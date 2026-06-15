import os
import cv2
import torch
import librosa
import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

TAGS = [
    'guitar', 'classical', 'slow', 'techno', 'string', 'drum', 'electro', 'rock', 'fast', 'piano', 
    'ambient', 'beat', 'violin', 'vocal', 'synth', 'female', 'india', 'opera', 'male', 'singer', 
    'no singer', 'harpsichord', 'loud', 'quiet', 'flute', 'pop', 'soft', 'sitar', 'solo', 'choir', 
    'new age', 'dance', 'female vocal', 'harp', 'cello', 'strange', 'country', 'heavy'
]

GENRES = ['classical', 'techno', 'electro', 'rock', 'india', 'opera', 'pop', 'new age', 'dance', 'country', 'heavy']
INSTRUMENTS = ['guitar', 'string', 'drum', 'piano', 'violin', 'vocal', 'synth', 'female', 'male', 'singer', 'no singer', 'harpsichord', 'flute', 'sitar', 'choir', 'female vocal', 'harp', 'cello']
MOODS = ['slow', 'fast', 'ambient', 'loud', 'quiet', 'soft', 'strange']

TAG_CATEGORIES = {
    'Inst': INSTRUMENTS,
    'Genre': GENRES,
    'Mood': MOODS
}


def get_prompt_and_labels(row, label_columns, mode='train'):
    orig_labels = row[label_columns].values.astype(float)

    if mode == 'train':
        active_tags = [tag for tag in label_columns if row[tag] == 1]
        if not active_tags:
            return "A piece of music.", torch.tensor(orig_labels, dtype=torch.float32)

        inst_list = [t for t in active_tags if t in TAG_CATEGORIES['Inst']]
        genre_list = [t for t in active_tags if t in TAG_CATEGORIES['Genre']]
        mood_list = [t for t in active_tags if t in TAG_CATEGORIES['Mood']]
        other_list = [t for t in active_tags if t not in TAG_CATEGORIES['Inst'] 
                      and t not in TAG_CATEGORIES['Genre'] 
                      and t not in TAG_CATEGORIES['Mood']]

        components = []
        if inst_list:
            components.append(f"featuring {' and '.join(inst_list)}")
        if genre_list:
            components.append(f"in the {' and '.join(genre_list)} style")
        if mood_list:
            components.append(f"with a {' and '.join(mood_list)} mood")
        if other_list:
            components.append(f"characterized by {' and '.join(other_list)}")

        random.shuffle(components)
        prompt_body = ' '.join(components) if components else 'A piece of music.'
        prompt = f"A music {prompt_body}." if components else prompt_body
        return prompt, torch.tensor(orig_labels, dtype=torch.float32)

    single_prompts = []
    for tag in label_columns:
        if tag in TAG_CATEGORIES['Inst']:
            single_prompts.append(f"A music featuring {tag} instrument.")
        elif tag in TAG_CATEGORIES['Genre']:
            single_prompts.append(f"A music of {tag} genre.")
        elif tag in TAG_CATEGORIES['Mood']:
            single_prompts.append(f"A music with a {tag} mood.")
        else:
            single_prompts.append(f"A music characterized by {tag}.")

    return single_prompts, torch.tensor(orig_labels, dtype=torch.float32)


def audio_to_ssm_image(waveform, sr=16000, target_size=(224, 224)):
    y = waveform.numpy().squeeze()
    if np.sum(np.abs(y)) == 0:
        return torch.zeros((3, target_size[0], target_size[1]), dtype=torch.float32)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    def compute_ssm(feature_matrix):
        mean = np.mean(feature_matrix, axis=1, keepdims=True)
        std = np.std(feature_matrix, axis=1, keepdims=True) + 1e-6
        norm_feat = (feature_matrix - mean) / std
        ssm = np.dot(norm_feat.T, norm_feat)

        ssm_min, ssm_max = ssm.min(), ssm.max()
        if ssm_max - ssm_min > 0:
            ssm = (ssm - ssm_min) / (ssm_max - ssm_min)
        else:
            ssm = np.zeros_like(ssm)

        return cv2.resize(ssm, target_size, interpolation=cv2.INTER_LINEAR)

    r_channel = compute_ssm(mel)
    g_channel = compute_ssm(chroma)
    b_channel = compute_ssm(mfcc)
    ssm_image = np.stack([r_channel, g_channel, b_channel], axis=0)
    return torch.tensor(ssm_image, dtype=torch.float32)


class MusicSSMDataset(Dataset):
    def __init__(self, csv_path, audio_dir, mode='train', sample_rate=16000, duration=29, segment_duration=3):
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.target_length = sample_rate * duration
        self.segment_length = sample_rate * segment_duration
        self.mode = mode

        df = pd.read_csv(csv_path, sep='\t' if csv_path.endswith('.csv') else ',')
        first_char = df['mp3_path'].str[0].str.lower()

        if mode == 'train':
            train_chars = [str(i) for i in range(8)]
            self.df = df[first_char.isin(train_chars)].reset_index(drop=True)
            self.df = self.df.sample(frac=0.2, random_state=42).reset_index(drop=True)
        elif mode == 'val':
            self.df = df[first_char == 'c'].reset_index(drop=True)
            self.df = self.df.sample(frac=0.5, random_state=42).reset_index(drop=True)
        elif mode == 'test':
            test_chars = ['d', 'e', 'f']
            self.df = df[first_char.isin(test_chars)].reset_index(drop=True)
        else:
            raise ValueError('Invalid mode!')

        print(f"🎬 [{mode.upper()}] dataset size: {len(self.df)}")
        self.label_columns = TAGS

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        full_audio_path = os.path.join(self.audio_dir, row['mp3_path'])

        try:
            waveform_np, _ = librosa.load(full_audio_path, sr=self.sample_rate, mono=True)
            waveform = torch.tensor(waveform_np).unsqueeze(0)
            if waveform.shape[1] > self.target_length:
                waveform = waveform[:, :self.target_length]
            else:
                waveform = torch.nn.functional.pad(waveform, (0, self.target_length - waveform.shape[1]))
        except Exception:
            waveform = torch.zeros((1, self.target_length))

        segments = waveform.unfold(1, self.segment_length, self.segment_length).permute(1, 0, 2)

        if self.mode == 'train':
            random_idx = np.random.randint(0, len(segments))
            selected_segment = segments[random_idx]
            ssm_images = audio_to_ssm_image(selected_segment, self.sample_rate)
        else:
            ssm_images = torch.stack([audio_to_ssm_image(seg, self.sample_rate) for seg in segments])

        text_data, labels = get_prompt_and_labels(row, self.label_columns, mode=self.mode)
        return ssm_images, text_data, labels


def generate_tag_prompts():
    prompts = []
    for tag in TAGS:
        if tag in GENRES:
            prompts.append(f"A music of {tag} genre.")
        elif tag in INSTRUMENTS:
            prompts.append(f"A music featuring {tag} instrument.")
        elif tag in MOODS:
            prompts.append(f"A music with a {tag} mood.")
        else:
            prompts.append(f"A music characterized by {tag}.")
    return prompts
