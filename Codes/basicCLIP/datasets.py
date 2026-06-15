# dataset.py
import os
import cv2
import torch
import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

TAGS = [
    'guitar', 'classical', 'slow', 'techno', 'string', 'drum', 'electro', 'rock', 'fast', 'piano', 
    'ambient', 'beat', 'violin', 'vocal', 'synth', 'female', 'india', 'opera', 'male', 'singer', 
    'no singer', 'harpsichord', 'loud', 'quiet', 'flute', 'pop', 'soft', 'sitar', 'solo', 'choir', 
    'new age', 'dance', 'female vocal', 'harp', 'cello', 'strange', 'country', 'heavy'
]

# 根據你的分類定義標籤屬性
GENRES = ['classical', 'techno', 'electro', 'rock', 'india', 'opera', 'pop', 'new age', 'dance', 'country', 'heavy']
INSTRUMENTS = ['guitar', 'string', 'drum', 'piano', 'violin', 'vocal', 'synth', 'female', 'male', 'singer', 'no singer', 'harpsichord', 'flute', 'sitar', 'choir', 'female vocal', 'harp', 'cello']
MOODS = ['slow', 'fast', 'ambient', 'loud', 'quiet', 'soft', 'strange']

def audio_to_ssm_image(waveform, sr=16000, target_size=(224, 224)):
    """將 3 秒音訊片段轉換為 3 通道的標準化 SSM 圖片 (Shape: 3, 224, 224)"""
    y = waveform.numpy().squeeze() 
    if np.sum(np.abs(y)) == 0:
        return torch.zeros((3, target_size[0], target_size[1]), dtype=torch.float32)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    def compute_and_format_ssm(feature_matrix):
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

    r_channel = compute_and_format_ssm(mel)
    g_channel = compute_and_format_ssm(chroma)
    b_channel = compute_and_format_ssm(mfcc)

    ssm_image = np.stack([r_channel, g_channel, b_channel], axis=0)
    return torch.tensor(ssm_image, dtype=torch.float32)

def audio_to_csm_image(waveform, sr=16000, target_size=(224, 224)):
    """
    將音訊片段轉換為 3 通道的標準化 CSM (Cross-Similarity Matrix) 圖片 (Shape: 3, 224, 224)
    橫軸 (X軸)：輸入音訊的時間點
    縱軸 (Y軸)：均勻分布的 C2 ~ C7 參考音高
    """
    y = waveform.numpy().squeeze() 
    if np.sum(np.abs(y)) == 0:
        return torch.zeros((3, target_size[0], target_size[1]), dtype=torch.float32)

    # 1. 提取輸入音訊的特徵
    # 為了能跟音高基準（C2~C7）做 CSM 比較，最適合的特徵是 CQT (Constant-Q Transform)
    # CQT 的頻域特徵直接對應到鋼琴音高。這裡設置每八度 12 個半音。
    # 預設範圍從 C2 (MIDI 36) 開始，涵蓋 6 個八度 (C2 ~ C8)，共 12 * 6 = 72 個頻段
    # 這樣一來，C2~C7 就會完美落在這個特徵矩陣的縱軸範圍內
    cqt = np.abs(librosa.cqt(y=y, sr=sr, fmin=librosa.note_to_hz('C2'), n_bins=72, bins_per_octave=12))
    
    # 同時保留原本的 Chroma，提供不同的音高解析度視角（Chroma 會折疊八度，著重在音名）
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
    
    # 為了維持 3 通道豐富度，第 3 個通道我們使用 Mel-spectrogram，但縱軸稍後會透過 CSM 映射與時間軸對齊
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)

    # 取得音訊在特徵提取後的「時間幀數」(Time frames)
    num_frames = cqt.shape[1]

    # 2. 建立基準矩陣 (Reference Matrix)
    # 我們需要在輸入音訊的時間長度內，讓 C2 到 C7 「均勻分布」
    # C2 到 C7 在 MIDI 號碼中分別是 36 到 84
    # 我們在時間軸上均勻線性插值出對應的 MIDI 頻率
    midi_reference = np.linspace(36, 84, num_frames)
    hz_reference = librosa.midi_to_hz(midi_reference)

    # 建立與輸入特徵相對應的「虛擬基準特徵矩陣」
    # R 通道基準：CQT 基準
    ref_cqt = np.zeros_like(cqt)
    # G 通道基準：Chroma 基準 (將 MIDI 轉成 0~11 的音名 index)
    ref_chroma = np.zeros_like(chroma)
    # B 通道基準：Mel 基準 (將 Hz 轉成對應的 Mel 頻段 index)
    ref_mel = np.zeros_like(mel)
    mel_frequencies = librosa.mel_frequencies(n_mels=64, fmin=0, fmax=sr/2)

    for t in range(num_frames):
        # 填充 R 通道基準 (CQT bin index): 尋找最接近的 CQT bin
        cqt_bin = int(np.clip((midi_reference[t] - 36), 0, 71))
        ref_cqt[cqt_bin, t] = 1.0
        
        # 填充 G 通道基準 (Chroma index)
        chroma_bin = int(midi_reference[t] % 12)
        ref_chroma[chroma_bin, t] = 1.0
        
        # 填充 B 通道基準 (Mel bin index)
        mel_bin = np.argmin(np.abs(mel_frequencies - hz_reference[t]))
        ref_mel[mel_bin, t] = 1.0

    # 3. 計算 CSM 的核心函式 (計算 輸入特徵 與 基準特徵 的點積相似度)
    def compute_and_format_csm(input_feat, ref_feat):
        # Z-score 標準化
        mean_in = np.mean(input_feat, axis=1, keepdims=True)
        std_in = np.std(input_feat, axis=1, keepdims=True) + 1e-6
        norm_in = (input_feat - mean_in) / std_in
        
        mean_ref = np.mean(ref_feat, axis=1, keepdims=True)
        std_ref = np.std(ref_feat, axis=1, keepdims=True) + 1e-6
        norm_ref = (ref_feat - mean_ref) / std_ref
        
        # CSM 計算：輸入特徵(時間軸) vs 基準特徵(時間軸上的C2~C7分佈)
        # 矩陣大小會是 (num_frames, num_frames)
        # 橫軸代表輸入音訊的時間，縱軸代表基準音高（隨著時間從 C2 爬升到 C7）
        csm = np.dot(norm_ref.T, norm_in)
        
        # Min-Max 最大最小化至 0~1
        csm_min, csm_max = csm.min(), csm.max()
        if csm_max - csm_min > 0:
            csm = (csm - csm_min) / (csm_max - csm_min)
        else:
            csm = np.zeros_like(csm)
            
        return cv2.resize(csm, target_size, interpolation=cv2.INTER_LINEAR)

    # 計算三個通道的 CSM
    r_channel = compute_and_format_csm(cqt, ref_cqt)       # 絕對音高層面的相似度 (CQT)
    g_channel = compute_and_format_csm(chroma, ref_chroma) # 音名層面的相似度 (Chroma)
    b_channel = compute_and_format_csm(mel, ref_mel)       # 頻譜包絡層面的相似度 (Mel)

    # 疊加成 3 通道圖片
    csm_image = np.stack([r_channel, g_channel, b_channel], axis=0)
    return torch.tensor(csm_image, dtype=torch.float32)

class MusicSSMDataset(Dataset):
    def __init__(self, csv_path, audio_dir, mode='train', sample_rate=16000, duration=29, segment_duration=3):
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.target_length = sample_rate * duration
        self.segment_length = sample_rate * segment_duration 
        self.mode = mode  # 👈 記得把 mode 存下來
        
        df = pd.read_csv(csv_path, sep='\t' if csv_path.endswith('.csv') else ',')
        first_char = df['mp3_path'].str[0].str.lower()
        
        if mode == 'train':
            #train_chars = [str(i) for i in range(10)] + ['a', 'b']
            train_chars = [str(i) for i in range(8)]
            self.df = df[first_char.isin(train_chars)].reset_index(drop=True)
            self.df = self.df.sample(frac=0.2, random_state=42).reset_index(drop=True)
        elif mode == 'val':
            self.df = df[first_char == 'c'].reset_index(drop=True)
            self.df = self.df.sample(frac=0.1, random_state=42).reset_index(drop=True)
        elif mode == 'test':
            test_chars = ['d', 'e', 'f']
            self.df = df[first_char.isin(test_chars)].reset_index(drop=True)
            #self.df = self.df.sample(frac=0.1, random_state=42).reset_index(drop=True)
        else:
            raise ValueError("Invalid mode!")
            
        
        print(f"🎬 [{mode.upper()}] 實際處理歌曲數量: {len(self.df)}")
        
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
        except Exception as e:
            waveform = torch.zeros((1, self.target_length))

        # 切分成 9 個 3 秒片段 (Shape: 9, 1, 48000)
        segments = waveform.unfold(1, self.segment_length, self.segment_length).permute(1, 0, 2)
        
        # 根據 mode 決定要拿多少片段 👇
        if self.mode == 'train':
            # 訓練模式：隨機挑選 1 個片段 (這樣能當作 Data Augmentation，增加模型泛化力)
            random_idx = np.random.randint(0, len(segments))
            selected_segment = segments[random_idx]
            ssm_images = audio_to_ssm_image(selected_segment, self.sample_rate)     # ssm
            #ssm_images = audio_to_csm_image(selected_segment, self.sample_rate)    # csm
            # 此時 ssm_images 的 Shape 會是 (3, 224, 224)
        else:
            # 驗證/測試模式：拿全部 9 個片段，之後丟給模型做投票
            ssm_images = torch.stack([audio_to_ssm_image(seg, self.sample_rate) for seg in segments])       # ssm
            #ssm_images = torch.stack([audio_to_csm_image(seg, self.sample_rate) for seg in segments])      # csm
            # 此時 ssm_images 的 Shape 會是 (9, 3, 224, 224)
        
        labels = torch.tensor(row[self.label_columns].values.astype(float), dtype=torch.float32)
        
        return ssm_images, labels

def generate_tag_prompts():
    """根據標籤屬性自動建構專屬的 Prompt 敘述"""
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


if __name__ == "__main__":
    # Define execution configurations
    CSV_PATH = "C://myCodes//python//MIR_project//MagnaTagATune//annotations_clean_top50.csv"
    AUDIO_DIR = "C://myCodes//python//MIR_project//MagnaTagATune//MagnaTagATune//"
    
    test_dataset = MusicSSMDataset(CSV_PATH, AUDIO_DIR, 'train')
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    ssm_images, labels = test_dataset[0]

    print("=== [Dataset] 第一筆資料詳細資訊 ===")
    print(f"🔹 1. SSM 圖片張量形狀: {ssm_images.shape}") 

    print(f"\n🔹 2. 這首歌被標記為 1 的標籤 (Active Tags):")
    # 找出這首歌 Multi-hot 向量中數值為 1 的索引，並還原成文字

    active_tags = [TAGS[i] for i, val in enumerate(labels) if val == 1]

    if active_tags:
        print(f"   👉 {active_tags}")
    else:
        print("   👉 這首歌在抽樣中沒有觸發任何 38 類標籤（屬於 Background music）")

    print(f"\n🔹 3. 原始標籤張量 (Labels Tensor):\n{labels}")