import os
import torch
import pandas as pd
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader
import laion_clap
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
from laion_clap.hook import CLAP_Module

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

GENRES = ['classical', 'techno', 'electro', 'rock', 'india', 'opera', 'pop', 'new age', 'dance', 'country', 'heavy']
INSTRUMENTS = ['guitar', 'string', 'drum', 'piano', 'violin', 'vocal', 'synth', 'female', 'male', 'singer', 'no singer', 'harpsichord', 'flute', 'sitar', 'choir', 'female vocal', 'harp', 'cello']
MOODS = ['slow', 'fast', 'ambient', 'loud', 'quiet', 'soft', 'strange']

TAGS = [
    'guitar', 'classical', 'slow', 'techno', 'string', 'drum', 'electro', 'rock', 'fast', 'piano', 
    'ambient', 'beat', 'violin', 'vocal', 'synth', 'female', 'india', 'opera', 'male', 'singer', 
    'no singer', 'harpsichord', 'loud', 'quiet', 'flute', 'pop', 'soft', 'sitar', 'solo', 'choir', 
    'new age', 'dance', 'female vocal', 'harp', 'cello', 'strange', 'country', 'heavy'
]


class MTATVotingDataset(Dataset):
    def __init__(self, csv_path, audio_dir, target_sr=48000, segment_duration=10, num_segments=9, sample_ratio=0.1):
        self.audio_dir = audio_dir
        self.target_sr = target_sr
        self.segment_len = target_sr * segment_duration  
        self.num_segments = num_segments
        
        df = pd.read_csv(csv_path, sep='\t')
        first_char = df['mp3_path'].str[0].str.lower()
        test_chars = ['c']  
        self.df = df[first_char.isin(test_chars)].reset_index(drop=True)
        
        if sample_ratio < 1.0:
            self.df = self.df.sample(frac=sample_ratio, random_state=42).reset_index(drop=True)
            
        print(f"🎵 [Dataset] 載入 MTAT 測試集。歌曲數: {len(self.df)} (重取樣至 {target_sr}Hz, 採 9 段投票制)")
        self.label_columns = TAGS

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        clean_mp3_path = row['mp3_path'].replace('/', os.sep)
        full_audio_path = os.path.join(self.audio_dir, clean_mp3_path)
        
        try:
            y, _ = librosa.load(full_audio_path, sr=self.target_sr, mono=True)
            total_len = len(y)
            if total_len > self.segment_len:
                stride = (total_len - self.segment_len) // (self.num_segments - 1)
                segments = [y[i*stride : i*stride + self.segment_len] for i in range(self.num_segments)]
            else:
                padded = np.pad(y, (0, self.segment_len - total_len))
                segments = [padded] * self.num_segments
                
            waveforms = np.stack(segments)  
            success = True
        except Exception as e:
            waveforms = np.zeros((self.num_segments, self.segment_len))
            success = False

        labels = torch.tensor(row[self.label_columns].values.astype(float), dtype=torch.float32)
        return torch.tensor(waveforms, dtype=torch.float32), labels, success


def calculate_sample_metrics(all_song_targets, all_song_predictions, threshold):
    """根據給定門檻值快速計算 Sample-level 指標"""
    pred_labels = (all_song_predictions >= threshold).astype(int)
    
    exact_matches = np.all(pred_labels == all_song_targets, axis=1)
    subset_accuracy = np.mean(exact_matches)
    
    sample_f1s = []
    for i in range(all_song_targets.shape[0]):
        y_true = all_song_targets[i]
        y_pred = pred_labels[i]
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        if np.sum(y_true) == 0 and np.sum(y_pred) == 0:
            score = 1.0
        elif (2 * tp + fp + fn) == 0:
            score = 0.0
        else:
            score = (2 * tp) / (2 * tp + fp + fn)
            
        sample_f1s.append(score)
        
    mean_sample_f1 = np.mean(sample_f1s)
    return subset_accuracy, mean_sample_f1


def evaluate_clap_pipeline(csv_path, audio_dir, sample_ratio=0.1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🎬 當前運算裝置: {device}")

    model = CLAP_Module(enable_fusion=False)
    model.load_ckpt()
    model.to(device)
    model.eval()
    print("✅ 模型加載完成。")

    dataset = MTATVotingDataset(csv_path, audio_dir, sample_ratio=sample_ratio)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2) 

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
    
    text_prompts = generate_tag_prompts()
    
    with torch.no_grad():
        text_embeds = model.get_text_embedding(text_prompts)
        text_embeds = torch.tensor(text_embeds, device=device).float()
        text_embeds /= text_embeds.norm(dim=-1, keepdim=True)

    all_song_predictions = []
    all_song_targets = []
    
    print("\n🚀 [正向傳播] 執行高度矩陣化的批次推理與 9 段投票...")
    with torch.no_grad():
        for waveforms, labels, successes in tqdm(loader, desc="Inference"):
            valid_mask = (successes == True)
            if not valid_mask.any(): continue
            
            waveforms = waveforms[valid_mask].to(device)  
            labels = labels[valid_mask]
            
            B, S, L = waveforms.shape
            flat_waveforms = waveforms.view(B * S, L)
            
            audio_embeds = model.get_audio_embedding_from_data(x=flat_waveforms, use_tensor=True)
            audio_embeds = audio_embeds.to(device).float()
            audio_embeds /= audio_embeds.norm(dim=-1, keepdim=True)
            
            flat_similarity = torch.matmul(audio_embeds, text_embeds.T)
            flat_probs = torch.sigmoid(flat_similarity)
            
            probs_per_segment = flat_probs.view(B, S, len(TAGS))
            voted_song_probs = probs_per_segment.mean(dim=1)
            
            all_song_predictions.append(voted_song_probs.cpu().numpy())
            all_song_targets.append(labels.numpy())

    all_song_predictions = np.vstack(all_song_predictions)
    all_song_targets = np.vstack(all_song_targets)

    # 1. 傳統類別指標計算 (Class-level)
    valid_classes = [i for i in range(all_song_targets.shape[1]) if len(np.unique(all_song_targets[:, i])) > 1]
    macro_auc = roc_auc_score(all_song_targets[:, valid_classes], all_song_predictions[:, valid_classes], average='macro')
    macro_map = average_precision_score(all_song_targets[:, valid_classes], all_song_predictions[:, valid_classes], average='macro')

    # 2. 🧠 動態網格搜尋最佳機率 Threshold (以 Sample-level F1 為優化目標)
    best_threshold = 0.5
    best_sample_f1 = -1.0
    best_subset_acc = 0.0
    
    print("\n🔍 正在尋找最佳機率 Threshold 門檻值...")
    threshold_candidates = np.arange(0.01, 1.0, 0.01)
    
    for th in threshold_candidates:
        subset_acc, sample_f1 = calculate_sample_metrics(all_song_targets, all_song_predictions, th)
        
        if sample_f1 > best_sample_f1:
            best_sample_f1 = sample_f1
            best_subset_acc = subset_acc
            best_threshold = th

    print(f"🎯 搜尋完成！最佳機率門檻值 (Best Threshold): {best_threshold:.2f}")

    print("\n======================= 📊 最終學術評估報告 (動態優化門檻) =======================")
    print(f" 🚀 評估歌曲總量 (Songs)      : {len(all_song_targets)} 首")
    print(f" 🌟 Macro AUC-ROC             : {macro_auc:.4f}")
    print(f" 🌟 Macro mAP (PR-AUC)        : {macro_map:.4f}")
    print("-" * 60)
    print(f" 🎯 最佳化歌曲標籤指標 (Sample-level, 最佳門檻 th={best_threshold:.2f}):")
    print(f"   - 完全命中率 (Subset Accuracy)       : {best_subset_acc * 100:.2f}%  (所有標籤全對才算對)")
    print(f"   - 平均歌曲得分 (Sample-level F1)     : {best_sample_f1 * 100:.2f}%  (綜合懲罰多預測與少預測)")
    print("=====================================================================")


if __name__ == "__main__":
    CSV_PATH = "/home/vincent071659/MIR/MagnaTagATune/annotations_clean_top50.csv"
    AUDIO_DIR = "/home/vincent071659/MIR/MagnaTagATune/MagnaTagATune"
    
    evaluate_clap_pipeline(CSV_PATH, AUDIO_DIR, sample_ratio=0.5)