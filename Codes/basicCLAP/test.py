import os
import time
import torch
import pandas as pd
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader
import laion_clap
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
from thop import profile 
from laion_clap.hook import CLAP_Module

GENRES = ['classical', 'techno', 'electro', 'rock', 'india', 'opera', 'pop', 'new age', 'dance', 'country', 'heavy']
INSTRUMENTS = ['guitar', 'string', 'drum', 'piano', 'violin', 'vocal', 'synth', 'female', 'male', 'singer', 'no singer', 'harpsichord', 'flute', 'sitar', 'choir', 'female vocal', 'harp', 'cello']
MOODS = ['slow', 'fast', 'ambient', 'loud', 'quiet', 'soft', 'strange']

TAGS = [
    'guitar', 'classical', 'slow', 'techno', 'string', 'drum', 'electro', 'rock', 'fast', 'piano', 
    'ambient', 'beat', 'violin', 'vocal', 'synth', 'female', 'india', 'opera', 'male', 'singer', 
    'no singer', 'harpsichord', 'loud', 'quiet', 'flute', 'pop', 'soft', 'sitar', 'solo', 'choir', 
    'new age', 'dance', 'female vocal', 'harp', 'cello', 'strange', 'country', 'heavy'
]

# =====================================================================
# 步驟 1：音訊重取樣與 9 段滑窗切片 Dataset
# =====================================================================
class MTATVotingDataset(Dataset):
    def __init__(self, csv_path, audio_dir, target_sr=48000, segment_duration=10, num_segments=9, sample_ratio=0.1):
        self.audio_dir = audio_dir
        self.target_sr = target_sr
        self.segment_len = target_sr * segment_duration  # 10秒 = 480000 點
        self.num_segments = num_segments
        
        df = pd.read_csv(csv_path, sep='\t')
        first_char = df['mp3_path'].str[0].str.lower()
        test_chars = ['d', 'e', 'f']  # 學術標準測試劃分
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
            # ⚡ 核心 1：自動將原生 16000Hz 音訊重取樣至 CLAP 期待的 48000Hz
            y, _ = librosa.load(full_audio_path, sr=self.target_sr, mono=True)
            
            # ⚡ 核心 2：將整首歌均勻/滑窗切成 9 個 10 秒片段
            # MTAT 歌曲約 29.1 秒 (1396800 點)。設定步長讓 9 段能均勻覆蓋
            total_len = len(y)
            if total_len > self.segment_len:
                stride = (total_len - self.segment_len) // (self.num_segments - 1)
                segments = [y[i*stride : i*stride + self.segment_len] for i in range(self.num_segments)]
            else:
                # 若音訊過短，進行重複填充
                padded = np.pad(y, (0, self.segment_len - total_len))
                segments = [padded] * self.num_segments
                
            waveforms = np.stack(segments)  # Shape: (9, 480000)
            success = True
        except Exception as e:
            waveforms = np.zeros((self.num_segments, self.segment_len))
            success = False

        labels = torch.tensor(row[self.label_columns].values.astype(float), dtype=torch.float32)
        return torch.tensor(waveforms, dtype=torch.float32), labels, success

# =====================================================================
# 步驟 2：模型計算量分析、批次推理與精確多標籤指標計算
# =====================================================================
def evaluate_clap_pipeline(csv_path, audio_dir, sample_ratio=0.1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🎬 當前運算裝置: {device}")

    model = CLAP_Module(enable_fusion=False)
    model.load_ckpt()
    
    model.to(device)
    model.eval()
    print("✅ 模型加載完成。")

    # =====================================================================
    # ⚙️ 計算模型的 Params、FLOPS 與 吞吐量 (Throughput)
    # =====================================================================
    print("\n⚙️  [效能分析] 正在追蹤音訊特徵提取器計算量...")
    
    # CLAP 內部接收的是 10 秒 48000Hz 的音訊，形狀為 (Batch, Samples)
    dummy_audio_input = torch.randn(1, 480000).to(device)
    
    # 追蹤音訊編碼骨幹 (model.model.audio_branch)
    structured_input = {
        "waveform": dummy_audio_input.to(device) # 或者是你定義的虛擬音訊張量
    }

    # 3. 呼叫 thop.profile (注意 inputs 必須是 tuple，所以最後要加個逗號)
    flops, params = profile(
        model.model.audio_branch, 
        inputs=(structured_input,), 
        verbose=False
    )
    
    gflops_per_image = flops / (1024 ** 3)  # 此處等同於每 10 秒音訊片段的 GFLOPS
    total_params_m = params / (1024 ** 2)
    
    print(f" ├─ Audio Backbone Params : {total_params_m:.2f} M")
    print(f" └─ Audio Backbone FLOPS  : {gflops_per_image:.2f} GFLOPS (per 10s segment)")

    # 2. 準備文字 Prompt 矩陣
    dataset = MTATVotingDataset(csv_path, audio_dir, sample_ratio=sample_ratio)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2) # 內含 9 段，實際批次量相當於 8 * 9 = 72

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
    
    text_prompts = generate_tag_prompts()
    
    with torch.no_grad():
        text_embeds = model.get_text_embedding(text_prompts)
        text_embeds = torch.tensor(text_embeds, device=device).float()
        text_embeds /= text_embeds.norm(dim=-1, keepdim=True)

    all_song_predictions = []
    all_song_targets = []
    
    # 吞吐量計時
    total_processed_segments = 0
    start_time = time.time()

    print("\n🚀 [正向傳播] 執行高度矩陣化的批次推理與 9 段投票...")
    with torch.no_grad():
        for waveforms, labels, successes in tqdm(loader, desc="Inference"):
            valid_mask = (successes == True)
            if not valid_mask.any(): continue
            
            waveforms = waveforms[valid_mask].to(device)  # Shape: (Batch, 9, 480000)
            labels = labels[valid_mask]
            
            B, S, L = waveforms.shape
            # ⚡ 矩陣加速拉平：將 (Batch, 9, 480000) 融合成 (Batch * 9, 480000) 一次丟給 GPU
            flat_waveforms = waveforms.view(B * S, L)
            
            # 提取所有音訊片段特徵
            audio_embeds = model.get_audio_embedding_from_data(x=flat_waveforms, use_tensor=True)
            audio_embeds = audio_embeds.to(device).float()
            audio_embeds /= audio_embeds.norm(dim=-1, keepdim=True)
            
            # 計算片段相似度 -> Shape: (Batch * 9, 38)
            flat_similarity = torch.matmul(audio_embeds, text_embeds.T)
            flat_probs = torch.sigmoid(flat_similarity)
            
            # ⚡ 還原形狀 -> Shape: (Batch, 9, 38)
            probs_per_segment = flat_probs.view(B, S, len(TAGS))
            
            # ⚡ 投票機制 (Voting)：對 9 個片段的機率取平均值 -> Shape: (Batch, 38)
            voted_song_probs = probs_per_segment.mean(dim=1)
            
            all_song_predictions.append(voted_song_probs.cpu().numpy())
            all_song_targets.append(labels.numpy())
            total_processed_segments += (B * S)

    end_time = time.time()
    
    # 計算吞吐量 (Throughput)
    total_time = end_time - start_time
    throughput_fps = total_processed_segments / total_time
    
    # ⚡ 新增：計算以「首」為單位的吞吐量（只計算成功處理的歌曲數）
    total_processed_songs = len(all_song_targets)
    throughput_songs = total_processed_songs / total_time

    print(f" ├─ 總推論消耗時間        : {total_time:.2f} 秒")
    print(f" ├─ 核心硬體吞吐量        : {throughput_fps:.2f} 片段/秒 (Segments per Second)")
    print(f" └─ 實際音樂處理吞吐量    : {throughput_songs:.2f} 首/秒 (Songs per Second) 🚀")

    # 聚合所有歌曲結果
    all_song_predictions = np.vstack(all_song_predictions)
    all_song_targets = np.vstack(all_song_targets)

    # =====================================================================
    # 📊 精確的多標籤學術指標計算 (Macro AUC & Macro mAP)
    # =====================================================================
    print("\n📊 [指標評估] 正在計算多標籤獨立解耦指標...")
    
    # 嚴格過濾：該標籤在測試集中必須同時包含正樣本(1)與負樣本(0)
    valid_classes = [i for i in range(all_song_targets.shape[1]) if len(np.unique(all_song_targets[:, i])) > 1]
    print(f" ℹ️  資訊：共 {len(valid_classes)}/{len(TAGS)} 個標籤符合學術計分標準（同時具備正負樣本）。")

    # 針對多標籤任務的標準計算方式：對各個獨立類別分別計算，最後取算術平均 (Macro)
    macro_auc = roc_auc_score(all_song_targets[:, valid_classes], all_song_predictions[:, valid_classes], average='macro')
    macro_map = average_precision_score(all_song_targets[:, valid_classes], all_song_predictions[:, valid_classes], average='macro')

    print("\n======================= 📊 最終學術評估報告 =======================")
    print(f" 🚀 評估歌曲總量 (Songs)      : {len(all_song_targets)} 首")
    print(f" ⚡ 硬體運算吞吐量 (Throughput): {throughput_fps:.2f} segments/s")
    print(f" 🎵 系統音樂吞吐量 (Throughput): {throughput_songs:.2f} songs/s  <-- 看這個最準")
    print(f" 🌟 Macro AUC-ROC             : {macro_auc:.4f}")
    print(f" 🌟 Macro mAP (PR-AUC)        : {macro_map:.4f}")
    print("=====================================================================")

if __name__ == "__main__":
    CSV_PATH = "/home/vincent071659/MIR/MagnaTagATune/annotations_clean_top50.csv"
    AUDIO_DIR = "/home/vincent071659/MIR/MagnaTagATune/MagnaTagATune"
    
    # 執行評估
    evaluate_clap_pipeline(CSV_PATH, AUDIO_DIR, sample_ratio=1)