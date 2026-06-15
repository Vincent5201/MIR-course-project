import torch
import yaml
import os
import time
from thop import profile
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 直接引入你 train.py 定義好的 evaluate，以及資料集和模型
from datasets import MusicSSMDataset 
from model import MusicCLIPClassifier

checkpoint_path = "checkpoints/clip_model_4176_8741.pth"

def evaluate(model, loader, device):
    model.eval()
    all_labels, all_scores = [], []

    with torch.no_grad():
        for images, labels in loader:
            batch_size, num_segments, channels, height, width = images.shape
            
            # 1. 壓平多個片段 -> (B * 9, 3, 224, 224) 丟給 CNN
            flat_images = images.view(-1, channels, height, width).to(device)
            
            # 2. 通過模型得到全體片段的 Logits 並轉成機率
            out = model(flat_images)
            flat_probs = torch.sigmoid(out['logits'])
            
            # 3. 還原形狀並在片段維度（dim=1）取平均（投票機制）
            probs_per_segment = flat_probs.view(batch_size, num_segments, -1)
            final_probs = torch.mean(probs_per_segment, dim=1)
            
            all_labels.append(labels)
            all_scores.append(final_probs.cpu())

    all_labels = torch.cat(all_labels).numpy()      # 形狀: (總歌曲數, 38)
    all_scores = torch.cat(all_scores).numpy()      # 形狀: (總歌曲數, 38)

    # 4. 逐類別檢查，確保同時存在 0 與 1 才計算指標，避免小樣本抽樣導致 nan 炸掉整體
    valid_aucs = []
    valid_aps = []
    for i in range(all_labels.shape[1]):
        # 該類別必須在驗證集中同時有正樣本（1）與負樣本（0）
        if np.sum(all_labels[:, i] == 1) > 0 and np.sum(all_labels[:, i] == 0) > 0:
            valid_aucs.append(roc_auc_score(all_labels[:, i], all_scores[:, i]))
            valid_aps.append(average_precision_score(all_labels[:, i], all_scores[:, i]))
            
    macro_auc = np.mean(valid_aucs) if valid_aucs else 0.0
    mean_ap = np.mean(valid_aps) if valid_aps else 0.0
    
    return mean_ap, macro_auc

def test_and_benchmark():
    # 1. 載入配置檔案
    with open("config.yaml", 'r') as rf:
        cfg = yaml.safe_load(rf)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("1. 正在初始化測試資料集與載入模型...")
    test_set = MusicSSMDataset(cfg['dataset']['csv_path'], cfg['dataset']['audio_dir'], 'test')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=cfg['batch'], shuffle=False)
    
    model = MusicCLIPClassifier(model_name="ViT-B/16", device=device)
    
    # 載入訓練好的最優權重
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"👉 成功載入最優權重: {checkpoint_path}")
    else:
        print(f"⚠️ 找不到權重，將使用 CLIP 初始權重。")
        
    model.to(device)
    model.eval()

    # 2. ⚙️ 計算 Params 與 FLOPS
    # 建立一個與 DataLoader 相同維度的 dummy 輸入 (1 張影像片段) 來追蹤計算量
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    flops, params = profile(model.model.visual, inputs=(dummy_input,), verbose=False)
    gflops_per_image = flops / (1024 ** 3)
    total_params_m = params / (1024 ** 2)

    print("2. 開始執行 Zero-shot 推論並測量 Throughput...")
    
    # 3. ⚡ 精準測量 Throughput 耗時
    # 在呼叫原生的 evaluate 前後進行同步與計時
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()

    # 🛑 直接呼叫你 train.py 的原創 evaluate 函式
    mean_ap, macro_auc = evaluate(model, test_loader, device)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()

    # 4. 計算總處理的影像片段張數 (總歌數 * 每首歌的片段數)
    # 透過讀取 Dataset 的實際資料長度與 images 的形狀特徵來回推
    total_songs = len(test_set)
    # 撈出第一筆資料來確認 num_segments (通常是 9)
    sample_images, _ = test_set[0] if isinstance(test_set[0], tuple) else (test_set[0][0], None)
    num_segments = sample_images.shape[0] if len(sample_images.shape) == 4 else 9
    total_images_processed = total_songs * num_segments

    # 5. Throughput 計算
    total_pure_time_sec = end_time - start_time
    throughput = total_images_processed / total_pure_time_sec if total_pure_time_sec > 0 else 0.0

    # 6. 輸出最終精簡報告
    print("\n" + "=" * 60)
    print(" 🚀 MUSIC-CLIP 最終評估報告")
    print("=" * 60)
    print(f"📈 演算法精準度指標:")
    print(f"   - Mean Average Precision (mAP) : {mean_ap:.4f}")
    print(f"   - Macro Area Under ROC (AUC)   : {macro_auc:.4f}")
    print("-" * 60)
    print(f"⚙️ 指定硬體效能基準指標:")
    print(f"   - 總參數數量 (Params)          : {total_params_m:.2f} M")
    print(f"   - 單張影像計算量 (FLOPS)       : {gflops_per_image:.2f} GFLOPS")
    print(f"   - 系統吞吐量 (Throughput)      : {throughput:.2f} Images/sec")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    test_and_benchmark()