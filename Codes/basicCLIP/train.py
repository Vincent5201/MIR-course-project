# train.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
import logging
import os
import time
import yaml
import numpy as np

import warnings
# 忽略所有非致命的 Warning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 載入你先前寫好的 dataset.py 與修正後的 model.py
from MIR.basicCLIP.datasets import MusicSSMDataset  
from MIR.basicCLIP.model import MusicCLIPClassifier

# 1. 載入配置檔案
with open("config.yaml", 'r') as rf:
    cfg = yaml.safe_load(rf)

# 2. 建立日誌與模型權重儲存路徑
os.makedirs("logs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

log_filename = f"logs/music_clip_train.log"
logging.basicConfig(
    filename=log_filename, filemode="w",
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(cfg['seed'])
torch.manual_seed(cfg['seed'])
np.random.seed(cfg['seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg['seed'])


logger.info("Initializing Music SSM Datasets...")
train_set = MusicSSMDataset(cfg['dataset']['csv_path'], cfg['dataset']['audio_dir'], 'train')
test_set = MusicSSMDataset(cfg['dataset']['csv_path'], cfg['dataset']['audio_dir'], 'val')

train_loader = DataLoader(train_set, batch_size=cfg['batch'], shuffle=True, drop_last=True)
test_loader = DataLoader(test_set, batch_size=cfg['batch'], shuffle=False)


logger.info(f"Loading MusicCLIPClassifier (Full-Trainable) on {device}...")
model = MusicCLIPClassifier(model_name="ViT-B/16", device=device)
model.to(device)


lr = float(cfg['lr'])
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)


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

train_iter = iter(train_loader)

max_iter = 40001
eval_step = 200
best_map, best_auc = 0.0, 0.0
criterion = nn.BCEWithLogitsLoss()

logger.info(f"Starting Joint Music-Text Training. Total Iterations: {max_iter}")


for iter_num in range(max_iter):
    model.train()
    
    try:
        images, labels = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        images, labels = next(train_iter)

    x = images.to(device).float()
    y = labels.to(device).float()  # BCE Loss 接收的標籤必須是 Float 格式

    optimizer.zero_grad()

    out = model(x)
    logits = out['logits']                  # 內部已乘上原生 logit_scale.exp()
    
    loss = criterion(logits, y)

    loss.backward()
    optimizer.step()

    if (iter_num + 1) % eval_step == 0:
        mAP, auc = evaluate(model, test_loader, device)
        
        # 多標籤任務通常以最高 mAP（或是 AUC）作為最優模型的儲存標準
        if mAP > best_map:
            best_map, best_auc = mAP, auc
            torch.save(model.state_dict(), f"checkpoints/clip_model.pth")
            logger.info(f"🔥 New Best Model Saved at Iteration {iter_num+1}!")

        logger.info(
            f"[{iter_num+1}/{max_iter}] "
            f"Loss: {loss.item():.4f} | "
            f"mAP: {mAP:.4f} (Best mAP: {best_map:.4f}) | "
            f"Macro AUC: {auc:.4f} (Best AUC: {best_auc:.4f})"
        )

logger.info(f"Final Best Metrics -> mAP: {best_map:.4f}, Macro AUC: {best_auc:.4f}")