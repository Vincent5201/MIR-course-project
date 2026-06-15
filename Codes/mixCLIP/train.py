import os
import time
import random
import yaml
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from datasets import MusicSSMDataset, get_prompt_and_labels
from model import MusicCLIPClassifier

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

with open('config.yaml', 'r') as rf:
    cfg = yaml.safe_load(rf)

os.makedirs('logs', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

log_filename = f"logs/mixclip_train.log"
logging.basicConfig(
    filename=log_filename, filemode='w',
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO
)
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())

seed = int(cfg.get('seed', 42))
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger.info('Initializing mixCLIP datasets...')
train_set = MusicSSMDataset(cfg['dataset']['csv_path'], cfg['dataset']['audio_dir'], 'train')
val_set = MusicSSMDataset(cfg['dataset']['csv_path'], cfg['dataset']['audio_dir'], 'val')

train_loader = DataLoader(train_set, batch_size=cfg['batch'], shuffle=True, drop_last=True)
val_loader = DataLoader(val_set, batch_size=cfg['batch'], shuffle=False)

logger.info(f'Loading MusicCLIPClassifier on {device}...')
model = MusicCLIPClassifier(model_name='ViT-B/16', device=device)
model.to(device)

lr = float(cfg.get('lr', 1e-5))
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
criterion = nn.BCEWithLogitsLoss()
info_nce_weight = float(cfg.get('info_nce_weight', 1.0))
bce_weight = float(cfg.get('bce_weight', 1.0))


def evaluate(model, loader, device, label_columns):
    model.eval()
    all_labels, all_scores = [], []

    dummy_row = loader.dataset.df.iloc[0]
    single_prompts, _ = get_prompt_and_labels(dummy_row, label_columns, mode='val')

    with torch.no_grad():
        text_features = model.encode_text(single_prompts)
        text_features = F.normalize(text_features, p=2, dim=-1)

        for images, _, labels in loader:
            batch_size, num_segments, channels, height, width = images.shape
            flat_images = images.view(-1, channels, height, width).to(device)
            audio_features = model.encode_audio(flat_images)
            audio_features = F.normalize(audio_features, p=2, dim=-1)
            audio_features = audio_features.view(batch_size, num_segments, -1)
            final_audio_features = torch.mean(audio_features, dim=1)
            final_audio_features = F.normalize(final_audio_features, p=2, dim=-1)
            logits = torch.matmul(final_audio_features, text_features.t()) * model.model.logit_scale.exp()
            scores = torch.sigmoid(logits)
            all_labels.append(labels)
            all_scores.append(scores.cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_scores = torch.cat(all_scores).numpy()

    valid_aucs, valid_aps = [], []
    for i in range(all_labels.shape[1]):
        if np.sum(all_labels[:, i] == 1) > 0 and np.sum(all_labels[:, i] == 0) > 0:
            valid_aucs.append(roc_auc_score(all_labels[:, i], all_scores[:, i]))
            valid_aps.append(average_precision_score(all_labels[:, i], all_scores[:, i]))

    macro_auc = np.mean(valid_aucs) if valid_aucs else 0.0
    mean_ap = np.mean(valid_aps) if valid_aps else 0.0
    return mean_ap, macro_auc


train_iter = iter(train_loader)
max_iter = int(cfg.get('max_iter', 5001))
eval_step = int(cfg.get('eval_step', 200))
best_map, best_auc = 0.0, 0.0

logger.info(f'Starting mixCLIP training. Total iterations: {max_iter}')
for iter_num in range(max_iter):
    model.train()
    try:
        images, text_prompts, labels = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        images, text_prompts, labels = next(train_iter)

    x = images.to(device).float()
    y = labels.to(device).float()

    optimizer.zero_grad()
    out = model(x, text_prompts=text_prompts)

    batch_size = x.size(0)
    ground_truth = torch.arange(batch_size, dtype=torch.long, device=device)
    info_nce_loss = (F.cross_entropy(out['logits_per_audio'], ground_truth) + F.cross_entropy(out['logits_per_text'], ground_truth)) / 2

    cls_out = model(x)
    bce_loss = criterion(cls_out['logits'], y)
    
    loss = info_nce_weight * info_nce_loss + bce_weight * bce_loss
    loss.backward()
    optimizer.step()

    if (iter_num + 1) % eval_step == 0:
        mAP, auc = evaluate(model, val_loader, device, train_set.label_columns)
        if mAP > best_map:
            best_map, best_auc = mAP, auc
            torch.save(model.state_dict(), 'checkpoints/clip_model.pth')
            logger.info(f'🔥 New best model saved at iteration {iter_num+1}!')

        logger.info(
            f'[{iter_num+1}/{max_iter}] '
            f'Loss: {loss.item():.4f} | '
            f'InfoNCE: {info_nce_loss.item():.4f} | '
            f'BCE: {bce_loss.item():.4f} | '
            f'mAP: {mAP:.4f} (best {best_map:.4f}) | '
            f'Macro AUC: {auc:.4f} (best {best_auc:.4f})'
        )

logger.info(f'Final best metrics -> mAP: {best_map:.4f}, Macro AUC: {best_auc:.4f}')
