import os
import time
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from datasets import MusicSSMDataset, TAGS, get_prompt_and_labels
from model import MusicCLIPClassifier

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

checkpoint_path = 'checkpoints/clip_model_4363_8773.pth'


def evaluate(model, loader, device):
    model.eval()
    all_labels, all_scores = [], []

    dummy_row = loader.dataset.df.iloc[0]
    single_prompts, _ = get_prompt_and_labels(dummy_row, TAGS, mode='test')

    with torch.no_grad():
        text_features = model.encode_text(single_prompts)
        text_features = F.normalize(text_features, p=2, dim=-1)

        for images, _, labels in loader:
            batch_size, num_segments, channels, height, width = images.shape
            flat_images = images.view(-1, channels, height, width).to(device)
            audio_features = model.encode_audio(flat_images)
            audio_features = F.normalize(audio_features, p=2, dim=-1)
            audio_features = audio_features.view(batch_size, num_segments, -1)
            final_features = torch.mean(audio_features, dim=1)
            final_features = F.normalize(final_features, p=2, dim=-1)
            logits = torch.matmul(final_features, text_features.t()) * model.model.logit_scale.exp()
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


def test_and_benchmark():
    with open('config.yaml', 'r') as rf:
        cfg = yaml.safe_load(rf)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_set = MusicSSMDataset(cfg['dataset']['csv_path'], cfg['dataset']['audio_dir'], 'test')
    test_loader = DataLoader(test_set, batch_size=cfg['batch'], shuffle=False)

    model = MusicCLIPClassifier(model_name='ViT-B/16', device=device)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f'Loaded weights from {checkpoint_path}')
    else:
        print('No checkpoint found. Using CLIP pretrained weights.')

    model.to(device)
    mean_ap, macro_auc = evaluate(model, test_loader, device)

    print('===== mixCLIP Test Results =====')
    print(f'mAP: {mean_ap:.4f}')
    print(f'Macro AUC: {macro_auc:.4f}')


if __name__ == '__main__':
    test_and_benchmark()
