import torch
import torch.nn as nn
import torch.nn.functional as F
from clip.clip import load, tokenize
from datasets import TAGS, generate_tag_prompts


class MusicCLIPClassifier(nn.Module):
    def __init__(self, model_name: str = 'ViT-B/16', device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = torch.device(device)
        self.model, _ = load(model_name, device=self.device)
        self.logit_scale = self.model.logit_scale

        self.tag_prompts = generate_tag_prompts()
        self.text_tokens = tokenize(self.tag_prompts).to(self.device)

    def _get_text_features(self) -> torch.Tensor:
        text_features = self.model.encode_text(self.text_tokens)
        return F.normalize(text_features, p=2, dim=-1)

    def encode_audio(self, x: torch.Tensor) -> torch.Tensor:
        audio_features = self.model.encode_image(x.to(self.device))
        return F.normalize(audio_features, p=2, dim=-1)

    def encode_text(self, text_prompts: list) -> torch.Tensor:
        tokens = tokenize(text_prompts).to(self.device)
        text_features = self.model.encode_text(tokens)
        return F.normalize(text_features, p=2, dim=-1)

    def forward(self, x: torch.Tensor, text_prompts: list = None) -> dict:
        audio_features = self.encode_audio(x)
        logit_scale = self.model.logit_scale.exp()

        if self.training and text_prompts is not None:
            text_features = self.encode_text(text_prompts)
            logits_per_audio = torch.matmul(audio_features, text_features.t()) * logit_scale
            logits_per_text = logits_per_audio.t()
            return {
                'logits_per_audio': logits_per_audio,
                'logits_per_text': logits_per_text
            }

        target_text_features = self._get_text_features()
        logits = torch.matmul(audio_features, target_text_features.t()) * logit_scale
        return {
            'logits': logits
        }


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MusicCLIPClassifier(model_name='ViT-B/16', device=device).to(device)
    mock_input = torch.randn(2, 3, 224, 224).to(device)
    out = model(mock_input)
    print(out['logits'].shape)
    mock_prompts = ['A music featuring guitar instrument.'] * 2
    out_train = model(mock_input, text_prompts=mock_prompts)
    print(out_train['logits_per_audio'].shape, out_train['logits_per_text'].shape)
