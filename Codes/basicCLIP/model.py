# model.py
import torch
import torch.nn as nn
from clip.clip import load, tokenize 
from datasets import TAGS, generate_tag_prompts

class MusicCLIPClassifier(nn.Module):
    def __init__(self, model_name: str = "ViT-B/16", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.device = torch.device(device)
        
        self.model, _ = load(model_name, device=device)
        
        self.tag_prompts = generate_tag_prompts()
        self.text_tokens = tokenize(self.tag_prompts).to(self.device)

    def _get_text_features(self) -> torch.Tensor:
        text_features = self.model.encode_text(self.text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def forward(self, x: torch.Tensor) -> dict:

        # x (Batch, 3, 224, 224)
    
        image_features = self.model.encode_image(x.to(self.device))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        text_features = self._get_text_features()
        
        similarity = image_features @ text_features.t()
        
        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * similarity
    
        
        return {
            'logits': logits
        }

if __name__ == "__main__":
    # 快速結構與梯度串接測試
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = MusicCLIPClassifier(model_name="ViT-B/16", device=device).to(device)
    mock_input = torch.randn(2, 9, 3, 224, 224).to(device)
    output = model(mock_input)
    
    print("\n✅ 模型的輸出結構檢查：")
    print(f"🔹 Logits 形狀 (預期 [2, 38]): {output['logits'].shape}")
    print(f"🔹 聚合後的歌曲影像特徵形狀 (預期 [2, 512]): {output['image_features'].shape}")
    print(f"🔹 標籤文字特徵形狀 (預期 [38, 512]): {output['text_features'].shape}")
    
    # 驗證 Text Encoder 是否帶有梯度
    print(f"\n🔹 文字特徵梯度檢查 (True 代表可訓練更新): {output['text_features'].requires_grad}")
    print(f"🔹 影像特徵梯度檢查 (True 代表可訓練更新): {output['image_features'].requires_grad}")