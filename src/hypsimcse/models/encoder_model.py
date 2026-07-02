"""SimCSE-style sentence encoder whose head lifts to the Lorentz model."""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from .lift import HyperbolicHead


def _pool(last_hidden, attention_mask, pooling):
    if pooling == "cls":
        return last_hidden[:, 0]
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


class SentenceEncoder(nn.Module):
    def __init__(self, model_name, out_dim, geometry, pooling="cls", c=1.0,
                 learnable_c=False, max_tangent_norm=None):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.pooling = pooling
        hidden = self.backbone.config.hidden_size
        self.head = HyperbolicHead(hidden, out_dim, geometry, c=c,
                                   learnable_c=learnable_c,
                                   max_tangent_norm=max_tangent_norm)

    def curvature(self):
        return self.head.curvature()

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = _pool(out.last_hidden_state, attention_mask, self.pooling)
        return self.head(pooled)

    @classmethod
    def build(cls, model_name, out_dim, geometry, **kw):
        tok = AutoTokenizer.from_pretrained(model_name)
        model = cls(model_name, out_dim, geometry, **kw)
        return model, tok


@torch.no_grad()
def encode_texts(model, tokenizer, texts, device, batch_size=64, max_length=64):
    model.eval()
    chunks = []
    for i in range(0, len(texts), batch_size):
        batch = tokenizer(texts[i:i + batch_size], return_tensors="pt",
                          padding=True, truncation=True, max_length=max_length)
        batch = {k: v.to(device) for k, v in batch.items()}
        chunks.append(model(batch["input_ids"], batch["attention_mask"]).cpu())
    return torch.cat(chunks, dim=0)
