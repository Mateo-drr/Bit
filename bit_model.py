from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel


class MishGLU(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hid_dim: int):
        super().__init__()
        self.mish = nn.Mish()
        self.lin1 = nn.Linear(in_dim, hid_dim)
        self.lin2 = nn.Linear(in_dim, hid_dim)
        self.out = nn.Linear(hid_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_mish = self.mish(self.lin1(x))
        x = self.lin2(x)
        return self.out(x_mish * x)


class BitClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int, hid_dim: int, pooling: str = "mean") -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)  # e.g. "answerdotai/ModernBERT-base"
        self.pooling = pooling
        h = self.bert.config.hidden_size
        self.mlp = nn.Sequential(
            MishGLU(h, hid_dim, hid_dim),
            nn.Mish(),
            MishGLU(hid_dim, num_labels, hid_dim),
        )

    def forward(self, input_ids, attention_mask=None, **kwargs):
        hs = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.pooling == "cls" or attention_mask is None:
            pooled = hs[:, 0]
        else:
            m = attention_mask.unsqueeze(-1).to(hs.dtype)
            pooled = (hs * m).sum(1) / m.sum(1).clamp(min=1)
        return self.mlp(pooled)
