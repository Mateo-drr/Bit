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
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        hidden_dims: tuple[int, ...] = (512,),
    ) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        bert_hidden = self.bert.config.hidden_size
        h = hidden_dims[0] if hidden_dims else bert_hidden

        self.mlp = nn.Sequential(
            MishGLU(bert_hidden, h, h * 2),
            nn.Mish(),
            MishGLU(h, num_labels, h * 2),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = outputs.pooler_output
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        return self.mlp(pooled)