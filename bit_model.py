from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel


class MishGLU(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hid_dim: int):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hid_dim)
        self.mish = nn.Mish()
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
        hid_dim: int,
        dropout: float = 0.0,
        glu_hid_dim: int | None = None,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)  # e.g. "answerdotai/ModernBERT-base"
        self.pooling = pooling
        h = self.bert.config.hidden_size
        glu_hid_dim = hid_dim if glu_hid_dim is None else glu_hid_dim
        self.mlp = nn.Sequential(
            nn.Dropout(p=dropout),
            MishGLU(h, hid_dim, glu_hid_dim),
            nn.Mish(),
            nn.Dropout(p=dropout),
            MishGLU(hid_dim, num_labels, glu_hid_dim),
        )

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        model_name: str,
        pooling: str = "mean",
    ) -> "BitClassifier":
        """Rebuild the classifier using the head shapes stored in a checkpoint."""
        required = (
            "mlp.0.lin1.weight",
            "mlp.0.out.weight",
            "mlp.2.lin1.weight",
            "mlp.2.out.weight",
        )
        for key in required:
            if key not in state_dict:
                raise ValueError(
                    f"Unsupported checkpoint architecture: missing {key}"
                )

        glu_hid_dim = int(state_dict["mlp.0.lin1.weight"].shape[0])
        hid_dim = int(state_dict["mlp.0.out.weight"].shape[0])
        num_labels = int(state_dict["mlp.2.out.weight"].shape[0])
        second_glu_hid_dim = int(state_dict["mlp.2.lin1.weight"].shape[0])
        if second_glu_hid_dim != glu_hid_dim:
            raise ValueError(
                "Unsupported checkpoint architecture: GLU layers have "
                f"different hidden widths ({glu_hid_dim} and "
                f"{second_glu_hid_dim})"
            )

        return cls(
            model_name=model_name,
            num_labels=num_labels,
            hid_dim=hid_dim,
            glu_hid_dim=glu_hid_dim,
            pooling=pooling,
        )

    def forward(self, input_ids, attention_mask=None, **kwargs):
        hs = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.pooling == "cls" or attention_mask is None:
            pooled = hs[:, 0]
        else:
            m = attention_mask.unsqueeze(-1).to(hs.dtype)
            pooled = (hs * m).sum(1) / m.sum(1).clamp(min=1)
        return self.mlp(pooled)
