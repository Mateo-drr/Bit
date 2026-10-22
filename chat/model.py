from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from bit_model import BitClassifier
from chat.config import ChatConfig


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


class Bit:
    def __init__(self, config: ChatConfig) -> None:
        self.config = config
        model_dir = Path(config.model_dir)
        metadata_path = model_dir / "training_meta.json"
        weights_path = model_dir / "bit_model.pt"
        if not metadata_path.exists() or not weights_path.exists():
            raise FileNotFoundError(
                f"Bit model not found in {model_dir}. "
                "Run `.venv/bin/python -m train.train` first."
            )

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.labels = tuple(metadata["labels"])
        self.device = resolve_device(config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.tokenizer.truncation_side = config.truncation_side
        self.model = BitClassifier(
            model_name=metadata["model_name"],
            num_labels=len(self.labels),
            hidden_dims=tuple(metadata["head_hidden_dims"]),
        )
        state_dict = torch.load(
            weights_path,
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> dict:
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        encoded = {
            name: tensor.to(self.device)
            for name, tensor in encoded.items()
        }
        with torch.inference_mode():
            logits = self.model(**encoded)
            probabilities = torch.softmax(logits, dim=-1)[0]

        index = int(probabilities.argmax().item())
        return {
            "label": self.labels[index],
            "confidence": float(probabilities[index].item()),
            "probabilities": {
                label: float(probabilities[i].item())
                for i, label in enumerate(self.labels)
            },
        }
