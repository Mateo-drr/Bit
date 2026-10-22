from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChatConfig:
    model_dir: str = "models/bit"
    max_length: int = 384
    truncation_side: str = "left"
    device: str = "auto"
    show_probabilities: bool = False
