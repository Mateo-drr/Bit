from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class TrainConfig:
    model_name: str = ["bert-base-cased", "answerdotai/ModernBERT-base"][1]
    data_dir: str = "data"
    output_dir: str = "models/bit"

    def __post_init__(self):
        self.output_dir = f"{self.output_dir}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    device: str = "cuda"
    use_fp16: bool = True
    seed: int = 1001001

    max_length: int = 384
    truncation_side: str = "left"

    epochs: int = 10
    batch_size: int = 8
    grad_accum_steps: int = 2

    learning_rate: float = 1e-3
    bert_learning_rate: float | None = 1e-5  # freeze BERT completely
    weight_decay: float = 0.2
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.05

    head_hidden_dims: int = 64
    dropout: float = 0.2

    num_workers: int = 4
    log_every: int = 1

    labels: tuple[str, ...] = ("yes", "no", "idk")

    wb: bool = True
    project_name: str = "Bit"

    @property
    def label2id(self) -> dict[str, int]:
        return {label: index for index, label in enumerate(self.labels)}

    @property
    def id2label(self) -> dict[int, str]:
        return {index: label for index, label in enumerate(self.labels)}
