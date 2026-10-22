from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    model_name: str = "bert-base-cased"
    data_dir: str = "data"
    output_dir: str = "models/bit"

    device: str = "cpu"
    use_fp16: bool = True
    seed: int = 42

    max_length: int = 384
    truncation_side: str = "left"

    epochs: int = 3
    batch_size: int = 1
    eval_batch_size: int = 1
    grad_accum_steps: int = 2

    learning_rate: float = 1e-4
    weight_decay: float = 0.2
    max_grad_norm: float = 1.0

    head_hidden_dims: tuple[int, ...] = (512,)

    num_workers: int = 0
    log_every: int = 1

    labels: tuple[str, ...] = ("yes", "no", "idk")

    @property
    def label2id(self) -> dict[str, int]:
        return {label: index for index, label in enumerate(self.labels)}

    @property
    def id2label(self) -> dict[int, str]:
        return {index: label for index, label in enumerate(self.labels)}
