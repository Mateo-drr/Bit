from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ChatConfig:
    model_dir: str = "models/bit"
    max_length: int = 384
    truncation_side: str = "left"
    device: str = "cpu"
    show_probabilities: bool = False


def list_available_models(models_root: str | Path = "models") -> list[tuple[str, Path]]:
    """Return list of (display_name, model_path) for valid models."""
    root = Path(models_root)
    models = []
    if not root.exists():
        return models

    # Check subdirectories
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            weights_path = entry / "bit_model.pt"
            metadata_path = entry / "training_meta.json"
            if weights_path.exists() and metadata_path.exists():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    labels = metadata.get("labels", [])
                    best_epoch = metadata.get("best_epoch")
                    display = entry.name
                    if best_epoch is not None:
                        display = f"{entry.name} (best_epoch={best_epoch}, labels={labels})"
                    models.append((display, entry))
                except Exception:
                    models.append((entry.name, entry))
    return models

