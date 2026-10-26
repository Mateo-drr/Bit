from __future__ import annotations

import os
from pathlib import Path

from chat.config import list_available_models

MODELS_ROOT = Path(__file__).resolve().parent / "models"
WHISPER_ROOT = MODELS_ROOT / "faster-whisper"
PIPER_ROOT = MODELS_ROOT / "piper"
HF_ROOT = MODELS_ROOT / "huggingface"
VOICE_INSTALL_HINT = (
    "Voice dependencies are missing. Install them with:\n"
    "  .venv/bin/pip install -e '.[voice]'"
)

# Set before transformers is imported so the Bit backbone is cached locally.
os.environ["HF_HOME"] = str(HF_ROOT)
os.environ["HF_HUB_CACHE"] = str(HF_ROOT / "hub")
# Ignore the legacy cache variable so an inherited value cannot relocate assets.
os.environ.pop("TRANSFORMERS_CACHE", None)


def resolve_model_dir(
    model: str | None = None,
    model_dir: str | None = None,
    models_root: str | Path = "models",
) -> Path:
    if model_dir:
        return Path(model_dir)

    root = Path(models_root)
    if model:
        candidate = root / model
        if candidate.exists():
            return candidate
        candidate = Path(model)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Could not find Bit model: {model}")

    models = list_available_models(root)
    if not models:
        raise FileNotFoundError(
            f"No trained Bit models found in {root}. "
            "Run `.venv/bin/python -m train.train` first."
        )
    return models[-1][1]
