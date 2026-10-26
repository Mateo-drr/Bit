import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, BertConfig, BertTokenizer

from bit_model import BitClassifier, MishGLU
from chat.config import ChatConfig
from chat.model import Bit

BACKBONE_HIDDEN_SIZE = 16
HEAD_HIDDEN_DIM = 8


def write_tiny_checkpoint(
    model_dir: Path,
    *,
    glu_hid_dim: int = HEAD_HIDDEN_DIM,
) -> None:
    backbone_config = BertConfig(
        vocab_size=32,
        hidden_size=BACKBONE_HIDDEN_SIZE,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
    )
    AutoModel.from_config(backbone_config).save_pretrained(model_dir)

    vocab_path = model_dir / "vocab.txt"
    vocab_path.write_text(
        "\n".join(
            [
                "[PAD]",
                "[UNK]",
                "[CLS]",
                "[SEP]",
                "[MASK]",
                "paris",
                "is",
                "in",
                "france",
            ]
        ),
        encoding="utf-8",
    )
    BertTokenizer(vocab_file=str(vocab_path)).save_pretrained(model_dir)

    (model_dir / "training_meta.json").write_text(
        json.dumps(
            {
                "model_name": str(model_dir),
                "labels": ["yes", "no", "idk"],
                "head_hidden_dims": HEAD_HIDDEN_DIM,
                "max_length": 16,
                "truncation_side": "left",
            }
        ),
        encoding="utf-8",
    )
    model = BitClassifier(
        model_name=str(model_dir),
        num_labels=3,
        hid_dim=HEAD_HIDDEN_DIM,
    )
    model.mlp = nn.Sequential(
        MishGLU(BACKBONE_HIDDEN_SIZE, HEAD_HIDDEN_DIM, glu_hid_dim),
        nn.Mish(),
        MishGLU(HEAD_HIDDEN_DIM, 3, glu_hid_dim),
    )
    torch.save(model.state_dict(), model_dir / "bit_model.pt")


class ChatModelTests(unittest.TestCase):
    def load_and_predict(self, model_dir: Path) -> None:
        bit = Bit(
            ChatConfig(model_dir=str(model_dir), max_length=16, device="cpu")
        )
        result = bit.predict("Paris is in France. Is Paris in France?")

        self.assertEqual(
            set(result), {"label", "confidence", "probabilities"}
        )
        self.assertIn(result["label"], {"yes", "no", "idk"})
        self.assertEqual(
            set(result["probabilities"]), {"yes", "no", "idk"}
        )
        self.assertAlmostEqual(
            sum(result["probabilities"].values()), 1.0, places=5
        )

    def test_loads_checkpoint_trained_with_current_head_design(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            write_tiny_checkpoint(model_dir)
            self.load_and_predict(model_dir)

    def test_loads_checkpoint_with_wider_mish_layers(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            write_tiny_checkpoint(model_dir, glu_hid_dim=16)
            self.load_and_predict(model_dir)


if __name__ == "__main__":
    unittest.main()
