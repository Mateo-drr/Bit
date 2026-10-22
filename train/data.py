from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

BASE_URL = (
    "https://raw.githubusercontent.com/CogComp/Yes-No-or-IDK/master/"
    "DATA/BoolQ_3L"
)
FILES = {
    "train": "train_full.json",
    "dev": "dev_full.json",
}
EXPECTED_COUNTS = {
    "train": 14141,
    "dev": 4906,
}


def map_answer(answer: Any) -> str:
    if answer is True:
        return "yes"
    if answer is False:
        return "no"
    if answer == "no-answer":
        return "idk"
    raise ValueError(f"Unknown BoolQ_3L answer: {answer!r}")


def build_text(passage: str, question: str) -> str:
    passage = passage.strip()
    question = question.strip()
    if not passage:
        return question
    if not question:
        return passage
    return f"{passage} {question}"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def ensure_data(data_dir: str | Path) -> dict[str, Path]:
    directory = Path(data_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split, filename in FILES.items():
        path = directory / filename
        if not path.exists() or path.stat().st_size == 0:
            url = f"{BASE_URL}/{filename}"
            temporary = path.with_name(path.name + ".tmp")
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, temporary)
            temporary.replace(path)
        paths[split] = path
    return paths


def load_split(data_dir: str | Path, split: str) -> list[dict[str, str]]:
    if split not in FILES:
        raise ValueError(f"Unknown split: {split!r}")
    path = ensure_data(data_dir)[split]
    rows = _read_jsonl(path)
    expected = EXPECTED_COUNTS[split]
    if len(rows) != expected:
        raise ValueError(
            f"{path} has {len(rows)} rows, expected {expected}. "
            "Delete it and rerun training to download it again."
        )
    examples: list[dict[str, str]] = []
    for row in rows:
        examples.append(
            {
                "text": build_text(row["passage"], row["question"]),
                "label": map_answer(row["answer"]),
            }
        )
    return examples


class BoolQDataset(Dataset):
    def __init__(
        self,
        examples: list[dict[str, str]],
        tokenizer: Any,
        label2id: dict[str, int],
        max_length: int,
        truncation_side: str = "left",
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.tokenizer.truncation_side = truncation_side

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        encoded = self.tokenizer(
            example["text"],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_attention_mask=True,
        )
        item = dict(encoded)
        item["labels"] = self.label2id[example["label"]]
        return item


def collate_fn(
    batch: list[dict[str, Any]],
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    max_length = max(len(item["input_ids"]) for item in batch)
    input_ids: list[list[int]] = []
    attention_masks: list[list[int]] = []
    token_type_ids: list[list[int]] = []
    labels: list[int] = []

    for item in batch:
        length = len(item["input_ids"])
        padding = max_length - length
        input_ids.append(item["input_ids"] + [pad_token_id] * padding)
        attention_masks.append(item["attention_mask"] + [0] * padding)
        token_type_ids.append(
            item.get("token_type_ids", [0] * length) + [0] * padding
        )
        labels.append(item["labels"])

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
