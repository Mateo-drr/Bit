from __future__ import annotations

import json
import random
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from bit_model import BitClassifier
from train.config import TrainConfig
from train.data import BoolQDataset, collate_fn, load_split


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def make_lr_lambda(total_steps: int, warmup_steps: int):
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(total_steps - current_step)
            / float(max(1, total_steps - warmup_steps)),
        )

    return lr_lambda


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    labels: tuple[str, ...],
) -> dict:
    model.eval()
    total_loss = 0.0
    all_predictions: list[int] = []
    all_labels: list[int] = []

    with torch.inference_mode():
        for batch in loader:
            targets = batch.pop("labels").to(device, non_blocking=True)
            inputs = {
                name: tensor.to(device, non_blocking=True)
                for name, tensor in batch.items()
            }
            logits = model(**inputs)
            loss = F.cross_entropy(logits, targets)
            total_loss += loss.item() * targets.size(0)
            all_predictions.extend(logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(targets.cpu().tolist())

    label_ids = list(range(len(labels)))
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        labels=label_ids,
        zero_division=0,
    )
    per_class = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    }
    return {
        "loss": total_loss / max(1, len(all_labels)),
        "accuracy": float(accuracy_score(all_labels, all_predictions)),
        "macro_f1": float(
            f1_score(
                all_labels,
                all_predictions,
                labels=label_ids,
                average="macro",
                zero_division=0,
            )
        ),
        "per_class": per_class,
    }


def save_checkpoint(
    model: torch.nn.Module,
    tokenizer,
    config: TrainConfig,
    metrics: dict,
    epoch: int,
) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "bit_model.pt")
    tokenizer.save_pretrained(output_dir)
    metadata = {
        "model_name": config.model_name,
        "labels": list(config.labels),
        "label2id": config.label2id,
        "id2label": config.id2label,
        "max_length": config.max_length,
        "truncation_side": config.truncation_side,
        "head_hidden_dims": list(config.head_hidden_dims),
        "input_format": "passage + ' ' + question",
        "best_epoch": epoch,
    }
    (output_dir / "training_meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "train_config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":

    config = TrainConfig()
    set_seed(config.seed)

    device = resolve_device(config.device)
    use_fp16 = config.use_fp16 and device.type == "cuda"
    if config.use_fp16 and not use_fp16:
        print("CUDA is unavailable; continuing without fp16.")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.truncation_side = config.truncation_side

    train_examples = load_split(config.data_dir, "train")
    dev_examples = load_split(config.data_dir, "dev")
    train_dataset = BoolQDataset(
        train_examples,
        tokenizer,
        config.label2id,
        config.max_length,
        config.truncation_side,
    )
    dev_dataset = BoolQDataset(
        dev_examples,
        tokenizer,
        config.label2id,
        config.max_length,
        config.truncation_side,
    )

    collate = partial(collate_fn, pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )

    model = BitClassifier(
        model_name=config.model_name,
        num_labels=len(config.labels),
        hidden_dims=config.head_hidden_dims,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    scaler = torch.amp.GradScaler(device=device.type)

    print(
        f"Train rows: {len(train_dataset)} | Dev rows: {len(dev_dataset)} | "
        f"Epochs: {config.epochs}"
    )

    best_f1 = -1.0
    best_epoch = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            targets = batch.pop("labels").to(device)
            inputs = {
                name: tensor.to(device)
                for name, tensor in batch.items()
            }

            with torch.amp.autocast(device_type=config.device):
                logits = model(**inputs)
                loss = F.cross_entropy(logits, targets)

            running_loss += loss.item()
            scaler.scale(loss / config.grad_accum_steps).backward()

            should_step = (
                step % config.grad_accum_steps == 0
                or step == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.max_grad_norm,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        scheduler.step()

        if step % config.log_every == 0 or step == len(train_loader):
                current_lr = scheduler.get_last_lr()[0]
                average_loss = running_loss / step
                print(
                    f"epoch {epoch} step {step}/{len(train_loader)} "
                    f"loss {average_loss:.4f} lr {current_lr:.2e}"
                )

        metrics = evaluate(model, dev_loader, device, config.labels)
        print(
            f"epoch {epoch} eval | loss {metrics['loss']:.4f} | "
            f"accuracy {metrics['accuracy']:.4f} | "
            f"macro_f1 {metrics['macro_f1']:.4f}"
        )
        for label, values in metrics["per_class"].items():
            print(
                f"  {label:>3} | precision {values['precision']:.4f} | "
                f"recall {values['recall']:.4f} | f1 {values['f1']:.4f} | "
                f"support {values['support']}"
            )

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_epoch = epoch
            save_checkpoint(model, tokenizer, config, metrics, epoch)
            print(f"Saved new best model to {config.output_dir}")

    print(f"Best macro F1: {best_f1:.4f} at epoch {best_epoch}")
