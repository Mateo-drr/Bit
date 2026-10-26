from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from bit_model import BitClassifier
from train.config import TrainConfig
from train.data import BoolQDataset, collate_fn, load_split

try:
    import wandb
except ImportError:
    wandb = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def split_batch(batch: dict, device: torch.device) -> tuple[dict, torch.Tensor]:
    targets = batch.pop("labels").to(device, non_blocking=True)
    inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    return inputs, targets


def compute_metrics(
    targets: list[int], preds: list[int], labels: tuple[str, ...]
) -> dict:
    ids = list(range(len(labels)))
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, preds, labels=ids, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(targets, preds)),
        "macro_f1": float(
            f1_score(targets, preds, labels=ids, average="macro", zero_division=0)
        ),
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        },
    }


@torch.inference_mode()
def evaluate(model, loader, device, labels, class_weights, amp) -> dict:
    model.eval()
    total_loss = 0.0
    preds: list[int] = []
    targets_all: list[int] = []

    for batch in loader:
        inputs, targets = split_batch(batch, device)
        with amp():
            logits = model(**inputs)
        logits = logits.float()
        total_loss += F.cross_entropy(logits, targets, weight=class_weights).item() * targets.size(0)
        preds.extend(logits.argmax(-1).tolist())
        targets_all.extend(targets.tolist())

    metrics = compute_metrics(targets_all, preds, labels)
    metrics["loss"] = total_loss / len(targets_all)
    return metrics


def save_checkpoint(model, tokenizer, config: TrainConfig, metrics: dict, epoch: int) -> None:
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "bit_model.pt")
    tokenizer.save_pretrained(out)

    meta = {
        "model_name": config.model_name,
        "labels": list(config.labels),
        "label2id": config.label2id,
        "id2label": config.id2label,
        "max_length": config.max_length,
        "truncation_side": config.truncation_side,
        "head_hidden_dims": config.head_hidden_dims,
        "input_format": "passage + ' ' + question",
        "best_epoch": epoch,
    }
    for name, payload in [
        ("training_meta.json", meta),
        ("metrics.json", metrics),
        ("train_config.json", asdict(config)),
    ]:
        (out / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    config = TrainConfig()
    set_seed(config.seed)
    device = resolve_device(config.device)

    # Mixed precision: bf16 when the GPU supports it (no scaler needed), else fp16 + scaler.
    use_amp = config.use_fp16 and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    amp = partial(torch.autocast, device_type=device.type, dtype=amp_dtype, enabled=use_amp)
    scaler = torch.amp.GradScaler(
        device.type, enabled=use_amp and amp_dtype == torch.float16
    )

    use_wandb = config.wb and wandb is not None
    if use_wandb:
        wandb.init(project=config.project_name, config=asdict(config))

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    train_examples = load_split(config.data_dir, "train")
    dev_examples = load_split(config.data_dir, "dev")
    make_dataset = lambda examples: BoolQDataset(
        examples, tokenizer, config.label2id, config.max_length, config.truncation_side
    )
    train_dataset, dev_dataset = make_dataset(train_examples), make_dataset(dev_examples)

    collate = partial(collate_fn, pad_token_id=tokenizer.pad_token_id)
    make_loader = lambda ds, shuffle: DataLoader(
        ds,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )
    train_loader, dev_loader = make_loader(train_dataset, True), make_loader(dev_dataset, False)

    counts = Counter(ex["label"] for ex in train_examples)
    total = sum(counts.values())
    class_weights = torch.tensor(
        [total / (len(config.labels) * counts[label]) for label in config.labels],
        dtype=torch.float,
        device=device,
    )

    model = BitClassifier(
        model_name=config.model_name,
        num_labels=len(config.labels),
        hid_dim=config.head_hidden_dims,
        dropout=config.dropout
    ).to(device)

    head_params = [p for n, p in model.named_parameters() if not n.startswith("bert.")]
    param_groups = [{"params": head_params, "lr": config.learning_rate}]
    if config.bert_learning_rate is None:
        model.bert.requires_grad_(False)
    else:
        param_groups.insert(0, {"params": model.bert.parameters(), "lr": config.bert_learning_rate})
    optimizer = AdamW(param_groups, weight_decay=config.weight_decay)

    steps_per_epoch = math.ceil(len(train_loader) / config.grad_accum_steps)
    total_steps = config.epochs * steps_per_epoch
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    print(
        f"Train rows: {len(train_dataset)} | Dev rows: {len(dev_dataset)} | "
        f"Epochs: {config.epochs} | Optimizer steps: {total_steps}"
    )
    print(model)

    best_f1, best_loss, best_epoch = -1.0, float("inf"), 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        train_preds: list[torch.Tensor] = []
        train_targets: list[torch.Tensor] = []

        bar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}")
        for step, batch in enumerate(bar, start=1):
            inputs, targets = split_batch(batch, device)

            with amp():
                logits = model(**inputs)
            logits = logits.float()
            loss = F.cross_entropy(
                logits,
                targets,
                weight=class_weights,
                label_smoothing=config.label_smoothing,
            )
            scaler.scale(loss / config.grad_accum_steps).backward()

            if step % config.grad_accum_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            train_preds.append(logits.argmax(-1).detach())
            train_targets.append(targets)
            bar.set_postfix(loss=f"{running_loss / step:.4f}")

        train_loss = running_loss / len(train_loader)
        train_metrics = compute_metrics(
            torch.cat(train_targets).tolist(), torch.cat(train_preds).tolist(), config.labels
        )
        lr = scheduler.get_last_lr()[0]
        print(
            f"epoch {epoch} | train loss {train_loss:.4f} | "
            f"acc {train_metrics['accuracy']:.4f} | macro_f1 {train_metrics['macro_f1']:.4f} | lr {lr:.2e}"
        )

        metrics = evaluate(model, dev_loader, device, config.labels, class_weights, amp)
        print(
            f"epoch {epoch} eval | loss {metrics['loss']:.4f} | "
            f"acc {metrics['accuracy']:.4f} | macro_f1 {metrics['macro_f1']:.4f}"
        )
        for label, v in metrics["per_class"].items():
            print(
                f"  {label:>3} | precision {v['precision']:.4f} | "
                f"recall {v['recall']:.4f} | f1 {v['f1']:.4f} | support {v['support']}"
            )

        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/accuracy": train_metrics["accuracy"],
                    "train/f1": train_metrics["macro_f1"],
                    "val/loss": metrics["loss"],
                    "val/accuracy": metrics["accuracy"],
                    "val/f1": metrics["macro_f1"],
                    "lr": lr,
                }
            )

        if metrics["macro_f1"] > best_f1 and metrics["loss"] < best_loss:
            best_f1, best_loss, best_epoch = metrics["macro_f1"], metrics["loss"], epoch
            save_checkpoint(model, tokenizer, config, metrics, epoch)
            print(f"Saved new best model to {config.output_dir}")

    print(f"Best macro F1: {best_f1:.4f} at epoch {best_epoch}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()