# Bit

Bit is a BERT-based three-class classifier that answers a question or statement with `yes`, `no`, or `idk`. It is trained on the BoolQ_3L dataset with a manual PyTorch loop and a small Mish-activated classification head.

## Dataset

BoolQ_3L extends BoolQ with unanswerable examples:

| Split | Rows |
| --- | ---: |
| Train | 14,141 |
| Dev | 4,906 |

The files are downloaded automatically into `data/` from the [CogComp/Yes-No-or-IDK](https://github.com/CogComp/Yes-No-or-IDK) repository. Labels are mapped as follows:

| Raw answer | Label |
| --- | --- |
| `true` | `yes` |
| `false` | `no` |
| `"no-answer"` | `idk` |

The dataset is licensed under CC BY-SA 3.0.

## Setup

Use Python 3.14, which is the version this project is set up with.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Training

Start training with the module entry point or the installed console script:

```bash
python -m train.train
# or
bit-train
```

The dataset is downloaded on first use. The best checkpoint, tokenizer, metrics, and training metadata are saved to `models/bit/`.

Training defaults live in `TrainConfig` in `train/config.py`:

- `bert-base-cased` backbone
- 384 token maximum length, left truncation
- 2 linear classifier layers with Mish and dropout
- 3 epochs, batch size 8, gradient accumulation of 2
- AdamW with learning rate `2e-5`, warmup, and gradient clipping
- FP16 when CUDA is available

## Chat

Run the interactive chat:

```bash
python -m chat.chat
# or
bit-chat
```

Send the context and question together as one normal message. There is no required delimiter:

```text
> Paris is the capital of France. Is Paris in France?
Bit: yes (0.982)
```

The chat also supports one-shot input and probability output:

```bash
python -m chat.chat --once "Paris is the capital of France. Is Paris in France?"
python -m chat.chat --probabilities
```

Interactive commands:

- `/help` — show input instructions
- `/example` — show an example message
- `/quit` or `/exit` — leave the chat

Chat settings live in `ChatConfig` in `chat/config.py`.

## Tests

```bash
python -m unittest discover -s tests
```

## Project Layout

```text
bit_model.py          Shared BERT backbone and Mish classification head
train/
  config.py           Training hyperparameters
  data.py             Dataset download, parsing, tokenization, collation
  train.py            Manual PyTorch training and evaluation loop
chat/
  config.py           Chat and inference settings
  model.py            Checkpoint loading and inference wrapper
  chat.py             Interactive chat and one-shot CLI
tests/
  test_core.py        Text building and label mapping tests
```

## Notes

Training text is built as `passage + " " + question`. The 3-class task is harder than binary BoolQ; expected accuracy is roughly 70–80%, with `idk` usually the most difficult class.
