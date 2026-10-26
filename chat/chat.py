from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from chat.config import ChatConfig, list_available_models
from chat.model import Bit

EXAMPLE = "Paris is the capital of France. Is Paris in France?"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chat with Bit.")
    parser.add_argument(
        "--once",
        metavar="TEXT",
        help="Send one message and exit.",
    )
    parser.add_argument(
        "--probabilities",
        action="store_true",
        help="Show the probability of every class.",
    )
    parser.add_argument(
        "--model-dir",
        metavar="PATH",
        help="Path to model directory to load.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit.",
    )
    parser.add_argument(
        "--model",
        metavar="NAME",
        help="Model directory name to load from models/ (e.g., bit or bit-20261023_224737).",
    )
    return parser


def print_result(result: dict, show_probabilities: bool) -> None:
    if show_probabilities:
        probabilities = " ".join(
            f"{label}={probability:.3f}"
            for label, probability in result["probabilities"].items()
        )
        print(
            f"Bit: {result['label']} ({result['confidence']:.3f}) "
            f"[{probabilities}]"
        )
    else:
        print(f"Bit: {result['label']} ({result['confidence']:.3f})")


def select_model_interactive(models: list[tuple[str, Path]]) -> Path | None:
    if not models:
        return None
    print("Available models:")
    for idx, (display, path) in enumerate(models, start=1):
        print(f"  {idx}. {display} [{path}]")
    print(f"  {len(models) + 1}. Use default ({ChatConfig().model_dir})")
    while True:
        try:
            choice = input("> Select model [1]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if not choice:
            choice = "1"
        try:
            num = int(choice)
        except ValueError:
            print("Please enter a number.")
            continue
        if 1 <= num <= len(models):
            return models[num - 1][1]
        if num == len(models) + 1:
            return Path(ChatConfig().model_dir)
        print("Invalid choice.")


def main() -> int:
    args = build_parser().parse_args()
    config = ChatConfig()
    if args.probabilities:
        config = replace(config, show_probabilities=True)

    models = list_available_models()

    if args.list_models:
        if not models:
            print("No trained models found in models/")
            print(f"Default: {config.model_dir}")
            return 0
        print("Available models:")
        for display, path in models:
            print(f"  - {display} ({path})")
        print(f"Default: {config.model_dir}")
        return 0

    model_path = None
    if args.model_dir:
        model_path = Path(args.model_dir)
    elif args.model:
        candidate = Path("models") / args.model
        if candidate.exists():
            model_path = candidate
        else:
            # try as-is
            model_path = Path(args.model)

    if model_path is not None:
        config = replace(config, model_dir=str(model_path))
    elif not args.once and sys.stdin.isatty():
        selected = select_model_interactive(models)
        if selected is not None:
            config = replace(config, model_dir=str(selected))

    try:
        bit = Bit(config)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"Loaded Bit model from {config.model_dir}")

    if args.once is not None:
        print_result(bit.predict(args.once), config.show_probabilities)
        return 0

    print("Bit is ready.")
    print(
        "Send context and question together as one normal message, "
        "with no separator."
    )
    print(f"Example: {EXAMPLE}")
    print("Commands: /help, /example, /models, /load <name>, /quit")

    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not text:
            continue
        if text in {"/quit", "/exit"}:
            break
        if text == "/help":
            print(
                "Type context followed by your question in one message. "
                "Bit answers yes, no, or idk.\n"
                "Commands: /help, /example, /models, /load <name>, /quit"
            )
            continue
        if text == "/example":
            print(EXAMPLE)
            continue
        if text == "/models":
            models_list = list_available_models()
            if not models_list:
                print("No trained models found in models/")
            else:
                print("Available models:")
                for idx, (display, path) in enumerate(models_list, start=1):
                    print(f"  {idx}. {display} ({path})")
                print(f"Currently loaded: {config.model_dir}")
            continue
        if text.startswith("/load"):
            parts = text.split(maxsplit=1)
            if len(parts) < 2:
                print("Usage: /load <name or number>")
                continue
            target = parts[1].strip()
            models_list = list_available_models()
            chosen_path = None
            try:
                num = int(target)
                if 1 <= num <= len(models_list):
                    chosen_path = models_list[num - 1][1]
            except ValueError:
                pass
            if chosen_path is None:
                candidate = Path("models") / target
                if candidate.exists():
                    chosen_path = candidate
                elif Path(target).exists():
                    chosen_path = Path(target)
            if chosen_path is None:
                print(f"Could not find model: {target}")
                continue
            try:
                new_config = replace(config, model_dir=str(chosen_path))
                bit = Bit(new_config)
                config = new_config
                print(f"Loaded Bit model from {config.model_dir}")
            except FileNotFoundError as exc:
                print(exc, file=sys.stderr)
            continue

        print_result(bit.predict(text), config.show_probabilities)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
