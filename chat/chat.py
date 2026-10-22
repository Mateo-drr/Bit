from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from chat.config import ChatConfig
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


def main() -> int:
    args = build_parser().parse_args()
    config = ChatConfig()
    if args.probabilities:
        config = replace(config, show_probabilities=True)

    try:
        bit = Bit(config)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    if args.once is not None:
        print_result(bit.predict(args.once), config.show_probabilities)
        return 0

    print("Bit is ready.")
    print(
        "Send context and question together as one normal message, "
        "with no separator."
    )
    print(f"Example: {EXAMPLE}")
    print("Commands: /help, /example, /quit")

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
                "Bit answers yes, no, or idk."
            )
            continue
        if text == "/example":
            print(EXAMPLE)
            continue

        print_result(bit.predict(text), config.show_probabilities)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
