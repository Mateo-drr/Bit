from __future__ import annotations

import argparse
import sys

import program.config  # noqa: F401  (sets HF_HOME before transformers loads)
from chat.config import ChatConfig, list_available_models
from chat.model import Bit
from program.audio import (
    AudioError,
    RecordSettings,
    Recorder,
    format_audio_devices,
)
from program.config import (
    MODELS_ROOT,
    PIPER_ROOT,
    WHISPER_ROOT,
    resolve_model_dir,
)
from program.speech import (
    DEFAULT_TTS_VOICE,
    DEFAULT_WHISPER_MODEL,
    Speaker,
    Transcriber,
)

SPOKEN_ANSWERS = {"yes": "Yes.", "no": "No.", "idk": "I don't know."}
QUIT_COMMANDS = {"quit", "exit", "stop"}
LISTEN_PROMPT = "Listening... speak your context and question."


def spoken_answer(label: str) -> str:
    return SPOKEN_ANSWERS.get(label.strip().lower(), "I don't know.")


def is_quit_command(text: str) -> bool:
    return text.strip(" \t\r\n.!?").lower() in QUIT_COMMANDS


def conversation_loop(bit, recorder, transcriber, speaker, output=print) -> int:
    try:
        while True:
            output(LISTEN_PROMPT)
            audio = recorder.record()
            if audio is None or len(audio) == 0:
                output("I didn't hear anything. Try again.")
                continue

            text = transcriber.transcribe(audio).strip()
            if not text:
                output("I couldn't understand that. Try again.")
                continue

            output(f"You: {text}")
            if is_quit_command(text):
                output("Goodbye.")
                return 0

            result = bit.predict(text)
            label = result["label"]
            output(f"Bit: {label} ({result['confidence']:.3f})")
            speaker.speak(spoken_answer(label))
    except KeyboardInterrupt:
        output("")
        output("Goodbye.")
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Talk to Bit with your voice, fully locally."
    )
    parser.add_argument(
        "--model",
        metavar="NAME",
        help="Model directory name to load from models/.",
    )
    parser.add_argument(
        "--model-dir",
        metavar="PATH",
        help="Path to a model directory to load.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available Bit models and exit.",
    )
    parser.add_argument(
        "--list-audio-devices",
        action="store_true",
        help="List audio input and output devices and exit.",
    )
    parser.add_argument(
        "--asr-model",
        default=DEFAULT_WHISPER_MODEL,
        metavar="NAME",
        help="Whisper speech-recognition model (default: %(default)s).",
    )
    parser.add_argument(
        "--tts-voice",
        default=DEFAULT_TTS_VOICE,
        metavar="NAME",
        help="Piper text-to-speech voice (default: %(default)s).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch/Whisper device, for example auto, cuda, or cpu "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--input-device",
        type=int,
        default=None,
        metavar="INDEX",
        help="Microphone device index (default: system default).",
    )
    parser.add_argument(
        "--output-device",
        type=int,
        default=None,
        metavar="INDEX",
        help="Speaker device index (default: system default).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        metavar="RMS",
        help="Fixed speech threshold; omit to calibrate from ambient noise.",
    )
    parser.add_argument(
        "--silence-seconds",
        type=float,
        default=0.8,
        metavar="SECONDS",
        help="Stop recording after this much silence (default: %(default)s).",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=15.0,
        metavar="SECONDS",
        help="Maximum recording length (default: %(default)s).",
    )
    return parser


def print_models() -> None:
    models = list_available_models()
    if not models:
        print("No trained models found in models/")
        return
    print("Available models:")
    for display, path in models:
        print(f"  - {display} ({path})")
    print(f"Newest: {models[-1][1]}")


def run_voice(args: argparse.Namespace) -> int:
    try:
        model_dir = resolve_model_dir(args.model, args.model_dir)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    config = ChatConfig(model_dir=str(model_dir), device=args.device)
    try:
        bit = Bit(config)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"Loaded Bit model from {model_dir}")

    settings = RecordSettings(
        threshold=args.threshold,
        silence_seconds=args.silence_seconds,
        max_seconds=args.max_seconds,
    )
    recorder = Recorder(settings, device=args.input_device)
    transcriber = Transcriber(
        model_name=args.asr_model,
        device=args.device,
        download_root=WHISPER_ROOT,
    )
    speaker = Speaker(
        voice=args.tts_voice,
        download_root=PIPER_ROOT,
        output_device=args.output_device,
    )

    # Fail fast on an unusable microphone, before any large model download.
    try:
        recorder.calibrate()
    except AudioError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"Loading speech models (first run downloads into {MODELS_ROOT})...")
    try:
        transcriber.ensure_loaded()
        speaker.ensure_loaded()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    print("Bit voice is ready.")
    try:
        return conversation_loop(bit, recorder, transcriber, speaker)
    except (AudioError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        if args.list_models:
            print_models()
            return 0

        if args.list_audio_devices:
            try:
                print(format_audio_devices())
            except AudioError as exc:
                print(exc, file=sys.stderr)
                return 1
            return 0

        return run_voice(args)
    except KeyboardInterrupt:
        print()
        print("Goodbye.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
