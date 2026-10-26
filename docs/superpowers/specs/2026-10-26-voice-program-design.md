# Bit voice program — design

Date: 2026-10-26
Status: awaiting approval

## Intent

Add a fully local voice interface to a trained Bit model. The user speaks a
context-and-question in one message, the program records it, transcribes it,
sends the text to Bit, prints `yes` / `no` / `idk`, and speaks the answer aloud.

## Goals

- Fully local after first-run model downloads.
- No API keys or cloud services.
- Reuse `chat.model.Bit` and `chat.config.ChatConfig`.
- Record with automatic silence detection.
- Print the transcript, answer, and confidence; speak the answer.
- Save every downloaded asset under `program/models/`.
- Work with any checkpoint layout supported by `BitClassifier.from_state_dict`.

## Non-goals

- Wake-word detection.
- Streaming/partial transcription.
- Speaker diarisation or multi-user support.
- Training or fine-tuning.
- GUI or web interface.

## Stack (chosen)

- Recording and playback: `sounddevice` over the installed PortAudio.
- Speech-to-text: `faster-whisper` with `small.en`, CUDA when available.
- Text-to-speech: `piper-tts` with the `en_US-lessac-medium` voice.
- Model: newest directory in `models/`, unless overridden.

Rejected alternatives: system `whisper-cpp` + `espeak-ng` (more shell plumbing,
robotic voice) and `edge-tts` (network dependency).

## Files

```text
program/
  __init__.py
  audio.py          Microphone capture, silence detection, playback
  speech.py         faster-whisper transcriber and Piper speaker
  voice_chat.py     CLI, model selection, conversation loop
tests/
  test_voice_program.py
```

Other changes:

- `pyproject.toml`: add optional `voice` dependencies and `bit-voice` script.
- `.gitignore`: ignore `program/models/`.
- `README.md`: short voice-usage section.

## Downloads

Everything goes under `program/models/`:

```text
program/models/
  faster-whisper/   Whisper weights
  piper/            Piper voice .onnx and .onnx.json
  huggingface/      HF cache used by the Bit backbone
```

`HF_HOME` is set before importing `transformers` so the backbone also uses the
local cache. First run downloads roughly 600 MB and prints its progress.

## Conversation loop

1. Load the newest model from `models/` (or the `--model` / `--model-dir`).
2. Load the Whisper model and the Piper voice from `program/models/`.
3. Print `Listening...`, record until ~0.8 s of silence, with a 15 s cap.
4. Transcribe the recording.
5. If the transcript is empty, prompt again.
6. If the transcript is `quit`, `exit`, or `stop`, exit.
7. Print `You: <transcript>`.
8. Run `Bit.predict(text)` and print `Bit: <label> (<confidence>)`.
9. Speak `Yes.`, `No.`, or `I don't know.`
10. Repeat until Ctrl+C or a quit command.

## Audio details

- 16 kHz mono float32 capture, 32 ms blocks.
- Energy-based VAD: 3 blocks above threshold to start, 20 below to stop.
- Keep 5 blocks of pre-roll so the first word is not clipped.
- Calibrate the noise threshold for 0.4 s at startup, with a fixed fallback.
- `--threshold`, `--silence-seconds`, and `--max-seconds` overrides.
- Playback blocks until the spoken answer finishes before listening again.

## CLI

```bash
.venv/bin/python -m program.voice_chat
.venv/bin/python -m program.voice_chat --model bit-20261024_232820
.venv/bin/python -m program.voice_chat --model-dir models/bit-20261024_232820
.venv/bin/python -m program.voice_chat --list-models
.venv/bin/python -m program.voice_chat --list-audio-devices
```

Additional options: `--asr-model`, `--device`, `--tts-voice`, `--input-device`,
`--output-device`, `--threshold`, `--silence-seconds`, `--max-seconds`.

## Error handling

- Missing voice dependencies: explain `pip install -e '.[voice]'`.
- No microphone or model directory: exit with a clear message.
- Empty or failed transcription: print the reason and keep listening.
- Ctrl+C: clean shutdown without a traceback.

## Testing

Unit tests use fakes for the microphone, transcriber, Bit model, and speaker:

- spoken-answer mapping
- VAD start/stop on synthetic audio
- conversation loop prints and speaks the expected answer
- quit command exits
- model resolution picks the newest checkpoint and honours overrides

Manual verification: run with the real mic, ask
`Paris is the capital of France. Is Paris in France?`, confirm the printed
transcript, `yes (0.956)`, and audible “Yes.”

## Constraints

- Python 3.14, CachyOS, PipeWire, RTX 2060 12 GB.
- New packages: `sounddevice`, `faster-whisper`, `piper-tts`.
- Everything runs from the project’s existing virtual environment.
