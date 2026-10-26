# Bit Voice Program Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fully local voice interface that records a spoken context-and-question, transcribes it, asks Bit, prints the answer, and speaks it aloud.

**Architecture:** `program/audio.py` captures 16 kHz mono audio with an RMS silence detector and plays Piper output. `program/speech.py` wraps faster-whisper and Piper, loading models lazily from `program/models/`. `program/voice_chat.py` resolves the newest checkpoint, wires the components, and runs a print-and-speak loop.

**Tech Stack:** Python 3.14, `sounddevice` 0.5.6, `faster-whisper` 1.2.1, `piper-tts` 1.8.0, existing `chat.model.Bit`, `unittest`.

**Spec:** `docs/superpowers/specs/2026-10-26-voice-program-design.md`

## Global Constraints

- Fully local after first-run downloads; no API keys or cloud services.
- Every downloaded asset lives under `program/models/`: `faster-whisper/`, `piper/`, `huggingface/`.
- Reuse `chat.config.ChatConfig` and `chat.model.Bit`; do not create a second checkpoint loader.
- Capture 16 kHz mono float32; stop after 0.8 s of silence; hard cap 15 s.
- Print `You: <transcript>` and `Bit: <label> (<confidence:.3f>)`.
- Speak exactly `Yes.`, `No.`, or `I don't know.`; keep listening after empty audio or transcript.
- Platform: CachyOS/PipeWire, Python 3.14, RTX 2060 12 GB; CUDA when available, CPU fallback.
- No wake word, streaming transcription, diarisation, GUI, or web interface.

## Review Focus

- The user says nothing or the microphone is silent → the program retries without sending empty text to Bit (Task 2 capture, Task 4 loop).
- Whisper returns an empty transcript → the program says it could not understand and keeps listening (Task 4).
- A question runs past the 15 s cap → capture stops at the limit and still makes a prediction (Task 2).
- No checkpoint exists or `--model` names an unknown directory → exit 1 with a clear message, no traceback (Task 1).
- A checkpoint contains an unexpected label → speak "I don't know." instead of raising `KeyError` (Task 4).

## File Structure

| File | Responsibility |
| --- | --- |
| `program/__init__.py` | Package marker |
| `program/config.py` | Local asset roots, HF cache, checkpoint resolution |
| `program/audio.py` | 16 kHz capture, RMS VAD, playback, device listing |
| `program/speech.py` | faster-whisper transcriber and Piper speaker |
| `program/voice_chat.py` | CLI wiring and conversation loop |
| `tests/test_voice_program.py` | Unit tests with fake mic/ASR/TTS/model |
| `pyproject.toml` | Voice extra, `program` package, `bit-voice` script |
| `.gitignore` | Ignore `program/models/` |
| `README.md` | Voice usage docs |

---

### Task 1: Program Paths and Model Resolution

**Files:**
- Create: `program/__init__.py`
- Create: `program/config.py`
- Modify: `pyproject.toml`
- Modify: `.gitignore`
- Test: `tests/test_voice_program.py`

**Interfaces:**
- Consumes: `chat.config.list_available_models(models_root: str | Path = "models") -> list[tuple[str, Path]]`
- Produces:
  - `MODELS_ROOT: Path` — `program/models`
  - `WHISPER_ROOT: Path` — `program/models/faster-whisper`
  - `PIPER_ROOT: Path` — `program/models/piper`
  - `HF_ROOT: Path` — `program/models/huggingface`
  - `VOICE_INSTALL_HINT: str` — message beginning `Voice dependencies are missing.`
  - `resolve_model_dir(model: str | None = None, model_dir: str | None = None, models_root: str | Path = "models") -> Path`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_voice_program.py`:

```python
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

import program
from program import config


class ConfigTests(unittest.TestCase):
    def write_model(self, root: Path, name: str) -> Path:
        model_dir = root / name
        model_dir.mkdir()
        (model_dir / "bit_model.pt").write_bytes(b"")
        (model_dir / "training_meta.json").write_text("{}", encoding="utf-8")
        return model_dir

    def test_download_roots_live_under_program_directory(self):
        self.assertEqual(
            config.MODELS_ROOT,
            Path(program.__file__).resolve().parent / "models",
        )
        self.assertEqual(config.WHISPER_ROOT, config.MODELS_ROOT / "faster-whisper")
        self.assertEqual(config.PIPER_ROOT, config.MODELS_ROOT / "piper")
        self.assertEqual(config.HF_ROOT, config.MODELS_ROOT / "huggingface")

    def test_resolve_model_dir_picks_newest_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_model(root, "bit-20260101_000000")
            newer = self.write_model(root, "bit-20260102_000000")
            self.assertEqual(config.resolve_model_dir(models_root=root), newer)

    def test_resolve_model_dir_honours_name_and_path_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            older = self.write_model(root, "bit-20260101_000000")
            self.write_model(root, "bit-20260102_000000")
            self.assertEqual(
                config.resolve_model_dir("bit-20260101_000000", models_root=root),
                older,
            )
            self.assertEqual(
                config.resolve_model_dir(model_dir=str(older), models_root=root),
                older,
            )

    def test_resolve_model_dir_reports_missing_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, "No trained Bit models"):
                config.resolve_model_dir(models_root=root)
            with self.assertRaisesRegex(FileNotFoundError, "Could not find Bit model"):
                config.resolve_model_dir("nope", models_root=root)

    def test_project_declares_voice_package_and_script(self):
        project_root = Path(__file__).resolve().parents[1]
        data = tomllib.loads(
            (project_root / "pyproject.toml").read_text(encoding="utf-8")
        )
        self.assertIn("program", data["tool"]["setuptools"]["packages"])
        self.assertEqual(
            data["project"]["scripts"]["bit-voice"],
            "program.voice_chat:main",
        )
        self.assertEqual(
            set(data["project"]["optional-dependencies"]["voice"]),
            {
                "sounddevice==0.5.6",
                "faster-whisper==1.2.1",
                "piper-tts==1.8.0",
            },
        )

    def test_voice_program_forces_local_hf_cache(self):
        project_root = Path(__file__).resolve().parents[1]
        env = {key: value for key, value in os.environ.items() if key != "HF_HOME"}
        env["HF_HOME"] = str(project_root / "wrong-cache")
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import os, program.config; print(os.environ['HF_HOME'])",
            ],
            cwd=project_root,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), str(config.HF_ROOT))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m unittest tests.test_voice_program.ConfigTests -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'program'`.

- [ ] **Step 3: Implement the package and resolution**

Create `program/__init__.py`:

```python
"""Fully local voice interface for Bit."""
```

Create `program/config.py`:

```python
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
```

Modify `pyproject.toml`:

```toml
[project.optional-dependencies]
voice = [
    "sounddevice==0.5.6",
    "faster-whisper==1.2.1",
    "piper-tts==1.8.0",
]

[project.scripts]
bit-train = "train.train:train"
bit-chat = "chat.chat:main"
bit-voice = "program.voice_chat:main"

[tool.setuptools]
py-modules = ["bit_model"]
packages = ["train", "chat", "program"]
```

Add this line to `.gitignore` after `models/`:

```text
program/models/
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m unittest tests.test_voice_program.ConfigTests -v`

Expected: PASS (6 tests).

- [ ] **Step 5: Verify the asset directory is ignored**

Run: `git check-ignore -v program/models/example`

Expected: output contains `.gitignore` and `program/models/`.

- [ ] **Step 6: Commit**

```bash
git add program/__init__.py program/config.py tests/test_voice_program.py pyproject.toml .gitignore docs/superpowers/specs/2026-10-26-voice-program-design.md docs/superpowers/plans/2026-10-26-voice-program.md
git commit -m "feat: scaffold local voice program"
```

---

### Task 2: Microphone Capture and Silence Detection

**Files:**
- Create: `program/audio.py`
- Modify: `tests/test_voice_program.py`
- Test: `tests/test_voice_program.py`

**Interfaces:**
- Consumes: `numpy`, `sounddevice` (installed by this task's extra).
- Produces:
  - `SAMPLE_RATE = 16000`, `BLOCK_SECONDS = 0.032`, `BLOCK_SIZE = 512`, `MIN_THRESHOLD = 0.01`, `NOISE_MULTIPLIER = 3.0`
  - `RecordSettings(sample_rate=16000, block_seconds=0.032, silence_seconds=0.8, max_seconds=15.0, threshold=None, calibration_seconds=0.4, start_blocks=3, preroll_blocks=5)` with property `block_size: int`
  - `AudioError(RuntimeError)`
  - `rms(block: np.ndarray) -> float`
  - `read_blocks(stream, block_size: int) -> Iterator[np.ndarray]`
  - `calibration_block_count(settings: RecordSettings) -> int`
  - `calibrate_threshold(blocks: Iterable[np.ndarray], settings: RecordSettings) -> float`
  - `record_blocks(blocks: Iterable[np.ndarray], settings: RecordSettings) -> np.ndarray`
  - `Recorder(settings: RecordSettings | None = None, device: int | None = None)` with `sample_rate: int` property, `calibrate() -> None`, and `record() -> np.ndarray`
  - `play(audio: np.ndarray, sample_rate: int, device: int | None = None) -> None`
  - `format_audio_devices() -> str`

- [ ] **Step 1: Install the voice extras**

Run: `.venv/bin/pip install -e '.[voice]'`

Expected: `Successfully installed ... sounddevice-0.5.6 ... faster-whisper-1.2.1 ... piper-tts-1.8.0 ...`

- [ ] **Step 2: Verify the audio dependency imports**

Run: `.venv/bin/python -c "import sounddevice; print(sounddevice.__version__)"`

Expected: `0.5.6`.

- [ ] **Step 3: Write the failing tests**

Ensure `import numpy as np` is present in `tests/test_voice_program.py`, add `from program.audio import RecordSettings, record_blocks`, and append:

```python
def make_blocks(seconds, amplitude, settings):
    count = round(seconds / settings.block_seconds)
    return [
        np.full(settings.block_size, amplitude, dtype=np.float32)
        for _ in range(count)
    ]


class AudioTests(unittest.TestCase):
    def test_record_blocks_captures_speech_and_stops_after_silence(self):
        settings = RecordSettings(
            threshold=0.05, silence_seconds=0.2, max_seconds=5.0
        )
        blocks = (
            make_blocks(0.16, 0.0, settings)
            + make_blocks(0.2, 0.2, settings)
            + make_blocks(0.5, 0.0, settings)
        )
        audio = record_blocks(blocks, settings)
        duration = len(audio) / settings.sample_rate
        self.assertGreaterEqual(duration, 0.4)
        self.assertLessEqual(duration, 0.8)
        self.assertGreater(float(np.max(np.abs(audio))), 0.1)

    def test_record_blocks_returns_empty_without_speech(self):
        settings = RecordSettings(
            threshold=0.05, silence_seconds=0.2, max_seconds=0.5
        )
        audio = record_blocks(make_blocks(1.0, 0.0, settings), settings)
        self.assertEqual(audio.size, 0)

    def test_record_blocks_stops_at_max_seconds(self):
        settings = RecordSettings(
            threshold=0.05, silence_seconds=1.0, max_seconds=0.64
        )
        audio = record_blocks(make_blocks(2.0, 0.2, settings), settings)
        duration = len(audio) / settings.sample_rate
        self.assertGreaterEqual(duration, 0.5)
        self.assertLessEqual(
            duration, settings.max_seconds + settings.block_seconds
        )

    def test_record_blocks_calibrates_threshold_from_noise(self):
        settings = RecordSettings(
            threshold=None, silence_seconds=0.2, max_seconds=2.0
        )
        blocks = (
            make_blocks(0.4, 0.05, settings)
            + make_blocks(0.3, 0.1, settings)
            + make_blocks(0.5, 0.0, settings)
        )
        audio = record_blocks(blocks, settings)
        self.assertEqual(audio.size, 0)
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `.venv/bin/python -m unittest tests.test_voice_program.AudioTests -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'program.audio'`.

- [ ] **Step 5: Implement `program/audio.py`**

Use these exact constants and rules:

```python
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import Iterable, Iterator

import numpy as np

SAMPLE_RATE = 16000
BLOCK_SECONDS = 0.032
BLOCK_SIZE = 512
MIN_THRESHOLD = 0.01
NOISE_MULTIPLIER = 3.0


class AudioError(RuntimeError):
    """Raised when the microphone cannot be opened."""


@dataclass(frozen=True)
class RecordSettings:
    sample_rate: int = SAMPLE_RATE
    block_seconds: float = BLOCK_SECONDS
    silence_seconds: float = 0.8
    max_seconds: float = 15.0
    threshold: float | None = None
    calibration_seconds: float = 0.4
    start_blocks: int = 3
    preroll_blocks: int = 5

    @property
    def block_size(self) -> int:
        return max(1, round(self.sample_rate * self.block_seconds))
```

Implement the remaining functions with these rules:

- `rms` flattens the block, converts to `float64`, and returns `sqrt(mean(square))`; an empty block returns `0.0`.
- `read_blocks` loops on `stream.read(block_size)` and yields the first element of the returned `(data, overflowed)` tuple.
- `calibration_block_count` returns `0` when `settings.threshold` is fixed, otherwise `max(1, round(calibration_seconds / block_seconds))`.
- `calibrate_threshold` returns the fixed threshold when one is set; otherwise it returns `max(MIN_THRESHOLD, mean(noise_rms) * NOISE_MULTIPLIER)`, using `MIN_THRESHOLD` when no blocks are available.
- `record_blocks`:
  - Consume `calibration_block_count(settings)` blocks, compute the threshold with `calibrate_threshold(calibration, settings)`, and use the fixed threshold when calibration was skipped.
  - Seed a `deque(maxlen=preroll_blocks)` with the last calibration blocks.
  - Before speech, append every block to the pre-roll and start when `start_blocks` consecutive blocks reach the threshold. Copy pre-roll blocks with `np.array(..., dtype=np.float32)` so reused buffers cannot alias.
  - After starting, append blocks, stop when the trailing silence reaches `silence_seconds`, and stop when `len(recorded) * block_seconds >= max_seconds`. The design's "20 blocks below" is the same intent; the exact count is derived from `--silence-seconds` (25 blocks at the 0.8 s / 32 ms default).
  - While waiting for speech, stop after `round(max_seconds / block_seconds)` blocks.
  - Return an empty `float32` array when speech never starts; otherwise return the concatenated recording.
- `Recorder.calibrate` is called once at startup. It stores `settings.threshold` when one is fixed. Otherwise it opens a short `sounddevice.InputStream`, reads `calibration_block_count(settings)` blocks, stores `calibrate_threshold(blocks, settings)`, and then closes the stream. It converts `sounddevice.PortAudioError` into `AudioError` naming the device.
- `Recorder.record` calls `calibrate()` if no threshold has been stored yet, then opens `sounddevice.InputStream(samplerate=..., blocksize=..., channels=1, dtype="float32", device=...)` as a context manager. It passes `read_blocks(stream, settings.block_size)` to `record_blocks` with a `dataclasses.replace(settings, threshold=stored_threshold)` copy so the first spoken word is not consumed by calibration. It converts `sounddevice.PortAudioError` into `AudioError` naming the device.
- `play` flattens the array, returns early when empty, then calls `sounddevice.play(..., samplerate=sample_rate, device=device)` followed by `sounddevice.wait()`.
- `format_audio_devices` returns one line per device: `"{index}: {name} in={max_input_channels} out={max_output_channels}"`, appending `" (default input)"`, `" (default output)"`, or both.

Keep all `sounddevice` imports inside functions so the module imports without audio hardware.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/bin/python -m unittest tests.test_voice_program.AudioTests -v`

Expected: PASS (4 tests).

- [ ] **Step 7: Commit**

```bash
git add program/audio.py tests/test_voice_program.py
git commit -m "feat: add silence-aware microphone capture"
```

---

### Task 3: Local Speech-to-Text and Text-to-Speech

**Files:**
- Create: `program/speech.py`
- Modify: `tests/test_voice_program.py`
- Test: `tests/test_voice_program.py`

**Interfaces:**
- Consumes: `program.audio.play`, `program.config.WHISPER_ROOT`, `program.config.PIPER_ROOT`, `program.config.VOICE_INSTALL_HINT`.
- Produces:
  - `DEFAULT_WHISPER_MODEL = "small.en"`
  - `DEFAULT_TTS_VOICE = "en_US-lessac-medium"`
  - `missing_dependency_error(exc: ImportError) -> RuntimeError`
  - `Transcriber(model_name="small.en", device="auto", download_root=WHISPER_ROOT, compute_type="default", model_factory=None)` with `ensure_loaded() -> None` and `transcribe(audio: np.ndarray) -> str`
  - `Speaker(voice="en_US-lessac-medium", download_root=PIPER_ROOT, use_cuda=False, output_device=None)` with `ensure_loaded() -> None`, `synthesize(text: str) -> tuple[np.ndarray, int]`, and `speak(text: str) -> None`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_voice_program.py`. Ensure these imports are present (`numpy` was added in Task 2): `from types import SimpleNamespace`, `from unittest import mock`, `from program import speech`, and `from program.speech import Speaker, Transcriber`.

```python
class FakeWhisperModel:
    def __init__(self, segments):
        self.segments = segments
        self.kwargs = None

    def transcribe(self, audio, **kwargs):
        self.kwargs = kwargs
        return iter(self.segments), SimpleNamespace(language="en")


class SpeechTests(unittest.TestCase):
    def test_missing_dependency_error_includes_install_hint(self):
        error = speech.missing_dependency_error(ImportError("nope"))
        self.assertIn("pip install -e '.[voice]'", str(error))
        self.assertIn("nope", str(error))

    def test_transcriber_joins_segments_and_uses_english_vad(self):
        model = FakeWhisperModel(
            [
                SimpleNamespace(text=" Hello "),
                SimpleNamespace(text="world "),
                SimpleNamespace(text=" "),
            ]
        )
        transcriber = Transcriber()
        transcriber._model = model
        transcript = transcriber.transcribe(np.zeros(1600, dtype=np.float32))
        self.assertEqual(transcript, "Hello world")
        self.assertEqual(model.kwargs["language"], "en")
        self.assertTrue(model.kwargs["vad_filter"])
        self.assertFalse(model.kwargs["condition_on_previous_text"])

    def test_transcriber_falls_back_to_cpu_when_gpu_load_fails(self):
        calls = []

        def factory(name, *, device, compute_type, download_root):
            calls.append((device, compute_type))
            if device != "cpu":
                raise RuntimeError("CUDA unavailable")
            return FakeWhisperModel([])

        transcriber = Transcriber(device="auto", model_factory=factory)
        transcriber.ensure_loaded()
        self.assertEqual(calls, [("auto", "default"), ("cpu", "int8")])

    def test_speaker_concatenates_chunks(self):
        voice = SimpleNamespace(
            synthesize=lambda text: iter(
                [
                    SimpleNamespace(
                        audio_float_array=np.array(
                            [0.1, -0.1], dtype=np.float32
                        ),
                        sample_rate=22050,
                    ),
                    SimpleNamespace(
                        audio_float_array=np.array([0.2], dtype=np.float32),
                        sample_rate=22050,
                    ),
                ]
            )
        )
        speaker = Speaker()
        speaker._voice = voice
        audio, sample_rate = speaker.synthesize("Yes.")
        np.testing.assert_allclose(audio, [0.1, -0.1, 0.2])
        self.assertEqual(sample_rate, 22050)

    def test_speaker_plays_synthesized_audio(self):
        speaker = Speaker(output_device=3)
        speaker.synthesize = mock.Mock(
            return_value=(np.array([0.5], dtype=np.float32), 22050)
        )
        with mock.patch("program.speech.play") as play:
            speaker.speak("Yes.")
        play.assert_called_once()
        audio, sample_rate = play.call_args.args
        np.testing.assert_allclose(audio, [0.5])
        self.assertEqual(sample_rate, 22050)
        self.assertEqual(play.call_args.kwargs["device"], 3)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m unittest tests.test_voice_program.SpeechTests -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'program.speech'`.

- [ ] **Step 3: Implement `program/speech.py`**

Implement with these exact behaviors:

- `missing_dependency_error` returns `RuntimeError(f"{VOICE_INSTALL_HINT}\nOriginal error: {exc}")`.
- `Transcriber.ensure_loaded` returns early when `_model` exists. Otherwise import `faster_whisper.WhisperModel` (wrapping `ImportError` with `missing_dependency_error`) or use the injected `model_factory`. Call it as `factory(model_name, device=device, compute_type=compute_type, download_root=str(download_root))`.
- If that load raises and `device != "cpu"`, print a fallback line and retry with `device="cpu", compute_type="int8"`. If CPU also fails, raise a `RuntimeError` naming the model and the underlying error.
- `Transcriber.transcribe` returns `""` for empty audio. Otherwise call `model.transcribe(samples, language="en", vad_filter=True, condition_on_previous_text=False)` and join non-empty `segment.text.strip()` values with single spaces.
- `Speaker.ensure_loaded` returns early when `_voice` exists. Otherwise create `download_root`, import `piper.download_voices.download_voice` and `piper.voice.PiperVoice` (wrapping `ImportError`), and download `"{voice}.onnx"` / `"{voice}.onnx.json"` with `download_voice(self.voice, self.download_root)` when either file is missing. Load with `PiperVoice.load(model_path, config_path=config_path, use_cuda=self.use_cuda)`. Wrap download and load failures in `RuntimeError` naming the voice and path.
- `Speaker.synthesize` returns `(empty float32 array, 22050)` when there are no chunks; otherwise concatenate `chunk.audio_float_array` into one `float32` array and return it with the first chunk's `sample_rate`.
- `Speaker.speak` calls `synthesize` and then `play(audio, sample_rate, device=self.output_device)`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m unittest tests.test_voice_program.SpeechTests -v`

Expected: PASS (5 tests).

- [ ] **Step 5: Verify the real speech packages import**

Run: `.venv/bin/python -c "from faster_whisper import WhisperModel; from piper.voice import PiperVoice; print('speech deps ok')"`

Expected: `speech deps ok` (warnings are fine).

- [ ] **Step 6: Commit**

```bash
git add program/speech.py tests/test_voice_program.py
git commit -m "feat: add local speech models"
```

---

### Task 4: Voice Conversation Loop and CLI

**Files:**
- Create: `program/voice_chat.py`
- Modify: `tests/test_voice_program.py`
- Modify: `README.md`
- Test: `tests/test_voice_program.py`

**Interfaces:**
- Consumes: `resolve_model_dir`, `MODELS_ROOT`, `WHISPER_ROOT`, `PIPER_ROOT`, `RecordSettings`, `Recorder`, `AudioError`, `format_audio_devices`, `Transcriber`, `Speaker`, `DEFAULT_WHISPER_MODEL`, `DEFAULT_TTS_VOICE`, `missing_dependency_error`, `ChatConfig`, `Bit`, `list_available_models`.
- Produces:
  - `SPOKEN_ANSWERS: dict[str, str]`
  - `spoken_answer(label: str) -> str`
  - `is_quit_command(text: str) -> bool`
  - `conversation_loop(bit, recorder, transcriber, speaker, output=print) -> int`
  - `build_parser() -> argparse.ArgumentParser`
  - `main(argv: list[str] | None = None) -> int`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_voice_program.py`. Ensure `numpy` is imported and add `from program.voice_chat import build_parser, conversation_loop, is_quit_command, spoken_answer`.

```python
class FakeRecorder:
    def __init__(self, recordings):
        self._recordings = iter(recordings)

    def record(self):
        return next(self._recordings)


class FakeTranscriber:
    def __init__(self, transcripts):
        self._transcripts = iter(transcripts)

    def transcribe(self, audio):
        return next(self._transcripts)


class FakeSpeaker:
    def __init__(self):
        self.spoken = []

    def speak(self, text):
        self.spoken.append(text)


class FakeBit:
    def __init__(self):
        self.questions = []

    def predict(self, text):
        self.questions.append(text)
        return {
            "label": "yes",
            "confidence": 0.9,
            "probabilities": {"yes": 0.9, "no": 0.05, "idk": 0.05},
        }


AUDIO = np.ones(1600, dtype=np.float32)


class VoiceChatTests(unittest.TestCase):
    def test_spoken_answer_maps_known_and_unknown_labels(self):
        self.assertEqual(spoken_answer("yes"), "Yes.")
        self.assertEqual(spoken_answer(" NO "), "No.")
        self.assertEqual(spoken_answer("idk"), "I don't know.")
        self.assertEqual(spoken_answer("maybe"), "I don't know.")

    def test_is_quit_command_accepts_spoken_stop_words(self):
        for text in ("quit", "Exit.", " stop ", "STOP!"):
            self.assertTrue(is_quit_command(text))
        for text in ("stop the car", "stopping", ""):
            self.assertFalse(is_quit_command(text))

    def test_conversation_loop_prints_and_speaks_answer(self):
        bit = FakeBit()
        speaker = FakeSpeaker()
        lines = []
        result = conversation_loop(
            bit,
            FakeRecorder([AUDIO, AUDIO]),
            FakeTranscriber(["Paris is in France. Is Paris in France?", "quit"]),
            speaker,
            output=lines.append,
        )
        self.assertEqual(result, 0)
        self.assertIn("You: Paris is in France. Is Paris in France?", lines)
        self.assertIn("Bit: yes (0.900)", lines)
        self.assertEqual(speaker.spoken, ["Yes."])
        self.assertEqual(
            bit.questions, ["Paris is in France. Is Paris in France?"]
        )
        self.assertEqual(lines[-1], "Goodbye.")

    def test_conversation_loop_retries_on_empty_transcript(self):
        speaker = FakeSpeaker()
        lines = []
        conversation_loop(
            FakeBit(),
            FakeRecorder([AUDIO, AUDIO]),
            FakeTranscriber(["", "quit"]),
            speaker,
            output=lines.append,
        )
        self.assertIn("I couldn't understand that. Try again.", lines)
        self.assertEqual(speaker.spoken, [])

    def test_conversation_loop_retries_on_empty_audio(self):
        lines = []
        conversation_loop(
            FakeBit(),
            FakeRecorder([np.zeros(0, dtype=np.float32), AUDIO]),
            FakeTranscriber(["quit"]),
            FakeSpeaker(),
            output=lines.append,
        )
        self.assertIn("I didn't hear anything. Try again.", lines)

    def test_conversation_loop_handles_keyboard_interrupt(self):
        class InterruptingRecorder:
            def record(self):
                raise KeyboardInterrupt

        lines = []
        result = conversation_loop(
            FakeBit(),
            InterruptingRecorder(),
            FakeTranscriber([]),
            FakeSpeaker(),
            output=lines.append,
        )
        self.assertEqual(result, 0)
        self.assertEqual(lines[-1], "Goodbye.")

    def test_parser_defaults(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.asr_model, "small.en")
        self.assertEqual(args.tts_voice, "en_US-lessac-medium")
        self.assertEqual(args.device, "auto")
        self.assertAlmostEqual(args.silence_seconds, 0.8)
        self.assertAlmostEqual(args.max_seconds, 15.0)
        self.assertIsNone(args.threshold)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m unittest tests.test_voice_program.VoiceChatTests -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'program.voice_chat'`.

- [ ] **Step 3: Implement `program/voice_chat.py`**

Import `program.config` before `chat.model` so `HF_HOME` is set before `transformers` loads. Use these exact rules:

- `SPOKEN_ANSWERS = {"yes": "Yes.", "no": "No.", "idk": "I don't know."}`.
- `spoken_answer` lowercases and strips the label, returning the mapping value or `"I don't know."`.
- `is_quit_command` strips whitespace and `".!? "` from both ends, lowercases, and returns whether the result is one of `{"quit", "exit", "stop"}`.
- `conversation_loop` prints `Listening... speak your context and question.` before each recording. It then:
  - retries with `I didn't hear anything. Try again.` for `None` or zero-length audio;
  - transcribes, strips, and retries with `I couldn't understand that. Try again.` for an empty transcript;
  - prints `You: <transcript>`;
  - exits with `Goodbye.` for a quit command;
  - prints `Bit: <label> (<confidence:.3f>)`, calls `speaker.speak(spoken_answer(label))`, then prompts again;
  - catches `KeyboardInterrupt`, prints a blank line and `Goodbye.`, and returns `0`.
- `build_parser` exposes all options from the spec: `--model`, `--model-dir`, `--list-models`, `--list-audio-devices`, `--asr-model` (default `small.en`), `--tts-voice` (default `en_US-lessac-medium`), `--device` (default `auto`), `--input-device` (`int`), `--output-device` (`int`), `--threshold` (`float`, default `None`), `--silence-seconds` (default `0.8`), and `--max-seconds` (default `15.0`).
- `main`:
  - handles `--list-models` with `list_available_models()`, printing `Available models:` and `Newest: <path>`;
  - handles `--list-audio-devices` with `format_audio_devices()`, converting `ImportError` to `missing_dependency_error`;
  - resolves the checkpoint with `resolve_model_dir`, builds `ChatConfig(model_dir=..., device=args.device)`, and loads `Bit`, printing `Loaded Bit model from ...`;
  - builds `RecordSettings(threshold=..., silence_seconds=..., max_seconds=...)`, `Recorder(..., device=args.input_device)`, `Transcriber(model_name=args.asr_model, device=args.device, download_root=WHISPER_ROOT)`, and `Speaker(voice=args.tts_voice, download_root=PIPER_ROOT, output_device=args.output_device)`;
  - prints `Loading speech models (first run downloads into <MODELS_ROOT>)...`, calls `ensure_loaded()` on both speech components, and returns `1` with a clear message if they raise `RuntimeError`;
  - calls `recorder.calibrate()` once before the loop so the first spoken word is not clipped, returning `1` with a clear message if it raises `AudioError`;
  - prints `Bit voice is ready.` and returns `conversation_loop(...)`;
  - catches `(AudioError, RuntimeError)` around the loop, prints the message to stderr, and returns `1`;
  - returns `1` for missing checkpoints and unknown `--model` values.

Add `if __name__ == "__main__": raise SystemExit(main())`.

Update `README.md` with a `## Voice Chat` section covering:

- install with `pip install -e '.[voice]'`;
- run with `python -m program.voice_chat` or `bit-voice`;
- speak context and question together, example `Paris is the capital of France. Is Paris in France?`;
- printed transcript and `Bit: yes (0.956)` followed by spoken “Yes.”;
- quit words `quit`, `exit`, `stop`;
- defaults to the newest `models/` checkpoint, overridden by `--model` / `--model-dir`;
- `--list-models`, `--list-audio-devices`, `--input-device`, `--output-device`, `--asr-model`, `--tts-voice`, `--silence-seconds`, `--max-seconds`;
- first run downloads into `program/models/` and nothing is sent to a cloud service.

Add the `program/` files to the README project layout.

- [ ] **Step 4: Run the full suite to verify it passes**

Run: `.venv/bin/python -m unittest discover -s tests -v`

Expected: PASS for the existing chat/core tests and all voice tests.

- [ ] **Step 5: Verify the CLI surfaces**

Run: `.venv/bin/python -m program.voice_chat --list-models`

Expected: model list including `Newest: models/bit-20261024_232820` (or another newest timestamp).

Run: `.venv/bin/python -m program.voice_chat --list-audio-devices`

Expected: one line per device, with at least one line marked `default input`.

Run: `.venv/bin/bit-voice --list-models`

Expected: same model list through the console script.

- [ ] **Step 6: Manual voice verification**

Run: `.venv/bin/python -m program.voice_chat --model-dir models/bit-20261024_232820`

Speak: `Paris is the capital of France. Is Paris in France?`

Expected: first run downloads Whisper and Piper assets into `program/models/`; the terminal shows `You: ...`, `Bit: yes (0.956)` (confidence may vary), and the machine says “Yes.”

Say `quit` to exit cleanly.

- [ ] **Step 7: Commit**

```bash
git add program/voice_chat.py tests/test_voice_program.py README.md
git commit -m "feat: add voice chat CLI"
```

---

## Self-Review Notes

- Spec coverage: local-only stack, local asset roots, HF cache, checkpoint resolution, silence-aware capture, 15 s cap, transcript/answer output, spoken labels, quit commands, error handling, CLI options, and fake-component tests all have tasks.
- Interfaces: `RecordSettings.block_size`, `record_blocks`, `Recorder.record`, `play`, `Transcriber.transcribe`, `Speaker.speak`, and `conversation_loop` are used consistently across tasks.
- Review Focus tests: silent input (Task 2 empty capture + Task 4 empty audio), empty transcript (Task 4), max duration (Task 2), missing checkpoint (Task 1), unexpected label (Task 4).
- Scope: no wake word, streaming, GUI, or extra checkpoint loader.
