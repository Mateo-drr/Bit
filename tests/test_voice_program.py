import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import program
from program import config, speech
from program.audio import RecordSettings, record_blocks
from program.speech import Speaker, Transcriber
from program.voice_chat import (
    build_parser,
    conversation_loop,
    is_quit_command,
    spoken_answer,
)


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


if __name__ == "__main__":
    unittest.main()
