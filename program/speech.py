from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from program.audio import play
from program.config import PIPER_ROOT, VOICE_INSTALL_HINT, WHISPER_ROOT

DEFAULT_WHISPER_MODEL = "small.en"
DEFAULT_TTS_VOICE = "en_US-lessac-medium"

ModelFactory = Callable[..., object]


def missing_dependency_error(exc: ImportError) -> RuntimeError:
    return RuntimeError(f"{VOICE_INSTALL_HINT}\nOriginal error: {exc}")


class Transcriber:
    def __init__(
        self,
        model_name: str = DEFAULT_WHISPER_MODEL,
        device: str = "auto",
        download_root: str | Path = WHISPER_ROOT,
        compute_type: str = "default",
        model_factory: ModelFactory | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.download_root = Path(download_root)
        self.compute_type = compute_type
        self.model_factory = model_factory
        self._model: object | None = None

    def ensure_loaded(self) -> None:
        if self._model is not None:
            return

        factory = self.model_factory
        if factory is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise missing_dependency_error(exc) from exc
            factory = WhisperModel

        try:
            self._model = factory(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                download_root=str(self.download_root),
            )
        except Exception as exc:
            if self.device == "cpu":
                raise RuntimeError(
                    f"Could not load Whisper model '{self.model_name}': {exc}"
                ) from exc

            print(
                f"Could not load Whisper model '{self.model_name}' on "
                f"{self.device}: {exc}. Falling back to CPU."
            )
            try:
                self._model = factory(
                    self.model_name,
                    device="cpu",
                    compute_type="int8",
                    download_root=str(self.download_root),
                )
                self.device = "cpu"
                self.compute_type = "int8"
            except Exception as retry_exc:
                raise RuntimeError(
                    f"Could not load Whisper model '{self.model_name}' on CPU: "
                    f"{retry_exc}"
                ) from retry_exc

    def transcribe(self, audio: np.ndarray) -> str:
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return ""

        self.ensure_loaded()
        segments, _info = self._model.transcribe(  # type: ignore[attr-defined]
            samples,
            language="en",
            vad_filter=True,
            condition_on_previous_text=False,
        )

        parts = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                parts.append(text)
        return " ".join(parts)


class Speaker:
    def __init__(
        self,
        voice: str = DEFAULT_TTS_VOICE,
        download_root: str | Path = PIPER_ROOT,
        use_cuda: bool = False,
        output_device: int | None = None,
    ) -> None:
        self.voice = voice
        self.download_root = Path(download_root)
        self.use_cuda = use_cuda
        self.output_device = output_device
        self._voice: object | None = None

    def ensure_loaded(self) -> None:
        if self._voice is not None:
            return

        try:
            from piper.download_voices import download_voice
            from piper.voice import PiperVoice
        except ImportError as exc:
            raise missing_dependency_error(exc) from exc

        self.download_root.mkdir(parents=True, exist_ok=True)

        model_path = self.download_root / f"{self.voice}.onnx"
        config_path = self.download_root / f"{self.voice}.onnx.json"

        if not model_path.exists() or not config_path.exists():
            try:
                download_voice(self.voice, self.download_root)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not download Piper voice '{self.voice}' into "
                    f"{self.download_root}: {exc}"
                ) from exc

        try:
            self._voice = PiperVoice.load(
                model_path,
                config_path=config_path,
                use_cuda=self.use_cuda,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not load Piper voice '{self.voice}' from {model_path}: {exc}"
            ) from exc

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        self.ensure_loaded()
        chunks = list(self._voice.synthesize(text))  # type: ignore[attr-defined]
        if not chunks:
            return np.empty(0, dtype=np.float32), 22050

        audio = np.concatenate(
            [
                np.asarray(chunk.audio_float_array, dtype=np.float32).reshape(-1)
                for chunk in chunks
            ]
        )
        return audio, chunks[0].sample_rate

    def speak(self, text: str) -> None:
        audio, sample_rate = self.synthesize(text)
        play(audio, sample_rate, device=self.output_device)
