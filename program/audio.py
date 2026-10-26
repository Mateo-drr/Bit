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


def rms(block: np.ndarray) -> float:
    data = np.asarray(block, dtype=np.float64).reshape(-1)
    if data.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(data))))


def _copy_block(block: np.ndarray) -> np.ndarray:
    """Copy a block so reused stream buffers cannot alias recorded audio."""
    return np.array(block, dtype=np.float32).reshape(-1)


def read_blocks(stream: object, block_size: int) -> Iterator[np.ndarray]:
    while True:
        data, _overflowed = stream.read(block_size)  # type: ignore[attr-defined]
        yield data


def calibration_block_count(settings: RecordSettings) -> int:
    if settings.threshold is not None:
        return 0
    return max(1, round(settings.calibration_seconds / settings.block_seconds))


def calibrate_threshold(
    blocks: Iterable[np.ndarray], settings: RecordSettings
) -> float:
    if settings.threshold is not None:
        return settings.threshold

    noise_rms = [rms(block) for block in blocks]
    if not noise_rms:
        return MIN_THRESHOLD
    return max(MIN_THRESHOLD, (sum(noise_rms) / len(noise_rms)) * NOISE_MULTIPLIER)


def record_blocks(
    blocks: Iterable[np.ndarray], settings: RecordSettings
) -> np.ndarray:
    iterator = iter(blocks)

    calibration: list[np.ndarray] = []
    for _ in range(calibration_block_count(settings)):
        try:
            calibration.append(next(iterator))
        except StopIteration:
            break

    threshold = calibrate_threshold(calibration, settings)
    if settings.threshold is not None:
        threshold = settings.threshold

    preroll: deque[np.ndarray] = deque(maxlen=settings.preroll_blocks)
    if settings.preroll_blocks > 0:
        for block in calibration[-settings.preroll_blocks :]:
            preroll.append(_copy_block(block))

    recorded: list[np.ndarray] = []
    consecutive = 0
    waited = 0
    max_wait_blocks = round(settings.max_seconds / settings.block_seconds)
    started = False

    for block in iterator:
        waited += 1
        preroll.append(_copy_block(block))

        if rms(block) >= threshold:
            consecutive += 1
        else:
            consecutive = 0

        if consecutive >= settings.start_blocks:
            recorded = [_copy_block(item) for item in preroll]
            started = True
            break

        if waited >= max_wait_blocks:
            break

    if not started:
        return np.empty(0, dtype=np.float32)

    silence_blocks = 0
    for block in iterator:
        recorded.append(_copy_block(block))

        if rms(block) >= threshold:
            silence_blocks = 0
        else:
            silence_blocks += 1
            if silence_blocks * settings.block_seconds >= settings.silence_seconds:
                break

        if len(recorded) * settings.block_seconds >= settings.max_seconds:
            break

    return np.concatenate(recorded) if recorded else np.empty(0, dtype=np.float32)


class Recorder:
    def __init__(
        self, settings: RecordSettings | None = None, device: int | None = None
    ) -> None:
        self.settings = settings or RecordSettings()
        self.device = device
        self._threshold: float | None = None

    @property
    def sample_rate(self) -> int:
        return self.settings.sample_rate

    def calibrate(self) -> None:
        if self._threshold is not None:
            return

        settings = self.settings
        if settings.threshold is not None:
            self._threshold = settings.threshold
            return

        import sounddevice as sd

        count = calibration_block_count(settings)
        try:
            with sd.InputStream(
                samplerate=settings.sample_rate,
                blocksize=settings.block_size,
                channels=1,
                dtype="float32",
                device=self.device,
            ) as stream:
                stream_blocks = read_blocks(stream, settings.block_size)
                calibration = [next(stream_blocks) for _ in range(count)]
        except sd.PortAudioError as exc:
            raise AudioError(
                f"Could not open microphone device {self._device_name()}: {exc}"
            ) from exc

        self._threshold = calibrate_threshold(calibration, settings)

    def record(self) -> np.ndarray:
        if self._threshold is None:
            self.calibrate()

        import sounddevice as sd

        settings = replace(self.settings, threshold=self._threshold)
        try:
            with sd.InputStream(
                samplerate=settings.sample_rate,
                blocksize=settings.block_size,
                channels=1,
                dtype="float32",
                device=self.device,
            ) as stream:
                return record_blocks(
                    read_blocks(stream, settings.block_size), settings
                )
        except sd.PortAudioError as exc:
            raise AudioError(
                f"Could not open microphone device {self._device_name()}: {exc}"
            ) from exc

    def _device_name(self) -> str:
        return "default" if self.device is None else str(self.device)


def play(audio: np.ndarray, sample_rate: int, device: int | None = None) -> None:
    samples = np.asarray(audio, dtype=np.float32).flatten()
    if samples.size == 0:
        return

    import sounddevice as sd

    sd.play(samples, samplerate=sample_rate, device=device)
    sd.wait()


def format_audio_devices() -> str:
    import sounddevice as sd

    default_input, default_output = sd.default.device
    lines = []
    for index, device in enumerate(sd.query_devices()):
        line = (
            f"{index}: {device['name']} "
            f"in={device['max_input_channels']} "
            f"out={device['max_output_channels']}"
        )
        if index == default_input:
            line += " (default input)"
        if index == default_output:
            line += " (default output)"
        lines.append(line)
    return "\n".join(lines)
