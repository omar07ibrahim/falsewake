"""Chunk-invariant streaming log-mel frontend for experiment 002.

The experiment keeps the public experiment-000 frontend geometry and numeric
definition.  Unlike :func:`falsewake.features.log_mel_spectrogram`, this module
does not right-pad or cap an utterance: it emits a frame only after all 400
samples for that frame have arrived.  ``end_utterance`` discards the incomplete
overlap/tail and resets the frame origin for the next utterance.

For a 16,000-sample clip the frame starts are 0, 160, ..., 15,520.  The 98
frames therefore cover samples ``[0, 15_920)``; the final 80 samples cannot
affect a frame before the utterance boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from falsewake.features import (
    DEFAULT_FRONTEND,
    FloatArray,
    FrontendConfig,
    mel_filterbank,
)


@dataclass(frozen=True, slots=True)
class StreamingFrontendState:
    """Read-only snapshot of the streaming frame alignment.

    ``buffered_samples`` begins at ``next_frame_start``.  Its length is the
    explicit phase within the next 400-sample analysis frame and is always
    smaller than one complete window after a successful ``push``.
    """

    total_samples: int
    frames_emitted: int
    next_frame_start: int
    frame_phase_samples: int
    buffered_samples: FloatArray


class StreamingLogMelFrontend:
    """Incrementally convert normalized mono float32 PCM into log-mel frames."""

    def __init__(self, config: FrontendConfig = DEFAULT_FRONTEND) -> None:
        if config.hop_samples > config.window_samples:
            raise ValueError(
                "streaming frontend requires hop_samples no greater than "
                "window_samples"
            )
        self._config = config
        self._periodic_hann = np.hanning(config.window_samples + 1)[:-1].astype(
            np.float32
        )
        self._mel_filters = mel_filterbank(config)
        self._buffer = np.empty(0, dtype=np.float32)
        self._total_samples = 0
        self._frames_emitted = 0

    @property
    def config(self) -> FrontendConfig:
        """Return the fixed frontend definition used by this stream."""

        return self._config

    @property
    def state(self) -> StreamingFrontendState:
        """Return a detached, non-writeable snapshot of the stream state."""

        buffered = self._buffer.copy()
        buffered.flags.writeable = False
        next_frame_start = self._frames_emitted * self._config.hop_samples
        return StreamingFrontendState(
            total_samples=self._total_samples,
            frames_emitted=self._frames_emitted,
            next_frame_start=next_frame_start,
            frame_phase_samples=int(buffered.size),
            buffered_samples=buffered,
        )

    def push(self, samples: FloatArray) -> FloatArray:
        """Consume one arbitrary-size PCM chunk and return newly complete frames.

        Validation happens before state mutation.  Empty chunks are valid and
        return a ``(0, mel_bins)`` array.
        """

        self._validate_chunk(samples)
        if samples.size == 0:
            return self._empty_frames()

        self._total_samples += int(samples.size)
        self._buffer = np.concatenate((self._buffer, samples))
        rows: list[FloatArray] = []
        while self._buffer.size >= self._config.window_samples:
            frame = self._buffer[: self._config.window_samples]
            rows.append(self._log_mel_frame(frame))
            self._frames_emitted += 1
            self._buffer = self._buffer[self._config.hop_samples :]

        # Do not retain an arbitrarily large caller-owned allocation through a
        # small NumPy view, and keep the snapshot layout deterministic.
        self._buffer = np.ascontiguousarray(self._buffer, dtype=np.float32).copy()
        if not rows:
            return self._empty_frames()
        return cast(FloatArray, np.stack(rows).astype(np.float32, copy=False))

    def end_utterance(self) -> None:
        """Discard an incomplete final frame and reset alignment to sample zero."""

        self._buffer = np.empty(0, dtype=np.float32)
        self._total_samples = 0
        self._frames_emitted = 0

    def _empty_frames(self) -> FloatArray:
        return np.empty((0, self._config.mel_bins), dtype=np.float32)

    def _log_mel_frame(self, frame: FloatArray) -> FloatArray:
        windowed = frame * self._periodic_hann
        spectrum = np.fft.rfft(windowed, n=self._config.fft_samples)
        power = (np.abs(spectrum) ** 2 / self._config.fft_samples).astype(np.float32)
        mel_power = power[np.newaxis, :] @ self._mel_filters.T
        result = np.log(np.maximum(mel_power, self._config.log_floor)).astype(
            np.float32
        )[0]
        if not np.all(np.isfinite(result)):
            raise ValueError("log-mel calculation produced non-finite values")
        return cast(FloatArray, result)

    @staticmethod
    def _validate_chunk(samples: FloatArray) -> None:
        if samples.dtype != np.dtype(np.float32):
            raise ValueError("samples must use normalized float32 PCM")
        if samples.ndim != 1:
            raise ValueError("samples must be one-dimensional mono audio")
        if not np.all(np.isfinite(samples)):
            raise ValueError("samples contain non-finite values")
        if samples.size and np.max(np.abs(samples.astype(np.float64))) > 1.0:
            raise ValueError("samples must stay in the normalized [-1, 1] range")


def complete_log_mel_frames(
    waveform: FloatArray, config: FrontendConfig = DEFAULT_FRONTEND
) -> FloatArray:
    """Return all complete streaming frames in one waveform, without padding."""

    frontend = StreamingLogMelFrontend(config)
    frames = frontend.push(waveform)
    frontend.end_utterance()
    return frames
