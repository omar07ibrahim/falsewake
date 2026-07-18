"""Small, dependency-light audio frontend for experiment 000."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class FrontendConfig:
    sample_rate: int = 16_000
    clip_samples: int = 16_000
    window_samples: int = 400
    hop_samples: int = 160
    fft_samples: int = 512
    mel_bins: int = 40
    min_hz: float = 20.0
    max_hz: float = 7_600.0
    log_floor: float = 1e-10

    def __post_init__(self) -> None:
        integer_fields = (
            self.sample_rate,
            self.clip_samples,
            self.window_samples,
            self.hop_samples,
            self.fft_samples,
            self.mel_bins,
        )
        if any(value <= 0 for value in integer_fields):
            raise ValueError("frontend integer settings must be positive")
        if self.clip_samples < self.window_samples:
            raise ValueError("clip_samples must be at least window_samples")
        if self.fft_samples < self.window_samples:
            raise ValueError("fft_samples must be at least window_samples")
        if self.mel_bins < 2:
            raise ValueError("mel_bins must be at least two")
        if not 0 <= self.min_hz < self.max_hz <= self.sample_rate / 2:
            raise ValueError("mel frequency bounds must fit below Nyquist")
        if not np.isfinite(self.log_floor) or self.log_floor <= 0:
            raise ValueError("log_floor must be finite and positive")

    @property
    def frame_count(self) -> int:
        return 1 + (self.clip_samples - self.window_samples) // self.hop_samples

    @property
    def summary_size(self) -> int:
        return self.mel_bins * 2


DEFAULT_FRONTEND = FrontendConfig()


def pcm16le_to_float32(contents: bytes) -> FloatArray:
    """Decode signed little-endian PCM16 with the fixed 1/32768 scale."""

    if len(contents) % 2:
        raise ValueError("PCM16 byte length must be even")
    integers = np.frombuffer(contents, dtype="<i2")
    return (integers.astype(np.float32) / np.float32(32_768.0)).astype(np.float32)


def mel_edge_frequencies(config: FrontendConfig = DEFAULT_FRONTEND) -> FloatArray:
    """Return the lower, center, and upper frequencies for all mel filters."""

    bounds = np.asarray([config.min_hz, config.max_hz], dtype=np.float64)
    mel_bounds = 2_595.0 * np.log10(1.0 + bounds / 700.0)
    mel_edges = np.linspace(
        mel_bounds[0], mel_bounds[1], config.mel_bins + 2, dtype=np.float64
    )
    frequency_edges = 700.0 * (np.power(10.0, mel_edges / 2_595.0) - 1.0)
    frequency_edges[0] = config.min_hz
    frequency_edges[-1] = config.max_hz
    result = frequency_edges.astype(np.float32)
    if not np.all(np.isfinite(result)) or not np.all(np.diff(result) > 0):
        raise ValueError("mel edges collapse at float32 precision")
    return result


def mel_filterbank(config: FrontendConfig = DEFAULT_FRONTEND) -> FloatArray:
    """Build area-normalized, HTK-spaced triangular mel filters."""

    edges = mel_edge_frequencies(config)
    frequencies = np.fft.rfftfreq(
        config.fft_samples, d=1.0 / config.sample_rate
    ).astype(np.float32)
    filters = np.empty((config.mel_bins, config.fft_samples // 2 + 1), dtype=np.float32)
    for index in range(config.mel_bins):
        lower, center, upper = edges[index : index + 3]
        rising = (frequencies - lower) / (center - lower)
        falling = (upper - frequencies) / (upper - center)
        triangle = np.maximum(0.0, np.minimum(rising, falling))
        filters[index] = triangle * (2.0 / (upper - lower))
    if not np.all(np.isfinite(filters)) or np.any(np.max(filters, axis=1) <= 0):
        raise ValueError("fft_samples is too small for the requested mel filterbank")
    return filters


def frame_waveform(
    waveform: FloatArray, config: FrontendConfig = DEFAULT_FRONTEND
) -> FloatArray:
    """Right-pad one mono clip and return overlapping analysis frames."""

    if waveform.dtype != np.dtype(np.float32):
        raise ValueError("waveform must use normalized float32 PCM samples")
    if waveform.ndim != 1:
        raise ValueError("waveform must be one-dimensional mono audio")
    if waveform.size > config.clip_samples:
        raise ValueError(
            f"waveform has {waveform.size} samples; limit is {config.clip_samples}"
        )
    if not np.all(np.isfinite(waveform)):
        raise ValueError("waveform contains non-finite samples")
    if waveform.size and np.max(np.abs(waveform.astype(np.float64))) > 1.0:
        raise ValueError("waveform samples must stay in the normalized [-1, 1] range")
    padded = np.zeros(config.clip_samples, dtype=np.float32)
    padded[: waveform.size] = waveform
    frames = sliding_window_view(padded, config.window_samples)[:: config.hop_samples]
    return np.asarray(frames[: config.frame_count], dtype=np.float32)


def log_mel_spectrogram(
    waveform: FloatArray, config: FrontendConfig = DEFAULT_FRONTEND
) -> FloatArray:
    """Compute an uncentered 25 ms / 10 ms log-mel representation."""

    frames = frame_waveform(waveform, config)
    periodic_hann = np.hanning(config.window_samples + 1)[:-1].astype(np.float32)
    spectrum = np.fft.rfft(frames * periodic_hann, n=config.fft_samples, axis=1)
    power = (np.abs(spectrum) ** 2 / config.fft_samples).astype(np.float32)
    mel_power = power @ mel_filterbank(config).T
    result = cast(
        FloatArray,
        np.log(np.maximum(mel_power, config.log_floor)).astype(np.float32),
    )
    if not np.all(np.isfinite(result)):
        raise ValueError("log-mel calculation produced non-finite values")
    return result


def summarize_log_mel(log_mel: FloatArray) -> FloatArray:
    """Concatenate per-band mean and population standard deviation."""

    if log_mel.ndim != 2 or log_mel.shape[0] == 0:
        raise ValueError("log_mel must contain at least one two-dimensional frame")
    if not np.all(np.isfinite(log_mel)):
        raise ValueError("log_mel contains non-finite values")
    means = np.mean(log_mel, axis=0, dtype=np.float64)
    deviations = np.std(log_mel, axis=0, dtype=np.float64)
    return np.concatenate((means, deviations)).astype(np.float32)


def extract_clip_features(
    waveform: FloatArray, config: FrontendConfig = DEFAULT_FRONTEND
) -> FloatArray:
    """Return the deliberately weak 80-value baseline representation."""

    return summarize_log_mel(log_mel_spectrogram(waveform, config))
