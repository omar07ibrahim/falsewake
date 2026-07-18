"""Deterministic log-mel normalization for Experiment 002."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class NormalizationStats:
    """Owned, immutable float32 normalization parameters."""

    means: FloatArray
    standard_deviations: FloatArray

    def __post_init__(self) -> None:
        _validate_stats_array("means", self.means)
        _validate_stats_array("standard_deviations", self.standard_deviations)
        if self.means.shape != self.standard_deviations.shape:
            raise ValueError("means and standard_deviations must have the same shape")
        if np.any(self.standard_deviations <= np.float32(0.0)):
            raise ValueError("standard_deviations must be strictly positive")

        means = np.array(self.means, dtype=np.float32, order="C", copy=True)
        standard_deviations = np.array(
            self.standard_deviations, dtype=np.float32, order="C", copy=True
        )
        means.flags.writeable = False
        standard_deviations.flags.writeable = False
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "standard_deviations", standard_deviations)

    @property
    def mel_bins(self) -> int:
        """Return the number of independently normalized mel bands."""

        return int(self.means.size)


class NormalizationAccumulator:
    """Accumulate float32 log-mel values in exact frame-major scalar order."""

    def __init__(self, mel_bins: int) -> None:
        if not isinstance(mel_bins, int) or isinstance(mel_bins, bool):
            raise TypeError("mel_bins must be an integer")
        if mel_bins < 1:
            raise ValueError("mel_bins must be positive")
        self._mel_bins = mel_bins
        self._frame_count = 0
        self._sums = np.zeros(mel_bins, dtype=np.float64)
        self._sumsq = np.zeros(mel_bins, dtype=np.float64)

    @property
    def mel_bins(self) -> int:
        """Return the fixed number of mel bands."""

        return self._mel_bins

    @property
    def frame_count(self) -> int:
        """Return the number of frames accumulated so far."""

        return self._frame_count

    def update(self, log_mel: FloatArray) -> None:
        """Accumulate one or more frames without changing scalar addition order."""

        _validate_log_mel(log_mel, self._mel_bins)
        for frame in log_mel:
            for mel_index in range(self._mel_bins):
                value = float(frame[mel_index])
                self._sums[mel_index] += value
                self._sumsq[mel_index] += value * value
            self._frame_count += 1

    def finalize(self, *, expected_frames: int) -> NormalizationStats:
        """Produce the registered float32 mean and population deviation values."""

        if not isinstance(expected_frames, int) or isinstance(expected_frames, bool):
            raise TypeError("expected_frames must be an integer")
        if expected_frames < 1:
            raise ValueError("expected_frames must be positive")
        if self._frame_count != expected_frames:
            raise ValueError(
                f"expected {expected_frames} frames, accumulated {self._frame_count}"
            )

        count = float(self._frame_count)
        means64 = self._sums / count
        variances64 = np.maximum(
            0.0,
            self._sumsq / count - means64 * means64,
        )
        deviations64 = np.sqrt(variances64)
        means = means64.astype(np.float32)
        deviations = deviations64.astype(np.float32)
        if not np.all(np.isfinite(means)) or not np.all(np.isfinite(deviations)):
            raise ValueError("normalization statistics are non-finite")
        if np.any(deviations <= np.float32(0.0)):
            raise ValueError("normalization standard deviation is zero")
        return NormalizationStats(means=means, standard_deviations=deviations)


def serialize_stats(stats: NormalizationStats) -> bytes:
    """Serialize means then deviations as exact little-endian binary32 values."""

    if not isinstance(stats, NormalizationStats):
        raise TypeError("stats must be NormalizationStats")
    return b"".join(
        (
            stats.means.astype("<f4", copy=False).tobytes(order="C"),
            stats.standard_deviations.astype("<f4", copy=False).tobytes(order="C"),
        )
    )


def load_stats(contents: bytes, *, mel_bins: int = 40) -> NormalizationStats:
    """Load an exact-size little-endian normalization artifact."""

    if not isinstance(contents, bytes):
        raise TypeError("contents must be bytes")
    if not isinstance(mel_bins, int) or isinstance(mel_bins, bool):
        raise TypeError("mel_bins must be an integer")
    if mel_bins < 1:
        raise ValueError("mel_bins must be positive")
    expected_bytes = 2 * mel_bins * np.dtype("<f4").itemsize
    if len(contents) != expected_bytes:
        raise ValueError(
            f"normalization artifact must contain exactly {expected_bytes} bytes"
        )
    values = np.frombuffer(contents, dtype="<f4")
    means = values[:mel_bins].astype(np.float32, copy=True)
    deviations = values[mel_bins:].astype(np.float32, copy=True)
    return NormalizationStats(means=means, standard_deviations=deviations)


def normalize_log_mel(log_mel: FloatArray, stats: NormalizationStats) -> FloatArray:
    """Apply separate float32 subtraction and division operations."""

    if not isinstance(stats, NormalizationStats):
        raise TypeError("stats must be NormalizationStats")
    _validate_log_mel(log_mel, stats.mel_bins)
    centered = np.subtract(log_mel, stats.means, dtype=np.float32)
    normalized = np.divide(
        centered,
        stats.standard_deviations,
        dtype=np.float32,
    )
    if not np.all(np.isfinite(normalized)):
        raise ValueError("normalization produced non-finite values")
    return np.ascontiguousarray(normalized, dtype=np.float32)


def _validate_stats_array(name: str, values: FloatArray) -> None:
    if not isinstance(values, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if values.dtype != np.dtype(np.float32):
        raise ValueError(f"{name} must use float32")
    if values.ndim != 1 or values.size < 1:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")


def _validate_log_mel(log_mel: FloatArray, mel_bins: int) -> None:
    if not isinstance(log_mel, np.ndarray):
        raise TypeError("log_mel must be a NumPy array")
    if log_mel.dtype != np.dtype(np.float32):
        raise ValueError("log_mel must use float32")
    if log_mel.ndim != 2 or log_mel.shape[0] < 1:
        raise ValueError("log_mel must contain at least one two-dimensional frame")
    if log_mel.shape[1] != mel_bins:
        raise ValueError(f"log_mel must contain exactly {mel_bins} mel bands")
    if not np.all(np.isfinite(log_mel)):
        raise ValueError("log_mel must contain only finite values")
