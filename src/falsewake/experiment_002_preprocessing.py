"""Exact waveform augmentation and feature preparation for Experiment 002."""

from __future__ import annotations

import math
from typing import Final

import numpy as np

from falsewake.experiment_002_data import (
    TRAIN_EPOCH_COUNT,
    TRAINING_SEEDS,
    WINDOW_SAMPLES,
    CommandExample,
    Experiment002Corpus,
    WindowExample,
    _require_registered_training_corpus,
    select_command_noise_window,
)
from falsewake.experiment_002_frontend import complete_log_mel_frames
from falsewake.experiment_002_normalization import (
    NormalizationStats,
    normalize_log_mel,
)
from falsewake.experiment_002_normalization_artifact import (
    VerifiedNormalization,
    verify_registered_normalization,
)
from falsewake.experiment_002_pcm_cache import PCMCacheSplitView
from falsewake.experiment_002_rng import bernoulli, uniform_integer, uniform_real
from falsewake.features import FloatArray

TIME_FRAMES: Final = 98
MEL_BINS: Final = 40
SHIFT_DRAW_BOUND: Final = 3_201
SHIFT_OFFSET: Final = 1_600
TIME_MASK_WIDTH_BOUND: Final = 11
MEL_MASK_WIDTH_BOUND: Final = 5
NOISE_PROBABILITY: Final = 0.8


class Experiment002PreprocessingError(ValueError):
    """An input or intermediate violates the registered numeric pipeline."""


class Experiment002TrainingPreprocessor:
    """Exact-corpus, training-cache-bound augmentation and feature pipeline."""

    __slots__ = (
        "_cache",
        "_corpus",
        "_stats",
        "_train_backgrounds",
        "_train_commands",
    )

    def __init__(
        self,
        corpus: Experiment002Corpus,
        cache: PCMCacheSplitView,
        normalization: VerifiedNormalization,
    ) -> None:
        _require_registered_training_corpus(corpus)
        if type(cache) is not PCMCacheSplitView:
            raise TypeError("cache must be a PCMCacheSplitView")
        if cache.split != "train":
            raise Experiment002PreprocessingError(
                "training preprocessing requires the training cache view"
            )
        if type(normalization) is not VerifiedNormalization:
            raise TypeError("normalization must be a VerifiedNormalization")
        verify_registered_normalization(normalization)
        self._corpus = corpus
        self._cache = cache
        self._stats = normalization.stats
        self._train_commands = corpus._train_command_sources
        self._train_backgrounds = frozenset(corpus.train_backgrounds)

    def command_waveform(
        self,
        example: CommandExample,
        *,
        seed: int,
        epoch: int,
    ) -> FloatArray:
        """Read and augment one identity-bound training command."""

        _require_training_context(seed, epoch)
        if not isinstance(example, CommandExample):
            raise TypeError("example must be a CommandExample")
        if example.source not in self._train_commands:
            raise Experiment002PreprocessingError(
                "command augmentation requires a corpus training source"
            )
        payload = self._cache.read_command_pcm16le(example.source)
        waveform = _decode_padded_command(
            payload,
            sample_count=example.source.sample_count,
        )

        shift = (
            uniform_integer(
                seed=seed,
                epoch=epoch,
                identity=example.identity,
                domain="command-shift",
                bound=SHIFT_DRAW_BOUND,
            )
            - SHIFT_OFFSET
        )
        shifted = _zero_fill_shift(waveform, shift)
        gained = _apply_gain(
            shifted,
            seed=seed,
            epoch=epoch,
            identity=example.identity,
            domain="command-gain-db",
        )

        if bernoulli(
            seed=seed,
            epoch=epoch,
            identity=example.identity,
            domain="command-noise-apply",
            probability=NOISE_PROBABILITY,
        ):
            noise_example = select_command_noise_window(
                self._corpus,
                example.source,
                seed=seed,
                epoch=epoch,
            )
            noise_payload = self._cache.read_background_window_pcm16le(
                noise_example.source,
                noise_example.start_sample,
            )
            noise = _decode_pcm16le_window(noise_payload)
            snr_db = uniform_real(
                seed=seed,
                epoch=epoch,
                identity=example.identity,
                domain="command-noise-snr-db",
                minimum=0.0,
                maximum_exclusive=20.0,
            )
            gained = _mix_at_snr(gained, noise, snr_db)

        return _clamp_waveform(gained)

    def silence_waveform(
        self,
        example: WindowExample,
        *,
        seed: int,
        epoch: int,
    ) -> FloatArray:
        """Read and augment one identity-bound training silence window."""

        _require_training_context(seed, epoch)
        if not isinstance(example, WindowExample):
            raise TypeError("example must be a WindowExample")
        if example.source not in self._train_backgrounds:
            raise Experiment002PreprocessingError(
                "silence augmentation requires a corpus training background"
            )
        payload = self._cache.read_background_window_pcm16le(
            example.source,
            example.start_sample,
        )
        waveform = _decode_pcm16le_window(payload)
        gained = _apply_gain(
            waveform,
            seed=seed,
            epoch=epoch,
            identity=example.identity,
            domain="silence-gain-db",
        )
        return _clamp_waveform(gained)

    def command_model_input(
        self,
        example: CommandExample,
        *,
        seed: int,
        epoch: int,
    ) -> FloatArray:
        """Return one augmented, normalized, masked command model input."""

        waveform = self.command_waveform(example, seed=seed, epoch=epoch)
        return _training_model_input(
            example,
            waveform,
            self._stats,
            seed=seed,
            epoch=epoch,
        )

    def silence_model_input(
        self,
        example: WindowExample,
        *,
        seed: int,
        epoch: int,
    ) -> FloatArray:
        """Return one augmented, normalized, masked silence model input."""

        waveform = self.silence_waveform(example, seed=seed, epoch=epoch)
        return _training_model_input(
            example,
            waveform,
            self._stats,
            seed=seed,
            epoch=epoch,
        )


def unaugmented_log_mel(waveform: FloatArray) -> FloatArray:
    """Produce authoritative time-major log-mel frames without any RNG."""

    source = _validated_waveform("waveform", waveform)
    log_mel = complete_log_mel_frames(source)
    if log_mel.dtype != np.dtype(np.float32) or log_mel.shape != (
        TIME_FRAMES,
        MEL_BINS,
    ):
        raise Experiment002PreprocessingError(
            "streaming frontend returned an invalid feature matrix"
        )
    if not np.all(np.isfinite(log_mel)):
        raise Experiment002PreprocessingError(
            "streaming frontend returned non-finite values"
        )
    return np.ascontiguousarray(log_mel, dtype=np.float32)


def _training_model_input(
    example: CommandExample | WindowExample,
    waveform: FloatArray,
    stats: NormalizationStats,
    *,
    seed: int,
    epoch: int,
) -> FloatArray:
    """Normalize, mask, and transpose one augmented training waveform."""

    _require_training_context(seed, epoch)
    if isinstance(example, CommandExample):
        domain_prefix = "command"
    elif isinstance(example, WindowExample):
        domain_prefix = "silence"
    else:
        raise TypeError("example must be a CommandExample or WindowExample")

    normalized = normalize_log_mel(unaugmented_log_mel(waveform), stats)
    masked = _apply_feature_masks(
        normalized,
        seed=seed,
        epoch=epoch,
        identity=example.identity,
        domain_prefix=domain_prefix,
    )
    return _model_layout(masked)


def unaugmented_model_input(
    waveform: FloatArray,
    normalization: VerifiedNormalization,
) -> FloatArray:
    """Normalize and transpose a validation, calibration, or replay clip."""

    if type(normalization) is not VerifiedNormalization:
        raise TypeError("normalization must be a VerifiedNormalization")
    verify_registered_normalization(normalization)
    stats = normalization.stats
    normalized = normalize_log_mel(unaugmented_log_mel(waveform), stats)
    return _model_layout(normalized)


def _require_training_context(seed: int, epoch: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if seed not in TRAINING_SEEDS:
        raise Experiment002PreprocessingError("seed is not registered for training")
    if not isinstance(epoch, int) or isinstance(epoch, bool):
        raise TypeError("epoch must be an integer")
    if not 0 <= epoch < TRAIN_EPOCH_COUNT:
        raise Experiment002PreprocessingError(
            "epoch is outside the registered training range"
        )


def _decode_padded_command(payload: bytes, *, sample_count: int) -> FloatArray:
    if not isinstance(sample_count, int) or isinstance(sample_count, bool):
        raise TypeError("sample_count must be an integer")
    if not 1 <= sample_count <= WINDOW_SAMPLES:
        raise Experiment002PreprocessingError(
            "command sample_count is outside the registered range"
        )
    decoded = _decode_pcm16le(payload, expected_samples=sample_count)
    padded = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    padded[:sample_count] = decoded
    return padded


def _decode_pcm16le_window(payload: bytes) -> FloatArray:
    return _decode_pcm16le(payload, expected_samples=WINDOW_SAMPLES)


def _decode_pcm16le(payload: bytes, *, expected_samples: int) -> FloatArray:
    if not isinstance(payload, bytes):
        raise TypeError("PCM payload must be bytes")
    if len(payload) != 2 * expected_samples:
        raise Experiment002PreprocessingError("PCM payload byte count differs")
    integers = np.frombuffer(payload, dtype="<i2")
    decoded = np.divide(
        integers.astype(np.float32),
        np.float32(32_768.0),
        dtype=np.float32,
    )
    if decoded.dtype != np.dtype(np.float32) or decoded.shape != (expected_samples,):
        raise Experiment002PreprocessingError("decoded PCM shape or dtype differs")
    if not np.all(np.isfinite(decoded)):
        raise Experiment002PreprocessingError("decoded PCM is non-finite")
    return np.ascontiguousarray(decoded, dtype=np.float32)


def _validated_waveform(name: str, waveform: FloatArray) -> FloatArray:
    if not isinstance(waveform, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if waveform.dtype != np.dtype(np.float32):
        raise Experiment002PreprocessingError(f"{name} must use float32")
    if waveform.shape != (WINDOW_SAMPLES,):
        raise Experiment002PreprocessingError(
            f"{name} must contain exactly {WINDOW_SAMPLES} samples"
        )
    if not np.all(np.isfinite(waveform)):
        raise Experiment002PreprocessingError(f"{name} contains non-finite values")
    maximum = np.max(np.abs(waveform.astype(np.float64)))
    if maximum > 1.0:
        raise Experiment002PreprocessingError(
            f"{name} leaves the normalized [-1, 1] interval"
        )
    return np.ascontiguousarray(waveform, dtype=np.float32)


def _zero_fill_shift(waveform: FloatArray, shift: int) -> FloatArray:
    if not -SHIFT_OFFSET <= shift <= SHIFT_OFFSET:
        raise Experiment002PreprocessingError("shift is outside the registered range")
    shifted = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    if shift >= 0:
        shifted[shift:] = waveform[: WINDOW_SAMPLES - shift]
    else:
        shifted[:shift] = waveform[-shift:]
    return shifted


def _apply_gain(
    waveform: FloatArray,
    *,
    seed: int,
    epoch: int,
    identity: bytes,
    domain: str,
) -> FloatArray:
    gain_db = uniform_real(
        seed=seed,
        epoch=epoch,
        identity=identity,
        domain=domain,
        minimum=-6.0,
        maximum_exclusive=6.0,
    )
    factor = np.float32(10.0 ** (gain_db / 20.0))
    if not np.isfinite(factor):
        raise Experiment002PreprocessingError("gain factor is non-finite")
    gained = np.multiply(waveform, factor, dtype=np.float32)
    if not np.all(np.isfinite(gained)):
        raise Experiment002PreprocessingError("gain produced non-finite values")
    return gained


def _mix_at_snr(
    command: FloatArray,
    noise: FloatArray,
    snr_db: float,
) -> FloatArray:
    command_squares = np.square(command, dtype=np.float64)
    command_sum = np.sum(command_squares, dtype=np.float64)
    command_rms = np.sqrt(command_sum / np.float64(WINDOW_SAMPLES))
    noise_squares = np.square(noise, dtype=np.float64)
    noise_sum = np.sum(noise_squares, dtype=np.float64)
    noise_rms = np.sqrt(noise_sum / np.float64(WINDOW_SAMPLES))
    if not np.isfinite(command_rms) or not np.isfinite(noise_rms):
        raise Experiment002PreprocessingError("RMS calculation is non-finite")
    if noise_rms == np.float64(0.0):
        raise Experiment002PreprocessingError("noise RMS must be positive")
    if command_rms == np.float64(0.0):
        scale = np.float32(0.0)
    else:
        ratio = 10.0 ** (snr_db / 20.0)
        scale64 = float(command_rms) / (float(noise_rms) * ratio)
        if not math.isfinite(ratio) or not math.isfinite(scale64):
            raise Experiment002PreprocessingError("SNR scale is non-finite")
        scale = np.float32(scale64)
    if not np.isfinite(scale):
        raise Experiment002PreprocessingError("float32 SNR scale is non-finite")
    scaled_noise = np.multiply(noise, scale, dtype=np.float32)
    mixed = np.add(command, scaled_noise, dtype=np.float32)
    if not np.all(np.isfinite(scaled_noise)) or not np.all(np.isfinite(mixed)):
        raise Experiment002PreprocessingError("noise mix is non-finite before clamp")
    return mixed


def _clamp_waveform(waveform: FloatArray) -> FloatArray:
    if not np.all(np.isfinite(waveform)):
        raise Experiment002PreprocessingError("waveform is non-finite before clamp")
    clamped = np.clip(waveform, np.float32(-1.0), np.float32(1.0))
    if clamped.dtype != np.dtype(np.float32) or not np.all(np.isfinite(clamped)):
        raise Experiment002PreprocessingError("clamp produced an invalid waveform")
    return np.ascontiguousarray(clamped, dtype=np.float32)


def _apply_feature_masks(
    normalized: FloatArray,
    *,
    seed: int,
    epoch: int,
    identity: bytes,
    domain_prefix: str,
) -> FloatArray:
    if normalized.dtype != np.dtype(np.float32) or normalized.shape != (
        TIME_FRAMES,
        MEL_BINS,
    ):
        raise Experiment002PreprocessingError(
            "normalized features have an invalid shape or dtype"
        )
    masked = normalized.copy(order="C")
    time_width = uniform_integer(
        seed=seed,
        epoch=epoch,
        identity=identity,
        domain=f"{domain_prefix}-time-width",
        bound=TIME_MASK_WIDTH_BOUND,
    )
    time_start = uniform_integer(
        seed=seed,
        epoch=epoch,
        identity=identity,
        domain=f"{domain_prefix}-time-start",
        bound=TIME_FRAMES - time_width + 1,
    )
    masked[time_start : time_start + time_width, :] = np.float32(0.0)

    mel_width = uniform_integer(
        seed=seed,
        epoch=epoch,
        identity=identity,
        domain=f"{domain_prefix}-mel-width",
        bound=MEL_MASK_WIDTH_BOUND,
    )
    mel_start = uniform_integer(
        seed=seed,
        epoch=epoch,
        identity=identity,
        domain=f"{domain_prefix}-mel-start",
        bound=MEL_BINS - mel_width + 1,
    )
    masked[:, mel_start : mel_start + mel_width] = np.float32(0.0)
    if not np.all(np.isfinite(masked)):
        raise Experiment002PreprocessingError(
            "feature masks produced non-finite values"
        )
    return masked


def _model_layout(time_major: FloatArray) -> FloatArray:
    if time_major.dtype != np.dtype(np.float32) or time_major.shape != (
        TIME_FRAMES,
        MEL_BINS,
    ):
        raise Experiment002PreprocessingError(
            "time-major features have an invalid shape or dtype"
        )
    model_input = np.ascontiguousarray(time_major.T, dtype=np.float32)
    if model_input.shape != (MEL_BINS, TIME_FRAMES) or not np.all(
        np.isfinite(model_input)
    ):
        raise Experiment002PreprocessingError("model input is invalid")
    return model_input
