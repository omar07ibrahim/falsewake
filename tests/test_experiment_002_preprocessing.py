from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import NoReturn, cast

import numpy as np
import pytest

import falsewake.experiment_002_preprocessing as preprocessing
from falsewake.experiment_002_data import (
    BackgroundSource,
    CommandExample,
    CommandSource,
    Experiment002Corpus,
    Experiment002DataError,
    WindowExample,
)
from falsewake.experiment_002_normalization import NormalizationStats
from falsewake.experiment_002_pcm_cache import PCMCacheSplitView
from falsewake.experiment_002_preprocessing import (
    Experiment002PreprocessingError,
    Experiment002TrainingPreprocessor,
    unaugmented_log_mel,
    unaugmented_model_input,
)
from falsewake.experiment_002_rng import uniform_integer
from falsewake.features import FloatArray

SEED = 20_260_719
COMMAND_SHA256 = "1" * 64
BACKGROUND_SHA256 = "2" * 64
VALIDATION_BACKGROUND_SHA256 = "3" * 64


class _FakeTrainingCache:
    def __init__(
        self,
        commands: dict[CommandSource, bytes],
        windows: dict[tuple[BackgroundSource, int], bytes],
        *,
        split: str = "train",
    ) -> None:
        self.split = split
        self._commands = commands
        self._windows = windows
        self.command_reads: list[CommandSource] = []
        self.window_reads: list[tuple[BackgroundSource, int]] = []

    def read_command_pcm16le(self, source: CommandSource) -> bytes:
        self.command_reads.append(source)
        try:
            return self._commands[source]
        except KeyError as error:
            raise AssertionError("unexpected command cache read") from error

    def read_background_window_pcm16le(
        self,
        source: BackgroundSource,
        start_sample: int,
    ) -> bytes:
        key = source, start_sample
        self.window_reads.append(key)
        try:
            return self._windows[key]
        except KeyError as error:
            raise AssertionError("unexpected background cache read") from error


def _command_source(
    path: str = "yes/alice_nohash_0.wav",
    *,
    sample_count: int = 15_997,
    manifest_index: int = 0,
) -> CommandSource:
    word = path.split("/", maxsplit=1)[0]
    return CommandSource(
        manifest_index=manifest_index,
        path=path,
        word=word,
        label=word,
        sample_count=sample_count,
        sha256=COMMAND_SHA256,
    )


def _command_example(source: CommandSource) -> CommandExample:
    return CommandExample(
        source=source,
        label=source.label,
        label_index=0,
        identity=source.identity,
    )


def _corpus(
    command: CommandSource,
    background: BackgroundSource,
    *,
    validation_commands: tuple[CommandSource, ...] = (),
) -> Experiment002Corpus:
    validation_background = BackgroundSource(
        path="_background_noise_/validation.wav",
        sample_count=16_000,
        sha256=VALIDATION_BACKGROUND_SHA256,
    )
    return Experiment002Corpus(
        train_commands=(command,),
        validation_commands=validation_commands,
        train_backgrounds=(background,),
        validation_background=validation_background,
        test_command_count=0,
        manifest_sha256="4" * 64,
    )


def _synthetic_command_pcm16() -> np.ndarray[tuple[int], np.dtype[np.int16]]:
    sample_index = np.arange(15_997, dtype=np.int64)
    return (((sample_index * 7_919 + 12_345) % 65_536) - 32_768).astype(np.int16)


def _synthetic_noise_pcm16() -> np.ndarray[tuple[int], np.dtype[np.int16]]:
    sample_index = np.arange(16_000, dtype=np.int64)
    return (((sample_index * 3_571 + 22_222) % 65_536) - 32_768).astype(np.int16)


def _pcm16le(values: np.ndarray[tuple[int], np.dtype[np.int16]]) -> bytes:
    return values.astype("<i2", copy=False).tobytes(order="C")


def _decode_and_pad(values: np.ndarray[tuple[int], np.dtype[np.int16]]) -> FloatArray:
    decoded = np.divide(
        values.astype(np.float32),
        np.float32(32_768.0),
        dtype=np.float32,
    )
    padded = np.zeros(16_000, dtype=np.float32)
    padded[: decoded.size] = decoded
    return padded


def _identity_stats() -> NormalizationStats:
    return NormalizationStats(
        means=np.zeros(40, dtype=np.float32),
        standard_deviations=np.ones(40, dtype=np.float32),
    )


def _sha256_float32(values: FloatArray) -> str:
    contents = values.astype("<f4", copy=False).tobytes(order="C")
    return hashlib.sha256(contents).hexdigest()


def _allow_synthetic_context(
    monkeypatch: pytest.MonkeyPatch,
    corpus: Experiment002Corpus,
    cache: _FakeTrainingCache,
    stats: NormalizationStats | None = None,
) -> Experiment002TrainingPreprocessor:
    monkeypatch.setattr(
        preprocessing,
        "_require_registered_training_corpus",
        lambda observed: None,
    )
    monkeypatch.setattr(preprocessing, "PCMCacheSplitView", _FakeTrainingCache)
    return Experiment002TrainingPreprocessor(
        corpus,
        cast(PCMCacheSplitView, cache),
        _identity_stats() if stats is None else stats,
    )


def test_module_has_one_validated_cache_bound_training_context() -> None:
    source = Path("src/falsewake/experiment_002_preprocessing.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    direct_imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert "falsewake.experiment_002_pcm_cache" in imported
    assert "_require_registered_training_corpus" in source
    assert "_train_command_sources" in source
    assert "NoiseWindowLoader" not in source
    assert "load_noise_window" not in source
    assert direct_imports == {"math", "numpy"}
    for forbidden in ("random", "torch", "mmap", "falsewake.features.log_mel"):
        assert forbidden not in source


def test_command_pipeline_matches_golden_and_reads_exact_selected_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _command_source()
    example = _command_example(command)
    background = BackgroundSource(
        path="_background_noise_/exercise_bike.wav",
        sample_count=799_586,
        sha256=BACKGROUND_SHA256,
    )
    noise_example = background.window(783_586)
    corpus = _corpus(command, background)
    cache = _FakeTrainingCache(
        {command: _pcm16le(_synthetic_command_pcm16())},
        {(background, 783_586): _pcm16le(_synthetic_noise_pcm16())},
    )
    processor = _allow_synthetic_context(monkeypatch, corpus, cache)
    monkeypatch.setattr(
        preprocessing,
        "select_command_noise_window",
        lambda *_args, **_kwargs: noise_example,
    )

    augmented = processor.command_waveform(example, seed=SEED, epoch=0)
    model_input = processor.command_model_input(example, seed=SEED, epoch=0)

    assert cache.command_reads == [command, command]
    assert cache.window_reads == [
        (noise_example.source, noise_example.start_sample),
        (noise_example.source, noise_example.start_sample),
    ]
    assert augmented.dtype == np.dtype(np.float32)
    assert augmented.shape == (16_000,)
    assert augmented.flags.c_contiguous
    assert _sha256_float32(augmented) == (
        "8692de6bacedd9036a87198ca5ff79cb0c063b91faab7096278003f1a1814d97"
    )
    assert model_input.dtype == np.dtype(np.float32)
    assert model_input.shape == (40, 98)
    assert model_input.flags.c_contiguous
    assert _sha256_float32(model_input) == (
        "e37f2d9b85a90eeb6639c855a668d0c00df524f423edcd9302fc9570700fa8ad"
    )


def test_false_noise_decision_never_selects_or_reads_background(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _command_source(
        "yes/golden_nohash_0.wav",
        sample_count=16_000,
    )
    example = _command_example(command)
    background = BackgroundSource(
        "_background_noise_/room.wav",
        16_000,
        BACKGROUND_SHA256,
    )
    corpus = _corpus(command, background)
    cache = _FakeTrainingCache(
        {command: _pcm16le(_synthetic_noise_pcm16())},
        {},
    )
    processor = _allow_synthetic_context(monkeypatch, corpus, cache)

    def forbidden(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("noise selection must not run")

    monkeypatch.setattr(preprocessing, "select_command_noise_window", forbidden)
    result = processor.command_waveform(example, seed=SEED, epoch=0)

    assert cache.command_reads == [command]
    assert cache.window_reads == []
    assert result.dtype == np.dtype(np.float32)
    assert np.all(np.isfinite(result))
    assert np.max(np.abs(result.astype(np.float64))) <= 1.0


def test_zero_noise_rms_fails_even_for_a_zero_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _command_source()
    example = _command_example(command)
    background = BackgroundSource(
        "_background_noise_/room.wav",
        16_000,
        BACKGROUND_SHA256,
    )
    noise_example = background.window(0)
    corpus = _corpus(command, background)
    cache = _FakeTrainingCache(
        {command: bytes(2 * command.sample_count)},
        {(background, 0): bytes(32_000)},
    )
    processor = _allow_synthetic_context(monkeypatch, corpus, cache)
    monkeypatch.setattr(
        preprocessing,
        "select_command_noise_window",
        lambda *_args, **_kwargs: noise_example,
    )

    with pytest.raises(Experiment002PreprocessingError, match="noise RMS"):
        processor.command_waveform(example, seed=SEED, epoch=0)


def test_pcm_decode_uses_exact_scale_and_positive_zero_padding() -> None:
    payload = _pcm16le(np.asarray([-32_768, 32_767], dtype=np.int16))
    padded = preprocessing._decode_padded_command(payload, sample_count=2)

    assert padded.dtype == np.dtype(np.float32)
    assert padded.shape == (16_000,)
    assert padded[0] == np.float32(-1.0)
    assert padded[1].view(np.uint32).item() == 0x3F7FFE00
    assert not np.any(padded[2:] != np.float32(0.0))
    assert not np.any(np.signbit(padded[2:]))


def test_silence_reads_one_window_then_uses_gain_without_command_noise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _command_source()
    background = BackgroundSource(
        "_background_noise_/pink_noise.wav",
        28_345,
        BACKGROUND_SHA256,
    )
    corpus = _corpus(command, background)
    example = background.window(12_345)
    payload = _pcm16le(_synthetic_noise_pcm16())
    cache = _FakeTrainingCache({}, {(background, 12_345): payload})
    processor = _allow_synthetic_context(monkeypatch, corpus, cache)

    def forbidden(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("silence must not select command noise")

    monkeypatch.setattr(preprocessing, "select_command_noise_window", forbidden)
    augmented = processor.silence_waveform(example, seed=SEED, epoch=0)
    model_input = processor.silence_model_input(example, seed=SEED, epoch=0)

    waveform = _decode_and_pad(_synthetic_noise_pcm16())
    expected_factor = np.asarray([0x3F9AEB55], dtype=np.uint32).view(np.float32)[0]
    expected = np.clip(
        np.multiply(waveform, expected_factor, dtype=np.float32),
        np.float32(-1.0),
        np.float32(1.0),
    )
    expected_features = unaugmented_log_mel(expected)
    expected_features[44:51, :] = np.float32(0.0)
    expected_features[:, 27:30] = np.float32(0.0)
    expected_model = np.ascontiguousarray(expected_features.T, dtype=np.float32)
    assert cache.window_reads == [(background, 12_345), (background, 12_345)]
    assert augmented.tobytes() == expected.tobytes()
    assert model_input.tobytes() == expected_model.tobytes()


def test_zero_width_masks_still_draw_both_start_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _command_source(
        "yes/golden_nohash_54.wav",
        sample_count=16_000,
    )
    example = _command_example(command)
    waveform = np.linspace(-0.8, 0.8, 16_000, dtype=np.float32)
    stats = _identity_stats()
    observed_domains: list[str] = []

    def recording_draw(
        *,
        seed: int,
        epoch: int,
        identity: bytes,
        domain: str,
        bound: int,
        counter: int = 0,
    ) -> int:
        observed_domains.append(domain)
        return uniform_integer(
            seed=seed,
            epoch=epoch,
            identity=identity,
            domain=domain,
            bound=bound,
            counter=counter,
        )

    monkeypatch.setattr(preprocessing, "uniform_integer", recording_draw)
    training = preprocessing._training_model_input(
        example,
        waveform,
        stats,
        seed=SEED,
        epoch=0,
    )
    unaugmented = unaugmented_model_input(waveform, stats)

    assert observed_domains == [
        "command-time-width",
        "command-time-start",
        "command-mel-width",
        "command-mel-start",
    ]
    assert training.tobytes() == unaugmented.tobytes()


def test_normalization_precedes_masks_with_nontrivial_statistics() -> None:
    command = _command_source(sample_count=16_000)
    example = _command_example(command)
    waveform = np.linspace(-0.73, 0.81, 16_000, dtype=np.float32)
    stats = NormalizationStats(
        means=np.linspace(-2.0, 2.0, 40, dtype=np.float32),
        standard_deviations=np.linspace(0.5, 1.5, 40, dtype=np.float32),
    )
    raw = unaugmented_log_mel(waveform)
    expected = np.divide(
        np.subtract(raw, stats.means, dtype=np.float32),
        stats.standard_deviations,
        dtype=np.float32,
    )
    expected[3:11, :] = np.float32(0.0)
    expected[:, 32:33] = np.float32(0.0)
    expected_model = np.ascontiguousarray(expected.T, dtype=np.float32)

    observed = preprocessing._training_model_input(
        example,
        waveform,
        stats,
        seed=SEED,
        epoch=0,
    )

    assert observed.tobytes() == expected_model.tobytes()
    assert not np.any(np.signbit(observed[:, 3:11]))
    assert not np.any(np.signbit(observed[32:33, :]))
    raw_masked_first = raw.copy()
    raw_masked_first[3:11, :] = np.float32(0.0)
    raw_masked_first[:, 32:33] = np.float32(0.0)
    wrong_order = np.divide(
        np.subtract(raw_masked_first, stats.means, dtype=np.float32),
        stats.standard_deviations,
        dtype=np.float32,
    )
    assert wrong_order.tobytes() != expected.tobytes()


def test_unaugmented_paths_consume_no_augmentation_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waveform = np.linspace(-0.5, 0.5, 16_000, dtype=np.float32)
    stats = _identity_stats()

    def forbidden(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("unaugmented path used augmentation RNG")

    monkeypatch.setattr(preprocessing, "uniform_integer", forbidden)
    monkeypatch.setattr(preprocessing, "uniform_real", forbidden)
    monkeypatch.setattr(preprocessing, "bernoulli", forbidden)

    log_mel = unaugmented_log_mel(waveform)
    model_input = unaugmented_model_input(waveform, stats)
    assert log_mel.shape == (98, 40)
    assert model_input.shape == (40, 98)
    assert model_input.tobytes() == np.ascontiguousarray(log_mel.T).tobytes()


def test_context_rejects_forgeable_corpus_before_cache_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _command_source()
    background = BackgroundSource(
        "_background_noise_/room.wav",
        16_000,
        BACKGROUND_SHA256,
    )
    corpus = _corpus(command, background)
    cache = _FakeTrainingCache({}, {})
    monkeypatch.setattr(preprocessing, "PCMCacheSplitView", _FakeTrainingCache)

    with pytest.raises(Experiment002DataError, match="registered manifest"):
        Experiment002TrainingPreprocessor(
            corpus,
            cast(PCMCacheSplitView, cache),
            _identity_stats(),
        )
    assert cache.command_reads == []
    assert cache.window_reads == []


def test_context_validates_once_then_uses_constant_time_membership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _command_source(
        "yes/golden_nohash_0.wav",
        sample_count=16_000,
    )
    background = BackgroundSource(
        "_background_noise_/room.wav",
        16_000,
        BACKGROUND_SHA256,
    )
    corpus = _corpus(command, background)
    cache = _FakeTrainingCache(
        {command: _pcm16le(_synthetic_noise_pcm16())},
        {},
    )
    validations: list[Experiment002Corpus] = []
    monkeypatch.setattr(preprocessing, "PCMCacheSplitView", _FakeTrainingCache)
    monkeypatch.setattr(
        preprocessing,
        "_require_registered_training_corpus",
        validations.append,
    )
    processor = Experiment002TrainingPreprocessor(
        corpus,
        cast(PCMCacheSplitView, cache),
        _identity_stats(),
    )
    monkeypatch.setattr(
        preprocessing,
        "select_command_noise_window",
        lambda *_args, **_kwargs: pytest.fail("false draw selected noise"),
    )

    processor.command_waveform(_command_example(command), seed=SEED, epoch=0)
    processor.command_waveform(_command_example(command), seed=SEED, epoch=0)

    assert validations == [corpus]
    assert processor._train_commands is corpus._train_command_sources
    assert isinstance(processor._train_backgrounds, frozenset)


def test_context_snapshots_normalization_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _command_source()
    background = BackgroundSource(
        "_background_noise_/room.wav",
        16_000,
        BACKGROUND_SHA256,
    )
    corpus = _corpus(command, background)
    cache = _FakeTrainingCache({}, {})
    stats = NormalizationStats(
        means=np.linspace(-1.0, 1.0, 40, dtype=np.float32),
        standard_deviations=np.linspace(0.5, 1.5, 40, dtype=np.float32),
    )
    expected_means = stats.means.copy()
    expected_deviations = stats.standard_deviations.copy()
    processor = _allow_synthetic_context(monkeypatch, corpus, cache, stats)

    stats.means.flags.writeable = True
    stats.standard_deviations.flags.writeable = True
    stats.means[:] = np.float32(99.0)
    stats.standard_deviations[:] = np.float32(77.0)

    np.testing.assert_array_equal(processor._stats.means, expected_means)
    np.testing.assert_array_equal(
        processor._stats.standard_deviations,
        expected_deviations,
    )
    assert not processor._stats.means.flags.writeable
    assert not processor._stats.standard_deviations.flags.writeable


def test_context_requires_concrete_training_split_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _command_source()
    background = BackgroundSource(
        "_background_noise_/room.wav",
        16_000,
        BACKGROUND_SHA256,
    )
    corpus = _corpus(command, background)
    monkeypatch.setattr(
        preprocessing,
        "_require_registered_training_corpus",
        lambda observed: None,
    )

    with pytest.raises(TypeError, match="PCMCacheSplitView"):
        Experiment002TrainingPreprocessor(
            corpus,
            cast(PCMCacheSplitView, object()),
            _identity_stats(),
        )

    monkeypatch.setattr(preprocessing, "PCMCacheSplitView", _FakeTrainingCache)
    validation_cache = _FakeTrainingCache({}, {}, split="validation")
    with pytest.raises(Experiment002PreprocessingError, match="training cache"):
        Experiment002TrainingPreprocessor(
            corpus,
            cast(PCMCacheSplitView, validation_cache),
            _identity_stats(),
        )


def test_cross_split_examples_cannot_enter_validated_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = _command_source("yes/train_nohash_0.wav", manifest_index=0)
    validation = _command_source(
        "yes/validation_nohash_0.wav",
        manifest_index=1,
    )
    background = BackgroundSource(
        "_background_noise_/room.wav",
        16_000,
        BACKGROUND_SHA256,
    )
    corpus = _corpus(train, background, validation_commands=(validation,))
    cache = _FakeTrainingCache({}, {})
    processor = _allow_synthetic_context(monkeypatch, corpus, cache)

    with pytest.raises(Experiment002PreprocessingError, match="training source"):
        processor.command_waveform(
            _command_example(validation),
            seed=SEED,
            epoch=0,
        )
    with pytest.raises(Experiment002PreprocessingError, match="training background"):
        processor.silence_waveform(
            corpus.validation_background.window(0),
            seed=SEED,
            epoch=0,
        )
    assert cache.command_reads == []
    assert cache.window_reads == []


@pytest.mark.parametrize("seed", [0, 20_260_718, 20_260_722, True])
def test_unregistered_training_seed_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    seed: int,
) -> None:
    command = _command_source(sample_count=16_000)
    background = BackgroundSource(
        "_background_noise_/room.wav",
        16_000,
        BACKGROUND_SHA256,
    )
    corpus = _corpus(command, background)
    cache = _FakeTrainingCache({}, {})
    processor = _allow_synthetic_context(monkeypatch, corpus, cache)

    with pytest.raises((TypeError, Experiment002PreprocessingError)):
        processor.command_waveform(
            _command_example(command),
            seed=seed,
            epoch=0,
        )


@pytest.mark.parametrize("epoch", [-1, 30, True])
def test_unregistered_training_epoch_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    epoch: int,
) -> None:
    command = _command_source(sample_count=16_000)
    background = BackgroundSource(
        "_background_noise_/room.wav",
        16_000,
        BACKGROUND_SHA256,
    )
    corpus = _corpus(command, background)
    cache = _FakeTrainingCache({}, {})
    processor = _allow_synthetic_context(monkeypatch, corpus, cache)

    with pytest.raises((TypeError, Experiment002PreprocessingError)):
        processor.command_waveform(
            _command_example(command),
            seed=SEED,
            epoch=epoch,
        )


@pytest.mark.parametrize(
    "waveform",
    [
        np.zeros(16_000, dtype=np.float64),
        np.zeros(15_999, dtype=np.float32),
        np.full(16_000, np.float32(1.01), dtype=np.float32),
        np.full(16_000, np.float32(np.inf), dtype=np.float32),
    ],
)
def test_invalid_unaugmented_waveforms_fail_before_frontend(
    waveform: np.ndarray[tuple[int], np.dtype[np.floating]],
) -> None:
    with pytest.raises(Experiment002PreprocessingError):
        unaugmented_log_mel(cast(FloatArray, waveform))


def test_cache_payload_shape_and_public_method_types_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _command_source()
    background = BackgroundSource(
        "_background_noise_/room.wav",
        16_000,
        BACKGROUND_SHA256,
    )
    corpus = _corpus(command, background)
    short_cache = _FakeTrainingCache({command: bytes(2)}, {})
    processor = _allow_synthetic_context(monkeypatch, corpus, short_cache)

    with pytest.raises(Experiment002PreprocessingError, match="byte count"):
        processor.command_waveform(
            _command_example(command),
            seed=SEED,
            epoch=0,
        )
    with pytest.raises(TypeError, match="example"):
        processor.command_waveform(
            cast(CommandExample, object()),
            seed=SEED,
            epoch=0,
        )
    with pytest.raises(TypeError, match="example"):
        processor.silence_waveform(
            cast(WindowExample, object()),
            seed=SEED,
            epoch=0,
        )
