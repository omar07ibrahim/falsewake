from __future__ import annotations

import ast
import copy
import hashlib
import json
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

import falsewake.experiment_002_data as experiment_data
import falsewake.experiment_002_normalization_artifact as normalization_artifact
import falsewake.experiment_002_preprocessing as preprocessing
import falsewake.experiment_002_validation as validation
from falsewake.experiment_002_data import (
    CLASS_ORDER,
    MANIFEST_SHA256,
    VALIDATION_COMMAND_INVENTORY_SHA256,
    VALIDATION_EXAMPLE_COUNT,
    VALIDATION_SILENCE_COUNT,
    BackgroundSource,
    CommandExample,
    CommandSource,
    Experiment002Corpus,
    WindowExample,
)
from falsewake.experiment_002_normalization import (
    NormalizationStats,
    serialize_stats,
)
from falsewake.experiment_002_normalization_artifact import (
    NormalizationArtifactIdentity,
    VerifiedNormalization,
    load_normalization_artifact,
)
from falsewake.experiment_002_pcm_cache import (
    REGISTERED_FILE_BYTES,
    PCMCacheIdentity,
    PCMCacheSplitView,
)
from falsewake.experiment_002_preprocessing import (
    _decode_padded_command,
    unaugmented_model_input,
)
from falsewake.experiment_002_validation import (
    Experiment002ValidationError,
    MaterializedValidationPopulation,
)
from falsewake.features import FloatArray

Int16Array = NDArray[np.int16]
Int64Array = NDArray[np.int64]
NORMALIZATION_PATH = Path("models/experiment-002-normalization.f32")
NORMALIZATION_SHA256 = (
    "891900d4c36fa8a71ba429384f3f4ff594de7c81347ed846c0f790dc4579268e"
)
PCM_CACHE_SHA256 = "b56270a4b99d235c62c162545b139a79b7cf3dc157259a3bbf5678d8a451f653"


class _FakeOwner:
    def __init__(
        self,
        *,
        closed: bool = False,
        identity: PCMCacheIdentity | None = None,
    ) -> None:
        self.closed = closed
        self.identity = (
            PCMCacheIdentity(
                byte_count=REGISTERED_FILE_BYTES,
                sha256=PCM_CACHE_SHA256,
            )
            if identity is None
            else identity
        )


class _FakeValidationCache:
    def __init__(
        self,
        commands: dict[CommandSource, bytes] | None = None,
        windows: dict[tuple[BackgroundSource, int], bytes] | None = None,
        *,
        split: str = "validation",
        owner: object | None = None,
        command_reader: Callable[[CommandSource], bytes] | None = None,
        window_reader: Callable[[BackgroundSource, int], bytes] | None = None,
    ) -> None:
        self._cache = _FakeOwner() if owner is None else owner
        self._split = split
        self._commands = {} if commands is None else commands
        self._windows = {} if windows is None else windows
        self._command_reader = command_reader
        self._window_reader = window_reader
        self.command_reads: list[CommandSource] = []
        self.window_reads: list[tuple[BackgroundSource, int]] = []

    @property
    def split(self) -> str:
        return self._split

    def read_command_pcm16le(self, source: CommandSource) -> bytes:
        self.command_reads.append(source)
        if self._command_reader is not None:
            return self._command_reader(source)
        try:
            return self._commands[source]
        except KeyError as error:
            raise AssertionError("unexpected command cache read") from error

    def read_background_window_pcm16le(
        self,
        source: BackgroundSource,
        start_sample: int,
    ) -> bytes:
        self.window_reads.append((source, start_sample))
        if self._window_reader is not None:
            return self._window_reader(source, start_sample)
        try:
            return self._windows[(source, start_sample)]
        except KeyError as error:
            raise AssertionError("unexpected window cache read") from error


def _command_source(
    *,
    label: str = "yes",
    ordinal: int = 0,
    sample_count: int = 5,
) -> CommandSource:
    word = label if label in CLASS_ORDER[:10] else "backward"
    return CommandSource(
        manifest_index=ordinal,
        path=f"{word}/speaker_{ordinal:05d}_nohash_0.wav",
        word=word,
        label=label,
        sample_count=sample_count,
        sha256=f"{ordinal % 16:x}" * 64,
    )


def _command_example(source: CommandSource) -> CommandExample:
    return CommandExample(
        source=source,
        label=source.label,
        label_index=CLASS_ORDER.index(source.label),
        identity=source.identity,
    )


def _background(*, sample_count: int = 978_488) -> BackgroundSource:
    return BackgroundSource(
        path="_background_noise_/running_tap.wav",
        sample_count=sample_count,
        sha256="a" * 64,
    )


def _identity_stats() -> NormalizationStats:
    return NormalizationStats(
        means=np.zeros(40, dtype=np.float32),
        standard_deviations=np.ones(40, dtype=np.float32),
    )


def _issued_normalization(stats: NormalizationStats) -> VerifiedNormalization:
    contents = serialize_stats(stats)
    identity = NormalizationArtifactIdentity(
        byte_count=len(contents),
        sha256=hashlib.sha256(contents).hexdigest(),
    )
    return normalization_artifact._issue_verified_normalization(
        contents,
        identity,
        normalization_artifact._REGISTERED_LAYOUT,
    )


def _registered_normalization() -> VerifiedNormalization:
    return load_normalization_artifact(
        NORMALIZATION_PATH,
        NormalizationArtifactIdentity(
            byte_count=320,
            sha256=NORMALIZATION_SHA256,
        ),
    )


def _pcm16le(values: Int16Array) -> bytes:
    return values.astype("<i2", copy=False).tobytes(order="C")


def _install_fake_cache_types(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(validation, "PCMCacheSplitView", _FakeValidationCache)
    monkeypatch.setattr(validation, "Experiment002PCMCache", _FakeOwner)


def _tiny_layout(
    *labels: int,
    frame_count: int = 98,
) -> validation._ValidationLayout:
    support = [0] * len(CLASS_ORDER)
    for label in labels:
        support[label] += 1
    return validation._ValidationLayout(
        example_count=len(labels),
        frame_count=frame_count,
        class_support=tuple(support),
    )


def _tiny_issued_population(
    *,
    registered: bool,
) -> tuple[
    Experiment002Corpus,
    tuple[CommandExample, ...],
    FloatArray,
    Int64Array,
    MaterializedValidationPopulation,
]:
    source = _command_source(sample_count=1)
    examples = (_command_example(source),)
    corpus = Experiment002Corpus(
        train_commands=(),
        validation_commands=(source,),
        train_backgrounds=(
            BackgroundSource(
                path="_background_noise_/train.wav",
                sample_count=16_000,
                sha256="b" * 64,
            ),
        ),
        validation_background=_background(sample_count=16_000),
        test_command_count=11_005,
        manifest_sha256=MANIFEST_SHA256,
    )
    model_inputs = np.zeros((1, 40, 98), dtype=np.float32)
    label_indices = np.array([0], dtype=np.int64)
    model_inputs.setflags(write=False)
    label_indices.setflags(write=False)
    population = validation._issue_materialized_validation(
        examples,
        model_inputs,
        label_indices,
        registered_corpus=corpus if registered else None,
        registered_marker=(
            validation._REGISTERED_MATERIALIZATION_MARKER if registered else None
        ),
    )
    return corpus, examples, model_inputs, label_indices, population


def _install_tiny_registered_population_guards(
    monkeypatch: pytest.MonkeyPatch,
    corpus: Experiment002Corpus,
    examples: tuple[CommandExample, ...],
) -> None:
    source = examples[0].source
    corpus_snapshot = (
        corpus.manifest_sha256,
        corpus.validation_commands,
        corpus.validation_background,
    )
    example_snapshot = (
        examples[0].source,
        examples[0].label,
        examples[0].label_index,
        examples[0].identity,
    )

    def require_corpus(observed: Experiment002Corpus) -> None:
        if (
            observed is not corpus
            or (
                observed.manifest_sha256,
                observed.validation_commands,
                observed.validation_background,
            )
            != corpus_snapshot
        ):
            raise Experiment002ValidationError("canonical corpus mismatch")

    def require_population(
        observed_corpus: Experiment002Corpus,
        observed_examples: tuple[CommandExample | WindowExample, ...],
    ) -> None:
        if (
            observed_corpus is not corpus
            or observed_examples is not examples
            or len(observed_examples) != 1
            or type(observed_examples[0]) is not CommandExample
        ):
            raise Experiment002ValidationError("canonical order mismatch")
        observed = observed_examples[0]
        if (
            observed.source is not source
            or (
                observed.source,
                observed.label,
                observed.label_index,
                observed.identity,
            )
            != example_snapshot
        ):
            raise Experiment002ValidationError("canonical metadata mismatch")

    monkeypatch.setattr(
        validation,
        "_require_registered_validation_corpus",
        require_corpus,
    )
    monkeypatch.setattr(
        validation,
        "_require_registered_validation_population",
        require_population,
    )


def test_validation_module_is_rng_free_and_reconciles_frozen_contracts() -> None:
    source_path = Path("src/falsewake/experiment_002_validation.py")
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )

    assert not any(
        name == forbidden or name.startswith(f"{forbidden}.")
        for name in imported
        for forbidden in ("random", "secrets", "torch", "falsewake.experiment_002_rng")
    )
    assert "np.random" not in source
    assert "_training_model_input" not in source
    assert "build_training_epoch" not in source
    assert "unaugmented_model_input" not in source

    trainer = json.loads(
        Path("configs/experiment-002-trainer.json").read_text(encoding="utf-8")
    )
    assert tuple(trainer["metrics"]["support"]) == validation.VALIDATION_CLASS_SUPPORT
    assert sum(validation.VALIDATION_CLASS_SUPPORT) == VALIDATION_EXAMPLE_COUNT
    assert (
        trainer["bindings"]["pcm_cache"]["artifact_sha256"]
        == validation.REGISTERED_PCM_CACHE_SHA256
    )
    assert (
        trainer["bindings"]["normalization"]["artifact_sha256"]
        == validation.REGISTERED_NORMALIZATION_SHA256
    )
    assert (
        validation._ValidationLayout(
            example_count=10_583,
            frame_count=98,
            class_support=validation.VALIDATION_CLASS_SUPPORT,
        )
        == validation._REGISTERED_LAYOUT
    )


def test_tiny_command_and_window_use_exact_pcm_and_owned_read_only_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _command_source(sample_count=5)
    command_example = _command_example(command)
    background = _background(sample_count=16_123)
    window_example = background.window(123)
    command_values = np.array([-32_768, -1, 0, 1, 32_767], dtype=np.int16)
    window_values = (
        (np.arange(16_000, dtype=np.int64) * 7_919 + 12_345) % 65_536 - 32_768
    ).astype(np.int16)
    fake = _FakeValidationCache(
        {command: _pcm16le(command_values)},
        {(background, 123): _pcm16le(window_values)},
    )
    captured: list[FloatArray] = []

    def extract(waveform: FloatArray, stats: NormalizationStats) -> FloatArray:
        assert stats is identity_stats
        captured.append(waveform.copy(order="C"))
        return np.full((40, 98), len(captured), dtype=np.float32)

    identity_stats = _identity_stats()
    monkeypatch.setattr(validation, "PCMCacheSplitView", _FakeValidationCache)
    result = validation._materialize_validation_population(
        (command_example, window_example),
        cast(PCMCacheSplitView, fake),
        identity_stats,
        layout=_tiny_layout(0, 11),
        model_input_extractor=extract,
    )

    assert fake.command_reads == [command]
    assert fake.window_reads == [(background, 123)]
    expected_command = np.divide(
        command_values.astype(np.float32),
        np.float32(32_768.0),
        dtype=np.float32,
    )
    np.testing.assert_array_equal(captured[0][:5], expected_command)
    assert np.all(captured[0][5:].view(np.uint32) == np.uint32(0))
    expected_window = np.divide(
        window_values.astype(np.float32),
        np.float32(32_768.0),
        dtype=np.float32,
    )
    np.testing.assert_array_equal(captured[1], expected_window)
    assert result.examples == (command_example, window_example)
    np.testing.assert_array_equal(result.label_indices, np.array([0, 11], np.int64))
    assert np.all(result.model_inputs[0] == np.float32(1.0))
    assert np.all(result.model_inputs[1] == np.float32(2.0))
    assert result.model_inputs.flags.c_contiguous
    assert result.model_inputs.flags.owndata
    assert not result.model_inputs.flags.writeable
    assert result.label_indices.flags.c_contiguous
    assert result.label_indices.flags.owndata
    assert not result.label_indices.flags.writeable
    private_state = validation._issued_materialized_state(result)
    assert private_state.registered_marker is None
    assert private_state.registered_corpus is None
    with pytest.raises(ValueError, match="read-only"):
        result.model_inputs[0, 0, 0] = np.float32(0.0)
    with pytest.raises(ValueError, match="read-only"):
        result.label_indices[0] = np.int64(1)


def test_one_row_matches_the_existing_unaugmented_path_byte_for_byte(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _command_source(sample_count=15_997)
    example = _command_example(command)
    indices = np.arange(command.sample_count, dtype=np.int64)
    values = (((indices * 3_571 + 22_222) % 65_536) - 32_768).astype(np.int16)
    payload = _pcm16le(values)
    fake = _FakeValidationCache({command: payload})
    stats = NormalizationStats(
        means=np.linspace(-3.0, 2.0, 40, dtype=np.float32),
        standard_deviations=np.linspace(0.5, 2.5, 40, dtype=np.float32),
    )
    capability = _issued_normalization(stats)

    def forbid_rng(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"unexpected augmentation RNG: {args!r}, {kwargs!r}")

    monkeypatch.setattr(preprocessing, "bernoulli", forbid_rng)
    monkeypatch.setattr(preprocessing, "uniform_integer", forbid_rng)
    monkeypatch.setattr(preprocessing, "uniform_real", forbid_rng)
    monkeypatch.setattr(validation, "PCMCacheSplitView", _FakeValidationCache)
    result = validation._materialize_validation_population(
        (example,),
        cast(PCMCacheSplitView, fake),
        capability.stats,
        layout=_tiny_layout(0),
    )
    expected = unaugmented_model_input(
        _decode_padded_command(payload, sample_count=command.sample_count),
        capability,
    )

    assert fake.command_reads == [command]
    assert result.model_inputs[0].tobytes(order="C") == expected.tobytes(order="C")


def test_full_population_order_support_and_copying_use_under_two_mib(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[CommandExample] = []
    ordinal = 0
    for label, count in zip(
        CLASS_ORDER[:-1],
        validation.VALIDATION_CLASS_SUPPORT[:-1],
        strict=True,
    ):
        for _ in range(count):
            source = _command_source(
                label=label,
                ordinal=ordinal,
                sample_count=1,
            )
            commands.append(_command_example(source))
            ordinal += 1
    background = _background()
    windows = tuple(background.window(index * 1_600) for index in range(602))
    examples: tuple[CommandExample | WindowExample, ...] = (*commands, *windows)
    assert len(examples) == VALIDATION_EXAMPLE_COUNT

    fake = _FakeValidationCache(
        command_reader=lambda source: bytes(2 * source.sample_count),
        window_reader=lambda source, start: bytes(32_000),
    )
    reusable = np.empty((40, 1), dtype=np.float32)
    calls = 0

    def extract(waveform: FloatArray, stats: NormalizationStats) -> FloatArray:
        nonlocal calls
        assert waveform.shape == (16_000,)
        assert stats is identity_stats
        calls += 1
        reusable.fill(np.float32(calls))
        return reusable

    identity_stats = _identity_stats()
    monkeypatch.setattr(validation, "PCMCacheSplitView", _FakeValidationCache)
    model_inputs, labels = validation._materialize_validation_arrays(
        examples,
        cast(PCMCacheSplitView, fake),
        identity_stats,
        layout=validation._ValidationLayout(
            example_count=10_583,
            frame_count=1,
            class_support=validation.VALIDATION_CLASS_SUPPORT,
        ),
        model_input_extractor=extract,
    )

    assert calls == 10_583
    assert fake.command_reads == [example.source for example in commands]
    assert fake.window_reads[0] == (background, 0)
    assert fake.window_reads[-1] == (background, 961_600)
    assert len(fake.window_reads) == VALIDATION_SILENCE_COUNT
    assert model_inputs.nbytes + labels.nbytes < 2 * 1_024**2
    assert np.all(model_inputs[0] == np.float32(1.0))
    assert np.all(model_inputs[1] == np.float32(2.0))
    assert np.all(model_inputs[-1] == np.float32(10_583.0))
    assert tuple(int(value) for value in np.bincount(labels, minlength=12)) == (
        validation.VALIDATION_CLASS_SUPPORT
    )
    assert not model_inputs.flags.writeable
    assert not labels.flags.writeable


def test_corpus_cache_and_normalization_trust_boundaries_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _command_source(ordinal=1)
    forged = _command_source(ordinal=2)
    train_background = BackgroundSource(
        path="_background_noise_/train.wav",
        sample_count=16_000,
        sha256="b" * 64,
    )
    corpus = Experiment002Corpus(
        train_commands=(),
        validation_commands=(original,),
        train_backgrounds=(train_background,),
        validation_background=_background(),
        test_command_count=11_005,
        manifest_sha256=MANIFEST_SHA256,
    )
    object.__setattr__(
        corpus,
        "_validation_inventory_sha256",
        VALIDATION_COMMAND_INVENTORY_SHA256,
    )
    object.__setattr__(corpus, "validation_commands", (forged,))
    canonical = experiment_data._require_registered_validation_corpus
    monkeypatch.setattr(
        validation,
        "_require_registered_data_corpus",
        lambda value: None,
    )
    with pytest.raises(Experiment002ValidationError, match="inventory"):
        validation._require_registered_validation_corpus(corpus)

    class CorpusSubclass(Experiment002Corpus):
        pass

    subclass = CorpusSubclass(
        train_commands=(),
        validation_commands=(original,),
        train_backgrounds=(train_background,),
        validation_background=_background(),
        test_command_count=11_005,
        manifest_sha256=MANIFEST_SHA256,
    )
    with pytest.raises(TypeError, match="Experiment002Corpus"):
        validation._require_registered_validation_corpus(subclass)

    monkeypatch.setattr(
        validation,
        "_require_registered_data_corpus",
        canonical,
    )
    registered_background = BackgroundSource(
        path="_background_noise_/running_tap.wav",
        sample_count=978_488,
        sha256="c199c5fd61f5bf9fd57f1c346c8a51c0c035d63d176348dd9f0f7d671d7eb8eb",
    )
    for field, value, message in (
        ("manifest_sha256", "0" * 64, "manifest"),
        ("test_command_count", 11_004, "test command count"),
        (
            "validation_background",
            _background(sample_count=978_487),
            "background identity",
        ),
    ):
        mutated = copy.copy(corpus)
        object.__setattr__(mutated, "validation_commands", (original,))
        object.__setattr__(mutated, "validation_background", registered_background)
        object.__setattr__(
            mutated,
            "_validation_inventory_sha256",
            VALIDATION_COMMAND_INVENTORY_SHA256,
        )
        object.__setattr__(mutated, field, value)
        with pytest.raises(ValueError, match=message):
            validation._require_registered_validation_corpus(mutated)

    _install_fake_cache_types(monkeypatch)
    good_cache = _FakeValidationCache()
    validation._require_registered_validation_cache(cast(PCMCacheSplitView, good_cache))

    class CacheSubclass(_FakeValidationCache):
        pass

    with pytest.raises(TypeError, match="PCMCacheSplitView"):
        validation._require_registered_validation_cache(
            cast(PCMCacheSplitView, CacheSubclass())
        )

    class OwnerSubclass(_FakeOwner):
        pass

    with pytest.raises(Experiment002ValidationError, match="owner"):
        validation._require_registered_validation_cache(
            cast(PCMCacheSplitView, _FakeValidationCache(owner=OwnerSubclass()))
        )

    class LyingSplit(str):
        def __eq__(self, other: object) -> bool:
            return True

    lying = _FakeValidationCache()
    lying._split = cast(str, LyingSplit("train"))
    with pytest.raises(Experiment002ValidationError, match="validation cache view"):
        validation._require_registered_validation_cache(cast(PCMCacheSplitView, lying))
    with pytest.raises(Experiment002ValidationError, match="closed"):
        validation._require_registered_validation_cache(
            cast(
                PCMCacheSplitView,
                _FakeValidationCache(owner=_FakeOwner(closed=True)),
            )
        )
    wrong_identity = PCMCacheIdentity(
        byte_count=REGISTERED_FILE_BYTES,
        sha256="0" * 64,
    )
    with pytest.raises(Experiment002ValidationError, match="frozen report"):
        validation._require_registered_validation_cache(
            cast(
                PCMCacheSplitView,
                _FakeValidationCache(
                    owner=_FakeOwner(identity=wrong_identity),
                ),
            )
        )

    capability = _registered_normalization()
    calls = 0
    real_verify = normalization_artifact.verify_registered_normalization

    def count_verify(value: VerifiedNormalization) -> None:
        nonlocal calls
        calls += 1
        real_verify(value)

    monkeypatch.setattr(validation, "verify_registered_normalization", count_verify)
    observed_stats = validation._registered_normalization_stats(capability)
    assert calls == 1
    assert observed_stats.means.shape == (40,)

    wrong_capability = _issued_normalization(_identity_stats())
    with pytest.raises(Experiment002ValidationError, match="frozen report"):
        validation._registered_normalization_stats(wrong_capability)
    with pytest.raises((TypeError, ValueError)):
        validation._registered_normalization_stats(copy.copy(capability))
    with pytest.raises((TypeError, ValueError)):
        validation._registered_normalization_stats(
            object.__new__(VerifiedNormalization)
        )


def test_public_entrypoint_verifies_each_trust_boundary_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _command_source(sample_count=1)
    examples = (_command_example(source),)
    fake_cache = _FakeValidationCache({source: bytes(2)})
    typed_cache = cast(PCMCacheSplitView, fake_cache)
    capability = _registered_normalization()
    sentinel = cast(MaterializedValidationPopulation, object())
    events: list[str] = []
    verify_calls = 0
    returned_stats: NormalizationStats | None = None
    real_normalization = validation._registered_normalization_stats
    real_verify = normalization_artifact.verify_registered_normalization

    def require_corpus(value: Experiment002Corpus) -> None:
        events.append("corpus")

    def require_cache(value: PCMCacheSplitView) -> None:
        events.append("cache")

    def verify(value: VerifiedNormalization) -> None:
        nonlocal verify_calls
        verify_calls += 1
        real_verify(value)

    def registered_stats(value: VerifiedNormalization) -> NormalizationStats:
        nonlocal returned_stats
        events.append("normalization")
        returned_stats = real_normalization(value)
        return returned_stats

    def build(value: Experiment002Corpus) -> tuple[CommandExample, ...]:
        events.append("build")
        return examples

    def require_population(
        value: Experiment002Corpus,
        observed: tuple[CommandExample | WindowExample, ...],
    ) -> None:
        events.append("population")
        assert observed == examples

    def materialize(
        observed: tuple[CommandExample | WindowExample, ...],
        cache: PCMCacheSplitView,
        stats: NormalizationStats,
        *,
        layout: validation._ValidationLayout,
        model_input_extractor: validation._ModelInputExtractor | None = None,
        registered_corpus: Experiment002Corpus | None = None,
        registered_marker: object | None = None,
    ) -> MaterializedValidationPopulation:
        events.append("materialize")
        assert observed == examples
        assert cache is typed_cache
        assert stats is returned_stats
        assert layout is validation._REGISTERED_LAYOUT
        assert model_input_extractor is None
        assert registered_corpus is corpus_argument
        assert registered_marker is validation._REGISTERED_MATERIALIZATION_MARKER
        return sentinel

    monkeypatch.setattr(
        validation, "_require_registered_validation_corpus", require_corpus
    )
    monkeypatch.setattr(
        validation, "_require_registered_validation_cache", require_cache
    )
    monkeypatch.setattr(validation, "verify_registered_normalization", verify)
    monkeypatch.setattr(validation, "_registered_normalization_stats", registered_stats)
    monkeypatch.setattr(validation, "build_validation_population", build)
    monkeypatch.setattr(
        validation,
        "_require_registered_validation_population",
        require_population,
    )
    monkeypatch.setattr(validation, "_materialize_validation_population", materialize)

    corpus_argument = cast(Experiment002Corpus, object())
    observed = validation.materialize_registered_validation(
        corpus_argument,
        typed_cache,
        capability,
    )
    assert observed is sentinel
    assert events == [
        "corpus",
        "cache",
        "normalization",
        "build",
        "population",
        "materialize",
    ]
    assert verify_calls == 1


def test_registered_population_guard_enforces_full_canonical_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources: list[CommandSource] = []
    ordinal = 0
    for label, count in zip(
        CLASS_ORDER[:-1],
        validation.VALIDATION_CLASS_SUPPORT[:-1],
        strict=True,
    ):
        for _ in range(count):
            sources.append(
                _command_source(
                    label=label,
                    ordinal=ordinal,
                    sample_count=1,
                )
            )
            ordinal += 1
    background = _background()
    corpus = Experiment002Corpus(
        train_commands=(),
        validation_commands=tuple(reversed(sources)),
        train_backgrounds=(
            BackgroundSource(
                path="_background_noise_/train.wav",
                sample_count=16_000,
                sha256="b" * 64,
            ),
        ),
        validation_background=background,
        test_command_count=11_005,
        manifest_sha256=MANIFEST_SHA256,
    )
    monkeypatch.setattr(
        experiment_data,
        "_require_registered_validation_corpus",
        lambda value: None,
    )
    examples = experiment_data.build_validation_population(corpus)

    validation._require_registered_validation_population(corpus, examples)
    command_count = VALIDATION_EXAMPLE_COUNT - VALIDATION_SILENCE_COUNT
    first_window = cast(WindowExample, examples[command_count])
    last_window = cast(WindowExample, examples[-1])
    assert len(examples) == VALIDATION_EXAMPLE_COUNT
    assert type(first_window) is WindowExample
    assert first_window.start_sample == 0
    assert type(last_window) is WindowExample
    assert last_window.start_sample == 961_600

    swapped = list(examples)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(Experiment002ValidationError, match="canonical source order"):
        validation._require_registered_validation_population(corpus, tuple(swapped))

    changed_window = list(examples)
    window_index = command_count + VALIDATION_SILENCE_COUNT // 2
    original_window = cast(WindowExample, changed_window[window_index])
    changed_window[window_index] = background.window(original_window.start_sample + 1)
    with pytest.raises(Experiment002ValidationError, match="canonical source order"):
        validation._require_registered_validation_population(
            corpus,
            tuple(changed_window),
        )

    changed_label = list(examples)
    forged_label = copy.copy(cast(CommandExample, changed_label[0]))
    object.__setattr__(forged_label, "label_index", 1)
    changed_label[0] = forged_label
    with pytest.raises(Experiment002ValidationError, match="command metadata"):
        validation._require_registered_validation_population(
            corpus,
            tuple(changed_label),
        )

    pair_swapped_labels = list(examples)
    left_index = 0
    left = cast(CommandExample, pair_swapped_labels[left_index])
    right_index = next(
        index
        for index, example in enumerate(pair_swapped_labels[:command_count])
        if cast(CommandExample, example).label_index != left.label_index
    )
    right = cast(CommandExample, pair_swapped_labels[right_index])
    forged_left = copy.copy(left)
    forged_right = copy.copy(right)
    object.__setattr__(forged_left, "label_index", right.label_index)
    object.__setattr__(forged_right, "label_index", left.label_index)
    pair_swapped_labels[left_index] = forged_left
    pair_swapped_labels[right_index] = forged_right
    assert tuple(
        sorted(example.label_index for example in pair_swapped_labels)
    ) == tuple(sorted(example.label_index for example in examples))
    with pytest.raises(Experiment002ValidationError, match="command metadata"):
        validation._require_registered_validation_population(
            corpus,
            tuple(pair_swapped_labels),
        )

    materialize_calls = 0

    def forbid_materialization(
        *args: object,
        **kwargs: object,
    ) -> MaterializedValidationPopulation:
        nonlocal materialize_calls
        materialize_calls += 1
        raise AssertionError("invalid metadata reached feature materialization")

    monkeypatch.setattr(
        validation,
        "_require_registered_validation_corpus",
        lambda value: None,
    )
    monkeypatch.setattr(
        validation,
        "_require_registered_validation_cache",
        lambda value: None,
    )
    monkeypatch.setattr(
        validation,
        "_registered_normalization_stats",
        lambda value: _identity_stats(),
    )
    monkeypatch.setattr(
        validation,
        "build_validation_population",
        lambda value: tuple(pair_swapped_labels),
    )
    monkeypatch.setattr(
        validation,
        "_materialize_validation_population",
        forbid_materialization,
    )
    with pytest.raises(Experiment002ValidationError, match="command metadata"):
        validation.materialize_registered_validation(
            corpus,
            cast(PCMCacheSplitView, object()),
            cast(VerifiedNormalization, object()),
        )
    assert materialize_calls == 0

    changed_identity = list(examples)
    forged_window = copy.copy(cast(WindowExample, changed_identity[-1]))
    object.__setattr__(forged_window, "identity", b"forged-window-identity")
    changed_identity[-1] = forged_window
    with pytest.raises(Experiment002ValidationError, match="window metadata"):
        validation._require_registered_validation_population(
            corpus,
            tuple(changed_identity),
        )


@pytest.mark.parametrize(
    "payload",
    [b"", b"\0\0\0\0", cast(bytes, "not-bytes")],
)
def test_bad_pcm_payloads_never_return_a_partial_population(
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes,
) -> None:
    source = _command_source(sample_count=1)
    example = _command_example(source)
    fake = _FakeValidationCache({source: payload})
    monkeypatch.setattr(validation, "PCMCacheSplitView", _FakeValidationCache)

    with pytest.raises((TypeError, ValueError)):
        validation._materialize_validation_population(
            (example,),
            cast(PCMCacheSplitView, fake),
            _identity_stats(),
            layout=_tiny_layout(0),
        )
    assert fake.command_reads == [source]


def test_extractor_failures_layout_order_and_result_invariants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _command_source(sample_count=1)
    example = _command_example(source)
    fake = _FakeValidationCache({source: bytes(2)})
    monkeypatch.setattr(validation, "PCMCacheSplitView", _FakeValidationCache)
    failures: tuple[Callable[[FloatArray, NormalizationStats], Any], ...] = (
        lambda waveform, stats: np.zeros((40, 98), dtype=np.float64),
        lambda waveform, stats: np.zeros((98, 40), dtype=np.float32),
        lambda waveform, stats: np.full((40, 98), np.nan, dtype=np.float32),
        lambda waveform, stats: np.zeros((40, 196), dtype=np.float32)[:, ::2],
        lambda waveform, stats: "not-an-array",
    )
    for extractor in failures:
        with pytest.raises((TypeError, Experiment002ValidationError)):
            validation._materialize_validation_population(
                (example,),
                cast(PCMCacheSplitView, fake),
                _identity_stats(),
                layout=_tiny_layout(0),
                model_input_extractor=cast(validation._ModelInputExtractor, extractor),
            )

    class BombLayout:
        def __getattribute__(self, name: str) -> Any:
            raise AssertionError(f"layout attribute accessed: {name}")

    with pytest.raises(TypeError, match="_ValidationLayout"):
        validation._materialize_validation_population(
            (example,),
            cast(PCMCacheSplitView, fake),
            _identity_stats(),
            layout=cast(validation._ValidationLayout, BombLayout()),
        )

    inputs = np.zeros((1, 40, 98), dtype=np.float32)
    labels = np.array([0], dtype=np.int64)
    inputs.setflags(write=False)
    labels.setflags(write=False)
    valid = validation._issue_materialized_validation((example,), inputs, labels)
    assert valid.examples == (example,)

    with pytest.raises(TypeError, match="issued"):
        MaterializedValidationPopulation()

    mismatched = np.array([1], dtype=np.int64)
    mismatched.setflags(write=False)
    with pytest.raises(Experiment002ValidationError, match="example order"):
        validation._issue_materialized_validation((example,), inputs, mismatched)

    writable_inputs = np.zeros((1, 40, 98), dtype=np.float32)
    with pytest.raises(Experiment002ValidationError, match="read-only"):
        validation._issue_materialized_validation((example,), writable_inputs, labels)

    wrong_frames = np.zeros((1, 40, 1), dtype=np.float32)
    wrong_frames.setflags(write=False)
    with pytest.raises(Experiment002ValidationError, match="shape or dtype"):
        validation._issue_materialized_validation((example,), wrong_frames, labels)

    nonfinite = np.zeros((1, 40, 98), dtype=np.float32)
    nonfinite[0, 0, 0] = np.float32(np.inf)
    nonfinite.setflags(write=False)
    with pytest.raises(Experiment002ValidationError, match="non-finite"):
        validation._issue_materialized_validation((example,), nonfinite, labels)


def test_registered_materialization_capability_binds_exact_issuer_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus, examples, model_inputs, label_indices, population = _tiny_issued_population(
        registered=True
    )
    _install_tiny_registered_population_guards(monkeypatch, corpus, examples)

    validation.verify_registered_materialized_validation(corpus, population)
    state = validation._issued_materialized_state(population)
    assert state.examples is examples
    assert state.model_inputs is model_inputs
    assert state.label_indices is label_indices
    assert state.registered_corpus is corpus
    assert state.registered_marker is validation._REGISTERED_MATERIALIZATION_MARKER
    assert (
        state.model_inputs_sha256
        == hashlib.sha256(model_inputs.tobytes(order="C")).hexdigest()
    )
    assert (
        state.label_indices_sha256
        == hashlib.sha256(label_indices.tobytes(order="C")).hexdigest()
    )
    assert population.examples is examples
    assert population.model_inputs is model_inputs
    assert population.label_indices is label_indices

    private = _tiny_issued_population(registered=False)[-1]
    with pytest.raises(Experiment002ValidationError, match="registered corpus"):
        validation.verify_registered_materialized_validation(corpus, private)

    equal_corpus = copy.copy(corpus)
    with pytest.raises(Experiment002ValidationError, match="registered corpus"):
        validation.verify_registered_materialized_validation(equal_corpus, population)


def test_materialized_population_rejects_copies_pickle_and_manual_forges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus, examples, model_inputs, label_indices, population = _tiny_issued_population(
        registered=True
    )
    _install_tiny_registered_population_guards(monkeypatch, corpus, examples)

    clones = (
        copy.copy(population),
        copy.deepcopy(population),
        pickle.loads(pickle.dumps(population)),
    )
    for clone in clones:
        assert clone is not population
        with pytest.raises(Experiment002ValidationError, match="not issued"):
            validation.verify_registered_materialized_validation(corpus, clone)
        with pytest.raises(Experiment002ValidationError, match="not issued"):
            _ = clone.model_inputs

    forged = object.__new__(MaterializedValidationPopulation)
    object.__setattr__(forged, "_examples", examples)
    object.__setattr__(forged, "_model_inputs", model_inputs)
    object.__setattr__(forged, "_label_indices", label_indices)
    object.__setattr__(
        forged,
        "_registered_marker",
        validation._REGISTERED_MATERIALIZATION_MARKER,
    )
    with pytest.raises(Experiment002ValidationError, match="not issued"):
        validation.verify_registered_materialized_validation(corpus, forged)

    class PopulationSubclass(MaterializedValidationPopulation):
        pass

    subclass = object.__new__(PopulationSubclass)
    with pytest.raises(TypeError, match="MaterializedValidationPopulation"):
        validation.verify_registered_materialized_validation(
            corpus,
            cast(MaterializedValidationPopulation, subclass),
        )


def test_registered_materialization_rejects_identity_and_byte_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fresh() -> tuple[
        Experiment002Corpus,
        tuple[CommandExample, ...],
        FloatArray,
        Int64Array,
        MaterializedValidationPopulation,
    ]:
        issued = _tiny_issued_population(registered=True)
        _install_tiny_registered_population_guards(
            monkeypatch,
            issued[0],
            issued[1],
        )
        return issued

    corpus, examples, model_inputs, _, population = fresh()
    replacement_inputs = model_inputs.copy(order="C")
    replacement_inputs.setflags(write=False)
    assert replacement_inputs.tobytes(order="C") == model_inputs.tobytes(order="C")
    object.__setattr__(population, "_model_inputs", replacement_inputs)
    with pytest.raises(Experiment002ValidationError, match="changed"):
        validation.verify_registered_materialized_validation(corpus, population)

    corpus, examples, _, label_indices, population = fresh()
    replacement_labels = label_indices.copy(order="C")
    replacement_labels.setflags(write=False)
    object.__setattr__(population, "_label_indices", replacement_labels)
    with pytest.raises(Experiment002ValidationError, match="changed"):
        validation.verify_registered_materialized_validation(corpus, population)

    corpus, examples, _, _, population = fresh()
    replacement_examples = tuple([*examples])
    assert replacement_examples == examples and replacement_examples is not examples
    object.__setattr__(population, "_examples", replacement_examples)
    with pytest.raises(Experiment002ValidationError, match="changed"):
        validation.verify_registered_materialized_validation(corpus, population)

    corpus, _, _, _, population = fresh()
    object.__setattr__(population, "_registered_marker", object())
    with pytest.raises(Experiment002ValidationError, match="changed"):
        validation.verify_registered_materialized_validation(corpus, population)

    corpus, _, model_inputs, _, population = fresh()
    model_inputs.setflags(write=True)
    model_inputs[0, 0, 0] = np.float32(-0.0)
    model_inputs.setflags(write=False)
    assert model_inputs[0, 0, 0] == np.float32(0.0)
    with pytest.raises(Experiment002ValidationError, match="bytes changed"):
        validation.verify_registered_materialized_validation(corpus, population)

    corpus, _, _, label_indices, population = fresh()
    label_indices.setflags(write=True)
    label_indices[0] = np.int64(1)
    label_indices.setflags(write=False)
    with pytest.raises(Experiment002ValidationError, match="example order"):
        validation.verify_registered_materialized_validation(corpus, population)


def test_registered_materialization_revalidates_layout_finiteness_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus, examples, model_inputs, _, population = _tiny_issued_population(
        registered=True
    )
    _install_tiny_registered_population_guards(monkeypatch, corpus, examples)
    model_inputs.resize((1, 20, 196), refcheck=False)
    with pytest.raises(Experiment002ValidationError, match="shape or dtype"):
        validation.verify_registered_materialized_validation(corpus, population)

    corpus, examples, model_inputs, _, population = _tiny_issued_population(
        registered=True
    )
    _install_tiny_registered_population_guards(monkeypatch, corpus, examples)
    model_inputs.setflags(write=True)
    model_inputs[0, 0, 0] = np.float32(np.nan)
    model_inputs.setflags(write=False)
    with pytest.raises(Experiment002ValidationError, match="non-finite"):
        validation.verify_registered_materialized_validation(corpus, population)

    corpus, examples, model_inputs, _, population = _tiny_issued_population(
        registered=True
    )
    _install_tiny_registered_population_guards(monkeypatch, corpus, examples)
    model_inputs.setflags(write=True)
    with pytest.raises(Experiment002ValidationError, match="read-only"):
        validation.verify_registered_materialized_validation(corpus, population)

    corpus, examples, _, _, population = _tiny_issued_population(registered=True)
    _install_tiny_registered_population_guards(monkeypatch, corpus, examples)
    object.__setattr__(corpus, "manifest_sha256", "0" * 64)
    with pytest.raises(Experiment002ValidationError, match="canonical corpus"):
        validation.verify_registered_materialized_validation(corpus, population)

    corpus, examples, _, _, population = _tiny_issued_population(registered=True)
    _install_tiny_registered_population_guards(monkeypatch, corpus, examples)
    object.__setattr__(examples[0], "identity", b"forged")
    with pytest.raises(Experiment002ValidationError, match="canonical metadata"):
        validation.verify_registered_materialized_validation(corpus, population)

    corpus, examples, _, _, population = _tiny_issued_population(registered=True)
    _install_tiny_registered_population_guards(monkeypatch, corpus, examples)
    equal_source = copy.copy(examples[0].source)
    object.__setattr__(examples[0], "source", equal_source)
    with pytest.raises(Experiment002ValidationError, match="canonical metadata"):
        validation.verify_registered_materialized_validation(corpus, population)
