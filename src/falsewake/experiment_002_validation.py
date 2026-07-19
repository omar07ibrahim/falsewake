"""Materialize the fixed, unaugmented Experiment 002 validation population."""

from __future__ import annotations

import hashlib
import hmac
import threading
import weakref
from dataclasses import dataclass
from typing import Final, Protocol

import numpy as np
from numpy.typing import NDArray

from falsewake.experiment_002_data import (
    CLASS_ORDER,
    VALIDATION_COMMAND_INVENTORY_SHA256,
    VALIDATION_EXAMPLE_COUNT,
    VALIDATION_HOP_SAMPLES,
    VALIDATION_SILENCE_COUNT,
    WINDOW_SAMPLES,
    BackgroundSource,
    CommandExample,
    CommandSource,
    Experiment002Corpus,
    WindowExample,
    _command_inventory_sha256,
    _command_source_order,
    build_validation_population,
)
from falsewake.experiment_002_data import (
    _require_registered_validation_corpus as _require_registered_data_corpus,
)
from falsewake.experiment_002_normalization import (
    NormalizationStats,
    normalize_log_mel,
)
from falsewake.experiment_002_normalization_artifact import (
    NormalizationArtifactIdentity,
    VerifiedNormalization,
    verify_registered_normalization,
)
from falsewake.experiment_002_pcm_cache import (
    REGISTERED_FILE_BYTES,
    Experiment002PCMCache,
    PCMCacheIdentity,
    PCMCacheSplitView,
)
from falsewake.experiment_002_preprocessing import (
    MEL_BINS,
    TIME_FRAMES,
    _decode_padded_command,
    _decode_pcm16le_window,
    _model_layout,
    unaugmented_log_mel,
)
from falsewake.features import FloatArray

Int64Array = NDArray[np.int64]
REGISTERED_PCM_CACHE_SHA256: Final = (
    "b56270a4b99d235c62c162545b139a79b7cf3dc157259a3bbf5678d8a451f653"
)
REGISTERED_NORMALIZATION_BYTES: Final = 320
REGISTERED_NORMALIZATION_SHA256: Final = (
    "891900d4c36fa8a71ba429384f3f4ff594de7c81347ed846c0f790dc4579268e"
)
VALIDATION_CLASS_SUPPORT: Final = (
    397,
    406,
    350,
    377,
    352,
    363,
    363,
    373,
    350,
    372,
    6_278,
    602,
)


class Experiment002ValidationError(ValueError):
    """The fixed validation materialization contract was violated."""


class _ModelInputExtractor(Protocol):
    def __call__(
        self,
        waveform: FloatArray,
        stats: NormalizationStats,
    ) -> FloatArray: ...


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
class MaterializedValidationPopulation:
    """Issuer-only owned arrays in one deterministic example order.

    NumPy's writeable flag is not a security capability: the later source-bound
    trainer remains the trusted sole consumer.  Registered provenance is checked
    separately by :func:`verify_registered_materialized_validation`.
    """

    _examples: tuple[CommandExample | WindowExample, ...]
    _model_inputs: FloatArray
    _label_indices: Int64Array
    _registered_marker: object | None

    def __init__(self) -> None:
        raise TypeError(
            "MaterializedValidationPopulation values are issued by this module"
        )

    @property
    def examples(self) -> tuple[CommandExample | WindowExample, ...]:
        """Return the exact immutable tuple supplied when this value was issued."""

        return _issued_materialized_state(self).examples

    @property
    def model_inputs(self) -> FloatArray:
        """Return the exact owned, read-only model-input array."""

        return _issued_materialized_state(self).model_inputs

    @property
    def label_indices(self) -> Int64Array:
        """Return the exact owned, read-only label array."""

        return _issued_materialized_state(self).label_indices


@dataclass(frozen=True, slots=True)
class _ValidationLayout:
    example_count: int
    frame_count: int
    class_support: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.example_count, int) or isinstance(
            self.example_count, bool
        ):
            raise TypeError("example_count must be an integer")
        if self.example_count < 1:
            raise Experiment002ValidationError("example_count must be positive")
        if not isinstance(self.frame_count, int) or isinstance(self.frame_count, bool):
            raise TypeError("frame_count must be an integer")
        if self.frame_count < 1:
            raise Experiment002ValidationError("frame_count must be positive")
        if type(self.class_support) is not tuple or len(self.class_support) != len(
            CLASS_ORDER
        ):
            raise Experiment002ValidationError("class_support has an invalid shape")
        if any(
            not isinstance(count, int) or isinstance(count, bool) or count < 0
            for count in self.class_support
        ):
            raise Experiment002ValidationError("class_support contains invalid counts")
        if sum(self.class_support) != self.example_count:
            raise Experiment002ValidationError(
                "class_support differs from example_count"
            )


_REGISTERED_LAYOUT: Final = _ValidationLayout(
    example_count=VALIDATION_EXAMPLE_COUNT,
    frame_count=TIME_FRAMES,
    class_support=VALIDATION_CLASS_SUPPORT,
)
_REGISTERED_MATERIALIZATION_MARKER: Final = object()


@dataclass(frozen=True, slots=True)
class _IssuedMaterializedValidationState:
    examples: tuple[CommandExample | WindowExample, ...]
    model_inputs: FloatArray
    label_indices: Int64Array
    model_inputs_sha256: str
    label_indices_sha256: str
    registered_corpus: Experiment002Corpus | None
    registered_marker: object | None


_ISSUED_MATERIALIZATIONS: weakref.WeakKeyDictionary[
    MaterializedValidationPopulation, _IssuedMaterializedValidationState
] = weakref.WeakKeyDictionary()
_ISSUED_MATERIALIZATIONS_LOCK = threading.Lock()


def materialize_registered_validation(
    corpus: Experiment002Corpus,
    cache: PCMCacheSplitView,
    normalization: VerifiedNormalization,
) -> MaterializedValidationPopulation:
    """Materialize all 10,583 registered validation examples without RNG."""

    _require_registered_validation_corpus(corpus)
    _require_registered_validation_cache(cache)
    stats = _registered_normalization_stats(normalization)
    examples = build_validation_population(corpus)
    _require_registered_validation_population(corpus, examples)
    return _materialize_validation_population(
        examples,
        cache,
        stats,
        layout=_REGISTERED_LAYOUT,
        registered_corpus=corpus,
        registered_marker=_REGISTERED_MATERIALIZATION_MARKER,
    )


def _materialize_validation_population(
    examples: tuple[CommandExample | WindowExample, ...],
    cache: PCMCacheSplitView,
    stats: NormalizationStats,
    *,
    layout: _ValidationLayout,
    model_input_extractor: _ModelInputExtractor | None = None,
    registered_corpus: Experiment002Corpus | None = None,
    registered_marker: object | None = None,
) -> MaterializedValidationPopulation:
    """Private small-population hook with the production numeric path."""

    if type(layout) is not _ValidationLayout:
        raise TypeError("layout must be a _ValidationLayout")
    if layout.frame_count != TIME_FRAMES:
        raise Experiment002ValidationError(
            "a public materialized population requires 98 frames"
        )
    model_inputs, label_indices = _materialize_validation_arrays(
        examples,
        cache,
        stats,
        layout=layout,
        model_input_extractor=model_input_extractor,
    )
    if registered_marker is None:
        if registered_corpus is not None:
            raise Experiment002ValidationError(
                "an unregistered materialization cannot bind a corpus"
            )
    elif registered_marker is not _REGISTERED_MATERIALIZATION_MARKER:
        raise Experiment002ValidationError(
            "materialization received an invalid registered-route marker"
        )
    elif type(registered_corpus) is not Experiment002Corpus:
        raise TypeError("registered_corpus must be an Experiment002Corpus")
    return _issue_materialized_validation(
        examples,
        model_inputs,
        label_indices,
        registered_corpus=registered_corpus,
        registered_marker=registered_marker,
    )


def verify_registered_materialized_validation(
    corpus: Experiment002Corpus,
    population: MaterializedValidationPopulation,
) -> None:
    """Verify exact process issuance, corpus provenance, and materialized bytes."""

    if type(corpus) is not Experiment002Corpus:
        raise TypeError("corpus must be an Experiment002Corpus")
    state = _issued_materialized_state(population)
    if state.registered_marker is not _REGISTERED_MATERIALIZATION_MARKER or (
        state.registered_corpus is not corpus
    ):
        raise Experiment002ValidationError(
            "materialized validation was not issued for this registered corpus"
        )

    first_model_inputs_sha256 = _array_sha256(state.model_inputs)
    first_label_indices_sha256 = _array_sha256(state.label_indices)
    _require_registered_validation_corpus(corpus)
    _validate_materialized_values(
        state.examples,
        state.model_inputs,
        state.label_indices,
    )
    _require_registered_validation_population(corpus, state.examples)
    _require_registered_validation_corpus(corpus)
    _validate_materialized_values(
        state.examples,
        state.model_inputs,
        state.label_indices,
    )
    second_model_inputs_sha256 = _array_sha256(state.model_inputs)
    second_label_indices_sha256 = _array_sha256(state.label_indices)
    if (
        not hmac.compare_digest(
            first_model_inputs_sha256,
            second_model_inputs_sha256,
        )
        or not hmac.compare_digest(
            second_model_inputs_sha256,
            state.model_inputs_sha256,
        )
        or not hmac.compare_digest(
            first_label_indices_sha256,
            second_label_indices_sha256,
        )
        or not hmac.compare_digest(
            second_label_indices_sha256,
            state.label_indices_sha256,
        )
    ):
        raise Experiment002ValidationError(
            "materialized validation bytes changed after issuance"
        )
    if _issued_materialized_state(population) is not state:
        raise Experiment002ValidationError(
            "materialized validation issuance changed during verification"
        )


def _issue_materialized_validation(
    examples: tuple[CommandExample | WindowExample, ...],
    model_inputs: FloatArray,
    label_indices: Int64Array,
    *,
    registered_corpus: Experiment002Corpus | None = None,
    registered_marker: object | None = None,
) -> MaterializedValidationPopulation:
    if registered_marker is None:
        if registered_corpus is not None:
            raise Experiment002ValidationError(
                "an unregistered materialization cannot bind a corpus"
            )
    elif registered_marker is not _REGISTERED_MATERIALIZATION_MARKER:
        raise Experiment002ValidationError(
            "materialization received an invalid registered-route marker"
        )
    elif type(registered_corpus) is not Experiment002Corpus:
        raise TypeError("registered_corpus must be an Experiment002Corpus")
    _validate_materialized_values(examples, model_inputs, label_indices)
    model_inputs_sha256 = _array_sha256(model_inputs)
    label_indices_sha256 = _array_sha256(label_indices)
    _validate_materialized_values(examples, model_inputs, label_indices)

    result = object.__new__(MaterializedValidationPopulation)
    object.__setattr__(result, "_examples", examples)
    object.__setattr__(result, "_model_inputs", model_inputs)
    object.__setattr__(result, "_label_indices", label_indices)
    object.__setattr__(result, "_registered_marker", registered_marker)
    state = _IssuedMaterializedValidationState(
        examples=examples,
        model_inputs=model_inputs,
        label_indices=label_indices,
        model_inputs_sha256=model_inputs_sha256,
        label_indices_sha256=label_indices_sha256,
        registered_corpus=registered_corpus,
        registered_marker=registered_marker,
    )
    with _ISSUED_MATERIALIZATIONS_LOCK:
        _ISSUED_MATERIALIZATIONS[result] = state
    return result


def _issued_materialized_state(
    population: MaterializedValidationPopulation,
) -> _IssuedMaterializedValidationState:
    if type(population) is not MaterializedValidationPopulation:
        raise TypeError("population must be a MaterializedValidationPopulation")
    with _ISSUED_MATERIALIZATIONS_LOCK:
        state = _ISSUED_MATERIALIZATIONS.get(population)
    if state is None:
        raise Experiment002ValidationError(
            "materialized validation was not issued by this process"
        )
    try:
        raw_examples = population._examples
        raw_model_inputs = population._model_inputs
        raw_label_indices = population._label_indices
        raw_registered_marker = population._registered_marker
    except AttributeError as error:
        raise Experiment002ValidationError(
            "materialized validation capability is incomplete"
        ) from error
    if (
        raw_examples is not state.examples
        or raw_model_inputs is not state.model_inputs
        or raw_label_indices is not state.label_indices
        or raw_registered_marker is not state.registered_marker
    ):
        raise Experiment002ValidationError(
            "materialized validation capability changed after issuance"
        )
    return state


def _validate_materialized_values(
    examples: tuple[CommandExample | WindowExample, ...],
    model_inputs: FloatArray,
    label_indices: Int64Array,
) -> None:
    if type(examples) is not tuple:
        raise TypeError("examples must be a tuple")
    if any(
        type(example) not in {CommandExample, WindowExample} for example in examples
    ):
        raise TypeError("examples contain an invalid example type")
    if type(model_inputs) is not np.ndarray:
        raise TypeError("model_inputs must be a NumPy array")
    if type(label_indices) is not np.ndarray:
        raise TypeError("label_indices must be a NumPy array")

    example_count = len(examples)
    if model_inputs.dtype != np.dtype(np.float32) or (
        model_inputs.shape != (example_count, MEL_BINS, TIME_FRAMES)
    ):
        raise Experiment002ValidationError(
            "model_inputs have an invalid shape or dtype"
        )
    if not model_inputs.flags.c_contiguous or not model_inputs.flags.owndata:
        raise Experiment002ValidationError(
            "model_inputs must be owned and C-contiguous"
        )
    if model_inputs.flags.writeable:
        raise Experiment002ValidationError("model_inputs must be read-only")
    if not np.all(np.isfinite(model_inputs)):
        raise Experiment002ValidationError("model_inputs contain non-finite values")

    if label_indices.dtype != np.dtype(np.int64) or label_indices.shape != (
        example_count,
    ):
        raise Experiment002ValidationError(
            "label_indices have an invalid shape or dtype"
        )
    if not label_indices.flags.c_contiguous or not label_indices.flags.owndata:
        raise Experiment002ValidationError(
            "label_indices must be owned and C-contiguous"
        )
    if label_indices.flags.writeable:
        raise Experiment002ValidationError("label_indices must be read-only")
    if example_count and (
        np.any(label_indices < 0) or np.any(label_indices >= len(CLASS_ORDER))
    ):
        raise Experiment002ValidationError("label_indices are outside class order")
    if any(
        int(label_indices[index]) != example.label_index
        for index, example in enumerate(examples)
    ):
        raise Experiment002ValidationError(
            "label_indices differ from the example order"
        )


def _array_sha256(values: np.ndarray[tuple[int, ...], np.dtype[np.generic]]) -> str:
    if type(values) is not np.ndarray:
        raise TypeError("values must be a NumPy array")
    if not values.flags.c_contiguous:
        raise Experiment002ValidationError("values must be C-contiguous")
    return hashlib.sha256(memoryview(values).cast("B")).hexdigest()


def _materialize_validation_arrays(
    examples: tuple[CommandExample | WindowExample, ...],
    cache: PCMCacheSplitView,
    stats: NormalizationStats,
    *,
    layout: _ValidationLayout,
    model_input_extractor: _ModelInputExtractor | None = None,
) -> tuple[FloatArray, Int64Array]:
    """Materialize a private registered or low-memory test layout."""

    if type(layout) is not _ValidationLayout:
        raise TypeError("layout must be a _ValidationLayout")
    if type(examples) is not tuple:
        raise TypeError("examples must be a tuple")
    if len(examples) != layout.example_count:
        raise Experiment002ValidationError(
            "validation population count differs from the layout"
        )
    if type(cache) is not PCMCacheSplitView:
        raise TypeError("cache must be a PCMCacheSplitView")
    if cache.split != "validation":
        raise Experiment002ValidationError(
            "validation materialization requires the validation cache view"
        )
    if type(stats) is not NormalizationStats:
        raise TypeError("stats must be NormalizationStats")
    extractor = (
        _registered_model_input
        if model_input_extractor is None
        else model_input_extractor
    )
    if not callable(extractor):
        raise TypeError("model_input_extractor must be callable")

    model_inputs = np.empty(
        (layout.example_count, MEL_BINS, layout.frame_count),
        dtype=np.float32,
        order="C",
    )
    label_indices = np.empty(layout.example_count, dtype=np.int64, order="C")

    for index, example in enumerate(examples):
        if type(example) is CommandExample:
            payload = cache.read_command_pcm16le(example.source)
            waveform = _decode_padded_command(
                payload,
                sample_count=example.source.sample_count,
            )
        elif type(example) is WindowExample:
            payload = cache.read_background_window_pcm16le(
                example.source,
                example.start_sample,
            )
            waveform = _decode_pcm16le_window(payload)
        else:
            raise TypeError("examples contain an invalid example type")

        model_input = extractor(waveform, stats)
        if type(model_input) is not np.ndarray:
            raise TypeError("model input extractor must return a NumPy array")
        if model_input.dtype != np.dtype(np.float32) or model_input.shape != (
            MEL_BINS,
            layout.frame_count,
        ):
            raise Experiment002ValidationError(
                "model input extractor returned an invalid shape or dtype"
            )
        if not model_input.flags.c_contiguous or not np.all(np.isfinite(model_input)):
            raise Experiment002ValidationError(
                "model input extractor returned an invalid matrix"
            )
        model_inputs[index] = model_input
        label_indices[index] = np.int64(example.label_index)

    model_inputs.setflags(write=False)
    label_indices.setflags(write=False)
    observed_support = tuple(
        int(value)
        for value in np.bincount(
            label_indices,
            minlength=len(CLASS_ORDER),
        )
    )
    if observed_support != layout.class_support:
        raise Experiment002ValidationError(
            "validation class support differs from the layout"
        )
    return model_inputs, label_indices


def _registered_model_input(
    waveform: FloatArray,
    stats: NormalizationStats,
) -> FloatArray:
    normalized = normalize_log_mel(unaugmented_log_mel(waveform), stats)
    return _model_layout(normalized)


def _require_registered_validation_corpus(corpus: Experiment002Corpus) -> None:
    if type(corpus) is not Experiment002Corpus:
        raise TypeError("corpus must be an Experiment002Corpus")
    _require_registered_data_corpus(corpus)
    if type(corpus.validation_commands) is not tuple or any(
        type(source) is not CommandSource for source in corpus.validation_commands
    ):
        raise Experiment002ValidationError(
            "validation command sources have invalid concrete types"
        )
    observed_inventory = _command_inventory_sha256(corpus.validation_commands)
    if not hmac.compare_digest(
        observed_inventory,
        VALIDATION_COMMAND_INVENTORY_SHA256,
    ):
        raise Experiment002ValidationError(
            "validation command inventory differs from the registered bytes"
        )
    if type(corpus.validation_background) is not BackgroundSource:
        raise Experiment002ValidationError(
            "validation background has an invalid concrete type"
        )


def _require_registered_validation_cache(cache: PCMCacheSplitView) -> None:
    """Check the opened cache at the registered source-bound process boundary.

    The cache module predates issuer registries.  These checks therefore reject
    accidental substitution and subclasses; the later committed run launcher is
    responsible for excluding adversarial co-resident Python code that could
    mutate private fields after ``Experiment002PCMCache.open`` verifies the file.
    """

    if type(cache) is not PCMCacheSplitView:
        raise TypeError("cache must be a PCMCacheSplitView")
    if type(cache._split) is not str or cache._split != "validation":
        raise Experiment002ValidationError(
            "validation materialization requires the validation cache view"
        )
    owner = cache._cache
    if type(owner) is not Experiment002PCMCache:
        raise Experiment002ValidationError(
            "validation cache view has an invalid concrete owner"
        )
    if owner.closed:
        raise Experiment002ValidationError("validation cache is closed")
    identity = owner.identity
    if type(identity) is not PCMCacheIdentity:
        raise Experiment002ValidationError("validation cache identity is invalid")
    if identity.byte_count != REGISTERED_FILE_BYTES or not hmac.compare_digest(
        identity.sha256,
        REGISTERED_PCM_CACHE_SHA256,
    ):
        raise Experiment002ValidationError(
            "validation cache differs from the frozen report identity"
        )


def _registered_normalization_stats(
    normalization: VerifiedNormalization,
) -> NormalizationStats:
    if type(normalization) is not VerifiedNormalization:
        raise TypeError("normalization must be a VerifiedNormalization")
    verify_registered_normalization(normalization)
    identity = normalization.identity
    if type(identity) is not NormalizationArtifactIdentity:
        raise Experiment002ValidationError("normalization identity is invalid")
    if identity.byte_count != REGISTERED_NORMALIZATION_BYTES or not (
        hmac.compare_digest(identity.sha256, REGISTERED_NORMALIZATION_SHA256)
    ):
        raise Experiment002ValidationError(
            "normalization differs from the frozen report identity"
        )
    return normalization.stats


def _require_registered_validation_population(
    corpus: Experiment002Corpus,
    examples: tuple[CommandExample | WindowExample, ...],
) -> None:
    if type(examples) is not tuple or len(examples) != VALIDATION_EXAMPLE_COUNT:
        raise Experiment002ValidationError(
            "validation population has an invalid concrete type or count"
        )
    command_count = VALIDATION_EXAMPLE_COUNT - VALIDATION_SILENCE_COUNT
    expected_commands = tuple(
        sorted(corpus.validation_commands, key=_command_source_order)
    )
    command_examples = examples[:command_count]
    if len(expected_commands) != command_count:
        raise Experiment002ValidationError(
            "validation command count differs from the registered population"
        )
    for example, source in zip(command_examples, expected_commands, strict=True):
        if (
            type(example) is not CommandExample
            or type(example.source) is not CommandSource
            or example.source is not source
        ):
            raise Experiment002ValidationError(
                "validation commands differ from canonical source order"
            )
        if (
            type(example.label) is not str
            or example.label != source.label
            or type(example.label_index) is not int
            or example.label_index != CLASS_ORDER.index(source.label)
            or type(example.identity) is not bytes
            or not hmac.compare_digest(example.identity, source.identity)
        ):
            raise Experiment002ValidationError(
                "validation command metadata differs from its canonical source"
            )

    windows = examples[command_count:]
    expected_starts = tuple(
        range(
            0,
            corpus.validation_background.sample_count - WINDOW_SAMPLES + 1,
            VALIDATION_HOP_SAMPLES,
        )
    )
    if len(expected_starts) != VALIDATION_SILENCE_COUNT:
        raise Experiment002ValidationError(
            "validation window count differs from the registered population"
        )
    for example, start_sample in zip(windows, expected_starts, strict=True):
        if (
            type(example) is not WindowExample
            or type(example.source) is not BackgroundSource
            or example.source is not corpus.validation_background
            or type(example.start_sample) is not int
            or example.start_sample != start_sample
        ):
            raise Experiment002ValidationError(
                "validation windows differ from canonical source order"
            )
        expected_window = corpus.validation_background.window(start_sample)
        if (
            type(example.label) is not str
            or example.label != expected_window.label
            or type(example.label_index) is not int
            or example.label_index != expected_window.label_index
            or type(example.identity) is not bytes
            or not hmac.compare_digest(example.identity, expected_window.identity)
        ):
            raise Experiment002ValidationError(
                "validation window metadata differs from its canonical source"
            )

    observed_support = [0] * len(CLASS_ORDER)
    for example in examples:
        if type(example) not in {CommandExample, WindowExample}:
            raise TypeError("examples contain an invalid example type")
        if (
            not isinstance(example.label_index, int)
            or isinstance(example.label_index, bool)
            or not 0 <= example.label_index < len(CLASS_ORDER)
        ):
            raise Experiment002ValidationError(
                "validation example has an invalid label index"
            )
        observed_support[example.label_index] += 1
    if tuple(observed_support) != VALIDATION_CLASS_SUPPORT:
        raise Experiment002ValidationError(
            "validation class support differs from the registered population"
        )
