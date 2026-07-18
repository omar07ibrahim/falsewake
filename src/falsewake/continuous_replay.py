"""Portable scoring and event primitives for continuous replay."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import stat
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2

from falsewake.features import FloatArray, extract_clip_features
from falsewake.speech_commands import TARGET_WORDS

MODEL_SHA256 = "d5ac3579f539288c6ed7769894dd30ad2272480956b2f9b24b7e730baaeb0cd3"
MAX_MODEL_BYTES = 1024 * 1024
CLIP_SAMPLES = 16_000
WINDOW_HOP_SAMPLES = 1_600
REFRACTORY_SAMPLES = 16_000
THRESHOLD_COUNT = 1_001
CLASS_ORDER = tuple(sorted((*TARGET_WORDS, "unknown", "silence")))
SAMPLES_PER_HOUR = 57_600_000
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_718

Float64Array = NDArray[np.float64]
Int64Array = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

_MODEL_KEYS = {
    "classifier",
    "examples_sha256",
    "experiment_config_sha256",
    "features_sha256",
    "input",
    "manifest_sha256",
    "scaler",
    "schema_version",
    "scikit_learn_version",
}
_IDENTITY_KEYS = (
    "examples_sha256",
    "experiment_config_sha256",
    "features_sha256",
    "manifest_sha256",
)
_THRESHOLDS = np.arange(THRESHOLD_COUNT, dtype=np.float64) / np.float64(1_000)
_THRESHOLDS.flags.writeable = False
_TARGET_POSITION = {word: index for index, word in enumerate(TARGET_WORDS)}
_INT64_MAX = int(np.iinfo(np.int64).max)
_MAX_BOOTSTRAP_MATRIX_ELEMENTS = 10_000_000


class ContinuousReplayError(ValueError):
    """A model, waveform, score, or event stream violates experiment 001."""


@dataclass(frozen=True, slots=True)
class PortableLinearModel:
    classes: tuple[str, ...]
    mean: Float64Array
    scale: Float64Array
    coef: Float64Array
    intercept: Float64Array


@dataclass(frozen=True, slots=True)
class ScoreBatch:
    logits: Float64Array
    probabilities: Float64Array
    predicted_indices: Int64Array


@dataclass(frozen=True, slots=True)
class EventGrid:
    counts: Int64Array
    counts_by_target: Int64Array


@dataclass(frozen=True, slots=True)
class DetectedEvent:
    start_sample: int
    predicted_target: str
    target_probability: float


@dataclass(frozen=True, slots=True)
class UtteranceEventGrid:
    """One utterance's registered event grid and negative exposure."""

    speaker_id: int
    scored_exposure_samples: int
    events: EventGrid

    def __post_init__(self) -> None:
        speaker_id = _require_integer(self.speaker_id, name="speaker_id", minimum=0)
        exposure = _require_exposure_samples(
            self.scored_exposure_samples,
            name="scored_exposure_samples",
            allow_zero=True,
        )
        events = _validated_event_grid(self.events, exposure=exposure)
        object.__setattr__(self, "speaker_id", speaker_id)
        object.__setattr__(self, "scored_exposure_samples", exposure)
        object.__setattr__(self, "events", events)


@dataclass(frozen=True, slots=True)
class AggregatedEventGrid:
    """Integer event sufficient statistics at corpus and speaker grain."""

    speaker_ids: Int64Array
    utterance_counts_by_speaker: Int64Array
    scored_exposure_samples_by_speaker: Int64Array
    counts_by_speaker: Int64Array
    counts_by_speaker_and_target: Int64Array
    scored_exposure_samples: int
    counts: Int64Array
    counts_by_target: Int64Array

    def __post_init__(self) -> None:
        _validate_aggregated_event_grid(self)


@dataclass(frozen=True, slots=True)
class CorrectAcceptGrid:
    """Positive-validation sufficient statistics on the registered grid."""

    clip_counts_by_target: Int64Array
    counts: Int64Array
    counts_by_target: Int64Array

    def __post_init__(self) -> None:
        _validate_correct_accept_grid(self)

    @property
    def baseline_correct_count(self) -> int:
        """Return the exact threshold-zero conditional-retention denominator."""

        return int(self.counts[0])

    @property
    def clip_count(self) -> int:
        """Return the target-validation denominator for absolute recall."""

        return sum(int(value) for value in self.clip_counts_by_target)


@dataclass(frozen=True, slots=True)
class ThresholdSelection:
    """Exact integer-gate outcome for all registered thresholds."""

    status: Literal["pass", "reject"]
    selected_threshold_milli: int | None
    false_event_gate: BoolArray
    retention_gate: BoolArray
    passing: BoolArray
    retention_frontier_milli: int
    negative_frontier_milli: int | None

    def __post_init__(self) -> None:
        _validate_threshold_selection(self)


@dataclass(frozen=True, slots=True)
class EventRateIntervals:
    """Point rates and registered descriptive 95% interval grids."""

    point: Float64Array
    garwood_lower: Float64Array
    garwood_upper: Float64Array
    speaker_bootstrap_lower: Float64Array
    speaker_bootstrap_upper: Float64Array

    def __post_init__(self) -> None:
        _validate_event_rate_intervals(self)


def _reject_json_constant(value: str) -> object:
    raise ContinuousReplayError(f"portable model contains invalid JSON value: {value}")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ContinuousReplayError(f"portable model repeats JSON key: {key!r}")
        result[key] = value
    return result


def _require_sha256(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise ContinuousReplayError(f"{name} must be a SHA-256 string")
    digest = value
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ContinuousReplayError(f"{name} is not a lowercase SHA-256 digest")
    return digest


def _require_object(value: object, *, name: str, keys: set[str]) -> dict[str, object]:
    if type(value) is not dict:
        raise ContinuousReplayError(f"{name} must be a JSON object")
    result = cast(dict[str, object], value)
    if set(result) != keys:
        raise ContinuousReplayError(f"{name} has unexpected or missing fields")
    return result


def _all_json_numbers(value: object) -> bool:
    if type(value) is list:
        return all(_all_json_numbers(item) for item in cast(list[object], value))
    return type(value) in {int, float}


def _float_array(value: object, *, name: str, shape: tuple[int, ...]) -> Float64Array:
    if not _all_json_numbers(value):
        raise ContinuousReplayError(f"{name} must contain only JSON numbers")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ContinuousReplayError(f"{name} cannot be converted to float64") from error
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ContinuousReplayError(f"{name} has an invalid shape or non-finite value")
    result = np.ascontiguousarray(result, dtype=np.float64)
    result.flags.writeable = False
    return result


def parse_portable_model(contents: bytes) -> PortableLinearModel:
    """Parse the exact experiment 000 portable model schema."""

    if len(contents) > MAX_MODEL_BYTES:
        raise ContinuousReplayError("portable model exceeds the size limit")
    try:
        document = json.loads(
            contents,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ContinuousReplayError(f"cannot decode portable model: {error}") from error
    model = _require_object(document, name="portable model", keys=_MODEL_KEYS)
    if type(model["schema_version"]) is not int or model["schema_version"] != 1:
        raise ContinuousReplayError("portable model schema_version must be integer one")
    if model["scikit_learn_version"] != "1.9.0":
        raise ContinuousReplayError("portable model scikit-learn version differs")
    for key in _IDENTITY_KEYS:
        _require_sha256(model[key], name=key)

    input_contract = _require_object(
        model["input"],
        name="portable model input",
        keys={"dtype", "feature_count", "formula"},
    )
    if input_contract != {
        "dtype": "float64",
        "feature_count": 80,
        "formula": "((x - scaler.mean) / scaler.scale) @ coef.T + intercept",
    }:
        raise ContinuousReplayError("portable model input contract differs")

    classifier = _require_object(
        model["classifier"],
        name="portable model classifier",
        keys={"classes", "coef", "intercept", "n_iter"},
    )
    raw_classes = classifier["classes"]
    if type(raw_classes) is not list or any(
        type(label) is not str for label in cast(list[object], raw_classes)
    ):
        raise ContinuousReplayError("portable model classes must be strings")
    classes = tuple(cast(list[str], raw_classes))
    if classes != CLASS_ORDER:
        raise ContinuousReplayError("portable model class order differs")
    n_iter = classifier["n_iter"]
    if (
        type(n_iter) is not list
        or len(cast(list[object], n_iter)) != 1
        or type(cast(list[object], n_iter)[0]) is not int
        or cast(list[int], n_iter)[0] < 1
    ):
        raise ContinuousReplayError("portable model n_iter is invalid")

    scaler = _require_object(
        model["scaler"],
        name="portable model scaler",
        keys={"mean", "scale"},
    )
    mean = _float_array(scaler["mean"], name="scaler mean", shape=(80,))
    scale = _float_array(scaler["scale"], name="scaler scale", shape=(80,))
    if np.any(scale <= 0):
        raise ContinuousReplayError("portable model scaler scale must be positive")
    coef = _float_array(classifier["coef"], name="classifier coef", shape=(12, 80))
    intercept = _float_array(
        classifier["intercept"], name="classifier intercept", shape=(12,)
    )
    return PortableLinearModel(
        classes=classes,
        mean=mean,
        scale=scale,
        coef=coef,
        intercept=intercept,
    )


def load_portable_model(
    path: Path, *, expected_sha256: str = MODEL_SHA256
) -> PortableLinearModel:
    """Load a regular non-symlink model only when its exact bytes match."""

    _require_sha256(expected_sha256, name="expected_sha256")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ContinuousReplayError(f"cannot open portable model: {error}") from error
    try:
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_MODEL_BYTES:
                raise ContinuousReplayError(
                    "portable model must be a bounded regular non-symlink file"
                )
            snapshot = (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            contents = stream.read(MAX_MODEL_BYTES + 1)
            stream.seek(0)
            repeated_contents = stream.read(MAX_MODEL_BYTES + 1)
            after = os.fstat(stream.fileno())
    except OSError as error:
        raise ContinuousReplayError(f"cannot read portable model: {error}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    final_snapshot = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if (
        len(contents) != before.st_size
        or contents != repeated_contents
        or snapshot != final_snapshot
    ):
        raise ContinuousReplayError("portable model changed while reading")
    observed = hashlib.sha256(contents).hexdigest()
    if not hmac.compare_digest(observed, expected_sha256):
        raise ContinuousReplayError(
            f"portable model SHA-256 differs: observed={observed}"
        )
    return parse_portable_model(contents)


def score_feature_rows(model: PortableLinearModel, features: FloatArray) -> ScoreBatch:
    """Apply the frozen scaler, linear logits, and stable float64 softmax."""

    if features.dtype != np.dtype(np.float32) or features.ndim != 2:
        raise ContinuousReplayError("features must be a two-dimensional float32 array")
    if features.shape[1:] != (80,) or not np.all(np.isfinite(features)):
        raise ContinuousReplayError(
            "features have an invalid shape or non-finite value"
        )
    selected = np.ascontiguousarray(features, dtype=np.float64)
    scaled = (selected - model.mean) / model.scale
    logits = scaled @ model.coef.T + model.intercept
    if not np.all(np.isfinite(logits)):
        raise ContinuousReplayError("portable model produced non-finite logits")
    if logits.shape[0] == 0:
        probabilities = np.empty((0, len(model.classes)), dtype=np.float64)
        predicted = np.empty(0, dtype=np.int64)
    else:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        probabilities = exponentials / np.sum(exponentials, axis=1, keepdims=True)
        predicted = np.argmax(logits, axis=1).astype(np.int64, copy=False)
    logits = np.ascontiguousarray(logits, dtype=np.float64)
    probabilities = np.ascontiguousarray(probabilities, dtype=np.float64)
    predicted = np.ascontiguousarray(predicted, dtype=np.int64)
    for values in (logits, probabilities, predicted):
        values.flags.writeable = False
    return ScoreBatch(
        logits=logits,
        probabilities=probabilities,
        predicted_indices=predicted,
    )


def window_count(sample_count: int) -> int:
    """Return the registered number of complete one-second windows."""

    if type(sample_count) is not int or sample_count < 0:
        raise ContinuousReplayError("sample_count must be a non-negative integer")
    if sample_count < CLIP_SAMPLES:
        return 0
    return 1 + (sample_count - CLIP_SAMPLES) // WINDOW_HOP_SAMPLES


def scored_exposure_samples(sample_count: int) -> int:
    """Return the union-span exposure of all registered windows."""

    count = window_count(sample_count)
    if count == 0:
        return 0
    return CLIP_SAMPLES + (count - 1) * WINDOW_HOP_SAMPLES


def score_utterance(
    model: PortableLinearModel, waveform: FloatArray
) -> tuple[Int64Array, ScoreBatch]:
    """Score every full window of one normalized mono utterance."""

    if waveform.dtype != np.dtype(np.float32) or waveform.ndim != 1:
        raise ContinuousReplayError("waveform must be one-dimensional float32")
    if not np.all(np.isfinite(waveform)) or (
        waveform.size > 0 and np.max(np.abs(waveform.astype(np.float64))) > 1.0
    ):
        raise ContinuousReplayError("waveform has invalid normalized samples")
    count = window_count(int(waveform.size))
    starts = np.arange(count, dtype=np.int64) * WINDOW_HOP_SAMPLES
    features = np.empty((count, 80), dtype=np.float32)
    for index, start in enumerate(starts.tolist()):
        features[index] = extract_clip_features(waveform[start : start + CLIP_SAMPLES])
    starts.flags.writeable = False
    return starts, score_feature_rows(model, features)


def _score_vectors(
    model: PortableLinearModel,
    scores: ScoreBatch,
) -> tuple[Int64Array, Float64Array]:
    row_count = scores.predicted_indices.size
    if (
        scores.logits.dtype != np.dtype(np.float64)
        or scores.probabilities.dtype != np.dtype(np.float64)
        or scores.predicted_indices.dtype != np.dtype(np.int64)
        or scores.logits.ndim != 2
        or scores.probabilities.ndim != 2
        or scores.predicted_indices.ndim != 1
        or scores.logits.shape != (row_count, len(model.classes))
        or scores.probabilities.shape != (row_count, len(model.classes))
        or np.any(scores.predicted_indices < 0)
        or np.any(scores.predicted_indices >= len(model.classes))
        or not np.all(np.isfinite(scores.logits))
        or not np.all(np.isfinite(scores.probabilities))
        or np.any(scores.probabilities < 0)
        or np.any(scores.probabilities > 1)
    ):
        raise ContinuousReplayError("score batch has invalid arrays")
    if row_count == 0:
        expected_probabilities = np.empty_like(scores.probabilities)
        expected_predictions = np.empty(0, dtype=np.int64)
    else:
        shifted = scores.logits - np.max(scores.logits, axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        expected_probabilities = exponentials / np.sum(
            exponentials, axis=1, keepdims=True
        )
        expected_predictions = np.argmax(scores.logits, axis=1).astype(
            np.int64, copy=False
        )
    if not np.array_equal(
        scores.predicted_indices, expected_predictions
    ) or not np.allclose(
        scores.probabilities,
        expected_probabilities,
        rtol=0,
        atol=4 * np.finfo(np.float64).eps,
    ):
        raise ContinuousReplayError(
            "score batch differs from stable softmax or logit argmax"
        )
    rows = np.arange(row_count, dtype=np.int64)
    probabilities = scores.probabilities[rows, scores.predicted_indices]
    return scores.predicted_indices, probabilities


def _validate_window_starts(starts: Int64Array, row_count: int) -> None:
    if starts.dtype != np.dtype(np.int64) or starts.ndim != 1:
        raise ContinuousReplayError(
            "window starts must be a one-dimensional int64 array"
        )
    expected = np.arange(row_count, dtype=np.int64) * WINDOW_HOP_SAMPLES
    if not np.array_equal(starts, expected):
        raise ContinuousReplayError(
            "window starts are not the complete registered grid"
        )


def replay_threshold_grid(
    model: PortableLinearModel,
    starts: Int64Array,
    scores: ScoreBatch,
) -> EventGrid:
    """Count events with independent refractory state at every threshold."""

    _validate_window_starts(starts, scores.predicted_indices.size)
    predicted, probabilities = _score_vectors(model, scores)
    counts = np.zeros(THRESHOLD_COUNT, dtype=np.int64)
    by_target = np.zeros((THRESHOLD_COUNT, len(TARGET_WORDS)), dtype=np.int64)
    next_allowed = np.zeros(THRESHOLD_COUNT, dtype=np.int64)

    for row, start in enumerate(starts.tolist()):
        label = model.classes[int(predicted[row])]
        target_position = _TARGET_POSITION.get(label)
        if target_position is None:
            continue
        emitted = (probabilities[row] >= _THRESHOLDS) & (start >= next_allowed)
        counts[emitted] += 1
        by_target[emitted, target_position] += 1
        next_allowed[emitted] = start + REFRACTORY_SAMPLES
    counts.flags.writeable = False
    by_target.flags.writeable = False
    return EventGrid(counts=counts, counts_by_target=by_target)


def events_at_threshold(
    model: PortableLinearModel,
    starts: Int64Array,
    scores: ScoreBatch,
    threshold_milli: int,
) -> tuple[DetectedEvent, ...]:
    """Return the exact event sequence at one registered integer threshold."""

    if type(threshold_milli) is not int or not 0 <= threshold_milli < THRESHOLD_COUNT:
        raise ContinuousReplayError("threshold_milli must be an integer from 0 to 1000")
    _validate_window_starts(starts, scores.predicted_indices.size)
    predicted, probabilities = _score_vectors(model, scores)
    threshold = np.float64(threshold_milli) / np.float64(1_000)
    next_allowed = 0
    events: list[DetectedEvent] = []
    for row, start in enumerate(starts.tolist()):
        label = model.classes[int(predicted[row])]
        probability = float(probabilities[row])
        if (
            label in _TARGET_POSITION
            and probability >= threshold
            and start >= next_allowed
        ):
            events.append(
                DetectedEvent(
                    start_sample=start,
                    predicted_target=label,
                    target_probability=probability,
                )
            )
            next_allowed = start + REFRACTORY_SAMPLES
    return tuple(events)


def _require_integer(
    value: object,
    *,
    name: str,
    minimum: int,
    maximum: int = _INT64_MAX,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ContinuousReplayError(
            f"{name} must be an integer from {minimum} through {maximum}"
        )
    return value


def _require_exposure_samples(
    value: object,
    *,
    name: str,
    allow_zero: bool,
) -> int:
    minimum = 0 if allow_zero else 1
    exposure = _require_integer(value, name=name, minimum=minimum)
    if exposure == 0:
        return exposure
    if exposure < CLIP_SAMPLES or (exposure - CLIP_SAMPLES) % WINDOW_HOP_SAMPLES:
        raise ContinuousReplayError(
            f"{name} is not a sum-compatible registered exposure"
        )
    return exposure


def _frozen_int64_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
) -> Int64Array:
    if (
        type(value) is not np.ndarray
        or cast(np.ndarray[tuple[int, ...], np.dtype[np.int64]], value).dtype
        != np.dtype(np.int64)
        or cast(np.ndarray[tuple[int, ...], np.dtype[np.int64]], value).shape != shape
    ):
        raise ContinuousReplayError(f"{name} must be an int64 array of shape {shape}")
    array = cast(Int64Array, value)
    if np.any(array < 0):
        raise ContinuousReplayError(f"{name} contains a negative integer")
    result = np.array(array, dtype=np.int64, order="C", copy=True)
    result.flags.writeable = False
    return result


def _frozen_bool_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
) -> BoolArray:
    if (
        type(value) is not np.ndarray
        or cast(np.ndarray[tuple[int, ...], np.dtype[np.bool_]], value).dtype
        != np.dtype(np.bool_)
        or cast(np.ndarray[tuple[int, ...], np.dtype[np.bool_]], value).shape != shape
    ):
        raise ContinuousReplayError(f"{name} must be a bool array of shape {shape}")
    result = np.array(value, dtype=np.bool_, order="C", copy=True)
    result.flags.writeable = False
    return result


def _frozen_float64_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
) -> Float64Array:
    if (
        type(value) is not np.ndarray
        or cast(np.ndarray[tuple[int, ...], np.dtype[np.float64]], value).dtype
        != np.dtype(np.float64)
        or cast(np.ndarray[tuple[int, ...], np.dtype[np.float64]], value).shape != shape
    ):
        raise ContinuousReplayError(f"{name} must be a float64 array of shape {shape}")
    array = cast(Float64Array, value)
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ContinuousReplayError(f"{name} contains an invalid rate")
    result = np.array(array, dtype=np.float64, order="C", copy=True)
    result.flags.writeable = False
    return result


def _checked_target_sums(
    counts_by_target: Int64Array,
    *,
    expected: Int64Array,
    name: str,
) -> None:
    flattened = counts_by_target.reshape((-1, len(TARGET_WORDS)))
    expected_flat = expected.reshape(-1)
    totals = np.zeros(expected_flat.shape, dtype=np.int64)
    for column in range(len(TARGET_WORDS)):
        values = flattened[:, column]
        if np.any(values > _INT64_MAX - totals):
            raise ContinuousReplayError(f"{name} overflows int64 while summing")
        totals += values
    if not np.array_equal(totals, expected_flat):
        raise ContinuousReplayError(f"{name} does not sum to its total counts")


def _checked_axis_zero_sum(values: Int64Array, *, name: str) -> Int64Array:
    result = np.zeros(values.shape[1:], dtype=np.int64)
    for row in values:
        if np.any(row > _INT64_MAX - result):
            raise ContinuousReplayError(f"{name} overflows int64 while summing")
        result += row
    return result


def _validated_event_grid(events: object, *, exposure: int) -> EventGrid:
    if type(events) is not EventGrid:
        raise ContinuousReplayError("events must be an EventGrid")
    selected = events
    counts = _frozen_int64_array(
        selected.counts,
        name="utterance event counts",
        shape=(THRESHOLD_COUNT,),
    )
    counts_by_target = _frozen_int64_array(
        selected.counts_by_target,
        name="utterance event counts by target",
        shape=(THRESHOLD_COUNT, len(TARGET_WORDS)),
    )
    _checked_target_sums(
        counts_by_target,
        expected=counts,
        name="utterance event counts by target",
    )
    if np.any(counts[1:] > counts[:-1]):
        raise ContinuousReplayError(
            "utterance event counts must not increase with threshold"
        )
    maximum_events = exposure // REFRACTORY_SAMPLES
    if np.any(counts > maximum_events):
        raise ContinuousReplayError("utterance event count exceeds its exposure")
    return EventGrid(counts=counts, counts_by_target=counts_by_target)


def _validate_aggregated_event_grid(grid: AggregatedEventGrid) -> None:
    if type(grid.scored_exposure_samples) is not int:
        raise ContinuousReplayError(
            "aggregated scored_exposure_samples must be an integer"
        )
    exposure = _require_exposure_samples(
        grid.scored_exposure_samples,
        name="aggregated scored_exposure_samples",
        allow_zero=False,
    )
    if type(grid.speaker_ids) is not np.ndarray or grid.speaker_ids.ndim != 1:
        raise ContinuousReplayError("speaker_ids must be a one-dimensional int64 array")
    speaker_count = int(grid.speaker_ids.size)
    if speaker_count == 0:
        raise ContinuousReplayError("aggregated event grid has no speakers")
    speaker_ids = _frozen_int64_array(
        grid.speaker_ids, name="speaker_ids", shape=(speaker_count,)
    )
    if np.any(speaker_ids[1:] <= speaker_ids[:-1]):
        raise ContinuousReplayError("speaker_ids are not strictly increasing")
    utterance_counts = _frozen_int64_array(
        grid.utterance_counts_by_speaker,
        name="utterance_counts_by_speaker",
        shape=(speaker_count,),
    )
    if np.any(utterance_counts == 0):
        raise ContinuousReplayError("every speaker must own at least one utterance")
    if sum(int(value) for value in utterance_counts) > _INT64_MAX:
        raise ContinuousReplayError("corpus utterance count overflows int64")
    exposures = _frozen_int64_array(
        grid.scored_exposure_samples_by_speaker,
        name="scored_exposure_samples_by_speaker",
        shape=(speaker_count,),
    )
    if np.any(exposures == 0):
        raise ContinuousReplayError("every speaker must have positive scored exposure")
    for value in exposures:
        _require_exposure_samples(
            int(value),
            name="speaker scored exposure",
            allow_zero=False,
        )
    speaker_counts = _frozen_int64_array(
        grid.counts_by_speaker,
        name="counts_by_speaker",
        shape=(speaker_count, THRESHOLD_COUNT),
    )
    speaker_target_counts = _frozen_int64_array(
        grid.counts_by_speaker_and_target,
        name="counts_by_speaker_and_target",
        shape=(speaker_count, THRESHOLD_COUNT, len(TARGET_WORDS)),
    )
    _checked_target_sums(
        speaker_target_counts,
        expected=speaker_counts,
        name="speaker event counts by target",
    )
    if np.any(speaker_counts[:, 1:] > speaker_counts[:, :-1]):
        raise ContinuousReplayError(
            "speaker event counts must not increase with threshold"
        )
    if np.any(speaker_counts > (exposures // REFRACTORY_SAMPLES)[:, None]):
        raise ContinuousReplayError("speaker event count exceeds its exposure")
    counts = _frozen_int64_array(
        grid.counts,
        name="aggregated event counts",
        shape=(THRESHOLD_COUNT,),
    )
    target_counts = _frozen_int64_array(
        grid.counts_by_target,
        name="aggregated event counts by target",
        shape=(THRESHOLD_COUNT, len(TARGET_WORDS)),
    )
    _checked_target_sums(
        target_counts,
        expected=counts,
        name="aggregated event counts by target",
    )
    if sum(int(value) for value in exposures) != exposure:
        raise ContinuousReplayError("speaker exposures do not sum to corpus exposure")
    if not np.array_equal(
        _checked_axis_zero_sum(speaker_counts, name="speaker event counts"), counts
    ):
        raise ContinuousReplayError("speaker event counts do not sum to corpus counts")
    if not np.array_equal(
        _checked_axis_zero_sum(
            speaker_target_counts, name="speaker event counts by target"
        ),
        target_counts,
    ):
        raise ContinuousReplayError(
            "speaker target counts do not sum to corpus target counts"
        )
    object.__setattr__(grid, "speaker_ids", speaker_ids)
    object.__setattr__(grid, "utterance_counts_by_speaker", utterance_counts)
    object.__setattr__(grid, "scored_exposure_samples_by_speaker", exposures)
    object.__setattr__(grid, "counts_by_speaker", speaker_counts)
    object.__setattr__(grid, "counts_by_speaker_and_target", speaker_target_counts)
    object.__setattr__(grid, "scored_exposure_samples", exposure)
    object.__setattr__(grid, "counts", counts)
    object.__setattr__(grid, "counts_by_target", target_counts)


def _validate_correct_accept_grid(grid: CorrectAcceptGrid) -> None:
    clip_counts = _frozen_int64_array(
        grid.clip_counts_by_target,
        name="positive clip counts by target",
        shape=(len(TARGET_WORDS),),
    )
    if not np.any(clip_counts):
        raise ContinuousReplayError("positive validation grid has no target clips")
    if sum(int(value) for value in clip_counts) > _INT64_MAX:
        raise ContinuousReplayError("positive target support overflows int64")
    counts = _frozen_int64_array(
        grid.counts,
        name="correct accept counts",
        shape=(THRESHOLD_COUNT,),
    )
    target_counts = _frozen_int64_array(
        grid.counts_by_target,
        name="correct accept counts by target",
        shape=(THRESHOLD_COUNT, len(TARGET_WORDS)),
    )
    _checked_target_sums(
        target_counts,
        expected=counts,
        name="correct accept counts by target",
    )
    if np.any(target_counts > clip_counts[None, :]):
        raise ContinuousReplayError("correct accepts exceed positive target support")
    if np.any(counts[1:] > counts[:-1]) or np.any(
        target_counts[1:] > target_counts[:-1]
    ):
        raise ContinuousReplayError(
            "correct accept counts must not increase with threshold"
        )
    if counts[0] == 0:
        raise ContinuousReplayError("baseline correct count must be positive")
    object.__setattr__(grid, "clip_counts_by_target", clip_counts)
    object.__setattr__(grid, "counts", counts)
    object.__setattr__(grid, "counts_by_target", target_counts)


def _validate_threshold_selection(selection: ThresholdSelection) -> None:
    if type(selection.status) is not str or selection.status not in {"pass", "reject"}:
        raise ContinuousReplayError("selection status is not pass or reject")
    if selection.selected_threshold_milli is not None and (
        type(selection.selected_threshold_milli) is not int
        or not 0 <= selection.selected_threshold_milli < THRESHOLD_COUNT
    ):
        raise ContinuousReplayError("selected threshold has an invalid schema")
    if (
        type(selection.retention_frontier_milli) is not int
        or not 0 <= selection.retention_frontier_milli < THRESHOLD_COUNT
    ):
        raise ContinuousReplayError("retention frontier has an invalid schema")
    if selection.negative_frontier_milli is not None and (
        type(selection.negative_frontier_milli) is not int
        or not 0 <= selection.negative_frontier_milli < THRESHOLD_COUNT
    ):
        raise ContinuousReplayError("negative frontier has an invalid schema")
    false_gate = _frozen_bool_array(
        selection.false_event_gate,
        name="false event gate",
        shape=(THRESHOLD_COUNT,),
    )
    retention_gate = _frozen_bool_array(
        selection.retention_gate,
        name="retention gate",
        shape=(THRESHOLD_COUNT,),
    )
    passing = _frozen_bool_array(
        selection.passing,
        name="passing threshold grid",
        shape=(THRESHOLD_COUNT,),
    )
    if np.any(false_gate[:-1] & ~false_gate[1:]):
        raise ContinuousReplayError("false event gate must remain true once met")
    if np.any(~retention_gate[:-1] & retention_gate[1:]):
        raise ContinuousReplayError(
            "retention gate cannot recover at a higher threshold"
        )
    if not np.array_equal(passing, false_gate & retention_gate):
        raise ContinuousReplayError("passing grid is not the conjunction of its gates")
    retained = np.flatnonzero(retention_gate)
    if retained.size == 0:
        raise ContinuousReplayError("retention gate has no passing threshold")
    retention_frontier = int(retained[-1])
    negative = np.flatnonzero(false_gate)
    negative_frontier = int(negative[0]) if negative.size else None
    passed = np.flatnonzero(passing)
    selected = int(passed[0]) if passed.size else None
    status: Literal["pass", "reject"] = "pass" if selected is not None else "reject"
    if selection.status != status:
        raise ContinuousReplayError("selection status differs from its exact gates")
    if selection.selected_threshold_milli != selected:
        raise ContinuousReplayError(
            "selected threshold is not the lowest passing point"
        )
    if selection.retention_frontier_milli != retention_frontier:
        raise ContinuousReplayError("retention frontier differs from its gate")
    if selection.negative_frontier_milli != negative_frontier:
        raise ContinuousReplayError("negative frontier differs from its gate")
    object.__setattr__(selection, "false_event_gate", false_gate)
    object.__setattr__(selection, "retention_gate", retention_gate)
    object.__setattr__(selection, "passing", passing)


def _validate_event_rate_intervals(intervals: EventRateIntervals) -> None:
    point = _frozen_float64_array(
        intervals.point, name="event rate point estimates", shape=(THRESHOLD_COUNT,)
    )
    garwood_lower = _frozen_float64_array(
        intervals.garwood_lower,
        name="Garwood lower bounds",
        shape=(THRESHOLD_COUNT,),
    )
    garwood_upper = _frozen_float64_array(
        intervals.garwood_upper,
        name="Garwood upper bounds",
        shape=(THRESHOLD_COUNT,),
    )
    bootstrap_lower = _frozen_float64_array(
        intervals.speaker_bootstrap_lower,
        name="speaker bootstrap lower bounds",
        shape=(THRESHOLD_COUNT,),
    )
    bootstrap_upper = _frozen_float64_array(
        intervals.speaker_bootstrap_upper,
        name="speaker bootstrap upper bounds",
        shape=(THRESHOLD_COUNT,),
    )
    if np.any(garwood_lower > point) or np.any(point > garwood_upper):
        raise ContinuousReplayError("Garwood interval does not contain its point rate")
    if np.any(bootstrap_lower > bootstrap_upper):
        raise ContinuousReplayError("speaker bootstrap interval bounds are reversed")
    object.__setattr__(intervals, "point", point)
    object.__setattr__(intervals, "garwood_lower", garwood_lower)
    object.__setattr__(intervals, "garwood_upper", garwood_upper)
    object.__setattr__(intervals, "speaker_bootstrap_lower", bootstrap_lower)
    object.__setattr__(intervals, "speaker_bootstrap_upper", bootstrap_upper)


def aggregate_event_grids(
    utterances: Sequence[UtteranceEventGrid],
) -> AggregatedEventGrid:
    """Aggregate immutable utterance grids in ascending integer speaker order."""

    if not isinstance(utterances, Sequence):
        raise ContinuousReplayError("utterances must be a finite sequence")
    if len(utterances) == 0:
        raise ContinuousReplayError("cannot aggregate an empty utterance sequence")
    total_exposure = 0
    by_speaker: dict[int, tuple[int, int, Int64Array, Int64Array]] = {}
    for record in utterances:
        if type(record) is not UtteranceEventGrid:
            raise ContinuousReplayError(
                "utterances must contain only UtteranceEventGrid records"
            )
        snapshot = UtteranceEventGrid(
            speaker_id=record.speaker_id,
            scored_exposure_samples=record.scored_exposure_samples,
            events=record.events,
        )
        total_exposure += snapshot.scored_exposure_samples
        if total_exposure > _INT64_MAX:
            raise ContinuousReplayError("corpus exposure overflows int64")
        previous = by_speaker.get(snapshot.speaker_id)
        if previous is None:
            by_speaker[snapshot.speaker_id] = (
                1,
                snapshot.scored_exposure_samples,
                snapshot.events.counts.copy(),
                snapshot.events.counts_by_target.copy(),
            )
            continue
        utterance_count, speaker_exposure, counts, target_counts = previous
        if utterance_count == _INT64_MAX:
            raise ContinuousReplayError("speaker utterance count overflows int64")
        speaker_exposure += snapshot.scored_exposure_samples
        if speaker_exposure > _INT64_MAX:
            raise ContinuousReplayError("speaker exposure overflows int64")
        if np.any(snapshot.events.counts > _INT64_MAX - counts):
            raise ContinuousReplayError("speaker event counts overflow int64")
        if np.any(snapshot.events.counts_by_target > _INT64_MAX - target_counts):
            raise ContinuousReplayError("speaker target counts overflow int64")
        counts += snapshot.events.counts
        target_counts += snapshot.events.counts_by_target
        by_speaker[snapshot.speaker_id] = (
            utterance_count + 1,
            speaker_exposure,
            counts,
            target_counts,
        )

    speaker_ids_list = sorted(by_speaker)
    speaker_count = len(speaker_ids_list)
    utterance_counts = np.zeros(speaker_count, dtype=np.int64)
    exposures = np.zeros(speaker_count, dtype=np.int64)
    speaker_counts = np.zeros((speaker_count, THRESHOLD_COUNT), dtype=np.int64)
    speaker_target_counts = np.zeros(
        (speaker_count, THRESHOLD_COUNT, len(TARGET_WORDS)), dtype=np.int64
    )
    for position, speaker_id in enumerate(speaker_ids_list):
        utterance_count, speaker_exposure, counts, target_counts = by_speaker[
            speaker_id
        ]
        utterance_counts[position] = utterance_count
        exposures[position] = speaker_exposure
        speaker_counts[position] = counts
        speaker_target_counts[position] = target_counts

    counts = _checked_axis_zero_sum(speaker_counts, name="speaker event counts")
    target_counts = _checked_axis_zero_sum(
        speaker_target_counts, name="speaker event counts by target"
    )
    return AggregatedEventGrid(
        speaker_ids=np.asarray(speaker_ids_list, dtype=np.int64),
        utterance_counts_by_speaker=utterance_counts,
        scored_exposure_samples_by_speaker=exposures,
        counts_by_speaker=speaker_counts,
        counts_by_speaker_and_target=speaker_target_counts,
        scored_exposure_samples=total_exposure,
        counts=counts,
        counts_by_target=target_counts,
    )


def correct_accept_threshold_grid(
    model: PortableLinearModel,
    target_positions: Int64Array,
    scores: ScoreBatch,
) -> CorrectAcceptGrid:
    """Count correct target argmax decisions surviving each threshold."""

    if model.classes != CLASS_ORDER:
        raise ContinuousReplayError("portable model class order differs")
    row_count = scores.predicted_indices.size
    if (
        type(target_positions) is not np.ndarray
        or target_positions.dtype != np.dtype(np.int64)
        or target_positions.shape != (row_count,)
        or np.any(target_positions < 0)
        or np.any(target_positions >= len(TARGET_WORDS))
    ):
        raise ContinuousReplayError(
            "target_positions must be an in-range int64 vector matching scores"
        )
    predicted, probabilities = _score_vectors(model, scores)
    model_target_indices = np.asarray(
        [model.classes.index(target) for target in TARGET_WORDS], dtype=np.int64
    )
    clip_counts = np.bincount(target_positions, minlength=len(TARGET_WORDS)).astype(
        np.int64, copy=False
    )
    target_counts = np.zeros((THRESHOLD_COUNT, len(TARGET_WORDS)), dtype=np.int64)
    for row, target_position in enumerate(target_positions.tolist()):
        if predicted[row] != model_target_indices[target_position]:
            continue
        accepted = probabilities[row] >= _THRESHOLDS
        target_counts[accepted, target_position] += 1
    counts = np.sum(target_counts, axis=1, dtype=np.int64)
    return CorrectAcceptGrid(
        clip_counts_by_target=clip_counts,
        counts=counts,
        counts_by_target=target_counts,
    )


def select_threshold(
    negative: AggregatedEventGrid,
    positive: CorrectAcceptGrid,
) -> ThresholdSelection:
    """Apply both experiment-001 gates with exact Python integer arithmetic."""

    _validate_aggregated_event_grid(negative)
    _validate_correct_accept_grid(positive)
    exposure = negative.scored_exposure_samples
    baseline_correct = positive.baseline_correct_count
    false_gate = np.fromiter(
        (
            int(event_count) * SAMPLES_PER_HOUR <= exposure
            for event_count in negative.counts
        ),
        dtype=np.bool_,
        count=THRESHOLD_COUNT,
    )
    retention_gate = np.fromiter(
        (
            int(correct_count) * 5 >= baseline_correct * 4
            for correct_count in positive.counts
        ),
        dtype=np.bool_,
        count=THRESHOLD_COUNT,
    )
    passing = false_gate & retention_gate
    retained = np.flatnonzero(retention_gate)
    negative_passes = np.flatnonzero(false_gate)
    passes = np.flatnonzero(passing)
    selected = int(passes[0]) if passes.size else None
    return ThresholdSelection(
        status="pass" if selected is not None else "reject",
        selected_threshold_milli=selected,
        false_event_gate=false_gate,
        retention_gate=retention_gate,
        passing=passing,
        retention_frontier_milli=int(retained[-1]),
        negative_frontier_milli=(
            int(negative_passes[0]) if negative_passes.size else None
        ),
    )


def garwood_rate_intervals(
    counts: Int64Array,
    scored_exposure_samples: int,
) -> tuple[Float64Array, Float64Array]:
    """Return nominal 95% Garwood event-rate bounds per scored hour."""

    exposure = _require_exposure_samples(
        scored_exposure_samples,
        name="Garwood scored_exposure_samples",
        allow_zero=False,
    )
    selected = _frozen_int64_array(
        counts, name="Garwood event counts", shape=(THRESHOLD_COUNT,)
    )
    if np.any(selected > exposure // REFRACTORY_SAMPLES):
        raise ContinuousReplayError("Garwood event count exceeds its exposure")
    values = selected.astype(np.float64)
    hours = np.float64(exposure) / np.float64(SAMPLES_PER_HOUR)
    lower = np.zeros(THRESHOLD_COUNT, dtype=np.float64)
    nonzero = selected != 0
    lower[nonzero] = (
        np.float64(0.5)
        * chi2.ppf(np.float64(0.025), np.float64(2.0) * values[nonzero])
        / hours
    )
    upper = (
        np.float64(0.5)
        * chi2.ppf(np.float64(0.975), np.float64(2.0) * (values + 1.0))
        / hours
    )
    lower = np.ascontiguousarray(lower, dtype=np.float64)
    upper = np.ascontiguousarray(upper, dtype=np.float64)
    lower.flags.writeable = False
    upper.flags.writeable = False
    return lower, upper


def speaker_bootstrap_indices(speaker_count: int) -> Int64Array:
    """Draw the one registered shared PCG64 speaker-resample matrix."""

    count = _require_integer(speaker_count, name="speaker_count", minimum=1)
    if BOOTSTRAP_RESAMPLES * count > _MAX_BOOTSTRAP_MATRIX_ELEMENTS:
        raise ContinuousReplayError("speaker bootstrap matrix exceeds its size limit")
    generator = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    indices = generator.integers(
        0,
        count,
        size=(BOOTSTRAP_RESAMPLES, count),
        dtype=np.int64,
    )
    indices.flags.writeable = False
    return indices


def _speaker_bootstrap_rate_intervals(
    aggregate: AggregatedEventGrid,
) -> tuple[Float64Array, Float64Array]:
    speaker_count = int(aggregate.speaker_ids.size)
    indices = speaker_bootstrap_indices(speaker_count)
    multiplicities = np.zeros((BOOTSTRAP_RESAMPLES, speaker_count), dtype=np.int64)
    rows = np.arange(BOOTSTRAP_RESAMPLES, dtype=np.int64)
    for column in range(speaker_count):
        np.add.at(multiplicities, (rows, indices[:, column]), 1)

    maximum_exposure = max(
        int(value) for value in aggregate.scored_exposure_samples_by_speaker
    )
    maximum_count = int(np.max(aggregate.counts_by_speaker))
    if (
        speaker_count * maximum_exposure > _INT64_MAX
        or speaker_count * maximum_count > _INT64_MAX
    ):
        raise ContinuousReplayError("speaker bootstrap sum can overflow int64")
    resampled_exposure = multiplicities @ aggregate.scored_exposure_samples_by_speaker
    if np.any(resampled_exposure <= 0):
        raise ContinuousReplayError("speaker bootstrap produced zero exposure")

    lower = np.empty(THRESHOLD_COUNT, dtype=np.float64)
    upper = np.empty(THRESHOLD_COUNT, dtype=np.float64)
    block_size = 32
    for start in range(0, THRESHOLD_COUNT, block_size):
        stop = min(start + block_size, THRESHOLD_COUNT)
        resampled_events = multiplicities @ aggregate.counts_by_speaker[:, start:stop]
        rates = (
            resampled_events.astype(np.float64)
            * np.float64(SAMPLES_PER_HOUR)
            / resampled_exposure[:, None].astype(np.float64)
        )
        percentiles = np.percentile(
            rates,
            (np.float64(2.5), np.float64(97.5)),
            axis=0,
            method="linear",
        )
        lower[start:stop] = percentiles[0]
        upper[start:stop] = percentiles[1]
    lower.flags.writeable = False
    upper.flags.writeable = False
    return lower, upper


def event_rate_intervals(aggregate: AggregatedEventGrid) -> EventRateIntervals:
    """Compute point, Garwood, and shared speaker-bootstrap rate grids."""

    _validate_aggregated_event_grid(aggregate)
    exposure = aggregate.scored_exposure_samples
    point = (
        aggregate.counts.astype(np.float64)
        * np.float64(SAMPLES_PER_HOUR)
        / np.float64(exposure)
    )
    garwood_lower, garwood_upper = garwood_rate_intervals(aggregate.counts, exposure)
    bootstrap_lower, bootstrap_upper = _speaker_bootstrap_rate_intervals(aggregate)
    return EventRateIntervals(
        point=point,
        garwood_lower=garwood_lower,
        garwood_upper=garwood_upper,
        speaker_bootstrap_lower=bootstrap_lower,
        speaker_bootstrap_upper=bootstrap_upper,
    )
