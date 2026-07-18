"""Portable scoring and event primitives for continuous replay."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from falsewake.features import FloatArray, extract_clip_features
from falsewake.speech_commands import TARGET_WORDS

MODEL_SHA256 = "d5ac3579f539288c6ed7769894dd30ad2272480956b2f9b24b7e730baaeb0cd3"
MAX_MODEL_BYTES = 1024 * 1024
CLIP_SAMPLES = 16_000
WINDOW_HOP_SAMPLES = 1_600
REFRACTORY_SAMPLES = 16_000
THRESHOLD_COUNT = 1_001
CLASS_ORDER = tuple(sorted((*TARGET_WORDS, "unknown", "silence")))

Float64Array = NDArray[np.float64]
Int64Array = NDArray[np.int64]

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
