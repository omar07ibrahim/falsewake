"""Exact validation arithmetic and gates for Experiment 002.

Population provenance belongs to the separate registered-evidence layer.
"""

from __future__ import annotations

import threading
import weakref
from dataclasses import dataclass
from fractions import Fraction
from typing import Final

import numpy as np
from numpy.typing import NDArray

from falsewake.experiment_002_data import CLASS_ORDER, VALIDATION_EXAMPLE_COUNT
from falsewake.experiment_002_validation import VALIDATION_CLASS_SUPPORT

Float32Array = NDArray[np.float32]
Int64Array = NDArray[np.int64]
CLASS_COUNT: Final = len(CLASS_ORDER)
TARGET_CLASS_COUNT: Final = 10
TARGET_POPULATION: Final = 3_703
_LAYOUT_ROUTE_MARKER: Final = object()


class Experiment002MetricsError(ValueError):
    """Validation predictions or metric state violate the frozen contract."""


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
class ValidationMetrics:
    """Issuer-only outputs from one exact validation computation."""

    _confusion_matrix: Int64Array
    _cross_entropy: np.float64
    _frame_97_logits: Float32Array
    _label_indices: Int64Array
    _layout: _MetricLayout
    _macro_f1: Fraction
    _predicted_indices: Int64Array
    _layout_route_marker: object | None

    def __init__(self) -> None:
        raise TypeError("ValidationMetrics values are issued by this module")

    @property
    def confusion_matrix(self) -> Int64Array:
        """Return a fresh immutable snapshot of the exact confusion matrix."""

        return _snapshot_int64(_issued_metric_state(self).confusion_matrix)

    @property
    def cross_entropy(self) -> np.float64:
        """Return the exact accumulated binary64 validation cross-entropy."""

        return np.float64(_issued_metric_state(self).cross_entropy)

    @property
    def macro_f1(self) -> Fraction:
        """Return the exact reduced macro-F1 fraction."""

        return _issued_metric_state(self).macro_f1

    @property
    def predicted_indices(self) -> Int64Array:
        """Return a fresh immutable snapshot in the issued input-row order."""

        return _snapshot_int64(_issued_metric_state(self).predicted_indices)

    @property
    def cross_entropy_hex(self) -> str:
        """Return the canonical lowercase CPython binary64 spelling."""

        return float(_issued_metric_state(self).cross_entropy).hex()


@dataclass(frozen=True, slots=True)
class _IssuedValidationMetricsState:
    confusion_matrix: Int64Array
    cross_entropy: np.float64
    frame_97_logits: Float32Array
    label_indices: Int64Array
    layout: _MetricLayout
    macro_f1: Fraction
    predicted_indices: Int64Array
    layout_route_marker: object | None


_ISSUED_METRICS: weakref.WeakKeyDictionary[
    ValidationMetrics, _IssuedValidationMetricsState
] = weakref.WeakKeyDictionary()
_ISSUED_METRICS_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class ValidationGateResults:
    """The five frozen validation predicates evaluated on one confusion matrix."""

    every_target_recall: bool
    macro_f1: bool
    silence_target_rate: bool
    target_accuracy: bool
    unknown_target_rate: bool

    def __post_init__(self) -> None:
        for name in (
            "every_target_recall",
            "macro_f1",
            "silence_target_rate",
            "target_accuracy",
            "unknown_target_rate",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")

    @property
    def passed(self) -> bool:
        """Return whether every frozen absolute predicate passed."""

        return (
            self.every_target_recall
            and self.macro_f1
            and self.silence_target_rate
            and self.target_accuracy
            and self.unknown_target_rate
        )


@dataclass(frozen=True, slots=True)
class _MetricLayout:
    example_count: int
    class_support: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.example_count) is not int:
            raise TypeError("example_count must be an integer")
        if self.example_count < 1:
            raise Experiment002MetricsError("example_count must be positive")
        if type(self.class_support) is not tuple or len(self.class_support) != (
            CLASS_COUNT
        ):
            raise Experiment002MetricsError("class_support has an invalid shape")
        if any(type(count) is not int or count < 0 for count in self.class_support):
            raise Experiment002MetricsError("class_support contains invalid counts")
        if sum(self.class_support) != self.example_count:
            raise Experiment002MetricsError("class_support differs from example_count")


_FROZEN_VALIDATION_LAYOUT: Final = _MetricLayout(
    example_count=VALIDATION_EXAMPLE_COUNT,
    class_support=VALIDATION_CLASS_SUPPORT,
)


def evaluate_validation_layout_logits(
    label_indices: Int64Array,
    frame_97_logits: Float32Array,
) -> ValidationMetrics:
    """Score arrays that match the frozen validation shape and class support.

    This route does not bind rows to registered population identities and therefore
    cannot, by itself, issue registered validation evidence.
    """

    _require_metric_inputs(label_indices, frame_97_logits, _FROZEN_VALIDATION_LAYOUT)
    return _compute_validation_metrics(
        label_indices,
        frame_97_logits,
        _FROZEN_VALIDATION_LAYOUT,
        layout_route_marker=_LAYOUT_ROUTE_MARKER,
    )


def verify_validation_layout_metrics(result: ValidationMetrics) -> None:
    """Reject values not issued through the frozen layout-only route."""

    state = _issued_metric_state(result)
    if state.layout_route_marker is not _LAYOUT_ROUTE_MARKER or (
        state.layout != _FROZEN_VALIDATION_LAYOUT
    ):
        raise Experiment002MetricsError(
            "validation metrics differ from the frozen layout route"
        )
    observed_support = tuple(
        _row_total(state.confusion_matrix, class_index)
        for class_index in range(CLASS_COUNT)
    )
    if observed_support != _FROZEN_VALIDATION_LAYOUT.class_support or (
        state.predicted_indices.shape != (_FROZEN_VALIDATION_LAYOUT.example_count,)
    ):
        raise Experiment002MetricsError(
            "validation metrics differ from the frozen layout route"
        )


def verify_validation_metrics_inputs(
    result: ValidationMetrics,
    label_indices: Int64Array,
    frame_97_logits: Float32Array,
) -> None:
    """Verify exact label/logit bytes, without asserting population provenance."""

    state = _issued_metric_state(result)
    _require_metric_inputs(label_indices, frame_97_logits, state.layout)
    if not _same_int64_bytes(label_indices, state.label_indices) or not (
        _same_float32_bytes(frame_97_logits, state.frame_97_logits)
    ):
        raise Experiment002MetricsError(
            "validation metric inputs differ from the issued computation"
        )


def evaluate_validation_gate_predicates(
    confusion_matrix: Int64Array,
) -> ValidationGateResults:
    """Evaluate preregistered predicates; this does not issue result evidence."""

    _require_confusion_matrix(confusion_matrix)
    observed_support = tuple(
        _row_total(confusion_matrix, class_index) for class_index in range(CLASS_COUNT)
    )
    if observed_support != VALIDATION_CLASS_SUPPORT:
        raise Experiment002MetricsError(
            "confusion matrix support differs from the frozen validation layout"
        )

    every_target_recall = all(
        10 * int(confusion_matrix[class_index, class_index])
        >= 7 * VALIDATION_CLASS_SUPPORT[class_index]
        for class_index in range(TARGET_CLASS_COUNT)
    )
    target_correct = sum(
        int(confusion_matrix[class_index, class_index])
        for class_index in range(TARGET_CLASS_COUNT)
    )
    unknown_target = sum(
        int(confusion_matrix[10, class_index])
        for class_index in range(TARGET_CLASS_COUNT)
    )
    silence_target = sum(
        int(confusion_matrix[11, class_index])
        for class_index in range(TARGET_CLASS_COUNT)
    )
    return ValidationGateResults(
        every_target_recall=every_target_recall,
        macro_f1=macro_f1_from_confusion(confusion_matrix) >= Fraction(4, 5),
        silence_target_rate=20 * silence_target <= VALIDATION_CLASS_SUPPORT[11],
        target_accuracy=20 * target_correct >= 17 * TARGET_POPULATION,
        unknown_target_rate=5 * unknown_target <= VALIDATION_CLASS_SUPPORT[10],
    )


def macro_f1_from_confusion(confusion_matrix: Int64Array) -> Fraction:
    """Compute the exact unweighted 12-class one-vs-rest macro-F1."""

    _require_confusion_matrix(confusion_matrix)
    total = Fraction(0, 1)
    for class_index in range(CLASS_COUNT):
        true_positive = int(confusion_matrix[class_index, class_index])
        false_positive = sum(
            int(confusion_matrix[row_index, class_index])
            for row_index in range(CLASS_COUNT)
            if row_index != class_index
        )
        false_negative = sum(
            int(confusion_matrix[class_index, column_index])
            for column_index in range(CLASS_COUNT)
            if column_index != class_index
        )
        denominator = 2 * true_positive + false_positive + false_negative
        class_f1 = (
            Fraction(0, 1)
            if denominator == 0
            else Fraction(2 * true_positive, denominator)
        )
        total += class_f1
    return total / CLASS_COUNT


def _evaluate_validation_logits(
    label_indices: Int64Array,
    frame_97_logits: Float32Array,
    *,
    layout: _MetricLayout,
) -> ValidationMetrics:
    """Private synthetic-layout seam with the frozen scalar arithmetic."""

    if type(layout) is not _MetricLayout:
        raise TypeError("layout must be a _MetricLayout")
    _require_metric_inputs(label_indices, frame_97_logits, layout)
    return _compute_validation_metrics(label_indices, frame_97_logits, layout)


def _compute_validation_metrics(
    label_indices: Int64Array,
    frame_97_logits: Float32Array,
    layout: _MetricLayout,
    *,
    layout_route_marker: object | None = None,
) -> ValidationMetrics:
    labels_snapshot = _snapshot_int64(label_indices)
    logits_snapshot = _snapshot_float32(frame_97_logits)
    layout_snapshot = _MetricLayout(layout.example_count, layout.class_support)
    _require_metric_inputs(labels_snapshot, logits_snapshot, layout_snapshot)
    predictions = np.array(
        np.argmax(logits_snapshot, axis=1),
        dtype=np.int64,
        order="C",
        copy=True,
    )
    confusion = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64, order="C")
    cross_entropy_sum = np.float64(0.0)

    for row_index in range(layout_snapshot.example_count):
        true_index = int(labels_snapshot[row_index])
        predicted_index = int(predictions[row_index])
        confusion[true_index, predicted_index] += np.int64(1)

        row = logits_snapshot[row_index]
        maximum = np.float64(row[0])
        for class_index in range(1, CLASS_COUNT):
            candidate = np.float64(row[class_index])
            if candidate > maximum:
                maximum = candidate
        exponential_sum = np.float64(0.0)
        for class_index in range(CLASS_COUNT):
            shifted = np.float64(np.float64(row[class_index]) - maximum)
            exponential_sum = np.float64(
                exponential_sum + np.exp(shifted, dtype=np.float64)
            )
        row_cross_entropy = np.float64(
            np.log(exponential_sum, dtype=np.float64)
            + maximum
            - np.float64(row[true_index])
        )
        if not np.isfinite(row_cross_entropy):
            raise Experiment002MetricsError(
                "validation cross-entropy became non-finite"
            )
        cross_entropy_sum = np.float64(cross_entropy_sum + row_cross_entropy)

    cross_entropy = np.float64(
        cross_entropy_sum / np.float64(layout_snapshot.example_count)
    )
    macro_f1 = macro_f1_from_confusion(confusion)
    _validate_metric_values(confusion, cross_entropy, macro_f1, predictions)

    result = object.__new__(ValidationMetrics)
    object.__setattr__(result, "_confusion_matrix", _snapshot_int64(confusion))
    object.__setattr__(result, "_cross_entropy", np.float64(cross_entropy))
    object.__setattr__(
        result,
        "_frame_97_logits",
        _snapshot_float32(logits_snapshot),
    )
    object.__setattr__(result, "_label_indices", _snapshot_int64(labels_snapshot))
    object.__setattr__(result, "_layout", layout_snapshot)
    object.__setattr__(result, "_macro_f1", macro_f1)
    object.__setattr__(result, "_predicted_indices", _snapshot_int64(predictions))
    object.__setattr__(
        result,
        "_layout_route_marker",
        layout_route_marker,
    )
    state = _IssuedValidationMetricsState(
        confusion_matrix=_snapshot_int64(confusion),
        cross_entropy=np.float64(cross_entropy),
        frame_97_logits=_snapshot_float32(logits_snapshot),
        label_indices=_snapshot_int64(labels_snapshot),
        layout=_MetricLayout(
            layout_snapshot.example_count, layout_snapshot.class_support
        ),
        macro_f1=macro_f1,
        predicted_indices=_snapshot_int64(predictions),
        layout_route_marker=layout_route_marker,
    )
    with _ISSUED_METRICS_LOCK:
        _ISSUED_METRICS[result] = state
    return result


def _require_metric_inputs(
    label_indices: Int64Array,
    frame_97_logits: Float32Array,
    layout: _MetricLayout,
) -> None:
    if type(layout) is not _MetricLayout:
        raise TypeError("layout must be a _MetricLayout")
    if type(label_indices) is not np.ndarray:
        raise TypeError("label_indices must be a NumPy array")
    if label_indices.dtype != np.dtype(np.int64) or label_indices.shape != (
        layout.example_count,
    ):
        raise Experiment002MetricsError("label_indices have an invalid shape or dtype")
    if (
        not label_indices.flags.c_contiguous
        or not label_indices.flags.owndata
        or label_indices.flags.writeable
    ):
        raise Experiment002MetricsError(
            "label_indices must be owned, C-contiguous, and read-only"
        )
    if np.any(label_indices < 0) or np.any(label_indices >= CLASS_COUNT):
        raise Experiment002MetricsError("label_indices are outside class order")
    observed_support = [0] * CLASS_COUNT
    for label_index in label_indices:
        observed_support[int(label_index)] += 1
    if tuple(observed_support) != layout.class_support:
        raise Experiment002MetricsError("label support differs from the metric layout")

    if type(frame_97_logits) is not np.ndarray:
        raise TypeError("frame_97_logits must be a NumPy array")
    if frame_97_logits.dtype != np.dtype(np.float32) or (
        frame_97_logits.shape != (layout.example_count, CLASS_COUNT)
    ):
        raise Experiment002MetricsError(
            "frame_97_logits have an invalid shape or dtype"
        )
    if (
        not frame_97_logits.flags.c_contiguous
        or not frame_97_logits.flags.owndata
        or frame_97_logits.flags.writeable
    ):
        raise Experiment002MetricsError(
            "frame_97_logits must be owned, C-contiguous, and read-only"
        )
    if not np.all(np.isfinite(frame_97_logits)):
        raise Experiment002MetricsError("frame_97_logits contain non-finite values")


def _require_confusion_matrix(confusion_matrix: Int64Array) -> None:
    if type(confusion_matrix) is not np.ndarray:
        raise TypeError("confusion_matrix must be a NumPy array")
    if confusion_matrix.dtype != np.dtype(np.int64) or confusion_matrix.shape != (
        CLASS_COUNT,
        CLASS_COUNT,
    ):
        raise Experiment002MetricsError(
            "confusion_matrix has an invalid shape or dtype"
        )
    if not confusion_matrix.flags.c_contiguous:
        raise Experiment002MetricsError("confusion_matrix must be C-contiguous")
    if np.any(confusion_matrix < 0):
        raise Experiment002MetricsError("confusion_matrix contains a negative count")


def _validate_metric_values(
    confusion_matrix: Int64Array,
    cross_entropy: np.float64,
    macro_f1: Fraction,
    predicted_indices: Int64Array,
) -> None:
    _require_confusion_matrix(confusion_matrix)
    if type(cross_entropy) is not np.float64 or not np.isfinite(cross_entropy):
        raise Experiment002MetricsError(
            "cross_entropy must be a finite NumPy float64 scalar"
        )
    if cross_entropy < np.float64(0.0):
        raise Experiment002MetricsError("cross_entropy must be nonnegative")
    if type(macro_f1) is not Fraction:
        raise TypeError("macro_f1 must be a Fraction")
    if not Fraction(0, 1) <= macro_f1 <= Fraction(1, 1):
        raise Experiment002MetricsError("macro_f1 must be within [0, 1]")
    if macro_f1 != macro_f1_from_confusion(confusion_matrix):
        raise Experiment002MetricsError("macro_f1 differs from the confusion matrix")
    if type(predicted_indices) is not np.ndarray:
        raise TypeError("predicted_indices must be a NumPy array")
    prediction_count = _confusion_total(confusion_matrix)
    if predicted_indices.dtype != np.dtype(np.int64) or (
        predicted_indices.shape != (prediction_count,)
    ):
        raise Experiment002MetricsError(
            "predicted_indices have an invalid shape or dtype"
        )
    if not predicted_indices.flags.c_contiguous:
        raise Experiment002MetricsError("predicted_indices must be C-contiguous")
    if prediction_count and (
        np.any(predicted_indices < 0) or np.any(predicted_indices >= CLASS_COUNT)
    ):
        raise Experiment002MetricsError("predicted_indices are outside class order")
    observed_predictions = [0] * CLASS_COUNT
    for predicted_index in predicted_indices:
        observed_predictions[int(predicted_index)] += 1
    expected_predictions = tuple(
        sum(
            int(confusion_matrix[row_index, class_index])
            for row_index in range(CLASS_COUNT)
        )
        for class_index in range(CLASS_COUNT)
    )
    if tuple(observed_predictions) != expected_predictions:
        raise Experiment002MetricsError(
            "predicted_indices differ from the confusion matrix"
        )


def _issued_metric_state(result: ValidationMetrics) -> _IssuedValidationMetricsState:
    if type(result) is not ValidationMetrics:
        raise TypeError("result must be ValidationMetrics")
    with _ISSUED_METRICS_LOCK:
        state = _ISSUED_METRICS.get(result)
    if state is None:
        raise Experiment002MetricsError(
            "validation metrics were not issued by this process"
        )
    _validate_metric_values(
        state.confusion_matrix,
        state.cross_entropy,
        state.macro_f1,
        state.predicted_indices,
    )
    _require_metric_inputs(
        state.label_indices,
        state.frame_97_logits,
        state.layout,
    )
    try:
        result_confusion = result._confusion_matrix
        result_cross_entropy = result._cross_entropy
        result_logits = result._frame_97_logits
        result_labels = result._label_indices
        result_layout = result._layout
        result_macro_f1 = result._macro_f1
        result_predictions = result._predicted_indices
        result_layout_route_marker = result._layout_route_marker
    except AttributeError as error:
        raise Experiment002MetricsError(
            "validation metric capability changed after issuance"
        ) from error
    if (
        not _same_int64_bytes(result_confusion, state.confusion_matrix)
        or type(result_cross_entropy) is not np.float64
        or result_cross_entropy.tobytes() != state.cross_entropy.tobytes()
        or not _same_float32_bytes(result_logits, state.frame_97_logits)
        or not _same_int64_bytes(result_labels, state.label_indices)
        or not _same_metric_layout(result_layout, state.layout)
        or type(result_macro_f1) is not Fraction
        or result_macro_f1 != state.macro_f1
        or not _same_int64_bytes(result_predictions, state.predicted_indices)
        or result_layout_route_marker is not state.layout_route_marker
    ):
        raise Experiment002MetricsError(
            "validation metric capability changed after issuance"
        )
    return state


def _snapshot_int64(values: Int64Array) -> Int64Array:
    if type(values) is not np.ndarray or values.dtype != np.dtype(np.int64):
        raise TypeError("metric snapshot must be an int64 NumPy array")
    snapshot = np.array(values, dtype=np.int64, order="C", copy=True)
    snapshot.setflags(write=False)
    return snapshot


def _snapshot_float32(values: Float32Array) -> Float32Array:
    if type(values) is not np.ndarray or values.dtype != np.dtype(np.float32):
        raise TypeError("metric snapshot must be a float32 NumPy array")
    snapshot = np.array(values, dtype=np.float32, order="C", copy=True)
    snapshot.setflags(write=False)
    return snapshot


def _same_int64_bytes(left: Int64Array, right: Int64Array) -> bool:
    return (
        type(left) is np.ndarray
        and left.dtype == np.dtype(np.int64)
        and left.shape == right.shape
        and left.flags.c_contiguous
        and left.flags.owndata
        and not left.flags.writeable
        and left.tobytes(order="C") == right.tobytes(order="C")
    )


def _same_float32_bytes(left: Float32Array, right: Float32Array) -> bool:
    return (
        type(left) is np.ndarray
        and left.dtype == np.dtype(np.float32)
        and left.shape == right.shape
        and left.flags.c_contiguous
        and left.flags.owndata
        and not left.flags.writeable
        and left.tobytes(order="C") == right.tobytes(order="C")
    )


def _same_metric_layout(left: _MetricLayout, right: _MetricLayout) -> bool:
    return (
        type(left) is _MetricLayout
        and type(left.example_count) is int
        and left.example_count == right.example_count
        and type(left.class_support) is tuple
        and all(type(count) is int for count in left.class_support)
        and left.class_support == right.class_support
    )


def _row_total(confusion_matrix: Int64Array, row_index: int) -> int:
    return sum(
        int(confusion_matrix[row_index, column_index])
        for column_index in range(CLASS_COUNT)
    )


def _confusion_total(confusion_matrix: Int64Array) -> int:
    return sum(
        _row_total(confusion_matrix, row_index) for row_index in range(CLASS_COUNT)
    )
