"""Population-bound validation evidence for Experiment 002.

The metrics module deliberately knows only labels, logits, and a fixed layout.
This module is the provenance boundary that binds those numeric results to the
issuer-verified registered corpus and materialized validation population.
"""

from __future__ import annotations

import hashlib
import hmac
import struct
import threading
import weakref
from dataclasses import dataclass
from fractions import Fraction
from typing import Final

import numpy as np
from numpy.typing import NDArray

from falsewake.experiment_002_data import (
    CLASS_ORDER,
    Experiment002Corpus,
)
from falsewake.experiment_002_metrics import (
    ValidationMetrics,
    verify_validation_layout_metrics,
    verify_validation_metrics_inputs,
)
from falsewake.experiment_002_validation import (
    MaterializedValidationPopulation,
    verify_registered_materialized_validation,
)

Float32Array = NDArray[np.float32]
Int64Array = NDArray[np.int64]

VALIDATION_INPUTS_DOMAIN: Final = b"falsewake-exp002-validation-inputs-v1\0"
VALIDATION_PREDICTIONS_DOMAIN: Final = b"falsewake-exp002-validation-predictions-v1\0"
_CLASS_COUNT: Final = len(CLASS_ORDER)
_UINT8_MAX: Final = (1 << 8) - 1
_UINT32_MAX: Final = (1 << 32) - 1
_UINT64_MAX: Final = (1 << 64) - 1
_UINT8: Final = struct.Struct("<B")
_UINT32LE: Final = struct.Struct("<I")
_UINT64LE: Final = struct.Struct("<Q")
_REGISTERED_INPUT_ROUTE_MARKER: Final = object()
_SYNTHETIC_INPUT_ROUTE_MARKER: Final = object()
_REGISTERED_EVIDENCE_ROUTE_MARKER: Final = object()
_SYNTHETIC_EVIDENCE_ROUTE_MARKER: Final = object()


class Experiment002EvidenceError(ValueError):
    """Validation provenance or evidence state violated the frozen contract."""


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
class RegisteredValidationInputs:
    """Issuer-only digest capability for one exact validation population."""

    _example_count: int
    _validation_inputs_sha256: str
    _route_marker: object

    def __init__(self) -> None:
        raise TypeError("RegisteredValidationInputs values are issued by this module")

    @property
    def example_count(self) -> int:
        """Return the bound population count after validating capability state."""

        return _issued_validation_inputs_state(self).example_count

    @property
    def validation_inputs_sha256(self) -> str:
        """Return the frozen custom-binary validation-input digest."""

        return _issued_validation_inputs_state(self).validation_inputs_sha256


@dataclass(frozen=True, slots=True)
class _IssuedValidationInputsState:
    corpus: Experiment002Corpus | None
    population: MaterializedValidationPopulation | None
    identity_bytes: tuple[bytes, ...]
    model_inputs: Float32Array
    label_indices: Int64Array
    example_count: int
    validation_inputs_sha256: str
    route_marker: object


_ISSUED_VALIDATION_INPUTS: weakref.WeakKeyDictionary[
    RegisteredValidationInputs, _IssuedValidationInputsState
] = weakref.WeakKeyDictionary()
_ISSUED_VALIDATION_INPUTS_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
class RegisteredValidationEvidence:
    """Issuer-only population-bound validation predictions and metric evidence."""

    _example_count: int
    _validation_inputs_sha256: str
    _validation_predictions_sha256: str
    _route_marker: object

    def __init__(self) -> None:
        raise TypeError("RegisteredValidationEvidence values are issued by this module")

    @property
    def example_count(self) -> int:
        """Return the number of population-bound predictions."""

        return _issued_validation_evidence_state(self).example_count

    @property
    def validation_inputs_sha256(self) -> str:
        """Return the digest of the exact population inputs used by this result."""

        return _issued_validation_evidence_state(self).validation_inputs_sha256

    @property
    def validation_predictions_sha256(self) -> str:
        """Return the frozen custom-binary validation-prediction digest."""

        return _issued_validation_evidence_state(self).validation_predictions_sha256

    @property
    def confusion_matrix(self) -> Int64Array:
        """Return a fresh immutable copy of the bound confusion matrix."""

        return _snapshot_int64(_issued_validation_evidence_state(self).confusion_matrix)

    @property
    def cross_entropy(self) -> np.float64:
        """Return the exact binary64 validation cross-entropy scalar."""

        return np.float64(_issued_validation_evidence_state(self).cross_entropy)

    @property
    def cross_entropy_hex(self) -> str:
        """Return the canonical lowercase CPython binary64 spelling."""

        return float(_issued_validation_evidence_state(self).cross_entropy).hex()

    @property
    def macro_f1(self) -> Fraction:
        """Return the exact reduced macro-F1 fraction."""

        return _issued_validation_evidence_state(self).macro_f1

    @property
    def predicted_indices(self) -> Int64Array:
        """Return a fresh immutable copy in registered population order."""

        return _snapshot_int64(
            _issued_validation_evidence_state(self).predicted_indices
        )


@dataclass(frozen=True, slots=True)
class _IssuedValidationEvidenceState:
    validation_inputs: RegisteredValidationInputs
    layout_metrics: ValidationMetrics
    frame_97_logits: Float32Array
    predicted_indices: Int64Array
    confusion_matrix: Int64Array
    cross_entropy: np.float64
    macro_f1: Fraction
    identity_bytes: tuple[bytes, ...]
    label_indices: Int64Array
    example_count: int
    validation_inputs_sha256: str
    validation_predictions_sha256: str
    route_marker: object


_ISSUED_VALIDATION_EVIDENCE: weakref.WeakKeyDictionary[
    RegisteredValidationEvidence, _IssuedValidationEvidenceState
] = weakref.WeakKeyDictionary()
_ISSUED_VALIDATION_EVIDENCE_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class _RegisteredValidationEvidenceSnapshot:
    """One atomic, fully verified state transfer to the history layer."""

    validation_inputs_sha256: str
    validation_predictions_sha256: str
    example_count: int
    confusion_matrix: tuple[tuple[int, ...], ...]
    cross_entropy: np.float64
    macro_f1: Fraction


def bind_registered_validation_inputs(
    corpus: Experiment002Corpus,
    population: MaterializedValidationPopulation,
) -> RegisteredValidationInputs:
    """Bind exact identities, labels, and features to registered provenance."""

    verify_registered_materialized_validation(corpus, population)
    result = _bind_registered_validation_inputs(corpus, population)
    verify_registered_materialized_validation(corpus, population)
    verify_registered_validation_inputs(result)
    return result


def verify_registered_validation_inputs(
    validation_inputs: RegisteredValidationInputs,
) -> None:
    """Revalidate original corpus/population issuance and every bound byte."""

    state = _issued_validation_inputs_state(validation_inputs)
    if state.route_marker is not _REGISTERED_INPUT_ROUTE_MARKER:
        raise Experiment002EvidenceError(
            "validation inputs were not issued through the registered route"
        )
    if type(state.corpus) is not Experiment002Corpus:
        raise Experiment002EvidenceError(
            "registered validation inputs have no exact corpus binding"
        )
    if type(state.population) is not MaterializedValidationPopulation:
        raise Experiment002EvidenceError(
            "registered validation inputs have no exact population binding"
        )

    first_digest = _validation_inputs_sha256(
        state.identity_bytes,
        state.label_indices,
        state.model_inputs,
    )
    verify_registered_materialized_validation(state.corpus, state.population)
    _require_population_matches_input_state(state)
    verify_registered_materialized_validation(state.corpus, state.population)
    _require_population_matches_input_state(state)
    second_digest = _validation_inputs_sha256(
        state.identity_bytes,
        state.label_indices,
        state.model_inputs,
    )
    if (
        not hmac.compare_digest(first_digest, second_digest)
        or not hmac.compare_digest(
            second_digest,
            state.validation_inputs_sha256,
        )
        or _issued_validation_inputs_state(validation_inputs) is not state
    ):
        raise Experiment002EvidenceError(
            "registered validation inputs changed during verification"
        )


def issue_registered_validation_evidence(
    validation_inputs: RegisteredValidationInputs,
    frame_97_logits: Float32Array,
    layout_metrics: ValidationMetrics,
) -> RegisteredValidationEvidence:
    """Issue evidence only for exact registered inputs and layout metrics."""

    verify_registered_validation_inputs(validation_inputs)
    verify_validation_layout_metrics(layout_metrics)
    result = _issue_registered_validation_evidence(
        validation_inputs,
        frame_97_logits,
        layout_metrics,
    )
    verify_registered_validation_inputs(validation_inputs)
    verify_validation_layout_metrics(layout_metrics)
    verify_registered_validation_evidence(result)
    return result


def verify_registered_validation_evidence(
    evidence: RegisteredValidationEvidence,
) -> None:
    """Revalidate complete population, prediction, and metric provenance."""

    state = _issued_validation_evidence_state(evidence)
    if state.route_marker is not _REGISTERED_EVIDENCE_ROUTE_MARKER:
        raise Experiment002EvidenceError(
            "validation evidence was not issued through the registered route"
        )
    first_prediction_digest = _validation_predictions_sha256(
        state.identity_bytes,
        state.label_indices,
        state.frame_97_logits,
        state.predicted_indices,
    )
    verify_registered_validation_inputs(state.validation_inputs)
    verify_validation_layout_metrics(state.layout_metrics)
    _require_evidence_metric_binding(state)
    verify_registered_validation_inputs(state.validation_inputs)
    verify_validation_layout_metrics(state.layout_metrics)
    _require_evidence_metric_binding(state)
    second_prediction_digest = _validation_predictions_sha256(
        state.identity_bytes,
        state.label_indices,
        state.frame_97_logits,
        state.predicted_indices,
    )
    if (
        not hmac.compare_digest(first_prediction_digest, second_prediction_digest)
        or not hmac.compare_digest(
            second_prediction_digest,
            state.validation_predictions_sha256,
        )
        or _issued_validation_evidence_state(evidence) is not state
    ):
        raise Experiment002EvidenceError(
            "registered validation evidence changed during verification"
        )


def _registered_validation_evidence_snapshot(
    evidence: RegisteredValidationEvidence,
) -> _RegisteredValidationEvidenceSnapshot:
    """Return one immutable history snapshot after full registered verification."""

    verify_registered_validation_evidence(evidence)
    state = _issued_validation_evidence_state(evidence)
    confusion = tuple(
        tuple(
            int(state.confusion_matrix[row, column]) for column in range(_CLASS_COUNT)
        )
        for row in range(_CLASS_COUNT)
    )
    snapshot = _RegisteredValidationEvidenceSnapshot(
        validation_inputs_sha256=state.validation_inputs_sha256,
        validation_predictions_sha256=state.validation_predictions_sha256,
        example_count=state.example_count,
        confusion_matrix=confusion,
        cross_entropy=np.float64(state.cross_entropy),
        macro_f1=state.macro_f1,
    )
    verify_registered_validation_evidence(evidence)
    if _issued_validation_evidence_state(evidence) is not state:
        raise Experiment002EvidenceError(
            "registered validation evidence changed while taking a snapshot"
        )
    return snapshot


def _bind_registered_validation_inputs(
    corpus: Experiment002Corpus,
    population: MaterializedValidationPopulation,
) -> RegisteredValidationInputs:
    """Implementation behind the public route; kept patchable by routing tests."""

    verify_registered_materialized_validation(corpus, population)
    identity_bytes = _population_identity_bytes(population)
    result = _issue_validation_inputs_capability(
        identity_bytes,
        population.model_inputs,
        population.label_indices,
        corpus=corpus,
        population=population,
        route_marker=_REGISTERED_INPUT_ROUTE_MARKER,
    )
    verify_registered_materialized_validation(corpus, population)
    _require_population_matches_input_state(_issued_validation_inputs_state(result))
    return result


def _bind_validation_inputs(
    identity_bytes: tuple[bytes, ...],
    model_inputs: Float32Array,
    label_indices: Int64Array,
) -> RegisteredValidationInputs:
    """Private tiny synthetic seam that can never carry the registered marker."""

    return _issue_validation_inputs_capability(
        identity_bytes,
        model_inputs,
        label_indices,
        corpus=None,
        population=None,
        route_marker=_SYNTHETIC_INPUT_ROUTE_MARKER,
    )


def _verify_synthetic_validation_inputs(
    validation_inputs: RegisteredValidationInputs,
) -> None:
    state = _issued_validation_inputs_state(validation_inputs)
    if state.route_marker is not _SYNTHETIC_INPUT_ROUTE_MARKER or (
        state.corpus is not None or state.population is not None
    ):
        raise Experiment002EvidenceError(
            "validation inputs were not issued by the synthetic seam"
        )


def _issue_validation_inputs_capability(
    identity_bytes: tuple[bytes, ...],
    model_inputs: Float32Array,
    label_indices: Int64Array,
    *,
    corpus: Experiment002Corpus | None,
    population: MaterializedValidationPopulation | None,
    route_marker: object,
) -> RegisteredValidationInputs:
    if route_marker is _REGISTERED_INPUT_ROUTE_MARKER:
        if type(corpus) is not Experiment002Corpus:
            raise TypeError("corpus must be an Experiment002Corpus")
        if type(population) is not MaterializedValidationPopulation:
            raise TypeError("population must be a MaterializedValidationPopulation")
    elif route_marker is _SYNTHETIC_INPUT_ROUTE_MARKER:
        if corpus is not None or population is not None:
            raise Experiment002EvidenceError(
                "synthetic validation inputs cannot bind registered provenance"
            )
    else:
        raise Experiment002EvidenceError("invalid validation-input route marker")

    _require_validation_input_payload(identity_bytes, model_inputs, label_indices)
    identities_snapshot = tuple(bytes(identity) for identity in identity_bytes)
    model_inputs_snapshot = _snapshot_float32(model_inputs)
    label_indices_snapshot = _snapshot_int64(label_indices)
    _require_validation_input_payload(
        identities_snapshot,
        model_inputs_snapshot,
        label_indices_snapshot,
    )
    digest = _validation_inputs_sha256(
        identities_snapshot,
        label_indices_snapshot,
        model_inputs_snapshot,
    )
    _require_validation_input_payload(identity_bytes, model_inputs, label_indices)
    if (
        identity_bytes != identities_snapshot
        or not _same_float32_bytes(model_inputs, model_inputs_snapshot)
        or not _same_int64_bytes(label_indices, label_indices_snapshot)
    ):
        raise Experiment002EvidenceError(
            "validation input bytes changed while taking the binding snapshot"
        )

    state = _IssuedValidationInputsState(
        corpus=corpus,
        population=population,
        identity_bytes=identities_snapshot,
        model_inputs=model_inputs_snapshot,
        label_indices=label_indices_snapshot,
        example_count=len(identities_snapshot),
        validation_inputs_sha256=digest,
        route_marker=route_marker,
    )
    result = object.__new__(RegisteredValidationInputs)
    object.__setattr__(result, "_example_count", state.example_count)
    object.__setattr__(
        result,
        "_validation_inputs_sha256",
        state.validation_inputs_sha256,
    )
    object.__setattr__(result, "_route_marker", state.route_marker)
    with _ISSUED_VALIDATION_INPUTS_LOCK:
        _ISSUED_VALIDATION_INPUTS[result] = state
    if _issued_validation_inputs_state(result) is not state:
        raise Experiment002EvidenceError(
            "validation input capability changed during issuance"
        )
    return result


def _issue_registered_validation_evidence(
    validation_inputs: RegisteredValidationInputs,
    frame_97_logits: Float32Array,
    layout_metrics: ValidationMetrics,
) -> RegisteredValidationEvidence:
    """Implementation behind the public route; kept patchable by routing tests."""

    verify_registered_validation_inputs(validation_inputs)
    verify_validation_layout_metrics(layout_metrics)
    return _issue_validation_evidence_capability(
        validation_inputs,
        frame_97_logits,
        layout_metrics,
        route_marker=_REGISTERED_EVIDENCE_ROUTE_MARKER,
    )


def _issue_validation_evidence(
    validation_inputs: RegisteredValidationInputs,
    frame_97_logits: Float32Array,
    layout_metrics: ValidationMetrics,
) -> RegisteredValidationEvidence:
    """Private tiny synthetic seam that can never carry the registered marker."""

    return _issue_validation_evidence_capability(
        validation_inputs,
        frame_97_logits,
        layout_metrics,
        route_marker=_SYNTHETIC_EVIDENCE_ROUTE_MARKER,
    )


def _verify_synthetic_validation_evidence(
    evidence: RegisteredValidationEvidence,
) -> None:
    state = _issued_validation_evidence_state(evidence)
    if state.route_marker is not _SYNTHETIC_EVIDENCE_ROUTE_MARKER:
        raise Experiment002EvidenceError(
            "validation evidence was not issued by the synthetic seam"
        )
    _verify_synthetic_validation_inputs(state.validation_inputs)
    _require_evidence_metric_binding(state)


def _issue_validation_evidence_capability(
    validation_inputs: RegisteredValidationInputs,
    frame_97_logits: Float32Array,
    layout_metrics: ValidationMetrics,
    *,
    route_marker: object,
) -> RegisteredValidationEvidence:
    input_state = _issued_validation_inputs_state(validation_inputs)
    if route_marker is _REGISTERED_EVIDENCE_ROUTE_MARKER:
        if input_state.route_marker is not _REGISTERED_INPUT_ROUTE_MARKER:
            raise Experiment002EvidenceError(
                "registered evidence requires registered validation inputs"
            )
        verify_registered_validation_inputs(validation_inputs)
        verify_validation_layout_metrics(layout_metrics)
    elif route_marker is _SYNTHETIC_EVIDENCE_ROUTE_MARKER:
        if input_state.route_marker is not _SYNTHETIC_INPUT_ROUTE_MARKER:
            raise Experiment002EvidenceError(
                "synthetic evidence requires synthetic validation inputs"
            )
        _verify_synthetic_validation_inputs(validation_inputs)
    else:
        raise Experiment002EvidenceError("invalid validation-evidence route marker")

    _require_frame_97_logits(frame_97_logits, input_state.example_count)
    verify_validation_metrics_inputs(
        layout_metrics,
        input_state.label_indices,
        frame_97_logits,
    )
    logits_snapshot = _snapshot_float32(frame_97_logits)
    verify_validation_metrics_inputs(
        layout_metrics,
        input_state.label_indices,
        logits_snapshot,
    )
    predicted_indices = layout_metrics.predicted_indices
    confusion_matrix = layout_metrics.confusion_matrix
    cross_entropy = layout_metrics.cross_entropy
    macro_f1 = layout_metrics.macro_f1
    predictions_snapshot = _snapshot_int64(predicted_indices)
    confusion_snapshot = _snapshot_int64(confusion_matrix)
    labels_snapshot = _snapshot_int64(input_state.label_indices)
    identities_snapshot = tuple(bytes(value) for value in input_state.identity_bytes)
    prediction_digest = _validation_predictions_sha256(
        identities_snapshot,
        labels_snapshot,
        logits_snapshot,
        predictions_snapshot,
    )

    state = _IssuedValidationEvidenceState(
        validation_inputs=validation_inputs,
        layout_metrics=layout_metrics,
        frame_97_logits=logits_snapshot,
        predicted_indices=predictions_snapshot,
        confusion_matrix=confusion_snapshot,
        cross_entropy=np.float64(cross_entropy),
        macro_f1=macro_f1,
        identity_bytes=identities_snapshot,
        label_indices=labels_snapshot,
        example_count=input_state.example_count,
        validation_inputs_sha256=input_state.validation_inputs_sha256,
        validation_predictions_sha256=prediction_digest,
        route_marker=route_marker,
    )
    _require_evidence_metric_binding(state)
    if route_marker is _REGISTERED_EVIDENCE_ROUTE_MARKER:
        verify_registered_validation_inputs(validation_inputs)
        verify_validation_layout_metrics(layout_metrics)
    else:
        _verify_synthetic_validation_inputs(validation_inputs)
    verify_validation_metrics_inputs(
        layout_metrics,
        labels_snapshot,
        logits_snapshot,
    )

    result = object.__new__(RegisteredValidationEvidence)
    object.__setattr__(result, "_example_count", state.example_count)
    object.__setattr__(
        result,
        "_validation_inputs_sha256",
        state.validation_inputs_sha256,
    )
    object.__setattr__(
        result,
        "_validation_predictions_sha256",
        state.validation_predictions_sha256,
    )
    object.__setattr__(result, "_route_marker", state.route_marker)
    with _ISSUED_VALIDATION_EVIDENCE_LOCK:
        _ISSUED_VALIDATION_EVIDENCE[result] = state
    if _issued_validation_evidence_state(result) is not state:
        raise Experiment002EvidenceError(
            "validation evidence capability changed during issuance"
        )
    return result


def _issued_validation_inputs_state(
    validation_inputs: RegisteredValidationInputs,
) -> _IssuedValidationInputsState:
    if type(validation_inputs) is not RegisteredValidationInputs:
        raise TypeError("validation_inputs must be RegisteredValidationInputs")
    with _ISSUED_VALIDATION_INPUTS_LOCK:
        state = _ISSUED_VALIDATION_INPUTS.get(validation_inputs)
    if state is None:
        raise Experiment002EvidenceError(
            "validation inputs were not issued by this process"
        )
    if type(state) is not _IssuedValidationInputsState:
        raise Experiment002EvidenceError("validation input issuer state is invalid")
    _validate_validation_inputs_state(state)
    try:
        object_count = validation_inputs._example_count
        object_digest = validation_inputs._validation_inputs_sha256
        object_marker = validation_inputs._route_marker
    except AttributeError as error:
        raise Experiment002EvidenceError(
            "validation input capability changed after issuance"
        ) from error
    if (
        type(object_count) is not int
        or object_count != state.example_count
        or type(object_digest) is not str
        or not hmac.compare_digest(object_digest, state.validation_inputs_sha256)
        or object_marker is not state.route_marker
    ):
        raise Experiment002EvidenceError(
            "validation input capability changed after issuance"
        )
    return state


def _issued_validation_evidence_state(
    evidence: RegisteredValidationEvidence,
) -> _IssuedValidationEvidenceState:
    if type(evidence) is not RegisteredValidationEvidence:
        raise TypeError("evidence must be RegisteredValidationEvidence")
    with _ISSUED_VALIDATION_EVIDENCE_LOCK:
        state = _ISSUED_VALIDATION_EVIDENCE.get(evidence)
    if state is None:
        raise Experiment002EvidenceError(
            "validation evidence was not issued by this process"
        )
    if type(state) is not _IssuedValidationEvidenceState:
        raise Experiment002EvidenceError("validation evidence issuer state is invalid")
    _validate_validation_evidence_state(state)
    try:
        object_count = evidence._example_count
        object_input_digest = evidence._validation_inputs_sha256
        object_prediction_digest = evidence._validation_predictions_sha256
        object_marker = evidence._route_marker
    except AttributeError as error:
        raise Experiment002EvidenceError(
            "validation evidence capability changed after issuance"
        ) from error
    if (
        type(object_count) is not int
        or object_count != state.example_count
        or type(object_input_digest) is not str
        or not hmac.compare_digest(
            object_input_digest,
            state.validation_inputs_sha256,
        )
        or type(object_prediction_digest) is not str
        or not hmac.compare_digest(
            object_prediction_digest,
            state.validation_predictions_sha256,
        )
        or object_marker is not state.route_marker
    ):
        raise Experiment002EvidenceError(
            "validation evidence capability changed after issuance"
        )
    return state


def _validate_validation_inputs_state(state: _IssuedValidationInputsState) -> None:
    if type(state.example_count) is not int or state.example_count < 1:
        raise Experiment002EvidenceError("validation input count is invalid")
    if state.example_count != len(state.identity_bytes):
        raise Experiment002EvidenceError(
            "validation input count differs from identity order"
        )
    _require_validation_input_payload(
        state.identity_bytes,
        state.model_inputs,
        state.label_indices,
    )
    _require_sha256(state.validation_inputs_sha256, "validation_inputs_sha256")
    observed_digest = _validation_inputs_sha256(
        state.identity_bytes,
        state.label_indices,
        state.model_inputs,
    )
    if not hmac.compare_digest(observed_digest, state.validation_inputs_sha256):
        raise Experiment002EvidenceError(
            "validation input digest differs from issuer state"
        )
    if state.route_marker is _REGISTERED_INPUT_ROUTE_MARKER:
        if (
            type(state.corpus) is not Experiment002Corpus
            or type(state.population) is not MaterializedValidationPopulation
        ):
            raise Experiment002EvidenceError(
                "registered validation input provenance is invalid"
            )
    elif state.route_marker is _SYNTHETIC_INPUT_ROUTE_MARKER:
        if state.corpus is not None or state.population is not None:
            raise Experiment002EvidenceError(
                "synthetic validation input provenance is invalid"
            )
    else:
        raise Experiment002EvidenceError("validation input route marker is invalid")


def _validate_validation_evidence_state(
    state: _IssuedValidationEvidenceState,
) -> None:
    if type(state.example_count) is not int or state.example_count < 1:
        raise Experiment002EvidenceError("validation evidence count is invalid")
    input_state = _issued_validation_inputs_state(state.validation_inputs)
    if (
        input_state.example_count != state.example_count
        or input_state.identity_bytes != state.identity_bytes
        or not _same_int64_bytes(input_state.label_indices, state.label_indices)
        or not hmac.compare_digest(
            input_state.validation_inputs_sha256,
            state.validation_inputs_sha256,
        )
    ):
        raise Experiment002EvidenceError(
            "validation evidence differs from its bound input capability"
        )
    _require_sha256(state.validation_inputs_sha256, "validation_inputs_sha256")
    _require_sha256(
        state.validation_predictions_sha256,
        "validation_predictions_sha256",
    )
    if state.route_marker is _REGISTERED_EVIDENCE_ROUTE_MARKER:
        if input_state.route_marker is not _REGISTERED_INPUT_ROUTE_MARKER:
            raise Experiment002EvidenceError(
                "registered evidence has unregistered validation inputs"
            )
        verify_validation_layout_metrics(state.layout_metrics)
    elif state.route_marker is _SYNTHETIC_EVIDENCE_ROUTE_MARKER:
        if input_state.route_marker is not _SYNTHETIC_INPUT_ROUTE_MARKER:
            raise Experiment002EvidenceError(
                "synthetic evidence has nonsynthetic validation inputs"
            )
    else:
        raise Experiment002EvidenceError("validation evidence route marker is invalid")
    _require_evidence_metric_binding(state)
    observed_digest = _validation_predictions_sha256(
        state.identity_bytes,
        state.label_indices,
        state.frame_97_logits,
        state.predicted_indices,
    )
    if not hmac.compare_digest(
        observed_digest,
        state.validation_predictions_sha256,
    ):
        raise Experiment002EvidenceError(
            "validation prediction digest differs from issuer state"
        )


def _require_population_matches_input_state(
    state: _IssuedValidationInputsState,
) -> None:
    if type(state.population) is not MaterializedValidationPopulation:
        raise Experiment002EvidenceError("validation input population is missing")
    current_identities = _population_identity_bytes(state.population)
    if (
        current_identities != state.identity_bytes
        or not _same_float32_bytes(
            state.population.model_inputs,
            state.model_inputs,
        )
        or not _same_int64_bytes(
            state.population.label_indices,
            state.label_indices,
        )
    ):
        raise Experiment002EvidenceError(
            "registered population bytes differ from validation input binding"
        )


def _require_evidence_metric_binding(
    state: _IssuedValidationEvidenceState,
) -> None:
    _require_frame_97_logits(state.frame_97_logits, state.example_count)
    _require_prediction_indices(state.predicted_indices, state.example_count)
    _require_confusion_matrix(state.confusion_matrix)
    if type(state.cross_entropy) is not np.float64 or not np.isfinite(
        state.cross_entropy
    ):
        raise Experiment002EvidenceError("evidence cross_entropy is invalid")
    if type(state.macro_f1) is not Fraction:
        raise TypeError("evidence macro_f1 must be a Fraction")
    verify_validation_metrics_inputs(
        state.layout_metrics,
        state.label_indices,
        state.frame_97_logits,
    )
    metric_predictions = state.layout_metrics.predicted_indices
    metric_confusion = state.layout_metrics.confusion_matrix
    metric_cross_entropy = state.layout_metrics.cross_entropy
    metric_macro_f1 = state.layout_metrics.macro_f1
    if (
        not _same_int64_bytes(metric_predictions, state.predicted_indices)
        or not _same_int64_bytes(metric_confusion, state.confusion_matrix)
        or type(metric_cross_entropy) is not np.float64
        or metric_cross_entropy.tobytes() != state.cross_entropy.tobytes()
        or type(metric_macro_f1) is not Fraction
        or metric_macro_f1 != state.macro_f1
    ):
        raise Experiment002EvidenceError(
            "validation evidence differs from its exact metric result"
        )

    expected_predictions = np.array(
        np.argmax(state.frame_97_logits, axis=1),
        dtype=np.int64,
        order="C",
        copy=True,
    )
    expected_predictions.setflags(write=False)
    if not _same_int64_bytes(expected_predictions, state.predicted_indices):
        raise Experiment002EvidenceError(
            "validation predictions differ from exact first-index argmax"
        )
    expected_confusion = np.zeros(
        (_CLASS_COUNT, _CLASS_COUNT),
        dtype=np.int64,
        order="C",
    )
    for row_index in range(state.example_count):
        expected_confusion[
            int(state.label_indices[row_index]),
            int(state.predicted_indices[row_index]),
        ] += np.int64(1)
    expected_confusion.setflags(write=False)
    if not _same_int64_bytes(expected_confusion, state.confusion_matrix):
        raise Experiment002EvidenceError(
            "validation confusion differs from exact labels and predictions"
        )


def _population_identity_bytes(
    population: MaterializedValidationPopulation,
) -> tuple[bytes, ...]:
    if type(population) is not MaterializedValidationPopulation:
        raise TypeError("population must be a MaterializedValidationPopulation")
    identities = tuple(example.identity for example in population.examples)
    if any(type(identity) is not bytes for identity in identities):
        raise Experiment002EvidenceError(
            "validation example identities must be exact bytes"
        )
    return identities


def _require_validation_input_payload(
    identity_bytes: tuple[bytes, ...],
    model_inputs: Float32Array,
    label_indices: Int64Array,
) -> None:
    if type(identity_bytes) is not tuple:
        raise TypeError("identity_bytes must be a tuple")
    if not 1 <= len(identity_bytes) <= _UINT32_MAX:
        raise Experiment002EvidenceError("validation identity count is invalid")
    if any(
        type(identity) is not bytes or len(identity) > _UINT32_MAX
        for identity in identity_bytes
    ):
        raise Experiment002EvidenceError(
            "validation identities have invalid concrete types or lengths"
        )
    if type(model_inputs) is not np.ndarray:
        raise TypeError("model_inputs must be a NumPy array")
    if model_inputs.dtype != np.dtype(np.float32) or model_inputs.ndim != 3:
        raise Experiment002EvidenceError("model_inputs have an invalid dtype or rank")
    if model_inputs.shape[0] != len(identity_bytes) or any(
        dimension < 1 for dimension in model_inputs.shape[1:]
    ):
        raise Experiment002EvidenceError("model_inputs have an invalid shape")
    if (
        not model_inputs.flags.c_contiguous
        or not model_inputs.flags.owndata
        or model_inputs.flags.writeable
    ):
        raise Experiment002EvidenceError(
            "model_inputs must be owned, C-contiguous, and read-only"
        )
    if not np.all(np.isfinite(model_inputs)):
        raise Experiment002EvidenceError("model_inputs contain non-finite values")
    if model_inputs[0].nbytes > _UINT64_MAX:
        raise Experiment002EvidenceError("model input row is too large to frame")

    if type(label_indices) is not np.ndarray:
        raise TypeError("label_indices must be a NumPy array")
    if label_indices.dtype != np.dtype(np.int64) or label_indices.shape != (
        len(identity_bytes),
    ):
        raise Experiment002EvidenceError("label_indices have an invalid shape or dtype")
    if (
        not label_indices.flags.c_contiguous
        or not label_indices.flags.owndata
        or label_indices.flags.writeable
    ):
        raise Experiment002EvidenceError(
            "label_indices must be owned, C-contiguous, and read-only"
        )
    if np.any(label_indices < 0) or np.any(label_indices >= _CLASS_COUNT):
        raise Experiment002EvidenceError("label_indices are outside class order")


def _require_frame_97_logits(
    frame_97_logits: Float32Array,
    example_count: int,
) -> None:
    if type(frame_97_logits) is not np.ndarray:
        raise TypeError("frame_97_logits must be a NumPy array")
    if frame_97_logits.dtype != np.dtype(np.float32) or (
        frame_97_logits.shape != (example_count, _CLASS_COUNT)
    ):
        raise Experiment002EvidenceError(
            "frame_97_logits have an invalid shape or dtype"
        )
    if (
        not frame_97_logits.flags.c_contiguous
        or not frame_97_logits.flags.owndata
        or frame_97_logits.flags.writeable
    ):
        raise Experiment002EvidenceError(
            "frame_97_logits must be owned, C-contiguous, and read-only"
        )
    if not np.all(np.isfinite(frame_97_logits)):
        raise Experiment002EvidenceError("frame_97_logits contain non-finite values")


def _require_prediction_indices(
    predicted_indices: Int64Array,
    example_count: int,
) -> None:
    if type(predicted_indices) is not np.ndarray:
        raise TypeError("predicted_indices must be a NumPy array")
    if predicted_indices.dtype != np.dtype(np.int64) or (
        predicted_indices.shape != (example_count,)
    ):
        raise Experiment002EvidenceError(
            "predicted_indices have an invalid shape or dtype"
        )
    if (
        not predicted_indices.flags.c_contiguous
        or not predicted_indices.flags.owndata
        or predicted_indices.flags.writeable
    ):
        raise Experiment002EvidenceError(
            "predicted_indices must be owned, C-contiguous, and read-only"
        )
    if np.any(predicted_indices < 0) or np.any(predicted_indices >= _CLASS_COUNT):
        raise Experiment002EvidenceError("predicted_indices are outside class order")


def _require_confusion_matrix(confusion_matrix: Int64Array) -> None:
    if type(confusion_matrix) is not np.ndarray:
        raise TypeError("confusion_matrix must be a NumPy array")
    if confusion_matrix.dtype != np.dtype(np.int64) or confusion_matrix.shape != (
        _CLASS_COUNT,
        _CLASS_COUNT,
    ):
        raise Experiment002EvidenceError(
            "confusion_matrix has an invalid shape or dtype"
        )
    if (
        not confusion_matrix.flags.c_contiguous
        or not confusion_matrix.flags.owndata
        or confusion_matrix.flags.writeable
    ):
        raise Experiment002EvidenceError(
            "confusion_matrix must be owned, C-contiguous, and read-only"
        )
    if np.any(confusion_matrix < 0):
        raise Experiment002EvidenceError("confusion_matrix contains a negative count")


def _validation_inputs_sha256(
    identity_bytes: tuple[bytes, ...],
    label_indices: Int64Array,
    model_inputs: Float32Array,
) -> str:
    """Hash the exact registered custom-binary validation-input framing."""

    _require_validation_input_payload(identity_bytes, model_inputs, label_indices)
    digest = hashlib.sha256(VALIDATION_INPUTS_DOMAIN)
    digest.update(_UINT32LE.pack(len(identity_bytes)))
    for index, identity in enumerate(identity_bytes):
        feature_payload = _little_endian_float32_bytes(model_inputs[index])
        digest.update(_UINT32LE.pack(len(identity)))
        digest.update(identity)
        digest.update(_UINT8.pack(int(label_indices[index])))
        digest.update(_UINT64LE.pack(len(feature_payload)))
        digest.update(feature_payload)
    return digest.hexdigest()


def _validation_predictions_sha256(
    identity_bytes: tuple[bytes, ...],
    label_indices: Int64Array,
    frame_97_logits: Float32Array,
    predicted_indices: Int64Array,
) -> str:
    """Hash the exact registered custom-binary prediction framing."""

    if type(identity_bytes) is not tuple or not 1 <= len(identity_bytes) <= _UINT32_MAX:
        raise Experiment002EvidenceError("prediction identity count is invalid")
    if any(
        type(identity) is not bytes or len(identity) > _UINT32_MAX
        for identity in identity_bytes
    ):
        raise Experiment002EvidenceError("prediction identities are invalid")
    _require_frame_97_logits(frame_97_logits, len(identity_bytes))
    _require_prediction_indices(predicted_indices, len(identity_bytes))
    if type(label_indices) is not np.ndarray:
        raise TypeError("label_indices must be a NumPy array")
    if label_indices.dtype != np.dtype(np.int64) or label_indices.shape != (
        len(identity_bytes),
    ):
        raise Experiment002EvidenceError(
            "prediction labels have an invalid shape or dtype"
        )
    if (
        not label_indices.flags.c_contiguous
        or not label_indices.flags.owndata
        or label_indices.flags.writeable
    ):
        raise Experiment002EvidenceError(
            "prediction labels must be owned, C-contiguous, and read-only"
        )
    if np.any(label_indices < 0) or np.any(label_indices >= _CLASS_COUNT):
        raise Experiment002EvidenceError("prediction labels are outside class order")

    digest = hashlib.sha256(VALIDATION_PREDICTIONS_DOMAIN)
    digest.update(_UINT32LE.pack(len(identity_bytes)))
    for index, identity in enumerate(identity_bytes):
        digest.update(_UINT32LE.pack(len(identity)))
        digest.update(identity)
        digest.update(_UINT8.pack(int(label_indices[index])))
        digest.update(_little_endian_float32_bytes(frame_97_logits[index]))
        digest.update(_UINT8.pack(int(predicted_indices[index])))
    return digest.hexdigest()


def _little_endian_float32_bytes(values: Float32Array) -> bytes:
    little_endian = np.asarray(
        values,
        dtype=np.dtype("<f4"),
        order="C",
    )
    return little_endian.tobytes(order="C")


def _snapshot_float32(values: Float32Array) -> Float32Array:
    if type(values) is not np.ndarray or values.dtype != np.dtype(np.float32):
        raise TypeError("float32 snapshot source must be a NumPy float32 array")
    snapshot = np.array(values, dtype=np.float32, order="C", copy=True)
    snapshot.setflags(write=False)
    return snapshot


def _snapshot_int64(values: Int64Array) -> Int64Array:
    if type(values) is not np.ndarray or values.dtype != np.dtype(np.int64):
        raise TypeError("int64 snapshot source must be a NumPy int64 array")
    snapshot = np.array(values, dtype=np.int64, order="C", copy=True)
    snapshot.setflags(write=False)
    return snapshot


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


def _require_sha256(value: str, name: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Experiment002EvidenceError(f"{name} must be lowercase SHA-256")
