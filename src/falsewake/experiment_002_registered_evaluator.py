"""Registered, source-bound validation evaluation for Experiment 002.

This module is the only bridge from a registered executor epoch handoff to
population-bound validation evidence.  The public route is deliberately
one-shot and fail-closed.  A small private kernel seam exists solely so the
model execution mechanics can be tested before the source-bound run
registration is committed; it cannot issue registered evidence.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import struct
import threading
import warnings
import weakref
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Final, Literal, NoReturn, SupportsIndex

import numpy as np
import torch
from numpy.typing import NDArray

from falsewake.causal_kws import (
    INPUT_MEL_BINS,
    MODEL_RECEPTIVE_FIELD_FRAMES,
    OUTPUT_CLASSES,
    CausalKWS,
)
from falsewake.experiment_002_evaluator import (
    _capture_model_parameter_authority,
    _capture_model_structure_authority,
    _gradient_snapshot,
    _require_direct_batch_slice,
    _require_direct_tensor_view,
    _require_logits,
    _require_model_parameter_authority,
    _require_model_ready_for_evaluation,
    _require_model_structure_authority,
    _require_retained_logits,
    _require_stream_state,
    _require_uniform_model_mode,
    _stable_model_tensor_sha256,
    _stream_state_payload,
    _torch_rng_sha256,
    _torch_rng_state,
)
from falsewake.experiment_002_evidence import (
    RegisteredValidationEvidence,
    RegisteredValidationInputs,
    _issued_validation_inputs_state,
    _registered_validation_evidence_snapshot,
    issue_registered_validation_evidence,
    verify_registered_validation_evidence,
    verify_registered_validation_inputs,
)
from falsewake.experiment_002_metrics import evaluate_validation_layout_logits
from falsewake.experiment_002_registered_executor import (
    RegisteredExecutorEpochHandoff,
    _fail_registered_epoch_handoff,
    _registered_epoch_handoff_model,
    _registered_epoch_handoff_snapshot,
    _RegisteredExecutorEpochHandoffSnapshot,
)
from falsewake.experiment_002_run_authority import (
    VerifiedRunRegistration,
    reverify_verified_run_registration,
)
from falsewake.experiment_002_training_evidence import (
    RegisteredModelTensorEvidence,
    capture_registered_model_tensors,
    verify_registered_model_tensor_evidence,
    verify_registered_model_tensors_match,
)

type Float32Array = NDArray[np.float32]
type _EvaluationPhase = Literal["EVALUATING", "COMPLETE", "FAILED"]
type _ResultPhase = Literal["COMPLETE", "FAILED"]

VALIDATION_EXAMPLE_COUNT: Final = 10_583
VALIDATION_BATCH_SIZE: Final = 128
VALIDATION_BATCH_COUNT: Final = 83
VALIDATION_FULL_BATCH_COUNT: Final = 82
VALIDATION_LAST_BATCH_SIZE: Final = 87
FINAL_ZERO_BASED_EPOCH: Final = 29
_MAX_SYNTHETIC_EXAMPLES: Final = 8
_FRAME_97: Final = MODEL_RECEPTIVE_FIELD_FRAMES - 1
_RESULT_AUTHORITY_DOMAIN: Final = b"falsewake-exp002-evaluated-epoch-v1\0"
_REGISTERED_ROUTE_MARKER: Final = object()
_SYNTHETIC_ROUTE_MARKER: Final = object()
_LOWER_HEX: Final = frozenset("0123456789abcdef")


class Experiment002RegisteredEvaluatorError(ValueError):
    """The registered evaluator violated its one-shot authority contract."""


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
class RegisteredEvaluatedEpoch:
    """Opaque evidence bundle for one exact registered epoch evaluation."""

    _seed: int
    _zero_based_epoch: int
    _validation_inputs_sha256: str
    _validation_predictions_sha256: str
    _model_tensor_sha256: str
    _authority_sha256: str
    _route_marker: object

    def __init__(self) -> None:
        raise TypeError("registered evaluated epochs are issuer-only")

    @property
    def seed(self) -> int:
        """Return the registered run seed after revalidating the capability."""

        return _issued_result_state(self).seed

    @property
    def zero_based_epoch(self) -> int:
        """Return the evaluated zero-based epoch after revalidation."""

        return _issued_result_state(self).zero_based_epoch

    @property
    def validation_inputs_sha256(self) -> str:
        """Return the exact registered validation-input digest."""

        return _issued_result_state(self).validation_inputs_sha256

    @property
    def validation_predictions_sha256(self) -> str:
        """Return the exact registered validation-prediction digest."""

        return _issued_result_state(self).validation_predictions_sha256

    @property
    def model_tensor_sha256(self) -> str:
        """Return the exact evaluated model-tensor digest."""

        return _issued_result_state(self).model_tensor_sha256

    @property
    def authority_sha256(self) -> str:
        """Return the digest binding every scalar source and evidence digest."""

        return _issued_result_state(self).authority_sha256

    def __copy__(self) -> NoReturn:
        raise TypeError("registered evaluated epochs cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        del memo
        raise TypeError("registered evaluated epochs cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("registered evaluated epochs cannot be serialized")

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        del protocol
        raise TypeError("registered evaluated epochs cannot be serialized")


@dataclass(frozen=True, slots=True)
class _SyntheticEvaluationLayout:
    """Private pre-registration layout, permanently capped at eight examples."""

    example_count: int
    batch_size: int

    def __post_init__(self) -> None:
        _require_positive_int(self.example_count, "example_count")
        _require_positive_int(self.batch_size, "batch_size")
        if self.example_count > _MAX_SYNTHETIC_EXAMPLES:
            raise Experiment002RegisteredEvaluatorError(
                "synthetic registered-evaluator kernel exceeds eight examples"
            )


@dataclass(frozen=True, slots=True)
class _KernelResult:
    example_count: int
    batch_sizes: tuple[int, ...]
    frame_97_logits_payload: bytes
    model_tensor_sha256: str
    torch_rng_sha256: str
    model_training_after: bool

    def _frame_97_logits(self) -> Float32Array:
        result = (
            np.frombuffer(self.frame_97_logits_payload, dtype=np.dtype("<f4"))
            .reshape(self.example_count, OUTPUT_CLASSES)
            .copy(order="C")
        )
        result.setflags(write=False)
        return result


@dataclass(slots=True)
class _HandoffEvaluationState:
    token: object
    process_id: int
    registration: VerifiedRunRegistration
    validation_inputs: RegisteredValidationInputs
    snapshot: _RegisteredExecutorEpochHandoffSnapshot
    phase: _EvaluationPhase
    result: RegisteredEvaluatedEpoch | None


@dataclass(frozen=True, slots=True)
class _HandoffEvaluationGuard:
    token: object
    process_id: int
    registration: VerifiedRunRegistration
    validation_inputs: RegisteredValidationInputs
    snapshot: _RegisteredExecutorEpochHandoffSnapshot


@dataclass(frozen=True, slots=True)
class _HandoffEvaluationLifecycle:
    token: object
    process_id: int
    phase: _EvaluationPhase
    result: RegisteredEvaluatedEpoch | None


@dataclass(slots=True)
class _HandoffIssuanceTruth:
    """Closure-owned append-only identity outside all public module registries."""

    handoff: RegisteredExecutorEpochHandoff
    state: _HandoffEvaluationState
    guard: _HandoffEvaluationGuard
    token: object
    process_id: int
    registration: VerifiedRunRegistration
    validation_inputs: RegisteredValidationInputs
    snapshot: _RegisteredExecutorEpochHandoffSnapshot
    snapshot_frame: tuple[object, ...]
    phase: _EvaluationPhase
    result: RegisteredEvaluatedEpoch | None


@dataclass(frozen=True, slots=True)
class _RegisteredEvaluatedEpochState:
    token: object
    route_marker: object
    process_id: int
    registration: VerifiedRunRegistration
    handoff: RegisteredExecutorEpochHandoff
    handoff_snapshot: _RegisteredExecutorEpochHandoffSnapshot
    validation_inputs: RegisteredValidationInputs
    validation_evidence: RegisteredValidationEvidence
    model_tensors: RegisteredModelTensorEvidence
    seed: int
    zero_based_epoch: int
    optimizer_generation: int
    batch_sizes: tuple[int, ...]
    registration_head_commit: str
    registration_sha256: str
    source_bundle_sha256: str
    validation_inputs_sha256: str
    validation_predictions_sha256: str
    model_tensor_sha256: str
    optimizer_sha256: str
    torch_rng_sha256: str
    authority_sha256: str


@dataclass(frozen=True, slots=True)
class _RegisteredEvaluatedEpochGuard:
    token: object
    route_marker: object
    process_id: int
    registration: VerifiedRunRegistration
    handoff: RegisteredExecutorEpochHandoff
    handoff_snapshot: _RegisteredExecutorEpochHandoffSnapshot
    validation_inputs: RegisteredValidationInputs
    validation_evidence: RegisteredValidationEvidence
    model_tensors: RegisteredModelTensorEvidence
    scalar_frame: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class _RegisteredEvaluatedEpochLifecycle:
    token: object
    process_id: int
    phase: _ResultPhase


@dataclass(slots=True)
class _ResultIssuanceTruth:
    """Closure-owned monotonic truth for one genuinely issued result object."""

    result: RegisteredEvaluatedEpoch
    state: _RegisteredEvaluatedEpochState
    guard: _RegisteredEvaluatedEpochGuard
    token: object
    process_id: int
    registration: VerifiedRunRegistration
    handoff: RegisteredExecutorEpochHandoff
    handoff_snapshot: _RegisteredExecutorEpochHandoffSnapshot
    validation_inputs: RegisteredValidationInputs
    validation_evidence: RegisteredValidationEvidence
    model_tensors: RegisteredModelTensorEvidence
    scalar_frame: tuple[object, ...]
    phase: _ResultPhase


@dataclass(frozen=True, slots=True)
class _RegisteredEvaluatedEpochSnapshot:
    """Verified capability transfer; it intentionally contains no raw tensors."""

    registration: VerifiedRunRegistration
    handoff: RegisteredExecutorEpochHandoff
    validation_inputs: RegisteredValidationInputs
    validation_evidence: RegisteredValidationEvidence
    model_tensors: RegisteredModelTensorEvidence
    process_id: int
    seed: int
    zero_based_epoch: int
    optimizer_generation: int
    batch_count: int
    another_training_epoch: bool
    registration_head_commit: str
    registration_sha256: str
    source_bundle_sha256: str
    validation_inputs_sha256: str
    validation_predictions_sha256: str
    model_tensor_sha256: str
    optimizer_sha256: str
    torch_rng_sha256: str
    authority_sha256: str


_HANDOFF_EVALUATIONS: weakref.WeakKeyDictionary[
    RegisteredExecutorEpochHandoff, _HandoffEvaluationState
] = weakref.WeakKeyDictionary()
_HANDOFF_EVALUATION_GUARDS: weakref.WeakKeyDictionary[
    RegisteredExecutorEpochHandoff, _HandoffEvaluationGuard
] = weakref.WeakKeyDictionary()
_HANDOFF_EVALUATION_LIFECYCLES: weakref.WeakKeyDictionary[
    RegisteredExecutorEpochHandoff, _HandoffEvaluationLifecycle
] = weakref.WeakKeyDictionary()
_ISSUED_HANDOFF_EVALUATIONS: weakref.WeakSet[RegisteredExecutorEpochHandoff] = (
    weakref.WeakSet()
)
_FAILED_HANDOFF_EVALUATIONS: weakref.WeakSet[RegisteredExecutorEpochHandoff] = (
    weakref.WeakSet()
)
_COMPLETED_HANDOFF_EVALUATIONS: weakref.WeakSet[RegisteredExecutorEpochHandoff] = (
    weakref.WeakSet()
)
_HANDOFF_EVALUATIONS_LOCK = threading.RLock()

_RESULTS: weakref.WeakKeyDictionary[
    RegisteredEvaluatedEpoch, _RegisteredEvaluatedEpochState
] = weakref.WeakKeyDictionary()
_RESULT_GUARDS: weakref.WeakKeyDictionary[
    RegisteredEvaluatedEpoch, _RegisteredEvaluatedEpochGuard
] = weakref.WeakKeyDictionary()
_RESULT_LIFECYCLES: weakref.WeakKeyDictionary[
    RegisteredEvaluatedEpoch, _RegisteredEvaluatedEpochLifecycle
] = weakref.WeakKeyDictionary()
_ISSUED_RESULTS: weakref.WeakSet[RegisteredEvaluatedEpoch] = weakref.WeakSet()
_FAILED_RESULTS: weakref.WeakSet[RegisteredEvaluatedEpoch] = weakref.WeakSet()
_RESULTS_LOCK = threading.RLock()


def _build_independent_issuance_truth() -> tuple[
    Callable[..., None],
    Callable[[RegisteredExecutorEpochHandoff], _EvaluationPhase | None],
    Callable[..., None],
    Callable[..., None],
    Callable[..., None],
    Callable[..., _ResultPhase],
    Callable[..., None],
]:
    """Create non-exported strong ledgers immune to registry replacement.

    The returned operations close over the only references to these dictionaries.
    Replacing every module-level weak registry, guard, lifecycle, or monotonic set
    therefore cannot mint a second issued identity or roll a terminal phase back.
    """

    lock = threading.RLock()
    handoffs: dict[RegisteredExecutorEpochHandoff, _HandoffIssuanceTruth] = {}
    results: dict[RegisteredEvaluatedEpoch, _ResultIssuanceTruth] = {}

    def snapshot_frame(
        snapshot: _RegisteredExecutorEpochHandoffSnapshot,
    ) -> tuple[object, ...]:
        runtime = snapshot.runtime_digests
        return (
            id(snapshot.executor),
            id(snapshot.registration),
            id(snapshot.validation_inputs),
            snapshot.process_id,
            id(snapshot.bridge_authority),
            id(snapshot.bridge_session_token),
            id(snapshot.population),
            id(snapshot.epoch_trace),
            snapshot.seed,
            snapshot.zero_based_epoch,
            snapshot.optimizer_generation,
            runtime.model_sha256,
            runtime.optimizer_sha256,
            runtime.rng_sha256,
            runtime.rng_state,
            id(snapshot.previous_history_barrier),
            id(snapshot.one_shot_token),
            snapshot.another_training_epoch,
        )

    def result_frame(
        state: _RegisteredEvaluatedEpochState,
    ) -> tuple[object, ...]:
        return (
            state.seed,
            state.zero_based_epoch,
            state.optimizer_generation,
            state.batch_sizes,
            state.registration_head_commit,
            state.registration_sha256,
            state.source_bundle_sha256,
            state.validation_inputs_sha256,
            state.validation_predictions_sha256,
            state.model_tensor_sha256,
            state.optimizer_sha256,
            state.torch_rng_sha256,
            state.authority_sha256,
        )

    def record_handoff(
        handoff: RegisteredExecutorEpochHandoff,
        state: _HandoffEvaluationState,
        guard: _HandoffEvaluationGuard,
    ) -> None:
        with lock:
            if handoff in handoffs:
                raise Experiment002RegisteredEvaluatorError(
                    "registered handoff already has issuance truth"
                )
            handoffs[handoff] = _HandoffIssuanceTruth(
                handoff=handoff,
                state=state,
                guard=guard,
                token=state.token,
                process_id=state.process_id,
                registration=state.registration,
                validation_inputs=state.validation_inputs,
                snapshot=state.snapshot,
                snapshot_frame=snapshot_frame(state.snapshot),
                phase="EVALUATING",
                result=None,
            )

    def handoff_phase(
        handoff: RegisteredExecutorEpochHandoff,
    ) -> _EvaluationPhase | None:
        with lock:
            truth = handoffs.get(handoff)
            return None if truth is None else truth.phase

    def require_handoff(
        handoff: RegisteredExecutorEpochHandoff,
        state: _HandoffEvaluationState,
        guard: _HandoffEvaluationGuard,
        phase: _EvaluationPhase,
        result: RegisteredEvaluatedEpoch | None,
    ) -> None:
        with lock:
            truth = handoffs.get(handoff)
            try:
                current_snapshot_frame = snapshot_frame(state.snapshot)
            except BaseException:
                if type(truth) is _HandoffIssuanceTruth:
                    truth.phase = "FAILED"
                    truth.result = None
                raise
            if (
                type(truth) is not _HandoffIssuanceTruth
                or truth.handoff is not handoff
                or truth.state is not state
                or truth.guard is not guard
                or truth.token is not state.token
                or truth.process_id != state.process_id
                or truth.registration is not state.registration
                or truth.validation_inputs is not state.validation_inputs
                or truth.snapshot is not state.snapshot
                or truth.snapshot_frame != current_snapshot_frame
                or truth.phase != phase
                or truth.result is not result
            ):
                if type(truth) is _HandoffIssuanceTruth:
                    truth.phase = "FAILED"
                    truth.result = None
                raise Experiment002RegisteredEvaluatorError(
                    "independent handoff issuance truth changed"
                )

    def transition_handoff(
        handoff: RegisteredExecutorEpochHandoff,
        state: _HandoffEvaluationState,
        *,
        expected: _EvaluationPhase,
        phase: _EvaluationPhase,
        result: RegisteredEvaluatedEpoch | None,
    ) -> None:
        with lock:
            truth = handoffs.get(handoff)
            if (
                type(truth) is not _HandoffIssuanceTruth
                or truth.state is not state
                or truth.token is not state.token
            ):
                if type(truth) is _HandoffIssuanceTruth:
                    truth.phase = "FAILED"
                    truth.result = None
                raise Experiment002RegisteredEvaluatorError(
                    "independent handoff issuance truth is missing"
                )
            if phase == "FAILED":
                truth.phase = "FAILED"
                truth.result = None
                return
            if truth.phase == "FAILED":
                raise Experiment002RegisteredEvaluatorError(
                    "failed handoff issuance truth cannot be rolled back"
                )
            if truth.phase != expected:
                raise Experiment002RegisteredEvaluatorError(
                    "handoff issuance truth transition is invalid"
                )
            if phase == "EVALUATING" or (phase == "COMPLETE" and result is None):
                raise Experiment002RegisteredEvaluatorError(
                    "handoff issuance truth target is invalid"
                )
            truth.phase = phase
            truth.result = result

    def record_result(
        result: RegisteredEvaluatedEpoch,
        state: _RegisteredEvaluatedEpochState,
        guard: _RegisteredEvaluatedEpochGuard,
    ) -> None:
        with lock:
            if result in results:
                raise Experiment002RegisteredEvaluatorError(
                    "registered result already has issuance truth"
                )
            results[result] = _ResultIssuanceTruth(
                result=result,
                state=state,
                guard=guard,
                token=state.token,
                process_id=state.process_id,
                registration=state.registration,
                handoff=state.handoff,
                handoff_snapshot=state.handoff_snapshot,
                validation_inputs=state.validation_inputs,
                validation_evidence=state.validation_evidence,
                model_tensors=state.model_tensors,
                scalar_frame=result_frame(state),
                phase="COMPLETE",
            )

    def require_result(
        result: RegisteredEvaluatedEpoch,
        state: _RegisteredEvaluatedEpochState | None,
        guard: _RegisteredEvaluatedEpochGuard | None,
    ) -> _ResultPhase:
        with lock:
            truth = results.get(result)
            try:
                current_result_frame = None if state is None else result_frame(state)
            except BaseException:
                if type(truth) is _ResultIssuanceTruth:
                    truth.phase = "FAILED"
                raise
            if (
                type(truth) is not _ResultIssuanceTruth
                or truth.result is not result
                or truth.state is not state
                or truth.guard is not guard
                or state is None
                or truth.token is not state.token
                or truth.process_id != state.process_id
                or truth.registration is not state.registration
                or truth.handoff is not state.handoff
                or truth.handoff_snapshot is not state.handoff_snapshot
                or truth.validation_inputs is not state.validation_inputs
                or truth.validation_evidence is not state.validation_evidence
                or truth.model_tensors is not state.model_tensors
                or truth.scalar_frame != current_result_frame
            ):
                if type(truth) is _ResultIssuanceTruth:
                    truth.phase = "FAILED"
                raise Experiment002RegisteredEvaluatorError(
                    "registered evaluated epoch has no independent issuance truth"
                )
            return truth.phase

    def fail_result(
        result: RegisteredEvaluatedEpoch,
        state: _RegisteredEvaluatedEpochState | None,
    ) -> None:
        with lock:
            truth = results.get(result)
            if truth is None:
                return
            if truth.state is not state:
                truth.phase = "FAILED"
                return
            truth.phase = "FAILED"

    return (
        record_handoff,
        handoff_phase,
        require_handoff,
        transition_handoff,
        record_result,
        require_result,
        fail_result,
    )


(
    _record_handoff_issuance_truth,
    _handoff_issuance_truth_phase,
    _require_handoff_issuance_truth,
    _transition_handoff_issuance_truth,
    _record_result_issuance_truth,
    _require_result_issuance_truth,
    _fail_result_issuance_truth,
) = _build_independent_issuance_truth()
del _build_independent_issuance_truth


def evaluate_registered_epoch(
    registration: VerifiedRunRegistration,
    handoff: RegisteredExecutorEpochHandoff,
    validation_inputs: RegisteredValidationInputs,
) -> RegisteredEvaluatedEpoch:
    """Evaluate exactly one registered epoch on the frozen validation layout."""

    if type(registration) is not VerifiedRunRegistration:
        raise TypeError("registration must be a VerifiedRunRegistration")
    if type(handoff) is not RegisteredExecutorEpochHandoff:
        raise TypeError("handoff must be a RegisteredExecutorEpochHandoff")
    if type(validation_inputs) is not RegisteredValidationInputs:
        raise TypeError("validation_inputs must be RegisteredValidationInputs")

    first_handoff_snapshot = _registered_epoch_handoff_snapshot(handoff)
    try:
        _require_handoff_arguments(
            registration,
            validation_inputs,
            first_handoff_snapshot,
        )
    except BaseException:
        _fail_registered_epoch_handoff(handoff)
        raise
    state = _claim_handoff_evaluation(
        handoff,
        registration,
        validation_inputs,
        first_handoff_snapshot,
    )
    try:
        reverify_verified_run_registration(registration)
        verify_registered_validation_inputs(validation_inputs)
        input_state = _issued_validation_inputs_state(validation_inputs)
        if input_state.example_count != VALIDATION_EXAMPLE_COUNT:
            raise Experiment002RegisteredEvaluatorError(
                "registered validation input count is not 10,583"
            )
        input_payload_before = input_state.model_inputs.tobytes(order="C")
        labels_payload_before = input_state.label_indices.tobytes(order="C")

        model = _registered_epoch_handoff_model(registration, handoff)
        if type(model) is not CausalKWS:
            raise Experiment002RegisteredEvaluatorError(
                "registered handoff did not return an exact CausalKWS"
            )
        outer_structure_authority = _capture_model_structure_authority(model)
        _require_model_ready_for_evaluation(model)
        outer_parameter_authority = _capture_model_parameter_authority(model)
        outer_gradients = _gradient_snapshot(model)
        rng_before = _torch_rng_state()
        model_tensors = capture_registered_model_tensors(
            model,
            seed=first_handoff_snapshot.seed,
            zero_based_epoch=first_handoff_snapshot.zero_based_epoch,
        )
        verify_registered_model_tensors_match(
            model,
            model_tensors,
            seed=first_handoff_snapshot.seed,
            zero_based_epoch=first_handoff_snapshot.zero_based_epoch,
        )
        if not hmac.compare_digest(
            model_tensors.sha256,
            first_handoff_snapshot.runtime_digests.model_sha256,
        ):
            raise Experiment002RegisteredEvaluatorError(
                "handoff model digest differs from registered tensor evidence"
            )

        kernel_result = _evaluate_exact_kernel(
            model,
            input_state.model_inputs,
            example_count=VALIDATION_EXAMPLE_COUNT,
            batch_size=VALIDATION_BATCH_SIZE,
            another_training_epoch=first_handoff_snapshot.another_training_epoch,
        )
        _require_registered_kernel_result(kernel_result, first_handoff_snapshot)
        verify_registered_model_tensors_match(
            model,
            model_tensors,
            seed=first_handoff_snapshot.seed,
            zero_based_epoch=first_handoff_snapshot.zero_based_epoch,
        )

        frame_97_logits = kernel_result._frame_97_logits()
        layout_metrics = evaluate_validation_layout_logits(
            input_state.label_indices,
            frame_97_logits,
        )
        validation_evidence = issue_registered_validation_evidence(
            validation_inputs,
            frame_97_logits,
            layout_metrics,
        )
        verify_registered_validation_evidence(validation_evidence)
        evidence_snapshot = _registered_validation_evidence_snapshot(
            validation_evidence
        )

        verify_registered_validation_inputs(validation_inputs)
        if _issued_validation_inputs_state(validation_inputs) is not input_state:
            raise Experiment002RegisteredEvaluatorError(
                "registered validation authority changed during evaluation"
            )
        if (
            input_state.model_inputs.tobytes(order="C") != input_payload_before
            or input_state.label_indices.tobytes(order="C") != labels_payload_before
        ):
            raise Experiment002RegisteredEvaluatorError(
                "registered validation bytes changed during evaluation"
            )
        verify_registered_model_tensors_match(
            model,
            model_tensors,
            seed=first_handoff_snapshot.seed,
            zero_based_epoch=first_handoff_snapshot.zero_based_epoch,
        )
        _require_uniform_model_mode(
            model,
            training=first_handoff_snapshot.another_training_epoch,
        )
        _require_model_structure_authority(model, outer_structure_authority)
        _require_model_parameter_authority(model, outer_parameter_authority)
        if _gradient_snapshot(model) != outer_gradients:
            raise Experiment002RegisteredEvaluatorError(
                "model gradients changed across registered evaluation"
            )
        if _torch_rng_state() != rng_before:
            raise Experiment002RegisteredEvaluatorError(
                "Torch CPU RNG changed across registered evaluation"
            )

        second_handoff_snapshot = _registered_epoch_handoff_snapshot(handoff)
        _require_same_handoff_snapshot(
            first_handoff_snapshot,
            second_handoff_snapshot,
        )
        reverify_verified_run_registration(registration)
        verify_registered_validation_evidence(validation_evidence)
        verify_registered_model_tensor_evidence(
            model_tensors,
            seed=first_handoff_snapshot.seed,
            zero_based_epoch=first_handoff_snapshot.zero_based_epoch,
        )
        result = _issue_registered_evaluated_epoch(
            registration=registration,
            handoff=handoff,
            handoff_snapshot=first_handoff_snapshot,
            validation_inputs=validation_inputs,
            validation_evidence=validation_evidence,
            model_tensors=model_tensors,
            validation_predictions_sha256=(
                evidence_snapshot.validation_predictions_sha256
            ),
            kernel_result=kernel_result,
        )
        _complete_handoff_evaluation(handoff, state, result)
        verify_registered_evaluated_epoch(result)
        return result
    except BaseException:
        try:
            _terminal_fail_handoff_evaluation(handoff, state)
        finally:
            _fail_registered_epoch_handoff(handoff)
        raise


def verify_registered_evaluated_epoch(result: RegisteredEvaluatedEpoch) -> None:
    """Revalidate every capability, identity, digest, and lifecycle binding."""

    state = _issued_result_state(result)
    try:
        reverify_verified_run_registration(state.registration)
        first_handoff_snapshot = _registered_epoch_handoff_snapshot(state.handoff)
        _require_same_handoff_snapshot(
            state.handoff_snapshot,
            first_handoff_snapshot,
        )
        verify_registered_validation_inputs(state.validation_inputs)
        verify_registered_validation_evidence(state.validation_evidence)
        verify_registered_model_tensor_evidence(
            state.model_tensors,
            seed=state.seed,
            zero_based_epoch=state.zero_based_epoch,
        )
        evidence_snapshot = _registered_validation_evidence_snapshot(
            state.validation_evidence
        )
        if (
            not hmac.compare_digest(
                evidence_snapshot.validation_inputs_sha256,
                state.validation_inputs_sha256,
            )
            or not hmac.compare_digest(
                evidence_snapshot.validation_predictions_sha256,
                state.validation_predictions_sha256,
            )
            or not hmac.compare_digest(
                state.validation_inputs.validation_inputs_sha256,
                state.validation_inputs_sha256,
            )
            or not hmac.compare_digest(
                state.model_tensors.sha256,
                state.model_tensor_sha256,
            )
            or not hmac.compare_digest(
                _authority_sha256(state),
                state.authority_sha256,
            )
        ):
            raise Experiment002RegisteredEvaluatorError(
                "registered evaluated-epoch evidence binding changed"
            )
        reverify_verified_run_registration(state.registration)
        second_handoff_snapshot = _registered_epoch_handoff_snapshot(state.handoff)
        _require_same_handoff_snapshot(
            first_handoff_snapshot,
            second_handoff_snapshot,
        )
        if _issued_result_state(result) is not state:
            raise Experiment002RegisteredEvaluatorError(
                "registered evaluated-epoch state changed during verification"
            )
    except BaseException:
        _terminal_fail_result(result, state)
        _terminal_fail_bound_handoff(state.handoff)
        raise


def _registered_evaluated_epoch_snapshot(
    result: RegisteredEvaluatedEpoch,
) -> _RegisteredEvaluatedEpochSnapshot:
    """Return verified opaque evidence capabilities and scalar bindings."""

    verify_registered_evaluated_epoch(result)
    state = _issued_result_state(result)
    snapshot = _RegisteredEvaluatedEpochSnapshot(
        registration=state.registration,
        handoff=state.handoff,
        validation_inputs=state.validation_inputs,
        validation_evidence=state.validation_evidence,
        model_tensors=state.model_tensors,
        process_id=state.process_id,
        seed=state.seed,
        zero_based_epoch=state.zero_based_epoch,
        optimizer_generation=state.optimizer_generation,
        batch_count=len(state.batch_sizes),
        another_training_epoch=state.zero_based_epoch < FINAL_ZERO_BASED_EPOCH,
        registration_head_commit=state.registration_head_commit,
        registration_sha256=state.registration_sha256,
        source_bundle_sha256=state.source_bundle_sha256,
        validation_inputs_sha256=state.validation_inputs_sha256,
        validation_predictions_sha256=state.validation_predictions_sha256,
        model_tensor_sha256=state.model_tensor_sha256,
        optimizer_sha256=state.optimizer_sha256,
        torch_rng_sha256=state.torch_rng_sha256,
        authority_sha256=state.authority_sha256,
    )
    verify_registered_evaluated_epoch(result)
    if _issued_result_state(result) is not state:
        raise Experiment002RegisteredEvaluatorError(
            "registered evaluated epoch changed while taking a snapshot"
        )
    return snapshot


def _evaluate_synthetic_kernel(
    model: CausalKWS,
    model_inputs: Float32Array,
    *,
    layout: _SyntheticEvaluationLayout,
    another_training_epoch: bool,
) -> _KernelResult:
    """Exercise only the model kernel; never issue registered evidence."""

    if type(layout) is not _SyntheticEvaluationLayout:
        raise TypeError("layout must be a _SyntheticEvaluationLayout")
    layout.__post_init__()
    return _evaluate_exact_kernel(
        model,
        model_inputs,
        example_count=layout.example_count,
        batch_size=layout.batch_size,
        another_training_epoch=another_training_epoch,
    )


def _evaluate_exact_kernel(
    model: CausalKWS,
    model_inputs: Float32Array,
    *,
    example_count: int,
    batch_size: int,
    another_training_epoch: bool,
) -> _KernelResult:
    if type(model) is not CausalKWS:
        raise TypeError("model must be an exact CausalKWS")
    _require_positive_int(example_count, "example_count")
    _require_positive_int(batch_size, "batch_size")
    if type(another_training_epoch) is not bool:
        raise TypeError("another_training_epoch must be a boolean")
    _require_exact_model_inputs(model_inputs, example_count)
    structure_authority = _capture_model_structure_authority(model)
    _require_model_ready_for_evaluation(model)
    parameter_authority = _capture_model_parameter_authority(model)
    input_payload_before = model_inputs.tobytes(order="C")
    model_sha256_before = _stable_model_tensor_sha256(model)
    gradients_before = _gradient_snapshot(model)
    rng_before = _torch_rng_state()

    frame_97_logits = np.empty(
        (example_count, OUTPUT_CLASSES),
        dtype=np.float32,
        order="C",
    )
    batch_sizes: list[int] = []
    try:
        model.eval()
        _require_uniform_model_mode(model, training=False)
        with torch.inference_mode():
            for start in range(0, example_count, batch_size):
                stop = min(start + batch_size, example_count)
                current_size = stop - start
                batch = model_inputs[start:stop]
                _require_direct_batch_slice(
                    batch,
                    model_inputs,
                    expected_size=current_size,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    features = torch.from_numpy(batch)
                _require_direct_tensor_view(features, batch)
                initial_state = model.initial_state(
                    current_size,
                    device=features.device,
                    dtype=features.dtype,
                )
                _require_stream_state(
                    initial_state,
                    batch_size=current_size,
                    expected_frames=0,
                    require_positive_zero=True,
                )
                initial_payload = _stream_state_payload(initial_state)
                logits, next_state = model.forward_stream(features, initial_state)
                _require_logits(logits, batch_size=current_size)
                _require_stream_state(
                    next_state,
                    batch_size=current_size,
                    expected_frames=MODEL_RECEPTIVE_FIELD_FRAMES,
                    require_positive_zero=False,
                )
                if _stream_state_payload(initial_state) != initial_payload:
                    raise Experiment002RegisteredEvaluatorError(
                        "model forward mutated its fresh initial state"
                    )
                retained = logits[:, _FRAME_97, :].contiguous()
                _require_retained_logits(retained, batch_size=current_size)
                frame_97_logits[start:stop] = retained.numpy()
                batch_sizes.append(current_size)
    finally:
        if another_training_epoch:
            model.train()
        else:
            model.eval()

    _require_uniform_model_mode(model, training=another_training_epoch)
    _require_model_structure_authority(model, structure_authority)
    _require_model_parameter_authority(model, parameter_authority)
    _require_exact_model_inputs(model_inputs, example_count)
    if model_inputs.tobytes(order="C") != input_payload_before:
        raise Experiment002RegisteredEvaluatorError(
            "model input bytes changed during evaluation"
        )
    model_sha256_after = _stable_model_tensor_sha256(model)
    if not hmac.compare_digest(model_sha256_before, model_sha256_after):
        raise Experiment002RegisteredEvaluatorError(
            "model tensor bytes changed during evaluation"
        )
    if _gradient_snapshot(model) != gradients_before:
        raise Experiment002RegisteredEvaluatorError(
            "model gradients changed during evaluation"
        )
    rng_after = _torch_rng_state()
    if rng_after != rng_before:
        raise Experiment002RegisteredEvaluatorError(
            "Torch CPU RNG changed during evaluation"
        )
    if not np.all(np.isfinite(frame_97_logits)):
        raise Experiment002RegisteredEvaluatorError(
            "frame-97 logits contain non-finite values"
        )
    expected_batch_sizes = tuple(
        min(batch_size, example_count - start)
        for start in range(0, example_count, batch_size)
    )
    if tuple(batch_sizes) != expected_batch_sizes:
        raise Experiment002RegisteredEvaluatorError(
            "evaluation batches did not cover the input exactly once"
        )
    payload = frame_97_logits.astype("<f4", copy=False).tobytes(order="C")
    if len(payload) != example_count * OUTPUT_CLASSES * 4:
        raise Experiment002RegisteredEvaluatorError(
            "frame-97 logits have an invalid byte count"
        )
    return _KernelResult(
        example_count=example_count,
        batch_sizes=tuple(batch_sizes),
        frame_97_logits_payload=payload,
        model_tensor_sha256=model_sha256_before,
        torch_rng_sha256=_torch_rng_sha256(rng_before),
        model_training_after=another_training_epoch,
    )


def _claim_handoff_evaluation(
    handoff: RegisteredExecutorEpochHandoff,
    registration: VerifiedRunRegistration,
    validation_inputs: RegisteredValidationInputs,
    snapshot: _RegisteredExecutorEpochHandoffSnapshot,
    _truth_phase: Callable[
        [RegisteredExecutorEpochHandoff], _EvaluationPhase | None
    ] = _handoff_issuance_truth_phase,
    _record_truth: Callable[..., None] = _record_handoff_issuance_truth,
) -> _HandoffEvaluationState:
    process_id = os.getpid()
    token = object()
    with _HANDOFF_EVALUATIONS_LOCK:
        truth_phase = _truth_phase(handoff)
        if truth_phase == "FAILED":
            raise Experiment002RegisteredEvaluatorError(
                "registered handoff evaluation is terminally failed"
            )
        if truth_phase == "COMPLETE":
            raise Experiment002RegisteredEvaluatorError(
                "registered handoff evaluation is one-shot"
            )
        if truth_phase == "EVALUATING":
            raise Experiment002RegisteredEvaluatorError(
                "registered handoff evaluation is already active"
            )
        existing = _HANDOFF_EVALUATIONS.get(handoff)
        if handoff in _FAILED_HANDOFF_EVALUATIONS:
            raise Experiment002RegisteredEvaluatorError(
                "registered handoff evaluation is terminally failed"
            )
        if handoff in _COMPLETED_HANDOFF_EVALUATIONS:
            raise Experiment002RegisteredEvaluatorError(
                "registered handoff evaluation is one-shot"
            )
        if existing is not None or handoff in _ISSUED_HANDOFF_EVALUATIONS:
            raise Experiment002RegisteredEvaluatorError(
                "registered handoff evaluation is already active"
            )
        state = _HandoffEvaluationState(
            token=token,
            process_id=process_id,
            registration=registration,
            validation_inputs=validation_inputs,
            snapshot=snapshot,
            phase="EVALUATING",
            result=None,
        )
        guard = _HandoffEvaluationGuard(
            token=token,
            process_id=process_id,
            registration=registration,
            validation_inputs=validation_inputs,
            snapshot=snapshot,
        )
        _record_truth(handoff, state, guard)
        _HANDOFF_EVALUATIONS[handoff] = state
        _HANDOFF_EVALUATION_GUARDS[handoff] = guard
        _HANDOFF_EVALUATION_LIFECYCLES[handoff] = _HandoffEvaluationLifecycle(
            token=token,
            process_id=process_id,
            phase="EVALUATING",
            result=None,
        )
        _ISSUED_HANDOFF_EVALUATIONS.add(handoff)
        return state


def _complete_handoff_evaluation(
    handoff: RegisteredExecutorEpochHandoff,
    state: _HandoffEvaluationState,
    result: RegisteredEvaluatedEpoch,
    _transition_truth: Callable[..., None] = _transition_handoff_issuance_truth,
) -> None:
    with _HANDOFF_EVALUATIONS_LOCK:
        _require_handoff_evaluation_authority(handoff, state)
        if state.phase != "EVALUATING" or state.result is not None:
            raise Experiment002RegisteredEvaluatorError(
                "registered handoff evaluation phase is invalid"
            )
        _transition_truth(
            handoff,
            state,
            expected="EVALUATING",
            phase="COMPLETE",
            result=result,
        )
        state.phase = "COMPLETE"
        state.result = result
        _HANDOFF_EVALUATION_LIFECYCLES[handoff] = _HandoffEvaluationLifecycle(
            token=state.token,
            process_id=state.process_id,
            phase="COMPLETE",
            result=result,
        )
        _COMPLETED_HANDOFF_EVALUATIONS.add(handoff)


def _terminal_fail_handoff_evaluation(
    handoff: RegisteredExecutorEpochHandoff,
    state: _HandoffEvaluationState,
    _transition_truth: Callable[..., None] = _transition_handoff_issuance_truth,
) -> None:
    with _HANDOFF_EVALUATIONS_LOCK:
        _FAILED_HANDOFF_EVALUATIONS.add(handoff)
        current = _HANDOFF_EVALUATIONS.get(handoff)
        if current is state:
            _transition_truth(
                handoff,
                state,
                expected=state.phase,
                phase="FAILED",
                result=None,
            )
            state.phase = "FAILED"
            state.result = None
            _HANDOFF_EVALUATION_LIFECYCLES[handoff] = _HandoffEvaluationLifecycle(
                token=state.token,
                process_id=state.process_id,
                phase="FAILED",
                result=None,
            )


def _terminal_fail_bound_handoff(
    handoff: RegisteredExecutorEpochHandoff,
    _transition_truth: Callable[..., None] = _transition_handoff_issuance_truth,
) -> None:
    try:
        with _HANDOFF_EVALUATIONS_LOCK:
            _FAILED_HANDOFF_EVALUATIONS.add(handoff)
            state = _HANDOFF_EVALUATIONS.get(handoff)
            if state is not None:
                _transition_truth(
                    handoff,
                    state,
                    expected=state.phase,
                    phase="FAILED",
                    result=None,
                )
                state.phase = "FAILED"
                state.result = None
                _HANDOFF_EVALUATION_LIFECYCLES[handoff] = _HandoffEvaluationLifecycle(
                    token=state.token,
                    process_id=state.process_id,
                    phase="FAILED",
                    result=None,
                )
    finally:
        _fail_registered_epoch_handoff(handoff)


def _require_handoff_evaluation_authority(
    handoff: RegisteredExecutorEpochHandoff,
    state: _HandoffEvaluationState,
    _require_truth: Callable[..., None] = _require_handoff_issuance_truth,
) -> None:
    guard = _HANDOFF_EVALUATION_GUARDS.get(handoff)
    lifecycle = _HANDOFF_EVALUATION_LIFECYCLES.get(handoff)
    if type(guard) is _HandoffEvaluationGuard:
        _require_truth(
            handoff,
            state,
            guard,
            state.phase,
            state.result,
        )
    if (
        handoff not in _ISSUED_HANDOFF_EVALUATIONS
        or handoff in _FAILED_HANDOFF_EVALUATIONS
        or _HANDOFF_EVALUATIONS.get(handoff) is not state
        or type(guard) is not _HandoffEvaluationGuard
        or type(lifecycle) is not _HandoffEvaluationLifecycle
        or state.token is not guard.token
        or state.token is not lifecycle.token
        or state.process_id != os.getpid()
        or state.process_id != guard.process_id
        or state.process_id != lifecycle.process_id
        or state.registration is not guard.registration
        or state.validation_inputs is not guard.validation_inputs
        or state.snapshot is not guard.snapshot
        or state.phase != lifecycle.phase
        or state.result is not lifecycle.result
    ):
        _FAILED_HANDOFF_EVALUATIONS.add(handoff)
        raise Experiment002RegisteredEvaluatorError(
            "registered handoff evaluation authority changed"
        )


def _issue_registered_evaluated_epoch(
    *,
    registration: VerifiedRunRegistration,
    handoff: RegisteredExecutorEpochHandoff,
    handoff_snapshot: _RegisteredExecutorEpochHandoffSnapshot,
    validation_inputs: RegisteredValidationInputs,
    validation_evidence: RegisteredValidationEvidence,
    model_tensors: RegisteredModelTensorEvidence,
    validation_predictions_sha256: str,
    kernel_result: _KernelResult,
    _record_truth: Callable[..., None] = _record_result_issuance_truth,
) -> RegisteredEvaluatedEpoch:
    token = object()
    process_id = os.getpid()
    provisional = _RegisteredEvaluatedEpochState(
        token=token,
        route_marker=_REGISTERED_ROUTE_MARKER,
        process_id=process_id,
        registration=registration,
        handoff=handoff,
        handoff_snapshot=handoff_snapshot,
        validation_inputs=validation_inputs,
        validation_evidence=validation_evidence,
        model_tensors=model_tensors,
        seed=handoff_snapshot.seed,
        zero_based_epoch=handoff_snapshot.zero_based_epoch,
        optimizer_generation=handoff_snapshot.optimizer_generation,
        batch_sizes=kernel_result.batch_sizes,
        registration_head_commit=registration.head_commit,
        registration_sha256=registration.registration_sha256,
        source_bundle_sha256=registration.source_bundle_sha256,
        validation_inputs_sha256=validation_inputs.validation_inputs_sha256,
        validation_predictions_sha256=validation_predictions_sha256,
        model_tensor_sha256=model_tensors.sha256,
        optimizer_sha256=handoff_snapshot.runtime_digests.optimizer_sha256,
        torch_rng_sha256=kernel_result.torch_rng_sha256,
        authority_sha256="",
    )
    state = _replace_authority_sha256(provisional)
    result = object.__new__(RegisteredEvaluatedEpoch)
    object.__setattr__(result, "_seed", state.seed)
    object.__setattr__(result, "_zero_based_epoch", state.zero_based_epoch)
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
    object.__setattr__(result, "_model_tensor_sha256", state.model_tensor_sha256)
    object.__setattr__(result, "_authority_sha256", state.authority_sha256)
    object.__setattr__(result, "_route_marker", _REGISTERED_ROUTE_MARKER)
    guard = _RegisteredEvaluatedEpochGuard(
        token=token,
        route_marker=_REGISTERED_ROUTE_MARKER,
        process_id=process_id,
        registration=registration,
        handoff=handoff,
        handoff_snapshot=handoff_snapshot,
        validation_inputs=validation_inputs,
        validation_evidence=validation_evidence,
        model_tensors=model_tensors,
        scalar_frame=_result_scalar_frame(state),
    )
    _record_truth(result, state, guard)
    with _RESULTS_LOCK:
        _RESULTS[result] = state
        _RESULT_GUARDS[result] = guard
        _RESULT_LIFECYCLES[result] = _RegisteredEvaluatedEpochLifecycle(
            token=token,
            process_id=process_id,
            phase="COMPLETE",
        )
        _ISSUED_RESULTS.add(result)
    return result


def _issued_result_state(
    result: RegisteredEvaluatedEpoch,
    _require_truth: Callable[..., _ResultPhase] = _require_result_issuance_truth,
) -> _RegisteredEvaluatedEpochState:
    if type(result) is not RegisteredEvaluatedEpoch:
        raise TypeError("result must be a RegisteredEvaluatedEpoch")
    with _RESULTS_LOCK:
        state = _RESULTS.get(result)
        guard = _RESULT_GUARDS.get(result)
        lifecycle = _RESULT_LIFECYCLES.get(result)
        issued = result in _ISSUED_RESULTS
        failed = result in _FAILED_RESULTS
    if failed:
        raise Experiment002RegisteredEvaluatorError(
            "registered evaluated epoch is terminally failed"
        )
    try:
        truth_phase = _require_truth(result, state, guard)
    except BaseException:
        with _RESULTS_LOCK:
            _FAILED_RESULTS.add(result)
        raise
    if truth_phase == "FAILED":
        raise Experiment002RegisteredEvaluatorError(
            "registered evaluated epoch is terminally failed"
        )
    if (
        not issued
        or type(state) is not _RegisteredEvaluatedEpochState
        or type(guard) is not _RegisteredEvaluatedEpochGuard
        or type(lifecycle) is not _RegisteredEvaluatedEpochLifecycle
    ):
        _terminal_fail_result(result, state)
        if type(state) is _RegisteredEvaluatedEpochState:
            _terminal_fail_bound_handoff(state.handoff)
        raise Experiment002RegisteredEvaluatorError(
            "registered evaluated epoch was not issued by this process"
        )
    try:
        _require_result_authority(result, state, guard, lifecycle)
    except BaseException:
        try:
            _terminal_fail_result(result, state)
        finally:
            _terminal_fail_bound_handoff(state.handoff)
        raise
    return state


def _require_result_authority(
    result: RegisteredEvaluatedEpoch,
    state: _RegisteredEvaluatedEpochState,
    guard: _RegisteredEvaluatedEpochGuard,
    lifecycle: _RegisteredEvaluatedEpochLifecycle,
) -> None:
    if (
        state.token is not guard.token
        or state.token is not lifecycle.token
        or state.route_marker is not _REGISTERED_ROUTE_MARKER
        or guard.route_marker is not _REGISTERED_ROUTE_MARKER
        or lifecycle.phase != "COMPLETE"
        or state.process_id != os.getpid()
        or state.process_id != guard.process_id
        or state.process_id != lifecycle.process_id
        or state.registration is not guard.registration
        or state.handoff is not guard.handoff
        or state.handoff_snapshot is not guard.handoff_snapshot
        or state.validation_inputs is not guard.validation_inputs
        or state.validation_evidence is not guard.validation_evidence
        or state.model_tensors is not guard.model_tensors
        or _result_scalar_frame(state) != guard.scalar_frame
    ):
        raise Experiment002RegisteredEvaluatorError(
            "registered evaluated-epoch authority changed"
        )
    try:
        object_frame = (
            result._seed,
            result._zero_based_epoch,
            result._validation_inputs_sha256,
            result._validation_predictions_sha256,
            result._model_tensor_sha256,
            result._authority_sha256,
            result._route_marker,
        )
    except AttributeError as error:
        raise Experiment002RegisteredEvaluatorError(
            "registered evaluated-epoch capability is incomplete"
        ) from error
    expected_frame = (
        state.seed,
        state.zero_based_epoch,
        state.validation_inputs_sha256,
        state.validation_predictions_sha256,
        state.model_tensor_sha256,
        state.authority_sha256,
        _REGISTERED_ROUTE_MARKER,
    )
    if object_frame != expected_frame:
        raise Experiment002RegisteredEvaluatorError(
            "registered evaluated-epoch capability changed after issuance"
        )


def _terminal_fail_result(
    result: RegisteredEvaluatedEpoch,
    state: _RegisteredEvaluatedEpochState | None,
    _fail_truth: Callable[..., None] = _fail_result_issuance_truth,
) -> None:
    with _RESULTS_LOCK:
        _FAILED_RESULTS.add(result)
        _fail_truth(result, state)
        if type(state) is _RegisteredEvaluatedEpochState:
            _RESULT_LIFECYCLES[result] = _RegisteredEvaluatedEpochLifecycle(
                token=state.token,
                process_id=state.process_id,
                phase="FAILED",
            )


def _replace_authority_sha256(
    state: _RegisteredEvaluatedEpochState,
) -> _RegisteredEvaluatedEpochState:
    return replace(state, authority_sha256=_authority_sha256(state))


def _authority_sha256(state: _RegisteredEvaluatedEpochState) -> str:
    payload = bytearray(_RESULT_AUTHORITY_DOMAIN)
    for value in (
        state.registration_head_commit,
        state.registration_sha256,
        state.source_bundle_sha256,
        state.validation_inputs_sha256,
        state.validation_predictions_sha256,
        state.model_tensor_sha256,
        state.optimizer_sha256,
        state.torch_rng_sha256,
    ):
        encoded = value.encode("ascii")
        payload.extend(struct.pack("<I", len(encoded)))
        payload.extend(encoded)
    payload.extend(struct.pack("<I", state.seed))
    payload.extend(struct.pack("<I", state.zero_based_epoch))
    payload.extend(struct.pack("<Q", state.optimizer_generation))
    payload.extend(struct.pack("<I", len(state.batch_sizes)))
    for size in state.batch_sizes:
        payload.extend(struct.pack("<I", size))
    return hashlib.sha256(payload).hexdigest()


def _result_scalar_frame(
    state: _RegisteredEvaluatedEpochState,
) -> tuple[object, ...]:
    return (
        state.seed,
        state.zero_based_epoch,
        state.optimizer_generation,
        state.batch_sizes,
        state.registration_head_commit,
        state.registration_sha256,
        state.source_bundle_sha256,
        state.validation_inputs_sha256,
        state.validation_predictions_sha256,
        state.model_tensor_sha256,
        state.optimizer_sha256,
        state.torch_rng_sha256,
        state.authority_sha256,
    )


def _require_handoff_arguments(
    registration: VerifiedRunRegistration,
    validation_inputs: RegisteredValidationInputs,
    snapshot: _RegisteredExecutorEpochHandoffSnapshot,
) -> None:
    if type(snapshot) is not _RegisteredExecutorEpochHandoffSnapshot:
        raise Experiment002RegisteredEvaluatorError(
            "registered handoff snapshot has an invalid type"
        )
    if snapshot.registration is not registration:
        raise Experiment002RegisteredEvaluatorError(
            "handoff belongs to a different run registration"
        )
    if snapshot.validation_inputs is not validation_inputs:
        raise Experiment002RegisteredEvaluatorError(
            "handoff belongs to different validation inputs"
        )
    if snapshot.process_id != os.getpid():
        raise Experiment002RegisteredEvaluatorError(
            "handoff belongs to a different process"
        )
    if snapshot.zero_based_epoch < 0 or (
        snapshot.zero_based_epoch > FINAL_ZERO_BASED_EPOCH
    ):
        raise Experiment002RegisteredEvaluatorError(
            "handoff epoch is outside the registered range"
        )
    expected_mode = snapshot.zero_based_epoch < FINAL_ZERO_BASED_EPOCH
    if snapshot.another_training_epoch is not expected_mode:
        raise Experiment002RegisteredEvaluatorError(
            "handoff post-evaluation model mode is invalid"
        )
    if snapshot.optimizer_generation != (snapshot.zero_based_epoch + 1) * 313:
        raise Experiment002RegisteredEvaluatorError(
            "handoff optimizer generation differs from the completed epoch"
        )
    _require_sha256(snapshot.runtime_digests.model_sha256, "model_sha256")
    _require_sha256(snapshot.runtime_digests.optimizer_sha256, "optimizer_sha256")
    _require_sha256(snapshot.runtime_digests.rng_sha256, "rng_sha256")


def _require_same_handoff_snapshot(
    left: _RegisteredExecutorEpochHandoffSnapshot,
    right: _RegisteredExecutorEpochHandoffSnapshot,
) -> None:
    if (
        type(left) is not _RegisteredExecutorEpochHandoffSnapshot
        or type(right) is not _RegisteredExecutorEpochHandoffSnapshot
        or left.executor is not right.executor
        or left.registration is not right.registration
        or left.validation_inputs is not right.validation_inputs
        or left.process_id != right.process_id
        or left.bridge_authority is not right.bridge_authority
        or left.bridge_session_token is not right.bridge_session_token
        or left.population is not right.population
        or left.epoch_trace is not right.epoch_trace
        or left.seed != right.seed
        or left.zero_based_epoch != right.zero_based_epoch
        or left.optimizer_generation != right.optimizer_generation
        or left.runtime_digests != right.runtime_digests
        or left.previous_history_barrier is not right.previous_history_barrier
        or left.one_shot_token is not right.one_shot_token
        or left.another_training_epoch is not right.another_training_epoch
    ):
        raise Experiment002RegisteredEvaluatorError(
            "registered handoff changed during evaluation"
        )


def _require_registered_kernel_result(
    result: _KernelResult,
    handoff_snapshot: _RegisteredExecutorEpochHandoffSnapshot,
) -> None:
    expected_sizes = (VALIDATION_BATCH_SIZE,) * VALIDATION_FULL_BATCH_COUNT + (
        VALIDATION_LAST_BATCH_SIZE,
    )
    expected_mode = handoff_snapshot.zero_based_epoch < FINAL_ZERO_BASED_EPOCH
    if (
        result.example_count != VALIDATION_EXAMPLE_COUNT
        or len(result.batch_sizes) != VALIDATION_BATCH_COUNT
        or result.batch_sizes != expected_sizes
        or result.model_training_after is not expected_mode
        or not hmac.compare_digest(
            result.model_tensor_sha256,
            handoff_snapshot.runtime_digests.model_sha256,
        )
        or not hmac.compare_digest(
            result.torch_rng_sha256,
            handoff_snapshot.runtime_digests.rng_sha256,
        )
    ):
        raise Experiment002RegisteredEvaluatorError(
            "registered evaluation kernel result differs from the frozen contract"
        )


def _require_exact_model_inputs(
    model_inputs: Float32Array,
    example_count: int,
) -> None:
    if type(model_inputs) is not np.ndarray:
        raise TypeError("model_inputs must be an exact NumPy array")
    if (
        model_inputs.dtype != np.dtype(np.float32)
        or model_inputs.shape
        != (example_count, INPUT_MEL_BINS, MODEL_RECEPTIVE_FIELD_FRAMES)
        or not model_inputs.flags.c_contiguous
        or not model_inputs.flags.owndata
        or model_inputs.flags.writeable
    ):
        raise Experiment002RegisteredEvaluatorError(
            "model inputs must be exact owned read-only contiguous float32"
        )
    if not np.all(np.isfinite(model_inputs)):
        raise Experiment002RegisteredEvaluatorError(
            "model inputs contain non-finite values"
        )


def _require_sha256(value: object, name: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise Experiment002RegisteredEvaluatorError(f"{name} must be lowercase SHA-256")


def _require_positive_int(value: object, name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise Experiment002RegisteredEvaluatorError(f"{name} must be positive")


# The bounded producers and all validation/transition/failure paths captured
# these closure-owned operations in their defaults.  Delete the temporary
# aliases so no caller can directly bless a counterfeit identity or replace
# the authority used by an already-defined critical path.
del _record_handoff_issuance_truth
del _handoff_issuance_truth_phase
del _require_handoff_issuance_truth
del _transition_handoff_issuance_truth
del _record_result_issuance_truth
del _require_result_issuance_truth
del _fail_result_issuance_truth
