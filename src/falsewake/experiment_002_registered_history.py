"""Source-bound, exact thirty-epoch training history for Experiment 002.

The public route in this module accepts only opaque evaluated-epoch evidence
issued by the registered evaluator.  Every accepted epoch produces a single
executor barrier.  The final history can be completed only by presenting the
registered complete update-trace capability containing the exact thirty epoch
trace capabilities already accepted here.

This module serializes no files.  Its completed capability owns the canonical
history bytes and preserves the winning evaluated-epoch capability for the
later checkpoint layer.

Threat model: the registered run executes frozen, reviewed source in a trusted
process.  The redundant stores and closure-owned mirrors below detect mutation
of module stores, lifecycle mirrors, and opaque capability slots within that
contract.  Arbitrary same-process reflection into function defaults or closure
cells is privileged code execution and is explicitly outside this boundary.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import os
import threading
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from typing import Final, Literal, NoReturn, SupportsIndex, cast

import numpy as np

from falsewake.experiment_002_evidence import (
    RegisteredValidationInputs,
    _registered_validation_evidence_snapshot,
    verify_registered_validation_evidence,
    verify_registered_validation_inputs,
)
from falsewake.experiment_002_registered_evaluator import (
    RegisteredEvaluatedEpoch,
    _registered_evaluated_epoch_snapshot,
    _RegisteredEvaluatedEpochSnapshot,
    verify_registered_evaluated_epoch,
)
from falsewake.experiment_002_registered_executor import (
    RegisteredExecutorEpochHandoff,
    RegisteredTrainingExecutor,
    _registered_epoch_handoff_snapshot,
    _registered_executor_snapshot,
    _RegisteredExecutorEpochHandoffSnapshot,
)
from falsewake.experiment_002_run_authority import (
    VerifiedRunRegistration,
    reverify_verified_run_registration,
)
from falsewake.experiment_002_training_evidence import (
    EPOCH_COUNT,
    REGISTERED_SEEDS,
    TOTAL_UPDATE_COUNT,
    TRAINING_EXAMPLE_COUNT,
    UPDATES_PER_EPOCH,
    CompleteUpdateTraceEvidence,
    EpochUpdateTraceEvidence,
    RegisteredModelTensorEvidence,
    verify_registered_complete_update_trace,
    verify_registered_epoch_update_trace,
    verify_registered_model_tensor_evidence,
)
from falsewake.experiment_002_training_history import (
    _EXPERIMENT,
    _HISTORY_DOMAIN,
    _SCHEMA_VERSION,
    _epoch_document,
    _EpochRecord,
    _macro_f1_from_confusion,
    _require_json_without_floats,
)
from falsewake.experiment_002_training_population import (
    REGISTERED_BATCH_COUNT,
    CompletedTrainingPopulation,
    verify_completed_registered_training_population,
)

type _HistoryPhase = Literal["OPEN", "COMPLETE", "FAILED"]
type _BarrierPhase = Literal["ISSUED", "CONSUMED", "RETIRED", "FAILED"]
type _CompletedPhase = Literal["COMPLETE", "FAILED"]
type _ConfusionMatrix = tuple[tuple[int, ...], ...]

VALIDATION_EXAMPLE_COUNT: Final = 10_583
VALIDATION_BATCH_COUNT: Final = 83
_CLASS_COUNT: Final = 12
_FINAL_ZERO_BASED_EPOCH: Final = EPOCH_COUNT - 1
_REGISTERED_ROUTE_MARKER: Final = object()
_HISTORY_RECORD_DOMAIN: Final = b"falsewake-exp002-registered-history-record-v1\0"
_LOWER_HEX: Final = frozenset("0123456789abcdef")


class Experiment002RegisteredHistoryError(ValueError):
    """The registered history violated its source-bound authority contract."""


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
class RegisteredTrainingHistory:
    """Opaque owner of one process-local, ordered thirty-epoch history."""

    _seed: int
    _token: object
    _route_marker: object

    def __init__(self) -> None:
        raise TypeError("registered training histories are issuer-only")

    def consume(
        self,
        evaluated_epoch: RegisteredEvaluatedEpoch,
    ) -> RegisteredHistoryBarrier:
        """Accept the exact next evaluated epoch and issue its one-use barrier."""

        return consume_registered_evaluated_epoch(self, evaluated_epoch)

    def complete(
        self,
        final_barrier: RegisteredHistoryBarrier,
        complete_update_trace: CompleteUpdateTraceEvidence,
    ) -> RegisteredCompletedTrainingHistory:
        """Complete after epoch 29 with the exact registered whole trace."""

        return complete_registered_training_history(
            self,
            final_barrier,
            complete_update_trace,
        )

    def __copy__(self) -> NoReturn:
        raise TypeError("registered training histories cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        del memo
        raise TypeError("registered training histories cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("registered training histories cannot be serialized")

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        del protocol
        raise TypeError("registered training histories cannot be serialized")


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
class RegisteredHistoryBarrier:
    """Opaque one-use proof that history accepted one exact epoch."""

    _seed: int
    _accepted_epoch: int
    _token: object
    _route_marker: object

    def __init__(self) -> None:
        raise TypeError("registered history barriers are issuer-only")

    def __copy__(self) -> NoReturn:
        raise TypeError("registered history barriers cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        del memo
        raise TypeError("registered history barriers cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("registered history barriers cannot be serialized")

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        del protocol
        raise TypeError("registered history barriers cannot be serialized")


@dataclass(frozen=True, slots=True)
class RegisteredEpochRank:
    """Immutable rank metadata; authority remains in the completed history."""

    zero_based_epoch: int
    macro_f1_exact_numerator: int
    macro_f1_exact_denominator: int
    validation_cross_entropy_float64_hex: str


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
class RegisteredCompletedTrainingHistory:
    """Opaque canonical history retaining the exact winning epoch evidence."""

    _seed: int
    _history_sha256: str
    _winner_epoch: int
    _token: object
    _route_marker: object

    def __init__(self) -> None:
        raise TypeError("completed registered histories are issuer-only")

    @property
    def seed(self) -> int:
        return _issued_completed_state(self).seed

    @property
    def epoch_count(self) -> int:
        return len(_issued_completed_state(self).records)

    @property
    def canonical_json_bytes(self) -> bytes:
        return _issued_completed_state(self).canonical_json_bytes

    @property
    def history_sha256(self) -> str:
        return _issued_completed_state(self).history_sha256

    @property
    def ranked_epochs(self) -> tuple[RegisteredEpochRank, ...]:
        return _issued_completed_state(self).ranked_epochs

    @property
    def winner(self) -> RegisteredEvaluatedEpoch:
        """Return the preserved exact evaluated capability for rank one."""

        return _issued_completed_state(self).winner

    def __copy__(self) -> NoReturn:
        raise TypeError("completed registered histories cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        del memo
        raise TypeError("completed registered histories cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("completed registered histories cannot be serialized")

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        del protocol
        raise TypeError("completed registered histories cannot be serialized")


@dataclass(frozen=True, slots=True)
class _BoundEpochRecord:
    record: _EpochRecord
    evaluated_epoch: RegisteredEvaluatedEpoch
    evaluated_snapshot: _RegisteredEvaluatedEpochSnapshot
    handoff_snapshot: _RegisteredExecutorEpochHandoffSnapshot
    population: CompletedTrainingPopulation
    epoch_trace: EpochUpdateTraceEvidence
    model_tensors: RegisteredModelTensorEvidence
    authority_fingerprint: str


@dataclass(slots=True)
class _HistoryState:
    token: object
    route_marker: object
    registration: VerifiedRunRegistration
    process_id: int
    executor: RegisteredTrainingExecutor
    validation_inputs: RegisteredValidationInputs
    seed: int
    validation_inputs_sha256: str
    lock: threading.RLock
    phase: _HistoryPhase = "OPEN"
    records: list[_BoundEpochRecord] | None = None
    active_barrier: RegisteredHistoryBarrier | None = None
    active_barrier_token: object | None = None
    completed: RegisteredCompletedTrainingHistory | None = None

    def __post_init__(self) -> None:
        if self.records is None:
            self.records = []


@dataclass(frozen=True, slots=True)
class _HistoryGuard:
    token: object
    route_marker: object
    registration: VerifiedRunRegistration
    process_id: int
    executor: RegisteredTrainingExecutor
    validation_inputs: RegisteredValidationInputs
    seed: int
    validation_inputs_sha256: str
    lock: threading.RLock


@dataclass(frozen=True, slots=True)
class _HistoryLifecycle:
    token: object
    process_id: int
    phase: _HistoryPhase
    records: tuple[_BoundEpochRecord, ...]
    record_fingerprints: tuple[str, ...]
    record_identity_frames: tuple[tuple[object, ...], ...]
    active_barrier: RegisteredHistoryBarrier | None
    active_barrier_token: object | None
    completed: RegisteredCompletedTrainingHistory | None


@dataclass(slots=True)
class _BarrierState:
    token: object
    route_marker: object
    history: RegisteredTrainingHistory
    history_token: object
    registration: VerifiedRunRegistration
    process_id: int
    executor: RegisteredTrainingExecutor
    validation_inputs: RegisteredValidationInputs
    seed: int
    accepted_epoch: int
    evaluated_epoch: RegisteredEvaluatedEpoch
    record_fingerprint: str
    phase: _BarrierPhase = "ISSUED"


@dataclass(frozen=True, slots=True)
class _BarrierGuard:
    token: object
    route_marker: object
    history: RegisteredTrainingHistory
    history_token: object
    registration: VerifiedRunRegistration
    process_id: int
    executor: RegisteredTrainingExecutor
    validation_inputs: RegisteredValidationInputs
    seed: int
    accepted_epoch: int
    evaluated_epoch: RegisteredEvaluatedEpoch
    record_fingerprint: str


@dataclass(frozen=True, slots=True)
class _BarrierLifecycle:
    token: object
    process_id: int
    phase: _BarrierPhase


@dataclass(frozen=True, slots=True)
class _CompletedState:
    token: object
    route_marker: object
    history: RegisteredTrainingHistory
    history_token: object
    registration: VerifiedRunRegistration
    process_id: int
    executor: RegisteredTrainingExecutor
    validation_inputs: RegisteredValidationInputs
    seed: int
    complete_update_trace: CompleteUpdateTraceEvidence
    records: tuple[_BoundEpochRecord, ...]
    record_fingerprints: tuple[str, ...]
    record_identity_frames: tuple[tuple[object, ...], ...]
    canonical_json_bytes: bytes
    history_sha256: str
    ranked_epochs: tuple[RegisteredEpochRank, ...]
    rank_frames: tuple[tuple[object, ...], ...]
    winner: RegisteredEvaluatedEpoch


@dataclass(frozen=True, slots=True)
class _CompletedGuard:
    token: object
    route_marker: object
    history: RegisteredTrainingHistory
    history_token: object
    registration: VerifiedRunRegistration
    process_id: int
    executor: RegisteredTrainingExecutor
    validation_inputs: RegisteredValidationInputs
    seed: int
    complete_update_trace: CompleteUpdateTraceEvidence
    records: tuple[_BoundEpochRecord, ...]
    record_fingerprints: tuple[str, ...]
    record_identity_frames: tuple[tuple[object, ...], ...]
    canonical_json_bytes: bytes
    history_sha256: str
    ranked_epochs: tuple[RegisteredEpochRank, ...]
    rank_frames: tuple[tuple[object, ...], ...]
    winner: RegisteredEvaluatedEpoch


@dataclass(frozen=True, slots=True)
class _CompletedLifecycle:
    token: object
    process_id: int
    phase: _CompletedPhase


@dataclass(frozen=True, slots=True)
class _RegisteredHistorySnapshot:
    registration: VerifiedRunRegistration
    executor: RegisteredTrainingExecutor
    validation_inputs: RegisteredValidationInputs
    process_id: int
    seed: int
    phase: _HistoryPhase
    accepted_epoch_count: int
    validation_inputs_sha256: str
    active_barrier: RegisteredHistoryBarrier | None


@dataclass(frozen=True, slots=True)
class _RegisteredHistoryBarrierSnapshot:
    registration: VerifiedRunRegistration
    history: RegisteredTrainingHistory
    executor: RegisteredTrainingExecutor
    validation_inputs: RegisteredValidationInputs
    process_id: int
    seed: int
    accepted_epoch: int
    next_zero_based_epoch: int | None
    evaluated_epoch: RegisteredEvaluatedEpoch
    evaluated_handoff: RegisteredExecutorEpochHandoff
    evaluated_authority_sha256: str
    phase: _BarrierPhase


@dataclass(frozen=True, slots=True)
class _RegisteredCompletedHistorySnapshot:
    registration: VerifiedRunRegistration
    history: RegisteredTrainingHistory
    executor: RegisteredTrainingExecutor
    validation_inputs: RegisteredValidationInputs
    complete_update_trace: CompleteUpdateTraceEvidence
    process_id: int
    seed: int
    epoch_count: int
    validation_inputs_sha256: str
    complete_update_trace_sha256: str
    canonical_json_bytes: bytes
    history_sha256: str
    ranked_epochs: tuple[RegisteredEpochRank, ...]
    evaluated_epochs: tuple[RegisteredEvaluatedEpoch, ...]
    winner: RegisteredEvaluatedEpoch


@dataclass(slots=True)
class _HistoryTruth:
    state: _HistoryState
    guard: _HistoryGuard
    token: object
    process_id: int
    registration: VerifiedRunRegistration
    executor: RegisteredTrainingExecutor
    validation_inputs: RegisteredValidationInputs
    seed: int
    validation_inputs_sha256: str
    phase: _HistoryPhase
    records: tuple[_BoundEpochRecord, ...]
    record_fingerprints: tuple[str, ...]
    record_identity_frames: tuple[tuple[object, ...], ...]
    active_barrier: RegisteredHistoryBarrier | None
    active_barrier_token: object | None
    completed: RegisteredCompletedTrainingHistory | None


@dataclass(slots=True)
class _BarrierTruth:
    state: _BarrierState
    guard: _BarrierGuard
    token: object
    process_id: int
    history: RegisteredTrainingHistory
    history_token: object
    registration: VerifiedRunRegistration
    executor: RegisteredTrainingExecutor
    validation_inputs: RegisteredValidationInputs
    seed: int
    accepted_epoch: int
    evaluated_epoch: RegisteredEvaluatedEpoch
    record_fingerprint: str
    phase: _BarrierPhase


@dataclass(slots=True)
class _CompletedTruth:
    state: _CompletedState
    guard: _CompletedGuard
    token: object
    process_id: int
    history: RegisteredTrainingHistory
    history_token: object
    registration: VerifiedRunRegistration
    executor: RegisteredTrainingExecutor
    validation_inputs: RegisteredValidationInputs
    seed: int
    complete_update_trace: CompleteUpdateTraceEvidence
    records: tuple[_BoundEpochRecord, ...]
    record_fingerprints: tuple[str, ...]
    record_identity_frames: tuple[tuple[object, ...], ...]
    canonical_json_bytes: bytes
    history_sha256: str
    ranked_epochs: tuple[RegisteredEpochRank, ...]
    rank_frames: tuple[tuple[object, ...], ...]
    winner: RegisteredEvaluatedEpoch
    phase: _CompletedPhase


_HISTORIES: weakref.WeakKeyDictionary[RegisteredTrainingHistory, _HistoryState] = (
    weakref.WeakKeyDictionary()
)
_HISTORY_GUARDS: weakref.WeakKeyDictionary[RegisteredTrainingHistory, _HistoryGuard] = (
    weakref.WeakKeyDictionary()
)
_HISTORY_LIFECYCLES: weakref.WeakKeyDictionary[
    RegisteredTrainingHistory, _HistoryLifecycle
] = weakref.WeakKeyDictionary()
_ISSUED_HISTORIES: weakref.WeakSet[RegisteredTrainingHistory] = weakref.WeakSet()
_FAILED_HISTORIES: weakref.WeakSet[RegisteredTrainingHistory] = weakref.WeakSet()

_BARRIERS: weakref.WeakKeyDictionary[RegisteredHistoryBarrier, _BarrierState] = (
    weakref.WeakKeyDictionary()
)
_BARRIER_GUARDS: weakref.WeakKeyDictionary[RegisteredHistoryBarrier, _BarrierGuard] = (
    weakref.WeakKeyDictionary()
)
_BARRIER_LIFECYCLES: weakref.WeakKeyDictionary[
    RegisteredHistoryBarrier, _BarrierLifecycle
] = weakref.WeakKeyDictionary()
_ISSUED_BARRIERS: weakref.WeakSet[RegisteredHistoryBarrier] = weakref.WeakSet()
_FAILED_BARRIERS: weakref.WeakSet[RegisteredHistoryBarrier] = weakref.WeakSet()

_COMPLETED: weakref.WeakKeyDictionary[
    RegisteredCompletedTrainingHistory, _CompletedState
] = weakref.WeakKeyDictionary()
_COMPLETED_GUARDS: weakref.WeakKeyDictionary[
    RegisteredCompletedTrainingHistory, _CompletedGuard
] = weakref.WeakKeyDictionary()
_COMPLETED_LIFECYCLES: weakref.WeakKeyDictionary[
    RegisteredCompletedTrainingHistory, _CompletedLifecycle
] = weakref.WeakKeyDictionary()
_ISSUED_COMPLETED: weakref.WeakSet[RegisteredCompletedTrainingHistory] = (
    weakref.WeakSet()
)
_FAILED_COMPLETED: weakref.WeakSet[RegisteredCompletedTrainingHistory] = (
    weakref.WeakSet()
)

_EVALUATED_BINDINGS: weakref.WeakKeyDictionary[
    RegisteredEvaluatedEpoch, RegisteredTrainingHistory
] = weakref.WeakKeyDictionary()
_REGISTRY_LOCK = threading.RLock()
_FLOW_LOCK = threading.RLock()


def _build_independent_authority_truth() -> tuple[
    Callable[..., None],
    Callable[..., None],
    Callable[..., None],
    Callable[..., None],
    Callable[..., RegisteredTrainingHistory | None],
    Callable[..., None],
    Callable[..., None],
    Callable[..., None],
    Callable[..., None],
    Callable[..., None],
    Callable[..., None],
    Callable[..., None],
    Callable[..., None],
]:
    """Build closure-owned issuance and monotonic lifecycle authority."""

    histories: weakref.WeakKeyDictionary[RegisteredTrainingHistory, _HistoryTruth] = (
        weakref.WeakKeyDictionary()
    )
    bound_executors: weakref.WeakKeyDictionary[
        RegisteredTrainingExecutor, RegisteredTrainingHistory
    ] = weakref.WeakKeyDictionary()
    barriers: weakref.WeakKeyDictionary[RegisteredHistoryBarrier, _BarrierTruth] = (
        weakref.WeakKeyDictionary()
    )
    completed: weakref.WeakKeyDictionary[
        RegisteredCompletedTrainingHistory, _CompletedTruth
    ] = weakref.WeakKeyDictionary()
    evaluated: weakref.WeakKeyDictionary[
        RegisteredEvaluatedEpoch, RegisteredTrainingHistory
    ] = weakref.WeakKeyDictionary()
    lock = threading.RLock()

    def record_history(
        history: RegisteredTrainingHistory,
        state: _HistoryState,
        guard: _HistoryGuard,
    ) -> None:
        with lock:
            if history in histories or state.executor in bound_executors:
                raise Experiment002RegisteredHistoryError(
                    "executor already has a registered history authority"
                )
            histories[history] = _HistoryTruth(
                state=state,
                guard=guard,
                token=state.token,
                process_id=state.process_id,
                registration=state.registration,
                executor=state.executor,
                validation_inputs=state.validation_inputs,
                seed=state.seed,
                validation_inputs_sha256=state.validation_inputs_sha256,
                phase="OPEN",
                records=(),
                record_fingerprints=(),
                record_identity_frames=(),
                active_barrier=None,
                active_barrier_token=None,
                completed=None,
            )
            bound_executors[state.executor] = history

    def require_history(
        history: RegisteredTrainingHistory,
        state: _HistoryState | None,
        guard: _HistoryGuard | None,
    ) -> None:
        with lock:
            truth = histories.get(history)
            try:
                raw_records = getattr(state, "records", None)
                observed_records = (
                    tuple(raw_records) if type(raw_records) is list else ()
                )
                observed_fingerprints = tuple(
                    _bound_record_fingerprint(record) for record in observed_records
                )
                observed_identity_frames = tuple(
                    _bound_record_identity_frame(record) for record in observed_records
                )
            except BaseException:
                if truth is not None:
                    truth.phase = "FAILED"
                raise
            if (
                truth is None
                or truth.state is not state
                or truth.guard is not guard
                or truth.token is not getattr(state, "token", None)
                or not _same_exact_frame(
                    truth.process_id,
                    getattr(state, "process_id", None),
                )
                or truth.registration is not getattr(state, "registration", None)
                or truth.executor is not getattr(state, "executor", None)
                or truth.validation_inputs
                is not getattr(state, "validation_inputs", None)
                or not _same_exact_frame(
                    truth.seed,
                    getattr(state, "seed", None),
                )
                or not _same_exact_frame(
                    truth.validation_inputs_sha256,
                    getattr(state, "validation_inputs_sha256", None),
                )
                or not _same_exact_frame(
                    truth.phase,
                    getattr(state, "phase", None),
                )
                or type(raw_records) is not list
                or not _same_exact_frame(truth.records, observed_records)
                or not _same_exact_frame(
                    truth.record_fingerprints,
                    observed_fingerprints,
                )
                or not _same_exact_frame(
                    truth.record_identity_frames,
                    observed_identity_frames,
                )
                or truth.active_barrier is not getattr(state, "active_barrier", None)
                or truth.active_barrier_token
                is not getattr(state, "active_barrier_token", None)
                or truth.completed is not getattr(state, "completed", None)
            ):
                if truth is not None:
                    truth.phase = "FAILED"
                raise Experiment002RegisteredHistoryError(
                    "closure-owned history authority changed"
                )

    def transition_history(
        history: RegisteredTrainingHistory,
        state: _HistoryState,
        guard: _HistoryGuard,
        *,
        expected_phase: _HistoryPhase,
        phase: _HistoryPhase,
        records: tuple[_BoundEpochRecord, ...],
        active_barrier: RegisteredHistoryBarrier | None,
        active_barrier_token: object | None,
        completed_history: RegisteredCompletedTrainingHistory | None,
    ) -> None:
        with lock:
            require_history(history, state, guard)
            truth = histories[history]
            if not _same_exact_frame(truth.phase, expected_phase):
                raise Experiment002RegisteredHistoryError(
                    "history lifecycle predecessor changed"
                )
            valid = (
                expected_phase == "OPEN"
                and type(expected_phase) is str
                and type(phase) is str
                and type(records) is tuple
                and phase in ("OPEN", "COMPLETE")
                and len(records) >= len(truth.records)
                and _same_exact_frame(
                    records[: len(truth.records)],
                    truth.records,
                )
                and len(records) - len(truth.records) in (0, 1)
            )
            if phase == "COMPLETE":
                valid = (
                    valid
                    and len(records) == EPOCH_COUNT
                    and completed_history is not None
                )
            if not valid:
                truth.phase = "FAILED"
                raise Experiment002RegisteredHistoryError(
                    "history lifecycle transition is invalid"
                )
            truth.phase = phase
            truth.records = records
            truth.record_fingerprints = tuple(
                _bound_record_fingerprint(record) for record in records
            )
            truth.record_identity_frames = tuple(
                _bound_record_identity_frame(record) for record in records
            )
            truth.active_barrier = active_barrier
            truth.active_barrier_token = active_barrier_token
            truth.completed = completed_history

    def fail_history(
        history: RegisteredTrainingHistory,
        state: _HistoryState | None,
    ) -> None:
        with lock:
            truth = histories.get(history)
            if truth is not None:
                truth.phase = "FAILED"
                truth.active_barrier = getattr(state, "active_barrier", None)
                truth.active_barrier_token = getattr(
                    state, "active_barrier_token", None
                )

    def claim_evaluated(
        evaluated_epoch: RegisteredEvaluatedEpoch,
        history: RegisteredTrainingHistory,
    ) -> None:
        with lock:
            owner = evaluated.get(evaluated_epoch)
            if owner is not None:
                raise Experiment002RegisteredHistoryError(
                    "registered evaluated epoch was already consumed"
                )
            evaluated[evaluated_epoch] = history

    def evaluated_owner(
        evaluated_epoch: RegisteredEvaluatedEpoch,
    ) -> RegisteredTrainingHistory | None:
        with lock:
            return evaluated.get(evaluated_epoch)

    def record_barrier(
        barrier: RegisteredHistoryBarrier,
        state: _BarrierState,
        guard: _BarrierGuard,
    ) -> None:
        with lock:
            if barrier in barriers:
                raise Experiment002RegisteredHistoryError(
                    "history barrier authority already exists"
                )
            barriers[barrier] = _BarrierTruth(
                state=state,
                guard=guard,
                token=state.token,
                process_id=state.process_id,
                history=state.history,
                history_token=state.history_token,
                registration=state.registration,
                executor=state.executor,
                validation_inputs=state.validation_inputs,
                seed=state.seed,
                accepted_epoch=state.accepted_epoch,
                evaluated_epoch=state.evaluated_epoch,
                record_fingerprint=state.record_fingerprint,
                phase="ISSUED",
            )

    def require_barrier(
        barrier: RegisteredHistoryBarrier,
        state: _BarrierState | None,
        guard: _BarrierGuard | None,
    ) -> None:
        with lock:
            truth = barriers.get(barrier)
            if (
                truth is None
                or truth.state is not state
                or truth.guard is not guard
                or truth.token is not getattr(state, "token", None)
                or not _same_exact_frame(
                    truth.process_id,
                    getattr(state, "process_id", None),
                )
                or truth.history is not getattr(state, "history", None)
                or truth.history_token is not getattr(state, "history_token", None)
                or truth.registration is not getattr(state, "registration", None)
                or truth.executor is not getattr(state, "executor", None)
                or truth.validation_inputs
                is not getattr(state, "validation_inputs", None)
                or not _same_exact_frame(
                    truth.seed,
                    getattr(state, "seed", None),
                )
                or not _same_exact_frame(
                    truth.accepted_epoch,
                    getattr(state, "accepted_epoch", None),
                )
                or truth.evaluated_epoch is not getattr(state, "evaluated_epoch", None)
                or not _same_exact_frame(
                    truth.record_fingerprint,
                    getattr(state, "record_fingerprint", None),
                )
                or not _same_exact_frame(
                    truth.phase,
                    getattr(state, "phase", None),
                )
            ):
                if truth is not None:
                    truth.phase = "FAILED"
                raise Experiment002RegisteredHistoryError(
                    "closure-owned history barrier authority changed"
                )

    def transition_barrier(
        barrier: RegisteredHistoryBarrier,
        state: _BarrierState,
        guard: _BarrierGuard,
        *,
        expected: _BarrierPhase,
        phase: _BarrierPhase,
    ) -> None:
        with lock:
            require_barrier(barrier, state, guard)
            truth = barriers[barrier]
            allowed = {
                ("ISSUED", "CONSUMED"),
                ("ISSUED", "RETIRED"),
                ("CONSUMED", "RETIRED"),
            }
            if (
                type(expected) is not str
                or type(phase) is not str
                or not _same_exact_frame(truth.phase, expected)
                or (expected, phase) not in allowed
            ):
                truth.phase = "FAILED"
                raise Experiment002RegisteredHistoryError(
                    "history barrier lifecycle transition is invalid"
                )
            truth.phase = phase

    def fail_barrier(
        barrier: RegisteredHistoryBarrier,
        state: _BarrierState | None,
    ) -> None:
        del state
        with lock:
            truth = barriers.get(barrier)
            if truth is not None:
                truth.phase = "FAILED"

    def record_completed(
        result: RegisteredCompletedTrainingHistory,
        state: _CompletedState,
        guard: _CompletedGuard,
    ) -> None:
        with lock:
            _require_completed_payload_types(state, guard)
            if result in completed:
                raise Experiment002RegisteredHistoryError(
                    "completed history authority already exists"
                )
            completed[result] = _CompletedTruth(
                state=state,
                guard=guard,
                token=state.token,
                process_id=state.process_id,
                history=state.history,
                history_token=state.history_token,
                registration=state.registration,
                executor=state.executor,
                validation_inputs=state.validation_inputs,
                seed=state.seed,
                complete_update_trace=state.complete_update_trace,
                records=state.records,
                record_fingerprints=tuple(
                    _bound_record_fingerprint(record) for record in state.records
                ),
                record_identity_frames=tuple(
                    _bound_record_identity_frame(record) for record in state.records
                ),
                canonical_json_bytes=state.canonical_json_bytes,
                history_sha256=state.history_sha256,
                ranked_epochs=state.ranked_epochs,
                rank_frames=tuple(_rank_frame(rank) for rank in state.ranked_epochs),
                winner=state.winner,
                phase="COMPLETE",
            )

    def require_completed(
        result: RegisteredCompletedTrainingHistory,
        state: _CompletedState | None,
        guard: _CompletedGuard | None,
    ) -> None:
        with lock:
            truth = completed.get(result)
            try:
                observed_records = getattr(state, "records", None)
                observed_ranks = getattr(state, "ranked_epochs", None)
                recomputed_fingerprints = (
                    tuple(
                        _bound_record_fingerprint(record) for record in observed_records
                    )
                    if type(observed_records) is tuple
                    else ()
                )
                recomputed_identity_frames = (
                    tuple(
                        _bound_record_identity_frame(record)
                        for record in observed_records
                    )
                    if type(observed_records) is tuple
                    else ()
                )
                recomputed_rank_frames = (
                    tuple(_rank_frame(rank) for rank in observed_ranks)
                    if type(observed_ranks) is tuple
                    else ()
                )
            except BaseException:
                if truth is not None:
                    truth.phase = "FAILED"
                raise
            if (
                truth is None
                or truth.state is not state
                or truth.guard is not guard
                or truth.token is not getattr(state, "token", None)
                or not _same_exact_frame(
                    truth.process_id,
                    getattr(state, "process_id", None),
                )
                or truth.history is not getattr(state, "history", None)
                or truth.history_token is not getattr(state, "history_token", None)
                or truth.registration is not getattr(state, "registration", None)
                or truth.executor is not getattr(state, "executor", None)
                or truth.validation_inputs
                is not getattr(state, "validation_inputs", None)
                or not _same_exact_frame(
                    truth.seed,
                    getattr(state, "seed", None),
                )
                or truth.complete_update_trace
                is not getattr(state, "complete_update_trace", None)
                or type(observed_records) is not tuple
                or not _same_exact_frame(truth.records, observed_records)
                or not _same_exact_frame(
                    truth.record_fingerprints,
                    getattr(state, "record_fingerprints", None),
                )
                or not _same_exact_frame(
                    truth.record_fingerprints,
                    recomputed_fingerprints,
                )
                or not _same_exact_frame(
                    truth.record_identity_frames,
                    getattr(state, "record_identity_frames", None),
                )
                or not _same_exact_frame(
                    truth.record_identity_frames,
                    recomputed_identity_frames,
                )
                or not _same_exact_frame(
                    truth.canonical_json_bytes,
                    getattr(state, "canonical_json_bytes", None),
                )
                or not _same_exact_frame(
                    truth.history_sha256,
                    getattr(state, "history_sha256", None),
                )
                or type(observed_ranks) is not tuple
                or not _same_exact_frame(truth.ranked_epochs, observed_ranks)
                or not _same_exact_frame(
                    truth.rank_frames,
                    getattr(state, "rank_frames", None),
                )
                or not _same_exact_frame(
                    truth.rank_frames,
                    recomputed_rank_frames,
                )
                or truth.winner is not getattr(state, "winner", None)
                or not _same_exact_frame(truth.phase, "COMPLETE")
            ):
                if truth is not None:
                    truth.phase = "FAILED"
                raise Experiment002RegisteredHistoryError(
                    "closure-owned completed history authority changed"
                )

    def fail_completed(
        result: RegisteredCompletedTrainingHistory,
        state: _CompletedState | None,
    ) -> None:
        del state
        with lock:
            truth = completed.get(result)
            if truth is not None:
                truth.phase = "FAILED"

    return (
        record_history,
        require_history,
        transition_history,
        fail_history,
        evaluated_owner,
        claim_evaluated,
        record_barrier,
        require_barrier,
        transition_barrier,
        fail_barrier,
        record_completed,
        require_completed,
        fail_completed,
    )


(
    _record_history_truth,
    _require_history_truth,
    _transition_history_truth,
    _fail_history_truth,
    _evaluated_truth_owner,
    _claim_evaluated_truth,
    _record_barrier_truth,
    _require_barrier_truth,
    _transition_barrier_truth,
    _fail_barrier_truth,
    _record_completed_truth,
    _require_completed_truth,
    _fail_completed_truth,
) = _build_independent_authority_truth()
del _build_independent_authority_truth


def create_registered_training_history(
    registration: VerifiedRunRegistration,
    executor: RegisteredTrainingExecutor,
    validation_inputs: RegisteredValidationInputs,
    *,
    seed: int,
) -> RegisteredTrainingHistory:
    """Bind one exact executor and validation population to a new history."""

    if type(registration) is not VerifiedRunRegistration:
        raise TypeError("registration must be a VerifiedRunRegistration")
    if type(executor) is not RegisteredTrainingExecutor:
        raise TypeError("executor must be a RegisteredTrainingExecutor")
    if type(validation_inputs) is not RegisteredValidationInputs:
        raise TypeError("validation_inputs must be RegisteredValidationInputs")
    _require_registered_seed(seed)
    process_id = os.getpid()
    reverify_verified_run_registration(registration)
    verify_registered_validation_inputs(validation_inputs)
    executor_snapshot = _registered_executor_snapshot(executor)
    if (
        executor_snapshot.registration is not registration
        or executor_snapshot.process_id != process_id
        or executor_snapshot.seed != seed
        or executor_snapshot.phase != "READY_TO_TRAIN"
        or executor_snapshot.zero_based_epoch != 0
        or executor_snapshot.optimizer_generation != 0
        or executor_snapshot.previous_history_barrier is not None
    ):
        raise Experiment002RegisteredHistoryError(
            "executor is not at the registered history creation boundary"
        )
    validation_sha256 = validation_inputs.validation_inputs_sha256
    _require_sha256(validation_sha256, "validation_inputs_sha256")
    result = _issue_registered_history(
        registration=registration,
        process_id=process_id,
        executor=executor,
        validation_inputs=validation_inputs,
        seed=seed,
        validation_inputs_sha256=validation_sha256,
    )
    reverify_verified_run_registration(registration)
    verify_registered_validation_inputs(validation_inputs)
    verify_registered_training_history(result)
    return result


def verify_registered_training_history(history: RegisteredTrainingHistory) -> None:
    """Revalidate the session and every accepted source capability."""

    state = _issued_history_state(history)
    lock = _trusted_history_lock(history, state)
    with _FLOW_LOCK, lock:
        _require_history_authority(history, state)
        if state.phase == "FAILED":
            raise Experiment002RegisteredHistoryError(
                "registered training history is terminally failed"
            )
        try:
            _verify_history_bindings(state)
            records = _history_records(state)
            for expected_epoch, bound in enumerate(records):
                _verify_bound_epoch(state, bound, expected_epoch=expected_epoch)
            if state.active_barrier is not None:
                barrier_state = _issued_barrier_state(state.active_barrier)
                if (
                    barrier_state.history is not history
                    or barrier_state.history_token is not state.token
                    or barrier_state.accepted_epoch != len(records) - 1
                ):
                    raise Experiment002RegisteredHistoryError(
                        "active history barrier differs from accepted records"
                    )
            _require_history_authority(history, state)
            _verify_history_bindings(state)
        except BaseException:
            _terminal_fail_history(history, state)
            raise


def consume_registered_evaluated_epoch(
    history: RegisteredTrainingHistory,
    evaluated_epoch: RegisteredEvaluatedEpoch,
) -> RegisteredHistoryBarrier:
    """Consume one exact next evaluator result and issue one barrier."""

    return _consume_registered_evaluated_epoch(history, evaluated_epoch)


def _consume_registered_evaluated_epoch(
    history: RegisteredTrainingHistory,
    evaluated_epoch: RegisteredEvaluatedEpoch,
    _claim_truth: Callable[..., None] = _claim_evaluated_truth,
    _owner_truth: Callable[
        [RegisteredEvaluatedEpoch], RegisteredTrainingHistory | None
    ] = _evaluated_truth_owner,
    _transition_truth: Callable[..., None] = _transition_history_truth,
) -> RegisteredHistoryBarrier:
    """Closure-bound implementation of the public epoch consumer."""

    if type(evaluated_epoch) is not RegisteredEvaluatedEpoch:
        raise TypeError("evaluated_epoch must be a RegisteredEvaluatedEpoch")
    state = _issued_history_state(history)
    lock = _trusted_history_lock(history, state)
    with _FLOW_LOCK, lock:
        _require_history_authority(history, state)
        if state.phase != "OPEN":
            raise Experiment002RegisteredHistoryError(
                f"registered history is {state.phase.lower()}"
            )
        expected_epoch = len(_history_records(state))
        previous = state.active_barrier
        previous_state: _BarrierState | None = None
        claimed = False
        try:
            if expected_epoch >= EPOCH_COUNT:
                raise Experiment002RegisteredHistoryError(
                    "all thirty registered epochs are already accepted"
                )
            _verify_history_bindings(state)
            evaluated_snapshot = _registered_evaluated_epoch_snapshot(evaluated_epoch)
            handoff_snapshot = _registered_epoch_handoff_snapshot(
                evaluated_snapshot.handoff
            )
            _require_evaluated_epoch_binding(
                state,
                evaluated_snapshot,
                handoff_snapshot,
                expected_epoch=expected_epoch,
            )
            if expected_epoch == 0:
                if (
                    previous is not None
                    or handoff_snapshot.previous_history_barrier is not None
                ):
                    raise Experiment002RegisteredHistoryError(
                        "epoch zero cannot have a prior history barrier"
                    )
            else:
                if type(previous) is not RegisteredHistoryBarrier:
                    raise Experiment002RegisteredHistoryError(
                        "the next epoch requires the prior history barrier"
                    )
                previous_state = _issued_barrier_state(previous)
                if (
                    previous_state.history is not history
                    or previous_state.history_token is not state.token
                    or previous_state.accepted_epoch != expected_epoch - 1
                    or previous_state.phase != "CONSUMED"
                    or handoff_snapshot.previous_history_barrier is not previous
                ):
                    raise Experiment002RegisteredHistoryError(
                        "executor did not consume the exact prior history barrier"
                    )
            bound = _build_bound_epoch(
                state,
                evaluated_epoch,
                evaluated_snapshot,
                handoff_snapshot,
            )
            _claim_truth(evaluated_epoch, history)
            claimed = True
            with _REGISTRY_LOCK:
                _EVALUATED_BINDINGS[evaluated_epoch] = history
            if previous is not None and previous_state is not None:
                _transition_barrier(previous, previous_state, phase="RETIRED")
            barrier = _issue_registered_barrier(history, state, bound)
            records = _history_records(state)
            barrier_token = _issued_barrier_state(barrier).token
            guard = _history_guard(history)
            _transition_truth(
                history,
                state,
                guard,
                expected_phase="OPEN",
                phase="OPEN",
                records=(*records, bound),
                active_barrier=barrier,
                active_barrier_token=barrier_token,
                completed_history=None,
            )
            records.append(bound)
            state.active_barrier = barrier
            state.active_barrier_token = barrier_token
            _publish_history_lifecycle(history, state)
            verify_registered_evaluated_epoch(evaluated_epoch)
            _verify_bound_epoch(state, bound, expected_epoch=expected_epoch)
            _require_history_authority(history, state)
            return barrier
        except BaseException:
            owner = _owner_truth(evaluated_epoch) if claimed else None
            if owner is not None and owner is not history:
                _terminal_fail_foreign_history(owner, history)
            _terminal_fail_history(history, state)
            raise


def verify_registered_history_barrier(barrier: RegisteredHistoryBarrier) -> None:
    """Verify an issued, consumed, or retired exact registered barrier."""

    state = _issued_barrier_state(barrier)
    history_state = _issued_history_state(state.history)
    lock = _trusted_history_lock(state.history, history_state)
    with _FLOW_LOCK, lock:
        try:
            _require_barrier_authority(barrier, state)
            _require_history_authority(state.history, history_state)
            if state.phase == "FAILED" or history_state.phase == "FAILED":
                raise Experiment002RegisteredHistoryError(
                    "registered history barrier is terminally failed"
                )
            _verify_history_bindings(history_state)
            records = _history_records(history_state)
            if (
                state.accepted_epoch >= len(records)
                or records[state.accepted_epoch].evaluated_epoch
                is not state.evaluated_epoch
                or records[state.accepted_epoch].authority_fingerprint
                != state.record_fingerprint
            ):
                raise Experiment002RegisteredHistoryError(
                    "history barrier no longer matches its accepted epoch"
                )
            _verify_bound_epoch(
                history_state,
                records[state.accepted_epoch],
                expected_epoch=state.accepted_epoch,
            )
            _require_barrier_authority(barrier, state)
        except BaseException:
            _terminal_fail_barrier(barrier, state)
            _terminal_fail_history(state.history, history_state)
            raise


def _consume_registered_history_barrier(
    registration: VerifiedRunRegistration,
    executor: RegisteredTrainingExecutor,
    barrier: RegisteredHistoryBarrier,
) -> _RegisteredHistoryBarrierSnapshot:
    """Consume one non-final barrier for the exact bound executor.

    This is the narrow typed handshake intended for the registered executor's
    next-epoch transition.  It returns only immutable identities and scalars.
    """

    if type(registration) is not VerifiedRunRegistration:
        raise TypeError("registration must be a VerifiedRunRegistration")
    if type(executor) is not RegisteredTrainingExecutor:
        raise TypeError("executor must be a RegisteredTrainingExecutor")
    if type(barrier) is not RegisteredHistoryBarrier:
        raise TypeError("barrier must be a RegisteredHistoryBarrier")
    state = _issued_barrier_state(barrier)
    history_state = _issued_history_state(state.history)
    lock = _trusted_history_lock(state.history, history_state)
    with _FLOW_LOCK, lock:
        try:
            _require_barrier_authority(barrier, state)
            _require_history_authority(state.history, history_state)
            if state.phase != "ISSUED":
                raise Experiment002RegisteredHistoryError(
                    "history barrier is not available for one-use consumption"
                )
            if state.accepted_epoch == _FINAL_ZERO_BASED_EPOCH:
                raise Experiment002RegisteredHistoryError(
                    "the final history barrier belongs to history completion"
                )
            if (
                state.registration is not registration
                or state.executor is not executor
                or history_state.active_barrier is not barrier
                or history_state.active_barrier_token is not state.token
            ):
                raise Experiment002RegisteredHistoryError(
                    "history barrier belongs to a different registered executor"
                )
            verify_registered_history_barrier(barrier)
            executor_snapshot = _registered_executor_snapshot(executor)
            if (
                executor_snapshot.registration is not registration
                or executor_snapshot.process_id != state.process_id
                or executor_snapshot.seed != state.seed
                or executor_snapshot.phase != "AWAITING_EVALUATION"
                or executor_snapshot.zero_based_epoch != state.accepted_epoch
                or executor_snapshot.optimizer_generation
                != (state.accepted_epoch + 1) * UPDATES_PER_EPOCH
            ):
                raise Experiment002RegisteredHistoryError(
                    "executor is not at the exact history barrier boundary"
                )
            _transition_barrier(barrier, state, phase="CONSUMED")
            return _registered_history_barrier_snapshot(barrier)
        except BaseException:
            _terminal_fail_barrier(barrier, state)
            _terminal_fail_history(state.history, history_state)
            raise


def _abort_consumed_registered_history_barrier(
    registration: VerifiedRunRegistration,
    executor: RegisteredTrainingExecutor,
    barrier: RegisteredHistoryBarrier,
) -> None:
    """Terminally abort one exact consumed non-final executor barrier.

    This post-consumption failure route intentionally consults no executor,
    evaluator, or public barrier verifier.  The caller invokes it only after
    releasing every executor and handoff lock.
    """

    if type(registration) is not VerifiedRunRegistration:
        raise TypeError("registration must be a VerifiedRunRegistration")
    if type(executor) is not RegisteredTrainingExecutor:
        raise TypeError("executor must be a RegisteredTrainingExecutor")
    if type(barrier) is not RegisteredHistoryBarrier:
        raise TypeError("barrier must be a RegisteredHistoryBarrier")
    with _REGISTRY_LOCK:
        barrier_state = _BARRIERS.get(barrier)
        barrier_guard = _BARRIER_GUARDS.get(barrier)
        barrier_issued = barrier in _ISSUED_BARRIERS
    if type(barrier_state) is not _BarrierState:
        raise Experiment002RegisteredHistoryError(
            "abort barrier has no exact issued state"
        )
    with _REGISTRY_LOCK:
        history_state = _HISTORIES.get(barrier_state.history)
        history_guard = _HISTORY_GUARDS.get(barrier_state.history)
        history_issued = barrier_state.history in _ISSUED_HISTORIES
    if (
        type(history_state) is not _HistoryState
        or type(barrier_guard) is not _BarrierGuard
        or type(history_guard) is not _HistoryGuard
    ):
        _terminal_fail_barrier(barrier, barrier_state)
        raise Experiment002RegisteredHistoryError(
            "abort boundary has no exact history guards"
        )
    lock = history_guard.lock
    with _FLOW_LOCK, lock:
        try:
            _require_barrier_payload_types(barrier_state, barrier_guard)
            _require_history_payload_types(history_state, history_guard)
            records = _history_records(history_state)
            bound = (
                records[barrier_state.accepted_epoch]
                if type(barrier_state.accepted_epoch) is int
                and 0 <= barrier_state.accepted_epoch < len(records)
                else None
            )
            if (
                not barrier_issued
                or not history_issued
                or barrier_state.history is not barrier_guard.history
                or barrier_state.token is not barrier_guard.token
                or history_state.token is not barrier_state.history_token
                or history_state.token is not history_guard.token
                or history_state.registration is not registration
                or history_state.executor is not executor
                or history_state.process_id != os.getpid()
                or history_state.lock is not lock
                or history_state.phase not in ("OPEN", "FAILED")
                or barrier_state.accepted_epoch != len(records) - 1
                or barrier_state.phase not in ("ISSUED", "CONSUMED", "FAILED")
                or not 0 <= barrier_state.accepted_epoch < _FINAL_ZERO_BASED_EPOCH
                or barrier_state.registration is not registration
                or history_state.registration is not registration
                or barrier_state.executor is not executor
                or history_state.executor is not executor
                or history_state.active_barrier is not barrier
                or history_state.active_barrier_token is not barrier_state.token
                or type(bound) is not _BoundEpochRecord
                or bound.evaluated_epoch is not barrier_state.evaluated_epoch
                or type(bound.evaluated_snapshot.handoff)
                is not RegisteredExecutorEpochHandoff
                or bound.handoff_snapshot.executor is not executor
                or _bound_record_fingerprint(bound) != bound.authority_fingerprint
            ):
                raise Experiment002RegisteredHistoryError(
                    "consumed barrier is not the exact abortable executor boundary"
                )
        except BaseException:
            _terminal_fail_barrier(barrier, barrier_state)
            _terminal_fail_history(barrier_state.history, history_state)
            raise
        _terminal_fail_barrier(barrier, barrier_state)
        _terminal_fail_history(barrier_state.history, history_state)


def _registered_history_barrier_snapshot(
    barrier: RegisteredHistoryBarrier,
) -> _RegisteredHistoryBarrierSnapshot:
    """Return immutable typed barrier bindings after structural verification."""

    state = _issued_barrier_state(barrier)
    _require_barrier_authority(barrier, state)
    history_state = _issued_history_state(state.history)
    records = _history_records(history_state)
    if state.accepted_epoch >= len(records):
        raise Experiment002RegisteredHistoryError(
            "history barrier has no accepted evaluated handoff"
        )
    bound = records[state.accepted_epoch]
    if bound.evaluated_epoch is not state.evaluated_epoch:
        raise Experiment002RegisteredHistoryError(
            "history barrier evaluated capability identity changed"
        )
    next_epoch = (
        state.accepted_epoch + 1
        if state.accepted_epoch < _FINAL_ZERO_BASED_EPOCH
        else None
    )
    result = _RegisteredHistoryBarrierSnapshot(
        registration=state.registration,
        history=state.history,
        executor=state.executor,
        validation_inputs=state.validation_inputs,
        process_id=state.process_id,
        seed=state.seed,
        accepted_epoch=state.accepted_epoch,
        next_zero_based_epoch=next_epoch,
        evaluated_epoch=state.evaluated_epoch,
        evaluated_handoff=bound.evaluated_snapshot.handoff,
        evaluated_authority_sha256=bound.evaluated_snapshot.authority_sha256,
        phase=state.phase,
    )
    _require_barrier_authority(barrier, state)
    return result


def complete_registered_training_history(
    history: RegisteredTrainingHistory,
    final_barrier: RegisteredHistoryBarrier,
    complete_update_trace: CompleteUpdateTraceEvidence,
) -> RegisteredCompletedTrainingHistory:
    """Seal exactly thirty epochs against their registered complete trace."""

    return _complete_registered_training_history(
        history,
        final_barrier,
        complete_update_trace,
    )


def _complete_registered_training_history(
    history: RegisteredTrainingHistory,
    final_barrier: RegisteredHistoryBarrier,
    complete_update_trace: CompleteUpdateTraceEvidence,
    _transition_truth: Callable[..., None] = _transition_history_truth,
) -> RegisteredCompletedTrainingHistory:
    """Closure-bound implementation of registered history completion."""

    if type(final_barrier) is not RegisteredHistoryBarrier:
        raise TypeError("final_barrier must be a RegisteredHistoryBarrier")
    if type(complete_update_trace) is not CompleteUpdateTraceEvidence:
        raise TypeError("complete_update_trace must be a CompleteUpdateTraceEvidence")
    state = _issued_history_state(history)
    lock = _trusted_history_lock(history, state)
    with _FLOW_LOCK, lock:
        _require_history_authority(history, state)
        if state.phase != "OPEN":
            raise Experiment002RegisteredHistoryError(
                f"registered history is {state.phase.lower()}"
            )
        barrier_state: _BarrierState | None = None
        try:
            records = tuple(_history_records(state))
            barrier_state = _issued_barrier_state(final_barrier)
            if (
                len(records) != EPOCH_COUNT
                or state.active_barrier is not final_barrier
                or state.active_barrier_token is not barrier_state.token
                or barrier_state.history is not history
                or barrier_state.history_token is not state.token
                or barrier_state.accepted_epoch != _FINAL_ZERO_BASED_EPOCH
                or barrier_state.phase != "ISSUED"
            ):
                raise Experiment002RegisteredHistoryError(
                    "completion requires the exact unconsumed epoch-29 barrier"
                )
            _verify_history_bindings(state)
            for expected_epoch, bound in enumerate(records):
                _verify_bound_epoch(state, bound, expected_epoch=expected_epoch)
            _require_complete_trace_binding(state, records, complete_update_trace)
            first = _canonical_registered_history_bytes(
                seed=state.seed,
                validation_inputs_sha256=state.validation_inputs_sha256,
                complete_update_trace_sha256=complete_update_trace.sha256,
                records=tuple(bound.record for bound in records),
            )
            second = _canonical_registered_history_bytes(
                seed=state.seed,
                validation_inputs_sha256=state.validation_inputs_sha256,
                complete_update_trace_sha256=complete_update_trace.sha256,
                records=tuple(bound.record for bound in records),
            )
            if first != second:
                raise Experiment002RegisteredHistoryError(
                    "canonical history changed during double serialization"
                )
            _require_history_authority(history, state)
            ranked = _rank_registered_records(tuple(bound.record for bound in records))
            winner_epoch = ranked[0].zero_based_epoch
            winner = records[winner_epoch].evaluated_epoch
            result = _issue_completed_history(
                history=history,
                state=state,
                complete_update_trace=complete_update_trace,
                records=records,
                canonical_json_bytes=first,
                ranked_epochs=ranked,
                winner=winner,
            )
            _transition_barrier(final_barrier, barrier_state, phase="RETIRED")
            guard = _history_guard(history)
            _transition_truth(
                history,
                state,
                guard,
                expected_phase="OPEN",
                phase="COMPLETE",
                records=records,
                active_barrier=None,
                active_barrier_token=None,
                completed_history=result,
            )
            state.phase = "COMPLETE"
            state.active_barrier = None
            state.active_barrier_token = None
            state.completed = result
            _publish_history_lifecycle(history, state)
            verify_registered_completed_training_history(result)
            return result
        except BaseException:
            if barrier_state is not None:
                _terminal_fail_barrier(final_barrier, barrier_state)
            _terminal_fail_history(history, state)
            raise


def verify_registered_completed_training_history(
    result: RegisteredCompletedTrainingHistory,
) -> None:
    """Fully revalidate canonical bytes, ranks, winner, and source evidence."""

    state = _issued_completed_state(result)
    try:
        history_state = _issued_history_state(state.history)
        _require_history_authority(state.history, history_state)
        if history_state.phase != "COMPLETE" or history_state.completed is not result:
            raise Experiment002RegisteredHistoryError(
                "completed history is not the session's terminal result"
            )
        _verify_history_bindings(history_state)
        for expected_epoch, bound in enumerate(state.records):
            _verify_bound_epoch(
                history_state,
                bound,
                expected_epoch=expected_epoch,
            )
        _require_complete_trace_binding(
            history_state,
            state.records,
            state.complete_update_trace,
        )
        expected_bytes = _canonical_registered_history_bytes(
            seed=state.seed,
            validation_inputs_sha256=history_state.validation_inputs_sha256,
            complete_update_trace_sha256=state.complete_update_trace.sha256,
            records=tuple(bound.record for bound in state.records),
        )
        expected_ranks = _rank_registered_records(
            tuple(bound.record for bound in state.records)
        )
        expected_winner = state.records[
            expected_ranks[0].zero_based_epoch
        ].evaluated_epoch
        if (
            expected_bytes != state.canonical_json_bytes
            or hashlib.sha256(_HISTORY_DOMAIN + expected_bytes).hexdigest()
            != state.history_sha256
            or expected_ranks != state.ranked_epochs
            or expected_winner is not state.winner
        ):
            raise Experiment002RegisteredHistoryError(
                "completed registered history bytes or ranking changed"
            )
        _require_completed_authority(result, state)
    except BaseException:
        _terminal_fail_completed(result, state)
        raise


def _registered_completed_history_snapshot(
    result: RegisteredCompletedTrainingHistory,
) -> _RegisteredCompletedHistorySnapshot:
    """Return a fully verified private transfer to checkpoint selection."""

    verify_registered_completed_training_history(result)
    state = _issued_completed_state(result)
    snapshot = _RegisteredCompletedHistorySnapshot(
        registration=state.registration,
        history=state.history,
        executor=state.executor,
        validation_inputs=state.validation_inputs,
        complete_update_trace=state.complete_update_trace,
        process_id=state.process_id,
        seed=state.seed,
        epoch_count=len(state.records),
        validation_inputs_sha256=state.validation_inputs.validation_inputs_sha256,
        complete_update_trace_sha256=state.complete_update_trace.sha256,
        canonical_json_bytes=state.canonical_json_bytes,
        history_sha256=state.history_sha256,
        ranked_epochs=state.ranked_epochs,
        evaluated_epochs=tuple(record.evaluated_epoch for record in state.records),
        winner=state.winner,
    )
    _require_completed_authority(result, state)
    return snapshot


def _registered_history_snapshot(
    history: RegisteredTrainingHistory,
) -> _RegisteredHistorySnapshot:
    state = _issued_history_state(history)
    _require_history_authority(history, state)
    return _RegisteredHistorySnapshot(
        registration=state.registration,
        executor=state.executor,
        validation_inputs=state.validation_inputs,
        process_id=state.process_id,
        seed=state.seed,
        phase=state.phase,
        accepted_epoch_count=len(_history_records(state)),
        validation_inputs_sha256=state.validation_inputs_sha256,
        active_barrier=state.active_barrier,
    )


def _issue_registered_history(
    *,
    registration: VerifiedRunRegistration,
    process_id: int,
    executor: RegisteredTrainingExecutor,
    validation_inputs: RegisteredValidationInputs,
    seed: int,
    validation_inputs_sha256: str,
    _record_truth: Callable[..., None] = _record_history_truth,
) -> RegisteredTrainingHistory:
    token = object()
    lock = threading.RLock()
    state = _HistoryState(
        token=token,
        route_marker=_REGISTERED_ROUTE_MARKER,
        registration=registration,
        process_id=process_id,
        executor=executor,
        validation_inputs=validation_inputs,
        seed=seed,
        validation_inputs_sha256=validation_inputs_sha256,
        lock=lock,
    )
    result = object.__new__(RegisteredTrainingHistory)
    object.__setattr__(result, "_seed", seed)
    object.__setattr__(result, "_token", token)
    object.__setattr__(result, "_route_marker", _REGISTERED_ROUTE_MARKER)
    guard = _HistoryGuard(
        token=token,
        route_marker=_REGISTERED_ROUTE_MARKER,
        registration=registration,
        process_id=process_id,
        executor=executor,
        validation_inputs=validation_inputs,
        seed=seed,
        validation_inputs_sha256=validation_inputs_sha256,
        lock=lock,
    )
    _record_truth(result, state, guard)
    with _REGISTRY_LOCK:
        _HISTORIES[result] = state
        _HISTORY_GUARDS[result] = guard
        _HISTORY_LIFECYCLES[result] = _lifecycle_from_state(state)
        _ISSUED_HISTORIES.add(result)
    return result


def _issue_registered_barrier(
    history: RegisteredTrainingHistory,
    history_state: _HistoryState,
    bound: _BoundEpochRecord,
    _record_truth: Callable[..., None] = _record_barrier_truth,
) -> RegisteredHistoryBarrier:
    token = object()
    state = _BarrierState(
        token=token,
        route_marker=_REGISTERED_ROUTE_MARKER,
        history=history,
        history_token=history_state.token,
        registration=history_state.registration,
        process_id=history_state.process_id,
        executor=history_state.executor,
        validation_inputs=history_state.validation_inputs,
        seed=history_state.seed,
        accepted_epoch=bound.record.zero_based_epoch,
        evaluated_epoch=bound.evaluated_epoch,
        record_fingerprint=bound.authority_fingerprint,
    )
    result = object.__new__(RegisteredHistoryBarrier)
    object.__setattr__(result, "_seed", state.seed)
    object.__setattr__(result, "_accepted_epoch", state.accepted_epoch)
    object.__setattr__(result, "_token", token)
    object.__setattr__(result, "_route_marker", _REGISTERED_ROUTE_MARKER)
    guard = _BarrierGuard(
        token=token,
        route_marker=_REGISTERED_ROUTE_MARKER,
        history=history,
        history_token=history_state.token,
        registration=history_state.registration,
        process_id=history_state.process_id,
        executor=history_state.executor,
        validation_inputs=history_state.validation_inputs,
        seed=history_state.seed,
        accepted_epoch=state.accepted_epoch,
        evaluated_epoch=bound.evaluated_epoch,
        record_fingerprint=bound.authority_fingerprint,
    )
    _record_truth(result, state, guard)
    with _REGISTRY_LOCK:
        _BARRIERS[result] = state
        _BARRIER_GUARDS[result] = guard
        _BARRIER_LIFECYCLES[result] = _BarrierLifecycle(
            token=token,
            process_id=state.process_id,
            phase="ISSUED",
        )
        _ISSUED_BARRIERS.add(result)
    return result


def _issue_completed_history(
    *,
    history: RegisteredTrainingHistory,
    state: _HistoryState,
    complete_update_trace: CompleteUpdateTraceEvidence,
    records: tuple[_BoundEpochRecord, ...],
    canonical_json_bytes: bytes,
    ranked_epochs: tuple[RegisteredEpochRank, ...],
    winner: RegisteredEvaluatedEpoch,
    _record_truth: Callable[..., None] = _record_completed_truth,
) -> RegisteredCompletedTrainingHistory:
    token = object()
    history_sha256 = hashlib.sha256(_HISTORY_DOMAIN + canonical_json_bytes).hexdigest()
    completed_state = _CompletedState(
        token=token,
        route_marker=_REGISTERED_ROUTE_MARKER,
        history=history,
        history_token=state.token,
        registration=state.registration,
        process_id=state.process_id,
        executor=state.executor,
        validation_inputs=state.validation_inputs,
        seed=state.seed,
        complete_update_trace=complete_update_trace,
        records=records,
        record_fingerprints=tuple(
            _bound_record_fingerprint(record) for record in records
        ),
        record_identity_frames=tuple(
            _bound_record_identity_frame(record) for record in records
        ),
        canonical_json_bytes=canonical_json_bytes,
        history_sha256=history_sha256,
        ranked_epochs=ranked_epochs,
        rank_frames=tuple(_rank_frame(rank) for rank in ranked_epochs),
        winner=winner,
    )
    result = object.__new__(RegisteredCompletedTrainingHistory)
    object.__setattr__(result, "_seed", state.seed)
    object.__setattr__(result, "_history_sha256", history_sha256)
    object.__setattr__(result, "_winner_epoch", ranked_epochs[0].zero_based_epoch)
    object.__setattr__(result, "_token", token)
    object.__setattr__(result, "_route_marker", _REGISTERED_ROUTE_MARKER)
    guard = _CompletedGuard(
        token=token,
        route_marker=_REGISTERED_ROUTE_MARKER,
        history=history,
        history_token=state.token,
        registration=state.registration,
        process_id=state.process_id,
        executor=state.executor,
        validation_inputs=state.validation_inputs,
        seed=state.seed,
        complete_update_trace=complete_update_trace,
        records=records,
        record_fingerprints=completed_state.record_fingerprints,
        record_identity_frames=completed_state.record_identity_frames,
        canonical_json_bytes=canonical_json_bytes,
        history_sha256=history_sha256,
        ranked_epochs=ranked_epochs,
        rank_frames=completed_state.rank_frames,
        winner=winner,
    )
    _record_truth(result, completed_state, guard)
    with _REGISTRY_LOCK:
        _COMPLETED[result] = completed_state
        _COMPLETED_GUARDS[result] = guard
        _COMPLETED_LIFECYCLES[result] = _CompletedLifecycle(
            token=token,
            process_id=state.process_id,
            phase="COMPLETE",
        )
        _ISSUED_COMPLETED.add(result)
    return result


def _issued_history_state(
    history: RegisteredTrainingHistory,
    _require_truth: Callable[..., None] = _require_history_truth,
) -> _HistoryState:
    if type(history) is not RegisteredTrainingHistory:
        raise TypeError("history must be a RegisteredTrainingHistory")
    with _REGISTRY_LOCK:
        state = _HISTORIES.get(history)
        guard = _HISTORY_GUARDS.get(history)
        issued = history in _ISSUED_HISTORIES
        failed = history in _FAILED_HISTORIES
    try:
        _require_truth(history, state, guard)
    except BaseException:
        if type(state) is _HistoryState:
            _terminal_fail_history(history, state)
        raise
    if failed or type(state) is not _HistoryState or not issued:
        if type(state) is _HistoryState:
            _terminal_fail_history(history, state)
        raise Experiment002RegisteredHistoryError(
            "registered history was not issued or is terminally failed"
        )
    try:
        _require_history_authority(history, state)
    except BaseException:
        _terminal_fail_history(history, state)
        raise
    return state


def _issued_barrier_state(
    barrier: RegisteredHistoryBarrier,
    _require_truth: Callable[..., None] = _require_barrier_truth,
) -> _BarrierState:
    if type(barrier) is not RegisteredHistoryBarrier:
        raise TypeError("barrier must be a RegisteredHistoryBarrier")
    with _REGISTRY_LOCK:
        state = _BARRIERS.get(barrier)
        guard = _BARRIER_GUARDS.get(barrier)
        issued = barrier in _ISSUED_BARRIERS
        failed = barrier in _FAILED_BARRIERS
    try:
        _require_truth(barrier, state, guard)
    except BaseException:
        if type(state) is _BarrierState:
            _terminal_fail_barrier(barrier, state)
        raise
    if failed or type(state) is not _BarrierState or not issued:
        if type(state) is _BarrierState:
            _terminal_fail_barrier(barrier, state)
        raise Experiment002RegisteredHistoryError(
            "registered history barrier was not issued or is failed"
        )
    try:
        _require_barrier_authority(barrier, state)
    except BaseException:
        _terminal_fail_barrier(barrier, state)
        raise
    return state


def _issued_completed_state(
    result: RegisteredCompletedTrainingHistory,
    _require_truth: Callable[..., None] = _require_completed_truth,
) -> _CompletedState:
    if type(result) is not RegisteredCompletedTrainingHistory:
        raise TypeError("result must be a RegisteredCompletedTrainingHistory")
    with _REGISTRY_LOCK:
        state = _COMPLETED.get(result)
        guard = _COMPLETED_GUARDS.get(result)
        lifecycle = _COMPLETED_LIFECYCLES.get(result)
        issued = result in _ISSUED_COMPLETED
        failed = result in _FAILED_COMPLETED
    try:
        _require_truth(result, state, guard)
    except BaseException:
        if type(state) is _CompletedState:
            _terminal_fail_completed(result, state)
        raise
    if (
        failed
        or not issued
        or type(state) is not _CompletedState
        or type(guard) is not _CompletedGuard
        or type(lifecycle) is not _CompletedLifecycle
        or lifecycle.token is not state.token
        or lifecycle.process_id != state.process_id
        or lifecycle.phase != "COMPLETE"
    ):
        if type(state) is _CompletedState:
            _terminal_fail_completed(result, state)
        raise Experiment002RegisteredHistoryError(
            "completed registered history was not issued or is failed"
        )
    try:
        _require_completed_authority(result, state)
    except BaseException:
        _terminal_fail_completed(result, state)
        raise
    return state


def _trusted_history_lock(
    history: RegisteredTrainingHistory,
    state: _HistoryState,
) -> threading.RLock:
    guard = _history_guard(history)
    if (
        state.lock is not guard.lock
        or type(state.lock) is not type(threading.RLock())
        or state.token is not guard.token
    ):
        _terminal_fail_history(history, state)
        raise Experiment002RegisteredHistoryError(
            "registered history lock authority changed"
        )
    return guard.lock


def _history_guard(history: RegisteredTrainingHistory) -> _HistoryGuard:
    with _REGISTRY_LOCK:
        guard = _HISTORY_GUARDS.get(history)
    if type(guard) is not _HistoryGuard:
        raise Experiment002RegisteredHistoryError("registered history guard is missing")
    return guard


def _history_records(state: _HistoryState) -> list[_BoundEpochRecord]:
    if type(state.records) is not list:
        raise Experiment002RegisteredHistoryError(
            "registered history record store has an invalid type"
        )
    return state.records


def _require_history_authority(
    history: RegisteredTrainingHistory,
    state: _HistoryState,
    _require_truth: Callable[..., None] = _require_history_truth,
) -> None:
    guard = _history_guard(history)
    _require_history_payload_types(state, guard)
    with _REGISTRY_LOCK:
        lifecycle = _HISTORY_LIFECYCLES.get(history)
        issued = history in _ISSUED_HISTORIES
        failed = history in _FAILED_HISTORIES
    _require_truth(history, state, guard)
    records = tuple(_history_records(state))
    expected_lifecycle = _lifecycle_from_state(state)
    try:
        object_frame = (history._seed, history._token, history._route_marker)
    except AttributeError as error:
        raise Experiment002RegisteredHistoryError(
            "registered history capability is incomplete"
        ) from error
    lifecycle_matches = type(lifecycle) is _HistoryLifecycle and _same_exact_frame(
        _history_lifecycle_frame(lifecycle),
        _history_lifecycle_frame(expected_lifecycle),
    )
    if (
        not issued
        or not lifecycle_matches
        or failed != (state.phase == "FAILED")
        or state.token is not guard.token
        or state.route_marker is not _REGISTERED_ROUTE_MARKER
        or guard.route_marker is not _REGISTERED_ROUTE_MARKER
        or state.registration is not guard.registration
        or not _same_exact_frame(state.process_id, guard.process_id)
        or state.process_id != os.getpid()
        or state.executor is not guard.executor
        or state.validation_inputs is not guard.validation_inputs
        or not _same_exact_frame(state.seed, guard.seed)
        or not _same_exact_frame(
            state.validation_inputs_sha256,
            guard.validation_inputs_sha256,
        )
        or state.lock is not guard.lock
        or state.phase not in ("OPEN", "COMPLETE", "FAILED")
        or len(records) > EPOCH_COUNT
        or (state.active_barrier is None) != (state.active_barrier_token is None)
        or (state.phase == "COMPLETE" and len(records) != EPOCH_COUNT)
        or (state.phase == "COMPLETE" and state.completed is None)
        or (state.phase == "OPEN" and state.completed is not None)
        or not _same_exact_frame(
            object_frame,
            (state.seed, state.token, _REGISTERED_ROUTE_MARKER),
        )
    ):
        raise Experiment002RegisteredHistoryError(
            "registered history authority is invalid"
        )
    for expected_epoch, bound in enumerate(records):
        if (
            type(bound) is not _BoundEpochRecord
            or bound.record.zero_based_epoch != expected_epoch
            or _bound_record_fingerprint(bound) != bound.authority_fingerprint
        ):
            raise Experiment002RegisteredHistoryError(
                "accepted registered history record authority changed"
            )


def _require_barrier_authority(
    barrier: RegisteredHistoryBarrier,
    state: _BarrierState,
    _require_truth: Callable[..., None] = _require_barrier_truth,
) -> None:
    with _REGISTRY_LOCK:
        guard = _BARRIER_GUARDS.get(barrier)
        lifecycle = _BARRIER_LIFECYCLES.get(barrier)
        issued = barrier in _ISSUED_BARRIERS
        failed = barrier in _FAILED_BARRIERS
    if type(guard) is _BarrierGuard:
        _require_truth(barrier, state, guard)
        _require_barrier_payload_types(state, guard)
    try:
        object_frame = (
            barrier._seed,
            barrier._accepted_epoch,
            barrier._token,
            barrier._route_marker,
        )
    except AttributeError as error:
        raise Experiment002RegisteredHistoryError(
            "registered history barrier is incomplete"
        ) from error
    lifecycle_matches = type(lifecycle) is _BarrierLifecycle and _same_exact_frame(
        _barrier_lifecycle_frame(lifecycle),
        (state.token, state.process_id, state.phase),
    )
    if (
        not issued
        or type(guard) is not _BarrierGuard
        or not lifecycle_matches
        or failed != (state.phase == "FAILED")
        or state.token is not guard.token
        or state.route_marker is not _REGISTERED_ROUTE_MARKER
        or guard.route_marker is not _REGISTERED_ROUTE_MARKER
        or state.history is not guard.history
        or state.history_token is not guard.history_token
        or state.registration is not guard.registration
        or not _same_exact_frame(state.process_id, guard.process_id)
        or state.process_id != os.getpid()
        or state.executor is not guard.executor
        or state.validation_inputs is not guard.validation_inputs
        or not _same_exact_frame(state.seed, guard.seed)
        or not _same_exact_frame(state.accepted_epoch, guard.accepted_epoch)
        or state.evaluated_epoch is not guard.evaluated_epoch
        or not _same_exact_frame(
            state.record_fingerprint,
            guard.record_fingerprint,
        )
        or state.phase not in ("ISSUED", "CONSUMED", "RETIRED", "FAILED")
        or not _same_exact_frame(
            object_frame,
            (
                state.seed,
                state.accepted_epoch,
                state.token,
                _REGISTERED_ROUTE_MARKER,
            ),
        )
    ):
        raise Experiment002RegisteredHistoryError(
            "registered history barrier authority is invalid"
        )


def _require_completed_authority(
    result: RegisteredCompletedTrainingHistory,
    state: _CompletedState,
    _require_truth: Callable[..., None] = _require_completed_truth,
) -> None:
    with _REGISTRY_LOCK:
        guard = _COMPLETED_GUARDS.get(result)
        lifecycle = _COMPLETED_LIFECYCLES.get(result)
        history_state = _HISTORIES.get(state.history)
        issued = result in _ISSUED_COMPLETED
        failed = result in _FAILED_COMPLETED
    if type(guard) is _CompletedGuard:
        _require_truth(result, state, guard)
        _require_completed_payload_types(state, guard)
    try:
        object_frame = (
            result._seed,
            result._history_sha256,
            result._winner_epoch,
            result._token,
            result._route_marker,
        )
    except AttributeError as error:
        raise Experiment002RegisteredHistoryError(
            "completed history capability is incomplete"
        ) from error
    if type(history_state) is _HistoryState:
        _require_history_authority(state.history, history_state)
    lifecycle_matches = type(lifecycle) is _CompletedLifecycle and _same_exact_frame(
        _completed_lifecycle_frame(lifecycle),
        (state.token, state.process_id, "COMPLETE"),
    )
    if (
        not issued
        or failed
        or type(history_state) is not _HistoryState
        or history_state.phase != "COMPLETE"
        or history_state.completed is not result
        or not _same_exact_frame(
            tuple(_history_records(history_state)),
            state.records,
        )
        or type(guard) is not _CompletedGuard
        or not lifecycle_matches
        or state.process_id != os.getpid()
        or state.route_marker is not _REGISTERED_ROUTE_MARKER
        or guard.route_marker is not _REGISTERED_ROUTE_MARKER
        or state.token is not guard.token
        or state.history is not guard.history
        or state.history_token is not guard.history_token
        or state.registration is not guard.registration
        or state.executor is not guard.executor
        or state.validation_inputs is not guard.validation_inputs
        or not _same_exact_frame(state.seed, guard.seed)
        or state.complete_update_trace is not guard.complete_update_trace
        or not _same_exact_frame(state.records, guard.records)
        or not _same_exact_frame(
            state.record_fingerprints,
            guard.record_fingerprints,
        )
        or not _same_exact_frame(
            state.record_fingerprints,
            tuple(_bound_record_fingerprint(record) for record in state.records),
        )
        or not _same_exact_frame(
            state.record_identity_frames,
            guard.record_identity_frames,
        )
        or not _same_exact_frame(
            state.record_identity_frames,
            tuple(_bound_record_identity_frame(record) for record in state.records),
        )
        or not _same_exact_frame(
            state.canonical_json_bytes,
            guard.canonical_json_bytes,
        )
        or not _same_exact_frame(state.history_sha256, guard.history_sha256)
        or not _same_exact_frame(state.ranked_epochs, guard.ranked_epochs)
        or not _same_exact_frame(state.rank_frames, guard.rank_frames)
        or not _same_exact_frame(
            state.rank_frames,
            tuple(_rank_frame(rank) for rank in state.ranked_epochs),
        )
        or state.winner is not guard.winner
        or len(state.records) != EPOCH_COUNT
        or not _same_exact_frame(
            object_frame,
            (
                state.seed,
                state.history_sha256,
                state.ranked_epochs[0].zero_based_epoch,
                state.token,
                _REGISTERED_ROUTE_MARKER,
            ),
        )
    ):
        raise Experiment002RegisteredHistoryError(
            "completed registered history authority is invalid"
        )


def _verify_history_bindings(state: _HistoryState) -> None:
    reverify_verified_run_registration(state.registration)
    verify_registered_validation_inputs(state.validation_inputs)
    if not hmac.compare_digest(
        state.validation_inputs.validation_inputs_sha256,
        state.validation_inputs_sha256,
    ):
        raise Experiment002RegisteredHistoryError(
            "registered validation input digest changed"
        )
    executor_snapshot = _registered_executor_snapshot(state.executor)
    if (
        executor_snapshot.registration is not state.registration
        or executor_snapshot.process_id != state.process_id
        or executor_snapshot.seed != state.seed
    ):
        raise Experiment002RegisteredHistoryError("registered executor binding changed")
    reverify_verified_run_registration(state.registration)


def _build_bound_epoch(
    state: _HistoryState,
    evaluated_epoch: RegisteredEvaluatedEpoch,
    evaluated_snapshot: _RegisteredEvaluatedEpochSnapshot,
    handoff_snapshot: _RegisteredExecutorEpochHandoffSnapshot,
) -> _BoundEpochRecord:
    epoch = evaluated_snapshot.zero_based_epoch
    verify_completed_registered_training_population(
        handoff_snapshot.population,
        seed=state.seed,
        zero_based_epoch=epoch,
    )
    verify_registered_epoch_update_trace(handoff_snapshot.epoch_trace)
    verify_registered_validation_evidence(evaluated_snapshot.validation_evidence)
    validation = _registered_validation_evidence_snapshot(
        evaluated_snapshot.validation_evidence
    )
    verify_registered_model_tensor_evidence(
        evaluated_snapshot.model_tensors,
        seed=state.seed,
        zero_based_epoch=epoch,
    )
    confusion = _validated_registered_confusion(validation.confusion_matrix)
    if validation.macro_f1 != _macro_f1_from_confusion(confusion):
        raise Experiment002RegisteredHistoryError(
            "registered macro-F1 differs from the confusion matrix"
        )
    record = _EpochRecord(
        zero_based_epoch=epoch,
        first_global_update=handoff_snapshot.epoch_trace.first_global_update,
        last_global_update_inclusive=(
            handoff_snapshot.epoch_trace.last_global_update_inclusive
        ),
        training_population_digest=handoff_snapshot.population.sha256,
        training_cross_entropy=np.float64(
            handoff_snapshot.epoch_trace.training_cross_entropy
        ),
        epoch_update_trace_digest=handoff_snapshot.epoch_trace.sha256,
        validation_input_digest=validation.validation_inputs_sha256,
        validation_confusion_matrix=confusion,
        validation_cross_entropy=np.float64(validation.cross_entropy),
        macro_f1=validation.macro_f1,
        validation_prediction_digest=validation.validation_predictions_sha256,
        model_tensor_digest=evaluated_snapshot.model_tensor_sha256,
    )
    _validate_registered_record(record, seed=state.seed)
    provisional = _BoundEpochRecord(
        record=record,
        evaluated_epoch=evaluated_epoch,
        evaluated_snapshot=evaluated_snapshot,
        handoff_snapshot=handoff_snapshot,
        population=handoff_snapshot.population,
        epoch_trace=handoff_snapshot.epoch_trace,
        model_tensors=evaluated_snapshot.model_tensors,
        authority_fingerprint="",
    )
    fingerprint = _bound_record_fingerprint(provisional)
    return _BoundEpochRecord(
        record=record,
        evaluated_epoch=evaluated_epoch,
        evaluated_snapshot=evaluated_snapshot,
        handoff_snapshot=handoff_snapshot,
        population=handoff_snapshot.population,
        epoch_trace=handoff_snapshot.epoch_trace,
        model_tensors=evaluated_snapshot.model_tensors,
        authority_fingerprint=fingerprint,
    )


def _verify_bound_epoch(
    state: _HistoryState,
    bound: _BoundEpochRecord,
    *,
    expected_epoch: int,
) -> None:
    if type(bound) is not _BoundEpochRecord:
        raise TypeError("bound history record has an invalid type")
    if (
        bound.record.zero_based_epoch != expected_epoch
        or _bound_record_fingerprint(bound) != bound.authority_fingerprint
    ):
        raise Experiment002RegisteredHistoryError(
            "bound registered epoch authority changed"
        )
    verify_registered_evaluated_epoch(bound.evaluated_epoch)
    evaluated = _registered_evaluated_epoch_snapshot(bound.evaluated_epoch)
    handoff = _registered_epoch_handoff_snapshot(evaluated.handoff)
    _require_evaluated_epoch_binding(
        state,
        evaluated,
        handoff,
        expected_epoch=expected_epoch,
    )
    if not _same_evaluated_snapshot(evaluated, bound.evaluated_snapshot):
        raise Experiment002RegisteredHistoryError(
            "registered evaluated-epoch snapshot changed"
        )
    if not _same_handoff_snapshot(handoff, bound.handoff_snapshot):
        raise Experiment002RegisteredHistoryError("registered handoff snapshot changed")
    if (
        handoff.population is not bound.population
        or handoff.epoch_trace is not bound.epoch_trace
        or evaluated.model_tensors is not bound.model_tensors
    ):
        raise Experiment002RegisteredHistoryError(
            "bound registered evidence identities changed"
        )
    verify_completed_registered_training_population(
        bound.population,
        seed=state.seed,
        zero_based_epoch=expected_epoch,
    )
    verify_registered_epoch_update_trace(bound.epoch_trace)
    verify_registered_validation_evidence(evaluated.validation_evidence)
    validation = _registered_validation_evidence_snapshot(evaluated.validation_evidence)
    verify_registered_model_tensor_evidence(
        bound.model_tensors,
        seed=state.seed,
        zero_based_epoch=expected_epoch,
    )
    expected_record = _build_record_from_verified_snapshots(
        state,
        evaluated,
        handoff,
        validation.confusion_matrix,
        validation.cross_entropy,
        validation.macro_f1,
        validation.validation_inputs_sha256,
        validation.validation_predictions_sha256,
    )
    if _epoch_document(expected_record) != _epoch_document(bound.record):
        raise Experiment002RegisteredHistoryError(
            "registered history record differs from its exact evidence"
        )


def _build_record_from_verified_snapshots(
    state: _HistoryState,
    evaluated: _RegisteredEvaluatedEpochSnapshot,
    handoff: _RegisteredExecutorEpochHandoffSnapshot,
    confusion_value: object,
    validation_cross_entropy: np.float64,
    macro_f1: Fraction,
    validation_inputs_sha256: str,
    validation_predictions_sha256: str,
) -> _EpochRecord:
    confusion = _validated_registered_confusion(confusion_value)
    record = _EpochRecord(
        zero_based_epoch=evaluated.zero_based_epoch,
        first_global_update=handoff.epoch_trace.first_global_update,
        last_global_update_inclusive=handoff.epoch_trace.last_global_update_inclusive,
        training_population_digest=handoff.population.sha256,
        training_cross_entropy=np.float64(handoff.epoch_trace.training_cross_entropy),
        epoch_update_trace_digest=handoff.epoch_trace.sha256,
        validation_input_digest=validation_inputs_sha256,
        validation_confusion_matrix=confusion,
        validation_cross_entropy=np.float64(validation_cross_entropy),
        macro_f1=macro_f1,
        validation_prediction_digest=validation_predictions_sha256,
        model_tensor_digest=evaluated.model_tensor_sha256,
    )
    _validate_registered_record(record, seed=state.seed)
    return record


def _require_evaluated_epoch_binding(
    state: _HistoryState,
    evaluated: _RegisteredEvaluatedEpochSnapshot,
    handoff: _RegisteredExecutorEpochHandoffSnapshot,
    *,
    expected_epoch: int,
) -> None:
    if (
        type(evaluated) is not _RegisteredEvaluatedEpochSnapshot
        or type(handoff) is not _RegisteredExecutorEpochHandoffSnapshot
        or evaluated.registration is not state.registration
        or evaluated.validation_inputs is not state.validation_inputs
        or handoff.registration is not state.registration
        or handoff.executor is not state.executor
        or handoff.validation_inputs is not state.validation_inputs
        or evaluated.process_id != state.process_id
        or handoff.process_id != state.process_id
        or evaluated.seed != state.seed
        or handoff.seed != state.seed
        or evaluated.zero_based_epoch != expected_epoch
        or handoff.zero_based_epoch != expected_epoch
        or evaluated.optimizer_generation != (expected_epoch + 1) * UPDATES_PER_EPOCH
        or handoff.optimizer_generation != (expected_epoch + 1) * UPDATES_PER_EPOCH
        or evaluated.batch_count != VALIDATION_BATCH_COUNT
        or evaluated.another_training_epoch
        is not (expected_epoch < _FINAL_ZERO_BASED_EPOCH)
        or handoff.another_training_epoch
        is not (expected_epoch < _FINAL_ZERO_BASED_EPOCH)
        or evaluated.registration_head_commit != state.registration.head_commit
        or evaluated.registration_sha256 != state.registration.registration_sha256
        or evaluated.source_bundle_sha256 != state.registration.source_bundle_sha256
        or evaluated.validation_inputs_sha256 != state.validation_inputs_sha256
        or evaluated.validation_inputs_sha256
        != evaluated.validation_evidence.validation_inputs_sha256
        or evaluated.validation_predictions_sha256
        != evaluated.validation_evidence.validation_predictions_sha256
        or evaluated.model_tensor_sha256 != evaluated.model_tensors.sha256
        or evaluated.model_tensor_sha256 != handoff.runtime_digests.model_sha256
        or evaluated.optimizer_sha256 != handoff.runtime_digests.optimizer_sha256
    ):
        raise Experiment002RegisteredHistoryError(
            "evaluated epoch differs from the registered history binding"
        )
    verify_completed_registered_training_population(
        handoff.population,
        seed=state.seed,
        zero_based_epoch=expected_epoch,
    )
    if (
        handoff.population.example_count != TRAINING_EXAMPLE_COUNT
        or handoff.population.batch_count != REGISTERED_BATCH_COUNT
    ):
        raise Experiment002RegisteredHistoryError(
            "registered training population has the wrong exact layout"
        )
    verify_registered_epoch_update_trace(handoff.epoch_trace)
    if (
        handoff.epoch_trace.seed != state.seed
        or handoff.epoch_trace.zero_based_epoch != expected_epoch
        or handoff.epoch_trace.update_count != UPDATES_PER_EPOCH
        or handoff.epoch_trace.first_global_update != expected_epoch * UPDATES_PER_EPOCH
        or handoff.epoch_trace.last_global_update_inclusive
        != (expected_epoch + 1) * UPDATES_PER_EPOCH - 1
    ):
        raise Experiment002RegisteredHistoryError(
            "registered epoch trace has the wrong exact layout"
        )


def _require_complete_trace_binding(
    state: _HistoryState,
    records: tuple[_BoundEpochRecord, ...],
    complete_trace: CompleteUpdateTraceEvidence,
) -> None:
    verify_registered_complete_update_trace(complete_trace)
    if (
        complete_trace.seed != state.seed
        or complete_trace.update_count != TOTAL_UPDATE_COUNT
        or len(complete_trace.epochs) != EPOCH_COUNT
    ):
        raise Experiment002RegisteredHistoryError(
            "complete update trace has the wrong registered layout"
        )
    for expected_epoch, (trace, bound) in enumerate(
        zip(complete_trace.epochs, records, strict=True)
    ):
        if (
            trace is not bound.epoch_trace
            or trace.zero_based_epoch != expected_epoch
            or not hmac.compare_digest(
                trace.sha256,
                bound.record.epoch_update_trace_digest,
            )
        ):
            raise Experiment002RegisteredHistoryError(
                "complete update trace differs from accepted epoch evidence"
            )
    verify_registered_complete_update_trace(complete_trace)


def _canonical_registered_history_bytes(
    *,
    seed: int,
    validation_inputs_sha256: str,
    complete_update_trace_sha256: str,
    records: tuple[_EpochRecord, ...],
) -> bytes:
    _require_registered_seed(seed)
    _require_sha256(validation_inputs_sha256, "validation_inputs_sha256")
    _require_sha256(
        complete_update_trace_sha256,
        "complete_update_trace_sha256",
    )
    if type(records) is not tuple or len(records) != EPOCH_COUNT:
        raise Experiment002RegisteredHistoryError(
            "canonical registered history requires exactly thirty records"
        )
    for expected_epoch, record in enumerate(records):
        _validate_registered_record(record, seed=seed)
        if (
            record.zero_based_epoch != expected_epoch
            or record.validation_input_digest != validation_inputs_sha256
        ):
            raise Experiment002RegisteredHistoryError(
                "canonical registered records are reordered or mismatched"
            )
    document: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "experiment": _EXPERIMENT,
        "seed": seed,
        "validation_input_digest": validation_inputs_sha256,
        "complete_update_trace_digest": complete_update_trace_sha256,
        "epochs": [_epoch_document(record) for record in records],
    }
    _require_json_without_floats(document)
    serialized = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return (serialized + "\n").encode("utf-8")


def _rank_registered_records(
    records: tuple[_EpochRecord, ...],
) -> tuple[RegisteredEpochRank, ...]:
    if type(records) is not tuple or len(records) != EPOCH_COUNT:
        raise Experiment002RegisteredHistoryError(
            "registered ranking requires exactly thirty records"
        )
    ordered = sorted(
        records,
        key=lambda record: (
            -record.macro_f1,
            record.validation_cross_entropy,
            record.zero_based_epoch,
        ),
    )
    result = tuple(
        RegisteredEpochRank(
            zero_based_epoch=record.zero_based_epoch,
            macro_f1_exact_numerator=record.macro_f1.numerator,
            macro_f1_exact_denominator=record.macro_f1.denominator,
            validation_cross_entropy_float64_hex=_registered_float64_hex(
                record.validation_cross_entropy
            ),
        )
        for record in ordered
    )
    if len({rank.zero_based_epoch for rank in result}) != EPOCH_COUNT:
        raise Experiment002RegisteredHistoryError(
            "registered history ranking is not a permutation"
        )
    return result


def _validate_registered_record(record: _EpochRecord, *, seed: int) -> None:
    _require_registered_seed(seed)
    if type(record) is not _EpochRecord:
        raise TypeError("record must be an exact _EpochRecord")
    if type(record.zero_based_epoch) is not int or not (
        0 <= record.zero_based_epoch < EPOCH_COUNT
    ):
        raise Experiment002RegisteredHistoryError("history epoch is out of range")
    first = record.zero_based_epoch * UPDATES_PER_EPOCH
    if (
        type(record.first_global_update) is not int
        or record.first_global_update != first
        or type(record.last_global_update_inclusive) is not int
        or record.last_global_update_inclusive != first + UPDATES_PER_EPOCH - 1
    ):
        raise Experiment002RegisteredHistoryError(
            "history record has the wrong registered update interval"
        )
    for name, digest_value in (
        ("training_population_digest", record.training_population_digest),
        ("epoch_update_trace_digest", record.epoch_update_trace_digest),
        ("validation_input_digest", record.validation_input_digest),
        ("validation_prediction_digest", record.validation_prediction_digest),
        ("model_tensor_digest", record.model_tensor_digest),
    ):
        _require_sha256(digest_value, name)
    _registered_float64_hex(record.training_cross_entropy)
    _registered_float64_hex(record.validation_cross_entropy)
    confusion = _validated_registered_confusion(record.validation_confusion_matrix)
    if type(record.macro_f1) is not Fraction:
        raise TypeError("macro_f1 must be an exact Fraction")
    if record.macro_f1 != _macro_f1_from_confusion(confusion):
        raise Experiment002RegisteredHistoryError(
            "history macro-F1 differs from the confusion matrix"
        )


def _validated_registered_confusion(value: object) -> _ConfusionMatrix:
    if type(value) is not tuple or len(value) != _CLASS_COUNT:
        raise Experiment002RegisteredHistoryError(
            "registered confusion matrix must have twelve rows"
        )
    rows: list[tuple[int, ...]] = []
    total = 0
    for row in value:
        if type(row) is not tuple or len(row) != _CLASS_COUNT:
            raise Experiment002RegisteredHistoryError(
                "registered confusion rows must have twelve cells"
            )
        checked: list[int] = []
        for cell in row:
            if type(cell) is not int:
                raise TypeError("registered confusion cells must be exact integers")
            if not 0 <= cell <= VALIDATION_EXAMPLE_COUNT:
                raise Experiment002RegisteredHistoryError(
                    "registered confusion cell is outside the population"
                )
            checked.append(cell)
            total += cell
        rows.append(tuple(checked))
    if total != VALIDATION_EXAMPLE_COUNT:
        raise Experiment002RegisteredHistoryError(
            "registered confusion matrix does not contain 10,583 examples"
        )
    return tuple(rows)


def _bound_record_fingerprint(bound: _BoundEpochRecord) -> str:
    if type(bound) is not _BoundEpochRecord:
        raise TypeError("bound record has an invalid type")
    document = _epoch_document(bound.record)
    _require_json_without_floats(document)
    scalar = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    snapshot_document = _bound_snapshot_scalar_document(bound)
    _require_json_without_floats(snapshot_document)
    snapshot_scalar = json.dumps(
        snapshot_document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    payload = bytearray(_HISTORY_RECORD_DOMAIN)
    payload.extend(len(scalar).to_bytes(4, "little"))
    payload.extend(scalar)
    payload.extend(len(snapshot_scalar).to_bytes(4, "little"))
    payload.extend(snapshot_scalar)
    for digest in (
        bound.evaluated_snapshot.authority_sha256,
        bound.evaluated_snapshot.validation_inputs_sha256,
        bound.evaluated_snapshot.validation_predictions_sha256,
        bound.evaluated_snapshot.model_tensor_sha256,
        bound.evaluated_snapshot.optimizer_sha256,
        bound.evaluated_snapshot.torch_rng_sha256,
        bound.handoff_snapshot.runtime_digests.model_sha256,
        bound.handoff_snapshot.runtime_digests.optimizer_sha256,
        bound.handoff_snapshot.runtime_digests.rng_sha256,
    ):
        _require_sha256(digest, "bound evidence digest")
        payload.extend(digest.encode("ascii"))
    return hashlib.sha256(payload).hexdigest()


def _bound_snapshot_scalar_document(
    bound: _BoundEpochRecord,
) -> dict[str, object]:
    evaluated = bound.evaluated_snapshot
    handoff = bound.handoff_snapshot
    for name, integer_value in (
        ("evaluated.process_id", evaluated.process_id),
        ("evaluated.seed", evaluated.seed),
        ("evaluated.zero_based_epoch", evaluated.zero_based_epoch),
        ("evaluated.optimizer_generation", evaluated.optimizer_generation),
        ("evaluated.batch_count", evaluated.batch_count),
        ("handoff.process_id", handoff.process_id),
        ("handoff.seed", handoff.seed),
        ("handoff.zero_based_epoch", handoff.zero_based_epoch),
        ("handoff.optimizer_generation", handoff.optimizer_generation),
    ):
        if type(integer_value) is not int:
            raise TypeError(f"{name} must be an exact integer")
    if (
        type(evaluated.another_training_epoch) is not bool
        or type(handoff.another_training_epoch) is not bool
    ):
        raise TypeError("snapshot training-continuation flags must be exact booleans")
    _require_lower_hex(
        evaluated.registration_head_commit,
        length=40,
        name="registration_head_commit",
    )
    for name, digest_value in (
        ("registration_sha256", evaluated.registration_sha256),
        ("source_bundle_sha256", evaluated.source_bundle_sha256),
        ("validation_inputs_sha256", evaluated.validation_inputs_sha256),
        (
            "validation_predictions_sha256",
            evaluated.validation_predictions_sha256,
        ),
        ("model_tensor_sha256", evaluated.model_tensor_sha256),
        ("optimizer_sha256", evaluated.optimizer_sha256),
        ("torch_rng_sha256", evaluated.torch_rng_sha256),
        ("authority_sha256", evaluated.authority_sha256),
        ("handoff_model_sha256", handoff.runtime_digests.model_sha256),
        ("handoff_optimizer_sha256", handoff.runtime_digests.optimizer_sha256),
        ("handoff_rng_sha256", handoff.runtime_digests.rng_sha256),
    ):
        _require_sha256(digest_value, name)
    if type(handoff.runtime_digests.rng_state) is not bytes:
        raise TypeError("handoff RNG state must be exact bytes")
    return {
        "evaluated": {
            "process_id": evaluated.process_id,
            "seed": evaluated.seed,
            "zero_based_epoch": evaluated.zero_based_epoch,
            "optimizer_generation": evaluated.optimizer_generation,
            "batch_count": evaluated.batch_count,
            "another_training_epoch": int(evaluated.another_training_epoch),
            "registration_head_commit": evaluated.registration_head_commit,
            "registration_sha256": evaluated.registration_sha256,
            "source_bundle_sha256": evaluated.source_bundle_sha256,
            "validation_inputs_sha256": evaluated.validation_inputs_sha256,
            "validation_predictions_sha256": (evaluated.validation_predictions_sha256),
            "model_tensor_sha256": evaluated.model_tensor_sha256,
            "optimizer_sha256": evaluated.optimizer_sha256,
            "torch_rng_sha256": evaluated.torch_rng_sha256,
            "authority_sha256": evaluated.authority_sha256,
        },
        "handoff": {
            "process_id": handoff.process_id,
            "seed": handoff.seed,
            "zero_based_epoch": handoff.zero_based_epoch,
            "optimizer_generation": handoff.optimizer_generation,
            "another_training_epoch": int(handoff.another_training_epoch),
            "model_sha256": handoff.runtime_digests.model_sha256,
            "optimizer_sha256": handoff.runtime_digests.optimizer_sha256,
            "rng_sha256": handoff.runtime_digests.rng_sha256,
            "rng_state_hex": handoff.runtime_digests.rng_state.hex(),
        },
    }


def _bound_record_identity_frame(bound: _BoundEpochRecord) -> tuple[object, ...]:
    """Capture process-local identities separately from canonical digests."""

    if type(bound) is not _BoundEpochRecord:
        raise TypeError("bound record has an invalid type")
    evaluated = bound.evaluated_snapshot
    handoff = bound.handoff_snapshot
    return (
        id(bound),
        id(bound.record),
        id(bound.evaluated_epoch),
        id(evaluated),
        id(evaluated.registration),
        id(evaluated.handoff),
        id(evaluated.validation_inputs),
        id(evaluated.validation_evidence),
        id(evaluated.model_tensors),
        id(handoff),
        id(handoff.executor),
        id(handoff.registration),
        id(handoff.validation_inputs),
        id(handoff.bridge_authority),
        id(handoff.bridge_session_token),
        id(handoff.population),
        id(handoff.epoch_trace),
        id(handoff.previous_history_barrier),
        id(handoff.one_shot_token),
        id(bound.population),
        id(bound.epoch_trace),
        id(bound.model_tensors),
    )


def _rank_frame(rank: RegisteredEpochRank) -> tuple[object, ...]:
    if type(rank) is not RegisteredEpochRank:
        raise TypeError("registered epoch rank has an invalid type")
    if type(rank.zero_based_epoch) is not int or not (
        0 <= rank.zero_based_epoch < EPOCH_COUNT
    ):
        raise Experiment002RegisteredHistoryError(
            "rank epoch must be an exact registered epoch integer"
        )
    if (
        type(rank.macro_f1_exact_numerator) is not int
        or type(rank.macro_f1_exact_denominator) is not int
        or rank.macro_f1_exact_denominator <= 0
        or not 0 <= rank.macro_f1_exact_numerator <= rank.macro_f1_exact_denominator
    ):
        raise Experiment002RegisteredHistoryError(
            "rank macro-F1 must be an exact nonnegative reduced fraction frame"
        )
    value = rank.validation_cross_entropy_float64_hex
    if type(value) is not str or value != value.lower():
        raise TypeError("rank validation cross-entropy hex must be an exact string")
    try:
        parsed = float.fromhex(value)
    except ValueError as error:
        raise Experiment002RegisteredHistoryError(
            "rank validation cross-entropy hex is invalid"
        ) from error
    if (
        not np.isfinite(parsed)
        or parsed < 0.0
        or np.signbit(parsed)
        or parsed.hex() != value
    ):
        raise Experiment002RegisteredHistoryError(
            "rank validation cross-entropy hex is not canonical finite binary64"
        )
    return (
        rank.zero_based_epoch,
        rank.macro_f1_exact_numerator,
        rank.macro_f1_exact_denominator,
        rank.validation_cross_entropy_float64_hex,
    )


def _same_exact_frame(left: object, right: object) -> bool:
    """Compare authority frames without bool/int or subclass equality leaks."""

    if type(left) is not type(right):
        return False
    if type(left) is tuple:
        left_tuple = cast(tuple[object, ...], left)
        right_tuple = cast(tuple[object, ...], right)
        return len(left_tuple) == len(right_tuple) and all(
            _same_exact_frame(left_item, right_item)
            for left_item, right_item in zip(left_tuple, right_tuple, strict=True)
        )
    if type(left) in (str, int, bool, bytes, type(None)):
        return left == right
    return left is right


def _require_history_payload_types(
    state: _HistoryState,
    guard: _HistoryGuard,
) -> None:
    if type(state) is not _HistoryState or type(guard) is not _HistoryGuard:
        raise TypeError("history state and guard must have exact issuer types")
    if type(state.token) is not object or type(guard.token) is not object:
        raise TypeError("history tokens must be exact objects")
    if type(state.process_id) is not int or type(guard.process_id) is not int:
        raise TypeError("history process IDs must be exact integers")
    if state.process_id <= 0 or guard.process_id <= 0:
        raise Experiment002RegisteredHistoryError(
            "history process IDs must be positive"
        )
    _require_registered_seed(state.seed)
    _require_registered_seed(guard.seed)
    _require_sha256(state.validation_inputs_sha256, "validation_inputs_sha256")
    _require_sha256(guard.validation_inputs_sha256, "guard validation_inputs_sha256")
    if type(state.phase) is not str or state.phase not in (
        "OPEN",
        "COMPLETE",
        "FAILED",
    ):
        raise Experiment002RegisteredHistoryError(
            "history phase must be an exact registered phase string"
        )
    if type(state.records) is not list:
        raise TypeError("history records must be an exact list")
    if type(state.lock) is not type(threading.RLock()) or type(guard.lock) is not type(
        threading.RLock()
    ):
        raise TypeError("history locks must be exact RLocks")
    if (
        state.active_barrier_token is not None
        and type(state.active_barrier_token) is not object
    ):
        raise TypeError("active history barrier token must be an exact object")


def _require_barrier_payload_types(
    state: _BarrierState,
    guard: _BarrierGuard,
) -> None:
    if type(state) is not _BarrierState or type(guard) is not _BarrierGuard:
        raise TypeError("barrier state and guard must have exact issuer types")
    if (
        type(state.token) is not object
        or type(guard.token) is not object
        or type(state.history_token) is not object
        or type(guard.history_token) is not object
    ):
        raise TypeError("barrier tokens must be exact objects")
    if type(state.process_id) is not int or type(guard.process_id) is not int:
        raise TypeError("barrier process IDs must be exact integers")
    if state.process_id <= 0 or guard.process_id <= 0:
        raise Experiment002RegisteredHistoryError(
            "barrier process IDs must be positive"
        )
    _require_registered_seed(state.seed)
    _require_registered_seed(guard.seed)
    for value, name in (
        (state.accepted_epoch, "accepted_epoch"),
        (guard.accepted_epoch, "guard accepted_epoch"),
    ):
        if type(value) is not int or not 0 <= value < EPOCH_COUNT:
            raise Experiment002RegisteredHistoryError(
                f"{name} must be an exact registered epoch integer"
            )
    _require_sha256(state.record_fingerprint, "barrier record_fingerprint")
    _require_sha256(guard.record_fingerprint, "guard record_fingerprint")
    if type(state.phase) is not str or state.phase not in (
        "ISSUED",
        "CONSUMED",
        "RETIRED",
        "FAILED",
    ):
        raise Experiment002RegisteredHistoryError(
            "barrier phase must be an exact registered phase string"
        )


def _require_completed_payload_types(
    state: _CompletedState,
    guard: _CompletedGuard,
) -> None:
    if type(state) is not _CompletedState or type(guard) is not _CompletedGuard:
        raise TypeError("completed state and guard must have exact issuer types")
    if (
        type(state.token) is not object
        or type(guard.token) is not object
        or type(state.history_token) is not object
        or type(guard.history_token) is not object
    ):
        raise TypeError("completed history tokens must be exact objects")
    if type(state.process_id) is not int or type(guard.process_id) is not int:
        raise TypeError("completed history process IDs must be exact integers")
    if state.process_id <= 0 or guard.process_id <= 0:
        raise Experiment002RegisteredHistoryError(
            "completed history process IDs must be positive"
        )
    _require_registered_seed(state.seed)
    _require_registered_seed(guard.seed)
    if type(state.records) is not tuple or type(guard.records) is not tuple:
        raise TypeError("completed history records must be exact tuples")
    if len(state.records) != EPOCH_COUNT or len(guard.records) != EPOCH_COUNT:
        raise Experiment002RegisteredHistoryError(
            "completed history must contain exactly thirty records"
        )
    if any(type(record) is not _BoundEpochRecord for record in state.records):
        raise TypeError("completed history records have invalid exact types")
    if (
        type(state.record_fingerprints) is not tuple
        or type(guard.record_fingerprints) is not tuple
    ):
        raise TypeError("completed record fingerprints must be exact tuples")
    if (
        len(state.record_fingerprints) != EPOCH_COUNT
        or len(guard.record_fingerprints) != EPOCH_COUNT
    ):
        raise Experiment002RegisteredHistoryError(
            "completed record fingerprint count is invalid"
        )
    for digest in (*state.record_fingerprints, *guard.record_fingerprints):
        _require_sha256(digest, "completed record fingerprint")
    if (
        type(state.record_identity_frames) is not tuple
        or type(guard.record_identity_frames) is not tuple
    ):
        raise TypeError("completed identity frames must be exact tuples")
    for frames in (state.record_identity_frames, guard.record_identity_frames):
        if len(frames) != EPOCH_COUNT or any(
            type(frame) is not tuple or any(type(value) is not int for value in frame)
            for frame in frames
        ):
            raise TypeError("completed identity frames contain non-exact values")
    if (
        type(state.canonical_json_bytes) is not bytes
        or type(guard.canonical_json_bytes) is not bytes
    ):
        raise TypeError("completed canonical JSON payloads must be exact bytes")
    if not state.canonical_json_bytes.endswith(
        b"\n"
    ) or not guard.canonical_json_bytes.endswith(b"\n"):
        raise Experiment002RegisteredHistoryError(
            "completed canonical JSON payload must end in one LF"
        )
    _require_sha256(state.history_sha256, "completed history_sha256")
    _require_sha256(guard.history_sha256, "guard history_sha256")
    if (
        state.history_sha256
        != hashlib.sha256(_HISTORY_DOMAIN + state.canonical_json_bytes).hexdigest()
    ):
        raise Experiment002RegisteredHistoryError(
            "completed history digest differs from canonical bytes"
        )
    if type(state.ranked_epochs) is not tuple or type(guard.ranked_epochs) is not tuple:
        raise TypeError("completed ranks must be exact tuples")
    if (
        len(state.ranked_epochs) != EPOCH_COUNT
        or len(guard.ranked_epochs) != EPOCH_COUNT
    ):
        raise Experiment002RegisteredHistoryError(
            "completed rank count must be exactly thirty"
        )
    observed_rank_frames = tuple(_rank_frame(rank) for rank in state.ranked_epochs)
    guard_rank_frames = tuple(_rank_frame(rank) for rank in guard.ranked_epochs)
    if type(state.rank_frames) is not tuple or type(guard.rank_frames) is not tuple:
        raise TypeError("completed rank frames must be exact tuples")
    if (
        not _same_exact_frame(state.rank_frames, observed_rank_frames)
        or not _same_exact_frame(guard.rank_frames, guard_rank_frames)
        or len({rank.zero_based_epoch for rank in state.ranked_epochs}) != EPOCH_COUNT
    ):
        raise Experiment002RegisteredHistoryError(
            "completed rank frames are invalid or not a permutation"
        )
    if (
        type(state.winner) is not RegisteredEvaluatedEpoch
        or type(guard.winner) is not RegisteredEvaluatedEpoch
    ):
        raise TypeError("completed winners must be exact evaluated capabilities")


def _history_lifecycle_frame(lifecycle: _HistoryLifecycle) -> tuple[object, ...]:
    if type(lifecycle) is not _HistoryLifecycle:
        raise TypeError("history lifecycle must have its exact issuer type")
    return (
        lifecycle.token,
        lifecycle.process_id,
        lifecycle.phase,
        lifecycle.records,
        lifecycle.record_fingerprints,
        lifecycle.record_identity_frames,
        lifecycle.active_barrier,
        lifecycle.active_barrier_token,
        lifecycle.completed,
    )


def _barrier_lifecycle_frame(lifecycle: _BarrierLifecycle) -> tuple[object, ...]:
    if type(lifecycle) is not _BarrierLifecycle:
        raise TypeError("barrier lifecycle must have its exact issuer type")
    return (lifecycle.token, lifecycle.process_id, lifecycle.phase)


def _completed_lifecycle_frame(
    lifecycle: _CompletedLifecycle,
) -> tuple[object, ...]:
    if type(lifecycle) is not _CompletedLifecycle:
        raise TypeError("completed lifecycle must have its exact issuer type")
    return (lifecycle.token, lifecycle.process_id, lifecycle.phase)


def _same_evaluated_snapshot(
    left: _RegisteredEvaluatedEpochSnapshot,
    right: _RegisteredEvaluatedEpochSnapshot,
) -> bool:
    return (
        type(left) is _RegisteredEvaluatedEpochSnapshot
        and type(right) is _RegisteredEvaluatedEpochSnapshot
        and left.registration is right.registration
        and left.handoff is right.handoff
        and left.validation_inputs is right.validation_inputs
        and left.validation_evidence is right.validation_evidence
        and left.model_tensors is right.model_tensors
        and left.process_id == right.process_id
        and left.seed == right.seed
        and left.zero_based_epoch == right.zero_based_epoch
        and left.optimizer_generation == right.optimizer_generation
        and left.batch_count == right.batch_count
        and left.another_training_epoch is right.another_training_epoch
        and left.registration_head_commit == right.registration_head_commit
        and left.registration_sha256 == right.registration_sha256
        and left.source_bundle_sha256 == right.source_bundle_sha256
        and left.validation_inputs_sha256 == right.validation_inputs_sha256
        and left.validation_predictions_sha256 == right.validation_predictions_sha256
        and left.model_tensor_sha256 == right.model_tensor_sha256
        and left.optimizer_sha256 == right.optimizer_sha256
        and left.torch_rng_sha256 == right.torch_rng_sha256
        and left.authority_sha256 == right.authority_sha256
    )


def _same_handoff_snapshot(
    left: _RegisteredExecutorEpochHandoffSnapshot,
    right: _RegisteredExecutorEpochHandoffSnapshot,
) -> bool:
    return (
        type(left) is _RegisteredExecutorEpochHandoffSnapshot
        and type(right) is _RegisteredExecutorEpochHandoffSnapshot
        and left.executor is right.executor
        and left.registration is right.registration
        and left.validation_inputs is right.validation_inputs
        and left.process_id == right.process_id
        and left.bridge_authority is right.bridge_authority
        and left.bridge_session_token is right.bridge_session_token
        and left.population is right.population
        and left.epoch_trace is right.epoch_trace
        and left.seed == right.seed
        and left.zero_based_epoch == right.zero_based_epoch
        and left.optimizer_generation == right.optimizer_generation
        and left.runtime_digests == right.runtime_digests
        and left.previous_history_barrier is right.previous_history_barrier
        and left.one_shot_token is right.one_shot_token
        and left.another_training_epoch is right.another_training_epoch
    )


def _lifecycle_from_state(state: _HistoryState) -> _HistoryLifecycle:
    records = tuple(_history_records(state))
    return _HistoryLifecycle(
        token=state.token,
        process_id=state.process_id,
        phase=state.phase,
        records=records,
        record_fingerprints=tuple(
            _bound_record_fingerprint(record) for record in records
        ),
        record_identity_frames=tuple(
            _bound_record_identity_frame(record) for record in records
        ),
        active_barrier=state.active_barrier,
        active_barrier_token=state.active_barrier_token,
        completed=state.completed,
    )


def _publish_history_lifecycle(
    history: RegisteredTrainingHistory,
    state: _HistoryState,
) -> None:
    with _REGISTRY_LOCK:
        _HISTORY_LIFECYCLES[history] = _lifecycle_from_state(state)


def _transition_barrier(
    barrier: RegisteredHistoryBarrier,
    state: _BarrierState,
    *,
    phase: Literal["CONSUMED", "RETIRED"],
    _transition_truth: Callable[..., None] = _transition_barrier_truth,
) -> None:
    with _REGISTRY_LOCK:
        guard = _BARRIER_GUARDS.get(barrier)
    if type(guard) is not _BarrierGuard:
        raise Experiment002RegisteredHistoryError("history barrier guard is missing")
    expected = state.phase
    _transition_truth(
        barrier,
        state,
        guard,
        expected=expected,
        phase=phase,
    )
    state.phase = phase
    with _REGISTRY_LOCK:
        _BARRIER_LIFECYCLES[barrier] = _BarrierLifecycle(
            token=state.token,
            process_id=state.process_id,
            phase=phase,
        )


def _terminal_fail_history(
    history: RegisteredTrainingHistory,
    state: _HistoryState,
    _fail_truth: Callable[..., None] = _fail_history_truth,
) -> None:
    _fail_truth(history, state)
    state.phase = "FAILED"
    with _REGISTRY_LOCK:
        _FAILED_HISTORIES.add(history)
        with contextlib.suppress(BaseException):
            _HISTORY_LIFECYCLES[history] = _lifecycle_from_state(state)
    if state.active_barrier is not None:
        with _REGISTRY_LOCK:
            barrier_state = _BARRIERS.get(state.active_barrier)
        if type(barrier_state) is _BarrierState:
            _terminal_fail_barrier(state.active_barrier, barrier_state)


def _terminal_fail_foreign_history(
    owner: RegisteredTrainingHistory,
    receiver: RegisteredTrainingHistory,
) -> None:
    if owner is receiver:
        return
    with _REGISTRY_LOCK:
        owner_state = _HISTORIES.get(owner)
    if type(owner_state) is _HistoryState:
        _terminal_fail_history(owner, owner_state)


def _terminal_fail_barrier(
    barrier: RegisteredHistoryBarrier,
    state: _BarrierState,
    _fail_truth: Callable[..., None] = _fail_barrier_truth,
) -> None:
    _fail_truth(barrier, state)
    state.phase = "FAILED"
    with _REGISTRY_LOCK:
        _FAILED_BARRIERS.add(barrier)
        _BARRIER_LIFECYCLES[barrier] = _BarrierLifecycle(
            token=state.token,
            process_id=state.process_id,
            phase="FAILED",
        )


def _terminal_fail_completed(
    result: RegisteredCompletedTrainingHistory,
    state: _CompletedState,
    _fail_truth: Callable[..., None] = _fail_completed_truth,
) -> None:
    _fail_truth(result, state)
    with _REGISTRY_LOCK:
        _FAILED_COMPLETED.add(result)
        _COMPLETED_LIFECYCLES[result] = _CompletedLifecycle(
            token=state.token,
            process_id=state.process_id,
            phase="FAILED",
        )


def _require_registered_seed(seed: object) -> None:
    if type(seed) is not int:
        raise TypeError("seed must be an exact integer")
    if seed not in REGISTERED_SEEDS:
        raise Experiment002RegisteredHistoryError(
            "seed is not one of the three registered training seeds"
        )


def _registered_float64_hex(value: object) -> str:
    if type(value) is not np.float64:
        raise TypeError("history cross-entropy must be an exact NumPy float64")
    if not np.isfinite(value) or value < np.float64(0.0) or np.signbit(value):
        raise Experiment002RegisteredHistoryError(
            "history cross-entropy must be finite, nonnegative, and not negative zero"
        )
    result = float(value).hex()
    if result != result.lower():
        raise Experiment002RegisteredHistoryError(
            "history float64 hex must be lowercase"
        )
    return result


def _require_sha256(value: object, name: str) -> None:
    _require_lower_hex(value, length=64, name=name)


def _require_lower_hex(value: object, *, length: int, name: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if len(value) != length or any(character not in _LOWER_HEX for character in value):
        raise Experiment002RegisteredHistoryError(
            f"{name} must be one lowercase {length}-character hex string"
        )


# Sanctioned producers and verifiers captured these closure callables in their
# defaults.  Removing convenience aliases prevents accidental/direct reuse; it
# is defense in depth, not secrecy from privileged same-process reflection.
del _record_history_truth
del _require_history_truth
del _transition_history_truth
del _fail_history_truth
del _evaluated_truth_owner
del _claim_evaluated_truth
del _record_barrier_truth
del _require_barrier_truth
del _transition_barrier_truth
del _fail_barrier_truth
del _record_completed_truth
del _require_completed_truth
del _fail_completed_truth
