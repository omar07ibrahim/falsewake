"""Tiny synthetic-only training-history kernel for Experiment 002.

The source-bound run registration does not exist yet.  This module therefore
cannot consume registered epochs, metrics, evidence, populations, checkpoints,
or artifacts.  It exercises only the canonical history serialization and
ranking mechanics with a process-local, deliberately tiny synthetic capability
chain.  Its completed value is explicitly unregistered and must never be
trusted by a future registered adapter.
"""

from __future__ import annotations

import hashlib
import json
import threading
import weakref
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Final, Literal, NoReturn

import numpy as np

_HISTORY_DOMAIN: Final = b"falsewake-exp002-history-v1\0"
_RECORD_AUTHORITY_DOMAIN: Final = b"falsewake-exp002-history-record-authority-v1\0"
_SCHEMA_VERSION: Final = 1
_EXPERIMENT: Final = "002"
_SYNTHETIC_SEED: Final = 0
_CLASS_COUNT: Final = 12
_MAX_SYNTHETIC_EPOCHS: Final = 4
_MAX_SYNTHETIC_UPDATES_PER_EPOCH: Final = 4
_MAX_SYNTHETIC_CONFUSION_CELL: Final = 4
_SYNTHETIC_ROUTE_MARKER: Final = object()
_LOWER_HEX: Final = frozenset("0123456789abcdef")

type _SessionPhase = Literal["OPEN", "COMPLETE", "FAILED"]
type _CapabilityPhase = Literal["ISSUED", "CONSUMED", "FAILED"]
type _ConfusionMatrix = tuple[tuple[int, ...], ...]
type _SessionAnchor = tuple[
    object,
    object,
    _SyntheticHistoryLayout,
    int,
    int,
    str,
    str,
    threading.RLock,
]
type _LifecycleAnchor = tuple[
    object,
    _SessionPhase,
    tuple[str, ...],
    EvaluatedEpoch | None,
    object | None,
    str | None,
    HistoryAcceptedEpoch | None,
    object | None,
    int | None,
]


class Experiment002TrainingHistoryError(ValueError):
    """The synthetic history kernel violated its exact local contract."""


@dataclass(frozen=True, slots=True)
class _SyntheticHistoryLayout:
    """Private bounded layout which cannot represent a full training run."""

    epoch_count: int
    updates_per_epoch: int = 1

    def __post_init__(self) -> None:
        _require_positive_int(self.epoch_count, "epoch_count")
        _require_positive_int(self.updates_per_epoch, "updates_per_epoch")
        if self.epoch_count > _MAX_SYNTHETIC_EPOCHS:
            raise Experiment002TrainingHistoryError(
                "synthetic history exceeds the tiny epoch limit"
            )
        if self.updates_per_epoch > _MAX_SYNTHETIC_UPDATES_PER_EPOCH:
            raise Experiment002TrainingHistoryError(
                "synthetic history exceeds the tiny update limit"
            )


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
class EvaluatedEpoch:
    """Opaque one-use synthetic epoch issued by this module only."""

    def __init__(self) -> None:
        raise TypeError("evaluated epochs are issuer-only")

    def __copy__(self) -> NoReturn:
        raise TypeError("evaluated epochs cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        del memo
        raise TypeError("evaluated epochs cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("evaluated epochs cannot be serialized")


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
class HistoryAcceptedEpoch:
    """Opaque one-use barrier proving one session accepted one exact epoch."""

    def __init__(self) -> None:
        raise TypeError("history acceptance barriers are issuer-only")

    def __copy__(self) -> NoReturn:
        raise TypeError("history acceptance barriers cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        del memo
        raise TypeError("history acceptance barriers cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("history acceptance barriers cannot be serialized")


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
class TrainingHistorySession:
    """Opaque terminal-on-failure owner of one tiny ordered history."""

    def __init__(self) -> None:
        raise TypeError("training history sessions are issuer-only")

    def consume(self, evaluated_epoch: EvaluatedEpoch) -> HistoryAcceptedEpoch:
        """Consume one exact next epoch and return its one-use barrier."""

        return _consume_synthetic_evaluated_epoch(self, evaluated_epoch)

    def complete(
        self,
        accepted_epoch: HistoryAcceptedEpoch,
    ) -> _UnregisteredCompletedTrainingHistory:
        """Complete only after the layout's final accepted epoch."""

        return _complete_synthetic_training_history(self, accepted_epoch)

    def __copy__(self) -> NoReturn:
        raise TypeError("training history sessions cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        del memo
        raise TypeError("training history sessions cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("training history sessions cannot be serialized")


@dataclass(frozen=True, slots=True)
class _EpochRecord:
    zero_based_epoch: int
    first_global_update: int
    last_global_update_inclusive: int
    training_population_digest: str
    training_cross_entropy: np.float64
    epoch_update_trace_digest: str
    validation_input_digest: str
    validation_confusion_matrix: _ConfusionMatrix
    validation_cross_entropy: np.float64
    macro_f1: Fraction
    validation_prediction_digest: str
    model_tensor_digest: str


@dataclass(frozen=True, slots=True)
class _UnregisteredEpochRank:
    """Non-authoritative rank metadata derived from one synthetic record."""

    zero_based_epoch: int
    macro_f1_exact_numerator: int
    macro_f1_exact_denominator: int
    validation_cross_entropy_float64_hex: str


@dataclass(frozen=True, slots=True)
class _UnregisteredCompletedTrainingHistory:
    """Forgeable value output, never evidence or checkpoint authority."""

    synthetic_seed: int
    epoch_count: int
    canonical_json_bytes: bytes
    history_sha256: str
    ranked_epochs: tuple[_UnregisteredEpochRank, ...]

    @property
    def winner(self) -> _UnregisteredEpochRank:
        """Return the best synthetic epoch under the frozen rank order."""

        return self.ranked_epochs[0]


@dataclass(frozen=True, slots=True)
class _SyntheticHistorySessionSnapshot:
    phase: _SessionPhase
    epoch_count: int
    accepted_epoch_count: int
    has_active_epoch: bool
    has_active_barrier: bool


@dataclass(slots=True)
class _SessionState:
    token: object
    route_marker: object
    layout: _SyntheticHistoryLayout
    validation_input_digest: str
    complete_update_trace_digest: str
    phase: _SessionPhase = "OPEN"
    records: list[_EpochRecord] = field(default_factory=list)
    active_epoch: EvaluatedEpoch | None = None
    active_epoch_token: object | None = None
    active_barrier: HistoryAcceptedEpoch | None = None
    active_barrier_token: object | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)


@dataclass(slots=True)
class _EvaluatedEpochState:
    owner: TrainingHistorySession
    session_token: object
    epoch_token: object
    route_marker: object
    record: _EpochRecord
    phase: _CapabilityPhase = "ISSUED"


@dataclass(slots=True)
class _AcceptedEpochState:
    owner: TrainingHistorySession
    session_token: object
    barrier_token: object
    route_marker: object
    accepted_epoch: int
    phase: _CapabilityPhase = "ISSUED"


@dataclass(frozen=True, slots=True)
class _SessionGuard:
    token: object
    route_marker: object
    layout: _SyntheticHistoryLayout
    epoch_count: int
    updates_per_epoch: int
    validation_input_digest: str
    complete_update_trace_digest: str
    lock: threading.RLock


@dataclass(frozen=True, slots=True)
class _SessionLifecycle:
    token: object
    phase: _SessionPhase
    record_fingerprints: tuple[str, ...]
    active_epoch: EvaluatedEpoch | None
    active_epoch_token: object | None
    active_epoch_fingerprint: str | None
    active_barrier: HistoryAcceptedEpoch | None
    active_barrier_token: object | None
    active_barrier_epoch: int | None


_SESSIONS: weakref.WeakKeyDictionary[TrainingHistorySession, _SessionState] = (
    weakref.WeakKeyDictionary()
)
_SESSION_GUARDS: weakref.WeakKeyDictionary[TrainingHistorySession, _SessionGuard] = (
    weakref.WeakKeyDictionary()
)
_SESSION_ANCHORS: weakref.WeakKeyDictionary[TrainingHistorySession, _SessionAnchor] = (
    weakref.WeakKeyDictionary()
)
_SESSION_LIFECYCLES: weakref.WeakKeyDictionary[
    TrainingHistorySession, _SessionLifecycle
] = weakref.WeakKeyDictionary()
_SESSION_LIFECYCLE_ANCHORS: weakref.WeakKeyDictionary[
    TrainingHistorySession, _LifecycleAnchor
] = weakref.WeakKeyDictionary()
_ISSUED_SESSIONS: weakref.WeakSet[TrainingHistorySession] = weakref.WeakSet()
_FAILED_SESSIONS: weakref.WeakSet[TrainingHistorySession] = weakref.WeakSet()
_EVALUATED_EPOCHS: weakref.WeakKeyDictionary[EvaluatedEpoch, _EvaluatedEpochState] = (
    weakref.WeakKeyDictionary()
)
_ACCEPTED_EPOCHS: weakref.WeakKeyDictionary[
    HistoryAcceptedEpoch, _AcceptedEpochState
] = weakref.WeakKeyDictionary()
_REGISTRY_LOCK = threading.RLock()
_FLOW_LOCK = threading.RLock()


def _create_synthetic_training_history_session(
    *,
    layout: _SyntheticHistoryLayout,
    validation_input_digest: str,
    complete_update_trace_digest: str,
) -> TrainingHistorySession:
    """Issue one unregistered history session with fixed synthetic seed zero."""

    _require_layout(layout)
    _require_digest(validation_input_digest, "validation_input_digest")
    _require_digest(complete_update_trace_digest, "complete_update_trace_digest")
    result = object.__new__(TrainingHistorySession)
    state = _SessionState(
        token=object(),
        route_marker=_SYNTHETIC_ROUTE_MARKER,
        layout=layout,
        validation_input_digest=validation_input_digest,
        complete_update_trace_digest=complete_update_trace_digest,
    )
    guard = _SessionGuard(
        token=state.token,
        route_marker=_SYNTHETIC_ROUTE_MARKER,
        layout=layout,
        epoch_count=layout.epoch_count,
        updates_per_epoch=layout.updates_per_epoch,
        validation_input_digest=validation_input_digest,
        complete_update_trace_digest=complete_update_trace_digest,
        lock=state.lock,
    )
    lifecycle = _SessionLifecycle(
        token=state.token,
        phase="OPEN",
        record_fingerprints=(),
        active_epoch=None,
        active_epoch_token=None,
        active_epoch_fingerprint=None,
        active_barrier=None,
        active_barrier_token=None,
        active_barrier_epoch=None,
    )
    anchor: _SessionAnchor = (
        state.token,
        _SYNTHETIC_ROUTE_MARKER,
        layout,
        layout.epoch_count,
        layout.updates_per_epoch,
        validation_input_digest,
        complete_update_trace_digest,
        state.lock,
    )
    with _REGISTRY_LOCK:
        _SESSIONS[result] = state
        _SESSION_GUARDS[result] = guard
        _SESSION_ANCHORS[result] = anchor
        _SESSION_LIFECYCLES[result] = lifecycle
        _SESSION_LIFECYCLE_ANCHORS[result] = _lifecycle_anchor(lifecycle)
        _ISSUED_SESSIONS.add(result)
    _synthetic_history_session_snapshot(result)
    return result


def _issue_synthetic_evaluated_epoch(
    session: TrainingHistorySession,
    *,
    accepted_previous: HistoryAcceptedEpoch | None,
    zero_based_epoch: int,
    training_population_digest: str,
    training_cross_entropy: np.float64,
    epoch_update_trace_digest: str,
    validation_confusion_matrix: _ConfusionMatrix,
    validation_cross_entropy: np.float64,
    validation_prediction_digest: str,
    model_tensor_digest: str,
) -> EvaluatedEpoch:
    """Issue one tiny raw synthetic observation as an opaque epoch capability."""

    state = _issued_session_state(session)
    trusted_lock = _trusted_session_lock(session, state)
    with _FLOW_LOCK, trusted_lock:
        _validate_session_or_fail(session, state)
        _require_open_session(state)
        previous_state: _AcceptedEpochState | None = None
        try:
            expected_epoch = len(state.records)
            if expected_epoch >= state.layout.epoch_count:
                raise Experiment002TrainingHistoryError(
                    "all synthetic epochs are already accepted"
                )
            if state.active_epoch is not None or state.active_epoch_token is not None:
                raise Experiment002TrainingHistoryError(
                    "one evaluated epoch is already active"
                )
            if expected_epoch == 0:
                if accepted_previous is not None:
                    if type(accepted_previous) is not HistoryAcceptedEpoch:
                        raise TypeError(
                            "accepted_previous must be an exact HistoryAcceptedEpoch"
                        )
                    previous_state = _issued_accepted_epoch_state(accepted_previous)
                    raise Experiment002TrainingHistoryError(
                        "epoch zero cannot consume a prior acceptance barrier"
                    )
                if state.active_barrier is not None:
                    raise Experiment002TrainingHistoryError(
                        "epoch zero cannot consume a prior acceptance barrier"
                    )
            else:
                if type(accepted_previous) is not HistoryAcceptedEpoch:
                    raise TypeError(
                        "accepted_previous must be an exact HistoryAcceptedEpoch"
                    )
                previous_state = _issued_accepted_epoch_state(accepted_previous)
                if (
                    state.active_barrier is not accepted_previous
                    or state.active_barrier_token is not previous_state.barrier_token
                    or previous_state.owner is not session
                    or previous_state.session_token is not state.token
                    or previous_state.route_marker is not _SYNTHETIC_ROUTE_MARKER
                    or previous_state.accepted_epoch != expected_epoch - 1
                    or previous_state.phase != "ISSUED"
                ):
                    raise Experiment002TrainingHistoryError(
                        "acceptance barrier does not match the next epoch"
                    )
            _require_exact_int(zero_based_epoch, "zero_based_epoch")
            if zero_based_epoch != expected_epoch:
                raise Experiment002TrainingHistoryError(
                    "synthetic epochs cannot skip, repeat, or reorder indices"
                )
            record = _build_epoch_record(
                state,
                zero_based_epoch=zero_based_epoch,
                training_population_digest=training_population_digest,
                training_cross_entropy=training_cross_entropy,
                epoch_update_trace_digest=epoch_update_trace_digest,
                validation_confusion_matrix=validation_confusion_matrix,
                validation_cross_entropy=validation_cross_entropy,
                validation_prediction_digest=validation_prediction_digest,
                model_tensor_digest=model_tensor_digest,
            )
            result = object.__new__(EvaluatedEpoch)
            epoch_token = object()
            issued = _EvaluatedEpochState(
                owner=session,
                session_token=state.token,
                epoch_token=epoch_token,
                route_marker=_SYNTHETIC_ROUTE_MARKER,
                record=record,
            )
            with _REGISTRY_LOCK:
                _EVALUATED_EPOCHS[result] = issued
            if previous_state is not None:
                previous_state.phase = "CONSUMED"
                state.active_barrier = None
                state.active_barrier_token = None
            state.active_epoch = result
            state.active_epoch_token = epoch_token
            _advance_session_lifecycle(session, state, event="ISSUE")
            return result
        except BaseException:
            if previous_state is not None:
                previous_state.phase = "FAILED"
                try:
                    _terminal_fail_foreign_owner(previous_state.owner, session)
                finally:
                    _terminal_fail_session(session, state)
            else:
                _terminal_fail_session(session, state)
            raise


def _consume_synthetic_evaluated_epoch(
    session: TrainingHistorySession,
    evaluated_epoch: EvaluatedEpoch,
) -> HistoryAcceptedEpoch:
    state = _issued_session_state(session)
    trusted_lock = _trusted_session_lock(session, state)
    with _FLOW_LOCK, trusted_lock:
        _validate_session_or_fail(session, state)
        _require_open_session(state)
        epoch_state: _EvaluatedEpochState | None = None
        try:
            if type(evaluated_epoch) is not EvaluatedEpoch:
                raise TypeError("evaluated_epoch must be an exact EvaluatedEpoch")
            epoch_state = _issued_evaluated_epoch_state(evaluated_epoch)
            expected_epoch = len(state.records)
            if (
                state.active_epoch is not evaluated_epoch
                or state.active_epoch_token is not epoch_state.epoch_token
                or state.active_barrier is not None
                or epoch_state.owner is not session
                or epoch_state.session_token is not state.token
                or epoch_state.route_marker is not _SYNTHETIC_ROUTE_MARKER
                or epoch_state.phase != "ISSUED"
                or epoch_state.record.zero_based_epoch != expected_epoch
            ):
                raise Experiment002TrainingHistoryError(
                    "evaluated epoch is replayed, foreign, or out of order"
                )
            _validate_epoch_record(epoch_state.record, state)
            barrier = object.__new__(HistoryAcceptedEpoch)
            barrier_token = object()
            accepted = _AcceptedEpochState(
                owner=session,
                session_token=state.token,
                barrier_token=barrier_token,
                route_marker=_SYNTHETIC_ROUTE_MARKER,
                accepted_epoch=expected_epoch,
            )
            with _REGISTRY_LOCK:
                _ACCEPTED_EPOCHS[barrier] = accepted
            state.records.append(epoch_state.record)
            epoch_state.phase = "CONSUMED"
            state.active_epoch = None
            state.active_epoch_token = None
            state.active_barrier = barrier
            state.active_barrier_token = barrier_token
            _advance_session_lifecycle(session, state, event="CONSUME")
            return barrier
        except BaseException:
            if epoch_state is not None:
                epoch_state.phase = "FAILED"
                try:
                    _terminal_fail_foreign_owner(epoch_state.owner, session)
                finally:
                    _terminal_fail_session(session, state)
            else:
                _terminal_fail_session(session, state)
            raise


def _complete_synthetic_training_history(
    session: TrainingHistorySession,
    accepted_epoch: HistoryAcceptedEpoch,
) -> _UnregisteredCompletedTrainingHistory:
    state = _issued_session_state(session)
    trusted_lock = _trusted_session_lock(session, state)
    with _FLOW_LOCK, trusted_lock:
        _validate_session_or_fail(session, state)
        _require_open_session(state)
        barrier_state: _AcceptedEpochState | None = None
        try:
            if type(accepted_epoch) is not HistoryAcceptedEpoch:
                raise TypeError("accepted_epoch must be an exact HistoryAcceptedEpoch")
            barrier_state = _issued_accepted_epoch_state(accepted_epoch)
            if (
                len(state.records) != state.layout.epoch_count
                or state.active_epoch is not None
                or state.active_barrier is not accepted_epoch
                or state.active_barrier_token is not barrier_state.barrier_token
                or barrier_state.owner is not session
                or barrier_state.session_token is not state.token
                or barrier_state.route_marker is not _SYNTHETIC_ROUTE_MARKER
                or barrier_state.accepted_epoch != state.layout.epoch_count - 1
                or barrier_state.phase != "ISSUED"
            ):
                raise Experiment002TrainingHistoryError(
                    "history completion requires the exact final epoch barrier"
                )
            for expected_epoch, record in enumerate(state.records):
                if record.zero_based_epoch != expected_epoch:
                    raise Experiment002TrainingHistoryError(
                        "history records are not in exact epoch order"
                    )
                _validate_epoch_record(record, state)
            first = _canonical_history_bytes(state)
            second = _canonical_history_bytes(state)
            if first != second:
                raise Experiment002TrainingHistoryError(
                    "canonical history bytes changed during double serialization"
                )
            _validate_session_or_fail(session, state)
            ranking = _rank_records(tuple(state.records))
            result = _UnregisteredCompletedTrainingHistory(
                synthetic_seed=_SYNTHETIC_SEED,
                epoch_count=state.layout.epoch_count,
                canonical_json_bytes=first,
                history_sha256=hashlib.sha256(_HISTORY_DOMAIN + first).hexdigest(),
                ranked_epochs=ranking,
            )
            _validate_unregistered_completed_history(result, state)
            barrier_state.phase = "CONSUMED"
            state.active_barrier = None
            state.active_barrier_token = None
            state.phase = "COMPLETE"
            _advance_session_lifecycle(session, state, event="COMPLETE")
            return result
        except BaseException:
            if barrier_state is not None:
                barrier_state.phase = "FAILED"
                try:
                    _terminal_fail_foreign_owner(barrier_state.owner, session)
                finally:
                    _terminal_fail_session(session, state)
            else:
                _terminal_fail_session(session, state)
            raise


def _synthetic_history_session_snapshot(
    session: TrainingHistorySession,
) -> _SyntheticHistorySessionSnapshot:
    state = _issued_session_state(session)
    trusted_lock = _trusted_session_lock(session, state)
    with _FLOW_LOCK, trusted_lock:
        _validate_session_or_fail(session, state)
        return _SyntheticHistorySessionSnapshot(
            phase=state.phase,
            epoch_count=state.layout.epoch_count,
            accepted_epoch_count=len(state.records),
            has_active_epoch=state.active_epoch is not None,
            has_active_barrier=state.active_barrier is not None,
        )


def _build_epoch_record(
    state: _SessionState,
    *,
    zero_based_epoch: int,
    training_population_digest: str,
    training_cross_entropy: np.float64,
    epoch_update_trace_digest: str,
    validation_confusion_matrix: _ConfusionMatrix,
    validation_cross_entropy: np.float64,
    validation_prediction_digest: str,
    model_tensor_digest: str,
) -> _EpochRecord:
    _require_digest(training_population_digest, "training_population_digest")
    _require_float64_cross_entropy(
        training_cross_entropy,
        "training_cross_entropy",
    )
    _require_digest(epoch_update_trace_digest, "epoch_update_trace_digest")
    confusion = _validated_confusion_matrix(validation_confusion_matrix)
    _require_float64_cross_entropy(
        validation_cross_entropy,
        "validation_cross_entropy",
    )
    _require_digest(validation_prediction_digest, "validation_prediction_digest")
    _require_digest(model_tensor_digest, "model_tensor_digest")
    first_update = zero_based_epoch * state.layout.updates_per_epoch
    record = _EpochRecord(
        zero_based_epoch=zero_based_epoch,
        first_global_update=first_update,
        last_global_update_inclusive=(
            first_update + state.layout.updates_per_epoch - 1
        ),
        training_population_digest=training_population_digest,
        training_cross_entropy=np.float64(training_cross_entropy),
        epoch_update_trace_digest=epoch_update_trace_digest,
        validation_input_digest=state.validation_input_digest,
        validation_confusion_matrix=confusion,
        validation_cross_entropy=np.float64(validation_cross_entropy),
        macro_f1=_macro_f1_from_confusion(confusion),
        validation_prediction_digest=validation_prediction_digest,
        model_tensor_digest=model_tensor_digest,
    )
    _validate_epoch_record(record, state)
    return record


def _validate_epoch_record(record: _EpochRecord, state: _SessionState) -> None:
    if type(record) is not _EpochRecord:
        raise TypeError("history record must be an exact _EpochRecord")
    _require_exact_int(record.zero_based_epoch, "zero_based_epoch")
    if not 0 <= record.zero_based_epoch < state.layout.epoch_count:
        raise Experiment002TrainingHistoryError("history epoch index is out of range")
    expected_first = record.zero_based_epoch * state.layout.updates_per_epoch
    if (
        type(record.first_global_update) is not int
        or record.first_global_update != expected_first
        or type(record.last_global_update_inclusive) is not int
        or record.last_global_update_inclusive
        != expected_first + state.layout.updates_per_epoch - 1
    ):
        raise Experiment002TrainingHistoryError(
            "history update interval differs from the synthetic layout"
        )
    _require_digest(record.training_population_digest, "training_population_digest")
    _require_float64_cross_entropy(
        record.training_cross_entropy,
        "training_cross_entropy",
    )
    _require_digest(record.epoch_update_trace_digest, "epoch_update_trace_digest")
    _require_digest(record.validation_input_digest, "validation_input_digest")
    if record.validation_input_digest != state.validation_input_digest:
        raise Experiment002TrainingHistoryError(
            "epoch validation input digest differs from the session"
        )
    confusion = _validated_confusion_matrix(record.validation_confusion_matrix)
    _require_float64_cross_entropy(
        record.validation_cross_entropy,
        "validation_cross_entropy",
    )
    if type(record.macro_f1) is not Fraction:
        raise TypeError("macro_f1 must be an exact Fraction")
    if record.macro_f1 != _macro_f1_from_confusion(confusion):
        raise Experiment002TrainingHistoryError(
            "macro_f1 differs from the synthetic confusion matrix"
        )
    _require_digest(
        record.validation_prediction_digest,
        "validation_prediction_digest",
    )
    _require_digest(record.model_tensor_digest, "model_tensor_digest")


def _canonical_history_bytes(state: _SessionState) -> bytes:
    document: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "experiment": _EXPERIMENT,
        "seed": _SYNTHETIC_SEED,
        "validation_input_digest": state.validation_input_digest,
        "complete_update_trace_digest": state.complete_update_trace_digest,
        "epochs": [_epoch_document(record) for record in state.records],
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


def _epoch_document(record: _EpochRecord) -> dict[str, object]:
    return {
        "zero_based_epoch": record.zero_based_epoch,
        "first_global_update": record.first_global_update,
        "last_global_update_inclusive": record.last_global_update_inclusive,
        "training_population_digest": record.training_population_digest,
        "training_cross_entropy_float64_hex": _float64_hex(
            record.training_cross_entropy
        ),
        "epoch_update_trace_digest": record.epoch_update_trace_digest,
        "validation_input_digest": record.validation_input_digest,
        "validation_confusion_matrix": [
            list(row) for row in record.validation_confusion_matrix
        ],
        "validation_cross_entropy_float64_hex": _float64_hex(
            record.validation_cross_entropy
        ),
        "macro_f1_exact_numerator": record.macro_f1.numerator,
        "macro_f1_exact_denominator": record.macro_f1.denominator,
        "validation_prediction_digest": record.validation_prediction_digest,
        "model_tensor_digest": record.model_tensor_digest,
    }


def _rank_records(
    records: tuple[_EpochRecord, ...],
) -> tuple[_UnregisteredEpochRank, ...]:
    if not records:
        raise Experiment002TrainingHistoryError("cannot rank an empty history")
    ordered = sorted(
        records,
        key=lambda record: (
            -record.macro_f1,
            record.validation_cross_entropy,
            record.zero_based_epoch,
        ),
    )
    return tuple(
        _UnregisteredEpochRank(
            zero_based_epoch=record.zero_based_epoch,
            macro_f1_exact_numerator=record.macro_f1.numerator,
            macro_f1_exact_denominator=record.macro_f1.denominator,
            validation_cross_entropy_float64_hex=_float64_hex(
                record.validation_cross_entropy
            ),
        )
        for record in ordered
    )


def _validate_unregistered_completed_history(
    result: _UnregisteredCompletedTrainingHistory,
    state: _SessionState,
) -> None:
    if (
        type(result) is not _UnregisteredCompletedTrainingHistory
        or result.synthetic_seed != _SYNTHETIC_SEED
        or result.epoch_count != state.layout.epoch_count
        or type(result.canonical_json_bytes) is not bytes
        or not result.canonical_json_bytes.endswith(b"\n")
        or type(result.history_sha256) is not str
        or result.history_sha256
        != hashlib.sha256(_HISTORY_DOMAIN + result.canonical_json_bytes).hexdigest()
        or type(result.ranked_epochs) is not tuple
        or len(result.ranked_epochs) != state.layout.epoch_count
        or len({rank.zero_based_epoch for rank in result.ranked_epochs})
        != state.layout.epoch_count
    ):
        raise Experiment002TrainingHistoryError(
            "unregistered completed history is internally inconsistent"
        )


def _validated_confusion_matrix(value: object) -> _ConfusionMatrix:
    if type(value) is not tuple or len(value) != _CLASS_COUNT:
        raise Experiment002TrainingHistoryError(
            "validation_confusion_matrix must contain exactly twelve rows"
        )
    rows: list[tuple[int, ...]] = []
    for row in value:
        if type(row) is not tuple or len(row) != _CLASS_COUNT:
            raise Experiment002TrainingHistoryError(
                "each confusion row must contain exactly twelve values"
            )
        checked: list[int] = []
        for cell in row:
            if type(cell) is not int:
                raise TypeError("confusion cells must be exact integers")
            if not 0 <= cell <= _MAX_SYNTHETIC_CONFUSION_CELL:
                raise Experiment002TrainingHistoryError(
                    "confusion cell exceeds the tiny synthetic limit"
                )
            checked.append(cell)
        rows.append(tuple(checked))
    return tuple(rows)


def _macro_f1_from_confusion(confusion: _ConfusionMatrix) -> Fraction:
    total = Fraction(0, 1)
    for index in range(_CLASS_COUNT):
        true_positive = confusion[index][index]
        false_positive = sum(
            confusion[row][index] for row in range(_CLASS_COUNT) if row != index
        )
        false_negative = sum(
            confusion[index][column]
            for column in range(_CLASS_COUNT)
            if column != index
        )
        denominator = 2 * true_positive + false_positive + false_negative
        if denominator:
            total += Fraction(2 * true_positive, denominator)
    return total / _CLASS_COUNT


def _require_json_without_floats(value: object) -> None:
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError("canonical history object keys must be strings")
            _require_json_without_floats(item)
        return
    if type(value) is list:
        for item in value:
            _require_json_without_floats(item)
        return
    if type(value) not in (str, int):
        raise Experiment002TrainingHistoryError(
            "canonical history may contain only JSON strings and integers"
        )


def _float64_hex(value: np.float64) -> str:
    _require_float64_cross_entropy(value, "float64 history value")
    result = float(value).hex()
    if result != result.lower():
        raise Experiment002TrainingHistoryError("float64 hex is not lowercase")
    return result


def _require_float64_cross_entropy(value: object, name: str) -> None:
    if type(value) is not np.float64:
        raise TypeError(f"{name} must be an exact NumPy float64 scalar")
    if not np.isfinite(value) or value < np.float64(0.0) or np.signbit(value):
        raise Experiment002TrainingHistoryError(
            f"{name} must be finite, nonnegative, and not negative zero"
        )


def _require_digest(value: object, name: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if len(value) != 64 or any(character not in _LOWER_HEX for character in value):
        raise Experiment002TrainingHistoryError(
            f"{name} must be one lowercase SHA-256 hex string"
        )


def _require_layout(layout: _SyntheticHistoryLayout) -> None:
    if type(layout) is not _SyntheticHistoryLayout:
        raise TypeError("layout must be an exact _SyntheticHistoryLayout")
    layout.__post_init__()


def _require_exact_int(value: object, name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")


def _require_positive_int(value: object, name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if value < 1:
        raise Experiment002TrainingHistoryError(f"{name} must be positive")


def _issued_session_state(session: TrainingHistorySession) -> _SessionState:
    if type(session) is not TrainingHistorySession:
        raise TypeError("session must be an exact TrainingHistorySession")
    with _REGISTRY_LOCK:
        state = _SESSIONS.get(session)
        anchored = session in _ISSUED_SESSIONS
        if state is None or not anchored:
            _FAILED_SESSIONS.add(session)
    if state is None or not anchored:
        raise Experiment002TrainingHistoryError(
            "training history session was not issued"
        )
    return state


def _issued_evaluated_epoch_state(epoch: EvaluatedEpoch) -> _EvaluatedEpochState:
    if type(epoch) is not EvaluatedEpoch:
        raise TypeError("epoch must be an exact EvaluatedEpoch")
    with _REGISTRY_LOCK:
        state = _EVALUATED_EPOCHS.get(epoch)
    if state is None:
        raise Experiment002TrainingHistoryError("evaluated epoch was not issued")
    return state


def _issued_accepted_epoch_state(
    barrier: HistoryAcceptedEpoch,
) -> _AcceptedEpochState:
    if type(barrier) is not HistoryAcceptedEpoch:
        raise TypeError("barrier must be an exact HistoryAcceptedEpoch")
    with _REGISTRY_LOCK:
        state = _ACCEPTED_EPOCHS.get(barrier)
    if state is None:
        raise Experiment002TrainingHistoryError(
            "history acceptance barrier was not issued"
        )
    return state


def _require_open_session(state: _SessionState) -> None:
    if state.phase == "FAILED":
        raise Experiment002TrainingHistoryError("training history session has failed")
    if state.phase == "COMPLETE":
        raise Experiment002TrainingHistoryError("training history session is complete")
    if state.phase != "OPEN":
        raise Experiment002TrainingHistoryError("training history phase is invalid")


def _trusted_session_lock(
    session: TrainingHistorySession,
    state: _SessionState,
) -> threading.RLock:
    with _REGISTRY_LOCK:
        guard = _SESSION_GUARDS.get(session)
        anchor = _SESSION_ANCHORS.get(session)
        anchored = session in _ISSUED_SESSIONS
    valid = (
        type(guard) is _SessionGuard
        and type(anchor) is tuple
        and len(anchor) == 8
        and anchored
        and guard.token is anchor[0]
        and guard.route_marker is anchor[1]
        and guard.layout is anchor[2]
        and guard.epoch_count == anchor[3]
        and guard.updates_per_epoch == anchor[4]
        and guard.validation_input_digest == anchor[5]
        and guard.complete_update_trace_digest == anchor[6]
        and guard.lock is anchor[7]
        and state.token is anchor[0]
        and state.route_marker is anchor[1]
        and state.layout is anchor[2]
        and state.layout.epoch_count == anchor[3]
        and state.layout.updates_per_epoch == anchor[4]
        and state.validation_input_digest == anchor[5]
        and state.complete_update_trace_digest == anchor[6]
        and state.lock is anchor[7]
    )
    if not valid:
        _terminal_fail_session(session, state)
        raise Experiment002TrainingHistoryError(
            "training history lock authority changed"
        )
    if type(guard) is not _SessionGuard:
        raise Experiment002TrainingHistoryError(
            "training history lock guard has an invalid type"
        )
    return guard.lock


def _validate_session_or_fail(
    session: TrainingHistorySession,
    state: _SessionState,
) -> None:
    try:
        _require_session_state(session, state)
    except BaseException:
        _terminal_fail_session(session, state)
        raise


def _require_session_state(
    session: TrainingHistorySession,
    state: _SessionState,
) -> None:
    with _REGISTRY_LOCK:
        guard = _SESSION_GUARDS.get(session)
        anchor = _SESSION_ANCHORS.get(session)
        lifecycle = _SESSION_LIFECYCLES.get(session)
        lifecycle_anchor = _SESSION_LIFECYCLE_ANCHORS.get(session)
        failed = session in _FAILED_SESSIONS
        if (
            guard is None
            or anchor is None
            or lifecycle is None
            or lifecycle_anchor is None
        ):
            _FAILED_SESSIONS.add(session)
    if guard is None or anchor is None or lifecycle is None or lifecycle_anchor is None:
        raise Experiment002TrainingHistoryError(
            "training history session guard was not issued"
        )
    if (
        type(state) is not _SessionState
        or state.token is not guard.token
        or lifecycle.token is not guard.token
        or not _lifecycle_matches_anchor(lifecycle, lifecycle_anchor)
        or guard.token is not anchor[0]
        or guard.route_marker is not anchor[1]
        or guard.layout is not anchor[2]
        or guard.epoch_count != anchor[3]
        or guard.updates_per_epoch != anchor[4]
        or guard.validation_input_digest != anchor[5]
        or guard.complete_update_trace_digest != anchor[6]
        or guard.lock is not anchor[7]
        or state.route_marker is not _SYNTHETIC_ROUTE_MARKER
        or guard.route_marker is not _SYNTHETIC_ROUTE_MARKER
        or state.layout is not guard.layout
        or state.layout.epoch_count != guard.epoch_count
        or state.layout.updates_per_epoch != guard.updates_per_epoch
        or state.validation_input_digest != guard.validation_input_digest
        or state.complete_update_trace_digest != guard.complete_update_trace_digest
        or state.lock is not guard.lock
        or type(state.records) is not list
        or len(state.records) > state.layout.epoch_count
        or (state.active_epoch is None) != (state.active_epoch_token is None)
        or (state.active_barrier is None) != (state.active_barrier_token is None)
        or (state.active_epoch is not None and state.active_barrier is not None)
        or state.phase not in ("OPEN", "COMPLETE", "FAILED")
    ):
        raise Experiment002TrainingHistoryError(
            "training history session authority is invalid"
        )
    _require_layout(state.layout)
    _require_digest(state.validation_input_digest, "validation_input_digest")
    _require_digest(
        state.complete_update_trace_digest,
        "complete_update_trace_digest",
    )
    if state.phase != lifecycle.phase or failed != (state.phase == "FAILED"):
        raise Experiment002TrainingHistoryError(
            "training history lifecycle is not monotonic"
        )
    if state.phase == "FAILED":
        return

    record_fingerprints: list[str] = []
    for expected_epoch, record in enumerate(state.records):
        if record.zero_based_epoch != expected_epoch:
            raise Experiment002TrainingHistoryError(
                "history records are not in exact epoch order"
            )
        _validate_epoch_record(record, state)
        record_fingerprints.append(_record_authority_fingerprint(record))
    if tuple(record_fingerprints) != lifecycle.record_fingerprints:
        raise Experiment002TrainingHistoryError(
            "accepted history record authority changed"
        )
    if (
        state.active_epoch is not lifecycle.active_epoch
        or state.active_epoch_token is not lifecycle.active_epoch_token
        or state.active_barrier is not lifecycle.active_barrier
        or state.active_barrier_token is not lifecycle.active_barrier_token
    ):
        raise Experiment002TrainingHistoryError(
            "active history capability lifecycle changed"
        )

    if state.active_epoch is None:
        if lifecycle.active_epoch_fingerprint is not None:
            raise Experiment002TrainingHistoryError(
                "active evaluated epoch fingerprint is inconsistent"
            )
    else:
        epoch_state = _issued_evaluated_epoch_state(state.active_epoch)
        if (
            epoch_state.owner is not session
            or epoch_state.session_token is not state.token
            or epoch_state.epoch_token is not state.active_epoch_token
            or epoch_state.route_marker is not _SYNTHETIC_ROUTE_MARKER
            or epoch_state.phase != "ISSUED"
        ):
            raise Experiment002TrainingHistoryError(
                "active evaluated epoch authority changed"
            )
        _validate_epoch_record(epoch_state.record, state)
        if _record_authority_fingerprint(epoch_state.record) != (
            lifecycle.active_epoch_fingerprint
        ):
            raise Experiment002TrainingHistoryError(
                "active evaluated epoch record changed"
            )

    if state.active_barrier is None:
        if lifecycle.active_barrier_epoch is not None:
            raise Experiment002TrainingHistoryError(
                "active history barrier epoch is inconsistent"
            )
    else:
        barrier_state = _issued_accepted_epoch_state(state.active_barrier)
        if (
            barrier_state.owner is not session
            or barrier_state.session_token is not state.token
            or barrier_state.barrier_token is not state.active_barrier_token
            or barrier_state.route_marker is not _SYNTHETIC_ROUTE_MARKER
            or barrier_state.phase != "ISSUED"
            or barrier_state.accepted_epoch != len(state.records) - 1
            or barrier_state.accepted_epoch != lifecycle.active_barrier_epoch
        ):
            raise Experiment002TrainingHistoryError(
                "active history barrier authority changed"
            )
    if state.phase == "COMPLETE" and (
        len(state.records) != state.layout.epoch_count
        or state.active_epoch is not None
        or state.active_barrier is not None
    ):
        raise Experiment002TrainingHistoryError(
            "completed history lifecycle is inconsistent"
        )


def _lifecycle_anchor(lifecycle: _SessionLifecycle) -> _LifecycleAnchor:
    return (
        lifecycle.token,
        lifecycle.phase,
        lifecycle.record_fingerprints,
        lifecycle.active_epoch,
        lifecycle.active_epoch_token,
        lifecycle.active_epoch_fingerprint,
        lifecycle.active_barrier,
        lifecycle.active_barrier_token,
        lifecycle.active_barrier_epoch,
    )


def _lifecycle_matches_anchor(
    lifecycle: _SessionLifecycle,
    anchor: _LifecycleAnchor,
) -> bool:
    return (
        lifecycle.token is anchor[0]
        and lifecycle.phase == anchor[1]
        and lifecycle.record_fingerprints == anchor[2]
        and lifecycle.active_epoch is anchor[3]
        and lifecycle.active_epoch_token is anchor[4]
        and lifecycle.active_epoch_fingerprint == anchor[5]
        and lifecycle.active_barrier is anchor[6]
        and lifecycle.active_barrier_token is anchor[7]
        and lifecycle.active_barrier_epoch == anchor[8]
    )


def _advance_session_lifecycle(
    session: TrainingHistorySession,
    state: _SessionState,
    *,
    event: Literal["ISSUE", "CONSUME", "COMPLETE"],
) -> None:
    if event not in ("ISSUE", "CONSUME", "COMPLETE"):
        raise Experiment002TrainingHistoryError("history lifecycle event is invalid")
    with _REGISTRY_LOCK:
        previous = _SESSION_LIFECYCLES.get(session)
        previous_anchor = _SESSION_LIFECYCLE_ANCHORS.get(session)
    if (
        previous is None
        or previous_anchor is None
        or not _lifecycle_matches_anchor(previous, previous_anchor)
        or previous.phase != "OPEN"
        or previous.token is not state.token
    ):
        raise Experiment002TrainingHistoryError(
            "history lifecycle transition has no valid predecessor"
        )
    current_fingerprints = tuple(
        _record_authority_fingerprint(record) for record in state.records
    )

    active_epoch_fingerprint: str | None = None
    if state.active_epoch is not None:
        active_epoch_state = _issued_evaluated_epoch_state(state.active_epoch)
        active_epoch_fingerprint = _record_authority_fingerprint(
            active_epoch_state.record
        )
    else:
        active_epoch_state = None
    active_barrier_epoch: int | None = None
    if state.active_barrier is not None:
        active_barrier_state = _issued_accepted_epoch_state(state.active_barrier)
        active_barrier_epoch = active_barrier_state.accepted_epoch
    else:
        active_barrier_state = None

    if event == "ISSUE":
        valid = (
            state.phase == "OPEN"
            and current_fingerprints == previous.record_fingerprints
            and previous.active_epoch is None
            and state.active_epoch is not None
            and active_epoch_state is not None
            and active_epoch_state.owner is session
            and active_epoch_state.session_token is state.token
            and active_epoch_state.epoch_token is state.active_epoch_token
            and active_epoch_state.route_marker is _SYNTHETIC_ROUTE_MARKER
            and active_epoch_state.phase == "ISSUED"
            and active_epoch_state.record.zero_based_epoch == len(state.records)
            and state.active_barrier is None
        )
        if len(state.records) == 0:
            valid = valid and previous.active_barrier is None
        else:
            previous_barrier = previous.active_barrier
            if previous_barrier is None:
                valid = False
            else:
                previous_barrier_state = _issued_accepted_epoch_state(previous_barrier)
                valid = valid and (
                    previous_barrier_state.owner is session
                    and previous_barrier_state.session_token is state.token
                    and previous_barrier_state.barrier_token
                    is previous.active_barrier_token
                    and previous_barrier_state.accepted_epoch == len(state.records) - 1
                    and previous_barrier_state.phase == "CONSUMED"
                )
    elif event == "CONSUME":
        previous_epoch = previous.active_epoch
        if previous_epoch is None:
            valid = False
        else:
            previous_epoch_state = _issued_evaluated_epoch_state(previous_epoch)
            valid = (
                state.phase == "OPEN"
                and previous.active_epoch_fingerprint is not None
                and current_fingerprints
                == previous.record_fingerprints + (previous.active_epoch_fingerprint,)
                and previous.active_barrier is None
                and previous_epoch_state.owner is session
                and previous_epoch_state.session_token is state.token
                and previous_epoch_state.epoch_token is previous.active_epoch_token
                and previous_epoch_state.phase == "CONSUMED"
                and state.active_epoch is None
                and state.active_barrier is not None
                and active_barrier_state is not None
                and active_barrier_state.owner is session
                and active_barrier_state.session_token is state.token
                and active_barrier_state.barrier_token is state.active_barrier_token
                and active_barrier_state.route_marker is _SYNTHETIC_ROUTE_MARKER
                and active_barrier_state.phase == "ISSUED"
                and active_barrier_state.accepted_epoch == len(state.records) - 1
            )
    else:
        previous_barrier = previous.active_barrier
        if previous_barrier is None:
            valid = False
        else:
            previous_barrier_state = _issued_accepted_epoch_state(previous_barrier)
            valid = (
                state.phase == "COMPLETE"
                and current_fingerprints == previous.record_fingerprints
                and len(state.records) == state.layout.epoch_count
                and previous.active_epoch is None
                and previous_barrier_state.owner is session
                and previous_barrier_state.session_token is state.token
                and previous_barrier_state.barrier_token
                is previous.active_barrier_token
                and previous_barrier_state.phase == "CONSUMED"
                and state.active_epoch is None
                and state.active_barrier is None
            )
    if not valid:
        raise Experiment002TrainingHistoryError(
            f"history lifecycle {event.lower()} transition is invalid"
        )

    lifecycle = _SessionLifecycle(
        token=state.token,
        phase=state.phase,
        record_fingerprints=current_fingerprints,
        active_epoch=state.active_epoch,
        active_epoch_token=state.active_epoch_token,
        active_epoch_fingerprint=active_epoch_fingerprint,
        active_barrier=state.active_barrier,
        active_barrier_token=state.active_barrier_token,
        active_barrier_epoch=active_barrier_epoch,
    )
    with _REGISTRY_LOCK:
        _SESSION_LIFECYCLES[session] = lifecycle
        _SESSION_LIFECYCLE_ANCHORS[session] = _lifecycle_anchor(lifecycle)


def _record_authority_fingerprint(record: _EpochRecord) -> str:
    document = _epoch_document(record)
    _require_json_without_floats(document)
    payload = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(_RECORD_AUTHORITY_DOMAIN + payload).hexdigest()


def _terminal_fail_session(
    session: TrainingHistorySession,
    state: _SessionState,
) -> None:
    state.phase = "FAILED"
    with _REGISTRY_LOCK:
        lifecycle = _SESSION_LIFECYCLES.get(session)
        lifecycle_anchor = _SESSION_LIFECYCLE_ANCHORS.get(session)
        _FAILED_SESSIONS.add(session)
        if lifecycle_anchor is not None:
            trusted_lifecycle = _SessionLifecycle(
                token=lifecycle_anchor[0],
                phase="FAILED",
                record_fingerprints=lifecycle_anchor[2],
                active_epoch=lifecycle_anchor[3],
                active_epoch_token=lifecycle_anchor[4],
                active_epoch_fingerprint=lifecycle_anchor[5],
                active_barrier=lifecycle_anchor[6],
                active_barrier_token=lifecycle_anchor[7],
                active_barrier_epoch=lifecycle_anchor[8],
            )
            _SESSION_LIFECYCLES[session] = trusted_lifecycle
            _SESSION_LIFECYCLE_ANCHORS[session] = _lifecycle_anchor(trusted_lifecycle)
        elif lifecycle is not None:
            trusted_lifecycle = replace(lifecycle, phase="FAILED")
            _SESSION_LIFECYCLES[session] = trusted_lifecycle
            _SESSION_LIFECYCLE_ANCHORS[session] = _lifecycle_anchor(trusted_lifecycle)
        else:
            trusted_lifecycle = None
    epoch_candidates = [state.active_epoch]
    barrier_candidates = [state.active_barrier]
    if trusted_lifecycle is not None:
        epoch_candidates.append(trusted_lifecycle.active_epoch)
        barrier_candidates.append(trusted_lifecycle.active_barrier)
    for candidate in epoch_candidates:
        if candidate is None:
            continue
        with _REGISTRY_LOCK:
            active_epoch = _EVALUATED_EPOCHS.get(candidate)
        if active_epoch is not None:
            active_epoch.phase = "FAILED"
    for barrier_candidate in barrier_candidates:
        if barrier_candidate is None:
            continue
        with _REGISTRY_LOCK:
            active_barrier = _ACCEPTED_EPOCHS.get(barrier_candidate)
        if active_barrier is not None:
            active_barrier.phase = "FAILED"


def _terminal_fail_foreign_owner(
    owner: TrainingHistorySession,
    current: TrainingHistorySession,
) -> None:
    if owner is current:
        return
    owner_state: _SessionState | None = None
    try:
        owner_state = _issued_session_state(owner)
        owner_lock = _trusted_session_lock(owner, owner_state)
        with owner_lock:
            if owner_state.phase != "FAILED":
                _terminal_fail_session(owner, owner_state)
    except BaseException:
        if owner_state is not None:
            _terminal_fail_session(owner, owner_state)
        return
