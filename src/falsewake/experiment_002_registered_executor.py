"""Source-bound registered training executor for Experiment 002.

This module owns the first production training slice: one child-local executor
may train one epoch at a time and issue one opaque handoff for the registered
evaluator.
The audited population, transition bridge, and update-trace modules remain the
authorities for data order and optimizer-update evidence.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import struct
import threading
import warnings
import weakref
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final, Literal, NoReturn, cast

import numpy as np
import torch
from torch import Tensor

import falsewake.experiment_002_training_executor as _numeric
from falsewake.causal_kws import (
    BLOCK_DILATIONS,
    DEPTHWISE_KERNEL_SIZE,
    ENCODER_CHANNELS,
    INPUT_MEL_BINS,
    MODEL_RECEPTIVE_FIELD_FRAMES,
    OUTPUT_CLASSES,
    POOL_HISTORY_FRAMES,
    TRAINABLE_PARAMETER_COUNT,
    CausalKWS,
    CausalKWSState,
)
from falsewake.experiment_002_data import TRAINING_SEEDS
from falsewake.experiment_002_evidence import (
    RegisteredValidationInputs,
    verify_registered_validation_inputs,
)
from falsewake.experiment_002_run_authority import (
    VerifiedRunRegistration,
    reverify_verified_run_registration,
)
from falsewake.experiment_002_training_bridge import (
    RegisteredExecutorSessionAuthority,
    RegisteredOptimizerTransition,
    TraceConsumedTransition,
    _executor_session_snapshot,
    _fail_optimizer_transition,
    _fail_registered_executor_session,
    _issue_registered_executor_session,
    _issue_registered_optimizer_transition,
    _transition_snapshot,
)
from falsewake.experiment_002_training_evidence import (
    CompleteUpdateTraceEvidence,
    EpochUpdateTraceEvidence,
    UpdateTraceAccumulator,
    _fail_registered_trace,
    registered_learning_rate,
    verify_registered_complete_update_trace,
    verify_registered_epoch_update_trace,
)
from falsewake.experiment_002_training_population import (
    REGISTERED_BATCH_COUNT,
    REGISTERED_LAST_BATCH_SIZE,
    TRAINING_BATCH_SIZE,
    CompletedTrainingPopulation,
    CompletedTrainingUpdate,
    MaterializedTrainingBatch,
    RegisteredTrainingEpoch,
    RegisteredTrainingInputSource,
    _begin_registered_epoch_for_executor,
    _executor_accept_completed_update,
    _executor_batch_binding,
    _executor_complete_update,
    _executor_fail_epoch,
    _executor_finish_epoch,
    _executor_next_batch,
    _executor_verify_pending_batch,
    _issued_source_state,
    _RegisteredBatchBinding,
    verify_completed_registered_training_population,
)

if TYPE_CHECKING:
    from falsewake.experiment_002_registered_evaluator import RegisteredEvaluatedEpoch
    from falsewake.experiment_002_registered_history import (
        RegisteredCompletedTrainingHistory,
        RegisteredHistoryBarrier,
        RegisteredTrainingHistory,
        _RegisteredHistoryBarrierSnapshot,
    )

type RegisteredExecutorPhase = Literal[
    "READY_TO_TRAIN", "TRAINING", "AWAITING_EVALUATION", "COMPLETE", "FAILED"
]

_REGISTERED_ROUTE: Final = object()
_UPDATES_PER_EPOCH: Final = REGISTERED_BATCH_COUNT
_REGISTERED_EPOCH_COUNT: Final = 30
_MAX_GRADIENT_NORM: Final = 5.0
_UINT32_MAX: Final = (1 << 32) - 1
_RLOCK_TYPE: Final = type(threading.RLock())
_EVENT_TYPE: Final = type(threading.Event())


class Experiment002RegisteredExecutorError(ValueError):
    """The source-bound registered executor violated its authority contract."""


class _BoundaryAdmissionConflict(Experiment002RegisteredExecutorError):
    """A different admitted boundary owns the executor without poisoning it."""


class _BoundaryTruthCorruption(Experiment002RegisteredExecutorError):
    """Retained boundary truth changed after its terminal publication."""


class _FinalBarrierRequiresCompletion(Experiment002RegisteredExecutorError):
    """A valid epoch-29 barrier was deliberately routed away from advance."""


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
class RegisteredTrainingExecutor:
    """Opaque, child-local authority for one registered training runtime."""

    def __init__(self) -> None:
        raise TypeError("registered training executors are issuer-only")

    def __copy__(self) -> NoReturn:
        raise TypeError("registered training executors cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        del memo
        raise TypeError("registered training executors cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("registered training executors cannot be serialized")


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
class RegisteredExecutorEpochHandoff:
    """Opaque one-use binding from a trained epoch to registered evaluation."""

    def __init__(self) -> None:
        raise TypeError("registered epoch handoffs are issuer-only")

    def __copy__(self) -> NoReturn:
        raise TypeError("registered epoch handoffs cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        del memo
        raise TypeError("registered epoch handoffs cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("registered epoch handoffs cannot be serialized")


@dataclass(frozen=True, slots=True)
class _RegisteredExecutorAnchor:
    executor: RegisteredTrainingExecutor
    ticket: _ExecutorIssuanceTicket
    token: object
    route_marker: object
    registration: VerifiedRunRegistration
    process_id: int
    seed: int
    training_inputs: RegisteredTrainingInputSource
    validation_inputs: RegisteredValidationInputs
    bridge_authority: RegisteredExecutorSessionAuthority
    bridge_session_token: object
    trace: UpdateTraceAccumulator
    runtime: _numeric._ExecutorState
    model: CausalKWS
    optimizer: torch.optim.AdamW
    parameter_names: tuple[str, ...]
    parameters: tuple[torch.nn.Parameter, ...]
    lock: threading.RLock


@dataclass(frozen=True, slots=True)
class _RegisteredExecutorGuard:
    anchor: _RegisteredExecutorAnchor
    ticket: _ExecutorIssuanceTicket
    token: object
    route_marker: object
    registration: VerifiedRunRegistration
    process_id: int
    seed: int
    training_inputs: RegisteredTrainingInputSource
    validation_inputs: RegisteredValidationInputs
    bridge_authority: RegisteredExecutorSessionAuthority
    bridge_session_token: object
    trace: UpdateTraceAccumulator
    runtime: _numeric._ExecutorState
    model: CausalKWS
    optimizer: torch.optim.AdamW
    parameter_names: tuple[str, ...]
    parameters: tuple[torch.nn.Parameter, ...]
    lock: threading.RLock


@dataclass(frozen=True, slots=True)
class _RegisteredExecutorLifecycle:
    phase: RegisteredExecutorPhase
    zero_based_epoch: int
    optimizer_generation: int
    expected: _numeric._RuntimeDigests
    active_epoch: RegisteredTrainingEpoch | None
    active_transition: RegisteredOptimizerTransition | None
    active_handoff: RegisteredExecutorEpochHandoff | None
    previous_history_barrier: object | None
    continuation_reservation: _ContinuationReservation | None
    finalization_reservation: _FinalizationReservation | None
    final_barrier: RegisteredHistoryBarrier | None
    complete_update_trace: CompleteUpdateTraceEvidence | None
    completed_history: RegisteredCompletedTrainingHistory | None


@dataclass(frozen=True, slots=True, eq=False)
class _RegisteredExecutorLifecycleAuthority:
    executor: RegisteredTrainingExecutor
    sequence: int
    lifecycle: _RegisteredExecutorLifecycle
    previous: _RegisteredExecutorLifecycleAuthority | None


@dataclass(slots=True)
class _RegisteredExecutorState:
    anchor: _RegisteredExecutorAnchor
    ticket: _ExecutorIssuanceTicket
    token: object
    route_marker: object
    registration: VerifiedRunRegistration
    process_id: int
    seed: int
    training_inputs: RegisteredTrainingInputSource
    validation_inputs: RegisteredValidationInputs
    bridge_authority: RegisteredExecutorSessionAuthority
    bridge_session_token: object
    trace: UpdateTraceAccumulator
    runtime: _numeric._ExecutorState
    expected: _numeric._RuntimeDigests
    lock: threading.RLock
    phase: RegisteredExecutorPhase = "READY_TO_TRAIN"
    zero_based_epoch: int = 0
    optimizer_generation: int = 0
    active_epoch: RegisteredTrainingEpoch | None = None
    active_transition: RegisteredOptimizerTransition | None = None
    active_handoff: RegisteredExecutorEpochHandoff | None = None
    previous_history_barrier: object | None = None
    continuation_reservation: _ContinuationReservation | None = None
    finalization_reservation: _FinalizationReservation | None = None
    final_barrier: RegisteredHistoryBarrier | None = None
    complete_update_trace: CompleteUpdateTraceEvidence | None = None
    completed_history: RegisteredCompletedTrainingHistory | None = None


@dataclass(frozen=True, slots=True)
class _RegisteredExecutorSnapshot:
    registration: VerifiedRunRegistration
    process_id: int
    seed: int
    phase: RegisteredExecutorPhase
    zero_based_epoch: int
    optimizer_generation: int
    model_sha256: str
    optimizer_sha256: str
    rng_sha256: str
    previous_history_barrier: object | None
    final_barrier: RegisteredHistoryBarrier | None
    complete_update_trace: CompleteUpdateTraceEvidence | None
    completed_history: RegisteredCompletedTrainingHistory | None


@dataclass(frozen=True, slots=True, eq=False)
class _ExecutorIssuanceTicket:
    executor: RegisteredTrainingExecutor
    token: object
    registration: VerifiedRunRegistration
    process_id: int


@dataclass(frozen=True, slots=True, eq=False)
class _ExecutorFailureEvent:
    ticket: _ExecutorIssuanceTicket


@dataclass(frozen=True, slots=True)
class _HandoffAnchor:
    handoff: RegisteredExecutorEpochHandoff
    executor: RegisteredTrainingExecutor
    executor_token: object
    handoff_token: object
    route_marker: object
    ticket: _HandoffIssuanceTicket


@dataclass(frozen=True, slots=True, eq=False)
class _HandoffIssuanceTicket:
    handoff: RegisteredExecutorEpochHandoff
    executor_ticket: _ExecutorIssuanceTicket
    handoff_token: object


@dataclass(frozen=True, slots=True, eq=False)
class _HandoffLifecycleEvent:
    ticket: _HandoffIssuanceTicket
    kind: Literal["USED", "FAILED"]


@dataclass(frozen=True, slots=True)
class _HandoffGuard:
    anchor: _HandoffAnchor
    executor: RegisteredTrainingExecutor
    executor_token: object
    handoff_token: object
    route_marker: object
    ticket: _HandoffIssuanceTicket
    registration: VerifiedRunRegistration
    validation_inputs: RegisteredValidationInputs
    process_id: int
    bridge_authority: RegisteredExecutorSessionAuthority
    bridge_session_token: object
    model: CausalKWS
    population: CompletedTrainingPopulation
    epoch_trace: EpochUpdateTraceEvidence
    seed: int
    zero_based_epoch: int
    optimizer_generation: int
    runtime_digests: _numeric._RuntimeDigests
    previous_history_barrier: object | None
    one_shot_token: object


@dataclass(slots=True)
class _HandoffState:
    anchor: _HandoffAnchor
    executor: RegisteredTrainingExecutor
    executor_token: object
    handoff_token: object
    route_marker: object
    ticket: _HandoffIssuanceTicket
    registration: VerifiedRunRegistration
    validation_inputs: RegisteredValidationInputs
    process_id: int
    bridge_authority: RegisteredExecutorSessionAuthority
    bridge_session_token: object
    model: CausalKWS
    population: CompletedTrainingPopulation
    epoch_trace: EpochUpdateTraceEvidence
    seed: int
    zero_based_epoch: int
    optimizer_generation: int
    runtime_digests: _numeric._RuntimeDigests
    previous_history_barrier: object | None
    one_shot_token: object
    used: bool = False
    failed: bool = False


@dataclass(frozen=True, slots=True)
class _HandoffLifecycle:
    used: bool
    failed: bool


@dataclass(frozen=True, slots=True, eq=False)
class _HandoffLifecycleAuthority:
    handoff: RegisteredExecutorEpochHandoff
    sequence: int
    lifecycle: _HandoffLifecycle
    previous: _HandoffLifecycleAuthority | None


type _ContinuationPhase = Literal["RESERVED", "CONSUMED", "COMMITTED", "FAILED"]


@dataclass(frozen=True, slots=True, eq=False)
class _ContinuationReservation:
    """Exact pre-consumption reservation for one non-final epoch advance."""

    token: object
    executor: RegisteredTrainingExecutor
    executor_ticket: _ExecutorIssuanceTicket
    handoff: RegisteredExecutorEpochHandoff
    handoff_ticket: _HandoffIssuanceTicket
    barrier: RegisteredHistoryBarrier
    barrier_type: type[RegisteredHistoryBarrier]
    evaluated_epoch: RegisteredEvaluatedEpoch
    evaluated_epoch_type: type[RegisteredEvaluatedEpoch]
    evaluated_authority_sha256: str
    registration: VerifiedRunRegistration
    validation_inputs: RegisteredValidationInputs
    process_id: int
    seed: int
    zero_based_epoch: int
    optimizer_generation: int
    runtime_digests: _numeric._RuntimeDigests
    previous_history_barrier: object | None
    handoff_token: object
    one_shot_token: object


@dataclass(frozen=True, slots=True)
class _RetiredHandoffBinding:
    """Immutable authority retained after the live executor leaves a handoff."""

    reservation: _ContinuationReservation
    executor: RegisteredTrainingExecutor
    handoff: RegisteredExecutorEpochHandoff
    barrier: RegisteredHistoryBarrier
    evaluated_epoch: RegisteredEvaluatedEpoch
    evaluated_authority_sha256: str
    registration: VerifiedRunRegistration
    validation_inputs: RegisteredValidationInputs
    process_id: int
    seed: int
    zero_based_epoch: int
    optimizer_generation: int
    runtime_digests: _numeric._RuntimeDigests
    previous_history_barrier: object | None
    handoff_token: object
    one_shot_token: object


type _FinalizationPhase = Literal[
    "RESERVED", "TRACE_COMPLETE", "HISTORY_COMPLETE", "COMMITTED", "FAILED"
]


@dataclass(frozen=True, slots=True, eq=False)
class _FinalizationReservation:
    """Exact epoch-29 boundary reserved before any irreversible completion."""

    token: object
    executor: RegisteredTrainingExecutor
    executor_ticket: _ExecutorIssuanceTicket
    handoff: RegisteredExecutorEpochHandoff
    handoff_ticket: _HandoffIssuanceTicket
    final_barrier: RegisteredHistoryBarrier
    barrier_type: type[RegisteredHistoryBarrier]
    history: RegisteredTrainingHistory
    history_type: type[RegisteredTrainingHistory]
    evaluated_epoch: RegisteredEvaluatedEpoch
    evaluated_epoch_type: type[RegisteredEvaluatedEpoch]
    evaluated_authority_sha256: str
    registration: VerifiedRunRegistration
    validation_inputs: RegisteredValidationInputs
    process_id: int
    seed: int
    zero_based_epoch: int
    optimizer_generation: int
    runtime_digests: _numeric._RuntimeDigests
    previous_history_barrier: object
    handoff_token: object
    one_shot_token: object
    trace: UpdateTraceAccumulator
    epoch_traces: tuple[EpochUpdateTraceEvidence, ...]


@dataclass(frozen=True, slots=True)
class _FinalizedHandoffBinding:
    """Immutable epoch-29 authority retained after full history completion."""

    reservation: _FinalizationReservation
    executor: RegisteredTrainingExecutor
    handoff: RegisteredExecutorEpochHandoff
    final_barrier: RegisteredHistoryBarrier
    history: RegisteredTrainingHistory
    evaluated_epoch: RegisteredEvaluatedEpoch
    evaluated_authority_sha256: str
    complete_update_trace: CompleteUpdateTraceEvidence
    complete_update_trace_type: type[CompleteUpdateTraceEvidence]
    completed_history: RegisteredCompletedTrainingHistory
    completed_history_type: type[RegisteredCompletedTrainingHistory]
    registration: VerifiedRunRegistration
    validation_inputs: RegisteredValidationInputs
    process_id: int
    seed: int
    zero_based_epoch: int
    optimizer_generation: int
    runtime_digests: _numeric._RuntimeDigests
    previous_history_barrier: object
    handoff_token: object
    one_shot_token: object


@dataclass(frozen=True, slots=True)
class _AdvanceAdmission:
    """Serialize public callers before any caller can consume history."""

    token: object
    executor: RegisteredTrainingExecutor
    barrier: RegisteredHistoryBarrier
    leader: bool
    completed: threading.Event
    outcome: list[BaseException | None]


@dataclass(frozen=True, slots=True)
class _CompletionOutcome:
    result: RegisteredCompletedTrainingHistory | None
    error: BaseException | None


@dataclass(frozen=True, slots=True)
class _CompletionAdmission:
    """Serialize one exact epoch-29 completion with every boundary operation."""

    token: object
    executor: RegisteredTrainingExecutor
    barrier: RegisteredHistoryBarrier
    leader: bool
    completed: threading.Event
    outcome: list[_CompletionOutcome]


@dataclass(frozen=True, slots=True)
class _BoundaryAdmissionState:
    kind: Literal["ADVANCE", "COMPLETE"]
    barrier: RegisteredHistoryBarrier
    token: object
    completed: threading.Event
    outcome: list[BaseException | None] | list[_CompletionOutcome]


@dataclass(frozen=True, slots=True)
class _RegisteredExecutorEpochHandoffSnapshot:
    executor: RegisteredTrainingExecutor
    registration: VerifiedRunRegistration
    validation_inputs: RegisteredValidationInputs
    process_id: int
    bridge_authority: RegisteredExecutorSessionAuthority
    bridge_session_token: object
    population: CompletedTrainingPopulation
    epoch_trace: EpochUpdateTraceEvidence
    seed: int
    zero_based_epoch: int
    optimizer_generation: int
    runtime_digests: _numeric._RuntimeDigests
    previous_history_barrier: object | None
    one_shot_token: object
    another_training_epoch: bool


@dataclass(frozen=True, slots=True)
class _BatchViews:
    model_inputs: np.ndarray[tuple[int, int, int], np.dtype[np.float32]]
    label_indices: np.ndarray[tuple[int], np.dtype[np.int64]]
    features: Tensor
    labels: Tensor
    batch_size: int


@dataclass(frozen=True, slots=True)
class _NumericUpdateOutcome:
    learning_rate: float
    batch_mean_training_loss: np.float32
    returned_preclip_l2_norm: np.float32
    rng_state_after_forward: bytes


_CREATION_LOCK = threading.Lock()
_CREATION_RESERVED_PROCESSES: set[int] = set()
_CREATION_RESERVATION_HISTORY: list[tuple[int, object]] = []
_EXECUTORS: weakref.WeakKeyDictionary[
    RegisteredTrainingExecutor, _RegisteredExecutorState
] = weakref.WeakKeyDictionary()
_EXECUTOR_GUARDS: weakref.WeakKeyDictionary[
    RegisteredTrainingExecutor, _RegisteredExecutorGuard
] = weakref.WeakKeyDictionary()
_EXECUTOR_LIFECYCLES: weakref.WeakKeyDictionary[
    RegisteredTrainingExecutor, _RegisteredExecutorLifecycle
] = weakref.WeakKeyDictionary()
_EXECUTOR_LIFECYCLE_HEADS: dict[
    RegisteredTrainingExecutor, _RegisteredExecutorLifecycleAuthority
] = {}
_EXECUTOR_LIFECYCLE_HISTORY: list[_RegisteredExecutorLifecycleAuthority] = []
_ISSUED_EXECUTOR_LIFECYCLES: set[_RegisteredExecutorLifecycleAuthority] = set()
_EXECUTOR_ANCHORS: dict[RegisteredTrainingExecutor, _RegisteredExecutorAnchor] = {}
_EXECUTOR_ANCHOR_HISTORY: list[_RegisteredExecutorAnchor] = []
_ISSUED_EXECUTOR_TICKETS: set[_ExecutorIssuanceTicket] = set()
_EXECUTOR_TICKET_HISTORY: list[_ExecutorIssuanceTicket] = []
_EXECUTOR_FAILURE_EVENTS: set[_ExecutorFailureEvent] = set()
_EXECUTOR_FAILURE_EVENT_HISTORY: list[_ExecutorFailureEvent] = []
_FAILED_EXECUTORS: weakref.WeakSet[RegisteredTrainingExecutor] = weakref.WeakSet()
_FAILED_EXECUTOR_HISTORY: set[RegisteredTrainingExecutor] = set()
_EXECUTORS_LOCK = threading.RLock()

_HANDOFFS: weakref.WeakKeyDictionary[RegisteredExecutorEpochHandoff, _HandoffState] = (
    weakref.WeakKeyDictionary()
)
_HANDOFF_GUARDS: weakref.WeakKeyDictionary[
    RegisteredExecutorEpochHandoff, _HandoffGuard
] = weakref.WeakKeyDictionary()
_HANDOFF_LIFECYCLES: weakref.WeakKeyDictionary[
    RegisteredExecutorEpochHandoff, _HandoffLifecycle
] = weakref.WeakKeyDictionary()
_HANDOFF_LIFECYCLE_HEADS: dict[
    RegisteredExecutorEpochHandoff, _HandoffLifecycleAuthority
] = {}
_HANDOFF_LIFECYCLE_HISTORY: list[_HandoffLifecycleAuthority] = []
_ISSUED_HANDOFF_LIFECYCLES: set[_HandoffLifecycleAuthority] = set()
_HANDOFF_ANCHORS: dict[RegisteredExecutorEpochHandoff, _HandoffAnchor] = {}
_HANDOFF_ANCHOR_HISTORY: list[_HandoffAnchor] = []
_ISSUED_HANDOFF_TICKETS: set[_HandoffIssuanceTicket] = set()
_HANDOFF_TICKET_HISTORY: list[_HandoffIssuanceTicket] = []
_HANDOFF_EVENTS: set[_HandoffLifecycleEvent] = set()
_HANDOFF_EVENT_HISTORY: list[_HandoffLifecycleEvent] = []
_FAILED_HANDOFFS: weakref.WeakSet[RegisteredExecutorEpochHandoff] = weakref.WeakSet()
_FAILED_HANDOFF_HISTORY: set[RegisteredExecutorEpochHandoff] = set()
_HANDOFFS_LOCK = threading.RLock()

_CONTINUATION_PHASES: dict[_ContinuationReservation, _ContinuationPhase] = {}
_CONTINUATION_HISTORY: list[_ContinuationReservation] = []
_RETIRED_HANDOFFS: dict[RegisteredExecutorEpochHandoff, _RetiredHandoffBinding] = {}
_FINALIZATION_PHASES: dict[_FinalizationReservation, _FinalizationPhase] = {}
_FINALIZATION_HISTORY: list[_FinalizationReservation] = []
_FINALIZED_HANDOFFS: dict[RegisteredExecutorEpochHandoff, _FinalizedHandoffBinding] = {}


@dataclass(frozen=True, slots=True)
class _AuthorityTruthRoutes:
    issue_executor: Callable[
        [
            RegisteredTrainingExecutor,
            _ExecutorIssuanceTicket,
            _RegisteredExecutorLifecycle,
        ],
        None,
    ]
    validate_executor_ticket: Callable[
        [RegisteredTrainingExecutor, _ExecutorIssuanceTicket], bool
    ]
    append_executor_lifecycle: Callable[
        [
            _ExecutorIssuanceTicket,
            _RegisteredExecutorLifecycle,
            _RegisteredExecutorLifecycle,
        ],
        bool,
    ]
    validate_executor_lifecycle: Callable[
        [_ExecutorIssuanceTicket, _RegisteredExecutorLifecycle], bool
    ]
    fail_executor: Callable[[_ExecutorIssuanceTicket], None]
    executor_failed: Callable[[_ExecutorIssuanceTicket], bool]
    issue_handoff: Callable[
        [RegisteredExecutorEpochHandoff, _HandoffIssuanceTicket, _HandoffLifecycle],
        None,
    ]
    validate_handoff_ticket: Callable[
        [RegisteredExecutorEpochHandoff, _HandoffIssuanceTicket], bool
    ]
    append_handoff_lifecycle: Callable[
        [_HandoffIssuanceTicket, _HandoffLifecycle, _HandoffLifecycle], bool
    ]
    validate_handoff_lifecycle: Callable[
        [_HandoffIssuanceTicket, _HandoffLifecycle], bool
    ]
    record_handoff_event: Callable[
        [_HandoffIssuanceTicket, Literal["USED", "FAILED"]], None
    ]
    handoff_has_event: Callable[
        [_HandoffIssuanceTicket, Literal["USED", "FAILED"]], bool
    ]
    begin_advance: Callable[
        [RegisteredTrainingExecutor, RegisteredHistoryBarrier, object],
        _AdvanceAdmission,
    ]
    recover_advance: Callable[
        [RegisteredTrainingExecutor, RegisteredHistoryBarrier, object],
        _AdvanceAdmission | None,
    ]
    recover_retained_advance: Callable[
        [RegisteredTrainingExecutor, RegisteredHistoryBarrier],
        _AdvanceAdmission | None,
    ]
    finish_advance: Callable[[_AdvanceAdmission, BaseException | None], None]
    begin_completion: Callable[
        [RegisteredTrainingExecutor, RegisteredHistoryBarrier, object],
        _CompletionAdmission,
    ]
    recover_completion: Callable[
        [RegisteredTrainingExecutor, RegisteredHistoryBarrier, object],
        _CompletionAdmission | None,
    ]
    recover_retained_completion: Callable[
        [RegisteredTrainingExecutor, RegisteredHistoryBarrier],
        _CompletionAdmission | None,
    ]
    finish_completion: Callable[
        [
            _CompletionAdmission,
            RegisteredCompletedTrainingHistory | None,
            BaseException | None,
        ],
        None,
    ]
    reserve_continuation: Callable[[_ContinuationReservation], None]
    validate_continuation: Callable[
        [_ContinuationReservation, _ContinuationPhase], bool
    ]
    transition_continuation: Callable[
        [_ContinuationReservation, _ContinuationPhase, _ContinuationPhase], None
    ]
    fail_continuation: Callable[[_ContinuationReservation], None]
    commit_retirement: Callable[
        [_ContinuationReservation, _RetiredHandoffBinding], None
    ]
    retired_handoff: Callable[
        [RegisteredExecutorEpochHandoff], _RetiredHandoffBinding | None
    ]
    reserve_finalization: Callable[[_FinalizationReservation], None]
    validate_finalization: Callable[
        [_FinalizationReservation, _FinalizationPhase], bool
    ]
    transition_finalization: Callable[
        [_FinalizationReservation, _FinalizationPhase, _FinalizationPhase], None
    ]
    fail_finalization: Callable[[_FinalizationReservation], None]
    executor_finalization: Callable[
        [RegisteredTrainingExecutor], _FinalizationReservation | None
    ]
    commit_finalization: Callable[
        [_FinalizationReservation, _FinalizedHandoffBinding], None
    ]
    finalized_handoff: Callable[
        [RegisteredExecutorEpochHandoff], _FinalizedHandoffBinding | None
    ]


def _build_authority_truth() -> _AuthorityTruthRoutes:
    """Create closure-owned issuance and monotonic lifecycle truth."""

    executor_tickets: dict[RegisteredTrainingExecutor, tuple[object, ...]] = {}
    executor_lifecycles: dict[_ExecutorIssuanceTicket, list[tuple[object, ...]]] = {}
    failed_executors: set[_ExecutorIssuanceTicket] = set()
    handoff_tickets: dict[
        RegisteredExecutorEpochHandoff,
        tuple[_HandoffIssuanceTicket, _ExecutorIssuanceTicket, object],
    ] = {}
    handoff_lifecycles: dict[_HandoffIssuanceTicket, list[tuple[bool, bool]]] = {}
    handoff_events: set[tuple[_HandoffIssuanceTicket, Literal["USED", "FAILED"]]] = (
        set()
    )
    active_boundaries: dict[RegisteredTrainingExecutor, _BoundaryAdmissionState] = {}
    completed_advances: dict[
        RegisteredTrainingExecutor,
        list[
            tuple[
                RegisteredHistoryBarrier,
                object,
                threading.Event,
                list[BaseException | None],
                tuple[object, ...],
            ]
        ],
    ] = {}
    completed_completions: dict[
        RegisteredTrainingExecutor,
        tuple[
            RegisteredHistoryBarrier,
            object,
            threading.Event,
            list[_CompletionOutcome],
            tuple[object, ...],
        ],
    ] = {}
    continuations: dict[
        _ContinuationReservation, tuple[tuple[object, ...], _ContinuationPhase]
    ] = {}
    executor_continuations: dict[
        RegisteredTrainingExecutor, list[_ContinuationReservation]
    ] = {}
    retired_handoffs: dict[RegisteredExecutorEpochHandoff, _RetiredHandoffBinding] = {}
    finalizations: dict[
        _FinalizationReservation, tuple[tuple[object, ...], _FinalizationPhase]
    ] = {}
    executor_finalizations: dict[
        RegisteredTrainingExecutor, _FinalizationReservation
    ] = {}
    finalized_handoffs: dict[
        RegisteredExecutorEpochHandoff,
        tuple[_FinalizedHandoffBinding, tuple[object, ...]],
    ] = {}
    lock = threading.RLock()

    def executor_frame(lifecycle: _RegisteredExecutorLifecycle) -> tuple[object, ...]:
        expected = lifecycle.expected
        return (
            (type(lifecycle.phase), lifecycle.phase),
            (type(lifecycle.zero_based_epoch), lifecycle.zero_based_epoch),
            (type(lifecycle.optimizer_generation), lifecycle.optimizer_generation),
            type(expected),
            (type(expected.model_sha256), expected.model_sha256),
            (type(expected.optimizer_sha256), expected.optimizer_sha256),
            (type(expected.rng_sha256), expected.rng_sha256),
            (type(expected.rng_state), expected.rng_state),
            (type(lifecycle.active_epoch), id(lifecycle.active_epoch)),
            (type(lifecycle.active_transition), id(lifecycle.active_transition)),
            (type(lifecycle.active_handoff), id(lifecycle.active_handoff)),
            (
                type(lifecycle.previous_history_barrier),
                id(lifecycle.previous_history_barrier),
            ),
            (
                type(lifecycle.continuation_reservation),
                id(lifecycle.continuation_reservation),
            ),
            (
                type(lifecycle.finalization_reservation),
                id(lifecycle.finalization_reservation),
            ),
            (type(lifecycle.final_barrier), id(lifecycle.final_barrier)),
            (
                type(lifecycle.complete_update_trace),
                id(lifecycle.complete_update_trace),
            ),
            (type(lifecycle.completed_history), id(lifecycle.completed_history)),
        )

    def executor_ticket_frame(
        ticket: _ExecutorIssuanceTicket,
    ) -> tuple[object, ...]:
        return (
            (type(ticket), id(ticket)),
            (type(ticket.executor), id(ticket.executor)),
            (type(ticket.token), id(ticket.token)),
            (type(ticket.registration), id(ticket.registration)),
            (type(ticket.process_id), ticket.process_id),
        )

    def issue_executor(
        executor: RegisteredTrainingExecutor,
        ticket: _ExecutorIssuanceTicket,
        lifecycle: _RegisteredExecutorLifecycle,
    ) -> None:
        with lock:
            if executor in executor_tickets or ticket in executor_lifecycles:
                raise Experiment002RegisteredExecutorError(
                    "executor truth was already issued"
                )
            executor_tickets[executor] = executor_ticket_frame(ticket)
            executor_lifecycles[ticket] = [executor_frame(lifecycle)]

    def validate_executor_ticket(
        executor: RegisteredTrainingExecutor, ticket: _ExecutorIssuanceTicket
    ) -> bool:
        with lock:
            frame = executor_tickets.get(executor)
            return (
                frame == executor_ticket_frame(ticket) and ticket.executor is executor
            )

    def append_executor_lifecycle(
        ticket: _ExecutorIssuanceTicket,
        previous: _RegisteredExecutorLifecycle,
        lifecycle: _RegisteredExecutorLifecycle,
    ) -> bool:
        with lock:
            frames = executor_lifecycles.get(ticket)
            if not frames or frames[-1] != executor_frame(previous):
                return False
            frames.append(executor_frame(lifecycle))
            return True

    def validate_executor_lifecycle(
        ticket: _ExecutorIssuanceTicket,
        lifecycle: _RegisteredExecutorLifecycle,
    ) -> bool:
        with lock:
            frames = executor_lifecycles.get(ticket)
            if not frames:
                return False
            return frames[-1] == executor_frame(lifecycle)

    def fail_executor(ticket: _ExecutorIssuanceTicket) -> None:
        with lock:
            if ticket in executor_lifecycles:
                failed_executors.add(ticket)

    def executor_failed(ticket: _ExecutorIssuanceTicket) -> bool:
        with lock:
            return ticket in failed_executors

    def issue_handoff(
        handoff: RegisteredExecutorEpochHandoff,
        ticket: _HandoffIssuanceTicket,
        lifecycle: _HandoffLifecycle,
    ) -> None:
        with lock:
            if (
                handoff in handoff_tickets
                or ticket in handoff_lifecycles
                or ticket.executor_ticket not in executor_lifecycles
            ):
                raise Experiment002RegisteredExecutorError(
                    "handoff truth was already issued or has no executor"
                )
            handoff_tickets[handoff] = (
                ticket,
                ticket.executor_ticket,
                ticket.handoff_token,
            )
            handoff_lifecycles[ticket] = [(lifecycle.used, lifecycle.failed)]

    def validate_handoff_ticket(
        handoff: RegisteredExecutorEpochHandoff, ticket: _HandoffIssuanceTicket
    ) -> bool:
        with lock:
            frame = handoff_tickets.get(handoff)
            return (
                frame
                == (
                    ticket,
                    ticket.executor_ticket,
                    ticket.handoff_token,
                )
                and ticket.handoff is handoff
            )

    def append_handoff_lifecycle(
        ticket: _HandoffIssuanceTicket,
        previous: _HandoffLifecycle,
        lifecycle: _HandoffLifecycle,
    ) -> bool:
        with lock:
            frames = handoff_lifecycles.get(ticket)
            previous_frame = (previous.used, previous.failed)
            if not frames or frames[-1] != previous_frame:
                return False
            frames.append((lifecycle.used, lifecycle.failed))
            return True

    def validate_handoff_lifecycle(
        ticket: _HandoffIssuanceTicket, lifecycle: _HandoffLifecycle
    ) -> bool:
        with lock:
            frames = handoff_lifecycles.get(ticket)
            if not frames:
                return False
            return frames[-1] == (
                lifecycle.used,
                lifecycle.failed,
            )

    def record_handoff_event(
        ticket: _HandoffIssuanceTicket, kind: Literal["USED", "FAILED"]
    ) -> None:
        with lock:
            if ticket in handoff_lifecycles:
                handoff_events.add((ticket, kind))

    def handoff_has_event(
        ticket: _HandoffIssuanceTicket, kind: Literal["USED", "FAILED"]
    ) -> bool:
        with lock:
            return (ticket, kind) in handoff_events

    def continuation_frame(
        reservation: _ContinuationReservation,
    ) -> tuple[object, ...]:
        runtime = reservation.runtime_digests
        return (
            (type(reservation.token), id(reservation.token)),
            (type(reservation.executor), id(reservation.executor)),
            (type(reservation.executor_ticket), id(reservation.executor_ticket)),
            (type(reservation.handoff), id(reservation.handoff)),
            (type(reservation.handoff_ticket), id(reservation.handoff_ticket)),
            (type(reservation.barrier), id(reservation.barrier)),
            (type(reservation.barrier_type), id(reservation.barrier_type)),
            (type(reservation.evaluated_epoch), id(reservation.evaluated_epoch)),
            (
                type(reservation.evaluated_epoch_type),
                id(reservation.evaluated_epoch_type),
            ),
            (
                type(reservation.evaluated_authority_sha256),
                reservation.evaluated_authority_sha256,
            ),
            (type(reservation.registration), id(reservation.registration)),
            (type(reservation.validation_inputs), id(reservation.validation_inputs)),
            (type(reservation.process_id), reservation.process_id),
            (type(reservation.seed), reservation.seed),
            (type(reservation.zero_based_epoch), reservation.zero_based_epoch),
            (type(reservation.optimizer_generation), reservation.optimizer_generation),
            (type(runtime.model_sha256), runtime.model_sha256),
            (type(runtime.optimizer_sha256), runtime.optimizer_sha256),
            (type(runtime.rng_sha256), runtime.rng_sha256),
            (type(runtime.rng_state), runtime.rng_state),
            (
                type(reservation.previous_history_barrier),
                id(reservation.previous_history_barrier),
            ),
            (type(reservation.handoff_token), id(reservation.handoff_token)),
            (type(reservation.one_shot_token), id(reservation.one_shot_token)),
        )

    def retirement_frame(binding: _RetiredHandoffBinding) -> tuple[object, ...]:
        runtime = binding.runtime_digests
        return (
            (type(binding.reservation), id(binding.reservation)),
            (type(binding.executor), id(binding.executor)),
            (type(binding.handoff), id(binding.handoff)),
            (type(binding.barrier), id(binding.barrier)),
            (type(binding.evaluated_epoch), id(binding.evaluated_epoch)),
            (
                type(binding.evaluated_authority_sha256),
                binding.evaluated_authority_sha256,
            ),
            (type(binding.registration), id(binding.registration)),
            (type(binding.validation_inputs), id(binding.validation_inputs)),
            (type(binding.process_id), binding.process_id),
            (type(binding.seed), binding.seed),
            (type(binding.zero_based_epoch), binding.zero_based_epoch),
            (type(binding.optimizer_generation), binding.optimizer_generation),
            (type(runtime.model_sha256), runtime.model_sha256),
            (type(runtime.optimizer_sha256), runtime.optimizer_sha256),
            (type(runtime.rng_sha256), runtime.rng_sha256),
            (type(runtime.rng_state), runtime.rng_state),
            (
                type(binding.previous_history_barrier),
                id(binding.previous_history_barrier),
            ),
            (type(binding.handoff_token), id(binding.handoff_token)),
            (type(binding.one_shot_token), id(binding.one_shot_token)),
        )

    def finalization_frame(
        reservation: _FinalizationReservation,
    ) -> tuple[object, ...]:
        runtime = reservation.runtime_digests
        return (
            (type(reservation.token), id(reservation.token)),
            (type(reservation.executor), id(reservation.executor)),
            (type(reservation.executor_ticket), id(reservation.executor_ticket)),
            (type(reservation.handoff), id(reservation.handoff)),
            (type(reservation.handoff_ticket), id(reservation.handoff_ticket)),
            (type(reservation.final_barrier), id(reservation.final_barrier)),
            (type(reservation.barrier_type), id(reservation.barrier_type)),
            (type(reservation.history), id(reservation.history)),
            (type(reservation.history_type), id(reservation.history_type)),
            (type(reservation.evaluated_epoch), id(reservation.evaluated_epoch)),
            (
                type(reservation.evaluated_epoch_type),
                id(reservation.evaluated_epoch_type),
            ),
            (
                type(reservation.evaluated_authority_sha256),
                reservation.evaluated_authority_sha256,
            ),
            (type(reservation.registration), id(reservation.registration)),
            (type(reservation.validation_inputs), id(reservation.validation_inputs)),
            (type(reservation.process_id), reservation.process_id),
            (type(reservation.seed), reservation.seed),
            (type(reservation.zero_based_epoch), reservation.zero_based_epoch),
            (type(reservation.optimizer_generation), reservation.optimizer_generation),
            (type(runtime.model_sha256), runtime.model_sha256),
            (type(runtime.optimizer_sha256), runtime.optimizer_sha256),
            (type(runtime.rng_sha256), runtime.rng_sha256),
            (type(runtime.rng_state), runtime.rng_state),
            (
                type(reservation.previous_history_barrier),
                id(reservation.previous_history_barrier),
            ),
            (type(reservation.handoff_token), id(reservation.handoff_token)),
            (type(reservation.one_shot_token), id(reservation.one_shot_token)),
            (type(reservation.trace), id(reservation.trace)),
            type(reservation.epoch_traces),
            tuple(
                (type(epoch_trace), id(epoch_trace))
                for epoch_trace in reservation.epoch_traces
            ),
        )

    def finalized_frame(binding: _FinalizedHandoffBinding) -> tuple[object, ...]:
        runtime = binding.runtime_digests
        return (
            (type(binding.reservation), id(binding.reservation)),
            (type(binding.executor), id(binding.executor)),
            (type(binding.handoff), id(binding.handoff)),
            (type(binding.final_barrier), id(binding.final_barrier)),
            (type(binding.history), id(binding.history)),
            (type(binding.evaluated_epoch), id(binding.evaluated_epoch)),
            (
                type(binding.evaluated_authority_sha256),
                binding.evaluated_authority_sha256,
            ),
            (type(binding.complete_update_trace), id(binding.complete_update_trace)),
            (
                type(binding.complete_update_trace_type),
                id(binding.complete_update_trace_type),
            ),
            (type(binding.completed_history), id(binding.completed_history)),
            (
                type(binding.completed_history_type),
                id(binding.completed_history_type),
            ),
            (type(binding.registration), id(binding.registration)),
            (type(binding.validation_inputs), id(binding.validation_inputs)),
            (type(binding.process_id), binding.process_id),
            (type(binding.seed), binding.seed),
            (type(binding.zero_based_epoch), binding.zero_based_epoch),
            (type(binding.optimizer_generation), binding.optimizer_generation),
            (type(runtime.model_sha256), runtime.model_sha256),
            (type(runtime.optimizer_sha256), runtime.optimizer_sha256),
            (type(runtime.rng_sha256), runtime.rng_sha256),
            (type(runtime.rng_state), runtime.rng_state),
            (
                type(binding.previous_history_barrier),
                id(binding.previous_history_barrier),
            ),
            (type(binding.handoff_token), id(binding.handoff_token)),
            (type(binding.one_shot_token), id(binding.one_shot_token)),
        )

    def advance_outcome_frame(
        outcome: list[BaseException | None],
    ) -> tuple[object, ...] | None:
        if type(outcome) is not list or len(outcome) != 1:
            return None
        value = outcome[0]
        return (type(value), id(value))

    def completion_outcome_frame(
        outcome: list[_CompletionOutcome],
    ) -> tuple[object, ...] | None:
        if (
            type(outcome) is not list
            or len(outcome) != 1
            or type(outcome[0]) is not _CompletionOutcome
        ):
            return None
        value = outcome[0]
        return (
            (type(value), id(value)),
            (type(value.result), id(value.result)),
            (type(value.error), id(value.error)),
        )

    def begin_advance(
        executor: RegisteredTrainingExecutor,
        barrier: RegisteredHistoryBarrier,
        caller_token: object,
    ) -> _AdvanceAdmission:
        with lock:
            for (
                completed_barrier,
                completed_token,
                completed,
                retained_outcome,
                retained_frame,
            ) in completed_advances.get(executor, []):
                if completed_barrier is barrier:
                    if type(completed) is not _EVENT_TYPE or not completed.is_set():
                        raise _BoundaryTruthCorruption(
                            "retained continuation event authority changed"
                        )
                    if advance_outcome_frame(retained_outcome) != retained_frame:
                        raise _BoundaryTruthCorruption(
                            "retained continuation outcome authority changed"
                        )
                    return _AdvanceAdmission(
                        completed_token,
                        executor,
                        barrier,
                        False,
                        completed,
                        retained_outcome,
                    )
            if executor in completed_completions:
                raise _BoundaryAdmissionConflict(
                    "completed executor rejects a new advance boundary"
                )
            active = active_boundaries.get(executor)
            if active is None:
                completed = threading.Event()
                outcome: list[BaseException | None] = []
                active_boundaries[executor] = _BoundaryAdmissionState(
                    "ADVANCE", barrier, caller_token, completed, outcome
                )
                return _AdvanceAdmission(
                    caller_token,
                    executor,
                    barrier,
                    True,
                    completed,
                    outcome,
                )
            if active.kind != "ADVANCE" or active.barrier is not barrier:
                raise _BoundaryAdmissionConflict(
                    "a different executor boundary is already active"
                )
            active_outcome = cast(list[BaseException | None], active.outcome)
            return _AdvanceAdmission(
                active.token,
                executor,
                barrier,
                False,
                active.completed,
                active_outcome,
            )

    def recover_advance(
        executor: RegisteredTrainingExecutor,
        barrier: RegisteredHistoryBarrier,
        caller_token: object,
    ) -> _AdvanceAdmission | None:
        with lock:
            active = active_boundaries.get(executor)
            if (
                active is not None
                and active.kind == "ADVANCE"
                and active.barrier is barrier
                and active.token is caller_token
            ):
                return _AdvanceAdmission(
                    caller_token,
                    executor,
                    barrier,
                    True,
                    active.completed,
                    cast(list[BaseException | None], active.outcome),
                )
            for (
                completed_barrier,
                completed_token,
                completed,
                retained_outcome,
                _retained_frame,
            ) in completed_advances.get(executor, []):
                if completed_barrier is barrier and completed_token is caller_token:
                    return _AdvanceAdmission(
                        caller_token,
                        executor,
                        barrier,
                        True,
                        completed,
                        retained_outcome,
                    )
            return None

    def recover_retained_advance(
        executor: RegisteredTrainingExecutor,
        barrier: RegisteredHistoryBarrier,
    ) -> _AdvanceAdmission | None:
        with lock:
            for (
                completed_barrier,
                completed_token,
                completed,
                retained_outcome,
                _retained_frame,
            ) in completed_advances.get(executor, []):
                if completed_barrier is barrier:
                    return _AdvanceAdmission(
                        completed_token,
                        executor,
                        barrier,
                        True,
                        completed,
                        retained_outcome,
                    )
            return None

    def finish_advance(
        admission: _AdvanceAdmission,
        outcome: BaseException | None,
    ) -> None:
        try:
            with lock:
                if type(admission.leader) is not bool or admission.leader is not True:
                    raise Experiment002RegisteredExecutorError(
                        "only the continuation leader may publish its outcome"
                    )
                retained = completed_advances.setdefault(admission.executor, [])
                exact_retained_index = next(
                    (
                        index
                        for index, item in enumerate(retained)
                        if item[0] is admission.barrier and item[1] is admission.token
                    ),
                    None,
                )
                active = active_boundaries.get(admission.executor)
                if exact_retained_index is not None:
                    exact_retained = retained[exact_retained_index]
                    if (
                        exact_retained[2] is not admission.completed
                        or exact_retained[3] is not admission.outcome
                    ):
                        raise Experiment002RegisteredExecutorError(
                            "retained continuation admission truth changed"
                        )
                    admission.outcome[:] = [outcome]
                    frame = advance_outcome_frame(admission.outcome)
                    assert frame is not None
                    retained[exact_retained_index] = (
                        admission.barrier,
                        admission.token,
                        admission.completed,
                        admission.outcome,
                        frame,
                    )
                    if (
                        active is not None
                        and active.kind == "ADVANCE"
                        and active.token is admission.token
                    ):
                        del active_boundaries[admission.executor]
                    return
                if (
                    active is None
                    or active.kind != "ADVANCE"
                    or active.barrier is not admission.barrier
                    or active.token is not admission.token
                    or active.completed is not admission.completed
                    or active.outcome is not admission.outcome
                ):
                    raise Experiment002RegisteredExecutorError(
                        "continuation admission truth changed"
                    )
                admission.outcome[:] = [outcome]
                if len(retained) >= _REGISTERED_EPOCH_COUNT - 1:
                    admission.outcome[:] = [
                        Experiment002RegisteredExecutorError(
                            "too many retained continuation outcomes"
                        )
                    ]
                else:
                    frame = advance_outcome_frame(admission.outcome)
                    assert frame is not None
                    retained.append(
                        (
                            admission.barrier,
                            admission.token,
                            admission.completed,
                            admission.outcome,
                            frame,
                        )
                    )
                del active_boundaries[admission.executor]
        finally:
            admission.completed.set()

    def begin_completion(
        executor: RegisteredTrainingExecutor,
        barrier: RegisteredHistoryBarrier,
        caller_token: object,
    ) -> _CompletionAdmission:
        with lock:
            retained = completed_completions.get(executor)
            if retained is not None:
                (
                    retained_barrier,
                    retained_token,
                    completed,
                    outcome,
                    retained_frame,
                ) = retained
                if retained_barrier is not barrier:
                    raise _BoundaryAdmissionConflict(
                        "executor already completed a different final boundary"
                    )
                if type(completed) is not _EVENT_TYPE or not completed.is_set():
                    raise _BoundaryTruthCorruption(
                        "retained completion event authority changed"
                    )
                if completion_outcome_frame(outcome) != retained_frame:
                    raise _BoundaryTruthCorruption(
                        "retained completion outcome authority changed"
                    )
                return _CompletionAdmission(
                    retained_token,
                    executor,
                    barrier,
                    False,
                    completed,
                    outcome,
                )
            active = active_boundaries.get(executor)
            if active is None:
                completed = threading.Event()
                completion_outcome: list[_CompletionOutcome] = []
                active_boundaries[executor] = _BoundaryAdmissionState(
                    "COMPLETE",
                    barrier,
                    caller_token,
                    completed,
                    completion_outcome,
                )
                return _CompletionAdmission(
                    caller_token,
                    executor,
                    barrier,
                    True,
                    completed,
                    completion_outcome,
                )
            if active.kind != "COMPLETE" or active.barrier is not barrier:
                raise _BoundaryAdmissionConflict(
                    "a different executor boundary is already active"
                )
            active_outcome = cast(list[_CompletionOutcome], active.outcome)
            return _CompletionAdmission(
                active.token,
                executor,
                barrier,
                False,
                active.completed,
                active_outcome,
            )

    def recover_completion(
        executor: RegisteredTrainingExecutor,
        barrier: RegisteredHistoryBarrier,
        caller_token: object,
    ) -> _CompletionAdmission | None:
        with lock:
            retained = completed_completions.get(executor)
            if retained is not None:
                (
                    retained_barrier,
                    retained_token,
                    completed,
                    outcome,
                    _retained_frame,
                ) = retained
                if retained_barrier is barrier and retained_token is caller_token:
                    return _CompletionAdmission(
                        caller_token,
                        executor,
                        barrier,
                        True,
                        completed,
                        outcome,
                    )
            active = active_boundaries.get(executor)
            if (
                active is not None
                and active.kind == "COMPLETE"
                and active.barrier is barrier
                and active.token is caller_token
            ):
                return _CompletionAdmission(
                    caller_token,
                    executor,
                    barrier,
                    True,
                    active.completed,
                    cast(list[_CompletionOutcome], active.outcome),
                )
            return None

    def recover_retained_completion(
        executor: RegisteredTrainingExecutor,
        barrier: RegisteredHistoryBarrier,
    ) -> _CompletionAdmission | None:
        with lock:
            retained = completed_completions.get(executor)
            if retained is None:
                return None
            (
                retained_barrier,
                retained_token,
                completed,
                outcome,
                _retained_frame,
            ) = retained
            if retained_barrier is not barrier:
                return None
            return _CompletionAdmission(
                retained_token,
                executor,
                barrier,
                True,
                completed,
                outcome,
            )

    def finish_completion(
        admission: _CompletionAdmission,
        result: RegisteredCompletedTrainingHistory | None,
        error: BaseException | None,
    ) -> None:
        try:
            with lock:
                if type(admission.leader) is not bool or admission.leader is not True:
                    raise Experiment002RegisteredExecutorError(
                        "only the completion leader may publish its outcome"
                    )
                if (result is None) == (error is None):
                    raise Experiment002RegisteredExecutorError(
                        "completion published an invalid outcome"
                    )
                retained = completed_completions.get(admission.executor)
                active = active_boundaries.get(admission.executor)
                if retained is not None:
                    (
                        retained_barrier,
                        retained_token,
                        retained_completed,
                        retained_outcome,
                        _retained_frame,
                    ) = retained
                    if (
                        retained_barrier is not admission.barrier
                        or retained_token is not admission.token
                        or retained_completed is not admission.completed
                        or retained_outcome is not admission.outcome
                    ):
                        raise Experiment002RegisteredExecutorError(
                            "retained completion admission truth changed"
                        )
                    admission.outcome[:] = [_CompletionOutcome(result, error)]
                    frame = completion_outcome_frame(admission.outcome)
                    assert frame is not None
                    completed_completions[admission.executor] = (
                        admission.barrier,
                        admission.token,
                        admission.completed,
                        admission.outcome,
                        frame,
                    )
                    if (
                        active is not None
                        and active.kind == "COMPLETE"
                        and active.token is admission.token
                    ):
                        del active_boundaries[admission.executor]
                    return
                if (
                    active is None
                    or active.kind != "COMPLETE"
                    or active.barrier is not admission.barrier
                    or active.token is not admission.token
                    or active.completed is not admission.completed
                    or active.outcome is not admission.outcome
                ):
                    raise Experiment002RegisteredExecutorError(
                        "completion admission truth changed"
                    )
                admission.outcome[:] = [_CompletionOutcome(result, error)]
                frame = completion_outcome_frame(admission.outcome)
                assert frame is not None
                completed_completions[admission.executor] = (
                    admission.barrier,
                    admission.token,
                    admission.completed,
                    admission.outcome,
                    frame,
                )
                del active_boundaries[admission.executor]
        finally:
            admission.completed.set()

    def reserve_continuation(reservation: _ContinuationReservation) -> None:
        with lock:
            if reservation in continuations:
                raise Experiment002RegisteredExecutorError(
                    "continuation reservation was already recorded"
                )
            prior = executor_continuations.setdefault(reservation.executor, [])
            if prior:
                previous = prior[-1]
                previous_truth = continuations.get(previous)
                if (
                    previous_truth is None
                    or previous_truth[1] != "COMMITTED"
                    or reservation.zero_based_epoch != previous.zero_based_epoch + 1
                    or reservation.previous_history_barrier is not previous.barrier
                ):
                    raise Experiment002RegisteredExecutorError(
                        "continuation reservation is not monotonic"
                    )
            elif (
                reservation.zero_based_epoch != 0
                or reservation.previous_history_barrier is not None
            ):
                raise Experiment002RegisteredExecutorError(
                    "the first continuation reservation is not epoch zero"
                )
            continuations[reservation] = (continuation_frame(reservation), "RESERVED")
            prior.append(reservation)

    def validate_continuation(
        reservation: _ContinuationReservation,
        phase: _ContinuationPhase,
    ) -> bool:
        with lock:
            truth = continuations.get(reservation)
            return truth == (continuation_frame(reservation), phase)

    def transition_continuation(
        reservation: _ContinuationReservation,
        expected: _ContinuationPhase,
        phase: _ContinuationPhase,
    ) -> None:
        with lock:
            truth = continuations.get(reservation)
            if truth != (continuation_frame(reservation), expected):
                raise Experiment002RegisteredExecutorError(
                    "continuation lifecycle lost closure-owned continuity"
                )
            allowed = {
                ("RESERVED", "CONSUMED"),
                ("RESERVED", "FAILED"),
                ("CONSUMED", "FAILED"),
                ("COMMITTED", "FAILED"),
            }
            if (expected, phase) not in allowed:
                raise Experiment002RegisteredExecutorError(
                    "continuation lifecycle transition is invalid"
                )
            continuations[reservation] = (truth[0], phase)

    def fail_continuation(reservation: _ContinuationReservation) -> None:
        with lock:
            truth = continuations.get(reservation)
            if truth is None or truth[1] == "FAILED":
                return
            continuations[reservation] = (truth[0], "FAILED")

    def commit_retirement(
        reservation: _ContinuationReservation,
        binding: _RetiredHandoffBinding,
    ) -> None:
        with lock:
            truth = continuations.get(reservation)
            if truth != (continuation_frame(reservation), "CONSUMED"):
                raise Experiment002RegisteredExecutorError(
                    "continuation is not consumed before retirement"
                )
            if (
                binding.reservation is not reservation
                or binding.handoff is not reservation.handoff
                or retirement_frame(binding)
                != retirement_frame(_retirement_from_reservation(reservation))
                or binding.handoff in retired_handoffs
            ):
                raise Experiment002RegisteredExecutorError(
                    "retired handoff binding differs from its reservation"
                )
            retired_handoffs[binding.handoff] = binding
            continuations[reservation] = (truth[0], "COMMITTED")

    def retired_handoff(
        handoff: RegisteredExecutorEpochHandoff,
    ) -> _RetiredHandoffBinding | None:
        with lock:
            binding = retired_handoffs.get(handoff)
            if binding is None:
                return None
            if retirement_frame(binding) != retirement_frame(
                _retirement_from_reservation(binding.reservation)
            ):
                return None
            truth = continuations.get(binding.reservation)
            if truth is None or truth[1] != "COMMITTED":
                return None
            return binding

    def reserve_finalization(reservation: _FinalizationReservation) -> None:
        with lock:
            if (
                reservation in finalizations
                or reservation.executor in executor_finalizations
            ):
                raise Experiment002RegisteredExecutorError(
                    "executor finalization was already reserved"
                )
            prior = executor_continuations.get(reservation.executor, [])
            if len(prior) != _REGISTERED_EPOCH_COUNT - 1:
                raise Experiment002RegisteredExecutorError(
                    "finalization lacks all non-final continuation truth"
                )
            previous_barrier: RegisteredHistoryBarrier | None = None
            for expected_epoch, previous in enumerate(prior):
                previous_truth = continuations.get(previous)
                if (
                    previous_truth != (continuation_frame(previous), "COMMITTED")
                    or previous.executor is not reservation.executor
                    or previous.executor_ticket is not reservation.executor_ticket
                    or previous.registration is not reservation.registration
                    or previous.validation_inputs is not reservation.validation_inputs
                    or previous.process_id != reservation.process_id
                    or previous.seed != reservation.seed
                    or previous.zero_based_epoch != expected_epoch
                    or previous.optimizer_generation
                    != (expected_epoch + 1) * _UPDATES_PER_EPOCH
                    or previous.previous_history_barrier is not previous_barrier
                    or previous.barrier_type is not reservation.barrier_type
                ):
                    raise Experiment002RegisteredExecutorError(
                        "finalization lacks an exact committed continuation chain"
                    )
                previous_barrier = previous.barrier
            if reservation.previous_history_barrier is not previous_barrier:
                raise Experiment002RegisteredExecutorError(
                    "finalization does not follow the exact epoch-28 barrier"
                )
            finalizations[reservation] = (
                finalization_frame(reservation),
                "RESERVED",
            )
            executor_finalizations[reservation.executor] = reservation

    def validate_finalization(
        reservation: _FinalizationReservation,
        phase: _FinalizationPhase,
    ) -> bool:
        with lock:
            return finalizations.get(reservation) == (
                finalization_frame(reservation),
                phase,
            )

    def transition_finalization(
        reservation: _FinalizationReservation,
        expected: _FinalizationPhase,
        phase: _FinalizationPhase,
    ) -> None:
        with lock:
            truth = finalizations.get(reservation)
            if truth != (finalization_frame(reservation), expected):
                raise Experiment002RegisteredExecutorError(
                    "finalization lifecycle lost closure-owned continuity"
                )
            allowed = {
                ("RESERVED", "TRACE_COMPLETE"),
                ("TRACE_COMPLETE", "HISTORY_COMPLETE"),
                ("HISTORY_COMPLETE", "COMMITTED"),
                ("RESERVED", "FAILED"),
                ("TRACE_COMPLETE", "FAILED"),
                ("HISTORY_COMPLETE", "FAILED"),
                ("COMMITTED", "FAILED"),
            }
            if (expected, phase) not in allowed:
                raise Experiment002RegisteredExecutorError(
                    "finalization lifecycle transition is invalid"
                )
            finalizations[reservation] = (truth[0], phase)

    def fail_finalization(reservation: _FinalizationReservation) -> None:
        with lock:
            truth = finalizations.get(reservation)
            if truth is None or truth[1] == "FAILED":
                return
            finalizations[reservation] = (truth[0], "FAILED")

    def executor_finalization(
        executor: RegisteredTrainingExecutor,
    ) -> _FinalizationReservation | None:
        with lock:
            candidates = [
                reservation
                for reservation, (frame, _phase) in finalizations.items()
                if reservation.executor is executor
                and frame == finalization_frame(reservation)
            ]
            if len(candidates) != 1:
                return None
            reservation = candidates[0]
            mapped = executor_finalizations.get(executor)
            if mapped is not None and mapped is not reservation:
                return None
            return reservation

    def commit_finalization(
        reservation: _FinalizationReservation,
        binding: _FinalizedHandoffBinding,
    ) -> None:
        with lock:
            truth = finalizations.get(reservation)
            if truth != (finalization_frame(reservation), "HISTORY_COMPLETE"):
                raise Experiment002RegisteredExecutorError(
                    "history is not complete before finalized handoff publication"
                )
            expected = _finalized_from_reservation(
                reservation,
                binding.complete_update_trace,
                binding.completed_history,
            )
            if (
                binding.reservation is not reservation
                or binding.handoff is not reservation.handoff
                or finalized_frame(binding) != finalized_frame(expected)
                or binding.handoff in finalized_handoffs
                or binding.handoff in retired_handoffs
            ):
                raise Experiment002RegisteredExecutorError(
                    "finalized handoff binding differs from its reservation"
                )
            finalized_handoffs[binding.handoff] = (
                binding,
                finalized_frame(binding),
            )
            finalizations[reservation] = (truth[0], "COMMITTED")

    def finalized_handoff(
        handoff: RegisteredExecutorEpochHandoff,
    ) -> _FinalizedHandoffBinding | None:
        with lock:
            retained = finalized_handoffs.get(handoff)
            if retained is None:
                return None
            binding, committed_frame = retained
            expected = _finalized_from_reservation(
                binding.reservation,
                binding.complete_update_trace,
                binding.completed_history,
            )
            if finalized_frame(
                binding
            ) != committed_frame or committed_frame != finalized_frame(expected):
                return None
            truth = finalizations.get(binding.reservation)
            if truth is None or truth[1] != "COMMITTED":
                return None
            return binding

    return _AuthorityTruthRoutes(
        issue_executor=issue_executor,
        validate_executor_ticket=validate_executor_ticket,
        append_executor_lifecycle=append_executor_lifecycle,
        validate_executor_lifecycle=validate_executor_lifecycle,
        fail_executor=fail_executor,
        executor_failed=executor_failed,
        issue_handoff=issue_handoff,
        validate_handoff_ticket=validate_handoff_ticket,
        append_handoff_lifecycle=append_handoff_lifecycle,
        validate_handoff_lifecycle=validate_handoff_lifecycle,
        record_handoff_event=record_handoff_event,
        handoff_has_event=handoff_has_event,
        begin_advance=begin_advance,
        recover_advance=recover_advance,
        recover_retained_advance=recover_retained_advance,
        finish_advance=finish_advance,
        begin_completion=begin_completion,
        recover_completion=recover_completion,
        recover_retained_completion=recover_retained_completion,
        finish_completion=finish_completion,
        reserve_continuation=reserve_continuation,
        validate_continuation=validate_continuation,
        transition_continuation=transition_continuation,
        fail_continuation=fail_continuation,
        commit_retirement=commit_retirement,
        retired_handoff=retired_handoff,
        reserve_finalization=reserve_finalization,
        validate_finalization=validate_finalization,
        transition_finalization=transition_finalization,
        fail_finalization=fail_finalization,
        executor_finalization=executor_finalization,
        commit_finalization=commit_finalization,
        finalized_handoff=finalized_handoff,
    )


_AUTHORITY_TRUTH: Final = _build_authority_truth()


def create_registered_training_executor(
    registration: VerifiedRunRegistration,
    training_inputs: RegisteredTrainingInputSource,
    validation_inputs: RegisteredValidationInputs,
    *,
    seed: int,
) -> RegisteredTrainingExecutor:
    """Create the process's sole source-bound registered training executor."""

    if type(registration) is not VerifiedRunRegistration:
        raise TypeError("registration must be a VerifiedRunRegistration")
    if type(training_inputs) is not RegisteredTrainingInputSource:
        raise TypeError("training_inputs must be a RegisteredTrainingInputSource")
    if type(validation_inputs) is not RegisteredValidationInputs:
        raise TypeError("validation_inputs must be RegisteredValidationInputs")
    _require_registered_seed(seed)
    process_id = os.getpid()
    _reserve_process_creation(process_id)

    bridge_authority: RegisteredExecutorSessionAuthority | None = None
    trace: UpdateTraceAccumulator | None = None
    try:
        reverify_verified_run_registration(registration)
        source_state = _issued_source_state(training_inputs)
        verify_registered_validation_inputs(validation_inputs)
        _numeric._configure_and_verify_runtime()
        torch.manual_seed(seed)
        model = CausalKWS()
        rng_after_constructor = _numeric._rng_state_bytes()

        named_parameters = tuple(model.named_parameters())
        parameter_names = tuple(name for name, _ in named_parameters)
        parameters = tuple(parameter for _, parameter in named_parameters)
        _numeric._validate_parameter_capture(model, parameter_names, parameters)
        optimizer = torch.optim.AdamW(
            parameters,
            lr=_numeric._ADAMW_CONSTRUCTOR_LR,
            betas=_numeric._ADAMW_BETAS,
            eps=_numeric._ADAMW_EPS,
            weight_decay=_numeric._ADAMW_WEIGHT_DECAY,
            amsgrad=False,
            maximize=False,
            foreach=False,
            capturable=False,
            differentiable=False,
            fused=False,
        )
        lock = threading.RLock()
        runtime = _numeric._ExecutorState(
            token=object(),
            route_marker=_REGISTERED_ROUTE,
            seed=seed,
            model=model,
            optimizer=optimizer,
            parameter_names=parameter_names,
            parameters=parameters,
            expected=_numeric._RuntimeDigests("", "", "", b""),
            lock=lock,
        )
        expected = _numeric._stable_runtime_digests(runtime, expected_generation=0)
        if expected.rng_state != rng_after_constructor:
            raise Experiment002RegisteredExecutorError(
                "executor construction consumed Torch RNG after CausalKWS"
            )
        runtime.expected = expected

        bridge_authority = _issue_registered_executor_session(seed)
        bridge_snapshot = _executor_session_snapshot(bridge_authority)
        trace = UpdateTraceAccumulator(seed)
        reverify_verified_run_registration(registration)
        if _issued_source_state(training_inputs) is not source_state:
            raise Experiment002RegisteredExecutorError(
                "registered training inputs changed during executor creation"
            )
        verify_registered_validation_inputs(validation_inputs)
        final_expected = _numeric._stable_runtime_digests(
            runtime, expected_generation=0
        )
        if final_expected != expected:
            raise Experiment002RegisteredExecutorError(
                "registered runtime changed during executor creation"
            )
        return _issue_registered_executor(
            registration=registration,
            process_id=process_id,
            seed=seed,
            training_inputs=training_inputs,
            validation_inputs=validation_inputs,
            bridge_authority=bridge_authority,
            bridge_session_token=bridge_snapshot.session_token,
            trace=trace,
            runtime=runtime,
            expected=expected,
            lock=lock,
        )
    except BaseException:
        if trace is not None:
            with contextlib.suppress(BaseException):
                _fail_registered_trace(trace)
        if bridge_authority is not None:
            with contextlib.suppress(BaseException):
                _fail_registered_executor_session(bridge_authority)
        raise


def execute_registered_training_epoch(
    executor: RegisteredTrainingExecutor,
) -> RegisteredExecutorEpochHandoff:
    """Train the executor's current epoch and stop at evaluation handoff."""

    state = _issued_executor_state(executor)
    lock = _trusted_executor_lock(executor, state)
    with lock:
        _validate_executor_or_fail(executor, state)
        if state.phase != "READY_TO_TRAIN":
            raise Experiment002RegisteredExecutorError(
                "executor is not ready to train one epoch"
            )
        epoch: RegisteredTrainingEpoch | None = None
        try:
            reverify_verified_run_registration(state.registration)
            _verify_bound_inputs(state)
            _numeric._verify_runtime()
            entry = _numeric._stable_runtime_digests(
                state.runtime, expected_generation=state.optimizer_generation
            )
            if entry != state.expected:
                raise Experiment002RegisteredExecutorError(
                    "runtime continuity changed before epoch execution"
                )
            state.phase = "TRAINING"
            _publish_executor_lifecycle(executor, state)
            epoch = _begin_registered_epoch_for_executor(
                state.training_inputs,
                executor_authority=state.bridge_authority,
                seed=state.seed,
                zero_based_epoch=state.zero_based_epoch,
            )
            state.active_epoch = epoch
            _publish_executor_lifecycle(executor, state)

            epoch_start = state.zero_based_epoch * _UPDATES_PER_EPOCH
            for offset in range(_UPDATES_PER_EPOCH):
                _execute_registered_update(
                    executor,
                    state,
                    epoch,
                    global_update=epoch_start + offset,
                )

            population = _executor_finish_epoch(epoch, state.bridge_authority)
            verify_completed_registered_training_population(
                population,
                seed=state.seed,
                zero_based_epoch=state.zero_based_epoch,
            )
            epoch_trace = state.trace._finish_registered_epoch(population)
            verify_registered_epoch_update_trace(epoch_trace)
            final_digests = _numeric._stable_runtime_digests(
                state.runtime, expected_generation=state.optimizer_generation
            )
            if (
                final_digests != state.expected
                or state.runtime.model.training is not True
            ):
                raise Experiment002RegisteredExecutorError(
                    "runtime changed across registered epoch finalization"
                )
            reverify_verified_run_registration(state.registration)
            _verify_bound_inputs(state)
            handoff = _issue_epoch_handoff(
                executor,
                state,
                population=population,
                epoch_trace=epoch_trace,
            )
            state.active_epoch = None
            state.active_transition = None
            state.active_handoff = handoff
            state.phase = "AWAITING_EVALUATION"
            _publish_executor_lifecycle(executor, state)
            _registered_epoch_handoff_snapshot(handoff)
            return handoff
        except BaseException:
            _terminal_fail_executor(executor, state)
            _propagate_executor_failure(state, epoch)
            raise


def advance_registered_training_executor(
    executor: RegisteredTrainingExecutor,
    barrier: RegisteredHistoryBarrier,
) -> None:
    """Retire one evaluated non-final handoff and enable its next epoch."""

    # Runtime imports are deliberately local: history imports this executor to
    # verify the opaque handoff evidence it owns.
    from falsewake.experiment_002_registered_history import RegisteredHistoryBarrier

    if type(executor) is not RegisteredTrainingExecutor:
        raise TypeError("executor must be a RegisteredTrainingExecutor")
    if type(barrier) is not RegisteredHistoryBarrier:
        raise TypeError("barrier must be a RegisteredHistoryBarrier")
    caller_token = object()
    admission: _AdvanceAdmission | None = None
    try:
        admission = _begin_advance_admission(executor, barrier, caller_token)
        _require_advance_admission(admission, executor=executor, barrier=barrier)
    except _BoundaryAdmissionConflict:
        raise
    except BaseException as admission_error:
        if admission is None:
            admission = _recover_advance_admission(
                executor,
                barrier,
                caller_token,
            )
        if admission is None and isinstance(
            admission_error,
            _BoundaryTruthCorruption,
        ):
            admission = _recover_retained_advance_admission(executor, barrier)
        if admission is None or not admission.leader:
            raise
        try:
            _terminal_fail_registered_continuation(
                executor,
                handoff=None,
                reservation=None,
            )
        finally:
            _publish_advance_admission(
                admission,
                admission_error,
                caller_token=caller_token,
                handoff=None,
                reservation=None,
                history_attempted=False,
            )
        raise
    assert admission is not None
    if not admission.leader:
        admission.completed.wait()
        if len(admission.outcome) != 1:
            raise Experiment002RegisteredExecutorError(
                "the leading continuation did not publish an outcome"
            )
        leader_outcome = admission.outcome[0]
        if leader_outcome is None:
            return
        raise Experiment002RegisteredExecutorError(
            "the concurrent continuation leader failed terminally"
        ) from leader_outcome

    handoff: RegisteredExecutorEpochHandoff | None = None
    reservation: _ContinuationReservation | None = None
    history_attempted = False
    outcome: BaseException | None = None
    try:
        issued_snapshot = _preflight_registered_history_barrier(executor, barrier)
        handoff = issued_snapshot.evaluated_handoff
        reservation = _reserve_registered_continuation(
            executor,
            handoff,
            barrier,
            issued_snapshot,
        )
        history_attempted = True
        consumed_snapshot = _consume_registered_continuation_history(
            reservation,
        )
        _require_same_history_barrier_snapshot(
            issued_snapshot,
            consumed_snapshot,
            expected_phase="CONSUMED",
        )
        _mark_registered_continuation_consumed(reservation)
        _commit_registered_nonfinal_continuation(
            executor,
            handoff,
            barrier,
            reservation,
            issued_snapshot,
            consumed_snapshot,
        )
        _registered_epoch_handoff_snapshot(handoff)
    except _FinalBarrierRequiresCompletion as error:
        outcome = error
        raise
    except BaseException as error:
        final_error = error
        try:
            _terminal_fail_registered_continuation(
                executor,
                handoff=handoff,
                reservation=reservation,
            )
        except BaseException as terminal_error:
            final_error = terminal_error
        if history_attempted:
            assert reservation is not None
            try:
                _abort_consumed_registered_history(
                    reservation.registration,
                    executor,
                    barrier,
                )
            except BaseException as abort_error:
                outcome = abort_error
                raise abort_error from final_error
        outcome = final_error
        if final_error is not error:
            raise final_error from error
        raise
    finally:
        _publish_advance_admission(
            admission,
            outcome,
            caller_token=caller_token,
            handoff=handoff,
            reservation=reservation,
            history_attempted=history_attempted,
        )


def complete_registered_training_executor(
    executor: RegisteredTrainingExecutor,
    final_barrier: RegisteredHistoryBarrier,
) -> RegisteredCompletedTrainingHistory:
    """Complete the exact epoch-29 boundary and return its canonical history."""

    from falsewake.experiment_002_registered_history import (
        RegisteredCompletedTrainingHistory,
        RegisteredHistoryBarrier,
    )

    if type(executor) is not RegisteredTrainingExecutor:
        raise TypeError("executor must be a RegisteredTrainingExecutor")
    if type(final_barrier) is not RegisteredHistoryBarrier:
        raise TypeError("final_barrier must be a RegisteredHistoryBarrier")

    caller_token = object()
    admission: _CompletionAdmission | None = None
    try:
        admission = _begin_completion_admission(
            executor,
            final_barrier,
            caller_token,
        )
        _require_completion_admission(
            admission,
            executor=executor,
            barrier=final_barrier,
        )
    except _BoundaryAdmissionConflict:
        raise
    except BaseException as admission_error:
        if admission is None:
            admission = _recover_completion_admission(
                executor,
                final_barrier,
                caller_token,
            )
        if admission is None and isinstance(
            admission_error,
            _BoundaryTruthCorruption,
        ):
            admission = _recover_retained_completion_admission(
                executor,
                final_barrier,
            )
        cleanup_error: BaseException = admission_error
        if admission is not None and admission.leader:
            local_terminal = False
            try:
                _ensure_terminal_fail_registered_finalization(
                    executor,
                    handoff=None,
                    reservation=None,
                )
                local_terminal = True
            except BaseException as terminal_error:
                cleanup_error = terminal_error
            if local_terminal:
                try:
                    _ensure_attempted_registered_history_completion_aborted(
                        executor,
                        final_barrier,
                        None,
                    )
                except BaseException as abort_error:
                    cleanup_error = abort_error
            _publish_completion_admission(
                admission,
                None,
                cleanup_error,
                executor=executor,
                final_barrier=final_barrier,
                caller_token=caller_token,
            )
        if cleanup_error is not admission_error:
            raise cleanup_error from admission_error
        raise

    assert admission is not None
    if not admission.leader:
        admission.completed.wait()
        if len(admission.outcome) != 1:
            raise Experiment002RegisteredExecutorError(
                "the leading completion did not publish an outcome"
            )
        leader_outcome = admission.outcome[0]
        if (
            type(leader_outcome) is _CompletionOutcome
            and leader_outcome.error is None
            and type(leader_outcome.result) is RegisteredCompletedTrainingHistory
        ):
            return _reverify_retained_registered_completion(
                executor,
                final_barrier,
                leader_outcome.result,
            )
        error = (
            leader_outcome.error if type(leader_outcome) is _CompletionOutcome else None
        )
        raise Experiment002RegisteredExecutorError(
            "the concurrent completion leader failed terminally"
        ) from error

    handoff: RegisteredExecutorEpochHandoff | None = None
    reservation: _FinalizationReservation | None = None
    complete_trace: CompleteUpdateTraceEvidence | None = None
    completed_history: RegisteredCompletedTrainingHistory | None = None
    outcome_error: BaseException | None = None
    try:
        issued_snapshot = _preflight_registered_final_history_barrier(
            executor,
            final_barrier,
        )
        handoff = issued_snapshot.evaluated_handoff
        reservation = _reserve_registered_finalization(
            executor,
            handoff,
            final_barrier,
            issued_snapshot,
        )
        complete_trace = _finish_registered_finalization_trace(reservation)
        _mark_registered_finalization_trace_complete(
            reservation,
            complete_trace,
        )
        completed_history = _complete_registered_finalization_history(
            reservation,
            complete_trace,
        )
        _mark_registered_finalization_history_complete(
            reservation,
            complete_trace,
            completed_history,
        )
        _commit_registered_finalization(
            executor,
            handoff,
            final_barrier,
            reservation,
            complete_trace,
            completed_history,
        )
        _verify_registered_finalization(
            executor,
            reservation,
            complete_trace,
            completed_history,
        )
    except BaseException as error:
        final_error: BaseException = error
        local_terminal = False
        try:
            _ensure_terminal_fail_registered_finalization(
                executor,
                handoff=handoff,
                reservation=reservation,
            )
            local_terminal = True
        except BaseException as terminal_error:
            final_error = terminal_error
        if local_terminal:
            try:
                _ensure_attempted_registered_history_completion_aborted(
                    executor,
                    final_barrier,
                    completed_history,
                )
            except BaseException as abort_error:
                final_error = abort_error
        outcome_error = final_error
        if final_error is not error:
            raise final_error from error
        raise
    finally:
        _publish_completion_admission(
            admission,
            completed_history if outcome_error is None else None,
            outcome_error,
            executor=executor,
            final_barrier=final_barrier,
            caller_token=caller_token,
        )

    assert completed_history is not None
    return completed_history


def _reverify_retained_registered_completion(
    executor: RegisteredTrainingExecutor,
    final_barrier: RegisteredHistoryBarrier,
    completed_history: RegisteredCompletedTrainingHistory,
) -> RegisteredCompletedTrainingHistory:
    """Revalidate a concurrent or late exact completion before returning it."""

    reservation: _FinalizationReservation | None = None
    try:
        state = _issued_executor_state(executor)
        with _trusted_executor_lock(executor, state):
            _validate_executor_or_fail(executor, state)
            reservation = state.finalization_reservation
            complete_trace = state.complete_update_trace
            if (
                state.phase != "COMPLETE"
                or type(reservation) is not _FinalizationReservation
                or type(complete_trace) is not CompleteUpdateTraceEvidence
                or state.final_barrier is not final_barrier
                or state.completed_history is not completed_history
            ):
                raise Experiment002RegisteredExecutorError(
                    "retained completion differs from executor authority"
                )
        _verify_registered_finalization(
            executor,
            reservation,
            complete_trace,
            completed_history,
        )
    except BaseException as error:
        final_error: BaseException = error
        local_terminal = False
        try:
            _ensure_terminal_fail_registered_finalization(
                executor,
                handoff=reservation.handoff if reservation is not None else None,
                reservation=reservation,
            )
            local_terminal = True
        except BaseException as terminal_error:
            final_error = terminal_error
        if local_terminal:
            try:
                _ensure_attempted_registered_history_completion_aborted(
                    executor,
                    final_barrier,
                    completed_history,
                )
            except BaseException as abort_error:
                final_error = abort_error
        if final_error is not error:
            raise final_error from error
        raise
    return completed_history


def _begin_advance_admission(
    executor: RegisteredTrainingExecutor,
    barrier: RegisteredHistoryBarrier,
    caller_token: object | None = None,
    _truth_begin: Callable[
        [RegisteredTrainingExecutor, RegisteredHistoryBarrier, object],
        _AdvanceAdmission,
    ] = _AUTHORITY_TRUTH.begin_advance,
) -> _AdvanceAdmission:
    if caller_token is None:
        caller_token = object()
    return _truth_begin(executor, barrier, caller_token)


def _recover_advance_admission(
    executor: RegisteredTrainingExecutor,
    barrier: RegisteredHistoryBarrier,
    caller_token: object,
    _truth_recover: Callable[
        [RegisteredTrainingExecutor, RegisteredHistoryBarrier, object],
        _AdvanceAdmission | None,
    ] = _AUTHORITY_TRUTH.recover_advance,
) -> _AdvanceAdmission | None:
    return _truth_recover(executor, barrier, caller_token)


def _recover_retained_advance_admission(
    executor: RegisteredTrainingExecutor,
    barrier: RegisteredHistoryBarrier,
    _truth_recover: Callable[
        [RegisteredTrainingExecutor, RegisteredHistoryBarrier],
        _AdvanceAdmission | None,
    ] = _AUTHORITY_TRUTH.recover_retained_advance,
) -> _AdvanceAdmission | None:
    return _truth_recover(executor, barrier)


def _finish_advance_admission(
    admission: _AdvanceAdmission,
    outcome: BaseException | None,
    _truth_finish: Callable[[_AdvanceAdmission, BaseException | None], None] = (
        _AUTHORITY_TRUTH.finish_advance
    ),
) -> None:
    _truth_finish(admission, outcome)


def _recover_finish_advance_admission(
    admission: _AdvanceAdmission,
    outcome: BaseException | None,
    _truth_finish: Callable[[_AdvanceAdmission, BaseException | None], None] = (
        _AUTHORITY_TRUTH.finish_advance
    ),
) -> None:
    _truth_finish(admission, outcome)


def _publish_advance_admission(
    admission: _AdvanceAdmission,
    outcome: BaseException | None,
    *,
    caller_token: object,
    handoff: RegisteredExecutorEpochHandoff | None,
    reservation: _ContinuationReservation | None,
    history_attempted: bool,
) -> None:
    try:
        _finish_advance_admission(admission, outcome)
    except BaseException as publication_error:
        recovered = _recover_advance_admission(
            admission.executor,
            admission.barrier,
            caller_token,
        )
        target = recovered if recovered is not None else admission
        if len(target.outcome) == 1 and target.outcome[0] is outcome:
            try:
                _recover_finish_advance_admission(target, outcome)
            finally:
                target.completed.set()
                admission.completed.set()
            raise
        if not isinstance(publication_error, Exception):
            try:
                _recover_finish_advance_admission(target, outcome)
            finally:
                target.completed.set()
                admission.completed.set()
            raise
        final_error: BaseException = publication_error
        try:
            _terminal_fail_registered_continuation(
                admission.executor,
                handoff=handoff,
                reservation=reservation,
            )
        except BaseException as terminal_error:
            final_error = terminal_error
        if history_attempted and reservation is not None:
            try:
                _abort_consumed_registered_history(
                    reservation.registration,
                    admission.executor,
                    admission.barrier,
                )
            except BaseException as abort_error:
                final_error = abort_error
        try:
            _recover_finish_advance_admission(target, final_error)
        except BaseException as recovery_error:
            final_error = recovery_error
            target.outcome[:] = [final_error]
        finally:
            target.completed.set()
            admission.completed.set()
        if final_error is not publication_error:
            raise final_error from publication_error
        raise


def _require_advance_admission(
    admission: _AdvanceAdmission,
    *,
    executor: RegisteredTrainingExecutor,
    barrier: RegisteredHistoryBarrier,
) -> None:
    if (
        type(admission) is not _AdvanceAdmission
        or type(admission.token) is not object
        or admission.executor is not executor
        or admission.barrier is not barrier
        or type(admission.leader) is not bool
        or type(admission.completed) is not _EVENT_TYPE
        or type(admission.outcome) is not list
        or (admission.leader and bool(admission.outcome))
        or (not admission.leader and len(admission.outcome) > 1)
    ):
        raise Experiment002RegisteredExecutorError(
            "continuation admission authority changed"
        )


def _begin_completion_admission(
    executor: RegisteredTrainingExecutor,
    barrier: RegisteredHistoryBarrier,
    caller_token: object | None = None,
    _truth_begin: Callable[
        [RegisteredTrainingExecutor, RegisteredHistoryBarrier, object],
        _CompletionAdmission,
    ] = _AUTHORITY_TRUTH.begin_completion,
) -> _CompletionAdmission:
    if caller_token is None:
        caller_token = object()
    return _truth_begin(executor, barrier, caller_token)


def _recover_completion_admission(
    executor: RegisteredTrainingExecutor,
    barrier: RegisteredHistoryBarrier,
    caller_token: object,
    _truth_recover: Callable[
        [RegisteredTrainingExecutor, RegisteredHistoryBarrier, object],
        _CompletionAdmission | None,
    ] = _AUTHORITY_TRUTH.recover_completion,
) -> _CompletionAdmission | None:
    return _truth_recover(executor, barrier, caller_token)


def _recover_retained_completion_admission(
    executor: RegisteredTrainingExecutor,
    barrier: RegisteredHistoryBarrier,
    _truth_recover: Callable[
        [RegisteredTrainingExecutor, RegisteredHistoryBarrier],
        _CompletionAdmission | None,
    ] = _AUTHORITY_TRUTH.recover_retained_completion,
) -> _CompletionAdmission | None:
    return _truth_recover(executor, barrier)


def _finish_completion_admission(
    admission: _CompletionAdmission,
    result: RegisteredCompletedTrainingHistory | None,
    error: BaseException | None,
    _truth_finish: Callable[
        [
            _CompletionAdmission,
            RegisteredCompletedTrainingHistory | None,
            BaseException | None,
        ],
        None,
    ] = _AUTHORITY_TRUTH.finish_completion,
) -> None:
    _truth_finish(admission, result, error)


def _recover_finish_completion_admission(
    admission: _CompletionAdmission,
    result: RegisteredCompletedTrainingHistory | None,
    error: BaseException | None,
    _truth_finish: Callable[
        [
            _CompletionAdmission,
            RegisteredCompletedTrainingHistory | None,
            BaseException | None,
        ],
        None,
    ] = _AUTHORITY_TRUTH.finish_completion,
) -> None:
    _truth_finish(admission, result, error)


def _publish_completion_admission(
    admission: _CompletionAdmission,
    result: RegisteredCompletedTrainingHistory | None,
    error: BaseException | None,
    *,
    executor: RegisteredTrainingExecutor,
    final_barrier: RegisteredHistoryBarrier,
    caller_token: object,
) -> None:
    try:
        _finish_completion_admission(admission, result, error)
    except BaseException as publication_error:
        recovered = _recover_completion_admission(
            executor,
            final_barrier,
            caller_token,
        )
        target = recovered if recovered is not None else admission
        if (
            len(target.outcome) == 1
            and type(target.outcome[0]) is _CompletionOutcome
            and target.outcome[0].result is result
            and target.outcome[0].error is error
        ):
            try:
                _recover_finish_completion_admission(target, result, error)
            finally:
                target.completed.set()
                admission.completed.set()
            raise
        if not isinstance(publication_error, Exception):
            try:
                _recover_finish_completion_admission(target, result, error)
            finally:
                target.completed.set()
                admission.completed.set()
            raise
        final_error: BaseException = publication_error
        local_terminal = False
        try:
            _ensure_terminal_fail_registered_finalization(
                executor,
                handoff=None,
                reservation=None,
            )
            local_terminal = True
        except BaseException as terminal_error:
            final_error = terminal_error
        if local_terminal:
            try:
                _ensure_attempted_registered_history_completion_aborted(
                    executor,
                    final_barrier,
                    result,
                )
            except BaseException as abort_error:
                final_error = abort_error
        recovered = _recover_completion_admission(
            executor,
            final_barrier,
            caller_token,
        )
        target = recovered if recovered is not None else admission
        try:
            _recover_finish_completion_admission(
                target,
                None,
                final_error,
            )
        except BaseException as recovery_error:
            final_error = recovery_error
            target.outcome[:] = [_CompletionOutcome(None, final_error)]
        finally:
            target.completed.set()
            admission.completed.set()
        if final_error is not publication_error:
            raise final_error from publication_error
        raise


def _require_completion_admission(
    admission: _CompletionAdmission,
    *,
    executor: RegisteredTrainingExecutor,
    barrier: RegisteredHistoryBarrier,
) -> None:
    if (
        type(admission) is not _CompletionAdmission
        or type(admission.token) is not object
        or admission.executor is not executor
        or admission.barrier is not barrier
        or type(admission.leader) is not bool
        or type(admission.completed) is not _EVENT_TYPE
        or type(admission.outcome) is not list
        or (admission.leader and bool(admission.outcome))
        or (not admission.leader and len(admission.outcome) > 1)
    ):
        raise Experiment002RegisteredExecutorError(
            "completion admission authority changed"
        )


def _preflight_registered_history_barrier(
    executor: RegisteredTrainingExecutor,
    barrier: RegisteredHistoryBarrier,
) -> _RegisteredHistoryBarrierSnapshot:
    """Read history only while no handoff or executor lock is held."""

    from falsewake.experiment_002_registered_evaluator import RegisteredEvaluatedEpoch
    from falsewake.experiment_002_registered_history import (
        _registered_history_barrier_snapshot,
        _RegisteredHistoryBarrierSnapshot,
        verify_registered_history_barrier,
    )

    verify_registered_history_barrier(barrier)
    snapshot = _registered_history_barrier_snapshot(barrier)
    if type(snapshot) is not _RegisteredHistoryBarrierSnapshot:
        raise Experiment002RegisteredExecutorError(
            "history returned an invalid barrier snapshot"
        )
    if (
        snapshot.executor is not executor
        or type(snapshot.evaluated_epoch) is not RegisteredEvaluatedEpoch
        or snapshot.phase != "ISSUED"
    ):
        raise Experiment002RegisteredExecutorError(
            "history barrier is not an issued non-final executor boundary"
        )
    if (
        snapshot.accepted_epoch == _REGISTERED_EPOCH_COUNT - 1
        and snapshot.next_zero_based_epoch is None
    ):
        raise _FinalBarrierRequiresCompletion(
            "epoch-29 is not a non-final boundary and belongs to completion"
        )
    if (
        snapshot.next_zero_based_epoch is None
        or not 0 <= snapshot.accepted_epoch < _REGISTERED_EPOCH_COUNT - 1
        or snapshot.next_zero_based_epoch != snapshot.accepted_epoch + 1
    ):
        raise Experiment002RegisteredExecutorError(
            "history barrier is not an issued non-final executor boundary"
        )
    _require_lower_sha256(
        snapshot.evaluated_authority_sha256,
        "evaluated_authority_sha256",
    )
    return snapshot


def _preflight_registered_final_history_barrier(
    executor: RegisteredTrainingExecutor,
    final_barrier: RegisteredHistoryBarrier,
) -> _RegisteredHistoryBarrierSnapshot:
    """Verify the final history boundary with every executor lock released."""

    from falsewake.experiment_002_registered_evaluator import RegisteredEvaluatedEpoch
    from falsewake.experiment_002_registered_history import (
        _registered_history_barrier_snapshot,
        _RegisteredHistoryBarrierSnapshot,
        verify_registered_history_barrier,
    )

    verify_registered_history_barrier(final_barrier)
    snapshot = _registered_history_barrier_snapshot(final_barrier)
    if type(snapshot) is not _RegisteredHistoryBarrierSnapshot:
        raise Experiment002RegisteredExecutorError(
            "history returned an invalid final barrier snapshot"
        )
    if (
        snapshot.executor is not executor
        or type(snapshot.evaluated_epoch) is not RegisteredEvaluatedEpoch
        or type(snapshot.evaluated_handoff) is not RegisteredExecutorEpochHandoff
        or snapshot.phase != "ISSUED"
        or snapshot.accepted_epoch != _REGISTERED_EPOCH_COUNT - 1
        or snapshot.next_zero_based_epoch is not None
    ):
        raise Experiment002RegisteredExecutorError(
            "history barrier is not the issued epoch-29 completion boundary"
        )
    _require_lower_sha256(
        snapshot.evaluated_authority_sha256,
        "evaluated_authority_sha256",
    )
    return snapshot


def _reserve_registered_continuation(
    executor: RegisteredTrainingExecutor,
    handoff: RegisteredExecutorEpochHandoff,
    barrier: RegisteredHistoryBarrier,
    snapshot: _RegisteredHistoryBarrierSnapshot,
    _truth_reserve: Callable[[_ContinuationReservation], None] = (
        _AUTHORITY_TRUTH.reserve_continuation
    ),
    _truth_retired: Callable[
        [RegisteredExecutorEpochHandoff], _RetiredHandoffBinding | None
    ] = _AUTHORITY_TRUTH.retired_handoff,
) -> _ContinuationReservation:
    """Reserve one continuation using only the executor-to-handoff lock order."""

    with _HANDOFFS_LOCK:
        handoff_state = _HANDOFFS.get(handoff)
        _validate_handoff(handoff, handoff_state)
        assert handoff_state is not None
        if _truth_retired(handoff) is not None or handoff in _RETIRED_HANDOFFS:
            raise Experiment002RegisteredExecutorError(
                "registered epoch handoff was already retired"
            )

    executor_state = _issued_executor_state(executor)
    with _trusted_executor_lock(executor, executor_state):
        _validate_executor_or_fail(executor, executor_state)
        if (
            snapshot.registration is not executor_state.registration
            or snapshot.registration is not handoff_state.registration
            or snapshot.validation_inputs is not executor_state.validation_inputs
            or snapshot.validation_inputs is not handoff_state.validation_inputs
            or snapshot.process_id != executor_state.process_id
            or snapshot.process_id != handoff_state.process_id
            or snapshot.process_id != os.getpid()
            or snapshot.seed != executor_state.seed
            or snapshot.seed != handoff_state.seed
            or snapshot.accepted_epoch != executor_state.zero_based_epoch
            or snapshot.accepted_epoch != handoff_state.zero_based_epoch
            or snapshot.evaluated_handoff is not handoff
            or executor_state.phase != "AWAITING_EVALUATION"
            or executor_state.active_handoff is not handoff
            or executor_state.active_epoch is not None
            or executor_state.active_transition is not None
            or executor_state.continuation_reservation is not None
            or not handoff_state.used
            or handoff_state.failed
            or executor_state.optimizer_generation
            != (snapshot.accepted_epoch + 1) * _UPDATES_PER_EPOCH
            or handoff_state.optimizer_generation != executor_state.optimizer_generation
            or handoff_state.runtime_digests != executor_state.expected
            or handoff_state.previous_history_barrier
            is not executor_state.previous_history_barrier
            or executor_state.runtime.model is not handoff_state.model
            or executor_state.runtime.model.training is not True
            or executor_state.trace.completed_epoch_count != snapshot.accepted_epoch + 1
            or executor_state.trace.next_global_update
            != executor_state.optimizer_generation
        ):
            raise Experiment002RegisteredExecutorError(
                "continuation identities differ from the evaluated boundary"
            )
        observed = _stable_handoff_runtime_digests(
            executor_state.runtime,
            expected_generation=executor_state.optimizer_generation,
            expected_training=True,
        )
        if observed != executor_state.expected:
            raise Experiment002RegisteredExecutorError(
                "runtime changed before continuation reservation"
            )
        reservation = _ContinuationReservation(
            token=object(),
            executor=executor,
            executor_ticket=executor_state.ticket,
            handoff=handoff,
            handoff_ticket=handoff_state.ticket,
            barrier=barrier,
            barrier_type=type(barrier),
            evaluated_epoch=snapshot.evaluated_epoch,
            evaluated_epoch_type=type(snapshot.evaluated_epoch),
            evaluated_authority_sha256=snapshot.evaluated_authority_sha256,
            registration=snapshot.registration,
            validation_inputs=snapshot.validation_inputs,
            process_id=snapshot.process_id,
            seed=snapshot.seed,
            zero_based_epoch=snapshot.accepted_epoch,
            optimizer_generation=executor_state.optimizer_generation,
            runtime_digests=replace(executor_state.expected),
            previous_history_barrier=executor_state.previous_history_barrier,
            handoff_token=handoff_state.handoff_token,
            one_shot_token=handoff_state.one_shot_token,
        )

    with _HANDOFFS_LOCK:
        if _HANDOFFS.get(handoff) is not handoff_state:
            raise Experiment002RegisteredExecutorError(
                "handoff state changed during continuation reservation"
            )
        _validate_handoff(handoff, handoff_state)
        _require_handoff_matches_reservation(handoff_state, reservation)
        _truth_reserve(reservation)
        _CONTINUATION_HISTORY.append(reservation)
        _CONTINUATION_PHASES[reservation] = "RESERVED"

    with _trusted_executor_lock(executor, executor_state):
        _validate_executor_or_fail(executor, executor_state)
        _require_executor_matches_reservation(
            executor_state,
            reservation,
            require_attached=False,
        )
        executor_state.continuation_reservation = reservation
        _publish_executor_lifecycle(executor, executor_state)
        _require_continuation_reservation(reservation, phase="RESERVED")
        _validate_executor_or_fail(executor, executor_state)

    with _HANDOFFS_LOCK:
        _validate_handoff(handoff, handoff_state)
        _require_handoff_matches_reservation(handoff_state, reservation)
    return reservation


def _require_handoff_matches_reservation(
    state: _HandoffState,
    reservation: _ContinuationReservation,
) -> None:
    if (
        state.executor is not reservation.executor
        or state.ticket is not reservation.handoff_ticket
        or state.registration is not reservation.registration
        or state.validation_inputs is not reservation.validation_inputs
        or state.process_id != reservation.process_id
        or state.seed != reservation.seed
        or state.zero_based_epoch != reservation.zero_based_epoch
        or state.optimizer_generation != reservation.optimizer_generation
        or state.runtime_digests != reservation.runtime_digests
        or state.previous_history_barrier is not reservation.previous_history_barrier
        or state.handoff_token is not reservation.handoff_token
        or state.one_shot_token is not reservation.one_shot_token
        or state.used is not True
        or state.failed is not False
    ):
        raise Experiment002RegisteredExecutorError(
            "handoff changed across continuation reservation"
        )


def _require_executor_matches_reservation(
    state: _RegisteredExecutorState,
    reservation: _ContinuationReservation,
    *,
    require_attached: bool,
) -> None:
    expected_reservation = reservation if require_attached else None
    if (
        state.ticket is not reservation.executor_ticket
        or state.registration is not reservation.registration
        or state.validation_inputs is not reservation.validation_inputs
        or state.process_id != reservation.process_id
        or state.process_id != os.getpid()
        or state.seed != reservation.seed
        or state.phase != "AWAITING_EVALUATION"
        or state.zero_based_epoch != reservation.zero_based_epoch
        or state.optimizer_generation != reservation.optimizer_generation
        or state.active_epoch is not None
        or state.active_transition is not None
        or state.active_handoff is not reservation.handoff
        or state.continuation_reservation is not expected_reservation
        or state.previous_history_barrier is not reservation.previous_history_barrier
        or state.expected != reservation.runtime_digests
        or state.runtime.expected != reservation.runtime_digests
        or state.runtime.optimizer_generation != reservation.optimizer_generation
        or state.runtime.model.training is not True
        or state.trace.completed_epoch_count != reservation.zero_based_epoch + 1
        or state.trace.next_global_update != reservation.optimizer_generation
    ):
        raise Experiment002RegisteredExecutorError(
            "executor changed across continuation reservation"
        )


def _consume_registered_continuation_history(
    reservation: _ContinuationReservation,
) -> _RegisteredHistoryBarrierSnapshot:
    """Consume history with every executor and handoff lock released."""

    from falsewake.experiment_002_registered_history import (
        _consume_registered_history_barrier,
    )

    snapshot = _consume_registered_history_barrier(
        reservation.registration,
        reservation.executor,
        reservation.barrier,
    )
    return snapshot


def _mark_registered_continuation_consumed(
    reservation: _ContinuationReservation,
    _truth_transition: Callable[
        [_ContinuationReservation, _ContinuationPhase, _ContinuationPhase], None
    ] = _AUTHORITY_TRUTH.transition_continuation,
) -> None:
    executor_state = _issued_executor_state(reservation.executor)
    with _trusted_executor_lock(reservation.executor, executor_state):
        _validate_executor_or_fail(reservation.executor, executor_state)
        if executor_state.continuation_reservation is not reservation:
            raise Experiment002RegisteredExecutorError(
                "executor lost its continuation before consumed publication"
            )
        with _HANDOFFS_LOCK:
            _require_continuation_reservation(reservation, phase="RESERVED")
            _truth_transition(reservation, "RESERVED", "CONSUMED")
            _CONTINUATION_PHASES[reservation] = "CONSUMED"
            _require_continuation_reservation(reservation, phase="CONSUMED")
        _validate_executor_or_fail(reservation.executor, executor_state)


def _commit_registered_nonfinal_continuation(
    executor: RegisteredTrainingExecutor,
    handoff: RegisteredExecutorEpochHandoff,
    barrier: RegisteredHistoryBarrier,
    reservation: _ContinuationReservation,
    issued_snapshot: _RegisteredHistoryBarrierSnapshot,
    consumed_snapshot: _RegisteredHistoryBarrierSnapshot,
    _truth_commit: Callable[
        [_ContinuationReservation, _RetiredHandoffBinding], None
    ] = _AUTHORITY_TRUTH.commit_retirement,
) -> None:
    """Commit using only the executor-to-handoff lock order when locks nest."""

    _require_same_history_barrier_snapshot(
        issued_snapshot,
        consumed_snapshot,
        expected_phase="CONSUMED",
    )
    with _HANDOFFS_LOCK:
        handoff_state = _HANDOFFS.get(handoff)
        _validate_handoff(handoff, handoff_state)
        assert handoff_state is not None
        _require_handoff_matches_reservation(handoff_state, reservation)
        binding = _retirement_from_reservation(reservation)

    executor_state = _issued_executor_state(executor)
    with _trusted_executor_lock(executor, executor_state):
        _validate_executor_or_fail(executor, executor_state)
        _require_continuation_reservation(reservation, phase="CONSUMED")
        _require_executor_matches_reservation(
            executor_state,
            reservation,
            require_attached=True,
        )
        observed = _stable_handoff_runtime_digests(
            executor_state.runtime,
            expected_generation=reservation.optimizer_generation,
            expected_training=True,
        )
        if observed != reservation.runtime_digests:
            raise Experiment002RegisteredExecutorError(
                "runtime changed after history consumed its barrier"
            )

    with _HANDOFFS_LOCK:
        _validate_handoff(handoff, handoff_state)
        _require_handoff_matches_reservation(handoff_state, reservation)
        _truth_commit(reservation, binding)
        _RETIRED_HANDOFFS[handoff] = binding
        _CONTINUATION_PHASES[reservation] = "COMMITTED"
        _require_retired_handoff_binding(handoff, handoff_state)

    with _trusted_executor_lock(executor, executor_state):
        _validate_executor_or_fail(executor, executor_state)
        _require_continuation_reservation(reservation, phase="COMMITTED")
        _require_executor_matches_reservation(
            executor_state,
            reservation,
            require_attached=True,
        )
        executor_state.active_handoff = None
        executor_state.previous_history_barrier = barrier
        executor_state.zero_based_epoch = reservation.zero_based_epoch + 1
        executor_state.phase = "READY_TO_TRAIN"
        executor_state.continuation_reservation = None
        _publish_executor_lifecycle(executor, executor_state)
        after = _stable_handoff_runtime_digests(
            executor_state.runtime,
            expected_generation=reservation.optimizer_generation,
            expected_training=True,
        )
        if (
            after != reservation.runtime_digests
            or executor_state.expected != reservation.runtime_digests
            or executor_state.runtime.expected != reservation.runtime_digests
            or executor_state.optimizer_generation != reservation.optimizer_generation
        ):
            raise Experiment002RegisteredExecutorError(
                "non-final continuation changed model, optimizer, or RNG"
            )
        _validate_executor_or_fail(executor, executor_state)

    with _HANDOFFS_LOCK:
        _require_retired_handoff_binding(handoff, handoff_state)


def _reserve_registered_finalization(
    executor: RegisteredTrainingExecutor,
    handoff: RegisteredExecutorEpochHandoff,
    final_barrier: RegisteredHistoryBarrier,
    snapshot: _RegisteredHistoryBarrierSnapshot,
    _truth_reserve: Callable[[_FinalizationReservation], None] = (
        _AUTHORITY_TRUTH.reserve_finalization
    ),
    _truth_retired: Callable[
        [RegisteredExecutorEpochHandoff], _RetiredHandoffBinding | None
    ] = _AUTHORITY_TRUTH.retired_handoff,
    _truth_finalized: Callable[
        [RegisteredExecutorEpochHandoff], _FinalizedHandoffBinding | None
    ] = _AUTHORITY_TRUTH.finalized_handoff,
) -> _FinalizationReservation:
    """Reserve the exact final boundary without nesting H before E."""

    executor_state = _issued_executor_state(executor)
    epoch_traces = _capture_registered_final_trace_epochs(executor_state.trace)

    with _HANDOFFS_LOCK:
        handoff_state = _HANDOFFS.get(handoff)
        _validate_handoff(handoff, handoff_state)
        assert handoff_state is not None
        if (
            _truth_retired(handoff) is not None
            or handoff in _RETIRED_HANDOFFS
            or _truth_finalized(handoff) is not None
            or handoff in _FINALIZED_HANDOFFS
        ):
            raise Experiment002RegisteredExecutorError(
                "epoch-29 handoff was already archived"
            )

    with _trusted_executor_lock(executor, executor_state):
        _validate_executor_or_fail(executor, executor_state)
        if (
            snapshot.registration is not executor_state.registration
            or snapshot.registration is not handoff_state.registration
            or snapshot.history is None
            or snapshot.validation_inputs is not executor_state.validation_inputs
            or snapshot.validation_inputs is not handoff_state.validation_inputs
            or snapshot.process_id != executor_state.process_id
            or snapshot.process_id != handoff_state.process_id
            or snapshot.process_id != os.getpid()
            or snapshot.seed != executor_state.seed
            or snapshot.seed != handoff_state.seed
            or snapshot.accepted_epoch != _REGISTERED_EPOCH_COUNT - 1
            or snapshot.next_zero_based_epoch is not None
            or snapshot.evaluated_handoff is not handoff
            or executor_state.phase != "AWAITING_EVALUATION"
            or executor_state.zero_based_epoch != _REGISTERED_EPOCH_COUNT - 1
            or executor_state.optimizer_generation
            != _REGISTERED_EPOCH_COUNT * _UPDATES_PER_EPOCH
            or executor_state.active_handoff is not handoff
            or executor_state.active_epoch is not None
            or executor_state.active_transition is not None
            or executor_state.continuation_reservation is not None
            or executor_state.finalization_reservation is not None
            or executor_state.final_barrier is not None
            or executor_state.complete_update_trace is not None
            or executor_state.completed_history is not None
            or not handoff_state.used
            or handoff_state.failed
            or handoff_state.zero_based_epoch != _REGISTERED_EPOCH_COUNT - 1
            or handoff_state.optimizer_generation
            != _REGISTERED_EPOCH_COUNT * _UPDATES_PER_EPOCH
            or handoff_state.runtime_digests != executor_state.expected
            or handoff_state.previous_history_barrier
            is not executor_state.previous_history_barrier
            or executor_state.previous_history_barrier is None
            or executor_state.runtime.model is not handoff_state.model
            or executor_state.runtime.model.training is not False
        ):
            raise Experiment002RegisteredExecutorError(
                "executor is not at the exact registered epoch-29 boundary"
            )
        observed = _stable_handoff_runtime_digests(
            executor_state.runtime,
            expected_generation=executor_state.optimizer_generation,
            expected_training=False,
        )
        if observed != executor_state.expected:
            raise Experiment002RegisteredExecutorError(
                "runtime changed before registered finalization"
            )
        if epoch_traces[-1] is not handoff_state.epoch_trace:
            raise Experiment002RegisteredExecutorError(
                "epoch-29 handoff is not the final trace capability"
            )
        reservation = _FinalizationReservation(
            token=object(),
            executor=executor,
            executor_ticket=executor_state.ticket,
            handoff=handoff,
            handoff_ticket=handoff_state.ticket,
            final_barrier=final_barrier,
            barrier_type=type(final_barrier),
            history=snapshot.history,
            history_type=type(snapshot.history),
            evaluated_epoch=snapshot.evaluated_epoch,
            evaluated_epoch_type=type(snapshot.evaluated_epoch),
            evaluated_authority_sha256=snapshot.evaluated_authority_sha256,
            registration=snapshot.registration,
            validation_inputs=snapshot.validation_inputs,
            process_id=snapshot.process_id,
            seed=snapshot.seed,
            zero_based_epoch=snapshot.accepted_epoch,
            optimizer_generation=executor_state.optimizer_generation,
            runtime_digests=replace(executor_state.expected),
            previous_history_barrier=executor_state.previous_history_barrier,
            handoff_token=handoff_state.handoff_token,
            one_shot_token=handoff_state.one_shot_token,
            trace=executor_state.trace,
            epoch_traces=epoch_traces,
        )

    with _HANDOFFS_LOCK:
        if _HANDOFFS.get(handoff) is not handoff_state:
            raise Experiment002RegisteredExecutorError(
                "handoff state changed during finalization reservation"
            )
        _validate_handoff(handoff, handoff_state)
        _require_handoff_matches_finalization(handoff_state, reservation)
        _truth_reserve(reservation)
        _FINALIZATION_HISTORY.append(reservation)
        _FINALIZATION_PHASES[reservation] = "RESERVED"
        _require_finalization_reservation(reservation, phase="RESERVED")

    with _trusted_executor_lock(executor, executor_state):
        _validate_executor_or_fail(executor, executor_state)
        _require_executor_matches_finalization(
            executor_state,
            reservation,
            require_attached=False,
        )
        executor_state.finalization_reservation = reservation
        _publish_executor_lifecycle(executor, executor_state)
        _require_finalization_reservation(reservation, phase="RESERVED")
        _validate_executor_or_fail(executor, executor_state)

    with _HANDOFFS_LOCK:
        _validate_handoff(handoff, handoff_state)
        _require_handoff_matches_finalization(handoff_state, reservation)
    return reservation


def _capture_registered_final_trace_epochs(
    trace: UpdateTraceAccumulator,
) -> tuple[EpochUpdateTraceEvidence, ...]:
    if type(trace) is not UpdateTraceAccumulator:
        raise Experiment002RegisteredExecutorError(
            "final trace accumulator has an invalid type"
        )
    with trace._lock:
        if (
            type(trace._records) is not list
            or len(trace._records) != _REGISTERED_EPOCH_COUNT * _UPDATES_PER_EPOCH
            or type(trace._epochs) is not list
            or len(trace._epochs) != _REGISTERED_EPOCH_COUNT
            or trace._complete is not None
            or type(trace._failed) is not bool
            or trace._failed
        ):
            raise Experiment002RegisteredExecutorError(
                "final trace accumulator is not complete and live"
            )
        epochs = tuple(trace._epochs)
    for expected_epoch, epoch_trace in enumerate(epochs):
        if type(epoch_trace) is not EpochUpdateTraceEvidence:
            raise Experiment002RegisteredExecutorError(
                "final trace contains an invalid epoch capability"
            )
        verify_registered_epoch_update_trace(epoch_trace)
        if (
            epoch_trace.seed != trace.seed
            or epoch_trace.zero_based_epoch != expected_epoch
            or epoch_trace.update_count != _UPDATES_PER_EPOCH
        ):
            raise Experiment002RegisteredExecutorError(
                "final trace epoch identities are not exact and ordered"
            )
    return epochs


def _require_handoff_matches_finalization(
    state: _HandoffState,
    reservation: _FinalizationReservation,
) -> None:
    if (
        state.executor is not reservation.executor
        or state.ticket is not reservation.handoff_ticket
        or state.registration is not reservation.registration
        or state.validation_inputs is not reservation.validation_inputs
        or state.process_id != reservation.process_id
        or state.seed != reservation.seed
        or state.zero_based_epoch != reservation.zero_based_epoch
        or state.optimizer_generation != reservation.optimizer_generation
        or state.runtime_digests != reservation.runtime_digests
        or state.previous_history_barrier is not reservation.previous_history_barrier
        or state.handoff_token is not reservation.handoff_token
        or state.one_shot_token is not reservation.one_shot_token
        or state.epoch_trace is not reservation.epoch_traces[-1]
        or state.used is not True
        or state.failed is not False
    ):
        raise Experiment002RegisteredExecutorError(
            "handoff changed across finalization reservation"
        )


def _require_executor_matches_finalization(
    state: _RegisteredExecutorState,
    reservation: _FinalizationReservation,
    *,
    require_attached: bool,
) -> None:
    expected_reservation = reservation if require_attached else None
    if (
        state.ticket is not reservation.executor_ticket
        or state.registration is not reservation.registration
        or state.validation_inputs is not reservation.validation_inputs
        or state.process_id != reservation.process_id
        or state.process_id != os.getpid()
        or state.seed != reservation.seed
        or state.phase != "AWAITING_EVALUATION"
        or state.zero_based_epoch != reservation.zero_based_epoch
        or state.optimizer_generation != reservation.optimizer_generation
        or state.active_epoch is not None
        or state.active_transition is not None
        or state.active_handoff is not reservation.handoff
        or state.continuation_reservation is not None
        or state.finalization_reservation is not expected_reservation
        or state.previous_history_barrier is not reservation.previous_history_barrier
        or state.final_barrier is not None
        or state.complete_update_trace is not None
        or state.completed_history is not None
        or state.trace is not reservation.trace
        or state.expected != reservation.runtime_digests
        or state.runtime.expected != reservation.runtime_digests
        or state.runtime.optimizer_generation != reservation.optimizer_generation
        or state.runtime.model.training is not False
    ):
        raise Experiment002RegisteredExecutorError(
            "executor changed across finalization reservation"
        )


def _finish_registered_finalization_trace(
    reservation: _FinalizationReservation,
) -> CompleteUpdateTraceEvidence:
    """Finish the exact trace while every executor and handoff lock is free."""

    evidence = reservation.trace.finish()
    _require_complete_trace_matches_finalization(reservation, evidence)
    return evidence


def _require_complete_trace_matches_finalization(
    reservation: _FinalizationReservation,
    evidence: CompleteUpdateTraceEvidence,
) -> None:
    if type(evidence) is not CompleteUpdateTraceEvidence:
        raise Experiment002RegisteredExecutorError(
            "trace finish returned an invalid complete capability"
        )
    verify_registered_complete_update_trace(evidence)
    epochs = evidence.epochs
    if (
        evidence.seed != reservation.seed
        or evidence.update_count != reservation.optimizer_generation
        or type(epochs) is not tuple
        or len(epochs) != _REGISTERED_EPOCH_COUNT
        or any(
            observed is not expected
            for observed, expected in zip(
                epochs,
                reservation.epoch_traces,
                strict=True,
            )
        )
    ):
        raise Experiment002RegisteredExecutorError(
            "complete trace differs from the reserved epoch identities"
        )
    verify_registered_complete_update_trace(evidence)


def _mark_registered_finalization_trace_complete(
    reservation: _FinalizationReservation,
    complete_trace: CompleteUpdateTraceEvidence,
    _truth_transition: Callable[
        [_FinalizationReservation, _FinalizationPhase, _FinalizationPhase], None
    ] = _AUTHORITY_TRUTH.transition_finalization,
) -> None:
    _require_complete_trace_matches_finalization(reservation, complete_trace)
    executor_state = _issued_executor_state(reservation.executor)
    with _trusted_executor_lock(reservation.executor, executor_state):
        _validate_executor_or_fail(reservation.executor, executor_state)
        _require_executor_matches_finalization(
            executor_state,
            reservation,
            require_attached=True,
        )
        with _HANDOFFS_LOCK:
            _require_finalization_reservation(reservation, phase="RESERVED")
            _truth_transition(reservation, "RESERVED", "TRACE_COMPLETE")
            _FINALIZATION_PHASES[reservation] = "TRACE_COMPLETE"
            _require_finalization_reservation(
                reservation,
                phase="TRACE_COMPLETE",
            )
        _validate_executor_or_fail(reservation.executor, executor_state)


def _complete_registered_finalization_history(
    reservation: _FinalizationReservation,
    complete_trace: CompleteUpdateTraceEvidence,
) -> RegisteredCompletedTrainingHistory:
    """Complete history with every executor and handoff lock released."""

    from falsewake.experiment_002_registered_history import (
        RegisteredCompletedTrainingHistory,
        complete_registered_training_history,
    )

    result = complete_registered_training_history(
        reservation.history,
        reservation.final_barrier,
        complete_trace,
    )
    if type(result) is not RegisteredCompletedTrainingHistory:
        raise Experiment002RegisteredExecutorError(
            "history completion returned an invalid capability"
        )
    return result


def _mark_registered_finalization_history_complete(
    reservation: _FinalizationReservation,
    complete_trace: CompleteUpdateTraceEvidence,
    completed_history: RegisteredCompletedTrainingHistory,
    _truth_transition: Callable[
        [_FinalizationReservation, _FinalizationPhase, _FinalizationPhase], None
    ] = _AUTHORITY_TRUTH.transition_finalization,
) -> None:
    _require_complete_trace_matches_finalization(reservation, complete_trace)
    _require_completed_history_matches_finalization(
        reservation,
        complete_trace,
        completed_history,
    )
    executor_state = _issued_executor_state(reservation.executor)
    with _trusted_executor_lock(reservation.executor, executor_state):
        _validate_executor_or_fail(reservation.executor, executor_state)
        _require_executor_matches_finalization(
            executor_state,
            reservation,
            require_attached=True,
        )
        with _HANDOFFS_LOCK:
            _require_finalization_reservation(
                reservation,
                phase="TRACE_COMPLETE",
            )
            _truth_transition(
                reservation,
                "TRACE_COMPLETE",
                "HISTORY_COMPLETE",
            )
            _FINALIZATION_PHASES[reservation] = "HISTORY_COMPLETE"
            _require_finalization_reservation(
                reservation,
                phase="HISTORY_COMPLETE",
            )
        _validate_executor_or_fail(reservation.executor, executor_state)


def _require_completed_history_matches_finalization(
    reservation: _FinalizationReservation,
    complete_trace: CompleteUpdateTraceEvidence,
    completed_history: RegisteredCompletedTrainingHistory,
) -> None:
    from falsewake.experiment_002_registered_history import (
        RegisteredCompletedTrainingHistory,
        _registered_completed_history_snapshot,
        _RegisteredCompletedHistorySnapshot,
    )

    if type(completed_history) is not RegisteredCompletedTrainingHistory:
        raise Experiment002RegisteredExecutorError(
            "completed history has an invalid type"
        )
    snapshot = _registered_completed_history_snapshot(completed_history)
    if (
        type(snapshot) is not _RegisteredCompletedHistorySnapshot
        or snapshot.registration is not reservation.registration
        or snapshot.history is not reservation.history
        or snapshot.executor is not reservation.executor
        or snapshot.validation_inputs is not reservation.validation_inputs
        or snapshot.complete_update_trace is not complete_trace
        or snapshot.process_id != reservation.process_id
        or snapshot.seed != reservation.seed
        or snapshot.epoch_count != _REGISTERED_EPOCH_COUNT
        or snapshot.evaluated_epochs[-1] is not reservation.evaluated_epoch
    ):
        raise Experiment002RegisteredExecutorError(
            "completed history differs from the reserved final boundary"
        )


def _commit_registered_finalization(
    executor: RegisteredTrainingExecutor,
    handoff: RegisteredExecutorEpochHandoff,
    final_barrier: RegisteredHistoryBarrier,
    reservation: _FinalizationReservation,
    complete_trace: CompleteUpdateTraceEvidence,
    completed_history: RegisteredCompletedTrainingHistory,
    _truth_commit: Callable[
        [_FinalizationReservation, _FinalizedHandoffBinding], None
    ] = _AUTHORITY_TRUTH.commit_finalization,
) -> None:
    """Publish finalized H truth, then the COMPLETE executor, without H -> E."""

    _require_complete_trace_matches_finalization(reservation, complete_trace)
    _require_completed_history_matches_finalization(
        reservation,
        complete_trace,
        completed_history,
    )
    with _HANDOFFS_LOCK:
        handoff_state = _HANDOFFS.get(handoff)
        _validate_handoff(handoff, handoff_state)
        assert handoff_state is not None
        _require_handoff_matches_finalization(handoff_state, reservation)
        binding = _finalized_from_reservation(
            reservation,
            complete_trace,
            completed_history,
        )

    executor_state = _issued_executor_state(executor)
    with _trusted_executor_lock(executor, executor_state):
        _validate_executor_or_fail(executor, executor_state)
        _require_executor_matches_finalization(
            executor_state,
            reservation,
            require_attached=True,
        )
        with _HANDOFFS_LOCK:
            _require_finalization_reservation(
                reservation,
                phase="HISTORY_COMPLETE",
            )

    with _HANDOFFS_LOCK:
        _validate_handoff(handoff, handoff_state)
        _require_handoff_matches_finalization(handoff_state, reservation)
        _truth_commit(reservation, binding)
        _FINALIZED_HANDOFFS[handoff] = binding
        _FINALIZATION_PHASES[reservation] = "COMMITTED"
        _require_finalized_handoff_binding(handoff, handoff_state)

    with _trusted_executor_lock(executor, executor_state):
        _validate_executor_or_fail(executor, executor_state)
        _require_finalization_reservation(reservation, phase="COMMITTED")
        _require_executor_matches_finalization(
            executor_state,
            reservation,
            require_attached=True,
        )
        executor_state.active_handoff = None
        executor_state.phase = "COMPLETE"
        executor_state.final_barrier = final_barrier
        executor_state.complete_update_trace = complete_trace
        executor_state.completed_history = completed_history
        _publish_executor_lifecycle(executor, executor_state)
        observed = _stable_handoff_runtime_digests(
            executor_state.runtime,
            expected_generation=reservation.optimizer_generation,
            expected_training=False,
        )
        if (
            observed != reservation.runtime_digests
            or executor_state.expected != reservation.runtime_digests
            or executor_state.runtime.expected != reservation.runtime_digests
            or executor_state.optimizer_generation != reservation.optimizer_generation
            or executor_state.previous_history_barrier
            is not reservation.previous_history_barrier
        ):
            raise Experiment002RegisteredExecutorError(
                "registered finalization changed model, optimizer, RNG, or B28"
            )
        _validate_executor_or_fail(executor, executor_state)

    with _HANDOFFS_LOCK:
        _require_finalized_handoff_binding(handoff, handoff_state)


def _verify_registered_finalization(
    executor: RegisteredTrainingExecutor,
    reservation: _FinalizationReservation,
    complete_trace: CompleteUpdateTraceEvidence,
    completed_history: RegisteredCompletedTrainingHistory,
) -> None:
    """Reverify every completed authority with E and H locks initially free."""

    _require_complete_trace_matches_finalization(reservation, complete_trace)
    _require_completed_history_matches_finalization(
        reservation,
        complete_trace,
        completed_history,
    )
    state = _issued_executor_state(executor)
    with _trusted_executor_lock(executor, state):
        _validate_executor_or_fail(executor, state)
        observed = _stable_handoff_runtime_digests(
            state.runtime,
            expected_generation=reservation.optimizer_generation,
            expected_training=False,
        )
        if (
            observed != reservation.runtime_digests
            or state.expected != reservation.runtime_digests
            or state.runtime.expected != reservation.runtime_digests
        ):
            raise Experiment002RegisteredExecutorError(
                "completed executor runtime differs from finalization authority"
            )
    snapshot = _registered_executor_snapshot(executor)
    if (
        snapshot.phase != "COMPLETE"
        or snapshot.zero_based_epoch != _REGISTERED_EPOCH_COUNT - 1
        or snapshot.optimizer_generation != _REGISTERED_EPOCH_COUNT * _UPDATES_PER_EPOCH
        or snapshot.previous_history_barrier is not reservation.previous_history_barrier
        or snapshot.final_barrier is not reservation.final_barrier
        or snapshot.complete_update_trace is not complete_trace
        or snapshot.completed_history is not completed_history
    ):
        raise Experiment002RegisteredExecutorError(
            "completed executor snapshot differs from finalization authority"
        )
    _registered_epoch_handoff_snapshot(reservation.handoff)


def _abort_attempted_registered_history_completion(
    executor: RegisteredTrainingExecutor,
    final_barrier: RegisteredHistoryBarrier,
    completed: RegisteredCompletedTrainingHistory | None,
) -> None:
    """Invoke history's exact final compensation route without E or H locks."""

    from falsewake.experiment_002_registered_history import (
        _abort_attempted_registered_history_completion as abort,
    )

    with _EXECUTORS_LOCK:
        state = _EXECUTORS.get(executor)
    if type(state) is not _RegisteredExecutorState:
        raise Experiment002RegisteredExecutorError(
            "completion abort cannot recover executor registration"
        )
    abort(
        state.registration,
        executor,
        final_barrier,
        completed,
    )


def _ensure_attempted_registered_history_completion_aborted(
    executor: RegisteredTrainingExecutor,
    final_barrier: RegisteredHistoryBarrier,
    completed: RegisteredCompletedTrainingHistory | None,
    _fallback: Callable[
        [
            RegisteredTrainingExecutor,
            RegisteredHistoryBarrier,
            RegisteredCompletedTrainingHistory | None,
        ],
        None,
    ] = _abort_attempted_registered_history_completion,
) -> None:
    try:
        _abort_attempted_registered_history_completion(
            executor,
            final_barrier,
            completed,
        )
    except BaseException as primary_error:
        try:
            _fallback(executor, final_barrier, completed)
        except BaseException as fallback_error:
            raise fallback_error from primary_error


def _abort_consumed_registered_history(
    registration: VerifiedRunRegistration,
    executor: RegisteredTrainingExecutor,
    barrier: RegisteredHistoryBarrier,
) -> None:
    """Invoke history's exact post-consumption failure route without locks."""

    from falsewake.experiment_002_registered_history import (
        _abort_consumed_registered_history_barrier,
    )

    _abort_consumed_registered_history_barrier(
        registration,
        executor,
        barrier,
    )


def _terminal_fail_registered_continuation(
    executor: RegisteredTrainingExecutor,
    *,
    handoff: RegisteredExecutorEpochHandoff | None,
    reservation: _ContinuationReservation | None,
    _truth_fail: Callable[[_ContinuationReservation], None] = (
        _AUTHORITY_TRUTH.fail_continuation
    ),
) -> None:
    """Poison a failed advance without acquiring or calling history."""

    with _EXECUTORS_LOCK:
        executor_state = _EXECUTORS.get(executor)
    with _HANDOFFS_LOCK:
        handoff_state = _HANDOFFS.get(handoff) if handoff is not None else None
        if type(handoff_state) is _HandoffState:
            assert handoff is not None
            _terminal_fail_handoff(handoff, handoff_state)
    if type(executor_state) is not _RegisteredExecutorState:
        return
    lock = executor_state.anchor.lock
    with lock:
        if reservation is None:
            reservation = executor_state.continuation_reservation
        if reservation is not None:
            with _HANDOFFS_LOCK:
                with contextlib.suppress(BaseException):
                    _truth_fail(reservation)
                with contextlib.suppress(BaseException):
                    _CONTINUATION_PHASES[reservation] = "FAILED"
        _terminal_fail_executor(executor, executor_state)
        _propagate_executor_failure(executor_state, None)


def _terminal_fail_registered_finalization(
    executor: RegisteredTrainingExecutor,
    *,
    handoff: RegisteredExecutorEpochHandoff | None,
    reservation: _FinalizationReservation | None,
    _truth_fail: Callable[[_FinalizationReservation], None] = (
        _AUTHORITY_TRUTH.fail_finalization
    ),
    _truth_lookup: Callable[
        [RegisteredTrainingExecutor], _FinalizationReservation | None
    ] = _AUTHORITY_TRUTH.executor_finalization,
) -> None:
    """Poison one attempted finalization without crossing into history."""

    with _EXECUTORS_LOCK:
        executor_state = _EXECUTORS.get(executor)
    if type(executor_state) is not _RegisteredExecutorState:
        return
    lock = executor_state.anchor.lock
    with lock:
        if reservation is None:
            reservation = executor_state.finalization_reservation
        if reservation is None:
            reservation = _truth_lookup(executor)
        if handoff is None:
            handoff = (
                reservation.handoff
                if reservation is not None
                else executor_state.active_handoff
            )
        with _HANDOFFS_LOCK:
            if reservation is not None:
                with contextlib.suppress(BaseException):
                    _truth_fail(reservation)
                with contextlib.suppress(BaseException):
                    if not any(
                        observed is reservation for observed in _FINALIZATION_HISTORY
                    ):
                        _FINALIZATION_HISTORY.append(reservation)
                with contextlib.suppress(BaseException):
                    _FINALIZATION_PHASES[reservation] = "FAILED"
            handoff_state = _HANDOFFS.get(handoff) if handoff is not None else None
            if type(handoff_state) is _HandoffState:
                assert handoff is not None
                _terminal_fail_handoff(handoff, handoff_state)
        _terminal_fail_executor(executor, executor_state)
        _propagate_executor_failure(executor_state, None)


def _ensure_terminal_fail_registered_finalization(
    executor: RegisteredTrainingExecutor,
    *,
    handoff: RegisteredExecutorEpochHandoff | None,
    reservation: _FinalizationReservation | None,
    _fallback: Callable[..., None] = _terminal_fail_registered_finalization,
) -> None:
    try:
        _terminal_fail_registered_finalization(
            executor,
            handoff=handoff,
            reservation=reservation,
        )
    except BaseException as primary_error:
        try:
            _fallback(
                executor,
                handoff=handoff,
                reservation=reservation,
            )
        except BaseException as fallback_error:
            raise fallback_error from primary_error
    _require_local_registered_finalization_failed(
        executor,
        handoff=handoff,
        reservation=reservation,
    )


def _require_local_registered_finalization_failed(
    executor: RegisteredTrainingExecutor,
    *,
    handoff: RegisteredExecutorEpochHandoff | None,
    reservation: _FinalizationReservation | None,
    _truth_lookup: Callable[
        [RegisteredTrainingExecutor], _FinalizationReservation | None
    ] = _AUTHORITY_TRUTH.executor_finalization,
    _truth_validate: Callable[
        [_FinalizationReservation, _FinalizationPhase], bool
    ] = _AUTHORITY_TRUTH.validate_finalization,
) -> None:
    with _EXECUTORS_LOCK:
        state = _EXECUTORS.get(executor)
    if type(state) is not _RegisteredExecutorState:
        return
    if reservation is None:
        reservation = state.finalization_reservation
    if reservation is None:
        reservation = _truth_lookup(executor)
    if handoff is None:
        handoff = (
            reservation.handoff if reservation is not None else state.active_handoff
        )
    with state.anchor.lock, _HANDOFFS_LOCK:
        handoff_state = _HANDOFFS.get(handoff) if handoff is not None else None
        if (
            state.phase != "FAILED"
            or executor not in _FAILED_EXECUTOR_HISTORY
            or (
                handoff is not None
                and (
                    type(handoff_state) is not _HandoffState
                    or not handoff_state.failed
                    or handoff not in _FAILED_HANDOFF_HISTORY
                )
            )
            or (
                reservation is not None
                and (
                    _FINALIZATION_PHASES.get(reservation) != "FAILED"
                    or not _truth_validate(reservation, "FAILED")
                )
            )
        ):
            raise Experiment002RegisteredExecutorError(
                "registered finalization was not locally terminalized"
            )


def _retirement_from_reservation(
    reservation: _ContinuationReservation,
) -> _RetiredHandoffBinding:
    return _RetiredHandoffBinding(
        reservation=reservation,
        executor=reservation.executor,
        handoff=reservation.handoff,
        barrier=reservation.barrier,
        evaluated_epoch=reservation.evaluated_epoch,
        evaluated_authority_sha256=reservation.evaluated_authority_sha256,
        registration=reservation.registration,
        validation_inputs=reservation.validation_inputs,
        process_id=reservation.process_id,
        seed=reservation.seed,
        zero_based_epoch=reservation.zero_based_epoch,
        optimizer_generation=reservation.optimizer_generation,
        runtime_digests=replace(reservation.runtime_digests),
        previous_history_barrier=reservation.previous_history_barrier,
        handoff_token=reservation.handoff_token,
        one_shot_token=reservation.one_shot_token,
    )


def _finalized_from_reservation(
    reservation: _FinalizationReservation,
    complete_trace: CompleteUpdateTraceEvidence,
    completed_history: RegisteredCompletedTrainingHistory,
) -> _FinalizedHandoffBinding:
    return _FinalizedHandoffBinding(
        reservation=reservation,
        executor=reservation.executor,
        handoff=reservation.handoff,
        final_barrier=reservation.final_barrier,
        history=reservation.history,
        evaluated_epoch=reservation.evaluated_epoch,
        evaluated_authority_sha256=reservation.evaluated_authority_sha256,
        complete_update_trace=complete_trace,
        complete_update_trace_type=type(complete_trace),
        completed_history=completed_history,
        completed_history_type=type(completed_history),
        registration=reservation.registration,
        validation_inputs=reservation.validation_inputs,
        process_id=reservation.process_id,
        seed=reservation.seed,
        zero_based_epoch=reservation.zero_based_epoch,
        optimizer_generation=reservation.optimizer_generation,
        runtime_digests=replace(reservation.runtime_digests),
        previous_history_barrier=reservation.previous_history_barrier,
        handoff_token=reservation.handoff_token,
        one_shot_token=reservation.one_shot_token,
    )


def _require_same_history_barrier_snapshot(
    issued: _RegisteredHistoryBarrierSnapshot,
    consumed: _RegisteredHistoryBarrierSnapshot,
    *,
    expected_phase: Literal["CONSUMED"],
) -> None:
    if (
        type(consumed) is not type(issued)
        or consumed.registration is not issued.registration
        or consumed.history is not issued.history
        or consumed.executor is not issued.executor
        or consumed.validation_inputs is not issued.validation_inputs
        or consumed.process_id != issued.process_id
        or consumed.seed != issued.seed
        or consumed.accepted_epoch != issued.accepted_epoch
        or consumed.next_zero_based_epoch != issued.next_zero_based_epoch
        or consumed.evaluated_epoch is not issued.evaluated_epoch
        or consumed.evaluated_handoff is not issued.evaluated_handoff
        or consumed.evaluated_authority_sha256 != issued.evaluated_authority_sha256
        or issued.phase != "ISSUED"
        or consumed.phase != expected_phase
    ):
        raise Experiment002RegisteredExecutorError(
            "history barrier changed across one-use consumption"
        )


def _require_lower_sha256(value: object, name: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Experiment002RegisteredExecutorError(
            f"{name} must be an exact lowercase SHA-256"
        )


def _execute_registered_update(
    executor: RegisteredTrainingExecutor,
    state: _RegisteredExecutorState,
    epoch: RegisteredTrainingEpoch,
    *,
    global_update: int,
) -> None:
    """Perform the exact population/optimizer/trace handshake for one update."""

    before = _entry_zero_grad_digests(state, global_update=global_update)
    batch = _next_registered_batch(state, epoch, global_update=global_update)
    _verify_registered_batch_pre(state, epoch, batch, global_update=global_update)
    views = _zero_copy_registered_batch(batch, global_update=global_update)
    outcome = _perform_registered_numeric_update(
        state,
        views,
        before=before,
        global_update=global_update,
    )
    _verify_registered_batch_post(state, epoch, batch, global_update=global_update)
    binding = _registered_batch_binding(state, epoch, batch)
    after = _capture_registered_update_after(
        state,
        before=before,
        outcome=outcome,
        global_update=global_update,
    )
    transition = _issue_update_transition(
        state,
        binding=binding,
        before=before,
        after=after,
        outcome=outcome,
        global_update=global_update,
    )
    state.active_transition = transition
    _publish_executor_lifecycle(executor, state)
    receipt = _complete_registered_update(state, epoch, batch, transition)
    consumption = _consume_registered_trace(state, receipt, transition)
    _accept_registered_population(
        state,
        epoch,
        receipt,
        transition,
        consumption,
    )
    _post_update_digests(
        state,
        expected=after,
        global_update=global_update,
    )
    state.expected = after
    state.runtime.expected = after
    state.optimizer_generation = global_update + 1
    state.runtime.optimizer_generation = global_update + 1
    state.active_transition = None
    _publish_executor_lifecycle(executor, state)


def _entry_zero_grad_digests(
    state: _RegisteredExecutorState, *, global_update: int
) -> _numeric._RuntimeDigests:
    if global_update != state.optimizer_generation:
        raise Experiment002RegisteredExecutorError(
            "global update differs from optimizer generation"
        )
    entry = _numeric._stable_runtime_digests(
        state.runtime, expected_generation=global_update
    )
    if entry != state.expected:
        raise Experiment002RegisteredExecutorError(
            "runtime continuity changed before registered update"
        )
    state.runtime.optimizer.zero_grad(set_to_none=True)
    if any(parameter.grad is not None for parameter in state.runtime.parameters):
        raise Experiment002RegisteredExecutorError(
            "zero_grad did not clear every registered gradient"
        )
    before = _numeric._stable_runtime_digests(
        state.runtime, expected_generation=global_update
    )
    if before != entry:
        raise Experiment002RegisteredExecutorError(
            "zero_grad changed model, optimizer, or Torch RNG"
        )
    return before


def _next_registered_batch(
    state: _RegisteredExecutorState,
    epoch: RegisteredTrainingEpoch,
    *,
    global_update: int,
) -> MaterializedTrainingBatch:
    return _executor_next_batch(epoch, state.bridge_authority, global_update)


def _verify_registered_batch_pre(
    state: _RegisteredExecutorState,
    epoch: RegisteredTrainingEpoch,
    batch: MaterializedTrainingBatch,
    *,
    global_update: int,
) -> None:
    _executor_verify_pending_batch(
        epoch,
        state.bridge_authority,
        batch,
        global_update=global_update,
        phase="PRE",
    )


def _zero_copy_registered_batch(
    batch: MaterializedTrainingBatch, *, global_update: int
) -> _BatchViews:
    model_inputs = batch.model_inputs
    label_indices = batch.label_indices
    batch_size = batch.batch_size
    expected_size = TRAINING_BATCH_SIZE
    if global_update % _UPDATES_PER_EPOCH == _UPDATES_PER_EPOCH - 1:
        expected_size = REGISTERED_LAST_BATCH_SIZE
    if (
        batch.zero_based_global_update != global_update
        or batch_size != expected_size
        or type(model_inputs) is not np.ndarray
        or model_inputs.dtype != np.dtype(np.float32)
        or model_inputs.shape
        != (batch_size, INPUT_MEL_BINS, MODEL_RECEPTIVE_FIELD_FRAMES)
        or not model_inputs.flags.c_contiguous
        or model_inputs.flags.writeable
        or not model_inputs.flags.owndata
        or model_inputs.base is not None
        or not bool(np.isfinite(model_inputs).all())
        or type(label_indices) is not np.ndarray
        or label_indices.dtype != np.dtype(np.int64)
        or label_indices.shape != (batch_size,)
        or not label_indices.flags.c_contiguous
        or label_indices.flags.writeable
        or not label_indices.flags.owndata
        or label_indices.base is not None
        or bool(np.any(label_indices < 0))
        or bool(np.any(label_indices >= OUTPUT_CLASSES))
    ):
        raise Experiment002RegisteredExecutorError(
            "registered materialized batch has an invalid layout"
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        features = torch.from_numpy(model_inputs)
        labels = torch.from_numpy(label_indices)
    if (
        type(features) is not Tensor
        or tuple(features.shape) != tuple(model_inputs.shape)
        or features.device.type != "cpu"
        or features.dtype != torch.float32
        or not features.is_contiguous()
        or features.data_ptr() != model_inputs.ctypes.data
        or type(labels) is not Tensor
        or tuple(labels.shape) != tuple(label_indices.shape)
        or labels.device.type != "cpu"
        or labels.dtype != torch.int64
        or not labels.is_contiguous()
        or labels.data_ptr() != label_indices.ctypes.data
    ):
        raise Experiment002RegisteredExecutorError(
            "Torch batch is not an exact zero-copy registered view"
        )
    return _BatchViews(
        model_inputs=cast(
            np.ndarray[tuple[int, int, int], np.dtype[np.float32]], model_inputs
        ),
        label_indices=label_indices,
        features=features,
        labels=labels,
        batch_size=batch_size,
    )


def _perform_registered_numeric_update(
    state: _RegisteredExecutorState,
    views: _BatchViews,
    *,
    before: _numeric._RuntimeDigests,
    global_update: int,
) -> _NumericUpdateOutcome:
    model = state.runtime.model
    model.train()
    initial_state = model.initial_state(
        views.batch_size,
        device=views.features.device,
        dtype=views.features.dtype,
    )
    _validate_registered_stream_state(
        initial_state,
        batch_size=views.batch_size,
        expected_frames=0,
        require_positive_zero=True,
    )
    initial_payload = _stream_state_payload(initial_state)
    rng_before_forward = _numeric._rng_state_bytes()
    if rng_before_forward != before.rng_state:
        raise Experiment002RegisteredExecutorError(
            "batch binding or initial state creation consumed Torch RNG"
        )
    logits, next_state = model.forward_stream(views.features, initial_state)
    rng_after_forward = _numeric._rng_state_bytes()
    if rng_after_forward == rng_before_forward:
        raise Experiment002RegisteredExecutorError(
            "registered training dropout did not consume Torch RNG"
        )
    _validate_registered_logits(logits, batch_size=views.batch_size)
    _validate_registered_stream_state(
        next_state,
        batch_size=views.batch_size,
        expected_frames=MODEL_RECEPTIVE_FIELD_FRAMES,
        require_positive_zero=False,
    )
    if _stream_state_payload(initial_state) != initial_payload:
        raise Experiment002RegisteredExecutorError(
            "registered forward mutated its fresh initial state"
        )
    loss = _numeric._training_cross_entropy(
        logits[:, MODEL_RECEPTIVE_FIELD_FRAMES - 1, :], views.labels
    )
    _numeric._validate_loss(loss)
    loss.backward()  # type: ignore[no-untyped-call]
    _numeric._validate_gradients(state.runtime.parameters)
    returned_norm = torch.nn.utils.clip_grad_norm_(
        state.runtime.parameters,
        _MAX_GRADIENT_NORM,
        norm_type=2.0,
        error_if_nonfinite=True,
        foreach=False,
    )
    _numeric._validate_preclip_norm(returned_norm)
    _numeric._validate_gradients(state.runtime.parameters)
    learning_rate = registered_learning_rate(global_update)
    if struct.pack("<d", learning_rate) != struct.pack(
        "<d", _numeric._registered_learning_rate(global_update)
    ):
        raise Experiment002RegisteredExecutorError(
            "registered learning-rate authorities disagree"
        )
    state.runtime.optimizer.param_groups[0]["lr"] = learning_rate
    state.runtime.optimizer.step()
    if _numeric._rng_state_bytes() != rng_after_forward:
        raise Experiment002RegisteredExecutorError(
            "Torch RNG changed outside registered model forward"
        )
    return _NumericUpdateOutcome(
        learning_rate=learning_rate,
        batch_mean_training_loss=np.float32(loss.detach().item()),
        returned_preclip_l2_norm=np.float32(returned_norm.detach().item()),
        rng_state_after_forward=rng_after_forward,
    )


def _verify_registered_batch_post(
    state: _RegisteredExecutorState,
    epoch: RegisteredTrainingEpoch,
    batch: MaterializedTrainingBatch,
    *,
    global_update: int,
) -> None:
    _executor_verify_pending_batch(
        epoch,
        state.bridge_authority,
        batch,
        global_update=global_update,
        phase="POST",
    )


def _registered_batch_binding(
    state: _RegisteredExecutorState,
    epoch: RegisteredTrainingEpoch,
    batch: MaterializedTrainingBatch,
) -> _RegisteredBatchBinding:
    return _executor_batch_binding(epoch, state.bridge_authority, batch)


def _capture_registered_update_after(
    state: _RegisteredExecutorState,
    *,
    before: _numeric._RuntimeDigests,
    outcome: _NumericUpdateOutcome,
    global_update: int,
) -> _numeric._RuntimeDigests:
    after = _numeric._stable_runtime_digests(
        state.runtime, expected_generation=global_update + 1
    )
    if (
        before.model_sha256 == after.model_sha256
        or before.optimizer_sha256 == after.optimizer_sha256
        or after.rng_state != outcome.rng_state_after_forward
    ):
        raise Experiment002RegisteredExecutorError(
            "registered optimizer update did not produce exact runtime continuity"
        )
    return after


def _issue_update_transition(
    state: _RegisteredExecutorState,
    *,
    binding: _RegisteredBatchBinding,
    before: _numeric._RuntimeDigests,
    after: _numeric._RuntimeDigests,
    outcome: _NumericUpdateOutcome,
    global_update: int,
) -> RegisteredOptimizerTransition:
    if (
        binding.executor_session_token is not state.bridge_session_token
        or binding.seed != state.seed
        or binding.zero_based_epoch != state.zero_based_epoch
        or binding.zero_based_global_update != global_update
    ):
        raise Experiment002RegisteredExecutorError(
            "registered batch binding differs from executor context"
        )
    return _issue_registered_optimizer_transition(
        executor_authority=state.bridge_authority,
        epoch_session_token=binding.epoch_session_token,
        batch_token=binding.batch_token,
        zero_based_epoch=state.zero_based_epoch,
        zero_based_global_update=global_update,
        batch_size=binding.batch_size,
        learning_rate=outcome.learning_rate,
        batch_mean_training_loss=outcome.batch_mean_training_loss,
        returned_preclip_l2_norm=outcome.returned_preclip_l2_norm,
        optimizer_generation_before=global_update,
        optimizer_generation_after=global_update + 1,
        model_sha256_before=before.model_sha256,
        model_sha256_after=after.model_sha256,
        optimizer_sha256_before=before.optimizer_sha256,
        optimizer_sha256_after=after.optimizer_sha256,
        rng_sha256_before=before.rng_sha256,
        rng_sha256_after=after.rng_sha256,
    )


def _complete_registered_update(
    state: _RegisteredExecutorState,
    epoch: RegisteredTrainingEpoch,
    batch: MaterializedTrainingBatch,
    transition: RegisteredOptimizerTransition,
) -> CompletedTrainingUpdate:
    snapshot = _transition_snapshot(transition, required_phase="ISSUED")
    if snapshot.optimizer_generation_after != state.optimizer_generation + 1:
        raise Experiment002RegisteredExecutorError(
            "transition generation differs before receipt issuance"
        )
    return _executor_complete_update(epoch, state.bridge_authority, batch, transition)


def _consume_registered_trace(
    state: _RegisteredExecutorState,
    receipt: CompletedTrainingUpdate,
    transition: RegisteredOptimizerTransition,
) -> TraceConsumedTransition:
    return state.trace._consume_registered_update(receipt, transition)


def _accept_registered_population(
    state: _RegisteredExecutorState,
    epoch: RegisteredTrainingEpoch,
    receipt: CompletedTrainingUpdate,
    transition: RegisteredOptimizerTransition,
    consumption: TraceConsumedTransition,
) -> None:
    _executor_accept_completed_update(
        epoch,
        state.bridge_authority,
        receipt,
        transition,
        consumption,
    )


def _post_update_digests(
    state: _RegisteredExecutorState,
    *,
    expected: _numeric._RuntimeDigests,
    global_update: int,
) -> None:
    observed = _numeric._stable_runtime_digests(
        state.runtime, expected_generation=global_update + 1
    )
    if observed != expected:
        raise Experiment002RegisteredExecutorError(
            "runtime changed during receipt/trace/population acceptance"
        )


def _registered_epoch_handoff_snapshot(
    handoff: RegisteredExecutorEpochHandoff,
) -> _RegisteredExecutorEpochHandoffSnapshot:
    """Return checked handoff identities and digests without exposing the model."""

    state = _access_registered_epoch_handoff(handoff, registration=None, consume=False)
    return _RegisteredExecutorEpochHandoffSnapshot(
        executor=state.executor,
        registration=state.registration,
        validation_inputs=state.validation_inputs,
        process_id=state.process_id,
        bridge_authority=state.bridge_authority,
        bridge_session_token=state.bridge_session_token,
        population=state.population,
        epoch_trace=state.epoch_trace,
        seed=state.seed,
        zero_based_epoch=state.zero_based_epoch,
        optimizer_generation=state.optimizer_generation,
        runtime_digests=replace(state.runtime_digests),
        previous_history_barrier=state.previous_history_barrier,
        one_shot_token=state.one_shot_token,
        another_training_epoch=state.zero_based_epoch + 1 < _REGISTERED_EPOCH_COUNT,
    )


def _registered_epoch_handoff_model(
    registration: VerifiedRunRegistration,
    handoff: RegisteredExecutorEpochHandoff,
) -> CausalKWS:
    """Consume one checked handoff and return its exact live model to evaluation."""

    state = _access_registered_epoch_handoff(
        handoff, registration=registration, consume=True
    )
    return state.model


def _fail_registered_epoch_handoff(
    handoff: RegisteredExecutorEpochHandoff,
) -> None:
    """Poison a claimed handoff and every owning training authority."""

    if type(handoff) is not RegisteredExecutorEpochHandoff:
        raise TypeError("handoff must be a RegisteredExecutorEpochHandoff")
    validation_error: BaseException | None = None
    with _HANDOFFS_LOCK:
        state = _HANDOFFS.get(handoff)
        if type(state) is not _HandoffState:
            raise Experiment002RegisteredExecutorError(
                "registered epoch handoff was not issued by this module"
            )
        if state.failed:
            return
        try:
            _validate_handoff(handoff, state)
        except BaseException as error:
            validation_error = error
    _poison_handoff_and_executor(handoff, state)
    if validation_error is not None:
        raise validation_error


def _access_registered_epoch_handoff(
    handoff: RegisteredExecutorEpochHandoff,
    *,
    registration: VerifiedRunRegistration | None,
    consume: bool,
    _truth_record: Callable[
        [_HandoffIssuanceTicket, Literal["USED", "FAILED"]], None
    ] = _AUTHORITY_TRUTH.record_handoff_event,
    _truth_retired: Callable[
        [RegisteredExecutorEpochHandoff], _RetiredHandoffBinding | None
    ] = _AUTHORITY_TRUTH.retired_handoff,
    _truth_finalized: Callable[
        [RegisteredExecutorEpochHandoff], _FinalizedHandoffBinding | None
    ] = _AUTHORITY_TRUTH.finalized_handoff,
) -> _HandoffState:
    """Validate a handoff against its awaiting executor and optionally claim it."""

    if type(handoff) is not RegisteredExecutorEpochHandoff:
        raise TypeError("handoff must be a RegisteredExecutorEpochHandoff")
    if registration is not None and type(registration) is not VerifiedRunRegistration:
        raise TypeError("registration must be a VerifiedRunRegistration")
    if type(consume) is not bool:
        raise TypeError("consume must be a bool")
    state: _HandoffState | None
    retired: _RetiredHandoffBinding | None = None
    finalized: _FinalizedHandoffBinding | None = None
    validation_error: BaseException | None = None
    with _HANDOFFS_LOCK:
        state = _HANDOFFS.get(handoff)
        try:
            _validate_handoff(handoff, state)
            assert state is not None
            retired = _truth_retired(handoff)
            finalized = _truth_finalized(handoff)
            if (retired is not None or handoff in _RETIRED_HANDOFFS) and (
                finalized is not None or handoff in _FINALIZED_HANDOFFS
            ):
                raise Experiment002RegisteredExecutorError(
                    "handoff has conflicting terminal archive truth"
                )
            if retired is not None or handoff in _RETIRED_HANDOFFS:
                _require_retired_handoff_binding(
                    handoff,
                    state,
                    _truth_retired=_truth_retired,
                )
            elif finalized is not None or handoff in _FINALIZED_HANDOFFS:
                _require_finalized_handoff_binding(
                    handoff,
                    state,
                    _truth_finalized=_truth_finalized,
                )
        except BaseException as error:
            validation_error = error
        if validation_error is None:
            assert state is not None
            if registration is not None and state.registration is not registration:
                raise Experiment002RegisteredExecutorError(
                    "handoff registration identity differs"
                )
            if state.failed or (consume and state.used):
                raise Experiment002RegisteredExecutorError(
                    "registered epoch handoff was already consumed or failed"
                )
    if type(state) is not _HandoffState:
        assert validation_error is not None
        raise validation_error
    if validation_error is not None:
        _poison_handoff_and_executor(handoff, state)
        raise validation_error

    reverify_verified_run_registration(state.registration)
    if retired is not None or finalized is not None:
        if consume:
            raise Experiment002RegisteredExecutorError(
                "archived registered epoch handoffs cannot return a model"
            )
        return state

    try:
        executor_state = _issued_executor_state(state.executor)
        with _trusted_executor_lock(state.executor, executor_state):
            _validate_executor_or_fail(state.executor, executor_state)
            expected_training = _expected_handoff_model_training(state)
            if (
                executor_state.phase != "AWAITING_EVALUATION"
                or executor_state.active_handoff is not handoff
                or executor_state.runtime.model is not state.model
                or executor_state.expected != state.runtime_digests
                or executor_state.optimizer_generation != state.optimizer_generation
                or executor_state.runtime.model.training is not expected_training
            ):
                raise Experiment002RegisteredExecutorError(
                    "handoff no longer matches its awaiting executor"
                )
            observed = _stable_handoff_runtime_digests(
                executor_state.runtime,
                expected_generation=state.optimizer_generation,
                expected_training=expected_training,
            )
            if observed != state.runtime_digests:
                raise Experiment002RegisteredExecutorError(
                    "handoff runtime changed before evaluation"
                )
    except BaseException as executor_error:
        if not consume:
            try:
                if _exact_retired_handoff_won_snapshot_race(
                    handoff,
                    state,
                    _truth_retired=_truth_retired,
                    _truth_finalized=_truth_finalized,
                ):
                    return state
            except BaseException as retirement_error:
                _poison_handoff_and_executor(handoff, state)
                raise retirement_error from executor_error
        _poison_handoff_and_executor(handoff, state)
        raise

    archived_during_access = False
    try:
        with _HANDOFFS_LOCK:
            if _HANDOFFS.get(handoff) is not state:
                raise Experiment002RegisteredExecutorError(
                    "handoff state changed during executor verification"
                )
            _validate_handoff(handoff, state)
            truth_retired = _truth_retired(handoff)
            truth_finalized = _truth_finalized(handoff)
            visible_retired = handoff in _RETIRED_HANDOFFS
            visible_finalized = handoff in _FINALIZED_HANDOFFS
            if (truth_retired is not None or visible_retired) and (
                truth_finalized is not None or visible_finalized
            ):
                raise Experiment002RegisteredExecutorError(
                    "handoff gained conflicting terminal archive truth"
                )
            if truth_retired is not None or visible_retired:
                if consume:
                    raise Experiment002RegisteredExecutorError(
                        "live handoff retired during evaluator access"
                    )
                binding = _require_retired_handoff_binding(
                    handoff,
                    state,
                    _truth_retired=_truth_retired,
                )
                if (
                    truth_retired is not binding
                    or _RETIRED_HANDOFFS.get(handoff) is not binding
                ):
                    raise Experiment002RegisteredExecutorError(
                        "retired handoff race lost exact authority"
                    )
                archived_during_access = True
            elif truth_finalized is not None or visible_finalized:
                if consume:
                    raise Experiment002RegisteredExecutorError(
                        "live handoff finalized during evaluator access"
                    )
                finalized_race_binding = _require_finalized_handoff_binding(
                    handoff,
                    state,
                    _truth_finalized=_truth_finalized,
                )
                if (
                    truth_finalized is not finalized_race_binding
                    or _FINALIZED_HANDOFFS.get(handoff) is not finalized_race_binding
                ):
                    raise Experiment002RegisteredExecutorError(
                        "finalized handoff race lost exact authority"
                    )
                archived_during_access = True
            elif consume:
                _truth_record(state.ticket, "USED")
                _record_handoff_event(state.ticket, "USED")
                state.used = True
                _publish_handoff_lifecycle(handoff, state)
    except BaseException:
        _poison_handoff_and_executor(handoff, state)
        raise
    if archived_during_access:
        return state

    try:
        with _trusted_executor_lock(state.executor, executor_state):
            _validate_executor_or_fail(state.executor, executor_state)
            if (
                executor_state.phase != "AWAITING_EVALUATION"
                or executor_state.active_handoff is not handoff
                or executor_state.runtime.model is not state.model
                or executor_state.expected != state.runtime_digests
                or executor_state.optimizer_generation != state.optimizer_generation
                or executor_state.runtime.model.training is not expected_training
            ):
                raise Experiment002RegisteredExecutorError(
                    "handoff changed after evaluator access"
                )
    except BaseException as executor_error:
        if not consume:
            try:
                if _exact_retired_handoff_won_snapshot_race(
                    handoff,
                    state,
                    _truth_retired=_truth_retired,
                    _truth_finalized=_truth_finalized,
                ):
                    return state
            except BaseException as retirement_error:
                _poison_handoff_and_executor(handoff, state)
                raise retirement_error from executor_error
        _poison_handoff_and_executor(handoff, state)
        raise
    return state


def _registered_executor_snapshot(
    executor: RegisteredTrainingExecutor,
) -> _RegisteredExecutorSnapshot:
    """Return only scalar/digest state; model and optimizer remain private."""

    state = _issued_executor_state(executor)
    with _trusted_executor_lock(executor, state):
        _validate_executor_or_fail(executor, state)
        if state.phase == "FAILED":
            raise Experiment002RegisteredExecutorError(
                "registered executor is terminally failed"
            )
        return _RegisteredExecutorSnapshot(
            registration=state.registration,
            process_id=state.process_id,
            seed=state.seed,
            phase=state.phase,
            zero_based_epoch=state.zero_based_epoch,
            optimizer_generation=state.optimizer_generation,
            model_sha256=state.expected.model_sha256,
            optimizer_sha256=state.expected.optimizer_sha256,
            rng_sha256=state.expected.rng_sha256,
            previous_history_barrier=state.previous_history_barrier,
            final_barrier=state.final_barrier,
            complete_update_trace=state.complete_update_trace,
            completed_history=state.completed_history,
        )


def _expected_handoff_model_training(state: _HandoffState) -> bool:
    """Return the exact mode allowed at this point in evaluator ownership."""

    if not 0 <= state.zero_based_epoch < _REGISTERED_EPOCH_COUNT:
        raise Experiment002RegisteredExecutorError(
            "handoff epoch is outside the registered layout"
        )
    return not state.used or state.zero_based_epoch < _REGISTERED_EPOCH_COUNT - 1


def _stable_handoff_runtime_digests(
    runtime: _numeric._ExecutorState,
    *,
    expected_generation: int,
    expected_training: bool,
) -> _numeric._RuntimeDigests:
    """Double-snapshot runtime state while preserving the required model mode."""

    if type(expected_training) is not bool:
        raise TypeError("expected_training must be a bool")
    if runtime.model.training is not expected_training:
        raise Experiment002RegisteredExecutorError(
            "handoff model mode differs before runtime digest verification"
        )
    if expected_training:
        return _numeric._stable_runtime_digests(
            runtime, expected_generation=expected_generation
        )
    first = _handoff_runtime_digests_in_eval_mode(
        runtime, expected_generation=expected_generation
    )
    second = _handoff_runtime_digests_in_eval_mode(
        runtime, expected_generation=expected_generation
    )
    if first != second or runtime.model.training is not False:
        raise Experiment002RegisteredExecutorError(
            "final handoff runtime changed during eval-mode double snapshot"
        )
    return first


def _handoff_runtime_digests_in_eval_mode(
    runtime: _numeric._ExecutorState, *, expected_generation: int
) -> _numeric._RuntimeDigests:
    _validate_eval_mode_parameter_capture(runtime)
    _numeric._validate_optimizer(runtime, expected_generation=expected_generation)
    rng_state = _numeric._rng_state_bytes()
    return _numeric._RuntimeDigests(
        model_sha256=_numeric._model_sha256(
            runtime.model,
            runtime.parameter_names,
            runtime.parameters,
        ),
        optimizer_sha256=_numeric._optimizer_sha256(
            runtime, expected_generation=expected_generation
        ),
        rng_sha256=hashlib.sha256(
            _numeric._RNG_DOMAIN + struct.pack("<Q", len(rng_state)) + rng_state
        ).hexdigest(),
        rng_state=rng_state,
    )


def _validate_eval_mode_parameter_capture(runtime: _numeric._ExecutorState) -> None:
    model = runtime.model
    names = runtime.parameter_names
    parameters = runtime.parameters
    current = tuple(model.named_parameters())
    if (
        type(model) is not CausalKWS
        or model.training is not False
        or any(module.training is not False for module in model.modules())
        or type(model.dropout) is not torch.nn.Dropout
        or type(model.dropout.p) is not float
        or struct.pack("<d", model.dropout.p) != struct.pack("<d", 0.10)
        or model.dropout.inplace is not False
        or len(current) != _numeric._MODEL_TENSOR_COUNT
        or len(names) != _numeric._MODEL_TENSOR_COUNT
        or len(parameters) != _numeric._MODEL_TENSOR_COUNT
        or tuple(name for name, _ in current) != names
        or len(set(names)) != len(names)
        or len({id(parameter) for parameter in parameters}) != len(parameters)
        or any(
            observed is not captured
            for (_, observed), captured in zip(current, parameters, strict=True)
        )
        or sum(parameter.numel() for parameter in parameters)
        != _numeric._MODEL_PARAMETER_VALUE_COUNT
        or TRAINABLE_PARAMETER_COUNT != _numeric._MODEL_PARAMETER_VALUE_COUNT
        or tuple(model.state_dict()) != names
    ):
        raise Experiment002RegisteredExecutorError(
            "eval-mode CausalKWS parameter capture differs from registration"
        )
    for name, parameter in zip(names, parameters, strict=True):
        if (
            type(name) is not str
            or type(parameter) is not torch.nn.Parameter
            or parameter.device.type != "cpu"
            or parameter.dtype != torch.float32
            or parameter.layout != torch.strided
            or not parameter.is_contiguous()
            or not parameter.requires_grad
            or not bool(torch.isfinite(parameter).all().item())
        ):
            raise Experiment002RegisteredExecutorError(
                f"eval-mode registered parameter is invalid ({name})"
            )


def _reserve_process_creation(process_id: int) -> None:
    _require_process_id(process_id)
    with _CREATION_LOCK:
        if process_id in _CREATION_RESERVED_PROCESSES or any(
            observed_pid == process_id
            for observed_pid, _ in _CREATION_RESERVATION_HISTORY
        ):
            raise Experiment002RegisteredExecutorError(
                "this process already reserved its registered executor"
            )
        reservation = (process_id, object())
        _CREATION_RESERVED_PROCESSES.add(process_id)
        _CREATION_RESERVATION_HISTORY.append(reservation)


def _issue_registered_executor(
    *,
    registration: VerifiedRunRegistration,
    process_id: int,
    seed: int,
    training_inputs: RegisteredTrainingInputSource,
    validation_inputs: RegisteredValidationInputs,
    bridge_authority: RegisteredExecutorSessionAuthority,
    bridge_session_token: object,
    trace: UpdateTraceAccumulator,
    runtime: _numeric._ExecutorState,
    expected: _numeric._RuntimeDigests,
    lock: threading.RLock,
    _truth_issue: Callable[
        [
            RegisteredTrainingExecutor,
            _ExecutorIssuanceTicket,
            _RegisteredExecutorLifecycle,
        ],
        None,
    ] = _AUTHORITY_TRUTH.issue_executor,
) -> RegisteredTrainingExecutor:
    result = object.__new__(RegisteredTrainingExecutor)
    token = object()
    ticket = _ExecutorIssuanceTicket(
        executor=result,
        token=object(),
        registration=registration,
        process_id=process_id,
    )
    anchor = _RegisteredExecutorAnchor(
        executor=result,
        ticket=ticket,
        token=token,
        route_marker=_REGISTERED_ROUTE,
        registration=registration,
        process_id=process_id,
        seed=seed,
        training_inputs=training_inputs,
        validation_inputs=validation_inputs,
        bridge_authority=bridge_authority,
        bridge_session_token=bridge_session_token,
        trace=trace,
        runtime=runtime,
        model=runtime.model,
        optimizer=runtime.optimizer,
        parameter_names=runtime.parameter_names,
        parameters=runtime.parameters,
        lock=lock,
    )
    state = _RegisteredExecutorState(
        anchor=anchor,
        ticket=ticket,
        token=token,
        route_marker=_REGISTERED_ROUTE,
        registration=registration,
        process_id=process_id,
        seed=seed,
        training_inputs=training_inputs,
        validation_inputs=validation_inputs,
        bridge_authority=bridge_authority,
        bridge_session_token=bridge_session_token,
        trace=trace,
        runtime=runtime,
        expected=expected,
        lock=lock,
    )
    guard = _guard_from_anchor(anchor)
    lifecycle = _lifecycle_from_state(state)
    lifecycle_authority = _RegisteredExecutorLifecycleAuthority(
        executor=result,
        sequence=0,
        lifecycle=lifecycle,
        previous=None,
    )
    _truth_issue(result, ticket, lifecycle)
    with _EXECUTORS_LOCK:
        _EXECUTORS[result] = state
        _EXECUTOR_GUARDS[result] = guard
        _EXECUTOR_LIFECYCLES[result] = lifecycle
        _EXECUTOR_LIFECYCLE_HEADS[result] = lifecycle_authority
        _ISSUED_EXECUTOR_LIFECYCLES.add(lifecycle_authority)
        _EXECUTOR_LIFECYCLE_HISTORY.append(lifecycle_authority)
        _EXECUTOR_ANCHORS[result] = anchor
        _EXECUTOR_ANCHOR_HISTORY.append(anchor)
        _ISSUED_EXECUTOR_TICKETS.add(ticket)
        _EXECUTOR_TICKET_HISTORY.append(ticket)
    _registered_executor_snapshot(result)
    return result


def _issue_epoch_handoff(
    executor: RegisteredTrainingExecutor,
    executor_state: _RegisteredExecutorState,
    *,
    population: CompletedTrainingPopulation,
    epoch_trace: EpochUpdateTraceEvidence,
    _truth_issue: Callable[
        [RegisteredExecutorEpochHandoff, _HandoffIssuanceTicket, _HandoffLifecycle],
        None,
    ] = _AUTHORITY_TRUTH.issue_handoff,
) -> RegisteredExecutorEpochHandoff:
    if type(population) is not CompletedTrainingPopulation:
        raise TypeError("population must be a CompletedTrainingPopulation")
    if type(epoch_trace) is not EpochUpdateTraceEvidence:
        raise TypeError("epoch_trace must be an EpochUpdateTraceEvidence")
    if (
        executor_state.phase != "TRAINING"
        or type(executor_state.active_epoch) is not RegisteredTrainingEpoch
        or executor_state.process_id != os.getpid()
        or executor_state.optimizer_generation
        != (executor_state.zero_based_epoch + 1) * _UPDATES_PER_EPOCH
        or executor_state.runtime.optimizer_generation
        != executor_state.optimizer_generation
        or executor_state.runtime.expected != executor_state.expected
        or executor_state.runtime.model.training is not True
        or executor_state.continuation_reservation is not None
        or (
            executor_state.zero_based_epoch == 0
            and executor_state.previous_history_barrier is not None
        )
    ):
        raise Experiment002RegisteredExecutorError(
            "executor is not at an exact registered epoch handoff boundary"
        )
    result = object.__new__(RegisteredExecutorEpochHandoff)
    handoff_token = object()
    one_shot_token = object()
    ticket = _HandoffIssuanceTicket(
        handoff=result,
        executor_ticket=executor_state.ticket,
        handoff_token=object(),
    )
    anchor = _HandoffAnchor(
        handoff=result,
        executor=executor,
        executor_token=executor_state.token,
        handoff_token=handoff_token,
        route_marker=_REGISTERED_ROUTE,
        ticket=ticket,
    )
    state = _HandoffState(
        anchor=anchor,
        executor=executor,
        executor_token=executor_state.token,
        handoff_token=handoff_token,
        route_marker=_REGISTERED_ROUTE,
        ticket=ticket,
        registration=executor_state.registration,
        validation_inputs=executor_state.validation_inputs,
        process_id=executor_state.process_id,
        bridge_authority=executor_state.bridge_authority,
        bridge_session_token=executor_state.bridge_session_token,
        model=executor_state.runtime.model,
        population=population,
        epoch_trace=epoch_trace,
        seed=executor_state.seed,
        zero_based_epoch=executor_state.zero_based_epoch,
        optimizer_generation=executor_state.optimizer_generation,
        runtime_digests=replace(executor_state.expected),
        previous_history_barrier=executor_state.previous_history_barrier,
        one_shot_token=one_shot_token,
    )
    guard = _handoff_guard_from_state(state)
    lifecycle = _HandoffLifecycle(False, False)
    lifecycle_authority = _HandoffLifecycleAuthority(
        handoff=result,
        sequence=0,
        lifecycle=lifecycle,
        previous=None,
    )
    _truth_issue(result, ticket, lifecycle)
    with _HANDOFFS_LOCK:
        _HANDOFFS[result] = state
        _HANDOFF_GUARDS[result] = guard
        _HANDOFF_LIFECYCLES[result] = lifecycle
        _HANDOFF_LIFECYCLE_HEADS[result] = lifecycle_authority
        _ISSUED_HANDOFF_LIFECYCLES.add(lifecycle_authority)
        _HANDOFF_LIFECYCLE_HISTORY.append(lifecycle_authority)
        _HANDOFF_ANCHORS[result] = anchor
        _HANDOFF_ANCHOR_HISTORY.append(anchor)
        _ISSUED_HANDOFF_TICKETS.add(ticket)
        _HANDOFF_TICKET_HISTORY.append(ticket)
    return result


def _issued_executor_state(
    executor: RegisteredTrainingExecutor,
) -> _RegisteredExecutorState:
    if type(executor) is not RegisteredTrainingExecutor:
        raise TypeError("executor must be a RegisteredTrainingExecutor")
    with _EXECUTORS_LOCK:
        state = _EXECUTORS.get(executor)
        if type(state) is not _RegisteredExecutorState:
            ticket = _find_executor_ticket(executor)
            if ticket is not None:
                _FAILED_EXECUTORS.add(executor)
                _FAILED_EXECUTOR_HISTORY.add(executor)
                _record_executor_failure(ticket)
            raise Experiment002RegisteredExecutorError(
                "registered executor was not issued by this module"
            )
        return state


def _trusted_executor_lock(
    executor: RegisteredTrainingExecutor,
    state: _RegisteredExecutorState,
    _truth_ticket: Callable[
        [RegisteredTrainingExecutor, _ExecutorIssuanceTicket], bool
    ] = _AUTHORITY_TRUTH.validate_executor_ticket,
) -> threading.RLock:
    anchor = _find_executor_anchor(executor)
    guard = _EXECUTOR_GUARDS.get(executor)
    if (
        type(anchor) is not _RegisteredExecutorAnchor
        or state.anchor is not anchor
        or state.ticket is not anchor.ticket
        or not _valid_executor_ticket(executor, state.ticket)
        or not _truth_ticket(executor, state.ticket)
        or type(guard) is not _RegisteredExecutorGuard
        or guard.anchor is not anchor
        or guard.ticket is not state.ticket
        or type(anchor.lock) is not _RLOCK_TYPE
        or state.lock is not anchor.lock
        or guard.lock is not anchor.lock
    ):
        active_epoch = state.active_epoch
        owns_ticket = _truth_ticket(executor, state.ticket)
        _terminal_fail_executor(executor, state)
        if owns_ticket:
            _propagate_executor_failure(state, active_epoch)
        raise Experiment002RegisteredExecutorError(
            "registered executor lock authority changed"
        )
    return anchor.lock


def _validate_executor_or_fail(
    executor: RegisteredTrainingExecutor,
    state: _RegisteredExecutorState,
    _truth_ticket: Callable[
        [RegisteredTrainingExecutor, _ExecutorIssuanceTicket], bool
    ] = _AUTHORITY_TRUTH.validate_executor_ticket,
) -> None:
    try:
        _validate_executor_authority(executor, state)
    except BaseException:
        active_epoch = state.active_epoch
        owns_ticket = _truth_ticket(executor, state.ticket)
        _terminal_fail_executor(executor, state)
        if owns_ticket:
            _propagate_executor_failure(state, active_epoch)
        raise


def _validate_executor_authority(
    executor: RegisteredTrainingExecutor,
    state: _RegisteredExecutorState,
    _truth_ticket: Callable[
        [RegisteredTrainingExecutor, _ExecutorIssuanceTicket], bool
    ] = _AUTHORITY_TRUTH.validate_executor_ticket,
    _truth_lifecycle: Callable[
        [_ExecutorIssuanceTicket, _RegisteredExecutorLifecycle], bool
    ] = _AUTHORITY_TRUTH.validate_executor_lifecycle,
    _truth_failed: Callable[[_ExecutorIssuanceTicket], bool] = (
        _AUTHORITY_TRUTH.executor_failed
    ),
    _truth_continuation: Callable[
        [_ContinuationReservation, _ContinuationPhase], bool
    ] = _AUTHORITY_TRUTH.validate_continuation,
    _truth_finalization: Callable[
        [_FinalizationReservation, _FinalizationPhase], bool
    ] = _AUTHORITY_TRUTH.validate_finalization,
    _truth_finalized: Callable[
        [RegisteredExecutorEpochHandoff], _FinalizedHandoffBinding | None
    ] = _AUTHORITY_TRUTH.finalized_handoff,
) -> None:
    anchor = _find_executor_anchor(executor)
    guard = _EXECUTOR_GUARDS.get(executor)
    lifecycle = _EXECUTOR_LIFECYCLES.get(executor)
    _require_executor_payload_types(state, lifecycle, guard, anchor)
    continuation = state.continuation_reservation
    finalization = state.finalization_reservation
    continuation_phase_valid = False
    finalization_phase: _FinalizationPhase | None = None
    finalization_phase_valid = False
    finalized_binding: _FinalizedHandoffBinding | None = None
    visible_finalized_binding: _FinalizedHandoffBinding | None = None
    with _HANDOFFS_LOCK:
        if type(continuation) is _ContinuationReservation:
            observed_phase = _CONTINUATION_PHASES.get(continuation)
            if type(observed_phase) is str and observed_phase in (
                "RESERVED",
                "CONSUMED",
                "COMMITTED",
            ):
                continuation_phase_valid = _truth_continuation(
                    continuation,
                    cast(_ContinuationPhase, observed_phase),
                )
        if type(finalization) is _FinalizationReservation:
            observed_finalization_phase = _FINALIZATION_PHASES.get(finalization)
            if type(observed_finalization_phase) is str and (
                observed_finalization_phase
                in (
                    "RESERVED",
                    "TRACE_COMPLETE",
                    "HISTORY_COMPLETE",
                    "COMMITTED",
                )
            ):
                finalization_phase = cast(
                    _FinalizationPhase,
                    observed_finalization_phase,
                )
                finalization_phase_valid = _truth_finalization(
                    finalization,
                    finalization_phase,
                )
            finalized_binding = _truth_finalized(finalization.handoff)
            visible_finalized_binding = _FINALIZED_HANDOFFS.get(finalization.handoff)
            if finalization_phase_valid and finalization_phase is not None:
                _require_finalization_reservation(
                    finalization,
                    phase=finalization_phase,
                    _truth_validate=_truth_finalization,
                )
                if finalization_phase == "COMMITTED":
                    final_handoff_state = _HANDOFFS.get(finalization.handoff)
                    if type(final_handoff_state) is not _HandoffState:
                        raise Experiment002RegisteredExecutorError(
                            "finalized executor lost its handoff state"
                        )
                    _require_finalized_handoff_binding(
                        finalization.handoff,
                        final_handoff_state,
                        _truth_finalized=_truth_finalized,
                    )
    if (
        type(anchor) is not _RegisteredExecutorAnchor
        or anchor.executor is not executor
        or state.anchor is not anchor
        or state.ticket is not anchor.ticket
        or not _valid_executor_ticket(executor, state.ticket)
        or not _truth_ticket(executor, state.ticket)
        or state.ticket.registration is not state.registration
        or state.ticket.process_id != state.process_id
        or type(guard) is not _RegisteredExecutorGuard
        or guard.anchor is not anchor
        or guard.ticket is not state.ticket
        or guard != _guard_from_anchor(anchor)
        or state.token is not anchor.token
        or state.route_marker is not _REGISTERED_ROUTE
        or state.registration is not anchor.registration
        or state.process_id != anchor.process_id
        or state.process_id != os.getpid()
        or state.seed != anchor.seed
        or state.training_inputs is not anchor.training_inputs
        or state.validation_inputs is not anchor.validation_inputs
        or state.bridge_authority is not anchor.bridge_authority
        or state.bridge_session_token is not anchor.bridge_session_token
        or state.trace is not anchor.trace
        or state.runtime is not anchor.runtime
        or state.runtime.model is not anchor.model
        or state.runtime.optimizer is not anchor.optimizer
        or state.runtime.parameter_names != anchor.parameter_names
        or any(
            observed is not expected
            for observed, expected in zip(
                state.runtime.parameters, anchor.parameters, strict=True
            )
        )
        or state.lock is not anchor.lock
        or lifecycle != _lifecycle_from_state(state)
        or not _valid_executor_lifecycle(executor, state.ticket, lifecycle)
        or (
            type(lifecycle) is _RegisteredExecutorLifecycle
            and not _truth_lifecycle(state.ticket, lifecycle)
        )
        or state.runtime.seed != state.seed
        or state.runtime.route_marker is not _REGISTERED_ROUTE
        or state.runtime.lock is not state.lock
        or state.runtime.expected != state.expected
        or state.runtime.optimizer_generation != state.optimizer_generation
        or executor in _FAILED_EXECUTORS
        or executor in _FAILED_EXECUTOR_HISTORY
        or _executor_has_failure_event(state.ticket)
        or _truth_failed(state.ticket)
        or state.phase == "FAILED"
        or (
            continuation is not None
            and (
                type(continuation) is not _ContinuationReservation
                or state.phase != "AWAITING_EVALUATION"
                or state.active_handoff is not continuation.handoff
                or continuation.executor is not executor
                or not continuation_phase_valid
            )
        )
        or (state.phase in ("READY_TO_TRAIN", "TRAINING") and continuation is not None)
        or (continuation is not None and finalization is not None)
        or (
            finalization is not None
            and (
                type(finalization) is not _FinalizationReservation
                or state.phase not in ("AWAITING_EVALUATION", "COMPLETE")
                or finalization.executor is not executor
                or finalization.executor_ticket is not state.ticket
                or finalization.registration is not state.registration
                or finalization.validation_inputs is not state.validation_inputs
                or finalization.process_id != state.process_id
                or finalization.seed != state.seed
                or finalization.zero_based_epoch != state.zero_based_epoch
                or finalization.optimizer_generation != state.optimizer_generation
                or finalization.trace is not state.trace
                or finalization.runtime_digests != state.expected
                or finalization.previous_history_barrier
                is not state.previous_history_barrier
                or not finalization_phase_valid
                or (
                    state.phase == "AWAITING_EVALUATION"
                    and (
                        state.active_handoff is not finalization.handoff
                        or state.final_barrier is not None
                        or state.complete_update_trace is not None
                        or state.completed_history is not None
                    )
                )
                or (
                    state.phase == "COMPLETE"
                    and (
                        finalization_phase != "COMMITTED"
                        or type(finalized_binding) is not _FinalizedHandoffBinding
                        or visible_finalized_binding is not finalized_binding
                        or finalized_binding.reservation is not finalization
                        or state.active_epoch is not None
                        or state.active_transition is not None
                        or state.active_handoff is not None
                        or state.final_barrier is not finalization.final_barrier
                        or state.complete_update_trace
                        is not finalized_binding.complete_update_trace
                        or state.completed_history
                        is not finalized_binding.completed_history
                    )
                )
            )
        )
        or (
            finalization is None
            and (
                state.phase == "COMPLETE"
                or state.final_barrier is not None
                or state.complete_update_trace is not None
                or state.completed_history is not None
            )
        )
        or (
            state.phase == "COMPLETE"
            and (
                state.zero_based_epoch != _REGISTERED_EPOCH_COUNT - 1
                or state.optimizer_generation
                != _REGISTERED_EPOCH_COUNT * _UPDATES_PER_EPOCH
                or state.runtime.model.training is not False
            )
        )
    ):
        raise Experiment002RegisteredExecutorError(
            "registered executor authority changed or crossed a process"
        )
    bridge = _executor_session_snapshot(state.bridge_authority)
    if (
        bridge.session_token is not state.bridge_session_token
        or bridge.seed != state.seed
    ):
        raise Experiment002RegisteredExecutorError(
            "executor bridge session differs from local authority"
        )


def _require_executor_payload_types(
    state: _RegisteredExecutorState,
    lifecycle: _RegisteredExecutorLifecycle | None,
    guard: _RegisteredExecutorGuard | None,
    anchor: _RegisteredExecutorAnchor | None,
) -> None:
    from falsewake.experiment_002_registered_history import (
        RegisteredCompletedTrainingHistory,
    )

    if (
        type(state) is not _RegisteredExecutorState
        or type(state.phase) is not str
        or state.phase
        not in (
            "READY_TO_TRAIN",
            "TRAINING",
            "AWAITING_EVALUATION",
            "COMPLETE",
            "FAILED",
        )
        or type(state.process_id) is not int
        or type(state.seed) is not int
        or type(state.zero_based_epoch) is not int
        or type(state.optimizer_generation) is not int
        or not 0 <= state.zero_based_epoch < _REGISTERED_EPOCH_COUNT
        or not 0
        <= state.optimizer_generation
        <= _REGISTERED_EPOCH_COUNT * _UPDATES_PER_EPOCH
        or type(state.runtime.seed) is not int
        or type(state.runtime.optimizer_generation) is not int
        or type(state.ticket) is not _ExecutorIssuanceTicket
        or state.ticket.executor is not state.anchor.executor
        or type(state.ticket.token) is not object
        or type(state.ticket.registration) is not VerifiedRunRegistration
        or type(state.ticket.process_id) is not int
        or type(guard) is not _RegisteredExecutorGuard
        or type(guard.process_id) is not int
        or type(guard.seed) is not int
        or type(guard.registration) is not VerifiedRunRegistration
        or type(anchor) is not _RegisteredExecutorAnchor
        or type(anchor.process_id) is not int
        or type(anchor.seed) is not int
        or type(anchor.registration) is not VerifiedRunRegistration
        or type(lifecycle) is not _RegisteredExecutorLifecycle
        or type(lifecycle.phase) is not str
        or type(lifecycle.zero_based_epoch) is not int
        or type(lifecycle.optimizer_generation) is not int
        or (
            state.finalization_reservation is not None
            and type(state.finalization_reservation) is not _FinalizationReservation
        )
        or (
            state.final_barrier is not None
            and (
                type(state.finalization_reservation) is not _FinalizationReservation
                or type(state.final_barrier)
                is not state.finalization_reservation.barrier_type
            )
        )
        or (
            state.complete_update_trace is not None
            and type(state.complete_update_trace) is not CompleteUpdateTraceEvidence
        )
        or (
            state.completed_history is not None
            and type(state.completed_history) is not RegisteredCompletedTrainingHistory
        )
    ):
        raise Experiment002RegisteredExecutorError(
            "registered executor scalar payload types changed"
        )
    _require_runtime_digest_payload(state.expected, name="executor expected")
    _require_runtime_digest_payload(state.runtime.expected, name="runtime expected")
    _require_runtime_digest_payload(lifecycle.expected, name="executor lifecycle")


def _require_runtime_digest_payload(
    digests: _numeric._RuntimeDigests,
    *,
    name: str,
) -> None:
    if (
        type(digests) is not _numeric._RuntimeDigests
        or type(digests.model_sha256) is not str
        or type(digests.optimizer_sha256) is not str
        or type(digests.rng_sha256) is not str
        or type(digests.rng_state) is not bytes
        or not digests.rng_state
    ):
        raise Experiment002RegisteredExecutorError(
            f"{name} runtime digest payload types changed"
        )
    _require_lower_sha256(digests.model_sha256, f"{name} model_sha256")
    _require_lower_sha256(digests.optimizer_sha256, f"{name} optimizer_sha256")
    _require_lower_sha256(digests.rng_sha256, f"{name} rng_sha256")


def _terminal_fail_executor(
    executor: RegisteredTrainingExecutor,
    state: _RegisteredExecutorState,
    _truth_fail: Callable[[_ExecutorIssuanceTicket], None] = (
        _AUTHORITY_TRUTH.fail_executor
    ),
    _truth_validate: Callable[
        [RegisteredTrainingExecutor, _ExecutorIssuanceTicket], bool
    ] = _AUTHORITY_TRUTH.validate_executor_ticket,
) -> None:
    state.phase = "FAILED"
    state.active_epoch = None
    with contextlib.suppress(BaseException):
        _FAILED_EXECUTORS.add(executor)
    with contextlib.suppress(BaseException):
        _FAILED_EXECUTOR_HISTORY.add(executor)
    owns_ticket = _truth_validate(executor, state.ticket)
    if owns_ticket:
        with contextlib.suppress(BaseException):
            _truth_fail(state.ticket)
        with contextlib.suppress(BaseException):
            _record_executor_failure(state.ticket)
    if owns_ticket:
        with contextlib.suppress(BaseException):
            _force_publish_executor_lifecycle(executor, state)
    handoff = state.active_handoff
    if owns_ticket and handoff is not None:
        with contextlib.suppress(BaseException):
            handoff_state = _HANDOFFS.get(handoff)
            if (
                type(handoff_state) is _HandoffState
                and not handoff_state.failed
                and not _handoff_has_event(handoff_state.ticket, "FAILED")
            ):
                _terminal_fail_handoff(handoff, handoff_state)


def _propagate_executor_failure(
    state: _RegisteredExecutorState,
    epoch: RegisteredTrainingEpoch | None,
) -> None:
    transition = state.active_transition
    with contextlib.suppress(BaseException):
        _fail_optimizer_transition(transition)
    with contextlib.suppress(BaseException):
        _executor_fail_epoch(epoch, state.bridge_authority)
    with contextlib.suppress(BaseException):
        _fail_registered_trace(state.trace)
    with contextlib.suppress(BaseException):
        _fail_registered_executor_session(state.bridge_authority)


def _publish_executor_lifecycle(
    executor: RegisteredTrainingExecutor,
    state: _RegisteredExecutorState,
) -> None:
    lifecycle = _lifecycle_from_state(state)
    with _EXECUTORS_LOCK:
        if _EXECUTORS.get(executor) is not state:
            raise Experiment002RegisteredExecutorError(
                "registered executor state changed before lifecycle publication"
            )
        previous = _trusted_executor_lifecycle_head(executor)
        if previous is None or _EXECUTOR_LIFECYCLES.get(executor) != previous.lifecycle:
            raise Experiment002RegisteredExecutorError(
                "registered executor lifecycle lost append-only continuity"
            )
        _append_executor_lifecycle_locked(executor, state.ticket, lifecycle, previous)


def _force_publish_executor_lifecycle(
    executor: RegisteredTrainingExecutor,
    state: _RegisteredExecutorState,
    _truth_validate: Callable[
        [_ExecutorIssuanceTicket, _RegisteredExecutorLifecycle], bool
    ] = _AUTHORITY_TRUTH.validate_executor_lifecycle,
) -> None:
    lifecycle = _lifecycle_from_state(state)
    with _EXECUTORS_LOCK:
        previous = _trusted_executor_lifecycle_head(executor)
        if previous is None:
            return
        if previous.lifecycle == lifecycle:
            if not _truth_validate(state.ticket, lifecycle):
                return
            _EXECUTOR_LIFECYCLES[executor] = lifecycle
            _EXECUTOR_LIFECYCLE_HEADS[executor] = previous
            return
        _append_executor_lifecycle_locked(executor, state.ticket, lifecycle, previous)


def _append_executor_lifecycle_locked(
    executor: RegisteredTrainingExecutor,
    ticket: _ExecutorIssuanceTicket,
    lifecycle: _RegisteredExecutorLifecycle,
    previous: _RegisteredExecutorLifecycleAuthority,
    _truth_append: Callable[
        [
            _ExecutorIssuanceTicket,
            _RegisteredExecutorLifecycle,
            _RegisteredExecutorLifecycle,
        ],
        bool,
    ] = _AUTHORITY_TRUTH.append_executor_lifecycle,
) -> None:
    if not _truth_append(ticket, previous.lifecycle, lifecycle):
        raise Experiment002RegisteredExecutorError(
            "closure-owned executor lifecycle continuity changed"
        )
    authority = _RegisteredExecutorLifecycleAuthority(
        executor=executor,
        sequence=previous.sequence + 1,
        lifecycle=lifecycle,
        previous=previous,
    )
    _ISSUED_EXECUTOR_LIFECYCLES.add(authority)
    _EXECUTOR_LIFECYCLE_HISTORY.append(authority)
    _EXECUTOR_LIFECYCLE_HEADS[executor] = authority
    _EXECUTOR_LIFECYCLES[executor] = lifecycle


def _lifecycle_from_state(
    state: _RegisteredExecutorState,
) -> _RegisteredExecutorLifecycle:
    return _RegisteredExecutorLifecycle(
        phase=state.phase,
        zero_based_epoch=state.zero_based_epoch,
        optimizer_generation=state.optimizer_generation,
        expected=replace(state.expected),
        active_epoch=state.active_epoch,
        active_transition=state.active_transition,
        active_handoff=state.active_handoff,
        previous_history_barrier=state.previous_history_barrier,
        continuation_reservation=state.continuation_reservation,
        finalization_reservation=state.finalization_reservation,
        final_barrier=state.final_barrier,
        complete_update_trace=state.complete_update_trace,
        completed_history=state.completed_history,
    )


def _trusted_executor_lifecycle_head(
    executor: RegisteredTrainingExecutor,
) -> _RegisteredExecutorLifecycleAuthority | None:
    entries = [
        authority
        for authority in _EXECUTOR_LIFECYCLE_HISTORY
        if type(authority) is _RegisteredExecutorLifecycleAuthority
        and authority.executor is executor
    ]
    previous: _RegisteredExecutorLifecycleAuthority | None = None
    for sequence, authority in enumerate(entries):
        if (
            authority not in _ISSUED_EXECUTOR_LIFECYCLES
            or authority.sequence != sequence
            or authority.previous is not previous
        ):
            return None
        previous = authority
    return previous


def _valid_executor_lifecycle(
    executor: RegisteredTrainingExecutor,
    ticket: _ExecutorIssuanceTicket,
    lifecycle: _RegisteredExecutorLifecycle | None,
    _truth_validate: Callable[
        [_ExecutorIssuanceTicket, _RegisteredExecutorLifecycle], bool
    ] = _AUTHORITY_TRUTH.validate_executor_lifecycle,
) -> bool:
    head = _trusted_executor_lifecycle_head(executor)
    return (
        head is not None
        and _EXECUTOR_LIFECYCLE_HEADS.get(executor) is head
        and lifecycle == head.lifecycle
        and _truth_validate(ticket, head.lifecycle)
    )


def _find_executor_ticket(
    executor: RegisteredTrainingExecutor,
) -> _ExecutorIssuanceTicket | None:
    tickets = [
        ticket
        for ticket in _EXECUTOR_TICKET_HISTORY
        if type(ticket) is _ExecutorIssuanceTicket
        and ticket.executor is executor
        and ticket in _ISSUED_EXECUTOR_TICKETS
    ]
    unique: list[_ExecutorIssuanceTicket] = []
    for ticket in tickets:
        if not any(observed is ticket for observed in unique):
            unique.append(ticket)
    return unique[0] if len(unique) == 1 else None


def _valid_executor_ticket(
    executor: RegisteredTrainingExecutor,
    ticket: _ExecutorIssuanceTicket,
    _truth_validate: Callable[
        [RegisteredTrainingExecutor, _ExecutorIssuanceTicket], bool
    ] = _AUTHORITY_TRUTH.validate_executor_ticket,
) -> bool:
    return (
        type(ticket) is _ExecutorIssuanceTicket
        and ticket.executor is executor
        and ticket in _ISSUED_EXECUTOR_TICKETS
        and _find_executor_ticket(executor) is ticket
        and _truth_validate(executor, ticket)
    )


def _record_executor_failure(
    ticket: _ExecutorIssuanceTicket,
    _truth_fail: Callable[[_ExecutorIssuanceTicket], None] = (
        _AUTHORITY_TRUTH.fail_executor
    ),
) -> None:
    _truth_fail(ticket)
    if any(event.ticket is ticket for event in _EXECUTOR_FAILURE_EVENTS):
        return
    event = _ExecutorFailureEvent(ticket)
    _EXECUTOR_FAILURE_EVENTS.add(event)
    _EXECUTOR_FAILURE_EVENT_HISTORY.append(event)


def _executor_has_failure_event(
    ticket: _ExecutorIssuanceTicket,
    _truth_failed: Callable[[_ExecutorIssuanceTicket], bool] = (
        _AUTHORITY_TRUTH.executor_failed
    ),
) -> bool:
    set_events = [event for event in _EXECUTOR_FAILURE_EVENTS if event.ticket is ticket]
    history_events = [
        event for event in _EXECUTOR_FAILURE_EVENT_HISTORY if event.ticket is ticket
    ]
    return _truth_failed(ticket) or bool(set_events) or bool(history_events)


def _guard_from_anchor(anchor: _RegisteredExecutorAnchor) -> _RegisteredExecutorGuard:
    return _RegisteredExecutorGuard(
        anchor=anchor,
        ticket=anchor.ticket,
        token=anchor.token,
        route_marker=anchor.route_marker,
        registration=anchor.registration,
        process_id=anchor.process_id,
        seed=anchor.seed,
        training_inputs=anchor.training_inputs,
        validation_inputs=anchor.validation_inputs,
        bridge_authority=anchor.bridge_authority,
        bridge_session_token=anchor.bridge_session_token,
        trace=anchor.trace,
        runtime=anchor.runtime,
        model=anchor.model,
        optimizer=anchor.optimizer,
        parameter_names=anchor.parameter_names,
        parameters=anchor.parameters,
        lock=anchor.lock,
    )


def _find_executor_anchor(
    executor: RegisteredTrainingExecutor,
) -> _RegisteredExecutorAnchor | None:
    mapped = _EXECUTOR_ANCHORS.get(executor)
    candidates = [
        anchor
        for anchor in _EXECUTOR_ANCHOR_HISTORY
        if type(anchor) is _RegisteredExecutorAnchor and anchor.executor is executor
    ]
    if type(mapped) is _RegisteredExecutorAnchor:
        candidates.append(mapped)
    unique: list[_RegisteredExecutorAnchor] = []
    for candidate in candidates:
        if not any(observed is candidate for observed in unique):
            unique.append(candidate)
    return unique[0] if len(unique) == 1 else None


def _validate_handoff(
    handoff: RegisteredExecutorEpochHandoff,
    state: _HandoffState | None,
    _truth_ticket: Callable[
        [RegisteredExecutorEpochHandoff, _HandoffIssuanceTicket], bool
    ] = _AUTHORITY_TRUTH.validate_handoff_ticket,
    _truth_lifecycle: Callable[
        [_HandoffIssuanceTicket, _HandoffLifecycle], bool
    ] = _AUTHORITY_TRUTH.validate_handoff_lifecycle,
    _truth_event: Callable[
        [_HandoffIssuanceTicket, Literal["USED", "FAILED"]], bool
    ] = _AUTHORITY_TRUTH.handoff_has_event,
) -> None:
    anchor = _find_handoff_anchor(handoff)
    guard = _HANDOFF_GUARDS.get(handoff)
    lifecycle = _HANDOFF_LIFECYCLES.get(handoff)
    _require_handoff_payload_types(state, guard, lifecycle)
    if (
        type(state) is not _HandoffState
        or type(anchor) is not _HandoffAnchor
        or anchor.handoff is not handoff
        or state.anchor is not anchor
        or state.ticket is not anchor.ticket
        or not _valid_handoff_ticket(handoff, state.ticket)
        or not _truth_ticket(handoff, state.ticket)
        or not _valid_executor_ticket(state.executor, state.ticket.executor_ticket)
        or type(guard) is not _HandoffGuard
        or guard.anchor is not anchor
        or guard.ticket is not state.ticket
        or guard != _handoff_guard_from_state(state)
        or lifecycle != _HandoffLifecycle(state.used, state.failed)
        or not _valid_handoff_lifecycle(handoff, state.ticket, lifecycle)
        or (
            type(lifecycle) is _HandoffLifecycle
            and not _truth_lifecycle(state.ticket, lifecycle)
        )
        or state.used != _handoff_has_event(state.ticket, "USED")
        or state.used != _truth_event(state.ticket, "USED")
        or state.failed != _handoff_has_event(state.ticket, "FAILED")
        or state.failed != _truth_event(state.ticket, "FAILED")
        or state.executor_token is not anchor.executor_token
        or state.handoff_token is not anchor.handoff_token
        or state.route_marker is not _REGISTERED_ROUTE
        or state.process_id != os.getpid()
        or state.failed
        or handoff in _FAILED_HANDOFFS
        or handoff in _FAILED_HANDOFF_HISTORY
    ):
        raise Experiment002RegisteredExecutorError(
            "registered epoch handoff authority changed or crossed a process"
        )


def _require_handoff_payload_types(
    state: _HandoffState | None,
    guard: _HandoffGuard | None,
    lifecycle: _HandoffLifecycle | None,
) -> None:
    if (
        type(state) is not _HandoffState
        or type(guard) is not _HandoffGuard
        or type(lifecycle) is not _HandoffLifecycle
        or type(state.process_id) is not int
        or type(state.seed) is not int
        or type(state.zero_based_epoch) is not int
        or type(state.optimizer_generation) is not int
        or type(state.used) is not bool
        or type(state.failed) is not bool
        or type(guard.process_id) is not int
        or type(guard.seed) is not int
        or type(guard.zero_based_epoch) is not int
        or type(guard.optimizer_generation) is not int
        or type(lifecycle.used) is not bool
        or type(lifecycle.failed) is not bool
        or not 0 <= state.zero_based_epoch < _REGISTERED_EPOCH_COUNT
        or state.optimizer_generation
        != (state.zero_based_epoch + 1) * _UPDATES_PER_EPOCH
    ):
        raise Experiment002RegisteredExecutorError(
            "registered handoff scalar payload types changed"
        )
    _require_runtime_digest_payload(state.runtime_digests, name="handoff state")
    _require_runtime_digest_payload(guard.runtime_digests, name="handoff guard")


def _require_continuation_reservation(
    reservation: _ContinuationReservation,
    *,
    phase: _ContinuationPhase,
    _truth_validate: Callable[
        [_ContinuationReservation, _ContinuationPhase], bool
    ] = _AUTHORITY_TRUTH.validate_continuation,
) -> None:
    with _HANDOFFS_LOCK:
        observed = _CONTINUATION_PHASES.get(reservation)
        matches_history = (
            sum(item is reservation for item in _CONTINUATION_HISTORY) == 1
        )
        truth_matches = _truth_validate(reservation, phase)
    if (
        type(reservation) is not _ContinuationReservation
        or type(phase) is not str
        or phase not in ("RESERVED", "CONSUMED", "COMMITTED", "FAILED")
        or type(observed) is not str
        or observed != phase
        or not matches_history
        or not truth_matches
        or type(reservation.token) is not object
        or type(reservation.executor) is not RegisteredTrainingExecutor
        or type(reservation.executor_ticket) is not _ExecutorIssuanceTicket
        or type(reservation.handoff) is not RegisteredExecutorEpochHandoff
        or type(reservation.handoff_ticket) is not _HandoffIssuanceTicket
        or type(reservation.barrier_type) is not type
        or type(reservation.barrier) is not reservation.barrier_type
        or type(reservation.evaluated_epoch_type) is not type
        or type(reservation.evaluated_epoch) is not reservation.evaluated_epoch_type
        or type(reservation.registration) is not VerifiedRunRegistration
        or type(reservation.validation_inputs) is not RegisteredValidationInputs
        or type(reservation.process_id) is not int
        or type(reservation.seed) is not int
        or type(reservation.zero_based_epoch) is not int
        or type(reservation.optimizer_generation) is not int
        or type(reservation.evaluated_authority_sha256) is not str
        or type(reservation.runtime_digests) is not _numeric._RuntimeDigests
        or type(reservation.handoff_token) is not object
        or type(reservation.one_shot_token) is not object
    ):
        raise Experiment002RegisteredExecutorError(
            "continuation reservation authority changed"
        )
    _require_lower_sha256(
        reservation.evaluated_authority_sha256,
        "evaluated_authority_sha256",
    )
    runtime = reservation.runtime_digests
    if (
        type(runtime.model_sha256) is not str
        or type(runtime.optimizer_sha256) is not str
        or type(runtime.rng_sha256) is not str
        or type(runtime.rng_state) is not bytes
        or not runtime.rng_state
        or (
            reservation.zero_based_epoch == 0
            and reservation.previous_history_barrier is not None
        )
        or (
            reservation.zero_based_epoch > 0
            and type(reservation.previous_history_barrier)
            is not reservation.barrier_type
        )
    ):
        raise Experiment002RegisteredExecutorError(
            "continuation runtime or prior-barrier payload is invalid"
        )
    _require_lower_sha256(runtime.model_sha256, "model_sha256")
    _require_lower_sha256(runtime.optimizer_sha256, "optimizer_sha256")
    _require_lower_sha256(runtime.rng_sha256, "rng_sha256")
    if (
        reservation.executor_ticket.executor is not reservation.executor
        or reservation.handoff_ticket.handoff is not reservation.handoff
        or reservation.handoff_ticket.executor_ticket is not reservation.executor_ticket
        or reservation.process_id != os.getpid()
        or reservation.seed not in TRAINING_SEEDS
        or not 0 <= reservation.zero_based_epoch < _REGISTERED_EPOCH_COUNT - 1
        or reservation.optimizer_generation
        != (reservation.zero_based_epoch + 1) * _UPDATES_PER_EPOCH
    ):
        raise Experiment002RegisteredExecutorError(
            "continuation reservation payload is invalid"
        )


def _require_finalization_reservation(
    reservation: _FinalizationReservation,
    *,
    phase: _FinalizationPhase,
    _truth_validate: Callable[
        [_FinalizationReservation, _FinalizationPhase], bool
    ] = _AUTHORITY_TRUTH.validate_finalization,
) -> None:
    with _HANDOFFS_LOCK:
        observed = _FINALIZATION_PHASES.get(reservation)
        handoff_state = _HANDOFFS.get(reservation.handoff)
        matches_history = (
            sum(item is reservation for item in _FINALIZATION_HISTORY) == 1
        )
        truth_matches = _truth_validate(reservation, phase)
    if (
        type(reservation) is not _FinalizationReservation
        or type(phase) is not str
        or phase
        not in (
            "RESERVED",
            "TRACE_COMPLETE",
            "HISTORY_COMPLETE",
            "COMMITTED",
            "FAILED",
        )
        or type(observed) is not str
        or observed != phase
        or not matches_history
        or not truth_matches
        or type(reservation.token) is not object
        or type(reservation.executor) is not RegisteredTrainingExecutor
        or type(reservation.executor_ticket) is not _ExecutorIssuanceTicket
        or type(reservation.handoff) is not RegisteredExecutorEpochHandoff
        or type(reservation.handoff_ticket) is not _HandoffIssuanceTicket
        or type(reservation.barrier_type) is not type
        or type(reservation.final_barrier) is not reservation.barrier_type
        or type(reservation.history_type) is not type
        or type(reservation.history) is not reservation.history_type
        or type(reservation.evaluated_epoch_type) is not type
        or type(reservation.evaluated_epoch) is not reservation.evaluated_epoch_type
        or type(reservation.evaluated_authority_sha256) is not str
        or type(reservation.registration) is not VerifiedRunRegistration
        or type(reservation.validation_inputs) is not RegisteredValidationInputs
        or type(reservation.process_id) is not int
        or type(reservation.seed) is not int
        or type(reservation.zero_based_epoch) is not int
        or type(reservation.optimizer_generation) is not int
        or type(reservation.runtime_digests) is not _numeric._RuntimeDigests
        or type(reservation.previous_history_barrier) is not reservation.barrier_type
        or reservation.previous_history_barrier is reservation.final_barrier
        or type(reservation.handoff_token) is not object
        or type(reservation.one_shot_token) is not object
        or type(reservation.trace) is not UpdateTraceAccumulator
        or type(reservation.epoch_traces) is not tuple
        or len(reservation.epoch_traces) != _REGISTERED_EPOCH_COUNT
        or any(
            type(epoch_trace) is not EpochUpdateTraceEvidence
            for epoch_trace in reservation.epoch_traces
        )
    ):
        raise Experiment002RegisteredExecutorError(
            "finalization reservation authority changed"
        )
    _require_lower_sha256(
        reservation.evaluated_authority_sha256,
        "evaluated_authority_sha256",
    )
    _require_runtime_digest_payload(
        reservation.runtime_digests,
        name="finalization reservation",
    )
    if (
        reservation.executor_ticket.executor is not reservation.executor
        or reservation.handoff_ticket.handoff is not reservation.handoff
        or reservation.handoff_ticket.executor_ticket is not reservation.executor_ticket
        or reservation.process_id != os.getpid()
        or reservation.seed not in TRAINING_SEEDS
        or reservation.zero_based_epoch != _REGISTERED_EPOCH_COUNT - 1
        or reservation.optimizer_generation
        != _REGISTERED_EPOCH_COUNT * _UPDATES_PER_EPOCH
        or reservation.epoch_traces[-1]
        is not getattr(handoff_state, "epoch_trace", None)
    ):
        raise Experiment002RegisteredExecutorError(
            "finalization reservation payload is invalid"
        )


def _require_retired_handoff_binding(
    handoff: RegisteredExecutorEpochHandoff,
    state: _HandoffState,
    *,
    _truth_retired: Callable[
        [RegisteredExecutorEpochHandoff], _RetiredHandoffBinding | None
    ] = _AUTHORITY_TRUTH.retired_handoff,
) -> _RetiredHandoffBinding:
    binding = _RETIRED_HANDOFFS.get(handoff)
    truth = _truth_retired(handoff)
    if (
        type(binding) is not _RetiredHandoffBinding
        or truth is not binding
        or type(binding.reservation) is not _ContinuationReservation
        or binding.executor is not state.executor
        or binding.handoff is not handoff
        or binding.reservation.handoff is not handoff
        or binding.barrier is not binding.reservation.barrier
        or binding.evaluated_epoch is not binding.reservation.evaluated_epoch
        or binding.evaluated_authority_sha256
        != binding.reservation.evaluated_authority_sha256
        or binding.registration is not state.registration
        or binding.registration is not binding.reservation.registration
        or binding.validation_inputs is not state.validation_inputs
        or binding.validation_inputs is not binding.reservation.validation_inputs
        or type(binding.process_id) is not int
        or binding.process_id != state.process_id
        or binding.process_id != os.getpid()
        or type(binding.seed) is not int
        or binding.seed != state.seed
        or type(binding.zero_based_epoch) is not int
        or binding.zero_based_epoch != state.zero_based_epoch
        or type(binding.optimizer_generation) is not int
        or binding.optimizer_generation != state.optimizer_generation
        or type(binding.runtime_digests) is not _numeric._RuntimeDigests
        or binding.runtime_digests != state.runtime_digests
        or binding.previous_history_barrier is not state.previous_history_barrier
        or binding.handoff_token is not state.handoff_token
        or binding.one_shot_token is not state.one_shot_token
        or not state.used
        or state.failed
    ):
        raise Experiment002RegisteredExecutorError("retired handoff authority changed")
    _require_continuation_reservation(binding.reservation, phase="COMMITTED")
    _require_lower_sha256(
        binding.evaluated_authority_sha256,
        "evaluated_authority_sha256",
    )
    return binding


def _require_finalized_handoff_binding(
    handoff: RegisteredExecutorEpochHandoff,
    state: _HandoffState,
    *,
    _truth_finalized: Callable[
        [RegisteredExecutorEpochHandoff], _FinalizedHandoffBinding | None
    ] = _AUTHORITY_TRUTH.finalized_handoff,
) -> _FinalizedHandoffBinding:
    binding = _FINALIZED_HANDOFFS.get(handoff)
    truth = _truth_finalized(handoff)
    if (
        type(binding) is not _FinalizedHandoffBinding
        or truth is not binding
        or type(binding.reservation) is not _FinalizationReservation
        or binding.executor is not state.executor
        or binding.handoff is not handoff
        or binding.reservation.handoff is not handoff
        or binding.final_barrier is not binding.reservation.final_barrier
        or binding.history is not binding.reservation.history
        or binding.evaluated_epoch is not binding.reservation.evaluated_epoch
        or binding.evaluated_authority_sha256
        != binding.reservation.evaluated_authority_sha256
        or type(binding.complete_update_trace_type) is not type
        or type(binding.complete_update_trace) is not binding.complete_update_trace_type
        or type(binding.completed_history_type) is not type
        or type(binding.completed_history) is not binding.completed_history_type
        or binding.registration is not state.registration
        or binding.registration is not binding.reservation.registration
        or binding.validation_inputs is not state.validation_inputs
        or binding.validation_inputs is not binding.reservation.validation_inputs
        or type(binding.process_id) is not int
        or binding.process_id != state.process_id
        or binding.process_id != os.getpid()
        or type(binding.seed) is not int
        or binding.seed != state.seed
        or type(binding.zero_based_epoch) is not int
        or binding.zero_based_epoch != _REGISTERED_EPOCH_COUNT - 1
        or binding.zero_based_epoch != state.zero_based_epoch
        or type(binding.optimizer_generation) is not int
        or binding.optimizer_generation != state.optimizer_generation
        or type(binding.runtime_digests) is not _numeric._RuntimeDigests
        or binding.runtime_digests != state.runtime_digests
        or binding.previous_history_barrier is not state.previous_history_barrier
        or binding.handoff_token is not state.handoff_token
        or binding.one_shot_token is not state.one_shot_token
        or not state.used
        or state.failed
    ):
        raise Experiment002RegisteredExecutorError(
            "finalized handoff authority changed"
        )
    _require_finalization_reservation(binding.reservation, phase="COMMITTED")
    _require_lower_sha256(
        binding.evaluated_authority_sha256,
        "evaluated_authority_sha256",
    )
    return binding


def _exact_retired_handoff_won_snapshot_race(
    handoff: RegisteredExecutorEpochHandoff,
    state: _HandoffState,
    *,
    _truth_retired: Callable[
        [RegisteredExecutorEpochHandoff], _RetiredHandoffBinding | None
    ] = _AUTHORITY_TRUTH.retired_handoff,
    _truth_finalized: Callable[
        [RegisteredExecutorEpochHandoff], _FinalizedHandoffBinding | None
    ] = _AUTHORITY_TRUTH.finalized_handoff,
) -> bool:
    """Return true only for exact terminal truth published during a snapshot."""

    with _HANDOFFS_LOCK:
        if _HANDOFFS.get(handoff) is not state:
            raise Experiment002RegisteredExecutorError(
                "handoff state changed during retired snapshot recheck"
            )
        _validate_handoff(handoff, state)
        retired_truth = _truth_retired(handoff)
        retired_visible = _RETIRED_HANDOFFS.get(handoff)
        finalized_truth = _truth_finalized(handoff)
        finalized_visible = _FINALIZED_HANDOFFS.get(handoff)
        retired_present = retired_truth is not None or handoff in _RETIRED_HANDOFFS
        finalized_present = (
            finalized_truth is not None or handoff in _FINALIZED_HANDOFFS
        )
        if retired_present and finalized_present:
            raise Experiment002RegisteredExecutorError(
                "handoff snapshot race found conflicting terminal truth"
            )
        if not retired_present and not finalized_present:
            return False
        if retired_present:
            binding = _require_retired_handoff_binding(
                handoff,
                state,
                _truth_retired=_truth_retired,
            )
            exact = retired_truth is binding and retired_visible is binding
        else:
            finalized_binding = _require_finalized_handoff_binding(
                handoff,
                state,
                _truth_finalized=_truth_finalized,
            )
            exact = (
                finalized_truth is finalized_binding
                and finalized_visible is finalized_binding
            )
        if not exact:
            raise Experiment002RegisteredExecutorError(
                "terminal handoff race lost exact authority"
            )
        return True


def _terminal_fail_handoff(
    handoff: RegisteredExecutorEpochHandoff,
    state: _HandoffState,
    _truth_validate: Callable[
        [RegisteredExecutorEpochHandoff, _HandoffIssuanceTicket], bool
    ] = _AUTHORITY_TRUTH.validate_handoff_ticket,
    _truth_record: Callable[
        [_HandoffIssuanceTicket, Literal["USED", "FAILED"]], None
    ] = _AUTHORITY_TRUTH.record_handoff_event,
) -> None:
    owns_ticket = _truth_validate(handoff, state.ticket)
    if owns_ticket:
        with contextlib.suppress(BaseException):
            _truth_record(state.ticket, "FAILED")
        with contextlib.suppress(BaseException):
            _record_handoff_event(state.ticket, "FAILED")
    state.failed = True
    with contextlib.suppress(BaseException):
        _FAILED_HANDOFFS.add(handoff)
    with contextlib.suppress(BaseException):
        _FAILED_HANDOFF_HISTORY.add(handoff)
    if owns_ticket:
        with contextlib.suppress(BaseException):
            _force_publish_handoff_lifecycle(handoff, state)


def _publish_handoff_lifecycle(
    handoff: RegisteredExecutorEpochHandoff, state: _HandoffState
) -> None:
    lifecycle = _HandoffLifecycle(state.used, state.failed)
    with _HANDOFFS_LOCK:
        previous = _trusted_handoff_lifecycle_head(handoff)
        if previous is None or _HANDOFF_LIFECYCLES.get(handoff) != previous.lifecycle:
            raise Experiment002RegisteredExecutorError(
                "registered handoff lifecycle lost append-only continuity"
            )
        _append_handoff_lifecycle_locked(handoff, state.ticket, lifecycle, previous)


def _force_publish_handoff_lifecycle(
    handoff: RegisteredExecutorEpochHandoff,
    state: _HandoffState,
    _truth_validate: Callable[
        [_HandoffIssuanceTicket, _HandoffLifecycle], bool
    ] = _AUTHORITY_TRUTH.validate_handoff_lifecycle,
) -> None:
    lifecycle = _HandoffLifecycle(state.used, state.failed)
    with _HANDOFFS_LOCK:
        previous = _trusted_handoff_lifecycle_head(handoff)
        if previous is None:
            return
        if previous.lifecycle == lifecycle:
            if not _truth_validate(state.ticket, lifecycle):
                return
            _HANDOFF_LIFECYCLES[handoff] = lifecycle
            _HANDOFF_LIFECYCLE_HEADS[handoff] = previous
            return
        _append_handoff_lifecycle_locked(handoff, state.ticket, lifecycle, previous)


def _append_handoff_lifecycle_locked(
    handoff: RegisteredExecutorEpochHandoff,
    ticket: _HandoffIssuanceTicket,
    lifecycle: _HandoffLifecycle,
    previous: _HandoffLifecycleAuthority,
    _truth_append: Callable[
        [_HandoffIssuanceTicket, _HandoffLifecycle, _HandoffLifecycle], bool
    ] = _AUTHORITY_TRUTH.append_handoff_lifecycle,
) -> None:
    if not _truth_append(ticket, previous.lifecycle, lifecycle):
        raise Experiment002RegisteredExecutorError(
            "closure-owned handoff lifecycle continuity changed"
        )
    authority = _HandoffLifecycleAuthority(
        handoff=handoff,
        sequence=previous.sequence + 1,
        lifecycle=lifecycle,
        previous=previous,
    )
    _ISSUED_HANDOFF_LIFECYCLES.add(authority)
    _HANDOFF_LIFECYCLE_HISTORY.append(authority)
    _HANDOFF_LIFECYCLE_HEADS[handoff] = authority
    _HANDOFF_LIFECYCLES[handoff] = lifecycle


def _trusted_handoff_lifecycle_head(
    handoff: RegisteredExecutorEpochHandoff,
) -> _HandoffLifecycleAuthority | None:
    entries = [
        authority
        for authority in _HANDOFF_LIFECYCLE_HISTORY
        if type(authority) is _HandoffLifecycleAuthority
        and authority.handoff is handoff
    ]
    previous: _HandoffLifecycleAuthority | None = None
    for sequence, authority in enumerate(entries):
        if (
            authority not in _ISSUED_HANDOFF_LIFECYCLES
            or authority.sequence != sequence
            or authority.previous is not previous
        ):
            return None
        previous = authority
    return previous


def _valid_handoff_lifecycle(
    handoff: RegisteredExecutorEpochHandoff,
    ticket: _HandoffIssuanceTicket,
    lifecycle: _HandoffLifecycle | None,
    _truth_validate: Callable[
        [_HandoffIssuanceTicket, _HandoffLifecycle], bool
    ] = _AUTHORITY_TRUTH.validate_handoff_lifecycle,
) -> bool:
    head = _trusted_handoff_lifecycle_head(handoff)
    return (
        head is not None
        and _HANDOFF_LIFECYCLE_HEADS.get(handoff) is head
        and lifecycle == head.lifecycle
        and _truth_validate(ticket, head.lifecycle)
    )


def _find_handoff_ticket(
    handoff: RegisteredExecutorEpochHandoff,
) -> _HandoffIssuanceTicket | None:
    tickets = [
        ticket
        for ticket in _HANDOFF_TICKET_HISTORY
        if type(ticket) is _HandoffIssuanceTicket
        and ticket.handoff is handoff
        and ticket in _ISSUED_HANDOFF_TICKETS
    ]
    unique: list[_HandoffIssuanceTicket] = []
    for ticket in tickets:
        if not any(observed is ticket for observed in unique):
            unique.append(ticket)
    return unique[0] if len(unique) == 1 else None


def _valid_handoff_ticket(
    handoff: RegisteredExecutorEpochHandoff,
    ticket: _HandoffIssuanceTicket,
    _truth_validate: Callable[
        [RegisteredExecutorEpochHandoff, _HandoffIssuanceTicket], bool
    ] = _AUTHORITY_TRUTH.validate_handoff_ticket,
) -> bool:
    return (
        type(ticket) is _HandoffIssuanceTicket
        and ticket.handoff is handoff
        and ticket in _ISSUED_HANDOFF_TICKETS
        and _find_handoff_ticket(handoff) is ticket
        and _truth_validate(handoff, ticket)
    )


def _record_handoff_event(
    ticket: _HandoffIssuanceTicket,
    kind: Literal["USED", "FAILED"],
    _truth_record: Callable[
        [_HandoffIssuanceTicket, Literal["USED", "FAILED"]], None
    ] = _AUTHORITY_TRUTH.record_handoff_event,
) -> None:
    _truth_record(ticket, kind)
    if any(event.ticket is ticket and event.kind == kind for event in _HANDOFF_EVENTS):
        return
    event = _HandoffLifecycleEvent(ticket, kind)
    _HANDOFF_EVENTS.add(event)
    _HANDOFF_EVENT_HISTORY.append(event)


def _handoff_has_event(
    ticket: _HandoffIssuanceTicket,
    kind: Literal["USED", "FAILED"],
    _truth_has: Callable[
        [_HandoffIssuanceTicket, Literal["USED", "FAILED"]], bool
    ] = _AUTHORITY_TRUTH.handoff_has_event,
) -> bool:
    return (
        _truth_has(ticket, kind)
        or any(
            event.ticket is ticket and event.kind == kind for event in _HANDOFF_EVENTS
        )
        or any(
            event.ticket is ticket and event.kind == kind
            for event in _HANDOFF_EVENT_HISTORY
        )
    )


def _poison_handoff_and_executor(
    handoff: RegisteredExecutorEpochHandoff,
    state: _HandoffState,
    _truth_validate: Callable[
        [RegisteredExecutorEpochHandoff, _HandoffIssuanceTicket], bool
    ] = _AUTHORITY_TRUTH.validate_handoff_ticket,
) -> None:
    owns_ticket = _truth_validate(handoff, state.ticket)
    with _HANDOFFS_LOCK:
        _terminal_fail_handoff(handoff, state)
    if not owns_ticket:
        return
    try:
        executor_state = _issued_executor_state(state.executor)
        lock = _trusted_executor_lock(state.executor, executor_state)
    except BaseException:
        return
    with lock:
        _terminal_fail_executor(state.executor, executor_state)
        _propagate_executor_failure(executor_state, None)


def _handoff_guard_from_state(state: _HandoffState) -> _HandoffGuard:
    return _HandoffGuard(
        anchor=state.anchor,
        ticket=state.ticket,
        executor=state.executor,
        executor_token=state.executor_token,
        handoff_token=state.handoff_token,
        route_marker=state.route_marker,
        registration=state.registration,
        validation_inputs=state.validation_inputs,
        process_id=state.process_id,
        bridge_authority=state.bridge_authority,
        bridge_session_token=state.bridge_session_token,
        model=state.model,
        population=state.population,
        epoch_trace=state.epoch_trace,
        seed=state.seed,
        zero_based_epoch=state.zero_based_epoch,
        optimizer_generation=state.optimizer_generation,
        runtime_digests=replace(state.runtime_digests),
        previous_history_barrier=state.previous_history_barrier,
        one_shot_token=state.one_shot_token,
    )


def _find_handoff_anchor(
    handoff: RegisteredExecutorEpochHandoff,
) -> _HandoffAnchor | None:
    mapped = _HANDOFF_ANCHORS.get(handoff)
    candidates = [
        anchor
        for anchor in _HANDOFF_ANCHOR_HISTORY
        if type(anchor) is _HandoffAnchor and anchor.handoff is handoff
    ]
    if type(mapped) is _HandoffAnchor:
        candidates.append(mapped)
    unique: list[_HandoffAnchor] = []
    for candidate in candidates:
        if not any(observed is candidate for observed in unique):
            unique.append(candidate)
    return unique[0] if len(unique) == 1 else None


def _verify_bound_inputs(state: _RegisteredExecutorState) -> None:
    if _issued_source_state(state.training_inputs) is not _issued_source_state(
        state.training_inputs
    ):
        raise Experiment002RegisteredExecutorError(
            "registered training input authority changed during verification"
        )
    verify_registered_validation_inputs(state.validation_inputs)


def _validate_registered_logits(logits: Tensor, *, batch_size: int) -> None:
    if (
        type(logits) is not Tensor
        or tuple(logits.shape)
        != (batch_size, MODEL_RECEPTIVE_FIELD_FRAMES, OUTPUT_CLASSES)
        or logits.device.type != "cpu"
        or logits.dtype != torch.float32
        or not bool(torch.isfinite(logits).all().item())
    ):
        raise Experiment002RegisteredExecutorError(
            "registered model logits are invalid"
        )


def _validate_registered_stream_state(
    state: CausalKWSState,
    *,
    batch_size: int,
    expected_frames: int,
    require_positive_zero: bool,
) -> None:
    if type(state) is not CausalKWSState or len(state.depthwise_states) != len(
        BLOCK_DILATIONS
    ):
        raise Experiment002RegisteredExecutorError(
            "registered stream state type or count is invalid"
        )
    for tensor, dilation in zip(state.depthwise_states, BLOCK_DILATIONS, strict=True):
        if (
            type(tensor) is not Tensor
            or tuple(tensor.shape)
            != (
                batch_size,
                ENCODER_CHANNELS,
                (DEPTHWISE_KERNEL_SIZE - 1) * dilation,
            )
            or tensor.device.type != "cpu"
            or tensor.dtype != torch.float32
            or not tensor.is_contiguous()
            or not bool(torch.isfinite(tensor).all().item())
            or (
                require_positive_zero
                and (
                    bool(torch.count_nonzero(tensor).item())
                    or bool(torch.signbit(tensor).any().item())
                )
            )
        ):
            raise Experiment002RegisteredExecutorError(
                "registered depthwise stream state is invalid"
            )
    pool = state.pool_state
    if (
        type(pool) is not Tensor
        or tuple(pool.shape) != (batch_size, ENCODER_CHANNELS, POOL_HISTORY_FRAMES)
        or pool.device.type != "cpu"
        or pool.dtype != torch.float32
        or not pool.is_contiguous()
        or not bool(torch.isfinite(pool).all().item())
        or (
            require_positive_zero
            and (
                bool(torch.count_nonzero(pool).item())
                or bool(torch.signbit(pool).any().item())
            )
        )
        or type(state.frames_seen) is not Tensor
        or tuple(state.frames_seen.shape) != (batch_size,)
        or state.frames_seen.device.type != "cpu"
        or state.frames_seen.dtype != torch.int64
        or not state.frames_seen.is_contiguous()
        or not bool(torch.all(state.frames_seen == expected_frames).item())
    ):
        raise Experiment002RegisteredExecutorError(
            "registered pool or frame-count stream state is invalid"
        )


def _stream_state_payload(state: CausalKWSState) -> bytes:
    tensors = (*state.depthwise_states, state.pool_state, state.frames_seen)
    return b"".join(tensor.detach().numpy().tobytes(order="C") for tensor in tensors)


def _require_registered_seed(seed: object) -> None:
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    if seed not in TRAINING_SEEDS:
        raise Experiment002RegisteredExecutorError(
            "seed is not one of the registered training seeds"
        )


def _require_process_id(process_id: object) -> None:
    if type(process_id) is not int:
        raise TypeError("process_id must be an integer")
    if not 1 <= process_id <= _UINT32_MAX:
        raise Experiment002RegisteredExecutorError(
            "process_id must be a positive UINT32"
        )


# Critical producers, validators, transitions, and poison paths captured the
# closure-owned routes in their function defaults.  Remove the temporary
# module aliases so callers cannot invoke or replace an issuer directly.
del _AUTHORITY_TRUTH
del _build_authority_truth
