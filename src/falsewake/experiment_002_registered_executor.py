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
from typing import Final, Literal, NoReturn, cast

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
    EpochUpdateTraceEvidence,
    UpdateTraceAccumulator,
    _fail_registered_trace,
    registered_learning_rate,
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

type RegisteredExecutorPhase = Literal[
    "READY_TO_TRAIN", "TRAINING", "AWAITING_EVALUATION", "FAILED"
]

_REGISTERED_ROUTE: Final = object()
_UPDATES_PER_EPOCH: Final = REGISTERED_BATCH_COUNT
_REGISTERED_EPOCH_COUNT: Final = 30
_MAX_GRADIENT_NORM: Final = 5.0
_UINT32_MAX: Final = (1 << 32) - 1
_RLOCK_TYPE: Final = type(threading.RLock())


class Experiment002RegisteredExecutorError(ValueError):
    """The source-bound registered executor violated its authority contract."""


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


def _build_authority_truth() -> _AuthorityTruthRoutes:
    """Create closure-owned issuance and monotonic lifecycle truth."""

    executor_tickets: dict[
        RegisteredTrainingExecutor,
        tuple[_ExecutorIssuanceTicket, object, VerifiedRunRegistration, int],
    ] = {}
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
    lock = threading.RLock()

    def executor_frame(lifecycle: _RegisteredExecutorLifecycle) -> tuple[object, ...]:
        expected = lifecycle.expected
        return (
            lifecycle.phase,
            lifecycle.zero_based_epoch,
            lifecycle.optimizer_generation,
            expected.model_sha256,
            expected.optimizer_sha256,
            expected.rng_sha256,
            expected.rng_state,
            lifecycle.active_epoch,
            lifecycle.active_transition,
            lifecycle.active_handoff,
            lifecycle.previous_history_barrier,
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
            executor_tickets[executor] = (
                ticket,
                ticket.token,
                ticket.registration,
                ticket.process_id,
            )
            executor_lifecycles[ticket] = [executor_frame(lifecycle)]

    def validate_executor_ticket(
        executor: RegisteredTrainingExecutor, ticket: _ExecutorIssuanceTicket
    ) -> bool:
        with lock:
            frame = executor_tickets.get(executor)
            return (
                frame
                == (
                    ticket,
                    ticket.token,
                    ticket.registration,
                    ticket.process_id,
                )
                and ticket.executor is executor
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
        except BaseException:
            _poison_handoff_and_executor(handoff, state)
            raise
        _poison_handoff_and_executor(handoff, state)


def _access_registered_epoch_handoff(
    handoff: RegisteredExecutorEpochHandoff,
    *,
    registration: VerifiedRunRegistration | None,
    consume: bool,
    _truth_record: Callable[
        [_HandoffIssuanceTicket, Literal["USED", "FAILED"]], None
    ] = _AUTHORITY_TRUTH.record_handoff_event,
) -> _HandoffState:
    """Validate a handoff against its awaiting executor and optionally claim it."""

    if type(handoff) is not RegisteredExecutorEpochHandoff:
        raise TypeError("handoff must be a RegisteredExecutorEpochHandoff")
    if registration is not None and type(registration) is not VerifiedRunRegistration:
        raise TypeError("registration must be a VerifiedRunRegistration")
    if type(consume) is not bool:
        raise TypeError("consume must be a bool")
    with _HANDOFFS_LOCK:
        state = _HANDOFFS.get(handoff)
        try:
            _validate_handoff(handoff, state)
        except BaseException:
            if type(state) is _HandoffState:
                _poison_handoff_and_executor(handoff, state)
            raise
        assert state is not None
        if registration is not None and state.registration is not registration:
            raise Experiment002RegisteredExecutorError(
                "handoff registration identity differs"
            )
        if state.failed or (consume and state.used):
            raise Experiment002RegisteredExecutorError(
                "registered epoch handoff was already consumed or failed"
            )
        reverify_verified_run_registration(state.registration)
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
                _terminal_fail_handoff(handoff, state)
                _terminal_fail_executor(state.executor, executor_state)
                _propagate_executor_failure(executor_state, None)
                raise Experiment002RegisteredExecutorError(
                    "handoff no longer matches its awaiting executor"
                )
            observed = _stable_handoff_runtime_digests(
                executor_state.runtime,
                expected_generation=state.optimizer_generation,
                expected_training=expected_training,
            )
            if observed != state.runtime_digests:
                _terminal_fail_handoff(handoff, state)
                _terminal_fail_executor(state.executor, executor_state)
                _propagate_executor_failure(executor_state, None)
                raise Experiment002RegisteredExecutorError(
                    "handoff runtime changed before evaluation"
                )
            if consume:
                _truth_record(state.ticket, "USED")
                _record_handoff_event(state.ticket, "USED")
                state.used = True
                _publish_handoff_lifecycle(handoff, state)
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
) -> None:
    anchor = _find_executor_anchor(executor)
    guard = _EXECUTOR_GUARDS.get(executor)
    lifecycle = _EXECUTOR_LIFECYCLES.get(executor)
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
            if type(handoff_state) is _HandoffState:
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
