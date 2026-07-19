from __future__ import annotations

import copy
import inspect
import os
import pickle
import threading
import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

import falsewake.experiment_002_registered_evaluator as registered_evaluator
import falsewake.experiment_002_registered_executor as registered
import falsewake.experiment_002_registered_history as registered_history
import falsewake.experiment_002_training_bridge as bridge
import falsewake.experiment_002_training_executor as numeric
from falsewake.causal_kws import CausalKWS
from falsewake.experiment_002_data import TRAINING_SEEDS
from falsewake.experiment_002_evidence import (
    RegisteredValidationEvidence,
    RegisteredValidationInputs,
)
from falsewake.experiment_002_registered_evaluator import RegisteredEvaluatedEpoch
from falsewake.experiment_002_registered_history import (
    RegisteredCompletedTrainingHistory,
    RegisteredHistoryBarrier,
    RegisteredTrainingHistory,
)
from falsewake.experiment_002_run_authority import VerifiedRunRegistration
from falsewake.experiment_002_training_bridge import (
    RegisteredExecutorSessionAuthority,
    RegisteredOptimizerTransition,
    TraceConsumedTransition,
)
from falsewake.experiment_002_training_evidence import (
    CompleteUpdateTraceEvidence,
    EpochUpdateTraceEvidence,
    RegisteredModelTensorEvidence,
    UpdateTraceAccumulator,
)
from falsewake.experiment_002_training_population import (
    CompletedTrainingPopulation,
    CompletedTrainingUpdate,
    MaterializedTrainingBatch,
    RegisteredTrainingEpoch,
    RegisteredTrainingInputSource,
    _RegisteredBatchBinding,
)


class _IntSubclass(int):
    pass


class _StrSubclass(str):
    pass


class _CounterfeitTrace:
    def __init__(self) -> None:
        self.epoch_trace = object.__new__(EpochUpdateTraceEvidence)
        self.completed_epochs = 0
        self.next_update = 0

    @property
    def completed_epoch_count(self) -> int:
        return self.completed_epochs

    @property
    def next_global_update(self) -> int:
        return self.next_update

    def _finish_registered_epoch(
        self, population: CompletedTrainingPopulation
    ) -> EpochUpdateTraceEvidence:
        del population
        return self.epoch_trace


def _counterfeit_exact[Value](value_type: type[Value]) -> Value:
    return object.__new__(value_type)


def _issue_counterfeit_bound_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    registered.RegisteredTrainingExecutor,
    registered._RegisteredExecutorState,
    CausalKWS,
    _CounterfeitTrace,
]:
    seed = TRAINING_SEEDS[0]
    registration = _counterfeit_exact(VerifiedRunRegistration)
    training_inputs = _counterfeit_exact(RegisteredTrainingInputSource)
    validation_inputs = _counterfeit_exact(RegisteredValidationInputs)
    bridge_authority = _counterfeit_exact(RegisteredExecutorSessionAuthority)
    bridge_token = object()
    trace_probe = _CounterfeitTrace()
    trace = cast(UpdateTraceAccumulator, trace_probe)
    model = cast(CausalKWS, SimpleNamespace(training=True))
    optimizer = cast(torch.optim.AdamW, object())
    expected = numeric._RuntimeDigests(
        model_sha256="1" * 64,
        optimizer_sha256="2" * 64,
        rng_sha256="3" * 64,
        rng_state=b"counterfeit-rng",
    )
    lock = threading.RLock()
    runtime = cast(
        numeric._ExecutorState,
        SimpleNamespace(
            token=object(),
            route_marker=registered._REGISTERED_ROUTE,
            seed=seed,
            model=model,
            optimizer=optimizer,
            parameter_names=(),
            parameters=(),
            expected=expected,
            optimizer_generation=0,
            lock=lock,
        ),
    )
    with monkeypatch.context() as local:
        local.setattr(
            registered,
            "_registered_executor_snapshot",
            lambda value: cast(registered._RegisteredExecutorSnapshot, object()),
        )
        executor = registered._issue_registered_executor(
            registration=registration,
            process_id=os.getpid(),
            seed=seed,
            training_inputs=training_inputs,
            validation_inputs=validation_inputs,
            bridge_authority=bridge_authority,
            bridge_session_token=bridge_token,
            trace=trace,
            runtime=runtime,
            expected=expected,
            lock=lock,
        )

    def bridge_snapshot(
        authority: RegisteredExecutorSessionAuthority,
        *,
        require_live: bool = True,
    ) -> bridge._ExecutorSessionSnapshot:
        del require_live
        assert authority is bridge_authority
        return bridge._ExecutorSessionSnapshot(bridge_token, seed, object())

    monkeypatch.setattr(registered, "_executor_session_snapshot", bridge_snapshot)
    state = registered._issued_executor_state(executor)
    return executor, state, model, trace_probe


def _claimed_nonfinal_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    registered.RegisteredTrainingExecutor,
    registered._RegisteredExecutorState,
    registered.RegisteredExecutorEpochHandoff,
    registered_history.RegisteredHistoryBarrier,
    registered_history._RegisteredHistoryBarrierSnapshot,
    registered_history._RegisteredHistoryBarrierSnapshot,
    CausalKWS,
]:
    executor, state, model, trace = _issue_counterfeit_bound_executor(monkeypatch)
    population = _counterfeit_exact(CompletedTrainingPopulation)
    state.optimizer_generation = 313
    state.runtime.optimizer_generation = 313
    state.phase = "TRAINING"
    state.active_epoch = _counterfeit_exact(RegisteredTrainingEpoch)
    trace.completed_epochs = 1
    trace.next_update = 313
    registered._publish_executor_lifecycle(executor, state)
    handoff = registered._issue_epoch_handoff(
        executor,
        state,
        population=population,
        epoch_trace=trace.epoch_trace,
    )
    state.active_epoch = None
    state.active_handoff = handoff
    state.phase = "AWAITING_EVALUATION"
    registered._publish_executor_lifecycle(executor, state)
    monkeypatch.setattr(
        registered,
        "reverify_verified_run_registration",
        lambda value: None,
    )
    monkeypatch.setattr(
        registered,
        "_stable_handoff_runtime_digests",
        lambda value, *, expected_generation, expected_training: state.expected,
    )
    assert (
        registered._registered_epoch_handoff_model(state.registration, handoff) is model
    )
    barrier = object.__new__(RegisteredHistoryBarrier)
    evaluated = object.__new__(RegisteredEvaluatedEpoch)
    owner = object.__new__(RegisteredTrainingHistory)
    issued = registered_history._RegisteredHistoryBarrierSnapshot(
        registration=state.registration,
        history=owner,
        executor=executor,
        validation_inputs=state.validation_inputs,
        process_id=state.process_id,
        seed=state.seed,
        accepted_epoch=0,
        next_zero_based_epoch=1,
        evaluated_epoch=evaluated,
        evaluated_handoff=handoff,
        evaluated_authority_sha256="a" * 64,
        phase="ISSUED",
    )
    consumed = replace(issued, phase="CONSUMED")
    return executor, state, handoff, barrier, issued, consumed, model


def _completed_nonfinal_advance(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    registered.RegisteredTrainingExecutor,
    registered._RegisteredExecutorState,
    registered.RegisteredExecutorEpochHandoff,
    registered_history.RegisteredHistoryBarrier,
]:
    executor, state, handoff, barrier, issued, consumed, _ = _claimed_nonfinal_boundary(
        monkeypatch
    )
    monkeypatch.setattr(
        registered,
        "_preflight_registered_history_barrier",
        lambda observed_executor, observed_barrier: issued,
    )
    monkeypatch.setattr(
        registered,
        "_consume_registered_continuation_history",
        lambda reservation: consumed,
    )
    registered.advance_registered_training_executor(executor, barrier)
    return executor, state, handoff, barrier


@dataclass(frozen=True, slots=True)
class _ClaimedFinalBoundary:
    executor: registered.RegisteredTrainingExecutor
    state: registered._RegisteredExecutorState
    handoff: registered.RegisteredExecutorEpochHandoff
    previous_barrier: RegisteredHistoryBarrier
    barrier: RegisteredHistoryBarrier
    issued: registered_history._RegisteredHistoryBarrierSnapshot
    complete_trace: CompleteUpdateTraceEvidence
    completed_history: RegisteredCompletedTrainingHistory
    model: CausalKWS
    prior_reservations: tuple[registered._ContinuationReservation, ...]
    calls: dict[str, int]
    abort_states: list[tuple[str, bool, str | None]]


def _claimed_final_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> _ClaimedFinalBoundary:
    seed = TRAINING_SEEDS[0]
    registration = _counterfeit_exact(VerifiedRunRegistration)
    training_inputs = _counterfeit_exact(RegisteredTrainingInputSource)
    validation_inputs = _counterfeit_exact(RegisteredValidationInputs)
    bridge_authority = _counterfeit_exact(RegisteredExecutorSessionAuthority)
    bridge_token = object()
    trace = UpdateTraceAccumulator(seed)
    model = cast(CausalKWS, SimpleNamespace(training=True))
    optimizer = cast(torch.optim.AdamW, object())
    expected = numeric._RuntimeDigests(
        model_sha256="1" * 64,
        optimizer_sha256="2" * 64,
        rng_sha256="3" * 64,
        rng_state=b"final-rng",
    )
    lock = threading.RLock()
    runtime = cast(
        numeric._ExecutorState,
        SimpleNamespace(
            token=object(),
            route_marker=registered._REGISTERED_ROUTE,
            seed=seed,
            model=model,
            optimizer=optimizer,
            parameter_names=(),
            parameters=(),
            expected=expected,
            optimizer_generation=0,
            lock=lock,
        ),
    )
    with monkeypatch.context() as local:
        local.setattr(
            registered,
            "_registered_executor_snapshot",
            lambda value: cast(registered._RegisteredExecutorSnapshot, object()),
        )
        executor = registered._issue_registered_executor(
            registration=registration,
            process_id=os.getpid(),
            seed=seed,
            training_inputs=training_inputs,
            validation_inputs=validation_inputs,
            bridge_authority=bridge_authority,
            bridge_session_token=bridge_token,
            trace=trace,
            runtime=runtime,
            expected=expected,
            lock=lock,
        )

    def bridge_snapshot(
        authority: RegisteredExecutorSessionAuthority,
        *,
        require_live: bool = True,
    ) -> bridge._ExecutorSessionSnapshot:
        del require_live
        assert authority is bridge_authority
        return bridge._ExecutorSessionSnapshot(bridge_token, seed, object())

    monkeypatch.setattr(registered, "_executor_session_snapshot", bridge_snapshot)
    monkeypatch.setattr(
        registered, "reverify_verified_run_registration", lambda value: None
    )
    state = registered._issued_executor_state(executor)

    truth_reserve = (
        inspect.signature(registered._reserve_registered_continuation)
        .parameters["_truth_reserve"]
        .default
    )
    truth_transition = (
        inspect.signature(registered._mark_registered_continuation_consumed)
        .parameters["_truth_transition"]
        .default
    )
    truth_commit = (
        inspect.signature(registered._commit_registered_nonfinal_continuation)
        .parameters["_truth_commit"]
        .default
    )
    previous_barrier: RegisteredHistoryBarrier | None = None
    prior_reservations: list[registered._ContinuationReservation] = []
    for zero_based_epoch in range(29):
        prior_handoff = object.__new__(registered.RegisteredExecutorEpochHandoff)
        handoff_ticket = registered._HandoffIssuanceTicket(
            handoff=prior_handoff,
            executor_ticket=state.ticket,
            handoff_token=object(),
        )
        barrier = object.__new__(RegisteredHistoryBarrier)
        reservation = registered._ContinuationReservation(
            token=object(),
            executor=executor,
            executor_ticket=state.ticket,
            handoff=prior_handoff,
            handoff_ticket=handoff_ticket,
            barrier=barrier,
            barrier_type=RegisteredHistoryBarrier,
            evaluated_epoch=object.__new__(RegisteredEvaluatedEpoch),
            evaluated_epoch_type=RegisteredEvaluatedEpoch,
            evaluated_authority_sha256="a" * 64,
            registration=registration,
            validation_inputs=validation_inputs,
            process_id=os.getpid(),
            seed=seed,
            zero_based_epoch=zero_based_epoch,
            optimizer_generation=(zero_based_epoch + 1) * 313,
            runtime_digests=replace(expected),
            previous_history_barrier=previous_barrier,
            handoff_token=object(),
            one_shot_token=object(),
        )
        truth_reserve(reservation)
        truth_transition(reservation, "RESERVED", "CONSUMED")
        truth_commit(
            reservation,
            registered._retirement_from_reservation(reservation),
        )
        prior_reservations.append(reservation)
        previous_barrier = barrier
    assert previous_barrier is not None

    epoch_traces = tuple(object.__new__(EpochUpdateTraceEvidence) for _ in range(30))
    state.zero_based_epoch = 29
    state.optimizer_generation = 30 * 313
    state.runtime.optimizer_generation = 30 * 313
    state.previous_history_barrier = previous_barrier
    state.phase = "TRAINING"
    state.active_epoch = _counterfeit_exact(RegisteredTrainingEpoch)
    registered._publish_executor_lifecycle(executor, state)
    handoff = registered._issue_epoch_handoff(
        executor,
        state,
        population=_counterfeit_exact(CompletedTrainingPopulation),
        epoch_trace=epoch_traces[-1],
    )
    state.active_epoch = None
    state.active_handoff = handoff
    state.phase = "AWAITING_EVALUATION"
    registered._publish_executor_lifecycle(executor, state)
    monkeypatch.setattr(
        registered,
        "_stable_handoff_runtime_digests",
        lambda value, *, expected_generation, expected_training: state.expected,
    )
    assert (
        registered._registered_epoch_handoff_model(state.registration, handoff) is model
    )
    model.training = False

    final_barrier = object.__new__(RegisteredHistoryBarrier)
    evaluated = object.__new__(RegisteredEvaluatedEpoch)
    history = object.__new__(RegisteredTrainingHistory)
    issued = registered_history._RegisteredHistoryBarrierSnapshot(
        registration=registration,
        history=history,
        executor=executor,
        validation_inputs=validation_inputs,
        process_id=os.getpid(),
        seed=seed,
        accepted_epoch=29,
        next_zero_based_epoch=None,
        evaluated_epoch=evaluated,
        evaluated_handoff=handoff,
        evaluated_authority_sha256="b" * 64,
        phase="ISSUED",
    )
    complete_trace = object.__new__(CompleteUpdateTraceEvidence)
    completed_history = object.__new__(RegisteredCompletedTrainingHistory)
    calls = {"trace": 0, "history": 0, "abort": 0}
    abort_states: list[tuple[str, bool, str | None]] = []

    monkeypatch.setattr(
        registered,
        "_preflight_registered_final_history_barrier",
        lambda observed_executor, observed_barrier: issued,
    )
    monkeypatch.setattr(
        registered,
        "_capture_registered_final_trace_epochs",
        lambda observed_trace: epoch_traces,
    )
    monkeypatch.setattr(
        registered,
        "_require_complete_trace_matches_finalization",
        lambda *args: None,
    )
    monkeypatch.setattr(
        registered,
        "_require_completed_history_matches_finalization",
        lambda *args: None,
    )

    def finish_trace(
        reservation: registered._FinalizationReservation,
    ) -> CompleteUpdateTraceEvidence:
        assert reservation.trace is trace
        calls["trace"] += 1
        return complete_trace

    def complete_history(
        reservation: registered._FinalizationReservation,
        observed_trace: CompleteUpdateTraceEvidence,
    ) -> RegisteredCompletedTrainingHistory:
        assert reservation.history is history
        assert observed_trace is complete_trace
        calls["history"] += 1
        return completed_history

    def abort(
        observed_executor: registered.RegisteredTrainingExecutor,
        observed_barrier: RegisteredHistoryBarrier,
        observed_completed: RegisteredCompletedTrainingHistory | None,
    ) -> None:
        assert observed_executor is executor
        assert observed_barrier is final_barrier
        del observed_completed
        calls["abort"] += 1
        handoff_state = registered._HANDOFFS[handoff]
        reservation = state.finalization_reservation
        phase = (
            registered._FINALIZATION_PHASES.get(reservation)
            if reservation is not None
            else None
        )
        abort_states.append((state.phase, handoff_state.failed, phase))

    monkeypatch.setattr(
        registered, "_finish_registered_finalization_trace", finish_trace
    )
    monkeypatch.setattr(
        registered, "_complete_registered_finalization_history", complete_history
    )
    monkeypatch.setattr(
        registered, "_abort_attempted_registered_history_completion", abort
    )
    return _ClaimedFinalBoundary(
        executor,
        state,
        handoff,
        previous_barrier,
        final_barrier,
        issued,
        complete_trace,
        completed_history,
        model,
        tuple(prior_reservations),
        calls,
        abort_states,
    )


def test_public_surface_is_opaque_and_has_exact_signatures() -> None:
    assert not hasattr(registered, "_AUTHORITY_TRUTH")
    assert not hasattr(registered, "_build_authority_truth")
    factory = inspect.signature(registered.create_registered_training_executor)
    assert tuple(factory.parameters) == (
        "registration",
        "training_inputs",
        "validation_inputs",
        "seed",
    )
    assert factory.parameters["seed"].kind is inspect.Parameter.KEYWORD_ONLY
    assert tuple(
        inspect.signature(registered.execute_registered_training_epoch).parameters
    ) == ("executor",)
    advance = inspect.signature(registered.advance_registered_training_executor)
    assert tuple(advance.parameters) == ("executor", "barrier")
    assert str(advance.return_annotation) == "None"
    complete = inspect.signature(registered.complete_registered_training_executor)
    assert tuple(complete.parameters) == ("executor", "final_barrier")
    assert str(complete.return_annotation) == "RegisteredCompletedTrainingHistory"
    assert tuple(
        inspect.signature(registered._registered_epoch_handoff_snapshot).parameters
    ) == ("handoff",)
    assert tuple(
        inspect.signature(registered._registered_epoch_handoff_model).parameters
    ) == ("registration", "handoff")

    with pytest.raises(TypeError, match="issuer-only"):
        registered.RegisteredTrainingExecutor()
    with pytest.raises(TypeError, match="issuer-only"):
        registered.RegisteredExecutorEpochHandoff()
    executor = object.__new__(registered.RegisteredTrainingExecutor)
    handoff = object.__new__(registered.RegisteredExecutorEpochHandoff)
    for value in (executor, handoff):
        with pytest.raises(TypeError, match="cannot be copied"):
            copy.copy(value)
        with pytest.raises(TypeError, match="cannot be copied"):
            copy.deepcopy(value)
        with pytest.raises(TypeError, match="cannot be serialized"):
            pickle.dumps(value)
        assert not hasattr(value, "model")
        assert not hasattr(value, "optimizer")


def test_public_factory_reserves_then_reverifies_before_any_registered_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    fake_pid = 424_242
    monkeypatch.setattr(os, "getpid", lambda: fake_pid)

    def reverify(value: VerifiedRunRegistration) -> None:
        del value
        events.append("reverify")
        raise RuntimeError("stale child-local registration")

    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("registered work ran before registration reverify")

    monkeypatch.setattr(registered, "reverify_verified_run_registration", reverify)
    monkeypatch.setattr(registered, "verify_registered_validation_inputs", forbidden)
    monkeypatch.setattr(registered, "_issued_source_state", forbidden)
    monkeypatch.setattr(torch, "manual_seed", forbidden)
    with pytest.raises(RuntimeError, match="stale child-local"):
        registered.create_registered_training_executor(
            _counterfeit_exact(VerifiedRunRegistration),
            _counterfeit_exact(RegisteredTrainingInputSource),
            _counterfeit_exact(RegisteredValidationInputs),
            seed=TRAINING_SEEDS[0],
        )
    assert events == ["reverify"]
    assert fake_pid in registered._CREATION_RESERVED_PROCESSES
    with pytest.raises(
        registered.Experiment002RegisteredExecutorError, match="reserved"
    ):
        registered._reserve_process_creation(fake_pid)


def test_counterfeit_public_capabilities_are_rejected() -> None:
    counterfeit_executor = object.__new__(registered.RegisteredTrainingExecutor)
    counterfeit_handoff = object.__new__(registered.RegisteredExecutorEpochHandoff)
    with pytest.raises(
        registered.Experiment002RegisteredExecutorError, match="not issued"
    ):
        registered.execute_registered_training_epoch(counterfeit_executor)
    with pytest.raises(registered.Experiment002RegisteredExecutorError):
        registered._registered_epoch_handoff_snapshot(counterfeit_handoff)
    with pytest.raises(registered.Experiment002RegisteredExecutorError):
        registered._fail_registered_epoch_handoff(counterfeit_handoff)


def test_private_update_seam_has_exact_handshake_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, state, _, _ = _issue_counterfeit_bound_executor(monkeypatch)
    epoch = _counterfeit_exact(RegisteredTrainingEpoch)
    batch = _counterfeit_exact(MaterializedTrainingBatch)
    transition = _counterfeit_exact(RegisteredOptimizerTransition)
    receipt = _counterfeit_exact(CompletedTrainingUpdate)
    consumption = _counterfeit_exact(TraceConsumedTransition)
    before = state.expected
    after = numeric._RuntimeDigests("4" * 64, "5" * 64, "6" * 64, b"after")
    outcome = registered._NumericUpdateOutcome(
        learning_rate=0.0003,
        batch_mean_training_loss=cast(Any, 1.0),
        returned_preclip_l2_norm=cast(Any, 2.0),
        rng_state_after_forward=b"after",
    )
    binding = _RegisteredBatchBinding(
        executor_session_token=state.bridge_session_token,
        epoch_session_token=object(),
        batch_token=object(),
        seed=state.seed,
        zero_based_epoch=0,
        zero_based_global_update=0,
        batch_size=128,
    )
    views = cast(registered._BatchViews, object())
    events: list[str] = []

    def seam(name: str, result: object = None) -> Callable[..., object]:
        def call(*args: object, **kwargs: object) -> object:
            del args, kwargs
            events.append(name)
            return result

        return call

    monkeypatch.setattr(
        registered, "_entry_zero_grad_digests", seam("entry_zero_grad", before)
    )
    monkeypatch.setattr(registered, "_next_registered_batch", seam("next", batch))
    monkeypatch.setattr(registered, "_verify_registered_batch_pre", seam("PRE"))
    monkeypatch.setattr(
        registered, "_zero_copy_registered_batch", seam("zero_copy", views)
    )
    monkeypatch.setattr(
        registered, "_perform_registered_numeric_update", seam("numeric", outcome)
    )
    monkeypatch.setattr(registered, "_verify_registered_batch_post", seam("POST"))
    monkeypatch.setattr(
        registered, "_registered_batch_binding", seam("binding", binding)
    )
    monkeypatch.setattr(
        registered, "_capture_registered_update_after", seam("after_digest", after)
    )
    monkeypatch.setattr(
        registered, "_issue_update_transition", seam("transition", transition)
    )
    monkeypatch.setattr(registered, "_publish_executor_lifecycle", seam("publish"))
    monkeypatch.setattr(
        registered, "_complete_registered_update", seam("receipt", receipt)
    )
    monkeypatch.setattr(
        registered, "_consume_registered_trace", seam("trace_consume", consumption)
    )
    monkeypatch.setattr(
        registered, "_accept_registered_population", seam("population_accept")
    )
    monkeypatch.setattr(registered, "_post_update_digests", seam("post_digest"))

    registered._execute_registered_update(executor, state, epoch, global_update=0)
    assert events == [
        "entry_zero_grad",
        "next",
        "PRE",
        "zero_copy",
        "numeric",
        "POST",
        "binding",
        "after_digest",
        "transition",
        "publish",
        "receipt",
        "trace_consume",
        "population_accept",
        "post_digest",
        "publish",
    ]
    assert state.optimizer_generation == 1
    assert state.expected == after
    assert state.runtime.expected == after
    assert state.active_transition is None


def test_execution_failure_marks_local_failed_before_dependency_propagation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, state, _, _ = _issue_counterfeit_bound_executor(monkeypatch)
    epoch = _counterfeit_exact(RegisteredTrainingEpoch)
    cleanup: list[tuple[str, registered.RegisteredExecutorPhase]] = []

    monkeypatch.setattr(
        registered, "reverify_verified_run_registration", lambda value: None
    )
    monkeypatch.setattr(registered, "_verify_bound_inputs", lambda value: None)
    monkeypatch.setattr(numeric, "_verify_runtime", lambda: None)
    monkeypatch.setattr(
        numeric,
        "_stable_runtime_digests",
        lambda value, *, expected_generation: state.expected,
    )
    monkeypatch.setattr(
        registered,
        "_begin_registered_epoch_for_executor",
        lambda *args, **kwargs: epoch,
    )

    def fail_update(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("synthetic failpoint")

    monkeypatch.setattr(registered, "_execute_registered_update", fail_update)

    def observe(name: str) -> Callable[..., None]:
        def call(*args: object, **kwargs: object) -> None:
            del args, kwargs
            cleanup.append((name, state.phase))

        return call

    monkeypatch.setattr(registered, "_fail_optimizer_transition", observe("transition"))
    monkeypatch.setattr(registered, "_executor_fail_epoch", observe("epoch"))
    monkeypatch.setattr(registered, "_fail_registered_trace", observe("trace"))
    monkeypatch.setattr(
        registered, "_fail_registered_executor_session", observe("bridge")
    )
    with pytest.raises(RuntimeError, match="synthetic failpoint"):
        registered.execute_registered_training_epoch(executor)
    assert cleanup == [
        ("transition", "FAILED"),
        ("epoch", "FAILED"),
        ("trace", "FAILED"),
        ("bridge", "FAILED"),
    ]
    assert state.phase == "FAILED"
    assert executor in registered._FAILED_EXECUTOR_HISTORY
    with pytest.raises(registered.Experiment002RegisteredExecutorError):
        registered._registered_executor_snapshot(executor)


def test_lifecycle_tamper_is_terminal_even_after_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, state, _, _ = _issue_counterfeit_bound_executor(monkeypatch)
    original = registered._EXECUTOR_LIFECYCLES[executor]
    state.optimizer_generation = 7
    with pytest.raises(registered.Experiment002RegisteredExecutorError):
        registered._registered_executor_snapshot(executor)
    state.optimizer_generation = original.optimizer_generation
    state.phase = original.phase
    registered._EXECUTOR_LIFECYCLES[executor] = original
    with pytest.raises(registered.Experiment002RegisteredExecutorError):
        registered._registered_executor_snapshot(executor)
    assert executor in registered._FAILED_EXECUTOR_HISTORY


def test_guard_removal_is_terminal_and_cannot_be_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, state, _, _ = _issue_counterfeit_bound_executor(monkeypatch)
    guard = registered._EXECUTOR_GUARDS.pop(executor)
    with pytest.raises(registered.Experiment002RegisteredExecutorError, match="lock"):
        registered._registered_executor_snapshot(executor)
    registered._EXECUTOR_GUARDS[executor] = guard
    state.phase = "READY_TO_TRAIN"
    registered._EXECUTOR_LIFECYCLES[executor] = registered._lifecycle_from_state(state)
    with pytest.raises(registered.Experiment002RegisteredExecutorError):
        registered._registered_executor_snapshot(executor)


def test_executor_authority_is_rejected_in_a_fork_without_poisoning_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, _, _, _ = _issue_counterfeit_bound_executor(monkeypatch)
    read_fd, write_fd = os.pipe()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            child = os.fork()
        if child == 0:
            os.close(read_fd)
            try:
                registered._registered_executor_snapshot(executor)
            except registered.Experiment002RegisteredExecutorError:
                os.write(write_fd, b"rejected")
            else:
                os.write(write_fd, b"accepted")
            finally:
                os.close(write_fd)
            os._exit(0)
        os.close(write_fd)
        write_fd = -1
        observed = os.read(read_fd, 32)
        waited, status = os.waitpid(child, 0)
        assert waited == child
        assert os.waitstatus_to_exitcode(status) == 0
        assert observed == b"rejected"
        assert registered._registered_executor_snapshot(executor).phase == (
            "READY_TO_TRAIN"
        )
    finally:
        os.close(read_fd)
        if write_fd >= 0:
            os.close(write_fd)


def test_concurrent_epoch_call_has_one_winner_and_nonpoisoning_loser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, state, _, trace_probe = _issue_counterfeit_bound_executor(monkeypatch)
    epoch = _counterfeit_exact(RegisteredTrainingEpoch)
    population = _counterfeit_exact(CompletedTrainingPopulation)
    handoff = object.__new__(registered.RegisteredExecutorEpochHandoff)
    entered = threading.Event()
    release = threading.Event()
    update_count = 0

    monkeypatch.setattr(
        registered, "reverify_verified_run_registration", lambda value: None
    )
    monkeypatch.setattr(registered, "_verify_bound_inputs", lambda value: None)
    monkeypatch.setattr(numeric, "_verify_runtime", lambda: None)
    monkeypatch.setattr(
        numeric,
        "_stable_runtime_digests",
        lambda value, *, expected_generation: state.expected,
    )
    monkeypatch.setattr(
        registered,
        "_begin_registered_epoch_for_executor",
        lambda *args, **kwargs: epoch,
    )

    def fake_update(
        observed_executor: registered.RegisteredTrainingExecutor,
        observed_state: registered._RegisteredExecutorState,
        observed_epoch: RegisteredTrainingEpoch,
        *,
        global_update: int,
    ) -> None:
        nonlocal update_count
        assert observed_executor is executor
        assert observed_state is state
        assert observed_epoch is epoch
        update_count += 1
        if global_update == 0:
            entered.set()
            assert release.wait(timeout=2.0)
        state.optimizer_generation = global_update + 1
        state.runtime.optimizer_generation = global_update + 1

    monkeypatch.setattr(registered, "_execute_registered_update", fake_update)
    monkeypatch.setattr(registered, "_executor_finish_epoch", lambda *args: population)
    monkeypatch.setattr(
        registered,
        "verify_completed_registered_training_population",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        registered,
        "verify_registered_epoch_update_trace",
        lambda value: None,
    )
    assert trace_probe.epoch_trace is not None
    monkeypatch.setattr(
        registered, "_issue_epoch_handoff", lambda *args, **kwargs: handoff
    )
    monkeypatch.setattr(
        registered, "_registered_epoch_handoff_snapshot", lambda value: object()
    )

    outcomes: list[object] = []

    def run() -> None:
        try:
            outcomes.append(registered.execute_registered_training_epoch(executor))
        except BaseException as error:
            outcomes.append(error)

    first = threading.Thread(target=run)
    second = threading.Thread(target=run)
    first.start()
    assert entered.wait(timeout=2.0)
    second.start()
    release.set()
    first.join(timeout=5.0)
    second.join(timeout=5.0)
    assert not first.is_alive() and not second.is_alive()
    assert sum(value is handoff for value in outcomes) == 1
    errors = [value for value in outcomes if isinstance(value, BaseException)]
    assert len(errors) == 1
    assert isinstance(errors[0], registered.Experiment002RegisteredExecutorError)
    assert update_count == 313
    assert state.phase == "AWAITING_EVALUATION"
    assert state.optimizer_generation == 313
    assert executor not in registered._FAILED_EXECUTOR_HISTORY


def test_handoff_snapshot_model_claim_and_failure_are_exact_and_one_shot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, state, model, trace_probe = _issue_counterfeit_bound_executor(monkeypatch)
    population = _counterfeit_exact(CompletedTrainingPopulation)
    state.optimizer_generation = 313
    state.runtime.optimizer_generation = 313
    state.phase = "TRAINING"
    state.active_epoch = _counterfeit_exact(RegisteredTrainingEpoch)
    registered._publish_executor_lifecycle(executor, state)
    handoff = registered._issue_epoch_handoff(
        executor,
        state,
        population=population,
        epoch_trace=trace_probe.epoch_trace,
    )
    state.active_handoff = handoff
    state.active_epoch = None
    state.phase = "AWAITING_EVALUATION"
    registered._publish_executor_lifecycle(executor, state)
    monkeypatch.setattr(
        registered, "reverify_verified_run_registration", lambda value: None
    )
    monkeypatch.setattr(
        numeric,
        "_stable_runtime_digests",
        lambda value, *, expected_generation: state.expected,
    )

    snapshot = registered._registered_epoch_handoff_snapshot(handoff)
    assert snapshot.executor is executor
    assert snapshot.registration is state.registration
    assert snapshot.validation_inputs is state.validation_inputs
    assert snapshot.population is population
    assert snapshot.epoch_trace is trace_probe.epoch_trace
    assert snapshot.seed == state.seed
    assert snapshot.zero_based_epoch == 0
    assert snapshot.optimizer_generation == 313
    assert snapshot.runtime_digests == state.expected
    assert snapshot.runtime_digests is not state.expected
    assert snapshot.previous_history_barrier is None
    assert snapshot.another_training_epoch is True
    assert not hasattr(snapshot, "model")
    assert not hasattr(snapshot, "optimizer")

    assert (
        registered._registered_epoch_handoff_model(state.registration, handoff) is model
    )
    with pytest.raises(
        registered.Experiment002RegisteredExecutorError, match="consumed"
    ):
        registered._registered_epoch_handoff_model(state.registration, handoff)
    assert registered._registered_epoch_handoff_snapshot(handoff) == snapshot

    monkeypatch.setattr(registered, "_fail_optimizer_transition", lambda value: None)
    monkeypatch.setattr(registered, "_executor_fail_epoch", lambda *args: None)
    monkeypatch.setattr(registered, "_fail_registered_trace", lambda value: None)
    monkeypatch.setattr(
        registered, "_fail_registered_executor_session", lambda value: None
    )
    registered._fail_registered_epoch_handoff(handoff)
    assert registered._issued_executor_state(executor).phase == "FAILED"
    assert executor in registered._FAILED_EXECUTOR_HISTORY
    assert handoff in registered._FAILED_HANDOFF_HISTORY
    with pytest.raises(registered.Experiment002RegisteredExecutorError):
        registered._registered_epoch_handoff_snapshot(handoff)


def test_handoff_tamper_poisoning_is_irreversible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, state, _, trace_probe = _issue_counterfeit_bound_executor(monkeypatch)
    population = _counterfeit_exact(CompletedTrainingPopulation)
    state.optimizer_generation = 313
    state.runtime.optimizer_generation = 313
    state.phase = "TRAINING"
    state.active_epoch = _counterfeit_exact(RegisteredTrainingEpoch)
    registered._publish_executor_lifecycle(executor, state)
    handoff = registered._issue_epoch_handoff(
        executor,
        state,
        population=population,
        epoch_trace=trace_probe.epoch_trace,
    )
    state.active_handoff = handoff
    state.active_epoch = None
    state.phase = "AWAITING_EVALUATION"
    registered._publish_executor_lifecycle(executor, state)
    handoff_state = registered._HANDOFFS[handoff]
    original_token = handoff_state.one_shot_token
    handoff_state.one_shot_token = object()
    with pytest.raises(registered.Experiment002RegisteredExecutorError):
        registered._registered_epoch_handoff_snapshot(handoff)
    handoff_state.one_shot_token = original_token
    registered._HANDOFF_LIFECYCLES[handoff] = registered._HandoffLifecycle(False, False)
    with pytest.raises(registered.Experiment002RegisteredExecutorError):
        registered._registered_epoch_handoff_snapshot(handoff)
    assert handoff in registered._FAILED_HANDOFF_HISTORY
    assert executor in registered._FAILED_EXECUTOR_HISTORY


def test_final_handoff_requires_train_before_claim_and_eval_after_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, state, model, trace_probe = _issue_counterfeit_bound_executor(monkeypatch)
    population = _counterfeit_exact(CompletedTrainingPopulation)
    state.zero_based_epoch = 29
    state.optimizer_generation = 30 * 313
    state.runtime.optimizer_generation = 30 * 313
    state.previous_history_barrier = object()
    state.phase = "TRAINING"
    state.active_epoch = _counterfeit_exact(RegisteredTrainingEpoch)
    registered._publish_executor_lifecycle(executor, state)
    handoff = registered._issue_epoch_handoff(
        executor,
        state,
        population=population,
        epoch_trace=trace_probe.epoch_trace,
    )
    state.active_handoff = handoff
    state.active_epoch = None
    state.phase = "AWAITING_EVALUATION"
    registered._publish_executor_lifecycle(executor, state)
    monkeypatch.setattr(
        registered, "reverify_verified_run_registration", lambda value: None
    )
    monkeypatch.setattr(
        registered,
        "_stable_handoff_runtime_digests",
        lambda value, *, expected_generation, expected_training: state.expected,
    )

    before = registered._registered_epoch_handoff_snapshot(handoff)
    assert before.another_training_epoch is False
    assert registered._expected_handoff_model_training(registered._HANDOFFS[handoff])
    assert (
        registered._registered_epoch_handoff_model(state.registration, handoff) is model
    )
    handoff_state = registered._HANDOFFS[handoff]
    assert handoff_state.used
    assert not registered._expected_handoff_model_training(handoff_state)

    model.training = False
    after = registered._registered_epoch_handoff_snapshot(handoff)
    assert after == before
    model.training = True
    with pytest.raises(registered.Experiment002RegisteredExecutorError):
        registered._registered_epoch_handoff_snapshot(handoff)
    assert executor in registered._FAILED_EXECUTOR_HISTORY


def test_nonfinal_claimed_handoff_still_requires_training_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, state, model, trace_probe = _issue_counterfeit_bound_executor(monkeypatch)
    population = _counterfeit_exact(CompletedTrainingPopulation)
    state.optimizer_generation = 313
    state.runtime.optimizer_generation = 313
    state.phase = "TRAINING"
    state.active_epoch = _counterfeit_exact(RegisteredTrainingEpoch)
    registered._publish_executor_lifecycle(executor, state)
    handoff = registered._issue_epoch_handoff(
        executor,
        state,
        population=population,
        epoch_trace=trace_probe.epoch_trace,
    )
    state.active_handoff = handoff
    state.active_epoch = None
    state.phase = "AWAITING_EVALUATION"
    registered._publish_executor_lifecycle(executor, state)
    monkeypatch.setattr(
        registered, "reverify_verified_run_registration", lambda value: None
    )
    monkeypatch.setattr(
        registered,
        "_stable_handoff_runtime_digests",
        lambda value, *, expected_generation, expected_training: state.expected,
    )
    assert (
        registered._registered_epoch_handoff_model(state.registration, handoff) is model
    )
    handoff_state = registered._HANDOFFS[handoff]
    assert registered._expected_handoff_model_training(handoff_state)
    assert registered._registered_epoch_handoff_snapshot(handoff).executor is executor
    model.training = False
    with pytest.raises(registered.Experiment002RegisteredExecutorError):
        registered._registered_epoch_handoff_snapshot(handoff)


def test_nonfinal_advance_retires_handoff_and_preserves_frozen_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, state, handoff, barrier, issued, consumed, model = (
        _claimed_nonfinal_boundary(monkeypatch)
    )
    before = registered._registered_epoch_handoff_snapshot(handoff)
    expected = state.expected
    lock_probe: list[bool] = []

    monkeypatch.setattr(
        registered,
        "_preflight_registered_history_barrier",
        lambda observed_executor, observed_barrier: issued,
    )

    def consume(
        reservation: registered._ContinuationReservation,
    ) -> registered_history._RegisteredHistoryBarrierSnapshot:
        def probe() -> None:
            acquired_handoff = registered._HANDOFFS_LOCK.acquire(timeout=1.0)
            acquired_executor = False
            try:
                if acquired_handoff:
                    acquired_executor = state.lock.acquire(timeout=1.0)
            finally:
                if acquired_executor:
                    state.lock.release()
                if acquired_handoff:
                    registered._HANDOFFS_LOCK.release()
            lock_probe.append(acquired_handoff and acquired_executor)

        thread = threading.Thread(target=probe)
        thread.start()
        thread.join(timeout=2.0)
        assert not thread.is_alive()
        assert reservation.barrier is barrier
        return consumed

    monkeypatch.setattr(
        registered,
        "_consume_registered_continuation_history",
        consume,
    )
    registered.advance_registered_training_executor(executor, barrier)

    assert lock_probe == [True]
    assert state.phase == "READY_TO_TRAIN"
    assert state.zero_based_epoch == 1
    assert state.optimizer_generation == 313
    assert state.active_epoch is None
    assert state.active_transition is None
    assert state.active_handoff is None
    assert state.previous_history_barrier is barrier
    assert state.continuation_reservation is None
    assert state.expected == expected
    assert state.runtime.expected == expected
    assert model.training is True
    retired = registered._RETIRED_HANDOFFS[handoff]
    assert registered._CONTINUATION_PHASES[retired.reservation] == "COMMITTED"

    model.training = False
    state.expected = numeric._RuntimeDigests("4" * 64, "5" * 64, "6" * 64, b"later")
    assert registered._registered_epoch_handoff_snapshot(handoff) == before
    with pytest.raises(
        registered.Experiment002RegisteredExecutorError,
        match="consumed|retired",
    ):
        registered._registered_epoch_handoff_model(state.registration, handoff)


@pytest.mark.parametrize(
    "window",
    (
        "before_first_executor_check",
        "after_first_executor_check",
        "before_final_executor_check",
    ),
)
def test_live_snapshot_survives_full_retirement_in_every_unlocked_window(
    monkeypatch: pytest.MonkeyPatch,
    window: str,
) -> None:
    executor, state, handoff, barrier, issued, consumed, _ = _claimed_nonfinal_boundary(
        monkeypatch
    )
    before = registered._registered_epoch_handoff_snapshot(handoff)
    snapshot_paused = threading.Event()
    release_snapshot = threading.Event()
    outcomes: list[object] = []
    trusted_calls = 0
    original_issued = registered._issued_executor_state
    original_trusted = registered._trusted_executor_lock

    monkeypatch.setattr(
        registered,
        "_preflight_registered_history_barrier",
        lambda observed_executor, observed_barrier: issued,
    )
    monkeypatch.setattr(
        registered,
        "_consume_registered_continuation_history",
        lambda reservation: consumed,
    )

    class _PauseAfterExit:
        def __init__(self, lock: Any) -> None:
            self._lock = lock

        def __enter__(self) -> None:
            self._lock.acquire()

        def __exit__(
            self,
            exc_type: Any,
            exc_value: Any,
            traceback: Any,
        ) -> None:
            del exc_type, exc_value, traceback
            self._lock.release()
            snapshot_paused.set()
            assert release_snapshot.wait(timeout=5.0)

    def issued_state(
        observed_executor: registered.RegisteredTrainingExecutor,
    ) -> registered._RegisteredExecutorState:
        if (
            threading.current_thread().name == "retirement-snapshot"
            and window == "before_first_executor_check"
        ):
            snapshot_paused.set()
            assert release_snapshot.wait(timeout=5.0)
        return original_issued(observed_executor)

    def trusted_lock(
        observed_executor: registered.RegisteredTrainingExecutor,
        observed_state: registered._RegisteredExecutorState,
    ) -> Any:
        nonlocal trusted_calls
        lock = original_trusted(observed_executor, observed_state)
        if threading.current_thread().name != "retirement-snapshot":
            return lock
        trusted_calls += 1
        if window == "after_first_executor_check" and trusted_calls == 1:
            return _PauseAfterExit(lock)
        if window == "before_final_executor_check" and trusted_calls == 2:
            snapshot_paused.set()
            assert release_snapshot.wait(timeout=5.0)
        return lock

    monkeypatch.setattr(registered, "_issued_executor_state", issued_state)
    monkeypatch.setattr(registered, "_trusted_executor_lock", trusted_lock)

    def snapshot() -> None:
        try:
            outcomes.append(registered._registered_epoch_handoff_snapshot(handoff))
        except BaseException as error:
            outcomes.append(error)

    reader = threading.Thread(target=snapshot, name="retirement-snapshot")
    reader.start()
    assert snapshot_paused.wait(timeout=5.0)

    registered.advance_registered_training_executor(executor, barrier)
    assert state.phase == "READY_TO_TRAIN"
    assert state.zero_based_epoch == 1
    assert executor not in registered._FAILED_EXECUTOR_HISTORY
    assert handoff not in registered._FAILED_HANDOFF_HISTORY

    release_snapshot.set()
    reader.join(timeout=5.0)
    assert not reader.is_alive()
    assert outcomes == [before]
    assert state.phase == "READY_TO_TRAIN"
    assert state.zero_based_epoch == 1
    assert executor not in registered._FAILED_EXECUTOR_HISTORY
    assert handoff not in registered._FAILED_HANDOFF_HISTORY


def test_concurrent_nonfinal_advance_has_one_history_consumer_and_safe_follower(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, _, handoff, barrier, issued, consumed, _ = _claimed_nonfinal_boundary(
        monkeypatch
    )
    entered = threading.Event()
    release = threading.Event()
    follower_admitted = threading.Event()
    counts = {"begin": 0, "preflight": 0, "consume": 0}
    count_lock = threading.Lock()
    original_begin = registered._begin_advance_admission

    def begin(
        observed_executor: registered.RegisteredTrainingExecutor,
        observed_barrier: registered_history.RegisteredHistoryBarrier,
        caller_token: object | None = None,
    ) -> registered._AdvanceAdmission:
        admission = original_begin(
            observed_executor,
            observed_barrier,
            caller_token,
        )
        with count_lock:
            counts["begin"] += 1
            if counts["begin"] == 2:
                follower_admitted.set()
        return admission

    def preflight(
        observed_executor: registered.RegisteredTrainingExecutor,
        observed_barrier: registered_history.RegisteredHistoryBarrier,
    ) -> registered_history._RegisteredHistoryBarrierSnapshot:
        del observed_executor, observed_barrier
        counts["preflight"] += 1
        return issued

    def consume(
        reservation: registered._ContinuationReservation,
    ) -> registered_history._RegisteredHistoryBarrierSnapshot:
        assert reservation.handoff is handoff
        counts["consume"] += 1
        entered.set()
        assert release.wait(timeout=2.0)
        return consumed

    monkeypatch.setattr(registered, "_begin_advance_admission", begin)
    monkeypatch.setattr(registered, "_preflight_registered_history_barrier", preflight)
    monkeypatch.setattr(
        registered,
        "_consume_registered_continuation_history",
        consume,
    )
    outcomes: list[BaseException | None] = []

    def run() -> None:
        try:
            registered.advance_registered_training_executor(executor, barrier)
        except BaseException as error:
            outcomes.append(error)
        else:
            outcomes.append(None)

    leader = threading.Thread(target=run)
    follower = threading.Thread(target=run)
    leader.start()
    assert entered.wait(timeout=2.0)
    follower.start()
    assert follower_admitted.wait(timeout=2.0)
    release.set()
    leader.join(timeout=5.0)
    follower.join(timeout=5.0)
    assert not leader.is_alive() and not follower.is_alive()
    assert outcomes == [None, None]
    assert counts == {"begin": 2, "preflight": 1, "consume": 1}
    assert registered._issued_executor_state(executor).phase == "READY_TO_TRAIN"


def test_admission_validation_failure_wakes_concurrent_follower(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, state, _, barrier, _, _, _ = _claimed_nonfinal_boundary(monkeypatch)
    leader_entered = threading.Event()
    follower_entered = threading.Event()
    cleanup_entered = threading.Event()
    release_cleanup = threading.Event()
    calls = 0
    calls_lock = threading.Lock()

    def fail_validation(
        admission: registered._AdvanceAdmission,
        *,
        executor: registered.RegisteredTrainingExecutor,
        barrier: registered_history.RegisteredHistoryBarrier,
    ) -> None:
        del admission, executor, barrier
        nonlocal calls
        with calls_lock:
            calls += 1
            current = calls
        if current == 1:
            leader_entered.set()
            assert follower_entered.wait(timeout=2.0)
            raise RuntimeError("admission validation failpoint")
        follower_entered.set()

    monkeypatch.setattr(registered, "_require_advance_admission", fail_validation)
    original_terminal = registered._terminal_fail_registered_continuation

    def blocked_terminal(
        observed_executor: registered.RegisteredTrainingExecutor,
        *,
        handoff: registered.RegisteredExecutorEpochHandoff | None,
        reservation: registered._ContinuationReservation | None,
    ) -> None:
        cleanup_entered.set()
        assert release_cleanup.wait(timeout=2.0)
        original_terminal(
            observed_executor,
            handoff=handoff,
            reservation=reservation,
        )

    monkeypatch.setattr(
        registered,
        "_terminal_fail_registered_continuation",
        blocked_terminal,
    )
    outcomes: list[BaseException] = []

    def run() -> None:
        try:
            registered.advance_registered_training_executor(executor, barrier)
        except BaseException as error:
            outcomes.append(error)

    leader = threading.Thread(target=run)
    follower = threading.Thread(target=run)
    leader.start()
    assert leader_entered.wait(timeout=2.0)
    follower.start()
    assert cleanup_entered.wait(timeout=2.0)
    assert follower.is_alive()
    assert state.phase == "AWAITING_EVALUATION"
    release_cleanup.set()
    leader.join(timeout=5.0)
    follower.join(timeout=5.0)
    assert not leader.is_alive() and not follower.is_alive()
    assert len(outcomes) == 2
    assert any(isinstance(error, RuntimeError) for error in outcomes)
    assert cast(str, state.phase) == "FAILED"
    assert executor in registered._FAILED_EXECUTOR_HISTORY


def test_only_admission_leader_can_publish_shared_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, _, _, barrier, _, _, _ = _claimed_nonfinal_boundary(monkeypatch)
    leader = registered._begin_advance_admission(executor, barrier)
    follower = registered._begin_advance_admission(executor, barrier)
    assert leader.leader is True
    assert follower.leader is False
    with pytest.raises(
        registered.Experiment002RegisteredExecutorError,
        match="leader",
    ):
        registered._finish_advance_admission(follower, None)
    second_follower = registered._begin_advance_admission(executor, barrier)
    assert second_follower.leader is False
    registered._finish_advance_admission(leader, None)
    assert follower.completed.wait(timeout=1.0)
    assert follower.outcome == [None]


def test_late_exact_advance_retry_is_durable_idempotent_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, state, handoff, barrier, issued, consumed, _ = _claimed_nonfinal_boundary(
        monkeypatch
    )
    consume_count = 0
    monkeypatch.setattr(
        registered,
        "_preflight_registered_history_barrier",
        lambda observed_executor, observed_barrier: issued,
    )

    def consume(
        reservation: registered._ContinuationReservation,
    ) -> registered_history._RegisteredHistoryBarrierSnapshot:
        nonlocal consume_count
        consume_count += 1
        return consumed

    monkeypatch.setattr(
        registered,
        "_consume_registered_continuation_history",
        consume,
    )
    registered.advance_registered_training_executor(executor, barrier)
    before = registered._registered_executor_snapshot(executor)
    registered.advance_registered_training_executor(executor, barrier)
    after = registered._registered_executor_snapshot(executor)
    assert before == after
    assert after.phase == "READY_TO_TRAIN"
    assert after.zero_based_epoch == 1
    assert consume_count == 1
    assert handoff not in registered._FAILED_HANDOFF_HISTORY


def test_terminal_cleanup_never_holds_handoff_lock_while_waiting_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, state, handoff, _, _, _, _ = _claimed_nonfinal_boundary(monkeypatch)
    executor_held = threading.Event()
    handoff_checked = threading.Event()
    acquired: list[bool] = []
    original_terminal_handoff = registered._terminal_fail_handoff

    def observed_terminal_handoff(
        observed_handoff: registered.RegisteredExecutorEpochHandoff,
        observed_state: registered._HandoffState,
    ) -> None:
        original_terminal_handoff(observed_handoff, observed_state)
        handoff_checked.set()

    monkeypatch.setattr(
        registered,
        "_terminal_fail_handoff",
        observed_terminal_handoff,
    )

    def hold_executor_then_take_handoff() -> None:
        with state.lock:
            executor_held.set()
            assert handoff_checked.wait(timeout=2.0)
            got = registered._HANDOFFS_LOCK.acquire(timeout=1.0)
            acquired.append(got)
            if got:
                registered._HANDOFFS_LOCK.release()

    holder = threading.Thread(target=hold_executor_then_take_handoff)
    holder.start()
    assert executor_held.wait(timeout=2.0)
    cleanup = threading.Thread(
        target=registered._terminal_fail_registered_continuation,
        kwargs={"executor": executor, "handoff": handoff, "reservation": None},
    )
    cleanup.start()
    holder.join(timeout=5.0)
    cleanup.join(timeout=5.0)
    assert not holder.is_alive() and not cleanup.is_alive()
    assert acquired == [True]
    assert state.phase == "FAILED"


def test_handoff_access_releases_handoff_lock_before_executor_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, state, handoff, _, _, _, _ = _claimed_nonfinal_boundary(monkeypatch)
    executor_held = threading.Event()
    handoff_validated = threading.Event()
    acquired: list[bool] = []
    outcomes: list[object] = []
    original_validate = registered._validate_handoff

    def observed_validate(
        observed_handoff: registered.RegisteredExecutorEpochHandoff,
        observed_state: registered._HandoffState | None,
    ) -> None:
        original_validate(observed_handoff, observed_state)
        if threading.current_thread().name == "snapshot-race":
            handoff_validated.set()

    monkeypatch.setattr(registered, "_validate_handoff", observed_validate)

    def hold_executor_then_take_handoff() -> None:
        with state.lock:
            executor_held.set()
            assert handoff_validated.wait(timeout=2.0)
            got = registered._HANDOFFS_LOCK.acquire(timeout=1.0)
            acquired.append(got)
            if got:
                registered._HANDOFFS_LOCK.release()

    def snapshot() -> None:
        try:
            outcomes.append(registered._registered_epoch_handoff_snapshot(handoff))
        except BaseException as error:
            outcomes.append(error)

    holder = threading.Thread(target=hold_executor_then_take_handoff)
    reader = threading.Thread(target=snapshot, name="snapshot-race")
    holder.start()
    assert executor_held.wait(timeout=2.0)
    reader.start()
    holder.join(timeout=5.0)
    reader.join(timeout=5.0)
    assert not holder.is_alive() and not reader.is_alive()
    assert acquired == [True]
    assert len(outcomes) == 1
    assert isinstance(outcomes[0], registered._RegisteredExecutorEpochHandoffSnapshot)


def test_consumed_truth_and_visible_phase_are_atomic_to_executor_readers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, state, handoff, barrier, issued, _, _ = _claimed_nonfinal_boundary(
        monkeypatch
    )
    reservation = registered._reserve_registered_continuation(
        executor,
        handoff,
        barrier,
        issued,
    )
    assignment_entered = threading.Event()
    release_assignment = threading.Event()
    snapshot_done = threading.Event()
    outcomes: list[object] = []
    backing = registered._CONTINUATION_PHASES

    class _BlockingPhases(dict[registered._ContinuationReservation, str]):
        def __setitem__(
            self,
            key: registered._ContinuationReservation,
            value: str,
        ) -> None:
            if key is reservation and value == "CONSUMED":
                assignment_entered.set()
                assert release_assignment.wait(timeout=2.0)
            backing[key] = cast(registered._ContinuationPhase, value)
            super().__setitem__(key, value)

    phases = _BlockingPhases(backing)
    monkeypatch.setattr(registered, "_CONTINUATION_PHASES", phases)

    def mark() -> None:
        try:
            registered._mark_registered_continuation_consumed(reservation)
        except BaseException as error:
            outcomes.append(error)

    def snapshot() -> None:
        try:
            outcomes.append(registered._registered_executor_snapshot(executor))
        except BaseException as error:
            outcomes.append(error)
        finally:
            snapshot_done.set()

    marker = threading.Thread(target=mark)
    reader = threading.Thread(target=snapshot)
    marker.start()
    assert assignment_entered.wait(timeout=2.0)
    reader.start()
    assert not snapshot_done.wait(timeout=0.1)
    release_assignment.set()
    marker.join(timeout=5.0)
    reader.join(timeout=5.0)
    assert not marker.is_alive() and not reader.is_alive()
    assert len(outcomes) == 1
    assert isinstance(outcomes[0], registered._RegisteredExecutorSnapshot)
    assert state.phase == "AWAITING_EVALUATION"
    assert backing[reservation] == "CONSUMED"


def test_committed_truth_and_visible_phase_are_atomic_to_executor_readers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, state, handoff, barrier, issued, consumed, _ = _claimed_nonfinal_boundary(
        monkeypatch
    )
    reservation = registered._reserve_registered_continuation(
        executor,
        handoff,
        barrier,
        issued,
    )
    registered._mark_registered_continuation_consumed(reservation)
    assignment_entered = threading.Event()
    release_assignment = threading.Event()
    snapshot_done = threading.Event()
    outcomes: list[object] = []
    errors: list[BaseException] = []
    backing = registered._CONTINUATION_PHASES

    class _BlockingCommitPhases(
        dict[registered._ContinuationReservation, registered._ContinuationPhase]
    ):
        def __setitem__(
            self,
            key: registered._ContinuationReservation,
            value: registered._ContinuationPhase,
        ) -> None:
            if key is reservation and value == "COMMITTED":
                assignment_entered.set()
                assert release_assignment.wait(timeout=5.0)
            backing[key] = value
            super().__setitem__(key, value)

    phases = _BlockingCommitPhases(backing)
    monkeypatch.setattr(registered, "_CONTINUATION_PHASES", phases)

    def commit() -> None:
        try:
            registered._commit_registered_nonfinal_continuation(
                executor,
                handoff,
                barrier,
                reservation,
                issued,
                consumed,
            )
        except BaseException as error:
            errors.append(error)

    def snapshot() -> None:
        try:
            outcomes.append(registered._registered_executor_snapshot(executor))
        except BaseException as error:
            errors.append(error)
        finally:
            snapshot_done.set()

    committer = threading.Thread(target=commit)
    reader = threading.Thread(target=snapshot)
    committer.start()
    assert assignment_entered.wait(timeout=5.0)
    reader.start()
    assert not snapshot_done.wait(timeout=0.1)
    release_assignment.set()
    committer.join(timeout=5.0)
    reader.join(timeout=5.0)
    assert not committer.is_alive() and not reader.is_alive()
    assert errors == []
    assert len(outcomes) == 1
    assert isinstance(outcomes[0], registered._RegisteredExecutorSnapshot)
    assert outcomes[0].phase == "AWAITING_EVALUATION"
    assert state.phase == "READY_TO_TRAIN"
    assert state.zero_based_epoch == 1
    assert backing[reservation] == "COMMITTED"
    assert executor not in registered._FAILED_EXECUTOR_HISTORY
    assert handoff not in registered._FAILED_HANDOFF_HISTORY


def test_retired_handoff_keeps_issued_evaluator_evidence_verifiable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, state, handoff, barrier, issued, _, model = _claimed_nonfinal_boundary(
        monkeypatch
    )
    handoff_snapshot = registered._registered_epoch_handoff_snapshot(handoff)
    validation_evidence = object.__new__(RegisteredValidationEvidence)
    model_tensors = object.__new__(RegisteredModelTensorEvidence)
    monkeypatch.setattr(
        VerifiedRunRegistration,
        "head_commit",
        property(lambda _self: "1" * 40),
    )
    monkeypatch.setattr(
        VerifiedRunRegistration,
        "registration_sha256",
        property(lambda _self: "2" * 64),
    )
    monkeypatch.setattr(
        VerifiedRunRegistration,
        "source_bundle_sha256",
        property(lambda _self: "3" * 64),
    )
    monkeypatch.setattr(
        RegisteredValidationInputs,
        "validation_inputs_sha256",
        property(lambda _self: "4" * 64),
    )
    monkeypatch.setattr(
        RegisteredModelTensorEvidence,
        "sha256",
        property(lambda _self: handoff_snapshot.runtime_digests.model_sha256),
    )
    result = registered_evaluator._issue_registered_evaluated_epoch(
        registration=state.registration,
        handoff=handoff,
        handoff_snapshot=handoff_snapshot,
        validation_inputs=state.validation_inputs,
        validation_evidence=validation_evidence,
        model_tensors=model_tensors,
        validation_predictions_sha256="5" * 64,
        kernel_result=registered_evaluator._KernelResult(
            example_count=10_583,
            batch_sizes=(128,) * 82 + (87,),
            frame_97_logits_payload=b"",
            model_tensor_sha256=handoff_snapshot.runtime_digests.model_sha256,
            torch_rng_sha256=handoff_snapshot.runtime_digests.rng_sha256,
            model_training_after=True,
        ),
    )
    result_state = registered_evaluator._RESULTS[result]
    issued = replace(
        issued,
        evaluated_epoch=result,
        evaluated_authority_sha256=result_state.authority_sha256,
    )
    consumed = replace(issued, phase="CONSUMED")
    monkeypatch.setattr(
        registered,
        "_preflight_registered_history_barrier",
        lambda observed_executor, observed_barrier: issued,
    )
    monkeypatch.setattr(
        registered,
        "_consume_registered_continuation_history",
        lambda reservation: consumed,
    )
    registered.advance_registered_training_executor(executor, barrier)
    model.training = False

    monkeypatch.setattr(
        registered_evaluator,
        "reverify_verified_run_registration",
        lambda value: None,
    )
    monkeypatch.setattr(
        registered_evaluator,
        "verify_registered_validation_inputs",
        lambda value: None,
    )
    monkeypatch.setattr(
        registered_evaluator,
        "verify_registered_validation_evidence",
        lambda value: None,
    )
    monkeypatch.setattr(
        registered_evaluator,
        "verify_registered_model_tensor_evidence",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        registered_evaluator,
        "_registered_validation_evidence_snapshot",
        lambda value: SimpleNamespace(
            validation_inputs_sha256="4" * 64,
            validation_predictions_sha256="5" * 64,
        ),
    )
    registered_evaluator.verify_registered_evaluated_epoch(result)
    assert registered_evaluator._registered_evaluated_epoch_snapshot(
        result
    ).handoff is (handoff)


@pytest.mark.parametrize("failpoint", ["reservation", "consume", "lifecycle"])
def test_nonfinal_advance_failpoints_are_terminal_and_never_reuse_history(
    monkeypatch: pytest.MonkeyPatch,
    failpoint: str,
) -> None:
    executor, state, handoff, barrier, issued, consumed, _ = _claimed_nonfinal_boundary(
        monkeypatch
    )
    consume_count = 0
    abort_count = 0
    original_publish = registered._publish_executor_lifecycle

    monkeypatch.setattr(
        registered,
        "_preflight_registered_history_barrier",
        lambda observed_executor, observed_barrier: issued,
    )

    def publish(
        observed_executor: registered.RegisteredTrainingExecutor,
        observed_state: registered._RegisteredExecutorState,
    ) -> None:
        if (
            failpoint == "reservation"
            and observed_state.continuation_reservation is not None
        ) or (failpoint == "lifecycle" and observed_state.phase == "READY_TO_TRAIN"):
            raise RuntimeError(f"{failpoint} failpoint")
        original_publish(observed_executor, observed_state)

    def consume(
        reservation: registered._ContinuationReservation,
    ) -> registered_history._RegisteredHistoryBarrierSnapshot:
        nonlocal consume_count
        consume_count += 1
        if failpoint == "consume":
            raise RuntimeError("consume failpoint")
        return consumed

    def abort(
        registration: VerifiedRunRegistration,
        observed_executor: registered.RegisteredTrainingExecutor,
        observed_barrier: registered_history.RegisteredHistoryBarrier,
    ) -> None:
        nonlocal abort_count
        assert registration is state.registration
        assert observed_executor is executor
        assert observed_barrier is barrier
        assert state.phase == "FAILED"
        assert handoff in registered._FAILED_HANDOFF_HISTORY
        abort_count += 1

    monkeypatch.setattr(registered, "_publish_executor_lifecycle", publish)
    monkeypatch.setattr(
        registered,
        "_consume_registered_continuation_history",
        consume,
    )
    monkeypatch.setattr(registered, "_abort_consumed_registered_history", abort)

    with pytest.raises(RuntimeError, match="failpoint"):
        registered.advance_registered_training_executor(executor, barrier)
    assert state.phase == "FAILED"
    assert executor in registered._FAILED_EXECUTOR_HISTORY
    assert handoff in registered._FAILED_HANDOFF_HISTORY
    assert consume_count == (0 if failpoint == "reservation" else 1)
    assert abort_count == (1 if failpoint in ("consume", "lifecycle") else 0)
    reservations = [
        item for item in registered._CONTINUATION_HISTORY if item.executor is executor
    ]
    assert len(reservations) == 1
    assert registered._CONTINUATION_PHASES[reservations[0]] == "FAILED"

    with pytest.raises(registered.Experiment002RegisteredExecutorError):
        registered.advance_registered_training_executor(executor, barrier)
    assert consume_count == (0 if failpoint == "reservation" else 1)


def test_history_attempt_invalid_return_still_aborts_after_terminal_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, state, handoff, barrier, issued, _, _ = _claimed_nonfinal_boundary(
        monkeypatch
    )
    aborted: list[bool] = []
    monkeypatch.setattr(
        registered,
        "_preflight_registered_history_barrier",
        lambda observed_executor, observed_barrier: issued,
    )
    monkeypatch.setattr(
        registered,
        "_consume_registered_continuation_history",
        lambda reservation: cast(Any, object()),
    )

    def abort(*args: object) -> None:
        del args
        assert state.phase == "FAILED"
        assert handoff in registered._FAILED_HANDOFF_HISTORY
        aborted.append(True)

    monkeypatch.setattr(registered, "_abort_consumed_registered_history", abort)
    with pytest.raises(registered.Experiment002RegisteredExecutorError):
        registered.advance_registered_training_executor(executor, barrier)
    assert aborted == [True]


def test_history_attempt_cleanup_runs_even_if_terminal_cleanup_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, _, _, barrier, issued, _, _ = _claimed_nonfinal_boundary(monkeypatch)
    aborted: list[bool] = []
    monkeypatch.setattr(
        registered,
        "_preflight_registered_history_barrier",
        lambda observed_executor, observed_barrier: issued,
    )
    monkeypatch.setattr(
        registered,
        "_consume_registered_continuation_history",
        lambda reservation: cast(Any, object()),
    )

    def terminal(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("terminal cleanup failpoint")

    def abort(*args: object) -> None:
        del args
        aborted.append(True)

    monkeypatch.setattr(
        registered,
        "_terminal_fail_registered_continuation",
        terminal,
    )
    monkeypatch.setattr(registered, "_abort_consumed_registered_history", abort)
    with pytest.raises(RuntimeError, match="terminal cleanup failpoint"):
        registered.advance_registered_training_executor(executor, barrier)
    assert aborted == [True]


def test_consumed_continuation_frame_type_confusion_aborts_after_poison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, state, handoff, barrier, issued, consumed, _ = _claimed_nonfinal_boundary(
        monkeypatch
    )
    aborted: list[bool] = []
    monkeypatch.setattr(
        registered,
        "_preflight_registered_history_barrier",
        lambda observed_executor, observed_barrier: issued,
    )

    def consume(
        reservation: registered._ContinuationReservation,
    ) -> registered_history._RegisteredHistoryBarrierSnapshot:
        object.__setattr__(reservation, "zero_based_epoch", False)
        return consumed

    def abort(*args: object) -> None:
        del args
        assert state.phase == "FAILED"
        aborted.append(True)

    monkeypatch.setattr(
        registered,
        "_consume_registered_continuation_history",
        consume,
    )
    monkeypatch.setattr(registered, "_abort_consumed_registered_history", abort)
    with pytest.raises(registered.Experiment002RegisteredExecutorError):
        registered.advance_registered_training_executor(executor, barrier)
    assert aborted == [True]
    assert handoff in registered._FAILED_HANDOFF_HISTORY
    object.__setattr__(
        next(
            item
            for item in registered._CONTINUATION_HISTORY
            if item.executor is executor
        ),
        "zero_based_epoch",
        0,
    )
    with pytest.raises(registered.Experiment002RegisteredExecutorError):
        registered._registered_epoch_handoff_snapshot(handoff)


def test_retired_handoff_store_and_scalar_rollback_are_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, _, handoff, barrier, issued, consumed, _ = _claimed_nonfinal_boundary(
        monkeypatch
    )
    monkeypatch.setattr(
        registered,
        "_preflight_registered_history_barrier",
        lambda observed_executor, observed_barrier: issued,
    )
    monkeypatch.setattr(
        registered,
        "_consume_registered_continuation_history",
        lambda reservation: consumed,
    )
    registered.advance_registered_training_executor(executor, barrier)
    binding = registered._RETIRED_HANDOFFS.pop(handoff)
    with pytest.raises(registered.Experiment002RegisteredExecutorError):
        registered._registered_epoch_handoff_snapshot(handoff)
    registered._RETIRED_HANDOFFS[handoff] = binding
    with pytest.raises(registered.Experiment002RegisteredExecutorError):
        registered._registered_epoch_handoff_snapshot(handoff)


@pytest.mark.parametrize(
    "tamper",
    [
        "state_epoch_bool",
        "state_phase_subclass",
        "state_rng_bytearray",
        "guard_process_subclass",
        "ticket_process_subclass",
        "lifecycle_epoch_bool",
    ],
)
def test_post_advance_executor_exact_type_tamper_is_irreversible(
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    executor, state, _, _ = _completed_nonfinal_advance(monkeypatch)
    guard = registered._EXECUTOR_GUARDS[executor]
    ticket = state.ticket
    lifecycle = registered._EXECUTOR_LIFECYCLES[executor]
    original = (
        state.phase,
        state.zero_based_epoch,
        state.expected,
        guard.process_id,
        ticket.process_id,
        lifecycle,
    )
    if tamper == "state_epoch_bool":
        state.zero_based_epoch = True
    elif tamper == "state_phase_subclass":
        state.phase = cast(
            registered.RegisteredExecutorPhase, _StrSubclass(state.phase)
        )
    elif tamper == "state_rng_bytearray":
        state.expected = replace(
            state.expected,
            rng_state=cast(bytes, bytearray(state.expected.rng_state)),
        )
    elif tamper == "guard_process_subclass":
        object.__setattr__(guard, "process_id", _IntSubclass(guard.process_id))
    elif tamper == "ticket_process_subclass":
        object.__setattr__(ticket, "process_id", _IntSubclass(ticket.process_id))
    else:
        registered._EXECUTOR_LIFECYCLES[executor] = replace(
            lifecycle,
            zero_based_epoch=True,
        )

    with pytest.raises(registered.Experiment002RegisteredExecutorError):
        registered._registered_executor_snapshot(executor)

    state.phase = original[0]
    state.zero_based_epoch = original[1]
    state.expected = original[2]
    object.__setattr__(guard, "process_id", original[3])
    object.__setattr__(ticket, "process_id", original[4])
    registered._EXECUTOR_LIFECYCLES[executor] = original[5]
    with pytest.raises(registered.Experiment002RegisteredExecutorError):
        registered._registered_executor_snapshot(executor)


def test_handoff_exact_payload_and_lifecycle_type_tamper_is_irreversible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, state, handoff, _, _, _, _ = _claimed_nonfinal_boundary(monkeypatch)
    handoff_state = registered._HANDOFFS[handoff]
    guard = registered._HANDOFF_GUARDS[handoff]
    lifecycle = registered._HANDOFF_LIFECYCLES[handoff]
    original = (
        handoff_state.zero_based_epoch,
        handoff_state.optimizer_generation,
        handoff_state.runtime_digests,
        guard.zero_based_epoch,
        lifecycle,
    )
    handoff_state.zero_based_epoch = False
    handoff_state.optimizer_generation = _IntSubclass(
        handoff_state.optimizer_generation
    )
    handoff_state.runtime_digests = replace(
        handoff_state.runtime_digests,
        rng_state=cast(bytes, bytearray(handoff_state.runtime_digests.rng_state)),
    )
    object.__setattr__(guard, "zero_based_epoch", False)
    registered._HANDOFF_LIFECYCLES[handoff] = registered._HandoffLifecycle(
        used=cast(bool, 1),
        failed=False,
    )
    with pytest.raises(registered.Experiment002RegisteredExecutorError):
        registered._registered_epoch_handoff_snapshot(handoff)

    handoff_state.zero_based_epoch = original[0]
    handoff_state.optimizer_generation = original[1]
    handoff_state.runtime_digests = original[2]
    handoff_state.failed = False
    object.__setattr__(guard, "zero_based_epoch", original[3])
    registered._HANDOFF_LIFECYCLES[handoff] = original[4]
    state.phase = "AWAITING_EVALUATION"
    with pytest.raises(registered.Experiment002RegisteredExecutorError):
        registered._registered_epoch_handoff_snapshot(handoff)


def test_retired_continuation_phase_subclass_is_terminal_after_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, handoff, _ = _completed_nonfinal_advance(monkeypatch)
    binding = registered._RETIRED_HANDOFFS[handoff]
    registered._CONTINUATION_PHASES[binding.reservation] = cast(
        registered._ContinuationPhase,
        _StrSubclass("COMMITTED"),
    )
    with pytest.raises(registered.Experiment002RegisteredExecutorError):
        registered._registered_epoch_handoff_snapshot(handoff)
    registered._CONTINUATION_PHASES[binding.reservation] = "COMMITTED"
    with pytest.raises(registered.Experiment002RegisteredExecutorError):
        registered._registered_epoch_handoff_snapshot(handoff)


def test_final_history_barrier_is_rejected_without_trace_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, state, handoff, barrier, issued, _, _ = _claimed_nonfinal_boundary(
        monkeypatch
    )
    final = replace(
        issued,
        accepted_epoch=29,
        next_zero_based_epoch=None,
        phase="ISSUED",
    )
    monkeypatch.setattr(
        registered_history,
        "verify_registered_history_barrier",
        lambda value: None,
    )
    monkeypatch.setattr(
        registered_history,
        "_registered_history_barrier_snapshot",
        lambda value: final,
    )
    with pytest.raises(
        registered.Experiment002RegisteredExecutorError,
        match="non-final",
    ):
        registered.advance_registered_training_executor(executor, barrier)
    assert state.phase == "AWAITING_EVALUATION"
    assert handoff not in registered._FAILED_HANDOFF_HISTORY
    assert not hasattr(state.trace, "finish")


def test_unregistered_eval_mode_digest_branch_preserves_exact_runtime() -> None:
    probe = numeric._create_synthetic_training_executor(seed=61_337)
    runtime = numeric._issued_executor_state(probe)
    before = numeric._stable_runtime_digests(runtime, expected_generation=0)
    runtime.model.eval()
    observed = registered._stable_handoff_runtime_digests(
        runtime,
        expected_generation=0,
        expected_training=False,
    )
    assert observed == before
    assert runtime.model.training is False
    assert all(module.training is False for module in runtime.model.modules())
    with pytest.raises(
        registered.Experiment002RegisteredExecutorError, match="mode differs"
    ):
        registered._stable_handoff_runtime_digests(
            runtime,
            expected_generation=0,
            expected_training=True,
        )


def test_coherent_executor_clone_full_store_rewrite_cannot_mint_truth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, state, _, _ = _issue_counterfeit_bound_executor(monkeypatch)
    clone = object.__new__(registered.RegisteredTrainingExecutor)
    forged_ticket = registered._ExecutorIssuanceTicket(
        executor=clone,
        token=object(),
        registration=state.registration,
        process_id=state.process_id,
    )
    forged_anchor = replace(state.anchor, executor=clone, ticket=forged_ticket)
    forged_state = replace(state, anchor=forged_anchor, ticket=forged_ticket)
    forged_guard = registered._guard_from_anchor(forged_anchor)
    forged_lifecycle = registered._lifecycle_from_state(forged_state)
    forged_lifecycle_authority = registered._RegisteredExecutorLifecycleAuthority(
        executor=clone,
        sequence=0,
        lifecycle=forged_lifecycle,
        previous=None,
    )

    saved_executors = dict(registered._EXECUTORS.items())
    saved_guards = dict(registered._EXECUTOR_GUARDS.items())
    saved_lifecycles = dict(registered._EXECUTOR_LIFECYCLES.items())
    saved_heads = dict(registered._EXECUTOR_LIFECYCLE_HEADS)
    saved_lifecycle_history = list(registered._EXECUTOR_LIFECYCLE_HISTORY)
    saved_issued_lifecycles = set(registered._ISSUED_EXECUTOR_LIFECYCLES)
    saved_anchors = dict(registered._EXECUTOR_ANCHORS)
    saved_anchor_history = list(registered._EXECUTOR_ANCHOR_HISTORY)
    saved_tickets = set(registered._ISSUED_EXECUTOR_TICKETS)
    saved_ticket_history = list(registered._EXECUTOR_TICKET_HISTORY)
    propagated: list[str] = []
    monkeypatch.setattr(
        registered,
        "_propagate_executor_failure",
        lambda *args: propagated.append("propagated"),
    )
    injected_truth = SimpleNamespace(issue_executor=lambda *args: None)
    monkeypatch.setattr(
        registered,
        "_AUTHORITY_TRUTH",
        injected_truth,
        raising=False,
    )
    injected_truth.issue_executor(clone, forged_ticket, forged_lifecycle)
    try:
        registered._EXECUTORS.clear()
        registered._EXECUTORS[clone] = forged_state
        registered._EXECUTOR_GUARDS.clear()
        registered._EXECUTOR_GUARDS[clone] = forged_guard
        registered._EXECUTOR_LIFECYCLES.clear()
        registered._EXECUTOR_LIFECYCLES[clone] = forged_lifecycle
        registered._EXECUTOR_LIFECYCLE_HEADS.clear()
        registered._EXECUTOR_LIFECYCLE_HEADS[clone] = forged_lifecycle_authority
        registered._EXECUTOR_LIFECYCLE_HISTORY.clear()
        registered._EXECUTOR_LIFECYCLE_HISTORY.append(forged_lifecycle_authority)
        registered._ISSUED_EXECUTOR_LIFECYCLES.clear()
        registered._ISSUED_EXECUTOR_LIFECYCLES.add(forged_lifecycle_authority)
        registered._EXECUTOR_ANCHORS.clear()
        registered._EXECUTOR_ANCHORS[clone] = forged_anchor
        registered._EXECUTOR_ANCHOR_HISTORY.clear()
        registered._EXECUTOR_ANCHOR_HISTORY.append(forged_anchor)
        registered._ISSUED_EXECUTOR_TICKETS.clear()
        registered._ISSUED_EXECUTOR_TICKETS.add(forged_ticket)
        registered._EXECUTOR_TICKET_HISTORY.clear()
        registered._EXECUTOR_TICKET_HISTORY.append(forged_ticket)
        with pytest.raises(registered.Experiment002RegisteredExecutorError):
            registered._registered_executor_snapshot(clone)
        assert propagated == []
    finally:
        registered._EXECUTORS.clear()
        registered._EXECUTORS.update(saved_executors)
        registered._EXECUTOR_GUARDS.clear()
        registered._EXECUTOR_GUARDS.update(saved_guards)
        registered._EXECUTOR_LIFECYCLES.clear()
        registered._EXECUTOR_LIFECYCLES.update(saved_lifecycles)
        registered._EXECUTOR_LIFECYCLE_HEADS.clear()
        registered._EXECUTOR_LIFECYCLE_HEADS.update(saved_heads)
        registered._EXECUTOR_LIFECYCLE_HISTORY.clear()
        registered._EXECUTOR_LIFECYCLE_HISTORY.extend(saved_lifecycle_history)
        registered._ISSUED_EXECUTOR_LIFECYCLES.clear()
        registered._ISSUED_EXECUTOR_LIFECYCLES.update(saved_issued_lifecycles)
        registered._EXECUTOR_ANCHORS.clear()
        registered._EXECUTOR_ANCHORS.update(saved_anchors)
        registered._EXECUTOR_ANCHOR_HISTORY.clear()
        registered._EXECUTOR_ANCHOR_HISTORY.extend(saved_anchor_history)
        registered._ISSUED_EXECUTOR_TICKETS.clear()
        registered._ISSUED_EXECUTOR_TICKETS.update(saved_tickets)
        registered._EXECUTOR_TICKET_HISTORY.clear()
        registered._EXECUTOR_TICKET_HISTORY.extend(saved_ticket_history)
    assert registered._registered_executor_snapshot(executor).phase == (
        "READY_TO_TRAIN"
    )


def test_failed_executor_full_module_rollback_cannot_erase_closure_truth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, state, _, _ = _issue_counterfeit_bound_executor(monkeypatch)
    initial_head = registered._EXECUTOR_LIFECYCLE_HEADS[executor]
    monkeypatch.setattr(registered, "_propagate_executor_failure", lambda *args: None)
    registered._terminal_fail_executor(executor, state)

    registered._FAILED_EXECUTORS.discard(executor)
    registered._FAILED_EXECUTOR_HISTORY.discard(executor)
    registered._EXECUTOR_FAILURE_EVENTS.difference_update(
        {
            event
            for event in registered._EXECUTOR_FAILURE_EVENTS
            if event.ticket is state.ticket
        }
    )
    registered._EXECUTOR_FAILURE_EVENT_HISTORY[:] = [
        event
        for event in registered._EXECUTOR_FAILURE_EVENT_HISTORY
        if event.ticket is not state.ticket
    ]
    state.phase = "READY_TO_TRAIN"
    registered._EXECUTOR_LIFECYCLES[executor] = initial_head.lifecycle
    registered._EXECUTOR_LIFECYCLE_HEADS[executor] = initial_head
    registered._EXECUTOR_LIFECYCLE_HISTORY[:] = [
        item
        for item in registered._EXECUTOR_LIFECYCLE_HISTORY
        if item.executor is not executor or item is initial_head
    ]
    registered._ISSUED_EXECUTOR_LIFECYCLES.intersection_update(
        registered._EXECUTOR_LIFECYCLE_HISTORY
    )
    with pytest.raises(registered.Experiment002RegisteredExecutorError):
        registered._registered_executor_snapshot(executor)
    assert executor in registered._FAILED_EXECUTOR_HISTORY


def test_coherent_handoff_clone_full_store_rewrite_cannot_poison_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, state, _, trace_probe = _issue_counterfeit_bound_executor(monkeypatch)
    population = _counterfeit_exact(CompletedTrainingPopulation)
    state.optimizer_generation = 313
    state.runtime.optimizer_generation = 313
    state.phase = "TRAINING"
    state.active_epoch = _counterfeit_exact(RegisteredTrainingEpoch)
    registered._publish_executor_lifecycle(executor, state)
    handoff = registered._issue_epoch_handoff(
        executor,
        state,
        population=population,
        epoch_trace=trace_probe.epoch_trace,
    )
    state.active_handoff = handoff
    state.active_epoch = None
    state.phase = "AWAITING_EVALUATION"
    registered._publish_executor_lifecycle(executor, state)
    original_state = registered._HANDOFFS[handoff]
    clone = object.__new__(registered.RegisteredExecutorEpochHandoff)
    forged_ticket = registered._HandoffIssuanceTicket(
        handoff=clone,
        executor_ticket=state.ticket,
        handoff_token=object(),
    )
    forged_anchor = replace(original_state.anchor, handoff=clone, ticket=forged_ticket)
    forged_state = replace(
        original_state,
        anchor=forged_anchor,
        ticket=forged_ticket,
    )
    forged_guard = registered._handoff_guard_from_state(forged_state)
    forged_lifecycle = registered._HandoffLifecycle(False, False)
    forged_lifecycle_authority = registered._HandoffLifecycleAuthority(
        handoff=clone,
        sequence=0,
        lifecycle=forged_lifecycle,
        previous=None,
    )
    saved_handoffs = dict(registered._HANDOFFS.items())
    saved_guards = dict(registered._HANDOFF_GUARDS.items())
    saved_lifecycles = dict(registered._HANDOFF_LIFECYCLES.items())
    saved_heads = dict(registered._HANDOFF_LIFECYCLE_HEADS)
    saved_lifecycle_history = list(registered._HANDOFF_LIFECYCLE_HISTORY)
    saved_issued_lifecycles = set(registered._ISSUED_HANDOFF_LIFECYCLES)
    saved_anchors = dict(registered._HANDOFF_ANCHORS)
    saved_anchor_history = list(registered._HANDOFF_ANCHOR_HISTORY)
    saved_tickets = set(registered._ISSUED_HANDOFF_TICKETS)
    saved_ticket_history = list(registered._HANDOFF_TICKET_HISTORY)
    injected_truth = SimpleNamespace(issue_handoff=lambda *args: None)
    monkeypatch.setattr(
        registered,
        "_AUTHORITY_TRUTH",
        injected_truth,
        raising=False,
    )
    injected_truth.issue_handoff(clone, forged_ticket, forged_lifecycle)
    try:
        registered._HANDOFFS.clear()
        registered._HANDOFFS[clone] = forged_state
        registered._HANDOFF_GUARDS.clear()
        registered._HANDOFF_GUARDS[clone] = forged_guard
        registered._HANDOFF_LIFECYCLES.clear()
        registered._HANDOFF_LIFECYCLES[clone] = forged_lifecycle
        registered._HANDOFF_LIFECYCLE_HEADS.clear()
        registered._HANDOFF_LIFECYCLE_HEADS[clone] = forged_lifecycle_authority
        registered._HANDOFF_LIFECYCLE_HISTORY.clear()
        registered._HANDOFF_LIFECYCLE_HISTORY.append(forged_lifecycle_authority)
        registered._ISSUED_HANDOFF_LIFECYCLES.clear()
        registered._ISSUED_HANDOFF_LIFECYCLES.add(forged_lifecycle_authority)
        registered._HANDOFF_ANCHORS.clear()
        registered._HANDOFF_ANCHORS[clone] = forged_anchor
        registered._HANDOFF_ANCHOR_HISTORY.clear()
        registered._HANDOFF_ANCHOR_HISTORY.append(forged_anchor)
        registered._ISSUED_HANDOFF_TICKETS.clear()
        registered._ISSUED_HANDOFF_TICKETS.add(forged_ticket)
        registered._HANDOFF_TICKET_HISTORY.clear()
        registered._HANDOFF_TICKET_HISTORY.append(forged_ticket)
        with pytest.raises(registered.Experiment002RegisteredExecutorError):
            registered._registered_epoch_handoff_snapshot(clone)
        assert executor not in registered._FAILED_EXECUTOR_HISTORY
    finally:
        registered._HANDOFFS.clear()
        registered._HANDOFFS.update(saved_handoffs)
        registered._HANDOFF_GUARDS.clear()
        registered._HANDOFF_GUARDS.update(saved_guards)
        registered._HANDOFF_LIFECYCLES.clear()
        registered._HANDOFF_LIFECYCLES.update(saved_lifecycles)
        registered._HANDOFF_LIFECYCLE_HEADS.clear()
        registered._HANDOFF_LIFECYCLE_HEADS.update(saved_heads)
        registered._HANDOFF_LIFECYCLE_HISTORY.clear()
        registered._HANDOFF_LIFECYCLE_HISTORY.extend(saved_lifecycle_history)
        registered._ISSUED_HANDOFF_LIFECYCLES.clear()
        registered._ISSUED_HANDOFF_LIFECYCLES.update(saved_issued_lifecycles)
        registered._HANDOFF_ANCHORS.clear()
        registered._HANDOFF_ANCHORS.update(saved_anchors)
        registered._HANDOFF_ANCHOR_HISTORY.clear()
        registered._HANDOFF_ANCHOR_HISTORY.extend(saved_anchor_history)
        registered._ISSUED_HANDOFF_TICKETS.clear()
        registered._ISSUED_HANDOFF_TICKETS.update(saved_tickets)
        registered._HANDOFF_TICKET_HISTORY.clear()
        registered._HANDOFF_TICKET_HISTORY.extend(saved_ticket_history)

    monkeypatch.setattr(
        registered, "reverify_verified_run_registration", lambda value: None
    )
    monkeypatch.setattr(
        numeric,
        "_stable_runtime_digests",
        lambda value, *, expected_generation: state.expected,
    )
    assert registered._registered_epoch_handoff_snapshot(handoff).executor is executor


def test_used_handoff_full_module_rollback_cannot_enable_second_model_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, state, model, trace_probe = _issue_counterfeit_bound_executor(monkeypatch)
    population = _counterfeit_exact(CompletedTrainingPopulation)
    state.optimizer_generation = 313
    state.runtime.optimizer_generation = 313
    state.phase = "TRAINING"
    state.active_epoch = _counterfeit_exact(RegisteredTrainingEpoch)
    registered._publish_executor_lifecycle(executor, state)
    handoff = registered._issue_epoch_handoff(
        executor,
        state,
        population=population,
        epoch_trace=trace_probe.epoch_trace,
    )
    state.active_handoff = handoff
    state.active_epoch = None
    state.phase = "AWAITING_EVALUATION"
    registered._publish_executor_lifecycle(executor, state)
    monkeypatch.setattr(
        registered, "reverify_verified_run_registration", lambda value: None
    )
    monkeypatch.setattr(
        numeric,
        "_stable_runtime_digests",
        lambda value, *, expected_generation: state.expected,
    )
    initial_head = registered._HANDOFF_LIFECYCLE_HEADS[handoff]
    assert (
        registered._registered_epoch_handoff_model(state.registration, handoff) is model
    )
    handoff_state = registered._HANDOFFS[handoff]
    assert handoff_state.used

    handoff_state.used = False
    registered._HANDOFF_EVENTS.difference_update(
        {
            event
            for event in registered._HANDOFF_EVENTS
            if event.ticket is handoff_state.ticket
        }
    )
    registered._HANDOFF_EVENT_HISTORY[:] = [
        event
        for event in registered._HANDOFF_EVENT_HISTORY
        if event.ticket is not handoff_state.ticket
    ]
    registered._HANDOFF_LIFECYCLES[handoff] = initial_head.lifecycle
    registered._HANDOFF_LIFECYCLE_HEADS[handoff] = initial_head
    registered._HANDOFF_LIFECYCLE_HISTORY[:] = [
        item
        for item in registered._HANDOFF_LIFECYCLE_HISTORY
        if item.handoff is not handoff or item is initial_head
    ]
    registered._ISSUED_HANDOFF_LIFECYCLES.intersection_update(
        registered._HANDOFF_LIFECYCLE_HISTORY
    )
    with pytest.raises(registered.Experiment002RegisteredExecutorError):
        registered._registered_epoch_handoff_model(state.registration, handoff)
    assert handoff in registered._FAILED_HANDOFF_HISTORY
    assert executor in registered._FAILED_EXECUTOR_HISTORY


def test_registered_completion_publishes_exact_terminal_authority_and_late_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final = _claimed_final_boundary(monkeypatch)
    before = registered._registered_epoch_handoff_snapshot(final.handoff)

    completed = registered.complete_registered_training_executor(
        final.executor,
        final.barrier,
    )

    assert completed is final.completed_history
    assert final.calls == {"trace": 1, "history": 1, "abort": 0}
    snapshot = registered._registered_executor_snapshot(final.executor)
    assert snapshot.phase == "COMPLETE"
    assert snapshot.zero_based_epoch == 29
    assert snapshot.optimizer_generation == 30 * 313
    assert snapshot.previous_history_barrier is final.previous_barrier
    assert snapshot.final_barrier is final.barrier
    assert snapshot.complete_update_trace is final.complete_trace
    assert snapshot.completed_history is completed
    assert final.state.active_handoff is None
    assert final.state.continuation_reservation is None
    assert final.state.finalization_reservation is not None
    assert final.model.training is False
    binding = registered._FINALIZED_HANDOFFS[final.handoff]
    assert binding.final_barrier is final.barrier
    assert binding.complete_update_trace is final.complete_trace
    assert binding.completed_history is completed
    assert registered._registered_epoch_handoff_snapshot(final.handoff) == before
    with pytest.raises(
        registered.Experiment002RegisteredExecutorError,
        match="consumed|archived",
    ):
        registered._registered_epoch_handoff_model(
            final.state.registration,
            final.handoff,
        )
    assert final.state.phase == "COMPLETE"

    late = registered.complete_registered_training_executor(
        final.executor,
        final.barrier,
    )
    assert late is completed
    assert final.calls == {"trace": 1, "history": 1, "abort": 0}


def test_concurrent_registered_completion_runs_irreversible_work_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final = _claimed_final_boundary(monkeypatch)
    trace_entered = threading.Event()
    release_trace = threading.Event()
    follower_admitted = threading.Event()
    original_begin = registered._begin_completion_admission

    def begin(
        executor: registered.RegisteredTrainingExecutor,
        barrier: RegisteredHistoryBarrier,
        caller_token: object | None = None,
    ) -> registered._CompletionAdmission:
        admission = original_begin(executor, barrier, caller_token)
        if not admission.leader:
            follower_admitted.set()
        return admission

    def finish_trace(
        reservation: registered._FinalizationReservation,
    ) -> CompleteUpdateTraceEvidence:
        assert reservation.executor is final.executor
        final.calls["trace"] += 1
        trace_entered.set()
        assert release_trace.wait(timeout=2.0)
        return final.complete_trace

    monkeypatch.setattr(registered, "_begin_completion_admission", begin)
    monkeypatch.setattr(
        registered, "_finish_registered_finalization_trace", finish_trace
    )
    outcomes: list[object] = []

    def run() -> None:
        try:
            outcomes.append(
                registered.complete_registered_training_executor(
                    final.executor,
                    final.barrier,
                )
            )
        except BaseException as error:
            outcomes.append(error)

    leader = threading.Thread(target=run)
    follower = threading.Thread(target=run)
    leader.start()
    assert trace_entered.wait(timeout=2.0)
    follower.start()
    assert follower_admitted.wait(timeout=2.0)
    release_trace.set()
    leader.join(timeout=5.0)
    follower.join(timeout=5.0)
    assert not leader.is_alive() and not follower.is_alive()
    assert outcomes == [final.completed_history, final.completed_history]
    assert outcomes[0] is outcomes[1]
    assert final.calls == {"trace": 1, "history": 1, "abort": 0}


def test_boundary_conflicts_are_not_admitted_and_do_not_poison_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final = _claimed_final_boundary(monkeypatch)
    trace_entered = threading.Event()
    release_trace = threading.Event()

    def finish_trace(
        reservation: registered._FinalizationReservation,
    ) -> CompleteUpdateTraceEvidence:
        del reservation
        final.calls["trace"] += 1
        trace_entered.set()
        assert release_trace.wait(timeout=2.0)
        return final.complete_trace

    monkeypatch.setattr(
        registered, "_finish_registered_finalization_trace", finish_trace
    )
    outcomes: list[object] = []

    def complete() -> None:
        try:
            outcomes.append(
                registered.complete_registered_training_executor(
                    final.executor,
                    final.barrier,
                )
            )
        except BaseException as error:
            outcomes.append(error)

    leader = threading.Thread(target=complete)
    leader.start()
    assert trace_entered.wait(timeout=2.0)
    with pytest.raises(
        registered.Experiment002RegisteredExecutorError,
        match="different executor boundary",
    ):
        registered.advance_registered_training_executor(
            final.executor,
            final.barrier,
        )
    with pytest.raises(
        registered.Experiment002RegisteredExecutorError,
        match="different executor boundary",
    ):
        registered.complete_registered_training_executor(
            final.executor,
            object.__new__(RegisteredHistoryBarrier),
        )
    assert final.state.phase == "AWAITING_EVALUATION"
    assert final.executor not in registered._FAILED_EXECUTOR_HISTORY
    assert final.handoff not in registered._FAILED_HANDOFF_HISTORY
    release_trace.set()
    leader.join(timeout=5.0)
    assert not leader.is_alive()
    assert outcomes == [final.completed_history]
    assert registered._registered_executor_snapshot(final.executor).phase == "COMPLETE"


@pytest.mark.parametrize("stage", ["trace", "history", "commit"])
def test_registered_completion_failpoints_poison_locally_before_history_abort(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    final = _claimed_final_boundary(monkeypatch)

    if stage == "trace":
        monkeypatch.setattr(
            registered,
            "_finish_registered_finalization_trace",
            lambda reservation: (_ for _ in ()).throw(RuntimeError("trace fail")),
        )
    elif stage == "history":
        monkeypatch.setattr(
            registered,
            "_complete_registered_finalization_history",
            lambda reservation, complete_trace: (_ for _ in ()).throw(
                RuntimeError("history fail")
            ),
        )
    else:
        monkeypatch.setattr(
            registered,
            "_commit_registered_finalization",
            lambda *args: (_ for _ in ()).throw(RuntimeError("commit fail")),
        )

    with pytest.raises(RuntimeError, match=f"{stage} fail"):
        registered.complete_registered_training_executor(
            final.executor,
            final.barrier,
        )

    reservation = final.state.finalization_reservation
    assert reservation is not None
    assert final.state.phase == "FAILED"
    assert registered._HANDOFFS[final.handoff].failed
    assert registered._FINALIZATION_PHASES[reservation] == "FAILED"
    assert final.calls["abort"] == 1
    assert final.abort_states == [("FAILED", True, "FAILED")]
    assert final.executor in registered._FAILED_EXECUTOR_HISTORY
    assert final.handoff in registered._FAILED_HANDOFF_HISTORY


def test_registered_completion_external_steps_run_with_executor_and_handoff_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final = _claimed_final_boundary(monkeypatch)
    observed: list[str] = []

    def require_locks_free(label: str) -> None:
        acquired: list[bool] = []

        def probe() -> None:
            executor_acquired = final.state.lock.acquire(timeout=1.0)
            handoff_acquired = False
            try:
                if executor_acquired:
                    handoff_acquired = registered._HANDOFFS_LOCK.acquire(timeout=1.0)
            finally:
                if handoff_acquired:
                    registered._HANDOFFS_LOCK.release()
                if executor_acquired:
                    final.state.lock.release()
            acquired.append(executor_acquired and handoff_acquired)

        thread = threading.Thread(target=probe)
        thread.start()
        thread.join(timeout=2.0)
        assert not thread.is_alive()
        assert acquired == [True]
        observed.append(label)

    def preflight(
        executor: registered.RegisteredTrainingExecutor,
        barrier: RegisteredHistoryBarrier,
    ) -> registered_history._RegisteredHistoryBarrierSnapshot:
        assert executor is final.executor and barrier is final.barrier
        require_locks_free("preflight")
        return final.issued

    def finish_trace(
        reservation: registered._FinalizationReservation,
    ) -> CompleteUpdateTraceEvidence:
        del reservation
        require_locks_free("trace")
        final.calls["trace"] += 1
        return final.complete_trace

    def complete_history(
        reservation: registered._FinalizationReservation,
        complete_trace: CompleteUpdateTraceEvidence,
    ) -> RegisteredCompletedTrainingHistory:
        del reservation, complete_trace
        require_locks_free("history")
        final.calls["history"] += 1
        return final.completed_history

    monkeypatch.setattr(
        registered, "_preflight_registered_final_history_barrier", preflight
    )
    monkeypatch.setattr(
        registered, "_finish_registered_finalization_trace", finish_trace
    )
    monkeypatch.setattr(
        registered, "_complete_registered_finalization_history", complete_history
    )
    assert (
        registered.complete_registered_training_executor(
            final.executor,
            final.barrier,
        )
        is final.completed_history
    )
    assert observed == ["preflight", "trace", "history"]


def test_valid_epoch_29_advance_rejection_is_nonpoisoning_then_completion_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final = _claimed_final_boundary(monkeypatch)
    monkeypatch.setattr(
        registered_history, "verify_registered_history_barrier", lambda value: None
    )
    monkeypatch.setattr(
        registered_history,
        "_registered_history_barrier_snapshot",
        lambda value: final.issued,
    )
    with pytest.raises(
        registered.Experiment002RegisteredExecutorError,
        match="non-final",
    ):
        registered.advance_registered_training_executor(
            final.executor,
            final.barrier,
        )
    assert final.state.phase == "AWAITING_EVALUATION"
    assert final.executor not in registered._FAILED_EXECUTOR_HISTORY
    assert (
        registered.complete_registered_training_executor(
            final.executor,
            final.barrier,
        )
        is final.completed_history
    )


def test_registered_completion_rejects_wrong_mode_and_barrier_subclass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BarrierSubclass(RegisteredHistoryBarrier):
        pass

    type_fixture = _claimed_final_boundary(monkeypatch)
    with pytest.raises(TypeError, match="final_barrier"):
        registered.complete_registered_training_executor(
            type_fixture.executor,
            object.__new__(_BarrierSubclass),
        )
    assert type_fixture.state.phase == "AWAITING_EVALUATION"

    mode_fixture = _claimed_final_boundary(monkeypatch)
    mode_fixture.model.training = True
    with pytest.raises(
        registered.Experiment002RegisteredExecutorError,
        match="epoch-29 boundary",
    ):
        registered.complete_registered_training_executor(
            mode_fixture.executor,
            mode_fixture.barrier,
        )
    assert mode_fixture.state.phase == "FAILED"
    assert mode_fixture.calls["abort"] == 1


def test_late_completion_reverifies_final_registry_before_returning_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final = _claimed_final_boundary(monkeypatch)
    assert (
        registered.complete_registered_training_executor(
            final.executor,
            final.barrier,
        )
        is final.completed_history
    )
    registered._FINALIZED_HANDOFFS.pop(final.handoff)

    with pytest.raises(registered.Experiment002RegisteredExecutorError):
        registered.complete_registered_training_executor(
            final.executor,
            final.barrier,
        )
    assert final.state.phase == "FAILED"
    assert registered._HANDOFFS[final.handoff].failed
    assert final.calls["abort"] == 1


def test_late_completion_reverifies_frozen_runtime_before_returning_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final = _claimed_final_boundary(monkeypatch)
    assert (
        registered.complete_registered_training_executor(
            final.executor,
            final.barrier,
        )
        is final.completed_history
    )
    changed = numeric._RuntimeDigests(
        model_sha256="4" * 64,
        optimizer_sha256="5" * 64,
        rng_sha256="6" * 64,
        rng_state=b"changed-final-rng",
    )
    monkeypatch.setattr(
        registered,
        "_stable_handoff_runtime_digests",
        lambda value, *, expected_generation, expected_training: changed,
    )

    with pytest.raises(
        registered.Experiment002RegisteredExecutorError,
        match="runtime differs",
    ):
        registered.complete_registered_training_executor(
            final.executor,
            final.barrier,
        )
    assert final.state.phase == "FAILED"
    assert registered._HANDOFFS[final.handoff].failed
    assert final.calls["abort"] == 1


def test_finalization_requires_all_29_continuations_still_committed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final = _claimed_final_boundary(monkeypatch)
    truth_fail = (
        inspect.signature(registered._terminal_fail_registered_continuation)
        .parameters["_truth_fail"]
        .default
    )
    truth_validate = (
        inspect.signature(registered._require_continuation_reservation)
        .parameters["_truth_validate"]
        .default
    )
    broken = final.prior_reservations[0]
    truth_fail(broken)
    assert truth_validate(broken, "FAILED")

    with pytest.raises(
        registered.Experiment002RegisteredExecutorError,
        match="committed continuation chain",
    ):
        registered.complete_registered_training_executor(
            final.executor,
            final.barrier,
        )

    assert final.state.phase == "FAILED"
    assert registered._HANDOFFS[final.handoff].failed
    assert final.calls["abort"] == 1
    assert truth_validate(broken, "FAILED")


def test_orphan_finalization_truth_is_recovered_after_reserve_interruption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final = _claimed_final_boundary(monkeypatch)
    original_reserve = registered._reserve_registered_finalization
    truth_reserve = (
        inspect.signature(original_reserve).parameters["_truth_reserve"].default
    )
    truth_validate = (
        inspect.signature(registered._require_finalization_reservation)
        .parameters["_truth_validate"]
        .default
    )
    captured: list[registered._FinalizationReservation] = []

    def reserve_then_interrupt(
        reservation: registered._FinalizationReservation,
    ) -> None:
        captured.append(reservation)
        truth_reserve(reservation)
        raise KeyboardInterrupt("after closure reserve")

    def reserve(
        executor: registered.RegisteredTrainingExecutor,
        handoff: registered.RegisteredExecutorEpochHandoff,
        barrier: RegisteredHistoryBarrier,
        snapshot: registered_history._RegisteredHistoryBarrierSnapshot,
    ) -> registered._FinalizationReservation:
        return original_reserve(
            executor,
            handoff,
            barrier,
            snapshot,
            _truth_reserve=reserve_then_interrupt,
        )

    monkeypatch.setattr(registered, "_reserve_registered_finalization", reserve)
    with pytest.raises(KeyboardInterrupt, match="closure reserve"):
        registered.complete_registered_training_executor(
            final.executor,
            final.barrier,
        )

    assert len(captured) == 1
    orphan = captured[0]
    assert final.state.finalization_reservation is None
    assert truth_validate(orphan, "FAILED")
    assert registered._FINALIZATION_PHASES[orphan] == "FAILED"
    assert sum(item is orphan for item in registered._FINALIZATION_HISTORY) == 1
    assert final.state.phase == "FAILED"
    assert registered._HANDOFFS[final.handoff].failed
    assert final.calls["abort"] == 1


def test_finalized_handoff_rejects_coherent_binding_identity_rewrite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final = _claimed_final_boundary(monkeypatch)
    assert (
        registered.complete_registered_training_executor(
            final.executor,
            final.barrier,
        )
        is final.completed_history
    )
    binding = registered._FINALIZED_HANDOFFS[final.handoff]
    original_trace = binding.complete_update_trace
    original_history = binding.completed_history
    foreign_trace = object.__new__(CompleteUpdateTraceEvidence)
    foreign_history = object.__new__(RegisteredCompletedTrainingHistory)
    object.__setattr__(binding, "complete_update_trace", foreign_trace)
    object.__setattr__(binding, "completed_history", foreign_history)
    try:
        with pytest.raises(registered.Experiment002RegisteredExecutorError):
            registered._registered_epoch_handoff_snapshot(final.handoff)
    finally:
        object.__setattr__(binding, "complete_update_trace", original_trace)
        object.__setattr__(binding, "completed_history", original_history)

    assert final.state.phase == "FAILED"
    assert registered._HANDOFFS[final.handoff].failed
    with pytest.raises(registered.Experiment002RegisteredExecutorError):
        registered._registered_epoch_handoff_snapshot(final.handoff)


@pytest.mark.parametrize("after_truth", [False, True])
def test_completion_finish_interruption_preserves_one_sticky_success(
    monkeypatch: pytest.MonkeyPatch,
    after_truth: bool,
) -> None:
    final = _claimed_final_boundary(monkeypatch)
    original_finish = registered._finish_completion_admission

    def interrupt(
        admission: registered._CompletionAdmission,
        result: RegisteredCompletedTrainingHistory | None,
        error: BaseException | None,
    ) -> None:
        if after_truth:
            original_finish(admission, result, error)
        raise KeyboardInterrupt("completion publication interrupted")

    monkeypatch.setattr(registered, "_finish_completion_admission", interrupt)
    with pytest.raises(KeyboardInterrupt, match="publication interrupted"):
        registered.complete_registered_training_executor(
            final.executor,
            final.barrier,
        )
    monkeypatch.setattr(
        registered,
        "_finish_completion_admission",
        original_finish,
    )

    late: list[object] = []

    def retry() -> None:
        try:
            late.append(
                registered.complete_registered_training_executor(
                    final.executor,
                    final.barrier,
                )
            )
        except BaseException as error:
            late.append(error)

    follower = threading.Thread(target=retry)
    follower.start()
    follower.join(timeout=3.0)
    assert not follower.is_alive()
    assert late == [final.completed_history]
    assert final.state.phase == "COMPLETE"
    assert final.calls == {"trace": 1, "history": 1, "abort": 0}


def test_completion_begin_interruption_recovers_only_its_active_leader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final = _claimed_final_boundary(monkeypatch)
    original_begin = registered._begin_completion_admission
    captured: list[registered._CompletionAdmission] = []

    def interrupt(
        executor: registered.RegisteredTrainingExecutor,
        barrier: RegisteredHistoryBarrier,
        caller_token: object | None = None,
    ) -> registered._CompletionAdmission:
        admission = original_begin(executor, barrier, caller_token)
        captured.append(admission)
        raise KeyboardInterrupt("after completion begin")

    monkeypatch.setattr(registered, "_begin_completion_admission", interrupt)
    with pytest.raises(KeyboardInterrupt, match="completion begin"):
        registered.complete_registered_training_executor(
            final.executor,
            final.barrier,
        )
    assert len(captured) == 1
    assert captured[0].completed.is_set()
    assert len(captured[0].outcome) == 1
    assert captured[0].outcome[0].error is not None
    assert final.state.phase == "FAILED"
    assert registered._HANDOFFS[final.handoff].failed
    assert final.calls["abort"] == 1

    monkeypatch.setattr(registered, "_begin_completion_admission", original_begin)
    with pytest.raises(registered.Experiment002RegisteredExecutorError):
        registered.complete_registered_training_executor(
            final.executor,
            final.barrier,
        )


def test_completion_admission_tamper_cannot_split_leader_and_follower_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final = _claimed_final_boundary(monkeypatch)
    original_begin = registered._begin_completion_admission
    original_trace = registered._finish_registered_finalization_trace
    captured: list[registered._CompletionAdmission] = []

    def begin(
        executor: registered.RegisteredTrainingExecutor,
        barrier: RegisteredHistoryBarrier,
        caller_token: object | None = None,
    ) -> registered._CompletionAdmission:
        admission = original_begin(executor, barrier, caller_token)
        captured.append(admission)
        return admission

    def tamper(
        reservation: registered._FinalizationReservation,
    ) -> CompleteUpdateTraceEvidence:
        assert len(captured) == 1
        object.__setattr__(captured[0], "token", object())
        return original_trace(reservation)

    monkeypatch.setattr(registered, "_begin_completion_admission", begin)
    monkeypatch.setattr(
        registered,
        "_finish_registered_finalization_trace",
        tamper,
    )
    with pytest.raises(
        registered.Experiment002RegisteredExecutorError,
        match="admission truth changed",
    ):
        registered.complete_registered_training_executor(
            final.executor,
            final.barrier,
        )

    assert final.state.phase == "FAILED"
    assert registered._HANDOFFS[final.handoff].failed
    assert final.calls["abort"] == 1
    assert captured[0].completed.is_set()
    assert len(captured[0].outcome) == 1
    assert captured[0].outcome[0].result is None
    assert captured[0].outcome[0].error is not None
    monkeypatch.setattr(registered, "_begin_completion_admission", original_begin)
    with pytest.raises(registered.Experiment002RegisteredExecutorError):
        registered.complete_registered_training_executor(
            final.executor,
            final.barrier,
        )


def test_advance_admission_tamper_cannot_split_leader_and_follower_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, state, handoff, barrier, issued, consumed, _ = _claimed_nonfinal_boundary(
        monkeypatch
    )
    original_begin = registered._begin_advance_admission
    captured: list[registered._AdvanceAdmission] = []
    abort_states: list[tuple[str, bool, str | None]] = []

    def begin(
        observed_executor: registered.RegisteredTrainingExecutor,
        observed_barrier: RegisteredHistoryBarrier,
        caller_token: object | None = None,
    ) -> registered._AdvanceAdmission:
        admission = original_begin(
            observed_executor,
            observed_barrier,
            caller_token,
        )
        captured.append(admission)
        return admission

    def consume(
        reservation: registered._ContinuationReservation,
    ) -> registered_history._RegisteredHistoryBarrierSnapshot:
        del reservation
        assert len(captured) == 1
        object.__setattr__(captured[0], "token", object())
        return consumed

    def abort(
        registration: VerifiedRunRegistration,
        observed_executor: registered.RegisteredTrainingExecutor,
        observed_barrier: RegisteredHistoryBarrier,
    ) -> None:
        del registration
        assert observed_executor is executor and observed_barrier is barrier
        reservation = next(
            item
            for item in registered._CONTINUATION_HISTORY
            if item.executor is executor
        )
        abort_states.append(
            (
                state.phase,
                registered._HANDOFFS[handoff].failed,
                registered._CONTINUATION_PHASES.get(reservation),
            )
        )

    monkeypatch.setattr(registered, "_begin_advance_admission", begin)
    monkeypatch.setattr(
        registered,
        "_preflight_registered_history_barrier",
        lambda observed_executor, observed_barrier: issued,
    )
    monkeypatch.setattr(
        registered,
        "_consume_registered_continuation_history",
        consume,
    )
    monkeypatch.setattr(registered, "_abort_consumed_registered_history", abort)
    with pytest.raises(
        registered.Experiment002RegisteredExecutorError,
        match="admission truth changed",
    ):
        registered.advance_registered_training_executor(executor, barrier)

    assert state.phase == "FAILED"
    assert registered._HANDOFFS[handoff].failed
    assert abort_states == [("FAILED", True, "FAILED")]
    assert captured[0].completed.is_set()
    assert len(captured[0].outcome) == 1
    assert isinstance(
        captured[0].outcome[0],
        registered.Experiment002RegisteredExecutorError,
    )
    monkeypatch.setattr(registered, "_begin_advance_admission", original_begin)
    with pytest.raises(registered.Experiment002RegisteredExecutorError):
        registered.advance_registered_training_executor(executor, barrier)


@pytest.mark.parametrize("cleanup", ["terminal", "history"])
def test_completion_cleanup_retries_default_captured_terminal_and_history_routes(
    monkeypatch: pytest.MonkeyPatch,
    cleanup: str,
) -> None:
    final = _claimed_final_boundary(monkeypatch)
    monkeypatch.setattr(
        registered,
        "_finish_registered_finalization_trace",
        lambda reservation: (_ for _ in ()).throw(RuntimeError("trace fail")),
    )
    if cleanup == "terminal":
        monkeypatch.setattr(
            registered,
            "_terminal_fail_registered_finalization",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                KeyboardInterrupt("terminal entry")
            ),
        )
    else:

        def lower_abort(
            registration: VerifiedRunRegistration,
            executor: registered.RegisteredTrainingExecutor,
            barrier: RegisteredHistoryBarrier,
            completed: RegisteredCompletedTrainingHistory | None,
        ) -> None:
            del registration, completed
            assert executor is final.executor and barrier is final.barrier
            assert final.state.phase == "FAILED"
            assert registered._HANDOFFS[final.handoff].failed
            reservation = final.state.finalization_reservation
            assert reservation is not None
            assert registered._FINALIZATION_PHASES[reservation] == "FAILED"
            final.calls["abort"] += 1

        monkeypatch.setattr(
            registered_history,
            "_abort_attempted_registered_history_completion",
            lower_abort,
        )
        monkeypatch.setattr(
            registered,
            "_abort_attempted_registered_history_completion",
            lambda *args: (_ for _ in ()).throw(
                KeyboardInterrupt("history abort entry")
            ),
        )

    with pytest.raises(RuntimeError, match="trace fail"):
        registered.complete_registered_training_executor(
            final.executor,
            final.barrier,
        )
    reservation = final.state.finalization_reservation
    assert reservation is not None
    assert final.state.phase == "FAILED"
    assert registered._HANDOFFS[final.handoff].failed
    assert registered._FINALIZATION_PHASES[reservation] == "FAILED"
    assert final.calls["abort"] == 1


def test_advance_begin_interruption_recovers_and_wakes_exact_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, state, _, barrier, _, _, _ = _claimed_nonfinal_boundary(monkeypatch)
    original_begin = registered._begin_advance_admission
    captured: list[registered._AdvanceAdmission] = []

    def interrupt(
        observed_executor: registered.RegisteredTrainingExecutor,
        observed_barrier: RegisteredHistoryBarrier,
        caller_token: object | None = None,
    ) -> registered._AdvanceAdmission:
        admission = original_begin(
            observed_executor,
            observed_barrier,
            caller_token,
        )
        captured.append(admission)
        raise KeyboardInterrupt("after advance begin")

    monkeypatch.setattr(registered, "_begin_advance_admission", interrupt)
    with pytest.raises(KeyboardInterrupt, match="advance begin"):
        registered.advance_registered_training_executor(executor, barrier)
    assert len(captured) == 1
    assert captured[0].completed.is_set()
    assert len(captured[0].outcome) == 1
    assert state.phase == "FAILED"

    monkeypatch.setattr(registered, "_begin_advance_admission", original_begin)
    with pytest.raises(registered.Experiment002RegisteredExecutorError):
        registered.advance_registered_training_executor(executor, barrier)


def test_epoch_29_advance_finish_interruption_cannot_block_real_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final = _claimed_final_boundary(monkeypatch)
    monkeypatch.setattr(
        registered_history, "verify_registered_history_barrier", lambda value: None
    )
    monkeypatch.setattr(
        registered_history,
        "_registered_history_barrier_snapshot",
        lambda value: final.issued,
    )
    original_finish = registered._finish_advance_admission
    monkeypatch.setattr(
        registered,
        "_finish_advance_admission",
        lambda admission, outcome: (_ for _ in ()).throw(
            KeyboardInterrupt("advance finish entry")
        ),
    )
    with pytest.raises(KeyboardInterrupt, match="advance finish"):
        registered.advance_registered_training_executor(
            final.executor,
            final.barrier,
        )
    assert final.state.phase == "AWAITING_EVALUATION"
    assert final.executor not in registered._FAILED_EXECUTOR_HISTORY
    monkeypatch.setattr(registered, "_finish_advance_admission", original_finish)
    assert (
        registered.complete_registered_training_executor(
            final.executor,
            final.barrier,
        )
        is final.completed_history
    )


def test_advance_follower_begin_interruption_cannot_poison_its_leader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final = _claimed_final_boundary(monkeypatch)
    original_begin = registered._begin_advance_admission
    entered = threading.Event()
    release = threading.Event()

    def begin(
        executor: registered.RegisteredTrainingExecutor,
        barrier: RegisteredHistoryBarrier,
        caller_token: object | None = None,
    ) -> registered._AdvanceAdmission:
        admission = original_begin(executor, barrier, caller_token)
        if not admission.leader:
            raise KeyboardInterrupt("advance follower begin")
        return admission

    def preflight(
        executor: registered.RegisteredTrainingExecutor,
        barrier: RegisteredHistoryBarrier,
    ) -> registered_history._RegisteredHistoryBarrierSnapshot:
        del executor, barrier
        entered.set()
        assert release.wait(timeout=2.0)
        raise registered._FinalBarrierRequiresCompletion("final route")

    monkeypatch.setattr(registered, "_begin_advance_admission", begin)
    monkeypatch.setattr(registered, "_preflight_registered_history_barrier", preflight)
    leader_outcomes: list[BaseException | None] = []

    def lead() -> None:
        try:
            registered.advance_registered_training_executor(
                final.executor,
                final.barrier,
            )
        except BaseException as error:
            leader_outcomes.append(error)
        else:
            leader_outcomes.append(None)

    leader = threading.Thread(target=lead)
    leader.start()
    assert entered.wait(timeout=2.0)
    with pytest.raises(KeyboardInterrupt, match="advance follower"):
        registered.advance_registered_training_executor(
            final.executor,
            final.barrier,
        )
    assert final.state.phase == "AWAITING_EVALUATION"
    assert final.executor not in registered._FAILED_EXECUTOR_HISTORY
    release.set()
    leader.join(timeout=3.0)
    assert not leader.is_alive()
    assert len(leader_outcomes) == 1
    assert isinstance(
        leader_outcomes[0],
        registered._FinalBarrierRequiresCompletion,
    )
    assert (
        registered.complete_registered_training_executor(
            final.executor,
            final.barrier,
        )
        is final.completed_history
    )


def test_retained_completion_outcome_rewrite_is_terminal_and_irreversible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final = _claimed_final_boundary(monkeypatch)
    assert (
        registered.complete_registered_training_executor(
            final.executor,
            final.barrier,
        )
        is final.completed_history
    )
    retained = registered._begin_completion_admission(
        final.executor,
        final.barrier,
    )
    assert not retained.leader and len(retained.outcome) == 1
    original = retained.outcome[0]
    object.__setattr__(original, "result", None)
    object.__setattr__(original, "error", RuntimeError("rewritten outcome"))

    with pytest.raises(
        registered.Experiment002RegisteredExecutorError,
        match="outcome authority changed",
    ):
        registered.complete_registered_training_executor(
            final.executor,
            final.barrier,
        )
    assert final.state.phase == "FAILED"
    assert registered._HANDOFFS[final.handoff].failed
    assert final.calls["abort"] == 1

    object.__setattr__(original, "result", final.completed_history)
    object.__setattr__(original, "error", None)
    retained.outcome[:] = [original]
    with pytest.raises(registered.Experiment002RegisteredExecutorError):
        registered.complete_registered_training_executor(
            final.executor,
            final.barrier,
        )
    assert final.state.phase == "FAILED"
    assert registered._HANDOFFS[final.handoff].failed


def test_retained_advance_outcome_rewrite_is_terminal_and_irreversible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, state, _, barrier = _completed_nonfinal_advance(monkeypatch)
    retained = registered._begin_advance_admission(executor, barrier)
    assert not retained.leader and retained.outcome == [None]
    retained.outcome[:] = [RuntimeError("rewritten advance outcome")]

    with pytest.raises(
        registered.Experiment002RegisteredExecutorError,
        match="outcome authority changed",
    ):
        registered.advance_registered_training_executor(executor, barrier)
    assert state.phase == "FAILED"

    retained.outcome[:] = [None]
    with pytest.raises(registered.Experiment002RegisteredExecutorError):
        registered.advance_registered_training_executor(executor, barrier)
    assert state.phase == "FAILED"


def test_retained_completion_event_clear_is_terminal_and_cannot_be_revived(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final = _claimed_final_boundary(monkeypatch)
    assert (
        registered.complete_registered_training_executor(
            final.executor,
            final.barrier,
        )
        is final.completed_history
    )
    retained = registered._begin_completion_admission(
        final.executor,
        final.barrier,
    )
    assert not retained.leader and retained.completed.is_set()
    retained.completed.clear()
    observed: list[BaseException | RegisteredCompletedTrainingHistory] = []

    def retry() -> None:
        try:
            observed.append(
                registered.complete_registered_training_executor(
                    final.executor,
                    final.barrier,
                )
            )
        except BaseException as error:
            observed.append(error)

    follower = threading.Thread(target=retry, daemon=True)
    follower.start()
    follower.join(timeout=3.0)
    assert not follower.is_alive()
    assert len(observed) == 1
    assert isinstance(observed[0], registered._BoundaryTruthCorruption)
    assert "event authority changed" in str(observed[0])
    assert retained.completed.is_set()
    assert final.state.phase == "FAILED"
    assert registered._HANDOFFS[final.handoff].failed
    assert final.calls["abort"] == 1

    retained.completed.set()
    with pytest.raises(registered.Experiment002RegisteredExecutorError):
        registered.complete_registered_training_executor(
            final.executor,
            final.barrier,
        )
    assert final.state.phase == "FAILED"
    assert registered._HANDOFFS[final.handoff].failed


def test_retained_advance_event_clear_is_terminal_and_cannot_be_revived(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, state, _, barrier = _completed_nonfinal_advance(monkeypatch)
    retained = registered._begin_advance_admission(executor, barrier)
    assert not retained.leader and retained.completed.is_set()
    retained.completed.clear()
    observed: list[BaseException | None] = []

    def retry() -> None:
        try:
            registered.advance_registered_training_executor(executor, barrier)
        except BaseException as error:
            observed.append(error)
        else:
            observed.append(None)

    follower = threading.Thread(target=retry, daemon=True)
    follower.start()
    follower.join(timeout=3.0)
    assert not follower.is_alive()
    assert len(observed) == 1
    assert isinstance(observed[0], registered._BoundaryTruthCorruption)
    assert "event authority changed" in str(observed[0])
    assert retained.completed.is_set()
    assert state.phase == "FAILED"

    retained.completed.set()
    with pytest.raises(registered.Experiment002RegisteredExecutorError):
        registered.advance_registered_training_executor(executor, barrier)
    assert state.phase == "FAILED"


def test_completion_partial_publication_is_recovered_without_stale_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final = _claimed_final_boundary(monkeypatch)
    original_finish = registered._finish_completion_admission
    captured: list[registered._CompletionAdmission] = []

    def interrupt(
        admission: registered._CompletionAdmission,
        result: RegisteredCompletedTrainingHistory | None,
        error: BaseException | None,
    ) -> None:
        captured.append(admission)
        admission.outcome[:] = [registered._CompletionOutcome(result, error)]
        raise KeyboardInterrupt("after completion outcome write")

    monkeypatch.setattr(registered, "_finish_completion_admission", interrupt)
    with pytest.raises(KeyboardInterrupt, match="completion outcome write"):
        registered.complete_registered_training_executor(
            final.executor,
            final.barrier,
        )
    assert len(captured) == 1
    retained = registered._recover_retained_completion_admission(
        final.executor,
        final.barrier,
    )
    assert retained is not None
    assert retained.completed is captured[0].completed
    assert retained.completed.is_set()
    assert len(retained.outcome) == 1
    assert retained.outcome[0].result is final.completed_history
    assert retained.outcome[0].error is None

    monkeypatch.setattr(registered, "_finish_completion_admission", original_finish)
    assert (
        registered.complete_registered_training_executor(
            final.executor,
            final.barrier,
        )
        is final.completed_history
    )
    assert final.state.phase == "COMPLETE"
    assert final.calls == {"trace": 1, "history": 1, "abort": 0}


def test_advance_partial_publication_is_recovered_without_stale_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, state, _, barrier, issued, consumed, _ = _claimed_nonfinal_boundary(
        monkeypatch
    )
    monkeypatch.setattr(
        registered,
        "_preflight_registered_history_barrier",
        lambda observed_executor, observed_barrier: issued,
    )
    monkeypatch.setattr(
        registered,
        "_consume_registered_continuation_history",
        lambda reservation: consumed,
    )
    original_finish = registered._finish_advance_admission
    captured: list[registered._AdvanceAdmission] = []

    def interrupt(
        admission: registered._AdvanceAdmission,
        outcome: BaseException | None,
    ) -> None:
        captured.append(admission)
        admission.outcome[:] = [outcome]
        raise KeyboardInterrupt("after advance outcome write")

    monkeypatch.setattr(registered, "_finish_advance_admission", interrupt)
    with pytest.raises(KeyboardInterrupt, match="advance outcome write"):
        registered.advance_registered_training_executor(executor, barrier)
    assert len(captured) == 1
    retained = registered._recover_retained_advance_admission(executor, barrier)
    assert retained is not None
    assert retained.completed is captured[0].completed
    assert retained.completed.is_set()
    assert retained.outcome == [None]

    monkeypatch.setattr(registered, "_finish_advance_admission", original_finish)
    registered.advance_registered_training_executor(executor, barrier)
    assert state.phase == "READY_TO_TRAIN"

    different_barrier = object.__new__(RegisteredHistoryBarrier)
    probe = registered._begin_advance_admission(executor, different_barrier)
    assert probe.leader
    registered._finish_advance_admission(probe, RuntimeError("probe outcome"))
    assert probe.completed.is_set()


def test_completion_follower_begin_interruption_cannot_poison_its_leader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final = _claimed_final_boundary(monkeypatch)
    original_begin = registered._begin_completion_admission
    original_trace = registered._finish_registered_finalization_trace
    entered = threading.Event()
    release = threading.Event()

    def begin(
        executor: registered.RegisteredTrainingExecutor,
        barrier: RegisteredHistoryBarrier,
        caller_token: object | None = None,
    ) -> registered._CompletionAdmission:
        admission = original_begin(executor, barrier, caller_token)
        if not admission.leader:
            raise KeyboardInterrupt("completion follower begin")
        return admission

    def finish_trace(
        reservation: registered._FinalizationReservation,
    ) -> CompleteUpdateTraceEvidence:
        entered.set()
        assert release.wait(timeout=2.0)
        return original_trace(reservation)

    monkeypatch.setattr(registered, "_begin_completion_admission", begin)
    monkeypatch.setattr(
        registered,
        "_finish_registered_finalization_trace",
        finish_trace,
    )
    leader_outcomes: list[object] = []

    def lead() -> None:
        try:
            leader_outcomes.append(
                registered.complete_registered_training_executor(
                    final.executor,
                    final.barrier,
                )
            )
        except BaseException as error:
            leader_outcomes.append(error)

    leader = threading.Thread(target=lead)
    leader.start()
    assert entered.wait(timeout=2.0)
    with pytest.raises(KeyboardInterrupt, match="completion follower"):
        registered.complete_registered_training_executor(
            final.executor,
            final.barrier,
        )
    assert final.state.phase == "AWAITING_EVALUATION"
    assert final.executor not in registered._FAILED_EXECUTOR_HISTORY
    release.set()
    leader.join(timeout=3.0)
    assert not leader.is_alive()
    assert leader_outcomes == [final.completed_history]
    assert registered._registered_executor_snapshot(final.executor).phase == "COMPLETE"
