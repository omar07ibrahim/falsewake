from __future__ import annotations

import copy
import inspect
import os
import pickle
import threading
import warnings
from collections.abc import Callable
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

import falsewake.experiment_002_registered_executor as registered
import falsewake.experiment_002_training_bridge as bridge
import falsewake.experiment_002_training_executor as numeric
from falsewake.causal_kws import CausalKWS
from falsewake.experiment_002_data import TRAINING_SEEDS
from falsewake.experiment_002_evidence import RegisteredValidationInputs
from falsewake.experiment_002_run_authority import VerifiedRunRegistration
from falsewake.experiment_002_training_bridge import (
    RegisteredExecutorSessionAuthority,
    RegisteredOptimizerTransition,
    TraceConsumedTransition,
)
from falsewake.experiment_002_training_evidence import (
    EpochUpdateTraceEvidence,
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


class _CounterfeitTrace:
    def __init__(self) -> None:
        self.epoch_trace = object.__new__(EpochUpdateTraceEvidence)

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
