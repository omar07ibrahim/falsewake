from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
import inspect
import json
import os
import pickle
import threading
import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import falsewake.experiment_002_registered_history as history
import falsewake.experiment_002_training_executor as numeric_executor
from falsewake.experiment_002_evidence import (
    RegisteredValidationEvidence,
    RegisteredValidationInputs,
)
from falsewake.experiment_002_registered_evaluator import (
    RegisteredEvaluatedEpoch,
    _RegisteredEvaluatedEpochSnapshot,
)
from falsewake.experiment_002_registered_executor import (
    RegisteredExecutorEpochHandoff,
    RegisteredTrainingExecutor,
    _RegisteredExecutorEpochHandoffSnapshot,
)
from falsewake.experiment_002_run_authority import VerifiedRunRegistration
from falsewake.experiment_002_training_bridge import (
    RegisteredExecutorSessionAuthority,
)
from falsewake.experiment_002_training_evidence import (
    EPOCH_COUNT,
    TOTAL_UPDATE_COUNT,
    TRAINING_EXAMPLE_COUNT,
    UPDATES_PER_EPOCH,
    CompleteUpdateTraceEvidence,
    EpochUpdateTraceEvidence,
    RegisteredModelTensorEvidence,
)
from falsewake.experiment_002_training_history import (
    _HISTORY_DOMAIN,
    _epoch_document,
    _EpochRecord,
    _macro_f1_from_confusion,
)
from falsewake.experiment_002_training_population import CompletedTrainingPopulation

type ConfusionMatrix = tuple[tuple[int, ...], ...]

_SEED = 20_260_719
_VALIDATION_DIGEST = "a" * 64
_COMPLETE_TRACE_DIGEST = "b" * 64
_SUPPORT = (397, 406, 350, 377, 352, 363, 363, 373, 350, 372, 6_278, 602)


class _IntSubclass(int):
    pass


class _StrSubclass(str):
    pass


def _perfect_confusion() -> ConfusionMatrix:
    return tuple(
        tuple(_SUPPORT[row] if row == column else 0 for column in range(12))
        for row in range(12)
    )


def _sparse_confusion() -> ConfusionMatrix:
    return tuple(
        tuple(10_583 if row == column == 0 else 0 for column in range(12))
        for row in range(12)
    )


def _record(
    epoch: int,
    *,
    validation_cross_entropy: np.float64 | None = None,
    confusion: ConfusionMatrix | None = None,
) -> _EpochRecord:
    matrix = _perfect_confusion() if confusion is None else confusion
    return _EpochRecord(
        zero_based_epoch=epoch,
        first_global_update=epoch * 313,
        last_global_update_inclusive=(epoch + 1) * 313 - 1,
        training_population_digest="c" * 64,
        training_cross_entropy=np.float64(epoch + 0.5),
        epoch_update_trace_digest="d" * 64,
        validation_input_digest=_VALIDATION_DIGEST,
        validation_confusion_matrix=matrix,
        validation_cross_entropy=(
            np.float64(epoch + 0.25)
            if validation_cross_entropy is None
            else validation_cross_entropy
        ),
        macro_f1=_macro_f1_from_confusion(matrix),
        validation_prediction_digest="e" * 64,
        model_tensor_digest="f" * 64,
    )


def _records() -> tuple[_EpochRecord, ...]:
    return tuple(_record(epoch) for epoch in range(30))


def _assert_no_json_floats(value: object) -> None:
    if type(value) is dict:
        for key, item in value.items():
            assert type(key) is str
            _assert_no_json_floats(item)
        return
    if type(value) is list:
        for item in value:
            _assert_no_json_floats(item)
        return
    assert type(value) in (str, int)


def _counterfeit_history() -> tuple[
    history.RegisteredTrainingHistory,
    history._HistoryState,
]:
    result = history._issue_registered_history(
        registration=object.__new__(VerifiedRunRegistration),
        process_id=os.getpid(),
        executor=object.__new__(RegisteredTrainingExecutor),
        validation_inputs=object.__new__(RegisteredValidationInputs),
        seed=_SEED,
        validation_inputs_sha256=_VALIDATION_DIGEST,
    )
    return result, history._issued_history_state(result)


def _counterfeit_bound(
    state: history._HistoryState,
    *,
    epoch: int = 0,
) -> history._BoundEpochRecord:
    handoff = object.__new__(RegisteredExecutorEpochHandoff)
    population = object.__new__(CompletedTrainingPopulation)
    epoch_trace = object.__new__(EpochUpdateTraceEvidence)
    evaluated = object.__new__(RegisteredEvaluatedEpoch)
    validation_evidence = object.__new__(RegisteredValidationEvidence)
    model_tensors = object.__new__(RegisteredModelTensorEvidence)
    runtime = numeric_executor._RuntimeDigests(
        model_sha256="1" * 64,
        optimizer_sha256="2" * 64,
        rng_sha256="3" * 64,
        rng_state=b"rng",
    )
    handoff_snapshot = _RegisteredExecutorEpochHandoffSnapshot(
        executor=state.executor,
        registration=state.registration,
        validation_inputs=state.validation_inputs,
        process_id=state.process_id,
        bridge_authority=object.__new__(RegisteredExecutorSessionAuthority),
        bridge_session_token=object(),
        population=population,
        epoch_trace=epoch_trace,
        seed=state.seed,
        zero_based_epoch=epoch,
        optimizer_generation=(epoch + 1) * 313,
        runtime_digests=runtime,
        previous_history_barrier=None,
        one_shot_token=object(),
        another_training_epoch=epoch < 29,
    )
    evaluated_snapshot = _RegisteredEvaluatedEpochSnapshot(
        registration=state.registration,
        handoff=handoff,
        validation_inputs=state.validation_inputs,
        validation_evidence=validation_evidence,
        model_tensors=model_tensors,
        process_id=state.process_id,
        seed=state.seed,
        zero_based_epoch=epoch,
        optimizer_generation=(epoch + 1) * 313,
        batch_count=83,
        another_training_epoch=epoch < 29,
        registration_head_commit="4" * 40,
        registration_sha256="5" * 64,
        source_bundle_sha256="6" * 64,
        validation_inputs_sha256=_VALIDATION_DIGEST,
        validation_predictions_sha256="7" * 64,
        model_tensor_sha256="1" * 64,
        optimizer_sha256="2" * 64,
        torch_rng_sha256="3" * 64,
        authority_sha256="8" * 64,
    )
    provisional = history._BoundEpochRecord(
        record=_record(epoch),
        evaluated_epoch=evaluated,
        evaluated_snapshot=evaluated_snapshot,
        handoff_snapshot=handoff_snapshot,
        population=population,
        epoch_trace=epoch_trace,
        model_tensors=model_tensors,
        authority_fingerprint="",
    )
    return dataclasses.replace(
        provisional,
        authority_fingerprint=history._bound_record_fingerprint(provisional),
    )


def _counterfeit_barrier() -> tuple[
    history.RegisteredTrainingHistory,
    history.RegisteredHistoryBarrier,
    history._BarrierState,
]:
    owner, owner_state = _counterfeit_history()
    barrier = history._issue_registered_barrier(
        owner,
        owner_state,
        _counterfeit_bound(owner_state),
    )
    return owner, barrier, history._issued_barrier_state(barrier)


def _counterfeit_completed() -> tuple[
    history.RegisteredCompletedTrainingHistory,
    history._CompletedState,
]:
    owner, owner_state = _counterfeit_history()
    records = tuple(_counterfeit_bound(owner_state, epoch=epoch) for epoch in range(30))
    epoch_records = tuple(bound.record for bound in records)
    canonical = history._canonical_registered_history_bytes(
        seed=owner_state.seed,
        validation_inputs_sha256=owner_state.validation_inputs_sha256,
        complete_update_trace_sha256=_COMPLETE_TRACE_DIGEST,
        records=epoch_records,
    )
    ranks = history._rank_registered_records(epoch_records)
    # Privileged white-box construction for structural tests only. Reflection
    # into defaults/closures is outside the production capability threat model.
    defaults = history._complete_registered_training_history.__defaults__
    assert defaults is not None and len(defaults) == 1
    transition = cast(Any, defaults[0])
    accepted: list[history._BoundEpochRecord] = []
    for bound in records:
        transition(
            owner,
            owner_state,
            history._HISTORY_GUARDS[owner],
            expected_phase="OPEN",
            phase="OPEN",
            records=(*accepted, bound),
            active_barrier=None,
            active_barrier_token=None,
            completed_history=None,
        )
        accepted.append(bound)
        owner_state.records = list(accepted)
        history._HISTORY_LIFECYCLES[owner] = history._lifecycle_from_state(owner_state)
    result = history._issue_completed_history(
        history=owner,
        state=owner_state,
        complete_update_trace=object.__new__(CompleteUpdateTraceEvidence),
        records=records,
        canonical_json_bytes=canonical,
        ranked_epochs=ranks,
        winner=records[ranks[0].zero_based_epoch].evaluated_epoch,
    )
    transition(
        owner,
        owner_state,
        history._HISTORY_GUARDS[owner],
        expected_phase="OPEN",
        phase="COMPLETE",
        records=records,
        active_barrier=None,
        active_barrier_token=None,
        completed_history=result,
    )
    owner_state.phase = "COMPLETE"
    owner_state.active_barrier = None
    owner_state.active_barrier_token = None
    owner_state.completed = result
    history._HISTORY_LIFECYCLES[owner] = history._lifecycle_from_state(owner_state)
    return result, history._issued_completed_state(result)


def _counterfeit_accepted_history() -> tuple[
    history.RegisteredTrainingHistory,
    history._HistoryState,
    history._BoundEpochRecord,
]:
    owner, state = _counterfeit_history()
    bound = _counterfeit_bound(state)
    barrier = history._issue_registered_barrier(owner, state, bound)
    barrier_state = history._issued_barrier_state(barrier)
    # Privileged white-box construction for structural tests only.
    defaults = history._consume_registered_evaluated_epoch.__defaults__
    assert defaults is not None and len(defaults) == 3
    transition = cast(Any, defaults[2])
    transition(
        owner,
        state,
        history._HISTORY_GUARDS[owner],
        expected_phase="OPEN",
        phase="OPEN",
        records=(bound,),
        active_barrier=barrier,
        active_barrier_token=barrier_state.token,
        completed_history=None,
    )
    state.records = [bound]
    state.active_barrier = barrier
    state.active_barrier_token = barrier_state.token
    history._HISTORY_LIFECYCLES[owner] = history._lifecycle_from_state(state)
    assert history._issued_history_state(owner) is state
    return owner, state, bound


def test_source_is_history_only_and_has_no_filesystem_publication_route() -> None:
    path = Path("src/falsewake/experiment_002_registered_history.py")
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported_roots.update(
                alias.name.split(".", maxsplit=1)[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".", maxsplit=1)[0])

    assert not imported_roots & {
        "pathlib",
        "safetensors",
        "subprocess",
        "torch",
    }
    assert "open(" not in source
    assert "Path(" not in source
    assert "experiment-002-run.json" not in source
    assert "EPOCH_COUNT" in source
    assert "CompleteUpdateTraceEvidence" in source
    assert "RegisteredEvaluatedEpoch" in source
    assert "RegisteredHistoryBarrier" in source


def test_source_states_the_trusted_process_reflection_boundary() -> None:
    source = Path("src/falsewake/experiment_002_registered_history.py").read_text(
        encoding="utf-8"
    )
    normalized = " ".join(source.split())
    assert "trusted process" in normalized
    assert "same-process reflection" in normalized
    assert "outside this boundary" in normalized
    assert "defense in depth, not secrecy" in normalized


def test_public_api_has_no_callback_or_generic_barrier_seam() -> None:
    consume = inspect.signature(history.consume_registered_evaluated_epoch)
    complete = inspect.signature(history.complete_registered_training_history)
    barrier = inspect.signature(history._consume_registered_history_barrier)
    abort = inspect.signature(history._abort_consumed_registered_history_barrier)
    assert tuple(consume.parameters) == ("history", "evaluated_epoch")
    assert tuple(complete.parameters) == (
        "history",
        "final_barrier",
        "complete_update_trace",
    )
    assert tuple(barrier.parameters) == ("registration", "executor", "barrier")
    assert tuple(abort.parameters) == ("registration", "executor", "barrier")
    assert not hasattr(history, "_build_independent_authority_truth")


def test_exact_consumed_barrier_abort_is_local_and_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, state, _ = _counterfeit_accepted_history()
    barrier = state.active_barrier
    assert type(barrier) is history.RegisteredHistoryBarrier
    barrier_state = history._issued_barrier_state(barrier)
    history._transition_barrier(barrier, barrier_state, phase="CONSUMED")

    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("abort crossed into executor/evaluator verification")

    monkeypatch.setattr(history, "_registered_executor_snapshot", forbidden)
    monkeypatch.setattr(history, "_registered_evaluated_epoch_snapshot", forbidden)
    monkeypatch.setattr(history, "verify_registered_history_barrier", forbidden)
    history._abort_consumed_registered_history_barrier(
        state.registration,
        state.executor,
        barrier,
    )
    assert state.phase == "FAILED"
    assert barrier_state.phase == "FAILED"
    assert owner in history._FAILED_HISTORIES
    assert barrier in history._FAILED_BARRIERS
    history._abort_consumed_registered_history_barrier(
        state.registration,
        state.executor,
        barrier,
    )


def test_consumed_barrier_abort_wrong_executor_fails_closed() -> None:
    _, state, _ = _counterfeit_accepted_history()
    barrier = state.active_barrier
    assert type(barrier) is history.RegisteredHistoryBarrier
    barrier_state = history._issued_barrier_state(barrier)
    history._transition_barrier(barrier, barrier_state, phase="CONSUMED")
    foreign = object.__new__(RegisteredTrainingExecutor)
    with pytest.raises(
        history.Experiment002RegisteredHistoryError,
        match="abortable",
    ):
        history._abort_consumed_registered_history_barrier(
            state.registration,
            foreign,
            barrier,
        )
    assert state.phase == "FAILED"
    assert barrier_state.phase == "FAILED"


@pytest.mark.parametrize("pre_failed", [False, True])
def test_attempted_barrier_abort_covers_issued_and_partial_failed_states(
    pre_failed: bool,
) -> None:
    owner, state, _ = _counterfeit_accepted_history()
    barrier = state.active_barrier
    assert type(barrier) is history.RegisteredHistoryBarrier
    barrier_state = history._issued_barrier_state(barrier)
    if pre_failed:
        history._terminal_fail_barrier(barrier, barrier_state)
        assert state.phase == "OPEN"
        assert barrier_state.phase == "FAILED"
    history._abort_consumed_registered_history_barrier(
        state.registration,
        state.executor,
        barrier,
    )
    assert state.phase == "FAILED"
    assert barrier_state.phase == "FAILED"
    assert owner in history._FAILED_HISTORIES
    history._abort_consumed_registered_history_barrier(
        state.registration,
        state.executor,
        barrier,
    )


def test_canonical_thirty_epoch_history_matches_frozen_json_contract() -> None:
    payload = history._canonical_registered_history_bytes(
        seed=_SEED,
        validation_inputs_sha256=_VALIDATION_DIGEST,
        complete_update_trace_sha256=_COMPLETE_TRACE_DIGEST,
        records=_records(),
    )
    parsed = json.loads(payload)
    assert payload.endswith(b"\n")
    assert not payload.endswith(b"\n\n")
    assert b" " not in payload
    assert parsed["schema_version"] == 1
    assert parsed["experiment"] == "002"
    assert parsed["seed"] == _SEED
    assert parsed["validation_input_digest"] == _VALIDATION_DIGEST
    assert parsed["complete_update_trace_digest"] == _COMPLETE_TRACE_DIGEST
    assert len(parsed["epochs"]) == 30
    assert parsed["epochs"][0]["first_global_update"] == 0
    assert parsed["epochs"][29]["last_global_update_inclusive"] == 9_389
    assert parsed["epochs"][0]["training_cross_entropy_float64_hex"] == (
        "0x1.0000000000000p-1"
    )
    assert parsed["epochs"][0]["validation_cross_entropy_float64_hex"] == (
        "0x1.0000000000000p-2"
    )
    _assert_no_json_floats(parsed)
    digest = hashlib.sha256(_HISTORY_DOMAIN + payload).hexdigest()
    assert len(digest) == 64


def test_canonical_contract_is_identical_to_frozen_epoch_document() -> None:
    records = _records()
    payload = history._canonical_registered_history_bytes(
        seed=_SEED,
        validation_inputs_sha256=_VALIDATION_DIGEST,
        complete_update_trace_sha256=_COMPLETE_TRACE_DIGEST,
        records=records,
    )
    expected = {
        "schema_version": 1,
        "experiment": "002",
        "seed": _SEED,
        "validation_input_digest": _VALIDATION_DIGEST,
        "complete_update_trace_digest": _COMPLETE_TRACE_DIGEST,
        "epochs": [_epoch_document(record) for record in records],
    }
    expected_payload = (
        json.dumps(
            expected,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    assert payload == expected_payload


def test_ranking_is_exact_macro_f1_then_binary64_ce_then_epoch() -> None:
    smallest = np.nextafter(np.float64(0.0), np.float64(1.0))
    above_quarter = np.nextafter(np.float64(0.25), np.float64(np.inf))
    records = list(_records())
    records[0] = _record(
        0,
        validation_cross_entropy=np.float64(0.0),
        confusion=_sparse_confusion(),
    )
    records[1] = _record(1, validation_cross_entropy=above_quarter)
    records[2] = _record(2, validation_cross_entropy=smallest)
    records[3] = _record(3, validation_cross_entropy=np.float64(0.0))
    records[4] = _record(4, validation_cross_entropy=np.float64(0.0))
    ranked = history._rank_registered_records(tuple(records))
    assert [rank.zero_based_epoch for rank in ranked[:5]] == [3, 4, 2, 1, 5]
    assert ranked[-1].zero_based_epoch == 0
    assert ranked[2].validation_cross_entropy_float64_hex == ("0x0.0000000000001p-1022")
    assert ranked[3].validation_cross_entropy_float64_hex == float(above_quarter).hex()
    assert ranked[0].macro_f1_exact_numerator == 1
    assert ranked[0].macro_f1_exact_denominator == 1


@pytest.mark.parametrize(
    ("mutation", "error_type"),
    [
        ("wrong_count", history.Experiment002RegisteredHistoryError),
        ("reordered", history.Experiment002RegisteredHistoryError),
        ("bool_epoch", history.Experiment002RegisteredHistoryError),
        ("python_float", TypeError),
        ("negative_zero", history.Experiment002RegisteredHistoryError),
        ("uppercase_digest", history.Experiment002RegisteredHistoryError),
        ("wrong_validation", history.Experiment002RegisteredHistoryError),
        ("bad_confusion_total", history.Experiment002RegisteredHistoryError),
    ],
)
def test_canonical_history_rejects_nonexact_records(
    mutation: str,
    error_type: type[BaseException],
) -> None:
    records = list(_records())
    if mutation == "wrong_count":
        records.pop()
    elif mutation == "reordered":
        records[0], records[1] = records[1], records[0]
    elif mutation == "bool_epoch":
        object.__setattr__(records[0], "zero_based_epoch", True)
    elif mutation == "python_float":
        object.__setattr__(records[0], "training_cross_entropy", 0.5)
    elif mutation == "negative_zero":
        object.__setattr__(records[0], "validation_cross_entropy", np.float64(-0.0))
    elif mutation == "uppercase_digest":
        object.__setattr__(records[0], "model_tensor_digest", "F" * 64)
    elif mutation == "wrong_validation":
        object.__setattr__(records[0], "validation_input_digest", "0" * 64)
    else:
        bad = [list(row) for row in _perfect_confusion()]
        bad[0][0] -= 1
        object.__setattr__(
            records[0],
            "validation_confusion_matrix",
            tuple(tuple(row) for row in bad),
        )
    with pytest.raises(error_type):
        history._canonical_registered_history_bytes(
            seed=_SEED,
            validation_inputs_sha256=_VALIDATION_DIGEST,
            complete_update_trace_sha256=_COMPLETE_TRACE_DIGEST,
            records=tuple(records),
        )


def test_capabilities_are_issuer_only_noncopyable_and_nonserializable() -> None:
    for capability_type in (
        history.RegisteredTrainingHistory,
        history.RegisteredHistoryBarrier,
        history.RegisteredCompletedTrainingHistory,
    ):
        with pytest.raises(TypeError, match="issuer-only"):
            capability_type()
        counterfeit = object.__new__(capability_type)
        with pytest.raises(TypeError, match="cannot be copied"):
            copy.copy(counterfeit)
        with pytest.raises(TypeError, match="cannot be copied"):
            copy.deepcopy(counterfeit)
        with pytest.raises(TypeError, match="cannot be serialized"):
            pickle.dumps(counterfeit)


def test_counterfeit_capabilities_are_rejected() -> None:
    with pytest.raises(history.Experiment002RegisteredHistoryError):
        history._issued_history_state(object.__new__(history.RegisteredTrainingHistory))
    with pytest.raises(history.Experiment002RegisteredHistoryError):
        history._issued_barrier_state(object.__new__(history.RegisteredHistoryBarrier))
    with pytest.raises(history.Experiment002RegisteredHistoryError):
        history._issued_completed_state(
            object.__new__(history.RegisteredCompletedTrainingHistory)
        )


def test_raw_closure_issuers_and_factory_are_absent() -> None:
    forbidden = (
        "_build_independent_authority_truth",
        "_record_history_truth",
        "_require_history_truth",
        "_transition_history_truth",
        "_fail_history_truth",
        "_evaluated_truth_owner",
        "_claim_evaluated_truth",
        "_record_barrier_truth",
        "_require_barrier_truth",
        "_transition_barrier_truth",
        "_fail_barrier_truth",
        "_record_completed_truth",
        "_require_completed_truth",
        "_fail_completed_truth",
    )
    assert all(not hasattr(history, name) for name in forbidden)


def test_injected_raw_issuer_alias_does_not_replace_captured_authority() -> None:
    calls = 0

    def injected(*args: object, **kwargs: object) -> None:
        nonlocal calls
        del args, kwargs
        calls += 1

    history.__dict__["_record_history_truth"] = injected
    try:
        issued, state = _counterfeit_history()
        assert history._issued_history_state(issued) is state
        assert calls == 0
    finally:
        history.__dict__.pop("_record_history_truth")


def test_history_clone_cannot_reuse_all_visible_stores() -> None:
    issued, state = _counterfeit_history()
    guard = history._HISTORY_GUARDS[issued]
    lifecycle = history._HISTORY_LIFECYCLES[issued]
    clone = object.__new__(history.RegisteredTrainingHistory)
    object.__setattr__(clone, "_seed", issued._seed)
    object.__setattr__(clone, "_token", issued._token)
    object.__setattr__(clone, "_route_marker", issued._route_marker)
    history._HISTORIES[clone] = state
    history._HISTORY_GUARDS[clone] = guard
    history._HISTORY_LIFECYCLES[clone] = lifecycle
    history._ISSUED_HISTORIES.add(clone)
    with pytest.raises(
        history.Experiment002RegisteredHistoryError,
        match="closure-owned",
    ):
        history._issued_history_state(clone)


def test_coherent_full_history_store_rewrite_cannot_replace_issuance() -> None:
    issued, state = _counterfeit_history()
    original_guard = history._HISTORY_GUARDS[issued]
    rewritten_state = dataclasses.replace(state, records=[])
    rewritten_guard = dataclasses.replace(original_guard)
    history._HISTORIES[issued] = rewritten_state
    history._HISTORY_GUARDS[issued] = rewritten_guard
    history._HISTORY_LIFECYCLES[issued] = history._lifecycle_from_state(rewritten_state)
    with pytest.raises(
        history.Experiment002RegisteredHistoryError,
        match="closure-owned",
    ):
        history._issued_history_state(issued)


def test_coherent_metadata_and_guard_rewrite_cannot_change_seed() -> None:
    issued, state = _counterfeit_history()
    guard = history._HISTORY_GUARDS[issued]
    state.seed = 20_260_720
    object.__setattr__(guard, "seed", 20_260_720)
    object.__setattr__(issued, "_seed", 20_260_720)
    history._HISTORY_LIFECYCLES[issued] = history._lifecycle_from_state(state)
    with pytest.raises(
        history.Experiment002RegisteredHistoryError,
        match="closure-owned",
    ):
        history._issued_history_state(issued)


def test_failed_history_cannot_be_rolled_back_by_rewriting_every_visible_store() -> (
    None
):
    issued, state = _counterfeit_history()
    old_lifecycle = history._HISTORY_LIFECYCLES[issued]
    history._terminal_fail_history(issued, state)
    state.phase = cast(Any, "OPEN")
    history._FAILED_HISTORIES.discard(issued)
    history._HISTORY_LIFECYCLES[issued] = old_lifecycle
    with pytest.raises(
        history.Experiment002RegisteredHistoryError,
        match="closure-owned",
    ):
        history._issued_history_state(issued)


@pytest.mark.parametrize("attack", ["lifecycle", "object_slot", "guard_mirror"])
def test_history_final_authority_failure_is_terminal_after_visible_restore(
    attack: str,
) -> None:
    issued, state = _counterfeit_history()
    guard = history._HISTORY_GUARDS[issued]
    original_lifecycle = history._HISTORY_LIFECYCLES[issued]
    original_seed = issued._seed
    original_guard_process = guard.process_id
    if attack == "lifecycle":
        history._HISTORY_LIFECYCLES[issued] = dataclasses.replace(
            original_lifecycle,
            phase=cast(Any, _StrSubclass("OPEN")),
        )
    elif attack == "object_slot":
        object.__setattr__(issued, "_seed", _IntSubclass(original_seed))
    else:
        object.__setattr__(
            guard,
            "process_id",
            _IntSubclass(original_guard_process),
        )
        history._HISTORY_LIFECYCLES[issued] = dataclasses.replace(
            original_lifecycle,
            process_id=_IntSubclass(original_lifecycle.process_id),
        )

    with pytest.raises((TypeError, history.Experiment002RegisteredHistoryError)):
        history._issued_history_state(issued)

    object.__setattr__(issued, "_seed", original_seed)
    object.__setattr__(guard, "process_id", original_guard_process)
    state.phase = "OPEN"
    history._FAILED_HISTORIES.discard(issued)
    history._HISTORY_LIFECYCLES[issued] = original_lifecycle
    with pytest.raises(
        history.Experiment002RegisteredHistoryError,
        match="closure-owned|failed",
    ):
        history._issued_history_state(issued)


def test_barrier_clone_and_coherent_store_rewrite_are_rejected() -> None:
    _, barrier, state = _counterfeit_barrier()
    guard = history._BARRIER_GUARDS[barrier]
    lifecycle = history._BARRIER_LIFECYCLES[barrier]
    clone = object.__new__(history.RegisteredHistoryBarrier)
    object.__setattr__(clone, "_seed", barrier._seed)
    object.__setattr__(clone, "_accepted_epoch", barrier._accepted_epoch)
    object.__setattr__(clone, "_token", barrier._token)
    object.__setattr__(clone, "_route_marker", barrier._route_marker)
    history._BARRIERS[clone] = state
    history._BARRIER_GUARDS[clone] = guard
    history._BARRIER_LIFECYCLES[clone] = lifecycle
    history._ISSUED_BARRIERS.add(clone)
    with pytest.raises(
        history.Experiment002RegisteredHistoryError,
        match="closure-owned",
    ):
        history._issued_barrier_state(clone)

    _, another, another_state = _counterfeit_barrier()
    another_guard = history._BARRIER_GUARDS[another]
    replacement = dataclasses.replace(another_state)
    history._BARRIERS[another] = replacement
    history._BARRIER_GUARDS[another] = dataclasses.replace(another_guard)
    history._BARRIER_LIFECYCLES[another] = history._BarrierLifecycle(
        token=replacement.token,
        process_id=replacement.process_id,
        phase=replacement.phase,
    )
    with pytest.raises(
        history.Experiment002RegisteredHistoryError,
        match="closure-owned",
    ):
        history._issued_barrier_state(another)


def test_barrier_lifecycle_rollback_is_rejected_by_closure_truth() -> None:
    _, barrier, state = _counterfeit_barrier()
    history._transition_barrier(barrier, state, phase="CONSUMED")
    state.phase = cast(Any, "ISSUED")
    history._BARRIER_LIFECYCLES[barrier] = history._BarrierLifecycle(
        token=state.token,
        process_id=state.process_id,
        phase="ISSUED",
    )
    with pytest.raises(
        history.Experiment002RegisteredHistoryError,
        match="closure-owned",
    ):
        history._issued_barrier_state(barrier)


@pytest.mark.parametrize("attack", ["lifecycle", "object_slot", "guard_mirror"])
def test_barrier_final_authority_failure_is_terminal_after_visible_restore(
    attack: str,
) -> None:
    _, barrier, state = _counterfeit_barrier()
    guard = history._BARRIER_GUARDS[barrier]
    original_lifecycle = history._BARRIER_LIFECYCLES[barrier]
    original_epoch = barrier._accepted_epoch
    original_guard_process = guard.process_id
    if attack == "lifecycle":
        history._BARRIER_LIFECYCLES[barrier] = dataclasses.replace(
            original_lifecycle,
            phase=cast(Any, _StrSubclass("ISSUED")),
        )
    elif attack == "object_slot":
        object.__setattr__(
            barrier,
            "_accepted_epoch",
            _IntSubclass(original_epoch),
        )
    else:
        object.__setattr__(
            guard,
            "process_id",
            _IntSubclass(original_guard_process),
        )
        history._BARRIER_LIFECYCLES[barrier] = dataclasses.replace(
            original_lifecycle,
            process_id=_IntSubclass(original_lifecycle.process_id),
        )

    with pytest.raises((TypeError, history.Experiment002RegisteredHistoryError)):
        history._issued_barrier_state(barrier)

    object.__setattr__(barrier, "_accepted_epoch", original_epoch)
    object.__setattr__(guard, "process_id", original_guard_process)
    state.phase = "ISSUED"
    history._FAILED_BARRIERS.discard(barrier)
    history._BARRIER_LIFECYCLES[barrier] = original_lifecycle
    with pytest.raises(
        history.Experiment002RegisteredHistoryError,
        match="closure-owned|failed",
    ):
        history._issued_barrier_state(barrier)


def test_completed_history_detects_nested_capability_identity_rewrite() -> None:
    completed, state = _counterfeit_completed()
    snapshot = state.records[0].evaluated_snapshot
    object.__setattr__(
        snapshot,
        "model_tensors",
        object.__new__(RegisteredModelTensorEvidence),
    )
    with pytest.raises(
        history.Experiment002RegisteredHistoryError,
        match="closure-owned",
    ):
        history._issued_completed_state(completed)


def test_completed_history_detects_in_place_rank_rewrite() -> None:
    completed, state = _counterfeit_completed()
    object.__setattr__(state.ranked_epochs[0], "zero_based_epoch", 29)
    with pytest.raises(
        history.Experiment002RegisteredHistoryError,
        match="closure-owned",
    ):
        history._issued_completed_state(completed)


@pytest.mark.parametrize("entrypoint", ["issued", "full_verifier"])
@pytest.mark.parametrize(
    "attack",
    [
        "state_seed_subclass",
        "state_process_subclass",
        "state_history_sha_subclass",
        "state_canonical_bytearray",
        "guard_seed_subclass",
        "guard_process_subclass",
        "guard_history_sha_subclass",
        "guard_canonical_bytearray",
        "rank_epoch_bool",
        "rank_numerator_bool",
        "rank_denominator_bool",
        "rank_ce_str_subclass",
        "state_rank_frame_bool",
        "guard_rank_frame_bool",
        "state_record_fingerprint_subclass",
        "guard_identity_frame_subclass",
        "object_seed_subclass",
        "object_history_sha_subclass",
        "object_winner_epoch_bool",
        "lifecycle_process_subclass",
        "lifecycle_phase_subclass",
    ],
)
def test_completed_exact_type_corruption_is_terminal_after_visible_restore(
    entrypoint: str,
    attack: str,
) -> None:
    completed, state = _counterfeit_completed()
    guard = history._COMPLETED_GUARDS[completed]
    original_lifecycle = history._COMPLETED_LIFECYCLES[completed]
    restores: list[tuple[object, str, object]] = []

    def replace(target: object, name: str, value: object) -> None:
        restores.append((target, name, getattr(target, name)))
        object.__setattr__(target, name, value)

    rank = state.ranked_epochs[0]
    if attack == "state_seed_subclass":
        replace(state, "seed", _IntSubclass(state.seed))
    elif attack == "state_process_subclass":
        replace(state, "process_id", _IntSubclass(state.process_id))
    elif attack == "state_history_sha_subclass":
        replace(state, "history_sha256", _StrSubclass(state.history_sha256))
    elif attack == "state_canonical_bytearray":
        replace(state, "canonical_json_bytes", bytearray(state.canonical_json_bytes))
    elif attack == "guard_seed_subclass":
        replace(guard, "seed", _IntSubclass(guard.seed))
    elif attack == "guard_process_subclass":
        replace(guard, "process_id", _IntSubclass(guard.process_id))
    elif attack == "guard_history_sha_subclass":
        replace(guard, "history_sha256", _StrSubclass(guard.history_sha256))
    elif attack == "guard_canonical_bytearray":
        replace(guard, "canonical_json_bytes", bytearray(guard.canonical_json_bytes))
    elif attack == "rank_epoch_bool":
        assert rank.zero_based_epoch == 0
        replace(rank, "zero_based_epoch", False)
    elif attack == "rank_numerator_bool":
        assert rank.macro_f1_exact_numerator == 1
        replace(rank, "macro_f1_exact_numerator", True)
    elif attack == "rank_denominator_bool":
        assert rank.macro_f1_exact_denominator == 1
        replace(rank, "macro_f1_exact_denominator", True)
    elif attack == "rank_ce_str_subclass":
        replace(
            rank,
            "validation_cross_entropy_float64_hex",
            _StrSubclass(rank.validation_cross_entropy_float64_hex),
        )
    elif attack == "state_rank_frame_bool":
        rank_frame = state.rank_frames[0]
        replace(
            state,
            "rank_frames",
            ((False, *rank_frame[1:]), *state.rank_frames[1:]),
        )
    elif attack == "guard_rank_frame_bool":
        guard_rank_frame = guard.rank_frames[0]
        replace(
            guard,
            "rank_frames",
            ((False, *guard_rank_frame[1:]), *guard.rank_frames[1:]),
        )
    elif attack == "state_record_fingerprint_subclass":
        fingerprint = state.record_fingerprints[0]
        replace(
            state,
            "record_fingerprints",
            (_StrSubclass(fingerprint), *state.record_fingerprints[1:]),
        )
    elif attack == "guard_identity_frame_subclass":
        identity_frame = guard.record_identity_frames[0]
        changed_first = (
            _IntSubclass(cast(int, identity_frame[0])),
            *identity_frame[1:],
        )
        replace(
            guard,
            "record_identity_frames",
            (changed_first, *guard.record_identity_frames[1:]),
        )
    elif attack == "object_seed_subclass":
        replace(completed, "_seed", _IntSubclass(completed._seed))
    elif attack == "object_history_sha_subclass":
        replace(
            completed,
            "_history_sha256",
            _StrSubclass(completed._history_sha256),
        )
    elif attack == "object_winner_epoch_bool":
        assert completed._winner_epoch == 0
        replace(completed, "_winner_epoch", False)
    elif attack == "lifecycle_process_subclass":
        history._COMPLETED_LIFECYCLES[completed] = dataclasses.replace(
            original_lifecycle,
            process_id=_IntSubclass(original_lifecycle.process_id),
        )
    else:
        history._COMPLETED_LIFECYCLES[completed] = dataclasses.replace(
            original_lifecycle,
            phase=cast(Any, _StrSubclass("COMPLETE")),
        )

    with pytest.raises((TypeError, history.Experiment002RegisteredHistoryError)):
        if entrypoint == "issued":
            history._issued_completed_state(completed)
        else:
            history.verify_registered_completed_training_history(completed)

    for target, name, value in reversed(restores):
        object.__setattr__(target, name, value)
    history._FAILED_COMPLETED.discard(completed)
    history._COMPLETED_LIFECYCLES[completed] = original_lifecycle
    with pytest.raises(
        history.Experiment002RegisteredHistoryError,
        match="closure-owned|failed",
    ):
        history._issued_completed_state(completed)


@pytest.mark.parametrize(
    "value",
    [
        _StrSubclass(float(np.float64(0.25)).hex()),
        "0x1p-2",
        "-0x0.0p+0",
        "-0x1.0000000000000p+0",
        "inf",
        "nan",
    ],
)
def test_rank_cross_entropy_requires_exact_canonical_finite_nonnegative_hex(
    value: str,
) -> None:
    rank = history.RegisteredEpochRank(
        zero_based_epoch=0,
        macro_f1_exact_numerator=1,
        macro_f1_exact_denominator=1,
        validation_cross_entropy_float64_hex=value,
    )
    with pytest.raises((TypeError, history.Experiment002RegisteredHistoryError)):
        history._rank_frame(rank)


@pytest.mark.parametrize(
    "mutation",
    [
        "evaluated_seed",
        "evaluated_epoch",
        "evaluated_generation",
        "evaluated_continuation",
        "registration_head",
        "source_bundle",
        "handoff_seed",
        "handoff_epoch",
        "handoff_generation",
        "handoff_continuation",
        "handoff_rng_state",
    ],
)
def test_bound_record_fingerprint_covers_every_snapshot_scalar(
    mutation: str,
) -> None:
    _, state = _counterfeit_history()
    bound = _counterfeit_bound(state)
    original = history._bound_record_fingerprint(bound)
    evaluated = bound.evaluated_snapshot
    handoff = bound.handoff_snapshot
    if mutation == "evaluated_seed":
        object.__setattr__(evaluated, "seed", 20_260_720)
    elif mutation == "evaluated_epoch":
        object.__setattr__(evaluated, "zero_based_epoch", 1)
    elif mutation == "evaluated_generation":
        object.__setattr__(evaluated, "optimizer_generation", 314)
    elif mutation == "evaluated_continuation":
        object.__setattr__(evaluated, "another_training_epoch", False)
    elif mutation == "registration_head":
        object.__setattr__(evaluated, "registration_head_commit", "a" * 40)
    elif mutation == "source_bundle":
        object.__setattr__(evaluated, "source_bundle_sha256", "a" * 64)
    elif mutation == "handoff_seed":
        object.__setattr__(handoff, "seed", 20_260_720)
    elif mutation == "handoff_epoch":
        object.__setattr__(handoff, "zero_based_epoch", 1)
    elif mutation == "handoff_generation":
        object.__setattr__(handoff, "optimizer_generation", 314)
    elif mutation == "handoff_continuation":
        object.__setattr__(handoff, "another_training_epoch", False)
    else:
        object.__setattr__(handoff.runtime_digests, "rng_state", b"changed")
    assert history._bound_record_fingerprint(bound) != original


@pytest.mark.parametrize("mutation", ["evaluated_seed", "handoff_rng_state"])
def test_accepted_history_immediately_rejects_nested_scalar_rewrite(
    mutation: str,
) -> None:
    owner, _, bound = _counterfeit_accepted_history()
    if mutation == "evaluated_seed":
        object.__setattr__(bound.evaluated_snapshot, "seed", 20_260_720)
    else:
        object.__setattr__(
            bound.handoff_snapshot.runtime_digests,
            "rng_state",
            b"changed",
        )
    with pytest.raises(
        history.Experiment002RegisteredHistoryError,
        match="closure-owned",
    ):
        history._issued_history_state(owner)


def test_concurrent_barrier_transition_has_one_winner_and_then_fails_closed() -> None:
    _, barrier, state = _counterfeit_barrier()
    start = threading.Barrier(3)
    outcomes: list[str] = []
    outcome_lock = threading.Lock()

    def consume() -> None:
        start.wait()
        try:
            history._transition_barrier(barrier, state, phase="CONSUMED")
        except history.Experiment002RegisteredHistoryError:
            outcome = "rejected"
        else:
            outcome = "accepted"
        with outcome_lock:
            outcomes.append(outcome)

    threads = (threading.Thread(target=consume), threading.Thread(target=consume))
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join()
    assert sorted(outcomes) == ["accepted", "rejected"]
    with pytest.raises(history.Experiment002RegisteredHistoryError):
        history._issued_barrier_state(barrier)


def test_history_authority_is_process_local_across_fork() -> None:
    if not hasattr(os, "fork"):
        pytest.skip("fork is required for the process-local capability test")
    issued, state = _counterfeit_history()
    read_fd, write_fd = os.pipe()
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"This process .* is multi-threaded, use of fork\(\)",
                category=DeprecationWarning,
            )
            child = os.fork()
        if child == 0:
            os.close(read_fd)
            try:
                history._issued_history_state(issued)
            except history.Experiment002RegisteredHistoryError:
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
        assert history._issued_history_state(issued) is state
    finally:
        os.close(read_fd)
        if write_fd >= 0:
            os.close(write_fd)


def test_exact_registered_constants_match_trainer_config() -> None:
    config = json.loads(Path("configs/experiment-002-trainer.json").read_text())
    assert EPOCH_COUNT == 30
    assert UPDATES_PER_EPOCH == 313
    assert TOTAL_UPDATE_COUNT == 9_390
    assert TRAINING_EXAMPLE_COUNT == 40_027
    assert history.VALIDATION_EXAMPLE_COUNT == 10_583
    assert config["checkpoint_selection"]["candidate_population"].startswith(
        "all_30_completed_epoch_checkpoints"
    )
    assert config["digests"]["history"]["domain"] == (_HISTORY_DOMAIN.decode("ascii"))
    assert config["checkpoint_selection"]["ranking"][:3] == [
        "maximize_exact_12_class_macro_F1_fraction",
        "minimize_finite_float64_validation_cross_entropy_with_no_tolerance_or_rounding",
        "prefer_lower_zero_based_epoch_index",
    ]


def test_run_registration_is_still_absent_and_tests_do_not_call_registered_routes() -> (
    None
):
    assert not Path("configs/experiment-002-run.json").exists()
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                called.add(node.func.attr)
            elif isinstance(node.func, ast.Name):
                called.add(node.func.id)
    assert not called & {
        "create_registered_training_history",
        "consume_registered_evaluated_epoch",
        "complete_registered_training_history",
        "evaluate_registered_epoch",
        "execute_registered_training_epoch",
    }


def test_no_generic_completed_value_or_filesystem_surface() -> None:
    for capability in (
        object.__new__(CompleteUpdateTraceEvidence),
        object.__new__(history.RegisteredCompletedTrainingHistory),
    ):
        assert not hasattr(capability, "path")
        assert not hasattr(capability, "publish")
        assert not hasattr(capability, "safetensors")
