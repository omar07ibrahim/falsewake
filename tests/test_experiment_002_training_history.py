from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import pickle
import threading
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import falsewake.experiment_002_training_history as history

type ConfusionMatrix = tuple[tuple[int, ...], ...]


def _digest(character: str) -> str:
    assert len(character) == 1 and character in "0123456789abcdef"
    return character * 64


def _perfect_confusion(value: int = 1) -> ConfusionMatrix:
    return tuple(
        tuple(value if row == column else 0 for column in range(12))
        for row in range(12)
    )


def _sparse_confusion() -> ConfusionMatrix:
    return tuple(
        tuple(1 if row == column == 0 else 0 for column in range(12))
        for row in range(12)
    )


def _session(
    epoch_count: int = 2,
    updates_per_epoch: int = 2,
    *,
    validation_digest: str | None = None,
    trace_digest: str | None = None,
) -> history.TrainingHistorySession:
    return history._create_synthetic_training_history_session(
        layout=history._SyntheticHistoryLayout(
            epoch_count=epoch_count,
            updates_per_epoch=updates_per_epoch,
        ),
        validation_input_digest=(
            _digest("a") if validation_digest is None else validation_digest
        ),
        complete_update_trace_digest=(
            _digest("b") if trace_digest is None else trace_digest
        ),
    )


def _issue(
    session: history.TrainingHistorySession,
    *,
    index: int,
    previous: history.HistoryAcceptedEpoch | None,
    confusion: ConfusionMatrix | None = None,
    training_cross_entropy: np.float64 | None = None,
    validation_cross_entropy: np.float64 | None = None,
) -> history.EvaluatedEpoch:
    return history._issue_synthetic_evaluated_epoch(
        session,
        accepted_previous=previous,
        zero_based_epoch=index,
        training_population_digest=_digest("c"),
        training_cross_entropy=(
            np.float64(index + 0.5)
            if training_cross_entropy is None
            else training_cross_entropy
        ),
        epoch_update_trace_digest=_digest("d"),
        validation_confusion_matrix=(
            _perfect_confusion() if confusion is None else confusion
        ),
        validation_cross_entropy=(
            np.float64(index + 0.25)
            if validation_cross_entropy is None
            else validation_cross_entropy
        ),
        validation_prediction_digest=_digest("e"),
        model_tensor_digest=_digest("f"),
    )


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


def test_source_is_a_strictly_unregistered_dependency_free_kernel() -> None:
    source = Path("src/falsewake/experiment_002_training_history.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    imported_modules: set[str] = set()
    module_functions: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name)
                imported_roots.add(alias.name.split(".", maxsplit=1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)
            imported_roots.add(node.module.split(".", maxsplit=1)[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            module_functions.append(node.name)

    assert imported_roots == {
        "__future__",
        "dataclasses",
        "fractions",
        "hashlib",
        "json",
        "numpy",
        "threading",
        "typing",
        "weakref",
    }
    assert not any(module.startswith("falsewake") for module in imported_modules)
    assert not imported_roots & {
        "os",
        "pathlib",
        "safetensors",
        "subprocess",
        "torch",
    }
    assert all(name.startswith("_") for name in module_functions)
    assert "202607" not in source
    assert "40027" not in source
    assert "10583" not in source
    assert "9390" not in source
    assert "313" not in source


@pytest.mark.parametrize(
    ("epoch_count", "updates_per_epoch", "error_type"),
    [
        (0, 1, history.Experiment002TrainingHistoryError),
        (-1, 1, history.Experiment002TrainingHistoryError),
        (5, 1, history.Experiment002TrainingHistoryError),
        (30, 1, history.Experiment002TrainingHistoryError),
        (1, 0, history.Experiment002TrainingHistoryError),
        (1, -1, history.Experiment002TrainingHistoryError),
        (1, 5, history.Experiment002TrainingHistoryError),
        (True, 1, TypeError),
        (1, False, TypeError),
        (1.0, 1, TypeError),
        (1, 1.0, TypeError),
    ],
)
def test_private_layout_is_hard_capped_and_exactly_typed(
    epoch_count: object,
    updates_per_epoch: object,
    error_type: type[BaseException],
) -> None:
    with pytest.raises(error_type):
        history._SyntheticHistoryLayout(
            epoch_count=cast(int, epoch_count),
            updates_per_epoch=cast(int, updates_per_epoch),
        )


def test_private_layout_accepts_only_the_tiny_boundary() -> None:
    layout = history._SyntheticHistoryLayout(4, 4)
    assert layout.epoch_count == history._MAX_SYNTHETIC_EPOCHS == 4
    assert layout.updates_per_epoch == history._MAX_SYNTHETIC_UPDATES_PER_EPOCH == 4
    assert history._SYNTHETIC_SEED == 0


def test_session_factory_rejects_invalid_layout_and_digest_inputs() -> None:
    with pytest.raises(TypeError, match="exact _SyntheticHistoryLayout"):
        history._create_synthetic_training_history_session(
            layout=cast(history._SyntheticHistoryLayout, object()),
            validation_input_digest=_digest("a"),
            complete_update_trace_digest=_digest("b"),
        )
    layout = history._SyntheticHistoryLayout(1)
    object.__setattr__(layout, "epoch_count", 30)
    with pytest.raises(history.Experiment002TrainingHistoryError, match="epoch limit"):
        history._create_synthetic_training_history_session(
            layout=layout,
            validation_input_digest=_digest("a"),
            complete_update_trace_digest=_digest("b"),
        )
    for validation_digest, trace_digest in (
        ("A" * 64, _digest("b")),
        (_digest("a"), "b" * 63),
    ):
        with pytest.raises(history.Experiment002TrainingHistoryError, match="SHA-256"):
            history._create_synthetic_training_history_session(
                layout=history._SyntheticHistoryLayout(1),
                validation_input_digest=validation_digest,
                complete_update_trace_digest=trace_digest,
            )


def test_one_epoch_golden_canonical_history_and_domain_digest() -> None:
    session = _session(epoch_count=1, updates_per_epoch=2)
    epoch = _issue(
        session,
        index=0,
        previous=None,
        training_cross_entropy=np.float64(0.5),
        validation_cross_entropy=np.float64(0.25),
    )
    barrier = session.consume(epoch)
    completed = session.complete(barrier)

    expected_epoch = {
        "zero_based_epoch": 0,
        "first_global_update": 0,
        "last_global_update_inclusive": 1,
        "training_population_digest": _digest("c"),
        "training_cross_entropy_float64_hex": "0x1.0000000000000p-1",
        "epoch_update_trace_digest": _digest("d"),
        "validation_input_digest": _digest("a"),
        "validation_confusion_matrix": [list(row) for row in _perfect_confusion()],
        "validation_cross_entropy_float64_hex": "0x1.0000000000000p-2",
        "macro_f1_exact_numerator": 1,
        "macro_f1_exact_denominator": 1,
        "validation_prediction_digest": _digest("e"),
        "model_tensor_digest": _digest("f"),
    }
    expected_document = {
        "schema_version": 1,
        "experiment": "002",
        "seed": 0,
        "validation_input_digest": _digest("a"),
        "complete_update_trace_digest": _digest("b"),
        "epochs": [expected_epoch],
    }
    expected_bytes = (
        json.dumps(
            expected_document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")

    assert completed.synthetic_seed == 0
    assert completed.epoch_count == 1
    assert completed.canonical_json_bytes == expected_bytes
    assert completed.history_sha256 == (
        "fd2aaa0fa9a53d214b1a0f9068f12525fce55c47f8402ba78bb9b3be74c00b68"
    )
    assert (
        completed.history_sha256
        == hashlib.sha256(history._HISTORY_DOMAIN + expected_bytes).hexdigest()
    )
    assert expected_bytes.endswith(b"\n")
    assert not expected_bytes.endswith(b"\n\n")
    assert b" " not in expected_bytes
    assert completed.winner.zero_based_epoch == 0
    assert completed.winner.macro_f1_exact_numerator == 1
    assert completed.winner.macro_f1_exact_denominator == 1

    parsed = json.loads(completed.canonical_json_bytes)
    assert parsed == expected_document
    _assert_no_json_floats(parsed)
    assert set(parsed) == {
        "schema_version",
        "experiment",
        "seed",
        "validation_input_digest",
        "complete_update_trace_digest",
        "epochs",
    }
    assert set(parsed["epochs"][0]) == {
        "zero_based_epoch",
        "first_global_update",
        "last_global_update_inclusive",
        "training_population_digest",
        "training_cross_entropy_float64_hex",
        "epoch_update_trace_digest",
        "validation_input_digest",
        "validation_confusion_matrix",
        "validation_cross_entropy_float64_hex",
        "macro_f1_exact_numerator",
        "macro_f1_exact_denominator",
        "validation_prediction_digest",
        "model_tensor_digest",
    }


def test_exact_ranking_uses_macro_f1_then_float64_ce_then_lower_epoch() -> None:
    session = _session(epoch_count=4, updates_per_epoch=1)
    barrier: history.HistoryAcceptedEpoch | None = None
    cases = (
        (_perfect_confusion(1), np.float64(0.5)),
        (_sparse_confusion(), np.float64(0.01)),
        (_perfect_confusion(2), np.float64(0.25)),
        (_perfect_confusion(3), np.float64(0.25)),
    )
    for index, (confusion, cross_entropy) in enumerate(cases):
        epoch = _issue(
            session,
            index=index,
            previous=barrier,
            confusion=confusion,
            validation_cross_entropy=cross_entropy,
        )
        barrier = session.consume(epoch)
    assert barrier is not None
    completed = session.complete(barrier)

    assert [rank.zero_based_epoch for rank in completed.ranked_epochs] == [2, 3, 0, 1]
    assert completed.ranked_epochs[0].macro_f1_exact_numerator == 1
    assert completed.ranked_epochs[0].macro_f1_exact_denominator == 1
    assert completed.ranked_epochs[-1].macro_f1_exact_numerator == 1
    assert completed.ranked_epochs[-1].macro_f1_exact_denominator == 12
    assert completed.ranked_epochs[0].validation_cross_entropy_float64_hex == (
        float(np.float64(0.25)).hex()
    )


def test_epoch_macro_f1_is_recomputed_exactly_from_confusion() -> None:
    confusion = _sparse_confusion()
    assert history._macro_f1_from_confusion(confusion) == Fraction(1, 12)
    session = _session(epoch_count=1)
    barrier = session.consume(
        _issue(session, index=0, previous=None, confusion=confusion)
    )
    completed = session.complete(barrier)
    document = json.loads(completed.canonical_json_bytes)
    record = document["epochs"][0]
    assert record["macro_f1_exact_numerator"] == 1
    assert record["macro_f1_exact_denominator"] == 12


def test_session_methods_accept_capabilities_only() -> None:
    assert tuple(
        inspect.signature(history.TrainingHistorySession.consume).parameters
    ) == (
        "self",
        "evaluated_epoch",
    )
    assert tuple(
        inspect.signature(history.TrainingHistorySession.complete).parameters
    ) == (
        "self",
        "accepted_epoch",
    )


def test_capability_constructors_copying_serialization_and_counterfeits_fail() -> None:
    for capability_type in (
        history.TrainingHistorySession,
        history.EvaluatedEpoch,
        history.HistoryAcceptedEpoch,
    ):
        with pytest.raises(TypeError, match="issuer-only"):
            capability_type()

    session = _session(epoch_count=2)
    epoch = _issue(session, index=0, previous=None)
    barrier = session.consume(epoch)
    for capability in (session, epoch, barrier):
        with pytest.raises(TypeError, match="cannot be copied"):
            copy.copy(capability)
        with pytest.raises(TypeError, match="cannot be copied"):
            copy.deepcopy(capability)
        with pytest.raises(TypeError, match="cannot be serialized"):
            pickle.dumps(capability)

    forged_session = object.__new__(history.TrainingHistorySession)
    forged_epoch = object.__new__(history.EvaluatedEpoch)
    forged_barrier = object.__new__(history.HistoryAcceptedEpoch)
    with pytest.raises(history.Experiment002TrainingHistoryError, match="not issued"):
        history._synthetic_history_session_snapshot(forged_session)
    with pytest.raises(history.Experiment002TrainingHistoryError, match="not issued"):
        history._issued_evaluated_epoch_state(forged_epoch)
    with pytest.raises(history.Experiment002TrainingHistoryError, match="not issued"):
        history._issued_accepted_epoch_state(forged_barrier)


def test_skip_is_rejected_and_terminal() -> None:
    session = _session()
    with pytest.raises(history.Experiment002TrainingHistoryError, match="skip"):
        _issue(session, index=1, previous=None)
    snapshot = history._synthetic_history_session_snapshot(session)
    assert snapshot.phase == "FAILED"
    with pytest.raises(history.Experiment002TrainingHistoryError, match="failed"):
        _issue(session, index=0, previous=None)


def test_duplicate_active_issuance_is_rejected_and_terminal() -> None:
    session = _session()
    epoch = _issue(session, index=0, previous=None)
    with pytest.raises(
        history.Experiment002TrainingHistoryError, match="already active"
    ):
        _issue(session, index=0, previous=None)
    snapshot = history._synthetic_history_session_snapshot(session)
    assert snapshot.phase == "FAILED"
    assert history._issued_evaluated_epoch_state(epoch).phase == "FAILED"


def test_replay_consumption_is_rejected_and_terminal() -> None:
    session = _session()
    epoch = _issue(session, index=0, previous=None)
    barrier = session.consume(epoch)
    assert history._issued_accepted_epoch_state(barrier).phase == "ISSUED"
    with pytest.raises(history.Experiment002TrainingHistoryError, match="replayed"):
        session.consume(epoch)
    assert history._synthetic_history_session_snapshot(session).phase == "FAILED"
    assert history._issued_accepted_epoch_state(barrier).phase == "FAILED"


def test_cross_session_epoch_poisoning_is_terminal_for_both_flows() -> None:
    owner = _session()
    foreign = _session()
    epoch = _issue(owner, index=0, previous=None)

    with pytest.raises(history.Experiment002TrainingHistoryError, match="foreign"):
        foreign.consume(epoch)
    assert history._synthetic_history_session_snapshot(foreign).phase == "FAILED"
    assert history._synthetic_history_session_snapshot(owner).phase == "FAILED"
    assert history._issued_evaluated_epoch_state(epoch).phase == "FAILED"

    with pytest.raises(history.Experiment002TrainingHistoryError, match="failed"):
        owner.consume(epoch)


def test_cross_session_barrier_is_rejected_and_terminal() -> None:
    owner = _session()
    owner_barrier = owner.consume(_issue(owner, index=0, previous=None))
    foreign = _session()
    foreign_barrier = foreign.consume(_issue(foreign, index=0, previous=None))
    assert foreign_barrier is not owner_barrier

    with pytest.raises(
        history.Experiment002TrainingHistoryError, match="does not match"
    ):
        _issue(foreign, index=1, previous=owner_barrier)
    assert history._synthetic_history_session_snapshot(foreign).phase == "FAILED"
    assert history._synthetic_history_session_snapshot(owner).phase == "FAILED"
    assert history._issued_accepted_epoch_state(owner_barrier).phase == "FAILED"

    with pytest.raises(history.Experiment002TrainingHistoryError, match="failed"):
        _issue(owner, index=1, previous=owner_barrier)


def test_cross_session_completion_barrier_poisoning_is_terminal() -> None:
    owner = _session(epoch_count=1)
    owner_barrier = owner.consume(_issue(owner, index=0, previous=None))
    foreign = _session(epoch_count=1)
    foreign.consume(_issue(foreign, index=0, previous=None))

    with pytest.raises(history.Experiment002TrainingHistoryError, match="final"):
        foreign.complete(owner_barrier)
    assert history._synthetic_history_session_snapshot(owner).phase == "FAILED"
    assert history._synthetic_history_session_snapshot(foreign).phase == "FAILED"
    assert history._issued_accepted_epoch_state(owner_barrier).phase == "FAILED"


def test_counterfeit_capability_use_fails_the_receiving_session_terminally() -> None:
    session = _session(epoch_count=1)
    counterfeit = object.__new__(history.EvaluatedEpoch)
    with pytest.raises(history.Experiment002TrainingHistoryError, match="not issued"):
        session.consume(counterfeit)
    assert history._synthetic_history_session_snapshot(session).phase == "FAILED"


def test_concurrent_replay_can_accept_at_most_once_and_then_fails_terminally() -> None:
    session = _session(epoch_count=2)
    epoch = _issue(session, index=0, previous=None)
    start = threading.Barrier(3)
    outcomes: list[str] = []
    outcomes_lock = threading.Lock()

    def consume() -> None:
        start.wait()
        try:
            session.consume(epoch)
        except history.Experiment002TrainingHistoryError:
            outcome = "rejected"
        else:
            outcome = "accepted"
        with outcomes_lock:
            outcomes.append(outcome)

    threads = (threading.Thread(target=consume), threading.Thread(target=consume))
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join()

    assert sorted(outcomes) == ["accepted", "rejected"]
    assert history._synthetic_history_session_snapshot(session).phase == "FAILED"


def test_early_completion_is_rejected_and_terminal() -> None:
    session = _session(epoch_count=2)
    barrier = session.consume(_issue(session, index=0, previous=None))
    with pytest.raises(history.Experiment002TrainingHistoryError, match="final"):
        session.complete(barrier)
    snapshot = history._synthetic_history_session_snapshot(session)
    assert snapshot.phase == "FAILED"
    with pytest.raises(history.Experiment002TrainingHistoryError, match="failed"):
        _issue(session, index=1, previous=barrier)


def test_completion_is_terminal_and_cannot_be_replayed() -> None:
    session = _session(epoch_count=1)
    epoch = _issue(session, index=0, previous=None)
    barrier = session.consume(epoch)
    completed = session.complete(barrier)
    snapshot = history._synthetic_history_session_snapshot(session)
    assert snapshot.phase == "COMPLETE"
    assert snapshot.accepted_epoch_count == 1
    assert not snapshot.has_active_epoch
    assert not snapshot.has_active_barrier
    with pytest.raises(history.Experiment002TrainingHistoryError, match="complete"):
        session.complete(barrier)
    with pytest.raises(history.Experiment002TrainingHistoryError, match="complete"):
        session.consume(epoch)
    assert completed.epoch_count == 1


@pytest.mark.parametrize(
    "corruption",
    [
        "epoch_bool",
        "uppercase_digest",
        "short_digest",
        "python_float_training_ce",
        "nan_training_ce",
        "infinite_validation_ce",
        "negative_validation_ce",
        "negative_zero_validation_ce",
        "confusion_list",
        "confusion_short",
        "confusion_row_list",
        "confusion_negative",
        "confusion_too_large",
        "confusion_bool",
    ],
)
def test_invalid_raw_synthetic_issuance_fails_terminally(corruption: str) -> None:
    session = _session(epoch_count=1)
    kwargs: dict[str, Any] = {
        "accepted_previous": None,
        "zero_based_epoch": 0,
        "training_population_digest": _digest("c"),
        "training_cross_entropy": np.float64(0.5),
        "epoch_update_trace_digest": _digest("d"),
        "validation_confusion_matrix": _perfect_confusion(),
        "validation_cross_entropy": np.float64(0.25),
        "validation_prediction_digest": _digest("e"),
        "model_tensor_digest": _digest("f"),
    }
    if corruption == "epoch_bool":
        kwargs["zero_based_epoch"] = True
    elif corruption == "uppercase_digest":
        kwargs["training_population_digest"] = "A" * 64
    elif corruption == "short_digest":
        kwargs["model_tensor_digest"] = "f" * 63
    elif corruption == "python_float_training_ce":
        kwargs["training_cross_entropy"] = 0.5
    elif corruption == "nan_training_ce":
        kwargs["training_cross_entropy"] = np.float64(np.nan)
    elif corruption == "infinite_validation_ce":
        kwargs["validation_cross_entropy"] = np.float64(np.inf)
    elif corruption == "negative_validation_ce":
        kwargs["validation_cross_entropy"] = np.float64(-0.1)
    elif corruption == "negative_zero_validation_ce":
        kwargs["validation_cross_entropy"] = np.float64(-0.0)
    elif corruption == "confusion_list":
        kwargs["validation_confusion_matrix"] = list(_perfect_confusion())
    elif corruption == "confusion_short":
        kwargs["validation_confusion_matrix"] = _perfect_confusion()[:-1]
    else:
        mutable = [list(row) for row in _perfect_confusion()]
        if corruption == "confusion_row_list":
            candidate: object = tuple(mutable)
        elif corruption == "confusion_negative":
            mutable[0][0] = -1
            candidate = tuple(tuple(row) for row in mutable)
        elif corruption == "confusion_too_large":
            mutable[0][0] = 5
            candidate = tuple(tuple(row) for row in mutable)
        else:
            mutable[0][0] = True
            candidate = tuple(tuple(row) for row in mutable)
        kwargs["validation_confusion_matrix"] = candidate

    with pytest.raises((TypeError, history.Experiment002TrainingHistoryError)):
        history._issue_synthetic_evaluated_epoch(session, **kwargs)
    snapshot = history._synthetic_history_session_snapshot(session)
    assert snapshot.phase == "FAILED"
    assert snapshot.accepted_epoch_count == 0


def test_tampered_epoch_state_is_rejected_before_history_acceptance() -> None:
    session = _session(epoch_count=1)
    epoch = _issue(session, index=0, previous=None)
    issued = history._issued_evaluated_epoch_state(epoch)
    object.__setattr__(issued.record, "validation_input_digest", _digest("0"))
    with pytest.raises(
        history.Experiment002TrainingHistoryError,
        match="differs from the session",
    ):
        session.consume(epoch)
    assert history._synthetic_history_session_snapshot(session).phase == "FAILED"


def test_reordered_internal_records_cannot_complete() -> None:
    session = _session(epoch_count=2)
    first = session.consume(_issue(session, index=0, previous=None))
    second = session.consume(_issue(session, index=1, previous=first))
    state = history._issued_session_state(session)
    state.records.reverse()
    with pytest.raises(
        history.Experiment002TrainingHistoryError, match="exact epoch order"
    ):
        session.complete(second)
    assert history._synthetic_history_session_snapshot(session).phase == "FAILED"


def test_tampered_epoch_route_marker_is_rejected_terminally() -> None:
    session = _session(epoch_count=1)
    epoch = _issue(session, index=0, previous=None)
    history._issued_evaluated_epoch_state(epoch).route_marker = object()
    with pytest.raises(history.Experiment002TrainingHistoryError, match="authority"):
        session.consume(epoch)
    assert history._synthetic_history_session_snapshot(session).phase == "FAILED"


def test_invalid_internal_phase_is_normalized_to_terminal_failure() -> None:
    session = _session(epoch_count=1)
    state = history._issued_session_state(session)
    state.phase = cast(Any, "BROKEN")
    with pytest.raises(history.Experiment002TrainingHistoryError, match="authority"):
        _issue(session, index=0, previous=None)
    assert history._synthetic_history_session_snapshot(session).phase == "FAILED"


def test_incoherent_active_capability_state_fails_terminally() -> None:
    session = _session(epoch_count=1)
    state = history._issued_session_state(session)
    state.active_epoch_token = object()
    with pytest.raises(history.Experiment002TrainingHistoryError, match="authority"):
        _issue(session, index=0, previous=None)
    assert state.phase == "FAILED"


def test_accepted_record_rewrite_is_detected_and_terminal() -> None:
    session = _session(epoch_count=1)
    barrier = session.consume(_issue(session, index=0, previous=None))
    state = history._issued_session_state(session)
    object.__setattr__(state.records[0], "training_population_digest", _digest("0"))
    with pytest.raises(
        history.Experiment002TrainingHistoryError,
        match="record authority changed",
    ):
        session.complete(barrier)
    assert state.phase == "FAILED"
    assert history._issued_accepted_epoch_state(barrier).phase == "FAILED"


def test_active_epoch_rewrite_is_detected_before_acceptance() -> None:
    session = _session(epoch_count=1)
    epoch = _issue(session, index=0, previous=None)
    epoch_state = history._issued_evaluated_epoch_state(epoch)
    object.__setattr__(epoch_state.record, "model_tensor_digest", _digest("0"))
    with pytest.raises(
        history.Experiment002TrainingHistoryError,
        match="record changed",
    ):
        session.consume(epoch)
    assert history._issued_session_state(session).phase == "FAILED"
    assert epoch_state.phase == "FAILED"


@pytest.mark.parametrize("corruption", ["validation_digest", "trace_digest", "layout"])
def test_session_metadata_and_layout_rewrites_are_terminal(corruption: str) -> None:
    session = _session(epoch_count=2, updates_per_epoch=2)
    state = history._issued_session_state(session)
    if corruption == "validation_digest":
        state.validation_input_digest = _digest("0")
    elif corruption == "trace_digest":
        state.complete_update_trace_digest = _digest("0")
    else:
        object.__setattr__(state.layout, "updates_per_epoch", 3)
    with pytest.raises(history.Experiment002TrainingHistoryError, match="authority"):
        _issue(session, index=0, previous=None)
    assert state.phase == "FAILED"


def test_record_and_barrier_rollback_cannot_replay_epoch_zero() -> None:
    session = _session(epoch_count=1)
    epoch = _issue(session, index=0, previous=None)
    barrier = session.consume(epoch)
    state = history._issued_session_state(session)
    state.records.clear()
    state.active_barrier = None
    state.active_barrier_token = None
    with pytest.raises(
        history.Experiment002TrainingHistoryError,
        match="record authority changed|capability lifecycle changed",
    ):
        _issue(session, index=0, previous=None)
    assert state.phase == "FAILED"
    assert history._issued_accepted_epoch_state(barrier).phase == "FAILED"


def test_failed_and_complete_phases_cannot_be_reopened() -> None:
    failed = _session(epoch_count=1)
    with pytest.raises(history.Experiment002TrainingHistoryError, match="skip"):
        _issue(failed, index=1, previous=None)
    failed_state = history._issued_session_state(failed)
    failed_state.phase = cast(Any, "OPEN")
    with pytest.raises(history.Experiment002TrainingHistoryError, match="lifecycle"):
        _issue(failed, index=0, previous=None)
    assert failed_state.phase == "FAILED"

    completed = _session(epoch_count=1)
    barrier = completed.consume(_issue(completed, index=0, previous=None))
    completed.complete(barrier)
    completed_state = history._issued_session_state(completed)
    completed_state.phase = cast(Any, "OPEN")
    with pytest.raises(history.Experiment002TrainingHistoryError, match="lifecycle"):
        _issue(completed, index=0, previous=None)
    assert completed_state.phase == "FAILED"


def test_session_registry_or_guard_removal_is_irreversible() -> None:
    session = _session(epoch_count=1)
    state = history._SESSIONS.pop(session)
    with pytest.raises(history.Experiment002TrainingHistoryError, match="not issued"):
        history._synthetic_history_session_snapshot(session)
    history._SESSIONS[session] = state
    with pytest.raises(history.Experiment002TrainingHistoryError, match="lifecycle"):
        _issue(session, index=0, previous=None)
    assert state.phase == "FAILED"

    another = _session(epoch_count=1)
    another_state = history._issued_session_state(another)
    guard = history._SESSION_GUARDS.pop(another)
    with pytest.raises(
        history.Experiment002TrainingHistoryError, match="lock authority"
    ):
        _issue(another, index=0, previous=None)
    history._SESSION_GUARDS[another] = guard
    another_state.phase = cast(Any, "OPEN")
    with pytest.raises(history.Experiment002TrainingHistoryError, match="lifecycle"):
        _issue(another, index=0, previous=None)
    assert another_state.phase == "FAILED"


def test_lock_is_verified_before_context_entry_and_failure_is_irreversible() -> None:
    session = _session(epoch_count=1)
    state = history._issued_session_state(session)
    trusted_lock = history._SESSION_ANCHORS[session][7]
    state.lock = cast(Any, object())
    with pytest.raises(
        history.Experiment002TrainingHistoryError, match="lock authority"
    ):
        _issue(session, index=0, previous=None)
    assert state.phase == "FAILED"
    state.lock = trusted_lock
    state.phase = cast(Any, "OPEN")
    with pytest.raises(history.Experiment002TrainingHistoryError, match="lifecycle"):
        _issue(session, index=0, previous=None)
    assert state.phase == "FAILED"


def test_foreign_owner_lock_tamper_still_poison_both_flows() -> None:
    owner = _session(epoch_count=2)
    barrier = owner.consume(_issue(owner, index=0, previous=None))
    owner_state = history._issued_session_state(owner)
    owner_state.lock = cast(Any, object())
    receiver = _session(epoch_count=1)
    with pytest.raises(history.Experiment002TrainingHistoryError, match="epoch zero"):
        _issue(receiver, index=0, previous=barrier)
    assert owner_state.phase == "FAILED"
    assert history._issued_session_state(receiver).phase == "FAILED"
    assert history._issued_accepted_epoch_state(barrier).phase == "FAILED"


def test_transition_specific_lifecycle_cannot_reseal_a_record_rewrite() -> None:
    session = _session(epoch_count=1)
    barrier = session.consume(_issue(session, index=0, previous=None))
    state = history._issued_session_state(session)
    object.__setattr__(state.records[0], "model_tensor_digest", _digest("0"))
    for event in ("ISSUE", "CONSUME", "COMPLETE"):
        with pytest.raises(
            history.Experiment002TrainingHistoryError, match="transition"
        ):
            history._advance_session_lifecycle(
                session,
                state,
                event=cast(Any, event),
            )
    with pytest.raises(
        history.Experiment002TrainingHistoryError,
        match="record authority changed",
    ):
        session.complete(barrier)
    assert state.phase == "FAILED"
    assert not hasattr(history, "_sync_session_lifecycle")


def test_coherent_soft_guard_and_state_mutation_cannot_change_anchor() -> None:
    session = _session(epoch_count=1)
    state = history._issued_session_state(session)
    guard = history._SESSION_GUARDS[session]
    state.validation_input_digest = _digest("0")
    object.__setattr__(guard, "validation_input_digest", _digest("0"))
    with pytest.raises(
        history.Experiment002TrainingHistoryError, match="lock authority"
    ):
        _issue(session, index=0, previous=None)
    assert state.phase == "FAILED"


def test_coherent_lifecycle_and_active_record_mutation_cannot_change_anchor() -> None:
    session = _session(epoch_count=1)
    epoch = _issue(session, index=0, previous=None)
    epoch_state = history._issued_evaluated_epoch_state(epoch)
    object.__setattr__(epoch_state.record, "model_tensor_digest", _digest("0"))
    lifecycle = history._SESSION_LIFECYCLES[session]
    object.__setattr__(
        lifecycle,
        "active_epoch_fingerprint",
        history._record_authority_fingerprint(epoch_state.record),
    )
    with pytest.raises(history.Experiment002TrainingHistoryError, match="authority"):
        session.consume(epoch)
    assert history._issued_session_state(session).phase == "FAILED"
    assert epoch_state.phase == "FAILED"


def test_completion_rechecks_fingerprints_after_double_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session(epoch_count=1)
    barrier = session.consume(_issue(session, index=0, previous=None))
    state = history._issued_session_state(session)
    original = history._canonical_history_bytes
    calls = 0

    def mutate_after_serialization(candidate: history._SessionState) -> bytes:
        nonlocal calls
        payload = original(candidate)
        calls += 1
        if calls == 2:
            object.__setattr__(
                candidate.records[0],
                "training_population_digest",
                _digest("0"),
            )
        return payload

    monkeypatch.setattr(history, "_canonical_history_bytes", mutate_after_serialization)
    with pytest.raises(
        history.Experiment002TrainingHistoryError,
        match="record authority changed",
    ):
        session.complete(barrier)
    assert calls == 2
    assert state.phase == "FAILED"


def test_foreign_barrier_at_epoch_zero_poison_both_sessions() -> None:
    owner = _session(epoch_count=2)
    barrier = owner.consume(_issue(owner, index=0, previous=None))
    receiver = _session(epoch_count=1)
    with pytest.raises(
        history.Experiment002TrainingHistoryError,
        match="epoch zero",
    ):
        _issue(receiver, index=0, previous=barrier)
    assert history._issued_session_state(owner).phase == "FAILED"
    assert history._issued_session_state(receiver).phase == "FAILED"
    assert history._issued_accepted_epoch_state(barrier).phase == "FAILED"


def test_ranking_preserves_adjacent_and_subnormal_float64_values() -> None:
    session = _session(epoch_count=4, updates_per_epoch=1)
    smallest_subnormal = np.nextafter(np.float64(0.0), np.float64(1.0))
    above_quarter = np.nextafter(np.float64(0.25), np.float64(np.inf))
    cross_entropies = (
        above_quarter,
        smallest_subnormal,
        np.float64(0.0),
        np.float64(0.0),
    )
    barrier: history.HistoryAcceptedEpoch | None = None
    for index, cross_entropy in enumerate(cross_entropies):
        barrier = session.consume(
            _issue(
                session,
                index=index,
                previous=barrier,
                validation_cross_entropy=cross_entropy,
            )
        )
    assert barrier is not None
    completed = session.complete(barrier)
    assert [rank.zero_based_epoch for rank in completed.ranked_epochs] == [2, 3, 1, 0]
    assert completed.ranked_epochs[2].validation_cross_entropy_float64_hex == (
        "0x0.0000000000001p-1022"
    )
    assert completed.ranked_epochs[3].validation_cross_entropy_float64_hex == (
        float(above_quarter).hex()
    )


def test_completed_value_is_explicitly_unregistered_and_has_no_artifact_route() -> None:
    session = _session(epoch_count=1)
    barrier = session.consume(_issue(session, index=0, previous=None))
    completed = session.complete(barrier)
    assert type(completed) is history._UnregisteredCompletedTrainingHistory
    assert type(completed.canonical_json_bytes) is bytes
    assert type(completed.ranked_epochs) is tuple
    for forbidden in (
        "checkpoint",
        "model",
        "parameters",
        "path",
        "publication",
        "safetensors",
        "tensors",
    ):
        assert not hasattr(completed, forbidden)

    forged = history._UnregisteredCompletedTrainingHistory(
        synthetic_seed=0,
        epoch_count=1,
        canonical_json_bytes=b"{}\n",
        history_sha256="not-authority",
        ranked_epochs=(
            history._UnregisteredEpochRank(
                zero_based_epoch=0,
                macro_f1_exact_numerator=1,
                macro_f1_exact_denominator=1,
                validation_cross_entropy_float64_hex="0x0.0p+0",
            ),
        ),
    )
    assert forged.history_sha256 == "not-authority"
    assert not isinstance(forged, history.EvaluatedEpoch)
    assert not isinstance(forged, history.HistoryAcceptedEpoch)
