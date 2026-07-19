from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import os
import pickle
import struct
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, NoReturn, cast

import pytest
import safetensors.torch
import torch
from torch import Tensor

import falsewake.experiment_002_registered_checkpoint as checkpoint
import falsewake.experiment_002_registered_evaluator as registered_evaluator
import falsewake.experiment_002_registered_history as registered_history
import falsewake.experiment_002_training_evidence as training_evidence
from falsewake.causal_kws import CausalKWS
from falsewake.experiment_002_evidence import RegisteredValidationInputs
from falsewake.experiment_002_registered_evaluator import RegisteredEvaluatedEpoch
from falsewake.experiment_002_registered_executor import RegisteredTrainingExecutor
from falsewake.experiment_002_registered_history import (
    RegisteredCompletedTrainingHistory,
    RegisteredEpochRank,
    RegisteredTrainingHistory,
)
from falsewake.experiment_002_run_authority import VerifiedRunRegistration
from falsewake.experiment_002_training_evidence import (
    CompleteUpdateTraceEvidence,
    RegisteredModelTensorEvidence,
)

_SEED = 20_260_719
_WINNER_EPOCH = 29
_HISTORY_SHA256 = "a" * 64
_VALIDATION_INPUTS_SHA256 = "b" * 64
_COMPLETE_TRACE_SHA256 = "c" * 64
_REGISTRATION_SHA256 = "d" * 64
_SOURCE_BUNDLE_SHA256 = "e" * 64
_VALIDATION_PREDICTIONS_SHA256 = "f" * 64
_OPTIMIZER_SHA256 = "1" * 64
_TORCH_RNG_SHA256 = "2" * 64
_EVALUATED_AUTHORITY_SHA256 = "3" * 64


@dataclass(frozen=True, slots=True)
class _SyntheticWorld:
    history: RegisteredCompletedTrainingHistory
    winner: RegisteredEvaluatedEpoch
    model_tensors: RegisteredModelTensorEvidence
    names: tuple[str, ...]
    tensors: tuple[Tensor, ...]
    model_sha256: str
    history_snapshot: registered_history._RegisteredCompletedHistorySnapshot
    evaluated_snapshot: registered_evaluator._RegisteredEvaluatedEpochSnapshot
    history_route_calls: list[RegisteredCompletedTrainingHistory]


def _causal_kws_tensors() -> tuple[tuple[str, ...], tuple[Tensor, ...]]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(_SEED)
        model = CausalKWS()
    items = tuple(
        sorted(model.named_parameters(), key=lambda item: item[0].encode("utf-8"))
    )
    return (
        tuple(name for name, _ in items),
        tuple(
            tensor.detach().clone(memory_format=torch.contiguous_format)
            for _, tensor in items
        ),
    )


def _synthetic_world(
    monkeypatch: pytest.MonkeyPatch,
    *,
    winner_epoch: int = _WINNER_EPOCH,
) -> _SyntheticWorld:
    assert 0 <= winner_epoch < training_evidence.EPOCH_COUNT
    names, tensors = _causal_kws_tensors()
    assert len(names) == training_evidence.MODEL_TENSOR_COUNT == 53
    assert sum(tensor.numel() for tensor in tensors) == 23_724
    model_payload = training_evidence._frame_model_tensors(names, tensors)
    model_sha256 = hashlib.sha256(
        training_evidence.MODEL_TENSOR_DOMAIN + model_payload
    ).hexdigest()

    history = object.__new__(RegisteredCompletedTrainingHistory)
    winner = object.__new__(RegisteredEvaluatedEpoch)
    evaluated_epochs = tuple(
        object.__new__(RegisteredEvaluatedEpoch)
        for _ in range(training_evidence.EPOCH_COUNT)
    )
    evaluated_epochs = (
        *evaluated_epochs[:winner_epoch],
        winner,
        *evaluated_epochs[winner_epoch + 1 :],
    )
    ranks = tuple(
        RegisteredEpochRank(
            zero_based_epoch=zero_based_epoch,
            macro_f1_exact_numerator=training_evidence.EPOCH_COUNT - rank,
            macro_f1_exact_denominator=training_evidence.EPOCH_COUNT,
            validation_cross_entropy_float64_hex="0x1.0000000000000p+0",
        )
        for rank, zero_based_epoch in enumerate(
            (
                winner_epoch,
                *(
                    epoch
                    for epoch in range(training_evidence.EPOCH_COUNT)
                    if epoch != winner_epoch
                ),
            )
        )
    )
    registration = cast(
        VerifiedRunRegistration,
        SimpleNamespace(
            head_commit="4" * 40,
            registration_sha256=_REGISTRATION_SHA256,
            source_bundle_sha256=_SOURCE_BUNDLE_SHA256,
        ),
    )
    model_tensors = cast(
        RegisteredModelTensorEvidence,
        SimpleNamespace(
            sha256=model_sha256,
            seed=_SEED,
            zero_based_epoch=winner_epoch,
            tensor_count=training_evidence.MODEL_TENSOR_COUNT,
            value_count=23_724,
        ),
    )
    validation_inputs = object.__new__(RegisteredValidationInputs)
    history_snapshot = registered_history._RegisteredCompletedHistorySnapshot(
        registration=registration,
        history=object.__new__(RegisteredTrainingHistory),
        executor=object.__new__(RegisteredTrainingExecutor),
        validation_inputs=validation_inputs,
        complete_update_trace=object.__new__(CompleteUpdateTraceEvidence),
        process_id=os.getpid(),
        seed=_SEED,
        epoch_count=training_evidence.EPOCH_COUNT,
        validation_inputs_sha256=_VALIDATION_INPUTS_SHA256,
        complete_update_trace_sha256=_COMPLETE_TRACE_SHA256,
        canonical_json_bytes=b'{"experiment":"experiment-002"}',
        history_sha256=_HISTORY_SHA256,
        ranked_epochs=ranks,
        evaluated_epochs=evaluated_epochs,
        winner=winner,
    )
    evaluated_snapshot = registered_evaluator._RegisteredEvaluatedEpochSnapshot(
        registration=registration,
        handoff=cast(Any, object()),
        validation_inputs=validation_inputs,
        validation_evidence=cast(Any, object()),
        model_tensors=model_tensors,
        process_id=os.getpid(),
        seed=_SEED,
        zero_based_epoch=winner_epoch,
        optimizer_generation=(winner_epoch + 1) * training_evidence.UPDATES_PER_EPOCH,
        batch_count=83,
        another_training_epoch=winner_epoch < training_evidence.EPOCH_COUNT - 1,
        registration_head_commit=registration.head_commit,
        registration_sha256=_REGISTRATION_SHA256,
        source_bundle_sha256=_SOURCE_BUNDLE_SHA256,
        validation_inputs_sha256=_VALIDATION_INPUTS_SHA256,
        validation_predictions_sha256=_VALIDATION_PREDICTIONS_SHA256,
        model_tensor_sha256=model_sha256,
        optimizer_sha256=_OPTIMIZER_SHA256,
        torch_rng_sha256=_TORCH_RNG_SHA256,
        authority_sha256=_EVALUATED_AUTHORITY_SHA256,
    )
    history_route_calls: list[RegisteredCompletedTrainingHistory] = []

    def history_sha256(_: RegisteredCompletedTrainingHistory) -> str:
        return _HISTORY_SHA256

    def history_seed(_: RegisteredCompletedTrainingHistory) -> int:
        return _SEED

    def completed_snapshot_route(
        candidate: RegisteredCompletedTrainingHistory,
    ) -> registered_history._RegisteredCompletedHistorySnapshot:
        assert candidate is history
        history_route_calls.append(candidate)
        return history_snapshot

    def evaluated_snapshot_route(
        candidate: RegisteredEvaluatedEpoch,
    ) -> registered_evaluator._RegisteredEvaluatedEpochSnapshot:
        assert candidate is winner
        return evaluated_snapshot

    def clone_route(
        evidence: RegisteredModelTensorEvidence,
    ) -> tuple[tuple[str, Tensor], ...]:
        assert evidence is model_tensors
        return tuple(
            (
                name,
                tensor.detach().clone(memory_format=torch.contiguous_format),
            )
            for name, tensor in zip(names, tensors, strict=True)
        )

    def verify_model_tensors(
        evidence: RegisteredModelTensorEvidence,
        *,
        seed: int | None = None,
        zero_based_epoch: int | None = None,
    ) -> None:
        assert evidence is model_tensors
        assert seed == _SEED
        assert zero_based_epoch == winner_epoch

    def reverify_registration(candidate: VerifiedRunRegistration) -> None:
        assert candidate is registration

    monkeypatch.setattr(
        RegisteredCompletedTrainingHistory,
        "history_sha256",
        property(history_sha256),
    )
    monkeypatch.setattr(
        RegisteredCompletedTrainingHistory,
        "seed",
        property(history_seed),
    )
    monkeypatch.setattr(
        checkpoint,
        "_registered_completed_history_snapshot",
        completed_snapshot_route,
    )
    monkeypatch.setattr(
        checkpoint,
        "_registered_evaluated_epoch_snapshot",
        evaluated_snapshot_route,
    )
    monkeypatch.setattr(checkpoint, "_model_tensor_clones", clone_route)
    monkeypatch.setattr(
        checkpoint,
        "verify_registered_model_tensor_evidence",
        verify_model_tensors,
    )
    monkeypatch.setattr(
        checkpoint,
        "reverify_verified_run_registration",
        reverify_registration,
    )
    return _SyntheticWorld(
        history=history,
        winner=winner,
        model_tensors=model_tensors,
        names=names,
        tensors=tensors,
        model_sha256=model_sha256,
        history_snapshot=history_snapshot,
        evaluated_snapshot=evaluated_snapshot,
        history_route_calls=history_route_calls,
    )


def _header_and_data(payload: bytes) -> tuple[dict[str, Any], bytes]:
    header_size = struct.unpack("<Q", payload[:8])[0]
    header_bytes = payload[8 : 8 + header_size].rstrip(b" ")
    return json.loads(header_bytes), payload[8 + header_size :]


def _pack_header(document: str, data: bytes) -> bytes:
    encoded = document.encode("utf-8")
    padded = encoded + b" " * (-len(encoded) % 8)
    return struct.pack("<Q", len(padded)) + padded + data


def _valid_payload_and_specs(
    world: _SyntheticWorld,
) -> tuple[bytes, tuple[checkpoint._TensorSpec, ...]]:
    specs = checkpoint._tensor_specs(world.names, world.tensors)
    payload = safetensors.torch.save(
        dict(zip(world.names, world.tensors, strict=True)),
        metadata=None,
    )
    checkpoint._require_safetensors_payload(payload, specs=specs)
    return payload, specs


def test_serializes_exact_causal_kws_population_and_preserves_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _synthetic_world(monkeypatch)
    real_save = safetensors.torch.save
    save_calls: list[int] = []

    def counted_save(
        tensors: dict[str, Tensor], metadata: dict[str, str] | None = None
    ) -> bytes:
        save_calls.append(len(tensors))
        return real_save(tensors, metadata=metadata)

    monkeypatch.setattr(safetensors.torch, "save", counted_save)
    rng_before = torch.get_rng_state().clone()

    result = checkpoint.serialize_registered_checkpoint(world.history)

    assert result.seed == _SEED
    assert result.zero_based_epoch == _WINNER_EPOCH
    assert result.history_sha256 == _HISTORY_SHA256
    assert result.model_tensor_sha256 == world.model_sha256
    assert result.safetensors_byte_count == 99_776
    assert (
        result.safetensors_sha256
        == hashlib.sha256(result.safetensors_bytes).hexdigest()
    )
    assert result.safetensors_bytes is result.safetensors_bytes
    assert save_calls == [53, 53]
    assert torch.equal(torch.get_rng_state(), rng_before)

    payload = result.safetensors_bytes
    header_size = struct.unpack("<Q", payload[:8])[0]
    header, data = _header_and_data(payload)
    assert header_size == 4_872
    assert len(data) == 94_896
    assert "__metadata__" not in header
    assert tuple(header) == world.names
    loaded = safetensors.torch.load(payload)
    assert tuple(sorted(loaded, key=lambda name: name.encode("utf-8"))) == world.names
    for name, expected in zip(world.names, world.tensors, strict=True):
        observed = loaded[name]
        assert tuple(observed.shape) == tuple(expected.shape)
        assert checkpoint._tensor_payload(observed) == checkpoint._tensor_payload(
            expected
        )
    checkpoint.verify_registered_serialized_checkpoint(result)
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_selects_nonfinal_rank_one_epoch_without_forcing_epoch_29(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    winner_epoch = 7
    world = _synthetic_world(monkeypatch, winner_epoch=winner_epoch)

    result = checkpoint.serialize_registered_checkpoint(world.history)

    assert world.history_snapshot.ranked_epochs[0].zero_based_epoch == winner_epoch
    assert result.zero_based_epoch == winner_epoch
    assert result.model_tensor_sha256 == world.model_sha256
    state = checkpoint._SERIALIZED[result]
    assert state.binding.winner is world.winner
    assert state.binding.optimizer_generation == (
        (winner_epoch + 1) * training_evidence.UPDATES_PER_EPOCH
    )
    assert state.binding.another_training_epoch is True
    checkpoint.verify_registered_serialized_checkpoint(result)


def test_concurrent_calls_coalesce_to_one_identity_and_two_saves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _synthetic_world(monkeypatch)
    real_begin = checkpoint._begin_serialization_admission
    real_save = safetensors.torch.save
    entrants = threading.Barrier(2)
    save_count = 0
    save_lock = threading.Lock()

    def synchronized_begin(
        history: RegisteredCompletedTrainingHistory, caller_token: object
    ) -> checkpoint._SerializationAdmission:
        admission = real_begin(history, caller_token)
        entrants.wait(timeout=10)
        return admission

    def counted_save(
        tensors: dict[str, Tensor], metadata: dict[str, str] | None = None
    ) -> bytes:
        nonlocal save_count
        with save_lock:
            save_count += 1
        return real_save(tensors, metadata=metadata)

    monkeypatch.setattr(
        checkpoint, "_begin_serialization_admission", synchronized_begin
    )
    monkeypatch.setattr(safetensors.torch, "save", counted_save)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(checkpoint.serialize_registered_checkpoint, world.history)
            for _ in range(2)
        ]
        left, right = (future.result(timeout=30) for future in futures)

    assert left is right
    assert save_count == 2
    monkeypatch.setattr(checkpoint, "_begin_serialization_admission", real_begin)
    late = checkpoint.serialize_registered_checkpoint(world.history)
    assert late is left
    assert save_count == 2


def test_serialization_failure_is_sticky_and_does_not_invalidate_upstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _synthetic_world(monkeypatch)
    failure = RuntimeError("synthetic serializer failure")
    save_count = 0

    def failing_save(
        tensors: dict[str, Tensor], metadata: dict[str, str] | None = None
    ) -> NoReturn:
        del tensors, metadata
        nonlocal save_count
        save_count += 1
        raise failure

    monkeypatch.setattr(safetensors.torch, "save", failing_save)
    with pytest.raises(RuntimeError) as first:
        checkpoint.serialize_registered_checkpoint(world.history)
    route_calls_after_failure = len(world.history_route_calls)
    with pytest.raises(RuntimeError) as second:
        checkpoint.serialize_registered_checkpoint(world.history)

    assert first.value is failure
    assert second.value is failure
    assert save_count == 1
    assert len(world.history_route_calls) == route_calls_after_failure
    snapshot_route = cast(
        Callable[
            [RegisteredCompletedTrainingHistory],
            registered_history._RegisteredCompletedHistorySnapshot,
        ],
        checkpoint._registered_completed_history_snapshot,  # type: ignore[attr-defined]
    )
    assert snapshot_route(world.history) is world.history_snapshot
    assert world.history.history_sha256 == _HISTORY_SHA256


@pytest.mark.parametrize("boundary", ("save", "load"))
def test_external_rng_consumption_is_restored_before_sticky_failure(
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    world = _synthetic_world(monkeypatch)
    rng_before = torch.get_rng_state().clone()
    real_load = safetensors.torch.load

    def consuming_save(
        tensors: dict[str, Tensor],
        metadata: dict[str, str] | None = None,
    ) -> NoReturn:
        del tensors, metadata
        torch.rand(1)
        raise RuntimeError("save consumed RNG")

    def consuming_load(payload: bytes) -> dict[str, Tensor]:
        loaded = real_load(payload)
        torch.rand(1)
        return loaded

    if boundary == "save":
        monkeypatch.setattr(safetensors.torch, "save", consuming_save)
    else:
        monkeypatch.setattr(safetensors.torch, "load", consuming_load)

    with pytest.raises(
        checkpoint.Experiment002RegisteredCheckpointError,
        match="Torch CPU RNG changed",
    ):
        checkpoint.serialize_registered_checkpoint(world.history)
    assert torch.equal(torch.get_rng_state(), rng_before)

    with pytest.raises(checkpoint.Experiment002RegisteredCheckpointError):
        checkpoint.serialize_registered_checkpoint(world.history)
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_capability_rejects_construction_copy_pickle_and_counterfeit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _synthetic_world(monkeypatch)
    issued = checkpoint.serialize_registered_checkpoint(world.history)
    counterfeit = object.__new__(checkpoint.RegisteredSerializedCheckpoint)

    with pytest.raises(TypeError):
        checkpoint.RegisteredSerializedCheckpoint()
    with pytest.raises(TypeError):
        checkpoint.serialize_registered_checkpoint(cast(Any, object()))
    with pytest.raises(checkpoint.Experiment002RegisteredCheckpointError):
        checkpoint.verify_registered_serialized_checkpoint(counterfeit)
    for operation in (
        lambda: copy.copy(issued),
        lambda: copy.deepcopy(issued),
        lambda: pickle.dumps(issued),
    ):
        with pytest.raises(TypeError):
            operation()


@pytest.mark.parametrize(
    "mutation",
    (
        "duplicate-key",
        "bool-offset",
        "gap",
        "trailing-data",
        "data-corruption",
        "metadata",
        "oversized-header",
    ),
)
def test_manual_parser_rejects_noncanonical_or_corrupt_payloads(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    world = _synthetic_world(monkeypatch)
    payload, specs = _valid_payload_and_specs(world)
    header, data = _header_and_data(payload)

    if mutation == "duplicate-key":
        items = list(header.items())
        duplicate_items = (items[0], items[0], *items[1:])
        document = (
            "{"
            + ",".join(
                f"{json.dumps(name)}:{json.dumps(record, separators=(',', ':'))}"
                for name, record in duplicate_items
            )
            + "}"
        )
        mutated = _pack_header(document, data)
    elif mutation == "bool-offset":
        first = cast(dict[str, Any], header[world.names[0]])
        first["data_offsets"][0] = False
        mutated = _pack_header(json.dumps(header, separators=(",", ":")), data)
    elif mutation == "gap":
        first = cast(dict[str, Any], header[world.names[0]])
        first["data_offsets"] = [1, first["data_offsets"][1] + 1]
        mutated = _pack_header(json.dumps(header, separators=(",", ":")), data)
    elif mutation == "trailing-data":
        mutated = payload + b"\0"
    elif mutation == "data-corruption":
        mutated = payload[:-1] + bytes((payload[-1] ^ 1,))
    elif mutation == "metadata":
        document = json.dumps({"__metadata__": {}, **header}, separators=(",", ":"))
        mutated = _pack_header(document, data)
    else:
        assert mutation == "oversized-header"
        mutated = struct.pack("<Q", checkpoint._MAX_HEADER_BYTES + 8)

    with pytest.raises(checkpoint.Experiment002RegisteredCheckpointError):
        checkpoint._require_safetensors_payload(mutated, specs=specs)


def test_registry_tamper_fails_terminally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _synthetic_world(monkeypatch)
    issued = checkpoint.serialize_registered_checkpoint(world.history)
    state = checkpoint._SERIALIZED[issued]
    copied_payload = memoryview(state.payload).tobytes()
    assert copied_payload == state.payload
    assert copied_payload is not state.payload
    checkpoint._SERIALIZED[issued] = replace(state, payload=copied_payload)

    with pytest.raises(
        checkpoint.Experiment002RegisteredCheckpointError,
        match="authority changed",
    ):
        checkpoint.verify_registered_serialized_checkpoint(issued)
    with pytest.raises(
        checkpoint.Experiment002RegisteredCheckpointError,
        match="terminally failed",
    ):
        checkpoint.verify_registered_serialized_checkpoint(issued)


def test_post_begin_admission_validation_interrupt_is_sticky_and_wakes_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _synthetic_world(monkeypatch)
    original_begin = checkpoint._begin_serialization_admission
    original_require = checkpoint._require_admission
    captured: list[checkpoint._SerializationAdmission] = []
    interruption = KeyboardInterrupt("after serialization begin")

    def begin(
        history: RegisteredCompletedTrainingHistory, caller_token: object
    ) -> checkpoint._SerializationAdmission:
        admission = original_begin(history, caller_token)
        captured.append(admission)
        return admission

    def interrupt(
        admission: checkpoint._SerializationAdmission,
        history: RegisteredCompletedTrainingHistory,
    ) -> NoReturn:
        del admission, history
        raise interruption

    monkeypatch.setattr(checkpoint, "_begin_serialization_admission", begin)
    monkeypatch.setattr(checkpoint, "_require_admission", interrupt)
    with pytest.raises(KeyboardInterrupt) as first:
        checkpoint.serialize_registered_checkpoint(world.history)

    assert first.value is interruption
    assert len(captured) == 1
    assert captured[0].completed.is_set()
    assert len(captured[0].outcome) == 1
    assert captured[0].outcome[0].error is interruption
    monkeypatch.setattr(checkpoint, "_begin_serialization_admission", original_begin)
    monkeypatch.setattr(checkpoint, "_require_admission", original_require)
    with pytest.raises(KeyboardInterrupt) as retry:
        checkpoint.serialize_registered_checkpoint(world.history)
    assert retry.value is interruption


def test_follower_cannot_publish_or_wake_the_active_leader() -> None:
    history = object.__new__(RegisteredCompletedTrainingHistory)
    leader = checkpoint._begin_serialization_admission(history, object())
    follower = checkpoint._begin_serialization_admission(history, object())
    assert leader.leader
    assert not follower.leader
    assert follower.completed is leader.completed
    assert follower.outcome is leader.outcome

    with pytest.raises(
        checkpoint.Experiment002RegisteredCheckpointError,
        match="only the serialization leader",
    ):
        checkpoint._finish_serialization_admission(
            follower,
            checkpoint._SerializationOutcome(None, RuntimeError("forged")),
        )
    assert not leader.completed.is_set()
    assert not leader.outcome

    expected = RuntimeError("leader outcome")
    checkpoint._finish_serialization_admission(
        leader,
        checkpoint._SerializationOutcome(None, expected),
    )
    assert leader.completed.is_set()
    assert leader.outcome[0].error is expected


def test_coherently_flipped_leader_flag_cannot_wait_on_its_own_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _synthetic_world(monkeypatch)
    original_begin = checkpoint._begin_serialization_admission
    captured: list[checkpoint._SerializationAdmission] = []

    def flipped_begin(
        history: RegisteredCompletedTrainingHistory,
        caller_token: object,
    ) -> checkpoint._SerializationAdmission:
        admission = original_begin(history, caller_token)
        captured.append(admission)
        object.__setattr__(admission, "leader", False)
        return admission

    monkeypatch.setattr(
        checkpoint,
        "_begin_serialization_admission",
        flipped_begin,
    )
    with pytest.raises(
        checkpoint.Experiment002RegisteredCheckpointError,
        match="terminally failed|closure truth changed|capability changed",
    ):
        checkpoint.serialize_registered_checkpoint(world.history)
    assert len(captured) == 1
    assert captured[0].completed.is_set()

    monkeypatch.setattr(
        checkpoint,
        "_begin_serialization_admission",
        original_begin,
    )
    with pytest.raises(checkpoint.Experiment002RegisteredCheckpointError):
        checkpoint.serialize_registered_checkpoint(world.history)


@pytest.mark.parametrize("consumer", ("leader", "follower"))
def test_terminal_consumer_cannot_accept_a_replaced_result_identity(
    consumer: str,
) -> None:
    history = object.__new__(RegisteredCompletedTrainingHistory)
    leader = checkpoint._begin_serialization_admission(history, object())
    expected = RuntimeError("expected terminal failure")
    checkpoint._finish_serialization_admission(
        leader,
        checkpoint._SerializationOutcome(None, expected),
    )
    admission = (
        leader
        if consumer == "leader"
        else checkpoint._begin_serialization_admission(history, object())
    )
    original = admission.outcome[0]
    foreign = object.__new__(checkpoint.RegisteredSerializedCheckpoint)
    admission.outcome[:] = [checkpoint._SerializationOutcome(foreign, None)]

    with pytest.raises(
        checkpoint.Experiment002RegisteredCheckpointError,
        match="truth changed",
    ):
        checkpoint._require_terminal_outcome(admission)

    admission.outcome[:] = [original]
    with pytest.raises(
        checkpoint.Experiment002RegisteredCheckpointError,
        match="terminally failed",
    ):
        checkpoint._begin_serialization_admission(history, object())


def test_publication_cannot_substitute_checkpoint_from_another_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    foreign_world = _synthetic_world(monkeypatch)
    foreign = checkpoint.serialize_registered_checkpoint(foreign_world.history)
    foreign_state = checkpoint._SERIALIZED[foreign]
    foreign_guard = checkpoint._SERIALIZED_GUARDS[foreign]

    target_world = _synthetic_world(monkeypatch)
    original_publish = checkpoint._publish_serialization_outcome

    def substitute(
        admission: checkpoint._SerializationAdmission,
        outcome: checkpoint._SerializationOutcome,
    ) -> None:
        replacement = (
            checkpoint._SerializationOutcome(foreign, None)
            if outcome.result is not None
            else outcome
        )
        original_publish(admission, replacement)

    monkeypatch.setattr(checkpoint, "_publish_serialization_outcome", substitute)
    with pytest.raises(
        checkpoint.Experiment002RegisteredCheckpointError,
        match="closure (authority|truth)|terminally failed",
    ):
        checkpoint.serialize_registered_checkpoint(target_world.history)
    assert checkpoint._validate_serialized_truth(
        foreign,
        foreign_state,
        foreign_guard,
    )


@pytest.mark.parametrize("boundary", ("verify", "rng"))
def test_post_issuance_interruption_terminalizes_captured_result(
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    world = _synthetic_world(monkeypatch)
    original_issue = checkpoint._issue_serialized_checkpoint
    original_verify = checkpoint._verify_serialized_state
    original_require_rng = checkpoint._require_rng_unchanged
    captured: list[checkpoint.RegisteredSerializedCheckpoint] = []
    interruption = KeyboardInterrupt(f"post-issuance {boundary} interruption")

    def issue(
        *,
        binding: checkpoint._SourceBinding,
        specs: tuple[checkpoint._TensorSpec, ...],
        model_payload: bytes,
        payload: bytes,
    ) -> checkpoint.RegisteredSerializedCheckpoint:
        result = original_issue(
            binding=binding,
            specs=specs,
            model_payload=model_payload,
            payload=payload,
        )
        captured.append(result)
        return result

    def interrupt_verify(
        result: checkpoint.RegisteredSerializedCheckpoint,
        *,
        reverify_source: bool,
    ) -> NoReturn:
        del result, reverify_source
        raise interruption

    def interrupt_rng(expected: bytes) -> None:
        if captured:
            raise interruption
        original_require_rng(expected)

    monkeypatch.setattr(checkpoint, "_issue_serialized_checkpoint", issue)
    if boundary == "verify":
        monkeypatch.setattr(checkpoint, "_verify_serialized_state", interrupt_verify)
    else:
        monkeypatch.setattr(checkpoint, "_require_rng_unchanged", interrupt_rng)

    with pytest.raises(KeyboardInterrupt) as observed:
        checkpoint.serialize_registered_checkpoint(world.history)
    assert observed.value is interruption
    assert len(captured) == 1

    monkeypatch.setattr(checkpoint, "_verify_serialized_state", original_verify)
    monkeypatch.setattr(checkpoint, "_require_rng_unchanged", original_require_rng)
    with pytest.raises(
        checkpoint.Experiment002RegisteredCheckpointError,
        match="terminally failed",
    ):
        checkpoint.verify_registered_serialized_checkpoint(captured[0])


@pytest.mark.parametrize("target", ("event", "outcome"))
def test_retained_admission_tamper_restore_is_irreversible(
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    world = _synthetic_world(monkeypatch)
    original_begin = checkpoint._begin_serialization_admission
    captured: list[checkpoint._SerializationAdmission] = []

    def begin(
        history: RegisteredCompletedTrainingHistory, caller_token: object
    ) -> checkpoint._SerializationAdmission:
        admission = original_begin(history, caller_token)
        captured.append(admission)
        return admission

    monkeypatch.setattr(checkpoint, "_begin_serialization_admission", begin)
    checkpoint.serialize_registered_checkpoint(world.history)
    assert len(captured) == 1
    retained = captured[0]
    assert retained.completed.is_set()
    assert len(retained.outcome) == 1
    if target == "event":
        retained.completed.clear()
    else:
        original = retained.outcome[0]
        retained.outcome.clear()

    monkeypatch.setattr(checkpoint, "_begin_serialization_admission", original_begin)
    with pytest.raises(
        checkpoint.Experiment002RegisteredCheckpointError,
        match="truth changed",
    ):
        checkpoint.serialize_registered_checkpoint(world.history)
    if target == "event":
        retained.completed.set()
    else:
        retained.outcome.append(original)
    with pytest.raises(
        checkpoint.Experiment002RegisteredCheckpointError,
        match="terminally failed",
    ):
        checkpoint.serialize_registered_checkpoint(world.history)


def test_coherent_serialized_registry_clone_cannot_mint_issuance_truth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _synthetic_world(monkeypatch)
    original = checkpoint.serialize_registered_checkpoint(world.history)
    original_state = checkpoint._SERIALIZED[original]
    original_guard = checkpoint._SERIALIZED_GUARDS[original]
    clone = object.__new__(checkpoint.RegisteredSerializedCheckpoint)
    token = object()
    cloned_state = replace(original_state, token=token)
    cloned_guard = replace(original_guard, token=token)
    for name, value in (
        ("_seed", cloned_state.binding.seed),
        ("_zero_based_epoch", cloned_state.binding.zero_based_epoch),
        ("_history_sha256", cloned_state.binding.history_sha256),
        ("_model_tensor_sha256", cloned_state.binding.model_tensor_sha256),
        ("_safetensors_sha256", cloned_state.safetensors_sha256),
        ("_safetensors_byte_count", len(cloned_state.payload)),
        ("_token", token),
        ("_route_marker", checkpoint._ROUTE_MARKER),
    ):
        object.__setattr__(clone, name, value)
    with checkpoint._REGISTRY_LOCK:
        checkpoint._SERIALIZED[clone] = cloned_state
        checkpoint._SERIALIZED_GUARDS[clone] = cloned_guard
        checkpoint._ISSUED_SERIALIZED.add(clone)
    try:
        with pytest.raises(checkpoint.Experiment002RegisteredCheckpointError):
            checkpoint.verify_registered_serialized_checkpoint(clone)
    finally:
        with checkpoint._REGISTRY_LOCK:
            checkpoint._SERIALIZED.pop(clone, None)
            checkpoint._SERIALIZED_GUARDS.pop(clone, None)
            checkpoint._ISSUED_SERIALIZED.discard(clone)
            checkpoint._FAILED_SERIALIZED.discard(clone)


def test_publication_entry_interruption_uses_captured_fallback_truth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _synthetic_world(monkeypatch)
    original_finish = checkpoint._finish_serialization_admission
    original_issue = checkpoint._issue_serialized_checkpoint
    captured: list[checkpoint.RegisteredSerializedCheckpoint] = []
    interruption = KeyboardInterrupt("publication entry interruption")

    def issue(
        *,
        binding: checkpoint._SourceBinding,
        specs: tuple[checkpoint._TensorSpec, ...],
        model_payload: bytes,
        payload: bytes,
    ) -> checkpoint.RegisteredSerializedCheckpoint:
        result = original_issue(
            binding=binding,
            specs=specs,
            model_payload=model_payload,
            payload=payload,
        )
        captured.append(result)
        return result

    def interrupt_finish(
        admission: checkpoint._SerializationAdmission,
        outcome: checkpoint._SerializationOutcome,
    ) -> NoReturn:
        del admission, outcome
        raise interruption

    monkeypatch.setattr(checkpoint, "_issue_serialized_checkpoint", issue)
    monkeypatch.setattr(checkpoint, "_finish_serialization_admission", interrupt_finish)
    with pytest.raises(KeyboardInterrupt) as observed:
        checkpoint.serialize_registered_checkpoint(world.history)
    assert observed.value is interruption
    assert len(captured) == 1

    monkeypatch.setattr(checkpoint, "_finish_serialization_admission", original_finish)
    assert checkpoint.serialize_registered_checkpoint(world.history) is captured[0]


def test_module_has_no_filesystem_checkpoint_route() -> None:
    tree = ast.parse(inspect.getsource(checkpoint))
    imported_modules: set[str] = set()
    forbidden_calls: set[str] = set()
    forbidden_names = {"open", "save_file", "load_file", "safe_open"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(
                alias.name.split(".", maxsplit=1)[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module.split(".", maxsplit=1)[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in forbidden_names:
                forbidden_calls.add(node.func.id)
            elif (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in forbidden_names
            ):
                forbidden_calls.add(node.func.attr)

    assert imported_modules.isdisjoint({"pathlib", "tempfile"})
    assert not forbidden_calls
