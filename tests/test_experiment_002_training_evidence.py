from __future__ import annotations

import copy
import hashlib
import json
import math
import pickle
import struct
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from torch import nn

import falsewake.experiment_002_training_bridge as training_bridge
import falsewake.experiment_002_training_evidence as evidence
import falsewake.experiment_002_training_population as training_population
from falsewake.causal_kws import CausalKWS
from falsewake.experiment_002_training_evidence import (
    CompleteUpdateTraceEvidence,
    EpochUpdateTraceEvidence,
    Experiment002TrainingEvidenceError,
    RegisteredModelTensorEvidence,
)


class _OrderedTinyModel(nn.Module):
    z_tensor: nn.Parameter
    a_tensor: nn.Parameter

    def __init__(self, *, signed_zero: float = 0.0) -> None:
        super().__init__()
        self.z_tensor = nn.Parameter(
            torch.tensor([signed_zero, 2.5], dtype=torch.float32)
        )
        self.a_tensor = nn.Parameter(torch.tensor([-1.25], dtype=torch.float32))


class _StringSubclass(str):
    pass


def _tiny_model_layout() -> evidence._ModelTensorLayout:
    return evidence._ModelTensorLayout(tensor_count=2, value_count=3)


def _golden_model_frame(model: _OrderedTinyModel) -> bytes:
    framed = bytearray(struct.pack("<I", 2))
    for name in ("a_tensor", "z_tensor"):
        tensor = model.state_dict()[name]
        name_bytes = name.encode("utf-8")
        payload = tensor.numpy().astype("<f4", copy=False).tobytes(order="C")
        framed.extend(struct.pack("<I", len(name_bytes)))
        framed.extend(name_bytes)
        framed.extend(struct.pack("<I", 5))
        framed.extend(b"F32LE")
        framed.extend(struct.pack("<I", tensor.ndim))
        for dimension in tensor.shape:
            framed.extend(struct.pack("<Q", dimension))
        framed.extend(struct.pack("<Q", len(payload)))
        framed.extend(payload)
    return bytes(framed)


def _tiny_trace_layout() -> evidence._TraceLayout:
    return evidence._TraceLayout(
        epochs=2,
        updates_per_epoch=3,
        population_count=10,
        batch_size=4,
        last_batch_size=2,
        warmup_updates=2,
        warmup_start_lr=0.1,
        maximum_lr=0.3,
        minimum_lr=0.01,
    )


def _golden_learning_rate(global_update: int, layout: evidence._TraceLayout) -> float:
    if global_update < layout.warmup_updates:
        return layout.warmup_start_lr + (
            (layout.maximum_lr - layout.warmup_start_lr)
            * global_update
            / (layout.warmup_updates - 1)
        )
    return layout.minimum_lr + 0.5 * (layout.maximum_lr - layout.minimum_lr) * (
        1.0
        + math.cos(
            math.pi
            * (global_update - layout.warmup_updates)
            / (layout.total_updates - 1 - layout.warmup_updates)
        )
    )


def _batch_size(global_update: int, layout: evidence._TraceLayout) -> int:
    if global_update % layout.updates_per_epoch == layout.updates_per_epoch - 1:
        return layout.last_batch_size
    return layout.batch_size


def _record(
    accumulator: evidence.UpdateTraceAccumulator,
    global_update: int,
    layout: evidence._TraceLayout,
    *,
    loss: np.float32 | None = None,
    norm: np.float32 | None = None,
) -> None:
    accumulator.record_update(
        global_update,
        _batch_size(global_update, layout),
        _golden_learning_rate(global_update, layout),
        np.float32(global_update + 0.25) if loss is None else loss,
        np.float32(global_update + 1.5) if norm is None else norm,
    )


def _finish_tiny_trace(
    seed: int = 7,
) -> tuple[
    evidence.UpdateTraceAccumulator,
    tuple[EpochUpdateTraceEvidence, ...],
    CompleteUpdateTraceEvidence,
]:
    layout = _tiny_trace_layout()
    accumulator = evidence.UpdateTraceAccumulator._for_layout(seed, layout)
    epochs: list[EpochUpdateTraceEvidence] = []
    for epoch in range(layout.epochs):
        first = epoch * layout.updates_per_epoch
        for global_update in range(first, first + layout.updates_per_epoch):
            _record(accumulator, global_update, layout)
        epochs.append(accumulator.finish_epoch())
    return accumulator, tuple(epochs), accumulator.finish()


def _registered_receipt_for_trace(
    *,
    transition: training_bridge.RegisteredOptimizerTransition,
    epoch_token: object,
    batch_token: object,
    receipt_token: object,
    transition_token: object,
    global_update: int = 0,
    batch_size: int = evidence.TRAINING_BATCH_SIZE,
    learning_rate: float | None = None,
    loss: np.float32 | None = None,
    norm: np.float32 | None = None,
) -> training_population.CompletedTrainingUpdate:
    if learning_rate is None:
        learning_rate = evidence.registered_learning_rate(global_update)
    if loss is None:
        loss = np.float32(0.75)
    if norm is None:
        norm = np.float32(1.25)
    snapshot = training_population._CompletedTrainingUpdateSnapshot(
        session_token=epoch_token,
        batch_token=batch_token,
        receipt_token=receipt_token,
        optimizer_transition_token=transition_token,
        seed=evidence.REGISTERED_SEEDS[0],
        zero_based_epoch=global_update // evidence.UPDATES_PER_EPOCH,
        batch_index=global_update % evidence.UPDATES_PER_EPOCH,
        zero_based_global_update=global_update,
        batch_size=batch_size,
        learning_rate=learning_rate,
        learning_rate_bytes=struct.pack("<d", learning_rate),
        batch_mean_training_loss=loss,
        batch_mean_training_loss_bytes=struct.pack("<f", loss),
        returned_preclip_l2_norm=norm,
        returned_preclip_l2_norm_bytes=struct.pack("<f", norm),
    )
    receipt = object.__new__(training_population.CompletedTrainingUpdate)
    state = training_population._IssuedReceiptState(
        snapshot=snapshot,
        authority_payload=training_population._frame_update_snapshot(snapshot),
        session_token=epoch_token,
        batch_token=batch_token,
        receipt_token=receipt_token,
        optimizer_transition_token=transition_token,
        optimizer_transition=transition,
    )
    with training_population._ISSUED_RECEIPTS_LOCK:
        training_population._ISSUED_RECEIPTS[receipt] = state
        training_population._RECEIPT_GUARDS[receipt] = (
            training_population._ReceiptIssuanceGuard(
                authority_payload=state.authority_payload,
                session_token=state.session_token,
                batch_token=state.batch_token,
                receipt_token=state.receipt_token,
                optimizer_transition_token=state.optimizer_transition_token,
                optimizer_transition=state.optimizer_transition,
            )
        )
    return receipt


def _registered_transition_snapshot_for_trace(
    *,
    executor_token: object,
    epoch_token: object,
    batch_token: object,
    transition_token: object,
    global_update: int = 0,
    batch_size: int = evidence.TRAINING_BATCH_SIZE,
    learning_rate: float | None = None,
    loss: np.float32 | None = None,
    norm: np.float32 | None = None,
) -> training_bridge._OptimizerTransitionSnapshot:
    if learning_rate is None:
        learning_rate = evidence.registered_learning_rate(global_update)
    if loss is None:
        loss = np.float32(0.75)
    if norm is None:
        norm = np.float32(1.25)
    digest = "0" * 64
    return training_bridge._OptimizerTransitionSnapshot(
        executor_session_token=executor_token,
        epoch_session_token=epoch_token,
        batch_token=batch_token,
        transition_token=transition_token,
        seed=evidence.REGISTERED_SEEDS[0],
        zero_based_epoch=global_update // evidence.UPDATES_PER_EPOCH,
        zero_based_global_update=global_update,
        batch_size=batch_size,
        learning_rate=learning_rate,
        learning_rate_bytes=struct.pack("<d", learning_rate),
        batch_mean_training_loss=loss,
        batch_mean_training_loss_bytes=struct.pack("<f", loss),
        returned_preclip_l2_norm=norm,
        returned_preclip_l2_norm_bytes=struct.pack("<f", norm),
        optimizer_generation_before=global_update,
        optimizer_generation_after=global_update + 1,
        model_sha256_before=digest,
        model_sha256_after=digest,
        optimizer_sha256_before=digest,
        optimizer_sha256_after=digest,
        rng_sha256_before=digest,
        rng_sha256_after=digest,
    )


def test_frozen_constants_and_formulas_reconcile_with_trainer_registration() -> None:
    config = json.loads(
        Path("configs/experiment-002-trainer.json").read_text(encoding="utf-8")
    )
    training = config["training_execution"]
    collation = config["collation"]["training"]

    assert tuple(training["seed_order"]) == evidence.REGISTERED_SEEDS
    assert evidence.EPOCH_COUNT == training["epochs"] == 30
    assert evidence.UPDATES_PER_EPOCH == training["updates_per_epoch"] == 313
    assert evidence.TOTAL_UPDATE_COUNT == training["total_updates_per_seed"] == 9_390
    assert training["population_count_per_epoch"] == evidence.TRAINING_EXAMPLE_COUNT
    assert evidence.TRAINING_BATCH_SIZE == collation["batch_size"] == 128
    assert evidence.LAST_BATCH_SIZE == collation["last_batch_size"] == 91
    assert (
        config["bindings"]["model"]["parameter_tensor_count"]
        == evidence.MODEL_TENSOR_COUNT
    )
    assert (
        config["bindings"]["model"]["parameter_value_count"]
        == evidence.MODEL_PARAMETER_VALUE_COUNT
    )

    assert evidence.registered_learning_rate(0) == 0.0003
    assert evidence.registered_learning_rate(312) == 0.003
    expected_cosine_start = 0.00003 + 0.5 * (0.003 - 0.00003) * (
        1 + math.cos(math.pi * (313 - 313) / (9_389 - 313))
    )
    expected_cosine_end = 0.00003 + 0.5 * (0.003 - 0.00003) * (
        1 + math.cos(math.pi * (9_389 - 313) / (9_389 - 313))
    )
    assert struct.pack("<d", evidence.registered_learning_rate(313)) == struct.pack(
        "<d", expected_cosine_start
    )
    assert struct.pack("<d", evidence.registered_learning_rate(9_389)) == struct.pack(
        "<d", expected_cosine_end
    )
    with pytest.raises(Experiment002TrainingEvidenceError):
        evidence.registered_learning_rate(9_390)
    with pytest.raises(TypeError):
        evidence.registered_learning_rate(cast(int, True))


def test_model_tensor_digest_has_an_independent_struct_golden() -> None:
    model = _OrderedTinyModel()
    observed = evidence._capture_model_tensors(
        model,
        layout=_tiny_model_layout(),
        seed=7,
        zero_based_epoch=2,
    )
    golden_payload = _golden_model_frame(model)
    expected = hashlib.sha256(
        b"falsewake-exp002-model-tensors-v1\0" + golden_payload
    ).hexdigest()

    assert observed.sha256 == expected
    assert observed.seed == 7
    assert observed.zero_based_epoch == 2
    assert observed.tensor_count == 2
    assert observed.value_count == 3
    state = evidence._issued_model_tensor_state(observed)
    assert state.names == ("a_tensor", "z_tensor")
    assert state.framed_payload == golden_payload
    for tensor in state.tensors:
        assert tensor.device.type == "cpu"
        assert tensor.dtype == torch.float32
        assert tensor.is_contiguous()
        assert not tensor.requires_grad


def test_model_digest_preserves_signed_zero_and_utf8_byte_order() -> None:
    positive = _OrderedTinyModel(signed_zero=0.0)
    negative = _OrderedTinyModel(signed_zero=-0.0)
    positive_evidence = evidence._capture_model_tensors(
        positive, layout=_tiny_model_layout()
    )
    negative_evidence = evidence._capture_model_tensors(
        negative, layout=_tiny_model_layout()
    )

    assert positive_evidence.sha256 != negative_evidence.sha256
    assert evidence._issued_model_tensor_state(positive_evidence).names == (
        "a_tensor",
        "z_tensor",
    )


def test_model_digest_frames_shape_even_when_the_value_count_is_unchanged() -> None:
    vector = nn.Module()
    vector.register_parameter(
        "weight", nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float32))
    )
    matrix = nn.Module()
    matrix.register_parameter(
        "weight", nn.Parameter(torch.tensor([[1.0, 2.0]], dtype=torch.float32))
    )
    layout = evidence._ModelTensorLayout(1, 2)

    vector_evidence = evidence._capture_model_tensors(vector, layout=layout)
    matrix_evidence = evidence._capture_model_tensors(matrix, layout=layout)
    assert vector_evidence.sha256 != matrix_evidence.sha256


def test_model_evidence_owns_clones_and_detects_current_model_mutation() -> None:
    model = _OrderedTinyModel()
    issued = evidence._capture_model_tensors(
        model,
        layout=_tiny_model_layout(),
        seed=7,
        zero_based_epoch=2,
    )
    original_digest = issued.sha256
    clones = evidence._model_tensor_clones(issued)
    with torch.no_grad():
        model.z_tensor[0] = 99.0
        clones[0][1].zero_()

    assert issued.sha256 == original_digest
    with pytest.raises(Experiment002TrainingEvidenceError, match="current model"):
        evidence._verify_model_matches_evidence(model, issued)


def test_model_evidence_context_prevents_cross_seed_and_cross_epoch_reuse() -> None:
    issued = evidence._capture_model_tensors(
        _OrderedTinyModel(),
        layout=_tiny_model_layout(),
        seed=7,
        zero_based_epoch=2,
    )
    evidence._verify_model_tensor_context(issued, 7, 2)
    with pytest.raises(Experiment002TrainingEvidenceError, match="different"):
        evidence._verify_model_tensor_context(issued, 8, 2)
    with pytest.raises(Experiment002TrainingEvidenceError, match="different"):
        evidence._verify_model_tensor_context(issued, 7, 3)

    with pytest.raises(Experiment002TrainingEvidenceError, match="together"):
        evidence._capture_model_tensors(
            _OrderedTinyModel(),
            layout=_tiny_model_layout(),
            seed=7,
        )


def test_model_capture_detects_a_mutation_between_its_two_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _OrderedTinyModel()
    original = evidence._snapshot_model_parameters
    calls = 0

    def snapshot(
        candidate: nn.Module, layout: evidence._ModelTensorLayout
    ) -> tuple[tuple[str, ...], tuple[torch.Tensor, ...]]:
        nonlocal calls
        result = original(candidate, layout)
        calls += 1
        if calls == 1:
            with torch.no_grad():
                model.z_tensor[0] = 3.0
        return result

    monkeypatch.setattr(evidence, "_snapshot_model_parameters", snapshot)
    with pytest.raises(Experiment002TrainingEvidenceError, match="changed during"):
        evidence._capture_model_tensors(model, layout=_tiny_model_layout())
    assert calls == 2


@pytest.mark.parametrize("kind", ["dtype", "nonfinite", "noncontiguous", "buffer"])
def test_model_contract_rejects_invalid_state(kind: str) -> None:
    model: nn.Module
    if kind == "dtype":
        model = nn.Linear(2, 1, bias=False, dtype=torch.float64)
        layout = evidence._ModelTensorLayout(1, 2)
    elif kind == "nonfinite":
        linear = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            linear.weight[0, 0] = float("nan")
        model = linear
        layout = evidence._ModelTensorLayout(1, 2)
    elif kind == "noncontiguous":
        model = nn.Module()
        model.register_parameter(
            "weight", nn.Parameter(torch.zeros((2, 3), dtype=torch.float32).t())
        )
        layout = evidence._ModelTensorLayout(1, 6)
    else:
        model = nn.Linear(2, 1, bias=False)
        model.register_buffer("extra", torch.zeros(1, dtype=torch.float32))
        layout = evidence._ModelTensorLayout(1, 2)

    with pytest.raises(Experiment002TrainingEvidenceError):
        evidence._capture_model_tensors(model, layout=layout)


def test_model_layout_counts_and_exact_public_route_are_enforced_without_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _OrderedTinyModel()
    with pytest.raises(Experiment002TrainingEvidenceError, match="tensor count"):
        evidence._capture_model_tensors(model, layout=evidence._ModelTensorLayout(1, 3))
    with pytest.raises(Experiment002TrainingEvidenceError, match="value count"):
        evidence._capture_model_tensors(model, layout=evidence._ModelTensorLayout(2, 4))

    exact_uninitialized_model = object.__new__(CausalKWS)
    sentinel = cast(RegisteredModelTensorEvidence, object())
    calls = 0

    def capture(
        candidate: nn.Module,
        *,
        layout: evidence._ModelTensorLayout,
        seed: int | None = None,
        zero_based_epoch: int | None = None,
        route_marker: object | None = None,
    ) -> RegisteredModelTensorEvidence:
        nonlocal calls
        calls += 1
        assert candidate is exact_uninitialized_model
        assert layout is evidence._REGISTERED_MODEL_LAYOUT
        assert seed == evidence.REGISTERED_SEEDS[0]
        assert zero_based_epoch == 29
        assert route_marker is evidence._MODEL_TENSOR_ROUTE_MARKER
        return sentinel

    monkeypatch.setattr(evidence, "_capture_model_tensors", capture)
    assert (
        evidence.capture_registered_model_tensors(
            exact_uninitialized_model,
            seed=evidence.REGISTERED_SEEDS[0],
            zero_based_epoch=29,
        )
        is sentinel
    )
    assert calls == 1
    with pytest.raises(TypeError, match="exact CausalKWS"):
        evidence.capture_registered_model_tensors(
            cast(CausalKWS, model),
            seed=evidence.REGISTERED_SEEDS[0],
            zero_based_epoch=0,
        )
    with pytest.raises(Experiment002TrainingEvidenceError, match="registered"):
        evidence.capture_registered_model_tensors(
            exact_uninitialized_model,
            seed=7,
            zero_based_epoch=0,
        )
    with pytest.raises(Experiment002TrainingEvidenceError, match="outside"):
        evidence.capture_registered_model_tensors(
            exact_uninitialized_model,
            seed=evidence.REGISTERED_SEEDS[0],
            zero_based_epoch=30,
        )


def test_model_capability_rejects_copy_pickle_forge_and_slot_mutation() -> None:
    model = _OrderedTinyModel()
    issued = evidence._capture_model_tensors(model, layout=_tiny_model_layout())
    with pytest.raises(TypeError):
        copy.copy(issued)
    with pytest.raises(TypeError):
        copy.deepcopy(issued)
    with pytest.raises(TypeError):
        pickle.dumps(issued)

    forged = object.__new__(RegisteredModelTensorEvidence)
    with pytest.raises(Experiment002TrainingEvidenceError, match="not issued"):
        _ = forged.sha256

    object.__setattr__(issued, "_sha256", "0" * 64)
    with pytest.raises(Experiment002TrainingEvidenceError, match="changed"):
        _ = issued.sha256


def test_model_capability_revalidates_internal_snapshot_tensor_types() -> None:
    issued = evidence._capture_model_tensors(
        _OrderedTinyModel(), layout=_tiny_model_layout()
    )
    state = evidence._issued_model_tensor_state(issued)
    changed_tensors = (state.tensors[0].to(torch.float64), *state.tensors[1:])
    object.__setattr__(state, "tensors", changed_tensors)

    with pytest.raises(Experiment002TrainingEvidenceError, match="float32"):
        _ = issued.sha256


@pytest.mark.parametrize(
    "field",
    ["seed", "zero_based_epoch", "names", "tensors", "framed_payload", "sha256"],
)
def test_model_capability_revalidates_every_internal_state_field_type(
    field: str,
) -> None:
    issued = evidence._capture_model_tensors(
        _OrderedTinyModel(),
        layout=_tiny_model_layout(),
        seed=7,
        zero_based_epoch=2,
    )
    state = evidence._issued_model_tensor_state(issued)
    assert state.seed is not None
    assert state.zero_based_epoch is not None
    replacements: dict[str, object] = {
        "seed": np.int64(state.seed),
        "zero_based_epoch": np.int64(state.zero_based_epoch),
        "names": list(state.names),
        "tensors": list(state.tensors),
        "framed_payload": bytearray(state.framed_payload),
        "sha256": _StringSubclass(state.sha256),
    }
    object.__setattr__(state, field, replacements[field])

    with pytest.raises((TypeError, Experiment002TrainingEvidenceError)):
        _ = issued.sha256


def test_model_capability_revalidates_nested_layout_field_types() -> None:
    issued = evidence._capture_model_tensors(
        _OrderedTinyModel(), layout=_tiny_model_layout()
    )
    state = evidence._issued_model_tensor_state(issued)
    object.__setattr__(state.layout, "tensor_count", np.int64(2))

    with pytest.raises(TypeError, match="tensor_count"):
        _ = issued.sha256


def test_trace_record_and_epoch_digest_have_independent_struct_goldens() -> None:
    seed = 7
    layout = _tiny_trace_layout()
    accumulator = evidence.UpdateTraceAccumulator._for_layout(seed, layout)
    records: list[bytes] = []
    losses: list[np.float32] = []
    for global_update in range(layout.updates_per_epoch):
        loss = np.float32(global_update + 0.25)
        norm = np.float32(global_update + 1.5)
        losses.append(loss)
        _record(accumulator, global_update, layout, loss=loss, norm=norm)
        records.append(
            struct.pack(
                "<IIdff",
                global_update,
                _batch_size(global_update, layout),
                _golden_learning_rate(global_update, layout),
                loss,
                norm,
            )
        )
    epoch = accumulator.finish_epoch()
    golden_payload = struct.pack("<III", seed, 0, 3) + b"".join(records)
    expected_digest = hashlib.sha256(
        b"falsewake-exp002-epoch-update-trace-v1\0" + golden_payload
    ).hexdigest()
    expected_ce = np.float64(0.0)
    for index, loss in enumerate(losses):
        expected_ce = np.float64(
            expected_ce + np.float64(loss) * np.float64(_batch_size(index, layout))
        )
    expected_ce = np.float64(expected_ce / np.float64(layout.population_count))

    assert struct.calcsize("<IIdff") == 24
    assert epoch.seed == seed
    assert epoch.zero_based_epoch == 0
    assert epoch.first_global_update == 0
    assert epoch.last_global_update_inclusive == 2
    assert epoch.update_count == 3
    assert epoch.sha256 == expected_digest
    assert epoch.training_cross_entropy.tobytes() == expected_ce.tobytes()
    assert epoch.training_cross_entropy_hex == float(expected_ce).hex()
    assert evidence._issued_epoch_trace_state(epoch).framed_payload == golden_payload


def test_complete_trace_digest_has_an_independent_struct_golden() -> None:
    seed = 11
    _, epochs, complete = _finish_tiny_trace(seed)
    layout = _tiny_trace_layout()
    record_bytes = []
    for global_update in range(layout.total_updates):
        record_bytes.append(
            struct.pack(
                "<IIdff",
                global_update,
                _batch_size(global_update, layout),
                _golden_learning_rate(global_update, layout),
                np.float32(global_update + 0.25),
                np.float32(global_update + 1.5),
            )
        )
    golden_payload = struct.pack("<II", seed, layout.total_updates) + b"".join(
        record_bytes
    )
    expected_digest = hashlib.sha256(
        b"falsewake-exp002-update-trace-v1\0" + golden_payload
    ).hexdigest()

    assert complete.seed == seed
    assert complete.update_count == layout.total_updates
    assert complete.sha256 == expected_digest
    assert complete.epochs == epochs
    assert complete.epoch_update_trace_digests == tuple(
        epoch.sha256 for epoch in epochs
    )
    assert evidence._issued_complete_trace_state(complete).framed_payload == (
        golden_payload
    )


def test_trace_digest_binds_seed_without_changing_update_records() -> None:
    _, first_epochs, first = _finish_tiny_trace(seed=7)
    _, second_epochs, second = _finish_tiny_trace(seed=8)

    assert first.sha256 != second.sha256
    assert first_epochs[0].sha256 != second_epochs[0].sha256


def test_trace_digest_preserves_signed_zero_in_the_same_float32_loss_field() -> None:
    layout = _tiny_trace_layout()
    digests: list[str] = []
    cross_entropies: list[bytes] = []
    for signed_zero in (np.float32(0.0), np.float32(-0.0)):
        accumulator = evidence.UpdateTraceAccumulator._for_layout(9, layout)
        for global_update in range(layout.updates_per_epoch):
            _record(
                accumulator,
                global_update,
                layout,
                loss=signed_zero if global_update == 0 else np.float32(1.0),
            )
        epoch = accumulator.finish_epoch()
        digests.append(epoch.sha256)
        cross_entropies.append(epoch.training_cross_entropy.tobytes())

    assert digests[0] != digests[1]
    assert cross_entropies[0] == cross_entropies[1]


def test_trace_rejects_wrong_order_learning_rate_and_batch_sizes() -> None:
    layout = _tiny_trace_layout()
    accumulator = evidence.UpdateTraceAccumulator._for_layout(7, layout)
    with pytest.raises(Experiment002TrainingEvidenceError, match="global_update"):
        _record(accumulator, 1, layout)

    expected_lr = _golden_learning_rate(0, layout)
    with pytest.raises(Experiment002TrainingEvidenceError, match="learning_rate"):
        accumulator.record_update(
            0,
            4,
            math.nextafter(expected_lr, math.inf),
            np.float32(1.0),
            np.float32(2.0),
        )
    with pytest.raises(TypeError, match="Python float"):
        accumulator.record_update(
            0,
            4,
            cast(float, np.float64(expected_lr)),
            np.float32(1.0),
            np.float32(2.0),
        )
    _record(accumulator, 0, layout)
    with pytest.raises(Experiment002TrainingEvidenceError, match="global_update"):
        _record(accumulator, 0, layout)
    _record(accumulator, 1, layout)
    with pytest.raises(Experiment002TrainingEvidenceError, match="batch_size"):
        accumulator.record_update(
            2,
            4,
            _golden_learning_rate(2, layout),
            np.float32(1.0),
            np.float32(2.0),
        )


@pytest.mark.parametrize(
    ("loss", "norm", "error"),
    [
        (cast(np.float32, 1.0), np.float32(1.0), TypeError),
        (np.float64(1.0), np.float32(1.0), TypeError),
        (np.float32(float("nan")), np.float32(1.0), Experiment002TrainingEvidenceError),
        (np.float32(1.0), np.float32(float("inf")), Experiment002TrainingEvidenceError),
    ],
)
def test_trace_rejects_wrong_scalar_types_and_nonfinite_values(
    loss: object, norm: object, error: type[Exception]
) -> None:
    layout = _tiny_trace_layout()
    accumulator = evidence.UpdateTraceAccumulator._for_layout(7, layout)
    with pytest.raises(error):
        accumulator.record_update(
            0,
            4,
            _golden_learning_rate(0, layout),
            cast(np.float32, loss),
            cast(np.float32, norm),
        )
    assert accumulator.next_global_update == 0


def test_epoch_boundaries_and_finish_state_machine_are_strict() -> None:
    layout = _tiny_trace_layout()
    accumulator = evidence.UpdateTraceAccumulator._for_layout(7, layout)
    with pytest.raises(Experiment002TrainingEvidenceError, match="complete pending"):
        accumulator.finish_epoch()
    with pytest.raises(Experiment002TrainingEvidenceError, match="every update"):
        accumulator.finish()

    for global_update in range(3):
        _record(accumulator, global_update, layout)
    with pytest.raises(Experiment002TrainingEvidenceError, match="finish_epoch"):
        _record(accumulator, 3, layout)
    first_epoch = accumulator.finish_epoch()
    with pytest.raises(Experiment002TrainingEvidenceError, match="complete pending"):
        accumulator.finish_epoch()
    with pytest.raises(Experiment002TrainingEvidenceError, match="every update"):
        accumulator.finish()

    for global_update in range(3, 6):
        _record(accumulator, global_update, layout)
    accumulator.finish_epoch()
    complete = accumulator.finish()
    assert first_epoch.zero_based_epoch == 0
    assert complete.update_count == 6
    with pytest.raises(Experiment002TrainingEvidenceError, match="already issued"):
        accumulator.finish()
    with pytest.raises(Experiment002TrainingEvidenceError, match="cannot append"):
        accumulator.record_update(
            6,
            4,
            0.1,
            np.float32(1.0),
            np.float32(1.0),
        )


def test_trace_capabilities_reject_copy_pickle_forge_and_slot_mutation() -> None:
    accumulator, epochs, complete = _finish_tiny_trace()
    epoch = epochs[0]
    for value in (epoch, complete, accumulator):
        with pytest.raises(TypeError):
            copy.copy(value)
        with pytest.raises(TypeError):
            copy.deepcopy(value)
        with pytest.raises(TypeError):
            pickle.dumps(value)

    forged_epoch = object.__new__(EpochUpdateTraceEvidence)
    with pytest.raises(Experiment002TrainingEvidenceError, match="not issued"):
        _ = forged_epoch.sha256
    forged_complete = object.__new__(CompleteUpdateTraceEvidence)
    with pytest.raises(Experiment002TrainingEvidenceError, match="not issued"):
        _ = forged_complete.sha256

    object.__setattr__(epoch, "_training_cross_entropy", np.float64(-0.0))
    with pytest.raises(Experiment002TrainingEvidenceError, match="changed"):
        _ = epoch.training_cross_entropy
    object.__setattr__(complete, "_update_count", 5)
    with pytest.raises(Experiment002TrainingEvidenceError, match="changed"):
        _ = complete.update_count


def test_trace_capability_revalidates_internal_record_scalar_types() -> None:
    _, epochs, _ = _finish_tiny_trace()
    state = evidence._issued_epoch_trace_state(epochs[0])
    object.__setattr__(state.records[0], "global_update", np.uint32(0))

    with pytest.raises(TypeError, match="record global_update"):
        _ = epochs[0].sha256


@pytest.mark.parametrize(
    "field",
    [
        "seed",
        "zero_based_epoch",
        "records",
        "framed_payload",
        "sha256",
        "training_cross_entropy",
    ],
)
def test_epoch_trace_revalidates_every_internal_state_field_type(field: str) -> None:
    _, epochs, _ = _finish_tiny_trace()
    state = evidence._issued_epoch_trace_state(epochs[0])
    replacements: dict[str, object] = {
        "seed": np.uint32(state.seed),
        "zero_based_epoch": np.uint32(state.zero_based_epoch),
        "records": list(state.records),
        "framed_payload": bytearray(state.framed_payload),
        "sha256": _StringSubclass(state.sha256),
        "training_cross_entropy": np.float32(state.training_cross_entropy),
    }
    object.__setattr__(state, field, replacements[field])

    with pytest.raises((TypeError, Experiment002TrainingEvidenceError)):
        _ = epochs[0].sha256


@pytest.mark.parametrize(
    "field", ["seed", "records", "epochs", "framed_payload", "sha256"]
)
def test_complete_trace_revalidates_every_internal_state_field_type(
    field: str,
) -> None:
    _, _, complete = _finish_tiny_trace()
    state = evidence._issued_complete_trace_state(complete)
    replacements: dict[str, object] = {
        "seed": np.uint32(state.seed),
        "records": list(state.records),
        "epochs": list(state.epochs),
        "framed_payload": bytearray(state.framed_payload),
        "sha256": _StringSubclass(state.sha256),
    }
    object.__setattr__(state, field, replacements[field])

    with pytest.raises((TypeError, Experiment002TrainingEvidenceError)):
        _ = complete.epochs


def test_trace_capabilities_revalidate_nested_layout_field_types() -> None:
    _, epochs, complete = _finish_tiny_trace()
    state = evidence._issued_complete_trace_state(complete)
    object.__setattr__(state.layout, "epochs", np.int64(state.layout.epochs))

    with pytest.raises(TypeError, match="epochs"):
        _ = complete.epochs
    with pytest.raises(TypeError, match="epochs"):
        _ = epochs[0].sha256


def test_unregistered_tiny_capabilities_cannot_pass_registered_verifiers() -> None:
    _, epochs, complete = _finish_tiny_trace()
    with pytest.raises(Experiment002TrainingEvidenceError, match="registered route"):
        evidence.verify_registered_epoch_update_trace(epochs[0])
    with pytest.raises(Experiment002TrainingEvidenceError, match="registered route"):
        evidence.verify_registered_complete_update_trace(complete)

    model_evidence = evidence._capture_model_tensors(
        _OrderedTinyModel(), layout=_tiny_model_layout()
    )
    with pytest.raises(Experiment002TrainingEvidenceError, match="registered route"):
        evidence.verify_registered_model_tensor_evidence(
            model_evidence,
            seed=evidence.REGISTERED_SEEDS[0],
            zero_based_epoch=0,
        )


def test_public_trace_constructor_binds_the_exact_layout_without_finishing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def initialize(
        self: evidence.UpdateTraceAccumulator,
        seed: int,
        layout: evidence._TraceLayout,
        route_marker: object | None,
    ) -> None:
        nonlocal calls
        del self
        calls += 1
        assert seed == evidence.REGISTERED_SEEDS[0]
        assert layout is evidence._REGISTERED_TRACE_LAYOUT
        assert route_marker is evidence._UPDATE_TRACE_ROUTE_MARKER

    monkeypatch.setattr(evidence.UpdateTraceAccumulator, "_initialize", initialize)
    evidence.UpdateTraceAccumulator(evidence.REGISTERED_SEEDS[0])
    assert calls == 1
    with pytest.raises(Experiment002TrainingEvidenceError, match="registered"):
        evidence.UpdateTraceAccumulator(7)
    with pytest.raises(TypeError):
        evidence.UpdateTraceAccumulator(cast(int, True))


def test_registered_trace_raw_record_and_public_epoch_finish_are_terminal() -> None:
    raw = evidence.UpdateTraceAccumulator(evidence.REGISTERED_SEEDS[0])
    with pytest.raises(Experiment002TrainingEvidenceError, match="raw scalar"):
        raw.record_update(
            0,
            evidence.TRAINING_BATCH_SIZE,
            evidence.registered_learning_rate(0),
            np.float32(0.5),
            np.float32(1.0),
        )
    assert raw._failed
    receipt = object.__new__(training_population.CompletedTrainingUpdate)
    transition = object.__new__(training_bridge.RegisteredOptimizerTransition)
    with pytest.raises(Experiment002TrainingEvidenceError, match="terminally failed"):
        raw._consume_registered_update(receipt, transition)

    public_finish = evidence.UpdateTraceAccumulator(evidence.REGISTERED_SEEDS[0])
    with pytest.raises(
        Experiment002TrainingEvidenceError, match="population finalization"
    ):
        public_finish.finish_epoch()
    assert public_finish._failed


@pytest.mark.parametrize(
    "field",
    ["seed", "layout", "route_marker", "trace_session_token", "authority_payload"],
)
def test_registered_trace_rejects_coherent_accumulator_authority_rewrites(
    field: str,
) -> None:
    accumulator = evidence.UpdateTraceAccumulator(evidence.REGISTERED_SEEDS[0])
    trace_state = evidence._TRACE_ACCUMULATOR_AUTHORITIES[accumulator]
    authority = trace_state.authority
    if field == "seed":
        replacement: object = evidence.REGISTERED_SEEDS[1]
        accumulator._seed = evidence.REGISTERED_SEEDS[1]
    elif field == "layout":
        layout_replacement = replace(
            accumulator._layout,
            warmup_start_lr=0.0004,
        )
        replacement = layout_replacement
        accumulator._layout = layout_replacement
    elif field == "route_marker":
        replacement = None
        accumulator._route_marker = None
    elif field == "trace_session_token":
        replacement = object()
        accumulator._trace_session_token = replacement
    else:
        replacement = b"changed"
    object.__setattr__(authority, field, replacement)
    if field != "authority_payload":
        object.__setattr__(
            authority,
            "authority_payload",
            evidence._frame_trace_accumulator_authority(
                authority.seed,
                authority.layout,
                authority.route_marker,
                authority.trace_session_token,
            ),
        )

    with pytest.raises(Experiment002TrainingEvidenceError, match="authority changed"):
        accumulator.record_update(
            0,
            evidence.TRAINING_BATCH_SIZE,
            evidence.registered_learning_rate(0),
            np.float32(0.5),
            np.float32(1.0),
        )
    assert accumulator._failed


def test_registered_trace_failed_set_defeats_coherent_failed_flag_reset() -> None:
    accumulator = evidence.UpdateTraceAccumulator(evidence.REGISTERED_SEEDS[0])
    evidence._fail_registered_trace(accumulator)
    trace_state = evidence._TRACE_ACCUMULATOR_AUTHORITIES[accumulator]
    accumulator._failed = False
    trace_state.failed = False

    with pytest.raises(Experiment002TrainingEvidenceError, match="authority changed"):
        accumulator.record_update(
            0,
            evidence.TRAINING_BATCH_SIZE,
            evidence.registered_learning_rate(0),
            np.float32(0.5),
            np.float32(1.0),
        )
    assert accumulator._failed


def test_registered_trace_derives_record_only_from_receipt_and_binds_consumption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accumulator = evidence.UpdateTraceAccumulator(evidence.REGISTERED_SEEDS[0])
    executor_token = object()
    epoch_token = object()
    batch_token = object()
    receipt_token = object()
    transition_token = object()
    consumption_token = object()
    transition = object.__new__(training_bridge.RegisteredOptimizerTransition)
    receipt = _registered_receipt_for_trace(
        transition=transition,
        epoch_token=epoch_token,
        batch_token=batch_token,
        receipt_token=receipt_token,
        transition_token=transition_token,
    )
    transition_snapshot = _registered_transition_snapshot_for_trace(
        executor_token=executor_token,
        epoch_token=epoch_token,
        batch_token=batch_token,
        transition_token=transition_token,
    )
    consumption = object.__new__(training_bridge.TraceConsumedTransition)
    consumption_snapshot = training_bridge._TraceConsumptionSnapshot(
        executor_session_token=executor_token,
        epoch_session_token=epoch_token,
        batch_token=batch_token,
        transition_token=transition_token,
        receipt_token=receipt_token,
        trace_session_token=accumulator._trace_session_token,
        consumption_token=consumption_token,
        zero_based_global_update=0,
        trace_record_index=0,
    )
    mark_calls = 0

    def transition_state(
        candidate: training_bridge.RegisteredOptimizerTransition,
        *,
        required_phase: training_bridge.TransitionPhase | None = None,
    ) -> training_bridge._OptimizerTransitionSnapshot:
        assert candidate is transition
        assert required_phase in (None, "ISSUED", "TRACE_CONSUMED")
        return transition_snapshot

    def mark(
        candidate: training_bridge.RegisteredOptimizerTransition,
        *,
        receipt_token: object,
        trace_session_token: object,
        trace_record_index: int,
    ) -> training_bridge.TraceConsumedTransition:
        nonlocal mark_calls
        mark_calls += 1
        assert candidate is transition
        assert receipt_token is consumption_snapshot.receipt_token
        assert trace_session_token is accumulator._trace_session_token
        assert trace_record_index == 0
        return consumption

    monkeypatch.setattr(evidence, "_transition_snapshot", transition_state)
    monkeypatch.setattr(training_population, "_transition_snapshot", transition_state)
    monkeypatch.setattr(evidence, "_mark_transition_trace_consumed", mark)
    monkeypatch.setattr(
        evidence,
        "_trace_consumption_snapshot",
        lambda candidate, **kwargs: consumption_snapshot,
    )

    assert accumulator._consume_registered_update(receipt, transition) is consumption
    assert mark_calls == 1
    assert accumulator.next_global_update == 1
    record = accumulator._records[0]
    expected_frame = struct.pack(
        "<IIdff",
        0,
        evidence.TRAINING_BATCH_SIZE,
        evidence.registered_learning_rate(0),
        np.float32(0.75),
        np.float32(1.25),
    )
    assert record.framed == expected_frame
    assert record.receipt is receipt
    assert record.receipt_token is receipt_token
    assert record.optimizer_transition_token is transition_token
    assert record.optimizer_transition is transition
    assert record.trace_session_token is accumulator._trace_session_token
    assert record.trace_consumption_token is consumption_token
    assert record.trace_consumption is consumption


def test_registered_trace_mismatch_fails_trace_without_sabotaging_foreign_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accumulator = evidence.UpdateTraceAccumulator(evidence.REGISTERED_SEEDS[0])
    epoch_token = object()
    batch_token = object()
    receipt_token = object()
    transition_token = object()
    transition = object.__new__(training_bridge.RegisteredOptimizerTransition)
    receipt = _registered_receipt_for_trace(
        transition=transition,
        epoch_token=epoch_token,
        batch_token=batch_token,
        receipt_token=receipt_token,
        transition_token=transition_token,
    )
    mismatched = _registered_transition_snapshot_for_trace(
        executor_token=object(),
        epoch_token=epoch_token,
        batch_token=batch_token,
        transition_token=transition_token,
        loss=np.float32(0.5),
    )
    marks = 0
    failures = 0

    def mark(
        *args: object, **kwargs: object
    ) -> training_bridge.TraceConsumedTransition:
        nonlocal marks
        del args, kwargs
        marks += 1
        return object.__new__(training_bridge.TraceConsumedTransition)

    def fail(candidate: training_bridge.RegisteredOptimizerTransition) -> None:
        nonlocal failures
        assert candidate is transition
        failures += 1

    monkeypatch.setattr(
        evidence, "_transition_snapshot", lambda *args, **kwargs: mismatched
    )
    monkeypatch.setattr(
        training_population,
        "_transition_snapshot",
        lambda *args, **kwargs: mismatched,
    )
    monkeypatch.setattr(evidence, "_mark_transition_trace_consumed", mark)
    monkeypatch.setattr(evidence, "_fail_optimizer_transition", fail)

    with pytest.raises(Experiment002TrainingEvidenceError, match="differs"):
        accumulator._consume_registered_update(receipt, transition)
    assert accumulator._failed
    assert accumulator.next_global_update == 0
    assert marks == 0
    assert failures == 0
    with pytest.raises(Experiment002TrainingEvidenceError, match="terminally failed"):
        accumulator._consume_registered_update(receipt, transition)


def test_registered_trace_post_claim_failure_fails_owned_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accumulator = evidence.UpdateTraceAccumulator(evidence.REGISTERED_SEEDS[0])
    executor_token = object()
    epoch_token = object()
    batch_token = object()
    receipt_token = object()
    transition_token = object()
    transition = object.__new__(training_bridge.RegisteredOptimizerTransition)
    receipt = _registered_receipt_for_trace(
        transition=transition,
        epoch_token=epoch_token,
        batch_token=batch_token,
        receipt_token=receipt_token,
        transition_token=transition_token,
    )
    transition_snapshot = _registered_transition_snapshot_for_trace(
        executor_token=executor_token,
        epoch_token=epoch_token,
        batch_token=batch_token,
        transition_token=transition_token,
    )
    consumption = object.__new__(training_bridge.TraceConsumedTransition)
    mismatched_consumption = training_bridge._TraceConsumptionSnapshot(
        executor_session_token=executor_token,
        epoch_session_token=epoch_token,
        batch_token=batch_token,
        transition_token=transition_token,
        receipt_token=object(),
        trace_session_token=accumulator._trace_session_token,
        consumption_token=object(),
        zero_based_global_update=0,
        trace_record_index=0,
    )
    failures = 0

    def fail(candidate: training_bridge.RegisteredOptimizerTransition) -> None:
        nonlocal failures
        assert candidate is transition
        failures += 1

    monkeypatch.setattr(
        evidence, "_transition_snapshot", lambda *args, **kwargs: transition_snapshot
    )
    monkeypatch.setattr(
        training_population,
        "_transition_snapshot",
        lambda *args, **kwargs: transition_snapshot,
    )
    monkeypatch.setattr(
        evidence, "_mark_transition_trace_consumed", lambda *args, **kwargs: consumption
    )
    monkeypatch.setattr(
        evidence,
        "_trace_consumption_snapshot",
        lambda candidate, **kwargs: mismatched_consumption,
    )
    monkeypatch.setattr(evidence, "_fail_optimizer_transition", fail)

    with pytest.raises(Experiment002TrainingEvidenceError, match="consumption"):
        accumulator._consume_registered_update(receipt, transition)
    assert accumulator._failed
    assert accumulator.next_global_update == 0
    assert failures == 1


def test_synthetic_trace_rejects_partial_or_complete_registered_authority() -> None:
    layout = _tiny_trace_layout()
    accumulator = evidence.UpdateTraceAccumulator._for_layout(7, layout)
    _record(accumulator, 0, layout)
    record = accumulator._records[0]
    object.__setattr__(record, "receipt_token", object())
    with pytest.raises(Experiment002TrainingEvidenceError, match="incomplete"):
        evidence._validate_record(record, 0, layout)

    transition = object.__new__(training_bridge.RegisteredOptimizerTransition)
    consumption = object.__new__(training_bridge.TraceConsumedTransition)
    receipt = object.__new__(training_population.CompletedTrainingUpdate)
    object.__setattr__(record, "receipt", receipt)
    object.__setattr__(record, "optimizer_transition_token", object())
    object.__setattr__(record, "optimizer_transition", transition)
    object.__setattr__(record, "trace_session_token", object())
    object.__setattr__(record, "trace_consumption_token", object())
    object.__setattr__(record, "trace_consumption", consumption)
    evidence._validate_record(record, 0, layout)
    with pytest.raises(Experiment002TrainingEvidenceError, match="synthetic"):
        evidence._validate_record_authority_route((record,), None)


def test_registered_epoch_finish_crosschecks_population_authority_without_framing_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accumulator = evidence.UpdateTraceAccumulator(evidence.REGISTERED_SEEDS[0])
    executor_token = object()
    epoch_token = object()
    records: list[evidence._UpdateRecord] = []
    updates: list[training_population._CompletedTrainingUpdateSnapshot] = []
    batch_tokens: list[object] = []
    receipt_tokens: list[object] = []
    transition_tokens: list[object] = []
    transitions: list[training_bridge.RegisteredOptimizerTransition] = []
    consumptions: list[training_bridge.TraceConsumedTransition] = []
    transition_snapshots: dict[int, training_bridge._OptimizerTransitionSnapshot] = {}
    consumption_snapshots: dict[int, training_bridge._TraceConsumptionSnapshot] = {}
    expected_frames: list[bytes] = []
    for global_update in range(evidence.UPDATES_PER_EPOCH):
        batch_size = (
            evidence.LAST_BATCH_SIZE
            if global_update == evidence.UPDATES_PER_EPOCH - 1
            else evidence.TRAINING_BATCH_SIZE
        )
        learning_rate = evidence.registered_learning_rate(global_update)
        loss = np.float32(global_update / 1000.0)
        norm = np.float32(1.0 + global_update / 1000.0)
        batch_token = object()
        receipt_token = object()
        transition_token = object()
        consumption_token = object()
        transition = object.__new__(training_bridge.RegisteredOptimizerTransition)
        consumption = object.__new__(training_bridge.TraceConsumedTransition)
        receipt = _registered_receipt_for_trace(
            transition=transition,
            epoch_token=epoch_token,
            batch_token=batch_token,
            receipt_token=receipt_token,
            transition_token=transition_token,
            global_update=global_update,
            batch_size=batch_size,
            learning_rate=learning_rate,
            loss=loss,
            norm=norm,
        )
        with training_population._ISSUED_RECEIPTS_LOCK:
            receipt_state = training_population._ISSUED_RECEIPTS[receipt]
            receipt_state.accepted = True
            training_population._ACCEPTED_RECEIPTS.add(receipt)
        transition_snapshot = _registered_transition_snapshot_for_trace(
            executor_token=executor_token,
            epoch_token=epoch_token,
            batch_token=batch_token,
            transition_token=transition_token,
            global_update=global_update,
            batch_size=batch_size,
            learning_rate=learning_rate,
            loss=loss,
            norm=norm,
        )
        consumption_snapshot = training_bridge._TraceConsumptionSnapshot(
            executor_session_token=executor_token,
            epoch_session_token=epoch_token,
            batch_token=batch_token,
            transition_token=transition_token,
            receipt_token=receipt_token,
            trace_session_token=accumulator._trace_session_token,
            consumption_token=consumption_token,
            zero_based_global_update=global_update,
            trace_record_index=global_update,
        )
        framed = struct.pack(
            "<IIdff", global_update, batch_size, learning_rate, loss, norm
        )
        expected_frames.append(framed)
        records.append(
            evidence._UpdateRecord(
                global_update=global_update,
                batch_size=batch_size,
                learning_rate=learning_rate,
                batch_mean_training_loss=loss,
                returned_preclip_l2_norm=norm,
                framed=framed,
                receipt=receipt,
                receipt_token=receipt_token,
                optimizer_transition_token=transition_token,
                optimizer_transition=transition,
                trace_session_token=accumulator._trace_session_token,
                trace_consumption_token=consumption_token,
                trace_consumption=consumption,
            )
        )
        updates.append(
            training_population._CompletedTrainingUpdateSnapshot(
                session_token=epoch_token,
                batch_token=batch_token,
                receipt_token=receipt_token,
                optimizer_transition_token=transition_token,
                seed=evidence.REGISTERED_SEEDS[0],
                zero_based_epoch=0,
                batch_index=global_update,
                zero_based_global_update=global_update,
                batch_size=batch_size,
                learning_rate=learning_rate,
                learning_rate_bytes=struct.pack("<d", learning_rate),
                batch_mean_training_loss=loss,
                batch_mean_training_loss_bytes=struct.pack("<f", loss),
                returned_preclip_l2_norm=norm,
                returned_preclip_l2_norm_bytes=struct.pack("<f", norm),
            )
        )
        batch_tokens.append(batch_token)
        receipt_tokens.append(receipt_token)
        transition_tokens.append(transition_token)
        transitions.append(transition)
        consumptions.append(consumption)
        transition_snapshots[id(transition)] = transition_snapshot
        consumption_snapshots[id(consumption)] = consumption_snapshot

    population_snapshot = training_population._CompletedTrainingPopulationSnapshot(
        session_token=epoch_token,
        seed=evidence.REGISTERED_SEEDS[0],
        zero_based_epoch=0,
        sha256="0" * 64,
        example_count=evidence.TRAINING_EXAMPLE_COUNT,
        batch_count=evidence.UPDATES_PER_EPOCH,
        updates=tuple(updates),
        ordered_batch_tokens=tuple(batch_tokens),
        ordered_receipt_tokens=tuple(receipt_tokens),
        ordered_optimizer_transition_tokens=tuple(transition_tokens),
        ordered_optimizer_transitions=tuple(transitions),
        ordered_trace_consumptions=tuple(consumptions),
        route_marker=object(),
    )
    population = object.__new__(training_population.CompletedTrainingPopulation)
    accumulator._records.extend(records)

    def transition_state(
        candidate: training_bridge.RegisteredOptimizerTransition,
        *,
        required_phase: training_bridge.TransitionPhase | None = None,
    ) -> training_bridge._OptimizerTransitionSnapshot:
        assert required_phase == "POPULATION_ACCEPTED"
        return transition_snapshots[id(candidate)]

    def consumption_state(
        candidate: training_bridge.TraceConsumedTransition,
        *,
        required_used: bool | None = None,
    ) -> training_bridge._TraceConsumptionSnapshot:
        assert required_used is True
        return consumption_snapshots[id(candidate)]

    def verify_population(
        candidate: training_population.CompletedTrainingPopulation,
        *,
        seed: int,
        zero_based_epoch: int,
    ) -> None:
        assert candidate is population
        assert seed == evidence.REGISTERED_SEEDS[0]
        assert zero_based_epoch == 0

    monkeypatch.setattr(evidence, "_transition_snapshot", transition_state)
    monkeypatch.setattr(
        training_population,
        "_transition_snapshot",
        lambda candidate, **kwargs: transition_snapshots[id(candidate)],
    )
    monkeypatch.setattr(evidence, "_trace_consumption_snapshot", consumption_state)
    monkeypatch.setattr(
        evidence, "verify_completed_registered_training_population", verify_population
    )
    monkeypatch.setattr(
        evidence,
        "_completed_training_population_snapshot",
        lambda candidate: population_snapshot,
    )

    epoch = accumulator._finish_registered_epoch(population)
    expected_payload = struct.pack(
        "<III", evidence.REGISTERED_SEEDS[0], 0, len(records)
    ) + b"".join(expected_frames)
    assert evidence._issued_epoch_trace_state(epoch).framed_payload == expected_payload
    assert (
        epoch.sha256
        == hashlib.sha256(
            evidence.EPOCH_UPDATE_TRACE_DOMAIN + expected_payload
        ).hexdigest()
    )
    object.__setattr__(records[0], "trace_consumption_token", object())
    with pytest.raises(
        Experiment002TrainingEvidenceError, match="transition authority"
    ):
        _ = epoch.sha256


@pytest.mark.parametrize(
    "kwargs",
    [
        {"population_count": 11},
        {"last_batch_size": 5},
        {"warmup_updates": 1},
        {"warmup_updates": 6},
        {"minimum_lr": float("nan")},
        {"minimum_lr": 0.2},
    ],
)
def test_private_trace_layout_rejects_incoherent_values(
    kwargs: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "epochs": 2,
        "updates_per_epoch": 3,
        "population_count": 10,
        "batch_size": 4,
        "last_batch_size": 2,
        "warmup_updates": 2,
        "warmup_start_lr": 0.1,
        "maximum_lr": 0.3,
        "minimum_lr": 0.01,
    }
    values.update(kwargs)
    with pytest.raises((TypeError, Experiment002TrainingEvidenceError)):
        evidence._TraceLayout(**values)  # type: ignore[arg-type]
