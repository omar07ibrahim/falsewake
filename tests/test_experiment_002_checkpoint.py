from __future__ import annotations

import ast
import builtins
import copy
import hashlib
import json
import pickle
import struct
import threading
from pathlib import Path
from typing import Any

import pytest
import safetensors as safetensors_package
import safetensors.torch as safetensors_module
import torch
from safetensors.torch import load as safetensors_load
from safetensors.torch import save as safetensors_save
from torch import Tensor

import falsewake.experiment_002_checkpoint as checkpoint


def _named_tensors() -> tuple[tuple[str, Tensor], ...]:
    return (
        ("é_tensor", torch.tensor([[-1.25]], dtype=torch.float32)),
        ("a_tensor", torch.tensor([-0.0, 2.5], dtype=torch.float32)),
    )


def _preserve() -> checkpoint._UnregisteredCheckpoint:
    return checkpoint._preserve_unregistered_checkpoint(_named_tensors())


def test_source_is_an_in_memory_unregistered_kernel_only() -> None:
    source = Path("src/falsewake/experiment_002_checkpoint.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    module_functions: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported_roots.update(
                alias.name.split(".", maxsplit=1)[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".", maxsplit=1)[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            module_functions.append(node.name)

    assert imported_roots == {
        "__future__",
        "dataclasses",
        "hashlib",
        "json",
        "math",
        "safetensors",
        "struct",
        "threading",
        "torch",
        "typing",
        "weakref",
    }
    assert all(name.startswith("_") for name in module_functions)
    assert "experiment_002_data" not in source
    assert "experiment_002_evidence" not in source
    assert "experiment_002_training_history" not in source
    assert "CausalKWS" not in source
    assert "save_file" not in source
    assert "load_file" not in source
    assert "safe_open" not in source
    assert "issue_registered" not in source
    assert "publish" not in source.lower().replace("publisher", "")


def test_preserve_serialize_and_roundtrip_match_independent_golden() -> None:
    named = _named_tensors()
    sources_before = tuple(tensor.clone() for _, tensor in named)
    preserved = checkpoint._preserve_unregistered_checkpoint(named)
    before = checkpoint._checkpoint_snapshot(preserved)

    expected_frame = bytearray(struct.pack("<I", 2))
    for name, shape, raw in (
        ("a_tensor", (2,), struct.pack("<ff", -0.0, 2.5)),
        ("é_tensor", (1, 1), struct.pack("<f", -1.25)),
    ):
        encoded = name.encode("utf-8")
        expected_frame.extend(struct.pack("<I", len(encoded)))
        expected_frame.extend(encoded)
        expected_frame.extend(struct.pack("<I", 5))
        expected_frame.extend(b"F32LE")
        expected_frame.extend(struct.pack("<I", len(shape)))
        for dimension in shape:
            expected_frame.extend(struct.pack("<Q", dimension))
        expected_frame.extend(struct.pack("<Q", len(raw)))
        expected_frame.extend(raw)
    expected_model_sha256 = hashlib.sha256(
        b"falsewake-exp002-model-tensors-v1\0" + expected_frame
    ).hexdigest()

    assert before.tensor_count == 2
    assert before.value_count == 3
    assert before.phase == "PRESERVED"
    assert before.model_tensor_sha256 == expected_model_sha256
    assert expected_model_sha256 == (
        "4ab7c4dd396d5ac19a00615830c57ea6bcf2225f6a66588b9b3aa436585d28e6"
    )

    serialized = checkpoint._serialize_unregistered_checkpoint(preserved)
    payload = checkpoint._serialized_checkpoint_bytes(serialized)
    after = checkpoint._checkpoint_snapshot(preserved)
    result = checkpoint._serialized_checkpoint_snapshot(serialized)

    assert after.phase == "SERIALIZED"
    assert result.tensor_count == 2
    assert result.value_count == 3
    assert result.model_tensor_sha256 == expected_model_sha256
    assert result.safetensors_byte_count == len(payload)
    assert result.safetensors_sha256 == hashlib.sha256(payload).hexdigest()
    assert checkpoint._serialized_checkpoint_bytes(serialized) == payload

    loaded = safetensors_load(payload)
    assert set(loaded) == {"a_tensor", "é_tensor"}
    assert struct.pack("<f", loaded["a_tensor"][0].item()) == struct.pack("<f", -0.0)
    torch.testing.assert_close(loaded["a_tensor"], sources_before[1], rtol=0, atol=0)
    torch.testing.assert_close(loaded["é_tensor"], sources_before[0], rtol=0, atol=0)


def test_preservation_is_owned_and_source_mutation_cannot_change_it() -> None:
    named = _named_tensors()
    preserved = checkpoint._preserve_unregistered_checkpoint(named)
    expected = checkpoint._checkpoint_snapshot(preserved)
    for _, tensor in named:
        tensor.add_(100.0)

    serialized = checkpoint._serialize_unregistered_checkpoint(preserved)
    observed = checkpoint._serialized_checkpoint_snapshot(serialized)
    loaded = safetensors_load(checkpoint._serialized_checkpoint_bytes(serialized))
    assert observed.model_tensor_sha256 == expected.model_tensor_sha256
    assert loaded["a_tensor"].tolist() == [-0.0, 2.5]
    assert loaded["é_tensor"].tolist() == [[-1.25]]


def test_save_is_called_twice_with_null_metadata_and_fresh_owned_mappings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Tensor]] = []
    metadata_values: list[dict[str, str] | None] = []

    def save_spy(
        tensors: dict[str, Tensor], metadata: dict[str, str] | None = None
    ) -> bytes:
        calls.append(dict(tensors))
        metadata_values.append(metadata)
        return safetensors_save(tensors, metadata=metadata)

    monkeypatch.setattr(safetensors_module, "save", save_spy)
    preserved = _preserve()
    state = checkpoint._CHECKPOINTS[preserved]
    checkpoint._serialize_unregistered_checkpoint(preserved)

    assert len(calls) == 2
    assert metadata_values == [None, None]
    assert tuple(calls[0]) == tuple(calls[1]) == ("a_tensor", "é_tensor")
    for first, second, internal in zip(
        calls[0].values(),
        calls[1].values(),
        state.tensors,
        strict=True,
    ):
        assert first is not second
        assert first is not internal
        assert second is not internal
        assert first.data_ptr() != second.data_ptr() != internal.data_ptr()
        assert first.storage_offset() == second.storage_offset() == 0
        assert first.untyped_storage().nbytes() == first.numel() * 4
        assert second.untyped_storage().nbytes() == second.numel() * 4


def test_frozen_safetensors_version_is_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preserved = _preserve()
    monkeypatch.setattr(safetensors_package, "__version__", "0.8.1")
    with pytest.raises(checkpoint.Experiment002CheckpointError, match="version"):
        checkpoint._serialize_unregistered_checkpoint(preserved)
    assert checkpoint._CHECKPOINTS[preserved].phase == "FAILED"


def test_save_cannot_mutate_preserved_tensors_through_its_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def hostile_save(
        tensors: dict[str, Tensor], metadata: dict[str, str] | None = None
    ) -> bytes:
        payload = safetensors_save(tensors, metadata=metadata)
        for tensor in tensors.values():
            tensor.add_(9.0)
        return payload

    monkeypatch.setattr(safetensors_module, "save", hostile_save)
    preserved = _preserve()
    expected = checkpoint._checkpoint_snapshot(preserved).model_tensor_sha256
    serialized = checkpoint._serialize_unregistered_checkpoint(preserved)
    assert (
        checkpoint._serialized_checkpoint_snapshot(serialized).model_tensor_sha256
        == expected
    )


def test_capabilities_are_issuer_only_noncopyable_and_one_shot() -> None:
    with pytest.raises(TypeError, match="issuer-only"):
        checkpoint._UnregisteredCheckpoint()
    counterfeit = object.__new__(checkpoint._UnregisteredCheckpoint)
    with pytest.raises(checkpoint.Experiment002CheckpointError, match="not issued"):
        checkpoint._checkpoint_snapshot(counterfeit)

    preserved = _preserve()
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.copy(preserved)
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.deepcopy(preserved)
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(preserved)

    serialized = checkpoint._serialize_unregistered_checkpoint(preserved)
    with pytest.raises(checkpoint.Experiment002CheckpointError, match="one-shot"):
        checkpoint._serialize_unregistered_checkpoint(preserved)
    assert checkpoint._checkpoint_snapshot(preserved).phase == "SERIALIZED"

    with pytest.raises(TypeError, match="issuer-only"):
        checkpoint._UnregisteredSerializedCheckpoint()
    counterfeit_serialized = object.__new__(
        checkpoint._UnregisteredSerializedCheckpoint
    )
    with pytest.raises(checkpoint.Experiment002CheckpointError, match="not issued"):
        checkpoint._serialized_checkpoint_snapshot(counterfeit_serialized)
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.copy(serialized)
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.deepcopy(serialized)
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(serialized)


def test_only_one_concurrent_serialization_can_succeed() -> None:
    preserved = _preserve()
    barrier = threading.Barrier(3)
    successes: list[checkpoint._UnregisteredSerializedCheckpoint] = []
    failures: list[BaseException] = []

    def run() -> None:
        barrier.wait()
        try:
            successes.append(checkpoint._serialize_unregistered_checkpoint(preserved))
        except BaseException as error:
            failures.append(error)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], checkpoint.Experiment002CheckpointError)
    assert "one-shot" in str(failures[0])
    assert checkpoint._checkpoint_snapshot(preserved).phase == "SERIALIZED"


@pytest.mark.parametrize(
    ("candidate", "error_type", "message"),
    [
        ([], TypeError, "exact tuple"),
        ((), checkpoint.Experiment002CheckpointError, "contain a tensor"),
        ((["a", torch.ones(1)],), TypeError, "exact pair"),
        (((1, torch.ones(1)),), TypeError, "exact strings"),
        ((("", torch.ones(1)),), checkpoint.Experiment002CheckpointError, "empty"),
        (
            (("\ud800", torch.ones(1)),),
            checkpoint.Experiment002CheckpointError,
            "valid UTF-8",
        ),
        (
            ((("a" * 257), torch.ones(1)),),
            checkpoint.Experiment002CheckpointError,
            "too long",
        ),
        (
            (("a", torch.ones(1)), ("a", torch.zeros(1))),
            checkpoint.Experiment002CheckpointError,
            "unique",
        ),
        (
            (("a", torch.nn.Parameter(torch.ones(1))),),
            TypeError,
            "exact Tensor",
        ),
        (
            (("a", torch.ones(1, requires_grad=True)),),
            checkpoint.Experiment002CheckpointError,
            "detached",
        ),
        (
            (("a", torch.ones(1, dtype=torch.float64)),),
            checkpoint.Experiment002CheckpointError,
            "float32",
        ),
        (
            (("a", torch.ones(2, 2).T),),
            checkpoint.Experiment002CheckpointError,
            "C-contiguous",
        ),
        (
            (("a", torch.empty(0)),),
            checkpoint.Experiment002CheckpointError,
            "cannot be empty",
        ),
        (
            (("a", torch.tensor([float("inf")])),),
            checkpoint.Experiment002CheckpointError,
            "finite",
        ),
        (
            (("a", torch.ones((1,) * 9)),),
            checkpoint.Experiment002CheckpointError,
            "rank",
        ),
        (
            (("a", torch.ones(65)),),
            checkpoint.Experiment002CheckpointError,
            "value limit",
        ),
    ],
)
def test_invalid_checkpoint_inputs_fail_closed(
    candidate: object,
    error_type: type[BaseException],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        checkpoint._preserve_unregistered_checkpoint(candidate)  # type: ignore[arg-type]


def test_sparse_tensor_and_large_population_are_rejected() -> None:
    sparse = torch.sparse_coo_tensor(
        torch.tensor([[0]]),
        torch.tensor([1.0]),
        size=(1,),
        check_invariants=False,
    )
    with pytest.raises(checkpoint.Experiment002CheckpointError, match="strided"):
        checkpoint._preserve_unregistered_checkpoint((("a", sparse),))

    too_many = tuple((f"tensor_{index}", torch.ones(1)) for index in range(5))
    with pytest.raises(checkpoint.Experiment002CheckpointError, match="tensor limit"):
        checkpoint._preserve_unregistered_checkpoint(too_many)
    assert checkpoint._MAX_UNREGISTERED_TENSORS == 4
    assert checkpoint._MAX_UNREGISTERED_VALUES == 64
    assert checkpoint._MAX_UNREGISTERED_TENSORS < 53
    assert checkpoint._MAX_UNREGISTERED_VALUES < 23_724


def test_utf8_name_order_controls_header_and_model_frame() -> None:
    named = (
        ("é", torch.tensor([3.0])),
        ("z", torch.tensor([2.0])),
        ("a", torch.tensor([1.0])),
    )
    preserved = checkpoint._preserve_unregistered_checkpoint(named)
    state = checkpoint._CHECKPOINTS[preserved]
    assert state.names == ("a", "z", "é")
    serialized = checkpoint._serialize_unregistered_checkpoint(preserved)
    payload = checkpoint._serialized_checkpoint_bytes(serialized)
    header_size = struct.unpack("<Q", payload[:8])[0]
    header = json.loads(payload[8 : 8 + header_size].rstrip(b" "))
    assert tuple(header) == ("a", "z", "é")
    assert "__metadata__" not in header


def test_nondeterministic_save_is_terminal_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0

    def changing_save(
        tensors: dict[str, Tensor], metadata: dict[str, str] | None = None
    ) -> bytes:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            first = next(iter(tensors.values()))
            first.add_(1.0)
        return safetensors_save(tensors, metadata=metadata)

    monkeypatch.setattr(safetensors_module, "save", changing_save)
    preserved = _preserve()
    with pytest.raises(
        checkpoint.Experiment002CheckpointError, match="not byte deterministic"
    ):
        checkpoint._serialize_unregistered_checkpoint(preserved)
    assert checkpoint._CHECKPOINTS[preserved].phase == "FAILED"
    with pytest.raises(checkpoint.Experiment002CheckpointError, match="terminally"):
        checkpoint._serialize_unregistered_checkpoint(preserved)


def test_metadata_or_nonbytes_save_is_terminal_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preserved = _preserve()

    def metadata_save(
        tensors: dict[str, Tensor], metadata: dict[str, str] | None = None
    ) -> bytes:
        del metadata
        return safetensors_save(tensors, metadata={"forbidden": "value"})

    monkeypatch.setattr(safetensors_module, "save", metadata_save)
    with pytest.raises(checkpoint.Experiment002CheckpointError, match="metadata"):
        checkpoint._serialize_unregistered_checkpoint(preserved)
    assert checkpoint._CHECKPOINTS[preserved].phase == "FAILED"

    another = _preserve()
    monkeypatch.setattr(safetensors_module, "save", lambda *_a, **_k: bytearray())
    with pytest.raises(checkpoint.Experiment002CheckpointError, match="return bytes"):
        checkpoint._serialize_unregistered_checkpoint(another)
    assert checkpoint._CHECKPOINTS[another].phase == "FAILED"


def test_corrupt_roundtrip_is_terminal_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preserved = _preserve()

    def corrupt_load(payload: bytes) -> dict[str, Tensor]:
        loaded = safetensors_load(payload)
        loaded["a_tensor"] = loaded["a_tensor"].clone()
        loaded["a_tensor"][0] = 0.0
        return loaded

    monkeypatch.setattr(safetensors_module, "load", corrupt_load)
    with pytest.raises(checkpoint.Experiment002CheckpointError, match="tensor bytes"):
        checkpoint._serialize_unregistered_checkpoint(preserved)
    assert checkpoint._CHECKPOINTS[preserved].phase == "FAILED"


def test_internal_tensor_or_authority_tampering_fails_terminally() -> None:
    preserved = _preserve()
    state = checkpoint._CHECKPOINTS[preserved]
    state.tensors[0].add_(1.0)
    with pytest.raises(
        checkpoint.Experiment002CheckpointError, match="tensors changed"
    ):
        checkpoint._serialize_unregistered_checkpoint(preserved)
    assert state.phase == "FAILED"

    another = _preserve()
    another_state = checkpoint._CHECKPOINTS[another]
    another_state.token = object()
    with pytest.raises(checkpoint.Experiment002CheckpointError, match="authority"):
        checkpoint._serialize_unregistered_checkpoint(another)
    assert another_state.phase == "FAILED"

    snapshot_target = _preserve()
    snapshot_state = checkpoint._CHECKPOINTS[snapshot_target]
    snapshot_state.tensors[0].mul_(2.0)
    with pytest.raises(
        checkpoint.Experiment002CheckpointError, match="tensors changed"
    ):
        checkpoint._checkpoint_snapshot(snapshot_target)
    assert snapshot_state.phase == "FAILED"


def test_serialized_payload_registry_tampering_is_detected() -> None:
    serialized = checkpoint._serialize_unregistered_checkpoint(_preserve())
    state = checkpoint._SERIALIZED[serialized]
    checkpoint._SERIALIZED[serialized] = checkpoint._SerializedState(
        token=state.token,
        route_marker=state.route_marker,
        tensor_count=state.tensor_count,
        value_count=state.value_count,
        model_tensor_sha256=state.model_tensor_sha256,
        safetensors_sha256="0" * 64,
        payload=state.payload,
    )
    with pytest.raises(checkpoint.Experiment002CheckpointError, match="authority"):
        checkpoint._serialized_checkpoint_bytes(serialized)


def test_serialized_counterfeit_and_coherent_replacement_fail_irreversibly() -> None:
    counterfeit = object.__new__(checkpoint._UnregisteredSerializedCheckpoint)
    forged_payload = b"not a safetensors file"
    forged_state = checkpoint._SerializedState(
        token=object(),
        route_marker=checkpoint._UNREGISTERED_ROUTE_MARKER,
        tensor_count=True,
        value_count=True,
        model_tensor_sha256="z" * 64,
        safetensors_sha256=hashlib.sha256(forged_payload).hexdigest(),
        payload=forged_payload,
    )
    checkpoint._SERIALIZED[counterfeit] = forged_state
    checkpoint._SERIALIZED_GUARDS[counterfeit] = (
        checkpoint._serialized_guard_from_state(forged_state)
    )
    with pytest.raises(checkpoint.Experiment002CheckpointError, match="not issued"):
        checkpoint._serialized_checkpoint_bytes(counterfeit)

    serialized = checkpoint._serialize_unregistered_checkpoint(_preserve())
    original = checkpoint._SERIALIZED[serialized]
    replacement_payload = b"still not a safetensors file"
    checkpoint._SERIALIZED[serialized] = checkpoint._SerializedState(
        token=original.token,
        route_marker=original.route_marker,
        tensor_count=original.tensor_count,
        value_count=original.value_count,
        model_tensor_sha256="0" * 64,
        safetensors_sha256=hashlib.sha256(replacement_payload).hexdigest(),
        payload=replacement_payload,
    )
    with pytest.raises(checkpoint.Experiment002CheckpointError, match="authority"):
        checkpoint._serialized_checkpoint_bytes(serialized)
    checkpoint._SERIALIZED[serialized] = original
    with pytest.raises(checkpoint.Experiment002CheckpointError, match="not issued"):
        checkpoint._serialized_checkpoint_bytes(serialized)


def test_completed_checkpoint_tamper_is_terminal_after_bytes_are_restored() -> None:
    preserved = _preserve()
    checkpoint._serialize_unregistered_checkpoint(preserved)
    state = checkpoint._CHECKPOINTS[preserved]
    original = state.tensors[0].clone()
    state.tensors[0].add_(1.0)
    with pytest.raises(
        checkpoint.Experiment002CheckpointError, match="tensors changed"
    ):
        checkpoint._serialize_unregistered_checkpoint(preserved)
    state.tensors[0].copy_(original)
    with pytest.raises(checkpoint.Experiment002CheckpointError, match="terminally"):
        checkpoint._checkpoint_snapshot(preserved)


def test_serialized_validation_rejects_bool_counts_and_nonhex_digests() -> None:
    payload = b"x"
    invalid = checkpoint._SerializedState(
        token=object(),
        route_marker=checkpoint._UNREGISTERED_ROUTE_MARKER,
        tensor_count=True,
        value_count=True,
        model_tensor_sha256="z" * 64,
        safetensors_sha256=hashlib.sha256(payload).hexdigest(),
        payload=payload,
    )
    guard = checkpoint._serialized_guard_from_state(invalid)
    with pytest.raises(checkpoint.Experiment002CheckpointError, match="counts"):
        checkpoint._validate_serialized_state(invalid, guard)


def test_guard_removal_and_phase_reset_are_irreversible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preserved = _preserve()
    guard = checkpoint._CHECKPOINT_GUARDS.pop(preserved)
    with pytest.raises(checkpoint.Experiment002CheckpointError, match="guard"):
        checkpoint._serialize_unregistered_checkpoint(preserved)
    checkpoint._CHECKPOINT_GUARDS[preserved] = guard
    checkpoint._CHECKPOINTS[preserved].phase = "PRESERVED"
    with pytest.raises(checkpoint.Experiment002CheckpointError, match="terminally"):
        checkpoint._serialize_unregistered_checkpoint(preserved)

    failed = _preserve()
    monkeypatch.setattr(safetensors_module, "save", lambda *_a, **_k: bytearray())
    with pytest.raises(checkpoint.Experiment002CheckpointError, match="return bytes"):
        checkpoint._serialize_unregistered_checkpoint(failed)
    checkpoint._CHECKPOINTS[failed].phase = "PRESERVED"
    monkeypatch.setattr(safetensors_module, "save", safetensors_save)
    with pytest.raises(checkpoint.Experiment002CheckpointError, match="terminally"):
        checkpoint._serialize_unregistered_checkpoint(failed)

    completed = _preserve()
    checkpoint._serialize_unregistered_checkpoint(completed)
    checkpoint._CHECKPOINTS[completed].phase = "PRESERVED"
    with pytest.raises(checkpoint.Experiment002CheckpointError, match="terminally"):
        checkpoint._serialize_unregistered_checkpoint(completed)


def test_serialized_guard_removal_is_irreversible() -> None:
    serialized = checkpoint._serialize_unregistered_checkpoint(_preserve())
    guard = checkpoint._SERIALIZED_GUARDS.pop(serialized)
    with pytest.raises(checkpoint.Experiment002CheckpointError, match="not issued"):
        checkpoint._serialized_checkpoint_bytes(serialized)
    checkpoint._SERIALIZED_GUARDS[serialized] = guard
    with pytest.raises(checkpoint.Experiment002CheckpointError, match="not issued"):
        checkpoint._serialized_checkpoint_bytes(serialized)


def test_kernel_does_not_open_or_create_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    before = tuple(tmp_path.iterdir())
    actual_open = builtins.open

    def reject_open(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("checkpoint kernel attempted filesystem access")

    monkeypatch.setattr(builtins, "open", reject_open)
    try:
        serialized = checkpoint._serialize_unregistered_checkpoint(_preserve())
        assert checkpoint._serialized_checkpoint_bytes(serialized)
    finally:
        monkeypatch.setattr(builtins, "open", actual_open)
    assert tuple(tmp_path.iterdir()) == before
