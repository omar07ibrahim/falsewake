"""In-memory checkpoint kernel for unregistered Experiment 002 probes.

This module intentionally accepts only a deliberately tiny synthetic tensor
population.  The source-bound registered checkpoint and canonical artifact
publisher are separate authorities and are never exposed through this kernel.
It keeps an owned snapshot in memory and verifies deterministic safetensors
bytes without creating a file or issuing a reusable registered capability.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import threading
import weakref
from dataclasses import dataclass, field
from typing import Final, Literal, NoReturn, cast

import safetensors as _safetensors
import safetensors.torch as _safetensors_torch
import torch
from torch import Tensor

_MODEL_TENSOR_DOMAIN: Final = b"falsewake-exp002-model-tensors-v1\0"
_DTYPE_TAG: Final = b"F32LE"
_MAX_UNREGISTERED_TENSORS: Final = 4
_MAX_UNREGISTERED_VALUES: Final = 64
_MAX_NAME_BYTES: Final = 256
_SAFETENSORS_VERSION: Final = "0.8.0"
_UNREGISTERED_ROUTE_MARKER: Final = object()

type _CheckpointPhase = Literal["PRESERVED", "SERIALIZED", "FAILED"]


class Experiment002CheckpointError(ValueError):
    """The unregistered in-memory checkpoint violated its exact contract."""


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
class _UnregisteredCheckpoint:
    """Issuer-only authority for one preserved synthetic tensor snapshot."""

    def __init__(self) -> None:
        raise TypeError("unregistered checkpoints are issuer-only")

    def __copy__(self) -> NoReturn:
        raise TypeError("unregistered checkpoints cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        del memo
        raise TypeError("unregistered checkpoints cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("unregistered checkpoints cannot be serialized")


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
class _UnregisteredSerializedCheckpoint:
    """Issuer-only handle for verified bytes that remain noncanonical."""

    def __init__(self) -> None:
        raise TypeError("serialized unregistered checkpoints are issuer-only")

    def __copy__(self) -> NoReturn:
        raise TypeError("serialized unregistered checkpoints cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        del memo
        raise TypeError("serialized unregistered checkpoints cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("serialized unregistered checkpoints cannot be serialized")


@dataclass(frozen=True, slots=True)
class _CheckpointSnapshot:
    tensor_count: int
    value_count: int
    model_tensor_sha256: str
    phase: _CheckpointPhase


@dataclass(frozen=True, slots=True)
class _SerializedCheckpointSnapshot:
    tensor_count: int
    value_count: int
    model_tensor_sha256: str
    safetensors_sha256: str
    safetensors_byte_count: int


@dataclass(frozen=True, slots=True)
class _CheckpointGuard:
    token: object
    route_marker: object
    names: tuple[str, ...]
    tensors: tuple[Tensor, ...]
    tensor_count: int
    value_count: int
    model_tensor_sha256: str
    lock: threading.RLock


@dataclass(slots=True)
class _CheckpointState:
    token: object
    route_marker: object
    names: tuple[str, ...]
    tensors: tuple[Tensor, ...]
    tensor_count: int
    value_count: int
    model_tensor_sha256: str
    phase: _CheckpointPhase = "PRESERVED"
    lock: threading.RLock = field(default_factory=threading.RLock)


@dataclass(frozen=True, slots=True)
class _SerializedState:
    token: object
    route_marker: object
    tensor_count: int
    value_count: int
    model_tensor_sha256: str
    safetensors_sha256: str
    payload: bytes


@dataclass(frozen=True, slots=True)
class _SerializedGuard:
    token: object
    route_marker: object
    tensor_count: int
    value_count: int
    model_tensor_sha256: str
    safetensors_sha256: str
    payload: bytes


_CHECKPOINTS: weakref.WeakKeyDictionary[_UnregisteredCheckpoint, _CheckpointState] = (
    weakref.WeakKeyDictionary()
)
_CHECKPOINT_GUARDS: weakref.WeakKeyDictionary[
    _UnregisteredCheckpoint, _CheckpointGuard
] = weakref.WeakKeyDictionary()
_SERIALIZED: weakref.WeakKeyDictionary[
    _UnregisteredSerializedCheckpoint, _SerializedState
] = weakref.WeakKeyDictionary()
_SERIALIZED_GUARDS: weakref.WeakKeyDictionary[
    _UnregisteredSerializedCheckpoint, _SerializedGuard
] = weakref.WeakKeyDictionary()
_ISSUED_SERIALIZED: weakref.WeakSet[_UnregisteredSerializedCheckpoint] = (
    weakref.WeakSet()
)
_FAILED_CHECKPOINTS: weakref.WeakSet[_UnregisteredCheckpoint] = weakref.WeakSet()
_COMPLETED_CHECKPOINTS: weakref.WeakSet[_UnregisteredCheckpoint] = weakref.WeakSet()
_FAILED_SERIALIZED: weakref.WeakSet[_UnregisteredSerializedCheckpoint] = (
    weakref.WeakSet()
)
_REGISTRY_LOCK = threading.RLock()


def _preserve_unregistered_checkpoint(
    named_tensors: tuple[tuple[str, Tensor], ...],
) -> _UnregisteredCheckpoint:
    """Clone a tiny synthetic tensor mapping into issuer-owned CPU memory."""

    names, tensors, value_count = _validate_and_clone_named_tensors(named_tensors)
    tensor_count = len(names)
    model_tensor_sha256 = _model_tensor_sha256(names, tensors)
    token = object()
    result = object.__new__(_UnregisteredCheckpoint)
    state = _CheckpointState(
        token=token,
        route_marker=_UNREGISTERED_ROUTE_MARKER,
        names=names,
        tensors=tensors,
        tensor_count=tensor_count,
        value_count=value_count,
        model_tensor_sha256=model_tensor_sha256,
    )
    guard = _CheckpointGuard(
        token=token,
        route_marker=_UNREGISTERED_ROUTE_MARKER,
        names=names,
        tensors=tensors,
        tensor_count=tensor_count,
        value_count=value_count,
        model_tensor_sha256=model_tensor_sha256,
        lock=state.lock,
    )
    with _REGISTRY_LOCK:
        _CHECKPOINTS[result] = state
        _CHECKPOINT_GUARDS[result] = guard
    _checkpoint_snapshot(result)
    return result


def _serialize_unregistered_checkpoint(
    checkpoint: _UnregisteredCheckpoint,
) -> _UnregisteredSerializedCheckpoint:
    """Serialize twice, roundtrip, and retain only verified in-memory bytes."""

    state = _issued_checkpoint_state(checkpoint)
    guard = _issued_checkpoint_guard(checkpoint)
    if state.lock is not guard.lock:
        _terminal_fail(checkpoint, state)
        raise Experiment002CheckpointError("checkpoint lock authority changed")
    with guard.lock:
        expected_one_shot_rejection = False
        try:
            _require_safetensors_runtime()
            _validate_checkpoint_authority(state, guard)
            if state.phase == "SERIALIZED":
                expected_one_shot_rejection = True
                raise Experiment002CheckpointError("checkpoint is one-shot")
            if state.phase == "FAILED":
                raise Experiment002CheckpointError("checkpoint is terminally failed")
            if state.phase != "PRESERVED":
                raise Experiment002CheckpointError("checkpoint phase is invalid")

            first = _serialize_once(state.names, state.tensors)
            _validate_checkpoint_authority(state, guard)
            second = _serialize_once(state.names, state.tensors)
            _validate_checkpoint_authority(state, guard)
            if first != second:
                raise Experiment002CheckpointError(
                    "safetensors serialization was not byte deterministic"
                )
            _verify_safetensors_roundtrip(
                first,
                names=state.names,
                tensors=state.tensors,
                expected_model_sha256=state.model_tensor_sha256,
            )
            _validate_checkpoint_authority(state, guard)

            serialized = object.__new__(_UnregisteredSerializedCheckpoint)
            serialized_state = _SerializedState(
                token=object(),
                route_marker=_UNREGISTERED_ROUTE_MARKER,
                tensor_count=state.tensor_count,
                value_count=state.value_count,
                model_tensor_sha256=state.model_tensor_sha256,
                safetensors_sha256=hashlib.sha256(first).hexdigest(),
                payload=first,
            )
            serialized_guard = _serialized_guard_from_state(serialized_state)
            with _REGISTRY_LOCK:
                _SERIALIZED[serialized] = serialized_state
                _SERIALIZED_GUARDS[serialized] = serialized_guard
                _ISSUED_SERIALIZED.add(serialized)
                state.phase = "SERIALIZED"
                _COMPLETED_CHECKPOINTS.add(checkpoint)
            _serialized_checkpoint_snapshot(serialized)
            return serialized
        except BaseException:
            if not expected_one_shot_rejection:
                _terminal_fail(checkpoint, state)
            raise


def _checkpoint_snapshot(checkpoint: _UnregisteredCheckpoint) -> _CheckpointSnapshot:
    """Return immutable metadata without exposing preserved tensor objects."""

    state = _issued_checkpoint_state(checkpoint)
    guard = _issued_checkpoint_guard(checkpoint)
    if state.lock is not guard.lock:
        _terminal_fail(checkpoint, state)
        raise Experiment002CheckpointError("checkpoint lock authority changed")
    with guard.lock:
        try:
            _validate_checkpoint_authority(state, guard)
        except BaseException:
            _terminal_fail(checkpoint, state)
            raise
        return _CheckpointSnapshot(
            tensor_count=state.tensor_count,
            value_count=state.value_count,
            model_tensor_sha256=state.model_tensor_sha256,
            phase=state.phase,
        )


def _serialized_checkpoint_snapshot(
    checkpoint: _UnregisteredSerializedCheckpoint,
) -> _SerializedCheckpointSnapshot:
    """Return immutable digest metadata for noncanonical serialized bytes."""

    state, guard = _issued_serialized_authority(checkpoint)
    try:
        _validate_serialized_state(state, guard)
    except BaseException:
        _terminal_fail_serialized(checkpoint)
        raise
    return _SerializedCheckpointSnapshot(
        tensor_count=state.tensor_count,
        value_count=state.value_count,
        model_tensor_sha256=state.model_tensor_sha256,
        safetensors_sha256=state.safetensors_sha256,
        safetensors_byte_count=len(state.payload),
    )


def _serialized_checkpoint_bytes(
    checkpoint: _UnregisteredSerializedCheckpoint,
) -> bytes:
    """Return the immutable noncanonical payload for isolated verification."""

    state, guard = _issued_serialized_authority(checkpoint)
    try:
        _validate_serialized_state(state, guard)
    except BaseException:
        _terminal_fail_serialized(checkpoint)
        raise
    return bytes(state.payload)


def _validate_and_clone_named_tensors(
    named_tensors: tuple[tuple[str, Tensor], ...],
) -> tuple[tuple[str, ...], tuple[Tensor, ...], int]:
    if type(named_tensors) is not tuple:
        raise TypeError("named_tensors must be an exact tuple")
    if not named_tensors:
        raise Experiment002CheckpointError("checkpoint must contain a tensor")
    if len(named_tensors) > _MAX_UNREGISTERED_TENSORS:
        raise Experiment002CheckpointError(
            "checkpoint exceeds the unregistered tensor limit"
        )

    captured: list[tuple[bytes, str, Tensor]] = []
    seen_names: set[str] = set()
    value_count = 0
    for entry in named_tensors:
        if type(entry) is not tuple or len(entry) != 2:
            raise TypeError("every named tensor must be an exact pair")
        name, tensor = entry
        if type(name) is not str:
            raise TypeError("tensor names must be exact strings")
        if not name:
            raise Experiment002CheckpointError("tensor names cannot be empty")
        try:
            encoded = name.encode("utf-8")
        except UnicodeEncodeError as error:
            raise Experiment002CheckpointError(
                "tensor names must be valid UTF-8"
            ) from error
        if len(encoded) > _MAX_NAME_BYTES:
            raise Experiment002CheckpointError("tensor name is too long")
        if name in seen_names:
            raise Experiment002CheckpointError("tensor names must be unique")
        seen_names.add(name)
        _require_source_tensor(tensor)
        value_count += tensor.numel()
        if value_count > _MAX_UNREGISTERED_VALUES:
            raise Experiment002CheckpointError(
                "checkpoint exceeds the unregistered value limit"
            )
        clone = tensor.detach().clone(memory_format=torch.contiguous_format)
        _require_owned_tensor_clone(clone, tensor)
        captured.append((encoded, name, clone))

    captured.sort(key=lambda item: item[0])
    names = tuple(name for _, name, _ in captured)
    tensors = tuple(tensor for _, _, tensor in captured)
    _require_stable_tensor_population(names, tensors, value_count=value_count)
    return names, tensors, value_count


def _require_source_tensor(tensor: Tensor) -> None:
    if type(tensor) is not Tensor:
        raise TypeError("checkpoint values must be exact Tensor objects")
    if tensor.layout is not torch.strided:
        raise Experiment002CheckpointError("checkpoint tensors must be strided")
    if tensor.device != torch.device("cpu"):
        raise Experiment002CheckpointError("checkpoint tensors must be on CPU")
    if tensor.dtype is not torch.float32:
        raise Experiment002CheckpointError("checkpoint tensors must be float32")
    if tensor.requires_grad or tensor.grad_fn is not None:
        raise Experiment002CheckpointError("checkpoint tensors must be detached")
    if not tensor.is_contiguous():
        raise Experiment002CheckpointError("checkpoint tensors must be C-contiguous")
    if tensor.numel() <= 0:
        raise Experiment002CheckpointError("checkpoint tensors cannot be empty")
    if tensor.ndim > 8:
        raise Experiment002CheckpointError("checkpoint tensor rank is too large")
    if not bool(torch.isfinite(tensor).all().item()):
        raise Experiment002CheckpointError("checkpoint tensors must be finite")


def _require_owned_tensor_clone(clone: Tensor, source: Tensor) -> None:
    _require_source_tensor(clone)
    if clone is source or clone.data_ptr() == source.data_ptr():
        raise Experiment002CheckpointError("checkpoint clone aliases its source")
    if clone.storage_offset() != 0:
        raise Experiment002CheckpointError("checkpoint clone has a storage offset")
    required_bytes = clone.numel() * clone.element_size()
    if clone.untyped_storage().nbytes() != required_bytes:
        raise Experiment002CheckpointError(
            "checkpoint clone does not own exact storage"
        )
    if clone.shape != source.shape or _tensor_payload(clone) != _tensor_payload(source):
        raise Experiment002CheckpointError("checkpoint clone changed tensor bytes")


def _require_stable_tensor_population(
    names: tuple[str, ...],
    tensors: tuple[Tensor, ...],
    *,
    value_count: int,
) -> None:
    if not 1 <= len(names) <= _MAX_UNREGISTERED_TENSORS:
        raise Experiment002CheckpointError("checkpoint tensor count is invalid")
    if len(tensors) != len(names) or len(set(names)) != len(names):
        raise Experiment002CheckpointError("checkpoint tensor mapping is invalid")
    if tuple(sorted(names, key=lambda name: name.encode("utf-8"))) != names:
        raise Experiment002CheckpointError("checkpoint names are not UTF-8 sorted")
    observed_values = 0
    for tensor in tensors:
        _require_source_tensor(tensor)
        if tensor.storage_offset() != 0 or (
            tensor.untyped_storage().nbytes() != tensor.numel() * tensor.element_size()
        ):
            raise Experiment002CheckpointError(
                "preserved checkpoint tensor does not own exact storage"
            )
        observed_values += tensor.numel()
    if (
        observed_values != value_count
        or not 1 <= value_count <= _MAX_UNREGISTERED_VALUES
    ):
        raise Experiment002CheckpointError("checkpoint value count is invalid")


def _serialize_once(names: tuple[str, ...], tensors: tuple[Tensor, ...]) -> bytes:
    mapping: dict[str, Tensor] = {}
    for name, tensor in zip(names, tensors, strict=True):
        clone = tensor.detach().clone(memory_format=torch.contiguous_format)
        _require_owned_tensor_clone(clone, tensor)
        mapping[name] = clone
    payload = _safetensors_torch.save(mapping, metadata=None)
    if type(payload) is not bytes:
        raise Experiment002CheckpointError("safetensors save did not return bytes")
    _require_safetensors_header(payload, names=names, tensors=tensors)
    return payload


def _require_safetensors_runtime() -> None:
    if _safetensors.__version__ != _SAFETENSORS_VERSION:
        raise Experiment002CheckpointError(
            "safetensors runtime version differs from the frozen serializer"
        )


def _verify_safetensors_roundtrip(
    payload: bytes,
    *,
    names: tuple[str, ...],
    tensors: tuple[Tensor, ...],
    expected_model_sha256: str,
) -> None:
    try:
        loaded_object = _safetensors_torch.load(payload)
    except Exception as error:
        raise Experiment002CheckpointError("safetensors roundtrip failed") from error
    if type(loaded_object) is not dict:
        raise Experiment002CheckpointError(
            "safetensors load returned an invalid mapping"
        )
    loaded = loaded_object
    if tuple(sorted(loaded, key=lambda name: name.encode("utf-8"))) != names:
        raise Experiment002CheckpointError("safetensors roundtrip names changed")
    roundtrip: list[Tensor] = []
    for name, expected in zip(names, tensors, strict=True):
        observed = loaded[name]
        if type(observed) is not Tensor:
            raise Experiment002CheckpointError(
                "safetensors roundtrip returned a non-Tensor value"
            )
        _require_source_tensor(observed)
        if observed.shape != expected.shape or (
            _tensor_payload(observed) != _tensor_payload(expected)
        ):
            raise Experiment002CheckpointError(
                "safetensors roundtrip changed tensor bytes"
            )
        roundtrip.append(observed)
    if _model_tensor_sha256(names, tuple(roundtrip)) != expected_model_sha256:
        raise Experiment002CheckpointError(
            "safetensors roundtrip changed the model tensor digest"
        )


def _require_safetensors_header(
    payload: bytes,
    *,
    names: tuple[str, ...],
    tensors: tuple[Tensor, ...],
) -> None:
    if len(payload) < 8:
        raise Experiment002CheckpointError("safetensors payload is truncated")
    header_size = struct.unpack("<Q", payload[:8])[0]
    if header_size == 0 or header_size % 8 != 0 or 8 + header_size > len(payload):
        raise Experiment002CheckpointError("safetensors header size is invalid")
    header_bytes = payload[8 : 8 + header_size]
    try:
        header_object = json.loads(header_bytes.rstrip(b" ").decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Experiment002CheckpointError("safetensors header is invalid") from error
    if type(header_object) is not dict or "__metadata__" in header_object:
        raise Experiment002CheckpointError("safetensors metadata must be null")
    header = cast(dict[str, object], header_object)
    if tuple(header) != names:
        raise Experiment002CheckpointError(
            "safetensors header names are not in UTF-8 order"
        )

    expected_offset = 0
    for name, tensor in zip(names, tensors, strict=True):
        record_object = header[name]
        if type(record_object) is not dict:
            raise Experiment002CheckpointError("safetensors tensor record is invalid")
        record = cast(dict[str, object], record_object)
        if set(record) != {"dtype", "shape", "data_offsets"}:
            raise Experiment002CheckpointError("safetensors tensor schema changed")
        shape_object = record["shape"]
        offsets_object = record["data_offsets"]
        if type(shape_object) is not list or type(offsets_object) is not list:
            raise Experiment002CheckpointError("safetensors tensor metadata is invalid")
        expected_end = expected_offset + tensor.numel() * tensor.element_size()
        if (
            record["dtype"] != "F32"
            or shape_object != list(tensor.shape)
            or (offsets_object != [expected_offset, expected_end])
        ):
            raise Experiment002CheckpointError("safetensors tensor metadata changed")
        expected_offset = expected_end
    if len(payload) != 8 + header_size + expected_offset:
        raise Experiment002CheckpointError("safetensors payload size is invalid")


def _model_tensor_sha256(names: tuple[str, ...], tensors: tuple[Tensor, ...]) -> str:
    return hashlib.sha256(
        _MODEL_TENSOR_DOMAIN + _frame_model_tensors(names, tensors)
    ).hexdigest()


def _frame_model_tensors(names: tuple[str, ...], tensors: tuple[Tensor, ...]) -> bytes:
    if len(names) != len(tensors):
        raise Experiment002CheckpointError("model tensor mapping is invalid")
    framed = bytearray(struct.pack("<I", len(names)))
    for name, tensor in zip(names, tensors, strict=True):
        encoded = name.encode("utf-8")
        raw = _tensor_payload(tensor)
        framed.extend(struct.pack("<I", len(encoded)))
        framed.extend(encoded)
        framed.extend(struct.pack("<I", len(_DTYPE_TAG)))
        framed.extend(_DTYPE_TAG)
        framed.extend(struct.pack("<I", tensor.ndim))
        for dimension in tensor.shape:
            framed.extend(struct.pack("<Q", dimension))
        framed.extend(struct.pack("<Q", len(raw)))
        framed.extend(raw)
    return bytes(framed)


def _tensor_payload(tensor: Tensor) -> bytes:
    values = tensor.reshape(-1).tolist()
    payload = bytearray()
    for value in values:
        if type(value) is not float or not math.isfinite(value):
            raise Experiment002CheckpointError("tensor payload is not finite float32")
        payload.extend(struct.pack("<f", value))
    expected_size = tensor.numel() * 4
    if len(payload) != expected_size:
        raise Experiment002CheckpointError("tensor payload size is invalid")
    return bytes(payload)


def _issued_checkpoint_state(
    checkpoint: _UnregisteredCheckpoint,
) -> _CheckpointState:
    if type(checkpoint) is not _UnregisteredCheckpoint:
        raise TypeError("checkpoint must be an exact _UnregisteredCheckpoint")
    with _REGISTRY_LOCK:
        state = _CHECKPOINTS.get(checkpoint)
        failed = checkpoint in _FAILED_CHECKPOINTS
        completed = checkpoint in _COMPLETED_CHECKPOINTS
        if state is None:
            _FAILED_CHECKPOINTS.add(checkpoint)
        elif state.phase == "FAILED" or completed != (state.phase == "SERIALIZED"):
            state.phase = "FAILED"
            _FAILED_CHECKPOINTS.add(checkpoint)
            failed = True
    if state is None:
        raise Experiment002CheckpointError("checkpoint was not issued here")
    if failed:
        _terminal_fail(checkpoint, state)
        raise Experiment002CheckpointError("checkpoint is terminally failed")
    return state


def _issued_checkpoint_guard(
    checkpoint: _UnregisteredCheckpoint,
) -> _CheckpointGuard:
    with _REGISTRY_LOCK:
        state = _CHECKPOINTS.get(checkpoint)
        guard = _CHECKPOINT_GUARDS.get(checkpoint)
        failed = checkpoint in _FAILED_CHECKPOINTS
        if guard is None:
            _FAILED_CHECKPOINTS.add(checkpoint)
    if guard is None:
        if state is not None:
            _terminal_fail(checkpoint, state)
        raise Experiment002CheckpointError("checkpoint guard was not issued here")
    if failed:
        if state is not None:
            _terminal_fail(checkpoint, state)
        raise Experiment002CheckpointError("checkpoint is terminally failed")
    return guard


def _issued_serialized_authority(
    checkpoint: _UnregisteredSerializedCheckpoint,
) -> tuple[_SerializedState, _SerializedGuard]:
    if type(checkpoint) is not _UnregisteredSerializedCheckpoint:
        raise TypeError("checkpoint must be an exact _UnregisteredSerializedCheckpoint")
    with _REGISTRY_LOCK:
        state = _SERIALIZED.get(checkpoint)
        guard = _SERIALIZED_GUARDS.get(checkpoint)
        anchored = checkpoint in _ISSUED_SERIALIZED
        failed = checkpoint in _FAILED_SERIALIZED
        if state is None or guard is None or not anchored:
            _FAILED_SERIALIZED.add(checkpoint)
    if state is None or guard is None or not anchored or failed:
        raise Experiment002CheckpointError("serialized checkpoint was not issued here")
    return state, guard


def _validate_checkpoint_authority(
    state: _CheckpointState,
    guard: _CheckpointGuard,
) -> None:
    if (
        state.token is not guard.token
        or state.route_marker is not _UNREGISTERED_ROUTE_MARKER
        or guard.route_marker is not _UNREGISTERED_ROUTE_MARKER
        or state.names is not guard.names
        or state.tensors is not guard.tensors
        or state.tensor_count != guard.tensor_count
        or state.value_count != guard.value_count
        or state.model_tensor_sha256 != guard.model_tensor_sha256
    ):
        raise Experiment002CheckpointError("checkpoint authority changed")
    _require_stable_tensor_population(
        state.names,
        state.tensors,
        value_count=state.value_count,
    )
    observed_sha256 = _model_tensor_sha256(state.names, state.tensors)
    if observed_sha256 != state.model_tensor_sha256:
        raise Experiment002CheckpointError("preserved checkpoint tensors changed")


def _serialized_guard_from_state(state: _SerializedState) -> _SerializedGuard:
    return _SerializedGuard(
        token=state.token,
        route_marker=state.route_marker,
        tensor_count=state.tensor_count,
        value_count=state.value_count,
        model_tensor_sha256=state.model_tensor_sha256,
        safetensors_sha256=state.safetensors_sha256,
        payload=state.payload,
    )


def _validate_serialized_state(
    state: _SerializedState,
    guard: _SerializedGuard,
) -> None:
    if (
        state.token is not guard.token
        or state.route_marker is not _UNREGISTERED_ROUTE_MARKER
        or guard.route_marker is not _UNREGISTERED_ROUTE_MARKER
        or state.tensor_count != guard.tensor_count
        or state.value_count != guard.value_count
        or state.model_tensor_sha256 != guard.model_tensor_sha256
        or state.safetensors_sha256 != guard.safetensors_sha256
        or state.payload is not guard.payload
    ):
        raise Experiment002CheckpointError("serialized checkpoint authority changed")
    if type(state.tensor_count) is not int or not (
        1 <= state.tensor_count <= _MAX_UNREGISTERED_TENSORS
    ):
        raise Experiment002CheckpointError("serialized checkpoint counts are invalid")
    if type(state.value_count) is not int or not (
        1 <= state.value_count <= _MAX_UNREGISTERED_VALUES
    ):
        raise Experiment002CheckpointError("serialized checkpoint counts are invalid")
    if not _is_sha256(state.model_tensor_sha256) or not _is_sha256(
        state.safetensors_sha256
    ):
        raise Experiment002CheckpointError("serialized checkpoint digest is invalid")
    if type(state.payload) is not bytes or hashlib.sha256(
        state.payload
    ).hexdigest() != (state.safetensors_sha256):
        raise Experiment002CheckpointError("serialized checkpoint payload changed")


def _terminal_fail(
    checkpoint: _UnregisteredCheckpoint,
    state: _CheckpointState,
) -> None:
    state.phase = "FAILED"
    with _REGISTRY_LOCK:
        _FAILED_CHECKPOINTS.add(checkpoint)


def _terminal_fail_serialized(
    checkpoint: _UnregisteredSerializedCheckpoint,
) -> None:
    with _REGISTRY_LOCK:
        _FAILED_SERIALIZED.add(checkpoint)


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
