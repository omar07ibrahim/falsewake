"""Registered in-memory winner checkpoint for Experiment 002.

Only a fully verified completed training history can reach this module.  The
rank-one evaluated epoch is selected from that history, its issuer-owned model
snapshot is cloned, and deterministic safetensors bytes are checked without
ever opening or creating a filesystem object.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import threading
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final, Literal, NoReturn, SupportsIndex, cast

import safetensors as _safetensors
import safetensors.torch as _safetensors_torch
import torch
from torch import Tensor

from falsewake.experiment_002_registered_evaluator import (
    FINAL_ZERO_BASED_EPOCH,
    VALIDATION_BATCH_COUNT,
    RegisteredEvaluatedEpoch,
    _registered_evaluated_epoch_snapshot,
    _RegisteredEvaluatedEpochSnapshot,
)
from falsewake.experiment_002_registered_history import (
    RegisteredCompletedTrainingHistory,
    RegisteredEpochRank,
    _registered_completed_history_snapshot,
    _RegisteredCompletedHistorySnapshot,
)
from falsewake.experiment_002_run_authority import (
    VerifiedRunRegistration,
    reverify_verified_run_registration,
)
from falsewake.experiment_002_training_evidence import (
    EPOCH_COUNT,
    MODEL_PARAMETER_VALUE_COUNT,
    MODEL_TENSOR_COUNT,
    MODEL_TENSOR_DOMAIN,
    UPDATES_PER_EPOCH,
    RegisteredModelTensorEvidence,
    _frame_model_tensors,
    _model_tensor_clones,
    verify_registered_model_tensor_evidence,
)

_SAFETENSORS_VERSION: Final = "0.8.0"
_F32LE_TAG: Final = "F32"
_MAX_HEADER_BYTES: Final = 64 * 1024
_MAX_NAME_BYTES: Final = 512
_LOWER_HEX: Final = frozenset("0123456789abcdef")
_ROUTE_MARKER: Final = object()
_EVENT_TYPE: Final = type(threading.Event())

type _AttemptPhase = Literal["SERIALIZING", "COMPLETE", "FAILED"]


class Experiment002RegisteredCheckpointError(ValueError):
    """The registered winner checkpoint violated its exact authority contract."""


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
class RegisteredSerializedCheckpoint:
    """Opaque terminal capability for verified in-memory safetensors bytes."""

    _seed: int
    _zero_based_epoch: int
    _history_sha256: str
    _model_tensor_sha256: str
    _safetensors_sha256: str
    _safetensors_byte_count: int
    _token: object
    _route_marker: object

    def __init__(self) -> None:
        raise TypeError("registered serialized checkpoints are issuer-only")

    @property
    def seed(self) -> int:
        return _verified_serialized_state(self).binding.seed

    @property
    def zero_based_epoch(self) -> int:
        return _verified_serialized_state(self).binding.zero_based_epoch

    @property
    def history_sha256(self) -> str:
        return _verified_serialized_state(self).binding.history_sha256

    @property
    def model_tensor_sha256(self) -> str:
        return _verified_serialized_state(self).binding.model_tensor_sha256

    @property
    def safetensors_sha256(self) -> str:
        return _verified_serialized_state(self).safetensors_sha256

    @property
    def safetensors_byte_count(self) -> int:
        return len(_verified_serialized_state(self).payload)

    @property
    def safetensors_bytes(self) -> bytes:
        """Return immutable bytes only after a fresh end-to-end verification."""

        return _verified_serialized_state(self).payload

    def __copy__(self) -> NoReturn:
        raise TypeError("registered serialized checkpoints cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        del memo
        raise TypeError("registered serialized checkpoints cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("registered serialized checkpoints cannot be serialized")

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        del protocol
        raise TypeError("registered serialized checkpoints cannot be serialized")


@dataclass(frozen=True, slots=True)
class _SourceBinding:
    completed_history: RegisteredCompletedTrainingHistory
    registration: VerifiedRunRegistration
    history: object
    executor: object
    validation_inputs: object
    complete_update_trace: object
    winner: RegisteredEvaluatedEpoch
    evaluated_handoff: object
    validation_evidence: object
    model_tensors: RegisteredModelTensorEvidence
    process_id: int
    seed: int
    epoch_count: int
    zero_based_epoch: int
    optimizer_generation: int
    batch_count: int
    another_training_epoch: bool
    registration_head_commit: str
    registration_sha256: str
    source_bundle_sha256: str
    validation_inputs_sha256: str
    validation_predictions_sha256: str
    model_tensor_sha256: str
    optimizer_sha256: str
    torch_rng_sha256: str
    evaluated_authority_sha256: str
    complete_update_trace_sha256: str
    canonical_json_bytes: bytes
    history_sha256: str
    ranked_epochs: tuple[RegisteredEpochRank, ...]
    evaluated_epochs: tuple[RegisteredEvaluatedEpoch, ...]


@dataclass(frozen=True, slots=True)
class _TensorSpec:
    name: str
    shape: tuple[int, ...]
    payload: bytes


@dataclass(frozen=True, slots=True)
class _SerializedState:
    token: object
    route_marker: object
    binding: _SourceBinding
    specs: tuple[_TensorSpec, ...]
    model_payload: bytes
    payload: bytes
    safetensors_sha256: str


@dataclass(frozen=True, slots=True)
class _SerializedGuard:
    token: object
    route_marker: object
    binding: _SourceBinding
    binding_frame: tuple[object, ...]
    specs: tuple[_TensorSpec, ...]
    specs_frame: tuple[object, ...]
    model_payload: bytes
    payload: bytes
    safetensors_sha256: str


@dataclass(frozen=True, slots=True)
class _SerializationOutcome:
    result: RegisteredSerializedCheckpoint | None
    error: BaseException | None


@dataclass(frozen=True, slots=True)
class _SerializationAdmission:
    token: object
    history: RegisteredCompletedTrainingHistory
    leader: bool
    completed: threading.Event
    outcome: list[_SerializationOutcome]


@dataclass(frozen=True, slots=True)
class _AttemptRecord:
    token: object
    completed: threading.Event
    outcome: list[_SerializationOutcome]
    phase: _AttemptPhase
    outcome_frame: tuple[object, ...] | None


@dataclass(frozen=True, slots=True)
class _TruthRoutes:
    begin: Callable[
        [RegisteredCompletedTrainingHistory, object], _SerializationAdmission
    ]
    recover: Callable[
        [RegisteredCompletedTrainingHistory, object],
        _SerializationAdmission | None,
    ]
    validate_admission: Callable[[_SerializationAdmission], bool]
    admission_role: Callable[[_SerializationAdmission], bool]
    terminal_outcome: Callable[[_SerializationAdmission], _SerializationOutcome]
    finish: Callable[[_SerializationAdmission, _SerializationOutcome], None]
    record_issued: Callable[
        [RegisteredSerializedCheckpoint, _SerializedState, _SerializedGuard], None
    ]
    validate_issued: Callable[[RegisteredSerializedCheckpoint, object, object], bool]
    fail_issued: Callable[[RegisteredSerializedCheckpoint], None]


_SERIALIZED: weakref.WeakKeyDictionary[
    RegisteredSerializedCheckpoint, _SerializedState
] = weakref.WeakKeyDictionary()
_SERIALIZED_GUARDS: weakref.WeakKeyDictionary[
    RegisteredSerializedCheckpoint, _SerializedGuard
] = weakref.WeakKeyDictionary()
_ISSUED_SERIALIZED: weakref.WeakSet[RegisteredSerializedCheckpoint] = weakref.WeakSet()
_FAILED_SERIALIZED: weakref.WeakSet[RegisteredSerializedCheckpoint] = weakref.WeakSet()
_REGISTRY_LOCK = threading.RLock()


def serialize_registered_checkpoint(
    completed_history: RegisteredCompletedTrainingHistory,
) -> RegisteredSerializedCheckpoint:
    """Serialize the exact rank-one model from one completed registered history."""

    if type(completed_history) is not RegisteredCompletedTrainingHistory:
        raise TypeError(
            "completed_history must be an exact RegisteredCompletedTrainingHistory"
        )
    caller_token = object()
    admission: _SerializationAdmission | None = None
    try:
        admission = _begin_serialization_admission(
            completed_history,
            caller_token,
        )
    except BaseException as begin_error:
        recovered = _recover_serialization_admission(
            completed_history,
            caller_token,
        )
        if recovered is not None:
            _publish_serialization_outcome(
                recovered,
                _SerializationOutcome(None, begin_error),
            )
        raise

    return _run_admitted_serialization(
        admission,
        completed_history,
        caller_token,
    )


def _run_admitted_serialization(
    admission: _SerializationAdmission,
    completed_history: RegisteredCompletedTrainingHistory,
    caller_token: object,
) -> RegisteredSerializedCheckpoint:
    leader_outcome: _SerializationOutcome | None = None
    try:
        leader = _require_admission(admission, completed_history)
        if not leader:
            admission.completed.wait()
            terminal = _require_terminal_outcome(admission)
            if terminal.error is not None:
                raise terminal.error
            assert terminal.result is not None
            verify_registered_serialized_checkpoint(terminal.result)
            return terminal.result

        result: RegisteredSerializedCheckpoint | None = None
        error: BaseException | None = None
        try:
            result = _serialize_leader(completed_history)
        except BaseException as caught:
            error = caught
        leader_outcome = _SerializationOutcome(result, error)
        _publish_serialization_outcome(admission, leader_outcome)
        terminal = _require_terminal_outcome(admission)
        if terminal.error is not None:
            raise terminal.error
        assert terminal.result is not None
        verify_registered_serialized_checkpoint(terminal.result)
        return terminal.result
    except BaseException as boundary_error:
        recovered = _recover_serialization_admission(
            completed_history,
            caller_token,
        )
        if recovered is not None and not recovered.completed.is_set():
            fallback = (
                leader_outcome
                if leader_outcome is not None
                else _SerializationOutcome(None, boundary_error)
            )
            try:
                _recover_finish_serialization_admission(recovered, fallback)
            finally:
                recovered.completed.set()
                admission.completed.set()
        raise


def verify_registered_serialized_checkpoint(
    checkpoint: RegisteredSerializedCheckpoint,
) -> None:
    """Fully reverify source authority, immutable bytes, header, and roundtrip."""

    _verified_serialized_state(checkpoint)


def _serialize_leader(
    completed_history: RegisteredCompletedTrainingHistory,
) -> RegisteredSerializedCheckpoint:
    rng_before = _torch_rng_state()
    try:
        return _serialize_leader_with_rng(completed_history, rng_before)
    finally:
        _require_rng_unchanged(rng_before)


def _serialize_leader_with_rng(
    completed_history: RegisteredCompletedTrainingHistory,
    rng_before: bytes,
) -> RegisteredSerializedCheckpoint:
    _require_safetensors_runtime()
    binding, names, tensors = _capture_source(completed_history)
    _require_rng_unchanged(rng_before)
    specs = _tensor_specs(names, tensors)
    model_payload = _frame_model_tensors(names, tensors)
    if hashlib.sha256(MODEL_TENSOR_DOMAIN + model_payload).hexdigest() != (
        binding.model_tensor_sha256
    ):
        raise Experiment002RegisteredCheckpointError(
            "winner model tensor digest changed before serialization"
        )

    first = _serialize_once(names, tensors)
    _require_rng_unchanged(rng_before)
    _require_safetensors_payload(first, specs=specs)
    second = _serialize_once(names, tensors)
    _require_rng_unchanged(rng_before)
    _require_safetensors_payload(second, specs=specs)
    if first != second:
        raise Experiment002RegisteredCheckpointError(
            "safetensors serialization was not byte deterministic"
        )
    _verify_safetensors_roundtrip(
        first,
        specs=specs,
        expected_model_sha256=binding.model_tensor_sha256,
    )
    _require_rng_unchanged(rng_before)

    second_binding, second_names, second_tensors = _capture_source(completed_history)
    _require_rng_unchanged(rng_before)
    _require_same_source_binding(binding, second_binding)
    if second_names != names or _tensor_specs(second_names, second_tensors) != specs:
        raise Experiment002RegisteredCheckpointError(
            "winner model tensors changed across serialization"
        )
    _require_rng_unchanged(rng_before)
    result: RegisteredSerializedCheckpoint | None = None
    try:
        result = _issue_serialized_checkpoint(
            binding=binding,
            specs=specs,
            model_payload=model_payload,
            payload=first,
        )
        _verify_serialized_state(result, reverify_source=True)
        _require_rng_unchanged(rng_before)
        return result
    except BaseException:
        if result is not None:
            _terminal_fail_serialized(result)
        raise


def _capture_source(
    completed_history: RegisteredCompletedTrainingHistory,
) -> tuple[_SourceBinding, tuple[str, ...], tuple[Tensor, ...]]:
    first_history = _registered_completed_history_snapshot(completed_history)
    _require_completed_snapshot(completed_history, first_history)
    first_evaluated = _registered_evaluated_epoch_snapshot(first_history.winner)
    binding = _source_binding(completed_history, first_history, first_evaluated)
    verify_registered_model_tensor_evidence(
        first_evaluated.model_tensors,
        seed=first_evaluated.seed,
        zero_based_epoch=first_evaluated.zero_based_epoch,
    )
    named_tensors = _model_tensor_clones(first_evaluated.model_tensors)
    names, tensors = _validate_cloned_tensors(named_tensors)
    verify_registered_model_tensor_evidence(
        first_evaluated.model_tensors,
        seed=first_evaluated.seed,
        zero_based_epoch=first_evaluated.zero_based_epoch,
    )

    second_history = _registered_completed_history_snapshot(completed_history)
    _require_completed_snapshot(completed_history, second_history)
    second_evaluated = _registered_evaluated_epoch_snapshot(second_history.winner)
    second_binding = _source_binding(
        completed_history,
        second_history,
        second_evaluated,
    )
    _require_same_source_binding(binding, second_binding)
    verify_registered_model_tensor_evidence(
        second_evaluated.model_tensors,
        seed=second_evaluated.seed,
        zero_based_epoch=second_evaluated.zero_based_epoch,
    )
    observed_model_sha256 = hashlib.sha256(
        MODEL_TENSOR_DOMAIN + _frame_model_tensors(names, tensors)
    ).hexdigest()
    if observed_model_sha256 != binding.model_tensor_sha256:
        raise Experiment002RegisteredCheckpointError(
            "cloned winner model tensor digest differs from evaluated evidence"
        )
    return binding, names, tensors


def _source_binding(
    completed_history: RegisteredCompletedTrainingHistory,
    history: _RegisteredCompletedHistorySnapshot,
    evaluated: _RegisteredEvaluatedEpochSnapshot,
) -> _SourceBinding:
    _require_completed_snapshot(completed_history, history)
    if type(evaluated) is not _RegisteredEvaluatedEpochSnapshot:
        raise Experiment002RegisteredCheckpointError(
            "winner evaluated snapshot has an invalid type"
        )
    winner_epoch = history.ranked_epochs[0].zero_based_epoch
    if (
        evaluated.registration is not history.registration
        or evaluated.validation_inputs is not history.validation_inputs
        or evaluated.process_id != history.process_id
        or evaluated.seed != history.seed
        or evaluated.zero_based_epoch != winner_epoch
        or evaluated.optimizer_generation
        != (evaluated.zero_based_epoch + 1) * UPDATES_PER_EPOCH
        or evaluated.batch_count != VALIDATION_BATCH_COUNT
        or evaluated.another_training_epoch
        is not (evaluated.zero_based_epoch < FINAL_ZERO_BASED_EPOCH)
        or evaluated.validation_inputs_sha256 != history.validation_inputs_sha256
    ):
        raise Experiment002RegisteredCheckpointError(
            "winner evaluated evidence differs from completed history"
        )
    reverify_verified_run_registration(history.registration)
    if (
        evaluated.registration_head_commit != history.registration.head_commit
        or evaluated.registration_sha256 != history.registration.registration_sha256
        or evaluated.source_bundle_sha256 != history.registration.source_bundle_sha256
        or evaluated.model_tensor_sha256 != evaluated.model_tensors.sha256
        or evaluated.model_tensors.seed != history.seed
        or evaluated.model_tensors.zero_based_epoch != winner_epoch
        or evaluated.model_tensors.tensor_count != MODEL_TENSOR_COUNT
        or evaluated.model_tensors.value_count != MODEL_PARAMETER_VALUE_COUNT
    ):
        raise Experiment002RegisteredCheckpointError(
            "winner source or model evidence binding changed"
        )
    for value, name in (
        (evaluated.registration_sha256, "registration sha256"),
        (evaluated.source_bundle_sha256, "source bundle sha256"),
        (evaluated.validation_inputs_sha256, "validation inputs sha256"),
        (evaluated.validation_predictions_sha256, "validation predictions sha256"),
        (evaluated.model_tensor_sha256, "model tensor sha256"),
        (evaluated.optimizer_sha256, "optimizer sha256"),
        (evaluated.torch_rng_sha256, "Torch RNG sha256"),
        (evaluated.authority_sha256, "evaluated authority sha256"),
        (history.complete_update_trace_sha256, "complete update trace sha256"),
        (history.history_sha256, "history sha256"),
    ):
        _require_sha256(value, name)
    return _SourceBinding(
        completed_history=completed_history,
        registration=history.registration,
        history=history.history,
        executor=history.executor,
        validation_inputs=history.validation_inputs,
        complete_update_trace=history.complete_update_trace,
        winner=history.winner,
        evaluated_handoff=evaluated.handoff,
        validation_evidence=evaluated.validation_evidence,
        model_tensors=evaluated.model_tensors,
        process_id=history.process_id,
        seed=history.seed,
        epoch_count=history.epoch_count,
        zero_based_epoch=evaluated.zero_based_epoch,
        optimizer_generation=evaluated.optimizer_generation,
        batch_count=evaluated.batch_count,
        another_training_epoch=evaluated.another_training_epoch,
        registration_head_commit=evaluated.registration_head_commit,
        registration_sha256=evaluated.registration_sha256,
        source_bundle_sha256=evaluated.source_bundle_sha256,
        validation_inputs_sha256=evaluated.validation_inputs_sha256,
        validation_predictions_sha256=evaluated.validation_predictions_sha256,
        model_tensor_sha256=evaluated.model_tensor_sha256,
        optimizer_sha256=evaluated.optimizer_sha256,
        torch_rng_sha256=evaluated.torch_rng_sha256,
        evaluated_authority_sha256=evaluated.authority_sha256,
        complete_update_trace_sha256=history.complete_update_trace_sha256,
        canonical_json_bytes=history.canonical_json_bytes,
        history_sha256=history.history_sha256,
        ranked_epochs=history.ranked_epochs,
        evaluated_epochs=history.evaluated_epochs,
    )


def _require_completed_snapshot(
    completed_history: RegisteredCompletedTrainingHistory,
    snapshot: _RegisteredCompletedHistorySnapshot,
) -> None:
    if type(snapshot) is not _RegisteredCompletedHistorySnapshot:
        raise Experiment002RegisteredCheckpointError(
            "completed history snapshot has an invalid type"
        )
    if (
        snapshot.process_id != os.getpid()
        or snapshot.epoch_count != EPOCH_COUNT
        or type(snapshot.canonical_json_bytes) is not bytes
        or not snapshot.canonical_json_bytes
        or len(snapshot.ranked_epochs) != EPOCH_COUNT
        or len(snapshot.evaluated_epochs) != EPOCH_COUNT
        or len({rank.zero_based_epoch for rank in snapshot.ranked_epochs})
        != EPOCH_COUNT
        or snapshot.winner
        is not snapshot.evaluated_epochs[snapshot.ranked_epochs[0].zero_based_epoch]
        or snapshot.history_sha256 != completed_history.history_sha256
        or snapshot.seed != completed_history.seed
    ):
        raise Experiment002RegisteredCheckpointError(
            "completed history does not bind one exact ranked winner"
        )
    if any(type(rank) is not RegisteredEpochRank for rank in snapshot.ranked_epochs):
        raise Experiment002RegisteredCheckpointError(
            "completed history ranks have invalid types"
        )


def _require_same_source_binding(left: _SourceBinding, right: _SourceBinding) -> None:
    if type(left) is not _SourceBinding or type(right) is not _SourceBinding:
        raise Experiment002RegisteredCheckpointError(
            "checkpoint source binding has an invalid type"
        )
    identity_fields = (
        "completed_history",
        "registration",
        "history",
        "executor",
        "validation_inputs",
        "complete_update_trace",
        "winner",
        "evaluated_handoff",
        "validation_evidence",
        "model_tensors",
    )
    if any(
        getattr(left, field) is not getattr(right, field) for field in identity_fields
    ):
        raise Experiment002RegisteredCheckpointError(
            "checkpoint source capability identity changed"
        )
    if not _same_evaluated_epoch_identities(
        left.evaluated_epochs,
        right.evaluated_epochs,
    ):
        raise Experiment002RegisteredCheckpointError(
            "completed history evaluated-epoch identities changed"
        )
    value_fields = (
        "process_id",
        "seed",
        "epoch_count",
        "zero_based_epoch",
        "optimizer_generation",
        "batch_count",
        "another_training_epoch",
        "registration_head_commit",
        "registration_sha256",
        "source_bundle_sha256",
        "validation_inputs_sha256",
        "validation_predictions_sha256",
        "model_tensor_sha256",
        "optimizer_sha256",
        "torch_rng_sha256",
        "evaluated_authority_sha256",
        "complete_update_trace_sha256",
        "canonical_json_bytes",
        "history_sha256",
        "ranked_epochs",
    )
    if any(
        type(getattr(left, field)) is not type(getattr(right, field))
        or getattr(left, field) != getattr(right, field)
        for field in value_fields
    ):
        raise Experiment002RegisteredCheckpointError(
            "checkpoint source scalar binding changed"
        )


def _same_evaluated_epoch_identities(
    left: tuple[RegisteredEvaluatedEpoch, ...],
    right: tuple[RegisteredEvaluatedEpoch, ...],
) -> bool:
    return (
        type(left) is tuple
        and type(right) is tuple
        and len(left) == len(right)
        and all(one is two for one, two in zip(left, right, strict=True))
    )


def _validate_cloned_tensors(
    named_tensors: tuple[tuple[str, Tensor], ...],
) -> tuple[tuple[str, ...], tuple[Tensor, ...]]:
    if type(named_tensors) is not tuple or len(named_tensors) != MODEL_TENSOR_COUNT:
        raise Experiment002RegisteredCheckpointError(
            "winner model tensor count is invalid"
        )
    names: list[str] = []
    tensors: list[Tensor] = []
    storage_ids: set[int] = set()
    value_count = 0
    for item in named_tensors:
        if type(item) is not tuple or len(item) != 2:
            raise Experiment002RegisteredCheckpointError(
                "winner model tensor entry is invalid"
            )
        name, tensor = item
        if type(name) is not str or not name or name == "__metadata__":
            raise Experiment002RegisteredCheckpointError(
                "winner model tensor name is invalid"
            )
        try:
            encoded = name.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            raise Experiment002RegisteredCheckpointError(
                "winner model tensor name is not UTF-8"
            ) from error
        if len(encoded) > _MAX_NAME_BYTES:
            raise Experiment002RegisteredCheckpointError(
                "winner model tensor name is too long"
            )
        _require_owned_f32_tensor(tensor, name)
        storage_id = tensor.untyped_storage()._cdata
        if storage_id in storage_ids:
            raise Experiment002RegisteredCheckpointError(
                "winner model tensor clones alias storage"
            )
        storage_ids.add(storage_id)
        names.append(name)
        tensors.append(tensor)
        value_count += tensor.numel()
    names_tuple = tuple(names)
    tensors_tuple = tuple(tensors)
    if (
        len(set(names_tuple)) != MODEL_TENSOR_COUNT
        or names_tuple
        != tuple(sorted(names_tuple, key=lambda value: value.encode("utf-8")))
        or value_count != MODEL_PARAMETER_VALUE_COUNT
    ):
        raise Experiment002RegisteredCheckpointError(
            "winner model tensor population is not canonical"
        )
    return names_tuple, tensors_tuple


def _require_owned_f32_tensor(tensor: Tensor, name: str) -> None:
    if type(tensor) is not Tensor:
        raise TypeError(f"winner model tensor {name!r} must be an exact Tensor")
    if (
        tensor.device.type != "cpu"
        or tensor.dtype != torch.float32
        or tensor.layout != torch.strided
        or not tensor.is_contiguous()
        or tensor.requires_grad
        or tensor.grad_fn is not None
        or not tensor.is_leaf
        or tensor.storage_offset() != 0
        or tensor.untyped_storage().nbytes() != tensor.numel() * tensor.element_size()
        or tensor._base is not None
        or not bool(torch.isfinite(tensor).all())
    ):
        raise Experiment002RegisteredCheckpointError(
            f"winner model tensor {name!r} is not an owned finite CPU float32 clone"
        )


def _tensor_specs(
    names: tuple[str, ...], tensors: tuple[Tensor, ...]
) -> tuple[_TensorSpec, ...]:
    return tuple(
        _TensorSpec(
            name=name,
            shape=tuple(tensor.shape),
            payload=_tensor_payload(tensor),
        )
        for name, tensor in zip(names, tensors, strict=True)
    )


def _serialize_once(names: tuple[str, ...], tensors: tuple[Tensor, ...]) -> bytes:
    mapping: dict[str, Tensor] = {}
    storage_ids: set[int] = set()
    for name, source in zip(names, tensors, strict=True):
        clone = source.detach().clone(memory_format=torch.contiguous_format)
        _require_owned_f32_tensor(clone, name)
        storage_id = clone.untyped_storage()._cdata
        if storage_id in storage_ids:
            raise Experiment002RegisteredCheckpointError(
                "serialization tensor clones alias storage"
            )
        storage_ids.add(storage_id)
        mapping[name] = clone
    payload = _safetensors_torch.save(mapping, metadata=None)
    if type(payload) is not bytes:
        raise Experiment002RegisteredCheckpointError(
            "safetensors save did not return exact bytes"
        )
    return payload


def _require_safetensors_runtime() -> None:
    if _safetensors.__version__ != _SAFETENSORS_VERSION:
        raise Experiment002RegisteredCheckpointError(
            "safetensors runtime version differs from the frozen serializer"
        )


def _require_safetensors_payload(
    payload: bytes,
    *,
    specs: tuple[_TensorSpec, ...],
) -> None:
    if type(payload) is not bytes or len(payload) < 8:
        raise Experiment002RegisteredCheckpointError("safetensors payload is truncated")
    header_size = struct.unpack("<Q", payload[:8])[0]
    if (
        header_size == 0
        or header_size % 8 != 0
        or header_size > _MAX_HEADER_BYTES
        or 8 + header_size > len(payload)
    ):
        raise Experiment002RegisteredCheckpointError(
            "safetensors header size is invalid"
        )
    padded_header = payload[8 : 8 + header_size]
    header_bytes = padded_header.rstrip(b" ")
    padding = padded_header[len(header_bytes) :]
    if not header_bytes or len(padding) > 7 or padding != b" " * len(padding):
        raise Experiment002RegisteredCheckpointError(
            "safetensors header padding is invalid"
        )
    try:
        decoded = header_bytes.decode("utf-8", errors="strict")
        parsed = json.loads(
            decoded,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_number,
            parse_float=_reject_json_number,
        )
    except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise Experiment002RegisteredCheckpointError(
            "safetensors header is not exact JSON"
        ) from error
    if type(parsed) is not dict or "__metadata__" in parsed:
        raise Experiment002RegisteredCheckpointError(
            "safetensors metadata must be null"
        )
    header = cast(dict[str, object], parsed)
    expected_names = tuple(spec.name for spec in specs)
    if tuple(header) != expected_names:
        raise Experiment002RegisteredCheckpointError(
            "safetensors header names are not in canonical UTF-8 order"
        )

    expected_offset = 0
    expected_data = bytearray()
    for spec in specs:
        record_object = header[spec.name]
        if type(record_object) is not dict:
            raise Experiment002RegisteredCheckpointError(
                "safetensors tensor record is invalid"
            )
        record = cast(dict[str, object], record_object)
        if set(record) != {"dtype", "shape", "data_offsets"}:
            raise Experiment002RegisteredCheckpointError(
                "safetensors tensor record schema changed"
            )
        shape = record["shape"]
        offsets = record["data_offsets"]
        if (
            type(record["dtype"]) is not str
            or record["dtype"] != _F32LE_TAG
            or type(shape) is not list
            or type(offsets) is not list
            or len(offsets) != 2
            or any(type(value) is not int or value < 0 for value in shape)
            or any(type(value) is not int or value < 0 for value in offsets)
        ):
            raise Experiment002RegisteredCheckpointError(
                "safetensors tensor metadata is invalid"
            )
        expected_end = expected_offset + len(spec.payload)
        if tuple(shape) != spec.shape or offsets != [expected_offset, expected_end]:
            raise Experiment002RegisteredCheckpointError(
                "safetensors tensor shape or contiguous offsets changed"
            )
        expected_data.extend(spec.payload)
        expected_offset = expected_end
    data = payload[8 + header_size :]
    if (
        len(data) != expected_offset
        or data != bytes(expected_data)
        or len(payload) != 8 + header_size + expected_offset
    ):
        raise Experiment002RegisteredCheckpointError(
            "safetensors data section differs from model tensor bytes"
        )


def _verify_safetensors_roundtrip(
    payload: bytes,
    *,
    specs: tuple[_TensorSpec, ...],
    expected_model_sha256: str,
) -> None:
    try:
        loaded_object = _safetensors_torch.load(payload)
    except Exception as error:
        raise Experiment002RegisteredCheckpointError(
            "safetensors in-memory roundtrip failed"
        ) from error
    if type(loaded_object) is not dict:
        raise Experiment002RegisteredCheckpointError(
            "safetensors load returned an invalid mapping"
        )
    loaded = loaded_object
    names = tuple(spec.name for spec in specs)
    if tuple(sorted(loaded, key=lambda value: value.encode("utf-8"))) != names:
        raise Experiment002RegisteredCheckpointError(
            "safetensors roundtrip names changed"
        )
    tensors: list[Tensor] = []
    for spec in specs:
        observed = loaded[spec.name]
        if type(observed) is not Tensor:
            raise Experiment002RegisteredCheckpointError(
                "safetensors roundtrip returned a non-Tensor value"
            )
        if (
            observed.device.type != "cpu"
            or observed.dtype != torch.float32
            or observed.layout != torch.strided
            or not observed.is_contiguous()
            or tuple(observed.shape) != spec.shape
            or _tensor_payload(observed) != spec.payload
            or not bool(torch.isfinite(observed).all())
        ):
            raise Experiment002RegisteredCheckpointError(
                "safetensors roundtrip changed a model tensor"
            )
        tensors.append(observed)
    if (
        hashlib.sha256(
            MODEL_TENSOR_DOMAIN + _frame_model_tensors(names, tuple(tensors))
        ).hexdigest()
        != expected_model_sha256
    ):
        raise Experiment002RegisteredCheckpointError(
            "safetensors roundtrip changed the model tensor digest"
        )


def _issue_serialized_checkpoint(
    *,
    binding: _SourceBinding,
    specs: tuple[_TensorSpec, ...],
    model_payload: bytes,
    payload: bytes,
) -> RegisteredSerializedCheckpoint:
    token = object()
    result = object.__new__(RegisteredSerializedCheckpoint)
    try:
        safetensors_sha256 = hashlib.sha256(payload).hexdigest()
        object.__setattr__(result, "_seed", binding.seed)
        object.__setattr__(result, "_zero_based_epoch", binding.zero_based_epoch)
        object.__setattr__(result, "_history_sha256", binding.history_sha256)
        object.__setattr__(
            result,
            "_model_tensor_sha256",
            binding.model_tensor_sha256,
        )
        object.__setattr__(result, "_safetensors_sha256", safetensors_sha256)
        object.__setattr__(result, "_safetensors_byte_count", len(payload))
        object.__setattr__(result, "_token", token)
        object.__setattr__(result, "_route_marker", _ROUTE_MARKER)
        state = _SerializedState(
            token=token,
            route_marker=_ROUTE_MARKER,
            binding=binding,
            specs=specs,
            model_payload=model_payload,
            payload=payload,
            safetensors_sha256=safetensors_sha256,
        )
        guard = _SerializedGuard(
            token=token,
            route_marker=_ROUTE_MARKER,
            binding=binding,
            binding_frame=_source_binding_frame(binding),
            specs=specs,
            specs_frame=_specs_frame(specs),
            model_payload=model_payload,
            payload=payload,
            safetensors_sha256=safetensors_sha256,
        )
        _record_serialized_truth(result, state, guard)
        with _REGISTRY_LOCK:
            _SERIALIZED[result] = state
            _SERIALIZED_GUARDS[result] = guard
            _ISSUED_SERIALIZED.add(result)
        if not _validate_serialized_truth(result, state, guard):
            raise Experiment002RegisteredCheckpointError(
                "serialized checkpoint closure truth changed during issuance"
            )
        return result
    except BaseException:
        _recover_fail_serialized_truth(result)
        with _REGISTRY_LOCK:
            _FAILED_SERIALIZED.add(result)
        raise


def _verified_serialized_state(
    checkpoint: RegisteredSerializedCheckpoint,
) -> _SerializedState:
    state, guard = _issued_serialized_authority(checkpoint)
    try:
        _validate_serialized_authority(checkpoint, state, guard)
        _verify_serialized_state(checkpoint, reverify_source=True)
    except BaseException:
        _terminal_fail_serialized(checkpoint)
        raise
    return state


def _verify_serialized_state(
    checkpoint: RegisteredSerializedCheckpoint,
    *,
    reverify_source: bool,
) -> None:
    rng_before = _torch_rng_state()
    try:
        _verify_serialized_state_with_rng(
            checkpoint,
            reverify_source=reverify_source,
            rng_before=rng_before,
        )
    finally:
        _require_rng_unchanged(rng_before)


def _verify_serialized_state_with_rng(
    checkpoint: RegisteredSerializedCheckpoint,
    *,
    reverify_source: bool,
    rng_before: bytes,
) -> None:
    state, guard = _issued_serialized_authority(checkpoint)
    _validate_serialized_authority(checkpoint, state, guard)
    _require_safetensors_runtime()
    if reverify_source:
        binding, names, tensors = _capture_source(state.binding.completed_history)
        _require_same_source_binding(state.binding, binding)
        specs = _tensor_specs(names, tensors)
        if specs != state.specs or _frame_model_tensors(names, tensors) != (
            state.model_payload
        ):
            raise Experiment002RegisteredCheckpointError(
                "serialized checkpoint no longer matches its winner model"
            )
        _require_rng_unchanged(rng_before)
    _require_safetensors_payload(state.payload, specs=state.specs)
    _verify_safetensors_roundtrip(
        state.payload,
        specs=state.specs,
        expected_model_sha256=state.binding.model_tensor_sha256,
    )
    _require_rng_unchanged(rng_before)
    if _issued_serialized_authority(checkpoint)[0] is not state:
        raise Experiment002RegisteredCheckpointError(
            "serialized checkpoint state changed during verification"
        )
    _require_rng_unchanged(rng_before)


def _issued_serialized_authority(
    checkpoint: RegisteredSerializedCheckpoint,
) -> tuple[_SerializedState, _SerializedGuard]:
    if type(checkpoint) is not RegisteredSerializedCheckpoint:
        raise TypeError("checkpoint must be an exact RegisteredSerializedCheckpoint")
    with _REGISTRY_LOCK:
        state = _SERIALIZED.get(checkpoint)
        guard = _SERIALIZED_GUARDS.get(checkpoint)
        issued = checkpoint in _ISSUED_SERIALIZED
        failed = checkpoint in _FAILED_SERIALIZED
        closure_valid = _validate_serialized_truth(checkpoint, state, guard)
        if state is None or guard is None or not issued or not closure_valid:
            _FAILED_SERIALIZED.add(checkpoint)
            _recover_fail_serialized_truth(checkpoint)
    if failed:
        raise Experiment002RegisteredCheckpointError(
            "serialized checkpoint is terminally failed"
        )
    if state is None or guard is None or not issued:
        raise Experiment002RegisteredCheckpointError(
            "serialized checkpoint was not issued by this process"
        )
    if not closure_valid:
        raise Experiment002RegisteredCheckpointError(
            "serialized checkpoint authority changed"
        )
    return state, guard


def _validate_serialized_authority(
    checkpoint: RegisteredSerializedCheckpoint,
    state: _SerializedState,
    guard: _SerializedGuard,
) -> None:
    if (
        type(state) is not _SerializedState
        or type(guard) is not _SerializedGuard
        or state.token is not guard.token
        or state.route_marker is not _ROUTE_MARKER
        or guard.route_marker is not _ROUTE_MARKER
        or state.binding is not guard.binding
        or _source_binding_frame(state.binding) != guard.binding_frame
        or state.specs is not guard.specs
        or _specs_frame(state.specs) != guard.specs_frame
        or state.model_payload is not guard.model_payload
        or state.payload is not guard.payload
        or state.safetensors_sha256 != guard.safetensors_sha256
    ):
        raise Experiment002RegisteredCheckpointError(
            "serialized checkpoint authority changed"
        )
    try:
        raw_frame = (
            checkpoint._seed,
            checkpoint._zero_based_epoch,
            checkpoint._history_sha256,
            checkpoint._model_tensor_sha256,
            checkpoint._safetensors_sha256,
            checkpoint._safetensors_byte_count,
            checkpoint._token,
            checkpoint._route_marker,
        )
    except AttributeError as error:
        raise Experiment002RegisteredCheckpointError(
            "serialized checkpoint capability is incomplete"
        ) from error
    expected_frame = (
        state.binding.seed,
        state.binding.zero_based_epoch,
        state.binding.history_sha256,
        state.binding.model_tensor_sha256,
        state.safetensors_sha256,
        len(state.payload),
        state.token,
        _ROUTE_MARKER,
    )
    if (
        any(
            type(observed) is not type(expected) or observed != expected
            for observed, expected in zip(
                raw_frame[:6], expected_frame[:6], strict=True
            )
        )
        or raw_frame[6] is not expected_frame[6]
        or raw_frame[7] is not expected_frame[7]
    ):
        raise Experiment002RegisteredCheckpointError(
            "serialized checkpoint capability fields changed"
        )
    _require_sha256(state.safetensors_sha256, "safetensors sha256")
    if (
        type(state.payload) is not bytes
        or not state.payload
        or hashlib.sha256(state.payload).hexdigest() != state.safetensors_sha256
        or type(state.model_payload) is not bytes
        or not state.model_payload
        or type(state.specs) is not tuple
        or len(state.specs) != MODEL_TENSOR_COUNT
    ):
        raise Experiment002RegisteredCheckpointError(
            "serialized checkpoint immutable bytes changed"
        )


def _source_binding_frame(binding: _SourceBinding) -> tuple[object, ...]:
    if type(binding) is not _SourceBinding:
        raise Experiment002RegisteredCheckpointError(
            "serialized checkpoint source binding has an invalid type"
        )
    identities = (
        binding.completed_history,
        binding.registration,
        binding.history,
        binding.executor,
        binding.validation_inputs,
        binding.complete_update_trace,
        binding.winner,
        binding.evaluated_handoff,
        binding.validation_evidence,
        binding.model_tensors,
    )
    scalar_names = (
        "process_id",
        "seed",
        "epoch_count",
        "zero_based_epoch",
        "optimizer_generation",
        "batch_count",
        "another_training_epoch",
        "registration_head_commit",
        "registration_sha256",
        "source_bundle_sha256",
        "validation_inputs_sha256",
        "validation_predictions_sha256",
        "model_tensor_sha256",
        "optimizer_sha256",
        "torch_rng_sha256",
        "evaluated_authority_sha256",
        "complete_update_trace_sha256",
        "canonical_json_bytes",
        "history_sha256",
    )
    rank_frame = tuple(
        (
            type(rank),
            type(rank.zero_based_epoch),
            rank.zero_based_epoch,
            type(rank.macro_f1_exact_numerator),
            rank.macro_f1_exact_numerator,
            type(rank.macro_f1_exact_denominator),
            rank.macro_f1_exact_denominator,
            type(rank.validation_cross_entropy_float64_hex),
            rank.validation_cross_entropy_float64_hex,
        )
        for rank in binding.ranked_epochs
    )
    evaluated_frame = tuple(
        (type(epoch), id(epoch)) for epoch in binding.evaluated_epochs
    )
    return (
        (type(binding), id(binding)),
        tuple((type(value), id(value)) for value in identities),
        tuple(
            (type(getattr(binding, name)), getattr(binding, name))
            for name in scalar_names
        ),
        (type(binding.ranked_epochs), rank_frame),
        (type(binding.evaluated_epochs), evaluated_frame),
    )


def _specs_frame(specs: tuple[_TensorSpec, ...]) -> tuple[object, ...]:
    if type(specs) is not tuple:
        raise Experiment002RegisteredCheckpointError(
            "serialized checkpoint tensor specs have an invalid type"
        )
    return (
        (type(specs), len(specs)),
        tuple(
            (
                type(spec),
                id(spec),
                type(spec.name),
                spec.name,
                type(spec.shape),
                tuple((type(value), value) for value in spec.shape),
                type(spec.payload),
                spec.payload,
            )
            for spec in specs
        ),
    )


def _serialized_truth_frame(
    checkpoint: RegisteredSerializedCheckpoint,
    state: _SerializedState,
    guard: _SerializedGuard,
) -> tuple[object, ...]:
    return (
        (type(checkpoint), id(checkpoint)),
        (type(state), id(state)),
        (type(guard), id(guard)),
        (type(state.token), id(state.token)),
        (type(guard.token), id(guard.token)),
        (type(state.route_marker), id(state.route_marker)),
        (type(guard.route_marker), id(guard.route_marker)),
        _source_binding_frame(state.binding),
        _source_binding_frame(guard.binding),
        _specs_frame(state.specs),
        _specs_frame(guard.specs),
        (type(state.model_payload), state.model_payload),
        (type(guard.model_payload), guard.model_payload),
        (type(state.payload), state.payload),
        (type(guard.payload), guard.payload),
        (type(state.safetensors_sha256), state.safetensors_sha256),
        (type(guard.safetensors_sha256), guard.safetensors_sha256),
        (
            type(checkpoint._seed),
            checkpoint._seed,
            type(checkpoint._zero_based_epoch),
            checkpoint._zero_based_epoch,
            type(checkpoint._history_sha256),
            checkpoint._history_sha256,
            type(checkpoint._model_tensor_sha256),
            checkpoint._model_tensor_sha256,
            type(checkpoint._safetensors_sha256),
            checkpoint._safetensors_sha256,
            type(checkpoint._safetensors_byte_count),
            checkpoint._safetensors_byte_count,
            type(checkpoint._token),
            id(checkpoint._token),
            type(checkpoint._route_marker),
            id(checkpoint._route_marker),
        ),
    )


def _terminal_fail_serialized(checkpoint: RegisteredSerializedCheckpoint) -> None:
    _recover_fail_serialized_truth(checkpoint)
    with _REGISTRY_LOCK:
        _FAILED_SERIALIZED.add(checkpoint)


def _build_truth_routes() -> _TruthRoutes:
    attempts: dict[RegisteredCompletedTrainingHistory, _AttemptRecord] = {}
    failed_attempts: set[RegisteredCompletedTrainingHistory] = set()
    retained_results: dict[
        RegisteredCompletedTrainingHistory,
        RegisteredSerializedCheckpoint | None,
    ] = {}
    retained_outcomes: dict[
        RegisteredCompletedTrainingHistory,
        _SerializationOutcome,
    ] = {}
    issued: dict[
        RegisteredSerializedCheckpoint,
        tuple[_SerializedState, _SerializedGuard, tuple[object, ...]],
    ] = {}
    issued_by_history: dict[
        RegisteredCompletedTrainingHistory,
        set[RegisteredSerializedCheckpoint],
    ] = {}
    failed_issued: set[RegisteredSerializedCheckpoint] = set()
    admission_tickets: dict[
        int,
        tuple[
            _SerializationAdmission,
            RegisteredCompletedTrainingHistory,
            _AttemptRecord,
            bool,
            tuple[object, ...],
        ],
    ] = {}
    lock = threading.RLock()

    def outcome_frame(
        outcome: list[_SerializationOutcome],
    ) -> tuple[object, ...] | None:
        if (
            type(outcome) is not list
            or len(outcome) != 1
            or type(outcome[0]) is not _SerializationOutcome
        ):
            return None
        value = outcome[0]
        return (
            (type(value), id(value)),
            (type(value.result), id(value.result)),
            (type(value.error), id(value.error)),
        )

    def validate_record(record: _AttemptRecord) -> None:
        if (
            type(record) is not _AttemptRecord
            or type(record.token) is not object
            or type(record.completed) is not _EVENT_TYPE
            or type(record.outcome) is not list
            or record.phase not in ("SERIALIZING", "COMPLETE", "FAILED")
        ):
            raise Experiment002RegisteredCheckpointError(
                "serialization admission truth changed"
            )
        if record.phase == "SERIALIZING":
            if (
                record.completed.is_set()
                or record.outcome
                or record.outcome_frame is not None
            ):
                raise Experiment002RegisteredCheckpointError(
                    "active serialization admission truth changed"
                )
        elif (
            not record.completed.is_set()
            or outcome_frame(record.outcome) != record.outcome_frame
        ):
            raise Experiment002RegisteredCheckpointError(
                "retained serialization outcome truth changed"
            )

    def poison_attempt(
        history: RegisteredCompletedTrainingHistory,
        record: _AttemptRecord,
    ) -> None:
        failed_attempts.add(history)
        retained = retained_results.get(history)
        if retained is not None:
            failed_issued.add(retained)
        failed_issued.update(issued_by_history.get(history, ()))
        record.completed.set()

    def admission_frame(
        admission: _SerializationAdmission,
    ) -> tuple[object, ...]:
        return (
            (type(admission), id(admission)),
            (type(admission.token), id(admission.token)),
            (type(admission.history), id(admission.history)),
            (type(admission.leader), admission.leader),
            (type(admission.completed), id(admission.completed)),
            (type(admission.outcome), id(admission.outcome)),
        )

    def issue_admission(
        *,
        token: object,
        history: RegisteredCompletedTrainingHistory,
        leader: bool,
        record: _AttemptRecord,
    ) -> _SerializationAdmission:
        admission = _SerializationAdmission(
            token,
            history,
            leader,
            record.completed,
            record.outcome,
        )
        admission_tickets[id(admission)] = (
            admission,
            history,
            record,
            leader,
            admission_frame(admission),
        )
        return admission

    def validate_admission(admission: _SerializationAdmission) -> bool:
        with lock:
            ticket = admission_tickets.get(id(admission))
            if ticket is None:
                record = attempts.get(admission.history)
                if record is not None:
                    poison_attempt(admission.history, record)
                return False
            original, history, issued_record, issued_leader, frame = ticket
            current_record = attempts.get(history)
            valid = (
                original is admission
                and current_record is not None
                and current_record.token is issued_record.token
                and current_record.completed is issued_record.completed
                and current_record.outcome is issued_record.outcome
                and admission_frame(admission) == frame
                and admission.history is history
                and admission.completed is issued_record.completed
                and admission.outcome is issued_record.outcome
                and (
                    (admission.leader and admission.token is issued_record.token)
                    or (
                        not admission.leader
                        and admission.token is not issued_record.token
                    )
                )
                and admission.leader is issued_leader
            )
            if valid:
                try:
                    assert current_record is not None
                    validate_record(current_record)
                except BaseException:
                    valid = False
            if not valid:
                poison_attempt(history, current_record or issued_record)
            return valid

    def admission_role(admission: _SerializationAdmission) -> bool:
        with lock:
            if not validate_admission(admission):
                raise Experiment002RegisteredCheckpointError(
                    "serialization admission closure truth changed"
                )
            ticket = admission_tickets[id(admission)]
            return ticket[3]

    def terminal_outcome(
        admission: _SerializationAdmission,
    ) -> _SerializationOutcome:
        with lock:
            if not validate_admission(admission):
                raise Experiment002RegisteredCheckpointError(
                    "serialization admission closure truth changed"
                )
            record = attempts.get(admission.history)
            retained = retained_outcomes.get(admission.history)
            if (
                record is None
                or record.phase not in ("COMPLETE", "FAILED")
                or retained is None
                or len(record.outcome) != 1
                or record.outcome[0] is not retained
            ):
                if record is not None:
                    poison_attempt(admission.history, record)
                raise Experiment002RegisteredCheckpointError(
                    "serialization terminal outcome closure truth changed"
                )
            return retained

    def record_issued(
        checkpoint: RegisteredSerializedCheckpoint,
        state: _SerializedState,
        guard: _SerializedGuard,
    ) -> None:
        with lock:
            if checkpoint in issued or checkpoint in failed_issued:
                failed_issued.add(checkpoint)
                raise Experiment002RegisteredCheckpointError(
                    "serialized checkpoint closure truth was already recorded"
                )
            issued[checkpoint] = (
                state,
                guard,
                _serialized_truth_frame(checkpoint, state, guard),
            )
            issued_by_history.setdefault(
                state.binding.completed_history,
                set(),
            ).add(checkpoint)

    def validate_issued(
        checkpoint: RegisteredSerializedCheckpoint,
        state_object: object,
        guard_object: object,
    ) -> bool:
        with lock:
            if checkpoint in failed_issued:
                return False
            truth = issued.get(checkpoint)
            if (
                truth is None
                or type(state_object) is not _SerializedState
                or type(guard_object) is not _SerializedGuard
                or truth[0] is not state_object
                or truth[1] is not guard_object
            ):
                failed_issued.add(checkpoint)
                return False
            try:
                valid = truth[2] == _serialized_truth_frame(
                    checkpoint,
                    state_object,
                    guard_object,
                )
            except BaseException:
                failed_issued.add(checkpoint)
                return False
            if not valid:
                failed_issued.add(checkpoint)
            return valid

    def fail_issued(checkpoint: RegisteredSerializedCheckpoint) -> None:
        with lock:
            failed_issued.add(checkpoint)

    def begin(
        history: RegisteredCompletedTrainingHistory,
        caller_token: object,
    ) -> _SerializationAdmission:
        with lock:
            record = attempts.get(history)
            if history in failed_attempts:
                if record is not None:
                    poison_attempt(history, record)
                raise Experiment002RegisteredCheckpointError(
                    "serialization attempt is terminally failed"
                )
            if record is not None:
                try:
                    validate_record(record)
                except BaseException:
                    poison_attempt(history, record)
                    raise
                return issue_admission(
                    token=caller_token,
                    history=history,
                    leader=False,
                    record=record,
                )
            completed = threading.Event()
            outcome: list[_SerializationOutcome] = []
            record = _AttemptRecord(
                caller_token,
                completed,
                outcome,
                "SERIALIZING",
                None,
            )
            attempts[history] = record
            return issue_admission(
                token=caller_token,
                history=history,
                leader=True,
                record=record,
            )

    def recover(
        history: RegisteredCompletedTrainingHistory,
        caller_token: object,
    ) -> _SerializationAdmission | None:
        with lock:
            record = attempts.get(history)
            if record is None or record.token is not caller_token:
                return None
            return issue_admission(
                token=caller_token,
                history=history,
                leader=True,
                record=record,
            )

    def finish(
        admission: _SerializationAdmission,
        outcome: _SerializationOutcome,
    ) -> None:
        if not validate_admission(admission):
            raise Experiment002RegisteredCheckpointError(
                "serialization admission closure truth changed"
            )
        if type(admission.leader) is not bool or admission.leader is not True:
            raise Experiment002RegisteredCheckpointError(
                "only the serialization leader may publish an outcome"
            )
        authorized = False
        try:
            with lock:
                record = attempts.get(admission.history)
                if (
                    record is None
                    or record.token is not admission.token
                    or record.completed is not admission.completed
                    or record.outcome is not admission.outcome
                ):
                    raise Experiment002RegisteredCheckpointError(
                        "serialization admission authority changed"
                    )
                authorized = True
                if admission.history in failed_attempts:
                    poison_attempt(admission.history, record)
                    raise Experiment002RegisteredCheckpointError(
                        "serialization attempt is terminally failed"
                    )
                if record.phase in ("COMPLETE", "FAILED"):
                    try:
                        validate_record(record)
                    except BaseException:
                        poison_attempt(admission.history, record)
                        raise
                    return
                if (outcome.result is None) == (outcome.error is None):
                    poison_attempt(admission.history, record)
                    raise Experiment002RegisteredCheckpointError(
                        "serialization terminal outcome is inconsistent"
                    )
                if outcome.result is not None:
                    issued_truth = issued.get(outcome.result)
                    if (
                        issued_truth is None
                        or outcome.result in failed_issued
                        or issued_truth[0].binding.completed_history
                        is not admission.history
                        or outcome.result
                        not in issued_by_history.get(admission.history, ())
                        or not validate_issued(
                            outcome.result,
                            issued_truth[0],
                            issued_truth[1],
                        )
                    ):
                        poison_attempt(admission.history, record)
                        raise Experiment002RegisteredCheckpointError(
                            "serialization result differs from closure authority"
                        )
                admission.outcome[:] = [outcome]
                phase: _AttemptPhase = (
                    "COMPLETE"
                    if outcome.result is not None and outcome.error is None
                    else "FAILED"
                )
                if phase == "FAILED":
                    failed_issued.update(issued_by_history.get(admission.history, ()))
                frame = outcome_frame(admission.outcome)
                if frame is None:
                    poison_attempt(admission.history, record)
                    raise Experiment002RegisteredCheckpointError(
                        "serialization terminal outcome is invalid"
                    )
                retained_results[admission.history] = outcome.result
                retained_outcomes[admission.history] = outcome
                attempts[admission.history] = _AttemptRecord(
                    admission.token,
                    admission.completed,
                    admission.outcome,
                    phase,
                    frame,
                )
        finally:
            if authorized:
                admission.completed.set()

    return _TruthRoutes(
        begin=begin,
        recover=recover,
        validate_admission=validate_admission,
        admission_role=admission_role,
        terminal_outcome=terminal_outcome,
        finish=finish,
        record_issued=record_issued,
        validate_issued=validate_issued,
        fail_issued=fail_issued,
    )


_TRUTH: Final = _build_truth_routes()


def _begin_serialization_admission(
    history: RegisteredCompletedTrainingHistory,
    caller_token: object,
    _begin: Callable[
        [RegisteredCompletedTrainingHistory, object], _SerializationAdmission
    ] = _TRUTH.begin,
) -> _SerializationAdmission:
    return _begin(history, caller_token)


def _recover_serialization_admission(
    history: RegisteredCompletedTrainingHistory,
    caller_token: object,
    _recover: Callable[
        [RegisteredCompletedTrainingHistory, object],
        _SerializationAdmission | None,
    ] = _TRUTH.recover,
) -> _SerializationAdmission | None:
    return _recover(history, caller_token)


def _validate_serialization_admission_truth(
    admission: _SerializationAdmission,
    _validate: Callable[[_SerializationAdmission], bool] = _TRUTH.validate_admission,
) -> bool:
    return _validate(admission)


def _serialization_admission_role(
    admission: _SerializationAdmission,
    _role: Callable[[_SerializationAdmission], bool] = _TRUTH.admission_role,
) -> bool:
    return _role(admission)


def _serialization_terminal_outcome_truth(
    admission: _SerializationAdmission,
    _outcome: Callable[
        [_SerializationAdmission], _SerializationOutcome
    ] = _TRUTH.terminal_outcome,
) -> _SerializationOutcome:
    return _outcome(admission)


def _finish_serialization_admission(
    admission: _SerializationAdmission,
    outcome: _SerializationOutcome,
    _finish: Callable[[_SerializationAdmission, _SerializationOutcome], None] = (
        _TRUTH.finish
    ),
) -> None:
    _finish(admission, outcome)


def _recover_finish_serialization_admission(
    admission: _SerializationAdmission,
    outcome: _SerializationOutcome,
    _finish: Callable[[_SerializationAdmission, _SerializationOutcome], None] = (
        _TRUTH.finish
    ),
) -> None:
    _finish(admission, outcome)


def _record_serialized_truth(
    checkpoint: RegisteredSerializedCheckpoint,
    state: _SerializedState,
    guard: _SerializedGuard,
    _record: Callable[
        [RegisteredSerializedCheckpoint, _SerializedState, _SerializedGuard], None
    ] = _TRUTH.record_issued,
) -> None:
    _record(checkpoint, state, guard)


def _validate_serialized_truth(
    checkpoint: RegisteredSerializedCheckpoint,
    state: object,
    guard: object,
    _validate: Callable[
        [RegisteredSerializedCheckpoint, object, object], bool
    ] = _TRUTH.validate_issued,
) -> bool:
    return _validate(checkpoint, state, guard)


def _fail_serialized_truth(
    checkpoint: RegisteredSerializedCheckpoint,
    _fail: Callable[[RegisteredSerializedCheckpoint], None] = _TRUTH.fail_issued,
) -> None:
    _fail(checkpoint)


def _recover_fail_serialized_truth(
    checkpoint: RegisteredSerializedCheckpoint,
    _fail: Callable[[RegisteredSerializedCheckpoint], None] = _TRUTH.fail_issued,
) -> None:
    _fail(checkpoint)


def _publish_serialization_outcome(
    admission: _SerializationAdmission,
    outcome: _SerializationOutcome,
) -> None:
    try:
        _finish_serialization_admission(admission, outcome)
    except BaseException:
        recovered = _recover_serialization_admission(
            admission.history,
            admission.token,
        )
        target = recovered if recovered is not None else admission
        try:
            _recover_finish_serialization_admission(target, outcome)
        finally:
            target.completed.set()
            admission.completed.set()
        raise


def _require_admission(
    admission: _SerializationAdmission,
    history: RegisteredCompletedTrainingHistory,
) -> bool:
    if type(admission) is not _SerializationAdmission:
        raise Experiment002RegisteredCheckpointError(
            "serialization admission capability changed"
        )
    leader = _serialization_admission_role(admission)
    if (
        type(admission.token) is not object
        or admission.history is not history
        or type(admission.leader) is not bool
        or admission.leader is not leader
        or type(admission.completed) is not _EVENT_TYPE
        or type(admission.outcome) is not list
        or (leader and admission.outcome)
        or (leader and admission.completed.is_set())
        or (not leader and len(admission.outcome) > 1)
        or (not leader and not admission.outcome and admission.completed.is_set())
        or (
            not leader
            and len(admission.outcome) == 1
            and not admission.completed.is_set()
        )
    ):
        raise Experiment002RegisteredCheckpointError(
            "serialization admission capability changed"
        )
    return leader


def _require_terminal_outcome(
    admission: _SerializationAdmission,
) -> _SerializationOutcome:
    outcome = _serialization_terminal_outcome_truth(admission)
    if (outcome.result is None) == (outcome.error is None):
        raise Experiment002RegisteredCheckpointError(
            "serialization terminal outcome is inconsistent"
        )
    return outcome


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise Experiment002RegisteredCheckpointError(
                "safetensors header contains a duplicate or invalid key"
            )
        result[key] = value
    return result


def _reject_json_number(value: str) -> NoReturn:
    raise Experiment002RegisteredCheckpointError(
        f"safetensors header contains a forbidden JSON number ({value})"
    )


def _tensor_payload(tensor: Tensor) -> bytes:
    return tensor.detach().numpy().astype("<f4", copy=False).tobytes(order="C")


def _torch_rng_state() -> bytes:
    state = torch.get_rng_state()
    if (
        type(state) is not Tensor
        or state.device.type != "cpu"
        or state.dtype != torch.uint8
        or state.layout != torch.strided
        or not state.is_contiguous()
    ):
        raise Experiment002RegisteredCheckpointError(
            "Torch CPU RNG state has an invalid representation"
        )
    return state.numpy().tobytes(order="C")


def _require_rng_unchanged(expected: bytes) -> None:
    if _torch_rng_state() != expected:
        restored = torch.frombuffer(bytearray(expected), dtype=torch.uint8).clone()
        torch.set_rng_state(restored)
        if _torch_rng_state() != expected:
            raise Experiment002RegisteredCheckpointError(
                "Torch CPU RNG could not be restored after checkpoint failure"
            )
        raise Experiment002RegisteredCheckpointError(
            "Torch CPU RNG changed across checkpoint serialization"
        )


def _require_sha256(value: object, name: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise Experiment002RegisteredCheckpointError(f"{name} is invalid")


del _TRUTH
