"""Byte-exact supplied-observation evidence for Experiment 002 training.

An update trace preserves values presented by trusted trainer integration.  It
does not, by itself, prove that an optimizer step occurred; that integration
must consume the matching issuer-only completed-update receipt.
"""

from __future__ import annotations

import hashlib
import math
import struct
import threading
import weakref
from dataclasses import dataclass
from typing import Final, NoReturn

import numpy as np
import torch
from torch import Tensor, nn

from falsewake.causal_kws import CausalKWS

REGISTERED_SEEDS: Final = (20_260_719, 20_260_720, 20_260_721)
EPOCH_COUNT: Final = 30
UPDATES_PER_EPOCH: Final = 313
TOTAL_UPDATE_COUNT: Final = 9_390
TRAINING_EXAMPLE_COUNT: Final = 40_027
TRAINING_BATCH_SIZE: Final = 128
LAST_BATCH_SIZE: Final = 91
MODEL_TENSOR_COUNT: Final = 53
MODEL_PARAMETER_VALUE_COUNT: Final = 23_724

MODEL_TENSOR_DOMAIN: Final = b"falsewake-exp002-model-tensors-v1\0"
EPOCH_UPDATE_TRACE_DOMAIN: Final = b"falsewake-exp002-epoch-update-trace-v1\0"
COMPLETE_UPDATE_TRACE_DOMAIN: Final = b"falsewake-exp002-update-trace-v1\0"
_F32LE_TAG: Final = b"F32LE"
_MODEL_TENSOR_ROUTE_MARKER: Final = object()
_UPDATE_TRACE_ROUTE_MARKER: Final = object()
_UINT32_MAX: Final = (1 << 32) - 1


class Experiment002TrainingEvidenceError(ValueError):
    """Model or update evidence violates the frozen trainer contract."""


def _require_positive_integer(value: object, name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise Experiment002TrainingEvidenceError(f"{name} must be positive")


def _require_uint32(value: object, name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if not 0 <= value <= _UINT32_MAX:
        raise Experiment002TrainingEvidenceError(f"{name} must fit UINT32LE")


@dataclass(frozen=True, slots=True)
class _ModelTensorLayout:
    tensor_count: int
    value_count: int

    def __post_init__(self) -> None:
        _require_positive_integer(self.tensor_count, "tensor_count")
        _require_positive_integer(self.value_count, "value_count")


_REGISTERED_MODEL_LAYOUT: Final = _ModelTensorLayout(
    tensor_count=MODEL_TENSOR_COUNT,
    value_count=MODEL_PARAMETER_VALUE_COUNT,
)


def _validate_model_tensor_layout(layout: _ModelTensorLayout) -> None:
    if type(layout) is not _ModelTensorLayout:
        raise TypeError("layout must be a _ModelTensorLayout")
    layout.__post_init__()


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
class RegisteredModelTensorEvidence:
    """Issuer-only digest and owned tensor snapshot from one model state."""

    _sha256: str
    _seed: int | None
    _zero_based_epoch: int | None
    _tensor_count: int
    _value_count: int
    _route_marker: object | None

    def __init__(self) -> None:
        raise TypeError("model tensor evidence is issued by this module")

    @property
    def sha256(self) -> str:
        """Return the canonical model-tensor SHA-256 digest."""

        return _issued_model_tensor_state(self).sha256

    @property
    def seed(self) -> int | None:
        """Return the run seed bound to this snapshot, if any."""

        return _issued_model_tensor_state(self).seed

    @property
    def zero_based_epoch(self) -> int | None:
        """Return the completed epoch bound to this snapshot, if any."""

        return _issued_model_tensor_state(self).zero_based_epoch

    @property
    def tensor_count(self) -> int:
        """Return the number of framed parameter tensors."""

        return _issued_model_tensor_state(self).layout.tensor_count

    @property
    def value_count(self) -> int:
        """Return the total number of framed float32 values."""

        return _issued_model_tensor_state(self).layout.value_count

    def __copy__(self) -> NoReturn:
        raise TypeError("model tensor evidence cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        del memo
        raise TypeError("model tensor evidence cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("model tensor evidence cannot be serialized")


@dataclass(frozen=True, slots=True)
class _IssuedModelTensorState:
    seed: int | None
    zero_based_epoch: int | None
    names: tuple[str, ...]
    tensors: tuple[Tensor, ...]
    framed_payload: bytes
    sha256: str
    layout: _ModelTensorLayout
    route_marker: object | None


_ISSUED_MODEL_TENSORS: weakref.WeakKeyDictionary[
    RegisteredModelTensorEvidence, _IssuedModelTensorState
] = weakref.WeakKeyDictionary()
_ISSUED_MODEL_TENSORS_LOCK = threading.Lock()


def capture_registered_model_tensors(
    model: CausalKWS,
    *,
    seed: int,
    zero_based_epoch: int,
) -> RegisteredModelTensorEvidence:
    """Capture the exact registered CausalKWS parameter state."""

    if type(model) is not CausalKWS:
        raise TypeError("model must be an exact CausalKWS")
    _require_registered_seed(seed)
    _require_registered_epoch(zero_based_epoch)
    return _capture_model_tensors(
        model,
        layout=_REGISTERED_MODEL_LAYOUT,
        seed=seed,
        zero_based_epoch=zero_based_epoch,
        route_marker=_MODEL_TENSOR_ROUTE_MARKER,
    )


def verify_registered_model_tensor_evidence(
    evidence: RegisteredModelTensorEvidence,
    *,
    seed: int,
    zero_based_epoch: int,
) -> None:
    """Require an intact capability issued through the exact public route."""

    _require_registered_seed(seed)
    _require_registered_epoch(zero_based_epoch)
    state = _issued_model_tensor_state(evidence)
    if state.route_marker is not _MODEL_TENSOR_ROUTE_MARKER or (
        state.layout != _REGISTERED_MODEL_LAYOUT
    ):
        raise Experiment002TrainingEvidenceError(
            "model tensor evidence was not issued by the registered route"
        )
    _verify_model_tensor_context(evidence, seed, zero_based_epoch)


def verify_registered_model_tensors_match(
    model: CausalKWS,
    evidence: RegisteredModelTensorEvidence,
    *,
    seed: int,
    zero_based_epoch: int,
) -> None:
    """Require the current exact model bytes to equal an issued snapshot."""

    if type(model) is not CausalKWS:
        raise TypeError("model must be an exact CausalKWS")
    verify_registered_model_tensor_evidence(
        evidence,
        seed=seed,
        zero_based_epoch=zero_based_epoch,
    )
    _verify_model_matches_evidence(model, evidence)


def _capture_model_tensors(
    model: nn.Module,
    *,
    layout: _ModelTensorLayout,
    seed: int | None = None,
    zero_based_epoch: int | None = None,
    route_marker: object | None = None,
) -> RegisteredModelTensorEvidence:
    """Private small-model seam using the production tensor framing."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if type(layout) is not _ModelTensorLayout:
        raise TypeError("layout must be a _ModelTensorLayout")
    if route_marker is not None and route_marker is not _MODEL_TENSOR_ROUTE_MARKER:
        raise Experiment002TrainingEvidenceError(
            "model tensor capture received an invalid route marker"
        )
    _require_optional_model_context(seed, zero_based_epoch)
    if route_marker is _MODEL_TENSOR_ROUTE_MARKER:
        if seed is None or zero_based_epoch is None:
            raise Experiment002TrainingEvidenceError(
                "registered model evidence requires a seed and epoch"
            )
        _require_registered_seed(seed)
        _require_registered_epoch(zero_based_epoch)

    first_names, first_tensors = _snapshot_model_parameters(model, layout)
    first_payload = _frame_model_tensors(first_names, first_tensors)
    second_names, second_tensors = _snapshot_model_parameters(model, layout)
    second_payload = _frame_model_tensors(second_names, second_tensors)
    if first_names != second_names or first_payload != second_payload:
        raise Experiment002TrainingEvidenceError(
            "model parameter bytes changed during evidence capture"
        )

    sha256 = hashlib.sha256(MODEL_TENSOR_DOMAIN + first_payload).hexdigest()
    result = object.__new__(RegisteredModelTensorEvidence)
    object.__setattr__(result, "_sha256", sha256)
    object.__setattr__(result, "_seed", seed)
    object.__setattr__(result, "_zero_based_epoch", zero_based_epoch)
    object.__setattr__(result, "_tensor_count", layout.tensor_count)
    object.__setattr__(result, "_value_count", layout.value_count)
    object.__setattr__(result, "_route_marker", route_marker)
    state = _IssuedModelTensorState(
        seed=seed,
        zero_based_epoch=zero_based_epoch,
        names=first_names,
        tensors=first_tensors,
        framed_payload=first_payload,
        sha256=sha256,
        layout=_ModelTensorLayout(layout.tensor_count, layout.value_count),
        route_marker=route_marker,
    )
    with _ISSUED_MODEL_TENSORS_LOCK:
        _ISSUED_MODEL_TENSORS[result] = state
    return result


def _verify_model_matches_evidence(
    model: nn.Module,
    evidence: RegisteredModelTensorEvidence,
) -> None:
    state = _issued_model_tensor_state(evidence)
    first_names, first_tensors = _snapshot_model_parameters(model, state.layout)
    first_payload = _frame_model_tensors(first_names, first_tensors)
    second_names, second_tensors = _snapshot_model_parameters(model, state.layout)
    second_payload = _frame_model_tensors(second_names, second_tensors)
    if (
        first_names != state.names
        or second_names != state.names
        or first_payload != state.framed_payload
        or second_payload != state.framed_payload
    ):
        raise Experiment002TrainingEvidenceError(
            "current model tensors differ from the issued evidence"
        )


def _verify_model_tensor_context(
    evidence: RegisteredModelTensorEvidence,
    seed: int,
    zero_based_epoch: int,
) -> None:
    """Require one capability's exact run/epoch context without reframing it."""

    _require_uint32(seed, "seed")
    _require_uint32(zero_based_epoch, "zero_based_epoch")
    state = _issued_model_tensor_state(evidence)
    if state.seed != seed or state.zero_based_epoch != zero_based_epoch:
        raise Experiment002TrainingEvidenceError(
            "model tensor evidence belongs to a different seed or epoch"
        )


def _model_tensor_clones(
    evidence: RegisteredModelTensorEvidence,
) -> tuple[tuple[str, Tensor], ...]:
    """Return fresh owned clones for later trusted checkpoint serialization."""

    state = _issued_model_tensor_state(evidence)
    return tuple(
        (name, tensor.detach().clone(memory_format=torch.contiguous_format))
        for name, tensor in zip(state.names, state.tensors, strict=True)
    )


def _snapshot_model_parameters(
    model: nn.Module,
    layout: _ModelTensorLayout,
) -> tuple[tuple[str, ...], tuple[Tensor, ...]]:
    named_parameters = tuple(model.named_parameters())
    if len(named_parameters) != layout.tensor_count:
        raise Experiment002TrainingEvidenceError(
            "model parameter tensor count differs from the layout"
        )
    names = tuple(name for name, _ in named_parameters)
    if any(type(name) is not str or not name for name in names):
        raise Experiment002TrainingEvidenceError("model parameter names are invalid")
    if len(set(names)) != len(names):
        raise Experiment002TrainingEvidenceError("model parameter names are not unique")
    parameter_ids = tuple(id(parameter) for _, parameter in named_parameters)
    if len(set(parameter_ids)) != len(parameter_ids):
        raise Experiment002TrainingEvidenceError(
            "model parameter objects are not unique"
        )

    state_dict = model.state_dict()
    if set(state_dict) != set(names) or len(state_dict) != len(names):
        raise Experiment002TrainingEvidenceError(
            "state_dict keys differ from named parameter keys"
        )

    sorted_names = tuple(sorted(names, key=lambda name: name.encode("utf-8")))
    snapshots: list[Tensor] = []
    value_count = 0
    for name in sorted_names:
        tensor = state_dict[name]
        _require_float32_parameter_tensor(tensor, name)
        value_count += tensor.numel()
        snapshot = tensor.detach().clone(memory_format=torch.contiguous_format)
        _require_float32_parameter_tensor(snapshot, name)
        snapshots.append(snapshot)
    if value_count != layout.value_count:
        raise Experiment002TrainingEvidenceError(
            "model parameter value count differs from the layout"
        )
    return sorted_names, tuple(snapshots)


def _require_float32_parameter_tensor(tensor: Tensor, name: str) -> None:
    if type(tensor) is not Tensor:
        raise TypeError(f"state_dict value {name!r} must be an exact Tensor")
    if tensor.device.type != "cpu":
        raise Experiment002TrainingEvidenceError(
            f"model parameter {name!r} must be on CPU"
        )
    if tensor.dtype != torch.float32:
        raise Experiment002TrainingEvidenceError(
            f"model parameter {name!r} must use float32"
        )
    if tensor.layout != torch.strided or not tensor.is_contiguous():
        raise Experiment002TrainingEvidenceError(
            f"model parameter {name!r} must be dense and C-contiguous"
        )
    if tensor.requires_grad:
        raise Experiment002TrainingEvidenceError(
            f"model parameter snapshot {name!r} must not require gradients"
        )
    if not bool(torch.isfinite(tensor).all()):
        raise Experiment002TrainingEvidenceError(
            f"model parameter {name!r} contains a non-finite value"
        )


def _validate_model_tensor_snapshot(
    names: tuple[str, ...],
    tensors: tuple[Tensor, ...],
    layout: _ModelTensorLayout,
) -> None:
    _validate_model_tensor_layout(layout)
    if (
        type(names) is not tuple
        or type(tensors) is not tuple
        or len(names) != layout.tensor_count
        or len(tensors) != layout.tensor_count
    ):
        raise Experiment002TrainingEvidenceError("model tensor snapshot is invalid")
    if any(type(name) is not str or not name for name in names):
        raise Experiment002TrainingEvidenceError("model parameter names are invalid")
    if len(set(names)) != len(names):
        raise Experiment002TrainingEvidenceError("model parameter names are not unique")
    if names != tuple(sorted(names, key=lambda name: name.encode("utf-8"))):
        raise Experiment002TrainingEvidenceError(
            "model parameter names are not in canonical UTF-8 byte order"
        )
    value_count = 0
    for name, tensor in zip(names, tensors, strict=True):
        _require_float32_parameter_tensor(tensor, name)
        value_count += tensor.numel()
    if value_count != layout.value_count:
        raise Experiment002TrainingEvidenceError(
            "model parameter value count differs from the layout"
        )


def _frame_model_tensors(names: tuple[str, ...], tensors: tuple[Tensor, ...]) -> bytes:
    if (
        type(names) is not tuple
        or type(tensors) is not tuple
        or len(names) != len(tensors)
    ):
        raise Experiment002TrainingEvidenceError("model tensor snapshot is invalid")
    framed = bytearray(struct.pack("<I", len(names)))
    for name, tensor in zip(names, tensors, strict=True):
        if type(name) is not str or not name:
            raise Experiment002TrainingEvidenceError(
                "model parameter names are invalid"
            )
        _require_float32_parameter_tensor(tensor, name)
        name_bytes = name.encode("utf-8")
        payload = tensor.detach().numpy().astype("<f4", copy=False).tobytes(order="C")
        framed.extend(struct.pack("<I", len(name_bytes)))
        framed.extend(name_bytes)
        framed.extend(struct.pack("<I", len(_F32LE_TAG)))
        framed.extend(_F32LE_TAG)
        framed.extend(struct.pack("<I", tensor.ndim))
        for dimension in tensor.shape:
            framed.extend(struct.pack("<Q", dimension))
        framed.extend(struct.pack("<Q", len(payload)))
        framed.extend(payload)
    return bytes(framed)


def _issued_model_tensor_state(
    evidence: RegisteredModelTensorEvidence,
) -> _IssuedModelTensorState:
    if type(evidence) is not RegisteredModelTensorEvidence:
        raise TypeError("evidence must be RegisteredModelTensorEvidence")
    with _ISSUED_MODEL_TENSORS_LOCK:
        state = _ISSUED_MODEL_TENSORS.get(evidence)
    if state is None:
        raise Experiment002TrainingEvidenceError(
            "model tensor evidence was not issued by this process"
        )
    _validate_issued_model_tensor_state(state)
    observed_payload = _frame_model_tensors(state.names, state.tensors)
    observed_sha256 = hashlib.sha256(MODEL_TENSOR_DOMAIN + observed_payload).hexdigest()
    try:
        raw_sha256 = evidence._sha256
        raw_seed = evidence._seed
        raw_zero_based_epoch = evidence._zero_based_epoch
        raw_tensor_count = evidence._tensor_count
        raw_value_count = evidence._value_count
        raw_route_marker = evidence._route_marker
    except AttributeError as error:
        raise Experiment002TrainingEvidenceError(
            "model tensor evidence capability is incomplete"
        ) from error
    if (
        observed_payload != state.framed_payload
        or observed_sha256 != state.sha256
        or type(raw_sha256) is not str
        or raw_sha256 != state.sha256
        or not _same_optional_int(raw_seed, state.seed)
        or not _same_optional_int(raw_zero_based_epoch, state.zero_based_epoch)
        or type(raw_tensor_count) is not int
        or raw_tensor_count != state.layout.tensor_count
        or type(raw_value_count) is not int
        or raw_value_count != state.layout.value_count
        or raw_route_marker is not state.route_marker
    ):
        raise Experiment002TrainingEvidenceError(
            "model tensor evidence capability changed after issuance"
        )
    return state


def _validate_issued_model_tensor_state(state: _IssuedModelTensorState) -> None:
    if type(state) is not _IssuedModelTensorState:
        raise TypeError("issued model tensor state has an invalid type")
    _require_optional_model_context(state.seed, state.zero_based_epoch)
    if state.route_marker is _MODEL_TENSOR_ROUTE_MARKER:
        if state.seed is None or state.zero_based_epoch is None:
            raise Experiment002TrainingEvidenceError(
                "registered model tensor state requires a seed and epoch"
            )
        _require_registered_seed(state.seed)
        _require_registered_epoch(state.zero_based_epoch)
    elif state.route_marker is not None:
        raise Experiment002TrainingEvidenceError(
            "issued model tensor state has an invalid route marker"
        )
    _validate_model_tensor_snapshot(state.names, state.tensors, state.layout)
    if type(state.framed_payload) is not bytes:
        raise TypeError("issued model tensor framing must be bytes")
    if type(state.sha256) is not str:
        raise TypeError("issued model tensor digest must be a string")


@dataclass(frozen=True, slots=True)
class _TraceLayout:
    epochs: int
    updates_per_epoch: int
    population_count: int
    batch_size: int
    last_batch_size: int
    warmup_updates: int
    warmup_start_lr: float
    maximum_lr: float
    minimum_lr: float

    def __post_init__(self) -> None:
        for field_name in (
            "epochs",
            "updates_per_epoch",
            "population_count",
            "batch_size",
            "last_batch_size",
            "warmup_updates",
        ):
            _require_positive_integer(getattr(self, field_name), field_name)
        if self.last_batch_size > self.batch_size:
            raise Experiment002TrainingEvidenceError(
                "last_batch_size cannot exceed batch_size"
            )
        expected_population = (
            self.batch_size * (self.updates_per_epoch - 1) + self.last_batch_size
        )
        if self.population_count != expected_population:
            raise Experiment002TrainingEvidenceError(
                "population_count differs from the batch layout"
            )
        if self.warmup_updates >= self.total_updates:
            raise Experiment002TrainingEvidenceError(
                "warmup_updates must leave at least one cosine update"
            )
        if self.warmup_updates < 2:
            raise Experiment002TrainingEvidenceError(
                "warmup_updates must contain at least two updates"
            )
        for field_name in ("warmup_start_lr", "maximum_lr", "minimum_lr"):
            value = getattr(self, field_name)
            if type(value) is not float or not math.isfinite(value) or value <= 0.0:
                raise Experiment002TrainingEvidenceError(
                    f"{field_name} must be a finite positive float"
                )
        if not self.minimum_lr < self.warmup_start_lr <= self.maximum_lr:
            raise Experiment002TrainingEvidenceError(
                "learning-rate endpoints are not ordered"
            )

    @property
    def total_updates(self) -> int:
        return self.epochs * self.updates_per_epoch


_REGISTERED_TRACE_LAYOUT: Final = _TraceLayout(
    epochs=EPOCH_COUNT,
    updates_per_epoch=UPDATES_PER_EPOCH,
    population_count=TRAINING_EXAMPLE_COUNT,
    batch_size=TRAINING_BATCH_SIZE,
    last_batch_size=LAST_BATCH_SIZE,
    warmup_updates=UPDATES_PER_EPOCH,
    warmup_start_lr=0.0003,
    maximum_lr=0.003,
    minimum_lr=0.00003,
)


def _validate_trace_layout(layout: _TraceLayout) -> None:
    if type(layout) is not _TraceLayout:
        raise TypeError("layout must be a _TraceLayout")
    layout.__post_init__()


def _validate_update_trace_route_marker(route_marker: object | None) -> None:
    if route_marker is not None and route_marker is not _UPDATE_TRACE_ROUTE_MARKER:
        raise Experiment002TrainingEvidenceError(
            "update trace state has an invalid route marker"
        )


@dataclass(frozen=True, slots=True)
class _UpdateRecord:
    global_update: int
    batch_size: int
    learning_rate: float
    batch_mean_training_loss: np.float32
    returned_preclip_l2_norm: np.float32
    framed: bytes


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
class EpochUpdateTraceEvidence:
    """Issuer-only digest and CE for one epoch of supplied observations."""

    _seed: int
    _zero_based_epoch: int
    _sha256: str
    _training_cross_entropy: np.float64
    _route_marker: object | None

    def __init__(self) -> None:
        raise TypeError("epoch update trace evidence is issued by this module")

    @property
    def seed(self) -> int:
        return _issued_epoch_trace_state(self).seed

    @property
    def zero_based_epoch(self) -> int:
        return _issued_epoch_trace_state(self).zero_based_epoch

    @property
    def first_global_update(self) -> int:
        state = _issued_epoch_trace_state(self)
        return state.zero_based_epoch * state.layout.updates_per_epoch

    @property
    def last_global_update_inclusive(self) -> int:
        state = _issued_epoch_trace_state(self)
        return (state.zero_based_epoch + 1) * state.layout.updates_per_epoch - 1

    @property
    def update_count(self) -> int:
        return _issued_epoch_trace_state(self).layout.updates_per_epoch

    @property
    def sha256(self) -> str:
        return _issued_epoch_trace_state(self).sha256

    @property
    def training_cross_entropy(self) -> np.float64:
        return np.float64(_issued_epoch_trace_state(self).training_cross_entropy)

    @property
    def training_cross_entropy_hex(self) -> str:
        return float(_issued_epoch_trace_state(self).training_cross_entropy).hex()

    def __copy__(self) -> NoReturn:
        raise TypeError("epoch update trace evidence cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        del memo
        raise TypeError("epoch update trace evidence cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("epoch update trace evidence cannot be serialized")


@dataclass(frozen=True, slots=True)
class _IssuedEpochTraceState:
    seed: int
    zero_based_epoch: int
    records: tuple[_UpdateRecord, ...]
    framed_payload: bytes
    sha256: str
    training_cross_entropy: np.float64
    layout: _TraceLayout
    route_marker: object | None


_ISSUED_EPOCH_TRACES: weakref.WeakKeyDictionary[
    EpochUpdateTraceEvidence, _IssuedEpochTraceState
] = weakref.WeakKeyDictionary()
_ISSUED_EPOCH_TRACES_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
class CompleteUpdateTraceEvidence:
    """Issuer-only digest for all supplied observations from one seed run."""

    _seed: int
    _sha256: str
    _update_count: int
    _route_marker: object | None

    def __init__(self) -> None:
        raise TypeError("complete update trace evidence is issued by this module")

    @property
    def seed(self) -> int:
        return _issued_complete_trace_state(self).seed

    @property
    def update_count(self) -> int:
        return _issued_complete_trace_state(self).layout.total_updates

    @property
    def sha256(self) -> str:
        return _issued_complete_trace_state(self).sha256

    @property
    def epochs(self) -> tuple[EpochUpdateTraceEvidence, ...]:
        return _issued_complete_trace_state(self).epochs

    @property
    def epoch_update_trace_digests(self) -> tuple[str, ...]:
        return tuple(
            epoch.sha256 for epoch in _issued_complete_trace_state(self).epochs
        )

    def __copy__(self) -> NoReturn:
        raise TypeError("complete update trace evidence cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        del memo
        raise TypeError("complete update trace evidence cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("complete update trace evidence cannot be serialized")


@dataclass(frozen=True, slots=True)
class _IssuedCompleteTraceState:
    seed: int
    records: tuple[_UpdateRecord, ...]
    epochs: tuple[EpochUpdateTraceEvidence, ...]
    framed_payload: bytes
    sha256: str
    layout: _TraceLayout
    route_marker: object | None


_ISSUED_COMPLETE_TRACES: weakref.WeakKeyDictionary[
    CompleteUpdateTraceEvidence, _IssuedCompleteTraceState
] = weakref.WeakKeyDictionary()
_ISSUED_COMPLETE_TRACES_LOCK = threading.Lock()


class UpdateTraceAccumulator:
    """Validate and frame supplied observations from one ordered seed run.

    This accumulator does not establish optimizer execution on its own.  The
    registered trainer must append each observation from the same issuer-only
    completed-update receipt that closes the corresponding population batch.
    """

    __slots__ = (
        "_seed",
        "_layout",
        "_route_marker",
        "_records",
        "_epochs",
        "_complete",
        "_lock",
    )

    def __init__(self, seed: int) -> None:
        _require_registered_seed(seed)
        self._initialize(seed, _REGISTERED_TRACE_LAYOUT, _UPDATE_TRACE_ROUTE_MARKER)

    @classmethod
    def _for_layout(cls, seed: int, layout: _TraceLayout) -> UpdateTraceAccumulator:
        """Construct an unregistered small-layout accumulator for unit tests."""

        _require_uint32(seed, "seed")
        if type(layout) is not _TraceLayout:
            raise TypeError("layout must be a _TraceLayout")
        result = object.__new__(cls)
        result._initialize(seed, layout, None)
        return result

    def _initialize(
        self,
        seed: int,
        layout: _TraceLayout,
        route_marker: object | None,
    ) -> None:
        _require_uint32(seed, "seed")
        _validate_trace_layout(layout)
        _validate_update_trace_route_marker(route_marker)
        if route_marker is _UPDATE_TRACE_ROUTE_MARKER:
            _require_registered_seed(seed)
            if layout != _REGISTERED_TRACE_LAYOUT:
                raise Experiment002TrainingEvidenceError(
                    "registered update trace requires the registered layout"
                )
        self._seed = seed
        self._layout = layout
        self._route_marker = route_marker
        self._records: list[_UpdateRecord] = []
        self._epochs: list[EpochUpdateTraceEvidence] = []
        self._complete: CompleteUpdateTraceEvidence | None = None
        self._lock = threading.Lock()

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def next_global_update(self) -> int:
        with self._lock:
            return len(self._records)

    @property
    def completed_epoch_count(self) -> int:
        with self._lock:
            return len(self._epochs)

    def record_update(
        self,
        global_update: int,
        batch_size: int,
        learning_rate: float,
        batch_mean_training_loss: np.float32,
        returned_preclip_l2_norm: np.float32,
    ) -> None:
        """Append one exact update after checking order, schedule, and scalars."""

        with self._lock:
            if self._complete is not None:
                raise Experiment002TrainingEvidenceError(
                    "cannot append after the complete trace was issued"
                )
            expected_update = len(self._records)
            if expected_update >= self._layout.total_updates:
                raise Experiment002TrainingEvidenceError(
                    "the update trace already contains every expected update"
                )
            if expected_update == (len(self._epochs) + 1) * (
                self._layout.updates_per_epoch
            ):
                raise Experiment002TrainingEvidenceError(
                    "finish_epoch must be called before the next update"
                )
            _require_uint32(global_update, "global_update")
            if global_update != expected_update:
                raise Experiment002TrainingEvidenceError(
                    "global_update differs from the next ordered update"
                )
            expected_batch_size = _batch_size_for_update(global_update, self._layout)
            if type(batch_size) is not int:
                raise TypeError("batch_size must be an integer")
            if batch_size != expected_batch_size:
                raise Experiment002TrainingEvidenceError(
                    "batch_size differs from the registered batch layout"
                )
            expected_learning_rate = _learning_rate_for_update(
                global_update, self._layout
            )
            if type(learning_rate) is not float:
                raise TypeError("learning_rate must be a Python float")
            if not math.isfinite(learning_rate) or struct.pack(
                "<d", learning_rate
            ) != struct.pack("<d", expected_learning_rate):
                raise Experiment002TrainingEvidenceError(
                    "learning_rate differs from the exact registered schedule"
                )
            _require_float32_scalar(
                batch_mean_training_loss, "batch_mean_training_loss"
            )
            _require_float32_scalar(
                returned_preclip_l2_norm, "returned_preclip_l2_norm"
            )
            framed = struct.pack(
                "<IIdff",
                global_update,
                batch_size,
                learning_rate,
                batch_mean_training_loss,
                returned_preclip_l2_norm,
            )
            self._records.append(
                _UpdateRecord(
                    global_update=global_update,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    batch_mean_training_loss=np.float32(batch_mean_training_loss),
                    returned_preclip_l2_norm=np.float32(returned_preclip_l2_norm),
                    framed=framed,
                )
            )

    def finish_epoch(self) -> EpochUpdateTraceEvidence:
        """Issue evidence after exactly one further complete epoch."""

        with self._lock:
            if self._complete is not None:
                raise Experiment002TrainingEvidenceError(
                    "the complete trace was already issued"
                )
            epoch = len(self._epochs)
            if epoch >= self._layout.epochs:
                raise Experiment002TrainingEvidenceError(
                    "every epoch trace was already issued"
                )
            expected_update_count = (epoch + 1) * self._layout.updates_per_epoch
            if len(self._records) != expected_update_count:
                raise Experiment002TrainingEvidenceError(
                    "finish_epoch requires exactly one complete pending epoch"
                )
            start = epoch * self._layout.updates_per_epoch
            records = tuple(self._records[start:expected_update_count])
            evidence = _issue_epoch_trace(
                self._seed,
                epoch,
                records,
                layout=self._layout,
                route_marker=self._route_marker,
            )
            self._epochs.append(evidence)
            return evidence

    def finish(self) -> CompleteUpdateTraceEvidence:
        """Issue evidence after all epochs and all updates were completed."""

        with self._lock:
            if self._complete is not None:
                raise Experiment002TrainingEvidenceError(
                    "the complete trace was already issued"
                )
            if (
                len(self._records) != self._layout.total_updates
                or len(self._epochs) != self._layout.epochs
            ):
                raise Experiment002TrainingEvidenceError(
                    "finish requires every update and every epoch evidence"
                )
            evidence = _issue_complete_trace(
                self._seed,
                tuple(self._records),
                tuple(self._epochs),
                layout=self._layout,
                route_marker=self._route_marker,
            )
            self._complete = evidence
            return evidence

    def __copy__(self) -> NoReturn:
        raise TypeError("an update trace accumulator cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        del memo
        raise TypeError("an update trace accumulator cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("an update trace accumulator cannot be serialized")


def registered_learning_rate(global_update: int) -> float:
    """Return the exact registered binary64 schedule value for one update."""

    return _learning_rate_for_update(global_update, _REGISTERED_TRACE_LAYOUT)


def verify_registered_epoch_update_trace(
    evidence: EpochUpdateTraceEvidence,
) -> None:
    """Require an intact epoch capability from a registered seed route."""

    state = _issued_epoch_trace_state(evidence)
    if (
        state.route_marker is not _UPDATE_TRACE_ROUTE_MARKER
        or state.layout != _REGISTERED_TRACE_LAYOUT
        or state.seed not in REGISTERED_SEEDS
    ):
        raise Experiment002TrainingEvidenceError(
            "epoch update trace was not issued by the registered route"
        )


def verify_registered_complete_update_trace(
    evidence: CompleteUpdateTraceEvidence,
) -> None:
    """Require an intact complete capability from a registered seed route."""

    state = _issued_complete_trace_state(evidence)
    if (
        state.route_marker is not _UPDATE_TRACE_ROUTE_MARKER
        or state.layout != _REGISTERED_TRACE_LAYOUT
        or state.seed not in REGISTERED_SEEDS
    ):
        raise Experiment002TrainingEvidenceError(
            "complete update trace was not issued by the registered route"
        )
    for epoch in state.epochs:
        verify_registered_epoch_update_trace(epoch)


def _learning_rate_for_update(global_update: int, layout: _TraceLayout) -> float:
    _validate_trace_layout(layout)
    _require_uint32(global_update, "global_update")
    if global_update >= layout.total_updates:
        raise Experiment002TrainingEvidenceError(
            "global_update is outside the trace layout"
        )
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


def _batch_size_for_update(global_update: int, layout: _TraceLayout) -> int:
    update_in_epoch = global_update % layout.updates_per_epoch
    if update_in_epoch == layout.updates_per_epoch - 1:
        return layout.last_batch_size
    return layout.batch_size


def _issue_epoch_trace(
    seed: int,
    zero_based_epoch: int,
    records: tuple[_UpdateRecord, ...],
    *,
    layout: _TraceLayout,
    route_marker: object | None,
) -> EpochUpdateTraceEvidence:
    _validate_update_trace_route_marker(route_marker)
    _validate_epoch_records(seed, zero_based_epoch, records, layout)
    framed_payload = _frame_epoch_trace(seed, zero_based_epoch, records)
    sha256 = hashlib.sha256(EPOCH_UPDATE_TRACE_DOMAIN + framed_payload).hexdigest()
    training_cross_entropy = _training_cross_entropy(records, layout.population_count)
    result = object.__new__(EpochUpdateTraceEvidence)
    object.__setattr__(result, "_seed", seed)
    object.__setattr__(result, "_zero_based_epoch", zero_based_epoch)
    object.__setattr__(result, "_sha256", sha256)
    object.__setattr__(
        result, "_training_cross_entropy", np.float64(training_cross_entropy)
    )
    object.__setattr__(result, "_route_marker", route_marker)
    state = _IssuedEpochTraceState(
        seed=seed,
        zero_based_epoch=zero_based_epoch,
        records=records,
        framed_payload=framed_payload,
        sha256=sha256,
        training_cross_entropy=np.float64(training_cross_entropy),
        layout=layout,
        route_marker=route_marker,
    )
    with _ISSUED_EPOCH_TRACES_LOCK:
        _ISSUED_EPOCH_TRACES[result] = state
    return result


def _issue_complete_trace(
    seed: int,
    records: tuple[_UpdateRecord, ...],
    epochs: tuple[EpochUpdateTraceEvidence, ...],
    *,
    layout: _TraceLayout,
    route_marker: object | None,
) -> CompleteUpdateTraceEvidence:
    _validate_trace_layout(layout)
    _validate_update_trace_route_marker(route_marker)
    _require_uint32(seed, "seed")
    if len(records) != layout.total_updates or len(epochs) != layout.epochs:
        raise Experiment002TrainingEvidenceError(
            "complete trace has an invalid record or epoch count"
        )
    for epoch_index, epoch in enumerate(epochs):
        epoch_state = _issued_epoch_trace_state(epoch)
        if (
            epoch_state.seed != seed
            or epoch_state.zero_based_epoch != epoch_index
            or epoch_state.layout != layout
            or epoch_state.route_marker is not route_marker
        ):
            raise Experiment002TrainingEvidenceError(
                "complete trace contains mismatched epoch evidence"
            )
        start = epoch_index * layout.updates_per_epoch
        end = start + layout.updates_per_epoch
        if epoch_state.records != records[start:end]:
            raise Experiment002TrainingEvidenceError(
                "complete trace records differ from epoch evidence"
            )
    _validate_complete_records(records, layout)
    framed_payload = _frame_complete_trace(seed, records)
    sha256 = hashlib.sha256(COMPLETE_UPDATE_TRACE_DOMAIN + framed_payload).hexdigest()
    result = object.__new__(CompleteUpdateTraceEvidence)
    object.__setattr__(result, "_seed", seed)
    object.__setattr__(result, "_sha256", sha256)
    object.__setattr__(result, "_update_count", layout.total_updates)
    object.__setattr__(result, "_route_marker", route_marker)
    complete_state = _IssuedCompleteTraceState(
        seed=seed,
        records=records,
        epochs=epochs,
        framed_payload=framed_payload,
        sha256=sha256,
        layout=layout,
        route_marker=route_marker,
    )
    with _ISSUED_COMPLETE_TRACES_LOCK:
        _ISSUED_COMPLETE_TRACES[result] = complete_state
    return result


def _frame_epoch_trace(
    seed: int,
    zero_based_epoch: int,
    records: tuple[_UpdateRecord, ...],
) -> bytes:
    return struct.pack("<III", seed, zero_based_epoch, len(records)) + b"".join(
        record.framed for record in records
    )


def _frame_complete_trace(seed: int, records: tuple[_UpdateRecord, ...]) -> bytes:
    return struct.pack("<II", seed, len(records)) + b"".join(
        record.framed for record in records
    )


def _training_cross_entropy(
    records: tuple[_UpdateRecord, ...], population_count: int
) -> np.float64:
    total = np.float64(0.0)
    for record in records:
        weighted_loss = np.float64(record.batch_mean_training_loss) * np.float64(
            record.batch_size
        )
        total = np.float64(total + weighted_loss)
    result = np.float64(total / np.float64(population_count))
    if not np.isfinite(result):
        raise Experiment002TrainingEvidenceError(
            "training cross-entropy became non-finite"
        )
    return result


def _validate_epoch_records(
    seed: int,
    zero_based_epoch: int,
    records: tuple[_UpdateRecord, ...],
    layout: _TraceLayout,
) -> None:
    _validate_trace_layout(layout)
    _require_uint32(seed, "seed")
    if type(zero_based_epoch) is not int or not 0 <= zero_based_epoch < layout.epochs:
        raise Experiment002TrainingEvidenceError("zero_based_epoch is invalid")
    if type(records) is not tuple or len(records) != layout.updates_per_epoch:
        raise Experiment002TrainingEvidenceError(
            "epoch trace has an invalid update count"
        )
    first_update = zero_based_epoch * layout.updates_per_epoch
    for offset, record in enumerate(records):
        _validate_record(record, first_update + offset, layout)


def _validate_complete_records(
    records: tuple[_UpdateRecord, ...], layout: _TraceLayout
) -> None:
    _validate_trace_layout(layout)
    if type(records) is not tuple or len(records) != layout.total_updates:
        raise Experiment002TrainingEvidenceError(
            "complete trace has an invalid update count"
        )
    for global_update, record in enumerate(records):
        _validate_record(record, global_update, layout)


def _validate_record(
    record: _UpdateRecord,
    expected_global_update: int,
    layout: _TraceLayout,
) -> None:
    if type(record) is not _UpdateRecord:
        raise TypeError("trace records must be _UpdateRecord values")
    if type(record.global_update) is not int:
        raise TypeError("record global_update must be an integer")
    if type(record.batch_size) is not int:
        raise TypeError("record batch_size must be an integer")
    if type(record.learning_rate) is not float:
        raise TypeError("record learning_rate must be a Python float")
    _require_float32_scalar(record.batch_mean_training_loss, "batch_mean_training_loss")
    _require_float32_scalar(record.returned_preclip_l2_norm, "returned_preclip_l2_norm")
    if type(record.framed) is not bytes:
        raise TypeError("record framing must be bytes")
    expected_batch_size = _batch_size_for_update(expected_global_update, layout)
    expected_learning_rate = _learning_rate_for_update(expected_global_update, layout)
    expected_framed = struct.pack(
        "<IIdff",
        expected_global_update,
        expected_batch_size,
        expected_learning_rate,
        record.batch_mean_training_loss,
        record.returned_preclip_l2_norm,
    )
    if (
        record.global_update != expected_global_update
        or record.batch_size != expected_batch_size
        or struct.pack("<d", record.learning_rate)
        != struct.pack("<d", expected_learning_rate)
        or record.framed != expected_framed
    ):
        raise Experiment002TrainingEvidenceError(
            "an update record changed after append"
        )


def _validate_issued_epoch_trace_state(state: _IssuedEpochTraceState) -> None:
    if type(state) is not _IssuedEpochTraceState:
        raise TypeError("issued epoch trace state has an invalid type")
    _validate_trace_layout(state.layout)
    _validate_update_trace_route_marker(state.route_marker)
    _validate_epoch_records(
        state.seed, state.zero_based_epoch, state.records, state.layout
    )
    if type(state.framed_payload) is not bytes:
        raise TypeError("issued epoch trace framing must be bytes")
    if type(state.sha256) is not str:
        raise TypeError("issued epoch trace digest must be a string")
    if type(state.training_cross_entropy) is not np.float64:
        raise TypeError("issued epoch cross-entropy must be a NumPy float64 scalar")
    if not np.isfinite(state.training_cross_entropy):
        raise Experiment002TrainingEvidenceError(
            "issued epoch cross-entropy must be finite"
        )


def _validate_issued_complete_trace_state(state: _IssuedCompleteTraceState) -> None:
    if type(state) is not _IssuedCompleteTraceState:
        raise TypeError("issued complete trace state has an invalid type")
    _validate_trace_layout(state.layout)
    _validate_update_trace_route_marker(state.route_marker)
    _require_uint32(state.seed, "seed")
    _validate_complete_records(state.records, state.layout)
    if type(state.epochs) is not tuple or len(state.epochs) != state.layout.epochs:
        raise Experiment002TrainingEvidenceError(
            "issued complete trace epochs are invalid"
        )
    if any(type(epoch) is not EpochUpdateTraceEvidence for epoch in state.epochs):
        raise TypeError("issued complete trace epochs have invalid types")
    if type(state.framed_payload) is not bytes:
        raise TypeError("issued complete trace framing must be bytes")
    if type(state.sha256) is not str:
        raise TypeError("issued complete trace digest must be a string")


def _issued_epoch_trace_state(
    evidence: EpochUpdateTraceEvidence,
) -> _IssuedEpochTraceState:
    if type(evidence) is not EpochUpdateTraceEvidence:
        raise TypeError("evidence must be EpochUpdateTraceEvidence")
    with _ISSUED_EPOCH_TRACES_LOCK:
        state = _ISSUED_EPOCH_TRACES.get(evidence)
    if state is None:
        raise Experiment002TrainingEvidenceError(
            "epoch update trace evidence was not issued by this process"
        )
    _validate_issued_epoch_trace_state(state)
    observed_payload = _frame_epoch_trace(
        state.seed, state.zero_based_epoch, state.records
    )
    observed_sha256 = hashlib.sha256(
        EPOCH_UPDATE_TRACE_DOMAIN + observed_payload
    ).hexdigest()
    observed_ce = _training_cross_entropy(state.records, state.layout.population_count)
    try:
        raw_seed = evidence._seed
        raw_epoch = evidence._zero_based_epoch
        raw_sha256 = evidence._sha256
        raw_ce = evidence._training_cross_entropy
        raw_route_marker = evidence._route_marker
    except AttributeError as error:
        raise Experiment002TrainingEvidenceError(
            "epoch update trace capability is incomplete"
        ) from error
    if (
        observed_payload != state.framed_payload
        or observed_sha256 != state.sha256
        or observed_ce.tobytes() != state.training_cross_entropy.tobytes()
        or type(raw_seed) is not int
        or raw_seed != state.seed
        or type(raw_epoch) is not int
        or raw_epoch != state.zero_based_epoch
        or type(raw_sha256) is not str
        or raw_sha256 != state.sha256
        or type(raw_ce) is not np.float64
        or raw_ce.tobytes() != state.training_cross_entropy.tobytes()
        or raw_route_marker is not state.route_marker
    ):
        raise Experiment002TrainingEvidenceError(
            "epoch update trace capability changed after issuance"
        )
    return state


def _issued_complete_trace_state(
    evidence: CompleteUpdateTraceEvidence,
) -> _IssuedCompleteTraceState:
    if type(evidence) is not CompleteUpdateTraceEvidence:
        raise TypeError("evidence must be CompleteUpdateTraceEvidence")
    with _ISSUED_COMPLETE_TRACES_LOCK:
        state = _ISSUED_COMPLETE_TRACES.get(evidence)
    if state is None:
        raise Experiment002TrainingEvidenceError(
            "complete update trace evidence was not issued by this process"
        )
    _validate_issued_complete_trace_state(state)
    for epoch_index, epoch in enumerate(state.epochs):
        epoch_state = _issued_epoch_trace_state(epoch)
        start = epoch_index * state.layout.updates_per_epoch
        end = start + state.layout.updates_per_epoch
        if (
            epoch_state.seed != state.seed
            or epoch_state.zero_based_epoch != epoch_index
            or epoch_state.records != state.records[start:end]
            or epoch_state.layout != state.layout
            or epoch_state.route_marker is not state.route_marker
        ):
            raise Experiment002TrainingEvidenceError(
                "complete update trace epoch capability changed"
            )
    observed_payload = _frame_complete_trace(state.seed, state.records)
    observed_sha256 = hashlib.sha256(
        COMPLETE_UPDATE_TRACE_DOMAIN + observed_payload
    ).hexdigest()
    try:
        raw_seed = evidence._seed
        raw_sha256 = evidence._sha256
        raw_update_count = evidence._update_count
        raw_route_marker = evidence._route_marker
    except AttributeError as error:
        raise Experiment002TrainingEvidenceError(
            "complete update trace capability is incomplete"
        ) from error
    if (
        observed_payload != state.framed_payload
        or observed_sha256 != state.sha256
        or type(raw_seed) is not int
        or raw_seed != state.seed
        or type(raw_sha256) is not str
        or raw_sha256 != state.sha256
        or type(raw_update_count) is not int
        or raw_update_count != state.layout.total_updates
        or raw_route_marker is not state.route_marker
    ):
        raise Experiment002TrainingEvidenceError(
            "complete update trace capability changed after issuance"
        )
    return state


def _require_registered_seed(seed: int) -> None:
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    if seed not in REGISTERED_SEEDS:
        raise Experiment002TrainingEvidenceError(
            "seed is not one of the three registered training seeds"
        )


def _require_registered_epoch(zero_based_epoch: int) -> None:
    if type(zero_based_epoch) is not int:
        raise TypeError("zero_based_epoch must be an integer")
    if not 0 <= zero_based_epoch < EPOCH_COUNT:
        raise Experiment002TrainingEvidenceError(
            "zero_based_epoch is outside the registered training run"
        )


def _require_optional_model_context(
    seed: int | None,
    zero_based_epoch: int | None,
) -> None:
    if (seed is None) != (zero_based_epoch is None):
        raise Experiment002TrainingEvidenceError(
            "model tensor seed and epoch must be supplied together"
        )
    if seed is not None:
        _require_uint32(seed, "seed")
    if zero_based_epoch is not None:
        _require_uint32(zero_based_epoch, "zero_based_epoch")


def _require_float32_scalar(value: object, name: str) -> None:
    if type(value) is not np.float32:
        raise TypeError(f"{name} must be a NumPy float32 scalar")
    if not np.isfinite(value):
        raise Experiment002TrainingEvidenceError(f"{name} must be finite")


def _same_optional_int(left: object, right: int | None) -> bool:
    if right is None:
        return left is None
    return type(left) is int and left == right
