"""Unregistered model-evaluation kernel for Experiment 002.

This module intentionally contains no registered corpus, validation-evidence,
metric, history, checkpoint, or artifact route.  It only exercises the exact
CausalKWS evaluation mechanics on caller-supplied synthetic arrays and returns
an explicitly unregistered result.
"""

from __future__ import annotations

import hashlib
import struct
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from typing import Final, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.nn.modules import module as torch_module

from falsewake.causal_kws import (
    BLOCK_DILATIONS,
    DEPTHWISE_KERNEL_SIZE,
    ENCODER_CHANNELS,
    INPUT_MEL_BINS,
    MODEL_RECEPTIVE_FIELD_FRAMES,
    OUTPUT_CLASSES,
    POOL_HISTORY_FRAMES,
    TRAINABLE_PARAMETER_COUNT,
    CausalDepthwiseSeparableBlock,
    CausalKWS,
    CausalKWSState,
    PerFrameLayerNorm,
)

type Float32Array = NDArray[np.float32]

_MODEL_TENSOR_COUNT: Final = 53
_MODEL_TENSOR_DOMAIN: Final = b"falsewake-exp002-model-tensors-v1\0"
_TORCH_RNG_DOMAIN: Final = b"falsewake-exp002-torch-cpu-rng-v1\0"
_FRAME_COUNT: Final = MODEL_RECEPTIVE_FIELD_FRAMES
_FRAME_97: Final = _FRAME_COUNT - 1
_MAX_UNREGISTERED_EXAMPLES: Final = 8
_LOCAL_HOOK_REGISTRIES: Final = (
    "_forward_pre_hooks",
    "_forward_pre_hooks_with_kwargs",
    "_forward_hooks",
    "_forward_hooks_with_kwargs",
    "_forward_hooks_always_called",
    "_backward_pre_hooks",
    "_backward_hooks",
    "_state_dict_pre_hooks",
    "_state_dict_hooks",
    "_load_state_dict_pre_hooks",
    "_load_state_dict_post_hooks",
)
_GLOBAL_HOOK_REGISTRIES: Final = (
    "_global_backward_hooks",
    "_global_backward_pre_hooks",
    "_global_buffer_registration_hooks",
    "_global_forward_hooks",
    "_global_forward_hooks_always_called",
    "_global_forward_hooks_with_kwargs",
    "_global_forward_pre_hooks",
    "_global_module_registration_hooks",
    "_global_parameter_registration_hooks",
)
_FORBIDDEN_INSTANCE_METHOD_OVERRIDES: Final = (
    "__call__",
    "_call_impl",
    "_compiled_call_impl",
    "_slow_forward",
    "_wrapped_call_impl",
    "children",
    "eval",
    "forward",
    "forward_stream",
    "initial_state",
    "modules",
    "named_children",
    "named_modules",
    "named_parameters",
    "parameters",
    "state_dict",
    "train",
)


class Experiment002EvaluatorError(ValueError):
    """The unregistered evaluation kernel violated its exact contract."""


@dataclass(frozen=True, slots=True)
class _EvaluationLayout:
    """Private small-population seam for pre-registration kernel tests."""

    example_count: int
    batch_size: int

    def __post_init__(self) -> None:
        _require_positive_int(self.example_count, "example_count")
        _require_positive_int(self.batch_size, "batch_size")
        if self.example_count > _MAX_UNREGISTERED_EXAMPLES:
            raise Experiment002EvaluatorError(
                "unregistered evaluation exceeds the synthetic example limit"
            )

    @property
    def batch_count(self) -> int:
        """Return the exact number of direct contiguous slices."""

        return (self.example_count + self.batch_size - 1) // self.batch_size


@dataclass(frozen=True, slots=True)
class _UnregisteredEvaluationResult:
    """Immutable bytes from a synthetic evaluation, never registered evidence."""

    example_count: int
    batch_count: int
    model_inputs_sha256: str
    model_tensor_sha256: str
    torch_rng_sha256: str
    frame_97_logits_payload: bytes
    model_training_after: bool

    @property
    def frame_97_logits(self) -> Float32Array:
        """Return a fresh owned, read-only C-contiguous float32 array."""

        result = (
            np.frombuffer(
                self.frame_97_logits_payload,
                dtype=np.dtype("<f4"),
            )
            .reshape(self.example_count, OUTPUT_CLASSES)
            .copy(order="C")
        )
        result.setflags(write=False)
        return result


@dataclass(frozen=True, slots=True)
class _ModelParameterAuthority:
    """Exact live parameter objects owned by the caller's model at entry."""

    names: tuple[str, ...]
    parameters: tuple[torch.nn.Parameter, ...]
    requires_grad: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class _ModelStructureAuthority:
    """Exact module objects and frozen architecture attributes at entry."""

    modules: tuple[torch.nn.Module, ...]


def _evaluate_unregistered_causal_model(
    model: CausalKWS,
    model_inputs: Float32Array,
    *,
    layout: _EvaluationLayout,
    another_training_epoch: bool,
) -> _UnregisteredEvaluationResult:
    """Evaluate synthetic arrays without issuing any registered capability."""

    if type(model) is not CausalKWS:
        raise TypeError("model must be an exact CausalKWS")
    if type(layout) is not _EvaluationLayout:
        raise TypeError("layout must be an exact _EvaluationLayout")
    layout.__post_init__()
    if type(another_training_epoch) is not bool:
        raise TypeError("another_training_epoch must be a boolean")
    structure_authority = _capture_model_structure_authority(model)
    _require_model_ready_for_evaluation(model)
    _require_model_inputs(model_inputs, layout)

    parameter_authority = _capture_model_parameter_authority(model)
    input_payload_before = model_inputs.tobytes(order="C")
    input_sha256 = hashlib.sha256(input_payload_before).hexdigest()
    model_sha256_before = _stable_model_tensor_sha256(model)
    gradients_before = _gradient_snapshot(model)
    rng_before = _torch_rng_state()
    rng_sha256 = _torch_rng_sha256(rng_before)

    frame_97_logits = np.empty(
        (layout.example_count, OUTPUT_CLASSES),
        dtype=np.float32,
        order="C",
    )
    try:
        model.eval()
        _require_uniform_model_mode(model, training=False)
        with torch.inference_mode():
            for batch_index, start in enumerate(
                range(0, layout.example_count, layout.batch_size)
            ):
                stop = min(start + layout.batch_size, layout.example_count)
                batch = model_inputs[start:stop]
                _require_direct_batch_slice(
                    batch,
                    model_inputs,
                    expected_size=stop - start,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    features = torch.from_numpy(batch)
                _require_direct_tensor_view(features, batch)
                initial_state = model.initial_state(
                    stop - start,
                    device=features.device,
                    dtype=features.dtype,
                )
                _require_stream_state(
                    initial_state,
                    batch_size=stop - start,
                    expected_frames=0,
                    require_positive_zero=True,
                )
                initial_payload = _stream_state_payload(initial_state)
                logits, next_state = model.forward_stream(features, initial_state)
                _require_logits(logits, batch_size=stop - start)
                _require_stream_state(
                    next_state,
                    batch_size=stop - start,
                    expected_frames=_FRAME_COUNT,
                    require_positive_zero=False,
                )
                if _stream_state_payload(initial_state) != initial_payload:
                    raise Experiment002EvaluatorError(
                        "model forward mutated its fresh initial state"
                    )
                retained = logits[:, _FRAME_97, :].contiguous()
                _require_retained_logits(retained, batch_size=stop - start)
                frame_97_logits[start:stop] = retained.numpy()
                if batch_index + 1 > layout.batch_count:
                    raise Experiment002EvaluatorError(
                        "evaluation produced too many batches"
                    )
    finally:
        if another_training_epoch:
            model.train()
        else:
            model.eval()

    _require_uniform_model_mode(model, training=another_training_epoch)
    _require_model_structure_authority(model, structure_authority)
    _require_model_parameter_authority(model, parameter_authority)

    _require_model_inputs(model_inputs, layout)
    input_payload_after = model_inputs.tobytes(order="C")
    if input_payload_after != input_payload_before:
        raise Experiment002EvaluatorError("model input bytes changed during evaluation")
    if _stable_model_tensor_sha256(model) != model_sha256_before:
        raise Experiment002EvaluatorError(
            "model tensor bytes changed during evaluation"
        )
    if _gradient_snapshot(model) != gradients_before:
        raise Experiment002EvaluatorError("model gradients changed during evaluation")
    rng_after = _torch_rng_state()
    if rng_after != rng_before:
        raise Experiment002EvaluatorError("Torch CPU RNG changed during evaluation")
    if not np.all(np.isfinite(frame_97_logits)):
        raise Experiment002EvaluatorError(
            "retained frame-97 logits contain non-finite values"
        )
    if frame_97_logits.shape != (layout.example_count, OUTPUT_CLASSES) or (
        frame_97_logits.dtype != np.dtype(np.float32)
        or not frame_97_logits.flags.c_contiguous
        or not frame_97_logits.flags.owndata
    ):
        raise Experiment002EvaluatorError(
            "retained frame-97 logits have invalid storage"
        )
    payload = frame_97_logits.astype("<f4", copy=False).tobytes(order="C")
    expected_payload_size = layout.example_count * OUTPUT_CLASSES * 4
    if len(payload) != expected_payload_size:
        raise Experiment002EvaluatorError(
            "retained frame-97 logits have an invalid byte count"
        )
    return _UnregisteredEvaluationResult(
        example_count=layout.example_count,
        batch_count=layout.batch_count,
        model_inputs_sha256=input_sha256,
        model_tensor_sha256=model_sha256_before,
        torch_rng_sha256=rng_sha256,
        frame_97_logits_payload=payload,
        model_training_after=another_training_epoch,
    )


def _require_model_ready_for_evaluation(model: CausalKWS) -> None:
    _require_uniform_model_mode(model, training=True)
    named_parameters = tuple(model.named_parameters())
    if len(named_parameters) != _MODEL_TENSOR_COUNT:
        raise Experiment002EvaluatorError(
            "model parameter tensor count differs from the contract"
        )
    if sum(parameter.numel() for _, parameter in named_parameters) != (
        TRAINABLE_PARAMETER_COUNT
    ):
        raise Experiment002EvaluatorError(
            "model parameter value count differs from the contract"
        )
    if len({name for name, _ in named_parameters}) != len(named_parameters):
        raise Experiment002EvaluatorError("model parameter names are not unique")
    if len({id(parameter) for _, parameter in named_parameters}) != len(
        named_parameters
    ):
        raise Experiment002EvaluatorError("model parameter objects are not unique")
    for name, parameter in named_parameters:
        _require_float32_tensor(
            parameter,
            name=f"parameter {name}",
            shape=tuple(parameter.shape),
        )
        if not parameter.requires_grad:
            raise Experiment002EvaluatorError(
                f"parameter {name} unexpectedly disables gradients"
            )


def _capture_model_parameter_authority(
    model: CausalKWS,
) -> _ModelParameterAuthority:
    named_parameters = tuple(model.named_parameters())
    authority = _ModelParameterAuthority(
        names=tuple(name for name, _ in named_parameters),
        parameters=tuple(parameter for _, parameter in named_parameters),
        requires_grad=tuple(
            parameter.requires_grad for _, parameter in named_parameters
        ),
    )
    _require_model_parameter_authority(model, authority)
    return authority


def _require_model_parameter_authority(
    model: CausalKWS,
    authority: _ModelParameterAuthority,
) -> None:
    if type(authority) is not _ModelParameterAuthority:
        raise TypeError("authority must be an exact _ModelParameterAuthority")
    named_parameters = tuple(model.named_parameters())
    names = tuple(name for name, _ in named_parameters)
    parameters = tuple(parameter for _, parameter in named_parameters)
    requires_grad = tuple(parameter.requires_grad for parameter in parameters)
    if (
        names != authority.names
        or len(parameters) != len(authority.parameters)
        or any(
            observed is not expected
            for observed, expected in zip(
                parameters,
                authority.parameters,
                strict=True,
            )
        )
        or requires_grad != authority.requires_grad
        or any(type(parameter) is not torch.nn.Parameter for parameter in parameters)
        or any(not value for value in requires_grad)
    ):
        raise Experiment002EvaluatorError(
            "model parameter identity or gradient authority changed during evaluation"
        )


def _capture_model_structure_authority(model: CausalKWS) -> _ModelStructureAuthority:
    modules = _exact_model_modules(model)
    authority = _ModelStructureAuthority(modules=modules)
    _require_model_structure_authority(model, authority)
    return authority


def _require_model_structure_authority(
    model: CausalKWS,
    authority: _ModelStructureAuthority,
) -> None:
    if type(authority) is not _ModelStructureAuthority:
        raise TypeError("authority must be an exact _ModelStructureAuthority")
    modules = _exact_model_modules(model)
    if len(modules) != len(authority.modules) or any(
        observed is not expected
        for observed, expected in zip(modules, authority.modules, strict=True)
    ):
        raise Experiment002EvaluatorError(
            "model module identity changed during evaluation"
        )


def _exact_model_modules(model: CausalKWS) -> tuple[torch.nn.Module, ...]:
    if type(model) is not CausalKWS or tuple(model._modules) != (
        "stem",
        "stem_norm",
        "stem_activation",
        "blocks",
        "dropout",
        "classifier",
    ):
        raise Experiment002EvaluatorError("CausalKWS module topology changed")
    _require_conv1d(
        model.stem,
        name="stem",
        in_channels=INPUT_MEL_BINS,
        out_channels=ENCODER_CHANNELS,
        kernel_size=1,
        dilation=1,
        groups=1,
    )
    _require_per_frame_layer_norm(model.stem_norm, "stem_norm")
    _require_relu(model.stem_activation, "stem_activation")
    if type(model.blocks) is not torch.nn.ModuleList or len(model.blocks) != len(
        BLOCK_DILATIONS
    ):
        raise Experiment002EvaluatorError("CausalKWS block topology changed")

    expected_modules: list[torch.nn.Module] = [
        model,
        model.stem,
        model.stem_norm,
        model.stem_norm.norm,
        model.stem_activation,
        model.blocks,
    ]
    for index, (block, dilation) in enumerate(
        zip(model.blocks, BLOCK_DILATIONS, strict=True)
    ):
        _require_causal_block(block, index=index, dilation=dilation)
        exact_block = cast(CausalDepthwiseSeparableBlock, block)
        expected_modules.extend(
            (
                exact_block,
                exact_block.depthwise,
                exact_block.depthwise_norm,
                exact_block.depthwise_norm.norm,
                exact_block.pointwise,
                exact_block.pointwise_norm,
                exact_block.pointwise_norm.norm,
                exact_block.activation,
            )
        )
    if (
        type(model.dropout) is not torch.nn.Dropout
        or not _same_float(model.dropout.p, 0.10)
        or model.dropout.inplace is not False
    ):
        raise Experiment002EvaluatorError("CausalKWS dropout configuration changed")
    if (
        type(model.classifier) is not torch.nn.Linear
        or model.classifier.in_features != ENCODER_CHANNELS
        or model.classifier.out_features != OUTPUT_CLASSES
        or type(model.classifier.weight) is not torch.nn.Parameter
        or type(model.classifier.bias) is not torch.nn.Parameter
    ):
        raise Experiment002EvaluatorError("CausalKWS classifier configuration changed")
    expected_modules.extend((model.dropout, model.classifier))
    _require_no_execution_hooks_or_overrides(tuple(expected_modules))
    observed_modules = tuple(model.modules())
    result = tuple(expected_modules)
    if len(observed_modules) != len(result) or any(
        observed is not expected
        for observed, expected in zip(observed_modules, result, strict=True)
    ):
        raise Experiment002EvaluatorError("CausalKWS module traversal changed")
    return result


def _require_no_execution_hooks_or_overrides(
    modules: tuple[torch.nn.Module, ...],
) -> None:
    for name in _GLOBAL_HOOK_REGISTRIES:
        registry = getattr(torch_module, name, None)
        if type(registry) is not OrderedDict or registry:
            raise Experiment002EvaluatorError(
                f"global Torch hook registry is not empty ({name})"
            )
    if torch_module._global_is_full_backward_hook is not None:
        raise Experiment002EvaluatorError("global Torch backward-hook mode is active")
    for index, module in enumerate(modules):
        if type(module) is not torch.nn.Module and not isinstance(
            module,
            torch.nn.Module,
        ):
            raise Experiment002EvaluatorError(
                f"model traversal contains an invalid module ({index})"
            )
        for name in _LOCAL_HOOK_REGISTRIES:
            registry = module.__dict__.get(name)
            if type(registry) is not OrderedDict or registry:
                raise Experiment002EvaluatorError(
                    f"model hook registry is not empty ({index}:{name})"
                )
        if module.__dict__.get("_is_full_backward_hook") is not None:
            raise Experiment002EvaluatorError(
                f"model backward-hook mode is active ({index})"
            )
        if getattr(module, "_compiled_call_impl", None) is not None:
            raise Experiment002EvaluatorError(
                f"model compiled call path is active ({index})"
            )
        overridden = tuple(
            name
            for name in _FORBIDDEN_INSTANCE_METHOD_OVERRIDES
            if name in module.__dict__
        )
        if overridden:
            raise Experiment002EvaluatorError(
                "model contains an instance execution-method override "
                f"({index}:{','.join(overridden)})"
            )


def _require_causal_block(
    block: torch.nn.Module,
    *,
    index: int,
    dilation: int,
) -> None:
    name = f"blocks.{index}"
    if (
        type(block) is not CausalDepthwiseSeparableBlock
        or tuple(block._modules)
        != (
            "depthwise",
            "depthwise_norm",
            "pointwise",
            "pointwise_norm",
            "activation",
        )
        or block.channels != ENCODER_CHANNELS
        or block.dilation != dilation
        or block.history_frames != (DEPTHWISE_KERNEL_SIZE - 1) * dilation
    ):
        raise Experiment002EvaluatorError(f"{name} configuration changed")
    _require_conv1d(
        block.depthwise,
        name=f"{name}.depthwise",
        in_channels=ENCODER_CHANNELS,
        out_channels=ENCODER_CHANNELS,
        kernel_size=DEPTHWISE_KERNEL_SIZE,
        dilation=dilation,
        groups=ENCODER_CHANNELS,
    )
    _require_per_frame_layer_norm(block.depthwise_norm, f"{name}.depthwise_norm")
    _require_conv1d(
        block.pointwise,
        name=f"{name}.pointwise",
        in_channels=ENCODER_CHANNELS,
        out_channels=ENCODER_CHANNELS,
        kernel_size=1,
        dilation=1,
        groups=1,
    )
    _require_per_frame_layer_norm(block.pointwise_norm, f"{name}.pointwise_norm")
    _require_relu(block.activation, f"{name}.activation")


def _require_conv1d(
    module: torch.nn.Module,
    *,
    name: str,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    dilation: int,
    groups: int,
) -> None:
    if (
        type(module) is not torch.nn.Conv1d
        or module.in_channels != in_channels
        or module.out_channels != out_channels
        or module.kernel_size != (kernel_size,)
        or module.stride != (1,)
        or module.padding != (0,)
        or module.dilation != (dilation,)
        or module.groups != groups
        or module.padding_mode != "zeros"
        or module.bias is not None
        or type(module.weight) is not torch.nn.Parameter
    ):
        raise Experiment002EvaluatorError(f"{name} Conv1d configuration changed")


def _require_per_frame_layer_norm(
    module: torch.nn.Module,
    name: str,
) -> None:
    if (
        type(module) is not PerFrameLayerNorm
        or tuple(module._modules) != ("norm",)
        or type(module.norm) is not torch.nn.LayerNorm
        or module.norm.normalized_shape != (ENCODER_CHANNELS,)
        or not _same_float(module.norm.eps, 1e-5)
        or module.norm.elementwise_affine is not True
        or type(module.norm.weight) is not torch.nn.Parameter
        or type(module.norm.bias) is not torch.nn.Parameter
    ):
        raise Experiment002EvaluatorError(f"{name} LayerNorm configuration changed")


def _require_relu(module: torch.nn.Module, name: str) -> None:
    if type(module) is not torch.nn.ReLU or module.inplace is not False:
        raise Experiment002EvaluatorError(f"{name} ReLU configuration changed")


def _same_float(value: object, expected: float) -> bool:
    return type(value) is float and struct.pack("<d", value) == struct.pack(
        "<d", expected
    )


def _require_model_inputs(
    model_inputs: Float32Array,
    layout: _EvaluationLayout,
) -> None:
    if type(model_inputs) is not np.ndarray:
        raise TypeError("model_inputs must be an exact NumPy array")
    expected_shape = (layout.example_count, INPUT_MEL_BINS, _FRAME_COUNT)
    if model_inputs.dtype != np.dtype(np.float32) or model_inputs.shape != (
        expected_shape
    ):
        raise Experiment002EvaluatorError("model_inputs have an invalid dtype or shape")
    if (
        not model_inputs.flags.c_contiguous
        or not model_inputs.flags.owndata
        or model_inputs.flags.writeable
    ):
        raise Experiment002EvaluatorError(
            "model_inputs must be owned, read-only, and C-contiguous"
        )
    if not np.all(np.isfinite(model_inputs)):
        raise Experiment002EvaluatorError("model_inputs contain non-finite values")


def _require_direct_batch_slice(
    batch: Float32Array,
    owner: Float32Array,
    *,
    expected_size: int,
) -> None:
    expected_shape = (expected_size, INPUT_MEL_BINS, _FRAME_COUNT)
    if (
        type(batch) is not np.ndarray
        or batch.dtype != np.dtype(np.float32)
        or batch.shape != expected_shape
        or not batch.flags.c_contiguous
        or batch.flags.owndata
        or batch.flags.writeable
        or not np.shares_memory(batch, owner)
    ):
        raise Experiment002EvaluatorError(
            "evaluation batch is not a direct contiguous input slice"
        )
    if not np.all(np.isfinite(batch)):
        raise Experiment002EvaluatorError("evaluation batch contains non-finite values")


def _require_direct_tensor_view(features: Tensor, batch: Float32Array) -> None:
    if (
        type(features) is not Tensor
        or features.device.type != "cpu"
        or features.dtype != torch.float32
        or tuple(features.shape) != tuple(batch.shape)
        or not features.is_contiguous()
        or features.requires_grad
        or not features.is_inference()
        or features.data_ptr() != int(batch.__array_interface__["data"][0])
    ):
        raise Experiment002EvaluatorError(
            "torch.from_numpy did not preserve the direct CPU float32 view"
        )


def _require_logits(logits: Tensor, *, batch_size: int) -> None:
    _require_float32_tensor(
        logits,
        name="all-frame logits",
        shape=(batch_size, _FRAME_COUNT, OUTPUT_CLASSES),
    )
    _require_plain_inference_tensor(logits, "all-frame logits")


def _require_retained_logits(logits: Tensor, *, batch_size: int) -> None:
    _require_float32_tensor(
        logits,
        name="frame-97 logits",
        shape=(batch_size, OUTPUT_CLASSES),
    )
    _require_plain_inference_tensor(logits, "frame-97 logits")


def _require_stream_state(
    state: CausalKWSState,
    *,
    batch_size: int,
    expected_frames: int,
    require_positive_zero: bool,
) -> None:
    if type(state) is not CausalKWSState:
        raise Experiment002EvaluatorError(
            "model returned an invalid streaming-state type"
        )
    expected_histories = tuple(
        (batch_size, ENCODER_CHANNELS, (DEPTHWISE_KERNEL_SIZE - 1) * dilation)
        for dilation in BLOCK_DILATIONS
    )
    for index, (history, shape) in enumerate(
        zip(state.depthwise_states, expected_histories, strict=True)
    ):
        _require_float32_tensor(history, name=f"dw_state_{index}", shape=shape)
        _require_plain_inference_tensor(history, f"dw_state_{index}")
        if require_positive_zero:
            _require_positive_zero_tensor(history, f"dw_state_{index}")
    _require_float32_tensor(
        state.pool_state,
        name="pool_state",
        shape=(batch_size, ENCODER_CHANNELS, POOL_HISTORY_FRAMES),
    )
    _require_plain_inference_tensor(state.pool_state, "pool_state")
    if require_positive_zero:
        _require_positive_zero_tensor(state.pool_state, "pool_state")
    frames_seen = state.frames_seen
    if (
        type(frames_seen) is not Tensor
        or frames_seen.device.type != "cpu"
        or frames_seen.dtype != torch.int64
        or tuple(frames_seen.shape) != (batch_size,)
        or not frames_seen.is_contiguous()
        or frames_seen.requires_grad
        or not frames_seen.is_inference()
        or torch.any(frames_seen != expected_frames).item()
    ):
        raise Experiment002EvaluatorError("frames_seen violates the exact contract")


def _require_float32_tensor(
    tensor: Tensor,
    *,
    name: str,
    shape: tuple[int, ...],
) -> None:
    if type(tensor) is not Tensor and type(tensor) is not torch.nn.Parameter:
        raise Experiment002EvaluatorError(f"{name} has an invalid tensor type")
    if (
        tensor.device.type != "cpu"
        or tensor.dtype != torch.float32
        or tuple(tensor.shape) != shape
        or tensor.layout != torch.strided
        or not tensor.is_contiguous()
        or not torch.all(torch.isfinite(tensor)).item()
    ):
        raise Experiment002EvaluatorError(
            f"{name} has invalid dtype, shape, device, layout, or values"
        )


def _require_positive_zero_tensor(tensor: Tensor, name: str) -> None:
    if torch.any(tensor != 0).item() or torch.any(torch.signbit(tensor)).item():
        raise Experiment002EvaluatorError(f"{name} is not fresh positive-zero state")


def _require_plain_inference_tensor(tensor: Tensor, name: str) -> None:
    if type(tensor) is not Tensor or tensor.requires_grad or not tensor.is_inference():
        raise Experiment002EvaluatorError(
            f"{name} must be an exact plain inference tensor without gradients"
        )


def _stream_state_payload(state: CausalKWSState) -> bytes:
    values = (*state.depthwise_states, state.pool_state, state.frames_seen)
    payload = bytearray()
    for tensor in values:
        payload.extend(tensor.detach().numpy().tobytes(order="C"))
    return bytes(payload)


def _stable_model_tensor_sha256(model: CausalKWS) -> str:
    first = _model_tensor_payload(model)
    second = _model_tensor_payload(model)
    if first != second:
        raise Experiment002EvaluatorError(
            "model tensor bytes changed while taking a digest"
        )
    return hashlib.sha256(_MODEL_TENSOR_DOMAIN + first).hexdigest()


def _model_tensor_payload(model: CausalKWS) -> bytes:
    named_parameters = tuple(model.named_parameters())
    if (
        len(named_parameters) != _MODEL_TENSOR_COUNT
        or sum(parameter.numel() for _, parameter in named_parameters)
        != TRAINABLE_PARAMETER_COUNT
    ):
        raise Experiment002EvaluatorError("model parameter layout changed")
    names = tuple(name for name, _ in named_parameters)
    state_dict = model.state_dict()
    if len(state_dict) != len(names) or set(state_dict) != set(names):
        raise Experiment002EvaluatorError(
            "model state_dict differs from named parameters"
        )
    framed = bytearray(struct.pack("<I", len(names)))
    for name in sorted(names, key=lambda value: value.encode("utf-8")):
        tensor = state_dict[name]
        _require_float32_tensor(
            tensor,
            name=f"state_dict tensor {name}",
            shape=tuple(tensor.shape),
        )
        name_bytes = name.encode("utf-8")
        tensor_payload = (
            tensor.detach().numpy().astype("<f4", copy=False).tobytes(order="C")
        )
        framed.extend(struct.pack("<I", len(name_bytes)))
        framed.extend(name_bytes)
        framed.extend(struct.pack("<I", len(b"F32LE")))
        framed.extend(b"F32LE")
        framed.extend(struct.pack("<I", tensor.ndim))
        for dimension in tensor.shape:
            framed.extend(struct.pack("<Q", dimension))
        framed.extend(struct.pack("<Q", len(tensor_payload)))
        framed.extend(tensor_payload)
    return bytes(framed)


def _gradient_snapshot(model: CausalKWS) -> tuple[bytes | None, ...]:
    result: list[bytes | None] = []
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            result.append(None)
            continue
        _require_float32_tensor(
            gradient,
            name=f"gradient {name}",
            shape=tuple(parameter.shape),
        )
        result.append(
            gradient.detach().numpy().astype("<f4", copy=False).tobytes(order="C")
        )
    if len(result) != _MODEL_TENSOR_COUNT:
        raise Experiment002EvaluatorError("gradient snapshot has an invalid count")
    return tuple(result)


def _torch_rng_state() -> bytes:
    state = torch.get_rng_state()
    if (
        type(state) is not Tensor
        or state.device.type != "cpu"
        or state.dtype != torch.uint8
        or state.ndim != 1
        or not state.is_contiguous()
    ):
        raise Experiment002EvaluatorError("Torch CPU RNG state is invalid")
    return state.numpy().tobytes(order="C")


def _torch_rng_sha256(state: bytes) -> str:
    if type(state) is not bytes or not state:
        raise Experiment002EvaluatorError("Torch CPU RNG bytes are invalid")
    payload = _TORCH_RNG_DOMAIN + struct.pack("<Q", len(state)) + state
    return hashlib.sha256(payload).hexdigest()


def _require_uniform_model_mode(model: CausalKWS, *, training: bool) -> None:
    if type(training) is not bool:
        raise TypeError("training must be a boolean")
    if any(module.training is not training for module in model.modules()):
        expected = "train" if training else "eval"
        raise Experiment002EvaluatorError(
            f"model modules are not uniformly in {expected} mode"
        )


def _require_positive_int(value: object, name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise Experiment002EvaluatorError(f"{name} must be positive")
