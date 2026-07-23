"""Synthetic-only exact optimizer kernel for Experiment 002.

This module exposes no registered executor capability.  The source-bound
registered executor is a separate authority that reuses only private numerical
validation helpers from this module.  Its local route accepts no external
training or validation data and can issue only an unregistered synthetic
optimizer transition.
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import struct
import threading
import weakref
from dataclasses import dataclass, field, replace
from typing import Final, Literal, NoReturn

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.nn import functional as F

from falsewake.causal_kws import (
    BLOCK_DILATIONS,
    DEPTHWISE_KERNEL_SIZE,
    ENCODER_CHANNELS,
    INPUT_MEL_BINS,
    MODEL_RECEPTIVE_FIELD_FRAMES,
    OUTPUT_CLASSES,
    POOL_HISTORY_FRAMES,
    TRAINABLE_PARAMETER_COUNT,
    CausalKWS,
    CausalKWSState,
)
from falsewake.experiment_002_training_bridge import (
    RegisteredOptimizerTransition,
    _fail_optimizer_transition,
    _issue_synthetic_optimizer_transition,
    _transition_snapshot,
)

_MODEL_TENSOR_COUNT: Final = 53
_MODEL_PARAMETER_VALUE_COUNT: Final = 23_724
_SYNTHETIC_BATCH_SIZE: Final = 1
_MAX_GRADIENT_NORM: Final = 5.0
_LABEL_SMOOTHING: Final = 0.05
_ADAMW_CONSTRUCTOR_LR: Final = 0.003
_ADAMW_BETAS: Final = (0.9, 0.999)
_ADAMW_EPS: Final = 1e-8
_ADAMW_WEIGHT_DECAY: Final = 1e-4
_MODEL_TENSOR_DOMAIN: Final = b"falsewake-exp002-model-tensors-v1\0"
_OPTIMIZER_DOMAIN: Final = b"falsewake-exp002-optimizer-state-v1\0"
_RNG_DOMAIN: Final = b"falsewake-exp002-torch-cpu-rng-v1\0"
_SYNTHETIC_ROUTE_MARKER: Final = object()
_UINT32_MAX: Final = (1 << 32) - 1
_RUNTIME_LOCK = threading.Lock()
_RUNTIME_CONFIGURED = False

type _Float32Array = NDArray[np.float32]
type _Int64Array = NDArray[np.int64]

type _ExecutorPhase = Literal["READY", "COMPLETE", "FAILED"]


class Experiment002TrainingExecutorError(ValueError):
    """The isolated optimizer probe violated its frozen execution contract."""


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
class _SyntheticTrainingExecutor:
    """Opaque one-shot authority for the synthetic zero-input probe."""

    def __init__(self) -> None:
        raise TypeError("synthetic training executors are issuer-only")

    def __copy__(self) -> NoReturn:
        raise TypeError("synthetic training executors cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        del memo
        raise TypeError("synthetic training executors cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("synthetic training executors cannot be serialized")


@dataclass(frozen=True, slots=True)
class _RuntimeDigests:
    model_sha256: str
    optimizer_sha256: str
    rng_sha256: str
    rng_state: bytes


@dataclass(frozen=True, slots=True)
class _SyntheticExecutorSnapshot:
    seed: int
    phase: _ExecutorPhase
    optimizer_generation: int
    model_sha256: str
    optimizer_sha256: str
    rng_sha256: str


@dataclass(frozen=True, slots=True)
class _ExecutorGuard:
    token: object
    route_marker: object
    seed: int
    model: CausalKWS
    optimizer: torch.optim.AdamW
    parameter_names: tuple[str, ...]
    parameters: tuple[torch.nn.Parameter, ...]
    lock: threading.RLock
    active_transition: RegisteredOptimizerTransition | None
    active_transition_token: object | None


@dataclass(frozen=True, slots=True)
class _ExecutorLifecycle:
    phase: _ExecutorPhase
    optimizer_generation: int
    model_sha256: str
    optimizer_sha256: str
    rng_sha256: str
    active_transition: RegisteredOptimizerTransition | None
    active_transition_token: object | None


@dataclass(slots=True)
class _ExecutorState:
    token: object
    route_marker: object
    seed: int
    model: CausalKWS
    optimizer: torch.optim.AdamW
    parameter_names: tuple[str, ...]
    parameters: tuple[torch.nn.Parameter, ...]
    expected: _RuntimeDigests
    phase: _ExecutorPhase = "READY"
    optimizer_generation: int = 0
    active_transition: RegisteredOptimizerTransition | None = None
    active_transition_token: object | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)


_EXECUTORS: weakref.WeakKeyDictionary[_SyntheticTrainingExecutor, _ExecutorState] = (
    weakref.WeakKeyDictionary()
)
_EXECUTOR_GUARDS: weakref.WeakKeyDictionary[
    _SyntheticTrainingExecutor, _ExecutorGuard
] = weakref.WeakKeyDictionary()
_EXECUTOR_LIFECYCLES: weakref.WeakKeyDictionary[
    _SyntheticTrainingExecutor, _ExecutorLifecycle
] = weakref.WeakKeyDictionary()
_EXECUTORS_LOCK = threading.Lock()


def _create_synthetic_training_executor(*, seed: int) -> _SyntheticTrainingExecutor:
    """Create one unregistered, hard-coded zero-input optimizer probe."""

    _require_uint32(seed, "seed")
    _configure_and_verify_runtime()
    torch.manual_seed(seed)
    model = CausalKWS()
    rng_after_constructor = _rng_state_bytes()

    named_parameters = tuple(model.named_parameters())
    parameter_names = tuple(name for name, _ in named_parameters)
    parameters = tuple(parameter for _, parameter in named_parameters)
    _validate_parameter_capture(model, parameter_names, parameters)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=_ADAMW_CONSTRUCTOR_LR,
        betas=_ADAMW_BETAS,
        eps=_ADAMW_EPS,
        weight_decay=_ADAMW_WEIGHT_DECAY,
        amsgrad=False,
        maximize=False,
        foreach=False,
        capturable=False,
        differentiable=False,
        fused=False,
    )
    token = object()
    state = _ExecutorState(
        token=token,
        route_marker=_SYNTHETIC_ROUTE_MARKER,
        seed=seed,
        model=model,
        optimizer=optimizer,
        parameter_names=parameter_names,
        parameters=parameters,
        active_transition=None,
        active_transition_token=None,
        expected=_RuntimeDigests("", "", "", b""),
    )
    expected = _stable_runtime_digests(state, expected_generation=0)
    if expected.rng_state != rng_after_constructor:
        raise Experiment002TrainingExecutorError(
            "optimizer construction or executor checks consumed Torch RNG"
        )
    state.expected = expected
    result = object.__new__(_SyntheticTrainingExecutor)
    guard = _ExecutorGuard(
        token=token,
        route_marker=_SYNTHETIC_ROUTE_MARKER,
        seed=seed,
        model=model,
        optimizer=optimizer,
        parameter_names=parameter_names,
        parameters=parameters,
        lock=state.lock,
        active_transition=None,
        active_transition_token=None,
    )
    lifecycle = _lifecycle_from_state(state)
    with _EXECUTORS_LOCK:
        _EXECUTORS[result] = state
        _EXECUTOR_GUARDS[result] = guard
        _EXECUTOR_LIFECYCLES[result] = lifecycle
    _synthetic_executor_snapshot(result)
    return result


def _execute_synthetic_optimizer_update(
    executor: _SyntheticTrainingExecutor,
) -> RegisteredOptimizerTransition:
    """Run exactly one discarded zero-input update and issue synthetic evidence."""

    state = _issued_executor_state(executor)
    with _trusted_executor_lock(executor, state):
        try:
            _verify_runtime()
        except BaseException:
            _terminal_fail_executor(executor, state)
            raise
        try:
            _validate_executor_authority(executor, state)
        except BaseException:
            _terminal_fail_executor(executor, state)
            raise
        if state.phase == "COMPLETE":
            raise Experiment002TrainingExecutorError(
                "synthetic optimizer probe is one-shot"
            )
        if state.phase == "FAILED":
            raise Experiment002TrainingExecutorError(
                "synthetic optimizer probe is terminally failed"
            )
        transition: RegisteredOptimizerTransition | None = None
        try:
            if state.phase != "READY" or state.optimizer_generation != 0:
                raise Experiment002TrainingExecutorError("executor phase is invalid")
            entry = _stable_runtime_digests(state, expected_generation=0)
            if entry != state.expected:
                raise Experiment002TrainingExecutorError(
                    "model, optimizer, or RNG continuity changed before update"
                )

            state.optimizer.zero_grad(set_to_none=True)
            if any(parameter.grad is not None for parameter in state.parameters):
                raise Experiment002TrainingExecutorError(
                    "zero_grad did not clear every parameter gradient"
                )
            before = _stable_runtime_digests(state, expected_generation=0)
            if before != entry:
                raise Experiment002TrainingExecutorError(
                    "zero_grad changed model, optimizer, or RNG state"
                )

            model_inputs, label_indices, features, labels = _synthetic_zero_batch()
            _validate_synthetic_batch(model_inputs, label_indices, features, labels)
            state.model.train()
            initial_state = state.model.initial_state(
                _SYNTHETIC_BATCH_SIZE,
                device=features.device,
                dtype=features.dtype,
            )
            _validate_stream_state(initial_state, expected_frames=0, require_zero=True)
            rng_before_forward = _rng_state_bytes()
            if rng_before_forward != before.rng_state:
                raise Experiment002TrainingExecutorError(
                    "batch collation or fresh state creation consumed Torch RNG"
                )
            logits, next_state = state.model.forward_stream(features, initial_state)
            rng_after_forward = _rng_state_bytes()
            if rng_after_forward == rng_before_forward:
                raise Experiment002TrainingExecutorError(
                    "training dropout did not consume the Torch RNG stream"
                )
            _validate_logits(logits)
            _validate_stream_state(
                next_state,
                expected_frames=MODEL_RECEPTIVE_FIELD_FRAMES,
                require_zero=False,
            )
            loss = _training_cross_entropy(
                logits[:, MODEL_RECEPTIVE_FIELD_FRAMES - 1, :], labels
            )
            _validate_loss(loss)
            loss.backward()  # type: ignore[no-untyped-call]
            _validate_gradients(state.parameters)
            returned_norm = torch.nn.utils.clip_grad_norm_(
                state.parameters,
                _MAX_GRADIENT_NORM,
                norm_type=2.0,
                error_if_nonfinite=True,
                foreach=False,
            )
            _validate_preclip_norm(returned_norm)
            _validate_gradients(state.parameters)
            learning_rate = _registered_learning_rate(0)
            state.optimizer.param_groups[0]["lr"] = learning_rate
            state.optimizer.step()

            after = _stable_runtime_digests(state, expected_generation=1)
            if before.model_sha256 == after.model_sha256:
                raise Experiment002TrainingExecutorError(
                    "synthetic optimizer step did not change model parameters"
                )
            if before.optimizer_sha256 == after.optimizer_sha256:
                raise Experiment002TrainingExecutorError(
                    "synthetic optimizer step did not change optimizer state"
                )
            if after.rng_state != rng_after_forward:
                raise Experiment002TrainingExecutorError(
                    "Torch RNG changed outside the model training forward"
                )
            loss_value = np.float32(loss.detach().item())
            norm_value = np.float32(returned_norm.detach().item())
            transition = _issue_synthetic_optimizer_transition(
                seed=state.seed,
                zero_based_epoch=0,
                zero_based_global_update=0,
                batch_size=_SYNTHETIC_BATCH_SIZE,
                learning_rate=learning_rate,
                batch_mean_training_loss=loss_value,
                returned_preclip_l2_norm=norm_value,
                optimizer_generation_before=0,
                optimizer_generation_after=1,
                model_sha256_before=before.model_sha256,
                model_sha256_after=after.model_sha256,
                optimizer_sha256_before=before.optimizer_sha256,
                optimizer_sha256_after=after.optimizer_sha256,
                rng_sha256_before=before.rng_sha256,
                rng_sha256_after=after.rng_sha256,
            )
            transition_token = _transition_snapshot(
                transition, required_phase="ISSUED"
            ).transition_token
            state.active_transition = transition
            state.active_transition_token = transition_token
            _bind_transition_guard(executor, state)
            post_issue = _stable_runtime_digests(state, expected_generation=1)
            if post_issue != after:
                raise Experiment002TrainingExecutorError(
                    "model, optimizer, or RNG changed during transition issuance"
                )
            state.expected = after
            state.optimizer_generation = 1
            state.phase = "COMPLETE"
            _publish_lifecycle(executor, state)
            return transition
        except BaseException:
            if transition is not None:
                state.active_transition = transition
                with contextlib.suppress(BaseException):
                    _fail_optimizer_transition(transition)
            _terminal_fail_executor(executor, state)
            raise


def _synthetic_executor_snapshot(
    executor: _SyntheticTrainingExecutor,
) -> _SyntheticExecutorSnapshot:
    """Return immutable scalar/digest state without exposing model or optimizer."""

    state = _issued_executor_state(executor)
    with _trusted_executor_lock(executor, state):
        try:
            _validate_executor_authority(executor, state)
            if state.phase == "FAILED":
                raise Experiment002TrainingExecutorError(
                    "synthetic optimizer probe is terminally failed"
                )
            observed = _stable_runtime_digests(
                state, expected_generation=state.optimizer_generation
            )
            if observed != state.expected:
                raise Experiment002TrainingExecutorError(
                    "cached executor snapshot differs from live runtime"
                )
        except BaseException:
            _terminal_fail_executor(executor, state)
            raise
        return _SyntheticExecutorSnapshot(
            seed=state.seed,
            phase=state.phase,
            optimizer_generation=state.optimizer_generation,
            model_sha256=state.expected.model_sha256,
            optimizer_sha256=state.expected.optimizer_sha256,
            rng_sha256=state.expected.rng_sha256,
        )


def _issued_executor_state(executor: _SyntheticTrainingExecutor) -> _ExecutorState:
    if type(executor) is not _SyntheticTrainingExecutor:
        raise TypeError("executor must be an exact _SyntheticTrainingExecutor")
    with _EXECUTORS_LOCK:
        state = _EXECUTORS.get(executor)
    if type(state) is not _ExecutorState:
        raise Experiment002TrainingExecutorError(
            "synthetic executor was not issued by this module"
        )
    return state


def _trusted_executor_lock(
    executor: _SyntheticTrainingExecutor, state: _ExecutorState
) -> threading.RLock:
    with _EXECUTORS_LOCK:
        guard = _EXECUTOR_GUARDS.get(executor)
        primary = _EXECUTORS.get(executor)
    if (
        type(guard) is not _ExecutorGuard
        or primary is not state
        or type(guard.lock) is not type(threading.RLock())
    ):
        raise Experiment002TrainingExecutorError(
            "synthetic executor has no trusted lock authority"
        )
    return guard.lock


def _validate_executor_authority(
    executor: _SyntheticTrainingExecutor, state: _ExecutorState
) -> None:
    with _EXECUTORS_LOCK:
        guard = _EXECUTOR_GUARDS.get(executor)
        lifecycle = _EXECUTOR_LIFECYCLES.get(executor)
    valid = (
        type(guard) is _ExecutorGuard
        and guard.token is state.token
        and guard.route_marker is _SYNTHETIC_ROUTE_MARKER
        and state.route_marker is _SYNTHETIC_ROUTE_MARKER
        and guard.seed == state.seed
        and guard.model is state.model
        and guard.optimizer is state.optimizer
        and guard.parameter_names == state.parameter_names
        and len(guard.parameters) == len(state.parameters)
        and all(
            guarded is observed
            for guarded, observed in zip(
                guard.parameters, state.parameters, strict=True
            )
        )
        and guard.lock is state.lock
        and guard.active_transition is state.active_transition
        and guard.active_transition_token is state.active_transition_token
        and type(lifecycle) is _ExecutorLifecycle
        and lifecycle == _lifecycle_from_state(state)
        and state.phase in {"READY", "COMPLETE", "FAILED"}
        and type(state.lock) is type(threading.RLock())
    )
    if not valid:
        _terminal_fail_executor(executor, state)
        raise Experiment002TrainingExecutorError(
            "synthetic executor differs from issuer authority"
        )
    if state.phase == "READY":
        if (
            state.active_transition is not None
            or state.active_transition_token is not None
        ):
            _terminal_fail_executor(executor, state)
            raise Experiment002TrainingExecutorError(
                "ready synthetic executor has an optimizer transition"
            )
    elif state.phase == "COMPLETE":
        transition = state.active_transition
        if type(transition) is not RegisteredOptimizerTransition:
            _terminal_fail_executor(executor, state)
            raise Experiment002TrainingExecutorError(
                "complete synthetic executor lost its transition"
            )
        transition_snapshot = _transition_snapshot(transition, required_phase="ISSUED")
        if transition_snapshot.transition_token is not state.active_transition_token:
            _terminal_fail_executor(executor, state)
            raise Experiment002TrainingExecutorError(
                "complete synthetic executor transition binding changed"
            )


def _terminal_fail_executor(
    executor: _SyntheticTrainingExecutor, state: _ExecutorState
) -> None:
    state.phase = "FAILED"
    with _EXECUTORS_LOCK:
        guard = _EXECUTOR_GUARDS.get(executor)
        lifecycle = _EXECUTOR_LIFECYCLES.get(executor)
    transition = _trusted_active_transition(state, guard, lifecycle)
    if transition is None:
        state.active_transition = None
        state.active_transition_token = None
    else:
        with contextlib.suppress(BaseException):
            _fail_optimizer_transition(transition)
    with contextlib.suppress(BaseException):
        _publish_lifecycle(executor, state)


def _trusted_active_transition(
    state: _ExecutorState,
    guard: _ExecutorGuard | None,
    lifecycle: _ExecutorLifecycle | None,
) -> RegisteredOptimizerTransition | None:
    candidates = [
        candidate
        for candidate in (
            state.active_transition,
            guard.active_transition if type(guard) is _ExecutorGuard else None,
            (
                lifecycle.active_transition
                if type(lifecycle) is _ExecutorLifecycle
                else None
            ),
        )
        if type(candidate) is RegisteredOptimizerTransition
    ]
    unique: list[RegisteredOptimizerTransition] = []
    for candidate in candidates:
        if not any(candidate is observed for observed in unique):
            unique.append(candidate)
    if not unique:
        return None
    support = tuple(
        sum(candidate is observed for observed in candidates) for candidate in unique
    )
    highest = max(support)
    if highest < 2 or support.count(highest) != 1:
        return None
    return unique[support.index(highest)]


def _bind_transition_guard(
    executor: _SyntheticTrainingExecutor, state: _ExecutorState
) -> None:
    with _EXECUTORS_LOCK:
        guard = _EXECUTOR_GUARDS.get(executor)
        if (
            type(guard) is not _ExecutorGuard
            or _EXECUTORS.get(executor) is not state
            or guard.active_transition is not None
            or guard.active_transition_token is not None
            or type(state.active_transition) is not RegisteredOptimizerTransition
            or type(state.active_transition_token) is not object
        ):
            raise Experiment002TrainingExecutorError(
                "synthetic transition guard cannot be bound"
            )
        _EXECUTOR_GUARDS[executor] = replace(
            guard,
            active_transition=state.active_transition,
            active_transition_token=state.active_transition_token,
        )


def _publish_lifecycle(
    executor: _SyntheticTrainingExecutor, state: _ExecutorState
) -> None:
    lifecycle = _lifecycle_from_state(state)
    with _EXECUTORS_LOCK:
        if _EXECUTORS.get(executor) is not state:
            raise Experiment002TrainingExecutorError(
                "synthetic executor primary authority changed"
            )
        guard = _EXECUTOR_GUARDS.get(executor)
        if type(guard) is not _ExecutorGuard:
            raise Experiment002TrainingExecutorError(
                "synthetic executor guard disappeared"
            )
        if guard.active_transition is None and state.active_transition is not None:
            _EXECUTOR_GUARDS[executor] = replace(
                guard,
                active_transition=state.active_transition,
                active_transition_token=state.active_transition_token,
            )
        elif (
            guard.active_transition is not state.active_transition
            or guard.active_transition_token is not state.active_transition_token
        ):
            raise Experiment002TrainingExecutorError(
                "synthetic transition guard differs from executor state"
            )
        _EXECUTOR_LIFECYCLES[executor] = lifecycle


def _lifecycle_from_state(state: _ExecutorState) -> _ExecutorLifecycle:
    return _ExecutorLifecycle(
        phase=state.phase,
        optimizer_generation=state.optimizer_generation,
        model_sha256=state.expected.model_sha256,
        optimizer_sha256=state.expected.optimizer_sha256,
        rng_sha256=state.expected.rng_sha256,
        active_transition=state.active_transition,
        active_transition_token=state.active_transition_token,
    )


def _stable_runtime_digests(
    state: _ExecutorState, *, expected_generation: int
) -> _RuntimeDigests:
    first = _runtime_digests(state, expected_generation=expected_generation)
    second = _runtime_digests(state, expected_generation=expected_generation)
    if first != second:
        raise Experiment002TrainingExecutorError(
            "model, optimizer, or RNG changed during double snapshot"
        )
    return first


def _runtime_digests(
    state: _ExecutorState, *, expected_generation: int
) -> _RuntimeDigests:
    _validate_parameter_capture(state.model, state.parameter_names, state.parameters)
    _validate_optimizer(state, expected_generation=expected_generation)
    rng_state = _rng_state_bytes()
    return _RuntimeDigests(
        model_sha256=_model_sha256(
            state.model, state.parameter_names, state.parameters
        ),
        optimizer_sha256=_optimizer_sha256(
            state, expected_generation=expected_generation
        ),
        rng_sha256=hashlib.sha256(
            _RNG_DOMAIN + struct.pack("<Q", len(rng_state)) + rng_state
        ).hexdigest(),
        rng_state=rng_state,
    )


def _validate_parameter_capture(
    model: CausalKWS,
    names: tuple[str, ...],
    parameters: tuple[torch.nn.Parameter, ...],
) -> None:
    if type(model) is not CausalKWS:
        raise TypeError("model must be an exact CausalKWS")
    current = tuple(model.named_parameters())
    if (
        model.training is not True
        or type(model.dropout) is not torch.nn.Dropout
        or type(model.dropout.p) is not float
        or struct.pack("<d", model.dropout.p) != struct.pack("<d", 0.10)
        or model.dropout.inplace is not False
        or len(current) != _MODEL_TENSOR_COUNT
        or len(names) != _MODEL_TENSOR_COUNT
        or len(parameters) != _MODEL_TENSOR_COUNT
        or tuple(name for name, _ in current) != names
        or len(set(names)) != len(names)
        or len({id(parameter) for parameter in parameters}) != len(parameters)
        or any(
            current_parameter is not captured_parameter
            for (_, current_parameter), captured_parameter in zip(
                current, parameters, strict=True
            )
        )
        or sum(parameter.numel() for parameter in parameters)
        != _MODEL_PARAMETER_VALUE_COUNT
        or TRAINABLE_PARAMETER_COUNT != _MODEL_PARAMETER_VALUE_COUNT
        or tuple(model.state_dict()) != names
    ):
        raise Experiment002TrainingExecutorError(
            "CausalKWS parameter capture differs from registration"
        )
    for name, parameter in zip(names, parameters, strict=True):
        if (
            type(name) is not str
            or type(parameter) is not torch.nn.Parameter
            or parameter.device.type != "cpu"
            or parameter.dtype != torch.float32
            or parameter.layout != torch.strided
            or not parameter.is_contiguous()
            or not parameter.requires_grad
            or not bool(torch.isfinite(parameter).all().item())
        ):
            raise Experiment002TrainingExecutorError(
                f"registered parameter is invalid ({name})"
            )


def _validate_optimizer(state: _ExecutorState, *, expected_generation: int) -> None:
    optimizer = state.optimizer
    if type(optimizer) is not torch.optim.AdamW or len(optimizer.param_groups) != 1:
        raise Experiment002TrainingExecutorError("optimizer must be one exact AdamW")
    group = optimizer.param_groups[0]
    expected_keys = {
        "amsgrad",
        "betas",
        "capturable",
        "decoupled_weight_decay",
        "differentiable",
        "eps",
        "foreach",
        "fused",
        "lr",
        "maximize",
        "params",
        "weight_decay",
    }
    group_parameters = group.get("params")
    defaults = optimizer.defaults
    expected_default_keys = expected_keys - {"params"}
    if (
        type(group) is not dict
        or set(group) != expected_keys
        or type(defaults) is not dict
        or set(defaults) != expected_default_keys
        or type(group_parameters) is not list
        or len(group_parameters) != len(state.parameters)
        or any(
            observed is not expected
            for observed, expected in zip(
                group_parameters, state.parameters, strict=True
            )
        )
        or type(group.get("lr")) is not float
        or not _exact_betas(group.get("betas"))
        or not _exact_float(group.get("eps"), _ADAMW_EPS)
        or not _exact_float(group.get("weight_decay"), _ADAMW_WEIGHT_DECAY)
        or group.get("amsgrad") is not False
        or group.get("maximize") is not False
        or group.get("foreach") is not False
        or group.get("capturable") is not False
        or group.get("differentiable") is not False
        or group.get("fused") is not False
        or group.get("decoupled_weight_decay") is not True
        or not _exact_float(defaults.get("lr"), _ADAMW_CONSTRUCTOR_LR)
        or not _exact_betas(defaults.get("betas"))
        or not _exact_float(defaults.get("eps"), _ADAMW_EPS)
        or not _exact_float(defaults.get("weight_decay"), _ADAMW_WEIGHT_DECAY)
        or defaults.get("amsgrad") is not False
        or defaults.get("maximize") is not False
        or defaults.get("foreach") is not False
        or defaults.get("capturable") is not False
        or defaults.get("differentiable") is not False
        or defaults.get("fused") is not False
        or defaults.get("decoupled_weight_decay") is not True
    ):
        raise Experiment002TrainingExecutorError(
            "AdamW parameter group differs from registration"
        )
    expected_lr = (
        _ADAMW_CONSTRUCTOR_LR
        if expected_generation == 0
        else _registered_learning_rate(expected_generation - 1)
    )
    if struct.pack("<d", group["lr"]) != struct.pack("<d", expected_lr):
        raise Experiment002TrainingExecutorError(
            "AdamW learning rate differs from optimizer generation"
        )
    optimizer_state = optimizer.state
    if expected_generation == 0:
        if len(optimizer_state) != 0:
            raise Experiment002TrainingExecutorError(
                "AdamW state must be empty before update zero"
            )
        return
    if len(optimizer_state) != len(state.parameters) or any(
        parameter not in optimizer_state for parameter in state.parameters
    ):
        raise Experiment002TrainingExecutorError(
            "AdamW state does not cover every parameter exactly"
        )
    for parameter in state.parameters:
        item = optimizer_state[parameter]
        if type(item) is not dict or set(item) != {"step", "exp_avg", "exp_avg_sq"}:
            raise Experiment002TrainingExecutorError(
                "AdamW state keys differ from registration"
            )
        step = item["step"]
        exp_avg = item["exp_avg"]
        exp_avg_sq = item["exp_avg_sq"]
        _validate_state_tensor(step, shape=(), name="step")
        if float(step.item()) != float(expected_generation):
            raise Experiment002TrainingExecutorError(
                "AdamW step counter differs from optimizer generation"
            )
        _validate_state_tensor(exp_avg, shape=tuple(parameter.shape), name="exp_avg")
        _validate_state_tensor(
            exp_avg_sq, shape=tuple(parameter.shape), name="exp_avg_sq"
        )


def _exact_float(value: object, expected: float) -> bool:
    return type(value) is float and struct.pack("<d", value) == struct.pack(
        "<d", expected
    )


def _exact_betas(value: object) -> bool:
    return (
        type(value) is tuple
        and len(value) == 2
        and _exact_float(value[0], _ADAMW_BETAS[0])
        and _exact_float(value[1], _ADAMW_BETAS[1])
    )


def _validate_state_tensor(
    tensor: object, *, shape: tuple[int, ...], name: str
) -> None:
    if (
        type(tensor) is not Tensor
        or tuple(tensor.shape) != shape
        or tensor.device.type != "cpu"
        or tensor.dtype != torch.float32
        or tensor.layout != torch.strided
        or not tensor.is_contiguous()
        or not bool(torch.isfinite(tensor).all().item())
    ):
        raise Experiment002TrainingExecutorError(f"AdamW {name} tensor is invalid")


def _model_sha256(
    model: CausalKWS,
    names: tuple[str, ...],
    parameters: tuple[torch.nn.Parameter, ...],
) -> str:
    del names, parameters
    first_names, first_tensors = _snapshot_model_parameters(model)
    first_payload = _frame_model_tensors(first_names, first_tensors)
    second_names, second_tensors = _snapshot_model_parameters(model)
    second_payload = _frame_model_tensors(second_names, second_tensors)
    if first_names != second_names or first_payload != second_payload:
        raise Experiment002TrainingExecutorError(
            "model parameter bytes changed during double snapshot"
        )
    return hashlib.sha256(_MODEL_TENSOR_DOMAIN + first_payload).hexdigest()


def _snapshot_model_parameters(
    model: CausalKWS,
) -> tuple[tuple[str, ...], tuple[Tensor, ...]]:
    named_parameters = tuple(model.named_parameters())
    names = tuple(name for name, _ in named_parameters)
    state_dict = model.state_dict()
    if (
        len(named_parameters) != _MODEL_TENSOR_COUNT
        or len(set(names)) != len(names)
        or set(state_dict) != set(names)
        or len(state_dict) != len(names)
    ):
        raise Experiment002TrainingExecutorError(
            "model parameter names differ from canonical snapshot layout"
        )
    sorted_names = tuple(sorted(names, key=lambda name: name.encode("utf-8")))
    snapshots: list[Tensor] = []
    value_count = 0
    for name in sorted_names:
        tensor = state_dict[name]
        _require_snapshot_tensor(tensor, name, allow_requires_grad=False)
        value_count += tensor.numel()
        snapshot = tensor.detach().clone(memory_format=torch.contiguous_format)
        _require_snapshot_tensor(snapshot, name, allow_requires_grad=False)
        snapshots.append(snapshot)
    if value_count != _MODEL_PARAMETER_VALUE_COUNT:
        raise Experiment002TrainingExecutorError(
            "model parameter value count differs from canonical layout"
        )
    return sorted_names, tuple(snapshots)


def _frame_model_tensors(names: tuple[str, ...], tensors: tuple[Tensor, ...]) -> bytes:
    """Frame detached float32 tensors with the canonical model evidence format."""

    if (
        type(names) is not tuple
        or type(tensors) is not tuple
        or len(names) != len(tensors)
        or names != tuple(sorted(names, key=lambda name: name.encode("utf-8")))
        or len(set(names)) != len(names)
    ):
        raise Experiment002TrainingExecutorError(
            "model tensor frame names are not in canonical order"
        )
    payload = bytearray(struct.pack("<I", len(names)))
    for name, tensor in zip(names, tensors, strict=True):
        if type(name) is not str or not name:
            raise Experiment002TrainingExecutorError(
                "model tensor frame name is invalid"
            )
        _require_snapshot_tensor(tensor, name, allow_requires_grad=False)
        name_bytes = name.encode("utf-8")
        raw = tensor.detach().numpy().astype("<f4", copy=False).tobytes(order="C")
        payload.extend(struct.pack("<I", len(name_bytes)))
        payload.extend(name_bytes)
        payload.extend(struct.pack("<I", 5))
        payload.extend(b"F32LE")
        payload.extend(struct.pack("<I", tensor.ndim))
        for dimension in tensor.shape:
            payload.extend(struct.pack("<Q", dimension))
        payload.extend(struct.pack("<Q", len(raw)))
        payload.extend(raw)
    return bytes(payload)


def _require_snapshot_tensor(
    tensor: Tensor, name: str, *, allow_requires_grad: bool
) -> None:
    if (
        type(tensor) is not Tensor
        or tensor.device.type != "cpu"
        or tensor.dtype != torch.float32
        or tensor.layout != torch.strided
        or not tensor.is_contiguous()
        or (not allow_requires_grad and tensor.requires_grad)
        or not bool(torch.isfinite(tensor).all().item())
    ):
        raise Experiment002TrainingExecutorError(
            f"model tensor snapshot is invalid ({name})"
        )


def _optimizer_sha256(state: _ExecutorState, *, expected_generation: int) -> str:
    group = state.optimizer.param_groups[0]
    payload = bytearray(
        struct.pack(
            "<I4d7BI",
            1,
            group["lr"],
            group["betas"][0],
            group["betas"][1],
            group["eps"],
            int(group["amsgrad"]),
            int(group["maximize"]),
            int(group["foreach"]),
            int(group["capturable"]),
            int(group["differentiable"]),
            int(group["fused"]),
            int(group["decoupled_weight_decay"]),
            len(state.parameters),
        )
    )
    payload.extend(struct.pack("<dI", group["weight_decay"], expected_generation))
    for parameter_index, (name, parameter) in enumerate(
        zip(state.parameter_names, state.parameters, strict=True)
    ):
        payload.extend(struct.pack("<I", parameter_index))
        name_bytes = name.encode("utf-8")
        payload.extend(struct.pack("<I", len(name_bytes)))
        payload.extend(name_bytes)
        item = state.optimizer.state.get(parameter)
        if item is None:
            payload.extend(b"\0")
            continue
        payload.extend(b"\1")
        for key in ("step", "exp_avg", "exp_avg_sq"):
            _append_f32_tensor(payload, key, item[key])
    return hashlib.sha256(_OPTIMIZER_DOMAIN + bytes(payload)).hexdigest()


def _append_f32_tensor(payload: bytearray, name: str, tensor: Tensor) -> None:
    name_bytes = name.encode("utf-8")
    raw = tensor.detach().numpy().astype("<f4", copy=False).tobytes(order="C")
    payload.extend(struct.pack("<I", len(name_bytes)))
    payload.extend(name_bytes)
    payload.extend(struct.pack("<I", 5))
    payload.extend(b"F32LE")
    payload.extend(struct.pack("<I", tensor.ndim))
    for dimension in tensor.shape:
        payload.extend(struct.pack("<Q", dimension))
    payload.extend(struct.pack("<Q", len(raw)))
    payload.extend(raw)


def _rng_state_bytes() -> bytes:
    rng = torch.get_rng_state()
    if (
        type(rng) is not Tensor
        or rng.device.type != "cpu"
        or rng.dtype != torch.uint8
        or rng.ndim != 1
        or not rng.is_contiguous()
    ):
        raise Experiment002TrainingExecutorError("Torch CPU RNG state is invalid")
    return rng.numpy().tobytes(order="C")


def _synthetic_zero_batch() -> tuple[_Float32Array, _Int64Array, Tensor, Tensor]:
    model_inputs = np.zeros(
        (_SYNTHETIC_BATCH_SIZE, INPUT_MEL_BINS, MODEL_RECEPTIVE_FIELD_FRAMES),
        dtype=np.float32,
        order="C",
    )
    label_indices = np.zeros((_SYNTHETIC_BATCH_SIZE,), dtype=np.int64, order="C")
    features = torch.from_numpy(model_inputs)
    labels = torch.from_numpy(label_indices)
    return model_inputs, label_indices, features, labels


def _validate_synthetic_batch(
    model_inputs: _Float32Array,
    label_indices: _Int64Array,
    features: Tensor,
    labels: Tensor,
) -> None:
    if (
        type(model_inputs) is not np.ndarray
        or model_inputs.dtype != np.dtype(np.float32)
        or model_inputs.shape
        != (_SYNTHETIC_BATCH_SIZE, INPUT_MEL_BINS, MODEL_RECEPTIVE_FIELD_FRAMES)
        or not model_inputs.flags.c_contiguous
        or not model_inputs.flags.owndata
        or model_inputs.base is not None
        or not bool(np.isfinite(model_inputs).all())
        or bool(np.count_nonzero(model_inputs))
        or bool(np.signbit(model_inputs).any())
        or type(label_indices) is not np.ndarray
        or label_indices.dtype != np.dtype(np.int64)
        or label_indices.shape != (_SYNTHETIC_BATCH_SIZE,)
        or not label_indices.flags.c_contiguous
        or not label_indices.flags.owndata
        or label_indices.base is not None
        or bool(np.count_nonzero(label_indices))
        or type(features) is not Tensor
        or tuple(features.shape)
        != (_SYNTHETIC_BATCH_SIZE, INPUT_MEL_BINS, MODEL_RECEPTIVE_FIELD_FRAMES)
        or features.device.type != "cpu"
        or features.dtype != torch.float32
        or not features.is_contiguous()
        or features.data_ptr() != model_inputs.ctypes.data
        or not bool(torch.isfinite(features).all().item())
        or bool(torch.count_nonzero(features).item())
        or bool(torch.signbit(features).any().item())
        or type(labels) is not Tensor
        or tuple(labels.shape) != (_SYNTHETIC_BATCH_SIZE,)
        or labels.device.type != "cpu"
        or labels.dtype != torch.int64
        or not labels.is_contiguous()
        or labels.data_ptr() != label_indices.ctypes.data
        or bool(torch.count_nonzero(labels).item())
    ):
        raise Experiment002TrainingExecutorError("synthetic zero batch is invalid")


def _validate_logits(logits: Tensor) -> None:
    if (
        type(logits) is not Tensor
        or tuple(logits.shape)
        != (
            _SYNTHETIC_BATCH_SIZE,
            MODEL_RECEPTIVE_FIELD_FRAMES,
            OUTPUT_CLASSES,
        )
        or logits.device.type != "cpu"
        or logits.dtype != torch.float32
        or not bool(torch.isfinite(logits).all().item())
    ):
        raise Experiment002TrainingExecutorError("model logits are invalid")


def _validate_stream_state(
    state: CausalKWSState, *, expected_frames: int, require_zero: bool
) -> None:
    if type(state) is not CausalKWSState:
        raise Experiment002TrainingExecutorError("stream state type is invalid")
    if len(state.depthwise_states) != len(BLOCK_DILATIONS):
        raise Experiment002TrainingExecutorError("depthwise state count is invalid")
    for tensor, dilation in zip(state.depthwise_states, BLOCK_DILATIONS, strict=True):
        if (
            type(tensor) is not Tensor
            or tuple(tensor.shape)
            != (
                _SYNTHETIC_BATCH_SIZE,
                ENCODER_CHANNELS,
                (DEPTHWISE_KERNEL_SIZE - 1) * dilation,
            )
            or tensor.device.type != "cpu"
            or tensor.dtype != torch.float32
            or not tensor.is_contiguous()
            or not bool(torch.isfinite(tensor).all().item())
            or (require_zero and bool(torch.count_nonzero(tensor).item()))
            or (require_zero and bool(torch.signbit(tensor).any().item()))
        ):
            raise Experiment002TrainingExecutorError("float stream state is invalid")
    pool = state.pool_state
    if (
        type(pool) is not Tensor
        or tuple(pool.shape)
        != (_SYNTHETIC_BATCH_SIZE, ENCODER_CHANNELS, POOL_HISTORY_FRAMES)
        or pool.device.type != "cpu"
        or pool.dtype != torch.float32
        or not pool.is_contiguous()
        or not bool(torch.isfinite(pool).all().item())
        or (require_zero and bool(torch.count_nonzero(pool).item()))
        or (require_zero and bool(torch.signbit(pool).any().item()))
    ):
        raise Experiment002TrainingExecutorError("pool stream state is invalid")
    if (
        type(state.frames_seen) is not Tensor
        or tuple(state.frames_seen.shape) != (_SYNTHETIC_BATCH_SIZE,)
        or state.frames_seen.device.type != "cpu"
        or state.frames_seen.dtype != torch.int64
        or not state.frames_seen.is_contiguous()
        or not bool(torch.all(state.frames_seen == expected_frames).item())
    ):
        raise Experiment002TrainingExecutorError("frames_seen state is invalid")


def _validate_loss(loss: Tensor) -> None:
    if (
        type(loss) is not Tensor
        or loss.ndim != 0
        or loss.device.type != "cpu"
        or loss.dtype != torch.float32
        or not bool(torch.isfinite(loss).item())
    ):
        raise Experiment002TrainingExecutorError("training loss is invalid")


def _training_cross_entropy(supervised_logits: Tensor, labels: Tensor) -> Tensor:
    return F.cross_entropy(
        supervised_logits,
        labels,
        weight=None,
        reduction="mean",
        label_smoothing=_LABEL_SMOOTHING,
    )


def _validate_gradients(parameters: tuple[torch.nn.Parameter, ...]) -> None:
    for parameter in parameters:
        gradient = parameter.grad
        if (
            type(gradient) is not Tensor
            or gradient.device.type != "cpu"
            or gradient.dtype != torch.float32
            or gradient.layout != torch.strided
            or tuple(gradient.shape) != tuple(parameter.shape)
            or not bool(torch.isfinite(gradient).all().item())
        ):
            raise Experiment002TrainingExecutorError(
                "parameter gradient is missing, sparse, or nonfinite"
            )


def _validate_preclip_norm(returned_norm: Tensor) -> None:
    if (
        type(returned_norm) is not Tensor
        or returned_norm.ndim != 0
        or returned_norm.device.type != "cpu"
        or returned_norm.dtype != torch.float32
        or not bool(torch.isfinite(returned_norm).item())
        or float(returned_norm.item()) < 0.0
    ):
        raise Experiment002TrainingExecutorError(
            "returned preclip gradient norm is invalid"
        )


def _registered_learning_rate(global_update: int) -> float:
    _require_uint32(global_update, "global_update")
    total_updates = 9_390
    warmup_updates = 313
    if global_update >= total_updates:
        raise Experiment002TrainingExecutorError(
            "global_update is outside the frozen schedule"
        )
    if global_update < warmup_updates:
        return 0.0003 + (0.003 - 0.0003) * global_update / (warmup_updates - 1)
    return 0.00003 + 0.5 * (0.003 - 0.00003) * (
        1.0
        + math.cos(
            math.pi
            * (global_update - warmup_updates)
            / (total_updates - 1 - warmup_updates)
        )
    )


def _require_uint32(value: object, name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if not 0 <= value <= _UINT32_MAX:
        raise Experiment002TrainingExecutorError(f"{name} must fit UINT32LE")


def _configure_and_verify_runtime() -> None:
    """Establish the registered CPU numerical runtime before model seeding."""

    global _RUNTIME_CONFIGURED
    with _RUNTIME_LOCK:
        if not _RUNTIME_CONFIGURED:
            if torch.cuda.is_available():
                raise Experiment002TrainingExecutorError(
                    "registered runtime requires CUDA to be unavailable"
                )
            torch.set_default_device("cpu")
            torch.set_default_dtype(torch.float32)
            torch.use_deterministic_algorithms(True, warn_only=False)
            torch.set_float32_matmul_precision("highest")
            if torch.set_flush_denormal(False) is not True:
                raise Experiment002TrainingExecutorError(
                    "Torch could not disable flush-denormal mode"
                )
            torch.set_num_threads(2)
            if torch.get_num_interop_threads() != 1:
                torch.set_num_interop_threads(1)
            torch.backends.mkldnn.set_flags(False)  # type: ignore[no-untyped-call]
            torch.backends.nnpack.set_flags(False)  # type: ignore[no-untyped-call]
            _RUNTIME_CONFIGURED = True
        _verify_runtime()


def _verify_runtime() -> None:
    subnormal_bits = np.array([1], dtype=np.uint32)
    positive_subnormal = torch.from_numpy(subnormal_bits.view(np.float32))
    if (
        torch.cuda.is_available()
        or torch.get_default_device() != torch.device("cpu")
        or torch.get_default_dtype() != torch.float32
        or not torch.are_deterministic_algorithms_enabled()
        or torch.is_deterministic_algorithms_warn_only_enabled()
        or torch.get_float32_matmul_precision() != "highest"
        or float(positive_subnormal.item()) == 0.0
        or torch.get_num_threads() != 2
        or torch.get_num_interop_threads() != 1
        or torch.backends.mkldnn.enabled
        or torch._C._get_nnpack_enabled()
    ):
        raise Experiment002TrainingExecutorError(
            "Torch runtime differs from registered CPU setup"
        )
