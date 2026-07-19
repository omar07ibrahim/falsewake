from __future__ import annotations

import copy
import hashlib
import inspect
import pickle
import struct
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor

import falsewake.experiment_002_evidence as validation_evidence
import falsewake.experiment_002_metrics as training_metrics
import falsewake.experiment_002_training_bridge as bridge
import falsewake.experiment_002_training_evidence as training_evidence
import falsewake.experiment_002_training_executor as executor
import falsewake.experiment_002_training_population as training_population
import falsewake.experiment_002_validation as training_validation
from falsewake.causal_kws import CausalKWS, CausalKWSState


def _forbidden(counter: dict[str, int], name: str) -> Any:
    def reject(*args: object, **kwargs: object) -> None:
        del args, kwargs
        counter[name] += 1
        raise AssertionError(f"registered route reached: {name}")

    return reject


def test_registered_executor_surface_is_intentionally_absent() -> None:
    source = Path(executor.__file__).read_text(encoding="utf-8")
    assert not hasattr(executor, "create_registered_training_executor")
    assert "_issue_registered_executor_session" not in source
    assert "_begin_registered_epoch_for_executor" not in source
    assert "_executor_next_batch" not in source
    assert "_consume_registered_update" not in source
    assert "experiment_002_training_population" not in source
    assert "experiment_002_training_evidence" not in source
    parameters = inspect.signature(
        executor._execute_synthetic_optimizer_update
    ).parameters
    assert tuple(parameters) == ("executor",)


def test_synthetic_executor_constructor_and_authority_are_opaque() -> None:
    with pytest.raises(TypeError, match="issuer-only"):
        executor._SyntheticTrainingExecutor()
    counterfeit = object.__new__(executor._SyntheticTrainingExecutor)
    with pytest.raises(executor.Experiment002TrainingExecutorError, match="not issued"):
        executor._synthetic_executor_snapshot(counterfeit)

    issued = executor._create_synthetic_training_executor(seed=17)
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.copy(issued)
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.deepcopy(issued)
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(issued)
    snapshot = executor._synthetic_executor_snapshot(issued)
    assert snapshot.phase == "READY"
    assert not hasattr(snapshot, "model")
    assert not hasattr(snapshot, "optimizer")
    assert not hasattr(issued, "model")
    assert not hasattr(issued, "optimizer")


def test_factory_calls_manual_seed_once_immediately_before_exact_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, int | None]] = []
    original_manual_seed = torch.manual_seed
    original_init = CausalKWS.__init__

    def manual_seed(seed: int) -> torch.Generator:
        events.append(("manual_seed", seed))
        return original_manual_seed(seed)

    def model_init(model: CausalKWS) -> None:
        events.append(("CausalKWS", None))
        original_init(model)

    monkeypatch.setattr(torch, "manual_seed", manual_seed)
    monkeypatch.setattr(CausalKWS, "__init__", model_init)
    issued = executor._create_synthetic_training_executor(seed=19)
    assert type(executor._issued_executor_state(issued).model) is CausalKWS
    assert events == [("manual_seed", 19), ("CausalKWS", None)]
    assert torch.get_default_device() == torch.device("cpu")
    assert torch.get_default_dtype() == torch.float32
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    assert torch.get_float32_matmul_precision() == "highest"
    assert torch.get_num_threads() == 2
    assert torch.get_num_interop_threads() == 1
    assert not torch.backends.mkldnn.enabled
    assert not torch._C._get_nnpack_enabled()


def test_model_frame_matches_evidence_oracle_and_tiny_golden_bytes() -> None:
    torch.manual_seed(23)
    model = CausalKWS()
    named = tuple(model.named_parameters())
    names = tuple(name for name, _ in named)
    parameters = tuple(parameter for _, parameter in named)
    observed = executor._model_sha256(model, names, parameters)
    oracle = training_evidence._capture_model_tensors(
        model,
        layout=training_evidence._REGISTERED_MODEL_LAYOUT,
    )
    assert observed == oracle.sha256

    golden_names = ("a_tensor", "é_tensor")
    golden_tensors = (
        torch.tensor([-0.0, 2.5], dtype=torch.float32),
        torch.tensor([[-1.25]], dtype=torch.float32),
    )
    expected = bytearray(struct.pack("<I", 2))
    for name, shape, raw in (
        ("a_tensor", (2,), struct.pack("<ff", -0.0, 2.5)),
        ("é_tensor", (1, 1), struct.pack("<f", -1.25)),
    ):
        encoded = name.encode("utf-8")
        expected.extend(struct.pack("<I", len(encoded)))
        expected.extend(encoded)
        expected.extend(struct.pack("<I", 5))
        expected.extend(b"F32LE")
        expected.extend(struct.pack("<I", len(shape)))
        for dimension in shape:
            expected.extend(struct.pack("<Q", dimension))
        expected.extend(struct.pack("<Q", len(raw)))
        expected.extend(raw)
    framed = executor._frame_model_tensors(golden_names, golden_tensors)
    assert framed == bytes(expected)
    assert hashlib.sha256(executor._MODEL_TENSOR_DOMAIN + framed).hexdigest() == (
        "4ab7c4dd396d5ac19a00615830c57ea6bcf2225f6a66588b9b3aa436585d28e6"
    )


def test_actual_synthetic_update_is_exact_and_never_reaches_registered_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered_calls = {
        "session": 0,
        "epoch": 0,
        "batch": 0,
        "trace": 0,
        "validation": 0,
        "metrics": 0,
        "evidence": 0,
    }
    monkeypatch.setattr(
        bridge,
        "_issue_registered_executor_session",
        _forbidden(registered_calls, "session"),
    )
    monkeypatch.setattr(
        training_population,
        "_begin_registered_epoch_for_executor",
        _forbidden(registered_calls, "epoch"),
    )
    monkeypatch.setattr(
        training_population,
        "_executor_next_batch",
        _forbidden(registered_calls, "batch"),
    )
    monkeypatch.setattr(
        training_evidence.UpdateTraceAccumulator,
        "_consume_registered_update",
        _forbidden(registered_calls, "trace"),
    )
    monkeypatch.setattr(
        training_validation,
        "materialize_registered_validation",
        _forbidden(registered_calls, "validation"),
    )
    monkeypatch.setattr(
        training_metrics,
        "evaluate_validation_layout_logits",
        _forbidden(registered_calls, "metrics"),
    )
    monkeypatch.setattr(
        validation_evidence,
        "issue_registered_validation_evidence",
        _forbidden(registered_calls, "evidence"),
    )

    issued = executor._create_synthetic_training_executor(seed=29)
    state = executor._issued_executor_state(issued)
    before = executor._synthetic_executor_snapshot(issued)
    assert len(state.parameter_names) == len(state.parameters) == 53
    assert sum(parameter.numel() for parameter in state.parameters) == 23_724
    assert len(state.optimizer.param_groups) == 1
    group = state.optimizer.param_groups[0]
    assert set(group) == {
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
    assert group["lr"] == 0.003
    assert group["betas"] == (0.9, 0.999)
    assert group["eps"] == 1e-8
    assert group["weight_decay"] == 1e-4
    assert group["decoupled_weight_decay"] is True
    assert all(
        group[name] is False
        for name in (
            "amsgrad",
            "capturable",
            "differentiable",
            "foreach",
            "fused",
            "maximize",
        )
    )
    assert all(
        observed is expected
        for observed, expected in zip(group["params"], state.parameters, strict=True)
    )
    assert not state.optimizer.state

    events: list[str] = []
    original_stable = executor._stable_runtime_digests
    original_zero_grad = state.optimizer.zero_grad
    original_batch = executor._synthetic_zero_batch
    original_batch_check = executor._validate_synthetic_batch
    original_train = state.model.train
    original_initial_state = state.model.initial_state
    original_state_check = executor._validate_stream_state
    original_forward = state.model.forward_stream
    original_logits_check = executor._validate_logits
    original_cross_entropy = executor._training_cross_entropy
    original_loss_check = executor._validate_loss
    original_gradient_check = executor._validate_gradients
    original_clip = torch.nn.utils.clip_grad_norm_
    original_norm_check = executor._validate_preclip_norm
    original_step = state.optimizer.step
    original_issue = executor._issue_synthetic_optimizer_transition  # type: ignore[attr-defined]
    gradient_checks = 0
    state_checks = 0

    def stable(
        candidate: executor._ExecutorState, *, expected_generation: int
    ) -> executor._RuntimeDigests:
        events.append(f"snapshot_{expected_generation}")
        return original_stable(candidate, expected_generation=expected_generation)

    def zero_grad(*, set_to_none: bool = True) -> None:
        assert set_to_none is True
        events.append("zero_grad")
        original_zero_grad(set_to_none=set_to_none)

    def synthetic_batch() -> tuple[
        executor._Float32Array,
        executor._Int64Array,
        Tensor,
        Tensor,
    ]:
        events.append("batch")
        return original_batch()

    def batch_check(
        model_inputs: executor._Float32Array,
        label_indices: executor._Int64Array,
        features: Tensor,
        labels: Tensor,
    ) -> None:
        assert model_inputs.flags.owndata and model_inputs.flags.c_contiguous
        assert label_indices.flags.owndata and label_indices.flags.c_contiguous
        assert features.data_ptr() == model_inputs.ctypes.data
        assert labels.data_ptr() == label_indices.ctypes.data
        events.append("batch_check")
        original_batch_check(model_inputs, label_indices, features, labels)

    def train(mode: bool = True) -> CausalKWS:
        assert mode is True
        events.append("train")
        return original_train(mode)

    def initial_state(
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> CausalKWSState:
        events.append("initial_state")
        return original_initial_state(batch_size, device=device, dtype=dtype)

    def state_check(
        stream_state: CausalKWSState, *, expected_frames: int, require_zero: bool
    ) -> None:
        nonlocal state_checks
        events.append(f"state_{state_checks}")
        state_checks += 1
        original_state_check(
            stream_state,
            expected_frames=expected_frames,
            require_zero=require_zero,
        )

    captured_logits: list[Tensor] = []

    def forward(
        features: Tensor, stream_state: CausalKWSState
    ) -> tuple[Tensor, CausalKWSState]:
        assert tuple(features.shape) == (1, 40, 98)
        events.append("forward")
        result = original_forward(features, stream_state)
        captured_logits.append(result[0])
        return result

    def logits_check(logits: Tensor) -> None:
        events.append("logits_check")
        original_logits_check(logits)

    def cross_entropy(
        supervised_logits: Tensor,
        labels: Tensor,
    ) -> Tensor:
        assert tuple(supervised_logits.shape) == (1, 12)
        assert tuple(labels.shape) == (1,)
        expected = captured_logits[0][:, 97, :]
        assert supervised_logits.data_ptr() == expected.data_ptr()
        assert supervised_logits.storage_offset() == expected.storage_offset()
        assert supervised_logits.stride() == expected.stride()
        assert torch.equal(supervised_logits, expected)
        events.append("cross_entropy")
        return original_cross_entropy(supervised_logits, labels)

    def loss_check(loss: Tensor) -> None:
        events.append("loss_check")
        original_loss_check(loss)

    def gradient_check(parameters: tuple[torch.nn.Parameter, ...]) -> None:
        nonlocal gradient_checks
        events.append(f"gradients_{gradient_checks}")
        gradient_checks += 1
        original_gradient_check(parameters)

    def clip(
        parameters: tuple[torch.nn.Parameter, ...],
        max_norm: float,
        *,
        norm_type: float,
        error_if_nonfinite: bool,
        foreach: bool | None,
    ) -> Tensor:
        assert parameters is state.parameters
        assert max_norm == 5.0
        assert norm_type == 2.0
        assert error_if_nonfinite is True
        assert foreach is False
        events.append("clip")
        return original_clip(
            parameters,
            max_norm,
            norm_type=norm_type,
            error_if_nonfinite=error_if_nonfinite,
            foreach=foreach,
        )

    def norm_check(norm: Tensor) -> None:
        events.append("norm_check")
        original_norm_check(norm)

    def step(closure: Callable[[], float] | None = None) -> None:
        assert closure is None
        assert struct.pack("<d", state.optimizer.param_groups[0]["lr"]) == struct.pack(
            "<d", 0.0003
        )
        events.append("step")
        original_step(closure)

    def issue(**kwargs: object) -> bridge.RegisteredOptimizerTransition:
        events.append("issue_synthetic")
        return original_issue(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(executor, "_stable_runtime_digests", stable)
    monkeypatch.setattr(state.optimizer, "zero_grad", zero_grad)
    monkeypatch.setattr(executor, "_synthetic_zero_batch", synthetic_batch)
    monkeypatch.setattr(executor, "_validate_synthetic_batch", batch_check)
    monkeypatch.setattr(state.model, "train", train)
    monkeypatch.setattr(state.model, "initial_state", initial_state)
    monkeypatch.setattr(executor, "_validate_stream_state", state_check)
    monkeypatch.setattr(state.model, "forward_stream", forward)
    monkeypatch.setattr(executor, "_validate_logits", logits_check)
    monkeypatch.setattr(executor, "_training_cross_entropy", cross_entropy)
    monkeypatch.setattr(executor, "_validate_loss", loss_check)
    monkeypatch.setattr(executor, "_validate_gradients", gradient_check)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    monkeypatch.setattr(executor, "_validate_preclip_norm", norm_check)
    monkeypatch.setattr(state.optimizer, "step", step)
    monkeypatch.setattr(
        executor,
        "_issue_synthetic_optimizer_transition",
        issue,
    )

    transition = executor._execute_synthetic_optimizer_update(issued)
    transition_snapshot = bridge._transition_snapshot(
        transition, required_phase="ISSUED"
    )
    after = executor._synthetic_executor_snapshot(issued)
    assert events == [
        "snapshot_0",
        "zero_grad",
        "snapshot_0",
        "batch",
        "batch_check",
        "train",
        "initial_state",
        "state_0",
        "forward",
        "logits_check",
        "state_1",
        "cross_entropy",
        "loss_check",
        "gradients_0",
        "clip",
        "norm_check",
        "gradients_1",
        "step",
        "snapshot_1",
        "issue_synthetic",
        "snapshot_1",
        "snapshot_1",
    ]
    assert registered_calls == {
        "session": 0,
        "epoch": 0,
        "batch": 0,
        "trace": 0,
        "validation": 0,
        "metrics": 0,
        "evidence": 0,
    }
    assert transition_snapshot.seed == 29
    assert transition_snapshot.zero_based_epoch == 0
    assert transition_snapshot.zero_based_global_update == 0
    assert transition_snapshot.batch_size == 1
    assert transition_snapshot.learning_rate == 0.0003
    assert transition_snapshot.optimizer_generation_before == 0
    assert transition_snapshot.optimizer_generation_after == 1
    assert transition_snapshot.batch_mean_training_loss == np.float32(
        transition_snapshot.batch_mean_training_loss
    )
    assert np.isfinite(transition_snapshot.batch_mean_training_loss)
    assert np.isfinite(transition_snapshot.returned_preclip_l2_norm)
    assert transition_snapshot.model_sha256_before == before.model_sha256
    assert transition_snapshot.optimizer_sha256_before == before.optimizer_sha256
    assert transition_snapshot.rng_sha256_before == before.rng_sha256
    assert transition_snapshot.model_sha256_after == after.model_sha256
    assert transition_snapshot.optimizer_sha256_after == after.optimizer_sha256
    assert transition_snapshot.rng_sha256_after == after.rng_sha256
    assert before.model_sha256 != after.model_sha256
    assert before.optimizer_sha256 != after.optimizer_sha256
    assert before.rng_sha256 != after.rng_sha256
    assert after.phase == "COMPLETE"
    assert after.optimizer_generation == 1
    assert len(state.optimizer.state) == 53
    for parameter in state.parameters:
        optimizer_state = state.optimizer.state[parameter]
        assert set(optimizer_state) == {"step", "exp_avg", "exp_avg_sq"}
        step_tensor = optimizer_state["step"]
        assert step_tensor.device.type == "cpu"
        assert step_tensor.dtype == torch.float32
        assert step_tensor.shape == torch.Size([])
        assert step_tensor.item() == 1.0
        for name in ("exp_avg", "exp_avg_sq"):
            tensor = optimizer_state[name]
            assert tensor.device.type == "cpu"
            assert tensor.dtype == torch.float32
            assert tensor.shape == parameter.shape
            assert tensor.is_contiguous()
            assert bool(torch.isfinite(tensor).all().item())

    first_parameter = state.parameters[0]
    original_parameter_value = first_parameter.view(-1)[0].detach().clone()
    with torch.no_grad():
        first_parameter.view(-1)[0] = float("inf")
    with pytest.raises(executor.Experiment002TrainingExecutorError, match="invalid"):
        executor._validate_parameter_capture(
            state.model, state.parameter_names, state.parameters
        )
    with torch.no_grad():
        first_parameter.view(-1)[0] = original_parameter_value

    first_optimizer_state = state.optimizer.state[first_parameter]
    removed = first_optimizer_state.pop("exp_avg_sq")
    with pytest.raises(executor.Experiment002TrainingExecutorError, match="state keys"):
        executor._validate_optimizer(state, expected_generation=1)
    first_optimizer_state["exp_avg_sq"] = removed

    step_tensor = first_optimizer_state["step"]
    step_tensor.fill_(2.0)
    with pytest.raises(
        executor.Experiment002TrainingExecutorError, match="step counter"
    ):
        executor._validate_optimizer(state, expected_generation=1)
    step_tensor.fill_(1.0)

    moment = first_optimizer_state["exp_avg"]
    original_moment = moment.view(-1)[0].clone()
    moment.view(-1)[0] = float("inf")
    with pytest.raises(
        executor.Experiment002TrainingExecutorError, match="exp_avg tensor"
    ):
        executor._validate_optimizer(state, expected_generation=1)
    moment.view(-1)[0] = original_moment
    executor._validate_optimizer(state, expected_generation=1)

    with pytest.raises(executor.Experiment002TrainingExecutorError, match="one-shot"):
        executor._execute_synthetic_optimizer_update(issued)
    assert bridge._issued_transition_phase(transition) == "ISSUED"
    assert executor._synthetic_executor_snapshot(issued).phase == "COMPLETE"
    bridge._fail_optimizer_transition(transition)


def test_rng_continuity_mismatch_fails_before_zero_grad_or_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    issued = executor._create_synthetic_training_executor(seed=31)
    state = executor._issued_executor_state(issued)
    calls = {"zero_grad": 0, "transition": 0}
    original_zero_grad = state.optimizer.zero_grad

    def zero_grad(*, set_to_none: bool = True) -> None:
        calls["zero_grad"] += 1
        original_zero_grad(set_to_none=set_to_none)

    def transition(**kwargs: object) -> bridge.RegisteredOptimizerTransition:
        del kwargs
        calls["transition"] += 1
        raise AssertionError("transition must not be issued")

    monkeypatch.setattr(state.optimizer, "zero_grad", zero_grad)
    monkeypatch.setattr(
        executor,
        "_issue_synthetic_optimizer_transition",
        transition,
    )
    torch.rand(1)
    with pytest.raises(
        executor.Experiment002TrainingExecutorError, match="continuity changed"
    ):
        executor._execute_synthetic_optimizer_update(issued)
    assert calls == {"zero_grad": 0, "transition": 0}
    assert not state.optimizer.state
    assert state.phase == "FAILED"


def test_snapshot_recomputes_live_runtime_instead_of_returning_cache() -> None:
    issued = executor._create_synthetic_training_executor(seed=35)
    state = executor._issued_executor_state(issued)
    torch.rand(1)
    with pytest.raises(
        executor.Experiment002TrainingExecutorError, match="cached executor snapshot"
    ):
        executor._synthetic_executor_snapshot(issued)
    assert state.phase == "FAILED"


def test_executor_uses_guarded_lock_and_rejects_raw_lock_replacement() -> None:
    issued = executor._create_synthetic_training_executor(seed=131)
    state = executor._issued_executor_state(issued)
    original_lock = state.lock
    state.lock = threading.RLock()
    with pytest.raises(
        executor.Experiment002TrainingExecutorError, match="issuer authority"
    ):
        executor._synthetic_executor_snapshot(issued)
    assert state.phase == "FAILED"
    assert state.lock is not original_lock


def test_post_issue_failure_terminally_invalidates_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    issued = executor._create_synthetic_training_executor(seed=37)
    captured: list[bridge.RegisteredOptimizerTransition] = []
    original_issue = executor._issue_synthetic_optimizer_transition  # type: ignore[attr-defined]

    def issue(**kwargs: object) -> bridge.RegisteredOptimizerTransition:
        transition = original_issue(**kwargs)  # type: ignore[arg-type]
        captured.append(transition)
        return transition

    def fail_publish(
        candidate: executor._SyntheticTrainingExecutor,
        state: executor._ExecutorState,
    ) -> None:
        del candidate, state
        raise RuntimeError("synthetic publish fault")

    monkeypatch.setattr(executor, "_issue_synthetic_optimizer_transition", issue)
    monkeypatch.setattr(executor, "_publish_lifecycle", fail_publish)
    with pytest.raises(RuntimeError, match="publish fault"):
        executor._execute_synthetic_optimizer_update(issued)
    assert len(captured) == 1
    assert bridge._issued_transition_phase(captured[0]) == "FAILED"
    assert executor._issued_executor_state(issued).phase == "FAILED"


@pytest.mark.parametrize("failure_window", ["transition_snapshot", "guard_bind"])
def test_transition_is_failed_in_every_post_issue_prebind_window(
    monkeypatch: pytest.MonkeyPatch, failure_window: str
) -> None:
    issued = executor._create_synthetic_training_executor(
        seed=101 if failure_window == "transition_snapshot" else 103
    )
    state = executor._issued_executor_state(issued)
    captured: list[bridge.RegisteredOptimizerTransition] = []
    original_issue = executor._issue_synthetic_optimizer_transition  # type: ignore[attr-defined]

    def issue(**kwargs: object) -> bridge.RegisteredOptimizerTransition:
        transition = original_issue(**kwargs)  # type: ignore[arg-type]
        captured.append(transition)
        return transition

    monkeypatch.setattr(
        executor,
        "_issue_synthetic_optimizer_transition",
        issue,
    )
    if failure_window == "transition_snapshot":

        def reject_snapshot(
            transition: bridge.RegisteredOptimizerTransition,
            *,
            required_phase: bridge.TransitionPhase | None = None,
        ) -> bridge._OptimizerTransitionSnapshot:
            del transition, required_phase
            raise RuntimeError("injected transition snapshot fault")

        monkeypatch.setattr(executor, "_transition_snapshot", reject_snapshot)
    else:

        def reject_bind(
            candidate: executor._SyntheticTrainingExecutor,
            candidate_state: executor._ExecutorState,
        ) -> None:
            del candidate, candidate_state
            raise RuntimeError("injected transition guard-bind fault")

        monkeypatch.setattr(executor, "_bind_transition_guard", reject_bind)

    with pytest.raises(RuntimeError, match="injected transition"):
        executor._execute_synthetic_optimizer_update(issued)
    assert len(captured) == 1
    assert bridge._issued_transition_phase(captured[0]) == "FAILED"
    assert state.phase == "FAILED"


def test_optimizer_tamper_is_terminal_before_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    issued = executor._create_synthetic_training_executor(seed=41)
    state = executor._issued_executor_state(issued)
    steps = 0
    original_step = state.optimizer.step

    def step(closure: Callable[[], float] | None = None) -> None:
        nonlocal steps
        steps += 1
        original_step(closure)

    monkeypatch.setattr(state.optimizer, "step", step)
    state.optimizer.param_groups[0]["weight_decay"] = 0.0
    with pytest.raises(
        executor.Experiment002TrainingExecutorError, match="parameter group"
    ):
        executor._execute_synthetic_optimizer_update(issued)
    assert steps == 0
    assert state.phase == "FAILED"


def test_stream_state_requires_exact_shapes_and_positive_zero() -> None:
    issued = executor._create_synthetic_training_executor(seed=43)
    model = executor._issued_executor_state(issued).model
    stream_state = model.initial_state(1)
    executor._validate_stream_state(stream_state, expected_frames=0, require_zero=True)

    wrong_shape = stream_state._replace(
        dw_state_0=torch.zeros((1, 48, 3), dtype=torch.float32)
    )
    with pytest.raises(
        executor.Experiment002TrainingExecutorError, match="float stream state"
    ):
        executor._validate_stream_state(
            wrong_shape, expected_frames=0, require_zero=True
        )

    negative_zero_pool = torch.full((1, 48, 37), -0.0, dtype=torch.float32)
    assert bool(torch.signbit(negative_zero_pool).all().item())
    negative_zero = stream_state._replace(pool_state=negative_zero_pool)
    with pytest.raises(
        executor.Experiment002TrainingExecutorError, match="pool stream state"
    ):
        executor._validate_stream_state(
            negative_zero, expected_frames=0, require_zero=True
        )


def test_gradient_and_preclip_validators_reject_all_nonfinite_forms() -> None:
    parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
    with pytest.raises(executor.Experiment002TrainingExecutorError, match="gradient"):
        executor._validate_gradients((parameter,))

    parameter.grad = torch.tensor([1.0, float("nan")], dtype=torch.float32)
    with pytest.raises(executor.Experiment002TrainingExecutorError, match="gradient"):
        executor._validate_gradients((parameter,))

    indices = torch.tensor([[0, 1]], dtype=torch.int64)
    values = torch.ones(2, dtype=torch.float32)
    parameter.grad = torch.sparse_coo_tensor(
        indices, values, (2,), check_invariants=True
    )
    with pytest.raises(executor.Experiment002TrainingExecutorError, match="gradient"):
        executor._validate_gradients((parameter,))

    wrong_dtype = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    wrong_dtype.grad = torch.ones(2, dtype=torch.float64)
    with pytest.raises(executor.Experiment002TrainingExecutorError, match="gradient"):
        executor._validate_gradients((wrong_dtype,))

    for value in (float("nan"), float("inf"), -1.0):
        with pytest.raises(
            executor.Experiment002TrainingExecutorError, match="preclip"
        ):
            executor._validate_preclip_norm(torch.tensor(value, dtype=torch.float32))


@pytest.mark.parametrize("mutation_stage", ["loss", "step"])
def test_rng_use_outside_forward_terminally_fails_without_transition(
    monkeypatch: pytest.MonkeyPatch, mutation_stage: str
) -> None:
    issued = executor._create_synthetic_training_executor(
        seed=47 if mutation_stage == "loss" else 53
    )
    state = executor._issued_executor_state(issued)
    transition_calls = 0

    def forbidden_transition(**kwargs: object) -> bridge.RegisteredOptimizerTransition:
        nonlocal transition_calls
        del kwargs
        transition_calls += 1
        raise AssertionError("RNG-corrupt update must not issue a transition")

    monkeypatch.setattr(
        executor,
        "_issue_synthetic_optimizer_transition",
        forbidden_transition,
    )
    if mutation_stage == "loss":
        original_loss = executor._training_cross_entropy

        def loss(logits: Tensor, labels: Tensor) -> Tensor:
            torch.rand(1)
            return original_loss(logits, labels)

        monkeypatch.setattr(executor, "_training_cross_entropy", loss)
    else:
        original_step = state.optimizer.step

        def step(closure: Callable[[], float] | None = None) -> None:
            original_step(closure)
            torch.rand(1)

        monkeypatch.setattr(state.optimizer, "step", step)

    with pytest.raises(
        executor.Experiment002TrainingExecutorError,
        match="RNG changed outside",
    ):
        executor._execute_synthetic_optimizer_update(issued)
    assert transition_calls == 0
    assert state.phase == "FAILED"
    assert len(state.optimizer.state) == 53
    assert state.active_transition is None


@pytest.mark.parametrize("fault", ["gradient", "clip"])
def test_pre_step_fault_never_steps_or_issues_transition(
    monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    issued = executor._create_synthetic_training_executor(
        seed=59 if fault == "gradient" else 61
    )
    state = executor._issued_executor_state(issued)
    steps = 0
    transitions = 0

    def step(closure: Callable[[], float] | None = None) -> None:
        nonlocal steps
        del closure
        steps += 1

    def transition(**kwargs: object) -> bridge.RegisteredOptimizerTransition:
        nonlocal transitions
        del kwargs
        transitions += 1
        raise AssertionError("pre-step fault issued a transition")

    monkeypatch.setattr(state.optimizer, "step", step)
    monkeypatch.setattr(
        executor,
        "_issue_synthetic_optimizer_transition",
        transition,
    )
    if fault == "gradient":

        def reject_gradients(
            parameters: tuple[torch.nn.Parameter, ...],
        ) -> None:
            del parameters
            raise executor.Experiment002TrainingExecutorError("injected gradient fault")

        monkeypatch.setattr(executor, "_validate_gradients", reject_gradients)
    else:

        def nonfinite_clip(
            parameters: tuple[torch.nn.Parameter, ...],
            max_norm: float,
            *,
            norm_type: float,
            error_if_nonfinite: bool,
            foreach: bool | None,
        ) -> Tensor:
            del parameters, max_norm, norm_type, error_if_nonfinite, foreach
            return torch.tensor(float("nan"), dtype=torch.float32)

        monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", nonfinite_clip)

    with pytest.raises((executor.Experiment002TrainingExecutorError, RuntimeError)):
        executor._execute_synthetic_optimizer_update(issued)
    assert steps == 0
    assert transitions == 0
    assert state.phase == "FAILED"
    assert state.active_transition is None
    assert not state.optimizer.state


@pytest.mark.parametrize("corruption", ["parameter", "state_key", "step", "moment"])
def test_live_snapshot_terminally_fails_corrupted_completed_transition(
    corruption: str,
) -> None:
    seed_by_corruption = {
        "parameter": 67,
        "state_key": 71,
        "step": 73,
        "moment": 79,
    }
    issued = executor._create_synthetic_training_executor(
        seed=seed_by_corruption[corruption]
    )
    transition = executor._execute_synthetic_optimizer_update(issued)
    state = executor._issued_executor_state(issued)
    parameter = state.parameters[0]
    optimizer_state = state.optimizer.state[parameter]
    if corruption == "parameter":
        with torch.no_grad():
            parameter.view(-1)[0] = float("inf")
    elif corruption == "state_key":
        optimizer_state.pop("exp_avg_sq")
    elif corruption == "step":
        optimizer_state["step"].fill_(2.0)
    else:
        optimizer_state["exp_avg"].view(-1)[0] = float("nan")

    with pytest.raises((executor.Experiment002TrainingExecutorError, ValueError)):
        executor._synthetic_executor_snapshot(issued)
    assert state.phase == "FAILED"
    assert bridge._issued_transition_phase(transition) == "FAILED"


def test_external_transition_failure_and_private_replacement_fail_closed() -> None:
    issued = executor._create_synthetic_training_executor(seed=83)
    transition = executor._execute_synthetic_optimizer_update(issued)
    state = executor._issued_executor_state(issued)
    bridge._fail_optimizer_transition(transition)
    with pytest.raises(ValueError):
        executor._execute_synthetic_optimizer_update(issued)
    assert state.phase == "FAILED"

    other = executor._create_synthetic_training_executor(seed=89)
    original = executor._execute_synthetic_optimizer_update(other)
    other_state = executor._issued_executor_state(other)
    foreign = bridge._issue_synthetic_optimizer_transition(
        seed=97,
        zero_based_epoch=0,
        zero_based_global_update=0,
        batch_size=1,
        learning_rate=0.0003,
        batch_mean_training_loss=np.float32(1.0),
        returned_preclip_l2_norm=np.float32(1.0),
        optimizer_generation_before=0,
        optimizer_generation_after=1,
        model_sha256_before="0" * 64,
        model_sha256_after="1" * 64,
        optimizer_sha256_before="2" * 64,
        optimizer_sha256_after="3" * 64,
        rng_sha256_before="4" * 64,
        rng_sha256_after="5" * 64,
    )
    other_state.active_transition = foreign
    with pytest.raises(executor.Experiment002TrainingExecutorError):
        executor._synthetic_executor_snapshot(other)
    assert other_state.phase == "FAILED"
    assert bridge._issued_transition_phase(original) == "FAILED"
    assert bridge._issued_transition_phase(foreign) == "ISSUED"
    bridge._fail_optimizer_transition(foreign)


def test_ready_foreign_transition_is_never_amplified_into_trusted_cleanup() -> None:
    issued = executor._create_synthetic_training_executor(seed=107)
    state = executor._issued_executor_state(issued)
    foreign = bridge._issue_synthetic_optimizer_transition(
        seed=109,
        zero_based_epoch=0,
        zero_based_global_update=0,
        batch_size=1,
        learning_rate=0.0003,
        batch_mean_training_loss=np.float32(1.0),
        returned_preclip_l2_norm=np.float32(1.0),
        optimizer_generation_before=0,
        optimizer_generation_after=1,
        model_sha256_before="0" * 64,
        model_sha256_after="1" * 64,
        optimizer_sha256_before="2" * 64,
        optimizer_sha256_after="3" * 64,
        rng_sha256_before="4" * 64,
        rng_sha256_after="5" * 64,
    )
    state.active_transition = foreign
    state.active_transition_token = bridge._transition_snapshot(
        foreign, required_phase="ISSUED"
    ).transition_token
    for _ in range(2):
        with pytest.raises(executor.Experiment002TrainingExecutorError):
            executor._synthetic_executor_snapshot(issued)
        assert bridge._issued_transition_phase(foreign) == "ISSUED"
    assert state.phase == "FAILED"
    assert state.active_transition is None
    assert state.active_transition_token is None
    bridge._fail_optimizer_transition(foreign)


@pytest.mark.parametrize("runtime_drift", ["matmul_precision", "flush_denormal"])
def test_runtime_drift_fails_executor_before_zero_grad_or_transition(
    monkeypatch: pytest.MonkeyPatch, runtime_drift: str
) -> None:
    issued = executor._create_synthetic_training_executor(
        seed=113 if runtime_drift == "matmul_precision" else 127
    )
    state = executor._issued_executor_state(issued)
    calls = {"zero_grad": 0, "transition": 0}

    def zero_grad(*, set_to_none: bool = True) -> None:
        del set_to_none
        calls["zero_grad"] += 1

    def transition(**kwargs: object) -> bridge.RegisteredOptimizerTransition:
        del kwargs
        calls["transition"] += 1
        raise AssertionError("runtime-drifted executor issued a transition")

    monkeypatch.setattr(state.optimizer, "zero_grad", zero_grad)
    monkeypatch.setattr(
        executor,
        "_issue_synthetic_optimizer_transition",
        transition,
    )
    if runtime_drift == "matmul_precision":
        torch.set_float32_matmul_precision("medium")
    else:
        assert torch.set_flush_denormal(True) is True
    try:
        with pytest.raises(
            executor.Experiment002TrainingExecutorError, match="Torch runtime"
        ):
            executor._execute_synthetic_optimizer_update(issued)
    finally:
        if runtime_drift == "matmul_precision":
            torch.set_float32_matmul_precision("highest")
        else:
            assert torch.set_flush_denormal(False) is True
    assert calls == {"zero_grad": 0, "transition": 0}
    assert state.phase == "FAILED"
    assert state.active_transition is None
