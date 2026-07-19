from __future__ import annotations

import ast
import copy
import hashlib
import types
import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from torch import Tensor

import falsewake.experiment_002_evaluator as evaluator
import falsewake.experiment_002_evidence as validation_evidence
import falsewake.experiment_002_metrics as metrics
import falsewake.experiment_002_training_evidence as training_evidence
import falsewake.experiment_002_training_population as training_population
import falsewake.experiment_002_validation as validation
from falsewake.causal_kws import (
    CausalDepthwiseSeparableBlock,
    CausalKWS,
    CausalKWSState,
)


def _model(seed: int = 20260719) -> CausalKWS:
    torch.manual_seed(seed)
    return CausalKWS()


def _inputs(example_count: int = 3) -> evaluator.Float32Array:
    values = (
        np.arange(example_count * 40 * 98, dtype=np.float32)
        .reshape(example_count, 40, 98)
        .copy(order="C")
    )
    values *= np.float32(1.0 / max(1, values.size))
    values.setflags(write=False)
    return values


def _run(
    model: CausalKWS,
    model_inputs: evaluator.Float32Array,
    *,
    batch_size: int = 2,
    another_training_epoch: bool = True,
) -> evaluator._UnregisteredEvaluationResult:
    return evaluator._evaluate_unregistered_causal_model(
        model,
        model_inputs,
        layout=evaluator._EvaluationLayout(
            example_count=len(model_inputs),
            batch_size=batch_size,
        ),
        another_training_epoch=another_training_epoch,
    )


def _manual_frame_97_logits(
    model: CausalKWS,
    model_inputs: evaluator.Float32Array,
    *,
    batch_size: int,
) -> evaluator.Float32Array:
    result = np.empty((len(model_inputs), 12), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(model_inputs), batch_size):
            stop = min(start + batch_size, len(model_inputs))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                features = torch.from_numpy(model_inputs[start:stop])
            state = model.initial_state(
                stop - start,
                device=features.device,
                dtype=features.dtype,
            )
            logits, _ = model.forward_stream(features, state)
            result[start:stop] = logits[:, 97, :].contiguous().numpy()
    result.setflags(write=False)
    return result


def _model_evidence_sha256(model: CausalKWS) -> str:
    evidence = training_evidence._capture_model_tensors(
        model,
        layout=training_evidence._ModelTensorLayout(
            tensor_count=53,
            value_count=23_724,
        ),
    )
    return evidence.sha256


def test_source_has_no_registered_data_metric_or_artifact_route() -> None:
    source = Path("src/falsewake/experiment_002_evaluator.py").read_text(
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
        "collections",
        "dataclasses",
        "falsewake",
        "hashlib",
        "numpy",
        "struct",
        "torch",
        "typing",
        "warnings",
    }
    assert all(name.startswith("_") for name in module_functions)
    assert "VALIDATION_EXAMPLE_COUNT" not in source
    assert "10_583" not in source
    assert "issue_registered" not in source
    assert "capture_registered" not in source
    assert "safetensors" not in source
    assert "json" not in imported_roots


def test_exact_happy_path_matches_an_independent_batched_oracle() -> None:
    model = _model()
    oracle_model = copy.deepcopy(model)
    model_inputs = _inputs(5)
    input_payload = model_inputs.tobytes(order="C")
    model_sha256 = _model_evidence_sha256(model)
    rng_before = torch.get_rng_state().numpy().tobytes(order="C")
    expected = _manual_frame_97_logits(
        oracle_model,
        model_inputs,
        batch_size=2,
    )
    assert torch.get_rng_state().numpy().tobytes(order="C") == rng_before

    result = _run(model, model_inputs, batch_size=2)

    assert result.example_count == 5
    assert result.batch_count == 3
    assert result.model_inputs_sha256 == hashlib.sha256(input_payload).hexdigest()
    assert result.model_tensor_sha256 == model_sha256
    assert result.torch_rng_sha256 == evaluator._torch_rng_sha256(rng_before)
    assert (
        result.torch_rng_sha256
        == hashlib.sha256(
            evaluator._TORCH_RNG_DOMAIN
            + len(rng_before).to_bytes(8, "little")
            + rng_before
        ).hexdigest()
    )
    assert result.model_training_after is True
    assert model.training
    assert all(module.training for module in model.modules())
    np.testing.assert_array_equal(result.frame_97_logits, expected)
    observed = result.frame_97_logits
    assert observed.dtype == np.dtype(np.float32)
    assert observed.shape == (5, 12)
    assert observed.flags.c_contiguous
    assert observed.flags.owndata
    assert not observed.flags.writeable
    assert result.frame_97_logits is not observed
    assert model_inputs.tobytes(order="C") == input_payload
    assert torch.get_rng_state().numpy().tobytes(order="C") == rng_before
    assert _model_evidence_sha256(model) == model_sha256


def test_final_evaluation_leaves_every_module_in_eval_mode() -> None:
    model = _model()
    result = _run(
        model,
        _inputs(1),
        batch_size=1,
        another_training_epoch=False,
    )
    assert result.model_training_after is False
    assert not model.training
    assert all(not module.training for module in model.modules())


def test_direct_views_fresh_states_and_one_forward_per_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    model_inputs = _inputs(5)
    numpy_batches: list[np.ndarray[Any, Any]] = []
    tensor_batches: list[Tensor] = []
    initial_states: list[CausalKWSState] = []
    forwarded_states: list[CausalKWSState] = []
    inference_flags: list[bool] = []
    actual_from_numpy = torch.from_numpy
    actual_initial_state = CausalKWS.initial_state
    actual_forward_stream = CausalKWS.forward_stream

    def from_numpy_spy(array: np.ndarray[Any, Any]) -> Tensor:
        numpy_batches.append(array)
        tensor = actual_from_numpy(array)
        tensor_batches.append(tensor)
        return tensor

    def initial_state_spy(
        self: CausalKWS,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> CausalKWSState:
        state = actual_initial_state(
            self,
            batch_size,
            device=device,
            dtype=dtype,
        )
        initial_states.append(state)
        return state

    def forward_spy(
        self: CausalKWS,
        features: Tensor,
        state: CausalKWSState,
    ) -> tuple[Tensor, CausalKWSState]:
        inference_flags.append(torch.is_inference_mode_enabled())
        forwarded_states.append(state)
        return actual_forward_stream(self, features, state)

    monkeypatch.setattr(torch, "from_numpy", from_numpy_spy)
    monkeypatch.setattr(CausalKWS, "initial_state", initial_state_spy)
    monkeypatch.setattr(CausalKWS, "forward_stream", forward_spy)

    result = _run(model, model_inputs, batch_size=2)

    assert result.batch_count == 3
    assert [array.shape[0] for array in numpy_batches] == [2, 2, 1]
    assert len(tensor_batches) == len(initial_states) == len(forwarded_states) == 3
    assert all(not array.flags.owndata for array in numpy_batches)
    assert all(np.shares_memory(array, model_inputs) for array in numpy_batches)
    assert all(array.flags.c_contiguous for array in numpy_batches)
    assert all(not array.flags.writeable for array in numpy_batches)
    assert all(
        tensor.data_ptr() == int(array.__array_interface__["data"][0])
        for tensor, array in zip(tensor_batches, numpy_batches, strict=True)
    )
    assert all(
        issued is forwarded
        for issued, forwarded in zip(initial_states, forwarded_states, strict=True)
    )
    assert len({id(state) for state in initial_states}) == 3
    assert inference_flags == [True, True, True]


def test_existing_gradients_are_preserved_byte_exactly() -> None:
    model = _model()
    before: list[bytes] = []
    for index, parameter in enumerate(model.parameters()):
        parameter.grad = torch.full_like(parameter, float(index + 1) / 64.0)
        before.append(parameter.grad.numpy().tobytes(order="C"))

    _run(model, _inputs(2), batch_size=1)

    after = [
        cast(Tensor, parameter.grad).numpy().tobytes(order="C")
        for parameter in model.parameters()
    ]
    assert after == before


def test_absent_gradients_remain_absent() -> None:
    model = _model()
    assert all(parameter.grad is None for parameter in model.parameters())
    _run(model, _inputs(2), batch_size=2)
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize(
    ("example_count", "batch_size", "error_type"),
    [
        (0, 1, evaluator.Experiment002EvaluatorError),
        (1, 0, evaluator.Experiment002EvaluatorError),
        (-1, 1, evaluator.Experiment002EvaluatorError),
        (1, -1, evaluator.Experiment002EvaluatorError),
        (9, 1, evaluator.Experiment002EvaluatorError),
        (10_583, 128, evaluator.Experiment002EvaluatorError),
        (True, 1, TypeError),
        (1, False, TypeError),
        (1.0, 1, TypeError),
        (1, 1.0, TypeError),
    ],
)
def test_layout_rejects_invalid_counts(
    example_count: object,
    batch_size: object,
    error_type: type[BaseException],
) -> None:
    with pytest.raises(error_type):
        evaluator._EvaluationLayout(
            example_count=cast(int, example_count),
            batch_size=cast(int, batch_size),
        )


def test_layout_batch_count_uses_exact_ceiling_division() -> None:
    assert evaluator._EvaluationLayout(1, 128).batch_count == 1
    assert evaluator._EvaluationLayout(4, 4).batch_count == 1
    assert evaluator._EvaluationLayout(5, 4).batch_count == 2
    assert evaluator._EvaluationLayout(8, 3).batch_count == 3


def test_unregistered_kernel_has_a_hard_tiny_population_boundary() -> None:
    assert evaluator._MAX_UNREGISTERED_EXAMPLES == 8
    assert evaluator._EvaluationLayout(8, 8).example_count == 8
    for forbidden_count in (9, 10_583, 2**32):
        with pytest.raises(
            evaluator.Experiment002EvaluatorError,
            match="synthetic example limit",
        ):
            evaluator._EvaluationLayout(forbidden_count, 128)


@pytest.mark.parametrize(
    "mutator",
    [
        "wrong_dtype",
        "wrong_shape",
        "writeable",
        "nonowner",
        "fortran",
        "nan",
        "positive_infinity",
        "negative_infinity",
    ],
)
def test_input_storage_and_values_fail_closed(mutator: str) -> None:
    model_inputs = _inputs(2)
    candidate: np.ndarray[Any, Any]
    if mutator == "wrong_dtype":
        candidate = model_inputs.astype(np.float64)
        candidate.setflags(write=False)
    elif mutator == "wrong_shape":
        candidate = np.zeros((2, 20, 196), dtype=np.float32)
        candidate.setflags(write=False)
    elif mutator == "writeable":
        candidate = model_inputs.copy(order="C")
    elif mutator == "nonowner":
        owner = np.zeros((3, 40, 98), dtype=np.float32)
        candidate = owner[1:]
        candidate.setflags(write=False)
    elif mutator == "fortran":
        candidate = np.asfortranarray(model_inputs)
        candidate.setflags(write=False)
    else:
        candidate = model_inputs.copy(order="C")
        replacement = {
            "nan": np.nan,
            "positive_infinity": np.inf,
            "negative_infinity": -np.inf,
        }[mutator]
        candidate[0, 0, 0] = np.float32(replacement)
        candidate.setflags(write=False)

    with pytest.raises(evaluator.Experiment002EvaluatorError):
        evaluator._evaluate_unregistered_causal_model(
            _model(),
            cast(evaluator.Float32Array, candidate),
            layout=evaluator._EvaluationLayout(2, 2),
            another_training_epoch=True,
        )


def test_exact_types_and_entry_mode_are_required() -> None:
    class DerivedCausalKWS(CausalKWS):
        pass

    with pytest.raises(TypeError, match="exact CausalKWS"):
        _run(cast(CausalKWS, DerivedCausalKWS()), _inputs(1), batch_size=1)

    model = _model()
    model.eval()
    with pytest.raises(evaluator.Experiment002EvaluatorError, match="uniformly"):
        _run(model, _inputs(1), batch_size=1)

    model = _model()
    model.blocks[0].eval()
    with pytest.raises(evaluator.Experiment002EvaluatorError, match="uniformly"):
        _run(model, _inputs(1), batch_size=1)

    with pytest.raises(TypeError, match="boolean"):
        evaluator._evaluate_unregistered_causal_model(
            _model(),
            _inputs(1),
            layout=evaluator._EvaluationLayout(1, 1),
            another_training_epoch=cast(bool, 1),
        )

    with pytest.raises(TypeError, match="exact _EvaluationLayout"):
        evaluator._evaluate_unregistered_causal_model(
            _model(),
            _inputs(1),
            layout=cast(evaluator._EvaluationLayout, object()),
            another_training_epoch=True,
        )


def test_a_copying_from_numpy_substitute_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def copied(array: np.ndarray[Any, Any]) -> Tensor:
        return torch.tensor(array, dtype=torch.float32)

    monkeypatch.setattr(torch, "from_numpy", copied)
    with pytest.raises(evaluator.Experiment002EvaluatorError, match="direct CPU"):
        _run(_model(), _inputs(1), batch_size=1)


def test_rng_consumption_inside_forward_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual = CausalKWS.forward_stream

    def consuming(
        self: CausalKWS,
        features: Tensor,
        state: CausalKWSState,
    ) -> tuple[Tensor, CausalKWSState]:
        result = actual(self, features, state)
        torch.rand(1)
        return result

    monkeypatch.setattr(CausalKWS, "forward_stream", consuming)
    with pytest.raises(evaluator.Experiment002EvaluatorError, match="RNG changed"):
        model = _model()
        _run(model, _inputs(1), batch_size=1)
    assert all(module.training for module in model.modules())


@pytest.mark.parametrize("another_training_epoch", [True, False])
def test_forward_exception_restores_the_required_model_mode(
    monkeypatch: pytest.MonkeyPatch,
    another_training_epoch: bool,
) -> None:
    def failing(
        self: CausalKWS,
        features: Tensor,
        state: CausalKWSState,
    ) -> tuple[Tensor, CausalKWSState]:
        del self, features, state
        raise RuntimeError("injected forward failure")

    monkeypatch.setattr(CausalKWS, "forward_stream", failing)
    model = _model()
    with pytest.raises(RuntimeError, match="injected forward failure"):
        _run(
            model,
            _inputs(1),
            batch_size=1,
            another_training_epoch=another_training_epoch,
        )
    assert all(module.training is another_training_epoch for module in model.modules())


def test_parameter_mutation_inside_forward_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual = CausalKWS.forward_stream

    def mutating(
        self: CausalKWS,
        features: Tensor,
        state: CausalKWSState,
    ) -> tuple[Tensor, CausalKWSState]:
        result = actual(self, features, state)
        self.stem.weight.add_(torch.ones_like(self.stem.weight))
        return result

    monkeypatch.setattr(CausalKWS, "forward_stream", mutating)
    with pytest.raises(
        evaluator.Experiment002EvaluatorError,
        match="model tensor bytes changed",
    ):
        _run(_model(), _inputs(1), batch_size=1)


@pytest.mark.parametrize("replacement_requires_grad", [True, False])
def test_byte_identical_parameter_replacement_fails_identity_authority(
    monkeypatch: pytest.MonkeyPatch,
    replacement_requires_grad: bool,
) -> None:
    actual = CausalKWS.forward_stream

    def replacing(
        self: CausalKWS,
        features: Tensor,
        state: CausalKWSState,
    ) -> tuple[Tensor, CausalKWSState]:
        result = actual(self, features, state)
        self.stem.weight = torch.nn.Parameter(
            self.stem.weight.detach().clone(),
            requires_grad=replacement_requires_grad,
        )
        return result

    monkeypatch.setattr(CausalKWS, "forward_stream", replacing)
    model = _model()
    original = model.stem.weight
    original_bytes = original.detach().numpy().tobytes(order="C")
    with pytest.raises(
        evaluator.Experiment002EvaluatorError,
        match="parameter identity",
    ):
        _run(model, _inputs(1), batch_size=1)
    assert model.stem.weight is not original
    assert model.stem.weight.detach().numpy().tobytes(order="C") == original_bytes
    assert all(module.training for module in model.modules())


@pytest.mark.parametrize(
    "mutation",
    ["dropout_probability", "stem_activation", "block_dilation", "layer_norm_eps"],
)
def test_model_structure_mutation_during_evaluation_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    actual = CausalKWS.forward_stream

    def mutating(
        self: CausalKWS,
        features: Tensor,
        state: CausalKWSState,
    ) -> tuple[Tensor, CausalKWSState]:
        result = actual(self, features, state)
        if mutation == "dropout_probability":
            self.dropout.p = 0.9
        elif mutation == "stem_activation":
            self.stem_activation = torch.nn.ReLU()
        elif mutation == "block_dilation":
            cast(CausalDepthwiseSeparableBlock, self.blocks[0]).dilation = 2
        else:
            self.stem_norm.norm.eps = 1e-4
        return result

    monkeypatch.setattr(CausalKWS, "forward_stream", mutating)
    model = _model()
    with pytest.raises(evaluator.Experiment002EvaluatorError, match="changed"):
        _run(model, _inputs(1), batch_size=1)
    assert all(module.training for module in model.modules())


@pytest.mark.parametrize(
    "mutation",
    ["dropout_probability", "relu_inplace", "conv_padding", "extra_module"],
)
def test_model_structure_tampering_at_entry_is_rejected(mutation: str) -> None:
    model = _model()
    if mutation == "dropout_probability":
        model.dropout.p = 0.9
    elif mutation == "relu_inplace":
        model.stem_activation.inplace = True
    elif mutation == "conv_padding":
        model.stem.padding = (1,)
    else:
        model.add_module("foreign", torch.nn.Identity())
    with pytest.raises(evaluator.Experiment002EvaluatorError, match="changed"):
        _run(model, _inputs(1), batch_size=1)
    assert all(module.training for module in model.modules())


def test_gradient_mutation_inside_forward_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    first_parameter = next(model.parameters())
    first_parameter.grad = torch.zeros_like(first_parameter)
    actual = CausalKWS.forward_stream

    def mutating(
        self: CausalKWS,
        features: Tensor,
        state: CausalKWSState,
    ) -> tuple[Tensor, CausalKWSState]:
        result = actual(self, features, state)
        cast(Tensor, first_parameter.grad).add_(1.0)
        return result

    monkeypatch.setattr(CausalKWS, "forward_stream", mutating)
    with pytest.raises(
        evaluator.Experiment002EvaluatorError,
        match="gradients changed",
    ):
        _run(model, _inputs(1), batch_size=1)


def test_input_mutation_during_forward_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_inputs = _inputs(2)
    actual = CausalKWS.forward_stream
    called = False

    def mutating(
        self: CausalKWS,
        features: Tensor,
        state: CausalKWSState,
    ) -> tuple[Tensor, CausalKWSState]:
        nonlocal called
        result = actual(self, features, state)
        if not called:
            model_inputs.setflags(write=True)
            model_inputs[0, 0, 0] = np.float32(-7.0)
            model_inputs.setflags(write=False)
            called = True
        return result

    monkeypatch.setattr(CausalKWS, "forward_stream", mutating)
    with pytest.raises(
        evaluator.Experiment002EvaluatorError,
        match="input bytes changed",
    ):
        _run(_model(), model_inputs, batch_size=1)


@pytest.mark.parametrize(
    "corruption",
    [
        "wrong_shape",
        "wrong_dtype",
        "nan",
        "noncontiguous",
        "parameter",
        "requires_grad",
    ],
)
def test_logit_corruption_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    actual = CausalKWS.forward_stream

    def corrupting(
        self: CausalKWS,
        features: Tensor,
        state: CausalKWSState,
    ) -> tuple[Tensor, CausalKWSState]:
        logits, next_state = actual(self, features, state)
        if corruption == "wrong_shape":
            logits = logits[:, :-1, :]
        elif corruption == "wrong_dtype":
            logits = logits.to(torch.float64)
        elif corruption == "nan":
            logits = logits.clone()
            logits[0, 0, 0] = torch.nan
        elif corruption == "parameter":
            logits = torch.nn.Parameter(logits.detach().clone(), requires_grad=False)
        elif corruption == "requires_grad":
            logits = torch.zeros(
                logits.shape,
                dtype=torch.float32,
                requires_grad=True,
            )
        else:
            logits = logits.transpose(1, 2)
        return logits, next_state

    monkeypatch.setattr(CausalKWS, "forward_stream", corrupting)
    with pytest.raises(evaluator.Experiment002EvaluatorError, match="logits"):
        _run(_model(), _inputs(1), batch_size=1)


@pytest.mark.parametrize(
    "corruption",
    [
        "wrong_type",
        "wrong_frames",
        "nan_history",
        "mutated_initial",
        "parameter_history",
        "requires_grad_history",
        "parameter_pool",
        "requires_grad_pool",
    ],
)
def test_stream_state_corruption_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    actual = CausalKWS.forward_stream

    def corrupting(
        self: CausalKWS,
        features: Tensor,
        state: CausalKWSState,
    ) -> tuple[Tensor, Any]:
        logits, next_state = actual(self, features, state)
        if corruption == "wrong_type":
            return logits, tuple(next_state)
        if corruption == "wrong_frames":
            return logits, next_state._replace(
                frames_seen=torch.zeros_like(next_state.frames_seen)
            )
        if corruption == "nan_history":
            changed = next_state.dw_state_0.clone()
            changed[0, 0, 0] = torch.nan
            return logits, next_state._replace(dw_state_0=changed)
        if corruption == "parameter_history":
            changed = torch.nn.Parameter(
                next_state.dw_state_0.detach().clone(),
                requires_grad=False,
            )
            return logits, next_state._replace(dw_state_0=changed)
        if corruption == "requires_grad_history":
            changed = torch.zeros(
                next_state.dw_state_0.shape,
                dtype=torch.float32,
                requires_grad=True,
            )
            return logits, next_state._replace(dw_state_0=changed)
        if corruption == "parameter_pool":
            changed = torch.nn.Parameter(
                next_state.pool_state.detach().clone(),
                requires_grad=False,
            )
            return logits, next_state._replace(pool_state=changed)
        if corruption == "requires_grad_pool":
            changed = torch.zeros(
                next_state.pool_state.shape,
                dtype=torch.float32,
                requires_grad=True,
            )
            return logits, next_state._replace(pool_state=changed)
        state.pool_state.add_(1.0)
        return logits, next_state

    monkeypatch.setattr(CausalKWS, "forward_stream", corrupting)
    with pytest.raises(evaluator.Experiment002EvaluatorError):
        _run(_model(), _inputs(1), batch_size=1)


@pytest.mark.parametrize("corruption", ["parameter", "requires_grad"])
def test_initial_state_must_use_plain_no_grad_tensors(
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    actual = CausalKWS.initial_state

    def corrupting(
        self: CausalKWS,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> CausalKWSState:
        state = actual(self, batch_size, device=device, dtype=dtype)
        changed: Tensor
        if corruption == "parameter":
            changed = torch.nn.Parameter(
                state.dw_state_0.detach().clone(),
                requires_grad=False,
            )
        else:
            changed = torch.zeros(
                state.dw_state_0.shape,
                dtype=torch.float32,
                requires_grad=True,
            )
        return state._replace(dw_state_0=changed)

    monkeypatch.setattr(CausalKWS, "initial_state", corrupting)
    model = _model()
    with pytest.raises(
        evaluator.Experiment002EvaluatorError,
        match="exact plain inference tensor",
    ):
        _run(model, _inputs(1), batch_size=1)
    assert all(module.training for module in model.modules())


def test_preinstalled_module_forward_hook_is_rejected() -> None:
    model = _model()
    handle = model.classifier.register_forward_hook(
        lambda module, args, output: torch.zeros_like(output)
    )
    try:
        with pytest.raises(
            evaluator.Experiment002EvaluatorError,
            match="hook registry is not empty",
        ):
            _run(model, _inputs(1), batch_size=1)
    finally:
        handle.remove()
    assert all(module.training for module in model.modules())


def test_preinstalled_global_forward_hook_is_rejected() -> None:
    handle = torch.nn.modules.module.register_module_forward_hook(
        lambda module, args, output: output
    )
    model = _model()
    try:
        with pytest.raises(
            evaluator.Experiment002EvaluatorError,
            match="global Torch hook registry is not empty",
        ):
            _run(model, _inputs(1), batch_size=1)
    finally:
        handle.remove()
    assert all(module.training for module in model.modules())


@pytest.mark.parametrize(
    "target",
    [
        "model",
        "submodule",
        "call_impl",
        "compiled_call_impl",
        "wrapped_call_impl",
        "slow_forward",
    ],
)
def test_instance_execution_method_override_is_rejected(target: str) -> None:
    model = _model()

    def replacement(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("instance override must never execute")

    if target == "model":
        object.__setattr__(
            model,
            "forward_stream",
            types.MethodType(replacement, model),
        )
    elif target == "submodule":
        object.__setattr__(
            model.stem_activation,
            "forward",
            types.MethodType(replacement, model.stem_activation),
        )
    else:
        attribute = {
            "call_impl": "_call_impl",
            "compiled_call_impl": "_compiled_call_impl",
            "wrapped_call_impl": "_wrapped_call_impl",
            "slow_forward": "_slow_forward",
        }[target]
        object.__setattr__(
            model.classifier,
            attribute,
            types.MethodType(replacement, model.classifier),
        )
    with pytest.raises(
        evaluator.Experiment002EvaluatorError,
        match=(
            "compiled call path"
            if target == "compiled_call_impl"
            else "instance execution-method override"
        ),
    ):
        _run(model, _inputs(1), batch_size=1)
    assert all(module.training for module in model.modules())


@pytest.mark.parametrize("method_name", ["modules", "named_parameters"])
def test_self_deleting_preflight_override_never_executes(method_name: str) -> None:
    model = _model()
    original = getattr(model, method_name)
    called = False

    def self_deleting(self: CausalKWS, *args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        object.__delattr__(self, method_name)
        return original(*args, **kwargs)

    object.__setattr__(
        model,
        method_name,
        types.MethodType(self_deleting, model),
    )
    with pytest.raises(
        evaluator.Experiment002EvaluatorError,
        match="instance execution-method override",
    ):
        _run(model, _inputs(1), batch_size=1)
    assert not called


def test_no_grad_outputs_are_not_accepted_as_inference_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual = CausalKWS.forward_stream

    def no_grad_only(
        self: CausalKWS,
        features: Tensor,
        state: CausalKWSState,
    ) -> tuple[Tensor, CausalKWSState]:
        with torch.inference_mode(False), torch.no_grad():
            normal_features = features.clone()
            normal_state = CausalKWSState(
                dw_state_0=state.dw_state_0.clone(),
                dw_state_1=state.dw_state_1.clone(),
                dw_state_2=state.dw_state_2.clone(),
                dw_state_3=state.dw_state_3.clone(),
                dw_state_4=state.dw_state_4.clone(),
                dw_state_5=state.dw_state_5.clone(),
                dw_state_6=state.dw_state_6.clone(),
                dw_state_7=state.dw_state_7.clone(),
                pool_state=state.pool_state.clone(),
                frames_seen=state.frames_seen.clone(),
            )
            return actual(self, normal_features, normal_state)

    monkeypatch.setattr(CausalKWS, "forward_stream", no_grad_only)
    model = _model()
    with pytest.raises(
        evaluator.Experiment002EvaluatorError,
        match="inference tensor",
    ):
        _run(model, _inputs(1), batch_size=1)
    assert all(module.training for module in model.modules())


def test_from_numpy_must_create_an_inference_tensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual = torch.from_numpy

    def ordinary_tensor(array: np.ndarray[Any, Any]) -> Tensor:
        with torch.inference_mode(False):
            return actual(array)

    monkeypatch.setattr(torch, "from_numpy", ordinary_tensor)
    with pytest.raises(
        evaluator.Experiment002EvaluatorError,
        match="direct CPU float32 view",
    ):
        _run(_model(), _inputs(1), batch_size=1)


def test_nonfinite_existing_gradient_fails_before_evaluation() -> None:
    model = _model()
    parameter = next(model.parameters())
    parameter.grad = torch.zeros_like(parameter)
    parameter.grad.reshape(-1)[0] = torch.nan
    with pytest.raises(evaluator.Experiment002EvaluatorError, match="gradient"):
        _run(model, _inputs(1), batch_size=1)


def test_registered_routes_are_never_called(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> Any:
        del args, kwargs
        raise AssertionError("registered route was called")

    for module, name in (
        (validation, "materialize_registered_validation"),
        (validation, "verify_registered_materialized_validation"),
        (validation_evidence, "bind_registered_validation_inputs"),
        (validation_evidence, "issue_registered_validation_evidence"),
        (metrics, "evaluate_validation_layout_logits"),
        (training_evidence, "capture_registered_model_tensors"),
        (training_population, "bind_registered_training_inputs"),
    ):
        monkeypatch.setattr(module, name, forbidden)

    result = _run(_model(), _inputs(1), batch_size=1)
    assert result.example_count == 1
    assert result.frame_97_logits.shape == (1, 12)


def test_result_payload_is_little_endian_float32_and_defensively_copied() -> None:
    result = _run(_model(), _inputs(2), batch_size=2)
    first = result.frame_97_logits
    second = result.frame_97_logits
    assert first is not second
    assert first.tobytes(order="C") == result.frame_97_logits_payload
    assert (
        np.frombuffer(
            result.frame_97_logits_payload,
            dtype=np.dtype("<f4"),
        ).tobytes()
        == result.frame_97_logits_payload
    )
    first.setflags(write=True)
    first[0, 0] = np.float32(123.0)
    np.testing.assert_array_equal(second, result.frame_97_logits)


def test_signed_zero_changes_the_model_digest_exactly_like_the_model_oracle() -> None:
    model = _model()
    first = next(model.parameters())
    with torch.no_grad():
        first.reshape(-1)[0] = 0.0
    positive = _run(model, _inputs(1), batch_size=1).model_tensor_sha256
    assert positive == _model_evidence_sha256(model)

    model = _model()
    first = next(model.parameters())
    with torch.no_grad():
        first.reshape(-1)[0] = -0.0
    negative = _run(model, _inputs(1), batch_size=1).model_tensor_sha256
    assert negative == _model_evidence_sha256(model)
    assert positive != negative


def test_batch_slices_cover_each_example_once_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_inputs = _inputs(7)
    observed_offsets: list[int] = []
    owner_address = int(model_inputs.__array_interface__["data"][0])
    bytes_per_example = 40 * 98 * 4
    actual = torch.from_numpy

    def recording(array: np.ndarray[Any, Any]) -> Tensor:
        address = int(array.__array_interface__["data"][0])
        observed_offsets.append((address - owner_address) // bytes_per_example)
        return actual(array)

    monkeypatch.setattr(torch, "from_numpy", recording)
    result = _run(_model(), model_inputs, batch_size=3)
    assert result.batch_count == 3
    assert observed_offsets == [0, 3, 6]


def test_output_does_not_retain_numpy_or_torch_input_storage() -> None:
    model_inputs = _inputs(2)
    result = _run(_model(), model_inputs, batch_size=1)
    logits = result.frame_97_logits
    assert not np.shares_memory(logits, model_inputs)
    assert isinstance(result.frame_97_logits_payload, bytes)
    assert len(result.frame_97_logits_payload) == 2 * 12 * 4


def test_forward_method_is_called_exactly_once_on_all_98_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int, int]] = []
    actual = CausalKWS.forward_stream

    def recording(
        self: CausalKWS,
        features: Tensor,
        state: CausalKWSState,
    ) -> tuple[Tensor, CausalKWSState]:
        calls.append(cast(tuple[int, int, int], tuple(features.shape)))
        return actual(self, features, state)

    monkeypatch.setattr(CausalKWS, "forward_stream", recording)
    _run(_model(), _inputs(5), batch_size=2)
    assert calls == [(2, 40, 98), (2, 40, 98), (1, 40, 98)]


def test_model_digest_has_no_dependency_on_module_training_flags() -> None:
    model = _model()
    training_digest = evaluator._stable_model_tensor_sha256(model)
    model.eval()
    evaluation_digest = evaluator._stable_model_tensor_sha256(model)
    assert training_digest == evaluation_digest


def test_unregistered_result_is_not_a_registered_evidence_type() -> None:
    result = _run(_model(), _inputs(1), batch_size=1)
    assert type(result) is evaluator._UnregisteredEvaluationResult
    assert not isinstance(
        result,
        validation_evidence.RegisteredValidationEvidence,
    )
    assert not isinstance(
        result,
        training_evidence.RegisteredModelTensorEvidence,
    )


def test_mixed_gradient_presence_is_preserved() -> None:
    model = _model()
    parameters = tuple(model.parameters())
    parameters[0].grad = torch.ones_like(parameters[0])
    parameters[-1].grad = torch.full_like(parameters[-1], 2.0)
    before = evaluator._gradient_snapshot(model)
    _run(model, _inputs(1), batch_size=1)
    assert evaluator._gradient_snapshot(model) == before


def test_model_parameter_layout_tampering_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    actual = model.state_dict

    def missing(self: CausalKWS) -> dict[str, Tensor]:
        state = actual()
        state.pop(next(iter(state)))
        return dict(state)

    monkeypatch.setattr(model, "state_dict", types.MethodType(missing, model))
    with pytest.raises(evaluator.Experiment002EvaluatorError, match="state_dict"):
        _run(model, _inputs(1), batch_size=1)
