from __future__ import annotations

import ast
import copy
import inspect
import os
import pickle
import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from torch import Tensor

import falsewake.experiment_002_evaluator as unregistered_kernel
import falsewake.experiment_002_registered_evaluator as evaluator
import falsewake.experiment_002_registered_executor as registered_executor
import falsewake.experiment_002_training_executor as numeric_executor
from falsewake.causal_kws import CausalKWS, CausalKWSState
from falsewake.experiment_002_evidence import (
    RegisteredValidationEvidence,
    RegisteredValidationInputs,
)
from falsewake.experiment_002_registered_executor import (
    RegisteredExecutorEpochHandoff,
    RegisteredTrainingExecutor,
)
from falsewake.experiment_002_run_authority import VerifiedRunRegistration
from falsewake.experiment_002_training_bridge import (
    RegisteredExecutorSessionAuthority,
)
from falsewake.experiment_002_training_evidence import (
    EpochUpdateTraceEvidence,
    RegisteredModelTensorEvidence,
)
from falsewake.experiment_002_training_population import CompletedTrainingPopulation


def _model(seed: int = 20_260_719) -> CausalKWS:
    torch.manual_seed(seed)
    return CausalKWS()


def _inputs(example_count: int = 3) -> evaluator.Float32Array:
    result = (
        np.arange(example_count * 40 * 98, dtype=np.float32)
        .reshape(example_count, 40, 98)
        .copy(order="C")
    )
    result *= np.float32(1.0 / max(result.size, 1))
    result.setflags(write=False)
    return result


def _run_synthetic(
    model: CausalKWS,
    inputs: evaluator.Float32Array,
    *,
    batch_size: int = 2,
    another_training_epoch: bool = True,
) -> evaluator._KernelResult:
    return evaluator._evaluate_synthetic_kernel(
        model,
        inputs,
        layout=evaluator._SyntheticEvaluationLayout(len(inputs), batch_size),
        another_training_epoch=another_training_epoch,
    )


def _manual_logits(
    model: CausalKWS,
    inputs: evaluator.Float32Array,
    *,
    batch_size: int,
) -> evaluator.Float32Array:
    result = np.empty((len(inputs), 12), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            stop = min(start + batch_size, len(inputs))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                features = torch.from_numpy(inputs[start:stop])
            state = model.initial_state(
                stop - start,
                device=features.device,
                dtype=features.dtype,
            )
            logits, _ = model.forward_stream(features, state)
            result[start:stop] = logits[:, 97, :].contiguous().numpy()
    result.setflags(write=False)
    return result


def _counterfeit_handoff_authority() -> tuple[
    RegisteredExecutorEpochHandoff,
    evaluator._HandoffEvaluationState,
    evaluator._HandoffEvaluationGuard,
]:
    registration = object.__new__(VerifiedRunRegistration)
    validation_inputs = object.__new__(RegisteredValidationInputs)
    handoff = object.__new__(RegisteredExecutorEpochHandoff)
    snapshot = registered_executor._RegisteredExecutorEpochHandoffSnapshot(
        executor=object.__new__(RegisteredTrainingExecutor),
        registration=registration,
        validation_inputs=validation_inputs,
        process_id=os.getpid(),
        bridge_authority=object.__new__(RegisteredExecutorSessionAuthority),
        bridge_session_token=object(),
        population=object.__new__(CompletedTrainingPopulation),
        epoch_trace=object.__new__(EpochUpdateTraceEvidence),
        seed=20_260_719,
        zero_based_epoch=0,
        optimizer_generation=313,
        runtime_digests=numeric_executor._RuntimeDigests(
            model_sha256="1" * 64,
            optimizer_sha256="2" * 64,
            rng_sha256="3" * 64,
            rng_state=b"counterfeit",
        ),
        previous_history_barrier=None,
        one_shot_token=object(),
        another_training_epoch=True,
    )
    token = object()
    state = evaluator._HandoffEvaluationState(
        token=token,
        process_id=os.getpid(),
        registration=registration,
        validation_inputs=validation_inputs,
        snapshot=snapshot,
        phase="EVALUATING",
        result=None,
    )
    guard = evaluator._HandoffEvaluationGuard(
        token=token,
        process_id=state.process_id,
        registration=registration,
        validation_inputs=validation_inputs,
        snapshot=snapshot,
    )
    return handoff, state, guard


def _claim_counterfeit_handoff_authority() -> tuple[
    RegisteredExecutorEpochHandoff,
    evaluator._HandoffEvaluationState,
    evaluator._HandoffEvaluationGuard,
]:
    handoff, candidate, _ = _counterfeit_handoff_authority()
    state = evaluator._claim_handoff_evaluation(
        handoff,
        candidate.registration,
        candidate.validation_inputs,
        candidate.snapshot,
    )
    return handoff, state, evaluator._HANDOFF_EVALUATION_GUARDS[handoff]


def _counterfeit_result_authority() -> tuple[
    evaluator.RegisteredEvaluatedEpoch,
    evaluator._RegisteredEvaluatedEpochState,
    evaluator._RegisteredEvaluatedEpochGuard,
]:
    handoff, handoff_state, _ = _counterfeit_handoff_authority()
    result = object.__new__(evaluator.RegisteredEvaluatedEpoch)
    token = object()
    validation_evidence = object.__new__(RegisteredValidationEvidence)
    model_tensors = object.__new__(RegisteredModelTensorEvidence)
    state = evaluator._RegisteredEvaluatedEpochState(
        token=token,
        route_marker=evaluator._REGISTERED_ROUTE_MARKER,
        process_id=os.getpid(),
        registration=handoff_state.registration,
        handoff=handoff,
        handoff_snapshot=handoff_state.snapshot,
        validation_inputs=handoff_state.validation_inputs,
        validation_evidence=validation_evidence,
        model_tensors=model_tensors,
        seed=20_260_719,
        zero_based_epoch=0,
        optimizer_generation=313,
        batch_sizes=(1,),
        registration_head_commit="a" * 40,
        registration_sha256="b" * 64,
        source_bundle_sha256="c" * 64,
        validation_inputs_sha256="d" * 64,
        validation_predictions_sha256="e" * 64,
        model_tensor_sha256="f" * 64,
        optimizer_sha256="1" * 64,
        torch_rng_sha256="2" * 64,
        authority_sha256="3" * 64,
    )
    guard = evaluator._RegisteredEvaluatedEpochGuard(
        token=token,
        route_marker=evaluator._REGISTERED_ROUTE_MARKER,
        process_id=state.process_id,
        registration=state.registration,
        handoff=handoff,
        handoff_snapshot=state.handoff_snapshot,
        validation_inputs=state.validation_inputs,
        validation_evidence=validation_evidence,
        model_tensors=model_tensors,
        scalar_frame=evaluator._result_scalar_frame(state),
    )
    object.__setattr__(result, "_seed", state.seed)
    object.__setattr__(result, "_zero_based_epoch", state.zero_based_epoch)
    object.__setattr__(
        result, "_validation_inputs_sha256", state.validation_inputs_sha256
    )
    object.__setattr__(
        result,
        "_validation_predictions_sha256",
        state.validation_predictions_sha256,
    )
    object.__setattr__(result, "_model_tensor_sha256", state.model_tensor_sha256)
    object.__setattr__(result, "_authority_sha256", state.authority_sha256)
    object.__setattr__(result, "_route_marker", evaluator._REGISTERED_ROUTE_MARKER)
    return result, state, guard


def _issue_counterfeit_result_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    evaluator.RegisteredEvaluatedEpoch,
    evaluator._RegisteredEvaluatedEpochState,
    evaluator._RegisteredEvaluatedEpochGuard,
]:
    _, candidate, _ = _counterfeit_result_authority()
    monkeypatch.setattr(
        VerifiedRunRegistration,
        "head_commit",
        property(lambda _self: candidate.registration_head_commit),
    )
    monkeypatch.setattr(
        VerifiedRunRegistration,
        "registration_sha256",
        property(lambda _self: candidate.registration_sha256),
    )
    monkeypatch.setattr(
        VerifiedRunRegistration,
        "source_bundle_sha256",
        property(lambda _self: candidate.source_bundle_sha256),
    )
    monkeypatch.setattr(
        RegisteredValidationInputs,
        "validation_inputs_sha256",
        property(lambda _self: candidate.validation_inputs_sha256),
    )
    monkeypatch.setattr(
        RegisteredModelTensorEvidence,
        "sha256",
        property(lambda _self: candidate.model_tensor_sha256),
    )
    result = evaluator._issue_registered_evaluated_epoch(
        registration=candidate.registration,
        handoff=candidate.handoff,
        handoff_snapshot=candidate.handoff_snapshot,
        validation_inputs=candidate.validation_inputs,
        validation_evidence=candidate.validation_evidence,
        model_tensors=candidate.model_tensors,
        validation_predictions_sha256=candidate.validation_predictions_sha256,
        kernel_result=evaluator._KernelResult(
            example_count=1,
            batch_sizes=candidate.batch_sizes,
            frame_97_logits_payload=b"",
            model_tensor_sha256=candidate.model_tensor_sha256,
            torch_rng_sha256=candidate.torch_rng_sha256,
            model_training_after=True,
        ),
    )
    return result, evaluator._RESULTS[result], evaluator._RESULT_GUARDS[result]


def test_public_signature_and_frozen_production_layout_are_exact() -> None:
    for name in (
        "_build_independent_issuance_truth",
        "_record_handoff_issuance_truth",
        "_handoff_issuance_truth_phase",
        "_require_handoff_issuance_truth",
        "_transition_handoff_issuance_truth",
        "_record_result_issuance_truth",
        "_require_result_issuance_truth",
        "_fail_result_issuance_truth",
    ):
        assert not hasattr(evaluator, name)
    assert str(inspect.signature(evaluator.evaluate_registered_epoch)) == (
        "(registration: 'VerifiedRunRegistration', handoff: "
        "'RegisteredExecutorEpochHandoff', validation_inputs: "
        "'RegisteredValidationInputs') -> 'RegisteredEvaluatedEpoch'"
    )
    assert evaluator.VALIDATION_EXAMPLE_COUNT == 10_583
    assert evaluator.VALIDATION_BATCH_SIZE == 128
    assert evaluator.VALIDATION_BATCH_COUNT == 83
    assert evaluator.VALIDATION_FULL_BATCH_COUNT == 82
    assert evaluator.VALIDATION_LAST_BATCH_SIZE == 87
    assert 82 * 128 + 87 == 10_583


def test_source_routes_registered_evaluation_without_the_tiny_old_route() -> None:
    source = Path("src/falsewake/experiment_002_registered_evaluator.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    public_functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    }
    assert public_functions == {
        "evaluate_registered_epoch",
        "verify_registered_evaluated_epoch",
    }
    assert "_evaluate_unregistered_causal_model" not in source
    assert "evaluate_validation_layout_logits" in source
    assert "issue_registered_validation_evidence" in source
    assert "capture_registered_model_tensors" in source
    assert "verify_registered_model_tensors_match" in source
    assert "_registered_epoch_handoff_snapshot" in source
    assert "_registered_epoch_handoff_model" in source
    assert "_fail_registered_epoch_handoff" in source


def test_registered_result_is_opaque_and_counterfeits_fail_closed() -> None:
    with pytest.raises(TypeError, match="issuer-only"):
        evaluator.RegisteredEvaluatedEpoch()
    counterfeit = object.__new__(evaluator.RegisteredEvaluatedEpoch)
    with pytest.raises(
        evaluator.Experiment002RegisteredEvaluatorError,
        match="issuance truth",
    ):
        evaluator.verify_registered_evaluated_epoch(counterfeit)
    with pytest.raises(
        evaluator.Experiment002RegisteredEvaluatorError,
        match="terminally failed",
    ):
        evaluator._registered_evaluated_epoch_snapshot(counterfeit)


def test_counterfeit_handoff_cannot_reach_registration_or_data_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration = object.__new__(VerifiedRunRegistration)
    handoff = object.__new__(RegisteredExecutorEpochHandoff)
    validation_inputs = object.__new__(RegisteredValidationInputs)

    def forbidden(*args: object, **kwargs: object) -> Any:
        del args, kwargs
        raise AssertionError("downstream registered route was reached")

    monkeypatch.setattr(evaluator, "reverify_verified_run_registration", forbidden)
    monkeypatch.setattr(evaluator, "verify_registered_validation_inputs", forbidden)
    monkeypatch.setattr(evaluator, "evaluate_validation_layout_logits", forbidden)
    monkeypatch.setattr(evaluator, "issue_registered_validation_evidence", forbidden)
    monkeypatch.setattr(evaluator, "capture_registered_model_tensors", forbidden)

    with pytest.raises(Exception, match="issued|handoff|authority"):
        evaluator.evaluate_registered_epoch(
            registration,
            handoff,
            validation_inputs,
        )


def test_independent_truth_rejects_coherent_handoff_clone_store_rewrite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original, original_state, original_guard = _claim_counterfeit_handoff_authority()
    assert evaluator._HANDOFF_EVALUATIONS[original] is original_state
    assert evaluator._HANDOFF_EVALUATION_GUARDS[original] is original_guard
    clone, clone_state, clone_guard = _counterfeit_handoff_authority()
    evaluator._HANDOFF_EVALUATIONS[clone] = clone_state
    evaluator._HANDOFF_EVALUATION_GUARDS[clone] = clone_guard
    evaluator._HANDOFF_EVALUATION_LIFECYCLES[clone] = (
        evaluator._HandoffEvaluationLifecycle(
            token=clone_state.token,
            process_id=clone_state.process_id,
            phase="EVALUATING",
            result=None,
        )
    )
    evaluator._ISSUED_HANDOFF_EVALUATIONS.add(clone)

    monkeypatch.setattr(
        evaluator,
        "_require_handoff_issuance_truth",
        lambda *args, **kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        evaluator,
        "_record_handoff_issuance_truth",
        lambda *args, **kwargs: None,
        raising=False,
    )
    evaluator._record_handoff_issuance_truth(clone, clone_state, clone_guard)
    with pytest.raises(
        evaluator.Experiment002RegisteredEvaluatorError,
        match="issuance truth",
    ):
        evaluator._require_handoff_evaluation_authority(clone, clone_state)


def test_independent_truth_rejects_in_place_handoff_snapshot_rewrite() -> None:
    handoff, state, _ = _claim_counterfeit_handoff_authority()
    object.__setattr__(state.snapshot, "zero_based_epoch", 1)

    with pytest.raises(
        evaluator.Experiment002RegisteredEvaluatorError,
        match="issuance truth",
    ):
        evaluator._require_handoff_evaluation_authority(handoff, state)


def test_failed_handoff_truth_cannot_be_rolled_back_with_every_public_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handoff, state, _ = _claim_counterfeit_handoff_authority()
    evaluator._terminal_fail_handoff_evaluation(handoff, state)

    evaluator._FAILED_HANDOFF_EVALUATIONS.discard(handoff)
    state.phase = "EVALUATING"
    evaluator._HANDOFF_EVALUATION_LIFECYCLES[handoff] = (
        evaluator._HandoffEvaluationLifecycle(
            token=state.token,
            process_id=state.process_id,
            phase="EVALUATING",
            result=None,
        )
    )
    monkeypatch.setattr(
        evaluator,
        "_handoff_issuance_truth_phase",
        lambda candidate: None,
        raising=False,
    )
    monkeypatch.setattr(
        evaluator,
        "_record_handoff_issuance_truth",
        lambda *args, **kwargs: None,
        raising=False,
    )
    with pytest.raises(
        evaluator.Experiment002RegisteredEvaluatorError,
        match="terminally failed",
    ):
        evaluator._claim_handoff_evaluation(
            handoff,
            state.registration,
            state.validation_inputs,
            state.snapshot,
        )


def test_independent_truth_rejects_coherent_result_clone_store_rewrite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original, original_state, original_guard = _issue_counterfeit_result_authority(
        monkeypatch
    )
    assert evaluator._RESULTS[original] is original_state
    assert evaluator._RESULT_GUARDS[original] is original_guard
    clone, clone_state, clone_guard = _counterfeit_result_authority()
    evaluator._RESULTS[clone] = clone_state
    evaluator._RESULT_GUARDS[clone] = clone_guard
    evaluator._RESULT_LIFECYCLES[clone] = evaluator._RegisteredEvaluatedEpochLifecycle(
        token=clone_state.token,
        process_id=clone_state.process_id,
        phase="COMPLETE",
    )
    evaluator._ISSUED_RESULTS.add(clone)
    monkeypatch.setattr(
        evaluator,
        "_require_result_issuance_truth",
        lambda *args, **kwargs: "COMPLETE",
        raising=False,
    )
    monkeypatch.setattr(
        evaluator,
        "_record_result_issuance_truth",
        lambda *args, **kwargs: None,
        raising=False,
    )
    evaluator._record_result_issuance_truth(clone, clone_state, clone_guard)

    with pytest.raises(
        evaluator.Experiment002RegisteredEvaluatorError,
        match="issuance truth",
    ):
        evaluator._issued_result_state(clone)


def test_independent_truth_rejects_in_place_result_and_guard_rewrite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, state, guard = _issue_counterfeit_result_authority(monkeypatch)
    evaluator._RESULTS[result] = state
    evaluator._RESULT_GUARDS[result] = guard
    evaluator._RESULT_LIFECYCLES[result] = evaluator._RegisteredEvaluatedEpochLifecycle(
        token=state.token,
        process_id=state.process_id,
        phase="COMPLETE",
    )
    evaluator._ISSUED_RESULTS.add(result)

    changed = "4" * 64
    object.__setattr__(state, "validation_predictions_sha256", changed)
    object.__setattr__(result, "_validation_predictions_sha256", changed)
    object.__setattr__(guard, "scalar_frame", evaluator._result_scalar_frame(state))
    with pytest.raises(
        evaluator.Experiment002RegisteredEvaluatorError,
        match="issuance truth",
    ):
        evaluator._issued_result_state(result)


def test_failed_result_truth_survives_failed_set_and_lifecycle_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, state, guard = _issue_counterfeit_result_authority(monkeypatch)
    evaluator._RESULTS[result] = state
    evaluator._RESULT_GUARDS[result] = guard
    evaluator._RESULT_LIFECYCLES[result] = evaluator._RegisteredEvaluatedEpochLifecycle(
        token=state.token,
        process_id=state.process_id,
        phase="COMPLETE",
    )
    evaluator._ISSUED_RESULTS.add(result)
    evaluator._terminal_fail_result(result, state)

    evaluator._FAILED_RESULTS.discard(result)
    evaluator._RESULT_LIFECYCLES[result] = evaluator._RegisteredEvaluatedEpochLifecycle(
        token=state.token,
        process_id=state.process_id,
        phase="COMPLETE",
    )
    monkeypatch.setattr(
        evaluator,
        "_require_result_issuance_truth",
        lambda *args, **kwargs: "COMPLETE",
        raising=False,
    )
    monkeypatch.setattr(
        evaluator,
        "_fail_result_issuance_truth",
        lambda *args, **kwargs: None,
        raising=False,
    )
    with pytest.raises(
        evaluator.Experiment002RegisteredEvaluatorError,
        match="terminally failed",
    ):
        evaluator._issued_result_state(result)


def test_public_route_requires_exact_capability_types() -> None:
    registration = object.__new__(VerifiedRunRegistration)
    handoff = object.__new__(RegisteredExecutorEpochHandoff)
    validation_inputs = object.__new__(RegisteredValidationInputs)
    with pytest.raises(TypeError, match="registration"):
        evaluator.evaluate_registered_epoch(
            cast(VerifiedRunRegistration, object()),
            handoff,
            validation_inputs,
        )
    with pytest.raises(TypeError, match="handoff"):
        evaluator.evaluate_registered_epoch(
            registration,
            cast(RegisteredExecutorEpochHandoff, object()),
            validation_inputs,
        )
    with pytest.raises(TypeError, match="validation_inputs"):
        evaluator.evaluate_registered_epoch(
            registration,
            handoff,
            cast(RegisteredValidationInputs, object()),
        )


def test_private_kernel_matches_independent_direct_batch_oracle() -> None:
    model = _model()
    oracle = copy.deepcopy(model)
    inputs = _inputs(5)
    rng_before = torch.get_rng_state().numpy().tobytes(order="C")
    expected = _manual_logits(oracle, inputs, batch_size=2)
    assert torch.get_rng_state().numpy().tobytes(order="C") == rng_before

    result = _run_synthetic(model, inputs, batch_size=2)

    assert result.example_count == 5
    assert result.batch_sizes == (2, 2, 1)
    assert result.model_training_after is True
    assert all(module.training for module in model.modules())
    np.testing.assert_array_equal(result._frame_97_logits(), expected)


@pytest.mark.parametrize(
    ("another_training_epoch", "expected_training"),
    ((True, True), (False, False)),
)
def test_private_kernel_restores_the_exact_requested_model_mode(
    another_training_epoch: bool,
    expected_training: bool,
) -> None:
    model = _model()
    result = _run_synthetic(
        model,
        _inputs(2),
        batch_size=1,
        another_training_epoch=another_training_epoch,
    )
    assert result.model_training_after is expected_training
    assert all(module.training is expected_training for module in model.modules())


def test_private_kernel_preserves_inputs_model_gradients_and_rng() -> None:
    model = _model()
    parameters = tuple(model.parameters())
    parameters[0].grad = torch.ones_like(parameters[0])
    parameters[-1].grad = torch.full_like(parameters[-1], 2.0)
    inputs = _inputs(4)
    input_payload = inputs.tobytes(order="C")
    model_digest = unregistered_kernel._stable_model_tensor_sha256(model)
    gradients = unregistered_kernel._gradient_snapshot(model)
    rng = torch.get_rng_state().numpy().tobytes(order="C")

    result = _run_synthetic(model, inputs, batch_size=3)

    assert inputs.tobytes(order="C") == input_payload
    assert unregistered_kernel._stable_model_tensor_sha256(model) == model_digest
    assert unregistered_kernel._gradient_snapshot(model) == gradients
    assert torch.get_rng_state().numpy().tobytes(order="C") == rng
    assert result.model_tensor_sha256 == model_digest


def test_private_kernel_uses_fresh_zero_state_for_every_direct_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_frames: list[tuple[int, ...]] = []
    observed_batches: list[int] = []
    actual = CausalKWS.forward_stream

    def recording(
        self: CausalKWS,
        features: Tensor,
        state: CausalKWSState,
    ) -> tuple[Tensor, CausalKWSState]:
        observed_frames.append(
            tuple(int(value) for value in state.frames_seen.tolist())
        )
        observed_batches.append(int(features.shape[0]))
        return actual(self, features, state)

    monkeypatch.setattr(CausalKWS, "forward_stream", recording)
    _run_synthetic(_model(), _inputs(7), batch_size=3)
    assert observed_batches == [3, 3, 1]
    assert observed_frames == [(0, 0, 0), (0, 0, 0), (0,)]


def test_private_kernel_cannot_call_any_registered_issuer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> Any:
        del args, kwargs
        raise AssertionError("private kernel called a registered issuer")

    for name in (
        "reverify_verified_run_registration",
        "verify_registered_validation_inputs",
        "evaluate_validation_layout_logits",
        "issue_registered_validation_evidence",
        "capture_registered_model_tensors",
        "verify_registered_model_tensors_match",
        "_registered_epoch_handoff_snapshot",
        "_registered_epoch_handoff_model",
    ):
        monkeypatch.setattr(evaluator, name, forbidden)

    result = _run_synthetic(_model(), _inputs(2), batch_size=1)
    assert result.example_count == 2
    assert result.batch_sizes == (1, 1)


def test_private_layout_is_permanently_capped_below_registered_population() -> None:
    assert evaluator._MAX_SYNTHETIC_EXAMPLES == 8
    with pytest.raises(
        evaluator.Experiment002RegisteredEvaluatorError,
        match="exceeds eight",
    ):
        evaluator._SyntheticEvaluationLayout(9, 1)
    with pytest.raises(TypeError, match="integer"):
        evaluator._SyntheticEvaluationLayout(cast(int, True), 1)
    with pytest.raises(
        evaluator.Experiment002RegisteredEvaluatorError,
        match="positive",
    ):
        evaluator._SyntheticEvaluationLayout(1, 0)


def test_kernel_result_exposes_only_defensive_private_logit_copies() -> None:
    result = _run_synthetic(_model(), _inputs(2), batch_size=2)
    first = result._frame_97_logits()
    second = result._frame_97_logits()
    assert first is not second
    assert not first.flags.writeable
    assert first.flags.owndata and first.flags.c_contiguous
    first.setflags(write=True)
    first[0, 0] = np.float32(123.0)
    assert not np.array_equal(first, second)


def test_kernel_detects_input_mutation_and_restores_training_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual = CausalKWS.forward_stream

    def mutating(
        self: CausalKWS,
        features: Tensor,
        state: CausalKWSState,
    ) -> tuple[Tensor, CausalKWSState]:
        features[0, 0, 0] = features[0, 0, 0] + 1
        return actual(self, features, state)

    monkeypatch.setattr(CausalKWS, "forward_stream", mutating)
    model = _model()
    with pytest.raises(
        evaluator.Experiment002RegisteredEvaluatorError,
        match="input bytes changed",
    ):
        _run_synthetic(model, _inputs(1), batch_size=1)
    assert all(module.training for module in model.modules())


def test_registered_result_declares_no_raw_model_or_logit_accessor() -> None:
    forbidden = {
        "model",
        "logits",
        "frame_97_logits",
        "frame_97_logits_payload",
    }
    assert forbidden.isdisjoint(vars(evaluator.RegisteredEvaluatedEpoch))
    snapshot_fields = set(evaluator._RegisteredEvaluatedEpochSnapshot.__slots__)
    assert forbidden.isdisjoint(snapshot_fields)


def test_source_has_no_filesystem_subprocess_or_artifact_publication_route() -> None:
    source = Path("src/falsewake/experiment_002_registered_evaluator.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported_roots.update(
                alias.name.split(".", maxsplit=1)[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".", maxsplit=1)[0])
    assert imported_roots.isdisjoint(
        {"json", "pathlib", "safetensors", "subprocess", "tempfile"}
    )
    assert "open(" not in source
    assert "write_bytes" not in source
    assert "torch.save" not in source


def test_counterfeit_result_cannot_be_copied_or_serialized_after_injection() -> None:
    counterfeit = object.__new__(evaluator.RegisteredEvaluatedEpoch)
    with pytest.raises(TypeError, match="copied"):
        copy.copy(counterfeit)
    with pytest.raises(TypeError, match="copied"):
        copy.deepcopy(counterfeit)
    with pytest.raises(TypeError, match="serialized"):
        pickle.dumps(counterfeit)
