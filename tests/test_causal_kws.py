from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, cast

import pytest
import torch
from torch import Tensor, nn

from falsewake.causal_kws import (
    BLOCK_DILATIONS,
    ENCODER_CHANNELS,
    INPUT_MEL_BINS,
    MODEL_RECEPTIVE_FIELD_FRAMES,
    OUTPUT_CLASSES,
    POOL_HISTORY_FRAMES,
    STREAMING_FLOAT_STATE_VALUES,
    TCN_RECEPTIVE_FIELD_FRAMES,
    TRAINABLE_PARAMETER_COUNT,
    CausalDepthwiseSeparableBlock,
    CausalKWS,
    CausalKWSState,
)


def _stream(
    model: CausalKWS, features: Tensor, chunk_sizes: Iterable[int]
) -> tuple[Tensor, CausalKWSState]:
    state = model.initial_state(
        features.shape[0], device=features.device, dtype=features.dtype
    )
    logits: list[Tensor] = []
    offset = 0
    for size in chunk_sizes:
        chunk = features[:, :, offset : offset + size]
        assert chunk.shape[2] == size
        chunk_logits, state = model.forward_stream(chunk, state)
        logits.append(chunk_logits)
        offset += size
    assert offset == features.shape[2]
    return torch.cat(logits, dim=1), state


def _assert_state_close(left: CausalKWSState, right: CausalKWSState) -> None:
    for left_history, right_history in zip(
        left.depthwise_states, right.depthwise_states, strict=True
    ):
        torch.testing.assert_close(left_history, right_history, rtol=1e-5, atol=1e-6)
    # Pooling reuses float32 accumulations across different chunk shapes. CPU
    # kernels may differ by a few ULPs here even when the causal logits agree.
    torch.testing.assert_close(left.pool_state, right.pool_state, rtol=1e-4, atol=3e-6)
    assert torch.equal(left.frames_seen, right.frames_seen)


def _replace_state(
    state: CausalKWSState,
    *,
    dw_state_0: Tensor | None = None,
    pool_state: Tensor | None = None,
    frames_seen: Tensor | None = None,
) -> CausalKWSState:
    return CausalKWSState(
        dw_state_0=state.dw_state_0 if dw_state_0 is None else dw_state_0,
        dw_state_1=state.dw_state_1,
        dw_state_2=state.dw_state_2,
        dw_state_3=state.dw_state_3,
        dw_state_4=state.dw_state_4,
        dw_state_5=state.dw_state_5,
        dw_state_6=state.dw_state_6,
        dw_state_7=state.dw_state_7,
        pool_state=state.pool_state if pool_state is None else pool_state,
        frames_seen=state.frames_seen if frames_seen is None else frames_seen,
    )


def test_registered_architecture_and_parameter_count() -> None:
    model = CausalKWS()

    assert sum(parameter.numel() for parameter in model.parameters()) == 23_724
    assert TRAINABLE_PARAMETER_COUNT == 23_724
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert model.stem.in_channels == INPUT_MEL_BINS
    assert model.stem.out_channels == ENCODER_CHANNELS
    assert model.stem.kernel_size == (1,)
    assert model.stem.bias is None
    assert len(model.blocks) == 8
    blocks = [cast(CausalDepthwiseSeparableBlock, block) for block in model.blocks]
    assert tuple(block.dilation for block in blocks) == BLOCK_DILATIONS
    for block in blocks:
        assert block.depthwise.kernel_size == (3,)
        assert block.depthwise.groups == ENCODER_CHANNELS
        assert block.depthwise.bias is None
        assert block.pointwise.kernel_size == (1,)
        assert block.pointwise.bias is None
    layer_norms = [
        module for module in model.modules() if isinstance(module, nn.LayerNorm)
    ]
    assert len(layer_norms) == 17
    assert not any(isinstance(module, nn.BatchNorm1d) for module in model.modules())
    assert model.dropout.p == pytest.approx(0.10)
    assert model.classifier.in_features == ENCODER_CHANNELS
    assert model.classifier.out_features == OUTPUT_CLASSES
    assert TCN_RECEPTIVE_FIELD_FRAMES == 61
    assert MODEL_RECEPTIVE_FIELD_FRAMES == 98
    assert STREAMING_FLOAT_STATE_VALUES == 4_656


def test_initialization_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    kaiming_calls: list[tuple[str, str]] = []
    xavier_calls = 0
    original_kaiming = nn.init.kaiming_normal_
    original_xavier = nn.init.xavier_uniform_

    def record_kaiming(
        tensor: Tensor,
        a: float = 0,
        mode: Literal["fan_in", "fan_out"] = "fan_in",
        nonlinearity: Literal["relu", "leaky_relu"] = "leaky_relu",
        generator: torch.Generator | None = None,
    ) -> Tensor:
        kaiming_calls.append((mode, nonlinearity))
        return original_kaiming(
            tensor,
            a=a,
            mode=mode,
            nonlinearity=nonlinearity,
            generator=generator,
        )

    def record_xavier(
        tensor: Tensor,
        gain: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        nonlocal xavier_calls
        xavier_calls += 1
        return original_xavier(tensor, gain=gain, generator=generator)

    monkeypatch.setattr(nn.init, "kaiming_normal_", record_kaiming)
    monkeypatch.setattr(nn.init, "xavier_uniform_", record_xavier)
    model = CausalKWS()

    assert kaiming_calls == [("fan_out", "relu")] * 17
    assert xavier_calls == 1
    for layer_norm in (
        module for module in model.modules() if isinstance(module, nn.LayerNorm)
    ):
        assert layer_norm.weight is not None
        assert layer_norm.bias is not None
        assert torch.equal(layer_norm.weight, torch.ones_like(layer_norm.weight))
        assert torch.equal(layer_norm.bias, torch.zeros_like(layer_norm.bias))
    assert model.classifier.bias is not None
    assert torch.equal(model.classifier.bias, torch.zeros_like(model.classifier.bias))


def test_state_shapes_and_saturated_frames_seen() -> None:
    model = CausalKWS().eval()
    state = model.initial_state(2)

    assert tuple(history.shape[2] for history in state.depthwise_states) == (
        2,
        4,
        8,
        16,
        2,
        4,
        8,
        16,
    )
    assert all(
        history.shape[:2] == (2, ENCODER_CHANNELS) for history in state.depthwise_states
    )
    assert state.pool_state.shape == (2, ENCODER_CHANNELS, POOL_HISTORY_FRAMES)
    assert state.frames_seen.shape == (2,)
    assert state.frames_seen.dtype == torch.int64
    assert torch.equal(state.frames_seen, torch.zeros(2, dtype=torch.int64))
    assert (
        sum(history.numel() for history in state.depthwise_states)
        + state.pool_state.numel()
        == 2 * STREAMING_FLOAT_STATE_VALUES
    )

    first_logits, state = model.forward_stream(torch.randn(2, 40, 71), state)
    second_logits, state = model.forward_stream(torch.randn(2, 40, 60), state)
    assert first_logits.shape == (2, 71, OUTPUT_CLASSES)
    assert second_logits.shape == (2, 60, OUTPUT_CLASSES)
    assert torch.equal(state.frames_seen, torch.full((2,), 98, dtype=torch.int64))
    for history in (*state.depthwise_states, state.pool_state):
        assert (
            history.untyped_storage().nbytes()
            == history.numel() * history.element_size()
        )


def test_full_one_frame_and_irregular_streaming_are_equivalent() -> None:
    torch.manual_seed(20_260_719)
    model = CausalKWS().eval()
    features = torch.randn(2, INPUT_MEL_BINS, 113)

    with torch.no_grad():
        full_logits, full_state = _stream(model, features, [113])
        one_frame_logits, one_frame_state = _stream(model, features, [1] * 113)
        irregular_logits, irregular_state = _stream(
            model, features, [7, 1, 19, 3, 32, 2, 11, 38]
        )
        forward_logits = model(features)

    torch.testing.assert_close(full_logits, forward_logits, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(full_logits, one_frame_logits, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(full_logits, irregular_logits, rtol=1e-5, atol=1e-6)
    _assert_state_close(full_state, one_frame_state)
    _assert_state_close(full_state, irregular_state)
    assert torch.equal(full_state.frames_seen, torch.full((2,), 98, dtype=torch.int64))


def test_logits_are_prefix_causal() -> None:
    torch.manual_seed(20_260_720)
    model = CausalKWS().eval()
    prefix_frames = 57
    features = torch.randn(1, INPUT_MEL_BINS, 91)
    changed_future = features.clone()
    changed_future[:, :, prefix_frames:] = (
        torch.randn_like(changed_future[:, :, prefix_frames:]) * 100.0
    )

    with torch.no_grad():
        original = model(features)
        modified = model(changed_future)
        prefix_only = model(features[:, :, :prefix_frames])

    torch.testing.assert_close(
        original[:, :prefix_frames], modified[:, :prefix_frames], rtol=0, atol=0
    )
    torch.testing.assert_close(
        original[:, :prefix_frames], prefix_only, rtol=1e-5, atol=1e-6
    )
    assert not torch.allclose(original[:, prefix_frames:], modified[:, prefix_frames:])


def test_dropout_is_disabled_in_eval_and_active_in_training() -> None:
    torch.manual_seed(20_260_721)
    model = CausalKWS()
    features = torch.randn(2, INPUT_MEL_BINS, 20)

    model.eval()
    with torch.no_grad():
        first_eval = model(features)
        second_eval = model(features)
    assert torch.equal(first_eval, second_eval)

    model.train()
    first_train = model(features)
    second_train = model(features)
    assert not torch.equal(first_train, second_train)


@pytest.mark.parametrize(
    ("features", "match"),
    [
        (torch.randn(2, 40), "shape"),
        (torch.randn(2, 39, 4), "40 mel"),
        (torch.randn(2, 40, 0), "at least one frame"),
        (torch.randn(2, 40, 4, dtype=torch.float64), "float32"),
        (torch.ones(2, 40, 4, dtype=torch.int64), "float32"),
    ],
)
def test_invalid_feature_contract_is_rejected(features: Tensor, match: str) -> None:
    model = CausalKWS()
    state = model.initial_state(2)
    with pytest.raises(ValueError, match=match):
        model.forward_stream(features, state)
    with pytest.raises(ValueError, match=match):
        model(features)


@pytest.mark.parametrize("batch_size", [0, -1])
def test_initial_state_rejects_nonpositive_batch(batch_size: int) -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        CausalKWS().initial_state(batch_size)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float16, torch.int64])
def test_initial_state_rejects_nonregistered_dtype(dtype: torch.dtype) -> None:
    with pytest.raises(ValueError, match="state must use float32"):
        CausalKWS().initial_state(1, dtype=dtype)


def test_model_weights_must_match_registered_float32_input() -> None:
    model = CausalKWS()
    state = model.initial_state(1)
    model.double()

    with pytest.raises(ValueError, match="model weights must use float32"):
        model.forward_stream(torch.randn(1, INPUT_MEL_BINS, 2), state)
    with pytest.raises(ValueError, match="model weights must use float32"):
        model.initial_state(1)

    model = CausalKWS()
    model.classifier.double()
    with pytest.raises(ValueError, match=r"classifier\.weight does not"):
        model.initial_state(1)


@pytest.mark.parametrize(
    ("replacement", "match"),
    [
        (torch.zeros(2, ENCODER_CHANNELS, 3), "dw_state_0 must have shape"),
        (
            torch.zeros(2, ENCODER_CHANNELS, 2, dtype=torch.float64),
            "dw_state_0 must use float32",
        ),
    ],
)
def test_bad_depthwise_state_is_rejected(replacement: Tensor, match: str) -> None:
    model = CausalKWS()
    features = torch.randn(2, INPUT_MEL_BINS, 4)
    state = _replace_state(model.initial_state(2), dw_state_0=replacement)

    with pytest.raises(ValueError, match=match):
        model.forward_stream(features, state)


@pytest.mark.parametrize(
    ("replacement", "match"),
    [
        (torch.zeros(2, ENCODER_CHANNELS, 36), "pool_state must have shape"),
        (
            torch.zeros(
                2,
                ENCODER_CHANNELS,
                POOL_HISTORY_FRAMES,
                dtype=torch.float64,
            ),
            "pool_state must use float32",
        ),
    ],
)
def test_bad_pool_state_is_rejected(replacement: Tensor, match: str) -> None:
    model = CausalKWS()
    features = torch.randn(2, INPUT_MEL_BINS, 4)
    state = _replace_state(model.initial_state(2), pool_state=replacement)

    with pytest.raises(ValueError, match=match):
        model.forward_stream(features, state)


@pytest.mark.parametrize(
    ("replacement", "match"),
    [
        (torch.zeros(2, 1, dtype=torch.int64), "shape"),
        (torch.zeros(2, dtype=torch.int32), "int64"),
        (torch.tensor([-1, 0], dtype=torch.int64), "within.*0, 98"),
        (torch.tensor([0, 99], dtype=torch.int64), "within.*0, 98"),
    ],
)
def test_bad_frames_seen_is_rejected(replacement: Tensor, match: str) -> None:
    model = CausalKWS()
    features = torch.randn(2, INPUT_MEL_BINS, 4)
    state = _replace_state(model.initial_state(2), frames_seen=replacement)

    with pytest.raises(ValueError, match=match):
        model.forward_stream(features, state)


def test_gradients_reach_every_trainable_parameter() -> None:
    torch.manual_seed(20_260_722)
    model = CausalKWS().train()
    features = torch.randn(2, INPUT_MEL_BINS, 12)

    model(features).square().mean().backward()

    assert all(
        parameter.grad is not None and torch.all(torch.isfinite(parameter.grad))
        for parameter in model.parameters()
    )
