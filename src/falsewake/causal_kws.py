"""Stateful causal neural network for Experiment 002 keyword spotting."""

from __future__ import annotations

from typing import NamedTuple, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

INPUT_MEL_BINS = 40
ENCODER_CHANNELS = 48
OUTPUT_CLASSES = 12
DEPTHWISE_KERNEL_SIZE = 3
BLOCK_DILATIONS = (1, 2, 4, 8, 1, 2, 4, 8)
POOL_FRAMES = 38
POOL_HISTORY_FRAMES = POOL_FRAMES - 1
TCN_RECEPTIVE_FIELD_FRAMES = 1 + (DEPTHWISE_KERNEL_SIZE - 1) * sum(BLOCK_DILATIONS)
MODEL_RECEPTIVE_FIELD_FRAMES = TCN_RECEPTIVE_FIELD_FRAMES + POOL_HISTORY_FRAMES
STREAMING_FLOAT_STATE_VALUES = ENCODER_CHANNELS * (
    (DEPTHWISE_KERNEL_SIZE - 1) * sum(BLOCK_DILATIONS) + POOL_HISTORY_FRAMES
)
TRAINABLE_PARAMETER_COUNT = 23_724


class CausalKWSState(NamedTuple):
    """Explicit histories required to continue a streamed utterance."""

    dw_state_0: Tensor
    dw_state_1: Tensor
    dw_state_2: Tensor
    dw_state_3: Tensor
    dw_state_4: Tensor
    dw_state_5: Tensor
    dw_state_6: Tensor
    dw_state_7: Tensor
    pool_state: Tensor
    frames_seen: Tensor

    @property
    def depthwise_states(self) -> tuple[Tensor, ...]:
        """Return depthwise histories in encoder block order."""

        return (
            self.dw_state_0,
            self.dw_state_1,
            self.dw_state_2,
            self.dw_state_3,
            self.dw_state_4,
            self.dw_state_5,
            self.dw_state_6,
            self.dw_state_7,
        )


class PerFrameLayerNorm(nn.Module):
    """Apply channel LayerNorm independently at every time frame."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-5, elementwise_affine=True)

    def forward(self, features: Tensor) -> Tensor:
        normalized = self.norm(features.transpose(1, 2)).transpose(1, 2)
        return cast(Tensor, normalized)


class CausalDepthwiseSeparableBlock(nn.Module):
    """One residual causal depthwise-separable temporal block."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.channels = channels
        self.dilation = dilation
        self.history_frames = dilation * (DEPTHWISE_KERNEL_SIZE - 1)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=DEPTHWISE_KERNEL_SIZE,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.depthwise_norm = PerFrameLayerNorm(channels)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.pointwise_norm = PerFrameLayerNorm(channels)
        self.activation = nn.ReLU()

    def forward(self, features: Tensor, history: Tensor) -> tuple[Tensor, Tensor]:
        combined = torch.cat((history, features), dim=2)
        next_history = combined[:, :, -self.history_frames :].contiguous()
        transformed = self.depthwise(combined)
        transformed = self.activation(self.depthwise_norm(transformed))
        transformed = self.pointwise_norm(self.pointwise(transformed))
        return self.activation(features + transformed), next_history


class CausalKWS(nn.Module):
    """The 23,724-parameter stateful CausalDS-TCN-24k model."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv1d(
            INPUT_MEL_BINS, ENCODER_CHANNELS, kernel_size=1, bias=False
        )
        self.stem_norm = PerFrameLayerNorm(ENCODER_CHANNELS)
        self.stem_activation = nn.ReLU()
        self.blocks = nn.ModuleList(
            CausalDepthwiseSeparableBlock(ENCODER_CHANNELS, dilation)
            for dilation in BLOCK_DILATIONS
        )
        self.dropout = nn.Dropout(p=0.10)
        self.classifier = nn.Linear(ENCODER_CHANNELS, OUTPUT_CLASSES, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Apply the registered Experiment 002 initialization."""

        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.LayerNorm):
                if module.elementwise_affine:
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> CausalKWSState:
        """Create zero histories for a new batch of utterances."""

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        reference = self.stem.weight
        self._validate_model_parameters(reference.device)
        resolved_device = reference.device if device is None else torch.device(device)
        resolved_dtype = reference.dtype if dtype is None else dtype
        if resolved_dtype != torch.float32:
            raise ValueError("streaming feature state must use float32")
        if resolved_device != reference.device:
            raise ValueError("streaming state device must match model weights")
        depthwise_states = tuple(
            torch.zeros(
                batch_size,
                ENCODER_CHANNELS,
                (DEPTHWISE_KERNEL_SIZE - 1) * dilation,
                device=resolved_device,
                dtype=resolved_dtype,
            )
            for dilation in BLOCK_DILATIONS
        )
        pool_state = torch.zeros(
            batch_size,
            ENCODER_CHANNELS,
            POOL_HISTORY_FRAMES,
            device=resolved_device,
            dtype=resolved_dtype,
        )
        frames_seen = torch.zeros(batch_size, device=resolved_device, dtype=torch.int64)
        return self._pack_state(depthwise_states, pool_state, frames_seen)

    def forward(self, features: Tensor) -> Tensor:
        """Score a full feature sequence with reset utterance state."""

        self._validate_features(features)
        state = self.initial_state(
            features.shape[0], device=features.device, dtype=features.dtype
        )
        logits, _ = self.forward_stream(features, state)
        return logits

    def forward_stream(
        self, features: Tensor, state: CausalKWSState
    ) -> tuple[Tensor, CausalKWSState]:
        """Score every frame in a chunk and return state for the next chunk."""

        self._validate_stream_inputs(features, state)
        encoded = self.stem_activation(self.stem_norm(self.stem(features)))
        next_depthwise: list[Tensor] = []
        for block, history in zip(self.blocks, state.depthwise_states, strict=True):
            encoded, next_history = block(encoded, history)
            next_depthwise.append(next_history)

        pool_input = torch.cat((state.pool_state, encoded), dim=2)
        pooled = F.avg_pool1d(pool_input, kernel_size=POOL_FRAMES, stride=1)
        next_pool_state = pool_input[:, :, -POOL_HISTORY_FRAMES:].contiguous()
        logits = self.classifier(self.dropout(pooled.transpose(1, 2)))
        chunk_frames = torch.full_like(state.frames_seen, features.shape[2])
        next_frames_seen = torch.clamp(
            state.frames_seen + chunk_frames, max=MODEL_RECEPTIVE_FIELD_FRAMES
        )
        next_state = self._pack_state(
            tuple(next_depthwise), next_pool_state, next_frames_seen
        )
        return logits, next_state

    @staticmethod
    def _pack_state(
        depthwise_states: tuple[Tensor, ...],
        pool_state: Tensor,
        frames_seen: Tensor,
    ) -> CausalKWSState:
        if len(depthwise_states) != len(BLOCK_DILATIONS):
            raise ValueError("one depthwise history is required per encoder block")
        return CausalKWSState(
            dw_state_0=depthwise_states[0],
            dw_state_1=depthwise_states[1],
            dw_state_2=depthwise_states[2],
            dw_state_3=depthwise_states[3],
            dw_state_4=depthwise_states[4],
            dw_state_5=depthwise_states[5],
            dw_state_6=depthwise_states[6],
            dw_state_7=depthwise_states[7],
            pool_state=pool_state,
            frames_seen=frames_seen,
        )

    def _validate_stream_inputs(self, features: Tensor, state: CausalKWSState) -> None:
        self._validate_features(features)

        batch_size = features.shape[0]
        for index, (history, dilation) in enumerate(
            zip(state.depthwise_states, BLOCK_DILATIONS, strict=True)
        ):
            expected = (
                batch_size,
                ENCODER_CHANNELS,
                (DEPTHWISE_KERNEL_SIZE - 1) * dilation,
            )
            if tuple(history.shape) != expected:
                raise ValueError(f"dw_state_{index} must have shape {expected}")
            if history.dtype != torch.float32:
                raise ValueError(f"dw_state_{index} must use float32")
            if history.device != features.device:
                raise ValueError(f"dw_state_{index} must match the feature device")
        pool_expected = (batch_size, ENCODER_CHANNELS, POOL_HISTORY_FRAMES)
        if tuple(state.pool_state.shape) != pool_expected:
            raise ValueError(f"pool_state must have shape {pool_expected}")
        if state.pool_state.dtype != torch.float32:
            raise ValueError("pool_state must use float32")
        if state.pool_state.device != features.device:
            raise ValueError("pool_state must match the feature device")
        if tuple(state.frames_seen.shape) != (batch_size,):
            raise ValueError("frames_seen must have shape [batch]")
        if state.frames_seen.device != features.device:
            raise ValueError("frames_seen must match the feature device")
        if state.frames_seen.dtype != torch.int64:
            raise ValueError("frames_seen must use int64")
        # A Python exception based on tensor data cannot be represented in a
        # dynamic Dynamo/ONNX graph. Exported callers retain the same [0, 98]
        # input contract; eager callers receive the explicit diagnostic here.
        if not torch.compiler.is_compiling() and bool(
            torch.any(
                (state.frames_seen < 0)
                | (state.frames_seen > MODEL_RECEPTIVE_FIELD_FRAMES)
            )
        ):
            raise ValueError("frames_seen values must be within [0, 98]")

    def _validate_features(self, features: Tensor) -> None:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, 40, frames]")
        if features.shape[0] < 1 or features.shape[2] < 1:
            raise ValueError("features must contain a batch and at least one frame")
        if features.shape[1] != INPUT_MEL_BINS:
            raise ValueError("features must contain exactly 40 mel channels")
        if features.dtype != torch.float32:
            raise ValueError("features must use the registered float32 dtype")
        self._validate_model_parameters(features.device)

    def _validate_model_parameters(self, device: torch.device) -> None:
        for name, parameter in self.named_parameters():
            if parameter.dtype != torch.float32:
                raise ValueError(
                    f"registered model weights must use float32 ({name} does not)"
                )
            if parameter.device != device:
                raise ValueError(
                    "registered model weights must match the input/state device "
                    f"({name} does not)"
                )
