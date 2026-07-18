from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from torch import Tensor

from falsewake.causal_kws import CausalKWS, CausalKWSState
from falsewake.experiment_002_frontend import (
    StreamingLogMelFrontend,
    complete_log_mel_frames,
)
from falsewake.features import FloatArray


def _waveform(sample_count: int, phase: float) -> FloatArray:
    samples = np.arange(sample_count, dtype=np.float64)
    waveform = 0.29 * np.sin(2.0 * np.pi * 613.0 * samples / 16_000.0 + phase)
    waveform += 0.17 * np.cos(
        2.0 * np.pi * 1_487.0 * samples / 16_000.0 - phase
    )
    return waveform.astype(np.float32)


def _partition(total: int, sizes: Iterable[int]) -> tuple[int, ...]:
    candidates = tuple(sizes)
    if not candidates or any(size <= 0 for size in candidates):
        raise ValueError("partition sizes must be positive")
    result: list[int] = []
    consumed = 0
    while consumed < total:
        size = min(candidates[len(result) % len(candidates)], total - consumed)
        result.append(size)
        consumed += size
    return tuple(result)


def _frames_to_nct(frames: FloatArray) -> Tensor:
    if frames.ndim != 2 or frames.shape[1] != 40:
        raise ValueError("frontend frames must have shape [T, 40]")
    if frames.dtype != np.dtype(np.float32):
        raise ValueError("frontend frames must use float32")
    nct = np.ascontiguousarray(frames.T[np.newaxis, :, :], dtype=np.float32)
    features = torch.from_numpy(nct)
    assert features.is_contiguous()
    return features


def _stream_utterance(
    frontend: StreamingLogMelFrontend,
    model: CausalKWS,
    state: CausalKWSState,
    waveform: FloatArray,
    chunk_sizes: Iterable[int],
) -> tuple[FloatArray, Tensor, CausalKWSState]:
    frame_chunks: list[FloatArray] = []
    logit_chunks: list[Tensor] = []
    offset = 0
    for size in chunk_sizes:
        chunk = waveform[offset : offset + size]
        if chunk.size != size:
            raise AssertionError("chunk partition exceeds the utterance")
        frames = frontend.push(chunk)
        if frames.shape[0]:
            features = _frames_to_nct(frames)
            logits, state = model.forward_stream(features, state)
            frame_chunks.append(frames)
            logit_chunks.append(logits)
        offset += size
    if offset != waveform.size:
        raise AssertionError("chunk partition does not consume the utterance")
    frontend.end_utterance()
    if not frame_chunks or not logit_chunks:
        raise AssertionError("utterance did not produce complete frontend frames")
    return np.concatenate(frame_chunks), torch.cat(logit_chunks, dim=1), state


def _assert_state_close(left: CausalKWSState, right: CausalKWSState) -> None:
    for left_history, right_history in zip(
        left.depthwise_states, right.depthwise_states, strict=True
    ):
        torch.testing.assert_close(left_history, right_history, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(left.pool_state, right.pool_state, rtol=1e-5, atol=1e-6)
    assert torch.equal(left.frames_seen, right.frames_seen)


def test_frontend_frames_convert_to_contiguous_nct_model_input() -> None:
    frames = complete_log_mel_frames(_waveform(16_000, phase=0.13))
    features = _frames_to_nct(frames)

    assert frames.shape == (98, 40)
    assert features.shape == (1, 40, 98)
    assert features.dtype == torch.float32

    torch.manual_seed(20_260_719)
    model = CausalKWS().eval()
    with torch.inference_mode():
        logits, state = model.forward_stream(features, model.initial_state(1))

    assert logits.shape == (1, 98, 12)
    assert logits.dtype == torch.float32
    assert torch.equal(state.frames_seen, torch.tensor([98], dtype=torch.int64))


def test_full_and_irregular_audio_streams_match_through_the_model() -> None:
    waveform = _waveform(20_080, phase=0.31)
    torch.manual_seed(20_260_720)
    model = CausalKWS().eval()

    with torch.inference_mode():
        full_frames, full_logits, full_state = _stream_utterance(
            StreamingLogMelFrontend(),
            model,
            model.initial_state(1),
            waveform,
            (waveform.size,),
        )
        irregular_frames, irregular_logits, irregular_state = _stream_utterance(
            StreamingLogMelFrontend(),
            model,
            model.initial_state(1),
            waveform,
            _partition(waveform.size, (1, 399, 17, 1_603, 80, 511, 2, 997)),
        )

    assert full_frames.shape == irregular_frames.shape == (124, 40)
    assert full_frames.tobytes() == irregular_frames.tobytes()
    assert full_logits.shape == irregular_logits.shape == (1, 124, 12)
    torch.testing.assert_close(full_logits, irregular_logits, rtol=1e-5, atol=1e-6)
    _assert_state_close(full_state, irregular_state)


def test_frontend_and_neural_state_reset_isolates_two_utterances() -> None:
    first = _waveform(16_000, phase=0.07)
    second = _waveform(16_000, phase=0.79)
    partition = _partition(second.size, (257, 1_031, 3, 509))
    torch.manual_seed(20_260_721)
    model = CausalKWS().eval()
    shared_frontend = StreamingLogMelFrontend()

    with torch.inference_mode():
        _, _, first_state = _stream_utterance(
            shared_frontend,
            model,
            model.initial_state(1),
            first,
            (first.size,),
        )
        reset_state = model.initial_state(1)
        second_frames, second_logits, second_state = _stream_utterance(
            shared_frontend,
            model,
            reset_state,
            second,
            partition,
        )
        fresh_frames, fresh_logits, fresh_state = _stream_utterance(
            StreamingLogMelFrontend(),
            model,
            model.initial_state(1),
            second,
            partition,
        )

    assert torch.equal(first_state.frames_seen, torch.tensor([98], dtype=torch.int64))
    assert torch.equal(reset_state.frames_seen, torch.zeros(1, dtype=torch.int64))
    assert shared_frontend.state.total_samples == 0
    assert shared_frontend.state.frames_emitted == 0
    assert second_frames.tobytes() == fresh_frames.tobytes()
    torch.testing.assert_close(second_logits, fresh_logits, rtol=0, atol=0)
    _assert_state_close(second_state, fresh_state)
