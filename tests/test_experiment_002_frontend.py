from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

import numpy as np
import pytest

from falsewake.experiment_002_frontend import (
    StreamingLogMelFrontend,
    complete_log_mel_frames,
)
from falsewake.features import (
    DEFAULT_FRONTEND,
    FloatArray,
    FrontendConfig,
    log_mel_spectrogram,
)


def _waveform(sample_count: int = 16_000, sample_rate: int = 16_000) -> FloatArray:
    samples = np.arange(sample_count, dtype=np.float64)
    waveform = 0.31 * np.sin(
        2.0 * np.pi * 731.0 * samples / sample_rate
    ) + 0.17 * np.cos(
        2.0 * np.pi * 1_913.0 * samples / sample_rate,
    )
    return waveform.astype(np.float32)


def _partition(total: int, sizes: Iterable[int]) -> tuple[int, ...]:
    result: list[int] = []
    consumed = 0
    candidates = tuple(sizes)
    if not candidates or any(size <= 0 for size in candidates):
        raise ValueError("partition sizes must be positive")
    while consumed < total:
        size = min(candidates[len(result) % len(candidates)], total - consumed)
        result.append(size)
        consumed += size
    return tuple(result)


def _stream(
    waveform: FloatArray,
    partition: tuple[int, ...],
    config: FrontendConfig = DEFAULT_FRONTEND,
) -> FloatArray:
    frontend = StreamingLogMelFrontend(config)
    rows: list[FloatArray] = []
    offset = 0
    for size in partition:
        chunk = waveform[offset : offset + size]
        emitted = frontend.push(chunk)
        if emitted.size:
            rows.append(emitted)
        offset += size
    assert offset == waveform.size
    frontend.end_utterance()
    if not rows:
        return np.empty((0, config.mel_bins), dtype=np.float32)
    return np.concatenate(rows)


def test_fixed_and_irregular_chunks_are_byte_identical() -> None:
    waveform = _waveform()
    one_chunk = complete_log_mel_frames(waveform)
    fixed = _stream(waveform, _partition(waveform.size, (160,)))
    irregular = _stream(
        waveform,
        _partition(waveform.size, (1, 399, 17, 1_603, 80, 511, 2, 997)),
    )

    assert one_chunk.shape == fixed.shape == irregular.shape == (98, 40)
    assert one_chunk.tobytes() == fixed.tobytes() == irregular.tobytes()

    single_frame_config = replace(DEFAULT_FRONTEND, clip_samples=400)
    public_single_frame = log_mel_spectrogram(waveform[:400], single_frame_config)
    assert one_chunk[:1].tobytes() == public_single_frame.tobytes()

    # The legacy clip frontend evaluates the same equations in a 98-row batch.
    # BLAS batching can change the last float32 bit, so parity is bounded to the
    # observed one-ULP-scale numeric tolerance rather than claimed byte equality.
    offline = log_mel_spectrogram(waveform)
    np.testing.assert_allclose(one_chunk, offline, rtol=0.0, atol=2.0e-6)


def test_clip_geometry_covers_15_920_samples_and_leaves_an_80_sample_tail() -> None:
    config = DEFAULT_FRONTEND
    coverage = (config.frame_count - 1) * config.hop_samples + config.window_samples
    assert config.frame_count == 98
    assert coverage == 15_920
    assert config.clip_samples - coverage == 80

    covered_impulse = np.zeros(config.clip_samples, dtype=np.float32)
    # The first sample beyond the preceding frame is unique to frame 98 and
    # receives a non-negligible Hann weight.  (The very last covered sample is
    # so close to the Hann endpoint that its power falls below the log floor.)
    covered_impulse[coverage - config.hop_samples] = 1.0
    covered = complete_log_mel_frames(covered_impulse)
    assert not np.array_equal(covered[-1], covered[-2])

    tail_changed = covered_impulse.copy()
    tail_changed[coverage:] = np.linspace(-1.0, 1.0, 80, dtype=np.float32)
    assert complete_log_mel_frames(tail_changed).tobytes() == covered.tobytes()

    frontend = StreamingLogMelFrontend()
    frontend.push(tail_changed)
    state = frontend.state
    assert state.frames_emitted == 98
    assert state.next_frame_start == 15_680
    assert state.frame_phase_samples == 320
    assert np.array_equal(state.buffered_samples, tail_changed[15_680:])


def test_phase_tracks_window_overlap_across_boundaries() -> None:
    waveform = _waveform(560)
    frontend = StreamingLogMelFrontend()

    assert frontend.push(waveform[:399]).shape == (0, 40)
    state = frontend.state
    assert (state.total_samples, state.frames_emitted) == (399, 0)
    assert (state.next_frame_start, state.frame_phase_samples) == (0, 399)

    assert frontend.push(waveform[399:400]).shape == (1, 40)
    state = frontend.state
    assert (state.total_samples, state.frames_emitted) == (400, 1)
    assert (state.next_frame_start, state.frame_phase_samples) == (160, 240)

    assert frontend.push(waveform[400:559]).shape == (0, 40)
    assert frontend.state.frame_phase_samples == 399
    assert frontend.push(waveform[559:]).shape == (1, 40)
    assert frontend.state.next_frame_start == 320
    assert frontend.state.frame_phase_samples == 240


def test_utterance_boundary_discards_tail_and_resets_frame_origin() -> None:
    first = _waveform(559)
    second = -_waveform()
    frontend = StreamingLogMelFrontend()

    assert frontend.push(first).shape == (1, 40)
    frontend.end_utterance()
    state = frontend.state
    assert state.total_samples == state.frames_emitted == 0
    assert state.next_frame_start == state.frame_phase_samples == 0
    assert state.buffered_samples.shape == (0,)
    assert not state.buffered_samples.flags.writeable

    observed = frontend.push(second)
    expected = complete_log_mel_frames(second)
    assert observed.tobytes() == expected.tobytes()


def test_continuous_waveform_is_chunk_invariant_beyond_one_second() -> None:
    waveform = _waveform(32_080)
    expected_frames = 1 + (waveform.size - 400) // 160
    one_chunk = complete_log_mel_frames(waveform)
    irregular = _stream(waveform, _partition(waveform.size, (37, 2_003, 160, 1)))

    assert expected_frames == 199
    assert one_chunk.shape == (expected_frames, 40)
    assert irregular.tobytes() == one_chunk.tobytes()


def test_hop_equal_to_window_is_supported_and_chunk_invariant() -> None:
    config = replace(DEFAULT_FRONTEND, hop_samples=DEFAULT_FRONTEND.window_samples)
    waveform = _waveform(2_001)

    one_chunk = complete_log_mel_frames(waveform, config)
    irregular = _stream(
        waveform,
        _partition(waveform.size, (1, 399, 17, 511)),
        config,
    )
    frontend = StreamingLogMelFrontend(config)
    frontend.push(waveform)

    assert one_chunk.shape == irregular.shape == (5, config.mel_bins)
    assert irregular.tobytes() == one_chunk.tobytes()
    assert frontend.state.next_frame_start == 2_000
    assert frontend.state.frame_phase_samples == 1


def test_nondefault_frontend_matches_offline_reference_across_chunks() -> None:
    config = FrontendConfig(
        clip_samples=2_000,
        window_samples=320,
        hop_samples=80,
        fft_samples=512,
        mel_bins=32,
    )
    waveform = _waveform(config.clip_samples, config.sample_rate)

    one_chunk = complete_log_mel_frames(waveform, config)
    irregular = _stream(
        waveform,
        _partition(waveform.size, (7, 121, 1, 257, 64)),
        config,
    )
    offline = log_mel_spectrogram(waveform, config)

    assert one_chunk.shape == irregular.shape == offline.shape == (22, 32)
    assert one_chunk.dtype == irregular.dtype == np.dtype(np.float32)
    assert irregular.tobytes() == one_chunk.tobytes()
    np.testing.assert_allclose(one_chunk, offline, rtol=0.0, atol=2.0e-6)


def test_nonoverlapping_gap_configuration_is_rejected() -> None:
    config = replace(DEFAULT_FRONTEND, hop_samples=DEFAULT_FRONTEND.window_samples + 1)

    with pytest.raises(ValueError, match="hop_samples no greater than window_samples"):
        StreamingLogMelFrontend(config)
    with pytest.raises(ValueError, match="hop_samples no greater than window_samples"):
        complete_log_mel_frames(np.empty(0, dtype=np.float32), config)


@pytest.mark.parametrize(
    ("sample_count", "expected_frames"),
    [
        (0, 0),
        (1, 0),
        (399, 0),
        (400, 1),
        (401, 1),
        (559, 1),
        (560, 2),
        (15_920, 98),
        (16_000, 98),
        (16_001, 98),
    ],
)
def test_frame_count_and_phase_at_analysis_boundaries(
    sample_count: int, expected_frames: int
) -> None:
    frontend = StreamingLogMelFrontend()
    frames = frontend.push(_waveform(sample_count))
    state = frontend.state

    assert frames.shape == (expected_frames, DEFAULT_FRONTEND.mel_bins)
    assert state.frames_emitted == expected_frames
    assert state.next_frame_start == expected_frames * DEFAULT_FRONTEND.hop_samples
    assert state.frame_phase_samples == sample_count - state.next_frame_start
    assert state.total_samples - state.next_frame_start == state.frame_phase_samples


def test_outputs_and_state_snapshots_are_owned_contiguous_float32() -> None:
    frontend = StreamingLogMelFrontend()
    initial = frontend.state
    empty = frontend.push(np.empty(0, dtype=np.float32))

    assert empty.shape == (0, DEFAULT_FRONTEND.mel_bins)
    assert empty.dtype == np.dtype(np.float32)
    assert empty.flags.c_contiguous
    after_empty = frontend.state
    assert after_empty.total_samples == initial.total_samples
    assert after_empty.frames_emitted == initial.frames_emitted
    assert after_empty.next_frame_start == initial.next_frame_start
    assert after_empty.frame_phase_samples == initial.frame_phase_samples
    assert np.array_equal(after_empty.buffered_samples, initial.buffered_samples)

    emitted = frontend.push(_waveform(DEFAULT_FRONTEND.window_samples))
    snapshot = frontend.state
    saved_buffer = snapshot.buffered_samples.copy()
    assert emitted.shape == (1, DEFAULT_FRONTEND.mel_bins)
    assert emitted.dtype == np.dtype(np.float32)
    assert emitted.flags.c_contiguous
    assert np.all(np.isfinite(emitted))
    assert snapshot.buffered_samples.dtype == np.dtype(np.float32)
    assert snapshot.buffered_samples.flags.c_contiguous
    assert not snapshot.buffered_samples.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        snapshot.buffered_samples[0] = np.float32(0.0)

    frontend.push(_waveform(DEFAULT_FRONTEND.hop_samples))
    assert np.array_equal(snapshot.buffered_samples, saved_buffer)
    assert not np.shares_memory(
        snapshot.buffered_samples, frontend.state.buffered_samples
    )


def test_exact_pcm_endpoints_are_accepted_and_infinities_are_rejected() -> None:
    frontend = StreamingLogMelFrontend()
    endpoints = np.asarray([-1.0, 1.0], dtype=np.float32)

    assert frontend.push(endpoints).shape == (0, DEFAULT_FRONTEND.mel_bins)
    before = frontend.state
    for value in (np.inf, -np.inf):
        with pytest.raises(ValueError, match="non-finite"):
            frontend.push(np.asarray([value], dtype=np.float32))
        after = frontend.state
        assert after.total_samples == before.total_samples
        assert after.frames_emitted == before.frames_emitted
        assert after.next_frame_start == before.next_frame_start
        assert after.frame_phase_samples == before.frame_phase_samples
        assert np.array_equal(after.buffered_samples, before.buffered_samples)


def test_invalid_chunks_are_rejected_without_mutating_state() -> None:
    frontend = StreamingLogMelFrontend()
    frontend.push(_waveform(257))
    before = frontend.state

    invalid: tuple[FloatArray, ...] = (
        np.zeros(10, dtype=np.float64),
        np.zeros((2, 10), dtype=np.float32),
        np.asarray([np.nan], dtype=np.float32),
        np.asarray([1.001], dtype=np.float32),
    )
    messages = ("float32", "one-dimensional", "non-finite", r"\[-1, 1\]")
    for chunk, message in zip(invalid, messages, strict=True):
        with pytest.raises(ValueError, match=message):
            frontend.push(chunk)
        after = frontend.state
        assert after.total_samples == before.total_samples
        assert after.frames_emitted == before.frames_emitted
        assert after.next_frame_start == before.next_frame_start
        assert after.frame_phase_samples == before.frame_phase_samples
        assert np.array_equal(after.buffered_samples, before.buffered_samples)

    assert frontend.push(np.empty(0, dtype=np.float32)).shape == (0, 40)
