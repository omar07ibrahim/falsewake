from __future__ import annotations

import struct

import numpy as np
import pytest

from falsewake.features import (
    DEFAULT_FRONTEND,
    FrontendConfig,
    extract_clip_features,
    frame_waveform,
    log_mel_spectrogram,
    mel_edge_frequencies,
    mel_filterbank,
    pcm16le_to_float32,
    summarize_log_mel,
)


def test_default_frontend_geometry() -> None:
    config = DEFAULT_FRONTEND
    assert config.frame_count == 98
    assert config.summary_size == 80

    filters = mel_filterbank(config)
    assert filters.shape == (40, 257)
    assert np.all(np.isfinite(filters))
    assert np.all(filters >= 0)
    assert np.all(np.max(filters, axis=1) > 0)

    edges = mel_edge_frequencies(config)
    assert edges.shape == (42,)
    assert np.all(np.diff(edges) > 0)
    assert edges[0] == pytest.approx(20.0, abs=1e-3)
    assert edges[-1] == pytest.approx(7_600.0, abs=1e-2)


def test_silence_has_the_log_floor_and_no_temporal_spread() -> None:
    waveform = np.zeros(DEFAULT_FRONTEND.clip_samples, dtype=np.float32)
    log_mel = log_mel_spectrogram(waveform)
    features = summarize_log_mel(log_mel)

    expected_floor = np.log(DEFAULT_FRONTEND.log_floor)
    assert log_mel.shape == (98, 40)
    assert np.allclose(log_mel, expected_floor, atol=1e-6)
    assert np.allclose(features[:40], expected_floor, atol=1e-6)
    assert np.array_equal(features[40:], np.zeros(40, dtype=np.float32))


def test_pcm16le_conversion_fixes_the_waveform_scale() -> None:
    contents = struct.pack("<hhhh", -32_768, -1, 0, 32_767)
    waveform = pcm16le_to_float32(contents)

    assert waveform.dtype == np.float32
    assert np.array_equal(
        waveform,
        np.asarray([-1.0, -1 / 32_768, 0.0, 32_767 / 32_768], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="even"):
        pcm16le_to_float32(b"\x00")


def test_small_frontend_matches_an_independent_numeric_golden() -> None:
    config = FrontendConfig(
        sample_rate=8,
        clip_samples=8,
        window_samples=4,
        hop_samples=2,
        fft_samples=4,
        mel_bins=2,
        min_hz=0.0,
        max_hz=4.0,
    )
    waveform = np.asarray(
        [0.25, -0.5, 0.75, -1.0, 0.5, 0.0, -0.25, 1.0], dtype=np.float32
    )

    assert np.allclose(
        mel_edge_frequencies(config),
        np.asarray([0.0, 1.33080168, 2.66413341, 4.0], dtype=np.float32),
        atol=1e-6,
    )
    assert np.allclose(
        mel_filterbank(config),
        np.asarray(
            [[0.0, 0.37393072, 0.0], [0.0, 0.37606748, 0.0]],
            dtype=np.float32,
        ),
        atol=1e-6,
    )
    assert np.allclose(
        log_mel_spectrogram(waveform, config),
        np.asarray(
            [
                [-2.83998274, -2.83428468],
                [-3.06312629, -3.05742823],
                [-3.53312992, -3.52743186],
            ],
            dtype=np.float32,
        ),
        atol=2e-6,
    )


def test_one_kilohertz_tone_peaks_in_the_nearest_mel_band() -> None:
    samples = np.arange(DEFAULT_FRONTEND.clip_samples, dtype=np.float32)
    waveform = (0.5 * np.sin(2 * np.pi * 1_000 * samples / 16_000)).astype(np.float32)
    log_mel = log_mel_spectrogram(waveform)
    centers = mel_edge_frequencies()[1:-1]

    observed = int(np.argmax(np.mean(log_mel, axis=0)))
    expected = int(np.argmin(np.abs(centers - 1_000)))
    assert observed == expected
    assert extract_clip_features(waveform).shape == (80,)


def test_short_clip_is_right_padded_without_mutating_input() -> None:
    waveform = np.linspace(-0.25, 0.25, 800, dtype=np.float32)
    original = waveform.copy()
    frames = frame_waveform(waveform)

    assert frames.shape == (98, 400)
    assert np.array_equal(waveform, original)
    assert np.array_equal(frames[0], waveform[:400])
    assert np.count_nonzero(frames[-1]) == 0


def test_pad_end_is_disabled_and_leaves_the_final_eighty_samples_unframed() -> None:
    waveform = np.zeros(DEFAULT_FRONTEND.clip_samples, dtype=np.float32)
    waveform[-1] = 1.0

    assert np.count_nonzero(frame_waveform(waveform)) == 0


def test_invalid_waveforms_and_configurations_are_rejected() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        frame_waveform(np.zeros((2, 100), dtype=np.float32))
    with pytest.raises(ValueError, match="limit"):
        frame_waveform(np.zeros(16_001, dtype=np.float32))
    with pytest.raises(ValueError, match="non-finite"):
        frame_waveform(np.asarray([np.nan], dtype=np.float32))
    with pytest.raises(ValueError, match="normalized float32"):
        frame_waveform(np.zeros(100, dtype=np.int16))
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        frame_waveform(np.asarray([1.001], dtype=np.float32))
    with pytest.raises(ValueError, match="Nyquist"):
        FrontendConfig(max_hz=8_001.0)
    with pytest.raises(ValueError, match="fft_samples"):
        FrontendConfig(fft_samples=128)
    with pytest.raises(ValueError, match="collapse"):
        mel_filterbank(FrontendConfig(min_hz=1_000.0, max_hz=1_000.00001))
