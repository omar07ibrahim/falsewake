from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from falsewake.experiment_002_normalization import (
    NormalizationAccumulator,
    NormalizationStats,
    load_stats,
    normalize_log_mel,
    serialize_stats,
)


def _golden_frames() -> tuple[np.ndarray[tuple[int, int], np.dtype[np.float32]], ...]:
    return (
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.asarray([[-1.0, 0.0], [5.0, 6.0]], dtype=np.float32),
    )


def test_normalization_dependencies_are_numpy_and_standard_library_only() -> None:
    source = Path("src/falsewake/experiment_002_normalization.py").read_text(
        encoding="utf-8"
    )
    imported_roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported_roots.update(
                alias.name.split(".", maxsplit=1)[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".", maxsplit=1)[0])

    assert imported_roots == {"__future__", "dataclasses", "numpy"}
    assert "np.sum" not in source


def test_scalar_accumulation_serialization_and_application_match_golden() -> None:
    accumulator = NormalizationAccumulator(mel_bins=2)
    first, second = _golden_frames()
    accumulator.update(first)
    accumulator.update(second)
    stats = accumulator.finalize(expected_frames=4)

    assert accumulator.frame_count == 4
    assert stats.mel_bins == accumulator.mel_bins == 2
    np.testing.assert_array_equal(stats.means, np.asarray([2.0, 3.0], np.float32))
    expected_deviation = np.float32(np.sqrt(np.float64(5.0)))
    np.testing.assert_array_equal(
        stats.standard_deviations,
        np.asarray([expected_deviation, expected_deviation], np.float32),
    )

    contents = serialize_stats(stats)
    assert contents.hex() == "0000004000004040bd1b0f40bd1b0f40"
    assert hashlib.sha256(contents).hexdigest() == (
        "df00d3fe6654fc9f35fe175a9b93e1021bc4c53356c1c1c11f465bf384b2ac1e"
    )
    restored = load_stats(contents, mel_bins=2)
    assert serialize_stats(restored) == contents

    normalized = normalize_log_mel(np.asarray([[1.0, 2.0]], dtype=np.float32), restored)
    np.testing.assert_array_equal(
        normalized.view(np.uint32),
        np.asarray([[0xBEE4F92E, 0xBEE4F92E]], dtype=np.uint32),
    )


def test_partitioning_updates_does_not_change_scalar_addition_order() -> None:
    pattern = np.asarray(
        [
            [1.0e30, -7.0, 1.0e-20],
            [1.0, 3.0e20, -5.0],
            [-1.0e30, 2.0, 8.0e-20],
            [9.0, -3.0e20, 11.0],
        ],
        dtype=np.float32,
    )
    frames = np.tile(pattern, (251, 1))

    one_update = NormalizationAccumulator(mel_bins=3)
    one_update.update(frames)
    one_stats = one_update.finalize(expected_frames=frames.shape[0])

    partitioned = NormalizationAccumulator(mel_bins=3)
    boundaries = (1, 17, 311, 777, frames.shape[0])
    start = 0
    for stop in boundaries:
        partitioned.update(frames[start:stop])
        start = stop
    partitioned_stats = partitioned.finalize(expected_frames=frames.shape[0])

    assert serialize_stats(partitioned_stats) == serialize_stats(one_stats)


def test_application_uses_two_separate_float32_operations() -> None:
    log_mel = np.asarray([[96.18840026855469]], dtype=np.float32)
    stats = NormalizationStats(
        means=np.asarray([-56.338844299316406], dtype=np.float32),
        standard_deviations=np.asarray([0.7969446182250977], dtype=np.float32),
    )

    observed = normalize_log_mel(log_mel, stats)
    fused_float64 = np.asarray(
        [
            [
                np.float32(
                    (np.float64(log_mel[0, 0]) - np.float64(stats.means[0]))
                    / np.float64(stats.standard_deviations[0])
                )
            ]
        ],
        dtype=np.float32,
    )

    assert int(observed.view(np.uint32)[0, 0]) == 0x433F63D9
    assert int(fused_float64.view(np.uint32)[0, 0]) == 0x433F63D8
    assert observed.dtype == np.dtype(np.float32)
    assert observed.flags.c_contiguous


def test_stats_take_immutable_owned_copies() -> None:
    means = np.asarray([1.0, 2.0], dtype=np.float32)
    deviations = np.asarray([3.0, 4.0], dtype=np.float32)
    stats = NormalizationStats(means=means, standard_deviations=deviations)
    means[:] = -1.0
    deviations[:] = -1.0

    np.testing.assert_array_equal(stats.means, np.asarray([1.0, 2.0], np.float32))
    np.testing.assert_array_equal(
        stats.standard_deviations, np.asarray([3.0, 4.0], np.float32)
    )
    assert not np.shares_memory(stats.means, means)
    assert not np.shares_memory(stats.standard_deviations, deviations)
    assert stats.means.flags.c_contiguous
    assert stats.standard_deviations.flags.c_contiguous
    assert not stats.means.flags.writeable
    assert not stats.standard_deviations.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        stats.means[0] = np.float32(0.0)


def test_accumulator_validation_happens_before_state_mutation() -> None:
    accumulator = NormalizationAccumulator(mel_bins=2)
    valid = np.asarray([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32)
    accumulator.update(valid)

    invalid_inputs = (
        np.asarray([[1.0, 2.0]], dtype=np.float64),
        np.asarray([1.0, 2.0], dtype=np.float32),
        np.empty((0, 2), dtype=np.float32),
        np.zeros((1, 3), dtype=np.float32),
        np.asarray([[np.nan, 2.0]], dtype=np.float32),
    )
    for invalid in invalid_inputs:
        with pytest.raises(ValueError):
            accumulator.update(
                cast(np.ndarray[tuple[int, int], np.dtype[np.float32]], invalid)
            )
        assert accumulator.frame_count == 2

    observed = accumulator.finalize(expected_frames=2)
    reference = NormalizationAccumulator(mel_bins=2)
    reference.update(valid)
    assert serialize_stats(observed) == serialize_stats(
        reference.finalize(expected_frames=2)
    )


@pytest.mark.parametrize(
    ("means", "deviations"),
    [
        (
            np.asarray([1.0], dtype=np.float64),
            np.asarray([1.0], dtype=np.float32),
        ),
        (
            np.asarray([[1.0]], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
        ),
        (
            np.asarray([1.0, 2.0], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
        ),
        (
            np.asarray([np.nan], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
        ),
        (
            np.asarray([1.0], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
        ),
        (
            np.asarray([1.0], dtype=np.float32),
            np.asarray([-1.0], dtype=np.float32),
        ),
    ],
)
def test_invalid_stats_are_rejected(
    means: np.ndarray[tuple[int, ...], np.dtype[np.floating]],
    deviations: np.ndarray[tuple[int, ...], np.dtype[np.floating]],
) -> None:
    with pytest.raises(ValueError):
        NormalizationStats(
            means=cast(np.ndarray[tuple[int], np.dtype[np.float32]], means),
            standard_deviations=cast(
                np.ndarray[tuple[int], np.dtype[np.float32]], deviations
            ),
        )


def test_zero_variance_and_frame_count_mismatches_fail_closed() -> None:
    accumulator = NormalizationAccumulator(mel_bins=2)
    accumulator.update(np.ones((3, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="expected 2 frames, accumulated 3"):
        accumulator.finalize(expected_frames=2)
    with pytest.raises(ValueError, match="standard deviation is zero"):
        accumulator.finalize(expected_frames=3)

    empty = NormalizationAccumulator(mel_bins=2)
    with pytest.raises(ValueError, match="expected_frames must be positive"):
        empty.finalize(expected_frames=0)
    with pytest.raises(TypeError):
        empty.finalize(expected_frames=cast(int, True))


def test_artifact_loader_requires_exact_size_and_valid_values() -> None:
    valid = bytes.fromhex("0000004000004040bd1b0f40bd1b0f40")
    for malformed in (valid[:-1], valid + b"\0"):
        with pytest.raises(ValueError, match="exactly 16 bytes"):
            load_stats(malformed, mel_bins=2)

    zero_deviation = np.asarray([1.0, 2.0, 0.0, 1.0], dtype="<f4").tobytes()
    with pytest.raises(ValueError, match="strictly positive"):
        load_stats(zero_deviation, mel_bins=2)
    nonfinite = np.asarray([1.0, 2.0, np.inf, 1.0], dtype="<f4").tobytes()
    with pytest.raises(ValueError, match="finite"):
        load_stats(nonfinite, mel_bins=2)


def test_normalize_rejects_invalid_inputs_without_modifying_them() -> None:
    stats = NormalizationStats(
        means=np.asarray([1.0, 2.0], dtype=np.float32),
        standard_deviations=np.asarray([2.0, 4.0], dtype=np.float32),
    )
    source = np.asarray([[3.0, 6.0]], dtype=np.float32)
    saved = source.copy()
    normalized = normalize_log_mel(source, stats)
    np.testing.assert_array_equal(source, saved)
    np.testing.assert_array_equal(normalized, np.asarray([[1.0, 1.0]], np.float32))

    invalid_inputs = (
        np.asarray([[3.0, 6.0]], dtype=np.float64),
        np.asarray([3.0, 6.0], dtype=np.float32),
        np.empty((0, 2), dtype=np.float32),
        np.zeros((1, 3), dtype=np.float32),
        np.asarray([[3.0, np.inf]], dtype=np.float32),
    )
    for invalid in invalid_inputs:
        with pytest.raises(ValueError):
            normalize_log_mel(
                cast(np.ndarray[tuple[int, int], np.dtype[np.float32]], invalid),
                stats,
            )


def test_registered_artifact_is_exactly_320_bytes_for_40_mel_bands() -> None:
    config = json.loads(
        Path("configs/experiment-002-training.json").read_text(encoding="utf-8")
    )
    normalization = config["normalization"]
    mel_bins = config["frontend"]["mel_bins"]
    assert mel_bins == 40
    assert normalization["total_frames"] == (
        normalization["clip_count"] * normalization["frame_count_per_clip"]
    )

    stats = NormalizationStats(
        means=np.zeros(mel_bins, dtype=np.float32),
        standard_deviations=np.ones(mel_bins, dtype=np.float32),
    )
    contents = serialize_stats(stats)
    assert len(contents) == 320
    assert load_stats(contents).mel_bins == 40
