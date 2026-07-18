from __future__ import annotations

import hashlib
import json
import wave
from dataclasses import replace
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from falsewake.baseline_data import ClipExample
from falsewake.feature_matrix import (
    FeatureMatrixError,
    config_sha256,
    extract_feature_matrix,
    feature_cache_metadata,
    feature_matrix_sha256,
    load_feature_cache,
    write_feature_cache,
)
from falsewake.features import extract_clip_features, pcm16le_to_float32

CONFIG_SHA256 = "c" * 64
MANIFEST_SHA256 = "d" * 64


def _write_wav(
    path: Path,
    samples: np.ndarray[tuple[int], np.dtype[np.int16]],
    *,
    sample_rate: int = 16_000,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(samples.astype("<i2").tobytes())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _example(
    *,
    kind: Literal["command", "silence"] = "command",
    path: str = "yes/alice_nohash_0.wav",
    label: str = "yes",
    start: int = 0,
    samples: int = 800,
    sha256: str = "a" * 64,
) -> ClipExample:
    return ClipExample(
        kind=kind,
        path=path,
        split="train",
        label=label,
        start_sample=start,
        source_sample_count=samples,
        source_sha256=sha256,
    )


def test_extracts_exact_command_and_silence_rows_and_round_trips_cache(
    tmp_path: Path,
) -> None:
    command_pcm = np.arange(-400, 400, dtype=np.int16)
    background_pcm = ((np.arange(32_000) * 17) % 20_000 - 10_000).astype(np.int16)
    command_sha = _write_wav(tmp_path / "yes/alice_nohash_0.wav", command_pcm)
    background_sha = _write_wav(
        tmp_path / "_background_noise_/room.wav", background_pcm
    )
    examples = (
        _example(sha256=command_sha),
        _example(
            kind="silence",
            path="_background_noise_/room.wav",
            label="silence",
            start=16_000,
            samples=32_000,
            sha256=background_sha,
        ),
    )

    first = extract_feature_matrix(tmp_path, examples)
    second = extract_feature_matrix(tmp_path, examples)
    expected_command = extract_clip_features(
        pcm16le_to_float32(command_pcm.astype("<i2").tobytes())
    )
    expected_silence = extract_clip_features(
        pcm16le_to_float32(background_pcm[16_000:].astype("<i2").tobytes())
    )

    assert first.dtype == np.float32
    assert first.shape == (2, 80)
    assert np.array_equal(first, second)
    assert np.array_equal(first[0], expected_command)
    assert np.array_equal(first[1], expected_silence)
    assert feature_matrix_sha256(first) == feature_matrix_sha256(second)

    metadata = write_feature_cache(
        tmp_path / "cache",
        first,
        examples,
        manifest_sha256=MANIFEST_SHA256,
        experiment_config_sha256=CONFIG_SHA256,
    )
    assert metadata == feature_cache_metadata(
        first,
        examples,
        manifest_sha256=MANIFEST_SHA256,
        experiment_config_sha256=CONFIG_SHA256,
    )
    loaded = load_feature_cache(
        tmp_path / "cache",
        examples,
        manifest_sha256=MANIFEST_SHA256,
        experiment_config_sha256=CONFIG_SHA256,
    )
    assert np.array_equal(loaded, first)


@pytest.mark.parametrize(
    "path",
    [
        "/yes/a.wav",
        "../yes/a.wav",
        "./yes/a.wav",
        "yes//a.wav",
        "yes\\a.wav",
        "yes/nested/a.wav",
        "yes/a.flac",
    ],
)
def test_rejects_noncanonical_source_paths(tmp_path: Path, path: str) -> None:
    (tmp_path / "yes").mkdir()
    with pytest.raises(FeatureMatrixError, match="not canonical"):
        extract_feature_matrix(tmp_path, (_example(path=path),))


def test_rejects_source_symlinks_and_wrong_source_kind(tmp_path: Path) -> None:
    samples = np.zeros(800, dtype=np.int16)
    source_sha = _write_wav(tmp_path / "yes/source.wav", samples)
    (tmp_path / "yes/link.wav").symlink_to(tmp_path / "yes/source.wav")
    (tmp_path / "alias").symlink_to(tmp_path / "yes", target_is_directory=True)

    with pytest.raises(FeatureMatrixError, match="symlinks"):
        extract_feature_matrix(
            tmp_path, (_example(path="yes/link.wav", sha256=source_sha),)
        )
    with pytest.raises(FeatureMatrixError, match="symlinks"):
        extract_feature_matrix(
            tmp_path, (_example(path="alias/source.wav", sha256=source_sha),)
        )
    with pytest.raises(FeatureMatrixError, match="invalid source or label"):
        extract_feature_matrix(
            tmp_path,
            (_example(path="yes/source.wav", label="silence", sha256=source_sha),),
        )


def test_rejects_payload_and_cached_background_identity_changes(tmp_path: Path) -> None:
    samples = np.zeros(32_000, dtype=np.int16)
    source_sha = _write_wav(tmp_path / "_background_noise_/room.wav", samples)
    first = _example(
        kind="silence",
        path="_background_noise_/room.wav",
        label="silence",
        samples=32_000,
        sha256=source_sha,
    )

    with pytest.raises(FeatureMatrixError, match="cached source digest conflicts"):
        extract_feature_matrix(
            tmp_path, (first, replace(first, start_sample=1, source_sha256="0" * 64))
        )

    path = tmp_path / "_background_noise_/room.wav"
    contents = bytearray(path.read_bytes())
    contents[-1] ^= 1
    path.write_bytes(contents)
    with pytest.raises(FeatureMatrixError, match="source digest differs"):
        extract_feature_matrix(tmp_path, (first,))


def test_rejects_invalid_wav_contract_and_window_bounds(tmp_path: Path) -> None:
    wrong_rate_sha = _write_wav(
        tmp_path / "yes/wrong_rate.wav",
        np.zeros(800, dtype=np.int16),
        sample_rate=8_000,
    )
    with pytest.raises(FeatureMatrixError, match="unsupported WAV"):
        extract_feature_matrix(
            tmp_path,
            (_example(path="yes/wrong_rate.wav", sha256=wrong_rate_sha),),
        )

    long_sha = _write_wav(tmp_path / "yes/long.wav", np.zeros(16_001, dtype=np.int16))
    with pytest.raises(FeatureMatrixError, match="longer than one clip"):
        extract_feature_matrix(
            tmp_path,
            (_example(path="yes/long.wav", samples=16_001, sha256=long_sha),),
        )

    background_sha = _write_wav(
        tmp_path / "_background_noise_/short.wav", np.zeros(16_000, dtype=np.int16)
    )
    silence = _example(
        kind="silence",
        path="_background_noise_/short.wav",
        label="silence",
        start=1,
        samples=16_000,
        sha256=background_sha,
    )
    with pytest.raises(FeatureMatrixError, match="falls outside"):
        extract_feature_matrix(tmp_path, (silence,))
    with pytest.raises(FeatureMatrixError, match="sample zero"):
        extract_feature_matrix(
            tmp_path,
            (
                _example(
                    path="yes/wrong_rate.wav",
                    start=1,
                    sha256=wrong_rate_sha,
                ),
            ),
        )


def test_cache_detects_metadata_and_matrix_tampering(tmp_path: Path) -> None:
    source_sha = _write_wav(
        tmp_path / "yes/alice_nohash_0.wav", np.zeros(800, dtype=np.int16)
    )
    examples = (_example(sha256=source_sha),)
    features = extract_feature_matrix(tmp_path, examples)
    cache = tmp_path / "cache"
    write_feature_cache(
        cache,
        features,
        examples,
        manifest_sha256=MANIFEST_SHA256,
        experiment_config_sha256=CONFIG_SHA256,
    )
    metadata_path = cache / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["examples_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(FeatureMatrixError, match="metadata or matrix"):
        load_feature_cache(
            cache,
            examples,
            manifest_sha256=MANIFEST_SHA256,
            experiment_config_sha256=CONFIG_SHA256,
        )

    metadata_path.unlink()
    np.save(cache / "features.npy", features + np.float32(1.0), allow_pickle=False)
    metadata_path.write_text(
        json.dumps(
            feature_cache_metadata(
                features,
                examples,
                manifest_sha256=MANIFEST_SHA256,
                experiment_config_sha256=CONFIG_SHA256,
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(FeatureMatrixError, match="metadata or matrix"):
        load_feature_cache(
            cache,
            examples,
            manifest_sha256=MANIFEST_SHA256,
            experiment_config_sha256=CONFIG_SHA256,
        )


def test_config_hash_requires_a_json_object(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_bytes(Path("configs/experiment-000.json").read_bytes())
    assert config_sha256(config) == hashlib.sha256(config.read_bytes()).hexdigest()

    document = json.loads(config.read_text(encoding="utf-8"))
    document["seed"] = 0
    config.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(FeatureMatrixError, match="differs from extraction runtime"):
        config_sha256(config)
