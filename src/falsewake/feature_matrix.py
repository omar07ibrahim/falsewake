"""Extract and cache the registered experiment 000 feature matrix."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import io
import json
import os
import shutil
import struct
import sys
import tempfile
import time
import wave
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import cast

import numpy as np

from falsewake.baseline_data import (
    EXPERIMENT_SEED,
    MANIFEST_SHA256,
    ClipExample,
    build_sampling_plan,
    load_audited_corpus,
    sampling_summary,
)
from falsewake.features import (
    DEFAULT_FRONTEND,
    FloatArray,
    FrontendConfig,
    extract_clip_features,
    pcm16le_to_float32,
)
from falsewake.speech_commands import (
    MAX_BACKGROUND_FILE_BYTES,
    MAX_COMMAND_FILE_BYTES,
    TARGET_WORDS,
)

FEATURE_CACHE_SCHEMA = 1
MAX_CONFIG_BYTES = 1024 * 1024
MAX_METADATA_BYTES = 64 * 1024
_MATRIX_DIGEST_DOMAIN = b"falsewake-feature-matrix-v1\0"


class FeatureMatrixError(ValueError):
    """A source or cache does not match the registered experiment."""


@dataclass(frozen=True, slots=True)
class _CachedSource:
    sha256: str
    sample_count: int
    waveform: FloatArray


Progress = Callable[[int, int], None]


def _require_sha256(value: str, *, name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise FeatureMatrixError(f"{name} is not a lowercase SHA-256 digest")


def _dataset_root(path: Path) -> Path:
    try:
        if path.is_symlink():
            raise FeatureMatrixError("dataset root must not be a symlink")
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise FeatureMatrixError(f"cannot resolve dataset root: {error}") from error
    if not resolved.is_dir():
        raise FeatureMatrixError(f"dataset root is not a directory: {resolved}")
    return resolved


def _source_path(root: Path, example: ClipExample) -> tuple[Path, int]:
    raw_path = example.path
    parsed = PurePosixPath(raw_path)
    invalid = (
        not raw_path
        or "\\" in raw_path
        or "\0" in raw_path
        or parsed.is_absolute()
        or len(parsed.parts) != 2
        or any(part in {"", ".", ".."} for part in parsed.parts)
        or parsed.as_posix() != raw_path
        or parsed.suffix != ".wav"
    )
    if invalid:
        raise FeatureMatrixError(f"source path is not canonical: {raw_path!r}")

    is_background = parsed.parts[0] == "_background_noise_"
    if example.kind == "command":
        if is_background or example.label == "silence":
            raise FeatureMatrixError("command example has an invalid source or label")
        size_limit = MAX_COMMAND_FILE_BYTES
    elif example.kind == "silence":
        if not is_background or example.label != "silence":
            raise FeatureMatrixError("silence example has an invalid source or label")
        size_limit = MAX_BACKGROUND_FILE_BYTES
    else:
        raise FeatureMatrixError(f"example has invalid kind: {example.kind!r}")

    candidate = root
    try:
        for part in parsed.parts:
            candidate /= part
            if candidate.is_symlink():
                raise FeatureMatrixError(
                    f"source path must not contain symlinks: {raw_path!r}"
                )
        if not candidate.is_file():
            raise FeatureMatrixError(f"source is not a regular file: {raw_path!r}")
        size = candidate.stat().st_size
    except OSError as error:
        raise FeatureMatrixError(
            f"cannot inspect source {raw_path!r}: {error}"
        ) from error
    if size > size_limit:
        raise FeatureMatrixError(
            f"source exceeds its size limit: path={raw_path!r}, "
            f"size={size}, limit={size_limit}"
        )
    return candidate, size


def _read_source(path: Path, expected_size: int, example: ClipExample) -> bytes:
    _require_sha256(example.source_sha256, name="source_sha256")
    try:
        contents = path.read_bytes()
    except OSError as error:
        raise FeatureMatrixError(
            f"cannot read source {example.path!r}: {error}"
        ) from error
    if len(contents) != expected_size:
        raise FeatureMatrixError(f"source size changed while reading {example.path!r}")
    observed = hashlib.sha256(contents).hexdigest()
    if not hmac.compare_digest(observed, example.source_sha256):
        raise FeatureMatrixError(
            f"source digest differs for {example.path!r}: observed={observed}"
        )
    return contents


def _decode_wav(
    contents: bytes, example: ClipExample, config: FrontendConfig
) -> FloatArray:
    try:
        with wave.open(io.BytesIO(contents), "rb") as audio:
            channels = audio.getnchannels()
            sample_width = audio.getsampwidth()
            sample_rate = audio.getframerate()
            sample_count = audio.getnframes()
            compression = audio.getcomptype()
            frames = audio.readframes(sample_count + 1)
    except (EOFError, wave.Error) as error:
        raise FeatureMatrixError(
            f"cannot decode WAV {example.path!r}: {error}"
        ) from error
    if (
        channels != 1
        or sample_width != 2
        or sample_rate != config.sample_rate
        or compression != "NONE"
        or sample_count < 1
    ):
        raise FeatureMatrixError(
            f"unsupported WAV format for {example.path!r}: channels={channels}, "
            f"sample_width={sample_width}, sample_rate={sample_rate}, "
            f"sample_count={sample_count}, compression={compression}"
        )
    if sample_count != example.source_sample_count:
        raise FeatureMatrixError(
            f"source frame count differs for {example.path!r}: "
            f"expected={example.source_sample_count}, observed={sample_count}"
        )
    expected_payload_bytes = sample_count * sample_width
    if len(frames) != expected_payload_bytes:
        raise FeatureMatrixError(
            f"truncated WAV payload for {example.path!r}: "
            f"expected={expected_payload_bytes}, read={len(frames)}"
        )
    return pcm16le_to_float32(frames)


def _example_waveform(
    root: Path,
    example: ClipExample,
    config: FrontendConfig,
    background_cache: dict[str, _CachedSource],
) -> FloatArray:
    if example.source_sample_count < 1:
        raise FeatureMatrixError("source_sample_count must be positive")
    if example.kind == "command" and example.start_sample != 0:
        raise FeatureMatrixError("command example must start at sample zero")
    if example.kind == "silence" and example.start_sample < 0:
        raise FeatureMatrixError("silence example must not start before sample zero")
    path, size = _source_path(root, example)

    cached = background_cache.get(example.path) if example.kind == "silence" else None
    if cached is not None:
        if not hmac.compare_digest(cached.sha256, example.source_sha256):
            raise FeatureMatrixError(
                f"cached source digest conflicts for {example.path!r}"
            )
        if cached.sample_count != example.source_sample_count:
            raise FeatureMatrixError(
                f"cached source frame count conflicts for {example.path!r}"
            )
        waveform = cached.waveform
    else:
        waveform = _decode_wav(_read_source(path, size, example), example, config)
        if example.kind == "silence":
            background_cache[example.path] = _CachedSource(
                sha256=example.source_sha256,
                sample_count=example.source_sample_count,
                waveform=waveform,
            )

    if example.kind == "command":
        if waveform.size > config.clip_samples:
            raise FeatureMatrixError(
                f"command source is longer than one clip: {example.path!r}"
            )
        return waveform

    start = example.start_sample
    stop = start + config.clip_samples
    if start < 0 or stop > waveform.size:
        raise FeatureMatrixError(
            f"silence window falls outside {example.path!r}: "
            f"start={start}, stop={stop}, samples={waveform.size}"
        )
    return waveform[start:stop]


def extract_feature_matrix(
    dataset_root: Path,
    examples: Sequence[ClipExample],
    *,
    config: FrontendConfig = DEFAULT_FRONTEND,
    progress: Progress | None = None,
) -> FloatArray:
    """Extract rows in the exact order supplied by the sampling plan."""

    if not examples:
        raise FeatureMatrixError("sampling plan is empty")
    root = _dataset_root(dataset_root)
    matrix = np.empty((len(examples), config.summary_size), dtype=np.float32)
    background_cache: dict[str, _CachedSource] = {}
    for index, example in enumerate(examples):
        features = extract_clip_features(
            _example_waveform(root, example, config, background_cache), config
        )
        if features.dtype != np.dtype(np.float32) or features.shape != (
            config.summary_size,
        ):
            raise FeatureMatrixError("frontend returned an invalid feature row")
        matrix[index] = features
        if progress is not None:
            progress(index + 1, len(examples))
    if not np.all(np.isfinite(matrix)):
        raise FeatureMatrixError("feature matrix contains non-finite values")
    return matrix


def feature_matrix_sha256(features: FloatArray) -> str:
    """Hash dimensions and canonical little-endian float32 values."""

    if features.dtype != np.dtype(np.float32) or features.ndim != 2:
        raise FeatureMatrixError("feature matrix must be two-dimensional float32")
    if not np.all(np.isfinite(features)):
        raise FeatureMatrixError("feature matrix contains non-finite values")
    canonical = np.ascontiguousarray(features, dtype="<f4")
    digest = hashlib.sha256()
    digest.update(_MATRIX_DIGEST_DOMAIN)
    digest.update(struct.pack("<QQ", *canonical.shape))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def config_sha256(path: Path) -> str:
    """Validate runtime identities and hash the registered experiment document."""

    try:
        if path.stat().st_size > MAX_CONFIG_BYTES:
            raise FeatureMatrixError("experiment config exceeds the size limit")
        contents = path.read_bytes()
        document = json.loads(contents)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise FeatureMatrixError(f"cannot read experiment config: {error}") from error
    if type(document) is not dict:
        raise FeatureMatrixError("experiment config is not a JSON object")
    registered = cast(dict[object, object], document)
    expected_frontend: dict[str, object] = {
        **asdict(DEFAULT_FRONTEND),
        "pcm16_divisor": 32_768,
    }
    expected_sampling = {
        "silence_count": "same_as_unknown",
        "silence_file": "sha256_modulo_equal_split_files",
        "silence_start": "sha256_modulo_valid_offsets",
        "target_clips": "all",
        "unknown_count": "median_low_target_count_per_split",
    }
    manifest = registered.get("manifest")
    manifest_matches = (
        type(manifest) is dict
        and cast(dict[object, object], manifest).get("jsonl_sha256") == MANIFEST_SHA256
    )
    if (
        registered.get("seed") != EXPERIMENT_SEED
        or registered.get("class_order") != [*TARGET_WORDS, "unknown", "silence"]
        or registered.get("frontend") != expected_frontend
        or registered.get("sampling") != expected_sampling
        or not manifest_matches
    ):
        raise FeatureMatrixError("experiment config differs from extraction runtime")
    return hashlib.sha256(contents).hexdigest()


def feature_cache_metadata(
    features: FloatArray,
    examples: Sequence[ClipExample],
    *,
    manifest_sha256: str,
    experiment_config_sha256: str,
    config: FrontendConfig = DEFAULT_FRONTEND,
) -> dict[str, object]:
    """Describe a matrix without depending on its container bytes."""

    _require_sha256(manifest_sha256, name="manifest_sha256")
    _require_sha256(experiment_config_sha256, name="experiment_config_sha256")
    if features.shape != (len(examples), config.summary_size):
        raise FeatureMatrixError("feature matrix shape differs from the sampling plan")
    selected = sampling_summary(examples)
    return {
        "dtype": "float32",
        "examples_sha256": selected["examples_sha256"],
        "experiment_config_sha256": experiment_config_sha256,
        "features_sha256": feature_matrix_sha256(features),
        "frontend": asdict(config),
        "label_counts_by_split": selected["label_counts_by_split"],
        "manifest_sha256": manifest_sha256,
        "schema_version": FEATURE_CACHE_SCHEMA,
        "shape": list(features.shape),
    }


def write_feature_cache(
    output: Path,
    features: FloatArray,
    examples: Sequence[ClipExample],
    *,
    manifest_sha256: str,
    experiment_config_sha256: str,
    config: FrontendConfig = DEFAULT_FRONTEND,
) -> dict[str, object]:
    """Publish one fail-if-present matrix directory through a staged rename."""

    metadata = feature_cache_metadata(
        features,
        examples,
        manifest_sha256=manifest_sha256,
        experiment_config_sha256=experiment_config_sha256,
        config=config,
    )
    output_parent = output.parent.resolve()
    if not output_parent.is_dir():
        raise FeatureMatrixError(f"cache parent is not a directory: {output_parent}")
    if output.exists() or output.is_symlink():
        raise FeatureMatrixError(f"cache output already exists: {output}")
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output_parent))
    try:
        np.save(stage / "features.npy", features, allow_pickle=False)
        serialized = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        (stage / "metadata.json").write_text(serialized, encoding="utf-8")
        os.replace(stage, output)
    except (OSError, ValueError) as error:
        shutil.rmtree(stage, ignore_errors=True)
        raise FeatureMatrixError(f"cannot publish feature cache: {error}") from error
    return metadata


def load_feature_cache(
    directory: Path,
    examples: Sequence[ClipExample],
    *,
    manifest_sha256: str,
    experiment_config_sha256: str,
    config: FrontendConfig = DEFAULT_FRONTEND,
) -> FloatArray:
    """Load a cache only when every registered identity still matches."""

    metadata_path = directory / "metadata.json"
    feature_path = directory / "features.npy"
    try:
        if directory.is_symlink() or not directory.is_dir():
            raise FeatureMatrixError("feature cache is not a regular directory")
        if metadata_path.is_symlink() or feature_path.is_symlink():
            raise FeatureMatrixError("feature cache must not contain symlinks")
        if metadata_path.stat().st_size > MAX_METADATA_BYTES:
            raise FeatureMatrixError("feature metadata exceeds the size limit")
        maximum_feature_bytes = (
            len(examples) * config.summary_size * np.dtype(np.float32).itemsize
            + 1024 * 1024
        )
        if feature_path.stat().st_size > maximum_feature_bytes:
            raise FeatureMatrixError("feature matrix file exceeds the size limit")
        metadata = json.loads(metadata_path.read_bytes())
        loaded = np.load(feature_path, allow_pickle=False)
    except (OSError, ValueError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise FeatureMatrixError(f"cannot load feature cache: {error}") from error
    if type(metadata) is not dict or not isinstance(loaded, np.ndarray):
        raise FeatureMatrixError("feature cache has an invalid container")
    features = cast(FloatArray, loaded)
    expected = feature_cache_metadata(
        features,
        examples,
        manifest_sha256=manifest_sha256,
        experiment_config_sha256=experiment_config_sha256,
        config=config,
    )
    if metadata != expected:
        raise FeatureMatrixError("feature cache metadata or matrix digest differs")
    return features


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract the byte-bound feature matrix for experiment 000."
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiment-000.json")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the exact Speech Commands feature extraction."""

    arguments = _parser().parse_args(argv)
    try:
        corpus = load_audited_corpus(
            arguments.manifest, expected_sha256=MANIFEST_SHA256
        )
        examples = build_sampling_plan(corpus)
        started = time.perf_counter()

        def show_progress(completed: int, total: int) -> None:
            if completed % 1_000 == 0 or completed == total:
                print(f"features {completed}/{total}", file=sys.stderr, flush=True)

        features = extract_feature_matrix(
            arguments.dataset_root, examples, progress=show_progress
        )
        metadata = write_feature_cache(
            arguments.output,
            features,
            examples,
            manifest_sha256=corpus.manifest_sha256,
            experiment_config_sha256=config_sha256(arguments.config),
        )
    except (FeatureMatrixError, ValueError) as error:
        print(f"feature extraction failed: {error}", file=sys.stderr)
        return 2
    report = {
        **metadata,
        "extraction_seconds": round(time.perf_counter() - started, 3),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
