from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn, cast

from falsewake.experiment_002_data import (
    MANIFEST_SHA256,
    TRAIN_COMMAND_INVENTORY_SHA256,
    VALIDATION_COMMAND_INVENTORY_SHA256,
)
from falsewake.experiment_002_normalization import serialize_stats
from falsewake.experiment_002_normalization_artifact import (
    FRAMES_PER_CLIP,
    MEL_BINS,
    NORMALIZATION_ARTIFACT_BYTES,
    REGISTERED_CLIP_COUNT,
    REGISTERED_FRAME_COUNT,
    NormalizationArtifactIdentity,
    load_normalization_artifact,
    verify_registered_normalization,
)
from falsewake.experiment_002_pcm_cache import REGISTERED_FILE_BYTES

ARTIFACT_PATH = Path("models/experiment-002-normalization.f32")
REPORT_PATH = Path("reports/experiment-002-normalization.json")
PCM_REPORT_PATH = Path("reports/experiment-002-pcm-cache.json")
TRAINING_CONFIG_PATH = Path("configs/experiment-002-training.json")
NUMERICS_CONFIG_PATH = Path("configs/experiment-002-numerics.json")

ARTIFACT_SHA256 = "891900d4c36fa8a71ba429384f3f4ff594de7c81347ed846c0f790dc4579268e"
MEANS_SHA256 = "3a74b4fb5d189b4eaf1a52d9cf5c8244da95568a26062f493647ede68477f74c"
DEVIATIONS_SHA256 = "8b2aeaf8a7d00a71dae73927bff31725cbb96b8c911299c8ba5e2476e7fa6465"
REPORT_SHA256 = "2b95565fec7dc956a4b1f667638b87e3df64cfaaca3026017942a8057c5187a8"
TRAINING_CONFIG_SHA256 = (
    "a4a21ff66ef04bf15950853acda0897f24ed01eecf566c795a7330e69593d44a"
)
NUMERICS_CONFIG_SHA256 = (
    "d11fc50ba8f609551dbf88d7863a5b45cd138c6681da6c10ca4c0a9eaf8f2be9"
)
PCM_REPORT_SHA256 = "a07b02a78bb6e0d2eef8548ec71a48e717f26c756548665f004b0ca1bf3e52ca"
PCM_CACHE_SHA256 = "b56270a4b99d235c62c162545b139a79b7cf3dc157259a3bbf5678d8a451f653"
IMPLEMENTATION_GIT_COMMIT = "c1d332722ed142fd1905a681a6d7ac81b755c45b"

SOURCE_SHA256 = {
    "src/falsewake/__init__.py": (
        "acc31446071a3860f5ee7629c148bd96247cae20ee02b47e99b61f9d399df87b"
    ),
    "src/falsewake/experiment_002_data.py": (
        "23b4902e0226b9ff9d792ca27a603a458d87e7401678cee3729f86a30b01d573"
    ),
    "src/falsewake/experiment_002_frontend.py": (
        "fdfa2534080cfbea2143c5b87905e1ce43be0d1e4aad4503063506d15e6c65ca"
    ),
    "src/falsewake/experiment_002_normalization.py": (
        "73bbe76aab35d352d2b702f67480f93645eb09cbd7627a9fafde86d3ecb69a81"
    ),
    "src/falsewake/experiment_002_normalization_artifact.py": (
        "103efbb001de4b75af81f5eaccc4fb8e3ae7dfb905b51860213aa3241e3e7dc9"
    ),
    "src/falsewake/experiment_002_pcm_cache.py": (
        "bcc05ecc797aebf809026675e9d4c71fc51dabfdb8177a04020f85c6d31fc961"
    ),
    "src/falsewake/experiment_002_preprocessing.py": (
        "d4438d9a52b33bbe610026df981ee23d2cb0956e0c1350ccce933c5a85903259"
    ),
    "src/falsewake/experiment_002_rng.py": (
        "174e445b47d6da586ae92ea3c4a694966bb6e2acdc2be92f31c1a7023a0fcd4e"
    ),
    "src/falsewake/features.py": (
        "3e7baf53c8dc772a68cd256ce6f4ca6428117c36f051c916f247dc409e515e3c"
    ),
    "src/falsewake/speech_commands.py": (
        "f80bd2bad56113d4adc168b04e1633ec22d8513afb973a4b5dc37d616901e234"
    ),
    "src/falsewake/speech_commands_pcm.py": (
        "f4cd5a2e8dd542d67b41cbe7db41ac031c93d3435787cecb941146c0017e8f33"
    ),
}
RUNTIME_EVIDENCE = {
    "byte_order": "little",
    "cpu_model": "AMD EPYC 7R13 Processor",
    "floating_authority": (
        "CPython_3.12.3_float_operations_and_power_using_the_host_libm_plus_"
        "NumPy_2.5.1_explicit_dtype_operations_and_the_bound_source_bytes"
    ),
    "machine": "x86_64",
    "numpy": "2.5.1",
    "platform": "Linux-6.17.0-1017-aws-x86_64-with-glibc2.39",
    "python": "3.12.3",
    "python_implementation": "CPython",
}


def _reject_nonfinite_json(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json(path: Path) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite_json,
        ),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, nested in value.items():
            yield from _strings(key)
            yield from _strings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _strings(nested)


def test_committed_normalization_is_an_independent_binary_oracle() -> None:
    contents = ARTIFACT_PATH.read_bytes()

    assert len(contents) == NORMALIZATION_ARTIFACT_BYTES == 320
    assert hashlib.sha256(contents).hexdigest() == ARTIFACT_SHA256
    assert hashlib.sha256(contents[:160]).hexdigest() == MEANS_SHA256
    assert hashlib.sha256(contents[160:]).hexdigest() == DEVIATIONS_SHA256

    values = struct.unpack("<80f", contents)
    means = values[:MEL_BINS]
    deviations = values[MEL_BINS:]
    assert len(means) == len(deviations) == MEL_BINS == 40
    assert all(math.isfinite(value) for value in values)
    assert all(value > 0.0 for value in deviations)
    assert float(min(means)).hex() == "-0x1.4c21540000000p+4"
    assert float(max(means)).hex() == "-0x1.bd10e80000000p+3"
    assert float(min(deviations)).hex() == "0x1.7afe5c0000000p+1"
    assert float(max(deviations)).hex() == "0x1.5540840000000p+2"

    identity = NormalizationArtifactIdentity(
        byte_count=NORMALIZATION_ARTIFACT_BYTES,
        sha256=ARTIFACT_SHA256,
    )
    verified = load_normalization_artifact(ARTIFACT_PATH, identity)
    verify_registered_normalization(verified)
    assert verified.identity == identity
    assert verified.clip_count == REGISTERED_CLIP_COUNT
    assert verified.frame_count == REGISTERED_FRAME_COUNT
    assert serialize_stats(verified.stats) == contents


def test_normalization_report_is_canonical_fixed_and_path_clean() -> None:
    raw = REPORT_PATH.read_text(encoding="utf-8")
    report = _strict_json(REPORT_PATH)

    assert hashlib.sha256(raw.encode("utf-8")).hexdigest() == REPORT_SHA256
    assert (
        raw
        == json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )
    assert set(report) == {
        "artifact",
        "bindings",
        "data_boundary",
        "experiment",
        "inputs",
        "measurements",
        "population",
        "runtime",
        "schema_version",
    }
    assert report["schema_version"] == 1
    assert report["experiment"] == "002"
    assert set(report["artifact"]) == {
        "byte_count",
        "format",
        "path",
        "role",
        "sha256",
    }
    assert set(report["bindings"]) == {
        "configs",
        "implementation_git_commit",
        "sources",
    }
    assert set(report["measurements"]) == {
        "independent_fresh_processes",
        "repeat_byte_identity",
        "rss_measurement",
        "runs",
        "statistics",
    }

    for value in _strings(report):
        lowered = value.lower()
        assert ".t/" not in lowered
        assert "falsewake-artifacts" not in lowered
        assert "\\" not in value
        assert not value.startswith(("/", "~"))
        assert not (len(value) >= 2 and value[1] == ":")
        if "/" in value:
            assert value.startswith(("configs/", "models/", "reports/", "src/"))
            assert ".." not in PurePosixPath(value).parts

    assert report["runtime"] == RUNTIME_EVIDENCE


def test_report_population_and_provenance_reconcile_with_frozen_inputs() -> None:
    report = _strict_json(REPORT_PATH)
    training = _strict_json(TRAINING_CONFIG_PATH)
    numerics = _strict_json(NUMERICS_CONFIG_PATH)
    pcm_report = _strict_json(PCM_REPORT_PATH)
    artifact = report["artifact"]
    population = report["population"]
    speech_commands = report["inputs"]["speech_commands"]
    pcm_cache = report["inputs"]["pcm_cache"]

    assert _sha256(TRAINING_CONFIG_PATH) == TRAINING_CONFIG_SHA256
    assert _sha256(NUMERICS_CONFIG_PATH) == NUMERICS_CONFIG_SHA256
    assert _sha256(PCM_REPORT_PATH) == PCM_REPORT_SHA256

    assert artifact == {
        "byte_count": NORMALIZATION_ARTIFACT_BYTES,
        "format": training["normalization"]["output"],
        "path": str(ARTIFACT_PATH),
        "role": "frozen_training_normalization_input",
        "sha256": ARTIFACT_SHA256,
    }
    assert population == {
        "clip_count": REGISTERED_CLIP_COUNT,
        "frame_count_per_clip": FRAMES_PER_CLIP,
        "mel_bins": MEL_BINS,
        "order": (
            "canonical_manifest_order_then_frame_0_through_97_then_mel_0_through_39"
        ),
        "source": "every_registered_training_command_exactly_once",
        "total_frames": REGISTERED_FRAME_COUNT,
    }
    assert REGISTERED_FRAME_COUNT == REGISTERED_CLIP_COUNT * FRAMES_PER_CLIP
    assert training["normalization"]["clip_count"] == REGISTERED_CLIP_COUNT
    assert training["normalization"]["frame_count_per_clip"] == FRAMES_PER_CLIP
    assert training["normalization"]["total_frames"] == REGISTERED_FRAME_COUNT
    assert training["data"]["manifest"]["split_counts"]["train"] == (
        REGISTERED_CLIP_COUNT
    )

    assert speech_commands == {
        "archive_sha256": training["data"]["archive"]["sha256"],
        "manifest_inventory_sha256": training["data"]["manifest"]["inventory_sha256"],
        "manifest_sha256": MANIFEST_SHA256,
        "train_command_inventory_sha256": TRAIN_COMMAND_INVENTORY_SHA256,
        "validation_command_inventory_sha256": (VALIDATION_COMMAND_INVENTORY_SHA256),
    }
    assert pcm_cache == {
        "byte_count": REGISTERED_FILE_BYTES,
        "external_identity_verified": True,
        "report_path": str(PCM_REPORT_PATH),
        "report_sha256": PCM_REPORT_SHA256,
        "sha256": PCM_CACHE_SHA256,
    }
    assert pcm_report["artifact"]["sha256"] == PCM_CACHE_SHA256
    assert pcm_cache["byte_count"] == pcm_report["artifact"]["byte_count"]
    assert pcm_report["inputs"]["manifest_sha256"] == MANIFEST_SHA256
    assert (
        pcm_report["inputs"]["manifest_inventory_sha256"]
        == (speech_commands["manifest_inventory_sha256"])
    )
    assert pcm_report["inputs"]["train_command_inventory_sha256"] == (
        TRAIN_COMMAND_INVENTORY_SHA256
    )
    assert pcm_report["inputs"]["validation_command_inventory_sha256"] == (
        VALIDATION_COMMAND_INVENTORY_SHA256
    )

    configs = report["bindings"]["configs"]
    assert configs == {
        str(NUMERICS_CONFIG_PATH): NUMERICS_CONFIG_SHA256,
        str(TRAINING_CONFIG_PATH): TRAINING_CONFIG_SHA256,
    }
    assert (
        numerics["identity"]["phase_1_training_config_sha256"]
        == configs[str(TRAINING_CONFIG_PATH)]
    )
    for path, digest in numerics["identity"]["bound_sources"].items():
        assert report["bindings"]["sources"][path] == digest
    assert report["bindings"]["implementation_git_commit"] == (
        IMPLEMENTATION_GIT_COMMIT
    )
    assert report["bindings"]["sources"] == SOURCE_SHA256
    for path, digest in SOURCE_SHA256.items():
        assert _sha256(Path(path)) == digest


def test_reported_runs_statistics_and_data_firewall_are_exact() -> None:
    report = _strict_json(REPORT_PATH)
    training = _strict_json(TRAINING_CONFIG_PATH)
    measurements = report["measurements"]
    runs = measurements["runs"]
    statistics = measurements["statistics"]

    assert measurements["independent_fresh_processes"] is True
    assert measurements["repeat_byte_identity"] is True
    assert set(runs) == {"first", "repeat"}
    for run in runs.values():
        assert set(run) == {
            "artifact_sha256",
            "compute_seconds",
            "maximum_rss_bytes",
            "total_wall_seconds",
        }
        assert run["artifact_sha256"] == ARTIFACT_SHA256
        assert math.isfinite(run["compute_seconds"])
        assert math.isfinite(run["total_wall_seconds"])
        assert 0.0 < run["compute_seconds"] < run["total_wall_seconds"]
        assert type(run["maximum_rss_bytes"]) is int
        assert (
            0
            < run["maximum_rss_bytes"]
            < training["resource_budgets"]["cache_build"]["peak_RSS_bytes_maximum"]
        )
        assert (
            run["total_wall_seconds"]
            < training["resource_budgets"]["cache_build"]["wall_seconds_maximum"]
        )

    contents = ARTIFACT_PATH.read_bytes()
    means = struct.unpack("<40f", contents[:160])
    deviations = struct.unpack("<40f", contents[160:])
    assert statistics == {
        "means": {
            "count": MEL_BINS,
            "maximum": max(means),
            "maximum_float32_hex": float(max(means)).hex(),
            "minimum": min(means),
            "minimum_float32_hex": float(min(means)).hex(),
            "slice_sha256": MEANS_SHA256,
        },
        "population_standard_deviations": {
            "count": MEL_BINS,
            "maximum": max(deviations),
            "maximum_float32_hex": float(max(deviations)).hex(),
            "minimum": min(deviations),
            "minimum_float32_hex": float(min(deviations)).hex(),
            "slice_sha256": DEVIATIONS_SHA256,
        },
    }
    assert report["data_boundary"] == {
        "augmentation_rng_calls_during_accumulation": 0,
        "cache_open_eager_background_payloads": True,
        "cache_open_full_file_hash_verified": True,
        "normalization_augmentation_enabled": False,
        "normalization_background_clips_accumulated": 0,
        "normalization_test_clips_accumulated": 0,
        "normalization_validation_clips_accumulated": 0,
        "test_command_count_only": 11_005,
        "test_identities_retained": False,
        "training_background_metadata_checked": True,
        "training_command_clips_accumulated": REGISTERED_CLIP_COUNT,
    }

    source = Path("src/falsewake/experiment_002_normalization_artifact.py").read_text(
        encoding="utf-8"
    )
    assert "sorted(corpus.train_commands" in source
    assert 'training.split != "train"' in source
    assert "read_command_pcm16le" in source
    assert "read_background_window_pcm16le" not in source
    assert "test_commands" not in source
    assert "experiment_002_rng" not in source

    cache_source = Path("src/falsewake/experiment_002_pcm_cache.py").read_text(
        encoding="utf-8"
    )
    assert "whole_sha256, payload_sha256 = _hash_verified_file(" in cache_source
    assert "background_payloads = _read_background_payloads(" in cache_source

    document = Path("docs/experiment-002.md").read_text(encoding="utf-8")
    assert "## Frozen normalization artifact" in document
    assert ARTIFACT_SHA256 in document
    assert "produced the same 320" in document
    assert "84,843 training commands and 8,314,614 frontend frames" in document
    assert "digest from the committed report" in document
    assert "does not\nitself pin one artifact digest" in document
    assert "eagerly materializes" in document
    assert "contribute zero normalization" in document
    assert "misreporting cache-verification I/O as absent" in document
