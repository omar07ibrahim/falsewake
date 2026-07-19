from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from falsewake.experiment_002_pcm_cache import (
    ENTRY_BYTES,
    HEADER_BYTES,
    PAYLOAD_ALIGNMENT,
    REGISTERED_ENTRY_COUNT,
    REGISTERED_FILE_BYTES,
    REGISTERED_PAYLOAD_BYTES,
    REGISTERED_PAYLOAD_OFFSET,
)

REPORT_PATH = Path("reports/experiment-002-pcm-cache.json")


def _report() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(REPORT_PATH.read_text(encoding="utf-8")))


def test_pcm_cache_report_is_canonical_and_bound_to_the_implementation() -> None:
    raw = REPORT_PATH.read_text(encoding="utf-8")
    report = json.loads(raw)

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
        "budget",
        "builder",
        "data_boundary",
        "experiment",
        "format",
        "inputs",
        "schema_version",
        "verification",
    }
    assert report["schema_version"] == 1
    assert report["experiment"] == "002"
    assert report["builder"] == {
        "git_commit": "0c460cd1060acc5f96c856ec4923cc885713e97e",
        "numpy": "2.5.1",
        "python": "3.12.3",
        "source": "src/falsewake/experiment_002_pcm_cache.py",
    }


def test_reported_layout_matches_the_frozen_binary_format() -> None:
    report = _report()
    layout = report["format"]
    artifact = report["artifact"]

    assert layout == {
        "entry_bytes": ENTRY_BYTES,
        "entry_count": REGISTERED_ENTRY_COUNT,
        "header_bytes": HEADER_BYTES,
        "payload_alignment_bytes": PAYLOAD_ALIGNMENT,
        "payload_bytes": REGISTERED_PAYLOAD_BYTES,
        "payload_offset": REGISTERED_PAYLOAD_OFFSET,
        "role_counts": {
            "train_backgrounds": 4,
            "train_commands": 84_843,
            "validation_backgrounds": 1,
            "validation_commands": 9_981,
        },
    }
    assert artifact["byte_count"] == REGISTERED_FILE_BYTES
    assert artifact["mode"] == "0444"
    assert artifact["committed_to_git"] is False
    assert artifact["role"].endswith("not_a_preregistered_trust_root")
    assert len(artifact["sha256"]) == 64


def test_reported_build_is_reproducible_and_within_every_budget() -> None:
    report = _report()
    artifact = report["artifact"]
    budget = report["budget"]
    verification = report["verification"]

    assert artifact["byte_count"] < budget["output_bytes_maximum"]
    assert (
        budget["observed_first_build_max_rss_bytes"] < budget["peak_rss_bytes_maximum"]
    )
    assert budget["observed_first_build_wall_seconds"] < budget["wall_seconds_maximum"]
    assert verification["first_build_sha256"] == artifact["sha256"]
    assert verification["repeat_build_sha256"] == artifact["sha256"]
    assert verification["repeat_byte_identity"] is True
    assert verification["external_identity_required_to_open"] is True
    assert verification["secure_wav_loader_parity"] is True
    assert verification["command_endpoint_comparisons"] == 4
    assert verification["background_boundary_comparisons"] == 10
    assert verification["parity_comparison_count"] == 14


def test_report_retains_only_the_registered_test_count_boundary() -> None:
    boundary = _report()["data_boundary"]

    assert boundary == {
        "cached_test_audio": False,
        "retained_test_identity": False,
        "test_command_count_only": 11_005,
        "white_noise_test_background_cached": False,
    }
    raw = REPORT_PATH.read_text(encoding="utf-8")
    assert "white_noise.wav" not in raw
    assert '"test_command_count_only": 11005' in raw
