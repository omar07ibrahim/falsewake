from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

from falsewake import holdout

REPORT_PATH = Path("reports/experiment-001-dev-replay.json")
SELECTION_PATH = Path("reports/experiment-001-selection.json")
REPRODUCIBILITY_PATH = Path("reports/experiment-001-reproducibility.json")

REPORT_SHA256 = "b8e30e26498af2600ece01f4cceb436e10a546b8390f039ca8a23cda45f00f2d"
SELECTION_SHA256 = (
    "1d44ae6ff06a5fab1567d0342299e293fe001b8c91f9972cfa8e79a2dabf2318"
)


def _load_canonical_json(path: Path) -> tuple[bytes, dict[str, object]]:
    contents = path.read_bytes()
    decoded = json.loads(contents)
    assert isinstance(decoded, dict)
    document = cast(dict[str, object], decoded)
    canonical = (
        json.dumps(
            document,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )
    assert contents == canonical
    return contents, document


def test_checked_in_replay_is_exact_canonical_and_rejected() -> None:
    report, _ = _load_canonical_json(REPORT_PATH)

    assert hashlib.sha256(report).hexdigest() == REPORT_SHA256
    assert holdout.recompute_selection(report) == ("reject", None)


def test_checked_in_selection_matches_the_rejected_replay() -> None:
    report = REPORT_PATH.read_bytes()
    selection_bytes, selection_document = _load_canonical_json(SELECTION_PATH)
    selection = holdout._parse_selection_artifact(selection_bytes)

    assert hashlib.sha256(selection_bytes).hexdigest() == SELECTION_SHA256
    assert selection_document == selection
    assert set(selection) == holdout.SELECTION_ARTIFACT_FIELDS
    assert selection["schema_version"] == holdout.SELECTION_SCHEMA_VERSION
    assert selection["status"] == "reject"
    assert selection["selected_threshold_milli"] is None
    assert selection["dev_replay_report_sha256"] == hashlib.sha256(
        report
    ).hexdigest()


def test_reproducibility_record_binds_two_independent_identical_runs() -> None:
    _, reproducibility = _load_canonical_json(REPRODUCIBILITY_PATH)
    selection = cast(
        dict[str, object], json.loads(SELECTION_PATH.read_bytes())
    )

    assert set(reproducibility) == {
        "byte_identical",
        "implementation_git_commit",
        "report_sha256",
        "runs",
        "runtime_identity_sha256",
        "schema_version",
        "selected_threshold_milli",
        "selection_artifact_sha256",
        "status",
        "thread_limits",
    }
    assert reproducibility["schema_version"] == 1
    assert reproducibility["byte_identical"] is True
    assert reproducibility["report_sha256"] == REPORT_SHA256
    assert reproducibility["selection_artifact_sha256"] == SELECTION_SHA256
    assert reproducibility["status"] == selection["status"] == "reject"
    assert reproducibility["selected_threshold_milli"] is None
    assert reproducibility["implementation_git_commit"] == selection[
        "implementation_git_commit"
    ]
    assert reproducibility["runtime_identity_sha256"] == selection[
        "runtime_identity_sha256"
    ]

    raw_runs = reproducibility["runs"]
    assert isinstance(raw_runs, list)
    runs = cast(list[dict[str, object]], raw_runs)
    assert len(runs) == 2
    assert {run["input_copy_set"] for run in runs} == {
        "primary",
        "independent_byte_identical_copies",
    }
    assert len({run["python_hash_seed"] for run in runs}) == 2
    for run in runs:
        assert set(run) == {
            "elapsed_seconds",
            "input_copy_set",
            "max_rss_kib",
            "python_hash_seed",
            "report_sha256",
            "selection_artifact_sha256",
            "user_cpu_seconds",
        }
        assert run["report_sha256"] == REPORT_SHA256
        assert run["selection_artifact_sha256"] == SELECTION_SHA256
