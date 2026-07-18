from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import cast

import pytest

ARTIFACT_PATH = Path("reports/experiment-001-analysis.artifact.json")
HTML_PATH = Path("reports/experiment-001-analysis.html")
VERIFICATION_PATH = Path("reports/experiment-001-analysis-verification.json")
REPLAY_PATH = Path("reports/experiment-001-dev-replay.json")
REPRODUCIBILITY_PATH = Path("reports/experiment-001-reproducibility.json")
CONFIG_PATH = Path("configs/experiment-001.json")

ARTIFACT_SHA256 = (
    "a328da94bbea7f72a1271059560140b57fc2409348786659a944ccdbc34c6863"
)
HTML_SHA256 = "5ed3a2423c7f3901a8d540afd96497316a29147f487dc88e2997e33a3fc7fadb"


def _load_document(path: Path) -> dict[str, object]:
    decoded = json.loads(path.read_bytes())
    assert isinstance(decoded, dict)
    return cast(dict[str, object], decoded)


def _mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _rows(value: object) -> list[dict[str, object]]:
    assert isinstance(value, list)
    assert all(isinstance(row, dict) for row in value)
    return cast(list[dict[str, object]], value)


def _projection_sources(
    artifact: dict[str, object],
) -> dict[str, dict[str, object]]:
    raw_sources = artifact["sources"]
    assert isinstance(raw_sources, list)
    sources = cast(list[dict[str, object]], raw_sources)
    indexed = {cast(str, source["id"]): source for source in sources}
    assert len(indexed) == len(sources)
    return indexed


def _execute_projection(
    source: dict[str, object], bindings: dict[str, str]
) -> list[dict[str, object]]:
    query = _mapping(source["query"])
    assert query["engine"] == "sqlite-json1"
    sql = query["sql"]
    assert isinstance(sql, str) and sql.lstrip().upper().startswith("WITH")
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in connection.execute(sql, bindings).fetchall()]
    finally:
        connection.close()


def _assert_semantic_rows_equal(
    actual: list[dict[str, object]], expected: list[dict[str, object]]
) -> None:
    assert len(actual) == len(expected)
    for actual_row, expected_row in zip(actual, expected, strict=True):
        assert actual_row.keys() == expected_row.keys()
        for field, expected_value in expected_row.items():
            actual_value = actual_row[field]
            if isinstance(expected_value, float):
                assert isinstance(actual_value, float)
                assert actual_value == pytest.approx(
                    expected_value, rel=1e-15, abs=1e-12
                )
            else:
                assert actual_value == expected_value


def test_portable_analysis_is_exact_validated_and_self_contained() -> None:
    artifact_contents = ARTIFACT_PATH.read_bytes()
    artifact = _load_document(ARTIFACT_PATH)
    html = HTML_PATH.read_bytes()

    assert hashlib.sha256(artifact_contents).hexdigest() == ARTIFACT_SHA256
    assert hashlib.sha256(html).hexdigest() == HTML_SHA256
    assert artifact["surface"] == "report"

    manifest = _mapping(artifact["manifest"])
    assert manifest["title"] == "FalseWake Experiment 001: Continuous Replay Result"
    assert len(cast(list[object], manifest["cards"])) == 5
    assert len(cast(list[object], manifest["charts"])) == 2
    assert len(cast(list[object], manifest["tables"])) == 1

    html_text = html.decode("utf-8")
    assert "data-analytics-portable-fallback" in html_text
    assert "The registered linear baseline is not deployable" in html_text
    assert "10,147" in html_text
    assert "false events" in html_text
    assert not re.search(
        r"<(?:img|script)\b[^>]*\bsrc\s*=\s*['\"]https?://", html_text, re.I
    )
    assert not re.search(
        r"<link\b[^>]*\bhref\s*=\s*['\"]https?://", html_text, re.I
    )

    verification_contents = VERIFICATION_PATH.read_bytes()
    verification = _load_document(VERIFICATION_PATH)
    assert verification_contents == json.dumps(
        verification,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"
    assert verification["artifact_sha256"] == ARTIFACT_SHA256
    assert verification["html_sha256"] == HTML_SHA256
    assert verification["repeated_build_byte_identical"] is True
    stages = _mapping(verification["stages"])
    assert stages == {
        "package": "passed",
        "validation": "passed",
        "verification": "structural_only",
    }
    source_reconciliation = _mapping(
        verification["source_query_reconciliation"]
    )
    assert source_reconciliation["query_count"] == 5
    assert source_reconciliation["dataset_row_count"] == 1_016
    assert source_reconciliation["status"] == "passed"


def test_embedded_sql_reproduces_every_exposed_dataset_row() -> None:
    artifact = _load_document(ARTIFACT_PATH)
    sources = _projection_sources(artifact)
    snapshot = _mapping(artifact["snapshot"])
    datasets = _mapping(snapshot["datasets"])
    replay_json = REPLAY_PATH.read_text(encoding="ascii")
    config_json = CONFIG_PATH.read_text(encoding="ascii")
    reproducibility_json = REPRODUCIBILITY_PATH.read_text(encoding="ascii")

    expected_summary = dict(_rows(datasets["summary"])[0])
    expected_replays = expected_summary.pop("byte_identical_replays")
    summary = _execute_projection(
        sources["summary-projection"],
        {"replay_json": replay_json, "config_json": config_json},
    )
    _assert_semantic_rows_equal(summary, [expected_summary])
    reproducibility = _execute_projection(
        sources["reproducibility-projection"],
        {"reproducibility_json": reproducibility_json},
    )
    _assert_semantic_rows_equal(
        reproducibility, [{"byte_identical_replays": expected_replays}]
    )

    for source_id, dataset_name in (
        ("threshold-gates-projection", "threshold_gates"),
        ("target-events-projection", "target_events"),
        ("frontier-checkpoints-projection", "frontier_checkpoints"),
    ):
        actual = _execute_projection(
            sources[source_id], {"replay_json": replay_json}
        )
        expected = _rows(datasets[dataset_name])
        _assert_semantic_rows_equal(actual, expected)

    assert len(_rows(datasets["threshold_gates"])) == 1_001
    assert len(_rows(datasets["target_events"])) == 10
    assert len(_rows(datasets["frontier_checkpoints"])) == 4
