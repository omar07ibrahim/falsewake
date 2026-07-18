from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from falsewake import holdout
from falsewake.result_plots import (
    FEASIBILITY_NEGATIVE_FRONTIER_MILLI,
    FEASIBILITY_QUESTION,
    FEASIBILITY_RETENTION_FRONTIER_MILLI,
    FEASIBILITY_SUBTITLE,
    FEASIBILITY_TAKEAWAY,
    ResultPlotError,
    class_recall_data,
    load_metrics,
    load_replay_feasibility,
    open_set_data,
    render_replay_feasibility,
    render_result_plots,
)


def _canonical_json(document: object) -> bytes:
    return (
        json.dumps(
            document,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _registered_replay_report() -> bytes:
    exposure = 308_310_400
    event_counts = [6 if threshold <= 990 else 5 for threshold in range(1_001)]
    baseline_by_target = list(holdout.BASELINE_CORRECT_BY_TARGET)
    retained_by_target = [289, 178, 193, 195, 185, 172, 231, 220, 8, 0]
    failed_by_target = [289, 178, 193, 195, 185, 172, 231, 220, 7, 0]
    thresholds: list[dict[str, object]] = []
    for threshold, events in enumerate(event_counts):
        if threshold == 0:
            correct_by_target = baseline_by_target
        elif threshold <= FEASIBILITY_RETENTION_FRONTIER_MILLI:
            correct_by_target = retained_by_target
        else:
            correct_by_target = failed_by_target
        correct = sum(correct_by_target)
        thresholds.append(
            {
                "threshold_milli": threshold,
                "dev_event_count": events,
                "dev_event_count_by_target": [events, *([0] * 9)],
                "false_events_per_hour": events * 57_600_000 / exposure,
                "garwood_95_percent": [0.0, 0.0],
                "speaker_bootstrap_95_percent": [0.0, 0.0],
                "validation_correct_accept_count": correct,
                "validation_correct_accept_count_by_target": correct_by_target,
                "correct_accept_recall": correct / holdout.TARGET_EXAMPLE_COUNT,
                "conditional_correct_retention": (
                    correct / holdout.BASELINE_CORRECT_COUNT
                ),
            }
        )

    speaker_rows: list[dict[str, object]] = []
    for speaker_id in range(1, 41):
        speaker_rows.append(
            {
                "speaker_id": speaker_id,
                "utterance_count": 1 if speaker_id < 40 else 2_664,
                "scored_exposure_samples": (
                    1 if speaker_id < 40 else exposure - 39
                ),
                "event_count": event_counts if speaker_id == 1 else [0] * 1_001,
            }
        )

    identities = {
        field: ("1" * 40 if field == "implementation_git_commit" else "a" * 64)
        for field in holdout.REPLAY_IDENTITY_FIELDS
    }
    return _canonical_json(
        {
            "schema_version": 1,
            "identities": identities,
            "runtime": holdout.current_runtime_identity(),
            "target_order": list(holdout.TARGET_ORDER),
            "dev": {
                "source_samples": 310_337_932,
                "scored_exposure_samples": exposure,
                "utterance_count": 2_703,
                "speaker_count": 40,
                "speaker_rows": speaker_rows,
            },
            "positive_validation": {
                "target_example_count": holdout.TARGET_EXAMPLE_COUNT,
                "baseline_correct_count": holdout.BASELINE_CORRECT_COUNT,
                "baseline_correct_by_target": baseline_by_target,
                "support_by_target": list(holdout.TARGET_SUPPORT),
            },
            "thresholds": thresholds,
            "selection": {
                "status": "reject",
                "selected_threshold_milli": None,
                "negative_frontier_milli": (
                    FEASIBILITY_NEGATIVE_FRONTIER_MILLI
                ),
                "retention_frontier_milli": (
                    FEASIBILITY_RETENTION_FRONTIER_MILLI
                ),
            },
            "top_false_events": [
                {"threshold_milli": threshold, "events": []}
                for threshold in (
                    0,
                    FEASIBILITY_RETENTION_FRONTIER_MILLI,
                    FEASIBILITY_NEGATIVE_FRONTIER_MILLI,
                )
            ],
        }
    )


def test_registered_plot_data_preserves_rates_and_denominators() -> None:
    metrics = load_metrics(Path("reports/experiment-000-linear.json"))
    recalls = class_recall_data(metrics)
    open_set = open_set_data(metrics)

    assert len(recalls) == 12
    assert recalls[0].label == "yes"
    assert recalls[0].test == pytest.approx(0.8257756563245824)
    assert recalls[-2].label == "unknown"
    assert recalls[-2].test_support == 405
    assert [(row.count, row.support) for row in open_set] == [
        (293, 363),
        (329, 405),
        (0, 363),
        (0, 405),
    ]


def test_static_plots_render_with_titles_and_without_remote_assets(
    tmp_path: Path,
) -> None:
    render_result_plots(Path("reports/experiment-000-linear.json"), tmp_path)

    for name in ("experiment-000-class-recall", "experiment-000-open-set"):
        png = tmp_path / f"{name}.png"
        svg = tmp_path / f"{name}.svg"
        assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        contents = svg.read_text(encoding="utf-8")
        assert "<svg" in contents
        assert "http://www.w3.org/2000/svg" in contents
        assert "https://" not in contents
        assert all(line == line.rstrip() for line in contents.splitlines())


def test_plot_loader_rejects_threshold_results(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    path.write_text(
        '{"class_order": ["yes"], "threshold_metrics": "evaluated"}',
        encoding="utf-8",
    )
    with pytest.raises(ResultPlotError, match="class order"):
        load_metrics(path)


def test_replay_loader_preserves_strict_registered_gates_and_frontiers(
    tmp_path: Path,
) -> None:
    report = tmp_path / "experiment-001-dev-replay.json"
    report.write_bytes(_registered_replay_report())

    data = load_replay_feasibility(report)

    assert len(data.thresholds) == 1_001
    assert data.thresholds[0] == 0
    assert data.thresholds[-1] == 1
    assert data.utterance_count == 2_703
    assert data.target_clip_count == 3_703
    assert data.scored_exposure_hours == pytest.approx(5.352611111111111)
    assert data.retention_frontier_milli == 395
    assert data.negative_frontier_milli == 991
    assert data.retention_gate[395]
    assert not data.retention_gate[396]
    assert not data.false_event_gate[990]
    assert data.false_event_gate[991]
    assert not any(
        retained and negative
        for retained, negative in zip(
            data.retention_gate, data.false_event_gate, strict=True
        )
    )


def test_replay_loader_rejects_forged_frontier_and_wrong_cohort(
    tmp_path: Path,
) -> None:
    document = cast(dict[str, object], json.loads(_registered_replay_report()))
    selection = cast(dict[str, object], document["selection"])
    selection["negative_frontier_milli"] = 990
    report = tmp_path / "forged-frontier.json"
    report.write_bytes(_canonical_json(document))
    with pytest.raises(ResultPlotError, match="reported selection"):
        load_replay_feasibility(report)

    document = cast(dict[str, object], json.loads(_registered_replay_report()))
    dev = cast(dict[str, object], document["dev"])
    dev["utterance_count"] = 2_704
    speaker_rows = cast(list[dict[str, object]], dev["speaker_rows"])
    speaker_rows[-1]["utterance_count"] = 2_665
    report = tmp_path / "wrong-cohort.json"
    report.write_bytes(_canonical_json(document))
    with pytest.raises(ResultPlotError, match="unexpected experiment-001 cohort"):
        load_replay_feasibility(report)


def test_replay_feasibility_svg_is_deterministic_and_has_no_remote_assets(
    tmp_path: Path,
) -> None:
    report = tmp_path / "experiment-001-dev-replay.json"
    report.write_bytes(_registered_replay_report())
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    render_replay_feasibility(report, first)
    render_replay_feasibility(report, second)

    first_png = first / "experiment-001-gate-feasibility.png"
    first_svg = first / "experiment-001-gate-feasibility.svg"
    second_svg = second / "experiment-001-gate-feasibility.svg"
    assert first_png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    contents = first_svg.read_text(encoding="utf-8")
    assert first_svg.read_bytes() == second_svg.read_bytes()
    assert FEASIBILITY_QUESTION in contents
    assert FEASIBILITY_SUBTITLE in contents
    assert FEASIBILITY_TAKEAWAY in contents
    assert "Retention frontier 0.395" in contents
    assert "False-event frontier 0.991" in contents
    assert "No registered threshold overlaps both pass regions." in contents
    assert "https://" not in contents
    assert "<image" not in contents
    assert 'xlink:href="http' not in contents
    assert all(line == line.rstrip() for line in contents.splitlines())


def test_replay_feasibility_requires_existing_output_directory(
    tmp_path: Path,
) -> None:
    report = tmp_path / "experiment-001-dev-replay.json"
    report.write_bytes(_registered_replay_report())

    with pytest.raises(ResultPlotError, match="output directory does not exist"):
        render_replay_feasibility(report, tmp_path / "missing")
