from __future__ import annotations

from pathlib import Path

import pytest

from falsewake.result_plots import (
    ResultPlotError,
    class_recall_data,
    load_metrics,
    open_set_data,
    render_result_plots,
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


def test_plot_loader_rejects_threshold_results(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    path.write_text(
        '{"class_order": ["yes"], "threshold_metrics": "evaluated"}',
        encoding="utf-8",
    )
    with pytest.raises(ResultPlotError, match="class order"):
        load_metrics(path)
