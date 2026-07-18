"""Render the reviewed experiment 000 clip metrics as static figures."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from falsewake.linear_baseline import CLASS_ORDER

BLUE = "#3A6EA5"
BLUE_DARK = "#234B73"
ORANGE = "#D97706"
NEUTRAL = "#6B7280"
INK = "#1F2937"
MUTED = "#5F6B7A"
GRID = "#D9DEE7"
BACKGROUND = "#FFFFFF"


class ResultPlotError(ValueError):
    """The metrics document cannot support the registered figures."""


@dataclass(frozen=True, slots=True)
class ClassRecall:
    label: str
    validation: float
    validation_support: int
    test: float
    test_support: int


@dataclass(frozen=True, slots=True)
class OpenSetRate:
    label: str
    rate: float
    count: int
    support: int
    color: str
    validation: bool


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ResultPlotError(f"{name} is not an object")
    return cast(dict[str, object], value)


def _list(value: object, *, name: str) -> list[object]:
    if type(value) is not list:
        raise ResultPlotError(f"{name} is not a list")
    return cast(list[object], value)


def _number(value: object, *, name: str) -> float:
    if type(value) not in {int, float}:
        raise ResultPlotError(f"{name} is not numeric")
    result = float(cast(float | int, value))
    if not np.isfinite(result):
        raise ResultPlotError(f"{name} is not finite")
    return result


def _integer(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ResultPlotError(f"{name} is not a non-negative integer")
    return value


def load_metrics(path: Path) -> dict[str, object]:
    """Load the standards-safe registered result document."""

    try:
        document = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ResultPlotError(f"cannot load metrics: {error}") from error
    metrics = _mapping(document, name="metrics")
    if metrics.get("class_order") != list(CLASS_ORDER):
        raise ResultPlotError("metrics use an unexpected class order")
    if metrics.get("threshold_metrics") != "not_evaluated":
        raise ResultPlotError("clip figures require unevaluated threshold metrics")
    return metrics


def class_recall_data(metrics: dict[str, object]) -> tuple[ClassRecall, ...]:
    """Return class-order recall and support for both reported splits."""

    splits = _mapping(metrics.get("splits"), name="splits")
    indexed: dict[str, dict[str, dict[str, object]]] = {}
    for split in ("validation", "test"):
        split_result = _mapping(splits.get(split), name=split)
        rows = _list(split_result.get("per_class"), name=f"{split}.per_class")
        parsed: dict[str, dict[str, object]] = {}
        for raw_row in rows:
            row = _mapping(raw_row, name=f"{split}.per_class row")
            label = row.get("label")
            if type(label) is not str or label in parsed:
                raise ResultPlotError(f"{split} has an invalid class label")
            parsed[label] = row
        if tuple(parsed) != CLASS_ORDER:
            raise ResultPlotError(f"{split} per-class rows are out of order")
        indexed[split] = parsed

    result_rows: list[ClassRecall] = []
    for label in CLASS_ORDER:
        validation = indexed["validation"][label]
        test = indexed["test"][label]
        validation_recall = _number(
            validation.get("recall"), name=f"validation.{label}.recall"
        )
        test_recall = _number(test.get("recall"), name=f"test.{label}.recall")
        if not 0 <= validation_recall <= 1 or not 0 <= test_recall <= 1:
            raise ResultPlotError("recall falls outside [0, 1]")
        result_rows.append(
            ClassRecall(
                label=label,
                validation=validation_recall,
                validation_support=_integer(
                    validation.get("support"), name=f"validation.{label}.support"
                ),
                test=test_recall,
                test_support=_integer(
                    test.get("support"), name=f"test.{label}.support"
                ),
            )
        )
    return tuple(result_rows)


def open_set_data(metrics: dict[str, object]) -> tuple[OpenSetRate, ...]:
    """Return the four registered open-set target-prediction rates."""

    splits = _mapping(metrics.get("splits"), name="splits")
    rows: list[OpenSetRate] = []
    for label, color in (("unknown", ORANGE), ("silence", NEUTRAL)):
        for split in ("validation", "test"):
            result = _mapping(splits.get(split), name=split)
            open_set = _mapping(
                result.get("open_set_target_prediction"), name=f"{split}.open_set"
            )
            values = _mapping(open_set.get(label), name=f"{split}.{label}")
            rate = _number(
                values.get("predicted_target_rate"), name=f"{split}.{label}.rate"
            )
            count = _integer(
                values.get("predicted_target_count"), name=f"{split}.{label}.count"
            )
            support = _integer(values.get("support"), name=f"{split}.{label}.support")
            if support == 0 or count > support or not 0 <= rate <= 1:
                raise ResultPlotError(f"{split}.{label} has an invalid rate base")
            if not np.isclose(rate, count / support, rtol=0, atol=1e-15):
                raise ResultPlotError(f"{split}.{label} rate does not match its counts")
            rows.append(
                OpenSetRate(
                    label=f"{split.capitalize()} · {label}",
                    rate=rate,
                    count=count,
                    support=support,
                    color=color,
                    validation=split == "validation",
                )
            )
    return tuple(rows)


def _style_axis(axis: Axes) -> None:
    axis.set_facecolor(BACKGROUND)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(MUTED)
    axis.tick_params(colors=INK, labelsize=9)
    axis.xaxis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)


def _save(figure: Figure, output_stem: Path) -> None:
    figure.savefig(
        output_stem.with_suffix(".png"),
        dpi=180,
        facecolor=BACKGROUND,
        metadata={"Software": "FalseWake experiment 000"},
    )
    figure.savefig(
        output_stem.with_suffix(".svg"),
        facecolor=BACKGROUND,
        metadata={"Creator": "FalseWake experiment 000", "Date": None},
    )
    plt.close(figure)


def plot_class_recall(rows: Sequence[ClassRecall], output_stem: Path) -> None:
    """Render validation/test recall with a shared zero-based scale."""

    if tuple(row.label for row in rows) != CLASS_ORDER:
        raise ResultPlotError("class recall rows do not use the registered order")
    figure, axis = plt.subplots(figsize=(9.2, 7.0), layout="constrained")
    y = np.arange(len(rows), dtype=np.float64)
    validation = np.asarray([row.validation for row in rows])
    test = np.asarray([row.test for row in rows])
    axis.barh(
        y - 0.18,
        validation,
        height=0.31,
        facecolor=BACKGROUND,
        edgecolor=BLUE_DARK,
        hatch="///",
        linewidth=1.1,
        label="Validation",
    )
    axis.barh(
        y + 0.18,
        test,
        height=0.31,
        color=BLUE,
        edgecolor=BLUE_DARK,
        linewidth=0.8,
        label="Test",
    )
    for position, value in zip(y + 0.18, test, strict=True):
        axis.text(
            min(value + 0.012, 0.985),
            position,
            f"{value:.0%}",
            ha="right" if value > 0.95 else "left",
            va="center",
            color=INK,
            fontsize=8,
        )
    axis.set_yticks(y, [row.label for row in rows])
    axis.invert_yaxis()
    axis.set_xlim(0, 1)
    axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.set_xlabel("Recall", color=INK, fontsize=10)
    axis.set_title(
        "Per-class recall — experiment 000\n"
        "Speech Commands v0.02; validation n=4,429, test n=4,884; argmax clips",
        loc="left",
        color=INK,
        fontsize=14,
        pad=16,
    )
    axis.legend(frameon=False, loc="lower right", fontsize=9)
    axis.text(
        0,
        -0.12,
        "Test percentages are labeled. No acceptance threshold or continuous "
        "stream is evaluated.",
        transform=axis.transAxes,
        color=MUTED,
        fontsize=8.5,
    )
    _style_axis(axis)
    _save(figure, output_stem)


def plot_open_set(rows: Sequence[OpenSetRate], output_stem: Path) -> None:
    """Render target-prediction rates with visible denominators and caveats."""

    if len(rows) != 4:
        raise ResultPlotError("open-set figure requires four split/class rows")
    figure, axis = plt.subplots(figsize=(9.2, 4.8), layout="constrained")
    y = np.arange(len(rows), dtype=np.float64)
    bars = axis.barh(
        y,
        [row.rate for row in rows],
        height=0.58,
        color=[BACKGROUND if row.validation else row.color for row in rows],
        edgecolor=[row.color for row in rows],
        linewidth=1.4,
    )
    for bar, row in zip(bars, rows, strict=True):
        if row.validation:
            bar.set_hatch("///")
        axis.text(
            max(row.rate + 0.018, 0.018),
            bar.get_y() + bar.get_height() / 2,
            f"{row.count}/{row.support} · {row.rate:.1%}",
            ha="left",
            va="center",
            color=INK,
            fontsize=9,
            fontweight="semibold",
        )
    axis.set_yticks(y, [row.label for row in rows])
    axis.invert_yaxis()
    axis.set_xlim(0, 1)
    axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.set_xlabel(
        "Predicted as any of the ten target commands", color=INK, fontsize=10
    )
    axis.set_title(
        "Open-set clips predicted as a target command\n"
        "Speech Commands v0.02; argmax clips; validation is hatched",
        loc="left",
        color=INK,
        fontsize=14,
        pad=16,
    )
    axis.text(
        0,
        -0.19,
        "Silence uses overlapping windows from one noise file per holdout split. "
        "This is not false accepts/hour.",
        transform=axis.transAxes,
        color=MUTED,
        fontsize=8.5,
    )
    _style_axis(axis)
    _save(figure, output_stem)


def render_result_plots(metrics_path: Path, output_directory: Path) -> None:
    """Render both registered result figures into an existing directory."""

    if not output_directory.is_dir():
        raise ResultPlotError(f"output directory does not exist: {output_directory}")
    metrics = load_metrics(metrics_path)
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
            "svg.hashsalt": "falsewake-experiment-000",
        }
    )
    plot_class_recall(
        class_recall_data(metrics), output_directory / "experiment-000-class-recall"
    )
    plot_open_set(open_set_data(metrics), output_directory / "experiment-000-open-set")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render the reviewed experiment 000 clip figures."
    )
    parser.add_argument(
        "metrics",
        type=Path,
        nargs="?",
        default=Path("reports/experiment-000-linear.json"),
    )
    parser.add_argument(
        "output_directory", type=Path, nargs="?", default=Path("reports")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Render both result figures from the exact checked-in metrics."""

    arguments = _parser().parse_args(argv)
    try:
        render_result_plots(arguments.metrics, arguments.output_directory)
    except ResultPlotError as error:
        print(f"result plotting failed: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
