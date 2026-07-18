"""Render the reviewed experiment 000 and 001 results as static figures."""

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

from falsewake import holdout
from falsewake.linear_baseline import CLASS_ORDER

BLUE = "#3A6EA5"
BLUE_DARK = "#234B73"
ORANGE = "#D97706"
NEUTRAL = "#6B7280"
INK = "#1F2937"
MUTED = "#5F6B7A"
GRID = "#D9DEE7"
BACKGROUND = "#FFFFFF"
FEASIBILITY_BACKGROUND = "#FAFBFD"

SAMPLES_PER_HOUR = 57_600_000
FEASIBILITY_UTTERANCE_COUNT = 2_703
FEASIBILITY_EXPOSURE_SAMPLES = 308_310_400
FEASIBILITY_TARGET_CLIP_COUNT = 3_703
FEASIBILITY_RETENTION_FRONTIER_MILLI = 395
FEASIBILITY_NEGATIVE_FRONTIER_MILLI = 991
FEASIBILITY_QUESTION = "Does any registered threshold meet both gates?"
FEASIBILITY_SUBTITLE = (
    "dev-clean 2,703 utterances / 5.35 scored hours; "
    "Speech Commands validation 3,703 target clips"
)
FEASIBILITY_TAKEAWAY = (
    "No\N{EM DASH}the \N{GREATER-THAN OR EQUAL TO}80% retention gate ends at "
    "0.395 while the \N{LESS-THAN OR EQUAL TO}1 false-event/hour gate begins "
    "at 0.991."
)


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


@dataclass(frozen=True, slots=True)
class FeasibilityData:
    """The exact registered gate states used by the experiment-001 figure."""

    thresholds: tuple[float, ...]
    retention_gate: tuple[bool, ...]
    false_event_gate: tuple[bool, ...]
    retention_frontier_milli: int
    negative_frontier_milli: int
    utterance_count: int
    scored_exposure_samples: int
    target_clip_count: int

    @property
    def scored_exposure_hours(self) -> float:
        """Return the registered development exposure in scored hours."""

        return self.scored_exposure_samples / SAMPLES_PER_HOUR


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


def _load_bounded_replay_report(path: Path) -> bytes:
    try:
        with path.open("rb") as stream:
            contents = stream.read(holdout.MAX_REPLAY_REPORT_BYTES + 1)
    except OSError as error:
        raise ResultPlotError(f"cannot load replay report: {error}") from error
    if len(contents) > holdout.MAX_REPLAY_REPORT_BYTES:
        raise ResultPlotError("replay report exceeds the registered size limit")
    return contents


def load_replay_feasibility(path: Path) -> FeasibilityData:
    """Load and verify the canonical registered experiment-001 replay result."""

    contents = _load_bounded_replay_report(path)
    try:
        recomputed_status, recomputed_threshold = holdout.recompute_selection(contents)
    except holdout.HoldoutAccessError as error:
        raise ResultPlotError(f"cannot validate replay report: {error}") from error
    try:
        document = _mapping(json.loads(contents), name="replay report")
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ResultPlotError(
            f"cannot decode validated replay report: {error}"
        ) from error

    dev = _mapping(document.get("dev"), name="replay report dev")
    positive = _mapping(
        document.get("positive_validation"), name="replay report positive validation"
    )
    utterance_count = _integer(
        dev.get("utterance_count"), name="dev.utterance_count"
    )
    exposure = _integer(
        dev.get("scored_exposure_samples"), name="dev.scored_exposure_samples"
    )
    target_clip_count = _integer(
        positive.get("target_example_count"),
        name="positive_validation.target_example_count",
    )
    if (
        utterance_count != FEASIBILITY_UTTERANCE_COUNT
        or exposure != FEASIBILITY_EXPOSURE_SAMPLES
        or target_clip_count != FEASIBILITY_TARGET_CLIP_COUNT
    ):
        raise ResultPlotError("replay report uses an unexpected experiment-001 cohort")

    baseline = _integer(
        positive.get("baseline_correct_count"),
        name="positive_validation.baseline_correct_count",
    )
    raw_thresholds = _list(document.get("thresholds"), name="thresholds")
    retention_gate: list[bool] = []
    false_event_gate: list[bool] = []
    for threshold, raw_row in enumerate(raw_thresholds):
        row = _mapping(raw_row, name=f"threshold row {threshold}")
        observed_threshold = _integer(
            row.get("threshold_milli"),
            name=f"threshold row {threshold}.threshold_milli",
        )
        if observed_threshold != threshold:
            raise ResultPlotError("replay threshold axis is not the registered grid")
        correct = _integer(
            row.get("validation_correct_accept_count"),
            name=f"threshold row {threshold}.validation_correct_accept_count",
        )
        events = _integer(
            row.get("dev_event_count"),
            name=f"threshold row {threshold}.dev_event_count",
        )
        retention_gate.append(correct * 5 >= baseline * 4)
        false_event_gate.append(events * SAMPLES_PER_HOUR <= exposure)

    retained = [index for index, passed in enumerate(retention_gate) if passed]
    negative = [index for index, passed in enumerate(false_event_gate) if passed]
    if not retained or not negative:
        raise ResultPlotError("replay report does not contain both gate frontiers")
    retention_frontier = retained[-1]
    negative_frontier = negative[0]
    if (
        recomputed_status != "reject"
        or recomputed_threshold is not None
        or retention_frontier != FEASIBILITY_RETENTION_FRONTIER_MILLI
        or negative_frontier != FEASIBILITY_NEGATIVE_FRONTIER_MILLI
        or any(
            retention and false_event
            for retention, false_event in zip(
                retention_gate, false_event_gate, strict=True
            )
        )
    ):
        raise ResultPlotError(
            "replay report differs from the registered experiment-001 "
            "feasibility result"
        )

    return FeasibilityData(
        thresholds=tuple(index / 1_000 for index in range(len(raw_thresholds))),
        retention_gate=tuple(retention_gate),
        false_event_gate=tuple(false_event_gate),
        retention_frontier_milli=retention_frontier,
        negative_frontier_milli=negative_frontier,
        utterance_count=utterance_count,
        scored_exposure_samples=exposure,
        target_clip_count=target_clip_count,
    )


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


def _save(
    figure: Figure,
    output_stem: Path,
    *,
    software: str = "FalseWake experiment 000",
    background: str = BACKGROUND,
) -> None:
    svg_path = output_stem.with_suffix(".svg")
    figure.savefig(
        output_stem.with_suffix(".png"),
        dpi=180,
        facecolor=background,
        metadata={"Software": software},
    )
    figure.savefig(
        svg_path,
        facecolor=background,
        metadata={"Creator": software, "Date": None},
    )
    plt.close(figure)
    try:
        lines = svg_path.read_text(encoding="utf-8").splitlines()
        svg_path.write_text(
            "\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8"
        )
    except OSError as error:
        raise ResultPlotError(f"cannot normalize SVG output: {error}") from error


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
    axis.legend(frameon=False, loc="upper right", fontsize=9)
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
            fontweight="bold",
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


def plot_replay_feasibility(data: FeasibilityData, output_stem: Path) -> None:
    """Render the registered experiment-001 gate feasibility result."""

    if (
        len(data.thresholds) != 1_001
        or len(data.retention_gate) != 1_001
        or len(data.false_event_gate) != 1_001
        or data.thresholds[0] != 0
        or data.thresholds[-1] != 1
    ):
        raise ResultPlotError("feasibility figure requires the full threshold grid")
    figure, axis = plt.subplots(figsize=(10.4, 6.0), layout="constrained")
    figure.set_facecolor(FEASIBILITY_BACKGROUND)
    axis.set_facecolor(FEASIBILITY_BACKGROUND)
    retention = np.asarray(data.retention_gate, dtype=np.int8)
    false_event = np.asarray(data.false_event_gate, dtype=np.int8)
    axis.step(
        data.thresholds,
        retention,
        where="post",
        color=BLUE,
        linestyle="-",
        linewidth=2.7,
        label="Retention gate \N{GREATER-THAN OR EQUAL TO}80%",
    )
    axis.step(
        data.thresholds,
        false_event,
        where="post",
        color=ORANGE,
        linestyle="--",
        linewidth=2.7,
        label="False-event gate \N{LESS-THAN OR EQUAL TO}1/hour",
    )
    retention_frontier = data.retention_frontier_milli / 1_000
    negative_frontier = data.negative_frontier_milli / 1_000
    axis.axvline(
        retention_frontier, color=BLUE, linestyle=":", linewidth=1.4, alpha=0.9
    )
    axis.axvline(
        negative_frontier, color=ORANGE, linestyle=":", linewidth=1.4, alpha=0.9
    )
    axis.text(
        retention_frontier - 0.012,
        0.50,
        "Retention frontier 0.395",
        rotation=90,
        ha="right",
        va="center",
        color=BLUE_DARK,
        fontsize=8.5,
        fontweight="bold",
    )
    axis.text(
        negative_frontier - 0.012,
        0.50,
        "False-event frontier 0.991",
        rotation=90,
        ha="right",
        va="center",
        color=ORANGE,
        fontsize=8.5,
        fontweight="bold",
    )
    axis.text(
        0.035,
        0.92,
        "Retention gate passes",
        color=BLUE_DARK,
        fontsize=9,
        fontweight="bold",
        bbox={"facecolor": FEASIBILITY_BACKGROUND, "edgecolor": "none", "pad": 2},
    )
    axis.text(
        0.64,
        0.08,
        "False-event gate fails",
        color=ORANGE,
        fontsize=9,
        fontweight="bold",
        bbox={"facecolor": FEASIBILITY_BACKGROUND, "edgecolor": "none", "pad": 2},
    )
    axis.set_xlim(0, 1)
    axis.set_ylim(-0.08, 1.08)
    axis.set_xticks(np.linspace(0, 1, 11))
    axis.set_xticklabels([f"{value:.1f}" for value in np.linspace(0, 1, 11)])
    axis.set_yticks([0, 1], ["Fail", "Pass"])
    axis.set_xlabel("Registered target-probability threshold", color=INK, fontsize=10)
    axis.set_title(
        f"{FEASIBILITY_QUESTION}\n{FEASIBILITY_SUBTITLE}",
        loc="left",
        color=INK,
        fontsize=14,
        pad=17,
    )
    axis.legend(frameon=False, loc="upper center", fontsize=9, ncols=2)
    axis.text(
        0,
        -0.19,
        FEASIBILITY_TAKEAWAY,
        transform=axis.transAxes,
        color=INK,
        fontsize=9,
        fontweight="bold",
    )
    axis.text(
        0,
        -0.25,
        "No registered threshold overlaps both pass regions.",
        transform=axis.transAxes,
        color=MUTED,
        fontsize=8.5,
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(MUTED)
    axis.tick_params(colors=INK, labelsize=9)
    axis.grid(True, color=GRID, linewidth=0.8, alpha=0.8)
    axis.set_axisbelow(True)
    _save(
        figure,
        output_stem,
        software="FalseWake experiment 001",
        background=FEASIBILITY_BACKGROUND,
    )


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


def render_replay_feasibility(report_path: Path, output_directory: Path) -> None:
    """Render the registered experiment-001 figure into an existing directory."""

    if not output_directory.is_dir():
        raise ResultPlotError(f"output directory does not exist: {output_directory}")
    data = load_replay_feasibility(report_path)
    with matplotlib.rc_context(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
            "svg.hashsalt": "falsewake-experiment-001",
        }
    ):
        plot_replay_feasibility(
            data, output_directory / "experiment-001-gate-feasibility"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render the reviewed FalseWake result figures."
    )
    parser.add_argument(
        "--replay-report",
        type=Path,
        help="also render the registered experiment 001 gate-feasibility figure",
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
    """Render reviewed result figures from exact checked-in evidence."""

    arguments = _parser().parse_args(argv)
    try:
        render_result_plots(arguments.metrics, arguments.output_directory)
        if arguments.replay_report is not None:
            render_replay_feasibility(
                arguments.replay_report, arguments.output_directory
            )
    except ResultPlotError as error:
        print(f"result plotting failed: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
