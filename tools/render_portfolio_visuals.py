"""Render reproducible, evidence-bound SVGs for the FalseWake README.

The renderer deliberately has a narrow trust boundary:

* source reads are restricted to ``SOURCE_ALLOWLIST``;
* project code is inspected as text/AST, never imported;
* the only project import is ``falsewake.result_plots`` for byte-current
  verification of the already reviewed Experiment 000/001 report figures;
* no registered experiment runner, coordinator, supervisor, seed worker, or
  authority issuer is imported or invoked.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import sys
import tempfile
import tomllib
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Final, cast

RENDERER_VERSION: Final = "1.0.0"
REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY: Final = REPOSITORY_ROOT / "docs" / "images" / "readme"
SAMPLES_PER_HOUR: Final = 57_600_000

SOURCE_ALLOWLIST: Final = frozenset(
    {
        ".github/workflows/ci.yml",
        ".python-version",
        "pyproject.toml",
        "requirements-dev.lock",
        "requirements-train-ci.lock",
        "reports/experiment-000-linear.json",
        "reports/experiment-001-dev-replay.json",
        "reports/experiment-002-execution-incident.json",
        "reports/experiment-003-execution-incident.json",
        "reports/experiment-004-preflight-incident.json",
        "reports/experiment-005-preflight-incident.json",
        "reports/experiment-006-execution-incident.json",
        "src/falsewake/causal_kws.py",
        "src/falsewake/continuous_replay.py",
        "src/falsewake/features.py",
        "src/falsewake/result_plots.py",
        "tests/test_causal_kws.py",
        "tools/render_portfolio_visuals.py",
    }
)

REPORT_ASSETS: Final = (
    "experiment-000-class-recall.png",
    "experiment-000-class-recall.svg",
    "experiment-000-open-set.png",
    "experiment-000-open-set.svg",
    "experiment-001-gate-feasibility.png",
    "experiment-001-gate-feasibility.svg",
)
VISUAL_FILENAMES: Final = (
    "system-path.svg",
    "data-boundaries.svg",
    "causal-tcn-state.svg",
    "experiment-lineage.svg",
    "threshold-tradeoff.svg",
    "unknown-word-recall.svg",
    "setup-verification.svg",
)

INK: Final = "#172033"
MUTED: Final = "#596579"
SUBTLE: Final = "#8791A3"
GRID: Final = "#D9E0EA"
PANEL: Final = "#F7F9FC"
WHITE: Final = "#FFFFFF"
BLUE: Final = "#356FAE"
BLUE_DARK: Final = "#234C78"
BLUE_LIGHT: Final = "#DCEAF7"
ORANGE: Final = "#D27A16"
ORANGE_DARK: Final = "#8E4E08"
ORANGE_LIGHT: Final = "#FBE9D2"
NEUTRAL_DARK: Final = "#4C566A"
FONT: Final = "Arial, Helvetica, sans-serif"
MONO: Final = "'DejaVu Sans Mono', 'Liberation Mono', monospace"


class PortfolioVisualError(ValueError):
    """Evidence or committed output does not satisfy the visual contract."""


@dataclass(frozen=True, slots=True)
class SourceRecord:
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class FrontendFacts:
    sample_rate: int
    clip_samples: int
    window_samples: int
    hop_samples: int
    fft_samples: int
    mel_bins: int
    frame_count: int
    summary_values: int
    replay_hop_samples: int
    refractory_samples: int
    threshold_count: int


@dataclass(frozen=True, slots=True)
class DatasetFacts:
    training_examples: int
    validation_examples: int
    test_examples: int
    validation_role: str
    headline_split: str
    dev_utterances: int
    dev_speakers: int
    dev_exposure_samples: int
    positive_validation_examples: int
    baseline_correct: int

    @property
    def dev_hours(self) -> float:
        return self.dev_exposure_samples / SAMPLES_PER_HOUR


@dataclass(frozen=True, slots=True)
class CausalFacts:
    mel_bins: int
    channels: int
    output_classes: int
    kernel_size: int
    dilations: tuple[int, ...]
    pool_frames: int
    tcn_receptive_frames: int
    model_receptive_frames: int
    state_values: int
    parameter_count: int


@dataclass(frozen=True, slots=True)
class ThresholdPoint:
    threshold_milli: int
    event_count: int
    correct_count: int
    false_events_per_hour: float
    retention: float


@dataclass(frozen=True, slots=True)
class ThresholdFacts:
    points: tuple[ThresholdPoint, ...]
    exposure_samples: int
    baseline_correct: int
    retention_frontier_milli: int
    negative_frontier_milli: int
    selection_status: str


@dataclass(frozen=True, slots=True)
class UnknownWord:
    source_word: str
    predicted_unknown_count: int
    support: int

    @property
    def recall(self) -> float:
        return self.predicted_unknown_count / self.support


@dataclass(frozen=True, slots=True)
class LineageRow:
    experiment: str
    boundary: str
    outcome: str
    detail: str


@dataclass(frozen=True, slots=True)
class Evidence:
    frontend: FrontendFacts
    dataset: DatasetFacts
    causal: CausalFacts
    threshold: ThresholdFacts
    unknown_words: tuple[UnknownWord, ...]
    lineage: tuple[LineageRow, ...]
    python_version: str
    ci_checks: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PortfolioBundle:
    visuals: Mapping[str, bytes]
    provenance: bytes


class Svg:
    """Small deterministic SVG writer with shared accessibility and styling."""

    def __init__(self, width: int, height: int, title: str, description: str) -> None:
        self.width = width
        self.height = height
        self._parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
                f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
                'aria-labelledby="title desc">'
            ),
            f'<title id="title">{escape(title)}</title>',
            f'<desc id="desc">{escape(description)}</desc>',
            (
                "<style>"
                f"text{{font-family:{FONT};fill:{INK}}}"
                f".mono{{font-family:{MONO}}}"
                ".title{font-size:34px;font-weight:700}"
                ".subtitle{font-size:17px}"
                ".section{font-size:20px;font-weight:700}"
                ".body{font-size:16px}"
                ".small{font-size:14px}"
                ".tiny{font-size:12px}"
                ".metric{font-size:29px;font-weight:700}"
                ".label{font-size:14px;font-weight:700;letter-spacing:.5px}"
                "</style>"
            ),
            f'<rect width="{width}" height="{height}" fill="{WHITE}"/>',
        ]

    def raw(self, value: str) -> None:
        self._parts.append(value)

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        fill: str = WHITE,
        stroke: str = GRID,
        radius: float = 14,
        stroke_width: float = 1.5,
    ) -> None:
        self.raw(
            f'<rect x="{x:g}" y="{y:g}" width="{width:g}" height="{height:g}" '
            f'rx="{radius:g}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{stroke_width:g}"/>'
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        stroke: str = GRID,
        width: float = 1.5,
        dash: str | None = None,
    ) -> None:
        dash_attribute = "" if dash is None else f' stroke-dasharray="{dash}"'
        self.raw(
            f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" '
            f'stroke="{stroke}" stroke-width="{width:g}"{dash_attribute}/>'
        )

    def circle(
        self,
        cx: float,
        cy: float,
        radius: float,
        *,
        fill: str,
        stroke: str = WHITE,
        stroke_width: float = 2,
    ) -> None:
        self.raw(
            f'<circle cx="{cx:g}" cy="{cy:g}" r="{radius:g}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{stroke_width:g}"/>'
        )

    def text(
        self,
        x: float,
        y: float,
        value: str,
        *,
        css_class: str = "body",
        fill: str = INK,
        anchor: str = "start",
        weight: int | None = None,
    ) -> None:
        weight_attribute = "" if weight is None else f' font-weight="{weight}"'
        self.raw(
            f'<text x="{x:g}" y="{y:g}" class="{css_class}" fill="{fill}" '
            f'text-anchor="{anchor}"{weight_attribute}>{escape(value)}</text>'
        )

    def multiline(
        self,
        x: float,
        y: float,
        lines: Sequence[str],
        *,
        css_class: str = "body",
        fill: str = INK,
        line_height: float = 24,
        anchor: str = "start",
        weight: int | None = None,
    ) -> None:
        weight_attribute = "" if weight is None else f' font-weight="{weight}"'
        spans = "".join(
            (
                f'<tspan x="{x:g}" dy="{0 if index == 0 else line_height:g}">'
                f"{escape(line)}</tspan>"
            )
            for index, line in enumerate(lines)
        )
        self.raw(
            f'<text x="{x:g}" y="{y:g}" class="{css_class}" fill="{fill}" '
            f'text-anchor="{anchor}"{weight_attribute}>{spans}</text>'
        )

    def path(
        self,
        points: Sequence[tuple[float, float]],
        *,
        stroke: str,
        width: float = 2.5,
        dash: str | None = None,
        fill: str = "none",
    ) -> None:
        if not points:
            raise PortfolioVisualError("an SVG path cannot be empty")
        commands = " ".join(
            f"{'M' if index == 0 else 'L'} {x:.2f} {y:.2f}"
            for index, (x, y) in enumerate(points)
        )
        dash_attribute = "" if dash is None else f' stroke-dasharray="{dash}"'
        self.raw(
            f'<path d="{commands}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{width:g}" stroke-linejoin="round" '
            f'stroke-linecap="round"{dash_attribute}/>'
        )

    def arrow(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        stroke: str = BLUE_DARK,
        width: float = 2.5,
    ) -> None:
        self.line(x1, y1, x2, y2, stroke=stroke, width=width)
        angle = math.atan2(y2 - y1, x2 - x1)
        size = 10
        left = (
            x2 - size * math.cos(angle - math.pi / 6),
            y2 - size * math.sin(angle - math.pi / 6),
        )
        right = (
            x2 - size * math.cos(angle + math.pi / 6),
            y2 - size * math.sin(angle + math.pi / 6),
        )
        self.raw(
            f'<path d="M {left[0]:.2f} {left[1]:.2f} L {x2:.2f} {y2:.2f} '
            f'L {right[0]:.2f} {right[1]:.2f}" fill="none" stroke="{stroke}" '
            f'stroke-width="{width:g}" stroke-linecap="round"/>'
        )

    def header(self, title: str, subtitle: str, eyebrow: str) -> None:
        self.text(70, 55, eyebrow.upper(), css_class="label", fill=BLUE_DARK)
        self.text(70, 103, title, css_class="title")
        self.text(70, 139, subtitle, css_class="subtitle", fill=MUTED)
        self.line(70, 165, self.width - 70, 165, stroke=GRID, width=1.5)

    def footer(self, source: str, nonclaim: str) -> None:
        y = self.height - 43
        self.line(70, y - 25, self.width - 70, y - 25, stroke=GRID)
        self.text(70, y, source, css_class="tiny", fill=MUTED)
        self.text(
            self.width - 70,
            y,
            nonclaim,
            css_class="tiny",
            fill=ORANGE_DARK,
            anchor="end",
            weight=700,
        )

    def finish(self) -> bytes:
        return ("\n".join((*self._parts, "</svg>")) + "\n").encode("utf-8")


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if type(value) is not dict:
        raise PortfolioVisualError(f"{name} is not an object")
    mapping = cast(dict[object, object], value)
    if not all(type(key) is str for key in mapping):
        raise PortfolioVisualError(f"{name} has a non-text key")
    return cast(dict[str, object], mapping)


def _list(value: object, *, name: str) -> list[object]:
    if type(value) is not list:
        raise PortfolioVisualError(f"{name} is not a list")
    return cast(list[object], value)


def _integer(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise PortfolioVisualError(f"{name} is not a non-negative integer")
    return value


def _boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise PortfolioVisualError(f"{name} is not a boolean")
    return value


def _text(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise PortfolioVisualError(f"{name} is not non-empty text")
    return value


def _source_path(root: Path, relative_path: str) -> Path:
    if relative_path not in SOURCE_ALLOWLIST:
        raise PortfolioVisualError(f"source is outside the allowlist: {relative_path}")
    if Path(relative_path).is_absolute() or ".." in Path(relative_path).parts:
        raise PortfolioVisualError(
            f"source path is not repository-relative: {relative_path}"
        )
    path = root / relative_path
    if path.is_symlink() or not path.is_file():
        raise PortfolioVisualError(
            f"source is missing or is a symlink: {relative_path}"
        )
    if path.resolve() != (root.resolve() / relative_path):
        raise PortfolioVisualError(f"source escapes repository root: {relative_path}")
    return path


def _read_source(root: Path, relative_path: str) -> bytes:
    try:
        return _source_path(root, relative_path).read_bytes()
    except OSError as error:
        raise PortfolioVisualError(f"cannot read {relative_path}: {error}") from error


def _load_json(root: Path, relative_path: str) -> dict[str, object]:
    def reject_non_finite(value: str) -> object:
        raise PortfolioVisualError(
            f"{relative_path} contains a non-finite JSON constant: {value}"
        )

    try:
        loaded = cast(
            object,
            json.loads(
                _read_source(root, relative_path),
                parse_constant=reject_non_finite,
            ),
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise PortfolioVisualError(f"cannot decode {relative_path}: {error}") from error
    return _mapping(loaded, name=relative_path)


def _sha256(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _source_records(root: Path, paths: Sequence[str]) -> list[dict[str, str]]:
    return [
        {"path": path, "sha256": _sha256(_read_source(root, path))}
        for path in sorted(paths)
    ]


def _eval_static(node: ast.expr, environment: Mapping[str, object]) -> object:
    if isinstance(node, ast.Constant) and type(node.value) in {int, float, str}:
        return cast(int | float | str, node.value)
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(_eval_static(element, environment) for element in node.elts)
    if isinstance(node, ast.Name) and node.id in environment:
        return environment[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_static(node.operand, environment)
        if type(value) not in {int, float}:
            raise PortfolioVisualError("static unary expression is not numeric")
        numeric = cast(int | float, value)
        return numeric if isinstance(node.op, ast.UAdd) else -numeric
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult)
    ):
        left = _eval_static(node.left, environment)
        right = _eval_static(node.right, environment)
        if type(left) is not int or type(right) is not int:
            raise PortfolioVisualError("static binary expression is not integer")
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        return left * right
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "sum"
        and len(node.args) == 1
        and not node.keywords
    ):
        values = _eval_static(node.args[0], environment)
        if type(values) is not tuple or not all(type(value) is int for value in values):
            raise PortfolioVisualError("static sum input is not an integer tuple")
        return sum(cast(tuple[int, ...], values))
    raise PortfolioVisualError(f"unsupported static expression: {ast.dump(node)}")


def _module_constants(
    root: Path, relative_path: str, names: Sequence[str]
) -> dict[str, object]:
    try:
        tree = ast.parse(_read_source(root, relative_path), filename=relative_path)
    except (SyntaxError, UnicodeDecodeError) as error:
        raise PortfolioVisualError(f"cannot parse {relative_path}: {error}") from error
    wanted = set(names)
    values: dict[str, object] = {}
    for statement in tree.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if isinstance(target, ast.Name) and target.id in wanted:
            values[target.id] = _eval_static(statement.value, values)
    missing = wanted - values.keys()
    if missing:
        raise PortfolioVisualError(
            f"{relative_path} omits static constants: {sorted(missing)}"
        )
    return values


def _frontend_defaults(root: Path) -> dict[str, int]:
    relative_path = "src/falsewake/features.py"
    try:
        tree = ast.parse(_read_source(root, relative_path), filename=relative_path)
    except (SyntaxError, UnicodeDecodeError) as error:
        raise PortfolioVisualError(f"cannot parse {relative_path}: {error}") from error
    names = {
        "sample_rate",
        "clip_samples",
        "window_samples",
        "hop_samples",
        "fft_samples",
        "mel_bins",
    }
    result: dict[str, int] = {}
    for statement in tree.body:
        if (
            not isinstance(statement, ast.ClassDef)
            or statement.name != "FrontendConfig"
        ):
            continue
        for item in statement.body:
            if (
                isinstance(item, ast.AnnAssign)
                and isinstance(item.target, ast.Name)
                and item.target.id in names
                and item.value is not None
            ):
                value = _eval_static(item.value, {})
                if type(value) is not int:
                    raise PortfolioVisualError(
                        f"FrontendConfig.{item.target.id} is not an integer literal"
                    )
                result[item.target.id] = value
    if result.keys() != names:
        raise PortfolioVisualError("FrontendConfig defaults are incomplete")
    return result


def _extract_frontend(root: Path) -> FrontendFacts:
    defaults = _frontend_defaults(root)
    replay = _module_constants(
        root,
        "src/falsewake/continuous_replay.py",
        (
            "CLIP_SAMPLES",
            "WINDOW_HOP_SAMPLES",
            "REFRACTORY_SAMPLES",
            "THRESHOLD_COUNT",
        ),
    )
    clip_samples = defaults["clip_samples"]
    if replay["CLIP_SAMPLES"] != clip_samples:
        raise PortfolioVisualError("clip and replay sample counts diverge")
    frame_count = (
        1 + (clip_samples - defaults["window_samples"]) // defaults["hop_samples"]
    )
    return FrontendFacts(
        sample_rate=defaults["sample_rate"],
        clip_samples=clip_samples,
        window_samples=defaults["window_samples"],
        hop_samples=defaults["hop_samples"],
        fft_samples=defaults["fft_samples"],
        mel_bins=defaults["mel_bins"],
        frame_count=frame_count,
        summary_values=defaults["mel_bins"] * 2,
        replay_hop_samples=cast(int, replay["WINDOW_HOP_SAMPLES"]),
        refractory_samples=cast(int, replay["REFRACTORY_SAMPLES"]),
        threshold_count=cast(int, replay["THRESHOLD_COUNT"]),
    )


def _per_class_support(split: dict[str, object], *, name: str) -> int:
    rows = _list(split.get("per_class"), name=f"{name}.per_class")
    return sum(
        _integer(_mapping(row, name=f"{name}.row").get("support"), name="support")
        for row in rows
    )


def _extract_dataset(root: Path) -> DatasetFacts:
    metrics = _load_json(root, "reports/experiment-000-linear.json")
    fit = _mapping(metrics.get("fit"), name="experiment-000.fit")
    splits = _mapping(metrics.get("splits"), name="experiment-000.splits")
    validation = _mapping(splits.get("validation"), name="validation")
    test = _mapping(splits.get("test"), name="test")
    replay = _load_json(root, "reports/experiment-001-dev-replay.json")
    dev = _mapping(replay.get("dev"), name="experiment-001.dev")
    positive = _mapping(
        replay.get("positive_validation"), name="experiment-001.positive_validation"
    )
    return DatasetFacts(
        training_examples=_integer(
            fit.get("training_examples"), name="fit.training_examples"
        ),
        validation_examples=_per_class_support(validation, name="validation"),
        test_examples=_per_class_support(test, name="test"),
        validation_role=_text(metrics.get("validation_role"), name="validation_role"),
        headline_split=_text(metrics.get("headline_split"), name="headline_split"),
        dev_utterances=_integer(dev.get("utterance_count"), name="dev.utterance_count"),
        dev_speakers=_integer(dev.get("speaker_count"), name="dev.speaker_count"),
        dev_exposure_samples=_integer(
            dev.get("scored_exposure_samples"), name="dev.scored_exposure_samples"
        ),
        positive_validation_examples=_integer(
            positive.get("target_example_count"),
            name="positive_validation.target_example_count",
        ),
        baseline_correct=_integer(
            positive.get("baseline_correct_count"),
            name="positive_validation.baseline_correct_count",
        ),
    )


def _extract_causal(root: Path) -> CausalFacts:
    values = _module_constants(
        root,
        "src/falsewake/causal_kws.py",
        (
            "INPUT_MEL_BINS",
            "ENCODER_CHANNELS",
            "OUTPUT_CLASSES",
            "DEPTHWISE_KERNEL_SIZE",
            "BLOCK_DILATIONS",
            "POOL_FRAMES",
            "POOL_HISTORY_FRAMES",
            "TCN_RECEPTIVE_FIELD_FRAMES",
            "MODEL_RECEPTIVE_FIELD_FRAMES",
            "STREAMING_FLOAT_STATE_VALUES",
            "TRAINABLE_PARAMETER_COUNT",
        ),
    )
    dilations_raw = values["BLOCK_DILATIONS"]
    if type(dilations_raw) is not tuple or not all(
        type(value) is int for value in dilations_raw
    ):
        raise PortfolioVisualError("BLOCK_DILATIONS is not an integer tuple")
    facts = CausalFacts(
        mel_bins=cast(int, values["INPUT_MEL_BINS"]),
        channels=cast(int, values["ENCODER_CHANNELS"]),
        output_classes=cast(int, values["OUTPUT_CLASSES"]),
        kernel_size=cast(int, values["DEPTHWISE_KERNEL_SIZE"]),
        dilations=cast(tuple[int, ...], dilations_raw),
        pool_frames=cast(int, values["POOL_FRAMES"]),
        tcn_receptive_frames=cast(int, values["TCN_RECEPTIVE_FIELD_FRAMES"]),
        model_receptive_frames=cast(int, values["MODEL_RECEPTIVE_FIELD_FRAMES"]),
        state_values=cast(int, values["STREAMING_FLOAT_STATE_VALUES"]),
        parameter_count=cast(int, values["TRAINABLE_PARAMETER_COUNT"]),
    )
    expected_tcn = 1 + (facts.kernel_size - 1) * sum(facts.dilations)
    expected_model = expected_tcn + facts.pool_frames - 1
    expected_state = facts.channels * (
        (facts.kernel_size - 1) * sum(facts.dilations) + facts.pool_frames - 1
    )
    if (
        facts.tcn_receptive_frames != expected_tcn
        or facts.model_receptive_frames != expected_model
        or facts.state_values != expected_state
    ):
        raise PortfolioVisualError("causal state arithmetic is internally inconsistent")
    test_source = _read_source(root, "tests/test_causal_kws.py").decode("utf-8")
    required_assertions = (
        "test_registered_architecture_and_parameter_count",
        "test_full_one_frame_and_irregular_streaming_are_equivalent",
        "test_logits_are_prefix_causal",
    )
    if not all(assertion in test_source for assertion in required_assertions):
        raise PortfolioVisualError("causal architecture proof tests are incomplete")
    return facts


def _extract_threshold(root: Path) -> ThresholdFacts:
    replay = _load_json(root, "reports/experiment-001-dev-replay.json")
    dev = _mapping(replay.get("dev"), name="dev")
    positive = _mapping(replay.get("positive_validation"), name="positive_validation")
    selection = _mapping(replay.get("selection"), name="selection")
    exposure = _integer(
        dev.get("scored_exposure_samples"), name="dev.scored_exposure_samples"
    )
    baseline = _integer(
        positive.get("baseline_correct_count"),
        name="positive_validation.baseline_correct_count",
    )
    if exposure == 0 or baseline == 0:
        raise PortfolioVisualError("threshold denominators must be positive")
    rows = _list(replay.get("thresholds"), name="thresholds")
    points: list[ThresholdPoint] = []
    for expected_threshold, raw_row in enumerate(rows):
        row = _mapping(raw_row, name=f"thresholds[{expected_threshold}]")
        threshold = _integer(
            row.get("threshold_milli"), name=f"thresholds[{expected_threshold}].milli"
        )
        if threshold != expected_threshold:
            raise PortfolioVisualError("threshold grid is not contiguous")
        events = _integer(row.get("dev_event_count"), name="dev_event_count")
        correct = _integer(
            row.get("validation_correct_accept_count"),
            name="validation_correct_accept_count",
        )
        if correct > baseline:
            raise PortfolioVisualError("threshold correct count exceeds baseline")
        points.append(
            ThresholdPoint(
                threshold_milli=threshold,
                event_count=events,
                correct_count=correct,
                false_events_per_hour=events * SAMPLES_PER_HOUR / exposure,
                retention=correct / baseline,
            )
        )
    if len(points) != 1_001:
        raise PortfolioVisualError("registered threshold grid must have 1,001 points")
    if any(
        current.correct_count > previous.correct_count
        for previous, current in zip(points, points[1:], strict=False)
    ):
        raise PortfolioVisualError(
            "validation acceptance count increases as the threshold rises"
        )
    retained = [
        point.threshold_milli
        for point in points
        if point.correct_count * 5 >= baseline * 4
    ]
    negative = [
        point.threshold_milli
        for point in points
        if point.event_count * SAMPLES_PER_HOUR <= exposure
    ]
    if not retained or not negative:
        raise PortfolioVisualError("threshold grid omits a gate frontier")
    retention_frontier = retained[-1]
    negative_frontier = negative[0]
    if (
        _integer(
            selection.get("retention_frontier_milli"),
            name="selection.retention_frontier_milli",
        )
        != retention_frontier
        or _integer(
            selection.get("negative_frontier_milli"),
            name="selection.negative_frontier_milli",
        )
        != negative_frontier
        or any(
            point.correct_count * 5 >= baseline * 4
            and point.event_count * SAMPLES_PER_HOUR <= exposure
            for point in points
        )
    ):
        raise PortfolioVisualError("reported gate outcome differs from raw counts")
    if (
        selection.get("status") != "reject"
        or selection.get("selected_threshold_milli") is not None
    ):
        raise PortfolioVisualError("registered rejection record changed")
    return ThresholdFacts(
        points=tuple(points),
        exposure_samples=exposure,
        baseline_correct=baseline,
        retention_frontier_milli=retention_frontier,
        negative_frontier_milli=negative_frontier,
        selection_status=_text(selection.get("status"), name="selection.status"),
    )


def _extract_unknown_words(root: Path) -> tuple[UnknownWord, ...]:
    metrics = _load_json(root, "reports/experiment-000-linear.json")
    splits = _mapping(metrics.get("splits"), name="splits")
    test = _mapping(splits.get("test"), name="splits.test")
    raw_rows = _list(
        test.get("unknown_by_source_word"), name="test.unknown_by_source_word"
    )
    rows: list[UnknownWord] = []
    for raw_row in raw_rows:
        row = _mapping(raw_row, name="unknown_by_source_word row")
        support = _integer(row.get("support"), name="unknown support")
        correct = _integer(
            row.get("predicted_unknown_count"), name="predicted_unknown_count"
        )
        if support == 0 or correct > support:
            raise PortfolioVisualError("unknown-word row has an invalid denominator")
        rows.append(
            UnknownWord(
                source_word=_text(row.get("source_word"), name="source_word"),
                predicted_unknown_count=correct,
                support=support,
            )
        )
    if len(rows) != 25 or sum(row.support for row in rows) != 405:
        raise PortfolioVisualError("unknown-word test slice has an unexpected cohort")
    return tuple(sorted(rows, key=lambda row: (-row.recall, row.source_word)))


def _extract_lineage(root: Path) -> tuple[LineageRow, ...]:
    replay = _load_json(root, "reports/experiment-001-dev-replay.json")
    selection = _mapping(replay.get("selection"), name="experiment-001.selection")
    incident_002 = _load_json(root, "reports/experiment-002-execution-incident.json")
    incident_003 = _load_json(root, "reports/experiment-003-execution-incident.json")
    incident_004 = _load_json(root, "reports/experiment-004-preflight-incident.json")
    incident_005 = _load_json(root, "reports/experiment-005-preflight-incident.json")
    incident_006 = _load_json(root, "reports/experiment-006-execution-incident.json")

    outcome_002 = _mapping(incident_002.get("outcome"), name="002.outcome")
    diagnosis_002 = _mapping(incident_002.get("diagnosis"), name="002.diagnosis")
    remediation_002 = _mapping(
        incident_002.get("remediation"), name="002.remediation"
    )
    attempt_003 = _mapping(incident_003.get("attempt"), name="003.attempt")
    outcome_003 = _mapping(incident_003.get("outcome"), name="003.outcome")
    attempt_004 = _mapping(incident_004.get("attempt"), name="004.attempt")
    attempt_005 = _mapping(incident_005.get("attempt"), name="005.attempt")
    attempt_006 = _mapping(incident_006.get("attempt"), name="006.attempt")
    diagnosis_006 = _mapping(incident_006.get("diagnosis"), name="006.diagnosis")
    outcome_006 = _mapping(incident_006.get("outcome"), name="006.outcome")
    terminal_006 = _mapping(
        incident_006.get("terminal_disposition"), name="006.terminal_disposition"
    )

    if (
        _text(incident_002.get("incident"), name="002.incident")
        != "registered_execution_failure"
        or _text(outcome_002.get("status"), name="002.outcome.status")
        != "execution_failure"
        or _boolean(
            outcome_002.get("checkpoint_reusable"),
            name="002.outcome.checkpoint_reusable",
        )
        or _boolean(
            diagnosis_002.get("published_scientific_result_admitted"),
            name="002.diagnosis.published_scientific_result_admitted",
        )
        or _boolean(
            remediation_002.get("reuse_experiment_002"),
            name="002.remediation.reuse_experiment_002",
        )
        or _boolean(
            remediation_002.get("scientific_protocol_change"),
            name="002.remediation.scientific_protocol_change",
        )
        or _integer(
            attempt_003.get("registered_runner_invocations"), name="003 invoked"
        )
        != 1
        or _integer(
            attempt_003.get("training_processes_started"),
            name="003 training processes",
        )
        != 0
        or _integer(
            attempt_003.get("registered_optimizer_updates"),
            name="003 optimizer updates",
        )
        != 0
        or _integer(
            attempt_003.get("registered_validation_examples"),
            name="003 validation examples",
        )
        != 0
        or _boolean(attempt_003.get("retry_allowed"), name="003 retry allowed")
        or _integer(attempt_004.get("registered_invocation_count"), name="004 invoked")
        != 0
        or _boolean(
            attempt_004.get("registered_attempt_consumed"),
            name="004 attempt consumed",
        )
        or _boolean(
            attempt_004.get("registered_authority_issuer_invoked"),
            name="004 authority invoked",
        )
        or _boolean(
            attempt_004.get("registered_coordinator_invoked"),
            name="004 coordinator invoked",
        )
        or _boolean(
            attempt_004.get("registered_runner_invoked"), name="004 runner invoked"
        )
        or _integer(attempt_005.get("registered_invocation_count"), name="005 invoked")
        != 0
        or _boolean(
            attempt_005.get("registered_attempt_consumed"),
            name="005 attempt consumed",
        )
        or _boolean(
            attempt_005.get("registered_authority_issuer_invoked"),
            name="005 authority invoked",
        )
        or _boolean(
            attempt_005.get("registered_coordinator_invoked"),
            name="005 coordinator invoked",
        )
        or _boolean(
            attempt_005.get("registered_runner_invoked"), name="005 runner invoked"
        )
        or _integer(attempt_006.get("registered_invocation_count"), name="006 invoked")
        != 1
        or not _boolean(
            attempt_006.get("registered_attempt_consumed"),
            name="006 attempt consumed",
        )
        or _boolean(attempt_006.get("retry_allowed"), name="006 retry allowed")
        or not _boolean(attempt_006.get("terminal"), name="006 terminal")
        or _boolean(
            outcome_006.get("scientific_result_available"),
            name="006 scientific result",
        )
        or _boolean(
            outcome_006.get("checkpoint_reusable"), name="006 checkpoint reusable"
        )
        or _boolean(
            terminal_006.get("experiment_006_retry_allowed"),
            name="006 terminal retry allowed",
        )
    ):
        raise PortfolioVisualError("incident invocation boundaries changed")
    root_cause = _text(
        diagnosis_006.get("root_cause_status"), name="006.root_cause_status"
    )
    if root_cause != "undetermined_from_retained_evidence":
        raise PortfolioVisualError("Experiment 006 root-cause boundary changed")
    return (
        LineageRow(
            "000",
            "MEASURED",
            "Clip baseline complete",
            "Argmax clip metrics; thresholds and streaming were not evaluated.",
        ),
        LineageRow(
            "001",
            "MEASURED · REJECTED",
            f"Replay grid {_text(selection.get('status'), name='001.status')}",
            "No threshold satisfied retention and false-event gates together.",
        ),
        LineageRow(
            "002",
            "INVOKED",
            _text(outcome_002.get("code"), name="002.outcome.code"),
            "Registered execution failed; no scientific result admitted.",
        ),
        LineageRow(
            "003",
            "INVOKED",
            _text(outcome_003.get("code"), name="003.outcome.code"),
            "Admission failed before training processes started.",
        ),
        LineageRow(
            "004",
            "PREFLIGHT REJECTED",
            _text(incident_004.get("status"), name="004.status"),
            "Registered authority/coordinator/runner were not invoked.",
        ),
        LineageRow(
            "005",
            "PREFLIGHT REJECTED",
            _text(incident_005.get("status"), name="005.status"),
            "Registered authority/coordinator/runner were not invoked.",
        ),
        LineageRow(
            "006",
            "INVOKED · TERMINAL",
            _text(outcome_006.get("code"), name="006.outcome.code"),
            "Root cause is undetermined from retained evidence; no retry.",
        ),
    )


def _extract_setup(root: Path) -> tuple[str, tuple[str, ...]]:
    version = _read_source(root, ".python-version").decode("ascii").strip()
    if version != "3.12.3":
        raise PortfolioVisualError("the registered Python version changed")
    pyproject = cast(
        dict[str, object], tomllib.loads(_read_source(root, "pyproject.toml").decode())
    )
    project = _mapping(pyproject.get("project"), name="pyproject.project")
    if project.get("requires-python") != ">=3.12":
        raise PortfolioVisualError("pyproject Python requirement changed")
    ci = _read_source(root, ".github/workflows/ci.yml").decode("utf-8")
    commands = (
        "python -m ruff check --no-cache --output-format=github .",
        "python tools/render_portfolio_visuals.py check",
        "tests/test_result_plots.py",
        "tests/test_portfolio_visuals.py",
        "python -m mypy --no-incremental --show-error-codes",
    )
    if not all(command in ci for command in commands):
        raise PortfolioVisualError("CI no longer contains the documented checks")
    development_lock = _read_source(root, "requirements-dev.lock")
    if (
        b"--hash=sha256:" not in development_lock
        or b"autogenerated by pip-compile" not in development_lock
    ):
        raise PortfolioVisualError("development lock is not hash-locked")
    training_lock = _read_source(root, "requirements-train-ci.lock")
    if (
        b"--hash=sha256:" not in training_lock
        or b"torch @ https://download.pytorch.org/whl/cpu/" not in training_lock
    ):
        raise PortfolioVisualError("CPU training-test lock is not hash-locked")
    return version, commands


def extract_evidence(root: Path = REPOSITORY_ROOT) -> Evidence:
    """Extract all visual facts without importing project implementation modules."""

    version, checks = _extract_setup(root)
    return Evidence(
        frontend=_extract_frontend(root),
        dataset=_extract_dataset(root),
        causal=_extract_causal(root),
        threshold=_extract_threshold(root),
        unknown_words=_extract_unknown_words(root),
        lineage=_extract_lineage(root),
        python_version=version,
        ci_checks=checks,
    )


def _render_system_path(evidence: Evidence) -> bytes:
    facts = evidence.frontend
    canvas = Svg(
        1600,
        900,
        "FalseWake measured system path",
        "The measured baseline path from audited audio through deterministic "
        "features, clip scoring, and continuous replay.",
    )
    canvas.header(
        "Measured system path",
        "One deterministic frontend supports the reviewed clip baseline and replay.",
        "FalseWake · implementation map",
    )
    boxes = (
        (
            80,
            250,
            245,
            210,
            "Audited audio",
            ("Speech Commands v0.02", "LibriSpeech dev-clean", "SHA-bound inputs"),
        ),
        (
            385,
            250,
            245,
            210,
            "Fixed frontend",
            (
                f"{facts.sample_rate // 1000} kHz mono PCM",
                f"{facts.window_samples / facts.sample_rate * 1000:.0f} ms window",
                f"{facts.hop_samples / facts.sample_rate * 1000:.0f} ms hop",
                f"{facts.mel_bins} log-mel bins",
            ),
        ),
        (
            690,
            250,
            245,
            210,
            "Clip baseline",
            (
                f"{facts.frame_count} × {facts.mel_bins} frames",
                f"mean + std → {facts.summary_values} values",
                "linear classifier",
            ),
        ),
        (
            995,
            250,
            245,
            210,
            "Continuous replay",
            (
                f"{facts.clip_samples / facts.sample_rate:.1f} s windows",
                f"{facts.replay_hop_samples / facts.sample_rate * 1000:.0f} ms stride",
                f"{facts.refractory_samples / facts.sample_rate:.1f} s refractory",
            ),
        ),
        (
            1300,
            250,
            220,
            210,
            "Gate decision",
            (
                f"{facts.threshold_count:,} thresholds",
                "retention + FEH",
                "registered reject",
            ),
        ),
    )
    for index, (x, y, width, height, title, lines) in enumerate(boxes):
        fill = BLUE_LIGHT if index in {1, 2} else PANEL
        if index == 4:
            fill = ORANGE_LIGHT
        canvas.rect(
            x,
            y,
            width,
            height,
            fill=fill,
            stroke=BLUE_DARK if index < 4 else ORANGE_DARK,
        )
        canvas.text(x + 24, y + 42, title, css_class="section")
        canvas.multiline(
            x + 24, y + 82, lines, css_class="body", fill=MUTED, line_height=28
        )
        if index < len(boxes) - 1:
            next_x = boxes[index + 1][0]
            canvas.arrow(x + width + 9, y + height / 2, next_x - 10, y + height / 2)
    canvas.rect(220, 545, 1160, 185, fill=WHITE, stroke=GRID)
    canvas.text(255, 590, "Evidence preserved at each boundary", css_class="section")
    canvas.multiline(
        255,
        630,
        (
            "Dataset identities and deterministic sampling",
            "Raw event counts and scored exposure",
        ),
        css_class="body",
        fill=MUTED,
        line_height=32,
    )
    canvas.multiline(
        820,
        630,
        (
            "Raw correct counts and baseline denominator",
            "Static SVG/PNG regenerated and byte-compared",
        ),
        css_class="body",
        fill=MUTED,
        line_height=32,
    )
    canvas.footer(
        "Sources: features.py · continuous_replay.py · experiment-000/001 reports",
        "Measured baseline path only",
    )
    return canvas.finish()


def _render_data_boundaries(evidence: Evidence) -> bytes:
    data = evidence.dataset
    canvas = Svg(
        1600,
        900,
        "FalseWake data boundaries",
        "Separate train, diagnostic validation, held-out test, positive-validation, "
        "and negative-replay roles with exact cohort sizes.",
    )
    canvas.header(
        "Data boundaries and decision roles",
        "Cohorts are separated by purpose; counts below come from committed reports.",
        "FalseWake · evidence boundaries",
    )
    cards = (
        (
            75,
            225,
            320,
            235,
            "TRAIN",
            f"{data.training_examples:,}",
            "Speech Commands clips",
            "Fits scaler + linear classifier",
            BLUE_LIGHT,
        ),
        (
            430,
            225,
            320,
            235,
            "VALIDATION",
            f"{data.validation_examples:,}",
            "Speech Commands clips",
            "Diagnostic only; no selection",
            PANEL,
        ),
        (
            785,
            225,
            320,
            235,
            "TEST",
            f"{data.test_examples:,}",
            "Speech Commands clips",
            "Experiment 000 headline split",
            PANEL,
        ),
        (
            1140,
            225,
            385,
            235,
            "NEGATIVE REPLAY",
            f"{data.dev_utterances:,}",
            f"dev-clean utterances · {data.dev_speakers} speakers",
            f"{data.dev_hours:.3f} scored hours",
            ORANGE_LIGHT,
        ),
    )
    for x, y, width, height, label, metric, context, role, fill in cards:
        canvas.rect(
            x,
            y,
            width,
            height,
            fill=fill,
            stroke=BLUE_DARK if fill != ORANGE_LIGHT else ORANGE_DARK,
        )
        canvas.text(
            x + 26,
            y + 40,
            label,
            css_class="label",
            fill=BLUE_DARK if fill != ORANGE_LIGHT else ORANGE_DARK,
        )
        canvas.text(x + 26, y + 91, metric, css_class="metric")
        canvas.text(x + 26, y + 126, context, css_class="small", fill=MUTED)
        canvas.text(x + 26, y + 186, role, css_class="body", weight=700)
    canvas.rect(250, 535, 1100, 190, fill=WHITE, stroke=GRID)
    canvas.text(290, 580, "Experiment 001 gate construction", css_class="section")
    canvas.rect(290, 615, 430, 70, fill=BLUE_LIGHT, stroke=BLUE_DARK, radius=10)
    canvas.text(
        315,
        646,
        f"{data.positive_validation_examples:,} target validation clips",
        css_class="body",
        weight=700,
    )
    canvas.text(
        315,
        672,
        f"{data.baseline_correct:,} baseline-correct denominator",
        css_class="small",
        fill=MUTED,
    )
    canvas.arrow(740, 650, 860, 650)
    canvas.rect(880, 615, 430, 70, fill=ORANGE_LIGHT, stroke=ORANGE_DARK, radius=10)
    canvas.text(
        905,
        646,
        f"{data.dev_utterances:,} negative replay utterances",
        css_class="body",
        weight=700,
    )
    canvas.text(
        905, 672, "Raw events / exact exposure → FEH", css_class="small", fill=MUTED
    )
    canvas.text(
        800,
        712,
        "both gates required at one threshold",
        css_class="small",
        fill=ORANGE_DARK,
        anchor="middle",
        weight=700,
    )
    canvas.footer(
        "Sources: experiment-000-linear.json · experiment-001-dev-replay.json",
        "No test-set threshold tuning",
    )
    return canvas.finish()


def _render_causal_tcn(evidence: Evidence) -> bytes:
    facts = evidence.causal
    canvas = Svg(
        1600,
        900,
        "Causal TCN architecture and streaming state",
        "The implemented and architecture-tested causal neural network, its residual "
        "dilation stack, receptive field, explicit state, and nonclaims.",
    )
    canvas.header(
        "Causal TCN architecture and streaming state",
        "Implemented and architecture-tested; explicit history makes "
        "chunking stateful.",
        "FalseWake · neural implementation",
    )
    stages = (
        (
            80,
            255,
            220,
            180,
            "Input",
            (f"{facts.mel_bins} log-mel bins", "float32 frames"),
        ),
        (
            355,
            255,
            220,
            180,
            "1×1 stem",
            (f"{facts.channels} channels", "LayerNorm + ReLU"),
        ),
        (
            630,
            225,
            420,
            240,
            "8 residual causal DS blocks",
            (
                f"kernel {facts.kernel_size}",
                f"dilations {facts.dilations[:4]} × 2",
                "depthwise → pointwise",
            ),
        ),
        (
            1105,
            255,
            220,
            180,
            "Causal pool",
            (f"{facts.pool_frames} frames", f"{facts.pool_frames - 1} history frames"),
        ),
        (1380, 255, 140, 180, "Head", (f"{facts.output_classes} logits", "per frame")),
    )
    for index, (x, y, width, height, title, lines) in enumerate(stages):
        fill = BLUE_LIGHT if index in {1, 2} else PANEL
        canvas.rect(x, y, width, height, fill=fill, stroke=BLUE_DARK)
        canvas.text(x + 22, y + 42, title, css_class="section")
        canvas.multiline(
            x + 22, y + 82, lines, css_class="body", fill=MUTED, line_height=29
        )
        if index < len(stages) - 1:
            canvas.arrow(
                x + width + 8,
                y + height / 2,
                stages[index + 1][0] - 9,
                stages[index + 1][1] + stages[index + 1][3] / 2,
            )
    canvas.rect(120, 535, 420, 150, fill=WHITE, stroke=GRID)
    canvas.text(150, 577, "Receptive field", css_class="section")
    canvas.text(
        150,
        625,
        f"{facts.tcn_receptive_frames} TCN frames",
        css_class="metric",
        fill=BLUE_DARK,
    )
    canvas.text(
        150,
        657,
        f"{facts.model_receptive_frames} frames including pooling",
        css_class="small",
        fill=MUTED,
    )
    canvas.rect(590, 535, 420, 150, fill=WHITE, stroke=GRID)
    canvas.text(620, 577, "Explicit streaming state", css_class="section")
    canvas.text(
        620,
        625,
        f"{facts.state_values:,} float values",
        css_class="metric",
        fill=BLUE_DARK,
    )
    canvas.text(
        620, 657, "8 depthwise histories + pool history", css_class="small", fill=MUTED
    )
    canvas.rect(1060, 535, 420, 150, fill=WHITE, stroke=GRID)
    canvas.text(1090, 577, "Trainable parameters", css_class="section")
    canvas.text(
        1090, 625, f"{facts.parameter_count:,}", css_class="metric", fill=BLUE_DARK
    )
    canvas.text(
        1090, 657, "source constant + architecture test", css_class="small", fill=MUTED
    )
    canvas.rect(230, 730, 1140, 70, fill=ORANGE_LIGHT, stroke=ORANGE_DARK, radius=10)
    canvas.text(
        800,
        760,
        "Boundary: no successful registered neural benchmark and no "
        "selected checkpoint",
        css_class="body",
        fill=ORANGE_DARK,
        anchor="middle",
        weight=700,
    )
    canvas.text(
        800,
        787,
        "The diagram documents implemented architecture—not measured model quality.",
        css_class="small",
        fill=ORANGE_DARK,
        anchor="middle",
    )
    canvas.footer(
        "Sources: causal_kws.py · test_causal_kws.py",
        "Architecture evidence ≠ benchmark evidence",
    )
    return canvas.finish()


def _render_lineage(evidence: Evidence) -> bytes:
    canvas = Svg(
        1600,
        1160,
        "FalseWake experiment lineage",
        "Experiments 000 through 006 with measured, invoked, preflight-rejected, "
        "and terminal boundaries distinguished.",
    )
    canvas.header(
        "Experiment lineage and fail-closed boundaries",
        "A preserved engineering record: successful measurements, "
        "rejections, and incidents.",
        "FalseWake · execution history",
    )
    line_x = 170
    start_y = 235
    gap = 112
    canvas.line(line_x, start_y, line_x, start_y + gap * 6, stroke=GRID, width=5)
    for index, row in enumerate(evidence.lineage):
        y = start_y + index * gap
        preflight = row.boundary == "PREFLIGHT REJECTED"
        measured = row.experiment in {"000", "001"}
        color = BLUE if measured else ORANGE if not preflight else NEUTRAL_DARK
        fill = BLUE_LIGHT if measured else ORANGE_LIGHT if not preflight else PANEL
        canvas.circle(line_x, y, 20, fill=color)
        canvas.text(105, y + 7, row.experiment, css_class="section", anchor="middle")
        canvas.rect(235, y - 43, 1280, 86, fill=fill, stroke=color, radius=12)
        canvas.text(265, y - 10, row.boundary, css_class="label", fill=color)
        canvas.text(520, y - 9, row.outcome, css_class="body", weight=700)
        canvas.text(265, y + 25, row.detail, css_class="small", fill=MUTED)
    canvas.rect(235, 965, 1280, 62, fill=WHITE, stroke=GRID, radius=10)
    canvas.text(
        875,
        992,
        "Experiment 006: plausible observer interference was recorded, "
        "but overlap and root cause were not established.",
        css_class="small",
        fill=ORANGE_DARK,
        anchor="middle",
        weight=700,
    )
    canvas.text(
        875,
        1017,
        "Incidents are evidence of controls and limits—not successful "
        "scientific benchmarks.",
        css_class="small",
        fill=MUTED,
        anchor="middle",
    )
    canvas.footer(
        "Sources: committed result and incident JSON for Experiments 000–006",
        "No experiment executed by this renderer",
    )
    return canvas.finish()


def _render_threshold_tradeoff(evidence: Evidence) -> bytes:
    facts = evidence.threshold
    canvas = Svg(
        1600,
        1080,
        "Experiment 001 threshold trade-off",
        "Conditional correct retention and false events per hour across all 1,001 "
        "registered thresholds, recomputed from raw counts and denominators.",
    )
    canvas.header(
        "Experiment 001 threshold trade-off",
        f"1,001 thresholds · {facts.exposure_samples / SAMPLES_PER_HOUR:.3f} "
        "scored hours · raw-count recomputation",
        "FalseWake · measured replay",
    )
    left = 155
    right = 1510
    width = right - left
    top_a, bottom_a = 235, 525
    top_b, bottom_b = 635, 850
    for tick in range(0, 11):
        x = left + width * tick / 10
        canvas.line(x, top_a, x, bottom_a, stroke=GRID)
        canvas.line(x, top_b, x, bottom_b, stroke=GRID)
        canvas.text(
            x, 882, f"{tick / 10:.1f}", css_class="tiny", fill=MUTED, anchor="middle"
        )
    for tick in range(0, 6):
        value = tick / 5
        y = bottom_a - (bottom_a - top_a) * value
        canvas.line(left, y, right, y, stroke=GRID)
        canvas.text(
            left - 18, y + 5, f"{value:.0%}", css_class="tiny", fill=MUTED, anchor="end"
        )
    feh_ticks = (0, 1, 10, 100, 1_000)
    max_log = math.log10(1 + max(point.false_events_per_hour for point in facts.points))
    for value in feh_ticks:
        y = bottom_b - (bottom_b - top_b) * math.log10(1 + value) / max_log
        canvas.line(left, y, right, y, stroke=GRID)
        canvas.text(
            left - 18, y + 5, f"{value:,}", css_class="tiny", fill=MUTED, anchor="end"
        )
    retention_points = [
        (
            left + width * point.threshold_milli / 1_000,
            bottom_a - (bottom_a - top_a) * point.retention,
        )
        for point in facts.points
    ]
    feh_points = [
        (
            left + width * point.threshold_milli / 1_000,
            bottom_b
            - (bottom_b - top_b)
            * math.log10(1 + point.false_events_per_hour)
            / max_log,
        )
        for point in facts.points
    ]
    canvas.path(retention_points, stroke=BLUE, width=3)
    canvas.path(feh_points, stroke=ORANGE, width=3)
    canvas.text(left, 210, "Conditional correct retention", css_class="section")
    canvas.text(
        right,
        210,
        f"correct count / {facts.baseline_correct:,} baseline-correct clips",
        css_class="small",
        fill=MUTED,
        anchor="end",
    )
    canvas.text(left, 610, "False events per hour", css_class="section")
    canvas.text(
        right,
        610,
        "event count × 57,600,000 / exposure · log(1 + rate) display scale",
        css_class="small",
        fill=MUTED,
        anchor="end",
    )
    for threshold, color, label in (
        (facts.retention_frontier_milli, BLUE_DARK, "retention frontier 0.395"),
        (facts.negative_frontier_milli, ORANGE_DARK, "FEH frontier 0.991"),
    ):
        x = left + width * threshold / 1_000
        canvas.line(x, top_a, x, bottom_b, stroke=color, width=2, dash="7 6")
        canvas.text(
            x - 8, 570, label, css_class="tiny", fill=color, anchor="end", weight=700
        )
    canvas.line(
        left,
        bottom_a - (bottom_a - top_a) * 0.8,
        right,
        bottom_a - (bottom_a - top_a) * 0.8,
        stroke=BLUE_DARK,
        dash="5 5",
    )
    feh_one_y = bottom_b - (bottom_b - top_b) * math.log10(2) / max_log
    canvas.line(left, feh_one_y, right, feh_one_y, stroke=ORANGE_DARK, dash="5 5")
    canvas.text(
        (left + right) / 2,
        920,
        "Registered target-probability threshold",
        css_class="body",
        anchor="middle",
        weight=700,
    )
    canvas.text(
        (left + right) / 2,
        952,
        "No threshold lies in both pass regions; registered selection status: reject.",
        css_class="small",
        fill=ORANGE_DARK,
        anchor="middle",
        weight=700,
    )
    canvas.footer(
        "Source: experiment-001-dev-replay.json · values recomputed from "
        "raw integer counts",
        "Derived float fields are not trusted",
    )
    return canvas.finish()


def _render_unknown_words(evidence: Evidence) -> bytes:
    rows = evidence.unknown_words
    canvas = Svg(
        1600,
        1280,
        "Test unknown-word recall by sampled source word",
        "Horizontal bars for all 25 exact Experiment 000 test unknown-word rows, "
        "with predicted-unknown counts and support.",
    )
    canvas.header(
        "Test unknown-word recall by sampled source word",
        f"Experiment 000 test slice · n={sum(row.support for row in rows)} "
        f"across {len(rows)} source words · exact counts shown",
        "FalseWake · open-set diagnostic",
    )
    left = 245
    right = 1435
    top = 215
    row_height = 34
    chart_width = right - left
    for tick in range(0, 5):
        value = tick / 4
        x = left + chart_width * value
        canvas.line(x, top - 10, x, top + row_height * len(rows), stroke=GRID)
        canvas.text(
            x, top - 22, f"{value:.0%}", css_class="tiny", fill=MUTED, anchor="middle"
        )
    for index, row in enumerate(rows):
        y = top + index * row_height
        bar_width = chart_width * row.recall
        canvas.text(
            left - 18,
            y + 20,
            row.source_word,
            css_class="small",
            anchor="end",
            weight=700,
        )
        canvas.rect(
            left,
            y + 3,
            max(bar_width, 1),
            22,
            fill=BLUE if bar_width else WHITE,
            stroke=BLUE_DARK,
            radius=3,
            stroke_width=1,
        )
        label_x = min(left + bar_width + 12, right - 5)
        anchor = "end" if label_x >= right - 5 else "start"
        canvas.text(
            label_x,
            y + 20,
            f"{row.predicted_unknown_count}/{row.support} · {row.recall:.1%}",
            css_class="tiny",
            fill=INK,
            anchor=anchor,
            weight=700,
        )
    canvas.rect(245, 1082, 1190, 62, fill=ORANGE_LIGHT, stroke=ORANGE_DARK, radius=10)
    canvas.text(
        840,
        1109,
        "Caveat: this is a deterministically sampled test slice, not a "
        "population-level per-word estimate.",
        css_class="small",
        fill=ORANGE_DARK,
        anchor="middle",
        weight=700,
    )
    canvas.text(
        840,
        1134,
        (
            f"Per-word supports are small ({min(row.support for row in rows)}–"
            f"{max(row.support for row in rows)} clips); compare exact "
            "denominators."
        ),
        css_class="small",
        fill=ORANGE_DARK,
        anchor="middle",
    )
    canvas.footer(
        "Source: experiment-000-linear.json · test.unknown_by_source_word",
        "Argmax clips; no streaming threshold",
    )
    return canvas.finish()


def _render_setup(evidence: Evidence) -> bytes:
    canvas = Svg(
        1600,
        1000,
        "FalseWake setup and safe verification",
        "Hash-locked setup, safe evidence checks, and the explicit boundary against "
        "running registered experiments 002 through 006.",
    )
    canvas.header(
        "Setup and safe verification",
        f"Python {evidence.python_version} · hash-locked dependencies · "
        "deterministic evidence checks",
        "FalseWake · reproducible workflow",
    )
    columns = (
        (75, 225, 455, 540, "1 · CREATE ENVIRONMENT", BLUE_LIGHT, BLUE_DARK),
        (572, 225, 455, 540, "2 · VERIFY SAFE EVIDENCE", PANEL, BLUE_DARK),
        (
            1069,
            225,
            455,
            540,
            "3 · RESPECT EXECUTION BOUNDARY",
            ORANGE_LIGHT,
            ORANGE_DARK,
        ),
    )
    for x, y, width, height, title, fill, stroke in columns:
        canvas.rect(x, y, width, height, fill=fill, stroke=stroke)
        canvas.text(x + 28, y + 45, title, css_class="label", fill=stroke)
    canvas.multiline(
        103,
        326,
        (
            "python3.12 -m venv .venv",
            ".venv/bin/python -m pip install \\",
            "  --require-hashes -r requirements-dev.lock",
            ".venv/bin/python -m pip install \\",
            "  --no-deps --no-build-isolation -e .",
        ),
        css_class="small mono",
        fill=INK,
        line_height=34,
    )
    canvas.text(103, 550, "Locked inputs", css_class="section")
    canvas.multiline(
        103,
        587,
        (
            ".python-version → 3.12.3",
            "requirements-dev.lock → hashes",
            "requirements-train-ci.lock → CPU hashes",
            "pyproject.toml → Python ≥3.12",
        ),
        css_class="body",
        fill=MUTED,
        line_height=32,
    )
    canvas.multiline(
        600,
        326,
        (
            "python tools/render_portfolio_visuals.py check",
            "",
            "python -m pytest -q \\",
            "  tests/test_result_plots.py \\",
            "  tests/test_portfolio_visuals.py",
        ),
        css_class="small mono",
        fill=INK,
        line_height=34,
    )
    canvas.text(600, 550, "Check performs", css_class="section")
    canvas.multiline(
        600,
        587,
        (
            "• recompute report-000/001 plots",
            "• byte-compare 3 SVG + 3 PNG",
            "• verify 7 SVG + provenance",
            "• read only the static source allowlist",
        ),
        css_class="body",
        fill=MUTED,
        line_height=32,
    )
    canvas.text(1097, 326, "SAFE", css_class="label", fill=BLUE_DARK)
    canvas.multiline(
        1097,
        365,
        (
            "Plotting Experiments 000/001",
            "Static source/report inspection",
            "Ruff · strict mypy · portable tests",
        ),
        css_class="body",
        fill=INK,
        line_height=34,
    )
    canvas.line(1097, 470, 1494, 470, stroke=ORANGE_DARK, width=2)
    canvas.text(1097, 518, "NOT PERFORMED", css_class="label", fill=ORANGE_DARK)
    canvas.multiline(
        1097,
        557,
        (
            "No Experiment 002–006 import",
            "No runner / coordinator / supervisor",
            "No seed worker / authority issuer",
            "No attempt-marker access",
            "No registered experiment execution",
        ),
        css_class="body",
        fill=ORANGE_DARK,
        line_height=34,
        weight=700,
    )
    canvas.rect(185, 825, 1230, 72, fill=WHITE, stroke=GRID, radius=10)
    canvas.text(
        800,
        855,
        "CI repeats Ruff, evidence verification, focused render tests, "
        "strict mypy, and portable tests.",
        css_class="body",
        anchor="middle",
        weight=700,
    )
    canvas.text(
        800,
        884,
        "The evidence renderer never turns documentation verification "
        "into an experiment attempt.",
        css_class="small",
        fill=MUTED,
        anchor="middle",
    )
    canvas.footer(
        "Sources: .python-version · dev/train locks · pyproject.toml · CI workflow",
        "Safe verification only",
    )
    return canvas.finish()


def _visuals(evidence: Evidence) -> dict[str, bytes]:
    return {
        "system-path.svg": _render_system_path(evidence),
        "data-boundaries.svg": _render_data_boundaries(evidence),
        "causal-tcn-state.svg": _render_causal_tcn(evidence),
        "experiment-lineage.svg": _render_lineage(evidence),
        "threshold-tradeoff.svg": _render_threshold_tradeoff(evidence),
        "unknown-word-recall.svg": _render_unknown_words(evidence),
        "setup-verification.svg": _render_setup(evidence),
    }


def _verify_report_assets(root: Path) -> dict[str, dict[str, object]]:
    """Regenerate only reviewed 000/001 plots and compare all committed bytes."""

    modules_before = frozenset(sys.modules)
    try:
        from falsewake.result_plots import (
            ResultPlotError,
            render_replay_feasibility,
            render_result_plots,
        )
    except ImportError as error:
        raise PortfolioVisualError(
            f"cannot import safe plot renderer: {error}"
        ) from error
    newly_imported = frozenset(sys.modules) - modules_before
    forbidden = sorted(
        name for name in newly_imported if name.startswith("falsewake.experiment_")
    )
    if forbidden:
        raise PortfolioVisualError(
            f"safe plot import loaded forbidden experiment modules: {forbidden}"
        )

    temporary_root = root.parent / ".t"
    try:
        temporary_root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise PortfolioVisualError(
            f"cannot create workspace temp root: {error}"
        ) from error
    records: dict[str, dict[str, object]] = {}
    try:
        with tempfile.TemporaryDirectory(
            prefix="falsewake-portfolio-plots-", dir=temporary_root
        ) as raw_directory:
            output = Path(raw_directory)
            render_result_plots(
                _source_path(root, "reports/experiment-000-linear.json"), output
            )
            render_replay_feasibility(
                _source_path(root, "reports/experiment-001-dev-replay.json"), output
            )
            newly_imported = frozenset(sys.modules) - modules_before
            forbidden = sorted(
                name
                for name in newly_imported
                if name.startswith("falsewake.experiment_")
            )
            if forbidden:
                raise PortfolioVisualError(
                    f"plot verification loaded forbidden modules: {forbidden}"
                )
            for name in REPORT_ASSETS:
                committed = root / "reports" / name
                if committed.is_symlink() or not committed.is_file():
                    raise PortfolioVisualError(
                        f"report asset is missing: reports/{name}"
                    )
                regenerated = (output / name).read_bytes()
                observed = committed.read_bytes()
                if regenerated != observed:
                    raise PortfolioVisualError(f"report asset is stale: reports/{name}")
                source_report = (
                    "reports/experiment-001-dev-replay.json"
                    if name.startswith("experiment-001")
                    else "reports/experiment-000-linear.json"
                )
                records[f"reports/{name}"] = {
                    "sha256": _sha256(observed),
                    "source_records": _source_records(
                        root,
                        (
                            source_report,
                            "requirements-dev.lock",
                            "src/falsewake/result_plots.py",
                        ),
                    ),
                    "verified_byte_current": True,
                }
    except (OSError, ResultPlotError) as error:
        if isinstance(error, PortfolioVisualError):
            raise
        raise PortfolioVisualError(f"cannot verify report assets: {error}") from error
    return records


def _provenance(
    root: Path,
    evidence: Evidence,
    visuals: Mapping[str, bytes],
    report_assets: Mapping[str, dict[str, object]],
) -> bytes:
    threshold = evidence.threshold
    key_thresholds = {
        f"{point.threshold_milli / 1_000:.3f}": {
            "correct_count": point.correct_count,
            "event_count": point.event_count,
            "false_events_per_hour_recomputed": point.false_events_per_hour,
            "retention_recomputed": point.retention,
        }
        for point in threshold.points
        if point.threshold_milli in {0, 395, 396, 990, 991, 1000}
    }
    unknown_rows = [
        {
            "predicted_unknown_count": row.predicted_unknown_count,
            "recall_recomputed": row.recall,
            "source_word": row.source_word,
            "support": row.support,
        }
        for row in evidence.unknown_words
    ]
    specs: dict[str, dict[str, object]] = {
        "system-path.svg": {
            "facts": {
                "clip_samples": evidence.frontend.clip_samples,
                "frame_count": evidence.frontend.frame_count,
                "frontend_hop_samples": evidence.frontend.hop_samples,
                "mel_bins": evidence.frontend.mel_bins,
                "replay_hop_samples": evidence.frontend.replay_hop_samples,
                "sample_rate": evidence.frontend.sample_rate,
                "summary_values": evidence.frontend.summary_values,
                "threshold_count": evidence.frontend.threshold_count,
                "window_samples": evidence.frontend.window_samples,
            },
            "nonclaims": ["Measured baseline path only."],
            "sources": (
                "src/falsewake/features.py",
                "src/falsewake/continuous_replay.py",
                "reports/experiment-000-linear.json",
                "reports/experiment-001-dev-replay.json",
            ),
        },
        "data-boundaries.svg": {
            "facts": {
                "baseline_correct": evidence.dataset.baseline_correct,
                "dev_exposure_samples": evidence.dataset.dev_exposure_samples,
                "dev_speakers": evidence.dataset.dev_speakers,
                "dev_utterances": evidence.dataset.dev_utterances,
                "headline_split": evidence.dataset.headline_split,
                "positive_validation_examples": (
                    evidence.dataset.positive_validation_examples
                ),
                "test_examples": evidence.dataset.test_examples,
                "training_examples": evidence.dataset.training_examples,
                "validation_examples": evidence.dataset.validation_examples,
                "validation_role": evidence.dataset.validation_role,
            },
            "nonclaims": ["No test-set threshold tuning."],
            "sources": (
                "reports/experiment-000-linear.json",
                "reports/experiment-001-dev-replay.json",
            ),
        },
        "causal-tcn-state.svg": {
            "facts": {
                "channels": evidence.causal.channels,
                "dilations": list(evidence.causal.dilations),
                "kernel_size": evidence.causal.kernel_size,
                "mel_bins": evidence.causal.mel_bins,
                "model_receptive_frames": evidence.causal.model_receptive_frames,
                "output_classes": evidence.causal.output_classes,
                "parameter_count": evidence.causal.parameter_count,
                "pool_frames": evidence.causal.pool_frames,
                "state_values": evidence.causal.state_values,
                "tcn_receptive_frames": evidence.causal.tcn_receptive_frames,
            },
            "nonclaims": [
                "Implemented and architecture-tested only.",
                "No successful registered neural benchmark.",
                "No selected checkpoint.",
            ],
            "sources": (
                "src/falsewake/causal_kws.py",
                "tests/test_causal_kws.py",
            ),
        },
        "experiment-lineage.svg": {
            "facts": {
                "rows": [
                    {
                        "boundary": row.boundary,
                        "detail": row.detail,
                        "experiment": row.experiment,
                        "outcome": row.outcome,
                    }
                    for row in evidence.lineage
                ]
            },
            "nonclaims": [
                "The renderer does not execute Experiments 002-006.",
                "Experiment 006 root cause is undetermined from retained evidence.",
                "Incidents are not successful scientific benchmarks.",
            ],
            "sources": tuple(
                "reports/experiment-"
                f"{number}-"
                f"{'preflight-' if number in {'004', '005'} else 'execution-'}"
                "incident.json"
                for number in ("002", "003", "004", "005", "006")
            )
            + (
                "reports/experiment-000-linear.json",
                "reports/experiment-001-dev-replay.json",
            ),
        },
        "threshold-tradeoff.svg": {
            "facts": {
                "baseline_correct": threshold.baseline_correct,
                "false_events_per_hour_formula": (
                    "dev_event_count * 57600000 / scored_exposure_samples"
                ),
                "key_thresholds": key_thresholds,
                "negative_frontier_milli": threshold.negative_frontier_milli,
                "point_count": len(threshold.points),
                "retention_formula": (
                    "validation_correct_accept_count / baseline_correct_count"
                ),
                "retention_frontier_milli": threshold.retention_frontier_milli,
                "scored_exposure_samples": threshold.exposure_samples,
                "selection_status": threshold.selection_status,
            },
            "nonclaims": [
                "Derived false-events/hour and retention fields are not trusted.",
                "No registered threshold satisfies both gates.",
            ],
            "sources": ("reports/experiment-001-dev-replay.json",),
        },
        "unknown-word-recall.svg": {
            "facts": {
                "correct_total": sum(
                    row.predicted_unknown_count for row in evidence.unknown_words
                ),
                "row_count": len(evidence.unknown_words),
                "rows": unknown_rows,
                "support_total": sum(row.support for row in evidence.unknown_words),
            },
            "nonclaims": [
                "Deterministically sampled test slice, not a "
                "population-level estimate.",
                "Argmax clips; no continuous-stream threshold.",
            ],
            "sources": ("reports/experiment-000-linear.json",),
        },
        "setup-verification.svg": {
            "facts": {
                "ci_checks": list(evidence.ci_checks),
                "python_version": evidence.python_version,
                "report_assets_byte_compared": len(REPORT_ASSETS),
                "visual_assets_checked": len(VISUAL_FILENAMES),
            },
            "nonclaims": [
                "No Experiment 002-006 imports or execution.",
                "No runner, coordinator, supervisor, seed worker, or authority issuer.",
                "No attempt-marker access.",
            ],
            "sources": (
                ".github/workflows/ci.yml",
                ".python-version",
                "pyproject.toml",
                "requirements-dev.lock",
                "requirements-train-ci.lock",
                "src/falsewake/result_plots.py",
            ),
        },
    }
    visual_records: dict[str, dict[str, object]] = {}
    for name in VISUAL_FILENAMES:
        spec = specs[name]
        source_paths = cast(Sequence[str], spec.pop("sources"))
        visual_records[name] = {
            **spec,
            "output_sha256": _sha256(visuals[name]),
            "path": f"docs/images/readme/{name}",
            "source_records": _source_records(root, source_paths),
        }
    document = {
        "existing_report_assets": dict(sorted(report_assets.items())),
        "renderer": {
            "command": "python tools/render_portfolio_visuals.py check",
            "name": "FalseWake evidence-bound portfolio renderer",
            "path": "tools/render_portfolio_visuals.py",
            "sha256": _sha256(
                _read_source(root, "tools/render_portfolio_visuals.py")
            ),
            "version": RENDERER_VERSION,
        },
        "schema_version": 1,
        "source_allowlist": sorted(SOURCE_ALLOWLIST),
        "visuals": visual_records,
    }
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


def build_bundle(root: Path = REPOSITORY_ROOT) -> PortfolioBundle:
    """Build every asset in memory and verify the six existing report plots."""

    evidence = extract_evidence(root)
    visuals = _visuals(evidence)
    if tuple(sorted(visuals)) != tuple(sorted(VISUAL_FILENAMES)):
        raise PortfolioVisualError("renderer output set differs from the contract")
    report_assets = _verify_report_assets(root)
    provenance = _provenance(root, evidence, visuals, report_assets)
    return PortfolioBundle(visuals=visuals, provenance=provenance)


def _write_if_changed(path: Path, contents: bytes) -> None:
    if path.is_symlink():
        raise PortfolioVisualError(f"refusing to replace symlink output: {path.name}")
    try:
        if path.exists() and path.read_bytes() == contents:
            if path.stat().st_mode & 0o777 != 0o644:
                path.chmod(0o644)
            return
    except OSError as error:
        raise PortfolioVisualError(
            f"cannot inspect output {path.name}: {error}"
        ) from error

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(contents)
            stream.flush()
        temporary_path.chmod(0o644)
        temporary_path.replace(path)
    except OSError as error:
        raise PortfolioVisualError(
            f"cannot publish output {path.name}: {error}"
        ) from error
    finally:
        if temporary_path is not None:
            with suppress(OSError):
                temporary_path.unlink(missing_ok=True)


def write_assets(
    root: Path = REPOSITORY_ROOT, output_directory: Path = OUTPUT_DIRECTORY
) -> None:
    """Write the complete deterministic portfolio-visual bundle."""

    bundle = build_bundle(root)
    output_directory.mkdir(parents=True, exist_ok=True)
    for name, contents in bundle.visuals.items():
        _write_if_changed(output_directory / name, contents)
    _write_if_changed(output_directory / "provenance.json", bundle.provenance)


def check_assets(
    root: Path = REPOSITORY_ROOT, output_directory: Path = OUTPUT_DIRECTORY
) -> None:
    """Fail if any committed portfolio visual or provenance byte is stale."""

    bundle = build_bundle(root)
    expected = {**bundle.visuals, "provenance.json": bundle.provenance}
    extras = sorted(
        path.name
        for path in output_directory.glob("*")
        if path.is_file() and path.name not in expected
    )
    if extras:
        raise PortfolioVisualError(f"unexpected README visual assets: {extras}")
    stale: list[str] = []
    for name, contents in expected.items():
        path = output_directory / name
        if not path.is_file() or path.is_symlink() or path.read_bytes() != contents:
            stale.append(name)
    if stale:
        raise PortfolioVisualError(
            "portfolio visual bundle is stale; run write: " + ", ".join(stale)
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render or verify evidence-bound FalseWake portfolio visuals."
    )
    parser.add_argument("command", choices=("write", "check"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fixed repository-local write/check interface."""

    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "write":
            write_assets()
            print(
                f"wrote {len(VISUAL_FILENAMES)} SVGs and provenance; "
                f"verified {len(REPORT_ASSETS)} report assets"
            )
        else:
            check_assets()
            print(
                f"verified {len(VISUAL_FILENAMES)} SVGs, provenance, and "
                f"{len(REPORT_ASSETS)} report assets"
            )
    except PortfolioVisualError as error:
        print(f"portfolio visual verification failed: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
