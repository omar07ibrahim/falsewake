"""Inspect FalseWake's immutable, tracked evidence without running experiments.

The command has a deliberately closed read surface.  It accepts no repository,
report, or output path and reads only the hash-pinned JSON records in
``SOURCE_SPECS``.  It imports no FalseWake module and never invokes Git,
subprocesses, experiment runners, coordinators, supervisors, authority issuers,
or attempt-marker code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Final, NoReturn, cast


class EvidenceInspectionError(ValueError):
    """A fixed evidence source failed its closed-world validation."""


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """One immutable source admitted to the inspector."""

    source_id: str
    directories: tuple[str, ...]
    filename: str
    byte_count: int
    sha256: str
    role: str

    @property
    def relative_path(self) -> str:
        """Return the stable repository-relative provenance path."""

        return "/".join((*self.directories, self.filename))


SOURCE_SPECS: Final = (
    SourceSpec(
        "experiment_000_report",
        ("reports",),
        "experiment-000-linear.json",
        20_042,
        "3ad2e6d630bc62810bb7a88418dd018cd0d7acd65b4173caa912af1fe33fcaf9",
        "measured clip-classification report",
    ),
    SourceSpec(
        "experiment_001_replay",
        ("reports",),
        "experiment-001-dev-replay.json",
        1_456_860,
        "b8e30e26498af2600ece01f4cceb436e10a546b8390f039ca8a23cda45f00f2d",
        "measured development replay and threshold grid",
    ),
    SourceSpec(
        "experiment_001_selection",
        ("reports",),
        "experiment-001-selection.json",
        828,
        "1d44ae6ff06a5fab1567d0342299e293fe001b8c91f9972cfa8e79a2dabf2318",
        "tracked threshold-selection disposition",
    ),
    SourceSpec(
        "experiment_002_terminal_report",
        ("reports",),
        "experiment-002-training.json",
        637,
        "494336d12e47f7bce952251e4c079770920f57032354e9618050250ee32bb99a",
        "automatic registered-execution terminal report",
    ),
    SourceSpec(
        "experiment_002_incident",
        ("reports",),
        "experiment-002-execution-incident.json",
        2_682,
        "353e33acc156e6e3a598d3dd777afbe5b15125b1b24f7f48266859f09f297b59",
        "registered-execution incident record",
    ),
    SourceSpec(
        "experiment_003_incident",
        ("reports",),
        "experiment-003-execution-incident.json",
        4_539,
        "0dc24fd211129cd2b81e29fb91ac9e66a0aad25f1deeae332f3b6ea420567688",
        "registered-admission incident record",
    ),
    SourceSpec(
        "experiment_004_incident",
        ("reports",),
        "experiment-004-preflight-incident.json",
        6_431,
        "d015b749acf6e98e87172372315c78ccfaad48b8ae6b3a29ae426890aa467f50",
        "pre-registration rejection record",
    ),
    SourceSpec(
        "experiment_005_incident",
        ("reports",),
        "experiment-005-preflight-incident.json",
        6_612,
        "3de6fbbd6912bee47f3ef01f5371b57ae8d5942b6fc9915fc0ec243b9e931e3d",
        "pre-registration rejection record",
    ),
    SourceSpec(
        "experiment_006_incident",
        ("reports",),
        "experiment-006-execution-incident.json",
        8_839,
        "c5cf002ea876890e3b3769af2bcc850433d3d08ac1071c864ed69ee2172af77d",
        "manual post-exit terminal incident record",
    ),
    SourceSpec(
        "visual_provenance",
        ("docs", "images", "readme"),
        "provenance.json",
        21_186,
        "8cf7c41bef86c4478b7993b1fbae3bfba5c34819250698d29e6e5107c8705082",
        "tracked visual source-hash provenance",
    ),
)

_SOURCE_BY_ID: Final = {spec.source_id: spec for spec in SOURCE_SPECS}
_REPORT_SOURCE_IDS: Final = tuple(
    spec.source_id for spec in SOURCE_SPECS if spec.source_id != "visual_provenance"
)

_EXPERIMENT_000_KEYS: Final = frozenset(
    {
        "class_order",
        "evaluation_seconds",
        "examples_sha256",
        "experiment_config_sha256",
        "features_sha256",
        "fit",
        "headline_split",
        "manifest_sha256",
        "portable_model_sha256",
        "runtime",
        "schema_version",
        "scikit_learn_version",
        "splits",
        "threshold_metrics",
        "validation_role",
    }
)
_SPLIT_KEYS: Final = frozenset(
    {
        "accuracy",
        "confusion_matrix",
        "correct",
        "example_count",
        "macro_f1",
        "open_set_target_prediction",
        "per_class",
        "target_argmax_error",
        "unknown_by_source_word",
    }
)
_THRESHOLD_ROW_KEYS: Final = frozenset(
    {
        "conditional_correct_retention",
        "correct_accept_recall",
        "dev_event_count",
        "dev_event_count_by_target",
        "false_events_per_hour",
        "garwood_95_percent",
        "speaker_bootstrap_95_percent",
        "threshold_milli",
        "validation_correct_accept_count",
        "validation_correct_accept_count_by_target",
    }
)


def _fail(code: str) -> NoReturn:
    """Raise a stable, path-free failure."""

    raise EvidenceInspectionError(code)


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if type(value) is not dict:
        _fail(f"schema:{name}:mapping")
    raw = cast(dict[object, object], value)
    if not all(type(key) is str for key in raw):
        _fail(f"schema:{name}:string_keys")
    return cast(dict[str, object], raw)


def _list(value: object, *, name: str) -> list[object]:
    if type(value) is not list:
        _fail(f"schema:{name}:list")
    return cast(list[object], value)


def _text(value: object, *, name: str) -> str:
    if type(value) is not str:
        _fail(f"schema:{name}:text")
    return value


def _integer(value: object, *, name: str) -> int:
    if type(value) is not int:
        _fail(f"schema:{name}:integer")
    return value


def _number(value: object, *, name: str) -> float:
    if type(value) not in (int, float):
        _fail(f"schema:{name}:number")
    number = float(cast(int | float, value))
    if not math.isfinite(number):
        _fail(f"schema:{name}:finite")
    return number


def _boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        _fail(f"schema:{name}:boolean")
    return value


def _keys(value: Mapping[str, object], expected: Iterable[str], *, name: str) -> None:
    if frozenset(value) != frozenset(expected):
        _fail(f"schema:{name}:keys")


def _equals(value: object, expected: object, *, name: str) -> None:
    if type(value) is not type(expected) or value != expected:
        _fail(f"schema:{name}:value")


def _expect_empty_list(value: object, *, name: str) -> None:
    if _list(value, name=name):
        _fail(f"schema:{name}:nonempty")


def _reject_json_constant(value: str) -> NoReturn:
    del value
    _fail("json:nonfinite")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail("json:duplicate_key")
        result[key] = value
    return result


def _parse_json(contents: bytes, *, source_id: str) -> dict[str, object]:
    try:
        text = contents.decode("ascii")
    except UnicodeDecodeError:
        _fail(f"source:{source_id}:ascii")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, RecursionError):
        _fail(f"source:{source_id}:json")
    return _mapping(value, name=source_id)


def _repository_root() -> Path:
    try:
        script = Path(__file__).resolve(strict=True)
    except OSError:
        _fail("repository:script_resolution")
    root = script.parents[1]
    if script.parent.name != "tools":
        _fail("repository:script_location")
    return root


def _no_follow_flag() -> int:
    flag = getattr(os, "O_NOFOLLOW", None)
    if type(flag) is not int or flag == 0:
        _fail("platform:o_nofollow_unavailable")
    return flag


def _directory_flags() -> int:
    return os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | _no_follow_flag()


def _file_flags() -> int:
    return os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK | _no_follow_flag()


def _open_root() -> int:
    try:
        descriptor = os.open(_repository_root(), _directory_flags())
    except OSError:
        _fail("repository:open")
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            _fail("repository:not_directory")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _open_child_directory(parent_fd: int, name: str, *, source_id: str) -> int:
    if not name or "/" in name or name in {".", ".."}:
        _fail(f"source:{source_id}:directory_contract")
    try:
        descriptor = os.open(name, _directory_flags(), dir_fd=parent_fd)
    except OSError:
        _fail(f"source:{source_id}:directory_open")
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            _fail(f"source:{source_id}:directory_type")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _stat_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_bound_source(root_fd: int, spec: SourceSpec) -> bytes:
    """Read one fixed regular file through pinned, no-follow directory handles."""

    descriptors: list[int] = []
    parent_fd = root_fd
    try:
        for directory in spec.directories:
            parent_fd = _open_child_directory(
                parent_fd, directory, source_id=spec.source_id
            )
            descriptors.append(parent_fd)
        if not spec.filename or "/" in spec.filename or spec.filename in {".", ".."}:
            _fail(f"source:{spec.source_id}:filename_contract")
        try:
            file_fd = os.open(spec.filename, _file_flags(), dir_fd=parent_fd)
        except OSError:
            _fail(f"source:{spec.source_id}:file_open")
        descriptors.append(file_fd)

        before = os.fstat(file_fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink < 1
            or before.st_size != spec.byte_count
        ):
            _fail(f"source:{spec.source_id}:file_identity")
        chunks: list[bytes] = []
        remaining = spec.byte_count + 1
        while remaining:
            try:
                chunk = os.read(file_fd, min(remaining, 131_072))
            except OSError:
                _fail(f"source:{spec.source_id}:read")
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        contents = b"".join(chunks)
        after = os.fstat(file_fd)
        if _stat_identity(before) != _stat_identity(after):
            _fail(f"source:{spec.source_id}:changed_during_read")
        if len(contents) != spec.byte_count:
            _fail(f"source:{spec.source_id}:byte_count")
        if hashlib.sha256(contents).hexdigest() != spec.sha256:
            _fail(f"source:{spec.source_id}:sha256")
        return contents
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _load_sources(root_fd: int) -> dict[str, dict[str, object]]:
    documents: dict[str, dict[str, object]] = {}
    for spec in SOURCE_SPECS:
        contents = _read_bound_source(root_fd, spec)
        documents[spec.source_id] = _parse_json(contents, source_id=spec.source_id)
    return documents


def _schema_version(document: Mapping[str, object], *, name: str) -> None:
    _equals(document.get("schema_version"), 1, name=f"{name}.schema_version")


def _validate_experiment_000(
    report: dict[str, object],
) -> dict[str, object]:
    _schema_version(report, name="000")
    _keys(report, _EXPERIMENT_000_KEYS, name="000")
    _equals(report.get("headline_split"), "test", name="000.headline_split")
    _equals(
        report.get("validation_role"),
        "diagnostic_only_no_selection",
        name="000.validation_role",
    )
    _equals(
        report.get("threshold_metrics"),
        "not_evaluated",
        name="000.threshold_metrics",
    )
    fit = _mapping(report.get("fit"), name="000.fit")
    _keys(
        fit,
        {"fit_seconds", "n_iter", "scaler_fit_examples", "training_examples"},
        name="000.fit",
    )
    _equals(fit.get("training_examples"), 36_941, name="000.training_examples")
    _equals(fit.get("scaler_fit_examples"), 36_941, name="000.scaler_fit_examples")

    splits = _mapping(report.get("splits"), name="000.splits")
    _keys(splits, {"validation", "test"}, name="000.splits")

    def split_payload(split_name: str) -> dict[str, object]:
        split = _mapping(splits.get(split_name), name=f"000.{split_name}")
        _keys(split, _SPLIT_KEYS, name=f"000.{split_name}")
        examples = _integer(
            split.get("example_count"), name=f"000.{split_name}.example_count"
        )
        accuracy = _number(split.get("accuracy"), name=f"000.{split_name}.accuracy")
        macro_f1 = _number(split.get("macro_f1"), name=f"000.{split_name}.macro_f1")
        target_error = _mapping(
            split.get("target_argmax_error"),
            name=f"000.{split_name}.target_argmax_error",
        )
        _keys(
            target_error,
            {"error_count", "error_rate", "support"},
            name=f"000.{split_name}.target_argmax_error",
        )
        open_set = _mapping(
            split.get("open_set_target_prediction"),
            name=f"000.{split_name}.open_set",
        )
        _keys(open_set, {"unknown", "silence"}, name=f"000.{split_name}.open_set")

        def open_set_payload(label: str) -> dict[str, object]:
            row = _mapping(
                open_set.get(label), name=f"000.{split_name}.open_set.{label}"
            )
            _keys(
                row,
                {"predicted_target_count", "predicted_target_rate", "support"},
                name=f"000.{split_name}.open_set.{label}",
            )
            return {
                "predicted_target_count": _integer(
                    row.get("predicted_target_count"),
                    name=f"000.{split_name}.{label}.count",
                ),
                "predicted_target_rate": _number(
                    row.get("predicted_target_rate"),
                    name=f"000.{split_name}.{label}.rate",
                ),
                "support": _integer(
                    row.get("support"), name=f"000.{split_name}.{label}.support"
                ),
            }

        return {
            "accuracy": accuracy,
            "example_count": examples,
            "macro_f1": macro_f1,
            "open_set_target_prediction": {
                "silence": open_set_payload("silence"),
                "unknown": open_set_payload("unknown"),
            },
            "target_argmax_error": {
                "error_count": _integer(
                    target_error.get("error_count"),
                    name=f"000.{split_name}.target_error.count",
                ),
                "error_rate": _number(
                    target_error.get("error_rate"),
                    name=f"000.{split_name}.target_error.rate",
                ),
                "support": _integer(
                    target_error.get("support"),
                    name=f"000.{split_name}.target_error.support",
                ),
            },
        }

    validation = split_payload("validation")
    test = split_payload("test")
    _equals(validation["example_count"], 4_429, name="000.validation.examples")
    _equals(test["example_count"], 4_884, name="000.test.examples")
    return {
        "disposition": "measured_clip_baseline",
        "evidence_class": "measured_clip_metrics",
        "experiment": "000",
        "metrics": {"test": test, "validation": validation},
        "nonclaims": [
            "Acceptance thresholds and continuous-stream metrics were not evaluated.",
            "Validation was diagnostic only and did not select the fixed model.",
        ],
        "sources": ["experiment_000_report"],
    }


def _threshold_payload(row: object, *, expected_milli: int) -> dict[str, object]:
    threshold = _mapping(row, name=f"001.thresholds[{expected_milli}]")
    _keys(
        threshold,
        _THRESHOLD_ROW_KEYS,
        name=f"001.thresholds[{expected_milli}]",
    )
    _equals(
        threshold.get("threshold_milli"),
        expected_milli,
        name=f"001.thresholds[{expected_milli}].milli",
    )
    return {
        "conditional_correct_retention": _number(
            threshold.get("conditional_correct_retention"),
            name=f"001.thresholds[{expected_milli}].retention",
        ),
        "dev_event_count": _integer(
            threshold.get("dev_event_count"),
            name=f"001.thresholds[{expected_milli}].events",
        ),
        "false_events_per_hour": _number(
            threshold.get("false_events_per_hour"),
            name=f"001.thresholds[{expected_milli}].rate",
        ),
        "threshold_milli": expected_milli,
        "validation_correct_accept_count": _integer(
            threshold.get("validation_correct_accept_count"),
            name=f"001.thresholds[{expected_milli}].correct",
        ),
    }


def _validate_experiment_001(
    replay: dict[str, object], selection_record: dict[str, object]
) -> dict[str, object]:
    _schema_version(replay, name="001.replay")
    _keys(
        replay,
        {
            "dev",
            "identities",
            "positive_validation",
            "runtime",
            "schema_version",
            "selection",
            "target_order",
            "thresholds",
            "top_false_events",
        },
        name="001.replay",
    )
    _schema_version(selection_record, name="001.selection_record")
    _keys(
        selection_record,
        {
            "dev_archive_sha256",
            "dev_audit_report_sha256",
            "dev_manifest_sha256",
            "dev_replay_report_sha256",
            "experiment_config_sha256",
            "implementation_git_commit",
            "runtime_identity_sha256",
            "schema_version",
            "scorer_source_sha256",
            "selected_threshold_milli",
            "status",
        },
        name="001.selection_record",
    )
    _equals(
        selection_record.get("dev_replay_report_sha256"),
        _SOURCE_BY_ID["experiment_001_replay"].sha256,
        name="001.selection_record.replay_sha256",
    )
    _equals(selection_record.get("status"), "reject", name="001.selection.status")
    if selection_record.get("selected_threshold_milli") is not None:
        _fail("schema:001.selection.selected_threshold:value")

    selection = _mapping(replay.get("selection"), name="001.selection")
    _keys(
        selection,
        {
            "negative_frontier_milli",
            "retention_frontier_milli",
            "selected_threshold_milli",
            "status",
        },
        name="001.selection",
    )
    _equals(selection.get("status"), "reject", name="001.replay.status")
    _equals(
        selection.get("retention_frontier_milli"),
        395,
        name="001.retention_frontier",
    )
    _equals(
        selection.get("negative_frontier_milli"),
        991,
        name="001.negative_frontier",
    )
    if selection.get("selected_threshold_milli") is not None:
        _fail("schema:001.replay.selected_threshold:value")

    dev = _mapping(replay.get("dev"), name="001.dev")
    _keys(
        dev,
        {
            "scored_exposure_samples",
            "source_samples",
            "speaker_count",
            "speaker_rows",
            "utterance_count",
        },
        name="001.dev",
    )
    positive = _mapping(
        replay.get("positive_validation"), name="001.positive_validation"
    )
    _keys(
        positive,
        {
            "baseline_correct_by_target",
            "baseline_correct_count",
            "support_by_target",
            "target_example_count",
        },
        name="001.positive_validation",
    )
    _equals(
        positive.get("baseline_correct_count"),
        2_088,
        name="001.baseline_correct",
    )
    _equals(
        positive.get("target_example_count"),
        3_703,
        name="001.target_examples",
    )
    thresholds = _list(replay.get("thresholds"), name="001.thresholds")
    if len(thresholds) != 1_001:
        _fail("schema:001.thresholds:length")
    for expected_milli, raw_row in enumerate(thresholds):
        row = _mapping(raw_row, name=f"001.thresholds[{expected_milli}]")
        _keys(
            row,
            _THRESHOLD_ROW_KEYS,
            name=f"001.thresholds[{expected_milli}]",
        )
        _equals(
            row.get("threshold_milli"),
            expected_milli,
            name=f"001.thresholds[{expected_milli}].milli",
        )

    return {
        "development_replay": {
            "scored_exposure_samples": _integer(
                dev.get("scored_exposure_samples"),
                name="001.dev.scored_exposure_samples",
            ),
            "speaker_count": _integer(
                dev.get("speaker_count"), name="001.dev.speaker_count"
            ),
            "utterance_count": _integer(
                dev.get("utterance_count"), name="001.dev.utterance_count"
            ),
        },
        "disposition": "reject",
        "evidence_class": "measured_development_replay",
        "experiment": "001",
        "frontier_rows": {
            "negative_fail_previous": _threshold_payload(
                thresholds[990], expected_milli=990
            ),
            "negative_pass_minimum": _threshold_payload(
                thresholds[991], expected_milli=991
            ),
            "retention_fail_next": _threshold_payload(
                thresholds[396], expected_milli=396
            ),
            "retention_pass_maximum": _threshold_payload(
                thresholds[395], expected_milli=395
            ),
        },
        "nonclaims": [
            "No registered threshold satisfied both gates.",
            "No threshold was selected and no holdout result is represented.",
            "Reported development rates are not production false-accept estimates.",
        ],
        "selected_threshold_milli": None,
        "sources": ["experiment_001_replay", "experiment_001_selection"],
    }


def _validate_experiment_002(
    report: dict[str, object], incident: dict[str, object]
) -> dict[str, object]:
    _schema_version(report, name="002.report")
    _keys(
        report,
        {
            "artifacts",
            "checkpoint_reusable",
            "experiment",
            "failure",
            "publication",
            "registration",
            "schema_version",
            "status",
        },
        name="002.report",
    )
    _schema_version(incident, name="002.incident")
    _keys(
        incident,
        {
            "diagnosis",
            "experiment",
            "incident",
            "outcome",
            "registration",
            "remediation",
            "runtime_observation",
            "schema_version",
        },
        name="002.incident",
    )
    _equals(report.get("experiment"), "002", name="002.report.experiment")
    _equals(incident.get("experiment"), "002", name="002.incident.experiment")
    _equals(
        incident.get("incident"),
        "registered_execution_failure",
        name="002.incident.kind",
    )
    outcome = _mapping(incident.get("outcome"), name="002.outcome")
    diagnosis = _mapping(incident.get("diagnosis"), name="002.diagnosis")
    _equals(
        outcome.get("report_sha256"),
        _SOURCE_BY_ID["experiment_002_terminal_report"].sha256,
        name="002.outcome.report_sha256",
    )
    _equals(outcome.get("status"), "execution_failure", name="002.status")
    _equals(outcome.get("code"), "seed_selection_failed", name="002.code")
    _equals(outcome.get("phase"), "registered_execution", name="002.phase")
    if _boolean(outcome.get("checkpoint_reusable"), name="002.checkpoint_reusable"):
        _fail("schema:002.checkpoint_reusable:value")
    if _boolean(
        diagnosis.get("published_scientific_result_admitted"),
        name="002.scientific_result",
    ):
        _fail("schema:002.scientific_result:value")
    if _boolean(report.get("checkpoint_reusable"), name="002.report.checkpoint"):
        _fail("schema:002.report.checkpoint:value")
    _expect_empty_list(report.get("artifacts"), name="002.report.artifacts")
    return {
        "disposition": _text(outcome.get("status"), name="002.status"),
        "evidence_class": "registered_execution_incident",
        "experiment": "002",
        "outcome_code": _text(outcome.get("code"), name="002.code"),
        "outcome_phase": _text(outcome.get("phase"), name="002.phase"),
        "nonclaims": [
            "No published scientific result was admitted.",
            "No artifact or reusable checkpoint was published.",
        ],
        "sources": [
            "experiment_002_terminal_report",
            "experiment_002_incident",
        ],
    }


def _validate_experiment_003(incident: dict[str, object]) -> dict[str, object]:
    _schema_version(incident, name="003")
    _keys(
        incident,
        {
            "attempt",
            "diagnosis",
            "execution_protocol",
            "experiment",
            "incident",
            "managed_outputs",
            "outcome",
            "registration",
            "remediation",
            "runtime_observation",
            "schema_version",
            "scientific_protocol",
        },
        name="003",
    )
    _equals(incident.get("experiment"), "003", name="003.experiment")
    _equals(
        incident.get("incident"),
        "registered_admission_failure",
        name="003.incident",
    )
    attempt = _mapping(incident.get("attempt"), name="003.attempt")
    outcome = _mapping(incident.get("outcome"), name="003.outcome")
    diagnosis = _mapping(incident.get("diagnosis"), name="003.diagnosis")
    managed = _mapping(incident.get("managed_outputs"), name="003.managed")
    _equals(attempt.get("registered_runner_invocations"), 1, name="003.invocations")
    for field in (
        "registered_optimizer_updates",
        "registered_seed_children_started",
        "registered_validation_examples",
        "training_processes_started",
    ):
        _equals(attempt.get(field), 0, name=f"003.{field}")
    if not _boolean(attempt.get("attempt_consumed"), name="003.attempt_consumed"):
        _fail("schema:003.attempt_consumed:value")
    if _boolean(attempt.get("retry_allowed"), name="003.retry_allowed"):
        _fail("schema:003.retry_allowed:value")
    _equals(outcome.get("status"), "execution_failure", name="003.status")
    _equals(outcome.get("code"), "parent_route_signature_mismatch", name="003.code")
    if _boolean(
        diagnosis.get("published_scientific_result_admitted"),
        name="003.scientific_result",
    ):
        _fail("schema:003.scientific_result:value")
    if _boolean(outcome.get("checkpoint_reusable"), name="003.checkpoint"):
        _fail("schema:003.checkpoint:value")
    if _boolean(
        outcome.get("automatic_terminal_report_published"),
        name="003.automatic_report",
    ):
        _fail("schema:003.automatic_report:value")
    _expect_empty_list(outcome.get("artifacts"), name="003.outcome.artifacts")
    if _boolean(managed.get("checkpoint_reusable"), name="003.managed.checkpoint"):
        _fail("schema:003.managed.checkpoint:value")
    if _boolean(managed.get("final_report_present"), name="003.managed.report"):
        _fail("schema:003.managed.report:value")
    _expect_empty_list(managed.get("artifacts"), name="003.artifacts")
    return {
        "disposition": _text(outcome.get("status"), name="003.status"),
        "evidence_class": "registered_admission_incident",
        "experiment": "003",
        "outcome_code": _text(outcome.get("code"), name="003.code"),
        "outcome_phase": _text(outcome.get("phase"), name="003.phase"),
        "nonclaims": [
            "No training process, optimizer update, or validation example was started.",
            "No scientific result, artifact, or reusable checkpoint was published.",
            "The consumed attempt is not retryable.",
        ],
        "sources": ["experiment_003_incident"],
    }


def _validate_preflight(
    incident: dict[str, object], *, experiment: str
) -> dict[str, object]:
    _schema_version(incident, name=experiment)
    expected_keys = {
        "attempt",
        "experiment",
        "implementation",
        "incident",
        "kind",
        "managed_outputs_absent",
        "preflight_evidence",
        "protocol",
        "recovery_requirements",
        "schema_version",
        "status",
        "violations",
    }
    if experiment == "005":
        expected_keys.add("preflight_boundary")
    _keys(incident, expected_keys, name=experiment)
    _equals(incident.get("experiment"), experiment, name=f"{experiment}.experiment")
    _equals(
        incident.get("incident"),
        "pre_registration_protocol_rejection",
        name=f"{experiment}.incident",
    )
    _equals(
        incident.get("kind"),
        "execution_protocol_preflight_incident",
        name=f"{experiment}.kind",
    )
    _equals(incident.get("status"), "preflight_rejected", name=f"{experiment}.status")
    attempt = _mapping(incident.get("attempt"), name=f"{experiment}.attempt")
    for field in (
        "optimizer_updates",
        "registered_invocation_count",
        "validation_examples",
    ):
        _equals(attempt.get(field), 0, name=f"{experiment}.{field}")
    for field in (
        "canonical_marker_present",
        "registered_attempt_consumed",
        "registered_authority_issuer_invoked",
        "registered_coordinator_invoked",
        "registered_runner_invoked",
    ):
        if _boolean(attempt.get(field), name=f"{experiment}.{field}"):
            _fail(f"schema:{experiment}.{field}:value")
    violations = _list(incident.get("violations"), name=f"{experiment}.violations")
    if not violations:
        _fail(f"schema:{experiment}.violations:empty")
    codes = [
        _text(
            _mapping(row, name=f"{experiment}.violation").get("code"),
            name=f"{experiment}.violation.code",
        )
        for row in violations
    ]
    expected_codes = {
        "004": [
            "facade_normalization_recipe_cannot_express_profile_004",
            "shared_authority_recipe_omits_required_dispatch",
        ],
        "005": ["facade_normalization_recipe_cannot_express_truthful_profile_005"],
    }[experiment]
    if codes != expected_codes:
        _fail(f"schema:{experiment}.violation_codes:value")
    return {
        "disposition": "preflight_rejected",
        "evidence_class": "pre_registration_incident",
        "experiment": experiment,
        "nonclaims": [
            "No registration or registered invocation was created.",
            "The authority issuer, coordinator, runner, and attempt marker "
            "were not invoked.",
            "No optimizer update, validation example, or scientific result "
            "was produced.",
        ],
        "sources": [f"experiment_{experiment}_incident"],
        "violation_codes": codes,
    }


def _validate_experiment_006(incident: dict[str, object]) -> dict[str, object]:
    _schema_version(incident, name="006")
    _keys(
        incident,
        {
            "attempt",
            "diagnosis",
            "execution_protocol",
            "experiment",
            "incident",
            "kind",
            "managed_state_after_exit",
            "observer_activity",
            "outcome",
            "record_provenance",
            "registration",
            "runtime_observation",
            "schema_version",
            "scientific_protocol",
            "terminal_disposition",
            "timing",
        },
        name="006",
    )
    _equals(incident.get("experiment"), "006", name="006.experiment")
    _equals(
        incident.get("incident"),
        "registered_terminal_authority_failure",
        name="006.incident",
    )
    attempt = _mapping(incident.get("attempt"), name="006.attempt")
    diagnosis = _mapping(incident.get("diagnosis"), name="006.diagnosis")
    hypothesis = _mapping(
        diagnosis.get("leading_hypothesis"), name="006.leading_hypothesis"
    )
    outcome = _mapping(incident.get("outcome"), name="006.outcome")
    terminal = _mapping(
        incident.get("terminal_disposition"), name="006.terminal_disposition"
    )
    managed = _mapping(
        incident.get("managed_state_after_exit"), name="006.managed_state"
    )
    _equals(attempt.get("registered_invocation_count"), 1, name="006.invocations")
    if not _boolean(attempt.get("registered_attempt_consumed"), name="006.consumed"):
        _fail("schema:006.consumed:value")
    if _boolean(attempt.get("retry_allowed"), name="006.retry_allowed"):
        _fail("schema:006.retry_allowed:value")
    if not _boolean(attempt.get("terminal"), name="006.terminal"):
        _fail("schema:006.terminal:value")
    _equals(outcome.get("status"), "execution_failure", name="006.status")
    _equals(
        outcome.get("code"),
        "final_registration_capability_rejected_after_prior_state_change",
        name="006.code",
    )
    if _boolean(
        outcome.get("scientific_result_available"), name="006.scientific_result"
    ):
        _fail("schema:006.scientific_result:value")
    if _boolean(outcome.get("checkpoint_reusable"), name="006.checkpoint"):
        _fail("schema:006.checkpoint:value")
    if _boolean(
        outcome.get("automatic_terminal_report_published"),
        name="006.automatic_report",
    ):
        _fail("schema:006.automatic_report:value")
    _expect_empty_list(outcome.get("published_artifacts"), name="006.artifacts")
    _equals(
        diagnosis.get("root_cause_status"),
        "undetermined_from_retained_evidence",
        name="006.root_cause",
    )
    _equals(
        hypothesis.get("classification"),
        "plausible_observer_interference_not_proven",
        name="006.hypothesis",
    )
    if _boolean(
        hypothesis.get("reverification_overlap_established"),
        name="006.overlap_established",
    ):
        _fail("schema:006.overlap_established:value")
    for field in (
        "optimizer_updates",
        "seed_children_started",
        "training_processes_started",
        "validation_examples",
    ):
        _equals(
            managed.get(field),
            "not_established_from_retained_evidence",
            name=f"006.{field}",
        )
    if _boolean(
        terminal.get("experiment_006_retry_allowed"), name="006.terminal_retry"
    ):
        _fail("schema:006.terminal_retry:value")
    if _boolean(terminal.get("checkpoint_reuse_allowed"), name="006.checkpoint_reuse"):
        _fail("schema:006.checkpoint_reuse:value")
    return {
        "disposition": _text(outcome.get("status"), name="006.status"),
        "evidence_class": "registered_terminal_incident",
        "experiment": "006",
        "outcome_code": _text(outcome.get("code"), name="006.code"),
        "outcome_phase": _text(outcome.get("phase"), name="006.phase"),
        "root_cause_status": _text(
            diagnosis.get("root_cause_status"), name="006.root_cause"
        ),
        "nonclaims": [
            "No scientific result, automatic terminal report, artifact, or "
            "reusable checkpoint exists.",
            "Hidden execution progress is not established from retained evidence.",
            "Observer interference is plausible but not proven as the root cause.",
            "The consumed terminal attempt is not retryable.",
        ],
        "sources": ["experiment_006_incident"],
    }


def _source_record_map(value: object, *, name: str) -> dict[str, str]:
    records = _list(value, name=name)
    result: dict[str, str] = {}
    for index, raw_record in enumerate(records):
        record = _mapping(raw_record, name=f"{name}[{index}]")
        _keys(record, {"path", "sha256"}, name=f"{name}[{index}]")
        path = _text(record.get("path"), name=f"{name}[{index}].path")
        digest = _text(record.get("sha256"), name=f"{name}[{index}].sha256")
        if path in result:
            _fail(f"schema:{name}:duplicate_path")
        result[path] = digest
    return result


def _validate_visual_provenance(provenance: dict[str, object]) -> None:
    _schema_version(provenance, name="provenance")
    _keys(
        provenance,
        {
            "existing_report_assets",
            "renderer",
            "schema_version",
            "source_allowlist",
            "visuals",
        },
        name="provenance",
    )
    visuals = _mapping(provenance.get("visuals"), name="provenance.visuals")
    lineage = _mapping(visuals.get("experiment-lineage.svg"), name="provenance.lineage")
    threshold = _mapping(
        visuals.get("threshold-tradeoff.svg"), name="provenance.threshold"
    )
    data = _mapping(
        visuals.get("data-boundaries.svg"), name="provenance.data_boundaries"
    )

    lineage_sources = _source_record_map(
        lineage.get("source_records"), name="provenance.lineage.sources"
    )
    expected_lineage = {
        _SOURCE_BY_ID[source_id].relative_path: _SOURCE_BY_ID[source_id].sha256
        for source_id in (
            "experiment_000_report",
            "experiment_001_replay",
            "experiment_002_incident",
            "experiment_003_incident",
            "experiment_004_incident",
            "experiment_005_incident",
            "experiment_006_incident",
        )
    }
    if lineage_sources != expected_lineage:
        _fail("schema:provenance.lineage.sources:value")
    threshold_sources = _source_record_map(
        threshold.get("source_records"), name="provenance.threshold.sources"
    )
    if threshold_sources != {
        _SOURCE_BY_ID["experiment_001_replay"].relative_path: _SOURCE_BY_ID[
            "experiment_001_replay"
        ].sha256
    }:
        _fail("schema:provenance.threshold.sources:value")
    data_sources = _source_record_map(
        data.get("source_records"), name="provenance.data.sources"
    )
    expected_data = {
        _SOURCE_BY_ID[source_id].relative_path: _SOURCE_BY_ID[source_id].sha256
        for source_id in ("experiment_000_report", "experiment_001_replay")
    }
    if data_sources != expected_data:
        _fail("schema:provenance.data.sources:value")


def build_inspection() -> dict[str, object]:
    """Return the validated, deterministic inspection payload."""

    root_fd = _open_root()
    try:
        documents = _load_sources(root_fd)
    finally:
        os.close(root_fd)
    _validate_visual_provenance(documents["visual_provenance"])
    experiments = [
        _validate_experiment_000(documents["experiment_000_report"]),
        _validate_experiment_001(
            documents["experiment_001_replay"],
            documents["experiment_001_selection"],
        ),
        _validate_experiment_002(
            documents["experiment_002_terminal_report"],
            documents["experiment_002_incident"],
        ),
        _validate_experiment_003(documents["experiment_003_incident"]),
        _validate_preflight(documents["experiment_004_incident"], experiment="004"),
        _validate_preflight(documents["experiment_005_incident"], experiment="005"),
        _validate_experiment_006(documents["experiment_006_incident"]),
    ]
    return {
        "experiments": experiments,
        "inspection": {
            "experiment_code_imported": False,
            "mode": "read_only_hash_pinned",
            "scientific_metrics_recomputed": False,
            "source_count": len(SOURCE_SPECS),
        },
        "schema_version": 1,
        "sources": [
            {
                "path": spec.relative_path,
                "role": spec.role,
                "sha256": spec.sha256,
                "source_id": spec.source_id,
            }
            for spec in SOURCE_SPECS
        ],
    }


def _as_mapping(value: object) -> dict[str, object]:
    return _mapping(value, name="render")


def _as_float(value: object) -> float:
    return _number(value, name="render.number")


def _as_int(value: object) -> int:
    return _integer(value, name="render.integer")


def _render_split(name: str, value: object) -> str:
    split = _as_mapping(value)
    target = _as_mapping(split["target_argmax_error"])
    open_set = _as_mapping(split["open_set_target_prediction"])
    unknown = _as_mapping(open_set["unknown"])
    silence = _as_mapping(open_set["silence"])
    return (
        f"  {name}: n={_as_int(split['example_count']):,}; "
        f"accuracy={_as_float(split['accuracy']):.2%}; "
        f"macro-F1={_as_float(split['macro_f1']):.3f}; "
        f"target argmax error={_as_float(target['error_rate']):.2%}; "
        f"unknown->target={_as_float(unknown['predicted_target_rate']):.2%}; "
        f"silence->target={_as_float(silence['predicted_target_rate']):.2%}"
    )


def _render_frontier(label: str, value: object) -> str:
    row = _as_mapping(value)
    return (
        f"  {label}: threshold={_as_int(row['threshold_milli']) / 1000:.3f}; "
        f"retention={_as_float(row['conditional_correct_retention']):.6f}; "
        f"events={_as_int(row['dev_event_count']):,}; "
        f"reported FEH={_as_float(row['false_events_per_hour']):.6f}"
    )


def render_human(payload: Mapping[str, object]) -> bytes:
    """Render a stable, path-relative human inspection."""

    experiments = _list(payload.get("experiments"), name="render.experiments")
    by_id = {
        _text(_as_mapping(item).get("experiment"), name="render.experiment"): (
            _as_mapping(item)
        )
        for item in experiments
    }
    exp000 = by_id["000"]
    metrics000 = _as_mapping(exp000["metrics"])
    exp001 = by_id["001"]
    replay001 = _as_mapping(exp001["development_replay"])
    frontiers001 = _as_mapping(exp001["frontier_rows"])
    lines = [
        "FalseWake tracked evidence inspection",
        "mode: read-only, hash-pinned, no experiment-code imports",
        f"sources: {len(SOURCE_SPECS)} fixed JSON records",
        "",
        "000 | MEASURED CLIP BASELINE",
        _render_split("validation", metrics000["validation"]),
        _render_split("test", metrics000["test"]),
        "  limits: thresholds/streaming not evaluated; validation diagnostic only",
        "",
        "001 | MEASURED DEVELOPMENT REPLAY | REJECT",
        (
            "  corpus: "
            f"{_as_int(replay001['utterance_count']):,} utterances; "
            f"{_as_int(replay001['speaker_count'])} speakers; "
            f"{_as_int(replay001['scored_exposure_samples']):,} scored samples"
        ),
        _render_frontier(
            "retention pass maximum", frontiers001["retention_pass_maximum"]
        ),
        _render_frontier("retention fail next", frontiers001["retention_fail_next"]),
        _render_frontier(
            "negative fail previous", frontiers001["negative_fail_previous"]
        ),
        _render_frontier(
            "negative pass minimum", frontiers001["negative_pass_minimum"]
        ),
        "  limits: no selected threshold; no holdout or production-rate claim",
        "",
        "002-006 | DISPOSITIONS AND NON-CLAIMS",
    ]
    for experiment_id in ("002", "003", "004", "005", "006"):
        experiment = by_id[experiment_id]
        raw_detail = experiment.get("outcome_code")
        if raw_detail is None:
            violations = _list(
                experiment.get("violation_codes"),
                name=f"render.{experiment_id}.violations",
            )
            if not violations:
                _fail(f"schema:render.{experiment_id}.violations:empty")
            raw_detail = violations[0]
        detail = _text(raw_detail, name=f"render.{experiment_id}.detail")
        lines.append(
            f"  {experiment_id}: "
            f"{_text(experiment['disposition'], name='render.disposition')} | {detail}"
        )
        for raw_nonclaim in _list(
            experiment["nonclaims"], name=f"render.{experiment_id}.nonclaims"
        ):
            lines.append(f"    - {_text(raw_nonclaim, name='render.nonclaim')}")
    lines.extend(("", "PROVENANCE (repository-relative; SHA-256)"))
    for source_id in _REPORT_SOURCE_IDS:
        spec = _SOURCE_BY_ID[source_id]
        lines.append(f"  {spec.relative_path}  {spec.sha256}")
    visual = _SOURCE_BY_ID["visual_provenance"]
    lines.append(f"  {visual.relative_path}  {visual.sha256}")
    return ("\n".join(lines) + "\n").encode("ascii")


def render_json(payload: Mapping[str, object]) -> bytes:
    """Render canonical, one-line ASCII JSON."""

    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        + b"\n"
    )


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        del message
        _fail("cli:arguments")


def _parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = _Parser(
        prog="inspect_evidence.py",
        description="Inspect hash-pinned tracked FalseWake evidence.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit canonical machine-readable JSON",
    )
    return parser.parse_args(argv)


def _write_all(descriptor: int, contents: bytes) -> None:
    offset = 0
    while offset < len(contents):
        written = os.write(descriptor, contents[offset:])
        if written <= 0:
            _fail("output:write")
        offset += written


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fixed inspection command."""

    try:
        arguments = _parse_arguments(sys.argv[1:] if argv is None else argv)
        payload = build_inspection()
        output = render_json(payload) if arguments.json else render_human(payload)
        _write_all(sys.stdout.fileno(), output)
    except (EvidenceInspectionError, OSError) as error:
        code = (
            str(error)
            if isinstance(error, EvidenceInspectionError)
            else "operating_system"
        )
        with suppress(EvidenceInspectionError, OSError):
            _write_all(
                sys.stderr.fileno(),
                f"evidence inspection failed: {code}\n".encode("ascii"),
            )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
