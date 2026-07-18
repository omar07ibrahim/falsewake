"""Fail-closed authorization for experiment 001 holdout access.

The firewall does not invoke the holdout supplier or expose a writable stream until
every development identity, the independently recomputed selection, and the clean
implementation snapshot have passed validation. Acquired bytes live only in an
anonymous, irreversibly sealed descriptor.
"""

from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import math
import os
import platform
import re
import shutil
import stat
import struct
import subprocess
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Self, cast

import numpy as np
import scipy
import soundfile as sf  # type: ignore[import-untyped]

MODULE_SOURCE_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = MODULE_SOURCE_PATH.parents[2]
EXECUTING_HOLDOUT_SHA256 = hashlib.sha256(MODULE_SOURCE_PATH.read_bytes()).hexdigest()

SELECTION_SCHEMA_VERSION = 1
EXPERIMENT_CONFIG_SHA256 = (
    "7e27ba554ef357a2de4a929088a99efc20b9ee0285cb0c242a05474cc944b217"
)
THRESHOLD_COUNT = 1_001
EXPOSURE_SAMPLES_PER_HOUR = 57_600_000
TARGET_EXAMPLE_COUNT = 3_703
BASELINE_CORRECT_COUNT = 2_088
TARGET_ORDER = (
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
)
TARGET_SUPPORT = (397, 406, 350, 377, 352, 363, 363, 373, 350, 372)
BASELINE_CORRECT_BY_TARGET = (289, 178, 193, 195, 185, 172, 231, 220, 222, 203)
DEV_MANIFEST_SHA256 = "6494fa36866b0c90eb12e8e1325339981fcd3cb5c61abd2d06dedd5d3ce6cff7"
DEV_AUDIT_REPORT_SHA256 = (
    "810bbd4966d3ad5a24cd2bdd3bb1a8afb4b7325fc285e19f180fb7b26a9dbe74"
)
DEV_SOURCE_SAMPLES = 310_337_932
DEV_SCORED_EXPOSURE_SAMPLES = 308_310_400
DEV_UTTERANCE_COUNT = 2_703
DEV_SPEAKER_COUNT = 40
TEST_ARCHIVE_BYTES = 346_663_984
TEST_ARCHIVE_MD5 = "32fa31d27d2e1cad72775fee3f4849a9"
TEST_ARCHIVE_SHA256 = (
    "39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23"
)

MAX_CONFIG_BYTES = 2 * 1024 * 1024
MAX_ARTIFACT_BYTES = 1024 * 1024
MAX_AUDIT_REPORT_BYTES = 2 * 1024 * 1024
MAX_REPLAY_REPORT_BYTES = 64 * 1024 * 1024
MAX_DEV_MANIFEST_BYTES = 32 * 1024 * 1024
MAX_DEV_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_SOURCE_BYTES = 4 * 1024 * 1024
MAX_TEST_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_GIT_OBJECT_SNAPSHOT_BYTES = 128 * 1024 * 1024
MAX_GIT_CONTROL_FILE_BYTES = 4 * 1024 * 1024
MAX_GIT_INDEX_BYTES = 64 * 1024 * 1024

EXPERIMENT_CONFIG_RELATIVE_PATH = "configs/experiment-001.json"
PORTABLE_MODEL_RELATIVE_PATH = "models/experiment-000-linear.json"
SELECTION_ARTIFACT_RELATIVE_PATH = "reports/experiment-001-selection.json"
ACQUISITION_CONTRACT = (
    "after_final_artifact_HEAD_config_index_scorer_model_runtime_and_development_"
    "evidence_revalidation_create_one_anonymous_memfd_with_sealing_enabled;invoke_"
    "one_trusted_supplier_once_with_only_a_writable_binary_stream;fsync_then_apply_"
    "F_SEAL_WRITE_F_SEAL_GROW_F_SEAL_SHRINK_F_SEAL_SEAL;require_exact_registered_"
    "test_clean_size_MD5_and_SHA256;run_no_Git_after_supplier;return_the_same_"
    "sealed_inode_through_an_independent_read_only_open_file_description"
)
GIT_VALIDATION_CONTRACT = (
    "copy_only_physical_bounded_object_files_into_a_new_bare_snapshot;ignore_"
    "repository_local_config_info_alternates_replacements_and_lazy_fetch;use_only_"
    "non_worktree_plumbing_with_full_commit_IDs;capture_one_physical_bounded_index_"
    "and_require_every_stage_zero_mode_OID_path_entry_to_equal_the_recursive_"
    "artifact_HEAD_tree;reject_unmerged_or_sparse_entries;never_run_status_hooks_"
    "filters_or_other_worktree_commands"
)

SCORER_SOURCE_FILES = (
    "src/falsewake/baseline_data.py",
    "src/falsewake/continuous_replay.py",
    "src/falsewake/feature_matrix.py",
    "src/falsewake/features.py",
    "src/falsewake/holdout.py",
    "src/falsewake/librispeech.py",
    "src/falsewake/speech_commands.py",
)
RUNTIME_FIELDS = (
    "python",
    "numpy",
    "scipy",
    "soundfile",
    "libsndfile",
    "platform",
)
SELECTION_ARTIFACT_FIELDS = {
    "schema_version",
    "status",
    "experiment_config_sha256",
    "scorer_source_sha256",
    "runtime_identity_sha256",
    "implementation_git_commit",
    "dev_archive_sha256",
    "dev_manifest_sha256",
    "dev_audit_report_sha256",
    "dev_replay_report_sha256",
    "selected_threshold_milli",
}
REPLAY_TOP_LEVEL_FIELDS = (
    "schema_version",
    "identities",
    "runtime",
    "target_order",
    "dev",
    "positive_validation",
    "thresholds",
    "selection",
    "top_false_events",
)
REPLAY_IDENTITY_FIELDS = (
    "experiment_config_sha256",
    "portable_model_sha256",
    "scorer_source_sha256",
    "runtime_identity_sha256",
    "implementation_git_commit",
    "dev_archive_sha256",
    "dev_manifest_sha256",
    "dev_audit_report_sha256",
    "positive_examples_sha256",
    "positive_feature_matrix_semantic_sha256",
)
REPLAY_THRESHOLD_FIELDS = (
    "threshold_milli",
    "dev_event_count",
    "dev_event_count_by_target",
    "false_events_per_hour",
    "garwood_95_percent",
    "speaker_bootstrap_95_percent",
    "validation_correct_accept_count",
    "validation_correct_accept_count_by_target",
    "correct_accept_recall",
    "conditional_correct_retention",
)

_SOURCE_DOMAIN = b"falsewake-exp001-scorer-source-v1\0"
_SNAPSHOT_ATTRIBUTES = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_nlink",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)


class HoldoutAccessError(ValueError):
    """Experiment 001 evidence does not authorize holdout access."""


@dataclass(frozen=True, slots=True)
class HoldoutEvidence:
    """All development-only inputs needed by the holdout firewall."""

    dev_archive: Path
    dev_manifest: Path
    dev_audit_report: Path
    dev_replay_report: Path


@dataclass(slots=True)
class AuthorizedTestClean:
    """A sealed official holdout stream and its captured evaluation identity."""

    stream: BinaryIO
    selected_threshold_milli: int
    implementation_git_commit: str
    portable_model_json: bytes
    archive_sha256: str

    def close(self) -> None:
        """Close the authorized stream."""

        self.stream.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object,
    ) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class _StableFile:
    sha256: str
    contents: bytes | None
    identity: tuple[int, int]


@dataclass(frozen=True, slots=True)
class _GitState:
    scorer_source_sha256: str
    files: tuple[_StableFile, ...]


@dataclass(frozen=True, slots=True)
class _SelectionArtifactState:
    file: _StableFile
    artifact: dict[str, object]
    head_commit: str


@dataclass(frozen=True, slots=True)
class _GitSnapshot:
    repository_root: Path
    git_dir: Path
    head_commit: str
    index_file: _StableFile


TestCleanSupplier = Callable[[BinaryIO], None]


@dataclass(frozen=True, slots=True)
class _Authorization:
    threshold_milli: int
    implementation_git_commit: str
    portable_model_json: bytes


@dataclass(frozen=True, slots=True)
class _Selection:
    status: str
    threshold_milli: int | None


@dataclass(frozen=True, slots=True)
class _ReplayReport:
    selection: _Selection
    identities: dict[str, object]
    runtime: dict[str, str]
    source_samples: int
    scored_exposure_samples: int
    utterance_count: int
    speaker_count: int


@dataclass(frozen=True, slots=True)
class _AuditPopulation:
    source_samples: int
    utterance_count: int
    speaker_count: int


@dataclass(frozen=True, slots=True)
class _ManifestPopulation:
    source_samples: int
    scored_exposure_samples: int
    utterance_count: int
    speaker_count: int


def _reject_json_constant(value: str) -> object:
    raise HoldoutAccessError(f"JSON contains invalid numeric constant: {value}")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise HoldoutAccessError(f"JSON repeats key: {key!r}")
        result[key] = value
    return result


def _parse_json(contents: bytes, *, name: str) -> object:
    try:
        document = json.loads(
            contents,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise HoldoutAccessError(f"cannot decode {name}: {error}") from error
    _reject_nonfinite_numbers(document, name=name)
    return document


def _reject_nonfinite_numbers(value: object, *, name: str) -> None:
    if type(value) is float and not math.isfinite(value):
        raise HoldoutAccessError(f"{name} contains a non-finite JSON number")
    if type(value) is list:
        for item in cast(list[object], value):
            _reject_nonfinite_numbers(item, name=name)
    elif type(value) is dict:
        for item in cast(dict[str, object], value).values():
            _reject_nonfinite_numbers(item, name=name)


def _require_object(
    value: object,
    *,
    name: str,
    exact_keys: set[str] | None = None,
) -> dict[str, object]:
    if type(value) is not dict:
        raise HoldoutAccessError(f"{name} must be a JSON object")
    result = cast(dict[str, object], value)
    if exact_keys is not None and set(result) != exact_keys:
        raise HoldoutAccessError(f"{name} has unexpected or missing fields")
    return result


def _require_int(
    value: object,
    *,
    name: str,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise HoldoutAccessError(f"{name} must be a JSON integer")
    result = value
    if result < minimum or (maximum is not None and result > maximum):
        raise HoldoutAccessError(f"{name} is outside its registered bounds")
    return result


def _require_number(value: object, *, name: str, minimum: float | None = None) -> float:
    if type(value) not in {int, float}:
        raise HoldoutAccessError(f"{name} must be a finite JSON number")
    result = float(cast(int | float, value))
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise HoldoutAccessError(f"{name} is outside its registered bounds")
    return result


def _require_int_vector(
    value: object,
    *,
    name: str,
    length: int,
    minimum: int = 0,
) -> tuple[int, ...]:
    if type(value) is not list or len(cast(list[object], value)) != length:
        raise HoldoutAccessError(f"{name} must contain exactly {length} integers")
    return tuple(
        _require_int(item, name=f"{name}[{index}]", minimum=minimum)
        for index, item in enumerate(cast(list[object], value))
    )


def _require_interval(value: object, *, name: str) -> tuple[float, float]:
    if type(value) is not list or len(cast(list[object], value)) != 2:
        raise HoldoutAccessError(f"{name} must contain two finite numbers")
    lower = _require_number(
        cast(list[object], value)[0], name=f"{name} lower", minimum=0
    )
    upper = _require_number(
        cast(list[object], value)[1], name=f"{name} upper", minimum=0
    )
    if lower > upper:
        raise HoldoutAccessError(f"{name} bounds are reversed")
    return lower, upper


def _require_sha256(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise HoldoutAccessError(f"{name} must be a SHA-256 string")
    digest = value
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise HoldoutAccessError(f"{name} is not a lowercase SHA-256 digest")
    return digest


def _require_git_commit(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise HoldoutAccessError(f"{name} must be a Git commit ID")
    commit = value
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise HoldoutAccessError(f"{name} is not a lowercase full Git commit ID")
    return commit


def _snapshot(metadata: os.stat_result) -> tuple[int, ...]:
    return tuple(cast(int, getattr(metadata, field)) for field in _SNAPSHOT_ATTRIBUTES)


def _resolve_existing_path(path: Path, *, name: str) -> Path:
    try:
        return path.resolve(strict=True)
    except OSError as error:
        raise HoldoutAccessError(f"cannot resolve {name}: {error}") from error


def _hash_stream(stream: BinaryIO, size: int) -> tuple[str, bytes | None]:
    digest = hashlib.sha256()
    captured: list[bytes] | None = [] if size <= MAX_REPLAY_REPORT_BYTES else None
    remaining = size
    while remaining:
        chunk = stream.read(min(1024 * 1024, remaining))
        if not chunk:
            raise HoldoutAccessError("bounded file ended before its recorded size")
        digest.update(chunk)
        if captured is not None:
            captured.append(chunk)
        remaining -= len(chunk)
    if stream.read(1):
        raise HoldoutAccessError("bounded file exceeds its recorded size")
    contents = b"".join(captured) if captured is not None else None
    return digest.hexdigest(), contents


def _read_stable_file(
    path: Path,
    *,
    name: str,
    maximum_bytes: int,
    keep_contents: bool,
) -> _StableFile:
    if not hasattr(os, "O_NOFOLLOW"):
        raise HoldoutAccessError("this firewall requires O_NOFOLLOW support")
    absolute_path = Path(os.path.abspath(path))
    if _resolve_existing_path(path, name=name) != absolute_path:
        raise HoldoutAccessError(f"{name} path must be fully physical")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            before = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink < 1
                or before.st_size < 0
                or before.st_size > maximum_bytes
            ):
                raise HoldoutAccessError(
                    f"{name} must be a bounded regular non-symlink file"
                )
            initial_snapshot = _snapshot(before)
            first_sha256, first_contents = _hash_stream(stream, before.st_size)
            stream.seek(0)
            second_sha256, second_contents = _hash_stream(stream, before.st_size)
            after = os.fstat(stream.fileno())
    except OSError as error:
        raise HoldoutAccessError(f"cannot read {name}: {error}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if (
        initial_snapshot != _snapshot(after)
        or not hmac.compare_digest(first_sha256, second_sha256)
        or first_contents != second_contents
    ):
        raise HoldoutAccessError(f"{name} changed while being read")
    if keep_contents and first_contents is None:
        raise HoldoutAccessError(f"{name} exceeds the in-memory parsing limit")
    return _StableFile(
        sha256=first_sha256,
        contents=first_contents if keep_contents else None,
        identity=(before.st_dev, before.st_ino),
    )


def _require_file_unchanged(
    path: Path,
    *,
    expected: _StableFile,
    name: str,
    maximum_bytes: int,
) -> None:
    observed = _read_stable_file(
        path,
        name=name,
        maximum_bytes=maximum_bytes,
        keep_contents=False,
    )
    if observed.identity != expected.identity or not hmac.compare_digest(
        observed.sha256, expected.sha256
    ):
        raise HoldoutAccessError(f"{name} changed during authorization")


def _mapping_at(
    document: dict[str, object], key: str, *, name: str
) -> dict[str, object]:
    if key not in document:
        raise HoldoutAccessError(f"{name} is missing")
    return _require_object(document[key], name=name)


def _validate_registered_config(contents: bytes) -> dict[str, object]:
    observed_sha256 = hashlib.sha256(contents).hexdigest()
    if not hmac.compare_digest(observed_sha256, EXPERIMENT_CONFIG_SHA256):
        raise HoldoutAccessError("experiment config identity is not experiment 001")
    config = _require_object(
        _parse_json(contents, name="experiment config"), name="experiment config"
    )
    if config.get("schema_version") != 1:
        raise HoldoutAccessError("experiment config schema_version differs")

    grid = _mapping_at(
        _mapping_at(config, "scoring", name="scoring"),
        "threshold_grid",
        name="threshold grid",
    )
    if grid != {
        "count": THRESHOLD_COUNT,
        "definition": "float64(integer_milli)_divided_by_float64(1000)",
        "start_milli": 0,
        "step_milli": 1,
        "stop_milli_inclusive": 1_000,
    }:
        raise HoldoutAccessError("threshold grid differs from experiment 001")

    selection = _mapping_at(config, "selection", name="selection")
    if selection.get("exact_integer_gates") != (
        "false_event_gate_is_events_times_57600000_lte_scored_exposure_samples;"
        "retention_gate_is_correct_accept_count_times_5_gte_baseline_correct_"
        "count_times_4"
    ):
        raise HoldoutAccessError("selection integer gates differ")
    if (
        selection.get("if_multiple_pass") != "lowest_threshold"
        or selection.get("max_dev_false_events_per_hour") != 1.0
        or selection.get("min_validation_conditional_correct_retention") != 0.8
    ):
        raise HoldoutAccessError("selection policy differs")

    identity = _mapping_at(
        config, "implementation_identity", name="implementation identity"
    )
    if identity.get("scorer_source_files") != list(SCORER_SOURCE_FILES):
        raise HoldoutAccessError("registered scorer source files differ")
    if identity.get("runtime_versions") != list(RUNTIME_FIELDS):
        raise HoldoutAccessError("registered runtime fields differ")

    negative = _mapping_at(config, "negative_source", name="negative source")
    test_identity = _mapping_at(
        negative,
        "test_clean_official_identity",
        name="official test-clean identity",
    )
    if test_identity != {
        "archive_bytes": TEST_ARCHIVE_BYTES,
        "archive_md5": TEST_ARCHIVE_MD5,
        "archive_sha256": TEST_ARCHIVE_SHA256,
        "archive_url": "https://www.openslr.org/resources/12/test-clean.tar.gz",
        "dataset": "LibriSpeech test-clean",
        "identity_provenance": (
            "OpenSLR_12_md5sum_and_openslr_librispeech_asr_download_checksum_"
            "metadata_registered_before_archive_access"
        ),
    }:
        raise HoldoutAccessError("official test-clean identity differs")
    archive_audit = _mapping_at(
        negative, "archive_audit", name="archive audit contract"
    )
    verified_output = _mapping_at(
        archive_audit,
        "verified_dev_clean_output",
        name="verified development output",
    )
    if verified_output != {
        "audit_report_sha256": DEV_AUDIT_REPORT_SHA256,
        "manifest_sha256": DEV_MANIFEST_SHA256,
        "scored_exposure_samples": DEV_SCORED_EXPOSURE_SAMPLES,
        "source_samples": DEV_SOURCE_SAMPLES,
        "speaker_count": DEV_SPEAKER_COUNT,
        "utterance_count": DEV_UTTERANCE_COUNT,
    }:
        raise HoldoutAccessError("verified development identities differ")

    positive = _mapping_at(config, "positive_validation", name="positive validation")
    if (
        positive.get("target_example_count") != TARGET_EXAMPLE_COUNT
        or positive.get("baseline_correct_count") != BASELINE_CORRECT_COUNT
        or positive.get("target_order") != list(TARGET_ORDER)
        or positive.get("target_support") != list(TARGET_SUPPORT)
        or positive.get("baseline_correct_by_target")
        != list(BASELINE_CORRECT_BY_TARGET)
    ):
        raise HoldoutAccessError("registered positive-validation denominators differ")

    replay = _mapping_at(config, "replay_report", name="replay report contract")
    if (
        replay.get("schema_version") != 1
        or replay.get("top_level_fields")
        != [
            "schema_version",
            "identities",
            "runtime",
            "target_order",
            "dev",
            "positive_validation",
            "thresholds",
            "selection",
            "top_false_events",
        ]
        or replay.get("identity_fields") != list(REPLAY_IDENTITY_FIELDS)
        or replay.get("dev_fields")
        != [
            "source_samples",
            "scored_exposure_samples",
            "utterance_count",
            "speaker_count",
            "speaker_rows",
        ]
        or replay.get("positive_validation_fields")
        != [
            "target_example_count",
            "baseline_correct_count",
            "baseline_correct_by_target",
            "support_by_target",
        ]
        or replay.get("threshold_row_fields") != list(REPLAY_THRESHOLD_FIELDS)
        or replay.get("selection_fields")
        != [
            "status",
            "selected_threshold_milli",
            "negative_frontier_milli",
            "retention_frontier_milli",
        ]
        or replay.get("top_false_events_fields") != ["threshold_milli", "events"]
        or replay.get("top_false_event_fields")
        != [
            "utterance_id",
            "window_start_sample",
            "predicted_target",
            "target_probability",
            "transcript",
        ]
    ):
        raise HoldoutAccessError("replay report schema differs from experiment 001")

    artifact = _mapping_at(config, "selection_artifact", name="selection artifact")
    if (
        artifact.get("schema_version") != SELECTION_SCHEMA_VERSION
        or artifact.get("enforcement_status") != "implemented_fail_closed"
        or artifact.get("acquisition_contract") != ACQUISITION_CONTRACT
        or artifact.get("git_validation_contract") != GIT_VALIDATION_CONTRACT
        or artifact.get("trusted_git_path") != SELECTION_ARTIFACT_RELATIVE_PATH
        or artifact.get("status_for_test_access") != "pass"
        or artifact.get("status_values") != ["pass", "reject"]
        or artifact.get("required_fields")
        != [
            "schema_version",
            "status",
            "experiment_config_sha256",
            "scorer_source_sha256",
            "runtime_identity_sha256",
            "implementation_git_commit",
            "dev_archive_sha256",
            "dev_manifest_sha256",
            "dev_audit_report_sha256",
            "dev_replay_report_sha256",
            "selected_threshold_milli",
        ]
    ):
        raise HoldoutAccessError("selection artifact contract differs")
    return config


def _parse_selection_artifact(contents: bytes) -> dict[str, object]:
    raw_document = _parse_json(contents, name="selection artifact")
    artifact = _require_object(
        raw_document,
        name="selection artifact",
        exact_keys=SELECTION_ARTIFACT_FIELDS,
    )
    try:
        canonical = (
            json.dumps(
                raw_document,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
            + b"\n"
        )
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise HoldoutAccessError("selection artifact is not canonical JSON") from error
    if canonical != contents:
        raise HoldoutAccessError("selection artifact serialization differs")
    if (
        artifact["schema_version"] != SELECTION_SCHEMA_VERSION
        or type(artifact["schema_version"]) is not int
    ):
        raise HoldoutAccessError(
            "selection artifact schema_version must be integer one"
        )
    if type(artifact["status"]) is not str or artifact["status"] not in {
        "pass",
        "reject",
    }:
        raise HoldoutAccessError("selection artifact status is invalid")
    for field in (
        "experiment_config_sha256",
        "scorer_source_sha256",
        "runtime_identity_sha256",
        "dev_archive_sha256",
        "dev_manifest_sha256",
        "dev_audit_report_sha256",
        "dev_replay_report_sha256",
    ):
        _require_sha256(artifact[field], name=field)
    _require_git_commit(
        artifact["implementation_git_commit"], name="implementation_git_commit"
    )
    threshold = artifact["selected_threshold_milli"]
    if artifact["status"] == "pass":
        _require_int(
            threshold,
            name="selected_threshold_milli",
            minimum=0,
            maximum=1_000,
        )
    elif threshold is not None:
        raise HoldoutAccessError("a rejected selection must have a null threshold")
    return artifact


def _validate_report_runtime(value: object) -> dict[str, str]:
    runtime = _require_object(
        value,
        name="development replay runtime",
        exact_keys=set(RUNTIME_FIELDS),
    )
    if any(
        type(runtime[field]) is not str or not runtime[field]
        for field in RUNTIME_FIELDS
    ):
        raise HoldoutAccessError("development replay runtime values must be strings")
    return {field: cast(str, runtime[field]) for field in RUNTIME_FIELDS}


def _validate_report_identities(value: object) -> dict[str, object]:
    identities = _require_object(
        value,
        name="development replay identities",
        exact_keys=set(REPLAY_IDENTITY_FIELDS),
    )
    for field in REPLAY_IDENTITY_FIELDS:
        if field == "implementation_git_commit":
            _require_git_commit(identities[field], name=field)
        else:
            _require_sha256(identities[field], name=field)
    return identities


def _validate_top_false_events(
    value: object,
    *,
    expected_thresholds: tuple[int, ...],
) -> None:
    if type(value) is not list:
        raise HoldoutAccessError("top_false_events must be a JSON array")
    groups = cast(list[object], value)
    if len(groups) != len(expected_thresholds):
        raise HoldoutAccessError("top_false_events has the wrong decision frontiers")
    for group_index, (raw_group, expected_threshold) in enumerate(
        zip(groups, expected_thresholds, strict=True)
    ):
        group = _require_object(
            raw_group,
            name=f"top false-event group {group_index}",
            exact_keys={"threshold_milli", "events"},
        )
        threshold = _require_int(
            group["threshold_milli"],
            name=f"top false-event group {group_index} threshold_milli",
            minimum=0,
            maximum=1_000,
        )
        if threshold != expected_threshold:
            raise HoldoutAccessError("top false-event thresholds are not canonical")
        if (
            type(group["events"]) is not list
            or len(cast(list[object], group["events"])) > 50
        ):
            raise HoldoutAccessError(
                "top false-event group must contain at most 50 events"
            )
        previous_rank: tuple[float, str, int] | None = None
        for event_index, raw_event in enumerate(cast(list[object], group["events"])):
            event = _require_object(
                raw_event,
                name=f"top false event {group_index}:{event_index}",
                exact_keys={
                    "utterance_id",
                    "window_start_sample",
                    "predicted_target",
                    "target_probability",
                    "transcript",
                },
            )
            if type(event["utterance_id"]) is not str or not event["utterance_id"]:
                raise HoldoutAccessError("top false-event utterance_id is invalid")
            start = _require_int(
                event["window_start_sample"],
                name="top false-event window_start_sample",
                minimum=0,
            )
            if (
                event["predicted_target"] not in TARGET_ORDER
                or type(event["predicted_target"]) is not str
            ):
                raise HoldoutAccessError("top false-event target is invalid")
            probability = _require_number(
                event["target_probability"],
                name="top false-event probability",
                minimum=0,
            )
            if probability > 1 or probability < threshold / 1_000:
                raise HoldoutAccessError("top false-event probability is invalid")
            if type(event["transcript"]) is not str:
                raise HoldoutAccessError("top false-event transcript must be a string")
            rank = (-probability, event["utterance_id"], start)
            if previous_rank is not None and rank < previous_rank:
                raise HoldoutAccessError(
                    "top false events are not in registered rank order"
                )
            previous_rank = rank


def _parse_replay_report(replay_report: bytes) -> _ReplayReport:
    raw_document = _parse_json(replay_report, name="development replay report")
    report = _require_object(
        raw_document,
        name="development replay report",
        exact_keys=set(REPLAY_TOP_LEVEL_FIELDS),
    )
    try:
        canonical = (
            json.dumps(
                raw_document,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
            + b"\n"
        )
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise HoldoutAccessError(
            "development replay report is not canonical JSON"
        ) from error
    if canonical != replay_report:
        raise HoldoutAccessError("development replay report serialization differs")
    if report["schema_version"] != 1 or type(report["schema_version"]) is not int:
        raise HoldoutAccessError(
            "development replay schema_version must be integer one"
        )

    identities = _validate_report_identities(report["identities"])
    runtime = _validate_report_runtime(report["runtime"])
    if report["target_order"] != list(TARGET_ORDER):
        raise HoldoutAccessError("development replay target order differs")

    positive = _require_object(
        report["positive_validation"],
        name="development replay positive validation",
        exact_keys={
            "target_example_count",
            "baseline_correct_count",
            "baseline_correct_by_target",
            "support_by_target",
        },
    )
    target_count = _require_int(
        positive["target_example_count"],
        name="target_example_count",
        minimum=1,
    )
    baseline = _require_int(
        positive["baseline_correct_count"],
        name="baseline_correct_count",
        minimum=1,
    )
    support = _require_int_vector(
        positive["support_by_target"],
        name="support_by_target",
        length=len(TARGET_ORDER),
    )
    baseline_by_target = _require_int_vector(
        positive["baseline_correct_by_target"],
        name="baseline_correct_by_target",
        length=len(TARGET_ORDER),
    )
    if (
        target_count != TARGET_EXAMPLE_COUNT
        or baseline != BASELINE_CORRECT_COUNT
        or support != TARGET_SUPPORT
        or baseline_by_target != BASELINE_CORRECT_BY_TARGET
        or sum(baseline_by_target) != baseline
        or sum(support) != target_count
    ):
        raise HoldoutAccessError("development replay positive denominators differ")

    dev = _require_object(
        report["dev"],
        name="development replay dev",
        exact_keys={
            "source_samples",
            "scored_exposure_samples",
            "utterance_count",
            "speaker_count",
            "speaker_rows",
        },
    )
    source_samples = _require_int(
        dev["source_samples"], name="source_samples", minimum=1
    )
    exposure = _require_int(
        dev["scored_exposure_samples"],
        name="scored_exposure_samples",
        minimum=1,
    )
    if exposure > source_samples:
        raise HoldoutAccessError("scored exposure exceeds development source samples")
    utterance_count = _require_int(
        dev["utterance_count"], name="utterance_count", minimum=1
    )
    speaker_count = _require_int(dev["speaker_count"], name="speaker_count", minimum=1)
    if (
        type(dev["speaker_rows"]) is not list
        or len(cast(list[object], dev["speaker_rows"])) != speaker_count
    ):
        raise HoldoutAccessError("speaker_rows differs from speaker_count")
    speaker_event_totals = [0] * THRESHOLD_COUNT
    speaker_exposure_total = 0
    speaker_utterance_total = 0
    previous_speaker_id: int | None = None
    for row_index, raw_speaker in enumerate(cast(list[object], dev["speaker_rows"])):
        speaker = _require_object(
            raw_speaker,
            name=f"speaker row {row_index}",
            exact_keys={
                "speaker_id",
                "utterance_count",
                "scored_exposure_samples",
                "event_count",
            },
        )
        speaker_id = _require_int(
            speaker["speaker_id"], name=f"speaker row {row_index} ID", minimum=0
        )
        if previous_speaker_id is not None and speaker_id <= previous_speaker_id:
            raise HoldoutAccessError(
                "speaker rows are not in ascending unique ID order"
            )
        previous_speaker_id = speaker_id
        speaker_utterance_total += _require_int(
            speaker["utterance_count"],
            name=f"speaker row {row_index} utterance_count",
            minimum=1,
        )
        speaker_exposure_total += _require_int(
            speaker["scored_exposure_samples"],
            name=f"speaker row {row_index} scored_exposure_samples",
            minimum=0,
        )
        speaker_events = _require_int_vector(
            speaker["event_count"],
            name=f"speaker row {row_index} event_count",
            length=THRESHOLD_COUNT,
        )
        for threshold, count in enumerate(speaker_events):
            speaker_event_totals[threshold] += count
    if speaker_exposure_total != exposure or speaker_utterance_total != utterance_count:
        raise HoldoutAccessError("speaker rows do not reproduce development totals")

    if (
        type(report["thresholds"]) is not list
        or len(cast(list[object], report["thresholds"])) != THRESHOLD_COUNT
    ):
        raise HoldoutAccessError("thresholds must contain the exact 1001-point grid")
    false_gate: list[int] = []
    retention_gate: list[int] = []
    passing: list[int] = []
    previous_events: int | None = None
    previous_correct: int | None = None
    previous_correct_by_target: tuple[int, ...] | None = None
    for threshold, raw_threshold in enumerate(cast(list[object], report["thresholds"])):
        row = _require_object(
            raw_threshold,
            name=f"threshold row {threshold}",
            exact_keys=set(REPLAY_THRESHOLD_FIELDS),
        )
        observed_threshold = _require_int(
            row["threshold_milli"],
            name=f"threshold row {threshold} threshold_milli",
            minimum=0,
            maximum=1_000,
        )
        if observed_threshold != threshold:
            raise HoldoutAccessError(
                "threshold rows are not the registered integer grid"
            )
        events = _require_int(
            row["dev_event_count"],
            name=f"threshold row {threshold} dev_event_count",
            minimum=0,
        )
        events_by_target = _require_int_vector(
            row["dev_event_count_by_target"],
            name=f"threshold row {threshold} dev_event_count_by_target",
            length=len(TARGET_ORDER),
        )
        correct = _require_int(
            row["validation_correct_accept_count"],
            name=f"threshold row {threshold} validation_correct_accept_count",
            minimum=0,
            maximum=baseline,
        )
        correct_by_target = _require_int_vector(
            row["validation_correct_accept_count_by_target"],
            name=f"threshold row {threshold} validation_correct_accept_count_by_target",
            length=len(TARGET_ORDER),
        )
        if events != sum(events_by_target) or events != speaker_event_totals[threshold]:
            raise HoldoutAccessError("development event sufficient statistics disagree")
        if correct != sum(correct_by_target) or any(
            count > support[index] for index, count in enumerate(correct_by_target)
        ):
            raise HoldoutAccessError("positive sufficient statistics disagree")
        if previous_events is not None and events > previous_events:
            raise HoldoutAccessError("development events increase across thresholds")
        if previous_correct is not None and correct > previous_correct:
            raise HoldoutAccessError("correct accepts increase across thresholds")
        if previous_correct_by_target is not None and any(
            count > previous_correct_by_target[index]
            for index, count in enumerate(correct_by_target)
        ):
            raise HoldoutAccessError(
                "per-target correct accepts increase across thresholds"
            )
        previous_events = events
        previous_correct = correct
        previous_correct_by_target = correct_by_target
        if threshold == 0 and correct_by_target != BASELINE_CORRECT_BY_TARGET:
            raise HoldoutAccessError("threshold-zero per-target baseline differs")

        expected_rate = events * EXPOSURE_SAMPLES_PER_HOUR / exposure
        if (
            _require_number(
                row["false_events_per_hour"],
                name=f"threshold row {threshold} false_events_per_hour",
                minimum=0,
            )
            != expected_rate
        ):
            raise HoldoutAccessError("false-events-per-hour derivation differs")
        _require_interval(
            row["garwood_95_percent"],
            name=f"threshold row {threshold} Garwood interval",
        )
        _require_interval(
            row["speaker_bootstrap_95_percent"],
            name=f"threshold row {threshold} speaker-bootstrap interval",
        )
        if (
            _require_number(
                row["correct_accept_recall"],
                name=f"threshold row {threshold} correct_accept_recall",
                minimum=0,
            )
            != correct / target_count
        ):
            raise HoldoutAccessError("correct-accept recall derivation differs")
        if (
            _require_number(
                row["conditional_correct_retention"],
                name=f"threshold row {threshold} conditional retention",
                minimum=0,
            )
            != correct / baseline
        ):
            raise HoldoutAccessError("conditional-retention derivation differs")

        negative_pass = events * EXPOSURE_SAMPLES_PER_HOUR <= exposure
        retention_pass = correct * 5 >= baseline * 4
        if negative_pass:
            false_gate.append(threshold)
        if retention_pass:
            retention_gate.append(threshold)
        if negative_pass and retention_pass:
            passing.append(threshold)
    if (
        previous_correct is None
        or cast(dict[str, object], cast(list[object], report["thresholds"])[0])[
            "validation_correct_accept_count"
        ]
        != baseline
    ):
        raise HoldoutAccessError("threshold-zero baseline count differs")

    expected_selection = _Selection(
        status="pass" if passing else "reject",
        threshold_milli=passing[0] if passing else None,
    )
    negative_frontier = false_gate[0] if false_gate else None
    retention_frontier = retention_gate[-1]
    raw_selection = _require_object(
        report["selection"],
        name="development replay selection",
        exact_keys={
            "status",
            "selected_threshold_milli",
            "negative_frontier_milli",
            "retention_frontier_milli",
        },
    )
    reported_status = raw_selection["status"]
    if type(reported_status) is not str or reported_status not in {"pass", "reject"}:
        raise HoldoutAccessError("reported selection status is invalid")
    raw_selected = raw_selection["selected_threshold_milli"]
    reported_selected = (
        None
        if raw_selected is None
        else _require_int(
            raw_selected,
            name="reported selected_threshold_milli",
            minimum=0,
            maximum=1_000,
        )
    )
    raw_negative_frontier = raw_selection["negative_frontier_milli"]
    reported_negative_frontier = (
        None
        if raw_negative_frontier is None
        else _require_int(
            raw_negative_frontier,
            name="reported negative_frontier_milli",
            minimum=0,
            maximum=1_000,
        )
    )
    reported_retention_frontier = _require_int(
        raw_selection["retention_frontier_milli"],
        name="reported retention_frontier_milli",
        minimum=0,
        maximum=1_000,
    )
    reported_selection: dict[str, object] = {
        "status": reported_status,
        "selected_threshold_milli": reported_selected,
        "negative_frontier_milli": reported_negative_frontier,
        "retention_frontier_milli": reported_retention_frontier,
    }
    if reported_selection != {
        "status": expected_selection.status,
        "selected_threshold_milli": expected_selection.threshold_milli,
        "negative_frontier_milli": negative_frontier,
        "retention_frontier_milli": retention_frontier,
    }:
        raise HoldoutAccessError(
            "reported selection differs from integer recomputation"
        )

    evidence_thresholds = tuple(
        sorted(
            {
                0,
                retention_frontier,
                *(() if negative_frontier is None else (negative_frontier,)),
                *(
                    ()
                    if expected_selection.threshold_milli is None
                    else (expected_selection.threshold_milli,)
                ),
            }
        )
    )
    _validate_top_false_events(
        report["top_false_events"], expected_thresholds=evidence_thresholds
    )
    return _ReplayReport(
        selection=expected_selection,
        identities=identities,
        runtime=runtime,
        source_samples=source_samples,
        scored_exposure_samples=exposure,
        utterance_count=utterance_count,
        speaker_count=speaker_count,
    )


def recompute_selection(replay_report: bytes) -> tuple[str, int | None]:
    """Strictly parse a replay report and independently recompute its selection."""

    selection = _parse_replay_report(replay_report).selection
    return selection.status, selection.threshold_milli


def canonical_runtime_identity(identity: Mapping[str, str]) -> bytes:
    """Serialize the exact registered runtime identity as canonical ASCII JSON."""

    if set(identity) != set(RUNTIME_FIELDS) or any(
        type(identity[field]) is not str or not identity[field]
        for field in RUNTIME_FIELDS
    ):
        raise HoldoutAccessError("runtime identity has missing or invalid fields")
    return (
        json.dumps(
            dict(identity),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def current_runtime_identity() -> dict[str, str]:
    """Return the runtime values registered by experiment 001."""

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "soundfile": sf.__version__,
        "libsndfile": sf.__libsndfile_version__,
        "platform": platform.platform(),
    }


def current_runtime_identity_sha256() -> str:
    """Hash the current canonical runtime identity document."""

    return hashlib.sha256(
        canonical_runtime_identity(current_runtime_identity())
    ).hexdigest()


def _physical_repository_root(repository_root: Path) -> Path:
    absolute_root = Path(os.path.abspath(repository_root))
    resolved_root = _resolve_existing_path(repository_root, name="repository root")
    if absolute_root != resolved_root or not resolved_root.is_dir():
        raise HoldoutAccessError("repository root must be one physical directory")
    git_dir = resolved_root / ".git"
    resolved_git_dir = _resolve_existing_path(git_dir, name="Git metadata directory")
    if resolved_git_dir != git_dir or not resolved_git_dir.is_dir():
        raise HoldoutAccessError("Git metadata must be one physical directory")
    return resolved_root


def _read_git_control_file(path: Path, *, name: str) -> bytes:
    absolute_path = Path(os.path.abspath(path))
    resolved_path = _resolve_existing_path(path, name=name)
    if resolved_path != absolute_path:
        raise HoldoutAccessError(f"{name} path must be fully physical")
    control = _read_stable_file(
        resolved_path,
        name=name,
        maximum_bytes=MAX_GIT_CONTROL_FILE_BYTES,
        keep_contents=True,
    )
    if control.contents is None:
        raise HoldoutAccessError(f"{name} contents are unavailable")
    return control.contents


def _commit_from_control_bytes(contents: bytes, *, name: str) -> str:
    if not contents.endswith(b"\n") or contents.count(b"\n") != 1:
        raise HoldoutAccessError(f"{name} must contain one LF-terminated commit")
    try:
        value = contents[:-1].decode("ascii")
    except UnicodeDecodeError as error:
        raise HoldoutAccessError(f"{name} is not ASCII") from error
    return _require_git_commit(value, name=name)


def _current_git_head(repository_root: Path) -> str:
    resolved_root = _physical_repository_root(repository_root)
    git_dir = resolved_root / ".git"
    head = _read_git_control_file(git_dir / "HEAD", name="Git HEAD")
    if not head.startswith(b"ref: "):
        return _commit_from_control_bytes(head, name="detached Git HEAD")
    if not head.endswith(b"\n") or head.count(b"\n") != 1:
        raise HoldoutAccessError("symbolic Git HEAD is malformed")
    try:
        ref_name = head[5:-1].decode("ascii")
    except UnicodeDecodeError as error:
        raise HoldoutAccessError("symbolic Git HEAD is not ASCII") from error
    if (
        re.fullmatch(r"refs/(?:[A-Za-z0-9._-]+/)*[A-Za-z0-9._-]+", ref_name)
        is None
        or ".." in ref_name
        or "@{" in ref_name
    ):
        raise HoldoutAccessError("symbolic Git HEAD ref is unsafe")
    loose_ref = git_dir / Path(ref_name)
    try:
        return _commit_from_control_bytes(
            _read_git_control_file(loose_ref, name="Git HEAD ref"),
            name="Git HEAD ref",
        )
    except HoldoutAccessError as loose_error:
        if loose_ref.exists() or loose_ref.is_symlink():
            raise
        packed_path = git_dir / "packed-refs"
        try:
            packed = _read_git_control_file(packed_path, name="packed Git refs")
        except HoldoutAccessError:
            raise loose_error from None
    try:
        lines = packed.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise HoldoutAccessError("packed Git refs are not ASCII") from error
    matches: list[str] = []
    for line in lines:
        if not line or line.startswith(("#", "^")):
            continue
        commit, separator, observed_ref = line.partition(" ")
        if separator and observed_ref == ref_name:
            matches.append(_require_git_commit(commit, name="packed Git ref"))
    if len(matches) != 1:
        raise HoldoutAccessError("packed Git refs do not contain one HEAD ref")
    return matches[0]


def _copy_git_object_file(source: Path, destination: Path, *, budget: int) -> int:
    descriptor = -1
    output = -1
    try:
        descriptor = os.open(source, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 1
            or before.st_size > budget
        ):
            raise HoldoutAccessError("Git object is not one bounded regular file")
        initial_snapshot = _snapshot(before)
        output = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o600,
        )
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise HoldoutAccessError("Git object ended before its recorded size")
            view = memoryview(chunk)
            while view:
                written = os.write(output, view)
                if written < 1:
                    raise HoldoutAccessError("cannot copy Git object snapshot")
                view = view[written:]
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise HoldoutAccessError("Git object exceeds its recorded size")
        if _snapshot(os.fstat(descriptor)) != initial_snapshot:
            raise HoldoutAccessError("Git object changed while being copied")
    except OSError as error:
        raise HoldoutAccessError(f"cannot snapshot Git object: {error}") from error
    finally:
        if output >= 0:
            os.close(output)
        if descriptor >= 0:
            os.close(descriptor)
    return before.st_size


def _copy_git_object_store(source: Path, destination: Path) -> None:
    resolved_source = _resolve_existing_path(source, name="Git object directory")
    if resolved_source != Path(os.path.abspath(source)) or not resolved_source.is_dir():
        raise HoldoutAccessError("Git object directory must be fully physical")
    destination.mkdir(mode=0o700)
    remaining = MAX_GIT_OBJECT_SNAPSHOT_BYTES
    with os.scandir(resolved_source) as entries:
        root_entries = sorted(entries, key=lambda entry: entry.name)
    for entry in root_entries:
        if entry.name == "info":
            if not entry.is_dir(follow_symlinks=False):
                raise HoldoutAccessError("Git objects/info is not a physical directory")
            continue
        is_loose_directory = re.fullmatch(r"[0-9a-f]{2}", entry.name) is not None
        if entry.name != "pack" and not is_loose_directory:
            raise HoldoutAccessError("Git object store contains an unexpected entry")
        if not entry.is_dir(follow_symlinks=False):
            raise HoldoutAccessError("Git object shard is not a physical directory")
        source_directory = Path(entry.path)
        destination_directory = destination / entry.name
        destination_directory.mkdir(mode=0o700)
        with os.scandir(source_directory) as children:
            child_entries = sorted(children, key=lambda child: child.name)
        for child in child_entries:
            if is_loose_directory:
                valid_name = re.fullmatch(r"[0-9a-f]{38}", child.name) is not None
            else:
                valid_name = (
                    re.fullmatch(
                        r"pack-[0-9a-f]{40}\.(?:pack|idx|rev)", child.name
                    )
                    is not None
                )
            if not valid_name or not child.is_file(follow_symlinks=False):
                raise HoldoutAccessError("Git object shard contains an unsafe entry")
            copied = _copy_git_object_file(
                Path(child.path), destination_directory / child.name, budget=remaining
            )
            remaining -= copied


@contextmanager
def _isolated_git_snapshot(repository_root: Path) -> Iterator[_GitSnapshot]:
    resolved_root = _physical_repository_root(repository_root)
    git_dir = resolved_root / ".git"
    head_commit = _current_git_head(resolved_root)
    index_file = _read_stable_file(
        git_dir / "index",
        name="Git index",
        maximum_bytes=MAX_GIT_INDEX_BYTES,
        keep_contents=True,
    )
    if index_file.contents is None:
        raise HoldoutAccessError("Git index contents are unavailable")
    try:
        with tempfile.TemporaryDirectory(
            prefix="falsewake-git-snapshot-", dir=git_dir
        ) as raw_snapshot:
            snapshot_dir = Path(raw_snapshot)
            (snapshot_dir / "refs").mkdir(mode=0o700)
            (snapshot_dir / "HEAD").write_text(
                "ref: refs/heads/unborn\n", encoding="ascii"
            )
            (snapshot_dir / "config").write_text(
                "[core]\n\trepositoryformatversion = 0\n\tbare = true\n",
                encoding="ascii",
            )
            (snapshot_dir / "index").write_bytes(index_file.contents)
            (snapshot_dir / "index").chmod(0o600)
            _copy_git_object_store(git_dir / "objects", snapshot_dir / "objects")
            yield _GitSnapshot(
                repository_root=resolved_root,
                git_dir=snapshot_dir,
                head_commit=head_commit,
                index_file=index_file,
            )
    except OSError as error:
        raise HoldoutAccessError(
            f"cannot create isolated Git snapshot: {error}"
        ) from error


def _run_git(snapshot: _GitSnapshot, arguments: list[str], *, name: str) -> bytes:
    executable = shutil.which("git", path=os.defpath)
    if executable is None:
        raise HoldoutAccessError("cannot find Git in the system executable path")
    environment = {
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_DIR": os.fspath(snapshot.git_dir),
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PAGER": "cat",
        "HOME": os.devnull,
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": os.defpath,
    }
    try:
        completed = subprocess.run(
            [
                executable,
                "--no-pager",
                "--no-replace-objects",
                "--git-dir",
                os.fspath(snapshot.git_dir),
                *arguments,
            ],
            check=False,
            capture_output=True,
            cwd=snapshot.git_dir,
            env=environment,
            stdin=subprocess.DEVNULL,
        )
    except OSError as error:
        raise HoldoutAccessError(f"cannot run Git for {name}: {error}") from error
    if completed.returncode != 0:
        detail = completed.stderr[:512].decode("utf-8", errors="replace").strip()
        raise HoldoutAccessError(f"Git rejected {name}: {detail}")
    return completed.stdout


def _git_blob(snapshot: _GitSnapshot, *, commit: str, relative_path: str) -> bytes:
    tree_entry = _run_git(
        snapshot,
        ["ls-tree", "-z", commit, "--", relative_path],
        name=f"tree entry for {relative_path}",
    )
    prefix = b"100644 blob "
    executable_prefix = b"100755 blob "
    if not tree_entry.endswith(b"\0") or not tree_entry.startswith(
        (prefix, executable_prefix)
    ):
        raise HoldoutAccessError(f"{relative_path} is not one regular Git blob")
    metadata, separator, observed_path = tree_entry[:-1].partition(b"\t")
    if not separator or observed_path != relative_path.encode("utf-8"):
        raise HoldoutAccessError(f"Git returned an ambiguous entry for {relative_path}")
    fields = metadata.split(b" ")
    if len(fields) != 3 or len(fields[2]) != 40:
        raise HoldoutAccessError(f"Git returned a malformed entry for {relative_path}")
    size_bytes = _run_git(
        snapshot,
        ["cat-file", "-s", fields[2].decode("ascii")],
        name=f"blob size for {relative_path}",
    )
    try:
        size = int(size_bytes.strip())
    except ValueError as error:
        raise HoldoutAccessError(
            f"Git returned an invalid size for {relative_path}"
        ) from error
    if size < 0 or size > MAX_SOURCE_BYTES:
        raise HoldoutAccessError(f"Git blob for {relative_path} exceeds the size limit")
    blob = _run_git(
        snapshot,
        ["cat-file", "blob", fields[2].decode("ascii")],
        name=f"blob for {relative_path}",
    )
    if len(blob) != size:
        raise HoldoutAccessError(f"Git blob size changed for {relative_path}")
    return blob


def _parse_git_entries(
    output: bytes,
    *,
    source: str,
) -> dict[bytes, tuple[bytes, bytes]]:
    if len(output) > MAX_GIT_INDEX_BYTES * 4:
        raise HoldoutAccessError(f"{source} output exceeds the size limit")
    if output and not output.endswith(b"\0"):
        raise HoldoutAccessError(f"{source} output is not NUL terminated")
    records = () if not output else output[:-1].split(b"\0")
    parsed: dict[bytes, tuple[bytes, bytes]] = {}
    for record in records:
        metadata, separator, path = record.partition(b"\t")
        fields = metadata.split(b" ")
        if not separator or not path or len(fields) != 3:
            raise HoldoutAccessError(f"{source} returned a malformed entry")
        mode, second, third = fields
        if source == "captured Git index":
            object_id = second
            if third != b"0":
                raise HoldoutAccessError("captured Git index has unmerged entries")
        else:
            object_id = third
            if (mode == b"160000" and second != b"commit") or (
                mode != b"160000" and second != b"blob"
            ):
                raise HoldoutAccessError(
                    "artifact HEAD tree has an invalid object type"
                )
        if (
            mode not in {b"100644", b"100755", b"120000", b"160000"}
            or len(object_id) != 40
            or any(byte not in b"0123456789abcdef" for byte in object_id)
        ):
            raise HoldoutAccessError(f"{source} returned an invalid entry identity")
        if path in parsed:
            raise HoldoutAccessError(f"{source} repeats a path")
        parsed[path] = (mode, object_id)
    return parsed


def _require_index_matches_head(
    snapshot: _GitSnapshot, *, head_commit: str
) -> None:
    index_entries = _parse_git_entries(
        _run_git(
            snapshot,
            ["ls-files", "--stage", "-z"],
            name="captured Git index entries",
        ),
        source="captured Git index",
    )
    head_entries = _parse_git_entries(
        _run_git(
            snapshot,
            ["ls-tree", "-r", "-z", "--full-tree", head_commit],
            name="artifact HEAD entries",
        ),
        source="artifact HEAD tree",
    )
    if index_entries != head_entries:
        raise HoldoutAccessError("Git index differs from the artifact HEAD")


def _validate_tracked_payload(
    snapshot: _GitSnapshot,
    *,
    implementation_commit: str,
    head_commit: str,
    relative_path: str,
    contents: bytes,
    name: str,
) -> None:
    if (
        _git_blob(
            snapshot,
            commit=implementation_commit,
            relative_path=relative_path,
        )
        != contents
    ):
        raise HoldoutAccessError(f"{name} differs from the implementation commit")
    if (
        _git_blob(snapshot, commit=head_commit, relative_path=relative_path)
        != contents
    ):
        raise HoldoutAccessError(f"artifact HEAD changes {name}")


def _validate_selection_artifact_commit(
    snapshot: _GitSnapshot,
) -> _SelectionArtifactState:
    resolved_root = snapshot.repository_root
    head_commit = snapshot.head_commit
    artifact_path = resolved_root / SELECTION_ARTIFACT_RELATIVE_PATH
    try:
        resolved_artifact = artifact_path.resolve(strict=True)
    except OSError as error:
        raise HoldoutAccessError(
            f"cannot resolve selection artifact: {error}"
        ) from error
    if resolved_artifact != Path(os.path.abspath(artifact_path)):
        raise HoldoutAccessError("selection artifact path must be fully physical")
    artifact_file = _read_stable_file(
        artifact_path,
        name="selection artifact",
        maximum_bytes=MAX_ARTIFACT_BYTES,
        keep_contents=True,
    )
    if artifact_file.contents is None:
        raise HoldoutAccessError("selection artifact contents are unavailable")
    artifact = _parse_selection_artifact(artifact_file.contents)
    committed = _git_blob(
        snapshot,
        commit=head_commit,
        relative_path=SELECTION_ARTIFACT_RELATIVE_PATH,
    )
    if committed != artifact_file.contents:
        raise HoldoutAccessError("selection artifact differs from its HEAD blob")
    if _current_git_head(resolved_root) != head_commit:
        raise HoldoutAccessError("Git HEAD changed while validating selection artifact")
    return _SelectionArtifactState(
        file=artifact_file,
        artifact=artifact,
        head_commit=head_commit,
    )


def _revalidate_selection_artifact_commit(
    snapshot: _GitSnapshot,
    expected: _SelectionArtifactState,
) -> None:
    observed = _validate_selection_artifact_commit(snapshot)
    if (
        observed.head_commit != expected.head_commit
        or observed.file.identity != expected.file.identity
        or not hmac.compare_digest(observed.file.sha256, expected.file.sha256)
        or observed.artifact != expected.artifact
    ):
        raise HoldoutAccessError("selection artifact changed during authorization")


def _validate_git_state(
    snapshot: _GitSnapshot,
    *,
    expected_head_commit: str,
    implementation_commit: str,
    config_contents: bytes,
) -> _GitState:
    resolved_root = snapshot.repository_root
    if snapshot.head_commit != expected_head_commit:
        raise HoldoutAccessError("captured Git HEAD differs from the artifact commit")
    if _current_git_head(resolved_root) != expected_head_commit:
        raise HoldoutAccessError("Git HEAD differs from the selection artifact commit")
    _require_index_matches_head(snapshot, head_commit=expected_head_commit)

    commit_type = _run_git(
        snapshot,
        ["cat-file", "-t", implementation_commit],
        name="implementation object type",
    )
    if commit_type != b"commit\n":
        raise HoldoutAccessError("implementation_git_commit is not a commit object")
    _run_git(
        snapshot,
        ["cat-file", "-e", f"{implementation_commit}^{{commit}}"],
        name="implementation commit",
    )
    _run_git(
        snapshot,
        [
            "merge-base",
            "--is-ancestor",
            implementation_commit,
            expected_head_commit,
        ],
        name="implementation commit ancestry",
    )
    if implementation_commit == expected_head_commit:
        raise HoldoutAccessError(
            "implementation commit must strictly precede the selection artifact"
        )
    implementation_artifact = _run_git(
        snapshot,
        [
            "ls-tree",
            "-z",
            implementation_commit,
            "--",
            SELECTION_ARTIFACT_RELATIVE_PATH,
        ],
        name="implementation selection-artifact absence",
    )
    if implementation_artifact:
        raise HoldoutAccessError(
            "selection artifact must be absent from the implementation commit"
        )
    config_blob = _git_blob(
        snapshot,
        commit=implementation_commit,
        relative_path="configs/experiment-001.json",
    )
    if config_blob != config_contents:
        raise HoldoutAccessError(
            "experiment config differs from the implementation commit"
        )
    if (
        _git_blob(
            snapshot,
            commit=expected_head_commit,
            relative_path="configs/experiment-001.json",
        )
        != config_contents
    ):
        raise HoldoutAccessError("current HEAD changes the experiment config")

    digest = hashlib.sha256()
    digest.update(_SOURCE_DOMAIN)
    source_files: list[_StableFile] = []
    for relative_path in SCORER_SOURCE_FILES:
        working = _read_stable_file(
            resolved_root / relative_path,
            name=f"scorer source {relative_path}",
            maximum_bytes=MAX_SOURCE_BYTES,
            keep_contents=True,
        )
        if working.contents is None:
            raise HoldoutAccessError(f"cannot retain scorer source {relative_path}")
        if relative_path == "src/falsewake/holdout.py" and not hmac.compare_digest(
            working.sha256, EXECUTING_HOLDOUT_SHA256
        ):
            raise HoldoutAccessError(
                "validated holdout source differs from the executing module"
            )
        committed = _git_blob(
            snapshot,
            commit=implementation_commit,
            relative_path=relative_path,
        )
        if working.contents != committed:
            raise HoldoutAccessError(
                f"scorer source {relative_path} differs from the implementation commit"
            )
        if (
            _git_blob(
                snapshot,
                commit=expected_head_commit,
                relative_path=relative_path,
            )
            != committed
        ):
            raise HoldoutAccessError(
                f"current HEAD changes scorer source {relative_path}"
            )
        encoded_path = relative_path.encode("utf-8")
        digest.update(struct.pack("<I", len(encoded_path)))
        digest.update(encoded_path)
        digest.update(struct.pack("<Q", len(working.contents)))
        digest.update(working.contents)
        source_files.append(working)
    if _current_git_head(resolved_root) != expected_head_commit:
        raise HoldoutAccessError("Git HEAD changed while validating implementation")
    return _GitState(
        scorer_source_sha256=digest.hexdigest(),
        files=tuple(source_files),
    )


def _verify_digest(observed: str, expected: object, *, name: str) -> None:
    registered = _require_sha256(expected, name=name)
    if not hmac.compare_digest(observed, registered):
        raise HoldoutAccessError(f"{name} differs")


def _validate_audit_report(
    contents: bytes,
    *,
    archive_sha256: str,
    manifest_sha256: str,
) -> _AuditPopulation:
    report = _require_object(
        _parse_json(contents, name="development audit report"),
        name="development audit report",
    )
    _verify_digest(
        archive_sha256, report.get("archive_sha256"), name="audit archive SHA-256"
    )
    _verify_digest(
        manifest_sha256,
        report.get("manifest_sha256"),
        name="audit manifest SHA-256",
    )
    source_samples = _require_int(
        report.get("source_samples"), name="audit source_samples", minimum=1
    )
    utterance_count = _require_int(
        report.get("utterance_count"), name="audit utterance_count", minimum=1
    )
    speaker_count = _require_int(
        report.get("speaker_count"), name="audit speaker_count", minimum=1
    )
    flac_count = _require_int(
        report.get("flac_count"), name="audit flac_count", minimum=1
    )
    if flac_count != utterance_count:
        raise HoldoutAccessError("audit FLAC and utterance counts differ")
    return _AuditPopulation(
        source_samples=source_samples,
        utterance_count=utterance_count,
        speaker_count=speaker_count,
    )


def _validate_dev_manifest(contents: bytes) -> _ManifestPopulation:
    if not contents or not contents.endswith(b"\n") or b"\r" in contents:
        raise HoldoutAccessError(
            "development manifest must be nonempty LF-terminated JSONL"
        )

    exact_keys = {
        "utterance_id",
        "relative_path",
        "speaker_id",
        "chapter_id",
        "sample_count",
        "raw_flac_sha256",
        "decoded_pcm16le_sha256",
        "transcript",
    }
    decimal_id = re.compile(r"[1-9][0-9]*", re.ASCII)
    utterance_id_pattern = re.compile(
        r"([1-9][0-9]*)-([1-9][0-9]*)-([0-9]{4})", re.ASCII
    )
    seen_utterances: set[str] = set()
    seen_paths: set[str] = set()
    speakers: set[int] = set()
    source_samples = 0
    scored_exposure_samples = 0
    previous_path: str | None = None

    for row_index, line in enumerate(contents.splitlines(keepends=True)):
        if line == b"\n" or not line.endswith(b"\n"):
            raise HoldoutAccessError(
                "development manifest contains a blank or unterminated row"
            )
        row_bytes = line[:-1]
        raw_row = _parse_json(row_bytes, name=f"development manifest row {row_index}")
        row = _require_object(
            raw_row,
            name=f"development manifest row {row_index}",
            exact_keys=exact_keys,
        )
        try:
            canonical_row = json.dumps(
                raw_row,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        except (TypeError, ValueError, UnicodeEncodeError) as error:
            raise HoldoutAccessError(
                f"development manifest row {row_index} is not canonical JSON"
            ) from error
        if canonical_row != row_bytes:
            raise HoldoutAccessError(
                f"development manifest row {row_index} serialization differs"
            )

        speaker_id = row["speaker_id"]
        chapter_id = row["chapter_id"]
        utterance_id = row["utterance_id"]
        relative_path = row["relative_path"]
        transcript = row["transcript"]
        if type(speaker_id) is not str or decimal_id.fullmatch(speaker_id) is None:
            raise HoldoutAccessError("development manifest speaker_id is invalid")
        if type(chapter_id) is not str or decimal_id.fullmatch(chapter_id) is None:
            raise HoldoutAccessError("development manifest chapter_id is invalid")
        if type(utterance_id) is not str:
            raise HoldoutAccessError("development manifest utterance_id is invalid")
        utterance_match = utterance_id_pattern.fullmatch(utterance_id)
        if utterance_match is None or utterance_match.group(1, 2) != (
            speaker_id,
            chapter_id,
        ):
            raise HoldoutAccessError("development manifest utterance_id is invalid")
        expected_path = (
            f"LibriSpeech/dev-clean/{speaker_id}/{chapter_id}/{utterance_id}.flac"
        )
        if type(relative_path) is not str or relative_path != expected_path:
            raise HoldoutAccessError("development manifest relative_path is invalid")
        if previous_path is not None and relative_path <= previous_path:
            raise HoldoutAccessError(
                "development manifest paths are not in ascending unique order"
            )
        previous_path = relative_path
        if utterance_id in seen_utterances or relative_path in seen_paths:
            raise HoldoutAccessError("development manifest repeats an utterance")
        seen_utterances.add(utterance_id)
        seen_paths.add(relative_path)

        sample_count = _require_int(
            row["sample_count"],
            name=f"development manifest row {row_index} sample_count",
            minimum=1,
            maximum=9_600_000,
        )
        _require_sha256(
            row["raw_flac_sha256"],
            name=f"development manifest row {row_index} raw FLAC SHA-256",
        )
        _require_sha256(
            row["decoded_pcm16le_sha256"],
            name=f"development manifest row {row_index} decoded PCM SHA-256",
        )
        if type(transcript) is not str or not transcript:
            raise HoldoutAccessError("development manifest transcript is invalid")

        source_samples += sample_count
        if sample_count >= 16_000:
            scored_exposure_samples += (
                16_000 + ((sample_count - 16_000) // 1_600) * 1_600
            )
        speakers.add(int(speaker_id))

    return _ManifestPopulation(
        source_samples=source_samples,
        scored_exposure_samples=scored_exposure_samples,
        utterance_count=len(seen_utterances),
        speaker_count=len(speakers),
    )


def _memfd_sealing_parameters() -> tuple[int, int, int, int]:
    flag_names = ("MFD_CLOEXEC", "MFD_ALLOW_SEALING")
    command_names = ("F_ADD_SEALS", "F_GET_SEALS")
    seal_names = ("F_SEAL_WRITE", "F_SEAL_GROW", "F_SEAL_SHRINK", "F_SEAL_SEAL")
    flags = tuple(getattr(os, name, None) for name in flag_names)
    commands = tuple(getattr(fcntl, name, None) for name in command_names)
    seals = tuple(getattr(fcntl, name, None) for name in seal_names)
    if (
        not callable(getattr(os, "memfd_create", None))
        or any(type(value) is not int for value in (*flags, *commands, *seals))
    ):
        raise HoldoutAccessError(
            "test-clean acquisition requires Linux memfd sealing support"
        )
    integer_flags = cast(tuple[int, int], flags)
    integer_commands = cast(tuple[int, int], commands)
    integer_seals = cast(tuple[int, int, int, int], seals)
    return (
        integer_flags[0] | integer_flags[1],
        integer_commands[0],
        integer_commands[1],
        integer_seals[0] | integer_seals[1] | integer_seals[2] | integer_seals[3],
    )


def _test_archive_digests(descriptor: int, size: int) -> tuple[str, str]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    offset = 0
    while offset < size:
        chunk = os.pread(descriptor, min(1024 * 1024, size - offset), offset)
        if not chunk:
            raise HoldoutAccessError(
                "sealed test-clean ended before its registered size"
            )
        md5.update(chunk)
        sha256.update(chunk)
        offset += len(chunk)
    return md5.hexdigest(), sha256.hexdigest()


def _acquire_and_open_test_archive(
    *,
    acquire: TestCleanSupplier,
    authorization: _Authorization,
) -> AuthorizedTestClean:
    memfd_flags, add_seals, get_seals, required_seals = _memfd_sealing_parameters()
    descriptor = -1
    read_descriptor = -1
    try:
        create_memfd = cast(Callable[[str, int], int], os.memfd_create)
        descriptor = create_memfd("falsewake-test-clean", memfd_flags)
        empty_metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(empty_metadata.st_mode)
            or empty_metadata.st_nlink != 0
            or empty_metadata.st_size != 0
        ):
            raise HoldoutAccessError(
                "test-clean staging must begin as one empty anonymous regular file"
            )
        supplier_descriptor = -1
        try:
            supplier_descriptor = os.dup(descriptor)
            writable = os.fdopen(supplier_descriptor, "wb", buffering=0)
            supplier_descriptor = -1
            with writable:
                result = cast(Callable[[BinaryIO], object], acquire)(writable)
                writable.flush()
        except Exception as error:
            raise HoldoutAccessError(
                f"test-clean acquisition failed: {error}"
            ) from error
        finally:
            if supplier_descriptor >= 0:
                os.close(supplier_descriptor)
        if result is not None:
            raise HoldoutAccessError("test-clean supplier must return None")
        os.fsync(descriptor)
        fcntl.fcntl(descriptor, add_seals, required_seals)
        observed_seals = fcntl.fcntl(descriptor, get_seals)
        if observed_seals != required_seals:
            raise HoldoutAccessError("test-clean memfd is not irreversibly sealed")
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 0
            or metadata.st_size != TEST_ARCHIVE_BYTES
            or metadata.st_size > MAX_TEST_ARCHIVE_BYTES
        ):
            raise HoldoutAccessError(
                "acquired test-clean size or anonymous inode identity differs"
            )
        observed_md5, observed_sha256 = _test_archive_digests(
            descriptor, metadata.st_size
        )
        if not hmac.compare_digest(observed_md5, TEST_ARCHIVE_MD5):
            raise HoldoutAccessError("test-clean official MD5 differs")
        if not hmac.compare_digest(observed_sha256, TEST_ARCHIVE_SHA256):
            raise HoldoutAccessError("test-clean registered SHA-256 differs")
        read_descriptor = os.open(
            f"/proc/self/fd/{descriptor}",
            os.O_RDONLY | os.O_CLOEXEC,
        )
        read_metadata = os.fstat(read_descriptor)
        if (
            (read_metadata.st_dev, read_metadata.st_ino)
            != (metadata.st_dev, metadata.st_ino)
            or not stat.S_ISREG(read_metadata.st_mode)
            or read_metadata.st_nlink != 0
            or read_metadata.st_size != metadata.st_size
            or fcntl.fcntl(read_descriptor, get_seals) != required_seals
        ):
            raise HoldoutAccessError(
                "read-only test-clean descriptor differs from the sealed memfd"
            )
        os.close(descriptor)
        descriptor = -1
        os.lseek(read_descriptor, 0, os.SEEK_SET)
        stream = os.fdopen(read_descriptor, "rb")
        read_descriptor = -1
    except OSError as error:
        raise HoldoutAccessError(
            f"cannot acquire and seal test-clean: {error}"
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if read_descriptor >= 0:
            os.close(read_descriptor)
    return AuthorizedTestClean(
        stream=stream,
        selected_threshold_milli=authorization.threshold_milli,
        implementation_git_commit=authorization.implementation_git_commit,
        portable_model_json=authorization.portable_model_json,
        archive_sha256=observed_sha256,
    )


def _authorize_from_snapshot(
    snapshot: _GitSnapshot, *, evidence: HoldoutEvidence
) -> _Authorization:
    artifact_state = _validate_selection_artifact_commit(
        snapshot,
    )
    artifact_file = artifact_state.file
    artifact = artifact_state.artifact

    config_path = snapshot.repository_root / EXPERIMENT_CONFIG_RELATIVE_PATH
    config_file = _read_stable_file(
        config_path,
        name="experiment config",
        maximum_bytes=MAX_CONFIG_BYTES,
        keep_contents=True,
    )
    if config_file.contents is None:
        raise HoldoutAccessError("experiment config contents are unavailable")
    config = _validate_registered_config(config_file.contents)
    _verify_digest(
        config_file.sha256,
        artifact["experiment_config_sha256"],
        name="experiment config SHA-256",
    )

    implementation_commit = _require_git_commit(
        artifact["implementation_git_commit"], name="implementation_git_commit"
    )
    git_state = _validate_git_state(
        snapshot,
        expected_head_commit=artifact_state.head_commit,
        implementation_commit=implementation_commit,
        config_contents=config_file.contents,
    )
    _verify_digest(
        git_state.scorer_source_sha256,
        artifact["scorer_source_sha256"],
        name="scorer source SHA-256",
    )
    model_contract = _mapping_at(config, "model", name="portable model identity")
    if model_contract.get("portable_json_path") != PORTABLE_MODEL_RELATIVE_PATH:
        raise HoldoutAccessError("portable model path differs from experiment 001")
    model_path = snapshot.repository_root / PORTABLE_MODEL_RELATIVE_PATH
    model_file = _read_stable_file(
        model_path,
        name="portable model",
        maximum_bytes=MAX_SOURCE_BYTES,
        keep_contents=True,
    )
    if model_file.contents is None:
        raise HoldoutAccessError("portable model contents are unavailable")
    _verify_digest(
        model_file.sha256,
        model_contract.get("portable_json_sha256"),
        name="portable model SHA-256",
    )
    _validate_tracked_payload(
        snapshot,
        implementation_commit=implementation_commit,
        head_commit=artifact_state.head_commit,
        relative_path=PORTABLE_MODEL_RELATIVE_PATH,
        contents=model_file.contents,
        name="portable model",
    )
    runtime = current_runtime_identity()
    runtime_sha256 = hashlib.sha256(canonical_runtime_identity(runtime)).hexdigest()
    _verify_digest(
        runtime_sha256,
        artifact["runtime_identity_sha256"],
        name="runtime identity SHA-256",
    )

    dev_archive = _read_stable_file(
        evidence.dev_archive,
        name="development archive",
        maximum_bytes=MAX_DEV_ARCHIVE_BYTES,
        keep_contents=False,
    )
    _verify_digest(
        dev_archive.sha256,
        artifact["dev_archive_sha256"],
        name="development archive SHA-256",
    )
    negative_source = _mapping_at(config, "negative_source", name="negative source")
    _verify_digest(
        dev_archive.sha256,
        negative_source.get("archive_sha256"),
        name="registered development archive SHA-256",
    )
    verified_output = _mapping_at(
        _mapping_at(
            negative_source,
            "archive_audit",
            name="archive audit contract",
        ),
        "verified_dev_clean_output",
        name="verified development output",
    )

    dev_manifest = _read_stable_file(
        evidence.dev_manifest,
        name="development manifest",
        maximum_bytes=MAX_DEV_MANIFEST_BYTES,
        keep_contents=True,
    )
    _verify_digest(
        dev_manifest.sha256,
        artifact["dev_manifest_sha256"],
        name="development manifest SHA-256",
    )
    _verify_digest(
        dev_manifest.sha256,
        verified_output.get("manifest_sha256"),
        name="registered development manifest SHA-256",
    )
    if dev_manifest.contents is None:
        raise HoldoutAccessError("development manifest contents are unavailable")
    manifest_population = _validate_dev_manifest(dev_manifest.contents)
    dev_audit = _read_stable_file(
        evidence.dev_audit_report,
        name="development audit report",
        maximum_bytes=MAX_AUDIT_REPORT_BYTES,
        keep_contents=True,
    )
    _verify_digest(
        dev_audit.sha256,
        artifact["dev_audit_report_sha256"],
        name="development audit report SHA-256",
    )
    _verify_digest(
        dev_audit.sha256,
        verified_output.get("audit_report_sha256"),
        name="registered development audit report SHA-256",
    )
    if dev_audit.contents is None:
        raise HoldoutAccessError("development audit report contents are unavailable")
    audit_population = _validate_audit_report(
        dev_audit.contents,
        archive_sha256=dev_archive.sha256,
        manifest_sha256=dev_manifest.sha256,
    )
    if (
        manifest_population.source_samples != audit_population.source_samples
        or manifest_population.utterance_count != audit_population.utterance_count
        or manifest_population.speaker_count != audit_population.speaker_count
    ):
        raise HoldoutAccessError(
            "development manifest population differs from the payload audit"
        )
    if (
        manifest_population.source_samples != verified_output.get("source_samples")
        or manifest_population.scored_exposure_samples
        != verified_output.get("scored_exposure_samples")
        or manifest_population.utterance_count != verified_output.get("utterance_count")
        or manifest_population.speaker_count != verified_output.get("speaker_count")
    ):
        raise HoldoutAccessError(
            "development manifest population differs from the registered audit"
        )

    dev_replay = _read_stable_file(
        evidence.dev_replay_report,
        name="development replay report",
        maximum_bytes=MAX_REPLAY_REPORT_BYTES,
        keep_contents=True,
    )
    _verify_digest(
        dev_replay.sha256,
        artifact["dev_replay_report_sha256"],
        name="development replay report SHA-256",
    )
    if dev_replay.contents is None:
        raise HoldoutAccessError("development replay report contents are unavailable")
    replay = _parse_replay_report(dev_replay.contents)
    recomputed = replay.selection
    if replay.runtime != runtime:
        raise HoldoutAccessError(
            "development replay runtime differs from current runtime"
        )
    if (
        replay.source_samples != manifest_population.source_samples
        or replay.scored_exposure_samples != manifest_population.scored_exposure_samples
        or replay.utterance_count != manifest_population.utterance_count
        or replay.speaker_count != manifest_population.speaker_count
    ):
        raise HoldoutAccessError(
            "development replay population or exposure differs from the manifest"
        )
    positive = _mapping_at(config, "positive_validation", name="positive validation")
    feature_cache = _mapping_at(
        positive, "feature_cache", name="positive feature cache"
    )
    expected_replay_identities: dict[str, object] = {
        "experiment_config_sha256": config_file.sha256,
        "portable_model_sha256": model_contract.get("portable_json_sha256"),
        "scorer_source_sha256": git_state.scorer_source_sha256,
        "runtime_identity_sha256": runtime_sha256,
        "implementation_git_commit": implementation_commit,
        "dev_archive_sha256": dev_archive.sha256,
        "dev_manifest_sha256": dev_manifest.sha256,
        "dev_audit_report_sha256": dev_audit.sha256,
        "positive_examples_sha256": positive.get("examples_sha256"),
        "positive_feature_matrix_semantic_sha256": feature_cache.get("features_sha256"),
    }
    if replay.identities != expected_replay_identities:
        raise HoldoutAccessError("development replay identities differ")
    if artifact["status"] != recomputed.status:
        raise HoldoutAccessError("selection artifact status differs from recomputation")
    if artifact["selected_threshold_milli"] != recomputed.threshold_milli:
        raise HoldoutAccessError("selected threshold differs from recomputation")
    if recomputed.status != "pass" or recomputed.threshold_milli is None:
        raise HoldoutAccessError("development selection rejected test-clean access")

    _revalidate_selection_artifact_commit(
        snapshot,
        artifact_state,
    )
    final_git_state = _validate_git_state(
        snapshot,
        expected_head_commit=artifact_state.head_commit,
        implementation_commit=implementation_commit,
        config_contents=config_file.contents,
    )
    if (
        final_git_state.scorer_source_sha256 != git_state.scorer_source_sha256
        or len(final_git_state.files) != len(git_state.files)
        or any(
            observed.identity != expected.identity
            or not hmac.compare_digest(observed.sha256, expected.sha256)
            for observed, expected in zip(
                final_git_state.files, git_state.files, strict=True
            )
        )
    ):
        raise HoldoutAccessError("scorer source changed during authorization")

    stable_inputs = (
        (
            snapshot.repository_root / SELECTION_ARTIFACT_RELATIVE_PATH,
            artifact_file,
            "selection artifact",
            MAX_ARTIFACT_BYTES,
        ),
        (
            config_path,
            config_file,
            "experiment config",
            MAX_CONFIG_BYTES,
        ),
        (
            model_path,
            model_file,
            "portable model",
            MAX_SOURCE_BYTES,
        ),
        (
            evidence.dev_archive,
            dev_archive,
            "development archive",
            MAX_DEV_ARCHIVE_BYTES,
        ),
        (
            evidence.dev_manifest,
            dev_manifest,
            "development manifest",
            MAX_DEV_MANIFEST_BYTES,
        ),
        (
            evidence.dev_audit_report,
            dev_audit,
            "development audit report",
            MAX_AUDIT_REPORT_BYTES,
        ),
        (
            evidence.dev_replay_report,
            dev_replay,
            "development replay report",
            MAX_REPLAY_REPORT_BYTES,
        ),
    )
    for path, expected_file, name, maximum_bytes in stable_inputs:
        _require_file_unchanged(
            path,
            expected=expected_file,
            name=name,
            maximum_bytes=maximum_bytes,
        )
    for relative_path, expected_file in zip(
        SCORER_SOURCE_FILES, git_state.files, strict=True
    ):
        _require_file_unchanged(
            snapshot.repository_root / relative_path,
            expected=expected_file,
            name=f"scorer source {relative_path}",
            maximum_bytes=MAX_SOURCE_BYTES,
        )
    if current_runtime_identity() != runtime:
        raise HoldoutAccessError("runtime changed during authorization")
    _require_file_unchanged(
        snapshot.repository_root / ".git/index",
        expected=snapshot.index_file,
        name="Git index",
        maximum_bytes=MAX_GIT_INDEX_BYTES,
    )
    return _Authorization(
        threshold_milli=recomputed.threshold_milli,
        implementation_git_commit=implementation_commit,
        portable_model_json=model_file.contents,
    )


def authorize_and_open_test_clean(
    *,
    evidence: HoldoutEvidence,
    acquire: TestCleanSupplier,
) -> AuthorizedTestClean:
    """Validate development evidence, then acquire and open test-clean exactly once."""

    if not callable(acquire):
        raise HoldoutAccessError("test-clean supplier must be callable")
    with _isolated_git_snapshot(REPOSITORY_ROOT) as snapshot:
        authorization = _authorize_from_snapshot(snapshot, evidence=evidence)
        authorized_head = snapshot.head_commit
    if _current_git_head(REPOSITORY_ROOT) != authorized_head:
        raise HoldoutAccessError("Git HEAD changed after authorization")
    return _acquire_and_open_test_archive(
        acquire=acquire,
        authorization=authorization,
    )
