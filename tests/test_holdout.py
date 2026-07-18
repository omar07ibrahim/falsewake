from __future__ import annotations

import fcntl
import hashlib
import json
import mmap
import os
import struct
import subprocess
from dataclasses import dataclass, fields
from pathlib import Path
from typing import BinaryIO, cast

import pytest

from falsewake import holdout
from falsewake.holdout import (
    HoldoutAccessError,
    HoldoutEvidence,
    authorize_and_open_test_clean,
    canonical_runtime_identity,
    current_runtime_identity,
    recompute_selection,
)

SYNTHETIC_SOURCE_SAMPLES = 5_760_000
SYNTHETIC_TEST_ARCHIVE = b"synthetic test archive"


@dataclass(slots=True)
class FirewallCase:
    evidence: HoldoutEvidence
    repository_root: Path
    artifact: dict[str, object]
    report: dict[str, object]
    source_path: Path


def _sha256(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _md5(contents: bytes) -> str:
    return hashlib.md5(contents, usedforsecurity=False).hexdigest()


def _json_bytes(document: object) -> bytes:
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


def _manifest_bytes(*rows: object) -> bytes:
    return b"".join(
        json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
        for row in rows
    )


def _write_json(path: Path, document: object) -> None:
    path.write_bytes(_json_bytes(document))


def _git(repository: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", os.fspath(repository), *arguments],
        check=True,
        capture_output=True,
    )
    return completed.stdout


def _git_index_entries_end(contents: bytes) -> int:
    if contents[:4] != b"DIRC" or len(contents) < 32:
        raise AssertionError("synthetic fixture has no regular Git index")
    version, entry_count = struct.unpack(">II", contents[4:12])
    if version not in {2, 3}:
        raise AssertionError("synthetic fixture requires Git index v2 or v3")
    offset = 12
    for _ in range(entry_count):
        entry_start = offset
        offset = contents.index(b"\0", offset + 62) + 1
        while (offset - entry_start) % 8:
            offset += 1
    return offset


def _source_sha256(repository: Path) -> str:
    digest = hashlib.sha256(b"falsewake-exp001-scorer-source-v1\0")
    for relative_path in holdout.SCORER_SOURCE_FILES:
        path_bytes = relative_path.encode("utf-8")
        contents = (repository / relative_path).read_bytes()
        digest.update(struct.pack("<I", len(path_bytes)))
        digest.update(path_bytes)
        digest.update(struct.pack("<Q", len(contents)))
        digest.update(contents)
    return digest.hexdigest()


def _make_report(
    identities: dict[str, object],
    *,
    negative_frontier: int = 500,
    retention_frontier: int = 700,
) -> dict[str, object]:
    exposure = SYNTHETIC_SOURCE_SAMPLES
    event_counts = [
        1 if threshold < negative_frontier else 0 for threshold in range(1001)
    ]
    thresholds: list[dict[str, object]] = []
    for threshold, events in enumerate(event_counts):
        correct_by_target = (
            list(holdout.BASELINE_CORRECT_BY_TARGET)
            if threshold <= retention_frontier
            else [0] * len(holdout.TARGET_ORDER)
        )
        correct = sum(correct_by_target)
        thresholds.append(
            {
                "threshold_milli": threshold,
                "dev_event_count": events,
                "dev_event_count_by_target": [events]
                + [0] * (len(holdout.TARGET_ORDER) - 1),
                "false_events_per_hour": (
                    events * holdout.EXPOSURE_SAMPLES_PER_HOUR / exposure
                ),
                "garwood_95_percent": [0.0, float(events + 1)],
                "speaker_bootstrap_95_percent": [float(events), float(events)],
                "validation_correct_accept_count": correct,
                "validation_correct_accept_count_by_target": correct_by_target,
                "correct_accept_recall": correct / holdout.TARGET_EXAMPLE_COUNT,
                "conditional_correct_retention": (
                    correct / holdout.BASELINE_CORRECT_COUNT
                ),
            }
        )
    passing = negative_frontier <= retention_frontier
    selected = negative_frontier if passing else None
    evidence_thresholds = sorted(
        {
            0,
            negative_frontier,
            retention_frontier,
            *(() if selected is None else (selected,)),
        }
    )
    return {
        "schema_version": 1,
        "identities": identities,
        "runtime": current_runtime_identity(),
        "target_order": list(holdout.TARGET_ORDER),
        "dev": {
            "source_samples": exposure,
            "scored_exposure_samples": exposure,
            "utterance_count": 1,
            "speaker_count": 1,
            "speaker_rows": [
                {
                    "speaker_id": 1,
                    "utterance_count": 1,
                    "scored_exposure_samples": exposure,
                    "event_count": event_counts,
                }
            ],
        },
        "positive_validation": {
            "target_example_count": holdout.TARGET_EXAMPLE_COUNT,
            "baseline_correct_count": holdout.BASELINE_CORRECT_COUNT,
            "baseline_correct_by_target": list(holdout.BASELINE_CORRECT_BY_TARGET),
            "support_by_target": list(holdout.TARGET_SUPPORT),
        },
        "thresholds": thresholds,
        "selection": {
            "status": "pass" if passing else "reject",
            "selected_threshold_milli": selected,
            "negative_frontier_milli": negative_frontier,
            "retention_frontier_milli": retention_frontier,
        },
        "top_false_events": [
            {"threshold_milli": threshold, "events": []}
            for threshold in evidence_thresholds
        ],
    }


def _refresh_artifact(case: FirewallCase) -> None:
    artifact_path = (
        case.repository_root / holdout.SELECTION_ARTIFACT_RELATIVE_PATH
    )
    _write_json(artifact_path, case.artifact)
    _git(case.repository_root, "add", holdout.SELECTION_ARTIFACT_RELATIVE_PATH)
    _git(
        case.repository_root,
        "-c",
        "user.name=FalseWake tests",
        "-c",
        "user.email=falsewake@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "update synthetic selection artifact",
    )


def _refresh_report(case: FirewallCase) -> None:
    _write_json(case.evidence.dev_replay_report, case.report)
    case.artifact["dev_replay_report_sha256"] = _sha256(
        case.evidence.dev_replay_report.read_bytes()
    )
    _refresh_artifact(case)


@pytest.fixture
def firewall_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FirewallCase:
    repository = tmp_path / "repo"
    repository.mkdir()
    monkeypatch.setattr(holdout, "REPOSITORY_ROOT", repository)
    config = json.loads(Path("configs/experiment-001.json").read_text(encoding="utf-8"))
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    archive = evidence_root / "dev-clean.tar.gz"
    manifest = evidence_root / "dev-clean.manifest.jsonl"
    audit = evidence_root / "dev-clean.audit.json"
    replay = evidence_root / "experiment-001-dev-replay.json"
    artifact_path = repository / holdout.SELECTION_ARTIFACT_RELATIVE_PATH
    archive.write_bytes(b"synthetic development archive")
    manifest.write_bytes(
        _manifest_bytes(
            {
                "chapter_id": "2",
                "decoded_pcm16le_sha256": "1" * 64,
                "raw_flac_sha256": "2" * 64,
                "relative_path": ("LibriSpeech/dev-clean/1/2/1-2-0003.flac"),
                "sample_count": SYNTHETIC_SOURCE_SAMPLES,
                "speaker_id": "1",
                "transcript": "SYNTHETIC DEVELOPMENT UTTERANCE",
                "utterance_id": "1-2-0003",
            }
        )
    )
    archive_sha256 = _sha256(archive.read_bytes())
    manifest_sha256 = _sha256(manifest.read_bytes())
    audit_document = {
        "archive_sha256": archive_sha256,
        "flac_count": 1,
        "manifest_sha256": manifest_sha256,
        "source_samples": SYNTHETIC_SOURCE_SAMPLES,
        "speaker_count": 1,
        "utterance_count": 1,
    }
    _write_json(audit, audit_document)
    audit_sha256 = _sha256(audit.read_bytes())
    config["negative_source"]["archive_sha256"] = archive_sha256
    test_identity = config["negative_source"]["test_clean_official_identity"]
    test_identity["archive_bytes"] = len(SYNTHETIC_TEST_ARCHIVE)
    test_identity["archive_md5"] = _md5(SYNTHETIC_TEST_ARCHIVE)
    test_identity["archive_sha256"] = _sha256(SYNTHETIC_TEST_ARCHIVE)
    verified_output = config["negative_source"]["archive_audit"][
        "verified_dev_clean_output"
    ]
    verified_output.update(
        {
            "audit_report_sha256": audit_sha256,
            "manifest_sha256": manifest_sha256,
            "scored_exposure_samples": SYNTHETIC_SOURCE_SAMPLES,
            "source_samples": SYNTHETIC_SOURCE_SAMPLES,
            "speaker_count": 1,
            "utterance_count": 1,
        }
    )
    monkeypatch.setattr(holdout, "DEV_AUDIT_REPORT_SHA256", audit_sha256)
    monkeypatch.setattr(holdout, "DEV_MANIFEST_SHA256", manifest_sha256)
    monkeypatch.setattr(
        holdout, "DEV_SCORED_EXPOSURE_SAMPLES", SYNTHETIC_SOURCE_SAMPLES
    )
    monkeypatch.setattr(holdout, "DEV_SOURCE_SAMPLES", SYNTHETIC_SOURCE_SAMPLES)
    monkeypatch.setattr(holdout, "DEV_SPEAKER_COUNT", 1)
    monkeypatch.setattr(holdout, "DEV_UTTERANCE_COUNT", 1)
    monkeypatch.setattr(holdout, "TEST_ARCHIVE_BYTES", len(SYNTHETIC_TEST_ARCHIVE))
    monkeypatch.setattr(holdout, "TEST_ARCHIVE_MD5", _md5(SYNTHETIC_TEST_ARCHIVE))
    monkeypatch.setattr(holdout, "TEST_ARCHIVE_SHA256", _sha256(SYNTHETIC_TEST_ARCHIVE))

    config_path = repository / "configs/experiment-001.json"
    config_path.parent.mkdir(parents=True)
    _write_json(config_path, config)
    monkeypatch.setattr(
        holdout, "EXPERIMENT_CONFIG_SHA256", _sha256(config_path.read_bytes())
    )
    for relative_path in holdout.SCORER_SOURCE_FILES:
        source = repository / relative_path
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(Path(relative_path).read_bytes())
    model_path = repository / holdout.PORTABLE_MODEL_RELATIVE_PATH
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(Path(holdout.PORTABLE_MODEL_RELATIVE_PATH).read_bytes())
    _git(repository, "init", "--quiet")
    _git(repository, "add", "configs", "models", "src")
    _git(
        repository,
        "-c",
        "user.name=FalseWake tests",
        "-c",
        "user.email=falsewake@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "synthetic implementation",
    )
    implementation_commit = _git(repository, "rev-parse", "HEAD").decode().strip()
    config_sha256 = _sha256(config_path.read_bytes())
    runtime_sha256 = _sha256(canonical_runtime_identity(current_runtime_identity()))
    identities: dict[str, object] = {
        "experiment_config_sha256": config_sha256,
        "portable_model_sha256": config["model"]["portable_json_sha256"],
        "scorer_source_sha256": _source_sha256(repository),
        "runtime_identity_sha256": runtime_sha256,
        "implementation_git_commit": implementation_commit,
        "dev_archive_sha256": archive_sha256,
        "dev_manifest_sha256": manifest_sha256,
        "dev_audit_report_sha256": audit_sha256,
        "positive_examples_sha256": config["positive_validation"]["examples_sha256"],
        "positive_feature_matrix_semantic_sha256": config["positive_validation"][
            "feature_cache"
        ]["features_sha256"],
    }
    report = _make_report(identities)
    _write_json(replay, report)
    artifact: dict[str, object] = {
        "schema_version": 1,
        "status": "pass",
        "experiment_config_sha256": config_sha256,
        "scorer_source_sha256": identities["scorer_source_sha256"],
        "runtime_identity_sha256": runtime_sha256,
        "implementation_git_commit": implementation_commit,
        "dev_archive_sha256": archive_sha256,
        "dev_manifest_sha256": manifest_sha256,
        "dev_audit_report_sha256": audit_sha256,
        "dev_replay_report_sha256": _sha256(replay.read_bytes()),
        "selected_threshold_milli": 500,
    }
    artifact_path.parent.mkdir(parents=True)
    _write_json(artifact_path, artifact)
    _git(repository, "add", holdout.SELECTION_ARTIFACT_RELATIVE_PATH)
    _git(
        repository,
        "-c",
        "user.name=FalseWake tests",
        "-c",
        "user.email=falsewake@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "record synthetic selection artifact",
    )
    evidence = HoldoutEvidence(
        dev_archive=archive,
        dev_manifest=manifest,
        dev_audit_report=audit,
        dev_replay_report=replay,
    )
    return FirewallCase(
        evidence=evidence,
        repository_root=repository,
        artifact=artifact,
        report=report,
        source_path=repository / holdout.SCORER_SOURCE_FILES[0],
    )


def _assert_rejected_without_supplier(case: FirewallCase, match: str) -> None:
    supplier_calls: list[BinaryIO] = []

    def supplier(destination: BinaryIO) -> None:
        supplier_calls.append(destination)
        destination.write(SYNTHETIC_TEST_ARCHIVE)

    with pytest.raises(HoldoutAccessError, match=match):
        authorize_and_open_test_clean(
            evidence=case.evidence,
            acquire=supplier,
        )
    assert supplier_calls == []


def test_valid_pass_opens_test_only_after_every_identity_matches(
    firewall_case: FirewallCase,
) -> None:
    def supplier(destination: BinaryIO) -> None:
        destination.write(SYNTHETIC_TEST_ARCHIVE)

    with authorize_and_open_test_clean(
        evidence=firewall_case.evidence,
        acquire=supplier,
    ) as authorized:
        assert authorized.selected_threshold_milli == 500
        assert (
            authorized.implementation_git_commit
            == firewall_case.artifact["implementation_git_commit"]
        )
        assert (
            authorized.portable_model_json
            == Path(holdout.PORTABLE_MODEL_RELATIVE_PATH).read_bytes()
        )
        assert authorized.archive_sha256 == _sha256(SYNTHETIC_TEST_ARCHIVE)
        assert os.fstat(authorized.stream.fileno()).st_nlink == 0
        required_seals = (
            fcntl.F_SEAL_WRITE
            | fcntl.F_SEAL_GROW
            | fcntl.F_SEAL_SHRINK
            | fcntl.F_SEAL_SEAL
        )
        assert fcntl.fcntl(authorized.stream.fileno(), fcntl.F_GET_SEALS) == (
            required_seals
        )
        assert authorized.stream.read() == SYNTHETIC_TEST_ARCHIVE


def test_retained_supplier_descriptor_cannot_mutate_the_sealed_archive(
    firewall_case: FirewallCase,
) -> None:
    retained_descriptor = -1

    def retaining_supplier(destination: BinaryIO) -> None:
        nonlocal retained_descriptor
        destination.write(SYNTHETIC_TEST_ARCHIVE)
        retained_descriptor = os.dup(destination.fileno())
        os.lseek(retained_descriptor, 0, os.SEEK_END)
        initial_flags = fcntl.fcntl(retained_descriptor, fcntl.F_GETFL)
        fcntl.fcntl(
            retained_descriptor,
            fcntl.F_SETFL,
            initial_flags | os.O_APPEND,
        )

    try:
        with authorize_and_open_test_clean(
            evidence=firewall_case.evidence,
            acquire=retaining_supplier,
        ) as authorized:
            os.lseek(retained_descriptor, 0, os.SEEK_END)
            retained_flags = fcntl.fcntl(retained_descriptor, fcntl.F_GETFL)
            fcntl.fcntl(
                retained_descriptor,
                fcntl.F_SETFL,
                retained_flags | os.O_APPEND,
            )
            with pytest.raises(OSError):
                os.pwrite(retained_descriptor, b"TAMPER", 0)
            assert authorized.stream.tell() == 0
            assert not (
                fcntl.fcntl(authorized.stream.fileno(), fcntl.F_GETFL) & os.O_APPEND
            )
            assert authorized.stream.read() == SYNTHETIC_TEST_ARCHIVE
    finally:
        if retained_descriptor >= 0:
            os.close(retained_descriptor)


def test_writable_mapping_prevents_authorization_instead_of_escaping_seals(
    firewall_case: FirewallCase,
) -> None:
    retained_mapping: mmap.mmap | None = None

    def mapping_supplier(destination: BinaryIO) -> None:
        nonlocal retained_mapping
        destination.write(SYNTHETIC_TEST_ARCHIVE)
        retained_mapping = mmap.mmap(
            destination.fileno(),
            len(SYNTHETIC_TEST_ARCHIVE),
            access=mmap.ACCESS_WRITE,
        )

    try:
        with pytest.raises(HoldoutAccessError, match="acquire and seal test-clean"):
            authorize_and_open_test_clean(
                evidence=firewall_case.evidence,
                acquire=mapping_supplier,
            )
    finally:
        if retained_mapping is not None:
            retained_mapping.close()


def test_supplier_failure_closes_private_staging_without_publishing_a_path(
    firewall_case: FirewallCase,
) -> None:
    def failing_supplier(destination: BinaryIO) -> None:
        destination.write(b"partial held-out bytes")
        raise RuntimeError("synthetic transfer failure")

    with pytest.raises(HoldoutAccessError, match="synthetic transfer failure"):
        authorize_and_open_test_clean(
            evidence=firewall_case.evidence,
            acquire=failing_supplier,
        )


def test_wrong_official_archive_bytes_are_rejected_after_sealing(
    firewall_case: FirewallCase,
) -> None:
    def wrong_supplier(destination: BinaryIO) -> None:
        destination.write(b"x" * len(SYNTHETIC_TEST_ARCHIVE))

    with pytest.raises(HoldoutAccessError, match="official MD5 differs"):
        authorize_and_open_test_clean(
            evidence=firewall_case.evidence,
            acquire=wrong_supplier,
        )


def test_git_fsmonitor_cannot_execute_bytes_acquired_after_the_gate(
    firewall_case: FirewallCase,
    tmp_path: Path,
) -> None:
    marker = tmp_path / "git-executed-helper"
    helper = tmp_path / "malicious-fsmonitor"
    helper.write_text(
        f"#!/bin/sh\ntouch {marker}\nprintf '{{}}\\n'\n",
        encoding="utf-8",
    )
    helper.chmod(0o700)
    _git(
        firewall_case.repository_root,
        "config",
        "core.fsmonitor",
        os.fspath(helper),
    )
    _git(firewall_case.repository_root, "status", "--short")
    assert marker.exists()
    marker.unlink()

    def supplier(destination: BinaryIO) -> None:
        destination.write(SYNTHETIC_TEST_ARCHIVE)

    with authorize_and_open_test_clean(
        evidence=firewall_case.evidence,
        acquire=supplier,
    ) as authorized:
        assert authorized.stream.read() == SYNTHETIC_TEST_ARCHIVE
    assert not marker.exists()


def test_git_clean_filter_cannot_run_during_authorization(
    firewall_case: FirewallCase,
    tmp_path: Path,
) -> None:
    marker = tmp_path / "git-executed-filter"
    helper = tmp_path / "malicious-clean-filter"
    helper.write_text(
        f"#!/bin/sh\ntouch {marker}\ncat\n",
        encoding="utf-8",
    )
    helper.chmod(0o700)
    relative_source = firewall_case.source_path.relative_to(
        firewall_case.repository_root
    )
    attributes = firewall_case.repository_root / ".git/info/attributes"
    attributes.write_text(f"{relative_source} filter=evil\n", encoding="utf-8")
    _git(
        firewall_case.repository_root,
        "config",
        "filter.evil.clean",
        os.fspath(helper),
    )
    os.utime(firewall_case.source_path)
    _git(firewall_case.repository_root, "status", "--short")
    assert marker.exists()
    marker.unlink()

    def supplier(destination: BinaryIO) -> None:
        destination.write(SYNTHETIC_TEST_ARCHIVE)

    with authorize_and_open_test_clean(
        evidence=firewall_case.evidence,
        acquire=supplier,
    ) as authorized:
        assert authorized.stream.read() == SYNTHETIC_TEST_ARCHIVE
    assert not marker.exists()


def test_missing_tracked_selection_artifact_fails_closed_before_supplier(
    firewall_case: FirewallCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_path = (
        firewall_case.repository_root
        / holdout.SELECTION_ARTIFACT_RELATIVE_PATH
    )
    artifact_path.unlink()
    _assert_rejected_without_supplier(
        firewall_case, "cannot resolve selection artifact"
    )


def test_symlinked_selection_artifact_fails_closed_before_supplier(
    firewall_case: FirewallCase,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact_path = (
        firewall_case.repository_root
        / holdout.SELECTION_ARTIFACT_RELATIVE_PATH
    )
    target = tmp_path / "selection-target.json"
    target.write_bytes(artifact_path.read_bytes())
    artifact_path.unlink()
    artifact_path.symlink_to(target)
    _assert_rejected_without_supplier(firewall_case, "path must be fully physical")


def test_head_switch_during_artifact_validation_fails_closed(
    firewall_case: FirewallCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_current_head = holdout._current_git_head
    calls = 0

    def switched_head(repository_root: Path) -> str:
        nonlocal calls
        calls += 1
        if calls == 2:
            return "0" * 40
        return real_current_head(repository_root)

    monkeypatch.setattr(holdout, "_current_git_head", switched_head)
    _assert_rejected_without_supplier(firewall_case, "HEAD changed")


def test_scorer_mutation_after_artifact_revalidation_blocks_acquisition(
    firewall_case: FirewallCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_revalidate = holdout._revalidate_selection_artifact_commit

    def mutate_scorer_after_artifact(
        snapshot: holdout._GitSnapshot,
        expected: holdout._SelectionArtifactState,
    ) -> None:
        real_revalidate(snapshot, expected)
        firewall_case.source_path.write_bytes(b"# raced scorer source\n")

    monkeypatch.setattr(
        holdout,
        "_revalidate_selection_artifact_commit",
        mutate_scorer_after_artifact,
    )
    _assert_rejected_without_supplier(
        firewall_case, "differs from the implementation commit"
    )


def test_registered_config_identity_rejects_self_consistent_protocol_drift() -> None:
    config = json.loads(Path("configs/experiment-001.json").read_text(encoding="utf-8"))
    config["scoring"]["refractory_samples"] = 1

    with pytest.raises(HoldoutAccessError, match="identity is not experiment 001"):
        holdout._validate_registered_config(_json_bytes(config))


def test_repository_config_path_is_not_caller_substitutable() -> None:
    evidence_fields = {field.name for field in fields(HoldoutEvidence)}
    assert "experiment_config" not in evidence_fields
    assert "repository_root" not in evidence_fields


def test_inherited_git_environment_cannot_redirect_validation(
    firewall_case: FirewallCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GIT_DIR", "/does/not/exist")
    monkeypatch.setenv("GIT_WORK_TREE", "/does/not/exist")
    monkeypatch.setenv("GIT_INDEX_FILE", "/does/not/exist")
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", "/does/not/exist")

    def supplier(destination: BinaryIO) -> None:
        destination.write(SYNTHETIC_TEST_ARCHIVE)

    with authorize_and_open_test_clean(
        evidence=firewall_case.evidence,
        acquire=supplier,
    ) as authorized:
        assert authorized.stream.read() == SYNTHETIC_TEST_ARCHIVE


def test_symlinked_repository_root_never_opens_test(
    firewall_case: FirewallCase,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    linked_root = tmp_path / "repo-link"
    linked_root.symlink_to(
        firewall_case.repository_root, target_is_directory=True
    )
    monkeypatch.setattr(holdout, "REPOSITORY_ROOT", linked_root)
    _assert_rejected_without_supplier(firewall_case, "physical directory")


@pytest.mark.parametrize(
    ("field", "match"),
    [
        ("experiment_config_sha256", "config SHA-256 differs"),
        ("scorer_source_sha256", "scorer source SHA-256 differs"),
        ("runtime_identity_sha256", "runtime identity SHA-256 differs"),
        ("dev_archive_sha256", "archive SHA-256 differs"),
        ("dev_manifest_sha256", "manifest SHA-256 differs"),
        ("dev_audit_report_sha256", "audit report SHA-256 differs"),
        ("dev_replay_report_sha256", "replay report SHA-256 differs"),
    ],
)
def test_artifact_identity_rejections_never_open_test(
    firewall_case: FirewallCase,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    match: str,
) -> None:
    firewall_case.artifact[field] = "0" * 64
    _refresh_artifact(firewall_case)
    _assert_rejected_without_supplier(firewall_case, match)


def test_wrong_artifact_identity_never_opens_test(
    firewall_case: FirewallCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_path = (
        firewall_case.repository_root
        / holdout.SELECTION_ARTIFACT_RELATIVE_PATH
    )
    firewall_case.artifact["selected_threshold_milli"] = 501
    _write_json(artifact_path, firewall_case.artifact)
    _assert_rejected_without_supplier(firewall_case, "differs from its HEAD blob")


def test_noncanonical_tracked_artifact_never_opens_test(
    firewall_case: FirewallCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_path = (
        firewall_case.repository_root
        / holdout.SELECTION_ARTIFACT_RELATIVE_PATH
    )
    artifact_path.write_bytes(
        json.dumps(firewall_case.artifact, sort_keys=True).encode("ascii")
    )
    _git(
        firewall_case.repository_root,
        "add",
        holdout.SELECTION_ARTIFACT_RELATIVE_PATH,
    )
    _git(
        firewall_case.repository_root,
        "-c",
        "user.name=FalseWake tests",
        "-c",
        "user.email=falsewake@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "record malformed selection artifact",
    )
    _assert_rejected_without_supplier(firewall_case, "artifact serialization differs")


def test_boolean_artifact_schema_version_never_opens_test(
    firewall_case: FirewallCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    firewall_case.artifact["schema_version"] = True
    _refresh_artifact(firewall_case)
    _assert_rejected_without_supplier(
        firewall_case, "schema_version must be integer one"
    )


def test_artifact_must_be_absent_from_the_implementation_commit(
    firewall_case: FirewallCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_commit = (
        _git(firewall_case.repository_root, "rev-parse", "HEAD")
        .decode()
        .strip()
    )
    firewall_case.artifact["implementation_git_commit"] = artifact_commit
    _refresh_artifact(firewall_case)
    _assert_rejected_without_supplier(
        firewall_case, "must be absent from the implementation commit"
    )


def test_nonancestor_implementation_commit_never_opens_test(
    firewall_case: FirewallCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = firewall_case.repository_root
    tree = _git(repository, "rev-parse", "HEAD^{tree}").decode().strip()
    side_commit = (
        _git(
            repository,
            "-c",
            "user.name=FalseWake tests",
            "-c",
            "user.email=falsewake@example.invalid",
            "commit-tree",
            tree,
            "-m",
            "synthetic nonancestor implementation",
        )
        .decode()
        .strip()
    )
    firewall_case.artifact["implementation_git_commit"] = side_commit
    _refresh_artifact(firewall_case)
    _assert_rejected_without_supplier(firewall_case, "implementation commit ancestry")


def test_wrong_selected_threshold_never_opens_test(
    firewall_case: FirewallCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    firewall_case.artifact["selected_threshold_milli"] = 501
    _refresh_artifact(firewall_case)
    _assert_rejected_without_supplier(firewall_case, "selected threshold differs")


def test_boolean_replay_selection_never_equals_an_integer_threshold(
    firewall_case: FirewallCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    identities = cast(dict[str, object], firewall_case.report["identities"])
    firewall_case.report = _make_report(identities, negative_frontier=1)
    selection = cast(dict[str, object], firewall_case.report["selection"])
    selection["selected_threshold_milli"] = True
    selection["negative_frontier_milli"] = True
    firewall_case.artifact["selected_threshold_milli"] = 1
    _refresh_report(firewall_case)

    _assert_rejected_without_supplier(firewall_case, "must be a JSON integer")


def test_handwritten_replay_identity_never_opens_test(
    firewall_case: FirewallCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    identities = cast(dict[str, object], firewall_case.report["identities"])
    identities["positive_examples_sha256"] = "0" * 64
    _refresh_report(firewall_case)
    _assert_rejected_without_supplier(firewall_case, "replay identities differ")


def test_payload_audit_tampering_is_rejected_by_the_registered_digest(
    firewall_case: FirewallCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit_path = firewall_case.evidence.dev_audit_report
    audit = cast(dict[str, object], json.loads(audit_path.read_text(encoding="utf-8")))
    audit["source_samples"] = cast(int, audit["source_samples"]) + 1
    _write_json(audit_path, audit)
    audit_sha256 = _sha256(audit_path.read_bytes())
    firewall_case.artifact["dev_audit_report_sha256"] = audit_sha256
    identities = cast(dict[str, object], firewall_case.report["identities"])
    identities["dev_audit_report_sha256"] = audit_sha256
    _refresh_report(firewall_case)

    _assert_rejected_without_supplier(
        firewall_case, "registered development audit report"
    )


def test_replay_exposure_must_be_derived_from_the_manifest(
    firewall_case: FirewallCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    dev = cast(dict[str, object], firewall_case.report["dev"])
    exposure = cast(int, dev["scored_exposure_samples"]) - 1_600
    dev["scored_exposure_samples"] = exposure
    speaker = cast(list[dict[str, object]], dev["speaker_rows"])[0]
    speaker["scored_exposure_samples"] = exposure
    for row in cast(list[dict[str, object]], firewall_case.report["thresholds"]):
        events = cast(int, row["dev_event_count"])
        row["false_events_per_hour"] = (
            events * holdout.EXPOSURE_SAMPLES_PER_HOUR / exposure
        )
    _refresh_report(firewall_case)

    _assert_rejected_without_supplier(
        firewall_case, "exposure differs from the manifest"
    )


def test_valid_reject_artifact_never_opens_test(
    firewall_case: FirewallCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    identities = cast(dict[str, object], firewall_case.report["identities"])
    firewall_case.report = _make_report(
        identities, negative_frontier=701, retention_frontier=700
    )
    firewall_case.artifact["status"] = "reject"
    firewall_case.artifact["selected_threshold_milli"] = None
    _refresh_report(firewall_case)
    _assert_rejected_without_supplier(
        firewall_case, "selection rejected test-clean access"
    )


def test_dirty_scorer_source_never_opens_test(
    firewall_case: FirewallCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    firewall_case.source_path.write_bytes(b"# dirty source\n")
    _assert_rejected_without_supplier(
        firewall_case, "differs from the implementation commit"
    )


def test_staged_scorer_drift_is_rejected_even_with_clean_working_bytes(
    firewall_case: FirewallCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = firewall_case.source_path.read_bytes()
    firewall_case.source_path.write_bytes(b"# staged malicious scorer\n")
    relative_path = firewall_case.source_path.relative_to(
        firewall_case.repository_root
    )
    _git(firewall_case.repository_root, "add", os.fspath(relative_path))
    firewall_case.source_path.write_bytes(original)
    assert _git(
        firewall_case.repository_root,
        "status",
        "--short",
        os.fspath(relative_path),
    ).startswith(b"MM ")

    _assert_rejected_without_supplier(
        firewall_case,
        "Git index differs from the artifact HEAD",
    )


def test_forged_clean_cache_tree_cannot_hide_a_malicious_index_entry(
    firewall_case: FirewallCase,
) -> None:
    repository = firewall_case.repository_root
    index_path = repository / ".git/index"
    clean_index = index_path.read_bytes()
    clean_entries_end = _git_index_entries_end(clean_index)
    assert b"TREE" in clean_index[clean_entries_end:-20]

    original = firewall_case.source_path.read_bytes()
    firewall_case.source_path.write_bytes(b"# cache-tree-hidden scorer\n")
    relative_path = firewall_case.source_path.relative_to(repository)
    _git(repository, "add", os.fspath(relative_path))
    firewall_case.source_path.write_bytes(original)
    dirty_index = index_path.read_bytes()
    dirty_entries_end = _git_index_entries_end(dirty_index)

    forged_body = dirty_index[:dirty_entries_end] + clean_index[clean_entries_end:-20]
    forged_checksum = hashlib.sha1(
        forged_body, usedforsecurity=False
    ).digest()
    index_path.write_bytes(forged_body + forged_checksum)

    staged = _git(repository, "ls-files", "--stage", "--", os.fspath(relative_path))
    head_blob = _git(repository, "rev-parse", f"HEAD:{relative_path}").strip()
    assert head_blob not in staged
    assert _git(repository, "write-tree").strip() == _git(
        repository, "rev-parse", "HEAD^{tree}"
    ).strip()
    _assert_rejected_without_supplier(
        firewall_case,
        "Git index differs from the artifact HEAD",
    )


def test_dirty_experiment_config_never_opens_test(
    firewall_case: FirewallCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = (
        firewall_case.repository_root / holdout.EXPERIMENT_CONFIG_RELATIVE_PATH
    )
    with config_path.open("ab") as stream:
        stream.write(b" \n")
    _assert_rejected_without_supplier(
        firewall_case, "config identity is not experiment 001"
    )


def test_repository_config_cannot_be_substituted_through_a_parent_symlink(
    firewall_case: FirewallCase,
    tmp_path: Path,
) -> None:
    config_parent = (
        firewall_case.repository_root
        / holdout.EXPERIMENT_CONFIG_RELATIVE_PATH
    ).parent
    external_parent = tmp_path / "external-configs"
    config_parent.rename(external_parent)
    config_parent.symlink_to(external_parent, target_is_directory=True)

    _assert_rejected_without_supplier(
        firewall_case,
        "experiment config path must be fully physical",
    )


def test_dirty_portable_model_never_reaches_the_supplier(
    firewall_case: FirewallCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = (
        firewall_case.repository_root / holdout.PORTABLE_MODEL_RELATIVE_PATH
    )
    model_path.write_bytes(b'{"tampered":true}\n')

    _assert_rejected_without_supplier(
        firewall_case,
        "portable model SHA-256 differs",
    )


def test_symlinked_development_evidence_never_opens_test(
    firewall_case: FirewallCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = firewall_case.evidence.dev_manifest
    target = manifest.with_name("manifest-target.jsonl")
    target.write_bytes(manifest.read_bytes())
    manifest.unlink()
    manifest.symlink_to(target)
    _assert_rejected_without_supplier(firewall_case, "path must be fully physical")


def test_unknown_implementation_commit_never_opens_test(
    firewall_case: FirewallCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    firewall_case.artifact["implementation_git_commit"] = "0" * 40
    _refresh_artifact(firewall_case)
    _assert_rejected_without_supplier(firewall_case, "implementation object")


def test_duplicate_artifact_key_never_opens_test(
    firewall_case: FirewallCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_path = (
        firewall_case.repository_root
        / holdout.SELECTION_ARTIFACT_RELATIVE_PATH
    )
    malformed = artifact_path.read_bytes().replace(
        b'{\n  "dev_archive_sha256"',
        b'{\n  "schema_version": 1,\n  "dev_archive_sha256"',
        1,
    )
    artifact_path.write_bytes(malformed)
    _assert_rejected_without_supplier(firewall_case, "repeats key")


def test_non_string_artifact_status_never_opens_test(
    firewall_case: FirewallCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    firewall_case.artifact["status"] = []
    _refresh_artifact(firewall_case)
    _assert_rejected_without_supplier(firewall_case, "artifact status is invalid")


def test_nonfinite_replay_number_never_opens_test(
    firewall_case: FirewallCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    malformed = firewall_case.evidence.dev_replay_report.read_bytes().replace(
        b'"false_events_per_hour": 10.0',
        b'"false_events_per_hour": 1e999',
        1,
    )
    firewall_case.evidence.dev_replay_report.write_bytes(malformed)
    firewall_case.artifact["dev_replay_report_sha256"] = _sha256(malformed)
    _refresh_artifact(firewall_case)
    _assert_rejected_without_supplier(firewall_case, "non-finite")


def test_recompute_selection_rejects_noncanonical_and_noninteger_grid(
    firewall_case: FirewallCase,
) -> None:
    compact = json.dumps(firewall_case.report, sort_keys=True).encode()
    with pytest.raises(HoldoutAccessError, match="serialization differs"):
        recompute_selection(compact)

    first = cast(list[dict[str, object]], firewall_case.report["thresholds"])[0]
    first["threshold_milli"] = 0.0
    with pytest.raises(HoldoutAccessError, match="must be a JSON integer"):
        recompute_selection(_json_bytes(firewall_case.report))
