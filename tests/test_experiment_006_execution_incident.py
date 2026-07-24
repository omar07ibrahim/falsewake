from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
INCIDENT_PATH = ROOT / "reports" / "experiment-006-execution-incident.json"
REGISTRATION_PATH = ROOT / "configs" / "experiment-006-run.json"

ADMISSION_COMMIT = "438651010c4ef1de4012a570f3211b1d27bb1e3a"
REGISTRATION_COMMIT = "b73ecb8be3871092aad431a6c497771296c67d60"
INCIDENT_COMMIT = "fd6b98e6122d2884800b359a61440e9b07832204"
INCIDENT_SHA256 = "c5cf002ea876890e3b3769af2bcc850433d3d08ac1071c864ed69ee2172af77d"
INCIDENT_GIT_BLOB = "44124d5bcf45a716fc84592b8781c84f04d27147"
INCIDENT_BYTES = 8_839
REGISTRATION_SHA256 = "53101c6424db09da01e23af4dc5956a57aa8a7e41226113782d4da1d822ab0b8"

MARKER_PATH = Path("/home/ubuntu/gitcode/.t/falsewake-experiment-006-attempt")
MARKER_BYTES = b"falsewake-experiment-006-attempt-v1\n"
MARKER_SHA256 = "11cb9ce51db5b5153d3839ac1edad1cd63ae582009417abfea6a8562e319bfbe"

MANAGED_SCIENTIFIC_OUTPUTS = (
    ROOT / "models" / "experiment-006-selected.safetensors",
    ROOT / "reports" / "experiment-006-seed-20260719-history.json",
    ROOT / "reports" / "experiment-006-seed-20260720-history.json",
    ROOT / "reports" / "experiment-006-seed-20260721-history.json",
    ROOT / "reports" / "experiment-006-selected-rerun-history.json",
    ROOT / "reports" / "experiment-006-training.json",
)


def _reject_duplicate_key(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"duplicate JSON key: {key}")
        document[key] = value
    return document


def _reject_float(value: str) -> None:
    raise ValueError(f"JSON floats are forbidden: {value}")


def _strict_json(payload: bytes) -> dict[str, Any]:
    assert not payload.startswith(b"\xef\xbb\xbf")
    document = json.loads(
        payload.decode("ascii"),
        object_pairs_hook=_reject_duplicate_key,
        parse_float=_reject_float,
        parse_constant=_reject_float,
    )
    assert type(document) is dict
    return document


def _canonical(document: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            document,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git(*arguments: str) -> bytes:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        timeout=30,
    )
    return completed.stdout


def _incident() -> dict[str, Any]:
    return _strict_json(INCIDENT_PATH.read_bytes())


def _stat_frame(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _marker_snapshot() -> tuple[tuple[int, ...], bytes] | None:
    try:
        named_before = MARKER_PATH.lstat()
    except FileNotFoundError:
        assert not os.path.lexists(MARKER_PATH)
        return None

    assert stat.S_ISREG(named_before.st_mode)
    descriptor = os.open(
        MARKER_PATH,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1 << 16):
            chunks.append(chunk)
        closed = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    frame = _stat_frame(named_before)
    assert _stat_frame(opened) == frame
    assert _stat_frame(closed) == frame
    assert _stat_frame(MARKER_PATH.lstat()) == frame
    return frame, b"".join(chunks)


def test_incident_is_exact_canonical_ascii_evidence() -> None:
    payload = INCIDENT_PATH.read_bytes()
    incident = _strict_json(payload)

    assert len(payload) == INCIDENT_BYTES
    assert _sha256(payload) == INCIDENT_SHA256
    assert _canonical(incident) == payload
    assert tuple(incident) == (
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
    )
    assert incident["schema_version"] == 1
    assert incident["experiment"] == "006"
    assert incident["scientific_protocol"] == "002"
    assert incident["kind"] == "registered_execution_incident"
    assert incident["incident"] == "registered_terminal_authority_failure"


def test_attempt_and_registration_bind_the_single_consumed_execution() -> None:
    incident = _incident()
    attempt = incident["attempt"]

    assert attempt == {
        "canonical_marker": str(MARKER_PATH),
        "exit_code": 1,
        "marker": {
            "byte_count": len(MARKER_BYTES),
            "bytes_ascii": MARKER_BYTES.decode("ascii"),
            "mode": "0444",
            "mtime_utc": "2026-07-24T01:48:08.576998118Z",
            "path": str(MARKER_PATH),
            "sha256": MARKER_SHA256,
        },
        "registered_attempt_consumed": True,
        "registered_authority_issuer_invoked": True,
        "registered_coordinator_invoked": True,
        "registered_invocation_count": 1,
        "registered_runner_invoked": True,
        "retry_allowed": False,
        "terminal": True,
    }

    registration_payload = REGISTRATION_PATH.read_bytes()
    assert len(registration_payload) == 14_811
    assert _sha256(registration_payload) == REGISTRATION_SHA256
    assert incident["registration"] == {
        "byte_count": 14_811,
        "commit": REGISTRATION_COMMIT,
        "git_blob": "02e087bbbbeeab87904a3ef5c5fd9bd7acc2985b",
        "implementation_commit": ADMISSION_COMMIT,
        "mode": "100644",
        "parent": ADMISSION_COMMIT,
        "path": "configs/experiment-006-run.json",
        "runtime_fingerprint_sha256": (
            "da1b542e1b6fe8adb6ba56d7060e35e06e39c85dacdf3c35474e404126bff5ea"
        ),
        "sha256": REGISTRATION_SHA256,
        "source_bundle_sha256": (
            "8150d34fa0db80efd073e78c740d0291051dadbafac3eda81cc3534d7c604e95"
        ),
    }


def test_terminal_outcome_preserves_epistemic_limits() -> None:
    incident = _incident()

    assert incident["outcome"] == {
        "automatic_terminal_report_published": False,
        "checkpoint_reusable": False,
        "code": "final_registration_capability_rejected_after_prior_state_change",
        "phase": "registered_final_authority_verification",
        "published_artifacts": [],
        "scientific_result_available": False,
        "status": "execution_failure",
        "terminal": True,
    }
    assert incident["managed_state_after_exit"] == {
        "canonical_managed_outputs": {
            "configs/experiment-006-run.json": "present",
            "models/experiment-006-selected.safetensors": "absent",
            "reports/experiment-006-seed-20260719-history.json": "absent",
            "reports/experiment-006-seed-20260720-history.json": "absent",
            "reports/experiment-006-seed-20260721-history.json": "absent",
            "reports/experiment-006-selected-rerun-history.json": "absent",
            "reports/experiment-006-training.json": "absent",
        },
        "caveat": (
            "The coordinator performs cleanup before its final authority check, so "
            "post-exit absence does not establish how far the hidden primary "
            "execution progressed."
        ),
        "optimizer_updates": "not_established_from_retained_evidence",
        "publication_temporaries": "absent",
        "repository_worktree": "clean",
        "scratch_root": "absent",
        "seed_children_started": "not_established_from_retained_evidence",
        "staging_root": "absent",
        "training_processes_started": "not_established_from_retained_evidence",
        "validation_examples": "not_established_from_retained_evidence",
    }
    assert incident["record_provenance"]["automatic_record"] is False
    assert incident["record_provenance"]["raw_stderr_bytes_retained"] is False
    assert incident["record_provenance"]["raw_stderr_sha256"] == "unavailable"


def test_trace_and_diagnosis_do_not_overstate_the_root_cause() -> None:
    incident = _incident()
    runtime = incident["runtime_observation"]
    diagnosis = incident["diagnosis"]

    assert runtime["final_registration_frame_verification_reached"] is True
    assert runtime["initial_coordinator_admission_succeeded"] is True
    assert runtime["original_primary_exception_available"] is False
    assert runtime["stdout_observed"] == "empty"
    assert runtime["terminal_traceback_observed"] is True
    assert [item["type"] for item in runtime["exception_chain"]] == [
        "falsewake.experiment_002_run_authority.Experiment002RunAuthorityError",
        "falsewake.experiment_006_coordinator.Experiment006CoordinatorError",
    ]
    assert [item["message"] for item in runtime["exception_chain"]] == [
        "run-registration capability was not issued by this verifier",
        "registered experiment final authority failed closed",
    ]

    assert diagnosis["root_cause_status"] == "undetermined_from_retained_evidence"
    assert diagnosis["leading_hypothesis"]["classification"] == (
        "plausible_observer_interference_not_proven"
    )
    assert diagnosis["leading_hypothesis"]["reverification_overlap_established"] is (
        False
    )
    assert diagnosis["leading_hypothesis"]["post_exit_mechanism_check"] == {
        "classification": "mechanism_reproduced_without_registered_routes",
        "git_directory_metadata_changed": True,
        "git_index_metadata_changed": False,
        "strace_events": [
            "openat .git/index read-only",
            "openat .git/index.lock with O_CREAT|O_EXCL",
            "unlink .git/index.lock",
        ],
    }
    assert incident["observer_activity"]["classification"] == (
        "plausible_interference_not_proven"
    )
    assert incident["observer_activity"]["reverification_overlap_established"] is False


def test_incident_commit_is_the_only_direct_child_change() -> None:
    assert _git("rev-list", "--parents", "-n", "1", INCIDENT_COMMIT) == (
        f"{INCIDENT_COMMIT} {REGISTRATION_COMMIT}\n".encode("ascii")
    )
    assert (
        _git(
            "diff-tree",
            "--no-ext-diff",
            "--no-textconv",
            "--no-commit-id",
            "--name-status",
            "-r",
            "-z",
            REGISTRATION_COMMIT,
            INCIDENT_COMMIT,
        )
        == b"A\0reports/experiment-006-execution-incident.json\0"
    )
    assert _git(
        "ls-tree",
        INCIDENT_COMMIT,
        "--",
        "reports/experiment-006-execution-incident.json",
    ) == (
        f"100644 blob {INCIDENT_GIT_BLOB}\t"
        "reports/experiment-006-execution-incident.json\n"
    ).encode("ascii")
    assert (
        _git(
            "show",
            f"{INCIDENT_COMMIT}:reports/experiment-006-execution-incident.json",
        )
        == INCIDENT_PATH.read_bytes()
    )

    identity = (
        "Omar Ibrahim <31526072+omar07ibrahim@users.noreply.github.com>\n"
        "Omar Ibrahim <31526072+omar07ibrahim@users.noreply.github.com>\n"
    ).encode("ascii")
    assert _git("show", "-s", "--format=%an <%ae>%n%cn <%ce>", INCIDENT_COMMIT) == (
        identity
    )


def test_no_scientific_output_or_retry_is_admitted() -> None:
    incident = _incident()

    for path in MANAGED_SCIENTIFIC_OUTPUTS:
        assert not os.path.lexists(path)
    assert incident["terminal_disposition"] == {
        "checkpoint_reuse_allowed": False,
        "experiment_006_retry_allowed": False,
        "further_recovery_profile_planned": False,
        "next_action": (
            "preserve the incident, end the recovery series, and continue portfolio "
            "work outside this execution lineage"
        ),
    }


def test_optional_external_marker_is_exact_and_unchanged() -> None:
    before = _marker_snapshot()
    if before is not None:
        frame, payload = before
        assert stat.S_IMODE(frame[2]) == 0o444
        assert frame[3] == 1
        assert frame[6] == len(MARKER_BYTES)
        assert payload == MARKER_BYTES
        assert _sha256(payload) == MARKER_SHA256
    assert _marker_snapshot() == before
