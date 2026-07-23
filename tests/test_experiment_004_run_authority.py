from __future__ import annotations

import dataclasses
import functools
import hashlib
import inspect
import json
import subprocess
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import pytest

from falsewake import experiment_002_run_authority as engine
from falsewake import experiment_003_run_authority as authority_003
from falsewake import experiment_004_run_authority as authority

ROOT = Path(__file__).resolve().parents[1]
P2 = engine._EXPERIMENT_002_PROFILE
P3 = engine._EXPERIMENT_003_PROFILE
P4 = engine._EXPERIMENT_004_PROFILE

P3_PROTOCOL_COMMIT = "e47bd581675abe22b529de7bcc825d39e84a00cd"
P3_PROTOCOL_PARENT = "650eefbf9f82b207ed58e3b1a1eac41197466b41"
P3_PROTOCOL_PATH = "configs/experiment-003-execution.json"
P3_PROTOCOL_SHA256 = "3f48cb48f6c3a56284f272fa9e308ae62b3749df64cb7a581e770da8bdd8218a"
P3_IMPLEMENTATION_COMMIT = "4475461d5fd3e5b8969020003424c73bb10d3c9b"
P3_REGISTRATION_COMMIT = "f7426fb038ae5dc0c24b5141d12d55323ac7c961"
P3_REGISTRATION_PATH = "configs/experiment-003-run.json"
P3_REGISTRATION_SHA256 = (
    "941a4ec67d2adbe0149861a678fc19e46ac4e2188c126b64c4b9c54f01425b2c"
)
P3_INCIDENT_COMMIT = "96f151d86ae060b402a0ef5d47de7ad8c1191c0a"
P3_INCIDENT_PATH = "reports/experiment-003-execution-incident.json"
P3_INCIDENT_SHA256 = "0dc24fd211129cd2b81e29fb91ac9e66a0aad25f1deeae332f3b6ea420567688"
P4_PROTOCOL_COMMIT = "c7a8b3c493e05211ff7afa3a7abb974fc1c8b4e2"
P4_PROTOCOL_PATH = "configs/experiment-004-execution.json"
P4_PROTOCOL_SHA256 = "aba6c1eca84ad33c7751e4768a8ef630ba2c063720fd29938dd1b69abf945a46"
SYNTHETIC_IMPLEMENTATION_COMMIT = "4" * 40

P4_PREDECESSOR: dict[str, Any] = {
    "execution_protocol": {
        "introduction_commit": P3_PROTOCOL_COMMIT,
        "path": P3_PROTOCOL_PATH,
        "sha256": P3_PROTOCOL_SHA256,
    },
    "experiment": "003",
    "implementation": {
        "commit": P3_IMPLEMENTATION_COMMIT,
        "runtime_fingerprint_sha256": (
            "c167d636026909879c952858c67c304ce5a1bbb981284242c3d6235efe205f8f"
        ),
        "source_bundle_sha256": (
            "28094defd76ba395a5b4eec477b6f0516686013df5995f24c3c126a35b09c0b6"
        ),
    },
    "incident": {
        "commit": P3_INCIDENT_COMMIT,
        "path": P3_INCIDENT_PATH,
        "sha256": P3_INCIDENT_SHA256,
    },
    "outcome": {
        "automatic_terminal_report_published": False,
        "checkpoint_reusable": False,
        "code": "parent_route_signature_mismatch",
        "phase": "registered_admission",
        "status": "execution_failure",
    },
    "registration": {
        "commit": P3_REGISTRATION_COMMIT,
        "path": P3_REGISTRATION_PATH,
        "sha256": P3_REGISTRATION_SHA256,
        "source_bundle_sha256": (
            "28094defd76ba395a5b4eec477b6f0516686013df5995f24c3c126a35b09c0b6"
        ),
    },
    "reuse_forbidden": True,
    "scientific_protocol": "002",
    "terminal": True,
    "topology": {
        "incident_commit": P3_INCIDENT_COMMIT,
        "incident_parent": P3_REGISTRATION_COMMIT,
        "outcome_report_path": "reports/experiment-003-training.json",
        "outcome_report_present": False,
        "registration_commit": P3_REGISTRATION_COMMIT,
        "registration_parent": P3_IMPLEMENTATION_COMMIT,
    },
}

P3_SOURCE_HASHES = (
    (
        "src/falsewake/experiment_003_coordinator.py",
        "f2b5ff3e8e874ddf408ba97475c69ca13875e7ac3a37c7462ba2637099a77d1a",
    ),
    (
        "src/falsewake/experiment_003_final_evidence.py",
        "6f0ebd8554e703886e63dc9d508ec83ab5a4ce3997b0f99714b441106216b9ad",
    ),
    (
        "src/falsewake/experiment_003_final_publication.py",
        "af61f1425a936b51711f70b097c8b5399571e32d39511fdf7dc947270fec59bb",
    ),
    (
        "src/falsewake/experiment_003_run_authority.py",
        "690ed90679928766a50219657ba467e0e0d39bf2763fa2b60128640fdf0938e3",
    ),
    (
        "src/falsewake/experiment_003_runner.py",
        "6e9eb84027487b17e4a375fe04f8c704c11aa4d78458405900625d462fcd5a4d",
    ),
    (
        "src/falsewake/experiment_003_seed_worker.py",
        "f9262a47d0bf2621ebb2837017b9947927980bee2c498171cabc34bd9ba800ad",
    ),
    (
        "src/falsewake/experiment_003_supervisor.py",
        "aa33a0f852d3b43f3eff0ded09873aa60911008ba1d37fa3acf4b5d34ab501e3",
    ),
)
P4_ADDED_SOURCE_PATHS = (
    "src/falsewake/experiment_004_coordinator.py",
    "src/falsewake/experiment_004_final_evidence.py",
    "src/falsewake/experiment_004_final_publication.py",
    "src/falsewake/experiment_004_run_authority.py",
    "src/falsewake/experiment_004_runner.py",
    "src/falsewake/experiment_004_seed_worker.py",
    "src/falsewake/experiment_004_supervisor.py",
)
P4_FORBIDDEN_OUTPUTS = (
    "configs/experiment-004-run.json",
    "models/experiment-004-selected.safetensors",
    "reports/experiment-004-seed-20260719-history.json",
    "reports/experiment-004-seed-20260720-history.json",
    "reports/experiment-004-seed-20260721-history.json",
    "reports/experiment-004-selected-rerun-history.json",
    "reports/experiment-004-training.json",
    "models/experiment-003-selected.safetensors",
    "reports/experiment-003-seed-20260719-history.json",
    "reports/experiment-003-seed-20260720-history.json",
    "reports/experiment-003-seed-20260721-history.json",
    "reports/experiment-003-selected-rerun-history.json",
    "reports/experiment-003-training.json",
)
EXPECTED_SOURCE_DELTA = b"".join(
    status + b"\0" + path.encode("ascii") + b"\0"
    for status, path in (
        (b"M", "src/falsewake/experiment_002_run_authority.py"),
        *((b"A", path) for path in P4_ADDED_SOURCE_PATHS),
    )
)


def _reset_issuer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine, "_ISSUED", weakref.WeakKeyDictionary())
    monkeypatch.setattr(engine, "_ISSUED_GUARDS", weakref.WeakKeyDictionary())
    monkeypatch.setattr(engine, "_FAILED", weakref.WeakSet())
    monkeypatch.setattr(engine, "_ISSUANCE_COMPLETE", False)


@pytest.fixture(autouse=True)
def reset_issuer(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_issuer(monkeypatch)


def _snapshot(profile: engine._AuthorityProfile) -> engine._RepositorySnapshot:
    return engine._RepositorySnapshot(
        repository_root=ROOT,
        head_commit="2" * 40,
        implementation_commit="1" * 40,
        registration_sha256="3" * 64,
        source_bundle_sha256="4" * 64,
        source_paths=("src/falsewake/experiment_002_run_authority.py",),
        profile=profile,
    )


@functools.cache
def _committed_payload(commit: str, path: str) -> bytes:
    completed = subprocess.run(
        ("git", "show", f"{commit}:{path}"),
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return completed.stdout


def test_facade_has_one_shared_capability_and_fixed_zero_argument_issuers() -> None:
    assert authority.VerifiedRunRegistration is engine.VerifiedRunRegistration
    assert authority.Experiment004RunAuthorityError is (
        engine.Experiment002RunAuthorityError
    )
    assert authority._RegisteredChildInputSnapshot is (
        engine._RegisteredChildInputSnapshot
    )
    assert authority._VerifiedState is engine._VerifiedState
    assert authority._MAX_CHILD_BUNDLE_BYTES == engine._MAX_CHILD_BUNDLE_BYTES
    assert authority._SEALED_CHILD_BUNDLE_FD == engine._SEALED_CHILD_BUNDLE_FD
    assert P4.sealed_child_memfd_target == authority._SEALED_CHILD_MEMFD_TARGET
    assert authority.__all__ == (
        "Experiment004RunAuthorityError",
        "VerifiedRunRegistration",
        "reverify_verified_run_registration",
        "verify_and_issue_experiment_004_run_registration",
        "verify_verified_run_registration",
    )
    assert (
        tuple(
            inspect.signature(
                authority.verify_and_issue_experiment_004_run_registration
            ).parameters
        )
        == ()
    )
    assert (
        tuple(
            inspect.signature(
                authority._verify_and_issue_experiment_004_sealed_child_registration
            ).parameters
        )
        == ()
    )
    for route in (
        authority.verify_verified_run_registration,
        authority.reverify_verified_run_registration,
        authority._registered_child_input_snapshot,
        authority._create_sealed_experiment_004_child_bundle_fd,
        authority._verified_state,
    ):
        assert tuple(inspect.signature(route).parameters) == ("registration",)
    assert "_ISSUANCE_COMPLETE" not in authority.__dict__
    assert "_ISSUED" not in authority.__dict__
    assert "_ISSUED_GUARDS" not in authority.__dict__


def test_facade_nonissuer_routes_bind_only_profile_004(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration = cast(engine.VerifiedRunRegistration, object())
    child_snapshot = cast(engine._RegisteredChildInputSnapshot, object())
    verified_state = cast(engine._VerifiedState, object())
    observed_profiles: list[engine._AuthorityProfile] = []
    observed_routes: list[str] = []

    def require_profile(
        candidate: engine.VerifiedRunRegistration,
        profile: engine._AuthorityProfile,
    ) -> engine._VerifiedState:
        assert candidate is registration
        observed_profiles.append(profile)
        return verified_state

    def child_input(
        candidate: engine.VerifiedRunRegistration,
    ) -> engine._RegisteredChildInputSnapshot:
        assert candidate is registration
        observed_routes.append("child-input")
        return child_snapshot

    def verify(candidate: engine.VerifiedRunRegistration) -> None:
        assert candidate is registration
        observed_routes.append("verify")

    def reverify(candidate: engine.VerifiedRunRegistration) -> None:
        assert candidate is registration
        observed_routes.append("reverify")

    def create_bundle(candidate: engine.VerifiedRunRegistration, /) -> int:
        assert candidate is registration
        observed_routes.append("bundle")
        return 17

    monkeypatch.setattr(
        engine,
        "_require_verified_registration_profile",
        require_profile,
    )
    monkeypatch.setattr(engine, "_registered_child_input_snapshot", child_input)
    monkeypatch.setattr(engine, "_verify_experiment_004_run_registration", verify)
    monkeypatch.setattr(engine, "_reverify_experiment_004_run_registration", reverify)
    monkeypatch.setattr(
        engine,
        "_create_sealed_experiment_004_child_bundle_fd",
        create_bundle,
    )

    authority.verify_verified_run_registration(registration)
    authority.reverify_verified_run_registration(registration)
    assert authority._registered_child_input_snapshot(registration) is child_snapshot
    assert authority._verified_state(registration) is verified_state
    assert authority._create_sealed_experiment_004_child_bundle_fd(registration) == 17
    assert observed_profiles == [P4, P4]
    assert observed_routes == ["verify", "reverify", "child-input", "bundle"]


def test_profile_004_binds_all_fourteen_execution_namespaces_exactly() -> None:
    assert P4 is not P2
    assert P4 is not P3
    assert dataclasses.astuple(P4) == (
        "004",
        "configs/experiment-004-run.json",
        b"falsewake-exp004-runtime-v1\0",
        b"FW4CHLD1",
        b"falsewake-exp004-source-bundle-v1\0",
        "falsewake-sealed://experiment-004/",
        "/memfd:falsewake-exp004-child-bundle (deleted)",
        "falsewake-exp004-child-bundle",
        b"FW4ACTV1",
        b"falsewake-exp004-activation-ticket-v1\0",
        "verify_and_issue_experiment_004_run_registration",
        "_verify_and_issue_experiment_004_sealed_child_registration",
        "src/falsewake/experiment_004_runner.py",
        "/home/ubuntu/gitcode/.t/falsewake-experiment-004-scratch",
    )
    assert engine._AUTHORITY_PROFILES[:3] == (P2, P3, P4)
    assert engine._require_authority_profile(P4) is P4


@pytest.mark.parametrize(
    ("source_profile", "target_verifier"),
    (
        (P2, authority_003.verify_verified_run_registration),
        (P2, authority.verify_verified_run_registration),
        (P3, engine.verify_verified_run_registration),
        (P3, authority.verify_verified_run_registration),
        (P4, engine.verify_verified_run_registration),
        (P4, authority_003.verify_verified_run_registration),
    ),
    ids=(
        "002-at-003",
        "002-at-004",
        "003-at-002",
        "003-at-004",
        "004-at-002",
        "004-at-003",
    ),
)
def test_all_six_cross_profile_directions_are_rejected_before_reverification(
    monkeypatch: pytest.MonkeyPatch,
    source_profile: engine._AuthorityProfile,
    target_verifier: Callable[[engine.VerifiedRunRegistration], None],
) -> None:
    def forbidden_reverification(
        _registration: engine.VerifiedRunRegistration,
    ) -> None:
        pytest.fail("cross-profile capability reached generic reverification")

    monkeypatch.setattr(
        engine,
        "reverify_verified_run_registration",
        forbidden_reverification,
    )
    capability = engine._issue_controlled_snapshot_for_tests(_snapshot(source_profile))

    with pytest.raises(
        engine.Experiment002RunAuthorityError,
        match="different authority profile",
    ):
        target_verifier(capability)
    assert engine._ISSUED[capability].profile is source_profile


def test_protocol_commit_sha_and_exact_predecessor_are_frozen() -> None:
    protocol_path = ROOT / P4_PROTOCOL_PATH
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    frozen = engine._expected_frozen_bindings(P4)

    assert engine._EXPERIMENT_004_PROTOCOL_INTRODUCTION_COMMIT == P4_PROTOCOL_COMMIT
    assert engine._EXPERIMENT_004_PROTOCOL_PATH == P4_PROTOCOL_PATH
    assert engine._EXPERIMENT_004_PROTOCOL_SHA256 == P4_PROTOCOL_SHA256
    assert hashlib.sha256(protocol_bytes).hexdigest() == P4_PROTOCOL_SHA256
    assert _committed_payload(P4_PROTOCOL_COMMIT, P4_PROTOCOL_PATH) == protocol_bytes
    assert frozen["execution_protocol"] == {
        "introduction_commit": P4_PROTOCOL_COMMIT,
        "path": P4_PROTOCOL_PATH,
        "sha256": P4_PROTOCOL_SHA256,
    }
    assert protocol["predecessor"] == P4_PREDECESSOR
    assert frozen["predecessor"] == P4_PREDECESSOR


def test_p4_frozen_files_take_the_no_outcome_file_branch() -> None:
    frozen = engine._expected_frozen_bindings(P4)
    predecessor = cast(dict[str, Any], frozen["predecessor"])
    outcome = cast(dict[str, Any], predecessor["outcome"])

    assert "path" not in outcome
    assert "sha256" not in outcome
    assert engine._frozen_file_bindings(P4) == (
        *engine._frozen_file_bindings(P2),
        (P4_PROTOCOL_PATH, P4_PROTOCOL_SHA256),
        (P3_PROTOCOL_PATH, P3_PROTOCOL_SHA256),
        (P3_REGISTRATION_PATH, P3_REGISTRATION_SHA256),
        (P3_INCIDENT_PATH, P3_INCIDENT_SHA256),
    )
    assert all(
        path != "reports/experiment-003-training.json"
        for path, _sha256 in engine._frozen_file_bindings(P4)
    )


@dataclass
class _HistoryTrace:
    git_calls: list[tuple[str, ...]]
    process_calls: list[tuple[str, ...]]
    blob_calls: list[tuple[str, str, int]]
    predecessor_history_calls: list[tuple[Path, str]]


_HistoryFault = Literal[
    "baseline-hash",
    "p3-protocol-topology",
    "p3-registration-topology",
    "p3-source-hash",
    "p4-protocol-hash",
    "p4-protocol-topology",
    "p4-source-mode",
    "source-delta",
]


def _install_history_oracle(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fault: _HistoryFault | None = None,
) -> _HistoryTrace:
    trace = _HistoryTrace([], [], [], [])
    p3_source_paths = {path for path, _sha256 in P3_SOURCE_HASHES}

    def fake_git(root: Path, *args: str) -> bytes:
        assert root == ROOT
        trace.git_calls.append(args)
        if args == (
            "rev-list",
            "--parents",
            "-n",
            "1",
            P4_PROTOCOL_COMMIT,
        ):
            parent = "0" * 40 if fault == "p4-protocol-topology" else P3_INCIDENT_COMMIT
            return f"{P4_PROTOCOL_COMMIT} {parent}\n".encode("ascii")
        if args == (
            "diff-tree",
            "--no-ext-diff",
            "--no-textconv",
            "--no-commit-id",
            "--name-status",
            "-r",
            "-z",
            P3_INCIDENT_COMMIT,
            P4_PROTOCOL_COMMIT,
        ):
            return b"A\0" + P4_PROTOCOL_PATH.encode("ascii") + b"\0"
        if args == (
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--name-status",
            "-z",
            P3_INCIDENT_COMMIT,
            SYNTHETIC_IMPLEMENTATION_COMMIT,
            "--",
            "src/falsewake",
        ):
            if fault == "source-delta":
                return EXPECTED_SOURCE_DELTA + b"A\0src/falsewake/unexpected.py\0"
            return EXPECTED_SOURCE_DELTA
        if (
            len(args) == 5
            and args[:2] == ("ls-tree", "-z")
            and args[3] == "--"
            and (
                (
                    args[2] == SYNTHETIC_IMPLEMENTATION_COMMIT
                    and args[4] in P4_FORBIDDEN_OUTPUTS
                )
                or (
                    args[2] == P3_INCIDENT_COMMIT
                    and args[4] == "reports/experiment-003-training.json"
                )
            )
        ):
            return b""
        if args == (
            "rev-list",
            "--parents",
            "-n",
            "1",
            P3_PROTOCOL_COMMIT,
        ):
            parent = "0" * 40 if fault == "p3-protocol-topology" else P3_PROTOCOL_PARENT
            return f"{P3_PROTOCOL_COMMIT} {parent}\n".encode("ascii")
        if args == (
            "diff-tree",
            "--no-ext-diff",
            "--no-textconv",
            "--no-commit-id",
            "--name-status",
            "-r",
            "-z",
            P3_PROTOCOL_PARENT,
            P3_PROTOCOL_COMMIT,
        ):
            return b"A\0" + P3_PROTOCOL_PATH.encode("ascii") + b"\0"
        if args == (
            "rev-list",
            "--parents",
            "-n",
            "1",
            P3_REGISTRATION_COMMIT,
        ):
            parent = (
                "0" * 40
                if fault == "p3-registration-topology"
                else P3_IMPLEMENTATION_COMMIT
            )
            return f"{P3_REGISTRATION_COMMIT} {parent}\n".encode("ascii")
        if args == (
            "rev-list",
            "--parents",
            "-n",
            "1",
            P3_INCIDENT_COMMIT,
        ):
            return (f"{P3_INCIDENT_COMMIT} {P3_REGISTRATION_COMMIT}\n").encode("ascii")
        if args == (
            "diff-tree",
            "--no-ext-diff",
            "--no-textconv",
            "--no-commit-id",
            "--name-status",
            "-r",
            "-z",
            P3_IMPLEMENTATION_COMMIT,
            P3_REGISTRATION_COMMIT,
        ):
            return b"A\0" + P3_REGISTRATION_PATH.encode("ascii") + b"\0"
        if args == (
            "diff-tree",
            "--no-ext-diff",
            "--no-textconv",
            "--no-commit-id",
            "--name-status",
            "-r",
            "-z",
            P3_REGISTRATION_COMMIT,
            P3_INCIDENT_COMMIT,
        ):
            return b"A\0" + P3_INCIDENT_PATH.encode("ascii") + b"\0"
        raise AssertionError(f"unexpected Git history query: {args!r}")

    def fake_git_process(
        root: Path,
        args: tuple[str, ...],
        *,
        allowed_returncodes: tuple[int, ...],
    ) -> subprocess.CompletedProcess[bytes]:
        assert root == ROOT
        assert allowed_returncodes == (0, 1)
        trace.process_calls.append(args)
        return subprocess.CompletedProcess(
            args=("git", *args),
            returncode=0,
            stdout=b"",
            stderr=b"",
        )

    def fake_committed_blob(
        root: Path,
        commit: str,
        path: str,
        *,
        maximum_bytes: int,
    ) -> engine._TreeBlob:
        assert root == ROOT
        trace.blob_calls.append((commit, path, maximum_bytes))
        mode = "100644"
        if (commit, path) == (P4_PROTOCOL_COMMIT, P4_PROTOCOL_PATH):
            payload = _committed_payload(commit, path)
            if fault == "p4-protocol-hash":
                payload += b"\n"
        elif (commit, path) == (P3_PROTOCOL_COMMIT, P3_PROTOCOL_PATH):
            payload = _committed_payload(commit, path)
        elif (commit, path) == (
            P3_INCIDENT_COMMIT,
            "src/falsewake/experiment_002_run_authority.py",
        ):
            payload = _committed_payload(commit, path)
            if fault == "baseline-hash":
                payload += b"\n"
        elif commit == SYNTHETIC_IMPLEMENTATION_COMMIT and path in p3_source_paths:
            payload = _committed_payload(P3_IMPLEMENTATION_COMMIT, path)
            if fault == "p3-source-hash" and path == P3_SOURCE_HASHES[0][0]:
                payload += b"\n"
        elif (
            commit == SYNTHETIC_IMPLEMENTATION_COMMIT
            and path == "src/falsewake/experiment_002_run_authority.py"
        ):
            payload = b"synthetic shared authority\n"
        elif (
            commit == SYNTHETIC_IMPLEMENTATION_COMMIT and path in P4_ADDED_SOURCE_PATHS
        ):
            payload = f"synthetic {path}\n".encode("ascii")
            if fault == "p4-source-mode" and path == P4_ADDED_SOURCE_PATHS[0]:
                mode = "100755"
        elif (commit, path) in {
            (P3_REGISTRATION_COMMIT, P3_REGISTRATION_PATH),
            (P3_INCIDENT_COMMIT, P3_INCIDENT_PATH),
        }:
            payload = _committed_payload(commit, path)
        else:
            raise AssertionError(f"unexpected committed blob query: {(commit, path)!r}")
        assert len(payload) <= maximum_bytes
        return engine._TreeBlob(mode=mode, oid="a" * 40, payload=payload)

    def fake_predecessor_history(
        root: Path,
        *,
        implementation_commit: str,
    ) -> None:
        trace.predecessor_history_calls.append((root, implementation_commit))

    monkeypatch.setattr(engine, "_git", fake_git)
    monkeypatch.setattr(engine, "_git_process", fake_git_process)
    monkeypatch.setattr(engine, "_committed_blob", fake_committed_blob)
    monkeypatch.setattr(
        engine,
        "_require_experiment_003_implementation_history",
        fake_predecessor_history,
    )
    return trace


def test_p4_implementation_history_checks_exact_topology_modes_and_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = _install_history_oracle(monkeypatch)

    engine._require_experiment_004_implementation_history(
        ROOT,
        implementation_commit=SYNTHETIC_IMPLEMENTATION_COMMIT,
    )

    assert trace.predecessor_history_calls == [(ROOT, P3_IMPLEMENTATION_COMMIT)]
    assert trace.process_calls == [
        (
            "merge-base",
            "--is-ancestor",
            P4_PROTOCOL_COMMIT,
            SYNTHETIC_IMPLEMENTATION_COMMIT,
        ),
        (
            "merge-base",
            "--is-ancestor",
            P3_PROTOCOL_COMMIT,
            P3_IMPLEMENTATION_COMMIT,
        ),
    ]
    assert (
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--name-status",
        "-z",
        P3_INCIDENT_COMMIT,
        SYNTHETIC_IMPLEMENTATION_COMMIT,
        "--",
        "src/falsewake",
    ) in trace.git_calls
    assert (
        "diff-tree",
        "--no-ext-diff",
        "--no-textconv",
        "--no-commit-id",
        "--name-status",
        "-r",
        "-z",
        P3_PROTOCOL_PARENT,
        P3_PROTOCOL_COMMIT,
    ) in trace.git_calls
    assert {
        path
        for commit, path, _maximum_bytes in trace.blob_calls
        if commit == SYNTHETIC_IMPLEMENTATION_COMMIT
    } == {
        "src/falsewake/experiment_002_run_authority.py",
        *(path for path, _sha256 in P3_SOURCE_HASHES),
        *P4_ADDED_SOURCE_PATHS,
    }
    for path in P4_FORBIDDEN_OUTPUTS:
        assert (
            "ls-tree",
            "-z",
            SYNTHETIC_IMPLEMENTATION_COMMIT,
            "--",
            path,
        ) in trace.git_calls


@pytest.mark.parametrize(
    ("fault", "message"),
    (
        ("p4-protocol-topology", "protocol introduction topology changed"),
        ("p4-protocol-hash", "protocol introduction blob changed"),
        ("source-delta", "implementation source delta is not exact"),
        ("p3-protocol-topology", "predecessor protocol topology changed"),
        ("p3-registration-topology", "predecessor topology changed"),
        ("baseline-hash", "shared-authority baseline changed"),
        ("p3-source-hash", "predecessor source changed"),
        ("p4-source-mode", "source mode is not exact"),
    ),
)
def test_p4_implementation_history_rejects_each_frozen_dimension(
    monkeypatch: pytest.MonkeyPatch,
    fault: _HistoryFault,
    message: str,
) -> None:
    _install_history_oracle(monkeypatch, fault=fault)

    with pytest.raises(engine.Experiment002RunAuthorityError, match=message):
        engine._require_experiment_004_implementation_history(
            ROOT,
            implementation_commit=SYNTHETIC_IMPLEMENTATION_COMMIT,
        )
