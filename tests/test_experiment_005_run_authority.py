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
from falsewake import experiment_004_run_authority as authority_004
from falsewake import experiment_005_run_authority as authority

ROOT = Path(__file__).resolve().parents[1]
P2 = engine._EXPERIMENT_002_PROFILE
P3 = engine._EXPERIMENT_003_PROFILE
P4 = engine._EXPERIMENT_004_PROFILE
P5 = engine._EXPERIMENT_005_PROFILE

P4_PROTOCOL_COMMIT = "c7a8b3c493e05211ff7afa3a7abb974fc1c8b4e2"
P4_PROTOCOL_PARENT = "96f151d86ae060b402a0ef5d47de7ad8c1191c0a"
P4_PROTOCOL_PATH = "configs/experiment-004-execution.json"
P4_PROTOCOL_SHA256 = "aba6c1eca84ad33c7751e4768a8ef630ba2c063720fd29938dd1b69abf945a46"
P4_IMPLEMENTATION_COMMIT = "f81cd142885068c27767d6d728da04248fb1a470"
P4_PREFLIGHT_BOUNDARY_COMMIT = "55043086af773e613502f81685662e7fcc58f413"
P4_INCIDENT_COMMIT = "462aeba306a0612fd6d64884e323d72db3569a89"
P4_INCIDENT_PATH = "reports/experiment-004-preflight-incident.json"
P4_INCIDENT_SHA256 = "d015b749acf6e98e87172372315c78ccfaad48b8ae6b3a29ae426890aa467f50"
P5_PROTOCOL_COMMIT = "0b6bf2cac2d4f6a04d6ced596bf8f835660eb072"
P5_PROTOCOL_PATH = "configs/experiment-005-execution.json"
P5_PROTOCOL_SHA256 = "824d1677cf8f75567cf1915381f0a9bbc778a74cba879ace80733c3ceb48e54d"
SYNTHETIC_IMPLEMENTATION_COMMIT = "5" * 40

P4_IMMUTABLE_BLOBS = (
    (
        "src/falsewake/experiment_002_run_authority.py",
        "52ce6039345b2159a4bf83ea294af2c3211cdcfbe87e9bc806133c67647b1a47",
    ),
    (
        "src/falsewake/experiment_004_coordinator.py",
        "6e4df88440c361376ce51a3574a16bbec79ecbc6ff4dd1fcf0cf8be86eac1daa",
    ),
    (
        "src/falsewake/experiment_004_final_evidence.py",
        "5fcff993e434c36d73c5e9b4f6d1e5954e8a54f2ce9fc17450ea615c5a4df5af",
    ),
    (
        "src/falsewake/experiment_004_final_publication.py",
        "b862490f15e88819e815cfe6d7769ff125e21e59cdde4642a4a700e5f2eaad16",
    ),
    (
        "src/falsewake/experiment_004_run_authority.py",
        "cacab13729f14bfdce10afa5b53b2a09116cf56abe21aee12ac9220a6ad2e5b5",
    ),
    (
        "src/falsewake/experiment_004_runner.py",
        "76f19ec33c10955fa50d26b8d35d56aeb15d712b72b22a17e88933f46d24accf",
    ),
    (
        "src/falsewake/experiment_004_seed_worker.py",
        "40cc3f0681006b03eac03f4747c60feb7a0ab42c102772afd6b90ef809929361",
    ),
    (
        "src/falsewake/experiment_004_supervisor.py",
        "750371018aefd8dc3cdbffff20a60c3785680992b05d6246f052f37609a6c8c9",
    ),
    (
        "tests/test_experiment_004_protocol.py",
        "8539610615717b8b1fb5ff41876aef41a36e8ac7a8dbda4eeb7307174ae7a3a8",
    ),
)
P5_ADDED_SOURCE_PATHS = (
    "src/falsewake/experiment_005_coordinator.py",
    "src/falsewake/experiment_005_final_evidence.py",
    "src/falsewake/experiment_005_final_publication.py",
    "src/falsewake/experiment_005_run_authority.py",
    "src/falsewake/experiment_005_runner.py",
    "src/falsewake/experiment_005_seed_worker.py",
    "src/falsewake/experiment_005_supervisor.py",
)
P4_MANAGED_OUTPUTS = (
    "configs/experiment-004-run.json",
    "models/experiment-004-selected.safetensors",
    "reports/experiment-004-seed-20260719-history.json",
    "reports/experiment-004-seed-20260720-history.json",
    "reports/experiment-004-seed-20260721-history.json",
    "reports/experiment-004-selected-rerun-history.json",
    "reports/experiment-004-training.json",
)
P5_MANAGED_OUTPUTS = (
    "configs/experiment-005-run.json",
    "models/experiment-005-selected.safetensors",
    "reports/experiment-005-seed-20260719-history.json",
    "reports/experiment-005-seed-20260720-history.json",
    "reports/experiment-005-seed-20260721-history.json",
    "reports/experiment-005-selected-rerun-history.json",
    "reports/experiment-005-training.json",
)
EXPECTED_SOURCE_DELTA = b"".join(
    status + b"\0" + path.encode("ascii") + b"\0"
    for status, path in (
        (b"M", "src/falsewake/experiment_002_run_authority.py"),
        *((b"A", path) for path in P5_ADDED_SOURCE_PATHS),
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


def _normalization_substitutions() -> tuple[tuple[str, str], ...]:
    document = json.loads((ROOT / P5_PROTOCOL_PATH).read_bytes())
    raw = document["execution_delta"]["parity_contract"]["normalization"][
        "ordered_literal_substitutions"
    ]
    return tuple((entry["candidate"], entry["reference"]) for entry in raw)


def test_facade_has_one_shared_capability_and_exact_route_shapes() -> None:
    assert authority.VerifiedRunRegistration is engine.VerifiedRunRegistration
    assert authority.Experiment005RunAuthorityError is (
        engine.Experiment002RunAuthorityError
    )
    assert authority._RegisteredChildInputSnapshot is (
        engine._RegisteredChildInputSnapshot
    )
    assert authority._VerifiedState is engine._VerifiedState
    assert authority._MAX_CHILD_BUNDLE_BYTES == engine._MAX_CHILD_BUNDLE_BYTES
    assert authority._SEALED_CHILD_BUNDLE_FD == engine._SEALED_CHILD_BUNDLE_FD
    assert P5.sealed_child_memfd_target == authority._SEALED_CHILD_MEMFD_TARGET
    assert authority.__all__ == (
        "Experiment005RunAuthorityError",
        "VerifiedRunRegistration",
        "reverify_verified_run_registration",
        "verify_and_issue_experiment_005_run_registration",
        "verify_verified_run_registration",
    )
    for zero_argument_route in (
        authority.verify_and_issue_experiment_005_run_registration,
        authority._verify_and_issue_experiment_005_sealed_child_registration,
        authority._required_child_bundle_seals,
    ):
        assert inspect.signature(zero_argument_route).parameters == {}
    for one_argument_route in (
        authority._registered_child_input_snapshot,
        authority._create_sealed_experiment_005_child_bundle_fd,
        authority._verified_state,
    ):
        code = one_argument_route.__code__
        assert (code.co_posonlyargcount, code.co_argcount) == (1, 1)
        assert one_argument_route.__defaults__ is None
        assert one_argument_route.__kwdefaults__ is None
    assert "_ISSUANCE_COMPLETE" not in authority.__dict__
    assert "_ISSUED" not in authority.__dict__
    assert "_ISSUED_GUARDS" not in authority.__dict__


def test_frozen_facade_recipe_exposes_one_honest_docstring_mismatch() -> None:
    candidate = (ROOT / "src/falsewake/experiment_005_run_authority.py").read_text()
    for source, target in _normalization_substitutions():
        candidate = candidate.replace(source, target)
    reference = (ROOT / "src/falsewake/experiment_004_run_authority.py").read_text()
    assert candidate != reference
    candidate_lines = candidate.splitlines()
    reference_lines = reference.splitlines()
    mismatches = [
        (index, candidate_line, reference_line)
        for index, (candidate_line, reference_line) in enumerate(
            zip(candidate_lines, reference_lines, strict=True),
            start=1,
        )
        if candidate_line != reference_line
    ]
    assert mismatches == [
        (
            87,
            '    """Return shared retained state only when its exact profile is '
            '``005``."""',
            '    """Return shared retained state only when its exact profile is '
            '``004``."""',
        )
    ]


def test_facade_nonissuer_routes_bind_only_profile_005(
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
    monkeypatch.setattr(engine, "_verify_experiment_005_run_registration", verify)
    monkeypatch.setattr(engine, "_reverify_experiment_005_run_registration", reverify)
    monkeypatch.setattr(
        engine,
        "_create_sealed_experiment_005_child_bundle_fd",
        create_bundle,
    )

    authority.verify_verified_run_registration(registration)
    authority.reverify_verified_run_registration(registration)
    assert authority._registered_child_input_snapshot(registration) is child_snapshot
    assert authority._verified_state(registration) is verified_state
    assert authority._create_sealed_experiment_005_child_bundle_fd(registration) == 17
    assert observed_profiles == [P5, P5]
    assert observed_routes == ["verify", "reverify", "child-input", "bundle"]


def test_profile_005_binds_all_fourteen_execution_namespaces_exactly() -> None:
    assert len({id(profile) for profile in (P2, P3, P4, P5)}) == 4
    assert dataclasses.astuple(P5) == (
        "005",
        "configs/experiment-005-run.json",
        b"falsewake-exp005-runtime-v1\0",
        b"FW5CHLD1",
        b"falsewake-exp005-source-bundle-v1\0",
        "falsewake-sealed://experiment-005/",
        "/memfd:falsewake-exp005-child-bundle (deleted)",
        "falsewake-exp005-child-bundle",
        b"FW5ACTV1",
        b"falsewake-exp005-activation-ticket-v1\0",
        "verify_and_issue_experiment_005_run_registration",
        "_verify_and_issue_experiment_005_sealed_child_registration",
        "src/falsewake/experiment_005_runner.py",
        "/home/ubuntu/gitcode/.t/falsewake-experiment-005-scratch",
    )
    assert engine._AUTHORITY_PROFILES[:4] == (P2, P3, P4, P5)
    assert engine._require_authority_profile(P5) is P5


VERIFIERS: dict[
    engine._AuthorityProfile,
    Callable[[engine.VerifiedRunRegistration], None],
] = {
    P2: engine.verify_verified_run_registration,
    P3: authority_003.verify_verified_run_registration,
    P4: authority_004.verify_verified_run_registration,
    P5: authority.verify_verified_run_registration,
}


@pytest.mark.parametrize(
    ("source_profile", "target_profile"),
    tuple(
        (source_profile, target_profile)
        for source_profile in (P2, P3, P4, P5)
        for target_profile in (P2, P3, P4, P5)
        if source_profile is not target_profile
    ),
    ids=tuple(
        f"{source.registration_experiment}-at-{target.registration_experiment}"
        for source in (P2, P3, P4, P5)
        for target in (P2, P3, P4, P5)
        if source is not target
    ),
)
def test_all_twelve_cross_profile_directions_are_rejected_before_reverification(
    monkeypatch: pytest.MonkeyPatch,
    source_profile: engine._AuthorityProfile,
    target_profile: engine._AuthorityProfile,
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
        VERIFIERS[target_profile](capability)
    assert engine._ISSUED[capability].profile is source_profile


def test_p5_parser_and_invocation_dispatch_bind_only_profile_005(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[engine._AuthorityProfile] = []

    def parse(
        payload: bytes,
        profile: engine._AuthorityProfile,
    ) -> engine._ChildBundleFrame:
        assert payload == b"controlled"
        observed.append(profile)
        return cast(engine._ChildBundleFrame, object())

    monkeypatch.setattr(engine, "_parse_child_bundle_for_profile", parse)
    engine._parse_experiment_005_child_bundle(b"controlled")
    assert observed == [P5]

    executable = "/home/ubuntu/gitcode/.t/falsewake-venv/bin/python"
    invocation: dict[str, Any] = {
        "argv": [P5.runner_entrypoint],
        "authority_process_role": "parent_bootstrap",
        "cwd": "/home/ubuntu/gitcode/falsewake",
        "environment": {
            "LANG": "C",
            "LC_ALL": "C",
            "MKL_DYNAMIC": "FALSE",
            "MKL_NUM_THREADS": "1",
            "OMP_DYNAMIC": "FALSE",
            "OMP_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "TMPDIR": P5.scratch_root,
            "TZ": "UTC",
        },
        "orig_argv": [executable, "-I", "-S", "-B", P5.runner_entrypoint],
        "python_flags": {
            "dont_write_bytecode": 1,
            "ignore_environment": 1,
            "isolated": 1,
            "no_site": 1,
            "no_user_site": 1,
        },
        "sys_path": ["/home/ubuntu/gitcode/falsewake/src"],
    }
    assert (
        engine._require_invocation_registration(
            invocation,
            {"python_executable": executable},
            P5,
        )
        is invocation
    )


def test_protocol_commit_exact_predecessor_and_frozen_files_are_bound() -> None:
    protocol_path = ROOT / P5_PROTOCOL_PATH
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    frozen = engine._expected_frozen_bindings(P5)

    assert engine._EXPERIMENT_005_PROTOCOL_INTRODUCTION_COMMIT == P5_PROTOCOL_COMMIT
    assert engine._EXPERIMENT_005_PROTOCOL_PATH == P5_PROTOCOL_PATH
    assert engine._EXPERIMENT_005_PROTOCOL_SHA256 == P5_PROTOCOL_SHA256
    assert hashlib.sha256(protocol_bytes).hexdigest() == P5_PROTOCOL_SHA256
    assert _committed_payload(P5_PROTOCOL_COMMIT, P5_PROTOCOL_PATH) == protocol_bytes
    assert frozen["execution_protocol"] == {
        "introduction_commit": P5_PROTOCOL_COMMIT,
        "path": P5_PROTOCOL_PATH,
        "sha256": P5_PROTOCOL_SHA256,
    }
    assert frozen["predecessor"] == protocol["predecessor"]
    assert engine._frozen_file_bindings(P5) == (
        *engine._frozen_file_bindings(P2),
        (P5_PROTOCOL_PATH, P5_PROTOCOL_SHA256),
        (P4_PROTOCOL_PATH, P4_PROTOCOL_SHA256),
        (P4_INCIDENT_PATH, P4_INCIDENT_SHA256),
    )
    frozen_paths = {path for path, _sha256 in engine._frozen_file_bindings(P5)}
    assert "configs/experiment-004-run.json" not in frozen_paths
    assert "reports/experiment-004-training.json" not in frozen_paths


def test_all_six_generic_dispatches_have_one_profile_005_arm() -> None:
    expected_fragments = {
        "_require_authority_profile": "elif profile is _EXPERIMENT_005_PROFILE:",
        "_verify_committed_repository": "elif profile is _EXPERIMENT_005_PROFILE:",
        "_require_invocation_registration": "_EXPERIMENT_005_PROFILE,",
        "_expected_frozen_bindings": "elif profile is _EXPERIMENT_005_PROFILE:",
        "_frozen_file_bindings": "if profile is _EXPERIMENT_005_PROFILE:",
    }
    for function_name, fragment in expected_fragments.items():
        source = inspect.getsource(getattr(engine, function_name))
        assert source.count(fragment) == 1
    profiles_source = inspect.getsource(engine._require_authority_profile)
    assert profiles_source.count("_EXPERIMENT_005_PROFILE") == 1
    assert engine._AUTHORITY_PROFILES.count(P5) == 1


@dataclass
class _HistoryTrace:
    git_calls: list[tuple[str, ...]]
    process_calls: list[tuple[str, ...]]
    blob_calls: list[tuple[str, str, int]]
    predecessor_history_calls: list[tuple[Path, str]]


_HistoryFault = Literal[
    "baseline-hash",
    "incident-hash",
    "incident-topology",
    "managed-output",
    "p4-protocol-hash",
    "p4-protocol-topology",
    "p4-source-hash",
    "p4-test-proof-hash",
    "p5-protocol-hash",
    "p5-protocol-topology",
    "p5-source-mode",
    "source-delta",
]


def _install_history_oracle(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fault: _HistoryFault | None = None,
) -> _HistoryTrace:
    trace = _HistoryTrace([], [], [], [])
    immutable_hashes = dict(P4_IMMUTABLE_BLOBS)

    def fake_git(root: Path, *args: str) -> bytes:
        assert root == ROOT
        trace.git_calls.append(args)
        if args == ("rev-list", "--parents", "-n", "1", P5_PROTOCOL_COMMIT):
            parent = "0" * 40 if fault == "p5-protocol-topology" else P4_INCIDENT_COMMIT
            return f"{P5_PROTOCOL_COMMIT} {parent}\n".encode("ascii")
        if args == (
            "diff-tree",
            "--no-ext-diff",
            "--no-textconv",
            "--no-commit-id",
            "--name-status",
            "-r",
            "-z",
            P4_INCIDENT_COMMIT,
            P5_PROTOCOL_COMMIT,
        ):
            return b"A\0" + P5_PROTOCOL_PATH.encode("ascii") + b"\0"
        if args == (
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--name-status",
            "-z",
            P4_INCIDENT_COMMIT,
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
                (args[2] == P4_INCIDENT_COMMIT and args[4] in P4_MANAGED_OUTPUTS)
                or (
                    args[2] == SYNTHETIC_IMPLEMENTATION_COMMIT
                    and args[4] in (*P4_MANAGED_OUTPUTS, *P5_MANAGED_OUTPUTS)
                )
            )
        ):
            if (
                fault == "managed-output"
                and args[2] == SYNTHETIC_IMPLEMENTATION_COMMIT
                and args[4] == P5_MANAGED_OUTPUTS[0]
            ):
                return b"100644 blob deadbeef\t" + args[4].encode() + b"\0"
            return b""
        if args == ("rev-list", "--parents", "-n", "1", P4_PROTOCOL_COMMIT):
            parent = "0" * 40 if fault == "p4-protocol-topology" else P4_PROTOCOL_PARENT
            return f"{P4_PROTOCOL_COMMIT} {parent}\n".encode("ascii")
        if args == (
            "diff-tree",
            "--no-ext-diff",
            "--no-textconv",
            "--no-commit-id",
            "--name-status",
            "-r",
            "-z",
            P4_PROTOCOL_PARENT,
            P4_PROTOCOL_COMMIT,
        ):
            return b"A\0" + P4_PROTOCOL_PATH.encode("ascii") + b"\0"
        if args == (
            "rev-list",
            "--parents",
            "-n",
            "1",
            P4_PREFLIGHT_BOUNDARY_COMMIT,
        ):
            return (
                f"{P4_PREFLIGHT_BOUNDARY_COMMIT} {P4_IMPLEMENTATION_COMMIT}\n"
            ).encode("ascii")
        if args == (
            "diff-tree",
            "--no-ext-diff",
            "--no-textconv",
            "--no-commit-id",
            "--name-status",
            "-r",
            "-z",
            P4_IMPLEMENTATION_COMMIT,
            P4_PREFLIGHT_BOUNDARY_COMMIT,
        ):
            return b"A\0docs/experiment-004.md\0"
        if args == ("rev-list", "--parents", "-n", "1", P4_INCIDENT_COMMIT):
            parent = (
                "0" * 40
                if fault == "incident-topology"
                else P4_PREFLIGHT_BOUNDARY_COMMIT
            )
            return f"{P4_INCIDENT_COMMIT} {parent}\n".encode("ascii")
        if args == (
            "diff-tree",
            "--no-ext-diff",
            "--no-textconv",
            "--no-commit-id",
            "--name-status",
            "-r",
            "-z",
            P4_PREFLIGHT_BOUNDARY_COMMIT,
            P4_INCIDENT_COMMIT,
        ):
            return b"A\0" + P4_INCIDENT_PATH.encode("ascii") + b"\0"
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
        if (commit, path) == (P5_PROTOCOL_COMMIT, P5_PROTOCOL_PATH):
            payload = _committed_payload(commit, path)
            if fault == "p5-protocol-hash":
                payload += b"\n"
        elif (commit, path) == (P4_PROTOCOL_COMMIT, P4_PROTOCOL_PATH):
            payload = _committed_payload(commit, path)
            if fault == "p4-protocol-hash":
                payload += b"\n"
        elif (commit, path) == (P4_INCIDENT_COMMIT, P4_INCIDENT_PATH):
            payload = _committed_payload(commit, path)
            if fault == "incident-hash":
                payload += b"\n"
        elif commit == P4_IMPLEMENTATION_COMMIT and path in immutable_hashes:
            payload = _committed_payload(commit, path)
            if (
                fault == "baseline-hash"
                and path == "src/falsewake/experiment_002_run_authority.py"
            ):
                payload += b"\n"
            if (
                fault == "p4-source-hash"
                and path == "src/falsewake/experiment_004_coordinator.py"
            ):
                payload += b"\n"
            if (
                fault == "p4-test-proof-hash"
                and path == "tests/test_experiment_004_protocol.py"
            ):
                payload += b"\n"
        elif (
            commit == SYNTHETIC_IMPLEMENTATION_COMMIT
            and path == "src/falsewake/experiment_002_run_authority.py"
        ):
            payload = b"synthetic shared authority\n"
        elif (
            commit == SYNTHETIC_IMPLEMENTATION_COMMIT and path in P5_ADDED_SOURCE_PATHS
        ):
            payload = f"synthetic {path}\n".encode("ascii")
            if fault == "p5-source-mode" and path == P5_ADDED_SOURCE_PATHS[0]:
                mode = "100755"
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
        "_require_experiment_004_implementation_history",
        fake_predecessor_history,
    )
    return trace


def test_p5_implementation_history_checks_exact_topology_modes_and_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = _install_history_oracle(monkeypatch)

    engine._require_experiment_005_implementation_history(
        ROOT,
        implementation_commit=SYNTHETIC_IMPLEMENTATION_COMMIT,
    )

    assert trace.predecessor_history_calls == [(ROOT, P4_IMPLEMENTATION_COMMIT)]
    assert trace.process_calls == [
        (
            "merge-base",
            "--is-ancestor",
            P5_PROTOCOL_COMMIT,
            SYNTHETIC_IMPLEMENTATION_COMMIT,
        ),
        (
            "merge-base",
            "--is-ancestor",
            P4_PROTOCOL_COMMIT,
            P4_IMPLEMENTATION_COMMIT,
        ),
        (
            "merge-base",
            "--is-ancestor",
            P4_IMPLEMENTATION_COMMIT,
            P4_INCIDENT_COMMIT,
        ),
    ]
    assert {
        path
        for commit, path, _maximum_bytes in trace.blob_calls
        if commit == P4_IMPLEMENTATION_COMMIT
    } == {path for path, _sha256 in P4_IMMUTABLE_BLOBS}
    assert {
        path
        for commit, path, _maximum_bytes in trace.blob_calls
        if commit == SYNTHETIC_IMPLEMENTATION_COMMIT
    } == {
        "src/falsewake/experiment_002_run_authority.py",
        *P5_ADDED_SOURCE_PATHS,
    }
    for path in P4_MANAGED_OUTPUTS:
        assert (
            "ls-tree",
            "-z",
            P4_INCIDENT_COMMIT,
            "--",
            path,
        ) in trace.git_calls
    for path in (*P4_MANAGED_OUTPUTS, *P5_MANAGED_OUTPUTS):
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
        ("p5-protocol-topology", "protocol introduction topology changed"),
        ("p5-protocol-hash", "protocol introduction blob changed"),
        ("source-delta", "implementation source delta is not exact"),
        ("managed-output", "history contains managed output"),
        ("p4-protocol-topology", "predecessor protocol topology changed"),
        ("p4-protocol-hash", "predecessor protocol blob changed"),
        ("incident-topology", "preflight incident topology changed"),
        ("incident-hash", "preflight incident blob changed"),
        ("baseline-hash", "predecessor implementation changed"),
        ("p4-source-hash", "predecessor implementation changed"),
        ("p4-test-proof-hash", "predecessor implementation changed"),
        ("p5-source-mode", "source mode is not exact"),
    ),
)
def test_p5_implementation_history_rejects_each_frozen_dimension(
    monkeypatch: pytest.MonkeyPatch,
    fault: _HistoryFault,
    message: str,
) -> None:
    _install_history_oracle(monkeypatch, fault=fault)

    with pytest.raises(engine.Experiment002RunAuthorityError, match=message):
        engine._require_experiment_005_implementation_history(
            ROOT,
            implementation_commit=SYNTHETIC_IMPLEMENTATION_COMMIT,
        )


def test_p5_history_never_inspects_or_mutates_the_external_p4_marker() -> None:
    source = inspect.getsource(engine._require_experiment_005_implementation_history)
    assert "/home/ubuntu/gitcode/.t/falsewake-experiment-004-attempt" not in source
    assert "lexists" not in source
    assert "unlink" not in source
    assert "open(" not in source
