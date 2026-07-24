from __future__ import annotations

import ast
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
from falsewake import experiment_005_run_authority as authority_005
from falsewake import experiment_006_run_authority as authority

ROOT = Path(__file__).resolve().parents[1]
P2 = engine._EXPERIMENT_002_PROFILE
P3 = engine._EXPERIMENT_003_PROFILE
P4 = engine._EXPERIMENT_004_PROFILE
P5 = engine._EXPERIMENT_005_PROFILE
P6 = engine._EXPERIMENT_006_PROFILE

P5_PROTOCOL_COMMIT = "0b6bf2cac2d4f6a04d6ced596bf8f835660eb072"
P5_PROTOCOL_PARENT = "462aeba306a0612fd6d64884e323d72db3569a89"
P5_PROTOCOL_PATH = "configs/experiment-005-execution.json"
P5_PROTOCOL_SHA256 = "824d1677cf8f75567cf1915381f0a9bbc778a74cba879ace80733c3ceb48e54d"
P5_IMPLEMENTATION_COMMIT = "04b634451d129faadadd921215123547e3467d65"
P5_PROTOCOL_PROOF_COMMIT = "1e20d6cf210a2d4a59cfc737a14eaccfc94a5b74"
P5_PROTOCOL_PROOF_PATH = "tests/test_experiment_005_protocol.py"
P5_PROTOCOL_PROOF_SHA256 = (
    "cf6d4ee2fcba31de37774b591571abb2d86998cc0f2a3e99a40abe1af4af5414"
)
P5_PREFLIGHT_BOUNDARY_COMMIT = "9ae549ccd477b13b9531f7c44d854d34549fd9c3"
P5_PREFLIGHT_BOUNDARY_PATH = "docs/experiment-005.md"
P5_PREFLIGHT_BOUNDARY_SHA256 = (
    "c9852a028f572e33b2379b98786f5c72e41ddae1300f03742ad977947a1dd600"
)
P5_INCIDENT_COMMIT = "594e2c491c057394f18860c8362a51190460645f"
P5_INCIDENT_PATH = "reports/experiment-005-preflight-incident.json"
P5_INCIDENT_SHA256 = "3de6fbbd6912bee47f3ef01f5371b57ae8d5942b6fc9915fc0ec243b9e931e3d"
P5_INCIDENT_BYTE_COUNT = 6_612
P6_PROTOCOL_COMMIT = "4adf524f9a3007d5382a851b3419e697d80ceeef"
P6_PROTOCOL_PATH = "configs/experiment-006-execution.json"
P6_PROTOCOL_SHA256 = "47ce6c6f11c3575944ef8c4ab1edcc0b42fdfa9bf1ede0318bfc5ea61fabf650"
SYNTHETIC_IMPLEMENTATION_COMMIT = "6" * 40

P5_IMMUTABLE_BLOBS = (
    (
        "src/falsewake/experiment_002_run_authority.py",
        "937ad120dd5828cb4029cc1382df2a0a7c06aef14c1d37bfdb848d93993416af",
    ),
    (
        "src/falsewake/experiment_005_coordinator.py",
        "4ad78de14ce12acdb63b4e103dd920ce878d7e60a06fa1c5893027d17401f812",
    ),
    (
        "src/falsewake/experiment_005_final_evidence.py",
        "1c99757fc9b78e612891a9d49e7dc40c2d0b90599a40927e4cd41dff781f355d",
    ),
    (
        "src/falsewake/experiment_005_final_publication.py",
        "84f7aba57f8ec0c096e5f18a03a8118800f53a9cfe4910c554bea677a9f79f07",
    ),
    (
        "src/falsewake/experiment_005_run_authority.py",
        "87d18e901b4b814ce31c7c17b60b6aa54f837edc84ad3bb5ee1af386df006f5e",
    ),
    (
        "src/falsewake/experiment_005_runner.py",
        "eeafface36baaabba5e1479750674884a069b41ccaeeb9dee9ccbc4b179a5336",
    ),
    (
        "src/falsewake/experiment_005_seed_worker.py",
        "0840a62f0f4327eec7649d17ad334edfa6941d6146bff6fe50e2d693f0b80f96",
    ),
    (
        "src/falsewake/experiment_005_supervisor.py",
        "7c8d58566c2b5e904ebf7f1e5fa1ab3b1c79c4387b38f296b08ef5ce7e312e71",
    ),
)
P6_ADDED_SOURCE_PATHS = (
    "src/falsewake/experiment_006_coordinator.py",
    "src/falsewake/experiment_006_final_evidence.py",
    "src/falsewake/experiment_006_final_publication.py",
    "src/falsewake/experiment_006_run_authority.py",
    "src/falsewake/experiment_006_runner.py",
    "src/falsewake/experiment_006_seed_worker.py",
    "src/falsewake/experiment_006_supervisor.py",
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
P6_MANAGED_OUTPUTS = (
    "configs/experiment-006-run.json",
    "models/experiment-006-selected.safetensors",
    "reports/experiment-006-seed-20260719-history.json",
    "reports/experiment-006-seed-20260720-history.json",
    "reports/experiment-006-seed-20260721-history.json",
    "reports/experiment-006-selected-rerun-history.json",
    "reports/experiment-006-training.json",
)
EXPECTED_SOURCE_DELTA = b"".join(
    status + b"\0" + path.encode("ascii") + b"\0"
    for status, path in (
        (b"M", "src/falsewake/experiment_002_run_authority.py"),
        *((b"A", path) for path in P6_ADDED_SOURCE_PATHS),
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


@functools.cache
def _protocol() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads((ROOT / P6_PROTOCOL_PATH).read_bytes()))


def _unparse_required(node: ast.expr | None) -> str:
    assert node is not None
    return ast.unparse(node)


def test_facade_has_one_shared_capability_and_exact_route_shapes() -> None:
    assert authority.VerifiedRunRegistration is engine.VerifiedRunRegistration
    assert authority.Experiment006RunAuthorityError is (
        engine.Experiment002RunAuthorityError
    )
    assert authority._RegisteredChildInputSnapshot is (
        engine._RegisteredChildInputSnapshot
    )
    assert authority._VerifiedState is engine._VerifiedState
    assert authority._MAX_CHILD_BUNDLE_BYTES == engine._MAX_CHILD_BUNDLE_BYTES
    assert authority._SEALED_CHILD_BUNDLE_FD == engine._SEALED_CHILD_BUNDLE_FD
    assert P6.sealed_child_memfd_target == authority._SEALED_CHILD_MEMFD_TARGET
    assert authority.__all__ == (
        "Experiment006RunAuthorityError",
        "VerifiedRunRegistration",
        "reverify_verified_run_registration",
        "verify_and_issue_experiment_006_run_registration",
        "verify_verified_run_registration",
    )
    for zero_argument_route in (
        authority.verify_and_issue_experiment_006_run_registration,
        authority._verify_and_issue_experiment_006_sealed_child_registration,
        authority._required_child_bundle_seals,
    ):
        assert inspect.signature(zero_argument_route).parameters == {}
    for one_argument_route in (
        authority._registered_child_input_snapshot,
        authority._create_sealed_experiment_006_child_bundle_fd,
        authority._verified_state,
    ):
        code = one_argument_route.__code__
        assert (code.co_posonlyargcount, code.co_argcount) == (1, 1)
        assert one_argument_route.__defaults__ is None
        assert one_argument_route.__kwdefaults__ is None
    assert "_ISSUANCE_COMPLETE" not in authority.__dict__
    assert "_ISSUED" not in authority.__dict__
    assert "_ISSUED_GUARDS" not in authority.__dict__


def test_facade_matches_the_exact_frozen_semantic_ast_and_docstring_contract() -> None:
    contract = _protocol()["admission_contract"]["facade_semantic_contract"]
    source = (ROOT / "src/falsewake/experiment_006_run_authority.py").read_text()
    tree = ast.parse(source, type_comments=True, feature_version=(3, 12))

    imports = [
        {
            "kind": "ImportFrom",
            "level": node.level,
            "module": node.module,
            "names": [
                {"asname": alias.asname, "name": alias.name} for alias in node.names
            ],
        }
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
    ]
    assert imports == contract["exact_imports"]

    assignments = {
        node.targets[0].id: ast.unparse(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id != "__all__"
    }
    assert assignments == contract["exact_aliases"]
    exports_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "__all__"
    )
    assert (
        list(ast.literal_eval(exports_node.value)) == contract["exact_public_exports"]
    )

    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    assert [node.name for node in functions] == contract["exact_route_order"]
    assert all(type(node) is ast.FunctionDef for node in functions)
    assert len(tree.body) == 1 + len(imports) + len(assignments) + 1 + len(functions)

    docstrings = {"module": ast.get_docstring(tree, clean=False)}
    for node in functions:
        docstrings[node.name] = ast.get_docstring(node, clean=False)
    assert docstrings == contract["docstring_truth"]["exact_docstrings"]

    frozen_routes = {route["name"]: route for route in contract["routes"]}
    forbidden_flags = (
        inspect.CO_ASYNC_GENERATOR | inspect.CO_COROUTINE | inspect.CO_GENERATOR
    )
    for node in functions:
        frozen = frozen_routes[node.name]
        assert frozen["node_kind"] == "FunctionDef"
        assert node.decorator_list == []
        assert not any(
            isinstance(descendant, (ast.Await, ast.Yield, ast.YieldFrom))
            for descendant in ast.walk(node)
        )
        signature = {
            "defaults": None
            if not node.args.defaults
            else [ast.unparse(default) for default in node.args.defaults],
            "keyword_defaults": None
            if not node.args.kw_defaults
            else [
                None if default is None else ast.unparse(default)
                for default in node.args.kw_defaults
            ],
            "keyword_only": [
                {"annotation": _unparse_required(arg.annotation), "name": arg.arg}
                for arg in node.args.kwonlyargs
            ],
            "positional_only": [
                {"annotation": _unparse_required(arg.annotation), "name": arg.arg}
                for arg in node.args.posonlyargs
            ],
            "positional_or_keyword": [
                {"annotation": _unparse_required(arg.annotation), "name": arg.arg}
                for arg in node.args.args
            ],
            "return_annotation": _unparse_required(node.returns),
            "variable_keyword": (
                None
                if node.args.kwarg is None
                else {
                    "annotation": _unparse_required(node.args.kwarg.annotation),
                    "name": node.args.kwarg.arg,
                }
            ),
            "variable_positional": (
                None
                if node.args.vararg is None
                else {
                    "annotation": _unparse_required(node.args.vararg.annotation),
                    "name": node.args.vararg.arg,
                }
            ),
        }
        assert signature == frozen["signature"]

        statements = node.body[1:]
        observed_statements: list[dict[str, object]] = []
        for statement in statements:
            if isinstance(statement, ast.Return):
                call = cast(ast.Call, statement.value)
                kind = "return_direct_call"
            else:
                call = cast(ast.Call, cast(ast.Expr, statement).value)
                kind = "expression_direct_call"
            observed_statements.append(
                {
                    "callee": ast.unparse(call.func),
                    "keyword_arguments": [],
                    "kind": kind,
                    "positional_arguments": [ast.unparse(arg) for arg in call.args],
                }
            )
            assert call.keywords == []
        assert observed_statements == frozen["statements"]

        route = cast(Callable[..., object], getattr(authority, node.name))
        assert inspect.isfunction(route)
        assert route.__code__.co_flags & forbidden_flags == 0


def test_facade_nonissuer_routes_bind_only_profile_006(
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
    monkeypatch.setattr(engine, "_verify_experiment_006_run_registration", verify)
    monkeypatch.setattr(engine, "_reverify_experiment_006_run_registration", reverify)
    monkeypatch.setattr(
        engine,
        "_create_sealed_experiment_006_child_bundle_fd",
        create_bundle,
    )

    authority.verify_verified_run_registration(registration)
    authority.reverify_verified_run_registration(registration)
    assert authority._registered_child_input_snapshot(registration) is child_snapshot
    assert authority._verified_state(registration) is verified_state
    assert authority._create_sealed_experiment_006_child_bundle_fd(registration) == 17
    assert observed_profiles == [P6, P6]
    assert observed_routes == ["verify", "reverify", "child-input", "bundle"]


def test_profile_006_binds_all_fourteen_execution_namespaces_exactly() -> None:
    assert len({id(profile) for profile in (P2, P3, P4, P5, P6)}) == 5
    assert dataclasses.astuple(P6) == (
        "006",
        "configs/experiment-006-run.json",
        b"falsewake-exp006-runtime-v1\0",
        b"FW6CHLD1",
        b"falsewake-exp006-source-bundle-v1\0",
        "falsewake-sealed://experiment-006/",
        "/memfd:falsewake-exp006-child-bundle (deleted)",
        "falsewake-exp006-child-bundle",
        b"FW6ACTV1",
        b"falsewake-exp006-activation-ticket-v1\0",
        "verify_and_issue_experiment_006_run_registration",
        "_verify_and_issue_experiment_006_sealed_child_registration",
        "src/falsewake/experiment_006_runner.py",
        "/home/ubuntu/gitcode/.t/falsewake-experiment-006-scratch",
    )
    assert engine._AUTHORITY_PROFILES[:5] == (P2, P3, P4, P5, P6)
    assert engine._require_authority_profile(P6) is P6


VERIFIERS: dict[
    engine._AuthorityProfile,
    Callable[[engine.VerifiedRunRegistration], None],
] = {
    P2: engine.verify_verified_run_registration,
    P3: authority_003.verify_verified_run_registration,
    P4: authority_004.verify_verified_run_registration,
    P5: authority_005.verify_verified_run_registration,
    P6: authority.verify_verified_run_registration,
}


@pytest.mark.parametrize(
    ("source_profile", "target_profile"),
    tuple(
        (source_profile, target_profile)
        for source_profile in (P2, P3, P4, P5, P6)
        for target_profile in (P2, P3, P4, P5, P6)
        if source_profile is not target_profile
    ),
    ids=tuple(
        f"{source.registration_experiment}-at-{target.registration_experiment}"
        for source in (P2, P3, P4, P5, P6)
        for target in (P2, P3, P4, P5, P6)
        if source is not target
    ),
)
def test_all_twenty_cross_profile_directions_are_rejected_before_reverification(
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


def test_p6_parser_and_invocation_dispatch_bind_only_profile_006(
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
    engine._parse_experiment_006_child_bundle(b"controlled")
    assert observed == [P6]

    executable = "/home/ubuntu/gitcode/.t/falsewake-venv/bin/python"
    invocation: dict[str, Any] = {
        "argv": [P6.runner_entrypoint],
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
            "TMPDIR": P6.scratch_root,
            "TZ": "UTC",
        },
        "orig_argv": [executable, "-I", "-S", "-B", P6.runner_entrypoint],
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
            P6,
        )
        is invocation
    )


def test_protocol_commit_exact_predecessor_and_frozen_files_are_bound() -> None:
    protocol_path = ROOT / P6_PROTOCOL_PATH
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    frozen = engine._expected_frozen_bindings(P6)

    assert engine._EXPERIMENT_006_PROTOCOL_INTRODUCTION_COMMIT == P6_PROTOCOL_COMMIT
    assert engine._EXPERIMENT_006_PROTOCOL_PATH == P6_PROTOCOL_PATH
    assert engine._EXPERIMENT_006_PROTOCOL_SHA256 == P6_PROTOCOL_SHA256
    assert hashlib.sha256(protocol_bytes).hexdigest() == P6_PROTOCOL_SHA256
    assert _committed_payload(P6_PROTOCOL_COMMIT, P6_PROTOCOL_PATH) == protocol_bytes
    assert frozen["execution_protocol"] == {
        "introduction_commit": P6_PROTOCOL_COMMIT,
        "path": P6_PROTOCOL_PATH,
        "sha256": P6_PROTOCOL_SHA256,
    }
    assert frozen["predecessor"] == protocol["predecessor"]
    assert engine._frozen_file_bindings(P6) == (
        *engine._frozen_file_bindings(P2),
        (P6_PROTOCOL_PATH, P6_PROTOCOL_SHA256),
        (P5_PROTOCOL_PATH, P5_PROTOCOL_SHA256),
        (P5_INCIDENT_PATH, P5_INCIDENT_SHA256),
    )
    predecessor = cast(dict[str, Any], frozen["predecessor"])
    assert predecessor["experiment"] == "005"
    assert predecessor["terminal"] is True
    assert predecessor["reuse_forbidden"] is True
    assert predecessor["incident"] == protocol["predecessor"]["incident"]
    frozen_paths = {path for path, _sha256 in engine._frozen_file_bindings(P6)}
    assert "configs/experiment-005-run.json" not in frozen_paths
    assert "reports/experiment-005-training.json" not in frozen_paths
    assert "configs/experiment-006-run.json" not in frozen_paths


def test_all_six_generic_dispatches_have_one_profile_006_arm() -> None:
    expected_fragments = {
        "_require_authority_profile": "elif profile is _EXPERIMENT_006_PROFILE:",
        "_verify_committed_repository": "elif profile is _EXPERIMENT_006_PROFILE:",
        "_require_invocation_registration": "_EXPERIMENT_006_PROFILE,",
        "_expected_frozen_bindings": "elif profile is _EXPERIMENT_006_PROFILE:",
        "_frozen_file_bindings": "if profile is _EXPERIMENT_006_PROFILE:",
    }
    for function_name, fragment in expected_fragments.items():
        source = inspect.getsource(getattr(engine, function_name))
        assert source.count(fragment) == 1
    profiles_source = inspect.getsource(engine._require_authority_profile)
    assert profiles_source.count("_EXPERIMENT_006_PROFILE") == 1
    assert engine._AUTHORITY_PROFILES.count(P6) == 1


@dataclass
class _HistoryTrace:
    git_calls: list[tuple[str, ...]]
    process_calls: list[tuple[str, ...]]
    blob_calls: list[tuple[str, str, int]]
    predecessor_history_calls: list[tuple[Path, str]]
    events: list[tuple[str, tuple[object, ...]]]


_HistoryFault = Literal[
    "p5-chain-delta",
    "p5-doc-hash",
    "p5-incident-hash",
    "p5-predecessor-topology",
    "p5-proof-hash",
    "p5-protocol-hash",
    "p5-source-hash",
    "p5-source-mode",
    "managed-output",
    "protocol-ancestry",
    "p6-protocol-delta",
    "p6-protocol-hash",
    "p6-protocol-mode",
    "p6-protocol-topology",
    "p6-source-mode",
    "source-delta",
    "transient-output",
]

P5_CHAIN = (
    (P5_PROTOCOL_COMMIT, P5_PROTOCOL_PARENT, P5_PROTOCOL_PATH),
    (P5_PROTOCOL_PROOF_COMMIT, P5_IMPLEMENTATION_COMMIT, P5_PROTOCOL_PROOF_PATH),
    (
        P5_PREFLIGHT_BOUNDARY_COMMIT,
        P5_PROTOCOL_PROOF_COMMIT,
        P5_PREFLIGHT_BOUNDARY_PATH,
    ),
    (P5_INCIDENT_COMMIT, P5_PREFLIGHT_BOUNDARY_COMMIT, P5_INCIDENT_PATH),
)
P5_FROZEN_BLOBS = (
    (
        P5_PROTOCOL_COMMIT,
        P5_PROTOCOL_PATH,
        P5_PROTOCOL_SHA256,
        engine._MAX_CONFIG_BYTES,
    ),
    (
        P5_PROTOCOL_PROOF_COMMIT,
        P5_PROTOCOL_PROOF_PATH,
        P5_PROTOCOL_PROOF_SHA256,
        engine._MAX_SOURCE_FILE_BYTES,
    ),
    (
        P5_PREFLIGHT_BOUNDARY_COMMIT,
        P5_PREFLIGHT_BOUNDARY_PATH,
        P5_PREFLIGHT_BOUNDARY_SHA256,
        engine._MAX_SOURCE_FILE_BYTES,
    ),
    (
        P5_INCIDENT_COMMIT,
        P5_INCIDENT_PATH,
        P5_INCIDENT_SHA256,
        engine._MAX_CONFIG_BYTES,
    ),
)


def _install_history_oracle(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fault: _HistoryFault | None = None,
) -> _HistoryTrace:
    trace = _HistoryTrace([], [], [], [], [])
    immutable_hashes = dict(P5_IMMUTABLE_BLOBS)

    def fake_git(root: Path, *args: str) -> bytes:
        assert root == ROOT
        trace.git_calls.append(args)
        trace.events.append(("git", args))
        if args == ("rev-list", "--parents", "-n", "1", P6_PROTOCOL_COMMIT):
            parent = "0" * 40 if fault == "p6-protocol-topology" else P5_INCIDENT_COMMIT
            return f"{P6_PROTOCOL_COMMIT} {parent}\n".encode("ascii")
        if args == (
            "diff-tree",
            "--no-ext-diff",
            "--no-textconv",
            "--no-commit-id",
            "--name-status",
            "-r",
            "-z",
            P5_INCIDENT_COMMIT,
            P6_PROTOCOL_COMMIT,
        ):
            if fault == "p6-protocol-delta":
                return b"A\0" + P6_PROTOCOL_PATH.encode() + b"\0A\0unexpected\0"
            return b"A\0" + P6_PROTOCOL_PATH.encode("ascii") + b"\0"
        if args == (
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--name-status",
            "-z",
            P5_INCIDENT_COMMIT,
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
                    args[2] in (P5_INCIDENT_COMMIT, SYNTHETIC_IMPLEMENTATION_COMMIT)
                    and args[4] in (*P4_MANAGED_OUTPUTS, *P5_MANAGED_OUTPUTS)
                )
                or (
                    args[2] == SYNTHETIC_IMPLEMENTATION_COMMIT
                    and args[4] in P6_MANAGED_OUTPUTS
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
        for index, (commit, parent, path) in enumerate(P5_CHAIN):
            if args == ("rev-list", "--parents", "-n", "1", commit):
                observed_parent = (
                    "0" * 40
                    if fault == "p5-predecessor-topology" and index == 1
                    else parent
                )
                return f"{commit} {observed_parent}\n".encode("ascii")
            if args == (
                "diff-tree",
                "--no-ext-diff",
                "--no-textconv",
                "--no-commit-id",
                "--name-status",
                "-r",
                "-z",
                parent,
                commit,
            ):
                suffix = (
                    b"A\0unexpected\0"
                    if fault == "p5-chain-delta" and index == 2
                    else b""
                )
                return b"A\0" + path.encode("ascii") + b"\0" + suffix
        if args == (
            "log",
            "--format=",
            "--name-only",
            f"{P5_INCIDENT_COMMIT}..{SYNTHETIC_IMPLEMENTATION_COMMIT}",
            "--",
            *P4_MANAGED_OUTPUTS,
            *P5_MANAGED_OUTPUTS,
            *P6_MANAGED_OUTPUTS,
        ):
            if fault == "transient-output":
                return b"reports/experiment-006-training.json\n"
            return b""
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
        trace.events.append(("git_process", args))
        return subprocess.CompletedProcess(
            args=("git", *args),
            returncode=1 if fault == "protocol-ancestry" else 0,
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
        trace.events.append(("committed_blob", (commit, path, maximum_bytes)))
        mode = "100644"
        if (commit, path) == (P6_PROTOCOL_COMMIT, P6_PROTOCOL_PATH):
            payload = _committed_payload(commit, path)
            if fault == "p6-protocol-hash":
                payload += b"\n"
            if fault == "p6-protocol-mode":
                mode = "100755"
        elif (commit, path) in {(row[0], row[1]) for row in P5_FROZEN_BLOBS}:
            payload = _committed_payload(commit, path)
            if fault == "p5-protocol-hash" and path == P5_PROTOCOL_PATH:
                payload += b"\n"
            if fault == "p5-proof-hash" and path == P5_PROTOCOL_PROOF_PATH:
                payload += b"\n"
            if fault == "p5-doc-hash" and path == P5_PREFLIGHT_BOUNDARY_PATH:
                payload += b"\n"
            if fault == "p5-incident-hash" and path == P5_INCIDENT_PATH:
                payload += b"\n"
        elif commit == P5_IMPLEMENTATION_COMMIT and path in immutable_hashes:
            payload = _committed_payload(commit, path)
            if (
                fault == "p5-source-hash"
                and path == "src/falsewake/experiment_005_coordinator.py"
            ):
                payload += b"\n"
            if (
                fault == "p5-source-mode"
                and path == "src/falsewake/experiment_005_coordinator.py"
            ):
                mode = "100755"
        elif (
            commit == SYNTHETIC_IMPLEMENTATION_COMMIT
            and path == "src/falsewake/experiment_002_run_authority.py"
        ):
            payload = b"synthetic shared authority\n"
        elif (
            commit == SYNTHETIC_IMPLEMENTATION_COMMIT and path in P6_ADDED_SOURCE_PATHS
        ):
            payload = f"synthetic {path}\n".encode("ascii")
            if fault == "p6-source-mode" and path == P6_ADDED_SOURCE_PATHS[0]:
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
        trace.events.append(("predecessor_history", (str(root), implementation_commit)))

    monkeypatch.setattr(engine, "_git", fake_git)
    monkeypatch.setattr(engine, "_git_process", fake_git_process)
    monkeypatch.setattr(engine, "_committed_blob", fake_committed_blob)
    monkeypatch.setattr(
        engine,
        "_require_experiment_005_implementation_history",
        fake_predecessor_history,
    )
    return trace


def test_p6_implementation_history_checks_exact_topology_modes_and_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = _install_history_oracle(monkeypatch)

    engine._require_experiment_006_implementation_history(
        ROOT,
        implementation_commit=SYNTHETIC_IMPLEMENTATION_COMMIT,
    )

    assert trace.predecessor_history_calls == [(ROOT, P5_IMPLEMENTATION_COMMIT)]
    assert trace.process_calls == [
        (
            "merge-base",
            "--is-ancestor",
            P6_PROTOCOL_COMMIT,
            SYNTHETIC_IMPLEMENTATION_COMMIT,
        ),
    ]
    assert len(trace.git_calls) == 47
    assert len(trace.blob_calls) == 21
    assert (
        len(trace.git_calls)
        + len(trace.process_calls)
        + len(trace.blob_calls)
        + len(trace.predecessor_history_calls)
        == 70
    )
    serialized_trace = json.dumps(
        trace.events,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    assert len(trace.events) == 70
    assert hashlib.sha256(serialized_trace).hexdigest() == (
        "1c62d5b26d586adb42a6f110642ff1a41c98db38e760f3c6ca9c5a2d3b3297c3"
    )
    assert {
        path
        for commit, path, _maximum_bytes in trace.blob_calls
        if commit == P5_IMPLEMENTATION_COMMIT
    } == {path for path, _sha256 in P5_IMMUTABLE_BLOBS}
    assert {
        path
        for commit, path, _maximum_bytes in trace.blob_calls
        if commit == SYNTHETIC_IMPLEMENTATION_COMMIT
    } == {
        "src/falsewake/experiment_002_run_authority.py",
        *P6_ADDED_SOURCE_PATHS,
    }
    assert len(_committed_payload(P5_INCIDENT_COMMIT, P5_INCIDENT_PATH)) == (
        P5_INCIDENT_BYTE_COUNT
    )
    for path in (*P4_MANAGED_OUTPUTS, *P5_MANAGED_OUTPUTS):
        assert (
            "ls-tree",
            "-z",
            P5_INCIDENT_COMMIT,
            "--",
            path,
        ) in trace.git_calls
    for path in (*P4_MANAGED_OUTPUTS, *P5_MANAGED_OUTPUTS, *P6_MANAGED_OUTPUTS):
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
        ("p6-protocol-topology", "protocol introduction topology changed"),
        ("p6-protocol-delta", "protocol introduction changed more"),
        ("p6-protocol-hash", "protocol introduction blob changed"),
        ("p6-protocol-mode", "protocol introduction blob changed"),
        ("protocol-ancestry", "does not descend from its protocol"),
        ("source-delta", "implementation source delta is not exact"),
        ("p5-predecessor-topology", "predecessor topology changed"),
        ("p5-chain-delta", "predecessor commit changed"),
        ("p5-protocol-hash", "predecessor blob changed"),
        ("p5-proof-hash", "predecessor blob changed"),
        ("p5-doc-hash", "predecessor blob changed"),
        ("p5-incident-hash", "predecessor blob changed"),
        ("p5-source-hash", "implementation source changed"),
        ("p5-source-mode", "implementation source changed"),
        ("managed-output", "predecessor history contains output"),
        ("p6-source-mode", "source mode is not exact"),
        ("transient-output", "transient managed output"),
    ),
)
def test_p6_implementation_history_rejects_each_frozen_dimension(
    monkeypatch: pytest.MonkeyPatch,
    fault: _HistoryFault,
    message: str,
) -> None:
    _install_history_oracle(monkeypatch, fault=fault)

    with pytest.raises(engine.Experiment002RunAuthorityError, match=message):
        engine._require_experiment_006_implementation_history(
            ROOT,
            implementation_commit=SYNTHETIC_IMPLEMENTATION_COMMIT,
        )


def test_p6_history_never_inspects_or_mutates_attempt_markers() -> None:
    source = inspect.getsource(engine._require_experiment_006_implementation_history)
    assert P5.scratch_root.removesuffix("-scratch") + "-attempt" not in source
    assert P6.scratch_root.removesuffix("-scratch") + "-attempt" not in source
    assert "lexists" not in source
    assert "unlink" not in source
    assert "open(" not in source
