from __future__ import annotations

import ast
import base64
import copy
import difflib
import hashlib
import inspect
import json
import os
import re
import stat
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs" / "experiment-006-execution.json"
PROTOCOL_COMMIT = "4adf524f9a3007d5382a851b3419e697d80ceeef"
PROTOCOL_PARENT = "594e2c491c057394f18860c8362a51190460645f"
PROTOCOL_SHA256 = "47ce6c6f11c3575944ef8c4ab1edcc0b42fdfa9bf1ede0318bfc5ea61fabf650"
PROTOCOL_GIT_BLOB = "9c7e11de8a3c81b74e32d56e8a429791376ea179"
PROTOCOL_BYTES = 129_719

ADMISSION_COMMIT = "438651010c4ef1de4012a570f3211b1d27bb1e3a"
ADMISSION_PARENT = "fe21f22feb4d258f291c4d8c8bfc9646f2618227"
REGISTRATION_COMMIT = "b73ecb8be3871092aad431a6c497771296c67d60"
REGISTRATION_PATH = "configs/experiment-006-run.json"
REGISTRATION_SHA256 = "53101c6424db09da01e23af4dc5956a57aa8a7e41226113782d4da1d822ab0b8"
REGISTRATION_BYTES = 14_811
INCIDENT_COMMIT = "fd6b98e6122d2884800b359a61440e9b07832204"
INCIDENT_PATH = "reports/experiment-006-execution-incident.json"
INCIDENT_SHA256 = "c5cf002ea876890e3b3769af2bcc850433d3d08ac1071c864ed69ee2172af77d"
INCIDENT_BYTES = 8_839

P5_PROTOCOL_COMMIT = "0b6bf2cac2d4f6a04d6ced596bf8f835660eb072"
P5_PROTOCOL_PARENT = "462aeba306a0612fd6d64884e323d72db3569a89"
P5_PROTOCOL_PATH = ROOT / "configs" / "experiment-005-execution.json"
P5_PROTOCOL_SHA256 = "824d1677cf8f75567cf1915381f0a9bbc778a74cba879ace80733c3ceb48e54d"
P5_IMPLEMENTATION_COMMIT = "04b634451d129faadadd921215123547e3467d65"
P5_PROTOCOL_PROOF_COMMIT = "1e20d6cf210a2d4a59cfc737a14eaccfc94a5b74"
P5_PROTOCOL_PROOF_SHA256 = (
    "cf6d4ee2fcba31de37774b591571abb2d86998cc0f2a3e99a40abe1af4af5414"
)
P5_PREFLIGHT_BOUNDARY_COMMIT = "9ae549ccd477b13b9531f7c44d854d34549fd9c3"
P5_PREFLIGHT_BOUNDARY_SHA256 = (
    "c9852a028f572e33b2379b98786f5c72e41ddae1300f03742ad977947a1dd600"
)
P5_INCIDENT_COMMIT = "594e2c491c057394f18860c8362a51190460645f"
P5_INCIDENT_SHA256 = "3de6fbbd6912bee47f3ef01f5371b57ae8d5942b6fc9915fc0ec243b9e931e3d"
P5_INCIDENT_BYTES = 6_612
P5_SHARED_AUTHORITY_SHA256 = (
    "937ad120dd5828cb4029cc1382df2a0a7c06aef14c1d37bfdb848d93993416af"
)

P4_MARKER = Path("/home/ubuntu/gitcode/.t/falsewake-experiment-004-attempt")
P5_MARKER = Path("/home/ubuntu/gitcode/.t/falsewake-experiment-005-attempt")
P6_MARKER = Path("/home/ubuntu/gitcode/.t/falsewake-experiment-006-attempt")

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

P5_SOURCE_BINDINGS: tuple[dict[str, str], ...] = (
    {
        "mode": "100644",
        "path": "src/falsewake/experiment_005_coordinator.py",
        "sha256": "4ad78de14ce12acdb63b4e103dd920ce878d7e60a06fa1c5893027d17401f812",
    },
    {
        "mode": "100644",
        "path": "src/falsewake/experiment_005_final_evidence.py",
        "sha256": "1c99757fc9b78e612891a9d49e7dc40c2d0b90599a40927e4cd41dff781f355d",
    },
    {
        "mode": "100644",
        "path": "src/falsewake/experiment_005_final_publication.py",
        "sha256": "84f7aba57f8ec0c096e5f18a03a8118800f53a9cfe4910c554bea677a9f79f07",
    },
    {
        "mode": "100644",
        "path": "src/falsewake/experiment_005_run_authority.py",
        "sha256": "87d18e901b4b814ce31c7c17b60b6aa54f837edc84ad3bb5ee1af386df006f5e",
    },
    {
        "mode": "100644",
        "path": "src/falsewake/experiment_005_runner.py",
        "sha256": "eeafface36baaabba5e1479750674884a069b41ccaeeb9dee9ccbc4b179a5336",
    },
    {
        "mode": "100644",
        "path": "src/falsewake/experiment_005_seed_worker.py",
        "sha256": "0840a62f0f4327eec7649d17ad334edfa6941d6146bff6fe50e2d693f0b80f96",
    },
    {
        "mode": "100644",
        "path": "src/falsewake/experiment_005_supervisor.py",
        "sha256": "7c8d58566c2b5e904ebf7f1e5fa1ab3b1c79c4387b38f296b08ef5ce7e312e71",
    },
)

ORDERED_SUBSTITUTIONS: tuple[dict[str, str], ...] = (
    {"candidate": "Experiment 006", "reference": "Experiment 005"},
    {"candidate": "Experiment006", "reference": "Experiment005"},
    {"candidate": "experiment_006", "reference": "experiment_005"},
    {"candidate": "experiment-006", "reference": "experiment-005"},
    {"candidate": "exp006", "reference": "exp005"},
    {"candidate": '"006"', "reference": '"005"'},
    {"candidate": "profile-006", "reference": "profile-005"},
    {"candidate": "profile ``006``", "reference": "profile ``005``"},
    {"candidate": "profile is ``006``", "reference": "profile is ``005``"},
    {
        "candidate": "_EXPERIMENT_006_PROFILE",
        "reference": "_EXPERIMENT_005_PROFILE",
    },
    {
        "candidate": "_EXPERIMENT_006_SEALED_CHILD_MEMFD_TARGET",
        "reference": "_EXPERIMENT_005_SEALED_CHILD_MEMFD_TARGET",
    },
    {"candidate": "FW6ACTV1", "reference": "FW5ACTV1"},
    {"candidate": "FW6CHLD1", "reference": "FW5CHLD1"},
)

SHARED_ADDED_SYMBOLS = (
    "_EXPERIMENT_006_CHILD_BUNDLE_MAGIC",
    "_EXPERIMENT_006_PROFILE",
    "_EXPERIMENT_006_PROTOCOL_INTRODUCTION_COMMIT",
    "_EXPERIMENT_006_PROTOCOL_PATH",
    "_EXPERIMENT_006_PROTOCOL_SHA256",
    "_EXPERIMENT_006_RUNTIME_FINGERPRINT_DOMAIN",
    "_EXPERIMENT_006_RUN_CONFIG_PATH",
    "_EXPERIMENT_006_SEALED_CHILD_MEMFD_TARGET",
    "_EXPERIMENT_006_SEALED_ORIGIN_PREFIX",
    "_EXPERIMENT_006_SOURCE_BUNDLE_DOMAIN",
    "_create_sealed_experiment_006_child_bundle_fd",
    "_parse_experiment_006_child_bundle",
    "_require_experiment_006_implementation_history",
    "_reverify_experiment_006_run_registration",
    "_verify_and_issue_experiment_006_run_registration",
    "_verify_and_issue_experiment_006_sealed_child_registration",
    "_verify_experiment_006_run_registration",
)
SHARED_DISPATCHES = (
    "_AUTHORITY_PROFILES",
    "_expected_frozen_bindings",
    "_frozen_file_bindings",
    "_require_authority_profile",
    "_require_invocation_registration",
    "_verify_committed_repository",
)

PROFILE_FIELDS = (
    "registration_experiment",
    "run_config_path",
    "runtime_fingerprint_domain",
    "child_bundle_magic",
    "source_bundle_domain",
    "sealed_origin_prefix",
    "sealed_child_memfd_target",
    "child_bundle_memfd_name",
    "activation_magic",
    "activation_ticket_domain",
    "issuer_route",
    "sealed_child_issuer_route",
    "runner_entrypoint",
    "scratch_root",
)
PROFILE_IDS = ("002", "003", "004", "005", "006")
CROSS_PROFILE_DIRECTIONS = tuple(
    f"{source}_to_{target}"
    for source in PROFILE_IDS
    for target in PROFILE_IDS
    if source != target
)
CROSS_PROFILE_REJECTIONS = tuple(
    f"experiment_{source}_capability_at_experiment_{target}_boundary"
    for source in PROFILE_IDS
    for target in PROFILE_IDS
    if source != target
)

EXPECTED_SOURCE_DELTA = (
    {"path": "src/falsewake/experiment_002_run_authority.py", "status": "M"},
    {"path": "src/falsewake/experiment_006_coordinator.py", "status": "A"},
    {"path": "src/falsewake/experiment_006_final_evidence.py", "status": "A"},
    {"path": "src/falsewake/experiment_006_final_publication.py", "status": "A"},
    {"path": "src/falsewake/experiment_006_run_authority.py", "status": "A"},
    {"path": "src/falsewake/experiment_006_runner.py", "status": "A"},
    {"path": "src/falsewake/experiment_006_seed_worker.py", "status": "A"},
    {"path": "src/falsewake/experiment_006_supervisor.py", "status": "A"},
)


def _reject_duplicate_key(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


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


def _path_snapshot(path: Path) -> tuple[tuple[int, ...], bytes] | None:
    if not os.path.lexists(path):
        return None
    before = path.lstat()
    assert stat.S_ISREG(before.st_mode)
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 16)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    named = path.lstat()
    frame = _stat_frame(before)
    assert _stat_frame(opened) == frame
    assert _stat_frame(after) == frame
    assert _stat_frame(named) == frame
    payload = b"".join(chunks)
    assert len(payload) == frame[6]
    return frame, payload


def _assert_terminal_p6_marker(snapshot: tuple[tuple[int, ...], bytes] | None) -> None:
    if snapshot is None:
        return
    frame, payload = snapshot
    assert stat.S_IMODE(frame[2]) == 0o444
    assert frame[3] == 1
    assert payload == b"falsewake-experiment-006-attempt-v1\n"
    assert _sha256(payload) == (
        "11cb9ce51db5b5153d3839ac1edad1cd63ae582009417abfea6a8562e319bfbe"
    )


def _protocol() -> dict[str, Any]:
    return _strict_json(PROTOCOL_PATH.read_bytes())


def _json_pointer(document: Any, pointer: str) -> Any:
    assert pointer.startswith("/")
    value = document
    for token in pointer[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            value = value[int(token)]
        else:
            assert isinstance(value, dict)
            value = value[token]
    return value


def _normalize_literals(source: str) -> str:
    for substitution in ORDERED_SUBSTITUTIONS:
        source = source.replace(
            substitution["candidate"],
            substitution["reference"],
        )
    return source


def _named_top_level_nodes(source: str) -> dict[str, ast.stmt]:
    result: dict[str, ast.stmt] = {}
    for node in ast.parse(source).body:
        name: str | None = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            name = node.targets[0].id
        elif isinstance(node, ast.TypeAlias) and isinstance(node.name, ast.Name):
            name = node.name.id
        if name is not None:
            assert name not in result
            result[name] = node
    return result


def _source_segment(source: str, node: ast.stmt) -> str:
    segment = ast.get_source_segment(source, node)
    assert segment is not None
    return segment


def _predecessor_regions(source: str) -> dict[str, tuple[int, int]]:
    body = ast.parse(source).body
    assignments = [
        (index, node)
        for index, node in enumerate(body)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id.startswith("_PREDECESSOR_")
    ]
    assert assignments
    first_index = assignments[0][0]
    last_index = assignments[-1][0]
    assert [index for index, _node in assignments] == list(
        range(first_index, last_index + 1)
    )
    assert last_index + 1 < len(body)
    functions = [
        (index, node)
        for index, node in enumerate(body)
        if isinstance(node, ast.FunctionDef) and node.name == "_predecessor_document"
    ]
    assert len(functions) == 1
    function_index, function = functions[0]
    assert function_index + 1 < len(body)
    return {
        "constants": (body[first_index].lineno, body[last_index + 1].lineno),
        "document": (function.lineno, body[function_index + 1].lineno),
    }


def _replace_predecessor_contract_regions(candidate: str, reference: str) -> str:
    candidate_regions = _predecessor_regions(candidate)
    reference_regions = _predecessor_regions(reference)
    candidate_lines = candidate.splitlines(keepends=True)
    reference_lines = reference.splitlines(keepends=True)
    for name in sorted(
        candidate_regions,
        key=lambda item: candidate_regions[item][0],
        reverse=True,
    ):
        candidate_first, candidate_last = candidate_regions[name]
        reference_first, reference_last = reference_regions[name]
        candidate_lines[candidate_first - 1 : candidate_last - 1] = reference_lines[
            reference_first - 1 : reference_last - 1
        ]
    return "".join(candidate_lines)


def _expression_dump(expression: ast.AST) -> str:
    return ast.dump(expression, include_attributes=False)


def _assert_expression(actual: ast.AST, expected: str) -> None:
    expected_node = ast.parse(expected, mode="eval").body
    assert _expression_dump(actual) == _expression_dump(expected_node)


def _argument_projection(argument: ast.arg) -> dict[str, str | None]:
    return {
        "annotation": (
            ast.unparse(argument.annotation)
            if argument.annotation is not None
            else None
        ),
        "name": argument.arg,
    }


def _assert_signature(
    function: ast.FunctionDef,
    expected: dict[str, Any],
) -> None:
    arguments = function.args
    assert [_argument_projection(item) for item in arguments.posonlyargs] == expected[
        "positional_only"
    ]
    assert [_argument_projection(item) for item in arguments.args] == expected[
        "positional_or_keyword"
    ]
    assert [_argument_projection(item) for item in arguments.kwonlyargs] == expected[
        "keyword_only"
    ]
    if arguments.defaults:
        actual_defaults: list[str] | None = [
            ast.unparse(item) for item in arguments.defaults
        ]
    else:
        actual_defaults = None
    if arguments.kw_defaults:
        actual_keyword_defaults: list[str | None] | None = [
            ast.unparse(item) if item is not None else None
            for item in arguments.kw_defaults
        ]
    else:
        actual_keyword_defaults = None
    assert actual_defaults == expected["defaults"]
    assert actual_keyword_defaults == expected["keyword_defaults"]
    assert (
        _argument_projection(arguments.vararg) if arguments.vararg is not None else None
    ) == expected["variable_positional"]
    assert (
        _argument_projection(arguments.kwarg) if arguments.kwarg is not None else None
    ) == expected["variable_keyword"]
    assert (
        ast.unparse(function.returns) if function.returns is not None else None
    ) == expected["return_annotation"]


def _body_without_docstring(function: ast.FunctionDef) -> list[ast.stmt]:
    body = list(function.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and type(body[0].value.value) is str
    ):
        body.pop(0)
    return body


def _assert_direct_statements(
    function: ast.FunctionDef,
    expected: list[dict[str, Any]],
) -> None:
    body = _body_without_docstring(function)
    assert len(body) == len(expected)
    for statement, contract in zip(body, expected, strict=True):
        if contract["kind"] == "return_direct_call":
            assert isinstance(statement, ast.Return)
            call = statement.value
        else:
            assert contract["kind"] == "expression_direct_call"
            assert isinstance(statement, ast.Expr)
            call = statement.value
        assert isinstance(call, ast.Call)
        assert ast.unparse(call.func) == contract["callee"]
        assert len(call.args) == len(contract["positional_arguments"])
        for actual, expression in zip(
            call.args,
            contract["positional_arguments"],
            strict=True,
        ):
            _assert_expression(actual, expression)
        expected_keywords = contract["keyword_arguments"]
        assert len(call.keywords) == len(expected_keywords)
        for actual_keyword, keyword in zip(
            call.keywords,
            expected_keywords,
            strict=True,
        ):
            assert actual_keyword.arg == keyword["name"]
            _assert_expression(actual_keyword.value, keyword["expression"])


class _RemoveProfile006(ast.NodeTransformer):
    _TEST = _expression_dump(
        ast.parse("profile is _EXPERIMENT_006_PROFILE", mode="eval").body
    )

    def visit_If(self, node: ast.If) -> ast.AST | list[ast.stmt] | None:
        visited = self.generic_visit(node)
        assert isinstance(visited, ast.If)
        if _expression_dump(visited.test) == self._TEST:
            return visited.orelse
        return visited

    def visit_Tuple(self, node: ast.Tuple) -> ast.AST:
        visited = self.generic_visit(node)
        assert isinstance(visited, ast.Tuple)
        visited.elts = [
            element
            for element in visited.elts
            if not (
                isinstance(element, ast.Name)
                and element.id == "_EXPERIMENT_006_PROFILE"
            )
        ]
        return visited


def _profile_if_nodes(node: ast.AST, profile: str) -> list[ast.If]:
    expected = _expression_dump(
        ast.parse(f"profile is _EXPERIMENT_{profile}_PROFILE", mode="eval").body
    )
    return [
        item
        for item in ast.walk(node)
        if isinstance(item, ast.If) and _expression_dump(item.test) == expected
    ]


def _nul_name_status(rows: list[dict[str, str]] | tuple[dict[str, str], ...]) -> bytes:
    return b"".join(
        row["status"].encode("ascii") + b"\0" + row["path"].encode("ascii") + b"\0"
        for row in rows
    )


def test_frozen_protocol_blob_topology_schema_and_lifecycle_are_exact() -> None:
    payload = PROTOCOL_PATH.read_bytes()
    assert len(payload) == PROTOCOL_BYTES
    assert _sha256(payload) == PROTOCOL_SHA256
    document = _strict_json(payload)
    assert _canonical(document) == payload
    assert tuple(document) == (
        "admission_contract",
        "attempt_contract",
        "authority_profile",
        "evidence_contract",
        "execution_delta",
        "experiment",
        "kind",
        "lifecycle",
        "namespace",
        "predecessor",
        "prefreeze_satisfiability",
        "schema_version",
        "scientific_inheritance",
        "verification_gate",
    )
    assert document["schema_version"] == 2
    assert document["experiment"] == "006"
    assert document["kind"] == "execution_only_preflight_protocol_recovery"

    assert (
        _git("rev-list", "--parents", "-n", "1", PROTOCOL_COMMIT)
        == f"{PROTOCOL_COMMIT} {PROTOCOL_PARENT}\n".encode()
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
            PROTOCOL_PARENT,
            PROTOCOL_COMMIT,
        )
        == b"A\0configs/experiment-006-execution.json\0"
    )
    assert (
        _git("ls-tree", PROTOCOL_COMMIT, "--", str(PROTOCOL_PATH.relative_to(ROOT)))
        == (
            f"100644 blob {PROTOCOL_GIT_BLOB}\tconfigs/experiment-006-execution.json\n"
        ).encode()
    )
    assert (
        _git(
            "cat-file", "-s", f"{PROTOCOL_COMMIT}:configs/experiment-006-execution.json"
        )
        == f"{PROTOCOL_BYTES}\n".encode()
    )
    assert (
        _git("show", f"{PROTOCOL_COMMIT}:configs/experiment-006-execution.json")
        == payload
    )

    lifecycle = document["lifecycle"]
    assert lifecycle["execution_protocol"] == {
        "committed_blob_verification_required": True,
        "introduction_commit_binding_required": True,
        "introduction_commit_must_be_direct_child_of": PROTOCOL_PARENT,
        "introduction_commit_must_precede_admission_commit": True,
        "only_change": "A configs/experiment-006-execution.json",
        "own_commit_value_in_document": "forbidden_self_reference",
        "own_sha256_value_in_document": "forbidden_self_reference",
        "path": "configs/experiment-006-execution.json",
        "required_mode": "100644",
        "run_config_binding_path": "frozen_bindings.execution_protocol",
        "run_config_sha256_binding_required": True,
    }
    assert lifecycle["registration_commit"] == {
        "execution_protocol_binding": "lifecycle.execution_protocol",
        "full_verification_gate_must_pass_before_creation": True,
        "only_change": "A configs/experiment-006-run.json",
        "own_commit_value_in_run_config": "forbidden_self_reference",
        "own_sha256_value_in_run_config": "forbidden_self_reference",
        "parent": "admission_commit",
        "required_mode": "100644",
    }
    assert lifecycle["attempt_limit"] == 1
    assert lifecycle["execution_config_authorizes_registered_training"] is False
    assert lifecycle["registration_config_authorizes_registered_training"] is True
    assert tuple(lifecycle["admission_commit"]["must_exclude"]) == P6_MANAGED_OUTPUTS
    assert tuple(lifecycle["self_reference_exclusions"]) == (
        "protocol_own_commit",
        "protocol_own_sha256",
        "future_admission_commit",
        "future_registration_commit",
        "future_source_sha256_values",
        "future_runner_sha256",
        "future_source_bundle_sha256",
        "future_incident_own_identity",
    )
    text = payload.decode("ascii")
    assert PROTOCOL_COMMIT not in text
    assert PROTOCOL_SHA256 not in text


def test_exact_p5_predecessor_topology_and_every_bound_blob_are_immutable() -> None:
    document = _protocol()
    predecessor = document["predecessor"]
    assert predecessor["experiment"] == "005"
    assert predecessor["scientific_protocol"] == "002"
    assert predecessor["terminal"] is True
    assert predecessor["reuse_forbidden"] is True
    assert predecessor["attempt"] == {
        "canonical_marker": str(P5_MARKER),
        "canonical_marker_present": False,
        "optimizer_updates": 0,
        "registered_attempt_consumed": False,
        "registered_authority_issuer_invoked": False,
        "registered_coordinator_invoked": False,
        "registered_invocation_count": 0,
        "registered_runner_invoked": False,
        "validation_examples": 0,
    }
    assert predecessor["execution_protocol"] == {
        "introduction_commit": P5_PROTOCOL_COMMIT,
        "mode": "100644",
        "path": "configs/experiment-005-execution.json",
        "sha256": P5_PROTOCOL_SHA256,
    }
    assert predecessor["implementation"] == {
        "commit": P5_IMPLEMENTATION_COMMIT,
        "production_sources": [
            {"path": item["path"], "sha256": item["sha256"]}
            for item in P5_SOURCE_BINDINGS
        ],
        "protocol_proof": {
            "commit": P5_PROTOCOL_PROOF_COMMIT,
            "path": "tests/test_experiment_005_protocol.py",
            "sha256": P5_PROTOCOL_PROOF_SHA256,
        },
        "shared_authority": {
            "path": "src/falsewake/experiment_002_run_authority.py",
            "sha256": P5_SHARED_AUTHORITY_SHA256,
        },
    }
    assert predecessor["preflight_boundary"] == {
        "commit": P5_PREFLIGHT_BOUNDARY_COMMIT,
        "mode": "100644",
        "parent": P5_PROTOCOL_PROOF_COMMIT,
        "path": "docs/experiment-005.md",
        "sha256": P5_PREFLIGHT_BOUNDARY_SHA256,
    }
    assert predecessor["incident"] == {
        "byte_count": P5_INCIDENT_BYTES,
        "commit": P5_INCIDENT_COMMIT,
        "mode": "100644",
        "parent": P5_PREFLIGHT_BOUNDARY_COMMIT,
        "path": "reports/experiment-005-preflight-incident.json",
        "sha256": P5_INCIDENT_SHA256,
    }
    assert tuple(predecessor["managed_outputs_absent"]) == P5_MANAGED_OUTPUTS
    assert predecessor["registration"] == {
        "path": "configs/experiment-005-run.json",
        "present": False,
    }
    assert predecessor["outcome"] == {
        "automatic_terminal_report_published": False,
        "checkpoint_reusable": False,
        "code": "facade_normalization_recipe_cannot_express_truthful_profile_005",
        "phase": "pre_registration_protocol_preflight",
        "reason": (
            "frozen_facade_normalization_cannot_express_the_truthful_"
            "profile_005_docstring"
        ),
        "status": "preflight_rejected",
    }
    assert predecessor["preflight_evidence"] == {
        "facade_candidate_normalized_sha256": (
            "9778140580667e95e120a4a8472d8b79ffb7034c3886c208fc3fd3c4da966c86"
        ),
        "facade_candidate_sha256": (
            "87d18e901b4b814ce31c7c17b60b6aa54f837edc84ad3bb5ee1af386df006f5e"
        ),
        "facade_reference_sha256": (
            "cacab13729f14bfdce10afa5b53b2a09116cf56abe21aee12ac9220a6ad2e5b5"
        ),
        "normalized_added_lines": 1,
        "normalized_deleted_lines": 1,
        "registration_gate": "not_run",
    }
    assert predecessor["topology"] == {
        "implementation_commit": P5_IMPLEMENTATION_COMMIT,
        "implementation_commit_is_ancestor_of_incident": True,
        "incident_commit": P5_INCIDENT_COMMIT,
        "incident_parent": P5_PREFLIGHT_BOUNDARY_COMMIT,
        "outcome_report_path": "reports/experiment-005-training.json",
        "outcome_report_present": False,
        "preflight_boundary_commit": P5_PREFLIGHT_BOUNDARY_COMMIT,
        "preflight_boundary_parent": P5_PROTOCOL_PROOF_COMMIT,
        "protocol_commit": P5_PROTOCOL_COMMIT,
        "protocol_parent": P5_PROTOCOL_PARENT,
        "protocol_proof_commit": P5_PROTOCOL_PROOF_COMMIT,
        "protocol_proof_parent": P5_IMPLEMENTATION_COMMIT,
        "registration_path": "configs/experiment-005-run.json",
        "registration_present": False,
    }

    chain = (
        (
            P5_PROTOCOL_COMMIT,
            P5_PROTOCOL_PARENT,
            "configs/experiment-005-execution.json",
            P5_PROTOCOL_SHA256,
            None,
        ),
        (
            P5_PROTOCOL_PROOF_COMMIT,
            P5_IMPLEMENTATION_COMMIT,
            "tests/test_experiment_005_protocol.py",
            P5_PROTOCOL_PROOF_SHA256,
            None,
        ),
        (
            P5_PREFLIGHT_BOUNDARY_COMMIT,
            P5_PROTOCOL_PROOF_COMMIT,
            "docs/experiment-005.md",
            P5_PREFLIGHT_BOUNDARY_SHA256,
            None,
        ),
        (
            P5_INCIDENT_COMMIT,
            P5_PREFLIGHT_BOUNDARY_COMMIT,
            "reports/experiment-005-preflight-incident.json",
            P5_INCIDENT_SHA256,
            P5_INCIDENT_BYTES,
        ),
    )
    for commit, parent, path, expected_sha256, expected_bytes in chain:
        assert (
            _git("rev-list", "--parents", "-n", "1", commit)
            == f"{commit} {parent}\n".encode()
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
                parent,
                commit,
            )
            == b"A\0" + path.encode() + b"\0"
        )
        payload = _git("show", f"{commit}:{path}")
        assert _sha256(payload) == expected_sha256
        if expected_bytes is not None:
            assert len(payload) == expected_bytes
        assert _git("ls-tree", commit, "--", path).startswith(b"100644 blob ")

    protocol_payload = P5_PROTOCOL_PATH.read_bytes()
    assert _sha256(protocol_payload) == P5_PROTOCOL_SHA256
    assert _git(
        "show", f"{P5_PROTOCOL_COMMIT}:configs/experiment-005-execution.json"
    ) == (protocol_payload)
    incident_payload = _git(
        "show",
        f"{P5_INCIDENT_COMMIT}:reports/experiment-005-preflight-incident.json",
    )
    incident = _strict_json(incident_payload)
    assert _canonical(incident) == incident_payload
    assert incident["attempt"] == predecessor["attempt"]
    assert incident["implementation"] == predecessor["implementation"]
    assert incident["status"] == "preflight_rejected"
    assert incident["incident"] == "pre_registration_protocol_rejection"

    baseline_path = "src/falsewake/experiment_002_run_authority.py"
    baseline = _git("show", f"{P5_IMPLEMENTATION_COMMIT}:{baseline_path}")
    assert _sha256(baseline) == P5_SHARED_AUTHORITY_SHA256
    for binding in P5_SOURCE_BINDINGS:
        payload = _git(
            "show",
            f"{P5_IMPLEMENTATION_COMMIT}:{binding['path']}",
        )
        assert _sha256(payload) == binding["sha256"]
        assert _git(
            "ls-tree",
            P5_IMPLEMENTATION_COMMIT,
            "--",
            binding["path"],
        ).startswith(b"100644 blob ")
    assert (
        subprocess.run(
            (
                "git",
                "merge-base",
                "--is-ancestor",
                P5_PROTOCOL_COMMIT,
                P5_IMPLEMENTATION_COMMIT,
            ),
            cwd=ROOT,
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )
    assert (
        subprocess.run(
            (
                "git",
                "merge-base",
                "--is-ancestor",
                P5_IMPLEMENTATION_COMMIT,
                P5_INCIDENT_COMMIT,
            ),
            cwd=ROOT,
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )


def test_actual_admission_source_delta_and_protocol_ancestry_are_exact() -> None:
    document = _protocol()
    source_delta = document["execution_delta"]["implementation_source_delta"]
    assert source_delta == {
        "base_commit": P5_INCIDENT_COMMIT,
        "name_status": list(EXPECTED_SOURCE_DELTA),
        "other_src_falsewake_changes": "forbidden",
    }
    head = _git("rev-parse", "HEAD").decode("ascii").strip()
    assert re.fullmatch(r"[0-9a-f]{40}", head)
    assert (
        subprocess.run(
            (
                "git",
                "merge-base",
                "--is-ancestor",
                PROTOCOL_COMMIT,
                head,
            ),
            cwd=ROOT,
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )
    assert _git(
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--name-status",
        "-z",
        P5_INCIDENT_COMMIT,
        head,
        "--",
        "src/falsewake",
    ) == _nul_name_status(EXPECTED_SOURCE_DELTA)
    assert tuple(document["execution_delta"]["production_additions"]) == tuple(
        row["path"] for row in EXPECTED_SOURCE_DELTA if row["status"] == "A"
    )


def test_all_six_nonfacade_recipes_restore_exact_p5_bytes() -> None:
    document = _protocol()
    parity = document["execution_delta"]["parity_contract"]
    assert parity["candidate"] == "experiment_006"
    assert parity["reference_profile"] == "experiment_005"
    assert parity["reference_commit"] == P5_IMPLEMENTATION_COMMIT
    assert parity["facade_global_literal_normalization"] == "forbidden"
    assert parity["facade_semantic_contract"] == (
        "admission_contract.facade_semantic_contract"
    )
    assert parity["unrestricted_textual_drift"] == (
        "forbidden_outside_the_facade_semantic_contract"
    )
    normalization = parity["normalization"]
    assert normalization["applies_to_facade"] is False
    assert normalization["direction"] == (
        "experiment_006_candidate_to_experiment_005_reference"
    )
    assert normalization["encoding"] == "UTF-8"
    assert tuple(normalization["ordered_literal_substitutions"]) == (
        ORDERED_SUBSTITUTIONS
    )
    assert tuple(normalization["scope"]) == (
        "src/falsewake/experiment_006_coordinator.py",
        "src/falsewake/experiment_006_final_evidence.py",
        "src/falsewake/experiment_006_final_publication.py",
        "src/falsewake/experiment_006_runner.py",
        "src/falsewake/experiment_006_seed_worker.py",
        "src/falsewake/experiment_006_supervisor.py",
    )
    assert normalization["unmatched_experiment_006_execution_literals"] == ("forbidden")

    recipes = parity["file_recipes"]
    assert len(recipes) == 7
    recipe_by_candidate = {item["candidate_path"]: item for item in recipes}
    facade_recipe = recipe_by_candidate["src/falsewake/experiment_006_run_authority.py"]
    assert facade_recipe == {
        "candidate_path": "src/falsewake/experiment_006_run_authority.py",
        "normalization": None,
        "reference_path": None,
        "reference_sha256": None,
        "semantic_contract": "admission_contract.facade_semantic_contract",
    }

    for suffix in ("coordinator", "runner", "seed_worker"):
        candidate_path = f"src/falsewake/experiment_006_{suffix}.py"
        reference_path = f"src/falsewake/experiment_005_{suffix}.py"
        recipe = recipe_by_candidate[candidate_path]
        assert recipe["reference_path"] == reference_path
        assert recipe["normalization"] == {
            "global_literals": (
                "parity_contract.normalization.ordered_literal_substitutions"
            ),
            "post_normalization_edits": [],
            "result": "byte_identical_to_reference",
        }
        reference = (ROOT / reference_path).read_text(encoding="utf-8")
        assert _sha256(reference.encode()) == recipe["reference_sha256"]
        candidate = _normalize_literals(
            (ROOT / candidate_path).read_text(encoding="utf-8")
        )
        assert candidate == reference

    for suffix in ("final_evidence", "final_publication"):
        candidate_path = f"src/falsewake/experiment_006_{suffix}.py"
        reference_path = f"src/falsewake/experiment_005_{suffix}.py"
        recipe = recipe_by_candidate[candidate_path]
        assert recipe["reference_path"] == reference_path
        assert recipe["normalization"]["result"] == "byte_identical_to_reference"
        edits = recipe["normalization"]["post_normalization_edits"]
        assert len(edits) == 1
        assert edits[0]["kind"] == "replace_exact_predecessor_contract_region_once"
        assert edits[0]["candidate_value_json_pointer"] == "/predecessor"
        assert edits[0]["reference_document_path"] == (
            "configs/experiment-005-execution.json"
        )
        assert edits[0]["reference_document_sha256"] == P5_PROTOCOL_SHA256
        assert edits[0]["reference_value_json_pointer"] == "/predecessor"
        reference = (ROOT / reference_path).read_text(encoding="utf-8")
        assert _sha256(reference.encode()) == recipe["reference_sha256"]
        candidate = _normalize_literals(
            (ROOT / candidate_path).read_text(encoding="utf-8")
        )
        candidate = _replace_predecessor_contract_regions(candidate, reference)
        assert candidate == reference

    runner = ROOT / "src" / "falsewake" / "experiment_006_runner.py"
    runner_sha256 = _sha256(runner.read_bytes())
    supervisor_path = ROOT / "src" / "falsewake" / "experiment_006_supervisor.py"
    supervisor = supervisor_path.read_text(encoding="utf-8")
    assignment = re.compile(r'(_RUNNER_SHA256: Final = \(\n    ")([0-9a-f]{64})("\n\))')
    matches = list(assignment.finditer(supervisor))
    assert len(matches) == 1
    assert matches[0].group(2) == runner_sha256
    supervisor_recipe = recipe_by_candidate[
        "src/falsewake/experiment_006_supervisor.py"
    ]
    reference_runner_sha256 = normalization["supervisor_runner_sha256_assignment"][
        "normalize_to_reference_value"
    ]
    assert reference_runner_sha256 == P5_SOURCE_BINDINGS[4]["sha256"]
    supervisor = assignment.sub(
        rf"\g<1>{reference_runner_sha256}\g<3>",
        supervisor,
        count=1,
    )
    supervisor = _normalize_literals(supervisor)
    supervisor_reference = (
        ROOT / "src" / "falsewake" / "experiment_005_supervisor.py"
    ).read_text(encoding="utf-8")
    assert (
        _sha256(supervisor_reference.encode()) == supervisor_recipe["reference_sha256"]
    )
    assert supervisor == supervisor_reference


def test_evidence_and_publication_bind_the_exact_p5_predecessor_value() -> None:
    predecessor = _protocol()["predecessor"]
    canonical_predecessor = json.dumps(
        predecessor,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    script = f"""
import json
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, {str(ROOT / "src")!r})
from falsewake import experiment_006_final_evidence as evidence
from falsewake import experiment_006_final_publication as publication

expected = json.loads({json.dumps(predecessor, sort_keys=True)!r})
assert evidence._predecessor_document() == expected
assert publication._predecessor_document() == expected
assert publication._PREDECESSOR_CANONICAL_JSON == {canonical_predecessor!r}
"""
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(
        (
            sys.executable,
            "-I",
            "-B",
            "-c",
            "import sys; exec(sys.stdin.read())",
        ),
        cwd=ROOT,
        env=environment,
        input=script,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_facade_has_the_exact_closed_semantic_ast_docs_and_sync_routes() -> None:
    contract = _protocol()["admission_contract"]["facade_semantic_contract"]
    path = ROOT / "src" / "falsewake" / "experiment_006_run_authority.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path), type_comments=True)

    docstrings = contract["docstring_truth"]
    assert (
        ast.get_docstring(tree, clean=False) == docstrings["exact_docstrings"]["module"]
    )
    assert tuple(docstrings["required_claim_scopes"]) == (
        "module",
        *contract["exact_route_order"],
    )
    assert set(docstrings["exact_docstrings"]) == set(
        docstrings["required_claim_scopes"]
    )
    assert docstrings["allowed_profile_claims"] == ["006"]
    assert tuple(docstrings["stale_profile_claims_forbidden"]) == (
        "002",
        "003",
        "004",
        "005",
    )
    assert docstrings["wording_otherwise_unrestricted"] is False

    body = tree.body
    assert (
        isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and type(body[0].value.value) is str
    )
    imports = [item for item in body if isinstance(item, ast.ImportFrom)]
    assert len(imports) == len(contract["exact_imports"]) == 2
    for actual, expected in zip(imports, contract["exact_imports"], strict=True):
        assert type(actual).__name__ == expected["kind"]
        assert actual.level == expected["level"]
        assert actual.module == expected["module"]
        assert [
            {"asname": alias.asname, "name": alias.name} for alias in actual.names
        ] == expected["names"]
        assert all(alias.name != "*" for alias in actual.names)

    assignments = [
        item
        for item in body
        if isinstance(item, ast.Assign)
        and len(item.targets) == 1
        and isinstance(item.targets[0], ast.Name)
    ]
    assert len(assignments) == len(contract["exact_aliases"]) + 1
    assignment_by_name = {
        item.targets[0].id: item
        for item in assignments
        if isinstance(item.targets[0], ast.Name)
    }
    assert set(assignment_by_name) == {*contract["exact_aliases"], "__all__"}
    for name, expression in contract["exact_aliases"].items():
        assert ast.unparse(assignment_by_name[name].value) == expression
    assert ast.literal_eval(assignment_by_name["__all__"].value) == tuple(
        contract["exact_public_exports"]
    )

    functions = [item for item in body if isinstance(item, ast.FunctionDef)]
    assert [item.name for item in functions] == contract["exact_route_order"]
    assert len(body) == 1 + len(imports) + len(assignments) + len(functions)
    function_by_name = {item.name: item for item in functions}
    assert len(function_by_name) == 8
    for route in contract["routes"]:
        function = function_by_name[route["name"]]
        assert type(function).__name__ == route["node_kind"] == "FunctionDef"
        assert function.decorator_list == []
        _assert_signature(function, route["signature"])
        _assert_direct_statements(function, route["statements"])
        assert (
            ast.get_docstring(function, clean=False)
            == docstrings["exact_docstrings"][function.name]
        )
        assert not any(
            isinstance(item, (ast.Await, ast.Yield, ast.YieldFrom))
            for item in ast.walk(function)
        )
        compiled = compile(
            ast.Module(body=[copy.deepcopy(function)], type_ignores=[]),
            filename=str(path),
            mode="exec",
            dont_inherit=True,
        )
        nested = next(
            item
            for item in compiled.co_consts
            if isinstance(item, type(compiled)) and item.co_name == function.name
        )
        assert not nested.co_flags & inspect.CO_COROUTINE
        assert not nested.co_flags & inspect.CO_GENERATOR
        assert not nested.co_flags & inspect.CO_ASYNC_GENERATOR


def test_shared_authority_restores_exact_p5_bytes_and_only_declared_ast() -> None:
    document = _protocol()
    recipe = document["execution_delta"]["shared_authority_recipe"]
    assert recipe["baseline_commit"] == P5_IMPLEMENTATION_COMMIT
    assert recipe["baseline_mode"] == "100644"
    assert recipe["baseline_path"] == ("src/falsewake/experiment_002_run_authority.py")
    assert recipe["baseline_sha256"] == P5_SHARED_AUTHORITY_SHA256
    assert tuple(recipe["added_symbols"]) == SHARED_ADDED_SYMBOLS
    assert tuple(recipe["modified_generic_dispatches"]) == SHARED_DISPATCHES

    path = recipe["baseline_path"]
    baseline = _git("show", f"{P5_IMPLEMENTATION_COMMIT}:{path}").decode("utf-8")
    candidate = (ROOT / path).read_text(encoding="utf-8")
    assert _sha256(baseline.encode()) == P5_SHARED_AUTHORITY_SHA256
    baseline_nodes = _named_top_level_nodes(baseline)
    candidate_nodes = _named_top_level_nodes(candidate)
    assert tuple(sorted(candidate_nodes.keys() - baseline_nodes.keys())) == tuple(
        sorted(SHARED_ADDED_SYMBOLS)
    )
    assert not baseline_nodes.keys() - candidate_nodes.keys()
    changed = tuple(
        sorted(
            name
            for name in baseline_nodes.keys() & candidate_nodes.keys()
            if _source_segment(baseline, baseline_nodes[name])
            != _source_segment(candidate, candidate_nodes[name])
        )
    )
    assert changed == tuple(sorted(SHARED_DISPATCHES))
    for name in baseline_nodes.keys() & candidate_nodes.keys():
        if name not in SHARED_DISPATCHES:
            assert _source_segment(baseline, baseline_nodes[name]) == _source_segment(
                candidate,
                candidate_nodes[name],
            )

    baseline_lines = baseline.splitlines(keepends=True)
    candidate_lines = candidate.splitlines(keepends=True)
    opcodes = difflib.SequenceMatcher(
        None,
        baseline_lines,
        candidate_lines,
        autojunk=False,
    ).get_opcodes()
    changed_opcodes = [item for item in opcodes if item[0] != "equal"]
    assert len(changed_opcodes) == 18
    assert {item[0] for item in changed_opcodes} == {"insert"}
    restored_lines: list[str] = []
    for tag, _base_first, _base_last, candidate_first, candidate_last in opcodes:
        if tag == "equal":
            restored_lines.extend(candidate_lines[candidate_first:candidate_last])
        else:
            assert tag == "insert"
    restored = "".join(restored_lines)
    assert restored == baseline
    assert _sha256(restored.encode()) == P5_SHARED_AUTHORITY_SHA256

    for name in SHARED_DISPATCHES:
        baseline_node = baseline_nodes[name]
        candidate_node = copy.deepcopy(candidate_nodes[name])
        profile_names = [
            item
            for item in ast.walk(candidate_node)
            if isinstance(item, ast.Name) and item.id == "_EXPERIMENT_006_PROFILE"
        ]
        assert len(profile_names) == 1
        stripped = _RemoveProfile006().visit(candidate_node)
        assert isinstance(stripped, ast.AST)
        assert _expression_dump(stripped) == _expression_dump(baseline_node)


def test_shared_ten_constants_and_six_simple_functions_are_exact() -> None:
    contract = _protocol()["execution_delta"]["shared_authority_recipe"][
        "positive_semantic_contract"
    ]
    source_path = ROOT / "src" / "falsewake" / "experiment_002_run_authority.py"
    source = source_path.read_text(encoding="utf-8")
    nodes = _named_top_level_nodes(source)
    constants = contract["constant_assignments"]
    assert contract["constant_assignment_count"] == len(constants) == 10
    assert {item["name"] for item in constants} == {
        name for name in SHARED_ADDED_SYMBOLS if name.startswith("_EXPERIMENT_006_")
    }
    for item in constants:
        node = nodes[item["name"]]
        assert isinstance(node, ast.AnnAssign)
        assert type(node).__name__ == item["node_kind"] == "AnnAssign"
        assert ast.unparse(node.annotation) == item["annotation"] == "Final"
        assert node.value is not None
        value = item["value"]
        if value.get("kind") == "post_freeze_literal":
            assert isinstance(node.value, ast.Constant)
            assert type(node.value.value) is str
            if item["name"].endswith("_INTRODUCTION_COMMIT"):
                assert node.value.value == PROTOCOL_COMMIT
                assert re.fullmatch(r"[0-9a-f]{40}", node.value.value)
            else:
                assert item["name"].endswith("_SHA256")
                assert node.value.value == PROTOCOL_SHA256
                assert re.fullmatch(r"[0-9a-f]{64}", node.value.value)
        elif value.get("kind") == "exact_keyword_call":
            assert isinstance(node.value, ast.Call)
            assert ast.unparse(node.value.func) == value["callee"]
            assert not node.value.args
            assert [keyword.arg for keyword in node.value.keywords] == [
                keyword["name"] for keyword in value["keyword_arguments"]
            ]
            for actual, expected in zip(
                node.value.keywords,
                value["keyword_arguments"],
                strict=True,
            ):
                _assert_expression(actual.value, expected["expression"])
        elif value["literal_kind"] == "bytes":
            assert isinstance(node.value, ast.Constant)
            assert type(node.value.value) is bytes
            assert node.value.value.hex() == value["utf8_hex"]
        else:
            assert value["literal_kind"] == "str"
            assert isinstance(node.value, ast.Constant)
            assert node.value.value == value["value"]

    simple_contracts = contract["function_contracts"]
    history_contract = contract["history_behavior"]
    assert len(simple_contracts) == 6
    truth = contract["function_docstring_truth"]
    assert set(truth["exact_docstrings"]) == {
        *(item["name"] for item in simple_contracts),
        history_contract["name"],
    }
    for item in simple_contracts:
        node = nodes[item["name"]]
        assert isinstance(node, ast.FunctionDef)
        assert type(node).__name__ == item["node_kind"] == "FunctionDef"
        assert node.decorator_list == []
        _assert_signature(node, item["signature"])
        _assert_direct_statements(node, item["statements"])
        assert (
            ast.get_docstring(node, clean=False) == truth["exact_docstrings"][node.name]
        )
        assert not any(
            isinstance(child, (ast.Await, ast.Yield, ast.YieldFrom))
            for child in ast.walk(node)
        )
        assert _body_without_docstring(node)
        assert not all(
            isinstance(statement, ast.Pass)
            for statement in _body_without_docstring(node)
        )

    history_node = nodes[history_contract["name"]]
    assert isinstance(history_node, ast.FunctionDef)
    _assert_signature(
        history_node,
        {
            **history_contract["signature"],
            "return_annotation": history_contract["return_annotation"],
        },
    )
    assert ast.get_docstring(history_node, clean=False) is None
    assert truth["exact_docstrings"][history_node.name] is None


def test_all_six_shared_dispatch_insertions_are_semantically_exact() -> None:
    document = _protocol()
    contract = document["execution_delta"]["shared_authority_recipe"][
        "positive_semantic_contract"
    ]
    insertions = contract["generic_dispatch_insertions"]
    assert contract["generic_dispatch_insertion_count"] == len(insertions) == 6
    assert tuple(item["target"] for item in insertions) == SHARED_DISPATCHES

    path = ROOT / "src" / "falsewake" / "experiment_002_run_authority.py"
    nodes = _named_top_level_nodes(path.read_text(encoding="utf-8"))
    profiles = nodes["_AUTHORITY_PROFILES"]
    assert isinstance(profiles, ast.AnnAssign)
    assert isinstance(profiles.value, ast.Tuple)
    assert [ast.unparse(item) for item in profiles.value.elts][-5:] == [
        "_EXPERIMENT_002_PROFILE",
        "_EXPERIMENT_003_PROFILE",
        "_EXPERIMENT_004_PROFILE",
        "_EXPERIMENT_005_PROFILE",
        "_EXPERIMENT_006_PROFILE",
    ]

    expected_bindings = nodes["_expected_frozen_bindings"]
    assert isinstance(expected_bindings, ast.FunctionDef)
    p5_branches = _profile_if_nodes(expected_bindings, "005")
    p6_branches = _profile_if_nodes(expected_bindings, "006")
    assert len(p5_branches) == len(p6_branches) == 1
    assert p6_branches[0].lineno > p5_branches[0].lineno
    assert len(p6_branches[0].body) == 2
    protocol_assignment, predecessor_assignment = p6_branches[0].body
    assert isinstance(protocol_assignment, ast.Assign)
    protocol_target = protocol_assignment.targets[0]
    assert isinstance(protocol_target, ast.Subscript)
    assert isinstance(protocol_target.value, ast.Name)
    assert protocol_target.value.id == "bindings"
    assert ast.literal_eval(protocol_target.slice) == "execution_protocol"
    assert isinstance(protocol_target.ctx, ast.Store)
    protocol_dict = protocol_assignment.value
    assert isinstance(protocol_dict, ast.Dict)
    assert all(key is not None for key in protocol_dict.keys)
    assert [ast.literal_eval(key) for key in protocol_dict.keys if key is not None] == [
        "introduction_commit",
        "path",
        "sha256",
    ]
    assert [ast.unparse(value) for value in protocol_dict.values] == [
        "_EXPERIMENT_006_PROTOCOL_INTRODUCTION_COMMIT",
        "_EXPERIMENT_006_PROTOCOL_PATH",
        "_EXPERIMENT_006_PROTOCOL_SHA256",
    ]
    assert isinstance(predecessor_assignment, ast.Assign)
    predecessor_target = predecessor_assignment.targets[0]
    assert isinstance(predecessor_target, ast.Subscript)
    assert isinstance(predecessor_target.value, ast.Name)
    assert predecessor_target.value.id == "bindings"
    assert ast.literal_eval(predecessor_target.slice) == "predecessor"
    assert isinstance(predecessor_target.ctx, ast.Store)
    assert ast.literal_eval(predecessor_assignment.value) == document["predecessor"]

    frozen = nodes["_frozen_file_bindings"]
    assert isinstance(frozen, ast.FunctionDef)
    p5_branches = _profile_if_nodes(frozen, "005")
    p6_branches = _profile_if_nodes(frozen, "006")
    assert len(p5_branches) == len(p6_branches) == 1
    assert p6_branches[0].lineno < p5_branches[0].lineno
    expected_frozen_branch = ast.parse(
        """
if profile is _EXPERIMENT_006_PROFILE:
    predecessor_execution_protocol = cast(
        dict[str, str],
        predecessor["execution_protocol"],
    )
    return (
        *bindings,
        (execution_protocol["path"], execution_protocol["sha256"]),
        (
            predecessor_execution_protocol["path"],
            predecessor_execution_protocol["sha256"],
        ),
        (predecessor_incident["path"], predecessor_incident["sha256"]),
    )
"""
    ).body[0]
    assert _expression_dump(p6_branches[0]) == _expression_dump(expected_frozen_branch)

    require_profile = nodes["_require_authority_profile"]
    assert isinstance(require_profile, ast.FunctionDef)
    p5_branches = _profile_if_nodes(require_profile, "005")
    p6_branches = _profile_if_nodes(require_profile, "006")
    assert len(p5_branches) == len(p6_branches) == 1
    assert p6_branches[0].lineno > p5_branches[0].lineno
    assert len(p6_branches[0].body) == 1
    expected_profile_values = document["authority_profile"]["profile_values"]
    actual_tuple = ast.literal_eval(
        p6_branches[0].body[0].value  # type: ignore[attr-defined]
    )
    assert actual_tuple == tuple(
        (
            expected_profile_values[field].encode("ascii")
            if field
            in {
                "runtime_fingerprint_domain",
                "child_bundle_magic",
                "source_bundle_domain",
                "activation_magic",
                "activation_ticket_domain",
            }
            else expected_profile_values[field]
        )
        for field in PROFILE_FIELDS
    )

    invocation = nodes["_require_invocation_registration"]
    assert isinstance(invocation, ast.FunctionDef)
    membership = [
        item
        for item in ast.walk(invocation)
        if isinstance(item, ast.Compare)
        and isinstance(item.left, ast.Name)
        and item.left.id == "profile"
        and len(item.ops) == 1
        and isinstance(item.ops[0], ast.In)
        and len(item.comparators) == 1
        and isinstance(item.comparators[0], ast.Tuple)
        and any(
            isinstance(element, ast.Name) and element.id == "_EXPERIMENT_006_PROFILE"
            for element in item.comparators[0].elts
        )
    ]
    assert len(membership) == 1
    membership_tuple = membership[0].comparators[0]
    assert isinstance(membership_tuple, ast.Tuple)
    assert [ast.unparse(item) for item in membership_tuple.elts] == [
        "_EXPERIMENT_003_PROFILE",
        "_EXPERIMENT_004_PROFILE",
        "_EXPERIMENT_005_PROFILE",
        "_EXPERIMENT_006_PROFILE",
    ]

    verify_repository = nodes["_verify_committed_repository"]
    assert isinstance(verify_repository, ast.FunctionDef)
    p5_branches = _profile_if_nodes(verify_repository, "005")
    p6_branches = _profile_if_nodes(verify_repository, "006")
    assert len(p5_branches) == len(p6_branches) == 1
    assert p6_branches[0].lineno > p5_branches[0].lineno
    expected_history_branch = ast.parse(
        """
if profile is _EXPERIMENT_006_PROFILE:
    _require_experiment_006_implementation_history(
        root,
        implementation_commit=document.implementation_commit,
    )
"""
    ).body[0]
    assert _expression_dump(p6_branches[0]) == _expression_dump(expected_history_branch)


def test_coordinator_ast_claims_all_fifteen_parent_routes_without_invocation() -> None:
    expected_rows = _protocol()["admission_contract"]["parent_route_matrix"]
    path = ROOT / "src" / "falsewake" / "experiment_006_coordinator.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    claims = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "claim_parent_operations"
    ]
    assert len(claims) == 1
    claim = claims[0]
    calls = sorted(
        (
            node
            for node in ast.walk(claim)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "require_function_identity"
        ),
        key=lambda node: (node.lineno, node.col_offset),
    )
    assert len(calls) == 13
    module_by_namespace = {
        "run_authority_namespace": "falsewake.experiment_006_run_authority",
        "supervisor_namespace": "falsewake.experiment_006_supervisor",
        "evidence_namespace": "falsewake.experiment_006_final_evidence",
        "publication_namespace": "falsewake.experiment_006_final_publication",
    }
    module_variable_by_namespace = {
        "run_authority_namespace": "run_authority_module_name",
        "supervisor_namespace": "supervisor_module_name",
        "evidence_namespace": "evidence_module_name",
        "publication_namespace": "publication_module_name",
    }
    static_rows: list[dict[str, object]] = []
    dynamic_calls: list[ast.Call] = []
    for call in calls:
        assert len(call.args) == 5
        assert not call.keywords
        lookup = call.args[0]
        assert isinstance(lookup, ast.Call)
        assert isinstance(lookup.func, ast.Name)
        assert lookup.func.id == "dictionary_get"
        assert len(lookup.args) == 2
        assert isinstance(lookup.args[0], ast.Name)
        namespace_name = lookup.args[0].id
        assert namespace_name in module_by_namespace
        module_node = call.args[2]
        assert isinstance(module_node, ast.Name)
        assert module_node.id == module_variable_by_namespace[namespace_name]
        binding_node = lookup.args[1]
        name_node = call.args[1]
        if isinstance(binding_node, ast.Name):
            assert binding_node.id == "binding_name"
            assert isinstance(name_node, ast.Name)
            assert name_node.id == "intrinsic_name"
            dynamic_calls.append(call)
            continue
        assert isinstance(binding_node, ast.Constant)
        assert type(binding_node.value) is str
        assert isinstance(name_node, ast.Constant)
        assert type(name_node.value) is str
        static_rows.append(
            {
                "binding": binding_node.value,
                "module": module_by_namespace[namespace_name],
                "name": name_node.value,
                "positional_only_arguments": ast.literal_eval(call.args[3]),
                "positional_or_keyword_arguments": ast.literal_eval(call.args[4]),
            }
        )
    assert len(dynamic_calls) == 1
    lifecycle_assignments = [
        node
        for node in ast.walk(claim)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "lifecycle_route_bindings"
    ]
    assert len(lifecycle_assignments) == 1
    lifecycle_bindings = ast.literal_eval(lifecycle_assignments[0].value)
    assert lifecycle_bindings == (
        ("_begin_registered_supervisor_child", "begin_registered_child"),
        (
            "_finish_registered_supervisor_child_success",
            "finish_registered_child_success",
        ),
        (
            "_finish_registered_supervisor_child_failure",
            "finish_registered_child_failure",
        ),
    )
    static_rows.extend(
        {
            "binding": binding,
            "module": "falsewake.experiment_006_supervisor",
            "name": name,
            "positional_only_arguments": 0,
            "positional_or_keyword_arguments": 0,
        }
        for binding, name in lifecycle_bindings
    )
    assert {row["binding"]: row for row in static_rows} == {
        row["binding"]: row for row in expected_rows
    }


def test_live_15_plus_4_routes_bindings_and_all_20_cross_directions() -> None:
    document = _protocol()
    assert not os.path.lexists(P4_MARKER)
    assert not os.path.lexists(P5_MARKER)
    marker_before = _path_snapshot(P6_MARKER)
    _assert_terminal_p6_marker(marker_before)
    admission = document["admission_contract"]
    assert len(admission["parent_route_matrix"]) == 15
    assert len(admission["wrapper_route_matrix"]) == 4
    routes = [
        *admission["parent_route_matrix"],
        *admission["wrapper_route_matrix"],
    ]
    required_modules = admission["pre_registration_proof"]["actual_modules_required"]
    profile_values = document["authority_profile"]["profile_values"]
    assert tuple(document["authority_profile"]["profile_binding_fields"]) == (
        PROFILE_FIELDS
    )
    assert tuple(document["authority_profile"]["cross_profile_rejections"]) == (
        CROSS_PROFILE_REJECTIONS
    )
    assert tuple(document["verification_gate"]["cross_profile_rejection"]) == (
        CROSS_PROFILE_DIRECTIONS
    )
    script = f"""
import importlib
import inspect
import json
import os
import stat
import sys
import weakref
from pathlib import Path
from types import FunctionType

sys.dont_write_bytecode = True
sys.path.insert(0, {str(ROOT / "src")!r})
p4_marker = {str(P4_MARKER)!r}
p5_marker = {str(P5_MARKER)!r}
p6_marker = {str(P6_MARKER)!r}
assert not os.path.lexists(p4_marker)
assert not os.path.lexists(p5_marker)

def marker_state(path):
    if not os.path.lexists(path):
        return None
    observed = os.lstat(path)
    assert stat.S_ISREG(observed.st_mode)
    return (
        observed.st_dev,
        observed.st_ino,
        observed.st_mode,
        observed.st_nlink,
        observed.st_uid,
        observed.st_gid,
        observed.st_size,
        observed.st_mtime_ns,
        observed.st_ctime_ns,
    )

p6_marker_before = marker_state(p6_marker)
route_spec = json.loads({json.dumps(routes, sort_keys=True)!r})
required_modules = json.loads({json.dumps(required_modules)!r})
modules = {{name: importlib.import_module(name) for name in required_modules}}
observed_routes = []
for expected in route_spec:
    route = vars(modules[expected["module"]]).get(expected["binding"])
    assert type(route) is FunctionType
    code = route.__code__
    observed_routes.append(
        {{
            "binding": expected["binding"],
            "module": route.__module__,
            "name": route.__name__,
            "positional_only_arguments": code.co_posonlyargcount,
            "positional_or_keyword_arguments": (
                code.co_argcount - code.co_posonlyargcount
            ),
            "keyword_only_arguments": code.co_kwonlyargcount,
            "defaults": route.__defaults__,
            "keyword_defaults": route.__kwdefaults__,
            "variable_positional_arguments": bool(code.co_flags & inspect.CO_VARARGS),
            "variable_keyword_arguments": bool(code.co_flags & inspect.CO_VARKEYWORDS),
            "coroutine": bool(code.co_flags & inspect.CO_COROUTINE),
            "generator": bool(code.co_flags & inspect.CO_GENERATOR),
            "async_generator": bool(code.co_flags & inspect.CO_ASYNC_GENERATOR),
        }}
    )

engine = importlib.import_module("falsewake.experiment_002_run_authority")
authorities = {{
    "002": engine,
    "003": importlib.import_module("falsewake.experiment_003_run_authority"),
    "004": importlib.import_module("falsewake.experiment_004_run_authority"),
    "005": importlib.import_module("falsewake.experiment_005_run_authority"),
    "006": importlib.import_module("falsewake.experiment_006_run_authority"),
}}
profiles = engine._AUTHORITY_PROFILES
assert tuple(item.registration_experiment for item in profiles) == {PROFILE_IDS!r}
profile = engine._EXPERIMENT_006_PROFILE
profile_fields = tuple(profile.__dataclass_fields__)
observed_profile = {{}}
for field in profile_fields:
    value = getattr(profile, field)
    if type(value) is bytes:
        value = value.decode("ascii")
    observed_profile[field] = value

bindings = engine._expected_frozen_bindings(profile)
assert bindings["execution_protocol"] == {{
    "introduction_commit": {PROTOCOL_COMMIT!r},
    "path": "configs/experiment-006-execution.json",
    "sha256": {PROTOCOL_SHA256!r},
}}
expected_predecessor = json.loads(
    {json.dumps(document["predecessor"], sort_keys=True)!r}
)
assert bindings["predecessor"] == expected_predecessor
assert engine._require_authority_profile(profile) is profile
assert engine._frozen_file_bindings(profile)[-3:] == (
    ("configs/experiment-006-execution.json", {PROTOCOL_SHA256!r}),
    ("configs/experiment-005-execution.json", {P5_PROTOCOL_SHA256!r}),
    ("reports/experiment-005-preflight-incident.json", {P5_INCIDENT_SHA256!r}),
)

engine._ISSUED = weakref.WeakKeyDictionary()
engine._ISSUED_GUARDS = weakref.WeakKeyDictionary()
engine._FAILED = weakref.WeakSet()
engine._ISSUANCE_COMPLETE = False
verifiers = {{
    key: module.verify_verified_run_registration
    for key, module in authorities.items()
}}

def forbidden_reverification(_registration):
    raise AssertionError("cross-profile route reached generic reverification")

engine.reverify_verified_run_registration = forbidden_reverification
directions = []
for source in profiles:
    for target in profiles:
        if source is target:
            continue
        engine._ISSUED = weakref.WeakKeyDictionary()
        engine._ISSUED_GUARDS = weakref.WeakKeyDictionary()
        engine._FAILED = weakref.WeakSet()
        engine._ISSUANCE_COMPLETE = False
        snapshot = engine._RepositorySnapshot(
            repository_root=Path({str(ROOT)!r}),
            head_commit="2" * 40,
            implementation_commit="1" * 40,
            registration_sha256="3" * 64,
            source_bundle_sha256="4" * 64,
            source_paths=("src/falsewake/experiment_002_run_authority.py",),
            profile=source,
        )
        capability = engine._issue_controlled_snapshot_for_tests(snapshot)
        try:
            verifiers[target.registration_experiment](capability)
        except engine.Experiment002RunAuthorityError as error:
            assert "different authority profile" in str(error)
        else:
            raise AssertionError("cross-profile capability was accepted")
        assert engine._ISSUED[capability].profile is source
        directions.append(
            source.registration_experiment
            + "_to_"
            + target.registration_experiment
        )

assert not os.path.lexists(p4_marker)
assert not os.path.lexists(p5_marker)
assert marker_state(p6_marker) == p6_marker_before
print(
    json.dumps(
        {{
            "routes": observed_routes,
            "profile_fields": profile_fields,
            "profile_values": observed_profile,
            "directions": directions,
        }},
        sort_keys=True,
    )
)
"""
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    completed = subprocess.run(
        (
            sys.executable,
            "-I",
            "-B",
            "-c",
            "import sys; exec(sys.stdin.read())",
        ),
        cwd=ROOT,
        env=environment,
        input=script,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    observed = json.loads(completed.stdout)
    expected_routes = [
        {
            **route,
            "keyword_only_arguments": 0,
            "defaults": None,
            "keyword_defaults": None,
            "variable_positional_arguments": False,
            "variable_keyword_arguments": False,
            "coroutine": False,
            "generator": False,
            "async_generator": False,
        }
        for route in routes
    ]
    assert observed["routes"] == expected_routes
    assert tuple(observed["profile_fields"]) == PROFILE_FIELDS
    assert observed["profile_values"] == profile_values
    assert tuple(observed["directions"]) == CROSS_PROFILE_DIRECTIONS
    assert not os.path.lexists(P4_MARKER)
    assert not os.path.lexists(P5_MARKER)
    marker_after = _path_snapshot(P6_MARKER)
    _assert_terminal_p6_marker(marker_after)
    assert marker_after == marker_before


def _history_trace_contract(document: dict[str, Any]) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        document["execution_delta"]["shared_authority_recipe"][
            "positive_semantic_contract"
        ]["history_behavior"]["fake_git_trace_contract"],
    )


def _history_plan(
    document: dict[str, Any], admission_commit: str
) -> list[dict[str, Any]]:
    trace = _history_trace_contract(document)
    events = trace["required_events"]
    assert [item["id"] for item in events] == [
        "p6_protocol_direct_parent",
        "p6_protocol_config_only_delta",
        "p6_protocol_blob",
        "p6_protocol_ancestry",
        "p6_exact_source_delta",
        "recursive_p5_history",
        "p5_protocol_proof_boundary_incident_topology",
        "p5_protocol_proof_doc_incident_blobs",
        "p5_shared_and_seven_production_blobs",
        "p4_p5_all_managed_output_absence",
        "p6_registration_and_managed_output_absence",
        "p6_shared_and_seven_source_modes",
        "p4_p5_p6_transient_managed_path_history_absence",
    ]
    event_by_id = {item["id"]: item for item in events}
    assert len(event_by_id) == 13
    plan: list[dict[str, Any]] = []

    def resolve(value: str) -> str:
        return {
            "$P6_PROTOCOL_COMMIT": PROTOCOL_COMMIT,
            "$P6_PROTOCOL_SHA256": PROTOCOL_SHA256,
            "$P6_ADMISSION_COMMIT": admission_commit,
        }.get(value, value)

    def add(
        event: str,
        callee: str,
        arguments: list[Any],
        *,
        keyword_arguments: dict[str, Any] | None = None,
        response: dict[str, Any],
    ) -> None:
        plan.append(
            {
                "event": event,
                "callee": callee,
                "arguments": arguments,
                "keyword_arguments": keyword_arguments or {},
                "response": response,
            }
        )

    def bytes_response(payload: bytes) -> dict[str, str]:
        return {
            "kind": "bytes",
            "base64": base64.b64encode(payload).decode("ascii"),
        }

    def blob_response(
        payload: bytes,
        *,
        mode: str = "100644",
    ) -> dict[str, str]:
        return {
            "kind": "blob",
            "mode": mode,
            "base64": base64.b64encode(payload).decode("ascii"),
        }

    event = event_by_id["p6_protocol_direct_parent"]
    expected_commit, expected_parent = (resolve(item) for item in event["expected"])
    add(
        event["id"],
        "_git",
        [
            str(ROOT),
            "rev-list",
            "--parents",
            "-n",
            "1",
            expected_commit,
        ],
        response=bytes_response(
            f"{expected_commit} {expected_parent}\n".encode("ascii")
        ),
    )

    event = event_by_id["p6_protocol_config_only_delta"]
    add(
        event["id"],
        "_git",
        [
            str(ROOT),
            "diff-tree",
            "--no-ext-diff",
            "--no-textconv",
            "--no-commit-id",
            "--name-status",
            "-r",
            "-z",
            resolve(event["from"]),
            resolve(event["to"]),
        ],
        response=bytes_response(_nul_name_status(event["expected"])),
    )

    event = event_by_id["p6_protocol_blob"]
    protocol_payload = _git("show", f"{PROTOCOL_COMMIT}:{event['path']}")
    assert _sha256(protocol_payload) == PROTOCOL_SHA256
    add(
        event["id"],
        "_committed_blob",
        [str(ROOT), resolve(event["treeish"]), event["path"]],
        keyword_arguments={"maximum_bytes": 1 << 20},
        response=blob_response(protocol_payload),
    )

    event = event_by_id["p6_protocol_ancestry"]
    add(
        event["id"],
        "_git_process",
        [
            str(ROOT),
            [
                "merge-base",
                "--is-ancestor",
                resolve(event["ancestor"]),
                resolve(event["descendant"]),
            ],
        ],
        keyword_arguments={"allowed_returncodes": [0, 1]},
        response={"kind": "process", "returncode": 0},
    )

    event = event_by_id["p6_exact_source_delta"]
    source_rows = _json_pointer(document, event["expected_json_pointer"])
    add(
        event["id"],
        "_git",
        [
            str(ROOT),
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--name-status",
            "-z",
            resolve(event["from"]),
            resolve(event["to"]),
            "--",
            event["pathspec"],
        ],
        response=bytes_response(_nul_name_status(source_rows)),
    )

    event = event_by_id["recursive_p5_history"]
    add(
        event["id"],
        "_require_experiment_005_implementation_history",
        [str(ROOT)],
        keyword_arguments=event["keyword_arguments"],
        response={"kind": "none"},
    )

    event = event_by_id["p5_protocol_proof_boundary_incident_topology"]
    for row in event["expected_chain"]:
        add(
            event["id"],
            "_git",
            [str(ROOT), "rev-list", "--parents", "-n", "1", row["commit"]],
            response=bytes_response(
                f"{row['commit']} {row['parent']}\n".encode("ascii")
            ),
        )
        status, path = row["only_change"].split(" ", maxsplit=1)
        add(
            event["id"],
            "_git",
            [
                str(ROOT),
                "diff-tree",
                "--no-ext-diff",
                "--no-textconv",
                "--no-commit-id",
                "--name-status",
                "-r",
                "-z",
                row["parent"],
                row["commit"],
            ],
            response=bytes_response(
                status.encode("ascii") + b"\0" + path.encode("ascii") + b"\0"
            ),
        )

    event = event_by_id["p5_protocol_proof_doc_incident_blobs"]
    for row in event["expected_blobs"]:
        payload = _git("show", f"{row['treeish']}:{row['path']}")
        assert _sha256(payload) == row["sha256"]
        if "byte_count" in row:
            assert len(payload) == row["byte_count"]
        maximum = 1 << 20 if row["maximum_bytes"] == "_MAX_CONFIG_BYTES" else 64 << 20
        add(
            event["id"],
            "_committed_blob",
            [str(ROOT), row["treeish"], row["path"]],
            keyword_arguments={"maximum_bytes": maximum},
            response=blob_response(payload, mode=row["mode"]),
        )

    event = event_by_id["p5_shared_and_seven_production_blobs"]
    for row in event["expected_blobs"]:
        treeish = event["treeish"]
        payload = _git("show", f"{treeish}:{row['path']}")
        assert _sha256(payload) == row["sha256"]
        add(
            event["id"],
            "_committed_blob",
            [str(ROOT), treeish, row["path"]],
            keyword_arguments={"maximum_bytes": 64 << 20},
            response=blob_response(payload, mode=row["mode"]),
        )

    event = event_by_id["p4_p5_all_managed_output_absence"]
    paths: list[str] = []
    for group in event["path_groups"]:
        if "paths" in group:
            paths.extend(group["paths"])
        else:
            paths.extend(_json_pointer(document, group["paths_json_pointer"]))
    for treeish in event["treeishes"]:
        for path in paths:
            add(
                event["id"],
                "_git",
                [str(ROOT), "ls-tree", "-z", resolve(treeish), "--", path],
                response=bytes_response(b""),
            )

    event = event_by_id["p6_registration_and_managed_output_absence"]
    for path in _json_pointer(document, event["paths_json_pointer"]):
        add(
            event["id"],
            "_git",
            [str(ROOT), "ls-tree", "-z", resolve(event["treeish"]), "--", path],
            response=bytes_response(b""),
        )

    event = event_by_id["p6_shared_and_seven_source_modes"]
    for path in event["paths"]:
        add(
            event["id"],
            "_committed_blob",
            [str(ROOT), resolve(event["treeish"]), path],
            keyword_arguments={"maximum_bytes": 64 << 20},
            response=blob_response(b"", mode=event["expected_mode"]),
        )

    event = event_by_id["p4_p5_p6_transient_managed_path_history_absence"]
    paths = []
    for group in event["path_groups"]:
        if "paths" in group:
            paths.extend(group["paths"])
        else:
            paths.extend(_json_pointer(document, group["paths_json_pointer"]))
    add(
        event["id"],
        "_git",
        [
            str(ROOT),
            "log",
            "--format=",
            "--name-only",
            (f"{resolve(event['from_exclusive'])}..{resolve(event['to_inclusive'])}"),
            "--",
            *paths,
        ],
        response=bytes_response(bytes.fromhex(event["expected_bytes_hex"])),
    )
    assert len(plan) == 70
    assert Counter(item["event"] for item in plan) == {
        "p6_protocol_direct_parent": 1,
        "p6_protocol_config_only_delta": 1,
        "p6_protocol_blob": 1,
        "p6_protocol_ancestry": 1,
        "p6_exact_source_delta": 1,
        "recursive_p5_history": 1,
        "p5_protocol_proof_boundary_incident_topology": 8,
        "p5_protocol_proof_doc_incident_blobs": 4,
        "p5_shared_and_seven_production_blobs": 8,
        "p4_p5_all_managed_output_absence": 28,
        "p6_registration_and_managed_output_absence": 7,
        "p6_shared_and_seven_source_modes": 8,
        "p4_p5_p6_transient_managed_path_history_absence": 1,
    }
    return plan


def test_history_function_has_closed_ast_effects_and_no_state_mutation() -> None:
    document = _protocol()
    positive = document["execution_delta"]["shared_authority_recipe"][
        "positive_semantic_contract"
    ]
    history = positive["history_behavior"]
    closure = history["ast_closure"]
    source_path = ROOT / "src" / "falsewake" / "experiment_002_run_authority.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    functions = [
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "_require_experiment_006_implementation_history"
    ]
    assert len(functions) == 1
    function = functions[0]
    assert not any(
        isinstance(
            item,
            (
                ast.AsyncFunctionDef,
                ast.Await,
                ast.Yield,
                ast.YieldFrom,
                ast.Import,
                ast.ImportFrom,
                ast.Global,
                ast.Nonlocal,
                ast.Delete,
                ast.AugAssign,
                ast.NamedExpr,
            ),
        )
        for item in ast.walk(function)
    )
    allowed_statements = set(closure["allowed_statement_kinds"])
    assert allowed_statements == {"Assign", "Expr", "For", "If", "Raise"}
    for item in ast.walk(function):
        if isinstance(item, ast.stmt) and item is not function:
            assert type(item).__name__ in allowed_statements
        if isinstance(item, ast.Assign):
            assert item.targets
            assert all(isinstance(target, ast.Name) for target in item.targets)
    assert not any(
        isinstance(item, (ast.Attribute, ast.Subscript))
        and isinstance(item.ctx, (ast.Store, ast.Del))
        for item in ast.walk(function)
    )
    assert not function.body or not isinstance(
        function.body[-1],
        (ast.Return, ast.Break, ast.Continue),
    )

    def call_target(call: ast.Call) -> str:
        if isinstance(call.func, ast.Name):
            return call.func.id
        assert isinstance(call.func, ast.Attribute)
        if call.func.attr == "encode":
            return "str.encode"
        if call.func.attr == "hexdigest":
            return "hash.hexdigest"
        if call.func.attr == "join" and isinstance(call.func.value, ast.Constant):
            assert type(call.func.value.value) is bytes
            return "bytes.join"
        return ast.unparse(call.func)

    effect_calls = set(closure["allowed_external_effect_call_targets"])
    pure_calls = set(closure["allowed_pure_call_targets"])
    observed_calls = [
        call_target(item) for item in ast.walk(function) if isinstance(item, ast.Call)
    ]
    assert set(observed_calls) <= effect_calls | pure_calls
    assert Counter(target for target in observed_calls if target in effect_calls) == {
        "_git": 8,
        "_committed_blob": 4,
        "_git_process": 1,
        "_require_experiment_005_implementation_history": 1,
    }
    assert closure["terminal_control_flow"] == (
        "fall_through_to_implicit_return_None_only"
    )


def test_history_strict_fake_trace_is_13_events_70_calls_and_fault_closed() -> None:
    document = _protocol()
    trace_contract = _history_trace_contract(document)
    assert trace_contract["event_cardinality"] == 13
    assert trace_contract["primitive_call_cardinality"] == 70
    assert trace_contract["event_order"] == "exactly_required_events_array_order"
    assert trace_contract["event_multiset"] == "exactly_required_events_once_each"
    assert tuple(trace_contract["instrumentation"]["intercepted_call_targets"]) == (
        "_committed_blob",
        "_git",
        "_git_process",
        "_require_experiment_005_implementation_history",
    )
    assert trace_contract["clean_trace_outcome"] == {
        "exception": None,
        "return_value": None,
        "status": "pass",
    }
    admission_commit = "a" * 40
    plan = _history_plan(document, admission_commit)
    fixture = {
        "admission_commit": admission_commit,
        "event_ids": [item["id"] for item in trace_contract["required_events"]],
        "plan": plan,
    }
    fixture_payload = json.dumps(
        fixture,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    script = f"""
import base64
import json
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, {str(ROOT / "src")!r})
from falsewake import experiment_002_run_authority as engine

fixture = json.loads({fixture_payload!r})

def normalized(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [normalized(item) for item in value]
    if isinstance(value, list):
        return [normalized(item) for item in value]
    if isinstance(value, dict):
        return {{key: normalized(item) for key, item in value.items()}}
    return value

class Adapter:
    def __init__(self, fault):
        self.fault = fault
        self.faulted = False
        self.index = 0
        self.trace = []

    def consume(self, callee, arguments, keyword_arguments):
        assert self.index < len(fixture["plan"])
        expected = fixture["plan"][self.index]
        self.index += 1
        assert expected["callee"] == callee
        assert expected["arguments"] == normalized(arguments)
        assert expected["keyword_arguments"] == normalized(keyword_arguments)
        self.trace.append((expected["event"], callee))
        if self.fault == expected["event"] and not self.faulted:
            self.faulted = True
            if callee == "_require_experiment_005_implementation_history":
                raise engine.Experiment002RunAuthorityError("injected recursive fault")
            if callee == "_git":
                return b"injected-fault"
            if callee == "_git_process":
                return subprocess.CompletedProcess((), 1, b"", b"")
            assert callee == "_committed_blob"
            response = expected["response"]
            return engine._TreeBlob(
                mode="100755",
                oid="0" * 40,
                payload=base64.b64decode(response["base64"]),
            )
        response = expected["response"]
        if response["kind"] == "none":
            return None
        if response["kind"] == "bytes":
            return base64.b64decode(response["base64"])
        if response["kind"] == "process":
            return subprocess.CompletedProcess(
                (),
                response["returncode"],
                b"",
                b"",
            )
        assert response["kind"] == "blob"
        return engine._TreeBlob(
            mode=response["mode"],
            oid="0" * 40,
            payload=base64.b64decode(response["base64"]),
        )

def run_case(fault):
    adapter = Adapter(fault)
    originals = (
        engine._committed_blob,
        engine._git,
        engine._git_process,
        engine._require_experiment_005_implementation_history,
    )

    def committed_blob(*args, **kwargs):
        return adapter.consume("_committed_blob", args, kwargs)

    def git(*args, **kwargs):
        return adapter.consume("_git", args, kwargs)

    def git_process(*args, **kwargs):
        return adapter.consume("_git_process", args, kwargs)

    def p5_history(*args, **kwargs):
        return adapter.consume(
            "_require_experiment_005_implementation_history",
            args,
            kwargs,
        )

    engine._committed_blob = committed_blob
    engine._git = git
    engine._git_process = git_process
    engine._require_experiment_005_implementation_history = p5_history
    caught = None
    result = object()
    try:
        result = engine._require_experiment_006_implementation_history(
            Path({str(ROOT)!r}),
            implementation_commit=fixture["admission_commit"],
        )
    except engine.Experiment002RunAuthorityError as error:
        caught = error
    finally:
        (
            engine._committed_blob,
            engine._git,
            engine._git_process,
            engine._require_experiment_005_implementation_history,
        ) = originals
    assert engine._committed_blob is originals[0]
    assert engine._git is originals[1]
    assert engine._git_process is originals[2]
    assert engine._require_experiment_005_implementation_history is originals[3]
    if fault is None:
        assert caught is None
        assert result is None
        assert adapter.index == 70
        assert len(adapter.trace) == 70
        assert not adapter.faulted
    else:
        assert caught is not None
        assert adapter.faulted
        assert adapter.trace[-1][0] == fault
    return adapter

clean = run_case(None)
assert [event for event, _callee in clean.trace] == [
    item["event"] for item in fixture["plan"]
]
fault_results = {{}}
for event_id in fixture["event_ids"]:
    adapter = run_case(event_id)
    fault_results[event_id] = {{
        "faulted": adapter.faulted,
        "last_event": adapter.trace[-1][0],
    }}
print(
    json.dumps(
        {{
            "clean_calls": len(clean.trace),
            "event_ids": fixture["event_ids"],
            "fault_results": fault_results,
        }},
        sort_keys=True,
    )
)
"""
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    completed = subprocess.run(
        (
            sys.executable,
            "-I",
            "-B",
            "-c",
            "import sys; exec(sys.stdin.read())",
        ),
        cwd=ROOT,
        env=environment,
        input=script,
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    observed = json.loads(completed.stdout)
    event_ids = [item["id"] for item in trace_contract["required_events"]]
    assert observed["clean_calls"] == 70
    assert observed["event_ids"] == event_ids
    assert set(observed["fault_results"]) == set(event_ids)
    assert all(
        item == {"faulted": True, "last_event": event_id}
        for event_id, item in observed["fault_results"].items()
    )


def test_every_live_route_has_exact_static_function_ast() -> None:
    admission = _protocol()["admission_contract"]
    routes = [
        *admission["parent_route_matrix"],
        *admission["wrapper_route_matrix"],
    ]
    trees: dict[str, ast.Module] = {}
    for route in routes:
        module = route["module"]
        if module not in trees:
            relative = Path("src") / Path(*module.split("."))
            path = ROOT / relative.with_suffix(".py")
            trees[module] = ast.parse(
                path.read_text(encoding="utf-8"),
                filename=str(path),
            )
        functions = [
            item
            for item in ast.walk(trees[module])
            if isinstance(item, ast.FunctionDef) and item.name == route["name"]
        ]
        assert len(functions) == 1
        function = functions[0]
        assert type(function).__name__ == "FunctionDef"
        assert not any(
            isinstance(item, (ast.Await, ast.Yield, ast.YieldFrom))
            for item in ast.walk(function)
        )
        assert len(function.args.posonlyargs) == route["positional_only_arguments"]
        assert len(function.args.args) == route["positional_or_keyword_arguments"]
        assert not function.args.kwonlyargs
        assert not function.args.defaults
        assert not function.args.kw_defaults
        assert function.args.vararg is None
        assert function.args.kwarg is None


def _synthetic_argument(item: dict[str, Any]) -> ast.arg:
    annotation = item["annotation"]
    return ast.arg(
        arg=item["name"],
        annotation=(
            ast.parse(annotation, mode="eval").body if annotation is not None else None
        ),
    )


def _synthetic_arguments(signature: dict[str, Any]) -> ast.arguments:
    defaults = signature["defaults"]
    keyword_defaults = signature["keyword_defaults"]
    variable_positional = signature["variable_positional"]
    variable_keyword = signature["variable_keyword"]
    return ast.arguments(
        posonlyargs=[
            _synthetic_argument(item) for item in signature["positional_only"]
        ],
        args=[_synthetic_argument(item) for item in signature["positional_or_keyword"]],
        vararg=(
            _synthetic_argument(variable_positional)
            if variable_positional is not None
            else None
        ),
        kwonlyargs=[_synthetic_argument(item) for item in signature["keyword_only"]],
        kw_defaults=(
            [
                ast.parse(item, mode="eval").body if item is not None else None
                for item in keyword_defaults
            ]
            if keyword_defaults is not None
            else []
        ),
        kwarg=(
            _synthetic_argument(variable_keyword)
            if variable_keyword is not None
            else None
        ),
        defaults=(
            [ast.parse(item, mode="eval").body for item in defaults]
            if defaults is not None
            else []
        ),
    )


def _synthetic_direct_statement(contract: dict[str, Any]) -> ast.stmt:
    call = ast.Call(
        func=ast.parse(contract["callee"], mode="eval").body,
        args=[
            ast.parse(item, mode="eval").body
            for item in contract["positional_arguments"]
        ],
        keywords=[
            ast.keyword(
                arg=item["name"],
                value=ast.parse(item["expression"], mode="eval").body,
            )
            for item in contract["keyword_arguments"]
        ],
    )
    if contract["kind"] == "return_direct_call":
        return ast.Return(value=call)
    assert contract["kind"] == "expression_direct_call"
    return ast.Expr(value=call)


def _synthetic_function(
    contract: dict[str, Any],
    docstring: str | None,
) -> ast.FunctionDef:
    body: list[ast.stmt] = []
    if docstring is not None:
        body.append(ast.Expr(value=ast.Constant(value=docstring)))
    body.extend(_synthetic_direct_statement(item) for item in contract["statements"])
    return ast.FunctionDef(
        name=contract["name"],
        args=_synthetic_arguments(contract["signature"]),
        body=body,
        decorator_list=[],
        returns=(
            ast.parse(contract["signature"]["return_annotation"], mode="eval").body
            if contract["signature"]["return_annotation"] is not None
            else None
        ),
        type_comment=None,
        type_params=[],
    )


def _synthesize_facade_fixture(document: dict[str, Any]) -> ast.Module:
    contract = document["admission_contract"]["facade_semantic_contract"]
    truth = contract["docstring_truth"]["exact_docstrings"]
    body: list[ast.stmt] = [
        ast.Expr(value=ast.Constant(value=truth["module"])),
    ]
    for item in contract["exact_imports"]:
        assert item["kind"] == "ImportFrom"
        body.append(
            ast.ImportFrom(
                module=item["module"],
                names=[
                    ast.alias(name=name["name"], asname=name["asname"])
                    for name in item["names"]
                ],
                level=item["level"],
            )
        )
    body.extend(
        ast.Assign(
            targets=[ast.Name(id=name, ctx=ast.Store())],
            value=ast.parse(expression, mode="eval").body,
        )
        for name, expression in contract["exact_aliases"].items()
    )
    body.append(
        ast.Assign(
            targets=[ast.Name(id="__all__", ctx=ast.Store())],
            value=ast.Tuple(
                elts=[
                    ast.Constant(value=item)
                    for item in contract["exact_public_exports"]
                ],
                ctx=ast.Load(),
            ),
        )
    )
    route_by_name = {item["name"]: item for item in contract["routes"]}
    body.extend(
        _synthetic_function(route_by_name[name], truth[name])
        for name in contract["exact_route_order"]
    )
    return ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))


def _synthesize_shared_fixture(document: dict[str, Any]) -> ast.Module:
    positive = document["execution_delta"]["shared_authority_recipe"][
        "positive_semantic_contract"
    ]
    truth = positive["function_docstring_truth"]["exact_docstrings"]
    body: list[ast.stmt] = [
        _synthetic_function(item, truth[item["name"]])
        for item in positive["function_contracts"]
    ]
    history = positive["history_behavior"]
    history_signature = {
        **history["signature"],
        "return_annotation": history["return_annotation"],
    }
    event_payloads = [
        json.dumps(
            item,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        for item in history["fake_git_trace_contract"]["required_events"]
    ]
    history_body: list[ast.stmt] = [
        ast.Assign(
            targets=[ast.Name(id="required_events", ctx=ast.Store())],
            value=ast.Tuple(
                elts=[ast.Constant(value=item) for item in event_payloads],
                ctx=ast.Load(),
            ),
        )
    ]
    history_docstring = truth[history["name"]]
    if history_docstring is not None:
        history_body.insert(
            0,
            ast.Expr(value=ast.Constant(value=history_docstring)),
        )
    body.append(
        ast.FunctionDef(
            name=history["name"],
            args=_synthetic_arguments(history_signature),
            body=history_body,
            decorator_list=[],
            returns=ast.parse(
                history_signature["return_annotation"],
                mode="eval",
            ).body,
            type_comment=None,
            type_params=[],
        )
    )
    return ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))


def _semantic_ast_projection(module: ast.Module) -> str:
    projected = copy.deepcopy(module)
    if (
        projected.body
        and isinstance(projected.body[0], ast.Expr)
        and isinstance(projected.body[0].value, ast.Constant)
        and type(projected.body[0].value.value) is str
    ):
        projected.body.pop(0)
    for item in ast.walk(projected):
        if (
            isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.body
            and isinstance(item.body[0], ast.Expr)
            and isinstance(item.body[0].value, ast.Constant)
            and type(item.body[0].value.value) is str
        ):
            item.body.pop(0)
    return ast.dump(projected, include_attributes=False)


def _synthetic_docstrings(module: ast.Module) -> dict[str, str | None]:
    result: dict[str, str | None] = {"module": ast.get_docstring(module, clean=False)}
    for item in module.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result[item.name] = ast.get_docstring(item, clean=False)
    return result


def _mutable_pointer_parent(document: Any, pointer: str) -> tuple[Any, str]:
    assert pointer.startswith("/")
    tokens = [
        item.replace("~1", "/").replace("~0", "~") for item in pointer[1:].split("/")
    ]
    assert tokens
    parent = document
    for token in tokens[:-1]:
        parent = parent[int(token)] if isinstance(parent, list) else parent[token]
    return parent, tokens[-1]


def _mutable_pointer_get(document: Any, pointer: str) -> Any:
    parent, token = _mutable_pointer_parent(document, pointer)
    return parent[int(token)] if isinstance(parent, list) else parent[token]


def _mutable_pointer_replace(document: Any, pointer: str, value: Any) -> None:
    parent, token = _mutable_pointer_parent(document, pointer)
    if isinstance(parent, list):
        parent[int(token)] = value
    else:
        parent[token] = value


def _mutable_pointer_pop(document: Any, pointer: str) -> Any:
    parent, token = _mutable_pointer_parent(document, pointer)
    return parent.pop(int(token)) if isinstance(parent, list) else parent.pop(token)


def _mutable_pointer_add(document: Any, pointer: str, value: Any) -> None:
    parent, token = _mutable_pointer_parent(document, pointer)
    if isinstance(parent, list):
        parent.insert(int(token), value)
    else:
        assert token not in parent
        parent[token] = value


def _synthetic_function_node(module: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        item
        for item in module.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    ]
    assert len(matches) == 1
    return matches[0]


def _replace_synthetic_docstring(
    module: ast.Module,
    *,
    name: str,
    before: str,
    after: str,
) -> None:
    function = _synthetic_function_node(module, name)
    assert ast.get_docstring(function, clean=False) == before
    first = function.body[0]
    assert isinstance(first, ast.Expr)
    assert isinstance(first.value, ast.Constant)
    first.value.value = after


def _append_synthetic_statement(
    function: ast.FunctionDef,
    statement: dict[str, Any],
) -> None:
    kind = statement["kind"]
    if kind == "Global":
        function.body.append(ast.Global(names=statement["names"]))
    elif kind == "Assign":
        function.body.append(
            ast.Assign(
                targets=[ast.Name(id=statement["target"], ctx=ast.Store())],
                value=ast.Constant(value=statement["value"]),
            )
        )
    elif kind == "Raise":
        function.body.append(
            ast.Raise(
                exc=ast.parse(statement["exception"], mode="eval").body,
                cause=None,
            )
        )
    else:
        assert kind == "ExprDirectCall"
        function.body.append(
            ast.Expr(
                value=ast.Call(
                    func=ast.parse(statement["callee"], mode="eval").body,
                    args=[
                        ast.parse(item, mode="eval").body
                        for item in statement["positional_arguments"]
                    ],
                    keywords=[],
                )
            )
        )


def _apply_json_replay_operator(
    document: dict[str, Any], operator: dict[str, Any]
) -> int:
    kind = operator["operator"]
    if kind == "json_replace":
        assert _mutable_pointer_get(document, operator["path"]) == operator["from"]
        _mutable_pointer_replace(
            document, operator["path"], copy.deepcopy(operator["to"])
        )
        return 1
    if kind == "json_remove_array_value":
        values = _mutable_pointer_get(document, operator["path"])
        assert isinstance(values, list)
        assert values.count(operator["value"]) == 1
        values.remove(operator["value"])
        return 1
    if kind == "json_append":
        values = _mutable_pointer_get(document, operator["path"])
        assert isinstance(values, list)
        values.append(copy.deepcopy(operator["value"]))
        return 1
    if kind == "json_add":
        generator = operator.get("value_generator")
        value = operator.get("value")
        if generator is not None:
            assert generator["kind"] == "repeat_ascii"
            value = generator["value"] * generator["count"]
        _mutable_pointer_add(document, operator["path"], value)
        return 1
    if kind == "json_move":
        value = _mutable_pointer_pop(document, operator["from"])
        _mutable_pointer_add(document, operator["to"], value)
        return 1
    if kind == "json_remove_first_matching_array_item":
        values = _mutable_pointer_get(document, operator["path"])
        assert isinstance(values, list)
        matches = [
            index
            for index, item in enumerate(values)
            if all(item[key] == value for key, value in operator["match"].items())
        ]
        assert len(matches) == 1
        values.pop(matches[0])
        return 1
    if kind == "json_duplicate_first_matching_array_item_after_itself":
        values = _mutable_pointer_get(document, operator["path"])
        assert isinstance(values, list)
        matches = [
            index
            for index, item in enumerate(values)
            if all(item[key] == value for key, value in operator["match"].items())
        ]
        assert len(matches) == 1
        index = matches[0]
        values.insert(index + 1, copy.deepcopy(values[index]))
        return 1
    if kind == "json_swap_array_items":
        values = _mutable_pointer_get(document, operator["path"])
        assert isinstance(values, list)
        first = operator["first_index"]
        second = operator["second_index"]
        assert first != second
        values[first], values[second] = values[second], values[first]
        return 1
    assert kind == "json_composite"
    assert operator["operations"]
    return sum(
        _apply_json_replay_operator(document, item) for item in operator["operations"]
    )


def _apply_prefreeze_operator(
    fixture: dict[str, Any],
    mutation_id: str,
    operator: dict[str, Any],
) -> int:
    kind = operator["operator"]
    if kind.startswith("json_"):
        return _apply_json_replay_operator(fixture["document"], operator)
    if kind == "replace_exact_function_docstring":
        target = (
            fixture["facade"] if operator["fixture"] == "facade" else fixture["shared"]
        )
        _replace_synthetic_docstring(
            target,
            name=operator["scope"],
            before=operator["from"],
            after=operator["to"],
        )
        return 1
    if kind == "replace_ast_node_kind":
        assert operator["fixture"] == "facade"
        assert operator["from"] == "FunctionDef"
        assert operator["to"] == "AsyncFunctionDef"
        module = fixture["facade"]
        function = _synthetic_function_node(module, operator["function"])
        index = module.body.index(function)
        replacement = ast.AsyncFunctionDef(
            name=function.name,
            args=function.args,
            body=function.body,
            decorator_list=function.decorator_list,
            returns=function.returns,
            type_comment=function.type_comment,
            type_params=function.type_params,
        )
        module.body[index] = replacement
        ast.fix_missing_locations(module)
        return 1
    if kind == "replace_ast_statement_kind":
        assert operator["fixture"] == "facade"
        function = _synthetic_function_node(
            fixture["facade"],
            operator["function"],
        )
        returns = [
            (index, item)
            for index, item in enumerate(function.body)
            if isinstance(item, ast.Return)
        ]
        assert len(returns) == 1
        index, original = returns[0]
        assert original.value is not None
        function.body[index] = ast.Expr(value=ast.Yield(value=original.value))
        ast.fix_missing_locations(fixture["facade"])
        return 1
    if kind == "replace_function_body":
        assert operator["fixture"] == "shared_added_functions"
        assert operator["statements"] == [{"kind": "Pass"}]
        function = _synthetic_function_node(
            fixture["shared"],
            operator["function"],
        )
        function.body = [ast.Pass()]
        ast.fix_missing_locations(fixture["shared"])
        return 1
    if kind in {"append_ast_statement", "append_ast_statements"}:
        assert operator["fixture"] == "shared_added_functions"
        function = _synthetic_function_node(
            fixture["shared"],
            operator["function"],
        )
        statements = (
            operator["statements"]
            if kind == "append_ast_statements"
            else [operator["statement"]]
        )
        for statement in statements:
            _append_synthetic_statement(function, statement)
        ast.fix_missing_locations(fixture["shared"])
        return len(statements)
    assert kind == "replace_event_result_then_force_success"
    assert operator["fixture"] == "fake_git_trace"
    event_id = operator["match"]["id"]
    assert event_id not in fixture["fault_overrides"]
    fixture["fault_overrides"][event_id] = copy.deepcopy(operator["trace_outcome"])
    fixture["fault_replacement"] = copy.deepcopy(operator["replacement_result"])
    fixture["fault_mutation_id"] = mutation_id
    return 1


def _pure_history_expansion(document: dict[str, Any]) -> list[tuple[str, int]]:
    events = _history_trace_contract(document)["required_events"]
    expansion: list[tuple[str, int]] = []
    for event in events:
        operation = event["operation"]
        if operation == "git_exact_commit_chain":
            count = 2 * len(event["expected_chain"])
        elif operation == "git_exact_committed_blobs":
            count = len(event["expected_blobs"])
        elif operation == "git_paths_absent":
            paths: list[str] = []
            if "paths_json_pointer" in event:
                paths.extend(_json_pointer(document, event["paths_json_pointer"]))
            for group in event.get("path_groups", []):
                if "paths" in group:
                    paths.extend(group["paths"])
                else:
                    paths.extend(_json_pointer(document, group["paths_json_pointer"]))
            tree_count = len(event.get("treeishes", [event.get("treeish")]))
            count = tree_count * len(paths)
        elif operation == "git_exact_path_modes":
            count = len(event["paths"])
        else:
            assert operation in {
                "exact_direct_call",
                "git_committed_blob",
                "git_diff_name_status",
                "git_diff_tree_name_status",
                "git_history_paths_absent",
                "git_merge_base_is_ancestor",
                "git_rev_list_parents",
            }
            count = 1
        expansion.extend((event["id"], index) for index in range(count))
    return expansion


def _prefreeze_replay_status(fixture: dict[str, Any]) -> str:
    document = fixture["document"]
    facade_expected = _synthesize_facade_fixture(document)
    shared_expected = _synthesize_shared_fixture(document)
    facade_semantic = _semantic_ast_projection(fixture["facade"]) == (
        _semantic_ast_projection(facade_expected)
    )
    shared_semantic = _semantic_ast_projection(fixture["shared"]) == (
        _semantic_ast_projection(shared_expected)
    )
    facade_docs = _synthetic_docstrings(fixture["facade"]) == (
        _synthetic_docstrings(facade_expected)
    )
    shared_docs = _synthetic_docstrings(fixture["shared"]) == (
        _synthetic_docstrings(shared_expected)
    )

    facade_contract = document["admission_contract"]["facade_semantic_contract"]
    shared_recipe = document["execution_delta"]["shared_authority_recipe"]
    positive = shared_recipe["positive_semantic_contract"]
    history_trace = positive["history_behavior"]["fake_git_trace_contract"]
    events = history_trace["required_events"]
    event_ids = [item["id"] for item in events]
    expected_event_ids = [
        "p6_protocol_direct_parent",
        "p6_protocol_config_only_delta",
        "p6_protocol_blob",
        "p6_protocol_ancestry",
        "p6_exact_source_delta",
        "recursive_p5_history",
        "p5_protocol_proof_boundary_incident_topology",
        "p5_protocol_proof_doc_incident_blobs",
        "p5_shared_and_seven_production_blobs",
        "p4_p5_all_managed_output_absence",
        "p6_registration_and_managed_output_absence",
        "p6_shared_and_seven_source_modes",
        "p4_p5_p6_transient_managed_path_history_absence",
    ]
    expansion = _pure_history_expansion(document)
    per_event = Counter(item[0] for item in expansion)
    derived_faults = {
        (event_id, mutation): "reject"
        for event_id in event_ids
        for mutation in ("remove_expected_result", "change_expected_result")
        if per_event[event_id] > 0
    }
    reported_faults = dict(derived_faults)
    for event_id, outcome in fixture["fault_overrides"].items():
        reported_faults[(event_id, "change_expected_result")] = outcome["status"]

    mutation_execution = document["prefreeze_satisfiability"]["mutation_execution"]
    config_checks = (
        facade_contract["top_level_policy"]
        == {
            "dynamic_getattr": "forbidden",
            "exception_wrappers": "forbidden",
            "extra_assignments": "forbidden",
            "extra_classes": "forbidden",
            "extra_executable_statements": "forbidden",
            "extra_functions": "forbidden",
            "star_imports": "forbidden",
        }
        and tuple(shared_recipe["modified_generic_dispatches"]) == SHARED_DISPATCHES
        and shared_recipe["restoration_contract"][
            "arbitrary_extra_logic_inside_allowed_dispatch"
        ]
        == "forbidden"
        and positive["generic_dispatch_insertions"][1]["inserted_ast"]["mutations"][1][
            "value_json_pointer"
        ]
        == "/predecessor"
        and document["lifecycle"]["registration_commit"]["parent"] == "admission_commit"
        and tuple(document["lifecycle"]["self_reference_exclusions"])
        == (
            "protocol_own_commit",
            "protocol_own_sha256",
            "future_admission_commit",
            "future_registration_commit",
            "future_source_sha256_values",
            "future_runner_sha256",
            "future_source_bundle_sha256",
            "future_incident_own_identity",
        )
        and set(document["prefreeze_satisfiability"]["result_digest"])
        == {"field_included_in_hashed_projection", "status"}
        and mutation_execution
        == {
            "application": "deep_copy_fixture_then_apply_exactly_one_operator",
            "config_fixture": (
                "this_exact_protocol_document_including_observed_result_fields"
            ),
            "operator_lookup": "mutation_operators[negative_mutations[].id]",
            "source_fixtures": {
                "facade": (
                    "synthesize_the_exact_admission_contract.facade_semantic_contract"
                ),
                "shared_added_functions": (
                    "synthesize_execution_delta.shared_authority_recipe."
                    "positive_semantic_contract"
                ),
            },
            "unexpected_mutations": "forbidden",
            "validator_outcome": (
                "the_mutated_fixture_must_match_negative_mutations[].expected"
            ),
        }
        and event_ids == expected_event_ids
        and history_trace["event_cardinality"] == len(events) == 13
        and history_trace["primitive_call_cardinality"] == len(expansion) == 70
        and per_event
        == {
            "p6_protocol_direct_parent": 1,
            "p6_protocol_config_only_delta": 1,
            "p6_protocol_blob": 1,
            "p6_protocol_ancestry": 1,
            "p6_exact_source_delta": 1,
            "recursive_p5_history": 1,
            "p5_protocol_proof_boundary_incident_topology": 8,
            "p5_protocol_proof_doc_incident_blobs": 4,
            "p5_shared_and_seven_production_blobs": 8,
            "p4_p5_all_managed_output_absence": 28,
            "p6_registration_and_managed_output_absence": 7,
            "p6_shared_and_seven_source_modes": 8,
            "p4_p5_p6_transient_managed_path_history_absence": 1,
        }
        and events[5]["keyword_arguments"]["implementation_commit"]
        == P5_IMPLEMENTATION_COMMIT
        and len(derived_faults) == 26
        and reported_faults == derived_faults
    )
    semantic_pass = config_checks and facade_semantic and shared_semantic
    docstring_pass = facade_docs and shared_docs
    if semantic_pass and docstring_pass:
        return "pass"
    if semantic_pass and not docstring_pass:
        return "semantic_ast_pass_and_docstring_truth_reject"
    return "reject"


def _replay_fixture_fingerprints(fixture: dict[str, Any]) -> dict[str, str]:
    fake_trace = {
        key: value
        for key, value in fixture.items()
        if key not in {"document", "facade", "shared"}
    }
    return {
        "document": json.dumps(
            fixture["document"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ),
        "facade": ast.dump(fixture["facade"], include_attributes=False),
        "shared_added_functions": ast.dump(
            fixture["shared"],
            include_attributes=False,
        ),
        "fake_git_trace": json.dumps(
            fake_trace,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ),
    }


def test_prefreeze_mutation_execution_replays_all_26_operators_in_memory() -> None:
    protocol_before = PROTOCOL_PATH.read_bytes()
    document = _strict_json(protocol_before)
    document_before = copy.deepcopy(document)
    loaded_before = frozenset(
        name for name in sys.modules if name.startswith("falsewake")
    )
    paths = (
        P4_MARKER,
        P5_MARKER,
        P6_MARKER,
        *(ROOT / item for item in P6_MANAGED_OUTPUTS),
        ROOT / INCIDENT_PATH,
    )
    filesystem_before = tuple(_path_snapshot(path) for path in paths)
    assert filesystem_before[0] is None
    assert filesystem_before[1] is None
    _assert_terminal_p6_marker(filesystem_before[2])
    assert filesystem_before[3] is not None
    assert filesystem_before[-1] is not None

    baseline_fixture: dict[str, Any] = {
        "document": copy.deepcopy(document),
        "facade": _synthesize_facade_fixture(document),
        "shared": _synthesize_shared_fixture(document),
        "fault_overrides": {},
    }
    assert _prefreeze_replay_status(baseline_fixture) == "pass"
    baseline_fingerprints = _replay_fixture_fingerprints(baseline_fixture)
    witness = document["prefreeze_satisfiability"]
    mutations = witness["negative_mutations"]
    operators = witness["mutation_operators"]
    assert [item["id"] for item in mutations] == list(
        dict.fromkeys(item["id"] for item in mutations)
    )
    observed: dict[str, str] = {}
    atomic_counts: dict[str, int] = {}
    for mutation in mutations:
        mutation_id = mutation["id"]
        fixture = copy.deepcopy(baseline_fixture)
        atomic_counts[mutation_id] = _apply_prefreeze_operator(
            fixture,
            mutation_id,
            operators[mutation_id],
        )
        fingerprints = _replay_fixture_fingerprints(fixture)
        changed_components = {
            name
            for name, fingerprint in fingerprints.items()
            if fingerprint != baseline_fingerprints[name]
        }
        expected_component = operators[mutation_id].get("fixture", "document")
        assert changed_components == {expected_component}
        observed[mutation_id] = _prefreeze_replay_status(fixture)
        assert observed[mutation_id] == mutation["expected"]
        assert document == document_before

    assert len(observed) == 26
    assert Counter(observed.values()) == {
        "reject": 23,
        "semantic_ast_pass_and_docstring_truth_reject": 3,
    }
    assert atomic_counts["protocol_or_witness_digest_self_reference_added"] == 2
    assert atomic_counts["shared_history_global_state_mutation_added"] == 2
    assert all(
        count == 1
        for mutation_id, count in atomic_counts.items()
        if mutation_id
        not in {
            "protocol_or_witness_digest_self_reference_added",
            "shared_history_global_state_mutation_added",
        }
    )
    assert _prefreeze_replay_status(baseline_fixture) == "pass"
    assert document == document_before
    assert PROTOCOL_PATH.read_bytes() == protocol_before
    assert _sha256(protocol_before) == PROTOCOL_SHA256
    assert (
        frozenset(name for name in sys.modules if name.startswith("falsewake"))
        == loaded_before
    )
    assert tuple(_path_snapshot(path) for path in paths) == filesystem_before


def test_prefreeze_witness_is_exactly_ten_positive_and_26_negative_cases() -> None:
    document = _protocol()
    witness = document["prefreeze_satisfiability"]
    assert witness["kind"] == "machine_checkable_synthetic_contract_v1"
    assert witness["claim_status"] == "pass"
    assert witness["execution_scope"] == {
        "canonical_marker_access": "forbidden",
        "coordinator_invocation": "forbidden",
        "filesystem_mutation": "forbidden",
        "issuer_invocation": "forbidden",
        "kind": "in_memory_non_importing_contract_evaluation",
        "registered_runner_invocation": "forbidden",
    }
    checklist = witness["checklist"]
    assert len(checklist) == 10
    assert len({item["id"] for item in checklist}) == 10
    assert all(item["observed"] == "pass" for item in checklist)
    assert all(item["status_before_protocol_commit"] == "pass" for item in checklist)
    for item in checklist:
        _json_pointer(document, item["contract_pointer"])

    mutations = witness["negative_mutations"]
    operators = witness["mutation_operators"]
    assert len(mutations) == len(operators) == 26
    assert len({item["id"] for item in mutations}) == 26
    assert set(operators) == {item["id"] for item in mutations}
    assert all(item["observed"] == item["expected"] for item in mutations)
    assert Counter(item["observed"] for item in mutations) == {
        "reject": 23,
        "semantic_ast_pass_and_docstring_truth_reject": 3,
    }
    fixture = witness["positive_fixture"]
    assert fixture == {
        "checklist_count": 10,
        "cross_profile_direction_count": 20,
        "facade_contract_pointer": ("/admission_contract/facade_semantic_contract"),
        "facade_docstring_scope_count": 9,
        "negative_mutation_count": 26,
        "negative_mutation_operator_count": 26,
        "p5_incident_sha256": P5_INCIDENT_SHA256,
        "p5_shared_authority_baseline_sha256": P5_SHARED_AUTHORITY_SHA256,
        "parent_route_count": 15,
        "shared_authority_added_symbol_count": 17,
        "shared_authority_constant_assignment_count": 10,
        "shared_authority_function_docstring_scope_count": 7,
        "shared_authority_history_event_count": 13,
        "shared_authority_history_primitive_call_count": 70,
        "shared_authority_modified_dispatch_count": 6,
        "shared_authority_simple_function_count": 6,
        "symbolic_commit_dag": [
            "experiment_005_incident",
            "experiment_006_protocol_freeze",
            "experiment_006_admission",
            "experiment_006_registration",
        ],
        "wrapper_route_count": 4,
    }
    assert witness["result_digest"] == {
        "field_included_in_hashed_projection": False,
        "status": "intentionally_absent_to_avoid_self_reference",
    }
    summary = witness["validation_summary"]
    assert summary["base_commit"] == P5_INCIDENT_COMMIT
    assert summary["status"] == "pass"
    assert summary["phase_1"] == {
        "assertion_count": 80,
        "id": "canonical_committed_bindings_lifecycle",
        "status": "pass",
    }
    assert summary["phase_2"] == {
        "assertion_count": 268,
        "id": "semantic_trace_and_negative_mutations",
        "negative_mutation_count": 26,
        "per_event_fault_case_count": 26,
        "primitive_trace_call_count": 70,
        "status": "pass",
    }
    assert summary["total_assertion_count"] == 348
    assert summary["canonical_marker_accesses"] == 0
    assert summary["coordinator_invocations"] == 0
    assert summary["issuer_invocations"] == 0
    assert summary["registered_runner_invocations"] == 0
    assert summary["production_module_imports"] == 0


def test_namespaces_and_historical_pre_registration_boundary_are_exact() -> None:
    document = _protocol()
    namespace = document["namespace"]
    assert namespace["attempt_marker"] == str(P6_MARKER)
    assert namespace["run_config"] == P6_MANAGED_OUTPUTS[0]
    assert tuple(namespace["managed_outputs"]["histories"]) == P6_MANAGED_OUTPUTS[2:5]
    assert namespace["managed_outputs"]["model"] == P6_MANAGED_OUTPUTS[1]
    assert (
        namespace["managed_outputs"]["selected_rerun_history"]
        == (P6_MANAGED_OUTPUTS[5])
    )
    assert namespace["managed_outputs"]["report"] == P6_MANAGED_OUTPUTS[6]
    assert namespace["authority_module"] == ("falsewake.experiment_006_run_authority")
    assert namespace["coordinator_module"] == ("falsewake.experiment_006_coordinator")
    assert namespace["runner"] == "src/falsewake/experiment_006_runner.py"
    assert namespace["scratch_root"] == (
        "/home/ubuntu/gitcode/.t/falsewake-experiment-006-scratch"
    )
    assert namespace["staging_root"] == (
        "/home/ubuntu/gitcode/.t/falsewake-experiment-006-staging"
    )

    p5_namespace = _strict_json(P5_PROTOCOL_PATH.read_bytes())["namespace"]
    for key in (
        "activation_thread_name",
        "attempt_marker",
        "authority_module",
        "child_bundle_memfd_name",
        "coordinator_module",
        "final_evidence_module",
        "final_publication_module",
        "launch_memfd_name",
        "run_config",
        "runner",
        "scratch_root",
        "sealed_child_memfd_target",
        "seed_worker_module",
        "staging_root",
        "supervisor_module",
        "temporary_publication_prefix",
    ):
        assert namespace[key] != p5_namespace[key]

    assert document["lifecycle"]["pre_registration_observations"] == {
        "registered_authority_issuer_invocations": 0,
        "registered_coordinator_invocations": 0,
        "registered_invocation_count": 0,
        "registered_optimizer_updates": 0,
        "registered_runner_invocations": 0,
        "registered_validation_examples": 0,
    }
    assert _git("rev-list", "--parents", "-n", "1", ADMISSION_COMMIT) == (
        f"{ADMISSION_COMMIT} {ADMISSION_PARENT}\n".encode("ascii")
    )
    assert not _git(
        "log",
        "--format=",
        "--name-only",
        f"{P5_INCIDENT_COMMIT}..{ADMISSION_COMMIT}",
        "--",
        *P4_MANAGED_OUTPUTS,
        *P5_MANAGED_OUTPUTS,
        *P6_MANAGED_OUTPUTS,
    )
    for path in (*P4_MANAGED_OUTPUTS, *P5_MANAGED_OUTPUTS, *P6_MANAGED_OUTPUTS):
        assert not _git("ls-tree", ADMISSION_COMMIT, "--", path)

    for path in document["execution_delta"]["production_additions"]:
        assert not _git("ls-tree", PROTOCOL_COMMIT, "--", path)
    frozen_shared = _git(
        "show",
        f"{PROTOCOL_COMMIT}:src/falsewake/experiment_002_run_authority.py",
    )
    assert b"_EXPERIMENT_006_PROFILE" not in frozen_shared
    assert not _git(
        "ls-tree",
        PROTOCOL_COMMIT,
        "--",
        "configs/experiment-006-run.json",
    )


def test_registration_and_terminal_incident_topology_are_exact() -> None:
    registration_payload = (ROOT / REGISTRATION_PATH).read_bytes()
    registration = _strict_json(registration_payload)
    assert len(registration_payload) == REGISTRATION_BYTES
    assert _sha256(registration_payload) == REGISTRATION_SHA256
    assert _canonical(registration) == registration_payload
    assert registration["implementation_commit"] == ADMISSION_COMMIT

    assert _git("rev-list", "--parents", "-n", "1", REGISTRATION_COMMIT) == (
        f"{REGISTRATION_COMMIT} {ADMISSION_COMMIT}\n".encode("ascii")
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
            ADMISSION_COMMIT,
            REGISTRATION_COMMIT,
        )
        == b"A\0" + REGISTRATION_PATH.encode("ascii") + b"\0"
    )
    registration_entry = _git(
        "ls-tree",
        REGISTRATION_COMMIT,
        "--",
        REGISTRATION_PATH,
    )
    assert registration_entry.startswith(b"100644 blob ")
    assert registration_entry.endswith(
        b"\t" + REGISTRATION_PATH.encode("ascii") + b"\n"
    )
    assert (
        _git("show", f"{REGISTRATION_COMMIT}:{REGISTRATION_PATH}")
        == registration_payload
    )

    incident_payload = (ROOT / INCIDENT_PATH).read_bytes()
    incident = _strict_json(incident_payload)
    assert len(incident_payload) == INCIDENT_BYTES
    assert _sha256(incident_payload) == INCIDENT_SHA256
    assert _canonical(incident) == incident_payload
    assert incident["experiment"] == "006"
    assert incident["incident"] == "registered_terminal_authority_failure"
    assert incident["outcome"]["automatic_terminal_report_published"] is False
    assert incident["diagnosis"]["root_cause_status"] == (
        "undetermined_from_retained_evidence"
    )

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
        == b"A\0" + INCIDENT_PATH.encode("ascii") + b"\0"
    )
    incident_entry = _git("ls-tree", INCIDENT_COMMIT, "--", INCIDENT_PATH)
    assert incident_entry.startswith(b"100644 blob ")
    assert incident_entry.endswith(b"\t" + INCIDENT_PATH.encode("ascii") + b"\n")
    assert _git("show", f"{INCIDENT_COMMIT}:{INCIDENT_PATH}") == incident_payload

    assert _git("ls-tree", INCIDENT_COMMIT, "--", REGISTRATION_PATH)
    for path in P6_MANAGED_OUTPUTS[1:]:
        assert not _git("ls-tree", INCIDENT_COMMIT, "--", path)


def test_terminal_outputs_and_external_marker_state_are_truthful() -> None:
    for path in (*P4_MANAGED_OUTPUTS, *P5_MANAGED_OUTPUTS):
        assert not os.path.lexists(ROOT / path)
    assert not os.path.lexists(P4_MARKER)
    assert not os.path.lexists(P5_MARKER)

    assert os.path.isfile(ROOT / REGISTRATION_PATH)
    assert os.path.isfile(ROOT / INCIDENT_PATH)
    for path in P6_MANAGED_OUTPUTS[1:]:
        assert not os.path.lexists(ROOT / path)

    marker_before = _path_snapshot(P6_MARKER)
    _assert_terminal_p6_marker(marker_before)
    marker_after = _path_snapshot(P6_MARKER)
    _assert_terminal_p6_marker(marker_after)
    assert marker_after == marker_before
