from __future__ import annotations

import ast
import difflib
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs" / "experiment-005-execution.json"
PROTOCOL_COMMIT = "0b6bf2cac2d4f6a04d6ced596bf8f835660eb072"
PROTOCOL_PARENT = "462aeba306a0612fd6d64884e323d72db3569a89"
PROTOCOL_SHA256 = "824d1677cf8f75567cf1915381f0a9bbc778a74cba879ace80733c3ceb48e54d"
PROTOCOL_GIT_BLOB = "38221d0314fd0b2cb0946c8ea24f6e74956a0c7a"
PROTOCOL_BYTES = 43_682

P4_PROTOCOL_PATH = ROOT / "configs" / "experiment-004-execution.json"
P4_PROTOCOL_COMMIT = "c7a8b3c493e05211ff7afa3a7abb974fc1c8b4e2"
P4_PROTOCOL_PARENT = "96f151d86ae060b402a0ef5d47de7ad8c1191c0a"
P4_PROTOCOL_SHA256 = "aba6c1eca84ad33c7751e4768a8ef630ba2c063720fd29938dd1b69abf945a46"
P4_PROTOCOL_GIT_BLOB = "c7474e5abf08634112939af72138ba913af5dff0"
P4_IMPLEMENTATION_COMMIT = "f81cd142885068c27767d6d728da04248fb1a470"
P4_PREFLIGHT_BOUNDARY_COMMIT = "55043086af773e613502f81685662e7fcc58f413"
P4_INCIDENT_COMMIT = "462aeba306a0612fd6d64884e323d72db3569a89"
P4_INCIDENT_PATH = ROOT / "reports" / "experiment-004-preflight-incident.json"
P4_INCIDENT_SHA256 = "d015b749acf6e98e87172372315c78ccfaad48b8ae6b3a29ae426890aa467f50"
P4_INCIDENT_GIT_BLOB = "2f23ef7d674930542911472c471f1f881b6004d0"
P4_PROTOCOL_PROOF_SHA256 = (
    "8539610615717b8b1fb5ff41876aef41a36e8ac7a8dbda4eeb7307174ae7a3a8"
)
P4_SHARED_AUTHORITY_SHA256 = (
    "52ce6039345b2159a4bf83ea294af2c3211cdcfbe87e9bc806133c67647b1a47"
)

P4_MARKER = Path("/home/ubuntu/gitcode/.t/falsewake-experiment-004-attempt")
P5_MARKER = Path("/home/ubuntu/gitcode/.t/falsewake-experiment-005-attempt")

P4_SOURCE_BINDINGS: tuple[dict[str, str], ...] = (
    {
        "mode": "100644",
        "path": "src/falsewake/experiment_004_coordinator.py",
        "sha256": "6e4df88440c361376ce51a3574a16bbec79ecbc6ff4dd1fcf0cf8be86eac1daa",
    },
    {
        "mode": "100644",
        "path": "src/falsewake/experiment_004_final_evidence.py",
        "sha256": "5fcff993e434c36d73c5e9b4f6d1e5954e8a54f2ce9fc17450ea615c5a4df5af",
    },
    {
        "mode": "100644",
        "path": "src/falsewake/experiment_004_final_publication.py",
        "sha256": "b862490f15e88819e815cfe6d7769ff125e21e59cdde4642a4a700e5f2eaad16",
    },
    {
        "mode": "100644",
        "path": "src/falsewake/experiment_004_run_authority.py",
        "sha256": "cacab13729f14bfdce10afa5b53b2a09116cf56abe21aee12ac9220a6ad2e5b5",
    },
    {
        "mode": "100644",
        "path": "src/falsewake/experiment_004_runner.py",
        "sha256": "76f19ec33c10955fa50d26b8d35d56aeb15d712b72b22a17e88933f46d24accf",
    },
    {
        "mode": "100644",
        "path": "src/falsewake/experiment_004_seed_worker.py",
        "sha256": "40cc3f0681006b03eac03f4747c60feb7a0ab42c102772afd6b90ef809929361",
    },
    {
        "mode": "100644",
        "path": "src/falsewake/experiment_004_supervisor.py",
        "sha256": "750371018aefd8dc3cdbffff20a60c3785680992b05d6246f052f37609a6c8c9",
    },
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

ORDERED_SUBSTITUTIONS: tuple[dict[str, str], ...] = (
    {"candidate": "Experiment 005", "reference": "Experiment 004"},
    {"candidate": "Experiment005", "reference": "Experiment004"},
    {"candidate": "experiment_005", "reference": "experiment_004"},
    {"candidate": "experiment-005", "reference": "experiment-004"},
    {"candidate": "exp005", "reference": "exp004"},
    {"candidate": '"005"', "reference": '"004"'},
    {"candidate": "profile-005", "reference": "profile-004"},
    {"candidate": "profile ``005``", "reference": "profile ``004``"},
    {
        "candidate": "_EXPERIMENT_005_PROFILE",
        "reference": "_EXPERIMENT_004_PROFILE",
    },
    {
        "candidate": "_EXPERIMENT_005_SEALED_CHILD_MEMFD_TARGET",
        "reference": "_EXPERIMENT_004_SEALED_CHILD_MEMFD_TARGET",
    },
    {"candidate": "FW5ACTV1", "reference": "FW4ACTV1"},
    {"candidate": "FW5CHLD1", "reference": "FW4CHLD1"},
)

SHARED_ADDED_SYMBOLS = (
    "_EXPERIMENT_005_CHILD_BUNDLE_MAGIC",
    "_EXPERIMENT_005_PROFILE",
    "_EXPERIMENT_005_PROTOCOL_INTRODUCTION_COMMIT",
    "_EXPERIMENT_005_PROTOCOL_PATH",
    "_EXPERIMENT_005_PROTOCOL_SHA256",
    "_EXPERIMENT_005_RUNTIME_FINGERPRINT_DOMAIN",
    "_EXPERIMENT_005_RUN_CONFIG_PATH",
    "_EXPERIMENT_005_SEALED_CHILD_MEMFD_TARGET",
    "_EXPERIMENT_005_SEALED_ORIGIN_PREFIX",
    "_EXPERIMENT_005_SOURCE_BUNDLE_DOMAIN",
    "_create_sealed_experiment_005_child_bundle_fd",
    "_parse_experiment_005_child_bundle",
    "_require_experiment_005_implementation_history",
    "_reverify_experiment_005_run_registration",
    "_verify_and_issue_experiment_005_run_registration",
    "_verify_and_issue_experiment_005_sealed_child_registration",
    "_verify_experiment_005_run_registration",
)
SHARED_DISPATCHES = (
    "_AUTHORITY_PROFILES",
    "_expected_frozen_bindings",
    "_frozen_file_bindings",
    "_require_authority_profile",
    "_require_invocation_registration",
    "_verify_committed_repository",
)

CROSS_PROFILE_REJECTIONS = (
    "experiment_002_capability_at_experiment_003_boundary",
    "experiment_002_capability_at_experiment_004_boundary",
    "experiment_002_capability_at_experiment_005_boundary",
    "experiment_003_capability_at_experiment_002_boundary",
    "experiment_003_capability_at_experiment_004_boundary",
    "experiment_003_capability_at_experiment_005_boundary",
    "experiment_004_capability_at_experiment_002_boundary",
    "experiment_004_capability_at_experiment_003_boundary",
    "experiment_004_capability_at_experiment_005_boundary",
    "experiment_005_capability_at_experiment_002_boundary",
    "experiment_005_capability_at_experiment_003_boundary",
    "experiment_005_capability_at_experiment_004_boundary",
)
CROSS_PROFILE_DIRECTIONS = (
    "002_to_003",
    "002_to_004",
    "002_to_005",
    "003_to_002",
    "003_to_004",
    "003_to_005",
    "004_to_002",
    "004_to_003",
    "004_to_005",
    "005_to_002",
    "005_to_003",
    "005_to_004",
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


def _protocol() -> dict[str, Any]:
    return _strict_json(PROTOCOL_PATH.read_bytes())


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
    tree = ast.parse(source)
    body = tree.body
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
        "constants": (
            body[first_index].lineno,
            body[last_index + 1].lineno,
        ),
        "document": (
            function.lineno,
            body[function_index + 1].lineno,
        ),
    }


def _replace_predecessor_contract_regions(
    candidate: str,
    reference: str,
) -> str:
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


def test_frozen_protocol_commit_blob_mode_size_and_topology_are_exact() -> None:
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
        "schema_version",
        "scientific_inheritance",
        "verification_gate",
    )
    assert document["schema_version"] == 1
    assert document["experiment"] == "005"
    assert document["kind"] == "execution_only_preflight_protocol_recovery"

    assert (
        _git(
            "rev-list",
            "--parents",
            "-n",
            "1",
            PROTOCOL_COMMIT,
        )
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
        == b"A\0configs/experiment-005-execution.json\0"
    )
    assert (
        _git(
            "ls-tree",
            PROTOCOL_COMMIT,
            "--",
            "configs/experiment-005-execution.json",
        )
        == (
            f"100644 blob {PROTOCOL_GIT_BLOB}\tconfigs/experiment-005-execution.json\n"
        ).encode()
    )
    assert (
        _git(
            "cat-file",
            "-s",
            f"{PROTOCOL_COMMIT}:configs/experiment-005-execution.json",
        )
        == f"{PROTOCOL_BYTES}\n".encode()
    )
    assert (
        _git(
            "show",
            f"{PROTOCOL_COMMIT}:configs/experiment-005-execution.json",
        )
        == payload
    )

    lifecycle = document["lifecycle"]["execution_protocol"]
    assert lifecycle == {
        "committed_blob_verification_required": True,
        "introduction_commit_binding_required": True,
        "introduction_commit_must_be_direct_child_of": PROTOCOL_PARENT,
        "introduction_commit_must_precede_implementation_commit": True,
        "only_change": "A configs/experiment-005-execution.json",
        "path": "configs/experiment-005-execution.json",
        "required_mode": "100644",
        "run_config_binding_path": "frozen_bindings.execution_protocol",
        "run_config_sha256_binding_required": True,
    }


def test_unregistered_p4_predecessor_and_incident_evidence_are_exact() -> None:
    predecessor = _protocol()["predecessor"]
    assert tuple(predecessor) == (
        "attempt",
        "execution_protocol",
        "experiment",
        "implementation",
        "incident",
        "managed_outputs_absent",
        "outcome",
        "registration",
        "reuse_forbidden",
        "scientific_protocol",
        "terminal",
        "topology",
    )
    assert predecessor["attempt"] == {
        "canonical_marker": str(P4_MARKER),
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
        "introduction_commit": P4_PROTOCOL_COMMIT,
        "mode": "100644",
        "path": "configs/experiment-004-execution.json",
        "sha256": P4_PROTOCOL_SHA256,
    }
    assert predecessor["experiment"] == "004"
    assert predecessor["implementation"] == {
        "commit": P4_IMPLEMENTATION_COMMIT,
        "production_sources": [
            {
                "path": binding["path"],
                "sha256": binding["sha256"],
            }
            for binding in P4_SOURCE_BINDINGS
        ],
        "protocol_proof": {
            "path": "tests/test_experiment_004_protocol.py",
            "sha256": P4_PROTOCOL_PROOF_SHA256,
        },
        "shared_authority": {
            "path": "src/falsewake/experiment_002_run_authority.py",
            "sha256": P4_SHARED_AUTHORITY_SHA256,
        },
    }
    assert predecessor["incident"] == {
        "commit": P4_INCIDENT_COMMIT,
        "mode": "100644",
        "path": "reports/experiment-004-preflight-incident.json",
        "sha256": P4_INCIDENT_SHA256,
    }
    assert tuple(predecessor["managed_outputs_absent"]) == P4_MANAGED_OUTPUTS
    assert predecessor["outcome"] == {
        "automatic_terminal_report_published": False,
        "checkpoint_reusable": False,
        "code": "pre_registration_protocol_rejection",
        "phase": "pre_registration_protocol_preflight",
        "reason": "frozen_protocol_contract_is_internally_unsatisfiable",
        "status": "preflight_rejected",
    }
    assert predecessor["registration"] == {
        "path": "configs/experiment-004-run.json",
        "present": False,
    }
    assert predecessor["reuse_forbidden"] is True
    assert predecessor["scientific_protocol"] == "002"
    assert predecessor["terminal"] is True
    assert predecessor["topology"] == {
        "implementation_commit": P4_IMPLEMENTATION_COMMIT,
        "implementation_commit_is_ancestor_of_incident": True,
        "incident_commit": P4_INCIDENT_COMMIT,
        "incident_parent": P4_PREFLIGHT_BOUNDARY_COMMIT,
        "outcome_report_path": "reports/experiment-004-training.json",
        "outcome_report_present": False,
        "preflight_boundary_commit": P4_PREFLIGHT_BOUNDARY_COMMIT,
        "preflight_boundary_parent": P4_IMPLEMENTATION_COMMIT,
        "registration_path": "configs/experiment-004-run.json",
        "registration_present": False,
    }

    incident_payload = P4_INCIDENT_PATH.read_bytes()
    assert _sha256(incident_payload) == P4_INCIDENT_SHA256
    incident = _strict_json(incident_payload)
    assert _canonical(incident) == incident_payload
    assert incident["attempt"] == predecessor["attempt"]
    assert incident["implementation"] == predecessor["implementation"]
    assert incident["protocol"] == {
        "commit": P4_PROTOCOL_COMMIT,
        "path": "configs/experiment-004-execution.json",
        "sha256": P4_PROTOCOL_SHA256,
    }
    assert incident["incident"] == "pre_registration_protocol_rejection"
    assert incident["kind"] == "execution_protocol_preflight_incident"
    assert incident["status"] == "preflight_rejected"
    assert tuple(incident["managed_outputs_absent"]) == P4_MANAGED_OUTPUTS
    assert [violation["code"] for violation in incident["violations"]] == [
        "facade_normalization_recipe_cannot_express_profile_004",
        "shared_authority_recipe_omits_required_dispatch",
    ]
    assert incident["violations"][1]["actual_required_dispatch"] == (
        "_frozen_file_bindings"
    )
    assert incident["violations"][1]["listed_generic_dispatches"] == [
        "_AUTHORITY_PROFILES",
        "_expected_frozen_bindings",
        "_require_authority_profile",
        "_require_invocation_registration",
        "_verify_committed_repository",
    ]


def test_p4_protocol_incident_topology_and_all_immutable_blobs_are_exact() -> None:
    assert (
        _git(
            "rev-list",
            "--parents",
            "-n",
            "1",
            P4_PROTOCOL_COMMIT,
        )
        == f"{P4_PROTOCOL_COMMIT} {P4_PROTOCOL_PARENT}\n".encode()
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
            P4_PROTOCOL_PARENT,
            P4_PROTOCOL_COMMIT,
        )
        == b"A\0configs/experiment-004-execution.json\0"
    )
    assert (
        _git(
            "ls-tree",
            P4_PROTOCOL_COMMIT,
            "--",
            "configs/experiment-004-execution.json",
        )
        == (
            f"100644 blob {P4_PROTOCOL_GIT_BLOB}\t"
            "configs/experiment-004-execution.json\n"
        ).encode()
    )
    p4_protocol_payload = P4_PROTOCOL_PATH.read_bytes()
    assert _sha256(p4_protocol_payload) == P4_PROTOCOL_SHA256
    assert (
        _git(
            "show",
            f"{P4_PROTOCOL_COMMIT}:configs/experiment-004-execution.json",
        )
        == p4_protocol_payload
    )
    assert (
        subprocess.run(
            (
                "git",
                "merge-base",
                "--is-ancestor",
                P4_PROTOCOL_COMMIT,
                P4_IMPLEMENTATION_COMMIT,
            ),
            cwd=ROOT,
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )

    assert (
        _git(
            "rev-list",
            "--parents",
            "-n",
            "1",
            P4_PREFLIGHT_BOUNDARY_COMMIT,
        )
        == f"{P4_PREFLIGHT_BOUNDARY_COMMIT} {P4_IMPLEMENTATION_COMMIT}\n".encode()
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
            P4_IMPLEMENTATION_COMMIT,
            P4_PREFLIGHT_BOUNDARY_COMMIT,
        )
        == b"A\0docs/experiment-004.md\0"
    )
    assert (
        _git(
            "rev-list",
            "--parents",
            "-n",
            "1",
            P4_INCIDENT_COMMIT,
        )
        == f"{P4_INCIDENT_COMMIT} {P4_PREFLIGHT_BOUNDARY_COMMIT}\n".encode()
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
            P4_PREFLIGHT_BOUNDARY_COMMIT,
            P4_INCIDENT_COMMIT,
        )
        == b"A\0reports/experiment-004-preflight-incident.json\0"
    )
    assert (
        _git(
            "ls-tree",
            P4_INCIDENT_COMMIT,
            "--",
            "reports/experiment-004-preflight-incident.json",
        )
        == (
            f"100644 blob {P4_INCIDENT_GIT_BLOB}\t"
            "reports/experiment-004-preflight-incident.json\n"
        ).encode()
    )
    assert (
        _git(
            "show",
            f"{P4_INCIDENT_COMMIT}:reports/experiment-004-preflight-incident.json",
        )
        == P4_INCIDENT_PATH.read_bytes()
    )

    frozen_bindings = _protocol()["execution_delta"]["immutable_predecessor_sources"][
        "experiment_004_modules"
    ]
    assert tuple(frozen_bindings) == P4_SOURCE_BINDINGS
    for binding in P4_SOURCE_BINDINGS:
        path = binding["path"]
        committed = _git("show", f"{P4_IMPLEMENTATION_COMMIT}:{path}")
        current = (ROOT / path).read_bytes()
        assert _sha256(committed) == binding["sha256"]
        assert _sha256(current) == binding["sha256"]
        assert current == committed
        assert _git(
            "ls-tree",
            P4_IMPLEMENTATION_COMMIT,
            "--",
            path,
        ).startswith(b"100644 blob ")

    immutable_auxiliary = (
        (
            "src/falsewake/experiment_002_run_authority.py",
            P4_SHARED_AUTHORITY_SHA256,
        ),
        (
            "tests/test_experiment_004_protocol.py",
            P4_PROTOCOL_PROOF_SHA256,
        ),
    )
    for path, expected_sha256 in immutable_auxiliary:
        committed = _git("show", f"{P4_IMPLEMENTATION_COMMIT}:{path}")
        assert _sha256(committed) == expected_sha256
        assert _git(
            "ls-tree",
            P4_IMPLEMENTATION_COMMIT,
            "--",
            path,
        ).startswith(b"100644 blob ")
    for path in P4_MANAGED_OUTPUTS:
        assert not _git("ls-tree", P4_INCIDENT_COMMIT, "--", path)
        assert not os.path.lexists(ROOT / path)


def test_frozen_routes_profile_cross_directions_and_substitutions_are_exact() -> None:
    document = _protocol()
    admission = document["admission_contract"]
    assert len(admission["parent_route_matrix"]) == 15
    assert len(admission["wrapper_route_matrix"]) == 4
    assert admission["route_invariants"] == {
        "defaults": None,
        "exact_type": "types.FunctionType",
        "keyword_defaults": None,
        "keyword_only_arguments": 0,
        "variable_keyword_arguments": False,
        "variable_positional_arguments": False,
    }
    assert admission["live_route_contract"] == {
        "argument_count": 1,
        "defaults": None,
        "keyword_defaults": None,
        "keyword_only_arguments": 0,
        "module": "falsewake.experiment_005_run_authority",
        "name": "_verified_state",
        "positional_only_arguments": 1,
        "positional_or_keyword_arguments": 0,
        "variable_keyword_arguments": False,
        "variable_positional_arguments": False,
    }

    authority = document["authority_profile"]
    assert tuple(authority["profile_binding_fields"]) == PROFILE_FIELDS
    assert len(authority["profile_values"]) == 14
    assert tuple(authority["cross_profile_rejections"]) == (CROSS_PROFILE_REJECTIONS)
    assert tuple(document["verification_gate"]["cross_profile_rejection"]) == (
        CROSS_PROFILE_DIRECTIONS
    )

    parity = document["execution_delta"]["parity_contract"]
    substitutions = parity["normalization"]["ordered_literal_substitutions"]
    assert tuple(substitutions) == ORDERED_SUBSTITUTIONS
    assert parity["normalization"]["direction"] == (
        "experiment_005_candidate_to_experiment_004_reference"
    )
    assert parity["normalization"]["encoding"] == "UTF-8"
    assert parity["unrestricted_textual_drift"] == "forbidden"

    shared = document["execution_delta"]["shared_authority_recipe"]
    assert tuple(shared["added_symbols"]) == SHARED_ADDED_SYMBOLS
    assert tuple(shared["modified_generic_dispatches"]) == SHARED_DISPATCHES
    assert shared["baseline_sha256"] == P4_SHARED_AUTHORITY_SHA256
    assert shared["normalization_result"] == (
        "removing_the_exact_profile_005_symbols_and_six_dispatch_arms_"
        "restores_the_baseline_bytes"
    )
    assert "_frozen_file_bindings" in shared["modified_generic_dispatches"]


def test_actual_15_plus_4_routes_profile_and_all_cross_directions_are_live() -> None:
    assert not os.path.lexists(P4_MARKER)
    assert not os.path.lexists(P5_MARKER)
    document = _protocol()
    admission = document["admission_contract"]
    routes = [
        *admission["parent_route_matrix"],
        *admission["wrapper_route_matrix"],
    ]
    required_modules = admission["pre_registration_proof"]["actual_modules_required"]
    profile_values = document["authority_profile"]["profile_values"]
    script = f"""
import importlib
import inspect
import json
import os
import sys
import weakref
from pathlib import Path
from types import FunctionType

sys.dont_write_bytecode = True
sys.path.insert(0, {str(ROOT / "src")!r})
p4_marker = {str(P4_MARKER)!r}
p5_marker = {str(P5_MARKER)!r}
assert not os.path.lexists(p4_marker)
assert not os.path.lexists(p5_marker)
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
        }}
    )

engine = importlib.import_module("falsewake.experiment_002_run_authority")
authority_003 = importlib.import_module("falsewake.experiment_003_run_authority")
authority_004 = importlib.import_module("falsewake.experiment_004_run_authority")
authority_005 = importlib.import_module("falsewake.experiment_005_run_authority")
profiles = engine._AUTHORITY_PROFILES
assert tuple(item.registration_experiment for item in profiles) == (
    "002",
    "003",
    "004",
    "005",
)
profile = engine._EXPERIMENT_005_PROFILE
profile_fields = tuple(profile.__dataclass_fields__)
observed_profile = {{}}
for field in profile_fields:
    value = getattr(profile, field)
    if type(value) is bytes:
        value = value.decode("ascii")
    observed_profile[field] = value

engine._ISSUED = weakref.WeakKeyDictionary()
engine._ISSUED_GUARDS = weakref.WeakKeyDictionary()
engine._FAILED = weakref.WeakSet()
engine._ISSUANCE_COMPLETE = False
verifiers = {{
    "002": engine.verify_verified_run_registration,
    "003": authority_003.verify_verified_run_registration,
    "004": authority_004.verify_verified_run_registration,
    "005": authority_005.verify_verified_run_registration,
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
        (sys.executable, "-I", "-B", "-c", script),
        cwd=ROOT,
        env=environment,
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
        }
        for route in routes
    ]
    assert observed["routes"] == expected_routes
    assert tuple(observed["profile_fields"]) == PROFILE_FIELDS
    assert observed["profile_values"] == profile_values
    assert tuple(observed["directions"]) == CROSS_PROFILE_DIRECTIONS
    assert not os.path.lexists(P4_MARKER)
    assert not os.path.lexists(P5_MARKER)


def test_coordinator_ast_claims_every_frozen_parent_route_without_invocation() -> None:
    expected_rows = _protocol()["admission_contract"]["parent_route_matrix"]
    path = ROOT / "src/falsewake/experiment_005_coordinator.py"
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
        "run_authority_namespace": "falsewake.experiment_005_run_authority",
        "supervisor_namespace": "falsewake.experiment_005_supervisor",
        "evidence_namespace": "falsewake.experiment_005_final_evidence",
        "publication_namespace": "falsewake.experiment_005_final_publication",
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
            "module": "falsewake.experiment_005_supervisor",
            "name": name,
            "positional_only_arguments": 0,
            "positional_or_keyword_arguments": 0,
        }
        for binding, name in lifecycle_bindings
    )
    assert {row["binding"]: row for row in static_rows} == {
        row["binding"]: row for row in expected_rows
    }


def test_simple_p5_to_p4_source_recipes_and_supervisor_hash_are_byte_exact() -> None:
    document = _protocol()
    recipes = document["execution_delta"]["parity_contract"]["file_recipes"]
    recipe_by_candidate = {recipe["candidate_path"]: recipe for recipe in recipes}
    for suffix in ("coordinator", "runner", "seed_worker"):
        candidate_path = f"src/falsewake/experiment_005_{suffix}.py"
        reference_path = f"src/falsewake/experiment_004_{suffix}.py"
        recipe = recipe_by_candidate[candidate_path]
        reference = (ROOT / reference_path).read_text(encoding="utf-8")
        assert _sha256(reference.encode()) == recipe["reference_sha256"]
        candidate = _normalize_literals(
            (ROOT / candidate_path).read_text(encoding="utf-8")
        )
        assert candidate == reference

    runner_path = ROOT / "src/falsewake/experiment_005_runner.py"
    runner_sha256 = _sha256(runner_path.read_bytes())
    supervisor_path = ROOT / "src/falsewake/experiment_005_supervisor.py"
    supervisor_candidate = supervisor_path.read_text(encoding="utf-8")
    runner_assignment = re.compile(
        r'(_RUNNER_SHA256: Final = \(\n    ")([0-9a-f]{64})("\n\))'
    )
    matches = list(runner_assignment.finditer(supervisor_candidate))
    assert len(matches) == 1
    assert matches[0].group(2) == runner_sha256
    reference_runner_sha256 = document["execution_delta"]["parity_contract"][
        "normalization"
    ]["supervisor_runner_sha256_assignment"]["normalize_to_reference_value"]
    assert reference_runner_sha256 == P4_SOURCE_BINDINGS[4]["sha256"]
    supervisor_candidate = runner_assignment.sub(
        rf"\g<1>{reference_runner_sha256}\g<3>",
        supervisor_candidate,
        count=1,
    )
    supervisor_candidate = _normalize_literals(supervisor_candidate)
    supervisor_reference = (
        ROOT / "src/falsewake/experiment_004_supervisor.py"
    ).read_text(encoding="utf-8")
    assert supervisor_candidate == supervisor_reference


def test_frozen_facade_recipe_has_one_exact_preflight_defect() -> None:
    document = _protocol()
    recipe = next(
        recipe
        for recipe in document["execution_delta"]["parity_contract"]["file_recipes"]
        if recipe["candidate_path"] == "src/falsewake/experiment_005_run_authority.py"
    )
    assert recipe["normalization"]["result"] == "byte_identical_to_reference"
    candidate = _normalize_literals(
        (ROOT / recipe["candidate_path"]).read_text(encoding="utf-8")
    )
    reference = (ROOT / recipe["reference_path"]).read_text(encoding="utf-8")
    assert _sha256(reference.encode()) == recipe["reference_sha256"]
    assert candidate != reference
    assert _sha256(candidate.encode()) == (
        "9778140580667e95e120a4a8472d8b79ffb7034c3886c208fc3fd3c4da966c86"
    )

    reference_line = (
        '    """Return shared retained state only when its exact profile '
        'is ``004``."""\n'
    )
    candidate_line = (
        '    """Return shared retained state only when its exact profile '
        'is ``005``."""\n'
    )
    assert reference.count(reference_line) == 1
    assert candidate.count(candidate_line) == 1
    assert candidate.count(reference_line) == 0
    opcodes = difflib.SequenceMatcher(
        None,
        reference.splitlines(keepends=True),
        candidate.splitlines(keepends=True),
        autojunk=False,
    ).get_opcodes()
    changed = [opcode for opcode in opcodes if opcode[0] != "equal"]
    assert len(changed) == 1
    tag, reference_first, reference_last, candidate_first, candidate_last = changed[0]
    assert tag == "replace"
    assert reference.splitlines(keepends=True)[reference_first:reference_last] == [
        reference_line
    ]
    assert candidate.splitlines(keepends=True)[candidate_first:candidate_last] == [
        candidate_line
    ]
    assert candidate.replace(candidate_line, reference_line) == reference
    assert "profile is ``005``" not in {
        substitution["candidate"] for substitution in ORDERED_SUBSTITUTIONS
    }


def test_evidence_and_publication_replace_only_the_exact_predecessor_regions() -> None:
    document = _protocol()
    recipes = document["execution_delta"]["parity_contract"]["file_recipes"]
    for suffix in ("final_evidence", "final_publication"):
        candidate_path = f"src/falsewake/experiment_005_{suffix}.py"
        reference_path = f"src/falsewake/experiment_004_{suffix}.py"
        recipe = next(
            item for item in recipes if item["candidate_path"] == candidate_path
        )
        assert recipe["reference_path"] == reference_path
        assert recipe["normalization"]["result"] == "byte_identical_to_reference"
        edits = recipe["normalization"]["post_normalization_edits"]
        assert len(edits) == 1
        assert edits[0]["kind"] == "replace_exact_predecessor_contract_region_once"
        assert edits[0]["candidate_value_json_pointer"] == "/predecessor"
        assert edits[0]["reference_document_path"] == (
            "configs/experiment-004-execution.json"
        )
        assert edits[0]["reference_document_sha256"] == P4_PROTOCOL_SHA256
        assert edits[0]["reference_value_json_pointer"] == "/predecessor"

        candidate = _normalize_literals(
            (ROOT / candidate_path).read_text(encoding="utf-8")
        )
        reference = (ROOT / reference_path).read_text(encoding="utf-8")
        assert _sha256(reference.encode()) == recipe["reference_sha256"]
        candidate = _replace_predecessor_contract_regions(candidate, reference)
        if suffix == "final_publication":
            remaining_symbol_replacements = (
                (
                    "_PREDECESSOR_PROTOCOL_PROOF_SHA256",
                    "_PREDECESSOR_RUNTIME_FINGERPRINT_SHA256",
                ),
                (
                    "_PREDECESSOR_SHARED_AUTHORITY_SHA256",
                    "_PREDECESSOR_SOURCE_BUNDLE_SHA256",
                ),
                (
                    "_PREDECESSOR_PREFLIGHT_BOUNDARY_COMMIT",
                    "_PREDECESSOR_REGISTRATION_COMMIT",
                ),
                (
                    "_PREDECESSOR_PREFLIGHT_BOUNDARY_PARENT",
                    "_PREDECESSOR_REGISTRATION_SHA256",
                ),
            )
            for candidate_symbol, reference_symbol in remaining_symbol_replacements:
                assert candidate.count(candidate_symbol) == 1
                candidate = candidate.replace(
                    candidate_symbol,
                    reference_symbol,
                )
        assert candidate == reference

    predecessor = document["predecessor"]
    canonical_predecessor = json.dumps(
        predecessor,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    script = f"""
import json
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, {str(ROOT / "src")!r})
assert not os.path.lexists({str(P4_MARKER)!r})
assert not os.path.lexists({str(P5_MARKER)!r})
from falsewake import experiment_005_final_evidence as evidence
from falsewake import experiment_005_final_publication as publication
expected = json.loads({json.dumps(predecessor, sort_keys=True)!r})
assert evidence._predecessor_document() == expected
assert publication._predecessor_document() == expected
assert publication._PREDECESSOR_CANONICAL_JSON == {canonical_predecessor!r}
assert not os.path.lexists({str(P4_MARKER)!r})
assert not os.path.lexists({str(P5_MARKER)!r})
"""
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(
        (sys.executable, "-I", "-B", "-c", script),
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_shared_authority_has_only_17_symbols_and_the_six_frozen_dispatches() -> None:
    path = "src/falsewake/experiment_002_run_authority.py"
    baseline = _git("show", f"{P4_IMPLEMENTATION_COMMIT}:{path}").decode()
    candidate = (ROOT / path).read_text(encoding="utf-8")
    assert _sha256(baseline.encode()) == P4_SHARED_AUTHORITY_SHA256

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

    baseline_lines = baseline.splitlines(keepends=True)
    candidate_lines = candidate.splitlines(keepends=True)
    matcher = difflib.SequenceMatcher(
        None,
        baseline_lines,
        candidate_lines,
        autojunk=False,
    )
    restored_lines: list[str] = []
    changed_opcodes = 0
    for (
        tag,
        baseline_first,
        baseline_last,
        candidate_first,
        candidate_last,
    ) in matcher.get_opcodes():
        if tag == "equal":
            restored_lines.extend(candidate_lines[candidate_first:candidate_last])
            continue
        changed_opcodes += 1
        restored_lines.extend(baseline_lines[baseline_first:baseline_last])
    assert changed_opcodes > 0
    restored = "".join(restored_lines)
    assert restored == baseline
    assert _sha256(restored.encode()) == P4_SHARED_AUTHORITY_SHA256

    source = candidate
    for symbol in SHARED_ADDED_SYMBOLS:
        assert symbol in source
    for dispatch in SHARED_DISPATCHES:
        assert dispatch in source
    assert "_frozen_file_bindings" in changed


def test_registration_outputs_and_both_permanent_markers_remain_absent() -> None:
    document = _protocol()
    assert document["lifecycle"]["pre_registration_observations"] == {
        "registered_authority_issuer_invocations": 0,
        "registered_coordinator_invocations": 0,
        "registered_invocation_count": 0,
        "registered_optimizer_updates": 0,
        "registered_runner_invocations": 0,
        "registered_validation_examples": 0,
    }
    assert document["predecessor"]["attempt"]["registered_attempt_consumed"] is False
    assert document["predecessor"]["registration"]["present"] is False
    for path in (*P4_MANAGED_OUTPUTS, *P5_MANAGED_OUTPUTS):
        assert not os.path.lexists(ROOT / path)
    assert not os.path.lexists(P4_MARKER)
    assert not os.path.lexists(P5_MARKER)
