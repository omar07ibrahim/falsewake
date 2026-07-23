from __future__ import annotations

import ast
import difflib
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports" / "experiment-004-preflight-incident.json"
PROTOCOL_PATH = ROOT / "configs" / "experiment-004-execution.json"
FACADE_PATH = ROOT / "src" / "falsewake" / "experiment_004_run_authority.py"
REFERENCE_FACADE_PATH = ROOT / "src" / "falsewake" / "experiment_003_run_authority.py"
SHARED_AUTHORITY_PATH = ROOT / "src" / "falsewake" / "experiment_002_run_authority.py"

BASE_COMMIT = "96f151d86ae060b402a0ef5d47de7ad8c1191c0a"
PROTOCOL_COMMIT = "c7a8b3c493e05211ff7afa3a7abb974fc1c8b4e2"
IMPLEMENTATION_COMMIT = "f81cd142885068c27767d6d728da04248fb1a470"
REPORT_SHA256 = "d015b749acf6e98e87172372315c78ccfaad48b8ae6b3a29ae426890aa467f50"
PROTOCOL_SHA256 = "aba6c1eca84ad33c7751e4768a8ef630ba2c063720fd29938dd1b69abf945a46"
NORMALIZED_FACADE_SHA256 = (
    "2e4909ab2af22c4018e2f543dfe7020b1bf711f53b575788b4aecafc6b959778"
)
REFERENCE_FACADE_SHA256 = (
    "690ed90679928766a50219657ba467e0e0d39bf2763fa2b60128640fdf0938e3"
)
ATTEMPT_MARKER = Path("/home/ubuntu/gitcode/.t/falsewake-experiment-004-attempt")

IMPLEMENTATION_CHAIN = (
    (
        "cce685788a7742cdbcd3ef1ebe3559047d4a414d",
        PROTOCOL_COMMIT,
    ),
    (
        "65daf0ebbe77cc7f906ef3e55d2455aea50943ba",
        "cce685788a7742cdbcd3ef1ebe3559047d4a414d",
    ),
    (
        "d4bb633bbe97cfe5b633d4ceb4d1129eab4e8094",
        "65daf0ebbe77cc7f906ef3e55d2455aea50943ba",
    ),
    (
        IMPLEMENTATION_COMMIT,
        "d4bb633bbe97cfe5b633d4ceb4d1129eab4e8094",
    ),
)

SOURCE_BINDINGS: tuple[dict[str, str], ...] = (
    {
        "path": "src/falsewake/experiment_004_coordinator.py",
        "sha256": "6e4df88440c361376ce51a3574a16bbec79ecbc6ff4dd1fcf0cf8be86eac1daa",
    },
    {
        "path": "src/falsewake/experiment_004_final_evidence.py",
        "sha256": "5fcff993e434c36d73c5e9b4f6d1e5954e8a54f2ce9fc17450ea615c5a4df5af",
    },
    {
        "path": "src/falsewake/experiment_004_final_publication.py",
        "sha256": "b862490f15e88819e815cfe6d7769ff125e21e59cdde4642a4a700e5f2eaad16",
    },
    {
        "path": "src/falsewake/experiment_004_run_authority.py",
        "sha256": "cacab13729f14bfdce10afa5b53b2a09116cf56abe21aee12ac9220a6ad2e5b5",
    },
    {
        "path": "src/falsewake/experiment_004_runner.py",
        "sha256": "76f19ec33c10955fa50d26b8d35d56aeb15d712b72b22a17e88933f46d24accf",
    },
    {
        "path": "src/falsewake/experiment_004_seed_worker.py",
        "sha256": "40cc3f0681006b03eac03f4747c60feb7a0ab42c102772afd6b90ef809929361",
    },
    {
        "path": "src/falsewake/experiment_004_supervisor.py",
        "sha256": "750371018aefd8dc3cdbffff20a60c3785680992b05d6246f052f37609a6c8c9",
    },
)
SHARED_AUTHORITY_BINDING = {
    "path": "src/falsewake/experiment_002_run_authority.py",
    "sha256": "52ce6039345b2159a4bf83ea294af2c3211cdcfbe87e9bc806133c67647b1a47",
}
PROTOCOL_PROOF_BINDING = {
    "path": "tests/test_experiment_004_protocol.py",
    "sha256": "8539610615717b8b1fb5ff41876aef41a36e8ac7a8dbda4eeb7307174ae7a3a8",
}

MANAGED_OUTPUTS = (
    "configs/experiment-004-run.json",
    "models/experiment-004-selected.safetensors",
    "reports/experiment-004-seed-20260719-history.json",
    "reports/experiment-004-seed-20260720-history.json",
    "reports/experiment-004-seed-20260721-history.json",
    "reports/experiment-004-selected-rerun-history.json",
    "reports/experiment-004-training.json",
)

EXPECTED_SOURCE_DELTA = (
    ("M", "src/falsewake/experiment_002_run_authority.py"),
    *(("A", binding["path"]) for binding in SOURCE_BINDINGS),
)

LITERAL_SUBSTITUTIONS: tuple[dict[str, str], ...] = (
    {"candidate": "Experiment 004", "reference": "Experiment 003"},
    {"candidate": "Experiment004", "reference": "Experiment003"},
    {"candidate": "experiment_004", "reference": "experiment_003"},
    {"candidate": "experiment-004", "reference": "experiment-003"},
    {"candidate": "exp004", "reference": "exp003"},
    {"candidate": '"004"', "reference": '"003"'},
    {"candidate": "FW4ACTV1", "reference": "FW3ACTV1"},
    {"candidate": "FW4CHLD1", "reference": "FW3CHLD1"},
)

LISTED_GENERIC_DISPATCHES = (
    "_AUTHORITY_PROFILES",
    "_expected_frozen_bindings",
    "_require_authority_profile",
    "_require_invocation_registration",
    "_verify_committed_repository",
)


def _reject_duplicate_key(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"duplicate JSON key: {key}")
        document[key] = value
    return document


def _reject_float(value: str) -> None:
    raise ValueError(f"non-integer JSON number is forbidden: {value}")


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
    try:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        pytest.fail(f"Git evidence is unavailable: {error}")
    return completed.stdout


def _report() -> dict[str, Any]:
    return _strict_json(REPORT_PATH.read_bytes())


def _protocol() -> dict[str, Any]:
    return _strict_json(PROTOCOL_PATH.read_bytes())


def _tree_blob(commit: str, path: str) -> bytes:
    entry = _git("ls-tree", commit, "--", path).decode("ascii")
    mode, object_type, object_and_path = entry.split(" ", 2)
    object_id, observed_path = object_and_path.rstrip("\n").split("\t", 1)
    assert mode == "100644"
    assert object_type == "blob"
    assert observed_path == path
    return _git("cat-file", "blob", object_id)


def _name_status(payload: bytes) -> tuple[tuple[str, str], ...]:
    fields = payload.split(b"\0")
    assert fields[-1] == b""
    fields.pop()
    assert len(fields) % 2 == 0
    return tuple(
        (
            fields[index].decode("ascii"),
            fields[index + 1].decode("ascii"),
        )
        for index in range(0, len(fields), 2)
    )


def test_preflight_incident_is_strict_canonical_ascii_json() -> None:
    payload = REPORT_PATH.read_bytes()
    document = _strict_json(payload)

    assert payload == _canonical(document)
    assert _sha256(payload) == REPORT_SHA256
    assert tuple(document) == (
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
    )
    assert document["schema_version"] == 1
    assert document["experiment"] == "004"
    assert document["kind"] == "execution_protocol_preflight_incident"
    assert document["incident"] == "pre_registration_protocol_rejection"
    assert document["status"] == "preflight_rejected"


def test_report_binds_exact_protocol_implementation_and_proof_blobs() -> None:
    document = _report()
    implementation = document["implementation"]

    assert document["protocol"] == {
        "commit": PROTOCOL_COMMIT,
        "path": "configs/experiment-004-execution.json",
        "sha256": PROTOCOL_SHA256,
    }
    assert implementation == {
        "commit": IMPLEMENTATION_COMMIT,
        "production_sources": list(SOURCE_BINDINGS),
        "protocol_proof": PROTOCOL_PROOF_BINDING,
        "shared_authority": SHARED_AUTHORITY_BINDING,
    }

    protocol_payload = PROTOCOL_PATH.read_bytes()
    assert _sha256(protocol_payload) == PROTOCOL_SHA256
    assert _tree_blob(PROTOCOL_COMMIT, document["protocol"]["path"]) == protocol_payload

    for binding in SOURCE_BINDINGS:
        path = binding["path"]
        current_payload = (ROOT / path).read_bytes()
        committed_payload = _tree_blob(IMPLEMENTATION_COMMIT, path)
        assert _sha256(current_payload) == binding["sha256"]
        assert _sha256(committed_payload) == binding["sha256"]
        assert current_payload == committed_payload

    shared_path = SHARED_AUTHORITY_BINDING["path"]
    committed_shared = _tree_blob(IMPLEMENTATION_COMMIT, shared_path)
    assert _sha256(committed_shared) == SHARED_AUTHORITY_BINDING["sha256"]

    proof_path = PROTOCOL_PROOF_BINDING["path"]
    committed_proof = _tree_blob(IMPLEMENTATION_COMMIT, proof_path)
    assert _sha256(committed_proof) == PROTOCOL_PROOF_BINDING["sha256"]


def test_implementation_topology_and_source_delta_are_exact() -> None:
    assert _git(
        "rev-list",
        "--parents",
        "--ancestry-path",
        "--reverse",
        f"{PROTOCOL_COMMIT}..{IMPLEMENTATION_COMMIT}",
    ).decode("ascii").splitlines() == [
        f"{commit} {parent}" for commit, parent in IMPLEMENTATION_CHAIN
    ]
    assert (
        _git("merge-base", "--is-ancestor", PROTOCOL_COMMIT, IMPLEMENTATION_COMMIT)
        == b""
    )

    observed_delta = _name_status(
        _git(
            "diff",
            "--name-status",
            "-z",
            BASE_COMMIT,
            IMPLEMENTATION_COMMIT,
            "--",
            "src/falsewake",
        )
    )
    assert observed_delta == EXPECTED_SOURCE_DELTA
    assert _protocol()["execution_delta"]["implementation_source_delta"] == {
        "base_commit": BASE_COMMIT,
        "name_status": [
            {"path": path, "status": status} for status, path in EXPECTED_SOURCE_DELTA
        ],
        "other_src_falsewake_changes": "forbidden",
    }


def test_no_registered_attempt_or_managed_output_exists() -> None:
    document = _report()

    assert tuple(document["managed_outputs_absent"]) == MANAGED_OUTPUTS
    assert not os.path.lexists(ATTEMPT_MARKER)
    for path in MANAGED_OUTPUTS:
        assert not os.path.lexists(ROOT / path)
        assert _git(
            "ls-tree", "-r", "--name-only", IMPLEMENTATION_COMMIT, "--", path
        ) == (b"")
    assert not os.path.lexists(ATTEMPT_MARKER)


def test_eight_frozen_substitutions_leave_exact_facade_mismatch() -> None:
    report = _report()
    protocol = _protocol()
    substitutions = protocol["execution_delta"]["parity_contract"]["normalization"][
        "ordered_literal_substitutions"
    ]
    assert tuple(substitutions) == LITERAL_SUBSTITUTIONS
    assert len(substitutions) == 8

    normalized = FACADE_PATH.read_text(encoding="utf-8")
    for substitution in substitutions:
        normalized = normalized.replace(
            substitution["candidate"],
            substitution["reference"],
        )
    reference = REFERENCE_FACADE_PATH.read_text(encoding="utf-8")

    matcher = difflib.SequenceMatcher(
        a=reference.splitlines(),
        b=normalized.splitlines(),
        autojunk=False,
    )
    deleted_lines = 0
    added_lines = 0
    changed_regions = 0
    for (
        tag,
        reference_start,
        reference_end,
        candidate_start,
        candidate_end,
    ) in matcher.get_opcodes():
        if tag == "equal":
            continue
        changed_regions += 1
        deleted_lines += reference_end - reference_start
        added_lines += candidate_end - candidate_start

    assert normalized != reference
    assert changed_regions == 11
    assert added_lines == 11
    assert deleted_lines == 11
    assert _sha256(normalized.encode("utf-8")) == NORMALIZED_FACADE_SHA256
    assert _sha256(reference.encode("utf-8")) == REFERENCE_FACADE_SHA256

    violation = report["violations"][0]
    assert violation == {
        "candidate_normalized_sha256": NORMALIZED_FACADE_SHA256,
        "candidate_path": "src/falsewake/experiment_004_run_authority.py",
        "code": "facade_normalization_recipe_cannot_express_profile_004",
        "explanation": (
            "The frozen recipe permits eight case-sensitive substitutions and no "
            "post-normalization edit. After all eight substitutions, the correct "
            "profile-004 facade still differs from the immutable Experiment 003 "
            "reference by 11 added and 11 deleted lines. The uncovered text "
            "includes profile-004 prose and the mandatory uppercase "
            "_EXPERIMENT_004_PROFILE and "
            "_EXPERIMENT_004_SEALED_CHILD_MEMFD_TARGET bindings. Replacing those "
            "bindings with profile 003 would make the implementation unsafe and "
            "semantically wrong."
        ),
        "normalized_added_lines": 11,
        "normalized_deleted_lines": 11,
        "protocol_json_pointer": "/execution_delta/parity_contract/file_recipes/3",
        "reference_path": "src/falsewake/experiment_003_run_authority.py",
        "reference_sha256": REFERENCE_FACADE_SHA256,
    }


def test_shared_recipe_omits_a_required_live_p4_dispatch() -> None:
    report = _report()
    protocol = _protocol()
    recipe = protocol["execution_delta"]["shared_authority_recipe"]

    assert tuple(recipe["modified_generic_dispatches"]) == LISTED_GENERIC_DISPATCHES
    assert "_frozen_file_bindings" not in recipe["modified_generic_dispatches"]
    assert report["violations"][1] == {
        "actual_required_dispatch": "_frozen_file_bindings",
        "code": "shared_authority_recipe_omits_required_dispatch",
        "explanation": (
            "The correct profile-004 implementation must add a "
            "_frozen_file_bindings dispatch because the exact Experiment 003 "
            "predecessor outcome has no artifact path or SHA-256. The frozen recipe "
            "lists five modified generic dispatches but omits this required sixth "
            "dispatch. Removing only the listed profile symbols and arms therefore "
            "cannot restore the baseline bytes as claimed."
        ),
        "listed_generic_dispatches": list(LISTED_GENERIC_DISPATCHES),
        "protocol_json_pointer": (
            "/execution_delta/shared_authority_recipe/modified_generic_dispatches"
        ),
    }

    tree = ast.parse(SHARED_AUTHORITY_PATH.read_text(encoding="utf-8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_frozen_file_bindings"
    ]
    assert len(functions) == 1
    p4_branches = [
        node
        for node in ast.walk(functions[0])
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "profile"
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], ast.Is)
        and len(node.test.comparators) == 1
        and isinstance(node.test.comparators[0], ast.Name)
        and node.test.comparators[0].id == "_EXPERIMENT_004_PROFILE"
    ]
    assert len(p4_branches) == 1
    branch_names = {
        node.id for node in ast.walk(p4_branches[0]) if isinstance(node, ast.Name)
    }
    assert {
        "_EXPERIMENT_004_PROFILE",
        "predecessor_execution_protocol",
        "predecessor_incident",
    }.issubset(branch_names)
    assert "predecessor_outcome" not in branch_names


def test_preflight_consumed_nothing_and_recovery_requires_new_identifier() -> None:
    report = _report()
    protocol = _protocol()

    assert report["attempt"] == {
        "canonical_marker": str(ATTEMPT_MARKER),
        "canonical_marker_present": False,
        "optimizer_updates": 0,
        "registered_attempt_consumed": False,
        "registered_authority_issuer_invoked": False,
        "registered_coordinator_invoked": False,
        "registered_invocation_count": 0,
        "registered_runner_invoked": False,
        "validation_examples": 0,
    }
    assert report["preflight_evidence"]["full_registration_gate"] == {
        "reason": "frozen_protocol_contract_is_internally_unsatisfiable",
        "status": "not_run",
    }
    assert (
        protocol["attempt_contract"][
            "preflight_rejection_consumes_registered_invocation"
        ]
        is False
    )
    assert protocol["lifecycle"]["pre_registration_observations"] == {
        "registered_optimizer_updates": 0,
        "registered_validation_examples": 0,
    }
    assert protocol["lifecycle"]["retry_policy"] == (
        "any_terminal_attempt_requires_a_new_experiment_identifier"
    )
    assert report["recovery_requirements"][1] == (
        "Use a new experiment identifier, authority profile, registration, "
        "transport namespace, publication namespace, scratch root, staging root, "
        "and permanent attempt marker."
    )
    assert not os.path.lexists(ATTEMPT_MARKER)
