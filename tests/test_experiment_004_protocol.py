from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs" / "experiment-004-execution.json"
PROTOCOL_COMMIT = "c7a8b3c493e05211ff7afa3a7abb974fc1c8b4e2"
PROTOCOL_PARENT = "96f151d86ae060b402a0ef5d47de7ad8c1191c0a"
PROTOCOL_SHA256 = "aba6c1eca84ad33c7751e4768a8ef630ba2c063720fd29938dd1b69abf945a46"
PROTOCOL_GIT_BLOB = "c7474e5abf08634112939af72138ba913af5dff0"
REFERENCE_IMPLEMENTATION = "4475461d5fd3e5b8969020003424c73bb10d3c9b"
ATTEMPT_MARKER = Path("/home/ubuntu/gitcode/.t/falsewake-experiment-004-attempt")

PARENT_ROUTE_MATRIX: tuple[dict[str, object], ...] = (
    {
        "binding": "_create_sealed_experiment_004_child_bundle_fd",
        "module": "falsewake.experiment_004_run_authority",
        "name": "_create_sealed_experiment_004_child_bundle_fd",
        "positional_only_arguments": 1,
        "positional_or_keyword_arguments": 0,
    },
    {
        "binding": "_required_child_bundle_seals",
        "module": "falsewake.experiment_004_run_authority",
        "name": "_required_child_bundle_seals",
        "positional_only_arguments": 0,
        "positional_or_keyword_arguments": 0,
    },
    {
        "binding": "_verified_state",
        "module": "falsewake.experiment_004_run_authority",
        "name": "_verified_state",
        "positional_only_arguments": 1,
        "positional_or_keyword_arguments": 0,
    },
    {
        "binding": "build_completed_final_evidence",
        "module": "falsewake.experiment_004_final_evidence",
        "name": "build_completed_final_evidence",
        "positional_only_arguments": 2,
        "positional_or_keyword_arguments": 0,
    },
    {
        "binding": "verify_completed_final_evidence",
        "module": "falsewake.experiment_004_final_evidence",
        "name": "verify_completed_final_evidence",
        "positional_only_arguments": 1,
        "positional_or_keyword_arguments": 0,
    },
    {
        "binding": "build_execution_failure_evidence",
        "module": "falsewake.experiment_004_final_publication",
        "name": "build_execution_failure_evidence",
        "positional_only_arguments": 3,
        "positional_or_keyword_arguments": 0,
    },
    {
        "binding": "publish_registered_final_evidence",
        "module": "falsewake.experiment_004_final_publication",
        "name": "publish_registered_final_evidence",
        "positional_only_arguments": 2,
        "positional_or_keyword_arguments": 0,
    },
    {
        "binding": "verify_execution_failure_evidence",
        "module": "falsewake.experiment_004_final_publication",
        "name": "verify_execution_failure_evidence",
        "positional_only_arguments": 1,
        "positional_or_keyword_arguments": 0,
    },
    {
        "binding": "_begin_registered_supervisor_child",
        "module": "falsewake.experiment_004_supervisor",
        "name": "begin_registered_child",
        "positional_only_arguments": 0,
        "positional_or_keyword_arguments": 0,
    },
    {
        "binding": "_capture_registered_cpu_ids",
        "module": "falsewake.experiment_004_supervisor",
        "name": "_capture_registered_cpu_ids",
        "positional_only_arguments": 0,
        "positional_or_keyword_arguments": 0,
    },
    {
        "binding": "_cleanup_registered_experiment_output_roots",
        "module": "falsewake.experiment_004_supervisor",
        "name": "_cleanup_registered_experiment_output_roots",
        "positional_only_arguments": 0,
        "positional_or_keyword_arguments": 0,
    },
    {
        "binding": "_finish_registered_supervisor_child_failure",
        "module": "falsewake.experiment_004_supervisor",
        "name": "finish_registered_child_failure",
        "positional_only_arguments": 0,
        "positional_or_keyword_arguments": 0,
    },
    {
        "binding": "_finish_registered_supervisor_child_success",
        "module": "falsewake.experiment_004_supervisor",
        "name": "finish_registered_child_success",
        "positional_only_arguments": 0,
        "positional_or_keyword_arguments": 0,
    },
    {
        "binding": "_prepare_registered_experiment_staging",
        "module": "falsewake.experiment_004_supervisor",
        "name": "_prepare_registered_experiment_staging",
        "positional_only_arguments": 0,
        "positional_or_keyword_arguments": 0,
    },
    {
        "binding": "_require_registered_parent_quiescence",
        "module": "falsewake.experiment_004_supervisor",
        "name": "_require_registered_parent_quiescence",
        "positional_only_arguments": 0,
        "positional_or_keyword_arguments": 0,
    },
)

WRAPPER_ROUTE_MATRIX: tuple[dict[str, object], ...] = (
    {
        "binding": "run_registered_experiment",
        "module": "falsewake.experiment_004_coordinator",
        "name": "run_registered_experiment",
        "positional_only_arguments": 1,
        "positional_or_keyword_arguments": 0,
    },
    {
        "binding": "_registered_child_input_snapshot",
        "module": "falsewake.experiment_004_run_authority",
        "name": "_registered_child_input_snapshot",
        "positional_only_arguments": 1,
        "positional_or_keyword_arguments": 0,
    },
    {
        "binding": "run_registered_seed_process",
        "module": "falsewake.experiment_004_seed_worker",
        "name": "run_registered_seed_process",
        "positional_only_arguments": 2,
        "positional_or_keyword_arguments": 0,
    },
    {
        "binding": "_supervise_registered_child",
        "module": "falsewake.experiment_004_supervisor",
        "name": "_supervise_registered_child",
        "positional_only_arguments": 4,
        "positional_or_keyword_arguments": 0,
    },
)

PROFILE_VALUES: dict[str, object] = {
    "activation_magic": "FW4ACTV1",
    "activation_ticket_domain": "falsewake-exp004-activation-ticket-v1\x00",
    "child_bundle_magic": "FW4CHLD1",
    "child_bundle_memfd_name": "falsewake-exp004-child-bundle",
    "issuer_route": "verify_and_issue_experiment_004_run_registration",
    "registration_experiment": "004",
    "run_config_path": "configs/experiment-004-run.json",
    "runner_entrypoint": "src/falsewake/experiment_004_runner.py",
    "runtime_fingerprint_domain": "falsewake-exp004-runtime-v1\x00",
    "scratch_root": "/home/ubuntu/gitcode/.t/falsewake-experiment-004-scratch",
    "sealed_child_issuer_route": (
        "_verify_and_issue_experiment_004_sealed_child_registration"
    ),
    "sealed_child_memfd_target": ("/memfd:falsewake-exp004-child-bundle (deleted)"),
    "sealed_origin_prefix": "falsewake-sealed://experiment-004/",
    "source_bundle_domain": "falsewake-exp004-source-bundle-v1\x00",
}

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

P3_SOURCE_HASHES: tuple[dict[str, str], ...] = (
    {
        "path": "src/falsewake/experiment_003_coordinator.py",
        "sha256": "f2b5ff3e8e874ddf408ba97475c69ca13875e7ac3a37c7462ba2637099a77d1a",
    },
    {
        "path": "src/falsewake/experiment_003_final_evidence.py",
        "sha256": "6f0ebd8554e703886e63dc9d508ec83ab5a4ce3997b0f99714b441106216b9ad",
    },
    {
        "path": "src/falsewake/experiment_003_final_publication.py",
        "sha256": "af61f1425a936b51711f70b097c8b5399571e32d39511fdf7dc947270fec59bb",
    },
    {
        "path": "src/falsewake/experiment_003_run_authority.py",
        "sha256": "690ed90679928766a50219657ba467e0e0d39bf2763fa2b60128640fdf0938e3",
    },
    {
        "path": "src/falsewake/experiment_003_runner.py",
        "sha256": "6e9eb84027487b17e4a375fe04f8c704c11aa4d78458405900625d462fcd5a4d",
    },
    {
        "path": "src/falsewake/experiment_003_seed_worker.py",
        "sha256": "f9262a47d0bf2621ebb2837017b9947927980bee2c498171cabc34bd9ba800ad",
    },
    {
        "path": "src/falsewake/experiment_003_supervisor.py",
        "sha256": "aa33a0f852d3b43f3eff0ded09873aa60911008ba1d37fa3acf4b5d34ab501e3",
    },
)

PRODUCTION_ADDITIONS = (
    "src/falsewake/experiment_004_coordinator.py",
    "src/falsewake/experiment_004_final_evidence.py",
    "src/falsewake/experiment_004_final_publication.py",
    "src/falsewake/experiment_004_run_authority.py",
    "src/falsewake/experiment_004_runner.py",
    "src/falsewake/experiment_004_seed_worker.py",
    "src/falsewake/experiment_004_supervisor.py",
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


def _payload_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    return _payload_sha256(path.read_bytes())


def _git(*arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        pytest.fail(f"Git repository evidence is unavailable: {error}")
    return completed.stdout


def _protocol() -> dict[str, Any]:
    return _strict_json(PROTOCOL_PATH.read_bytes())


def _literal_substitutions(document: dict[str, Any]) -> tuple[dict[str, str], ...]:
    raw = document["execution_delta"]["parity_contract"]["normalization"][
        "ordered_literal_substitutions"
    ]
    assert type(raw) is list
    result: list[dict[str, str]] = []
    for entry in raw:
        assert type(entry) is dict
        assert tuple(entry) == ("candidate", "reference")
        assert type(entry["candidate"]) is str
        assert type(entry["reference"]) is str
        result.append(
            {
                "candidate": entry["candidate"],
                "reference": entry["reference"],
            }
        )
    return tuple(result)


def _normalize_literals(source: str, document: dict[str, Any]) -> str:
    for substitution in _literal_substitutions(document):
        source = source.replace(
            substitution["candidate"],
            substitution["reference"],
        )
    return source


def _without_docstring(function: ast.FunctionDef) -> list[ast.stmt]:
    statements = list(function.body)
    if (
        statements
        and isinstance(statements[0], ast.Expr)
        and isinstance(statements[0].value, ast.Constant)
        and type(statements[0].value.value) is str
    ):
        statements.pop(0)
    return statements


def test_protocol_blob_has_one_exact_config_only_introduction() -> None:
    payload = PROTOCOL_PATH.read_bytes()
    document = _strict_json(payload)

    assert payload == _canonical(document)
    assert len(payload) == 39_325
    assert _payload_sha256(payload) == PROTOCOL_SHA256
    assert _git("rev-list", "--parents", "-n", "1", PROTOCOL_COMMIT).decode(
        "ascii"
    ).strip().split() == [PROTOCOL_COMMIT, PROTOCOL_PARENT]
    assert (
        _git(
            "diff-tree",
            "--no-commit-id",
            "--name-status",
            "-r",
            "-z",
            PROTOCOL_PARENT,
            PROTOCOL_COMMIT,
        )
        == b"A\x00configs/experiment-004-execution.json\x00"
    )
    tree_entry = _git(
        "ls-tree",
        PROTOCOL_COMMIT,
        "--",
        "configs/experiment-004-execution.json",
    ).decode("ascii")
    assert tree_entry == (
        f"100644 blob {PROTOCOL_GIT_BLOB}\tconfigs/experiment-004-execution.json\n"
    )
    assert _git("cat-file", "blob", PROTOCOL_GIT_BLOB) == payload
    assert (
        _git(
            "show",
            f"{PROTOCOL_COMMIT}:configs/experiment-004-execution.json",
        )
        == payload
    )


def test_protocol_route_profile_namespace_and_source_delta_are_exact() -> None:
    document = _protocol()
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
    assert document["experiment"] == "004"
    assert document["kind"] == "execution_only_interface_recovery"

    admission = document["admission_contract"]
    assert tuple(admission["parent_route_matrix"]) == PARENT_ROUTE_MATRIX
    assert tuple(admission["wrapper_route_matrix"]) == WRAPPER_ROUTE_MATRIX
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
        "module": "falsewake.experiment_004_run_authority",
        "name": "_verified_state",
        "positional_only_arguments": 1,
        "positional_or_keyword_arguments": 0,
        "variable_keyword_arguments": False,
        "variable_positional_arguments": False,
    }
    assert admission["pre_registration_proof"] == {
        "actual_cross_module_route_matrix_required": True,
        "actual_modules_required": [
            "falsewake.experiment_004_coordinator",
            "falsewake.experiment_004_final_evidence",
            "falsewake.experiment_004_final_publication",
            "falsewake.experiment_004_run_authority",
            "falsewake.experiment_004_runner",
            "falsewake.experiment_004_seed_worker",
            "falsewake.experiment_004_supervisor",
        ],
        "coordinator_claim_closure_invocation": "forbidden",
        "coordinator_require_function_identity_calls_checked": True,
        "function_code_objects_checked": True,
        "isolated_process_required": True,
        "method": "actual_module_objects_and_coordinator_AST_route_matrix",
        "registered_authority_issuer_invocation": "forbidden",
        "registered_coordinator_invocation": "forbidden",
        "registered_runner_invocation": "forbidden",
    }

    authority = document["authority_profile"]
    assert authority == {
        "capability_state_profile": "004",
        "capability_type": (
            "falsewake.experiment_002_run_authority.VerifiedRunRegistration"
        ),
        "coordinator_profile_check_required": True,
        "cross_profile_acceptance": "forbidden",
        "cross_profile_rejections": [
            "experiment_002_capability_at_experiment_003_boundary",
            "experiment_002_capability_at_experiment_004_boundary",
            "experiment_003_capability_at_experiment_002_boundary",
            "experiment_003_capability_at_experiment_004_boundary",
            "experiment_004_capability_at_experiment_002_boundary",
            "experiment_004_capability_at_experiment_003_boundary",
        ],
        "facade": "src/falsewake/experiment_004_run_authority.py",
        "issuer_caller_supplied_values": [],
        "issuer_route": "verify_and_issue_experiment_004_run_registration",
        "logical_authority": "004",
        "process_issuance_latch": "shared_across_all_profiles",
        "profile_binding_fields": list(PROFILE_FIELDS),
        "profile_values": PROFILE_VALUES,
        "publisher_profile_check_required": True,
        "reverify_dispatch": "retained_capability_state_profile_only",
        "sealed_child_issuer_route": (
            "_verify_and_issue_experiment_004_sealed_child_registration"
        ),
        "shared_engine": "src/falsewake/experiment_002_run_authority.py",
    }

    histories = [
        "reports/experiment-004-seed-20260719-history.json",
        "reports/experiment-004-seed-20260720-history.json",
        "reports/experiment-004-seed-20260721-history.json",
    ]
    namespace = document["namespace"]
    assert namespace == {
        "activation_thread_name": "falsewake-exp004-child-activation",
        "attempt_marker": str(ATTEMPT_MARKER),
        "authority_module": "falsewake.experiment_004_run_authority",
        "child_bundle_memfd_name": "falsewake-exp004-child-bundle",
        "child_control_fd": 3,
        "child_source_bundle_fd": 7,
        "coordinator_module": "falsewake.experiment_004_coordinator",
        "final_evidence_module": "falsewake.experiment_004_final_evidence",
        "final_publication_module": "falsewake.experiment_004_final_publication",
        "launch_memfd_name": "falsewake-experiment-004-launch",
        "managed_outputs": {
            "histories": histories,
            "model": "models/experiment-004-selected.safetensors",
            "report": "reports/experiment-004-training.json",
            "selected_rerun_history": (
                "reports/experiment-004-selected-rerun-history.json"
            ),
        },
        "required_sealed_sources": [
            "src/falsewake/__init__.py",
            "src/falsewake/experiment_002_process_guard.py",
            "src/falsewake/experiment_002_run_authority.py",
            "src/falsewake/experiment_004_coordinator.py",
            "src/falsewake/experiment_004_run_authority.py",
        ],
        "run_config": "configs/experiment-004-run.json",
        "runner": "src/falsewake/experiment_004_runner.py",
        "scratch_root": ("/home/ubuntu/gitcode/.t/falsewake-experiment-004-scratch"),
        "sealed_child_memfd_target": ("/memfd:falsewake-exp004-child-bundle (deleted)"),
        "seed_worker_module": "falsewake.experiment_004_seed_worker",
        "staging_root": ("/home/ubuntu/gitcode/.t/falsewake-experiment-004-staging"),
        "supervisor_module": "falsewake.experiment_004_supervisor",
        "temporary_publication_prefix": ".falsewake-exp004-publication-",
        "transport": {
            "activation_magic": "FW4ACTV1",
            "activation_ticket_domain": ("falsewake-exp004-activation-ticket-v1\x00"),
            "authority_sealed_origin_prefix": ("falsewake-sealed://experiment-004/"),
            "child_bundle_magic": "FW4CHLD1",
            "inherited_child_result_envelope_domain": (
                "falsewake-exp002-child-result-envelope-v1\x00"
            ),
            "inherited_history_domain": "falsewake-exp002-history-v1\x00",
            "inherited_model_tensor_domain": ("falsewake-exp002-model-tensors-v1\x00"),
            "inherited_process_guard_domain": (
                "falsewake-exp002-process-guard-filter-v1\x00"
            ),
            "runner_sealed_origin_prefix": "falsewake-sealed://experiment-004",
            "runtime_fingerprint_domain": "falsewake-exp004-runtime-v1\x00",
            "source_bundle_domain": "falsewake-exp004-source-bundle-v1\x00",
        },
    }

    delta = document["execution_delta"]
    assert tuple(delta) == (
        "classification",
        "immutable_predecessor_sources",
        "implementation_source_delta",
        "parity_contract",
        "permitted_existing_source_changes",
        "production_additions",
        "scientific_protocol_change",
        "shared_authority_recipe",
    )
    assert delta["classification"] == "interface_contract_recovery"
    assert delta["scientific_protocol_change"] is False
    assert tuple(delta["production_additions"]) == PRODUCTION_ADDITIONS
    assert delta["permitted_existing_source_changes"] == [
        {
            "path": "src/falsewake/experiment_002_run_authority.py",
            "reference_sha256": (
                "1c19b920df0e75a401bbfcc95c36adfc550b24e77d5e13e5a711d61a0f45f315"
            ),
            "scope": (
                "add_one_fixed_profile_004_without_changing_profiles_002_or_003_"
                "or_scientific_execution_semantics"
            ),
        }
    ]
    assert delta["implementation_source_delta"] == {
        "base_commit": PROTOCOL_PARENT,
        "name_status": [
            {
                "path": "src/falsewake/experiment_002_run_authority.py",
                "status": "M",
            },
            *[{"path": path, "status": "A"} for path in PRODUCTION_ADDITIONS],
        ],
        "other_src_falsewake_changes": "forbidden",
    }
    assert delta["immutable_predecessor_sources"] == {
        "experiment_002_scientific_modules": {
            "document_path": "configs/experiment-003-execution.json",
            "document_sha256": (
                "3f48cb48f6c3a56284f272fa9e308ae62b3749df64cb7a581e770da8bdd8218a"
            ),
            "json_pointer": "/scientific_inheritance/source_blobs",
        },
        "experiment_003_modules": list(P3_SOURCE_HASHES),
    }

    parity = delta["parity_contract"]
    assert parity["candidate"] == "experiment_004"
    assert parity["reference_profile"] == "experiment_003"
    assert parity["reference_commit"] == REFERENCE_IMPLEMENTATION
    assert parity["unrestricted_textual_drift"] == "forbidden"
    assert parity["allowed_semantic_deltas"] == [
        "coordinator_accepts_one_positional_only_verified_state_argument",
        (
            "runner_claims_the_permanent_attempt_marker_before_authority_or_"
            "coordinator_import"
        ),
        "supervisor_binds_the_exact_experiment_004_runner_sha256",
        (
            "final_evidence_and_publication_bind_experiment_003_incident_and_"
            "experiment_004_namespaces"
        ),
        "run_authority_facade_binds_profile_004",
    ]
    assert parity["normalization"] == {
        "direction": "experiment_004_candidate_to_experiment_003_reference",
        "encoding": "UTF-8",
        "ordered_literal_substitutions": [
            {"candidate": "Experiment 004", "reference": "Experiment 003"},
            {"candidate": "Experiment004", "reference": "Experiment003"},
            {"candidate": "experiment_004", "reference": "experiment_003"},
            {"candidate": "experiment-004", "reference": "experiment-003"},
            {"candidate": "exp004", "reference": "exp003"},
            {"candidate": '"004"', "reference": '"003"'},
            {"candidate": "FW4ACTV1", "reference": "FW3ACTV1"},
            {"candidate": "FW4CHLD1", "reference": "FW3CHLD1"},
        ],
        "supervisor_runner_sha256_assignment": {
            "candidate_value": "one_exact_future_64_lowercase_hex_digest",
            "normalize_to_reference_value": (
                "6e9eb84027487b17e4a375fe04f8c704c11aa4d78458405900625d462fcd5a4d"
            ),
            "target": "_RUNNER_SHA256",
        },
    }
    assert [
        (
            recipe["candidate_path"],
            recipe["reference_path"],
            recipe["reference_sha256"],
        )
        for recipe in parity["file_recipes"]
    ] == [
        (
            f"src/falsewake/experiment_004_{suffix}.py",
            binding["path"],
            binding["sha256"],
        )
        for suffix, binding in zip(
            (
                "coordinator",
                "final_evidence",
                "final_publication",
                "run_authority",
                "runner",
                "seed_worker",
                "supervisor",
            ),
            P3_SOURCE_HASHES,
            strict=True,
        )
    ]


def test_predecessor_experiment_003_sources_are_byte_immutable() -> None:
    protocol_bindings = _protocol()["execution_delta"]["immutable_predecessor_sources"][
        "experiment_003_modules"
    ]
    assert tuple(protocol_bindings) == P3_SOURCE_HASHES

    for binding in P3_SOURCE_HASHES:
        current = (ROOT / binding["path"]).read_bytes()
        reference = _git(
            "show",
            f"{REFERENCE_IMPLEMENTATION}:{binding['path']}",
        )
        assert _payload_sha256(current) == binding["sha256"]
        assert _payload_sha256(reference) == binding["sha256"]
        assert current == reference


def test_actual_route_matrix_and_shared_profile_in_isolated_process() -> None:
    assert not os.path.lexists(ATTEMPT_MARKER)
    route_spec = json.dumps(
        [*PARENT_ROUTE_MATRIX, *WRAPPER_ROUTE_MATRIX],
        sort_keys=True,
    )
    source_root = str(ROOT / "src")
    script = f"""
import importlib
import inspect
import json
import sys
from types import FunctionType

sys.dont_write_bytecode = True
sys.path.insert(0, {source_root!r})
route_spec = json.loads({route_spec!r})
required_modules = (
    "falsewake.experiment_004_coordinator",
    "falsewake.experiment_004_final_evidence",
    "falsewake.experiment_004_final_publication",
    "falsewake.experiment_004_run_authority",
    "falsewake.experiment_004_runner",
    "falsewake.experiment_004_seed_worker",
    "falsewake.experiment_004_supervisor",
)
modules = {{name: importlib.import_module(name) for name in required_modules}}
rows = []
for expected in route_spec:
    route = getattr(modules[expected["module"]], expected["binding"])
    code = route.__code__
    rows.append(
        {{
            "binding": expected["binding"],
            "module": route.__module__,
            "name": route.__name__,
            "exact_function": type(route) is FunctionType,
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
profile = engine._EXPERIMENT_004_PROFILE
profile_fields = tuple(profile.__dataclass_fields__)
profile_values = {{}}
for field in profile_fields:
    value = getattr(profile, field)
    if type(value) is bytes:
        value = value.decode("ascii")
    profile_values[field] = value
print(
    json.dumps(
        {{
            "routes": rows,
            "profile_fields": profile_fields,
            "profile_values": profile_values,
            "profiles": [
                item.registration_experiment for item in engine._AUTHORITY_PROFILES
            ],
            "profile_is_last": engine._AUTHORITY_PROFILES[-1] is profile,
        }},
        sort_keys=True,
    )
)
"""
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    completed = subprocess.run(
        (sys.executable, "-I", "-c", script),
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    result = json.loads(completed.stdout)
    assert type(result) is dict

    expected_routes = []
    for row in (*PARENT_ROUTE_MATRIX, *WRAPPER_ROUTE_MATRIX):
        expected_routes.append(
            {
                **row,
                "defaults": None,
                "exact_function": True,
                "keyword_defaults": None,
                "keyword_only_arguments": 0,
                "variable_keyword_arguments": False,
                "variable_positional_arguments": False,
            }
        )
    assert result["routes"] == expected_routes
    assert tuple(result["profile_fields"]) == PROFILE_FIELDS
    assert result["profile_values"] == PROFILE_VALUES
    assert result["profiles"] == ["002", "003", "004"]
    assert result["profile_is_last"] is True
    assert not os.path.lexists(ATTEMPT_MARKER)


def test_coordinator_ast_claims_every_parent_route_signature() -> None:
    source = (ROOT / "src/falsewake/experiment_004_coordinator.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
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
        "run_authority_namespace": "falsewake.experiment_004_run_authority",
        "supervisor_namespace": "falsewake.experiment_004_supervisor",
        "evidence_namespace": "falsewake.experiment_004_final_evidence",
        "publication_namespace": "falsewake.experiment_004_final_publication",
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
        binding_node = lookup.args[1]
        name_node = call.args[1]
        module_node = call.args[2]
        assert isinstance(module_node, ast.Name)
        expected_module_variable = {
            "run_authority_namespace": "run_authority_module_name",
            "supervisor_namespace": "supervisor_module_name",
            "evidence_namespace": "evidence_module_name",
            "publication_namespace": "publication_module_name",
        }[namespace_name]
        assert module_node.id == expected_module_variable
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
        positional_only = ast.literal_eval(call.args[3])
        positional_or_keyword = ast.literal_eval(call.args[4])
        assert type(positional_only) is int
        assert type(positional_or_keyword) is int
        static_rows.append(
            {
                "binding": binding_node.value,
                "module": module_by_namespace[namespace_name],
                "name": name_node.value,
                "positional_only_arguments": positional_only,
                "positional_or_keyword_arguments": positional_or_keyword,
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
            "module": "falsewake.experiment_004_supervisor",
            "name": name,
            "positional_only_arguments": 0,
            "positional_or_keyword_arguments": 0,
        }
        for binding, name in lifecycle_bindings
    )
    assert {row["binding"]: row for row in static_rows} == {
        row["binding"]: row for row in PARENT_ROUTE_MATRIX
    }


def test_run_authority_facade_has_only_fixed_profile_004_targets() -> None:
    source = (ROOT / "src/falsewake/experiment_004_run_authority.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    expected_statements = {
        "verify_and_issue_experiment_004_run_registration": [
            "return _engine._verify_and_issue_experiment_004_run_registration()"
        ],
        "_verify_and_issue_experiment_004_sealed_child_registration": [
            (
                "return _engine."
                "_verify_and_issue_experiment_004_sealed_child_registration()"
            )
        ],
        "verify_verified_run_registration": [
            "_engine._verify_experiment_004_run_registration(registration)"
        ],
        "reverify_verified_run_registration": [
            "_engine._reverify_experiment_004_run_registration(registration)"
        ],
        "_registered_child_input_snapshot": [
            (
                "_engine._require_verified_registration_profile("
                "registration, _engine._EXPERIMENT_004_PROFILE)"
            ),
            "return _engine._registered_child_input_snapshot(registration)",
        ],
        "_create_sealed_experiment_004_child_bundle_fd": [
            (
                "return _engine."
                "_create_sealed_experiment_004_child_bundle_fd(registration)"
            )
        ],
        "_verified_state": [
            (
                "return _engine._require_verified_registration_profile("
                "registration, _engine._EXPERIMENT_004_PROFILE)"
            )
        ],
        "_required_child_bundle_seals": [
            "return _engine._required_child_bundle_seals()"
        ],
    }
    assert tuple(functions) == tuple(expected_statements)
    for name, expected in expected_statements.items():
        function = functions[name]
        assert [ast.unparse(node) for node in _without_docstring(function)] == expected
        assert all(argument.arg != "profile" for argument in function.args.args)
        assert all(argument.arg != "profile" for argument in function.args.posonlyargs)
        assert not function.args.kwonlyargs
        assert function.args.vararg is None
        assert function.args.kwarg is None

    assignments = {
        node.targets[0].id: ast.unparse(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    assert assignments["_SEALED_CHILD_MEMFD_TARGET"] == (
        "_engine._EXPERIMENT_004_SEALED_CHILD_MEMFD_TARGET"
    )
    assert "_EXPERIMENT_003_PROFILE" not in source
    assert "_EXPERIMENT_002_PROFILE" not in source


def _remove_attempt_marker_regions(source: str) -> str:
    tree = ast.parse(source)
    marker_symbols = {
        "_ATTEMPT_MARKER_BYTES",
        "_ATTEMPT_MARKER_CREATION_MODE",
        "_ATTEMPT_MARKER_DIRECTORY",
        "_ATTEMPT_MARKER_FINAL_MODE",
        "_ATTEMPT_MARKER_LEAF",
        "_ATTEMPT_MARKER_PATH",
    }
    regions: list[tuple[int, int]] = []
    observed_symbols: set[str] = set()
    top_level = list(tree.body)
    for index, node in enumerate(top_level):
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id in marker_symbols
        ):
            assert node.end_lineno is not None
            observed_symbols.add(node.target.id)
            regions.append((node.lineno, node.end_lineno))
        if isinstance(node, ast.FunctionDef) and node.name == (
            "_claim_permanent_parent_attempt"
        ):
            assert index + 1 < len(top_level)
            next_node = top_level[index + 1]
            assert isinstance(next_node, ast.FunctionDef)
            assert next_node.name == "_issue_registration"
            regions.append((node.lineno, next_node.lineno - 1))
    assert observed_symbols == marker_symbols

    main_functions = [
        node
        for node in top_level
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    ]
    assert len(main_functions) == 1
    claim_calls = [
        node
        for node in ast.walk(main_functions[0])
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "_claim_permanent_parent_attempt"
    ]
    assert len(claim_calls) == 1
    call = claim_calls[0]
    assert call.end_lineno is not None
    regions.append((call.lineno, call.end_lineno))

    removed_lines: set[int] = set()
    for first, last in regions:
        assert first <= last
        assert not removed_lines.intersection(range(first, last + 1))
        removed_lines.update(range(first, last + 1))
    return "".join(
        line
        for line_number, line in enumerate(source.splitlines(keepends=True), start=1)
        if line_number not in removed_lines
    )


def test_feasible_p4_to_p3_source_recipes_are_byte_exact() -> None:
    document = _protocol()

    seed_candidate = _normalize_literals(
        (ROOT / "src/falsewake/experiment_004_seed_worker.py").read_text(
            encoding="utf-8"
        ),
        document,
    )
    seed_reference = (ROOT / "src/falsewake/experiment_003_seed_worker.py").read_text(
        encoding="utf-8"
    )
    assert seed_candidate == seed_reference

    coordinator_candidate = _normalize_literals(
        (ROOT / "src/falsewake/experiment_004_coordinator.py").read_text(
            encoding="utf-8"
        ),
        document,
    )
    candidate_signature = (
        '            "_verified_state",\n'
        "            run_authority_module_name,\n"
        "            1,\n"
        "            0,\n"
    )
    reference_signature = (
        '            "_verified_state",\n'
        "            run_authority_module_name,\n"
        "            0,\n"
        "            1,\n"
    )
    assert coordinator_candidate.count(candidate_signature) == 1
    coordinator_candidate = coordinator_candidate.replace(
        candidate_signature,
        reference_signature,
    )
    coordinator_reference = (
        ROOT / "src/falsewake/experiment_003_coordinator.py"
    ).read_text(encoding="utf-8")
    assert coordinator_candidate == coordinator_reference

    runner_path = ROOT / "src/falsewake/experiment_004_runner.py"
    runner_sha256 = _sha256(runner_path)
    supervisor_candidate = (
        ROOT / "src/falsewake/experiment_004_supervisor.py"
    ).read_text(encoding="utf-8")
    runner_assignment = re.compile(
        r'(_RUNNER_SHA256: Final = \(\n    ")([0-9a-f]{64})("\n\))'
    )
    matches = list(runner_assignment.finditer(supervisor_candidate))
    assert len(matches) == 1
    assert matches[0].group(2) == runner_sha256
    reference_runner_sha256 = document["execution_delta"]["parity_contract"][
        "normalization"
    ]["supervisor_runner_sha256_assignment"]["normalize_to_reference_value"]
    supervisor_candidate = runner_assignment.sub(
        rf"\g<1>{reference_runner_sha256}\g<3>",
        supervisor_candidate,
        count=1,
    )
    supervisor_candidate = _normalize_literals(supervisor_candidate, document)
    supervisor_reference = (
        ROOT / "src/falsewake/experiment_003_supervisor.py"
    ).read_text(encoding="utf-8")
    assert supervisor_candidate == supervisor_reference

    runner_candidate = _normalize_literals(
        runner_path.read_text(encoding="utf-8"),
        document,
    )
    runner_candidate = _remove_attempt_marker_regions(runner_candidate)
    runner_reference = (ROOT / "src/falsewake/experiment_003_runner.py").read_text(
        encoding="utf-8"
    )
    assert runner_candidate == runner_reference
    assert not os.path.lexists(ATTEMPT_MARKER)
