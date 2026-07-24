from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs" / "experiment-006-execution.json"
REFERENCE_PATH = ROOT / "src" / "falsewake" / "experiment_005_coordinator.py"
CANDIDATE_PATH = ROOT / "src" / "falsewake" / "experiment_006_coordinator.py"
ATTEMPT_MARKER = Path("/home/ubuntu/gitcode/.t/falsewake-experiment-006-attempt")
PROTOCOL_SHA256 = "47ce6c6f11c3575944ef8c4ab1edcc0b42fdfa9bf1ede0318bfc5ea61fabf650"
REFERENCE_SHA256 = "4ad78de14ce12acdb63b4e103dd920ce878d7e60a06fa1c5893027d17401f812"


def _protocol() -> dict[str, Any]:
    document = json.loads(PROTOCOL_PATH.read_text(encoding="ascii"))
    assert type(document) is dict
    return document


def _normalized_candidate(document: dict[str, Any]) -> str:
    source = CANDIDATE_PATH.read_text(encoding="utf-8")
    substitutions = document["execution_delta"]["parity_contract"]["normalization"][
        "ordered_literal_substitutions"
    ]
    assert type(substitutions) is list
    assert len(substitutions) == 13
    for substitution in substitutions:
        assert type(substitution) is dict
        assert tuple(substitution) == ("candidate", "reference")
        candidate = substitution["candidate"]
        reference = substitution["reference"]
        assert type(candidate) is str
        assert type(reference) is str
        source = source.replace(candidate, reference)
    return source


def test_coordinator_is_the_exact_frozen_experiment_006_parity_delta() -> None:
    protocol_bytes = PROTOCOL_PATH.read_bytes()
    assert hashlib.sha256(protocol_bytes).hexdigest() == PROTOCOL_SHA256
    document = _protocol()
    recipes = document["execution_delta"]["parity_contract"]["file_recipes"]
    recipe = next(
        item
        for item in recipes
        if item["candidate_path"] == "src/falsewake/experiment_006_coordinator.py"
    )
    assert recipe == {
        "candidate_path": "src/falsewake/experiment_006_coordinator.py",
        "normalization": {
            "global_literals": (
                "parity_contract.normalization.ordered_literal_substitutions"
            ),
            "post_normalization_edits": [],
            "result": "byte_identical_to_reference",
        },
        "reference_path": "src/falsewake/experiment_005_coordinator.py",
        "reference_sha256": REFERENCE_SHA256,
    }
    reference = REFERENCE_PATH.read_text(encoding="utf-8")
    assert hashlib.sha256(reference.encode()).hexdigest() == REFERENCE_SHA256
    assert _normalized_candidate(document) == reference


def test_coordinator_uses_only_profile_006_control_namespaces() -> None:
    source = CANDIDATE_PATH.read_text(encoding="utf-8")

    expected_control_bindings = (
        '_WORKER_MODULE: Final = "falsewake.experiment_006_seed_worker"',
        '_SUPERVISOR_MODULE: Final = "falsewake.experiment_006_supervisor"',
        ('_RUN_AUTHORITY_MODULE: Final = "falsewake.experiment_006_run_authority"'),
        ('_FINAL_EVIDENCE_MODULE: Final = "falsewake.experiment_006_final_evidence"'),
        (
            "_FINAL_PUBLICATION_MODULE: Final = "
            '"falsewake.experiment_006_final_publication"'
        ),
        '_FRAME_MAGIC: Final = b"FW6ACTV1"',
        ('_TICKET_DIGEST_DOMAIN: Final = b"falsewake-exp006-activation-ticket-v1\\0"'),
        "_create_sealed_experiment_006_child_bundle_fd",
    )
    for binding in expected_control_bindings:
        assert binding in source

    assert "Experiment005" not in source
    assert "experiment_005" not in source
    assert "experiment-005" not in source
    assert "exp005" not in source
    assert "FW5ACTV1" not in source
    assert "FW5CHLD1" not in source


def test_coordinator_retains_the_registered_experiment_002_scientific_domains() -> None:
    source = CANDIDATE_PATH.read_text(encoding="utf-8")

    assert (
        '_PROCESS_GUARD_MODULE: Final = "falsewake.experiment_002_process_guard"'
        in source
    )
    assert (
        '_CHILD_RESULT_MODULE: Final = "falsewake.experiment_002_child_result"'
        in source
    )
    assert 'history_domain = b"falsewake-exp002-history-v1\\0"' in source
    assert 'envelope_domain = b"falsewake-exp002-child-result-envelope-v1\\0"' in source
    assert "falsewake.experiment_006_process_guard" not in source
    assert "falsewake.experiment_006_child_result" not in source


def test_coordinator_ast_claims_every_frozen_parent_route_signature() -> None:
    document = _protocol()
    expected_rows = document["admission_contract"]["parent_route_matrix"]
    assert type(expected_rows) is list
    assert len(expected_rows) == 15

    tree = ast.parse(
        CANDIDATE_PATH.read_text(encoding="utf-8"),
        filename=str(CANDIDATE_PATH),
    )
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


def test_all_frozen_routes_have_live_exact_functions_in_an_isolated_process() -> None:
    assert not os.path.lexists(ATTEMPT_MARKER)
    document = _protocol()
    admission = document["admission_contract"]
    routes = [
        *admission["parent_route_matrix"],
        *admission["wrapper_route_matrix"],
    ]
    assert len(routes) == 19
    required_modules = admission["pre_registration_proof"]["actual_modules_required"]
    script = f"""
import importlib
import inspect
import json
import os
import sys
from types import FunctionType

sys.dont_write_bytecode = True
sys.path.insert(0, {str(ROOT / "src")!r})
marker = {str(ATTEMPT_MARKER)!r}
assert not os.path.lexists(marker)
route_spec = json.loads({json.dumps(routes, sort_keys=True)!r})
required_modules = json.loads({json.dumps(required_modules)!r})
modules = {{name: importlib.import_module(name) for name in required_modules}}
observed = []
for expected in route_spec:
    route = vars(modules[expected["module"]]).get(expected["binding"])
    assert type(route) is FunctionType
    code = route.__code__
    observed.append(
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
assert not os.path.lexists(marker)
print(json.dumps(observed, sort_keys=True))
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
    expected = [
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
    assert observed == expected
    assert not os.path.lexists(ATTEMPT_MARKER)
