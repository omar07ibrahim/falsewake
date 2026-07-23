from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import FunctionType

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "src" / "falsewake" / "experiment_002_coordinator.py"
CANDIDATE_PATH = ROOT / "src" / "falsewake" / "experiment_004_coordinator.py"


def _replace_once(source: str, old: str, new: str) -> str:
    assert source.count(old) == 1, old
    return source.replace(old, new)


def _expected_coordinator_source() -> str:
    source = REFERENCE_PATH.read_text(encoding="utf-8")
    source = source.replace("Experiment 002", "Experiment 004")
    source = source.replace(
        "Experiment002Coordinator",
        "Experiment004Coordinator",
    )
    source = _replace_once(
        source,
        "from falsewake.experiment_002_run_authority import",
        "from falsewake.experiment_004_run_authority import",
    )
    replacements = (
        (
            '"falsewake.experiment_002_seed_worker"',
            '"falsewake.experiment_004_seed_worker"',
        ),
        (
            '"falsewake.experiment_002_supervisor"',
            '"falsewake.experiment_004_supervisor"',
        ),
        (
            '"falsewake.experiment_002_run_authority"',
            '"falsewake.experiment_004_run_authority"',
        ),
        (
            '"falsewake.experiment_002_final_evidence"',
            '"falsewake.experiment_004_final_evidence"',
        ),
        (
            '"falsewake.experiment_002_final_publication"',
            '"falsewake.experiment_004_final_publication"',
        ),
        ("FW2ACTV1", "FW4ACTV1"),
        (
            "falsewake-exp002-activation-ticket-v1",
            "falsewake-exp004-activation-ticket-v1",
        ),
    )
    for old, new in replacements:
        source = _replace_once(source, old, new)
    source = source.replace(
        "_create_sealed_experiment_002_child_bundle_fd",
        "_create_sealed_experiment_004_child_bundle_fd",
    )
    source = _replace_once(
        source,
        "build_completed_evidence: Callable[[object], object]",
        (
            "build_completed_evidence: "
            "Callable[[VerifiedRunRegistration, object], object]"
        ),
    )
    source = _replace_once(
        source,
        '"Callable[[object], object], Callable[[object], object], "',
        (
            '"Callable[[VerifiedRunRegistration, object], object], "\n'
            '            "Callable[[object], object], "'
        ),
    )
    source = _replace_once(
        source,
        (
            '            "build_completed_final_evidence",\n'
            "            evidence_module_name,\n"
            "            1,\n"
            "            0,\n"
            "        )"
        ),
        (
            '            "build_completed_final_evidence",\n'
            "            evidence_module_name,\n"
            "            2,\n"
            "            0,\n"
            "        )"
        ),
    )
    source = _replace_once(
        source,
        (
            "        verified_state_route = require_function_identity(\n"
            '            dictionary_get(run_authority_namespace, "_verified_state"),\n'
            '            "_verified_state",\n'
            "            run_authority_module_name,\n"
            "            0,\n"
            "            1,\n"
            "        )"
        ),
        (
            "        verified_state_route = require_function_identity(\n"
            '            dictionary_get(run_authority_namespace, "_verified_state"),\n'
            '            "_verified_state",\n'
            "            run_authority_module_name,\n"
            "            1,\n"
            "            0,\n"
            "        )"
        ),
    )
    return _replace_once(
        source,
        (
            '                "completed-evidence construction",\n'
            "                selection,\n"
        ),
        (
            '                "completed-evidence construction",\n'
            "                registration,\n"
            "                selection,\n"
        ),
    )


def test_coordinator_is_the_exact_registered_experiment_004_delta() -> None:
    assert CANDIDATE_PATH.read_text(encoding="utf-8") == (
        _expected_coordinator_source()
    )


def test_coordinator_uses_the_committed_shared_authority_map() -> None:
    source = CANDIDATE_PATH.read_text(encoding="utf-8")

    assert '_WORKER_MODULE: Final = "falsewake.experiment_004_seed_worker"' in source
    assert (
        '_PROCESS_GUARD_MODULE: Final = "falsewake.experiment_002_process_guard"'
        in source
    )
    assert '_SUPERVISOR_MODULE: Final = "falsewake.experiment_004_supervisor"' in source
    assert (
        '_RUN_AUTHORITY_MODULE: Final = "falsewake.experiment_004_run_authority"'
        in source
    )
    assert (
        '_FINAL_EVIDENCE_MODULE: Final = "falsewake.experiment_004_final_evidence"'
        in source
    )
    assert (
        "_FINAL_PUBLICATION_MODULE: Final = "
        '"falsewake.experiment_004_final_publication"' in source
    )
    assert (
        '_CHILD_RESULT_MODULE: Final = "falsewake.experiment_002_child_result"'
        in source
    )
    assert "falsewake.experiment_004_process_guard" not in source
    assert "falsewake.experiment_004_child_result" not in source


def test_coordinator_binds_activation_and_evidence_to_profile_004() -> None:
    source = CANDIDATE_PATH.read_text(encoding="utf-8")

    assert '_FRAME_MAGIC: Final = b"FW4ACTV1"' in source
    assert (
        "_TICKET_DIGEST_DOMAIN: Final = "
        'b"falsewake-exp004-activation-ticket-v1\\0"' in source
    )
    assert 'history_domain = b"falsewake-exp002-history-v1\\0"' in source
    assert 'envelope_domain = b"falsewake-exp002-child-result-envelope-v1\\0"' in source
    assert "_create_sealed_experiment_004_child_bundle_fd" in source
    assert "_create_sealed_experiment_002_child_bundle_fd" not in source
    assert (
        "build_completed_evidence: "
        "Callable[[VerifiedRunRegistration, object], object]" in source
    )
    assert (
        '                "completed-evidence construction",\n'
        "                registration,\n"
        "                selection,\n" in source
    )
    assert (
        '            "build_completed_final_evidence",\n'
        "            evidence_module_name,\n"
        "            2,\n"
        "            0,\n" in source
    )


def test_live_verified_state_route_matches_the_coordinator_ast_contract() -> None:
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
    calls = [
        node
        for node in ast.walk(claims[0])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "require_function_identity"
        and len(node.args) == 5
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == "_verified_state"
    ]
    assert len(calls) == 1
    call = calls[0]
    assert ast.literal_eval(call.args[3]) == 1
    assert ast.literal_eval(call.args[4]) == 0

    authority = importlib.import_module("falsewake.experiment_004_run_authority")
    route = vars(authority).get("_verified_state")
    assert type(route) is FunctionType
    assert route.__name__ == "_verified_state"
    assert route.__module__ == "falsewake.experiment_004_run_authority"
    assert route.__defaults__ is None
    assert route.__kwdefaults__ is None
    code = route.__code__
    assert code.co_argcount == 1
    assert code.co_posonlyargcount == 1
    assert code.co_argcount - code.co_posonlyargcount == 0
    assert code.co_kwonlyargcount == 0
    assert code.co_flags & 0x0C == 0
