from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "src" / "falsewake" / "experiment_002_coordinator.py"
CANDIDATE_PATH = ROOT / "src" / "falsewake" / "experiment_003_coordinator.py"


def _replace_once(source: str, old: str, new: str) -> str:
    assert source.count(old) == 1, old
    return source.replace(old, new)


def _expected_coordinator_source() -> str:
    source = REFERENCE_PATH.read_text(encoding="utf-8")
    source = source.replace("Experiment 002", "Experiment 003")
    source = source.replace(
        "Experiment002Coordinator",
        "Experiment003Coordinator",
    )
    source = _replace_once(
        source,
        "from falsewake.experiment_002_run_authority import",
        "from falsewake.experiment_003_run_authority import",
    )
    replacements = (
        (
            '"falsewake.experiment_002_seed_worker"',
            '"falsewake.experiment_003_seed_worker"',
        ),
        (
            '"falsewake.experiment_002_supervisor"',
            '"falsewake.experiment_003_supervisor"',
        ),
        (
            '"falsewake.experiment_002_run_authority"',
            '"falsewake.experiment_003_run_authority"',
        ),
        (
            '"falsewake.experiment_002_final_evidence"',
            '"falsewake.experiment_003_final_evidence"',
        ),
        (
            '"falsewake.experiment_002_final_publication"',
            '"falsewake.experiment_003_final_publication"',
        ),
        ("FW2ACTV1", "FW3ACTV1"),
        (
            "falsewake-exp002-activation-ticket-v1",
            "falsewake-exp003-activation-ticket-v1",
        ),
    )
    for old, new in replacements:
        source = _replace_once(source, old, new)
    source = source.replace(
        "_create_sealed_experiment_002_child_bundle_fd",
        "_create_sealed_experiment_003_child_bundle_fd",
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


def test_coordinator_is_the_exact_registered_experiment_003_delta() -> None:
    assert CANDIDATE_PATH.read_text(encoding="utf-8") == (
        _expected_coordinator_source()
    )


def test_coordinator_uses_the_committed_shared_authority_map() -> None:
    source = CANDIDATE_PATH.read_text(encoding="utf-8")

    assert '_WORKER_MODULE: Final = "falsewake.experiment_003_seed_worker"' in source
    assert (
        '_PROCESS_GUARD_MODULE: Final = "falsewake.experiment_002_process_guard"'
        in source
    )
    assert '_SUPERVISOR_MODULE: Final = "falsewake.experiment_003_supervisor"' in source
    assert (
        '_RUN_AUTHORITY_MODULE: Final = "falsewake.experiment_003_run_authority"'
        in source
    )
    assert (
        '_FINAL_EVIDENCE_MODULE: Final = "falsewake.experiment_003_final_evidence"'
        in source
    )
    assert (
        "_FINAL_PUBLICATION_MODULE: Final = "
        '"falsewake.experiment_003_final_publication"' in source
    )
    assert (
        '_CHILD_RESULT_MODULE: Final = "falsewake.experiment_002_child_result"'
        in source
    )
    assert "falsewake.experiment_003_process_guard" not in source
    assert "falsewake.experiment_003_child_result" not in source


def test_coordinator_binds_activation_and_evidence_to_profile_003() -> None:
    source = CANDIDATE_PATH.read_text(encoding="utf-8")

    assert '_FRAME_MAGIC: Final = b"FW3ACTV1"' in source
    assert (
        "_TICKET_DIGEST_DOMAIN: Final = "
        'b"falsewake-exp003-activation-ticket-v1\\0"' in source
    )
    assert 'history_domain = b"falsewake-exp002-history-v1\\0"' in source
    assert 'envelope_domain = b"falsewake-exp002-child-result-envelope-v1\\0"' in source
    assert "_create_sealed_experiment_003_child_bundle_fd" in source
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
