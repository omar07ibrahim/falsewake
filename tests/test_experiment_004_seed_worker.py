from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "src" / "falsewake" / "experiment_002_seed_worker.py"
CANDIDATE_PATH = ROOT / "src" / "falsewake" / "experiment_004_seed_worker.py"


def _expected_seed_worker_source() -> str:
    source = REFERENCE_PATH.read_text(encoding="utf-8")
    replacements = (
        ("Experiment 002 seed", "Experiment 004 seed"),
        (
            "falsewake.experiment_002_coordinator",
            "falsewake.experiment_004_coordinator",
        ),
        (
            "falsewake.experiment_002_run_authority",
            "falsewake.experiment_004_run_authority",
        ),
        ("Experiment002SeedWorker", "Experiment004SeedWorker"),
    )
    for old, new in replacements:
        assert old in source
        source = source.replace(old, new)
    coordinator_import = (
        "from falsewake.experiment_004_coordinator import (\n"
        "    REGISTERED_SEEDS,\n"
        "    VerifiedChildActivation,\n"
        "    verify_verified_child_activation,\n"
        ")\n"
    )
    authority_import = (
        "from falsewake.experiment_004_run_authority import (\n"
        "    VerifiedRunRegistration,\n"
        "    _registered_child_input_snapshot,\n"
        "    _RegisteredChildInputSnapshot,\n"
        ")\n"
    )
    assert source.count(coordinator_import) == 1
    assert source.count(authority_import) == 1
    source = source.replace(coordinator_import, "")
    source = source.replace(authority_import, "")
    validation_import = (
        "from falsewake.experiment_002_validation import "
        "materialize_registered_validation\n"
    )
    assert source.count(validation_import) == 1
    source = source.replace(
        validation_import,
        validation_import + coordinator_import + authority_import,
    )
    return source


def test_seed_worker_is_the_exact_registered_experiment_004_delta() -> None:
    assert CANDIDATE_PATH.read_text(encoding="utf-8") == (
        _expected_seed_worker_source()
    )


def test_seed_worker_uses_profile_004_activation_and_authority() -> None:
    source = CANDIDATE_PATH.read_text(encoding="utf-8")

    assert "from falsewake.experiment_004_coordinator import (" in source
    assert "from falsewake.experiment_004_run_authority import (" in source
    assert "Experiment004SeedWorkerError" in source
    assert "Experiment002SeedWorker" not in source
    assert "falsewake.experiment_002_coordinator" not in source
    assert "falsewake.experiment_002_run_authority" not in source


def test_seed_worker_inherits_the_registered_scientific_pipeline() -> None:
    source = CANDIDATE_PATH.read_text(encoding="utf-8")

    inherited_modules = (
        "experiment_002_child_result",
        "experiment_002_data",
        "experiment_002_evidence",
        "experiment_002_normalization_artifact",
        "experiment_002_pcm_cache",
        "experiment_002_registered_checkpoint",
        "experiment_002_registered_evaluator",
        "experiment_002_registered_executor",
        "experiment_002_registered_history",
        "experiment_002_training_population",
        "experiment_002_validation",
    )
    for module_name in inherited_modules:
        assert f"from falsewake.{module_name} import" in source

    assert "Experiment002PCMCache" in source
    assert '_HISTORY_DOMAIN: Final = b"falsewake-exp002-history-v1\\0"' in source
    assert "falsewake.experiment_004_child_result" not in source
    assert "falsewake.experiment_004_registered_" not in source
