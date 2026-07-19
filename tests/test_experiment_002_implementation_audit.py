from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, NoReturn, cast

AUDIT_PATH = Path("reports/experiment-002-implementation-audit.json")
METRICS_PATH = Path("src/falsewake/experiment_002_metrics.py")
TRAINER_PATH = Path("configs/experiment-002-trainer.json")
AUDIT_SHA256 = "0b8033e7f407550d1c8576d31485e5bbcf825592de0d3b86f42ce98c1fca7116"
TRAINER_SHA256 = "768720d031a1f2f6d0a36466fd9b4fa095a2f1b9a6fbbc382d4af5fc51514b48"


def _reject_nonfinite(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _document(path: Path) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite,
        ),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_implementation_audit_is_canonical_and_bound_to_the_incident() -> None:
    raw = AUDIT_PATH.read_text(encoding="utf-8")
    audit = _document(AUDIT_PATH)

    assert _sha256(AUDIT_PATH) == AUDIT_SHA256
    assert raw == (
        json.dumps(
            audit,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )
    assert set(audit) == {
        "bindings",
        "chronology",
        "contract_assessment",
        "experiment",
        "inputs",
        "outputs",
        "probes",
        "resolution",
        "schema_version",
        "scope_exclusions",
    }
    assert audit["schema_version"] == 1
    assert audit["experiment"] == "002"

    trainer_binding = audit["bindings"]["trainer_registration"]
    assert trainer_binding == {
        "commit": "28cb87ecd73e4e12f7fb4cb6fe49912b0951a35b",
        "path": TRAINER_PATH.as_posix(),
        "sha256": TRAINER_SHA256,
    }
    assert _sha256(TRAINER_PATH) == TRAINER_SHA256
    assert (
        "no_registered_validation_metric"
        in (_document(TRAINER_PATH)["registration"]["run_prohibition"])
    )
    assert audit["bindings"]["metrics_implementation"] == {
        "last_committed_commit": "4713acb44b9656f8465bb62019480dcd257efe1b",
        "review_candidate_source_sha256": (
            "c3cfd393b41f806e8fcce17af81bccad888e7a7b7f462a4744e71ac165f7e379"
        ),
        "review_candidate_tests_sha256": (
            "3e044b248dd66cce62c46b4e8a773e7886bd28850d3601f9261fcdb95b2464a9"
        ),
    }


def test_audit_records_the_literal_boundary_crossing_without_result_evidence() -> None:
    audit = _document(AUDIT_PATH)
    chronology = audit["chronology"]
    assessment = audit["contract_assessment"]

    assert chronology == {
        "occurred_after_trainer_registration": True,
        "occurred_before_source_bound_run_registration": True,
        "run_registration_path": "configs/experiment-002-run.json",
        "run_registration_present_at_probe_time": False,
    }
    assert assessment["literal_trainer_run_prohibition_boundary_crossed"] is True
    assert (
        assessment["population_bound_registered_validation_evidence_created"] is False
    )
    assert assessment["scientific_performance_observed"] is False
    assert assessment["selection_or_gate_decision_made"] is False
    assert all(value is False for value in audit["outputs"].values())
    assert all(value is False for value in audit["scope_exclusions"].values())

    labels = audit["inputs"]["label_indices"]
    logits = audit["inputs"]["frame_97_logits"]
    assert labels["shape"] == [10_583]
    assert sum(labels["class_support"]) == 10_583
    assert labels["example_identity_bound"] is False
    assert logits["shape"] == [10_583, 12]
    assert logits["values"] == "all_numeric_zero"


def test_audit_reconciles_every_discarded_synthetic_scoring() -> None:
    audit = _document(AUDIT_PATH)
    probes = audit["probes"]
    public = probes["public_layout_marked_route"]
    private = probes["private_unmarked_route"]

    assert public["computations"] == 2
    assert public["examples_per_computation"] == 10_583
    assert public["examples_scored"] == 2 * 10_583 == 21_166
    assert public["historical_registered_issuance_marker_present"] is True
    assert public["registered_layout_verifier_acceptance_observed"] is True
    assert private["computations"] == 1
    assert private["examples_per_computation"] == 10_583
    assert private["examples_scored"] == 10_583
    assert private["layout_issuance_marker_present"] is False
    assert probes["totals"] == {
        "computations": public["computations"] + private["computations"],
        "examples_scored": public["examples_scored"] + private["examples_scored"],
    }
    assert probes["totals"] == {"computations": 3, "examples_scored": 31_749}


def test_metrics_api_exposes_layout_only_names_after_the_audit() -> None:
    source = METRICS_PATH.read_text(encoding="utf-8")
    resolution = _document(AUDIT_PATH)["resolution"]

    assert "def evaluate_registered_validation_logits" not in source
    assert "def verify_registered_validation_metrics" not in source
    assert (
        f"def {resolution['layout_only_public_entrypoint']}" in source
        and f"def {resolution['layout_only_verifier']}" in source
    )
    assert resolution["full_shape_synthetic_scoring_before_run_registration_forbidden"]
    assert resolution["run_registration_must_bind_this_report"]
