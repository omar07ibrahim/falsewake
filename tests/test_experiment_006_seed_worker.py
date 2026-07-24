from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs" / "experiment-006-execution.json"
REFERENCE_PATH = ROOT / "src" / "falsewake" / "experiment_005_seed_worker.py"
CANDIDATE_PATH = ROOT / "src" / "falsewake" / "experiment_006_seed_worker.py"
REFERENCE_SHA256 = "0840a62f0f4327eec7649d17ad334edfa6941d6146bff6fe50e2d693f0b80f96"


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
        candidate = substitution["candidate"]
        reference = substitution["reference"]
        assert type(candidate) is str
        assert type(reference) is str
        source = source.replace(candidate, reference)
    return source


def test_seed_worker_is_the_exact_frozen_experiment_006_parity_delta() -> None:
    document = _protocol()
    recipes = document["execution_delta"]["parity_contract"]["file_recipes"]
    recipe = next(
        item
        for item in recipes
        if item["candidate_path"] == "src/falsewake/experiment_006_seed_worker.py"
    )
    assert recipe == {
        "candidate_path": "src/falsewake/experiment_006_seed_worker.py",
        "normalization": {
            "global_literals": (
                "parity_contract.normalization.ordered_literal_substitutions"
            ),
            "post_normalization_edits": [],
            "result": "byte_identical_to_reference",
        },
        "reference_path": "src/falsewake/experiment_005_seed_worker.py",
        "reference_sha256": REFERENCE_SHA256,
    }
    reference = REFERENCE_PATH.read_text(encoding="utf-8")
    assert hashlib.sha256(reference.encode()).hexdigest() == REFERENCE_SHA256
    assert _normalized_candidate(document) == reference


def test_seed_worker_uses_profile_006_activation_and_authority() -> None:
    source = CANDIDATE_PATH.read_text(encoding="utf-8")

    assert "from falsewake.experiment_006_coordinator import (" in source
    assert "from falsewake.experiment_006_run_authority import (" in source
    assert "Experiment006SeedWorkerError" in source
    assert "Experiment005" not in source
    assert "experiment_005" not in source
    assert "experiment-005" not in source
    assert "exp005" not in source
    assert "FW5ACTV1" not in source
    assert "FW5CHLD1" not in source


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
    assert "falsewake.experiment_006_child_result" not in source
    assert "falsewake.experiment_006_registered_" not in source


def test_seed_worker_wrapper_has_the_exact_frozen_signature_without_calling_it() -> (
    None
):
    document = _protocol()
    wrapper = next(
        route
        for route in document["admission_contract"]["wrapper_route_matrix"]
        if route["binding"] == "run_registered_seed_process"
    )
    tree = ast.parse(
        CANDIDATE_PATH.read_text(encoding="utf-8"),
        filename=str(CANDIDATE_PATH),
    )
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "run_registered_seed_process"
    ]
    assert len(functions) == 1
    arguments = functions[0].args
    assert len(arguments.posonlyargs) == wrapper["positional_only_arguments"] == 2
    assert len(arguments.args) == wrapper["positional_or_keyword_arguments"] == 0
    assert not arguments.kwonlyargs
    assert arguments.vararg is None
    assert arguments.kwarg is None
    assert not arguments.defaults
    assert not arguments.kw_defaults
