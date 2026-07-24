from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/experiment-006-execution.json"
CANDIDATE_PATH = ROOT / "src/falsewake/experiment_006_supervisor.py"
RUNNER_PATH = ROOT / "src/falsewake/experiment_006_runner.py"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _parity_contract() -> dict[str, Any]:
    protocol = cast(
        dict[str, Any],
        json.loads(PROTOCOL_PATH.read_text(encoding="utf-8")),
    )
    return cast(
        dict[str, Any],
        protocol["execution_delta"]["parity_contract"],
    )


def test_supervisor_is_exact_protocol_normalized_parity() -> None:
    parity = _parity_contract()
    normalization = cast(dict[str, Any], parity["normalization"])
    encoding = cast(str, normalization["encoding"])
    candidate_bytes = CANDIDATE_PATH.read_bytes()
    candidate_text = candidate_bytes.decode(encoding)
    runner_bytes = RUNNER_PATH.read_bytes()
    assert "Experiment 006" in candidate_text
    assert "falsewake-experiment-006-scratch" in candidate_text
    assert "falsewake-experiment-006-staging" in candidate_text
    assert "falsewake-exp006-child-activation" in candidate_text
    assert (
        candidate_text.count("from falsewake.experiment_002_child_result import") == 1
    )
    assert "Experiment 005" not in candidate_text
    assert "experiment_005" not in candidate_text
    assert "Experiment 004" not in candidate_text
    assert "experiment_004" not in candidate_text

    runner_matches = re.findall(
        r'^_RUNNER_SHA256: Final = \(\n    "([0-9a-f]{64})"\n\)$',
        candidate_text,
        flags=re.MULTILINE,
    )
    assert runner_matches == [_sha256(runner_bytes)]

    candidate_assignment = f'_RUNNER_SHA256: Final = (\n    "{runner_matches[0]}"\n)'
    runner_assignment = cast(
        dict[str, str],
        normalization["supervisor_runner_sha256_assignment"],
    )
    reference_assignment = (
        '_RUNNER_SHA256: Final = (\n    "'
        f'{runner_assignment["normalize_to_reference_value"]}"\n)'
    )
    assert candidate_text.count(candidate_assignment) == 1
    normalized_text = candidate_text.replace(
        candidate_assignment,
        reference_assignment,
        1,
    )
    for substitution in cast(
        list[dict[str, str]],
        normalization["ordered_literal_substitutions"],
    ):
        candidate_literal = substitution["candidate"]
        normalized_text = normalized_text.replace(
            candidate_literal,
            substitution["reference"],
        )

    recipes = cast(list[dict[str, Any]], parity["file_recipes"])
    recipe = next(
        value
        for value in recipes
        if value["candidate_path"] == "src/falsewake/experiment_006_supervisor.py"
    )
    reference_path = ROOT / cast(str, recipe["reference_path"])
    reference_bytes = reference_path.read_bytes()
    assert _sha256(reference_bytes) == recipe["reference_sha256"]
    assert normalized_text.encode(encoding) == reference_bytes


def test_supervisor_imports_only_the_inherited_child_result_module() -> None:
    tree = ast.parse(CANDIDATE_PATH.read_bytes(), filename=str(CANDIDATE_PATH))
    falsewake_imports = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith("falsewake.")
    ]

    assert falsewake_imports == ["falsewake.experiment_002_child_result"]
