from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
import os
import struct
import weakref
from collections.abc import Iterator
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
from pathlib import Path
from types import FunctionType
from typing import Any, cast

import pytest

import falsewake.experiment_004_final_evidence as final_evidence
from falsewake import experiment_002_run_authority as authority_engine

ROOT = Path(__file__).resolve().parents[1]
_PROTOCOL = json.loads((ROOT / "configs/experiment-004-execution.json").read_bytes())
_PREDECESSOR = _PROTOCOL["predecessor"]
_HISTORY_DOMAIN = b"falsewake-exp002-history-v1\0"
_ENVELOPE_DOMAIN = b"falsewake-exp002-child-result-envelope-v1\0"
_MODEL_DOMAIN = b"falsewake-exp002-model-tensors-v1\0"
_SEEDS = (20_260_719, 20_260_720, 20_260_721)
_SUPPORT = (397, 406, 350, 377, 352, 363, 363, 373, 350, 372, 6_278, 602)
_REPORT_SHA256 = {
    "pass": "373c47f0b693695459b08e737c9be64d65e49c5461c1f561839959ccee5b4c31",
    "gate_failure": "187d9b35e4f29bf40af3c36094eeb86eca0b7db5c611b38ff9bfaa2e45b5e37c",
}

type _Snapshot = tuple[object, ...]
type _Child = tuple[object, ...]
type _Selector = tuple[object, ...]


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _canonical(document: object) -> bytes:
    return (
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _model_specs() -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple(
        sorted(
            (
                *(
                    item
                    for block in range(8)
                    for item in (
                        (f"blocks.{block}.depthwise.weight", (48, 1, 3)),
                        (f"blocks.{block}.depthwise_norm.norm.bias", (48,)),
                        (f"blocks.{block}.depthwise_norm.norm.weight", (48,)),
                        (f"blocks.{block}.pointwise.weight", (48, 48, 1)),
                        (f"blocks.{block}.pointwise_norm.norm.bias", (48,)),
                        (f"blocks.{block}.pointwise_norm.norm.weight", (48,)),
                    )
                ),
                ("classifier.bias", (12,)),
                ("classifier.weight", (12, 48)),
                ("stem.weight", (48, 40, 1)),
                ("stem_norm.norm.bias", (48,)),
                ("stem_norm.norm.weight", (48,)),
            ),
            key=lambda item: item[0].encode("utf-8"),
        )
    )


def _safetensors(variant: int) -> tuple[bytes, str]:
    specs = _model_specs()
    header: dict[str, object] = {}
    data = bytearray()
    framed = bytearray(struct.pack("<I", len(specs)))
    offset = 0
    for tensor_index, (name, shape) in enumerate(specs):
        value_count = math.prod(shape)
        value = (variant * 100 + tensor_index) / 10_000.0
        raw = struct.pack("<f", value) * value_count
        end = offset + len(raw)
        header[name] = {
            "dtype": "F32",
            "shape": list(shape),
            "data_offsets": [offset, end],
        }
        data.extend(raw)
        name_bytes = name.encode("utf-8")
        framed.extend(struct.pack("<I", len(name_bytes)))
        framed.extend(name_bytes)
        framed.extend(struct.pack("<I", 5))
        framed.extend(b"F32LE")
        framed.extend(struct.pack("<I", len(shape)))
        for dimension in shape:
            framed.extend(struct.pack("<Q", dimension))
        framed.extend(struct.pack("<Q", len(raw)))
        framed.extend(raw)
        offset = end
    header_bytes = json.dumps(
        header,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    padded = header_bytes + b" " * (-len(header_bytes) % 8)
    payload = struct.pack("<Q", len(padded)) + padded + bytes(data)
    assert len(specs) == 53
    assert len(data) == 94_896
    assert len(padded) == 4_872
    assert len(payload) == 99_776
    return payload, hashlib.sha256(_MODEL_DOMAIN + bytes(framed)).hexdigest()


def _perfect_confusion() -> list[list[int]]:
    result: list[list[int]] = []
    for index, support in enumerate(_SUPPORT):
        row = [0] * len(_SUPPORT)
        row[index] = support
        result.append(row)
    return result


def _unknown_gate_failure_confusion() -> list[list[int]]:
    result = _perfect_confusion()
    result[10][10] -= 1_256
    result[10][0] += 1_256
    return result


def _recall_gate_failure_confusion() -> list[list[int]]:
    result = _perfect_confusion()
    result[0][0] = 277
    result[0][1] = _SUPPORT[0] - 277
    return result


def _macro_f1(confusion: list[list[int]]) -> Fraction:
    total = Fraction(0, 1)
    for class_index, support in enumerate(_SUPPORT):
        true_positive = confusion[class_index][class_index]
        false_positive = sum(
            confusion[row][class_index]
            for row in range(len(_SUPPORT))
            if row != class_index
        )
        false_negative = support - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        total += (
            Fraction(0, 1)
            if denominator == 0
            else Fraction(2 * true_positive, denominator)
        )
    return total / len(_SUPPORT)


def _history(
    seed: int,
    winner_model_sha256: str,
    winner_ce: float,
    confusion: list[list[int]],
    *,
    winner_epoch: int = 7,
    validation_salt: str = "registered",
    prediction_salt: str = "registered",
    nonwinner_model_salt: str = "registered",
) -> bytes:
    validation_input_digest = _sha(f"validation-{validation_salt}")
    macro_f1 = _macro_f1(confusion)
    epochs: list[dict[str, object]] = []
    for epoch in range(30):
        model_digest = (
            winner_model_sha256
            if epoch == winner_epoch
            else _sha(f"model-{seed}-{epoch}-{nonwinner_model_salt}")
        )
        validation_ce = winner_ce if epoch == winner_epoch else 4.0 + epoch
        epochs.append(
            {
                "epoch_update_trace_digest": _sha(f"epoch-trace-{seed}-{epoch}"),
                "first_global_update": epoch * 313,
                "last_global_update_inclusive": (epoch + 1) * 313 - 1,
                "macro_f1_exact_denominator": macro_f1.denominator,
                "macro_f1_exact_numerator": macro_f1.numerator,
                "model_tensor_digest": model_digest,
                "training_cross_entropy_float64_hex": float(1 + epoch).hex(),
                "training_population_digest": _sha(f"population-{seed}-{epoch}"),
                "validation_confusion_matrix": confusion,
                "validation_cross_entropy_float64_hex": validation_ce.hex(),
                "validation_input_digest": validation_input_digest,
                "validation_prediction_digest": _sha(
                    f"prediction-{seed}-{epoch}-{prediction_salt}"
                ),
                "zero_based_epoch": epoch,
            }
        )
    return _canonical(
        {
            "complete_update_trace_digest": _sha(f"complete-trace-{seed}"),
            "epochs": epochs,
            "experiment": "002",
            "schema_version": 1,
            "seed": seed,
            "validation_input_digest": validation_input_digest,
        }
    )


def _envelope(
    role: str,
    ordinal: int,
    seed: int,
    history: bytes,
    safetensors: bytes,
    model_tensor_sha256: str,
    winner_epoch: int = 7,
) -> bytes:
    return _canonical(
        {
            "experiment": "002",
            "history": {
                "byte_count": len(history),
                "domain_sha256": hashlib.sha256(_HISTORY_DOMAIN + history).hexdigest(),
                "filename": "experiment-002-history.json",
                "model_tensor_sha256": model_tensor_sha256,
                "winner_epoch": winner_epoch,
            },
            "ordinal": ordinal,
            "registration": {
                "head_commit": "a" * 40,
                "implementation_commit": "b" * 40,
                "registration_sha256": "c" * 64,
                "source_bundle_sha256": "d" * 64,
            },
            "role": role,
            "safetensors": {
                "byte_count": len(safetensors),
                "filename": "experiment-002-winner.safetensors",
                "sha256": hashlib.sha256(safetensors).hexdigest(),
            },
            "schema_version": 1,
            "seed": seed,
        }
    )


def _snapshot(
    role: str,
    ordinal: int,
    seed: int,
    history: bytes,
    safetensors: bytes,
    model_tensor_sha256: str,
    winner_ce: float,
    confusion: list[list[int]],
    *,
    winner_epoch: int = 7,
) -> _Snapshot:
    envelope = _envelope(
        role,
        ordinal,
        seed,
        history,
        safetensors,
        model_tensor_sha256,
        winner_epoch,
    )
    macro_f1 = _macro_f1(confusion)
    return (
        role,
        ordinal,
        seed,
        "a" * 40,
        "b" * 40,
        "c" * 64,
        "d" * 64,
        history,
        len(history),
        hashlib.sha256(_HISTORY_DOMAIN + history).hexdigest(),
        safetensors,
        len(safetensors),
        hashlib.sha256(safetensors).hexdigest(),
        envelope,
        len(envelope),
        hashlib.sha256(_ENVELOPE_DOMAIN + envelope).hexdigest(),
        winner_epoch,
        macro_f1.numerator,
        macro_f1.denominator,
        winner_ce.hex(),
        model_tensor_sha256,
    )


def _child(
    pid: int,
    snapshot: _Snapshot,
    *,
    cpu_ids: tuple[int, int] = (2, 3),
) -> _Child:
    return (pid, cpu_ids, 10_000 + pid, 1_000_000 + pid, 200_000 + pid, snapshot)


def _selector(
    confusion: list[list[int]] | None = None,
    *,
    confusions: tuple[list[list[int]], list[list[int]], list[list[int]]] | None = None,
    winner_cross_entropies: tuple[float, float, float] = (0.4, 0.2, 0.6),
    winner_epochs: tuple[int, int, int] = (7, 7, 7),
    selected: int = 1,
) -> _Selector:
    if confusion is not None and confusions is not None:
        raise ValueError("choose one confusion input")
    registered_confusions = confusions or (
        confusion or _perfect_confusion(),
        confusion or _perfect_confusion(),
        confusion or _perfect_confusion(),
    )
    children: list[_Child] = []
    histories: list[bytes] = []
    safetensors_values: list[bytes] = []
    tensor_digests: list[str] = []
    for ordinal, seed in enumerate(_SEEDS):
        registered_confusion = registered_confusions[ordinal]
        safetensors, model_digest = _safetensors(ordinal)
        history = _history(
            seed,
            model_digest,
            winner_cross_entropies[ordinal],
            registered_confusion,
            winner_epoch=winner_epochs[ordinal],
        )
        histories.append(history)
        safetensors_values.append(safetensors)
        tensor_digests.append(model_digest)
        children.append(
            _child(
                10_001 + ordinal,
                _snapshot(
                    "training_seed",
                    ordinal,
                    seed,
                    history,
                    safetensors,
                    model_digest,
                    winner_cross_entropies[ordinal],
                    registered_confusion,
                    winner_epoch=winner_epochs[ordinal],
                ),
            )
        )
    selected_confusion = registered_confusions[selected]
    rerun = _child(
        10_004,
        _snapshot(
            "selected_seed_rerun",
            3,
            _SEEDS[selected],
            histories[selected],
            safetensors_values[selected],
            tensor_digests[selected],
            winner_cross_entropies[selected],
            selected_confusion,
            winner_epoch=winner_epochs[selected],
        ),
    )
    return (tuple(children), selected, rerun)


_ACTIVE_REGISTRATION: authority_engine.VerifiedRunRegistration | None = None


@pytest.fixture(autouse=True)
def _profile_004_registration() -> Iterator[None]:
    global _ACTIVE_REGISTRATION

    previous = (
        authority_engine._ISSUED,
        authority_engine._ISSUED_GUARDS,
        authority_engine._FAILED,
        authority_engine._ISSUANCE_COMPLETE,
    )
    authority_engine._ISSUED = weakref.WeakKeyDictionary()
    authority_engine._ISSUED_GUARDS = weakref.WeakKeyDictionary()
    authority_engine._FAILED = weakref.WeakSet()
    authority_engine._ISSUANCE_COMPLETE = False
    snapshot = authority_engine._RepositorySnapshot(
        repository_root=ROOT,
        head_commit="a" * 40,
        implementation_commit="b" * 40,
        registration_sha256="c" * 64,
        source_bundle_sha256="d" * 64,
        source_paths=("src/falsewake/experiment_002_run_authority.py",),
        profile=authority_engine._EXPERIMENT_004_PROFILE,
    )
    _ACTIVE_REGISTRATION = authority_engine._issue_controlled_snapshot_for_tests(
        snapshot
    )
    try:
        yield
    finally:
        _ACTIVE_REGISTRATION = None
        (
            authority_engine._ISSUED,
            authority_engine._ISSUED_GUARDS,
            authority_engine._FAILED,
            authority_engine._ISSUANCE_COMPLETE,
        ) = previous


def _build(selector: object) -> final_evidence.FinalCompletedEvidence:
    assert _ACTIVE_REGISTRATION is not None
    return final_evidence.build_completed_final_evidence(
        _ACTIVE_REGISTRATION,
        selector,
    )


def _replace_tuple(
    value: tuple[object, ...], index: int, replacement: object
) -> tuple[object, ...]:
    fields = list(value)
    fields[index] = replacement
    return tuple(fields)


def _replace_rerun(selector: _Selector, rerun: _Child) -> _Selector:
    return (selector[0], selector[1], rerun)


def _walk(value: object) -> tuple[object, ...]:
    values: list[object] = [value]
    if type(value) is dict:
        for key, item in cast(dict[object, object], value).items():
            values.extend(_walk(key))
            values.extend(_walk(item))
    elif type(value) in (list, tuple):
        for item in cast(list[object] | tuple[object, ...], value):
            values.extend(_walk(item))
    return tuple(values)


class _ForcePass:
    def __get__(self, instance: object, owner: type[object] | None = None) -> str:
        del instance, owner
        return "pass"


class _MaskedPid:
    def __init__(self, wrapped: object) -> None:
        self._wrapped = wrapped

    def __get__(self, instance: object, owner: type[object] | None = None) -> object:
        if instance is None:
            return self
        evidence = self._wrapped.__get__(instance, owner)  # type: ignore[attr-defined]
        return replace(evidence, pid=os.getpid())


class _PropertyLedger:
    def __init__(self, ledger: tuple[object, ...]) -> None:
        self._ledger = ledger

    @property
    def authority_frame(self) -> object:
        return self._ledger[0]

    @property
    def report_bytes(self) -> object:
        return self._ledger[1]

    @property
    def publication_snapshot(self) -> object:
        return self._ledger[2]


def _closure_value(function: FunctionType, name: str) -> object:
    closure = function.__closure__
    assert closure is not None
    return closure[function.__code__.co_freevars.index(name)].cell_contents


def _issued_ledgers() -> Any:
    report_implementation = cast(
        FunctionType,
        _closure_value(
            cast(FunctionType, final_evidence.canonical_completed_report_bytes),
            "implementation",
        ),
    )
    guarded_verified_ledger = cast(
        FunctionType, _closure_value(report_implementation, "verified_route")
    )
    verified_ledger = cast(
        FunctionType,
        _closure_value(guarded_verified_ledger, "verified_ledger"),
    )
    return _closure_value(verified_ledger, "issued")


def test_predecessor_is_exact_terminal_p3_failure_without_fabricated_outcome() -> None:
    predecessor = final_evidence._predecessor_document()

    assert predecessor == _PREDECESSOR
    outcome = cast(dict[str, object], predecessor["outcome"])
    assert outcome == {
        "automatic_terminal_report_published": False,
        "checkpoint_reusable": False,
        "code": "parent_route_signature_mismatch",
        "phase": "registered_admission",
        "status": "execution_failure",
    }
    assert not {"artifacts", "commit", "path", "sha256"} & outcome.keys()
    topology = cast(dict[str, object], predecessor["topology"])
    assert topology["outcome_report_path"] == "reports/experiment-003-training.json"
    assert topology["outcome_report_present"] is False
    assert not (ROOT / "reports/experiment-003-training.json").exists()


def test_p2_scientific_domains_and_child_artifact_names_are_preserved() -> None:
    assert final_evidence._HISTORY_DOMAIN == _HISTORY_DOMAIN
    assert final_evidence._ENVELOPE_DOMAIN == _ENVELOPE_DOMAIN
    assert final_evidence._MODEL_TENSOR_DOMAIN == _MODEL_DOMAIN
    assert final_evidence._HISTORY_FILENAME == "experiment-002-history.json"
    assert final_evidence._SAFETENSORS_FILENAME == "experiment-002-winner.safetensors"


def test_surface_is_standard_library_only_fixed_and_defaultless() -> None:
    source_path = Path("src/falsewake/experiment_004_final_evidence.py")
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.partition(".")[0])
    assert imported_roots <= {
        "__future__",
        "builtins",
        "collections",
        "dataclasses",
        "fractions",
        "falsewake",
        "hashlib",
        "json",
        "math",
        "os",
        "struct",
        "threading",
        "types",
        "typing",
        "weakref",
    }
    assert "experiment_002_coordinator" not in source
    assert "_make_completed_evidence_routes" not in vars(final_evidence)
    routes = (
        final_evidence.build_completed_final_evidence,
        final_evidence.verify_completed_final_evidence,
        final_evidence.canonical_completed_report_bytes,
        final_evidence.snapshot_completed_publication,
    )
    assert all(type(route) is FunctionType for route in routes)
    route_functions = tuple(cast(FunctionType, route) for route in routes)
    route_closures = [route.__closure__ for route in route_functions]
    required_cells = {
        "fail_route",
        "guard_authority",
        "guard_route",
        "implementation",
        "match_authority",
    }
    assert all(
        set(route.__code__.co_freevars) == required_cells for route in route_functions
    )
    assert all(closure is not None for closure in route_closures)
    assert len(
        {
            id(cell)
            for closure in route_closures
            for cell in cast(tuple[object, ...], closure)
        }
    ) == sum(len(cast(tuple[object, ...], closure)) for closure in route_closures)
    for index, route in enumerate(route_functions):
        parameters = tuple(inspect.signature(route).parameters.values())
        assert len(parameters) == (2 if index == 0 else 1)
        assert all(
            parameter.kind is inspect.Parameter.POSITIONAL_ONLY
            and parameter.default is inspect.Parameter.empty
            for parameter in parameters
        )
    assert final_evidence.TRAINING_HISTORY_PATHS == (
        "reports/experiment-004-seed-20260719-history.json",
        "reports/experiment-004-seed-20260720-history.json",
        "reports/experiment-004-seed-20260721-history.json",
    )
    assert (
        final_evidence.SELECTED_RERUN_HISTORY_PATH
        == "reports/experiment-004-selected-rerun-history.json"
    )
    assert (
        final_evidence.SELECTED_SAFETENSORS_PATH
        == "models/experiment-004-selected.safetensors"
    )
    assert final_evidence.FINAL_REPORT_PATH == "reports/experiment-004-training.json"
    assert final_evidence.FINAL_REPORT_BYTES_MAXIMUM == 256 << 10
    assert final_evidence.COMPLETED_PUBLICATION_BYTES_MAXIMUM == 8 << 20


def test_builder_rejects_cross_profile_and_mismatched_live_registration() -> None:
    def issue(
        profile: authority_engine._AuthorityProfile,
        *,
        head_commit: str,
    ) -> authority_engine.VerifiedRunRegistration:
        authority_engine._ISSUED = weakref.WeakKeyDictionary()
        authority_engine._ISSUED_GUARDS = weakref.WeakKeyDictionary()
        authority_engine._FAILED = weakref.WeakSet()
        authority_engine._ISSUANCE_COMPLETE = False
        return authority_engine._issue_controlled_snapshot_for_tests(
            authority_engine._RepositorySnapshot(
                repository_root=ROOT,
                head_commit=head_commit,
                implementation_commit="b" * 40,
                registration_sha256="c" * 64,
                source_bundle_sha256="d" * 64,
                source_paths=("src/falsewake/experiment_002_run_authority.py",),
                profile=profile,
            )
        )

    experiment_002_registration = issue(
        authority_engine._EXPERIMENT_002_PROFILE,
        head_commit="a" * 40,
    )
    with pytest.raises(
        authority_engine.Experiment002RunAuthorityError,
        match="different authority profile",
    ):
        final_evidence.build_completed_final_evidence(
            experiment_002_registration,
            _selector(),
        )

    mismatched_registration = issue(
        authority_engine._EXPERIMENT_004_PROFILE,
        head_commit="e" * 40,
    )
    with pytest.raises(
        final_evidence.Experiment004FinalEvidenceError,
        match="differs from the live Experiment 004 capability",
    ):
        final_evidence.build_completed_final_evidence(
            mismatched_registration,
            _selector(),
        )


def test_pass_issues_exact_evidence_artifacts_resources_and_canonical_report() -> None:
    evidence = _build(_selector())

    assert type(evidence) is final_evidence.FinalCompletedEvidence
    assert final_evidence.verify_completed_final_evidence(evidence) is None
    assert evidence.schema_version == 1
    assert evidence.experiment == "004"
    assert evidence.status == "pass"
    assert evidence.selection.selected_training_ordinal == 1
    assert evidence.selection.selected_seed == _SEEDS[1]
    assert evidence.selection.ranked_training_ordinals == (1, 0, 2)
    assert tuple(child.ordinal for child in evidence.children) == (0, 1, 2, 3)
    assert tuple(child.pid for child in evidence.children) == (
        10_001,
        10_002,
        10_003,
        10_004,
    )
    assert all(child.cpu_ids == (2, 3) for child in evidence.children)
    assert evidence.rerun.history_byte_identical is True
    assert evidence.rerun.safetensors_byte_identical is True
    assert evidence.rerun.all_validation_prediction_digests_identical is True
    assert evidence.rerun.all_model_tensor_digests_identical is True
    assert evidence.gates.all_passed is True
    assert evidence.gates.target_accuracy_correct == 3_703
    assert evidence.gates.target_accuracy_minimum_correct == 3_148
    assert tuple(item.minimum_correct for item in evidence.gates.target_recalls) == (
        278,
        285,
        245,
        264,
        247,
        255,
        255,
        262,
        245,
        261,
    )
    assert evidence.gates.unknown_maximum_target_predictions == 1_255
    assert evidence.gates.silence_maximum_target_predictions == 30
    assert tuple(artifact.path for artifact in evidence.artifacts) == (
        *final_evidence.TRAINING_HISTORY_PATHS,
        final_evidence.SELECTED_RERUN_HISTORY_PATH,
        final_evidence.SELECTED_SAFETENSORS_PATH,
    )
    assert all(artifact.mode == "0444" for artifact in evidence.artifacts)
    assert evidence.artifacts[-1].source_child_ordinal == 3
    assert evidence.artifacts[-1].model_tensor_sha256 == (
        evidence.selection.selected_model_tensor_sha256
    )

    report = final_evidence.canonical_completed_report_bytes(evidence)
    parsed = json.loads(report)
    assert type(parsed) is dict
    assert report == _canonical(parsed)
    assert report.endswith(b"\n")
    assert b'"status":"pass"' in report
    assert hashlib.sha256(report).hexdigest() == _REPORT_SHA256["pass"]
    assert not any(type(value) is float for value in _walk(parsed))
    report_document = cast(dict[str, Any], parsed)
    assert set(report_document) == {
        "artifacts",
        "checkpoint_reusable",
        "children",
        "experiment",
        "failure",
        "gates",
        "predecessor",
        "publication",
        "registration",
        "rerun",
        "schema_version",
        "scientific_protocol",
        "selection",
        "status",
    }
    assert report_document["experiment"] == "004"
    assert report_document["scientific_protocol"] == "002"
    assert report_document["predecessor"] == _PREDECESSOR
    assert report_document["checkpoint_reusable"] is True
    assert report_document["failure"] is None
    report_artifacts = cast(list[dict[str, object]], report_document["artifacts"])
    assert len(report_artifacts) == 5
    assert report_artifacts[-1]["path"] == final_evidence.SELECTED_SAFETENSORS_PATH
    assert "contents" not in report_artifacts[-1]


def test_gate_failure_is_completed_evidence_but_omits_selected_model() -> None:
    evidence = _build(_selector(_unknown_gate_failure_confusion()))

    assert evidence.status == "gate_failure"
    assert evidence.rerun.history_byte_identical is True
    assert evidence.gates.unknown_target_predictions == 1_256
    assert evidence.gates.unknown_target_rate_passed is False
    assert evidence.gates.target_accuracy_passed is True
    assert evidence.gates.macro_f1_passed is True
    assert evidence.gates.all_passed is False
    assert len(evidence.artifacts) == 4
    assert all(
        artifact.kind != "selected_safetensors" for artifact in evidence.artifacts
    )
    report_bytes = final_evidence.canonical_completed_report_bytes(evidence)
    assert hashlib.sha256(report_bytes).hexdigest() == _REPORT_SHA256["gate_failure"]
    report = json.loads(report_bytes)
    report_document = cast(dict[str, object], report)
    assert report_document["status"] == "gate_failure"
    assert report_document["failure"] is None
    assert report_document["checkpoint_reusable"] is False
    assert report_document["scientific_protocol"] == "002"
    assert len(cast(list[object], report_document["artifacts"])) == 4


@pytest.mark.parametrize(
    ("confusion", "expected_status", "expected_payload_count"),
    [
        (_perfect_confusion(), "pass", 5),
        (_unknown_gate_failure_confusion(), "gate_failure", 4),
    ],
)
def test_publication_snapshot_is_ordered_primitive_and_report_last(
    confusion: list[list[int]],
    expected_status: str,
    expected_payload_count: int,
) -> None:
    evidence = _build(_selector(confusion))

    snapshot = final_evidence.snapshot_completed_publication(evidence)
    status, registration, plans, report_path, report_mode, report_bytes = snapshot

    assert type(snapshot) is tuple
    assert status == expected_status
    assert registration == ("a" * 40, "b" * 40, "c" * 64, "d" * 64)
    assert type(plans) is tuple
    assert len(plans) == expected_payload_count
    assert tuple(plan[0] for plan in plans) == (
        *final_evidence.TRAINING_HISTORY_PATHS,
        final_evidence.SELECTED_RERUN_HISTORY_PATH,
        *(
            (final_evidence.SELECTED_SAFETENSORS_PATH,)
            if expected_status == "pass"
            else ()
        ),
    )
    assert all(type(plan) is tuple and type(plan[3]) is bytes for plan in plans)
    assert all(plan[4] == len(plan[3]) for plan in plans)
    assert all(hashlib.sha256(plan[3]).hexdigest() == plan[5] for plan in plans)
    assert report_path == final_evidence.FINAL_REPORT_PATH
    assert report_mode == "0444"
    assert report_path not in {plan[0] for plan in plans}
    assert report_bytes is final_evidence.canonical_completed_report_bytes(evidence)
    assert snapshot[-1] is report_bytes
    assert len(report_bytes) <= final_evidence.FINAL_REPORT_BYTES_MAXIMUM
    assert sum(plan[4] for plan in plans) + len(report_bytes) <= (
        final_evidence.COMPLETED_PUBLICATION_BYTES_MAXIMUM
    )
    parsed = cast(dict[str, object], json.loads(report_bytes))
    assert parsed["status"] == status
    assert parsed["failure"] is None
    assert (
        cast(dict[str, str], parsed["registration"])["head_commit"] == registration[0]
    )


def test_exact_recall_gate_is_recomputed_from_rerun_winner_matrix() -> None:
    evidence = _build(_selector(_recall_gate_failure_confusion()))

    first = evidence.gates.target_recalls[0]
    assert (first.correct, first.support, first.minimum_correct, first.passed) == (
        277,
        397,
        278,
        False,
    )
    assert evidence.gates.target_accuracy_passed is True
    assert evidence.status == "gate_failure"


def test_rank_uses_macro_f1_before_cross_entropy() -> None:
    evidence = _build(
        _selector(
            confusions=(
                _recall_gate_failure_confusion(),
                _unknown_gate_failure_confusion(),
                _perfect_confusion(),
            ),
            winner_cross_entropies=(0.1, 0.1, 0.9),
            selected=2,
        )
    )

    assert evidence.selection.ranked_training_ordinals == (2, 0, 1)


@pytest.mark.parametrize(
    ("cross_entropies", "winner_epochs", "selected", "expected_ranking"),
    [
        ((0.3, 0.1, 0.2), (7, 7, 7), 1, (1, 2, 0)),
        ((0.2, 0.2, 0.2), (4, 2, 3), 1, (1, 2, 0)),
        ((0.2, 0.2, 0.2), (7, 7, 7), 0, (0, 1, 2)),
    ],
)
def test_rank_tie_breaks_cross_entropy_epoch_then_seed(
    cross_entropies: tuple[float, float, float],
    winner_epochs: tuple[int, int, int],
    selected: int,
    expected_ranking: tuple[int, int, int],
) -> None:
    evidence = _build(
        _selector(
            winner_cross_entropies=cross_entropies,
            winner_epochs=winner_epochs,
            selected=selected,
        )
    )

    assert evidence.selection.ranked_training_ordinals == expected_ranking


def test_all_exact_gate_formula_boundaries() -> None:
    assert final_evidence._target_accuracy_gate(3_148, 3_703) is True
    assert final_evidence._target_accuracy_gate(3_147, 3_703) is False
    for support in _SUPPORT[:10]:
        minimum = (7 * support + 9) // 10
        assert final_evidence._target_recall_gate(minimum, support) is True
        assert final_evidence._target_recall_gate(minimum - 1, support) is False
    assert final_evidence._macro_f1_gate(Fraction(4, 5)) is True
    assert final_evidence._macro_f1_gate(Fraction(799, 1_000)) is False
    assert final_evidence._unknown_target_rate_gate(1_255, 6_278) is True
    assert final_evidence._unknown_target_rate_gate(1_256, 6_278) is False
    assert final_evidence._silence_target_rate_gate(30, 602) is True
    assert final_evidence._silence_target_rate_gate(31, 602) is False


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: list(value),
        lambda value: (value[0], True, value[2]),
        lambda value: (cast(tuple[object, ...], value[0])[:2], value[1], value[2]),
        lambda value: (value[0], 0, value[2]),
        lambda value: (
            value[0],
            value[1],
            cast(tuple[object, ...], value[2])[:5],
        ),
    ],
)
def test_selector_result_shape_selection_and_process_identity_fail_closed(
    mutator: Any,
) -> None:
    with pytest.raises((TypeError, final_evidence.Experiment004FinalEvidenceError)):
        _build(mutator(_selector()))


def test_training_role_ordinal_order_and_common_cpu_binding_are_exact() -> None:
    selector = _selector()
    training = list(cast(tuple[object, ...], selector[0]))
    first = cast(_Child, training[0])
    first_snapshot = cast(_Snapshot, first[5])
    training[0] = _replace_tuple(
        first,
        5,
        _replace_tuple(first_snapshot, 0, "selected_seed_rerun"),
    )
    with pytest.raises(final_evidence.Experiment004FinalEvidenceError):
        _build((tuple(training), selector[1], selector[2]))

    different_cpu_rerun = _replace_tuple(cast(_Child, selector[2]), 1, (4, 5))
    with pytest.raises(
        final_evidence.Experiment004FinalEvidenceError,
        match="same captured CPU",
    ):
        _build(_replace_rerun(selector, different_cpu_rerun))

    parent_pid_child = _replace_tuple(
        cast(_Child, cast(tuple[object, ...], selector[0])[0]),
        0,
        os.getpid(),
    )
    parent_pid_training = list(cast(tuple[object, ...], selector[0]))
    parent_pid_training[0] = parent_pid_child
    with pytest.raises(
        final_evidence.Experiment004FinalEvidenceError,
        match="equals the evidence parent",
    ):
        _build((tuple(parent_pid_training), selector[1], selector[2]))


def test_sequential_children_may_legitimately_reuse_a_reaped_pid() -> None:
    selector = _selector()
    training = list(cast(tuple[object, ...], selector[0]))
    first_pid = cast(int, cast(_Child, training[0])[0])
    training[1] = _replace_tuple(cast(_Child, training[1]), 0, first_pid)

    evidence = _build((tuple(training), selector[1], selector[2]))

    assert evidence.children[0].pid == evidence.children[1].pid == first_pid
    assert final_evidence.verify_completed_final_evidence(evidence) is None


@pytest.mark.parametrize(
    "bad_history",
    [
        b'{"schema_version":1,"schema_version":1}\n',
        b'{"schema_version":1.0}\n',
        b'{"schema_version":1}\n\n',
        b"{}",
    ],
)
def test_each_history_is_independently_strict_canonical_json(
    bad_history: bytes,
) -> None:
    selector = _selector()
    training = list(cast(tuple[object, ...], selector[0]))
    first = cast(_Child, training[0])
    snapshot = cast(_Snapshot, first[5])
    training[0] = _replace_tuple(first, 5, _replace_tuple(snapshot, 7, bad_history))
    with pytest.raises(final_evidence.Experiment004FinalEvidenceError):
        _build((tuple(training), selector[1], selector[2]))


def test_malformed_envelope_fails_even_with_coherent_outer_count_and_digest() -> None:
    selector = _selector()
    rerun = cast(_Child, selector[2])
    snapshot = cast(_Snapshot, rerun[5])
    malformed = b'{"schema_version":1,"schema_version":1}\n'
    changed = _replace_tuple(snapshot, 13, malformed)
    changed = _replace_tuple(changed, 14, len(malformed))
    changed = _replace_tuple(
        changed,
        15,
        hashlib.sha256(_ENVELOPE_DOMAIN + malformed).hexdigest(),
    )

    with pytest.raises(final_evidence.Experiment004FinalEvidenceError):
        _build(_replace_rerun(selector, _replace_tuple(rerun, 5, changed)))


def test_malformed_safetensors_fails_even_with_coherent_outer_hash() -> None:
    selector = _selector()
    rerun = cast(_Child, selector[2])
    snapshot = cast(_Snapshot, rerun[5])
    malformed = b"\0" * len(cast(bytes, snapshot[10]))
    changed = _replace_tuple(snapshot, 10, malformed)
    changed = _replace_tuple(changed, 11, len(malformed))
    changed = _replace_tuple(changed, 12, hashlib.sha256(malformed).hexdigest())

    with pytest.raises(final_evidence.Experiment004FinalEvidenceError):
        _build(_replace_rerun(selector, _replace_tuple(rerun, 5, changed)))


@pytest.mark.parametrize(
    ("field_index", "replacement"),
    [
        (8, 1),
        (9, "0" * 64),
        (11, 1),
        (12, "0" * 64),
        (14, 1),
        (15, "0" * 64),
        (16, 8),
        (17, 0),
        (18, 2),
        (19, (0.3).hex()),
        (20, "0" * 64),
    ],
)
def test_snapshot_counts_digests_and_winner_fields_are_not_trusted(
    field_index: int,
    replacement: object,
) -> None:
    selector = _selector()
    rerun = cast(_Child, selector[2])
    snapshot = cast(_Snapshot, rerun[5])
    bad_rerun = _replace_tuple(
        rerun,
        5,
        _replace_tuple(snapshot, field_index, replacement),
    )
    with pytest.raises(final_evidence.Experiment004FinalEvidenceError):
        _build(_replace_rerun(selector, bad_rerun))


@pytest.mark.parametrize(
    ("validation_salt", "prediction_salt", "model_salt"),
    [
        ("changed", "registered", "registered"),
        ("registered", "changed", "registered"),
        ("registered", "registered", "changed"),
    ],
)
def test_rerun_rejects_coherent_history_component_mismatches(
    validation_salt: str,
    prediction_salt: str,
    model_salt: str,
) -> None:
    selector = _selector()
    selected_child = cast(_Child, cast(tuple[object, ...], selector[0])[1])
    selected_snapshot = cast(_Snapshot, selected_child[5])
    model_digest = cast(str, selected_snapshot[20])
    safetensors = cast(bytes, selected_snapshot[10])
    confusion = _perfect_confusion()
    changed_history = _history(
        _SEEDS[1],
        model_digest,
        0.2,
        confusion,
        validation_salt=validation_salt,
        prediction_salt=prediction_salt,
        nonwinner_model_salt=model_salt,
    )
    changed_rerun = _child(
        10_004,
        _snapshot(
            "selected_seed_rerun",
            3,
            _SEEDS[1],
            changed_history,
            safetensors,
            model_digest,
            0.2,
            confusion,
        ),
    )
    with pytest.raises(
        final_evidence.Experiment004FinalEvidenceError,
        match="rerun differs",
    ):
        _build(_replace_rerun(selector, changed_rerun))


def test_rerun_rejects_coherent_selected_checkpoint_mismatch() -> None:
    selector = _selector()
    safetensors, model_digest = _safetensors(99)
    confusion = _perfect_confusion()
    changed_history = _history(_SEEDS[1], model_digest, 0.2, confusion)
    changed_rerun = _child(
        10_004,
        _snapshot(
            "selected_seed_rerun",
            3,
            _SEEDS[1],
            changed_history,
            safetensors,
            model_digest,
            0.2,
            confusion,
        ),
    )
    with pytest.raises(
        final_evidence.Experiment004FinalEvidenceError,
        match="rerun differs",
    ):
        _build(_replace_rerun(selector, changed_rerun))


def test_artifact_and_final_values_are_immutable_and_issuer_verified() -> None:
    evidence = _build(_selector())
    with pytest.raises(FrozenInstanceError):
        evidence.artifacts[0].__setattr__("path", "reports/forged.json")
    with pytest.raises(TypeError, match="issued only"):
        final_evidence.FinalCompletedEvidence()

    object.__setattr__(evidence, "status", "gate_failure")
    with pytest.raises(
        final_evidence.Experiment004FinalEvidenceError,
        match="changed after issuance",
    ):
        final_evidence.verify_completed_final_evidence(evidence)
    with pytest.raises(final_evidence.Experiment004FinalEvidenceError):
        final_evidence.canonical_completed_report_bytes(evidence)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda evidence: object.__setattr__(
            evidence.registration, "head_commit", "e" * 40
        ),
        lambda evidence: object.__setattr__(
            evidence.children[0], "seed", evidence.children[0].seed + 1
        ),
        lambda evidence: object.__setattr__(
            evidence.selection, "selected_seed", evidence.selection.selected_seed + 1
        ),
        lambda evidence: object.__setattr__(
            evidence.rerun, "selected_seed", evidence.rerun.selected_seed + 1
        ),
        lambda evidence: object.__setattr__(
            evidence.gates.target_recalls[0],
            "correct",
            evidence.gates.target_recalls[0].correct - 1,
        ),
        lambda evidence: object.__setattr__(
            evidence.gates, "all_passed", not evidence.gates.all_passed
        ),
        lambda evidence: object.__setattr__(
            evidence.artifacts[0], "path", "reports/forged.json"
        ),
    ],
)
def test_every_nested_evidence_dataclass_is_primitive_ledger_verified(
    mutator: Any,
) -> None:
    evidence = _build(_selector())
    mutator(evidence)

    with pytest.raises(
        final_evidence.Experiment004FinalEvidenceError,
        match="changed after issuance",
    ):
        final_evidence.verify_completed_final_evidence(evidence)


def test_nested_class_property_masking_fails_issuer_integrity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _build(_selector())
    monkeypatch.setattr(
        final_evidence.ChildEvidence,
        "seed",
        property(lambda _self: _SEEDS[0]),
    )

    with pytest.raises(
        final_evidence.Experiment004FinalEvidenceError,
        match="issuer integrity",
    ):
        final_evidence.verify_completed_final_evidence(evidence)


@pytest.mark.parametrize(
    ("class_name", "field_name", "descriptor"),
    [
        (
            "_ParsedEpoch",
            "macro_f1",
            property(lambda _instance: Fraction(1, 1)),
        ),
        (
            "_ParsedHistory",
            "winner",
            property(lambda _instance: object()),
        ),
        (
            "_ParsedSnapshot",
            "winner_epoch",
            property(lambda _instance: 0),
        ),
        (
            "_ParsedChild",
            "evidence",
            _MaskedPid(vars(final_evidence._ParsedChild)["evidence"]),
        ),
        ("_CompletedFields", "status", _ForcePass()),
    ],
    ids=(
        "parsed-epoch-force-pass",
        "parsed-history-property",
        "parsed-snapshot-property",
        "parsed-child-masked-pid",
        "completed-fields-force-pass",
    ),
)
def test_every_internal_trust_dataclass_rejects_persistent_descriptor_masking(
    monkeypatch: pytest.MonkeyPatch,
    class_name: str,
    field_name: str,
    descriptor: object,
) -> None:
    internal_class = cast(type[object], getattr(final_evidence, class_name))
    monkeypatch.setattr(internal_class, field_name, descriptor)

    with pytest.raises(
        final_evidence.Experiment004FinalEvidenceError,
        match="issuer integrity",
    ):
        _build(_selector())

    monkeypatch.undo()
    final_evidence.verify_completed_final_evidence(_build(_selector()))


@pytest.mark.parametrize(
    "class_name",
    (
        "_ParsedEpoch",
        "_ParsedHistory",
        "_ParsedSnapshot",
        "_ParsedChild",
        "_CompletedFields",
    ),
)
def test_every_internal_trust_dataclass_rejects_global_class_rebinding(
    monkeypatch: pytest.MonkeyPatch,
    class_name: str,
) -> None:
    monkeypatch.setattr(final_evidence, class_name, type(class_name, (), {}))

    with pytest.raises(
        final_evidence.Experiment004FinalEvidenceError,
        match="issuer integrity",
    ):
        _build(_selector())


@pytest.mark.parametrize(
    "class_name",
    (
        "_ParsedEpoch",
        "_ParsedHistory",
        "_ParsedSnapshot",
        "_ParsedChild",
        "_CompletedFields",
    ),
)
def test_every_internal_trust_dataclass_rejects_namespace_extension(
    monkeypatch: pytest.MonkeyPatch,
    class_name: str,
) -> None:
    internal_class = cast(type[object], getattr(final_evidence, class_name))
    monkeypatch.setattr(
        internal_class,
        "_persistent_force_pass",
        _ForcePass(),
        raising=False,
    )

    with pytest.raises(
        final_evidence.Experiment004FinalEvidenceError,
        match="issuer integrity",
    ):
        _build(_selector())


def test_issued_state_surface_is_replaced_by_an_exact_primitive_tuple_ledger() -> None:
    assert "_IssuedState" not in vars(final_evidence)
    evidence = _build(_selector())
    issued = _issued_ledgers()
    ledger = issued[evidence]

    assert type(ledger) is tuple
    assert len(ledger) == 3
    assert {type(item) for item in _walk(ledger)} <= {
        tuple,
        bytes,
        bool,
        int,
        str,
        type(None),
    }
    assert final_evidence.canonical_completed_report_bytes(evidence) is ledger[1]
    assert final_evidence.snapshot_completed_publication(evidence) is ledger[2]


def test_property_issued_state_is_rejected_by_primitive_ledger() -> None:
    evidence = _build(_selector())
    issued = _issued_ledgers()
    ledger = cast(tuple[object, ...], issued[evidence])
    issued[evidence] = _PropertyLedger(ledger)

    with pytest.raises(
        final_evidence.Experiment004FinalEvidenceError,
        match="lost exact tuple authority",
    ):
        final_evidence.canonical_completed_report_bytes(evidence)
    assert evidence not in issued


@pytest.mark.parametrize(
    "forge",
    (
        lambda ledger: (ledger[0], bytearray(cast(bytes, ledger[1])), ledger[2]),
        lambda ledger: (list(cast(tuple[object, ...], ledger[0])), *ledger[1:]),
        lambda ledger: (*ledger[:2], list(cast(tuple[object, ...], ledger[2]))),
        lambda ledger: (*ledger, None),
    ),
    ids=(
        "bytes-substitution",
        "authority-frame-substitution",
        "snapshot-substitution",
        "length-extension",
    ),
)
def test_each_issued_ledger_component_is_locally_exact_and_revoked(
    forge: Any,
) -> None:
    evidence = _build(_selector())
    issued = _issued_ledgers()
    ledger = cast(tuple[object, ...], issued[evidence])
    issued[evidence] = forge(ledger)

    with pytest.raises(final_evidence.Experiment004FinalEvidenceError):
        final_evidence.snapshot_completed_publication(evidence)
    assert evidence not in issued


def test_module_and_hash_route_rebinding_fail_issuer_integrity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _build(_selector())
    monkeypatch.setattr(final_evidence, "hashlib", object())
    with pytest.raises(
        final_evidence.Experiment004FinalEvidenceError,
        match="issuer integrity",
    ):
        final_evidence.canonical_completed_report_bytes(evidence)

    monkeypatch.undo()
    monkeypatch.setattr(hashlib, "sha256", lambda _payload=b"": object())
    with pytest.raises(
        final_evidence.Experiment004FinalEvidenceError,
        match="issuer integrity",
    ):
        final_evidence.snapshot_completed_publication(evidence)


def test_json_encoder_replacement_cannot_forge_the_final_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selector = _selector()
    original_encoder = json.JSONEncoder
    forged_documents: list[object] = []

    class ForgingJSONEncoder:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._delegate = original_encoder(*args, **kwargs)

        def encode(self, document: object) -> str:
            if type(document) is dict and {
                "artifacts",
                "children",
                "publication",
                "status",
            } <= set(cast(dict[str, object], document)):
                forged_documents.append(document)
                return '{"forged":true}'
            return self._delegate.encode(document)

    monkeypatch.setattr(json, "JSONEncoder", ForgingJSONEncoder)
    with pytest.raises(
        final_evidence.Experiment004FinalEvidenceError,
        match="issuer integrity",
    ):
        _build(selector)
    assert forged_documents == []


def test_json_decoder_replacement_cannot_take_over_canonical_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selector = _selector()
    original_decoder = json.JSONDecoder
    forged_documents: list[dict[str, object]] = []

    class ForgingJSONDecoder:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._delegate = original_decoder(*args, **kwargs)

        def decode(self, text: str) -> Any:
            document = self._delegate.decode(text)
            if type(document) is dict and "epochs" in document and "seed" in document:
                reordered = dict(
                    reversed(tuple(cast(dict[str, object], document).items()))
                )
                forged_documents.append(reordered)
                return reordered
            return document

    monkeypatch.setattr(json, "JSONDecoder", ForgingJSONDecoder)
    with pytest.raises(
        final_evidence.Experiment004FinalEvidenceError,
        match="issuer integrity",
    ):
        _build(selector)
    assert forged_documents == []


@pytest.mark.parametrize(
    ("class_name", "method_name"),
    (("JSONEncoder", "encode"), ("JSONDecoder", "decode")),
)
def test_json_method_code_mutation_fails_issuer_integrity(
    class_name: str,
    method_name: str,
) -> None:
    selector = _selector()
    json_class = cast(type[Any], getattr(json, class_name))
    method = cast(FunctionType, getattr(json_class, method_name))

    def forged_method(_instance: object, _value: object) -> object:
        return {}

    original_code = method.__code__
    method.__code__ = forged_method.__code__
    try:
        with pytest.raises(
            final_evidence.Experiment004FinalEvidenceError,
            match="issuer integrity",
        ):
            _build(selector)
    finally:
        method.__code__ = original_code


@pytest.mark.parametrize("class_name", ("JSONEncoder", "JSONDecoder"))
def test_json_class_namespace_mutation_fails_issuer_integrity(
    monkeypatch: pytest.MonkeyPatch,
    class_name: str,
) -> None:
    selector = _selector()
    json_class = cast(type[Any], getattr(json, class_name))
    monkeypatch.setattr(
        json_class,
        "_persistent_evidence_forge",
        object(),
        raising=False,
    )

    with pytest.raises(
        final_evidence.Experiment004FinalEvidenceError,
        match="issuer integrity",
    ):
        _build(selector)


@pytest.mark.parametrize(
    ("class_name", "method_name"),
    (("JSONEncoder", "encode"), ("JSONDecoder", "decode")),
)
def test_json_method_namespace_mutation_fails_issuer_integrity(
    monkeypatch: pytest.MonkeyPatch,
    class_name: str,
    method_name: str,
) -> None:
    selector = _selector()
    json_class = cast(type[Any], getattr(json, class_name))
    method = cast(FunctionType, getattr(json_class, method_name))
    monkeypatch.setattr(
        method,
        "_persistent_evidence_forge",
        object(),
        raising=False,
    )

    with pytest.raises(
        final_evidence.Experiment004FinalEvidenceError,
        match="issuer integrity",
    ):
        _build(selector)


@pytest.mark.parametrize(
    ("class_name", "method_name", "global_name"),
    (
        ("JSONEncoder", "encode", "encode_basestring_ascii"),
        ("JSONDecoder", "__init__", "JSONObject"),
    ),
)
def test_json_method_global_namespace_mutation_fails_issuer_integrity(
    monkeypatch: pytest.MonkeyPatch,
    class_name: str,
    method_name: str,
    global_name: str,
) -> None:
    selector = _selector()
    json_class = cast(type[Any], getattr(json, class_name))
    method = cast(FunctionType, getattr(json_class, method_name))
    monkeypatch.setitem(method.__globals__, global_name, object())

    with pytest.raises(
        final_evidence.Experiment004FinalEvidenceError,
        match="issuer integrity",
    ):
        _build(selector)


@pytest.mark.parametrize(
    ("class_name", "keyword_name", "replacement"),
    (("JSONEncoder", "sort_keys", True), ("JSONDecoder", "strict", False)),
)
def test_json_method_keyword_default_mutation_fails_issuer_integrity(
    monkeypatch: pytest.MonkeyPatch,
    class_name: str,
    keyword_name: str,
    replacement: object,
) -> None:
    selector = _selector()
    json_class = cast(type[Any], getattr(json, class_name))
    initializer = cast(FunctionType, json_class.__init__)
    keyword_defaults = initializer.__kwdefaults__
    assert keyword_defaults is not None
    monkeypatch.setitem(keyword_defaults, keyword_name, replacement)

    with pytest.raises(
        final_evidence.Experiment004FinalEvidenceError,
        match="issuer integrity",
    ):
        _build(selector)


def test_helper_global_and_function_code_mutation_fail_issuer_integrity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _build(_selector())
    original_builder = final_evidence._completed_report_bytes
    monkeypatch.setattr(
        final_evidence, "_completed_report_bytes", lambda _value: b"{}\n"
    )
    with pytest.raises(
        final_evidence.Experiment004FinalEvidenceError,
        match="issuer integrity",
    ):
        final_evidence.verify_completed_final_evidence(evidence)

    monkeypatch.undo()

    def forged_builder(_value: object) -> bytes:
        return b"{}\n"

    monkeypatch.setattr(original_builder, "__code__", forged_builder.__code__)
    with pytest.raises(
        final_evidence.Experiment004FinalEvidenceError,
        match="issuer integrity",
    ):
        final_evidence.verify_completed_final_evidence(evidence)


@pytest.mark.parametrize(
    ("target_name", "checker_name"),
    [
        ("build_completed_final_evidence", "verify_completed_final_evidence"),
        ("verify_completed_final_evidence", "canonical_completed_report_bytes"),
        ("canonical_completed_report_bytes", "snapshot_completed_publication"),
        ("snapshot_completed_publication", "build_completed_final_evidence"),
    ],
)
def test_each_public_route_closure_mutation_fails_from_an_independent_route(
    target_name: str,
    checker_name: str,
) -> None:
    evidence = _build(_selector())
    target = cast(FunctionType, getattr(final_evidence, target_name))
    checker = cast(FunctionType, getattr(final_evidence, checker_name))
    closure = target.__closure__
    assert closure is not None
    assert {"guard_route", "implementation"} <= set(target.__code__.co_freevars)
    index = target.__code__.co_freevars.index("implementation")
    cell = closure[index]
    original = cell.cell_contents
    cell.cell_contents = lambda _value: None
    try:
        with pytest.raises(
            final_evidence.Experiment004FinalEvidenceError,
            match="issuer integrity",
        ):
            if checker_name == "build_completed_final_evidence":
                assert _ACTIVE_REGISTRATION is not None
                checker(_ACTIVE_REGISTRATION, _selector())
            else:
                checker(evidence)
    finally:
        cell.cell_contents = original


@pytest.mark.parametrize(
    ("target_name", "cell_name"),
    [
        (route_name, cell_name)
        for route_name in (
            "build_completed_final_evidence",
            "verify_completed_final_evidence",
            "canonical_completed_report_bytes",
            "snapshot_completed_publication",
        )
        for cell_name in ("guard_route", "implementation")
    ],
)
def test_each_public_route_rejects_its_own_single_cell_mutation(
    target_name: str,
    cell_name: str,
) -> None:
    evidence = _build(_selector())
    target = cast(FunctionType, getattr(final_evidence, target_name))
    closure = target.__closure__
    assert closure is not None
    index = target.__code__.co_freevars.index(cell_name)
    cell = closure[index]
    original = cell.cell_contents
    cell.cell_contents = lambda _value=None: None
    try:
        with pytest.raises(
            final_evidence.Experiment004FinalEvidenceError,
            match="issuer integrity",
        ):
            if target_name == "build_completed_final_evidence":
                assert _ACTIVE_REGISTRATION is not None
                target(_ACTIVE_REGISTRATION, _selector())
            else:
                target(evidence)
    finally:
        cell.cell_contents = original


def test_guard_code_and_closure_authority_are_preflight_checked() -> None:
    evidence = _build(_selector())
    route = final_evidence.verify_completed_final_evidence
    route_closure = route.__closure__
    assert route_closure is not None
    guard_index = route.__code__.co_freevars.index("guard_route")
    guard = cast(FunctionType, route_closure[guard_index].cell_contents)

    original_code = guard.__code__
    guard.__code__ = original_code.replace(co_name="tampered_require_integrity")
    try:
        with pytest.raises(
            final_evidence.Experiment004FinalEvidenceError,
            match="issuer integrity",
        ):
            route(evidence)
    finally:
        guard.__code__ = original_code

    guard_closure = guard.__closure__
    assert guard_closure is not None
    authority_index = guard.__code__.co_freevars.index("route_function_authorities")
    authority_cell = guard_closure[authority_index]
    original_authorities = authority_cell.cell_contents
    authority_cell.cell_contents = ()
    try:
        with pytest.raises(
            final_evidence.Experiment004FinalEvidenceError,
            match="issuer integrity",
        ):
            route(evidence)
    finally:
        authority_cell.cell_contents = original_authorities


@pytest.mark.parametrize(
    ("target_name", "checker_name"),
    [
        ("build_completed_final_evidence", "verify_completed_final_evidence"),
        ("verify_completed_final_evidence", "canonical_completed_report_bytes"),
        ("canonical_completed_report_bytes", "snapshot_completed_publication"),
        ("snapshot_completed_publication", "verify_completed_final_evidence"),
    ],
)
def test_each_internal_implementation_closure_is_authority_checked(
    target_name: str,
    checker_name: str,
) -> None:
    evidence = _build(_selector())
    target = cast(FunctionType, getattr(final_evidence, target_name))
    wrapper_closure = target.__closure__
    assert wrapper_closure is not None
    implementation_index = target.__code__.co_freevars.index("implementation")
    implementation = cast(
        FunctionType, wrapper_closure[implementation_index].cell_contents
    )
    implementation_closure = implementation.__closure__
    assert implementation_closure is not None
    cell = implementation_closure[0]
    original = cell.cell_contents
    cell.cell_contents = object()
    try:
        with pytest.raises(
            final_evidence.Experiment004FinalEvidenceError,
            match="issuer integrity",
        ):
            cast(FunctionType, getattr(final_evidence, checker_name))(evidence)
    finally:
        cell.cell_contents = original


def test_internal_implementation_code_is_authority_checked() -> None:
    evidence = _build(_selector())
    verify_wrapper = final_evidence.verify_completed_final_evidence
    report_wrapper = final_evidence.canonical_completed_report_bytes
    assert verify_wrapper.__closure__ is not None
    assert report_wrapper.__closure__ is not None
    verify_index = verify_wrapper.__code__.co_freevars.index("implementation")
    report_index = report_wrapper.__code__.co_freevars.index("implementation")
    verify_implementation = cast(
        FunctionType, verify_wrapper.__closure__[verify_index].cell_contents
    )
    report_implementation = cast(
        FunctionType, report_wrapper.__closure__[report_index].cell_contents
    )
    original_code = verify_implementation.__code__
    verify_implementation.__code__ = report_implementation.__code__
    try:
        with pytest.raises(
            final_evidence.Experiment004FinalEvidenceError,
            match="issuer integrity",
        ):
            final_evidence.snapshot_completed_publication(evidence)
    finally:
        verify_implementation.__code__ = original_code


@pytest.mark.parametrize(
    ("child_field", "replacement"),
    [
        (2, 21_600 * 1_000_000_000 + 1),
        (3, 8_589_934_593),
        (4, 4_294_967_297),
    ],
)
def test_registered_resource_budgets_are_revalidated(
    child_field: int,
    replacement: int,
) -> None:
    selector = _selector()
    training = list(cast(tuple[object, ...], selector[0]))
    training[0] = _replace_tuple(cast(_Child, training[0]), child_field, replacement)
    with pytest.raises(final_evidence.Experiment004FinalEvidenceError):
        _build((tuple(training), selector[1], selector[2]))


def test_registered_resource_and_uint32_maxima_are_accepted_exactly() -> None:
    selector = _selector()
    training = list(cast(tuple[object, ...], selector[0]))
    first = cast(_Child, training[0])
    first = _replace_tuple(first, 0, (1 << 32) - 1)
    first = _replace_tuple(first, 1, ((1 << 32) - 2, (1 << 32) - 1))
    first = _replace_tuple(first, 2, 21_600 * 1_000_000_000)
    first = _replace_tuple(first, 3, 8_589_934_592)
    first = _replace_tuple(first, 4, 4_294_967_296)
    training[0] = first
    common_cpu = cast(tuple[int, int], first[1])
    training[1] = _replace_tuple(cast(_Child, training[1]), 1, common_cpu)
    training[2] = _replace_tuple(cast(_Child, training[2]), 1, common_cpu)
    rerun = _replace_tuple(cast(_Child, selector[2]), 1, common_cpu)

    evidence = _build((tuple(training), selector[1], rerun))

    assert evidence.children[0].pid == (1 << 32) - 1
    assert evidence.children[0].cpu_ids == common_cpu
    assert evidence.children[0].elapsed_nanoseconds == 21_600 * 1_000_000_000
    assert evidence.children[0].maximum_rss_bytes == 8_589_934_592
    assert evidence.children[0].output_and_scratch_bytes == 4_294_967_296


@pytest.mark.parametrize(
    ("field_index", "replacement"), [(0, 1 << 32), (1, (2, 1 << 32))]
)
def test_pid_and_cpu_identifiers_above_uint32_fail_closed(
    field_index: int, replacement: object
) -> None:
    selector = _selector()
    training = list(cast(tuple[object, ...], selector[0]))
    training[0] = _replace_tuple(cast(_Child, training[0]), field_index, replacement)
    with pytest.raises(
        final_evidence.Experiment004FinalEvidenceError,
        match="unsigned 32-bit",
    ):
        _build((tuple(training), selector[1], selector[2]))


def test_publication_budget_exact_boundaries_and_one_byte_overflow() -> None:
    report_maximum = final_evidence.FINAL_REPORT_BYTES_MAXIMUM
    aggregate_maximum = final_evidence.COMPLETED_PUBLICATION_BYTES_MAXIMUM
    report = b"r" * report_maximum
    remaining = aggregate_maximum - report_maximum

    final_evidence._require_publication_budgets(
        report,
        (remaining,),
        report_maximum,
        aggregate_maximum,
    )
    with pytest.raises(final_evidence.Experiment004FinalEvidenceError):
        final_evidence._require_publication_budgets(
            report + b"r",
            (),
            report_maximum,
            aggregate_maximum,
        )
    with pytest.raises(final_evidence.Experiment004FinalEvidenceError):
        final_evidence._require_publication_budgets(
            report,
            (remaining + 1,),
            report_maximum,
            aggregate_maximum,
        )
