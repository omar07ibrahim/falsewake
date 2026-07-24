from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import pickle
import stat
import subprocess
import sys
import weakref
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import pytest

import falsewake.experiment_006_final_publication as publication

ROOT = Path(__file__).resolve().parents[1]
_PROTOCOL = json.loads((ROOT / "configs/experiment-006-execution.json").read_bytes())
_PREDECESSOR = _PROTOCOL["predecessor"]
_REGISTRATION = ("a" * 40, "b" * 40, "c" * 64, "d" * 64)
_HISTORY_PATHS = (
    "reports/experiment-006-seed-20260719-history.json",
    "reports/experiment-006-seed-20260720-history.json",
    "reports/experiment-006-seed-20260721-history.json",
    "reports/experiment-006-selected-rerun-history.json",
)
_MODEL_PATH = "models/experiment-006-selected.safetensors"
_REPORT_PATH = "reports/experiment-006-training.json"
_MANAGED_PATHS = (*_HISTORY_PATHS, _MODEL_PATH, _REPORT_PATH)
_TEMP_PREFIX = ".falsewake-exp006-publication-"
_HISTORY_DOMAIN = b"falsewake-exp002-history-v1\0"
_REGISTERED_SEEDS = (20_260_719, 20_260_720, 20_260_721)
_CLASS_NAMES = (
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
    "unknown",
    "silence",
)
_CLASS_SUPPORT = (397, 406, 350, 377, 352, 363, 363, 373, 350, 372, 6_278, 602)

type _Plan = tuple[
    str,
    str,
    str,
    bytes,
    int,
    str,
    str | None,
    str | None,
    int,
]
type _Snapshot = tuple[
    str, tuple[str, str, str, str], tuple[_Plan, ...], str, str, bytes
]


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


def _macro_f1(confusion: list[list[int]]) -> Fraction:
    total = Fraction()
    for class_index in range(len(_CLASS_SUPPORT)):
        true_positive = confusion[class_index][class_index]
        false_positive = sum(
            confusion[row][class_index]
            for row in range(len(_CLASS_SUPPORT))
            if row != class_index
        )
        false_negative = _CLASS_SUPPORT[class_index] - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        total += (
            Fraction() if denominator == 0 else Fraction(2 * true_positive, denominator)
        )
    return total / len(_CLASS_SUPPORT)


def _history_bytes(seed: int, status: str, rank_bias: int) -> bytes:
    if status == "pass":
        confusion = [
            [support if column == row else 0 for column in range(12)]
            for row, support in enumerate(_CLASS_SUPPORT)
        ]
    else:
        confusion = [
            [support if column == 10 else 0 for column in range(12)]
            for support in _CLASS_SUPPORT
        ]
    macro_f1 = _macro_f1(confusion)
    validation_input = hashlib.sha256(f"validation-{seed}".encode()).hexdigest()
    epochs: list[dict[str, object]] = []
    for epoch in range(30):
        validation_cross_entropy = 0.1 * (rank_bias + 1) + epoch / 1_000
        epochs.append(
            {
                "epoch_update_trace_digest": hashlib.sha256(
                    f"updates-{seed}-{epoch}".encode()
                ).hexdigest(),
                "first_global_update": epoch * 313,
                "last_global_update_inclusive": (epoch + 1) * 313 - 1,
                "macro_f1_exact_denominator": macro_f1.denominator,
                "macro_f1_exact_numerator": macro_f1.numerator,
                "model_tensor_digest": hashlib.sha256(
                    f"model-tensors-{seed}-{epoch}".encode()
                ).hexdigest(),
                "training_cross_entropy_float64_hex": (
                    1.0 + rank_bias / 10 + epoch / 1_000
                ).hex(),
                "training_population_digest": hashlib.sha256(
                    f"population-{seed}-{epoch}".encode()
                ).hexdigest(),
                "validation_confusion_matrix": confusion,
                "validation_cross_entropy_float64_hex": (
                    validation_cross_entropy.hex()
                ),
                "validation_input_digest": validation_input,
                "validation_prediction_digest": hashlib.sha256(
                    f"predictions-{seed}-{epoch}".encode()
                ).hexdigest(),
                "zero_based_epoch": epoch,
            }
        )
    return _canonical(
        {
            "complete_update_trace_digest": hashlib.sha256(
                f"complete-{seed}".encode()
            ).hexdigest(),
            "epochs": epochs,
            "experiment": "002",
            "schema_version": 1,
            "seed": seed,
            "validation_input_digest": validation_input,
        }
    )


def _history_plan(path: str, kind: str, ordinal: int, contents: bytes) -> _Plan:
    return (
        path,
        kind,
        "0444",
        contents,
        len(contents),
        hashlib.sha256(contents).hexdigest(),
        hashlib.sha256(_HISTORY_DOMAIN + contents).hexdigest(),
        None,
        ordinal,
    )


def _report_artifact(plan: _Plan) -> dict[str, object]:
    result: dict[str, object] = {
        "byte_count": plan[4],
        "kind": plan[1],
        "mode": plan[2],
        "path": plan[0],
        "sha256": plan[5],
        "source_child_ordinal": plan[8],
    }
    if plan[6] is not None:
        result["domain_sha256"] = plan[6]
    if plan[7] is not None:
        result["model_tensor_sha256"] = plan[7]
        result["tensor_count"] = 53
        result["value_count"] = 23_724
    return result


def _completed_snapshot(status: str = "pass") -> _Snapshot:
    training_contents = tuple(
        _history_bytes(seed, status, rank_bias)
        for seed, rank_bias in zip(_REGISTERED_SEEDS, (2, 0, 1), strict=True)
    )
    parsed_training = tuple(
        publication._COMPLETED_HISTORY_PARSER(contents, seed)
        for contents, seed in zip(
            training_contents,
            _REGISTERED_SEEDS,
            strict=True,
        )
    )
    ranking = tuple(
        sorted(
            range(3),
            key=lambda index: (
                -parsed_training[index].winner.macro_f1,
                parsed_training[index].winner.validation_cross_entropy,
                parsed_training[index].winner.zero_based_epoch,
                _REGISTERED_SEEDS[index],
            ),
        )
    )
    selected_ordinal = ranking[0]
    selected_seed = _REGISTERED_SEEDS[selected_ordinal]
    rerun_contents = training_contents[selected_ordinal]
    histories = tuple(
        _history_plan(
            path,
            "selected_rerun_history" if index == 3 else "training_history",
            index,
            (training_contents[index] if index < 3 else rerun_contents),
        )
        for index, path in enumerate(_HISTORY_PATHS)
    )
    plans = histories
    model_contents = (b"falsewake-selected-model\0" * 4_338)[:99_776]
    assert len(model_contents) == 99_776
    model_sha256 = hashlib.sha256(model_contents).hexdigest()
    rerun_history = publication._COMPLETED_HISTORY_PARSER(
        rerun_contents,
        selected_seed,
    )
    model_tensor_sha256 = rerun_history.winner.model_tensor_digest
    if status == "pass":
        plans = (
            *histories,
            (
                _MODEL_PATH,
                "selected_safetensors",
                "0444",
                model_contents,
                len(model_contents),
                model_sha256,
                None,
                model_tensor_sha256,
                3,
            ),
        )
    assert status in {"pass", "gate_failure"}
    selected_safetensors_sha256 = (
        model_sha256
        if status == "pass"
        else hashlib.sha256(b"gate-failure-selected-model").hexdigest()
    )
    children: list[dict[str, object]] = []
    for index, (plan, history) in enumerate(
        zip(histories, (*parsed_training, rerun_history), strict=True)
    ):
        child_seed = _REGISTERED_SEEDS[index] if index < 3 else selected_seed
        child_safetensors_sha256 = (
            selected_safetensors_sha256
            if index in {selected_ordinal, 3}
            else hashlib.sha256(f"safetensors-{child_seed}".encode()).hexdigest()
        )
        winner = history.winner
        children.append(
            {
                "binding": {
                    "ordinal": index,
                    "role": ("training_seed" if index < 3 else "selected_seed_rerun"),
                    "seed": child_seed,
                },
                "envelope": {
                    "byte_count": 1_024 + index,
                    "domain_sha256": hashlib.sha256(
                        f"envelope-{index}".encode()
                    ).hexdigest(),
                },
                "history": {
                    "byte_count": plan[4],
                    "domain_sha256": plan[6],
                    "sha256": plan[5],
                },
                "resources": {
                    "cpu_ids": [2, 3],
                    "elapsed_nanoseconds": 1_000_000 + index,
                    "maximum_rss_bytes": 100_000_000 + index,
                    "output_and_scratch_bytes": 200_000 + index,
                    "pid": 4_000_000_000 + index,
                },
                "safetensors": {
                    "byte_count": 99_776,
                    "sha256": child_safetensors_sha256,
                },
                "winner": {
                    "macro_f1_exact_denominator": winner.macro_f1.denominator,
                    "macro_f1_exact_numerator": winner.macro_f1.numerator,
                    "model_tensor_sha256": winner.model_tensor_digest,
                    "validation_cross_entropy_float64_hex": (
                        winner.validation_cross_entropy_float64_hex
                    ),
                    "zero_based_epoch": winner.zero_based_epoch,
                },
            }
        )
    selected_winner = parsed_training[selected_ordinal].winner
    gates = publication._COMPLETED_GATES_DOCUMENT(
        publication._COMPLETED_GATES_COMPUTER(rerun_history)
    )
    report = _canonical(
        {
            "artifacts": [_report_artifact(plan) for plan in plans],
            "checkpoint_reusable": status == "pass",
            "children": children,
            "experiment": "006",
            "failure": None,
            "gates": gates,
            "predecessor": _PREDECESSOR,
            "publication": {"report_mode": "0444", "report_path": _REPORT_PATH},
            "registration": {
                "head_commit": _REGISTRATION[0],
                "implementation_commit": _REGISTRATION[1],
                "path": "configs/experiment-006-run.json",
                "registration_sha256": _REGISTRATION[2],
                "source_bundle_sha256": _REGISTRATION[3],
            },
            "rerun": {
                "all_model_tensor_digests_identical": True,
                "all_validation_prediction_digests_identical": True,
                "history_byte_identical": True,
                "rerun_ordinal": 3,
                "safetensors_byte_identical": True,
                "selected_checkpoint_digest_identical": True,
                "selected_rank_identical": True,
                "selected_seed": selected_seed,
                "selected_training_ordinal": selected_ordinal,
                "validation_input_digest_identical": True,
            },
            "schema_version": 1,
            "scientific_protocol": "002",
            "selection": {
                "ranked_training_ordinals": list(ranking),
                "selected_macro_f1_denominator": (selected_winner.macro_f1.denominator),
                "selected_macro_f1_numerator": selected_winner.macro_f1.numerator,
                "selected_model_tensor_sha256": (selected_winner.model_tensor_digest),
                "selected_seed": selected_seed,
                "selected_training_ordinal": selected_ordinal,
                "selected_validation_cross_entropy_float64_hex": (
                    selected_winner.validation_cross_entropy_float64_hex
                ),
                "selected_winner_epoch": selected_winner.zero_based_epoch,
            },
            "status": status,
        }
    )
    return status, _REGISTRATION, plans, _REPORT_PATH, "0444", report


def _failure_snapshot(
    phase: publication.FailurePhase = "registered_execution",
    code: publication.FailureCode = "seed_selection_failed",
) -> _Snapshot:
    evidence = publication.build_execution_failure_evidence(_REGISTRATION, phase, code)
    return publication.snapshot_execution_failure_publication(evidence)


def _layout(root: Path) -> publication._PublicationLayout:
    root.mkdir()
    (root / "reports").mkdir()
    (root / "models").mkdir()
    return publication._PublicationLayout(root.as_posix())


def _path(root: Path, relative: str) -> Path:
    return root / relative


def _fresh_callback(
    root: Path, calls: list[str], action: Any | None = None
) -> publication._FinalAuthorityCheck:
    def callback() -> None:
        calls.append("revalidate")
        assert all(not _path(root, path).exists() for path in _MANAGED_PATHS)
        assert not any(
            entry.name.startswith(_TEMP_PREFIX)
            for directory in (root / "reports", root / "models")
            for entry in directory.iterdir()
        )
        if action is not None:
            action()

    return callback


def _assert_file(path: Path, contents: bytes) -> None:
    observed = path.stat()
    assert stat.S_ISREG(observed.st_mode)
    assert stat.S_IMODE(observed.st_mode) == 0o444
    assert observed.st_nlink == 1
    assert path.read_bytes() == contents


def test_predecessor_is_exact_terminal_p5_preflight_rejection() -> None:
    predecessor = publication._predecessor_document()

    assert predecessor == _PREDECESSOR
    assert (
        json.dumps(
            _PREDECESSOR,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        == publication._PREDECESSOR_CANONICAL_JSON
    )
    attempt = cast(dict[str, object], predecessor["attempt"])
    assert attempt == {
        "canonical_marker": (
            "/home/ubuntu/gitcode/.t/falsewake-experiment-005-attempt"
        ),
        "canonical_marker_present": False,
        "optimizer_updates": 0,
        "registered_attempt_consumed": False,
        "registered_authority_issuer_invoked": False,
        "registered_coordinator_invoked": False,
        "registered_invocation_count": 0,
        "registered_runner_invoked": False,
        "validation_examples": 0,
    }
    outcome = cast(dict[str, object], predecessor["outcome"])
    assert outcome == {
        "automatic_terminal_report_published": False,
        "checkpoint_reusable": False,
        "code": "facade_normalization_recipe_cannot_express_truthful_profile_005",
        "phase": "pre_registration_protocol_preflight",
        "reason": (
            "frozen_facade_normalization_cannot_express_the_truthful_profile_005_docstring"
        ),
        "status": "preflight_rejected",
    }
    assert not {"artifacts", "commit", "path", "sha256"} & outcome.keys()
    assert predecessor["registration"] == {
        "path": "configs/experiment-005-run.json",
        "present": False,
    }
    topology = cast(dict[str, object], predecessor["topology"])
    assert topology["outcome_report_path"] == "reports/experiment-005-training.json"
    assert topology["outcome_report_present"] is False
    assert topology["registration_present"] is False
    assert not (ROOT / "configs/experiment-005-run.json").exists()
    assert not (ROOT / "reports/experiment-005-training.json").exists()


def test_public_api_is_defaultless_positional_only_and_factory_is_deleted() -> None:
    build = inspect.signature(publication.build_execution_failure_evidence)
    publish = inspect.signature(publication.publish_registered_final_evidence)

    assert tuple(build.parameters) == ("registration_frame", "phase", "code")
    assert all(
        parameter.kind is inspect.Parameter.POSITIONAL_ONLY
        and parameter.default is inspect.Parameter.empty
        for parameter in build.parameters.values()
    )
    assert tuple(publish.parameters) == ("registration", "evidence")
    assert all(
        parameter.kind is inspect.Parameter.POSITIONAL_ONLY
        and parameter.default is inspect.Parameter.empty
        for parameter in publish.parameters.values()
    )
    assert not hasattr(publication, "_make_failure_evidence_routes")
    assert not hasattr(publication, "_make_registered_publisher")


def test_public_route_rejects_wrong_exact_types_without_filesystem_access() -> None:
    with pytest.raises(TypeError, match="registration"):
        publication.publish_registered_final_evidence(cast(Any, object()), object())


def test_public_route_closure_rejects_constant_os_and_class_tampering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closure = inspect.getclosurevars(publication.publish_registered_final_evidence)
    require_routes = cast(Any, closure.nonlocals["require_routes"])
    require_routes()

    with monkeypatch.context() as scoped:
        scoped.setattr(publication, "_REPORT_PATH", "reports/forged.json")
        with pytest.raises(
            publication.Experiment006FinalPublicationError, match="routes changed"
        ):
            require_routes()
    require_routes()

    def forged_link(*args: object, **kwargs: object) -> None:
        del args, kwargs

    with monkeypatch.context() as scoped:
        scoped.setattr(os, "link", forged_link)
        with pytest.raises(
            publication.Experiment006FinalPublicationError, match="routes changed"
        ):
            require_routes()
    require_routes()

    with monkeypatch.context() as scoped:
        scoped.setattr(os, "O_NOFOLLOW", 0)
        with pytest.raises(
            publication.Experiment006FinalPublicationError, match="routes changed"
        ):
            require_routes()
    require_routes()

    with monkeypatch.context() as scoped:
        scoped.setattr(publication._FilePlan, "forged", object(), raising=False)
        with pytest.raises(
            publication.Experiment006FinalPublicationError, match="routes changed"
        ):
            require_routes()
    require_routes()

    with monkeypatch.context() as scoped:
        scoped.setattr(publication, "_FailureState", object)
        with pytest.raises(
            publication.Experiment006FinalPublicationError, match="routes changed"
        ):
            require_routes()
    require_routes()

    linkat = cast(Any, publication._LINKAT)
    linkat.errcheck = lambda result, _function, _arguments: result
    try:
        with pytest.raises(
            publication.Experiment006FinalPublicationError, match="routes changed"
        ):
            require_routes()
    finally:
        del linkat.errcheck
    require_routes()

    original_linkat_flags = publication._LINKAT._flags_
    with monkeypatch.context() as scoped:
        scoped.setattr(publication._LINKAT, "_flags_", 0)
        with pytest.raises(
            publication.Experiment006FinalPublicationError, match="routes changed"
        ):
            require_routes()
    assert publication._LINKAT._flags_ == original_linkat_flags
    require_routes()

    layout = cast(Any, closure.nonlocals["layout"])
    original_repository_root = layout.repository_root
    object.__setattr__(layout, "repository_root", "/tmp/forged-publication-root")
    try:
        with pytest.raises(
            publication.Experiment006FinalPublicationError, match="routes changed"
        ):
            require_routes()
    finally:
        object.__setattr__(layout, "repository_root", original_repository_root)
    require_routes()

    open_directory_kwdefaults = publication._open_directory.__kwdefaults__
    assert open_directory_kwdefaults == {"dir_fd": None}
    open_directory_kwdefaults["dir_fd"] = 0
    try:
        with pytest.raises(
            publication.Experiment006FinalPublicationError,
            match="function authority changed",
        ):
            require_routes()
    finally:
        open_directory_kwdefaults["dir_fd"] = None
    require_routes()

    failure_route_closure = inspect.getclosurevars(
        publication.snapshot_execution_failure_publication
    )
    nested_failure_validator = cast(
        Any, failure_route_closure.nonlocals["validated_snapshot"]
    )
    original_nested_defaults = nested_failure_validator.__defaults__
    nested_failure_validator.__defaults__ = (object(),)
    try:
        with pytest.raises(
            publication.Experiment006FinalPublicationError,
            match="function authority changed",
        ):
            require_routes()
    finally:
        nested_failure_validator.__defaults__ = original_nested_defaults
    require_routes()

    authority_checker = cast(
        Any,
        inspect.getclosurevars(require_routes).nonlocals["require_function_authority"],
    )
    original_authority_checker_code = authority_checker.__code__

    def bypass_authority(_authority: object) -> None:
        return None

    authority_checker.__code__ = bypass_authority.__code__
    try:
        with pytest.raises(
            publication.Experiment006FinalPublicationError,
            match="routes changed",
        ):
            require_routes()
    finally:
        authority_checker.__code__ = original_authority_checker_code
    require_routes()

    def forged_sha256(_value: object = b"") -> object:
        return object()

    with monkeypatch.context() as scoped:
        scoped.setattr(hashlib, "sha256", forged_sha256)
        with pytest.raises(
            publication.Experiment006FinalPublicationError, match="routes changed"
        ):
            require_routes()
    require_routes()

    with monkeypatch.context() as scoped:
        scoped.setattr(json.JSONEncoder, "forged", object(), raising=False)
        with pytest.raises(
            publication.Experiment006FinalPublicationError, match="routes changed"
        ):
            require_routes()
    require_routes()

    decoder = cast(Any, json.JSONDecoder.__dict__["decode"])
    original_decoder_code = decoder.__code__

    def forged_decode(_self: object, _source: str) -> dict[str, object]:
        return {}

    decoder.__code__ = forged_decode.__code__
    try:
        with pytest.raises(
            publication.Experiment006FinalPublicationError,
            match="function authority changed",
        ):
            require_routes()
    finally:
        decoder.__code__ = original_decoder_code
    require_routes()

    scanner_module = cast(Any, json.__dict__["scanner"])
    with monkeypatch.context() as scoped:
        scoped.setattr(scanner_module, "make_scanner", lambda _context: object())
        with pytest.raises(
            publication.Experiment006FinalPublicationError, match="routes changed"
        ):
            require_routes()
    require_routes()

    encoder_module = cast(Any, json.encoder)
    with monkeypatch.context() as scoped:
        scoped.setattr(encoder_module, "c_make_encoder", lambda *args: args)
        with pytest.raises(
            publication.Experiment006FinalPublicationError, match="routes changed"
        ):
            require_routes()
    require_routes()

    with monkeypatch.context() as scoped:
        scoped.setitem(encoder_module.ESCAPE_DCT, "forged", "forged")
        with pytest.raises(
            publication.Experiment006FinalPublicationError, match="routes changed"
        ):
            require_routes()
    require_routes()

    route_closure = inspect.getclosurevars(require_routes)
    run_authority_module = cast(Any, route_closure.nonlocals["run_authority_module"])
    with monkeypatch.context() as scoped:
        scoped.setattr(run_authority_module, "_verified_state", lambda _value: object())
        with pytest.raises(
            publication.Experiment006FinalPublicationError, match="routes changed"
        ):
            require_routes()
    require_routes()

    registration_type = cast(Any, closure.nonlocals["registration_type"])
    getter = cast(Any, registration_type.__dict__["head_commit"].fget)
    original_getter_code = getter.__code__

    def forged_getter(_self: object) -> str:
        return "f" * 40

    getter.__code__ = forged_getter.__code__
    try:
        with pytest.raises(
            publication.Experiment006FinalPublicationError,
            match="function authority changed",
        ):
            require_routes()
    finally:
        getter.__code__ = original_getter_code
    require_routes()


@pytest.mark.parametrize("phase,code", sorted(publication.EXECUTION_FAILURE_PAIRS))
def test_each_closed_failure_pair_issues_exact_canonical_evidence(
    phase: publication.FailurePhase,
    code: publication.FailureCode,
) -> None:
    evidence = publication.build_execution_failure_evidence(_REGISTRATION, phase, code)
    publication.verify_execution_failure_evidence(evidence)
    snapshot = publication.snapshot_execution_failure_publication(evidence)

    assert snapshot[:5] == (
        "execution_failure",
        _REGISTRATION,
        (),
        _REPORT_PATH,
        "0444",
    )
    report = cast(dict[str, Any], json.loads(snapshot[5]))
    assert snapshot[5] == _canonical(report)
    assert report["failure"] == {"phase": phase, "code": code}
    assert report["artifacts"] == []
    assert report["checkpoint_reusable"] is False
    assert report["experiment"] == "006"
    assert report["scientific_protocol"] == "002"
    assert report["predecessor"] == _PREDECESSOR
    lowered = snapshot[5].lower()
    assert b"exception" not in lowered
    assert b"traceback" not in lowered
    assert b"message" not in lowered


@pytest.mark.parametrize(
    "phase,code",
    [
        ("parent_setup", "seed_selection_failed"),
        ("registered_execution", "staging_prepare_failed"),
        ("completed_evidence", "pre_cleanup_quiescence_failed"),
        ("unknown", "unknown"),
    ],
)
def test_unregistered_failure_pair_is_rejected(
    phase: str,
    code: str,
) -> None:
    with pytest.raises(
        publication.Experiment006FinalPublicationError, match="not registered"
    ):
        publication.build_execution_failure_evidence(
            _REGISTRATION,
            cast(publication.FailurePhase, phase),
            cast(publication.FailureCode, code),
        )


def test_failure_evidence_rejects_forgery_copy_pickle_and_mutation() -> None:
    with pytest.raises(TypeError, match="issued"):
        publication.FinalExecutionFailureEvidence()
    forged = object.__new__(publication.FinalExecutionFailureEvidence)
    with pytest.raises(publication.Experiment006FinalPublicationError):
        publication.verify_execution_failure_evidence(forged)

    evidence = publication.build_execution_failure_evidence(
        _REGISTRATION, "registered_execution", "seed_selection_failed"
    )
    with pytest.raises(TypeError, match="copied"):
        copy.copy(evidence)
    with pytest.raises(TypeError, match="serialized"):
        pickle.dumps(evidence)
    object.__setattr__(evidence, "code", "source_bundle_close_failed")
    with pytest.raises(publication.Experiment006FinalPublicationError, match="changed"):
        publication.snapshot_execution_failure_publication(evidence)


def test_failure_issuer_registry_ignores_weakref_route_forgery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legitimate = publication.build_execution_failure_evidence(
        _REGISTRATION,
        "registered_execution",
        "seed_selection_failed",
    )
    snapshot = publication.snapshot_execution_failure_publication(legitimate)
    field_names = (
        "schema_version",
        "experiment",
        "status",
        "registration",
        "phase",
        "code",
        "report_path",
        "report_mode",
    )
    values = tuple(getattr(legitimate, name) for name in field_names)
    forged = object.__new__(publication.FinalExecutionFailureEvidence)
    for name, value in zip(field_names, values, strict=True):
        object.__setattr__(forged, name, value)
    forged_state = publication._FailureState(frame=values, snapshot=snapshot)

    with pytest.raises(
        publication.Experiment006FinalPublicationError,
        match="not issued",
    ):
        publication.verify_execution_failure_evidence(forged)

    with monkeypatch.context() as scoped:
        scoped.setattr(
            weakref.WeakKeyDictionary,
            "get",
            lambda _self, _key, _default=None: forged_state,
        )
        with pytest.raises(
            publication.Experiment006FinalPublicationError,
            match="not issued",
        ):
            publication.verify_execution_failure_evidence(forged)

    legitimate_ref = weakref.ref(legitimate)
    with monkeypatch.context() as scoped:
        scoped.setattr(
            weakref,
            "ref",
            lambda _value, _callback=None: legitimate_ref,
        )
        with pytest.raises(
            publication.Experiment006FinalPublicationError,
            match="not issued",
        ):
            publication.snapshot_execution_failure_publication(forged)

    snapshot_closure = inspect.getclosurevars(
        publication.snapshot_execution_failure_publication
    )
    validator = cast(Any, snapshot_closure.nonlocals["validated_snapshot"])
    registry = inspect.getclosurevars(validator).nonlocals["issued"]
    assert type(registry) is dict


def test_failure_issuer_rejects_force_pair_and_persistent_descriptor_tampering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ForcePair(str):
        def __eq__(self, other: object) -> bool:
            del other
            return True

        def __hash__(self) -> int:
            return hash("parent_setup")

    with pytest.raises(TypeError, match="exact strings"):
        publication.build_execution_failure_evidence(
            _REGISTRATION,
            cast(Any, ForcePair("forged")),
            cast(Any, ForcePair("forged")),
        )

    evidence = publication.build_execution_failure_evidence(
        _REGISTRATION,
        "registered_execution",
        "seed_selection_failed",
    )
    with monkeypatch.context() as scoped:
        scoped.setattr(
            publication.FinalExecutionFailureEvidence,
            "code",
            property(lambda _self: "seed_selection_failed"),
        )
        with pytest.raises(
            publication.Experiment006FinalPublicationError,
            match="issuer authority changed",
        ):
            publication.verify_execution_failure_evidence(evidence)
    publication.verify_execution_failure_evidence(evidence)

    with monkeypatch.context() as scoped:
        scoped.setattr(
            publication._FailureState,
            "snapshot",
            property(lambda _self: _failure_snapshot()),
        )
        with pytest.raises(
            publication.Experiment006FinalPublicationError,
            match="issuer authority changed",
        ):
            publication.verify_execution_failure_evidence(evidence)
    publication.verify_execution_failure_evidence(evidence)


def test_failure_issuer_rejects_helper_code_tamper_before_report_construction() -> None:
    evidence = publication.build_execution_failure_evidence(
        _REGISTRATION,
        "registered_execution",
        "seed_selection_failed",
    )
    original_code = publication._failure_report_bytes.__code__

    def forged_report(
        _registration: object,
        _phase: object,
        _code: object,
    ) -> bytes:
        return b'{"exception":"attacker-controlled"}\n'

    publication._failure_report_bytes.__code__ = forged_report.__code__
    try:
        with pytest.raises(
            publication.Experiment006FinalPublicationError,
            match="issuer function authority changed",
        ):
            publication.build_execution_failure_evidence(
                _REGISTRATION,
                "registered_execution",
                "seed_selection_failed",
            )
        with pytest.raises(
            publication.Experiment006FinalPublicationError,
            match="issuer function authority changed",
        ):
            publication.verify_execution_failure_evidence(evidence)
        with pytest.raises(
            publication.Experiment006FinalPublicationError,
            match="issuer function authority changed",
        ):
            publication.snapshot_execution_failure_publication(evidence)
    finally:
        publication._failure_report_bytes.__code__ = original_code

    publication.verify_execution_failure_evidence(evidence)
    assert (
        b"exception"
        not in publication.snapshot_execution_failure_publication(evidence)[5]
    )


def test_failure_report_is_independent_of_mutable_json_encoder_globals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = publication.snapshot_execution_failure_publication(
        publication.build_execution_failure_evidence(
            _REGISTRATION,
            "registered_execution",
            "seed_selection_failed",
        )
    )

    class EvilJSONEncoder:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def encode(self, _value: object) -> str:
            return '{"exception":"attacker-controlled"}'

    with monkeypatch.context() as scoped:
        scoped.setattr(json, "JSONEncoder", EvilJSONEncoder)
        observed = publication.snapshot_execution_failure_publication(
            publication.build_execution_failure_evidence(
                _REGISTRATION,
                "registered_execution",
                "seed_selection_failed",
            )
        )

    assert observed == expected
    assert b"exception" not in observed[5]


def test_failure_builder_rejects_nonexact_or_invalid_registration_frame() -> None:
    class TupleSubclass(tuple[object, ...]):
        pass

    for value in (
        TupleSubclass(_REGISTRATION),
        ("a" * 39, *_REGISTRATION[1:]),
        (_REGISTRATION[1], _REGISTRATION[1], *_REGISTRATION[2:]),
        (*_REGISTRATION, "extra"),
    ):
        with pytest.raises(publication.Experiment006FinalPublicationError):
            publication.build_execution_failure_evidence(
                cast(Any, value), "registered_execution", "seed_selection_failed"
            )


@pytest.mark.parametrize("status", ["pass", "gate_failure"])
def test_completed_publication_writes_exact_set_report_last_and_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    root = tmp_path / status
    layout = _layout(root)
    snapshot = _completed_snapshot(status)
    calls: list[str] = []
    link_order: list[str] = []
    original_link = publication._link_descriptor_no_overwrite

    def install_recorder() -> None:
        def recording_link(
            source: int,
            destination_parent: int,
            destination: str,
        ) -> None:
            link_order.append(destination)
            original_link(source, destination_parent, destination)

        monkeypatch.setattr(
            publication, "_link_descriptor_no_overwrite", recording_link
        )

    publication._publish_snapshot(
        snapshot,
        layout,
        _fresh_callback(root, calls, install_recorder),
    )

    expected_paths = [plan[0] for plan in snapshot[2]] + [_REPORT_PATH]
    assert calls == ["revalidate"]
    assert link_order[-1] == Path(_REPORT_PATH).name
    assert link_order == [Path(path).name for path in expected_paths]
    for plan in snapshot[2]:
        _assert_file(_path(root, plan[0]), plan[3])
    _assert_file(_path(root, _REPORT_PATH), snapshot[5])
    assert _path(root, _MODEL_PATH).exists() is (status == "pass")
    assert not any(
        entry.name.startswith(_TEMP_PREFIX)
        for directory in (root / "reports", root / "models")
        for entry in directory.iterdir()
    )


def test_execution_failure_publication_writes_only_report(tmp_path: Path) -> None:
    root = tmp_path / "failure"
    layout = _layout(root)
    snapshot = _failure_snapshot()
    calls: list[str] = []

    publication._publish_snapshot(snapshot, layout, _fresh_callback(root, calls))

    assert calls == ["revalidate"]
    _assert_file(_path(root, _REPORT_PATH), snapshot[5])
    assert all(not _path(root, path).exists() for path in _MANAGED_PATHS[:-1])


def test_publication_never_touches_experiment_002_outputs(tmp_path: Path) -> None:
    root = tmp_path / "predecessor-sentinels"
    layout = _layout(root)
    predecessor_paths = (
        "reports/experiment-002-seed-20260719-history.json",
        "reports/experiment-002-seed-20260720-history.json",
        "reports/experiment-002-seed-20260721-history.json",
        "reports/experiment-002-selected-rerun-history.json",
        "models/experiment-002-selected.safetensors",
        "reports/experiment-002-training.json",
    )
    sentinel = b"immutable experiment 002 output"
    fingerprints: dict[str, tuple[int, int, int]] = {}
    for relative in predecessor_paths:
        path = _path(root, relative)
        path.write_bytes(sentinel)
        path.chmod(0o444)
        observed = path.stat()
        fingerprints[relative] = (
            observed.st_ino,
            stat.S_IMODE(observed.st_mode),
            observed.st_nlink,
        )

    snapshot = _failure_snapshot()
    publication._publish_snapshot(
        snapshot,
        layout,
        _fresh_callback(root, []),
    )

    for relative in predecessor_paths:
        path = _path(root, relative)
        observed = path.stat()
        assert path.read_bytes() == sentinel
        assert (
            observed.st_ino,
            stat.S_IMODE(observed.st_mode),
            observed.st_nlink,
        ) == fingerprints[relative]
    _assert_file(_path(root, _REPORT_PATH), snapshot[5])


@pytest.mark.parametrize("relative", _MANAGED_PATHS)
def test_every_preexisting_managed_leaf_is_preserved_and_blocks_before_revalidation(
    tmp_path: Path,
    relative: str,
) -> None:
    root = tmp_path / hashlib.sha256(relative.encode()).hexdigest()[:8]
    layout = _layout(root)
    destination = _path(root, relative)
    sentinel = b"foreign preexisting bytes"
    destination.write_bytes(sentinel)
    calls: list[str] = []

    with pytest.raises(
        publication.Experiment006FinalPublicationError, match="not fresh"
    ):
        publication._publish_snapshot(
            _completed_snapshot(), layout, _fresh_callback(root, calls)
        )

    assert calls == []
    assert destination.read_bytes() == sentinel
    assert stat.S_IMODE(destination.stat().st_mode) != 0o444


def test_symlink_destination_and_crash_prefix_are_preserved_and_blocked(
    tmp_path: Path,
) -> None:
    for variant in ("symlink", "crash"):
        root = tmp_path / variant
        layout = _layout(root)
        outside = root / "outside"
        outside.write_bytes(b"outside")
        if variant == "symlink":
            (root / _REPORT_PATH).symlink_to(outside)
        else:
            (root / "reports" / f"{_TEMP_PREFIX}foreign.tmp").write_bytes(b"debris")
        calls: list[str] = []
        with pytest.raises(publication.Experiment006FinalPublicationError):
            publication._publish_snapshot(
                _failure_snapshot(), layout, _fresh_callback(root, calls)
            )
        assert calls == []
        assert outside.read_bytes() == b"outside"


def test_revalidation_failure_happens_before_any_temporary_or_destination(
    tmp_path: Path,
) -> None:
    root = tmp_path / "revalidation"
    layout = _layout(root)

    def reject() -> None:
        raise RuntimeError("synthetic authority rejection")

    with pytest.raises(RuntimeError, match="authority rejection"):
        publication._publish_snapshot(_completed_snapshot(), layout, reject)

    assert all(not _path(root, path).exists() for path in _MANAGED_PATHS)
    assert not any(
        entry.name.startswith(_TEMP_PREFIX)
        for directory in (root / "reports", root / "models")
        for entry in directory.iterdir()
    )


def test_parent_replacement_after_revalidation_is_detected_before_write(
    tmp_path: Path,
) -> None:
    root = tmp_path / "parent-swap"
    layout = _layout(root)
    calls: list[str] = []

    def replace_parent() -> None:
        (root / "reports").rename(root / "reports-owned")
        (root / "reports").mkdir()

    with pytest.raises(
        publication.Experiment006FinalPublicationError, match="directory name changed"
    ):
        publication._publish_snapshot(
            _completed_snapshot(),
            layout,
            _fresh_callback(root, calls, replace_parent),
        )

    assert calls == ["revalidate"]
    assert list((root / "reports-owned").iterdir()) == []
    assert list((root / "reports").iterdir()) == []


def test_symlinked_parent_is_rejected_without_following_it(tmp_path: Path) -> None:
    root = tmp_path / "symlinked-parent"
    layout = _layout(root)
    (root / "reports").rmdir()
    outside = root / "outside-reports"
    outside.mkdir()
    (root / "reports").symlink_to(outside, target_is_directory=True)

    with pytest.raises(
        publication.Experiment006FinalPublicationError, match="securely opened"
    ):
        publication._publish_snapshot(
            _failure_snapshot(), layout, _fresh_callback(root, [])
        )

    assert list(outside.iterdir()) == []


def test_partial_and_interrupted_descriptor_io_is_completed_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "partial-io"
    layout = _layout(root)
    snapshot = _failure_snapshot()
    original_pwrite = os.pwrite
    original_pread = os.pread
    interrupted_write = False
    interrupted_read = False

    def partial_pwrite(descriptor: int, contents: Any, offset: int) -> int:
        nonlocal interrupted_write
        if not interrupted_write:
            interrupted_write = True
            raise InterruptedError
        return original_pwrite(descriptor, memoryview(contents)[:3], offset)

    def partial_pread(descriptor: int, count: int, offset: int) -> bytes:
        nonlocal interrupted_read
        if not interrupted_read:
            interrupted_read = True
            raise InterruptedError
        return original_pread(descriptor, min(count, 2), offset)

    monkeypatch.setattr(os, "pwrite", partial_pwrite)
    monkeypatch.setattr(os, "pread", partial_pread)
    publication._publish_snapshot(snapshot, layout, _fresh_callback(root, []))

    _assert_file(_path(root, _REPORT_PATH), snapshot[5])
    assert interrupted_write is True
    assert interrupted_read is True


def test_prelink_write_failure_cleans_only_owned_temporaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "write-failure"
    layout = _layout(root)
    original_pwrite = os.pwrite
    writes = 0

    def failing_pwrite(descriptor: int, contents: Any, offset: int) -> int:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("synthetic write failure")
        return original_pwrite(descriptor, contents, offset)

    monkeypatch.setattr(os, "pwrite", failing_pwrite)
    with pytest.raises(OSError, match="write failure"):
        publication._publish_snapshot(
            _completed_snapshot(), layout, _fresh_callback(root, [])
        )

    assert all(not _path(root, path).exists() for path in _MANAGED_PATHS)
    assert not any(
        entry.name.startswith(_TEMP_PREFIX)
        for directory in (root / "reports", root / "models")
        for entry in directory.iterdir()
    )


@pytest.mark.parametrize(("failure_call", "temporary_slot"), [(1, 0), (5, 4)])
def test_local_prepare_cleanup_fsyncs_its_exact_parent_after_unlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_call: int,
    temporary_slot: int,
) -> None:
    root = tmp_path / f"cleanup-fsync-{failure_call}"
    layout = _layout(root)
    original_write_all = publication._write_all
    original_unlink = os.unlink
    original_fsync = os.fsync
    write_calls = 0
    events: list[tuple[str, int, str]] = []

    def failing_write_all(descriptor: int, contents: bytes) -> None:
        nonlocal write_calls
        write_calls += 1
        if write_calls == failure_call:
            raise RuntimeError("synthetic local prepare failure")
        original_write_all(descriptor, contents)

    def tracking_unlink(path: str, *, dir_fd: int | None = None) -> None:
        original_unlink(path, dir_fd=dir_fd)
        if path.startswith(_TEMP_PREFIX):
            assert dir_fd is not None
            events.append(("unlink", dir_fd, path))

    def tracking_fsync(descriptor: int) -> None:
        events.append(("fsync", descriptor, ""))
        original_fsync(descriptor)

    def install_failure() -> None:
        monkeypatch.setattr(publication, "_write_all", failing_write_all)
        monkeypatch.setattr(os, "unlink", tracking_unlink)
        monkeypatch.setattr(os, "fsync", tracking_fsync)

    with pytest.raises(RuntimeError, match="local prepare failure"):
        publication._publish_snapshot(
            _completed_snapshot(), layout, _fresh_callback(root, [], install_failure)
        )

    suffix = f"-{temporary_slot:02d}.tmp"
    unlink_index, unlink_event = next(
        (index, event)
        for index, event in enumerate(events)
        if event[0] == "unlink" and event[2].endswith(suffix)
    )
    assert any(
        event == ("fsync", unlink_event[1], "") for event in events[unlink_index + 1 :]
    )
    assert all(not _path(root, path).exists() for path in _MANAGED_PATHS)
    assert not any(
        entry.name.startswith(_TEMP_PREFIX)
        for directory in (root / "reports", root / "models")
        for entry in directory.iterdir()
    )


def test_local_prepare_cleanup_parent_fsync_error_is_surfaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "cleanup-fsync-error"
    layout = _layout(root)
    original_fsync = os.fsync

    def fail_write(_descriptor: int, _contents: bytes) -> None:
        raise RuntimeError("synthetic write failure")

    def fail_directory_fsync(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("synthetic cleanup directory fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(publication, "_write_all", fail_write)
    monkeypatch.setattr(os, "fsync", fail_directory_fsync)

    with pytest.raises(
        publication.Experiment006FinalPublicationError,
        match="preparation failed with cleanup errors",
    ):
        publication._publish_snapshot(
            _failure_snapshot(), layout, _fresh_callback(root, [])
        )

    assert not _path(root, _REPORT_PATH).exists()
    assert not any(
        entry.name.startswith(_TEMP_PREFIX) for entry in (root / "reports").iterdir()
    )


def test_link_race_never_overwrites_or_unlinks_concurrent_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "link-race"
    layout = _layout(root)
    original_link = publication._link_descriptor_no_overwrite
    sentinel = b"concurrent publisher"

    def install_race() -> None:
        raced = False

        def racing_link(
            source: int,
            destination_parent: int,
            destination: str,
        ) -> None:
            nonlocal raced
            if not raced:
                raced = True
                descriptor = os.open(
                    destination,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=destination_parent,
                )
                try:
                    os.write(descriptor, sentinel)
                finally:
                    os.close(descriptor)
            original_link(source, destination_parent, destination)

        monkeypatch.setattr(publication, "_link_descriptor_no_overwrite", racing_link)

    with pytest.raises(
        publication.Experiment006FinalPublicationError, match="appeared"
    ):
        publication._publish_snapshot(
            _completed_snapshot(),
            layout,
            _fresh_callback(root, [], install_race),
        )

    assert _path(root, _HISTORY_PATHS[0]).read_bytes() == sentinel
    assert not _path(root, _REPORT_PATH).exists()


def test_failure_after_first_commit_preserves_exact_prefix_and_crash_temporaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "immutable-prefix"
    layout = _layout(root)
    snapshot = _completed_snapshot()
    original_link = publication._link_descriptor_no_overwrite
    link_count = 0

    def install_failure() -> None:
        def failing_link(
            source: int,
            destination_parent: int,
            destination: str,
        ) -> None:
            nonlocal link_count
            link_count += 1
            if link_count == 2:
                raise OSError("synthetic second-link failure")
            original_link(source, destination_parent, destination)

        monkeypatch.setattr(publication, "_link_descriptor_no_overwrite", failing_link)

    with pytest.raises(OSError, match="second-link failure"):
        publication._publish_snapshot(
            snapshot, layout, _fresh_callback(root, [], install_failure)
        )

    _assert_file(_path(root, _HISTORY_PATHS[0]), snapshot[2][0][3])
    assert not _path(root, _HISTORY_PATHS[1]).exists()
    assert not _path(root, _REPORT_PATH).exists()
    crash_names = [
        entry.name
        for directory in (root / "reports", root / "models")
        for entry in directory.iterdir()
        if entry.name.startswith(_TEMP_PREFIX)
    ]
    assert crash_names
    assert all(_REGISTRATION[2] in name for name in crash_names)


@pytest.mark.parametrize(
    ("status", "failure_index"),
    (
        *(("pass", index) for index in range(6)),
        *(("gate_failure", index) for index in range(5)),
        ("execution_failure", 0),
    ),
)
def test_every_precommit_link_failure_preserves_only_the_exact_committed_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    failure_index: int,
) -> None:
    root = tmp_path / f"link-fault-{status}-{failure_index}"
    layout = _layout(root)
    snapshot = (
        _failure_snapshot()
        if status == "execution_failure"
        else _completed_snapshot(status)
    )
    original_link = publication._link_descriptor_no_overwrite
    observed_index = 0

    def install_failure() -> None:
        def failing_link(
            source: int,
            destination_parent: int,
            destination: str,
        ) -> None:
            nonlocal observed_index
            current = observed_index
            observed_index += 1
            if current == failure_index:
                raise OSError("synthetic indexed link failure")
            original_link(source, destination_parent, destination)

        monkeypatch.setattr(publication, "_link_descriptor_no_overwrite", failing_link)

    with pytest.raises(OSError, match="indexed link failure"):
        publication._publish_snapshot(
            snapshot,
            layout,
            _fresh_callback(root, [], install_failure),
        )

    ordered_paths = [plan[0] for plan in snapshot[2]] + [_REPORT_PATH]
    ordered_contents = [plan[3] for plan in snapshot[2]] + [snapshot[5]]
    for path, contents in zip(
        ordered_paths[:failure_index],
        ordered_contents[:failure_index],
        strict=True,
    ):
        _assert_file(_path(root, path), contents)
    for path in ordered_paths[failure_index:]:
        assert not _path(root, path).exists()
    crash_temporaries = [
        entry
        for directory in (root / "reports", root / "models")
        for entry in directory.iterdir()
        if entry.name.startswith(_TEMP_PREFIX)
    ]
    assert bool(crash_temporaries) is (failure_index > 0)
    assert not _path(root, _REPORT_PATH).exists()


def test_payload_tamper_is_rejected_before_report_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "payload-tamper"
    layout = _layout(root)
    snapshot = _completed_snapshot()
    original_commit = publication._commit_file
    corrupted = False

    def install_tamper() -> None:
        def tampering_commit(
            pinned: publication._PinnedLayout,
            prepared: publication._PreparedFile,
            state: publication._CommitState,
        ) -> None:
            nonlocal corrupted
            original_commit(pinned, prepared, state)
            if not corrupted and prepared.plan.path == _HISTORY_PATHS[0]:
                corrupted = True
                destination = _path(root, _HISTORY_PATHS[0])
                destination.chmod(0o644)
                destination.write_bytes(b"concurrent payload corruption")

        monkeypatch.setattr(publication, "_commit_file", tampering_commit)

    with pytest.raises(
        publication.Experiment006FinalPublicationError,
        match="mode, size, owner, or link count",
    ):
        publication._publish_snapshot(
            snapshot,
            layout,
            _fresh_callback(root, [], install_tamper),
        )

    assert corrupted is True
    assert not _path(root, _REPORT_PATH).exists()


def test_report_name_race_before_commit_is_preserved_and_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "report-race"
    layout = _layout(root)
    snapshot = _completed_snapshot()
    original_commit = publication._commit_file
    sentinel = b"concurrent report"
    raced = False

    def install_race() -> None:
        def racing_commit(
            pinned: publication._PinnedLayout,
            prepared: publication._PreparedFile,
            state: publication._CommitState,
        ) -> None:
            nonlocal raced
            original_commit(pinned, prepared, state)
            if not raced and prepared.plan.path == _HISTORY_PATHS[0]:
                raced = True
                _path(root, _REPORT_PATH).write_bytes(sentinel)

        monkeypatch.setattr(publication, "_commit_file", racing_commit)

    with pytest.raises(
        publication.Experiment006FinalPublicationError,
        match="appeared before final report",
    ):
        publication._publish_snapshot(
            snapshot,
            layout,
            _fresh_callback(root, [], install_race),
        )

    assert raced is True
    assert _path(root, _REPORT_PATH).read_bytes() == sentinel


def test_final_report_inode_swap_during_postverify_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "postverify-report-swap"
    layout = _layout(root)
    snapshot = _failure_snapshot()
    original_listdir = os.listdir
    swapped = False

    def install_swap() -> None:
        def swapping_listdir(path: Any) -> list[str]:
            nonlocal swapped
            report = _path(root, _REPORT_PATH)
            if report.exists() and not swapped:
                swapped = True
                report.rename(report.with_name(f"{report.name}.original"))
                report.write_bytes(snapshot[5])
                report.chmod(0o444)
            return original_listdir(path)

        monkeypatch.setattr(os, "listdir", swapping_listdir)

    with pytest.raises(
        publication.Experiment006FinalPublicationError,
        match="built inode",
    ):
        publication._publish_snapshot(
            snapshot,
            layout,
            _fresh_callback(root, [], install_swap),
        )

    assert swapped is True
    assert _path(root, _REPORT_PATH).read_bytes() == snapshot[5]


def test_snapshot_rejects_wrong_payload_set_digest_mode_and_budget() -> None:
    good = _completed_snapshot()
    mutations: list[object] = []
    mutations.append(("pass", good[1], good[2][:-1], *good[3:]))
    bad_mode = list(good[2][0])
    bad_mode[2] = "0644"
    mutations.append((good[0], good[1], (tuple(bad_mode), *good[2][1:]), *good[3:]))
    bad_digest = list(good[2][0])
    bad_digest[5] = "0" * 64
    mutations.append((good[0], good[1], (tuple(bad_digest), *good[2][1:]), *good[3:]))
    oversized_report = b"x" * ((256 << 10) + 1)
    mutations.append((*good[:5], oversized_report))

    for value in mutations:
        with pytest.raises(publication.Experiment006FinalPublicationError):
            publication._require_publication_snapshot(value)


def test_snapshot_rejects_unregistered_report_and_artifact_fields() -> None:
    failure = _failure_snapshot()
    failure_document = cast(dict[str, object], json.loads(failure[5]))
    failure_document["schema_version"] = True
    with pytest.raises(
        publication.Experiment006FinalPublicationError, match="identity or status"
    ):
        publication._require_publication_snapshot(
            (*failure[:5], _canonical(failure_document))
        )

    failure_document["schema_version"] = 1
    failure_document["exception"] = "forbidden exception text"
    with pytest.raises(
        publication.Experiment006FinalPublicationError, match="unexpected or missing"
    ):
        publication._require_publication_snapshot(
            (*failure[:5], _canonical(failure_document))
        )

    completed = _completed_snapshot()
    completed_document = cast(dict[str, Any], json.loads(completed[5]))
    artifacts = cast(list[dict[str, object]], completed_document["artifacts"])
    artifacts[0]["source_child_ordinal"] = False
    with pytest.raises(
        publication.Experiment006FinalPublicationError,
        match="differs from its payload plan",
    ):
        publication._require_publication_snapshot(
            (*completed[:5], _canonical(completed_document))
        )

    artifacts[0]["source_child_ordinal"] = 0
    artifacts[0]["exception"] = "forbidden artifact field"
    with pytest.raises(
        publication.Experiment006FinalPublicationError, match="unexpected or missing"
    ):
        publication._require_publication_snapshot(
            (*completed[:5], _canonical(completed_document))
        )


def test_completed_snapshot_rejects_schema_shell_and_nested_semantic_forgery() -> None:
    completed = _completed_snapshot()
    base = cast(dict[str, Any], json.loads(completed[5]))
    mutations: list[dict[str, Any]] = []

    shell = copy.deepcopy(base)
    shell["children"] = []
    shell["selection"] = {}
    shell["rerun"] = {}
    shell["gates"] = {"all_passed": True}
    mutations.append(shell)

    wrong_selection = copy.deepcopy(base)
    wrong_selection["selection"]["selected_training_ordinal"] = 0
    mutations.append(wrong_selection)

    bool_alias = copy.deepcopy(base)
    bool_alias["selection"]["selected_training_ordinal"] = True
    mutations.append(bool_alias)

    false_rerun = copy.deepcopy(base)
    false_rerun["rerun"]["history_byte_identical"] = False
    mutations.append(false_rerun)

    fabricated_gates = copy.deepcopy(base)
    fabricated_gates["gates"]["all_passed"] = False
    mutations.append(fabricated_gates)

    forged_child = copy.deepcopy(base)
    forged_child["children"][0]["history"]["sha256"] = "0" * 64
    mutations.append(forged_child)

    forged_winner = copy.deepcopy(base)
    forged_winner["children"][3]["winner"]["zero_based_epoch"] = 29
    mutations.append(forged_winner)

    for document in mutations:
        with pytest.raises(publication.Experiment006FinalPublicationError):
            publication._require_publication_snapshot(
                (*completed[:5], _canonical(document))
            )


def test_aggregate_budget_exact_boundary_and_one_byte_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _completed_snapshot()
    other_bytes = sum(plan[4] for plan in base[2][1:])
    target = publication._PUBLICATION_BYTES_MAXIMUM - len(base[5]) - other_bytes

    def accept_report(_snapshot: object) -> None:
        return None

    monkeypatch.setattr(
        publication,
        "_require_report_matches_snapshot",
        accept_report,
    )

    def snapshot_with_first_payload_size(byte_count: int) -> _Snapshot:
        contents = b"x" * byte_count
        digest = hashlib.sha256(contents).hexdigest()
        first = list(base[2][0])
        first[3] = contents
        first[4] = byte_count
        first[5] = digest
        plans = (cast(_Plan, tuple(first)), *base[2][1:])
        document = cast(dict[str, Any], json.loads(base[5]))
        artifacts = cast(list[dict[str, object]], document["artifacts"])
        artifacts[0]["byte_count"] = byte_count
        artifacts[0]["sha256"] = digest
        return base[0], base[1], plans, base[3], base[4], _canonical(document)

    for _ in range(4):
        candidate = snapshot_with_first_payload_size(target)
        adjusted = (
            publication._PUBLICATION_BYTES_MAXIMUM - len(candidate[5]) - other_bytes
        )
        if adjusted == target:
            break
        target = adjusted
    boundary = snapshot_with_first_payload_size(target)
    assert len(boundary[5]) + sum(plan[4] for plan in boundary[2]) == (
        publication._PUBLICATION_BYTES_MAXIMUM
    )
    assert publication._require_publication_snapshot(boundary) == boundary

    oversized = snapshot_with_first_payload_size(target + 1)
    assert len(oversized[5]) + sum(plan[4] for plan in oversized[2]) == (
        publication._PUBLICATION_BYTES_MAXIMUM + 1
    )
    with pytest.raises(
        publication.Experiment006FinalPublicationError, match="byte budget"
    ):
        publication._require_publication_snapshot(oversized)


def test_public_publisher_rejects_thread_and_fork_and_burns_failed_attempt() -> None:
    script = r"""
import inspect
import os
import threading

import falsewake.experiment_006_final_publication as publication

frame = ("a" * 40, "b" * 40, "c" * 64, "d" * 64)
evidence = publication.build_execution_failure_evidence(
    frame, "registered_execution", "seed_selection_failed"
)
closure = inspect.getclosurevars(publication.publish_registered_final_evidence)
registration_type = closure.nonlocals["registration_type"]
registration = object.__new__(registration_type)

publisher_lock = closure.nonlocals["lock"]
assert publisher_lock.acquire(blocking=False)
try:
    try:
        publication.publish_registered_final_evidence(registration, evidence)
    except publication.Experiment006FinalPublicationError as error:
        assert "overlaps another call" in str(error)
    else:
        raise AssertionError("overlapping publication was accepted")
finally:
    publisher_lock.release()

thread_errors = []
def call_from_thread():
    try:
        publication.publish_registered_final_evidence(registration, evidence)
    except BaseException as error:
        thread_errors.append(str(error))

thread = threading.Thread(target=call_from_thread)
thread.start()
thread.join()
assert len(thread_errors) == 1 and "thread changed" in thread_errors[0]

read_descriptor, write_descriptor = os.pipe()
child = os.fork()
if child == 0:
    os.close(read_descriptor)
    try:
        publication.publish_registered_final_evidence(registration, evidence)
    except BaseException as error:
        os.write(write_descriptor, str(error).encode("ascii"))
        os._exit(0)
    os._exit(7)
os.close(write_descriptor)
fork_error = os.read(read_descriptor, 4096).decode("ascii")
os.close(read_descriptor)
_, wait_status = os.waitpid(child, 0)
assert os.waitstatus_to_exitcode(wait_status) == 0
assert "inherited by a fork" in fork_error

expected_main_error = (
    "managed publication namespace is not fresh"
    if os.path.isfile("reports/experiment-006-training.json")
    else "not issued"
)
try:
    publication.publish_registered_final_evidence(registration, evidence)
except BaseException as error:
    assert expected_main_error in str(error)
else:
    raise AssertionError("forged capability unexpectedly published")

try:
    publication.publish_registered_final_evidence(registration, evidence)
except publication.Experiment006FinalPublicationError as error:
    assert "already attempted" in str(error)
else:
    raise AssertionError("failed publication attempt was replayable")

print("publisher-guards-ok")
"""
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "publisher-guards-ok\n"


def test_replaced_temporary_name_cannot_publish_attacker_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "source-name-swap"
    layout = _layout(root)
    snapshot = _failure_snapshot()
    original_link = publication._link_descriptor_no_overwrite
    attacker = b"attacker-controlled report bytes"

    def install_swap() -> None:
        def swapping_link(
            source: int,
            destination_parent: int,
            destination: str,
        ) -> None:
            temporary_names = [
                name
                for name in os.listdir(destination_parent)
                if name.startswith(_TEMP_PREFIX)
            ]
            assert len(temporary_names) == 1
            moved_name = f"{temporary_names[0]}-moved"
            os.rename(
                temporary_names[0],
                moved_name,
                src_dir_fd=destination_parent,
                dst_dir_fd=destination_parent,
            )
            descriptor = os.open(
                temporary_names[0],
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=destination_parent,
            )
            try:
                os.write(descriptor, attacker)
            finally:
                os.close(descriptor)
            original_link(source, destination_parent, destination)

        monkeypatch.setattr(publication, "_link_descriptor_no_overwrite", swapping_link)

    with pytest.raises(
        publication.Experiment006FinalPublicationError, match="names do not identify"
    ):
        publication._publish_snapshot(
            snapshot, layout, _fresh_callback(root, [], install_swap)
        )

    assert _path(root, _REPORT_PATH).read_bytes() == snapshot[5]
    assert _path(root, _REPORT_PATH).read_bytes() != attacker
    assert any(
        entry.name.startswith(_TEMP_PREFIX) for entry in (root / "reports").iterdir()
    )


def test_report_is_never_accepted_before_payloads_in_source_contract() -> None:
    source = Path(publication.__file__).read_text(encoding="utf-8")
    public_source = inspect.getsource(publication.publish_registered_final_evidence)
    commit_loop = "for index, item in enumerate(prepared)"
    boundary_call = "_require_report_commit_boundary(pinned, prepared, snapshot)"
    assert commit_loop in source
    assert boundary_call in source
    assert "result.append(" in source
    assert (
        source.index(commit_loop)
        < source.index(boundary_call)
        < source.index("_postverify(pinned, prepared, snapshot)")
    )
    assert "os.unlink(prepared.destination_leaf" not in source
    assert "os.chmod" not in source
    assert "os.rename" not in source
    assert "_link_descriptor_no_overwrite(" in source
    reverify_line = "result = reverify(registration)"
    frame_line = "observed_frame = registration_frame(registration)"
    second_snapshot_line = "second = snapshot_validator(adapter(evidence))"
    publish_line = "publish_core(first, layout, final_authority_check)"
    assert public_source.count(reverify_line) == 1
    assert (
        public_source.index(reverify_line)
        < public_source.index(frame_line)
        < public_source.index(second_snapshot_line)
        < public_source.index(publish_line)
    )
