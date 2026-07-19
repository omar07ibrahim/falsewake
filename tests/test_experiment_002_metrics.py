from __future__ import annotations

import ast
import copy
import json
import pickle
from fractions import Fraction
from pathlib import Path
from typing import cast

import numpy as np
import pytest

import falsewake.experiment_002_metrics as metrics
from falsewake.experiment_002_data import CLASS_ORDER, VALIDATION_EXAMPLE_COUNT
from falsewake.experiment_002_metrics import (
    Experiment002MetricsError,
    ValidationMetrics,
)
from falsewake.experiment_002_validation import VALIDATION_CLASS_SUPPORT


def _immutable(array: np.ndarray[tuple[int, ...], np.dtype[np.generic]]) -> None:
    array.setflags(write=False)


def _tiny_inputs() -> tuple[
    np.ndarray[tuple[int], np.dtype[np.int64]],
    np.ndarray[tuple[int, int], np.dtype[np.float32]],
    metrics._MetricLayout,
]:
    labels = np.asarray([0, 1, 2, 11], dtype=np.int64)
    logits = np.asarray(
        [
            [0.0] * 12,
            [-3.0, -2.0, 4.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0],
            [0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [
                80.0,
                -80.0,
                -80.0,
                -80.0,
                -80.0,
                -80.0,
                -80.0,
                -80.0,
                -80.0,
                -80.0,
                -80.0,
                -80.0,
            ],
        ],
        dtype=np.float32,
    )
    _immutable(labels)
    _immutable(logits)
    support = [0] * 12
    for label in labels:
        support[int(label)] += 1
    return labels, logits, metrics._MetricLayout(4, tuple(support))


def _frozen_layout_confusion() -> np.ndarray[tuple[int, int], np.dtype[np.int64]]:
    confusion = np.zeros((12, 12), dtype=np.int64)
    for class_index, support in enumerate(VALIDATION_CLASS_SUPPORT):
        confusion[class_index, class_index] = support
    return confusion


def test_metric_module_is_pure_and_reconciles_the_frozen_registration() -> None:
    source = Path("src/falsewake/experiment_002_metrics.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(
                alias.name.split(".", maxsplit=1)[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".", maxsplit=1)[0])

    assert imported_roots == {
        "__future__",
        "dataclasses",
        "falsewake",
        "fractions",
        "numpy",
        "threading",
        "typing",
        "weakref",
    }
    assert not any(name in source for name in ("torch", "random", "secrets"))
    assert "np.argmax(logits_snapshot, axis=1)" in source
    assert "np.sum" not in source

    config = json.loads(
        Path("configs/experiment-002-trainer.json").read_text(encoding="utf-8")
    )
    assert tuple(config["metrics"]["class_order"]) == CLASS_ORDER
    assert tuple(config["metrics"]["support"]) == VALIDATION_CLASS_SUPPORT
    assert config["metrics"]["target_population"] == metrics.TARGET_POPULATION
    assert config["validation_execution"]["population_count"] == (
        metrics._FROZEN_VALIDATION_LAYOUT.example_count
    )


def test_synthetic_predictions_use_exact_scalar_order_and_first_tie_index() -> None:
    labels, logits, layout = _tiny_inputs()
    observed = metrics._evaluate_validation_logits(
        labels,
        logits,
        layout=layout,
    )

    np.testing.assert_array_equal(
        observed.predicted_indices,
        np.asarray([0, 2, 1, 0], dtype=np.int64),
    )
    expected_confusion = np.zeros((12, 12), dtype=np.int64)
    expected_confusion[0, 0] = 1
    expected_confusion[1, 2] = 1
    expected_confusion[2, 1] = 1
    expected_confusion[11, 0] = 1
    np.testing.assert_array_equal(observed.confusion_matrix, expected_confusion)
    assert observed.macro_f1 == Fraction(1, 18)
    assert observed.cross_entropy_hex == "0x1.5365c69d5f7a4p+5"
    assert not observed.confusion_matrix.flags.writeable
    assert observed.confusion_matrix.flags.owndata
    assert observed.confusion_matrix.flags.c_contiguous
    assert not observed.predicted_indices.flags.writeable
    assert observed.predicted_indices.flags.owndata
    assert observed.predicted_indices.flags.c_contiguous


def test_macro_f1_uses_zero_for_empty_classes_and_reduced_fractions() -> None:
    empty = np.zeros((12, 12), dtype=np.int64)
    assert metrics.macro_f1_from_confusion(empty) == Fraction(0, 1)

    confusion = np.zeros((12, 12), dtype=np.int64)
    confusion[0, 0] = 3
    confusion[0, 1] = 1
    confusion[1, 0] = 2
    confusion[1, 1] = 4
    expected = (Fraction(6, 9) + Fraction(8, 11)) / 12
    assert metrics.macro_f1_from_confusion(confusion) == expected


def test_frozen_gate_predicates_use_exact_integer_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    perfect = _frozen_layout_confusion()
    perfect_gates = metrics.evaluate_validation_gate_predicates(perfect)
    assert perfect_gates.passed

    target_recall = perfect.copy(order="C")
    target_recall[0] = 0
    target_recall[0, 0] = 277
    target_recall[0, 10] = VALIDATION_CLASS_SUPPORT[0] - 277
    assert not metrics.evaluate_validation_gate_predicates(
        target_recall
    ).every_target_recall
    target_recall[0, 0] += 1
    target_recall[0, 10] -= 1
    assert metrics.evaluate_validation_gate_predicates(
        target_recall
    ).every_target_recall

    target_accuracy = perfect.copy(order="C")
    errors = [56, 56, 56, 56, 56, 55, 55, 55, 55, 55]
    for class_index, error_count in enumerate(errors):
        target_accuracy[class_index, class_index] -= error_count
        target_accuracy[class_index, 10] += error_count
    assert sum(int(target_accuracy[index, index]) for index in range(10)) == 3_148
    assert metrics.evaluate_validation_gate_predicates(target_accuracy).target_accuracy
    target_accuracy[9, 9] -= 1
    target_accuracy[9, 10] += 1
    assert not metrics.evaluate_validation_gate_predicates(
        target_accuracy
    ).target_accuracy

    unknown_rate = perfect.copy(order="C")
    unknown_rate[10, 10] -= 1_255
    unknown_rate[10, 0] += 1_255
    assert metrics.evaluate_validation_gate_predicates(unknown_rate).unknown_target_rate
    unknown_rate[10, 10] -= 1
    unknown_rate[10, 0] += 1
    assert not metrics.evaluate_validation_gate_predicates(
        unknown_rate
    ).unknown_target_rate

    silence_rate = perfect.copy(order="C")
    silence_rate[11, 11] -= 30
    silence_rate[11, 0] += 30
    assert metrics.evaluate_validation_gate_predicates(silence_rate).silence_target_rate
    silence_rate[11, 11] -= 1
    silence_rate[11, 0] += 1
    assert not metrics.evaluate_validation_gate_predicates(
        silence_rate
    ).silence_target_rate

    macro_failure = np.zeros((12, 12), dtype=np.int64)
    for class_index, support in enumerate(VALIDATION_CLASS_SUPPORT):
        macro_failure[class_index, 0] = support
    assert not metrics.evaluate_validation_gate_predicates(macro_failure).macro_f1

    monkeypatch.setattr(
        metrics,
        "macro_f1_from_confusion",
        lambda value: Fraction(4, 5),
    )
    assert metrics.evaluate_validation_gate_predicates(perfect).macro_f1
    monkeypatch.setattr(
        metrics,
        "macro_f1_from_confusion",
        lambda value: Fraction(4, 5) - Fraction(1, 10**30),
    )
    assert not metrics.evaluate_validation_gate_predicates(perfect).macro_f1


def test_public_entrypoint_binds_only_the_frozen_layout_without_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = np.repeat(
        np.arange(12, dtype=np.int64),
        np.asarray(VALIDATION_CLASS_SUPPORT, dtype=np.int64),
    )
    logits = np.zeros((VALIDATION_EXAMPLE_COUNT, 12), dtype=np.float32)
    _immutable(labels)
    _immutable(logits)
    sentinel = cast(ValidationMetrics, object())
    calls = 0

    def compute(
        observed_labels: metrics.Int64Array,
        observed_logits: metrics.Float32Array,
        layout: metrics._MetricLayout,
        *,
        layout_route_marker: object | None = None,
    ) -> ValidationMetrics:
        nonlocal calls
        calls += 1
        assert observed_labels is labels
        assert observed_logits is logits
        assert layout is metrics._FROZEN_VALIDATION_LAYOUT
        assert layout_route_marker is metrics._LAYOUT_ROUTE_MARKER
        return sentinel

    monkeypatch.setattr(metrics, "_compute_validation_metrics", compute)
    observed = metrics.evaluate_validation_layout_logits(labels, logits)

    assert observed is sentinel
    assert calls == 1


def test_input_contracts_fail_before_metric_state_is_created(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels, logits, layout = _tiny_inputs()
    calls = 0

    def compute(
        observed_labels: metrics.Int64Array,
        observed_logits: metrics.Float32Array,
        observed_layout: metrics._MetricLayout,
    ) -> ValidationMetrics:
        nonlocal calls
        calls += 1
        raise AssertionError("invalid inputs reached metric computation")

    monkeypatch.setattr(metrics, "_compute_validation_metrics", compute)
    bad_labels = (
        labels.astype(np.int32),
        labels[:-1].copy(),
        np.asarray([0, 1, 2, 12], dtype=np.int64),
        np.asarray([0, 1, 1, 11], dtype=np.int64),
        cast(np.ndarray[tuple[int], np.dtype[np.int64]], [0, 1, 2, 11]),
    )
    for label_candidate in bad_labels:
        if isinstance(label_candidate, np.ndarray):
            label_candidate.setflags(write=False)
        with pytest.raises((TypeError, Experiment002MetricsError)):
            metrics._evaluate_validation_logits(
                cast(metrics.Int64Array, label_candidate),
                logits,
                layout=layout,
            )

    writable_labels = np.asarray([0, 1, 2, 11], dtype=np.int64)
    with pytest.raises(Experiment002MetricsError, match="read-only"):
        metrics._evaluate_validation_logits(
            writable_labels,
            logits,
            layout=layout,
        )
    writable_logits = logits.copy(order="C")
    with pytest.raises(Experiment002MetricsError, match="read-only"):
        metrics._evaluate_validation_logits(
            labels,
            writable_logits,
            layout=layout,
        )

    bad_logits = (
        logits.astype(np.float64),
        logits[:, :11].copy(),
        np.full((4, 12), np.nan, dtype=np.float32),
        np.zeros((4, 24), dtype=np.float32)[:, ::2],
        cast(np.ndarray[tuple[int, int], np.dtype[np.float32]], "not-an-array"),
    )
    for logit_candidate in bad_logits:
        if isinstance(logit_candidate, np.ndarray):
            logit_candidate.setflags(write=False)
        with pytest.raises((TypeError, Experiment002MetricsError)):
            metrics._evaluate_validation_logits(
                labels,
                cast(metrics.Float32Array, logit_candidate),
                layout=layout,
            )
    assert calls == 0


def test_results_are_issuer_only_and_return_fresh_immutable_snapshots() -> None:
    labels, logits, layout = _tiny_inputs()
    observed = metrics._evaluate_validation_logits(labels, logits, layout=layout)
    first_confusion = observed.confusion_matrix
    second_confusion = observed.confusion_matrix
    first_predictions = observed.predicted_indices
    second_predictions = observed.predicted_indices

    np.testing.assert_array_equal(first_confusion, second_confusion)
    np.testing.assert_array_equal(first_predictions, second_predictions)
    assert not np.shares_memory(first_confusion, second_confusion)
    assert not np.shares_memory(first_predictions, second_predictions)
    assert first_confusion.flags.owndata and first_confusion.flags.c_contiguous
    assert first_predictions.flags.owndata and first_predictions.flags.c_contiguous
    assert not first_confusion.flags.writeable
    assert not first_predictions.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        first_confusion[0, 0] = 0
    with pytest.raises(ValueError, match="read-only"):
        first_predictions[0] = 11
    metrics.verify_validation_metrics_inputs(observed, labels, logits)
    changed_logits = logits.copy(order="C")
    changed_logits[0, 0] = np.float32(-0.0)
    changed_logits.setflags(write=False)
    assert np.array_equal(changed_logits, logits)
    assert changed_logits.tobytes(order="C") != logits.tobytes(order="C")
    with pytest.raises(Experiment002MetricsError, match="inputs differ"):
        metrics.verify_validation_metrics_inputs(observed, labels, changed_logits)
    changed_labels = labels.copy(order="C")
    changed_labels[[0, 1]] = changed_labels[[1, 0]]
    changed_labels.setflags(write=False)
    with pytest.raises(Experiment002MetricsError, match="inputs differ"):
        metrics.verify_validation_metrics_inputs(observed, changed_labels, logits)
    with pytest.raises(Experiment002MetricsError, match="frozen layout route"):
        metrics.verify_validation_layout_metrics(observed)

    with pytest.raises(TypeError, match="issued"):
        ValidationMetrics()
    clones = (
        copy.copy(observed),
        copy.deepcopy(observed),
        pickle.loads(pickle.dumps(observed)),
    )
    for clone in clones:
        with pytest.raises(Experiment002MetricsError, match="not issued"):
            _ = clone.cross_entropy

    forged = object.__new__(ValidationMetrics)
    pair_swapped = observed.predicted_indices.copy(order="C")
    pair_swapped[[0, 1]] = pair_swapped[[1, 0]]
    object.__setattr__(forged, "_confusion_matrix", observed.confusion_matrix)
    object.__setattr__(forged, "_cross_entropy", np.float64(0.0))
    object.__setattr__(forged, "_macro_f1", observed.macro_f1)
    object.__setattr__(forged, "_predicted_indices", pair_swapped)
    with pytest.raises(Experiment002MetricsError, match="not issued"):
        _ = forged.cross_entropy

    object.__setattr__(observed, "_cross_entropy", np.float64(0.0))
    with pytest.raises(Experiment002MetricsError, match="changed"):
        _ = observed.cross_entropy

    class EqualityLiar:
        def __ne__(self, other: object) -> bool:
            return False

    layout_tampered = metrics._evaluate_validation_logits(labels, logits, layout=layout)
    object.__setattr__(
        layout_tampered,
        "_layout",
        cast(metrics._MetricLayout, EqualityLiar()),
    )
    with pytest.raises(Experiment002MetricsError, match="changed"):
        _ = layout_tampered.macro_f1

    macro_tampered = metrics._evaluate_validation_logits(labels, logits, layout=layout)
    object.__setattr__(
        macro_tampered,
        "_macro_f1",
        cast(Fraction, EqualityLiar()),
    )
    with pytest.raises(Experiment002MetricsError, match="changed"):
        _ = macro_tampered.macro_f1


def test_metric_computation_uses_a_private_pre_scoring_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels, logits, layout = _tiny_inputs()
    reference = metrics._evaluate_validation_logits(labels, logits, layout=layout)
    original_snapshot = metrics._snapshot_float32

    def snapshot(values: metrics.Float32Array) -> metrics.Float32Array:
        captured = original_snapshot(values)
        if values is logits:
            values.setflags(write=True)
            values[0, 0] = np.float32(100.0)
            values.setflags(write=False)
        return captured

    monkeypatch.setattr(metrics, "_snapshot_float32", snapshot)
    observed = metrics._evaluate_validation_logits(labels, logits, layout=layout)

    assert observed.cross_entropy_hex == reference.cross_entropy_hex
    np.testing.assert_array_equal(
        observed.predicted_indices,
        reference.predicted_indices,
    )
    with pytest.raises(Experiment002MetricsError, match="inputs differ"):
        metrics.verify_validation_metrics_inputs(observed, labels, logits)


def test_metric_computation_uses_the_private_layout_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels, logits, layout = _tiny_inputs()
    reference = metrics._evaluate_validation_logits(labels, logits, layout=layout)
    real_require = metrics._require_metric_inputs
    calls = 0

    def require(
        observed_labels: metrics.Int64Array,
        observed_logits: metrics.Float32Array,
        observed_layout: metrics._MetricLayout,
    ) -> None:
        nonlocal calls
        calls += 1
        real_require(observed_labels, observed_logits, observed_layout)
        if calls == 2:
            object.__setattr__(layout, "example_count", 1)
            object.__setattr__(layout, "class_support", (1,) + (0,) * 11)

    monkeypatch.setattr(metrics, "_require_metric_inputs", require)
    observed = metrics._evaluate_validation_logits(labels, logits, layout=layout)

    assert observed.cross_entropy_hex == reference.cross_entropy_hex
    np.testing.assert_array_equal(
        observed.confusion_matrix,
        reference.confusion_matrix,
    )


def test_layout_and_confusion_types_fail_closed() -> None:
    with pytest.raises(TypeError, match="integer"):
        metrics._MetricLayout(cast(int, True), (1,) + (0,) * 11)
    with pytest.raises(Experiment002MetricsError, match="shape"):
        metrics._MetricLayout(1, (1,))
    with pytest.raises(Experiment002MetricsError, match="example_count"):
        metrics._MetricLayout(2, (1,) + (0,) * 11)

    valid = np.zeros((12, 12), dtype=np.int64)
    invalid_confusions = (
        valid.astype(np.int32),
        valid[:11],
        np.full((12, 12), -1, dtype=np.int64),
        np.zeros((12, 24), dtype=np.int64)[:, ::2],
        cast(np.ndarray[tuple[int, int], np.dtype[np.int64]], object()),
    )
    for confusion in invalid_confusions:
        with pytest.raises((TypeError, Experiment002MetricsError)):
            metrics.macro_f1_from_confusion(cast(metrics.Int64Array, confusion))

    wrong_support = _frozen_layout_confusion()
    wrong_support[0, 0] -= 1
    with pytest.raises(Experiment002MetricsError, match="support"):
        metrics.evaluate_validation_gate_predicates(wrong_support)

    gates = metrics.ValidationGateResults(True, True, True, True, True)
    assert gates.passed
    with pytest.raises(TypeError, match="bool"):
        metrics.ValidationGateResults(cast(bool, 1), True, True, True, True)
