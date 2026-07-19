from __future__ import annotations

import ast
import copy
import hashlib
import json
import pickle
import struct
from fractions import Fraction
from pathlib import Path
from typing import cast

import numpy as np
import pytest

import falsewake.experiment_002_evidence as evidence
import falsewake.experiment_002_metrics as metrics
from falsewake.experiment_002_data import Experiment002Corpus
from falsewake.experiment_002_metrics import ValidationMetrics
from falsewake.experiment_002_validation import MaterializedValidationPopulation


def _immutable(values: np.ndarray[tuple[int, ...], np.dtype[np.generic]]) -> None:
    values.setflags(write=False)


def _tiny_arrays() -> tuple[
    tuple[bytes, ...],
    evidence.Float32Array,
    evidence.Int64Array,
    evidence.Float32Array,
]:
    identities = (b"a", b"\x00B")
    model_inputs = np.asarray(
        [
            [[0.0, -0.0]],
            [[1.5, -2.25]],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([1, 11], dtype=np.int64)
    logits = np.asarray(
        [
            [
                0.0,
                2.0,
                2.0,
                -1.0,
                -2.0,
                -3.0,
                -4.0,
                -5.0,
                -6.0,
                -7.0,
                -8.0,
                -9.0,
            ],
            [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0],
        ],
        dtype=np.float32,
    )
    _immutable(model_inputs)
    _immutable(labels)
    _immutable(logits)
    return identities, model_inputs, labels, logits


def _tiny_metrics(
    labels: evidence.Int64Array,
    logits: evidence.Float32Array,
) -> ValidationMetrics:
    support = [0] * 12
    for label in labels:
        support[int(label)] += 1
    return metrics._evaluate_validation_logits(
        labels,
        logits,
        layout=metrics._MetricLayout(len(labels), tuple(support)),
    )


def _tiny_capabilities() -> tuple[
    evidence.RegisteredValidationInputs,
    evidence.RegisteredValidationEvidence,
    ValidationMetrics,
]:
    identities, model_inputs, labels, logits = _tiny_arrays()
    input_capability = evidence._bind_validation_inputs(
        identities,
        model_inputs,
        labels,
    )
    metric_result = _tiny_metrics(labels, logits)
    result_capability = evidence._issue_validation_evidence(
        input_capability,
        logits,
        metric_result,
    )
    return input_capability, result_capability, metric_result


def _stdlib_input_vector() -> bytes:
    identities, model_inputs, labels, _ = _tiny_arrays()
    payload = bytearray(b"falsewake-exp002-validation-inputs-v1\0")
    payload.extend(struct.pack("<I", 2))
    for index, identity in enumerate(identities):
        payload.extend(struct.pack("<I", len(identity)))
        payload.extend(identity)
        payload.extend(struct.pack("<B", int(labels[index])))
        row = model_inputs[index].reshape(-1)
        feature_payload = struct.pack("<2f", float(row[0]), float(row[1]))
        payload.extend(struct.pack("<Q", len(feature_payload)))
        payload.extend(feature_payload)
    return bytes(payload)


def _stdlib_prediction_vector(predictions: evidence.Int64Array) -> bytes:
    identities, _, labels, logits = _tiny_arrays()
    payload = bytearray(b"falsewake-exp002-validation-predictions-v1\0")
    payload.extend(struct.pack("<I", 2))
    for index, identity in enumerate(identities):
        payload.extend(struct.pack("<I", len(identity)))
        payload.extend(identity)
        payload.extend(struct.pack("<B", int(labels[index])))
        payload.extend(
            struct.pack(
                "<12f",
                *(float(value) for value in logits[index]),
            )
        )
        payload.extend(struct.pack("<B", int(predictions[index])))
    return bytes(payload)


def test_source_is_pure_and_reconciles_the_frozen_digest_registration() -> None:
    source = Path("src/falsewake/experiment_002_evidence.py").read_text(
        encoding="utf-8"
    )
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
        "hashlib",
        "hmac",
        "numpy",
        "struct",
        "threading",
        "typing",
        "weakref",
    }
    assert not any(name in source for name in ("torch", "random", "secrets"))

    config = json.loads(
        Path("configs/experiment-002-trainer.json").read_text(encoding="utf-8")
    )
    digests = config["digests"]
    assert digests["validation_inputs"]["domain"].encode("ascii") == (
        evidence.VALIDATION_INPUTS_DOMAIN
    )
    assert digests["validation_predictions"]["domain"].encode("ascii") == (
        evidence.VALIDATION_PREDICTIONS_DOMAIN
    )
    assert (
        "UINT64LE_feature_payload_byte_count"
        in (digests["validation_inputs"]["framing"])
    )
    assert (
        "12_raw_little_endian_float32_frame_97_logits"
        in (digests["validation_predictions"]["framing"])
    )
    assert 'struct.Struct("<I")' in source
    assert 'struct.Struct("<Q")' in source


def test_independent_stdlib_golden_vectors_match_exact_module_framing() -> None:
    identities, model_inputs, labels, logits = _tiny_arrays()
    metric_result = _tiny_metrics(labels, logits)
    input_capability = evidence._bind_validation_inputs(
        identities,
        model_inputs,
        labels,
    )
    result_capability = evidence._issue_validation_evidence(
        input_capability,
        logits,
        metric_result,
    )

    input_vector = _stdlib_input_vector()
    prediction_vector = _stdlib_prediction_vector(metric_result.predicted_indices)
    assert len(input_vector) == 87
    assert len(prediction_vector) == 158
    assert hashlib.sha256(input_vector).hexdigest() == (
        "75190d8b7979e5fed30b38f239ba1f25e5ca2e0112995bcc6ddea0fef84f16d6"
    )
    assert hashlib.sha256(prediction_vector).hexdigest() == (
        "559e32cc4bcaa468eb39d71fd250dda8aeacd527adecd66dd29f47236fc453be"
    )
    assert (
        input_capability.validation_inputs_sha256
        == hashlib.sha256(input_vector).hexdigest()
    )
    assert (
        result_capability.validation_predictions_sha256
        == hashlib.sha256(prediction_vector).hexdigest()
    )

    positive_zero_features = model_inputs.copy(order="C")
    positive_zero_features[0, 0, 1] = np.float32(0.0)
    _immutable(positive_zero_features)
    changed = evidence._bind_validation_inputs(
        identities,
        positive_zero_features,
        labels,
    )
    assert changed.validation_inputs_sha256 != (
        input_capability.validation_inputs_sha256
    )


def test_supplied_full_row_stdlib_golden_vectors_match_exact_framing() -> None:
    identity = b"cmd:\0yes"
    feature_values = [0.0] * (40 * 98)
    feature_values[0] = -0.0
    feature_values[1] = 1.5
    feature_values[-1] = -2.25
    feature_payload = struct.pack("<3920f", *feature_values)
    input_vector = b"".join(
        (
            b"falsewake-exp002-validation-inputs-v1\0",
            struct.pack("<I", 1),
            struct.pack("<I", len(identity)),
            identity,
            struct.pack("<B", 3),
            struct.pack("<Q", len(feature_payload)),
            feature_payload,
        )
    )
    logits_values = [0.0, 2.0, 2.0, -0.0, -1.0, -2.0, -3.0, -4.0]
    logits_values.extend([-5.0, -6.0, -7.0, -8.0])
    prediction_vector = b"".join(
        (
            b"falsewake-exp002-validation-predictions-v1\0",
            struct.pack("<I", 1),
            struct.pack("<I", len(identity)),
            identity,
            struct.pack("<B", 3),
            struct.pack("<12f", *logits_values),
            struct.pack("<B", 1),
        )
    )
    assert len(input_vector) == 15_743
    assert len(prediction_vector) == 109
    assert hashlib.sha256(input_vector).hexdigest() == (
        "99200d41f0fe9a1bdbf85741856b0518d855110843b9d29931ccce8cf71bd458"
    )
    assert hashlib.sha256(prediction_vector).hexdigest() == (
        "da4de86a80e981f7aa93e97386c1ba5f38933df04315a01edd594c4ed083b357"
    )

    model_inputs = (
        np.asarray(feature_values, dtype=np.float32).reshape(1, 40, 98).copy(order="C")
    )
    labels = np.asarray([3], dtype=np.int64)
    logits = np.asarray([logits_values], dtype=np.float32)
    _immutable(model_inputs)
    _immutable(labels)
    _immutable(logits)
    input_capability = evidence._bind_validation_inputs(
        (identity,),
        model_inputs,
        labels,
    )
    metric_result = _tiny_metrics(labels, logits)
    result_capability = evidence._issue_validation_evidence(
        input_capability,
        logits,
        metric_result,
    )
    assert metric_result.predicted_indices.tolist() == [1]
    assert (
        input_capability.validation_inputs_sha256
        == hashlib.sha256(input_vector).hexdigest()
    )
    assert (
        result_capability.validation_predictions_sha256
        == hashlib.sha256(prediction_vector).hexdigest()
    )


def test_private_tiny_route_binds_metrics_without_registered_authority() -> None:
    inputs, result, metric_result = _tiny_capabilities()
    evidence._verify_synthetic_validation_inputs(inputs)
    evidence._verify_synthetic_validation_evidence(result)

    assert inputs.example_count == 2
    assert result.example_count == 2
    assert result.validation_inputs_sha256 == inputs.validation_inputs_sha256
    assert result.cross_entropy_hex == "0x1.b5d4d880758d6p-1"
    assert result.macro_f1 == Fraction(1, 12)
    np.testing.assert_array_equal(
        result.predicted_indices,
        np.asarray([1, 0], dtype=np.int64),
    )
    expected_confusion = np.zeros((12, 12), dtype=np.int64)
    expected_confusion[1, 1] = 1
    expected_confusion[11, 0] = 1
    np.testing.assert_array_equal(result.confusion_matrix, expected_confusion)
    assert result.cross_entropy.tobytes() == metric_result.cross_entropy.tobytes()
    assert not result.predicted_indices.flags.writeable
    assert result.predicted_indices.flags.owndata
    assert not result.confusion_matrix.flags.writeable
    assert result.confusion_matrix.flags.owndata

    with pytest.raises(
        evidence.Experiment002EvidenceError,
        match="registered route",
    ):
        evidence.verify_registered_validation_inputs(inputs)
    with pytest.raises(
        evidence.Experiment002EvidenceError,
        match="registered route",
    ):
        evidence.verify_registered_validation_evidence(result)
    with pytest.raises(
        evidence.Experiment002EvidenceError,
        match="registered route",
    ):
        evidence._registered_validation_evidence_snapshot(result)


def test_public_input_entrypoint_only_routes_registered_issuer_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = cast(Experiment002Corpus, object())
    population = cast(MaterializedValidationPopulation, object())
    sentinel = cast(evidence.RegisteredValidationInputs, object())
    calls: list[str] = []

    def verify_materialized(
        observed_corpus: Experiment002Corpus,
        observed_population: MaterializedValidationPopulation,
    ) -> None:
        assert observed_corpus is corpus
        assert observed_population is population
        calls.append("materialized")

    def bind_internal(
        observed_corpus: Experiment002Corpus,
        observed_population: MaterializedValidationPopulation,
    ) -> evidence.RegisteredValidationInputs:
        assert observed_corpus is corpus
        assert observed_population is population
        calls.append("bind")
        return sentinel

    def verify_inputs(
        observed: evidence.RegisteredValidationInputs,
    ) -> None:
        assert observed is sentinel
        calls.append("inputs")

    monkeypatch.setattr(
        evidence,
        "verify_registered_materialized_validation",
        verify_materialized,
    )
    monkeypatch.setattr(evidence, "_bind_registered_validation_inputs", bind_internal)
    monkeypatch.setattr(
        evidence,
        "verify_registered_validation_inputs",
        verify_inputs,
    )

    observed = evidence.bind_registered_validation_inputs(corpus, population)

    assert observed is sentinel
    assert calls == ["materialized", "bind", "materialized", "inputs"]


def test_public_evidence_entrypoint_only_routes_registered_issuer_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = cast(evidence.RegisteredValidationInputs, object())
    logits = cast(evidence.Float32Array, object())
    metric_result = cast(ValidationMetrics, object())
    sentinel = cast(evidence.RegisteredValidationEvidence, object())
    calls: list[str] = []

    def verify_inputs(
        observed: evidence.RegisteredValidationInputs,
    ) -> None:
        assert observed is inputs
        calls.append("inputs")

    def verify_layout(observed: ValidationMetrics) -> None:
        assert observed is metric_result
        calls.append("layout")

    def issue_internal(
        observed_inputs: evidence.RegisteredValidationInputs,
        observed_logits: evidence.Float32Array,
        observed_metrics: ValidationMetrics,
    ) -> evidence.RegisteredValidationEvidence:
        assert observed_inputs is inputs
        assert observed_logits is logits
        assert observed_metrics is metric_result
        calls.append("issue")
        return sentinel

    def verify_result(observed: evidence.RegisteredValidationEvidence) -> None:
        assert observed is sentinel
        calls.append("evidence")

    monkeypatch.setattr(
        evidence,
        "verify_registered_validation_inputs",
        verify_inputs,
    )
    monkeypatch.setattr(evidence, "verify_validation_layout_metrics", verify_layout)
    monkeypatch.setattr(
        evidence,
        "_issue_registered_validation_evidence",
        issue_internal,
    )
    monkeypatch.setattr(
        evidence,
        "verify_registered_validation_evidence",
        verify_result,
    )

    observed = evidence.issue_registered_validation_evidence(
        inputs,
        logits,
        metric_result,
    )

    assert observed is sentinel
    assert calls == [
        "inputs",
        "layout",
        "issue",
        "inputs",
        "layout",
        "evidence",
    ]


def test_raw_metrics_and_synthetic_inputs_cannot_cross_registered_route() -> None:
    identities, model_inputs, labels, logits = _tiny_arrays()
    input_capability = evidence._bind_validation_inputs(
        identities,
        model_inputs,
        labels,
    )
    metric_result = _tiny_metrics(labels, logits)

    with pytest.raises(
        evidence.Experiment002EvidenceError,
        match="registered route",
    ):
        evidence.issue_registered_validation_evidence(
            input_capability,
            logits,
            metric_result,
        )
    with pytest.raises(TypeError, match="RegisteredValidationInputs"):
        evidence.issue_registered_validation_evidence(
            cast(evidence.RegisteredValidationInputs, metric_result),
            logits,
            metric_result,
        )
    with pytest.raises(
        evidence.Experiment002EvidenceError,
        match="requires registered validation inputs",
    ):
        evidence._issue_validation_evidence_capability(
            input_capability,
            logits,
            metric_result,
            route_marker=evidence._REGISTERED_EVIDENCE_ROUTE_MARKER,
        )


def test_capabilities_are_issuer_only_and_fail_closed_after_slot_changes() -> None:
    with pytest.raises(TypeError, match="issued by this module"):
        evidence.RegisteredValidationInputs()
    with pytest.raises(TypeError, match="issued by this module"):
        evidence.RegisteredValidationEvidence()

    forged_inputs = object.__new__(evidence.RegisteredValidationInputs)
    object.__setattr__(forged_inputs, "_example_count", 2)
    object.__setattr__(forged_inputs, "_validation_inputs_sha256", "0" * 64)
    object.__setattr__(
        forged_inputs,
        "_route_marker",
        evidence._REGISTERED_INPUT_ROUTE_MARKER,
    )
    with pytest.raises(
        evidence.Experiment002EvidenceError,
        match="not issued by this process",
    ):
        _ = forged_inputs.validation_inputs_sha256

    forged_evidence = object.__new__(evidence.RegisteredValidationEvidence)
    object.__setattr__(forged_evidence, "_example_count", 2)
    object.__setattr__(forged_evidence, "_validation_inputs_sha256", "0" * 64)
    object.__setattr__(
        forged_evidence,
        "_validation_predictions_sha256",
        "0" * 64,
    )
    object.__setattr__(
        forged_evidence,
        "_route_marker",
        evidence._REGISTERED_EVIDENCE_ROUTE_MARKER,
    )
    with pytest.raises(
        evidence.Experiment002EvidenceError,
        match="not issued by this process",
    ):
        _ = forged_evidence.validation_predictions_sha256

    inputs, result, _ = _tiny_capabilities()
    for clone in (
        copy.copy(inputs),
        copy.deepcopy(inputs),
        pickle.loads(pickle.dumps(inputs)),
    ):
        with pytest.raises(
            evidence.Experiment002EvidenceError,
            match="not issued by this process",
        ):
            _ = clone.example_count
    for clone in (
        copy.copy(result),
        copy.deepcopy(result),
        pickle.loads(pickle.dumps(result)),
    ):
        with pytest.raises(
            evidence.Experiment002EvidenceError,
            match="not issued by this process",
        ):
            _ = clone.example_count

    object.__setattr__(inputs, "_example_count", 3)
    with pytest.raises(
        evidence.Experiment002EvidenceError,
        match="changed after issuance",
    ):
        _ = inputs.example_count

    object.__setattr__(result, "_validation_predictions_sha256", "0" * 64)
    with pytest.raises(
        evidence.Experiment002EvidenceError,
        match="changed after issuance",
    ):
        _ = result.validation_predictions_sha256


def test_input_contracts_reject_noncanonical_arrays_and_identity_types() -> None:
    identities, model_inputs, labels, _ = _tiny_arrays()

    writable = model_inputs.copy(order="C")
    with pytest.raises(
        evidence.Experiment002EvidenceError,
        match="read-only",
    ):
        evidence._bind_validation_inputs(identities, writable, labels)

    nonfinite = model_inputs.copy(order="C")
    nonfinite[0, 0, 0] = np.float32(np.nan)
    _immutable(nonfinite)
    with pytest.raises(
        evidence.Experiment002EvidenceError,
        match="non-finite",
    ):
        evidence._bind_validation_inputs(identities, nonfinite, labels)

    wrong_labels = np.asarray([1, 12], dtype=np.int64)
    _immutable(wrong_labels)
    with pytest.raises(
        evidence.Experiment002EvidenceError,
        match="outside class order",
    ):
        evidence._bind_validation_inputs(identities, model_inputs, wrong_labels)

    class BytesSubclass(bytes):
        pass

    with pytest.raises(
        evidence.Experiment002EvidenceError,
        match="concrete types",
    ):
        evidence._bind_validation_inputs(
            (BytesSubclass(b"a"), b"b"),
            model_inputs,
            labels,
        )

    wrong_rank = np.asarray([[0.0], [1.0]], dtype=np.float32)
    _immutable(wrong_rank)
    with pytest.raises(
        evidence.Experiment002EvidenceError,
        match="dtype or rank",
    ):
        evidence._bind_validation_inputs(identities, wrong_rank, labels)

    float64_inputs = np.asarray(model_inputs, dtype=np.float64, order="C")
    _immutable(float64_inputs)
    with pytest.raises(
        evidence.Experiment002EvidenceError,
        match="dtype or rank",
    ):
        evidence._bind_validation_inputs(
            identities,
            cast(evidence.Float32Array, float64_inputs),
            labels,
        )

    int32_labels = np.asarray(labels, dtype=np.int32, order="C")
    _immutable(int32_labels)
    with pytest.raises(
        evidence.Experiment002EvidenceError,
        match="shape or dtype",
    ):
        evidence._bind_validation_inputs(
            identities,
            model_inputs,
            cast(evidence.Int64Array, int32_labels),
        )

    feature_backing = np.zeros((2, 1, 4), dtype=np.float32, order="C")
    noncontiguous_inputs = feature_backing[:, :, ::2]
    _immutable(noncontiguous_inputs)
    assert not noncontiguous_inputs.flags.c_contiguous
    with pytest.raises(
        evidence.Experiment002EvidenceError,
        match="C-contiguous",
    ):
        evidence._bind_validation_inputs(
            identities,
            noncontiguous_inputs,
            labels,
        )


def test_logits_contract_rejects_wrong_dtype_and_noncontiguous_storage() -> None:
    identities, model_inputs, labels, logits = _tiny_arrays()
    input_capability = evidence._bind_validation_inputs(
        identities,
        model_inputs,
        labels,
    )
    metric_result = _tiny_metrics(labels, logits)

    float64_logits = np.asarray(logits, dtype=np.float64, order="C")
    _immutable(float64_logits)
    with pytest.raises(
        evidence.Experiment002EvidenceError,
        match="shape or dtype",
    ):
        evidence._issue_validation_evidence(
            input_capability,
            cast(evidence.Float32Array, float64_logits),
            metric_result,
        )

    logits_backing = np.zeros((2, 24), dtype=np.float32, order="C")
    noncontiguous_logits = logits_backing[:, ::2]
    _immutable(noncontiguous_logits)
    assert noncontiguous_logits.shape == (2, 12)
    assert not noncontiguous_logits.flags.c_contiguous
    with pytest.raises(
        evidence.Experiment002EvidenceError,
        match="C-contiguous",
    ):
        evidence._issue_validation_evidence(
            input_capability,
            noncontiguous_logits,
            metric_result,
        )


def test_row_pair_swaps_change_both_registered_binary_digests() -> None:
    identities, model_inputs, labels, logits = _tiny_arrays()
    original_inputs = evidence._bind_validation_inputs(
        identities,
        model_inputs,
        labels,
    )
    original_metrics = _tiny_metrics(labels, logits)
    original_evidence = evidence._issue_validation_evidence(
        original_inputs,
        logits,
        original_metrics,
    )

    swapped_identities = tuple(reversed(identities))
    swapped_inputs_array = np.array(model_inputs[::-1], dtype=np.float32, order="C")
    swapped_labels = np.array(labels[::-1], dtype=np.int64, order="C")
    swapped_logits = np.array(logits[::-1], dtype=np.float32, order="C")
    _immutable(swapped_inputs_array)
    _immutable(swapped_labels)
    _immutable(swapped_logits)
    swapped_inputs = evidence._bind_validation_inputs(
        swapped_identities,
        swapped_inputs_array,
        swapped_labels,
    )
    swapped_metrics = _tiny_metrics(swapped_labels, swapped_logits)
    swapped_evidence = evidence._issue_validation_evidence(
        swapped_inputs,
        swapped_logits,
        swapped_metrics,
    )

    assert swapped_inputs.validation_inputs_sha256 != (
        original_inputs.validation_inputs_sha256
    )
    assert swapped_evidence.validation_predictions_sha256 != (
        original_evidence.validation_predictions_sha256
    )

    identity_only_swap = evidence._bind_validation_inputs(
        swapped_identities,
        model_inputs,
        labels,
    )
    identity_only_evidence = evidence._issue_validation_evidence(
        identity_only_swap,
        logits,
        original_metrics,
    )
    assert identity_only_swap.validation_inputs_sha256 != (
        original_inputs.validation_inputs_sha256
    )
    assert identity_only_evidence.validation_predictions_sha256 != (
        original_evidence.validation_predictions_sha256
    )


def test_exact_metric_input_bytes_are_required_even_when_predictions_match() -> None:
    identities, model_inputs, labels, logits = _tiny_arrays()
    input_capability = evidence._bind_validation_inputs(
        identities,
        model_inputs,
        labels,
    )
    metric_result = _tiny_metrics(labels, logits)

    changed_logits = logits.copy(order="C")
    changed_logits[0, 0] = np.float32(-0.0)
    assert np.argmax(changed_logits, axis=1).tolist() == [1, 0]
    _immutable(changed_logits)
    with pytest.raises(
        metrics.Experiment002MetricsError,
        match="differ from the issued computation",
    ):
        evidence._issue_validation_evidence(
            input_capability,
            changed_logits,
            metric_result,
        )


def test_input_binding_detects_mutation_during_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identities, model_inputs, labels, _ = _tiny_arrays()
    original_digest = evidence._validation_inputs_sha256
    mutated = False

    def racing_digest(
        observed_identities: tuple[bytes, ...],
        observed_labels: evidence.Int64Array,
        observed_features: evidence.Float32Array,
    ) -> str:
        nonlocal mutated
        if not mutated:
            mutated = True
            model_inputs.setflags(write=True)
            model_inputs[0, 0, 0] = np.float32(9.0)
            model_inputs.setflags(write=False)
        return original_digest(
            observed_identities,
            observed_labels,
            observed_features,
        )

    monkeypatch.setattr(evidence, "_validation_inputs_sha256", racing_digest)
    with pytest.raises(
        evidence.Experiment002EvidenceError,
        match="changed while taking the binding snapshot",
    ):
        evidence._bind_validation_inputs(identities, model_inputs, labels)


def test_evidence_uses_an_exact_logits_snapshot_during_digest_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identities, model_inputs, labels, logits = _tiny_arrays()
    input_capability = evidence._bind_validation_inputs(
        identities,
        model_inputs,
        labels,
    )
    metric_result = _tiny_metrics(labels, logits)
    stable_prediction_digest = hashlib.sha256(
        _stdlib_prediction_vector(metric_result.predicted_indices)
    ).hexdigest()
    original_digest = evidence._validation_predictions_sha256
    mutated = False

    def racing_digest(
        observed_identities: tuple[bytes, ...],
        observed_labels: evidence.Int64Array,
        observed_logits: evidence.Float32Array,
        observed_predictions: evidence.Int64Array,
    ) -> str:
        nonlocal mutated
        if not mutated:
            mutated = True
            logits.setflags(write=True)
            logits[0, 0] = np.float32(7.0)
            logits.setflags(write=False)
        return original_digest(
            observed_identities,
            observed_labels,
            observed_logits,
            observed_predictions,
        )

    monkeypatch.setattr(evidence, "_validation_predictions_sha256", racing_digest)
    result = evidence._issue_validation_evidence(
        input_capability,
        logits,
        metric_result,
    )

    evidence._verify_synthetic_validation_evidence(result)
    assert result.validation_predictions_sha256 == stable_prediction_digest
    assert float(logits[0, 0]) == 7.0


def test_evidence_detects_input_capability_mutation_during_issuance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identities, model_inputs, labels, logits = _tiny_arrays()
    input_capability = evidence._bind_validation_inputs(
        identities,
        model_inputs,
        labels,
    )
    metric_result = _tiny_metrics(labels, logits)
    original_digest = evidence._validation_predictions_sha256
    mutated = False

    def racing_digest(
        observed_identities: tuple[bytes, ...],
        observed_labels: evidence.Int64Array,
        observed_logits: evidence.Float32Array,
        observed_predictions: evidence.Int64Array,
    ) -> str:
        nonlocal mutated
        if not mutated:
            mutated = True
            object.__setattr__(
                input_capability,
                "_validation_inputs_sha256",
                "0" * 64,
            )
        return original_digest(
            observed_identities,
            observed_labels,
            observed_logits,
            observed_predictions,
        )

    monkeypatch.setattr(evidence, "_validation_predictions_sha256", racing_digest)
    with pytest.raises(
        evidence.Experiment002EvidenceError,
        match="changed after issuance",
    ):
        evidence._issue_validation_evidence(
            input_capability,
            logits,
            metric_result,
        )
