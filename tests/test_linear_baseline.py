from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from falsewake.baseline_data import MANIFEST_SHA256, ClipExample
from falsewake.linear_baseline import (
    CLASS_ORDER,
    LinearBaselineError,
    fit_linear_baseline,
    write_baseline_run,
)
from falsewake.speech_commands import Split


def _experiment() -> tuple[np.ndarray, tuple[ClipExample, ...]]:
    rows: list[np.ndarray] = []
    examples: list[ClipExample] = []
    splits: tuple[Split, ...] = ("test", "train", "validation")
    for split_index, split in enumerate(splits):
        repeats = 2 if split == "train" else 1
        for class_index, label in enumerate(CLASS_ORDER):
            for repeat in range(repeats):
                row = np.zeros(80, dtype=np.float32)
                row[class_index] = np.float32(4.0 + split_index * 0.1)
                row[20] = np.float32(repeat * 0.01)
                rows.append(row)
                word = "bed" if label == "unknown" else label
                path = (
                    f"_background_noise_/{split}.wav"
                    if label == "silence"
                    else f"{word}/speaker_nohash_{repeat}.wav"
                )
                examples.append(
                    ClipExample(
                        kind="silence" if label == "silence" else "command",
                        path=path,
                        split=split,
                        label=label,
                        start_sample=repeat if label == "silence" else 0,
                        source_sample_count=32_000 if label == "silence" else 16_000,
                        source_sha256=f"{class_index + 1:064x}",
                    )
                )
    return np.asarray(rows, dtype=np.float32), tuple(examples)


def test_fit_uses_explicit_train_rows_and_reports_fixed_class_order() -> None:
    features, examples = _experiment()
    run = fit_linear_baseline(
        features,
        examples,
        config_path=Path("configs/experiment-000.json"),
        manifest_sha256=MANIFEST_SHA256,
    )

    assert run.metrics["class_order"] == list(CLASS_ORDER)
    assert run.metrics["headline_split"] == "test"
    fit = run.metrics["fit"]
    assert isinstance(fit, dict)
    assert fit["training_examples"] == 24
    assert fit["scaler_fit_examples"] == 24
    splits = run.metrics["splits"]
    assert isinstance(splits, dict)
    for split in ("validation", "test"):
        metrics = splits[split]
        assert isinstance(metrics, dict)
        assert metrics["example_count"] == 12
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["macro_f1"] == pytest.approx(1.0)
        assert len(metrics["confusion_matrix"]) == 12
        assert len(metrics["per_class"]) == 12
        unknown = metrics["unknown_by_source_word"]
        assert isinstance(unknown, list)
        assert unknown == [
            {
                "predicted_unknown_count": 1,
                "predicted_unknown_recall": 1.0,
                "source_word": "bed",
                "support": 1,
            }
        ]

    model_bytes = (json.dumps(run.model, indent=2, sort_keys=True) + "\n").encode()
    assert (
        run.metrics["portable_model_sha256"] == hashlib.sha256(model_bytes).hexdigest()
    )
    classifier = run.model["classifier"]
    assert isinstance(classifier, dict)
    assert classifier["classes"] == sorted(CLASS_ORDER)
    scaler = run.model["scaler"]
    assert isinstance(scaler, dict)
    train_mask = np.asarray([example.split == "train" for example in examples])
    assert np.asarray(scaler["mean"]) == pytest.approx(
        np.mean(features[train_mask].astype(np.float64), axis=0)
    )


def test_rejects_non_float_cache_missing_class_and_changed_protocol(
    tmp_path: Path,
) -> None:
    features, examples = _experiment()
    with pytest.raises(LinearBaselineError, match="float32"):
        fit_linear_baseline(
            features.astype(np.float64),
            examples,
            config_path=Path("configs/experiment-000.json"),
            manifest_sha256=MANIFEST_SHA256,
        )

    keep = np.asarray([example.label != "yes" for example in examples])
    with pytest.raises(LinearBaselineError, match="every class"):
        fit_linear_baseline(
            features[keep],
            tuple(example for example in examples if example.label != "yes"),
            config_path=Path("configs/experiment-000.json"),
            manifest_sha256=MANIFEST_SHA256,
        )

    config = json.loads(Path("configs/experiment-000.json").read_text())
    config["classifier"]["parameters"]["C"] = 2.0
    changed = tmp_path / "experiment.json"
    changed.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(LinearBaselineError, match="fit protocol"):
        fit_linear_baseline(
            features,
            examples,
            config_path=changed,
            manifest_sha256=MANIFEST_SHA256,
        )


def test_run_artifacts_are_published_together_and_fail_if_present(
    tmp_path: Path,
) -> None:
    features, examples = _experiment()
    run = fit_linear_baseline(
        features,
        examples,
        config_path=Path("configs/experiment-000.json"),
        manifest_sha256=MANIFEST_SHA256,
    )
    output = tmp_path / "run"
    write_baseline_run(output, run)

    assert json.loads((output / "model.json").read_text()) == run.model
    assert json.loads((output / "metrics.json").read_text()) == run.metrics
    with pytest.raises(LinearBaselineError, match="already exists"):
        write_baseline_run(output, run)


def test_registered_result_reconciles_with_its_provenance() -> None:
    model_path = Path("models/experiment-000-linear.json")
    model = json.loads(model_path.read_text(encoding="utf-8"))
    metrics = json.loads(
        Path("reports/experiment-000-linear.json").read_text(encoding="utf-8")
    )
    features = json.loads(
        Path("reports/experiment-000-features.json").read_text(encoding="utf-8")
    )
    sampling = json.loads(
        Path("reports/experiment-000-sampling.json").read_text(encoding="utf-8")
    )

    assert (
        metrics["portable_model_sha256"]
        == hashlib.sha256(model_path.read_bytes()).hexdigest()
    )
    assert (
        metrics["experiment_config_sha256"]
        == hashlib.sha256(Path("configs/experiment-000.json").read_bytes()).hexdigest()
    )
    assert metrics["examples_sha256"] == sampling["examples_sha256"]
    assert metrics["features_sha256"] == features["features_sha256"]
    assert metrics["manifest_sha256"] == MANIFEST_SHA256
    assert metrics["threshold_metrics"] == "not_evaluated"

    classifier = model["classifier"]
    assert isinstance(classifier, dict)
    assert classifier["classes"] == sorted(CLASS_ORDER)
    assert np.asarray(classifier["coef"]).shape == (12, 80)
    assert np.asarray(classifier["intercept"]).shape == (12,)
    assert np.all(np.isfinite(np.asarray(classifier["coef"])))

    for split in ("validation", "test"):
        result = metrics["splits"][split]
        matrix = np.asarray(result["confusion_matrix"], dtype=np.int64)
        assert matrix.shape == (12, 12)
        assert int(matrix.sum()) == result["example_count"]
        assert int(np.trace(matrix)) == result["correct"]
        assert result["accuracy"] == pytest.approx(
            result["correct"] / result["example_count"], abs=1e-15
        )
        assert result["macro_f1"] == pytest.approx(
            np.mean([row["f1"] for row in result["per_class"]]), abs=1e-15
        )
        unknown_index = CLASS_ORDER.index("unknown")
        silence_index = CLASS_ORDER.index("silence")
        assert sum(
            row["predicted_unknown_count"] for row in result["unknown_by_source_word"]
        ) == int(matrix[unknown_index, unknown_index])
        assert sum(row["support"] for row in result["unknown_by_source_word"]) == int(
            matrix[unknown_index].sum()
        )
        assert result["open_set_target_prediction"]["unknown"][
            "predicted_target_count"
        ] == int(matrix[unknown_index, :10].sum())
        assert result["open_set_target_prediction"]["silence"][
            "predicted_target_count"
        ] == int(matrix[silence_index, :10].sum())
