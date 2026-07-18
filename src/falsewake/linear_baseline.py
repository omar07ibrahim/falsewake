"""Fit and evaluate the frozen experiment 000 linear clip baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
import time
import warnings
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

import numpy as np
import scipy
import sklearn
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler

from falsewake.baseline_data import (
    MANIFEST_SHA256,
    ClipExample,
    build_sampling_plan,
    load_audited_corpus,
    sampling_summary,
)
from falsewake.feature_matrix import (
    FeatureMatrixError,
    config_sha256,
    feature_matrix_sha256,
    load_feature_cache,
)
from falsewake.features import FloatArray
from falsewake.speech_commands import TARGET_WORDS, Split

CLASS_ORDER = (*TARGET_WORDS, "unknown", "silence")
SCIKIT_LEARN_VERSION = "1.9.0"
MAX_CONFIG_BYTES = 1024 * 1024

_CLASSIFIER = {
    "implementation": "sklearn.linear_model.LogisticRegression",
    "parameters": {
        "C": 1.0,
        "class_weight": None,
        "dual": False,
        "fit_intercept": True,
        "intercept_scaling": 1,
        "l1_ratio": 0.0,
        "max_iter": 1000,
        "n_jobs": None,
        "random_state": 20_260_718,
        "solver": "lbfgs",
        "tol": 0.0001,
        "verbose": 0,
        "warm_start": False,
    },
    "scikit_learn_version": SCIKIT_LEARN_VERSION,
    "standard_scaler": {
        "copy": True,
        "fit_split": "train",
        "with_mean": True,
        "with_std": True,
    },
}

_TRAINING = {
    "cache_dtype": "float32",
    "convergence_warning": "error",
    "fit_input_dtype": "float64_c_contiguous",
    "fit_split": "train",
    "store_n_iter": True,
    "validation_role": "diagnostic_only_no_selection",
}

_EVALUATION = {
    "clip_prediction": "argmax",
    "confusion_matrix_labels": "class_order",
    "decision_columns": "classifier.classes_",
    "headline_split": "test",
    "macro_f1": {"labels": "class_order", "zero_division": 0},
    "open_set_target_prediction_rate": ["unknown", "silence"],
    "per_class": "precision_recall_f1_support",
    "reported_splits": ["validation", "test"],
    "target_argmax_error_rate": {
        "denominator": "clips_with_one_of_the_ten_target_labels",
        "numerator": "argmax_prediction_differs_from_label",
    },
    "threshold_metrics": "deferred_to_continuous_experiment",
    "unknown_word_slice": {
        "metric": "predicted_unknown_recall",
        "source_word": "first_manifest_path_component",
        "support": True,
    },
}


class LinearBaselineError(ValueError):
    """The frozen training or evaluation contract was violated."""


@dataclass(frozen=True, slots=True)
class BaselineRun:
    model: dict[str, object]
    metrics: dict[str, object]


def _json_bytes(document: dict[str, object]) -> bytes:
    return (
        json.dumps(document, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _registered_config(path: Path) -> str:
    """Require the exact classifier, training, and evaluation protocol."""

    digest = config_sha256(path)
    try:
        if path.stat().st_size > MAX_CONFIG_BYTES:
            raise LinearBaselineError("experiment config exceeds the size limit")
        contents = path.read_bytes()
        document = json.loads(contents)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise LinearBaselineError(f"cannot read experiment config: {error}") from error
    if hashlib.sha256(contents).hexdigest() != digest or type(document) is not dict:
        raise LinearBaselineError("experiment config changed while reading")
    registered = cast(dict[object, object], document)
    if (
        registered.get("classifier") != _CLASSIFIER
        or registered.get("training") != _TRAINING
        or registered.get("evaluation") != _EVALUATION
    ):
        raise LinearBaselineError("experiment config differs from the fit protocol")
    return digest


def _split_mask(examples: Sequence[ClipExample], split: Split) -> NDArray[np.bool_]:
    return np.fromiter(
        (example.split == split for example in examples),
        dtype=np.bool_,
        count=len(examples),
    )


def _labels(examples: Sequence[ClipExample]) -> NDArray[np.str_]:
    return np.asarray([example.label for example in examples], dtype=np.str_)


def _fit(
    features: FloatArray, examples: Sequence[ClipExample]
) -> tuple[StandardScaler, LogisticRegression, float]:
    train_mask = _split_mask(examples, "train")
    train_labels = _labels(examples)[train_mask]
    if not np.any(train_mask) or set(train_labels.tolist()) != set(CLASS_ORDER):
        raise LinearBaselineError("training split does not contain every class")

    train_features = np.ascontiguousarray(features[train_mask], dtype=np.float64)
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaled = scaler.fit_transform(train_features)
    if scaler.n_samples_seen_ != train_features.shape[0]:
        raise LinearBaselineError("scaler was not fit on exactly the training rows")

    classifier = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=0.0,
        max_iter=1000,
        n_jobs=None,
        random_state=20_260_718,
        solver="lbfgs",
        tol=0.0001,
        verbose=0,
        warm_start=False,
    )
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            classifier.fit(scaled, train_labels)
        except ConvergenceWarning as error:
            raise LinearBaselineError("linear baseline did not converge") from error
    fit_seconds = time.perf_counter() - started
    fitted_arrays = (
        scaler.mean_,
        scaler.scale_,
        classifier.coef_,
        classifier.intercept_,
    )
    if any(not np.all(np.isfinite(values)) for values in fitted_arrays) or np.any(
        scaler.scale_ <= 0
    ):
        raise LinearBaselineError("fitted scaler or classifier is non-finite")
    if tuple(classifier.classes_.tolist()) != tuple(sorted(CLASS_ORDER)):
        raise LinearBaselineError("classifier class columns differ from the data")
    return scaler, classifier, fit_seconds


def _predict(
    features: FloatArray,
    mask: NDArray[np.bool_],
    scaler: StandardScaler,
    classifier: LogisticRegression,
) -> NDArray[np.str_]:
    selected = np.ascontiguousarray(features[mask], dtype=np.float64)
    scaled = scaler.transform(selected)
    predicted = cast(NDArray[np.str_], classifier.predict(scaled))
    scores = scaled @ classifier.coef_.T + classifier.intercept_
    if not np.all(np.isfinite(scores)):
        raise LinearBaselineError("linear baseline produced non-finite scores")
    manual = classifier.classes_[np.argmax(scores, axis=1)]
    if not np.array_equal(predicted, manual):
        raise LinearBaselineError("portable linear logits differ from sklearn predict")
    return predicted


def _unknown_word_metrics(
    examples: Sequence[ClipExample],
    mask: NDArray[np.bool_],
    predictions: NDArray[np.str_],
) -> list[dict[str, object]]:
    selected_examples = [
        example for example, include in zip(examples, mask, strict=True) if include
    ]
    counts: dict[str, list[bool]] = defaultdict(list)
    for example, predicted in zip(selected_examples, predictions, strict=True):
        if example.label != "unknown":
            continue
        word = PurePosixPath(example.path).parts[0]
        counts[word].append(predicted == "unknown")
    return [
        {
            "predicted_unknown_count": sum(values),
            "predicted_unknown_recall": sum(values) / len(values),
            "source_word": word,
            "support": len(values),
        }
        for word, values in sorted(counts.items())
    ]


def _split_metrics(
    split: Split,
    examples: Sequence[ClipExample],
    labels: NDArray[np.str_],
    predictions: NDArray[np.str_],
    mask: NDArray[np.bool_],
) -> dict[str, object]:
    true = labels[mask]
    if true.size == 0:
        raise LinearBaselineError(f"{split} split is empty")
    precision, recall, f1, support = precision_recall_fscore_support(
        true,
        predictions,
        labels=CLASS_ORDER,
        zero_division=0,
    )
    per_class = [
        {
            "f1": float(f1[index]),
            "label": label,
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(CLASS_ORDER)
    ]
    target_mask = np.isin(true, TARGET_WORDS)
    target_errors = int(np.count_nonzero(predictions[target_mask] != true[target_mask]))
    target_support = int(np.count_nonzero(target_mask))
    if target_support == 0:
        raise LinearBaselineError(f"{split} split has no target clips")

    open_set: dict[str, object] = {}
    targets = set(TARGET_WORDS)
    for label in ("unknown", "silence"):
        label_mask = true == label
        label_support = int(np.count_nonzero(label_mask))
        if label_support == 0:
            raise LinearBaselineError(f"{split} split has no {label} clips")
        predicted_target = sum(
            prediction in targets for prediction in predictions[label_mask]
        )
        open_set[label] = {
            "predicted_target_count": predicted_target,
            "predicted_target_rate": predicted_target / label_support,
            "support": label_support,
        }

    return {
        "accuracy": float(accuracy_score(true, predictions)),
        "confusion_matrix": confusion_matrix(
            true, predictions, labels=CLASS_ORDER
        ).tolist(),
        "correct": int(np.count_nonzero(true == predictions)),
        "example_count": int(true.size),
        "macro_f1": float(
            f1_score(
                true,
                predictions,
                labels=CLASS_ORDER,
                average="macro",
                zero_division=0,
            )
        ),
        "open_set_target_prediction": open_set,
        "per_class": per_class,
        "target_argmax_error": {
            "error_count": target_errors,
            "error_rate": target_errors / target_support,
            "support": target_support,
        },
        "unknown_by_source_word": _unknown_word_metrics(examples, mask, predictions),
    }


def _model_document(
    scaler: StandardScaler,
    classifier: LogisticRegression,
    *,
    config_digest: str,
    examples_digest: str,
    features_digest: str,
    manifest_sha256: str,
) -> dict[str, object]:
    return {
        "classifier": {
            "classes": classifier.classes_.tolist(),
            "coef": classifier.coef_.tolist(),
            "intercept": classifier.intercept_.tolist(),
            "n_iter": classifier.n_iter_.tolist(),
        },
        "experiment_config_sha256": config_digest,
        "examples_sha256": examples_digest,
        "features_sha256": features_digest,
        "input": {
            "dtype": "float64",
            "feature_count": int(classifier.n_features_in_),
            "formula": "((x - scaler.mean) / scaler.scale) @ coef.T + intercept",
        },
        "manifest_sha256": manifest_sha256,
        "scaler": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        },
        "schema_version": 1,
        "scikit_learn_version": sklearn.__version__,
    }


def fit_linear_baseline(
    features: FloatArray,
    examples: Sequence[ClipExample],
    *,
    config_path: Path,
    manifest_sha256: str,
) -> BaselineRun:
    """Fit on explicit train rows and report validation/test without tuning."""

    if sklearn.__version__ != SCIKIT_LEARN_VERSION:
        raise LinearBaselineError(
            f"scikit-learn version differs: expected={SCIKIT_LEARN_VERSION}, "
            f"observed={sklearn.__version__}"
        )
    config_digest = _registered_config(config_path)
    if features.dtype != np.dtype(np.float32) or features.ndim != 2:
        raise LinearBaselineError("features must be a two-dimensional float32 matrix")
    if features.shape != (len(examples), 80) or not np.all(np.isfinite(features)):
        raise LinearBaselineError("features differ from the sampling plan or frontend")
    if not examples:
        raise LinearBaselineError("sampling plan is empty")
    if manifest_sha256 != MANIFEST_SHA256:
        raise LinearBaselineError("manifest_sha256 differs from experiment 000")

    selected = sampling_summary(examples)
    examples_digest = cast(str, selected["examples_sha256"])
    features_digest = feature_matrix_sha256(features)
    labels = _labels(examples)
    if any(label not in CLASS_ORDER for label in labels):
        raise LinearBaselineError("sampling plan contains an unregistered label")

    scaler, classifier, fit_seconds = _fit(features, examples)
    evaluation_started = time.perf_counter()
    splits: dict[str, object] = {}
    for split in ("validation", "test"):
        mask = _split_mask(examples, split)
        predictions = _predict(features, mask, scaler, classifier)
        splits[split] = _split_metrics(split, examples, labels, predictions, mask)
    evaluation_seconds = time.perf_counter() - evaluation_started

    model = _model_document(
        scaler,
        classifier,
        config_digest=config_digest,
        examples_digest=examples_digest,
        features_digest=features_digest,
        manifest_sha256=manifest_sha256,
    )
    model_sha256 = hashlib.sha256(_json_bytes(model)).hexdigest()
    metrics: dict[str, object] = {
        "class_order": list(CLASS_ORDER),
        "evaluation_seconds": evaluation_seconds,
        "experiment_config_sha256": config_digest,
        "examples_sha256": examples_digest,
        "features_sha256": features_digest,
        "fit": {
            "fit_seconds": fit_seconds,
            "n_iter": classifier.n_iter_.tolist(),
            "scaler_fit_examples": int(
                np.count_nonzero(_split_mask(examples, "train"))
            ),
            "training_examples": int(np.count_nonzero(_split_mask(examples, "train"))),
        },
        "headline_split": "test",
        "manifest_sha256": manifest_sha256,
        "portable_model_sha256": model_sha256,
        "runtime": {
            "numpy": np.__version__,
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__,
        },
        "schema_version": 1,
        "scikit_learn_version": sklearn.__version__,
        "splits": splits,
        "threshold_metrics": "not_evaluated",
        "validation_role": "diagnostic_only_no_selection",
    }
    return BaselineRun(model=model, metrics=metrics)


def write_baseline_run(output: Path, run: BaselineRun) -> None:
    """Publish model and metrics together through a staged rename."""

    output_parent = output.parent.resolve()
    if not output_parent.is_dir():
        raise LinearBaselineError(f"run parent is not a directory: {output_parent}")
    if output.exists() or output.is_symlink():
        raise LinearBaselineError(f"run output already exists: {output}")
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output_parent))
    try:
        (stage / "model.json").write_bytes(_json_bytes(run.model))
        (stage / "metrics.json").write_bytes(_json_bytes(run.metrics))
        os.replace(stage, output)
    except (OSError, TypeError, ValueError) as error:
        shutil.rmtree(stage, ignore_errors=True)
        raise LinearBaselineError(f"cannot publish baseline run: {error}") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit the frozen experiment 000 linear baseline."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("feature_cache", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiment-000.json")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load the bound cache, fit once, and publish portable JSON artifacts."""

    arguments = _parser().parse_args(argv)
    try:
        corpus = load_audited_corpus(
            arguments.manifest, expected_sha256=MANIFEST_SHA256
        )
        examples = build_sampling_plan(corpus)
        digest = config_sha256(arguments.config)
        features = load_feature_cache(
            arguments.feature_cache,
            examples,
            manifest_sha256=corpus.manifest_sha256,
            experiment_config_sha256=digest,
        )
        run = fit_linear_baseline(
            features,
            examples,
            config_path=arguments.config,
            manifest_sha256=corpus.manifest_sha256,
        )
        write_baseline_run(arguments.output, run)
    except (FeatureMatrixError, LinearBaselineError, ValueError) as error:
        print(f"linear baseline failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(run.metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
