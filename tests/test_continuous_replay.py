from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from falsewake.continuous_replay import (
    CLASS_ORDER,
    MODEL_SHA256,
    ContinuousReplayError,
    PortableLinearModel,
    ScoreBatch,
    events_at_threshold,
    load_portable_model,
    parse_portable_model,
    replay_threshold_grid,
    score_feature_rows,
    score_utterance,
    scored_exposure_samples,
    window_count,
)


def _model_document() -> dict[str, object]:
    return cast(
        dict[str, object],
        json.loads(
            Path("models/experiment-000-linear.json").read_text(encoding="utf-8")
        ),
    )


def _model_bytes(document: dict[str, object]) -> bytes:
    return (json.dumps(document, allow_nan=False, sort_keys=True) + "\n").encode()


def _zero_model(model: PortableLinearModel) -> PortableLinearModel:
    coef = np.zeros_like(model.coef)
    intercept = np.zeros_like(model.intercept)
    coef.flags.writeable = False
    intercept.flags.writeable = False
    return replace(model, coef=coef, intercept=intercept)


def _scores(
    model: PortableLinearModel,
    labels: list[str],
    probabilities: list[float],
) -> ScoreBatch:
    row_count = len(labels)
    values = np.zeros((row_count, len(model.classes)), dtype=np.float64)
    predicted = np.empty(row_count, dtype=np.int64)
    for row, (label, probability) in enumerate(zip(labels, probabilities, strict=True)):
        index = model.classes.index(label)
        predicted[row] = index
        remainder = (1.0 - probability) / (len(model.classes) - 1)
        values[row] = remainder
        values[row, index] = probability
    logits = np.log(np.maximum(values, np.finfo(np.float64).tiny))
    return ScoreBatch(logits=logits, probabilities=values, predicted_indices=predicted)


def _sparse_target_scores(
    model: PortableLinearModel,
    count: int,
    events: dict[int, tuple[str, float]],
) -> ScoreBatch:
    labels = ["unknown"] * count
    probabilities = [1.0] * count
    for index, (label, probability) in events.items():
        labels[index] = label
        probabilities[index] = probability
    return _scores(model, labels, probabilities)


def test_checked_in_portable_model_scores_with_stable_softmax() -> None:
    model = load_portable_model(Path("models/experiment-000-linear.json"))
    features = np.zeros((2, 80), dtype=np.float32)

    scores = score_feature_rows(model, features)

    assert model.classes == CLASS_ORDER
    assert scores.logits.shape == scores.probabilities.shape == (2, 12)
    assert np.allclose(np.sum(scores.probabilities, axis=1), 1.0, rtol=0, atol=1e-15)
    assert np.array_equal(scores.predicted_indices, np.argmax(scores.logits, axis=1))
    assert not scores.probabilities.flags.writeable
    observed_model_sha256 = hashlib.sha256(
        Path("models/experiment-000-linear.json").read_bytes()
    ).hexdigest()
    assert observed_model_sha256 == MODEL_SHA256


def test_exact_argmax_tie_uses_the_first_model_class() -> None:
    model = _zero_model(load_portable_model(Path("models/experiment-000-linear.json")))

    scores = score_feature_rows(model, np.zeros((1, 80), dtype=np.float32))

    assert scores.predicted_indices.tolist() == [0]
    assert model.classes[0] == "down"
    assert scores.probabilities[0, 0] == pytest.approx(1 / 12)


def test_logit_argmax_survives_a_sub_ulp_softmax_tie() -> None:
    base = _zero_model(load_portable_model(Path("models/experiment-000-linear.json")))
    intercept = np.zeros_like(base.intercept)
    intercept[1] = np.nextafter(np.float64(0), np.float64(1))
    intercept.flags.writeable = False
    model = replace(base, intercept=intercept)

    scores = score_feature_rows(model, np.zeros((1, 80), dtype=np.float32))
    grid = replay_threshold_grid(model, np.asarray([0], dtype=np.int64), scores)

    assert scores.logits[0, 1] > scores.logits[0, 0]
    assert scores.probabilities[0, 1] == scores.probabilities[0, 0]
    assert scores.predicted_indices.tolist() == [1]
    assert model.classes[1] == "go"
    assert grid.counts[0] == 1


def test_model_parser_rejects_duplicate_keys_and_nonfinite_numbers() -> None:
    with pytest.raises(ContinuousReplayError, match="repeats JSON key"):
        parse_portable_model(b'{"schema_version":1,"schema_version":1}')
    with pytest.raises(ContinuousReplayError, match="invalid JSON value"):
        parse_portable_model(b'{"value":NaN}')


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("classes", "class order"),
        ("coef", "invalid shape"),
        ("scale", "positive"),
        ("schema", "schema_version"),
        ("input", "input contract"),
    ],
)
def test_model_parser_rejects_schema_and_array_drift(
    mutation: str, message: str
) -> None:
    document = _model_document()
    classifier = cast(dict[str, object], document["classifier"])
    scaler = cast(dict[str, object], document["scaler"])
    if mutation == "classes":
        classifier["classes"] = list(reversed(cast(list[str], classifier["classes"])))
    elif mutation == "coef":
        classifier["coef"] = cast(list[object], classifier["coef"])[:-1]
    elif mutation == "scale":
        values = cast(list[float], scaler["scale"])
        values[0] = 0.0
    elif mutation == "schema":
        document["schema_version"] = True
    else:
        cast(dict[str, object], document["input"])["dtype"] = "float32"

    with pytest.raises(ContinuousReplayError, match=message):
        parse_portable_model(_model_bytes(document))


def test_model_loader_rejects_digest_mismatch_and_symlink(tmp_path: Path) -> None:
    source = Path("models/experiment-000-linear.json")
    copy = tmp_path / "model.json"
    copy.write_bytes(source.read_bytes())
    with pytest.raises(ContinuousReplayError, match="SHA-256 differs"):
        load_portable_model(copy, expected_sha256="0" * 64)

    link = tmp_path / "model-link.json"
    link.symlink_to(source.resolve())
    with pytest.raises(ContinuousReplayError, match="cannot open"):
        load_portable_model(link)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("float_prediction", "invalid arrays"),
        ("nan_logit", "invalid arrays"),
        ("inconsistent_probability", "differs from stable softmax"),
    ],
)
def test_event_replay_rejects_malformed_score_batches(
    mutation: str, message: str
) -> None:
    model = load_portable_model(Path("models/experiment-000-linear.json"))
    base = score_feature_rows(model, np.zeros((1, 80), dtype=np.float32))
    if mutation == "float_prediction":
        malformed = replace(
            base, predicted_indices=base.predicted_indices.astype(np.float64)
        )
    elif mutation == "nan_logit":
        logits = base.logits.copy()
        logits[0, 0] = np.nan
        malformed = replace(base, logits=logits)
    else:
        probabilities = base.probabilities.copy()
        probabilities[0] = np.roll(probabilities[0], 1)
        malformed = replace(base, probabilities=probabilities)

    with pytest.raises(ContinuousReplayError, match=message):
        replay_threshold_grid(model, np.asarray([0], dtype=np.int64), malformed)


@pytest.mark.parametrize(
    ("samples", "count", "exposure"),
    [
        (0, 0, 0),
        (15_999, 0, 0),
        (16_000, 1, 16_000),
        (17_599, 1, 16_000),
        (17_600, 2, 17_600),
        (32_001, 11, 32_000),
    ],
)
def test_window_count_and_scored_exposure_are_exact(
    samples: int, count: int, exposure: int
) -> None:
    assert window_count(samples) == count
    assert scored_exposure_samples(samples) == exposure


def test_score_utterance_drops_tail_and_never_crosses_a_boundary() -> None:
    model = load_portable_model(Path("models/experiment-000-linear.json"))
    waveform = np.zeros(17_600, dtype=np.float32)

    starts, scores = score_utterance(model, waveform)

    assert starts.tolist() == [0, 1_600]
    assert scores.probabilities.shape == (2, 12)


def test_grid_uses_global_inclusive_refractory_state_per_threshold() -> None:
    model = load_portable_model(Path("models/experiment-000-linear.json"))
    starts = np.arange(21, dtype=np.int64) * 1_600
    scores = _sparse_target_scores(
        model,
        21,
        {
            0: ("down", 0.5),
            1: ("go", 0.9),
            10: ("go", 0.5),
            11: ("down", 0.9),
            20: ("go", 0.5),
        },
    )

    grid = replay_threshold_grid(model, starts, scores)

    assert grid.counts[500] == 3
    assert grid.counts[501] == 2
    assert grid.counts[900] == 2
    assert grid.counts[901] == 0
    assert grid.counts_by_target[500].sum() == grid.counts[500]
    assert not grid.counts.flags.writeable


def test_threshold_states_shift_events_independently_and_reset_per_utterance() -> None:
    model = load_portable_model(Path("models/experiment-000-linear.json"))
    starts = np.arange(16, dtype=np.int64) * 1_600
    scores = _sparse_target_scores(
        model,
        16,
        {
            0: ("down", 0.5),
            5: ("down", 0.9),
            10: ("down", 0.5),
            15: ("down", 0.9),
        },
    )

    low = events_at_threshold(model, starts, scores, 500)
    high = events_at_threshold(model, starts, scores, 900)
    reset = events_at_threshold(
        model,
        np.asarray([0], dtype=np.int64),
        _scores(model, ["down"], [0.9]),
        900,
    )

    assert [event.start_sample for event in low] == [0, 16_000]
    assert [event.start_sample for event in high] == [8_000, 24_000]
    assert [event.start_sample for event in reset] == [0]


def test_non_target_argmax_never_qualifies() -> None:
    model = load_portable_model(Path("models/experiment-000-linear.json"))
    starts = np.asarray([0, 1_600], dtype=np.int64)
    scores = _scores(model, ["unknown", "silence"], [1.0, 1.0])

    grid = replay_threshold_grid(model, starts, scores)

    assert not np.any(grid.counts)
