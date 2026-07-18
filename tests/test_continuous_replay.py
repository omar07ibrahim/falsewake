from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from falsewake.continuous_replay import (
    BOOTSTRAP_RESAMPLES,
    CLASS_ORDER,
    MODEL_SHA256,
    SAMPLES_PER_HOUR,
    THRESHOLD_COUNT,
    ContinuousReplayError,
    CorrectAcceptGrid,
    EventGrid,
    PortableLinearModel,
    ScoreBatch,
    UtteranceEventGrid,
    aggregate_event_grids,
    correct_accept_threshold_grid,
    event_rate_intervals,
    events_at_threshold,
    garwood_rate_intervals,
    load_portable_model,
    parse_portable_model,
    replay_threshold_grid,
    score_feature_rows,
    score_utterance,
    scored_exposure_samples,
    select_threshold,
    speaker_bootstrap_indices,
    window_count,
)
from falsewake.speech_commands import TARGET_WORDS


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


def _event_grid(counts: np.ndarray, *, target_position: int = 0) -> EventGrid:
    selected = np.asarray(counts, dtype=np.int64)
    assert selected.shape == (THRESHOLD_COUNT,)
    by_target = np.zeros((THRESHOLD_COUNT, len(TARGET_WORDS)), dtype=np.int64)
    by_target[:, target_position] = selected
    return EventGrid(counts=selected, counts_by_target=by_target)


def _correct_grid(counts: np.ndarray, *, support: int) -> CorrectAcceptGrid:
    selected = np.asarray(counts, dtype=np.int64)
    by_target = np.zeros((THRESHOLD_COUNT, len(TARGET_WORDS)), dtype=np.int64)
    by_target[:, 0] = selected
    clip_counts = np.zeros(len(TARGET_WORDS), dtype=np.int64)
    clip_counts[0] = support
    return CorrectAcceptGrid(
        clip_counts_by_target=clip_counts,
        counts=selected,
        counts_by_target=by_target,
    )


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


def test_utterance_grids_aggregate_by_ascending_integer_speaker() -> None:
    first_counts = np.zeros(THRESHOLD_COUNT, dtype=np.int64)
    first_counts[:501] = 1
    second_counts = np.zeros(THRESHOLD_COUNT, dtype=np.int64)
    second_counts[:101] = 2
    second_counts[101:901] = 1
    third_counts = np.zeros(THRESHOLD_COUNT, dtype=np.int64)
    third_counts[:201] = 1
    utterances = (
        UtteranceEventGrid(7, 16_000, _event_grid(first_counts, target_position=9)),
        UtteranceEventGrid(2, 32_000, _event_grid(second_counts)),
        UtteranceEventGrid(2, 16_000, _event_grid(third_counts, target_position=1)),
    )

    aggregate = aggregate_event_grids(utterances)

    assert aggregate.speaker_ids.tolist() == [2, 7]
    assert aggregate.utterance_counts_by_speaker.tolist() == [2, 1]
    assert aggregate.scored_exposure_samples_by_speaker.tolist() == [48_000, 16_000]
    assert aggregate.scored_exposure_samples == 64_000
    assert aggregate.counts_by_speaker[:, 0].tolist() == [3, 1]
    assert aggregate.counts[[0, 150, 700, 950]].tolist() == [4, 3, 1, 0]
    assert np.array_equal(aggregate.counts_by_target.sum(axis=1), aggregate.counts)
    assert aggregate.counts_by_target[0, [0, 1, 9]].tolist() == [2, 1, 1]
    for values in (
        aggregate.speaker_ids,
        aggregate.utterance_counts_by_speaker,
        aggregate.scored_exposure_samples_by_speaker,
        aggregate.counts_by_speaker,
        aggregate.counts_by_speaker_and_target,
        aggregate.counts,
        aggregate.counts_by_target,
    ):
        assert not values.flags.writeable


def test_correct_accept_grid_uses_true_target_and_inclusive_threshold() -> None:
    model = load_portable_model(Path("models/experiment-000-linear.json"))
    scores = _scores(model, ["down", "go", "go"], [0.5, 0.8, 0.9])
    target_positions = np.asarray(
        [TARGET_WORDS.index("down"), TARGET_WORDS.index("go"), 0],
        dtype=np.int64,
    )

    grid = correct_accept_threshold_grid(model, target_positions, scores)

    assert grid.clip_count == 3
    assert grid.baseline_correct_count == 2
    assert grid.counts[[0, 500, 501, 800, 801]].tolist() == [2, 2, 1, 1, 0]
    assert grid.counts_by_target[500, TARGET_WORDS.index("down")] == 1
    assert grid.counts_by_target[800, TARGET_WORDS.index("go")] == 1
    assert not grid.counts.flags.writeable


def test_selection_uses_exact_integer_boundaries_and_lowest_intersection() -> None:
    negative_counts = np.ones(THRESHOLD_COUNT, dtype=np.int64)
    negative_counts[:400] = 2
    negative = aggregate_event_grids(
        (
            UtteranceEventGrid(
                11,
                SAMPLES_PER_HOUR,
                _event_grid(negative_counts),
            ),
        )
    )
    correct_counts = np.full(THRESHOLD_COUNT, 8, dtype=np.int64)
    correct_counts[:400] = 10
    correct_counts[601:] = 7
    positive = _correct_grid(correct_counts, support=10)

    selection = select_threshold(negative, positive)

    assert selection.status == "pass"
    assert selection.selected_threshold_milli == 400
    assert selection.negative_frontier_milli == 400
    assert selection.retention_frontier_milli == 600
    assert not selection.false_event_gate[399]
    assert selection.false_event_gate[400]
    assert positive.baseline_correct_count * 4 == 8 * 5
    assert selection.retention_gate[600]
    assert not selection.retention_gate[601]


def test_selection_rejects_disjoint_gates_and_preserves_null_frontier() -> None:
    negative_counts = np.full(THRESHOLD_COUNT, 1, dtype=np.int64)
    negative = aggregate_event_grids(
        (
            UtteranceEventGrid(
                3,
                SAMPLES_PER_HOUR - 1_600,
                _event_grid(negative_counts),
            ),
        )
    )
    correct_counts = np.full(THRESHOLD_COUNT, 8, dtype=np.int64)
    correct_counts[0] = 10
    correct_counts[601:] = 7

    selection = select_threshold(negative, _correct_grid(correct_counts, support=10))

    assert selection.status == "reject"
    assert selection.selected_threshold_milli is None
    assert selection.negative_frontier_milli is None
    assert selection.retention_frontier_milli == 600
    assert not np.any(selection.passing)


def test_selection_products_do_not_overflow_int64() -> None:
    maximum = int(np.iinfo(np.int64).max)
    exposure = 16_000 + ((maximum - 16_000) // 1_600) * 1_600
    event_limit = exposure // SAMPLES_PER_HOUR
    negative_counts = np.full(THRESHOLD_COUNT, event_limit, dtype=np.int64)
    negative_counts[:10] += 1
    negative = aggregate_event_grids(
        (UtteranceEventGrid(1, exposure, _event_grid(negative_counts)),)
    )
    positive = _correct_grid(np.ones(THRESHOLD_COUNT, dtype=np.int64), support=1)

    selection = select_threshold(negative, positive)

    assert (event_limit + 1) * SAMPLES_PER_HOUR > maximum
    assert event_limit * SAMPLES_PER_HOUR <= exposure
    assert selection.selected_threshold_milli == 10


def test_garwood_and_shared_speaker_bootstrap_have_fixed_values() -> None:
    first_counts = np.zeros(THRESHOLD_COUNT, dtype=np.int64)
    first_counts[0] = 2
    second_counts = np.zeros(THRESHOLD_COUNT, dtype=np.int64)
    second_counts[0] = 4
    second_counts[1] = 2
    aggregate = aggregate_event_grids(
        (
            UtteranceEventGrid(1, SAMPLES_PER_HOUR, _event_grid(first_counts)),
            UtteranceEventGrid(2, SAMPLES_PER_HOUR, _event_grid(second_counts)),
        )
    )

    intervals = event_rate_intervals(aggregate)

    assert intervals.point[:3].tolist() == [3.0, 1.0, 0.0]
    assert intervals.garwood_lower[2] == 0.0
    assert intervals.garwood_lower[0] == pytest.approx(1.1009471267454254)
    assert intervals.garwood_upper[0] == pytest.approx(6.529737011259343)
    assert intervals.garwood_upper[2] == pytest.approx(1.84444, abs=1e-6)
    assert intervals.speaker_bootstrap_lower[:3].tolist() == [2.0, 0.0, 0.0]
    assert intervals.speaker_bootstrap_upper[:3].tolist() == [4.0, 2.0, 0.0]
    assert not intervals.speaker_bootstrap_lower.flags.writeable


def test_registered_bootstrap_matrix_and_linear_percentiles_are_reproducible() -> None:
    indices = speaker_bootstrap_indices(3)

    assert indices.shape == (BOOTSTRAP_RESAMPLES, 3)
    assert indices[:5].tolist() == [
        [2, 2, 0],
        [2, 1, 1],
        [0, 1, 2],
        [1, 1, 0],
        [2, 1, 2],
    ]
    assert np.array_equal(indices, speaker_bootstrap_indices(3))
    assert not indices.flags.writeable


def test_bootstrap_reuses_one_matrix_for_unequal_speaker_exposures() -> None:
    speaker_counts = (
        np.asarray([1, 0, *([0] * (THRESHOLD_COUNT - 2))], dtype=np.int64),
        np.asarray([4, 2, *([0] * (THRESHOLD_COUNT - 2))], dtype=np.int64),
        np.asarray([9, 3, *([0] * (THRESHOLD_COUNT - 2))], dtype=np.int64),
    )
    speaker_exposures = np.asarray([1, 2, 3], dtype=np.int64) * SAMPLES_PER_HOUR
    aggregate = aggregate_event_grids(
        tuple(
            UtteranceEventGrid(
                speaker_id,
                int(speaker_exposures[speaker_id - 1]),
                _event_grid(speaker_counts[speaker_id - 1]),
            )
            for speaker_id in (1, 2, 3)
        )
    )

    intervals = event_rate_intervals(aggregate)
    indices = speaker_bootstrap_indices(3)
    for threshold in (0, 1):
        events = np.asarray(
            [counts[threshold] for counts in speaker_counts], dtype=np.int64
        )
        resampled_rates = (
            events[indices].sum(axis=1, dtype=np.int64)
            * SAMPLES_PER_HOUR
            / speaker_exposures[indices].sum(axis=1, dtype=np.int64)
        )
        expected = np.percentile(resampled_rates, [2.5, 97.5], method="linear")
        assert intervals.speaker_bootstrap_lower[threshold] == expected[0]
        assert intervals.speaker_bootstrap_upper[threshold] == expected[1]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("dtype", "int64 array"),
        ("shape", "shape"),
        ("target_sum", "does not sum"),
        ("increasing", "must not increase"),
        ("exposure", "registered exposure"),
        ("speaker", "speaker_id"),
    ],
)
def test_utterance_grid_schema_is_strict(mutation: str, message: str) -> None:
    counts = np.zeros(THRESHOLD_COUNT, dtype=np.int64)
    by_target = np.zeros((THRESHOLD_COUNT, len(TARGET_WORDS)), dtype=np.int64)
    exposure = 16_000
    speaker: object = 1
    if mutation == "dtype":
        counts = counts.astype(np.int32)
    elif mutation == "shape":
        counts = counts[:-1]
    elif mutation == "target_sum":
        counts[0] = 1
    elif mutation == "increasing":
        counts[1] = 1
        by_target[1, 0] = 1
    elif mutation == "exposure":
        exposure = 16_001
    else:
        speaker = np.int64(1)

    with pytest.raises(ContinuousReplayError, match=message):
        UtteranceEventGrid(
            cast(int, speaker),
            exposure,
            EventGrid(counts=counts, counts_by_target=by_target),
        )


def test_aggregate_rejects_empty_input_and_int64_exposure_overflow() -> None:
    with pytest.raises(ContinuousReplayError, match="empty"):
        aggregate_event_grids(())

    maximum = int(np.iinfo(np.int64).max)
    exposure = 16_000 + ((maximum - 16_000) // 1_600) * 1_600
    empty = _event_grid(np.zeros(THRESHOLD_COUNT, dtype=np.int64))
    utterances = (
        UtteranceEventGrid(1, exposure, empty),
        UtteranceEventGrid(2, exposure, empty),
    )
    with pytest.raises(ContinuousReplayError, match="overflows int64"):
        aggregate_event_grids(utterances)


def test_interval_and_positive_grid_validation_rejects_invalid_inputs() -> None:
    counts = np.zeros(THRESHOLD_COUNT, dtype=np.int64)
    with pytest.raises(ContinuousReplayError, match="positive"):
        _correct_grid(counts, support=0)
    with pytest.raises(ContinuousReplayError, match="exceeds"):
        garwood_rate_intervals(np.full(THRESHOLD_COUNT, 2, dtype=np.int64), 16_000)
    with pytest.raises(ContinuousReplayError, match="size limit"):
        speaker_bootstrap_indices(1_001)
