from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from falsewake import continuous_replay as replay
from falsewake import holdout
from falsewake.continuous_replay import (
    THRESHOLD_COUNT,
    ContinuousReplayError,
    CorrectAcceptGrid,
    EventGrid,
    ReplayEventAccumulator,
    ReplayInputs,
    ReplayTrace,
    UtteranceEventGrid,
    _load_positive_grid,
    _publish_artifacts,
    aggregate_event_grids,
    build_replay_artifacts,
    load_portable_model,
)
from falsewake.speech_commands import TARGET_WORDS


def _sha256(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _identities(runtime: dict[str, str]) -> dict[str, str]:
    runtime_sha256 = _sha256(holdout.canonical_runtime_identity(runtime))
    result = {
        field: ("1" * 40 if field == "implementation_git_commit" else "a" * 64)
        for field in holdout.REPLAY_IDENTITY_FIELDS
    }
    result["runtime_identity_sha256"] = runtime_sha256
    return result


def _positive_grid(retention_frontier: int) -> CorrectAcceptGrid:
    support = np.asarray(holdout.TARGET_SUPPORT, dtype=np.int64)
    baseline = np.asarray(holdout.BASELINE_CORRECT_BY_TARGET, dtype=np.int64)
    by_target = np.zeros(
        (THRESHOLD_COUNT, len(TARGET_WORDS)), dtype=np.int64
    )
    by_target[: retention_frontier + 1] = baseline
    return CorrectAcceptGrid(
        clip_counts_by_target=support,
        counts=np.sum(by_target, axis=1, dtype=np.int64),
        counts_by_target=by_target,
    )


def _one_target_grid(
    *, speaker_id: int, exposure: int, last_event_threshold: int
) -> UtteranceEventGrid:
    counts = np.zeros(THRESHOLD_COUNT, dtype=np.int64)
    counts[: last_event_threshold + 1] = 1
    by_target = np.zeros(
        (THRESHOLD_COUNT, len(TARGET_WORDS)), dtype=np.int64
    )
    by_target[:, 0] = counts
    return UtteranceEventGrid(
        speaker_id=speaker_id,
        scored_exposure_samples=exposure,
        events=EventGrid(counts=counts, counts_by_target=by_target),
    )


def _trace(
    *,
    utterance_id: str,
    speaker_id: int,
    starts: list[int],
    probabilities: list[float],
) -> ReplayTrace:
    return ReplayTrace(
        utterance_id=utterance_id,
        speaker_id=speaker_id,
        transcript=f"TRANSCRIPT {utterance_id}",
        window_starts=np.asarray(starts, dtype=np.int64),
        target_positions=np.zeros(len(starts), dtype=np.int64),
        target_probabilities=np.asarray(probabilities, dtype=np.float64),
    )


def test_streaming_accumulator_matches_canonical_aggregation() -> None:
    records = (
        _one_target_grid(speaker_id=2, exposure=32_000, last_event_threshold=200),
        _one_target_grid(speaker_id=1, exposure=16_000, last_event_threshold=400),
        _one_target_grid(speaker_id=2, exposure=48_000, last_event_threshold=600),
    )
    accumulator = ReplayEventAccumulator()
    for record in reversed(records):
        accumulator.add(record)
    streamed = accumulator.aggregate()
    canonical = aggregate_event_grids(records)

    assert streamed.scored_exposure_samples == canonical.scored_exposure_samples
    assert streamed.speaker_ids.tolist() == [1, 2]
    assert streamed.utterance_counts_by_speaker.tolist() == [1, 2]
    for field in (
        "scored_exposure_samples_by_speaker",
        "counts_by_speaker",
        "counts_by_speaker_and_target",
        "counts",
        "counts_by_target",
    ):
        assert np.array_equal(
            cast(
                np.ndarray[tuple[int, ...], np.dtype[np.int64]],
                getattr(streamed, field),
            ),
            cast(
                np.ndarray[tuple[int, ...], np.dtype[np.int64]],
                getattr(canonical, field),
            ),
        )


def test_canonical_replay_is_byte_reproducible_and_hash_bound() -> None:
    exposure = 5_760_000
    negative = aggregate_event_grids(
        (
            _one_target_grid(
                speaker_id=1,
                exposure=exposure,
                last_event_threshold=499,
            ),
        )
    )
    traces = (
        _trace(
            utterance_id="1-2-0003",
            speaker_id=1,
            starts=[0],
            probabilities=[0.499],
        ),
    )
    positive = _positive_grid(retention_frontier=700)
    runtime = holdout.current_runtime_identity()
    first = build_replay_artifacts(
        source_samples=exposure,
        negative=negative,
        positive=positive,
        traces=traces,
        identities=_identities(runtime),
        runtime=runtime,
    )
    second = build_replay_artifacts(
        source_samples=exposure,
        negative=negative,
        positive=positive,
        traces=traces,
        identities=_identities(runtime),
        runtime=runtime,
    )

    assert first == second
    assert first.status == "pass"
    assert first.selected_threshold_milli == 500
    assert first.report.endswith(b"\n")
    assert first.report_sha256 == _sha256(first.report)
    assert first.selection_artifact_sha256 == _sha256(first.selection_artifact)
    assert holdout.recompute_selection(first.report) == ("pass", 500)
    artifact = json.loads(first.selection_artifact)
    assert artifact["dev_replay_report_sha256"] == first.report_sha256
    report = json.loads(first.report)
    assert [row["threshold_milli"] for row in report["top_false_events"]] == [
        0,
        500,
        700,
    ]
    assert report["top_false_events"][0]["events"][0]["utterance_id"] == (
        "1-2-0003"
    )
    assert report["top_false_events"][1]["events"] == []


def test_top_events_replay_refractory_state_at_each_threshold() -> None:
    exposure = 57_600_000
    negative = aggregate_event_grids(
        (
            _one_target_grid(
                speaker_id=1,
                exposure=exposure,
                last_event_threshold=900,
            ),
        )
    )
    traces = (
        _trace(
            utterance_id="1-2-0003",
            speaker_id=1,
            starts=[0, 1_600],
            probabilities=[0.4, 0.9],
        ),
    )
    runtime = holdout.current_runtime_identity()
    artifacts = build_replay_artifacts(
        source_samples=exposure,
        negative=negative,
        positive=_positive_grid(retention_frontier=500),
        traces=traces,
        identities=_identities(runtime),
        runtime=runtime,
    )
    report = json.loads(artifacts.report)
    groups = {
        row["threshold_milli"]: row["events"]
        for row in report["top_false_events"]
    }

    assert groups[0][0]["window_start_sample"] == 0
    assert groups[500][0]["window_start_sample"] == 1_600
    assert groups[500][0]["target_probability"] == 0.9


def test_top_events_are_ranked_and_capped_at_fifty() -> None:
    records = tuple(
        _one_target_grid(
            speaker_id=1,
            exposure=16_000,
            last_event_threshold=499,
        )
        for _ in range(52)
    )
    traces = tuple(
        _trace(
            utterance_id=f"1-2-{index:04d}",
            speaker_id=1,
            starts=[0],
            probabilities=[0.499],
        )
        for index in reversed(range(52))
    )
    runtime = holdout.current_runtime_identity()
    artifacts = build_replay_artifacts(
        source_samples=52 * 16_000,
        negative=aggregate_event_grids(records),
        positive=_positive_grid(retention_frontier=700),
        traces=traces,
        identities=_identities(runtime),
        runtime=runtime,
    )
    groups = json.loads(artifacts.report)["top_false_events"]
    zero_events = groups[0]["events"]

    assert len(zero_events) == 50
    assert [event["utterance_id"] for event in zero_events] == [
        f"1-2-{index:04d}" for index in range(50)
    ]


@pytest.mark.parametrize("mismatch", ("metadata", "matrix"))
def test_raw_feature_cache_digests_are_checked_before_loading(
    tmp_path: Path, mismatch: str
) -> None:
    speech_manifest = tmp_path / "speech.jsonl"
    speech_manifest.write_bytes(b"not parsed because raw cache identity fails\n")
    feature_cache = tmp_path / "cache"
    feature_cache.mkdir()
    metadata = b'{"schema_version":1}\n'
    matrix = b"not-a-npy"
    (feature_cache / "metadata.json").write_bytes(metadata)
    (feature_cache / "features.npy").write_bytes(matrix)
    inputs = ReplayInputs(
        repository_root=tmp_path,
        speech_commands_manifest=speech_manifest,
        feature_cache=feature_cache,
        dev_archive=tmp_path / "unused-archive",
        dev_manifest=tmp_path / "unused-manifest",
        dev_audit_report=tmp_path / "unused-audit",
    )
    config: dict[str, object] = {
        "positive_validation": {
            "feature_cache": {
                "manifest_sha256": _sha256(speech_manifest.read_bytes()),
                "metadata_json_sha256": (
                    "0" * 64 if mismatch == "metadata" else _sha256(metadata)
                ),
                "features_npy_sha256": (
                    "0" * 64 if mismatch == "matrix" else _sha256(matrix)
                ),
            }
        }
    }

    with pytest.raises(
        ContinuousReplayError, match="feature-cache container identity differs"
    ):
        _load_positive_grid(
            inputs,
            config,
            load_portable_model(Path("models/experiment-000-linear.json")),
        )


def test_executing_scorer_must_come_from_requested_checkout(tmp_path: Path) -> None:
    with pytest.raises(
        ContinuousReplayError, match="executing scorer module path differs"
    ):
        replay._validate_executing_scorer_paths(tmp_path)


def test_git_blob_size_is_bounded_before_payload_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    relative_path = "src/falsewake/continuous_replay.py"
    object_id = "a" * 40

    def git_output(
        _: Path, arguments: Sequence[str], *, name: str
    ) -> bytes:
        assert arguments[:2] == ["ls-tree", "-z"]
        assert name.startswith("tree entry")
        return (
            f"100644 blob {object_id}\t{relative_path}".encode("ascii") + b"\0"
        )

    def bounded_output(
        _: Path,
        arguments: Sequence[str],
        *,
        name: str,
        maximum_bytes: int,
    ) -> bytes:
        assert name.startswith("blob size")
        assert arguments[:2] == ["cat-file", "-s"]
        assert maximum_bytes == 64
        return b"11\n"

    monkeypatch.setattr(replay, "_run_git", git_output)
    monkeypatch.setattr(replay, "_run_git_bounded", bounded_output)

    with pytest.raises(ContinuousReplayError, match="Git blob exceeds the limit"):
        replay._git_blob(
            tmp_path,
            commit="1" * 40,
            relative_path=relative_path,
            maximum_bytes=10,
        )


def test_artifacts_publish_as_one_fail_if_present_directory(tmp_path: Path) -> None:
    exposure = 5_760_000
    negative = aggregate_event_grids(
        (
            _one_target_grid(
                speaker_id=1,
                exposure=exposure,
                last_event_threshold=499,
            ),
        )
    )
    runtime = holdout.current_runtime_identity()
    artifacts = build_replay_artifacts(
        source_samples=exposure,
        negative=negative,
        positive=_positive_grid(retention_frontier=700),
        traces=(
            _trace(
                utterance_id="1-2-0003",
                speaker_id=1,
                starts=[0],
                probabilities=[0.499],
            ),
        ),
        identities=_identities(runtime),
        runtime=runtime,
    )
    first = tmp_path / "run-a"
    second = tmp_path / "run-b"
    _publish_artifacts(first, artifacts)
    _publish_artifacts(second, artifacts)

    first_files = {path.name: path.read_bytes() for path in first.iterdir()}
    second_files = {path.name: path.read_bytes() for path in second.iterdir()}
    assert first_files == second_files == {
        "experiment-001-dev-replay.json": artifacts.report,
        "experiment-001-selection.json": artifacts.selection_artifact,
    }
    with pytest.raises(ContinuousReplayError, match="output"):
        _publish_artifacts(first, artifacts)
    assert {path.name: path.read_bytes() for path in first.iterdir()} == first_files


def test_synthetic_runner_revalidates_and_reproduces_two_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exposure = 5_760_000
    negative = aggregate_event_grids(
        (
            _one_target_grid(
                speaker_id=1,
                exposure=exposure,
                last_event_threshold=499,
            ),
        )
    )
    traces = (
        _trace(
            utterance_id="1-2-0003",
            speaker_id=1,
            starts=[0],
            probabilities=[0.499],
        ),
    )
    runtime = holdout.current_runtime_identity()
    model_bytes = Path("models/experiment-000-linear.json").read_bytes()
    implementation = replay._ImplementationSnapshot(
        commit="1" * 40,
        config={
            "positive_validation": {
                "examples_sha256": "b" * 64,
                "feature_cache": {"features_sha256": "c" * 64},
            }
        },
        config_bytes=b"{}\n",
        config_sha256="a" * 64,
        model_bytes=model_bytes,
        model_sha256=_sha256(model_bytes),
        scorer_source_sha256="d" * 64,
        runtime=runtime,
        runtime_sha256=_sha256(holdout.canonical_runtime_identity(runtime)),
    )
    development = replay._DevReplayState(
        source_samples=exposure,
        aggregate=negative,
        traces=traces,
        archive_sha256="e" * 64,
        manifest_sha256="f" * 64,
        audit_report_sha256="0" * 64,
    )
    calls: list[str] = []

    def capture(_: Path) -> replay._ImplementationSnapshot:
        calls.append("implementation")
        return implementation

    def capture_inputs(_: ReplayInputs) -> object:
        calls.append("inputs")
        return object()

    def load_positive(
        _: ReplayInputs,
        __: object,
        ___: replay.PortableLinearModel,
    ) -> CorrectAcceptGrid:
        calls.append("positive")
        return _positive_grid(retention_frontier=700)

    def stage_dev(
        _: ReplayInputs,
        __: object,
        ___: replay.PortableLinearModel,
    ) -> replay._DevReplayState:
        calls.append("development")
        return development

    def revalidate(
        _: ReplayInputs,
        __: object,
        *,
        expected_archive_sha256: str,
    ) -> None:
        assert expected_archive_sha256 == development.archive_sha256
        calls.append("revalidate")

    monkeypatch.setattr(replay, "_capture_implementation", capture)
    monkeypatch.setattr(replay, "_capture_replay_inputs", capture_inputs)
    monkeypatch.setattr(replay, "_load_positive_grid", load_positive)
    monkeypatch.setattr(replay, "_stage_dev_replay", stage_dev)
    monkeypatch.setattr(replay, "_revalidate_replay_inputs", revalidate)
    inputs = ReplayInputs(
        repository_root=tmp_path,
        speech_commands_manifest=tmp_path / "synthetic-speech",
        feature_cache=tmp_path / "synthetic-cache",
        dev_archive=tmp_path / "synthetic-dev",
        dev_manifest=tmp_path / "synthetic-manifest",
        dev_audit_report=tmp_path / "synthetic-audit",
    )
    first = replay.run_dev_replay(inputs, tmp_path / "run-a")
    second = replay.run_dev_replay(inputs, tmp_path / "run-b")

    assert first == second
    assert (tmp_path / "run-a/experiment-001-dev-replay.json").read_bytes() == (
        tmp_path / "run-b/experiment-001-dev-replay.json"
    ).read_bytes()
    assert calls == [
        "implementation",
        "inputs",
        "positive",
        "development",
        "revalidate",
        "implementation",
    ] * 2

    def fail_revalidation(
        _: ReplayInputs,
        __: object,
        *,
        expected_archive_sha256: str,
    ) -> None:
        assert expected_archive_sha256 == development.archive_sha256
        raise ContinuousReplayError("synthetic late mutation")

    monkeypatch.setattr(replay, "_revalidate_replay_inputs", fail_revalidation)
    failed_output = tmp_path / "failed-run"
    with pytest.raises(ContinuousReplayError, match="synthetic late mutation"):
        replay.run_dev_replay(inputs, failed_output)
    assert not failed_output.exists()


def test_reject_result_keeps_null_threshold() -> None:
    exposure = 5_760_000
    runtime = holdout.current_runtime_identity()
    artifacts = build_replay_artifacts(
        source_samples=exposure,
        negative=aggregate_event_grids(
            (
                _one_target_grid(
                    speaker_id=1,
                    exposure=exposure,
                    last_event_threshold=1_000,
                ),
            )
        ),
        positive=_positive_grid(retention_frontier=700),
        traces=(
            _trace(
                utterance_id="1-2-0003",
                speaker_id=1,
                starts=[0],
                probabilities=[1.0],
            ),
        ),
        identities=_identities(runtime),
        runtime=runtime,
    )
    artifact = json.loads(artifacts.selection_artifact)

    assert artifacts.status == "reject"
    assert artifacts.selected_threshold_milli is None
    assert artifact["status"] == "reject"
    assert artifact["selected_threshold_milli"] is None
