from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from falsewake.baseline_data import (
    EXPERIMENT_SEED,
    MANIFEST_SHA256,
    AuditedCorpus,
    BackgroundSource,
    BaselineDataError,
    CommandSource,
    build_sampling_plan,
    load_audited_corpus,
    sampling_summary,
)
from falsewake.features import DEFAULT_FRONTEND
from falsewake.speech_commands import TARGET_WORDS, Split


def _corpus() -> AuditedCorpus:
    commands: list[CommandSource] = []
    splits: tuple[Split, ...] = ("train", "validation", "test")
    for split in splits:
        for label, count in (("yes", 2), ("no", 3), ("unknown", 5)):
            commands.extend(
                CommandSource(
                    path=f"{label}/{split}-{label}-{index}.wav",
                    split=split,
                    label=label,
                    sample_count=16_000,
                    sha256=f"{index + 1:064x}",
                )
                for index in range(count)
            )
    background = tuple(
        BackgroundSource(
            path=f"_background_noise_/{split}.wav",
            split=split,
            sample_count=32_000,
            sha256=f"{index + 100:064x}",
        )
        for index, split in enumerate(splits)
    )
    return AuditedCorpus(tuple(commands), background, "a" * 64)


def test_sampling_plan_is_stable_balanced_and_split_local() -> None:
    corpus = _corpus()
    first = build_sampling_plan(corpus, target_words=("yes", "no"))
    second = build_sampling_plan(
        AuditedCorpus(tuple(reversed(corpus.commands)), corpus.background, "a" * 64),
        target_words=("yes", "no"),
    )

    assert first == second
    for split in ("train", "validation", "test"):
        selected = [example for example in first if example.split == split]
        counts = {
            label: sum(example.label == label for example in selected)
            for label in ("yes", "no", "unknown", "silence")
        }
        assert counts == {"yes": 2, "no": 3, "unknown": 2, "silence": 2}
        silence = [example for example in selected if example.label == "silence"]
        assert all(
            example.path == f"_background_noise_/{split}.wav" for example in silence
        )
        assert all(0 <= example.start_sample <= 16_000 for example in silence)
        assert len(
            {(example.path, example.start_sample) for example in silence}
        ) == len(silence)

    summary = sampling_summary(first)
    assert summary["example_count"] == 27
    assert summary["seed"] == EXPERIMENT_SEED
    assert summary["examples_sha256"] == (
        "7decd466ba7223086977c67160498b8296116c1725a2bce6be602ee552b3bf1d"
    )


def test_sampling_requires_every_target_and_split_background() -> None:
    corpus = _corpus()
    without_test_background = AuditedCorpus(
        corpus.commands,
        tuple(record for record in corpus.background if record.split != "test"),
        corpus.manifest_sha256,
    )

    with pytest.raises(BaselineDataError, match="every target"):
        build_sampling_plan(corpus, target_words=("yes", "missing"))
    with pytest.raises(BaselineDataError, match="test has no usable"):
        build_sampling_plan(without_test_background, target_words=("yes", "no"))


def test_manifest_loader_binds_bytes_and_extracts_source_rows(tmp_path: Path) -> None:
    rows = [
        {
            "kind": "command",
            "label": "yes",
            "path": "yes/alice_nohash_0.wav",
            "sample_count": 16_000,
            "sha256": "a" * 64,
            "split": "train",
        },
        {
            "kind": "background",
            "path": "_background_noise_/room.wav",
            "sample_count": 32_000,
            "sha256": "b" * 64,
            "split": "train",
        },
    ]
    contents = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_bytes(contents)
    digest = hashlib.sha256(contents).hexdigest()

    corpus = load_audited_corpus(manifest, expected_sha256=digest)
    assert [asdict(record) for record in corpus.commands] == [
        {
            "label": "yes",
            "path": "yes/alice_nohash_0.wav",
            "sample_count": 16_000,
            "sha256": "a" * 64,
            "split": "train",
        }
    ]
    assert corpus.background[0].path == "_background_noise_/room.wav"

    manifest.write_bytes(contents + b"\n")
    with pytest.raises(BaselineDataError, match="digest differs"):
        load_audited_corpus(manifest, expected_sha256=digest)


def test_registered_config_matches_code_constants() -> None:
    config = json.loads(Path("configs/experiment-000.json").read_text(encoding="utf-8"))
    assert config["seed"] == EXPERIMENT_SEED
    assert config["manifest"]["jsonl_sha256"] == MANIFEST_SHA256
    assert config["class_order"] == [*TARGET_WORDS, "unknown", "silence"]
    frontend = dict(config["frontend"])
    assert frontend.pop("pcm16_divisor") == 32_768
    assert frontend == asdict(DEFAULT_FRONTEND)
