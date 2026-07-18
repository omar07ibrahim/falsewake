from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from falsewake.speech_commands import (
    SpeechCommandsError,
    build_speech_commands_manifest,
    inspect_speech_commands,
)


def _write_wav(path: Path, sample_count: int = 320) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\x00\x00" * sample_count)


def _fixture(root: Path) -> Path:
    _write_wav(root / "yes" / "alice_nohash_0.wav")
    _write_wav(root / "no" / "alice_nohash_1.wav")
    _write_wav(root / "go" / "bob_nohash_0.wav")
    _write_wav(root / "cat" / "carol_nohash_0.wav")
    _write_wav(root / "_background_noise_" / "room.wav", sample_count=32_000)
    (root / "validation_list.txt").write_text("go/bob_nohash_0.wav\n", encoding="utf-8")
    (root / "testing_list.txt").write_text("cat/carol_nohash_0.wav\n", encoding="utf-8")
    return root


def test_manifest_is_stable_and_keeps_speakers_disjoint(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "dataset")
    first = build_speech_commands_manifest(root, tmp_path / "first")
    second = build_speech_commands_manifest(root, tmp_path / "second")

    assert first == second
    assert first.jsonl_sha256 == second.jsonl_sha256
    assert [record.path for record in first.records] == sorted(
        record.path for record in first.records
    )
    assert {record.speaker_id: record.split for record in first.records} == {
        "alice": "train",
        "bob": "validation",
        "carol": "test",
    }
    assert next(record for record in first.records if record.word == "cat").label == (
        "unknown"
    )

    summary = json.loads(
        (tmp_path / "first" / "speech-commands-summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["record_count"] == 4
    assert summary["split_counts"] == {"test": 1, "train": 2, "validation": 1}
    assert summary["speaker_counts"] == {
        "test": 1,
        "train": 1,
        "validation": 1,
    }
    assert summary["label_counts"] == {
        "go": 1,
        "no": 1,
        "unknown": 1,
        "yes": 1,
    }
    assert summary["background"] == [
        {"path": "_background_noise_/room.wav", "sample_count": 32_000}
    ]


def test_speaker_crossing_splits_is_rejected(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "dataset")
    _write_wav(root / "right" / "bob_nohash_1.wav")
    (root / "testing_list.txt").write_text(
        "cat/carol_nohash_0.wav\nright/bob_nohash_1.wav\n", encoding="utf-8"
    )

    with pytest.raises(SpeechCommandsError, match="speaker 'bob' crosses"):
        inspect_speech_commands(root)


def test_split_entry_for_missing_audio_is_rejected(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "dataset")
    (root / "validation_list.txt").write_text(
        "go/bob_nohash_0.wav\nleft/nobody_nohash_0.wav\n", encoding="utf-8"
    )

    with pytest.raises(SpeechCommandsError, match="missing file"):
        inspect_speech_commands(root)


def test_non_pcm_command_audio_is_rejected(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "dataset")
    bad = root / "yes" / "alice_nohash_0.wav"
    with wave.open(str(bad), "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\x00\x00" * 320)

    with pytest.raises(SpeechCommandsError, match="unsupported WAV format"):
        inspect_speech_commands(root)
