from __future__ import annotations

import hashlib
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
    first = build_speech_commands_manifest(
        root, tmp_path / "first", source_identity=None
    )
    second = build_speech_commands_manifest(
        root, tmp_path / "second", source_identity=None
    )

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
    assert summary["schema_version"] == 2
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
    assert summary["background_count"] == 1
    assert summary["background"][0]["path"] == "_background_noise_/room.wav"
    assert summary["background"][0]["split"] == "train"
    assert summary["background"][0]["sample_count"] == 32_000
    assert summary["source"] == "unverified"

    rows = (tmp_path / "first" / "speech-commands.jsonl").read_bytes()
    assert rows.count(b"\n") == 5
    assert rows == (tmp_path / "second" / "speech-commands.jsonl").read_bytes()
    assert first.jsonl_sha256 == (
        "63867b45349e73c8765aeaa93fd47d536f33da3fe90505fe0acef9dac4084982"
    )
    assert (
        hashlib.sha256(
            (tmp_path / "first" / "speech-commands-summary.json").read_bytes()
        ).hexdigest()
        == "43ee8f813b5a9231d7038e97713663c190729171d3cfd9e6b0286bf7fd6bd173"
    )


def test_speaker_crossing_splits_is_rejected(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "dataset")
    _write_wav(root / "right" / "bob_nohash_1.wav")
    (root / "testing_list.txt").write_text(
        "cat/carol_nohash_0.wav\nright/bob_nohash_1.wav\n", encoding="utf-8"
    )

    with pytest.raises(SpeechCommandsError, match="speaker 'bob' crosses"):
        inspect_speech_commands(root, source_identity=None)


def test_split_entry_for_missing_audio_is_rejected(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "dataset")
    (root / "validation_list.txt").write_text(
        "go/bob_nohash_0.wav\nleft/nobody_nohash_0.wav\n", encoding="utf-8"
    )

    with pytest.raises(SpeechCommandsError, match="missing file"):
        inspect_speech_commands(root, source_identity=None)


def test_non_pcm_command_audio_is_rejected(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "dataset")
    bad = root / "yes" / "alice_nohash_0.wav"
    with wave.open(str(bad), "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\x00\x00" * 320)

    with pytest.raises(SpeechCommandsError, match="unsupported WAV format"):
        inspect_speech_commands(root, source_identity=None)


def test_registered_source_rejects_an_unregistered_split(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "dataset")

    with pytest.raises(SpeechCommandsError, match="registered source"):
        inspect_speech_commands(root)


def test_truncated_wav_payload_is_rejected(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "dataset")
    audio = root / "yes" / "alice_nohash_0.wav"
    audio.write_bytes(audio.read_bytes()[:-20])

    with pytest.raises(SpeechCommandsError, match="truncated WAV payload"):
        inspect_speech_commands(root, source_identity=None)


def test_output_symlink_is_rejected(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "dataset")
    destination = tmp_path / "destination"
    destination.mkdir()
    output = tmp_path / "output"
    output.symlink_to(destination, target_is_directory=True)

    with pytest.raises(SpeechCommandsError, match="output path"):
        build_speech_commands_manifest(root, output, source_identity=None)
