"""Build and audit a Speech Commands v0.02 experiment manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import wave
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

TARGET_WORDS = (
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
)

Split = Literal["train", "validation", "test"]


class SpeechCommandsError(ValueError):
    """Raised when a source tree cannot support the registered experiment."""


@dataclass(frozen=True, slots=True)
class SpeechCommandRecord:
    path: str
    split: Split
    word: str
    label: str
    speaker_id: str
    utterance_index: int
    sample_count: int


@dataclass(frozen=True, slots=True)
class BackgroundRecord:
    path: str
    sample_count: int


@dataclass(frozen=True, slots=True)
class SpeechCommandsManifest:
    records: tuple[SpeechCommandRecord, ...]
    background: tuple[BackgroundRecord, ...]
    jsonl_sha256: str

    def summary(self) -> dict[str, object]:
        split_counts = Counter(record.split for record in self.records)
        label_counts = Counter(record.label for record in self.records)
        speakers: dict[str, set[str]] = {
            "train": set(),
            "validation": set(),
            "test": set(),
        }
        for record in self.records:
            speakers[record.split].add(record.speaker_id)
        return {
            "background": [asdict(record) for record in self.background],
            "jsonl_sha256": self.jsonl_sha256,
            "label_counts": dict(sorted(label_counts.items())),
            "record_count": len(self.records),
            "schema_version": 1,
            "source": "speech_commands_v0.02",
            "speaker_counts": {
                split: len(values) for split, values in speakers.items()
            },
            "split_counts": {
                split: split_counts[split] for split in ("train", "validation", "test")
            },
            "target_words": list(TARGET_WORDS),
        }


def _read_split_list(root: Path, filename: str) -> set[str]:
    path = root / filename
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise SpeechCommandsError(f"cannot read {filename}: {error}") from error

    entries: set[str] = set()
    for line_number, raw_line in enumerate(lines, start=1):
        entry = raw_line.strip()
        if not entry:
            continue
        parsed = PurePosixPath(entry)
        if (
            parsed.is_absolute()
            or len(parsed.parts) != 2
            or any(part in {"", ".", ".."} for part in parsed.parts)
            or "\\" in entry
            or parsed.suffix != ".wav"
            or parsed.parts[0] == "_background_noise_"
        ):
            raise SpeechCommandsError(
                f"{filename}:{line_number} has an invalid audio path: {entry!r}"
            )
        normalized = parsed.as_posix()
        if normalized in entries:
            raise SpeechCommandsError(
                f"{filename}:{line_number} repeats {normalized!r}"
            )
        entries.add(normalized)
    return entries


def _parse_utterance(path: str) -> tuple[str, int]:
    stem = PurePosixPath(path).stem
    speaker_id, separator, raw_index = stem.rpartition("_nohash_")
    if (
        not separator
        or not speaker_id
        or not raw_index.isascii()
        or not raw_index.isdigit()
    ):
        raise SpeechCommandsError(f"cannot derive speaker and utterance from {path!r}")
    return speaker_id, int(raw_index)


def _wav_sample_count(path: Path, *, allow_long: bool) -> int:
    if path.is_symlink():
        raise SpeechCommandsError(f"audio file must not be a symbolic link: {path}")
    try:
        with wave.open(str(path), "rb") as audio:
            channels = audio.getnchannels()
            sample_width = audio.getsampwidth()
            sample_rate = audio.getframerate()
            sample_count = audio.getnframes()
            compression = audio.getcomptype()
    except (OSError, EOFError, wave.Error) as error:
        raise SpeechCommandsError(
            f"cannot read WAV metadata for {path}: {error}"
        ) from error
    if (
        channels != 1
        or sample_width != 2
        or sample_rate != 16_000
        or compression != "NONE"
        or sample_count < 1
        or (not allow_long and sample_count > 16_000)
    ):
        raise SpeechCommandsError(
            f"unsupported WAV format for {path}: channels={channels}, "
            f"sample_width={sample_width}, sample_rate={sample_rate}, "
            f"sample_count={sample_count}, compression={compression}"
        )
    return sample_count


def inspect_speech_commands(
    root: Path,
) -> tuple[tuple[SpeechCommandRecord, ...], tuple[BackgroundRecord, ...]]:
    """Inspect one extracted v0.02 tree and return its audited rows."""

    root = root.resolve()
    if not root.is_dir():
        raise SpeechCommandsError(f"dataset root is not a directory: {root}")

    validation = _read_split_list(root, "validation_list.txt")
    testing = _read_split_list(root, "testing_list.txt")
    overlap = validation & testing
    if overlap:
        example = min(overlap)
        raise SpeechCommandsError(
            f"validation and testing lists overlap at {example!r}"
        )

    audio_paths = sorted(
        path.relative_to(root).as_posix()
        for path in root.glob("*/*.wav")
        if path.parent.name != "_background_noise_"
    )
    if not audio_paths:
        raise SpeechCommandsError("dataset contains no command utterances")
    audio_set = set(audio_paths)
    missing = (validation | testing) - audio_set
    if missing:
        raise SpeechCommandsError(
            f"split lists refer to a missing file: {min(missing)!r}"
        )

    records: list[SpeechCommandRecord] = []
    speaker_split: dict[str, Split] = {}
    for relative_path in audio_paths:
        parsed = PurePosixPath(relative_path)
        word = parsed.parts[0]
        speaker_id, utterance_index = _parse_utterance(relative_path)
        split: Split
        if relative_path in validation:
            split = "validation"
        elif relative_path in testing:
            split = "test"
        else:
            split = "train"
        previous = speaker_split.setdefault(speaker_id, split)
        if previous != split:
            raise SpeechCommandsError(
                f"speaker {speaker_id!r} crosses {previous!r} and {split!r}"
            )
        records.append(
            SpeechCommandRecord(
                path=relative_path,
                split=split,
                word=word,
                label=word if word in TARGET_WORDS else "unknown",
                speaker_id=speaker_id,
                utterance_index=utterance_index,
                sample_count=_wav_sample_count(root / relative_path, allow_long=False),
            )
        )

    background_paths = sorted((root / "_background_noise_").glob("*.wav"))
    if not background_paths:
        raise SpeechCommandsError("dataset contains no background-noise recordings")
    background = tuple(
        BackgroundRecord(
            path=path.relative_to(root).as_posix(),
            sample_count=_wav_sample_count(path, allow_long=True),
        )
        for path in background_paths
    )
    return tuple(records), background


def build_speech_commands_manifest(root: Path, output: Path) -> SpeechCommandsManifest:
    """Write the JSONL rows and a small reviewable summary."""

    records, background = inspect_speech_commands(root)
    rows = b"".join(
        (
            json.dumps(asdict(record), sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        for record in records
    )
    manifest = SpeechCommandsManifest(
        records=records,
        background=background,
        jsonl_sha256=hashlib.sha256(rows).hexdigest(),
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "speech-commands.jsonl").write_bytes(rows)
    (output / "speech-commands-summary.json").write_text(
        json.dumps(manifest.summary(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit Speech Commands v0.02 and write the experiment manifest."
    )
    parser.add_argument("root", type=Path, help="extracted speech_commands_v0.02 root")
    parser.add_argument("output", type=Path, help="directory for generated manifests")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = build_speech_commands_manifest(args.root, args.output)
    print(json.dumps(manifest.summary(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
