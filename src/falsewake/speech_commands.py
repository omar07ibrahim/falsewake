"""Build and audit a Speech Commands v0.02 experiment manifest."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import tempfile
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
MAX_COMMAND_FILE_BYTES = 1024 * 1024
MAX_BACKGROUND_FILE_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    name: str
    archive_sha256: str
    validation_list_sha256: str
    testing_list_sha256: str
    command_count: int
    inventory_sha256: str
    background_splits: tuple[tuple[str, Split], ...]


SPEECH_COMMANDS_V002 = SourceIdentity(
    name="speech_commands_v0.02",
    archive_sha256=("af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58"),
    validation_list_sha256=(
        "5747407275538b4056e823982f0db1fc993776ab532048196a19be701bdc87d2"
    ),
    testing_list_sha256=(
        "2d17c6b3faf63be43eda93cfeb0c747cfd79b7b236282039dbac65a2cb5f1df5"
    ),
    command_count=105_829,
    inventory_sha256=(
        "c9596927b6de1aa9bb8174bea25f613ebb7bab9e671304c525961b66e0812c5c"
    ),
    background_splits=(
        ("_background_noise_/doing_the_dishes.wav", "train"),
        ("_background_noise_/dude_miaowing.wav", "train"),
        ("_background_noise_/exercise_bike.wav", "train"),
        ("_background_noise_/pink_noise.wav", "train"),
        ("_background_noise_/running_tap.wav", "validation"),
        ("_background_noise_/white_noise.wav", "test"),
    ),
)


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
    sha256: str


@dataclass(frozen=True, slots=True)
class BackgroundRecord:
    path: str
    split: Split
    sample_count: int
    sha256: str


@dataclass(frozen=True, slots=True)
class SpeechCommandsInspection:
    records: tuple[SpeechCommandRecord, ...]
    background: tuple[BackgroundRecord, ...]
    source: str
    validation_list_sha256: str
    testing_list_sha256: str
    inventory_sha256: str


@dataclass(frozen=True, slots=True)
class SpeechCommandsManifest:
    records: tuple[SpeechCommandRecord, ...]
    background: tuple[BackgroundRecord, ...]
    jsonl_sha256: str
    source: str
    validation_list_sha256: str
    testing_list_sha256: str
    inventory_sha256: str
    archive_sha256: str | None

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
            "archive_sha256": self.archive_sha256,
            "background": [asdict(record) for record in self.background],
            "background_count": len(self.background),
            "inventory_sha256": self.inventory_sha256,
            "jsonl_sha256": self.jsonl_sha256,
            "label_counts": dict(sorted(label_counts.items())),
            "record_count": len(self.records),
            "schema_version": 2,
            "source": self.source,
            "source_files": {
                "testing_list_sha256": self.testing_list_sha256,
                "validation_list_sha256": self.validation_list_sha256,
            },
            "speaker_counts": {
                split: len(values) for split, values in speakers.items()
            },
            "split_counts": {
                split: split_counts[split] for split in ("train", "validation", "test")
            },
            "target_words": list(TARGET_WORDS),
        }


def _require_regular_source_file(root: Path, path: Path) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise SpeechCommandsError(
            f"source path escapes the dataset root: {path}"
        ) from error

    candidate = root
    for part in relative.parts:
        candidate /= part
        if candidate.is_symlink():
            raise SpeechCommandsError(f"source path must not contain symlinks: {path}")
    if not path.is_file():
        raise SpeechCommandsError(f"source path is not a regular file: {path}")


def _read_split_list(root: Path, filename: str) -> tuple[set[str], str]:
    path = root / filename
    _require_regular_source_file(root, path)
    try:
        contents = path.read_bytes()
        lines = contents.decode("utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
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
    if not entries:
        raise SpeechCommandsError(f"{filename} is empty")
    return entries, hashlib.sha256(contents).hexdigest()


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


def _wav_identity(root: Path, path: Path, *, allow_long: bool) -> tuple[int, str]:
    _require_regular_source_file(root, path)
    try:
        file_size = path.stat().st_size
        size_limit = MAX_BACKGROUND_FILE_BYTES if allow_long else MAX_COMMAND_FILE_BYTES
        if file_size > size_limit:
            raise SpeechCommandsError(
                f"WAV file exceeds the audit limit for {path}: "
                f"size={file_size}, limit={size_limit}"
            )
        contents = path.read_bytes()
        with wave.open(io.BytesIO(contents), "rb") as audio:
            channels = audio.getnchannels()
            sample_width = audio.getsampwidth()
            sample_rate = audio.getframerate()
            sample_count = audio.getnframes()
            compression = audio.getcomptype()
            frames = audio.readframes(sample_count + 1)
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
    expected_payload_bytes = sample_count * channels * sample_width
    if len(frames) != expected_payload_bytes:
        raise SpeechCommandsError(
            f"truncated WAV payload for {path}: expected={expected_payload_bytes}, "
            f"read={len(frames)}"
        )
    return sample_count, hashlib.sha256(contents).hexdigest()


def _inventory_sha256(
    records: Sequence[SpeechCommandRecord],
    background: Sequence[BackgroundRecord],
) -> str:
    digest = hashlib.sha256()
    entries = sorted(
        [(record.path, record.sha256) for record in records]
        + [(record.path, record.sha256) for record in background]
    )
    for path, sha256 in entries:
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def inspect_speech_commands(
    root: Path,
    *,
    source_identity: SourceIdentity | None = SPEECH_COMMANDS_V002,
) -> SpeechCommandsInspection:
    """Inspect one extracted v0.02 tree and return its audited rows."""

    root = root.resolve()
    if not root.is_dir():
        raise SpeechCommandsError(f"dataset root is not a directory: {root}")

    validation, validation_sha256 = _read_split_list(root, "validation_list.txt")
    testing, testing_sha256 = _read_split_list(root, "testing_list.txt")
    if source_identity is not None:
        if validation_sha256 != source_identity.validation_list_sha256:
            raise SpeechCommandsError(
                "validation_list.txt does not match the registered source: "
                f"observed={validation_sha256}"
            )
        if testing_sha256 != source_identity.testing_list_sha256:
            raise SpeechCommandsError(
                "testing_list.txt does not match the registered source: "
                f"observed={testing_sha256}"
            )
    overlap = validation & testing
    if overlap:
        example = min(overlap)
        raise SpeechCommandsError(
            f"validation and testing lists overlap at {example!r}"
        )

    audio_files = sorted(
        (
            path
            for path in root.glob("*/*.wav")
            if path.parent.name != "_background_noise_"
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    for path in audio_files:
        _require_regular_source_file(root, path)
    audio_paths = [path.relative_to(root).as_posix() for path in audio_files]
    if not audio_paths:
        raise SpeechCommandsError("dataset contains no command utterances")
    if (
        source_identity is not None
        and len(audio_paths) != source_identity.command_count
    ):
        raise SpeechCommandsError(
            "command inventory does not match the registered source: "
            f"expected={source_identity.command_count}, observed={len(audio_paths)}"
        )
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
        sample_count, sha256 = _wav_identity(
            root, root / relative_path, allow_long=False
        )
        records.append(
            SpeechCommandRecord(
                path=relative_path,
                split=split,
                word=word,
                label=word if word in TARGET_WORDS else "unknown",
                speaker_id=speaker_id,
                utterance_index=utterance_index,
                sample_count=sample_count,
                sha256=sha256,
            )
        )

    background_paths = sorted((root / "_background_noise_").glob("*.wav"))
    if not background_paths:
        raise SpeechCommandsError("dataset contains no background-noise recordings")
    split_by_background = (
        dict(source_identity.background_splits) if source_identity is not None else {}
    )
    relative_background = {
        path.relative_to(root).as_posix() for path in background_paths
    }
    if source_identity is not None and relative_background != set(split_by_background):
        raise SpeechCommandsError(
            "background inventory does not match the registered source"
        )
    background_records: list[BackgroundRecord] = []
    for path in background_paths:
        relative_path = path.relative_to(root).as_posix()
        sample_count, sha256 = _wav_identity(root, path, allow_long=True)
        background_records.append(
            BackgroundRecord(
                path=relative_path,
                split=split_by_background.get(relative_path, "train"),
                sample_count=sample_count,
                sha256=sha256,
            )
        )

    background = tuple(background_records)
    inventory_sha256 = _inventory_sha256(records, background)
    if (
        source_identity is not None
        and inventory_sha256 != source_identity.inventory_sha256
    ):
        raise SpeechCommandsError(
            "audio payload inventory does not match the registered source: "
            f"observed={inventory_sha256}"
        )
    return SpeechCommandsInspection(
        records=tuple(records),
        background=background,
        source=source_identity.name if source_identity is not None else "unverified",
        validation_list_sha256=validation_sha256,
        testing_list_sha256=testing_sha256,
        inventory_sha256=inventory_sha256,
    )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise SpeechCommandsError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _manifest_bytes(inspection: SpeechCommandsInspection) -> bytes:
    rows: list[dict[str, object]] = []
    for command in inspection.records:
        rows.append({"kind": "command", **asdict(command)})
    for background in inspection.background:
        rows.append({"kind": "background", **asdict(background)})
    return b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for row in rows
    )


def _prepare_new_output_path(output: Path) -> Path:
    output = output.absolute()
    current = Path(output.anchor)
    for part in output.parent.parts[1:]:
        current /= part
        if current.exists() and current.is_symlink():
            raise SpeechCommandsError(
                f"output path must not contain symlinks: {output}"
            )
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise SpeechCommandsError(
            f"cannot create output parent {output.parent}: {error}"
        ) from error
    if output.exists() or output.is_symlink():
        raise SpeechCommandsError(f"output path already exists: {output}")
    return output


def _atomic_write(path: Path, contents: bytes) -> None:
    if path.is_symlink():
        raise SpeechCommandsError(f"output file must not be a symlink: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(contents)
            destination.flush()
            os.fsync(destination.fileno())
        temporary.replace(path)
    except OSError as error:
        raise SpeechCommandsError(f"cannot publish {path}: {error}") from error
    finally:
        temporary.unlink(missing_ok=True)


def build_speech_commands_manifest(
    root: Path,
    output: Path,
    *,
    source_identity: SourceIdentity | None = SPEECH_COMMANDS_V002,
    archive: Path | None = None,
) -> SpeechCommandsManifest:
    """Write the JSONL rows and a small reviewable summary."""

    inspection = inspect_speech_commands(root, source_identity=source_identity)
    rows = _manifest_bytes(inspection)
    archive_sha256 = _hash_file(archive) if archive is not None else None
    if (
        archive_sha256 is not None
        and source_identity is not None
        and archive_sha256 != source_identity.archive_sha256
    ):
        raise SpeechCommandsError(
            f"archive does not match the registered source: observed={archive_sha256}"
        )
    manifest = SpeechCommandsManifest(
        records=inspection.records,
        background=inspection.background,
        jsonl_sha256=hashlib.sha256(rows).hexdigest(),
        source=inspection.source,
        validation_list_sha256=inspection.validation_list_sha256,
        testing_list_sha256=inspection.testing_list_sha256,
        inventory_sha256=inspection.inventory_sha256,
        archive_sha256=archive_sha256,
    )
    output = _prepare_new_output_path(output)
    staging = Path(
        tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.", suffix=".tmp")
    )
    try:
        _atomic_write(staging / "speech-commands.jsonl", rows)
        _atomic_write(
            staging / "speech-commands-summary.json",
            (json.dumps(manifest.summary(), indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            ),
        )
        staging.replace(output)
    except OSError as error:
        raise SpeechCommandsError(
            f"cannot publish output directory {output}: {error}"
        ) from error
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit Speech Commands v0.02 and write the experiment manifest."
    )
    parser.add_argument("root", type=Path, help="extracted speech_commands_v0.02 root")
    parser.add_argument("output", type=Path, help="directory for generated manifests")
    parser.add_argument(
        "--archive",
        type=Path,
        help="optional original archive to verify against the registered SHA-256",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = build_speech_commands_manifest(
        args.root, args.output, archive=args.archive
    )
    print(json.dumps(manifest.summary(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
