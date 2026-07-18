"""Validate and audit the registered LibriSpeech development archive."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import hmac
import io
import json
import os
import re
import shutil
import stat
import struct
import sys
import tarfile
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import IO, BinaryIO, Literal, cast

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]
from numpy.typing import NDArray

MAX_ARCHIVE_BYTES = 1024 * 1024 * 1024
MAX_MEMBER_COUNT = 100_000
MAX_REGULAR_BYTES = 8 * 1024 * 1024 * 1024
MAX_FLAC_BYTES = 64 * 1024 * 1024
MAX_TRANSCRIPT_BYTES = 4 * 1024 * 1024
MAX_METADATA_BYTES = 64 * 1024 * 1024

METADATA_PATHS = (
    "LibriSpeech/BOOKS.TXT",
    "LibriSpeech/CHAPTERS.TXT",
    "LibriSpeech/LICENSE.TXT",
    "LibriSpeech/README.TXT",
    "LibriSpeech/SPEAKERS.TXT",
)

_FLAC_PATTERN = re.compile(
    r"\ALibriSpeech/dev-clean/([1-9][0-9]*)/([1-9][0-9]*)/"
    r"\1-\2-([0-9]{4})\.flac\Z",
    re.ASCII,
)
_TRANSCRIPT_PATTERN = re.compile(
    r"\ALibriSpeech/dev-clean/([1-9][0-9]*)/([1-9][0-9]*)/"
    r"\1-\2\.trans\.txt\Z",
    re.ASCII,
)
_UTTERANCE_ID_PATTERN = re.compile(
    r"\A([1-9][0-9]*)-([1-9][0-9]*)-([0-9]{4})\Z", re.ASCII
)
_SAFE_COMPONENT = re.compile(r"\A[A-Za-z0-9._-]+\Z", re.ASCII)

_RAW_FLAC_INVENTORY_DOMAIN = b"falsewake-exp001-librispeech-raw-flac-inventory-v1\0"
_DECODED_PCM16LE_INVENTORY_DOMAIN = (
    b"falsewake-exp001-librispeech-decoded-pcm16le-inventory-v1\0"
)
_TRANSCRIPT_INVENTORY_DOMAIN = b"falsewake-exp001-librispeech-transcript-inventory-v1\0"
_METADATA_INVENTORY_DOMAIN = b"falsewake-exp001-librispeech-metadata-inventory-v1\0"

_AT_FDCWD = -100
_RENAME_NOREPLACE = 1

MemberKind = Literal["flac", "transcript", "metadata"]


class LibriSpeechError(ValueError):
    """An archive cannot satisfy the registered LibriSpeech contract."""


def _require_digest(value: str, *, length: int, name: str) -> None:
    if len(value) != length or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} is not a lowercase hexadecimal digest")


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    name: str
    archive_bytes: int
    archive_md5: str
    archive_sha256: str

    def __post_init__(self) -> None:
        if not self.name or self.archive_bytes < 1:
            raise ValueError("source identity name and size must be non-empty")
        _require_digest(self.archive_md5, length=32, name="archive_md5")
        _require_digest(self.archive_sha256, length=64, name="archive_sha256")


DEV_CLEAN = SourceIdentity(
    name="LibriSpeech dev-clean",
    archive_bytes=337_926_286,
    archive_md5="42e2234ba48799c1f50f24a7926300a1",
    archive_sha256=("76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3"),
)


@dataclass(frozen=True, slots=True)
class ArchiveMember:
    path: str
    size: int
    kind: MemberKind
    speaker_id: str | None = None
    chapter_id: str | None = None
    utterance_id: str | None = None


@dataclass(frozen=True, slots=True)
class ArchiveInspection:
    archive_bytes: int
    archive_md5: str
    archive_sha256: str
    source: str
    directory_paths: tuple[str, ...]
    regular_bytes: int
    members: tuple[ArchiveMember, ...]

    @property
    def directory_count(self) -> int:
        """Return the exact number of validated directory headers."""

        return len(self.directory_paths)

    def summary(self) -> dict[str, object]:
        kinds = tuple(member.kind for member in self.members)
        speakers = {
            member.speaker_id
            for member in self.members
            if member.speaker_id is not None
        }
        chapters = {
            (member.speaker_id, member.chapter_id)
            for member in self.members
            if member.kind == "flac"
        }
        return {
            "archive_bytes": self.archive_bytes,
            "archive_md5": self.archive_md5,
            "archive_sha256": self.archive_sha256,
            "chapter_count": len(chapters),
            "directory_count": self.directory_count,
            "flac_count": kinds.count("flac"),
            "member_count": self.directory_count + len(self.members),
            "metadata_count": kinds.count("metadata"),
            "regular_bytes": self.regular_bytes,
            "regular_count": len(self.members),
            "schema_version": 1,
            "source": self.source,
            "speaker_count": len(speakers),
            "transcript_count": kinds.count("transcript"),
        }


@dataclass(frozen=True, slots=True)
class LibriSpeechUtterance:
    """One canonical manifest row bound to source and decoded payloads."""

    utterance_id: str
    relative_path: str
    speaker_id: str
    chapter_id: str
    sample_count: int
    raw_flac_sha256: str
    decoded_pcm16le_sha256: str
    transcript: str


@dataclass(frozen=True, slots=True)
class LibriSpeechAudit:
    """Deterministic whole-population payload audit."""

    inspection: ArchiveInspection
    utterances: tuple[LibriSpeechUtterance, ...]
    manifest_sha256: str
    raw_flac_inventory_sha256: str
    decoded_pcm16le_inventory_sha256: str
    transcript_inventory_sha256: str
    metadata_inventory_sha256: str
    source_samples: int
    soundfile_version: str
    libsndfile_version: str

    def report(self) -> dict[str, object]:
        """Return the stable report document without paths, clocks, or timings."""

        speakers = {row.speaker_id for row in self.utterances}
        chapters = {(row.speaker_id, row.chapter_id) for row in self.utterances}
        return {
            "archive_bytes": self.inspection.archive_bytes,
            "archive_md5": self.inspection.archive_md5,
            "archive_sha256": self.inspection.archive_sha256,
            "chapter_count": len(chapters),
            "decoder_versions": {
                "libsndfile": self.libsndfile_version,
                "soundfile": self.soundfile_version,
            },
            "decoded_pcm16le_inventory_sha256": (self.decoded_pcm16le_inventory_sha256),
            "flac_count": len(self.utterances),
            "manifest_sha256": self.manifest_sha256,
            "metadata_count": sum(
                member.kind == "metadata" for member in self.inspection.members
            ),
            "metadata_inventory_sha256": self.metadata_inventory_sha256,
            "raw_flac_inventory_sha256": self.raw_flac_inventory_sha256,
            "sample_rate": 16_000,
            "schema_version": 1,
            "source": self.inspection.source,
            "source_duration_hours": self.source_samples / 57_600_000,
            "source_duration_seconds": self.source_samples / 16_000,
            "source_samples": self.source_samples,
            "speaker_count": len(speakers),
            "transcript_count": sum(
                member.kind == "transcript" for member in self.inspection.members
            ),
            "transcript_inventory_sha256": self.transcript_inventory_sha256,
            "utterance_count": len(self.utterances),
        }


def _canonical_member_name(raw_name: str) -> str:
    if not raw_name or "\\" in raw_name or "\0" in raw_name:
        raise LibriSpeechError(f"archive member path is unsafe: {raw_name!r}")
    if not raw_name.isascii():
        raise LibriSpeechError(f"archive member path is not ASCII: {raw_name!r}")
    parsed = PurePosixPath(raw_name)
    if (
        parsed.is_absolute()
        or any(part in {"", ".", ".."} for part in parsed.parts)
        or parsed.as_posix() != raw_name
        or any(_SAFE_COMPONENT.fullmatch(part) is None for part in parsed.parts)
    ):
        raise LibriSpeechError(f"archive member path is not canonical: {raw_name!r}")
    return raw_name


def _classify_regular(path: str, size: int) -> ArchiveMember:
    if size < 1:
        raise LibriSpeechError(f"regular archive member is empty: {path!r}")
    if path in METADATA_PATHS:
        if size > MAX_METADATA_BYTES:
            raise LibriSpeechError(f"metadata member exceeds its size limit: {path!r}")
        return ArchiveMember(path=path, size=size, kind="metadata")

    flac = _FLAC_PATTERN.fullmatch(path)
    if flac is not None:
        if size > MAX_FLAC_BYTES:
            raise LibriSpeechError(f"FLAC member exceeds its size limit: {path!r}")
        speaker_id, chapter_id, utterance_number = flac.groups()
        return ArchiveMember(
            path=path,
            size=size,
            kind="flac",
            speaker_id=speaker_id,
            chapter_id=chapter_id,
            utterance_id=f"{speaker_id}-{chapter_id}-{utterance_number}",
        )

    transcript = _TRANSCRIPT_PATTERN.fullmatch(path)
    if transcript is not None:
        if size > MAX_TRANSCRIPT_BYTES:
            raise LibriSpeechError(
                f"transcript member exceeds its size limit: {path!r}"
            )
        speaker_id, chapter_id = transcript.groups()
        return ArchiveMember(
            path=path,
            size=size,
            kind="transcript",
            speaker_id=speaker_id,
            chapter_id=chapter_id,
        )

    raise LibriSpeechError(f"unexpected regular archive member: {path!r}")


def _header_inventory(
    archive: tarfile.TarFile,
) -> tuple[tuple[str, ...], int, tuple[ArchiveMember, ...]]:
    seen_paths: set[str] = set()
    directory_paths: set[str] = set()
    members: list[ArchiveMember] = []
    regular_bytes = 0

    try:
        for index, header in enumerate(archive, start=1):
            if index > MAX_MEMBER_COUNT:
                raise LibriSpeechError("archive member count exceeds the audit limit")
            path = _canonical_member_name(header.name)
            if path in seen_paths:
                raise LibriSpeechError(f"archive member path is repeated: {path!r}")
            seen_paths.add(path)

            if header.type == tarfile.DIRTYPE:
                if header.size != 0:
                    raise LibriSpeechError(f"archive directory has a payload: {path!r}")
                directory_paths.add(path)
                continue
            if (
                header.type not in {tarfile.REGTYPE, tarfile.AREGTYPE}
                or header.issparse()
            ):
                raise LibriSpeechError(
                    f"archive member is not a plain regular file or directory: {path!r}"
                )

            member = _classify_regular(path, header.size)
            regular_bytes += member.size
            if regular_bytes > MAX_REGULAR_BYTES:
                raise LibriSpeechError("archive payload exceeds the audit size limit")
            members.append(member)
    except (tarfile.TarError, OSError) as error:
        raise LibriSpeechError(f"cannot read archive headers: {error}") from error

    if not members:
        raise LibriSpeechError("archive has no regular members")

    metadata = {member.path for member in members if member.kind == "metadata"}
    missing_metadata = set(METADATA_PATHS) - metadata
    if missing_metadata:
        raise LibriSpeechError(
            f"archive is missing metadata member: {min(missing_metadata)!r}"
        )

    flac_members = [member for member in members if member.kind == "flac"]
    if not flac_members:
        raise LibriSpeechError("archive contains no canonical dev-clean FLAC members")
    flac_chapters = {(member.speaker_id, member.chapter_id) for member in flac_members}
    transcript_chapters = {
        (member.speaker_id, member.chapter_id)
        for member in members
        if member.kind == "transcript"
    }
    if flac_chapters != transcript_chapters:
        missing = flac_chapters - transcript_chapters
        extra = transcript_chapters - flac_chapters
        detail = min(missing or extra)
        problem = "missing" if missing else "unexpected"
        raise LibriSpeechError(
            f"archive has a {problem} chapter transcript: {detail!r}"
        )

    regular_paths = {member.path for member in members}
    expected_directories = {
        ancestor.as_posix()
        for path in regular_paths
        for ancestor in PurePosixPath(path).parents
        if ancestor != PurePosixPath(".")
    }
    unexpected_directories = directory_paths - expected_directories
    if unexpected_directories:
        raise LibriSpeechError(
            f"archive has an unexpected directory: {min(unexpected_directories)!r}"
        )

    ordered_directories = tuple(
        sorted(directory_paths, key=lambda path: path.encode("utf-8"))
    )
    ordered = tuple(sorted(members, key=lambda member: member.path.encode("utf-8")))
    return ordered_directories, regular_bytes, ordered


def _archive_digests(stream: BinaryIO) -> tuple[int, str, str]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    byte_count = 0
    while True:
        chunk = stream.read(1024 * 1024)
        if not chunk:
            break
        if not isinstance(chunk, bytes):
            raise LibriSpeechError("archive stream did not return bytes")
        byte_count += len(chunk)
        if byte_count > MAX_ARCHIVE_BYTES:
            raise LibriSpeechError("archive exceeds the audit size limit")
        md5.update(chunk)
        sha256.update(chunk)
    return byte_count, md5.hexdigest(), sha256.hexdigest()


def _file_snapshot(status: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _inspect_open_stream(
    stream: BinaryIO,
    status: os.stat_result,
    source_identity: SourceIdentity | None,
) -> tuple[ArchiveInspection, tuple[int, int, int, int, int, int]]:
    if not stat.S_ISREG(status.st_mode):
        raise LibriSpeechError("archive must be a regular non-symlink file")
    initial_snapshot = _file_snapshot(status)
    archive_bytes, archive_md5, archive_sha256 = _archive_digests(stream)
    if status.st_size != archive_bytes:
        raise LibriSpeechError("archive size changed while hashing")
    if source_identity is not None:
        if archive_bytes != source_identity.archive_bytes:
            raise LibriSpeechError(
                "archive size differs from the registered source: "
                f"expected={source_identity.archive_bytes}, observed={archive_bytes}"
            )
        if not hmac.compare_digest(archive_md5, source_identity.archive_md5):
            raise LibriSpeechError(
                "archive MD5 differs from the registered source: "
                f"observed={archive_md5}"
            )
        if not hmac.compare_digest(archive_sha256, source_identity.archive_sha256):
            raise LibriSpeechError(
                "archive SHA-256 differs from the registered source: "
                f"observed={archive_sha256}"
            )

    stream.seek(0)
    try:
        with tarfile.open(fileobj=stream, mode="r:gz") as archive:
            directory_paths, regular_bytes, members = _header_inventory(archive)
    except (tarfile.TarError, OSError, EOFError) as error:
        raise LibriSpeechError(f"cannot open tar archive: {error}") from error

    try:
        stream.seek(0)
        repeated_identity = _archive_digests(stream)
        final_snapshot = _file_snapshot(os.fstat(stream.fileno()))
    except OSError as error:
        raise LibriSpeechError(f"cannot recheck archive identity: {error}") from error
    if (
        repeated_identity != (archive_bytes, archive_md5, archive_sha256)
        or final_snapshot != initial_snapshot
    ):
        raise LibriSpeechError("archive changed during header inspection")

    return (
        ArchiveInspection(
            archive_bytes=archive_bytes,
            archive_md5=archive_md5,
            archive_sha256=archive_sha256,
            source=(
                source_identity.name if source_identity is not None else "unverified"
            ),
            directory_paths=directory_paths,
            regular_bytes=regular_bytes,
            members=members,
        ),
        initial_snapshot,
    )


def inspect_librispeech_archive(
    path: Path,
    *,
    source_identity: SourceIdentity | None = DEV_CLEAN,
) -> ArchiveInspection:
    """Hash one archive and validate its headers without opening member payloads."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise LibriSpeechError(f"cannot open archive: {error}") from error

    try:
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            status = os.fstat(stream.fileno())
            inspection, _ = _inspect_open_stream(stream, status, source_identity)
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    return inspection


def _read_member_payload(stream: IO[bytes], member: ArchiveMember) -> bytes:
    payload = bytearray()
    try:
        while len(payload) < member.size:
            chunk = stream.read(min(1024 * 1024, member.size - len(payload)))
            if not chunk:
                break
            if not isinstance(chunk, bytes):
                raise LibriSpeechError(
                    f"archive payload stream did not return bytes: {member.path!r}"
                )
            payload.extend(chunk)
        trailing = stream.read(1)
    except (tarfile.TarError, OSError, EOFError) as error:
        raise LibriSpeechError(
            f"cannot read archive payload {member.path!r}: {error}"
        ) from error
    if len(payload) != member.size or trailing:
        raise LibriSpeechError(
            f"archive payload size differs from its header: {member.path!r}"
        )
    return bytes(payload)


def _decode_flac(payload: bytes, path: str) -> tuple[int, str]:
    decoded_sha256 = hashlib.sha256()
    sample_count = 0
    try:
        with sf.SoundFile(io.BytesIO(payload), mode="r") as decoder:
            if (
                decoder.format != "FLAC"
                or decoder.subtype != "PCM_16"
                or decoder.channels != 1
                or decoder.samplerate != 16_000
            ):
                raise LibriSpeechError(
                    "unsupported FLAC format for "
                    f"{path!r}: format={decoder.format}, subtype={decoder.subtype}, "
                    f"channels={decoder.channels}, sample_rate={decoder.samplerate}"
                )
            declared_frames = int(decoder.frames)
            if declared_frames < 1:
                raise LibriSpeechError(f"FLAC has no decoded samples: {path!r}")
            while True:
                samples = cast(
                    NDArray[np.int16],
                    decoder.read(65_536, dtype="int16", always_2d=False),
                )
                if samples.size == 0:
                    break
                if samples.ndim != 1 or samples.dtype != np.dtype(np.int16):
                    raise LibriSpeechError(
                        f"FLAC decoder returned non-mono PCM16 samples: {path!r}"
                    )
                canonical = np.asarray(samples, dtype=np.dtype("<i2"))
                decoded_sha256.update(canonical.tobytes(order="C"))
                sample_count += int(samples.size)
            if sample_count != declared_frames:
                raise LibriSpeechError(
                    "decoded FLAC sample count differs from its stream metadata: "
                    f"{path!r}, expected={declared_frames}, observed={sample_count}"
                )
    except LibriSpeechError:
        raise
    except (OSError, RuntimeError, ValueError, EOFError) as error:
        raise LibriSpeechError(
            f"cannot decode FLAC payload {path!r}: {error}"
        ) from error
    return sample_count, decoded_sha256.hexdigest()


def _parse_transcript(payload: bytes, member: ArchiveMember) -> dict[str, str]:
    if b"\r" in payload:
        raise LibriSpeechError(f"transcript contains a CR byte: {member.path!r}")
    if not payload.endswith(b"\n"):
        raise LibriSpeechError(
            f"transcript is not terminated by an LF byte: {member.path!r}"
        )
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise LibriSpeechError(
            f"transcript is not strict UTF-8: {member.path!r}"
        ) from error

    records: dict[str, str] = {}
    for line_number, line in enumerate(text[:-1].split("\n"), start=1):
        identifier, separator, transcript = line.partition(" ")
        match = _UTTERANCE_ID_PATTERN.fullmatch(identifier)
        if (
            not separator
            or not transcript
            or transcript.startswith(" ")
            or match is None
        ):
            raise LibriSpeechError(
                "transcript line must be '<utterance-id><space><nonempty text>': "
                f"{member.path!r}:{line_number}"
            )
        speaker_id, chapter_id, _ = match.groups()
        if speaker_id != member.speaker_id or chapter_id != member.chapter_id:
            raise LibriSpeechError(
                "transcript line belongs to a different chapter: "
                f"{member.path!r}:{line_number}"
            )
        if identifier in records:
            raise LibriSpeechError(
                f"transcript repeats utterance ID {identifier!r}: {member.path!r}"
            )
        records[identifier] = transcript
    return records


def _inventory_sha256(domain: bytes, entries: Sequence[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256(domain)
    for path, payload in sorted(entries, key=lambda item: item[0].encode("utf-8")):
        path_bytes = path.encode("utf-8")
        digest.update(struct.pack("<I", len(path_bytes)))
        digest.update(path_bytes)
        digest.update(struct.pack("<Q", len(payload)))
        digest.update(payload)
    return digest.hexdigest()


def _manifest_bytes(utterances: Sequence[LibriSpeechUtterance]) -> bytes:
    return b"".join(
        (
            json.dumps(
                asdict(utterance),
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        for utterance in utterances
    )


def _payload_inventory(
    archive: tarfile.TarFile, inspection: ArchiveInspection
) -> LibriSpeechAudit:
    expected = {member.path: member for member in inspection.members}
    expected_directories = set(inspection.directory_paths)
    seen: set[str] = set()
    seen_directories: set[str] = set()
    flacs: dict[str, tuple[ArchiveMember, int, str, str]] = {}
    transcripts: dict[str, str] = {}
    raw_flac_entries: list[tuple[str, bytes]] = []
    decoded_entries: list[tuple[str, bytes]] = []
    transcript_entries: list[tuple[str, bytes]] = []
    metadata_entries: list[tuple[str, bytes]] = []

    try:
        for header in archive:
            path = _canonical_member_name(header.name)
            if header.type == tarfile.DIRTYPE:
                if (
                    header.size != 0
                    or path in seen_directories
                    or path not in expected_directories
                ):
                    raise LibriSpeechError(
                        f"payload pass differs from validated directories: {path!r}"
                    )
                seen_directories.add(path)
                continue
            member = expected.get(path)
            if (
                member is None
                or path in seen
                or header.size != member.size
                or header.type not in {tarfile.REGTYPE, tarfile.AREGTYPE}
                or header.issparse()
            ):
                raise LibriSpeechError(
                    f"payload pass differs from validated headers: {path!r}"
                )
            seen.add(path)
            extracted = archive.extractfile(header)
            if extracted is None:
                raise LibriSpeechError(f"cannot open archive payload: {path!r}")
            with extracted:
                payload = _read_member_payload(extracted, member)

            if member.kind == "metadata":
                metadata_entries.append((path, payload))
                continue
            if member.kind == "transcript":
                transcript_entries.append((path, payload))
                for transcript_id, transcript in _parse_transcript(
                    payload, member
                ).items():
                    if transcript_id in transcripts:
                        raise LibriSpeechError(
                            f"transcripts repeat utterance ID {transcript_id!r}"
                        )
                    transcripts[transcript_id] = transcript
                continue

            utterance_id = member.utterance_id
            if utterance_id is None:
                raise LibriSpeechError(f"FLAC has no canonical utterance ID: {path!r}")
            if utterance_id in flacs:
                raise LibriSpeechError(
                    f"FLAC utterance ID is repeated: {utterance_id!r}"
                )
            sample_count, decoded_sha256 = _decode_flac(payload, path)
            raw_sha256 = hashlib.sha256(payload).hexdigest()
            flacs[utterance_id] = (
                member,
                sample_count,
                raw_sha256,
                decoded_sha256,
            )
            raw_flac_entries.append((path, bytes.fromhex(raw_sha256)))
            decoded_entries.append(
                (
                    path,
                    struct.pack("<Q", sample_count) + bytes.fromhex(decoded_sha256),
                )
            )
    except (tarfile.TarError, OSError, EOFError) as error:
        raise LibriSpeechError(f"cannot read archive payload pass: {error}") from error

    if seen != set(expected) or seen_directories != expected_directories:
        raise LibriSpeechError("payload pass does not match the validated member union")
    missing_transcripts = set(flacs) - set(transcripts)
    extra_transcripts = set(transcripts) - set(flacs)
    if missing_transcripts:
        raise LibriSpeechError(
            "transcript/FLAC bijection is missing utterance ID "
            f"{min(missing_transcripts)!r}"
        )
    if extra_transcripts:
        raise LibriSpeechError(
            "transcript/FLAC bijection has unexpected utterance ID "
            f"{min(extra_transcripts)!r}"
        )

    rows: list[LibriSpeechUtterance] = []
    for utterance_id, (
        member,
        sample_count,
        raw_sha256,
        decoded_sha256,
    ) in flacs.items():
        if member.speaker_id is None or member.chapter_id is None:
            raise LibriSpeechError(
                f"FLAC has incomplete canonical IDs: {member.path!r}"
            )
        rows.append(
            LibriSpeechUtterance(
                utterance_id=utterance_id,
                relative_path=member.path,
                speaker_id=member.speaker_id,
                chapter_id=member.chapter_id,
                sample_count=sample_count,
                raw_flac_sha256=raw_sha256,
                decoded_pcm16le_sha256=decoded_sha256,
                transcript=transcripts[utterance_id],
            )
        )
    utterances = tuple(
        sorted(rows, key=lambda utterance: utterance.relative_path.encode("utf-8"))
    )
    manifest = _manifest_bytes(utterances)
    return LibriSpeechAudit(
        inspection=inspection,
        utterances=utterances,
        manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        raw_flac_inventory_sha256=_inventory_sha256(
            _RAW_FLAC_INVENTORY_DOMAIN, raw_flac_entries
        ),
        decoded_pcm16le_inventory_sha256=_inventory_sha256(
            _DECODED_PCM16LE_INVENTORY_DOMAIN, decoded_entries
        ),
        transcript_inventory_sha256=_inventory_sha256(
            _TRANSCRIPT_INVENTORY_DOMAIN, transcript_entries
        ),
        metadata_inventory_sha256=_inventory_sha256(
            _METADATA_INVENTORY_DOMAIN, metadata_entries
        ),
        source_samples=sum(utterance.sample_count for utterance in utterances),
        soundfile_version=str(sf.__version__),
        libsndfile_version=str(sf.__libsndfile_version__),
    )


def audit_librispeech_archive(
    path: Path,
    *,
    source_identity: SourceIdentity | None = DEV_CLEAN,
) -> LibriSpeechAudit:
    """Audit every allowed payload through one descriptor without extraction."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise LibriSpeechError(f"cannot open archive: {error}") from error

    try:
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            status = os.fstat(stream.fileno())
            inspection, initial_snapshot = _inspect_open_stream(
                stream, status, source_identity
            )
            stream.seek(0)
            try:
                with tarfile.open(fileobj=stream, mode="r:gz") as archive:
                    audit = _payload_inventory(archive, inspection)
            except (tarfile.TarError, OSError, EOFError) as error:
                raise LibriSpeechError(
                    f"cannot open tar archive for payload audit: {error}"
                ) from error

            try:
                stream.seek(0)
                final_identity = _archive_digests(stream)
                final_snapshot = _file_snapshot(os.fstat(stream.fileno()))
            except OSError as error:
                raise LibriSpeechError(
                    f"cannot recheck archive after payload audit: {error}"
                ) from error
            expected_identity = (
                inspection.archive_bytes,
                inspection.archive_md5,
                inspection.archive_sha256,
            )
            if (
                final_identity != expected_identity
                or final_snapshot != initial_snapshot
            ):
                raise LibriSpeechError("archive changed during payload audit")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return audit


def _prepare_new_output_path(output: Path) -> Path:
    output = output.absolute()
    current = Path(output.anchor)
    for part in output.parent.parts[1:]:
        current /= part
        if current.exists() and current.is_symlink():
            raise LibriSpeechError(f"output path must not contain symlinks: {output}")
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise LibriSpeechError(
            f"cannot create output parent {output.parent}: {error}"
        ) from error
    if output.exists() or output.is_symlink():
        raise LibriSpeechError(f"output path already exists: {output}")
    return output


def _write_new_file(path: Path, contents: bytes) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(contents)
            destination.flush()
            os.fsync(destination.fileno())
    except OSError as error:
        raise LibriSpeechError(
            f"cannot write staged audit file {path.name}: {error}"
        ) from error


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as error:
        raise LibriSpeechError(
            "atomic no-replace directory publication is unavailable"
        ) from error
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise LibriSpeechError(f"output path already exists: {destination}")
    raise LibriSpeechError(
        f"cannot atomically publish output directory: {os.strerror(error_number)}"
    )


def build_librispeech_audit(
    archive: Path,
    output: Path,
    *,
    source_identity: SourceIdentity | None = DEV_CLEAN,
) -> LibriSpeechAudit:
    """Audit an archive and atomically publish its JSONL manifest and report."""

    output = _prepare_new_output_path(output)
    audit = audit_librispeech_archive(archive, source_identity=source_identity)
    manifest = _manifest_bytes(audit.utterances)
    if hashlib.sha256(manifest).hexdigest() != audit.manifest_sha256:
        raise LibriSpeechError("manifest changed between audit and publication")
    report = (
        json.dumps(
            audit.report(),
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")

    staging = Path(
        tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.", suffix=".tmp")
    )
    published = False
    try:
        _write_new_file(staging / "dev-clean.manifest.jsonl", manifest)
        _write_new_file(staging / "dev-clean.audit.json", report)
        _fsync_directory(staging)
        _rename_directory_noreplace(staging, output)
        published = True
        _fsync_directory(output.parent)
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return audit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate or fully audit the registered LibriSpeech dev archive."
    )
    parser.add_argument("archive", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="write a full payload manifest and report to this new directory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Print a deterministic header summary or publish a full payload audit."""

    arguments = _parser().parse_args(argv)
    try:
        if arguments.output is not None:
            audit = build_librispeech_audit(arguments.archive, arguments.output)
            document = audit.report()
        else:
            inspection = inspect_librispeech_archive(arguments.archive)
            document = inspection.summary()
    except LibriSpeechError as error:
        print(f"LibriSpeech inspection failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
