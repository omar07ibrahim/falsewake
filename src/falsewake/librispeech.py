"""Validate the container and member inventory of the LibriSpeech dev archive."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import stat
import sys
import tarfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Literal

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
_SAFE_COMPONENT = re.compile(r"\A[A-Za-z0-9._-]+\Z", re.ASCII)

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
    directory_count: int
    regular_bytes: int
    members: tuple[ArchiveMember, ...]

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
) -> tuple[int, int, tuple[ArchiveMember, ...]]:
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

    ordered = tuple(sorted(members, key=lambda member: member.path.encode("utf-8")))
    return len(directory_paths), regular_bytes, ordered


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
                        f"expected={source_identity.archive_bytes}, "
                        f"observed={archive_bytes}"
                    )
                if not hmac.compare_digest(archive_md5, source_identity.archive_md5):
                    raise LibriSpeechError(
                        "archive MD5 differs from the registered source: "
                        f"observed={archive_md5}"
                    )
                if not hmac.compare_digest(
                    archive_sha256, source_identity.archive_sha256
                ):
                    raise LibriSpeechError(
                        "archive SHA-256 differs from the registered source: "
                        f"observed={archive_sha256}"
                    )

            stream.seek(0)
            try:
                with tarfile.open(fileobj=stream, mode="r:gz") as archive:
                    directory_count, regular_bytes, members = _header_inventory(archive)
            except (tarfile.TarError, OSError, EOFError) as error:
                raise LibriSpeechError(f"cannot open tar archive: {error}") from error

            try:
                stream.seek(0)
                repeated_identity = _archive_digests(stream)
                final_snapshot = _file_snapshot(os.fstat(stream.fileno()))
            except OSError as error:
                raise LibriSpeechError(
                    f"cannot recheck archive identity: {error}"
                ) from error
            if (
                repeated_identity
                != (
                    archive_bytes,
                    archive_md5,
                    archive_sha256,
                )
                or final_snapshot != initial_snapshot
            ):
                raise LibriSpeechError("archive changed during header inspection")
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    return ArchiveInspection(
        archive_bytes=archive_bytes,
        archive_md5=archive_md5,
        archive_sha256=archive_sha256,
        source=source_identity.name if source_identity is not None else "unverified",
        directory_count=directory_count,
        regular_bytes=regular_bytes,
        members=members,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the registered LibriSpeech dev-clean tar headers."
    )
    parser.add_argument("archive", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Print a deterministic header inventory for the registered dev archive."""

    arguments = _parser().parse_args(argv)
    try:
        inspection = inspect_librispeech_archive(arguments.archive)
    except LibriSpeechError as error:
        print(f"LibriSpeech inspection failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(inspection.summary(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
