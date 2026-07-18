from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

import falsewake.librispeech as librispeech
from falsewake.librispeech import (
    DEV_CLEAN,
    METADATA_PATHS,
    LibriSpeechError,
    SourceIdentity,
    _canonical_member_name,
    inspect_librispeech_archive,
)


def _regular(name: str, contents: bytes = b"payload") -> tuple[tarfile.TarInfo, bytes]:
    member = tarfile.TarInfo(name)
    member.size = len(contents)
    return member, contents


def _special(name: str, member_type: bytes) -> tuple[tarfile.TarInfo, bytes]:
    member = tarfile.TarInfo(name)
    member.type = member_type
    member.size = 0
    return member, b""


def _valid_members() -> list[tuple[tarfile.TarInfo, bytes]]:
    members = [_regular(path, path.encode("ascii")) for path in METADATA_PATHS]
    members.extend(
        [
            _regular(
                "LibriSpeech/dev-clean/1272/128104/1272-128104.trans.txt",
                b"1272-128104-0000 A TEST UTTERANCE\n",
            ),
            _regular(
                "LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac",
                b"fLaCfake",
            ),
            _regular(
                "LibriSpeech/dev-clean/1272/128104/1272-128104-0001.flac",
                b"fLaCmore",
            ),
        ]
    )
    return members


def _write_archive(path: Path, members: list[tuple[tarfile.TarInfo, bytes]]) -> Path:
    with tarfile.open(path, "w:gz", format=tarfile.USTAR_FORMAT) as archive:
        for member, contents in members:
            archive.addfile(member, io.BytesIO(contents) if member.isreg() else None)
    return path


def _identity(path: Path) -> SourceIdentity:
    contents = path.read_bytes()
    return SourceIdentity(
        name="synthetic dev-clean",
        archive_bytes=len(contents),
        archive_md5=hashlib.md5(contents, usedforsecurity=False).hexdigest(),
        archive_sha256=hashlib.sha256(contents).hexdigest(),
    )


def test_header_inspection_is_stable_and_covers_the_exact_union(tmp_path: Path) -> None:
    archive = _write_archive(tmp_path / "valid.tar.gz", _valid_members())
    identity = _identity(archive)

    first = inspect_librispeech_archive(archive, source_identity=identity)
    second = inspect_librispeech_archive(archive, source_identity=identity)

    assert first == second
    assert [member.path for member in first.members] == sorted(
        member.path for member in first.members
    )
    assert first.summary() == {
        "archive_bytes": archive.stat().st_size,
        "archive_md5": identity.archive_md5,
        "archive_sha256": identity.archive_sha256,
        "chapter_count": 1,
        "directory_count": 0,
        "flac_count": 2,
        "member_count": 8,
        "metadata_count": 5,
        "regular_bytes": sum(member.size for member in first.members),
        "regular_count": 8,
        "schema_version": 1,
        "source": "synthetic dev-clean",
        "speaker_count": 1,
        "transcript_count": 1,
    }
    flac = next(member for member in first.members if member.kind == "flac")
    assert flac.utterance_id == "1272-128104-0000"


def test_registered_source_identity_matches_the_protocol() -> None:
    config = json.loads(
        Path("configs/experiment-001.json").read_text(encoding="utf-8")
    )["negative_source"]

    assert DEV_CLEAN.archive_bytes == config["archive_bytes"]
    assert DEV_CLEAN.archive_md5 == config["archive_md5"]
    assert DEV_CLEAN.archive_sha256 == config["archive_sha256"]


def test_canonical_directory_entries_and_legacy_regular_type_are_allowed(
    tmp_path: Path,
) -> None:
    directories = [
        _special("LibriSpeech", tarfile.DIRTYPE),
        _special("LibriSpeech/dev-clean", tarfile.DIRTYPE),
        _special("LibriSpeech/dev-clean/1272", tarfile.DIRTYPE),
        _special("LibriSpeech/dev-clean/1272/128104", tarfile.DIRTYPE),
    ]
    members = directories + _valid_members()
    legacy = next(
        member for member, _ in members if member.name.endswith("1272-128104-0001.flac")
    )
    legacy.type = tarfile.AREGTYPE
    archive = _write_archive(tmp_path / "directories.tar.gz", members)

    inspection = inspect_librispeech_archive(archive, source_identity=None)

    assert inspection.directory_count == 4
    assert inspection.directory_paths == tuple(member.name for member, _ in directories)
    assert inspection.summary()["member_count"] == 12


@pytest.mark.parametrize(
    "name",
    [
        "/LibriSpeech/dev-clean/1/2/1-2-0000.flac",
        "../LibriSpeech/dev-clean/1/2/1-2-0000.flac",
        "LibriSpeech/dev-clean/1/../2/1-2-0000.flac",
        "LibriSpeech//dev-clean/1/2/1-2-0000.flac",
        "LibriSpeech\\dev-clean\\1\\2\\1-2-0000.flac",
        "LibriSpeech/dev-clean/1/2/bad name.flac",
        "LibriSpeech/dev-clean/1/2/bad\0name.flac",
        "LibriSpeech/dev-clean/é/2/é-2-0000.flac",
    ],
)
def test_unsafe_or_noncanonical_member_names_are_rejected(name: str) -> None:
    with pytest.raises(LibriSpeechError):
        _canonical_member_name(name)


def test_duplicate_member_path_is_rejected(tmp_path: Path) -> None:
    members = _valid_members()
    members.append(_regular("LibriSpeech/README.TXT", b"duplicate"))
    archive = _write_archive(tmp_path / "duplicate.tar.gz", members)

    with pytest.raises(LibriSpeechError, match="repeated"):
        inspect_librispeech_archive(archive, source_identity=None)


@pytest.mark.parametrize(
    "member_type",
    [
        tarfile.SYMTYPE,
        tarfile.LNKTYPE,
        tarfile.CHRTYPE,
        tarfile.BLKTYPE,
        tarfile.FIFOTYPE,
        tarfile.GNUTYPE_SPARSE,
        tarfile.CONTTYPE,
        b"Z",
    ],
)
def test_non_plain_member_types_are_rejected(
    tmp_path: Path, member_type: bytes
) -> None:
    members = [
        item for item in _valid_members() if item[0].name != "LibriSpeech/README.TXT"
    ]
    members.append(_special("LibriSpeech/README.TXT", member_type))
    archive = _write_archive(tmp_path / f"type-{member_type.hex()}.tar.gz", members)

    with pytest.raises(LibriSpeechError, match="plain regular"):
        inspect_librispeech_archive(archive, source_identity=None)


def test_unexpected_regular_member_is_rejected(tmp_path: Path) -> None:
    members = _valid_members()
    members.append(_regular("LibriSpeech/dev-clean/README.txt"))
    archive = _write_archive(tmp_path / "unexpected.tar.gz", members)

    with pytest.raises(LibriSpeechError, match="unexpected regular"):
        inspect_librispeech_archive(archive, source_identity=None)


def test_noncanonical_utterance_width_is_rejected(tmp_path: Path) -> None:
    members = [
        item
        for item in _valid_members()
        if not item[0].name.endswith("1272-128104-0001.flac")
    ]
    members.append(_regular("LibriSpeech/dev-clean/1272/128104/1272-128104-1.flac"))
    archive = _write_archive(tmp_path / "utterance-width.tar.gz", members)

    with pytest.raises(LibriSpeechError, match="unexpected regular"):
        inspect_librispeech_archive(archive, source_identity=None)


def test_missing_or_extra_chapter_transcript_is_rejected(tmp_path: Path) -> None:
    missing = [
        item for item in _valid_members() if not item[0].name.endswith(".trans.txt")
    ]
    missing_archive = _write_archive(tmp_path / "missing.tar.gz", missing)
    with pytest.raises(LibriSpeechError, match="missing chapter transcript"):
        inspect_librispeech_archive(missing_archive, source_identity=None)

    extra = _valid_members()
    extra.append(
        _regular("LibriSpeech/dev-clean/3/4/3-4.trans.txt", b"3-4-0000 EXTRA\n")
    )
    extra_archive = _write_archive(tmp_path / "extra.tar.gz", extra)
    with pytest.raises(LibriSpeechError, match="unexpected chapter transcript"):
        inspect_librispeech_archive(extra_archive, source_identity=None)


def test_missing_metadata_and_unexpected_directory_are_rejected(tmp_path: Path) -> None:
    missing = [item for item in _valid_members() if item[0].name != METADATA_PATHS[0]]
    missing_archive = _write_archive(tmp_path / "missing-metadata.tar.gz", missing)
    with pytest.raises(LibriSpeechError, match="missing metadata"):
        inspect_librispeech_archive(missing_archive, source_identity=None)

    members = _valid_members()
    members.append(_special("unrelated", tarfile.DIRTYPE))
    directory_archive = _write_archive(tmp_path / "directory.tar.gz", members)
    with pytest.raises(LibriSpeechError, match="unexpected directory"):
        inspect_librispeech_archive(directory_archive, source_identity=None)


def test_archive_identity_and_regular_file_are_required(tmp_path: Path) -> None:
    archive = _write_archive(tmp_path / "valid.tar.gz", _valid_members())
    wrong = SourceIdentity(
        name="wrong",
        archive_bytes=archive.stat().st_size,
        archive_md5="0" * 32,
        archive_sha256="0" * 64,
    )
    with pytest.raises(LibriSpeechError, match="MD5 differs"):
        inspect_librispeech_archive(archive, source_identity=wrong)

    wrong_size = SourceIdentity(
        name="wrong size",
        archive_bytes=archive.stat().st_size + 1,
        archive_md5=_identity(archive).archive_md5,
        archive_sha256=_identity(archive).archive_sha256,
    )
    with pytest.raises(LibriSpeechError, match="size differs"):
        inspect_librispeech_archive(archive, source_identity=wrong_size)

    correct = _identity(archive)
    wrong_sha = SourceIdentity(
        name="wrong SHA",
        archive_bytes=correct.archive_bytes,
        archive_md5=correct.archive_md5,
        archive_sha256="0" * 64,
    )
    with pytest.raises(LibriSpeechError, match="SHA-256 differs"):
        inspect_librispeech_archive(archive, source_identity=wrong_sha)

    link = tmp_path / "archive-link.tar.gz"
    link.symlink_to(archive)
    with pytest.raises(LibriSpeechError, match="cannot open archive"):
        inspect_librispeech_archive(link, source_identity=None)


def test_corrupt_gzip_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "corrupt.tar.gz"
    archive.write_bytes(b"not a gzip archive")

    with pytest.raises(LibriSpeechError, match="cannot open tar archive"):
        inspect_librispeech_archive(archive, source_identity=None)


def test_same_inode_mutation_during_scan_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _write_archive(tmp_path / "mutable.tar.gz", _valid_members())
    original = librispeech._header_inventory

    def inspect_then_mutate(
        opened: tarfile.TarFile,
    ) -> tuple[tuple[str, ...], int, tuple[librispeech.ArchiveMember, ...]]:
        result = original(opened)
        with archive.open("r+b") as stream:
            stream.seek(-1, io.SEEK_END)
            final = stream.read(1)
            stream.seek(-1, io.SEEK_END)
            stream.write(bytes([final[0] ^ 1]))
        return result

    monkeypatch.setattr(librispeech, "_header_inventory", inspect_then_mutate)

    with pytest.raises(LibriSpeechError, match="changed during header inspection"):
        inspect_librispeech_archive(archive, source_identity=None)


def test_header_inspector_never_opens_a_member_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _write_archive(tmp_path / "headers-only.tar.gz", _valid_members())

    def forbidden_extractfile(*args: object, **kwargs: object) -> None:
        raise AssertionError("header inspection must not open member payloads")

    monkeypatch.setattr(tarfile.TarFile, "extractfile", forbidden_extractfile)

    inspect_librispeech_archive(archive, source_identity=None)


def test_registered_wrong_size_is_rejected_before_any_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "wrong-size.tar.gz"
    archive.write_bytes(b"one byte too short")
    identity = SourceIdentity(
        name="synthetic registered source",
        archive_bytes=archive.stat().st_size + 1,
        archive_md5="0" * 32,
        archive_sha256="0" * 64,
    )

    def forbidden_digest(stream: object) -> tuple[int, str, str]:
        raise AssertionError("wrong-size registered archive must not be read")

    monkeypatch.setattr(librispeech, "_archive_digests", forbidden_digest)

    with pytest.raises(LibriSpeechError, match="archive size differs"):
        inspect_librispeech_archive(archive, source_identity=identity)


def test_duplicate_directory_alias_is_rejected(tmp_path: Path) -> None:
    members = [
        _special("LibriSpeech", tarfile.DIRTYPE),
        _special("LibriSpeech/", tarfile.DIRTYPE),
        *_valid_members(),
    ]
    archive = _write_archive(tmp_path / "directory-alias.tar.gz", members)

    with pytest.raises(LibriSpeechError, match="repeated"):
        inspect_librispeech_archive(archive, source_identity=None)


@pytest.mark.parametrize(
    ("limit_name", "limit_value", "message"),
    [
        ("MAX_MEMBER_COUNT", 7, "member count"),
        ("MAX_REGULAR_BYTES", 1, "payload"),
        ("MAX_METADATA_BYTES", 1, "metadata member"),
        ("MAX_ARCHIVE_BYTES", 1, "archive exceeds"),
    ],
)
def test_header_limits_are_enforced_before_payload_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
    limit_value: int,
    message: str,
) -> None:
    archive = _write_archive(tmp_path / f"limit-{limit_name}.tar.gz", _valid_members())
    monkeypatch.setattr(librispeech, limit_name, limit_value)

    with pytest.raises(LibriSpeechError, match=message):
        inspect_librispeech_archive(archive, source_identity=None)
