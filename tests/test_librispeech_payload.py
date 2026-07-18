from __future__ import annotations

import hashlib
import io
import json
import os
import struct
import tarfile
from collections.abc import Sequence
from dataclasses import fields, replace
from pathlib import Path
from typing import IO, BinaryIO

import numpy as np
import pytest
import soundfile as sf  # type: ignore[import-untyped]

import falsewake.librispeech as librispeech
from falsewake.librispeech import (
    METADATA_PATHS,
    LibriSpeechError,
    SourceIdentity,
    audit_librispeech_archive,
    build_librispeech_audit,
)

_TRANSCRIPT_PATH = "LibriSpeech/dev-clean/1272/128104/1272-128104.trans.txt"
_FIRST_FLAC_PATH = "LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac"
_SECOND_FLAC_PATH = "LibriSpeech/dev-clean/1272/128104/1272-128104-0001.flac"
_VALID_TRANSCRIPT = (
    "1272-128104-0000 A CAFÉ TEST\n1272-128104-0001 SECOND UTTERANCE\n".encode()
)


def _regular(name: str, contents: bytes) -> tuple[tarfile.TarInfo, bytes]:
    member = tarfile.TarInfo(name)
    member.size = len(contents)
    return member, contents


def _flac_bytes(
    *,
    sample_rate: int = 16_000,
    channels: int = 1,
    subtype: str = "PCM_16",
    sample_count: int = 480,
) -> bytes:
    base = ((np.arange(sample_count, dtype=np.int32) * 257) % 65_536 - 32_768).astype(
        np.int16
    )
    samples = base if channels == 1 else np.column_stack([base, -base])
    destination = io.BytesIO()
    sf.write(
        destination,
        samples,
        sample_rate,
        format="FLAC",
        subtype=subtype,
    )
    return destination.getvalue()


def _valid_members() -> list[tuple[tarfile.TarInfo, bytes]]:
    members = [_regular(path, path.encode("ascii")) for path in METADATA_PATHS]
    members.extend(
        [
            _regular(_TRANSCRIPT_PATH, _VALID_TRANSCRIPT),
            _regular(_FIRST_FLAC_PATH, _flac_bytes(sample_count=320)),
            _regular(_SECOND_FLAC_PATH, _flac_bytes(sample_count=480)),
        ]
    )
    return members


def _replace_payload(
    members: Sequence[tuple[tarfile.TarInfo, bytes]], name: str, payload: bytes
) -> list[tuple[tarfile.TarInfo, bytes]]:
    return [
        _regular(member.name, payload if member.name == name else contents)
        for member, contents in members
    ]


def _write_archive(
    path: Path, members: Sequence[tuple[tarfile.TarInfo, bytes]]
) -> Path:
    with tarfile.open(path, "w:gz", format=tarfile.USTAR_FORMAT) as archive:
        for member, contents in members:
            archive.addfile(member, io.BytesIO(contents))
    return path


def _identity(path: Path) -> SourceIdentity:
    contents = path.read_bytes()
    return SourceIdentity(
        name="synthetic dev-clean",
        archive_bytes=len(contents),
        archive_md5=hashlib.md5(contents, usedforsecurity=False).hexdigest(),
        archive_sha256=hashlib.sha256(contents).hexdigest(),
    )


def _framed_digest(domain: bytes, entries: Sequence[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256(domain)
    for path, payload in sorted(entries, key=lambda item: item[0].encode()):
        path_bytes = path.encode()
        digest.update(struct.pack("<I", len(path_bytes)))
        digest.update(path_bytes)
        digest.update(struct.pack("<Q", len(payload)))
        digest.update(payload)
    return digest.hexdigest()


def test_payload_audit_is_byte_stable_and_matches_registered_framing(
    tmp_path: Path,
) -> None:
    members = _valid_members()
    archive = _write_archive(tmp_path / "valid.tar.gz", members)
    identity = _identity(archive)

    first = build_librispeech_audit(
        archive, tmp_path / "first", source_identity=identity
    )
    second = build_librispeech_audit(
        archive, tmp_path / "second", source_identity=identity
    )

    assert first == second
    first_manifest = (tmp_path / "first/dev-clean.manifest.jsonl").read_bytes()
    first_report = (tmp_path / "first/dev-clean.audit.json").read_bytes()
    assert first_manifest == (tmp_path / "second/dev-clean.manifest.jsonl").read_bytes()
    assert first_report == (tmp_path / "second/dev-clean.audit.json").read_bytes()
    assert hashlib.sha256(first_manifest).hexdigest() == first.manifest_sha256

    rows = [json.loads(line) for line in first_manifest.splitlines()]
    assert [row["relative_path"] for row in rows] == [
        _FIRST_FLAC_PATH,
        _SECOND_FLAC_PATH,
    ]
    assert rows[0] == {
        "chapter_id": "128104",
        "decoded_pcm16le_sha256": hashlib.sha256(
            np.asarray(
                ((np.arange(320, dtype=np.int32) * 257) % 65_536 - 32_768),
                dtype="<i2",
            ).tobytes()
        ).hexdigest(),
        "raw_flac_sha256": hashlib.sha256(
            next(
                contents
                for member, contents in members
                if member.name == _FIRST_FLAC_PATH
            )
        ).hexdigest(),
        "relative_path": _FIRST_FLAC_PATH,
        "sample_count": 320,
        "speaker_id": "1272",
        "transcript": "A CAFÉ TEST",
        "utterance_id": "1272-128104-0000",
    }
    assert b"CAF\\u00c9" in first_manifest

    flac_payloads = [
        (member.name, contents)
        for member, contents in members
        if member.name.endswith(".flac")
    ]
    raw_entries = [
        (path, hashlib.sha256(contents).digest()) for path, contents in flac_payloads
    ]
    decoded_entries = [
        (
            row["relative_path"],
            struct.pack("<Q", row["sample_count"])
            + bytes.fromhex(row["decoded_pcm16le_sha256"]),
        )
        for row in rows
    ]
    transcript_entries = [(_TRANSCRIPT_PATH, _VALID_TRANSCRIPT)]
    metadata_entries = [(path, path.encode("ascii")) for path in METADATA_PATHS]
    assert first.raw_flac_inventory_sha256 == _framed_digest(
        librispeech._RAW_FLAC_INVENTORY_DOMAIN, raw_entries
    )
    assert first.decoded_pcm16le_inventory_sha256 == _framed_digest(
        librispeech._DECODED_PCM16LE_INVENTORY_DOMAIN, decoded_entries
    )
    assert first.transcript_inventory_sha256 == _framed_digest(
        librispeech._TRANSCRIPT_INVENTORY_DOMAIN, transcript_entries
    )
    assert first.metadata_inventory_sha256 == _framed_digest(
        librispeech._METADATA_INVENTORY_DOMAIN, metadata_entries
    )

    report = json.loads(first_report)
    assert report["source"] == "synthetic dev-clean"
    assert report["source_samples"] == 800
    assert report["source_duration_seconds"] == 0.05
    assert report["utterance_count"] == report["flac_count"] == 2
    assert report["speaker_count"] == report["chapter_count"] == 1
    assert report["decoder_versions"] == {
        "libsndfile": str(sf.__libsndfile_version__),
        "soundfile": str(sf.__version__),
    }
    report_text = first_report.decode()
    assert str(tmp_path) not in report_text
    assert "timestamp" not in report_text
    assert "elapsed" not in report_text


def test_manifest_row_schema_matches_the_registered_contract(tmp_path: Path) -> None:
    archive = _write_archive(tmp_path / "schema.tar.gz", _valid_members())
    audit = audit_librispeech_archive(archive, source_identity=_identity(archive))
    config = json.loads(
        Path("configs/experiment-001.json").read_text(encoding="utf-8")
    )["negative_source"]["archive_audit"]

    assert [field.name for field in fields(audit.utterances[0])] == config[
        "manifest_fields"
    ]
    assert config["manifest_serialization"] == (
        "rows_sorted_by_relative_path;UTF8_json_dumps_sort_keys_true_separators_"
        "comma_colon_ensure_ascii_true_allow_nan_false_plus_LF"
    )
    assert config["output_files"] == {
        "audit_report": "dev-clean.audit.json",
        "manifest": "dev-clean.manifest.jsonl",
    }
    assert config["inventory_domain_hex"] == {
        "decoded_pcm16le": librispeech._DECODED_PCM16LE_INVENTORY_DOMAIN.hex(),
        "metadata": librispeech._METADATA_INVENTORY_DOMAIN.hex(),
        "raw_flac": librispeech._RAW_FLAC_INVENTORY_DOMAIN.hex(),
        "transcript": librispeech._TRANSCRIPT_INVENTORY_DOMAIN.hex(),
    }


def test_payload_pass_revalidates_the_exact_directory_paths(tmp_path: Path) -> None:
    archive = _write_archive(tmp_path / "directories.tar.gz", _valid_members())
    inspection = librispeech.inspect_librispeech_archive(
        archive, source_identity=_identity(archive)
    )
    inconsistent = replace(inspection, directory_paths=("LibriSpeech",))

    with (
        tarfile.open(archive, mode="r:gz") as opened,
        pytest.raises(LibriSpeechError, match="validated member union"),
    ):
        librispeech._payload_inventory(opened, inconsistent)


def test_cli_payload_mode_publishes_the_same_synthetic_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    archive = _write_archive(tmp_path / "cli.tar.gz", _valid_members())
    output = tmp_path / "cli-output"
    original = librispeech.build_librispeech_audit

    def build_unverified(path: Path, destination: Path) -> librispeech.LibriSpeechAudit:
        return original(path, destination, source_identity=None)

    monkeypatch.setattr(librispeech, "build_librispeech_audit", build_unverified)

    assert librispeech.main([str(archive), "--output", str(output)]) == 0

    printed = json.loads(capsys.readouterr().out)
    written = json.loads((output / "dev-clean.audit.json").read_text(encoding="utf-8"))
    assert printed == written


@pytest.mark.parametrize(
    ("transcript", "message"),
    [
        (
            b"1272-128104-0000 FIRST\n1272-128104-0000 DUPLICATE\n",
            "repeats utterance ID",
        ),
        (b"1272-128104-0000 ONLY ONE\n", "bijection is missing"),
        (
            _VALID_TRANSCRIPT + b"1272-128104-0002 EXTRA\n",
            "bijection has unexpected",
        ),
        (_VALID_TRANSCRIPT.replace(b"\n", b"\r\n"), "contains a CR"),
        (b"1272-128104-0000 BAD \xff\n", "not strict UTF-8"),
        (b"1272-128104-0000\n", "must be"),
        (b"1272-128104-0000 \n", "must be"),
        (b"1272-128104-0000  LEADING SPACE\n", "must be"),
        (b"\n", "must be"),
        (b"1272-128104-0000 NO TERMINATOR", "terminated by an LF"),
        (
            b"1272-999-0000 WRONG CHAPTER\n1272-128104-0001 SECOND\n",
            "different chapter",
        ),
    ],
)
def test_strict_transcript_and_exact_flac_bijection(
    tmp_path: Path, transcript: bytes, message: str
) -> None:
    members = _replace_payload(_valid_members(), _TRANSCRIPT_PATH, transcript)
    archive = _write_archive(tmp_path / "transcript.tar.gz", members)

    with pytest.raises(LibriSpeechError, match=message):
        audit_librispeech_archive(archive, source_identity=_identity(archive))


def _invalid_audio_cases() -> list[tuple[bytes, str]]:
    long_flac = _flac_bytes(sample_count=8_192)
    return [
        (b"not a FLAC stream", "cannot decode FLAC"),
        (_flac_bytes(channels=2), "unsupported FLAC format"),
        (_flac_bytes(sample_rate=8_000), "unsupported FLAC format"),
        (_flac_bytes(subtype="PCM_24"), "unsupported FLAC format"),
        (long_flac[: len(long_flac) // 2], "cannot decode FLAC|sample count differs"),
    ]


@pytest.mark.parametrize(("audio", "message"), _invalid_audio_cases())
def test_invalid_or_noncanonical_flac_is_rejected(
    tmp_path: Path, audio: bytes, message: str
) -> None:
    members = _replace_payload(_valid_members(), _FIRST_FLAC_PATH, audio)
    archive = _write_archive(tmp_path / "audio.tar.gz", members)

    with pytest.raises(LibriSpeechError, match=message):
        audit_librispeech_archive(archive, source_identity=_identity(archive))


def test_payload_pass_uses_extractfile_only_after_the_double_identity_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _write_archive(tmp_path / "ordered.tar.gz", _valid_members())
    digest_calls = 0
    extract_calls = 0
    original_digest = librispeech._archive_digests
    original_extractfile = tarfile.TarFile.extractfile

    def counted_digest(stream: BinaryIO) -> tuple[int, str, str]:
        nonlocal digest_calls
        digest_calls += 1
        return original_digest(stream)

    def checked_extractfile(
        self: tarfile.TarFile, member: str | tarfile.TarInfo
    ) -> IO[bytes] | None:
        nonlocal extract_calls
        assert digest_calls >= 2
        extract_calls += 1
        return original_extractfile(self, member)

    monkeypatch.setattr(librispeech, "_archive_digests", counted_digest)
    monkeypatch.setattr(tarfile.TarFile, "extractfile", checked_extractfile)

    audit_librispeech_archive(archive, source_identity=_identity(archive))

    assert digest_calls == 3
    assert extract_calls == len(_valid_members())


def test_payload_audit_opens_the_archive_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _write_archive(tmp_path / "one-fd.tar.gz", _valid_members())
    archive_open_calls = 0
    original_open = os.open

    def counted_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal archive_open_calls
        if os.fsdecode(path) == os.fspath(archive):
            archive_open_calls += 1
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", counted_open)

    audit_librispeech_archive(archive, source_identity=_identity(archive))

    assert archive_open_calls == 1


def test_archive_mutation_during_payload_pass_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _write_archive(tmp_path / "mutable.tar.gz", _valid_members())
    original = librispeech._payload_inventory

    def audit_then_mutate(
        opened: tarfile.TarFile, inspection: librispeech.ArchiveInspection
    ) -> librispeech.LibriSpeechAudit:
        result = original(opened, inspection)
        with archive.open("r+b") as destination:
            destination.seek(-1, io.SEEK_END)
            final = destination.read(1)
            destination.seek(-1, io.SEEK_END)
            destination.write(bytes([final[0] ^ 1]))
        return result

    monkeypatch.setattr(librispeech, "_payload_inventory", audit_then_mutate)

    with pytest.raises(LibriSpeechError, match="changed during payload audit"):
        audit_librispeech_archive(archive, source_identity=_identity(archive))


def test_payload_audit_never_uses_filesystem_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _write_archive(tmp_path / "no-extract.tar.gz", _valid_members())

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("filesystem extraction is forbidden")

    monkeypatch.setattr(tarfile.TarFile, "extract", forbidden)
    monkeypatch.setattr(tarfile.TarFile, "extractall", forbidden)

    build_librispeech_audit(
        archive, tmp_path / "audit", source_identity=_identity(archive)
    )


def test_existing_and_symlink_outputs_are_rejected_without_changes(
    tmp_path: Path,
) -> None:
    archive = _write_archive(tmp_path / "output.tar.gz", _valid_members())
    identity = _identity(archive)
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "sentinel"
    sentinel.write_text("keep", encoding="utf-8")
    destination = tmp_path / "destination"
    destination.mkdir()
    link = tmp_path / "link"
    link.symlink_to(destination, target_is_directory=True)

    with pytest.raises(LibriSpeechError, match="already exists"):
        build_librispeech_audit(archive, existing, source_identity=identity)
    with pytest.raises(LibriSpeechError, match="already exists"):
        build_librispeech_audit(archive, link, source_identity=identity)

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert list(destination.iterdir()) == []


def test_atomic_publish_refuses_a_destination_created_during_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _write_archive(tmp_path / "race.tar.gz", _valid_members())
    identity = _identity(archive)
    output = tmp_path / "raced-output"
    original = librispeech.audit_librispeech_archive

    def audit_then_claim(
        path: Path, *, source_identity: SourceIdentity | None = librispeech.DEV_CLEAN
    ) -> librispeech.LibriSpeechAudit:
        result = original(path, source_identity=source_identity)
        output.mkdir()
        (output / "sentinel").write_text("racer", encoding="utf-8")
        return result

    monkeypatch.setattr(librispeech, "audit_librispeech_archive", audit_then_claim)

    with pytest.raises(LibriSpeechError, match="already exists"):
        build_librispeech_audit(archive, output, source_identity=identity)

    assert (output / "sentinel").read_text(encoding="utf-8") == "racer"
    assert not list(tmp_path.glob(".raced-output.*.tmp"))


def test_failed_payload_audit_publishes_no_partial_directory(tmp_path: Path) -> None:
    members = _replace_payload(_valid_members(), _FIRST_FLAC_PATH, b"not a FLAC stream")
    archive = _write_archive(tmp_path / "invalid.tar.gz", members)
    output = tmp_path / "failed-output"

    with pytest.raises(LibriSpeechError, match="cannot decode FLAC"):
        build_librispeech_audit(archive, output, source_identity=_identity(archive))

    assert not output.exists()
    assert not list(tmp_path.glob(".failed-output.*.tmp"))
