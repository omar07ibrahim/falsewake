from __future__ import annotations

import hashlib
import os
import wave
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pytest

import falsewake.speech_commands_pcm as pcm
from falsewake.baseline_data import ClipExample
from falsewake.feature_matrix import extract_feature_matrix, feature_matrix_sha256
from falsewake.features import extract_clip_features, pcm16le_to_float32
from falsewake.speech_commands import (
    MAX_BACKGROUND_FILE_BYTES,
    MAX_COMMAND_FILE_BYTES,
)
from falsewake.speech_commands_pcm import (
    COMMAND_SAMPLE_LIMIT,
    PCMSourceIdentity,
    SpeechCommandsPCMError,
    SpeechCommandsPCMLoader,
    VerifiedPCM16LE,
)


def _write_wav(
    root: Path,
    relative_path: str,
    payload: bytes,
    *,
    channels: int = 1,
    sample_width: int = 2,
    sample_rate: int = 16_000,
) -> PCMSourceIdentity:
    path = root.joinpath(*relative_path.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(sample_width)
        audio.setframerate(sample_rate)
        audio.writeframes(payload)
    with wave.open(str(path), "rb") as audio:
        sample_count = audio.getnframes()
    return PCMSourceIdentity(
        path=relative_path,
        sample_count=sample_count,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _write_pcm16(
    root: Path,
    relative_path: str,
    samples: np.ndarray[tuple[int], np.dtype[np.int16]],
    *,
    channels: int = 1,
    sample_rate: int = 16_000,
) -> PCMSourceIdentity:
    return _write_wav(
        root,
        relative_path,
        samples.astype("<i2").tobytes(),
        channels=channels,
        sample_width=2,
        sample_rate=sample_rate,
    )


def _clip_example(
    source: PCMSourceIdentity,
    *,
    kind: Literal["command", "silence"] = "command",
    label: str = "yes",
    start_sample: int = 0,
) -> ClipExample:
    return ClipExample(
        kind=kind,
        path=source.path,
        split="train",
        label=label,
        start_sample=start_sample,
        source_sample_count=source.sample_count,
        source_sha256=source.sha256,
    )


def test_exact_pcm_payload_conversion_and_ownership(tmp_path: Path) -> None:
    samples = np.asarray([-32_768, -1, 0, 1, 32_767], dtype=np.int16)
    source = _write_pcm16(tmp_path, "yes/endpoints.wav", samples)

    with SpeechCommandsPCMLoader(tmp_path) as loader:
        verified = loader.load_command(source)
        first = verified.to_float32()
        second = verified.to_float32()

    assert verified.path == source.path
    assert verified.sha256 == source.sha256
    assert verified.sample_count == samples.size
    assert verified.payload == samples.astype("<i2").tobytes()
    expected = np.asarray(
        [-1.0, -1.0 / 32_768.0, 0.0, 1.0 / 32_768.0, 32_767.0 / 32_768.0],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(first, expected)
    assert first.dtype == second.dtype == np.dtype(np.float32)
    assert first.flags.c_contiguous and second.flags.c_contiguous
    assert first.flags.owndata and second.flags.owndata
    assert not np.shares_memory(first, second)
    first[:] = 0.0
    np.testing.assert_array_equal(second, expected)
    np.testing.assert_array_equal(pcm16le_to_float32(verified.payload), expected)


@pytest.mark.parametrize("sample_count", [15_999, 16_000])
def test_command_sample_boundaries_are_accepted(
    tmp_path: Path, sample_count: int
) -> None:
    samples = np.arange(sample_count, dtype=np.int16)
    source = _write_pcm16(tmp_path, "yes/boundary.wav", samples)
    with SpeechCommandsPCMLoader(tmp_path) as loader:
        observed = loader.load_command(source)
    assert observed.sample_count == sample_count
    assert observed.payload == samples.astype("<i2").tobytes()


def test_long_command_is_rejected_but_background_is_loaded_whole(
    tmp_path: Path,
) -> None:
    command_samples = np.zeros(COMMAND_SAMPLE_LIMIT + 1, dtype=np.int16)
    command = _write_pcm16(tmp_path, "yes/too_long.wav", command_samples)
    background_samples = np.arange(32_000, dtype=np.int16)
    background = _write_pcm16(
        tmp_path, "_background_noise_/room.wav", background_samples
    )
    short_background = _write_pcm16(
        tmp_path, "_background_noise_/short.wav", np.arange(80, dtype=np.int16)
    )

    with SpeechCommandsPCMLoader(tmp_path) as loader:
        with pytest.raises(SpeechCommandsPCMError, match="command exceeds"):
            loader.load_command(command)
        complete = loader.load_background(background)
        short = loader.load_background(short_background)

    assert complete.sample_count == 32_000
    assert complete.payload == background_samples.astype("<i2").tobytes()
    assert short.sample_count == 80


def test_new_loader_is_byte_identical_to_frozen_feature_extraction(
    tmp_path: Path,
) -> None:
    command_pcm = np.arange(-400, 400, dtype=np.int16)
    background_pcm = ((np.arange(32_000) * 17) % 20_000 - 10_000).astype(np.int16)
    command = _write_pcm16(tmp_path, "yes/alice_nohash_0.wav", command_pcm)
    background = _write_pcm16(tmp_path, "_background_noise_/room.wav", background_pcm)
    examples = (
        _clip_example(command),
        _clip_example(
            background,
            kind="silence",
            label="silence",
            start_sample=16_000,
        ),
    )

    legacy = extract_feature_matrix(tmp_path, examples)
    with SpeechCommandsPCMLoader(tmp_path) as loader:
        command_waveform = loader.load_command(command).to_float32()
        background_waveform = loader.load_background(background).to_float32()
    shared = np.stack(
        (
            extract_clip_features(command_waveform),
            extract_clip_features(background_waveform[16_000:32_000]),
        )
    ).astype(np.float32)

    assert shared.tobytes(order="C") == legacy.tobytes(order="C")
    assert feature_matrix_sha256(shared) == (
        "77fb6a45ffee1aeaac0b35472eb98a883423cce8f815a8f9abb32c1ac34c9845"
    )
    assert hashlib.sha256(shared.astype("<f4").tobytes()).hexdigest() == (
        "16473a1a015d59925cba8078d857df10f9a9289fe026b9a771b3ea43e8320d68"
    )


@pytest.mark.parametrize(
    "path",
    [
        "",
        "/yes/a.wav",
        "../yes/a.wav",
        "./yes/a.wav",
        "yes//a.wav",
        "yes/a.wav/",
        "yes\\a.wav",
        "yes/a\0.wav",
        "yes/nested/a.wav",
        "yes/a.flac",
        "yes/a.WAV",
    ],
)
def test_noncanonical_paths_are_rejected(path: str) -> None:
    with pytest.raises(SpeechCommandsPCMError, match="not canonical"):
        PCMSourceIdentity(path=path, sample_count=1, sha256="0" * 64)


def test_source_paths_must_encode_as_valid_utf8() -> None:
    with pytest.raises(SpeechCommandsPCMError, match="valid UTF-8"):
        PCMSourceIdentity(
            path="yes/bad-\ud800.wav",
            sample_count=1,
            sha256="0" * 64,
        )


def test_identity_fields_are_strictly_validated() -> None:
    with pytest.raises(TypeError):
        PCMSourceIdentity(path=cast(str, b"yes/a.wav"), sample_count=1, sha256="0" * 64)
    for sample_count in (0, -1):
        with pytest.raises(SpeechCommandsPCMError, match="positive"):
            PCMSourceIdentity(
                path="yes/a.wav", sample_count=sample_count, sha256="0" * 64
            )
    with pytest.raises(TypeError):
        PCMSourceIdentity(
            path="yes/a.wav", sample_count=cast(int, True), sha256="0" * 64
        )
    with pytest.raises(TypeError):
        PCMSourceIdentity(path="yes/a.wav", sample_count=1, sha256=cast(str, 123))
    for digest in ("0" * 63, "A" * 64, "g" * 64):
        with pytest.raises(SpeechCommandsPCMError, match="lowercase SHA-256"):
            PCMSourceIdentity(path="yes/a.wav", sample_count=1, sha256=digest)


def test_command_and_background_paths_cannot_cross_kinds(tmp_path: Path) -> None:
    command = _write_pcm16(tmp_path, "yes/command.wav", np.zeros(20, dtype=np.int16))
    background = _write_pcm16(
        tmp_path, "_background_noise_/room.wav", np.zeros(20, dtype=np.int16)
    )
    with SpeechCommandsPCMLoader(tmp_path) as loader:
        with pytest.raises(SpeechCommandsPCMError, match="invalid path"):
            loader.load_command(background)
        with pytest.raises(SpeechCommandsPCMError, match="invalid path"):
            loader.load_background(command)


def test_root_parent_and_leaf_symlinks_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    outside = tmp_path / "outside"
    source = _write_pcm16(outside, "yes/source.wav", np.zeros(20, dtype=np.int16))
    root.symlink_to(outside, target_is_directory=True)
    with pytest.raises(SpeechCommandsPCMError, match="dataset root"):
        SpeechCommandsPCMLoader(root)
    root.unlink()
    root.mkdir()
    (root / "alias").symlink_to(outside / "yes", target_is_directory=True)
    parent_link = PCMSourceIdentity(
        path="alias/source.wav",
        sample_count=source.sample_count,
        sha256=source.sha256,
    )
    (root / "yes").mkdir()
    (root / "yes/leaf.wav").symlink_to(outside / "yes/source.wav")
    leaf_link = PCMSourceIdentity(
        path="yes/leaf.wav",
        sample_count=source.sample_count,
        sha256=source.sha256,
    )

    with SpeechCommandsPCMLoader(root) as loader:
        for linked in (parent_link, leaf_link):
            with pytest.raises(SpeechCommandsPCMError, match="securely open"):
                loader.load_command(linked)


def test_missing_and_nonregular_sources_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "yes").mkdir()
    missing = PCMSourceIdentity(path="yes/missing.wav", sample_count=1, sha256="0" * 64)
    (tmp_path / "yes/directory.wav").mkdir()
    directory = PCMSourceIdentity(
        path="yes/directory.wav", sample_count=1, sha256="0" * 64
    )
    fifo_path = tmp_path / "yes/fifo.wav"
    os.mkfifo(fifo_path)
    fifo = PCMSourceIdentity(path="yes/fifo.wav", sample_count=1, sha256="0" * 64)

    with SpeechCommandsPCMLoader(tmp_path) as loader:
        with pytest.raises(SpeechCommandsPCMError, match="securely open"):
            loader.load_command(missing)
        for nonregular in (directory, fifo):
            with pytest.raises(SpeechCommandsPCMError, match="regular file"):
                loader.load_command(nonregular)


def test_digest_sample_count_and_same_size_mutation_are_rejected(
    tmp_path: Path,
) -> None:
    samples = np.arange(200, dtype=np.int16)
    source = _write_pcm16(tmp_path, "yes/source.wav", samples)
    wrong_digest = PCMSourceIdentity(
        path=source.path, sample_count=source.sample_count, sha256="0" * 64
    )
    wrong_count = PCMSourceIdentity(
        path=source.path,
        sample_count=source.sample_count + 1,
        sha256=source.sha256,
    )
    with SpeechCommandsPCMLoader(tmp_path) as loader:
        with pytest.raises(SpeechCommandsPCMError, match="digest differs"):
            loader.load_command(wrong_digest)
        with pytest.raises(SpeechCommandsPCMError, match="frame count differs"):
            loader.load_command(wrong_count)

    path = tmp_path / "yes/source.wav"
    contents = bytearray(path.read_bytes())
    contents[-1] ^= 1
    path.write_bytes(contents)
    with (
        SpeechCommandsPCMLoader(tmp_path) as loader,
        pytest.raises(SpeechCommandsPCMError, match="digest differs"),
    ):
        loader.load_command(source)


@pytest.mark.parametrize(
    ("relative_path", "payload", "channels", "sample_width", "sample_rate"),
    [
        ("yes/stereo.wav", b"\0" * 80, 2, 2, 16_000),
        ("yes/eight_bit.wav", b"\x80" * 40, 1, 1, 16_000),
        ("yes/twenty_four_bit.wav", b"\0" * 120, 1, 3, 16_000),
        ("yes/wrong_rate.wav", b"\0" * 80, 1, 2, 8_000),
    ],
)
def test_unsupported_wav_formats_are_rejected(
    tmp_path: Path,
    relative_path: str,
    payload: bytes,
    channels: int,
    sample_width: int,
    sample_rate: int,
) -> None:
    written = _write_wav(
        tmp_path,
        relative_path,
        payload,
        channels=channels,
        sample_width=sample_width,
        sample_rate=sample_rate,
    )
    source = PCMSourceIdentity(
        path=written.path,
        sample_count=max(1, written.sample_count),
        sha256=written.sha256,
    )
    with (
        SpeechCommandsPCMLoader(tmp_path) as loader,
        pytest.raises(SpeechCommandsPCMError, match="unsupported WAV format"),
    ):
        loader.load_command(source)


def test_zero_frame_wav_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "yes/empty.wav"
    path.parent.mkdir()
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"")
    source = PCMSourceIdentity(
        path="yes/empty.wav",
        sample_count=1,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    with (
        SpeechCommandsPCMLoader(tmp_path) as loader,
        pytest.raises(SpeechCommandsPCMError, match="unsupported WAV format"),
    ):
        loader.load_command(source)


def test_malformed_and_truncated_wavs_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "yes").mkdir()
    malformed_path = tmp_path / "yes/malformed.wav"
    malformed_path.write_bytes(b"not a WAV file")
    malformed = PCMSourceIdentity(
        path="yes/malformed.wav",
        sample_count=1,
        sha256=hashlib.sha256(malformed_path.read_bytes()).hexdigest(),
    )
    valid = _write_pcm16(tmp_path, "yes/truncated.wav", np.arange(100, dtype=np.int16))
    truncated_path = tmp_path / "yes/truncated.wav"
    truncated_path.write_bytes(truncated_path.read_bytes()[:-20])
    truncated = PCMSourceIdentity(
        path=valid.path,
        sample_count=valid.sample_count,
        sha256=hashlib.sha256(truncated_path.read_bytes()).hexdigest(),
    )

    with SpeechCommandsPCMLoader(tmp_path) as loader:
        with pytest.raises(SpeechCommandsPCMError, match="cannot decode WAV"):
            loader.load_command(malformed)
        with pytest.raises(SpeechCommandsPCMError, match="truncated WAV payload"):
            loader.load_command(truncated)


@pytest.mark.parametrize(
    ("relative_path", "size_limit"),
    [
        ("yes/oversized.wav", MAX_COMMAND_FILE_BYTES),
        ("_background_noise_/oversized.wav", MAX_BACKGROUND_FILE_BYTES),
    ],
)
def test_oversized_sources_are_rejected_before_reading(
    tmp_path: Path, relative_path: str, size_limit: int
) -> None:
    path = tmp_path.joinpath(*relative_path.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        output.truncate(size_limit + 1)
    source = PCMSourceIdentity(
        path=relative_path,
        sample_count=1,
        sha256="0" * 64,
    )
    with SpeechCommandsPCMLoader(tmp_path) as loader:
        operation = (
            loader.load_background
            if relative_path.startswith("_background_noise_")
            else loader.load_command
        )
        with pytest.raises(SpeechCommandsPCMError, match="exceeds its size limit"):
            operation(source)


def test_pinned_leaf_descriptor_defeats_symlink_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "dataset"
    source = _write_pcm16(root, "yes/source.wav", np.arange(80, dtype=np.int16))
    outside = tmp_path / "outside.wav"
    with outside.open("wb") as output:
        output.truncate(MAX_COMMAND_FILE_BYTES + 1)
    original_open = os.open
    swapped = False

    def swapping_open(
        path: str | bytes | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if dir_fd is None:
            descriptor = original_open(path, flags, mode)
        else:
            descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if path == "source.wav" and dir_fd is not None and not swapped:
            swapped = True
            (root / "yes/source.wav").rename(root / "yes/original.wav")
            (root / "yes/source.wav").symlink_to(outside)
        return descriptor

    with SpeechCommandsPCMLoader(root) as loader:
        monkeypatch.setattr(os, "open", swapping_open)
        observed = loader.load_command(source)

    assert swapped
    assert observed.payload == np.arange(80, dtype=np.int16).astype("<i2").tobytes()


def test_pinned_parent_descriptor_defeats_directory_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "dataset"
    source = _write_pcm16(root, "yes/source.wav", np.arange(60, dtype=np.int16))
    outside = tmp_path / "outside"
    _write_pcm16(outside, "yes/source.wav", np.full(60, 777, dtype=np.int16))
    original_open = os.open
    swapped = False

    def swapping_open(
        path: str | bytes | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if dir_fd is None:
            descriptor = original_open(path, flags, mode)
        else:
            descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if path == "yes" and dir_fd is not None and not swapped:
            swapped = True
            (root / "yes").rename(root / "original-yes")
            (root / "yes").symlink_to(outside / "yes", target_is_directory=True)
        return descriptor

    with SpeechCommandsPCMLoader(root) as loader:
        monkeypatch.setattr(os, "open", swapping_open)
        observed = loader.load_command(source)

    assert swapped
    assert observed.payload == np.arange(60, dtype=np.int16).astype("<i2").tobytes()


def test_pinned_root_descriptor_survives_root_path_replacement(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    source = _write_pcm16(root, "yes/source.wav", np.arange(40, dtype=np.int16))
    loader = SpeechCommandsPCMLoader(root)
    pinned = tmp_path / "pinned"
    root.rename(pinned)
    replacement = tmp_path / "replacement"
    _write_pcm16(replacement, "yes/source.wav", np.full(40, 123, dtype=np.int16))
    root.symlink_to(replacement, target_is_directory=True)
    try:
        observed = loader.load_command(source)
    finally:
        loader.close()

    assert observed.payload == np.arange(40, dtype=np.int16).astype("<i2").tobytes()


def test_concurrent_close_cannot_redirect_a_load_through_fd_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "dataset"
    outside = tmp_path / "outside"
    _write_pcm16(root, "yes/source.wav", np.zeros(40, dtype=np.int16))
    outside_source = _write_pcm16(
        outside,
        "yes/source.wav",
        np.full(40, 777, dtype=np.int16),
    )
    loader = SpeechCommandsPCMLoader(root)
    original_root_fd = loader._root_fd
    original_open = os.open
    replacement_fd = -1
    intercepted_dir_fd = -1

    def reusing_open(
        path: str | bytes | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal intercepted_dir_fd, replacement_fd
        if path == "yes" and dir_fd is not None and replacement_fd < 0:
            intercepted_dir_fd = dir_fd
            loader.close()
            root_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
            replacement_fd = original_open(outside, root_flags)
            assert replacement_fd == original_root_fd
        if dir_fd is None:
            return original_open(path, flags, mode)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    try:
        monkeypatch.setattr(os, "open", reusing_open)
        with pytest.raises(SpeechCommandsPCMError, match="digest differs"):
            loader.load_command(outside_source)
    finally:
        loader.close()
        if replacement_fd >= 0:
            os.close(replacement_fd)

    assert intercepted_dir_fd >= 0
    assert intercepted_dir_fd != original_root_fd


def test_changed_fstat_fingerprint_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_pcm16(tmp_path, "yes/source.wav", np.arange(40, dtype=np.int16))
    original_fstat = os.fstat
    calls = 0

    def changed_fstat(descriptor: int) -> os.stat_result:
        nonlocal calls
        calls += 1
        observed = original_fstat(descriptor)
        if calls == 2:
            fields = list(observed)
            fields[6] = observed.st_size + 1
            return os.stat_result(fields)
        return observed

    with SpeechCommandsPCMLoader(tmp_path) as loader:
        monkeypatch.setattr(os, "fstat", changed_fstat)
        with pytest.raises(SpeechCommandsPCMError, match="changed while reading"):
            loader.load_command(source)


def test_loader_lifecycle_and_verified_payload_validation(tmp_path: Path) -> None:
    source = _write_pcm16(tmp_path, "yes/source.wav", np.arange(20, dtype=np.int16))
    loader = SpeechCommandsPCMLoader(tmp_path)
    assert not loader.closed
    loader.close()
    loader.close()
    assert loader.closed
    with pytest.raises(SpeechCommandsPCMError, match="closed"):
        loader.load_command(source)
    with (
        SpeechCommandsPCMLoader(tmp_path) as open_loader,
        pytest.raises(TypeError, match="source must"),
    ):
        open_loader.load_command(cast(PCMSourceIdentity, object()))
    with pytest.raises(SpeechCommandsPCMError, match="exactly 2 bytes"):
        pcm._new_verified_pcm16le(
            path="yes/source.wav",
            sample_count=1,
            sha256="0" * 64,
            payload=b"",
        )
    with pytest.raises(TypeError, match="PCM loader"):
        VerifiedPCM16LE()


def test_loader_rejects_invalid_roots(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="must be a Path"):
        SpeechCommandsPCMLoader(cast(Path, str(tmp_path)))
    with pytest.raises(SpeechCommandsPCMError, match="dataset root"):
        SpeechCommandsPCMLoader(tmp_path / "missing")
    regular_file = tmp_path / "not-a-directory"
    regular_file.write_bytes(b"x")
    with pytest.raises(SpeechCommandsPCMError, match="dataset root"):
        SpeechCommandsPCMLoader(regular_file)


def test_frozen_predecessor_sources_remain_byte_identical() -> None:
    expected = {
        "src/falsewake/feature_matrix.py": (
            "82ca61ebda6ec833d0047acb4a88091411efa16cbc5667ae7b16ca85d8842aff"
        ),
        "src/falsewake/features.py": (
            "3e7baf53c8dc772a68cd256ce6f4ca6428117c36f051c916f247dc409e515e3c"
        ),
        "src/falsewake/speech_commands.py": (
            "f80bd2bad56113d4adc168b04e1633ec22d8513afb973a4b5dc37d616901e234"
        ),
    }
    assert {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in expected
    } == expected
