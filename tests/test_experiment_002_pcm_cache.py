from __future__ import annotations

import ast
import hashlib
import os
import stat
import struct
import threading
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Final, cast

import numpy as np
import pytest

import falsewake.experiment_002_pcm_cache as pcm_cache
import falsewake.speech_commands_pcm as pcm
from falsewake.experiment_002_data import (
    BackgroundSource,
    CommandSource,
    Experiment002Corpus,
)
from falsewake.features import pcm16le_to_float32
from falsewake.speech_commands_pcm import PCMSourceIdentity, VerifiedPCM16LE

HEADER_BYTES: Final = 320
ENTRY_BYTES: Final = 96
ALIGNMENT: Final = 4096
MANIFEST_INVENTORY_SHA256: Final = "2" * 64
ARCHIVE_SHA256: Final = "3" * 64

_HEADER = struct.Struct("<16sIIIIQQ32s32s32s32s32s32s32s4I32s")
_ENTRY = struct.Struct("<B3sIIIQ32s32s8s")


def _pcm(samples: list[int] | np.ndarray[tuple[int], np.dtype[np.int16]]) -> bytes:
    return np.asarray(samples, dtype="<i2").tobytes()


def _pattern(sample_count: int, *, offset: int) -> bytes:
    values = (
        (np.arange(sample_count, dtype=np.int32) * 257 + offset) % 65_536 - 32_768
    ).astype("<i2")
    return values.tobytes()


def _command(
    manifest_index: int,
    path: str,
    payload: bytes,
    *,
    label: str | None = None,
) -> CommandSource:
    word = path.split("/", maxsplit=1)[0]
    return CommandSource(
        manifest_index=manifest_index,
        path=path,
        word=word,
        label=label if label is not None else word,
        sample_count=len(payload) // 2,
        sha256=hashlib.sha256(b"wav:" + path.encode()).hexdigest(),
    )


def _background(path: str, payload: bytes) -> BackgroundSource:
    return BackgroundSource(
        path=path,
        sample_count=len(payload) // 2,
        sha256=hashlib.sha256(b"wav:" + path.encode()).hexdigest(),
    )


class _SyntheticCorpus:
    def __init__(self) -> None:
        self.payloads: dict[str, bytes] = {
            "yes/a_nohash_0.wav": _pcm([-32_768, -1, 0, 32_767]),
            "no/b_nohash_0.wav": _pcm([1, 2, 3]),
            "up/c_nohash_0.wav": _pcm([-9, 8, -7, 6, -5]),
            "_background_noise_/hum.wav": _pattern(16_003, offset=17),
            "_background_noise_/room.wav": _pattern(16_001, offset=29),
            "_background_noise_/running.wav": _pattern(16_005, offset=43),
        }
        train_commands = (
            _command(9, "yes/a_nohash_0.wav", self.payloads["yes/a_nohash_0.wav"]),
            _command(2, "no/b_nohash_0.wav", self.payloads["no/b_nohash_0.wav"]),
        )
        validation_commands = (
            _command(15, "up/c_nohash_0.wav", self.payloads["up/c_nohash_0.wav"]),
        )
        train_backgrounds = (
            _background(
                "_background_noise_/room.wav",
                self.payloads["_background_noise_/room.wav"],
            ),
            _background(
                "_background_noise_/hum.wav",
                self.payloads["_background_noise_/hum.wav"],
            ),
        )
        validation_background = _background(
            "_background_noise_/running.wav",
            self.payloads["_background_noise_/running.wav"],
        )
        self.corpus = Experiment002Corpus(
            train_commands=train_commands,
            validation_commands=validation_commands,
            train_backgrounds=train_backgrounds,
            validation_background=validation_background,
            # Deliberately nonzero: the cache must retain only this count, never an
            # identity or a loader request for a test source.
            test_command_count=7,
            manifest_sha256="1" * 64,
        )
        self.calls: list[tuple[str, str]] = []

    def loader_factory(self, dataset_root: Path) -> _FakeLoader:
        return _FakeLoader(dataset_root, self.payloads, self.calls)


class _FakeLoader:
    def __init__(
        self,
        dataset_root: Path,
        payloads: Mapping[str, bytes],
        calls: list[tuple[str, str]],
    ) -> None:
        self.dataset_root = dataset_root
        self._payloads = payloads
        self._calls = calls
        self.closed = False

    def __enter__(self) -> _FakeLoader:
        if self.closed:
            raise AssertionError("fake loader was reused after close")
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self.closed = True

    def load_command(self, source: PCMSourceIdentity) -> VerifiedPCM16LE:
        return self._load("command", source)

    def load_background(self, source: PCMSourceIdentity) -> VerifiedPCM16LE:
        return self._load("background", source)

    def _load(self, kind: str, source: PCMSourceIdentity) -> VerifiedPCM16LE:
        if self.closed:
            raise AssertionError("fake loader was closed")
        self._calls.append((kind, source.path))
        try:
            payload = self._payloads[source.path]
        except KeyError as error:
            raise AssertionError(
                f"cache attempted an unregistered load: {source.path}"
            ) from error
        assert source.sample_count == len(payload) // 2
        return pcm._new_verified_pcm16le(
            path=source.path,
            sample_count=source.sample_count,
            sha256=source.sha256,
            payload=payload,
        )


def _layout(synthetic: _SyntheticCorpus) -> pcm_cache._CacheLayout:
    return pcm_cache._CacheLayout.for_corpus(
        synthetic.corpus,
        manifest_inventory_sha256=MANIFEST_INVENTORY_SHA256,
        archive_sha256=ARCHIVE_SHA256,
    )


def _identity(path: Path) -> pcm_cache.PCMCacheIdentity:
    contents = path.read_bytes()
    return pcm_cache.PCMCacheIdentity(
        byte_count=len(contents), sha256=hashlib.sha256(contents).hexdigest()
    )


def _build(
    synthetic: _SyntheticCorpus,
    destination: Path,
    *,
    layout: pcm_cache._CacheLayout | None = None,
) -> pcm_cache.PCMCacheIdentity:
    return pcm_cache._build_pcm_cache(
        synthetic.corpus,
        destination.parent / "unused-dataset-root",
        destination,
        layout=_layout(synthetic) if layout is None else layout,
        loader_factory=synthetic.loader_factory,
    )


@contextmanager
def _opened(
    synthetic: _SyntheticCorpus,
    path: Path,
    expected: pcm_cache.PCMCacheIdentity | None = None,
) -> Iterator[pcm_cache.Experiment002PCMCache]:
    opened = pcm_cache.Experiment002PCMCache._open(
        synthetic.corpus,
        path,
        _identity(path) if expected is None else expected,
        layout=_layout(synthetic),
    )
    try:
        yield opened
    finally:
        opened.close()


def _header(contents: bytes) -> tuple[object, ...]:
    return cast(tuple[object, ...], _HEADER.unpack_from(contents))


def _entries(contents: bytes) -> list[tuple[object, ...]]:
    count = cast(int, _header(contents)[4])
    return [
        cast(
            tuple[object, ...],
            _ENTRY.unpack_from(contents, HEADER_BYTES + i * ENTRY_BYTES),
        )
        for i in range(count)
    ]


def _rewrite_header_digest(contents: bytearray, digest_offset: int) -> None:
    if digest_offset == 0xD0:
        entry_count = struct.unpack_from("<I", contents, 0x1C)[0]
        section = contents[HEADER_BYTES : HEADER_BYTES + entry_count * ENTRY_BYTES]
    elif digest_offset == 0xF0:
        payload_offset = struct.unpack_from("<Q", contents, 0x20)[0]
        payload_bytes = struct.unpack_from("<Q", contents, 0x28)[0]
        section = contents[payload_offset : payload_offset + payload_bytes]
    else:
        raise AssertionError("unsupported embedded digest offset")
    contents[digest_offset : digest_offset + 32] = hashlib.sha256(section).digest()


def _write_forgery(path: Path, contents: bytearray) -> pcm_cache.PCMCacheIdentity:
    path.chmod(0o600)
    path.write_bytes(contents)
    return _identity(path)


@pytest.fixture
def synthetic() -> _SyntheticCorpus:
    return _SyntheticCorpus()


def test_exact_little_endian_layout_alignment_and_source_order(
    tmp_path: Path, synthetic: _SyntheticCorpus
) -> None:
    destination = tmp_path / "tiny.pcmcache"
    identity = _build(synthetic, destination)
    contents = destination.read_bytes()
    layout = _layout(synthetic)

    assert identity == _identity(destination)
    assert len(contents) == layout.file_bytes
    assert layout.entry_count == 6
    assert layout.payload_offset == ALIGNMENT
    assert layout.payload_offset % ALIGNMENT == 0
    assert _HEADER.size == HEADER_BYTES
    assert _ENTRY.size == ENTRY_BYTES

    header = _header(contents)
    assert header[:7] == (
        b"FALSEWAKEPCM002\0",
        1,
        HEADER_BYTES,
        ENTRY_BYTES,
        6,
        ALIGNMENT,
        layout.payload_bytes,
    )
    assert header[7:12] == (
        bytes.fromhex(synthetic.corpus.manifest_sha256),
        bytes.fromhex(MANIFEST_INVENTORY_SHA256),
        bytes.fromhex(ARCHIVE_SHA256),
        bytes.fromhex(synthetic.corpus._train_inventory_sha256),
        bytes.fromhex(synthetic.corpus._validation_inventory_sha256),
    )
    index = contents[HEADER_BYTES : HEADER_BYTES + 6 * ENTRY_BYTES]
    payload = contents[ALIGNMENT:]
    assert header[12:14] == (
        hashlib.sha256(index).digest(),
        hashlib.sha256(payload).digest(),
    )
    assert header[14:18] == (2, 1, 2, 1)
    assert header[18] == bytes(32)
    assert contents[HEADER_BYTES + 6 * ENTRY_BYTES : ALIGNMENT] == bytes(
        ALIGNMENT - HEADER_BYTES - 6 * ENTRY_BYTES
    )

    ordered_sources: tuple[CommandSource | BackgroundSource, ...] = (
        synthetic.corpus.train_commands[1],
        synthetic.corpus.train_commands[0],
        synthetic.corpus.validation_commands[0],
        synthetic.corpus.train_backgrounds[1],
        synthetic.corpus.train_backgrounds[0],
        synthetic.corpus.validation_background,
    )
    entries = _entries(contents)
    assert [entry[0] for entry in entries] == [0, 0, 1, 2, 2, 3]
    assert [entry[2] for entry in entries] == [2, 9, 15, 0, 1, 0]
    cursor = ALIGNMENT
    for entry, source in zip(entries, ordered_sources, strict=True):
        raw = synthetic.payloads[source.path]
        assert entry[1] == bytes(3)
        assert entry[3] == source.sample_count
        assert entry[4] == 0
        assert entry[5] == cursor
        assert entry[6] == bytes.fromhex(source.sha256)
        assert entry[7] == hashlib.sha256(raw).digest()
        assert entry[8] == bytes(8)
        assert contents[cursor : cursor + len(raw)] == raw
        cursor += len(raw)
    assert cursor == len(contents)
    assert stat.S_IMODE(destination.stat().st_mode) & 0o222 == 0


def test_registered_layout_constants_match_the_frozen_plan() -> None:
    assert pcm_cache.HEADER_BYTES == 320
    assert pcm_cache.ENTRY_BYTES == 96
    assert pcm_cache.PAYLOAD_ALIGNMENT == 4096
    assert pcm_cache.REGISTERED_ENTRY_COUNT == 94_829
    assert pcm_cache.REGISTERED_PAYLOAD_BYTES == 2_986_671_540
    assert pcm_cache.REGISTERED_PAYLOAD_OFFSET == 9_105_408
    assert pcm_cache.REGISTERED_FILE_BYTES == 2_995_776_948
    assert pcm_cache._REGISTERED_LAYOUT.role_counts == (84_843, 9_981, 4, 1)
    assert pcm_cache._REGISTERED_LAYOUT.file_bytes < 4 * 1024**3


def test_public_api_refuses_a_tiny_nonregistered_corpus_before_io(
    tmp_path: Path, synthetic: _SyntheticCorpus
) -> None:
    destination = tmp_path / "public.cache"
    with pytest.raises(
        pcm_cache.Experiment002PCMCacheError, match="registered|population|corpus"
    ):
        pcm_cache.build_registered_pcm_cache(
            synthetic.corpus,
            tmp_path / "dataset",
            destination,
        )
    assert not destination.exists()

    with pytest.raises(
        pcm_cache.Experiment002PCMCacheError, match="registered|population|corpus"
    ):
        pcm_cache.Experiment002PCMCache.open(
            synthetic.corpus,
            tmp_path / "missing.cache",
            pcm_cache.PCMCacheIdentity(1, "0" * 64),
        )


def test_build_is_byte_deterministic_and_never_loads_test_sources(
    tmp_path: Path, synthetic: _SyntheticCorpus
) -> None:
    first = tmp_path / "first.cache"
    second = tmp_path / "second.cache"

    first_identity = _build(synthetic, first)
    second_identity = _build(synthetic, second)

    assert first_identity == second_identity
    assert first.read_bytes() == second.read_bytes()
    expected_paths = {
        *(source.path for source in synthetic.corpus.train_commands),
        *(source.path for source in synthetic.corpus.validation_commands),
        *(source.path for source in synthetic.corpus.train_backgrounds),
        synthetic.corpus.validation_background.path,
    }
    assert {path for _, path in synthetic.calls} == expected_paths
    assert len(synthetic.calls) == 2 * len(expected_paths)
    assert all(
        "test" not in path and "white_noise" not in path for _, path in synthetic.calls
    )
    assert not any(path.name.startswith(".first.cache") for path in tmp_path.iterdir())
    assert not any(path.name.startswith(".second.cache") for path in tmp_path.iterdir())


def test_split_views_enforce_training_validation_firewall_and_exact_reads(
    tmp_path: Path, synthetic: _SyntheticCorpus
) -> None:
    path = tmp_path / "views.cache"
    expected = _build(synthetic, path)
    train_command = synthetic.corpus.train_commands[0]
    validation_command = synthetic.corpus.validation_commands[0]
    train_background = synthetic.corpus.train_backgrounds[0]
    validation_background = synthetic.corpus.validation_background

    with _opened(synthetic, path, expected) as opened:
        assert opened.training.split == "train"
        assert opened.validation.split == "validation"
        assert (
            opened.training.read_command_pcm16le(train_command)
            == synthetic.payloads[train_command.path]
        )
        assert (
            opened.validation.read_command_pcm16le(validation_command)
            == synthetic.payloads[validation_command.path]
        )
        with pytest.raises(pcm_cache.Experiment002PCMCacheError, match="split"):
            opened.training.read_command_pcm16le(validation_command)
        with pytest.raises(pcm_cache.Experiment002PCMCacheError, match="split"):
            opened.validation.read_command_pcm16le(train_command)

        final_start = train_background.sample_count - 16_000
        expected_final = synthetic.payloads[train_background.path][2 * final_start :]
        assert (
            opened.training.read_background_window_pcm16le(
                train_background, final_start
            )
            == expected_final
        )
        assert (
            opened.validation.read_background_window_pcm16le(validation_background, 0)
            == synthetic.payloads[validation_background.path][:32_000]
        )
        with pytest.raises(pcm_cache.Experiment002PCMCacheError, match="split"):
            opened.training.read_background_window_pcm16le(validation_background, 0)
        with pytest.raises(pcm_cache.Experiment002PCMCacheError, match="split"):
            opened.validation.read_background_window_pcm16le(train_background, 0)


def test_raw_pcm_endpoints_owned_conversion_and_direct_preprocessing_parity(
    tmp_path: Path, synthetic: _SyntheticCorpus
) -> None:
    path = tmp_path / "parity.cache"
    _build(synthetic, path)
    source = synthetic.corpus.train_commands[0]
    background = synthetic.corpus.validation_background

    with _opened(synthetic, path) as opened:
        command_raw = opened.training.read_command_pcm16le(source)
        first_window = opened.validation.read_background_window_pcm16le(background, 0)
        last_start = background.sample_count - 16_000
        last_window = opened.validation.read_background_window_pcm16le(
            background, last_start
        )

    assert isinstance(command_raw, bytes)
    converted = pcm16le_to_float32(command_raw)
    assert np.array_equal(
        converted,
        np.asarray([-1.0, -1.0 / 32_768.0, 0.0, 32_767.0 / 32_768.0], dtype=np.float32),
    )
    padded = np.zeros(16_000, dtype=np.float32)
    padded[: source.sample_count] = converted
    direct_padded = np.zeros(16_000, dtype=np.float32)
    direct_padded[: source.sample_count] = pcm16le_to_float32(
        synthetic.payloads[source.path]
    )
    assert np.array_equal(padded, direct_padded)
    assert first_window == synthetic.payloads[background.path][:32_000]
    assert last_window == synthetic.payloads[background.path][-32_000:]


def test_output_budget_is_rejected_before_loader_or_temporary_file(
    tmp_path: Path, synthetic: _SyntheticCorpus
) -> None:
    destination = tmp_path / "over-budget.cache"
    normal = _layout(synthetic)
    over_budget = replace(normal, output_bytes_maximum=normal.file_bytes - 1)

    with pytest.raises(pcm_cache.Experiment002PCMCacheError, match="budget|maximum"):
        _build(synthetic, destination, layout=over_budget)

    assert synthetic.calls == []
    assert list(tmp_path.iterdir()) == []


def test_existing_destination_is_never_overwritten_or_loaded(
    tmp_path: Path, synthetic: _SyntheticCorpus
) -> None:
    destination = tmp_path / "existing.cache"
    sentinel = b"owned by another process"
    destination.write_bytes(sentinel)

    with pytest.raises(pcm_cache.Experiment002PCMCacheError, match="exist"):
        _build(synthetic, destination)

    assert destination.read_bytes() == sentinel
    assert synthetic.calls == []
    assert list(tmp_path.iterdir()) == [destination]


def test_concurrent_publish_has_exactly_one_winner(
    tmp_path: Path, synthetic: _SyntheticCorpus
) -> None:
    destination = tmp_path / "contended.cache"
    barrier = threading.Barrier(2)

    def contend() -> pcm_cache.PCMCacheIdentity | BaseException:
        barrier.wait(timeout=10)
        try:
            return _build(synthetic, destination)
        except BaseException as error:  # captured for assertions in the parent thread
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: contend(), range(2)))

    identities = [
        result for result in results if isinstance(result, pcm_cache.PCMCacheIdentity)
    ]
    failures = [result for result in results if isinstance(result, BaseException)]
    assert len(identities) == len(failures) == 1
    assert isinstance(failures[0], pcm_cache.Experiment002PCMCacheError)
    assert identities[0] == _identity(destination)
    assert [path for path in tmp_path.iterdir()] == [destination]


HeaderMutation = tuple[int, bytes]


@pytest.mark.parametrize(
    ("offset", "replacement"),
    [
        (0x00, b"X"),
        (0x10, struct.pack("<I", 2)),
        (0x14, struct.pack("<I", HEADER_BYTES - 1)),
        (0x18, struct.pack("<I", ENTRY_BYTES - 1)),
        (0x1C, struct.pack("<I", 7)),
        (0x20, struct.pack("<Q", ALIGNMENT + 1)),
        (0x28, struct.pack("<Q", 1)),
        (0x30, b"\xff"),
        (0x50, b"\xff"),
        (0x70, b"\xff"),
        (0x90, b"\xff"),
        (0xB0, b"\xff"),
        (0xD0, b"\xff"),
        (0xF0, b"\xff"),
        (0x110, struct.pack("<I", 3)),
        (0x120, b"\x01"),
    ],
)
def test_header_and_provenance_corruption_is_rejected(
    tmp_path: Path,
    synthetic: _SyntheticCorpus,
    offset: int,
    replacement: bytes,
) -> None:
    path = tmp_path / "header.cache"
    _build(synthetic, path)
    forged = bytearray(path.read_bytes())
    forged[offset : offset + len(replacement)] = replacement
    expected = _write_forgery(path, forged)

    with pytest.raises(pcm_cache.Experiment002PCMCacheError):
        pcm_cache.Experiment002PCMCache._open(
            synthetic.corpus,
            path,
            expected,
            layout=_layout(synthetic),
        )


def test_nonzero_alignment_padding_is_rejected(
    tmp_path: Path, synthetic: _SyntheticCorpus
) -> None:
    path = tmp_path / "padding.cache"
    _build(synthetic, path)
    forged = bytearray(path.read_bytes())
    forged[HEADER_BYTES + _layout(synthetic).entry_count * ENTRY_BYTES] = 1
    expected = _write_forgery(path, forged)

    with pytest.raises(pcm_cache.Experiment002PCMCacheError, match="padding|zero"):
        pcm_cache.Experiment002PCMCache._open(
            synthetic.corpus, path, expected, layout=_layout(synthetic)
        )


@pytest.mark.parametrize(
    ("relative_offset", "replacement"),
    [
        (0, b"\xff"),
        (1, b"\x01"),
        (4, struct.pack("<I", 123)),
        (8, struct.pack("<I", 123)),
        (12, struct.pack("<I", 1)),
        (16, struct.pack("<Q", ALIGNMENT + 1)),
        (24, b"\x00" * 32),
        (88, b"\x01"),
    ],
)
def test_every_index_field_and_reserved_region_is_validated(
    tmp_path: Path,
    synthetic: _SyntheticCorpus,
    relative_offset: int,
    replacement: bytes,
) -> None:
    path = tmp_path / "entry.cache"
    _build(synthetic, path)
    forged = bytearray(path.read_bytes())
    entry_offset = HEADER_BYTES + relative_offset
    forged[entry_offset : entry_offset + len(replacement)] = replacement
    _rewrite_header_digest(forged, 0xD0)
    expected = _write_forgery(path, forged)

    with pytest.raises(pcm_cache.Experiment002PCMCacheError):
        pcm_cache.Experiment002PCMCache._open(
            synthetic.corpus, path, expected, layout=_layout(synthetic)
        )


def test_forged_per_record_pcm_digest_is_rejected_when_command_is_read(
    tmp_path: Path, synthetic: _SyntheticCorpus
) -> None:
    path = tmp_path / "record-digest.cache"
    _build(synthetic, path)
    forged = bytearray(path.read_bytes())
    forged[HEADER_BYTES + 56 : HEADER_BYTES + 88] = bytes(32)
    _rewrite_header_digest(forged, 0xD0)
    expected = _write_forgery(path, forged)

    with (
        _opened(synthetic, path, expected) as opened,
        pytest.raises(pcm_cache.Experiment002PCMCacheError, match="PCM|digest"),
    ):
        opened.training.read_command_pcm16le(synthetic.corpus.train_commands[1])


@pytest.mark.parametrize("delta", [-2, 2])
def test_index_offsets_cannot_overlap_or_leave_gaps(
    tmp_path: Path, synthetic: _SyntheticCorpus, delta: int
) -> None:
    path = tmp_path / "offset.cache"
    _build(synthetic, path)
    forged = bytearray(path.read_bytes())
    second_offset_field = HEADER_BYTES + ENTRY_BYTES + 16
    observed = struct.unpack_from("<Q", forged, second_offset_field)[0]
    struct.pack_into("<Q", forged, second_offset_field, observed + delta)
    _rewrite_header_digest(forged, 0xD0)
    expected = _write_forgery(path, forged)

    with pytest.raises(pcm_cache.Experiment002PCMCacheError, match="offset|contiguous"):
        pcm_cache.Experiment002PCMCache._open(
            synthetic.corpus, path, expected, layout=_layout(synthetic)
        )


@pytest.mark.parametrize("change", ["truncate", "append"])
def test_truncation_and_trailing_bytes_are_rejected_even_with_matching_external_hash(
    tmp_path: Path, synthetic: _SyntheticCorpus, change: str
) -> None:
    path = tmp_path / "length.cache"
    _build(synthetic, path)
    contents = path.read_bytes()
    path.chmod(0o600)
    path.write_bytes(contents[:-1] if change == "truncate" else contents + b"\0")
    expected = _identity(path)

    with pytest.raises(pcm_cache.Experiment002PCMCacheError, match="byte|size|length"):
        pcm_cache.Experiment002PCMCache._open(
            synthetic.corpus, path, expected, layout=_layout(synthetic)
        )


def test_payload_corruption_is_rejected_by_embedded_digest(
    tmp_path: Path, synthetic: _SyntheticCorpus
) -> None:
    path = tmp_path / "payload.cache"
    _build(synthetic, path)
    forged = bytearray(path.read_bytes())
    forged[_layout(synthetic).payload_offset] ^= 1
    expected = _write_forgery(path, forged)

    with pytest.raises(pcm_cache.Experiment002PCMCacheError, match="payload|digest"):
        pcm_cache.Experiment002PCMCache._open(
            synthetic.corpus, path, expected, layout=_layout(synthetic)
        )


def test_self_consistent_forgery_still_requires_the_external_identity(
    tmp_path: Path, synthetic: _SyntheticCorpus
) -> None:
    path = tmp_path / "forged.cache"
    original = _build(synthetic, path)
    forged = bytearray(path.read_bytes())
    forged[_layout(synthetic).payload_offset] ^= 1
    first_pcm = forged[
        _layout(synthetic).payload_offset : _layout(synthetic).payload_offset + 6
    ]
    forged[HEADER_BYTES + 56 : HEADER_BYTES + 88] = hashlib.sha256(first_pcm).digest()
    _rewrite_header_digest(forged, 0xD0)
    _rewrite_header_digest(forged, 0xF0)
    path.chmod(0o600)
    path.write_bytes(forged)

    with pytest.raises(
        pcm_cache.Experiment002PCMCacheError, match="external|digest|identity"
    ):
        pcm_cache.Experiment002PCMCache._open(
            synthetic.corpus, path, original, layout=_layout(synthetic)
        )


def test_post_open_command_payload_mutation_is_detected_per_record(
    tmp_path: Path, synthetic: _SyntheticCorpus
) -> None:
    path = tmp_path / "mutable.cache"
    _build(synthetic, path)
    first_source = synthetic.corpus.train_commands[1]
    opened = pcm_cache.Experiment002PCMCache._open(
        synthetic.corpus, path, _identity(path), layout=_layout(synthetic)
    )
    try:
        path.chmod(0o600)
        descriptor = os.open(path, os.O_RDWR | os.O_CLOEXEC)
        try:
            observed = os.pread(descriptor, 1, _layout(synthetic).payload_offset)
            os.pwrite(
                descriptor,
                bytes([observed[0] ^ 1]),
                _layout(synthetic).payload_offset,
            )
        finally:
            os.close(descriptor)

        with pytest.raises(pcm_cache.Experiment002PCMCacheError, match="PCM|digest"):
            opened.training.read_command_pcm16le(first_source)
    finally:
        opened.close()


def test_backgrounds_are_owned_verified_snapshots_after_open(
    tmp_path: Path, synthetic: _SyntheticCorpus
) -> None:
    path = tmp_path / "background-copy.cache"
    _build(synthetic, path)
    source = synthetic.corpus.validation_background
    expected = synthetic.payloads[source.path][:32_000]
    contents = path.read_bytes()
    final_entry = _entries(contents)[-1]
    payload_offset = cast(int, final_entry[5])

    with _opened(synthetic, path) as opened:
        path.chmod(0o600)
        descriptor = os.open(path, os.O_WRONLY | os.O_CLOEXEC)
        try:
            os.pwrite(descriptor, b"\0" * 32_000, payload_offset)
        finally:
            os.close(descriptor)
        assert opened.validation.read_background_window_pcm16le(source, 0) == expected


def test_open_reader_is_bound_to_inode_across_path_replacement(
    tmp_path: Path, synthetic: _SyntheticCorpus
) -> None:
    path = tmp_path / "pinned.cache"
    _build(synthetic, path)
    source = synthetic.corpus.train_commands[0]
    expected = synthetic.payloads[source.path]

    with _opened(synthetic, path) as opened:
        held = tmp_path / "original-inode.cache"
        path.rename(held)
        path.write_bytes(b"untrusted pathname replacement")
        assert opened.training.read_command_pcm16le(source) == expected


def test_open_rejects_symlink_fifo_and_directory_without_blocking(
    tmp_path: Path, synthetic: _SyntheticCorpus
) -> None:
    regular = tmp_path / "regular.cache"
    _build(synthetic, regular)
    identity = _identity(regular)
    symlink = tmp_path / "link.cache"
    symlink.symlink_to(regular)
    fifo = tmp_path / "fifo.cache"
    os.mkfifo(fifo)
    directory = tmp_path / "directory.cache"
    directory.mkdir()

    for candidate in (symlink, fifo, directory):
        with pytest.raises(pcm_cache.Experiment002PCMCacheError):
            pcm_cache.Experiment002PCMCache._open(
                synthetic.corpus,
                candidate,
                identity,
                layout=_layout(synthetic),
            )


def test_builder_rejects_symlink_fifo_and_directory_destinations_without_loading(
    tmp_path: Path, synthetic: _SyntheticCorpus
) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"sentinel")
    symlink = tmp_path / "link"
    symlink.symlink_to(target)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    directory = tmp_path / "directory"
    directory.mkdir()

    for candidate in (symlink, fifo, directory):
        with pytest.raises(pcm_cache.Experiment002PCMCacheError, match="exist"):
            _build(synthetic, candidate)

    assert target.read_bytes() == b"sentinel"
    assert symlink.is_symlink()
    assert stat.S_ISFIFO(fifo.stat().st_mode)
    assert directory.is_dir()
    assert synthetic.calls == []


def test_symlinked_parent_is_rejected_for_build_and_open_before_source_io(
    tmp_path: Path, synthetic: _SyntheticCorpus
) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(pcm_cache.Experiment002PCMCacheError, match="parent|secure"):
        _build(synthetic, linked_parent / "cache.bin")
    assert synthetic.calls == []
    assert list(real_parent.iterdir()) == []

    regular = real_parent / "cache.bin"
    _build(synthetic, regular)
    with pytest.raises(pcm_cache.Experiment002PCMCacheError, match="parent|secure"):
        pcm_cache.Experiment002PCMCache._open(
            synthetic.corpus,
            linked_parent / "cache.bin",
            _identity(regular),
            layout=_layout(synthetic),
        )


def test_loader_identity_failure_closes_loader_and_cleans_temporary(
    tmp_path: Path, synthetic: _SyntheticCorpus
) -> None:
    destination = tmp_path / "wrong-loader.cache"

    class WrongIdentityLoader:
        def __init__(self, dataset_root: Path) -> None:
            self.dataset_root = dataset_root
            self.closed = False
            loaders.append(self)

        def load_command(self, source: PCMSourceIdentity) -> VerifiedPCM16LE:
            payload = synthetic.payloads[source.path]
            return pcm._new_verified_pcm16le(
                path=source.path,
                sample_count=source.sample_count,
                sha256="f" * 64,
                payload=payload,
            )

        def load_background(self, source: PCMSourceIdentity) -> VerifiedPCM16LE:
            raise AssertionError("identity mismatch should stop before backgrounds")

        def close(self) -> None:
            self.closed = True

    loaders: list[WrongIdentityLoader] = []

    with pytest.raises(pcm_cache.Experiment002PCMCacheError, match="identity"):
        pcm_cache._build_pcm_cache(
            synthetic.corpus,
            tmp_path / "dataset",
            destination,
            layout=_layout(synthetic),
            loader_factory=WrongIdentityLoader,
        )

    assert len(loaders) == 1 and loaders[0].closed
    assert list(tmp_path.iterdir()) == []


def test_close_is_idempotent_and_all_views_fail_after_close(
    tmp_path: Path, synthetic: _SyntheticCorpus
) -> None:
    path = tmp_path / "lifecycle.cache"
    _build(synthetic, path)
    opened = pcm_cache.Experiment002PCMCache._open(
        synthetic.corpus, path, _identity(path), layout=_layout(synthetic)
    )
    training = opened.training
    validation = opened.validation
    opened.close()
    opened.close()

    assert opened.closed
    with pytest.raises(pcm_cache.Experiment002PCMCacheError, match="closed"):
        training.read_command_pcm16le(synthetic.corpus.train_commands[0])
    with pytest.raises(pcm_cache.Experiment002PCMCacheError, match="closed"):
        validation.read_background_window_pcm16le(
            synthetic.corpus.validation_background, 0
        )
    with (
        pytest.raises(pcm_cache.Experiment002PCMCacheError, match="closed"),
        opened,
    ):
        pass


def test_repeated_open_close_does_not_leak_descriptors(
    tmp_path: Path, synthetic: _SyntheticCorpus
) -> None:
    path = tmp_path / "fd.cache"
    _build(synthetic, path)
    before = len(os.listdir("/proc/self/fd"))

    for _ in range(100):
        with _opened(synthetic, path) as opened:
            opened.training.read_command_pcm16le(synthetic.corpus.train_commands[0])

    assert len(os.listdir("/proc/self/fd")) == before


def test_read_api_rejects_wrong_source_identity_and_window_bounds(
    tmp_path: Path, synthetic: _SyntheticCorpus
) -> None:
    path = tmp_path / "arguments.cache"
    _build(synthetic, path)
    command = synthetic.corpus.train_commands[0]
    forged_command = _command(
        command.manifest_index,
        command.path,
        synthetic.payloads[command.path] + b"\0\0",
    )
    background = synthetic.corpus.train_backgrounds[0]

    with _opened(synthetic, path) as opened:
        with pytest.raises(
            pcm_cache.Experiment002PCMCacheError, match="source|identity"
        ):
            opened.training.read_command_pcm16le(forged_command)
        for start_sample in (-1, background.sample_count - 16_000 + 1):
            with pytest.raises(
                pcm_cache.Experiment002PCMCacheError, match="window|start|outside"
            ):
                opened.training.read_background_window_pcm16le(background, start_sample)
        with pytest.raises(TypeError, match="integer"):
            opened.training.read_background_window_pcm16le(background, True)


@pytest.mark.parametrize(
    ("byte_count", "sha256", "exception"),
    [
        (-1, "0" * 64, pcm_cache.Experiment002PCMCacheError),
        (0, "0" * 64, pcm_cache.Experiment002PCMCacheError),
        (True, "0" * 64, TypeError),
        (1, "0" * 63, pcm_cache.Experiment002PCMCacheError),
        (1, "G" * 64, pcm_cache.Experiment002PCMCacheError),
    ],
)
def test_external_identity_is_strict(
    byte_count: int, sha256: str, exception: type[BaseException]
) -> None:
    with pytest.raises(exception):
        pcm_cache.PCMCacheIdentity(byte_count=byte_count, sha256=sha256)


def test_open_requires_exact_external_byte_count_and_digest(
    tmp_path: Path, synthetic: _SyntheticCorpus
) -> None:
    path = tmp_path / "external.cache"
    actual = _build(synthetic, path)
    wrong_count = pcm_cache.PCMCacheIdentity(
        byte_count=actual.byte_count + 1,
        sha256=actual.sha256,
    )
    wrong_digest = pcm_cache.PCMCacheIdentity(
        byte_count=actual.byte_count,
        sha256="0" * 64 if actual.sha256 != "0" * 64 else "1" * 64,
    )

    for expected in (wrong_count, wrong_digest):
        with pytest.raises(
            pcm_cache.Experiment002PCMCacheError, match="external|identity|digest|byte"
        ):
            pcm_cache.Experiment002PCMCache._open(
                synthetic.corpus,
                path,
                expected,
                layout=_layout(synthetic),
            )


def test_open_and_reads_handle_partial_pread_results(
    tmp_path: Path,
    synthetic: _SyntheticCorpus,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "partial-read.cache"
    expected = _build(synthetic, path)
    original_pread = os.pread
    partial_calls = 0

    def partial_pread(descriptor: int, length: int, offset: int) -> bytes:
        nonlocal partial_calls
        if length > 1:
            partial_calls += 1
            length = max(1, length // 2)
        return original_pread(descriptor, length, offset)

    monkeypatch.setattr(os, "pread", partial_pread)

    with _opened(synthetic, path, expected) as opened:
        assert (
            opened.training.read_command_pcm16le(synthetic.corpus.train_commands[0])
            == synthetic.payloads[synthetic.corpus.train_commands[0].path]
        )

    assert partial_calls > 0


def test_mutation_during_full_file_verification_is_rejected(
    tmp_path: Path,
    synthetic: _SyntheticCorpus,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "verify-race.cache"
    expected = _build(synthetic, path)
    original_pread = os.pread
    mutated = False

    def mutate_after_first_read(descriptor: int, length: int, offset: int) -> bytes:
        nonlocal mutated
        result = original_pread(descriptor, length, offset)
        if not mutated and offset == 0 and result:
            mutated = True
            status = path.stat()
            os.utime(
                path,
                ns=(status.st_atime_ns, status.st_mtime_ns + 1_000_000_000),
            )
        return result

    monkeypatch.setattr(os, "pread", mutate_after_first_read)

    with pytest.raises(pcm_cache.Experiment002PCMCacheError, match="changed"):
        pcm_cache.Experiment002PCMCache._open(
            synthetic.corpus,
            path,
            expected,
            layout=_layout(synthetic),
        )
    assert mutated


def test_cache_module_has_no_bulk_scanner_mmap_or_training_dependency() -> None:
    module_path = Path(pcm_cache.__file__)
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    called_attributes: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(
                alias.name.split(".", maxsplit=1)[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module.split(".", maxsplit=1)[0])
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            called_attributes.add(node.func.attr)

    assert imported.isdisjoint(
        {"glob", "mmap", "numpy", "onnx", "onnxruntime", "torch"}
    )
    assert called_attributes.isdisjoint({"glob", "rglob", "walk"})
    assert "os.replace" not in source
    assert "white_noise.wav" not in source


def test_builder_handles_partial_pwrite_results(
    tmp_path: Path,
    synthetic: _SyntheticCorpus,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "partial.cache"
    original_pwrite = os.pwrite
    partial_pwrites = 0

    def partial_pwrite(
        descriptor: int,
        data: bytes | bytearray | memoryview,
        offset: int,
    ) -> int:
        nonlocal partial_pwrites
        view = memoryview(data)
        if len(view) > 1:
            partial_pwrites += 1
            view = view[: max(1, len(view) // 2)]
        return original_pwrite(descriptor, view, offset)

    monkeypatch.setattr(os, "pwrite", partial_pwrite)

    built = _build(synthetic, destination)

    assert built == _identity(destination)
    assert partial_pwrites > 0
    with _opened(synthetic, destination, built) as opened:
        assert (
            opened.training.read_command_pcm16le(synthetic.corpus.train_commands[0])
            == synthetic.payloads[synthetic.corpus.train_commands[0].path]
        )


@pytest.mark.parametrize("operation", ["pwrite", "fsync", "link"])
def test_publish_failure_removes_partial_inode_and_destination(
    tmp_path: Path,
    synthetic: _SyntheticCorpus,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    destination = tmp_path / f"failed-{operation}.cache"

    if operation == "pwrite":

        def fail_pwrite(
            descriptor: int,
            data: bytes | bytearray | memoryview,
            offset: int,
        ) -> int:
            raise OSError("injected pwrite failure")

        monkeypatch.setattr(os, "pwrite", fail_pwrite)
    elif operation == "fsync":

        def fail_fsync(descriptor: int) -> None:
            raise OSError("injected fsync failure")

        monkeypatch.setattr(os, "fsync", fail_fsync)
    else:

        def fail_link(*args: object, **kwargs: object) -> None:
            raise OSError("injected link failure")

        monkeypatch.setattr(os, "link", fail_link)

    with pytest.raises(pcm_cache.Experiment002PCMCacheError):
        _build(synthetic, destination)

    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_parent_fsync_failure_preserves_committed_destination_and_cleans_temp(
    tmp_path: Path,
    synthetic: _SyntheticCorpus,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "parent-fsync.cache"
    original_fsync = os.fsync
    calls = 0

    def fail_parent_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("injected parent fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_parent_fsync)

    with pytest.raises(pcm_cache.Experiment002PCMCacheError):
        _build(synthetic, destination)

    assert calls == 3
    assert [path for path in tmp_path.iterdir()] == [destination]
    observed = _identity(destination)
    with _opened(synthetic, destination, observed) as opened:
        source = synthetic.corpus.train_commands[0]
        assert (
            opened.training.read_command_pcm16le(source)
            == synthetic.payloads[source.path]
        )


def test_temporary_unlink_failure_after_link_preserves_committed_destination(
    tmp_path: Path,
    synthetic: _SyntheticCorpus,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "unlink-failure.cache"
    original_unlink = os.unlink
    failed_once = False

    def fail_temporary_once(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        dir_fd: int | None = None,
    ) -> None:
        nonlocal failed_once
        if not failed_once and os.fsdecode(path).endswith(".tmp"):
            failed_once = True
            raise OSError("injected temporary unlink failure")
        original_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(os, "unlink", fail_temporary_once)

    with pytest.raises(pcm_cache.Experiment002PCMCacheError):
        _build(synthetic, destination)

    assert failed_once
    assert [path for path in tmp_path.iterdir()] == [destination]
    observed = _identity(destination)
    with _opened(synthetic, destination, observed) as opened:
        source = synthetic.corpus.validation_commands[0]
        assert (
            opened.validation.read_command_pcm16le(source)
            == synthetic.payloads[source.path]
        )


def test_publish_uses_atomic_same_directory_no_overwrite_link_and_parent_fsync(
    tmp_path: Path,
    synthetic: _SyntheticCorpus,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "published.cache"
    original_link = os.link
    original_fsync = os.fsync
    link_observations: list[
        tuple[
            str | bytes | os.PathLike[str] | os.PathLike[bytes],
            str | bytes | os.PathLike[str] | os.PathLike[bytes],
            int | None,
            int | None,
            bool,
        ]
    ] = []
    fsync_modes: list[int] = []

    def observe_link(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        destination_name: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        link_observations.append(
            (
                source,
                destination_name,
                src_dir_fd,
                dst_dir_fd,
                follow_symlinks,
            )
        )
        original_link(
            source,
            destination_name,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    def observe_fsync(descriptor: int) -> None:
        fsync_modes.append(os.fstat(descriptor).st_mode)
        original_fsync(descriptor)

    monkeypatch.setattr(os, "link", observe_link)
    monkeypatch.setattr(os, "fsync", observe_fsync)

    _build(synthetic, destination)

    assert len(link_observations) == 1
    source, leaf, source_fd, destination_fd, follows = link_observations[0]
    assert os.fspath(leaf) == destination.name
    assert os.fspath(source) != os.fspath(destination)
    assert source_fd is not None and source_fd == destination_fd
    assert not follows
    assert any(stat.S_ISREG(mode) for mode in fsync_modes)
    assert stat.S_ISDIR(fsync_modes[-1])


def test_destination_name_replacement_after_link_is_detected_without_deleting_replacer(
    tmp_path: Path,
    synthetic: _SyntheticCorpus,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "raced.cache"
    held = tmp_path / "attacker-held.cache"
    sentinel = b"replacement owned by concurrent publisher"
    original_link = os.link

    def replace_after_link(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        destination_name: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        original_link(
            source,
            destination_name,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )
        if dst_dir_fd is None:
            raise AssertionError("publisher did not pin the destination parent")
        os.rename(
            destination_name,
            held.name,
            src_dir_fd=dst_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )
        descriptor = os.open(
            destination.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o600,
            dir_fd=dst_dir_fd,
        )
        try:
            assert os.write(descriptor, sentinel) == len(sentinel)
        finally:
            os.close(descriptor)

    monkeypatch.setattr(os, "link", replace_after_link)

    with pytest.raises(
        pcm_cache.Experiment002PCMCacheError, match="published|inode|name"
    ):
        _build(synthetic, destination)

    assert destination.read_bytes() == sentinel
    assert held.exists()
    assert _identity(held).byte_count == _layout(synthetic).file_bytes
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        [destination.name, held.name]
    )


def test_read_close_fd_reuse_cannot_redirect_a_command_read(
    tmp_path: Path,
    synthetic: _SyntheticCorpus,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "race.cache"
    _build(synthetic, path)
    source = synthetic.corpus.train_commands[1]
    expected_payload = synthetic.payloads[source.path]
    attacker = tmp_path / "attacker.cache"
    attacker_contents = bytearray(path.read_bytes())
    payload_offset = _layout(synthetic).payload_offset
    attacker_contents[payload_offset : payload_offset + len(expected_payload)] = (
        b"\x55" * len(expected_payload)
    )
    attacker.write_bytes(attacker_contents)
    opened = pcm_cache.Experiment002PCMCache._open(
        synthetic.corpus, path, _identity(path), layout=_layout(synthetic)
    )
    original_pread = os.pread
    entered = threading.Event()
    release = threading.Event()
    blocked_fd = -1

    def blocked_pread(descriptor: int, length: int, offset: int) -> bytes:
        nonlocal blocked_fd
        if offset == _layout(synthetic).payload_offset and length == len(
            expected_payload
        ):
            blocked_fd = descriptor
            entered.set()
            assert release.wait(timeout=10)
        return original_pread(descriptor, length, offset)

    monkeypatch.setattr(os, "pread", blocked_pread)
    attacker_fds: list[int] = []
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            read_future = executor.submit(opened.training.read_command_pcm16le, source)
            assert entered.wait(timeout=10)
            close_future = executor.submit(opened.close)
            try:
                close_future.result(timeout=0.2)
            except TimeoutError:
                closed_during_read = False
            else:
                closed_during_read = True
            if closed_during_read:
                for _ in range(64):
                    descriptor = os.open(attacker, os.O_RDONLY | os.O_CLOEXEC)
                    attacker_fds.append(descriptor)
                    if descriptor == blocked_fd:
                        break
            release.set()
            assert read_future.result(timeout=10) == expected_payload
            close_future.result(timeout=10)
    finally:
        release.set()
        for descriptor in attacker_fds:
            os.close(descriptor)
        opened.close()
