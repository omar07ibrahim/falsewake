"""Deterministic, identity-bound PCM cache for Experiment 002.

The cache is an operational acceleration artifact rather than a trust root.  A
reader therefore requires the byte count and SHA-256 digest supplied out of
band by the builder.  Paths are never stored in the cache and there is no
generic path lookup API: callers can read only sources retained by the exact
training or validation corpus used while opening it.
"""

from __future__ import annotations

import errno
import hashlib
import hmac
import itertools
import os
import stat
import struct
import threading
import weakref
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Final, Literal, Protocol

from falsewake.experiment_002_data import (
    MANIFEST_SHA256,
    TRAIN_COMMAND_INVENTORY_SHA256,
    VALIDATION_COMMAND_INVENTORY_SHA256,
    WINDOW_SAMPLES,
    BackgroundSource,
    CommandSource,
    Experiment002Corpus,
    Experiment002DataError,
    _require_registered_training_corpus,
    _require_registered_validation_corpus,
)
from falsewake.speech_commands_pcm import (
    PCMSourceIdentity,
    SpeechCommandsPCMError,
    SpeechCommandsPCMLoader,
    VerifiedPCM16LE,
)

HEADER_BYTES: Final = 320
ENTRY_BYTES: Final = 96
PAYLOAD_ALIGNMENT: Final = 4_096
REGISTERED_ENTRY_COUNT: Final = 94_829
REGISTERED_PAYLOAD_BYTES: Final = 2_986_671_540
REGISTERED_PAYLOAD_OFFSET: Final = 9_105_408
REGISTERED_FILE_BYTES: Final = 2_995_776_948

_MAGIC: Final = b"FALSEWAKEPCM002\0"
_VERSION: Final = 1
_ROLE_TRAIN_COMMAND: Final = 0
_ROLE_VALIDATION_COMMAND: Final = 1
_ROLE_TRAIN_BACKGROUND: Final = 2
_ROLE_VALIDATION_BACKGROUND: Final = 3
_REGISTERED_ROLE_COUNTS: Final = (84_843, 9_981, 4, 1)
_MANIFEST_INVENTORY_SHA256: Final = (
    "c9596927b6de1aa9bb8174bea25f613ebb7bab9e671304c525961b66e0812c5c"
)
_ARCHIVE_SHA256: Final = (
    "af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58"
)
_OUTPUT_BYTES_MAXIMUM: Final = 4 * 1_024**3
_HASH_CHUNK_BYTES: Final = 1 * 1_024**2
_ZERO32: Final = bytes(32)
_ENTRY_STRUCT: Final = struct.Struct("<B3sIIIQ32s32s8s")
_TEMP_COUNTER = itertools.count()
_TEMP_COUNTER_LOCK = threading.Lock()
_OPEN_SUPPORTS_DIR_FD: Final = os.open in os.supports_dir_fd
_LINK_SUPPORTS_DIR_FD: Final = os.link in os.supports_dir_fd
_STAT_SUPPORTS_DIR_FD: Final = os.stat in os.supports_dir_fd
_UNLINK_SUPPORTS_DIR_FD: Final = os.unlink in os.supports_dir_fd

type _Role = Literal[0, 1, 2, 3]
type _Split = Literal["train", "validation"]
type _Source = CommandSource | BackgroundSource


class Experiment002PCMCacheError(ValueError):
    """A cache, corpus, source, or filesystem operation violated the contract."""


@dataclass(frozen=True, slots=True)
class PCMCacheIdentity:
    """The external whole-file identity required by every cache reader."""

    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        _require_positive_integer("byte_count", self.byte_count)
        if self.byte_count > 0xFFFF_FFFF_FFFF_FFFF:
            raise Experiment002PCMCacheError("byte_count exceeds uint64")
        _validate_sha256(self.sha256)


@dataclass(frozen=True, slots=True)
class _CacheLayout:
    """Binary layout and provenance, with a private tiny-corpus constructor."""

    manifest_sha256: str
    manifest_inventory_sha256: str
    archive_sha256: str
    train_inventory_sha256: str
    validation_inventory_sha256: str
    role_counts: tuple[int, int, int, int]
    payload_bytes: int
    output_bytes_maximum: int = _OUTPUT_BYTES_MAXIMUM

    def __post_init__(self) -> None:
        for name, value in (
            ("manifest_sha256", self.manifest_sha256),
            ("manifest_inventory_sha256", self.manifest_inventory_sha256),
            ("archive_sha256", self.archive_sha256),
            ("train_inventory_sha256", self.train_inventory_sha256),
            ("validation_inventory_sha256", self.validation_inventory_sha256),
        ):
            try:
                _validate_sha256(value)
            except (TypeError, Experiment002PCMCacheError) as error:
                raise type(error)(f"invalid {name}: {error}") from error
        if not isinstance(self.role_counts, tuple) or len(self.role_counts) != 4:
            raise TypeError("role_counts must be a four-integer tuple")
        for role_count in self.role_counts:
            _require_nonnegative_integer("role count", role_count)
            if role_count > 0xFFFF_FFFF:
                raise Experiment002PCMCacheError("role count exceeds uint32")
        _require_nonnegative_integer("payload_bytes", self.payload_bytes)
        _require_positive_integer("output_bytes_maximum", self.output_bytes_maximum)
        if self.entry_count < 1:
            raise Experiment002PCMCacheError("cache must contain at least one entry")
        if self.entry_count > 0xFFFF_FFFF:
            raise Experiment002PCMCacheError("entry count exceeds uint32")
        if self.payload_bytes > 0xFFFF_FFFF_FFFF_FFFF:
            raise Experiment002PCMCacheError("payload size exceeds uint64")
        if self.file_bytes > 0xFFFF_FFFF_FFFF_FFFF:
            raise Experiment002PCMCacheError("cache size exceeds uint64")

    @classmethod
    def for_corpus(
        cls,
        corpus: Experiment002Corpus,
        *,
        manifest_inventory_sha256: str,
        archive_sha256: str,
        output_bytes_maximum: int = _OUTPUT_BYTES_MAXIMUM,
    ) -> _CacheLayout:
        """Derive a hermetic layout without weakening the registered public API."""

        if not isinstance(corpus, Experiment002Corpus):
            raise TypeError("corpus must be an Experiment002Corpus")
        role_counts = (
            len(corpus.train_commands),
            len(corpus.validation_commands),
            len(corpus.train_backgrounds),
            1,
        )
        sources: tuple[_Source, ...] = (
            *corpus.train_commands,
            *corpus.validation_commands,
            *corpus.train_backgrounds,
            corpus.validation_background,
        )
        payload_bytes = sum(2 * source.sample_count for source in sources)
        return cls(
            manifest_sha256=corpus.manifest_sha256,
            manifest_inventory_sha256=manifest_inventory_sha256,
            archive_sha256=archive_sha256,
            train_inventory_sha256=corpus._train_inventory_sha256,
            validation_inventory_sha256=corpus._validation_inventory_sha256,
            role_counts=role_counts,
            payload_bytes=payload_bytes,
            output_bytes_maximum=output_bytes_maximum,
        )

    @property
    def entry_count(self) -> int:
        return sum(self.role_counts)

    @property
    def payload_offset(self) -> int:
        index_end = HEADER_BYTES + self.entry_count * ENTRY_BYTES
        return _align_up(index_end, PAYLOAD_ALIGNMENT)

    @property
    def file_bytes(self) -> int:
        return self.payload_offset + self.payload_bytes


class _PCMLoader(Protocol):
    def load_command(self, source: PCMSourceIdentity) -> VerifiedPCM16LE: ...

    def load_background(self, source: PCMSourceIdentity) -> VerifiedPCM16LE: ...

    def close(self) -> None: ...


class _LoaderFactory(Protocol):
    def __call__(self, dataset_root: Path) -> _PCMLoader: ...


@dataclass(frozen=True, slots=True)
class _SourceRecord:
    role: int
    source_index: int
    source: _Source


@dataclass(frozen=True, slots=True)
class _IndexEntry:
    payload_offset: int
    sample_count: int
    pcm_sha256: bytes


@dataclass(frozen=True, slots=True)
class _ParsedIndex:
    train_commands: dict[CommandSource, _IndexEntry]
    validation_commands: dict[CommandSource, _IndexEntry]
    train_backgrounds: dict[BackgroundSource, _IndexEntry]
    validation_backgrounds: dict[BackgroundSource, _IndexEntry]


@dataclass(frozen=True, slots=True)
class PCMCacheSplitView:
    """A role-bound read-only view of one opened cache."""

    _cache: Experiment002PCMCache
    _split: _Split

    def __post_init__(self) -> None:
        if not isinstance(self._cache, Experiment002PCMCache):
            raise TypeError("cache must be an Experiment002PCMCache")
        if self._split not in {"train", "validation"}:
            raise Experiment002PCMCacheError("cache view has an invalid split")

    @property
    def split(self) -> _Split:
        """Return the only source role accepted by this view."""

        return self._split

    def read_command_pcm16le(self, source: CommandSource) -> bytes:
        """Read one exact, unpadded command payload from this split."""

        return self._cache._read_command(self._split, source)

    def read_background_window_pcm16le(
        self,
        source: BackgroundSource,
        start_sample: int,
    ) -> bytes:
        """Read one owned 16,000-sample background window from this split."""

        return self._cache._read_background_window(
            self._split,
            source,
            start_sample,
        )


class Experiment002PCMCache:
    """A verified descriptor-pinned cache with role-separated read views."""

    __slots__ = (
        "_background_payloads",
        "_descriptor",
        "_finalizer",
        "_identity",
        "_lifecycle_lock",
        "_parsed_index",
        "_path",
        "__weakref__",
    )

    _background_payloads: dict[BackgroundSource, bytes]
    _descriptor: int
    _finalizer: weakref.finalize[[int], Experiment002PCMCache]
    _identity: PCMCacheIdentity
    _lifecycle_lock: threading.Lock
    _parsed_index: _ParsedIndex
    _path: Path

    def __init__(self) -> None:
        raise TypeError("Experiment002PCMCache instances are created by open()")

    @classmethod
    def open(
        cls,
        corpus: Experiment002Corpus,
        path: Path,
        expected: PCMCacheIdentity,
    ) -> Experiment002PCMCache:
        """Open only the exact registered cache with an external identity."""

        _require_registered_corpus(corpus)
        return cls._open(corpus, path, expected, layout=_REGISTERED_LAYOUT)

    @classmethod
    def _open(
        cls,
        corpus: Experiment002Corpus,
        path: Path,
        expected: PCMCacheIdentity,
        *,
        layout: _CacheLayout,
    ) -> Experiment002PCMCache:
        """Private layout hook used by small hermetic format tests."""

        _require_layout_matches_corpus(corpus, layout)
        if not isinstance(path, Path):
            raise TypeError("path must be a Path")
        if not isinstance(expected, PCMCacheIdentity):
            raise TypeError("expected must be a PCMCacheIdentity")
        if expected.byte_count != layout.file_bytes:
            raise Experiment002PCMCacheError(
                "external cache bytes differ from the required layout"
            )

        descriptor = _open_cache_file(path)
        try:
            before = _checked_file_stat(descriptor, layout)
            whole_sha256, payload_sha256 = _hash_verified_file(
                descriptor,
                file_bytes=layout.file_bytes,
                payload_offset=layout.payload_offset,
            )
            if not hmac.compare_digest(whole_sha256, expected.sha256):
                raise Experiment002PCMCacheError(
                    "whole-file cache digest differs from the external identity: "
                    f"observed={whole_sha256}"
                )

            header = _read_exact_at(descriptor, HEADER_BYTES, 0)
            index_bytes = _read_exact_at(
                descriptor,
                layout.entry_count * ENTRY_BYTES,
                HEADER_BYTES,
            )
            padding_offset = HEADER_BYTES + len(index_bytes)
            padding = _read_exact_at(
                descriptor,
                layout.payload_offset - padding_offset,
                padding_offset,
            )
            if any(padding):
                raise Experiment002PCMCacheError("cache alignment padding is nonzero")

            _validate_header(
                header,
                layout=layout,
                index_sha256=hashlib.sha256(index_bytes).digest(),
                payload_sha256=payload_sha256,
            )
            records = _ordered_source_records(corpus)
            parsed_index = _parse_index(index_bytes, records, layout)
            background_payloads = _read_background_payloads(
                descriptor,
                parsed_index,
            )
            after = os.fstat(descriptor)
            if _stat_fingerprint(before) != _stat_fingerprint(after):
                raise Experiment002PCMCacheError(
                    "cache changed while it was being verified"
                )
            instance = cls._from_verified(
                descriptor=descriptor,
                path=path,
                identity=expected,
                parsed_index=parsed_index,
                background_payloads=background_payloads,
            )
            descriptor = -1
            return instance
        except OSError as error:
            raise Experiment002PCMCacheError(
                f"cannot verify PCM cache: {error}"
            ) from error
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    @classmethod
    def _from_verified(
        cls,
        *,
        descriptor: int,
        path: Path,
        identity: PCMCacheIdentity,
        parsed_index: _ParsedIndex,
        background_payloads: dict[BackgroundSource, bytes],
    ) -> Experiment002PCMCache:
        instance = object.__new__(cls)
        instance._path = path
        instance._identity = identity
        instance._descriptor = descriptor
        instance._parsed_index = parsed_index
        instance._background_payloads = background_payloads
        instance._lifecycle_lock = threading.Lock()
        instance._finalizer = weakref.finalize(instance, os.close, descriptor)
        return instance

    @property
    def path(self) -> Path:
        """Return the path used to open the pinned cache inode."""

        return self._path

    @property
    def identity(self) -> PCMCacheIdentity:
        """Return the externally supplied identity verified during open."""

        return self._identity

    @property
    def training(self) -> PCMCacheSplitView:
        """Return a view that accepts training sources only."""

        return PCMCacheSplitView(self, "train")

    @property
    def validation(self) -> PCMCacheSplitView:
        """Return a view that accepts validation sources only."""

        return PCMCacheSplitView(self, "validation")

    @property
    def closed(self) -> bool:
        """Return whether the pinned cache descriptor has been released."""

        with self._lifecycle_lock:
            return not self._finalizer.alive

    def close(self) -> None:
        """Release the pinned cache descriptor exactly once."""

        with self._lifecycle_lock:
            self._finalizer()

    def __enter__(self) -> Experiment002PCMCache:
        with self._lifecycle_lock:
            if not self._finalizer.alive:
                raise Experiment002PCMCacheError("PCM cache is closed")
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _read_command(self, split: _Split, source: CommandSource) -> bytes:
        if not isinstance(source, CommandSource):
            raise TypeError("source must be a CommandSource")
        entries = (
            self._parsed_index.train_commands
            if split == "train"
            else self._parsed_index.validation_commands
        )
        entry = entries.get(source)
        if entry is None:
            raise Experiment002PCMCacheError(
                f"command source identity is not a member of the {split} split view"
            )
        with self._lifecycle_lock:
            if not self._finalizer.alive:
                raise Experiment002PCMCacheError("PCM cache is closed")
            try:
                payload = _read_entry_payload(
                    self._descriptor,
                    entry,
                )
            except OSError as error:
                raise Experiment002PCMCacheError(
                    f"cannot read command PCM: {error}"
                ) from error
        return payload

    def _read_background_window(
        self,
        split: _Split,
        source: BackgroundSource,
        start_sample: int,
    ) -> bytes:
        if not isinstance(source, BackgroundSource):
            raise TypeError("source must be a BackgroundSource")
        _require_nonnegative_integer("start_sample", start_sample)
        entries = (
            self._parsed_index.train_backgrounds
            if split == "train"
            else self._parsed_index.validation_backgrounds
        )
        if source not in entries:
            raise Experiment002PCMCacheError(
                f"background source identity is not a member of the {split} split view"
            )
        if start_sample + WINDOW_SAMPLES > source.sample_count:
            raise Experiment002PCMCacheError(
                "background window falls outside its registered source"
            )
        with self._lifecycle_lock:
            if not self._finalizer.alive:
                raise Experiment002PCMCacheError("PCM cache is closed")
            payload = self._background_payloads[source]
        start_byte = 2 * start_sample
        end_byte = start_byte + 2 * WINDOW_SAMPLES
        window = payload[start_byte:end_byte]
        if len(window) != 2 * WINDOW_SAMPLES:
            raise Experiment002PCMCacheError(
                "cached background window has an invalid byte count"
            )
        return window


def build_registered_pcm_cache(
    corpus: Experiment002Corpus,
    dataset_root: Path,
    destination: Path,
) -> PCMCacheIdentity:
    """Build and atomically publish the exact registered operational cache."""

    _require_registered_corpus(corpus)
    return _build_pcm_cache(
        corpus,
        dataset_root,
        destination,
        layout=_REGISTERED_LAYOUT,
    )


def _build_pcm_cache(
    corpus: Experiment002Corpus,
    dataset_root: Path,
    destination: Path,
    *,
    layout: _CacheLayout,
    loader_factory: _LoaderFactory | None = None,
) -> PCMCacheIdentity:
    """Private small-layout builder preserving all production I/O semantics."""

    _require_layout_matches_corpus(corpus, layout)
    if not isinstance(dataset_root, Path):
        raise TypeError("dataset_root must be a Path")
    if not isinstance(destination, Path):
        raise TypeError("destination must be a Path")
    records = _ordered_source_records(corpus)
    parent, destination_leaf = _parent_and_leaf(destination)
    _require_secure_filesystem_support()
    parent_fd = _open_parent(parent)
    temporary_leaf: str | None = None
    temporary_fd = -1
    try:
        _require_destination_absent(parent_fd, destination_leaf)
        temporary_leaf, temporary_fd = _create_temporary(parent_fd)
        try:
            os.ftruncate(temporary_fd, layout.payload_offset)
            index_parts, payload_sha256, final_offset = _write_payloads(
                temporary_fd,
                dataset_root=dataset_root,
                records=records,
                payload_offset=layout.payload_offset,
                loader_factory=loader_factory,
            )
            if final_offset != layout.file_bytes:
                raise Experiment002PCMCacheError(
                    "decoded PCM payload byte count differs from the layout"
                )
            index_bytes = b"".join(index_parts)
            if len(index_bytes) != layout.entry_count * ENTRY_BYTES:
                raise Experiment002PCMCacheError(
                    "built cache index has an invalid byte count"
                )
            header = _build_header(
                layout,
                index_sha256=hashlib.sha256(index_bytes).digest(),
                payload_sha256=payload_sha256,
            )
            _write_all_at(temporary_fd, index_bytes, HEADER_BYTES)
            _write_all_at(temporary_fd, header, 0)
            os.ftruncate(temporary_fd, layout.file_bytes)
            _checked_file_stat(temporary_fd, layout)
            os.fsync(temporary_fd)
            os.fchmod(temporary_fd, 0o444)
            os.fsync(temporary_fd)
            hash_before = os.fstat(temporary_fd)
            identity_sha256, observed_payload_sha256 = _hash_verified_file(
                temporary_fd,
                file_bytes=layout.file_bytes,
                payload_offset=layout.payload_offset,
            )
            if not hmac.compare_digest(
                observed_payload_sha256,
                payload_sha256,
            ):
                raise Experiment002PCMCacheError(
                    "built cache payload changed before publication"
                )
            observed_header = _read_exact_at(temporary_fd, HEADER_BYTES, 0)
            observed_index = _read_exact_at(
                temporary_fd,
                len(index_bytes),
                HEADER_BYTES,
            )
            padding_offset = HEADER_BYTES + len(index_bytes)
            observed_padding = _read_exact_at(
                temporary_fd,
                layout.payload_offset - padding_offset,
                padding_offset,
            )
            if not hmac.compare_digest(observed_header, header):
                raise Experiment002PCMCacheError(
                    "built cache header changed before publication"
                )
            if not hmac.compare_digest(observed_index, index_bytes):
                raise Experiment002PCMCacheError(
                    "built cache index changed before publication"
                )
            if any(observed_padding):
                raise Experiment002PCMCacheError(
                    "built cache padding changed before publication"
                )
            after_hash = os.fstat(temporary_fd)
            if _stat_fingerprint(hash_before) != _stat_fingerprint(after_hash):
                raise Experiment002PCMCacheError(
                    "built cache inode changed before publication"
                )
            identity = PCMCacheIdentity(layout.file_bytes, identity_sha256)
            os.link(
                temporary_leaf,
                destination_leaf,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
            _require_published_inode(parent_fd, destination_leaf, after_hash)
            os.unlink(temporary_leaf, dir_fd=parent_fd)
            temporary_leaf = None
            os.fsync(parent_fd)
            return identity
        except FileExistsError as error:
            raise Experiment002PCMCacheError(
                "cache destination already exists"
            ) from error
        except (Experiment002PCMCacheError, SpeechCommandsPCMError):
            raise
        except OSError as error:
            raise Experiment002PCMCacheError(
                f"cannot build PCM cache: {error}"
            ) from error
    finally:
        if temporary_fd >= 0:
            os.close(temporary_fd)
        if temporary_leaf is not None:
            with suppress(FileNotFoundError):
                os.unlink(temporary_leaf, dir_fd=parent_fd)
        os.close(parent_fd)


def _write_payloads(
    descriptor: int,
    *,
    dataset_root: Path,
    records: tuple[_SourceRecord, ...],
    payload_offset: int,
    loader_factory: _LoaderFactory | None,
) -> tuple[list[bytes], bytes, int]:
    loader: _PCMLoader
    if loader_factory is None:
        loader = SpeechCommandsPCMLoader(dataset_root)
    else:
        loader = loader_factory(dataset_root)
    index_parts: list[bytes] = []
    payload_hasher = hashlib.sha256()
    offset = payload_offset
    try:
        for record in records:
            if isinstance(record.source, CommandSource):
                snapshot = loader.load_command(record.source.pcm_identity)
            else:
                snapshot = loader.load_background(record.source.pcm_identity)
            payload = _verified_payload(record.source, snapshot)
            _write_all_at(descriptor, payload, offset)
            payload_hasher.update(payload)
            pcm_sha256 = hashlib.sha256(payload).digest()
            index_parts.append(
                _ENTRY_STRUCT.pack(
                    record.role,
                    b"\0\0\0",
                    record.source_index,
                    record.source.sample_count,
                    0,
                    offset,
                    bytes.fromhex(record.source.sha256),
                    pcm_sha256,
                    bytes(8),
                )
            )
            offset += len(payload)
    finally:
        loader.close()
    return index_parts, payload_hasher.digest(), offset


def _verified_payload(
    source: _Source,
    snapshot: VerifiedPCM16LE,
) -> bytes:
    if not isinstance(snapshot, VerifiedPCM16LE):
        raise TypeError("PCM loader must return VerifiedPCM16LE")
    if (
        snapshot.path != source.path
        or snapshot.sample_count != source.sample_count
        or not hmac.compare_digest(snapshot.sha256, source.sha256)
    ):
        raise Experiment002PCMCacheError("PCM snapshot identity differs")
    if not isinstance(snapshot.payload, bytes):
        raise TypeError("verified PCM payload must be bytes")
    if len(snapshot.payload) != 2 * source.sample_count:
        raise Experiment002PCMCacheError("verified PCM payload length differs")
    return snapshot.payload


def _build_header(
    layout: _CacheLayout,
    *,
    index_sha256: bytes,
    payload_sha256: bytes,
) -> bytes:
    _require_digest_bytes("index_sha256", index_sha256)
    _require_digest_bytes("payload_sha256", payload_sha256)
    header = bytearray(HEADER_BYTES)
    header[0x000:0x010] = _MAGIC
    struct.pack_into(
        "<IIIIQQ",
        header,
        0x010,
        _VERSION,
        HEADER_BYTES,
        ENTRY_BYTES,
        layout.entry_count,
        layout.payload_offset,
        layout.payload_bytes,
    )
    header[0x030:0x050] = bytes.fromhex(layout.manifest_sha256)
    header[0x050:0x070] = bytes.fromhex(layout.manifest_inventory_sha256)
    header[0x070:0x090] = bytes.fromhex(layout.archive_sha256)
    header[0x090:0x0B0] = bytes.fromhex(layout.train_inventory_sha256)
    header[0x0B0:0x0D0] = bytes.fromhex(layout.validation_inventory_sha256)
    header[0x0D0:0x0F0] = index_sha256
    header[0x0F0:0x110] = payload_sha256
    struct.pack_into("<IIII", header, 0x110, *layout.role_counts)
    return bytes(header)


def _validate_header(
    header: bytes,
    *,
    layout: _CacheLayout,
    index_sha256: bytes,
    payload_sha256: bytes,
) -> None:
    if len(header) != HEADER_BYTES:
        raise Experiment002PCMCacheError("cache header byte count differs")
    if header[0x000:0x010] != _MAGIC:
        raise Experiment002PCMCacheError("cache magic differs")
    observed_numbers = struct.unpack_from("<IIIIQQ", header, 0x010)
    expected_numbers = (
        _VERSION,
        HEADER_BYTES,
        ENTRY_BYTES,
        layout.entry_count,
        layout.payload_offset,
        layout.payload_bytes,
    )
    if observed_numbers != expected_numbers:
        raise Experiment002PCMCacheError("cache header layout fields differ")
    expected_digests = (
        bytes.fromhex(layout.manifest_sha256),
        bytes.fromhex(layout.manifest_inventory_sha256),
        bytes.fromhex(layout.archive_sha256),
        bytes.fromhex(layout.train_inventory_sha256),
        bytes.fromhex(layout.validation_inventory_sha256),
        index_sha256,
        payload_sha256,
    )
    observed_digests = tuple(
        header[offset : offset + 32]
        for offset in (0x030, 0x050, 0x070, 0x090, 0x0B0, 0x0D0, 0x0F0)
    )
    if any(
        not hmac.compare_digest(observed, expected)
        for observed, expected in zip(
            observed_digests,
            expected_digests,
            strict=True,
        )
    ):
        raise Experiment002PCMCacheError("cache provenance or content digest differs")
    if struct.unpack_from("<IIII", header, 0x110) != layout.role_counts:
        raise Experiment002PCMCacheError("cache role counts differ")
    if header[0x120:0x140] != _ZERO32:
        raise Experiment002PCMCacheError("cache header reserved bytes are nonzero")


def _parse_index(
    index_bytes: bytes,
    records: tuple[_SourceRecord, ...],
    layout: _CacheLayout,
) -> _ParsedIndex:
    if len(records) != layout.entry_count:
        raise Experiment002PCMCacheError("corpus source count differs from layout")
    if len(index_bytes) != layout.entry_count * ENTRY_BYTES:
        raise Experiment002PCMCacheError("cache index byte count differs")
    train_commands: dict[CommandSource, _IndexEntry] = {}
    validation_commands: dict[CommandSource, _IndexEntry] = {}
    train_backgrounds: dict[BackgroundSource, _IndexEntry] = {}
    validation_backgrounds: dict[BackgroundSource, _IndexEntry] = {}
    expected_offset = layout.payload_offset
    for position, record in enumerate(records):
        raw = index_bytes[position * ENTRY_BYTES : (position + 1) * ENTRY_BYTES]
        (
            role,
            reserved3,
            source_index,
            sample_count,
            reserved4,
            payload_offset,
            wav_sha256,
            pcm_sha256,
            reserved8,
        ) = _ENTRY_STRUCT.unpack(raw)
        if reserved3 != bytes(3) or reserved4 != 0 or reserved8 != bytes(8):
            raise Experiment002PCMCacheError(
                f"cache entry {position} reserved bytes are nonzero"
            )
        if role != record.role:
            raise Experiment002PCMCacheError(f"cache entry {position} role differs")
        if source_index != record.source_index:
            raise Experiment002PCMCacheError(
                f"cache entry {position} source index differs"
            )
        if sample_count != record.source.sample_count:
            raise Experiment002PCMCacheError(
                f"cache entry {position} sample count differs"
            )
        if payload_offset != expected_offset:
            raise Experiment002PCMCacheError(
                f"cache entry {position} payload offset is not contiguous"
            )
        if not hmac.compare_digest(
            wav_sha256,
            bytes.fromhex(record.source.sha256),
        ):
            raise Experiment002PCMCacheError(
                f"cache entry {position} WAV identity differs"
            )
        entry = _IndexEntry(
            payload_offset=payload_offset,
            sample_count=sample_count,
            pcm_sha256=pcm_sha256,
        )
        if role == _ROLE_TRAIN_COMMAND:
            if not isinstance(record.source, CommandSource):
                raise Experiment002PCMCacheError("training command type differs")
            train_commands[record.source] = entry
        elif role == _ROLE_VALIDATION_COMMAND:
            if not isinstance(record.source, CommandSource):
                raise Experiment002PCMCacheError("validation command type differs")
            validation_commands[record.source] = entry
        elif role == _ROLE_TRAIN_BACKGROUND:
            if not isinstance(record.source, BackgroundSource):
                raise Experiment002PCMCacheError("training background type differs")
            train_backgrounds[record.source] = entry
        else:
            if role != _ROLE_VALIDATION_BACKGROUND or not isinstance(
                record.source,
                BackgroundSource,
            ):
                raise Experiment002PCMCacheError("validation background type differs")
            validation_backgrounds[record.source] = entry
        expected_offset += 2 * sample_count
    if expected_offset != layout.file_bytes:
        raise Experiment002PCMCacheError("cache entries do not span the payload")
    observed_counts = (
        len(train_commands),
        len(validation_commands),
        len(train_backgrounds),
        len(validation_backgrounds),
    )
    if observed_counts != layout.role_counts:
        raise Experiment002PCMCacheError("parsed cache role counts differ")
    return _ParsedIndex(
        train_commands=train_commands,
        validation_commands=validation_commands,
        train_backgrounds=train_backgrounds,
        validation_backgrounds=validation_backgrounds,
    )


def _read_background_payloads(
    descriptor: int,
    parsed: _ParsedIndex,
) -> dict[BackgroundSource, bytes]:
    result: dict[BackgroundSource, bytes] = {}
    for entries in (parsed.train_backgrounds, parsed.validation_backgrounds):
        for source, entry in entries.items():
            result[source] = _read_entry_payload(descriptor, entry)
    return result


def _read_entry_payload(descriptor: int, entry: _IndexEntry) -> bytes:
    byte_count = 2 * entry.sample_count
    payload = _read_exact_at(descriptor, byte_count, entry.payload_offset)
    observed = hashlib.sha256(payload).digest()
    if not hmac.compare_digest(observed, entry.pcm_sha256):
        raise Experiment002PCMCacheError("cached PCM payload digest differs")
    return payload


def _ordered_source_records(
    corpus: Experiment002Corpus,
) -> tuple[_SourceRecord, ...]:
    if not isinstance(corpus, Experiment002Corpus):
        raise TypeError("corpus must be an Experiment002Corpus")
    records: list[_SourceRecord] = []
    for role, commands in (
        (_ROLE_TRAIN_COMMAND, corpus.train_commands),
        (_ROLE_VALIDATION_COMMAND, corpus.validation_commands),
    ):
        for source in sorted(commands, key=lambda item: item.manifest_index):
            records.append(
                _SourceRecord(
                    role=role,
                    source_index=source.manifest_index,
                    source=source,
                )
            )
    for role, backgrounds in (
        (_ROLE_TRAIN_BACKGROUND, corpus.train_backgrounds),
        (_ROLE_VALIDATION_BACKGROUND, (corpus.validation_background,)),
    ):
        ordered = sorted(backgrounds, key=lambda item: item.path.encode("utf-8"))
        records.extend(
            _SourceRecord(role=role, source_index=index, source=source)
            for index, source in enumerate(ordered)
        )
    return tuple(records)


def _require_registered_corpus(corpus: Experiment002Corpus) -> None:
    if not isinstance(corpus, Experiment002Corpus):
        raise TypeError("corpus must be an Experiment002Corpus")
    try:
        _require_registered_training_corpus(corpus)
        _require_registered_validation_corpus(corpus)
    except Experiment002DataError as error:
        raise Experiment002PCMCacheError(
            f"corpus does not match the registered cache population: {error}"
        ) from error
    derived = _CacheLayout.for_corpus(
        corpus,
        manifest_inventory_sha256=_MANIFEST_INVENTORY_SHA256,
        archive_sha256=_ARCHIVE_SHA256,
    )
    if derived != _REGISTERED_LAYOUT:
        raise Experiment002PCMCacheError(
            "corpus byte totals or provenance differ from the registered cache"
        )


def _require_layout_matches_corpus(
    corpus: Experiment002Corpus,
    layout: _CacheLayout,
) -> None:
    if not isinstance(corpus, Experiment002Corpus):
        raise TypeError("corpus must be an Experiment002Corpus")
    if not isinstance(layout, _CacheLayout):
        raise TypeError("layout must be a _CacheLayout")
    derived = _CacheLayout.for_corpus(
        corpus,
        manifest_inventory_sha256=layout.manifest_inventory_sha256,
        archive_sha256=layout.archive_sha256,
        output_bytes_maximum=layout.output_bytes_maximum,
    )
    if derived != layout:
        raise Experiment002PCMCacheError("layout differs from the supplied corpus")
    if layout.file_bytes > layout.output_bytes_maximum:
        raise Experiment002PCMCacheError(
            "cache exceeds its output byte budget maximum: "
            f"size={layout.file_bytes}, limit={layout.output_bytes_maximum}"
        )
    records = _ordered_source_records(corpus)
    if len(records) != layout.entry_count:
        raise Experiment002PCMCacheError("corpus entry count differs from layout")
    for record in records:
        if record.role not in {
            _ROLE_TRAIN_COMMAND,
            _ROLE_VALIDATION_COMMAND,
            _ROLE_TRAIN_BACKGROUND,
            _ROLE_VALIDATION_BACKGROUND,
        }:
            raise Experiment002PCMCacheError("corpus source role is invalid")
        if not 0 <= record.source_index <= 0xFFFF_FFFF:
            raise Experiment002PCMCacheError("source index exceeds uint32")
        if not 1 <= record.source.sample_count <= 0xFFFF_FFFF:
            raise Experiment002PCMCacheError("source sample count exceeds uint32")
    if sum(2 * record.source.sample_count for record in records) != (
        layout.payload_bytes
    ):
        raise Experiment002PCMCacheError("corpus payload bytes differ from layout")


def _open_cache_file(path: Path) -> int:
    _require_secure_filesystem_support()
    parent, leaf = _parent_and_leaf(path)
    parent_fd = _open_parent(parent)
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
    try:
        try:
            descriptor = os.open(leaf, flags, dir_fd=parent_fd)
        except OSError as error:
            raise Experiment002PCMCacheError(
                f"cannot securely open PCM cache {path}: {error}"
            ) from error
    finally:
        os.close(parent_fd)
    try:
        source_stat = os.fstat(descriptor)
        if not stat.S_ISREG(source_stat.st_mode):
            raise Experiment002PCMCacheError("PCM cache is not a regular file")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _open_parent(parent: Path) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        descriptor = os.open(parent, flags)
    except (OSError, ValueError, UnicodeError) as error:
        raise Experiment002PCMCacheError(
            f"cannot securely open cache parent {parent}: {error}"
        ) from error
    try:
        parent_stat = os.fstat(descriptor)
        if not stat.S_ISDIR(parent_stat.st_mode):
            raise Experiment002PCMCacheError("cache parent is not a directory")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _parent_and_leaf(path: Path) -> tuple[Path, str]:
    leaf = path.name
    if not leaf or leaf in {".", ".."} or "/" in leaf or "\0" in leaf:
        raise Experiment002PCMCacheError(
            f"cache destination must have one valid leaf: {path}"
        )
    try:
        leaf.encode("utf-8")
    except UnicodeEncodeError as error:
        raise Experiment002PCMCacheError(
            "cache leaf must encode as valid UTF-8"
        ) from error
    return path.parent, leaf


def _require_destination_absent(parent_fd: int, leaf: str) -> None:
    try:
        os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as error:
        raise Experiment002PCMCacheError(
            f"cannot inspect cache destination: {error}"
        ) from error
    exists_error = FileExistsError(
        errno.EEXIST,
        "cache destination already exists",
        leaf,
    )
    raise Experiment002PCMCacheError("cache destination already exists") from (
        exists_error
    )


def _create_temporary(parent_fd: int) -> tuple[str, int]:
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | os.O_NOFOLLOW
        | os.O_CLOEXEC
        | os.O_NONBLOCK
    )
    for _ in range(1_024):
        with _TEMP_COUNTER_LOCK:
            counter = next(_TEMP_COUNTER)
        leaf = f".falsewake-pcm-{os.getpid():x}-{counter:x}.tmp"
        try:
            descriptor = os.open(leaf, flags, 0o600, dir_fd=parent_fd)
        except FileExistsError:
            continue
        try:
            source_stat = os.fstat(descriptor)
            if not stat.S_ISREG(source_stat.st_mode):
                raise Experiment002PCMCacheError(
                    "new cache temporary is not a regular file"
                )
        except BaseException:
            os.close(descriptor)
            with suppress(FileNotFoundError):
                os.unlink(leaf, dir_fd=parent_fd)
            raise
        return leaf, descriptor
    raise Experiment002PCMCacheError("cannot allocate a unique cache temporary")


def _require_published_inode(
    parent_fd: int,
    destination_leaf: str,
    expected: os.stat_result,
) -> None:
    observed = os.stat(
        destination_leaf,
        dir_fd=parent_fd,
        follow_symlinks=False,
    )
    if (
        not stat.S_ISREG(observed.st_mode)
        or observed.st_dev != expected.st_dev
        or observed.st_ino != expected.st_ino
        or observed.st_size != expected.st_size
    ):
        raise Experiment002PCMCacheError(
            "published cache name does not reference the built inode"
        )


def _checked_file_stat(
    descriptor: int,
    layout: _CacheLayout,
) -> os.stat_result:
    source_stat = os.fstat(descriptor)
    if not stat.S_ISREG(source_stat.st_mode):
        raise Experiment002PCMCacheError("PCM cache is not a regular file")
    if source_stat.st_size != layout.file_bytes:
        raise Experiment002PCMCacheError(
            "PCM cache byte count differs: "
            f"observed={source_stat.st_size}, expected={layout.file_bytes}"
        )
    return source_stat


def _hash_verified_file(
    descriptor: int,
    *,
    file_bytes: int,
    payload_offset: int,
) -> tuple[str, bytes]:
    whole_hasher = hashlib.sha256()
    payload_hasher = hashlib.sha256()
    offset = 0
    while offset < file_bytes:
        request = min(_HASH_CHUNK_BYTES, file_bytes - offset)
        try:
            chunk = os.pread(descriptor, request, offset)
        except InterruptedError:
            continue
        if not chunk:
            raise Experiment002PCMCacheError(f"PCM cache is truncated at byte {offset}")
        whole_hasher.update(chunk)
        chunk_end = offset + len(chunk)
        if chunk_end > payload_offset:
            first_payload_byte = max(payload_offset, offset)
            payload_hasher.update(chunk[first_payload_byte - offset :])
        offset = chunk_end
    if os.pread(descriptor, 1, file_bytes):
        raise Experiment002PCMCacheError("PCM cache has trailing bytes")
    return whole_hasher.hexdigest(), payload_hasher.digest()


def _read_exact_at(descriptor: int, byte_count: int, offset: int) -> bytes:
    _require_nonnegative_integer("byte_count", byte_count)
    _require_nonnegative_integer("offset", offset)
    result = bytearray()
    while len(result) < byte_count:
        try:
            chunk = os.pread(
                descriptor,
                byte_count - len(result),
                offset + len(result),
            )
        except InterruptedError:
            continue
        if not chunk:
            raise Experiment002PCMCacheError(
                f"PCM cache is truncated at byte {offset + len(result)}"
            )
        result.extend(chunk)
    return bytes(result)


def _write_all_at(descriptor: int, contents: bytes, offset: int) -> None:
    _require_nonnegative_integer("offset", offset)
    written = 0
    view = memoryview(contents)
    while written < len(view):
        try:
            count = os.pwrite(descriptor, view[written:], offset + written)
        except InterruptedError:
            continue
        if count <= 0:
            raise Experiment002PCMCacheError(
                f"short PCM cache write at byte {offset + written}"
            )
        written += count


def _require_secure_filesystem_support() -> None:
    missing = [
        name
        for name in ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")
        if not hasattr(os, name)
    ]
    supports_dir_fd = (
        _OPEN_SUPPORTS_DIR_FD
        and _LINK_SUPPORTS_DIR_FD
        and _STAT_SUPPORTS_DIR_FD
        and _UNLINK_SUPPORTS_DIR_FD
    )
    if missing or not supports_dir_fd:
        detail = ", ".join(missing) if missing else "descriptor-relative operations"
        raise Experiment002PCMCacheError(
            f"secure PCM cache filesystem operations are unavailable: {detail}"
        )


def _stat_fingerprint(source_stat: os.stat_result) -> tuple[int, ...]:
    return (
        source_stat.st_dev,
        source_stat.st_ino,
        source_stat.st_mode,
        source_stat.st_size,
        source_stat.st_mtime_ns,
        source_stat.st_ctime_ns,
    )


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _require_digest_bytes(name: str, value: bytes) -> None:
    if not isinstance(value, bytes) or len(value) != 32:
        raise TypeError(f"{name} must contain exactly 32 bytes")


def _validate_sha256(value: str) -> None:
    if not isinstance(value, str):
        raise TypeError("SHA-256 digest must be a string")
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise Experiment002PCMCacheError(
            "SHA-256 digest must use 64 lowercase hexadecimal characters"
        )


def _require_nonnegative_integer(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise Experiment002PCMCacheError(f"{name} must be nonnegative")


def _require_positive_integer(name: str, value: int) -> None:
    _require_nonnegative_integer(name, value)
    if value < 1:
        raise Experiment002PCMCacheError(f"{name} must be positive")


_REGISTERED_LAYOUT: Final = _CacheLayout(
    manifest_sha256=MANIFEST_SHA256,
    manifest_inventory_sha256=_MANIFEST_INVENTORY_SHA256,
    archive_sha256=_ARCHIVE_SHA256,
    train_inventory_sha256=TRAIN_COMMAND_INVENTORY_SHA256,
    validation_inventory_sha256=VALIDATION_COMMAND_INVENTORY_SHA256,
    role_counts=_REGISTERED_ROLE_COUNTS,
    payload_bytes=REGISTERED_PAYLOAD_BYTES,
)

if (
    _REGISTERED_LAYOUT.entry_count != REGISTERED_ENTRY_COUNT
    or _REGISTERED_LAYOUT.payload_offset != REGISTERED_PAYLOAD_OFFSET
    or _REGISTERED_LAYOUT.file_bytes != REGISTERED_FILE_BYTES
):
    raise RuntimeError("registered PCM cache size constants are inconsistent")
