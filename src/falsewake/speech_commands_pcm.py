"""Secure, identity-bound PCM snapshots from an extracted Speech Commands tree."""

from __future__ import annotations

import hashlib
import hmac
import io
import os
import stat
import threading
import wave
import weakref
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import TracebackType

import numpy as np

from falsewake.features import FloatArray, pcm16le_to_float32
from falsewake.speech_commands import (
    MAX_BACKGROUND_FILE_BYTES,
    MAX_COMMAND_FILE_BYTES,
)

COMMAND_SAMPLE_LIMIT = 16_000
SAMPLE_RATE = 16_000


class SpeechCommandsPCMError(ValueError):
    """A source path or WAV snapshot does not match its audited identity."""


@dataclass(frozen=True, slots=True)
class PCMSourceIdentity:
    """The manifest fields required to bind one source WAV."""

    path: str
    sample_count: int
    sha256: str

    def __post_init__(self) -> None:
        _canonical_parts(self.path)
        _validate_sample_count(self.sample_count)
        _validate_sha256(self.sha256)


@dataclass(frozen=True, slots=True)
class VerifiedPCM16LE:
    """An immutable raw PCM payload decoded from one verified WAV snapshot."""

    path: str
    sample_count: int
    sha256: str
    payload: bytes

    def __post_init__(self) -> None:
        _canonical_parts(self.path)
        _validate_sample_count(self.sample_count)
        _validate_sha256(self.sha256)
        if not isinstance(self.payload, bytes):
            raise TypeError("payload must be bytes")
        expected_bytes = self.sample_count * 2
        if len(self.payload) != expected_bytes:
            raise SpeechCommandsPCMError(
                f"PCM payload must contain exactly {expected_bytes} bytes"
            )

    def to_float32(self) -> FloatArray:
        """Return a fresh C-contiguous array using the fixed 1/32768 scale."""

        waveform = pcm16le_to_float32(self.payload)
        if waveform.dtype != np.dtype(np.float32) or waveform.shape != (
            self.sample_count,
        ):
            raise SpeechCommandsPCMError("PCM conversion returned an invalid array")
        if not waveform.flags.c_contiguous or not np.all(np.isfinite(waveform)):
            raise SpeechCommandsPCMError("PCM conversion returned invalid values")
        return waveform


class SpeechCommandsPCMLoader:
    """Load WAVs below a directory descriptor pinned for this loader's lifetime."""

    def __init__(self, dataset_root: Path) -> None:
        if not isinstance(dataset_root, Path):
            raise TypeError("dataset_root must be a Path")
        self._require_secure_open_support()
        self._dataset_root = dataset_root
        self._lifecycle_lock = threading.Lock()
        root_fd = self._open_root(dataset_root)
        self._root_fd = root_fd
        self._finalizer = weakref.finalize(self, os.close, root_fd)

    @property
    def dataset_root(self) -> Path:
        """Return the path whose directory descriptor was pinned."""

        return self._dataset_root

    @property
    def closed(self) -> bool:
        """Return whether the pinned directory descriptor has been released."""

        with self._lifecycle_lock:
            return not self._finalizer.alive

    def close(self) -> None:
        """Release the pinned dataset directory descriptor once."""

        with self._lifecycle_lock:
            self._finalizer()

    def __enter__(self) -> SpeechCommandsPCMLoader:
        with self._lifecycle_lock:
            if not self._finalizer.alive:
                raise SpeechCommandsPCMError("PCM loader is closed")
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def load_command(self, source: PCMSourceIdentity) -> VerifiedPCM16LE:
        """Load one non-background command of at most 16,000 samples."""

        return self._load(source, background=False)

    def load_background(self, source: PCMSourceIdentity) -> VerifiedPCM16LE:
        """Load one complete registered background recording."""

        return self._load(source, background=True)

    def _load(self, source: PCMSourceIdentity, *, background: bool) -> VerifiedPCM16LE:
        if not isinstance(source, PCMSourceIdentity):
            raise TypeError("source must be PCMSourceIdentity")
        parts = _canonical_parts(source.path)
        is_background_path = parts[0] == "_background_noise_"
        if background != is_background_path:
            expected_kind = "background" if background else "command"
            raise SpeechCommandsPCMError(
                f"{expected_kind} source has an invalid path: {source.path!r}"
            )
        if not background and source.sample_count > COMMAND_SAMPLE_LIMIT:
            raise SpeechCommandsPCMError(
                f"command exceeds {COMMAND_SAMPLE_LIMIT} samples: {source.path!r}"
            )

        size_limit = MAX_BACKGROUND_FILE_BYTES if background else MAX_COMMAND_FILE_BYTES
        root_fd = self._duplicate_root()
        try:
            contents = self._read_snapshot(
                parts,
                size_limit=size_limit,
                root_fd=root_fd,
            )
        finally:
            os.close(root_fd)
        observed_sha256 = hashlib.sha256(contents).hexdigest()
        if not hmac.compare_digest(observed_sha256, source.sha256):
            raise SpeechCommandsPCMError(
                f"source digest differs for {source.path!r}: observed={observed_sha256}"
            )
        payload = _decode_pcm_payload(contents, source)
        return VerifiedPCM16LE(
            path=source.path,
            sample_count=source.sample_count,
            sha256=source.sha256,
            payload=payload,
        )

    def _duplicate_root(self) -> int:
        with self._lifecycle_lock:
            if not self._finalizer.alive:
                raise SpeechCommandsPCMError("PCM loader is closed")
            try:
                return os.dup(self._root_fd)
            except OSError as error:
                raise SpeechCommandsPCMError(
                    f"cannot duplicate the pinned dataset root: {error}"
                ) from error

    def _read_snapshot(
        self,
        parts: tuple[str, str],
        *,
        size_limit: int,
        root_fd: int,
    ) -> bytes:
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        file_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
        try:
            directory_fd = os.open(parts[0], directory_flags, dir_fd=root_fd)
        except OSError as error:
            raise SpeechCommandsPCMError(
                f"cannot securely open source directory {parts[0]!r}: {error}"
            ) from error
        try:
            try:
                source_fd = os.open(parts[1], file_flags, dir_fd=directory_fd)
            except OSError as error:
                raise SpeechCommandsPCMError(
                    f"cannot securely open source {parts[0]}/{parts[1]}: {error}"
                ) from error
        finally:
            os.close(directory_fd)

        try:
            before = os.fstat(source_fd)
            if not stat.S_ISREG(before.st_mode):
                raise SpeechCommandsPCMError("source is not a regular file")
            if before.st_size > size_limit:
                raise SpeechCommandsPCMError(
                    "source exceeds its size limit: "
                    f"size={before.st_size}, limit={size_limit}"
                )
            source_file = os.fdopen(source_fd, "rb", closefd=True)
            source_fd = -1
            with source_file:
                contents = source_file.read(size_limit + 1)
                after = os.fstat(source_file.fileno())
        except OSError as error:
            raise SpeechCommandsPCMError(
                f"cannot read source snapshot: {error}"
            ) from error
        finally:
            if source_fd >= 0:
                os.close(source_fd)

        if len(contents) > size_limit:
            raise SpeechCommandsPCMError(
                f"source exceeds its size limit while reading: limit={size_limit}"
            )
        if len(contents) != before.st_size or _stat_fingerprint(before) != (
            _stat_fingerprint(after)
        ):
            raise SpeechCommandsPCMError("source changed while reading")
        return contents

    @staticmethod
    def _require_secure_open_support() -> None:
        required_flags = ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")
        missing = [name for name in required_flags if not hasattr(os, name)]
        if missing or os.open not in os.supports_dir_fd:
            details = ", ".join(missing) if missing else "dir_fd"
            raise SpeechCommandsPCMError(
                f"secure descriptor-relative opens are unavailable: {details}"
            )

    @staticmethod
    def _open_root(dataset_root: Path) -> int:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        try:
            root_fd = os.open(dataset_root, flags)
        except (OSError, ValueError) as error:
            raise SpeechCommandsPCMError(
                f"cannot securely open dataset root {dataset_root}: {error}"
            ) from error
        try:
            root_stat = os.fstat(root_fd)
        except OSError as error:
            os.close(root_fd)
            raise SpeechCommandsPCMError(
                f"cannot inspect dataset root {dataset_root}: {error}"
            ) from error
        if not stat.S_ISDIR(root_stat.st_mode):
            os.close(root_fd)
            raise SpeechCommandsPCMError("dataset root is not a directory")
        return root_fd


def _decode_pcm_payload(contents: bytes, source: PCMSourceIdentity) -> bytes:
    try:
        with wave.open(io.BytesIO(contents), "rb") as audio:
            channels = audio.getnchannels()
            sample_width = audio.getsampwidth()
            sample_rate = audio.getframerate()
            sample_count = audio.getnframes()
            compression = audio.getcomptype()
            payload = audio.readframes(sample_count + 1)
    except (EOFError, OSError, wave.Error) as error:
        raise SpeechCommandsPCMError(
            f"cannot decode WAV {source.path!r}: {error}"
        ) from error
    if (
        channels != 1
        or sample_width != 2
        or sample_rate != SAMPLE_RATE
        or compression != "NONE"
        or sample_count < 1
    ):
        raise SpeechCommandsPCMError(
            f"unsupported WAV format for {source.path!r}: channels={channels}, "
            f"sample_width={sample_width}, sample_rate={sample_rate}, "
            f"sample_count={sample_count}, compression={compression}"
        )
    if sample_count != source.sample_count:
        raise SpeechCommandsPCMError(
            f"source frame count differs for {source.path!r}: "
            f"expected={source.sample_count}, observed={sample_count}"
        )
    expected_payload_bytes = sample_count * sample_width
    if len(payload) != expected_payload_bytes:
        raise SpeechCommandsPCMError(
            f"truncated WAV payload for {source.path!r}: "
            f"expected={expected_payload_bytes}, read={len(payload)}"
        )
    return payload


def _canonical_parts(path: str) -> tuple[str, str]:
    if not isinstance(path, str):
        raise TypeError("path must be a string")
    try:
        path.encode("utf-8")
    except UnicodeEncodeError as error:
        raise SpeechCommandsPCMError(
            "source path must encode as valid UTF-8"
        ) from error
    parsed = PurePosixPath(path)
    invalid = (
        not path
        or "\\" in path
        or "\0" in path
        or parsed.is_absolute()
        or len(parsed.parts) != 2
        or any(part in {"", ".", ".."} for part in parsed.parts)
        or parsed.as_posix() != path
        or parsed.suffix != ".wav"
    )
    if invalid:
        raise SpeechCommandsPCMError(f"source path is not canonical: {path!r}")
    return parsed.parts[0], parsed.parts[1]


def _validate_sample_count(sample_count: int) -> None:
    if not isinstance(sample_count, int) or isinstance(sample_count, bool):
        raise TypeError("sample_count must be an integer")
    if sample_count < 1:
        raise SpeechCommandsPCMError("sample_count must be positive")


def _validate_sha256(value: str) -> None:
    if not isinstance(value, str):
        raise TypeError("sha256 must be a string")
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise SpeechCommandsPCMError("sha256 must be a lowercase SHA-256 digest")


def _stat_fingerprint(source_stat: os.stat_result) -> tuple[int, ...]:
    return (
        source_stat.st_dev,
        source_stat.st_ino,
        source_stat.st_mode,
        source_stat.st_size,
        source_stat.st_mtime_ns,
        source_stat.st_ctime_ns,
    )
