"""Compute, publish, and verify the Experiment 002 normalization artifact.

The 320-byte artifact is the sole canonical backing state of a verified
normalization capability.  NumPy arrays are decoded afresh on every ``stats``
access so callers never receive a mutable view of capability-owned state.
"""

from __future__ import annotations

import errno
import hashlib
import hmac
import itertools
import os
import stat
import threading
import weakref
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

import numpy as np

from falsewake.experiment_002_data import (
    WINDOW_SAMPLES,
    Experiment002Corpus,
    Experiment002DataError,
    _require_registered_training_corpus,
)
from falsewake.experiment_002_frontend import complete_log_mel_frames
from falsewake.experiment_002_normalization import (
    NormalizationAccumulator,
    NormalizationStats,
    load_stats,
    serialize_stats,
)
from falsewake.experiment_002_pcm_cache import PCMCacheSplitView
from falsewake.features import FloatArray

REGISTERED_CLIP_COUNT: Final = 84_843
REGISTERED_FRAME_COUNT: Final = 8_314_614
NORMALIZATION_ARTIFACT_BYTES: Final = 320
FRAMES_PER_CLIP: Final = 98
MEL_BINS: Final = 40

_TEMP_COUNTER = itertools.count()
_TEMP_COUNTER_LOCK = threading.Lock()
_OPEN_SUPPORTS_DIR_FD: Final = os.open in os.supports_dir_fd
_LINK_SUPPORTS_DIR_FD: Final = os.link in os.supports_dir_fd
_STAT_SUPPORTS_DIR_FD: Final = os.stat in os.supports_dir_fd
_UNLINK_SUPPORTS_DIR_FD: Final = os.unlink in os.supports_dir_fd


class Experiment002NormalizationArtifactError(ValueError):
    """A computation, capability, or artifact violated the fixed protocol."""


@dataclass(frozen=True, slots=True)
class NormalizationArtifactIdentity:
    """The external whole-file identity of one normalization artifact."""

    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        _require_positive_integer("byte_count", self.byte_count)
        if self.byte_count > 0xFFFF_FFFF_FFFF_FFFF:
            raise Experiment002NormalizationArtifactError("byte_count exceeds uint64")
        _validate_sha256(self.sha256)


@dataclass(frozen=True, slots=True)
class _NormalizationLayout:
    """Clip/frame population contract used by the registered and tiny paths."""

    clip_count: int
    frames_per_clip: int

    def __post_init__(self) -> None:
        _require_positive_integer("clip_count", self.clip_count)
        _require_positive_integer("frames_per_clip", self.frames_per_clip)
        if self.frame_count > 0xFFFF_FFFF_FFFF_FFFF:
            raise Experiment002NormalizationArtifactError(
                "normalization frame count exceeds uint64"
            )

    @classmethod
    def for_corpus(
        cls,
        corpus: Experiment002Corpus,
        *,
        frames_per_clip: int = FRAMES_PER_CLIP,
    ) -> _NormalizationLayout:
        if not isinstance(corpus, Experiment002Corpus):
            raise TypeError("corpus must be an Experiment002Corpus")
        return cls(
            clip_count=len(corpus.train_commands),
            frames_per_clip=frames_per_clip,
        )

    @property
    def frame_count(self) -> int:
        return self.clip_count * self.frames_per_clip


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
class VerifiedNormalization:
    """Issuer-only normalization capability backed by canonical immutable bytes."""

    _clip_count: int
    _contents: bytes
    _frame_count: int
    _identity: NormalizationArtifactIdentity

    def __init__(self) -> None:
        raise TypeError("VerifiedNormalization values are issued by this module")

    @property
    def contents(self) -> bytes:
        """Return the immutable canonical little-endian artifact bytes."""

        return _issued_state(self).contents

    @property
    def identity(self) -> NormalizationArtifactIdentity:
        """Return the verified external identity of ``contents``."""

        return _snapshot_identity(_issued_state(self).identity)

    @property
    def clip_count(self) -> int:
        """Return the exact number of clips used by this capability."""

        return _issued_state(self).clip_count

    @property
    def frame_count(self) -> int:
        """Return the exact number of frames used by this capability."""

        return _issued_state(self).frame_count

    @property
    def stats(self) -> NormalizationStats:
        """Decode fresh owned, immutable arrays from the canonical bytes."""

        return load_stats(_issued_state(self).contents, mel_bins=MEL_BINS)


@dataclass(frozen=True, slots=True)
class _IssuedNormalizationState:
    contents: bytes
    identity: NormalizationArtifactIdentity
    clip_count: int
    frame_count: int


_ISSUED_NORMALIZATIONS: weakref.WeakKeyDictionary[
    VerifiedNormalization, _IssuedNormalizationState
] = weakref.WeakKeyDictionary()
_ISSUED_NORMALIZATIONS_LOCK = threading.Lock()


class _FeatureExtractor(Protocol):
    def __call__(self, waveform: FloatArray) -> FloatArray: ...


def compute_registered_normalization(
    corpus: Experiment002Corpus,
    training: PCMCacheSplitView,
) -> VerifiedNormalization:
    """Compute the registered scalar-order statistics once over every train clip."""

    try:
        _require_registered_training_corpus(corpus)
    except Experiment002DataError as error:
        raise Experiment002NormalizationArtifactError(
            f"corpus does not match the registered training population: {error}"
        ) from error
    return _compute_normalization(
        corpus,
        training,
        layout=_REGISTERED_LAYOUT,
    )


def _compute_normalization(
    corpus: Experiment002Corpus,
    training: PCMCacheSplitView,
    *,
    layout: _NormalizationLayout,
    feature_extractor: _FeatureExtractor | None = None,
) -> VerifiedNormalization:
    """Private tiny-population hook with the production numeric and cache path."""

    _require_layout_matches_corpus(corpus, layout)
    if type(training) is not PCMCacheSplitView:
        raise TypeError("training must be a PCMCacheSplitView")
    if training.split != "train":
        raise Experiment002NormalizationArtifactError(
            "normalization requires the concrete training cache view"
        )
    extractor = (
        _authoritative_log_mel if feature_extractor is None else feature_extractor
    )
    if not callable(extractor):
        raise TypeError("feature_extractor must be callable")

    accumulator = NormalizationAccumulator(mel_bins=MEL_BINS)
    observed_clips = 0
    commands = sorted(corpus.train_commands, key=lambda source: source.manifest_index)
    for source in commands:
        payload = training.read_command_pcm16le(source)
        waveform = _decode_and_pad_command(
            payload,
            sample_count=source.sample_count,
        )
        features = extractor(waveform)
        _validate_feature_matrix(features, frames_per_clip=layout.frames_per_clip)
        accumulator.update(features)
        observed_clips += 1

    if observed_clips != layout.clip_count:
        raise Experiment002NormalizationArtifactError(
            "normalization clip count differs from the layout"
        )
    if accumulator.frame_count != layout.frame_count:
        raise Experiment002NormalizationArtifactError(
            "normalization frame count differs from the layout"
        )
    stats = accumulator.finalize(expected_frames=layout.frame_count)
    contents = serialize_stats(stats)
    identity = _identity_for_contents(contents)
    return _issue_verified_normalization(contents, identity, layout)


def publish_normalization_artifact(
    result: VerifiedNormalization,
    destination: Path,
) -> NormalizationArtifactIdentity:
    """Atomically publish a registered capability without replacing any name."""

    return _publish_normalization_artifact(
        result,
        destination,
        layout=_REGISTERED_LAYOUT,
    )


def verify_registered_normalization(result: VerifiedNormalization) -> None:
    """Reject values not issued for the exact registered population."""

    _revalidate_capability(result, _REGISTERED_LAYOUT)


def _publish_normalization_artifact(
    result: VerifiedNormalization,
    destination: Path,
    *,
    layout: _NormalizationLayout,
) -> NormalizationArtifactIdentity:
    """Private layout hook retaining the production publication semantics."""

    state = _validated_capability_state(result, layout)
    if not isinstance(destination, Path):
        raise TypeError("destination must be a Path")
    parent, destination_leaf = _parent_and_leaf(destination)
    _require_secure_filesystem_support()
    parent_fd = _open_parent(parent)
    temporary_leaf: str | None = None
    temporary_fd = -1
    try:
        _require_destination_absent(parent_fd, destination_leaf)
        temporary_leaf, temporary_fd = _create_temporary(parent_fd)
        try:
            _write_all_at(temporary_fd, state.contents, 0)
            os.ftruncate(temporary_fd, NORMALIZATION_ARTIFACT_BYTES)
            _checked_regular_size(temporary_fd)
            os.fsync(temporary_fd)
            os.fchmod(temporary_fd, 0o444)
            os.fsync(temporary_fd)

            before = os.fstat(temporary_fd)
            observed = _read_exact_at(
                temporary_fd,
                NORMALIZATION_ARTIFACT_BYTES,
                0,
            )
            if os.pread(temporary_fd, 1, NORMALIZATION_ARTIFACT_BYTES):
                raise Experiment002NormalizationArtifactError(
                    "normalization artifact has trailing bytes"
                )
            after = os.fstat(temporary_fd)
            if _stat_fingerprint(before) != _stat_fingerprint(after):
                raise Experiment002NormalizationArtifactError(
                    "normalization temporary changed while being verified"
                )
            if not hmac.compare_digest(observed, state.contents):
                raise Experiment002NormalizationArtifactError(
                    "normalization temporary contents differ"
                )
            if not hmac.compare_digest(
                hashlib.sha256(observed).hexdigest(),
                state.identity.sha256,
            ):
                raise Experiment002NormalizationArtifactError(
                    "normalization temporary digest differs"
                )

            os.link(
                temporary_leaf,
                destination_leaf,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
            _require_published_inode(parent_fd, destination_leaf, after)
            os.unlink(temporary_leaf, dir_fd=parent_fd)
            temporary_leaf = None
            os.fsync(parent_fd)
            return _snapshot_identity(state.identity)
        except FileExistsError as error:
            raise Experiment002NormalizationArtifactError(
                "normalization destination already exists"
            ) from error
        except Experiment002NormalizationArtifactError:
            raise
        except OSError as error:
            raise Experiment002NormalizationArtifactError(
                f"cannot publish normalization artifact: {error}"
            ) from error
    finally:
        if temporary_fd >= 0:
            os.close(temporary_fd)
        if temporary_leaf is not None:
            with suppress(FileNotFoundError):
                os.unlink(temporary_leaf, dir_fd=parent_fd)
        os.close(parent_fd)


def load_normalization_artifact(
    path: Path,
    expected: NormalizationArtifactIdentity,
) -> VerifiedNormalization:
    """Load the registered artifact only under its external whole-file identity."""

    return _load_normalization_artifact(
        path,
        expected,
        layout=_REGISTERED_LAYOUT,
    )


def _load_normalization_artifact(
    path: Path,
    expected: NormalizationArtifactIdentity,
    *,
    layout: _NormalizationLayout,
) -> VerifiedNormalization:
    """Private layout hook that retains verified tiny-population counts."""

    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    if type(expected) is not NormalizationArtifactIdentity:
        raise TypeError("expected must be a NormalizationArtifactIdentity")
    if not isinstance(layout, _NormalizationLayout):
        raise TypeError("layout must be a _NormalizationLayout")
    expected_snapshot = _snapshot_identity(expected)
    if expected_snapshot.byte_count != NORMALIZATION_ARTIFACT_BYTES:
        raise Experiment002NormalizationArtifactError(
            "external normalization byte count differs"
        )
    _require_secure_filesystem_support()
    descriptor = _open_artifact(path)
    try:
        before = _checked_regular_size(descriptor)
        contents = _read_exact_at(descriptor, NORMALIZATION_ARTIFACT_BYTES, 0)
        if os.pread(descriptor, 1, NORMALIZATION_ARTIFACT_BYTES):
            raise Experiment002NormalizationArtifactError(
                "normalization artifact has trailing bytes"
            )
        observed_sha256 = hashlib.sha256(contents).hexdigest()
        if not hmac.compare_digest(observed_sha256, expected_snapshot.sha256):
            raise Experiment002NormalizationArtifactError(
                "normalization artifact digest differs from external identity: "
                f"observed={observed_sha256}"
            )
        after = os.fstat(descriptor)
        if _stat_fingerprint(before) != _stat_fingerprint(after):
            raise Experiment002NormalizationArtifactError(
                "normalization artifact changed while being verified"
            )
        return _issue_verified_normalization(contents, expected_snapshot, layout)
    except OSError as error:
        raise Experiment002NormalizationArtifactError(
            f"cannot load normalization artifact: {error}"
        ) from error
    finally:
        os.close(descriptor)


def _issue_verified_normalization(
    contents: bytes,
    identity: NormalizationArtifactIdentity,
    layout: _NormalizationLayout,
) -> VerifiedNormalization:
    if type(contents) is not bytes:
        raise TypeError("contents must be bytes")
    if type(identity) is not NormalizationArtifactIdentity:
        raise TypeError("identity must be a NormalizationArtifactIdentity")
    if not isinstance(layout, _NormalizationLayout):
        raise TypeError("layout must be a _NormalizationLayout")
    identity_snapshot = _snapshot_identity(identity)
    _validate_canonical_contents(contents, identity_snapshot)

    result = object.__new__(VerifiedNormalization)
    object.__setattr__(result, "_contents", contents)
    object.__setattr__(result, "_identity", _snapshot_identity(identity_snapshot))
    object.__setattr__(result, "_clip_count", layout.clip_count)
    object.__setattr__(result, "_frame_count", layout.frame_count)
    state = _IssuedNormalizationState(
        contents=contents,
        identity=_snapshot_identity(identity_snapshot),
        clip_count=layout.clip_count,
        frame_count=layout.frame_count,
    )
    with _ISSUED_NORMALIZATIONS_LOCK:
        _ISSUED_NORMALIZATIONS[result] = state
    return result


def _validate_canonical_contents(
    contents: bytes,
    identity: NormalizationArtifactIdentity,
) -> None:
    if type(contents) is not bytes:
        raise TypeError("contents must be bytes")
    if type(identity) is not NormalizationArtifactIdentity:
        raise TypeError("identity must be a NormalizationArtifactIdentity")
    if len(contents) != NORMALIZATION_ARTIFACT_BYTES:
        raise Experiment002NormalizationArtifactError(
            f"normalization contents must contain {NORMALIZATION_ARTIFACT_BYTES} bytes"
        )
    if identity.byte_count != len(contents):
        raise Experiment002NormalizationArtifactError(
            "normalization contents differ from their byte-count identity"
        )
    observed_sha256 = hashlib.sha256(contents).hexdigest()
    if not hmac.compare_digest(observed_sha256, identity.sha256):
        raise Experiment002NormalizationArtifactError(
            "normalization contents differ from their digest identity"
        )
    try:
        stats = load_stats(contents, mel_bins=MEL_BINS)
    except (TypeError, ValueError) as error:
        raise Experiment002NormalizationArtifactError(
            f"normalization contents are invalid: {error}"
        ) from error
    if serialize_stats(stats) != contents:
        raise Experiment002NormalizationArtifactError(
            "normalization contents are not canonical"
        )


def _revalidate_capability(
    result: VerifiedNormalization,
    layout: _NormalizationLayout,
) -> None:
    _validated_capability_state(result, layout)


def _validated_capability_state(
    result: VerifiedNormalization,
    layout: _NormalizationLayout,
) -> _IssuedNormalizationState:
    if type(result) is not VerifiedNormalization:
        raise TypeError("result must be a VerifiedNormalization")
    if not isinstance(layout, _NormalizationLayout):
        raise TypeError("layout must be a _NormalizationLayout")
    with _ISSUED_NORMALIZATIONS_LOCK:
        state = _ISSUED_NORMALIZATIONS.get(result)
    if state is None:
        raise Experiment002NormalizationArtifactError(
            "normalization capability was not issued by this process"
        )
    try:
        raw_contents = result._contents
        raw_identity = result._identity
        raw_clip_count = result._clip_count
        raw_frame_count = result._frame_count
    except AttributeError as error:
        raise Experiment002NormalizationArtifactError(
            "normalization capability is incomplete"
        ) from error
    raw_identity_matches = (
        type(raw_identity) is NormalizationArtifactIdentity
        and raw_identity.byte_count == state.identity.byte_count
        and hmac.compare_digest(raw_identity.sha256, state.identity.sha256)
    )
    if (
        type(raw_contents) is not bytes
        or raw_contents != state.contents
        or not raw_identity_matches
        or type(raw_clip_count) is not int
        or raw_clip_count != state.clip_count
        or type(raw_frame_count) is not int
        or raw_frame_count != state.frame_count
    ):
        raise Experiment002NormalizationArtifactError(
            "normalization capability state changed after issuance"
        )
    if state.clip_count != layout.clip_count or state.frame_count != layout.frame_count:
        raise Experiment002NormalizationArtifactError(
            "normalization capability population counts differ"
        )
    _validate_canonical_contents(state.contents, state.identity)
    return state


def _issued_state(result: VerifiedNormalization) -> _IssuedNormalizationState:
    if type(result) is not VerifiedNormalization:
        raise TypeError("result must be a VerifiedNormalization")
    with _ISSUED_NORMALIZATIONS_LOCK:
        state = _ISSUED_NORMALIZATIONS.get(result)
    if state is None:
        raise Experiment002NormalizationArtifactError(
            "normalization capability was not issued by this process"
        )
    return state


def _identity_for_contents(contents: bytes) -> NormalizationArtifactIdentity:
    if type(contents) is not bytes:
        raise TypeError("contents must be bytes")
    return NormalizationArtifactIdentity(
        byte_count=len(contents),
        sha256=hashlib.sha256(contents).hexdigest(),
    )


def _snapshot_identity(
    identity: NormalizationArtifactIdentity,
) -> NormalizationArtifactIdentity:
    if type(identity) is not NormalizationArtifactIdentity:
        raise TypeError("identity must be a NormalizationArtifactIdentity")
    return NormalizationArtifactIdentity(
        byte_count=identity.byte_count,
        sha256=identity.sha256,
    )


def _authoritative_log_mel(waveform: FloatArray) -> FloatArray:
    features = complete_log_mel_frames(waveform)
    _validate_feature_matrix(features, frames_per_clip=FRAMES_PER_CLIP)
    return np.ascontiguousarray(features, dtype=np.float32)


def _decode_and_pad_command(payload: bytes, *, sample_count: int) -> FloatArray:
    if not isinstance(payload, bytes):
        raise TypeError("PCM payload must be bytes")
    if not isinstance(sample_count, int) or isinstance(sample_count, bool):
        raise TypeError("sample_count must be an integer")
    if not 1 <= sample_count <= WINDOW_SAMPLES:
        raise Experiment002NormalizationArtifactError(
            "command sample count is outside the registered range"
        )
    if len(payload) != 2 * sample_count:
        raise Experiment002NormalizationArtifactError("command PCM byte count differs")
    integers = np.frombuffer(payload, dtype="<i2")
    decoded = np.divide(
        integers.astype(np.float32),
        np.float32(32_768.0),
        dtype=np.float32,
    )
    if decoded.dtype != np.dtype(np.float32) or decoded.shape != (sample_count,):
        raise Experiment002NormalizationArtifactError(
            "decoded command PCM shape or dtype differs"
        )
    if not np.all(np.isfinite(decoded)):
        raise Experiment002NormalizationArtifactError(
            "decoded command PCM is non-finite"
        )
    waveform = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    waveform[:sample_count] = decoded
    return waveform


def _validate_feature_matrix(
    features: FloatArray,
    *,
    frames_per_clip: int,
) -> None:
    if not isinstance(features, np.ndarray):
        raise TypeError("feature extractor must return a NumPy array")
    if features.dtype != np.dtype(np.float32):
        raise Experiment002NormalizationArtifactError(
            "feature extractor must return float32"
        )
    if features.shape != (frames_per_clip, MEL_BINS):
        raise Experiment002NormalizationArtifactError(
            "feature extractor returned an invalid shape"
        )
    if not features.flags.c_contiguous:
        raise Experiment002NormalizationArtifactError(
            "feature extractor must return a C-contiguous matrix"
        )
    if not np.all(np.isfinite(features)):
        raise Experiment002NormalizationArtifactError(
            "feature extractor returned non-finite values"
        )


def _require_layout_matches_corpus(
    corpus: Experiment002Corpus,
    layout: _NormalizationLayout,
) -> None:
    if not isinstance(corpus, Experiment002Corpus):
        raise TypeError("corpus must be an Experiment002Corpus")
    if not isinstance(layout, _NormalizationLayout):
        raise TypeError("layout must be a _NormalizationLayout")
    derived = _NormalizationLayout.for_corpus(
        corpus,
        frames_per_clip=layout.frames_per_clip,
    )
    if derived != layout:
        raise Experiment002NormalizationArtifactError(
            "normalization layout differs from the corpus"
        )


def _open_artifact(path: Path) -> int:
    parent, leaf = _parent_and_leaf(path)
    parent_fd = _open_parent(parent)
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
    try:
        try:
            descriptor = os.open(leaf, flags, dir_fd=parent_fd)
        except OSError as error:
            raise Experiment002NormalizationArtifactError(
                f"cannot securely open normalization artifact {path}: {error}"
            ) from error
    finally:
        os.close(parent_fd)
    try:
        source_stat = os.fstat(descriptor)
        if not stat.S_ISREG(source_stat.st_mode):
            raise Experiment002NormalizationArtifactError(
                "normalization artifact is not a regular file"
            )
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _open_parent(parent: Path) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        descriptor = os.open(parent, flags)
    except (OSError, ValueError, UnicodeError) as error:
        raise Experiment002NormalizationArtifactError(
            f"cannot securely open normalization parent {parent}: {error}"
        ) from error
    try:
        parent_stat = os.fstat(descriptor)
        if not stat.S_ISDIR(parent_stat.st_mode):
            raise Experiment002NormalizationArtifactError(
                "normalization parent is not a directory"
            )
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _parent_and_leaf(path: Path) -> tuple[Path, str]:
    leaf = path.name
    if not leaf or leaf in {".", ".."} or "/" in leaf or "\0" in leaf:
        raise Experiment002NormalizationArtifactError(
            f"normalization path must have one valid leaf: {path}"
        )
    try:
        leaf.encode("utf-8")
    except UnicodeEncodeError as error:
        raise Experiment002NormalizationArtifactError(
            "normalization leaf must encode as valid UTF-8"
        ) from error
    return path.parent, leaf


def _require_destination_absent(parent_fd: int, leaf: str) -> None:
    try:
        os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as error:
        raise Experiment002NormalizationArtifactError(
            f"cannot inspect normalization destination: {error}"
        ) from error
    exists_error = FileExistsError(
        errno.EEXIST,
        "normalization destination already exists",
        leaf,
    )
    raise Experiment002NormalizationArtifactError(
        "normalization destination already exists"
    ) from exists_error


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
        leaf = f".falsewake-normalization-{os.getpid():x}-{counter:x}.tmp"
        try:
            descriptor = os.open(leaf, flags, 0o600, dir_fd=parent_fd)
        except FileExistsError:
            continue
        try:
            source_stat = os.fstat(descriptor)
            if not stat.S_ISREG(source_stat.st_mode):
                raise Experiment002NormalizationArtifactError(
                    "new normalization temporary is not a regular file"
                )
        except BaseException:
            os.close(descriptor)
            with suppress(FileNotFoundError):
                os.unlink(leaf, dir_fd=parent_fd)
            raise
        return leaf, descriptor
    raise Experiment002NormalizationArtifactError(
        "cannot allocate a unique normalization temporary"
    )


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
        raise Experiment002NormalizationArtifactError(
            "published normalization name does not reference the built inode"
        )


def _checked_regular_size(descriptor: int) -> os.stat_result:
    source_stat = os.fstat(descriptor)
    if not stat.S_ISREG(source_stat.st_mode):
        raise Experiment002NormalizationArtifactError(
            "normalization artifact is not a regular file"
        )
    if source_stat.st_size != NORMALIZATION_ARTIFACT_BYTES:
        raise Experiment002NormalizationArtifactError(
            "normalization artifact byte count differs: "
            f"observed={source_stat.st_size}, "
            f"expected={NORMALIZATION_ARTIFACT_BYTES}"
        )
    return source_stat


def _read_exact_at(descriptor: int, byte_count: int, offset: int) -> bytes:
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
            raise Experiment002NormalizationArtifactError(
                f"normalization artifact is truncated at byte {offset + len(result)}"
            )
        result.extend(chunk)
    return bytes(result)


def _write_all_at(descriptor: int, contents: bytes, offset: int) -> None:
    written = 0
    view = memoryview(contents)
    while written < len(view):
        try:
            count = os.pwrite(descriptor, view[written:], offset + written)
        except InterruptedError:
            continue
        if count <= 0:
            raise Experiment002NormalizationArtifactError(
                f"short normalization write at byte {offset + written}"
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
        raise Experiment002NormalizationArtifactError(
            f"secure normalization filesystem operations are unavailable: {detail}"
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


def _validate_sha256(value: str) -> None:
    if type(value) is not str:
        raise TypeError("sha256 must be a string")
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise Experiment002NormalizationArtifactError(
            "sha256 must use 64 lowercase hexadecimal characters"
        )


def _require_positive_integer(name: str, value: int) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise Experiment002NormalizationArtifactError(f"{name} must be positive")


_REGISTERED_LAYOUT: Final = _NormalizationLayout(
    clip_count=REGISTERED_CLIP_COUNT,
    frames_per_clip=FRAMES_PER_CLIP,
)

if _REGISTERED_LAYOUT.frame_count != REGISTERED_FRAME_COUNT:
    raise RuntimeError("registered normalization counts are inconsistent")
