"""Source-bound run-registration authority for Experiment 002.

The canonical registration file is intentionally absent today.  This module is
therefore a closed gate: it can issue an authority only after a later, separately
committed ``configs/experiment-002-run.json`` passes every repository, source,
and frozen-binding check below.  The verifier is deliberately standard-library
only so callers can run it before importing any numerical or registered-data
module.
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib.machinery
import importlib.metadata
import json
import os
import platform
import selectors
import shutil
import signal
import stat
import struct
import subprocess
import sys
import threading
import time
import weakref
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final, NoReturn, Protocol, SupportsIndex, cast

_RUN_CONFIG_PATH: Final = "configs/experiment-002-run.json"
_AUTHORITY_SOURCE_PATH: Final = "src/falsewake/experiment_002_run_authority.py"
_CANONICAL_REPOSITORY_ROOT: Final = Path("/home/ubuntu/gitcode/falsewake")
_CANONICAL_PYTHON_EXECUTABLE: Final = (
    "/home/ubuntu/gitcode/.t/falsewake-venv/bin/python"
)
_TRUSTED_RUNTIME_SYS_PATHS: Final = (
    "/usr/lib/python312.zip",
    "/usr/lib/python3.12",
    "/usr/lib/python3.12/lib-dynload",
    "/home/ubuntu/gitcode/.t/falsewake-venv/lib/python3.12/site-packages",
)
_SOURCE_BUNDLE_DOMAIN: Final = b"falsewake-exp002-source-bundle-v1\0"
_RUNTIME_FINGERPRINT_DOMAIN: Final = b"falsewake-exp002-runtime-v1\0"
_MAX_CONFIG_BYTES: Final = 1 << 20
_MAX_SOURCE_FILE_BYTES: Final = 64 << 20
_MAX_SOURCE_BUNDLE_BYTES: Final = 256 << 20
_MAX_SOURCE_FILES: Final = 4_096
_MAX_FROZEN_FILE_BYTES: Final = 64 << 20
_MAX_FROZEN_BUNDLE_BYTES: Final = 256 << 20
_MAX_TRACKED_FILES: Final = 4_096
_MAX_TRACKED_DIRECTORIES: Final = 4_096
_MAX_TRACKED_FILE_BYTES: Final = 64 << 20
_MAX_TRACKED_BYTES: Final = 256 << 20
_MAX_WORKTREE_ENTRIES: Final = 16_384
_MAX_WORKTREE_DEPTH: Final = 64
_MAX_GIT_OUTPUT_BYTES: Final = 32 << 20
_GIT_TIMEOUT_SECONDS: Final = 30
_CHILD_BUNDLE_MAGIC: Final = b"FW2CHLD1"
_CHILD_BUNDLE_VERSION: Final = 1
_CHILD_BUNDLE_HEADER: Final = struct.Struct("<8sI40s40s64s64sIQQ")
_MAX_CHILD_BUNDLE_BYTES: Final = 256 << 20
_SEALED_CHILD_BUNDLE_FD: Final = 7
_SEALED_CHILD_MEMFD_TARGET: Final = "/memfd:falsewake-exp002-child-bundle (deleted)"
_SEALED_ORIGIN_PREFIX: Final = "falsewake-sealed://experiment-002/"
_PARENT_ORIGIN_KIND: Final = "parent_repository"
_SEALED_CHILD_ORIGIN_KIND: Final = "sealed_child"
_LOWER_HEX_40 = frozenset("0123456789abcdef")
_LOWER_HEX_64 = _LOWER_HEX_40
_ISSUER_MARKER: Final = object()
_AUTHORITY_PROCESS_ID: Final = os.getpid()
_PATH_TYPE: Final = type(Path())


class Experiment002RunAuthorityError(ValueError):
    """The source-bound Experiment 002 registration failed closed."""


class _ForkReinitializableLock(Protocol):
    def _at_fork_reinit(self) -> None: ...


class _SourceDataLoader(Protocol):
    def get_data(self, origin: str) -> bytes: ...


class _GetDataRoute(Protocol):
    def __call__(self, origin: str) -> bytes: ...


class _FinderVerifierRoute(Protocol):
    def __call__(self) -> None: ...


def _capture_loaded_authority_source() -> tuple[
    str,
    Path,
    str,
    object,
    object,
    object,
    object,
    object | None,
    object | None,
    tuple[object, ...],
    object | None,
    tuple[str, ...],
    str,
]:
    spec = globals().get("__spec__")
    module = sys.modules.get(__name__)
    if type(spec) is not importlib.machinery.ModuleSpec or module is None:
        raise Experiment002RunAuthorityError(
            "run authority has no exact import module specification"
        )
    loader = spec.loader
    origin = spec.origin
    spec_name = spec.name
    spec_search_locations = spec.submodule_search_locations
    if loader is None or type(origin) is not str or not origin:
        raise Experiment002RunAuthorityError(
            "run authority import origin or loader is invalid"
        )
    get_data_route = getattr(loader, "get_data", None)
    if not callable(get_data_route):
        raise Experiment002RunAuthorityError(
            "run authority loader has no source-data route"
        )
    if origin.startswith(_SEALED_ORIGIN_PREFIX):
        suffix = origin.removeprefix(_SEALED_ORIGIN_PREFIX)
        try:
            source_digest, relative_path = suffix.split("/", maxsplit=1)
        except ValueError as error:
            raise Experiment002RunAuthorityError(
                "sealed run-authority origin is malformed"
            ) from error
        if type(sys.meta_path) is not list or type(sys.path) is not list:
            raise Experiment002RunAuthorityError(
                "sealed run-authority import surfaces have invalid types"
            )
        meta_path_object = cast(object, sys.meta_path)
        meta_path_frame = tuple(cast(list[object], sys.meta_path))
        sys_path_object = cast(object, sys.path)
        sys_path_frame = tuple(sys.path)
        verifier_route = getattr(loader, "_verify_sealed_import_state", None)
        if (
            len(source_digest) != 64
            or any(character not in _LOWER_HEX_64 for character in source_digest)
            or relative_path != _AUTHORITY_SOURCE_PATH
            or not sys.meta_path
            or cast(object, sys.meta_path[0]) is not cast(object, loader)
            or not callable(verifier_route)
            or any(type(path) is not str for path in sys.path)
            or sys_path_frame != _TRUSTED_RUNTIME_SYS_PATHS
            or getattr(module, "__spec__", None) is not spec
            or getattr(module, "__loader__", None) is not loader
            or type(getattr(module, "__file__", None)) is not str
            or getattr(module, "__file__", None) != origin
            or type(getattr(module, "__package__", None)) is not str
            or getattr(module, "__package__", None) != "falsewake"
            or type(spec_name) is not str
            or spec_name != __name__
            or spec_search_locations is not None
        ):
            raise Experiment002RunAuthorityError(
                "sealed run-authority import identity is invalid"
            )
        try:
            verifier_result = cast(_FinderVerifierRoute, verifier_route)()
            payload = cast(_GetDataRoute, get_data_route)(origin)
            final_verifier_result = cast(_FinderVerifierRoute, verifier_route)()
        except BaseException as error:
            raise Experiment002RunAuthorityError(
                "sealed run-authority source cannot be captured"
            ) from error
        if (
            verifier_result is not None
            or final_verifier_result is not None
            or type(payload) is not bytes
            or not 0 < len(payload) <= _MAX_SOURCE_FILE_BYTES
            or getattr(loader, "get_data", None) is not get_data_route
            or getattr(loader, "_verify_sealed_import_state", None)
            is not verifier_route
            or cast(object, sys.meta_path) is not meta_path_object
            or len(sys.meta_path) != len(meta_path_frame)
            or any(
                observed is not expected
                for observed, expected in zip(
                    cast(list[object], sys.meta_path),
                    meta_path_frame,
                    strict=True,
                )
            )
            or cast(object, sys.path) is not sys_path_object
            or tuple(sys.path) != sys_path_frame
        ):
            raise Experiment002RunAuthorityError(
                "sealed run-authority source payload is invalid"
            )
        executing_file = _CANONICAL_REPOSITORY_ROOT / _AUTHORITY_SOURCE_PATH
        return (
            _SEALED_CHILD_ORIGIN_KIND,
            executing_file,
            origin,
            loader,
            spec,
            module,
            get_data_route,
            verifier_route,
            meta_path_object,
            meta_path_frame,
            sys_path_object,
            sys_path_frame,
            hashlib.sha256(payload).hexdigest(),
        )

    canonical_path = os.fspath(_CANONICAL_REPOSITORY_ROOT / _AUTHORITY_SOURCE_PATH)
    if (
        type(loader) is not importlib.machinery.SourceFileLoader
        or origin != canonical_path
        or globals().get("__file__") != canonical_path
        or loader.name != __name__
        or loader.path != canonical_path
        or getattr(module, "__spec__", None) is not spec
        or getattr(module, "__loader__", None) is not loader
    ):
        raise Experiment002RunAuthorityError(
            "parent run-authority import identity is not canonical"
        )
    try:
        payload = cast(_SourceDataLoader, loader).get_data(origin)
    except BaseException as error:
        raise Experiment002RunAuthorityError(
            "parent run-authority source cannot be captured"
        ) from error
    if type(payload) is not bytes or not 0 < len(payload) <= _MAX_SOURCE_FILE_BYTES:
        raise Experiment002RunAuthorityError(
            "parent run-authority source payload is invalid"
        )
    return (
        _PARENT_ORIGIN_KIND,
        Path(canonical_path),
        origin,
        loader,
        spec,
        module,
        get_data_route,
        None,
        None,
        (),
        None,
        (),
        hashlib.sha256(payload).hexdigest(),
    )


(
    _LOADED_SOURCE_ORIGIN_KIND,
    _EXECUTING_FILE,
    _LOADED_SOURCE_ORIGIN,
    _LOADED_SOURCE_LOADER,
    _LOADED_MODULE_SPEC,
    _LOADED_MODULE,
    _LOADED_SOURCE_GET_DATA_ROUTE,
    _LOADED_FINDER_VERIFY_ROUTE,
    _LOADED_META_PATH_OBJECT,
    _LOADED_META_PATH_FRAME,
    _LOADED_SYS_PATH_OBJECT,
    _LOADED_SYS_PATH_FRAME,
    _LOADED_SOURCE_SHA256,
) = _capture_loaded_authority_source()


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
class VerifiedRunRegistration:
    """Opaque process-local proof of one fully verified committed registration."""

    def __init__(self) -> None:
        raise TypeError("verified run registrations are issuer-only")

    @property
    def head_commit(self) -> str:
        """Return the exact committed HEAD verified at issuance."""

        return _verified_state(self).head_commit

    @property
    def implementation_commit(self) -> str:
        """Return the exact registered implementation commit."""

        return _verified_state(self).implementation_commit

    @property
    def registration_sha256(self) -> str:
        """Return the SHA-256 of the canonical run-registration bytes."""

        return _verified_state(self).registration_sha256

    @property
    def source_bundle_sha256(self) -> str:
        """Return the verified committed-source bundle digest."""

        return _verified_state(self).source_bundle_sha256

    def __copy__(self) -> NoReturn:
        raise TypeError("verified run registrations cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        del memo
        raise TypeError("verified run registrations cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("verified run registrations cannot be serialized")

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        del protocol
        raise TypeError("verified run registrations cannot be serialized")


@dataclass(frozen=True, slots=True)
class _AuthorityOriginBinding:
    kind: str
    parent_process_id: int
    bundle_descriptor: int
    bundle_stat_frame: tuple[int, ...]
    bundle_seals: int
    bundle_proc_target: str
    bundle_offset: int
    source_loader: object | None
    source_finder: object | None
    loader_get_data_route: object | None
    finder_verify_route: object | None
    module_object: object | None
    module_spec: object | None
    module_origin: str
    meta_path: tuple[object, ...]
    runtime_sys_path: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _VerifiedState:
    issuer_marker: object
    repository_root: Path
    head_commit: str
    implementation_commit: str
    registration_sha256: str
    source_bundle_sha256: str
    source_paths: tuple[str, ...]
    registration_bytes: bytes
    source_bundle_payload: bytes
    frozen_blobs: tuple[_FrozenBlob, ...]
    origin_binding: _AuthorityOriginBinding
    process_id: int
    nonce: object


@dataclass(frozen=True, slots=True)
class _RepositorySnapshot:
    repository_root: Path
    head_commit: str
    implementation_commit: str
    registration_sha256: str
    source_bundle_sha256: str
    source_paths: tuple[str, ...]
    registration_bytes: bytes = b""
    source_bundle_payload: bytes = b""
    frozen_blobs: tuple[_FrozenBlob, ...] = ()


@dataclass(frozen=True, slots=True)
class _IssuedGuard:
    issuer_marker: object
    repository_root: Path
    head_commit: str
    implementation_commit: str
    registration_sha256: str
    source_bundle_sha256: str
    source_paths: tuple[str, ...]
    registration_bytes: bytes
    source_bundle_payload: bytes
    frozen_blobs: tuple[_FrozenBlob, ...]
    origin_binding: _AuthorityOriginBinding
    process_id: int
    nonce: object


@dataclass(frozen=True, slots=True)
class _RegistrationDocument:
    implementation_commit: str
    import_roots: tuple[str, ...]
    source_paths: tuple[str, ...]
    source_payload_byte_count: int
    source_bundle_sha256: str
    external_inputs: dict[str, Any]
    runtime: dict[str, Any]
    invocation: dict[str, Any]
    raw_bytes: bytes


@dataclass(frozen=True, slots=True)
class _TreeBlob:
    mode: str
    oid: str
    payload: bytes


@dataclass(frozen=True, slots=True)
class _TreeBlobIdentity:
    mode: str
    oid: str


@dataclass(frozen=True, slots=True)
class _PreflightTrackedBlob:
    path: str
    identity: _TreeBlobIdentity
    byte_count: int


@dataclass(frozen=True, slots=True)
class _FrozenBlob:
    path: str
    payload: bytes


@dataclass(frozen=True, slots=True)
class _ChildBundleFrame:
    head_commit: str
    implementation_commit: str
    registration_sha256: str
    source_bundle_sha256: str
    registration_bytes: bytes
    source_bundle_payload: bytes
    source_blobs: tuple[_FrozenBlob, ...]
    frozen_blobs: tuple[_FrozenBlob, ...]


@dataclass(frozen=True, slots=True)
class _SealedChildSnapshot:
    frame: _ChildBundleFrame
    origin_binding: _AuthorityOriginBinding


@dataclass(frozen=True, slots=True)
class _SealedDescriptorSnapshot:
    payload: bytes
    stat_frame: tuple[int, ...]
    seals: int
    proc_target: str
    offset: int


_ISSUED: weakref.WeakKeyDictionary[VerifiedRunRegistration, _VerifiedState] = (
    weakref.WeakKeyDictionary()
)
_ISSUED_GUARDS: weakref.WeakKeyDictionary[VerifiedRunRegistration, _IssuedGuard] = (
    weakref.WeakKeyDictionary()
)
_FAILED: weakref.WeakSet[VerifiedRunRegistration] = weakref.WeakSet()
_ISSUED_LOCK = threading.RLock()
os.register_at_fork(
    after_in_child=cast(_ForkReinitializableLock, _ISSUED_LOCK)._at_fork_reinit
)
_ISSUANCE_COMPLETE = False


def _require_authority_process() -> None:
    if os.getpid() != _AUTHORITY_PROCESS_ID:
        raise Experiment002RunAuthorityError(
            "run-registration authority cannot be inherited across a process fork"
        )


def verify_and_issue_experiment_002_run_registration() -> VerifiedRunRegistration:
    """Verify the later committed registration and issue an opaque authority.

    No repository, digest, source list, commit, or other trust value is accepted
    as an argument.  The physical repository is derived from this module's own
    verified source location.
    """

    _require_authority_process()
    if _LOADED_SOURCE_ORIGIN_KIND != _PARENT_ORIGIN_KIND:
        raise Experiment002RunAuthorityError(
            "parent run-registration issuer requires its canonical file origin"
        )
    with _ISSUED_LOCK:
        _require_authority_process()
        if _ISSUANCE_COMPLETE:
            raise Experiment002RunAuthorityError(
                "a run registration was already issued in this process"
            )
    root = _repository_root()

    # Deliberately precede Git, runtime, source, and data inspection.  The real
    # repository currently stops here because this canonical file is absent.
    config_path = root / _RUN_CONFIG_PATH
    raw_config = _read_regular_file(config_path, maximum_bytes=_MAX_CONFIG_BYTES)
    document = _parse_registration(raw_config)

    snapshot = _verify_committed_repository(root, document)
    _require_production_activation(root, document)
    with _ISSUED_LOCK:
        _require_authority_process()
        if _ISSUANCE_COMPLETE:
            raise Experiment002RunAuthorityError(
                "a concurrent run-registration issuance already completed"
            )
        final_snapshot = _verify_committed_repository(root, document)
        _require_production_activation(root, document)
        if final_snapshot != snapshot:
            raise Experiment002RunAuthorityError(
                "repository changed across production activation"
            )
        return _record_verified_snapshot_locked(snapshot)


def _verify_and_issue_experiment_002_sealed_child_registration() -> (
    VerifiedRunRegistration
):
    """Verify fixed sealed-child state and mint one process-local capability."""

    _require_authority_process()
    if _LOADED_SOURCE_ORIGIN_KIND != _SEALED_CHILD_ORIGIN_KIND:
        raise Experiment002RunAuthorityError(
            "sealed-child issuer requires its exact virtual source origin"
        )
    with _ISSUED_LOCK:
        _require_authority_process()
        if _ISSUANCE_COMPLETE:
            raise Experiment002RunAuthorityError(
                "a run registration was already issued in this process"
            )
    observed = _verify_sealed_child_local_state()
    with _ISSUED_LOCK:
        _require_authority_process()
        if _ISSUANCE_COMPLETE:
            raise Experiment002RunAuthorityError(
                "a concurrent run-registration issuance already completed"
            )
        final_observed = _verify_sealed_child_local_state()
        if not _sealed_child_snapshots_match(observed, final_observed):
            raise Experiment002RunAuthorityError(
                "sealed-child authority inputs changed across issuance"
            )
        return _record_verified_state_locked(
            _state_from_sealed_child_snapshot(observed)
        )


def verify_verified_run_registration(
    registration: VerifiedRunRegistration,
) -> None:
    """Reverify repository state and reject any forged or stale capability."""

    _require_authority_process()
    reverify_verified_run_registration(registration)


def reverify_verified_run_registration(
    registration: VerifiedRunRegistration,
) -> None:
    """Fully recheck repository state, poisoning the capability on mismatch."""

    _require_authority_process()
    state = _verified_state(registration)
    try:
        if state.origin_binding.kind == _SEALED_CHILD_ORIGIN_KIND:
            observed_child = _verify_sealed_child_local_state()
            with _ISSUED_LOCK:
                _require_authority_process()
                current_state = _verified_state(registration)
                final_child = _verify_sealed_child_local_state()
                if (
                    current_state is not state
                    or not _sealed_child_snapshots_match(observed_child, final_child)
                    or not _sealed_child_snapshot_matches_state(final_child, state)
                ):
                    raise Experiment002RunAuthorityError(
                        "sealed-child registration no longer matches its authority"
                    )
        else:
            raw_config = _read_regular_file(
                state.repository_root / _RUN_CONFIG_PATH,
                maximum_bytes=_MAX_CONFIG_BYTES,
            )
            document = _parse_registration(raw_config)
            observed = _verify_committed_repository(state.repository_root, document)
            _require_registered_runtime_identity(state.repository_root, document)
            with _ISSUED_LOCK:
                _require_authority_process()
                current_state = _verified_state(registration)
                final_observed = _verify_committed_repository(
                    state.repository_root, document
                )
                _require_registered_runtime_identity(state.repository_root, document)
                if (
                    current_state is not state
                    or final_observed != observed
                    or not _snapshot_matches_state(final_observed, state)
                ):
                    raise Experiment002RunAuthorityError(
                        "run registration no longer matches its issued authority"
                    )
    except BaseException:
        with _ISSUED_LOCK:
            _require_authority_process()
            _FAILED.add(registration)
        raise


def _create_sealed_experiment_002_child_bundle_fd(
    registration: VerifiedRunRegistration, /
) -> int:
    """Return an O_RDONLY fd with mode 0400 and byte seals verified at return."""

    _require_authority_process()
    writer: int | None = None
    reader: int | None = None
    try:
        reverify_verified_run_registration(registration)
        state = _verified_state(registration)
        if state.origin_binding.kind != _PARENT_ORIGIN_KIND:
            raise Experiment002RunAuthorityError(
                "sealed child bundles require a parent-origin authority"
            )
        bundle = _child_bundle_bytes_from_state(state)
        frame = _parse_experiment_002_child_bundle(bundle)
        _require_child_bundle_matches_state(frame, state)

        memfd_flags, required_seals = _memfd_requirements()
        try:
            created_writer = os.memfd_create(
                "falsewake-exp002-child-bundle",
                flags=memfd_flags,
            )
        except (AttributeError, OSError) as error:
            raise Experiment002RunAuthorityError(
                "anonymous child-bundle memfd creation failed"
            ) from error
        writer = _require_descriptor_number(created_writer, "child-bundle writer")
        _write_descriptor_exactly(writer, bundle)
        try:
            os.fchmod(writer, 0o400)
            os.fsync(writer)
            fcntl.fcntl(writer, fcntl.F_ADD_SEALS, required_seals)
        except (AttributeError, OSError) as error:
            raise Experiment002RunAuthorityError(
                "child-bundle memfd could not be finalized and sealed"
            ) from error
        writer_stat = _verify_sealed_bundle_descriptor(
            writer,
            expected_bytes=bundle,
            state=state,
            required_seals=required_seals,
            expected_access_mode=os.O_RDWR,
        )

        open_flags = os.O_RDONLY | _required_os_constant("O_CLOEXEC")
        try:
            created_reader = os.open(f"/proc/self/fd/{writer}", open_flags)
        except OSError as error:
            raise Experiment002RunAuthorityError(
                "sealed child bundle could not be reopened read-only"
            ) from error
        reader = _require_descriptor_number(created_reader, "child-bundle reader")
        if reader == writer:
            raise Experiment002RunAuthorityError(
                "child-bundle reader did not receive an independent descriptor"
            )
        reader_stat = _verify_sealed_bundle_descriptor(
            reader,
            expected_bytes=bundle,
            state=state,
            required_seals=required_seals,
            expected_access_mode=os.O_RDONLY,
        )
        if (reader_stat.st_dev, reader_stat.st_ino) != (
            writer_stat.st_dev,
            writer_stat.st_ino,
        ):
            raise Experiment002RunAuthorityError(
                "child-bundle descriptors do not identify the same memfd"
            )
        _require_independent_open_file_descriptions(writer, reader, len(bundle))

        os.close(writer)
        writer = None
        _verify_sealed_bundle_descriptor(
            reader,
            expected_bytes=bundle,
            state=state,
            required_seals=required_seals,
            expected_access_mode=os.O_RDONLY,
        )

        reverify_verified_run_registration(registration)
        if _verified_state(registration) is not state:
            raise Experiment002RunAuthorityError(
                "run-registration authority changed during child-bundle creation"
            )
        _verify_sealed_bundle_descriptor(
            reader,
            expected_bytes=bundle,
            state=state,
            required_seals=required_seals,
            expected_access_mode=os.O_RDONLY,
        )
        result = reader
        reader = None
        return result
    finally:
        for descriptor in (reader, writer):
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    continue


def _verify_synthetic_repository_for_tests(
    repository_root: Path,
) -> _RepositorySnapshot:
    """Exercise repository verification without minting a genuine capability."""

    root = _explicit_test_repository_root(repository_root)
    raw_config = _read_regular_file(
        root / _RUN_CONFIG_PATH,
        maximum_bytes=_MAX_CONFIG_BYTES,
    )
    document = _parse_registration(raw_config)
    return _verify_committed_repository(root, document)


def _issue_controlled_snapshot_for_tests(
    snapshot: _RepositorySnapshot,
) -> VerifiedRunRegistration:
    """Exercise the production capability lifecycle from a controlled snapshot.

    Synthetic repository verification itself never calls this seam.  It exists
    only so lifecycle tests can stress the exact one-shot production recorder.
    """

    _require_authority_process()
    with _ISSUED_LOCK:
        _require_authority_process()
        return _record_verified_snapshot_locked(snapshot)


def _record_verified_snapshot_locked(
    snapshot: _RepositorySnapshot,
) -> VerifiedRunRegistration:
    _require_authority_process()
    return _record_verified_state_locked(_state_from_snapshot(snapshot))


def _record_verified_state_locked(
    verified: _VerifiedState,
) -> VerifiedRunRegistration:
    global _ISSUANCE_COMPLETE

    _require_authority_process()
    if _ISSUANCE_COMPLETE:
        raise Experiment002RunAuthorityError(
            "a run registration was already issued in this process"
        )
    _require_verified_state_frame(verified)
    guard = _guard_from_state(verified)
    capability = object.__new__(VerifiedRunRegistration)
    _ISSUED[capability] = verified
    _ISSUED_GUARDS[capability] = guard
    _ISSUANCE_COMPLETE = True
    return capability


def _verified_state(registration: VerifiedRunRegistration) -> _VerifiedState:
    _require_authority_process()
    if type(registration) is not VerifiedRunRegistration:
        raise TypeError("registration must be an exact VerifiedRunRegistration")
    with _ISSUED_LOCK:
        _require_authority_process()
        state = _ISSUED.get(registration)
        guard = _ISSUED_GUARDS.get(registration)
        failed = registration in _FAILED
        if (
            state is None
            or guard is None
            or failed
            or state.issuer_marker is not _ISSUER_MARKER
            or state.process_id != _AUTHORITY_PROCESS_ID
            or not _guard_matches_state(guard, state)
        ):
            _FAILED.add(registration)
            raise Experiment002RunAuthorityError(
                "run-registration capability was not issued by this verifier"
            )
        return state


def _guard_from_state(state: _VerifiedState) -> _IssuedGuard:
    _require_verified_state_frame(state)
    return _IssuedGuard(
        issuer_marker=state.issuer_marker,
        repository_root=state.repository_root,
        head_commit=state.head_commit,
        implementation_commit=state.implementation_commit,
        registration_sha256=state.registration_sha256,
        source_bundle_sha256=state.source_bundle_sha256,
        source_paths=state.source_paths,
        registration_bytes=state.registration_bytes,
        source_bundle_payload=state.source_bundle_payload,
        frozen_blobs=tuple(
            _FrozenBlob(path=blob.path, payload=blob.payload)
            for blob in state.frozen_blobs
        ),
        origin_binding=_clone_origin_binding(state.origin_binding),
        process_id=state.process_id,
        nonce=state.nonce,
    )


def _parent_origin_binding() -> _AuthorityOriginBinding:
    if _LOADED_SOURCE_ORIGIN_KIND != _PARENT_ORIGIN_KIND:
        raise Experiment002RunAuthorityError(
            "repository authority cannot be minted from a sealed-child origin"
        )
    return _AuthorityOriginBinding(
        kind=_PARENT_ORIGIN_KIND,
        parent_process_id=0,
        bundle_descriptor=-1,
        bundle_stat_frame=(),
        bundle_seals=0,
        bundle_proc_target="",
        bundle_offset=-1,
        source_loader=None,
        source_finder=None,
        loader_get_data_route=None,
        finder_verify_route=None,
        module_object=None,
        module_spec=None,
        module_origin=_LOADED_SOURCE_ORIGIN,
        meta_path=(),
        runtime_sys_path=(),
    )


def _clone_origin_binding(
    binding: _AuthorityOriginBinding,
) -> _AuthorityOriginBinding:
    _require_origin_binding_frame(binding, None)
    return _AuthorityOriginBinding(
        kind=binding.kind,
        parent_process_id=binding.parent_process_id,
        bundle_descriptor=binding.bundle_descriptor,
        bundle_stat_frame=tuple(binding.bundle_stat_frame),
        bundle_seals=binding.bundle_seals,
        bundle_proc_target=binding.bundle_proc_target,
        bundle_offset=binding.bundle_offset,
        source_loader=binding.source_loader,
        source_finder=binding.source_finder,
        loader_get_data_route=binding.loader_get_data_route,
        finder_verify_route=binding.finder_verify_route,
        module_object=binding.module_object,
        module_spec=binding.module_spec,
        module_origin=binding.module_origin,
        meta_path=tuple(binding.meta_path),
        runtime_sys_path=tuple(binding.runtime_sys_path),
    )


def _identity_sequences_match(
    left: tuple[object, ...], right: tuple[object, ...]
) -> bool:
    return len(left) == len(right) and all(
        left_item is right_item
        for left_item, right_item in zip(left, right, strict=True)
    )


def _origin_bindings_match(
    left: _AuthorityOriginBinding,
    right: _AuthorityOriginBinding,
) -> bool:
    if (
        type(left) is not _AuthorityOriginBinding
        or type(right) is not _AuthorityOriginBinding
    ):
        return False
    return (
        left.kind == right.kind
        and left.parent_process_id == right.parent_process_id
        and left.bundle_descriptor == right.bundle_descriptor
        and left.bundle_stat_frame == right.bundle_stat_frame
        and left.bundle_seals == right.bundle_seals
        and left.bundle_proc_target == right.bundle_proc_target
        and left.bundle_offset == right.bundle_offset
        and left.source_loader is right.source_loader
        and left.source_finder is right.source_finder
        and left.loader_get_data_route is right.loader_get_data_route
        and left.finder_verify_route is right.finder_verify_route
        and left.module_object is right.module_object
        and left.module_spec is right.module_spec
        and left.module_origin == right.module_origin
        and _identity_sequences_match(left.meta_path, right.meta_path)
        and left.runtime_sys_path == right.runtime_sys_path
    )


def _require_origin_binding_frame(
    binding: _AuthorityOriginBinding,
    source_bundle_sha256: str | None,
) -> None:
    if type(binding) is not _AuthorityOriginBinding:
        raise Experiment002RunAuthorityError(
            "run-registration origin binding has an invalid type"
        )
    if (
        type(binding.kind) is not str
        or type(binding.parent_process_id) is not int
        or type(binding.bundle_descriptor) is not int
        or type(binding.bundle_stat_frame) is not tuple
        or any(type(value) is not int for value in binding.bundle_stat_frame)
        or type(binding.bundle_seals) is not int
        or type(binding.bundle_proc_target) is not str
        or type(binding.bundle_offset) is not int
        or type(binding.module_origin) is not str
        or type(binding.meta_path) is not tuple
        or type(binding.runtime_sys_path) is not tuple
        or any(type(path) is not str for path in binding.runtime_sys_path)
    ):
        raise Experiment002RunAuthorityError(
            "run-registration origin binding fields have invalid types"
        )
    if binding.kind == _PARENT_ORIGIN_KIND:
        expected_parent = (
            binding.parent_process_id == 0
            and binding.bundle_descriptor == -1
            and binding.bundle_stat_frame == ()
            and binding.bundle_seals == 0
            and binding.bundle_proc_target == ""
            and binding.bundle_offset == -1
            and binding.source_loader is None
            and binding.source_finder is None
            and binding.loader_get_data_route is None
            and binding.finder_verify_route is None
            and binding.module_object is None
            and binding.module_spec is None
            and binding.module_origin == _LOADED_SOURCE_ORIGIN
            and _LOADED_SOURCE_ORIGIN_KIND == _PARENT_ORIGIN_KIND
            and binding.meta_path == ()
            and binding.runtime_sys_path == ()
        )
        if not expected_parent:
            raise Experiment002RunAuthorityError(
                "parent run-registration origin binding changed"
            )
        return
    if binding.kind != _SEALED_CHILD_ORIGIN_KIND:
        raise Experiment002RunAuthorityError("run-registration origin kind is invalid")
    if source_bundle_sha256 is None:
        source_bundle_sha256 = _sealed_origin_source_digest(binding.module_origin)
    expected_origin = _sealed_authority_origin(source_bundle_sha256)
    if (
        binding.parent_process_id < 1
        or binding.parent_process_id == _AUTHORITY_PROCESS_ID
        or binding.parent_process_id != os.getppid()
        or binding.bundle_descriptor != _SEALED_CHILD_BUNDLE_FD
        or len(binding.bundle_stat_frame) != 9
        or binding.bundle_seals != _required_child_bundle_seals()
        or binding.bundle_proc_target != _SEALED_CHILD_MEMFD_TARGET
        or binding.bundle_offset != 0
        or binding.source_loader is None
        or binding.source_finder is not binding.source_loader
        or binding.source_loader is not _LOADED_SOURCE_LOADER
        or binding.loader_get_data_route is not _LOADED_SOURCE_GET_DATA_ROUTE
        or binding.finder_verify_route is not _LOADED_FINDER_VERIFY_ROUTE
        or not callable(binding.loader_get_data_route)
        or not callable(binding.finder_verify_route)
        or binding.module_object is not _LOADED_MODULE
        or binding.module_spec is not _LOADED_MODULE_SPEC
        or binding.module_origin != expected_origin
        or binding.module_origin != _LOADED_SOURCE_ORIGIN
        or not binding.meta_path
        or binding.meta_path[0] is not binding.source_loader
        or not _identity_sequences_match(binding.meta_path, _LOADED_META_PATH_FRAME)
        or cast(object, sys.meta_path) is not _LOADED_META_PATH_OBJECT
        or type(sys.meta_path) is not list
        or not _identity_sequences_match(
            tuple(cast(list[object], sys.meta_path)), binding.meta_path
        )
        or binding.runtime_sys_path != _TRUSTED_RUNTIME_SYS_PATHS
        or binding.runtime_sys_path != _LOADED_SYS_PATH_FRAME
        or cast(object, sys.path) is not _LOADED_SYS_PATH_OBJECT
        or type(sys.path) is not list
        or tuple(sys.path) != binding.runtime_sys_path
    ):
        raise Experiment002RunAuthorityError(
            "sealed-child run-registration origin binding changed"
        )


def _guard_matches_state(guard: _IssuedGuard, state: _VerifiedState) -> bool:
    try:
        _require_verified_state_frame(state)
        _require_guard_frame(guard)
    except (TypeError, Experiment002RunAuthorityError):
        return False
    return (
        guard.issuer_marker is state.issuer_marker
        and guard.repository_root == state.repository_root
        and guard.head_commit == state.head_commit
        and guard.implementation_commit == state.implementation_commit
        and guard.registration_sha256 == state.registration_sha256
        and guard.source_bundle_sha256 == state.source_bundle_sha256
        and guard.source_paths == state.source_paths
        and guard.registration_bytes == state.registration_bytes
        and guard.source_bundle_payload == state.source_bundle_payload
        and guard.frozen_blobs == state.frozen_blobs
        and _origin_bindings_match(guard.origin_binding, state.origin_binding)
        and guard.process_id == state.process_id
        and guard.nonce is state.nonce
    )


def _require_verified_state_frame(state: _VerifiedState) -> None:
    if type(state) is not _VerifiedState:
        raise Experiment002RunAuthorityError(
            "run-registration state has an invalid type"
        )
    _require_authority_frame(
        issuer_marker=state.issuer_marker,
        repository_root=state.repository_root,
        head_commit=state.head_commit,
        implementation_commit=state.implementation_commit,
        registration_sha256=state.registration_sha256,
        source_bundle_sha256=state.source_bundle_sha256,
        source_paths=state.source_paths,
        registration_bytes=state.registration_bytes,
        source_bundle_payload=state.source_bundle_payload,
        frozen_blobs=state.frozen_blobs,
        origin_binding=state.origin_binding,
        process_id=state.process_id,
        nonce=state.nonce,
    )


def _require_guard_frame(guard: _IssuedGuard) -> None:
    if type(guard) is not _IssuedGuard:
        raise Experiment002RunAuthorityError(
            "run-registration guard has an invalid type"
        )
    _require_authority_frame(
        issuer_marker=guard.issuer_marker,
        repository_root=guard.repository_root,
        head_commit=guard.head_commit,
        implementation_commit=guard.implementation_commit,
        registration_sha256=guard.registration_sha256,
        source_bundle_sha256=guard.source_bundle_sha256,
        source_paths=guard.source_paths,
        registration_bytes=guard.registration_bytes,
        source_bundle_payload=guard.source_bundle_payload,
        frozen_blobs=guard.frozen_blobs,
        origin_binding=guard.origin_binding,
        process_id=guard.process_id,
        nonce=guard.nonce,
    )


def _require_authority_frame(
    *,
    issuer_marker: object,
    repository_root: Path,
    head_commit: str,
    implementation_commit: str,
    registration_sha256: str,
    source_bundle_sha256: str,
    source_paths: tuple[str, ...],
    registration_bytes: bytes,
    source_bundle_payload: bytes,
    frozen_blobs: tuple[_FrozenBlob, ...],
    origin_binding: _AuthorityOriginBinding,
    process_id: int,
    nonce: object,
) -> None:
    if (
        issuer_marker is not _ISSUER_MARKER
        or type(repository_root) is not _PATH_TYPE
        or repository_root != _CANONICAL_REPOSITORY_ROOT
    ):
        raise Experiment002RunAuthorityError("run-registration authority frame changed")
    _require_hex(head_commit, length=40, name="authority HEAD commit")
    _require_hex(
        implementation_commit, length=40, name="authority implementation commit"
    )
    _require_hex(registration_sha256, length=64, name="authority registration sha256")
    _require_hex(source_bundle_sha256, length=64, name="authority source bundle sha256")
    if type(source_paths) is not tuple or not source_paths:
        raise Experiment002RunAuthorityError("authority source paths changed")
    _require_path_list(list(source_paths), "authority source paths")
    _require_retained_launch_material(
        implementation_commit=implementation_commit,
        registration_sha256=registration_sha256,
        source_bundle_sha256=source_bundle_sha256,
        source_paths=source_paths,
        registration_bytes=registration_bytes,
        source_bundle_payload=source_bundle_payload,
        frozen_blobs=frozen_blobs,
    )
    _require_origin_binding_frame(origin_binding, source_bundle_sha256)
    if (
        type(process_id) is not int
        or process_id != _AUTHORITY_PROCESS_ID
        or type(nonce) is not object
    ):
        raise Experiment002RunAuthorityError("authority process frame changed")


def _state_from_snapshot(snapshot: _RepositorySnapshot) -> _VerifiedState:
    if type(snapshot) is not _RepositorySnapshot:
        raise Experiment002RunAuthorityError(
            "verified repository snapshot has an invalid type"
        )
    return _VerifiedState(
        issuer_marker=_ISSUER_MARKER,
        repository_root=snapshot.repository_root,
        head_commit=snapshot.head_commit,
        implementation_commit=snapshot.implementation_commit,
        registration_sha256=snapshot.registration_sha256,
        source_bundle_sha256=snapshot.source_bundle_sha256,
        source_paths=snapshot.source_paths,
        registration_bytes=snapshot.registration_bytes,
        source_bundle_payload=snapshot.source_bundle_payload,
        frozen_blobs=tuple(
            _FrozenBlob(path=blob.path, payload=blob.payload)
            for blob in snapshot.frozen_blobs
        ),
        origin_binding=_parent_origin_binding(),
        process_id=_AUTHORITY_PROCESS_ID,
        nonce=object(),
    )


def _snapshot_matches_state(
    snapshot: _RepositorySnapshot, state: _VerifiedState
) -> bool:
    return (
        type(snapshot) is _RepositorySnapshot
        and type(state) is _VerifiedState
        and snapshot.repository_root == state.repository_root
        and snapshot.head_commit == state.head_commit
        and snapshot.implementation_commit == state.implementation_commit
        and snapshot.registration_sha256 == state.registration_sha256
        and snapshot.source_bundle_sha256 == state.source_bundle_sha256
        and snapshot.source_paths == state.source_paths
        and snapshot.registration_bytes == state.registration_bytes
        and snapshot.source_bundle_payload == state.source_bundle_payload
        and snapshot.frozen_blobs == state.frozen_blobs
        and state.origin_binding.kind == _PARENT_ORIGIN_KIND
        and state.process_id == _AUTHORITY_PROCESS_ID
    )


def _verify_committed_repository(
    root: Path,
    document: _RegistrationDocument,
) -> _RepositorySnapshot:
    _require_git_toplevel(root)
    head_commit = _head_commit(root)

    config_blob = _committed_blob(
        root,
        head_commit,
        _RUN_CONFIG_PATH,
        maximum_bytes=_MAX_CONFIG_BYTES,
    )
    if config_blob.mode != "100644":
        raise Experiment002RunAuthorityError(
            "run registration must be a nonexecutable regular HEAD blob"
        )
    if config_blob.payload != document.raw_bytes:
        raise Experiment002RunAuthorityError(
            "run registration differs from its committed HEAD blob"
        )
    _require_worktree_blob(root, _RUN_CONFIG_PATH, config_blob)
    _require_clean_worktree(root, head_commit)

    _require_implementation_ancestor(
        root,
        implementation_commit=document.implementation_commit,
        head_commit=head_commit,
    )
    _require_registration_commit_shape(
        root,
        implementation_commit=document.implementation_commit,
        head_commit=head_commit,
    )
    minimum_source_byte_count = _minimum_blob_frame_byte_count(document.source_paths)
    maximum_frozen_payload_bytes = min(
        _MAX_FROZEN_BUNDLE_BYTES,
        _MAX_CHILD_BUNDLE_BYTES
        - _CHILD_BUNDLE_HEADER.size
        - len(document.raw_bytes)
        - minimum_source_byte_count,
    )
    if maximum_frozen_payload_bytes < 4:
        raise Experiment002RunAuthorityError(
            "child-bundle framing leaves no room for frozen blobs"
        )
    frozen_blobs = _capture_frozen_files(
        root,
        head_commit,
        maximum_payload_bytes=maximum_frozen_payload_bytes,
    )
    frozen_payload_byte_count = len(_frozen_blob_payload(frozen_blobs))
    maximum_source_payload_bytes = min(
        _MAX_SOURCE_BUNDLE_BYTES,
        _MAX_CHILD_BUNDLE_BYTES
        - _CHILD_BUNDLE_HEADER.size
        - len(document.raw_bytes)
        - frozen_payload_byte_count,
    )
    if document.source_payload_byte_count > maximum_source_payload_bytes:
        raise Experiment002RunAuthorityError(
            "registered source payload exceeds the exact child-bundle budget"
        )
    source_payload_limit = min(
        maximum_source_payload_bytes,
        document.source_payload_byte_count,
    )

    observed_paths = _enumerate_import_roots(root, document.import_roots)
    if observed_paths != document.source_paths:
        raise Experiment002RunAuthorityError(
            "registered source paths do not exactly cover the import roots"
        )
    if _AUTHORITY_SOURCE_PATH not in document.source_paths:
        raise Experiment002RunAuthorityError(
            "run authority source is absent from the committed source bundle"
        )
    _require_invocation_source_coverage(root, document)

    source_blobs: list[tuple[str, bytes]] = []
    tree_blobs: dict[str, _TreeBlob] = {}
    framed_source_byte_count = 4
    source_frame_overheads = tuple(
        12 + len(path.encode("utf-8")) for path in document.source_paths
    )
    remaining_frame_overhead = sum(source_frame_overheads)
    for path, frame_overhead in zip(
        document.source_paths, source_frame_overheads, strict=True
    ):
        remaining_frame_overhead -= frame_overhead
        framed_source_byte_count += frame_overhead
        if framed_source_byte_count > source_payload_limit:
            raise Experiment002RunAuthorityError(
                "committed source framing exceeds its retained byte budget"
            )
        maximum_blob_bytes = min(
            _MAX_SOURCE_FILE_BYTES,
            source_payload_limit - framed_source_byte_count - remaining_frame_overhead,
        )
        implementation_blob = _committed_blob(
            root,
            document.implementation_commit,
            path,
            maximum_bytes=maximum_blob_bytes,
        )
        head_identity = _committed_blob_identity(root, head_commit, path)
        if (
            implementation_blob.mode != head_identity.mode
            or implementation_blob.oid != head_identity.oid
        ):
            raise Experiment002RunAuthorityError(
                f"registered source changed after implementation commit ({path})"
            )
        _require_worktree_blob(root, path, implementation_blob)
        if (
            path == _AUTHORITY_SOURCE_PATH
            and hashlib.sha256(implementation_blob.payload).hexdigest()
            != _LOADED_SOURCE_SHA256
        ):
            raise Experiment002RunAuthorityError(
                "executing run-authority source differs from its bound blob"
            )
        source_blobs.append((path, implementation_blob.payload))
        tree_blobs[path] = implementation_blob
        framed_source_byte_count += len(implementation_blob.payload)

    source_bundle_payload = _source_bundle_payload(tuple(source_blobs))
    payload_byte_count = len(source_bundle_payload)
    if payload_byte_count != framed_source_byte_count:
        raise Experiment002RunAuthorityError(
            "committed source framing byte count changed"
        )
    source_digest = hashlib.sha256(
        _SOURCE_BUNDLE_DOMAIN + source_bundle_payload
    ).hexdigest()
    if payload_byte_count != document.source_payload_byte_count:
        raise Experiment002RunAuthorityError(
            "committed source bundle byte count differs from registration"
        )
    if source_digest != document.source_bundle_sha256:
        raise Experiment002RunAuthorityError(
            "committed source bundle digest differs from registration"
        )

    # Close ordinary HEAD/worktree races before issuance.  All object reads use
    # the captured full commit ID, and every bound working file is checked again.
    if _head_commit(root) != head_commit:
        raise Experiment002RunAuthorityError(
            "repository HEAD changed during verification"
        )
    _require_worktree_blob(root, _RUN_CONFIG_PATH, config_blob)
    if (
        _capture_frozen_files(
            root,
            head_commit,
            maximum_payload_bytes=maximum_frozen_payload_bytes,
        )
        != frozen_blobs
    ):
        raise Experiment002RunAuthorityError(
            "frozen bindings changed during verification"
        )
    for path, blob in tree_blobs.items():
        _require_worktree_blob(root, path, blob)
    if _enumerate_import_roots(root, document.import_roots) != document.source_paths:
        raise Experiment002RunAuthorityError(
            "registered import roots changed during verification"
        )
    # This is the closing raw-object/worktree closure: it reloads and hashes every
    # HEAD blob, then byte-compares every tracked file through O_NOFOLLOW fds.
    _require_clean_worktree(root, head_commit)
    if _head_commit(root) != head_commit:
        raise Experiment002RunAuthorityError(
            "repository HEAD changed during closing raw verification"
        )
    _require_git_topology(root)
    if _head_commit(root) != head_commit:
        raise Experiment002RunAuthorityError(
            "repository HEAD changed during final topology verification"
        )

    return _RepositorySnapshot(
        repository_root=root,
        head_commit=head_commit,
        implementation_commit=document.implementation_commit,
        registration_sha256=hashlib.sha256(document.raw_bytes).hexdigest(),
        source_bundle_sha256=source_digest,
        source_paths=document.source_paths,
        registration_bytes=document.raw_bytes,
        source_bundle_payload=source_bundle_payload,
        frozen_blobs=frozen_blobs,
    )


def _parse_registration(raw: bytes) -> _RegistrationDocument:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise Experiment002RunAuthorityError(
            "run registration is not strict UTF-8"
        ) from error
    if text.startswith("\ufeff"):
        raise Experiment002RunAuthorityError("run registration has a UTF-8 BOM")
    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_number,
            parse_float=_reject_json_number,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise Experiment002RunAuthorityError(
            "run registration is not exact JSON"
        ) from error
    if type(parsed) is not dict:
        raise Experiment002RunAuthorityError("run registration must be a JSON object")
    document = cast(dict[str, Any], parsed)
    canonical = (
        json.dumps(
            document,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    if raw != canonical:
        raise Experiment002RunAuthorityError(
            "run registration is not canonical indented ASCII JSON plus LF"
        )
    _require_exact_keys(
        document,
        {
            "experiment",
            "external_inputs",
            "frozen_bindings",
            "implementation_commit",
            "invocation",
            "runtime",
            "schema_version",
            "source_bundle",
        },
        "run registration",
    )
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise Experiment002RunAuthorityError("run schema_version must be integer 1")
    if type(document["experiment"]) is not str or document["experiment"] != "002":
        raise Experiment002RunAuthorityError("run experiment must be exact string 002")
    implementation_commit = _require_hex(
        document["implementation_commit"],
        length=40,
        name="implementation_commit",
    )
    expected_bindings = _expected_frozen_bindings()
    if not _exact_json_equal(document["frozen_bindings"], expected_bindings):
        raise Experiment002RunAuthorityError(
            "run registration does not contain the exact frozen bindings"
        )

    source = document["source_bundle"]
    if type(source) is not dict:
        raise Experiment002RunAuthorityError("source_bundle must be an object")
    source_object = cast(dict[str, Any], source)
    _require_exact_keys(
        source_object,
        {"import_roots", "paths", "payload_byte_count", "sha256"},
        "source_bundle",
    )
    import_roots = _require_path_list(source_object["import_roots"], "import_roots")
    source_paths = _require_path_list(source_object["paths"], "source paths")
    if not import_roots:
        raise Experiment002RunAuthorityError("at least one import root is required")
    if len(import_roots) > _MAX_SOURCE_FILES:
        raise Experiment002RunAuthorityError("too many registered import roots")
    if not source_paths:
        raise Experiment002RunAuthorityError("at least one source path is required")
    if len(source_paths) > _MAX_SOURCE_FILES:
        raise Experiment002RunAuthorityError("too many registered source paths")
    seen_import_roots: set[str] = set()
    for import_root in import_roots:
        parts = PurePosixPath(import_root).parts
        if any(
            "/".join(parts[:depth]) in seen_import_roots
            for depth in range(1, len(parts))
        ):
            raise Experiment002RunAuthorityError(
                "registered import roots must not overlap"
            )
        seen_import_roots.add(import_root)
    byte_count = source_object["payload_byte_count"]
    if type(byte_count) is not int or not 0 <= byte_count <= _MAX_SOURCE_BUNDLE_BYTES:
        raise Experiment002RunAuthorityError(
            "source bundle payload_byte_count is invalid"
        )
    minimum_source_byte_count = _minimum_blob_frame_byte_count(source_paths)
    if byte_count < minimum_source_byte_count:
        raise Experiment002RunAuthorityError(
            "source bundle payload_byte_count is below its minimum framing"
        )
    frozen_paths = tuple(
        path
        for path, _ in sorted(
            _frozen_file_bindings(), key=lambda item: item[0].encode("utf-8")
        )
    )
    minimum_frozen_byte_count = _minimum_blob_frame_byte_count(frozen_paths)
    _require_child_bundle_total_byte_count(
        len(raw),
        byte_count,
        minimum_frozen_byte_count,
    )
    source_sha256 = _require_hex(
        source_object["sha256"], length=64, name="source bundle sha256"
    )
    external_inputs = _require_external_inputs(document["external_inputs"])
    runtime = _require_runtime_registration(document["runtime"])
    invocation = _require_invocation_registration(document["invocation"], runtime)
    return _RegistrationDocument(
        implementation_commit=implementation_commit,
        import_roots=import_roots,
        source_paths=source_paths,
        source_payload_byte_count=byte_count,
        source_bundle_sha256=source_sha256,
        external_inputs=external_inputs,
        runtime=runtime,
        invocation=invocation,
        raw_bytes=raw,
    )


def _require_external_inputs(value: object) -> dict[str, Any]:
    if type(value) is not dict:
        raise Experiment002RunAuthorityError("external_inputs must be an object")
    inputs = cast(dict[str, Any], value)
    _require_exact_keys(inputs, {"archive", "manifest", "pcm_cache"}, "external_inputs")
    archive = _require_exact_object(
        inputs["archive"], {"path", "sha256"}, "external archive"
    )
    manifest = _require_exact_object(
        inputs["manifest"],
        {"inventory_sha256", "path", "record_count", "sha256"},
        "external manifest",
    )
    pcm_cache = _require_exact_object(
        inputs["pcm_cache"],
        {"byte_count", "path", "sha256"},
        "external PCM cache",
    )
    archive_path = _require_external_path(archive["path"], "archive path")
    manifest_path = _require_external_path(manifest["path"], "manifest path")
    pcm_path = _require_external_path(pcm_cache["path"], "PCM cache path")
    if len({archive_path, manifest_path, pcm_path}) != 3:
        raise Experiment002RunAuthorityError("external input paths must be distinct")
    expected = _expected_frozen_bindings()["inputs"]
    expected_inputs = cast(dict[str, Any], expected)
    _require_exact_scalar(
        archive["sha256"], expected_inputs["archive_sha256"], "archive sha256"
    )
    _require_exact_scalar(
        manifest["sha256"], expected_inputs["manifest_sha256"], "manifest sha256"
    )
    _require_exact_scalar(
        manifest["inventory_sha256"],
        expected_inputs["manifest_inventory_sha256"],
        "manifest inventory sha256",
    )
    _require_exact_scalar(
        manifest["record_count"],
        expected_inputs["manifest_record_count"],
        "manifest record count",
    )
    artifacts = cast(dict[str, Any], _expected_frozen_bindings()["artifacts"])
    expected_pcm = cast(dict[str, Any], artifacts["pcm_cache"])
    _require_exact_scalar(
        pcm_cache["sha256"], expected_pcm["sha256"], "PCM cache sha256"
    )
    _require_exact_scalar(
        pcm_cache["byte_count"],
        expected_pcm["byte_count"],
        "PCM cache byte count",
    )
    return inputs


def _require_runtime_registration(value: object) -> dict[str, Any]:
    runtime = _require_exact_object(
        value,
        {
            "fingerprint_sha256",
            "numpy_version",
            "python_executable",
            "python_sys_path_roots",
            "python_version",
            "safetensors_version",
            "torch_version",
        },
        "runtime",
    )
    executable = _require_absolute_ascii_path(
        runtime["python_executable"], "Python executable"
    )
    if executable != _CANONICAL_PYTHON_EXECUTABLE:
        raise Experiment002RunAuthorityError(
            "Python executable is not the frozen Experiment 002 runtime"
        )
    runtime_sys_path_roots = _require_string_list(
        runtime["python_sys_path_roots"],
        "runtime sys.path roots",
        absolute_paths=True,
    )
    if runtime_sys_path_roots != _TRUSTED_RUNTIME_SYS_PATHS:
        raise Experiment002RunAuthorityError(
            "runtime sys.path roots are not the frozen exact allowlist"
        )
    expected_versions = {
        "numpy_version": "2.5.1",
        "python_version": "3.12.3",
        "safetensors_version": "0.8.0",
        "torch_version": "2.13.0+cpu",
    }
    for key, expected in expected_versions.items():
        _require_exact_scalar(runtime[key], expected, key)
    fingerprint_fields = {
        "numpy_version": runtime["numpy_version"],
        "python_executable": executable,
        "python_sys_path_roots": list(runtime_sys_path_roots),
        "python_version": runtime["python_version"],
        "safetensors_version": runtime["safetensors_version"],
        "torch_version": runtime["torch_version"],
    }
    expected_fingerprint = _runtime_fingerprint_sha256(fingerprint_fields)
    _require_exact_scalar(
        runtime["fingerprint_sha256"],
        expected_fingerprint,
        "runtime fingerprint sha256",
    )
    return runtime


def _runtime_fingerprint_sha256(fields: dict[str, Any]) -> str:
    payload = (
        json.dumps(
            fields,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    return hashlib.sha256(_RUNTIME_FINGERPRINT_DOMAIN + payload).hexdigest()


def _require_invocation_registration(
    value: object,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    invocation = _require_exact_object(
        value,
        {
            "argv",
            "authority_process_role",
            "cwd",
            "environment",
            "orig_argv",
            "python_flags",
            "sys_path",
        },
        "invocation",
    )
    _require_exact_scalar(
        invocation["authority_process_role"],
        "parent_bootstrap",
        "authority process role",
    )
    argv = _require_string_list(invocation["argv"], "argv", absolute_paths=False)
    orig_argv = _require_string_list(
        invocation["orig_argv"], "orig_argv", absolute_paths=False
    )
    entrypoint = _require_relative_path(argv[0], "argv entrypoint")
    expected_orig_argv = (
        runtime["python_executable"],
        "-I",
        "-S",
        "-B",
        *argv,
    )
    if orig_argv != expected_orig_argv:
        raise Experiment002RunAuthorityError(
            "orig_argv must exactly bind the frozen interpreter, flags, and argv"
        )
    if entrypoint != orig_argv[4]:
        raise Experiment002RunAuthorityError("invocation entrypoint is ambiguous")
    _require_absolute_ascii_path(invocation["cwd"], "invocation cwd")
    _require_string_list(invocation["sys_path"], "sys_path", absolute_paths=True)
    environment = _require_exact_object(
        invocation["environment"],
        {
            "LANG",
            "LC_ALL",
            "MKL_DYNAMIC",
            "MKL_NUM_THREADS",
            "OMP_DYNAMIC",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "PYTHONDONTWRITEBYTECODE",
            "PYTHONHASHSEED",
            "PYTHONNOUSERSITE",
            "TMPDIR",
            "TZ",
        },
        "invocation environment",
    )
    expected_environment = {
        "LANG": "C",
        "LC_ALL": "C",
        "MKL_DYNAMIC": "FALSE",
        "MKL_NUM_THREADS": "1",
        "OMP_DYNAMIC": "FALSE",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "TZ": "UTC",
    }
    for key, expected in expected_environment.items():
        _require_exact_scalar(environment[key], expected, f"environment {key}")
    _require_external_path(environment["TMPDIR"], "TMPDIR")
    flags = _require_exact_object(
        invocation["python_flags"],
        {
            "dont_write_bytecode",
            "ignore_environment",
            "isolated",
            "no_site",
            "no_user_site",
        },
        "Python flags",
    )
    for key in flags:
        _require_exact_scalar(flags[key], 1, f"Python flag {key}")
    return invocation


def _require_invocation_source_coverage(
    root: Path,
    document: _RegistrationDocument,
) -> None:
    invocation = document.invocation
    argv = cast(list[str], invocation["argv"])
    entrypoint = _require_relative_path(argv[0], "argv entrypoint")
    source_paths = frozenset(document.source_paths)
    if entrypoint not in source_paths or not entrypoint.startswith("src/"):
        raise Experiment002RunAuthorityError(
            "invocation entrypoint is absent from the committed source bundle"
        )
    runtime_roots = cast(list[str], document.runtime["python_sys_path_roots"])
    expected_sys_path = (os.fspath(root / "src"), *runtime_roots)
    if tuple(cast(list[str], invocation["sys_path"])) != expected_sys_path:
        raise Experiment002RunAuthorityError(
            "sys.path is not the exact source-bound runtime allowlist"
        )
    observed_source_paths = _enumerate_import_roots(root, ("src",))
    if observed_source_paths != document.source_paths:
        raise Experiment002RunAuthorityError(
            "root/src exposes source outside the committed source bundle"
        )


def _require_production_activation(root: Path, document: _RegistrationDocument) -> None:
    _require_registered_runtime_identity(root, document)
    forbidden_modules = {"numpy", "safetensors", "site", "torch"}
    if forbidden_modules.intersection(sys.modules):
        raise Experiment002RunAuthorityError(
            "runtime or numerical modules loaded before run authorization"
        )


def _require_registered_runtime_identity(
    root: Path,
    document: _RegistrationDocument,
) -> None:
    runtime = document.runtime
    invocation = document.invocation
    if platform.python_version() != runtime["python_version"]:
        raise Experiment002RunAuthorityError("running Python version is not registered")
    if os.path.abspath(sys.executable) != runtime["python_executable"]:
        raise Experiment002RunAuthorityError(
            "running Python executable is not registered"
        )
    package_versions = {
        "numpy_version": importlib.metadata.version("numpy"),
        "safetensors_version": importlib.metadata.version("safetensors"),
        "torch_version": importlib.metadata.version("torch"),
    }
    for key, observed in package_versions.items():
        if observed != runtime[key]:
            raise Experiment002RunAuthorityError(
                f"running package version is not registered ({key})"
            )
    if Path.cwd().resolve(strict=True) != root or invocation["cwd"] != os.fspath(root):
        raise Experiment002RunAuthorityError("running cwd is not the repository root")
    if (
        list(sys.argv) != invocation["argv"]
        or list(sys.orig_argv) != invocation["orig_argv"]
    ):
        raise Experiment002RunAuthorityError("running argv is not registered")
    if list(sys.path) != invocation["sys_path"]:
        raise Experiment002RunAuthorityError("running sys.path is not registered")
    if dict(os.environ) != invocation["environment"]:
        raise Experiment002RunAuthorityError("running environment is not registered")
    expected_flags = cast(dict[str, int], invocation["python_flags"])
    for key, expected in expected_flags.items():
        if getattr(sys.flags, key) != expected:
            raise Experiment002RunAuthorityError(
                f"running Python flag is not registered ({key})"
            )


def _require_exact_object(
    value: object, expected_keys: set[str], name: str
) -> dict[str, Any]:
    if type(value) is not dict:
        raise Experiment002RunAuthorityError(f"{name} must be an exact object")
    result = cast(dict[str, Any], value)
    _require_exact_keys(result, expected_keys, name)
    return result


def _require_exact_scalar(observed: object, expected: object, name: str) -> None:
    if type(observed) is not type(expected) or observed != expected:
        raise Experiment002RunAuthorityError(f"{name} does not match its frozen value")


def _require_string_list(
    value: object, name: str, *, absolute_paths: bool
) -> tuple[str, ...]:
    if type(value) is not list or not value:
        raise Experiment002RunAuthorityError(f"{name} must be a nonempty exact array")
    result: list[str] = []
    for item in cast(list[object], value):
        if type(item) is not str or not item or "\x00" in item:
            raise Experiment002RunAuthorityError(f"{name} contains an invalid string")
        text = item
        if absolute_paths:
            _require_absolute_ascii_path(text, name)
        result.append(text)
    if len(set(result)) != len(result):
        raise Experiment002RunAuthorityError(f"{name} must not contain duplicates")
    return tuple(result)


def _require_absolute_ascii_path(value: object, name: str) -> str:
    if type(value) is not str:
        raise Experiment002RunAuthorityError(f"{name} must be an exact string")
    path = value
    if (
        not path.startswith("/")
        or path == "/"
        or path.endswith("/")
        or "//" in path
        or "\\" in path
        or "\x00" in path
        or any(ord(character) < 0x21 or ord(character) > 0x7E for character in path)
        or PurePosixPath(path).as_posix() != path
        or any(part in {".", ".."} for part in PurePosixPath(path).parts)
    ):
        raise Experiment002RunAuthorityError(f"{name} is not a canonical absolute path")
    return path


def _require_external_path(value: object, name: str) -> str:
    path = _require_absolute_ascii_path(value, name)
    if not path.startswith("/home/ubuntu/gitcode/.t/"):
        raise Experiment002RunAuthorityError(
            f"{name} must be beneath /home/ubuntu/gitcode/.t"
        )
    return path


def _expected_frozen_bindings() -> dict[str, Any]:
    """Return a fresh exact copy of the already frozen trust bindings."""

    return {
        "artifacts": {
            "normalization": {
                "byte_count": 320,
                "path": "models/experiment-002-normalization.f32",
                "sha256": (
                    "891900d4c36fa8a71ba429384f3f4ff594de7c81347ed846c0f790dc4579268e"
                ),
            },
            "normalization_report": {
                "path": "reports/experiment-002-normalization.json",
                "sha256": (
                    "2b95565fec7dc956a4b1f667638b87e3df64cfaaca3026017942a8057c5187a8"
                ),
            },
            "pcm_cache": {
                "byte_count": 2_995_776_948,
                "sha256": (
                    "b56270a4b99d235c62c162545b139a79b7cf3dc157259a3bbf5678d8a451f653"
                ),
            },
            "pcm_cache_report": {
                "path": "reports/experiment-002-pcm-cache.json",
                "sha256": (
                    "a07b02a78bb6e0d2eef8548ec71a48e717f26c756548665f004b0ca1bf3e52ca"
                ),
            },
        },
        "configs": {
            "numerics": {
                "path": "configs/experiment-002-numerics.json",
                "sha256": (
                    "d11fc50ba8f609551dbf88d7863a5b45cd138c6681da6c10ca4c0a9eaf8f2be9"
                ),
            },
            "phase_1": {
                "path": "configs/experiment-002-training.json",
                "sha256": (
                    "a4a21ff66ef04bf15950853acda0897f24ed01eecf566c795a7330e69593d44a"
                ),
            },
        },
        "implementation_audit": {
            "path": "reports/experiment-002-implementation-audit.json",
            "sha256": (
                "0b8033e7f407550d1c8576d31485e5bbcf825592de0d3b86f42ce98c1fca7116"
            ),
        },
        "inputs": {
            "archive_sha256": (
                "af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58"
            ),
            "manifest_inventory_sha256": (
                "c9596927b6de1aa9bb8174bea25f613ebb7bab9e671304c525961b66e0812c5c"
            ),
            "manifest_record_count": 105_829,
            "manifest_sha256": (
                "d28e6993101bd6bc452033bcb7355b25e3b84ab5c51a1458cc097dc93c60f78b"
            ),
            "train_command_inventory_sha256": (
                "74c0e622b3b30df7600cd452f2d88ed62cabcbca040aba54a89f8fcf819c3546"
            ),
            "validation_command_inventory_sha256": (
                "c5373d67a0bb97bb54ff0bd576ec7b089aa398a273e25115286a677ddb4140ce"
            ),
        },
        "model": {
            "parameter_tensor_count": 53,
            "parameter_value_count": 23_724,
            "path": "src/falsewake/causal_kws.py",
            "sha256": (
                "fa5d86aa218eaad7503b6e66ed9faa18e34e917696ccb98dc5ee4f1b2267c4a6"
            ),
        },
        "trainer": {
            "path": "configs/experiment-002-trainer.json",
            "sha256": (
                "768720d031a1f2f6d0a36466fd9b4fa095a2f1b9a6fbbc382d4af5fc51514b48"
            ),
        },
    }


def _frozen_file_bindings() -> tuple[tuple[str, str], ...]:
    frozen = _expected_frozen_bindings()
    artifacts = cast(dict[str, Any], frozen["artifacts"])
    configs = cast(dict[str, Any], frozen["configs"])
    return (
        (
            cast(dict[str, str], frozen["trainer"])["path"],
            cast(dict[str, str], frozen["trainer"])["sha256"],
        ),
        (
            cast(dict[str, str], frozen["implementation_audit"])["path"],
            cast(dict[str, str], frozen["implementation_audit"])["sha256"],
        ),
        (
            cast(dict[str, str], configs["numerics"])["path"],
            cast(dict[str, str], configs["numerics"])["sha256"],
        ),
        (
            cast(dict[str, str], configs["phase_1"])["path"],
            cast(dict[str, str], configs["phase_1"])["sha256"],
        ),
        (
            cast(dict[str, Any], frozen["model"])["path"],
            cast(dict[str, Any], frozen["model"])["sha256"],
        ),
        (
            cast(dict[str, Any], artifacts["normalization"])["path"],
            cast(dict[str, Any], artifacts["normalization"])["sha256"],
        ),
        (
            cast(dict[str, str], artifacts["normalization_report"])["path"],
            cast(dict[str, str], artifacts["normalization_report"])["sha256"],
        ),
        (
            cast(dict[str, str], artifacts["pcm_cache_report"])["path"],
            cast(dict[str, str], artifacts["pcm_cache_report"])["sha256"],
        ),
    )


def _capture_frozen_files(
    root: Path,
    head_commit: str,
    *,
    maximum_payload_bytes: int,
) -> tuple[_FrozenBlob, ...]:
    bindings = tuple(
        sorted(_frozen_file_bindings(), key=lambda item: item[0].encode("utf-8"))
    )
    if len({path for path, _ in bindings}) != len(bindings):
        raise Experiment002RunAuthorityError("frozen file bindings are not unique")
    binding_paths = tuple(path for path, _ in bindings)
    minimum_byte_count = _minimum_blob_frame_byte_count(binding_paths)
    if (
        type(maximum_payload_bytes) is not int
        or maximum_payload_bytes < minimum_byte_count
        or maximum_payload_bytes > _MAX_FROZEN_BUNDLE_BYTES
    ):
        raise Experiment002RunAuthorityError(
            "frozen payload retained byte budget is invalid"
        )
    result: list[_FrozenBlob] = []
    frame_overheads = tuple(12 + len(path.encode("utf-8")) for path in binding_paths)
    remaining_frame_overhead = sum(frame_overheads)
    framed_byte_count = 4
    for (path, expected_sha256), frame_overhead in zip(
        bindings, frame_overheads, strict=True
    ):
        remaining_frame_overhead -= frame_overhead
        framed_byte_count += frame_overhead
        maximum_blob_bytes = min(
            _MAX_FROZEN_FILE_BYTES,
            maximum_payload_bytes - framed_byte_count - remaining_frame_overhead,
        )
        blob = _committed_blob(
            root,
            head_commit,
            path,
            maximum_bytes=maximum_blob_bytes,
        )
        framed_byte_count += len(blob.payload)
        observed_sha256 = hashlib.sha256(blob.payload).hexdigest()
        if observed_sha256 != expected_sha256:
            raise Experiment002RunAuthorityError(
                f"frozen committed file digest mismatch ({path})"
            )
        if (
            path == "models/experiment-002-normalization.f32"
            and len(blob.payload) != 320
        ):
            raise Experiment002RunAuthorityError(
                "normalization artifact byte count mismatch"
            )
        _require_worktree_blob(root, path, blob)
        result.append(_FrozenBlob(path=path, payload=blob.payload))
    return tuple(result)


def _require_frozen_files(root: Path, head_commit: str) -> None:
    _capture_frozen_files(
        root,
        head_commit,
        maximum_payload_bytes=_MAX_FROZEN_BUNDLE_BYTES,
    )


def _source_bundle_digest(
    sources: tuple[tuple[str, bytes], ...],
) -> tuple[int, str]:
    payload = _source_bundle_payload(sources)
    digest = hashlib.sha256(_SOURCE_BUNDLE_DOMAIN + payload).hexdigest()
    return len(payload), digest


def _minimum_blob_frame_byte_count(paths: tuple[str, ...]) -> int:
    if type(paths) is not tuple or not paths:
        raise Experiment002RunAuthorityError("blob-frame paths are invalid")
    total_byte_count = 4
    previous_path_bytes: bytes | None = None
    for path in paths:
        _require_relative_path(path, "blob-frame path")
        path_bytes = path.encode("utf-8")
        if previous_path_bytes is not None and path_bytes <= previous_path_bytes:
            raise Experiment002RunAuthorityError(
                "blob-frame paths are duplicated or not UTF-8 sorted"
            )
        previous_path_bytes = path_bytes
        total_byte_count += 12 + len(path_bytes)
        if total_byte_count > _MAX_CHILD_BUNDLE_BYTES:
            raise Experiment002RunAuthorityError(
                "minimum blob framing exceeds the child-bundle limit"
            )
    return total_byte_count


def _source_bundle_payload(
    sources: tuple[tuple[str, bytes], ...],
) -> bytes:
    if not sources or len(sources) > _MAX_SOURCE_FILES:
        raise Experiment002RunAuthorityError("source bundle file count is invalid")
    paths = tuple(path for path, _ in sources)
    if paths != tuple(sorted(paths, key=lambda value: value.encode("utf-8"))):
        raise Experiment002RunAuthorityError("source bundle paths are not UTF-8 sorted")
    if len(set(paths)) != len(paths):
        raise Experiment002RunAuthorityError("source bundle paths are not unique")
    payload = bytearray(struct.pack("<I", len(sources)))
    for path, blob in sources:
        _require_relative_path(path, "source path")
        if type(blob) is not bytes or len(blob) > _MAX_SOURCE_FILE_BYTES:
            raise Experiment002RunAuthorityError("source blob byte count is invalid")
        path_bytes = path.encode("utf-8")
        payload.extend(struct.pack("<I", len(path_bytes)))
        payload.extend(path_bytes)
        payload.extend(struct.pack("<Q", len(blob)))
        payload.extend(blob)
        if len(payload) > _MAX_SOURCE_BUNDLE_BYTES:
            raise Experiment002RunAuthorityError("source bundle is too large")
    return bytes(payload)


def _parse_blob_frames(
    payload: bytes,
    *,
    name: str,
    maximum_files: int,
    maximum_file_bytes: int,
    maximum_payload_bytes: int,
) -> tuple[_FrozenBlob, ...]:
    if type(payload) is not bytes:
        raise Experiment002RunAuthorityError(f"{name} payload type is invalid")
    if len(payload) < 4 or len(payload) > maximum_payload_bytes:
        raise Experiment002RunAuthorityError(f"{name} payload byte count is invalid")
    file_count = struct.unpack_from("<I", payload, 0)[0]
    if file_count < 1 or file_count > maximum_files:
        raise Experiment002RunAuthorityError(f"{name} file count is invalid")
    offset = 4
    previous_path_bytes: bytes | None = None
    result: list[_FrozenBlob] = []
    for _ in range(file_count):
        if len(payload) - offset < 4:
            raise Experiment002RunAuthorityError(f"{name} path frame is truncated")
        path_byte_count = struct.unpack_from("<I", payload, offset)[0]
        offset += 4
        if (
            path_byte_count < 1
            or path_byte_count > 4_096
            or path_byte_count > len(payload) - offset
        ):
            raise Experiment002RunAuthorityError(f"{name} path byte count is invalid")
        path_bytes = payload[offset : offset + path_byte_count]
        offset += path_byte_count
        try:
            path = path_bytes.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise Experiment002RunAuthorityError(
                f"{name} path is not strict UTF-8"
            ) from error
        _require_relative_path(path, f"{name} path")
        if previous_path_bytes is not None and path_bytes <= previous_path_bytes:
            raise Experiment002RunAuthorityError(
                f"{name} paths are duplicated or not UTF-8 sorted"
            )
        previous_path_bytes = path_bytes
        if len(payload) - offset < 8:
            raise Experiment002RunAuthorityError(f"{name} blob frame is truncated")
        blob_byte_count = struct.unpack_from("<Q", payload, offset)[0]
        offset += 8
        if (
            blob_byte_count > maximum_file_bytes
            or blob_byte_count > len(payload) - offset
        ):
            raise Experiment002RunAuthorityError(f"{name} blob byte count is invalid")
        blob = payload[offset : offset + blob_byte_count]
        offset += blob_byte_count
        result.append(_FrozenBlob(path=path, payload=blob))
    if offset != len(payload):
        raise Experiment002RunAuthorityError(f"{name} payload has trailing bytes")
    return tuple(result)


def _parse_source_bundle_payload(payload: bytes) -> tuple[_FrozenBlob, ...]:
    return _parse_blob_frames(
        payload,
        name="source bundle",
        maximum_files=_MAX_SOURCE_FILES,
        maximum_file_bytes=_MAX_SOURCE_FILE_BYTES,
        maximum_payload_bytes=_MAX_SOURCE_BUNDLE_BYTES,
    )


def _require_frozen_blob_sequence(blobs: tuple[_FrozenBlob, ...]) -> None:
    if type(blobs) is not tuple:
        raise Experiment002RunAuthorityError("frozen blob sequence type is invalid")
    expected_bindings = tuple(
        sorted(_frozen_file_bindings(), key=lambda item: item[0].encode("utf-8"))
    )
    if len(blobs) != len(expected_bindings):
        raise Experiment002RunAuthorityError("frozen blob file count is invalid")
    expected_paths = tuple(path for path, _ in expected_bindings)
    observed_paths: list[str] = []
    total_bytes = 4
    for blob, (expected_path, expected_sha256) in zip(
        blobs, expected_bindings, strict=True
    ):
        if type(blob) is not _FrozenBlob:
            raise Experiment002RunAuthorityError("frozen blob frame type is invalid")
        if type(blob.path) is not str or type(blob.payload) is not bytes:
            raise Experiment002RunAuthorityError("frozen blob frame changed")
        _require_relative_path(blob.path, "frozen blob path")
        observed_paths.append(blob.path)
        if blob.path != expected_path:
            raise Experiment002RunAuthorityError(
                "frozen blob paths are not the exact UTF-8-sorted bindings"
            )
        if len(blob.payload) > _MAX_FROZEN_FILE_BYTES:
            raise Experiment002RunAuthorityError("frozen blob is too large")
        if hashlib.sha256(blob.payload).hexdigest() != expected_sha256:
            raise Experiment002RunAuthorityError(
                f"frozen blob digest mismatch ({blob.path})"
            )
        if (
            blob.path == "models/experiment-002-normalization.f32"
            and len(blob.payload) != 320
        ):
            raise Experiment002RunAuthorityError(
                "normalization artifact byte count mismatch"
            )
        total_bytes += 12 + len(blob.path.encode("utf-8")) + len(blob.payload)
        if total_bytes > _MAX_FROZEN_BUNDLE_BYTES:
            raise Experiment002RunAuthorityError("frozen blob payload is too large")
    if tuple(observed_paths) != expected_paths:
        raise Experiment002RunAuthorityError("frozen blob sequence changed")


def _frozen_blob_payload(blobs: tuple[_FrozenBlob, ...]) -> bytes:
    _require_frozen_blob_sequence(blobs)
    payload = bytearray(struct.pack("<I", len(blobs)))
    for blob in blobs:
        path_bytes = blob.path.encode("utf-8")
        payload.extend(struct.pack("<I", len(path_bytes)))
        payload.extend(path_bytes)
        payload.extend(struct.pack("<Q", len(blob.payload)))
        payload.extend(blob.payload)
    if len(payload) > _MAX_FROZEN_BUNDLE_BYTES:
        raise Experiment002RunAuthorityError("frozen blob payload is too large")
    return bytes(payload)


def _parse_frozen_blob_payload(payload: bytes) -> tuple[_FrozenBlob, ...]:
    blobs = _parse_blob_frames(
        payload,
        name="frozen bundle",
        maximum_files=len(_frozen_file_bindings()),
        maximum_file_bytes=_MAX_FROZEN_FILE_BYTES,
        maximum_payload_bytes=_MAX_FROZEN_BUNDLE_BYTES,
    )
    _require_frozen_blob_sequence(blobs)
    return blobs


def _require_retained_launch_material(
    *,
    implementation_commit: str,
    registration_sha256: str,
    source_bundle_sha256: str,
    source_paths: tuple[str, ...],
    registration_bytes: bytes,
    source_bundle_payload: bytes,
    frozen_blobs: tuple[_FrozenBlob, ...],
) -> None:
    if (
        type(registration_bytes) is not bytes
        or type(source_bundle_payload) is not bytes
        or type(frozen_blobs) is not tuple
    ):
        raise Experiment002RunAuthorityError(
            "retained child-launch material has an invalid type"
        )
    present = (
        bool(registration_bytes),
        bool(source_bundle_payload),
        bool(frozen_blobs),
    )
    if not any(present):
        return
    if not all(present):
        raise Experiment002RunAuthorityError(
            "retained child-launch material is incomplete"
        )
    if len(registration_bytes) > _MAX_CONFIG_BYTES:
        raise Experiment002RunAuthorityError("retained registration is too large")
    if hashlib.sha256(registration_bytes).hexdigest() != registration_sha256:
        raise Experiment002RunAuthorityError("retained registration digest changed")
    document = _parse_registration(registration_bytes)
    if document.implementation_commit != implementation_commit:
        raise Experiment002RunAuthorityError(
            "retained registration implementation changed"
        )
    if document.source_paths != source_paths:
        raise Experiment002RunAuthorityError("retained registration paths changed")
    if document.source_payload_byte_count != len(source_bundle_payload):
        raise Experiment002RunAuthorityError(
            "retained source payload byte count changed"
        )
    observed_source_sha256 = hashlib.sha256(
        _SOURCE_BUNDLE_DOMAIN + source_bundle_payload
    ).hexdigest()
    if (
        observed_source_sha256 != source_bundle_sha256
        or document.source_bundle_sha256 != source_bundle_sha256
    ):
        raise Experiment002RunAuthorityError("retained source payload digest changed")
    source_blobs = _parse_source_bundle_payload(source_bundle_payload)
    if tuple(blob.path for blob in source_blobs) != source_paths:
        raise Experiment002RunAuthorityError("retained source payload paths changed")
    _require_frozen_blob_sequence(frozen_blobs)
    frozen_payload_byte_count = len(_frozen_blob_payload(frozen_blobs))
    _require_child_bundle_total_byte_count(
        len(registration_bytes),
        len(source_bundle_payload),
        frozen_payload_byte_count,
    )
    source_by_path = {blob.path: blob.payload for blob in source_blobs}
    for frozen_blob in frozen_blobs:
        source_payload = source_by_path.get(frozen_blob.path)
        if source_payload is not None and source_payload != frozen_blob.payload:
            raise Experiment002RunAuthorityError(
                f"retained source and frozen blobs differ ({frozen_blob.path})"
            )


def _require_child_bundle_total_byte_count(
    registration_byte_count: int,
    source_payload_byte_count: int,
    frozen_payload_byte_count: int,
) -> int:
    counts = (
        registration_byte_count,
        source_payload_byte_count,
        frozen_payload_byte_count,
    )
    if any(type(value) is not int or value < 0 for value in counts):
        raise Experiment002RunAuthorityError(
            "child bundle component byte count is invalid"
        )
    total_byte_count = _CHILD_BUNDLE_HEADER.size + sum(counts)
    if total_byte_count > _MAX_CHILD_BUNDLE_BYTES:
        raise Experiment002RunAuthorityError(
            "child bundle exceeds the exact total byte limit"
        )
    return total_byte_count


def _child_bundle_bytes_from_state(state: _VerifiedState) -> bytes:
    _require_verified_state_frame(state)
    if state.origin_binding.kind != _PARENT_ORIGIN_KIND:
        raise Experiment002RunAuthorityError(
            "a child-origin authority cannot create another child bundle"
        )
    if (
        not state.registration_bytes
        or not state.source_bundle_payload
        or not state.frozen_blobs
    ):
        raise Experiment002RunAuthorityError(
            "verified authority has no retained child-launch material"
        )
    frozen_payload = _frozen_blob_payload(state.frozen_blobs)
    expected_byte_count = _require_child_bundle_total_byte_count(
        len(state.registration_bytes),
        len(state.source_bundle_payload),
        len(frozen_payload),
    )
    header = _CHILD_BUNDLE_HEADER.pack(
        _CHILD_BUNDLE_MAGIC,
        _CHILD_BUNDLE_VERSION,
        state.head_commit.encode("ascii"),
        state.implementation_commit.encode("ascii"),
        state.registration_sha256.encode("ascii"),
        state.source_bundle_sha256.encode("ascii"),
        len(state.registration_bytes),
        len(state.source_bundle_payload),
        len(frozen_payload),
    )
    bundle = b"".join(
        (
            header,
            state.registration_bytes,
            state.source_bundle_payload,
            frozen_payload,
        )
    )
    if len(bundle) != expected_byte_count:
        raise Experiment002RunAuthorityError("child bundle byte count changed")
    return bundle


def _decode_fixed_ascii(value: bytes, name: str) -> str:
    try:
        return value.decode("ascii", errors="strict")
    except UnicodeDecodeError as error:
        raise Experiment002RunAuthorityError(f"{name} is not ASCII") from error


def _sealed_authority_origin(source_bundle_sha256: str) -> str:
    digest = _require_hex(
        source_bundle_sha256,
        length=64,
        name="sealed authority source bundle sha256",
    )
    return f"{_SEALED_ORIGIN_PREFIX}{digest}/{_AUTHORITY_SOURCE_PATH}"


def _sealed_origin_source_digest(origin: str) -> str:
    if type(origin) is not str or not origin.startswith(_SEALED_ORIGIN_PREFIX):
        raise Experiment002RunAuthorityError("sealed authority origin is invalid")
    suffix = origin.removeprefix(_SEALED_ORIGIN_PREFIX)
    try:
        digest, path = suffix.split("/", maxsplit=1)
    except ValueError as error:
        raise Experiment002RunAuthorityError(
            "sealed authority origin is malformed"
        ) from error
    _require_hex(digest, length=64, name="sealed authority origin digest")
    if path != _AUTHORITY_SOURCE_PATH or origin != _sealed_authority_origin(digest):
        raise Experiment002RunAuthorityError(
            "sealed authority origin path is not exact"
        )
    return digest


def _parse_experiment_002_child_bundle(payload: bytes) -> _ChildBundleFrame:
    if type(payload) is not bytes:
        raise Experiment002RunAuthorityError("child bundle must be exact bytes")
    if len(payload) < _CHILD_BUNDLE_HEADER.size:
        raise Experiment002RunAuthorityError("child bundle header is truncated")
    if len(payload) > _MAX_CHILD_BUNDLE_BYTES:
        raise Experiment002RunAuthorityError("child bundle is too large")
    (
        magic,
        version,
        head_bytes,
        implementation_bytes,
        registration_sha256_bytes,
        source_bundle_sha256_bytes,
        registration_byte_count,
        source_payload_byte_count,
        frozen_payload_byte_count,
    ) = _CHILD_BUNDLE_HEADER.unpack_from(payload)
    if magic != _CHILD_BUNDLE_MAGIC or version != _CHILD_BUNDLE_VERSION:
        raise Experiment002RunAuthorityError("child bundle magic or version is invalid")
    if registration_byte_count < 1 or registration_byte_count > _MAX_CONFIG_BYTES:
        raise Experiment002RunAuthorityError(
            "child bundle registration byte count is invalid"
        )
    if (
        source_payload_byte_count < 1
        or source_payload_byte_count > _MAX_SOURCE_BUNDLE_BYTES
    ):
        raise Experiment002RunAuthorityError(
            "child bundle source byte count is invalid"
        )
    if (
        frozen_payload_byte_count < 1
        or frozen_payload_byte_count > _MAX_FROZEN_BUNDLE_BYTES
    ):
        raise Experiment002RunAuthorityError(
            "child bundle frozen byte count is invalid"
        )
    expected_byte_count = _require_child_bundle_total_byte_count(
        registration_byte_count,
        source_payload_byte_count,
        frozen_payload_byte_count,
    )
    if expected_byte_count != len(payload):
        raise Experiment002RunAuthorityError(
            "child bundle is truncated or has trailing bytes"
        )
    head_commit = _require_hex(
        _decode_fixed_ascii(head_bytes, "child bundle HEAD"),
        length=40,
        name="child bundle HEAD",
    )
    implementation_commit = _require_hex(
        _decode_fixed_ascii(implementation_bytes, "child bundle implementation"),
        length=40,
        name="child bundle implementation",
    )
    if head_commit == implementation_commit:
        raise Experiment002RunAuthorityError(
            "child bundle HEAD and implementation commits are identical"
        )
    registration_sha256 = _require_hex(
        _decode_fixed_ascii(
            registration_sha256_bytes, "child bundle registration sha256"
        ),
        length=64,
        name="child bundle registration sha256",
    )
    source_bundle_sha256 = _require_hex(
        _decode_fixed_ascii(source_bundle_sha256_bytes, "child bundle source sha256"),
        length=64,
        name="child bundle source sha256",
    )
    offset = _CHILD_BUNDLE_HEADER.size
    registration_bytes = payload[offset : offset + registration_byte_count]
    offset += registration_byte_count
    source_bundle_payload = payload[offset : offset + source_payload_byte_count]
    offset += source_payload_byte_count
    frozen_payload = payload[offset:]
    if hashlib.sha256(registration_bytes).hexdigest() != registration_sha256:
        raise Experiment002RunAuthorityError(
            "child bundle registration digest mismatch"
        )
    document = _parse_registration(registration_bytes)
    if document.implementation_commit != implementation_commit:
        raise Experiment002RunAuthorityError(
            "child bundle implementation binding mismatch"
        )
    if document.source_payload_byte_count != len(source_bundle_payload):
        raise Experiment002RunAuthorityError(
            "child bundle source byte-count binding mismatch"
        )
    observed_source_sha256 = hashlib.sha256(
        _SOURCE_BUNDLE_DOMAIN + source_bundle_payload
    ).hexdigest()
    if (
        observed_source_sha256 != source_bundle_sha256
        or document.source_bundle_sha256 != source_bundle_sha256
    ):
        raise Experiment002RunAuthorityError(
            "child bundle source digest binding mismatch"
        )
    source_blobs = _parse_source_bundle_payload(source_bundle_payload)
    if tuple(blob.path for blob in source_blobs) != document.source_paths:
        raise Experiment002RunAuthorityError(
            "child bundle source path binding mismatch"
        )
    frozen_blobs = _parse_frozen_blob_payload(frozen_payload)
    source_by_path = {blob.path: blob.payload for blob in source_blobs}
    for frozen_blob in frozen_blobs:
        source_blob = source_by_path.get(frozen_blob.path)
        if source_blob is not None and source_blob != frozen_blob.payload:
            raise Experiment002RunAuthorityError(
                f"child bundle source/frozen mismatch ({frozen_blob.path})"
            )
    return _ChildBundleFrame(
        head_commit=head_commit,
        implementation_commit=implementation_commit,
        registration_sha256=registration_sha256,
        source_bundle_sha256=source_bundle_sha256,
        registration_bytes=registration_bytes,
        source_bundle_payload=source_bundle_payload,
        source_blobs=source_blobs,
        frozen_blobs=frozen_blobs,
    )


def _require_child_bundle_matches_state(
    frame: _ChildBundleFrame, state: _VerifiedState
) -> None:
    if type(frame) is not _ChildBundleFrame or type(state) is not _VerifiedState:
        raise Experiment002RunAuthorityError("child bundle frame type is invalid")
    if (
        frame.head_commit != state.head_commit
        or frame.implementation_commit != state.implementation_commit
        or frame.registration_sha256 != state.registration_sha256
        or frame.source_bundle_sha256 != state.source_bundle_sha256
        or frame.registration_bytes != state.registration_bytes
        or frame.source_bundle_payload != state.source_bundle_payload
        or frame.frozen_blobs != state.frozen_blobs
        or tuple(blob.path for blob in frame.source_blobs) != state.source_paths
    ):
        raise Experiment002RunAuthorityError(
            "child bundle does not exactly match its verified authority"
        )


def _verify_sealed_child_local_state() -> _SealedChildSnapshot:
    if _LOADED_SOURCE_ORIGIN_KIND != _SEALED_CHILD_ORIGIN_KIND:
        raise Experiment002RunAuthorityError(
            "sealed-child verification requires a sealed source origin"
        )
    descriptor = _read_fixed_sealed_child_bundle()
    frame = _parse_experiment_002_child_bundle(descriptor.payload)
    document = _parse_registration(frame.registration_bytes)
    _require_sealed_child_document_bindings(frame, document)
    _require_sealed_child_runtime_identity(document)
    binding = _capture_sealed_child_origin_binding(frame, descriptor)
    _require_origin_binding_frame(binding, frame.source_bundle_sha256)
    return _SealedChildSnapshot(frame=frame, origin_binding=binding)


def _require_sealed_child_document_bindings(
    frame: _ChildBundleFrame,
    document: _RegistrationDocument,
) -> None:
    if (
        type(frame) is not _ChildBundleFrame
        or type(document) is not _RegistrationDocument
    ):
        raise Experiment002RunAuthorityError(
            "sealed-child registration frame has an invalid type"
        )
    _require_retained_launch_material(
        implementation_commit=frame.implementation_commit,
        registration_sha256=frame.registration_sha256,
        source_bundle_sha256=frame.source_bundle_sha256,
        source_paths=document.source_paths,
        registration_bytes=frame.registration_bytes,
        source_bundle_payload=frame.source_bundle_payload,
        frozen_blobs=frame.frozen_blobs,
    )
    invocation = document.invocation
    argv = cast(list[str], invocation["argv"])
    entrypoint = _require_relative_path(argv[0], "sealed-child argv entrypoint")
    expected_parent_sys_path = (
        os.fspath(_CANONICAL_REPOSITORY_ROOT / "src"),
        *_TRUSTED_RUNTIME_SYS_PATHS,
    )
    if (
        frame.implementation_commit != document.implementation_commit
        or frame.source_bundle_sha256 != document.source_bundle_sha256
        or tuple(blob.path for blob in frame.source_blobs) != document.source_paths
        or _AUTHORITY_SOURCE_PATH not in document.source_paths
        or entrypoint not in document.source_paths
        or not entrypoint.startswith("src/")
        or invocation["cwd"] != os.fspath(_CANONICAL_REPOSITORY_ROOT)
        or tuple(cast(list[str], invocation["sys_path"])) != expected_parent_sys_path
    ):
        raise Experiment002RunAuthorityError(
            "sealed-child registration bindings are not canonical"
        )


def _require_sealed_child_runtime_identity(
    document: _RegistrationDocument,
) -> None:
    runtime = document.runtime
    invocation = document.invocation
    python_version = platform.python_version()
    if type(python_version) is not str or python_version != runtime["python_version"]:
        raise Experiment002RunAuthorityError(
            "sealed-child Python version is not registered"
        )
    if (
        type(sys.executable) is not str
        or os.path.abspath(sys.executable) != runtime["python_executable"]
    ):
        raise Experiment002RunAuthorityError(
            "sealed-child Python executable is not registered"
        )
    package_versions = {
        "numpy_version": importlib.metadata.version("numpy"),
        "safetensors_version": importlib.metadata.version("safetensors"),
        "torch_version": importlib.metadata.version("torch"),
    }
    for key, observed in package_versions.items():
        if type(observed) is not str or observed != runtime[key]:
            raise Experiment002RunAuthorityError(
                f"sealed-child package version is not registered ({key})"
            )
    try:
        current_directory = Path.cwd().resolve(strict=True)
    except OSError as error:
        raise Experiment002RunAuthorityError(
            "sealed-child cwd cannot be resolved"
        ) from error
    if current_directory != _CANONICAL_REPOSITORY_ROOT or invocation[
        "cwd"
    ] != os.fspath(_CANONICAL_REPOSITORY_ROOT):
        raise Experiment002RunAuthorityError(
            "sealed-child cwd is not the canonical repository root"
        )
    if (
        type(sys.argv) is not list
        or type(sys.orig_argv) is not list
        or any(type(argument) is not str for argument in sys.argv)
        or any(type(argument) is not str for argument in sys.orig_argv)
        or list(sys.argv) != invocation["argv"]
        or list(sys.orig_argv) != invocation["orig_argv"]
    ):
        raise Experiment002RunAuthorityError("sealed-child argv is not registered")
    if (
        type(sys.path) is not list
        or any(type(path) is not str for path in sys.path)
        or tuple(sys.path) != _TRUSTED_RUNTIME_SYS_PATHS
    ):
        raise Experiment002RunAuthorityError(
            "sealed-child sys.path is not the exact runtime-only allowlist"
        )
    observed_environment = dict(os.environ)
    if (
        any(
            type(key) is not str or type(value) is not str
            for key, value in observed_environment.items()
        )
        or observed_environment != invocation["environment"]
    ):
        raise Experiment002RunAuthorityError(
            "sealed-child environment is not registered"
        )
    expected_flags = cast(dict[str, int], invocation["python_flags"])
    for key, expected in expected_flags.items():
        observed_flag = getattr(sys.flags, key)
        if type(observed_flag) is not int or observed_flag != expected:
            raise Experiment002RunAuthorityError(
                f"sealed-child Python flag is not registered ({key})"
            )


def _capture_sealed_child_origin_binding(
    frame: _ChildBundleFrame,
    descriptor: _SealedDescriptorSnapshot,
) -> _AuthorityOriginBinding:
    module = sys.modules.get(__name__)
    spec = getattr(module, "__spec__", None)
    loader = getattr(spec, "loader", None)
    origin = getattr(spec, "origin", None)
    if (
        type(sys.meta_path) is not list
        or not sys.meta_path
        or type(sys.path) is not list
        or any(type(path) is not str for path in sys.path)
    ):
        raise Experiment002RunAuthorityError(
            "sealed-child import surfaces have invalid types"
        )
    finder = cast(object, sys.meta_path[0])
    get_data_route = getattr(loader, "get_data", None)
    verifier_route = getattr(finder, "_verify_sealed_import_state", None)
    spec_name = getattr(spec, "name", None)
    module_package = getattr(module, "__package__", None)
    module_file = getattr(module, "__file__", None)
    expected_origin = _sealed_authority_origin(frame.source_bundle_sha256)
    if (
        module is not _LOADED_MODULE
        or spec is not _LOADED_MODULE_SPEC
        or type(spec) is not importlib.machinery.ModuleSpec
        or loader is not _LOADED_SOURCE_LOADER
        or finder is not loader
        or get_data_route is not _LOADED_SOURCE_GET_DATA_ROUTE
        or verifier_route is not _LOADED_FINDER_VERIFY_ROUTE
        or not callable(get_data_route)
        or not callable(verifier_route)
        or type(origin) is not str
        or origin != expected_origin
        or origin != _LOADED_SOURCE_ORIGIN
        or type(spec_name) is not str
        or spec_name != __name__
        or spec.submodule_search_locations is not None
        or getattr(module, "__loader__", None) is not loader
        or getattr(module, "__spec__", None) is not spec
        or type(module_package) is not str
        or module_package != "falsewake"
        or type(module_file) is not str
        or module_file != expected_origin
        or cast(object, sys.meta_path) is not _LOADED_META_PATH_OBJECT
        or not _identity_sequences_match(
            tuple(cast(list[object], sys.meta_path)), _LOADED_META_PATH_FRAME
        )
        or cast(object, sys.path) is not _LOADED_SYS_PATH_OBJECT
        or tuple(sys.path) != _LOADED_SYS_PATH_FRAME
    ):
        raise Experiment002RunAuthorityError(
            "sealed-child module, finder, loader, or origin identity is invalid"
        )
    meta_path = tuple(cast(list[object], sys.meta_path))
    runtime_sys_path = tuple(sys.path)
    try:
        verifier_result = cast(_FinderVerifierRoute, verifier_route)()
        authority_payload = cast(_GetDataRoute, get_data_route)(expected_origin)
        final_verifier_result = cast(_FinderVerifierRoute, verifier_route)()
    except BaseException as error:
        raise Experiment002RunAuthorityError(
            "sealed-child source finder could not reverify itself"
        ) from error
    if verifier_result is not None or final_verifier_result is not None:
        raise Experiment002RunAuthorityError(
            "sealed-child source finder verifier returned a value"
        )
    authority_blobs = tuple(
        blob for blob in frame.source_blobs if blob.path == _AUTHORITY_SOURCE_PATH
    )
    if (
        type(authority_payload) is not bytes
        or len(authority_blobs) != 1
        or authority_blobs[0].payload != authority_payload
        or hashlib.sha256(authority_payload).hexdigest() != _LOADED_SOURCE_SHA256
        or _sealed_origin_source_digest(expected_origin) != frame.source_bundle_sha256
        or type(sys.meta_path) is not list
        or not _identity_sequences_match(
            tuple(cast(list[object], sys.meta_path)), meta_path
        )
        or getattr(loader, "get_data", None) is not get_data_route
        or getattr(finder, "_verify_sealed_import_state", None) is not verifier_route
        or getattr(module, "__spec__", None) is not spec
        or getattr(module, "__loader__", None) is not loader
        or type(getattr(spec, "origin", None)) is not str
        or getattr(spec, "origin", None) != expected_origin
        or tuple(sys.path) != runtime_sys_path
    ):
        raise Experiment002RunAuthorityError(
            "sealed-child source identity changed while being captured"
        )
    parent_process_id = os.getppid()
    if (
        type(parent_process_id) is not int
        or parent_process_id < 1
        or parent_process_id == _AUTHORITY_PROCESS_ID
    ):
        raise Experiment002RunAuthorityError(
            "sealed-child parent process identity is invalid"
        )
    return _AuthorityOriginBinding(
        kind=_SEALED_CHILD_ORIGIN_KIND,
        parent_process_id=parent_process_id,
        bundle_descriptor=_SEALED_CHILD_BUNDLE_FD,
        bundle_stat_frame=descriptor.stat_frame,
        bundle_seals=descriptor.seals,
        bundle_proc_target=descriptor.proc_target,
        bundle_offset=descriptor.offset,
        source_loader=loader,
        source_finder=finder,
        loader_get_data_route=get_data_route,
        finder_verify_route=verifier_route,
        module_object=module,
        module_spec=spec,
        module_origin=expected_origin,
        meta_path=meta_path,
        runtime_sys_path=runtime_sys_path,
    )


def _sealed_child_snapshots_match(
    left: _SealedChildSnapshot,
    right: _SealedChildSnapshot,
) -> bool:
    return (
        type(left) is _SealedChildSnapshot
        and type(right) is _SealedChildSnapshot
        and left.frame == right.frame
        and _origin_bindings_match(left.origin_binding, right.origin_binding)
    )


def _state_from_sealed_child_snapshot(
    snapshot: _SealedChildSnapshot,
) -> _VerifiedState:
    if type(snapshot) is not _SealedChildSnapshot:
        raise Experiment002RunAuthorityError(
            "sealed-child snapshot has an invalid type"
        )
    frame = snapshot.frame
    state = _VerifiedState(
        issuer_marker=_ISSUER_MARKER,
        repository_root=_CANONICAL_REPOSITORY_ROOT,
        head_commit=frame.head_commit,
        implementation_commit=frame.implementation_commit,
        registration_sha256=frame.registration_sha256,
        source_bundle_sha256=frame.source_bundle_sha256,
        source_paths=tuple(blob.path for blob in frame.source_blobs),
        registration_bytes=frame.registration_bytes,
        source_bundle_payload=frame.source_bundle_payload,
        frozen_blobs=tuple(
            _FrozenBlob(path=blob.path, payload=blob.payload)
            for blob in frame.frozen_blobs
        ),
        origin_binding=_clone_origin_binding(snapshot.origin_binding),
        process_id=_AUTHORITY_PROCESS_ID,
        nonce=object(),
    )
    _require_verified_state_frame(state)
    return state


def _sealed_child_snapshot_matches_state(
    snapshot: _SealedChildSnapshot,
    state: _VerifiedState,
) -> bool:
    if type(snapshot) is not _SealedChildSnapshot or type(state) is not _VerifiedState:
        return False
    frame = snapshot.frame
    return (
        state.origin_binding.kind == _SEALED_CHILD_ORIGIN_KIND
        and frame.head_commit == state.head_commit
        and frame.implementation_commit == state.implementation_commit
        and frame.registration_sha256 == state.registration_sha256
        and frame.source_bundle_sha256 == state.source_bundle_sha256
        and tuple(blob.path for blob in frame.source_blobs) == state.source_paths
        and frame.registration_bytes == state.registration_bytes
        and frame.source_bundle_payload == state.source_bundle_payload
        and frame.frozen_blobs == state.frozen_blobs
        and _origin_bindings_match(snapshot.origin_binding, state.origin_binding)
        and state.process_id == _AUTHORITY_PROCESS_ID
    )


def _required_os_constant(name: str) -> int:
    value = getattr(os, name, None)
    if type(value) is not int or value < 0:
        raise Experiment002RunAuthorityError(
            f"required operating-system constant is unavailable ({name})"
        )
    return value


def _required_fcntl_constant(name: str) -> int:
    value = getattr(fcntl, name, None)
    if type(value) is not int or value < 0:
        raise Experiment002RunAuthorityError(
            f"required sealing constant is unavailable ({name})"
        )
    return value


def _memfd_requirements() -> tuple[int, int]:
    if not callable(getattr(os, "memfd_create", None)):
        raise Experiment002RunAuthorityError("memfd_create is unavailable")
    flags = _required_os_constant("MFD_CLOEXEC") | _required_os_constant(
        "MFD_ALLOW_SEALING"
    )
    _required_fcntl_constant("F_ADD_SEALS")
    _required_fcntl_constant("F_GET_SEALS")
    return flags, _required_child_bundle_seals()


def _required_child_bundle_seals() -> int:
    _required_fcntl_constant("F_GET_SEALS")
    return (
        _required_fcntl_constant("F_SEAL_WRITE")
        | _required_fcntl_constant("F_SEAL_GROW")
        | _required_fcntl_constant("F_SEAL_SHRINK")
        | _required_fcntl_constant("F_SEAL_SEAL")
    )


def _require_descriptor_number(descriptor: object, name: str) -> int:
    if type(descriptor) is not int or descriptor < 0:
        raise Experiment002RunAuthorityError(f"{name} is invalid")
    return descriptor


def _write_descriptor_exactly(descriptor: int, payload: bytes) -> None:
    _require_descriptor_number(descriptor, "child-bundle writer")
    if type(payload) is not bytes or not payload:
        raise Experiment002RunAuthorityError("child-bundle write payload is invalid")
    view = memoryview(payload)
    offset = 0
    while offset < len(view):
        try:
            written = os.write(descriptor, view[offset:])
        except InterruptedError:
            continue
        except OSError as error:
            raise Experiment002RunAuthorityError(
                "child-bundle memfd write failed"
            ) from error
        if type(written) is not int or written < 1 or written > len(view) - offset:
            raise Experiment002RunAuthorityError(
                "child-bundle memfd write made invalid progress"
            )
        offset += written
    try:
        observed_offset = os.lseek(descriptor, 0, os.SEEK_CUR)
    except OSError as error:
        raise Experiment002RunAuthorityError(
            "child-bundle writer offset cannot be verified"
        ) from error
    if observed_offset != len(payload):
        raise Experiment002RunAuthorityError(
            "child-bundle writer offset is inconsistent"
        )


def _read_descriptor_exactly(descriptor: int, byte_count: int) -> bytes:
    _require_descriptor_number(descriptor, "child-bundle descriptor")
    if (
        type(byte_count) is not int
        or byte_count < 1
        or byte_count > _MAX_CHILD_BUNDLE_BYTES
    ):
        raise Experiment002RunAuthorityError(
            "child-bundle descriptor byte count is invalid"
        )
    chunks: list[bytes] = []
    offset = 0
    while offset < byte_count:
        try:
            chunk = os.pread(descriptor, min(1 << 20, byte_count - offset), offset)
        except InterruptedError:
            continue
        except OSError as error:
            raise Experiment002RunAuthorityError(
                "child-bundle descriptor read failed"
            ) from error
        if type(chunk) is not bytes or not chunk:
            raise Experiment002RunAuthorityError(
                "child-bundle descriptor is unexpectedly truncated"
            )
        chunks.append(chunk)
        offset += len(chunk)
    try:
        trailing = os.pread(descriptor, 1, byte_count)
    except OSError as error:
        raise Experiment002RunAuthorityError(
            "child-bundle descriptor length cannot be verified"
        ) from error
    if trailing != b"":
        raise Experiment002RunAuthorityError(
            "child-bundle descriptor has trailing bytes"
        )
    return b"".join(chunks)


def _read_fixed_sealed_child_bundle() -> _SealedDescriptorSnapshot:
    descriptor = _SEALED_CHILD_BUNDLE_FD
    required_seals = _required_child_bundle_seals()
    try:
        before = os.fstat(descriptor)
        descriptor_flags = fcntl.fcntl(descriptor, fcntl.F_GETFD)
        status_flags = fcntl.fcntl(descriptor, fcntl.F_GETFL)
        seals = fcntl.fcntl(descriptor, fcntl.F_GET_SEALS)
        inheritable = os.get_inheritable(descriptor)
        proc_target = os.readlink(f"/proc/self/fd/{descriptor}")
        offset = os.lseek(descriptor, 0, os.SEEK_CUR)
    except (AttributeError, OSError, ValueError) as error:
        raise Experiment002RunAuthorityError(
            "fixed sealed-child bundle descriptor cannot be inspected"
        ) from error
    before_frame = _stat_frame(before)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 0
        or before.st_uid != os.geteuid()
        or before.st_gid != os.getegid()
        or stat.S_IMODE(before.st_mode) != 0o400
        or not 0 < before.st_size <= _MAX_CHILD_BUNDLE_BYTES
    ):
        raise Experiment002RunAuthorityError(
            "fixed sealed-child bundle is not an exact anonymous regular file"
        )
    if (
        type(descriptor_flags) is not int
        or descriptor_flags != fcntl.FD_CLOEXEC
        or inheritable is not False
        or type(status_flags) is not int
        or status_flags & os.O_ACCMODE != os.O_RDONLY
        or status_flags & (os.O_APPEND | os.O_NONBLOCK)
        or type(seals) is not int
        or seals != required_seals
        or type(proc_target) is not str
        or proc_target != _SEALED_CHILD_MEMFD_TARGET
        or type(offset) is not int
        or offset != 0
    ):
        raise Experiment002RunAuthorityError(
            "fixed sealed-child bundle descriptor flags or identity are invalid"
        )
    payload = _read_descriptor_exactly(descriptor, before.st_size)
    try:
        after = os.fstat(descriptor)
        final_descriptor_flags = fcntl.fcntl(descriptor, fcntl.F_GETFD)
        final_status_flags = fcntl.fcntl(descriptor, fcntl.F_GETFL)
        final_seals = fcntl.fcntl(descriptor, fcntl.F_GET_SEALS)
        final_inheritable = os.get_inheritable(descriptor)
        final_target = os.readlink(f"/proc/self/fd/{descriptor}")
        final_offset = os.lseek(descriptor, 0, os.SEEK_CUR)
    except (AttributeError, OSError, ValueError) as error:
        raise Experiment002RunAuthorityError(
            "fixed sealed-child bundle descriptor changed while being read"
        ) from error
    if (
        _stat_frame(after) != before_frame
        or final_descriptor_flags != descriptor_flags
        or final_status_flags != status_flags
        or final_seals != seals
        or final_inheritable is not False
        or final_target != proc_target
        or final_offset != offset
        or len(payload) != before.st_size
    ):
        raise Experiment002RunAuthorityError(
            "fixed sealed-child bundle descriptor changed while being read"
        )
    return _SealedDescriptorSnapshot(
        payload=payload,
        stat_frame=before_frame,
        seals=seals,
        proc_target=proc_target,
        offset=offset,
    )


def _verify_sealed_bundle_descriptor(
    descriptor: int,
    *,
    expected_bytes: bytes,
    state: _VerifiedState,
    required_seals: int,
    expected_access_mode: int,
) -> os.stat_result:
    _require_descriptor_number(descriptor, "child-bundle descriptor")
    try:
        before = os.fstat(descriptor)
        descriptor_flags = fcntl.fcntl(descriptor, fcntl.F_GETFD)
        status_flags = fcntl.fcntl(descriptor, fcntl.F_GETFL)
        observed_seals = fcntl.fcntl(descriptor, fcntl.F_GET_SEALS)
        inheritable = os.get_inheritable(descriptor)
    except (AttributeError, OSError) as error:
        raise Experiment002RunAuthorityError(
            "child-bundle descriptor metadata cannot be verified"
        ) from error
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 0
        or stat.S_IMODE(before.st_mode) != 0o400
        or before.st_size != len(expected_bytes)
    ):
        raise Experiment002RunAuthorityError(
            "child-bundle descriptor stat frame is invalid"
        )
    if (
        type(descriptor_flags) is not int
        or descriptor_flags & fcntl.FD_CLOEXEC == 0
        or inheritable is not False
        or type(status_flags) is not int
        or status_flags & os.O_ACCMODE != expected_access_mode
        or type(observed_seals) is not int
        or observed_seals != required_seals
    ):
        raise Experiment002RunAuthorityError(
            "child-bundle descriptor flags or seals are invalid"
        )
    observed_bytes = _read_descriptor_exactly(descriptor, len(expected_bytes))
    if observed_bytes != expected_bytes:
        raise Experiment002RunAuthorityError(
            "child-bundle descriptor bytes do not match the authority"
        )
    frame = _parse_experiment_002_child_bundle(observed_bytes)
    _require_child_bundle_matches_state(frame, state)
    try:
        after = os.fstat(descriptor)
        final_seals = fcntl.fcntl(descriptor, fcntl.F_GET_SEALS)
    except OSError as error:
        raise Experiment002RunAuthorityError(
            "child-bundle descriptor changed while being verified"
        ) from error
    if _stat_frame(after) != _stat_frame(before) or final_seals != required_seals:
        raise Experiment002RunAuthorityError(
            "child-bundle descriptor changed while being verified"
        )
    return after


def _require_independent_open_file_descriptions(
    writer: int, reader: int, bundle_byte_count: int
) -> None:
    try:
        writer_offset = os.lseek(writer, 0, os.SEEK_CUR)
        reader_offset = os.lseek(reader, 0, os.SEEK_CUR)
        if writer_offset != bundle_byte_count or reader_offset != 0:
            raise Experiment002RunAuthorityError(
                "child-bundle descriptor offsets are invalid"
            )
        os.lseek(reader, 1, os.SEEK_SET)
        if os.lseek(writer, 0, os.SEEK_CUR) != writer_offset:
            raise Experiment002RunAuthorityError(
                "child-bundle descriptors share one open file description"
            )
        os.lseek(reader, 0, os.SEEK_SET)
    except OSError as error:
        raise Experiment002RunAuthorityError(
            "child-bundle descriptor independence cannot be verified"
        ) from error


def _enumerate_import_roots(
    root: Path, import_roots: tuple[str, ...]
) -> tuple[str, ...]:
    result: list[str] = []
    visited_entry_count = 0
    for relative_root in import_roots:
        root_path = _validated_worktree_path(root, relative_root)
        try:
            root_stat = root_path.lstat()
        except OSError as error:
            raise Experiment002RunAuthorityError(
                f"registered import root is inaccessible ({relative_root})"
            ) from error
        if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
            raise Experiment002RunAuthorityError(
                "registered import root is not a nonsymlink directory "
                f"({relative_root})"
            )
        pending: list[tuple[Path, str]] = [(root_path, relative_root)]
        while pending:
            directory, relative_directory = pending.pop()
            try:
                entries = os.scandir(directory)
            except OSError as error:
                raise Experiment002RunAuthorityError(
                    "registered import root cannot be enumerated "
                    f"({relative_directory})"
                ) from error
            try:
                with entries:
                    for entry in entries:
                        visited_entry_count += 1
                        if visited_entry_count > _MAX_SOURCE_FILES:
                            raise Experiment002RunAuthorityError(
                                "registered import roots contain too many entries"
                            )
                        child_relative = f"{relative_directory}/{entry.name}"
                        _require_relative_path(child_relative, "import-root entry")
                        child_stat = entry.stat(follow_symlinks=False)
                        if stat.S_ISLNK(child_stat.st_mode):
                            raise Experiment002RunAuthorityError(
                                "registered import root contains a symlink "
                                f"({child_relative})"
                            )
                        if stat.S_ISDIR(child_stat.st_mode):
                            pending.append((Path(entry.path), child_relative))
                        elif stat.S_ISREG(child_stat.st_mode):
                            result.append(child_relative)
                        else:
                            raise Experiment002RunAuthorityError(
                                "registered import root contains a nonregular entry "
                                f"({child_relative})"
                            )
            except OSError as error:
                raise Experiment002RunAuthorityError(
                    "registered import-root entry is inaccessible "
                    f"({relative_directory})"
                ) from error
    return tuple(sorted(result, key=lambda value: value.encode("utf-8")))


def _repository_root() -> Path:
    try:
        source_stat = _EXECUTING_FILE.lstat()
    except OSError as error:
        raise Experiment002RunAuthorityError(
            "executing run-authority source is inaccessible"
        ) from error
    if stat.S_ISLNK(source_stat.st_mode) or not stat.S_ISREG(source_stat.st_mode):
        raise Experiment002RunAuthorityError(
            "executing run-authority source is not a regular nonsymlink file"
        )
    resolved_source = _EXECUTING_FILE.resolve(strict=True)
    if resolved_source != _EXECUTING_FILE:
        raise Experiment002RunAuthorityError(
            "executing run-authority path contains symlink ambiguity"
        )
    if len(_EXECUTING_FILE.parents) < 3:
        raise Experiment002RunAuthorityError(
            "executing run-authority path has no repository root"
        )
    absolute = _EXECUTING_FILE.parents[2]
    try:
        candidate_stat = absolute.lstat()
    except OSError as error:
        raise Experiment002RunAuthorityError(
            "repository root is inaccessible"
        ) from error
    if stat.S_ISLNK(candidate_stat.st_mode) or not stat.S_ISDIR(candidate_stat.st_mode):
        raise Experiment002RunAuthorityError(
            "repository root must be a nonsymlink directory"
        )
    resolved = absolute.resolve(strict=True)
    if resolved != absolute:
        raise Experiment002RunAuthorityError(
            "repository root must not contain symlink ambiguity"
        )
    if resolved != _CANONICAL_REPOSITORY_ROOT:
        raise Experiment002RunAuthorityError(
            "executing authority is outside the canonical Experiment 002 repository"
        )
    try:
        relative_source = _EXECUTING_FILE.relative_to(resolved).as_posix()
    except ValueError as error:
        raise Experiment002RunAuthorityError(
            "executing run-authority source is outside its repository"
        ) from error
    if relative_source != _AUTHORITY_SOURCE_PATH:
        raise Experiment002RunAuthorityError(
            "executing run-authority source path is not canonical"
        )
    return resolved


def _explicit_test_repository_root(repository_root: Path) -> Path:
    if type(repository_root) is not _PATH_TYPE:
        raise TypeError("synthetic repository_root must be an exact pathlib path")
    absolute = repository_root.absolute()
    try:
        root_stat = absolute.lstat()
    except OSError as error:
        raise Experiment002RunAuthorityError(
            "synthetic repository root is inaccessible"
        ) from error
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise Experiment002RunAuthorityError(
            "synthetic repository root must be a nonsymlink directory"
        )
    resolved = absolute.resolve(strict=True)
    if resolved != absolute:
        raise Experiment002RunAuthorityError(
            "synthetic repository root contains symlink ambiguity"
        )
    return resolved


def _validated_worktree_path(root: Path, relative_path: str) -> Path:
    _require_relative_path(relative_path, "repository path")
    current = root
    parts = PurePosixPath(relative_path).parts
    for index, part in enumerate(parts):
        current = current / part
        if index == len(parts) - 1:
            break
        try:
            component = current.lstat()
        except OSError as error:
            raise Experiment002RunAuthorityError(
                f"repository path ancestor is inaccessible ({relative_path})"
            ) from error
        if stat.S_ISLNK(component.st_mode) or not stat.S_ISDIR(component.st_mode):
            raise Experiment002RunAuthorityError(
                "repository path ancestor is not a nonsymlink directory "
                f"({relative_path})"
            )
    return current


def _read_regular_file(path: Path, *, maximum_bytes: int) -> bytes:
    try:
        before = path.lstat()
    except OSError as error:
        raise Experiment002RunAuthorityError(
            f"required regular file is absent or inaccessible ({path})"
        ) from error
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
    ):
        raise Experiment002RunAuthorityError(
            f"required path is not a singly-linked regular nonsymlink file ({path})"
        )
    if before.st_size > maximum_bytes:
        raise Experiment002RunAuthorityError(f"required file is too large ({path})")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise Experiment002RunAuthorityError(
            f"required regular file cannot be opened safely ({path})"
        ) from error
    try:
        opened = os.fstat(descriptor)
        before_frame = _stat_frame(before)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or _stat_frame(opened) != before_frame
        ):
            raise Experiment002RunAuthorityError(
                f"required regular file identity changed ({path})"
            )
        chunks: list[bytes] = []
        byte_count = 0
        while True:
            chunk = os.read(descriptor, min(1 << 20, maximum_bytes + 1 - byte_count))
            if not chunk:
                break
            chunks.append(chunk)
            byte_count += len(chunk)
            if byte_count > maximum_bytes:
                raise Experiment002RunAuthorityError(
                    f"required file exceeds its byte limit ({path})"
                )
        after = os.fstat(descriptor)
        if _stat_frame(after) != before_frame:
            raise Experiment002RunAuthorityError(
                f"required regular file changed while reading ({path})"
            )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _stat_frame(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _require_worktree_blob(root: Path, path: str, blob: _TreeBlob) -> None:
    worktree_path = _validated_worktree_path(root, path)
    maximum_bytes = max(len(blob.payload), 1)
    first = _read_regular_file(worktree_path, maximum_bytes=maximum_bytes)
    second = _read_regular_file(worktree_path, maximum_bytes=maximum_bytes)
    if first != second or first != blob.payload:
        raise Experiment002RunAuthorityError(
            f"worktree file differs from committed HEAD blob ({path})"
        )
    file_mode = worktree_path.stat(follow_symlinks=False).st_mode
    executable = bool(file_mode & 0o111)
    if executable != (blob.mode == "100755"):
        raise Experiment002RunAuthorityError(
            f"worktree executable mode differs from committed HEAD ({path})"
        )


def _require_git_toplevel(root: Path) -> None:
    output = _git(root, "rev-parse", "--show-toplevel")
    try:
        text = output.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise Experiment002RunAuthorityError(
            "Git top-level path is not UTF-8"
        ) from error
    if "\x00" in text or not text.endswith("\n") or text.count("\n") != 1:
        raise Experiment002RunAuthorityError("Git top-level output is ambiguous")
    top_level = Path(text[:-1])
    if not top_level.is_absolute() or top_level != root:
        raise Experiment002RunAuthorityError(
            "repository_root is not the exact Git worktree top level"
        )
    _require_git_topology(root)


def _require_git_topology(root: Path) -> None:
    expected = f"{root / '.git'}\n"
    outputs = (
        (
            _git(root, "rev-parse", "--path-format=absolute", "--git-dir"),
            "Git metadata directory",
        ),
        (
            _git(root, "rev-parse", "--path-format=absolute", "--git-common-dir"),
            "Git common metadata directory",
        ),
    )
    for output, name in outputs:
        if len(output) > 4_097:
            raise Experiment002RunAuthorityError(f"{name} output is too long")
        try:
            observed = output.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise Experiment002RunAuthorityError(f"{name} is not UTF-8") from error
        if observed != expected or "\x00" in observed or observed.count("\n") != 1:
            raise Experiment002RunAuthorityError(
                f"{name} is not the exact top-level .git directory"
            )


def _head_commit(root: Path) -> str:
    output = _git(root, "rev-parse", "--verify", "HEAD^{commit}")
    return _single_git_hex_line(output, length=40, name="HEAD commit")


def _require_clean_worktree(root: Path, head_commit: str) -> None:
    _require_git_topology(root)
    tagged = _git(root, "ls-files", "-v", "-z")
    tagged_records = tagged.split(b"\0")
    if not tagged_records or tagged_records[-1] != b"":
        raise Experiment002RunAuthorityError("Git index flag output is ambiguous")
    if any(not record.startswith(b"H ") for record in tagged_records[:-1]):
        raise Experiment002RunAuthorityError(
            "Git index contains assume-unchanged, skip-worktree, or unusual entries"
        )

    index_entries = _index_entries(root)
    tree_entries = _tree_entries(root, head_commit)
    if index_entries != tree_entries:
        raise Experiment002RunAuthorityError(
            "Git stage-zero index does not exactly equal the HEAD tree"
        )
    _require_raw_tracked_worktree(root, tree_entries)
    _require_git_topology(root)


def _validated_tracked_entry(
    mode: bytes,
    oid: bytes,
    path: bytes,
    *,
    name: str,
) -> tuple[bytes, bytes, bytes]:
    if mode not in {b"100644", b"100755"}:
        raise Experiment002RunAuthorityError(
            f"{name} contains a nonregular or unusual mode"
        )
    try:
        oid_text = oid.decode("ascii", errors="strict")
        path_text = path.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise Experiment002RunAuthorityError(
            f"{name} contains a noncanonical identity or path"
        ) from error
    _require_hex(oid_text, length=40, name=f"{name} blob ID")
    _require_relative_path(path_text, f"{name} path")
    if path_text.encode("utf-8") != path:
        raise Experiment002RunAuthorityError(f"{name} path is not canonical UTF-8")
    return mode, oid, path


def _require_raw_tracked_worktree(
    root: Path,
    tree_entries: tuple[tuple[bytes, bytes, bytes], ...],
) -> None:
    if type(tree_entries) is not tuple or len(tree_entries) > _MAX_TRACKED_FILES:
        raise Experiment002RunAuthorityError("tracked tree entry count is invalid")
    tracked_paths: set[str] = set()
    tracked_directories: set[str] = set()
    decoded_entries: list[tuple[str, _TreeBlobIdentity]] = []
    previous_path_bytes: bytes | None = None
    for mode, oid, path_bytes in tree_entries:
        validated_mode, validated_oid, validated_path = _validated_tracked_entry(
            mode,
            oid,
            path_bytes,
            name="tracked tree",
        )
        if previous_path_bytes is not None and validated_path <= previous_path_bytes:
            raise Experiment002RunAuthorityError(
                "tracked tree paths are duplicated or not byte sorted"
            )
        previous_path_bytes = validated_path
        path = validated_path.decode("utf-8", errors="strict")
        parts = PurePosixPath(path).parts
        if len(parts) - 1 > _MAX_WORKTREE_DEPTH:
            raise Experiment002RunAuthorityError(
                "tracked tree path exceeds the worktree depth limit"
            )
        if path in tracked_paths or path in tracked_directories:
            raise Experiment002RunAuthorityError(
                "tracked tree contains a file/directory path collision"
            )
        for directory_depth in range(1, len(parts)):
            directory = "/".join(parts[:directory_depth])
            if directory in tracked_paths:
                raise Experiment002RunAuthorityError(
                    "tracked tree contains a file/directory path collision"
                )
            tracked_directories.add(directory)
            if len(tracked_directories) > _MAX_TRACKED_DIRECTORIES:
                raise Experiment002RunAuthorityError(
                    "tracked tree contains too many directory prefixes"
                )
        tracked_paths.add(path)
        decoded_entries.append(
            (
                path,
                _TreeBlobIdentity(
                    mode=validated_mode.decode("ascii"),
                    oid=validated_oid.decode("ascii"),
                ),
            )
        )
    if len(tracked_paths) + len(tracked_directories) + 1 > _MAX_WORKTREE_ENTRIES:
        raise Experiment002RunAuthorityError(
            "tracked tree exceeds the raw worktree entry budget"
        )

    preflighted: list[_PreflightTrackedBlob] = []
    cumulative_byte_count = 0
    for path, identity in decoded_entries:
        byte_count = _preflight_committed_blob_size(
            root,
            identity,
            path,
            maximum_bytes=_MAX_TRACKED_FILE_BYTES,
        )
        cumulative_byte_count += byte_count
        if cumulative_byte_count > _MAX_TRACKED_BYTES:
            raise Experiment002RunAuthorityError(
                "tracked tree exceeds its cumulative raw byte budget"
            )
        preflighted.append(
            _PreflightTrackedBlob(
                path=path,
                identity=identity,
                byte_count=byte_count,
            )
        )
    _require_raw_worktree_inventory(
        root,
        tuple(preflighted),
        frozenset(tracked_directories),
    )


def _require_raw_worktree_inventory(
    root: Path,
    tracked_blobs: tuple[_PreflightTrackedBlob, ...],
    tracked_directories: frozenset[str],
) -> None:
    if (
        type(tracked_blobs) is not tuple
        or type(tracked_directories) is not frozenset
        or len(tracked_blobs) > _MAX_TRACKED_FILES
        or len(tracked_directories) > _MAX_TRACKED_DIRECTORIES
    ):
        raise Experiment002RunAuthorityError("raw worktree frame is invalid")
    tracked_by_path: dict[str, _PreflightTrackedBlob] = {}
    reconstructed_directories: set[str] = set()
    cumulative_byte_count = 0
    previous_path_bytes: bytes | None = None
    for tracked in tracked_blobs:
        if type(tracked) is not _PreflightTrackedBlob:
            raise Experiment002RunAuthorityError("raw tracked blob frame is invalid")
        _require_relative_path(tracked.path, "raw tracked blob path")
        path_bytes = tracked.path.encode("utf-8")
        if previous_path_bytes is not None and path_bytes <= previous_path_bytes:
            raise Experiment002RunAuthorityError(
                "raw tracked blob paths are not uniquely byte sorted"
            )
        previous_path_bytes = path_bytes
        if tracked.path in tracked_by_path or tracked.path in tracked_directories:
            raise Experiment002RunAuthorityError(
                "raw tracked blob paths are duplicated or collide"
            )
        if (
            type(tracked.identity) is not _TreeBlobIdentity
            or tracked.identity.mode not in {"100644", "100755"}
            or type(tracked.byte_count) is not int
            or not 0 <= tracked.byte_count <= _MAX_TRACKED_FILE_BYTES
        ):
            raise Experiment002RunAuthorityError("raw tracked blob frame is invalid")
        _require_hex(
            tracked.identity.oid,
            length=40,
            name="raw tracked blob ID",
        )
        cumulative_byte_count += tracked.byte_count
        if cumulative_byte_count > _MAX_TRACKED_BYTES:
            raise Experiment002RunAuthorityError(
                "raw tracked blobs exceed their cumulative byte budget"
            )
        parts = PurePosixPath(tracked.path).parts
        if len(parts) - 1 > _MAX_WORKTREE_DEPTH:
            raise Experiment002RunAuthorityError(
                "raw tracked blob path exceeds the worktree depth limit"
            )
        for directory_depth in range(1, len(parts)):
            reconstructed_directories.add("/".join(parts[:directory_depth]))
            if len(reconstructed_directories) > _MAX_TRACKED_DIRECTORIES:
                raise Experiment002RunAuthorityError(
                    "raw tracked blobs contain too many directory prefixes"
                )
        tracked_by_path[tracked.path] = tracked
    if reconstructed_directories != set(tracked_directories):
        raise Experiment002RunAuthorityError(
            "raw tracked directory frame does not match tracked blob paths"
        )
    if len(tracked_by_path) + len(tracked_directories) + 1 > _MAX_WORKTREE_ENTRIES:
        raise Experiment002RunAuthorityError(
            "raw worktree frame exceeds its entry budget"
        )

    directory_flags = (
        os.O_RDONLY
        | _required_os_constant("O_CLOEXEC")
        | _required_os_constant("O_DIRECTORY")
        | _required_os_constant("O_NOFOLLOW")
        | _required_os_constant("O_NONBLOCK")
    )
    file_flags = (
        os.O_RDONLY
        | _required_os_constant("O_CLOEXEC")
        | _required_os_constant("O_NOFOLLOW")
        | _required_os_constant("O_NONBLOCK")
    )
    observed_paths: set[str] = set()
    observed_directories: set[str] = set()
    visited_file_identities: set[tuple[int, int]] = set()
    visited_directory_identities: set[tuple[int, int]] = set()
    visited_entry_count = 0
    git_metadata_seen = False
    root_device = -1
    git_metadata_frame: tuple[int, ...] | None = None

    def require_descriptor_payload(
        descriptor: int,
        payload: bytes,
        path: str,
        expected_frame: tuple[int, ...],
    ) -> None:
        for _pass_index in range(2):
            offset = 0
            while offset < len(payload):
                try:
                    chunk = os.pread(
                        descriptor,
                        min(1 << 20, len(payload) - offset),
                        offset,
                    )
                except InterruptedError:
                    continue
                except OSError as error:
                    raise Experiment002RunAuthorityError(
                        f"tracked file cannot be read safely ({path})"
                    ) from error
                if (
                    type(chunk) is not bytes
                    or not chunk
                    or chunk != payload[offset : offset + len(chunk)]
                ):
                    raise Experiment002RunAuthorityError(
                        f"worktree file differs from committed HEAD blob ({path})"
                    )
                offset += len(chunk)
            try:
                trailing = os.pread(descriptor, 1, len(payload))
                after = os.fstat(descriptor)
            except OSError as error:
                raise Experiment002RunAuthorityError(
                    f"tracked file changed while being read ({path})"
                ) from error
            if trailing != b"" or _stat_frame(after) != expected_frame:
                raise Experiment002RunAuthorityError(
                    f"tracked file changed while being read ({path})"
                )

    def require_named_frame(
        directory_descriptor: int,
        name: str,
        expected_frame: tuple[int, ...],
        path: str,
    ) -> None:
        try:
            observed = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except OSError as error:
            raise Experiment002RunAuthorityError(
                f"worktree name changed during verification ({path})"
            ) from error
        if _stat_frame(observed) != expected_frame:
            raise Experiment002RunAuthorityError(
                f"worktree name changed during verification ({path})"
            )

    def visit_directory(
        directory_descriptor: int,
        relative_directory: str,
        depth: int,
    ) -> None:
        nonlocal git_metadata_seen, visited_entry_count
        try:
            before = os.fstat(directory_descriptor)
        except OSError as error:
            raise Experiment002RunAuthorityError(
                "worktree directory metadata is inaccessible"
            ) from error
        if not stat.S_ISDIR(before.st_mode):
            raise Experiment002RunAuthorityError(
                "worktree traversal entered a nondirectory"
            )
        if before.st_dev != root_device:
            raise Experiment002RunAuthorityError(
                "worktree traversal crossed a filesystem boundary"
            )
        before_frame = _stat_frame(before)
        collected: list[tuple[str, str, bool]] = []
        local_names: set[str] = set()
        try:
            entries = os.scandir(directory_descriptor)
        except OSError as error:
            raise Experiment002RunAuthorityError(
                "worktree directory cannot be enumerated"
            ) from error
        try:
            with entries:
                for entry in entries:
                    if type(entry.name) is not str or entry.name in local_names:
                        raise Experiment002RunAuthorityError(
                            "worktree enumeration returned a duplicate or invalid name"
                        )
                    local_names.add(entry.name)
                    visited_entry_count += 1
                    if visited_entry_count > _MAX_WORKTREE_ENTRIES:
                        raise Experiment002RunAuthorityError(
                            "worktree contains too many filesystem entries"
                        )
                    if relative_directory == "" and entry.name == ".git":
                        collected.append((entry.name, ".git", True))
                        continue
                    relative_path = (
                        entry.name
                        if relative_directory == ""
                        else f"{relative_directory}/{entry.name}"
                    )
                    try:
                        _require_relative_path(relative_path, "worktree entry")
                    except UnicodeEncodeError as error:
                        raise Experiment002RunAuthorityError(
                            "worktree contains a non-UTF-8 filesystem entry"
                        ) from error
                    if (
                        relative_path not in tracked_by_path
                        and relative_path not in tracked_directories
                    ):
                        try:
                            unexpected = os.stat(
                                entry.name,
                                dir_fd=directory_descriptor,
                                follow_symlinks=False,
                            )
                        except OSError as error:
                            raise Experiment002RunAuthorityError(
                                f"worktree entry is inaccessible ({relative_path})"
                            ) from error
                        if stat.S_ISREG(unexpected.st_mode):
                            kind = "regular file"
                        elif stat.S_ISDIR(unexpected.st_mode):
                            kind = "directory"
                        else:
                            kind = "symlink or special filesystem entry"
                        raise Experiment002RunAuthorityError(
                            f"worktree contains an untracked {kind} ({relative_path})"
                        )
                    collected.append((entry.name, relative_path, False))
        finally:
            try:
                after = os.fstat(directory_descriptor)
            except OSError as error:
                raise Experiment002RunAuthorityError(
                    "worktree directory changed while being enumerated"
                ) from error
            if _stat_frame(after) != before_frame:
                raise Experiment002RunAuthorityError(
                    "worktree directory changed while being enumerated"
                )

        for name, relative_path, is_git_metadata in sorted(
            collected,
            key=lambda item: item[0].encode("utf-8"),
        ):
            try:
                metadata = os.stat(
                    name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
            except OSError as error:
                raise Experiment002RunAuthorityError(
                    f"worktree entry is inaccessible ({relative_path})"
                ) from error
            metadata_frame = _stat_frame(metadata)
            if is_git_metadata:
                if (
                    git_metadata_seen
                    or git_metadata_frame is None
                    or metadata_frame != git_metadata_frame
                ):
                    raise Experiment002RunAuthorityError(
                        "top-level .git metadata changed during inventory"
                    )
                git_metadata_seen = True
                continue
            if relative_path in tracked_directories:
                if not stat.S_ISDIR(metadata.st_mode):
                    raise Experiment002RunAuthorityError(
                        "tracked worktree directory became a non-directory "
                        f"({relative_path})"
                    )
                if depth >= _MAX_WORKTREE_DEPTH:
                    raise Experiment002RunAuthorityError(
                        "worktree directory depth exceeds its limit"
                    )
                child_descriptor: int | None = None
                try:
                    child_descriptor = os.open(
                        name,
                        directory_flags,
                        dir_fd=directory_descriptor,
                    )
                    opened = os.fstat(child_descriptor)
                    identity = (opened.st_dev, opened.st_ino)
                    if (
                        not stat.S_ISDIR(opened.st_mode)
                        or opened.st_dev != root_device
                        or _stat_frame(opened) != metadata_frame
                        or identity in visited_directory_identities
                    ):
                        raise Experiment002RunAuthorityError(
                            f"worktree directory identity is invalid ({relative_path})"
                        )
                    visited_directory_identities.add(identity)
                    visit_directory(
                        child_descriptor,
                        relative_path,
                        depth + 1,
                    )
                    if _stat_frame(os.fstat(child_descriptor)) != metadata_frame:
                        raise Experiment002RunAuthorityError(
                            "worktree directory changed during traversal "
                            f"({relative_path})"
                        )
                    require_named_frame(
                        directory_descriptor,
                        name,
                        metadata_frame,
                        relative_path,
                    )
                    observed_directories.add(relative_path)
                except OSError as error:
                    raise Experiment002RunAuthorityError(
                        f"worktree directory cannot be opened safely ({relative_path})"
                    ) from error
                finally:
                    if child_descriptor is not None:
                        os.close(child_descriptor)
                continue

            tracked = tracked_by_path[relative_path]
            if not stat.S_ISREG(metadata.st_mode):
                raise Experiment002RunAuthorityError(
                    "tracked worktree path is a symlink or special filesystem entry "
                    f"({relative_path})"
                )
            file_descriptor: int | None = None
            try:
                file_descriptor = os.open(
                    name,
                    file_flags,
                    dir_fd=directory_descriptor,
                )
                opened = os.fstat(file_descriptor)
                opened_frame = _stat_frame(opened)
                identity = (opened.st_dev, opened.st_ino)
                executable = bool(opened.st_mode & 0o111)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or opened.st_nlink != 1
                    or opened.st_dev != root_device
                    or opened.st_size != tracked.byte_count
                    or opened_frame != metadata_frame
                    or identity in visited_file_identities
                    or executable != (tracked.identity.mode == "100755")
                ):
                    raise Experiment002RunAuthorityError(
                        f"tracked worktree file identity is invalid ({relative_path})"
                    )
                visited_file_identities.add(identity)
                blob = _load_preflighted_committed_blob(
                    root,
                    tracked.identity,
                    tracked.path,
                    declared_byte_count=tracked.byte_count,
                )
                require_descriptor_payload(
                    file_descriptor,
                    blob.payload,
                    relative_path,
                    opened_frame,
                )
                require_named_frame(
                    directory_descriptor,
                    name,
                    opened_frame,
                    relative_path,
                )
                observed_paths.add(relative_path)
            except OSError as error:
                raise Experiment002RunAuthorityError(
                    f"tracked worktree file cannot be opened safely ({relative_path})"
                ) from error
            finally:
                if file_descriptor is not None:
                    os.close(file_descriptor)

        try:
            final_directory = os.fstat(directory_descriptor)
        except OSError as error:
            raise Experiment002RunAuthorityError(
                "worktree directory changed during traversal"
            ) from error
        if _stat_frame(final_directory) != before_frame:
            raise Experiment002RunAuthorityError(
                "worktree directory changed during traversal"
            )

    root_descriptor: int | None = None
    git_descriptor: int | None = None
    try:
        root_before = root.lstat()
        if stat.S_ISLNK(root_before.st_mode) or not stat.S_ISDIR(root_before.st_mode):
            raise Experiment002RunAuthorityError(
                "worktree root is not a nonsymlink directory"
            )
        root_descriptor = _open_absolute_nonsymlink_directory(root, directory_flags)
        opened_root = os.fstat(root_descriptor)
        if _stat_frame(opened_root) != _stat_frame(root_before):
            raise Experiment002RunAuthorityError(
                "worktree root identity changed before inventory"
            )
        root_device = opened_root.st_dev
        visited_directory_identities.add((opened_root.st_dev, opened_root.st_ino))
        try:
            git_named = os.stat(
                ".git",
                dir_fd=root_descriptor,
                follow_symlinks=False,
            )
            git_descriptor = os.open(
                ".git",
                directory_flags,
                dir_fd=root_descriptor,
            )
            git_opened = os.fstat(git_descriptor)
        except OSError as error:
            raise Experiment002RunAuthorityError(
                "top-level .git metadata cannot be opened safely"
            ) from error
        git_identity = (git_opened.st_dev, git_opened.st_ino)
        git_metadata_frame = _stat_frame(git_opened)
        if (
            not stat.S_ISDIR(git_opened.st_mode)
            or git_opened.st_dev != root_device
            or git_metadata_frame != _stat_frame(git_named)
            or git_identity in visited_directory_identities
        ):
            raise Experiment002RunAuthorityError(
                "top-level .git metadata identity is invalid"
            )
        visited_directory_identities.add(git_identity)

        visit_directory(root_descriptor, "", 0)
        if not git_metadata_seen or git_metadata_frame is None:
            raise Experiment002RunAuthorityError(
                "worktree has no exact top-level .git metadata directory"
            )
        if _stat_frame(os.fstat(git_descriptor)) != git_metadata_frame:
            raise Experiment002RunAuthorityError(
                "top-level .git metadata changed during inventory"
            )
        require_named_frame(
            root_descriptor,
            ".git",
            git_metadata_frame,
            ".git",
        )
        root_after = root.lstat()
        if _stat_frame(root_after) != _stat_frame(root_before):
            raise Experiment002RunAuthorityError(
                "worktree root path identity changed during inventory"
            )
        reopened_root = _open_absolute_nonsymlink_directory(root, directory_flags)
        try:
            if _stat_frame(os.fstat(reopened_root)) != _stat_frame(opened_root):
                raise Experiment002RunAuthorityError(
                    "worktree root component chain changed during inventory"
                )
        finally:
            os.close(reopened_root)
    except OSError as error:
        raise Experiment002RunAuthorityError(
            "worktree root cannot be opened safely"
        ) from error
    finally:
        if git_descriptor is not None:
            os.close(git_descriptor)
        if root_descriptor is not None:
            os.close(root_descriptor)
    if observed_paths != set(tracked_by_path):
        raise Experiment002RunAuthorityError(
            "raw worktree inventory does not exactly equal the tracked tree"
        )
    if observed_directories != set(tracked_directories):
        raise Experiment002RunAuthorityError(
            "raw worktree inventory does not exactly equal tracked directories"
        )


def _open_absolute_nonsymlink_directory(path: Path, flags: int) -> int:
    if type(path) is not _PATH_TYPE or not path.is_absolute():
        raise Experiment002RunAuthorityError(
            "anchored directory path is not an exact absolute pathlib path"
        )
    parts = path.parts
    if not parts or parts[0] != os.sep:
        raise Experiment002RunAuthorityError(
            "anchored directory path has no filesystem root"
        )
    descriptor: int | None = None
    try:
        descriptor = os.open(os.sep, flags)
        for part in parts[1:]:
            named = os.stat(
                part,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
            if not stat.S_ISDIR(named.st_mode):
                raise Experiment002RunAuthorityError(
                    "anchored directory path contains a symlink or nondirectory"
                )
            child = os.open(part, flags, dir_fd=descriptor)
            try:
                if _stat_frame(os.fstat(child)) != _stat_frame(named):
                    raise Experiment002RunAuthorityError(
                        "anchored directory component identity changed"
                    )
            except BaseException:
                os.close(child)
                raise
            os.close(descriptor)
            descriptor = child
        result = descriptor
        descriptor = None
        return result
    except OSError as error:
        raise Experiment002RunAuthorityError(
            "anchored directory path cannot be opened safely"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _index_entries(root: Path) -> tuple[tuple[bytes, bytes, bytes], ...]:
    output = _git(root, "ls-files", "--stage", "-z")
    records = output.split(b"\0")
    if not records or records[-1] != b"":
        raise Experiment002RunAuthorityError("Git index output is ambiguous")
    if len(records) - 1 > _MAX_TRACKED_FILES:
        raise Experiment002RunAuthorityError("Git index contains too many entries")
    result: list[tuple[bytes, bytes, bytes]] = []
    previous_path: bytes | None = None
    for record in records[:-1]:
        try:
            metadata, path = record.split(b"\t", maxsplit=1)
            mode, oid, stage = metadata.split(b" ", maxsplit=2)
        except ValueError as error:
            raise Experiment002RunAuthorityError(
                "Git index entry is malformed"
            ) from error
        if stage != b"0" or not path or b"\0" in path:
            raise Experiment002RunAuthorityError(
                "Git index contains a non-stage-zero or invalid entry"
            )
        validated = _validated_tracked_entry(
            mode,
            oid,
            path,
            name="Git index",
        )
        if previous_path is not None and path <= previous_path:
            raise Experiment002RunAuthorityError(
                "Git index paths are duplicated or not byte sorted"
            )
        previous_path = path
        result.append(validated)
    return tuple(result)


def _tree_entries(
    root: Path, head_commit: str
) -> tuple[tuple[bytes, bytes, bytes], ...]:
    output = _git(root, "ls-tree", "-r", "-z", head_commit)
    records = output.split(b"\0")
    if not records or records[-1] != b"":
        raise Experiment002RunAuthorityError("Git HEAD tree output is ambiguous")
    if len(records) - 1 > _MAX_TRACKED_FILES:
        raise Experiment002RunAuthorityError("Git HEAD tree contains too many entries")
    result: list[tuple[bytes, bytes, bytes]] = []
    previous_path: bytes | None = None
    for record in records[:-1]:
        try:
            metadata, path = record.split(b"\t", maxsplit=1)
            mode, object_type, oid = metadata.split(b" ", maxsplit=2)
        except ValueError as error:
            raise Experiment002RunAuthorityError(
                "Git HEAD tree entry is malformed"
            ) from error
        if object_type != b"blob" or not path or b"\0" in path:
            raise Experiment002RunAuthorityError("Git HEAD tree path is invalid")
        validated = _validated_tracked_entry(
            mode,
            oid,
            path,
            name="Git HEAD tree",
        )
        if previous_path is not None and path <= previous_path:
            raise Experiment002RunAuthorityError(
                "Git HEAD tree paths are duplicated or not byte sorted"
            )
        previous_path = path
        result.append(validated)
    return tuple(result)


def _require_implementation_ancestor(
    root: Path,
    *,
    implementation_commit: str,
    head_commit: str,
) -> None:
    resolved = _git(
        root,
        "rev-parse",
        "--verify",
        f"{implementation_commit}^{{commit}}",
    )
    if _single_git_hex_line(resolved, length=40, name="implementation commit") != (
        implementation_commit
    ):
        raise Experiment002RunAuthorityError(
            "implementation commit does not resolve to its exact full ID"
        )
    if implementation_commit == head_commit:
        raise Experiment002RunAuthorityError(
            "run registration must be committed after its implementation commit"
        )
    result = _git_process(
        root,
        ("merge-base", "--is-ancestor", implementation_commit, head_commit),
        allowed_returncodes=(0, 1),
    )
    if result.returncode != 0:
        raise Experiment002RunAuthorityError(
            "implementation commit is not an ancestor of registration HEAD"
        )
    earlier_entry = _git(
        root,
        "ls-tree",
        "-z",
        implementation_commit,
        "--",
        _RUN_CONFIG_PATH,
    )
    if earlier_entry:
        raise Experiment002RunAuthorityError(
            "run registration already existed at the implementation commit"
        )


def _require_registration_commit_shape(
    root: Path,
    *,
    implementation_commit: str,
    head_commit: str,
) -> None:
    parents = _git(root, "rev-list", "--parents", "-n", "1", head_commit)
    if not parents.endswith(b"\n") or parents.count(b"\n") != 1:
        raise Experiment002RunAuthorityError(
            "registration commit parent output is ambiguous"
        )
    try:
        fields = parents[:-1].decode("ascii").split(" ")
    except UnicodeDecodeError as error:
        raise Experiment002RunAuthorityError(
            "registration commit parent output is not ASCII"
        ) from error
    if fields != [head_commit, implementation_commit]:
        raise Experiment002RunAuthorityError(
            "registration HEAD must have only the implementation commit as parent"
        )
    changes = _git(
        root,
        "diff-tree",
        "--no-ext-diff",
        "--no-textconv",
        "--no-commit-id",
        "--name-status",
        "-r",
        "-z",
        implementation_commit,
        head_commit,
    )
    if changes != b"A\0" + _RUN_CONFIG_PATH.encode("ascii") + b"\0":
        raise Experiment002RunAuthorityError(
            "registration commit must add only the canonical run config"
        )


def _committed_blob_identity(root: Path, commit: str, path: str) -> _TreeBlobIdentity:
    _require_relative_path(path, "committed path")
    output = _git(root, "ls-tree", "-z", commit, "--", path)
    if not output.endswith(b"\0") or output.count(b"\0") != 1:
        raise Experiment002RunAuthorityError(
            f"committed path is absent or ambiguous ({path})"
        )
    record = output[:-1]
    try:
        metadata, observed_path = record.split(b"\t", maxsplit=1)
        mode, object_type, oid = metadata.split(b" ", maxsplit=2)
    except ValueError as error:
        raise Experiment002RunAuthorityError(
            f"committed tree entry is malformed ({path})"
        ) from error
    if observed_path != path.encode("utf-8"):
        raise Experiment002RunAuthorityError(
            f"committed tree path is not exact ({path})"
        )
    if mode not in {b"100644", b"100755"} or object_type != b"blob":
        raise Experiment002RunAuthorityError(
            f"committed path is not a regular nonsymlink blob ({path})"
        )
    try:
        oid_text = oid.decode("ascii")
    except UnicodeDecodeError as error:
        raise Experiment002RunAuthorityError(
            f"committed blob ID is not ASCII ({path})"
        ) from error
    _require_hex(oid_text, length=40, name=f"blob ID for {path}")
    return _TreeBlobIdentity(mode=mode.decode("ascii"), oid=oid_text)


def _committed_blob(
    root: Path,
    commit: str,
    path: str,
    *,
    maximum_bytes: int,
) -> _TreeBlob:
    identity = _committed_blob_identity(root, commit, path)
    declared_byte_count = _preflight_committed_blob_size(
        root,
        identity,
        path,
        maximum_bytes=maximum_bytes,
    )
    return _load_preflighted_committed_blob(
        root,
        identity,
        path,
        declared_byte_count=declared_byte_count,
    )


def _preflight_committed_blob_size(
    root: Path,
    identity: _TreeBlobIdentity,
    path: str,
    *,
    maximum_bytes: int,
) -> int:
    if (
        type(maximum_bytes) is not int
        or not 0 <= maximum_bytes <= _MAX_CHILD_BUNDLE_BYTES
    ):
        raise Experiment002RunAuthorityError(
            f"committed blob retained byte budget is invalid ({path})"
        )
    if type(identity) is not _TreeBlobIdentity:
        raise Experiment002RunAuthorityError(
            f"committed blob identity is invalid ({path})"
        )
    declared_byte_count = _single_git_size_line(
        _git(root, "cat-file", "-s", identity.oid),
        name=f"blob size for {path}",
    )
    if declared_byte_count > maximum_bytes:
        raise Experiment002RunAuthorityError(
            f"committed blob exceeds its retained byte budget ({path})"
        )
    return declared_byte_count


def _load_preflighted_committed_blob(
    root: Path,
    identity: _TreeBlobIdentity,
    path: str,
    *,
    declared_byte_count: int,
) -> _TreeBlob:
    if (
        type(identity) is not _TreeBlobIdentity
        or type(declared_byte_count) is not int
        or not 0 <= declared_byte_count <= _MAX_CHILD_BUNDLE_BYTES
    ):
        raise Experiment002RunAuthorityError(
            f"committed blob preflight frame is invalid ({path})"
        )
    payload = _git(
        root,
        "cat-file",
        "blob",
        identity.oid,
        maximum_stdout_bytes=declared_byte_count,
    )
    if len(payload) != declared_byte_count:
        raise Experiment002RunAuthorityError(
            f"committed blob differs from its size preflight ({path})"
        )
    object_digest = hashlib.sha1(usedforsecurity=False)
    object_digest.update(b"blob " + str(len(payload)).encode("ascii") + b"\0")
    object_digest.update(payload)
    observed_oid = object_digest.hexdigest()
    if observed_oid != identity.oid:
        raise Experiment002RunAuthorityError(
            f"committed blob payload does not match its Git object ID ({path})"
        )
    return _TreeBlob(mode=identity.mode, oid=identity.oid, payload=payload)


def _git(
    root: Path,
    *arguments: str,
    maximum_stdout_bytes: int = _MAX_GIT_OUTPUT_BYTES,
) -> bytes:
    return _git_process(
        root,
        arguments,
        allowed_returncodes=(0,),
        maximum_stdout_bytes=maximum_stdout_bytes,
    ).stdout


def _git_process(
    root: Path,
    arguments: tuple[str, ...],
    *,
    allowed_returncodes: tuple[int, ...],
    maximum_stdout_bytes: int = _MAX_GIT_OUTPUT_BYTES,
) -> subprocess.CompletedProcess[bytes]:
    if (
        type(maximum_stdout_bytes) is not int
        or maximum_stdout_bytes < 0
        or maximum_stdout_bytes > _MAX_CHILD_BUNDLE_BYTES
    ):
        raise Experiment002RunAuthorityError("isolated Git output limit is invalid")
    executable = shutil.which("git", path="/usr/bin:/bin")
    if executable is None:
        raise Experiment002RunAuthorityError(
            "an absolute trusted Git executable is absent"
        )
    executable_path = Path(executable)
    try:
        executable_stat = executable_path.lstat()
    except OSError as error:
        raise Experiment002RunAuthorityError(
            "Git executable is inaccessible"
        ) from error
    if (
        not executable_path.is_absolute()
        or stat.S_ISLNK(executable_stat.st_mode)
        or not stat.S_ISREG(executable_stat.st_mode)
    ):
        raise Experiment002RunAuthorityError(
            "Git executable must be an absolute regular nonsymlink file"
        )
    command = (
        os.fspath(executable_path),
        "--no-pager",
        "--literal-pathspecs",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.untrackedCache=false",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "submodule.recurse=false",
        *arguments,
    )
    environment = {
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_LITERAL_PATHSPECS": "1",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "HOME": "/nonexistent",
        "LANG": "C",
        "LC_ALL": "C",
        "XDG_CONFIG_HOME": "/nonexistent",
    }
    process: subprocess.Popen[bytes] | None = None
    process_descriptor: int | None = None
    deadline = time.monotonic() + _GIT_TIMEOUT_SECONDS
    try:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            close_fds=True,
            start_new_session=True,
        )
        process_descriptor = os.pidfd_open(process.pid, 0)
        completed = _collect_bounded_git_process(
            process,
            process_descriptor=process_descriptor,
            command=command,
            deadline=deadline,
            maximum_stdout_bytes=maximum_stdout_bytes,
        )
    except (AttributeError, OSError, subprocess.TimeoutExpired) as error:
        raise Experiment002RunAuthorityError("isolated Git command failed") from error
    finally:
        if process is not None:
            _close_kill_and_reap_git_process(process)
        if process_descriptor is not None:
            try:
                os.close(process_descriptor)
            except OSError:
                process_descriptor = None
    if (
        type(completed.returncode) is not int
        or completed.returncode not in allowed_returncodes
    ):
        raise Experiment002RunAuthorityError(
            "isolated Git command returned an unexpected status"
        )
    if type(completed.stdout) is not bytes or type(completed.stderr) is not bytes:
        raise Experiment002RunAuthorityError("isolated Git output type is invalid")
    if (
        len(completed.stdout) > maximum_stdout_bytes
        or len(completed.stderr) > _MAX_GIT_OUTPUT_BYTES
    ):
        raise Experiment002RunAuthorityError("isolated Git output is too large")
    return completed


def _collect_bounded_git_process(
    process: subprocess.Popen[bytes],
    *,
    process_descriptor: int,
    command: tuple[str, ...],
    deadline: float,
    maximum_stdout_bytes: int,
) -> subprocess.CompletedProcess[bytes]:
    stdout = process.stdout
    stderr = process.stderr
    if stdout is None or stderr is None:
        raise Experiment002RunAuthorityError("isolated Git pipes are unavailable")
    stdout_bytes = bytearray()
    stderr_bytes = bytearray()
    selector = selectors.DefaultSelector()
    try:
        stdout_fd = stdout.fileno()
        stderr_fd = stderr.fileno()
        os.set_blocking(stdout_fd, False)
        os.set_blocking(stderr_fd, False)
        selector.register(stdout_fd, selectors.EVENT_READ, data="stdout")
        selector.register(stderr_fd, selectors.EVENT_READ, data="stderr")
        selector.register(
            process_descriptor,
            selectors.EVENT_READ,
            data="process",
        )
        while selector.get_map():
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                raise subprocess.TimeoutExpired(command, _GIT_TIMEOUT_SECONDS)
            try:
                events = selector.select(remaining_seconds)
            except InterruptedError:
                continue
            if not events:
                raise subprocess.TimeoutExpired(command, _GIT_TIMEOUT_SECONDS)
            for key, _event_mask in events:
                if key.data == "process":
                    selector.unregister(key.fd)
                    continue
                if key.data == "stdout":
                    target = stdout_bytes
                    byte_limit = maximum_stdout_bytes
                else:
                    target = stderr_bytes
                    byte_limit = _MAX_GIT_OUTPUT_BYTES
                read_byte_count = min(1 << 20, byte_limit + 1 - len(target))
                if read_byte_count < 1:
                    read_byte_count = 1
                try:
                    chunk = os.read(key.fd, read_byte_count)
                except (BlockingIOError, InterruptedError):
                    continue
                if not chunk:
                    selector.unregister(key.fd)
                    continue
                target.extend(chunk)
                if len(target) > byte_limit:
                    raise Experiment002RunAuthorityError(
                        "isolated Git output is too large"
                    )
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            raise subprocess.TimeoutExpired(command, _GIT_TIMEOUT_SECONDS)
        _kill_git_process_group(process)
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            raise subprocess.TimeoutExpired(command, _GIT_TIMEOUT_SECONDS)
        returncode = process.wait(timeout=remaining_seconds)
    finally:
        selector.close()
    return subprocess.CompletedProcess(
        args=command,
        returncode=returncode,
        stdout=bytes(stdout_bytes),
        stderr=bytes(stderr_bytes),
    )


def _close_kill_and_reap_git_process(process: subprocess.Popen[bytes]) -> None:
    for stream in (process.stdout, process.stderr):
        if stream is not None:
            try:
                stream.close()
            except OSError:
                continue
    if process.returncode is None:
        _kill_git_process_group(process)
    try:
        process.wait(timeout=1.0)
    except (OSError, subprocess.TimeoutExpired):
        try:
            process.kill()
        except OSError as error:
            del error
        try:
            process.wait(timeout=1.0)
        except (OSError, subprocess.TimeoutExpired) as error:
            del error


def _kill_git_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError:
        try:
            process.kill()
        except OSError:
            return


def _single_git_hex_line(output: bytes, *, length: int, name: str) -> str:
    if not output.endswith(b"\n") or output.count(b"\n") != 1 or b"\0" in output:
        raise Experiment002RunAuthorityError(f"{name} output is ambiguous")
    try:
        text = output[:-1].decode("ascii")
    except UnicodeDecodeError as error:
        raise Experiment002RunAuthorityError(f"{name} is not ASCII") from error
    return _require_hex(text, length=length, name=name)


def _single_git_size_line(output: bytes, *, name: str) -> int:
    if not output.endswith(b"\n") or output.count(b"\n") != 1 or b"\0" in output:
        raise Experiment002RunAuthorityError(f"{name} output is ambiguous")
    try:
        text = output[:-1].decode("ascii", errors="strict")
    except UnicodeDecodeError as error:
        raise Experiment002RunAuthorityError(f"{name} is not ASCII") from error
    if not text or not text.isdecimal() or (len(text) > 1 and text.startswith("0")):
        raise Experiment002RunAuthorityError(f"{name} is not canonical decimal")
    value = int(text)
    if str(value) != text or value > _MAX_CHILD_BUNDLE_BYTES:
        raise Experiment002RunAuthorityError(f"{name} exceeds the absolute byte cap")
    return value


def _require_path_list(value: object, name: str) -> tuple[str, ...]:
    if type(value) is not list:
        raise Experiment002RunAuthorityError(f"{name} must be an exact JSON array")
    result: list[str] = []
    for item in cast(list[object], value):
        result.append(_require_relative_path(item, name))
    expected = sorted(result, key=lambda path: path.encode("utf-8"))
    if result != expected:
        raise Experiment002RunAuthorityError(f"{name} must be UTF-8 byte sorted")
    if len(set(result)) != len(result):
        raise Experiment002RunAuthorityError(f"{name} must not contain duplicates")
    return tuple(result)


def _require_relative_path(value: object, name: str) -> str:
    if type(value) is not str:
        raise Experiment002RunAuthorityError(f"{name} must contain exact strings")
    path = value
    if (
        not path
        or len(path.encode("utf-8")) > 4_096
        or path.startswith("/")
        or path.endswith("/")
        or "//" in path
        or "\\" in path
        or "\x00" in path
        or any(ord(character) < 0x21 or ord(character) > 0x7E for character in path)
    ):
        raise Experiment002RunAuthorityError(f"{name} contains an invalid path")
    parsed = PurePosixPath(path)
    if parsed.as_posix() != path or any(part in {".", ".."} for part in parsed.parts):
        raise Experiment002RunAuthorityError(f"{name} contains path traversal")
    if parsed.parts[0] == ".git":
        raise Experiment002RunAuthorityError(f"{name} must not enter .git")
    return path


def _require_hex(value: object, *, length: int, name: str) -> str:
    if type(value) is not str:
        raise Experiment002RunAuthorityError(f"{name} must be an exact string")
    text = value
    alphabet = _LOWER_HEX_40 if length == 40 else _LOWER_HEX_64
    if len(text) != length or any(character not in alphabet for character in text):
        raise Experiment002RunAuthorityError(
            f"{name} must be exactly {length} lowercase hexadecimal characters"
        )
    return text


def _require_exact_keys(value: dict[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected or any(type(key) is not str for key in value):
        raise Experiment002RunAuthorityError(f"{name} has invalid keys")


def _exact_json_equal(observed: object, expected: object) -> bool:
    if type(observed) is not type(expected):
        return False
    if type(expected) is dict:
        observed_dict = cast(dict[object, object], observed)
        expected_dict = cast(dict[object, object], expected)
        return set(observed_dict) == set(expected_dict) and all(
            _exact_json_equal(observed_dict[key], value)
            for key, value in expected_dict.items()
        )
    if type(expected) is list:
        observed_list = cast(list[object], observed)
        expected_list = cast(list[object], expected)
        return len(observed_list) == len(expected_list) and all(
            _exact_json_equal(left, right)
            for left, right in zip(observed_list, expected_list, strict=True)
        )
    return observed == expected


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise Experiment002RunAuthorityError(
                "run registration contains a duplicate or invalid JSON key"
            )
        result[key] = value
    return result


def _reject_json_number(value: str) -> NoReturn:
    raise Experiment002RunAuthorityError(
        f"run registration contains a forbidden JSON number ({value})"
    )
