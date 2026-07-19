"""Source-bound run-registration authority for Experiment 002.

The canonical registration file is intentionally absent today.  This module is
therefore a closed gate: it can issue an authority only after a later, separately
committed ``configs/experiment-002-run.json`` passes every repository, source,
and frozen-binding check below.  The verifier is deliberately standard-library
only so callers can run it before importing any numerical or registered-data
module.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import stat
import struct
import subprocess
import sys
import threading
import weakref
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final, NoReturn, SupportsIndex, cast

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
_MAX_GIT_OUTPUT_BYTES: Final = 32 << 20
_GIT_TIMEOUT_SECONDS: Final = 30
_LOWER_HEX_40 = frozenset("0123456789abcdef")
_LOWER_HEX_64 = _LOWER_HEX_40
_ISSUER_MARKER: Final = object()
_PATH_TYPE: Final = type(Path())
_EXECUTING_FILE: Final = Path(__file__).absolute()
_LOADED_SOURCE_SHA256: Final = hashlib.sha256(_EXECUTING_FILE.read_bytes()).hexdigest()


class Experiment002RunAuthorityError(ValueError):
    """The source-bound Experiment 002 registration failed closed."""


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
class _VerifiedState:
    issuer_marker: object
    repository_root: Path
    head_commit: str
    implementation_commit: str
    registration_sha256: str
    source_bundle_sha256: str
    source_paths: tuple[str, ...]
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


@dataclass(frozen=True, slots=True)
class _IssuedGuard:
    issuer_marker: object
    repository_root: Path
    head_commit: str
    implementation_commit: str
    registration_sha256: str
    source_bundle_sha256: str
    source_paths: tuple[str, ...]
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


_ISSUED: weakref.WeakKeyDictionary[VerifiedRunRegistration, _VerifiedState] = (
    weakref.WeakKeyDictionary()
)
_ISSUED_GUARDS: weakref.WeakKeyDictionary[VerifiedRunRegistration, _IssuedGuard] = (
    weakref.WeakKeyDictionary()
)
_FAILED: weakref.WeakSet[VerifiedRunRegistration] = weakref.WeakSet()
_ISSUED_LOCK = threading.RLock()
_ISSUANCE_COMPLETE = False


def verify_and_issue_experiment_002_run_registration() -> VerifiedRunRegistration:
    """Verify the later committed registration and issue an opaque authority.

    No repository, digest, source list, commit, or other trust value is accepted
    as an argument.  The physical repository is derived from this module's own
    verified source location.
    """

    with _ISSUED_LOCK:
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


def verify_verified_run_registration(
    registration: VerifiedRunRegistration,
) -> None:
    """Reverify repository state and reject any forged or stale capability."""

    reverify_verified_run_registration(registration)


def reverify_verified_run_registration(
    registration: VerifiedRunRegistration,
) -> None:
    """Fully recheck repository state, poisoning the capability on mismatch."""

    state = _verified_state(registration)
    try:
        raw_config = _read_regular_file(
            state.repository_root / _RUN_CONFIG_PATH,
            maximum_bytes=_MAX_CONFIG_BYTES,
        )
        document = _parse_registration(raw_config)
        observed = _verify_committed_repository(state.repository_root, document)
        _require_registered_runtime_identity(state.repository_root, document)
        with _ISSUED_LOCK:
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
            _FAILED.add(registration)
        raise


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

    with _ISSUED_LOCK:
        return _record_verified_snapshot_locked(snapshot)


def _record_verified_snapshot_locked(
    snapshot: _RepositorySnapshot,
) -> VerifiedRunRegistration:
    global _ISSUANCE_COMPLETE

    if _ISSUANCE_COMPLETE:
        raise Experiment002RunAuthorityError(
            "a run registration was already issued in this process"
        )
    verified = _state_from_snapshot(snapshot)
    capability = object.__new__(VerifiedRunRegistration)
    guard = _guard_from_state(verified)
    _ISSUED[capability] = verified
    _ISSUED_GUARDS[capability] = guard
    _ISSUANCE_COMPLETE = True
    return capability


def _verified_state(registration: VerifiedRunRegistration) -> _VerifiedState:
    if type(registration) is not VerifiedRunRegistration:
        raise TypeError("registration must be an exact VerifiedRunRegistration")
    with _ISSUED_LOCK:
        state = _ISSUED.get(registration)
        guard = _ISSUED_GUARDS.get(registration)
        failed = registration in _FAILED
        if (
            state is None
            or guard is None
            or failed
            or state.issuer_marker is not _ISSUER_MARKER
            or state.process_id != os.getpid()
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
        process_id=state.process_id,
        nonce=state.nonce,
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
    if type(process_id) is not int or process_id < 1 or type(nonce) is not object:
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
        process_id=os.getpid(),
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
        and state.process_id == os.getpid()
    )


def _verify_committed_repository(
    root: Path,
    document: _RegistrationDocument,
) -> _RepositorySnapshot:
    _require_git_toplevel(root)
    head_commit = _head_commit(root)

    config_blob = _committed_blob(root, head_commit, _RUN_CONFIG_PATH)
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
    _require_frozen_files(root, head_commit)

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
    for path in document.source_paths:
        implementation_blob = _committed_blob(
            root, document.implementation_commit, path
        )
        head_blob = _committed_blob(root, head_commit, path)
        if (
            implementation_blob.mode != head_blob.mode
            or implementation_blob.payload != head_blob.payload
        ):
            raise Experiment002RunAuthorityError(
                f"registered source changed after implementation commit ({path})"
            )
        _require_worktree_blob(root, path, head_blob)
        if (
            path == _AUTHORITY_SOURCE_PATH
            and hashlib.sha256(implementation_blob.payload).hexdigest()
            != _LOADED_SOURCE_SHA256
        ):
            raise Experiment002RunAuthorityError(
                "executing run-authority source differs from its bound blob"
            )
        source_blobs.append((path, implementation_blob.payload))
        tree_blobs[path] = head_blob

    payload_byte_count, source_digest = _source_bundle_digest(tuple(source_blobs))
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
    _require_clean_worktree(root, head_commit)
    _require_worktree_blob(root, _RUN_CONFIG_PATH, config_blob)
    for path, expected_sha256 in _frozen_file_bindings():
        blob = _committed_blob(root, head_commit, path)
        if hashlib.sha256(blob.payload).hexdigest() != expected_sha256:
            raise Experiment002RunAuthorityError(
                f"frozen binding changed during verification ({path})"
            )
        _require_worktree_blob(root, path, blob)
    for path, blob in tree_blobs.items():
        _require_worktree_blob(root, path, blob)
    if _enumerate_import_roots(root, document.import_roots) != document.source_paths:
        raise Experiment002RunAuthorityError(
            "registered import roots changed during verification"
        )

    return _RepositorySnapshot(
        repository_root=root,
        head_commit=head_commit,
        implementation_commit=document.implementation_commit,
        registration_sha256=hashlib.sha256(document.raw_bytes).hexdigest(),
        source_bundle_sha256=source_digest,
        source_paths=document.source_paths,
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
    if not source_paths:
        raise Experiment002RunAuthorityError("at least one source path is required")
    if len(source_paths) > _MAX_SOURCE_FILES:
        raise Experiment002RunAuthorityError("too many registered source paths")
    for first_index, first in enumerate(import_roots):
        first_path = PurePosixPath(first)
        for second in import_roots[first_index + 1 :]:
            second_path = PurePosixPath(second)
            if first_path in second_path.parents or second_path in first_path.parents:
                raise Experiment002RunAuthorityError(
                    "registered import roots must not overlap"
                )
    byte_count = source_object["payload_byte_count"]
    if type(byte_count) is not int or not 0 <= byte_count <= _MAX_SOURCE_BUNDLE_BYTES:
        raise Experiment002RunAuthorityError(
            "source bundle payload_byte_count is invalid"
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


def _require_frozen_files(root: Path, head_commit: str) -> None:
    for path, expected_sha256 in _frozen_file_bindings():
        blob = _committed_blob(root, head_commit, path)
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


def _source_bundle_digest(
    sources: tuple[tuple[str, bytes], ...],
) -> tuple[int, str]:
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
    digest = hashlib.sha256(_SOURCE_BUNDLE_DOMAIN + payload).hexdigest()
    return len(payload), digest


def _enumerate_import_roots(
    root: Path, import_roots: tuple[str, ...]
) -> tuple[str, ...]:
    result: list[str] = []
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
                entries = sorted(
                    os.scandir(directory),
                    key=lambda entry: entry.name.encode("utf-8"),
                    reverse=True,
                )
            except OSError as error:
                raise Experiment002RunAuthorityError(
                    "registered import root cannot be enumerated "
                    f"({relative_directory})"
                ) from error
            for entry in entries:
                child_relative = f"{relative_directory}/{entry.name}"
                _require_relative_path(child_relative, "import-root entry")
                try:
                    child_stat = entry.stat(follow_symlinks=False)
                except OSError as error:
                    raise Experiment002RunAuthorityError(
                        "registered import-root entry is inaccessible "
                        f"({child_relative})"
                    ) from error
                if stat.S_ISLNK(child_stat.st_mode):
                    raise Experiment002RunAuthorityError(
                        f"registered import root contains a symlink ({child_relative})"
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
                if len(result) + len(pending) > _MAX_SOURCE_FILES:
                    raise Experiment002RunAuthorityError(
                        "registered import roots contain too many entries"
                    )
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
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise Experiment002RunAuthorityError(
            f"required path is not a regular nonsymlink file ({path})"
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
        if not stat.S_ISREG(opened.st_mode) or _stat_frame(opened) != before_frame:
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


def _head_commit(root: Path) -> str:
    output = _git(root, "rev-parse", "--verify", "HEAD^{commit}")
    return _single_git_hex_line(output, length=40, name="HEAD commit")


def _require_clean_worktree(root: Path, head_commit: str) -> None:
    output = _git(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )
    if output:
        raise Experiment002RunAuthorityError(
            "repository worktree or index is not clean, including untracked files"
        )
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


def _index_entries(root: Path) -> tuple[tuple[bytes, bytes, bytes], ...]:
    output = _git(root, "ls-files", "--stage", "-z")
    records = output.split(b"\0")
    if not records or records[-1] != b"":
        raise Experiment002RunAuthorityError("Git index output is ambiguous")
    result: list[tuple[bytes, bytes, bytes]] = []
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
        result.append((mode, oid, path))
    return tuple(result)


def _tree_entries(
    root: Path, head_commit: str
) -> tuple[tuple[bytes, bytes, bytes], ...]:
    output = _git(root, "ls-tree", "-r", "-z", head_commit)
    records = output.split(b"\0")
    if not records or records[-1] != b"":
        raise Experiment002RunAuthorityError("Git HEAD tree output is ambiguous")
    result: list[tuple[bytes, bytes, bytes]] = []
    for record in records[:-1]:
        try:
            metadata, path = record.split(b"\t", maxsplit=1)
            mode, _object_type, oid = metadata.split(b" ", maxsplit=2)
        except ValueError as error:
            raise Experiment002RunAuthorityError(
                "Git HEAD tree entry is malformed"
            ) from error
        if not path or b"\0" in path:
            raise Experiment002RunAuthorityError("Git HEAD tree path is invalid")
        result.append((mode, oid, path))
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


def _committed_blob(root: Path, commit: str, path: str) -> _TreeBlob:
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
    payload = _git(root, "cat-file", "blob", oid_text)
    if len(payload) > _MAX_SOURCE_FILE_BYTES and path not in {
        binding_path for binding_path, _ in _frozen_file_bindings()
    }:
        raise Experiment002RunAuthorityError(
            f"committed source blob is too large ({path})"
        )
    return _TreeBlob(mode=mode.decode("ascii"), oid=oid_text, payload=payload)


def _git(root: Path, *arguments: str) -> bytes:
    return _git_process(root, arguments, allowed_returncodes=(0,)).stdout


def _git_process(
    root: Path,
    arguments: tuple[str, ...],
    *,
    allowed_returncodes: tuple[int, ...],
) -> subprocess.CompletedProcess[bytes]:
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
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "HOME": "/nonexistent",
        "LANG": "C",
        "LC_ALL": "C",
        "XDG_CONFIG_HOME": "/nonexistent",
    }
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            env=environment,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise Experiment002RunAuthorityError("isolated Git command failed") from error
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
        len(completed.stdout) > _MAX_GIT_OUTPUT_BYTES
        or len(completed.stderr) > _MAX_GIT_OUTPUT_BYTES
    ):
        raise Experiment002RunAuthorityError("isolated Git output is too large")
    return completed


def _single_git_hex_line(output: bytes, *, length: int, name: str) -> str:
    if not output.endswith(b"\n") or output.count(b"\n") != 1 or b"\0" in output:
        raise Experiment002RunAuthorityError(f"{name} output is ambiguous")
    try:
        text = output[:-1].decode("ascii")
    except UnicodeDecodeError as error:
        raise Experiment002RunAuthorityError(f"{name} is not ASCII") from error
    return _require_hex(text, length=length, name=name)


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
