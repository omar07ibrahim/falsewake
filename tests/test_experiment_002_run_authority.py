from __future__ import annotations

import ast
import copy
import errno
import fcntl
import hashlib
import importlib.machinery
import importlib.metadata
import inspect
import json
import os
import pickle
import platform
import select
import shutil
import signal
import socket
import stat
import struct
import subprocess
import sys
import threading
import time
import warnings
import weakref
import zlib
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from pathlib import Path, PosixPath
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

import falsewake.experiment_002_run_authority as authority

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_CONFIG_PATH = Path("configs/experiment-002-run.json")
SOURCE_DOMAIN = b"falsewake-exp002-source-bundle-v1\0"
RUNTIME_DOMAIN = b"falsewake-exp002-runtime-v1\0"
BUNDLE_MAGIC = b"FW2CHLD1"
BUNDLE_VERSION = 1
BUNDLE_HEADER = struct.Struct("<8sI40s40s64s64sIQQ")
PYTHON_EXECUTABLE = "/home/ubuntu/gitcode/.t/falsewake-venv/bin/python"
RUNTIME_SYS_PATH_ROOTS = (
    "/usr/lib/python312.zip",
    "/usr/lib/python3.12",
    "/usr/lib/python3.12/lib-dynload",
    "/home/ubuntu/gitcode/.t/falsewake-venv/lib/python3.12/site-packages",
)
ENTRYPOINT_PATH = "src/falsewake/experiment_002_runner.py"
FROZEN_FILE_PATHS = (
    "configs/experiment-002-numerics.json",
    "configs/experiment-002-trainer.json",
    "configs/experiment-002-training.json",
    "models/experiment-002-normalization.f32",
    "reports/experiment-002-implementation-audit.json",
    "reports/experiment-002-normalization.json",
    "reports/experiment-002-pcm-cache.json",
    "src/falsewake/causal_kws.py",
)


@dataclass(slots=True)
class RepositoryCase:
    root: Path
    implementation_commit: str
    head_commit: str
    document: dict[str, Any]
    source_paths: tuple[str, ...]
    source_payload_byte_count: int
    source_sha256: str


class FakeSealedSourceFinder:
    get_data: Callable[[str], bytes]
    _verify_sealed_import_state: Callable[[], None]

    def __init__(self, source_map: dict[str, bytes]) -> None:
        self.source_map = dict(source_map)
        self.valid = True
        self.verification_count = 0
        self.meta_path_object: object | None = None
        self.meta_path_frame: tuple[object, ...] | None = None
        self.sys_path_object: object | None = None

        def get_data(origin: str) -> bytes:
            if type(origin) is not str or origin not in self.source_map:
                raise FileNotFoundError(origin)
            return self.source_map[origin]

        def verify() -> None:
            self.verification_count += 1
            if (
                not self.valid
                or type(sys.modules) is not dict
                or any(type(name) is not str for name in sys.modules)
                or type(sys.meta_path) is not list
                or not sys.meta_path
                or cast(object, sys.meta_path[0]) is not self
                or type(sys.path) is not list
                or any(type(path) is not str for path in sys.path)
                or tuple(sys.path) != RUNTIME_SYS_PATH_ROOTS
                or type(self.source_map) is not dict
                or any(
                    type(origin) is not str or type(payload) is not bytes
                    for origin, payload in self.source_map.items()
                )
            ):
                raise RuntimeError("sealed finder state changed")
            if self.meta_path_object is None:
                self.meta_path_object = sys.meta_path
                self.meta_path_frame = tuple(cast(list[object], sys.meta_path))
                self.sys_path_object = sys.path
            if (
                cast(object, sys.meta_path) is not self.meta_path_object
                or self.meta_path_frame is None
                or len(sys.meta_path) != len(self.meta_path_frame)
                or any(
                    observed is not expected
                    for observed, expected in zip(
                        cast(list[object], sys.meta_path),
                        self.meta_path_frame,
                        strict=True,
                    )
                )
                or cast(object, sys.path) is not self.sys_path_object
            ):
                raise RuntimeError("sealed finder import surfaces changed")

        self.get_data = get_data
        self._verify_sealed_import_state = verify


@dataclass(slots=True)
class SealedChildHarness:
    case: RepositoryCase
    bundle: bytes
    frame: authority._ChildBundleFrame
    finder: FakeSealedSourceFinder
    spec: importlib.machinery.ModuleSpec
    origin: str


class EqualStringSubclass(str):
    pass


class EqualIntegerSubclass(int):
    pass


def _git(repository: Path, *arguments: str, input_bytes: bytes | None = None) -> bytes:
    environment = dict(os.environ)
    environment.update(
        {
            "GIT_AUTHOR_DATE": "2026-07-19T00:00:00+00:00",
            "GIT_AUTHOR_EMAIL": "authority@example.invalid",
            "GIT_AUTHOR_NAME": "Run Authority Test",
            "GIT_COMMITTER_DATE": "2026-07-19T00:00:00+00:00",
            "GIT_COMMITTER_EMAIL": "authority@example.invalid",
            "GIT_COMMITTER_NAME": "Run Authority Test",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": "/dev/null",
        }
    )
    completed = subprocess.run(
        ("/usr/bin/git", *arguments),
        cwd=repository,
        env=environment,
        input=input_bytes,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(f"git {' '.join(arguments)} failed: {completed.stderr!r}")
    return completed.stdout


def _canonical(document: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            document,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _worktree_snapshot(repository: Path) -> tuple[tuple[str, str, int, str], ...]:
    result: list[tuple[str, str, int, str]] = []
    for path in sorted(repository.rglob("*"), key=lambda value: value.as_posix()):
        relative = path.relative_to(repository)
        if relative.parts and relative.parts[0] == ".git":
            continue
        metadata = path.lstat()
        if path.is_symlink():
            kind = "symlink"
            digest = os.readlink(path)
        elif path.is_dir():
            kind = "directory"
            digest = ""
        else:
            kind = "file"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        result.append((relative.as_posix(), kind, metadata.st_mode, digest))
    return tuple(result)


def _source_payload(repository: Path, source_paths: tuple[str, ...]) -> bytes:
    payload = bytearray(struct.pack("<I", len(source_paths)))
    for path in source_paths:
        path_bytes = path.encode("utf-8")
        blob = (repository / path).read_bytes()
        payload.extend(struct.pack("<I", len(path_bytes)))
        payload.extend(path_bytes)
        payload.extend(struct.pack("<Q", len(blob)))
        payload.extend(blob)
    return bytes(payload)


def _source_digest(repository: Path, source_paths: tuple[str, ...]) -> tuple[int, str]:
    payload = _source_payload(repository, source_paths)
    return len(payload), hashlib.sha256(SOURCE_DOMAIN + payload).hexdigest()


def _blob_payload(blobs: tuple[tuple[str, bytes], ...]) -> bytes:
    payload = bytearray(struct.pack("<I", len(blobs)))
    for path, blob in blobs:
        path_bytes = path.encode("utf-8")
        payload.extend(struct.pack("<I", len(path_bytes)))
        payload.extend(path_bytes)
        payload.extend(struct.pack("<Q", len(blob)))
        payload.extend(blob)
    return bytes(payload)


def _frozen_payload(repository: Path) -> bytes:
    return _blob_payload(
        tuple((path, (repository / path).read_bytes()) for path in FROZEN_FILE_PATHS)
    )


def _bundle_oracle(case: RepositoryCase) -> bytes:
    registration = (case.root / RUN_CONFIG_PATH).read_bytes()
    source = _source_payload(case.root, case.source_paths)
    frozen = _frozen_payload(case.root)
    return b"".join(
        (
            BUNDLE_HEADER.pack(
                BUNDLE_MAGIC,
                BUNDLE_VERSION,
                case.head_commit.encode("ascii"),
                case.implementation_commit.encode("ascii"),
                hashlib.sha256(registration).hexdigest().encode("ascii"),
                hashlib.sha256(SOURCE_DOMAIN + source).hexdigest().encode("ascii"),
                len(registration),
                len(source),
                len(frozen),
            ),
            registration,
            source,
            frozen,
        )
    )


def _independent_bundle_sections(
    payload: bytes,
) -> tuple[tuple[bytes | int, ...], bytes, bytes, bytes]:
    assert len(payload) >= BUNDLE_HEADER.size
    header = BUNDLE_HEADER.unpack_from(payload)
    registration_size = cast(int, header[6])
    source_size = cast(int, header[7])
    frozen_size = cast(int, header[8])
    expected_size = BUNDLE_HEADER.size + registration_size + source_size + frozen_size
    assert len(payload) == expected_size
    offset = BUNDLE_HEADER.size
    registration = payload[offset : offset + registration_size]
    offset += registration_size
    source = payload[offset : offset + source_size]
    offset += source_size
    frozen = payload[offset:]
    return header, registration, source, frozen


def _open_descriptors() -> set[str]:
    return set(os.listdir("/proc/self/fd"))


def _runtime() -> dict[str, Any]:
    fields = {
        "numpy_version": "2.5.1",
        "python_executable": PYTHON_EXECUTABLE,
        "python_sys_path_roots": list(RUNTIME_SYS_PATH_ROOTS),
        "python_version": "3.12.3",
        "safetensors_version": "0.8.0",
        "torch_version": "2.13.0+cpu",
    }
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
    return {
        **fields,
        "fingerprint_sha256": hashlib.sha256(RUNTIME_DOMAIN + payload).hexdigest(),
    }


def _external_inputs() -> dict[str, Any]:
    frozen_inputs = cast(
        dict[str, Any], authority._expected_frozen_bindings()["inputs"]
    )
    pcm = cast(
        dict[str, Any],
        cast(dict[str, Any], authority._expected_frozen_bindings()["artifacts"])[
            "pcm_cache"
        ],
    )
    return {
        "archive": {
            "path": "/home/ubuntu/gitcode/.t/authority-test/archive.tar.gz",
            "sha256": frozen_inputs["archive_sha256"],
        },
        "manifest": {
            "inventory_sha256": frozen_inputs["manifest_inventory_sha256"],
            "path": "/home/ubuntu/gitcode/.t/authority-test/manifest.jsonl",
            "record_count": frozen_inputs["manifest_record_count"],
            "sha256": frozen_inputs["manifest_sha256"],
        },
        "pcm_cache": {
            "byte_count": pcm["byte_count"],
            "path": "/home/ubuntu/gitcode/.t/authority-test/pcm-cache.bin",
            "sha256": pcm["sha256"],
        },
    }


def _invocation(repository: Path) -> dict[str, Any]:
    return {
        "argv": [ENTRYPOINT_PATH],
        "authority_process_role": "parent_bootstrap",
        "cwd": os.fspath(repository),
        "environment": {
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
            "TMPDIR": "/home/ubuntu/gitcode/.t/authority-test/scratch",
            "TZ": "UTC",
        },
        "orig_argv": [
            PYTHON_EXECUTABLE,
            "-I",
            "-S",
            "-B",
            ENTRYPOINT_PATH,
        ],
        "python_flags": {
            "dont_write_bytecode": 1,
            "ignore_environment": 1,
            "isolated": 1,
            "no_site": 1,
            "no_user_site": 1,
        },
        "sys_path": [
            os.fspath(repository / "src"),
            *RUNTIME_SYS_PATH_ROOTS,
        ],
    }


def _implementation_repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "synthetic-run-repository"
    repository.mkdir(parents=True)
    _git(repository, "init", "--quiet", "--initial-branch=main")
    for relative in FROZEN_FILE_PATHS:
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT_ROOT / relative, destination)
    authority_destination = repository / "src/falsewake/experiment_002_run_authority.py"
    shutil.copyfile(
        PROJECT_ROOT / "src/falsewake/experiment_002_run_authority.py",
        authority_destination,
    )
    runner = repository / ENTRYPOINT_PATH
    runner.parent.mkdir(parents=True, exist_ok=True)
    runner.write_text("VALUE = 'synthetic committed source'\n", encoding="utf-8")
    (repository / ".gitignore").write_text("*.ignored\n", encoding="utf-8")
    _git(repository, "add", "--all")
    _git(repository, "commit", "--quiet", "-m", "synthetic implementation")
    implementation = _git(repository, "rev-parse", "HEAD").decode().strip()
    return repository, implementation


def _valid_repository(tmp_path: Path) -> RepositoryCase:
    repository, implementation = _implementation_repository(tmp_path)
    return _register_repository(repository, implementation)


def _register_repository(repository: Path, implementation: str) -> RepositoryCase:
    import_roots = ("src/falsewake",)
    source_paths = tuple(
        sorted(
            (
                ENTRYPOINT_PATH,
                "src/falsewake/causal_kws.py",
                "src/falsewake/experiment_002_run_authority.py",
            ),
            key=lambda value: value.encode("utf-8"),
        )
    )
    payload_byte_count, source_sha256 = _source_digest(repository, source_paths)
    document: dict[str, Any] = {
        "experiment": "002",
        "external_inputs": _external_inputs(),
        "frozen_bindings": authority._expected_frozen_bindings(),
        "implementation_commit": implementation,
        "invocation": _invocation(repository),
        "runtime": _runtime(),
        "schema_version": 1,
        "source_bundle": {
            "import_roots": list(import_roots),
            "paths": list(source_paths),
            "payload_byte_count": payload_byte_count,
            "sha256": source_sha256,
        },
    }
    run_path = repository / RUN_CONFIG_PATH
    run_path.write_bytes(_canonical(document))
    _git(repository, "add", RUN_CONFIG_PATH.as_posix())
    _git(repository, "commit", "--quiet", "-m", "register synthetic run")
    head = _git(repository, "rev-parse", "HEAD").decode().strip()
    return RepositoryCase(
        root=repository,
        implementation_commit=implementation,
        head_commit=head,
        document=document,
        source_paths=source_paths,
        source_payload_byte_count=payload_byte_count,
        source_sha256=source_sha256,
    )


def _amend_document(case: RepositoryCase, document: dict[str, Any]) -> None:
    (case.root / RUN_CONFIG_PATH).write_bytes(_canonical(document))
    _git(case.root, "add", RUN_CONFIG_PATH.as_posix())
    _git(case.root, "commit", "--quiet", "--amend", "--no-edit")


def _prepare_controlled_capability_lifecycle(
    case: RepositoryCase,
    monkeypatch: pytest.MonkeyPatch,
    *,
    patch_runtime_identity: bool = True,
) -> authority._RepositorySnapshot:
    snapshot = authority._verify_synthetic_repository_for_tests(case.root)
    monkeypatch.setattr(authority, "_CANONICAL_REPOSITORY_ROOT", case.root)
    monkeypatch.setattr(
        authority,
        "_ISSUED",
        weakref.WeakKeyDictionary(),
    )
    monkeypatch.setattr(
        authority,
        "_ISSUED_GUARDS",
        weakref.WeakKeyDictionary(),
    )
    monkeypatch.setattr(authority, "_FAILED", weakref.WeakSet())
    monkeypatch.setattr(authority, "_ISSUANCE_COMPLETE", False)

    def controlled_runtime_identity(
        root: Path,
        document: authority._RegistrationDocument,
    ) -> None:
        del root, document

    if patch_runtime_identity:
        monkeypatch.setattr(
            authority,
            "_require_registered_runtime_identity",
            controlled_runtime_identity,
        )
    return snapshot


@contextmanager
def _fixed_sealed_child_bundle_fd(bundle: bytes) -> Iterator[None]:
    saved_descriptor = -1
    original_inheritable = False
    writer = -1
    reader = -1
    try:
        try:
            original_inheritable = os.get_inheritable(7)
            saved_descriptor = fcntl.fcntl(7, fcntl.F_DUPFD_CLOEXEC, 64)
        except OSError as error:
            if error.errno != errno.EBADF:
                raise
            placeholder = os.open("/dev/null", os.O_RDONLY | os.O_CLOEXEC)
            if placeholder != 7:
                os.dup2(placeholder, 7, inheritable=False)
                os.close(placeholder)
        writer = os.memfd_create(
            "falsewake-exp002-child-bundle",
            os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING,
        )
        authority._write_descriptor_exactly(writer, bundle)
        os.fchmod(writer, 0o400)
        os.fsync(writer)
        fcntl.fcntl(
            writer,
            fcntl.F_ADD_SEALS,
            fcntl.F_SEAL_WRITE
            | fcntl.F_SEAL_GROW
            | fcntl.F_SEAL_SHRINK
            | fcntl.F_SEAL_SEAL,
        )
        reader = os.open(f"/proc/self/fd/{writer}", os.O_RDONLY | os.O_CLOEXEC)
        os.dup2(reader, 7, inheritable=False)
        if reader != 7:
            os.close(reader)
            reader = -1
        os.close(writer)
        writer = -1
        assert os.lseek(7, 0, os.SEEK_CUR) == 0
        yield
    finally:
        for descriptor in (reader, writer):
            if descriptor >= 0 and descriptor != 7:
                with suppress(OSError):
                    os.close(descriptor)
        with suppress(OSError):
            os.close(7)
        if saved_descriptor >= 0:
            os.dup2(
                saved_descriptor,
                7,
                inheritable=original_inheritable,
            )
            os.close(saved_descriptor)


def _replace_fixed_sealed_child_bundle_fd(
    bundle: bytes,
    *,
    name: str = "falsewake-exp002-child-bundle",
    mode: int = 0o400,
    seals: int | None = None,
) -> None:
    if seals is None:
        seals = (
            fcntl.F_SEAL_WRITE
            | fcntl.F_SEAL_GROW
            | fcntl.F_SEAL_SHRINK
            | fcntl.F_SEAL_SEAL
        )
    writer = os.memfd_create(name, os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    reader = -1
    try:
        authority._write_descriptor_exactly(writer, bundle)
        os.fchmod(writer, mode)
        os.fsync(writer)
        if seals:
            fcntl.fcntl(writer, fcntl.F_ADD_SEALS, seals)
        reader = os.open(f"/proc/self/fd/{writer}", os.O_RDONLY | os.O_CLOEXEC)
        os.dup2(reader, 7, inheritable=False)
    finally:
        with suppress(OSError):
            os.close(reader)
        os.close(writer)


def _prepare_sealed_child_harness(
    case: RepositoryCase,
    monkeypatch: pytest.MonkeyPatch,
) -> SealedChildHarness:
    snapshot = authority._verify_synthetic_repository_for_tests(case.root)
    monkeypatch.setattr(authority, "_CANONICAL_REPOSITORY_ROOT", case.root)
    parent_state = authority._state_from_snapshot(snapshot)
    bundle = authority._child_bundle_bytes_from_state(parent_state)
    frame = authority._parse_experiment_002_child_bundle(bundle)
    origin_prefix = f"falsewake-sealed://experiment-002/{frame.source_bundle_sha256}/"
    source_map = {
        f"{origin_prefix}{blob.path}": blob.payload for blob in frame.source_blobs
    }
    finder = FakeSealedSourceFinder(source_map)
    origin = f"{origin_prefix}{authority._AUTHORITY_SOURCE_PATH}"
    spec = importlib.machinery.ModuleSpec(
        authority.__name__,
        cast(Any, finder),
        origin=origin,
    )
    spec.has_location = True
    document = authority._parse_registration(frame.registration_bytes)
    invocation = document.invocation
    runtime = document.runtime
    flags = cast(dict[str, int], invocation["python_flags"])
    environment = cast(dict[str, str], invocation["environment"])
    versions = {
        "numpy": cast(str, runtime["numpy_version"]),
        "safetensors": cast(str, runtime["safetensors_version"]),
        "torch": cast(str, runtime["torch_version"]),
    }

    monkeypatch.setattr(authority, "_LOADED_SOURCE_ORIGIN_KIND", "sealed_child")
    monkeypatch.setattr(authority, "_LOADED_SOURCE_ORIGIN", origin)
    monkeypatch.setattr(authority, "_LOADED_SOURCE_LOADER", finder)
    monkeypatch.setattr(authority, "_LOADED_MODULE_SPEC", spec)
    monkeypatch.setattr(authority, "_LOADED_MODULE", authority)
    monkeypatch.setattr(
        authority,
        "_LOADED_SOURCE_GET_DATA_ROUTE",
        finder.get_data,
    )
    monkeypatch.setattr(
        authority,
        "_LOADED_FINDER_VERIFY_ROUTE",
        finder._verify_sealed_import_state,
    )
    authority_blob = next(
        blob
        for blob in frame.source_blobs
        if blob.path == authority._AUTHORITY_SOURCE_PATH
    )
    monkeypatch.setattr(
        authority,
        "_LOADED_SOURCE_SHA256",
        hashlib.sha256(authority_blob.payload).hexdigest(),
    )
    monkeypatch.setattr(authority, "__spec__", spec)
    monkeypatch.setattr(authority, "__loader__", finder)
    monkeypatch.setattr(authority, "__file__", origin)
    monkeypatch.setattr(authority, "__package__", "falsewake")
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
    monkeypatch.setattr(sys, "path", list(authority._TRUSTED_RUNTIME_SYS_PATHS))
    monkeypatch.setattr(authority, "_LOADED_META_PATH_OBJECT", sys.meta_path)
    monkeypatch.setattr(
        authority,
        "_LOADED_META_PATH_FRAME",
        tuple(cast(list[object], sys.meta_path)),
    )
    monkeypatch.setattr(authority, "_LOADED_SYS_PATH_OBJECT", sys.path)
    monkeypatch.setattr(
        authority,
        "_LOADED_SYS_PATH_FRAME",
        tuple(sys.path),
    )
    monkeypatch.setattr(sys, "argv", list(invocation["argv"]))
    monkeypatch.setattr(sys, "orig_argv", list(invocation["orig_argv"]))
    monkeypatch.setattr(sys, "executable", cast(str, runtime["python_executable"]))
    monkeypatch.setattr(sys, "flags", SimpleNamespace(**flags))
    monkeypatch.setattr(os, "environ", dict(environment))
    monkeypatch.setattr(platform, "python_version", lambda: runtime["python_version"])
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda distribution_name: versions[distribution_name],
    )
    monkeypatch.chdir(case.root)
    monkeypatch.setattr(authority, "_ISSUED", weakref.WeakKeyDictionary())
    monkeypatch.setattr(authority, "_ISSUED_GUARDS", weakref.WeakKeyDictionary())
    monkeypatch.setattr(authority, "_FAILED", weakref.WeakSet())
    monkeypatch.setattr(authority, "_ISSUANCE_COMPLETE", False)
    return SealedChildHarness(
        case=case,
        bundle=bundle,
        frame=frame,
        finder=finder,
        spec=spec,
        origin=origin,
    )


def test_source_is_strictly_standard_library_only() -> None:
    source_path = PROJECT_ROOT / "src/falsewake/experiment_002_run_authority.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".", 1)[0])
    assert imported_roots == {
        "__future__",
        "dataclasses",
        "fcntl",
        "hashlib",
        "importlib",
        "json",
        "os",
        "pathlib",
        "platform",
        "selectors",
        "shutil",
        "signal",
        "stat",
        "struct",
        "subprocess",
        "sys",
        "threading",
        "time",
        "typing",
        "weakref",
    }
    assert not {"falsewake", "numpy", "safetensors", "torch"}.intersection(
        imported_roots
    )


def test_actual_repository_is_closed_before_git_or_any_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_path = PROJECT_ROOT / RUN_CONFIG_PATH
    status_before = _git(PROJECT_ROOT, "status", "--porcelain=v1", "-z")
    files_before = _worktree_snapshot(PROJECT_ROOT)
    called = False

    def forbidden(*args: object, **kwargs: object) -> bytes:
        nonlocal called
        del args, kwargs
        called = True
        raise AssertionError("Git must not run before the absent config rejects")

    monkeypatch.setattr(authority, "_git", forbidden)
    assert not run_path.exists()
    with pytest.raises(
        authority.Experiment002RunAuthorityError,
        match="absent or inaccessible",
    ):
        authority.verify_and_issue_experiment_002_run_registration()
    assert not called
    assert not run_path.exists()
    assert _git(PROJECT_ROOT, "status", "--porcelain=v1", "-z") == status_before
    assert _worktree_snapshot(PROJECT_ROOT) == files_before


def test_public_issuer_has_no_repository_or_digest_override() -> None:
    signature = inspect.signature(
        authority.verify_and_issue_experiment_002_run_registration
    )
    assert tuple(signature.parameters) == ()


def test_sealed_child_issuer_has_no_override_and_parent_issuer_rejects_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _valid_repository(tmp_path)
    _prepare_sealed_child_harness(case, monkeypatch)
    signature = inspect.signature(
        authority._verify_and_issue_experiment_002_sealed_child_registration
    )
    assert tuple(signature.parameters) == ()
    git_called = False

    def forbidden_git(*args: object, **kwargs: object) -> bytes:
        nonlocal git_called
        del args, kwargs
        git_called = True
        raise AssertionError("Git must not run from a sealed-child issuer")

    monkeypatch.setattr(authority, "_git", forbidden_git)
    with pytest.raises(
        authority.Experiment002RunAuthorityError,
        match="canonical file origin",
    ):
        authority.verify_and_issue_experiment_002_run_registration()
    assert not git_called


def test_sealed_child_source_capture_uses_stable_loader_routes_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _valid_repository(tmp_path)
    harness = _prepare_sealed_child_harness(case, monkeypatch)

    def forbidden_path_read(_path: Path) -> bytes:
        raise AssertionError("sealed source capture must not read __file__")

    monkeypatch.setattr(Path, "read_bytes", forbidden_path_read)
    captured = authority._capture_loaded_authority_source()
    (
        kind,
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
        source_sha256,
    ) = captured
    assert kind == authority._SEALED_CHILD_ORIGIN_KIND
    assert executing_file == case.root / authority._AUTHORITY_SOURCE_PATH
    assert origin == harness.origin
    assert loader is harness.finder
    assert spec is harness.spec
    assert module is authority
    assert get_data_route is harness.finder.get_data
    assert verifier_route is harness.finder._verify_sealed_import_state
    assert meta_path_object is sys.meta_path
    assert meta_path_frame == tuple(sys.meta_path)
    assert sys_path_object is sys.path
    assert sys_path_frame == authority._TRUSTED_RUNTIME_SYS_PATHS
    assert (
        source_sha256
        == hashlib.sha256(harness.finder.source_map[harness.origin]).hexdigest()
    )
    assert harness.finder.verification_count == 2
    with pytest.raises(FileNotFoundError):
        harness.finder.get_data(f"{harness.origin}.unregistered")


def test_parent_verification_rejects_stable_loaded_source_digest_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _valid_repository(tmp_path)
    monkeypatch.setattr(authority, "_LOADED_SOURCE_SHA256", "0" * 64)
    with pytest.raises(
        authority.Experiment002RunAuthorityError,
        match="executing run-authority source differs from its bound blob",
    ):
        authority._verify_synthetic_repository_for_tests(case.root)


def test_sealed_child_issuer_and_reverification_never_use_git_or_popen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _valid_repository(tmp_path)
    harness = _prepare_sealed_child_harness(case, monkeypatch)

    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("sealed-child authority must stay process-local")

    monkeypatch.setattr(authority, "_git", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    with _fixed_sealed_child_bundle_fd(harness.bundle):
        offset_before = os.lseek(7, 0, os.SEEK_CUR)
        capability = (
            authority._verify_and_issue_experiment_002_sealed_child_registration()
        )
        assert os.lseek(7, 0, os.SEEK_CUR) == offset_before == 0
        assert capability.head_commit == case.head_commit
        assert capability.implementation_commit == case.implementation_commit
        assert (
            capability.registration_sha256
            == hashlib.sha256(harness.frame.registration_bytes).hexdigest()
        )
        assert capability.source_bundle_sha256 == case.source_sha256

        state = authority._ISSUED[capability]
        guard = authority._ISSUED_GUARDS[capability]
        assert state.source_paths == case.source_paths
        assert state.origin_binding.kind == authority._SEALED_CHILD_ORIGIN_KIND
        assert state.origin_binding is not guard.origin_binding
        assert authority._origin_bindings_match(
            state.origin_binding,
            guard.origin_binding,
        )
        assert state.origin_binding.bundle_descriptor == 7
        assert state.origin_binding.bundle_offset == 0
        assert state.origin_binding.bundle_proc_target == (
            "/memfd:falsewake-exp002-child-bundle (deleted)"
        )
        assert state.origin_binding.source_loader is harness.finder
        assert state.origin_binding.source_finder is harness.finder
        assert state.origin_binding.loader_get_data_route is harness.finder.get_data
        assert (
            state.origin_binding.finder_verify_route
            is harness.finder._verify_sealed_import_state
        )
        assert state.origin_binding.module_object is authority
        assert state.origin_binding.module_spec is harness.spec
        assert state.origin_binding.module_origin == harness.origin
        assert state.origin_binding.runtime_sys_path == RUNTIME_SYS_PATH_ROOTS

        for module_name in ("numpy", "safetensors", "torch"):
            monkeypatch.setitem(sys.modules, module_name, ModuleType(module_name))
        authority.reverify_verified_run_registration(capability)
        assert os.lseek(7, 0, os.SEEK_CUR) == 0
        assert capability not in authority._FAILED

        with pytest.raises(
            authority.Experiment002RunAuthorityError,
            match="parent-origin authority",
        ):
            authority._create_sealed_experiment_002_child_bundle_fd(capability)
        assert capability not in authority._FAILED
        with pytest.raises(
            authority.Experiment002RunAuthorityError,
            match="already issued",
        ):
            authority._verify_and_issue_experiment_002_sealed_child_registration()


@pytest.mark.parametrize(
    "tamper",
    [
        "descriptor_closed",
        "descriptor_offset",
        "descriptor_replaced",
        "descriptor_mode",
        "finder_failed",
        "get_data_route",
        "verifier_route",
        "source_payload",
        "module_origin",
        "module_origin_string_subclass",
        "module_file",
        "module_file_string_subclass",
        "module_package_string_subclass",
        "module_loader",
        "module_spec",
        "spec_name_string_subclass",
        "meta_path_identity",
        "sys_modules_key_subclass",
        "sys_path_identity",
        "sys_path_string_subclass",
        "argv_string_subclass",
        "orig_argv_string_subclass",
        "environment_string_subclass",
        "python_version_string_subclass",
        "python_executable_string_subclass",
        "package_version_string_subclass",
        "flag_integer_subclass",
        "parent_process",
    ],
)
def test_sealed_child_reverification_detects_tamper_and_permanently_poisons(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    case = _valid_repository(tmp_path)
    harness = _prepare_sealed_child_harness(case, monkeypatch)
    with _fixed_sealed_child_bundle_fd(harness.bundle):
        capability = (
            authority._verify_and_issue_experiment_002_sealed_child_registration()
        )
        if tamper == "descriptor_closed":
            os.close(7)
        elif tamper == "descriptor_offset":
            os.lseek(7, 1, os.SEEK_SET)
        elif tamper == "descriptor_replaced":
            _replace_fixed_sealed_child_bundle_fd(harness.bundle)
        elif tamper == "descriptor_mode":
            os.fchmod(7, 0o600)
        elif tamper == "finder_failed":
            harness.finder.valid = False
        elif tamper == "get_data_route":
            original_get_data = harness.finder.get_data

            def replacement_get_data(origin: str) -> bytes:
                return original_get_data(origin)

            harness.finder.get_data = replacement_get_data
        elif tamper == "verifier_route":
            original_verifier = harness.finder._verify_sealed_import_state

            def replacement_verifier() -> None:
                original_verifier()

            harness.finder._verify_sealed_import_state = replacement_verifier
        elif tamper == "source_payload":
            harness.finder.source_map[harness.origin] += b"\n"
        elif tamper == "module_origin":
            harness.spec.origin = f"{harness.origin}.changed"
        elif tamper == "module_origin_string_subclass":
            harness.spec.origin = EqualStringSubclass(harness.origin)
        elif tamper == "module_file":
            monkeypatch.setattr(authority, "__file__", f"{harness.origin}.changed")
        elif tamper == "module_file_string_subclass":
            monkeypatch.setattr(
                authority,
                "__file__",
                EqualStringSubclass(harness.origin),
            )
        elif tamper == "module_package_string_subclass":
            monkeypatch.setattr(
                authority,
                "__package__",
                EqualStringSubclass("falsewake"),
            )
        elif tamper == "module_loader":
            monkeypatch.setattr(authority, "__loader__", object())
        elif tamper == "module_spec":
            monkeypatch.setattr(authority, "__spec__", None)
        elif tamper == "spec_name_string_subclass":
            harness.spec.name = EqualStringSubclass(authority.__name__)
        elif tamper == "meta_path_identity":
            monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
        elif tamper == "sys_modules_key_subclass":
            spoofed_name = EqualStringSubclass("falsewake.spoofed")
            monkeypatch.setitem(sys.modules, spoofed_name, ModuleType(spoofed_name))
        elif tamper == "sys_path_identity":
            monkeypatch.setattr(sys, "path", list(sys.path))
        elif tamper == "sys_path_string_subclass":
            sys.path[0] = EqualStringSubclass(sys.path[0])
        elif tamper == "argv_string_subclass":
            sys.argv[0] = EqualStringSubclass(sys.argv[0])
        elif tamper == "orig_argv_string_subclass":
            sys.orig_argv[0] = EqualStringSubclass(sys.orig_argv[0])
        elif tamper == "environment_string_subclass":
            environment_key = next(iter(os.environ))
            os.environ[environment_key] = EqualStringSubclass(
                os.environ[environment_key]
            )
        elif tamper == "python_version_string_subclass":
            runtime = cast(dict[str, Any], harness.case.document["runtime"])
            expected_python_version = cast(str, runtime["python_version"])
            monkeypatch.setattr(
                platform,
                "python_version",
                lambda: EqualStringSubclass(expected_python_version),
            )
        elif tamper == "python_executable_string_subclass":
            monkeypatch.setattr(
                sys,
                "executable",
                EqualStringSubclass(sys.executable),
            )
        elif tamper == "package_version_string_subclass":
            versions = cast(dict[str, Any], harness.case.document["runtime"])
            package_versions = {
                "numpy": cast(str, versions["numpy_version"]),
                "safetensors": cast(str, versions["safetensors_version"]),
                "torch": cast(str, versions["torch_version"]),
            }
            monkeypatch.setattr(
                importlib.metadata,
                "version",
                lambda name: EqualStringSubclass(package_versions[name]),
            )
        elif tamper == "flag_integer_subclass":
            mutated_flags = vars(sys.flags).copy()
            mutated_flags["isolated"] = EqualIntegerSubclass(
                cast(int, mutated_flags["isolated"])
            )
            monkeypatch.setattr(sys, "flags", SimpleNamespace(**mutated_flags))
        else:
            parent_process_id = os.getppid()
            monkeypatch.setattr(os, "getppid", lambda: parent_process_id + 1)

        with pytest.raises(authority.Experiment002RunAuthorityError):
            authority.reverify_verified_run_registration(capability)
        assert capability in authority._FAILED
        with pytest.raises(
            authority.Experiment002RunAuthorityError,
            match="not issued",
        ):
            _ = capability.head_commit


@pytest.mark.parametrize("tamper_target", ["state", "guard"])
def test_sealed_child_origin_frame_tamper_permanently_poisons_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper_target: str,
) -> None:
    case = _valid_repository(tmp_path)
    harness = _prepare_sealed_child_harness(case, monkeypatch)
    with _fixed_sealed_child_bundle_fd(harness.bundle):
        capability = (
            authority._verify_and_issue_experiment_002_sealed_child_registration()
        )
        original_state = authority._ISSUED[capability]
        original_guard = authority._ISSUED_GUARDS[capability]
        if tamper_target == "state":
            authority._ISSUED[capability] = dataclass_replace(
                original_state,
                origin_binding=dataclass_replace(
                    original_state.origin_binding,
                    bundle_offset=1,
                ),
            )
        else:
            authority._ISSUED_GUARDS[capability] = dataclass_replace(
                original_guard,
                origin_binding=dataclass_replace(
                    original_guard.origin_binding,
                    bundle_offset=1,
                ),
            )
        with pytest.raises(
            authority.Experiment002RunAuthorityError,
            match="not issued",
        ):
            _ = capability.head_commit
        authority._ISSUED[capability] = original_state
        authority._ISSUED_GUARDS[capability] = original_guard
        with pytest.raises(
            authority.Experiment002RunAuthorityError,
            match="not issued",
        ):
            authority.verify_verified_run_registration(capability)


@pytest.mark.parametrize(
    "mutation",
    ["wrong_memfd_name", "missing_seals", "inheritable"],
)
def test_fixed_sealed_child_descriptor_rejects_nonexact_kernel_identity(
    mutation: str,
) -> None:
    payload = b"sealed-child-descriptor-probe"
    with _fixed_sealed_child_bundle_fd(payload):
        if mutation == "wrong_memfd_name":
            _replace_fixed_sealed_child_bundle_fd(payload, name="wrong-name")
        elif mutation == "missing_seals":
            _replace_fixed_sealed_child_bundle_fd(payload, seals=0)
        else:
            os.set_inheritable(7, True)
        with pytest.raises(authority.Experiment002RunAuthorityError):
            authority._read_fixed_sealed_child_bundle()


def test_sealed_child_issuance_is_concurrent_one_shot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _valid_repository(tmp_path)
    harness = _prepare_sealed_child_harness(case, monkeypatch)
    barrier = threading.Barrier(3)

    def attempt_issue() -> authority.VerifiedRunRegistration:
        barrier.wait()
        return authority._verify_and_issue_experiment_002_sealed_child_registration()

    with _fixed_sealed_child_bundle_fd(harness.bundle):
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(attempt_issue) for _ in range(2)]
            barrier.wait()
            outcomes: list[authority.VerifiedRunRegistration | BaseException] = []
            for future in futures:
                try:
                    outcomes.append(future.result())
                except authority.Experiment002RunAuthorityError as error:
                    outcomes.append(error)
        issued = [
            outcome
            for outcome in outcomes
            if isinstance(outcome, authority.VerifiedRunRegistration)
        ]
        rejected = [
            outcome
            for outcome in outcomes
            if isinstance(outcome, authority.Experiment002RunAuthorityError)
        ]
        assert len(issued) == 1
        assert len(rejected) == 1
        authority.reverify_verified_run_registration(issued[0])


def test_sealed_child_capability_is_process_local_across_fork(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not hasattr(os, "fork"):
        pytest.skip("fork is required for the process-local capability test")
    case = _valid_repository(tmp_path)
    harness = _prepare_sealed_child_harness(case, monkeypatch)
    with _fixed_sealed_child_bundle_fd(harness.bundle):
        capability = (
            authority._verify_and_issue_experiment_002_sealed_child_registration()
        )
        read_descriptor, write_descriptor = os.pipe()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"This process .* is multi-threaded, use of fork\(\)",
                category=DeprecationWarning,
            )
            child_pid = os.fork()
        if child_pid == 0:
            os.close(read_descriptor)
            try:
                _ = capability.head_commit
            except authority.Experiment002RunAuthorityError:
                os.write(write_descriptor, b"rejected")
                os.close(write_descriptor)
                os._exit(0)
            os.close(write_descriptor)
            os._exit(2)

        os.close(write_descriptor)
        observed = os.read(read_descriptor, 64)
        os.close(read_descriptor)
        waited_pid, status = os.waitpid(child_pid, 0)
        assert waited_pid == child_pid
        assert os.WIFEXITED(status)
        assert os.WEXITSTATUS(status) == 0
        assert observed == b"rejected"
        authority.reverify_verified_run_registration(capability)
        assert capability.head_commit == case.head_commit


def test_public_authority_rejects_a_physical_foreign_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copied_source = (
        tmp_path / "foreign-repository/src/falsewake/experiment_002_run_authority.py"
    )
    copied_source.parent.mkdir(parents=True)
    shutil.copyfile(
        PROJECT_ROOT / "src/falsewake/experiment_002_run_authority.py",
        copied_source,
    )
    monkeypatch.setattr(authority, "_EXECUTING_FILE", copied_source)

    with pytest.raises(
        authority.Experiment002RunAuthorityError,
        match="outside the canonical",
    ):
        authority._repository_root()


def test_isolated_committed_source_success_is_explicitly_noncapability(
    tmp_path: Path,
) -> None:
    case = _valid_repository(tmp_path)
    issued_before = len(authority._ISSUED)
    status_before = _git(case.root, "status", "--porcelain=v1", "-z")

    snapshot = authority._verify_synthetic_repository_for_tests(case.root)

    assert type(snapshot) is authority._RepositorySnapshot
    assert snapshot.head_commit == case.head_commit
    assert snapshot.implementation_commit == case.implementation_commit
    assert snapshot.source_paths == case.source_paths
    assert snapshot.source_bundle_sha256 == case.source_sha256
    assert snapshot.registration_bytes == (case.root / RUN_CONFIG_PATH).read_bytes()
    assert snapshot.source_bundle_payload == _source_payload(
        case.root, case.source_paths
    )
    assert tuple(blob.path for blob in snapshot.frozen_blobs) == FROZEN_FILE_PATHS
    assert tuple(blob.payload for blob in snapshot.frozen_blobs) == tuple(
        (case.root / path).read_bytes() for path in FROZEN_FILE_PATHS
    )
    assert len(authority._ISSUED) == issued_before
    assert not isinstance(snapshot, authority.VerifiedRunRegistration)
    assert _git(case.root, "status", "--porcelain=v1", "-z") == status_before == b""


def test_source_bundle_framing_matches_an_independent_oracle(tmp_path: Path) -> None:
    case = _valid_repository(tmp_path)
    sources = tuple(
        (path, (case.root / path).read_bytes()) for path in case.source_paths
    )
    byte_count, digest = authority._source_bundle_digest(sources)
    assert byte_count == case.source_payload_byte_count
    assert digest == case.source_sha256
    assert authority._source_bundle_payload(sources) == _source_payload(
        case.root, case.source_paths
    )


def test_child_bundle_api_is_positional_only_and_exactly_sealed_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signature = inspect.signature(
        authority._create_sealed_experiment_002_child_bundle_fd
    )
    assert tuple(signature.parameters) == ("registration",)
    parameter = signature.parameters["registration"]
    assert parameter.kind is inspect.Parameter.POSITIONAL_ONLY
    assert parameter.default is inspect.Parameter.empty

    case = _valid_repository(tmp_path)
    snapshot = _prepare_controlled_capability_lifecycle(case, monkeypatch)
    capability = authority._issue_controlled_snapshot_for_tests(snapshot)
    original_reverify = authority.reverify_verified_run_registration
    reverify_calls = 0

    def counted_reverify(
        registration: authority.VerifiedRunRegistration,
    ) -> None:
        nonlocal reverify_calls
        reverify_calls += 1
        original_reverify(registration)

    monkeypatch.setattr(
        authority,
        "reverify_verified_run_registration",
        counted_reverify,
    )
    descriptor = authority._create_sealed_experiment_002_child_bundle_fd(capability)
    try:
        expected = _bundle_oracle(case)
        metadata = os.fstat(descriptor)
        observed = os.pread(descriptor, metadata.st_size, 0)
        assert observed == expected
        header, registration, source, frozen = _independent_bundle_sections(observed)
        assert header[:6] == (
            BUNDLE_MAGIC,
            BUNDLE_VERSION,
            case.head_commit.encode("ascii"),
            case.implementation_commit.encode("ascii"),
            hashlib.sha256(registration).hexdigest().encode("ascii"),
            hashlib.sha256(SOURCE_DOMAIN + source).hexdigest().encode("ascii"),
        )
        assert registration == (case.root / RUN_CONFIG_PATH).read_bytes()
        assert source == _source_payload(case.root, case.source_paths)
        assert frozen == _frozen_payload(case.root)
        assert stat.S_ISREG(metadata.st_mode)
        assert stat.S_IMODE(metadata.st_mode) == 0o400
        assert metadata.st_nlink == 0
        assert fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE == os.O_RDONLY
        assert fcntl.fcntl(descriptor, fcntl.F_GETFD) & fcntl.FD_CLOEXEC
        expected_seals = (
            fcntl.F_SEAL_WRITE
            | fcntl.F_SEAL_GROW
            | fcntl.F_SEAL_SHRINK
            | fcntl.F_SEAL_SEAL
        )
        assert fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) == expected_seals
        assert not os.get_inheritable(descriptor)
        with pytest.raises(OSError) as write_error:
            os.write(descriptor, b"x")
        assert write_error.value.errno == errno.EBADF
        with pytest.raises(OSError) as truncate_error:
            os.ftruncate(descriptor, 0)
        assert truncate_error.value.errno in {errno.EBADF, errno.EINVAL}

        with pytest.raises(OSError) as writable_open_error:
            os.open(f"/proc/self/fd/{descriptor}", os.O_RDWR | os.O_CLOEXEC)
        assert writable_open_error.value.errno in {errno.EACCES, errno.EPERM}
    finally:
        os.close(descriptor)
    assert reverify_calls == 2

    state = authority._ISSUED[capability]
    guard = authority._ISSUED_GUARDS[capability]
    assert state.registration_bytes == guard.registration_bytes
    assert state.source_bundle_payload == guard.source_bundle_payload
    assert state.frozen_blobs == guard.frozen_blobs
    assert state.frozen_blobs is not guard.frozen_blobs
    assert all(
        state_blob is not guard_blob
        for state_blob, guard_blob in zip(
            state.frozen_blobs, guard.frozen_blobs, strict=True
        )
    )


def test_child_bundle_mode_is_verified_at_return_and_rechecked_after_chmod(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _valid_repository(tmp_path)
    snapshot = _prepare_controlled_capability_lifecycle(case, monkeypatch)
    capability = authority._issue_controlled_snapshot_for_tests(snapshot)
    descriptor = authority._create_sealed_experiment_002_child_bundle_fd(capability)
    try:
        expected = _bundle_oracle(case)
        state = authority._ISSUED[capability]
        _, required_seals = authority._memfd_requirements()
        assert stat.S_IMODE(os.fstat(descriptor).st_mode) == 0o400
        os.fchmod(descriptor, 0o700)
        assert stat.S_IMODE(os.fstat(descriptor).st_mode) == 0o700
        assert fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) == required_seals
        with pytest.raises(
            authority.Experiment002RunAuthorityError,
            match="stat frame is invalid",
        ):
            authority._verify_sealed_bundle_descriptor(
                descriptor,
                expected_bytes=expected,
                state=state,
                required_seals=required_seals,
                expected_access_mode=os.O_RDONLY,
            )
    finally:
        os.close(descriptor)


@pytest.mark.parametrize(
    "mutation",
    [
        "magic",
        "version",
        "implementation",
        "registration_sha256",
        "source_sha256",
        "registration_payload",
        "source_payload",
        "frozen_payload",
        "truncate",
        "trailing",
        "oversize_registration",
        "oversize_source",
        "oversize_frozen",
        "oversize_total",
    ],
)
def test_child_bundle_parser_rejects_corruption_and_nonexact_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    case = _valid_repository(tmp_path)
    _prepare_controlled_capability_lifecycle(case, monkeypatch)
    original = _bundle_oracle(case)
    header, registration, source, _frozen = _independent_bundle_sections(original)
    registration_offset = BUNDLE_HEADER.size
    source_offset = registration_offset + len(registration)
    frozen_offset = source_offset + len(source)
    mutated = bytearray(original)
    if mutation == "magic":
        mutated[0] ^= 1
    elif mutation == "version":
        struct.pack_into("<I", mutated, 8, BUNDLE_VERSION + 1)
    elif mutation == "implementation":
        implementation_offset = 8 + 4 + 40
        mutated[implementation_offset] = (
            ord("0") if mutated[implementation_offset] != ord("0") else ord("1")
        )
    elif mutation == "registration_sha256":
        registration_sha_offset = 8 + 4 + 40 + 40
        mutated[registration_sha_offset] = (
            ord("0") if mutated[registration_sha_offset] != ord("0") else ord("1")
        )
    elif mutation == "source_sha256":
        source_sha_offset = 8 + 4 + 40 + 40 + 64
        mutated[source_sha_offset] = (
            ord("0") if mutated[source_sha_offset] != ord("0") else ord("1")
        )
    elif mutation == "registration_payload":
        mutated[registration_offset] ^= 1
    elif mutation == "source_payload":
        mutated[source_offset + len(source) - 1] ^= 1
    elif mutation == "frozen_payload":
        mutated[-1] ^= 1
    elif mutation == "truncate":
        mutated = mutated[:-1]
    elif mutation == "trailing":
        mutated.extend(b"x")
    elif mutation == "oversize_registration":
        registration_size_offset = struct.calcsize("<8sI40s40s64s64s")
        struct.pack_into("<I", mutated, registration_size_offset, (1 << 20) + 1)
    elif mutation == "oversize_source":
        source_size_offset = struct.calcsize("<8sI40s40s64s64sI")
        struct.pack_into("<Q", mutated, source_size_offset, (256 << 20) + 1)
    elif mutation == "oversize_frozen":
        frozen_size_offset = struct.calcsize("<8sI40s40s64s64sIQ")
        struct.pack_into("<Q", mutated, frozen_size_offset, (256 << 20) + 1)
    else:
        registration_size_offset = struct.calcsize("<8sI40s40s64s64s")
        source_size_offset = struct.calcsize("<8sI40s40s64s64sI")
        frozen_size_offset = struct.calcsize("<8sI40s40s64s64sIQ")
        struct.pack_into("<I", mutated, registration_size_offset, 1)
        struct.pack_into("<Q", mutated, source_size_offset, 128 << 20)
        struct.pack_into("<Q", mutated, frozen_size_offset, 128 << 20)
    assert cast(bytes, header[0]) == BUNDLE_MAGIC
    assert frozen_offset < len(original)
    with pytest.raises(authority.Experiment002RunAuthorityError):
        authority._parse_experiment_002_child_bundle(bytes(mutated))


def test_child_bundle_exact_total_limit_boundary_and_overflow_oracle() -> None:
    exact_limit = 256 << 20
    independent_header_size = struct.calcsize("<8sI40s40s64s64sIQQ")
    final_component_at_boundary = exact_limit - independent_header_size - 2
    assert exact_limit == authority._MAX_CHILD_BUNDLE_BYTES
    assert independent_header_size == authority._CHILD_BUNDLE_HEADER.size
    assert (
        authority._require_child_bundle_total_byte_count(
            1,
            1,
            final_component_at_boundary,
        )
        == exact_limit
    )
    with pytest.raises(
        authority.Experiment002RunAuthorityError,
        match="exact total byte limit",
    ):
        authority._require_child_bundle_total_byte_count(
            1,
            1,
            final_component_at_boundary + 1,
        )


def test_child_bundle_total_gate_runs_before_capability_is_recorded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _valid_repository(tmp_path)
    snapshot = _prepare_controlled_capability_lifecycle(case, monkeypatch)

    def reject_total(
        registration_byte_count: int,
        source_payload_byte_count: int,
        frozen_payload_byte_count: int,
    ) -> int:
        del (
            registration_byte_count,
            source_payload_byte_count,
            frozen_payload_byte_count,
        )
        raise authority.Experiment002RunAuthorityError(
            "injected exact total byte limit rejection"
        )

    monkeypatch.setattr(
        authority,
        "_require_child_bundle_total_byte_count",
        reject_total,
    )
    with pytest.raises(
        authority.Experiment002RunAuthorityError,
        match="exact total byte limit rejection",
    ):
        authority._issue_controlled_snapshot_for_tests(snapshot)
    assert not authority._ISSUANCE_COMPLETE
    assert len(authority._ISSUED) == 0
    assert len(authority._ISSUED_GUARDS) == 0


def test_child_bundle_head_is_checked_against_immutable_authority_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _valid_repository(tmp_path)
    snapshot = _prepare_controlled_capability_lifecycle(case, monkeypatch)
    capability = authority._issue_controlled_snapshot_for_tests(snapshot)
    mutated = bytearray(_bundle_oracle(case))
    head_offset = 8 + 4
    mutated[head_offset] = ord("0") if mutated[head_offset] != ord("0") else ord("1")
    frame = authority._parse_experiment_002_child_bundle(bytes(mutated))
    assert frame.head_commit != case.head_commit
    with pytest.raises(
        authority.Experiment002RunAuthorityError,
        match="does not exactly match",
    ):
        authority._require_child_bundle_matches_state(
            frame, authority._ISSUED[capability]
        )


@pytest.mark.parametrize("bundle_kind", ["source", "frozen"])
@pytest.mark.parametrize("mutation", ["unsorted", "duplicate", "oversize"])
def test_blob_frame_parsers_reject_order_duplicates_and_oversize(
    tmp_path: Path,
    bundle_kind: str,
    mutation: str,
) -> None:
    case = _valid_repository(tmp_path)
    if bundle_kind == "source":
        blobs = tuple(
            (path, (case.root / path).read_bytes()) for path in case.source_paths
        )
        parser = authority._parse_source_bundle_payload
        maximum_file_bytes = 64 << 20
    else:
        blobs = tuple(
            (path, (case.root / path).read_bytes()) for path in FROZEN_FILE_PATHS
        )
        parser = authority._parse_frozen_blob_payload
        maximum_file_bytes = 64 << 20
    if mutation == "unsorted":
        malformed = _blob_payload(tuple(reversed(blobs)))
    elif mutation == "duplicate":
        malformed = _blob_payload((blobs[0], blobs[0], *blobs[2:]))
    else:
        path_bytes = blobs[0][0].encode("utf-8")
        malformed = b"".join(
            (
                struct.pack("<I", 1),
                struct.pack("<I", len(path_bytes)),
                path_bytes,
                struct.pack("<Q", maximum_file_bytes + 1),
            )
        )
    with pytest.raises(authority.Experiment002RunAuthorityError):
        parser(malformed)


@pytest.mark.parametrize(
    "failure_stage",
    ["write", "seal", "reopen", "post_reverify"],
)
def test_child_bundle_failure_paths_close_every_owned_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    case = _valid_repository(tmp_path)
    snapshot = _prepare_controlled_capability_lifecycle(case, monkeypatch)
    capability = authority._issue_controlled_snapshot_for_tests(snapshot)
    descriptors_before = _open_descriptors()
    if failure_stage == "write":

        def injected_write(descriptor: int, payload: object) -> int:
            del descriptor, payload
            raise OSError(errno.EIO, "injected memfd write failure")

        monkeypatch.setattr(os, "write", injected_write)
        expected_message = "write failed"
    elif failure_stage == "seal":
        original_fcntl = fcntl.fcntl

        def injected_fcntl(descriptor: int, command: int, argument: int = 0) -> int:
            if command == fcntl.F_ADD_SEALS:
                raise OSError(errno.EIO, "injected sealing failure")
            return original_fcntl(descriptor, command, argument)

        monkeypatch.setattr(fcntl, "fcntl", injected_fcntl)
        expected_message = "finalized and sealed"
    elif failure_stage == "reopen":
        original_open = os.open

        def injected_open(
            path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
            flags: int,
            mode: int = 0o777,
            *,
            dir_fd: int | None = None,
        ) -> int:
            if type(path) is str and path.startswith("/proc/self/fd/"):
                raise OSError(errno.EIO, "injected read-only reopen failure")
            if dir_fd is None:
                return original_open(path, flags, mode)
            return original_open(path, flags, mode, dir_fd=dir_fd)

        monkeypatch.setattr(os, "open", injected_open)
        expected_message = "reopened read-only"
    else:
        original_reverify = authority.reverify_verified_run_registration
        calls = 0

        def injected_reverify(
            registration: authority.VerifiedRunRegistration,
        ) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise authority.Experiment002RunAuthorityError(
                    "injected post-creation reverification failure"
                )
            original_reverify(registration)

        monkeypatch.setattr(
            authority,
            "reverify_verified_run_registration",
            injected_reverify,
        )
        expected_message = "post-creation reverification"
    with pytest.raises(
        authority.Experiment002RunAuthorityError,
        match=expected_message,
    ):
        authority._create_sealed_experiment_002_child_bundle_fd(capability)
    assert _open_descriptors() == descriptors_before


def test_child_bundle_rejects_forged_and_stale_capabilities_without_fd_leaks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forged = object.__new__(authority.VerifiedRunRegistration)
    descriptors_before = _open_descriptors()
    with pytest.raises(authority.Experiment002RunAuthorityError, match="not issued"):
        authority._create_sealed_experiment_002_child_bundle_fd(forged)
    assert _open_descriptors() == descriptors_before

    case = _valid_repository(tmp_path)
    snapshot = _prepare_controlled_capability_lifecycle(case, monkeypatch)
    capability = authority._issue_controlled_snapshot_for_tests(snapshot)
    entrypoint = case.root / ENTRYPOINT_PATH
    original = entrypoint.read_bytes()
    entrypoint.write_bytes(b"STALE_BEFORE_BUNDLE = True\n")
    try:
        with pytest.raises(authority.Experiment002RunAuthorityError):
            authority._create_sealed_experiment_002_child_bundle_fd(capability)
    finally:
        entrypoint.write_bytes(original)
    assert _open_descriptors() == descriptors_before
    with pytest.raises(authority.Experiment002RunAuthorityError, match="not issued"):
        authority._create_sealed_experiment_002_child_bundle_fd(capability)
    assert _open_descriptors() == descriptors_before


def test_concurrent_child_bundle_creation_returns_distinct_identical_memfds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _valid_repository(tmp_path)
    snapshot = _prepare_controlled_capability_lifecycle(case, monkeypatch)
    capability = authority._issue_controlled_snapshot_for_tests(snapshot)
    barrier = threading.Barrier(3)

    def create_bundle() -> int:
        barrier.wait()
        return authority._create_sealed_experiment_002_child_bundle_fd(capability)

    descriptors: list[int] = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(create_bundle) for _ in range(2)]
        barrier.wait()
        descriptors = [future.result() for future in futures]
    try:
        assert descriptors[0] != descriptors[1]
        metadata = tuple(os.fstat(descriptor) for descriptor in descriptors)
        assert (metadata[0].st_dev, metadata[0].st_ino) != (
            metadata[1].st_dev,
            metadata[1].st_ino,
        )
        payloads = tuple(
            os.pread(descriptor, item.st_size, 0)
            for descriptor, item in zip(descriptors, metadata, strict=True)
        )
        assert payloads == (_bundle_oracle(case),) * 2
    finally:
        for descriptor in descriptors:
            os.close(descriptor)


@pytest.mark.parametrize("tamper_target", ["state", "guard"])
@pytest.mark.parametrize(
    "material",
    ["registration_bytes", "source_bundle_payload", "frozen_blobs"],
)
def test_controlled_capability_poisoned_by_retained_material_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper_target: str,
    material: str,
) -> None:
    case = _valid_repository(tmp_path)
    snapshot = _prepare_controlled_capability_lifecycle(case, monkeypatch)
    capability = authority._issue_controlled_snapshot_for_tests(snapshot)
    original_state = authority._ISSUED[capability]
    original_guard = authority._ISSUED_GUARDS[capability]
    original = original_state if tamper_target == "state" else original_guard
    if material == "registration_bytes":
        mutated = dataclass_replace(
            original,
            registration_bytes=original.registration_bytes + b" ",
        )
    elif material == "source_bundle_payload":
        mutated = dataclass_replace(
            original,
            source_bundle_payload=original.source_bundle_payload[:-1] + b"!",
        )
    else:
        first = original.frozen_blobs[0]
        mutated_blobs = (
            authority._FrozenBlob(path=first.path, payload=first.payload + b"!"),
            *original.frozen_blobs[1:],
        )
        mutated = dataclass_replace(original, frozen_blobs=mutated_blobs)
    if tamper_target == "state":
        authority._ISSUED[capability] = cast(authority._VerifiedState, mutated)
    else:
        authority._ISSUED_GUARDS[capability] = cast(authority._IssuedGuard, mutated)
    with pytest.raises(authority.Experiment002RunAuthorityError, match="not issued"):
        authority._verified_state(capability)

    authority._ISSUED[capability] = original_state
    authority._ISSUED_GUARDS[capability] = original_guard
    with pytest.raises(authority.Experiment002RunAuthorityError, match="not issued"):
        authority._verified_state(capability)


def test_verified_registration_is_opaque_noncopyable_and_nonpickleable() -> None:
    with pytest.raises(TypeError, match="issuer-only"):
        authority.VerifiedRunRegistration()
    forged = object.__new__(authority.VerifiedRunRegistration)
    with pytest.raises(authority.Experiment002RunAuthorityError, match="not issued"):
        authority.verify_verified_run_registration(forged)
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.copy(forged)
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.deepcopy(forged)
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(forged)


@pytest.mark.parametrize("tamper_target", ["state", "guard"])
def test_controlled_capability_poisoned_by_independent_frame_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper_target: str,
) -> None:
    case = _valid_repository(tmp_path)
    snapshot = _prepare_controlled_capability_lifecycle(case, monkeypatch)
    capability = authority._issue_controlled_snapshot_for_tests(snapshot)
    original_state = authority._ISSUED[capability]
    original_guard = authority._ISSUED_GUARDS[capability]
    if tamper_target == "state":
        authority._ISSUED[capability] = dataclass_replace(
            original_state,
            registration_sha256="0" * 64,
        )
    else:
        authority._ISSUED_GUARDS[capability] = dataclass_replace(
            original_guard,
            registration_sha256="0" * 64,
        )

    with pytest.raises(authority.Experiment002RunAuthorityError, match="not issued"):
        authority._verified_state(capability)

    authority._ISSUED[capability] = original_state
    authority._ISSUED_GUARDS[capability] = original_guard
    with pytest.raises(authority.Experiment002RunAuthorityError, match="not issued"):
        authority._verified_state(capability)


def test_controlled_capability_is_process_local_across_fork(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not hasattr(os, "fork"):
        pytest.skip("fork is required for the process-local capability test")
    case = _valid_repository(tmp_path)
    snapshot = _prepare_controlled_capability_lifecycle(case, monkeypatch)
    capability = authority._issue_controlled_snapshot_for_tests(snapshot)
    read_descriptor, write_descriptor = os.pipe()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"This process .* is multi-threaded, use of fork\(\)",
            category=DeprecationWarning,
        )
        child_pid = os.fork()
    if child_pid == 0:
        os.close(read_descriptor)
        try:
            _ = capability.head_commit
        except authority.Experiment002RunAuthorityError:
            os.write(write_descriptor, b"rejected")
            os.close(write_descriptor)
            os._exit(0)
        os.close(write_descriptor)
        os._exit(2)

    os.close(write_descriptor)
    observed = os.read(read_descriptor, 64)
    os.close(read_descriptor)
    waited_pid, status = os.waitpid(child_pid, 0)
    assert waited_pid == child_pid
    assert os.WIFEXITED(status)
    assert os.WEXITSTATUS(status) == 0
    assert observed == b"rejected"
    assert capability.head_commit == case.head_commit


def test_forked_authority_fails_fast_while_another_thread_holds_issuer_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not hasattr(os, "fork"):
        pytest.skip("fork is required for the inherited-lock liveness test")
    case = _valid_repository(tmp_path)
    snapshot = _prepare_controlled_capability_lifecycle(case, monkeypatch)
    capability = authority._issue_controlled_snapshot_for_tests(snapshot)
    lock_held = threading.Event()
    release_lock = threading.Event()

    def hold_issuer_lock() -> None:
        with authority._ISSUED_LOCK:
            lock_held.set()
            release_lock.wait()

    holder = threading.Thread(target=hold_issuer_lock, daemon=True)
    holder.start()
    assert lock_held.wait(timeout=2.0)

    read_descriptor, write_descriptor = os.pipe()
    child_pid = -1
    child_reaped = False
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"This process .* is multi-threaded, use of fork\(\)",
                category=DeprecationWarning,
            )
            child_pid = os.fork()
        if child_pid == 0:
            os.close(read_descriptor)
            if not authority._ISSUED_LOCK.acquire(blocking=False):
                os.write(write_descriptor, b"inherited-locked-rlock")
                os.close(write_descriptor)
                os._exit(90)
            authority._ISSUED_LOCK.release()
            rejected = 0
            for action in (
                "property",
                "verify",
                "reverify",
                "bundle",
                "controlled_issue",
                "public_issue",
            ):
                try:
                    if action == "property":
                        _ = capability.head_commit
                    elif action == "verify":
                        authority.verify_verified_run_registration(capability)
                    elif action == "reverify":
                        authority.reverify_verified_run_registration(capability)
                    elif action == "bundle":
                        authority._create_sealed_experiment_002_child_bundle_fd(
                            capability
                        )
                    elif action == "controlled_issue":
                        authority._issue_controlled_snapshot_for_tests(snapshot)
                    else:
                        authority.verify_and_issue_experiment_002_run_registration()
                except authority.Experiment002RunAuthorityError:
                    rejected += 1
                except BaseException:
                    os.write(write_descriptor, b"unexpected-error")
                    os.close(write_descriptor)
                    os._exit(91)
                else:
                    os.write(write_descriptor, b"accepted")
                    os.close(write_descriptor)
                    os._exit(92)
            os.write(write_descriptor, str(rejected).encode("ascii"))
            os.close(write_descriptor)
            os._exit(0)

        os.close(write_descriptor)
        write_descriptor = -1
        readable, _, _ = select.select([read_descriptor], [], [], 3.0)
        if not readable:
            os.kill(child_pid, 9)
            os.waitpid(child_pid, 0)
            child_reaped = True
            pytest.fail("forked authority blocked on the inherited issuer lock")
        observed = os.read(read_descriptor, 64)
        waited_pid, status = os.waitpid(child_pid, 0)
        child_reaped = True
        assert waited_pid == child_pid
        assert os.WIFEXITED(status)
        assert os.WEXITSTATUS(status) == 0
        assert observed == b"6"
    finally:
        release_lock.set()
        holder.join(timeout=3.0)
        os.close(read_descriptor)
        if write_descriptor >= 0:
            os.close(write_descriptor)
        if child_pid > 0 and not child_reaped:
            try:
                os.kill(child_pid, 9)
            except ProcessLookupError:
                child_reaped = True
            try:
                os.waitpid(child_pid, 0)
            except ChildProcessError:
                child_reaped = True
    assert not holder.is_alive()
    assert capability.head_commit == case.head_commit
    authority.verify_verified_run_registration(capability)


def test_stale_public_reverification_permanently_poisons_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _valid_repository(tmp_path)
    snapshot = _prepare_controlled_capability_lifecycle(case, monkeypatch)
    capability = authority._issue_controlled_snapshot_for_tests(snapshot)
    authority.verify_verified_run_registration(capability)

    entrypoint = case.root / ENTRYPOINT_PATH
    original = entrypoint.read_bytes()
    entrypoint.write_bytes(b"DIRTY = True\n")
    with pytest.raises(authority.Experiment002RunAuthorityError):
        authority.verify_verified_run_registration(capability)
    entrypoint.write_bytes(original)

    with pytest.raises(authority.Experiment002RunAuthorityError, match="not issued"):
        authority.verify_verified_run_registration(capability)


def test_locked_final_reverification_closes_the_activation_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _valid_repository(tmp_path)
    snapshot = _prepare_controlled_capability_lifecycle(case, monkeypatch)
    capability = authority._issue_controlled_snapshot_for_tests(snapshot)
    entrypoint = case.root / ENTRYPOINT_PATH
    original = entrypoint.read_bytes()
    mutated = False

    def mutate_during_activation(
        root: Path,
        document: authority._RegistrationDocument,
    ) -> None:
        nonlocal mutated
        del root, document
        if not mutated:
            entrypoint.write_bytes(b"MUTATED_DURING_ACTIVATION = True\n")
            mutated = True

    monkeypatch.setattr(
        authority,
        "_require_registered_runtime_identity",
        mutate_during_activation,
    )
    try:
        with pytest.raises(authority.Experiment002RunAuthorityError):
            authority.verify_verified_run_registration(capability)
    finally:
        entrypoint.write_bytes(original)

    assert mutated
    with pytest.raises(authority.Experiment002RunAuthorityError, match="not issued"):
        authority.verify_verified_run_registration(capability)


def test_late_public_reverification_allows_bound_numerical_modules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _valid_repository(tmp_path)
    snapshot = _prepare_controlled_capability_lifecycle(
        case,
        monkeypatch,
        patch_runtime_identity=False,
    )
    capability = authority._issue_controlled_snapshot_for_tests(snapshot)
    document = authority._parse_registration((case.root / RUN_CONFIG_PATH).read_bytes())
    invocation = document.invocation
    runtime = document.runtime
    flags = cast(dict[str, int], invocation["python_flags"])
    environment = cast(dict[str, str], invocation["environment"])

    monkeypatch.chdir(case.root)
    monkeypatch.setattr(platform, "python_version", lambda: "3.12.3")
    monkeypatch.setattr(sys, "executable", PYTHON_EXECUTABLE)
    monkeypatch.setattr(sys, "argv", list(invocation["argv"]))
    monkeypatch.setattr(sys, "orig_argv", list(invocation["orig_argv"]))
    monkeypatch.setattr(sys, "path", list(invocation["sys_path"]))
    monkeypatch.setattr(sys, "flags", SimpleNamespace(**flags))
    monkeypatch.setattr(os, "environ", dict(environment))

    versions = {
        "numpy": cast(str, runtime["numpy_version"]),
        "safetensors": cast(str, runtime["safetensors_version"]),
        "torch": cast(str, runtime["torch_version"]),
    }

    def registered_version(distribution_name: str) -> str:
        return versions[distribution_name]

    monkeypatch.setattr(
        importlib.metadata,
        "version",
        registered_version,
    )
    for module_name in ("numpy", "safetensors", "torch"):
        monkeypatch.setitem(sys.modules, module_name, ModuleType(module_name))

    with pytest.raises(
        authority.Experiment002RunAuthorityError,
        match="loaded before run authorization",
    ):
        authority._require_production_activation(case.root, document)

    authority.verify_verified_run_registration(capability)
    assert capability.head_commit == case.head_commit
    assert capability not in authority._FAILED


def test_controlled_capability_issuance_is_concurrent_one_shot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _valid_repository(tmp_path)
    snapshot = _prepare_controlled_capability_lifecycle(case, monkeypatch)
    barrier = threading.Barrier(3)

    def attempt_issue() -> authority.VerifiedRunRegistration:
        barrier.wait()
        return authority._issue_controlled_snapshot_for_tests(snapshot)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(attempt_issue) for _ in range(2)]
        barrier.wait()
        outcomes: list[authority.VerifiedRunRegistration | BaseException] = []
        for future in futures:
            try:
                outcomes.append(future.result())
            except authority.Experiment002RunAuthorityError as error:
                outcomes.append(error)

    issued = [
        outcome
        for outcome in outcomes
        if isinstance(outcome, authority.VerifiedRunRegistration)
    ]
    rejected = [
        outcome
        for outcome in outcomes
        if isinstance(outcome, authority.Experiment002RunAuthorityError)
    ]
    assert len(issued) == 1
    assert len(rejected) == 1
    assert issued[0].head_commit == case.head_commit


def test_untracked_run_config_cannot_open_the_gate(tmp_path: Path) -> None:
    repository, implementation = _implementation_repository(tmp_path)
    case = _valid_repository(tmp_path / "other")
    document = copy.deepcopy(case.document)
    document["implementation_commit"] = implementation
    document["invocation"]["cwd"] = os.fspath(repository)
    (repository / RUN_CONFIG_PATH).write_bytes(_canonical(document))
    with pytest.raises(authority.Experiment002RunAuthorityError):
        authority._verify_synthetic_repository_for_tests(repository)


def test_committed_symlink_run_config_is_rejected(tmp_path: Path) -> None:
    repository, _implementation = _implementation_repository(tmp_path)
    run_path = repository / RUN_CONFIG_PATH
    run_path.symlink_to("experiment-002-trainer.json")
    _git(repository, "add", RUN_CONFIG_PATH.as_posix())
    _git(repository, "commit", "--quiet", "-m", "malicious symlink config")
    with pytest.raises(
        authority.Experiment002RunAuthorityError,
        match="regular nonsymlink",
    ):
        authority._verify_synthetic_repository_for_tests(repository)


def test_executable_run_config_blob_is_rejected(tmp_path: Path) -> None:
    case = _valid_repository(tmp_path)
    run_path = case.root / RUN_CONFIG_PATH
    run_path.chmod(0o755)
    _git(case.root, "add", RUN_CONFIG_PATH.as_posix())
    _git(case.root, "commit", "--quiet", "--amend", "--no-edit")
    with pytest.raises(
        authority.Experiment002RunAuthorityError,
        match="nonexecutable",
    ):
        authority._verify_synthetic_repository_for_tests(case.root)


@pytest.mark.parametrize("kind", ["tracked", "staged", "untracked"])
def test_dirty_or_untracked_worktree_is_rejected(tmp_path: Path, kind: str) -> None:
    case = _valid_repository(tmp_path)
    if kind == "tracked":
        (case.root / ENTRYPOINT_PATH).write_text("DIRTY = True\n", encoding="utf-8")
    elif kind == "staged":
        (case.root / ENTRYPOINT_PATH).write_text("STAGED = True\n", encoding="utf-8")
        _git(case.root, "add", ENTRYPOINT_PATH)
    else:
        (case.root / "untracked.txt").write_text("untracked\n", encoding="utf-8")
    with pytest.raises(
        authority.Experiment002RunAuthorityError,
        match=(
            "worktree file differs|tracked worktree file identity|"
            "stage-zero index|untracked regular file"
        ),
    ):
        authority._verify_synthetic_repository_for_tests(case.root)


def test_ignored_extra_file_inside_import_root_is_rejected(tmp_path: Path) -> None:
    case = _valid_repository(tmp_path)
    (case.root / "src/falsewake/extra.ignored").write_text(
        "ignored\n", encoding="utf-8"
    )
    assert _git(case.root, "status", "--porcelain=v1", "-z") == b""
    with pytest.raises(
        authority.Experiment002RunAuthorityError,
        match="untracked regular file|do not exactly cover",
    ):
        authority._verify_synthetic_repository_for_tests(case.root)


@pytest.mark.parametrize("attributes_source", ["git-info", "tracked"])
def test_clean_filter_cannot_hide_same_length_raw_mutation_or_execute(
    tmp_path: Path,
    attributes_source: str,
) -> None:
    if attributes_source == "git-info":
        case = _valid_repository(tmp_path)
        attributes = case.root / ".git/info/attributes"
        attributes.write_text(".gitignore filter=evil\n", encoding="utf-8")
    else:
        repository, _implementation = _implementation_repository(tmp_path)
        (repository / ".gitattributes").write_text(
            ".gitignore filter=evil\n",
            encoding="utf-8",
        )
        _git(repository, "add", ".gitattributes")
        _git(repository, "commit", "--quiet", "--amend", "--no-edit")
        implementation = _git(repository, "rev-parse", "HEAD").decode().strip()
        case = _register_repository(repository, implementation)
    marker = case.root / ".git/filter-clean-ran"
    helper = case.root / ".git/evil-clean"
    helper.write_text(
        "#!/bin/sh\n"
        "/bin/cat >/dev/null\n"
        "printf invoked > .git/filter-clean-ran\n"
        "printf '%s\\n' '*.ignored'\n",
        encoding="utf-8",
    )
    helper.chmod(0o700)
    _git(case.root, "config", "filter.evil.clean", "./.git/evil-clean")
    original = (case.root / ".gitignore").read_bytes()
    mutation = b"*.changed\n"
    assert len(original) == len(mutation)
    (case.root / ".gitignore").write_bytes(mutation)

    assert (
        _git(
            case.root,
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        )
        == b""
    )
    assert marker.read_bytes() == b"invoked"
    marker.unlink()

    with pytest.raises(
        authority.Experiment002RunAuthorityError,
        match="worktree file differs from committed HEAD blob",
    ):
        authority._verify_synthetic_repository_for_tests(case.root)
    assert not marker.exists()


def test_filter_process_from_local_config_is_never_executed(tmp_path: Path) -> None:
    case = _valid_repository(tmp_path)
    (case.root / ".git/info/attributes").write_text(
        "*.py filter=process-probe\n",
        encoding="utf-8",
    )
    marker = case.root / ".git/filter-process-ran"
    helper = case.root / ".git/evil-process"
    helper.write_text(
        "#!/bin/sh\nprintf invoked > .git/filter-process-ran\nexit 91\n",
        encoding="utf-8",
    )
    helper.chmod(0o700)
    _git(
        case.root,
        "config",
        "filter.process-probe.process",
        "./.git/evil-process",
    )

    snapshot = authority._verify_synthetic_repository_for_tests(case.root)

    assert snapshot.head_commit == case.head_commit
    assert not marker.exists()


@pytest.mark.parametrize("mechanism", ["gitignore", "info", "core"])
def test_raw_inventory_rejects_files_hidden_by_every_exclude_layer(
    tmp_path: Path,
    mechanism: str,
) -> None:
    case = _valid_repository(tmp_path)
    if mechanism == "gitignore":
        relative_path = "hidden.ignored"
    elif mechanism == "info":
        relative_path = "hidden.info"
        (case.root / ".git/info/exclude").write_text(
            f"{relative_path}\n",
            encoding="utf-8",
        )
    else:
        relative_path = "hidden.global"
        excludes = case.root / ".git/global-excludes"
        excludes.write_text(f"{relative_path}\n", encoding="utf-8")
        _git(case.root, "config", "core.excludesFile", os.fspath(excludes))
    (case.root / relative_path).write_text("hidden\n", encoding="utf-8")
    assert (
        _git(
            case.root,
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        )
        == b""
    )

    with pytest.raises(
        authority.Experiment002RunAuthorityError,
        match="untracked regular file",
    ):
        authority._verify_synthetic_repository_for_tests(case.root)


@pytest.mark.parametrize("kind", ["symlink", "fifo", "socket"])
def test_raw_inventory_rejects_untracked_symlink_or_special_leaf(
    tmp_path: Path,
    kind: str,
) -> None:
    case = _valid_repository(tmp_path)
    path = case.root / f"untracked-{kind}"
    listener: socket.socket | None = None
    directory_descriptor: int | None = None
    try:
        if kind == "symlink":
            path.symlink_to(".gitignore")
        elif kind == "fifo":
            os.mkfifo(path)
        else:
            directory_descriptor = os.open(
                case.root,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
            )
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(f"/proc/self/fd/{directory_descriptor}/{path.name}")
        with pytest.raises(
            authority.Experiment002RunAuthorityError,
            match="symlink or special filesystem entry",
        ):
            authority._verify_synthetic_repository_for_tests(case.root)
    finally:
        if listener is not None:
            listener.close()
        if directory_descriptor is not None:
            os.close(directory_descriptor)


def test_raw_inventory_rejects_untracked_empty_directory(tmp_path: Path) -> None:
    case = _valid_repository(tmp_path)
    (case.root / "empty-untracked-directory").mkdir()

    with pytest.raises(
        authority.Experiment002RunAuthorityError,
        match="untracked directory",
    ):
        authority._verify_synthetic_repository_for_tests(case.root)


def test_tracked_hardlink_alias_is_rejected(tmp_path: Path) -> None:
    case = _valid_repository(tmp_path)
    tracked = case.root / ENTRYPOINT_PATH
    alias = case.root / ".git/tracked-hardlink-alias"
    alias.write_bytes(tracked.read_bytes())
    tracked.unlink()
    os.link(alias, tracked)
    assert tracked.stat().st_nlink == 2

    with pytest.raises(
        authority.Experiment002RunAuthorityError,
        match="singly-linked|multiply-linked|file identity is invalid",
    ):
        authority._verify_synthetic_repository_for_tests(case.root)


def test_corrupt_loose_blob_is_rejected_by_full_verifier(tmp_path: Path) -> None:
    case = _valid_repository(tmp_path)
    path = ".gitignore"
    old_oid = _git(case.root, "rev-parse", f"HEAD:{path}").decode().strip()
    old_payload = (case.root / path).read_bytes()
    replacement = b"*.changed\n"
    assert len(old_payload) == len(replacement)
    replacement_frame = (
        b"blob " + str(len(replacement)).encode("ascii") + b"\0" + replacement
    )
    replacement_oid = hashlib.sha1(
        replacement_frame,
        usedforsecurity=False,
    ).hexdigest()
    assert replacement_oid != old_oid
    loose_object = case.root / ".git/objects" / old_oid[:2] / old_oid[2:]
    assert loose_object.is_file()
    loose_object.chmod(stat.S_IMODE(loose_object.stat().st_mode) | stat.S_IWUSR)
    loose_object.write_bytes(zlib.compress(replacement_frame))
    (case.root / path).write_bytes(replacement)
    assert _git(case.root, "cat-file", "blob", old_oid) == replacement

    with pytest.raises(
        authority.Experiment002RunAuthorityError,
        match="does not match its Git object ID",
    ):
        authority._verify_synthetic_repository_for_tests(case.root)


def test_tracked_entry_count_is_bounded_before_object_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = tuple(
        (b"100644", b"0" * 40, f"path-{index:04d}".encode("ascii"))
        for index in range(4_097)
    )
    object_read = False

    def forbidden_preflight(*args: object, **kwargs: object) -> int:
        nonlocal object_read
        del args, kwargs
        object_read = True
        raise AssertionError("tracked entry count must reject before object reads")

    monkeypatch.setattr(
        authority,
        "_preflight_committed_blob_size",
        forbidden_preflight,
    )
    with pytest.raises(
        authority.Experiment002RunAuthorityError,
        match="tracked tree entry count",
    ):
        authority._require_raw_tracked_worktree(tmp_path, entries)
    assert not object_read


def test_tracked_cumulative_budget_is_checked_before_any_payload_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = tuple(
        (b"100644", b"0" * 40, f"path-{index}".encode("ascii")) for index in range(5)
    )
    preflight_count = 0
    payload_loaded = False

    def maximum_preflight(*args: object, **kwargs: object) -> int:
        nonlocal preflight_count
        del args, kwargs
        preflight_count += 1
        return 64 << 20

    def forbidden_load(*args: object, **kwargs: object) -> authority._TreeBlob:
        nonlocal payload_loaded
        del args, kwargs
        payload_loaded = True
        raise AssertionError("aggregate size must reject before payload loading")

    monkeypatch.setattr(
        authority,
        "_preflight_committed_blob_size",
        maximum_preflight,
    )
    monkeypatch.setattr(
        authority,
        "_load_preflighted_committed_blob",
        forbidden_load,
    )
    with pytest.raises(
        authority.Experiment002RunAuthorityError,
        match="cumulative raw byte budget",
    ):
        authority._require_raw_tracked_worktree(tmp_path, entries)
    assert preflight_count == 5
    assert not payload_loaded


@pytest.mark.parametrize("kind", ["gitdir-file", "symlink"])
def test_noncanonical_top_level_git_indirection_is_rejected(
    tmp_path: Path,
    kind: str,
) -> None:
    case = _valid_repository(tmp_path)
    git_directory = case.root / ".git"
    external_git_directory = tmp_path / f"external-{kind}-metadata"
    git_directory.rename(external_git_directory)
    if kind == "gitdir-file":
        git_directory.write_text(
            f"gitdir: {external_git_directory}\n",
            encoding="utf-8",
        )
    else:
        git_directory.symlink_to(external_git_directory, target_is_directory=True)
    assert _git(case.root, "rev-parse", "HEAD").decode().strip() == case.head_commit

    with pytest.raises(
        authority.Experiment002RunAuthorityError,
        match=r"metadata directory|top-level \.git|opened safely",
    ):
        authority._verify_synthetic_repository_for_tests(case.root)


def test_external_git_common_directory_is_rejected(tmp_path: Path) -> None:
    case = _valid_repository(tmp_path)
    git_directory = case.root / ".git"
    common_directory = tmp_path / "external-common-metadata"
    git_directory.rename(common_directory)
    git_directory.mkdir()
    shutil.move(common_directory / "HEAD", git_directory / "HEAD")
    shutil.move(common_directory / "index", git_directory / "index")
    (git_directory / "commondir").write_text(
        f"{common_directory}\n",
        encoding="utf-8",
    )
    assert _git(case.root, "rev-parse", "HEAD").decode().strip() == case.head_commit
    assert _git(
        case.root,
        "rev-parse",
        "--path-format=absolute",
        "--git-common-dir",
    ).decode().strip() == os.fspath(common_directory)

    with pytest.raises(
        authority.Experiment002RunAuthorityError,
        match="common metadata directory",
    ):
        authority._verify_synthetic_repository_for_tests(case.root)


@pytest.mark.parametrize("resource", ["depth", "directory-prefixes"])
def test_tracked_directory_resources_reject_before_object_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resource: str,
) -> None:
    paths: tuple[str, ...]
    if resource == "depth":
        paths = ("/".join((*(["d"] * 65), "file")),)
    else:
        paths = tuple(f"d{index:04d}/nested/file" for index in range(4_096))
    entries = tuple((b"100644", b"0" * 40, path.encode("ascii")) for path in paths)
    object_read = False

    def forbidden_preflight(*args: object, **kwargs: object) -> int:
        nonlocal object_read
        del args, kwargs
        object_read = True
        raise AssertionError("directory resources must reject before object reads")

    monkeypatch.setattr(
        authority,
        "_preflight_committed_blob_size",
        forbidden_preflight,
    )
    with pytest.raises(
        authority.Experiment002RunAuthorityError,
        match="depth limit|too many directory prefixes",
    ):
        authority._require_raw_tracked_worktree(tmp_path, entries)
    assert not object_read


def test_fd_walk_detects_atomic_tracked_file_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _valid_repository(tmp_path)
    original_load = authority._load_preflighted_committed_blob
    replaced = False

    def replace_after_open(
        root: Path,
        identity: authority._TreeBlobIdentity,
        path: str,
        *,
        declared_byte_count: int,
    ) -> authority._TreeBlob:
        nonlocal replaced
        blob = original_load(
            root,
            identity,
            path,
            declared_byte_count=declared_byte_count,
        )
        if path == ENTRYPOINT_PATH and not replaced:
            replacement = root / "src/falsewake/.replacement"
            replacement.write_bytes(blob.payload)
            os.replace(replacement, root / path)
            replaced = True
        return blob

    monkeypatch.setattr(
        authority,
        "_load_preflighted_committed_blob",
        replace_after_open,
    )
    with pytest.raises(
        authority.Experiment002RunAuthorityError,
        match="name changed|directory changed|file changed",
    ):
        authority._verify_synthetic_repository_for_tests(case.root)
    assert replaced


def test_fd_walk_regular_to_fifo_race_is_nonblocking_and_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _valid_repository(tmp_path)
    original_open = os.open
    target_name = Path(ENTRYPOINT_PATH).name
    swapped = False

    def swap_before_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if path == target_name and dir_fd is not None and flags & os.O_NONBLOCK:
            assert flags & os.O_NOFOLLOW
            os.unlink(path, dir_fd=dir_fd)
            os.mkfifo(path, dir_fd=dir_fd)
            swapped = True
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", swap_before_open)
    with pytest.raises(
        authority.Experiment002RunAuthorityError,
        match="file identity is invalid|symlink or special",
    ):
        authority._verify_synthetic_repository_for_tests(case.root)
    assert swapped


@pytest.mark.parametrize("injected_failure", [False, True])
def test_fd_walk_closes_every_descriptor_on_success_and_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    injected_failure: bool,
) -> None:
    case = _valid_repository(tmp_path)
    if injected_failure:
        original_load = authority._load_preflighted_committed_blob

        def fail_after_descriptors_open(
            root: Path,
            identity: authority._TreeBlobIdentity,
            path: str,
            *,
            declared_byte_count: int,
        ) -> authority._TreeBlob:
            if path == ENTRYPOINT_PATH:
                raise authority.Experiment002RunAuthorityError(
                    "injected descriptor cleanup failure"
                )
            return original_load(
                root,
                identity,
                path,
                declared_byte_count=declared_byte_count,
            )

        monkeypatch.setattr(
            authority,
            "_load_preflighted_committed_blob",
            fail_after_descriptors_open,
        )
    descriptors_before = set(os.listdir("/proc/self/fd"))

    if injected_failure:
        with pytest.raises(
            authority.Experiment002RunAuthorityError,
            match="injected descriptor cleanup failure",
        ):
            authority._verify_synthetic_repository_for_tests(case.root)
    else:
        authority._verify_synthetic_repository_for_tests(case.root)

    assert set(os.listdir("/proc/self/fd")) == descriptors_before


def test_final_head_check_follows_the_last_git_topology_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _valid_repository(tmp_path)
    original_git = authority._git
    topology_call_count = 0
    mutated = False

    def mutate_during_last_topology(
        root: Path,
        *arguments: str,
        maximum_stdout_bytes: int = 32 << 20,
    ) -> bytes:
        nonlocal mutated, topology_call_count
        output = original_git(
            root,
            *arguments,
            maximum_stdout_bytes=maximum_stdout_bytes,
        )
        if arguments == (
            "rev-parse",
            "--path-format=absolute",
            "--git-dir",
        ):
            topology_call_count += 1
            if topology_call_count == 6:
                _git(root, "update-ref", "HEAD", case.implementation_commit)
                mutated = True
        return output

    monkeypatch.setattr(authority, "_git", mutate_during_last_topology)
    with pytest.raises(
        authority.Experiment002RunAuthorityError,
        match="HEAD changed during final topology verification",
    ):
        authority._verify_synthetic_repository_for_tests(case.root)

    assert topology_call_count == 6
    assert mutated
    assert _git(case.root, "rev-parse", "HEAD").decode().strip() == (
        case.implementation_commit
    )


def test_import_root_count_is_bounded_before_overlap_or_git_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _valid_repository(tmp_path)
    document = copy.deepcopy(case.document)
    source = cast(dict[str, Any], document["source_bundle"])
    source["import_roots"] = [f"roots/r{index:04d}" for index in range(4_097)]
    _amend_document(case, document)
    git_called = False

    def forbidden_git(*args: object, **kwargs: object) -> bytes:
        nonlocal git_called
        del args, kwargs
        git_called = True
        raise AssertionError("import-root count must fail before Git")

    monkeypatch.setattr(authority, "_git", forbidden_git)
    with pytest.raises(
        authority.Experiment002RunAuthorityError,
        match="too many registered import roots",
    ):
        authority._verify_synthetic_repository_for_tests(case.root)
    assert not git_called


def test_import_root_enumeration_counts_empty_directories_cumulatively(
    tmp_path: Path,
) -> None:
    import_root = tmp_path / "roots"
    import_root.mkdir()
    for index in range(4_097):
        (import_root / f"d{index:04d}").mkdir()
    with pytest.raises(
        authority.Experiment002RunAuthorityError,
        match="too many entries",
    ):
        authority._enumerate_import_roots(tmp_path, ("roots",))


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_index_hiding_flags_are_rejected(tmp_path: Path, flag: str) -> None:
    case = _valid_repository(tmp_path)
    _git(case.root, "update-index", flag, ENTRYPOINT_PATH)
    with pytest.raises(
        authority.Experiment002RunAuthorityError,
        match="assume-unchanged|skip-worktree|unusual",
    ):
        authority._verify_synthetic_repository_for_tests(case.root)


def test_worktree_blob_mismatch_is_rejected(tmp_path: Path) -> None:
    case = _valid_repository(tmp_path)
    document = copy.deepcopy(case.document)
    document["schema_version"] = 1
    (case.root / RUN_CONFIG_PATH).write_bytes(_canonical(document) + b" ")
    with pytest.raises(authority.Experiment002RunAuthorityError):
        authority._verify_synthetic_repository_for_tests(case.root)


@pytest.mark.parametrize(
    "mutation",
    ["traversal", "duplicate", "unsorted", "uppercase_hex", "digest_mismatch"],
)
def test_invalid_source_registration_fails_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    case = _valid_repository(tmp_path)
    document = copy.deepcopy(case.document)
    source = cast(dict[str, Any], document["source_bundle"])
    paths = cast(list[str], source["paths"])
    if mutation == "traversal":
        paths[0] = "../src/falsewake/experiment_002_runner.py"
    elif mutation == "duplicate":
        paths[1] = paths[0]
    elif mutation == "unsorted":
        paths.reverse()
    elif mutation == "uppercase_hex":
        source["sha256"] = cast(str, source["sha256"]).upper()
    else:
        source["sha256"] = "0" * 64
    _amend_document(case, document)
    with pytest.raises(authority.Experiment002RunAuthorityError):
        authority._verify_synthetic_repository_for_tests(case.root)


def test_declared_source_overflow_fails_before_any_git_object_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _valid_repository(tmp_path)
    document = copy.deepcopy(case.document)
    source = cast(dict[str, Any], document["source_bundle"])
    source["payload_byte_count"] = 256 << 20
    _amend_document(case, document)
    git_called = False

    def forbidden_git(*args: object, **kwargs: object) -> bytes:
        nonlocal git_called
        del args, kwargs
        git_called = True
        raise AssertionError("declared total overflow must fail before Git")

    monkeypatch.setattr(authority, "_git", forbidden_git)
    with pytest.raises(
        authority.Experiment002RunAuthorityError,
        match="exact total byte limit",
    ):
        authority._verify_synthetic_repository_for_tests(case.root)
    assert not git_called


def test_oversized_tree_blob_fails_size_preflight_before_payload_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _valid_repository(tmp_path)
    runner_oid = (
        _git(
            case.root,
            "rev-parse",
            f"{case.implementation_commit}:{ENTRYPOINT_PATH}",
        )
        .decode("ascii")
        .strip()
    )
    original_git = authority._git
    size_preflighted = False
    payload_requested = False

    def declared_oversized_blob(
        root: Path,
        *arguments: str,
        maximum_stdout_bytes: int = 32 << 20,
    ) -> bytes:
        nonlocal size_preflighted, payload_requested
        if arguments == ("cat-file", "-s", runner_oid):
            size_preflighted = True
            return f"{(64 << 20) + 1}\n".encode("ascii")
        if arguments == ("cat-file", "blob", runner_oid):
            payload_requested = True
            raise AssertionError("oversized blob payload must not be requested")
        return original_git(
            root,
            *arguments,
            maximum_stdout_bytes=maximum_stdout_bytes,
        )

    monkeypatch.setattr(authority, "_git", declared_oversized_blob)
    with pytest.raises(
        authority.Experiment002RunAuthorityError,
        match="exceeds its retained byte budget",
    ):
        authority._verify_synthetic_repository_for_tests(case.root)
    assert size_preflighted
    assert not payload_requested


def test_head_identity_check_reuses_one_preflighted_source_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _valid_repository(tmp_path)
    runner_oid = (
        _git(
            case.root,
            "rev-parse",
            f"{case.implementation_commit}:{ENTRYPOINT_PATH}",
        )
        .decode("ascii")
        .strip()
    )
    original_git = authority._git
    events: list[str] = []

    def observed_git(
        root: Path,
        *arguments: str,
        maximum_stdout_bytes: int = 32 << 20,
    ) -> bytes:
        if arguments == ("cat-file", "-s", runner_oid):
            events.append("size")
        elif arguments == ("cat-file", "blob", runner_oid):
            events.append("payload")
        return original_git(
            root,
            *arguments,
            maximum_stdout_bytes=maximum_stdout_bytes,
        )

    monkeypatch.setattr(authority, "_git", observed_git)
    snapshot = authority._verify_synthetic_repository_for_tests(case.root)
    assert snapshot.source_paths == case.source_paths
    assert events == ["size", "payload"] * 3


def test_frozen_binding_cannot_be_overridden_in_config(tmp_path: Path) -> None:
    case = _valid_repository(tmp_path)
    document = copy.deepcopy(case.document)
    document["frozen_bindings"]["trainer"]["sha256"] = "0" * 64
    _amend_document(case, document)
    with pytest.raises(
        authority.Experiment002RunAuthorityError,
        match="exact frozen bindings",
    ):
        authority._verify_synthetic_repository_for_tests(case.root)


@pytest.mark.parametrize("missing", ["external_inputs", "invocation", "runtime"])
def test_activation_schema_cannot_be_omitted(tmp_path: Path, missing: str) -> None:
    case = _valid_repository(tmp_path)
    document = copy.deepcopy(case.document)
    del document[missing]
    _amend_document(case, document)
    with pytest.raises(authority.Experiment002RunAuthorityError, match="invalid keys"):
        authority._verify_synthetic_repository_for_tests(case.root)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("unbound_entrypoint", "entrypoint is absent"),
        ("external_sys_path", "exact source-bound runtime allowlist"),
        ("repository_sys_path", "exact source-bound runtime allowlist"),
    ],
)
def test_invocation_cannot_expose_unregistered_executable_code(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    case = _valid_repository(tmp_path)
    document = copy.deepcopy(case.document)
    invocation = cast(dict[str, Any], document["invocation"])
    if mutation == "unbound_entrypoint":
        invocation["argv"][0] = "configs/experiment-002-trainer.json"
        invocation["orig_argv"][4] = "configs/experiment-002-trainer.json"
    elif mutation == "external_sys_path":
        invocation["sys_path"].append("/tmp/unregistered-python-root")
    else:
        invocation["sys_path"].append(os.fspath(case.root / "configs"))
    _amend_document(case, document)

    with pytest.raises(authority.Experiment002RunAuthorityError, match=message):
        authority._verify_synthetic_repository_for_tests(case.root)


def test_frozen_committed_blob_mismatch_is_rejected(tmp_path: Path) -> None:
    repository, implementation = _implementation_repository(tmp_path)
    trainer = repository / "configs/experiment-002-trainer.json"
    trainer.write_bytes(trainer.read_bytes() + b"\n")
    _git(repository, "add", "configs/experiment-002-trainer.json")
    _git(repository, "commit", "--quiet", "--amend", "--no-edit")
    implementation = _git(repository, "rev-parse", "HEAD").decode().strip()

    source_paths = tuple(
        sorted(
            (
                ENTRYPOINT_PATH,
                "src/falsewake/causal_kws.py",
                "src/falsewake/experiment_002_run_authority.py",
            ),
            key=lambda value: value.encode("utf-8"),
        )
    )
    payload_count, digest = _source_digest(repository, source_paths)
    document = {
        "experiment": "002",
        "external_inputs": _external_inputs(),
        "frozen_bindings": authority._expected_frozen_bindings(),
        "implementation_commit": implementation,
        "invocation": _invocation(repository),
        "runtime": _runtime(),
        "schema_version": 1,
        "source_bundle": {
            "import_roots": ["src/falsewake"],
            "paths": list(source_paths),
            "payload_byte_count": payload_count,
            "sha256": digest,
        },
    }
    (repository / RUN_CONFIG_PATH).write_bytes(_canonical(document))
    _git(repository, "add", RUN_CONFIG_PATH.as_posix())
    _git(repository, "commit", "--quiet", "-m", "register altered roots")
    with pytest.raises(
        authority.Experiment002RunAuthorityError,
        match="frozen committed file digest mismatch",
    ):
        authority._verify_synthetic_repository_for_tests(repository)


def test_nonancestor_implementation_commit_is_rejected(tmp_path: Path) -> None:
    case = _valid_repository(tmp_path)
    unrelated = (
        _git(
            case.root,
            "commit-tree",
            f"{case.implementation_commit}^{{tree}}",
            input_bytes=b"unrelated implementation\n",
        )
        .decode()
        .strip()
    )
    document = copy.deepcopy(case.document)
    document["implementation_commit"] = unrelated
    _amend_document(case, document)
    with pytest.raises(
        authority.Experiment002RunAuthorityError,
        match="not an ancestor",
    ):
        authority._verify_synthetic_repository_for_tests(case.root)


def test_registration_commit_may_add_only_the_run_config(tmp_path: Path) -> None:
    case = _valid_repository(tmp_path)
    (case.root / "also-added.txt").write_text("extra\n", encoding="utf-8")
    _git(case.root, "add", "also-added.txt")
    _git(case.root, "commit", "--quiet", "--amend", "--no-edit")
    with pytest.raises(
        authority.Experiment002RunAuthorityError,
        match="add only",
    ):
        authority._verify_synthetic_repository_for_tests(case.root)


def test_duplicate_json_key_and_float_are_rejected_before_git(tmp_path: Path) -> None:
    repository, _implementation = _implementation_repository(tmp_path)
    run_path = repository / RUN_CONFIG_PATH
    run_path.write_text(
        '{"experiment":"002","experiment":"002","value":1.5}\n',
        encoding="ascii",
    )
    with pytest.raises(authority.Experiment002RunAuthorityError, match="exact JSON"):
        authority._verify_synthetic_repository_for_tests(repository)


def test_subprocess_output_ambiguity_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _valid_repository(tmp_path)

    def ambiguous_git(
        root: Path,
        *arguments: str,
        maximum_stdout_bytes: int = 32 << 20,
    ) -> bytes:
        del root, arguments, maximum_stdout_bytes
        return b"/first\n/second\n"

    monkeypatch.setattr(authority, "_git", ambiguous_git)
    with pytest.raises(
        authority.Experiment002RunAuthorityError,
        match="ambiguous",
    ):
        authority._verify_synthetic_repository_for_tests(case.root)


def test_bounded_git_stdout_overflow_kills_and_reaps_incremental_producer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid_path = tmp_path / "overflow.pid"
    executable = tmp_path / "fake-git-overflow"
    executable.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$$\" > {os.fspath(pid_path)!r}\n"
        "while :; do printf '0123456789abcdef0123456789abcdef'; done\n",
        encoding="ascii",
    )
    executable.chmod(0o755)
    monkeypatch.setattr(shutil, "which", lambda _name, path: os.fspath(executable))
    started = time.monotonic()
    with pytest.raises(
        authority.Experiment002RunAuthorityError,
        match="output is too large",
    ):
        authority._git_process(
            tmp_path,
            ("ignored",),
            allowed_returncodes=(0,),
            maximum_stdout_bytes=1_024,
        )
    assert time.monotonic() - started < 2.0
    producer_pid = int(pid_path.read_text(encoding="ascii"))
    with pytest.raises(ProcessLookupError):
        os.kill(producer_pid, 0)


def test_bounded_git_timeout_kills_and_reaps_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid_path = tmp_path / "timeout.pid"
    executable = tmp_path / "fake-git-timeout"
    executable.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$$\" > {os.fspath(pid_path)!r}\n"
        "exec /bin/sleep 60\n",
        encoding="ascii",
    )
    executable.chmod(0o755)
    monkeypatch.setattr(shutil, "which", lambda _name, path: os.fspath(executable))
    monkeypatch.setattr(authority, "_GIT_TIMEOUT_SECONDS", 0.2)
    started = time.monotonic()
    with pytest.raises(
        authority.Experiment002RunAuthorityError,
        match="isolated Git command failed",
    ):
        authority._git_process(
            tmp_path,
            ("ignored",),
            allowed_returncodes=(0,),
            maximum_stdout_bytes=1_024,
        )
    assert time.monotonic() - started < 2.0
    producer_pid = int(pid_path.read_text(encoding="ascii"))
    with pytest.raises(ProcessLookupError):
        os.kill(producer_pid, 0)


def test_bounded_git_drains_stdout_and_stderr_without_pipe_deadlock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "fake-git-dual-stream"
    executable.write_text(
        "#!/bin/sh\n"
        "(/usr/bin/head -c 131072 /dev/zero) &\n"
        "(/usr/bin/head -c 131072 /dev/zero >&2) &\n"
        "wait\n",
        encoding="ascii",
    )
    executable.chmod(0o755)
    monkeypatch.setattr(shutil, "which", lambda _name, path: os.fspath(executable))
    completed = authority._git_process(
        tmp_path,
        ("ignored",),
        allowed_returncodes=(0,),
        maximum_stdout_bytes=131_072,
    )
    assert completed.returncode == 0
    assert completed.stdout == b"\0" * 131_072
    assert completed.stderr == b"\0" * 131_072


def test_bounded_git_kills_same_group_descendant_after_clean_leader_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descendant_pid_path = tmp_path / "descendant.pid"
    executable = tmp_path / "fake-git-descendant"
    executable.write_text(
        "#!/bin/sh\n"
        "/bin/sleep 60 </dev/null >/dev/null 2>&1 &\n"
        f"printf '%s\\n' \"$!\" > {os.fspath(descendant_pid_path)!r}\n"
        "exit 0\n",
        encoding="ascii",
    )
    executable.chmod(0o755)
    monkeypatch.setattr(shutil, "which", lambda _name, path: os.fspath(executable))
    completed = authority._git_process(
        tmp_path,
        ("ignored",),
        allowed_returncodes=(0,),
        maximum_stdout_bytes=1_024,
    )
    assert completed.returncode == 0
    descendant_pid = int(descendant_pid_path.read_text(encoding="ascii"))
    deadline = time.monotonic() + 2.0
    while Path(f"/proc/{descendant_pid}").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not Path(f"/proc/{descendant_pid}").exists()


def test_successful_git_process_group_is_killed_once_before_leader_reap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "fake-git-clean-exit"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="ascii")
    executable.chmod(0o755)
    monkeypatch.setattr(shutil, "which", lambda _name, path: os.fspath(executable))
    original_killpg = os.killpg
    calls: list[tuple[int, int, bool]] = []

    def observed_killpg(process_group: int, supplied_signal: int) -> None:
        calls.append(
            (
                process_group,
                supplied_signal,
                Path(f"/proc/{process_group}").exists(),
            )
        )
        original_killpg(process_group, supplied_signal)

    monkeypatch.setattr(os, "killpg", observed_killpg)
    completed = authority._git_process(
        tmp_path,
        ("ignored",),
        allowed_returncodes=(0,),
        maximum_stdout_bytes=1_024,
    )
    assert completed.returncode == 0
    assert len(calls) == 1
    process_group, supplied_signal, leader_existed = calls[0]
    assert supplied_signal == signal.SIGKILL
    assert leader_existed
    assert not Path(f"/proc/{process_group}").exists()


def test_isolated_git_disables_lazy_fetch_and_pager_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment_path = tmp_path / "git-environment.txt"
    arguments_path = tmp_path / "git-arguments.txt"
    executable = tmp_path / "fake-git-environment"
    executable.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$GIT_NO_LAZY_FETCH\" > {os.fspath(environment_path)!r}\n"
        f"printf '%s\\n' \"$@\" > {os.fspath(arguments_path)!r}\n"
        "exit 0\n",
        encoding="ascii",
    )
    executable.chmod(0o755)
    monkeypatch.setattr(shutil, "which", lambda _name, path: os.fspath(executable))
    completed = authority._git_process(
        tmp_path,
        ("ignored",),
        allowed_returncodes=(0,),
        maximum_stdout_bytes=1_024,
    )
    assert completed.returncode == 0
    assert environment_path.read_text(encoding="ascii") == "1\n"
    arguments = arguments_path.read_text(encoding="ascii").splitlines()
    assert arguments[0] == "--no-pager"
    assert "--literal-pathspecs" in arguments


def test_synthetic_root_rejects_path_subclasses_before_execution(
    tmp_path: Path,
) -> None:
    class DerivedPath(PosixPath):
        pass

    derived = DerivedPath(tmp_path)
    with pytest.raises(TypeError, match="exact pathlib path"):
        authority._verify_synthetic_repository_for_tests(derived)
