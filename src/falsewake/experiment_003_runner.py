"""Exact parent-or-sealed-child bootstrap for Experiment 003.

The runner itself is executed from the supervisor's pinned descriptor.  After
validating that isolated interpreter invocation it classifies fixed descriptor
3 exactly once.  ``EBADF`` selects the parent path; every open descriptor 3
irrevocably selects the child path, where project code is admitted exclusively
from the byte-sealed bundle on descriptor 7.

The child loader is deliberately implemented here, before any ``falsewake``
import.  It removes the repository from ``sys.path``, hard-denies source
fallback for every ``falsewake`` name, and keeps the sealed mmap alive while
the child-local authority and guarded coordinator run.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import mmap
import os
import stat
import struct
import sys
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from types import MappingProxyType, ModuleType
from typing import Any, Final, NoReturn, Protocol, cast

_REPOSITORY_ROOT: Final = "/home/ubuntu/gitcode/falsewake"
_ENTRYPOINT: Final = "src/falsewake/experiment_003_runner.py"
_CANONICAL_FILE: Final = f"{_REPOSITORY_ROOT}/{_ENTRYPOINT}"
_PYTHON_EXECUTABLE: Final = "/home/ubuntu/gitcode/.t/falsewake-venv/bin/python"
_BOOTSTRAP_SYS_PATH: Final = (
    "/usr/lib/python312.zip",
    "/usr/lib/python3.12",
    "/usr/lib/python3.12/lib-dynload",
)
_RUNTIME_SYS_PATH_ROOTS: Final = (
    *_BOOTSTRAP_SYS_PATH,
    "/home/ubuntu/gitcode/.t/falsewake-venv/lib/python3.12/site-packages",
)
_REGISTERED_SYS_PATH: Final = (
    f"{_REPOSITORY_ROOT}/src",
    *_RUNTIME_SYS_PATH_ROOTS,
)
_EXPECTED_ARGV: Final = (_ENTRYPOINT,)
_EXPECTED_ORIG_ARGV: Final = (
    _PYTHON_EXECUTABLE,
    "-I",
    "-S",
    "-B",
    _ENTRYPOINT,
)
_EXPECTED_ENVIRONMENT: Final = {
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
    "TMPDIR": "/home/ubuntu/gitcode/.t/falsewake-experiment-003-scratch",
    "TZ": "UTC",
}
_EXPECTED_FLAGS: Final = {
    "dont_write_bytecode": 1,
    "ignore_environment": 1,
    "isolated": 1,
    "no_site": 1,
    "no_user_site": 1,
}
_FORBIDDEN_BOOTSTRAP_MODULE_ROOTS: Final = frozenset(
    {"falsewake", "numpy", "safetensors", "site", "torch"}
)
_AUTHORITY_MODULE: Final = "falsewake.experiment_003_run_authority"
_COORDINATOR_MODULE: Final = "falsewake.experiment_003_coordinator"
_SEALED_CHILD_ISSUER: Final = (
    "_verify_and_issue_experiment_003_sealed_child_registration"
)
_GUARDED_CHILD_ROUTE: Final = "_run_guarded_registered_seed_child"

_PARENT_ROLE: Final = "parent"
_CHILD_ROLE: Final = "child"
_CHILD_CONTROL_FD: Final = 3
_CHILD_SOURCE_BUNDLE_FD: Final = 7
_CHILD_BUNDLE_MAGIC: Final = b"FW3CHLD1"
_CHILD_BUNDLE_VERSION: Final = 1
_CHILD_BUNDLE_HEADER: Final = struct.Struct("<8sI40s40s64s64sIQQ")
_SOURCE_BUNDLE_DOMAIN: Final = b"falsewake-exp003-source-bundle-v1\0"
_MAX_CONFIG_BYTES: Final = 1 << 20
_MAX_SOURCE_FILE_BYTES: Final = 64 << 20
_MAX_SOURCE_BUNDLE_BYTES: Final = 256 << 20
_MAX_SOURCE_FILES: Final = 4_096
_MAX_FROZEN_FILE_BYTES: Final = 64 << 20
_MAX_FROZEN_BUNDLE_BYTES: Final = 256 << 20
_MAX_FROZEN_FILES: Final = 4_096
_MAX_CHILD_BUNDLE_BYTES: Final = 256 << 20
_SOURCE_BUNDLE_SEALS: Final = (
    fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL
)
_LOWER_HEX: Final = frozenset("0123456789abcdef")
_SEALED_ORIGIN_PREFIX: Final = "falsewake-sealed://experiment-003"
_FALSEWAKE_SOURCE_PREFIX: Final = "src/falsewake/"
_REQUIRED_SEALED_SOURCES: Final = frozenset(
    {
        "src/falsewake/__init__.py",
        "src/falsewake/experiment_002_run_authority.py",
        "src/falsewake/experiment_003_run_authority.py",
        "src/falsewake/experiment_003_coordinator.py",
        "src/falsewake/experiment_002_process_guard.py",
    }
)


class Experiment003RunnerError(RuntimeError):
    """The exact Experiment 003 bootstrap failed closed."""


class _ParentCoordinator(Protocol):
    def run_registered_experiment(self, registration: object, /) -> None:
        """Run the single registered parent experiment."""


class _GuardedChildCoordinator(Protocol):
    def _run_guarded_registered_seed_child(
        self,
        registration: object,
        /,
    ) -> None:
        """Run one sealed child through the fixed guarded route."""


@dataclass(frozen=True, slots=True)
class _BlobFrame:
    path: str
    offset: int
    byte_count: int


@dataclass(frozen=True, slots=True)
class _DescriptorFrame:
    stat_frame: tuple[int, ...]
    seals: int
    proc_target: str


@dataclass(frozen=True, slots=True)
class _ModuleFrame:
    fullname: str
    blob: _BlobFrame
    origin: str
    is_package: bool


@dataclass(frozen=True, slots=True)
class _ExecutedModule:
    module: ModuleType
    spec: importlib.machinery.ModuleSpec
    module_name: str
    module_package: str
    module_file: str
    spec_name: str
    spec_origin: str
    package_path: object | None
    package_path_frame: tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class _SealedChildBundle:
    """One structurally and cryptographically admitted FD7 container."""

    descriptor: int
    descriptor_frame: _DescriptorFrame
    descriptor_offset: int
    mapping: mmap.mmap
    head_commit: str
    implementation_commit: str
    registration_sha256: str
    source_bundle_sha256: str
    source_blobs: tuple[_BlobFrame, ...]
    frozen_blobs: tuple[_BlobFrame, ...]

    def close(self) -> None:
        """Close only the private mapping; descriptor ownership stays external."""

        if not self.mapping.closed:
            self.mapping.close()


class _SealedSourceFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """One-shot hard-deny meta finder and loader backed by the FD7 mmap."""

    get_data: Callable[[str], bytes]
    _verify_sealed_import_state: Callable[[], None]

    __slots__ = (
        "_attempted",
        "_bundle",
        "_executing",
        "_failed",
        "_loaded",
        "_lock",
        "_meta_path_frame",
        "_meta_path_registry",
        "_module_registry",
        "_modules",
        "_origins",
        "_sys_path_frame",
        "_sys_path_registry",
        "_verify_sealed_import_state",
        "get_data",
    )

    def __init__(self, bundle: _SealedChildBundle) -> None:
        if type(bundle) is not _SealedChildBundle:
            raise TypeError("sealed child bundle must have the exact internal type")
        modules, origins = _module_frames(bundle)
        self._bundle = bundle
        self._modules: Mapping[str, _ModuleFrame] = MappingProxyType(modules)
        self._origins: Mapping[str, _BlobFrame] = MappingProxyType(origins)
        self._attempted: set[str] = set()
        self._loaded: dict[str, _ExecutedModule] = {}
        self._executing: dict[str, _ExecutedModule] = {}
        self._failed = False
        self._lock = threading.RLock()
        self._meta_path_frame: tuple[object, ...] | None = None
        self._meta_path_registry: object | None = None
        self._module_registry: object | None = None
        self._sys_path_frame: tuple[str, ...] | None = None
        self._sys_path_registry: object | None = None

        def get_data(origin: str, /) -> bytes:
            with self._lock:
                try:
                    self._require_active()
                    if type(origin) is not str:
                        self._poison("sealed loader origin must be an exact string")
                    blob = self._origins.get(origin)
                    if blob is None:
                        self._poison("sealed loader origin is not registered")
                    return self._blob_bytes(blob)
                except BaseException:
                    self._failed = True
                    raise

        def verify() -> None:
            with self._lock:
                try:
                    self._verify_locked()
                except BaseException:
                    self._failed = True
                    raise

        # Stable closure-valued routes let the authority capture and compare
        # callable identity instead of receiving a fresh bound method per lookup.
        self.get_data = get_data
        self._verify_sealed_import_state = verify

    def _bind_import_surface(self) -> None:
        with self._lock:
            if (
                self._meta_path_frame is not None
                or self._meta_path_registry is not None
                or self._module_registry is not None
                or self._sys_path_frame is not None
                or self._sys_path_registry is not None
                or type(sys.modules) is not dict
                or type(sys.meta_path) is not list
                or type(sys.path) is not list
                or any(type(item) is not str for item in sys.path)
                or not sys.meta_path
                or sys.meta_path[0] is not self
            ):
                self._poison("sealed import surface cannot be rebound")
            self._meta_path_frame = tuple(sys.meta_path)
            self._meta_path_registry = sys.meta_path
            self._module_registry = sys.modules
            self._sys_path_frame = tuple(sys.path)
            self._sys_path_registry = sys.path

    def _poison(self, message: str) -> NoReturn:
        self._failed = True
        raise Experiment003RunnerError(message)

    def _require_active(self) -> None:
        if self._failed:
            raise Experiment003RunnerError("sealed source loader is terminally failed")
        meta_path_frame = self._meta_path_frame
        sys_path_frame = self._sys_path_frame
        if (
            meta_path_frame is None
            or self._meta_path_registry is not sys.meta_path
            or type(sys.meta_path) is not list
            or len(sys.meta_path) != len(meta_path_frame)
            or any(
                observed is not expected
                for observed, expected in zip(
                    sys.meta_path,
                    meta_path_frame,
                    strict=True,
                )
            )
            or not sys.meta_path
            or sys.meta_path[0] is not self
        ):
            self._poison("sealed source loader is not first in meta_path")
        if (
            sys_path_frame is None
            or self._sys_path_registry is not sys.path
            or type(sys.path) is not list
            or len(sys.path) != len(sys_path_frame)
            or any(
                type(observed) is not str or observed is not expected
                for observed, expected in zip(
                    sys.path,
                    sys_path_frame,
                    strict=True,
                )
            )
        ):
            self._poison("sealed child sys.path changed")
        if self._module_registry is not sys.modules or type(sys.modules) is not dict:
            self._poison("sealed child module registry changed")
        _require_sealed_child_bundle_stable(self._bundle)

    def _verify_locked(self) -> None:
        self._require_active()
        module_keys = tuple(sys.modules)
        if any(type(name) is not str for name in module_keys):
            self._poison("module registry contains a non-exact string key")
        falsewake_names = {
            name
            for name in module_keys
            if name == "falsewake" or name.startswith("falsewake.")
        }
        expected_names = set(self._loaded) | set(self._executing)
        if falsewake_names != expected_names:
            self._poison("falsewake import registry is not exactly sealed")
        for name, executed in (*self._loaded.items(), *self._executing.items()):
            module = executed.module
            frame = self._modules.get(name)
            if frame is None:
                self._poison("sealed module identity or origin changed")
            spec = getattr(module, "__spec__", None)
            package_path = getattr(module, "__path__", None)
            module_name = getattr(module, "__name__", None)
            module_package = getattr(module, "__package__", None)
            module_file = getattr(module, "__file__", None)
            module_dict = getattr(module, "__dict__", None)
            expected_package = name if frame.is_package else name.rpartition(".")[0]
            if (
                sys.modules.get(name) is not module
                or type(module) is not ModuleType
                or type(module_dict) is not dict
                or type(module_name) is not str
                or module_name is not executed.module_name
                or type(module_package) is not str
                or module_package is not executed.module_package
                or getattr(module, "__loader__", None) is not self
                or type(module_file) is not str
                or module_file is not executed.module_file
                or spec is not executed.spec
                or type(spec) is not importlib.machinery.ModuleSpec
                or type(spec.name) is not str
                or spec.name is not executed.spec_name
                or spec.loader is not self
                or type(spec.origin) is not str
                or spec.origin is not executed.spec_origin
                or spec.has_location is not True
                or type(executed.module_name) is not str
                or executed.module_name is not name
                or type(executed.module_package) is not str
                or executed.module_package != expected_package
                or type(executed.module_file) is not str
                or executed.module_file is not frame.origin
                or type(executed.spec_name) is not str
                or executed.spec_name is not executed.module_name
                or type(executed.spec_origin) is not str
                or executed.spec_origin is not frame.origin
                or module_name is not spec.name
                or module_file is not spec.origin
                or spec.submodule_search_locations is not executed.package_path
                or package_path is not executed.package_path
            ):
                self._poison("sealed module identity or origin changed")
            if frame.is_package:
                if (
                    type(package_path) is not list
                    or package_path
                    or executed.package_path_frame != ()
                ):
                    self._poison("sealed module identity or origin changed")
            elif (
                package_path is not None
                or "__path__" in cast(dict[str, object], module_dict)
                or executed.package_path is not None
                or executed.package_path_frame is not None
                or spec.submodule_search_locations is not None
            ):
                self._poison("sealed module identity or origin changed")

    def _blob_bytes(self, blob: _BlobFrame) -> bytes:
        if type(blob) is not _BlobFrame:
            self._poison("sealed source blob frame changed")
        return self._bundle.mapping[blob.offset : blob.offset + blob.byte_count]

    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: object = None,
    ) -> Any:
        del path
        with self._lock:
            try:
                self._require_active()
                if type(fullname) is not str:
                    self._poison("sealed loader module name must be an exact string")
                if not (fullname == "falsewake" or fullname.startswith("falsewake.")):
                    return None
                if target is not None:
                    self._poison("sealed falsewake modules cannot be reloaded")
                if fullname in self._attempted:
                    self._poison("sealed falsewake module execution cannot be retried")
                frame = self._modules.get(fullname)
                if frame is None:
                    raise ModuleNotFoundError(
                        f"sealed source bundle has no module {fullname!r}",
                        name=fullname,
                    )
                spec = importlib.util.spec_from_loader(
                    fullname,
                    self,
                    origin=frame.origin,
                    is_package=frame.is_package,
                )
                if (
                    type(spec) is not importlib.machinery.ModuleSpec
                    or type(spec.name) is not str
                    or spec.name is not fullname
                    or spec.loader is not self
                    or type(spec.origin) is not str
                    or spec.origin is not frame.origin
                    or (
                        type(spec.submodule_search_locations) is not list
                        or spec.submodule_search_locations
                        if frame.is_package
                        else spec.submodule_search_locations is not None
                    )
                ):
                    self._poison("sealed module spec could not be constructed exactly")
                spec.has_location = True
                if spec.has_location is not True:
                    self._poison("sealed module spec location could not be fixed")
                return spec
            except BaseException:
                self._failed = True
                raise

    def create_module(self, spec: object) -> None:
        with self._lock:
            try:
                self._require_active()
                if (
                    type(spec) is not importlib.machinery.ModuleSpec
                    or spec.loader is not self
                ):
                    self._poison("sealed loader received a forged module spec")
                return None
            except BaseException:
                self._failed = True
                raise

    def exec_module(self, module: ModuleType) -> None:
        with self._lock:
            try:
                self._require_active()
                if type(module) is not ModuleType:
                    self._poison("sealed loader received an invalid module")
                module_name = getattr(module, "__name__", None)
                module_package = getattr(module, "__package__", None)
                module_file = getattr(module, "__file__", None)
                module_dict = getattr(module, "__dict__", None)
                spec = getattr(module, "__spec__", None)
                if type(module_name) is not str:
                    self._poison("sealed loader received an invalid module")
                if type(spec) is not importlib.machinery.ModuleSpec:
                    self._poison("sealed module execution is forged or repeated")
                module_spec: importlib.machinery.ModuleSpec = spec
                frame = self._modules.get(module_name)
                if frame is None:
                    self._poison("sealed module execution is forged or repeated")
                expected_package = (
                    module_name if frame.is_package else module_name.rpartition(".")[0]
                )
                if (
                    module_name in self._attempted
                    or type(module_dict) is not dict
                    or type(module_package) is not str
                    or module_package != expected_package
                    or getattr(module, "__loader__", None) is not self
                    or type(module_file) is not str
                    or type(module_spec.name) is not str
                    or module_spec.name is not module_name
                    or module_spec.loader is not self
                    or type(module_spec.origin) is not str
                    or module_spec.origin is not frame.origin
                    or module_file is not module_spec.origin
                    or getattr(module, "__spec__", None) is not module_spec
                    or module_spec.has_location is not True
                ):
                    self._poison("sealed module execution is forged or repeated")
                package_path = getattr(module, "__path__", None)
                if frame.is_package:
                    if (
                        type(package_path) is not list
                        or package_path
                        or module_spec.submodule_search_locations is not package_path
                    ):
                        self._poison("sealed package path has an invalid type")
                    package_path_frame: tuple[str, ...] | None = ()
                else:
                    if (
                        package_path is not None
                        or "__path__" in cast(dict[str, object], module_dict)
                        or module_spec.submodule_search_locations is not None
                    ):
                        self._poison("sealed nonpackage exposes a package path")
                    package_path_frame = None
                executed = _ExecutedModule(
                    module=module,
                    spec=module_spec,
                    module_name=module_name,
                    module_package=module_package,
                    module_file=module_file,
                    spec_name=module_spec.name,
                    spec_origin=module_spec.origin,
                    package_path=package_path,
                    package_path_frame=package_path_frame,
                )
                self._attempted.add(module_name)
                self._executing[module_name] = executed
                view = memoryview(self._bundle.mapping)[
                    frame.blob.offset : frame.blob.offset + frame.blob.byte_count
                ]
                try:
                    code = compile(
                        cast(Any, view),
                        frame.origin,
                        "exec",
                        dont_inherit=True,
                        optimize=sys.flags.optimize,
                    )
                finally:
                    view.release()
                exec(code, module.__dict__)
                self._verify_locked()
                del self._executing[module_name]
                self._loaded[module_name] = executed
                self._verify_locked()
            except BaseException:
                self._failed = True
                raise

    def get_filename(self, fullname: str, /) -> str:
        with self._lock:
            try:
                self._require_active()
                if type(fullname) is not str:
                    self._poison("sealed loader module name must be an exact string")
                frame = self._modules.get(fullname)
                if frame is None:
                    self._poison("sealed loader filename is not registered")
                return frame.origin
            except BaseException:
                self._failed = True
                raise

    def is_package(self, fullname: str, /) -> bool:
        with self._lock:
            try:
                self._require_active()
                if type(fullname) is not str:
                    self._poison("sealed loader module name must be an exact string")
                frame = self._modules.get(fullname)
                if frame is None:
                    self._poison("sealed loader package name is not registered")
                return frame.is_package
            except BaseException:
                self._failed = True
                raise


def _fail(message: str) -> NoReturn:
    raise Experiment003RunnerError(message)


def _require_exact_sequence(
    observed: object,
    expected: tuple[str, ...],
    name: str,
) -> None:
    if type(observed) is not list or type(expected) is not tuple:
        _fail(f"{name} is not the exact registered sequence")
    values = cast(list[object], observed)
    if len(values) != len(expected) or any(
        type(value) is not str or type(registered) is not str or value != registered
        for value, registered in zip(values, expected, strict=True)
    ):
        _fail(f"{name} is not the exact registered sequence")


def _require_clean_bootstrap_imports(module_names: object) -> None:
    if type(module_names) is not tuple or any(
        type(name) is not str for name in cast(tuple[object, ...], module_names)
    ):
        _fail("bootstrap module-name snapshot is invalid")
    forbidden = sorted(
        name
        for name in cast(tuple[str, ...], module_names)
        if name.partition(".")[0] in _FORBIDDEN_BOOTSTRAP_MODULE_ROOTS
    )
    if forbidden:
        _fail(f"forbidden module loaded before authorization: {forbidden[0]}")


def _validate_bootstrap() -> None:
    if type(__file__) is not str or __file__ != _CANONICAL_FILE:
        _fail("runner __file__ is not the canonical entrypoint")
    if os.path.abspath(__file__) != _CANONICAL_FILE:
        _fail("runner __file__ is not absolute and canonical")
    if os.path.realpath(__file__) != _CANONICAL_FILE:
        _fail("runner entrypoint resolves outside its canonical path")
    if os.getcwd() != _REPOSITORY_ROOT:
        _fail("runner cwd is not the canonical repository root")
    if os.path.realpath(os.getcwd()) != _REPOSITORY_ROOT:
        _fail("runner cwd resolves outside the canonical repository root")
    if sys.executable != _PYTHON_EXECUTABLE:
        _fail("runner Python executable is not the frozen interpreter")
    if tuple(sys.version_info[:3]) != (3, 12, 3):
        _fail("runner Python version is not 3.12.3")
    _require_exact_sequence(sys.argv, _EXPECTED_ARGV, "argv")
    _require_exact_sequence(sys.orig_argv, _EXPECTED_ORIG_ARGV, "orig_argv")
    _require_exact_sequence(sys.path, _BOOTSTRAP_SYS_PATH, "bootstrap sys.path")
    if dict(os.environ) != _EXPECTED_ENVIRONMENT:
        _fail("runner environment is not the exact registered environment")
    for flag_name, expected in _EXPECTED_FLAGS.items():
        if getattr(sys.flags, flag_name, None) != expected:
            _fail(f"runner Python flag is not registered: {flag_name}")
    _require_clean_bootstrap_imports(tuple(sys.modules))


def _classify_execution_role() -> str:
    """Classify descriptor 3 once; only exact EBADF selects the parent."""

    try:
        descriptor_flags = fcntl.fcntl(_CHILD_CONTROL_FD, fcntl.F_GETFD)
    except OSError as error:
        if error.errno == errno.EBADF:
            return _PARENT_ROLE
        raise Experiment003RunnerError(
            "child control descriptor 3 could not be classified"
        ) from error
    if type(descriptor_flags) is not int or descriptor_flags != fcntl.FD_CLOEXEC:
        _fail("open child control descriptor 3 is not exact close-on-exec")
    return _CHILD_ROLE


def _activate_registered_sys_path() -> None:
    sys.path[:] = _REGISTERED_SYS_PATH
    _require_exact_sequence(sys.path, _REGISTERED_SYS_PATH, "registered sys.path")


def _issue_registration() -> object:
    authority = importlib.import_module(_AUTHORITY_MODULE)
    issuer = cast(
        Callable[[], object],
        authority.verify_and_issue_experiment_003_run_registration,
    )
    return issuer()


def _run_coordinator(registration: object) -> None:
    coordinator = cast(
        _ParentCoordinator,
        importlib.import_module(_COORDINATOR_MODULE),
    )
    coordinator.run_registered_experiment(registration)


def _run_registered_parent() -> None:
    _activate_registered_sys_path()
    registration = _issue_registration()
    _run_coordinator(registration)


def _stat_frame(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_blocks,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _inspect_source_bundle_descriptor(descriptor: int) -> tuple[_DescriptorFrame, int]:
    if type(descriptor) is not int or descriptor != _CHILD_SOURCE_BUNDLE_FD:
        _fail("source bundle is not fixed descriptor 7")
    try:
        before = os.fstat(descriptor)
        status_flags = fcntl.fcntl(descriptor, fcntl.F_GETFL)
        descriptor_flags = fcntl.fcntl(descriptor, fcntl.F_GETFD)
        seals = fcntl.fcntl(descriptor, fcntl.F_GET_SEALS)
        proc_target = os.readlink(f"/proc/self/fd/{descriptor}")
        descriptor_offset = os.lseek(descriptor, 0, os.SEEK_CUR)
        after = os.fstat(descriptor)
    except (OSError, ValueError) as error:
        raise Experiment003RunnerError(
            "source bundle descriptor 7 cannot be inspected"
        ) from error
    frame = _DescriptorFrame(
        stat_frame=_stat_frame(before),
        seals=seals,
        proc_target=proc_target,
    )
    if _stat_frame(after) != frame.stat_frame:
        _fail("source bundle descriptor changed while being inspected")
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 0
        or before.st_uid != os.geteuid()
        or before.st_gid != os.getegid()
        or stat.S_IMODE(before.st_mode) != 0o400
        or not 0 < before.st_size <= _MAX_CHILD_BUNDLE_BYTES
    ):
        _fail("source bundle is not an exact bounded anonymous regular file")
    if type(status_flags) is not int or status_flags & os.O_ACCMODE != os.O_RDONLY:
        _fail("source bundle descriptor is not read-only")
    if type(descriptor_flags) is not int or descriptor_flags != fcntl.FD_CLOEXEC:
        _fail("source bundle descriptor is not exact close-on-exec")
    if type(seals) is not int or seals != _SOURCE_BUNDLE_SEALS:
        _fail("source bundle descriptor lacks the exact byte seals")
    if (
        type(proc_target) is not str
        or not proc_target.startswith("/memfd:")
        or not proc_target.endswith(" (deleted)")
    ):
        _fail("source bundle descriptor is not an anonymous Linux memfd")
    if type(descriptor_offset) is not int or descriptor_offset < 0:
        _fail("source bundle descriptor offset is invalid")
    return frame, descriptor_offset


def _require_sealed_child_bundle_stable(bundle: _SealedChildBundle) -> None:
    if type(bundle) is not _SealedChildBundle or bundle.mapping.closed:
        _fail("sealed child bundle is closed or forged")
    frame, descriptor_offset = _inspect_source_bundle_descriptor(bundle.descriptor)
    if (
        frame != bundle.descriptor_frame
        or descriptor_offset != bundle.descriptor_offset
        or len(bundle.mapping) != frame.stat_frame[6]
    ):
        _fail("sealed child bundle descriptor or offset changed")


def _decode_fixed_ascii(value: bytes, name: str) -> str:
    try:
        return value.decode("ascii", errors="strict")
    except UnicodeDecodeError as error:
        raise Experiment003RunnerError(f"{name} is not ASCII") from error


def _require_lower_hex(value: object, length: int, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != length
        or any(character not in _LOWER_HEX for character in value)
    ):
        _fail(f"{name} must be exactly {length} lowercase hexadecimal characters")
    return value


def _require_relative_path(value: object, name: str) -> str:
    if type(value) is not str:
        _fail(f"{name} must contain exact strings")
    path = value
    try:
        path_bytes = path.encode("utf-8")
    except UnicodeEncodeError as error:
        raise Experiment003RunnerError(f"{name} path is not UTF-8") from error
    if (
        not path
        or len(path_bytes) > 4_096
        or path.startswith("/")
        or path.endswith("/")
        or "//" in path
        or "\\" in path
        or "\x00" in path
        or any(ord(character) < 0x21 or ord(character) > 0x7E for character in path)
    ):
        _fail(f"{name} contains an invalid path")
    parsed = PurePosixPath(path)
    if parsed.as_posix() != path or any(part in {".", ".."} for part in parsed.parts):
        _fail(f"{name} contains path traversal")
    if parsed.parts[0] == ".git":
        _fail(f"{name} must not enter .git")
    return path


def _require_path_list(value: object, name: str) -> tuple[str, ...]:
    if type(value) is not list:
        _fail(f"{name} must be an exact JSON array")
    paths = tuple(_require_relative_path(item, name) for item in value)
    if paths != tuple(sorted(paths, key=lambda item: item.encode("utf-8"))):
        _fail(f"{name} must be UTF-8 byte sorted")
    if len(set(paths)) != len(paths):
        _fail(f"{name} must not contain duplicates")
    return paths


def _parse_blob_frames(
    mapping: mmap.mmap,
    *,
    offset: int,
    byte_count: int,
    name: str,
    maximum_files: int,
    maximum_file_bytes: int,
) -> tuple[_BlobFrame, ...]:
    end = offset + byte_count
    if (
        type(offset) is not int
        or type(byte_count) is not int
        or byte_count < 4
        or offset < 0
        or end > len(mapping)
    ):
        _fail(f"{name} payload byte count is invalid")
    file_count = struct.unpack_from("<I", mapping, offset)[0]
    if file_count < 1 or file_count > maximum_files:
        _fail(f"{name} file count is invalid")
    cursor = offset + 4
    previous_path_bytes: bytes | None = None
    result: list[_BlobFrame] = []
    for _ in range(file_count):
        if end - cursor < 4:
            _fail(f"{name} path frame is truncated")
        path_byte_count = struct.unpack_from("<I", mapping, cursor)[0]
        cursor += 4
        if (
            path_byte_count < 1
            or path_byte_count > 4_096
            or path_byte_count > end - cursor
        ):
            _fail(f"{name} path byte count is invalid")
        path_bytes = mapping[cursor : cursor + path_byte_count]
        cursor += path_byte_count
        try:
            path = path_bytes.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise Experiment003RunnerError(
                f"{name} path is not strict UTF-8"
            ) from error
        _require_relative_path(path, f"{name} path")
        if previous_path_bytes is not None and path_bytes <= previous_path_bytes:
            _fail(f"{name} paths are duplicated or not UTF-8 sorted")
        previous_path_bytes = path_bytes
        if end - cursor < 8:
            _fail(f"{name} blob frame is truncated")
        blob_byte_count = struct.unpack_from("<Q", mapping, cursor)[0]
        cursor += 8
        if blob_byte_count > maximum_file_bytes or blob_byte_count > end - cursor:
            _fail(f"{name} blob byte count is invalid")
        result.append(_BlobFrame(path=path, offset=cursor, byte_count=blob_byte_count))
        cursor += blob_byte_count
    if cursor != end:
        _fail(f"{name} payload has trailing bytes")
    return tuple(result)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate registration key: {key}")
        result[key] = value
    return result


def _reject_json_number(value: str) -> NoReturn:
    raise ValueError(f"non-integer registration number is forbidden: {value}")


def _parse_registration_bindings(
    raw: bytes,
    *,
    implementation_commit: str,
    source_payload_byte_count: int,
    source_bundle_sha256: str,
    source_paths: tuple[str, ...],
) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise Experiment003RunnerError(
            "run registration is not strict UTF-8"
        ) from error
    if text.startswith("\ufeff"):
        _fail("run registration has a UTF-8 BOM")
    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_number,
            parse_float=_reject_json_number,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise Experiment003RunnerError("run registration is not exact JSON") from error
    if type(parsed) is not dict:
        _fail("run registration must be a JSON object")
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
        _fail("run registration is not canonical indented ASCII JSON plus LF")
    expected_keys = {
        "experiment",
        "external_inputs",
        "frozen_bindings",
        "implementation_commit",
        "invocation",
        "runtime",
        "schema_version",
        "source_bundle",
    }
    if set(document) != expected_keys or any(type(key) is not str for key in document):
        _fail("run registration fields are not exact")
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        _fail("run registration schema version is invalid")
    if type(document["experiment"]) is not str or document["experiment"] != "003":
        _fail("run registration experiment is invalid")
    if document["implementation_commit"] != implementation_commit:
        _fail("run registration implementation binding is invalid")
    for key in ("external_inputs", "frozen_bindings", "invocation", "runtime"):
        if type(document[key]) is not dict:
            _fail(f"run registration {key} binding is not an object")
    source = document["source_bundle"]
    if type(source) is not dict:
        _fail("run registration source_bundle is not an object")
    source_object = cast(dict[str, Any], source)
    if set(source_object) != {"import_roots", "paths", "payload_byte_count", "sha256"}:
        _fail("run registration source_bundle fields are not exact")
    import_roots = _require_path_list(source_object["import_roots"], "import roots")
    registered_paths = _require_path_list(source_object["paths"], "source paths")
    if not import_roots or not registered_paths:
        _fail("run registration source coverage is empty")
    seen_roots: set[str] = set()
    for root in import_roots:
        parts = PurePosixPath(root).parts
        if any("/".join(parts[:depth]) in seen_roots for depth in range(1, len(parts))):
            _fail("run registration import roots overlap")
        seen_roots.add(root)
    if (
        type(source_object["payload_byte_count"]) is not int
        or source_object["payload_byte_count"] != source_payload_byte_count
        or source_object["sha256"] != source_bundle_sha256
        or registered_paths != source_paths
    ):
        _fail("run registration source bundle binding is invalid")
    return document


def _sha256_mapping_slice(
    mapping: mmap.mmap,
    offset: int,
    byte_count: int,
    *,
    domain: bytes = b"",
) -> str:
    hasher = hashlib.sha256()
    hasher.update(domain)
    view = memoryview(mapping)[offset : offset + byte_count]
    try:
        hasher.update(view)
    finally:
        view.release()
    return hasher.hexdigest()


def _parse_child_bundle(
    descriptor: int,
    descriptor_frame: _DescriptorFrame,
    descriptor_offset: int,
    mapping: mmap.mmap,
) -> _SealedChildBundle:
    if len(mapping) < _CHILD_BUNDLE_HEADER.size:
        _fail("child bundle header is truncated")
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
    ) = _CHILD_BUNDLE_HEADER.unpack_from(mapping)
    if magic != _CHILD_BUNDLE_MAGIC or version != _CHILD_BUNDLE_VERSION:
        _fail("child bundle magic or version is invalid")
    if not 0 < registration_byte_count <= _MAX_CONFIG_BYTES:
        _fail("child bundle registration byte count is invalid")
    if not 4 <= source_payload_byte_count <= _MAX_SOURCE_BUNDLE_BYTES:
        _fail("child bundle source payload byte count is invalid")
    if not 4 <= frozen_payload_byte_count <= _MAX_FROZEN_BUNDLE_BYTES:
        _fail("child bundle frozen payload byte count is invalid")
    expected_size = (
        _CHILD_BUNDLE_HEADER.size
        + registration_byte_count
        + source_payload_byte_count
        + frozen_payload_byte_count
    )
    if expected_size != len(mapping) or expected_size > _MAX_CHILD_BUNDLE_BYTES:
        _fail("child bundle is truncated, trailing, or over its aggregate limit")
    head_commit = _require_lower_hex(
        _decode_fixed_ascii(head_bytes, "child bundle HEAD"),
        40,
        "child bundle HEAD",
    )
    implementation_commit = _require_lower_hex(
        _decode_fixed_ascii(implementation_bytes, "child bundle implementation"),
        40,
        "child bundle implementation",
    )
    if head_commit == implementation_commit:
        _fail("child bundle HEAD and implementation commits are identical")
    registration_sha256 = _require_lower_hex(
        _decode_fixed_ascii(
            registration_sha256_bytes,
            "child bundle registration sha256",
        ),
        64,
        "child bundle registration sha256",
    )
    source_bundle_sha256 = _require_lower_hex(
        _decode_fixed_ascii(
            source_bundle_sha256_bytes,
            "child bundle source sha256",
        ),
        64,
        "child bundle source sha256",
    )
    registration_offset = _CHILD_BUNDLE_HEADER.size
    source_offset = registration_offset + registration_byte_count
    frozen_offset = source_offset + source_payload_byte_count
    registration_bytes = mapping[
        registration_offset : registration_offset + registration_byte_count
    ]
    if hashlib.sha256(registration_bytes).hexdigest() != registration_sha256:
        _fail("child bundle registration digest mismatch")
    observed_source_sha256 = _sha256_mapping_slice(
        mapping,
        source_offset,
        source_payload_byte_count,
        domain=_SOURCE_BUNDLE_DOMAIN,
    )
    if observed_source_sha256 != source_bundle_sha256:
        _fail("child bundle source digest mismatch")
    source_blobs = _parse_blob_frames(
        mapping,
        offset=source_offset,
        byte_count=source_payload_byte_count,
        name="source bundle",
        maximum_files=_MAX_SOURCE_FILES,
        maximum_file_bytes=_MAX_SOURCE_FILE_BYTES,
    )
    frozen_blobs = _parse_blob_frames(
        mapping,
        offset=frozen_offset,
        byte_count=frozen_payload_byte_count,
        name="frozen bundle",
        maximum_files=_MAX_FROZEN_FILES,
        maximum_file_bytes=_MAX_FROZEN_FILE_BYTES,
    )
    source_paths = tuple(blob.path for blob in source_blobs)
    if not _REQUIRED_SEALED_SOURCES.issubset(source_paths):
        _fail("child bundle lacks a required sealed bootstrap source")
    _parse_registration_bindings(
        registration_bytes,
        implementation_commit=implementation_commit,
        source_payload_byte_count=source_payload_byte_count,
        source_bundle_sha256=source_bundle_sha256,
        source_paths=source_paths,
    )
    return _SealedChildBundle(
        descriptor=descriptor,
        descriptor_frame=descriptor_frame,
        descriptor_offset=descriptor_offset,
        mapping=mapping,
        head_commit=head_commit,
        implementation_commit=implementation_commit,
        registration_sha256=registration_sha256,
        source_bundle_sha256=source_bundle_sha256,
        source_blobs=source_blobs,
        frozen_blobs=frozen_blobs,
    )


def _admit_sealed_child_bundle() -> _SealedChildBundle:
    descriptor = _CHILD_SOURCE_BUNDLE_FD
    frame, descriptor_offset = _inspect_source_bundle_descriptor(descriptor)
    try:
        mapping = mmap.mmap(
            descriptor,
            frame.stat_frame[6],
            access=mmap.ACCESS_READ,
        )
    except (OSError, ValueError) as error:
        raise Experiment003RunnerError(
            "source bundle could not be mapped read-only"
        ) from error
    try:
        bundle = _parse_child_bundle(
            descriptor,
            frame,
            descriptor_offset,
            mapping,
        )
        _require_sealed_child_bundle_stable(bundle)
        return bundle
    except BaseException:
        mapping.close()
        raise


def _module_name_from_path(path: str) -> tuple[str, bool] | None:
    if not path.startswith(_FALSEWAKE_SOURCE_PREFIX) or not path.endswith(".py"):
        return None
    relative = path[len(_FALSEWAKE_SOURCE_PREFIX) :]
    parts = relative.split("/")
    is_package = parts[-1] == "__init__.py"
    module_parts = parts[:-1] if is_package else [*parts[:-1], parts[-1][:-3]]
    if any(not part or not part.isidentifier() for part in module_parts):
        _fail("sealed source path is not an importable Python module")
    fullname = ".".join(("falsewake", *module_parts))
    return fullname, is_package


def _module_frames(
    bundle: _SealedChildBundle,
) -> tuple[dict[str, _ModuleFrame], dict[str, _BlobFrame]]:
    modules: dict[str, _ModuleFrame] = {}
    origins: dict[str, _BlobFrame] = {}
    for blob in bundle.source_blobs:
        origin = f"{_SEALED_ORIGIN_PREFIX}/{bundle.source_bundle_sha256}/{blob.path}"
        if origin in origins:
            _fail("sealed source origin is duplicated")
        origins[origin] = blob
        module_binding = _module_name_from_path(blob.path)
        if module_binding is None:
            continue
        fullname, is_package = module_binding
        if fullname in modules:
            _fail("sealed source maps multiple paths to one module")
        modules[fullname] = _ModuleFrame(
            fullname=fullname,
            blob=blob,
            origin=origin,
            is_package=is_package,
        )
    required_modules = {
        "falsewake",
        _AUTHORITY_MODULE,
        _COORDINATOR_MODULE,
    }
    if not required_modules.issubset(modules):
        _fail("sealed source map lacks a required bootstrap module")
    if not modules["falsewake"].is_package:
        _fail("sealed falsewake root is not a package")
    return modules, origins


def _install_sealed_source_finder(
    bundle: _SealedChildBundle,
) -> _SealedSourceFinder:
    _require_clean_bootstrap_imports(tuple(sys.modules))
    sys.path[:] = _RUNTIME_SYS_PATH_ROOTS
    _require_exact_sequence(sys.path, _RUNTIME_SYS_PATH_ROOTS, "sealed child sys.path")
    sys.path_importer_cache.clear()
    finder = _SealedSourceFinder(bundle)
    sys.meta_path.insert(0, finder)
    finder._bind_import_surface()
    finder._verify_sealed_import_state()
    return finder


def _run_sealed_registered_child() -> None:
    bundle = _admit_sealed_child_bundle()
    finder = _install_sealed_source_finder(bundle)

    package = importlib.import_module("falsewake")
    if type(package) is not ModuleType:
        _fail("sealed falsewake package has an invalid type")
    finder._verify_sealed_import_state()

    authority = importlib.import_module(_AUTHORITY_MODULE)
    finder._verify_sealed_import_state()
    issuer = getattr(authority, _SEALED_CHILD_ISSUER, None)
    if not callable(issuer):
        _fail("sealed child registration issuer is unavailable")
    registration = issuer()
    finder._verify_sealed_import_state()

    coordinator = cast(
        _GuardedChildCoordinator,
        importlib.import_module(_COORDINATOR_MODULE),
    )
    finder._verify_sealed_import_state()
    route = getattr(coordinator, _GUARDED_CHILD_ROUTE, None)
    if not callable(route):
        _fail("guarded sealed-child coordinator route is unavailable")
    result = route(registration)
    if result is not None:
        _fail("guarded sealed-child coordinator returned an unexpected value")
    finder._verify_sealed_import_state()


def main() -> None:
    """Validate and execute exactly one parent or sealed-child route."""

    _validate_bootstrap()
    role = _classify_execution_role()
    if role == _CHILD_ROLE:
        _run_sealed_registered_child()
        return
    if role != _PARENT_ROLE:
        _fail("runner execution role is invalid")
    _run_registered_parent()


if __name__ == "__main__":
    main()
