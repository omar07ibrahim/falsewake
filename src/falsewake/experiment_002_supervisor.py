"""Fail-closed parent resource kernel for Experiment 002 children.

The coordinator imports this module only on the registered parent path and uses
its private composition points so the process and filesystem controls remain
independently reviewable.

This kernel freezes only the immediate taskset, Python, and runner launch bytes.
Its trust claim stops before the runner's later project imports: the parent
authority provides those from a separately verified sealed bundle.
Pre-existing same-process callbacks and audit hooks are trusted not to mutate
protected FDs, change signal dispositions, or create processes.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import os
import signal
import socket
import stat
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import MemberDescriptorType
from typing import Final, NoReturn, Protocol, cast

from falsewake.experiment_002_child_result import (
    ChildResultBinding,
    VerifiedChildResult,
    load_registered_child_result,
    verify_verified_child_result,
)

__all__ = ("Experiment002SupervisorError",)

_REPOSITORY_ROOT: Final = "/home/ubuntu/gitcode/falsewake"
_TEMP_ROOT: Final = "/home/ubuntu/gitcode/.t"
_SCRATCH_ROOT: Final = f"{_TEMP_ROOT}/falsewake-experiment-002-scratch"
_STAGING_ROOT: Final = f"{_TEMP_ROOT}/falsewake-experiment-002-staging"
_TASKSET_EXECUTABLE: Final = "/usr/bin/taskset"
_TASKSET_SHA256: Final = (
    "a9c851792e54e91fba7b827019380abee54e715b6817899c835e4f221354b260"
)
_PYTHON_EXECUTABLE: Final = "/home/ubuntu/gitcode/.t/falsewake-venv/bin/python"
_PYTHON_LINK_CHAIN: Final = (
    ("/home/ubuntu/gitcode/.t/falsewake-venv/bin/python", "python3"),
    ("/home/ubuntu/gitcode/.t/falsewake-venv/bin/python3", "/usr/bin/python3"),
    ("/usr/bin/python3", "python3.12"),
)
_PYTHON_RESOLVED_EXECUTABLE: Final = "/usr/bin/python3.12"
_PYTHON_RESOLVED_SHA256: Final = (
    "1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118"
)
_RUNNER_ENTRYPOINT: Final = "src/falsewake/experiment_002_runner.py"
_RUNNER_PATH: Final = f"{_REPOSITORY_ROOT}/{_RUNNER_ENTRYPOINT}"
_RUNNER_SHA256: Final = (
    "3aaa25b54670c357d528887d582d39e5fb0b815edb86201a7eef793c92b4931d"
)
_CHILD_CONTROL_FD: Final = 3
_CHILD_RUNNER_FD: Final = 4
_CHILD_PYTHON_FD: Final = 5
_CHILD_TASKSET_FD: Final = 6
_CHILD_SOURCE_BUNDLE_FD: Final = 7
_SAFE_SOURCE_FD_MINIMUM: Final = 8
_LAUNCH_FILE_BYTES_MAXIMUM: Final = 64 << 20
_SOURCE_BUNDLE_BYTES_MAXIMUM: Final = 256 << 20
_LAUNCH_MEMFD_FLAGS: Final = os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING
_LAUNCH_MEMFD_SEALS: Final = (
    fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL
)
_SOURCE_BUNDLE_MEMFD_SEALS: Final = _LAUNCH_MEMFD_SEALS
_CPU_COUNT: Final = 2
_WALL_SECONDS_MAXIMUM: Final = 21_600
_WALL_NANOSECONDS_MAXIMUM: Final = _WALL_SECONDS_MAXIMUM * 1_000_000_000
_RSS_BYTES_MAXIMUM: Final = 8_589_934_592
_OUTPUT_BYTES_MAXIMUM: Final = 4_294_967_296
_POLL_INTERVAL_NANOSECONDS: Final = 25_000_000
_POLL_INTERVAL_MAXIMUM_NANOSECONDS: Final = 100_000_000
_TERMINATION_NANOSECONDS_MAXIMUM: Final = 10_000_000_000
_ACTIVATION_JOIN_SECONDS_MAXIMUM: Final = 35.0
_ACTIVATION_START_GRACE_SECONDS: Final = 0.05
_THREAD_RETIRE_NANOSECONDS_MAXIMUM: Final = 100_000_000
_PROC_FILE_BYTES_MAXIMUM: Final = 1 << 20
_OUTPUT_ENTRY_COUNT_MAXIMUM: Final = 65_536
_OUTPUT_DEPTH_MAXIMUM: Final = 32
_KIB: Final = 1_024
_DEVNULL: Final = "/dev/null"
_RUNNER_BOOTSTRAP: Final = (
    "import os,sys;"
    "os.set_inheritable(3,False);os.set_inheritable(7,False);"
    "os.close(6);os.close(5);"
    f"sys.executable={_PYTHON_EXECUTABLE!r};"
    f"sys.argv=[{_RUNNER_ENTRYPOINT!r}];"
    "sys.orig_argv=[sys.executable,'-I','-S','-B',sys.argv[0]];"
    f"_p={_RUNNER_PATH!r};"
    "_s=open(4,'rb',closefd=True);_b=_s.read();_s.close();"
    "_g={'__name__':'__main__','__file__':_p,'__package__':None,"
    "'__spec__':None,'__builtins__':__builtins__};"
    "exec(compile(_b,_p,'exec'),_g)"
)
_REGISTERED_ENVIRONMENT_ITEMS: Final = (
    ("LANG", "C"),
    ("LC_ALL", "C"),
    ("MKL_DYNAMIC", "FALSE"),
    ("MKL_NUM_THREADS", "1"),
    ("OMP_DYNAMIC", "FALSE"),
    ("OMP_NUM_THREADS", "2"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("PYTHONDONTWRITEBYTECODE", "1"),
    ("PYTHONHASHSEED", "0"),
    ("PYTHONNOUSERSITE", "1"),
    ("TMPDIR", _SCRATCH_ROOT),
    ("TZ", "UTC"),
)

type _CloseAction = tuple[int, int]
type _Dup2Action = tuple[int, int, int]
type _FileAction = _CloseAction | _Dup2Action
type _Activation = Callable[[socket.socket, int], None]
type _SignalDispositionFrame = tuple[tuple[int, int], ...]


class _SigSet(ctypes.Structure):
    _fields_ = (("values", ctypes.c_ulong * 16),)


class _SchedParam(ctypes.Structure):
    _fields_ = (("priority", ctypes.c_int),)


class _SpawnAttributes(ctypes.Structure):
    _fields_ = (
        ("flags", ctypes.c_short),
        ("process_group", ctypes.c_int),
        ("signal_default", _SigSet),
        ("signal_mask", _SigSet),
        ("scheduler_parameters", _SchedParam),
        ("scheduler_policy", ctypes.c_int),
        ("cgroup", ctypes.c_int),
        ("padding", ctypes.c_int * 15),
    )


class _SpawnFileActions(ctypes.Structure):
    _fields_ = (
        ("allocated", ctypes.c_int),
        ("used", ctypes.c_int),
        ("actions", ctypes.c_void_p),
        ("padding", ctypes.c_int * 16),
    )


_LIBC: Final = ctypes.CDLL(
    "/lib/x86_64-linux-gnu/libc.so.6",
    use_errno=True,
)
_LIBC.posix_spawn_file_actions_init.argtypes = (ctypes.POINTER(_SpawnFileActions),)
_LIBC.posix_spawn_file_actions_init.restype = ctypes.c_int
_LIBC.posix_spawn_file_actions_destroy.argtypes = (ctypes.POINTER(_SpawnFileActions),)
_LIBC.posix_spawn_file_actions_destroy.restype = ctypes.c_int
_LIBC.posix_spawn_file_actions_addclose.argtypes = (
    ctypes.POINTER(_SpawnFileActions),
    ctypes.c_int,
)
_LIBC.posix_spawn_file_actions_addclose.restype = ctypes.c_int
_LIBC.posix_spawn_file_actions_adddup2.argtypes = (
    ctypes.POINTER(_SpawnFileActions),
    ctypes.c_int,
    ctypes.c_int,
)
_LIBC.posix_spawn_file_actions_adddup2.restype = ctypes.c_int
_LIBC.posix_spawn_file_actions_addclosefrom_np.argtypes = (
    ctypes.POINTER(_SpawnFileActions),
    ctypes.c_int,
)
_LIBC.posix_spawn_file_actions_addclosefrom_np.restype = ctypes.c_int
_LIBC.posix_spawnattr_init.argtypes = (ctypes.POINTER(_SpawnAttributes),)
_LIBC.posix_spawnattr_init.restype = ctypes.c_int
_LIBC.posix_spawnattr_destroy.argtypes = (ctypes.POINTER(_SpawnAttributes),)
_LIBC.posix_spawnattr_destroy.restype = ctypes.c_int
_LIBC.posix_spawnattr_setflags.argtypes = (
    ctypes.POINTER(_SpawnAttributes),
    ctypes.c_short,
)
_LIBC.posix_spawnattr_setflags.restype = ctypes.c_int
_LIBC.posix_spawnattr_setpgroup.argtypes = (
    ctypes.POINTER(_SpawnAttributes),
    ctypes.c_int,
)
_LIBC.posix_spawnattr_setpgroup.restype = ctypes.c_int
_LIBC.posix_spawnattr_setsigmask.argtypes = (
    ctypes.POINTER(_SpawnAttributes),
    ctypes.POINTER(_SigSet),
)
_LIBC.posix_spawnattr_setsigmask.restype = ctypes.c_int
_LIBC.sigemptyset.argtypes = (ctypes.POINTER(_SigSet),)
_LIBC.sigemptyset.restype = ctypes.c_int
_LIBC.posix_spawn.argtypes = (
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_char_p,
    ctypes.POINTER(_SpawnFileActions),
    ctypes.POINTER(_SpawnAttributes),
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.c_char_p),
)
_LIBC.posix_spawn.restype = ctypes.c_int
_POSIX_SPAWN_SETPGROUP: Final = 0x02
_POSIX_SPAWN_SETSIGMASK: Final = 0x08


class Experiment002SupervisorError(RuntimeError):
    """The registered child could not be supervised without ambiguity."""


@dataclass(frozen=True, slots=True)
class _Limits:
    wall_nanoseconds: int
    rss_bytes: int
    output_bytes: int
    poll_nanoseconds: int


@dataclass(frozen=True, slots=True)
class _LaunchPlan:
    repository_root: str
    temporary_root: str
    scratch_root: str
    staging_root: str
    taskset_executable: str
    python_executable: str
    runner_entrypoint: str
    runner_path: str
    environment_items: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class _WaitResult:
    pid: int
    status: int
    maximum_rss_kib: int


@dataclass(frozen=True, slots=True)
class _ProcessSample:
    rss_bytes: int | None
    child_pids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _SourceBundleFrame:
    metadata: tuple[int, ...]
    seals: int


@dataclass(slots=True)
class _PinnedParent:
    path: str
    descriptors: tuple[int, ...]
    identities: tuple[tuple[int, int, int], ...]
    components: tuple[str, ...]

    @property
    def descriptor(self) -> int:
        if not self.descriptors:
            return -1
        return self.descriptors[-1]

    def close(self) -> None:
        failures: list[OSError] = []
        descriptors = self.descriptors
        self.descriptors = ()
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError as error:
                failures.append(error)
        if failures:
            raise failures[0]


@dataclass(slots=True)
class _PinnedRoot:
    path: str
    parent: _PinnedParent
    descriptor: int
    identity: tuple[int, int, int]

    @property
    def parent_fd(self) -> int:
        return self.parent.descriptor

    def close(self) -> None:
        failures: list[OSError] = []
        if self.descriptor >= 0:
            descriptor = self.descriptor
            self.descriptor = -1
            try:
                os.close(descriptor)
            except OSError as error:
                failures.append(error)
        try:
            self.parent.close()
        except OSError as error:
            failures.append(error)
        if failures:
            raise failures[0]


type _RegisteredStagingLeaseFrame = tuple[
    _PinnedRoot,
    int,
    tuple[int, int, int],
    _PinnedParent,
    tuple[int, ...],
    tuple[tuple[int, int, int], ...],
    tuple[str, ...],
]


@dataclass(slots=True, eq=False)
class _RegisteredParentLifecycleState:
    """One stable identity containing every authorized lifecycle mutation."""

    owner_thread: int | None
    phase: int
    lease: _PinnedRoot | None
    lease_frame: _RegisteredStagingLeaseFrame | None
    namespace_owned: bool
    children_started: int
    children_completed: int
    cleanup_attempted: bool


@dataclass(frozen=True, slots=True)
class _ChildResourceMetrics:
    pid: int
    cpu_ids: tuple[int, int]
    elapsed_nanoseconds: int
    maximum_rss_bytes: int
    output_and_scratch_bytes: int


@dataclass(frozen=True, slots=True)
class _SupervisedChildResult:
    pid: int
    cpu_ids: tuple[int, int]
    elapsed_nanoseconds: int
    maximum_rss_bytes: int
    output_and_scratch_bytes: int
    verified_child_result: VerifiedChildResult

    def __post_init__(self) -> None:
        if type(self.verified_child_result) is not VerifiedChildResult:
            raise TypeError(
                "verified_child_result must be an exact VerifiedChildResult"
            )


@dataclass(slots=True)
class _ChildHandle:
    pid: int
    reaped: bool = False
    wait_result: _WaitResult | None = None


@dataclass(slots=True)
class _ActivationState:
    completed: threading.Event
    lock: threading.Lock
    error: BaseException | None = None


@dataclass(slots=True)
class _CleanupNode:
    name: str | None
    parent: _CleanupNode | None
    identity: tuple[int, int, int]
    expanded: bool = False


class _Kernel(Protocol):
    def parent_affinity(self) -> set[int]: ...

    def parent_child_pids(self) -> tuple[int, ...]: ...

    def process_affinity(self, pid: int) -> set[int]: ...

    def monotonic_ns(self) -> int: ...

    def sleep(self, seconds: float) -> None: ...

    def spawn(
        self,
        executable_path: str,
        argv: tuple[str, ...],
        environment: Mapping[str, str],
        file_actions: tuple[_FileAction, ...],
        source_bundle_fd: int,
    ) -> int: ...

    def wait4(self, pid: int, options: int) -> _WaitResult: ...

    def kill_process_group(self, pid: int) -> None: ...

    def kill_process(self, pid: int) -> None: ...

    def sample_process(self, pid: int) -> _ProcessSample: ...


_REGISTERED_LIMITS: Final = _Limits(
    wall_nanoseconds=_WALL_NANOSECONDS_MAXIMUM,
    rss_bytes=_RSS_BYTES_MAXIMUM,
    output_bytes=_OUTPUT_BYTES_MAXIMUM,
    poll_nanoseconds=_POLL_INTERVAL_NANOSECONDS,
)
_REGISTERED_PLAN: Final = _LaunchPlan(
    repository_root=_REPOSITORY_ROOT,
    temporary_root=_TEMP_ROOT,
    scratch_root=_SCRATCH_ROOT,
    staging_root=_STAGING_ROOT,
    taskset_executable=_TASKSET_EXECUTABLE,
    python_executable=_PYTHON_EXECUTABLE,
    runner_entrypoint=_RUNNER_ENTRYPOINT,
    runner_path=_RUNNER_PATH,
    environment_items=_REGISTERED_ENVIRONMENT_ITEMS,
)


def _fail(message: str) -> NoReturn:
    raise Experiment002SupervisorError(message)


def _snapshot_child_result_binding(
    binding: ChildResultBinding, /
) -> ChildResultBinding:
    """Validate and detach one expected result binding before any side effect."""

    if type(binding) is not ChildResultBinding:
        raise TypeError("result binding must be an exact ChildResultBinding")
    role = binding.role
    ordinal = binding.ordinal
    seed = binding.seed
    registration_head_commit = binding.registration_head_commit
    implementation_commit = binding.implementation_commit
    registration_sha256 = binding.registration_sha256
    source_bundle_sha256 = binding.source_bundle_sha256
    snapshot = ChildResultBinding(
        role=role,
        ordinal=ordinal,
        seed=seed,
        registration_head_commit=registration_head_commit,
        implementation_commit=implementation_commit,
        registration_sha256=registration_sha256,
        source_bundle_sha256=source_bundle_sha256,
    )
    second = (
        binding.role,
        binding.ordinal,
        binding.seed,
        binding.registration_head_commit,
        binding.implementation_commit,
        binding.registration_sha256,
        binding.source_bundle_sha256,
    )
    if (
        role,
        ordinal,
        seed,
        registration_head_commit,
        implementation_commit,
        registration_sha256,
        source_bundle_sha256,
    ) != second:
        _fail("result binding changed while it was snapshotted")
    return snapshot


def _require_exact_integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(f"{name} is not an integer at least {minimum}")
    return value


def _require_limits(limits: _Limits) -> None:
    if type(limits) is not _Limits:
        raise TypeError("limits must be an exact _Limits")
    _require_exact_integer(limits.wall_nanoseconds, "wall limit", minimum=1)
    _require_exact_integer(limits.rss_bytes, "RSS limit", minimum=1)
    _require_exact_integer(limits.output_bytes, "output limit", minimum=1)
    poll = _require_exact_integer(limits.poll_nanoseconds, "poll interval", minimum=1)
    if poll > _POLL_INTERVAL_MAXIMUM_NANOSECONDS:
        _fail("poll interval exceeds 100 milliseconds")


def _require_plan(plan: _LaunchPlan) -> None:
    if type(plan) is not _LaunchPlan:
        raise TypeError("launch plan must be an exact _LaunchPlan")
    string_fields = (
        plan.repository_root,
        plan.temporary_root,
        plan.scratch_root,
        plan.staging_root,
        plan.taskset_executable,
        plan.python_executable,
        plan.runner_entrypoint,
        plan.runner_path,
    )
    if any(
        type(value) is not str or not value or "\0" in value for value in string_fields
    ):
        _fail("launch plan contains an invalid string")
    for value in (
        plan.repository_root,
        plan.temporary_root,
        plan.scratch_root,
        plan.staging_root,
        plan.taskset_executable,
        plan.python_executable,
        plan.runner_path,
    ):
        if not os.path.isabs(value):
            _fail("launch plan path is not absolute")
    if os.path.isabs(plan.runner_entrypoint):
        _fail("runner entrypoint must be relative")
    expected_runner = os.path.join(plan.repository_root, plan.runner_entrypoint)
    if expected_runner != plan.runner_path:
        _fail("runner path and entrypoint disagree")
    expected_roots = {plan.scratch_root, plan.staging_root}
    if len(expected_roots) != 2:
        _fail("scratch and staging roots must be distinct")
    for root in expected_roots:
        if os.path.dirname(root) != plan.temporary_root:
            _fail("output root is outside the registered temporary root")
    if (
        type(plan.environment_items) is not tuple
        or not plan.environment_items
        or any(
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) is not str
            for item in plan.environment_items
        )
    ):
        _fail("registered environment items are invalid")
    keys = tuple(item[0] for item in plan.environment_items)
    if len(keys) != len(set(keys)):
        _fail("registered environment contains a duplicate key")
    expected_environment_items = tuple(
        (key, plan.scratch_root if key == "TMPDIR" else value)
        for key, value in _REGISTERED_ENVIRONMENT_ITEMS
    )
    if plan.environment_items != expected_environment_items:
        _fail("registered environment differs from the fixed environment")


def _capture_two_lowest_cpu_ids(kernel: _Kernel) -> tuple[int, int]:
    observed = kernel.parent_affinity()
    if type(observed) is not set or any(
        type(value) is not int or value < 0 for value in observed
    ):
        _fail("parent affinity is not an exact set of nonnegative CPU IDs")
    ordered = sorted(observed)
    if len(ordered) < _CPU_COUNT:
        _fail("fewer than two logical CPUs are available")
    selected = (ordered[0], ordered[1])
    if selected[0] == selected[1]:
        _fail("captured CPU IDs are not distinct")
    return selected


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


def _identity_frame(value: os.stat_result) -> tuple[int, int, int]:
    return value.st_dev, value.st_ino, stat.S_IFMT(value.st_mode)


def _open_validated_regular_path(
    path: str,
    *,
    executable: bool,
    expected_sha256: str | None = None,
) -> int:
    if type(path) is not str or not os.path.isabs(path) or "\0" in path:
        _fail("fixed launch path is not an absolute string")
    try:
        before = os.lstat(path)
    except OSError as error:
        raise Experiment002SupervisorError(
            f"fixed launch path is inaccessible: {path}"
        ) from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        _fail(f"fixed launch path is not a regular nonsymlink file: {path}")
    if os.path.realpath(path) != path:
        _fail(f"fixed launch path contains symlink ambiguity: {path}")
    if executable and before.st_mode & 0o111 == 0:
        _fail(f"fixed executable is not executable: {path}")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise Experiment002SupervisorError(
            f"fixed launch path cannot be opened: {path}"
        ) from error
    try:
        opened = os.fstat(descriptor)
        digest: str | None = None
        if expected_sha256 is not None:
            hasher = hashlib.sha256()
            while True:
                try:
                    chunk = os.read(descriptor, 1 << 20)
                except InterruptedError:
                    continue
                if not chunk:
                    break
                hasher.update(chunk)
            digest = hasher.hexdigest()
        after = os.lstat(path)
        if (
            not stat.S_ISREG(opened.st_mode)
            or _stat_frame(before) != _stat_frame(opened)
            or _stat_frame(opened) != _stat_frame(after)
        ):
            _fail(f"fixed launch path changed during validation: {path}")
        if expected_sha256 is not None and digest != expected_sha256:
            _fail(f"fixed launch path digest is not registered: {path}")
        os.lseek(descriptor, 0, os.SEEK_SET)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _validate_regular_path(
    path: str,
    *,
    executable: bool,
    expected_sha256: str | None = None,
) -> None:
    descriptor = _open_validated_regular_path(
        path,
        executable=executable,
        expected_sha256=expected_sha256,
    )
    os.close(descriptor)


def _sealed_launch_snapshot(
    descriptor: int,
    *,
    executable: bool,
    expected_sha256: str | None,
) -> int:
    if type(descriptor) is not int or descriptor < 0:
        _fail("launch snapshot source descriptor is invalid")
    source_before = os.fstat(descriptor)
    if (
        not stat.S_ISREG(source_before.st_mode)
        or source_before.st_size < 0
        or source_before.st_size > _LAUNCH_FILE_BYTES_MAXIMUM
    ):
        _fail("launch snapshot source is not a bounded regular file")
    snapshot = -1
    try:
        snapshot = os.memfd_create(
            "falsewake-experiment-002-launch",
            _LAUNCH_MEMFD_FLAGS,
        )
        os.lseek(descriptor, 0, os.SEEK_SET)
        copied_hasher = hashlib.sha256()
        copied_bytes = 0
        while True:
            try:
                chunk = os.read(descriptor, 1 << 20)
            except InterruptedError:
                continue
            if not chunk:
                break
            copied_bytes += len(chunk)
            if copied_bytes > _LAUNCH_FILE_BYTES_MAXIMUM:
                _fail("launch snapshot source exceeds its byte limit")
            copied_hasher.update(chunk)
            remaining = memoryview(chunk)
            while remaining:
                try:
                    written = os.write(snapshot, remaining)
                except InterruptedError:
                    continue
                if type(written) is not int or written <= 0:
                    _fail("launch snapshot write made no progress")
                remaining = remaining[written:]
        source_after = os.fstat(descriptor)
        if (
            _stat_frame(source_before) != _stat_frame(source_after)
            or copied_bytes != source_after.st_size
        ):
            _fail("launch source changed while its immutable snapshot was built")
        copied_digest = copied_hasher.hexdigest()
        if expected_sha256 is not None and copied_digest != expected_sha256:
            _fail("launch snapshot digest is not registered")

        os.fchmod(snapshot, 0o500 if executable else 0o400)
        added = fcntl.fcntl(
            snapshot,
            fcntl.F_ADD_SEALS,
            _LAUNCH_MEMFD_SEALS,
        )
        if type(added) is not int or added != 0:
            _fail("launch snapshot seals did not apply exactly")
        observed_seals = fcntl.fcntl(snapshot, fcntl.F_GET_SEALS)
        if observed_seals != _LAUNCH_MEMFD_SEALS:
            _fail("launch snapshot does not have the exact immutable seals")

        os.lseek(snapshot, 0, os.SEEK_SET)
        sealed_hasher = hashlib.sha256()
        sealed_bytes = 0
        while True:
            try:
                chunk = os.read(snapshot, 1 << 20)
            except InterruptedError:
                continue
            if not chunk:
                break
            sealed_bytes += len(chunk)
            sealed_hasher.update(chunk)
        if sealed_bytes != copied_bytes or sealed_hasher.hexdigest() != copied_digest:
            _fail("sealed launch snapshot differs from its validated source")
        os.lseek(snapshot, 0, os.SEEK_SET)
        return snapshot
    except BaseException:
        if snapshot >= 0:
            with suppress(OSError):
                os.close(snapshot)
        raise


def _open_validated_launch_snapshot(
    path: str,
    *,
    executable: bool,
    expected_sha256: str | None = None,
) -> int:
    source = _open_validated_regular_path(
        path,
        executable=executable,
        expected_sha256=expected_sha256,
    )
    try:
        return _sealed_launch_snapshot(
            source,
            executable=executable,
            expected_sha256=expected_sha256,
        )
    finally:
        os.close(source)


def _validate_python_executable_chain() -> None:
    for path, expected_target in _PYTHON_LINK_CHAIN:
        try:
            value = os.lstat(path)
            target = os.readlink(path)
        except OSError as error:
            raise Experiment002SupervisorError(
                "frozen Python executable link chain is inaccessible"
            ) from error
        if not stat.S_ISLNK(value.st_mode) or target != expected_target:
            _fail("frozen Python executable link chain differs")
    if os.path.realpath(_PYTHON_EXECUTABLE) != _PYTHON_RESOLVED_EXECUTABLE:
        _fail("frozen Python executable resolves to an unexpected path")
    _validate_regular_path(
        _PYTHON_RESOLVED_EXECUTABLE,
        executable=True,
        expected_sha256=_PYTHON_RESOLVED_SHA256,
    )


def _validate_launch_surface(plan: _LaunchPlan) -> None:
    _require_plan(plan)
    try:
        cwd = os.getcwd()
    except OSError as error:
        raise Experiment002SupervisorError("cannot inspect the parent cwd") from error
    if cwd != plan.repository_root or os.path.realpath(cwd) != plan.repository_root:
        _fail("parent cwd is not the registered repository root")
    try:
        temporary = os.lstat(plan.temporary_root)
    except OSError as error:
        raise Experiment002SupervisorError(
            "registered temporary root is inaccessible"
        ) from error
    if stat.S_ISLNK(temporary.st_mode) or not stat.S_ISDIR(temporary.st_mode):
        _fail("registered temporary root is not a nonsymlink directory")
    if os.path.realpath(plan.temporary_root) != plan.temporary_root:
        _fail("registered temporary root contains symlink ambiguity")
    if plan.taskset_executable == _TASKSET_EXECUTABLE:
        _validate_regular_path(
            plan.taskset_executable,
            executable=True,
            expected_sha256=_TASKSET_SHA256,
        )
    else:
        _validate_regular_path(plan.taskset_executable, executable=True)
    if plan.python_executable == _PYTHON_EXECUTABLE:
        _validate_python_executable_chain()
    else:
        _validate_regular_path(plan.python_executable, executable=True)
    _validate_regular_path(
        plan.runner_path,
        executable=False,
        expected_sha256=(_RUNNER_SHA256 if plan.runner_path == _RUNNER_PATH else None),
    )


def _pin_launch_files(plan: _LaunchPlan) -> tuple[int, int, int]:
    descriptors: list[int] = []
    try:
        taskset_fd = _open_validated_launch_snapshot(
            plan.taskset_executable,
            executable=True,
            expected_sha256=(
                _TASKSET_SHA256
                if plan.taskset_executable == _TASKSET_EXECUTABLE
                else None
            ),
        )
        descriptors.append(taskset_fd)
        if plan.python_executable == _PYTHON_EXECUTABLE:
            _validate_python_executable_chain()
            python_path = _PYTHON_RESOLVED_EXECUTABLE
            python_digest: str | None = _PYTHON_RESOLVED_SHA256
        else:
            python_path = plan.python_executable
            python_digest = None
        python_fd = _open_validated_launch_snapshot(
            python_path,
            executable=True,
            expected_sha256=python_digest,
        )
        descriptors.append(python_fd)
        runner_fd = _open_validated_launch_snapshot(
            plan.runner_path,
            executable=False,
            expected_sha256=(
                _RUNNER_SHA256 if plan.runner_path == _RUNNER_PATH else None
            ),
        )
        descriptors.append(runner_fd)
        return taskset_fd, python_fd, runner_fd
    except BaseException:
        for descriptor in reversed(descriptors):
            with suppress(OSError):
                os.close(descriptor)
        raise


def _parent_and_leaf(path: str) -> tuple[str, str]:
    parent, leaf = os.path.split(path)
    if not parent or not leaf or leaf in {".", ".."} or "/" in leaf or "\0" in leaf:
        _fail("output root must have one valid leaf")
    return parent, leaf


def _open_directory(path: str) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise Experiment002SupervisorError(
            f"cannot securely open directory: {path}"
        ) from error
    try:
        value = os.fstat(descriptor)
        if not stat.S_ISDIR(value.st_mode):
            _fail(f"opened path is not a directory: {path}")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _absolute_components(path: str) -> tuple[str, ...]:
    if (
        type(path) is not str
        or not os.path.isabs(path)
        or os.path.normpath(path) != path
        or "\0" in path
    ):
        _fail("pinned directory path is not canonical and absolute")
    components = tuple(part for part in path.split("/") if part)
    if any(part in {".", ".."} for part in components):
        _fail("pinned directory path contains an invalid component")
    return components


def _pin_directory_chain(path: str) -> _PinnedParent:
    components = _absolute_components(path)
    descriptors: list[int] = []
    identities: list[tuple[int, int, int]] = []
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        current = os.open("/", flags)
        descriptors.append(current)
        root_value = os.fstat(current)
        if not stat.S_ISDIR(root_value.st_mode):
            _fail("filesystem root is not a directory")
        identities.append(_identity_frame(root_value))
        for component in components:
            before = os.stat(component, dir_fd=current, follow_symlinks=False)
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                _fail("pinned directory chain contains a nondirectory component")
            child = os.open(component, flags, dir_fd=current)
            descriptors.append(child)
            opened = os.fstat(child)
            named_after = os.stat(
                component,
                dir_fd=current,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISDIR(opened.st_mode)
                or _identity_frame(before) != _identity_frame(opened)
                or _identity_frame(opened) != _identity_frame(named_after)
            ):
                _fail("pinned directory component changed while opening")
            identities.append(_identity_frame(opened))
            current = child
        return _PinnedParent(
            path=path,
            descriptors=tuple(descriptors),
            identities=tuple(identities),
            components=components,
        )
    except OSError as error:
        for descriptor in reversed(descriptors):
            with suppress(OSError):
                os.close(descriptor)
        raise Experiment002SupervisorError(
            f"cannot securely pin directory chain: {path}"
        ) from error
    except BaseException:
        for descriptor in reversed(descriptors):
            with suppress(OSError):
                os.close(descriptor)
        raise


def _verify_pinned_parent(
    parent: _PinnedParent,
    *,
    edge_count: int | None = None,
) -> None:
    if (
        type(parent) is not _PinnedParent
        or not parent.descriptors
        or len(parent.descriptors) != len(parent.identities)
        or len(parent.descriptors) != len(parent.components) + 1
        or "/" + "/".join(parent.components) != parent.path
        or any(
            type(descriptor) is not int or descriptor < 0
            for descriptor in parent.descriptors
        )
        or any(
            type(identity) is not tuple
            or len(identity) != 3
            or any(type(value) is not int or value < 0 for value in identity)
            for identity in parent.identities
        )
    ):
        _fail("pinned parent chain metadata is invalid")
    if edge_count is None:
        edge_count = len(parent.components)
    if type(edge_count) is not int or not 0 <= edge_count <= len(parent.components):
        _fail("pinned parent verification bound is invalid")
    try:
        for descriptor, identity in zip(
            parent.descriptors,
            parent.identities,
            strict=True,
        ):
            opened = os.fstat(descriptor)
            if not stat.S_ISDIR(opened.st_mode) or _identity_frame(opened) != identity:
                _fail("pinned parent descriptor identity changed")
        for index in range(edge_count):
            named = os.stat(
                parent.components[index],
                dir_fd=parent.descriptors[index],
                follow_symlinks=False,
            )
            if _identity_frame(named) != parent.identities[index + 1]:
                _fail("pinned parent name identity changed")
    except OSError as error:
        raise Experiment002SupervisorError(
            "pinned parent chain cannot be verified"
        ) from error


def _pin_existing_root(path: str) -> _PinnedRoot:
    parent, leaf = _parent_and_leaf(path)
    pinned_parent = _pin_directory_chain(parent)
    descriptor = -1
    transferred = False
    try:
        try:
            before = os.stat(
                leaf,
                dir_fd=pinned_parent.descriptor,
                follow_symlinks=False,
            )
        except OSError as error:
            raise Experiment002SupervisorError(
                f"output root is inaccessible: {path}"
            ) from error
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
            _fail(f"output root is not a nonsymlink directory: {path}")
        descriptor = os.open(
            leaf,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=pinned_parent.descriptor,
        )
        opened = os.fstat(descriptor)
        named_after = os.stat(
            leaf,
            dir_fd=pinned_parent.descriptor,
            follow_symlinks=False,
        )
        identity = _identity_frame(opened)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or identity != _identity_frame(before)
            or identity != _identity_frame(named_after)
            or stat.S_IMODE(opened.st_mode) != 0o700
        ):
            _fail(f"output root identity or mode is invalid: {path}")
        pinned = _PinnedRoot(path, pinned_parent, descriptor, identity)
        descriptor = -1
        transferred = True
        return pinned
    except OSError as error:
        raise Experiment002SupervisorError(
            f"cannot securely pin output root: {path}"
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if not transferred:
            pinned_parent.close()


def _create_and_pin_fresh_root(path: str) -> _PinnedRoot:
    parent, leaf = _parent_and_leaf(path)
    pinned_parent = _pin_directory_chain(parent)
    descriptor = -1
    created = False
    transferred = False
    try:
        try:
            os.mkdir(leaf, 0o700, dir_fd=pinned_parent.descriptor)
            created = True
        except FileExistsError as error:
            raise Experiment002SupervisorError(
                f"fresh output root already exists: {path}"
            ) from error
        descriptor = os.open(
            leaf,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=pinned_parent.descriptor,
        )
        os.fchmod(descriptor, 0o700)
        opened = os.fstat(descriptor)
        named = os.stat(
            leaf,
            dir_fd=pinned_parent.descriptor,
            follow_symlinks=False,
        )
        identity = _identity_frame(opened)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or identity != _identity_frame(named)
            or stat.S_IMODE(opened.st_mode) != 0o700
            or os.listdir(descriptor)
        ):
            _fail(f"fresh output root is not an empty mode-0700 directory: {path}")
        pinned = _PinnedRoot(path, pinned_parent, descriptor, identity)
        descriptor = -1
        transferred = True
        return pinned
    except BaseException as primary:
        cleanup_failure: BaseException | None = None
        if descriptor >= 0:
            closing = descriptor
            descriptor = -1
            try:
                os.close(closing)
            except BaseException as error:
                cleanup_failure = error
        if created:
            try:
                _remove_named_tree(pinned_parent.descriptor, leaf)
            except BaseException as error:
                if cleanup_failure is None:
                    cleanup_failure = error
        if cleanup_failure is not None:
            raise Experiment002SupervisorError(
                "fresh output root creation failed with cleanup errors"
            ) from primary
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if not transferred:
            pinned_parent.close()


def _create_fresh_directory(path: str) -> None:
    pinned = _create_and_pin_fresh_root(path)
    pinned.close()


def _require_output_root_absent(path: str) -> None:
    parent, leaf = _parent_and_leaf(path)
    pinned_parent = _pin_directory_chain(parent)
    try:
        try:
            os.stat(
                leaf,
                dir_fd=pinned_parent.descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return
        except OSError as error:
            raise Experiment002SupervisorError(
                f"fresh output root cannot be inspected: {path}"
            ) from error
        _fail(f"fresh output root already exists: {path}")
    finally:
        pinned_parent.close()


def _prepare_fresh_output_roots(roots: tuple[str, str]) -> None:
    if type(roots) is not tuple or len(roots) != 2 or roots[0] == roots[1]:
        _fail("output roots are not one exact scratch/staging pair")
    created: list[str] = []
    try:
        for root in roots:
            _create_fresh_directory(root)
            created.append(root)
    except BaseException:
        for root in reversed(created):
            with suppress(BaseException):
                _remove_directory_tree(root)
        raise


def _prepare_experiment_staging(plan: _LaunchPlan = _REGISTERED_PLAN) -> None:
    """Create the one experiment-lifetime staging root before the first child."""

    _require_plan(plan)
    _create_fresh_directory(plan.staging_root)


def _require_existing_output_root(path: str) -> None:
    try:
        pinned = _pin_existing_root(path)
    except Experiment002SupervisorError as error:
        raise Experiment002SupervisorError(
            "experiment staging root is absent or invalid"
        ) from error
    pinned.close()


def _scan_directory(
    descriptor: int,
    *,
    seen: set[tuple[int, int]],
    entries_seen: list[int],
    depth: int,
    maximum_bytes: int,
    total: int,
) -> int:
    if depth > _OUTPUT_DEPTH_MAXIMUM:
        _fail("output directory nesting exceeds its limit")
    try:
        names = os.listdir(descriptor)
    except OSError as error:
        raise Experiment002SupervisorError(
            "output directory cannot be enumerated"
        ) from error
    if any(
        type(name) is not str or not name or "/" in name or "\0" in name
        for name in names
    ):
        _fail("output directory contains an invalid name")
    try:
        ordered = sorted(names, key=lambda name: name.encode("utf-8"))
    except UnicodeEncodeError as error:
        raise Experiment002SupervisorError("output name is not valid UTF-8") from error
    for name in ordered:
        entries_seen[0] += 1
        if entries_seen[0] > _OUTPUT_ENTRY_COUNT_MAXIMUM:
            _fail("output tree contains too many entries")
        try:
            before = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except OSError as error:
            raise Experiment002SupervisorError(
                "output entry raced during inspection"
            ) from error
        if stat.S_ISLNK(before.st_mode):
            _fail("output tree contains a symlink")
        if stat.S_ISDIR(before.st_mode):
            child_fd = -1
            try:
                child_fd = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=descriptor,
                )
                opened = os.fstat(child_fd)
                if _identity_frame(before) != _identity_frame(opened):
                    _fail("output directory identity changed while opening")
                total = _scan_directory(
                    child_fd,
                    seen=seen,
                    entries_seen=entries_seen,
                    depth=depth + 1,
                    maximum_bytes=maximum_bytes,
                    total=total,
                )
                after = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if _identity_frame(opened) != _identity_frame(after):
                    _fail("output directory identity changed during traversal")
            except OSError as error:
                raise Experiment002SupervisorError(
                    "output directory cannot be traversed safely"
                ) from error
            finally:
                if child_fd >= 0:
                    os.close(child_fd)
            continue
        if not stat.S_ISREG(before.st_mode):
            _fail("output tree contains a nonregular leaf")
        leaf_fd = -1
        try:
            leaf_fd = os.open(
                name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
                dir_fd=descriptor,
            )
            opened = os.fstat(leaf_fd)
            after_open = os.fstat(leaf_fd)
            named_after = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if (
                not stat.S_ISREG(opened.st_mode)
                or _stat_frame(before) != _stat_frame(opened)
                or _stat_frame(opened) != _stat_frame(after_open)
                or _stat_frame(after_open) != _stat_frame(named_after)
            ):
                _fail("output regular file changed during inspection")
        except OSError as error:
            raise Experiment002SupervisorError(
                "output regular file cannot be inspected safely"
            ) from error
        finally:
            if leaf_fd >= 0:
                os.close(leaf_fd)
        identity = (opened.st_dev, opened.st_ino)
        if identity in seen:
            continue
        seen.add(identity)
        if opened.st_size < 0 or opened.st_blocks < 0:
            _fail("output regular file has invalid accounting metadata")
        contribution = max(opened.st_size, 512 * opened.st_blocks)
        total += contribution
        if total > maximum_bytes:
            _fail("output and scratch byte budget was exceeded")
    return total


def _require_pinned_roots(roots: tuple[_PinnedRoot, _PinnedRoot]) -> None:
    if (
        type(roots) is not tuple
        or len(roots) != 2
        or any(type(root) is not _PinnedRoot for root in roots)
        or roots[0].path == roots[1].path
        or roots[0].identity == roots[1].identity
    ):
        _fail("pinned roots are not one exact scratch/staging pair")
    for root in roots:
        if (
            type(root.path) is not str
            or not root.path
            or type(root.parent) is not _PinnedParent
            or type(root.descriptor) is not int
            or root.descriptor < 0
            or type(root.identity) is not tuple
            or len(root.identity) != 3
            or any(type(value) is not int or value < 0 for value in root.identity)
        ):
            _fail("pinned output root metadata is invalid")


def _account_pinned_roots(
    roots: tuple[_PinnedRoot, _PinnedRoot], maximum_bytes: int
) -> int:
    _require_exact_integer(maximum_bytes, "output limit", minimum=1)
    _require_pinned_roots(roots)
    seen: set[tuple[int, int]] = set()
    entries_seen = [0]
    total = 0
    for root in roots:
        _verify_pinned_parent(root.parent)
        try:
            opened = os.fstat(root.descriptor)
        except OSError as error:
            raise Experiment002SupervisorError(
                "pinned output root descriptor is inaccessible"
            ) from error
        if (
            not stat.S_ISDIR(opened.st_mode)
            or _identity_frame(opened) != root.identity
            or stat.S_IMODE(opened.st_mode) != 0o700
        ):
            _fail("pinned output root identity or mode is invalid")
        total = _scan_directory(
            root.descriptor,
            seen=seen,
            entries_seen=entries_seen,
            depth=0,
            maximum_bytes=maximum_bytes,
            total=total,
        )
        try:
            after = os.fstat(root.descriptor)
            _verify_pinned_parent(root.parent)
            _parent, leaf = _parent_and_leaf(root.path)
            named_after = os.stat(
                leaf,
                dir_fd=root.parent_fd,
                follow_symlinks=False,
            )
        except OSError as error:
            raise Experiment002SupervisorError(
                "output root name changed during traversal"
            ) from error
        if (
            _identity_frame(after) != root.identity
            or stat.S_IMODE(after.st_mode) != 0o700
            or _identity_frame(named_after) != root.identity
        ):
            _fail("output root identity changed during traversal")
    return total


def _account_output_roots(roots: tuple[str, str], maximum_bytes: int) -> int:
    _require_exact_integer(maximum_bytes, "output limit", minimum=1)
    if type(roots) is not tuple or len(roots) != 2 or roots[0] == roots[1]:
        _fail("output roots are not one exact scratch/staging pair")
    pinned: list[_PinnedRoot] = []
    try:
        for root_path in roots:
            pinned.append(_pin_existing_root(root_path))
        return _account_pinned_roots((pinned[0], pinned[1]), maximum_bytes)
    finally:
        for pinned_root in reversed(pinned):
            pinned_root.close()


def _list_directory_retry(descriptor: int) -> tuple[str, ...]:
    while True:
        try:
            names = os.listdir(descriptor)
            break
        except InterruptedError:
            continue
    if any(
        type(name) is not str or not name or "/" in name or "\0" in name
        for name in names
    ):
        _fail("cleanup directory contains an invalid name")
    return tuple(names)


def _cleanup_node_chain(node: _CleanupNode) -> list[_CleanupNode]:
    chain: list[_CleanupNode] = []
    cursor: _CleanupNode | None = node
    while cursor is not None:
        chain.append(cursor)
        cursor = cursor.parent
    chain.reverse()
    if not chain or chain[0].name is not None or chain[0].parent is not None:
        _fail("cleanup node chain lacks its root")
    return chain


def _open_cleanup_node(
    root_descriptor: int,
    node: _CleanupNode,
) -> tuple[int, bool]:
    chain = _cleanup_node_chain(node)
    current = root_descriptor
    owned = False
    try:
        root_value = os.fstat(current)
        if _identity_frame(root_value) != chain[0].identity:
            _fail("cleanup root descriptor identity changed")
        for component in chain[1:]:
            if component.name is None:
                _fail("cleanup path contains an unnamed component")
            while True:
                try:
                    child = os.open(
                        component.name,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                        dir_fd=current,
                    )
                    break
                except InterruptedError:
                    continue
            try:
                opened = os.fstat(child)
                if _identity_frame(opened) != component.identity:
                    _fail("cleanup path component identity changed")
            except BaseException:
                with suppress(OSError):
                    os.close(child)
                raise
            if owned:
                os.close(current)
            current = child
            owned = True
        return current, owned
    except BaseException:
        if owned:
            with suppress(OSError):
                os.close(current)
        raise


def _remove_directory_contents(descriptor: int) -> None:
    if type(descriptor) is not int or descriptor < 0:
        _fail("cleanup directory descriptor is invalid")
    root_value = os.fstat(descriptor)
    if not stat.S_ISDIR(root_value.st_mode):
        _fail("cleanup root descriptor is not a directory")
    root = _CleanupNode(
        name=None,
        parent=None,
        identity=_identity_frame(root_value),
    )
    pending = [root]
    while pending:
        node = pending.pop()
        if node.expanded:
            if node.parent is None:
                continue
            if node.name is None:
                _fail("cleanup directory node lacks its name")
            parent_fd, parent_owned = _open_cleanup_node(descriptor, node.parent)
            try:
                while True:
                    try:
                        named_after = os.stat(
                            node.name,
                            dir_fd=parent_fd,
                            follow_symlinks=False,
                        )
                        break
                    except InterruptedError:
                        continue
                if _identity_frame(named_after) != node.identity:
                    _fail("cleanup directory identity changed during traversal")
                while True:
                    try:
                        os.rmdir(node.name, dir_fd=parent_fd)
                        break
                    except InterruptedError:
                        continue
            finally:
                if parent_owned:
                    os.close(parent_fd)
            continue

        directory_fd, directory_owned = _open_cleanup_node(descriptor, node)
        children: list[_CleanupNode] = []
        try:
            for name in _list_directory_retry(directory_fd):
                while True:
                    try:
                        value = os.stat(
                            name,
                            dir_fd=directory_fd,
                            follow_symlinks=False,
                        )
                        break
                    except InterruptedError:
                        continue
                    except FileNotFoundError:
                        value = None
                        break
                if value is None:
                    continue
                if stat.S_ISDIR(value.st_mode) and not stat.S_ISLNK(value.st_mode):
                    children.append(
                        _CleanupNode(
                            name=name,
                            parent=node,
                            identity=_identity_frame(value),
                        )
                    )
                    continue
                while True:
                    try:
                        os.unlink(name, dir_fd=directory_fd)
                        break
                    except InterruptedError:
                        continue
        finally:
            if directory_owned:
                os.close(directory_fd)
        node.expanded = True
        pending.append(node)
        pending.extend(children)


def _remove_named_tree(parent_fd: int, leaf: str) -> None:
    descriptor = -1
    try:
        try:
            value = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
            while True:
                try:
                    os.unlink(leaf, dir_fd=parent_fd)
                    break
                except InterruptedError:
                    continue
            return
        while True:
            try:
                descriptor = os.open(
                    leaf,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=parent_fd,
                )
                break
            except InterruptedError:
                continue
        opened = os.fstat(descriptor)
        if _identity_frame(value) != _identity_frame(opened):
            _fail("cleanup root identity changed")
        _remove_directory_contents(descriptor)
        os.close(descriptor)
        descriptor = -1
        named_after = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        if _identity_frame(opened) != _identity_frame(named_after):
            _fail("cleanup root was replaced")
        while True:
            try:
                os.rmdir(leaf, dir_fd=parent_fd)
                break
            except InterruptedError:
                continue
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _remove_directory_tree(path: str) -> None:
    parent, leaf = _parent_and_leaf(path)
    parent_fd = _open_directory(parent)
    try:
        _remove_named_tree(parent_fd, leaf)
    finally:
        os.close(parent_fd)


def _names_for_identity(
    parent_fd: int,
    identity: tuple[int, int, int],
) -> list[str]:
    matches: list[str] = []
    for name in _list_directory_retry(parent_fd):
        try:
            value = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        except OSError as error:
            raise Experiment002SupervisorError(
                "pinned parent entry cannot be inspected"
            ) from error
        if _identity_frame(value) == identity:
            matches.append(name)
    return matches


def _first_parent_name_mismatch(parent: _PinnedParent) -> int | None:
    _verify_pinned_parent(parent, edge_count=0)
    for index, component in enumerate(parent.components):
        try:
            named = os.stat(
                component,
                dir_fd=parent.descriptors[index],
                follow_symlinks=False,
            )
        except OSError:
            return index
        if _identity_frame(named) != parent.identities[index + 1]:
            return index
    return None


def _clean_registered_replacement_chain(
    parent: _PinnedParent,
    mismatch_index: int,
    managed_leaves: tuple[str, str],
) -> bool:
    if not 0 <= mismatch_index < len(parent.components):
        _fail("replacement chain mismatch index is invalid")
    current = parent.descriptors[mismatch_index]
    opened: list[tuple[int, int, str, tuple[int, int, int]]] = []
    registered_path_terminated = False
    try:
        for component in parent.components[mismatch_index:]:
            try:
                value = os.stat(
                    component,
                    dir_fd=current,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                registered_path_terminated = True
                break
            if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
                while True:
                    try:
                        os.unlink(component, dir_fd=current)
                        break
                    except InterruptedError:
                        continue
                registered_path_terminated = True
                break
            child = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=current,
            )
            child_value = os.fstat(child)
            if _identity_frame(child_value) != _identity_frame(value):
                os.close(child)
                _fail("replacement ancestor changed while opening")
            opened.append((child, current, component, _identity_frame(child_value)))
            current = child
        else:
            for leaf in managed_leaves:
                _remove_named_tree(current, leaf)

        removed_top = False
        for child, owner, component, identity in reversed(opened):
            if _list_directory_retry(child):
                break
            os.close(child)
            opened.remove((child, owner, component, identity))
            named_after = os.stat(
                component,
                dir_fd=owner,
                follow_symlinks=False,
            )
            if _identity_frame(named_after) != identity:
                _fail("replacement ancestor changed during cleanup")
            while True:
                try:
                    os.rmdir(component, dir_fd=owner)
                    break
                except InterruptedError:
                    continue
            if owner == parent.descriptors[mismatch_index]:
                removed_top = True
        if not opened and registered_path_terminated:
            removed_top = True
        return removed_top
    finally:
        for child, _owner, _component, _identity in reversed(opened):
            with suppress(OSError):
                os.close(child)


def _recover_pinned_parent(roots: tuple[_PinnedRoot, _PinnedRoot]) -> None:
    _require_pinned_roots(roots)
    left, right = roots
    if (
        left.parent.path != right.parent.path
        or left.parent.components != right.parent.components
        or left.parent.identities != right.parent.identities
    ):
        _fail("scratch and staging do not share one pinned parent identity")
    parent = left.parent
    if not parent.components:
        _verify_pinned_parent(parent)
        return
    managed = tuple(_parent_and_leaf(root.path)[1] for root in roots)
    managed_leaves = (managed[0], managed[1])
    for _attempt in range(len(parent.components)):
        mismatch_index = _first_parent_name_mismatch(parent)
        if mismatch_index is None:
            _verify_pinned_parent(left.parent)
            _verify_pinned_parent(right.parent)
            return
        owner = parent.descriptors[mismatch_index]
        registered_component = parent.components[mismatch_index]
        expected_identity = parent.identities[mismatch_index + 1]
        matches = _names_for_identity(owner, expected_identity)
        removed = _clean_registered_replacement_chain(
            parent,
            mismatch_index,
            managed_leaves,
        )
        if not removed:
            _fail("replacement ancestor contains unmanaged entries")
        if len(matches) != 1:
            _fail("pinned ancestor cannot be located uniquely")
        while True:
            try:
                os.rename(
                    matches[0],
                    registered_component,
                    src_dir_fd=owner,
                    dst_dir_fd=owner,
                )
                break
            except InterruptedError:
                continue
        restored = os.stat(
            registered_component,
            dir_fd=owner,
            follow_symlinks=False,
        )
        if _identity_frame(restored) != expected_identity:
            _fail("pinned ancestor identity changed while restoring its name")
    _fail("pinned parent chain could not be recovered")


def _restore_pinned_root_name(root: _PinnedRoot) -> None:
    if type(root) is not _PinnedRoot:
        raise TypeError("output root must be an exact _PinnedRoot")
    _verify_pinned_parent(root.parent)
    matches = _names_for_identity(root.parent_fd, root.identity)
    _parent, registered_leaf = _parent_and_leaf(root.path)
    if matches != [registered_leaf]:
        _remove_named_tree(root.parent_fd, registered_leaf)
        if len(matches) != 1:
            _fail("pinned output root cannot be located uniquely")
        while True:
            try:
                os.rename(
                    matches[0],
                    registered_leaf,
                    src_dir_fd=root.parent_fd,
                    dst_dir_fd=root.parent_fd,
                )
                break
            except InterruptedError:
                continue
    named = os.stat(
        registered_leaf,
        dir_fd=root.parent_fd,
        follow_symlinks=False,
    )
    if _identity_frame(named) != root.identity:
        _fail("pinned output root could not be restored")


def _cleanup_pinned_staging(root: _PinnedRoot) -> None:
    if type(root) is not _PinnedRoot:
        raise TypeError("staging root must be an exact _PinnedRoot")
    try:
        opened = os.fstat(root.descriptor)
    except OSError as error:
        raise Experiment002SupervisorError(
            "pinned staging descriptor is inaccessible during cleanup"
        ) from error
    if _identity_frame(opened) != root.identity or not stat.S_ISDIR(opened.st_mode):
        _fail("pinned staging identity changed before cleanup")
    _remove_directory_contents(root.descriptor)
    if _identity_frame(os.fstat(root.descriptor)) != root.identity:
        _fail("pinned staging identity changed during cleanup")
    _restore_pinned_root_name(root)


def _cleanup_pinned_scratch(root: _PinnedRoot) -> None:
    if type(root) is not _PinnedRoot:
        raise TypeError("scratch root must be an exact _PinnedRoot")
    if root.parent_fd < 0 or root.descriptor < 0:
        _fail("scratch root was not retained for cleanup")
    try:
        opened = os.fstat(root.descriptor)
    except OSError as error:
        raise Experiment002SupervisorError(
            "pinned scratch descriptor is inaccessible during cleanup"
        ) from error
    if _identity_frame(opened) != root.identity or not stat.S_ISDIR(opened.st_mode):
        _fail("pinned scratch identity changed before cleanup")
    _remove_directory_contents(root.descriptor)
    if _identity_frame(os.fstat(root.descriptor)) != root.identity:
        _fail("pinned scratch identity changed during cleanup")

    matches = _names_for_identity(root.parent_fd, root.identity)
    direct_identity_missing = len(matches) != 1
    if len(matches) > 1:
        _fail("pinned scratch identity appears under multiple parent names")
    if matches:
        while True:
            try:
                os.rmdir(matches[0], dir_fd=root.parent_fd)
                break
            except InterruptedError:
                continue

    _parent, registered_leaf = _parent_and_leaf(root.path)
    if not matches or matches[0] != registered_leaf:
        _remove_named_tree(root.parent_fd, registered_leaf)
    if direct_identity_missing:
        _fail("pinned scratch was moved outside its registered parent")


def _cleanup_output_roots(roots: tuple[str, str]) -> None:
    failures: list[BaseException] = []
    for root in reversed(roots):
        try:
            _remove_directory_tree(root)
        except BaseException as error:
            failures.append(error)
    if failures:
        raise Experiment002SupervisorError(
            "failed to clean supervised output roots"
        ) from failures[0]


def _parse_vmrss_bytes(contents: bytes) -> int:
    if (
        type(contents) is not bytes
        or not contents
        or len(contents) > _PROC_FILE_BYTES_MAXIMUM
    ):
        _fail("/proc status payload has an invalid size or type")
    if b"\0" in contents or not contents.endswith(b"\n"):
        _fail("/proc status payload is not a complete text record")
    matches = [line for line in contents.splitlines() if line.startswith(b"VmRSS:")]
    if len(matches) != 1:
        _fail("/proc status does not contain exactly one VmRSS record")
    fields = matches[0].split()
    if len(fields) != 3 or fields[0] != b"VmRSS:" or fields[2] != b"kB":
        _fail("/proc VmRSS record has an invalid format")
    digits = fields[1]
    if (
        not digits
        or not digits.isdigit()
        or (len(digits) > 1 and digits.startswith(b"0"))
    ):
        _fail("/proc VmRSS value is not a canonical nonnegative integer")
    kib = int(digits)
    if kib > (1 << 63) // _KIB:
        _fail("/proc VmRSS value is out of range")
    return kib * _KIB


def _require_userspace_task_status(contents: bytes) -> None:
    if (
        type(contents) is not bytes
        or not contents
        or len(contents) > _PROC_FILE_BYTES_MAXIMUM
        or b"\0" in contents
        or not contents.endswith(b"\n")
    ):
        _fail("/proc status payload is not a complete userspace task record")
    matches = [
        line
        for line in contents.splitlines()
        if line.lstrip().lower().startswith(b"kthread")
    ]
    if matches != [b"Kthread:\t0"]:
        _fail("/proc status does not identify exactly one userspace task")


def _parse_task_vmrss_bytes(contents: bytes) -> int | None:
    _require_userspace_task_status(contents)
    try:
        return _parse_vmrss_bytes(contents)
    except Experiment002SupervisorError:
        if (
            type(contents) is not bytes
            or not contents
            or len(contents) > _PROC_FILE_BYTES_MAXIMUM
            or b"\0" in contents
            or not contents.endswith(b"\n")
        ):
            raise
        vmrss_like = any(
            line.lstrip().lower().startswith(b"vmrss") for line in contents.splitlines()
        )
        if vmrss_like:
            raise
        return None


def _parse_children_bytes(contents: bytes) -> tuple[int, ...]:
    if (
        type(contents) is not bytes
        or len(contents) > _PROC_FILE_BYTES_MAXIMUM
        or b"\0" in contents
    ):
        _fail("/proc children payload has an invalid size or type")
    stripped = contents.strip()
    if not stripped:
        return ()
    fields = stripped.split()
    children: list[int] = []
    for field in fields:
        if not field.isdigit() or field.startswith(b"0"):
            _fail("/proc children payload contains an invalid PID")
        child = int(field)
        if child < 1 or child in children:
            _fail("/proc children payload contains a duplicate or invalid PID")
        children.append(child)
    return tuple(children)


def _read_bounded_proc_file(path: str) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError as error:
        raise ProcessLookupError(path) from error
    except OSError as error:
        raise Experiment002SupervisorError(
            f"cannot open process status file: {path}"
        ) from error
    try:
        value = os.fstat(descriptor)
        if not stat.S_ISREG(value.st_mode):
            _fail("process status path is not a regular procfs file")
        chunks: list[bytes] = []
        count = 0
        while True:
            try:
                chunk = os.read(
                    descriptor, min(65_536, _PROC_FILE_BYTES_MAXIMUM + 1 - count)
                )
            except InterruptedError:
                continue
            if not chunk:
                break
            chunks.append(chunk)
            count += len(chunk)
            if count > _PROC_FILE_BYTES_MAXIMUM:
                _fail("process status file exceeds its byte limit")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _read_process_sample(pid: int) -> _ProcessSample:
    _require_exact_integer(pid, "child PID", minimum=1)
    status_contents = _read_bounded_proc_file(f"/proc/{pid}/status")
    leader_rss = _parse_task_vmrss_bytes(status_contents)
    task_path = f"/proc/{pid}/task"
    try:
        names = os.listdir(task_path)
    except FileNotFoundError as error:
        raise ProcessLookupError(task_path) from error
    except OSError as error:
        raise Experiment002SupervisorError("cannot enumerate child tasks") from error
    tids: list[int] = []
    for name in names:
        if (
            type(name) is not str
            or not name.isascii()
            or not name.isdigit()
            or name.startswith("0")
        ):
            _fail("child task directory contains an invalid TID")
        tids.append(int(name))
    if pid not in tids or len(tids) != len(set(tids)):
        _fail("child task directory lacks its leader or repeats a TID")
    children: list[int] = []
    rss_values = [] if leader_rss is None else [leader_rss]
    for tid in sorted(tids):
        if leader_rss is None:
            task_status: bytes | None
            if tid == pid:
                task_status = status_contents
            else:
                try:
                    task_status = _read_bounded_proc_file(f"{task_path}/{tid}/status")
                except ProcessLookupError:
                    task_status = None
            if task_status is not None:
                task_rss = _parse_task_vmrss_bytes(task_status)
                if task_rss is not None:
                    rss_values.append(task_rss)
        try:
            payload = _read_bounded_proc_file(f"{task_path}/{tid}/children")
        except ProcessLookupError:
            if tid == pid:
                raise
            continue
        for child in _parse_children_bytes(payload):
            if child not in children:
                children.append(child)
    rss_bytes = max(rss_values) if rss_values else None
    return _ProcessSample(rss_bytes=rss_bytes, child_pids=tuple(children))


def _read_parent_child_pids() -> tuple[int, ...]:
    payload = _read_bounded_proc_file(f"/proc/self/task/{os.getpid()}/children")
    return _parse_children_bytes(payload)


def _require_libc_success(result: object, operation: str) -> None:
    if type(result) is not int:
        _fail(f"{operation} returned a noninteger result")
    if result != 0:
        raise Experiment002SupervisorError(f"{operation} failed") from OSError(
            result,
            os.strerror(result),
        )


def _require_source_bundle_descriptor(
    descriptor: int,
    *,
    safe_source: bool = False,
) -> _SourceBundleFrame:
    """Admit only one immutable anonymous read-only Linux memfd."""

    if (
        type(descriptor) is not int
        or descriptor < 0
        or type(safe_source) is not bool
        or (safe_source and descriptor < _SAFE_SOURCE_FD_MINIMUM)
    ):
        _fail("source bundle descriptor is invalid")
    try:
        before = os.fstat(descriptor)
        status_flags = fcntl.fcntl(descriptor, fcntl.F_GETFL)
        descriptor_flags = fcntl.fcntl(descriptor, fcntl.F_GETFD)
        seals = fcntl.fcntl(descriptor, fcntl.F_GET_SEALS)
        proc_target = os.readlink(f"/proc/self/fd/{descriptor}")
        after = os.fstat(descriptor)
    except (OSError, ValueError) as error:
        raise Experiment002SupervisorError(
            "source bundle descriptor cannot be inspected"
        ) from error
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 0
        or before.st_uid != os.geteuid()
        or before.st_gid != os.getegid()
        or stat.S_IMODE(before.st_mode) != 0o400
        or not 0 < before.st_size <= _SOURCE_BUNDLE_BYTES_MAXIMUM
        or _stat_frame(before) != _stat_frame(after)
    ):
        _fail("source bundle is not an exact bounded anonymous regular file")
    if type(status_flags) is not int or status_flags & os.O_ACCMODE != os.O_RDONLY:
        _fail("source bundle descriptor is not read-only")
    if (
        type(descriptor_flags) is not int
        or descriptor_flags & fcntl.FD_CLOEXEC != fcntl.FD_CLOEXEC
    ):
        _fail("source bundle descriptor is not close-on-exec")
    if type(seals) is not int or seals != _SOURCE_BUNDLE_MEMFD_SEALS:
        _fail("source bundle memfd does not have the exact immutable seals")
    if (
        type(proc_target) is not str
        or not proc_target.startswith("/memfd:")
        or not proc_target.endswith(" (deleted)")
    ):
        _fail("source bundle descriptor is not an anonymous Linux memfd")
    return _SourceBundleFrame(metadata=_stat_frame(before), seals=seals)


def _require_matching_source_bundle_descriptors(
    caller_descriptor: int,
    independent_descriptor: int,
) -> None:
    caller = _require_source_bundle_descriptor(caller_descriptor)
    independent = _require_source_bundle_descriptor(
        independent_descriptor,
        safe_source=True,
    )
    if caller != independent:
        _fail("independent source bundle descriptor does not exactly match caller")


def _require_raw_file_actions(
    file_actions: tuple[_FileAction, ...],
    source_bundle_fd: int,
) -> None:
    dup2_actions = cast(tuple[_Dup2Action, ...], file_actions)
    expected_targets = (
        0,
        1,
        2,
        _CHILD_CONTROL_FD,
        _CHILD_RUNNER_FD,
        _CHILD_PYTHON_FD,
        _CHILD_TASKSET_FD,
        _CHILD_SOURCE_BUNDLE_FD,
    )
    if (
        type(file_actions) is not tuple
        or len(file_actions) != len(expected_targets)
        or any(
            type(action) is not tuple
            or len(action) != 3
            or action[0] != os.POSIX_SPAWN_DUP2
            or type(action[1]) is not int
            or action[1] < _SAFE_SOURCE_FD_MINIMUM
            or type(action[2]) is not int
            for action in file_actions
        )
        or tuple(action[2] for action in dup2_actions) != expected_targets
        or not (dup2_actions[0][1] == dup2_actions[1][1] == dup2_actions[2][1])
        or len(
            {
                dup2_actions[0][1],
                *(action[1] for action in dup2_actions[3:]),
            }
        )
        != 6
        or dup2_actions[-1]
        != (
            os.POSIX_SPAWN_DUP2,
            source_bundle_fd,
            _CHILD_SOURCE_BUNDLE_FD,
        )
    ):
        _fail("raw posix_spawn file actions are not the exact fixed FD layout")


def _raw_posix_spawn(
    executable_path: str,
    argv: tuple[str, ...],
    environment: Mapping[str, str],
    file_actions: tuple[_FileAction, ...],
    source_bundle_fd: int,
) -> int:
    if (
        ctypes.sizeof(_SigSet) != 128
        or ctypes.sizeof(_SpawnAttributes) != 336
        or ctypes.sizeof(_SpawnFileActions) != 80
    ):
        _fail("host glibc spawn ABI layout is not registered")
    if (
        type(executable_path) is not str
        or not executable_path.startswith("/proc/self/fd/")
        or "\0" in executable_path
        or type(argv) is not tuple
        or not argv
        or any(type(value) is not str or "\0" in value for value in argv)
    ):
        _fail("raw posix_spawn arguments are invalid")
    if type(environment) is not dict or any(
        type(key) is not str
        or not key
        or "=" in key
        or "\0" in key
        or type(value) is not str
        or "\0" in value
        for key, value in environment.items()
    ):
        _fail("raw posix_spawn environment is invalid")
    _require_source_bundle_descriptor(source_bundle_fd, safe_source=True)
    _require_raw_file_actions(file_actions, source_bundle_fd)

    argv_bytes = tuple(os.fsencode(value) for value in argv)
    environment_bytes = tuple(
        os.fsencode(f"{key}={value}") for key, value in environment.items()
    )
    argv_array = (ctypes.c_char_p * (len(argv_bytes) + 1))(
        *argv_bytes,
        None,
    )
    environment_array = (ctypes.c_char_p * (len(environment_bytes) + 1))(
        *environment_bytes,
        None,
    )
    actions = _SpawnFileActions()
    attributes = _SpawnAttributes()
    empty_mask = _SigSet()
    actions_initialized = False
    attributes_initialized = False
    primary: BaseException | None = None
    spawned_pid = -1
    try:
        _require_libc_success(
            _LIBC.posix_spawn_file_actions_init(ctypes.byref(actions)),
            "posix_spawn file-actions initialization",
        )
        actions_initialized = True
        for action in file_actions:
            if len(action) == 3 and action[0] == os.POSIX_SPAWN_DUP2:
                _require_libc_success(
                    _LIBC.posix_spawn_file_actions_adddup2(
                        ctypes.byref(actions),
                        action[1],
                        action[2],
                    ),
                    "posix_spawn dup2 action",
                )
            elif len(action) == 2 and action[0] == os.POSIX_SPAWN_CLOSE:
                _require_libc_success(
                    _LIBC.posix_spawn_file_actions_addclose(
                        ctypes.byref(actions),
                        action[1],
                    ),
                    "posix_spawn close action",
                )
            else:
                _fail("raw posix_spawn received an unknown file action")
        _require_libc_success(
            _LIBC.posix_spawn_file_actions_addclosefrom_np(
                ctypes.byref(actions),
                _SAFE_SOURCE_FD_MINIMUM,
            ),
            "posix_spawn closefrom action",
        )
        _require_libc_success(
            _LIBC.posix_spawnattr_init(ctypes.byref(attributes)),
            "posix_spawn attribute initialization",
        )
        attributes_initialized = True
        _require_libc_success(
            _LIBC.sigemptyset(ctypes.byref(empty_mask)),
            "empty child signal-mask initialization",
        )
        _require_libc_success(
            _LIBC.posix_spawnattr_setpgroup(ctypes.byref(attributes), 0),
            "posix_spawn process-group attribute",
        )
        _require_libc_success(
            _LIBC.posix_spawnattr_setsigmask(
                ctypes.byref(attributes),
                ctypes.byref(empty_mask),
            ),
            "posix_spawn signal-mask attribute",
        )
        _require_libc_success(
            _LIBC.posix_spawnattr_setflags(
                ctypes.byref(attributes),
                _POSIX_SPAWN_SETPGROUP | _POSIX_SPAWN_SETSIGMASK,
            ),
            "posix_spawn flags attribute",
        )
        pid_value = ctypes.c_int()
        result = _LIBC.posix_spawn(
            ctypes.byref(pid_value),
            os.fsencode(executable_path),
            ctypes.byref(actions),
            ctypes.byref(attributes),
            argv_array,
            environment_array,
        )
        _require_libc_success(result, "raw glibc posix_spawn")
        spawned_pid = pid_value.value
    except BaseException as error:
        primary = error

    cleanup_failures: list[BaseException] = []
    if attributes_initialized:
        try:
            _require_libc_success(
                _LIBC.posix_spawnattr_destroy(ctypes.byref(attributes)),
                "posix_spawn attribute destruction",
            )
        except BaseException as error:
            cleanup_failures.append(error)
    if actions_initialized:
        try:
            _require_libc_success(
                _LIBC.posix_spawn_file_actions_destroy(ctypes.byref(actions)),
                "posix_spawn file-actions destruction",
            )
        except BaseException as error:
            cleanup_failures.append(error)
    if primary is not None:
        if cleanup_failures:
            raise Experiment002SupervisorError(
                "raw posix_spawn failed with cleanup errors"
            ) from primary
        raise primary
    if cleanup_failures:
        raise Experiment002SupervisorError(
            "raw posix_spawn completed with cleanup errors"
        ) from cleanup_failures[0]
    if spawned_pid < 1:
        _fail("raw posix_spawn returned an invalid PID")
    return spawned_pid


class _RealKernel:
    def parent_affinity(self) -> set[int]:
        return os.sched_getaffinity(0)

    def parent_child_pids(self) -> tuple[int, ...]:
        return _read_parent_child_pids()

    def process_affinity(self, pid: int) -> set[int]:
        try:
            return os.sched_getaffinity(pid)
        except ProcessLookupError:
            raise
        except OSError as error:
            raise Experiment002SupervisorError(
                "cannot inspect child CPU affinity"
            ) from error

    def monotonic_ns(self) -> int:
        return time.monotonic_ns()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def spawn(
        self,
        executable_path: str,
        argv: tuple[str, ...],
        environment: Mapping[str, str],
        file_actions: tuple[_FileAction, ...],
        source_bundle_fd: int,
    ) -> int:
        try:
            return _raw_posix_spawn(
                executable_path,
                argv,
                dict(environment),
                file_actions,
                source_bundle_fd,
            )
        except OSError as error:
            raise Experiment002SupervisorError(
                "registered child posix_spawn failed"
            ) from error

    def wait4(self, pid: int, options: int) -> _WaitResult:
        waited, status_value, usage = os.wait4(pid, options)
        maximum = usage.ru_maxrss
        if type(maximum) is not int:
            _fail("wait4 ru_maxrss is not an exact integer")
        return _WaitResult(waited, status_value, maximum)

    def kill_process_group(self, pid: int) -> None:
        os.killpg(pid, signal.SIGKILL)

    def kill_process(self, pid: int) -> None:
        os.kill(pid, signal.SIGKILL)

    def sample_process(self, pid: int) -> _ProcessSample:
        return _read_process_sample(pid)


_REAL_KERNEL: Final = _RealKernel()


def _duplicate_safe_descriptor(
    descriptor: int,
    *,
    minimum: int = _SAFE_SOURCE_FD_MINIMUM,
) -> int:
    if (
        type(descriptor) is not int
        or descriptor < 0
        or type(minimum) is not int
        or minimum < _SAFE_SOURCE_FD_MINIMUM
    ):
        _fail("descriptor cannot be duplicated safely")
    try:
        duplicated = fcntl.fcntl(descriptor, fcntl.F_DUPFD_CLOEXEC, minimum)
    except OSError as error:
        raise Experiment002SupervisorError(
            "descriptor cannot be duplicated safely"
        ) from error
    try:
        inheritable = os.get_inheritable(duplicated)
    except OSError as error:
        with suppress(OSError):
            os.close(duplicated)
        raise Experiment002SupervisorError(
            "duplicated descriptor flags cannot be inspected"
        ) from error
    if duplicated < minimum or duplicated == descriptor or inheritable:
        with suppress(OSError):
            os.close(duplicated)
        _fail("safe descriptor duplication returned an invalid descriptor")
    return duplicated


def _open_independent_source_bundle_descriptor(descriptor: int) -> int:
    """Reopen the bundle onto an OFD whose offset the caller does not share."""

    caller_before = _require_source_bundle_descriptor(descriptor)
    reopened = -1
    independent = -1
    try:
        try:
            reopened = os.open(
                f"/proc/self/fd/{descriptor}",
                os.O_RDONLY | os.O_CLOEXEC,
            )
        except OSError as error:
            raise Experiment002SupervisorError(
                "source bundle cannot be reopened onto an independent descriptor"
            ) from error
        if reopened < _SAFE_SOURCE_FD_MINIMUM:
            independent = _duplicate_safe_descriptor(reopened)
            os.close(reopened)
            reopened = -1
        else:
            independent = reopened
            reopened = -1
        if independent == descriptor:
            _fail("source bundle reopen did not return an independent descriptor")
        caller_after = _require_source_bundle_descriptor(descriptor)
        independent_frame = _require_source_bundle_descriptor(
            independent,
            safe_source=True,
        )
        if caller_before != caller_after or caller_after != independent_frame:
            _fail("independently reopened source bundle does not exactly match caller")
        result = independent
        independent = -1
        return result
    finally:
        if reopened >= 0:
            with suppress(OSError):
                os.close(reopened)
        if independent >= 0:
            with suppress(OSError):
                os.close(independent)


def _build_file_actions(
    *,
    parent_channel_fd: int,
    child_channel_fd: int,
    null_fd: int,
    runner_fd: int,
    python_fd: int,
    taskset_exec_fd: int,
    source_bundle_fd: int,
) -> tuple[_FileAction, ...]:
    descriptors = (
        parent_channel_fd,
        child_channel_fd,
        null_fd,
        runner_fd,
        python_fd,
        taskset_exec_fd,
        source_bundle_fd,
    )
    if any(type(value) is not int or value < 0 for value in descriptors):
        _fail("spawn descriptor is invalid")
    if len(set(descriptors)) != len(descriptors):
        _fail("spawn descriptors must be distinct")
    if any(
        descriptor < _SAFE_SOURCE_FD_MINIMUM
        for descriptor in (
            child_channel_fd,
            null_fd,
            runner_fd,
            python_fd,
            taskset_exec_fd,
            source_bundle_fd,
        )
    ):
        _fail("spawn sources must be safe duplicated descriptors")
    return (
        (os.POSIX_SPAWN_DUP2, null_fd, 0),
        (os.POSIX_SPAWN_DUP2, null_fd, 1),
        (os.POSIX_SPAWN_DUP2, null_fd, 2),
        (os.POSIX_SPAWN_DUP2, child_channel_fd, _CHILD_CONTROL_FD),
        (os.POSIX_SPAWN_DUP2, runner_fd, _CHILD_RUNNER_FD),
        (os.POSIX_SPAWN_DUP2, python_fd, _CHILD_PYTHON_FD),
        (os.POSIX_SPAWN_DUP2, taskset_exec_fd, _CHILD_TASKSET_FD),
        (
            os.POSIX_SPAWN_DUP2,
            source_bundle_fd,
            _CHILD_SOURCE_BUNDLE_FD,
        ),
    )


def _registered_argv(plan: _LaunchPlan, cpu_ids: tuple[int, int]) -> tuple[str, ...]:
    if (
        type(cpu_ids) is not tuple
        or len(cpu_ids) != 2
        or any(type(value) is not int or value < 0 for value in cpu_ids)
        or cpu_ids[0] >= cpu_ids[1]
    ):
        _fail("captured CPU IDs are invalid")
    return (
        plan.taskset_executable,
        "--cpu-list",
        f"{cpu_ids[0]},{cpu_ids[1]}",
        plan.python_executable,
        "-I",
        "-S",
        "-B",
        plan.runner_entrypoint,
    )


def _pinned_spawn_argv(
    plan: _LaunchPlan,
    cpu_ids: tuple[int, int],
) -> tuple[str, ...]:
    _registered_argv(plan, cpu_ids)
    return (
        plan.taskset_executable,
        "--cpu-list",
        f"{cpu_ids[0]},{cpu_ids[1]}",
        f"/proc/self/fd/{_CHILD_PYTHON_FD}",
        "-I",
        "-S",
        "-B",
        "-c",
        _RUNNER_BOOTSTRAP,
    )


def _kernel_child_journal(kernel: _Kernel) -> tuple[int, ...]:
    children = kernel.parent_child_pids()
    if (
        type(children) is not tuple
        or len(children) != len(set(children))
        or any(
            type(child) is not int or child < 1 or child == os.getpid()
            for child in children
        )
    ):
        _fail("parent child journal is invalid")
    return children


def _wait4_retry(kernel: _Kernel, pid: int, options: int) -> _WaitResult:
    while True:
        try:
            result = kernel.wait4(pid, options)
        except InterruptedError:
            continue
        if type(result) is not _WaitResult:
            _fail("wait4 returned an invalid result type")
        if result.pid not in {0, pid}:
            _fail("wait4 returned an unrelated PID")
        if result.pid == 0 and options == 0:
            _fail("blocking wait4 returned no child")
        if result.pid == pid:
            _require_exact_integer(result.status, "wait status")
            _require_exact_integer(result.maximum_rss_kib, "wait4 maximum RSS")
        return result


def _record_reap(handle: _ChildHandle, result: _WaitResult) -> None:
    if handle.reaped or handle.wait_result is not None or result.pid != handle.pid:
        _fail("child reap state is ambiguous")
    handle.reaped = True
    handle.wait_result = result


def _kill_and_reap(handle: _ChildHandle, kernel: _Kernel) -> None:
    if handle.reaped:
        return
    signal_failures: list[BaseException] = []
    for operation in (kernel.kill_process_group, kernel.kill_process):
        try:
            operation(handle.pid)
        except KeyboardInterrupt:
            signal_failures.append(KeyboardInterrupt())
        except ProcessLookupError:
            pass
        except OSError as error:
            if error.errno != errno.ESRCH:
                signal_failures.append(error)
    try:
        started = kernel.monotonic_ns()
    except KeyboardInterrupt:
        started = kernel.monotonic_ns()
    while True:
        try:
            result = _wait4_retry(kernel, handle.pid, os.WNOHANG)
        except KeyboardInterrupt:
            continue
        if result.pid == handle.pid:
            _record_reap(handle, result)
            return
        try:
            now = kernel.monotonic_ns()
        except KeyboardInterrupt:
            continue
        if now < started:
            _fail("monotonic clock moved backwards during child termination")
        if now - started > _TERMINATION_NANOSECONDS_MAXIMUM:
            if signal_failures:
                raise Experiment002SupervisorError(
                    "child could not be killed and reaped within the termination bound"
                ) from signal_failures[0]
            _fail("child was not reaped within the termination bound")
        remaining = _TERMINATION_NANOSECONDS_MAXIMUM - (now - started)
        try:
            kernel.sleep(min(_POLL_INTERVAL_NANOSECONDS, remaining) / 1_000_000_000)
        except KeyboardInterrupt:
            continue


def _require_authoritative_wait4() -> None:
    if signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL:
        _fail(
            "SIGCHLD is not default, so authoritative wait4 accounting is unavailable"
        )


def _blockable_signals() -> set[int]:
    blocked = {int(value) for value in signal.valid_signals()}
    blocked.discard(int(signal.SIGKILL))
    blocked.discard(int(signal.SIGSTOP))
    if int(signal.SIGINT) not in blocked:
        _fail("SIGINT is absent from the blockable signal set")
    return blocked


def _capture_safe_signal_dispositions() -> _SignalDispositionFrame:
    frame: list[tuple[int, int]] = []
    for signal_number in sorted(_blockable_signals()):
        disposition = signal.getsignal(signal_number)
        if signal_number == int(signal.SIGINT):
            if disposition is not signal.default_int_handler:
                _fail("parent has an untrusted SIGINT disposition")
            disposition_code = 2
        elif disposition == signal.SIG_DFL:
            disposition_code = 0
        elif disposition == signal.SIG_IGN:
            disposition_code = 1
        else:
            _fail("parent has an untrusted signal disposition")
        frame.append((signal_number, disposition_code))
    return tuple(frame)


def _require_safe_signal_dispositions_unchanged(
    expected: _SignalDispositionFrame,
) -> None:
    if type(expected) is not tuple or _capture_safe_signal_dispositions() != expected:
        _fail("parent signal dispositions changed during supervision")


def _restore_safe_signal_dispositions(frame: _SignalDispositionFrame) -> None:
    if type(frame) is not tuple:
        _fail("parent signal-disposition frame is invalid")
    for signal_number, disposition_code in frame:
        if disposition_code == 0:
            signal.signal(signal_number, signal.SIG_DFL)
        elif disposition_code == 1:
            signal.signal(signal_number, signal.SIG_IGN)
        elif disposition_code == 2 and signal_number == int(signal.SIGINT):
            signal.signal(signal_number, signal.default_int_handler)
        else:
            _fail("parent signal-disposition frame is invalid")


def _require_single_parent_thread() -> None:
    try:
        task_names = os.listdir("/proc/self/task")
    except OSError as error:
        raise Experiment002SupervisorError("cannot enumerate parent tasks") from error
    expected = {str(os.getpid())}
    if set(task_names) != expected or len(task_names) != 1:
        _fail("parent must have exactly one thread before posix_spawn")


def _start_activation(
    callback: _Activation,
    channel: socket.socket,
    pid: int,
) -> tuple[_ActivationState, threading.Thread]:
    if not callable(callback):
        raise TypeError("activation callback must be callable")
    state = _ActivationState(completed=threading.Event(), lock=threading.Lock())

    def activate() -> None:
        error: BaseException | None = None
        try:
            callback(channel, pid)
        except BaseException as caught:
            error = caught
        with state.lock:
            state.error = error
            state.completed.set()

    thread = threading.Thread(
        target=activate,
        name="falsewake-exp002-child-activation",
        daemon=True,
    )
    thread.start()
    return state, thread


def _activation_error(state: _ActivationState) -> BaseException | None:
    with state.lock:
        if not state.completed.is_set():
            return None
    return state.error


def _join_activation_thread(thread: threading.Thread) -> None:
    thread.join(timeout=_ACTIVATION_JOIN_SECONDS_MAXIMUM)
    if thread.is_alive():
        _fail("activation thread survived containment")
    native_id = thread.native_id
    if native_id is None:
        _fail("activation thread lacks its native task ID")
    started = time.monotonic_ns()
    task_path = f"/proc/self/task/{native_id}"
    while os.path.exists(task_path):
        now = time.monotonic_ns()
        if now < started:
            _fail("monotonic clock moved backwards during thread retirement")
        if now - started > _THREAD_RETIRE_NANOSECONDS_MAXIMUM:
            _fail("activation native task did not retire")
        time.sleep(0.001)


def _raise_activation_error(state: _ActivationState) -> None:
    error = _activation_error(state)
    if error is not None:
        raise error


def _require_process_affinity(
    kernel: _Kernel,
    pid: int,
    expected: tuple[int, int],
) -> None:
    observed = kernel.process_affinity(pid)
    if type(observed) is not set or observed != set(expected):
        _fail("child CPU affinity is not the captured two-CPU set")


def _validate_terminal_result(
    wait_result: _WaitResult,
    *,
    elapsed_nanoseconds: int,
    observed_rss_bytes: int,
    output_bytes: int,
    limits: _Limits,
) -> int:
    if not os.WIFEXITED(wait_result.status) or os.WEXITSTATUS(wait_result.status) != 0:
        _fail("registered child did not exit successfully")
    maximum_rss = wait_result.maximum_rss_kib * _KIB
    if maximum_rss > limits.rss_bytes or observed_rss_bytes > limits.rss_bytes:
        _fail("registered child exceeded its peak RSS budget")
    if elapsed_nanoseconds > limits.wall_nanoseconds:
        _fail("registered child exceeded its wall-time budget")
    if output_bytes > limits.output_bytes:
        _fail("registered child exceeded its output budget")
    return max(maximum_rss, observed_rss_bytes)


def _monitor_child(
    handle: _ChildHandle,
    *,
    cpu_ids: tuple[int, int],
    roots: tuple[_PinnedRoot, _PinnedRoot],
    start_nanoseconds: int,
    activation: _ActivationState,
    kernel: _Kernel,
    limits: _Limits,
) -> _ChildResourceMetrics:
    _require_pinned_roots(roots)
    observed_rss = 0
    rss_unavailable_since: int | None = None
    affinity_verified = False
    previous_cycle_started = start_nanoseconds
    while True:
        cycle_started = kernel.monotonic_ns()
        if cycle_started < previous_cycle_started:
            _fail("monotonic clock moved backwards")
        if cycle_started - previous_cycle_started > _POLL_INTERVAL_MAXIMUM_NANOSECONDS:
            _fail("resource observations were separated by more than 100 milliseconds")
        previous_cycle_started = cycle_started
        if (
            activation.completed.is_set()
            and not affinity_verified
            and not handle.reaped
        ):
            _raise_activation_error(activation)
            _require_process_affinity(kernel, handle.pid, cpu_ids)
            affinity_verified = True
        if handle.reaped:
            if handle.wait_result is None:
                _fail("reaped child lacks its terminal wait4 result")
            wait_result = handle.wait_result
        else:
            wait_result = _wait4_retry(kernel, handle.pid, os.WNOHANG)
        if wait_result.pid == handle.pid:
            if not handle.reaped:
                _record_reap(handle, wait_result)
            end = kernel.monotonic_ns()
            elapsed = end - start_nanoseconds
            if elapsed < 0:
                _fail("monotonic clock moved backwards")
            if not activation.completed.wait(
                min(limits.poll_nanoseconds / 1_000_000_000, 0.1)
            ):
                _fail("child exited before activation completed")
            _raise_activation_error(activation)
            if not affinity_verified:
                _fail("child exited before its exact CPU affinity was verified")
            output_bytes = _account_pinned_roots(roots, limits.output_bytes)
            cycle_finished = kernel.monotonic_ns()
            if cycle_finished < cycle_started:
                _fail("monotonic clock moved backwards")
            if cycle_finished - cycle_started > _POLL_INTERVAL_MAXIMUM_NANOSECONDS:
                _fail("one resource sampling cycle exceeded 100 milliseconds")
            peak = _validate_terminal_result(
                wait_result,
                elapsed_nanoseconds=elapsed,
                observed_rss_bytes=observed_rss,
                output_bytes=output_bytes,
                limits=limits,
            )
            return _ChildResourceMetrics(
                pid=handle.pid,
                cpu_ids=cpu_ids,
                elapsed_nanoseconds=elapsed,
                maximum_rss_bytes=peak,
                output_and_scratch_bytes=output_bytes,
            )

        elapsed = cycle_started - start_nanoseconds
        if elapsed < 0:
            _fail("monotonic clock moved backwards")
        if elapsed > limits.wall_nanoseconds:
            _fail("registered child exceeded its wall-time budget")
        if activation.completed.is_set():
            _raise_activation_error(activation)
            if not affinity_verified:
                _require_process_affinity(kernel, handle.pid, cpu_ids)
                affinity_verified = True
        try:
            sample = kernel.sample_process(handle.pid)
        except ProcessLookupError:
            raced = _wait4_retry(kernel, handle.pid, os.WNOHANG)
            if raced.pid == handle.pid:
                _record_reap(handle, raced)
                continue
            _fail("child disappeared before it could be reaped")
        if type(sample) is not _ProcessSample:
            _fail("process sampler returned an invalid result")
        if type(sample.child_pids) is not tuple or any(
            type(value) is not int or value < 1 for value in sample.child_pids
        ):
            _fail("process sampler returned invalid child PIDs")
        # This catches persistent descendants and is deliberately defense in depth.
        # The later child bootstrap must install the registered pre-numerics seccomp
        # filter to close the fork-and-exit gap between two procfs observations.
        if sample.child_pids:
            _fail("registered worker created a child process")
        if (
            rss_unavailable_since is not None
            and cycle_started - rss_unavailable_since
            > _POLL_INTERVAL_MAXIMUM_NANOSECONDS
        ):
            _fail("child RSS remained unavailable for more than 100 milliseconds")
        if sample.rss_bytes is None:
            if rss_unavailable_since is None:
                rss_unavailable_since = cycle_started
        else:
            _require_exact_integer(sample.rss_bytes, "sampled RSS")
            observed_rss = max(observed_rss, sample.rss_bytes)
            if observed_rss > limits.rss_bytes:
                _fail("registered child exceeded its observed RSS budget")
        _account_pinned_roots(roots, limits.output_bytes)

        cycle_finished = kernel.monotonic_ns()
        if cycle_finished < cycle_started:
            _fail("monotonic clock moved backwards")
        cycle_nanoseconds = cycle_finished - cycle_started
        if cycle_nanoseconds > _POLL_INTERVAL_MAXIMUM_NANOSECONDS:
            _fail("one resource sampling cycle exceeded 100 milliseconds")
        if (
            rss_unavailable_since is not None
            and cycle_finished - rss_unavailable_since
            > _POLL_INTERVAL_MAXIMUM_NANOSECONDS
        ):
            _fail("child RSS remained unavailable for more than 100 milliseconds")
        if sample.rss_bytes is not None:
            rss_unavailable_since = None
        remaining = limits.wall_nanoseconds - (cycle_finished - start_nanoseconds)
        if remaining < 0:
            _fail("registered child exceeded its wall-time budget")
        sleep_nanoseconds = min(
            max(0, limits.poll_nanoseconds - cycle_nanoseconds),
            remaining,
        )
        kernel.sleep(sleep_nanoseconds / 1_000_000_000)


def _close_activation_channel(channel: socket.socket) -> None:
    with suppress(OSError):
        channel.shutdown(socket.SHUT_RDWR)
    channel.close()


def _supervise_child_masked(
    cpu_ids: tuple[int, int],
    activation_callback: _Activation,
    source_bundle_fd: int,
    result_binding: ChildResultBinding,
    /,
    *,
    signal_dispositions: _SignalDispositionFrame,
    plan: _LaunchPlan = _REGISTERED_PLAN,
    limits: _Limits = _REGISTERED_LIMITS,
    kernel: _Kernel = _REAL_KERNEL,
) -> _SupervisedChildResult:
    """Run one fixed child with registered resources and signals already blocked."""

    expected_result_binding = _snapshot_child_result_binding(result_binding)
    _require_plan(plan)
    _require_limits(limits)
    _require_source_bundle_descriptor(source_bundle_fd)
    _registered_argv(plan, cpu_ids)
    _require_authoritative_wait4()
    _require_single_parent_thread()
    _require_safe_signal_dispositions_unchanged(signal_dispositions)
    _validate_launch_surface(plan)
    staging_root: _PinnedRoot | None = None
    scratch_root: _PinnedRoot | None = None
    parent_channel: socket.socket | None = None
    child_channel: socket.socket | None = None
    null_fd = -1
    taskset_fd = -1
    python_fd = -1
    runner_fd = -1
    child_source_fd = -1
    null_source_fd = -1
    taskset_source_fd = -1
    python_source_fd = -1
    runner_source_fd = -1
    bundle_source_fd = -1
    handle: _ChildHandle | None = None
    journaled_handles: list[_ChildHandle] = []
    activation_state: _ActivationState | None = None
    activation_thread: threading.Thread | None = None
    try:
        try:
            staging_root = _pin_existing_root(plan.staging_root)
        except Experiment002SupervisorError as error:
            raise Experiment002SupervisorError(
                "experiment staging root is absent or invalid"
            ) from error
        scratch_root = _create_and_pin_fresh_root(plan.scratch_root)
        roots = (scratch_root, staging_root)
        parent_channel, child_channel = socket.socketpair(
            socket.AF_UNIX,
            socket.SOCK_SEQPACKET,
        )
        null_fd = os.open(_DEVNULL, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
        if not stat.S_ISCHR(os.fstat(null_fd).st_mode):
            _fail("/dev/null is not a character device")
        taskset_fd, python_fd, runner_fd = _pin_launch_files(plan)
        bundle_source_fd = _open_independent_source_bundle_descriptor(source_bundle_fd)
        child_source_fd = _duplicate_safe_descriptor(child_channel.fileno())
        null_source_fd = _duplicate_safe_descriptor(null_fd)
        taskset_source_fd = _duplicate_safe_descriptor(taskset_fd)
        python_source_fd = _duplicate_safe_descriptor(python_fd)
        runner_source_fd = _duplicate_safe_descriptor(runner_fd)
        executable_path = f"/proc/self/fd/{_CHILD_TASKSET_FD}"
        argv = _pinned_spawn_argv(plan, cpu_ids)
        environment = dict(plan.environment_items)
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, _blockable_signals())
        try:
            _require_authoritative_wait4()
            _require_single_parent_thread()
            _require_safe_signal_dispositions_unchanged(signal_dispositions)
            if _kernel_child_journal(kernel):
                _fail("parent already has a child before the registered posix_spawn")
            _require_safe_signal_dispositions_unchanged(signal_dispositions)
            file_actions = _build_file_actions(
                parent_channel_fd=parent_channel.fileno(),
                child_channel_fd=child_source_fd,
                null_fd=null_source_fd,
                runner_fd=runner_source_fd,
                python_fd=python_source_fd,
                taskset_exec_fd=taskset_source_fd,
                source_bundle_fd=bundle_source_fd,
            )
            start = kernel.monotonic_ns()
            try:
                _require_matching_source_bundle_descriptors(
                    source_bundle_fd,
                    bundle_source_fd,
                )
                pid = kernel.spawn(
                    executable_path,
                    argv,
                    environment,
                    file_actions,
                    bundle_source_fd,
                )
            except BaseException:
                ambiguous_children = _kernel_child_journal(kernel)
                journaled_handles.extend(
                    _ChildHandle(pid=child) for child in ambiguous_children
                )
                raise
            journaled_children = _kernel_child_journal(kernel)
            journaled_handles.extend(
                _ChildHandle(pid=child) for child in journaled_children
            )
            if len(journaled_handles) == 1:
                handle = journaled_handles[0]
            if type(pid) is not int or pid < 1 or pid == os.getpid():
                _fail("posix_spawn returned an invalid child PID")
            if journaled_children != (pid,):
                _fail(
                    "posix_spawn child journal does not contain the exact returned PID"
                )
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        if handle is None:
            _fail("registered child lacks its authoritative journal handle")
        child_channel.close()
        child_channel = None
        os.close(null_fd)
        null_fd = -1
        os.close(child_source_fd)
        child_source_fd = -1
        os.close(null_source_fd)
        null_source_fd = -1
        os.close(taskset_fd)
        taskset_fd = -1
        os.close(python_fd)
        python_fd = -1
        os.close(runner_fd)
        runner_fd = -1
        os.close(taskset_source_fd)
        taskset_source_fd = -1
        os.close(python_source_fd)
        python_source_fd = -1
        os.close(runner_source_fd)
        runner_source_fd = -1
        os.close(bundle_source_fd)
        bundle_source_fd = -1
        activation_state, activation_thread = _start_activation(
            activation_callback,
            parent_channel,
            pid,
        )
        activation_state.completed.wait(_ACTIVATION_START_GRACE_SECONDS)
        resource_metrics = _monitor_child(
            handle,
            cpu_ids=cpu_ids,
            roots=roots,
            start_nanoseconds=start,
            activation=activation_state,
            kernel=kernel,
            limits=limits,
        )
        _join_activation_thread(activation_thread)
        before_load_bytes = _account_pinned_roots(roots, limits.output_bytes)
        if before_load_bytes != resource_metrics.output_and_scratch_bytes:
            _fail("supervised output changed before child-result loading")
        verified_child_result = load_registered_child_result(
            Path(scratch_root.path),
            expected_result_binding,
        )
        if type(verified_child_result) is not VerifiedChildResult:
            _fail("child-result loader returned an invalid exact type")
        verification = cast(
            Callable[[VerifiedChildResult], object],
            verify_verified_child_result,
        )(verified_child_result)
        if verification is not None:
            _fail("child-result verifier returned an unexpected value")
        after_load_bytes = _account_pinned_roots(roots, limits.output_bytes)
        if after_load_bytes != resource_metrics.output_and_scratch_bytes:
            _fail("supervised output changed during child-result loading")
        result = _SupervisedChildResult(
            pid=resource_metrics.pid,
            cpu_ids=resource_metrics.cpu_ids,
            elapsed_nanoseconds=resource_metrics.elapsed_nanoseconds,
            maximum_rss_bytes=resource_metrics.maximum_rss_bytes,
            output_and_scratch_bytes=resource_metrics.output_and_scratch_bytes,
            verified_child_result=verified_child_result,
        )
        _cleanup_pinned_scratch(scratch_root)
        return result
    except BaseException as primary:
        cleanup_failures: list[BaseException] = []
        if parent_channel is not None:
            try:
                _close_activation_channel(parent_channel)
            except BaseException as error:
                cleanup_failures.append(error)
            parent_channel = None
        for journaled_handle in journaled_handles:
            try:
                _kill_and_reap(journaled_handle, kernel)
            except BaseException as error:
                cleanup_failures.append(error)
        if activation_thread is not None:
            try:
                _join_activation_thread(activation_thread)
            except BaseException as error:
                cleanup_failures.append(error)
        if scratch_root is not None and staging_root is not None:
            try:
                _recover_pinned_parent((scratch_root, staging_root))
            except BaseException as error:
                cleanup_failures.append(error)
        if staging_root is not None:
            try:
                if journaled_handles:
                    _cleanup_pinned_staging(staging_root)
                else:
                    _restore_pinned_root_name(staging_root)
            except BaseException as error:
                cleanup_failures.append(error)
        if scratch_root is not None:
            try:
                _cleanup_pinned_scratch(scratch_root)
            except BaseException as error:
                cleanup_failures.append(error)
        if cleanup_failures:
            raise Experiment002SupervisorError(
                "child failure was contained with cleanup errors"
            ) from primary
        raise
    finally:
        if child_channel is not None:
            with suppress(OSError):
                child_channel.close()
        if parent_channel is not None:
            with suppress(OSError):
                _close_activation_channel(parent_channel)
        if null_fd >= 0:
            with suppress(OSError):
                os.close(null_fd)
        if child_source_fd >= 0:
            with suppress(OSError):
                os.close(child_source_fd)
        if null_source_fd >= 0:
            with suppress(OSError):
                os.close(null_source_fd)
        if taskset_fd >= 0:
            with suppress(OSError):
                os.close(taskset_fd)
        if python_fd >= 0:
            with suppress(OSError):
                os.close(python_fd)
        if runner_fd >= 0:
            with suppress(OSError):
                os.close(runner_fd)
        if taskset_source_fd >= 0:
            with suppress(OSError):
                os.close(taskset_source_fd)
        if python_source_fd >= 0:
            with suppress(OSError):
                os.close(python_source_fd)
        if runner_source_fd >= 0:
            with suppress(OSError):
                os.close(runner_source_fd)
        if bundle_source_fd >= 0:
            with suppress(OSError):
                os.close(bundle_source_fd)
        if scratch_root is not None:
            with suppress(OSError):
                scratch_root.close()
        if staging_root is not None:
            with suppress(OSError):
                staging_root.close()


def _contain_unregistered_parent_children(kernel: _Kernel) -> bool:
    contained_any = False
    for _attempt in range(4):
        children = _kernel_child_journal(kernel)
        if not children:
            return contained_any
        contained_any = True
        for child in children:
            _kill_and_reap(_ChildHandle(pid=child), kernel)
    _fail("unauthorized parent children persisted through containment")


def _recover_registered_staging_parent(
    root: _PinnedRoot,
    scratch_path: str,
    /,
) -> None:
    """Recover the pinned registered parent without requiring a scratch lease."""

    if type(root) is not _PinnedRoot:
        raise TypeError("staging lease must be an exact _PinnedRoot")
    parent = root.parent
    if not parent.components:
        _verify_pinned_parent(parent)
        return
    scratch_parent, scratch_leaf = _parent_and_leaf(scratch_path)
    staging_parent, staging_leaf = _parent_and_leaf(root.path)
    if scratch_parent != staging_parent or scratch_leaf == staging_leaf:
        _fail("registered output lease paths are inconsistent")
    managed_leaves = (scratch_leaf, staging_leaf)
    for _attempt in range(len(parent.components)):
        mismatch_index = _first_parent_name_mismatch(parent)
        if mismatch_index is None:
            _verify_pinned_parent(parent)
            return
        owner = parent.descriptors[mismatch_index]
        registered_component = parent.components[mismatch_index]
        expected_identity = parent.identities[mismatch_index + 1]
        matches = _names_for_identity(owner, expected_identity)
        removed = _clean_registered_replacement_chain(
            parent,
            mismatch_index,
            managed_leaves,
        )
        if not removed:
            _fail("replacement ancestor contains unmanaged entries")
        if len(matches) != 1:
            _fail("pinned ancestor cannot be located uniquely")
        while True:
            try:
                os.rename(
                    matches[0],
                    registered_component,
                    src_dir_fd=owner,
                    dst_dir_fd=owner,
                )
                break
            except InterruptedError:
                continue
        restored = os.stat(
            registered_component,
            dir_fd=owner,
            follow_symlinks=False,
        )
        if _identity_frame(restored) != expected_identity:
            _fail("pinned ancestor identity changed while restoring its name")
    _fail("pinned staging parent chain could not be recovered")


def _remove_pinned_registered_staging(root: _PinnedRoot, /) -> None:
    """Remove one empty pinned staging inode and any fixed-name replacement."""

    if type(root) is not _PinnedRoot:
        raise TypeError("staging lease must be an exact _PinnedRoot")
    _verify_pinned_parent(root.parent)
    matches = _names_for_identity(root.parent_fd, root.identity)
    _parent, registered_leaf = _parent_and_leaf(root.path)
    if len(matches) > 1:
        _fail("pinned staging identity appears under multiple parent names")
    if not matches:
        _remove_named_tree(root.parent_fd, registered_leaf)
        _fail(
            "pinned staging moved outside its registered parent after its "
            "contents were scrubbed"
        )
    owned_name = matches[0]
    if owned_name != registered_leaf:
        _remove_named_tree(root.parent_fd, registered_leaf)
    named = os.stat(owned_name, dir_fd=root.parent_fd, follow_symlinks=False)
    if _identity_frame(named) != root.identity:
        _fail("pinned staging identity changed before removal")
    _remove_named_tree(root.parent_fd, owned_name)
    if owned_name != registered_leaf:
        _remove_named_tree(root.parent_fd, registered_leaf)


def _bind_registered_parent_lifecycle() -> tuple[
    Callable[[], None],
    Callable[[], None],
    Callable[[], None],
    Callable[[], None],
    Callable[[], None],
    Callable[[], None],
]:
    """Bind the one-attempt four-child parent lifecycle to hidden state."""

    fresh = 0
    preparing = 1
    active = 2
    child_running = 3
    cleaning = 4
    closed = 5
    failed_without_lease = 6
    failed_with_lease = 7
    maximum_children = 4

    plan = _REGISTERED_PLAN
    _require_plan(plan)
    if (
        plan.temporary_root != _TEMP_ROOT
        or plan.scratch_root != _SCRATCH_ROOT
        or plan.staging_root != _STAGING_ROOT
    ):
        _fail("registered experiment output paths are not exact")
    scratch_path = plan.scratch_root
    staging_path = plan.staging_root
    scratch_parent, scratch_leaf = _parent_and_leaf(scratch_path)
    staging_parent, _staging_leaf = _parent_and_leaf(staging_path)
    if scratch_parent != staging_parent:
        _fail("registered output roots do not share one parent")

    kernel = _REAL_KERNEL
    require_absent = _require_output_root_absent
    create_and_pin = _create_and_pin_fresh_root
    recover_parent = _recover_registered_staging_parent
    restore_name = _restore_pinned_root_name
    scrub_directory = _remove_directory_contents
    remove_staging = _remove_pinned_registered_staging
    remove_named_tree = _remove_named_tree
    require_wait4 = _require_authoritative_wait4
    require_single_parent_thread = _require_single_parent_thread
    child_journal = _kernel_child_journal
    kill_and_reap = _kill_and_reap
    child_handle_type = _ChildHandle
    exact_type = type
    get_process_id = os.getpid
    get_thread_id = threading.get_ident
    error_type = Experiment002SupervisorError
    base_exception_type = BaseException
    module_globals = globals()
    state_type = _RegisteredParentLifecycleState
    pinned_root_type = _PinnedRoot
    pinned_parent_type = _PinnedParent
    state_field_names = (
        "owner_thread",
        "phase",
        "lease",
        "lease_frame",
        "namespace_owned",
        "children_started",
        "children_completed",
        "cleanup_attempted",
    )
    state_namespace = state_type.__dict__
    state_namespace_names = tuple(sorted(state_namespace))
    state_namespace_values = tuple(
        state_namespace[name] for name in state_namespace_names
    )
    state_descriptors = tuple(state_namespace.get(name) for name in state_field_names)
    if any(
        exact_type(descriptor) is not MemberDescriptorType
        for descriptor in state_descriptors
    ):
        raise RuntimeError("registered parent lifecycle state slots are unavailable")
    state_lock = threading.Lock()
    owner_process = get_process_id()
    state = state_type(
        owner_thread=None,
        phase=fresh,
        lease=None,
        lease_frame=None,
        namespace_owned=False,
        children_started=0,
        children_completed=0,
        cleanup_attempted=False,
    )

    def fail_invalid_state(message: str) -> NoReturn:
        raise error_type(f"registered parent lifecycle state is invalid: {message}")

    def require_state_class_authority() -> None:
        current_namespace = state_type.__dict__
        if (
            module_globals.get("_RegisteredParentLifecycleState") is not state_type
            or tuple(sorted(current_namespace)) != state_namespace_names
            or any(
                current_namespace[name] is not state_namespace_values[index]
                for index, name in enumerate(state_namespace_names)
            )
            or any(
                current_namespace.get(name) is not state_descriptors[index]
                for index, name in enumerate(state_field_names)
            )
        ):
            fail_invalid_state("class authority changed")

    def require_state_invariants() -> None:
        require_state_class_authority()
        owner_thread = state.owner_thread
        current_phase = state.phase
        current_lease = state.lease
        current_frame = state.lease_frame
        namespace_is_owned = state.namespace_owned
        started = state.children_started
        completed = state.children_completed
        cleanup_was_attempted = state.cleanup_attempted
        if (
            exact_type(state) is not state_type
            or (
                owner_thread is not None
                and (exact_type(owner_thread) is not int or owner_thread < 1)
            )
            or exact_type(current_phase) is not int
            or current_phase
            not in {
                fresh,
                preparing,
                active,
                child_running,
                cleaning,
                closed,
                failed_without_lease,
                failed_with_lease,
            }
            or exact_type(namespace_is_owned) is not bool
            or exact_type(started) is not int
            or exact_type(completed) is not int
            or not 0 <= completed <= started <= maximum_children
            or started - completed not in {0, 1}
            or exact_type(cleanup_was_attempted) is not bool
        ):
            fail_invalid_state("field types, ranges, or phase are inconsistent")

        lease_is_absent = current_lease is None and current_frame is None
        lease_is_paired = (
            exact_type(current_lease) is pinned_root_type
            and exact_type(current_frame) is tuple
            and len(cast(tuple[object, ...], current_frame)) == 7
            and cast(tuple[object, ...], current_frame)[0] is current_lease
            and cast(tuple[object, ...], current_frame)[1]
            == cast(_PinnedRoot, current_lease).descriptor
            and cast(tuple[object, ...], current_frame)[2]
            == cast(_PinnedRoot, current_lease).identity
            and cast(tuple[object, ...], current_frame)[3]
            is cast(_PinnedRoot, current_lease).parent
            and exact_type(cast(_PinnedRoot, current_lease).parent)
            is pinned_parent_type
            and cast(tuple[object, ...], current_frame)[4]
            == cast(_PinnedRoot, current_lease).parent.descriptors
            and cast(tuple[object, ...], current_frame)[5]
            == cast(_PinnedRoot, current_lease).parent.identities
            and cast(tuple[object, ...], current_frame)[6]
            == cast(_PinnedRoot, current_lease).parent.components
        )
        if not lease_is_absent and not lease_is_paired:
            fail_invalid_state("staging lease and frame are not exactly paired")

        if current_phase == fresh:
            legal = (
                lease_is_absent
                and not namespace_is_owned
                and started == completed == 0
                and not cleanup_was_attempted
            )
        elif current_phase == preparing:
            legal = (
                not namespace_is_owned
                and started == completed == 0
                and not cleanup_was_attempted
            )
        elif current_phase == active:
            legal = (
                lease_is_paired
                and namespace_is_owned
                and started == completed
                and not cleanup_was_attempted
            )
        elif current_phase == child_running:
            legal = (
                lease_is_paired
                and namespace_is_owned
                and started == completed + 1
                and not cleanup_was_attempted
            )
        elif current_phase == cleaning:
            legal = cleanup_was_attempted
        elif current_phase == closed:
            legal = lease_is_absent and cleanup_was_attempted
        elif current_phase == failed_without_lease:
            legal = lease_is_absent
        else:
            legal = lease_is_paired and not cleanup_was_attempted
        if not legal:
            fail_invalid_state("phase and retained authority disagree")

    def enter() -> None:
        if get_process_id() != owner_process:
            raise error_type("registered parent lifecycle was inherited by a fork")
        if not state_lock.acquire(blocking=False):
            raise error_type("registered parent lifecycle call overlaps another call")
        try:
            require_state_invariants()
            current_thread = get_thread_id()
            if exact_type(current_thread) is not int or current_thread < 1:
                raise error_type("registered parent thread identity is invalid")
            if state.owner_thread is None:
                state.owner_thread = current_thread
            elif current_thread != state.owner_thread:
                raise error_type("registered parent lifecycle caller changed")
            require_state_invariants()
        except base_exception_type:
            state_lock.release()
            raise

    def leave() -> None:
        try:
            require_state_invariants()
        finally:
            state_lock.release()

    def poison() -> None:
        state.phase = (
            failed_with_lease if state.lease is not None else failed_without_lease
        )

    def require_lease() -> _PinnedRoot:
        current = state.lease
        frame = state.lease_frame
        if (
            current is None
            or frame is None
            or exact_type(current) is not pinned_root_type
            or exact_type(frame) is not tuple
            or len(frame) != 7
            or frame[0] is not current
            or current.descriptor != frame[1]
            or current.identity != frame[2]
            or current.parent is not frame[3]
            or current.parent.descriptors != frame[4]
            or current.parent.identities != frame[5]
            or current.parent.components != frame[6]
        ):
            raise error_type("registered staging lease authority changed")
        typed_current = current
        opened = os.fstat(typed_current.descriptor)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or _identity_frame(opened) != typed_current.identity
            or stat.S_IMODE(opened.st_mode) != 0o700
        ):
            raise error_type("registered staging lease identity changed")
        recover_parent(typed_current, scratch_path)
        restore_name(typed_current)
        named = os.stat(
            _parent_and_leaf(staging_path)[1],
            dir_fd=typed_current.parent_fd,
            follow_symlinks=False,
        )
        if _identity_frame(named) != typed_current.identity:
            raise error_type("registered staging lease name changed")
        return typed_current

    def contain_direct_children() -> tuple[bool, tuple[BaseException, ...]]:
        observed = False
        failures: list[BaseException] = []
        try:
            require_single_parent_thread()
        except base_exception_type as error:
            failures.append(error)
        try:
            require_wait4()
        except base_exception_type as error:
            failures.append(error)
        for _round in range(4):
            try:
                children = child_journal(kernel)
            except base_exception_type as error:
                failures.append(error)
                break
            if not children:
                return observed, tuple(failures)
            observed = True
            for child in children:
                try:
                    kill_and_reap(child_handle_type(pid=child), kernel)
                except base_exception_type as error:
                    failures.append(error)
        try:
            remaining = child_journal(kernel)
        except base_exception_type as error:
            failures.append(error)
        else:
            if remaining:
                observed = True
                failures.append(
                    error_type(
                        "unauthorized parent children persisted through containment"
                    )
                )
        return observed, tuple(failures)

    def raise_for_quiescence(
        observed: bool,
        failures: tuple[BaseException, ...],
        /,
    ) -> None:
        if failures:
            raise error_type(
                "registered parent quiescence failed during child containment"
            ) from failures[0]
        if observed:
            raise error_type(
                "an unauthorized direct child was contained at a parent boundary"
            )

    def cleanup_owned_lease(*, remove_scratch: bool) -> tuple[BaseException, ...]:
        current = state.lease
        frame = state.lease_frame
        if current is None:
            return ()
        failures: list[BaseException] = []
        descriptor = frame[1] if frame is not None else -1
        identity = frame[2] if frame is not None else None
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISDIR(opened.st_mode) or _identity_frame(opened) != identity:
                raise error_type("registered staging lease changed before cleanup")
            scrub_directory(descriptor)
        except base_exception_type as error:
            failures.append(error)
        try:
            recover_parent(current, scratch_path)
        except base_exception_type as error:
            failures.append(error)
        try:
            remove_staging(current)
        except base_exception_type as error:
            failures.append(error)
        if remove_scratch:
            try:
                remove_named_tree(current.parent_fd, scratch_leaf)
            except base_exception_type as error:
                failures.append(error)
        try:
            current.close()
        except base_exception_type as error:
            failures.append(error)
        state.lease = None
        state.lease_frame = None
        return tuple(failures)

    def _prepare_registered_experiment_staging() -> None:
        enter()
        try:
            if state.phase != fresh:
                raise error_type("registered parent lifecycle prepare was replayed")
            state.phase = preparing
            require_state_invariants()
            try:
                observed, containment_failures = contain_direct_children()
                raise_for_quiescence(observed, containment_failures)
                require_absent(scratch_path)
                require_absent(staging_path)
                created = create_and_pin(staging_path)
                state.lease = created
                state.lease_frame = (
                    created,
                    created.descriptor,
                    created.identity,
                    created.parent,
                    created.parent.descriptors,
                    created.parent.identities,
                    created.parent.components,
                )
                require_absent(scratch_path)
                require_lease()
                state.namespace_owned = True
                state.phase = active
                require_state_invariants()
            except base_exception_type as primary:
                poison()
                cleanup_failures = cleanup_owned_lease(remove_scratch=False)
                state.phase = failed_without_lease
                if cleanup_failures:
                    raise error_type(
                        "registered staging preparation failed with cleanup errors"
                    ) from primary
                raise
        finally:
            leave()

    def _require_registered_parent_quiescence() -> None:
        enter()
        try:
            terminal_phase = state.phase in {
                failed_without_lease,
                failed_with_lease,
            }
            if state.phase not in {
                fresh,
                active,
                closed,
                failed_without_lease,
                failed_with_lease,
            }:
                raise error_type(
                    "registered parent quiescence was requested in an active phase"
                )
            observed, failures = contain_direct_children()
            if observed or failures:
                if state.phase != closed:
                    poison()
                raise_for_quiescence(observed, failures)
            if terminal_phase:
                raise error_type("registered parent lifecycle is terminally failed")
        finally:
            leave()

    def begin_registered_child() -> None:
        enter()
        try:
            if state.phase != active:
                if state.phase != closed:
                    poison()
                raise error_type("registered child requires one active staging lease")
            if state.children_started >= maximum_children:
                poison()
                raise error_type("registered parent attempted more than four children")
            try:
                observed, failures = contain_direct_children()
                raise_for_quiescence(observed, failures)
                require_lease()
            except base_exception_type:
                poison()
                raise
            state.children_started += 1
            state.phase = child_running
            require_state_invariants()
        finally:
            leave()

    def finish_registered_child_success() -> None:
        enter()
        try:
            if state.phase != child_running:
                poison()
                raise error_type("registered child completion phase is invalid")
            try:
                observed, failures = contain_direct_children()
                raise_for_quiescence(observed, failures)
                require_lease()
            except base_exception_type:
                poison()
                raise
            state.children_completed += 1
            if state.children_completed != state.children_started:
                poison()
                raise error_type("registered child completion count is invalid")
            state.phase = active
            require_state_invariants()
        finally:
            leave()

    def finish_registered_child_failure() -> None:
        enter()
        try:
            if state.phase != child_running:
                poison()
                raise error_type("registered child failure phase is invalid")
            boundary_failures: list[BaseException] = []
            observed, failures = contain_direct_children()
            if observed:
                boundary_failures.append(
                    error_type(
                        "an unauthorized direct child was contained after child failure"
                    )
                )
            boundary_failures.extend(failures)
            try:
                require_lease()
            except base_exception_type as error:
                boundary_failures.append(error)
            poison()
            if boundary_failures:
                raise error_type(
                    "registered child failure crossed a damaged parent boundary"
                ) from boundary_failures[0]
        finally:
            leave()

    def _cleanup_registered_experiment_output_roots() -> None:
        enter()
        try:
            if state.phase == closed:
                return
            if state.phase in {child_running, cleaning}:
                raise error_type(
                    "registered output cleanup overlaps an active lifecycle phase"
                )
            if state.cleanup_attempted:
                raise error_type("failed registered output cleanup was replayed")
            state.cleanup_attempted = True
            state.phase = cleaning
            require_state_invariants()
            failures: list[BaseException] = []
            observed_before, before_failures = contain_direct_children()
            if observed_before:
                failures.append(
                    error_type(
                        "an unauthorized direct child was contained before cleanup"
                    )
                )
            failures.extend(before_failures)
            failures.extend(cleanup_owned_lease(remove_scratch=state.namespace_owned))
            require_state_invariants()
            observed_after, after_failures = contain_direct_children()
            if observed_after:
                failures.append(
                    error_type(
                        "an unauthorized direct child was contained after cleanup"
                    )
                )
            failures.extend(after_failures)
            if failures:
                state.phase = failed_without_lease
                raise error_type(
                    "registered output cleanup failed after all containment phases"
                ) from failures[0]
            state.phase = closed
            require_state_invariants()
        finally:
            leave()

    return (
        _prepare_registered_experiment_staging,
        _require_registered_parent_quiescence,
        _cleanup_registered_experiment_output_roots,
        begin_registered_child,
        finish_registered_child_success,
        finish_registered_child_failure,
    )


(
    _prepare_registered_experiment_staging,
    _require_registered_parent_quiescence,
    _cleanup_registered_experiment_output_roots,
    _begin_registered_supervisor_child,
    _finish_registered_supervisor_child_success,
    _finish_registered_supervisor_child_failure,
) = _bind_registered_parent_lifecycle()
del _bind_registered_parent_lifecycle


def _supervise_child(
    cpu_ids: tuple[int, int],
    activation_callback: _Activation,
    source_bundle_fd: int,
    result_binding: ChildResultBinding,
    /,
    *,
    plan: _LaunchPlan = _REGISTERED_PLAN,
    limits: _Limits = _REGISTERED_LIMITS,
    kernel: _Kernel = _REAL_KERNEL,
) -> _SupervisedChildResult:
    """Run one child while the parent signal surface remains fully blocked."""

    expected_result_binding = _snapshot_child_result_binding(result_binding)
    signal_dispositions = _capture_safe_signal_dispositions()
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, _blockable_signals())
    result: _SupervisedChildResult | None = None
    primary: BaseException | None = None
    boundary_violations: list[BaseException] = []
    cleanup_failures: list[BaseException] = []
    unauthorized_children = False
    safe_to_unmask = True
    try:
        result = _supervise_child_masked(
            cpu_ids,
            activation_callback,
            source_bundle_fd,
            expected_result_binding,
            signal_dispositions=signal_dispositions,
            plan=plan,
            limits=limits,
            kernel=kernel,
        )
    except BaseException as error:
        primary = error

    try:
        _require_safe_signal_dispositions_unchanged(signal_dispositions)
    except BaseException as error:
        boundary_violations.append(error)
    try:
        _restore_safe_signal_dispositions(signal_dispositions)
    except BaseException as error:
        cleanup_failures.append(error)
        safe_to_unmask = False
    try:
        unauthorized_children |= _contain_unregistered_parent_children(kernel)
    except BaseException as error:
        cleanup_failures.append(error)

    try:
        _require_safe_signal_dispositions_unchanged(signal_dispositions)
    except BaseException as error:
        boundary_violations.append(error)
        try:
            _restore_safe_signal_dispositions(signal_dispositions)
            _require_safe_signal_dispositions_unchanged(signal_dispositions)
        except BaseException as restore_error:
            cleanup_failures.append(restore_error)
            safe_to_unmask = False

    if safe_to_unmask:
        try:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            unauthorized_children |= _contain_unregistered_parent_children(kernel)
        except BaseException as error:
            if primary is None:
                primary = error
            else:
                cleanup_failures.append(error)
    else:
        cleanup_failures.append(
            Experiment002SupervisorError(
                "unsafe signal dispositions prevented parent mask restoration"
            )
        )

    if unauthorized_children:
        boundary_violations.append(
            Experiment002SupervisorError(
                "an unauthorized direct child was contained during supervision"
            )
        )
    if primary is None and boundary_violations:
        primary = boundary_violations[0]
    elif boundary_violations:
        cleanup_failures.extend(boundary_violations)
    if primary is not None:
        if cleanup_failures:
            raise Experiment002SupervisorError(
                "supervision failed with parent-boundary containment errors"
            ) from primary
        raise primary
    if cleanup_failures:
        raise Experiment002SupervisorError(
            "parent-boundary cleanup failed after child supervision"
        ) from cleanup_failures[0]
    if result is None:
        _fail("supervision completed without an exact child result")
    return result


def _make_supervise_registered_child() -> Callable[..., _SupervisedChildResult]:
    module_globals = globals()
    exact_type = type
    base_exception_type = BaseException
    error_type = Experiment002SupervisorError
    supervise = _supervise_child
    supervise_code = supervise.__code__
    supervise_defaults = supervise.__defaults__
    supervise_keyword_defaults = supervise.__kwdefaults__
    if supervise_keyword_defaults is None:
        raise RuntimeError("registered supervisor defaults are unavailable")
    supervise_keyword_names = tuple(sorted(supervise_keyword_defaults))
    supervise_keyword_values = tuple(
        supervise_keyword_defaults[name] for name in supervise_keyword_names
    )
    supervise_closure = supervise.__closure__
    plan = _REGISTERED_PLAN
    limits = _REGISTERED_LIMITS
    kernel = _REAL_KERNEL
    begin_registered_child = _begin_registered_supervisor_child
    finish_registered_child_success = _finish_registered_supervisor_child_success
    finish_registered_child_failure = _finish_registered_supervisor_child_failure
    plan_type = _LaunchPlan
    limits_type = _Limits
    plan_field_names = (
        "repository_root",
        "temporary_root",
        "scratch_root",
        "staging_root",
        "taskset_executable",
        "python_executable",
        "runner_entrypoint",
        "runner_path",
        "environment_items",
    )
    limits_field_names = (
        "wall_nanoseconds",
        "rss_bytes",
        "output_bytes",
        "poll_nanoseconds",
    )
    plan_namespace = plan_type.__dict__
    limits_namespace = limits_type.__dict__
    plan_descriptors = tuple(plan_namespace.get(name) for name in plan_field_names)
    limits_descriptors = tuple(
        limits_namespace.get(name) for name in limits_field_names
    )
    if any(
        exact_type(descriptor) is not MemberDescriptorType
        for descriptor in (*plan_descriptors, *limits_descriptors)
    ):
        raise RuntimeError("registered supervisor slot authority is unavailable")
    typed_plan_descriptors = cast(
        tuple[MemberDescriptorType, ...],
        plan_descriptors,
    )
    typed_limits_descriptors = cast(
        tuple[MemberDescriptorType, ...],
        limits_descriptors,
    )
    plan_frame = tuple(
        descriptor.__get__(plan, plan_type) for descriptor in typed_plan_descriptors
    )
    limits_frame = tuple(
        descriptor.__get__(limits, limits_type)
        for descriptor in typed_limits_descriptors
    )

    def require_registered_authority(
        current_supervise: object,
        current_plan: object,
        current_limits: object,
        current_kernel: object,
        /,
    ) -> None:
        if (
            current_supervise is not supervise
            or current_plan is not plan
            or current_limits is not limits
            or current_kernel is not kernel
            or module_globals.get("_supervise_child") is not supervise
            or module_globals.get("_REGISTERED_PLAN") is not plan
            or module_globals.get("_REGISTERED_LIMITS") is not limits
            or module_globals.get("_REAL_KERNEL") is not kernel
            or supervise.__code__ is not supervise_code
            or supervise.__defaults__ is not supervise_defaults
            or supervise.__kwdefaults__ is not supervise_keyword_defaults
            or supervise.__closure__ is not supervise_closure
            or tuple(sorted(supervise_keyword_defaults)) != supervise_keyword_names
            or any(
                supervise_keyword_defaults[name] is not supervise_keyword_values[index]
                for index, name in enumerate(supervise_keyword_names)
            )
        ):
            raise error_type("registered supervisor executable authority changed")
        if any(
            plan_type.__dict__.get(name) is not typed_plan_descriptors[index]
            for index, name in enumerate(plan_field_names)
        ) or any(
            limits_type.__dict__.get(name) is not typed_limits_descriptors[index]
            for index, name in enumerate(limits_field_names)
        ):
            raise error_type("registered supervisor class authority changed")
        current_plan_frame = tuple(
            descriptor.__get__(plan, plan_type) for descriptor in typed_plan_descriptors
        )
        current_limits_frame = tuple(
            descriptor.__get__(limits, limits_type)
            for descriptor in typed_limits_descriptors
        )
        if any(
            current_plan_frame[index] is not plan_frame[index]
            for index in range(len(plan_frame))
        ) or any(
            current_limits_frame[index] is not limits_frame[index]
            for index in range(len(limits_frame))
        ):
            raise error_type("registered supervisor plan or limits changed")

    def _supervise_registered_child(
        cpu_ids: tuple[int, int],
        activation_callback: _Activation,
        source_bundle_fd: int,
        result_binding: ChildResultBinding,
        /,
    ) -> _SupervisedChildResult:
        require_registered_authority(supervise, plan, limits, kernel)
        begin_registered_child()
        try:
            result = supervise(
                cpu_ids,
                activation_callback,
                source_bundle_fd,
                result_binding,
                plan=plan,
                limits=limits,
                kernel=kernel,
            )
        except base_exception_type as primary:
            boundary_failures: list[BaseException] = []
            try:
                finish_registered_child_failure()
            except base_exception_type as error:
                boundary_failures.append(error)
            try:
                require_registered_authority(supervise, plan, limits, kernel)
            except base_exception_type as error:
                boundary_failures.append(error)
            if boundary_failures:
                raise error_type(
                    "registered supervision failed after its parent boundary changed"
                ) from primary
            raise
        finish_registered_child_success()
        require_registered_authority(supervise, plan, limits, kernel)
        return result

    return _supervise_registered_child


_supervise_registered_child = _make_supervise_registered_child()
del _make_supervise_registered_child


def _capture_registered_cpu_ids() -> tuple[int, int]:
    """Capture the production CPU pair once for later four-child integration."""

    _validate_launch_surface(_REGISTERED_PLAN)
    return _capture_two_lowest_cpu_ids(_REAL_KERNEL)
