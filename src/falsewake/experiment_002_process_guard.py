"""Import-sealed no-process boundary for an Experiment 002 registered child.

The sealed child bootstrap dynamically imports this module at its exact
post-authority, pre-ACK boundary.  Import itself synchronously installs the
fixed Linux/x86-64 seccomp-BPF program across the thread group before any
public getter exists or module import can return.  The program permits ordinary
computation and future threads, but denies process creation and image
replacement:

* the x32 syscall namespace and ``clone3`` report ``ENOSYS``;
* ``fork``, ``vfork``, ``execve``, and ``execveat`` report ``EPERM``; and
* legacy ``clone`` is permitted only when ``CLONE_THREAD`` is present.

``PR_SET_NO_NEW_PRIVS`` uses a captured ``libc.prctl`` pointer; filter
installation uses captured ``libc.syscall`` and raw x86-64 ``seccomp(2)`` with
``SECCOMP_FILTER_FLAG_TSYNC``.  No syscall number, action, architecture, filter
byte, or runtime identity is accepted from a caller.  A valid capability is
issued only if the child remained one-threaded and pre-numerical throughout
the import audit.  TSYNC applies the fixed filter to every thread that remains
in the thread group when syscall 317 executes; post-install audits then
withhold the capability if a thread or numerical import raced the bootstrap.
Those post-audits are defense-in-depth: an untrusted pre-existing callback may
fork or leave the thread group before TSYNC and is not contained by this module.

This boundary assumes the sealed bootstrap described above.  Arbitrary
pre-existing same-interpreter callbacks, tracing, or mutation are outside its
trust model.  Detected reentry and invariant changes latch failure for this one
module execution and cannot be overwritten by its later success path.  A
failed import is terminal for the sealed caller; Python removes the failed
module, so a deliberate reimport would create fresh module state.
"""

from __future__ import annotations

import builtins
import ctypes
import errno
import hashlib
import os
import struct
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, NoReturn, Protocol, SupportsIndex

if "_PROCESS_GUARD_MODULE_EXECUTION_SENTINEL" in globals():
    raise RuntimeError("the process-guard module cannot be reloaded")
_PROCESS_GUARD_MODULE_EXECUTION_SENTINEL = object()

__all__ = (
    "Experiment002ProcessGuardError",
    "VerifiedChildProcessGuard",
    "get_registered_child_process_guard",
    "verify_verified_child_process_guard",
)

_AUDIT_ARCH_X86_64: Final = 0xC000_003E
_X32_SYSCALL_BIT: Final = 0x4000_0000
_CLONE_THREAD: Final = 0x0001_0000

_SYS_CLONE: Final = 56
_SYS_FORK: Final = 57
_SYS_VFORK: Final = 58
_SYS_EXECVE: Final = 59
_SYS_EXECVEAT: Final = 322
_SYS_CLONE3: Final = 435

_PR_GET_SECCOMP: Final = 21
_PR_SET_NO_NEW_PRIVS: Final = 38
_PR_GET_NO_NEW_PRIVS: Final = 39
_SECCOMP_MODE_FILTER: Final = 2
_SYS_SECCOMP: Final = 317
_SECCOMP_SET_MODE_FILTER: Final = 1
_SECCOMP_FILTER_FLAG_TSYNC: Final = 1

_BPF_LD_W_ABS: Final = 0x20
_BPF_JMP_JEQ_K: Final = 0x15
_BPF_JMP_JSET_K: Final = 0x45
_BPF_RET_K: Final = 0x06
_SECCOMP_RET_KILL_PROCESS: Final = 0x8000_0000
_SECCOMP_RET_ERRNO: Final = 0x0005_0000
_SECCOMP_RET_ALLOW: Final = 0x7FFF_0000

_SECCOMP_DATA_NR_OFFSET: Final = 0
_SECCOMP_DATA_ARCH_OFFSET: Final = 4
_SECCOMP_DATA_ARGUMENT_ZERO_OFFSET: Final = 16

_FILTER_DIGEST_DOMAIN: Final = b"falsewake-exp002-process-guard-filter-v1\0"
_FILTER_FRAME_MAGIC: Final = b"FW2SCMP1"
_REGISTERED_FILTER_SHA256: Final = (
    "e2004b41f7a30715c9ff00456b0e9fd0f1f974500d49aeaa728006d01bacc9ab"
)
_MAX_PROC_STATUS_BYTES: Final = 1 << 20
_NUMERICAL_IMPORT_ROOTS: Final = frozenset(
    {
        "matplotlib",
        "numpy",
        "onnx",
        "onnxruntime",
        "safetensors",
        "scipy",
        "sklearn",
        "soundfile",
        "torch",
    }
)

# sock_filter is exactly ``unsigned short, unsigned char, unsigned char,
# unsigned int``.  Each tuple is therefore also its canonical little-endian
# kernel frame.  Jump offsets below are intentionally local and auditable.
_FILTER_INSTRUCTIONS: Final = (
    (_BPF_LD_W_ABS, 0, 0, _SECCOMP_DATA_ARCH_OFFSET),  # 00: load arch
    (_BPF_JMP_JEQ_K, 1, 0, _AUDIT_ARCH_X86_64),  # 01: x86-64 -> 03
    (_BPF_RET_K, 0, 0, _SECCOMP_RET_KILL_PROCESS),  # 02: wrong arch
    (_BPF_LD_W_ABS, 0, 0, _SECCOMP_DATA_NR_OFFSET),  # 03: load nr
    (_BPF_JMP_JSET_K, 0, 1, _X32_SYSCALL_BIT),  # 04: clear -> 06
    (_BPF_RET_K, 0, 0, _SECCOMP_RET_ERRNO | 38),  # 05: x32 / ENOSYS
    (_BPF_JMP_JEQ_K, 0, 1, _SYS_FORK),  # 06: unequal -> 08
    (_BPF_RET_K, 0, 0, _SECCOMP_RET_ERRNO | 1),  # 07: EPERM
    (_BPF_JMP_JEQ_K, 0, 1, _SYS_VFORK),  # 08: unequal -> 10
    (_BPF_RET_K, 0, 0, _SECCOMP_RET_ERRNO | 1),  # 09: EPERM
    (_BPF_JMP_JEQ_K, 0, 1, _SYS_EXECVE),  # 10: unequal -> 12
    (_BPF_RET_K, 0, 0, _SECCOMP_RET_ERRNO | 1),  # 11: EPERM
    (_BPF_JMP_JEQ_K, 0, 1, _SYS_EXECVEAT),  # 12: unequal -> 14
    (_BPF_RET_K, 0, 0, _SECCOMP_RET_ERRNO | 1),  # 13: EPERM
    (_BPF_JMP_JEQ_K, 0, 1, _SYS_CLONE3),  # 14: unequal -> 16
    (_BPF_RET_K, 0, 0, _SECCOMP_RET_ERRNO | 38),  # 15: ENOSYS
    (_BPF_JMP_JEQ_K, 0, 3, _SYS_CLONE),  # 16: unequal -> allow at 20
    (
        _BPF_LD_W_ABS,
        0,
        0,
        _SECCOMP_DATA_ARGUMENT_ZERO_OFFSET,
    ),  # 17: clone flags low word
    (_BPF_JMP_JSET_K, 1, 0, _CLONE_THREAD),  # 18: thread -> allow at 20
    (_BPF_RET_K, 0, 0, _SECCOMP_RET_ERRNO | 1),  # 19: EPERM
    (_BPF_RET_K, 0, 0, _SECCOMP_RET_ALLOW),  # 20
)

_FILTER_BYTES: Final = b"".join(
    struct.pack("<HBBI", code, jump_true, jump_false, value)
    for code, jump_true, jump_false, value in _FILTER_INSTRUCTIONS
)
_FILTER_FRAME: Final = (
    _FILTER_FRAME_MAGIC + struct.pack("<I", len(_FILTER_INSTRUCTIONS)) + _FILTER_BYTES
)
_FILTER_SHA256: Final = hashlib.sha256(
    _FILTER_DIGEST_DOMAIN + _FILTER_FRAME
).hexdigest()


class Experiment002ProcessGuardError(RuntimeError):
    """The registered child process boundary failed closed."""


@dataclass(frozen=True, slots=True)
class _GuardSnapshot:
    process_id: int
    parent_process_id: int
    native_thread_id: int
    filter_sha256: str
    filter_instruction_count: int
    seccomp_filter_count: int


class _GuardStateRoute(Protocol):
    def __call__(self, guard: object, /) -> _GuardSnapshot:
        """Return closure-owned verified guard state."""


def _make_state_relay() -> tuple[
    _GuardStateRoute,
    Callable[[_GuardStateRoute], None],
]:
    route: _GuardStateRoute | None = None

    def relay(guard: object, /) -> _GuardSnapshot:
        if route is None:
            raise Experiment002ProcessGuardError(
                "process-guard authority route is not initialized"
            )
        return route(guard)

    def bind(candidate: _GuardStateRoute, /) -> None:
        nonlocal route
        if route is not None:
            raise Experiment002ProcessGuardError(
                "process-guard authority route was already initialized"
            )
        route = candidate

    return relay, bind


_GUARD_STATE_RELAY, _bind_guard_state_relay = _make_state_relay()


def _make_guard_properties(
    state: _GuardStateRoute,
) -> tuple[property, property, property, property, property, property]:
    def process_id(guard: object, /) -> int:
        return state(guard).process_id

    def parent_process_id(guard: object, /) -> int:
        return state(guard).parent_process_id

    def native_thread_id(guard: object, /) -> int:
        return state(guard).native_thread_id

    def filter_sha256(guard: object, /) -> str:
        return state(guard).filter_sha256

    def filter_instruction_count(guard: object, /) -> int:
        return state(guard).filter_instruction_count

    def seccomp_filter_count(guard: object, /) -> int:
        return state(guard).seccomp_filter_count

    return (
        property(process_id),
        property(parent_process_id),
        property(native_thread_id),
        property(filter_sha256),
        property(filter_instruction_count),
        property(seccomp_filter_count),
    )


(
    _PROCESS_ID_PROPERTY,
    _PARENT_PROCESS_ID_PROPERTY,
    _NATIVE_THREAD_ID_PROPERTY,
    _FILTER_SHA256_PROPERTY,
    _FILTER_INSTRUCTION_COUNT_PROPERTY,
    _SECCOMP_FILTER_COUNT_PROPERTY,
) = _make_guard_properties(_GUARD_STATE_RELAY)


@dataclass(frozen=True, slots=True, init=False, eq=False)
class VerifiedChildProcessGuard:
    """Opaque proof that the fixed child seccomp boundary was installed."""

    def __init__(self) -> None:
        raise TypeError("child process guards are issued only by the installer")

    process_id = _PROCESS_ID_PROPERTY
    parent_process_id = _PARENT_PROCESS_ID_PROPERTY
    native_thread_id = _NATIVE_THREAD_ID_PROPERTY
    filter_sha256 = _FILTER_SHA256_PROPERTY
    filter_instruction_count = _FILTER_INSTRUCTION_COUNT_PROPERTY
    seccomp_filter_count = _SECCOMP_FILTER_COUNT_PROPERTY

    def __copy__(self) -> NoReturn:
        raise TypeError("child process guards cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        del memo
        raise TypeError("child process guards cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("child process guards cannot be serialized")

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        del protocol
        raise TypeError("child process guards cannot be serialized")


@dataclass(frozen=True, slots=True)
class _IssuedGuardState:
    marker: object
    process_id: int
    parent_process_id: int
    native_thread_id: int
    filter_sha256: str
    filter_instruction_count: int
    seccomp_filter_count: int
    token: object


@dataclass(frozen=True, slots=True)
class _IssuedGuardSeal:
    marker: object
    process_id: int
    parent_process_id: int
    native_thread_id: int
    filter_sha256: str
    filter_instruction_count: int
    seccomp_filter_count: int
    token: object


class _SockFilter(ctypes.Structure):
    _fields_ = (
        ("code", ctypes.c_ushort),
        ("jt", ctypes.c_ubyte),
        ("jf", ctypes.c_ubyte),
        ("k", ctypes.c_uint32),
    )


class _SockFprog(ctypes.Structure):
    _fields_ = (
        ("length", ctypes.c_ushort),
        ("filter", ctypes.POINTER(_SockFilter)),
    )


def _registered_filter_contract_for_tests(
    _instructions: tuple[tuple[int, int, int, int], ...] = _FILTER_INSTRUCTIONS,
    _frame: bytes = _FILTER_FRAME,
    _digest: str = _FILTER_SHA256,
) -> tuple[tuple[tuple[int, int, int, int], ...], bytes, str]:
    """Return the immutable static filter contract without installing it."""

    return _instructions, _frame, _digest


def _make_import_bootstrap() -> Callable[
    [],
    tuple[
        Callable[[], VerifiedChildProcessGuard],
        Callable[[VerifiedChildProcessGuard], None],
        _GuardStateRoute,
    ],
]:
    """Capture trusted routes and return the sole import-time bootstrap."""

    module_globals = globals()
    guard_type = VerifiedChildProcessGuard
    error_type = Experiment002ProcessGuardError
    state_type = _IssuedGuardState
    seal_type = _IssuedGuardSeal
    snapshot_type = _GuardSnapshot
    sock_filter_type = _SockFilter
    sock_fprog_type = _SockFprog
    instructions = _FILTER_INSTRUCTIONS
    filter_frame = _FILTER_FRAME
    filter_sha256 = _FILTER_SHA256
    registered_filter_sha256 = _REGISTERED_FILTER_SHA256
    filter_frame_magic = _FILTER_FRAME_MAGIC
    filter_digest_domain = _FILTER_DIGEST_DOMAIN
    numerical_roots = _NUMERICAL_IMPORT_ROOTS
    max_status_bytes = _MAX_PROC_STATUS_BYTES
    seccomp_mode_filter = _SECCOMP_MODE_FILTER
    seccomp_syscall_number = _SYS_SECCOMP
    seccomp_set_mode_filter = _SECCOMP_SET_MODE_FILTER
    seccomp_tsync = _SECCOMP_FILTER_FLAG_TSYNC
    pr_get_seccomp = _PR_GET_SECCOMP
    pr_set_no_new_privs = _PR_SET_NO_NEW_PRIVS
    pr_get_no_new_privs = _PR_GET_NO_NEW_PRIVS
    original_modules = sys.modules
    original_modules_identity = id(original_modules)

    builtin_id = builtins.id
    builtin_int = builtins.int
    builtin_isinstance = builtins.isinstance
    builtin_len = builtins.len
    list_type = builtins.list
    builtin_object = builtins.object
    builtin_open = builtins.open
    builtin_set = builtins.set
    builtin_sorted = builtins.sorted
    builtin_str = builtins.str
    builtin_tuple = builtins.tuple
    builtin_type = builtins.type
    bytes_type = builtins.bytes
    base_exception_type = builtins.BaseException
    exception_type = builtins.Exception
    os_error_type = builtins.OSError

    getpid = os.getpid
    getppid = os.getppid
    gettid = threading.get_native_id
    listdir = os.listdir
    strerror = os.strerror
    uname = os.uname
    lifecycle_lock = threading.Lock()
    pack = struct.pack
    calcsize = struct.calcsize
    byteorder = sys.byteorder
    platform_name = sys.platform
    os_name = os.name
    implementation_name = sys.implementation.name

    ctypes_sizeof = ctypes.sizeof
    ctypes_alignment = ctypes.alignment
    ctypes_byref = ctypes.byref
    ctypes_cast = ctypes.cast
    ctypes_get_errno = ctypes.get_errno
    ctypes_set_errno = ctypes.set_errno
    ctypes_c_int = ctypes.c_int
    ctypes_c_long = ctypes.c_long
    ctypes_c_uint32 = ctypes.c_uint32
    ctypes_c_ulong = ctypes.c_ulong
    ctypes_c_void_p = ctypes.c_void_p
    hash_sha256 = hashlib.sha256
    errno_eio = errno.EIO
    errno_eperm = errno.EPERM
    errno_enosys = errno.ENOSYS

    libc = ctypes.CDLL(None, use_errno=True)
    prctl_address = ctypes_cast(libc.prctl, ctypes_c_void_p).value
    syscall_address = ctypes_cast(libc.syscall, ctypes_c_void_p).value
    if prctl_address is None or syscall_address is None:
        raise error_type("libc does not expose usable prctl and syscall symbols")
    prctl = ctypes.CFUNCTYPE(
        ctypes_c_int,
        ctypes_c_int,
        ctypes_c_ulong,
        ctypes_c_void_p,
        ctypes_c_ulong,
        ctypes_c_ulong,
        use_errno=True,
    )(prctl_address)
    raw_seccomp = ctypes.CFUNCTYPE(
        ctypes_c_long,
        ctypes_c_long,
        ctypes_c_ulong,
        ctypes_c_ulong,
        ctypes_c_void_p,
        use_errno=True,
    )(syscall_address)

    phase_new = 0
    phase_installing = 1
    phase_installed = 2
    phase_failed = 3
    phase = phase_new
    failure_latched = False
    validation_active = False
    capability_claimed = False
    issued_capability: VerifiedChildProcessGuard | None = None
    issued_state: _IssuedGuardState | None = None
    issued_seal: _IssuedGuardSeal | None = None
    trusted_snapshot: _GuardSnapshot | None = None
    trusted_state_frame: bytes | None = None
    trusted_seal_frame: bytes | None = None
    expected_getter: object | None = None
    expected_verify: object | None = None
    marker = builtin_object()
    token = builtin_object()

    property_names = (
        "process_id",
        "parent_process_id",
        "native_thread_id",
        "filter_sha256",
        "filter_instruction_count",
        "seccomp_filter_count",
    )
    guarded_class_members = {
        name: guard_type.__dict__[name]
        for name in (
            "__init__",
            "__copy__",
            "__deepcopy__",
            "__reduce__",
            "__reduce_ex__",
            *property_names,
        )
    }

    def fail(message: str) -> NoReturn:
        raise error_type(message)

    def poison() -> None:
        nonlocal failure_latched
        nonlocal phase
        failure_latched = True
        if phase != phase_installing:
            phase = phase_failed

    def assert_installing(stage: str) -> None:
        if failure_latched or phase != phase_installing:
            poison()
            fail(f"process-guard bootstrap lost authority before {stage}")

    def require_public_routes() -> None:
        if (
            module_globals.get("Experiment002ProcessGuardError") is not error_type
            or module_globals.get("VerifiedChildProcessGuard") is not guard_type
            or module_globals.get("get_registered_child_process_guard")
            is not expected_getter
            or module_globals.get("verify_verified_child_process_guard")
            is not expected_verify
        ):
            fail("process-guard public authority route was replaced")
        for name, member in guarded_class_members.items():
            if guard_type.__dict__.get(name) is not member:
                fail("process-guard capability class was replaced")
        if builtin_id(original_modules) != original_modules_identity:
            fail("the captured import registry identity changed")

    def require_runtime() -> None:
        observed_uname = uname()
        if (
            os_name != "posix"
            or platform_name != "linux"
            or implementation_name != "cpython"
            or byteorder != "little"
            or observed_uname.sysname != "Linux"
            or observed_uname.machine != "x86_64"
            or calcsize("P") != 8
            or ctypes_sizeof(ctypes_c_long) != 8
            or ctypes_sizeof(ctypes_c_uint32) != 4
            or ctypes_sizeof(sock_filter_type) != 8
            or ctypes_alignment(sock_filter_type) != 4
            or sock_filter_type.code.offset != 0
            or sock_filter_type.jt.offset != 2
            or sock_filter_type.jf.offset != 3
            or sock_filter_type.k.offset != 4
            or sock_fprog_type.length.offset != 0
            or sock_fprog_type.filter.offset != 8
            or ctypes_sizeof(sock_fprog_type) != 16
            or errno_eio != 5
            or errno_eperm != 1
            or errno_enosys != 38
            or seccomp_syscall_number != 317
            or seccomp_set_mode_filter != 1
            or seccomp_tsync != 1
            or filter_sha256 != registered_filter_sha256
            or hash_sha256(filter_digest_domain + filter_frame).hexdigest()
            != registered_filter_sha256
        ):
            fail("process guard requires the exact Linux x86-64 CPython ABI")

    def numerical_imports() -> tuple[str, ...]:
        observed = {
            name.partition(".")[0]
            for name in builtin_tuple(original_modules)
            if builtin_type(name) is builtin_str
        }
        return builtin_tuple(builtin_sorted(observed & numerical_roots))

    def task_ids() -> tuple[int, ...]:
        try:
            entries = listdir("/proc/self/task")
        except os_error_type as exc:
            raise error_type("cannot inspect /proc/self/task") from exc
        if builtin_type(entries) is not list_type:
            fail("invalid /proc/self/task listing")
        result: list[int] = []
        for entry in entries:
            if (
                builtin_type(entry) is not builtin_str
                or not entry.isascii()
                or not entry.isdigit()
            ):
                fail("invalid native thread identifier in /proc/self/task")
            value = builtin_int(entry, 10)
            if value <= 0:
                fail("invalid native thread identifier in /proc/self/task")
            result.append(value)
        if not result or builtin_len(builtin_set(result)) != builtin_len(result):
            fail("ambiguous /proc/self/task listing")
        return builtin_tuple(builtin_sorted(result))

    def record_invariants(
        stage: str,
        process_id: int,
        native_thread_id: int,
        violations: set[str],
    ) -> None:
        before_numerics = numerical_imports()
        observed_tasks = task_ids()
        after_numerics = numerical_imports()
        observed_numerics = builtin_tuple(
            builtin_sorted(builtin_set(before_numerics) | builtin_set(after_numerics))
        )
        if observed_numerics:
            violations.add(
                f"{stage}: numerical imports: {', '.join(observed_numerics)}"
            )
        if native_thread_id != process_id or observed_tasks != (process_id,):
            violations.add(
                f"{stage}: expected only main TID {process_id}, observed "
                + ",".join(builtin_str(value) for value in observed_tasks)
            )

    def identity() -> tuple[int, int, int]:
        process_id = getpid()
        parent_process_id = getppid()
        native_thread_id = gettid()
        if (
            builtin_type(process_id) is not builtin_int
            or builtin_type(parent_process_id) is not builtin_int
            or builtin_type(native_thread_id) is not builtin_int
            or process_id <= 1
            or parent_process_id <= 0
            or parent_process_id == process_id
            or native_thread_id <= 0
        ):
            fail("process guard observed an invalid child process identity")
        return process_id, parent_process_id, native_thread_id

    def read_status() -> tuple[int, int, int]:
        try:
            with builtin_open("/proc/self/status", "rb", buffering=0) as stream:
                payload = stream.read(max_status_bytes + 1)
        except os_error_type as exc:
            raise error_type("cannot inspect /proc/self/status") from exc
        if (
            builtin_type(payload) is not bytes_type
            or builtin_len(payload) > max_status_bytes
        ):
            fail("invalid /proc/self/status payload")
        fields: dict[bytes, int] = {}
        required = (b"NoNewPrivs", b"Seccomp", b"Seccomp_filters")
        for line in payload.splitlines():
            key, separator, value = line.partition(b":")
            if key not in required:
                continue
            if not separator or key in fields:
                fail("ambiguous seccomp fields in /proc/self/status")
            stripped = value.strip()
            if not stripped or not stripped.isascii() or not stripped.isdigit():
                fail("invalid seccomp field in /proc/self/status")
            fields[key] = builtin_int(stripped, 10)
        if builtin_set(fields) != builtin_set(required):
            fail("missing seccomp fields in /proc/self/status")
        return (
            fields[b"NoNewPrivs"],
            fields[b"Seccomp"],
            fields[b"Seccomp_filters"],
        )

    def call_prctl(option: int, argument_two: int) -> int:
        ctypes_set_errno(0)
        result = builtin_int(prctl(option, argument_two, None, 0, 0))
        if result < 0:
            observed_errno = ctypes_get_errno()
            if observed_errno <= 0:
                observed_errno = errno_eio
            raise os_error_type(observed_errno, strerror(observed_errno))
        return result

    def install_tsync(program: _SockFprog) -> None:
        ctypes_set_errno(0)
        result = builtin_int(
            raw_seccomp(
                seccomp_syscall_number,
                seccomp_set_mode_filter,
                seccomp_tsync,
                ctypes_byref(program),
            )
        )
        if result != 0:
            observed_errno = ctypes_get_errno()
            if result > 0:
                fail(f"seccomp TSYNC rejected native thread {result}")
            if observed_errno <= 0:
                observed_errno = errno_eio
            raise os_error_type(observed_errno, strerror(observed_errno))

    def state_frame(candidate: _IssuedGuardState | _IssuedGuardSeal) -> bytes:
        digest_bytes = candidate.filter_sha256.encode("ascii")
        return b"".join(
            (
                pack("<Q", builtin_id(candidate.marker)),
                pack("<Q", candidate.process_id),
                pack("<Q", candidate.parent_process_id),
                pack("<Q", candidate.native_thread_id),
                pack("<I", candidate.filter_instruction_count),
                pack("<I", candidate.seccomp_filter_count),
                pack("<I", builtin_len(digest_bytes)),
                digest_bytes,
                pack("<Q", builtin_id(candidate.token)),
            )
        )

    def kernel_matches(snapshot: _GuardSnapshot) -> None:
        process_id, parent_process_id, native_thread_id = identity()
        if (
            process_id != snapshot.process_id
            or parent_process_id != snapshot.parent_process_id
            or native_thread_id != snapshot.native_thread_id
        ):
            fail("process-guard PID, PPID, or installer TID changed")
        no_new_privs, seccomp_mode, filter_count = read_status()
        if (
            no_new_privs != 1
            or seccomp_mode != seccomp_mode_filter
            or filter_count != snapshot.seccomp_filter_count
            or call_prctl(pr_get_no_new_privs, 0) != 1
            or call_prctl(pr_get_seccomp, 0) != seccomp_mode_filter
        ):
            fail("installed process-guard kernel state changed")

    def validate_locked(candidate: object, /) -> _GuardSnapshot:
        require_public_routes()
        if failure_latched or phase != phase_installed:
            fail("process-guard authority is not active")
        if (
            builtin_type(candidate) is not guard_type
            or candidate is not issued_capability
            or issued_state is None
            or issued_seal is None
            or trusted_snapshot is None
            or trusted_state_frame is None
            or trusted_seal_frame is None
            or builtin_type(issued_state) is not state_type
            or builtin_type(issued_seal) is not seal_type
            or issued_state.marker is not marker
            or issued_seal.marker is not marker
            or issued_state.token is not token
            or issued_seal.token is not token
            or state_frame(issued_state) != trusted_state_frame
            or state_frame(issued_seal) != trusted_seal_frame
            or issued_state.process_id != trusted_snapshot.process_id
            or issued_seal.process_id != trusted_snapshot.process_id
            or issued_state.parent_process_id != trusted_snapshot.parent_process_id
            or issued_seal.parent_process_id != trusted_snapshot.parent_process_id
            or issued_state.native_thread_id != trusted_snapshot.native_thread_id
            or issued_seal.native_thread_id != trusted_snapshot.native_thread_id
            or issued_state.filter_sha256 != trusted_snapshot.filter_sha256
            or issued_seal.filter_sha256 != trusted_snapshot.filter_sha256
            or issued_state.filter_instruction_count
            != trusted_snapshot.filter_instruction_count
            or issued_seal.filter_instruction_count
            != trusted_snapshot.filter_instruction_count
            or issued_state.seccomp_filter_count
            != trusted_snapshot.seccomp_filter_count
            or issued_seal.seccomp_filter_count != trusted_snapshot.seccomp_filter_count
        ):
            fail("process-guard authority is forged or changed")
        kernel_matches(trusted_snapshot)
        if failure_latched or phase != phase_installed:
            fail("process-guard authority failed during validation")
        return snapshot_type(
            process_id=trusted_snapshot.process_id,
            parent_process_id=trusted_snapshot.parent_process_id,
            native_thread_id=trusted_snapshot.native_thread_id,
            filter_sha256=trusted_snapshot.filter_sha256,
            filter_instruction_count=trusted_snapshot.filter_instruction_count,
            seccomp_filter_count=trusted_snapshot.seccomp_filter_count,
        )

    def verified_snapshot(candidate: object, /) -> _GuardSnapshot:
        nonlocal phase
        nonlocal validation_active
        if validation_active:
            poison()
            fail("recursive process-guard validation is forbidden")
        validation_active = True
        try:
            with lifecycle_lock:
                try:
                    return validate_locked(candidate)
                except base_exception_type:
                    poison()
                    raise
        finally:
            validation_active = False

    def get_capability() -> VerifiedChildProcessGuard:
        nonlocal capability_claimed
        nonlocal phase
        nonlocal validation_active
        if validation_active:
            poison()
            fail("recursive process-guard capability claim is forbidden")
        validation_active = True
        try:
            with lifecycle_lock:
                try:
                    require_public_routes()
                    if (
                        failure_latched
                        or phase != phase_installed
                        or issued_capability is None
                    ):
                        fail("process-guard authority is not active")
                    if capability_claimed:
                        poison()
                        fail("process-guard capability was already claimed")
                    validate_locked(issued_capability)
                    if failure_latched or phase != phase_installed:
                        fail("process-guard authority failed during capability claim")
                    capability_claimed = True
                    return issued_capability
                except base_exception_type:
                    poison()
                    raise
        finally:
            validation_active = False

    def verify(candidate: VerifiedChildProcessGuard, /) -> None:
        verified_snapshot(candidate)

    expected_getter = get_capability
    expected_verify = verify

    def bootstrap() -> tuple[
        Callable[[], VerifiedChildProcessGuard],
        Callable[[VerifiedChildProcessGuard], None],
        _GuardStateRoute,
    ]:
        nonlocal phase
        nonlocal issued_capability
        nonlocal issued_state
        nonlocal issued_seal
        nonlocal trusted_snapshot
        nonlocal trusted_state_frame
        nonlocal trusted_seal_frame

        if failure_latched or phase != phase_new:
            poison()
            fail("process-guard import bootstrap is one-shot and non-reentrant")
        with lifecycle_lock:
            if failure_latched or phase != phase_new:
                poison()
                fail("process-guard import bootstrap is one-shot and non-reentrant")
            phase = phase_installing
            violations: set[str] = builtin_set()
            try:
                require_runtime()
                process_id, parent_process_id, native_thread_id = identity()
                record_invariants(
                    "initial import audit",
                    process_id,
                    native_thread_id,
                    violations,
                )

                before_no_new_privs, before_mode, before_filters = read_status()
                record_invariants(
                    "after pre-install status audit",
                    process_id,
                    native_thread_id,
                    violations,
                )
                if (
                    before_no_new_privs not in (0, 1)
                    or before_mode not in (0, seccomp_mode_filter)
                    or before_filters < 0
                    or (before_mode == 0 and before_filters != 0)
                    or (before_mode == seccomp_mode_filter and before_filters < 1)
                    or call_prctl(pr_get_no_new_privs, 0) != before_no_new_privs
                    or call_prctl(pr_get_seccomp, 0) != before_mode
                ):
                    fail("unexpected pre-install seccomp kernel state")

                assert_installing("PR_SET_NO_NEW_PRIVS")
                call_prctl(pr_set_no_new_privs, 1)
                assert_installing("post-no-new-privileges audit")
                no_new_privs_status = read_status()
                record_invariants(
                    "after no-new-privileges status audit",
                    process_id,
                    native_thread_id,
                    violations,
                )
                if (
                    no_new_privs_status[0] != 1
                    or call_prctl(pr_get_no_new_privs, 0) != 1
                ):
                    fail("PR_SET_NO_NEW_PRIVS did not become effective")

                filter_array_type = sock_filter_type * builtin_len(instructions)
                filter_array = filter_array_type(
                    *(sock_filter_type(*instruction) for instruction in instructions)
                )
                observed_filter_bytes = bytes_type(filter_array)
                observed_frame = (
                    filter_frame_magic
                    + pack("<I", builtin_len(instructions))
                    + observed_filter_bytes
                )
                if (
                    builtin_type(observed_filter_bytes) is not bytes_type
                    or observed_frame != filter_frame
                    or hash_sha256(filter_digest_domain + observed_frame).hexdigest()
                    != filter_sha256
                ):
                    fail("in-memory seccomp filter differs from its registration")
                program = sock_fprog_type(builtin_len(instructions), filter_array)

                record_invariants(
                    "immediately before TSYNC",
                    process_id,
                    native_thread_id,
                    violations,
                )
                assert_installing("seccomp TSYNC")
                install_tsync(program)
                assert_installing("post-seccomp TSYNC audit")
                record_invariants(
                    "immediately after TSYNC",
                    process_id,
                    native_thread_id,
                    violations,
                )

                after_no_new_privs, after_mode, after_filters = read_status()
                record_invariants(
                    "after installed-status audit",
                    process_id,
                    native_thread_id,
                    violations,
                )
                if (
                    after_no_new_privs != 1
                    or after_mode != seccomp_mode_filter
                    or after_filters != before_filters + 1
                    or call_prctl(pr_get_no_new_privs, 0) != 1
                    or call_prctl(pr_get_seccomp, 0) != seccomp_mode_filter
                ):
                    fail("seccomp TSYNC did not bind exact kernel state")
                if identity() != (process_id, parent_process_id, native_thread_id):
                    fail("process identity changed during guard import")

                snapshot = snapshot_type(
                    process_id=process_id,
                    parent_process_id=parent_process_id,
                    native_thread_id=native_thread_id,
                    filter_sha256=filter_sha256,
                    filter_instruction_count=builtin_len(instructions),
                    seccomp_filter_count=after_filters,
                )
                state = state_type(
                    marker=marker,
                    process_id=process_id,
                    parent_process_id=parent_process_id,
                    native_thread_id=native_thread_id,
                    filter_sha256=filter_sha256,
                    filter_instruction_count=builtin_len(instructions),
                    seccomp_filter_count=after_filters,
                    token=token,
                )
                seal = seal_type(
                    marker=marker,
                    process_id=process_id,
                    parent_process_id=parent_process_id,
                    native_thread_id=native_thread_id,
                    filter_sha256=filter_sha256,
                    filter_instruction_count=builtin_len(instructions),
                    seccomp_filter_count=after_filters,
                    token=token,
                )
                kernel_matches(snapshot)
                record_invariants(
                    "final pre-issuance audit",
                    process_id,
                    native_thread_id,
                    violations,
                )
                if violations:
                    fail(
                        "process-guard import invariants changed; filter sealed but "
                        "capability withheld: " + "; ".join(builtin_sorted(violations))
                    )

                assert_installing("final capability issuance")
                capability = builtin_object.__new__(guard_type)
                state_frame_value = state_frame(state)
                seal_frame_value = state_frame(seal)
                assert_installing("authority publication")
                issued_capability = capability
                issued_state = state
                issued_seal = seal
                trusted_snapshot = snapshot
                trusted_state_frame = state_frame_value
                trusted_seal_frame = seal_frame_value
                phase = phase_installed
                if failure_latched:
                    phase = phase_failed
                    fail(
                        "process-guard bootstrap failure was latched before publication"
                    )
                return get_capability, verify, verified_snapshot
            except exception_type as exc:
                poison()
                if builtin_isinstance(exc, error_type):
                    raise
                raise error_type("process-guard import failed closed") from exc
            except base_exception_type:
                poison()
                raise

    return bootstrap


_import_bootstrap = _make_import_bootstrap()
del _make_import_bootstrap

(
    get_registered_child_process_guard,
    verify_verified_child_process_guard,
    _guard_state_route,
) = _import_bootstrap()
_bind_guard_state_relay(_guard_state_route)

del _import_bootstrap
del _bind_guard_state_relay
del _guard_state_route
del _GUARD_STATE_RELAY
del _PROCESS_ID_PROPERTY
del _PARENT_PROCESS_ID_PROPERTY
del _NATIVE_THREAD_ID_PROPERTY
del _FILTER_SHA256_PROPERTY
del _FILTER_INSTRUCTION_COUNT_PROPERTY
del _SECCOMP_FILTER_COUNT_PROPERTY
del _make_guard_properties
del _make_state_relay
