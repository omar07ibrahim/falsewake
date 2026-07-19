from __future__ import annotations

import ast
import hashlib
import json
import os
import struct
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
SOURCE_PATH = SOURCE_ROOT / "falsewake/experiment_002_process_guard.py"
FILTER_DOMAIN = b"falsewake-exp002-process-guard-filter-v1\0"
FILTER_MAGIC = b"FW2SCMP1"
FILTER_SHA256 = "e2004b41f7a30715c9ff00456b0e9fd0f1f974500d49aeaa728006d01bacc9ab"

LD_W_ABS = 0x20
JMP_JEQ_K = 0x15
JMP_JSET_K = 0x45
RET_K = 0x06
RET_KILL_PROCESS = 0x8000_0000
RET_ERRNO = 0x0005_0000
RET_ALLOW = 0x7FFF_0000
ARCH_X86_64 = 0xC000_003E
X32_BIT = 0x4000_0000
CLONE_THREAD = 0x0001_0000

EXPECTED_INSTRUCTIONS = (
    (LD_W_ABS, 0, 0, 4),
    (JMP_JEQ_K, 1, 0, ARCH_X86_64),
    (RET_K, 0, 0, RET_KILL_PROCESS),
    (LD_W_ABS, 0, 0, 0),
    (JMP_JSET_K, 0, 1, X32_BIT),
    (RET_K, 0, 0, RET_ERRNO | 38),
    (JMP_JEQ_K, 0, 1, 57),
    (RET_K, 0, 0, RET_ERRNO | 1),
    (JMP_JEQ_K, 0, 1, 58),
    (RET_K, 0, 0, RET_ERRNO | 1),
    (JMP_JEQ_K, 0, 1, 59),
    (RET_K, 0, 0, RET_ERRNO | 1),
    (JMP_JEQ_K, 0, 1, 322),
    (RET_K, 0, 0, RET_ERRNO | 1),
    (JMP_JEQ_K, 0, 1, 435),
    (RET_K, 0, 0, RET_ERRNO | 38),
    (JMP_JEQ_K, 0, 3, 56),
    (LD_W_ABS, 0, 0, 16),
    (JMP_JSET_K, 1, 0, CLONE_THREAD),
    (RET_K, 0, 0, RET_ERRNO | 1),
    (RET_K, 0, 0, RET_ALLOW),
)


def _read_seccomp_status() -> dict[str, int]:
    result: dict[str, int] = {}
    for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
        key, separator, value = line.partition(":")
        if separator and key in {"NoNewPrivs", "Seccomp", "Seccomp_filters"}:
            result[key] = int(value.strip())
    assert set(result) == {"NoNewPrivs", "Seccomp", "Seccomp_filters"}
    return result


ROOT_STATUS = _read_seccomp_status()


def _run_fresh(
    script: str, *, timeout: float = 20.0
) -> subprocess.CompletedProcess[str]:
    source = (
        "import sys\n"
        f"sys.path.insert(0, {os.fspath(SOURCE_ROOT)!r})\n" + textwrap.dedent(script)
    )
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        (sys.executable, "-I", "-S", "-B", "-c", source),
        env=environment,
        capture_output=True,
        check=False,
        text=True,
        timeout=timeout,
    )


def _fresh_json(script: str, *, timeout: float = 20.0) -> Any:
    completed = _run_fresh(script, timeout=timeout)
    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.splitlines()
    assert lines, completed.stderr
    return json.loads(lines[-1])


def _evaluate_filter(*, architecture: int, syscall: int, argument_zero: int = 0) -> int:
    accumulator = 0
    program_counter = 0
    for _ in range(64):
        code, jump_true, jump_false, value = EXPECTED_INSTRUCTIONS[program_counter]
        if code == LD_W_ABS:
            if value == 0:
                accumulator = syscall & 0xFFFF_FFFF
            elif value == 4:
                accumulator = architecture & 0xFFFF_FFFF
            elif value == 16:
                accumulator = argument_zero & 0xFFFF_FFFF
            else:
                raise AssertionError(f"unexpected absolute load offset {value}")
            program_counter += 1
        elif code == JMP_JEQ_K:
            jump = jump_true if accumulator == value else jump_false
            program_counter += jump + 1
        elif code == JMP_JSET_K:
            jump = jump_true if accumulator & value else jump_false
            program_counter += jump + 1
        elif code == RET_K:
            return value
        else:
            raise AssertionError(f"unexpected BPF opcode {code:#x}")
    raise AssertionError("filter failed to terminate")


def test_source_is_standard_library_only_and_root_never_imports_it() -> None:
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    imported = {
        node.names[0].name.partition(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
    }
    imported.update(
        (node.module or "").partition(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    assert imported <= {
        "__future__",
        "builtins",
        "collections",
        "ctypes",
        "dataclasses",
        "errno",
        "hashlib",
        "os",
        "struct",
        "sys",
        "threading",
        "typing",
    }
    assert "falsewake.experiment_002_process_guard" not in sys.modules
    assert _read_seccomp_status() == ROOT_STATUS


def test_source_exposes_getter_not_a_post_import_installer() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level_functions = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert "install_registered_child_process_guard" not in source
    assert "get_registered_child_process_guard" in source
    assert "_make_import_bootstrap" in top_level_functions
    assert "_SYS_SECCOMP: Final = 317" in source
    assert "_SECCOMP_FILTER_FLAG_TSYNC: Final = 1" in source
    assert "threading.Lock()" in source
    assert "threading.RLock()" not in source


def test_filter_frame_digest_and_kernel_program_are_exact() -> None:
    result = _fresh_json(
        """
        import json
        import falsewake.experiment_002_process_guard as guard_module

        contract = guard_module._registered_filter_contract_for_tests
        instructions, frame, digest = contract()
        print(json.dumps({
            "instructions": instructions,
            "frame_hex": frame.hex(),
            "digest": digest,
        }))
        """
    )
    expected_bytes = b"".join(
        struct.pack("<HBBI", *instruction) for instruction in EXPECTED_INSTRUCTIONS
    )
    expected_frame = FILTER_MAGIC + struct.pack("<I", 21) + expected_bytes
    assert (
        tuple(tuple(item) for item in result["instructions"]) == EXPECTED_INSTRUCTIONS
    )
    assert bytes.fromhex(result["frame_hex"]) == expected_frame
    assert len(expected_frame) == 180
    assert hashlib.sha256(FILTER_DOMAIN + expected_frame).hexdigest() == FILTER_SHA256
    assert result["digest"] == FILTER_SHA256


def test_filter_jump_offsets_are_forward_bounded_and_every_instruction_reachable() -> (
    None
):
    reachable = {0}
    frontier = [0]
    while frontier:
        program_counter = frontier.pop()
        code, jump_true, jump_false, _ = EXPECTED_INSTRUCTIONS[program_counter]
        if code == RET_K:
            continue
        if code == LD_W_ABS:
            successors: tuple[int, ...] = (program_counter + 1,)
        else:
            assert code in (JMP_JEQ_K, JMP_JSET_K)
            successors = (
                program_counter + jump_true + 1,
                program_counter + jump_false + 1,
            )
        for successor in successors:
            assert program_counter < successor < len(EXPECTED_INSTRUCTIONS)
            if successor not in reachable:
                reachable.add(successor)
                frontier.append(successor)
    assert reachable == set(range(len(EXPECTED_INSTRUCTIONS)))


@pytest.mark.parametrize("syscall", [57, 58, 59, 322])
def test_filter_interpreter_denies_process_and_exec_syscalls(syscall: int) -> None:
    assert _evaluate_filter(architecture=ARCH_X86_64, syscall=syscall) == (
        RET_ERRNO | 1
    )


def test_filter_interpreter_denies_x32_and_clone3_with_enosys() -> None:
    assert _evaluate_filter(architecture=ARCH_X86_64, syscall=435) == (RET_ERRNO | 38)
    for syscall in (X32_BIT, X32_BIT | 39, X32_BIT | 435):
        assert _evaluate_filter(architecture=ARCH_X86_64, syscall=syscall) == (
            RET_ERRNO | 38
        )


def test_filter_interpreter_allows_only_thread_clone_and_ordinary_syscalls() -> None:
    assert _evaluate_filter(architecture=ARCH_X86_64, syscall=56) == (RET_ERRNO | 1)
    assert (
        _evaluate_filter(
            architecture=ARCH_X86_64,
            syscall=56,
            argument_zero=CLONE_THREAD,
        )
        == RET_ALLOW
    )
    for syscall in (0, 1, 39, 202, 231, 334):
        assert _evaluate_filter(architecture=ARCH_X86_64, syscall=syscall) == RET_ALLOW


def test_filter_interpreter_kills_every_non_x86_64_architecture() -> None:
    for architecture in (0, 0x4000_003E, 0xC000_00B7, 0xFFFF_FFFF):
        assert _evaluate_filter(architecture=architecture, syscall=39) == (
            RET_KILL_PROCESS
        )


def test_import_synchronously_installs_guard_before_getter_is_obtainable() -> None:
    result = _fresh_json(
        """
        import importlib
        import json
        import os
        import threading

        def status():
            return {
                key: int(value)
                for key, value in (
                    line.split(":", 1)
                    for line in open("/proc/self/status", encoding="ascii")
                    if line.startswith(("NoNewPrivs:", "Seccomp:", "Seccomp_filters:"))
                )
            }

        before = status()
        guard_module = importlib.import_module("falsewake.experiment_002_process_guard")
        after_import = status()
        guard = guard_module.get_registered_child_process_guard()
        guard_module.verify_verified_child_process_guard(guard)
        print(json.dumps({
            "before": before,
            "after": after_import,
            "all": guard_module.__all__,
            "has_installer": hasattr(
                guard_module,
                "install_registered_child_process_guard",
            ),
            "pid": guard.process_id,
            "ppid": guard.parent_process_id,
            "tid": guard.native_thread_id,
            "digest": guard.filter_sha256,
            "instructions": guard.filter_instruction_count,
            "filters": guard.seccomp_filter_count,
            "actual": [os.getpid(), os.getppid(), threading.get_native_id()],
        }, sort_keys=True))
        """
    )
    assert result["after"]["NoNewPrivs"] == 1
    assert result["after"]["Seccomp"] == 2
    assert result["after"]["Seccomp_filters"] == (
        result["before"]["Seccomp_filters"] + 1
    )
    assert result["all"] == [
        "Experiment002ProcessGuardError",
        "VerifiedChildProcessGuard",
        "get_registered_child_process_guard",
        "verify_verified_child_process_guard",
    ]
    assert result["has_installer"] is False
    assert [result["pid"], result["ppid"], result["tid"]] == result["actual"]
    assert result["pid"] == result["tid"]
    assert result["digest"] == FILTER_SHA256
    assert result["instructions"] == 21
    assert result["filters"] == result["after"]["Seccomp_filters"]


def test_threads_created_after_import_inherit_filter_and_still_run() -> None:
    result = _fresh_json(
        """
        import json
        import os
        import threading
        import falsewake.experiment_002_process_guard as guard_module

        guard = guard_module.get_registered_child_process_guard()
        observed = []
        def worker():
            try:
                os.fork()
            except OSError as exc:
                observed.append(["ran", exc.errno])
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        guard_module.verify_verified_child_process_guard(guard)
        print(json.dumps(observed))
        """
    )
    assert result == [["ran", 1]]


def test_runtime_denies_fork_popen_execve_and_execveat_with_eperm() -> None:
    result = _fresh_json(
        """
        import ctypes
        import json
        import os
        import subprocess
        import falsewake.experiment_002_process_guard

        observed = {}
        try:
            child = os.fork()
        except OSError as exc:
            observed["fork"] = exc.errno
        else:
            if child == 0:
                os._exit(91)
            os.waitpid(child, 0)
            observed["fork"] = "allowed"
        try:
            subprocess.Popen(("/bin/true",))
        except OSError as exc:
            observed["popen"] = exc.errno
        else:
            observed["popen"] = "allowed"
        libc = ctypes.CDLL(None, use_errno=True)
        ctypes.set_errno(0)
        observed["execveat_result"] = int(libc.syscall(322, -100, 0, 0, 0, 0))
        observed["execveat_errno"] = ctypes.get_errno()
        try:
            os.execve("/bin/true", ("true",), {})
        except OSError as exc:
            observed["execve"] = exc.errno
        print(json.dumps(observed, sort_keys=True))
        """
    )
    assert result == {
        "execve": 1,
        "execveat_errno": 1,
        "execveat_result": -1,
        "fork": 1,
        "popen": 1,
    }


def test_runtime_clone3_x32_vfork_and_process_clone_actions_are_exact() -> None:
    result = _fresh_json(
        """
        import ctypes
        import json
        import os
        import falsewake.experiment_002_process_guard

        libc = ctypes.CDLL(None, use_errno=True)
        observed = {}
        for name, syscall, arguments in (
            ("clone3", 435, (0, 0)),
            ("x32_getpid", 0x40000027, ()),
            ("vfork", 58, ()),
            ("process_clone", 56, (17, 0, 0, 0, 0)),
        ):
            ctypes.set_errno(0)
            value = int(libc.syscall(syscall, *arguments))
            error = ctypes.get_errno()
            if value == 0:
                os._exit(97)
            if value > 0:
                os.waitpid(value, 0)
            observed[name] = [value, error]
        print(json.dumps(observed, sort_keys=True))
        """
    )
    assert result == {
        "clone3": [-1, 38],
        "process_clone": [-1, 1],
        "vfork": [-1, 1],
        "x32_getpid": [-1, 38],
    }


def test_capability_getter_is_one_shot_and_failure_is_sticky() -> None:
    result = _fresh_json(
        """
        import json
        import falsewake.experiment_002_process_guard as guard_module

        guard = guard_module.get_registered_child_process_guard()
        failures = []
        for operation in (
            guard_module.get_registered_child_process_guard,
            lambda: guard_module.verify_verified_child_process_guard(guard),
        ):
            try:
                operation()
            except guard_module.Experiment002ProcessGuardError as exc:
                failures.append(str(exc))
        print(json.dumps(failures))
        """
    )
    assert "already claimed" in result[0]
    assert "not active" in result[1]


def test_forged_capability_fails_and_stickily_poisons_real_capability() -> None:
    result = _fresh_json(
        """
        import json
        import falsewake.experiment_002_process_guard as guard_module

        constructor_failed = False
        try:
            guard_module.VerifiedChildProcessGuard()
        except TypeError:
            constructor_failed = True
        real = guard_module.get_registered_child_process_guard()
        forged = object.__new__(guard_module.VerifiedChildProcessGuard)
        failures = []
        for candidate in (forged, real):
            try:
                guard_module.verify_verified_child_process_guard(candidate)
            except guard_module.Experiment002ProcessGuardError as exc:
                failures.append(str(exc))
        print(json.dumps({"constructor": constructor_failed, "failures": failures}))
        """
    )
    assert result["constructor"] is True
    assert "forged" in result["failures"][0]
    assert "not active" in result["failures"][1]


def test_public_getter_route_replacement_fails_sticky() -> None:
    result = _fresh_json(
        """
        import json
        import falsewake.experiment_002_process_guard as guard_module

        getter = guard_module.get_registered_child_process_guard
        guard_module.get_registered_child_process_guard = lambda: None
        failures = []
        try:
            getter()
        except guard_module.Experiment002ProcessGuardError as exc:
            failures.append(str(exc))
        guard_module.get_registered_child_process_guard = getter
        try:
            getter()
        except guard_module.Experiment002ProcessGuardError as exc:
            failures.append(str(exc))
        print(json.dumps(failures))
        """
    )
    assert "route was replaced" in result[0]
    assert "not active" in result[1]


def test_class_member_replacement_is_detected_sticky() -> None:
    result = _fresh_json(
        """
        import json
        import falsewake.experiment_002_process_guard as guard_module

        guard = guard_module.get_registered_child_process_guard()
        original = guard_module.VerifiedChildProcessGuard.process_id
        guard_module.VerifiedChildProcessGuard.process_id = property(lambda self: 1)
        failures = []
        try:
            guard_module.verify_verified_child_process_guard(guard)
        except guard_module.Experiment002ProcessGuardError as exc:
            failures.append(str(exc))
        guard_module.VerifiedChildProcessGuard.process_id = original
        try:
            guard_module.verify_verified_child_process_guard(guard)
        except guard_module.Experiment002ProcessGuardError as exc:
            failures.append(str(exc))
        print(json.dumps(failures))
        """
    )
    assert "capability class was replaced" in result[0]
    assert "not active" in result[1]


def test_captured_routes_ignore_private_dependency_replacement_after_import() -> None:
    result = _fresh_json(
        """
        import json
        import falsewake.experiment_002_process_guard as guard_module

        getter = guard_module.get_registered_child_process_guard
        verify = guard_module.verify_verified_child_process_guard
        for name in (
            "builtins", "ctypes", "errno", "hashlib", "os", "struct", "sys", "threading"
        ):
            setattr(guard_module, name, None)
        guard = getter()
        verify(guard)
        print(json.dumps({
            "digest": guard.filter_sha256,
            "count": guard.filter_instruction_count,
        }))
        """
    )
    assert result == {"digest": FILTER_SHA256, "count": 21}


def test_published_authority_closures_have_no_late_module_global_routes() -> None:
    result = _fresh_json(
        """
        import inspect
        import json
        import types
        import falsewake.experiment_002_process_guard as guard_module

        pending = [
            guard_module.get_registered_child_process_guard,
            guard_module.verify_verified_child_process_guard,
        ]
        observed = set()
        late_globals = []
        while pending:
            function = pending.pop()
            if id(function) in observed:
                continue
            observed.add(id(function))
            closure = inspect.getclosurevars(function)
            if closure.globals:
                late_globals.append([function.__qualname__, sorted(closure.globals)])
            for cell in function.__closure__ or ():
                value = cell.cell_contents
                if (
                    isinstance(value, types.FunctionType)
                    and value.__module__ == guard_module.__name__
                ):
                    pending.append(value)
        property_parameter_counts = []
        for name in (
            "process_id",
            "parent_process_id",
            "native_thread_id",
            "filter_sha256",
            "filter_instruction_count",
            "seccomp_filter_count",
        ):
            descriptor = vars(guard_module.VerifiedChildProcessGuard)[name]
            property_parameter_counts.append(
                len(inspect.signature(descriptor.fget).parameters)
            )
        print(json.dumps({
            "closure_count": len(observed),
            "late_globals": late_globals,
            "property_parameters": property_parameter_counts,
        }))
        """
    )
    assert result["closure_count"] >= 10
    assert result["late_globals"] == []
    assert result["property_parameters"] == [1, 1, 1, 1, 1, 1]


def test_mutating_closure_issued_state_is_detected_by_independent_seal() -> None:
    result = _fresh_json(
        """
        import json
        import types
        import falsewake.experiment_002_process_guard as guard_module

        guard = guard_module.get_registered_child_process_guard()
        pending = [guard_module.verify_verified_child_process_guard]
        observed = set()
        issued = None
        while pending:
            function = pending.pop()
            if id(function) in observed:
                continue
            observed.add(id(function))
            for cell in function.__closure__ or ():
                value = cell.cell_contents
                if type(value).__name__ == "_IssuedGuardState":
                    issued = value
                elif isinstance(value, types.FunctionType):
                    pending.append(value)
        object.__setattr__(issued, "process_id", issued.process_id + 1)
        failures = []
        for _ in range(2):
            try:
                guard_module.verify_verified_child_process_guard(guard)
            except guard_module.Experiment002ProcessGuardError as exc:
                failures.append(str(exc))
        print(json.dumps(failures))
        """
    )
    assert "forged or changed" in result[0]
    assert "not active" in result[1]


def test_wrong_architecture_import_fails_before_filter_installation() -> None:
    result = _fresh_json(
        """
        import importlib
        import json
        import os
        import types

        def count():
            for line in open("/proc/self/status", encoding="ascii"):
                if line.startswith("Seccomp_filters:"):
                    return int(line.split(":", 1)[1])
            raise AssertionError

        real_uname = os.uname
        observed = real_uname()
        os.uname = lambda: types.SimpleNamespace(
            sysname=observed.sysname,
            nodename=observed.nodename,
            release=observed.release,
            version=observed.version,
            machine="aarch64",
        )
        before = count()
        try:
            importlib.import_module("falsewake.experiment_002_process_guard")
        except BaseException as exc:
            failure = str(exc)
        os.uname = real_uname
        print(json.dumps({"before": before, "after": count(), "failure": failure}))
        """
    )
    assert result["before"] == result["after"]
    assert "Linux x86-64 CPython ABI" in result["failure"]


def test_audit_hook_numerical_injection_is_detected_after_filter_is_sealed() -> None:
    result = _fresh_json(
        """
        import importlib
        import json
        import os
        import sys
        import types

        injected = False
        def hook(event, arguments):
            global injected
            if (
                event == "open"
                and arguments
                and arguments[0] == "/proc/self/status"
                and not injected
            ):
                injected = True
                sys.modules["numpy"] = types.ModuleType("numpy")
        sys.addaudithook(hook)
        try:
            importlib.import_module("falsewake.experiment_002_process_guard")
        except BaseException as exc:
            failure = str(exc)
        del sys.modules["numpy"]
        try:
            os.fork()
        except OSError as exc:
            fork_errno = exc.errno
        else:
            fork_errno = 0
        print(json.dumps({
            "injected": injected,
            "failure": failure,
            "fork_errno": fork_errno,
        }))
        """
    )
    assert result["injected"] is True
    assert "numerical imports: numpy" in result["failure"]
    assert "filter sealed" in result["failure"]
    assert result["fork_errno"] == 1


def test_audit_hook_thread_alive_through_tsync_is_sealed_then_detected() -> None:
    result = _fresh_json(
        """
        import importlib
        import json
        import os
        import sys
        import threading

        started = False
        ready = threading.Event()
        release = threading.Event()
        observed = []
        raced_thread = None

        def worker():
            ready.set()
            release.wait()
            try:
                child = os.fork()
            except OSError as exc:
                observed.append(exc.errno)
            else:
                if child == 0:
                    os._exit(97)
                os.waitpid(child, 0)
                observed.append(0)

        def hook(event, arguments):
            global started, raced_thread
            if (
                event == "open"
                and arguments
                and arguments[0] == "/proc/self/status"
                and not started
            ):
                started = True
                raced_thread = threading.Thread(target=worker)
                raced_thread.start()
                ready.wait()

        sys.addaudithook(hook)
        try:
            importlib.import_module("falsewake.experiment_002_process_guard")
        except BaseException as exc:
            failure = str(exc)
        release.set()
        raced_thread.join()
        try:
            os.fork()
        except OSError as exc:
            main_fork_errno = exc.errno
        else:
            main_fork_errno = 0
        print(json.dumps({
            "started": started,
            "failure": failure,
            "thread_fork": observed,
            "main_fork": main_fork_errno,
        }))
        """
    )
    assert result["started"] is True
    assert "expected only main TID" in result["failure"]
    assert "filter sealed" in result["failure"]
    assert result["thread_fork"] == [1]
    assert result["main_fork"] == 1


def test_recursive_private_bootstrap_poison_cannot_be_overwritten_by_outer() -> None:
    result = _fresh_json(
        """
        import importlib
        import json
        import sys

        def count():
            for line in open("/proc/self/status", encoding="ascii"):
                if line.startswith("Seccomp_filters:"):
                    return int(line.split(":", 1)[1])
            raise AssertionError

        recursed = False
        recursive_failure = None
        def hook(event, arguments):
            global recursed, recursive_failure
            if (
                event == "open"
                and arguments
                and arguments[0] == "/proc/self/status"
                and not recursed
            ):
                module = sys.modules.get("falsewake.experiment_002_process_guard")
                bootstrap = getattr(module, "_import_bootstrap", None)
                if bootstrap is not None:
                    recursed = True
                    try:
                        bootstrap()
                    except BaseException as exc:
                        recursive_failure = str(exc)

        before = count()
        sys.addaudithook(hook)
        try:
            importlib.import_module("falsewake.experiment_002_process_guard")
        except BaseException as exc:
            outer_failure = str(exc)
        print(json.dumps({
            "before": before,
            "after": count(),
            "recursed": recursed,
            "recursive": recursive_failure,
            "outer": outer_failure,
        }))
        """
    )
    assert result["recursed"] is True
    assert "non-reentrant" in result["recursive"]
    assert "lost authority" in result["outer"]
    assert result["after"] == result["before"]


def test_module_reload_is_rejected_without_adding_a_filter() -> None:
    result = _fresh_json(
        """
        import importlib
        import json
        import falsewake.experiment_002_process_guard as guard_module

        def count():
            for line in open("/proc/self/status", encoding="ascii"):
                if line.startswith("Seccomp_filters:"):
                    return int(line.split(":", 1)[1])
            raise AssertionError

        guard = guard_module.get_registered_child_process_guard()
        before = count()
        try:
            importlib.reload(guard_module)
        except RuntimeError as exc:
            failure = str(exc)
        guard_module.verify_verified_child_process_guard(guard)
        print(json.dumps({"before": before, "after": count(), "failure": failure}))
        """
    )
    assert result["after"] == result["before"]
    assert "cannot be reloaded" in result["failure"]


def test_verification_from_noninstaller_thread_stickily_fails() -> None:
    result = _fresh_json(
        """
        import json
        import threading
        import falsewake.experiment_002_process_guard as guard_module

        guard = guard_module.get_registered_child_process_guard()
        failures = []
        def worker():
            try:
                guard_module.verify_verified_child_process_guard(guard)
            except guard_module.Experiment002ProcessGuardError as exc:
                failures.append(str(exc))
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        try:
            guard_module.verify_verified_child_process_guard(guard)
        except guard_module.Experiment002ProcessGuardError as exc:
            failures.append(str(exc))
        print(json.dumps(failures))
        """
    )
    assert "installer TID changed" in result[0]
    assert "not active" in result[1]


def test_adding_unregistered_filter_invalidates_exact_filter_identity() -> None:
    result = _fresh_json(
        """
        import ctypes
        import json
        import falsewake.experiment_002_process_guard as guard_module

        guard = guard_module.get_registered_child_process_guard()
        class Filter(ctypes.Structure):
            _fields_ = (
                ("code", ctypes.c_ushort),
                ("jt", ctypes.c_ubyte),
                ("jf", ctypes.c_ubyte),
                ("k", ctypes.c_uint32),
            )
        class Program(ctypes.Structure):
            _fields_ = (
                ("length", ctypes.c_ushort),
                ("filter", ctypes.POINTER(Filter)),
            )
        filters = (Filter * 1)(Filter(6, 0, 0, 0x7fff0000))
        program = Program(1, filters)
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl.argtypes = (
            ctypes.c_int,
            ctypes.c_ulong,
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_ulong,
        )
        libc.prctl.restype = ctypes.c_int
        installed = libc.prctl(22, 2, ctypes.byref(program), 0, 0)
        failures = []
        for _ in range(2):
            try:
                guard_module.verify_verified_child_process_guard(guard)
            except guard_module.Experiment002ProcessGuardError as exc:
                failures.append(str(exc))
        print(json.dumps({"installed": installed, "failures": failures}))
        """
    )
    assert result["installed"] == 0
    assert "kernel state changed" in result["failures"][0]
    assert "not active" in result["failures"][1]


def test_ppid_change_invalidates_guard_identity() -> None:
    result = _fresh_json(
        """
        import json
        import os
        import time

        reader, writer = os.pipe()
        child = os.fork()
        if child != 0:
            os.close(writer)
            os.read(reader, 1)
            os._exit(0)
        os.close(reader)
        import falsewake.experiment_002_process_guard as guard_module
        guard = guard_module.get_registered_child_process_guard()
        installed_ppid = guard.parent_process_id
        os.write(writer, b"x")
        os.close(writer)
        deadline = time.monotonic() + 5.0
        while os.getppid() == installed_ppid and time.monotonic() < deadline:
            time.sleep(0.01)
        try:
            guard_module.verify_verified_child_process_guard(guard)
        except guard_module.Experiment002ProcessGuardError as exc:
            failure = str(exc)
        print(json.dumps({
            "before": installed_ppid,
            "after": os.getppid(),
            "failure": failure,
        }), flush=True)
        """,
        timeout=10.0,
    )
    assert result["after"] != result["before"]
    assert "PPID" in result["failure"]


def test_root_pytest_process_remains_unfiltered_by_production_module() -> None:
    assert "falsewake.experiment_002_process_guard" not in sys.modules
    assert _read_seccomp_status() == ROOT_STATUS
