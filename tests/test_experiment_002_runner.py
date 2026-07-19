from __future__ import annotations

import ast
import errno
import fcntl
import hashlib
import importlib
import json
import os
import struct
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

import falsewake.experiment_002_runner as runner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "src/falsewake/experiment_002_runner.py"
ENTRYPOINT = "src/falsewake/experiment_002_runner.py"
PYTHON_EXECUTABLE = "/home/ubuntu/gitcode/.t/falsewake-venv/bin/python"
BOOTSTRAP_SYS_PATH = (
    "/usr/lib/python312.zip",
    "/usr/lib/python3.12",
    "/usr/lib/python3.12/lib-dynload",
)
REGISTERED_SYS_PATH = (
    "/home/ubuntu/gitcode/falsewake/src",
    *BOOTSTRAP_SYS_PATH,
    "/home/ubuntu/gitcode/.t/falsewake-venv/lib/python3.12/site-packages",
)
EXPECTED_ENVIRONMENT = {
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
    "TMPDIR": "/home/ubuntu/gitcode/.t/falsewake-experiment-002-scratch",
    "TZ": "UTC",
}

HEAD_COMMIT = "2" * 40
IMPLEMENTATION_COMMIT = "1" * 40
DEFAULT_SOURCES = {
    "src/falsewake/__init__.py": b"ORIGIN = __spec__.origin\n",
    "src/falsewake/experiment_002_coordinator.py": (
        b"CALLED = None\n"
        b"def _run_guarded_registered_seed_child(registration, /):\n"
        b"    global CALLED\n"
        b"    CALLED = registration\n"
    ),
    "src/falsewake/experiment_002_process_guard.py": b"GUARD_STUB = True\n",
    "src/falsewake/experiment_002_run_authority.py": (
        b"CAPABILITY = object()\n"
        b"def _verify_and_issue_experiment_002_sealed_child_registration():\n"
        b"    return CAPABILITY\n"
    ),
}


def _blob_payload(
    blobs: dict[str, bytes],
    *,
    sort_paths: bool = True,
) -> bytes:
    items = list(blobs.items())
    if sort_paths:
        items.sort(key=lambda item: item[0].encode("utf-8"))
    result = bytearray(struct.pack("<I", len(items)))
    for path, payload in items:
        path_bytes = path.encode("utf-8")
        result.extend(struct.pack("<I", len(path_bytes)))
        result.extend(path_bytes)
        result.extend(struct.pack("<Q", len(payload)))
        result.extend(payload)
    return bytes(result)


def _synthetic_bundle(
    *,
    sources: dict[str, bytes] | None = None,
    frozen: dict[str, bytes] | None = None,
    canonical_registration: bool = True,
    source_payload: bytes | None = None,
    frozen_payload: bytes | None = None,
    magic: bytes = b"FW2CHLD1",
    version: int = 1,
    registration_sha256: str | None = None,
    source_bundle_sha256: str | None = None,
    trailing: bytes = b"",
) -> bytes:
    source_files = dict(DEFAULT_SOURCES if sources is None else sources)
    source_frame = (
        _blob_payload(source_files) if source_payload is None else source_payload
    )
    frozen_frame = (
        _blob_payload({"configs/synthetic.json": b"{}\n"})
        if frozen_payload is None and frozen is None
        else _blob_payload(frozen or {})
        if frozen_payload is None
        else frozen_payload
    )
    observed_source_sha256 = hashlib.sha256(
        runner._SOURCE_BUNDLE_DOMAIN + source_frame
    ).hexdigest()
    source_sha256 = (
        observed_source_sha256 if source_bundle_sha256 is None else source_bundle_sha256
    )
    paths = sorted(source_files, key=lambda value: value.encode("utf-8"))
    document = {
        "experiment": "002",
        "external_inputs": {},
        "frozen_bindings": {},
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "invocation": {},
        "runtime": {},
        "schema_version": 1,
        "source_bundle": {
            "import_roots": ["src"],
            "paths": paths,
            "payload_byte_count": len(source_frame),
            "sha256": source_sha256,
        },
    }
    if canonical_registration:
        registration = (
            json.dumps(
                document,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    else:
        registration = json.dumps(document, sort_keys=True).encode("ascii")
    registration_digest = (
        hashlib.sha256(registration).hexdigest()
        if registration_sha256 is None
        else registration_sha256
    )
    header = runner._CHILD_BUNDLE_HEADER.pack(
        magic,
        version,
        HEAD_COMMIT.encode("ascii"),
        IMPLEMENTATION_COMMIT.encode("ascii"),
        registration_digest.encode("ascii"),
        source_sha256.encode("ascii"),
        len(registration),
        len(source_frame),
        len(frozen_frame),
    )
    return header + registration + source_frame + frozen_frame + trailing


def _source_bundle_fd(
    payload: bytes = b"x",
    *,
    byte_count: int | None = None,
    mode: int = 0o400,
    seals: int = runner._SOURCE_BUNDLE_SEALS,
    read_only: bool = True,
) -> int:
    writer = os.memfd_create(
        "falsewake-runner-test-bundle",
        os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING,
    )
    result = -1
    try:
        size = len(payload) if byte_count is None else byte_count
        os.ftruncate(writer, size)
        if payload:
            os.pwrite(writer, payload[:size], 0)
        os.fchmod(writer, mode)
        fcntl.fcntl(writer, fcntl.F_ADD_SEALS, seals)
        if read_only:
            result = os.open(
                f"/proc/self/fd/{writer}",
                os.O_RDONLY | os.O_CLOEXEC,
            )
        else:
            result = writer
            writer = -1
        os.set_inheritable(result, False)
        return result
    except BaseException:
        if result >= 0:
            with suppress(OSError):
                os.close(result)
        raise
    finally:
        if writer >= 0:
            os.close(writer)


@contextmanager
def _admitted_bundle(
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes | None = None,
) -> Iterator[tuple[int, runner._SealedChildBundle]]:
    descriptor = _source_bundle_fd(_synthetic_bundle() if payload is None else payload)
    monkeypatch.setattr(runner, "_CHILD_SOURCE_BUNDLE_FD", descriptor)
    bundle: runner._SealedChildBundle | None = None
    try:
        bundle = runner._admit_sealed_child_bundle()
        yield descriptor, bundle
    finally:
        if bundle is not None:
            bundle.close()
        os.close(descriptor)


@contextmanager
def _clean_falsewake_import_surface() -> Iterator[None]:
    saved_modules = {
        name: module
        for name, module in tuple(sys.modules.items())
        if name == "falsewake" or name.startswith("falsewake.")
    }
    saved_meta_path = list(sys.meta_path)
    saved_path = list(sys.path)
    saved_cache = dict(sys.path_importer_cache)
    for name in saved_modules:
        sys.modules.pop(name, None)
    try:
        yield
    finally:
        for name in tuple(sys.modules):
            if name == "falsewake" or name.startswith("falsewake."):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
        sys.meta_path[:] = saved_meta_path
        sys.path[:] = saved_path
        sys.path_importer_cache.clear()
        sys.path_importer_cache.update(saved_cache)


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1
    return matches[0]


def _call_name(statement: ast.stmt) -> str:
    assert isinstance(statement, ast.Expr)
    assert isinstance(statement.value, ast.Call)
    assert isinstance(statement.value.func, ast.Name)
    assert not statement.value.args
    assert not statement.value.keywords
    return statement.value.func.id


def _install_valid_bootstrap(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, ...]]:
    observed_modules: list[tuple[str, ...]] = []
    monkeypatch.setattr(runner, "__file__", os.fspath(SOURCE_PATH))
    monkeypatch.setattr(os, "getcwd", lambda: os.fspath(PROJECT_ROOT))
    monkeypatch.setattr(sys, "executable", PYTHON_EXECUTABLE)
    monkeypatch.setattr(sys, "version_info", (3, 12, 3))
    monkeypatch.setattr(sys, "argv", [ENTRYPOINT])
    monkeypatch.setattr(
        sys,
        "orig_argv",
        [PYTHON_EXECUTABLE, "-I", "-S", "-B", ENTRYPOINT],
    )
    monkeypatch.setattr(sys, "path", list(BOOTSTRAP_SYS_PATH))
    monkeypatch.setattr(os, "environ", dict(EXPECTED_ENVIRONMENT))
    monkeypatch.setattr(
        sys,
        "flags",
        SimpleNamespace(
            dont_write_bytecode=1,
            ignore_environment=1,
            isolated=1,
            no_site=1,
            no_user_site=1,
        ),
    )
    monkeypatch.setattr(
        runner,
        "_require_clean_bootstrap_imports",
        lambda names: observed_modules.append(names),
    )
    return observed_modules


def test_source_has_only_standard_library_top_level_imports() -> None:
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    imports = [
        node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]

    assert imports
    for node in imports:
        if isinstance(node, ast.Import):
            assert {alias.name.partition(".")[0] for alias in node.names} <= {
                "errno",
                "fcntl",
                "hashlib",
                "importlib",
                "json",
                "mmap",
                "os",
                "stat",
                "struct",
                "sys",
                "threading",
            }
        else:
            assert node.module in {
                "__future__",
                "collections.abc",
                "dataclasses",
                "pathlib",
                "types",
                "typing",
            }

    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        and (
            (
                isinstance(node, ast.Import)
                and any(alias.name.startswith("falsewake") for alias in node.names)
            )
            or (
                isinstance(node, ast.ImportFrom)
                and node.module is not None
                and node.module.startswith("falsewake")
            )
        )
        for node in ast.walk(tree)
    )


def test_main_has_the_exact_fail_closed_sequence_and_no_arguments() -> None:
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    main = _function(tree, "main")

    assert not main.args.posonlyargs
    assert not main.args.args
    assert not main.args.kwonlyargs
    assert main.args.vararg is None
    assert main.args.kwarg is None
    assert len(main.body) == 6
    assert isinstance(main.body[0], ast.Expr)
    assert isinstance(main.body[0].value, ast.Constant)
    assert _call_name(main.body[1]) == "_validate_bootstrap"
    assignment = main.body[2]
    assert isinstance(assignment, ast.Assign)
    assert len(assignment.targets) == 1
    assert isinstance(assignment.targets[0], ast.Name)
    assert assignment.targets[0].id == "role"
    assert isinstance(assignment.value, ast.Call)
    assert isinstance(assignment.value.func, ast.Name)
    assert assignment.value.func.id == "_classify_execution_role"
    assert not assignment.value.args
    assert not assignment.value.keywords
    child_branch = main.body[3]
    assert isinstance(child_branch, ast.If)
    assert ast.unparse(child_branch.test) == "role == _CHILD_ROLE"
    assert _call_name(child_branch.body[0]) == "_run_sealed_registered_child"
    assert isinstance(child_branch.body[1], ast.Return)
    parent_guard = main.body[4]
    assert isinstance(parent_guard, ast.If)
    assert ast.unparse(parent_guard.test) == "role != _PARENT_ROLE"
    assert ast.unparse(parent_guard.body[0]).startswith("_fail(")
    assert _call_name(main.body[5]) == "_run_registered_parent"


def test_script_guard_calls_only_main() -> None:
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    guards = [node for node in tree.body if isinstance(node, ast.If)]

    assert len(guards) == 1
    guard = guards[0]
    assert ast.unparse(guard.test) == "__name__ == '__main__'"
    assert len(guard.body) == 1
    assert _call_name(guard.body[0]) == "main"
    assert not guard.orelse


def test_frozen_command_and_runtime_roots_are_exact() -> None:
    assert os.fspath(PROJECT_ROOT) == runner._REPOSITORY_ROOT
    assert os.fspath(SOURCE_PATH) == runner._CANONICAL_FILE
    assert runner._ENTRYPOINT == ENTRYPOINT
    assert runner._PYTHON_EXECUTABLE == PYTHON_EXECUTABLE
    assert runner._BOOTSTRAP_SYS_PATH == BOOTSTRAP_SYS_PATH
    assert runner._REGISTERED_SYS_PATH == REGISTERED_SYS_PATH
    assert runner._EXPECTED_ARGV == (ENTRYPOINT,)
    assert runner._EXPECTED_ORIG_ARGV == (
        PYTHON_EXECUTABLE,
        "-I",
        "-S",
        "-B",
        ENTRYPOINT,
    )
    assert runner._EXPECTED_ENVIRONMENT == EXPECTED_ENVIRONMENT


def test_registered_path_activation_replaces_the_bootstrap_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = ["attacker", *BOOTSTRAP_SYS_PATH, "tail"]
    monkeypatch.setattr(sys, "path", original)

    runner._activate_registered_sys_path()

    assert sys.path == list(REGISTERED_SYS_PATH)
    assert sys.path is original


@pytest.mark.parametrize(
    "observed",
    [
        BOOTSTRAP_SYS_PATH,
        [*BOOTSTRAP_SYS_PATH, "/unregistered"],
        [BOOTSTRAP_SYS_PATH[1], BOOTSTRAP_SYS_PATH[0], BOOTSTRAP_SYS_PATH[2]],
        [*BOOTSTRAP_SYS_PATH[:2]],
    ],
)
def test_exact_sequence_rejects_wrong_type_order_extras_and_omissions(
    observed: object,
) -> None:
    with pytest.raises(runner.Experiment002RunnerError, match="exact registered"):
        runner._require_exact_sequence(observed, BOOTSTRAP_SYS_PATH, "test path")


def test_exact_sequence_rejects_a_string_subclass_with_lying_equality() -> None:
    class LyingString(str):
        def __eq__(self, other: object) -> bool:
            del other
            return True

    observed = list(BOOTSTRAP_SYS_PATH)
    observed[0] = LyingString("/attacker-controlled")
    assert observed == list(BOOTSTRAP_SYS_PATH)

    with pytest.raises(runner.Experiment002RunnerError, match="exact registered"):
        runner._require_exact_sequence(observed, BOOTSTRAP_SYS_PATH, "test path")


def test_clean_import_check_rejects_each_protected_module_root() -> None:
    for name in (
        "falsewake",
        "falsewake.experiment_002_data",
        "numpy.core",
        "safetensors",
        "site",
        "torch.nn",
    ):
        with pytest.raises(
            runner.Experiment002RunnerError,
            match="forbidden module loaded before authorization",
        ):
            runner._require_clean_bootstrap_imports(("sys", name))


def test_clean_import_check_accepts_only_an_exact_tuple_of_names() -> None:
    runner._require_clean_bootstrap_imports(("sys", "os", "importlib"))

    with pytest.raises(runner.Experiment002RunnerError, match="snapshot is invalid"):
        runner._require_clean_bootstrap_imports(["sys", "os"])
    with pytest.raises(runner.Experiment002RunnerError, match="snapshot is invalid"):
        runner._require_clean_bootstrap_imports(("sys", object()))


def test_valid_bootstrap_is_accepted_before_any_path_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_modules = _install_valid_bootstrap(monkeypatch)
    original_path = sys.path

    runner._validate_bootstrap()

    assert sys.path is original_path
    assert sys.path == list(BOOTSTRAP_SYS_PATH)
    assert observed_modules == [tuple(sys.modules)]


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("file", "__file__"),
        ("cwd", "cwd"),
        ("executable", "executable"),
        ("version", "version"),
        ("argv", "argv"),
        ("orig_argv", "orig_argv"),
        ("sys_path", "bootstrap sys.path"),
        ("environment", "environment"),
        ("flag", "Python flag"),
    ],
)
def test_bootstrap_rejects_every_mutable_process_surface(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    message: str,
) -> None:
    _install_valid_bootstrap(monkeypatch)
    if field == "file":
        monkeypatch.setattr(runner, "__file__", f"{SOURCE_PATH}.copy")
    elif field == "cwd":
        monkeypatch.setattr(os, "getcwd", lambda: "/home/ubuntu/gitcode")
    elif field == "executable":
        monkeypatch.setattr(sys, "executable", "/usr/bin/python3.12")
    elif field == "version":
        monkeypatch.setattr(sys, "version_info", (3, 12, 4))
    elif field == "argv":
        monkeypatch.setattr(sys, "argv", [ENTRYPOINT, "--override"])
    elif field == "orig_argv":
        monkeypatch.setattr(
            sys,
            "orig_argv",
            [PYTHON_EXECUTABLE, "-I", "-S", "-B", ENTRYPOINT, "--override"],
        )
    elif field == "sys_path":
        monkeypatch.setattr(sys, "path", [*BOOTSTRAP_SYS_PATH, "/tmp"])
    elif field == "environment":
        environment = dict(EXPECTED_ENVIRONMENT)
        environment["PYTHONPATH"] = "/tmp"
        monkeypatch.setattr(os, "environ", environment)
    else:
        monkeypatch.setattr(
            sys,
            "flags",
            SimpleNamespace(
                dont_write_bytecode=1,
                ignore_environment=1,
                isolated=0,
                no_site=1,
                no_user_site=1,
            ),
        )

    with pytest.raises(runner.Experiment002RunnerError, match=message):
        runner._validate_bootstrap()


def test_noncanonical_entrypoint_resolution_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_valid_bootstrap(monkeypatch)

    def diverted_realpath(path: str) -> str:
        if path == os.fspath(SOURCE_PATH):
            return (
                "/home/ubuntu/gitcode/falsewake-copy/src/falsewake/"
                "experiment_002_runner.py"
            )
        if path == os.fspath(PROJECT_ROOT):
            return os.fspath(PROJECT_ROOT)
        raise AssertionError(f"unexpected path: {path}")

    monkeypatch.setattr(os.path, "realpath", diverted_realpath)

    with pytest.raises(runner.Experiment002RunnerError, match="resolves outside"):
        runner._validate_bootstrap()


def test_authority_import_and_issuance_use_one_fixed_zero_argument_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability = object()
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
    authority = ModuleType("falsewake.experiment_002_run_authority")

    def issue(*args: object, **kwargs: object) -> object:
        calls.append(("issue", args, kwargs))
        return capability

    authority.__dict__["verify_and_issue_experiment_002_run_registration"] = issue

    def import_module(name: str) -> ModuleType:
        calls.append((name, (), {}))
        return authority

    monkeypatch.setattr(importlib, "import_module", import_module)

    observed = runner._issue_registration()

    assert observed is capability
    assert calls == [
        ("falsewake.experiment_002_run_authority", (), {}),
        ("issue", (), {}),
    ]


def test_future_coordinator_is_dynamically_imported_and_gets_only_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability = object()
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
    coordinator = ModuleType("falsewake.experiment_002_coordinator")

    def run_registered(*args: object, **kwargs: object) -> None:
        calls.append(("run", args, kwargs))

    coordinator.__dict__["run_registered_experiment"] = run_registered

    def import_module(name: str) -> ModuleType:
        calls.append((name, (), {}))
        return coordinator

    monkeypatch.setattr(importlib, "import_module", import_module)

    runner._run_coordinator(capability)

    assert calls == [
        ("falsewake.experiment_002_coordinator", (), {}),
        ("run", (capability,), {}),
    ]


def test_missing_fixed_coordinator_function_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = ModuleType("falsewake.experiment_002_coordinator")
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: coordinator,
    )

    with pytest.raises(AttributeError, match="run_registered_experiment"):
        runner._run_coordinator(object())


def test_exact_isolated_command_reaches_the_absent_registration_gate() -> None:
    completed = subprocess.run(
        (
            PYTHON_EXECUTABLE,
            "-I",
            "-S",
            "-B",
            ENTRYPOINT,
        ),
        cwd=PROJECT_ROOT,
        env=EXPECTED_ENVIRONMENT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode != 0
    assert completed.stdout == ""
    assert "Experiment002RunAuthorityError" in completed.stderr
    assert "required regular file is absent or inaccessible" in completed.stderr
    assert "configs/experiment-002-run.json" in completed.stderr
    assert "Experiment002RunnerError" not in completed.stderr
    assert "experiment_002_coordinator" not in completed.stderr


def test_main_orders_validation_activation_issuance_and_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        runner,
        "_validate_bootstrap",
        lambda: events.append("validate"),
    )

    def classify_parent() -> str:
        events.append("classify")
        return runner._PARENT_ROLE

    monkeypatch.setattr(
        runner,
        "_classify_execution_role",
        classify_parent,
    )
    monkeypatch.setattr(
        runner,
        "_run_registered_parent",
        lambda: events.append("parent"),
    )
    monkeypatch.setattr(
        runner,
        "_run_sealed_registered_child",
        lambda: events.append("child"),
    )

    runner.main()

    assert events == ["validate", "classify", "parent"]


@pytest.mark.parametrize("failed_step", ["validate", "classify", "parent"])
def test_main_never_advances_past_a_failed_gate(
    monkeypatch: pytest.MonkeyPatch,
    failed_step: str,
) -> None:
    events: list[str] = []

    def gate(name: str) -> Callable[[], None]:
        def execute() -> None:
            events.append(name)
            if name == failed_step:
                raise RuntimeError(name)

        return execute

    monkeypatch.setattr(runner, "_validate_bootstrap", gate("validate"))

    def classify() -> str:
        gate("classify")()
        return runner._PARENT_ROLE

    monkeypatch.setattr(runner, "_classify_execution_role", classify)
    monkeypatch.setattr(
        runner,
        "_run_registered_parent",
        gate("parent"),
    )
    monkeypatch.setattr(
        runner,
        "_run_sealed_registered_child",
        lambda: events.append("child"),
    )

    with pytest.raises(RuntimeError, match=failed_step):
        runner.main()

    expected = ["validate", "classify", "parent"]
    assert events == expected[: expected.index(failed_step) + 1]


def test_fd3_exact_ebadf_is_the_only_parent_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = os.dup(0)
    os.close(descriptor)
    monkeypatch.setattr(runner, "_CHILD_CONTROL_FD", descriptor)

    assert runner._classify_execution_role() == runner._PARENT_ROLE


@pytest.mark.parametrize("kind", ["file", "pipe"])
def test_any_open_cloexec_fd3_irrevocably_classifies_as_child(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    kind: str,
) -> None:
    cleanup: list[int] = []
    if kind == "file":
        path = tmp_path / "not-a-socket"
        path.write_bytes(b"malformed child control")
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    else:
        descriptor, writer = os.pipe2(os.O_CLOEXEC)
        cleanup.append(writer)
    cleanup.append(descriptor)
    try:
        monkeypatch.setattr(runner, "_CHILD_CONTROL_FD", descriptor)
        assert runner._classify_execution_role() == runner._CHILD_ROLE
    finally:
        for opened in cleanup:
            os.close(opened)


@pytest.mark.parametrize("kind", ["file", "pipe"])
def test_malformed_open_fd3_child_failure_never_runs_parent_route(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    kind: str,
) -> None:
    cleanup: list[int] = []
    if kind == "file":
        path = tmp_path / "fd3-malformed"
        path.write_bytes(b"not seqpacket")
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    else:
        descriptor, writer = os.pipe2(os.O_CLOEXEC)
        cleanup.append(writer)
    cleanup.append(descriptor)
    events: list[str] = []
    try:
        monkeypatch.setattr(runner, "_CHILD_CONTROL_FD", descriptor)
        monkeypatch.setattr(runner, "_validate_bootstrap", lambda: None)

        def fail_child() -> None:
            events.append("child")
            raise RuntimeError("malformed child route")

        monkeypatch.setattr(runner, "_run_sealed_registered_child", fail_child)
        monkeypatch.setattr(
            runner,
            "_run_registered_parent",
            lambda: events.append("parent"),
        )
        with pytest.raises(RuntimeError, match="malformed child route"):
            runner.main()
        assert events == ["child"]
    finally:
        for opened in cleanup:
            os.close(opened)


def test_open_fd3_without_cloexec_is_terminal_not_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor, writer = os.pipe()
    try:
        os.set_inheritable(descriptor, True)
        monkeypatch.setattr(runner, "_CHILD_CONTROL_FD", descriptor)
        with pytest.raises(
            runner.Experiment002RunnerError,
            match="control descriptor 3.*close-on-exec",
        ):
            runner._classify_execution_role()
    finally:
        os.close(descriptor)
        os.close(writer)


def test_fd3_non_ebadf_inspection_error_is_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_fcntl(descriptor: int, operation: int) -> int:
        del descriptor, operation
        raise OSError(errno.EIO, "synthetic inspection failure")

    monkeypatch.setattr(fcntl, "fcntl", fail_fcntl)
    with pytest.raises(
        runner.Experiment002RunnerError,
        match="could not be classified",
    ):
        runner._classify_execution_role()


def test_child_failure_never_falls_back_to_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        runner,
        "_validate_bootstrap",
        lambda: events.append("validate"),
    )

    def classify_child() -> str:
        events.append("classify")
        return runner._CHILD_ROLE

    monkeypatch.setattr(
        runner,
        "_classify_execution_role",
        classify_child,
    )

    def fail_child() -> None:
        events.append("child")
        raise RuntimeError("sealed child failed")

    monkeypatch.setattr(runner, "_run_sealed_registered_child", fail_child)
    monkeypatch.setattr(
        runner,
        "_run_registered_parent",
        lambda: events.append("parent"),
    )
    with pytest.raises(RuntimeError, match="sealed child failed"):
        runner.main()
    assert events == ["validate", "classify", "child"]


def test_fd7_exact_aggregate_size_bound_is_inclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = _source_bundle_fd(
        b"",
        byte_count=runner._MAX_CHILD_BUNDLE_BYTES,
    )
    try:
        monkeypatch.setattr(runner, "_CHILD_SOURCE_BUNDLE_FD", descriptor)
        frame, offset = runner._inspect_source_bundle_descriptor(descriptor)
        assert frame.stat_frame[6] == 256 << 20
        assert offset == 0
    finally:
        os.close(descriptor)


def test_fd7_rejects_one_byte_over_aggregate_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = _source_bundle_fd(
        b"",
        byte_count=runner._MAX_CHILD_BUNDLE_BYTES + 1,
    )
    try:
        monkeypatch.setattr(runner, "_CHILD_SOURCE_BUNDLE_FD", descriptor)
        with pytest.raises(
            runner.Experiment002RunnerError,
            match="bounded anonymous regular file",
        ):
            runner._inspect_source_bundle_descriptor(descriptor)
    finally:
        os.close(descriptor)


@pytest.mark.parametrize("kind", ["mode", "seal", "writable", "inheritable"])
def test_fd7_rejects_wrong_geometry(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    descriptor = _source_bundle_fd(
        _synthetic_bundle(),
        mode=0o440 if kind == "mode" else 0o400,
        seals=(
            runner._SOURCE_BUNDLE_SEALS & ~fcntl.F_SEAL_WRITE
            if kind == "seal"
            else runner._SOURCE_BUNDLE_SEALS
        ),
        read_only=kind != "writable",
    )
    try:
        if kind == "inheritable":
            os.set_inheritable(descriptor, True)
        monkeypatch.setattr(runner, "_CHILD_SOURCE_BUNDLE_FD", descriptor)
        with pytest.raises(runner.Experiment002RunnerError, match="source bundle"):
            runner._inspect_source_bundle_descriptor(descriptor)
    finally:
        os.close(descriptor)


@pytest.mark.parametrize("kind", ["disk", "pipe"])
def test_fd7_rejects_non_memfd_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    kind: str,
) -> None:
    cleanup: list[int] = []
    if kind == "disk":
        path = tmp_path / "bundle"
        path.write_bytes(_synthetic_bundle())
        path.chmod(0o400)
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    else:
        descriptor, writer = os.pipe2(os.O_CLOEXEC)
        cleanup.append(writer)
    cleanup.append(descriptor)
    try:
        monkeypatch.setattr(runner, "_CHILD_SOURCE_BUNDLE_FD", descriptor)
        with pytest.raises(runner.Experiment002RunnerError, match="source bundle"):
            runner._inspect_source_bundle_descriptor(descriptor)
    finally:
        for opened in cleanup:
            os.close(opened)


def test_fd7_admission_and_loader_reads_do_not_change_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _admitted_bundle(monkeypatch) as (descriptor, bundle):
        os.lseek(descriptor, 13, os.SEEK_SET)
        # Admission recorded offset zero, so restore that exact admitted frame
        # before its invariant check and exercise an arbitrary stable offset in
        # a separately admitted instance below.
        os.lseek(descriptor, bundle.descriptor_offset, os.SEEK_SET)
        assert bundle.descriptor_offset == 0
        initial = os.lseek(descriptor, 0, os.SEEK_CUR)
        modules, _origins = runner._module_frames(bundle)
        authority = modules[runner._AUTHORITY_MODULE]
        view = memoryview(bundle.mapping)[
            authority.blob.offset : authority.blob.offset + authority.blob.byte_count
        ]
        try:
            compile(view, authority.origin, "exec")
        finally:
            view.release()
        assert os.lseek(descriptor, 0, os.SEEK_CUR) == initial


def test_fd7_nonzero_offset_is_preserved_across_full_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = _source_bundle_fd(_synthetic_bundle())
    bundle: runner._SealedChildBundle | None = None
    try:
        os.lseek(descriptor, 17, os.SEEK_SET)
        monkeypatch.setattr(runner, "_CHILD_SOURCE_BUNDLE_FD", descriptor)
        bundle = runner._admit_sealed_child_bundle()
        assert bundle.descriptor_offset == 17
        assert os.lseek(descriptor, 0, os.SEEK_CUR) == 17
    finally:
        if bundle is not None:
            bundle.close()
        os.close(descriptor)


def test_fd7_fchmod_after_admission_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _admitted_bundle(monkeypatch) as (descriptor, bundle):
        os.fchmod(descriptor, 0o600)
        with pytest.raises(
            runner.Experiment002RunnerError,
            match="source bundle|descriptor",
        ):
            runner._require_sealed_child_bundle_stable(bundle)


def test_fd7_stat_race_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    descriptor = _source_bundle_fd(_synthetic_bundle())
    monkeypatch.setattr(runner, "_CHILD_SOURCE_BUNDLE_FD", descriptor)
    real_stat_frame = runner._stat_frame
    calls = 0

    def changing_stat_frame(value: os.stat_result) -> tuple[int, ...]:
        nonlocal calls
        calls += 1
        frame = real_stat_frame(value)
        if calls == 2:
            return (*frame[:-1], frame[-1] + 1)
        return frame

    monkeypatch.setattr(runner, "_stat_frame", changing_stat_frame)
    try:
        with pytest.raises(
            runner.Experiment002RunnerError,
            match="changed while being inspected",
        ):
            runner._inspect_source_bundle_descriptor(descriptor)
    finally:
        os.close(descriptor)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (_synthetic_bundle(magic=b"BADMAGIC"), "magic or version"),
        (_synthetic_bundle(version=2), "magic or version"),
        (_synthetic_bundle(registration_sha256="0" * 64), "registration digest"),
        (_synthetic_bundle(source_bundle_sha256="0" * 64), "source digest"),
        (_synthetic_bundle(canonical_registration=False), "not canonical"),
        (_synthetic_bundle(trailing=b"x"), "truncated, trailing"),
        (_synthetic_bundle(frozen_payload=struct.pack("<I", 0)), "file count"),
    ],
)
def test_child_bundle_parser_rejects_header_digest_and_frame_tampering(
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes,
    message: str,
) -> None:
    with (
        pytest.raises(runner.Experiment002RunnerError, match=message),
        _admitted_bundle(monkeypatch, payload),
    ):
        pass


@pytest.mark.parametrize(
    "missing_path",
    [
        "src/falsewake/experiment_002_coordinator.py",
        "src/falsewake/experiment_002_process_guard.py",
    ],
)
def test_child_bundle_parser_requires_all_authorization_boundary_sources(
    monkeypatch: pytest.MonkeyPatch,
    missing_path: str,
) -> None:
    sources = dict(DEFAULT_SOURCES)
    del sources[missing_path]
    with (
        pytest.raises(
            runner.Experiment002RunnerError,
            match="required sealed bootstrap source",
        ),
        _admitted_bundle(monkeypatch, _synthetic_bundle(sources=sources)),
    ):
        pass


def test_child_bundle_parser_rejects_unsorted_source_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = dict(reversed(tuple(DEFAULT_SOURCES.items())))
    unsorted_payload = _blob_payload(sources, sort_paths=False)
    with (
        pytest.raises(
            runner.Experiment002RunnerError,
            match="not UTF-8 sorted",
        ),
        _admitted_bundle(
            monkeypatch,
            _synthetic_bundle(sources=sources, source_payload=unsorted_payload),
        ),
    ):
        pass


def test_child_bundle_parser_rejects_path_traversal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = dict(DEFAULT_SOURCES)
    sources["src/falsewake/../escape.py"] = b"ESCAPE = True\n"
    with (
        pytest.raises(runner.Experiment002RunnerError, match="path traversal"),
        _admitted_bundle(monkeypatch, _synthetic_bundle(sources=sources)),
    ):
        pass


def test_sealed_finder_loads_exact_origins_and_stable_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _admitted_bundle(monkeypatch) as (descriptor, bundle):
        initial_offset = os.lseek(descriptor, 0, os.SEEK_CUR)
        with _clean_falsewake_import_surface():
            monkeypatch.setattr(
                runner,
                "_require_clean_bootstrap_imports",
                lambda names: None,
            )
            finder = runner._install_sealed_source_finder(bundle)
            get_data = finder.get_data
            verify = finder._verify_sealed_import_state
            assert finder.get_data is get_data
            assert finder._verify_sealed_import_state is verify
            assert sys.meta_path[0] is finder
            assert tuple(sys.path) == runner._RUNTIME_SYS_PATH_ROOTS
            assert f"{runner._REPOSITORY_ROOT}/src" not in sys.path
            assert not sys.path_importer_cache

            package = importlib.import_module("falsewake")
            authority = importlib.import_module(runner._AUTHORITY_MODULE)
            coordinator = importlib.import_module(runner._COORDINATOR_MODULE)
            verify()
            prefix = f"falsewake-sealed://experiment-002/{bundle.source_bundle_sha256}/"
            assert package.__file__ == prefix + "src/falsewake/__init__.py"
            assert authority.__file__ == (
                prefix + "src/falsewake/experiment_002_run_authority.py"
            )
            assert coordinator.__file__ == (
                prefix + "src/falsewake/experiment_002_coordinator.py"
            )
            assert authority.__loader__ is finder
            assert authority.__spec__ is not None
            assert authority.__spec__.loader is finder
            assert (
                get_data(authority.__file__)
                == DEFAULT_SOURCES["src/falsewake/experiment_002_run_authority.py"]
            )
            assert os.lseek(descriptor, 0, os.SEEK_CUR) == initial_offset


def test_real_authority_captures_the_exact_sealed_finder_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_source = (
        PROJECT_ROOT / "src/falsewake/experiment_002_run_authority.py"
    ).read_bytes()
    sources = dict(DEFAULT_SOURCES)
    sources["src/falsewake/experiment_002_run_authority.py"] = authority_source
    with (
        _admitted_bundle(
            monkeypatch,
            _synthetic_bundle(sources=sources),
        ) as (_descriptor, bundle),
        _clean_falsewake_import_surface(),
    ):
        monkeypatch.setattr(
            runner,
            "_require_clean_bootstrap_imports",
            lambda names: None,
        )
        finder = runner._install_sealed_source_finder(bundle)
        importlib.import_module("falsewake")
        authority = importlib.import_module(runner._AUTHORITY_MODULE)

        assert authority._LOADED_SOURCE_ORIGIN_KIND == "sealed_child"
        assert authority._LOADED_SOURCE_ORIGIN is authority.__file__
        assert authority.__spec__ is not None
        assert authority._LOADED_SOURCE_ORIGIN is authority.__spec__.origin
        assert authority._LOADED_SOURCE_LOADER is finder
        assert authority._LOADED_MODULE_SPEC is authority.__spec__
        assert authority._LOADED_MODULE is authority
        assert authority._LOADED_SOURCE_GET_DATA_ROUTE is finder.get_data
        assert (
            authority._LOADED_FINDER_VERIFY_ROUTE is finder._verify_sealed_import_state
        )
        assert authority._LOADED_META_PATH_OBJECT is sys.meta_path
        assert len(authority._LOADED_META_PATH_FRAME) == len(sys.meta_path)
        assert all(
            captured is observed
            for captured, observed in zip(
                authority._LOADED_META_PATH_FRAME,
                sys.meta_path,
                strict=True,
            )
        )
        assert authority._LOADED_SYS_PATH_OBJECT is sys.path
        assert len(authority._LOADED_SYS_PATH_FRAME) == len(sys.path)
        assert all(
            captured is observed
            for captured, observed in zip(
                authority._LOADED_SYS_PATH_FRAME,
                sys.path,
                strict=True,
            )
        )
        assert (
            hashlib.sha256(authority_source).hexdigest()
            == authority._LOADED_SOURCE_SHA256
        )
        assert finder.get_data(authority._LOADED_SOURCE_ORIGIN) == authority_source
        finder._verify_sealed_import_state()


def test_sealed_finder_hard_denies_poisoned_worktree_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    poisoned = tmp_path / "falsewake"
    poisoned.mkdir()
    (poisoned / "missing.py").write_text(
        "raise AssertionError('poisoned worktree imported')\n",
        encoding="utf-8",
    )
    with (
        _admitted_bundle(monkeypatch) as (_descriptor, bundle),
        _clean_falsewake_import_surface(),
    ):
        monkeypatch.setattr(
            runner,
            "_RUNTIME_SYS_PATH_ROOTS",
            (*runner._RUNTIME_SYS_PATH_ROOTS, os.fspath(tmp_path)),
        )
        monkeypatch.setattr(
            runner,
            "_require_clean_bootstrap_imports",
            lambda names: None,
        )
        finder = runner._install_sealed_source_finder(bundle)
        importlib.import_module("falsewake")
        with pytest.raises(ModuleNotFoundError, match="sealed source bundle"):
            importlib.import_module("falsewake.missing")
        with pytest.raises(
            runner.Experiment002RunnerError,
            match="terminally failed",
        ):
            finder._verify_sealed_import_state()


def test_loader_get_data_rejects_every_unregistered_virtual_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with (
        _admitted_bundle(monkeypatch) as (_descriptor, bundle),
        _clean_falsewake_import_surface(),
    ):
        monkeypatch.setattr(
            runner,
            "_require_clean_bootstrap_imports",
            lambda names: None,
        )
        finder = runner._install_sealed_source_finder(bundle)
        with pytest.raises(
            runner.Experiment002RunnerError,
            match="origin is not registered",
        ):
            finder.get_data(
                "falsewake-sealed://experiment-002/"
                f"{bundle.source_bundle_sha256}/src/falsewake/missing.py"
            )
        with pytest.raises(
            runner.Experiment002RunnerError,
            match="terminally failed",
        ):
            finder._verify_sealed_import_state()


def test_verifier_rejects_a_sys_modules_string_subclass_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Alias(str):
        pass

    with (
        _admitted_bundle(monkeypatch) as (_descriptor, bundle),
        _clean_falsewake_import_surface(),
    ):
        monkeypatch.setattr(
            runner,
            "_require_clean_bootstrap_imports",
            lambda names: None,
        )
        finder = runner._install_sealed_source_finder(bundle)
        importlib.import_module("falsewake")
        alias = Alias("falsewake.missing")
        fake = ModuleType("falsewake.missing")
        try:
            sys.modules[alias] = fake
            assert sys.modules["falsewake.missing"] is fake
            with pytest.raises(
                runner.Experiment002RunnerError,
                match="non-exact string key",
            ):
                finder._verify_sealed_import_state()
        finally:
            sys.modules.pop(alias, None)
        with pytest.raises(
            runner.Experiment002RunnerError,
            match="terminally failed",
        ):
            finder._verify_sealed_import_state()


def test_verifier_rejects_a_sys_path_subclass_with_lying_equality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LyingPath(str):
        def __eq__(self, other: object) -> bool:
            del other
            return True

    with (
        _admitted_bundle(monkeypatch) as (_descriptor, bundle),
        _clean_falsewake_import_surface(),
    ):
        monkeypatch.setattr(
            runner,
            "_require_clean_bootstrap_imports",
            lambda names: None,
        )
        finder = runner._install_sealed_source_finder(bundle)
        original = sys.path[0]
        sys.path[0] = LyingPath("/attacker-controlled")
        assert sys.path == list(runner._RUNTIME_SYS_PATH_ROOTS)
        with pytest.raises(
            runner.Experiment002RunnerError,
            match="sealed child sys.path changed",
        ):
            finder._verify_sealed_import_state()
        sys.path[0] = original
        with pytest.raises(
            runner.Experiment002RunnerError,
            match="terminally failed",
        ):
            finder._verify_sealed_import_state()


@pytest.mark.parametrize("registry", ["meta_path", "modules", "path"])
def test_verifier_holds_strong_identity_for_each_import_registry(
    monkeypatch: pytest.MonkeyPatch,
    registry: str,
) -> None:
    with (
        _admitted_bundle(monkeypatch) as (_descriptor, bundle),
        _clean_falsewake_import_surface(),
    ):
        monkeypatch.setattr(
            runner,
            "_require_clean_bootstrap_imports",
            lambda names: None,
        )
        finder = runner._install_sealed_source_finder(bundle)
        importlib.import_module("falsewake")
        original_meta_path = sys.meta_path
        original_modules = sys.modules
        original_path = sys.path
        try:
            if registry == "meta_path":
                sys.meta_path = list(sys.meta_path)
            elif registry == "modules":
                sys.modules = dict(sys.modules)
            else:
                sys.path = list(sys.path)
            with pytest.raises(runner.Experiment002RunnerError):
                finder._verify_sealed_import_state()
        finally:
            sys.meta_path = original_meta_path
            sys.modules = original_modules
            sys.path = original_path
        with pytest.raises(
            runner.Experiment002RunnerError,
            match="terminally failed",
        ):
            finder._verify_sealed_import_state()


def test_transient_fd7_offset_failure_latches_the_finder_terminally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with (
        _admitted_bundle(monkeypatch) as (descriptor, bundle),
        _clean_falsewake_import_surface(),
    ):
        monkeypatch.setattr(
            runner,
            "_require_clean_bootstrap_imports",
            lambda names: None,
        )
        finder = runner._install_sealed_source_finder(bundle)
        importlib.import_module("falsewake")
        os.lseek(descriptor, bundle.descriptor_offset + 1, os.SEEK_SET)
        with pytest.raises(
            runner.Experiment002RunnerError,
            match="descriptor or offset changed",
        ):
            finder._verify_sealed_import_state()
        os.lseek(descriptor, bundle.descriptor_offset, os.SEEK_SET)
        with pytest.raises(
            runner.Experiment002RunnerError,
            match="terminally failed",
        ):
            finder._verify_sealed_import_state()


@pytest.mark.parametrize("operation", ["reload", "reimport"])
def test_sealed_module_reload_and_deletion_reimport_are_terminal(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    with (
        _admitted_bundle(monkeypatch) as (_descriptor, bundle),
        _clean_falsewake_import_surface(),
    ):
        monkeypatch.setattr(
            runner,
            "_require_clean_bootstrap_imports",
            lambda names: None,
        )
        finder = runner._install_sealed_source_finder(bundle)
        package = importlib.import_module("falsewake")
        if operation == "reload":
            with pytest.raises(
                runner.Experiment002RunnerError,
                match="cannot be reloaded",
            ):
                importlib.reload(package)
        else:
            del sys.modules["falsewake"]
            with pytest.raises(
                runner.Experiment002RunnerError,
                match="cannot be retried",
            ):
                importlib.import_module("falsewake")
        with pytest.raises(runner.Experiment002RunnerError):
            finder._verify_sealed_import_state()


def test_failed_sealed_module_import_cannot_be_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = dict(DEFAULT_SOURCES)
    sources["src/falsewake/experiment_002_run_authority.py"] = b"not valid python !\n"
    with (
        _admitted_bundle(
            monkeypatch,
            _synthetic_bundle(sources=sources),
        ) as (_descriptor, bundle),
        _clean_falsewake_import_surface(),
    ):
        monkeypatch.setattr(
            runner,
            "_require_clean_bootstrap_imports",
            lambda names: None,
        )
        runner._install_sealed_source_finder(bundle)
        importlib.import_module("falsewake")
        with pytest.raises(SyntaxError):
            importlib.import_module(runner._AUTHORITY_MODULE)
        with pytest.raises(
            runner.Experiment002RunnerError,
            match="terminally failed",
        ):
            importlib.import_module(runner._AUTHORITY_MODULE)


def test_sealed_finder_supports_nested_exact_source_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = dict(DEFAULT_SOURCES)
    sources["src/falsewake/helper.py"] = b"VALUE = 17\n"
    sources["src/falsewake/experiment_002_run_authority.py"] = (
        b"from falsewake import helper\n"
        b"CAPABILITY = helper.VALUE\n"
        b"def _verify_and_issue_experiment_002_sealed_child_registration():\n"
        b"    return CAPABILITY\n"
    )
    with (
        _admitted_bundle(
            monkeypatch,
            _synthetic_bundle(sources=sources),
        ) as (_descriptor, bundle),
        _clean_falsewake_import_surface(),
    ):
        monkeypatch.setattr(
            runner,
            "_require_clean_bootstrap_imports",
            lambda names: None,
        )
        finder = runner._install_sealed_source_finder(bundle)
        importlib.import_module("falsewake")
        authority = importlib.import_module(runner._AUTHORITY_MODULE)
        assert authority.CAPABILITY == 17
        finder._verify_sealed_import_state()


@pytest.mark.parametrize(
    "mutation",
    [
        "tail",
        "spec",
        "spec_clone",
        "spec_locations",
        "spec_name",
        "spec_origin",
        "spec_has_location",
        "package_path",
        "module_name",
        "module_package",
        "module_file",
    ],
)
def test_finder_verifier_rejects_import_surface_identity_mutation(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    with (
        _admitted_bundle(monkeypatch) as (_descriptor, bundle),
        _clean_falsewake_import_surface(),
    ):
        monkeypatch.setattr(
            runner,
            "_require_clean_bootstrap_imports",
            lambda names: None,
        )
        finder = runner._install_sealed_source_finder(bundle)
        package = importlib.import_module("falsewake")
        if mutation == "tail":
            sys.meta_path.append(cast(Any, object()))
        elif mutation == "spec":
            package.__spec__ = cast(
                Any,
                SimpleNamespace(
                    name="falsewake",
                    loader=finder,
                    origin=package.__file__,
                    submodule_search_locations=[],
                ),
            )
        elif mutation == "spec_clone":
            assert package.__spec__ is not None
            cloned_spec = importlib.machinery.ModuleSpec(
                package.__spec__.name,
                finder,
                origin=package.__spec__.origin,
                is_package=True,
            )
            cloned_spec.has_location = True
            cloned_spec.submodule_search_locations = cast(list[str], package.__path__)
            package.__spec__ = cloned_spec
        elif mutation == "spec_locations":
            assert package.__spec__ is not None
            package.__spec__.submodule_search_locations = []
        elif mutation == "spec_name":
            assert package.__spec__ is not None
            package.__spec__.name = type("Alias", (str,), {})(package.__spec__.name)
        elif mutation == "spec_origin":
            assert package.__spec__ is not None
            assert package.__spec__.origin is not None
            package.__spec__.origin = type("Alias", (str,), {})(package.__spec__.origin)
        elif mutation == "spec_has_location":
            assert package.__spec__ is not None
            package.__spec__.has_location = False
        elif mutation == "package_path":
            package.__path__.append("poisoned")
        elif mutation == "module_name":
            package.__name__ = type("Alias", (str,), {})(package.__name__)
        elif mutation == "module_package":
            package.__package__ = type("Alias", (str,), {})(package.__package__)
        else:
            assert package.__file__ is not None
            package.__file__ = type("Alias", (str,), {})(package.__file__)
        with pytest.raises(runner.Experiment002RunnerError):
            finder._verify_sealed_import_state()


def test_full_synthetic_child_route_uses_only_sealed_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = _source_bundle_fd(_synthetic_bundle())
    finder: runner._SealedSourceFinder | None = None
    try:
        monkeypatch.setattr(runner, "_CHILD_SOURCE_BUNDLE_FD", descriptor)
        monkeypatch.setattr(
            runner,
            "_require_clean_bootstrap_imports",
            lambda names: None,
        )
        with _clean_falsewake_import_surface():
            runner._run_sealed_registered_child()
            finder = cast(runner._SealedSourceFinder, sys.meta_path[0])
            authority = importlib.import_module(runner._AUTHORITY_MODULE)
            coordinator = importlib.import_module(runner._COORDINATOR_MODULE)
            assert coordinator.CALLED is authority.CAPABILITY
            assert isinstance(authority.__file__, str)
            assert authority.__file__.startswith("falsewake-sealed://experiment-002/")
            assert finder._bundle.mapping.closed is False
            finder._verify_sealed_import_state()
    finally:
        if finder is not None:
            finder._bundle.close()
        os.close(descriptor)
