from __future__ import annotations

import ast
import importlib
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace

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
                "importlib",
                "os",
                "sys",
            }
        else:
            assert node.module in {"__future__", "collections.abc", "typing"}

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
    assert len(main.body) == 5
    assert isinstance(main.body[0], ast.Expr)
    assert isinstance(main.body[0].value, ast.Constant)
    assert _call_name(main.body[1]) == "_validate_bootstrap"
    assert _call_name(main.body[2]) == "_activate_registered_sys_path"
    assignment = main.body[3]
    assert isinstance(assignment, ast.Assign)
    assert len(assignment.targets) == 1
    assert isinstance(assignment.targets[0], ast.Name)
    assert assignment.targets[0].id == "registration"
    assert isinstance(assignment.value, ast.Call)
    assert isinstance(assignment.value.func, ast.Name)
    assert assignment.value.func.id == "_issue_registration"
    assert not assignment.value.args
    assert not assignment.value.keywords
    final_call = main.body[4]
    assert isinstance(final_call, ast.Expr)
    assert isinstance(final_call.value, ast.Call)
    assert isinstance(final_call.value.func, ast.Name)
    assert final_call.value.func.id == "_run_coordinator"
    assert len(final_call.value.args) == 1
    assert isinstance(final_call.value.args[0], ast.Name)
    assert final_call.value.args[0].id == "registration"
    assert not final_call.value.keywords


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
    capability = object()
    events: list[tuple[str, object | None]] = []
    monkeypatch.setattr(
        runner,
        "_validate_bootstrap",
        lambda: events.append(("validate", None)),
    )
    monkeypatch.setattr(
        runner,
        "_activate_registered_sys_path",
        lambda: events.append(("activate", None)),
    )

    def issue() -> object:
        events.append(("issue", None))
        return capability

    monkeypatch.setattr(runner, "_issue_registration", issue)
    monkeypatch.setattr(
        runner,
        "_run_coordinator",
        lambda registration: events.append(("run", registration)),
    )

    runner.main()

    assert events == [
        ("validate", None),
        ("activate", None),
        ("issue", None),
        ("run", capability),
    ]


@pytest.mark.parametrize("failed_step", ["validate", "activate", "issue"])
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
    monkeypatch.setattr(runner, "_activate_registered_sys_path", gate("activate"))

    def issue() -> object:
        gate("issue")()
        return object()

    monkeypatch.setattr(runner, "_issue_registration", issue)
    monkeypatch.setattr(
        runner,
        "_run_coordinator",
        lambda registration: events.append("run"),
    )

    with pytest.raises(RuntimeError, match=failed_step):
        runner.main()

    expected = ["validate", "activate", "issue"]
    assert events == expected[: expected.index(failed_step) + 1]
