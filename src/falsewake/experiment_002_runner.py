"""Exact source-bound parent bootstrap for Experiment 002.

This file is a script entrypoint, not a configurable command-line interface.
It admits one frozen interpreter invocation, replaces the isolated interpreter's
bootstrap path with the registered import path, and only then imports the run
authority and the future experiment coordinator.
"""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Callable
from typing import Final, NoReturn, Protocol, cast

_REPOSITORY_ROOT: Final = "/home/ubuntu/gitcode/falsewake"
_ENTRYPOINT: Final = "src/falsewake/experiment_002_runner.py"
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
    "TMPDIR": "/home/ubuntu/gitcode/.t/falsewake-experiment-002-scratch",
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
_AUTHORITY_MODULE: Final = "falsewake.experiment_002_run_authority"
_COORDINATOR_MODULE: Final = "falsewake.experiment_002_coordinator"


class Experiment002RunnerError(RuntimeError):
    """The exact Experiment 002 parent bootstrap failed closed."""


class _Coordinator(Protocol):
    def run_registered_experiment(self, registration: object, /) -> None:
        """Run the single registered experiment without caller overrides."""


def _fail(message: str) -> NoReturn:
    raise Experiment002RunnerError(message)


def _require_exact_sequence(
    observed: object,
    expected: tuple[str, ...],
    name: str,
) -> None:
    if type(observed) is not list or tuple(cast(list[object], observed)) != expected:
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


def _activate_registered_sys_path() -> None:
    sys.path[:] = _REGISTERED_SYS_PATH
    _require_exact_sequence(sys.path, _REGISTERED_SYS_PATH, "registered sys.path")


def _issue_registration() -> object:
    authority = importlib.import_module(_AUTHORITY_MODULE)
    issuer = cast(
        Callable[[], object],
        authority.verify_and_issue_experiment_002_run_registration,
    )
    return issuer()


def _run_coordinator(registration: object) -> None:
    coordinator = cast(_Coordinator, importlib.import_module(_COORDINATOR_MODULE))
    coordinator.run_registered_experiment(registration)


def main() -> None:
    """Validate, authorize, and execute the one registered Experiment 002 run."""

    _validate_bootstrap()
    _activate_registered_sys_path()
    registration = _issue_registration()
    _run_coordinator(registration)


if __name__ == "__main__":
    main()
