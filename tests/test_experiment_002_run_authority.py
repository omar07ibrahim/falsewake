from __future__ import annotations

import ast
import copy
import hashlib
import importlib.metadata
import inspect
import json
import os
import pickle
import platform
import shutil
import struct
import subprocess
import sys
import threading
import warnings
import weakref
from concurrent.futures import ThreadPoolExecutor
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


def _source_digest(repository: Path, source_paths: tuple[str, ...]) -> tuple[int, str]:
    payload = bytearray(struct.pack("<I", len(source_paths)))
    for path in source_paths:
        path_bytes = path.encode("utf-8")
        blob = (repository / path).read_bytes()
        payload.extend(struct.pack("<I", len(path_bytes)))
        payload.extend(path_bytes)
        payload.extend(struct.pack("<Q", len(blob)))
        payload.extend(blob)
    return len(payload), hashlib.sha256(SOURCE_DOMAIN + payload).hexdigest()


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
        "hashlib",
        "importlib",
        "json",
        "os",
        "pathlib",
        "platform",
        "shutil",
        "stat",
        "struct",
        "subprocess",
        "sys",
        "threading",
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
    with pytest.raises(authority.Experiment002RunAuthorityError, match="not clean"):
        authority._verify_synthetic_repository_for_tests(case.root)


def test_ignored_extra_file_inside_import_root_is_rejected(tmp_path: Path) -> None:
    case = _valid_repository(tmp_path)
    (case.root / "src/falsewake/extra.ignored").write_text(
        "ignored\n", encoding="utf-8"
    )
    assert _git(case.root, "status", "--porcelain=v1", "-z") == b""
    with pytest.raises(
        authority.Experiment002RunAuthorityError,
        match="do not exactly cover",
    ):
        authority._verify_synthetic_repository_for_tests(case.root)


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

    def ambiguous(
        *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        del args, kwargs
        return subprocess.CompletedProcess(
            args=("git",),
            returncode=0,
            stdout=b"/first\n/second\n",
            stderr=b"",
        )

    monkeypatch.setattr(subprocess, "run", ambiguous)
    with pytest.raises(
        authority.Experiment002RunAuthorityError,
        match="ambiguous",
    ):
        authority._verify_synthetic_repository_for_tests(case.root)


def test_synthetic_root_rejects_path_subclasses_before_execution(
    tmp_path: Path,
) -> None:
    class DerivedPath(PosixPath):
        pass

    derived = DerivedPath(tmp_path)
    with pytest.raises(TypeError, match="exact pathlib path"):
        authority._verify_synthetic_repository_for_tests(derived)
