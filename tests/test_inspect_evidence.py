from __future__ import annotations

import ast
import errno
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

TOOLS_DIRECTORY = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIRECTORY))

import inspect_evidence as evidence  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
COMMAND = [sys.executable, str(TOOLS_DIRECTORY / "inspect_evidence.py")]
SYNTHETIC_CONTENTS = b'{"evidence":"original"}\n'


def _mapping(value: object) -> dict[str, object]:
    assert type(value) is dict
    raw = cast(dict[object, object], value)
    assert all(type(key) is str for key in raw)
    return cast(dict[str, object], raw)


def _list(value: object) -> list[object]:
    assert type(value) is list
    return cast(list[object], value)


def _run(*arguments: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [*COMMAND, *arguments],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        env={
            "LC_ALL": "C",
            "PATH": os.environ.get("PATH", ""),
            "PYTHONHASHSEED": "0",
            "TZ": "UTC",
        },
    )


def _synthetic_spec(
    *,
    directories: tuple[str, ...] = ("evidence",),
    filename: str = "record.json",
    contents: bytes = SYNTHETIC_CONTENTS,
    byte_count: int | None = None,
    sha256: str | None = None,
) -> evidence.SourceSpec:
    return evidence.SourceSpec(
        source_id="synthetic",
        directories=directories,
        filename=filename,
        byte_count=len(contents) if byte_count is None else byte_count,
        sha256=hashlib.sha256(contents).hexdigest() if sha256 is None else sha256,
        role="adversarial test fixture",
    )


def _open_test_root(path: Path) -> int:
    return os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY)


def _assert_descriptor_closed(descriptor: int) -> None:
    with pytest.raises(OSError) as error:
        os.fstat(descriptor)
    assert error.value.errno == errno.EBADF


def test_json_output_is_exact_deterministic_and_evidence_bound() -> None:
    first = _run("--json")
    second = _run("--json")

    assert first.returncode == second.returncode == 0
    assert first.stderr == second.stderr == b""
    assert first.stdout == second.stdout
    assert hashlib.sha256(first.stdout).hexdigest() == (
        "69c59643071781105c378b3fb3c0b1014b3108550fa86b40eebfb7d03d37579b"
    )
    assert first.stdout.endswith(b"\n")
    assert first.stdout.count(b"\n") == 1
    assert b"/home/" not in first.stdout
    assert str(REPOSITORY_ROOT).encode() not in first.stdout
    assert b"falsewake-experiment-006-attempt" not in first.stdout

    payload = _mapping(json.loads(first.stdout))
    assert payload["schema_version"] == 1
    inspection = _mapping(payload["inspection"])
    assert inspection == {
        "experiment_code_imported": False,
        "mode": "read_only_hash_pinned",
        "scientific_metrics_recomputed": False,
        "source_count": 10,
    }
    experiments = {
        cast(str, row["experiment"]): row
        for raw_row in _list(payload["experiments"])
        if (row := _mapping(raw_row))
    }
    assert set(experiments) == {"000", "001", "002", "003", "004", "005", "006"}
    assert experiments["001"]["disposition"] == "reject"
    assert experiments["001"]["selected_threshold_milli"] is None
    assert experiments["002"]["outcome_code"] == "seed_selection_failed"
    assert experiments["004"]["disposition"] == "preflight_rejected"
    assert experiments["005"]["disposition"] == "preflight_rejected"
    assert experiments["006"]["root_cause_status"] == (
        "undetermined_from_retained_evidence"
    )


def test_human_output_is_exact_path_safe_and_deterministic() -> None:
    first = _run()
    second = _run()

    assert first.returncode == second.returncode == 0
    assert first.stderr == second.stderr == b""
    assert first.stdout == second.stdout
    assert hashlib.sha256(first.stdout).hexdigest() == (
        "61e2d8915443fbda039a79decc11f2fb2c8c93d41100688e59e6e2288e9a1584"
    )
    assert first.stdout.startswith(b"FalseWake tracked evidence inspection\n")
    assert b"000 | MEASURED CLIP BASELINE\n" in first.stdout
    assert b"001 | MEASURED DEVELOPMENT REPLAY | REJECT\n" in first.stdout
    assert b"002-006 | DISPOSITIONS AND NON-CLAIMS\n" in first.stdout
    assert b"/home/" not in first.stdout
    assert str(REPOSITORY_ROOT).encode() not in first.stdout
    assert b"falsewake-experiment-006-attempt" not in first.stdout


def test_source_surface_is_closed_hash_pinned_and_regular_file_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_paths = {
        "reports/experiment-000-linear.json",
        "reports/experiment-001-dev-replay.json",
        "reports/experiment-001-selection.json",
        "reports/experiment-002-training.json",
        "reports/experiment-002-execution-incident.json",
        "reports/experiment-003-execution-incident.json",
        "reports/experiment-004-preflight-incident.json",
        "reports/experiment-005-preflight-incident.json",
        "reports/experiment-006-execution-incident.json",
        "docs/images/readme/provenance.json",
    }
    assert {spec.relative_path for spec in evidence.SOURCE_SPECS} == expected_paths
    assert len({spec.source_id for spec in evidence.SOURCE_SPECS}) == 10
    assert all(len(spec.sha256) == 64 for spec in evidence.SOURCE_SPECS)
    assert all(not path.startswith("/") and ".." not in path for path in expected_paths)

    observed: list[str] = []
    original = evidence._read_bound_source

    def recording_reader(root_fd: int, spec: evidence.SourceSpec) -> bytes:
        observed.append(spec.relative_path)
        return original(root_fd, spec)

    monkeypatch.setattr(evidence, "_read_bound_source", recording_reader)
    modules_before = frozenset(sys.modules)
    evidence.build_inspection()
    assert observed == [spec.relative_path for spec in evidence.SOURCE_SPECS]
    assert not [
        module
        for module in frozenset(sys.modules) - modules_before
        if module == "falsewake" or module.startswith("falsewake.")
    ]

    observed_flags: list[int] = []
    original_open = os.open

    def recording_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o600,
        *,
        dir_fd: int | None = None,
    ) -> int:
        observed_flags.append(flags)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", recording_open)
    evidence.build_inspection()
    write_flags = (
        os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_EXCL | os.O_TRUNC
    )
    assert observed_flags
    assert all(flags & write_flags == 0 for flags in observed_flags)
    assert all(flags & os.O_NOFOLLOW for flags in observed_flags)


def test_ast_surface_has_no_project_import_execution_or_write_primitive() -> None:
    source = (TOOLS_DIRECTORY / "inspect_evidence.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)

    assert not any(
        module == "falsewake" or module.startswith("falsewake.")
        for module in imported_modules
    )
    assert imported_modules.isdisjoint(
        {
            "asyncio",
            "multiprocessing",
            "runpy",
            "shutil",
            "subprocess",
        }
    )
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "os"
    ]
    called_os_members = {cast(ast.Attribute, call.func).attr for call in calls}
    assert called_os_members <= {"close", "fstat", "open", "read", "write"}
    assert (
        not {
            "chmod",
            "link",
            "mkdir",
            "remove",
            "rename",
            "replace",
            "symlink",
            "unlink",
        }
        & called_os_members
    )
    forbidden_tokens = (
        "experiment_002_runner",
        "experiment_003_coordinator",
        "experiment_004_run_authority",
        "experiment_005_supervisor",
        "experiment_006_runner",
        "falsewake-experiment-006-attempt",
        ".git",
    )
    assert not any(token in source for token in forbidden_tokens)


def test_schema_helpers_and_hash_gate_fail_closed() -> None:
    with pytest.raises(evidence.EvidenceInspectionError, match="duplicate_key"):
        evidence._parse_json(b'{"a":1,"a":2}\n', source_id="synthetic")
    with pytest.raises(evidence.EvidenceInspectionError, match="nonfinite"):
        evidence._parse_json(b'{"a":NaN}\n', source_id="synthetic")
    with pytest.raises(evidence.EvidenceInspectionError, match="schema:000:keys"):
        evidence._validate_experiment_000({"schema_version": 1})

    spec = evidence.SOURCE_SPECS[0]
    root_fd = os.open(REPOSITORY_ROOT, os.O_RDONLY | os.O_DIRECTORY)
    try:
        altered = evidence.SourceSpec(
            source_id=spec.source_id,
            directories=spec.directories,
            filename=spec.filename,
            byte_count=spec.byte_count,
            sha256="0" * 64,
            role=spec.role,
        )
        with pytest.raises(evidence.EvidenceInspectionError, match="sha256"):
            evidence._read_bound_source(root_fd, altered)
    finally:
        os.close(root_fd)


def test_symlink_directories_and_files_are_rejected(
    tmp_path: Path,
) -> None:
    real_directory = tmp_path / "real"
    real_directory.mkdir()
    (real_directory / "record.json").write_bytes(SYNTHETIC_CONTENTS)
    (tmp_path / "directory-link").symlink_to(real_directory, target_is_directory=True)

    root_fd = _open_test_root(tmp_path)
    try:
        with pytest.raises(evidence.EvidenceInspectionError, match="directory_open"):
            evidence._read_bound_source(
                root_fd,
                _synthetic_spec(directories=("directory-link",)),
            )
    finally:
        os.close(root_fd)

    admitted_directory = tmp_path / "admitted"
    admitted_directory.mkdir()
    (admitted_directory / "record.json").symlink_to(real_directory / "record.json")
    root_fd = _open_test_root(tmp_path)
    try:
        with pytest.raises(evidence.EvidenceInspectionError, match="file_open"):
            evidence._read_bound_source(
                root_fd,
                _synthetic_spec(directories=("admitted",)),
            )
    finally:
        os.close(root_fd)


def test_fifo_is_rejected_as_non_regular_without_blocking(tmp_path: Path) -> None:
    directory = tmp_path / "evidence"
    directory.mkdir()
    os.mkfifo(directory / "record.json", mode=0o600)

    root_fd = _open_test_root(tmp_path)
    try:
        with pytest.raises(evidence.EvidenceInspectionError, match="file_identity"):
            evidence._read_bound_source(root_fd, _synthetic_spec())
    finally:
        os.close(root_fd)


def test_path_replacement_after_open_never_supplies_replacement_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = b'{"value":"aaaa"}\n'
    replacement = b'{"value":"bbbb"}\n'
    directory = tmp_path / "evidence"
    directory.mkdir()
    (directory / "record.json").write_bytes(original)
    replacement_path = directory / "replacement-ready.json"
    replacement_path.write_bytes(replacement)
    root_fd = _open_test_root(tmp_path)
    original_open = os.open
    replaced = False

    def replacing_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o600,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replaced
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if path == "record.json" and dir_fd is not None and not replaced:
            replaced = True
            os.replace(replacement_path, directory / "record.json")
        return descriptor

    monkeypatch.setattr(os, "open", replacing_open)
    try:
        with pytest.raises(evidence.EvidenceInspectionError, match="file_identity"):
            evidence._read_bound_source(
                root_fd,
                _synthetic_spec(contents=original),
            )
    finally:
        os.close(root_fd)
    assert replaced
    assert (directory / "record.json").read_bytes() == replacement


def test_identity_change_during_read_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "evidence"
    directory.mkdir()
    (directory / "record.json").write_bytes(SYNTHETIC_CONTENTS)
    root_fd = _open_test_root(tmp_path)
    original_identity = evidence._stat_identity
    identity_calls = 0

    def changing_identity(metadata: os.stat_result) -> tuple[int, ...]:
        nonlocal identity_calls
        identity_calls += 1
        identity = original_identity(metadata)
        if identity_calls == 2:
            return (*identity[:-1], identity[-1] + 1)
        return identity

    monkeypatch.setattr(evidence, "_stat_identity", changing_identity)
    try:
        with pytest.raises(
            evidence.EvidenceInspectionError, match="changed_during_read"
        ):
            evidence._read_bound_source(root_fd, _synthetic_spec())
    finally:
        os.close(root_fd)
    assert identity_calls == 2


def test_wrong_registered_byte_count_fails_closed(tmp_path: Path) -> None:
    directory = tmp_path / "evidence"
    directory.mkdir()
    (directory / "record.json").write_bytes(SYNTHETIC_CONTENTS)
    root_fd = _open_test_root(tmp_path)
    try:
        with pytest.raises(evidence.EvidenceInspectionError, match="file_identity"):
            evidence._read_bound_source(
                root_fd,
                _synthetic_spec(byte_count=len(SYNTHETIC_CONTENTS) + 1),
            )
    finally:
        os.close(root_fd)


def test_short_read_fails_the_post_read_byte_count_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "evidence"
    directory.mkdir()
    (directory / "record.json").write_bytes(SYNTHETIC_CONTENTS)
    root_fd = _open_test_root(tmp_path)
    original_read = os.read
    shortened = False

    def short_read(descriptor: int, count: int) -> bytes:
        nonlocal shortened
        chunk = original_read(descriptor, count)
        if chunk and not shortened:
            shortened = True
            return chunk[:-1]
        return chunk

    monkeypatch.setattr(os, "read", short_read)
    try:
        with pytest.raises(evidence.EvidenceInspectionError, match="byte_count"):
            evidence._read_bound_source(root_fd, _synthetic_spec())
    finally:
        os.close(root_fd)
    assert shortened


def test_all_opened_descriptors_close_after_late_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "evidence"
    directory.mkdir()
    (directory / "record.json").write_bytes(SYNTHETIC_CONTENTS)
    root_fd = _open_test_root(tmp_path)
    original_open = os.open
    opened: list[int] = []

    def recording_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o600,
        *,
        dir_fd: int | None = None,
    ) -> int:
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        opened.append(descriptor)
        return descriptor

    monkeypatch.setattr(os, "open", recording_open)
    try:
        with pytest.raises(evidence.EvidenceInspectionError, match="sha256"):
            evidence._read_bound_source(
                root_fd,
                _synthetic_spec(sha256="0" * 64),
            )
    finally:
        os.close(root_fd)
    assert len(opened) == 2
    for descriptor in opened:
        _assert_descriptor_closed(descriptor)


def test_opened_directory_closes_after_nested_symlink_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outer = tmp_path / "outer"
    outer.mkdir()
    real = tmp_path / "real"
    real.mkdir()
    (outer / "nested-link").symlink_to(real, target_is_directory=True)
    root_fd = _open_test_root(tmp_path)
    original_open = os.open
    opened: list[int] = []

    def recording_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o600,
        *,
        dir_fd: int | None = None,
    ) -> int:
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        opened.append(descriptor)
        return descriptor

    monkeypatch.setattr(os, "open", recording_open)
    try:
        with pytest.raises(evidence.EvidenceInspectionError, match="directory_open"):
            evidence._read_bound_source(
                root_fd,
                _synthetic_spec(directories=("outer", "nested-link")),
            )
    finally:
        os.close(root_fd)
    assert len(opened) == 1
    _assert_descriptor_closed(opened[0])


def test_missing_no_follow_support_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(os, "O_NOFOLLOW")
    with pytest.raises(
        evidence.EvidenceInspectionError, match="o_nofollow_unavailable"
    ):
        evidence._directory_flags()
    with pytest.raises(
        evidence.EvidenceInspectionError, match="o_nofollow_unavailable"
    ):
        evidence._file_flags()


def test_cli_rejects_all_path_and_unknown_arguments_without_reading_sources() -> None:
    for argument in (
        "--repo-root=.",
        "--report=reports/experiment-000-linear.json",
        "--output=-",
        "reports/experiment-001-dev-replay.json",
        "--unknown",
    ):
        result = _run(argument)
        assert result.returncode == 2
        assert result.stdout == b""
        assert result.stderr == b"evidence inspection failed: cli:arguments\n"
        assert b"/home/" not in result.stderr

    help_result = _run("--help")
    assert help_result.returncode == 0
    assert help_result.stderr == b""
    assert help_result.stdout.startswith(b"usage: inspect_evidence.py")
    assert b"/home/" not in help_result.stdout
    assert str(REPOSITORY_ROOT).encode() not in help_result.stdout
