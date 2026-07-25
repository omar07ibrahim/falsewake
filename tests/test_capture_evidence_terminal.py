from __future__ import annotations

import ast
import functools
import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import cast

import pytest

TOOLS_DIRECTORY = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIRECTORY))

import capture_evidence_terminal as capture  # noqa: E402
import inspect_evidence as inspector  # noqa: E402


def _synthetic_transcript() -> bytes:
    lines = [f"synthetic evidence line {number:02d}" for number in range(1, 51)]
    lines[0] = "FalseWake tracked evidence inspection"
    lines[4] = "000 | MEASURED CLIP BASELINE"
    lines[5] = "  validation: x < y & y > z; label=\"measured\"; apostrophe='ok'"
    lines[9] = "001 | MEASURED DEVELOPMENT REPLAY | REJECT"
    lines[17] = "002-006 | DISPOSITIONS AND NON-CLAIMS"
    lines[39] = "PROVENANCE (repository-relative; SHA-256)"
    return ("\n".join(lines) + "\n").encode("ascii")


@functools.cache
def _actual_transcript() -> bytes:
    return inspector.render_human(inspector.build_inspection())


@pytest.fixture
def isolated_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Path:
    output = tmp_path / "docs" / "images" / "terminal"
    output.parent.mkdir(parents=True)
    monkeypatch.setattr(capture, "OUTPUT_DIRECTORY", output)

    def open_isolated_output(*, create: bool) -> int:
        if create:
            output.mkdir(mode=0o755, exist_ok=True)
        elif not output.is_dir():
            raise capture.TerminalEvidenceError("artifacts:directory")
        return capture._open_directory(output)

    def accept_synthetic_base(
        _: str,
        *,
        prior_manifest_sha256: str | None,
        snapshot: capture.InputSnapshot,
    ) -> None:
        del prior_manifest_sha256, snapshot

    monkeypatch.setattr(capture, "_open_output_directory", open_isolated_output)
    monkeypatch.setattr(capture, "_validate_base_state", accept_synthetic_base)
    return output


def _install_valid_artifacts(output: Path) -> dict[str, bytes]:
    snapshot = capture._snapshot_bound_inputs()
    base_commit = capture._repository_head()
    artifacts = capture._build_artifacts(
        _actual_transcript(),
        base_commit=base_commit,
        prior_manifest_sha256=capture._prior_manifest_digest(base_commit),
        snapshot=snapshot,
    )
    capture._write_artifacts(artifacts)
    assert output.is_dir()
    return artifacts


def _manifest(output: Path) -> dict[str, object]:
    value = json.loads((output / capture.MANIFEST_FILENAME).read_text("ascii"))
    assert type(value) is dict
    return cast(dict[str, object], value)


def _rewrite_manifest(output: Path, value: dict[str, object]) -> None:
    (output / capture.MANIFEST_FILENAME).write_bytes(capture._canonical_json(value))


def _mapping(value: object) -> dict[str, object]:
    assert type(value) is dict
    return cast(dict[str, object], value)


def _list(value: object) -> list[object]:
    assert type(value) is list
    return cast(list[object], value)


def test_cli_is_a_literal_two_operation_allowlist(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(capture, "_record", lambda: calls.append("record"))
    monkeypatch.setattr(
        capture,
        "_check_artifacts",
        lambda: calls.append("check"),
    )

    assert capture.main(["record"]) == 0
    assert capture.main(["check"]) == 0
    assert calls == ["record", "check"]

    for arguments in (
        [],
        ["--help"],
        ["record", "--output", "elsewhere"],
        ["check", "extra"],
        ["tools/inspect_evidence.py"],
    ):
        assert capture.main(arguments) == 2
    assert calls == ["record", "check"]
    assert "cli:arguments" in capsys.readouterr().err


def test_source_surface_has_no_experiment_import_or_dynamic_child_command() -> None:
    source = (TOOLS_DIRECTORY / "capture_evidence_terminal.py").read_text("utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any(
        name == "falsewake" or name.startswith("falsewake.") for name in imported
    )
    assert capture.SYMBOLIC_ARGV == (
        "{current-python}",
        "tools/inspect_evidence.py",
    )
    assert capture.OUTPUT_DIRECTORY.parts[-3:] == ("docs", "images", "terminal")
    assert capture.OUTPUT_DIRECTORY_PARTS == ("docs", "images", "terminal")
    assert {
        "evidence-inspection.txt",
        "evidence-metrics-frontiers.svg",
        "evidence-lineage-nonclaims.svg",
        "evidence-provenance.svg",
        "manifest.json",
    } == capture.EXPECTED_ARTIFACT_NAMES
    forbidden = (
        "experiment_002_runner",
        "experiment_003_coordinator",
        "experiment_004_run_authority",
        "experiment_005_supervisor",
        "experiment_006_runner",
        "falsewake-experiment-006-attempt",
    )
    assert all(token not in source for token in forbidden)


def test_inspector_subprocess_is_exactly_fixed_shell_free_and_scrubbed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(
        arguments: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        observed.append((arguments, kwargs))
        return subprocess.CompletedProcess(
            arguments,
            0,
            stdout=_synthetic_transcript(),
            stderr=b"",
        )

    monkeypatch.setenv("FALSEWAKE_SECRET_SENTINEL", "must-not-propagate")
    monkeypatch.setattr(subprocess, "run", fake_run)
    result = capture._run_inspector()

    assert result.returncode == 0
    assert len(observed) == 1
    arguments, kwargs = observed[0]
    assert arguments == [sys.executable, "tools/inspect_evidence.py"]
    assert kwargs["shell"] is False
    assert kwargs["check"] is False
    assert kwargs["capture_output"] is True
    assert kwargs["cwd"] == capture.REPOSITORY_ROOT
    assert kwargs["env"] == capture.CAPTURE_ENVIRONMENT
    assert "FALSEWAKE_SECRET_SENTINEL" not in cast(dict[str, str], kwargs["env"])


def test_record_invokes_inspector_once_and_rechecks_state(
    isolated_output: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    head = capture._repository_head()
    snapshot = capture._snapshot_bound_inputs()
    repository_checks: list[str] = []
    snapshots: list[capture.InputSnapshot] = []
    invocations = 0

    def clean_repository() -> str:
        repository_checks.append(head)
        return head

    def snapshot_inputs() -> capture.InputSnapshot:
        snapshots.append(snapshot)
        return snapshot

    def run_inspector() -> subprocess.CompletedProcess[bytes]:
        nonlocal invocations
        invocations += 1
        return subprocess.CompletedProcess(
            [sys.executable, capture.INSPECTOR_RELATIVE_PATH],
            0,
            stdout=_actual_transcript(),
            stderr=b"",
        )

    monkeypatch.setattr(capture, "_require_clean_repository", clean_repository)
    monkeypatch.setattr(capture, "_snapshot_bound_inputs", snapshot_inputs)
    monkeypatch.setattr(capture, "_run_inspector", run_inspector)
    capture._record()

    assert invocations == 1
    assert len(repository_checks) == 2
    assert len(snapshots) >= 3
    assert set(path.name for path in isolated_output.iterdir()) == (
        capture.EXPECTED_ARTIFACT_NAMES
    )


def test_record_rejects_dirty_tree_before_inspector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def dirty_status(
        arguments: tuple[str, ...],
    ) -> subprocess.CompletedProcess[bytes]:
        assert "status" in arguments
        return subprocess.CompletedProcess(
            ["git", *arguments],
            0,
            stdout=b"?? untracked.txt\x00",
            stderr=b"",
        )

    def forbidden_inspector() -> subprocess.CompletedProcess[bytes]:
        nonlocal calls
        calls += 1
        raise AssertionError("inspector must not run")

    monkeypatch.setattr(capture, "_run_git", dirty_status)
    monkeypatch.setattr(capture, "_run_inspector", forbidden_inspector)
    with pytest.raises(capture.TerminalEvidenceError, match="git:dirty"):
        capture._record()
    assert calls == 0


def test_record_fails_if_inputs_change_around_the_single_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = capture._snapshot_bound_inputs()
    changed = capture.InputSnapshot(
        original.recorder_byte_count,
        original.recorder_sha256,
        original.inspector_byte_count,
        "0" * 64,
        original.sources,
    )
    snapshots = iter((original, changed))
    writes = 0

    monkeypatch.setattr(capture, "_require_clean_repository", capture._repository_head)
    monkeypatch.setattr(capture, "_snapshot_bound_inputs", lambda: next(snapshots))
    monkeypatch.setattr(
        capture,
        "_run_inspector",
        lambda: subprocess.CompletedProcess(
            ["inspector"],
            0,
            stdout=_actual_transcript(),
            stderr=b"",
        ),
    )

    def forbidden_write(_: dict[str, bytes]) -> None:
        nonlocal writes
        writes += 1

    monkeypatch.setattr(capture, "_write_artifacts", forbidden_write)
    with pytest.raises(capture.TerminalEvidenceError, match="inputs_changed"):
        capture._record()
    assert writes == 0


def test_check_validates_complete_artifact_set_without_running_inspector(
    isolated_output: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_valid_artifacts(isolated_output)

    def forbidden() -> subprocess.CompletedProcess[bytes]:
        raise AssertionError("check must never execute the inspector")

    monkeypatch.setattr(capture, "_run_inspector", forbidden)
    capture._check_artifacts()
    assert capture.main(["check"]) == 0


@pytest.mark.parametrize(
    "mutation",
    [
        "schema",
        "source_hash",
        "inspector_hash",
        "line_range",
        "excerpt_hash",
        "output_hash",
        "python_version",
    ],
)
def test_check_rejects_canonical_manifest_tampering(
    isolated_output: Path,
    mutation: str,
) -> None:
    _install_valid_artifacts(isolated_output)
    manifest = _manifest(isolated_output)
    if mutation == "schema":
        manifest.pop("artifact_set")
    elif mutation == "source_hash":
        sources = _list(manifest["sources"])
        _mapping(sources[0])["sha256"] = "0" * 64
    elif mutation == "inspector_hash":
        _mapping(manifest["inspector"])["sha256"] = "0" * 64
    elif mutation == "line_range":
        panels = _list(manifest["panels"])
        _mapping(panels[0])["line_end"] = 15
    elif mutation == "excerpt_hash":
        panels = _list(manifest["panels"])
        _mapping(panels[0])["excerpt_sha256"] = "f" * 64
    elif mutation == "python_version":
        _mapping(manifest["capture"])["python_version"] = "9.9.9"
    else:
        _mapping(manifest["transcript"])["output_sha256"] = "e" * 64
    _rewrite_manifest(isolated_output, manifest)

    with pytest.raises(capture.TerminalEvidenceError, match="manifest"):
        capture._check_artifacts()


@pytest.mark.parametrize(
    ("contents", "error"),
    [
        (b'{"schema_version":1,"schema_version":1}\n', "duplicate_key"),
        (b'{"schema_version":NaN}\n', "nonfinite"),
        (b'{"schema_version":1}', "canonical"),
        (b"[]\n", "schema"),
    ],
)
def test_manifest_parser_rejects_duplicate_nonfinite_and_noncanonical_json(
    contents: bytes,
    error: str,
) -> None:
    with pytest.raises(capture.TerminalEvidenceError, match=error):
        capture._parse_manifest(contents)


def test_manifest_parser_bounds_size_and_normalizes_excessive_nesting() -> None:
    with pytest.raises(capture.TerminalEvidenceError, match="too_large"):
        capture._parse_manifest(b" " * (capture.MAX_MANIFEST_BYTE_COUNT + 1))
    deeply_nested = b"[" * 2_000 + b"0" + b"]" * 2_000 + b"\n"
    with pytest.raises(capture.TerminalEvidenceError, match="manifest"):
        capture._parse_manifest(deeply_nested)


@pytest.mark.parametrize("artifact", ["transcript", "panel"])
def test_check_rejects_artifact_hash_or_byte_tampering(
    isolated_output: Path,
    artifact: str,
) -> None:
    _install_valid_artifacts(isolated_output)
    path = (
        isolated_output / capture.TRANSCRIPT_FILENAME
        if artifact == "transcript"
        else isolated_output / capture.PANELS[0].filename
    )
    path.write_bytes(path.read_bytes() + b"tamper\n")

    with pytest.raises(capture.TerminalEvidenceError):
        capture._check_artifacts()


def test_hard_pinned_stdout_rejects_self_consistent_rebuilt_forgery(
    isolated_output: Path,
) -> None:
    _install_valid_artifacts(isolated_output)
    original = (isolated_output / capture.TRANSCRIPT_FILENAME).read_bytes()
    forged = original.replace(b"read-only", b"read-fake", 1)
    assert len(forged) == len(original)
    assert forged != original
    forged_lines = capture._structural_transcript_lines(forged)
    (isolated_output / capture.TRANSCRIPT_FILENAME).write_bytes(forged)

    manifest = _manifest(isolated_output)
    transcript_record = _mapping(manifest["transcript"])
    forged_digest = hashlib.sha256(forged).hexdigest()
    transcript_record["sha256"] = forged_digest
    transcript_record["output_sha256"] = forged_digest
    raw_panels = _list(manifest["panels"])
    for panel, raw_record in zip(capture.PANELS, raw_panels, strict=True):
        contents = capture._render_panel(forged_lines, panel)
        (isolated_output / panel.filename).write_bytes(contents)
        record = _mapping(raw_record)
        record["byte_count"] = len(contents)
        record["sha256"] = hashlib.sha256(contents).hexdigest()
        record["excerpt_sha256"] = hashlib.sha256(
            capture._excerpt_bytes(forged_lines, panel)
        ).hexdigest()
    _rewrite_manifest(isolated_output, manifest)

    with pytest.raises(capture.TerminalEvidenceError, match="transcript:sha256"):
        capture._check_artifacts()


def test_check_rejects_extra_file_and_symlink(
    isolated_output: Path,
) -> None:
    _install_valid_artifacts(isolated_output)
    extra = isolated_output / "unexpected.txt"
    extra.write_text("not admitted", encoding="ascii")
    with pytest.raises(capture.TerminalEvidenceError, match="extra_or_missing"):
        capture._check_artifacts()
    extra.unlink()
    (isolated_output / "unexpected-link").symlink_to(capture.TRANSCRIPT_FILENAME)
    with pytest.raises(capture.TerminalEvidenceError):
        capture._check_artifacts()


def test_descriptor_reader_rejects_symlink_fifo_and_mid_read_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "reader"
    directory.mkdir()
    regular = directory / "record.txt"
    regular.write_bytes(b"A" * 64)
    (directory / "link.txt").symlink_to("record.txt")
    os.mkfifo(directory / "pipe")
    directory_fd = capture._open_directory(directory)
    try:
        for filename in ("link.txt", "pipe"):
            with pytest.raises(capture.TerminalEvidenceError):
                capture._file_bytes_at(directory_fd, filename)

        original_read = os.read
        mutated = False

        def mutating_read(descriptor: int, length: int) -> bytes:
            nonlocal mutated
            contents = original_read(descriptor, length)
            if contents and not mutated:
                mutated = True
                regular.write_bytes(b"B" * 80)
            return contents

        monkeypatch.setattr(os, "read", mutating_read)
        with pytest.raises(capture.TerminalEvidenceError, match="changed_during_read"):
            capture._file_bytes_at(directory_fd, "record.txt")
    finally:
        os.close(directory_fd)


def test_output_walk_rejects_intermediate_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repository"
    outside = tmp_path / "outside"
    (root / "docs").mkdir(parents=True)
    outside.mkdir()
    (root / "docs" / "images").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(capture, "REPOSITORY_ROOT", root)

    with pytest.raises(capture.TerminalEvidenceError, match="artifacts:directory"):
        capture._open_output_directory(create=True)
    assert not (outside / "terminal").exists()


def test_svg_is_readable_exact_escaped_line_range() -> None:
    lines = capture._structural_transcript_lines(_synthetic_transcript())
    panel = capture.PANELS[0]
    contents = capture._render_panel(lines, panel)
    decoded = contents.decode("ascii")

    assert "DejaVu Sans Mono" in decoded
    assert 'width="1720"' in decoded
    assert "&lt;" in decoded
    assert "&gt;" in decoded
    assert "&amp;" in decoded
    assert "&quot;" in decoded
    assert "exact line-range excerpt" in decoded

    root = ET.fromstring(contents)
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    terminal_rows = [
        element
        for element in root.findall("svg:text", namespace)
        if element.attrib.get("class") == "terminal"
    ]
    excerpt = lines[panel.line_start - 1 : panel.line_end]
    assert len(terminal_rows) == len(excerpt)
    for row, expected in zip(terminal_rows, excerpt, strict=True):
        spans = row.findall("svg:tspan", namespace)
        assert len(spans) == 2
        assert (spans[1].text or "") == expected


def test_panel_specs_cover_only_the_three_truthful_sections() -> None:
    assert [(panel.line_start, panel.line_end) for panel in capture.PANELS] == [
        (1, 16),
        (18, 38),
        (40, 50),
    ]
    assert len({panel.panel_id for panel in capture.PANELS}) == 3
    assert len({panel.filename for panel in capture.PANELS}) == 3
    transcript = _synthetic_transcript()
    lines = capture._structural_transcript_lines(transcript)
    assert sum(
        len(capture._excerpt_bytes(lines, panel)) for panel in capture.PANELS
    ) < len(transcript)


def test_main_normalizes_errors_without_paths_or_tracebacks(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def failure() -> None:
        raise capture.TerminalEvidenceError("manifest:transcript")

    monkeypatch.setattr(capture, "_check_artifacts", failure)
    assert capture.main(["check"]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "terminal evidence failed: manifest:transcript\n"
    assert str(capture.REPOSITORY_ROOT) not in output.err


def test_capture_manifest_binds_every_source_and_every_artifact() -> None:
    snapshot = capture._snapshot_bound_inputs()
    base_commit = capture._repository_head()
    artifacts = capture._build_artifacts(
        _actual_transcript(),
        base_commit=base_commit,
        prior_manifest_sha256=capture._prior_manifest_digest(base_commit),
        snapshot=snapshot,
    )
    manifest = capture._parse_manifest(artifacts[capture.MANIFEST_FILENAME])
    sources = _list(manifest["sources"])
    panels = _list(manifest["panels"])

    assert len(sources) == 10
    assert {_mapping(source)["path"] for source in sources} == {
        source.relative_path for source in capture.EVIDENCE_SOURCES
    }
    assert all(_mapping(source)["byte_count"] for source in sources)
    assert all(len(cast(str, _mapping(source)["sha256"])) == 64 for source in sources)
    assert len(panels) == 3
    assert all(_mapping(panel)["line_start"] for panel in panels)
    assert all(_mapping(panel)["line_end"] for panel in panels)
    assert all(len(cast(str, _mapping(panel)["sha256"])) == 64 for panel in panels)
    assert all(
        len(cast(str, _mapping(panel)["excerpt_sha256"])) == 64 for panel in panels
    )


def test_base_state_binds_recorder_inputs_and_prior_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = capture._snapshot_bound_inputs()
    blobs = {
        capture.RECORDER_RELATIVE_PATH: capture._repository_file_bytes(
            capture.RECORDER_RELATIVE_PATH
        ),
        capture.INSPECTOR_RELATIVE_PATH: capture._repository_file_bytes(
            capture.INSPECTOR_RELATIVE_PATH
        ),
        **{
            source.relative_path: capture._repository_file_bytes(source.relative_path)
            for source in snapshot.sources
        },
    }
    good = "a" * 40
    missing_recorder = "b" * 40
    changed_prior = "c" * 40
    monkeypatch.setattr(capture, "_base_is_ancestor", lambda _: None)

    def fake_blob(commit: str, relative_path: str) -> bytes | None:
        if relative_path == capture.MANIFEST_RELATIVE_PATH:
            return b"prior manifest\n" if commit == changed_prior else None
        if (
            commit == missing_recorder
            and relative_path == capture.RECORDER_RELATIVE_PATH
        ):
            return None
        return blobs[relative_path]

    monkeypatch.setattr(capture, "_git_blob_at_commit", fake_blob)
    capture._validate_base_state(
        good,
        prior_manifest_sha256=None,
        snapshot=snapshot,
    )
    with pytest.raises(capture.TerminalEvidenceError, match="base_inputs"):
        capture._validate_base_state(
            missing_recorder,
            prior_manifest_sha256=None,
            snapshot=snapshot,
        )
    with pytest.raises(capture.TerminalEvidenceError, match="base_prior_manifest"):
        capture._validate_base_state(
            changed_prior,
            prior_manifest_sha256=None,
            snapshot=snapshot,
        )
