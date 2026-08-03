"""Capture reproducible terminal evidence from the closed evidence inspector.

The public interface is deliberately literal: ``record`` captures one invocation
of ``tools/inspect_evidence.py`` and ``check`` validates committed artifacts
without invoking it.  No CLI-provided child command, path, option, or inherited
environment value reaches the inspector subprocess.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Final, NoReturn, cast


class TerminalEvidenceError(ValueError):
    """The closed capture or its artifact set failed validation."""


@dataclass(frozen=True, slots=True)
class BoundFile:
    """One immutable input admitted to the capture."""

    source_id: str
    relative_path: str
    byte_count: int
    sha256: str


@dataclass(frozen=True, slots=True)
class Panel:
    """A fixed, inclusive transcript range rendered as one SVG."""

    panel_id: str
    filename: str
    title: str
    line_start: int
    line_end: int


@dataclass(frozen=True, slots=True)
class InputSnapshot:
    """Hashes observed immediately around the inspector invocation."""

    recorder_byte_count: int
    recorder_sha256: str
    inspector_byte_count: int
    inspector_sha256: str
    sources: tuple[BoundFile, ...]


REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]
INSPECTOR_PATH: Final = REPOSITORY_ROOT / "tools" / "inspect_evidence.py"
OUTPUT_DIRECTORY: Final = REPOSITORY_ROOT / "docs" / "images" / "terminal"
OUTPUT_DIRECTORY_PARTS: Final = ("docs", "images", "terminal")

RECORDER_RELATIVE_PATH: Final = "tools/capture_evidence_terminal.py"
INSPECTOR_RELATIVE_PATH: Final = "tools/inspect_evidence.py"
INSPECTOR_BYTE_COUNT: Final = 47_470
INSPECTOR_SHA256: Final = (
    "e6f74a4ed3443a412b237300b7d01fd3242d0b4974eeaa76832ed21e25d2620a"
)
TRANSCRIPT_FILENAME: Final = "evidence-inspection.txt"
MANIFEST_FILENAME: Final = "manifest.json"
MANIFEST_RELATIVE_PATH: Final = "docs/images/terminal/manifest.json"
DISPLAY_COMMAND: Final = "python3 tools/inspect_evidence.py"
SYMBOLIC_ARGV: Final = ("{current-python}", INSPECTOR_RELATIVE_PATH)
PYTHON_VERSION: Final = "3.12.3"
EXPECTED_TRANSCRIPT_BYTE_COUNT: Final = 3_675
EXPECTED_TRANSCRIPT_SHA256: Final = (
    "813d6d8a0e3eea8882caa4af8668f92da0a00dfcacad351e3b6c81e318d42491"
)
MAX_MANIFEST_BYTE_COUNT: Final = 65_536
MAX_TRANSCRIPT_BYTE_COUNT: Final = 16_384
MAX_PANEL_BYTE_COUNT: Final = 65_536

EVIDENCE_SOURCES: Final = (
    BoundFile(
        "experiment_000_report",
        "reports/experiment-000-linear.json",
        20_042,
        "3ad2e6d630bc62810bb7a88418dd018cd0d7acd65b4173caa912af1fe33fcaf9",
    ),
    BoundFile(
        "experiment_001_replay",
        "reports/experiment-001-dev-replay.json",
        1_456_860,
        "b8e30e26498af2600ece01f4cceb436e10a546b8390f039ca8a23cda45f00f2d",
    ),
    BoundFile(
        "experiment_001_selection",
        "reports/experiment-001-selection.json",
        828,
        "1d44ae6ff06a5fab1567d0342299e293fe001b8c91f9972cfa8e79a2dabf2318",
    ),
    BoundFile(
        "experiment_002_terminal_report",
        "reports/experiment-002-training.json",
        637,
        "494336d12e47f7bce952251e4c079770920f57032354e9618050250ee32bb99a",
    ),
    BoundFile(
        "experiment_002_incident",
        "reports/experiment-002-execution-incident.json",
        2_682,
        "353e33acc156e6e3a598d3dd777afbe5b15125b1b24f7f48266859f09f297b59",
    ),
    BoundFile(
        "experiment_003_incident",
        "reports/experiment-003-execution-incident.json",
        4_539,
        "0dc24fd211129cd2b81e29fb91ac9e66a0aad25f1deeae332f3b6ea420567688",
    ),
    BoundFile(
        "experiment_004_incident",
        "reports/experiment-004-preflight-incident.json",
        6_431,
        "d015b749acf6e98e87172372315c78ccfaad48b8ae6b3a29ae426890aa467f50",
    ),
    BoundFile(
        "experiment_005_incident",
        "reports/experiment-005-preflight-incident.json",
        6_612,
        "3de6fbbd6912bee47f3ef01f5371b57ae8d5942b6fc9915fc0ec243b9e931e3d",
    ),
    BoundFile(
        "experiment_006_incident",
        "reports/experiment-006-execution-incident.json",
        8_839,
        "c5cf002ea876890e3b3769af2bcc850433d3d08ac1071c864ed69ee2172af77d",
    ),
    BoundFile(
        "visual_provenance",
        "docs/images/readme/provenance.json",
        21_186,
        "5b32752075839e8218f161c7c681b753fde69a596ed1bde34300d8f1aa0fbdd7",
    ),
)

PANELS: Final = (
    Panel(
        "metrics_frontiers",
        "evidence-metrics-frontiers.svg",
        "Measured metrics and development frontier",
        1,
        16,
    ),
    Panel(
        "lineage_nonclaims",
        "evidence-lineage-nonclaims.svg",
        "Experiments 002-006: dispositions and non-claims",
        18,
        38,
    ),
    Panel(
        "provenance",
        "evidence-provenance.svg",
        "Tracked evidence provenance",
        40,
        50,
    ),
)

EXPECTED_ARTIFACT_NAMES: Final = frozenset(
    {
        TRANSCRIPT_FILENAME,
        MANIFEST_FILENAME,
        *(panel.filename for panel in PANELS),
    }
)
CAPTURE_ENVIRONMENT: Final = {
    "HOME": "/nonexistent",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "TMPDIR": "/nonexistent",
    "TZ": "UTC",
}
MANIFEST_ENVIRONMENT: Final = {
    key: ("<isolated>" if key in {"HOME", "TMPDIR"} else value)
    for key, value in CAPTURE_ENVIRONMENT.items()
}
GIT_ENVIRONMENT: Final = {
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "HOME": "/nonexistent",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": os.defpath,
}
_COMMIT_PATTERN: Final = re.compile(r"[0-9a-f]{40}\Z")


def _fail(code: str) -> NoReturn:
    raise TerminalEvidenceError(code)


def _sha256(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _descriptor_bytes(
    descriptor: int,
    *,
    expected_size: int | None = None,
    maximum_size: int | None = None,
) -> bytes:
    try:
        before = os.fstat(descriptor)
    except OSError:
        _fail("input:stat")
    if not stat.S_ISREG(before.st_mode):
        _fail("input:not_regular")
    if expected_size is not None and before.st_size != expected_size:
        _fail("input:byte_count")
    if maximum_size is not None and before.st_size > maximum_size:
        _fail("input:too_large")
    chunks: list[bytes] = []
    remaining = before.st_size
    try:
        while remaining:
            chunk = os.read(descriptor, min(remaining, 65_536))
            if not chunk:
                _fail("input:changed_during_read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            _fail("input:changed_during_read")
        after = os.fstat(descriptor)
    except OSError:
        _fail("input:read")
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_after != identity_before:
        _fail("input:changed_during_read")
    return b"".join(chunks)


def _open_directory(path: Path) -> int:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
    except OSError:
        _fail("input:directory")
    try:
        metadata = os.fstat(descriptor)
    except OSError:
        with suppress(OSError):
            os.close(descriptor)
        _fail("input:directory")
    if not stat.S_ISDIR(metadata.st_mode):
        with suppress(OSError):
            os.close(descriptor)
        _fail("input:directory")
    return descriptor


def _open_output_directory(*, create: bool) -> int:
    directory_fd = _open_directory(REPOSITORY_ROOT)
    try:
        for index, component in enumerate(OUTPUT_DIRECTORY_PARTS):
            try:
                next_fd = os.open(
                    component,
                    os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=directory_fd,
                )
            except FileNotFoundError:
                if not create or index != len(OUTPUT_DIRECTORY_PARTS) - 1:
                    _fail("artifacts:directory")
                try:
                    os.mkdir(component, mode=0o755, dir_fd=directory_fd)
                    next_fd = os.open(
                        component,
                        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=directory_fd,
                    )
                except OSError:
                    _fail("artifacts:create_directory")
            except OSError:
                _fail("artifacts:directory")
            os.close(directory_fd)
            directory_fd = next_fd
        return directory_fd
    except TerminalEvidenceError:
        with suppress(OSError):
            os.close(directory_fd)
        raise


def _file_bytes_at(
    directory_fd: int,
    filename: str,
    *,
    expected_size: int | None = None,
    maximum_size: int | None = None,
) -> bytes:
    if not filename or filename in {".", ".."} or "/" in filename:
        _fail("input:filename")
    try:
        descriptor = os.open(
            filename,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=directory_fd,
        )
    except OSError:
        _fail("input:open")
    try:
        return _descriptor_bytes(
            descriptor,
            expected_size=expected_size,
            maximum_size=maximum_size,
        )
    finally:
        with suppress(OSError):
            os.close(descriptor)


def _repository_file_bytes(
    relative_path: str,
    *,
    expected_size: int | None = None,
) -> bytes:
    parts = tuple(relative_path.split("/"))
    if (
        not parts
        or any(not part or part in {".", ".."} for part in parts)
        or relative_path.startswith("/")
    ):
        _fail("input:path")
    directory_fd = _open_directory(REPOSITORY_ROOT)
    try:
        for component in parts[:-1]:
            try:
                next_fd = os.open(
                    component,
                    os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=directory_fd,
                )
            except OSError:
                _fail("input:path")
            os.close(directory_fd)
            directory_fd = next_fd
        return _file_bytes_at(
            directory_fd,
            parts[-1],
            expected_size=expected_size,
        )
    finally:
        with suppress(OSError):
            os.close(directory_fd)


def _observe_bound_file(bound: BoundFile) -> BoundFile:
    contents = _repository_file_bytes(
        bound.relative_path,
        expected_size=bound.byte_count,
    )
    digest = _sha256(contents)
    if digest != bound.sha256:
        _fail(f"input:{bound.source_id}:sha256")
    return BoundFile(
        bound.source_id,
        bound.relative_path,
        len(contents),
        digest,
    )


def _snapshot_bound_inputs() -> InputSnapshot:
    recorder = _repository_file_bytes(RECORDER_RELATIVE_PATH)
    inspector = _repository_file_bytes(
        INSPECTOR_RELATIVE_PATH,
        expected_size=INSPECTOR_BYTE_COUNT,
    )
    inspector_digest = _sha256(inspector)
    if inspector_digest != INSPECTOR_SHA256:
        _fail("input:inspector:sha256")
    return InputSnapshot(
        recorder_byte_count=len(recorder),
        recorder_sha256=_sha256(recorder),
        inspector_byte_count=len(inspector),
        inspector_sha256=inspector_digest,
        sources=tuple(_observe_bound_file(source) for source in EVIDENCE_SOURCES),
    )


def _run_git(arguments: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=REPOSITORY_ROOT,
            env=GIT_ENVIRONMENT,
            shell=False,
            check=False,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        _fail("git:execution")


def _repository_head() -> str:
    result = _run_git(("rev-parse", "--verify", "HEAD"))
    if result.returncode != 0 or result.stderr:
        _fail("git:head")
    try:
        head = result.stdout.decode("ascii").strip()
    except UnicodeDecodeError:
        _fail("git:head")
    if _COMMIT_PATTERN.fullmatch(head) is None:
        _fail("git:head")
    return head


def _require_clean_repository() -> str:
    status = _run_git(
        (
            "-c",
            "core.fsmonitor=false",
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--no-renames",
        )
    )
    if status.returncode != 0 or status.stderr:
        _fail("git:status")
    if status.stdout:
        _fail("git:dirty")
    return _repository_head()


def _base_is_ancestor(base_commit: str) -> None:
    result = _run_git(("merge-base", "--is-ancestor", base_commit, _repository_head()))
    if result.returncode != 0 or result.stdout or result.stderr:
        _fail("manifest:base_commit")


def _git_blob_at_commit(commit: str, relative_path: str) -> bytes | None:
    admitted_paths = {
        RECORDER_RELATIVE_PATH,
        INSPECTOR_RELATIVE_PATH,
        MANIFEST_RELATIVE_PATH,
        *(source.relative_path for source in EVIDENCE_SOURCES),
    }
    if _COMMIT_PATTERN.fullmatch(commit) is None or relative_path not in admitted_paths:
        _fail("git:blob_arguments")
    listing = _run_git(("ls-tree", "-z", "--full-tree", commit, "--", relative_path))
    if listing.returncode != 0 or listing.stderr:
        _fail("git:blob_listing")
    if not listing.stdout:
        return None
    if listing.stdout.count(b"\x00") != 1:
        _fail("git:blob_listing")
    result = _run_git(("show", f"{commit}:{relative_path}"))
    if result.returncode != 0 or result.stderr:
        _fail("git:blob_read")
    return result.stdout


def _prior_manifest_digest(base_commit: str) -> str | None:
    prior_manifest = _git_blob_at_commit(base_commit, MANIFEST_RELATIVE_PATH)
    return None if prior_manifest is None else _sha256(prior_manifest)


def _validate_base_state(
    base_commit: str,
    *,
    prior_manifest_sha256: str | None,
    snapshot: InputSnapshot,
) -> None:
    _base_is_ancestor(base_commit)
    expected_blobs = (
        (
            RECORDER_RELATIVE_PATH,
            snapshot.recorder_byte_count,
            snapshot.recorder_sha256,
        ),
        (
            INSPECTOR_RELATIVE_PATH,
            snapshot.inspector_byte_count,
            snapshot.inspector_sha256,
        ),
        *(
            (source.relative_path, source.byte_count, source.sha256)
            for source in snapshot.sources
        ),
    )
    for relative_path, byte_count, digest in expected_blobs:
        contents = _git_blob_at_commit(base_commit, relative_path)
        if (
            contents is None
            or len(contents) != byte_count
            or _sha256(contents) != digest
        ):
            _fail("manifest:base_inputs")
    if _prior_manifest_digest(base_commit) != prior_manifest_sha256:
        _fail("manifest:base_prior_manifest")


def _run_inspector() -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            [sys.executable, INSPECTOR_RELATIVE_PATH],
            cwd=REPOSITORY_ROOT,
            env=CAPTURE_ENVIRONMENT,
            shell=False,
            check=False,
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        _fail("capture:execution")


def _structural_transcript_lines(transcript: bytes) -> list[str]:
    if not transcript or not transcript.endswith(b"\n"):
        _fail("transcript:final_newline")
    if b"\r" in transcript or b"\x1b" in transcript or b"\x00" in transcript:
        _fail("transcript:control")
    try:
        decoded = transcript.decode("ascii")
    except UnicodeDecodeError:
        _fail("transcript:ascii")
    lines = decoded[:-1].split("\n")
    if len(lines) != 50:
        _fail("transcript:line_count")
    expected_markers = {
        1: "FalseWake tracked evidence inspection",
        5: "000 | MEASURED CLIP BASELINE",
        10: "001 | MEASURED DEVELOPMENT REPLAY | REJECT",
        18: "002-006 | DISPOSITIONS AND NON-CLAIMS",
        40: "PROVENANCE (repository-relative; SHA-256)",
    }
    if any(lines[line - 1] != marker for line, marker in expected_markers.items()):
        _fail("transcript:structure")
    forbidden = (
        "/home/",
        "/Users/",
        "AKIA",
        "BEGIN PRIVATE KEY",
        "github_pat_",
        "ghp_",
    )
    if any(token in decoded for token in forbidden):
        _fail("transcript:sensitive")
    return lines


def _transcript_lines(transcript: bytes) -> list[str]:
    if len(transcript) != EXPECTED_TRANSCRIPT_BYTE_COUNT:
        _fail("transcript:byte_count")
    if _sha256(transcript) != EXPECTED_TRANSCRIPT_SHA256:
        _fail("transcript:sha256")
    return _structural_transcript_lines(transcript)


def _excerpt_bytes(lines: Sequence[str], panel: Panel) -> bytes:
    if not 1 <= panel.line_start <= panel.line_end <= len(lines):
        _fail("panel:line_range")
    return ("\n".join(lines[panel.line_start - 1 : panel.line_end]) + "\n").encode(
        "ascii"
    )


def _render_panel(lines: Sequence[str], panel: Panel) -> bytes:
    excerpt = lines[panel.line_start - 1 : panel.line_end]
    if len(excerpt) != panel.line_end - panel.line_start + 1:
        _fail("panel:line_range")
    width = 1_720
    height = 132 + len(excerpt) * 24
    rendered_lines = []
    for offset, line in enumerate(excerpt):
        number = panel.line_start + offset
        y = 112 + offset * 24
        rendered_lines.append(
            f'  <text x="28" y="{y}" class="terminal">'
            f'<tspan class="line-number">{number:02d}</tspan>'
            f'<tspan x="72">{html.escape(line, quote=True)}</tspan></text>'
        )
    title = html.escape(panel.title, quote=True)
    title_id = f"{panel.panel_id}-title"
    description_id = f"{panel.panel_id}-description"
    description = html.escape(
        (
            f"Exact lines {panel.line_start}-{panel.line_end} from "
            f"{TRANSCRIPT_FILENAME}."
        ),
        quote=True,
    )
    svg = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" '
            f'role="img" aria-labelledby="{title_id} {description_id}">'
        ),
        f'  <title id="{title_id}">{title}</title>',
        f'  <desc id="{description_id}">{description}</desc>',
        "  <style>",
        "    text { font-family: 'DejaVu Sans Mono', monospace; }",
        "    .title { fill: #f8fafc; font-size: 21px; font-weight: 700; }",
        "    .meta { fill: #94a3b8; font-size: 14px; }",
        "    .terminal { fill: #e2e8f0; font-size: 16px; }",
        "    .line-number { fill: #64748b; }",
        "  </style>",
        (
            f'  <rect x="1" y="1" width="{width - 2}" height="{height - 2}" '
            'rx="12" fill="#0b1020" stroke="#334155" stroke-width="2"/>'
        ),
        '  <circle cx="28" cy="25" r="6" fill="#fb7185"/>',
        '  <circle cx="48" cy="25" r="6" fill="#fbbf24"/>',
        '  <circle cx="68" cy="25" r="6" fill="#4ade80"/>',
        f'  <text x="92" y="32" class="title">{title}</text>',
        (
            f'  <text x="28" y="63" class="meta">$ {DISPLAY_COMMAND} | '
            f"stdout lines {panel.line_start}-{panel.line_end}</text>"
        ),
        (
            f'  <text x="28" y="85" class="meta">source: '
            f"{TRANSCRIPT_FILENAME} | exact line-range excerpt</text>"
        ),
        *rendered_lines,
        "</svg>",
        "",
    ]
    return "\n".join(svg).encode("ascii")


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def _source_manifest(source: BoundFile) -> dict[str, object]:
    return {
        "byte_count": source.byte_count,
        "path": source.relative_path,
        "sha256": source.sha256,
        "source_id": source.source_id,
    }


def _build_artifacts(
    transcript: bytes,
    *,
    base_commit: str,
    prior_manifest_sha256: str | None,
    snapshot: InputSnapshot,
) -> dict[str, bytes]:
    lines = _transcript_lines(transcript)
    panel_files = {panel.filename: _render_panel(lines, panel) for panel in PANELS}
    panel_records = []
    for panel in PANELS:
        contents = panel_files[panel.filename]
        panel_records.append(
            {
                "byte_count": len(contents),
                "excerpt_sha256": _sha256(_excerpt_bytes(lines, panel)),
                "line_end": panel.line_end,
                "line_start": panel.line_start,
                "panel_id": panel.panel_id,
                "path": f"docs/images/terminal/{panel.filename}",
                "sha256": _sha256(contents),
                "title": panel.title,
            }
        )
    transcript_digest = _sha256(transcript)
    manifest: dict[str, object] = {
        "artifact_set": "falsewake_terminal_evidence",
        "capture": {
            "base_commit": base_commit,
            "command_argv": list(SYMBOLIC_ARGV),
            "command_display": DISPLAY_COMMAND,
            "environment": MANIFEST_ENVIRONMENT,
            "exit_code": 0,
            "invocation_count": 1,
            "prior_manifest_sha256": prior_manifest_sha256,
            "python_version": PYTHON_VERSION,
            "shell": False,
            "working_directory": "repository-root",
        },
        "inspector": {
            "byte_count": snapshot.inspector_byte_count,
            "path": INSPECTOR_RELATIVE_PATH,
            "sha256": snapshot.inspector_sha256,
        },
        "panels": panel_records,
        "recorder": {
            "byte_count": snapshot.recorder_byte_count,
            "path": RECORDER_RELATIVE_PATH,
            "sha256": snapshot.recorder_sha256,
        },
        "schema_version": 1,
        "sources": [_source_manifest(source) for source in snapshot.sources],
        "transcript": {
            "byte_count": len(transcript),
            "line_count": len(lines),
            "output_sha256": transcript_digest,
            "path": f"docs/images/terminal/{TRANSCRIPT_FILENAME}",
            "sha256": transcript_digest,
        },
    }
    return {
        TRANSCRIPT_FILENAME: transcript,
        **panel_files,
        MANIFEST_FILENAME: _canonical_json(manifest),
    }


def _directory_inventory(directory_fd: int) -> frozenset[str]:
    try:
        entries = tuple(os.listdir(directory_fd))
    except OSError:
        _fail("artifacts:inventory")
    names = frozenset(entries)
    if len(names) != len(entries):
        _fail("artifacts:inventory")
    for entry in entries:
        try:
            entry_metadata = os.stat(
                entry,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except OSError:
            _fail("artifacts:inventory")
        if not stat.S_ISREG(entry_metadata.st_mode):
            _fail("artifacts:not_regular")
    return names


def _prepare_output_directory() -> int:
    directory_fd = _open_output_directory(create=True)
    try:
        existing = _directory_inventory(directory_fd)
        if not existing <= EXPECTED_ARTIFACT_NAMES:
            _fail("artifacts:extra_file")
    except TerminalEvidenceError:
        with suppress(OSError):
            os.close(directory_fd)
        raise
    return directory_fd


def _write_one(directory_fd: int, filename: str, contents: bytes) -> None:
    temporary = f".{filename}.tmp"
    try:
        os.stat(temporary, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    except OSError:
        _fail("artifacts:temporary")
    else:
        _fail("artifacts:temporary_exists")
    try:
        metadata = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    except OSError:
        _fail("artifacts:destination")
    else:
        if not stat.S_ISREG(metadata.st_mode):
            _fail("artifacts:destination")
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o644,
            dir_fd=directory_fd,
        )
        offset = 0
        while offset < len(contents):
            written = os.write(descriptor, contents[offset:])
            if written <= 0:
                _fail("artifacts:write")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(
            temporary,
            filename,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
    except (OSError, TerminalEvidenceError):
        if descriptor >= 0:
            with suppress(OSError):
                os.close(descriptor)
        with suppress(OSError):
            os.unlink(temporary, dir_fd=directory_fd)
        _fail("artifacts:write")


def _write_artifacts(artifacts: Mapping[str, bytes]) -> None:
    if frozenset(artifacts) != EXPECTED_ARTIFACT_NAMES:
        _fail("artifacts:set")
    directory_fd = _prepare_output_directory()
    ordered = [
        TRANSCRIPT_FILENAME,
        *(panel.filename for panel in PANELS),
        MANIFEST_FILENAME,
    ]
    try:
        for filename in ordered:
            _write_one(directory_fd, filename, artifacts[filename])
        if _directory_inventory(directory_fd) != EXPECTED_ARTIFACT_NAMES:
            _fail("artifacts:inventory")
    finally:
        with suppress(OSError):
            os.close(directory_fd)


def _reject_constant(_: str) -> NoReturn:
    _fail("manifest:nonfinite")


def _reject_float(_: str) -> NoReturn:
    _fail("manifest:float")


def _unique_mapping(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail("manifest:duplicate_key")
        result[key] = value
    return result


def _parse_manifest(contents: bytes) -> dict[str, object]:
    if len(contents) > MAX_MANIFEST_BYTE_COUNT:
        _fail("manifest:too_large")
    try:
        decoded = contents.decode("ascii")
        value = json.loads(
            decoded,
            object_pairs_hook=_unique_mapping,
            parse_constant=_reject_constant,
            parse_float=_reject_float,
        )
    except TerminalEvidenceError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError):
        _fail("manifest:json")
    if type(value) is not dict:
        _fail("manifest:schema")
    manifest = cast(dict[str, object], value)
    try:
        canonical = _canonical_json(manifest)
    except (RecursionError, TypeError, ValueError):
        _fail("manifest:json")
    if canonical != contents:
        _fail("manifest:canonical")
    return manifest


def _mapping(value: object) -> dict[str, object]:
    if type(value) is not dict:
        _fail("manifest:schema")
    raw = cast(dict[object, object], value)
    if not all(type(key) is str for key in raw):
        _fail("manifest:schema")
    return cast(dict[str, object], raw)


def _list(value: object) -> list[object]:
    if type(value) is not list:
        _fail("manifest:schema")
    return cast(list[object], value)


def _text(value: object) -> str:
    if type(value) is not str:
        _fail("manifest:schema")
    return value


def _integer(value: object) -> int:
    if type(value) is not int:
        _fail("manifest:schema")
    return value


def _boolean(value: object) -> bool:
    if type(value) is not bool:
        _fail("manifest:schema")
    return value


def _optional_sha256(value: object) -> str | None:
    if value is None:
        return None
    digest = _text(value)
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        _fail("manifest:schema")
    return digest


def _exact_equal(value: object, expected: object) -> bool:
    if type(value) is not type(expected):
        return False
    if type(value) is dict:
        actual_mapping = cast(dict[object, object], value)
        expected_mapping = cast(dict[object, object], expected)
        return set(actual_mapping) == set(expected_mapping) and all(
            _exact_equal(actual_mapping[key], expected_mapping[key])
            for key in actual_mapping
        )
    if type(value) is list:
        actual_list = cast(list[object], value)
        expected_list = cast(list[object], expected)
        return len(actual_list) == len(expected_list) and all(
            _exact_equal(actual, wanted)
            for actual, wanted in zip(actual_list, expected_list, strict=True)
        )
    return value == expected


def _keys(value: Mapping[str, object], expected: set[str]) -> None:
    if set(value) != expected:
        _fail("manifest:schema")


def _validate_manifest(
    manifest: dict[str, object],
    *,
    snapshot: InputSnapshot,
    transcript: bytes,
    panel_files: Mapping[str, bytes],
) -> tuple[str, str | None]:
    _keys(
        manifest,
        {
            "artifact_set",
            "capture",
            "inspector",
            "panels",
            "recorder",
            "schema_version",
            "sources",
            "transcript",
        },
    )
    if (
        _integer(manifest["schema_version"]) != 1
        or _text(manifest["artifact_set"]) != "falsewake_terminal_evidence"
    ):
        _fail("manifest:schema")
    capture = _mapping(manifest["capture"])
    _keys(
        capture,
        {
            "base_commit",
            "command_argv",
            "command_display",
            "environment",
            "exit_code",
            "invocation_count",
            "prior_manifest_sha256",
            "python_version",
            "shell",
            "working_directory",
        },
    )
    base_commit = _text(capture["base_commit"])
    if _COMMIT_PATTERN.fullmatch(base_commit) is None:
        _fail("manifest:base_commit")
    prior_manifest_sha256 = _optional_sha256(capture["prior_manifest_sha256"])
    if (
        _list(capture["command_argv"]) != list(SYMBOLIC_ARGV)
        or _text(capture["command_display"]) != DISPLAY_COMMAND
        or not _exact_equal(_mapping(capture["environment"]), MANIFEST_ENVIRONMENT)
        or _integer(capture["exit_code"]) != 0
        or _integer(capture["invocation_count"]) != 1
        or _text(capture["python_version"]) != PYTHON_VERSION
        or _boolean(capture["shell"])
        or _text(capture["working_directory"]) != "repository-root"
    ):
        _fail("manifest:capture")

    inspector = _mapping(manifest["inspector"])
    _keys(inspector, {"byte_count", "path", "sha256"})
    if not _exact_equal(
        inspector,
        {
            "byte_count": snapshot.inspector_byte_count,
            "path": INSPECTOR_RELATIVE_PATH,
            "sha256": snapshot.inspector_sha256,
        },
    ):
        _fail("manifest:inspector")

    recorder = _mapping(manifest["recorder"])
    _keys(recorder, {"byte_count", "path", "sha256"})
    if not _exact_equal(
        recorder,
        {
            "byte_count": snapshot.recorder_byte_count,
            "path": RECORDER_RELATIVE_PATH,
            "sha256": snapshot.recorder_sha256,
        },
    ):
        _fail("manifest:recorder")

    source_records = _list(manifest["sources"])
    expected_sources = [_source_manifest(source) for source in snapshot.sources]
    if not _exact_equal(source_records, expected_sources):
        _fail("manifest:sources")

    lines = _transcript_lines(transcript)
    transcript_record = _mapping(manifest["transcript"])
    _keys(
        transcript_record,
        {"byte_count", "line_count", "output_sha256", "path", "sha256"},
    )
    transcript_digest = _sha256(transcript)
    if not _exact_equal(
        transcript_record,
        {
            "byte_count": len(transcript),
            "line_count": len(lines),
            "output_sha256": transcript_digest,
            "path": f"docs/images/terminal/{TRANSCRIPT_FILENAME}",
            "sha256": transcript_digest,
        },
    ):
        _fail("manifest:transcript")

    raw_panels = _list(manifest["panels"])
    if len(raw_panels) != len(PANELS):
        _fail("manifest:panels")
    expected_panels = []
    for panel in PANELS:
        contents = panel_files[panel.filename]
        regenerated = _render_panel(lines, panel)
        if contents != regenerated:
            _fail("artifacts:panel")
        expected_panels.append(
            {
                "byte_count": len(contents),
                "excerpt_sha256": _sha256(_excerpt_bytes(lines, panel)),
                "line_end": panel.line_end,
                "line_start": panel.line_start,
                "panel_id": panel.panel_id,
                "path": f"docs/images/terminal/{panel.filename}",
                "sha256": _sha256(contents),
                "title": panel.title,
            }
        )
    if not _exact_equal(raw_panels, expected_panels):
        _fail("manifest:panels")
    return base_commit, prior_manifest_sha256


def _check_artifacts() -> None:
    directory_fd = _open_output_directory(create=False)
    try:
        if _directory_inventory(directory_fd) != EXPECTED_ARTIFACT_NAMES:
            _fail("artifacts:extra_or_missing")
        manifest_bytes = _file_bytes_at(
            directory_fd,
            MANIFEST_FILENAME,
            maximum_size=MAX_MANIFEST_BYTE_COUNT,
        )
        transcript = _file_bytes_at(
            directory_fd,
            TRANSCRIPT_FILENAME,
            maximum_size=MAX_TRANSCRIPT_BYTE_COUNT,
        )
        panels = {
            panel.filename: _file_bytes_at(
                directory_fd,
                panel.filename,
                maximum_size=MAX_PANEL_BYTE_COUNT,
            )
            for panel in PANELS
        }
        if _directory_inventory(directory_fd) != EXPECTED_ARTIFACT_NAMES:
            _fail("artifacts:extra_or_missing")
    finally:
        with suppress(OSError):
            os.close(directory_fd)
    snapshot = _snapshot_bound_inputs()
    manifest = _parse_manifest(manifest_bytes)
    base_commit, prior_manifest_sha256 = _validate_manifest(
        manifest,
        snapshot=snapshot,
        transcript=transcript,
        panel_files=panels,
    )
    _validate_base_state(
        base_commit,
        prior_manifest_sha256=prior_manifest_sha256,
        snapshot=snapshot,
    )


def _record() -> None:
    runtime_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    if runtime_version != PYTHON_VERSION:
        _fail("capture:python_version")
    base_commit = _require_clean_repository()
    before = _snapshot_bound_inputs()
    prior_manifest_sha256 = _prior_manifest_digest(base_commit)
    result = _run_inspector()
    if result.returncode != 0:
        _fail("capture:exit")
    if result.stderr:
        _fail("capture:stderr")
    _transcript_lines(result.stdout)
    after = _snapshot_bound_inputs()
    if after != before:
        _fail("capture:inputs_changed")
    if _require_clean_repository() != base_commit:
        _fail("capture:repository_changed")
    artifacts = _build_artifacts(
        result.stdout,
        base_commit=base_commit,
        prior_manifest_sha256=prior_manifest_sha256,
        snapshot=before,
    )
    _write_artifacts(artifacts)
    _check_artifacts()


def main(argv: Sequence[str] | None = None) -> int:
    """Run exactly one literal recorder operation."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1 or arguments[0] not in {"record", "check"}:
        sys.stderr.write("terminal evidence failed: cli:arguments\n")
        return 2
    try:
        if arguments[0] == "record":
            _record()
            sys.stdout.write(
                "recorded docs/images/terminal evidence from one inspector invocation\n"
            )
        else:
            _check_artifacts()
            sys.stdout.write("terminal evidence artifacts verified\n")
    except (TerminalEvidenceError, OSError) as error:
        code = (
            str(error)
            if isinstance(error, TerminalEvidenceError)
            else "operating_system"
        )
        sys.stderr.write(f"terminal evidence failed: {code}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
