from __future__ import annotations

import ast
import errno
import fcntl
import hashlib
import os
import resource
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import replace
from pathlib import Path

import pytest

from falsewake import experiment_002_supervisor as supervisor

SOURCE_PATH = (
    Path(__file__).parents[1] / "src" / "falsewake" / "experiment_002_supervisor.py"
)


class FakeKernel:
    def __init__(
        self,
        *,
        wait_results: list[supervisor._WaitResult] | None = None,
        samples: list[supervisor._ProcessSample | BaseException] | None = None,
        times: list[int] | None = None,
        pid: int = 41_001,
    ) -> None:
        self.allowed = {9, 2, 7}
        self.child_affinity = {2, 7}
        self.wait_results = deque(
            wait_results
            if wait_results is not None
            else [
                supervisor._WaitResult(0, 0, 0),
                supervisor._WaitResult(pid, 0, 32),
            ]
        )
        self.samples = deque(
            samples
            if samples is not None
            else [supervisor._ProcessSample(16 * 1024, ())]
        )
        self.times = deque(times if times is not None else [0, 1, 2])
        self.last_time = self.times[-1] if self.times else 0
        self.pid = pid
        self.spawned = False
        self.spawn_calls: list[
            tuple[
                str,
                tuple[str, ...],
                dict[str, str],
                tuple[supervisor._FileAction, ...],
            ]
        ] = []
        self.wait_calls: list[tuple[int, int]] = []
        self.sleep_calls: list[float] = []
        self.affinity_calls: list[int] = []
        self.kill_group_calls: list[int] = []
        self.kill_pid_calls: list[int] = []

    def parent_affinity(self) -> set[int]:
        return set(self.allowed)

    def parent_child_pids(self) -> tuple[int, ...]:
        return (self.pid,) if self.spawned else ()

    def process_affinity(self, pid: int) -> set[int]:
        assert pid == self.pid
        self.affinity_calls.append(pid)
        return set(self.child_affinity)

    def monotonic_ns(self) -> int:
        if self.times:
            self.last_time = self.times.popleft()
        else:
            self.last_time += 1
        return self.last_time

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        time.sleep(0)

    def spawn(
        self,
        executable_path: str,
        argv: tuple[str, ...],
        environment: Mapping[str, str],
        file_actions: tuple[supervisor._FileAction, ...],
    ) -> int:
        self.spawn_calls.append(
            (executable_path, argv, dict(environment), file_actions)
        )
        self.spawned = True
        return self.pid

    def wait4(self, pid: int, options: int) -> supervisor._WaitResult:
        self.wait_calls.append((pid, options))
        if not self.wait_results:
            raise AssertionError("unexpected wait4 call")
        result = self.wait_results.popleft()
        if isinstance(result, BaseException):
            raise result
        if result.pid == self.pid:
            self.spawned = False
        return result

    def kill_process_group(self, pid: int) -> None:
        self.kill_group_calls.append(pid)

    def kill_process(self, pid: int) -> None:
        self.kill_pid_calls.append(pid)

    def sample_process(self, pid: int) -> supervisor._ProcessSample:
        assert pid == self.pid
        if not self.samples:
            raise AssertionError("unexpected process sample")
        result = self.samples.popleft()
        if isinstance(result, BaseException):
            raise result
        return result


def _test_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> supervisor._LaunchPlan:
    repository = tmp_path / "repo"
    temporary = tmp_path / "external"
    runner = repository / supervisor._RUNNER_ENTRYPOINT
    taskset = tmp_path / "taskset"
    python = tmp_path / "python"
    runner.parent.mkdir(parents=True)
    temporary.mkdir()
    runner.write_bytes(b"runner\n")
    taskset.write_bytes(b"taskset\n")
    python.write_bytes(b"python\n")
    taskset.chmod(0o755)
    python.chmod(0o755)
    monkeypatch.chdir(repository)
    scratch = temporary / "scratch"
    staging = temporary / "staging"
    environment = tuple(
        (key, os.fspath(scratch) if key == "TMPDIR" else value)
        for key, value in supervisor._REGISTERED_ENVIRONMENT_ITEMS
    )
    return supervisor._LaunchPlan(
        repository_root=os.fspath(repository),
        temporary_root=os.fspath(temporary),
        scratch_root=os.fspath(scratch),
        staging_root=os.fspath(staging),
        taskset_executable=os.fspath(taskset),
        python_executable=os.fspath(python),
        runner_entrypoint=supervisor._RUNNER_ENTRYPOINT,
        runner_path=os.fspath(runner),
        environment_items=environment,
    )


def _roots(tmp_path: Path) -> tuple[Path, Path]:
    left = tmp_path / "scratch"
    right = tmp_path / "staging"
    left.mkdir(mode=0o700)
    right.mkdir(mode=0o700)
    left.chmod(0o700)
    right.chmod(0o700)
    return left, right


def _root_strings(roots: tuple[Path, Path]) -> tuple[str, str]:
    return os.fspath(roots[0]), os.fspath(roots[1])


def _limits(
    *,
    wall: int = 1_000,
    rss: int = 1_000_000,
    output: int = 1_000_000,
    poll: int = 10,
) -> supervisor._Limits:
    return supervisor._Limits(wall, rss, output, poll)


def _prepare_staging(plan: supervisor._LaunchPlan) -> None:
    supervisor._prepare_experiment_staging(plan)


@pytest.fixture(autouse=True)
def _isolate_fake_supervision_from_native_test_pools(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if (
        request.node.name
        == "test_single_thread_preflight_rejects_additional_parent_task"
    ):
        return
    monkeypatch.setattr(supervisor, "_require_single_parent_thread", lambda: None)


def test_source_is_standard_library_only_and_not_enabled() -> None:
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module)
    assert imports == {
        "__future__",
        "collections.abc",
        "contextlib",
        "ctypes",
        "dataclasses",
        "errno",
        "fcntl",
        "hashlib",
        "os",
        "signal",
        "socket",
        "stat",
        "threading",
        "time",
        "typing",
    }
    source = SOURCE_PATH.read_text(encoding="utf-8")
    assert "subprocess" not in source
    assert "preexec_fn" not in source
    assert "numpy" not in source
    assert "torch" not in source
    assert supervisor.__all__ == ("Experiment002SupervisorError",)
    assert not hasattr(supervisor, "run_registered_experiment")


def test_registered_production_constants_are_exact() -> None:
    assert supervisor._TASKSET_EXECUTABLE == "/usr/bin/taskset"
    assert (
        supervisor._PYTHON_EXECUTABLE
        == "/home/ubuntu/gitcode/.t/falsewake-venv/bin/python"
    )
    assert supervisor._RUNNER_PATH == (
        "/home/ubuntu/gitcode/falsewake/src/falsewake/experiment_002_runner.py"
    )
    assert (
        supervisor._Limits(
            wall_nanoseconds=21_600_000_000_000,
            rss_bytes=8_589_934_592,
            output_bytes=4_294_967_296,
            poll_nanoseconds=25_000_000,
        )
        == supervisor._REGISTERED_LIMITS
    )
    assert dict(supervisor._REGISTERED_ENVIRONMENT_ITEMS) == {
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


def test_launch_plan_rejects_any_environment_addition_or_reordering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _test_plan(tmp_path, monkeypatch)
    for environment_items in (
        plan.environment_items + (("PATH", "/usr/bin"),),
        tuple(reversed(plan.environment_items)),
    ):
        with pytest.raises(
            supervisor.Experiment002SupervisorError,
            match="fixed environment",
        ):
            supervisor._require_plan(replace(plan, environment_items=environment_items))


def test_capture_chooses_exactly_two_lowest_allowed_cpus() -> None:
    kernel = FakeKernel()
    assert supervisor._capture_two_lowest_cpu_ids(kernel) == (2, 7)
    kernel.allowed = {31, 30}
    assert supervisor._capture_two_lowest_cpu_ids(kernel) == (30, 31)


@pytest.mark.parametrize("allowed", [set(), {1}, {True, 2}, {-1, 0}])
def test_capture_rejects_insufficient_or_noncanonical_affinity(
    allowed: set[int],
) -> None:
    kernel = FakeKernel()
    kernel.allowed = allowed
    with pytest.raises(supervisor.Experiment002SupervisorError, match="CPU|affinity"):
        supervisor._capture_two_lowest_cpu_ids(kernel)


def test_registered_argv_uses_taskset_then_exact_isolated_runner() -> None:
    assert supervisor._registered_argv(supervisor._REGISTERED_PLAN, (0, 3)) == (
        "/usr/bin/taskset",
        "--cpu-list",
        "0,3",
        "/home/ubuntu/gitcode/.t/falsewake-venv/bin/python",
        "-I",
        "-S",
        "-B",
        "src/falsewake/experiment_002_runner.py",
    )
    for value in ((3, 3), (4, 2), (False, 2), (-1, 2)):
        with pytest.raises(supervisor.Experiment002SupervisorError):
            supervisor._registered_argv(supervisor._REGISTERED_PLAN, value)


def test_launch_path_validation_rejects_symlink(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"x")
    target.chmod(0o755)
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(supervisor.Experiment002SupervisorError, match="nonsymlink"):
        supervisor._validate_regular_path(os.fspath(link), executable=True)


def test_launch_snapshot_is_sealed_against_same_inode_content_mutation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "launch-object"
    original = b"registered launch bytes\n"
    source.write_bytes(original)
    source.chmod(0o755)
    snapshot = supervisor._open_validated_launch_snapshot(
        os.fspath(source),
        executable=True,
        expected_sha256=hashlib.sha256(original).hexdigest(),
    )
    try:
        assert stat.S_IMODE(os.fstat(snapshot).st_mode) == 0o500
        assert (
            fcntl.fcntl(snapshot, fcntl.F_GET_SEALS) == supervisor._LAUNCH_MEMFD_SEALS
        )
        source.write_bytes(b"mutated after validation\n")
        os.lseek(snapshot, 0, os.SEEK_SET)
        assert os.read(snapshot, len(original) + 64) == original
        with pytest.raises(OSError) as caught:
            os.pwrite(snapshot, b"malicious", 0)
        assert caught.value.errno == errno.EPERM
    finally:
        os.close(snapshot)


def test_fresh_roots_are_new_empty_mode_0700_and_cleanup_is_safe(
    tmp_path: Path,
) -> None:
    scratch = tmp_path / "scratch"
    staging = tmp_path / "staging"
    roots = os.fspath(scratch), os.fspath(staging)
    supervisor._prepare_fresh_output_roots(roots)
    assert stat.S_IMODE(scratch.stat().st_mode) == 0o700
    assert stat.S_IMODE(staging.stat().st_mode) == 0o700
    (scratch / "nested").mkdir()
    (scratch / "nested" / "file").write_bytes(b"payload")
    (staging / "link").symlink_to(tmp_path)
    supervisor._cleanup_output_roots(roots)
    assert not scratch.exists()
    assert not staging.exists()
    assert tmp_path.exists()
    supervisor._prepare_fresh_output_roots(roots)
    assert scratch.is_dir() and staging.is_dir()
    supervisor._cleanup_output_roots(roots)


def test_fresh_root_refuses_existing_name_without_removing_it(tmp_path: Path) -> None:
    existing = tmp_path / "scratch"
    existing.mkdir()
    marker = existing / "owned"
    marker.write_bytes(b"keep")
    with pytest.raises(supervisor.Experiment002SupervisorError, match="already exists"):
        supervisor._create_fresh_directory(os.fspath(existing))
    assert marker.read_bytes() == b"keep"


def test_accounting_uses_size_or_allocated_blocks_and_shared_total(
    tmp_path: Path,
) -> None:
    roots = _roots(tmp_path)
    first = roots[0] / "a"
    second = roots[1] / "b"
    first.write_bytes(b"a")
    second.write_bytes(b"b" * 5000)
    expected = sum(
        max(path.stat().st_size, 512 * path.stat().st_blocks)
        for path in (first, second)
    )
    assert supervisor._account_output_roots(_root_strings(roots), expected) == expected
    with pytest.raises(supervisor.Experiment002SupervisorError, match="budget"):
        supervisor._account_output_roots(_root_strings(roots), expected - 1)


def test_sparse_file_exact_budget_passes_and_plus_one_fails(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    sparse = roots[0] / "sparse"
    limit = 4_294_967_296
    with sparse.open("wb") as stream:
        stream.truncate(limit)
    assert supervisor._account_output_roots(_root_strings(roots), limit) == limit
    with sparse.open("r+b") as stream:
        stream.truncate(limit + 1)
    with pytest.raises(supervisor.Experiment002SupervisorError, match="budget"):
        supervisor._account_output_roots(_root_strings(roots), limit)


def test_pinned_accounting_charges_renamed_root_and_cleanup_unlinks_replacement(
    tmp_path: Path,
) -> None:
    roots = _roots(tmp_path)
    scratch = supervisor._pin_existing_root(os.fspath(roots[0]))
    staging = supervisor._pin_existing_root(os.fspath(roots[1]))
    hidden = tmp_path / "hidden-scratch"
    victim = tmp_path / "victim"
    victim.mkdir()
    marker = victim / "marker"
    marker.write_bytes(b"preserve")
    limit = 4_294_967_296
    try:
        roots[0].rename(hidden)
        with (hidden / "oversized").open("wb") as stream:
            stream.truncate(limit + 1)
        roots[0].symlink_to(victim, target_is_directory=True)
        with pytest.raises(supervisor.Experiment002SupervisorError, match="budget"):
            supervisor._account_pinned_roots((scratch, staging), limit)
        supervisor._cleanup_pinned_scratch(scratch)
        assert not hidden.exists()
        assert not roots[0].exists()
        assert not roots[0].is_symlink()
        assert marker.read_bytes() == b"preserve"
    finally:
        scratch.close()
        staging.close()
        supervisor._remove_directory_tree(os.fspath(roots[1]))


def test_pinned_parent_chain_detects_and_recovers_temporary_root_rename(
    tmp_path: Path,
) -> None:
    arena = tmp_path / "arena"
    temporary = arena / "temporary"
    scratch_path = temporary / "scratch"
    staging_path = temporary / "staging"
    staging_path.mkdir(parents=True, mode=0o700)
    scratch_path.mkdir(mode=0o700)
    scratch_path.chmod(0o700)
    staging_path.chmod(0o700)
    scratch = supervisor._pin_existing_root(os.fspath(scratch_path))
    staging = supervisor._pin_existing_root(os.fspath(staging_path))
    hidden_temporary = arena / "hidden-temporary"
    limit = 4_294_967_296
    try:
        temporary.rename(hidden_temporary)
        replacement_scratch = temporary / "scratch"
        replacement_staging = temporary / "staging"
        replacement_scratch.mkdir(parents=True, mode=0o700)
        replacement_staging.mkdir(mode=0o700)
        with (replacement_scratch / "unaccounted").open("wb") as stream:
            stream.truncate(limit + 1)
        with pytest.raises(
            supervisor.Experiment002SupervisorError,
            match="parent",
        ):
            supervisor._account_pinned_roots((scratch, staging), limit)

        supervisor._recover_pinned_parent((scratch, staging))
        supervisor._restore_pinned_root_name(staging)
        assert temporary.is_dir()
        assert not hidden_temporary.exists()
        assert not (scratch_path / "unaccounted").exists()
        assert (
            supervisor._identity_frame(temporary.stat())
            == staging.parent.identities[-1]
        )
        supervisor._cleanup_pinned_scratch(scratch)
        assert not scratch_path.exists()
        assert staging_path.is_dir()
    finally:
        scratch.close()
        staging.close()
        supervisor._remove_directory_tree(os.fspath(staging_path))


def test_parent_recovery_removes_managed_replacement_when_original_is_nested(
    tmp_path: Path,
) -> None:
    arena = tmp_path / "arena"
    temporary = arena / "temporary"
    scratch_path = temporary / "scratch"
    staging_path = temporary / "staging"
    staging_path.mkdir(parents=True, mode=0o700)
    scratch_path.mkdir(mode=0o700)
    scratch_path.chmod(0o700)
    staging_path.chmod(0o700)
    scratch = supervisor._pin_existing_root(os.fspath(scratch_path))
    staging = supervisor._pin_existing_root(os.fspath(staging_path))
    holding = arena / "holding"
    holding.mkdir()
    moved_temporary = holding / "temporary"
    try:
        temporary.rename(moved_temporary)
        replacement_scratch = temporary / "scratch"
        replacement_staging = temporary / "staging"
        replacement_scratch.mkdir(parents=True, mode=0o700)
        replacement_staging.mkdir(mode=0o700)
        payload = replacement_scratch / "oversized"
        with payload.open("wb") as stream:
            stream.truncate(4_294_967_297)
        with pytest.raises(
            supervisor.Experiment002SupervisorError,
            match="cannot be located",
        ):
            supervisor._recover_pinned_parent((scratch, staging))
        assert not payload.exists()
        assert not temporary.exists()
        supervisor._cleanup_pinned_scratch(scratch)
        assert not (moved_temporary / "scratch").exists()
    finally:
        scratch.close()
        staging.close()


def test_parent_recovery_cleans_payload_after_higher_ancestor_rename(
    tmp_path: Path,
) -> None:
    arena = tmp_path / "arena"
    temporary = arena / "temporary"
    scratch_path = temporary / "scratch"
    staging_path = temporary / "staging"
    staging_path.mkdir(parents=True, mode=0o700)
    scratch_path.mkdir(mode=0o700)
    scratch = supervisor._pin_existing_root(os.fspath(scratch_path))
    staging = supervisor._pin_existing_root(os.fspath(staging_path))
    hidden_arena = tmp_path / "hidden-arena"
    payload: Path | None = None
    try:
        arena.rename(hidden_arena)
        replacement_scratch = arena / "temporary" / "scratch"
        replacement_staging = arena / "temporary" / "staging"
        replacement_scratch.mkdir(parents=True, mode=0o700)
        replacement_staging.mkdir(mode=0o700)
        payload = replacement_staging / "unaccounted"
        with payload.open("wb") as stream:
            stream.truncate(4_294_967_297)

        with pytest.raises(supervisor.Experiment002SupervisorError, match="parent"):
            supervisor._account_pinned_roots(
                (scratch, staging),
                4_294_967_296,
            )
        supervisor._recover_pinned_parent((scratch, staging))
        assert arena.is_dir()
        assert not hidden_arena.exists()
        assert payload is not None and not payload.exists()
        assert supervisor._identity_frame(arena.stat()) == scratch.parent.identities[-2]
        supervisor._cleanup_pinned_scratch(scratch)
        assert not scratch_path.exists()
    finally:
        scratch.close()
        staging.close()
        if staging_path.exists():
            supervisor._remove_directory_tree(os.fspath(staging_path))


def test_staging_cleanup_removes_registered_replacement_when_original_is_nested(
    tmp_path: Path,
) -> None:
    staging_path = tmp_path / "staging"
    staging_path.mkdir(mode=0o700)
    staging = supervisor._pin_existing_root(os.fspath(staging_path))
    holding = tmp_path / "holding"
    holding.mkdir()
    moved_staging = holding / "staging"
    payload = staging_path / "unaccounted"
    try:
        staging_path.rename(moved_staging)
        staging_path.mkdir(mode=0o700)
        with payload.open("wb") as stream:
            stream.truncate(4_294_967_297)
        with pytest.raises(
            supervisor.Experiment002SupervisorError,
            match="cannot be located",
        ):
            supervisor._cleanup_pinned_staging(staging)
        assert not payload.exists()
        assert not staging_path.exists()
        assert list(moved_staging.iterdir()) == []
    finally:
        staging.close()
        if holding.exists():
            supervisor._remove_directory_tree(os.fspath(holding))


def test_accounting_charges_cross_root_hardlink_inode_exactly_once(
    tmp_path: Path,
) -> None:
    roots = _roots(tmp_path)
    original = roots[0] / "source"
    original.write_bytes(b"payload")
    os.link(original, roots[1] / "alias")
    contribution = max(
        original.stat().st_size,
        512 * original.stat().st_blocks,
    )
    assert (
        supervisor._account_output_roots(_root_strings(roots), contribution)
        == contribution
    )
    with pytest.raises(supervisor.Experiment002SupervisorError, match="budget"):
        supervisor._account_output_roots(_root_strings(roots), contribution - 1)


@pytest.mark.parametrize("kind", ["symlink", "fifo", "socket"])
def test_accounting_rejects_nonregular_leaf(tmp_path: Path, kind: str) -> None:
    roots = _roots(tmp_path)
    leaf = roots[0] / kind
    opened: socket.socket | None = None
    if kind == "symlink":
        leaf.symlink_to(roots[1])
    elif kind == "fifo":
        os.mkfifo(leaf)
    else:
        opened = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        opened.bind(os.fspath(leaf))
    try:
        with pytest.raises(
            supervisor.Experiment002SupervisorError,
            match="symlink|nonregular",
        ):
            supervisor._account_output_roots(_root_strings(roots), 1_000_000)
    finally:
        if opened is not None:
            opened.close()


def test_accounting_requires_root_mode_0700(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    roots[1].chmod(0o755)
    with pytest.raises(supervisor.Experiment002SupervisorError, match="mode"):
        supervisor._account_output_roots(_root_strings(roots), 1_000_000)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (b"Name:\tworker\nVmRSS:\t0 kB\n", 0),
        (b"VmRSS: 123 kB\nState:\tR\n", 123 * 1024),
        (b"State:\tS\nVmRSS:\t999999 kB\n", 999_999 * 1024),
    ],
)
def test_vmrss_parser_accepts_only_canonical_kib(payload: bytes, expected: int) -> None:
    assert supervisor._parse_vmrss_bytes(payload) == expected


@pytest.mark.parametrize(
    "payload",
    [
        b"State:\tR\n",
        b"VmRSS: 1 kB\nVmRSS: 2 kB\n",
        b"VmRSS: -1 kB\n",
        b"VmRSS: +1 kB\n",
        b"VmRSS: 01 kB\n",
        b"VmRSS: 1 KB\n",
        b"vmrss: 1 kB\n",
        b"VmRSS: 1 kB",
        b"VmRSS: 1 kB\0\n",
    ],
)
def test_vmrss_parser_rejects_missing_ambiguous_or_malformed(payload: bytes) -> None:
    with pytest.raises(supervisor.Experiment002SupervisorError):
        supervisor._parse_vmrss_bytes(payload)


def test_zombie_status_is_distinguished_for_terminal_wait_race() -> None:
    assert supervisor._status_is_zombie(b"Name:\tx\nState:\tZ (zombie)\n")
    assert not supervisor._status_is_zombie(b"State:\tR (running)\n")
    assert not supervisor._status_is_zombie(b"State:\tZ\nState:\tZ\n")


@pytest.mark.parametrize(
    "payload",
    [
        b"State:\tZ (zombie)\nVmRSS: malformed kB\n",
        b"State:\tZ (zombie)\nVmRSS: 1 kB\nVmRSS: 2 kB\n",
    ],
)
def test_zombie_does_not_pardon_malformed_or_duplicate_vmrss(
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes,
) -> None:
    monkeypatch.setattr(supervisor, "_read_bounded_proc_file", lambda _path: payload)
    with pytest.raises(supervisor.Experiment002SupervisorError, match="VmRSS"):
        supervisor._read_process_sample(1234)


def test_zombie_pardons_only_entirely_absent_vmrss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        supervisor,
        "_read_bounded_proc_file",
        lambda _path: b"Name:\tworker\nState:\tZ (zombie)\n",
    )
    with pytest.raises(ProcessLookupError, match="zombie"):
        supervisor._read_process_sample(1234)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [(b"", ()), (b" \n", ()), (b"12 99\n", (12, 99))],
)
def test_children_parser(payload: bytes, expected: tuple[int, ...]) -> None:
    assert supervisor._parse_children_bytes(payload) == expected


@pytest.mark.parametrize("payload", [b"0\n", b"01\n", b"1 1\n", b"-1\n", b"x\n"])
def test_children_parser_rejects_invalid_pids(payload: bytes) -> None:
    with pytest.raises(supervisor.Experiment002SupervisorError):
        supervisor._parse_children_bytes(payload)


def test_real_process_sampler_reads_current_process() -> None:
    sample = supervisor._read_process_sample(os.getpid())
    assert sample.rss_bytes > 0
    assert sample.child_pids == ()


def test_file_actions_are_the_exact_fixed_descriptor_layout() -> None:
    actions = supervisor._build_file_actions(
        parent_channel_fd=3,
        child_channel_fd=7,
        null_fd=8,
        runner_fd=9,
        python_fd=10,
        taskset_exec_fd=11,
    )
    assert actions == (
        (os.POSIX_SPAWN_DUP2, 8, 0),
        (os.POSIX_SPAWN_DUP2, 8, 1),
        (os.POSIX_SPAWN_DUP2, 8, 2),
        (os.POSIX_SPAWN_DUP2, 7, 3),
        (os.POSIX_SPAWN_DUP2, 9, 4),
        (os.POSIX_SPAWN_DUP2, 10, 5),
        (os.POSIX_SPAWN_DUP2, 11, 6),
    )


def test_file_actions_reject_unsafe_fd3_source() -> None:
    with pytest.raises(supervisor.Experiment002SupervisorError, match="safe"):
        supervisor._build_file_actions(
            parent_channel_fd=4,
            child_channel_fd=3,
            null_fd=8,
            runner_fd=9,
            python_fd=10,
            taskset_exec_fd=11,
        )


def test_safe_cloexec_source_is_duplicated_to_child_fd3_across_exec() -> None:
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    null_fd = os.open(os.devnull, os.O_RDWR | os.O_CLOEXEC)
    child_source = supervisor._duplicate_safe_descriptor(child.fileno())
    null_source = supervisor._duplicate_safe_descriptor(null_fd)
    runner_source = supervisor._duplicate_safe_descriptor(null_fd)
    python_source = supervisor._open_validated_launch_snapshot(
        supervisor._PYTHON_RESOLVED_EXECUTABLE,
        executable=True,
        expected_sha256=supervisor._PYTHON_RESOLVED_SHA256,
    )
    executable_source = supervisor._open_validated_launch_snapshot(
        supervisor._TASKSET_EXECUTABLE,
        executable=True,
        expected_sha256=supervisor._TASKSET_SHA256,
    )
    extra = supervisor._duplicate_safe_descriptor(null_fd, minimum=20)
    os.set_inheritable(extra, True)
    pid = -1
    try:
        assert child_source >= 7 and null_source >= 7
        assert not os.get_inheritable(child_source)
        assert not os.get_inheritable(null_source)
        actions = supervisor._build_file_actions(
            parent_channel_fd=parent.fileno(),
            child_channel_fd=child_source,
            null_fd=null_source,
            runner_fd=runner_source,
            python_fd=python_source,
            taskset_exec_fd=executable_source,
        )
        argv = (
            supervisor._TASKSET_EXECUTABLE,
            "--cpu-list",
            str(min(os.sched_getaffinity(0))),
            f"/proc/self/fd/{supervisor._CHILD_PYTHON_FD}",
            "-I",
            "-S",
            "-c",
            (
                "import os;"
                f"\ntry: os.fstat({extra})"
                "\nexcept OSError: os.write(3,b'fd3-ok')"
                "\nelse: os.write(3,b'leaked-high-fd')"
            ),
        )
        pid = supervisor._raw_posix_spawn(
            f"/proc/self/fd/{supervisor._CHILD_TASKSET_FD}",
            argv,
            {"LC_ALL": "C"},
            actions,
        )
        child.close()
        assert parent.recv(64) == b"fd3-ok"
        waited, status_value = os.waitpid(pid, 0)
        pid = -1
        assert waited > 0
        assert os.WIFEXITED(status_value)
        assert os.WEXITSTATUS(status_value) == 0
    finally:
        if pid > 0:
            with suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
        parent.close()
        child.close()
        os.close(null_fd)
        os.close(child_source)
        os.close(null_source)
        os.close(runner_source)
        os.close(python_source)
        os.close(executable_source)
        os.close(extra)


def test_wait4_retries_eintr_and_never_waits_for_any_child() -> None:
    kernel = FakeKernel(wait_results=[])
    calls = 0

    def wait4(pid: int, options: int) -> supervisor._WaitResult:
        nonlocal calls
        calls += 1
        assert pid == kernel.pid
        assert options == os.WNOHANG
        if calls < 3:
            raise InterruptedError
        return supervisor._WaitResult(pid, 0, 1)

    kernel.wait4 = wait4  # type: ignore[method-assign]
    result = supervisor._wait4_retry(kernel, kernel.pid, os.WNOHANG)
    assert result.pid == kernel.pid
    assert calls == 3


def test_wait4_rejects_unrelated_pid() -> None:
    kernel = FakeKernel(
        wait_results=[supervisor._WaitResult(99_999, 0, 1)],
    )
    with pytest.raises(supervisor.Experiment002SupervisorError, match="unrelated"):
        supervisor._wait4_retry(kernel, kernel.pid, os.WNOHANG)


def test_kill_and_reap_calls_group_and_pid_then_retries_interrupts() -> None:
    kernel = FakeKernel(wait_results=[])
    results: deque[supervisor._WaitResult | BaseException] = deque(
        [
            InterruptedError(),
            KeyboardInterrupt(),
            supervisor._WaitResult(kernel.pid, 9, 3),
        ]
    )

    def wait4(pid: int, options: int) -> supervisor._WaitResult:
        assert (pid, options) == (kernel.pid, os.WNOHANG)
        value = results.popleft()
        if isinstance(value, BaseException):
            raise value
        return value

    kernel.wait4 = wait4  # type: ignore[method-assign]
    handle = supervisor._ChildHandle(kernel.pid)
    supervisor._kill_and_reap(handle, kernel)
    assert handle.reaped
    assert kernel.kill_group_calls == [kernel.pid]
    assert kernel.kill_pid_calls == [kernel.pid]
    supervisor._kill_and_reap(handle, kernel)
    assert kernel.kill_group_calls == [kernel.pid]


def test_terminal_limits_allow_exact_edges_and_reject_one_over() -> None:
    limits = _limits(wall=100, rss=10 * 1024, output=20)
    result = supervisor._WaitResult(1, 0, 10)
    assert (
        supervisor._validate_terminal_result(
            result,
            elapsed_nanoseconds=100,
            observed_rss_bytes=10 * 1024,
            output_bytes=20,
            limits=limits,
        )
        == 10 * 1024
    )
    for field in ("wall", "rss", "output"):
        arguments = {
            "elapsed_nanoseconds": 100,
            "observed_rss_bytes": 10 * 1024,
            "output_bytes": 20,
        }
        if field == "wall":
            arguments["elapsed_nanoseconds"] += 1
        elif field == "rss":
            arguments["observed_rss_bytes"] += 1
        else:
            arguments["output_bytes"] += 1
        with pytest.raises(
            supervisor.Experiment002SupervisorError,
            match=field if field != "rss" else "RSS",
        ):
            supervisor._validate_terminal_result(result, limits=limits, **arguments)


def test_supervise_nominal_spawn_is_exact_and_keeps_success_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _test_plan(tmp_path, monkeypatch)
    _prepare_staging(plan)
    kernel = FakeKernel(times=[10, 20, 21, 22])
    activations: list[tuple[int, int]] = []

    def activate(channel: socket.socket, pid: int) -> None:
        activations.append((channel.family, pid))

    result = supervisor._supervise_child(
        (2, 7),
        activate,
        plan=plan,
        limits=_limits(),
        kernel=kernel,
    )
    assert result == supervisor._SupervisedChildResult(
        pid=kernel.pid,
        cpu_ids=(2, 7),
        elapsed_nanoseconds=13,
        maximum_rss_bytes=32 * 1024,
        output_and_scratch_bytes=0,
    )
    assert activations == [(socket.AF_UNIX, kernel.pid)]
    assert kernel.affinity_calls == [kernel.pid]
    assert len(kernel.spawn_calls) == 1
    executable_path, argv, environment, actions = kernel.spawn_calls[0]
    assert executable_path == "/proc/self/fd/6"
    assert argv == supervisor._pinned_spawn_argv(plan, (2, 7))
    assert environment == dict(plan.environment_items)
    assert set(environment) == {
        key for key, _value in supervisor._REGISTERED_ENVIRONMENT_ITEMS
    }
    assert "PATH" not in environment
    assert "HOME" not in environment
    assert "PYTHONPATH" not in environment
    assert any(
        action[0] == os.POSIX_SPAWN_DUP2 and action[-1] == 3 for action in actions
    )
    assert Path(plan.scratch_root).is_dir()
    assert Path(plan.staging_root).is_dir()
    supervisor._cleanup_output_roots((plan.scratch_root, plan.staging_root))


def test_inheritable_fd_snapshot_is_built_while_all_catchable_signals_are_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _test_plan(tmp_path, monkeypatch)
    _prepare_staging(plan)
    kernel = FakeKernel(times=[0, 1, 2, 3, 4])
    original = supervisor._build_file_actions
    observed_masks: list[set[int]] = []

    def inspect_mask(
        *,
        parent_channel_fd: int,
        child_channel_fd: int,
        null_fd: int,
        runner_fd: int,
        python_fd: int,
        taskset_exec_fd: int,
    ) -> tuple[supervisor._FileAction, ...]:
        observed_masks.append(
            {int(value) for value in signal.pthread_sigmask(signal.SIG_BLOCK, set())}
        )
        return original(
            parent_channel_fd=parent_channel_fd,
            child_channel_fd=child_channel_fd,
            null_fd=null_fd,
            runner_fd=runner_fd,
            python_fd=python_fd,
            taskset_exec_fd=taskset_exec_fd,
        )

    monkeypatch.setattr(supervisor, "_build_file_actions", inspect_mask)
    supervisor._supervise_child(
        (2, 7),
        lambda _channel, _pid: None,
        plan=plan,
        limits=_limits(),
        kernel=kernel,
    )
    assert len(observed_masks) == 1
    assert supervisor._blockable_signals() <= observed_masks[0]
    supervisor._cleanup_output_roots((plan.scratch_root, plan.staging_root))


@pytest.mark.parametrize(
    ("sample", "message"),
    [
        (supervisor._ProcessSample(1_000_001, ()), "RSS"),
        (supervisor._ProcessSample(1, (55,)), "child process"),
    ],
)
def test_supervise_resource_breach_kills_reaps_and_cleans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample: supervisor._ProcessSample,
    message: str,
) -> None:
    plan = _test_plan(tmp_path, monkeypatch)
    _prepare_staging(plan)
    kernel = FakeKernel(
        wait_results=[
            supervisor._WaitResult(0, 0, 0),
            supervisor._WaitResult(41_001, 9, 64),
        ],
        samples=[sample],
        times=[0, 1],
    )
    with pytest.raises(supervisor.Experiment002SupervisorError, match=message):
        supervisor._supervise_child(
            (2, 7),
            lambda _channel, _pid: None,
            plan=plan,
            limits=_limits(),
            kernel=kernel,
        )
    assert kernel.kill_group_calls == [kernel.pid]
    assert kernel.kill_pid_calls == [kernel.pid]
    assert kernel.wait_calls == [
        (kernel.pid, os.WNOHANG),
        (kernel.pid, os.WNOHANG),
    ]
    assert not Path(plan.scratch_root).exists()
    assert Path(plan.staging_root).is_dir()
    supervisor._remove_directory_tree(plan.staging_root)


def test_supervise_timeout_kills_and_reaps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _test_plan(tmp_path, monkeypatch)
    _prepare_staging(plan)
    kernel = FakeKernel(
        wait_results=[
            supervisor._WaitResult(0, 0, 0),
            supervisor._WaitResult(41_001, 9, 1),
        ],
        times=[0, 101],
    )
    with pytest.raises(supervisor.Experiment002SupervisorError, match="wall-time"):
        supervisor._supervise_child(
            (2, 7),
            lambda _channel, _pid: None,
            plan=plan,
            limits=_limits(wall=100),
            kernel=kernel,
        )
    assert kernel.kill_group_calls == [kernel.pid]
    assert kernel.kill_pid_calls == [kernel.pid]
    supervisor._remove_directory_tree(plan.staging_root)


def test_terminal_ru_maxrss_failure_never_signals_reaped_pid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _test_plan(tmp_path, monkeypatch)
    _prepare_staging(plan)
    kernel = FakeKernel(
        wait_results=[
            supervisor._WaitResult(0, 0, 0),
            supervisor._WaitResult(41_001, 0, 1_001),
        ],
        times=[0, 1, 2, 3],
    )
    with pytest.raises(supervisor.Experiment002SupervisorError, match="RSS"):
        supervisor._supervise_child(
            (2, 7),
            lambda _channel, _pid: None,
            plan=plan,
            limits=_limits(rss=1_000 * 1024),
            kernel=kernel,
        )
    assert kernel.kill_group_calls == []
    assert kernel.kill_pid_calls == []
    assert kernel.wait_calls == [
        (kernel.pid, os.WNOHANG),
        (kernel.pid, os.WNOHANG),
    ]
    supervisor._remove_directory_tree(plan.staging_root)


def test_proc_disappearance_is_pardoned_only_after_exact_terminal_wait(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _test_plan(tmp_path, monkeypatch)
    _prepare_staging(plan)
    kernel = FakeKernel(
        wait_results=[
            supervisor._WaitResult(0, 0, 0),
            supervisor._WaitResult(41_001, 0, 3),
        ],
        samples=[ProcessLookupError("zombie")],
        times=[0, 1, 2],
    )
    result = supervisor._supervise_child(
        (2, 7),
        lambda _channel, _pid: None,
        plan=plan,
        limits=_limits(),
        kernel=kernel,
    )
    assert result.maximum_rss_bytes == 3 * 1024
    assert kernel.wait_calls == [
        (kernel.pid, os.WNOHANG),
        (kernel.pid, os.WNOHANG),
    ]
    supervisor._cleanup_output_roots((plan.scratch_root, plan.staging_root))


def test_activation_is_monitored_concurrently_and_failure_is_contained(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _test_plan(tmp_path, monkeypatch)
    _prepare_staging(plan)
    kernel = FakeKernel(
        wait_results=[
            supervisor._WaitResult(0, 0, 0),
            supervisor._WaitResult(41_001, 9, 1),
        ],
        samples=[supervisor._ProcessSample(1_000_001, ())],
        times=[0, 1],
    )
    entered = threading.Event()

    def blocked_activation(channel: socket.socket, _pid: int) -> None:
        entered.set()
        channel.recv(1)

    with pytest.raises(supervisor.Experiment002SupervisorError, match="RSS"):
        supervisor._supervise_child(
            (2, 7),
            blocked_activation,
            plan=plan,
            limits=_limits(),
            kernel=kernel,
        )
    assert entered.is_set()
    assert kernel.kill_group_calls == [kernel.pid]
    assert kernel.kill_pid_calls == [kernel.pid]
    supervisor._remove_directory_tree(plan.staging_root)


def test_activation_keyboard_interrupt_still_kills_reaps_and_cleans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _test_plan(tmp_path, monkeypatch)
    _prepare_staging(plan)
    kernel = FakeKernel(
        wait_results=[
            supervisor._WaitResult(0, 0, 0),
            supervisor._WaitResult(41_001, 9, 1),
        ],
        times=[0, 1],
    )

    def interrupt(_channel: socket.socket, _pid: int) -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        supervisor._supervise_child(
            (2, 7),
            interrupt,
            plan=plan,
            limits=_limits(),
            kernel=kernel,
        )
    assert kernel.kill_group_calls == [kernel.pid]
    assert kernel.kill_pid_calls == [kernel.pid]
    assert not Path(plan.scratch_root).exists()
    supervisor._remove_directory_tree(plan.staging_root)


def test_spawn_failure_cleans_child_scratch_and_preserves_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _test_plan(tmp_path, monkeypatch)
    _prepare_staging(plan)
    kernel = FakeKernel(times=[0])

    def fail_spawn(
        executable_path: str,
        argv: tuple[str, ...],
        environment: Mapping[str, str],
        file_actions: tuple[supervisor._FileAction, ...],
    ) -> int:
        del executable_path, argv, environment, file_actions
        raise OSError("synthetic spawn failure")

    kernel.spawn = fail_spawn  # type: ignore[method-assign]
    with pytest.raises(OSError, match="synthetic"):
        supervisor._supervise_child(
            (2, 7),
            lambda _channel, _pid: None,
            plan=plan,
            limits=_limits(),
            kernel=kernel,
        )
    assert not Path(plan.scratch_root).exists()
    assert Path(plan.staging_root).is_dir()
    supervisor._remove_directory_tree(plan.staging_root)


@pytest.mark.parametrize(
    ("returned_pid", "message"),
    [(0, "invalid child PID"), (99_999, "child journal")],
)
def test_invalid_or_mismatched_spawn_pid_contains_authoritative_journal_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    returned_pid: int,
    message: str,
) -> None:
    plan = _test_plan(tmp_path, monkeypatch)
    _prepare_staging(plan)
    kernel = FakeKernel(
        wait_results=[supervisor._WaitResult(41_001, 9, 1)],
        times=[0, 1],
    )

    def invalid_spawn(
        executable_path: str,
        argv: tuple[str, ...],
        environment: Mapping[str, str],
        file_actions: tuple[supervisor._FileAction, ...],
    ) -> int:
        del executable_path, argv, environment, file_actions
        kernel.spawned = True
        return returned_pid

    kernel.spawn = invalid_spawn  # type: ignore[method-assign]
    with pytest.raises(supervisor.Experiment002SupervisorError, match=message):
        supervisor._supervise_child(
            (2, 7),
            lambda _channel, _pid: None,
            plan=plan,
            limits=_limits(),
            kernel=kernel,
        )
    assert kernel.kill_group_calls == [kernel.pid]
    assert kernel.kill_pid_calls == [kernel.pid]
    assert kernel.wait_calls == [(kernel.pid, os.WNOHANG)]
    assert not Path(plan.scratch_root).exists()
    assert Path(plan.staging_root).is_dir()
    supervisor._remove_directory_tree(plan.staging_root)


def test_ambiguous_spawn_keyboard_interrupt_is_journaled_killed_and_reaped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _test_plan(tmp_path, monkeypatch)
    _prepare_staging(plan)
    kernel = FakeKernel(
        wait_results=[supervisor._WaitResult(41_001, 9, 1)],
        times=[0, 1],
    )

    def interrupted_spawn(
        executable_path: str,
        argv: tuple[str, ...],
        environment: Mapping[str, str],
        file_actions: tuple[supervisor._FileAction, ...],
    ) -> int:
        del executable_path, argv, environment, file_actions
        kernel.spawned = True
        raise KeyboardInterrupt

    kernel.spawn = interrupted_spawn  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        supervisor._supervise_child(
            (2, 7),
            lambda _channel, _pid: None,
            plan=plan,
            limits=_limits(),
            kernel=kernel,
        )
    assert kernel.kill_group_calls == [kernel.pid]
    assert kernel.kill_pid_calls == [kernel.pid]
    assert kernel.wait_calls == [(kernel.pid, os.WNOHANG)]
    assert not Path(plan.scratch_root).exists()
    assert Path(plan.staging_root).is_dir()
    supervisor._remove_directory_tree(plan.staging_root)


def test_failed_child_clears_and_restores_exact_staging_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _test_plan(tmp_path, monkeypatch)
    _prepare_staging(plan)
    staging = Path(plan.staging_root)
    original_identity = supervisor._identity_frame(staging.stat())
    hidden = staging.with_name("hidden-staging")
    marker = hidden / "preserved"
    kernel = FakeKernel(
        wait_results=[
            supervisor._WaitResult(0, 0, 0),
            supervisor._WaitResult(41_001, 9, 1),
        ],
        times=[0, 1, 2],
    )

    def replace_staging(_channel: socket.socket, _pid: int) -> None:
        staging.rename(hidden)
        marker.write_bytes(b"prior verified result")
        staging.mkdir(mode=0o700)
        (staging / "replacement").write_bytes(b"discard")

    with pytest.raises(
        supervisor.Experiment002SupervisorError,
        match="identity changed",
    ):
        supervisor._supervise_child(
            (2, 7),
            replace_staging,
            plan=plan,
            limits=_limits(),
            kernel=kernel,
        )
    assert supervisor._identity_frame(staging.stat()) == original_identity
    assert not (staging / "preserved").exists()
    assert not (staging / "replacement").exists()
    assert not hidden.exists()
    assert not Path(plan.scratch_root).exists()
    supervisor._remove_directory_tree(plan.staging_root)


def test_staging_budget_breach_is_removed_after_child_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _test_plan(tmp_path, monkeypatch)
    _prepare_staging(plan)
    staging = Path(plan.staging_root)
    payload = staging / "oversized"
    kernel = FakeKernel(
        wait_results=[
            supervisor._WaitResult(0, 0, 0),
            supervisor._WaitResult(41_001, 9, 1),
        ],
        times=[0, 1, 2],
    )

    def exceed_budget(_channel: socket.socket, _pid: int) -> None:
        with payload.open("wb") as stream:
            stream.truncate(4_294_967_297)

    with pytest.raises(supervisor.Experiment002SupervisorError, match="budget"):
        supervisor._supervise_child(
            (2, 7),
            exceed_budget,
            plan=plan,
            limits=_limits(output=4_294_967_296),
            kernel=kernel,
        )
    assert staging.is_dir()
    assert not payload.exists()
    assert list(staging.iterdir()) == []
    assert not Path(plan.scratch_root).exists()
    supervisor._remove_directory_tree(plan.staging_root)


@pytest.mark.parametrize("handler", [signal.SIG_IGN, lambda _signum, _frame: None])
def test_nondefault_sigchld_is_rejected_before_spawn(
    monkeypatch: pytest.MonkeyPatch,
    handler: object,
) -> None:
    monkeypatch.setattr(signal, "getsignal", lambda _signal: handler)
    with pytest.raises(supervisor.Experiment002SupervisorError, match="SIGCHLD"):
        supervisor._require_authoritative_wait4()


def test_default_sigchld_allows_authoritative_wait4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(signal, "getsignal", lambda _signal: signal.SIG_DFL)
    supervisor._require_authoritative_wait4()


def test_signal_disposition_frame_rejects_custom_handler_before_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _test_plan(tmp_path, monkeypatch)
    _prepare_staging(plan)
    kernel = FakeKernel()
    original_getsignal = signal.getsignal

    def getsignal(signal_number: int) -> object:
        if signal_number == int(signal.SIGUSR1):
            return lambda _signum, _frame: None
        return original_getsignal(signal_number)

    monkeypatch.setattr(signal, "getsignal", getsignal)
    with pytest.raises(
        supervisor.Experiment002SupervisorError,
        match="untrusted signal disposition",
    ):
        supervisor._supervise_child(
            (2, 7),
            lambda _channel, _pid: None,
            plan=plan,
            limits=_limits(),
            kernel=kernel,
        )
    assert kernel.spawn_calls == []
    assert not Path(plan.scratch_root).exists()
    supervisor._remove_directory_tree(plan.staging_root)


def test_parent_signals_remain_blocked_through_activation_and_monitor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _test_plan(tmp_path, monkeypatch)
    _prepare_staging(plan)
    kernel = FakeKernel(times=[0, 1, 2, 3])
    observed_masks: list[set[int]] = []
    original_sample = kernel.sample_process
    original_mask = {
        int(value) for value in signal.pthread_sigmask(signal.SIG_BLOCK, set())
    }

    def sample(pid: int) -> supervisor._ProcessSample:
        observed_masks.append(
            {int(value) for value in signal.pthread_sigmask(signal.SIG_BLOCK, set())}
        )
        return original_sample(pid)

    def activate(_channel: socket.socket, _pid: int) -> None:
        observed_masks.append(
            {int(value) for value in signal.pthread_sigmask(signal.SIG_BLOCK, set())}
        )

    kernel.sample_process = sample  # type: ignore[method-assign]
    supervisor._supervise_child(
        (2, 7),
        activate,
        plan=plan,
        limits=_limits(),
        kernel=kernel,
    )
    assert len(observed_masks) == 2
    assert all(supervisor._blockable_signals() <= mask for mask in observed_masks)
    restored_mask = {
        int(value) for value in signal.pthread_sigmask(signal.SIG_BLOCK, set())
    }
    assert restored_mask == original_mask
    supervisor._cleanup_output_roots((plan.scratch_root, plan.staging_root))


def test_unregistered_parent_child_is_killed_and_reaped() -> None:
    kernel = FakeKernel(
        wait_results=[supervisor._WaitResult(41_001, int(signal.SIGKILL), 1)],
    )
    kernel.spawned = True
    assert supervisor._contain_unregistered_parent_children(kernel)
    assert kernel.kill_group_calls == [kernel.pid]
    assert kernel.kill_pid_calls == [kernel.pid]
    assert kernel.wait_calls == [(kernel.pid, os.WNOHANG)]
    assert kernel.parent_child_pids() == ()


def test_actual_production_paths_and_registered_hashes_validate_without_spawn() -> None:
    supervisor._validate_launch_surface(supervisor._REGISTERED_PLAN)


def test_production_launch_objects_are_exact_sealed_snapshots() -> None:
    descriptors = supervisor._pin_launch_files(supervisor._REGISTERED_PLAN)
    expected = (
        (supervisor._TASKSET_SHA256, 0o500),
        (supervisor._PYTHON_RESOLVED_SHA256, 0o500),
        (supervisor._RUNNER_SHA256, 0o400),
    )
    try:
        for descriptor, (expected_digest, expected_mode) in zip(
            descriptors,
            expected,
            strict=True,
        ):
            assert stat.S_IMODE(os.fstat(descriptor).st_mode) == expected_mode
            assert (
                fcntl.fcntl(descriptor, fcntl.F_GET_SEALS)
                == supervisor._LAUNCH_MEMFD_SEALS
            )
            os.lseek(descriptor, 0, os.SEEK_SET)
            hasher = hashlib.sha256()
            while chunk := os.read(descriptor, 1 << 20):
                hasher.update(chunk)
            assert hasher.hexdigest() == expected_digest
    finally:
        for descriptor in descriptors:
            os.close(descriptor)


def test_monitor_sleep_request_is_strictly_below_100ms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _test_plan(tmp_path, monkeypatch)
    _prepare_staging(plan)
    kernel = FakeKernel(times=[0, 1, 2, 3])
    result = supervisor._supervise_child(
        (2, 7),
        lambda _channel, _pid: None,
        plan=plan,
        limits=_limits(wall=1_000_000_000, poll=99_999_999),
        kernel=kernel,
    )
    assert result.pid == kernel.pid
    assert kernel.sleep_calls == [pytest.approx(0.099999998)]
    supervisor._cleanup_output_roots((plan.scratch_root, plan.staging_root))


def test_cleanup_never_follows_replaced_root_symlink(tmp_path: Path) -> None:
    victim = tmp_path / "victim"
    victim.mkdir()
    marker = victim / "marker"
    marker.write_bytes(b"preserve")
    root = tmp_path / "scratch"
    root.symlink_to(victim, target_is_directory=True)
    supervisor._remove_directory_tree(os.fspath(root))
    assert not root.exists()
    assert marker.read_bytes() == b"preserve"


def test_cleanup_retries_interrupted_unlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "scratch"
    root.mkdir()
    (root / "file").write_bytes(b"payload")
    original = os.unlink
    injected = False

    def interrupted_once(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        dir_fd: int | None = None,
    ) -> None:
        nonlocal injected
        if not injected and os.fsdecode(path) == "file":
            injected = True
            raise InterruptedError
        original(path, dir_fd=dir_fd)

    monkeypatch.setattr(os, "unlink", interrupted_once)
    supervisor._remove_directory_tree(os.fspath(root))
    assert injected
    assert not root.exists()


def test_cleanup_ignores_accounting_depth_limit_and_removes_deep_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "scratch"
    nested = root / "one" / "two"
    nested.mkdir(parents=True)
    with monkeypatch.context() as scoped:
        scoped.setattr(supervisor, "_OUTPUT_DEPTH_MAXIMUM", 0)
        supervisor._remove_directory_tree(os.fspath(root))
    assert not root.exists()


def test_cleanup_removes_sparse_payload_beyond_accounting_depth_bound(
    tmp_path: Path,
) -> None:
    roots = _roots(tmp_path)
    current = roots[0]
    for index in range(supervisor._OUTPUT_DEPTH_MAXIMUM + 2):
        current /= f"d{index}"
        current.mkdir()
    payload = current / "oversized"
    with payload.open("wb") as stream:
        stream.truncate(4_294_967_297)
    with pytest.raises(supervisor.Experiment002SupervisorError, match="nesting"):
        supervisor._account_output_roots(_root_strings(roots), 4_294_967_296)
    supervisor._remove_directory_tree(os.fspath(roots[0]))
    assert not roots[0].exists()
    assert not payload.exists()
    supervisor._remove_directory_tree(os.fspath(roots[1]))


def test_cleanup_uses_constant_descriptor_space_under_low_nofile_limit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "scratch"
    root.mkdir()
    current = root
    for index in range(96):
        current /= f"d{index}"
        current.mkdir()
    payload = current / "payload"
    payload.write_bytes(b"bounded cleanup")
    original_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    if original_limit[0] < 32 or original_limit[1] < 32:
        pytest.skip("test process already has an unusually low RLIMIT_NOFILE")
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, original_limit[1]))
    try:
        supervisor._remove_directory_tree(os.fspath(root))
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, original_limit)
    assert not payload.exists()
    assert not root.exists()


def test_output_entry_and_depth_bounds_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = _roots(tmp_path)
    (roots[0] / "one").write_bytes(b"")
    (roots[0] / "two").write_bytes(b"")
    monkeypatch.setattr(supervisor, "_OUTPUT_ENTRY_COUNT_MAXIMUM", 1)
    with pytest.raises(supervisor.Experiment002SupervisorError, match="too many"):
        supervisor._account_output_roots(_root_strings(roots), 1_000_000)
    monkeypatch.setattr(supervisor, "_OUTPUT_ENTRY_COUNT_MAXIMUM", 10)
    monkeypatch.setattr(supervisor, "_OUTPUT_DEPTH_MAXIMUM", 0)
    (roots[1] / "nested").mkdir()
    with pytest.raises(supervisor.Experiment002SupervisorError, match="nesting"):
        supervisor._account_output_roots(_root_strings(roots), 1_000_000)


def test_experiment_staging_survives_across_fresh_children(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _test_plan(tmp_path, monkeypatch)
    _prepare_staging(plan)
    marker = Path(plan.staging_root) / "seed-0.result"

    def first_activation(_channel: socket.socket, _pid: int) -> None:
        marker.write_bytes(b"verified seed result")

    first = FakeKernel(times=[0, 1, 2, 3])
    supervisor._supervise_child(
        (2, 7),
        first_activation,
        plan=plan,
        limits=_limits(),
        kernel=first,
    )
    supervisor._remove_directory_tree(plan.scratch_root)
    second = FakeKernel(times=[10, 11, 12, 13])

    def second_activation(_channel: socket.socket, _pid: int) -> None:
        assert marker.read_bytes() == b"verified seed result"

    result = supervisor._supervise_child(
        (2, 7),
        second_activation,
        plan=plan,
        limits=_limits(),
        kernel=second,
    )
    expected = max(marker.stat().st_size, 512 * marker.stat().st_blocks)
    assert result.output_and_scratch_bytes == expected
    supervisor._cleanup_output_roots((plan.scratch_root, plan.staging_root))


def test_fast_exit_cannot_succeed_without_pre_reap_affinity_observation(
    tmp_path: Path,
) -> None:
    roots = _roots(tmp_path)
    scratch = supervisor._pin_existing_root(os.fspath(roots[0]))
    staging = supervisor._pin_existing_root(os.fspath(roots[1]))
    kernel = FakeKernel(
        wait_results=[],
        times=[1, 2],
    )
    state = supervisor._ActivationState(
        completed=threading.Event(),
        lock=threading.Lock(),
    )

    def terminal_wait(_pid: int, _options: int) -> supervisor._WaitResult:
        state.completed.set()
        return supervisor._WaitResult(kernel.pid, 0, 1)

    kernel.wait4 = terminal_wait  # type: ignore[assignment]
    handle = supervisor._ChildHandle(kernel.pid)
    try:
        with pytest.raises(supervisor.Experiment002SupervisorError, match="affinity"):
            supervisor._monitor_child(
                handle,
                cpu_ids=(2, 7),
                roots=(scratch, staging),
                start_nanoseconds=0,
                activation=state,
                kernel=kernel,
                limits=_limits(),
            )
    finally:
        scratch.close()
        staging.close()
    assert handle.reaped
    assert kernel.affinity_calls == []


def test_kill_group_permission_error_still_uses_pid_and_reaps() -> None:
    kernel = FakeKernel(
        wait_results=[
            supervisor._WaitResult(0, 0, 0),
            supervisor._WaitResult(41_001, 9, 1),
        ],
        times=[0, 1],
    )

    def denied(_pid: int) -> None:
        raise PermissionError(errno.EPERM, "denied")

    kernel.kill_process_group = denied  # type: ignore[assignment]
    handle = supervisor._ChildHandle(kernel.pid)
    supervisor._kill_and_reap(handle, kernel)
    assert handle.reaped
    assert kernel.kill_pid_calls == [kernel.pid]
    assert kernel.sleep_calls


def test_failed_kills_and_nonexiting_child_hit_termination_bound() -> None:
    kernel = FakeKernel(
        wait_results=[supervisor._WaitResult(0, 0, 0)],
        times=[0, supervisor._TERMINATION_NANOSECONDS_MAXIMUM + 1],
    )

    def denied(_pid: int) -> None:
        raise PermissionError(errno.EPERM, "denied")

    kernel.kill_process_group = denied  # type: ignore[assignment]
    kernel.kill_process = denied  # type: ignore[assignment]
    with pytest.raises(supervisor.Experiment002SupervisorError, match="termination"):
        supervisor._kill_and_reap(supervisor._ChildHandle(kernel.pid), kernel)


def test_sampling_cycle_over_100ms_is_contained(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _test_plan(tmp_path, monkeypatch)
    _prepare_staging(plan)
    kernel = FakeKernel(
        wait_results=[
            supervisor._WaitResult(0, 0, 0),
            supervisor._WaitResult(41_001, 9, 1),
        ],
        times=[0, 1, 100_000_002, 100_000_003],
    )
    with pytest.raises(
        supervisor.Experiment002SupervisorError, match="100 milliseconds"
    ):
        supervisor._supervise_child(
            (2, 7),
            lambda _channel, _pid: None,
            plan=plan,
            limits=_limits(wall=1_000_000_000),
            kernel=kernel,
        )
    assert kernel.kill_pid_calls == [kernel.pid]
    supervisor._remove_directory_tree(plan.staging_root)


def test_scheduler_oversleep_between_cycle_starts_is_contained(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _test_plan(tmp_path, monkeypatch)
    _prepare_staging(plan)
    kernel = FakeKernel(
        wait_results=[
            supervisor._WaitResult(0, 0, 0),
            supervisor._WaitResult(41_001, 9, 1),
        ],
        samples=[supervisor._ProcessSample(1, ())],
        times=[0, 1, 2, 100_000_002, 100_000_003],
    )
    with pytest.raises(
        supervisor.Experiment002SupervisorError,
        match="observations.*100 milliseconds",
    ):
        supervisor._supervise_child(
            (2, 7),
            lambda _channel, _pid: None,
            plan=plan,
            limits=_limits(wall=1_000_000_000),
            kernel=kernel,
        )
    assert kernel.kill_pid_calls == [kernel.pid]
    supervisor._remove_directory_tree(plan.staging_root)


def test_real_kernel_spawn_uses_setpgroup_empty_signal_mask_and_copied_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_spawn(
        executable_path: str,
        argv: tuple[str, ...],
        environment: dict[str, str],
        file_actions: tuple[supervisor._FileAction, ...],
    ) -> int:
        observed.update(
            executable_path=executable_path,
            argv=argv,
            environment=environment,
            file_actions=file_actions,
        )
        return 1234

    monkeypatch.setattr(supervisor, "_raw_posix_spawn", fake_spawn)
    environment = {"ONLY": "registered"}
    actions: tuple[supervisor._FileAction, ...] = ((os.POSIX_SPAWN_CLOSE, 9),)
    result = supervisor._RealKernel().spawn(
        "/proc/self/fd/6",
        ("/registered/taskset", "argument"),
        environment,
        actions,
    )
    assert result == 1234
    assert observed == {
        "executable_path": "/proc/self/fd/6",
        "argv": ("/registered/taskset", "argument"),
        "environment": environment,
        "file_actions": actions,
    }
    assert observed["environment"] is not environment


def test_supervisor_requires_experiment_staging_before_scratch_or_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _test_plan(tmp_path, monkeypatch)
    kernel = FakeKernel()
    with pytest.raises(supervisor.Experiment002SupervisorError, match="staging"):
        supervisor._supervise_child(
            (2, 7),
            lambda _channel, _pid: None,
            plan=plan,
            limits=_limits(),
            kernel=kernel,
        )
    assert kernel.spawn_calls == []
    assert not Path(plan.scratch_root).exists()


def test_single_thread_preflight_rejects_additional_parent_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(os, "listdir", lambda _path: [str(os.getpid()), "99999"])
    with pytest.raises(supervisor.Experiment002SupervisorError, match="one thread"):
        supervisor._require_single_parent_thread()


def test_single_thread_preflight_accepts_fresh_production_process() -> None:
    source_root = SOURCE_PATH.parents[1]
    code = (
        "import sys;"
        f"sys.path.insert(0,{os.fspath(source_root)!r});"
        "from falsewake import experiment_002_supervisor as supervisor;"
        "supervisor._require_single_parent_thread();"
        "print('single-thread-ok')"
    )
    completed = subprocess.run(
        (sys.executable, "-I", "-S", "-B", "-c", code),
        cwd=SOURCE_PATH.parents[2],
        env={"LC_ALL": "C"},
        check=False,
        capture_output=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert completed.stdout == b"single-thread-ok\n"
    assert completed.stderr == b""


def test_private_seams_have_no_production_path_or_budget_overrides() -> None:
    assert supervisor._REGISTERED_PLAN.scratch_root.startswith(
        "/home/ubuntu/gitcode/.t/"
    )
    assert supervisor._REGISTERED_PLAN.staging_root.startswith(
        "/home/ubuntu/gitcode/.t/"
    )
    assert supervisor._POLL_INTERVAL_NANOSECONDS < 100_000_000
    assert supervisor._CPU_COUNT == 2
    assert not any(
        name.startswith("run_") or name.startswith("main")
        for name in supervisor.__dict__
        if not name.startswith("__")
    )
