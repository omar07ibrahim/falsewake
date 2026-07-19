from __future__ import annotations

import ast
import errno
import fcntl
import hashlib
import inspect
import json
import os
import resource
import signal
import socket
import stat
import struct
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterator, Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, cast

import pytest

from falsewake import experiment_002_child_result as child_result
from falsewake import experiment_002_supervisor as supervisor

SOURCE_PATH = (
    Path(__file__).parents[1] / "src" / "falsewake" / "experiment_002_supervisor.py"
)
_TEST_SOURCE_BUNDLE = b"falsewake synthetic sealed source bundle\n"
_HISTORY_DOMAIN = b"falsewake-exp002-history-v1\0"
_MODEL_TENSOR_DOMAIN = b"falsewake-exp002-model-tensors-v1\0"
_CLASS_SUPPORT = (397, 406, 350, 377, 352, 363, 363, 373, 350, 372, 6_278, 602)


@dataclass(frozen=True, slots=True)
class _ResultCase:
    binding: child_result.ChildResultBinding
    history: bytes
    history_sha256: str
    safetensors: bytes
    safetensors_sha256: str
    winner_epoch: int
    model_tensor_sha256: str


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _canonical(document: dict[str, object]) -> bytes:
    return (
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _result_safetensors() -> tuple[bytes, str]:
    header: dict[str, object] = {}
    data = bytearray()
    framed = bytearray(struct.pack("<I", len(child_result._REGISTERED_MODEL_SPECS)))
    offset = 0
    for tensor_index, (name, shape) in enumerate(child_result._REGISTERED_MODEL_SPECS):
        value_count = 1
        for dimension in shape:
            value_count *= dimension
        raw = struct.pack("<f", float(tensor_index) / 100.0) * value_count
        end = offset + len(raw)
        header[name] = {
            "dtype": "F32",
            "shape": list(shape),
            "data_offsets": [offset, end],
        }
        data.extend(raw)
        name_bytes = name.encode("utf-8")
        framed.extend(struct.pack("<I", len(name_bytes)))
        framed.extend(name_bytes)
        framed.extend(struct.pack("<I", 5))
        framed.extend(b"F32LE")
        framed.extend(struct.pack("<I", len(shape)))
        for dimension in shape:
            framed.extend(struct.pack("<Q", dimension))
        framed.extend(struct.pack("<Q", len(raw)))
        framed.extend(raw)
        offset = end
    header_bytes = json.dumps(
        header,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    padded_header = header_bytes + b" " * (-len(header_bytes) % 8)
    payload = struct.pack("<Q", len(padded_header)) + padded_header + bytes(data)
    return payload, hashlib.sha256(_MODEL_TENSOR_DOMAIN + bytes(framed)).hexdigest()


def _result_history(*, seed: int, model_tensor_sha256: str) -> bytes:
    validation_input_digest = _sha("validation-inputs")
    confusion: list[list[int]] = []
    for index, support in enumerate(_CLASS_SUPPORT):
        row = [0] * len(_CLASS_SUPPORT)
        row[index] = support
        confusion.append(row)
    epochs: list[dict[str, object]] = []
    winner_epoch = 7
    for epoch in range(30):
        validation_cross_entropy = 0.25 if epoch == winner_epoch else 2.0 + epoch
        epochs.append(
            {
                "epoch_update_trace_digest": _sha(f"epoch-trace-{epoch}"),
                "first_global_update": epoch * 313,
                "last_global_update_inclusive": (epoch + 1) * 313 - 1,
                "macro_f1_exact_denominator": 1,
                "macro_f1_exact_numerator": 1,
                "model_tensor_digest": (
                    model_tensor_sha256
                    if epoch == winner_epoch
                    else _sha(f"model-{seed}-{epoch}")
                ),
                "training_cross_entropy_float64_hex": float(1.0 + epoch).hex(),
                "training_population_digest": _sha(f"population-{seed}-{epoch}"),
                "validation_confusion_matrix": confusion,
                "validation_cross_entropy_float64_hex": (
                    validation_cross_entropy.hex()
                ),
                "validation_input_digest": validation_input_digest,
                "validation_prediction_digest": _sha(f"predictions-{seed}-{epoch}"),
                "zero_based_epoch": epoch,
            }
        )
    return _canonical(
        {
            "complete_update_trace_digest": _sha(f"complete-update-trace-{seed}"),
            "epochs": epochs,
            "experiment": "002",
            "schema_version": 1,
            "seed": seed,
            "validation_input_digest": validation_input_digest,
        }
    )


def _result_case(
    *,
    role: Literal["training_seed", "selected_seed_rerun"] = "training_seed",
    ordinal: int = 0,
) -> _ResultCase:
    seed = 20_260_719 if ordinal in {0, 3} else 20_260_719 + ordinal
    binding = child_result.ChildResultBinding(
        role=role,
        ordinal=ordinal,
        seed=seed,
        registration_head_commit="a" * 40,
        implementation_commit="b" * 40,
        registration_sha256="c" * 64,
        source_bundle_sha256="d" * 64,
    )
    safetensors, model_tensor_sha256 = _result_safetensors()
    history = _result_history(seed=seed, model_tensor_sha256=model_tensor_sha256)
    return _ResultCase(
        binding=binding,
        history=history,
        history_sha256=hashlib.sha256(_HISTORY_DOMAIN + history).hexdigest(),
        safetensors=safetensors,
        safetensors_sha256=hashlib.sha256(safetensors).hexdigest(),
        winner_epoch=7,
        model_tensor_sha256=model_tensor_sha256,
    )


def _write_result(path: Path, case: _ResultCase) -> None:
    child_result.write_registered_child_result(
        path,
        case.binding,
        history_json_bytes=case.history,
        history_sha256=case.history_sha256,
        safetensors_bytes=case.safetensors,
        safetensors_sha256=case.safetensors_sha256,
        winner_epoch=case.winner_epoch,
        model_tensor_sha256=case.model_tensor_sha256,
    )


_DEFAULT_RESULT_CASE = _result_case()
_REAL_LOAD_REGISTERED_CHILD_RESULT = child_result.load_registered_child_result
_REAL_VERIFY_VERIFIED_CHILD_RESULT = child_result.verify_verified_child_result
_SYNTHETIC_VERIFIED_CHILD_RESULT = object.__new__(child_result.VerifiedChildResult)


def _source_bundle_fd(
    payload: bytes = _TEST_SOURCE_BUNDLE,
    *,
    byte_count: int | None = None,
    mode: int = 0o400,
    seals: int = supervisor._SOURCE_BUNDLE_MEMFD_SEALS,
    read_only: bool = True,
    inheritable: bool = False,
) -> int:
    owner = os.memfd_create(
        "falsewake-test-source-bundle",
        os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING,
    )
    result = -1
    try:
        size = len(payload) if byte_count is None else byte_count
        os.ftruncate(owner, size)
        if payload:
            os.pwrite(owner, payload[:size], 0)
        os.fchmod(owner, mode)
        fcntl.fcntl(owner, fcntl.F_ADD_SEALS, seals)
        if read_only:
            result = os.open(
                f"/proc/self/fd/{owner}",
                os.O_RDONLY | os.O_CLOEXEC,
            )
        else:
            result = owner
            owner = -1
        os.set_inheritable(result, inheritable)
        return result
    except BaseException:
        if result >= 0:
            with suppress(OSError):
                os.close(result)
        raise
    finally:
        if owner >= 0:
            os.close(owner)


def _supervise_child(
    cpu_ids: tuple[int, int],
    activation_callback: supervisor._Activation,
    result_binding: child_result.ChildResultBinding = _DEFAULT_RESULT_CASE.binding,
    /,
    **kwargs: object,
) -> supervisor._SupervisedChildResult:
    source_bundle = _source_bundle_fd()
    try:
        return supervisor._supervise_child(
            cpu_ids,
            activation_callback,
            source_bundle,
            result_binding,
            **kwargs,  # type: ignore[arg-type]
        )
    finally:
        os.close(source_bundle)


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
                int,
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
        source_bundle_fd: int,
    ) -> int:
        self.spawn_calls.append(
            (
                executable_path,
                argv,
                dict(environment),
                file_actions,
                source_bundle_fd,
            )
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


def _result_test_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> supervisor._LaunchPlan:
    plan = _test_plan(tmp_path, monkeypatch)
    temporary = Path("/home/ubuntu/gitcode/.t") / (
        f"falsewake-supervisor-result-test-{uuid.uuid4().hex}"
    )
    temporary.mkdir(mode=0o700)
    scratch = temporary / "scratch"
    staging = temporary / "staging"
    environment = tuple(
        (key, os.fspath(scratch) if key == "TMPDIR" else value)
        for key, value in plan.environment_items
    )
    return replace(
        plan,
        temporary_root=os.fspath(temporary),
        scratch_root=os.fspath(scratch),
        staging_root=os.fspath(staging),
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

    def load_result(
        _scratch_directory: Path,
        _expected: child_result.ChildResultBinding,
        /,
    ) -> child_result.VerifiedChildResult:
        return _SYNTHETIC_VERIFIED_CHILD_RESULT

    def verify_result(
        result: child_result.VerifiedChildResult,
        /,
    ) -> None:
        assert result is _SYNTHETIC_VERIFIED_CHILD_RESULT

    monkeypatch.setattr(supervisor, "load_registered_child_result", load_result)
    monkeypatch.setattr(supervisor, "verify_verified_child_result", verify_result)


@pytest.fixture
def result_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[supervisor._LaunchPlan]:
    plan = _result_test_plan(tmp_path, monkeypatch)
    try:
        yield plan
    finally:
        for path in (plan.scratch_root, plan.staging_root, plan.temporary_root):
            if Path(path).exists():
                supervisor._remove_directory_tree(path)


@pytest.fixture
def registered_output_roots() -> Iterator[tuple[Path, Path]]:
    roots = (
        Path(supervisor._REGISTERED_PLAN.scratch_root),
        Path(supervisor._REGISTERED_PLAN.staging_root),
    )
    root_strings = _root_strings(roots)
    cleanup_roots = supervisor._cleanup_output_roots
    cleanup_roots(root_strings)
    try:
        yield roots
    finally:
        cleanup_roots(root_strings)


def _use_real_result_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        supervisor,
        "load_registered_child_result",
        _REAL_LOAD_REGISTERED_CHILD_RESULT,
    )
    monkeypatch.setattr(
        supervisor,
        "verify_verified_child_result",
        _REAL_VERIFY_VERIFIED_CHILD_RESULT,
    )


def _allocated_regular_bytes(path: Path) -> int:
    total = 0
    for child in path.iterdir():
        metadata = child.stat(follow_symlinks=False)
        assert stat.S_ISREG(metadata.st_mode)
        total += max(metadata.st_size, 512 * metadata.st_blocks)
    return total


def test_source_uses_only_stdlib_and_fixed_child_result_authority() -> None:
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
        "falsewake.experiment_002_child_result",
        "hashlib",
        "os",
        "pathlib",
        "signal",
        "socket",
        "stat",
        "threading",
        "time",
        "types",
        "typing",
    }
    source = SOURCE_PATH.read_text(encoding="utf-8")
    assert "subprocess" not in source
    assert "preexec_fn" not in source
    assert "numpy" not in source
    assert "torch" not in source
    assert supervisor.__all__ == ("Experiment002SupervisorError",)
    assert not hasattr(supervisor, "run_registered_experiment")


@pytest.mark.parametrize(
    "route",
    [supervisor._supervise_child_masked, supervisor._supervise_child],
)
def test_supervise_routes_require_exact_positional_result_binding(
    route: Callable[..., object],
) -> None:
    signature = inspect.signature(route)
    parameter = signature.parameters["result_binding"]
    assert parameter.kind is inspect.Parameter.POSITIONAL_ONLY
    assert parameter.default is inspect.Parameter.empty
    assert parameter.annotation in {
        child_result.ChildResultBinding,
        "ChildResultBinding",
    }


def test_registered_supervisor_wrapper_has_exact_closure_bound_surface() -> None:
    route = supervisor._supervise_registered_child
    signature = inspect.signature(route)
    parameters = tuple(signature.parameters.values())
    assert [parameter.name for parameter in parameters] == [
        "cpu_ids",
        "activation_callback",
        "source_bundle_fd",
        "result_binding",
    ]
    assert all(
        parameter.kind is inspect.Parameter.POSITIONAL_ONLY
        and parameter.default is inspect.Parameter.empty
        for parameter in parameters
    )
    assert route.__defaults__ is None
    assert route.__kwdefaults__ is None
    assert route.__name__ == "_supervise_registered_child"
    assert route.__module__ == supervisor.__name__
    closure = {
        name: cell.cell_contents
        for name, cell in zip(
            route.__code__.co_freevars,
            route.__closure__ or (),
            strict=True,
        )
    }
    assert closure["supervise"] is supervisor._supervise_child
    assert closure["plan"] is supervisor._REGISTERED_PLAN
    assert closure["limits"] is supervisor._REGISTERED_LIMITS
    assert closure["kernel"] is supervisor._REAL_KERNEL


@pytest.mark.parametrize("mutation", ["plan", "limits", "kwdefaults"])
def test_registered_supervisor_wrapper_rejects_mutable_authority_spoofs(
    mutation: str,
) -> None:
    plan = supervisor._REGISTERED_PLAN
    limits = supervisor._REGISTERED_LIMITS
    keyword_defaults = supervisor._supervise_child.__kwdefaults__
    assert keyword_defaults is not None
    original: object
    if mutation == "plan":
        original = plan.runner_path
        object.__setattr__(plan, "runner_path", "/tmp/forged-runner.py")
    elif mutation == "limits":
        original = limits.wall_nanoseconds
        object.__setattr__(limits, "wall_nanoseconds", 10**30)
    else:
        original = keyword_defaults["plan"]
        keyword_defaults["plan"] = object()
    try:
        with pytest.raises(
            supervisor.Experiment002SupervisorError,
            match="authority|plan|limits",
        ):
            supervisor._supervise_registered_child(
                (0, 1),
                lambda _channel, _pid: None,
                -1,
                _DEFAULT_RESULT_CASE.binding,
            )
    finally:
        if mutation == "plan":
            object.__setattr__(plan, "runner_path", original)
        elif mutation == "limits":
            object.__setattr__(limits, "wall_nanoseconds", original)
        else:
            keyword_defaults["plan"] = original


def test_registered_supervisor_wrapper_rejects_pre_call_route_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replacement_calls: list[object] = []

    def replacement(*_args: object, **_kwargs: object) -> object:
        replacement_calls.append(object())
        return object()

    monkeypatch.setattr(supervisor, "_supervise_child", replacement)
    with pytest.raises(
        supervisor.Experiment002SupervisorError,
        match="executable authority changed",
    ):
        supervisor._supervise_registered_child(
            (0, 1),
            lambda _channel, _pid: None,
            -1,
            _DEFAULT_RESULT_CASE.binding,
        )
    assert replacement_calls == []


def test_registered_production_constants_are_exact() -> None:
    assert supervisor._CHILD_SOURCE_BUNDLE_FD == 7
    assert supervisor._SAFE_SOURCE_FD_MINIMUM == 8
    assert supervisor._SOURCE_BUNDLE_BYTES_MAXIMUM == 256 << 20
    assert (
        supervisor._SOURCE_BUNDLE_MEMFD_SEALS
        == fcntl.F_SEAL_WRITE
        | fcntl.F_SEAL_GROW
        | fcntl.F_SEAL_SHRINK
        | fcntl.F_SEAL_SEAL
    )
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


@pytest.mark.parametrize(
    "route",
    [
        supervisor._prepare_registered_experiment_staging,
        supervisor._cleanup_registered_experiment_output_roots,
    ],
)
def test_registered_output_lifecycle_routes_have_exact_zero_argument_signatures(
    route: Callable[[], None],
) -> None:
    signature = inspect.signature(route)
    assert tuple(signature.parameters.values()) == ()
    assert signature.return_annotation in {None, "None"}
    assert route.__defaults__ is None
    assert route.__kwdefaults__ is None
    with pytest.raises(TypeError):
        cast(Callable[[object], None], route)(supervisor._REGISTERED_PLAN)
    with pytest.raises(TypeError):
        cast(Callable[..., None], route)(plan=supervisor._REGISTERED_PLAN)


def test_registered_staging_prepare_and_cleanup_are_fixed_and_idempotent(
    registered_output_roots: tuple[Path, Path],
) -> None:
    scratch, staging = registered_output_roots

    supervisor._prepare_registered_experiment_staging()

    assert not os.path.lexists(scratch)
    assert staging.is_dir()
    assert stat.S_IMODE(staging.stat().st_mode) == 0o700
    assert list(staging.iterdir()) == []
    with pytest.raises(
        supervisor.Experiment002SupervisorError,
        match="already exists",
    ):
        supervisor._prepare_registered_experiment_staging()
    assert list(staging.iterdir()) == []

    supervisor._cleanup_registered_experiment_output_roots()
    assert not os.path.lexists(scratch)
    assert not os.path.lexists(staging)

    supervisor._cleanup_registered_experiment_output_roots()
    assert not os.path.lexists(scratch)
    assert not os.path.lexists(staging)


@pytest.mark.parametrize("existing_index", [0, 1])
def test_registered_staging_prepare_rejects_any_existing_output_root(
    registered_output_roots: tuple[Path, Path],
    existing_index: int,
) -> None:
    scratch, staging = registered_output_roots
    existing = registered_output_roots[existing_index]
    existing.mkdir(mode=0o700)
    marker = existing / "owned"
    marker.write_bytes(b"preserve")

    with pytest.raises(
        supervisor.Experiment002SupervisorError,
        match="already exists",
    ):
        supervisor._prepare_registered_experiment_staging()

    assert marker.read_bytes() == b"preserve"
    if existing == scratch:
        assert not os.path.lexists(staging)
    else:
        assert not os.path.lexists(scratch)


def test_registered_cleanup_handles_replaced_symlink_deep_and_sparse_roots(
    registered_output_roots: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    scratch, staging = registered_output_roots
    victim = tmp_path / "victim"
    victim.mkdir()
    marker = victim / "marker"
    marker.write_bytes(b"preserve")

    supervisor._prepare_registered_experiment_staging()
    moved_staging = tmp_path / "moved-staging"
    staging.rename(moved_staging)
    staging.mkdir(mode=0o700)
    cursor = staging
    for index in range(180):
        cursor /= f"d{index:03d}"
        cursor.mkdir()
    with (cursor / "sparse").open("wb") as stream:
        stream.truncate(4_294_967_297)
    (cursor / "external-link").symlink_to(victim, target_is_directory=True)
    scratch.symlink_to(victim, target_is_directory=True)

    supervisor._cleanup_registered_experiment_output_roots()

    assert not os.path.lexists(scratch)
    assert not os.path.lexists(staging)
    assert moved_staging.is_dir()
    assert marker.read_bytes() == b"preserve"


def test_registered_output_lifecycle_captures_paths_and_routes_at_import(
    registered_output_roots: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch, staging = registered_output_roots
    fake_temporary = tmp_path / "fake-temporary"
    fake_temporary.mkdir()
    fake_scratch = fake_temporary / "scratch"
    fake_staging = fake_temporary / "staging"
    original_plan = supervisor._REGISTERED_PLAN
    fake_environment = tuple(
        (key, os.fspath(fake_scratch) if key == "TMPDIR" else value)
        for key, value in original_plan.environment_items
    )
    monkeypatch.setattr(
        supervisor,
        "_REGISTERED_PLAN",
        replace(
            original_plan,
            temporary_root=os.fspath(fake_temporary),
            scratch_root=os.fspath(fake_scratch),
            staging_root=os.fspath(fake_staging),
            environment_items=fake_environment,
        ),
    )
    monkeypatch.setattr(supervisor, "_SCRATCH_ROOT", os.fspath(fake_scratch))
    monkeypatch.setattr(supervisor, "_STAGING_ROOT", os.fspath(fake_staging))

    def unexpected_route(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("registered lifecycle consulted a mutable global route")

    monkeypatch.setattr(supervisor, "_require_output_root_absent", unexpected_route)
    monkeypatch.setattr(supervisor, "_create_fresh_directory", unexpected_route)
    monkeypatch.setattr(supervisor, "_cleanup_output_roots", unexpected_route)

    supervisor._prepare_registered_experiment_staging()
    assert not os.path.lexists(scratch)
    assert staging.is_dir()
    assert not os.path.lexists(fake_scratch)
    assert not os.path.lexists(fake_staging)

    supervisor._cleanup_registered_experiment_output_roots()
    assert not os.path.lexists(scratch)
    assert not os.path.lexists(staging)


def test_registered_cleanup_attempts_both_roots_and_aggregates_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch = supervisor._REGISTERED_PLAN.scratch_root
    staging = supervisor._REGISTERED_PLAN.staging_root
    supervisor._cleanup_output_roots((scratch, staging))
    calls: list[str] = []
    staging_error = PermissionError("staging cleanup failed")
    scratch_error = OSError("scratch cleanup failed")

    def fail_cleanup(path: str) -> None:
        calls.append(path)
        if path == staging:
            raise staging_error
        if path == scratch:
            raise scratch_error
        raise AssertionError("unexpected cleanup path")

    monkeypatch.setattr(supervisor, "_remove_directory_tree", fail_cleanup)

    with pytest.raises(
        supervisor.Experiment002SupervisorError,
        match="failed to clean supervised output roots",
    ) as caught:
        supervisor._cleanup_registered_experiment_output_roots()

    assert calls == [staging, scratch]
    assert caught.value.__cause__ is staging_error


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


def test_source_bundle_descriptor_is_exact_and_offset_independent() -> None:
    descriptor = _source_bundle_fd()
    try:
        os.lseek(descriptor, 7, os.SEEK_SET)
        supervisor._require_source_bundle_descriptor(descriptor)
        assert os.lseek(descriptor, 0, os.SEEK_CUR) == 7
        metadata = os.fstat(descriptor)
        assert stat.S_ISREG(metadata.st_mode)
        assert metadata.st_nlink == 0
        assert stat.S_IMODE(metadata.st_mode) == 0o400
        assert fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) == (
            supervisor._SOURCE_BUNDLE_MEMFD_SEALS
        )
        assert fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE == (os.O_RDONLY)
    finally:
        os.close(descriptor)


def test_source_bundle_full_container_bound_is_inclusive() -> None:
    descriptor = _source_bundle_fd(byte_count=supervisor._SOURCE_BUNDLE_BYTES_MAXIMUM)
    try:
        supervisor._require_source_bundle_descriptor(descriptor)
        assert os.fstat(descriptor).st_size == 256 << 20
    finally:
        os.close(descriptor)


def test_source_bundle_reopen_has_an_independent_open_file_description() -> None:
    descriptor = _source_bundle_fd()
    independent = -1
    try:
        os.lseek(descriptor, 7, os.SEEK_SET)
        independent = supervisor._open_independent_source_bundle_descriptor(descriptor)
        assert independent >= supervisor._SAFE_SOURCE_FD_MINIMUM
        assert os.lseek(independent, 0, os.SEEK_CUR) == 0
        assert os.read(independent, 1) == _TEST_SOURCE_BUNDLE[:1]
        assert os.lseek(independent, 0, os.SEEK_CUR) == 1
        assert os.lseek(descriptor, 0, os.SEEK_CUR) == 7
        assert os.lseek(independent, 3, os.SEEK_SET) == 3
        assert os.read(independent, 1) == _TEST_SOURCE_BUNDLE[3:4]
        assert os.lseek(descriptor, 0, os.SEEK_CUR) == 7
        supervisor._require_matching_source_bundle_descriptors(
            descriptor,
            independent,
        )
    finally:
        if independent >= 0:
            os.close(independent)
        os.close(descriptor)


def test_source_bundle_match_rejects_a_different_sealed_memfd() -> None:
    descriptor = _source_bundle_fd()
    different_caller = _source_bundle_fd()
    different = -1
    try:
        different = supervisor._open_independent_source_bundle_descriptor(
            different_caller
        )
        with pytest.raises(
            supervisor.Experiment002SupervisorError,
            match="does not exactly match caller",
        ):
            supervisor._require_matching_source_bundle_descriptors(
                descriptor,
                different,
            )
    finally:
        if different >= 0:
            os.close(different)
        os.close(different_caller)
        os.close(descriptor)


@pytest.mark.parametrize(
    "missing_seal",
    [
        fcntl.F_SEAL_WRITE,
        fcntl.F_SEAL_GROW,
        fcntl.F_SEAL_SHRINK,
        fcntl.F_SEAL_SEAL,
    ],
)
def test_source_bundle_rejects_each_missing_seal(missing_seal: int) -> None:
    descriptor = _source_bundle_fd(
        seals=supervisor._SOURCE_BUNDLE_MEMFD_SEALS & ~missing_seal
    )
    try:
        with pytest.raises(
            supervisor.Experiment002SupervisorError,
            match="source bundle.*seals",
        ):
            supervisor._require_source_bundle_descriptor(descriptor)
    finally:
        os.close(descriptor)


@pytest.mark.parametrize(
    "kind",
    [
        "zero",
        "oversized",
        "wrong-mode",
        "writable",
        "inheritable",
        "closed",
        "disk",
        "pipe",
    ],
)
def test_source_bundle_rejects_malformed_descriptor_geometry(
    tmp_path: Path,
    kind: str,
) -> None:
    cleanup: list[int] = []
    if kind == "zero":
        descriptor = _source_bundle_fd(payload=b"", byte_count=0)
    elif kind == "oversized":
        descriptor = _source_bundle_fd(
            byte_count=supervisor._SOURCE_BUNDLE_BYTES_MAXIMUM + 1
        )
    elif kind == "wrong-mode":
        descriptor = _source_bundle_fd(mode=0o440)
    elif kind == "writable":
        descriptor = _source_bundle_fd(read_only=False)
    elif kind == "inheritable":
        descriptor = _source_bundle_fd(inheritable=True)
    elif kind == "closed":
        descriptor = _source_bundle_fd()
        os.close(descriptor)
    elif kind == "disk":
        path = tmp_path / "bundle"
        path.write_bytes(_TEST_SOURCE_BUNDLE)
        path.chmod(0o400)
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    else:
        descriptor, write_descriptor = os.pipe2(os.O_CLOEXEC)
        cleanup.append(write_descriptor)
    if kind != "closed":
        cleanup.append(descriptor)
    try:
        with pytest.raises(
            supervisor.Experiment002SupervisorError,
            match="source bundle",
        ):
            supervisor._require_source_bundle_descriptor(descriptor)
    finally:
        for opened in cleanup:
            os.close(opened)


@pytest.mark.parametrize("source_bundle_fd", [7, 12])
def test_file_actions_reject_source_bundle_low_fd_or_collision(
    source_bundle_fd: int,
) -> None:
    with pytest.raises(
        supervisor.Experiment002SupervisorError,
        match="safe|distinct",
    ):
        supervisor._build_file_actions(
            parent_channel_fd=3,
            child_channel_fd=8,
            null_fd=9,
            runner_fd=10,
            python_fd=11,
            taskset_exec_fd=12,
            source_bundle_fd=source_bundle_fd,
        )


def test_file_actions_are_the_exact_fixed_descriptor_layout() -> None:
    actions = supervisor._build_file_actions(
        parent_channel_fd=3,
        child_channel_fd=8,
        null_fd=9,
        runner_fd=10,
        python_fd=11,
        taskset_exec_fd=12,
        source_bundle_fd=13,
    )
    assert actions == (
        (os.POSIX_SPAWN_DUP2, 9, 0),
        (os.POSIX_SPAWN_DUP2, 9, 1),
        (os.POSIX_SPAWN_DUP2, 9, 2),
        (os.POSIX_SPAWN_DUP2, 8, 3),
        (os.POSIX_SPAWN_DUP2, 10, 4),
        (os.POSIX_SPAWN_DUP2, 11, 5),
        (os.POSIX_SPAWN_DUP2, 12, 6),
        (os.POSIX_SPAWN_DUP2, 13, 7),
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
            source_bundle_fd=12,
        )


def test_source_bundle_safe_duplication_fails_under_low_nofile_limit() -> None:
    source_bundle = _source_bundle_fd()
    original_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    if original_limit[1] < supervisor._SAFE_SOURCE_FD_MINIMUM:
        os.close(source_bundle)
        pytest.skip("hard RLIMIT_NOFILE is below the fixed source range")
    try:
        resource.setrlimit(
            resource.RLIMIT_NOFILE,
            (supervisor._SAFE_SOURCE_FD_MINIMUM, original_limit[1]),
        )
        with pytest.raises(
            supervisor.Experiment002SupervisorError,
            match="duplicated safely",
        ):
            supervisor._duplicate_safe_descriptor(source_bundle)
        assert os.fstat(source_bundle).st_size == len(_TEST_SOURCE_BUNDLE)
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, original_limit)
        os.close(source_bundle)


def test_source_bundle_independent_reopen_fails_closed_under_low_nofile_limit() -> None:
    source_bundle = _source_bundle_fd()
    original_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    if original_limit[1] < supervisor._SAFE_SOURCE_FD_MINIMUM:
        os.close(source_bundle)
        pytest.skip("hard RLIMIT_NOFILE is below the fixed source range")
    try:
        resource.setrlimit(
            resource.RLIMIT_NOFILE,
            (supervisor._SAFE_SOURCE_FD_MINIMUM, original_limit[1]),
        )
        with pytest.raises(
            supervisor.Experiment002SupervisorError,
            match="independent|duplicated safely",
        ):
            supervisor._open_independent_source_bundle_descriptor(source_bundle)
        assert os.fstat(source_bundle).st_size == len(_TEST_SOURCE_BUNDLE)
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, original_limit)
        os.close(source_bundle)


def test_safe_cloexec_source_is_duplicated_to_child_fd3_across_exec() -> None:
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    null_fd = os.open(os.devnull, os.O_RDWR | os.O_CLOEXEC)
    source_bundle_fd = _source_bundle_fd()
    child_source = supervisor._duplicate_safe_descriptor(child.fileno())
    null_source = supervisor._duplicate_safe_descriptor(null_fd)
    runner_source = supervisor._duplicate_safe_descriptor(null_fd)
    bundle_source = supervisor._open_independent_source_bundle_descriptor(
        source_bundle_fd
    )
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
        assert child_source >= 8 and null_source >= 8
        assert not os.get_inheritable(child_source)
        assert not os.get_inheritable(null_source)
        actions = supervisor._build_file_actions(
            parent_channel_fd=parent.fileno(),
            child_channel_fd=child_source,
            null_fd=null_source,
            runner_fd=runner_source,
            python_fd=python_source,
            taskset_exec_fd=executable_source,
            source_bundle_fd=bundle_source,
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
            bundle_source,
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
        os.close(source_bundle_fd)
        os.close(child_source)
        os.close(null_source)
        os.close(runner_source)
        os.close(bundle_source)
        os.close(python_source)
        os.close(executable_source)
        os.close(extra)


def test_fd7_bundle_survives_real_taskset_python_runner_bootstrap_and_closes(
    tmp_path: Path,
) -> None:
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < 2:
        pytest.skip("live registered taskset chain requires two logical CPUs")
    cpu_ids = (allowed[0], allowed[1])
    payload = b"sealed FD7 live bootstrap probe\x00\xff"
    source_bundle = _source_bundle_fd(payload)
    os.lseek(source_bundle, len(payload), os.SEEK_SET)
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    parent.settimeout(10.0)
    null_fd = os.open(os.devnull, os.O_RDWR | os.O_CLOEXEC)
    extra = supervisor._duplicate_safe_descriptor(null_fd, minimum=32)
    os.set_inheritable(extra, True)
    runner_path = tmp_path / "fd7-runner.py"
    runner_path.write_text(
        "import fcntl,os,stat\n"
        f"_payload={payload!r}\n"
        f"_extra={extra}\n"
        "_fd=7\n"
        "_meta=os.fstat(_fd)\n"
        "_first=os.read(7,1)\n"
        "_remaining=os.read(_fd,len(_payload))\n"
        "_ok=(stat.S_ISREG(_meta.st_mode) and _meta.st_nlink==0 "
        "and stat.S_IMODE(_meta.st_mode)==0o400 "
        "and _first+_remaining==_payload "
        "and fcntl.fcntl(_fd,fcntl.F_GET_SEALS)=="
        f"{supervisor._SOURCE_BUNDLE_MEMFD_SEALS} "
        "and fcntl.fcntl(_fd,fcntl.F_GETFL)&os.O_ACCMODE==os.O_RDONLY "
        "and not os.get_inheritable(3) "
        "and not os.get_inheritable(_fd))\n"
        "try:\n os.fstat(_extra)\n"
        "except OSError:\n pass\n"
        "else:\n _ok=False\n"
        "os.close(_fd)\n"
        "try:\n os.fstat(_fd)\n"
        "except OSError:\n pass\n"
        "else:\n _ok=False\n"
        "os.write(3,b'fd7-bootstrap-ok' if _ok else b'fd7-bootstrap-bad')\n",
        encoding="utf-8",
    )
    runner_master = supervisor._open_validated_launch_snapshot(
        os.fspath(runner_path),
        executable=False,
    )
    python_master = supervisor._open_validated_launch_snapshot(
        supervisor._PYTHON_RESOLVED_EXECUTABLE,
        executable=True,
        expected_sha256=supervisor._PYTHON_RESOLVED_SHA256,
    )
    taskset_master = supervisor._open_validated_launch_snapshot(
        supervisor._TASKSET_EXECUTABLE,
        executable=True,
        expected_sha256=supervisor._TASKSET_SHA256,
    )
    child_source = supervisor._duplicate_safe_descriptor(child.fileno())
    null_source = supervisor._duplicate_safe_descriptor(null_fd)
    runner_source = supervisor._duplicate_safe_descriptor(runner_master)
    python_source = supervisor._duplicate_safe_descriptor(python_master)
    taskset_source = supervisor._duplicate_safe_descriptor(taskset_master)
    bundle_source = supervisor._open_independent_source_bundle_descriptor(source_bundle)
    pid = -1
    try:
        actions = supervisor._build_file_actions(
            parent_channel_fd=parent.fileno(),
            child_channel_fd=child_source,
            null_fd=null_source,
            runner_fd=runner_source,
            python_fd=python_source,
            taskset_exec_fd=taskset_source,
            source_bundle_fd=bundle_source,
        )
        pid = supervisor._raw_posix_spawn(
            f"/proc/self/fd/{supervisor._CHILD_TASKSET_FD}",
            supervisor._pinned_spawn_argv(supervisor._REGISTERED_PLAN, cpu_ids),
            dict(supervisor._REGISTERED_ENVIRONMENT_ITEMS),
            actions,
            bundle_source,
        )
        child.close()
        assert parent.recv(64) == b"fd7-bootstrap-ok"
        waited, status_value = os.waitpid(pid, 0)
        pid = -1
        assert waited > 0
        assert os.WIFEXITED(status_value)
        assert os.WEXITSTATUS(status_value) == 0
        assert os.lseek(source_bundle, 0, os.SEEK_CUR) == len(payload)
    finally:
        if pid > 0:
            with suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
        parent.close()
        child.close()
        for descriptor in (
            null_fd,
            extra,
            source_bundle,
            runner_master,
            python_master,
            taskset_master,
            child_source,
            null_source,
            runner_source,
            python_source,
            taskset_source,
            bundle_source,
        ):
            os.close(descriptor)


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


def test_supervise_nominal_spawn_removes_scratch_and_keeps_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _test_plan(tmp_path, monkeypatch)
    _prepare_staging(plan)
    kernel = FakeKernel(times=[10, 20, 21, 22])
    activations: list[tuple[int, int]] = []

    def activate(channel: socket.socket, pid: int) -> None:
        activations.append((channel.family, pid))

    result = _supervise_child(
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
        verified_child_result=_SYNTHETIC_VERIFIED_CHILD_RESULT,
    )
    assert activations == [(socket.AF_UNIX, kernel.pid)]
    assert kernel.affinity_calls == [kernel.pid]
    assert len(kernel.spawn_calls) == 1
    executable_path, argv, environment, actions, source_bundle_fd = kernel.spawn_calls[
        0
    ]
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
    assert actions[-1] == (
        os.POSIX_SPAWN_DUP2,
        source_bundle_fd,
        supervisor._CHILD_SOURCE_BUNDLE_FD,
    )
    with pytest.raises(OSError) as caught:
        os.fstat(source_bundle_fd)
    assert caught.value.errno == errno.EBADF
    assert not Path(plan.scratch_root).exists()
    assert Path(plan.staging_root).is_dir()
    supervisor._remove_directory_tree(plan.staging_root)


def test_real_child_result_survives_secure_scratch_deletion_and_is_accounted(
    result_plan: supervisor._LaunchPlan,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_real_result_routes(monkeypatch)
    _prepare_staging(result_plan)
    kernel = FakeKernel(times=[10, 20, 21, 22])
    accounted_bytes: list[int] = []

    def activate(_channel: socket.socket, _pid: int) -> None:
        scratch = Path(result_plan.scratch_root)
        _write_result(scratch, _DEFAULT_RESULT_CASE)
        accounted_bytes.append(_allocated_regular_bytes(scratch))

    result = _supervise_child(
        (2, 7),
        activate,
        _DEFAULT_RESULT_CASE.binding,
        plan=result_plan,
        limits=_limits(output=2_000_000),
        kernel=kernel,
    )
    assert type(result.verified_child_result) is child_result.VerifiedChildResult
    _REAL_VERIFY_VERIFIED_CHILD_RESULT(result.verified_child_result)
    assert result.verified_child_result.binding == _DEFAULT_RESULT_CASE.binding
    assert (
        result.verified_child_result.canonical_history_bytes
        == _DEFAULT_RESULT_CASE.history
    )
    assert result.verified_child_result.safetensors_bytes == (
        _DEFAULT_RESULT_CASE.safetensors
    )
    assert accounted_bytes == [result.output_and_scratch_bytes]
    assert not Path(result_plan.scratch_root).exists()
    assert Path(result_plan.staging_root).is_dir()


@pytest.mark.parametrize(
    "failure",
    ["missing", "partial", "extra", "tampered", "mismatched"],
)
def test_exit_zero_invalid_child_result_fails_closed_and_clears_roots(
    result_plan: supervisor._LaunchPlan,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    _use_real_result_routes(monkeypatch)
    _prepare_staging(result_plan)
    staging = Path(result_plan.staging_root)
    (staging / "prior-result").write_bytes(b"must be cleared")
    kernel = FakeKernel(times=[0, 1, 2, 3])

    def activate(_channel: socket.socket, _pid: int) -> None:
        scratch = Path(result_plan.scratch_root)
        if failure == "missing":
            return
        if failure == "partial":
            (scratch / child_result._HISTORY_FILENAME).write_bytes(
                _DEFAULT_RESULT_CASE.history
            )
            return
        if failure == "mismatched":
            _write_result(scratch, _result_case(ordinal=1))
            return
        _write_result(scratch, _DEFAULT_RESULT_CASE)
        if failure == "extra":
            (scratch / "unexpected-result").write_bytes(b"extra")
            return
        target = scratch / child_result._SAFETENSORS_FILENAME
        payload = bytearray(target.read_bytes())
        payload[-1] ^= 1
        target.write_bytes(payload)

    with pytest.raises(child_result.Experiment002ChildResultError):
        _supervise_child(
            (2, 7),
            activate,
            _DEFAULT_RESULT_CASE.binding,
            plan=result_plan,
            limits=_limits(output=2_000_000),
            kernel=kernel,
        )
    assert not Path(result_plan.scratch_root).exists()
    assert staging.is_dir()
    assert list(staging.iterdir()) == []


def test_success_order_is_monitor_join_account_load_verify_account_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _test_plan(tmp_path, monkeypatch)
    _prepare_staging(plan)
    kernel = FakeKernel(times=[0, 1, 2, 3])
    events: list[str] = []
    original_account = supervisor._account_pinned_roots
    original_join = supervisor._join_activation_thread
    original_cleanup = supervisor._cleanup_pinned_scratch
    loaded_bindings: list[child_result.ChildResultBinding] = []

    def account(
        roots: tuple[supervisor._PinnedRoot, supervisor._PinnedRoot],
        maximum_bytes: int,
    ) -> int:
        events.append("account")
        return original_account(roots, maximum_bytes)

    def join(thread: threading.Thread) -> None:
        events.append("join")
        original_join(thread)

    def load(
        _scratch_directory: Path,
        expected: child_result.ChildResultBinding,
        /,
    ) -> child_result.VerifiedChildResult:
        events.append("load")
        loaded_bindings.append(expected)
        return _SYNTHETIC_VERIFIED_CHILD_RESULT

    def verify(_result: child_result.VerifiedChildResult, /) -> None:
        events.append("verify")

    def cleanup(root: supervisor._PinnedRoot) -> None:
        events.append("cleanup")
        original_cleanup(root)

    monkeypatch.setattr(supervisor, "_account_pinned_roots", account)
    monkeypatch.setattr(supervisor, "_join_activation_thread", join)
    monkeypatch.setattr(supervisor, "load_registered_child_result", load)
    monkeypatch.setattr(supervisor, "verify_verified_child_result", verify)
    monkeypatch.setattr(supervisor, "_cleanup_pinned_scratch", cleanup)
    result = _supervise_child(
        (2, 7),
        lambda _channel, _pid: None,
        plan=plan,
        limits=_limits(),
        kernel=kernel,
    )
    assert events[:2] == ["account", "account"]
    assert events[-6:] == [
        "join",
        "account",
        "load",
        "verify",
        "account",
        "cleanup",
    ]
    assert len(loaded_bindings) == 1
    assert loaded_bindings[0] == _DEFAULT_RESULT_CASE.binding
    assert loaded_bindings[0] is not _DEFAULT_RESULT_CASE.binding
    assert result.verified_child_result is _SYNTHETIC_VERIFIED_CHILD_RESULT
    assert not Path(plan.scratch_root).exists()
    supervisor._remove_directory_tree(plan.staging_root)


def test_supervision_reuses_issuer_bundle_without_owning_it_and_closes_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _test_plan(tmp_path, monkeypatch)
    _prepare_staging(plan)
    source_bundle = _source_bundle_fd()
    os.lseek(source_bundle, len(_TEST_SOURCE_BUNDLE), os.SEEK_SET)
    try:
        for ordinal in range(2):
            kernel = FakeKernel(
                pid=41_001 + ordinal,
                times=[10 * ordinal, 10 * ordinal + 1, 10 * ordinal + 2],
            )
            supervisor._supervise_child(
                (2, 7),
                lambda _channel, _pid: None,
                source_bundle,
                _DEFAULT_RESULT_CASE.binding,
                plan=plan,
                limits=_limits(),
                kernel=kernel,
            )
            assert os.fstat(source_bundle).st_size == len(_TEST_SOURCE_BUNDLE)
            assert os.lseek(source_bundle, 0, os.SEEK_CUR) == len(_TEST_SOURCE_BUNDLE)
            safe_source = kernel.spawn_calls[0][-1]
            assert safe_source != source_bundle
            with pytest.raises(OSError) as caught:
                os.fstat(safe_source)
            assert caught.value.errno == errno.EBADF
            assert not Path(plan.scratch_root).exists()
    finally:
        os.close(source_bundle)
        supervisor._remove_directory_tree(plan.staging_root)


def test_real_results_reuse_fresh_scratch_automatically_and_keep_staging(
    result_plan: supervisor._LaunchPlan,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_real_result_routes(monkeypatch)
    _prepare_staging(result_plan)
    staging = Path(result_plan.staging_root)
    marker = staging / "seed-zero-retained"
    marker.write_bytes(b"trusted parent staging")
    cases = (_DEFAULT_RESULT_CASE, _result_case(ordinal=1))
    results: list[supervisor._SupervisedChildResult] = []

    for ordinal, case in enumerate(cases):
        kernel = FakeKernel(
            pid=41_001 + ordinal,
            times=[10 * ordinal, 10 * ordinal + 1, 10 * ordinal + 2],
        )

        def activate(
            _channel: socket.socket,
            _pid: int,
            *,
            current: _ResultCase = case,
        ) -> None:
            assert marker.read_bytes() == b"trusted parent staging"
            _write_result(Path(result_plan.scratch_root), current)

        results.append(
            _supervise_child(
                (2, 7),
                activate,
                case.binding,
                plan=result_plan,
                limits=_limits(output=2_000_000),
                kernel=kernel,
            )
        )
        assert not Path(result_plan.scratch_root).exists()
        assert marker.read_bytes() == b"trusted parent staging"

    assert [result.verified_child_result.seed for result in results] == [
        20_260_719,
        20_260_720,
    ]
    for result in results:
        _REAL_VERIFY_VERIFIED_CHILD_RESULT(result.verified_child_result)


def test_result_binding_exact_type_is_rejected_before_scratch_or_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BindingSubclass(child_result.ChildResultBinding):
        pass

    plan = _test_plan(tmp_path, monkeypatch)
    _prepare_staging(plan)
    kernel = FakeKernel()
    invalid_values = (object(), object.__new__(BindingSubclass))
    for value in invalid_values:
        with pytest.raises(TypeError, match="exact ChildResultBinding"):
            _supervise_child(
                (2, 7),
                lambda _channel, _pid: None,
                cast(child_result.ChildResultBinding, value),
                plan=plan,
                limits=_limits(),
                kernel=kernel,
            )
        assert kernel.spawn_calls == []
        assert not Path(plan.scratch_root).exists()
    assert Path(plan.staging_root).is_dir()
    supervisor._remove_directory_tree(plan.staging_root)


def test_invalid_bundle_fails_before_scratch_or_spawn_and_remains_caller_owned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _test_plan(tmp_path, monkeypatch)
    _prepare_staging(plan)
    kernel = FakeKernel()
    source_bundle = _source_bundle_fd(read_only=False)
    try:
        with pytest.raises(
            supervisor.Experiment002SupervisorError,
            match="source bundle.*read-only",
        ):
            supervisor._supervise_child(
                (2, 7),
                lambda _channel, _pid: None,
                source_bundle,
                _DEFAULT_RESULT_CASE.binding,
                plan=plan,
                limits=_limits(),
                kernel=kernel,
            )
        assert os.fstat(source_bundle).st_size == len(_TEST_SOURCE_BUNDLE)
        assert kernel.spawn_calls == []
        assert not Path(plan.scratch_root).exists()
        assert Path(plan.staging_root).is_dir()
    finally:
        os.close(source_bundle)
        supervisor._remove_directory_tree(plan.staging_root)


def test_bundle_source_independent_reopen_failure_cleans_every_owned_resource(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _test_plan(tmp_path, monkeypatch)
    _prepare_staging(plan)
    kernel = FakeKernel()
    source_bundle = _source_bundle_fd()
    original = supervisor._open_independent_source_bundle_descriptor

    def reopen(descriptor: int) -> int:
        if descriptor == source_bundle:
            raise supervisor.Experiment002SupervisorError(
                "synthetic source bundle independent reopen failure"
            )
        return original(descriptor)

    monkeypatch.setattr(
        supervisor,
        "_open_independent_source_bundle_descriptor",
        reopen,
    )
    try:
        with pytest.raises(
            supervisor.Experiment002SupervisorError,
            match="synthetic source bundle independent reopen",
        ):
            supervisor._supervise_child(
                (2, 7),
                lambda _channel, _pid: None,
                source_bundle,
                _DEFAULT_RESULT_CASE.binding,
                plan=plan,
                limits=_limits(),
                kernel=kernel,
            )
        assert os.fstat(source_bundle).st_size == len(_TEST_SOURCE_BUNDLE)
        assert kernel.spawn_calls == []
        assert not Path(plan.scratch_root).exists()
        assert Path(plan.staging_root).is_dir()
    finally:
        os.close(source_bundle)
        supervisor._remove_directory_tree(plan.staging_root)


def test_bundle_caller_swap_immediately_before_spawn_fails_revalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _test_plan(tmp_path, monkeypatch)
    _prepare_staging(plan)
    kernel = FakeKernel()
    source_bundle = _source_bundle_fd()
    replacement = _source_bundle_fd(b"different immutable source bundle\n")
    captured_sources: list[int] = []
    original_open = supervisor._open_independent_source_bundle_descriptor

    def capture_source(descriptor: int) -> int:
        result = original_open(descriptor)
        captured_sources.append(result)
        return result

    def swap_caller() -> int:
        os.dup2(replacement, source_bundle, inheritable=False)
        return 0

    monkeypatch.setattr(
        supervisor,
        "_open_independent_source_bundle_descriptor",
        capture_source,
    )
    kernel.monotonic_ns = swap_caller  # type: ignore[method-assign]
    try:
        with pytest.raises(
            supervisor.Experiment002SupervisorError,
            match="does not exactly match caller",
        ):
            supervisor._supervise_child(
                (2, 7),
                lambda _channel, _pid: None,
                source_bundle,
                _DEFAULT_RESULT_CASE.binding,
                plan=plan,
                limits=_limits(),
                kernel=kernel,
            )
        assert kernel.spawn_calls == []
        assert len(captured_sources) == 1
        with pytest.raises(OSError) as caught:
            os.fstat(captured_sources[0])
        assert caught.value.errno == errno.EBADF
        assert not Path(plan.scratch_root).exists()
        assert Path(plan.staging_root).is_dir()
    finally:
        os.close(replacement)
        os.close(source_bundle)
        supervisor._remove_directory_tree(plan.staging_root)


def test_scratch_root_replacement_during_load_is_detected_and_contained(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _test_plan(tmp_path, monkeypatch)
    _prepare_staging(plan)
    kernel = FakeKernel(times=[0, 1, 2, 3])
    hidden = Path(plan.scratch_root).with_name("hidden-original-scratch")

    def replace_during_load(
        scratch_directory: Path,
        _expected: child_result.ChildResultBinding,
        /,
    ) -> child_result.VerifiedChildResult:
        scratch_directory.rename(hidden)
        scratch_directory.mkdir(mode=0o700)
        return _SYNTHETIC_VERIFIED_CHILD_RESULT

    monkeypatch.setattr(
        supervisor,
        "load_registered_child_result",
        replace_during_load,
    )
    with pytest.raises(
        supervisor.Experiment002SupervisorError,
        match="identity changed",
    ):
        _supervise_child(
            (2, 7),
            lambda _channel, _pid: None,
            plan=plan,
            limits=_limits(),
            kernel=kernel,
        )
    assert not Path(plan.scratch_root).exists()
    assert not hidden.exists()
    assert Path(plan.staging_root).is_dir()
    assert list(Path(plan.staging_root).iterdir()) == []
    supervisor._remove_directory_tree(plan.staging_root)


def test_result_load_byte_mutation_breaks_terminal_account_equality(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _test_plan(tmp_path, monkeypatch)
    _prepare_staging(plan)
    staging = Path(plan.staging_root)
    (staging / "prior-result").write_bytes(b"clear on failure")
    kernel = FakeKernel(times=[0, 1, 2, 3])

    def mutate_during_load(
        scratch_directory: Path,
        _expected: child_result.ChildResultBinding,
        /,
    ) -> child_result.VerifiedChildResult:
        (scratch_directory / "late-result-byte").write_bytes(b"changed")
        return _SYNTHETIC_VERIFIED_CHILD_RESULT

    monkeypatch.setattr(
        supervisor,
        "load_registered_child_result",
        mutate_during_load,
    )
    with pytest.raises(
        supervisor.Experiment002SupervisorError,
        match="changed during child-result loading",
    ):
        _supervise_child(
            (2, 7),
            lambda _channel, _pid: None,
            plan=plan,
            limits=_limits(),
            kernel=kernel,
        )
    assert not Path(plan.scratch_root).exists()
    assert staging.is_dir()
    assert list(staging.iterdir()) == []
    supervisor._remove_directory_tree(plan.staging_root)


def test_verified_result_reverification_failure_uses_child_containment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _test_plan(tmp_path, monkeypatch)
    _prepare_staging(plan)
    staging = Path(plan.staging_root)
    (staging / "prior-result").write_bytes(b"clear on failure")
    kernel = FakeKernel(times=[0, 1, 2, 3])

    def fail_verification(
        _result: child_result.VerifiedChildResult,
        /,
    ) -> None:
        raise child_result.Experiment002ChildResultError(
            "synthetic result authority failure"
        )

    monkeypatch.setattr(
        supervisor,
        "verify_verified_child_result",
        fail_verification,
    )
    with pytest.raises(
        child_result.Experiment002ChildResultError,
        match="authority failure",
    ):
        _supervise_child(
            (2, 7),
            lambda _channel, _pid: None,
            plan=plan,
            limits=_limits(),
            kernel=kernel,
        )
    assert not Path(plan.scratch_root).exists()
    assert staging.is_dir()
    assert list(staging.iterdir()) == []
    supervisor._remove_directory_tree(plan.staging_root)


def test_inheritable_fd_snapshot_is_built_while_all_catchable_signals_are_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _test_plan(tmp_path, monkeypatch)
    _prepare_staging(plan)
    kernel = FakeKernel(times=[0, 1, 2, 3, 4])
    original = supervisor._build_file_actions
    original_match = supervisor._require_matching_source_bundle_descriptors
    observed_masks: list[set[int]] = []
    revalidation_masks: list[set[int]] = []

    def inspect_mask(
        *,
        parent_channel_fd: int,
        child_channel_fd: int,
        null_fd: int,
        runner_fd: int,
        python_fd: int,
        taskset_exec_fd: int,
        source_bundle_fd: int,
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
            source_bundle_fd=source_bundle_fd,
        )

    def inspect_match(caller_descriptor: int, independent_descriptor: int) -> None:
        revalidation_masks.append(
            {int(value) for value in signal.pthread_sigmask(signal.SIG_BLOCK, set())}
        )
        original_match(caller_descriptor, independent_descriptor)

    monkeypatch.setattr(supervisor, "_build_file_actions", inspect_mask)
    monkeypatch.setattr(
        supervisor,
        "_require_matching_source_bundle_descriptors",
        inspect_match,
    )
    _supervise_child(
        (2, 7),
        lambda _channel, _pid: None,
        plan=plan,
        limits=_limits(),
        kernel=kernel,
    )
    assert len(observed_masks) == 1
    assert supervisor._blockable_signals() <= observed_masks[0]
    assert len(revalidation_masks) == 1
    assert supervisor._blockable_signals() <= revalidation_masks[0]
    supervisor._remove_directory_tree(plan.staging_root)


@pytest.mark.parametrize("failure", ["activation", "nonzero", "resource", "spawn"])
def test_early_failures_never_invoke_child_result_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    plan = _test_plan(tmp_path, monkeypatch)
    _prepare_staging(plan)
    loader_calls: list[Path] = []

    def forbidden_loader(
        scratch_directory: Path,
        _expected: child_result.ChildResultBinding,
        /,
    ) -> child_result.VerifiedChildResult:
        loader_calls.append(scratch_directory)
        raise AssertionError("child-result loader was reached")

    monkeypatch.setattr(
        supervisor,
        "load_registered_child_result",
        forbidden_loader,
    )

    def activation(_channel: socket.socket, _pid: int) -> None:
        return None

    expected_exception: type[BaseException]
    if failure == "activation":
        kernel = FakeKernel(times=[0, 1, 2])

        def fail_activation(_channel: socket.socket, _pid: int) -> None:
            raise RuntimeError("synthetic activation failure")

        activation = fail_activation
        expected_exception = RuntimeError
    elif failure == "nonzero":
        kernel = FakeKernel(
            wait_results=[
                supervisor._WaitResult(0, 0, 0),
                supervisor._WaitResult(41_001, 9, 1),
            ],
            times=[0, 1, 2],
        )
        expected_exception = supervisor.Experiment002SupervisorError
    elif failure == "resource":
        kernel = FakeKernel(
            wait_results=[
                supervisor._WaitResult(0, 0, 0),
                supervisor._WaitResult(41_001, 9, 1),
            ],
            samples=[supervisor._ProcessSample(1_000_001, ())],
            times=[0, 1, 2],
        )
        expected_exception = supervisor.Experiment002SupervisorError
    else:
        kernel = FakeKernel(times=[0])

        def fail_spawn(
            executable_path: str,
            argv: tuple[str, ...],
            environment: Mapping[str, str],
            file_actions: tuple[supervisor._FileAction, ...],
            source_bundle_fd: int,
        ) -> int:
            del executable_path, argv, environment, file_actions, source_bundle_fd
            raise OSError("synthetic spawn failure")

        kernel.spawn = fail_spawn  # type: ignore[method-assign]
        expected_exception = OSError

    with pytest.raises(expected_exception):
        _supervise_child(
            (2, 7),
            activation,
            plan=plan,
            limits=_limits(),
            kernel=kernel,
        )
    assert loader_calls == []
    assert not Path(plan.scratch_root).exists()
    assert Path(plan.staging_root).is_dir()
    supervisor._remove_directory_tree(plan.staging_root)


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
        _supervise_child(
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
        _supervise_child(
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
        _supervise_child(
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
    result = _supervise_child(
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
    supervisor._remove_directory_tree(plan.staging_root)


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
        _supervise_child(
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
        _supervise_child(
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
        source_bundle_fd: int,
    ) -> int:
        del executable_path, argv, environment, file_actions, source_bundle_fd
        raise OSError("synthetic spawn failure")

    kernel.spawn = fail_spawn  # type: ignore[method-assign]
    with pytest.raises(OSError, match="synthetic"):
        _supervise_child(
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
        source_bundle_fd: int,
    ) -> int:
        del executable_path, argv, environment, file_actions, source_bundle_fd
        kernel.spawned = True
        return returned_pid

    kernel.spawn = invalid_spawn  # type: ignore[method-assign]
    with pytest.raises(supervisor.Experiment002SupervisorError, match=message):
        _supervise_child(
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
        source_bundle_fd: int,
    ) -> int:
        del executable_path, argv, environment, file_actions, source_bundle_fd
        kernel.spawned = True
        raise KeyboardInterrupt

    kernel.spawn = interrupted_spawn  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        _supervise_child(
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
        _supervise_child(
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
        _supervise_child(
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
        _supervise_child(
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
    _supervise_child(
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
    supervisor._remove_directory_tree(plan.staging_root)


def test_post_masked_parent_boundary_failure_cannot_escape_a_trusted_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    internal_result = supervisor._SupervisedChildResult(
        pid=41_001,
        cpu_ids=(2, 7),
        elapsed_nanoseconds=1,
        maximum_rss_bytes=1,
        output_and_scratch_bytes=1,
        verified_child_result=_SYNTHETIC_VERIFIED_CHILD_RESULT,
    )
    masked_bindings: list[child_result.ChildResultBinding] = []

    def masked(*args: object, **_kwargs: object) -> supervisor._SupervisedChildResult:
        assert len(args) == 4
        binding = args[3]
        assert type(binding) is child_result.ChildResultBinding
        masked_bindings.append(binding)
        return internal_result

    def fail_boundary(_frame: supervisor._SignalDispositionFrame) -> None:
        raise supervisor.Experiment002SupervisorError(
            "synthetic post-masked parent boundary failure"
        )

    monkeypatch.setattr(supervisor, "_supervise_child_masked", masked)
    monkeypatch.setattr(
        supervisor,
        "_require_safe_signal_dispositions_unchanged",
        fail_boundary,
    )
    source_bundle = _source_bundle_fd()
    escaped: list[supervisor._SupervisedChildResult] = []

    def invoke() -> None:
        escaped.append(
            supervisor._supervise_child(
                (2, 7),
                lambda _channel, _pid: None,
                source_bundle,
                _DEFAULT_RESULT_CASE.binding,
            )
        )

    try:
        with pytest.raises(
            supervisor.Experiment002SupervisorError,
            match="parent-boundary containment",
        ):
            invoke()
    finally:
        os.close(source_bundle)
    assert escaped == []
    assert len(masked_bindings) == 1
    assert masked_bindings[0] == _DEFAULT_RESULT_CASE.binding
    assert masked_bindings[0] is not _DEFAULT_RESULT_CASE.binding


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
    result = _supervise_child(
        (2, 7),
        lambda _channel, _pid: None,
        plan=plan,
        limits=_limits(wall=1_000_000_000, poll=99_999_999),
        kernel=kernel,
    )
    assert result.pid == kernel.pid
    assert kernel.sleep_calls == [pytest.approx(0.099999998)]
    supervisor._remove_directory_tree(plan.staging_root)


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
    _supervise_child(
        (2, 7),
        first_activation,
        plan=plan,
        limits=_limits(),
        kernel=first,
    )
    assert not Path(plan.scratch_root).exists()
    second = FakeKernel(times=[10, 11, 12, 13])

    def second_activation(_channel: socket.socket, _pid: int) -> None:
        assert marker.read_bytes() == b"verified seed result"

    result = _supervise_child(
        (2, 7),
        second_activation,
        plan=plan,
        limits=_limits(),
        kernel=second,
    )
    expected = max(marker.stat().st_size, 512 * marker.stat().st_blocks)
    assert result.output_and_scratch_bytes == expected
    supervisor._remove_directory_tree(plan.staging_root)


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
        _supervise_child(
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
        _supervise_child(
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
        source_bundle_fd: int,
    ) -> int:
        observed.update(
            executable_path=executable_path,
            argv=argv,
            environment=environment,
            file_actions=file_actions,
            source_bundle_fd=source_bundle_fd,
        )
        return 1234

    monkeypatch.setattr(supervisor, "_raw_posix_spawn", fake_spawn)
    environment = {"ONLY": "registered"}
    actions: tuple[supervisor._FileAction, ...] = ((os.POSIX_SPAWN_CLOSE, 9),)
    source_bundle = _source_bundle_fd()
    try:
        result = supervisor._RealKernel().spawn(
            "/proc/self/fd/6",
            ("/registered/taskset", "argument"),
            environment,
            actions,
            source_bundle,
        )
    finally:
        os.close(source_bundle)
    assert result == 1234
    assert observed == {
        "executable_path": "/proc/self/fd/6",
        "argv": ("/registered/taskset", "argument"),
        "environment": environment,
        "file_actions": actions,
        "source_bundle_fd": source_bundle,
    }
    assert observed["environment"] is not environment


def test_supervisor_requires_experiment_staging_before_scratch_or_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _test_plan(tmp_path, monkeypatch)
    kernel = FakeKernel()
    with pytest.raises(supervisor.Experiment002SupervisorError, match="staging"):
        _supervise_child(
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
