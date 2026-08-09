from __future__ import annotations

import builtins
import hashlib
import itertools
import os
import select
import shutil
import signal
import sys
import tempfile
import threading
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

import falsewake

_SOURCE = (
    Path(__file__).resolve().parents[1] / "src/falsewake/experiment_002_seed_worker.py"
)
_SCRATCH_ROOT = Path("/home/ubuntu/gitcode/.t")
_SEEDS = (20_260_719, 20_260_720, 20_260_721)
_HISTORY_DOMAIN = b"falsewake-exp002-history-v1\0"
_MODULE_COUNTER = itertools.count()


class _SyntheticFailure(RuntimeError):
    pass


@dataclass(slots=True)
class _World:
    trace: list[str]
    counts: Counter[str]
    arguments: dict[str, list[tuple[tuple[object, ...], dict[str, object]]]]
    fail_at: str | None
    first_snapshot: Any
    second_snapshot: Any
    registration: Any
    activation: Any
    corpus: Any
    normalization: Any
    validation_population: Any
    validation_inputs: Any
    training_inputs: Any
    executor: Any
    history: Any
    completed: Any
    checkpoint: Any
    cache: Any = None
    fail_after_write: bool = False
    writer_result: object = None

    def call(
        self,
        name: str,
        *args: object,
        result: object = None,
        **kwargs: object,
    ) -> object:
        self.trace.append(name)
        self.counts[name] += 1
        self.arguments[name].append((args, kwargs))
        occurrence = self.counts[name]
        if self.fail_at in {name, f"{name}:{occurrence}"}:
            raise _SyntheticFailure(name)
        return result


@pytest.fixture
def scratch_directory() -> Iterator[Path]:
    path = Path(
        tempfile.mkdtemp(
            prefix="falsewake-exp002-worker-test-",
            dir=_SCRATCH_ROOT,
        )
    )
    os.chmod(path, 0o700)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _fake_module(name: str, **attributes: object) -> ModuleType:
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _synthetic_modules(
    scratch_directory: Path,
    *,
    fail_at: str | None = None,
    role: object = "training_seed",
    seed: object = _SEEDS[0],
    ordinal: object = 0,
    child_pid: object | None = None,
    snapshot_change: str | None = None,
    terminal_change: str | None = None,
    writer_result: object = None,
) -> tuple[_World, dict[str, ModuleType], dict[str, type[object]]]:
    class Registration:
        def __init__(self) -> None:
            self.head_commit = "1" * 40
            self.implementation_commit = "2" * 40
            self.registration_sha256 = "3" * 64
            self.source_bundle_sha256 = "4" * 64

    class Activation:
        def __init__(self) -> None:
            self.role = role
            self.seed = seed
            self.ordinal = ordinal
            self.child_pid = os.getpid() if child_pid is None else child_pid

    @dataclass(frozen=True, slots=True)
    class Snapshot:
        manifest_path: Path
        pcm_cache_path: Path
        pcm_cache_byte_count: int
        pcm_cache_sha256: str
        normalization_path: Path
        normalization_byte_count: int
        normalization_sha256: str
        scratch_directory: Path

    @dataclass(frozen=True, slots=True)
    class ChildResultBinding:
        role: str
        ordinal: int
        seed: int
        registration_head_commit: str
        implementation_commit: str
        registration_sha256: str
        source_bundle_sha256: str

    @dataclass(frozen=True, slots=True)
    class NormalizationIdentity:
        byte_count: int
        sha256: str

        def __post_init__(self) -> None:
            world.call("normalization_identity", self.byte_count, self.sha256)

    @dataclass(frozen=True, slots=True)
    class PCMIdentity:
        byte_count: int
        sha256: str

        def __post_init__(self) -> None:
            world.call("pcm_identity", self.byte_count, self.sha256)

    class Evaluated:
        def __init__(
            self,
            *,
            seed: object,
            zero_based_epoch: object,
            model_tensor_sha256: object,
        ) -> None:
            self.seed = seed
            self.zero_based_epoch = zero_based_epoch
            self.model_tensor_sha256 = model_tensor_sha256

    class Completed:
        def __init__(
            self,
            *,
            seed: object,
            epoch_count: object,
            canonical_json_bytes: object,
            history_sha256: object,
            winner: object,
        ) -> None:
            self.seed = seed
            self.epoch_count = epoch_count
            self.canonical_json_bytes = canonical_json_bytes
            self.history_sha256 = history_sha256
            self.winner = winner

    class Checkpoint:
        def __init__(
            self,
            *,
            seed: object,
            zero_based_epoch: object,
            history_sha256: object,
            model_tensor_sha256: object,
            safetensors_bytes: object,
            safetensors_sha256: object,
            safetensors_byte_count: object,
        ) -> None:
            self.seed = seed
            self.zero_based_epoch = zero_based_epoch
            self.history_sha256 = history_sha256
            self.model_tensor_sha256 = model_tensor_sha256
            self.safetensors_bytes = safetensors_bytes
            self.safetensors_sha256 = safetensors_sha256
            self.safetensors_byte_count = safetensors_byte_count

    first_snapshot = Snapshot(
        manifest_path=scratch_directory / "manifest.jsonl",
        pcm_cache_path=scratch_directory / "pcm-cache.bin",
        pcm_cache_byte_count=2_995_776_948,
        pcm_cache_sha256="5" * 64,
        normalization_path=scratch_directory / "normalization.bin",
        normalization_byte_count=320,
        normalization_sha256="6" * 64,
        scratch_directory=scratch_directory,
    )
    second_snapshot = replace(first_snapshot)
    if snapshot_change == "pcm_cache_sha256":
        second_snapshot = replace(first_snapshot, pcm_cache_sha256="7" * 64)
    elif snapshot_change == "scratch_directory":
        second_snapshot = replace(
            first_snapshot,
            scratch_directory=scratch_directory / "changed",
        )
    elif snapshot_change == "alias":
        second_snapshot = first_snapshot

    history_bytes: Any = b'{"experiment":"002"}\n'
    history_sha256: Any = hashlib.sha256(_HISTORY_DOMAIN + history_bytes).hexdigest()
    safetensors_bytes: Any = b"synthetic-safetensors-payload"
    safetensors_sha256: Any = hashlib.sha256(safetensors_bytes).hexdigest()
    model_sha256: Any = "8" * 64
    winner_seed: Any = seed
    winner_epoch: Any = 17
    completed_seed: Any = seed
    completed_epoch_count: Any = 30
    checkpoint_seed: Any = seed
    checkpoint_epoch: Any = winner_epoch
    checkpoint_history_sha256: Any = history_sha256
    checkpoint_model_sha256: Any = model_sha256
    checkpoint_byte_count: Any = len(safetensors_bytes)

    if terminal_change == "history_bytes":
        history_bytes = b"not-canonical"
    elif terminal_change == "history_sha256":
        history_sha256 = "9" * 64
    elif terminal_change == "completed_seed":
        completed_seed = _SEEDS[1]
    elif terminal_change == "epoch_count_bool":
        completed_epoch_count = True
    elif terminal_change == "winner_seed_bool":
        winner_seed = True
    elif terminal_change == "winner_epoch_bool":
        winner_epoch = True
    elif terminal_change == "checkpoint_seed":
        checkpoint_seed = _SEEDS[1]
    elif terminal_change == "checkpoint_epoch":
        checkpoint_epoch = 18
    elif terminal_change == "checkpoint_history":
        checkpoint_history_sha256 = "a" * 64
    elif terminal_change == "checkpoint_model":
        checkpoint_model_sha256 = "b" * 64
    elif terminal_change == "checkpoint_bytes":
        safetensors_bytes = b"changed"
    elif terminal_change == "checkpoint_sha256":
        safetensors_sha256 = "c" * 64
    elif terminal_change == "checkpoint_byte_count_bool":
        checkpoint_byte_count = True

    registration = Registration()
    activation = Activation()
    corpus = object()
    normalization = object()
    validation_population = object()
    validation_inputs = object()
    training_inputs = object()
    executor = object()
    history = object()
    winner = Evaluated(
        seed=winner_seed,
        zero_based_epoch=winner_epoch,
        model_tensor_sha256=model_sha256,
    )
    completed = Completed(
        seed=completed_seed,
        epoch_count=completed_epoch_count,
        canonical_json_bytes=history_bytes,
        history_sha256=history_sha256,
        winner=winner,
    )
    checkpoint = Checkpoint(
        seed=checkpoint_seed,
        zero_based_epoch=checkpoint_epoch,
        history_sha256=checkpoint_history_sha256,
        model_tensor_sha256=checkpoint_model_sha256,
        safetensors_bytes=safetensors_bytes,
        safetensors_sha256=safetensors_sha256,
        safetensors_byte_count=checkpoint_byte_count,
    )
    world = _World(
        trace=[],
        counts=Counter(),
        arguments=defaultdict(list),
        fail_at=fail_at,
        first_snapshot=first_snapshot,
        second_snapshot=second_snapshot,
        registration=registration,
        activation=activation,
        corpus=corpus,
        normalization=normalization,
        validation_population=validation_population,
        validation_inputs=validation_inputs,
        training_inputs=training_inputs,
        executor=executor,
        history=history,
        completed=completed,
        checkpoint=checkpoint,
        writer_result=writer_result,
    )

    validation_view = object()
    training_view = object()

    class Cache:
        def __init__(self) -> None:
            self.validation = validation_view
            self.training = training_view
            self.closed = False

        @classmethod
        def open(cls, *args: object) -> Cache:
            cache = cls()
            world.cache = cache
            return world.call("cache_open", *args, result=cache)  # type: ignore[return-value]

        def __enter__(self) -> Cache:
            return world.call("cache_enter", self, result=self)  # type: ignore[return-value]

        def __exit__(
            self,
            exception_type: object,
            exception: object,
            traceback: object,
        ) -> None:
            self.closed = True
            world.trace.append("cache_exit")
            world.counts["cache_exit"] += 1

    def verify_activation(candidate_registration: object, candidate: object) -> None:
        assert candidate_registration is registration
        assert candidate is activation
        world.call("activation_verify", candidate_registration, candidate)
        if world.fail_after_write and world.counts["write"] == 1:
            raise _SyntheticFailure("post-write activation verification")

    def snapshot(candidate: object) -> object:
        assert candidate is registration
        result = first_snapshot if world.counts["snapshot"] == 0 else second_snapshot
        return world.call("snapshot", candidate, result=result)

    def load_corpus(path: object) -> object:
        return world.call("load_corpus", path, result=corpus)

    def load_normalization(path: object, identity: object) -> object:
        return world.call(
            "load_normalization",
            path,
            identity,
            result=normalization,
        )

    def materialize(*args: object) -> object:
        return world.call("materialize", *args, result=validation_population)

    def bind_validation(*args: object) -> object:
        return world.call("bind_validation", *args, result=validation_inputs)

    def bind_training(*args: object) -> object:
        return world.call("bind_training", *args, result=training_inputs)

    def create_executor(*args: object, **kwargs: object) -> object:
        return world.call(
            "create_executor",
            *args,
            **kwargs,
            result=executor,
        )

    def create_history(*args: object, **kwargs: object) -> object:
        return world.call("create_history", *args, **kwargs, result=history)

    def execute(candidate: object) -> object:
        epoch = world.counts["execute"]
        return world.call("execute", candidate, result=("handoff", epoch))

    def evaluate(*args: object) -> object:
        epoch = world.counts["evaluate"]
        return world.call("evaluate", *args, result=("evaluated", epoch))

    def consume(*args: object) -> object:
        epoch = world.counts["consume"]
        return world.call("consume", *args, result=("barrier", epoch))

    def advance(*args: object) -> None:
        world.call("advance", *args)

    def complete(*args: object) -> object:
        return world.call("complete", *args, result=completed)

    def verify_completed(candidate: object) -> None:
        world.call("verify_history", candidate)

    def serialize(candidate: object) -> object:
        return world.call("serialize", candidate, result=checkpoint)

    def verify_serialized(candidate: object) -> None:
        world.call("verify_checkpoint", candidate)

    def write(*args: object, **kwargs: object) -> object:
        return world.call("write", *args, **kwargs, result=world.writer_result)

    modules = {
        "falsewake.experiment_002_child_result": _fake_module(
            "falsewake.experiment_002_child_result",
            ChildResultBinding=ChildResultBinding,
            write_registered_child_result=write,
        ),
        "falsewake.experiment_002_coordinator": _fake_module(
            "falsewake.experiment_002_coordinator",
            REGISTERED_SEEDS=_SEEDS,
            VerifiedChildActivation=Activation,
            verify_verified_child_activation=verify_activation,
        ),
        "falsewake.experiment_002_data": _fake_module(
            "falsewake.experiment_002_data",
            load_registered_corpus=load_corpus,
        ),
        "falsewake.experiment_002_evidence": _fake_module(
            "falsewake.experiment_002_evidence",
            bind_registered_validation_inputs=bind_validation,
        ),
        "falsewake.experiment_002_normalization_artifact": _fake_module(
            "falsewake.experiment_002_normalization_artifact",
            NormalizationArtifactIdentity=NormalizationIdentity,
            load_normalization_artifact=load_normalization,
        ),
        "falsewake.experiment_002_pcm_cache": _fake_module(
            "falsewake.experiment_002_pcm_cache",
            Experiment002PCMCache=Cache,
            PCMCacheIdentity=PCMIdentity,
        ),
        "falsewake.experiment_002_registered_checkpoint": _fake_module(
            "falsewake.experiment_002_registered_checkpoint",
            RegisteredSerializedCheckpoint=Checkpoint,
            serialize_registered_checkpoint=serialize,
            verify_registered_serialized_checkpoint=verify_serialized,
        ),
        "falsewake.experiment_002_registered_evaluator": _fake_module(
            "falsewake.experiment_002_registered_evaluator",
            RegisteredEvaluatedEpoch=Evaluated,
            evaluate_registered_epoch=evaluate,
        ),
        "falsewake.experiment_002_registered_executor": _fake_module(
            "falsewake.experiment_002_registered_executor",
            advance_registered_training_executor=advance,
            complete_registered_training_executor=complete,
            create_registered_training_executor=create_executor,
            execute_registered_training_epoch=execute,
        ),
        "falsewake.experiment_002_registered_history": _fake_module(
            "falsewake.experiment_002_registered_history",
            RegisteredCompletedTrainingHistory=Completed,
            consume_registered_evaluated_epoch=consume,
            create_registered_training_history=create_history,
            verify_registered_completed_training_history=verify_completed,
        ),
        "falsewake.experiment_002_run_authority": _fake_module(
            "falsewake.experiment_002_run_authority",
            VerifiedRunRegistration=Registration,
            _registered_child_input_snapshot=snapshot,
            _RegisteredChildInputSnapshot=Snapshot,
        ),
        "falsewake.experiment_002_training_population": _fake_module(
            "falsewake.experiment_002_training_population",
            bind_registered_training_inputs=bind_training,
        ),
        "falsewake.experiment_002_validation": _fake_module(
            "falsewake.experiment_002_validation",
            materialize_registered_validation=materialize,
        ),
    }
    types: dict[str, type[object]] = {
        "Activation": Activation,
        "Cache": Cache,
        "Checkpoint": Checkpoint,
        "Completed": Completed,
        "Registration": Registration,
        "Snapshot": Snapshot,
    }
    return world, modules, types


def _load_worker(
    monkeypatch: pytest.MonkeyPatch,
    modules: dict[str, ModuleType],
    *,
    forbid_io: bool = False,
) -> ModuleType:
    source = _SOURCE.read_text(encoding="utf-8")
    code = compile(source, os.fspath(_SOURCE), "exec")
    name = f"_falsewake_seed_worker_test_{next(_MODULE_COUNTER)}"
    worker = ModuleType(name)
    worker.__file__ = os.fspath(_SOURCE)
    worker.__package__ = "falsewake"
    worker.__spec__ = None

    def forbidden(*args: object, **kwargs: object) -> Any:
        del args, kwargs
        raise AssertionError("seed-worker import attempted filesystem I/O")

    with monkeypatch.context() as context:
        for module_name, module in modules.items():
            context.setitem(sys.modules, module_name, module)
            context.setattr(
                falsewake,
                module_name.rsplit(".", maxsplit=1)[1],
                module,
                raising=False,
            )
        context.setitem(sys.modules, name, worker)
        if forbid_io:
            context.setattr(builtins, "open", forbidden)
            context.setattr(os, "open", forbidden)
            context.setattr(os, "lstat", forbidden)
            context.setattr(os, "fstat", forbidden)
            context.setattr(os, "listdir", forbidden)
        exec(code, worker.__dict__)
    sys.modules.pop(name, None)
    return worker


def _major_trace(world: _World) -> list[str]:
    names = {
        "advance",
        "bind_training",
        "bind_validation",
        "cache_enter",
        "cache_exit",
        "cache_open",
        "complete",
        "consume",
        "create_executor",
        "create_history",
        "evaluate",
        "execute",
        "load_corpus",
        "load_normalization",
        "materialize",
        "serialize",
        "snapshot",
        "verify_checkpoint",
        "verify_history",
        "write",
    }
    return [name for name in world.trace if name in names]


def test_exact_synthetic_pipeline_and_terminal_arguments(
    monkeypatch: pytest.MonkeyPatch,
    scratch_directory: Path,
) -> None:
    world, modules, _ = _synthetic_modules(scratch_directory)
    worker = _load_worker(monkeypatch, modules)

    assert (
        worker.run_registered_seed_process(world.registration, world.activation) is None
    )

    assert world.counts["execute"] == 30
    assert world.counts["evaluate"] == 30
    assert world.counts["consume"] == 30
    assert world.counts["advance"] == 29
    assert world.counts["complete"] == 1
    assert world.counts["serialize"] == 1
    assert world.counts["write"] == 1
    assert world.counts["materialize"] == 1
    assert world.counts["verify_history"] == 1
    assert world.counts["verify_checkpoint"] == 1
    assert world.counts["snapshot"] == 2
    assert world.counts["cache_exit"] == 1
    assert world.cache is not None and world.cache.closed is True

    major = _major_trace(world)
    assert major.index("materialize") < major.index("bind_validation")
    assert major.index("bind_validation") < major.index("bind_training")
    assert major.index("bind_training") < major.index("create_executor")
    assert major.index("create_executor") < major.index("create_history")
    assert major[-2:] == ["write", "cache_exit"]
    epoch_trace_start = world.trace.index("execute") - 1
    epoch_trace_end = world.trace.index("verify_history")
    expected_epoch_trace: list[str] = []
    for zero_based_epoch in range(30):
        expected_epoch_trace.extend(
            (
                "activation_verify",
                "execute",
                "activation_verify",
                "evaluate",
                "activation_verify",
                "consume",
                "activation_verify",
                "advance" if zero_based_epoch < 29 else "complete",
                "activation_verify",
            )
        )
    assert world.trace[epoch_trace_start:epoch_trace_end] == expected_epoch_trace

    normalization_identity_args = world.arguments["normalization_identity"][0][0]
    pcm_identity_args = world.arguments["pcm_identity"][0][0]
    snapshot = world.first_snapshot
    assert normalization_identity_args == (
        snapshot.normalization_byte_count,
        snapshot.normalization_sha256,
    )
    assert pcm_identity_args == (
        snapshot.pcm_cache_byte_count,
        snapshot.pcm_cache_sha256,
    )
    cache_args = world.arguments["cache_open"][0][0]
    assert cache_args[0] is world.corpus
    assert cache_args[1] == snapshot.pcm_cache_path
    cache_identity = cast(Any, cache_args[2])
    assert cache_identity.byte_count == snapshot.pcm_cache_byte_count
    assert cache_identity.sha256 == snapshot.pcm_cache_sha256

    write_args, write_kwargs = world.arguments["write"][0]
    assert write_args[0] == scratch_directory
    binding = cast(Any, write_args[1])
    assert binding.role == world.activation.role
    assert binding.ordinal == world.activation.ordinal
    assert binding.seed == world.activation.seed
    assert binding.registration_head_commit == world.registration.head_commit
    assert binding.implementation_commit == world.registration.implementation_commit
    assert binding.registration_sha256 == world.registration.registration_sha256
    assert binding.source_bundle_sha256 == world.registration.source_bundle_sha256
    assert write_kwargs == {
        "history_json_bytes": world.completed.canonical_json_bytes,
        "history_sha256": world.completed.history_sha256,
        "safetensors_bytes": world.checkpoint.safetensors_bytes,
        "safetensors_sha256": world.checkpoint.safetensors_sha256,
        "winner_epoch": world.completed.winner.zero_based_epoch,
        "model_tensor_sha256": world.completed.winner.model_tensor_sha256,
    }


@pytest.mark.parametrize(
    ("role", "seed", "ordinal"),
    (
        ("training_seed", _SEEDS[0], 0),
        ("training_seed", _SEEDS[1], 1),
        ("training_seed", _SEEDS[2], 2),
        ("selected_seed_rerun", _SEEDS[0], 3),
        ("selected_seed_rerun", _SEEDS[1], 3),
        ("selected_seed_rerun", _SEEDS[2], 3),
    ),
)
def test_exact_registered_assignment_matrix_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
    scratch_directory: Path,
    role: str,
    seed: int,
    ordinal: int,
) -> None:
    world, modules, _ = _synthetic_modules(
        scratch_directory,
        role=role,
        seed=seed,
        ordinal=ordinal,
    )
    worker = _load_worker(monkeypatch, modules)
    worker.run_registered_seed_process(world.registration, world.activation)
    assert world.counts["write"] == 1


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("role", "unregistered"),
        ("seed", True),
        ("ordinal", True),
        ("child_pid", True),
        ("child_pid", -1),
        ("child_pid", 999_999_999),
        ("ordinal", 1),
    ),
)
def test_assignment_bool_pid_and_matrix_mismatches_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    scratch_directory: Path,
    field: str,
    value: object,
) -> None:
    world, modules, _ = _synthetic_modules(scratch_directory)
    setattr(world.activation, field, value)
    worker = _load_worker(monkeypatch, modules)
    with pytest.raises(worker.Experiment002SeedWorkerError):
        worker.run_registered_seed_process(world.registration, world.activation)
    assert world.counts["snapshot"] == 0
    assert world.counts["write"] == 0


@pytest.mark.parametrize("field", ("role", "seed", "ordinal", "child_pid"))
def test_assignment_scalar_subclasses_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    scratch_directory: Path,
    field: str,
) -> None:
    class StringSubclass(str):
        pass

    class IntegerSubclass(int):
        pass

    world, modules, _ = _synthetic_modules(scratch_directory)
    original = getattr(world.activation, field)
    replacement = (
        StringSubclass(original) if field == "role" else IntegerSubclass(original)
    )
    setattr(world.activation, field, replacement)
    worker = _load_worker(monkeypatch, modules)
    with pytest.raises(worker.Experiment002SeedWorkerError):
        worker.run_registered_seed_process(world.registration, world.activation)
    assert world.counts["snapshot"] == 0


@pytest.mark.parametrize("kind", ("registration", "activation"))
def test_capability_subclasses_are_rejected_after_latch_consumption(
    monkeypatch: pytest.MonkeyPatch,
    scratch_directory: Path,
    kind: str,
) -> None:
    world, modules, types = _synthetic_modules(scratch_directory)
    base = types["Registration"] if kind == "registration" else types["Activation"]
    subclass = type("CapabilitySubclass", (base,), {})
    candidate = subclass()
    worker = _load_worker(monkeypatch, modules)
    registration = candidate if kind == "registration" else world.registration
    activation = candidate if kind == "activation" else world.activation
    with pytest.raises(TypeError):
        worker.run_registered_seed_process(registration, activation)
    with pytest.raises(worker.Experiment002SeedWorkerError):
        worker.run_registered_seed_process(world.registration, world.activation)


def test_one_shot_latch_blocks_retry_after_success(
    monkeypatch: pytest.MonkeyPatch,
    scratch_directory: Path,
) -> None:
    world, modules, _ = _synthetic_modules(scratch_directory)
    worker = _load_worker(monkeypatch, modules)
    worker.run_registered_seed_process(world.registration, world.activation)
    with pytest.raises(worker.Experiment002SeedWorkerError):
        worker.run_registered_seed_process(world.registration, world.activation)
    assert world.counts["snapshot"] == 2
    assert world.counts["write"] == 1


def test_one_shot_latch_blocks_retry_after_failure(
    monkeypatch: pytest.MonkeyPatch,
    scratch_directory: Path,
) -> None:
    world, modules, _ = _synthetic_modules(
        scratch_directory,
        fail_at="snapshot",
    )
    worker = _load_worker(monkeypatch, modules)
    with pytest.raises(_SyntheticFailure):
        worker.run_registered_seed_process(world.registration, world.activation)
    with pytest.raises(worker.Experiment002SeedWorkerError):
        worker.run_registered_seed_process(world.registration, world.activation)
    assert world.counts["snapshot"] == 1


def test_activation_assignment_mutation_between_boundaries_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    scratch_directory: Path,
) -> None:
    world, modules, _ = _synthetic_modules(scratch_directory)
    coordinator_module = modules["falsewake.experiment_002_coordinator"]
    original = cast(
        Any,
        coordinator_module.__dict__["verify_verified_child_activation"],
    )

    def mutating_verify(registration: object, activation: object) -> None:
        original(registration, activation)
        if world.counts["activation_verify"] == 3:
            world.activation.seed = _SEEDS[1]

    coordinator_module.__dict__["verify_verified_child_activation"] = mutating_verify
    worker = _load_worker(monkeypatch, modules)
    with pytest.raises(worker.Experiment002SeedWorkerError):
        worker.run_registered_seed_process(world.registration, world.activation)
    assert world.counts["snapshot"] == 1
    assert world.counts["write"] == 0


def test_one_shot_latch_rejects_concurrent_follower(
    monkeypatch: pytest.MonkeyPatch,
    scratch_directory: Path,
) -> None:
    world, modules, _ = _synthetic_modules(scratch_directory)
    entered = threading.Event()
    release = threading.Event()
    coordinator_module = modules["falsewake.experiment_002_coordinator"]
    original = cast(
        Any,
        coordinator_module.__dict__["verify_verified_child_activation"],
    )

    def blocking_verify(registration: object, activation: object) -> None:
        entered.set()
        assert release.wait(timeout=5)
        original(registration, activation)
        raise _SyntheticFailure("leader stops after admission")

    coordinator_module.__dict__["verify_verified_child_activation"] = blocking_verify
    worker = _load_worker(monkeypatch, modules)
    errors: list[BaseException] = []

    def leader() -> None:
        try:
            worker.run_registered_seed_process(world.registration, world.activation)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=leader)
    thread.start()
    assert entered.wait(timeout=5)
    with pytest.raises(worker.Experiment002SeedWorkerError):
        worker.run_registered_seed_process(world.registration, world.activation)
    release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], _SyntheticFailure)
    assert world.counts["snapshot"] == 0


def test_forked_child_rejects_before_inherited_held_latch(
    monkeypatch: pytest.MonkeyPatch,
    scratch_directory: Path,
) -> None:
    world, modules, _ = _synthetic_modules(scratch_directory)
    worker = _load_worker(monkeypatch, modules)
    entrypoint = worker.run_registered_seed_process
    closure = dict(
        zip(
            entrypoint.__code__.co_freevars,
            entrypoint.__closure__ or (),
            strict=True,
        )
    )
    inherited_lock = cast(Any, closure["attempt_lock"].cell_contents)
    read_fd, write_fd = os.pipe()
    payload = b""
    timed_out = False
    assert inherited_lock.acquire()
    process_id = os.fork()
    if process_id == 0:
        os.close(read_fd)
        try:
            entrypoint(world.registration, world.activation)
        except BaseException as error:
            child_payload = type(error).__name__.encode("ascii", errors="replace")
        else:
            child_payload = b"unexpected-success"
        try:
            os.write(write_fd, child_payload)
        finally:
            os.close(write_fd)
        os._exit(0)

    os.close(write_fd)
    try:
        readable, _, _ = select.select([read_fd], [], [], 2.0)
        if not readable:
            timed_out = True
            os.kill(process_id, signal.SIGKILL)
        else:
            payload = os.read(read_fd, 256)
    finally:
        inherited_lock.release()
        os.close(read_fd)
        os.waitpid(process_id, 0)

    assert timed_out is False
    assert payload == b"Experiment002SeedWorkerError"
    assert world.counts["activation_verify"] == 0
    assert world.counts["snapshot"] == 0


@pytest.mark.parametrize(
    "fail_at",
    (
        "snapshot",
        "load_corpus",
        "normalization_identity",
        "pcm_identity",
        "load_normalization",
        "cache_open",
        "materialize",
        "bind_validation",
        "bind_training",
        "create_executor",
        "create_history",
        "execute",
        "evaluate",
        "consume",
        "advance",
        "complete",
        "verify_history",
        "serialize",
        "verify_checkpoint",
        "snapshot:2",
    ),
)
def test_failure_matrix_stops_before_write_and_closes_open_cache(
    monkeypatch: pytest.MonkeyPatch,
    scratch_directory: Path,
    fail_at: str,
) -> None:
    world, modules, _ = _synthetic_modules(scratch_directory, fail_at=fail_at)
    worker = _load_worker(monkeypatch, modules)
    with pytest.raises(_SyntheticFailure):
        worker.run_registered_seed_process(world.registration, world.activation)
    assert world.counts["write"] == 0
    failed_name, separator, occurrence_text = fail_at.partition(":")
    failed_occurrence = int(occurrence_text) if separator else 1
    failed_indices = [
        index for index, name in enumerate(world.trace) if name == failed_name
    ]
    failure_index = failed_indices[failed_occurrence - 1]
    expected_cleanup = ["cache_exit"] if world.counts["cache_enter"] else []
    assert world.trace[failure_index + 1 :] == expected_cleanup
    if world.counts["cache_enter"]:
        assert world.cache is not None and world.cache.closed is True
        assert world.counts["cache_exit"] == 1


@pytest.mark.parametrize(
    "change",
    (
        "history_bytes",
        "history_sha256",
        "completed_seed",
        "epoch_count_bool",
        "winner_seed_bool",
        "winner_epoch_bool",
        "checkpoint_seed",
        "checkpoint_epoch",
        "checkpoint_history",
        "checkpoint_model",
        "checkpoint_bytes",
        "checkpoint_sha256",
        "checkpoint_byte_count_bool",
    ),
)
def test_manual_terminal_agreement_rejects_every_scalar_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    scratch_directory: Path,
    change: str,
) -> None:
    world, modules, _ = _synthetic_modules(
        scratch_directory,
        terminal_change=change,
    )
    worker = _load_worker(monkeypatch, modules)
    with pytest.raises((TypeError, worker.Experiment002SeedWorkerError)):
        worker.run_registered_seed_process(world.registration, world.activation)
    assert world.counts["write"] == 0
    assert world.cache is not None and world.cache.closed is True


def test_second_snapshot_change_blocks_result_write(
    monkeypatch: pytest.MonkeyPatch,
    scratch_directory: Path,
) -> None:
    world, modules, _ = _synthetic_modules(
        scratch_directory,
        snapshot_change="pcm_cache_sha256",
    )
    worker = _load_worker(monkeypatch, modules)
    with pytest.raises(worker.Experiment002SeedWorkerError):
        worker.run_registered_seed_process(world.registration, world.activation)
    assert world.counts["snapshot"] == 2
    assert world.counts["write"] == 0
    assert world.cache is not None and world.cache.closed is True


def test_aliased_second_snapshot_blocks_result_write(
    monkeypatch: pytest.MonkeyPatch,
    scratch_directory: Path,
) -> None:
    world, modules, _ = _synthetic_modules(
        scratch_directory,
        snapshot_change="alias",
    )
    worker = _load_worker(monkeypatch, modules)
    with pytest.raises(worker.Experiment002SeedWorkerError):
        worker.run_registered_seed_process(world.registration, world.activation)
    assert world.first_snapshot is world.second_snapshot
    assert world.counts["snapshot"] == 2
    assert world.counts["write"] == 0
    assert world.cache is not None and world.cache.closed is True


def test_post_write_activation_failure_propagates_after_durable_call(
    monkeypatch: pytest.MonkeyPatch,
    scratch_directory: Path,
) -> None:
    world, modules, _ = _synthetic_modules(scratch_directory)
    world.fail_after_write = True
    worker = _load_worker(monkeypatch, modules)
    with pytest.raises(_SyntheticFailure, match="post-write"):
        worker.run_registered_seed_process(world.registration, world.activation)
    assert world.counts["write"] == 1
    assert world.cache is not None and world.cache.closed is True
    assert _major_trace(world)[-2:] == ["write", "cache_exit"]


def test_writer_exception_closes_cache_and_consumes_one_shot_latch(
    monkeypatch: pytest.MonkeyPatch,
    scratch_directory: Path,
) -> None:
    world, modules, _ = _synthetic_modules(scratch_directory, fail_at="write")
    worker = _load_worker(monkeypatch, modules)
    with pytest.raises(_SyntheticFailure, match="write"):
        worker.run_registered_seed_process(world.registration, world.activation)
    assert world.counts["write"] == 1
    assert world.cache is not None and world.cache.closed is True
    with pytest.raises(worker.Experiment002SeedWorkerError):
        worker.run_registered_seed_process(world.registration, world.activation)
    assert world.counts["write"] == 1


def test_non_none_writer_still_runs_post_write_reverification_then_fails(
    monkeypatch: pytest.MonkeyPatch,
    scratch_directory: Path,
) -> None:
    world, modules, _ = _synthetic_modules(
        scratch_directory,
        writer_result=object(),
    )
    worker = _load_worker(monkeypatch, modules)
    with pytest.raises(
        worker.Experiment002SeedWorkerError,
        match="writer returned an unexpected value",
    ):
        worker.run_registered_seed_process(world.registration, world.activation)
    write_index = world.trace.index("write")
    assert world.trace[write_index + 1 :] == ["activation_verify", "cache_exit"]
    assert world.counts["write"] == 1
    assert world.cache is not None and world.cache.closed is True


@pytest.mark.parametrize(
    ("module_name", "route_name", "trace_name"),
    (
        (
            "falsewake.experiment_002_registered_executor",
            "advance_registered_training_executor",
            "advance",
        ),
        (
            "falsewake.experiment_002_registered_history",
            "verify_registered_completed_training_history",
            "verify_history",
        ),
        (
            "falsewake.experiment_002_registered_checkpoint",
            "verify_registered_serialized_checkpoint",
            "verify_checkpoint",
        ),
    ),
)
def test_non_none_void_route_returns_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    scratch_directory: Path,
    module_name: str,
    route_name: str,
    trace_name: str,
) -> None:
    world, modules, _ = _synthetic_modules(scratch_directory)
    route_module = modules[module_name]
    original = cast(Any, route_module.__dict__[route_name])

    def non_none_route(*args: object, **kwargs: object) -> object:
        original(*args, **kwargs)
        return object()

    route_module.__dict__[route_name] = non_none_route
    worker = _load_worker(monkeypatch, modules)
    with pytest.raises(worker.Experiment002SeedWorkerError):
        worker.run_registered_seed_process(world.registration, world.activation)
    assert world.counts[trace_name] == 1
    assert world.counts["write"] == 0
    assert world.cache is not None and world.cache.closed is True


def test_cache_context_identity_change_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    scratch_directory: Path,
) -> None:
    world, modules, types = _synthetic_modules(scratch_directory)
    cache_type = types["Cache"]

    def invalid_enter(cache: object) -> object:
        return world.call("cache_enter", cache, result=object())

    monkeypatch.setattr(cache_type, "__enter__", invalid_enter)
    worker = _load_worker(monkeypatch, modules)
    with pytest.raises(worker.Experiment002SeedWorkerError):
        worker.run_registered_seed_process(world.registration, world.activation)
    assert world.counts["materialize"] == 0
    assert world.cache is not None and world.cache.closed is True


def test_critical_routes_types_constants_and_helpers_are_closure_captured(
    monkeypatch: pytest.MonkeyPatch,
    scratch_directory: Path,
) -> None:
    world, modules, _ = _synthetic_modules(scratch_directory)
    worker = _load_worker(monkeypatch, modules)

    def bomb(*args: object, **kwargs: object) -> Any:
        del args, kwargs
        raise AssertionError("mutable module global reached production entrypoint")

    for name in (
        "_child_input_snapshots_match",
        "_child_result_binding",
        "_directory_stat_frame",
        "_initial_activation_binding",
        "_preflight_scratch_directory",
        "_registered_child_input_snapshot",
        "_require_child_input_snapshot",
        "_require_lower_hex",
        "_require_owned_empty_directory_stat",
        "_require_positive_int",
        "_require_sha256",
        "_terminal_payload",
        "_verify_activation_boundary",
        "advance_registered_training_executor",
        "bind_registered_training_inputs",
        "bind_registered_validation_inputs",
        "complete_registered_training_executor",
        "consume_registered_evaluated_epoch",
        "create_registered_training_executor",
        "create_registered_training_history",
        "evaluate_registered_epoch",
        "execute_registered_training_epoch",
        "load_normalization_artifact",
        "load_registered_corpus",
        "materialize_registered_validation",
        "serialize_registered_checkpoint",
        "verify_registered_completed_training_history",
        "verify_registered_serialized_checkpoint",
        "verify_verified_child_activation",
        "write_registered_child_result",
    ):
        setattr(worker, name, bomb)
    for name in (
        "ChildResultBinding",
        "Experiment002PCMCache",
        "NormalizationArtifactIdentity",
        "PCMCacheIdentity",
        "RegisteredCompletedTrainingHistory",
        "RegisteredEvaluatedEpoch",
        "RegisteredSerializedCheckpoint",
        "VerifiedChildActivation",
        "VerifiedRunRegistration",
        "_ActivationBinding",
        "_RegisteredChildInputSnapshot",
        "_TerminalPayload",
    ):
        setattr(worker, name, SimpleNamespace)
    worker.__dict__.update(
        {
            "_EPOCH_COUNT": 0,
            "REGISTERED_SEEDS": (),
            "_SCRATCH_ROOT": Path("/"),
            "os": SimpleNamespace(getpid=lambda: -1),
            "hashlib": SimpleNamespace(sha256=bomb),
            "threading": SimpleNamespace(Lock=bomb),
        }
    )
    for name in (
        "TypeError",
        "OSError",
        "any",
        "bytes",
        "cast",
        "int",
        "len",
        "list",
        "range",
        "set",
        "str",
        "type",
    ):
        setattr(worker, name, bomb)

    worker.run_registered_seed_process(world.registration, world.activation)
    assert world.counts["write"] == 1
    assert world.counts["execute"] == 30


def test_import_has_no_filesystem_io(
    monkeypatch: pytest.MonkeyPatch,
    scratch_directory: Path,
) -> None:
    _, modules, _ = _synthetic_modules(scratch_directory)
    worker = _load_worker(monkeypatch, modules, forbid_io=True)
    assert callable(worker.run_registered_seed_process)


def test_worker_source_has_no_archive_route() -> None:
    source = _SOURCE.read_text(encoding="utf-8")
    assert "archive" not in source


def test_scratch_preflight_rejects_nonempty_mode_and_symlink_paths(
    monkeypatch: pytest.MonkeyPatch,
    scratch_directory: Path,
) -> None:
    world, modules, _ = _synthetic_modules(scratch_directory)
    worker = _load_worker(monkeypatch, modules)
    worker._preflight_scratch_directory(scratch_directory)

    occupied = scratch_directory / "occupied"
    occupied.mkdir(mode=0o700)
    (occupied / "entry").write_bytes(b"x")
    with pytest.raises(worker.Experiment002SeedWorkerError):
        worker._preflight_scratch_directory(occupied)

    wrong_mode = scratch_directory / "wrong-mode"
    wrong_mode.mkdir(mode=0o755)
    # Deliberately permissive: the owner-only preflight must reject this fixture.
    wrong_mode.chmod(0o755)
    with pytest.raises(worker.Experiment002SeedWorkerError):
        worker._preflight_scratch_directory(wrong_mode)

    target = scratch_directory / "target"
    target.mkdir(mode=0o700)
    symlink = scratch_directory / "symlink"
    symlink.symlink_to(target, target_is_directory=True)
    with pytest.raises(worker.Experiment002SeedWorkerError):
        worker._preflight_scratch_directory(symlink)


def test_scratch_preflight_rejects_path_subclass_and_outside_root(
    monkeypatch: pytest.MonkeyPatch,
    scratch_directory: Path,
) -> None:
    _, modules, _ = _synthetic_modules(scratch_directory)
    worker = _load_worker(monkeypatch, modules)
    path_subclass = type("PathSubclass", (type(Path()),), {})

    with pytest.raises(TypeError):
        worker._preflight_scratch_directory(path_subclass(scratch_directory))
    with pytest.raises(worker.Experiment002SeedWorkerError):
        worker._preflight_scratch_directory(Path("/tmp"))


@pytest.mark.parametrize(("field_index", "value"), ((3, 1), (3, 0), (4, -1)))
def test_scratch_metadata_rejects_invalid_link_count_and_owner(
    monkeypatch: pytest.MonkeyPatch,
    scratch_directory: Path,
    field_index: int,
    value: int,
) -> None:
    _, modules, _ = _synthetic_modules(scratch_directory)
    worker = _load_worker(monkeypatch, modules)
    fields = list(os.lstat(scratch_directory))
    fields[field_index] = (
        os.geteuid() + 1 if field_index == 4 and value == -1 else value
    )
    metadata = os.stat_result(fields)

    with pytest.raises(worker.Experiment002SeedWorkerError):
        worker._require_owned_empty_directory_stat(metadata)
