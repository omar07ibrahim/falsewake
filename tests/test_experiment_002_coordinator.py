from __future__ import annotations

import array
import ast
import copy
import errno
import fcntl
import hashlib
import importlib
import inspect
import itertools
import json
import os
import pickle
import socket
import stat
import subprocess
import sys
import textwrap
import threading
import warnings
import weakref
from collections.abc import Callable
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from types import FunctionType, ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

import falsewake.experiment_002_coordinator as coordinator
import falsewake.experiment_002_run_authority as authority

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
SOURCE_PATH = PROJECT_ROOT / "src/falsewake/experiment_002_coordinator.py"
TEST_BINDING = coordinator._RegistrationBinding(
    head_commit="1" * 40,
    implementation_commit="2" * 40,
    registration_sha256="3" * 64,
    source_bundle_sha256="4" * 64,
)
_TICKET_IDS = itertools.count(1)


def _registered_experiment_state() -> coordinator._RegisteredExperimentState:
    closure = dict(
        zip(
            coordinator.run_registered_experiment.__code__.co_freevars,
            coordinator.run_registered_experiment.__closure__ or (),
            strict=True,
        )
    )
    state = closure["state"].cell_contents
    assert type(state) is coordinator._RegisteredExperimentState
    return state


@pytest.fixture(autouse=True)
def isolated_coordinator_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(coordinator, "_ACTIVATIONS", weakref.WeakKeyDictionary())
    monkeypatch.setattr(coordinator, "_ACTIVATION_GUARDS", weakref.WeakKeyDictionary())
    monkeypatch.setattr(coordinator, "_ACTIVATION_ANCHORS", weakref.WeakKeyDictionary())
    monkeypatch.setattr(coordinator, "_ACTIVATION_PHASES", weakref.WeakKeyDictionary())
    monkeypatch.setattr(coordinator, "_DISPATCHED_ACTIVATIONS", weakref.WeakSet())
    monkeypatch.setattr(coordinator, "_COMPLETED_ACTIVATIONS", weakref.WeakSet())
    monkeypatch.setattr(coordinator, "_FAILED_ACTIVATIONS", weakref.WeakSet())
    monkeypatch.setattr(coordinator, "_ACCEPTED_TICKET_DIGESTS", set())
    monkeypatch.setattr(coordinator, "_ACCEPTED_TICKET_NONCES", set())
    monkeypatch.setattr(coordinator, "_ISSUED_TICKET_DIGESTS", set())
    monkeypatch.setattr(coordinator, "_ISSUED_TICKET_NONCES", set())


@pytest.fixture
def synthetic_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> coordinator.VerifiedRunRegistration:
    monkeypatch.setattr(authority, "_ISSUED", weakref.WeakKeyDictionary())
    monkeypatch.setattr(authority, "_ISSUED_GUARDS", weakref.WeakKeyDictionary())
    monkeypatch.setattr(authority, "_FAILED", weakref.WeakSet())
    monkeypatch.setattr(authority, "_ISSUANCE_COMPLETE", False)
    monkeypatch.setattr(
        authority,
        "reverify_verified_run_registration",
        lambda _registration: None,
    )
    snapshot = authority._RepositorySnapshot(
        repository_root=PROJECT_ROOT,
        head_commit=TEST_BINDING.head_commit,
        implementation_commit=TEST_BINDING.implementation_commit,
        registration_sha256=TEST_BINDING.registration_sha256,
        source_bundle_sha256=TEST_BINDING.source_bundle_sha256,
        source_paths=("src/falsewake/experiment_002_run_authority.py",),
    )
    return authority._issue_controlled_snapshot_for_tests(snapshot)


type _SyntheticActivation = Callable[[socket.socket, int], None]
type _SyntheticSupervisorBehavior = Callable[
    [tuple[int, int], _SyntheticActivation, int, object],
    object,
]
type _SyntheticSnapshotBehavior = Callable[[object], object]


def _complete_synthetic_parent_activation(
    activation: _SyntheticActivation,
    registration: coordinator.VerifiedRunRegistration,
    role: str,
    seed: int,
    activations: list[
        tuple[
            socket.socket,
            coordinator.VerifiedRunRegistration,
            str,
            int,
            int,
        ]
    ],
    /,
) -> int:
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    child_pid = _fork()
    if child_pid == 0:
        try:
            parent.close()
            packet, _credentials = coordinator._receive_packet(child)
            ticket = coordinator._decode_packet(packet)
            acknowledgement = {
                "child_pid": ticket["child_pid"],
                "kind": "activation_ack",
                "nonce": ticket["nonce"],
                "ordinal": ticket["ordinal"],
                "parent_pid": ticket["parent_pid"],
                "role": ticket["role"],
                "schema_version": ticket["schema_version"],
                "seed": ticket["seed"],
                "status": "accepted",
                "ticket_sha256": coordinator._ticket_sha256(packet),
            }
            coordinator._send_packet(
                child,
                coordinator._encode_packet(acknowledgement),
            )
            child.close()
            os._exit(0)
        except BaseException:
            os._exit(91)
    child.close()
    try:
        try:
            activation(parent, child_pid)
        except BaseException:
            waited_pid, status = os.waitpid(child_pid, 0)
            if (
                waited_pid == child_pid
                and os.WIFEXITED(status)
                and os.WEXITSTATUS(status) == 0
            ):
                activations.append((parent, registration, role, seed, child_pid))
            raise
        activations.append((parent, registration, role, seed, child_pid))
        _wait_success(child_pid)
    finally:
        parent.close()
    return child_pid


def _fail_synthetic_parent_activation(
    activation: _SyntheticActivation,
    /,
) -> int:
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    child_pid = _fork()
    if child_pid == 0:
        parent.close()
        child.close()
        os._exit(0)
    child.close()
    try:
        activation(parent, child_pid)
    finally:
        parent.close()
        _wait_success(child_pid)
    return child_pid


@dataclass(slots=True)
class _SyntheticParentHarness:
    supervisor_module: ModuleType
    child_result_module: ModuleType
    binding_type: type[Any]
    verified_result_type: type[Any]
    supervised_result_type: type[Any]
    behavior: list[_SyntheticSupervisorBehavior]
    snapshot_behavior: list[_SyntheticSnapshotBehavior]
    imports: list[str]
    calls: list[tuple[tuple[int, int], _SyntheticActivation, int, object]]
    activations: list[
        tuple[
            socket.socket,
            coordinator.VerifiedRunRegistration,
            str,
            int,
            int,
        ]
    ]
    binding_frames: list[tuple[str, int, int, str, str, str, str]]
    verification_calls: list[object]
    snapshot_calls: list[object]


def _set_route_identity(value: Any, name: str, module_name: str) -> None:
    value.__name__ = name
    value.__qualname__ = name
    value.__module__ = module_name


def _install_synthetic_parent_execution(
    monkeypatch: pytest.MonkeyPatch,
    registration: coordinator.VerifiedRunRegistration,
) -> _SyntheticParentHarness:
    supervisor_module = ModuleType(coordinator._SUPERVISOR_MODULE)
    child_result_module = ModuleType(coordinator._CHILD_RESULT_MODULE)
    calls: list[tuple[tuple[int, int], _SyntheticActivation, int, object]] = []
    imports: list[str] = []
    activations: list[
        tuple[
            socket.socket,
            coordinator.VerifiedRunRegistration,
            str,
            int,
            int,
        ]
    ] = []
    binding_frames: list[tuple[str, int, int, str, str, str, str]] = []
    verification_calls: list[object] = []
    snapshot_calls: list[object] = []

    @dataclass(frozen=True, slots=True)
    class SyntheticChildResultBinding:
        role: str
        ordinal: int
        seed: int
        registration_head_commit: str
        implementation_commit: str
        registration_sha256: str
        source_bundle_sha256: str

    _set_route_identity(
        SyntheticChildResultBinding,
        coordinator._CHILD_RESULT_BINDING_TYPE,
        coordinator._CHILD_RESULT_MODULE,
    )

    class SyntheticVerifiedChildResult:
        def __init__(self, binding: SyntheticChildResultBinding) -> None:
            self._binding = binding

        @property
        def binding(self) -> SyntheticChildResultBinding:
            return self._binding

        @property
        def role(self) -> str:
            return self._binding.role

        @property
        def ordinal(self) -> int:
            return self._binding.ordinal

        @property
        def seed(self) -> int:
            return self._binding.seed

        @property
        def registration_head_commit(self) -> str:
            return self._binding.registration_head_commit

        @property
        def implementation_commit(self) -> str:
            return self._binding.implementation_commit

        @property
        def registration_sha256(self) -> str:
            return self._binding.registration_sha256

        @property
        def source_bundle_sha256(self) -> str:
            return self._binding.source_bundle_sha256

        @property
        def winner_epoch(self) -> int:
            return 7

        @property
        def winner_macro_f1(self) -> Fraction:
            return Fraction(3, 4)

        @property
        def winner_validation_cross_entropy(self) -> float:
            return 0.5

        @property
        def model_tensor_sha256(self) -> str:
            return "5" * 64

    _set_route_identity(
        SyntheticVerifiedChildResult,
        coordinator._VERIFIED_CHILD_RESULT_TYPE,
        coordinator._CHILD_RESULT_MODULE,
    )

    @dataclass(frozen=True, slots=True)
    class SyntheticSupervisedChildResult:
        pid: int
        cpu_ids: tuple[int, int]
        elapsed_nanoseconds: int
        maximum_rss_bytes: int
        output_and_scratch_bytes: int
        verified_child_result: object

    _set_route_identity(
        SyntheticSupervisedChildResult,
        coordinator._SUPERVISED_CHILD_RESULT_TYPE,
        coordinator._SUPERVISOR_MODULE,
    )

    def load_registered_child_result(_path: object, _binding: object, /) -> object:
        raise AssertionError("synthetic coordinator adapter must not call the loader")

    _set_route_identity(
        load_registered_child_result,
        coordinator._CHILD_RESULT_LOADER_FUNCTION,
        coordinator._CHILD_RESULT_MODULE,
    )

    def verify_verified_child_result(result: object, /) -> None:
        if type(result) is not SyntheticVerifiedChildResult:
            raise TypeError("not an exact synthetic verified child result")
        verification_calls.append(result)

    _set_route_identity(
        verify_verified_child_result,
        coordinator._CHILD_RESULT_VERIFIER_FUNCTION,
        coordinator._CHILD_RESULT_MODULE,
    )

    history_bytes = b'{"synthetic":true}'
    safetensors_bytes = b"synthetic-safetensors"
    envelope_bytes = b'{"synthetic-envelope":true}'

    def default_snapshot(result: object, /) -> object:
        if type(result) is not SyntheticVerifiedChildResult:
            raise TypeError("not an exact synthetic verified child result")
        binding = cast(Any, result)._binding
        return (
            binding.role,
            binding.ordinal,
            binding.seed,
            binding.registration_head_commit,
            binding.implementation_commit,
            binding.registration_sha256,
            binding.source_bundle_sha256,
            history_bytes,
            len(history_bytes),
            hashlib.sha256(
                b"falsewake-exp002-history-v1\0" + history_bytes
            ).hexdigest(),
            safetensors_bytes,
            len(safetensors_bytes),
            hashlib.sha256(safetensors_bytes).hexdigest(),
            envelope_bytes,
            len(envelope_bytes),
            hashlib.sha256(
                b"falsewake-exp002-child-result-envelope-v1\0" + envelope_bytes
            ).hexdigest(),
            7,
            3,
            4,
            (0.5).hex(),
            "5" * 64,
        )

    snapshot_behavior: list[_SyntheticSnapshotBehavior] = [default_snapshot]

    def snapshot_verified_child_result(result: object, /) -> object:
        snapshot_calls.append(result)
        return snapshot_behavior[0](result)

    _set_route_identity(
        snapshot_verified_child_result,
        coordinator._CHILD_RESULT_SNAPSHOTTER_FUNCTION,
        coordinator._CHILD_RESULT_MODULE,
    )

    def default_behavior(
        cpu_ids: tuple[int, int],
        activation: _SyntheticActivation,
        _source_bundle_fd: int,
        binding: object,
    ) -> object:
        if type(binding) is not SyntheticChildResultBinding:
            raise TypeError("not an exact synthetic child-result binding")
        child_pid = _complete_synthetic_parent_activation(
            activation,
            registration,
            binding.role,
            binding.seed,
            activations,
        )
        verified = SyntheticVerifiedChildResult(binding)
        return SyntheticSupervisedChildResult(
            pid=child_pid,
            cpu_ids=cpu_ids,
            elapsed_nanoseconds=11,
            maximum_rss_bytes=12,
            output_and_scratch_bytes=13,
            verified_child_result=verified,
        )

    behavior: list[_SyntheticSupervisorBehavior] = [default_behavior]

    registered_plan = object()
    registered_limits = object()
    real_kernel = object()

    def supervise_child(
        cpu_ids: tuple[int, int],
        activation: _SyntheticActivation,
        source_bundle_fd: int,
        binding: object,
        /,
        *,
        plan: object = registered_plan,
        limits: object = registered_limits,
        kernel: object = real_kernel,
    ) -> object:
        if (
            plan is not registered_plan
            or limits is not registered_limits
            or kernel is not real_kernel
        ):
            raise AssertionError("synthetic supervisor authority changed")
        if type(binding) is SyntheticChildResultBinding:
            binding_frames.append(
                (
                    binding.role,
                    binding.ordinal,
                    binding.seed,
                    binding.registration_head_commit,
                    binding.implementation_commit,
                    binding.registration_sha256,
                    binding.source_bundle_sha256,
                )
            )
        calls.append((cpu_ids, activation, source_bundle_fd, binding))
        return behavior[0](cpu_ids, activation, source_bundle_fd, binding)

    _set_route_identity(
        supervise_child,
        coordinator._INJECTABLE_SUPERVISE_CHILD_FUNCTION,
        coordinator._SUPERVISOR_MODULE,
    )

    def make_registered_supervisor() -> FunctionType:
        supervise = supervise_child
        plan = registered_plan
        limits = registered_limits
        kernel = real_kernel

        def supervise_registered_child(
            cpu_ids: tuple[int, int],
            activation: _SyntheticActivation,
            source_bundle_fd: int,
            binding: object,
            /,
        ) -> object:
            return supervise(
                cpu_ids,
                activation,
                source_bundle_fd,
                binding,
                plan=plan,
                limits=limits,
                kernel=kernel,
            )

        return cast(FunctionType, supervise_registered_child)

    supervise_registered_child = make_registered_supervisor()
    _set_route_identity(
        supervise_registered_child,
        coordinator._SUPERVISE_CHILD_FUNCTION,
        coordinator._SUPERVISOR_MODULE,
    )

    setattr(
        child_result_module,
        coordinator._CHILD_RESULT_BINDING_TYPE,
        SyntheticChildResultBinding,
    )
    setattr(
        child_result_module,
        coordinator._VERIFIED_CHILD_RESULT_TYPE,
        SyntheticVerifiedChildResult,
    )
    setattr(
        child_result_module,
        coordinator._CHILD_RESULT_LOADER,
        load_registered_child_result,
    )
    setattr(
        child_result_module,
        coordinator._CHILD_RESULT_VERIFIER,
        verify_verified_child_result,
    )
    setattr(
        child_result_module,
        coordinator._CHILD_RESULT_SNAPSHOTTER,
        snapshot_verified_child_result,
    )
    setattr(
        supervisor_module,
        coordinator._SUPERVISE_CHILD_FUNCTION,
        supervise_registered_child,
    )
    setattr(
        supervisor_module,
        coordinator._INJECTABLE_SUPERVISE_CHILD_FUNCTION,
        supervise_child,
    )
    setattr(supervisor_module, coordinator._REGISTERED_SUPERVISOR_PLAN, registered_plan)
    setattr(
        supervisor_module,
        coordinator._REGISTERED_SUPERVISOR_LIMITS,
        registered_limits,
    )
    setattr(supervisor_module, coordinator._REGISTERED_SUPERVISOR_KERNEL, real_kernel)
    setattr(
        supervisor_module,
        coordinator._SUPERVISED_CHILD_RESULT_TYPE,
        SyntheticSupervisedChildResult,
    )
    for name in (
        coordinator._CHILD_RESULT_BINDING_TYPE,
        coordinator._VERIFIED_CHILD_RESULT_TYPE,
        coordinator._CHILD_RESULT_LOADER,
        coordinator._CHILD_RESULT_VERIFIER,
    ):
        setattr(supervisor_module, name, getattr(child_result_module, name))

    def dynamic_import(name: str) -> ModuleType:
        imports.append(name)
        if name == coordinator._SUPERVISOR_MODULE:
            return supervisor_module
        if name == coordinator._CHILD_RESULT_MODULE:
            return child_result_module
        raise AssertionError(f"unexpected parent execution import: {name}")

    monkeypatch.setattr(importlib, "import_module", dynamic_import)

    return _SyntheticParentHarness(
        supervisor_module=supervisor_module,
        child_result_module=child_result_module,
        binding_type=SyntheticChildResultBinding,
        verified_result_type=SyntheticVerifiedChildResult,
        supervised_result_type=SyntheticSupervisedChildResult,
        behavior=behavior,
        snapshot_behavior=snapshot_behavior,
        imports=imports,
        calls=calls,
        activations=activations,
        binding_frames=binding_frames,
        verification_calls=verification_calls,
        snapshot_calls=snapshot_calls,
    )


def _synthetic_supervised_result(
    harness: _SyntheticParentHarness,
    binding: object,
    *,
    pid: int,
    cpu_ids: tuple[int, int],
    elapsed_nanoseconds: object = 11,
    maximum_rss_bytes: object = 12,
    output_and_scratch_bytes: object = 13,
    verified_child_result: object | None = None,
) -> object:
    child_result = (
        harness.verified_result_type(binding)
        if verified_child_result is None
        else verified_child_result
    )
    return harness.supervised_result_type(
        pid=pid,
        cpu_ids=cpu_ids,
        elapsed_nanoseconds=elapsed_nanoseconds,
        maximum_rss_bytes=maximum_rss_bytes,
        output_and_scratch_bytes=output_and_scratch_bytes,
        verified_child_result=child_result,
    )


type _SelectionMetric = tuple[int, int, int, float]
type _SnapshotMutation = Callable[[list[object]], None]


def _selection_metrics(
    *values: _SelectionMetric,
) -> dict[int, _SelectionMetric]:
    metrics = values or ((7, 3, 4, 0.5),) * 3
    assert len(metrics) == 3
    return dict(zip(coordinator.REGISTERED_SEEDS, metrics, strict=True))


def _install_selection_snapshot_behavior(
    harness: _SyntheticParentHarness,
    metrics: dict[int, _SelectionMetric],
    *,
    rerun_mutation: _SnapshotMutation | None = None,
) -> None:
    base_snapshot = harness.snapshot_behavior[0]

    def selection_snapshot(result: object, /) -> object:
        fields = list(cast(tuple[object, ...], base_snapshot(result)))
        binding = cast(Any, result)._binding
        epoch, numerator, denominator, cross_entropy = metrics[binding.seed]
        fields[16:20] = (
            epoch,
            numerator,
            denominator,
            cross_entropy.hex(),
        )
        if binding.role == coordinator._RERUN_ROLE and rerun_mutation is not None:
            rerun_mutation(fields)
        return tuple(fields)

    harness.snapshot_behavior[0] = selection_snapshot


def _issue_additional_synthetic_registration() -> coordinator.VerifiedRunRegistration:
    authority._ISSUANCE_COMPLETE = False
    snapshot = authority._RepositorySnapshot(
        repository_root=PROJECT_ROOT,
        head_commit=TEST_BINDING.head_commit,
        implementation_commit=TEST_BINDING.implementation_commit,
        registration_sha256=TEST_BINDING.registration_sha256,
        source_bundle_sha256=TEST_BINDING.source_bundle_sha256,
        source_paths=("src/falsewake/experiment_002_run_authority.py",),
    )
    return authority._issue_controlled_snapshot_for_tests(snapshot)


def _mutate_required_rerun_snapshot_field(
    fields: list[object],
    index: int,
) -> None:
    if index in {7, 8, 9}:
        history_bytes = b'{"rerun-history":true}'
        if index == 8:
            history_bytes += b"x"
        fields[7:10] = (
            history_bytes,
            len(history_bytes),
            hashlib.sha256(
                b"falsewake-exp002-history-v1\0" + history_bytes
            ).hexdigest(),
        )
    elif index in {10, 11, 12}:
        safetensors_bytes = b"rerun-safetensors"
        if index == 11:
            safetensors_bytes += b"x"
        fields[10:13] = (
            safetensors_bytes,
            len(safetensors_bytes),
            hashlib.sha256(safetensors_bytes).hexdigest(),
        )
    elif index == 16:
        fields[16] = 6
    elif index == 17:
        fields[17] = 1
    elif index == 18:
        fields[18] = 5
    elif index == 19:
        fields[19] = (0.25).hex()
    elif index == 20:
        fields[20] = "6" * 64
    else:
        raise AssertionError(f"unsupported rerun field: {index}")


def _registered_seed_selection_implementation() -> FunctionType:
    closure = dict(
        zip(
            coordinator._run_registered_seed_selection.__code__.co_freevars,
            coordinator._run_registered_seed_selection.__closure__ or (),
            strict=True,
        )
    )
    return cast(FunctionType, closure["implementation"].cell_contents)


def _ticket(
    *,
    role: str = coordinator._SEED_ROLE,
    seed: int = coordinator.REGISTERED_SEEDS[0],
    nonce: str | None = None,
    digest: str | None = None,
) -> coordinator._Ticket:
    ticket_id = next(_TICKET_IDS)
    return coordinator._Ticket(
        assignment=coordinator._assignment(role, seed),
        parent_pid=os.getppid(),
        child_pid=os.getpid(),
        nonce=f"{ticket_id:064x}" if nonce is None else nonce,
        packet_sha256=f"{ticket_id + 1_000_000:064x}" if digest is None else digest,
    )


def _activation(
    registration: coordinator.VerifiedRunRegistration,
    *,
    role: str = coordinator._SEED_ROLE,
    seed: int = coordinator.REGISTERED_SEEDS[0],
) -> coordinator.VerifiedChildActivation:
    ticket = _ticket(role=role, seed=seed)
    receipt = coordinator._reserve_accepted_ticket(ticket)
    return coordinator._issue_activation(
        registration,
        ticket,
        receipt,
        _process_guard_binding(),
    )


def _process_guard_binding() -> coordinator._ProcessGuardBinding:
    guard = object()

    def verify(value: object, /) -> None:
        if value is not guard:
            raise coordinator.Experiment002CoordinatorError(
                "synthetic process guard identity changed"
            )

    return coordinator._ProcessGuardBinding(guard=guard, verifier=verify)


def _raw_packet(payload: bytes, *, magic: bytes | None = None) -> bytes:
    return (
        coordinator._FRAME_HEADER.pack(
            coordinator._FRAME_MAGIC if magic is None else magic,
            len(payload),
        )
        + payload
    )


def _wait_success(pid: int) -> None:
    waited_pid, status = os.waitpid(pid, 0)
    assert waited_pid == pid
    assert os.WIFEXITED(status)
    assert os.WEXITSTATUS(status) == 0


def _fork() -> int:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"This process .* is multi-threaded, use of fork\(\)",
            category=DeprecationWarning,
        )
        return os.fork()


def _install_forked_synthetic_registration_authority() -> None:
    synthetic_state = SimpleNamespace(
        head_commit=TEST_BINDING.head_commit,
        implementation_commit=TEST_BINDING.implementation_commit,
        registration_sha256=TEST_BINDING.registration_sha256,
        source_bundle_sha256=TEST_BINDING.source_bundle_sha256,
        profile=authority._EXPERIMENT_002_PROFILE,
    )
    mutable_authority = cast(Any, authority)
    mutable_authority._AUTHORITY_PROCESS_ID = os.getpid()
    mutable_authority.reverify_verified_run_registration = lambda _registration: None
    mutable_authority._verified_state = lambda _registration: synthetic_state


def _fork_fixed_fd_round_trip(
    registration: coordinator.VerifiedRunRegistration,
    *,
    role: str,
    seed: int,
) -> None:
    expected_ordinal = coordinator._assignment(role, seed).ordinal
    parent_channel, child_channel = socket.socketpair(
        socket.AF_UNIX,
        socket.SOCK_SEQPACKET,
    )
    pid = _fork()
    if pid == 0:
        try:
            _install_forked_synthetic_registration_authority()
            parent_channel.close()
            child_fd = child_channel.fileno()
            if child_fd == coordinator._CHILD_CONTROL_FD:
                child_channel.detach()
            else:
                os.dup2(child_fd, coordinator._CHILD_CONTROL_FD, inheritable=False)
                child_channel.close()

            process_guard_binding = _process_guard_binding()
            coordinator._claim_and_verify_child_process_guard = lambda: (
                process_guard_binding
            )

            def worker(
                supplied_registration: coordinator.VerifiedRunRegistration,
                activation: coordinator.VerifiedChildActivation,
                /,
            ) -> None:
                if supplied_registration is not registration:
                    raise AssertionError("worker registration identity changed")
                if (
                    activation.role != role
                    or activation.seed != seed
                    or activation.ordinal != expected_ordinal
                    or activation.child_pid != os.getpid()
                ):
                    raise AssertionError("worker activation fields changed")
                try:
                    os.fstat(coordinator._CHILD_CONTROL_FD)
                except OSError as error:
                    if error.errno != errno.EBADF:
                        raise
                else:
                    raise AssertionError("worker inherited open control FD3")
                coordinator.verify_verified_child_activation(
                    registration,
                    activation,
                )

            def dynamic_import(name: str) -> object:
                if name != coordinator._WORKER_MODULE:
                    raise AssertionError(f"unexpected child import: {name}")
                return SimpleNamespace(run_registered_seed_process=worker)

            cast(Any, importlib).import_module = dynamic_import
            coordinator._run_guarded_registered_seed_child(registration)
            try:
                coordinator._run_registered_seed_child(registration)
            except coordinator.Experiment002CoordinatorError as error:
                if "already attempted" not in str(error):
                    raise
            else:
                raise AssertionError("successful child route was retried")
            os._exit(0)
        except BaseException:
            os._exit(92)
    child_channel.close()
    try:
        coordinator._parent_activate_child(
            parent_channel,
            registration,
            role=role,
            seed=seed,
            child_pid=pid,
        )
        _wait_success(pid)
        parent_channel.settimeout(3.0)
        assert parent_channel.recv(1) == b""
    finally:
        parent_channel.close()


def _send_valid_ticket(channel: socket.socket, *, child_pid: int) -> None:
    assignment = coordinator._assignment(
        coordinator._SEED_ROLE,
        coordinator.REGISTERED_SEEDS[0],
    )
    nonce = f"{next(_TICKET_IDS):064x}"
    packet = coordinator._encode_packet(
        coordinator._ticket_document(
            TEST_BINDING,
            assignment,
            parent_pid=os.getpid(),
            child_pid=child_pid,
            nonce=nonce,
        )
    )
    coordinator._send_packet(channel, packet)


def _assert_no_ack(channel: socket.socket) -> None:
    channel.settimeout(3.0)
    try:
        observed = channel.recv(1)
    except ConnectionResetError:
        return
    assert observed == b""


def _fork_guarded_route_failure(
    registration: coordinator.VerifiedRunRegistration,
    *,
    stage: str,
) -> None:
    parent_channel, child_channel = socket.socketpair(
        socket.AF_UNIX,
        socket.SOCK_SEQPACKET,
    )
    pid = _fork()
    if pid == 0:
        try:
            _install_forked_synthetic_registration_authority()
            parent_channel.close()
            child_fd = child_channel.fileno()
            inheritable = stage == "inheritable"
            if child_fd == coordinator._CHILD_CONTROL_FD:
                child_channel.detach()
                os.set_inheritable(coordinator._CHILD_CONTROL_FD, inheritable)
            else:
                os.dup2(
                    child_fd,
                    coordinator._CHILD_CONTROL_FD,
                    inheritable=inheritable,
                )
                child_channel.close()

            if stage == "status_flags":
                status_flags = fcntl.fcntl(
                    coordinator._CHILD_CONTROL_FD,
                    fcntl.F_GETFL,
                )
                fcntl.fcntl(
                    coordinator._CHILD_CONTROL_FD,
                    fcntl.F_SETFL,
                    status_flags | os.O_ASYNC,
                )

            triggered = stage in {"inheritable", "status_flags"}
            guard_claims = 0
            worker_called = False
            issued: list[coordinator.VerifiedChildActivation] = []
            process_guard_binding = _process_guard_binding()
            mutable_coordinator = cast(Any, coordinator)

            def fail(message: str) -> None:
                nonlocal triggered
                triggered = True
                raise coordinator.Experiment002CoordinatorError(message)

            def claim_guard() -> coordinator._ProcessGuardBinding:
                nonlocal guard_claims, triggered
                guard_claims += 1
                if stage == "guard":
                    fail("injected guard failure")
                if stage == "frame_tamper":
                    triggered = True
                    os.set_inheritable(coordinator._CHILD_CONTROL_FD, True)
                return process_guard_binding

            mutable_coordinator._claim_and_verify_child_process_guard = claim_guard

            if stage == "peek":
                mutable_coordinator._peek_packet = lambda _channel: fail(
                    "injected peek"
                )
            elif stage == "registration":
                mutable_coordinator._registration_binding = lambda _registration: fail(
                    "injected registration"
                )
            elif stage == "decode":
                mutable_coordinator._decode_packet = lambda _packet: fail(
                    "injected decode"
                )
            elif stage in {"consume_mismatch", "credential_mismatch"}:
                original_consume = coordinator._consume_packet

                def mismatched_consume(
                    channel: socket.socket,
                ) -> coordinator._ReceivedPacket:
                    nonlocal triggered
                    triggered = True
                    received = original_consume(channel)
                    if stage == "consume_mismatch":
                        return replace(received, packet=received.packet + b"x")
                    return replace(
                        received,
                        credentials=replace(
                            received.credentials,
                            pid=received.credentials.pid + 1,
                        ),
                    )

                mutable_coordinator._consume_packet = mismatched_consume
            elif stage == "guard_reverify":
                mutable_coordinator._require_process_guard_binding = lambda _binding: (
                    fail("injected guard reverify")
                )
            elif stage == "reserve":
                mutable_coordinator._reserve_accepted_ticket = lambda _ticket: fail(
                    "injected reserve"
                )
            elif stage == "issue":
                mutable_coordinator._issue_activation = lambda *_args: fail(
                    "injected issue"
                )
            elif stage == "activation_verify":
                mutable_coordinator.verify_verified_child_activation = lambda *_args: (
                    fail("injected activation verification")
                )
            elif stage == "ack_encode":
                mutable_coordinator._ack_document = lambda _ticket: fail(
                    "injected acknowledgement encoding"
                )
            elif stage == "ack_send":
                mutable_coordinator._send_packet_prepared = lambda *_args: fail(
                    "injected acknowledgement send"
                )

            if stage != "issue":
                original_issue = coordinator._issue_activation

                def capture_issue(
                    supplied_registration: coordinator.VerifiedRunRegistration,
                    ticket: coordinator._Ticket,
                    receipt: coordinator._AcceptedTicketReceipt,
                    binding: coordinator._ProcessGuardBinding,
                ) -> coordinator.VerifiedChildActivation:
                    activation = original_issue(
                        supplied_registration,
                        ticket,
                        receipt,
                        binding,
                    )
                    issued.append(activation)
                    return activation

                mutable_coordinator._issue_activation = capture_issue

            def dispatch(
                _registration: coordinator.VerifiedRunRegistration,
                _activation: coordinator.VerifiedChildActivation,
            ) -> None:
                nonlocal worker_called
                worker_called = True

            mutable_coordinator._dispatch_registered_seed_process = dispatch

            try:
                coordinator._run_guarded_registered_seed_child(registration)
            except BaseException:
                pass
            else:
                os._exit(71)

            if not triggered or worker_called:
                os._exit(72)
            if stage in {"inheritable", "status_flags"} and guard_claims != 0:
                os._exit(73)
            if stage in {
                "peek",
                "guard",
                "frame_tamper",
                "registration",
                "decode",
                "consume_mismatch",
                "credential_mismatch",
                "guard_reverify",
                "reserve",
                "inheritable",
                "status_flags",
            } and (
                coordinator._ACCEPTED_TICKET_DIGESTS
                or coordinator._ACCEPTED_TICKET_NONCES
            ):
                os._exit(74)
            if stage in {"activation_verify", "ack_encode", "ack_send"} and (
                len(issued) != 1 or issued[0] not in coordinator._FAILED_ACTIVATIONS
            ):
                os._exit(75)

            coordinator._ACCEPTED_TICKET_DIGESTS.clear()
            coordinator._ACCEPTED_TICKET_NONCES.clear()
            coordinator._FAILED_ACTIVATIONS.clear()
            try:
                coordinator._run_registered_seed_child(registration)
            except coordinator.Experiment002CoordinatorError as error:
                if "already attempted" not in str(error):
                    os._exit(76)
            else:
                os._exit(77)
            os._exit(0)
        except BaseException:
            os._exit(78)

    child_channel.close()
    try:
        try:
            _send_valid_ticket(parent_channel, child_pid=pid)
        except coordinator.Experiment002CoordinatorError:
            if stage not in {"inheritable", "status_flags", "peek"}:
                raise
        else:
            _assert_no_ack(parent_channel)
    finally:
        parent_channel.close()
    _wait_success(pid)


def test_module_imports_only_stdlib_and_run_authority() -> None:
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
        "dataclasses",
        "errno",
        "falsewake.experiment_002_run_authority",
        "fcntl",
        "hashlib",
        "importlib",
        "json",
        "math",
        "os",
        "secrets",
        "socket",
        "stat",
        "struct",
        "threading",
        "types",
        "typing",
        "weakref",
    }
    source = SOURCE_PATH.read_text(encoding="utf-8")
    assert "falsewake.experiment_002_seed_worker" in source
    assert "falsewake.experiment_002_supervisor" in source
    assert "falsewake.experiment_002_child_result" in source
    assert "numpy" not in source
    assert "torch" not in source
    assert "experiment_002_training" not in source


def test_coordinator_import_does_not_load_parent_execution_modules() -> None:
    script = textwrap.dedent(
        f"""
        import json
        import sys

        sys.path.insert(0, {os.fspath(SOURCE_ROOT)!r})
        import falsewake.experiment_002_coordinator

        names = (
            'falsewake.experiment_002_supervisor',
            'falsewake.experiment_002_child_result',
            'falsewake.experiment_002_final_evidence',
            'falsewake.experiment_002_final_publication',
        )
        print(json.dumps({{name: name in sys.modules for name in names}}))
        """
    )
    completed = subprocess.run(
        (sys.executable, "-I", "-S", "-B", "-c", script),
        capture_output=True,
        check=False,
        text=True,
        timeout=20.0,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "falsewake.experiment_002_supervisor": False,
        "falsewake.experiment_002_child_result": False,
        "falsewake.experiment_002_final_evidence": False,
        "falsewake.experiment_002_final_publication": False,
    }


def test_real_parent_execution_routes_claim_in_an_isolated_process() -> None:
    script = textwrap.dedent(
        f"""
        import json
        import sys

        sys.path.insert(0, {os.fspath(SOURCE_ROOT)!r})
        import falsewake.experiment_002_coordinator as coordinator

        routes = coordinator._claim_and_verify_parent_execution_routes()
        coordinator._require_parent_execution_routes_unchanged(routes)
        print(json.dumps({{
            'supervisor': routes[0].__name__,
            'child_result': routes[1].__name__,
            'supervise_child': routes[2].__name__,
            'binding_type': routes[4].__name__,
            'result_type': routes[5].__name__,
            'snapshot_result': routes[8].__name__,
        }}, sort_keys=True))
        """
    )
    completed = subprocess.run(
        (sys.executable, "-I", "-S", "-B", "-c", script),
        capture_output=True,
        check=False,
        text=True,
        timeout=20.0,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "binding_type": coordinator._CHILD_RESULT_BINDING_TYPE,
        "child_result": coordinator._CHILD_RESULT_MODULE,
        "result_type": coordinator._VERIFIED_CHILD_RESULT_TYPE,
        "snapshot_result": coordinator._CHILD_RESULT_SNAPSHOTTER_FUNCTION,
        "supervise_child": coordinator._SUPERVISE_CHILD_FUNCTION,
        "supervisor": coordinator._SUPERVISOR_MODULE,
    }


def test_real_process_guard_claim_runs_only_in_a_fresh_subprocess() -> None:
    module_name = "falsewake.experiment_002_process_guard"
    assert module_name not in sys.modules
    script = textwrap.dedent(
        f"""
        import json
        import os
        import pathlib
        import sys

        sys.path.insert(0, {os.fspath(SOURCE_ROOT)!r})
        import falsewake.experiment_002_coordinator as coordinator

        def status():
            observed = {{}}
            for line in pathlib.Path('/proc/self/status').read_text(
                encoding='ascii'
            ).splitlines():
                key, separator, value = line.partition(':')
                if separator and key in {{'NoNewPrivs', 'Seccomp', 'Seccomp_filters'}}:
                    observed[key] = int(value.strip())
            return observed

        assert {module_name!r} not in sys.modules
        before = status()
        binding = coordinator._claim_and_verify_child_process_guard()
        coordinator._require_process_guard_binding(binding)
        after = status()
        print(json.dumps({{
            'before': before,
            'after': after,
            'module_loaded': {module_name!r} in sys.modules,
            'os_pid': os.getpid(),
            'process_id': binding.guard.process_id,
            'parent_process_id': binding.guard.parent_process_id,
            'verifier_module': binding.verifier.__module__,
        }}, sort_keys=True))
        """
    )
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        (sys.executable, "-I", "-S", "-B", "-c", script),
        env=environment,
        capture_output=True,
        check=False,
        text=True,
        timeout=20.0,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.splitlines()[-1])
    assert result["module_loaded"] is True
    assert result["process_id"] == result["os_pid"]
    assert result["parent_process_id"] == os.getpid()
    assert result["verifier_module"] == module_name
    assert result["after"]["NoNewPrivs"] == 1
    assert result["after"]["Seccomp"] == 2
    assert result["after"]["Seccomp_filters"] == (
        result["before"]["Seccomp_filters"] + 1
    )
    assert module_name not in sys.modules


def test_full_guarded_fd3_route_uses_real_process_guard_in_fresh_subprocess() -> None:
    module_name = "falsewake.experiment_002_process_guard"
    assert module_name not in sys.modules
    script = textwrap.dedent(
        f"""
        import errno
        import importlib
        import json
        import os
        import socket
        import sys
        from types import SimpleNamespace

        sys.path.insert(0, {os.fspath(SOURCE_ROOT)!r})
        import falsewake.experiment_002_coordinator as coordinator
        import falsewake.experiment_002_run_authority as authority

        authority.reverify_verified_run_registration = lambda _registration: None
        synthetic_state = SimpleNamespace(
            head_commit='1' * 40,
            implementation_commit='2' * 40,
            registration_sha256='3' * 64,
            source_bundle_sha256='4' * 64,
            profile=authority._EXPERIMENT_002_PROFILE,
        )
        authority._verified_state = lambda _registration: synthetic_state
        registration = object.__new__(coordinator.VerifiedRunRegistration)
        parent_channel, child_channel = socket.socketpair(
            socket.AF_UNIX,
            socket.SOCK_SEQPACKET,
        )
        child_pid = os.fork()
        if child_pid == 0:
            try:
                authority._AUTHORITY_PROCESS_ID = os.getpid()
                parent_channel.close()
                child_fd = child_channel.fileno()
                if child_fd == coordinator._CHILD_CONTROL_FD:
                    child_channel.detach()
                    os.set_inheritable(coordinator._CHILD_CONTROL_FD, False)
                else:
                    os.dup2(
                        child_fd,
                        coordinator._CHILD_CONTROL_FD,
                        inheritable=False,
                    )
                    child_channel.close()

                original_import = importlib.import_module

                def worker(supplied_registration, activation, /):
                    if supplied_registration is not registration:
                        raise AssertionError('registration identity changed')
                    coordinator.verify_verified_child_activation(
                        registration,
                        activation,
                    )
                    try:
                        os.fstat(coordinator._CHILD_CONTROL_FD)
                    except OSError as error:
                        if error.errno != errno.EBADF:
                            raise
                    else:
                        raise AssertionError('FD3 remained open in worker')
                    try:
                        unexpected_pid = os.fork()
                    except OSError as error:
                        if error.errno != errno.EPERM:
                            raise
                    else:
                        if unexpected_pid == 0:
                            os._exit(97)
                        os.waitpid(unexpected_pid, 0)
                        raise AssertionError('process guard allowed fork')

                def dynamic_import(name):
                    if name == coordinator._WORKER_MODULE:
                        return SimpleNamespace(
                            run_registered_seed_process=worker
                        )
                    return original_import(name)

                importlib.import_module = dynamic_import
                coordinator._run_guarded_registered_seed_child(registration)
                os._exit(0)
            except BaseException:
                os._exit(91)

        child_channel.close()
        coordinator._parent_activate_child(
            parent_channel,
            registration,
            role=coordinator._SEED_ROLE,
            seed=coordinator.REGISTERED_SEEDS[0],
            child_pid=child_pid,
        )
        parent_channel.close()
        waited_pid, status = os.waitpid(child_pid, 0)
        assert waited_pid == child_pid
        assert os.WIFEXITED(status)
        assert os.WEXITSTATUS(status) == 0
        print(json.dumps({{'status': 'ok'}}))
        """
    )
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        (sys.executable, "-I", "-S", "-B", "-c", script),
        env=environment,
        capture_output=True,
        check=False,
        text=True,
        timeout=20.0,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout.splitlines()[-1]) == {"status": "ok"}
    assert module_name not in sys.modules


def test_public_parent_surface_has_one_positional_only_argument() -> None:
    signature = inspect.signature(coordinator.run_registered_experiment)
    parameters = list(signature.parameters.values())
    assert len(parameters) == 1
    assert parameters[0].name == "registration"
    assert parameters[0].kind is inspect.Parameter.POSITIONAL_ONLY
    assert signature.return_annotation in {None, "None"}
    with pytest.raises(TypeError):
        cast(Any, coordinator.run_registered_experiment)(registration=object())


def test_public_parent_burns_before_rejecting_mutated_admission_route() -> None:
    script = textwrap.dedent(
        f"""
        import json
        import sys

        sys.path.insert(0, {os.fspath(SOURCE_ROOT)!r})
        import falsewake.experiment_002_coordinator as coordinator

        registration = object.__new__(coordinator.VerifiedRunRegistration)
        original = coordinator.verify_verified_run_registration
        calls = []

        def replacement(value):
            calls.append(value)

        coordinator.verify_verified_run_registration = replacement
        first = None
        first_cause = None
        try:
            coordinator.run_registered_experiment(registration)
        except coordinator.Experiment002CoordinatorError as error:
            first = str(error)
            first_cause = str(error.__cause__)
        coordinator.verify_verified_run_registration = original
        second = None
        try:
            coordinator.run_registered_experiment(registration)
        except coordinator.Experiment002CoordinatorError as error:
            second = str(error)
        closure = dict(zip(
            coordinator.run_registered_experiment.__code__.co_freevars,
            coordinator.run_registered_experiment.__closure__ or (),
            strict=True,
        ))
        state = closure['state'].cell_contents
        print(json.dumps({{
            'calls': len(calls),
            'first': first,
            'first_cause': first_cause,
            'second': second,
            'attempted': state.attempted,
            'in_flight': state.in_flight,
        }}, sort_keys=True))
        """
    )
    completed = subprocess.run(
        (sys.executable, "-I", "-S", "-B", "-c", script),
        capture_output=True,
        check=False,
        text=True,
        timeout=20.0,
    )
    assert completed.returncode == 0, completed.stderr
    observed = json.loads(completed.stdout)
    assert observed == {
        "attempted": True,
        "calls": 0,
        "first": "registered experiment admission failed closed",
        "first_cause": "registered experiment coordinator authority changed",
        "in_flight": True,
        "second": "registered experiment was already attempted",
    }


def test_public_parent_rejects_forged_and_subclassed_registrations() -> None:
    class Subclass(authority.VerifiedRunRegistration):
        pass

    for value in (object(), object.__new__(Subclass)):
        with pytest.raises(TypeError, match="exact VerifiedRunRegistration"):
            coordinator.run_registered_experiment(cast(Any, value))


def test_private_parent_child_surface_has_no_override_parameters() -> None:
    signature = inspect.signature(coordinator._run_one_registered_parent_child)
    parameters = list(signature.parameters.values())
    assert [parameter.name for parameter in parameters] == [
        "registration",
        "assignment",
        "cpu_ids",
        "source_bundle_fd",
    ]
    assert all(
        parameter.kind is inspect.Parameter.POSITIONAL_ONLY
        and parameter.default is inspect.Parameter.empty
        for parameter in parameters
    )
    assert signature.return_annotation in {
        coordinator._RegisteredParentChildResult,
        "_RegisteredParentChildResult",
    }
    forbidden = {"plan", "limits", "kernel", "callback", "environment", "path"}
    assert forbidden.isdisjoint(signature.parameters)


def test_private_parent_child_happy_path_snapshots_exact_authorities_and_borrows_fd(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = _install_synthetic_parent_execution(
        monkeypatch,
        synthetic_registration,
    )
    assignment = coordinator._assignment(
        coordinator._SEED_ROLE,
        coordinator.REGISTERED_SEEDS[1],
    )
    cpu_ids = (2, 7)
    bundle_path = tmp_path / "borrowed-bundle"
    bundle_path.write_bytes(b"synthetic sealed bytes")
    source_bundle_fd = os.open(bundle_path, os.O_RDONLY | os.O_CLOEXEC)
    os.lseek(source_bundle_fd, 5, os.SEEK_SET)
    before = os.fstat(source_bundle_fd)
    before_offset = os.lseek(source_bundle_fd, 0, os.SEEK_CUR)
    before_inheritable = os.get_inheritable(source_bundle_fd)
    try:
        result = coordinator._run_one_registered_parent_child(
            synthetic_registration,
            assignment,
            cpu_ids,
            source_bundle_fd,
        )
        after = os.fstat(source_bundle_fd)
        after_offset = os.lseek(source_bundle_fd, 0, os.SEEK_CUR)
        after_inheritable = os.get_inheritable(source_bundle_fd)
    finally:
        os.close(source_bundle_fd)

    assert type(result) is tuple
    (
        result_pid,
        result_cpu_ids,
        elapsed_nanoseconds,
        maximum_rss_bytes,
        output_and_scratch_bytes,
        verified_child_snapshot,
    ) = result
    assert result_pid > 0
    assert result_pid != os.getpid()
    assert result_cpu_ids == cpu_ids
    assert elapsed_nanoseconds == 11
    assert maximum_rss_bytes == 12
    assert output_and_scratch_bytes == 13
    assert type(verified_child_snapshot) is tuple
    assert len(verified_child_snapshot) == 21
    assert verified_child_snapshot[:3] == (
        assignment.role,
        assignment.ordinal,
        assignment.seed,
    )
    assert verified_child_snapshot[17:19] == (3, 4)
    assert verified_child_snapshot[19] == (0.5).hex()
    assert harness.imports == [
        coordinator._SUPERVISOR_MODULE,
        coordinator._CHILD_RESULT_MODULE,
    ]
    assert len(harness.calls) == 1
    call_cpu_ids, _activation, call_fd, child_binding = harness.calls[0]
    assert call_cpu_ids == cpu_ids
    assert call_fd == source_bundle_fd
    assert type(child_binding) is harness.binding_type
    expected_binding = (
        assignment.role,
        assignment.ordinal,
        assignment.seed,
        TEST_BINDING.head_commit,
        TEST_BINDING.implementation_commit,
        TEST_BINDING.registration_sha256,
        TEST_BINDING.source_bundle_sha256,
    )
    assert harness.binding_frames == [expected_binding]
    assert len(harness.activations) == 1
    _channel, supplied_registration, role, seed, child_pid = harness.activations[0]
    assert supplied_registration is synthetic_registration
    assert (role, seed, child_pid) == (
        assignment.role,
        assignment.seed,
        result_pid,
    )
    assert len(harness.snapshot_calls) == 2
    assert harness.snapshot_calls[0] is harness.snapshot_calls[1]
    assert harness.verification_calls == [harness.snapshot_calls[0]]
    assert all(value is not harness.snapshot_calls[0] for value in result)
    assert (after.st_dev, after.st_ino, after.st_size) == (
        before.st_dev,
        before.st_ino,
        before.st_size,
    )
    assert after_offset == before_offset
    assert after_inheritable is before_inheritable


def test_parent_child_uses_detached_assignment_after_caller_object_mutation(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_synthetic_parent_execution(
        monkeypatch,
        synthetic_registration,
    )
    assignment = coordinator._assignment(
        coordinator._SEED_ROLE,
        coordinator.REGISTERED_SEEDS[0],
    )
    original_behavior = harness.behavior[0]

    def behavior(
        cpu_ids: tuple[int, int],
        activation: _SyntheticActivation,
        source_bundle_fd: int,
        binding: object,
    ) -> object:
        object.__setattr__(assignment, "seed", coordinator.REGISTERED_SEEDS[1])
        object.__setattr__(assignment, "ordinal", 1)
        return original_behavior(cpu_ids, activation, source_bundle_fd, binding)

    harness.behavior[0] = behavior
    result = coordinator._run_one_registered_parent_child(
        synthetic_registration,
        assignment,
        (2, 7),
        91,
    )
    assert result[5][2] == coordinator.REGISTERED_SEEDS[0]
    assert result[5][1] == 0
    assert harness.activations[0][3] == coordinator.REGISTERED_SEEDS[0]


def test_parent_child_input_snapshot_authority_ignores_pre_call_route_replacement(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_synthetic_parent_execution(
        monkeypatch,
        synthetic_registration,
    )
    original_assignment = coordinator._assignment(
        coordinator._SEED_ROLE,
        coordinator.REGISTERED_SEEDS[0],
    )
    redirected_assignment = coordinator._assignment(
        coordinator._SEED_ROLE,
        coordinator.REGISTERED_SEEDS[1],
    )

    def redirect_assignment(
        _assignment: coordinator._Assignment,
        /,
    ) -> coordinator._Assignment:
        return redirected_assignment

    def redirect_cpu_ids(_cpu_ids: tuple[int, int], /) -> tuple[int, int]:
        return (11, 13)

    monkeypatch.setattr(
        coordinator,
        "_snapshot_registered_assignment",
        redirect_assignment,
    )
    monkeypatch.setattr(
        coordinator,
        "_snapshot_registered_cpu_ids",
        redirect_cpu_ids,
    )
    result = coordinator._run_one_registered_parent_child(
        synthetic_registration,
        original_assignment,
        (2, 7),
        91,
    )
    assert result[1] == (2, 7)
    assert result[5][2] == coordinator.REGISTERED_SEEDS[0]
    assert result[5][1] == 0
    assert harness.calls[0][0] == (2, 7)
    assert harness.activations[0][3] == coordinator.REGISTERED_SEEDS[0]


def test_parent_child_rejects_pre_call_assignment_descriptor_replacement(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_synthetic_parent_execution(
        monkeypatch,
        synthetic_registration,
    )
    assignment = coordinator._assignment(
        coordinator._SEED_ROLE,
        coordinator.REGISTERED_SEEDS[0],
    )
    monkeypatch.setattr(
        coordinator._Assignment,
        "role",
        property(lambda _assignment: coordinator._SEED_ROLE),
    )
    monkeypatch.setattr(
        coordinator._Assignment,
        "seed",
        property(lambda _assignment: coordinator.REGISTERED_SEEDS[1]),
    )
    monkeypatch.setattr(
        coordinator._Assignment,
        "ordinal",
        property(lambda _assignment: 1),
    )
    with pytest.raises(
        coordinator.Experiment002CoordinatorError,
        match="input descriptors changed",
    ):
        coordinator._run_one_registered_parent_child(
            synthetic_registration,
            assignment,
            (2, 7),
            91,
        )
    assert harness.calls == []
    assert harness.activations == []


def test_parent_child_rejects_pre_call_registration_descriptor_replacement(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_synthetic_parent_execution(
        monkeypatch,
        synthetic_registration,
    )
    replacements = {
        "head_commit": "a" * 40,
        "implementation_commit": "b" * 40,
        "registration_sha256": "c" * 64,
        "source_bundle_sha256": "d" * 64,
    }

    def constant_property(value: str) -> property:
        def read(_binding: object) -> str:
            return value

        return property(read)

    for name, replacement in replacements.items():
        monkeypatch.setattr(
            coordinator._RegistrationBinding,
            name,
            constant_property(replacement),
        )
    with pytest.raises(
        coordinator.Experiment002CoordinatorError,
        match="input descriptors changed",
    ):
        coordinator._run_one_registered_parent_child(
            synthetic_registration,
            coordinator._assignment(
                coordinator._SEED_ROLE,
                coordinator.REGISTERED_SEEDS[0],
            ),
            (2, 7),
            91,
        )
    assert harness.calls == []
    assert harness.activations == []


@pytest.mark.parametrize(
    "invalid_input",
    ["assignment", "cpu-list", "cpu-bool", "cpu-order", "fd-bool"],
)
def test_parent_child_rejects_invalid_exact_inputs_before_dynamic_import(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
    invalid_input: str,
) -> None:
    harness = _install_synthetic_parent_execution(
        monkeypatch,
        synthetic_registration,
    )
    assignment = coordinator._assignment(
        coordinator._SEED_ROLE,
        coordinator.REGISTERED_SEEDS[0],
    )
    cpu_ids: object = (2, 7)
    source_bundle_fd: object = 91
    if invalid_input == "assignment":
        assignment = coordinator._Assignment(
            role=coordinator._SEED_ROLE,
            seed=coordinator.REGISTERED_SEEDS[0],
            ordinal=3,
        )
    elif invalid_input == "cpu-list":
        cpu_ids = [2, 7]
    elif invalid_input == "cpu-bool":
        cpu_ids = (True, 7)
    elif invalid_input == "cpu-order":
        cpu_ids = (7, 2)
    else:
        source_bundle_fd = True
    with pytest.raises(coordinator.Experiment002CoordinatorError):
        coordinator._run_one_registered_parent_child(
            synthetic_registration,
            assignment,
            cast(Any, cpu_ids),
            cast(Any, source_bundle_fd),
        )
    assert harness.imports == []


@pytest.mark.parametrize(
    "tamper",
    [
        "module_type",
        "module_name",
        "function_type",
        "function_name",
        "function_module",
        "result_type_module",
        "child_binding_alias",
        "child_loader_alias",
        "child_snapshot_type",
        "child_snapshot_name",
        "child_snapshot_module",
    ],
)
def test_parent_execution_route_claim_rejects_spoofed_exact_identities(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    harness = _install_synthetic_parent_execution(
        monkeypatch,
        synthetic_registration,
    )
    if tamper == "module_type":
        original_import = importlib.import_module

        def dynamic_import(name: str) -> object:
            if name == coordinator._SUPERVISOR_MODULE:
                return SimpleNamespace(__name__=name)
            return original_import(name)

        monkeypatch.setattr(importlib, "import_module", dynamic_import)
    elif tamper == "module_name":
        harness.supervisor_module.__name__ = "falsewake.spoofed_supervisor"
    elif tamper == "function_type":
        setattr(
            harness.supervisor_module,
            coordinator._SUPERVISE_CHILD_FUNCTION,
            object(),
        )
    elif tamper == "function_name":
        route = getattr(
            harness.supervisor_module,
            coordinator._SUPERVISE_CHILD_FUNCTION,
        )
        cast(Any, route).__name__ = "spoofed_supervise_child"
    elif tamper == "function_module":
        route = getattr(
            harness.supervisor_module,
            coordinator._SUPERVISE_CHILD_FUNCTION,
        )
        cast(Any, route).__module__ = "falsewake.spoofed_supervisor"
    elif tamper == "result_type_module":
        cast(
            Any, harness.supervised_result_type
        ).__module__ = "falsewake.spoofed_supervisor"
    elif tamper == "child_binding_alias":
        setattr(
            harness.supervisor_module,
            coordinator._CHILD_RESULT_BINDING_TYPE,
            type("ChildResultBinding", (), {}),
        )
    elif tamper == "child_loader_alias":

        def different_loader(_path: object, _binding: object, /) -> object:
            raise AssertionError

        _set_route_identity(
            different_loader,
            coordinator._CHILD_RESULT_LOADER_FUNCTION,
            coordinator._CHILD_RESULT_MODULE,
        )
        setattr(
            harness.supervisor_module,
            coordinator._CHILD_RESULT_LOADER,
            different_loader,
        )
    elif tamper == "child_snapshot_type":
        setattr(
            harness.child_result_module,
            coordinator._CHILD_RESULT_SNAPSHOTTER,
            object(),
        )
    else:
        snapshot_route = getattr(
            harness.child_result_module,
            coordinator._CHILD_RESULT_SNAPSHOTTER,
        )
        if tamper == "child_snapshot_name":
            cast(Any, snapshot_route).__name__ = "spoofed_snapshot_route"
        else:
            cast(Any, snapshot_route).__module__ = "falsewake.spoofed_child_result"

    with pytest.raises(
        coordinator.Experiment002CoordinatorError,
        match="identity|routes",
    ):
        coordinator._run_one_registered_parent_child(
            synthetic_registration,
            coordinator._assignment(
                coordinator._SEED_ROLE,
                coordinator.REGISTERED_SEEDS[0],
            ),
            (2, 7),
            91,
        )
    assert harness.calls == []


@pytest.mark.parametrize(
    "mutation",
    ["module_attribute", "function_metadata", "snapshot_module_attribute"],
)
def test_parent_execution_route_mutation_during_supervision_fails_closed(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    harness = _install_synthetic_parent_execution(
        monkeypatch,
        synthetic_registration,
    )
    original_behavior = harness.behavior[0]

    def behavior(
        cpu_ids: tuple[int, int],
        activation: _SyntheticActivation,
        source_bundle_fd: int,
        binding: object,
    ) -> object:
        result = original_behavior(cpu_ids, activation, source_bundle_fd, binding)
        route = getattr(
            harness.supervisor_module,
            coordinator._SUPERVISE_CHILD_FUNCTION,
        )
        if mutation == "snapshot_module_attribute":

            def replacement_snapshot(_result: object, /) -> object:
                raise AssertionError

            _set_route_identity(
                replacement_snapshot,
                coordinator._CHILD_RESULT_SNAPSHOTTER_FUNCTION,
                coordinator._CHILD_RESULT_MODULE,
            )
            setattr(
                harness.child_result_module,
                coordinator._CHILD_RESULT_SNAPSHOTTER,
                replacement_snapshot,
            )
            return result
        if mutation == "module_attribute":

            def replacement(*_args: object) -> object:
                raise AssertionError

            _set_route_identity(
                replacement,
                coordinator._SUPERVISE_CHILD_FUNCTION,
                coordinator._SUPERVISOR_MODULE,
            )
            setattr(
                harness.supervisor_module,
                coordinator._SUPERVISE_CHILD_FUNCTION,
                replacement,
            )
        else:
            cast(Any, route).__module__ = "falsewake.spoofed_supervisor"
        return result

    harness.behavior[0] = behavior
    with pytest.raises(
        coordinator.Experiment002CoordinatorError,
        match="identity|routes",
    ):
        coordinator._run_one_registered_parent_child(
            synthetic_registration,
            coordinator._assignment(
                coordinator._SEED_ROLE,
                coordinator.REGISTERED_SEEDS[0],
            ),
            (2, 7),
            91,
        )


def test_parent_execution_route_mutation_is_a_boundary_failure_on_supervisor_error(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_synthetic_parent_execution(
        monkeypatch,
        synthetic_registration,
    )

    def behavior(
        _cpu_ids: tuple[int, int],
        _activation: _SyntheticActivation,
        _source_bundle_fd: int,
        _binding: object,
    ) -> object:
        harness.supervisor_module.__name__ = "falsewake.spoofed_supervisor"
        raise RuntimeError("synthetic supervisor failure")

    harness.behavior[0] = behavior
    with pytest.raises(
        coordinator.Experiment002CoordinatorError,
        match="parent boundary changed",
    ) as caught:
        coordinator._run_one_registered_parent_child(
            synthetic_registration,
            coordinator._assignment(
                coordinator._SEED_ROLE,
                coordinator.REGISTERED_SEEDS[0],
            ),
            (2, 7),
            91,
        )
    assert type(caught.value.__cause__) is RuntimeError


@pytest.mark.parametrize(
    "free_variable",
    ["verified_state", "verified_truth", "authority_values"],
)
def test_real_route_integrity_detects_nested_closure_cell_replacement(
    free_variable: str,
) -> None:
    routes = coordinator._claim_and_verify_parent_execution_routes()
    integrity_frame = routes[11]
    matching_cells = [
        cell_frame
        for node in integrity_frame
        for cell_frame in node[11]
        if cell_frame[1] == free_variable
        and cell_frame[2]
        and type(cell_frame[3]) is FunctionType
    ]
    assert matching_cells
    cell, _name, _occupied, original = matching_cells[0]
    cell.cell_contents = lambda _value: None
    try:
        with pytest.raises(
            coordinator.Experiment002CoordinatorError,
            match="closure content changed after claim",
        ):
            coordinator._require_parent_execution_routes_unchanged(routes)
    finally:
        cell.cell_contents = original
    coordinator._require_parent_execution_routes_unchanged(routes)


def test_claimed_snapshotter_and_verifier_code_replacement_fails_closed(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_synthetic_parent_execution(monkeypatch, synthetic_registration)
    original_behavior = harness.behavior[0]
    snapshotter = cast(
        FunctionType,
        getattr(harness.child_result_module, coordinator._CHILD_RESULT_SNAPSHOTTER),
    )
    verifier = cast(
        FunctionType,
        getattr(harness.child_result_module, coordinator._CHILD_RESULT_VERIFIER),
    )
    snapshotter_code = snapshotter.__code__
    verifier_code = verifier.__code__

    def behavior(
        cpu_ids: tuple[int, int],
        activation: _SyntheticActivation,
        source_bundle_fd: int,
        binding: object,
    ) -> object:
        result = original_behavior(cpu_ids, activation, source_bundle_fd, binding)
        snapshotter.__code__ = snapshotter_code.replace(co_name="forged_snapshot")
        verifier.__code__ = verifier_code.replace(co_name="forged_verifier")
        return result

    harness.behavior[0] = behavior
    try:
        with pytest.raises(
            coordinator.Experiment002CoordinatorError,
            match="executable route changed after claim|parent boundary changed",
        ):
            coordinator._run_one_registered_parent_child(
                synthetic_registration,
                coordinator._assignment(
                    coordinator._SEED_ROLE,
                    coordinator.REGISTERED_SEEDS[0],
                ),
                (2, 7),
                91,
            )
    finally:
        snapshotter.__code__ = snapshotter_code
        verifier.__code__ = verifier_code
    assert harness.snapshot_calls == []
    assert harness.verification_calls == []


def test_supervised_slot_descriptor_replacement_after_claim_fails_closed(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_synthetic_parent_execution(monkeypatch, synthetic_registration)
    original_behavior = harness.behavior[0]

    def behavior(
        cpu_ids: tuple[int, int],
        activation: _SyntheticActivation,
        source_bundle_fd: int,
        binding: object,
    ) -> object:
        result = original_behavior(cpu_ids, activation, source_bundle_fd, binding)
        monkeypatch.setattr(
            harness.supervised_result_type,
            "elapsed_nanoseconds",
            property(lambda _result: 0),
        )
        return result

    harness.behavior[0] = behavior
    with pytest.raises(
        coordinator.Experiment002CoordinatorError,
        match="dynamic class changed|parent boundary changed",
    ):
        coordinator._run_one_registered_parent_child(
            synthetic_registration,
            coordinator._assignment(
                coordinator._SEED_ROLE,
                coordinator.REGISTERED_SEEDS[0],
            ),
            (2, 7),
            91,
        )


def test_supervised_constructor_code_replacement_after_claim_fails_closed(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_synthetic_parent_execution(monkeypatch, synthetic_registration)
    original_behavior = harness.behavior[0]
    constructor = cast(
        FunctionType,
        harness.supervised_result_type.__dict__["__init__"],
    )
    constructor_code = constructor.__code__

    def behavior(
        cpu_ids: tuple[int, int],
        activation: _SyntheticActivation,
        source_bundle_fd: int,
        binding: object,
    ) -> object:
        result = original_behavior(cpu_ids, activation, source_bundle_fd, binding)
        constructor.__code__ = constructor_code.replace(co_name="forged_init")
        return result

    harness.behavior[0] = behavior
    try:
        with pytest.raises(
            coordinator.Experiment002CoordinatorError,
            match="executable route changed after claim|parent boundary changed",
        ):
            coordinator._run_one_registered_parent_child(
                synthetic_registration,
                coordinator._assignment(
                    coordinator._SEED_ROLE,
                    coordinator.REGISTERED_SEEDS[0],
                ),
                (2, 7),
                91,
            )
    finally:
        constructor.__code__ = constructor_code


@pytest.mark.parametrize(
    "route_name",
    [
        "_require_registration",
        "_registration_binding",
        "_parent_activate_child",
        "_verified_child_result_snapshot_frame",
    ],
)
def test_pre_call_critical_coordinator_route_replacement_is_rejected(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
    route_name: str,
) -> None:
    harness = _install_synthetic_parent_execution(monkeypatch, synthetic_registration)
    replacement_calls: list[object] = []

    def replacement(*_args: object, **_kwargs: object) -> object:
        replacement_calls.append(object())
        return None

    monkeypatch.setattr(coordinator, route_name, replacement)
    with pytest.raises(
        coordinator.Experiment002CoordinatorError,
        match="coordinator.*routes changed",
    ):
        coordinator._run_one_registered_parent_child(
            synthetic_registration,
            coordinator._assignment(
                coordinator._SEED_ROLE,
                coordinator.REGISTERED_SEEDS[0],
            ),
            (2, 7),
            91,
        )
    assert replacement_calls == []
    assert harness.calls == []
    assert harness.activations == []


def test_child_result_binding_constructor_is_not_an_authority_route(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_synthetic_parent_execution(
        monkeypatch,
        synthetic_registration,
    )

    constructor_calls: list[object] = []

    def forbidden_constructor(
        _cls: type[object],
        *_args: object,
        **_kwargs: object,
    ) -> object:
        constructor_calls.append(object())
        return object()

    monkeypatch.setattr(
        harness.binding_type,
        "__new__",
        staticmethod(forbidden_constructor),
    )
    result = coordinator._run_one_registered_parent_child(
        synthetic_registration,
        coordinator._assignment(
            coordinator._SEED_ROLE,
            coordinator.REGISTERED_SEEDS[0],
        ),
        (2, 7),
        91,
    )
    assert result[5][:3] == (
        coordinator._SEED_ROLE,
        0,
        coordinator.REGISTERED_SEEDS[0],
    )
    assert constructor_calls == []


def test_child_result_binding_stable_property_spoof_is_rejected_at_claim(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_synthetic_parent_execution(
        monkeypatch,
        synthetic_registration,
    )

    class PropertySpoofBinding:
        role = property(lambda _self: coordinator._SEED_ROLE)
        ordinal = property(lambda _self: 0)
        seed = property(lambda _self: coordinator.REGISTERED_SEEDS[0])
        registration_head_commit = property(lambda _self: TEST_BINDING.head_commit)
        implementation_commit = property(
            lambda _self: TEST_BINDING.implementation_commit
        )
        registration_sha256 = property(lambda _self: TEST_BINDING.registration_sha256)
        source_bundle_sha256 = property(lambda _self: TEST_BINDING.source_bundle_sha256)

    _set_route_identity(
        PropertySpoofBinding,
        coordinator._CHILD_RESULT_BINDING_TYPE,
        coordinator._CHILD_RESULT_MODULE,
    )
    setattr(
        harness.child_result_module,
        coordinator._CHILD_RESULT_BINDING_TYPE,
        PropertySpoofBinding,
    )
    setattr(
        harness.supervisor_module,
        coordinator._CHILD_RESULT_BINDING_TYPE,
        PropertySpoofBinding,
    )
    with pytest.raises(
        coordinator.Experiment002CoordinatorError,
        match="slot descriptors are invalid",
    ):
        coordinator._run_one_registered_parent_child(
            synthetic_registration,
            coordinator._assignment(
                coordinator._SEED_ROLE,
                coordinator.REGISTERED_SEEDS[0],
            ),
            (2, 7),
            91,
        )
    assert harness.calls == []


def test_pre_call_registration_binding_route_replacement_is_rejected(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_synthetic_parent_execution(
        monkeypatch,
        synthetic_registration,
    )
    binding_calls = 0

    def registration_binding(
        supplied: coordinator.VerifiedRunRegistration,
    ) -> coordinator._RegistrationBinding:
        nonlocal binding_calls
        assert supplied is synthetic_registration
        binding_calls += 1
        return replace(TEST_BINDING, head_commit="a" * 40)

    monkeypatch.setattr(coordinator, "_registration_binding", registration_binding)
    with pytest.raises(
        coordinator.Experiment002CoordinatorError,
        match="coordinator.*routes changed",
    ):
        coordinator._run_one_registered_parent_child(
            synthetic_registration,
            coordinator._assignment(
                coordinator._SEED_ROLE,
                coordinator.REGISTERED_SEEDS[0],
            ),
            (2, 7),
            91,
        )
    assert binding_calls == 0
    assert harness.calls == []


@pytest.mark.parametrize(
    "activation_case",
    ["missing", "double", "swallowed_failure", "pid_mismatch"],
)
def test_parent_owned_activation_closure_requires_one_success_matching_result(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
    activation_case: str,
) -> None:
    harness = _install_synthetic_parent_execution(
        monkeypatch,
        synthetic_registration,
    )
    child_pid = os.getpid() + 100_000

    def behavior(
        cpu_ids: tuple[int, int],
        activation: _SyntheticActivation,
        _source_bundle_fd: int,
        binding: object,
    ) -> object:
        activated_child_pid = child_pid
        if activation_case == "swallowed_failure":
            with pytest.raises(coordinator.Experiment002CoordinatorError):
                activated_child_pid = _fail_synthetic_parent_activation(activation)
        elif activation_case != "missing":
            typed_binding = cast(Any, binding)
            activated_child_pid = _complete_synthetic_parent_activation(
                activation,
                synthetic_registration,
                typed_binding.role,
                typed_binding.seed,
                harness.activations,
            )
        if activation_case == "double":
            parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
            try:
                with pytest.raises(coordinator.Experiment002CoordinatorError):
                    activation(parent, activated_child_pid)
            finally:
                parent.close()
                child.close()
        result_pid = (
            activated_child_pid + 1
            if activation_case == "pid_mismatch"
            else activated_child_pid
        )
        return _synthetic_supervised_result(
            harness,
            binding,
            pid=result_pid,
            cpu_ids=cpu_ids,
        )

    harness.behavior[0] = behavior
    with pytest.raises(coordinator.Experiment002CoordinatorError):
        coordinator._run_one_registered_parent_child(
            synthetic_registration,
            coordinator._assignment(
                coordinator._SEED_ROLE,
                coordinator.REGISTERED_SEEDS[0],
            ),
            (2, 7),
            91,
        )
    assert len(harness.calls) == 1


@pytest.mark.parametrize("supervisor_outcome", ["error", "missing_activation"])
def test_retained_activation_callback_is_closed_after_supervisor_finishes(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
    supervisor_outcome: str,
) -> None:
    harness = _install_synthetic_parent_execution(
        monkeypatch,
        synthetic_registration,
    )
    retained: list[_SyntheticActivation] = []
    child_pid = os.getpid() + 100_000

    def behavior(
        cpu_ids: tuple[int, int],
        activation: _SyntheticActivation,
        _source_bundle_fd: int,
        binding: object,
    ) -> object:
        retained.append(activation)
        if supervisor_outcome == "error":
            raise RuntimeError("stable retained-callback supervisor failure")
        return _synthetic_supervised_result(
            harness,
            binding,
            pid=child_pid,
            cpu_ids=cpu_ids,
        )

    harness.behavior[0] = behavior
    if supervisor_outcome == "error":
        with pytest.raises(RuntimeError, match="retained-callback supervisor failure"):
            coordinator._run_one_registered_parent_child(
                synthetic_registration,
                coordinator._assignment(
                    coordinator._SEED_ROLE,
                    coordinator.REGISTERED_SEEDS[0],
                ),
                (2, 7),
                91,
            )
    else:
        with pytest.raises(
            coordinator.Experiment002CoordinatorError,
            match="returned without one completed child activation",
        ):
            coordinator._run_one_registered_parent_child(
                synthetic_registration,
                coordinator._assignment(
                    coordinator._SEED_ROLE,
                    coordinator.REGISTERED_SEEDS[0],
                ),
                (2, 7),
                91,
            )

    assert len(retained) == 1
    assert harness.activations == []
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    try:
        with pytest.raises(
            coordinator.Experiment002CoordinatorError,
            match="no longer accepting",
        ):
            retained[0](parent, child_pid)
    finally:
        parent.close()
        child.close()
    assert harness.activations == []


def test_activation_uses_captured_coordinator_route_and_detects_global_replacement(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_synthetic_parent_execution(
        monkeypatch,
        synthetic_registration,
    )
    original_behavior = harness.behavior[0]
    bypass_calls: list[int] = []

    def bypass(
        _channel: socket.socket,
        _registration: coordinator.VerifiedRunRegistration,
        *,
        role: str,
        seed: int,
        child_pid: int,
    ) -> None:
        del role, seed
        bypass_calls.append(child_pid)

    def behavior(
        cpu_ids: tuple[int, int],
        activation: _SyntheticActivation,
        source_bundle_fd: int,
        binding: object,
    ) -> object:
        monkeypatch.setattr(coordinator, "_parent_activate_child", bypass)
        return original_behavior(cpu_ids, activation, source_bundle_fd, binding)

    harness.behavior[0] = behavior
    with pytest.raises(
        coordinator.Experiment002CoordinatorError,
        match="coordinator parent execution routes changed",
    ):
        coordinator._run_one_registered_parent_child(
            synthetic_registration,
            coordinator._assignment(
                coordinator._SEED_ROLE,
                coordinator.REGISTERED_SEEDS[0],
            ),
            (2, 7),
            91,
        )
    assert len(harness.activations) == 1
    assert bypass_calls == []


@pytest.mark.parametrize(
    "result_case",
    [
        "wrong_supervised_type",
        "supervised_subclass",
        "bool_elapsed",
        "negative_rss",
        "cpu_mismatch",
        "wrong_child_type",
        "child_subclass",
    ],
)
def test_supervised_result_exact_type_metric_and_child_matrix_fails_closed(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
    result_case: str,
) -> None:
    harness = _install_synthetic_parent_execution(
        monkeypatch,
        synthetic_registration,
    )

    def behavior(
        cpu_ids: tuple[int, int],
        activation: _SyntheticActivation,
        _source_bundle_fd: int,
        binding: object,
    ) -> object:
        typed_binding = cast(Any, binding)
        child_pid = _complete_synthetic_parent_activation(
            activation,
            synthetic_registration,
            typed_binding.role,
            typed_binding.seed,
            harness.activations,
        )
        if result_case == "wrong_supervised_type":
            return object()
        verified: object | None = None
        elapsed: object = 11
        maximum_rss: object = 12
        result_cpu_ids = cpu_ids
        if result_case == "supervised_subclass":
            subclass = type(
                "SyntheticSupervisedSubclass",
                (harness.supervised_result_type,),
                {},
            )
            return subclass(
                pid=child_pid,
                cpu_ids=cpu_ids,
                elapsed_nanoseconds=11,
                maximum_rss_bytes=12,
                output_and_scratch_bytes=13,
                verified_child_result=harness.verified_result_type(binding),
            )
        if result_case == "bool_elapsed":
            elapsed = True
        elif result_case == "negative_rss":
            maximum_rss = -1
        elif result_case == "cpu_mismatch":
            result_cpu_ids = (cpu_ids[0], cpu_ids[1] + 1)
        elif result_case == "wrong_child_type":
            verified = object()
        elif result_case == "child_subclass":
            child_subclass = type(
                "SyntheticVerifiedChildSubclass",
                (harness.verified_result_type,),
                {},
            )
            verified = child_subclass(binding)
        return _synthetic_supervised_result(
            harness,
            binding,
            pid=child_pid,
            cpu_ids=result_cpu_ids,
            elapsed_nanoseconds=elapsed,
            maximum_rss_bytes=maximum_rss,
            verified_child_result=verified,
        )

    harness.behavior[0] = behavior
    with pytest.raises(coordinator.Experiment002CoordinatorError):
        coordinator._run_one_registered_parent_child(
            synthetic_registration,
            coordinator._assignment(
                coordinator._SEED_ROLE,
                coordinator.REGISTERED_SEEDS[0],
            ),
            (2, 7),
            91,
        )


def test_verified_child_result_verifier_must_return_none(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_synthetic_parent_execution(
        monkeypatch,
        synthetic_registration,
    )

    def unexpected_verifier(_result: object, /) -> object:
        return object()

    _set_route_identity(
        unexpected_verifier,
        coordinator._CHILD_RESULT_VERIFIER_FUNCTION,
        coordinator._CHILD_RESULT_MODULE,
    )
    setattr(
        harness.child_result_module,
        coordinator._CHILD_RESULT_VERIFIER,
        unexpected_verifier,
    )
    setattr(
        harness.supervisor_module,
        coordinator._CHILD_RESULT_VERIFIER,
        unexpected_verifier,
    )
    with pytest.raises(
        coordinator.Experiment002CoordinatorError,
        match="verifier returned an unexpected value",
    ):
        coordinator._run_one_registered_parent_child(
            synthetic_registration,
            coordinator._assignment(
                coordinator._SEED_ROLE,
                coordinator.REGISTERED_SEEDS[0],
            ),
            (2, 7),
            91,
        )


def test_verifier_mutation_is_caught_between_authoritative_snapshots(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_synthetic_parent_execution(
        monkeypatch,
        synthetic_registration,
    )
    replacement_binding = harness.binding_type(
        role=coordinator._SEED_ROLE,
        ordinal=1,
        seed=coordinator.REGISTERED_SEEDS[1],
        registration_head_commit=TEST_BINDING.head_commit,
        implementation_commit=TEST_BINDING.implementation_commit,
        registration_sha256=TEST_BINDING.registration_sha256,
        source_bundle_sha256=TEST_BINDING.source_bundle_sha256,
    )
    verifier_calls: list[object] = []

    def mutating_verifier(result: object, /) -> None:
        if type(result) is not harness.verified_result_type:
            raise TypeError("not an exact synthetic result")
        verifier_calls.append(result)
        cast(Any, result)._binding = replacement_binding

    _set_route_identity(
        mutating_verifier,
        coordinator._CHILD_RESULT_VERIFIER_FUNCTION,
        coordinator._CHILD_RESULT_MODULE,
    )
    setattr(
        harness.child_result_module,
        coordinator._CHILD_RESULT_VERIFIER,
        mutating_verifier,
    )
    setattr(
        harness.supervisor_module,
        coordinator._CHILD_RESULT_VERIFIER,
        mutating_verifier,
    )
    with pytest.raises(
        coordinator.Experiment002CoordinatorError,
        match="snapshots changed across verification",
    ):
        coordinator._run_one_registered_parent_child(
            synthetic_registration,
            coordinator._assignment(
                coordinator._SEED_ROLE,
                coordinator.REGISTERED_SEEDS[0],
            ),
            (2, 7),
            91,
        )
    assert len(verifier_calls) == 1


def test_two_valid_but_different_authoritative_snapshots_are_rejected(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_synthetic_parent_execution(
        monkeypatch,
        synthetic_registration,
    )
    original_snapshot = harness.snapshot_behavior[0]
    snapshot_calls = 0

    def changing_snapshot(result: object, /) -> object:
        nonlocal snapshot_calls
        snapshot_calls += 1
        snapshot = cast(tuple[object, ...], original_snapshot(result))
        if snapshot_calls == 1:
            return snapshot
        fields = list(snapshot)
        fields[16] = 8
        return tuple(fields)

    harness.snapshot_behavior[0] = changing_snapshot
    with pytest.raises(
        coordinator.Experiment002CoordinatorError,
        match="snapshots changed across verification",
    ):
        coordinator._run_one_registered_parent_child(
            synthetic_registration,
            coordinator._assignment(
                coordinator._SEED_ROLE,
                coordinator.REGISTERED_SEEDS[0],
            ),
            (2, 7),
            91,
        )
    assert snapshot_calls == 2
    assert len(harness.verification_calls) == 1


@pytest.mark.parametrize(
    "snapshot_case",
    ["wrong_type", "wrong_shape", "wrong_field_type", "wrong_binding"],
)
def test_authoritative_snapshot_exact_type_shape_and_binding_fail_closed(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
    snapshot_case: str,
) -> None:
    harness = _install_synthetic_parent_execution(
        monkeypatch,
        synthetic_registration,
    )
    original_snapshot = harness.snapshot_behavior[0]

    def malformed_snapshot(result: object, /) -> object:
        snapshot = cast(tuple[object, ...], original_snapshot(result))
        if snapshot_case == "wrong_type":
            return list(snapshot)
        if snapshot_case == "wrong_shape":
            return snapshot[:-1]
        fields = list(snapshot)
        if snapshot_case == "wrong_field_type":
            fields[8] = True
        else:
            fields[1] = 1
            fields[2] = coordinator.REGISTERED_SEEDS[1]
        return tuple(fields)

    harness.snapshot_behavior[0] = malformed_snapshot
    with pytest.raises(
        coordinator.Experiment002CoordinatorError,
        match="snapshot",
    ):
        coordinator._run_one_registered_parent_child(
            synthetic_registration,
            coordinator._assignment(
                coordinator._SEED_ROLE,
                coordinator.REGISTERED_SEEDS[0],
            ),
            (2, 7),
            91,
        )
    assert harness.verification_calls == []


def test_stable_post_supervision_verifier_failure_preserves_primary_error(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_synthetic_parent_execution(
        monkeypatch,
        synthetic_registration,
    )

    def failing_verifier(_result: object, /) -> None:
        raise RuntimeError("stable post-supervision verifier failure")

    _set_route_identity(
        failing_verifier,
        coordinator._CHILD_RESULT_VERIFIER_FUNCTION,
        coordinator._CHILD_RESULT_MODULE,
    )
    setattr(
        harness.child_result_module,
        coordinator._CHILD_RESULT_VERIFIER,
        failing_verifier,
    )
    setattr(
        harness.supervisor_module,
        coordinator._CHILD_RESULT_VERIFIER,
        failing_verifier,
    )
    with pytest.raises(
        RuntimeError,
        match="stable post-supervision verifier failure",
    ):
        coordinator._run_one_registered_parent_child(
            synthetic_registration,
            coordinator._assignment(
                coordinator._SEED_ROLE,
                coordinator.REGISTERED_SEEDS[0],
            ),
            (2, 7),
            91,
        )


def test_post_supervision_failure_is_masked_when_parent_boundary_also_changes(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_synthetic_parent_execution(
        monkeypatch,
        synthetic_registration,
    )

    def mutating_failing_verifier(_result: object, /) -> None:
        harness.child_result_module.__name__ = "falsewake.spoofed_child_result"
        raise RuntimeError("post-supervision failure after boundary mutation")

    _set_route_identity(
        mutating_failing_verifier,
        coordinator._CHILD_RESULT_VERIFIER_FUNCTION,
        coordinator._CHILD_RESULT_MODULE,
    )
    setattr(
        harness.child_result_module,
        coordinator._CHILD_RESULT_VERIFIER,
        mutating_failing_verifier,
    )
    setattr(
        harness.supervisor_module,
        coordinator._CHILD_RESULT_VERIFIER,
        mutating_failing_verifier,
    )
    with pytest.raises(
        coordinator.Experiment002CoordinatorError,
        match="parent execution routes changed",
    ) as caught:
        coordinator._run_one_registered_parent_child(
            synthetic_registration,
            coordinator._assignment(
                coordinator._SEED_ROLE,
                coordinator.REGISTERED_SEEDS[0],
            ),
            (2, 7),
            91,
        )
    assert type(caught.value.__cause__) is RuntimeError


def test_authoritative_snapshot_ignores_public_identity_property_spoof(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_synthetic_parent_execution(
        monkeypatch,
        synthetic_registration,
    )

    class StringSubclass(str):
        pass

    monkeypatch.setattr(
        harness.verified_result_type,
        "role",
        property(lambda result: StringSubclass(result.binding.role)),
    )
    result = coordinator._run_one_registered_parent_child(
        synthetic_registration,
        coordinator._assignment(
            coordinator._SEED_ROLE,
            coordinator.REGISTERED_SEEDS[0],
        ),
        (2, 7),
        91,
    )
    assert result[5][:3] == (
        coordinator._SEED_ROLE,
        0,
        coordinator.REGISTERED_SEEDS[0],
    )
    assert type(result[5][0]) is str


@pytest.mark.parametrize(
    ("property_name", "value"),
    [
        ("winner_epoch", 30),
        ("winner_epoch", True),
        ("winner_macro_f1", Fraction(2, 1)),
        ("winner_validation_cross_entropy", float("inf")),
        ("winner_validation_cross_entropy", float("nan")),
        ("winner_validation_cross_entropy", -0.0),
        ("model_tensor_sha256", "not-a-digest"),
    ],
    ids=[
        "epoch-bound",
        "epoch-bool",
        "f1-range",
        "ce-inf",
        "ce-nan",
        "ce-negzero",
        "model",
    ],
)
def test_authoritative_snapshot_ignores_public_rank_property_spoofs(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
    property_name: str,
    value: object,
) -> None:
    harness = _install_synthetic_parent_execution(
        monkeypatch,
        synthetic_registration,
    )
    monkeypatch.setattr(
        harness.verified_result_type,
        property_name,
        property(lambda _result: value),
    )
    result = coordinator._run_one_registered_parent_child(
        synthetic_registration,
        coordinator._assignment(
            coordinator._SEED_ROLE,
            coordinator.REGISTERED_SEEDS[0],
        ),
        (2, 7),
        91,
    )
    assert result[5][16:] == (7, 3, 4, (0.5).hex(), "5" * 64)


def test_authoritative_snapshot_never_reads_mutating_public_rank_property(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_synthetic_parent_execution(
        monkeypatch,
        synthetic_registration,
    )
    epochs = iter((7, 8))
    monkeypatch.setattr(
        harness.verified_result_type,
        "winner_epoch",
        property(lambda _result: next(epochs)),
    )
    result = coordinator._run_one_registered_parent_child(
        synthetic_registration,
        coordinator._assignment(
            coordinator._SEED_ROLE,
            coordinator.REGISTERED_SEEDS[0],
        ),
        (2, 7),
        91,
    )
    assert result[5][16] == 7
    assert next(epochs) == 7


def test_supervised_result_requires_exact_member_descriptors(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_synthetic_parent_execution(
        monkeypatch,
        synthetic_registration,
    )

    class UnstableSupervisedResult:
        def __init__(
            self,
            *,
            pid: int,
            cpu_ids: tuple[int, int],
            verified_child_result: object,
        ) -> None:
            self.pid = pid
            self.cpu_ids = cpu_ids
            self._elapsed_reads = 0
            self.maximum_rss_bytes = 12
            self.output_and_scratch_bytes = 13
            self.verified_child_result = verified_child_result

        @property
        def elapsed_nanoseconds(self) -> int:
            self._elapsed_reads += 1
            return 10 + self._elapsed_reads

    _set_route_identity(
        UnstableSupervisedResult,
        coordinator._SUPERVISED_CHILD_RESULT_TYPE,
        coordinator._SUPERVISOR_MODULE,
    )
    setattr(
        harness.supervisor_module,
        coordinator._SUPERVISED_CHILD_RESULT_TYPE,
        UnstableSupervisedResult,
    )

    with pytest.raises(
        coordinator.Experiment002CoordinatorError,
        match="slot descriptors are invalid",
    ):
        coordinator._run_one_registered_parent_child(
            synthetic_registration,
            coordinator._assignment(
                coordinator._SEED_ROLE,
                coordinator.REGISTERED_SEEDS[0],
            ),
            (2, 7),
            91,
        )
    assert harness.calls == []


def test_verified_child_binding_must_match_registered_assignment(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_synthetic_parent_execution(
        monkeypatch,
        synthetic_registration,
    )

    def behavior(
        cpu_ids: tuple[int, int],
        activation: _SyntheticActivation,
        _source_bundle_fd: int,
        binding: object,
    ) -> object:
        typed_binding = cast(Any, binding)
        child_pid = _complete_synthetic_parent_activation(
            activation,
            synthetic_registration,
            typed_binding.role,
            typed_binding.seed,
            harness.activations,
        )
        different = harness.binding_type(
            role=coordinator._SEED_ROLE,
            ordinal=1,
            seed=coordinator.REGISTERED_SEEDS[1],
            registration_head_commit=TEST_BINDING.head_commit,
            implementation_commit=TEST_BINDING.implementation_commit,
            registration_sha256=TEST_BINDING.registration_sha256,
            source_bundle_sha256=TEST_BINDING.source_bundle_sha256,
        )
        return _synthetic_supervised_result(
            harness,
            different,
            pid=child_pid,
            cpu_ids=cpu_ids,
        )

    harness.behavior[0] = behavior
    with pytest.raises(
        coordinator.Experiment002CoordinatorError,
        match="differs from its registered assignment",
    ):
        coordinator._run_one_registered_parent_child(
            synthetic_registration,
            coordinator._assignment(
                coordinator._SEED_ROLE,
                coordinator.REGISTERED_SEEDS[0],
            ),
            (2, 7),
            91,
        )


def test_registration_change_during_result_verification_fails_closed(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_synthetic_parent_execution(
        monkeypatch,
        synthetic_registration,
    )
    changed_state = SimpleNamespace(
        head_commit=TEST_BINDING.head_commit,
        implementation_commit=TEST_BINDING.implementation_commit,
        registration_sha256=TEST_BINDING.registration_sha256,
        source_bundle_sha256="a" * 64,
    )

    def changing_verifier(result: object, /) -> None:
        if type(result) is not harness.verified_result_type:
            raise TypeError("not an exact synthetic result")
        monkeypatch.setattr(
            authority,
            "_verified_state",
            lambda _registration: changed_state,
        )

    _set_route_identity(
        changing_verifier,
        coordinator._CHILD_RESULT_VERIFIER_FUNCTION,
        coordinator._CHILD_RESULT_MODULE,
    )
    setattr(
        harness.child_result_module,
        coordinator._CHILD_RESULT_VERIFIER,
        changing_verifier,
    )
    setattr(
        harness.supervisor_module,
        coordinator._CHILD_RESULT_VERIFIER,
        changing_verifier,
    )
    with pytest.raises(coordinator.Experiment002CoordinatorError, match="registration"):
        coordinator._run_one_registered_parent_child(
            synthetic_registration,
            coordinator._assignment(
                coordinator._SEED_ROLE,
                coordinator.REGISTERED_SEEDS[0],
            ),
            (2, 7),
            91,
        )


def test_parent_child_helper_reuses_one_borrowed_fd_and_cpu_pair(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = _install_synthetic_parent_execution(
        monkeypatch,
        synthetic_registration,
    )
    bundle_path = tmp_path / "reused-bundle"
    bundle_path.write_bytes(b"one parent-owned descriptor")
    source_bundle_fd = os.open(bundle_path, os.O_RDONLY | os.O_CLOEXEC)
    os.lseek(source_bundle_fd, 4, os.SEEK_SET)
    before = os.fstat(source_bundle_fd)
    before_offset = os.lseek(source_bundle_fd, 0, os.SEEK_CUR)
    cpu_ids = (2, 7)
    try:
        results = [
            coordinator._run_one_registered_parent_child(
                synthetic_registration,
                coordinator._assignment(coordinator._SEED_ROLE, seed),
                cpu_ids,
                source_bundle_fd,
            )
            for seed in coordinator.REGISTERED_SEEDS[:2]
        ]
        after = os.fstat(source_bundle_fd)
        after_offset = os.lseek(source_bundle_fd, 0, os.SEEK_CUR)
    finally:
        os.close(source_bundle_fd)
    assert [result[5][2] for result in results] == list(
        coordinator.REGISTERED_SEEDS[:2]
    )
    assert len(harness.calls) == 2
    assert all(
        call[0] == cpu_ids and call[2] == source_bundle_fd for call in harness.calls
    )
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
    assert after_offset == before_offset


def test_stable_supervisor_failure_propagates_and_keeps_borrowed_fd_open(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = _install_synthetic_parent_execution(
        monkeypatch,
        synthetic_registration,
    )

    def behavior(
        _cpu_ids: tuple[int, int],
        _activation: _SyntheticActivation,
        _source_bundle_fd: int,
        _binding: object,
    ) -> object:
        raise RuntimeError("stable synthetic supervisor failure")

    harness.behavior[0] = behavior
    bundle_path = tmp_path / "failed-bundle"
    bundle_path.write_bytes(b"still caller owned")
    source_bundle_fd = os.open(bundle_path, os.O_RDONLY | os.O_CLOEXEC)
    os.lseek(source_bundle_fd, 3, os.SEEK_SET)
    before = os.fstat(source_bundle_fd)
    before_offset = os.lseek(source_bundle_fd, 0, os.SEEK_CUR)
    try:
        with pytest.raises(RuntimeError, match="stable synthetic supervisor failure"):
            coordinator._run_one_registered_parent_child(
                synthetic_registration,
                coordinator._assignment(
                    coordinator._SEED_ROLE,
                    coordinator.REGISTERED_SEEDS[0],
                ),
                (2, 7),
                source_bundle_fd,
            )
        after = os.fstat(source_bundle_fd)
        after_offset = os.lseek(source_bundle_fd, 0, os.SEEK_CUR)
    finally:
        os.close(source_bundle_fd)
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
    assert after_offset == before_offset


@pytest.mark.parametrize(
    ("role", "seed", "ordinal"),
    [
        (coordinator._SEED_ROLE, coordinator.REGISTERED_SEEDS[0], 0),
        (coordinator._SEED_ROLE, coordinator.REGISTERED_SEEDS[1], 1),
        (coordinator._SEED_ROLE, coordinator.REGISTERED_SEEDS[2], 2),
        (coordinator._RERUN_ROLE, coordinator.REGISTERED_SEEDS[0], 3),
        (coordinator._RERUN_ROLE, coordinator.REGISTERED_SEEDS[2], 3),
    ],
)
def test_only_registered_seed_and_rerun_assignments_exist(
    role: str,
    seed: int,
    ordinal: int,
) -> None:
    assert coordinator._assignment(role, seed) == coordinator._Assignment(
        role=role,
        seed=seed,
        ordinal=ordinal,
    )


@pytest.mark.parametrize(
    ("role", "seed"),
    [
        (coordinator._SEED_ROLE, 7),
        (coordinator._RERUN_ROLE, 7),
        ("benchmark", coordinator.REGISTERED_SEEDS[0]),
        ("", coordinator.REGISTERED_SEEDS[0]),
    ],
)
def test_unregistered_assignments_fail_closed(role: str, seed: int) -> None:
    with pytest.raises(coordinator.Experiment002CoordinatorError):
        coordinator._assignment(role, seed)
    with pytest.raises(TypeError):
        coordinator._assignment(role, cast(Any, True))


def test_assignment_rejects_equality_compatible_primitive_subclasses() -> None:
    class IntegerSubclass(int):
        pass

    class StringSubclass(str):
        pass

    with pytest.raises(TypeError, match="role"):
        coordinator._assignment(
            cast(Any, StringSubclass(coordinator._SEED_ROLE)),
            coordinator.REGISTERED_SEEDS[0],
        )
    with pytest.raises(TypeError, match="seed"):
        coordinator._assignment(
            coordinator._SEED_ROLE,
            cast(Any, IntegerSubclass(coordinator.REGISTERED_SEEDS[0])),
        )
    assignment = coordinator._Assignment(
        role=coordinator._SEED_ROLE,
        seed=coordinator.REGISTERED_SEEDS[1],
        ordinal=cast(Any, True),
    )
    with pytest.raises(
        coordinator.Experiment002CoordinatorError,
        match="exact types",
    ):
        coordinator._require_assignment(assignment)


def test_canonical_ticket_packet_round_trip_binds_every_field() -> None:
    assignment = coordinator._assignment(
        coordinator._SEED_ROLE,
        coordinator.REGISTERED_SEEDS[1],
    )
    document = coordinator._ticket_document(
        TEST_BINDING,
        assignment,
        parent_pid=101,
        child_pid=202,
        nonce="a" * 64,
    )
    packet = coordinator._encode_packet(document)
    assert coordinator._decode_packet(packet) == document
    magic, length = coordinator._FRAME_HEADER.unpack(
        packet[: coordinator._FRAME_HEADER.size]
    )
    payload = packet[coordinator._FRAME_HEADER.size :]
    assert magic == coordinator._FRAME_MAGIC
    assert length == len(payload)
    assert payload.endswith(b"\n")
    assert payload == (
        json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")
    assert len(packet) <= coordinator._MAX_PACKET_BYTES


@pytest.mark.parametrize(
    "packet",
    [
        b"",
        b"short",
        _raw_packet(b'{"a":1,"a":2}\n'),
        _raw_packet(b'{"value":1.0}\n'),
        _raw_packet(b'{"value":NaN}\n'),
        _raw_packet(b'{ "value":1}\n'),
        _raw_packet(b'{"value":1}'),
        _raw_packet(b"{}\n", magic=b"BADMAGIC"),
        coordinator._FRAME_HEADER.pack(coordinator._FRAME_MAGIC, 99) + b"{}\n",
    ],
)
def test_noncanonical_or_malformed_packets_are_rejected(packet: bytes) -> None:
    with pytest.raises(coordinator.Experiment002CoordinatorError):
        coordinator._decode_packet(packet)


def test_packet_size_limits_are_enforced() -> None:
    with pytest.raises(coordinator.Experiment002CoordinatorError, match="JSON size"):
        coordinator._encode_packet({"value": "x" * coordinator._MAX_JSON_BYTES})
    oversized = b"x" * (coordinator._MAX_PACKET_BYTES + 1)
    with pytest.raises(coordinator.Experiment002CoordinatorError, match="size"):
        coordinator._decode_packet(oversized)


def test_seqpacket_transport_delivers_kernel_credentials() -> None:
    sender, receiver = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    packet = coordinator._encode_packet({"kind": "synthetic"})
    try:
        coordinator._prepare_channel(receiver)
        coordinator._send_packet(sender, packet)
        observed, credentials = coordinator._receive_packet(receiver)
    finally:
        sender.close()
        receiver.close()
    assert observed == packet
    assert credentials == coordinator._PeerCredentials(
        pid=os.getpid(),
        uid=os.getuid(),
        gid=os.getgid(),
    )


def test_real_msg_peek_is_nondestructive_and_consume_matches_exact_frame() -> None:
    sender, receiver = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    packet = coordinator._encode_packet({"kind": "peek_probe"})
    try:
        coordinator._prepare_channel(receiver)
        coordinator._send_packet(sender, packet)
        first = coordinator._peek_packet(receiver)
        second = coordinator._peek_packet(receiver)
        consumed = coordinator._consume_packet(receiver)
    finally:
        sender.close()
        receiver.close()

    coordinator._require_same_received_packet(first, second)
    coordinator._require_same_received_packet(first, consumed)
    assert first.packet == packet
    assert first.address is None
    assert second.address is None
    assert consumed.address is None
    assert first.credentials == coordinator._PeerCredentials(
        pid=os.getpid(),
        uid=os.getuid(),
        gid=os.getgid(),
    )


def test_msg_peek_rejects_scm_rights_without_leaking_received_descriptor() -> None:
    sender, receiver = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    source_descriptor = os.open("/dev/null", os.O_RDONLY)
    packet = coordinator._encode_packet({"kind": "rights_probe"})
    credentials = coordinator._CREDENTIALS.pack(
        os.getpid(),
        os.getuid(),
        os.getgid(),
    )
    ancillary = [
        (socket.SOL_SOCKET, socket.SCM_CREDENTIALS, credentials),
        (
            socket.SOL_SOCKET,
            socket.SCM_RIGHTS,
            array.array("i", [source_descriptor]).tobytes(),
        ),
    ]
    try:
        coordinator._prepare_channel(receiver)
        sender.sendmsg([packet], ancillary)
        descriptor_count = len(os.listdir("/proc/self/fd"))
        with pytest.raises(
            coordinator.Experiment002CoordinatorError,
            match="SCM_RIGHTS",
        ):
            coordinator._peek_packet(receiver)
        assert len(os.listdir("/proc/self/fd")) == descriptor_count
    finally:
        os.close(source_descriptor)
        sender.close()
        receiver.close()


def test_rights_cleanup_sweeps_all_descriptors_before_reporting_close_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempted: list[int] = []

    def close(descriptor: int) -> None:
        attempted.append(descriptor)
        if descriptor == 101:
            raise OSError("injected close failure")

    monkeypatch.setattr(os, "close", close)
    ancillary = [
        (
            socket.SOL_SOCKET,
            socket.SCM_RIGHTS,
            array.array("i", [101, 102]).tobytes(),
        )
    ]
    with pytest.raises(
        coordinator.Experiment002CoordinatorError,
        match="could not be closed",
    ):
        coordinator._close_received_rights(ancillary)
    assert attempted == [101, 102]


def test_stream_and_inet_channels_are_rejected() -> None:
    first, second = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        with pytest.raises(
            coordinator.Experiment002CoordinatorError,
            match="SOCK_SEQPACKET",
        ):
            coordinator._prepare_channel(first)
    finally:
        first.close()
        second.close()
    inet = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        with pytest.raises(
            coordinator.Experiment002CoordinatorError,
            match="AF_UNIX",
        ):
            coordinator._prepare_channel(inet)
    finally:
        inet.close()


def test_ticket_validation_binds_kernel_sender_pids_and_registration() -> None:
    assignment = coordinator._assignment(
        coordinator._SEED_ROLE,
        coordinator.REGISTERED_SEEDS[0],
    )
    document = coordinator._ticket_document(
        TEST_BINDING,
        assignment,
        parent_pid=os.getppid(),
        child_pid=os.getpid(),
        nonce="7" * 64,
    )
    packet = coordinator._encode_packet(document)
    credentials = coordinator._PeerCredentials(
        pid=os.getppid(), uid=os.getuid(), gid=os.getgid()
    )
    observed = coordinator._validate_ticket(
        document,
        credentials,
        binding=TEST_BINDING,
        packet=packet,
    )
    assert observed.assignment == assignment
    assert observed.parent_pid == os.getppid()
    assert observed.child_pid == os.getpid()
    assert observed.packet_sha256 == coordinator._ticket_sha256(packet)

    wrong_sender = replace(credentials, pid=os.getppid() + 100_000)
    with pytest.raises(coordinator.Experiment002CoordinatorError, match="sender"):
        coordinator._validate_ticket(
            document,
            wrong_sender,
            binding=TEST_BINDING,
            packet=packet,
        )

    wrong_binding = replace(TEST_BINDING, registration_sha256="8" * 64)
    with pytest.raises(coordinator.Experiment002CoordinatorError, match="registration"):
        coordinator._validate_ticket(
            document,
            credentials,
            binding=wrong_binding,
            packet=packet,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("kind", "ticket", "kind"),
        ("schema_version", 2, "schema"),
        ("parent_pid", 1, "sender"),
        ("child_pid", 1, "binding"),
        ("ordinal", 2, "ordinal"),
        ("role", "benchmark", "registered"),
        ("seed", 7, "registered"),
        ("nonce", "A" * 64, "lowercase hex"),
    ],
)
def test_ticket_scalar_tampering_is_rejected(
    field: str,
    value: object,
    message: str,
) -> None:
    assignment = coordinator._assignment(
        coordinator._SEED_ROLE,
        coordinator.REGISTERED_SEEDS[0],
    )
    document = coordinator._ticket_document(
        TEST_BINDING,
        assignment,
        parent_pid=os.getppid(),
        child_pid=os.getpid(),
        nonce="9" * 64,
    )
    document[field] = value
    packet = coordinator._encode_packet(document)
    credentials = coordinator._PeerCredentials(
        pid=os.getppid(), uid=os.getuid(), gid=os.getgid()
    )
    with pytest.raises(coordinator.Experiment002CoordinatorError, match=message):
        coordinator._validate_ticket(
            document,
            credentials,
            binding=TEST_BINDING,
            packet=packet,
        )


def test_extra_and_missing_ticket_fields_are_rejected() -> None:
    assignment = coordinator._assignment(
        coordinator._SEED_ROLE,
        coordinator.REGISTERED_SEEDS[0],
    )
    base = coordinator._ticket_document(
        TEST_BINDING,
        assignment,
        parent_pid=os.getppid(),
        child_pid=os.getpid(),
        nonce="a" * 64,
    )
    credentials = coordinator._PeerCredentials(
        pid=os.getppid(), uid=os.getuid(), gid=os.getgid()
    )
    variants = (
        {**base, "callback": "forbidden"},
        {k: v for k, v in base.items() if k != "seed"},
    )
    for document in variants:
        packet = coordinator._encode_packet(document)
        with pytest.raises(coordinator.Experiment002CoordinatorError, match="fields"):
            coordinator._validate_ticket(
                document,
                credentials,
                binding=TEST_BINDING,
                packet=packet,
            )


def test_acknowledgement_is_exact_and_pid_bound() -> None:
    ticket = _ticket()
    document = coordinator._ack_document(ticket)
    credentials = coordinator._PeerCredentials(
        pid=os.getpid(), uid=os.getuid(), gid=os.getgid()
    )
    coordinator._validate_ack(
        document,
        credentials,
        assignment=ticket.assignment,
        parent_pid=ticket.parent_pid,
        child_pid=ticket.child_pid,
        nonce=ticket.nonce,
        packet_sha256=ticket.packet_sha256,
    )
    for field, value in (
        ("status", "ready"),
        ("ticket_sha256", "f" * 64),
        ("seed", coordinator.REGISTERED_SEEDS[1]),
        ("nonce", "0" * 64),
    ):
        changed = dict(document)
        changed[field] = value
        with pytest.raises(
            coordinator.Experiment002CoordinatorError, match="does not match"
        ):
            coordinator._validate_ack(
                changed,
                credentials,
                assignment=ticket.assignment,
                parent_pid=ticket.parent_pid,
                child_pid=ticket.child_pid,
                nonce=ticket.nonce,
                packet_sha256=ticket.packet_sha256,
            )
    with pytest.raises(
        coordinator.Experiment002CoordinatorError, match="does not match"
    ):
        coordinator._validate_ack(
            document,
            replace(credentials, pid=os.getpid() + 1),
            assignment=ticket.assignment,
            parent_pid=ticket.parent_pid,
            child_pid=ticket.child_pid,
            nonce=ticket.nonce,
            packet_sha256=ticket.packet_sha256,
        )
    boolean_ordinal = dict(document)
    boolean_ordinal["ordinal"] = False
    with pytest.raises(TypeError, match="ordinal"):
        coordinator._validate_ack(
            boolean_ordinal,
            credentials,
            assignment=ticket.assignment,
            parent_pid=ticket.parent_pid,
            child_pid=ticket.child_pid,
            nonce=ticket.nonce,
            packet_sha256=ticket.packet_sha256,
        )


def test_ticket_and_nonce_reservations_are_irreversibly_one_shot() -> None:
    ticket = _ticket()
    coordinator._reserve_issued_ticket(ticket.nonce, ticket.packet_sha256)
    with pytest.raises(
        coordinator.Experiment002CoordinatorError, match="already issued"
    ):
        coordinator._reserve_issued_ticket(ticket.nonce, "7" * 64)
    with pytest.raises(
        coordinator.Experiment002CoordinatorError, match="already issued"
    ):
        coordinator._reserve_issued_ticket("8" * 64, ticket.packet_sha256)
    coordinator._ISSUED_TICKET_NONCES.clear()
    coordinator._ISSUED_TICKET_DIGESTS.clear()
    with pytest.raises(
        coordinator.Experiment002CoordinatorError, match="already issued"
    ):
        coordinator._reserve_issued_ticket(ticket.nonce, ticket.packet_sha256)

    coordinator._reserve_accepted_ticket(ticket)
    with pytest.raises(coordinator.Experiment002CoordinatorError, match="replayed"):
        coordinator._reserve_accepted_ticket(ticket)
    same_nonce = replace(ticket, packet_sha256="9" * 64)
    with pytest.raises(coordinator.Experiment002CoordinatorError, match="replayed"):
        coordinator._reserve_accepted_ticket(same_nonce)
    same_digest = replace(ticket, nonce="a" * 64)
    with pytest.raises(coordinator.Experiment002CoordinatorError, match="replayed"):
        coordinator._reserve_accepted_ticket(same_digest)
    coordinator._ACCEPTED_TICKET_NONCES.clear()
    coordinator._ACCEPTED_TICKET_DIGESTS.clear()
    with pytest.raises(coordinator.Experiment002CoordinatorError, match="replayed"):
        coordinator._reserve_accepted_ticket(ticket)


def test_accepted_ticket_receipt_is_exact_bound_and_one_shot(
    synthetic_registration: coordinator.VerifiedRunRegistration,
) -> None:
    ticket = _ticket()
    forged_receipt = object.__new__(coordinator._AcceptedTicketReceipt)
    with pytest.raises(
        coordinator.Experiment002CoordinatorError,
        match="receipt",
    ):
        coordinator._issue_activation(
            synthetic_registration,
            ticket,
            forged_receipt,
            _process_guard_binding(),
        )

    receipt = coordinator._reserve_accepted_ticket(ticket)
    mismatched = replace(ticket, nonce="e" * 64)
    with pytest.raises(
        coordinator.Experiment002CoordinatorError,
        match="receipt",
    ):
        coordinator._issue_activation(
            synthetic_registration,
            mismatched,
            receipt,
            _process_guard_binding(),
        )
    activation = coordinator._issue_activation(
        synthetic_registration,
        ticket,
        receipt,
        _process_guard_binding(),
    )
    coordinator.verify_verified_child_activation(synthetic_registration, activation)
    with pytest.raises(
        coordinator.Experiment002CoordinatorError,
        match="receipt",
    ):
        coordinator._issue_activation(
            synthetic_registration,
            ticket,
            receipt,
            _process_guard_binding(),
        )


@pytest.mark.parametrize(
    ("role", "seed"),
    [
        (coordinator._SEED_ROLE, coordinator.REGISTERED_SEEDS[0]),
        (coordinator._SEED_ROLE, coordinator.REGISTERED_SEEDS[1]),
        (coordinator._SEED_ROLE, coordinator.REGISTERED_SEEDS[2]),
        (coordinator._RERUN_ROLE, coordinator.REGISTERED_SEEDS[1]),
    ],
)
def test_fixed_fd_protocol_round_trip_for_exact_plan_roles(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    role: str,
    seed: int,
) -> None:
    _fork_fixed_fd_round_trip(synthetic_registration, role=role, seed=seed)


@pytest.mark.parametrize(
    "stage",
    [
        "inheritable",
        "status_flags",
        "peek",
        "guard",
        "frame_tamper",
        "registration",
        "decode",
        "consume_mismatch",
        "credential_mismatch",
        "guard_reverify",
        "reserve",
        "issue",
        "activation_verify",
        "ack_encode",
        "ack_send",
    ],
)
def test_guarded_route_failure_matrix_has_no_ack_worker_or_retry(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    stage: str,
) -> None:
    _fork_guarded_route_failure(synthetic_registration, stage=stage)


def test_guarded_route_latch_is_claimed_before_argument_validation() -> None:
    pid = _fork()
    if pid == 0:
        try:
            with pytest.raises(TypeError, match="exact VerifiedRunRegistration"):
                cast(Any, coordinator._run_guarded_registered_seed_child)(object())
            with pytest.raises(
                coordinator.Experiment002CoordinatorError,
                match="already attempted",
            ):
                cast(Any, coordinator._run_registered_seed_child)(object())
            os._exit(0)
        except BaseException:
            os._exit(79)
    _wait_success(pid)


def test_parent_rejects_ack_sent_by_wrong_pid(
    synthetic_registration: coordinator.VerifiedRunRegistration,
) -> None:
    parent_channel, child_channel = socket.socketpair(
        socket.AF_UNIX,
        socket.SOCK_SEQPACKET,
    )
    fake_child_pid = os.getpid() + 100_000
    try:
        assignment = coordinator._assignment(
            coordinator._SEED_ROLE,
            coordinator.REGISTERED_SEEDS[0],
        )
        nonce = "b" * 64
        packet = coordinator._encode_packet(
            coordinator._ticket_document(
                TEST_BINDING,
                assignment,
                parent_pid=os.getpid(),
                child_pid=fake_child_pid,
                nonce=nonce,
            )
        )
        ticket = coordinator._Ticket(
            assignment=assignment,
            parent_pid=os.getpid(),
            child_pid=fake_child_pid,
            nonce=nonce,
            packet_sha256=coordinator._ticket_sha256(packet),
        )
        coordinator._prepare_channel(parent_channel)
        coordinator._send_packet(
            child_channel, coordinator._encode_packet(coordinator._ack_document(ticket))
        )
        ack_packet, credentials = coordinator._receive_packet(parent_channel)
        with pytest.raises(
            coordinator.Experiment002CoordinatorError, match="child PID"
        ):
            coordinator._validate_ack(
                coordinator._decode_packet(ack_packet),
                credentials,
                assignment=assignment,
                parent_pid=os.getpid(),
                child_pid=fake_child_pid,
                nonce=nonce,
                packet_sha256=ticket.packet_sha256,
            )
    finally:
        parent_channel.close()
        child_channel.close()


def test_activation_is_opaque_noncopyable_and_nonserializable(
    synthetic_registration: coordinator.VerifiedRunRegistration,
) -> None:
    with pytest.raises(TypeError, match="issued"):
        coordinator.VerifiedChildActivation()
    activation = _activation(synthetic_registration)
    assert activation.role == coordinator._SEED_ROLE
    assert activation.seed == coordinator.REGISTERED_SEEDS[0]
    assert activation.ordinal == 0
    assert activation.child_pid == os.getpid()
    coordinator.verify_verified_child_activation(synthetic_registration, activation)
    with pytest.raises(TypeError, match="copied"):
        copy.copy(activation)
    with pytest.raises(TypeError, match="copied"):
        copy.deepcopy(activation)
    with pytest.raises(TypeError, match="serialized"):
        pickle.dumps(activation)


def test_process_guard_is_identity_bound_reverified_and_failure_is_terminal(
    synthetic_registration: coordinator.VerifiedRunRegistration,
) -> None:
    guard = object()
    healthy = True
    calls: list[object] = []

    def verify(value: object, /) -> None:
        calls.append(value)
        if value is not guard or not healthy:
            raise coordinator.Experiment002CoordinatorError(
                "synthetic process guard failed"
            )

    binding = coordinator._ProcessGuardBinding(guard=guard, verifier=verify)
    ticket = _ticket()
    receipt = coordinator._reserve_accepted_ticket(ticket)
    activation = coordinator._issue_activation(
        synthetic_registration,
        ticket,
        receipt,
        binding,
    )
    state = coordinator._ACTIVATIONS[activation]
    assert state.process_guard is guard
    assert state.process_guard_verifier is verify
    assert coordinator._ACTIVATION_GUARDS[activation].process_guard is guard
    assert coordinator._ACTIVATION_ANCHORS[activation].process_guard is guard

    before = len(calls)
    coordinator.verify_verified_child_activation(synthetic_registration, activation)
    assert len(calls) >= before + 3
    assert all(value is guard for value in calls)

    healthy = False
    with pytest.raises(coordinator.Experiment002CoordinatorError):
        coordinator.verify_verified_child_activation(
            synthetic_registration,
            activation,
        )
    healthy = True
    coordinator._FAILED_ACTIVATIONS.discard(activation)
    with pytest.raises(coordinator.Experiment002CoordinatorError, match="corrupted"):
        coordinator.verify_verified_child_activation(
            synthetic_registration,
            activation,
        )


def test_coherent_process_guard_store_rewrite_fails_closure_identity_anchor(
    synthetic_registration: coordinator.VerifiedRunRegistration,
) -> None:
    class EqualityCompatibleGuard:
        def __eq__(self, other: object) -> bool:
            return isinstance(other, EqualityCompatibleGuard)

    original_guard = EqualityCompatibleGuard()
    replacement_guard = EqualityCompatibleGuard()

    def verifier(value: object, /) -> None:
        if value is not original_guard and value is not replacement_guard:
            raise AssertionError("unexpected equality-compatible guard")

    ticket = _ticket()
    receipt = coordinator._reserve_accepted_ticket(ticket)
    activation = coordinator._issue_activation(
        synthetic_registration,
        ticket,
        receipt,
        coordinator._ProcessGuardBinding(
            guard=original_guard,
            verifier=verifier,
        ),
    )
    original = coordinator._ACTIVATIONS[activation]
    assert original_guard == replacement_guard
    assert original_guard is not replacement_guard

    rewritten = replace(
        original,
        process_guard=replacement_guard,
    )
    coordinator._ACTIVATIONS[activation] = rewritten
    coordinator._ACTIVATION_GUARDS[activation] = coordinator._guard_from_state(
        rewritten
    )
    coordinator._ACTIVATION_ANCHORS[activation] = coordinator._anchor_from_state(
        rewritten
    )
    with pytest.raises(coordinator.Experiment002CoordinatorError, match="corrupted"):
        coordinator.verify_verified_child_activation(
            synthetic_registration,
            activation,
        )


def test_guard_and_anchor_matchers_reject_bool_before_scalar_equality(
    synthetic_registration: coordinator.VerifiedRunRegistration,
) -> None:
    activation = _activation(synthetic_registration)
    state = coordinator._ACTIVATIONS[activation]
    guard = replace(coordinator._guard_from_state(state), ordinal=cast(Any, True))
    anchor = replace(coordinator._anchor_from_state(state), seed=cast(Any, True))
    assert not coordinator._activation_guard_matches_state(guard, state)
    assert not coordinator._activation_anchor_matches_state(anchor, state)


def test_trusted_verifier_exception_irreversibly_poisons_activation(
    synthetic_registration: coordinator.VerifiedRunRegistration,
) -> None:
    activation = _activation(synthetic_registration)

    def explode(
        _activation: coordinator.VerifiedChildActivation,
        _state: object,
    ) -> tuple[int | None, bool, bool]:
        raise RuntimeError("injected trusted verifier failure")

    with pytest.raises(RuntimeError, match="trusted verifier"):
        coordinator._activation_state(activation, _trusted_verify=explode)
    coordinator._FAILED_ACTIVATIONS.discard(activation)
    with pytest.raises(coordinator.Experiment002CoordinatorError, match="corrupted"):
        coordinator.verify_verified_child_activation(
            synthetic_registration,
            activation,
        )


def test_activation_rejects_other_registration_and_stays_poisoned(
    synthetic_registration: coordinator.VerifiedRunRegistration,
) -> None:
    activation = _activation(synthetic_registration)
    other = object.__new__(coordinator.VerifiedRunRegistration)
    with pytest.raises(coordinator.Experiment002CoordinatorError, match="different"):
        coordinator.verify_verified_child_activation(other, activation)
    with pytest.raises(coordinator.Experiment002CoordinatorError, match="corrupted"):
        coordinator.verify_verified_child_activation(synthetic_registration, activation)


def test_activation_state_tamper_is_terminal_after_restore(
    synthetic_registration: coordinator.VerifiedRunRegistration,
) -> None:
    activation = _activation(synthetic_registration)
    original = coordinator._ACTIVATIONS[activation]
    coordinator._ACTIVATIONS[activation] = replace(
        original,
        seed=coordinator.REGISTERED_SEEDS[1],
    )
    with pytest.raises(coordinator.Experiment002CoordinatorError, match="corrupted"):
        coordinator.verify_verified_child_activation(synthetic_registration, activation)
    coordinator._ACTIVATIONS[activation] = original
    coordinator._ACTIVATION_GUARDS[activation] = coordinator._guard_from_state(original)
    with pytest.raises(coordinator.Experiment002CoordinatorError, match="corrupted"):
        coordinator.verify_verified_child_activation(synthetic_registration, activation)


def test_coherent_state_and_guard_tamper_is_caught_by_anchor(
    synthetic_registration: coordinator.VerifiedRunRegistration,
) -> None:
    activation = _activation(synthetic_registration)
    original = coordinator._ACTIVATIONS[activation]
    changed = replace(original, nonce="c" * 64, ticket_sha256="d" * 64)
    coordinator._ACTIVATIONS[activation] = changed
    coordinator._ACTIVATION_GUARDS[activation] = coordinator._guard_from_state(changed)
    with pytest.raises(coordinator.Experiment002CoordinatorError, match="corrupted"):
        coordinator.verify_verified_child_activation(synthetic_registration, activation)


def test_mutated_ticket_and_all_mutable_activation_stores_cannot_rewrite_issuance(
    synthetic_registration: coordinator.VerifiedRunRegistration,
) -> None:
    ticket = _ticket()
    receipt = coordinator._reserve_accepted_ticket(ticket)
    activation = coordinator._issue_activation(
        synthetic_registration,
        ticket,
        receipt,
        _process_guard_binding(),
    )
    original = coordinator._ACTIVATIONS[activation]
    original_assignment = ticket.assignment
    original_nonce = ticket.nonce
    original_digest = ticket.packet_sha256
    rewritten_assignment = coordinator._assignment(
        coordinator._SEED_ROLE,
        coordinator.REGISTERED_SEEDS[1],
    )
    object.__setattr__(ticket, "assignment", rewritten_assignment)
    object.__setattr__(ticket, "nonce", "c" * 64)
    object.__setattr__(ticket, "packet_sha256", "d" * 64)
    rewritten = replace(
        original,
        role=rewritten_assignment.role,
        seed=rewritten_assignment.seed,
        ordinal=rewritten_assignment.ordinal,
        nonce=ticket.nonce,
        ticket_sha256=ticket.packet_sha256,
    )
    coordinator._ACTIVATIONS[activation] = rewritten
    coordinator._ACTIVATION_GUARDS[activation] = coordinator._guard_from_state(
        rewritten
    )
    coordinator._ACTIVATION_ANCHORS[activation] = coordinator._anchor_from_state(
        rewritten
    )

    with pytest.raises(coordinator.Experiment002CoordinatorError, match="corrupted"):
        coordinator.verify_verified_child_activation(
            synthetic_registration,
            activation,
        )
    object.__setattr__(ticket, "assignment", original_assignment)
    object.__setattr__(ticket, "nonce", original_nonce)
    object.__setattr__(ticket, "packet_sha256", original_digest)
    coordinator._ACTIVATIONS[activation] = original
    coordinator._ACTIVATION_GUARDS[activation] = coordinator._guard_from_state(original)
    coordinator._ACTIVATION_ANCHORS[activation] = coordinator._anchor_from_state(
        original
    )
    coordinator._FAILED_ACTIVATIONS.discard(activation)
    with pytest.raises(coordinator.Experiment002CoordinatorError, match="corrupted"):
        coordinator.verify_verified_child_activation(
            synthetic_registration,
            activation,
        )


def test_full_store_coherent_assignment_rewrite_cannot_replace_trusted_issuance(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activation = _activation(synthetic_registration)
    original = coordinator._ACTIVATIONS[activation]
    assignment = coordinator._assignment(
        coordinator._RERUN_ROLE,
        coordinator.REGISTERED_SEEDS[2],
    )
    changed = replace(
        original,
        role=assignment.role,
        seed=assignment.seed,
        ordinal=assignment.ordinal,
        nonce="c" * 64,
        ticket_sha256="d" * 64,
    )

    monkeypatch.setattr(
        coordinator,
        "_ACTIVATIONS",
        weakref.WeakKeyDictionary({activation: changed}),
    )
    monkeypatch.setattr(
        coordinator,
        "_ACTIVATION_GUARDS",
        weakref.WeakKeyDictionary({activation: coordinator._guard_from_state(changed)}),
    )
    monkeypatch.setattr(
        coordinator,
        "_ACTIVATION_ANCHORS",
        weakref.WeakKeyDictionary(
            {activation: coordinator._anchor_from_state(changed)}
        ),
    )
    monkeypatch.setattr(
        coordinator,
        "_ACTIVATION_PHASES",
        weakref.WeakKeyDictionary({activation: coordinator._PHASE_ISSUED}),
    )
    monkeypatch.setattr(coordinator, "_DISPATCHED_ACTIVATIONS", weakref.WeakSet())
    monkeypatch.setattr(coordinator, "_COMPLETED_ACTIVATIONS", weakref.WeakSet())
    monkeypatch.setattr(coordinator, "_FAILED_ACTIVATIONS", weakref.WeakSet())
    assert not hasattr(coordinator, "_TRUSTED_VERIFY_ACTIVATION")
    monkeypatch.setattr(
        coordinator,
        "_TRUSTED_VERIFY_ACTIVATION",
        lambda _activation, _state: (coordinator._PHASE_ISSUED, False, True),
        raising=False,
    )

    with pytest.raises(coordinator.Experiment002CoordinatorError, match="corrupted"):
        coordinator.verify_verified_child_activation(synthetic_registration, activation)

    coordinator._ACTIVATIONS[activation] = original
    coordinator._ACTIVATION_GUARDS[activation] = coordinator._guard_from_state(original)
    coordinator._ACTIVATION_ANCHORS[activation] = coordinator._anchor_from_state(
        original
    )
    coordinator._ACTIVATION_PHASES[activation] = coordinator._PHASE_ISSUED
    coordinator._FAILED_ACTIVATIONS.discard(activation)
    with pytest.raises(coordinator.Experiment002CoordinatorError, match="corrupted"):
        coordinator.verify_verified_child_activation(synthetic_registration, activation)


def test_object_new_clone_cannot_copy_activation_authority(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activation = _activation(synthetic_registration)
    clone = object.__new__(coordinator.VerifiedChildActivation)
    coordinator._ACTIVATIONS[clone] = coordinator._ACTIVATIONS[activation]
    coordinator._ACTIVATION_GUARDS[clone] = coordinator._ACTIVATION_GUARDS[activation]
    coordinator._ACTIVATION_ANCHORS[clone] = coordinator._ACTIVATION_ANCHORS[activation]
    coordinator._ACTIVATION_PHASES[clone] = coordinator._ACTIVATION_PHASES[activation]
    imports: list[str] = []
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: imports.append(name),
    )

    with pytest.raises(coordinator.Experiment002CoordinatorError, match="corrupted"):
        coordinator.verify_verified_child_activation(synthetic_registration, clone)
    with pytest.raises(coordinator.Experiment002CoordinatorError, match="corrupted"):
        coordinator._dispatch_registered_seed_process(synthetic_registration, clone)
    assert imports == []
    coordinator.verify_verified_child_activation(synthetic_registration, activation)


def test_guard_removal_and_restore_cannot_recover_activation(
    synthetic_registration: coordinator.VerifiedRunRegistration,
) -> None:
    activation = _activation(synthetic_registration)
    guard = coordinator._ACTIVATION_GUARDS.pop(activation)
    with pytest.raises(coordinator.Experiment002CoordinatorError, match="corrupted"):
        coordinator.verify_verified_child_activation(synthetic_registration, activation)
    coordinator._ACTIVATION_GUARDS[activation] = guard
    with pytest.raises(coordinator.Experiment002CoordinatorError, match="corrupted"):
        coordinator.verify_verified_child_activation(synthetic_registration, activation)


def test_coherent_dispatched_phase_rollback_is_terminal(
    synthetic_registration: coordinator.VerifiedRunRegistration,
) -> None:
    activation = _activation(synthetic_registration)
    issued = coordinator._ACTIVATIONS[activation]
    coordinator._transition_activation(
        activation,
        expected_phase=coordinator._PHASE_ISSUED,
        next_phase=coordinator._PHASE_DISPATCHED,
    )
    assert activation in coordinator._DISPATCHED_ACTIVATIONS
    coordinator._ACTIVATIONS[activation] = issued
    coordinator._ACTIVATION_GUARDS[activation] = coordinator._guard_from_state(issued)
    coordinator._ACTIVATION_PHASES[activation] = coordinator._PHASE_ISSUED
    with pytest.raises(coordinator.Experiment002CoordinatorError, match="corrupted"):
        coordinator.verify_verified_child_activation(synthetic_registration, activation)
    assert activation in coordinator._DISPATCHED_ACTIVATIONS
    assert activation in coordinator._FAILED_ACTIVATIONS


def test_completed_activation_cannot_be_coherently_reset_to_issued(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activation = _activation(synthetic_registration)
    issued = coordinator._ACTIVATIONS[activation]
    worker = SimpleNamespace(run_registered_seed_process=lambda *_args: None)
    monkeypatch.setattr(importlib, "import_module", lambda _name: worker)
    coordinator._dispatch_registered_seed_process(synthetic_registration, activation)
    assert activation in coordinator._DISPATCHED_ACTIVATIONS
    assert activation in coordinator._COMPLETED_ACTIVATIONS
    coordinator._ACTIVATIONS[activation] = issued
    coordinator._ACTIVATION_GUARDS[activation] = coordinator._guard_from_state(issued)
    coordinator._ACTIVATION_PHASES[activation] = coordinator._PHASE_ISSUED
    with pytest.raises(coordinator.Experiment002CoordinatorError, match="corrupted"):
        coordinator.verify_verified_child_activation(synthetic_registration, activation)
    assert activation in coordinator._DISPATCHED_ACTIVATIONS
    assert activation in coordinator._COMPLETED_ACTIVATIONS
    assert activation in coordinator._FAILED_ACTIVATIONS


def test_full_lifecycle_store_replacement_cannot_roll_completed_back_to_issued(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activation = _activation(synthetic_registration)
    issued = coordinator._ACTIVATIONS[activation]
    worker = SimpleNamespace(run_registered_seed_process=lambda *_args: None)
    monkeypatch.setattr(importlib, "import_module", lambda _name: worker)
    coordinator._dispatch_registered_seed_process(synthetic_registration, activation)

    monkeypatch.setattr(
        coordinator,
        "_ACTIVATIONS",
        weakref.WeakKeyDictionary({activation: issued}),
    )
    monkeypatch.setattr(
        coordinator,
        "_ACTIVATION_GUARDS",
        weakref.WeakKeyDictionary({activation: coordinator._guard_from_state(issued)}),
    )
    monkeypatch.setattr(
        coordinator,
        "_ACTIVATION_ANCHORS",
        weakref.WeakKeyDictionary({activation: coordinator._anchor_from_state(issued)}),
    )
    monkeypatch.setattr(
        coordinator,
        "_ACTIVATION_PHASES",
        weakref.WeakKeyDictionary({activation: coordinator._PHASE_ISSUED}),
    )
    monkeypatch.setattr(coordinator, "_DISPATCHED_ACTIVATIONS", weakref.WeakSet())
    monkeypatch.setattr(coordinator, "_COMPLETED_ACTIVATIONS", weakref.WeakSet())
    monkeypatch.setattr(coordinator, "_FAILED_ACTIVATIONS", weakref.WeakSet())
    assert not hasattr(coordinator, "_TRUSTED_ADVANCE_ACTIVATION")
    assert not hasattr(coordinator, "_TRUSTED_POISON_ACTIVATION")
    monkeypatch.setattr(
        coordinator,
        "_TRUSTED_ADVANCE_ACTIVATION",
        lambda *_args: None,
        raising=False,
    )
    monkeypatch.setattr(
        coordinator,
        "_TRUSTED_POISON_ACTIVATION",
        lambda _activation: None,
        raising=False,
    )

    with pytest.raises(coordinator.Experiment002CoordinatorError, match="corrupted"):
        coordinator.verify_verified_child_activation(synthetic_registration, activation)
    coordinator._FAILED_ACTIVATIONS.discard(activation)
    with pytest.raises(coordinator.Experiment002CoordinatorError, match="corrupted"):
        coordinator.verify_verified_child_activation(synthetic_registration, activation)


def test_activation_is_rejected_after_fork(
    synthetic_registration: coordinator.VerifiedRunRegistration,
) -> None:
    activation = _activation(synthetic_registration)
    pid = _fork()
    if pid == 0:
        try:
            coordinator.verify_verified_child_activation(
                synthetic_registration,
                activation,
            )
        except coordinator.Experiment002CoordinatorError:
            os._exit(0)
        os._exit(93)
    _wait_success(pid)
    coordinator.verify_verified_child_activation(synthetic_registration, activation)


def test_real_run_registration_authority_rejects_fork_inheritance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(authority, "_ISSUED", weakref.WeakKeyDictionary())
    monkeypatch.setattr(authority, "_ISSUED_GUARDS", weakref.WeakKeyDictionary())
    monkeypatch.setattr(authority, "_FAILED", weakref.WeakSet())
    monkeypatch.setattr(authority, "_ISSUANCE_COMPLETE", False)
    snapshot = authority._RepositorySnapshot(
        repository_root=Path("/home/ubuntu/gitcode/falsewake"),
        head_commit="1" * 40,
        implementation_commit="2" * 40,
        registration_sha256="3" * 64,
        source_bundle_sha256="4" * 64,
        source_paths=("src/falsewake/experiment_002_run_authority.py",),
    )
    registration = authority._issue_controlled_snapshot_for_tests(snapshot)
    authority._verified_state(registration)
    pid = _fork()
    if pid == 0:
        try:
            authority._verified_state(registration)
        except authority.Experiment002RunAuthorityError:
            os._exit(0)
        os._exit(94)
    _wait_success(pid)
    authority._verified_state(registration)


def test_dispatch_dynamically_imports_worker_after_claim_and_is_one_shot(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activation = _activation(synthetic_registration)
    events: list[tuple[str, object]] = []

    def worker(
        registration: coordinator.VerifiedRunRegistration,
        supplied: coordinator.VerifiedChildActivation,
        /,
    ) -> None:
        coordinator.verify_verified_child_activation(registration, supplied)
        events.append((supplied.role, supplied.seed))

    def dynamic_import(name: str) -> object:
        events.append(("import", name))
        return SimpleNamespace(run_registered_seed_process=worker)

    monkeypatch.setattr(importlib, "import_module", dynamic_import)
    coordinator._dispatch_registered_seed_process(
        synthetic_registration,
        activation,
    )
    assert events == [
        ("import", "falsewake.experiment_002_seed_worker"),
        (coordinator._SEED_ROLE, coordinator.REGISTERED_SEEDS[0]),
    ]
    with pytest.raises(coordinator.Experiment002CoordinatorError, match="corrupted"):
        coordinator.verify_verified_child_activation(
            synthetic_registration,
            activation,
        )
    with pytest.raises(coordinator.Experiment002CoordinatorError):
        coordinator._dispatch_registered_seed_process(
            synthetic_registration,
            activation,
        )
    assert events == [
        ("import", "falsewake.experiment_002_seed_worker"),
        (coordinator._SEED_ROLE, coordinator.REGISTERED_SEEDS[0]),
    ]


@pytest.mark.parametrize("failure", ["import", "worker", "return"])
def test_dispatch_failure_irreversibly_poisons_activation(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    activation = _activation(synthetic_registration)

    def worker(*args: object) -> object:
        del args
        if failure == "worker":
            raise RuntimeError("synthetic worker failure")
        if failure == "return":
            return object()
        return None

    def dynamic_import(name: str) -> object:
        assert name == coordinator._WORKER_MODULE
        if failure == "import":
            raise ModuleNotFoundError(name)
        return SimpleNamespace(run_registered_seed_process=worker)

    monkeypatch.setattr(importlib, "import_module", dynamic_import)
    with pytest.raises(
        (ModuleNotFoundError, RuntimeError, coordinator.Experiment002CoordinatorError)
    ):
        coordinator._dispatch_registered_seed_process(
            synthetic_registration,
            activation,
        )
    assert activation in coordinator._DISPATCHED_ACTIVATIONS
    assert activation in coordinator._FAILED_ACTIVATIONS
    with pytest.raises(coordinator.Experiment002CoordinatorError, match="corrupted"):
        coordinator.verify_verified_child_activation(
            synthetic_registration,
            activation,
        )


def test_guarded_child_route_is_the_only_exact_legacy_entrypoint() -> None:
    assert not hasattr(coordinator, "_receive_child_activation")
    assert not hasattr(coordinator, "_receive_activation_from_fixed_fd")
    assert (
        coordinator._run_registered_seed_child
        is coordinator._run_guarded_registered_seed_child
    )
    signature = inspect.signature(coordinator._run_guarded_registered_seed_child)
    parameters = list(signature.parameters.values())
    assert len(parameters) == 1
    assert parameters[0].name == "registration"
    assert parameters[0].kind is inspect.Parameter.POSITIONAL_ONLY
    source = inspect.getsource(coordinator._run_guarded_registered_seed_child)
    assert "fileno=_CHILD_CONTROL_FD" in source
    assert "attempted = True" in source
    for forbidden in ("argv", "callback", "command", "environment", "path=", "seed="):
        assert forbidden not in source


def test_guarded_child_route_source_preserves_the_security_order() -> None:
    source = inspect.getsource(coordinator._run_guarded_registered_seed_child)
    ordered_fragments = (
        "peeked = _peek_packet(channel)",
        "process_guard_binding = _claim_and_verify_child_process_guard()",
        "registration_binding = _registration_binding(registration)",
        "ticket = _validate_ticket(",
        "consumed = _consume_packet(channel)",
        "_require_same_received_packet(peeked, consumed)",
        "_require_process_guard_binding(process_guard_binding)",
        "receipt = _reserve_accepted_ticket(ticket)",
        "activation = _issue_activation(",
        "verify_verified_child_activation(registration, activation)",
        "acknowledgement = _encode_packet(_ack_document(ticket))",
        "_send_packet_prepared(channel, acknowledgement)",
        "_dispatch_registered_seed_process(registration, activation)",
    )
    positions = [source.index(fragment) for fragment in ordered_fragments]
    assert positions == sorted(positions)
    assert source.count("_claim_and_verify_child_process_guard()") == 1
    assert source.count("_send_packet_prepared(channel, acknowledgement)") == 1


def test_registered_seed_selection_surface_is_exact_and_factory_is_deleted() -> None:
    signature = inspect.signature(coordinator._run_registered_seed_selection)
    parameters = list(signature.parameters.values())
    assert [parameter.name for parameter in parameters] == [
        "registration",
        "cpu_ids",
        "source_bundle_fd",
    ]
    assert all(
        parameter.kind is inspect.Parameter.POSITIONAL_ONLY
        and parameter.default is inspect.Parameter.empty
        for parameter in parameters
    )
    assert coordinator._run_registered_seed_selection.__defaults__ is None
    assert coordinator._run_registered_seed_selection.__kwdefaults__ is None
    assert signature.return_annotation in {
        coordinator._RegisteredSeedSelection,
        "_RegisteredSeedSelection",
    }
    assert not hasattr(coordinator, "_make_registered_seed_selection")
    assert not hasattr(coordinator, "_bind_registered_seed_selection_integrity")
    implementation = _registered_seed_selection_implementation()
    closure = dict(
        zip(
            implementation.__code__.co_freevars,
            implementation.__closure__ or (),
            strict=True,
        )
    )
    assert (
        closure["run_one"].cell_contents is coordinator._run_one_registered_parent_child
    )
    assert closure["make_assignment"].cell_contents is coordinator._assignment
    verify_registration = cast(
        FunctionType, closure["verify_registration"].cell_contents
    )
    verify_closure = dict(
        zip(
            verify_registration.__code__.co_freevars,
            verify_registration.__closure__ or (),
            strict=True,
        )
    )
    assert (
        verify_closure["registration_verifier"].cell_contents
        is cast(Any, coordinator).verify_verified_run_registration
    )
    assert (
        closure["snapshot_frame"].cell_contents
        is coordinator._verified_child_result_snapshot_frame
    )
    assert (
        closure["result_frame"].cell_contents
        is coordinator._registered_parent_child_result_frame
    )


def test_registered_seed_selection_happy_trace_is_exact_detached_and_borrows_fd(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = _install_synthetic_parent_execution(monkeypatch, synthetic_registration)
    _install_selection_snapshot_behavior(harness, _selection_metrics())
    cpu_ids = (2, 7)
    bundle_path = tmp_path / "selection-borrowed-bundle"
    bundle_path.write_bytes(b"synthetic sealed bytes")
    source_bundle_fd = os.open(bundle_path, os.O_RDONLY | os.O_CLOEXEC)
    os.lseek(source_bundle_fd, 5, os.SEEK_SET)
    before = (
        os.fstat(source_bundle_fd),
        os.lseek(source_bundle_fd, 0, os.SEEK_CUR),
        os.get_inheritable(source_bundle_fd),
    )
    try:
        selection = coordinator._run_registered_seed_selection(
            synthetic_registration,
            cpu_ids,
            source_bundle_fd,
        )
        after = (
            os.fstat(source_bundle_fd),
            os.lseek(source_bundle_fd, 0, os.SEEK_CUR),
            os.get_inheritable(source_bundle_fd),
        )
    finally:
        os.close(source_bundle_fd)

    assert before == after
    assert type(selection) is tuple
    training, selected_ordinal, rerun = selection
    assert type(training) is tuple
    assert len(training) == 3
    assert selected_ordinal == 0
    registration_frame = (
        TEST_BINDING.head_commit,
        TEST_BINDING.implementation_commit,
        TEST_BINDING.registration_sha256,
        TEST_BINDING.source_bundle_sha256,
    )
    assert harness.binding_frames == [
        (
            coordinator._SEED_ROLE,
            0,
            coordinator.REGISTERED_SEEDS[0],
            *registration_frame,
        ),
        (
            coordinator._SEED_ROLE,
            1,
            coordinator.REGISTERED_SEEDS[1],
            *registration_frame,
        ),
        (
            coordinator._SEED_ROLE,
            2,
            coordinator.REGISTERED_SEEDS[2],
            *registration_frame,
        ),
        (
            coordinator._RERUN_ROLE,
            3,
            coordinator.REGISTERED_SEEDS[0],
            *registration_frame,
        ),
    ]
    assert len(harness.calls) == 4
    assert all(
        call[0] == cpu_ids and call[2] == source_bundle_fd for call in harness.calls
    )
    assert [result[5][:3] for result in training] == [
        (coordinator._SEED_ROLE, 0, coordinator.REGISTERED_SEEDS[0]),
        (coordinator._SEED_ROLE, 1, coordinator.REGISTERED_SEEDS[1]),
        (coordinator._SEED_ROLE, 2, coordinator.REGISTERED_SEEDS[2]),
    ]
    assert rerun[5][:3] == (
        coordinator._RERUN_ROLE,
        3,
        training[selected_ordinal][5][2],
    )
    for result in (*training, rerun):
        assert type(result) is tuple
        assert len(result) == 6
        assert type(result[1]) is tuple
        assert type(result[5]) is tuple
        assert len(result[5]) == 21
        assert not any(
            type(value)
            in {
                harness.verified_result_type,
                harness.supervised_result_type,
                harness.binding_type,
            }
            for value in result
        )
    with pytest.raises(
        coordinator.Experiment002CoordinatorError,
        match="already attempted",
    ):
        coordinator._run_registered_seed_selection(
            synthetic_registration,
            cpu_ids,
            source_bundle_fd,
        )
    assert len(harness.calls) == 4


@pytest.mark.parametrize(
    ("metrics", "expected_ordinal"),
    [
        (
            _selection_metrics((7, 3, 4, 0.1), (7, 4, 5, 0.9), (7, 2, 3, 0.01)),
            1,
        ),
        (
            _selection_metrics((7, 3, 4, 0.5), (7, 3, 4, 0.4), (7, 3, 4, 0.2)),
            2,
        ),
        (
            _selection_metrics((6, 3, 4, 0.5), (2, 3, 4, 0.5), (4, 3, 4, 0.5)),
            1,
        ),
        (
            _selection_metrics((7, 3, 4, 0.5), (7, 3, 4, 0.5), (7, 3, 4, 0.5)),
            0,
        ),
    ],
    ids=("f1-cross-product", "cross-entropy", "epoch", "seed"),
)
def test_registered_seed_selection_applies_each_exact_rank_tiebreak(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
    metrics: dict[int, _SelectionMetric],
    expected_ordinal: int,
) -> None:
    harness = _install_synthetic_parent_execution(monkeypatch, synthetic_registration)
    _install_selection_snapshot_behavior(harness, metrics)
    training, selected_ordinal, rerun = coordinator._run_registered_seed_selection(
        synthetic_registration,
        (2, 7),
        91,
    )
    assert selected_ordinal == expected_ordinal
    assert rerun[5][2] == training[selected_ordinal][5][2]


@pytest.mark.parametrize(
    "ordered_metrics",
    tuple(
        itertools.permutations(
            (
                (7, 4, 5, 0.9),
                (2, 3, 4, 0.1),
                (1, 2, 3, 0.01),
            )
        )
    ),
)
def test_registered_seed_selection_ranking_is_metric_order_independent(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
    ordered_metrics: tuple[_SelectionMetric, _SelectionMetric, _SelectionMetric],
) -> None:
    harness = _install_synthetic_parent_execution(monkeypatch, synthetic_registration)
    metrics = _selection_metrics(*ordered_metrics)
    _install_selection_snapshot_behavior(harness, metrics)
    _training, selected_ordinal, rerun = coordinator._run_registered_seed_selection(
        synthetic_registration,
        (2, 7),
        91,
    )
    expected_seed = coordinator.REGISTERED_SEEDS[ordered_metrics.index((7, 4, 5, 0.9))]
    assert selected_ordinal == coordinator.REGISTERED_SEEDS.index(expected_seed)
    assert rerun[5][2] == expected_seed


def test_selected_seed_rerun_allows_envelope_and_resource_differences(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_synthetic_parent_execution(monkeypatch, synthetic_registration)

    def mutate_envelope(fields: list[object]) -> None:
        envelope_bytes = b'{"rerun-envelope":"different"}'
        fields[13:16] = (
            envelope_bytes,
            len(envelope_bytes),
            hashlib.sha256(
                b"falsewake-exp002-child-result-envelope-v1\0" + envelope_bytes
            ).hexdigest(),
        )

    _install_selection_snapshot_behavior(
        harness,
        _selection_metrics(),
        rerun_mutation=mutate_envelope,
    )
    base_behavior = harness.behavior[0]

    def resource_behavior(
        cpu_ids: tuple[int, int],
        activation: _SyntheticActivation,
        source_bundle_fd: int,
        binding: object,
    ) -> object:
        result = base_behavior(cpu_ids, activation, source_bundle_fd, binding)
        if cast(Any, binding).role == coordinator._RERUN_ROLE:
            return replace(
                cast(Any, result),
                elapsed_nanoseconds=101,
                maximum_rss_bytes=102,
                output_and_scratch_bytes=103,
            )
        return result

    harness.behavior[0] = resource_behavior
    training, selected_ordinal, rerun = coordinator._run_registered_seed_selection(
        synthetic_registration,
        (2, 7),
        91,
    )
    selected = training[selected_ordinal]
    assert rerun[2:5] == (101, 102, 103)
    assert rerun[2:5] != selected[2:5]
    assert rerun[5][13:16] != selected[5][13:16]
    assert all(
        rerun[5][index] == selected[5][index]
        for index in (7, 8, 9, 10, 11, 12, 16, 17, 18, 19, 20)
    )


@pytest.mark.parametrize(
    "index",
    (7, 8, 9, 10, 11, 12, 16, 17, 18, 19, 20),
)
def test_selected_seed_rerun_rejects_each_required_field_mismatch(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
    index: int,
) -> None:
    harness = _install_synthetic_parent_execution(monkeypatch, synthetic_registration)
    _install_selection_snapshot_behavior(
        harness,
        _selection_metrics(),
        rerun_mutation=lambda fields: _mutate_required_rerun_snapshot_field(
            fields,
            index,
        ),
    )
    with pytest.raises(
        coordinator.Experiment002CoordinatorError,
        match="selected-seed rerun differs",
    ):
        coordinator._run_registered_seed_selection(
            synthetic_registration,
            (2, 7),
            91,
        )
    assert len(harness.calls) == 4


@pytest.mark.parametrize(
    "invalid",
    ("result-type", "snapshot-shape", "snapshot-type", "snapshot-binding"),
)
def test_registered_seed_selection_rejects_invalid_child_return_contracts(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
    invalid: str,
) -> None:
    harness = _install_synthetic_parent_execution(monkeypatch, synthetic_registration)
    if invalid == "result-type":
        base_behavior = harness.behavior[0]

        def invalid_result(
            cpu_ids: tuple[int, int],
            activation: _SyntheticActivation,
            source_bundle_fd: int,
            binding: object,
        ) -> object:
            base_behavior(cpu_ids, activation, source_bundle_fd, binding)
            return object()

        harness.behavior[0] = invalid_result
    else:
        base_snapshot = harness.snapshot_behavior[0]

        def invalid_snapshot(result: object, /) -> object:
            fields = list(cast(tuple[object, ...], base_snapshot(result)))
            if invalid == "snapshot-shape":
                return tuple(fields[:-1])
            if invalid == "snapshot-type":
                fields[16] = True
            else:
                fields[2] = coordinator.REGISTERED_SEEDS[1]
            return tuple(fields)

        harness.snapshot_behavior[0] = invalid_snapshot

    with pytest.raises(coordinator.Experiment002CoordinatorError):
        coordinator._run_registered_seed_selection(
            synthetic_registration,
            (2, 7),
            91,
        )
    assert len(harness.calls) == 1
    with pytest.raises(
        coordinator.Experiment002CoordinatorError,
        match="already attempted",
    ):
        coordinator._run_registered_seed_selection(
            synthetic_registration,
            (2, 7),
            91,
        )
    assert len(harness.calls) == 1


@pytest.mark.parametrize("failure_call", (0, 1, 2, 3))
def test_registered_seed_selection_failure_stops_later_calls_burns_and_borrows_fd(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_call: int,
) -> None:
    harness = _install_synthetic_parent_execution(monkeypatch, synthetic_registration)
    _install_selection_snapshot_behavior(harness, _selection_metrics())
    base_behavior = harness.behavior[0]

    def fail_selected_call(
        cpu_ids: tuple[int, int],
        activation: _SyntheticActivation,
        source_bundle_fd: int,
        binding: object,
    ) -> object:
        if len(harness.calls) - 1 == failure_call:
            raise RuntimeError(f"synthetic selection failure {failure_call}")
        return base_behavior(cpu_ids, activation, source_bundle_fd, binding)

    harness.behavior[0] = fail_selected_call
    bundle_path = tmp_path / f"selection-failure-{failure_call}"
    bundle_path.write_bytes(b"synthetic sealed bytes")
    source_bundle_fd = os.open(bundle_path, os.O_RDONLY | os.O_CLOEXEC)
    os.lseek(source_bundle_fd, 5, os.SEEK_SET)
    before = (
        os.fstat(source_bundle_fd),
        os.lseek(source_bundle_fd, 0, os.SEEK_CUR),
        os.get_inheritable(source_bundle_fd),
    )
    try:
        with pytest.raises(RuntimeError, match="synthetic selection failure"):
            coordinator._run_registered_seed_selection(
                synthetic_registration,
                (2, 7),
                source_bundle_fd,
            )
        assert len(harness.calls) == failure_call + 1
        with pytest.raises(
            coordinator.Experiment002CoordinatorError,
            match="already attempted",
        ):
            coordinator._run_registered_seed_selection(
                synthetic_registration,
                (2, 7),
                source_bundle_fd,
            )
        after = (
            os.fstat(source_bundle_fd),
            os.lseek(source_bundle_fd, 0, os.SEEK_CUR),
            os.get_inheritable(source_bundle_fd),
        )
    finally:
        os.close(source_bundle_fd)
    assert before == after
    assert len(harness.calls) == failure_call + 1


def test_registered_seed_selection_burns_before_argument_validation(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_synthetic_parent_execution(monkeypatch, synthetic_registration)
    with pytest.raises(
        coordinator.Experiment002CoordinatorError,
        match="CPU IDs",
    ):
        coordinator._run_registered_seed_selection(
            synthetic_registration,
            cast(Any, (True, 7)),
            91,
        )
    with pytest.raises(
        coordinator.Experiment002CoordinatorError,
        match="already attempted",
    ):
        coordinator._run_registered_seed_selection(
            synthetic_registration,
            (2, 7),
            91,
        )
    assert harness.calls == []


def test_registered_seed_selection_is_global_single_flight_and_fresh_can_follow(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    concurrent_registration = _issue_additional_synthetic_registration()
    harness = _install_synthetic_parent_execution(monkeypatch, synthetic_registration)
    _install_selection_snapshot_behavior(harness, _selection_metrics())
    base_behavior = harness.behavior[0]
    entered = threading.Event()
    release = threading.Event()
    primary_results: list[coordinator._RegisteredSeedSelection] = []
    primary_errors: list[BaseException] = []
    blocked_once = False

    def blocking_behavior(
        cpu_ids: tuple[int, int],
        activation: _SyntheticActivation,
        source_bundle_fd: int,
        binding: object,
    ) -> object:
        nonlocal blocked_once
        if not blocked_once:
            blocked_once = True
            entered.set()
            if not release.wait(5.0):
                raise AssertionError("selection concurrency release timed out")
        return base_behavior(cpu_ids, activation, source_bundle_fd, binding)

    harness.behavior[0] = blocking_behavior

    def run_primary() -> None:
        try:
            primary_results.append(
                coordinator._run_registered_seed_selection(
                    synthetic_registration,
                    (2, 7),
                    91,
                )
            )
        except BaseException as error:
            primary_errors.append(error)

    thread = threading.Thread(target=run_primary)
    thread.start()
    assert entered.wait(5.0)
    try:
        with pytest.raises(
            coordinator.Experiment002CoordinatorError,
            match="already attempted",
        ):
            coordinator._run_registered_seed_selection(
                synthetic_registration,
                (2, 7),
                91,
            )
        with pytest.raises(
            coordinator.Experiment002CoordinatorError,
            match="already in flight",
        ):
            coordinator._run_registered_seed_selection(
                concurrent_registration,
                (2, 7),
                91,
            )
    finally:
        release.set()
        thread.join(10.0)
    assert not thread.is_alive()
    assert primary_errors == []
    assert len(primary_results) == 1
    assert len(harness.calls) == 4
    with pytest.raises(
        coordinator.Experiment002CoordinatorError,
        match="already attempted",
    ):
        coordinator._run_registered_seed_selection(
            concurrent_registration,
            (2, 7),
            91,
        )

    fresh_registration = _issue_additional_synthetic_registration()
    fresh_harness = _install_synthetic_parent_execution(
        monkeypatch,
        fresh_registration,
    )
    _install_selection_snapshot_behavior(fresh_harness, _selection_metrics())
    fresh_selection = coordinator._run_registered_seed_selection(
        fresh_registration,
        (2, 7),
        91,
    )
    assert type(fresh_selection) is tuple
    assert len(fresh_harness.calls) == 4


@pytest.mark.parametrize(
    "route_name",
    (
        "_run_one_registered_parent_child",
        "_assignment",
        "verify_verified_run_registration",
        "_verified_child_result_snapshot_frame",
        "_registered_parent_child_result_frame",
    ),
)
def test_registered_seed_selection_rejects_pre_call_global_route_replacement(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
    route_name: str,
) -> None:
    harness = _install_synthetic_parent_execution(monkeypatch, synthetic_registration)
    original = getattr(coordinator, route_name)
    replacement_calls: list[tuple[object, ...]] = []

    def replacement(*args: object, **kwargs: object) -> object:
        replacement_calls.append((*args, kwargs))
        return object()

    monkeypatch.setattr(coordinator, route_name, replacement)
    with pytest.raises(
        coordinator.Experiment002CoordinatorError,
        match="routes changed",
    ):
        coordinator._run_registered_seed_selection(
            synthetic_registration,
            (2, 7),
            91,
        )
    assert replacement_calls == []
    assert harness.calls == []
    monkeypatch.setattr(coordinator, route_name, original)
    with pytest.raises(
        coordinator.Experiment002CoordinatorError,
        match="already attempted",
    ):
        coordinator._run_registered_seed_selection(
            synthetic_registration,
            (2, 7),
            91,
        )
    assert harness.calls == []


@pytest.mark.parametrize("tamper", ("code", "defaults", "closure"))
def test_registered_seed_selection_rejects_mid_call_dynamic_authority_mutation(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    harness = _install_synthetic_parent_execution(monkeypatch, synthetic_registration)
    _install_selection_snapshot_behavior(harness, _selection_metrics())
    implementation = _registered_seed_selection_implementation()
    closure = dict(
        zip(
            implementation.__code__.co_freevars,
            implementation.__closure__ or (),
            strict=True,
        )
    )
    better_result = cast(FunctionType, closure["better_result"].cell_contents)
    verify_registration = cast(
        FunctionType,
        closure["verify_registration"].cell_contents,
    )
    rerun_indices_cell = closure["rerun_match_indices"]
    better_code = better_result.__code__
    verifier_defaults = verify_registration.__defaults__
    rerun_indices = rerun_indices_cell.cell_contents
    base_behavior = harness.behavior[0]
    mutated = False

    def tampering_behavior(
        cpu_ids: tuple[int, int],
        activation: _SyntheticActivation,
        source_bundle_fd: int,
        binding: object,
    ) -> object:
        nonlocal mutated
        result = base_behavior(cpu_ids, activation, source_bundle_fd, binding)
        if not mutated:
            mutated = True
            if tamper == "code":
                better_result.__code__ = better_code.replace(co_name="forged_rank")
            elif tamper == "defaults":
                verify_registration.__defaults__ = (None,)
            else:
                rerun_indices_cell.cell_contents = ()
        return result

    harness.behavior[0] = tampering_behavior
    try:
        with pytest.raises(
            coordinator.Experiment002CoordinatorError,
            match="authority changed|routes changed|closure content changed",
        ):
            coordinator._run_registered_seed_selection(
                synthetic_registration,
                (2, 7),
                91,
            )
    finally:
        better_result.__code__ = better_code
        verify_registration.__defaults__ = verifier_defaults
        rerun_indices_cell.cell_contents = rerun_indices
    assert mutated
    assert 1 <= len(harness.calls) <= 4


def test_registered_seed_selection_burns_before_pre_call_integrity_failure(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_synthetic_parent_execution(monkeypatch, synthetic_registration)
    implementation = _registered_seed_selection_implementation()
    closure = dict(
        zip(
            implementation.__code__.co_freevars,
            implementation.__closure__ or (),
            strict=True,
        )
    )
    better_result = cast(FunctionType, closure["better_result"].cell_contents)
    original_code = better_result.__code__
    better_result.__code__ = original_code.replace(co_name="forged_pre_call_rank")
    try:
        with pytest.raises(
            coordinator.Experiment002CoordinatorError,
            match="authority changed",
        ):
            coordinator._run_registered_seed_selection(
                synthetic_registration,
                (2, 7),
                91,
            )
    finally:
        better_result.__code__ = original_code
    with pytest.raises(
        coordinator.Experiment002CoordinatorError,
        match="already attempted",
    ):
        coordinator._run_registered_seed_selection(
            synthetic_registration,
            (2, 7),
            91,
        )
    assert harness.calls == []


def test_registered_seed_selection_rejects_inherited_registration_before_child(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_synthetic_parent_execution(monkeypatch, synthetic_registration)
    _install_selection_snapshot_behavior(harness, _selection_metrics())
    pid = _fork()
    if pid == 0:
        try:
            coordinator._run_registered_seed_selection(
                synthetic_registration,
                (2, 7),
                91,
            )
        except authority.Experiment002RunAuthorityError:
            os._exit(0)
        except BaseException:
            os._exit(92)
        os._exit(93)
    _wait_success(pid)
    assert harness.calls == []
    selection = coordinator._run_registered_seed_selection(
        synthetic_registration,
        (2, 7),
        91,
    )
    assert type(selection) is tuple
    assert len(harness.calls) == 4


def test_registered_seed_selection_source_has_no_extra_resource_or_import_routes() -> (
    None
):
    source = inspect.getsource(_registered_seed_selection_implementation())
    assert "run_one(" in source
    assert "selected_seed = training[selected_ordinal][5][2]" in source
    assert "rerun_match_indices" in source
    for forbidden in (
        "import_module",
        "os.close",
        "os.dup",
        "os.lseek",
        "cleanup",
        "retry",
    ):
        assert forbidden not in source


def test_public_parent_terminal_ast_has_no_blocked_stub_or_override_surface() -> None:
    source = textwrap.dedent(inspect.getsource(coordinator.run_registered_experiment))
    function = cast(ast.FunctionDef, ast.parse(source).body[0])
    assert [argument.arg for argument in function.args.posonlyargs] == ["registration"]
    assert function.args.args == []
    assert function.args.kwonlyargs == []
    assert "supervisor and resource-control layer are not enabled" not in source
    assert "claim_parent_operations(registration)" in source
    assert "own_operation_route(" in source
    assert "typed_operations_descriptors" in source
    assert "_run_registered_seed_selection" not in source


def test_registered_parent_claim_uses_captured_names_and_real_lifecycle_intrinsics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    public_closure = dict(
        zip(
            coordinator.run_registered_experiment.__code__.co_freevars,
            coordinator.run_registered_experiment.__closure__ or (),
            strict=True,
        )
    )
    claim = cast(FunctionType, public_closure["claim_parent_operations"].cell_contents)
    claim_closure = dict(
        zip(
            claim.__code__.co_freevars,
            claim.__closure__ or (),
            strict=True,
        )
    )
    source = inspect.getsource(claim)
    assert "dynamic_import(run_authority_module_name)" in source
    assert "dynamic_import(supervisor_module_name)" in source
    assert "dynamic_import(evidence_module_name)" in source
    assert "dynamic_import(publication_module_name)" in source
    for mutable_name in (
        "_RUN_AUTHORITY_MODULE",
        "_SUPERVISOR_MODULE",
        "_FINAL_EVIDENCE_MODULE",
        "_FINAL_PUBLICATION_MODULE",
        "_LOWER_HEX",
    ):
        assert mutable_name not in source

    require_authority = cast(
        FunctionType,
        claim_closure["require_coordinator_authority"].cell_contents,
    )
    canonical_supervisor = cast(
        str,
        claim_closure["supervisor_module_name"].cell_contents,
    )
    monkeypatch.setattr(coordinator, "_SUPERVISOR_MODULE", "evil.redirect")
    with pytest.raises(
        coordinator.Experiment002CoordinatorError,
        match="coordinator authority changed",
    ):
        require_authority()

    supervisor = importlib.import_module(canonical_supervisor)
    expected = (
        ("_begin_registered_supervisor_child", "begin_registered_child"),
        (
            "_finish_registered_supervisor_child_success",
            "finish_registered_child_success",
        ),
        (
            "_finish_registered_supervisor_child_failure",
            "finish_registered_child_failure",
        ),
    )
    for binding_name, intrinsic_name in expected:
        route = cast(FunctionType, getattr(supervisor, binding_name))
        assert type(route) is FunctionType
        assert route.__name__ == intrinsic_name
        assert route.__module__ == canonical_supervisor
        signature = inspect.signature(route)
        assert list(signature.parameters) == []


def test_registered_parent_claim_composes_real_routes_without_execution() -> None:
    script = textwrap.dedent(
        f"""
        import json
        import sys

        sys.path.insert(0, {os.fspath(SOURCE_ROOT)!r})
        import falsewake.experiment_002_run_authority as authority

        state_type = authority._VerifiedState
        state = object.__new__(state_type)
        values = (
            ('head_commit', '1' * 40),
            ('implementation_commit', '2' * 40),
            ('registration_sha256', '3' * 64),
            ('source_bundle_sha256', '4' * 64),
        )
        for name, value in values:
            state_type.__dict__[name].__set__(state, value)

        def verify_verified_run_registration(value):
            if type(value) is not authority.VerifiedRunRegistration:
                raise TypeError

        def reverify_verified_run_registration(value):
            return verify_verified_run_registration(value)

        def _verified_state(value):
            verify_verified_run_registration(value)
            return state

        for route in (
            verify_verified_run_registration,
            reverify_verified_run_registration,
            _verified_state,
        ):
            route.__module__ = authority.__name__
        authority.verify_verified_run_registration = verify_verified_run_registration
        authority.reverify_verified_run_registration = (
            reverify_verified_run_registration
        )
        authority._verified_state = _verified_state

        import falsewake.experiment_002_coordinator as coordinator

        registration = object.__new__(authority.VerifiedRunRegistration)
        public = coordinator.run_registered_experiment
        public_closure = dict(zip(
            public.__code__.co_freevars,
            public.__closure__,
            strict=True,
        ))
        parent_state = public_closure['state'].cell_contents
        parent_state_type = public_closure['state_type'].cell_contents
        state_descriptors = public_closure['typed_state_descriptors'].cell_contents
        state_descriptors[0].__set__(parent_state, True)
        state_descriptors[1].__set__(parent_state, True)
        assert type(parent_state) is parent_state_type

        claim = public_closure['claim_parent_operations'].cell_contents
        frame, operations = claim(registration)
        assert type(operations) is coordinator._RegisteredExperimentOperations
        assert operations.require_authority() is None
        checker = operations.require_authority
        assert 'publish_final' not in checker.__code__.co_freevars
        assert all(
            node[0] is not operations.publish_final_evidence
            for node in operations.authority_integrity
        )
        state_descriptors[1].__set__(parent_state, False)
        try:
            operations.require_authority()
        except coordinator.Experiment002CoordinatorError:
            pass
        else:
            raise AssertionError('parent attempt state loss was accepted')
        state_descriptors[1].__set__(parent_state, True)
        checker_closure = dict(zip(
            checker.__code__.co_freevars,
            checker.__closure__,
            strict=True,
        ))
        require_run_authority = checker_closure[
            'require_run_authority'
        ].cell_contents
        run_checker_closure = dict(zip(
            require_run_authority.__code__.co_freevars,
            require_run_authority.__closure__,
            strict=True,
        ))
        required_seals_route = run_checker_closure[
            'required_seals_route'
        ].cell_contents
        original_required_seals_code = required_seals_route.__code__

        def executed_if_authority_check_is_late():
            raise AssertionError('mutated seals route executed')

        required_seals_route.__code__ = (
            executed_if_authority_check_is_late.__code__
        )
        try:
            operations.require_authority()
        except coordinator.Experiment002CoordinatorError:
            pass
        else:
            raise AssertionError('mutated seals route was accepted')
        finally:
            required_seals_route.__code__ = original_required_seals_code
        publication = sys.modules[
            'falsewake.experiment_002_final_publication'
        ]
        publication_namespace = publication.__dict__
        saved_publication_route = publication_namespace.pop(
            'build_execution_failure_evidence'
        )
        fallback_lookups = []

        def module_fallback(name):
            fallback_lookups.append(name)
            return saved_publication_route

        publication_namespace['__getattr__'] = module_fallback
        try:
            try:
                operations.require_authority()
            except coordinator.Experiment002CoordinatorError:
                pass
            else:
                raise AssertionError(
                    'module fallback replaced a missing authority binding'
                )
            assert fallback_lookups == []
        finally:
            publication_namespace[
                'build_execution_failure_evidence'
            ] = saved_publication_route
            publication_namespace.pop('__getattr__')
        colliding_binding_name = 'build_execution_failure_evidence'
        saved_colliding_route = publication_namespace.pop(
            colliding_binding_name
        )
        key_comparisons = []

        class CollidingModuleKey:
            def __hash__(self):
                return hash(colliding_binding_name)

            def __eq__(self, other):
                key_comparisons.append(other)
                return True

        colliding_key = CollidingModuleKey()
        publication_namespace[colliding_key] = saved_colliding_route
        key_comparisons.clear()
        try:
            try:
                operations.require_authority()
            except coordinator.Experiment002CoordinatorError:
                pass
            else:
                raise AssertionError(
                    'hostile module key retained parent authority'
                )
            assert key_comparisons == []
        finally:
            publication_namespace.pop(colliding_key)
            publication_namespace[
                colliding_binding_name
            ] = saved_colliding_route
        original_publication_type = type(publication)
        module_dictionary_reads = []

        class MutatedPublicationModule(original_publication_type):
            def __getattribute__(self, name):
                if name == '__dict__':
                    module_dictionary_reads.append(name)
                return super().__getattribute__(name)

        publication.__class__ = MutatedPublicationModule
        try:
            try:
                operations.require_authority()
            except coordinator.Experiment002CoordinatorError:
                pass
            else:
                raise AssertionError(
                    'mutated module type retained parent authority'
                )
            assert module_dictionary_reads == []
        finally:
            publication.__class__ = original_publication_type
        publisher_support_integrity = checker_closure[
            'publisher_support_integrity'
        ].cell_contents
        assert all(
            node[0] is not operations.publish_final_evidence
            and 'attempted' not in node[1].co_freevars
            and 'in_flight' not in node[1].co_freevars
            for node in publisher_support_integrity
        )
        publisher = operations.publish_final_evidence
        publisher_closure = dict(zip(
            publisher.__code__.co_freevars,
            publisher.__closure__,
            strict=True,
        ))
        publisher_helper = publisher_closure['require_routes'].cell_contents
        publisher_helper.__code__ = publisher_helper.__code__.replace(
            co_name='tampered_publisher_helper'
        )
        try:
            operations.require_authority()
        except coordinator.Experiment002CoordinatorError:
            pass
        else:
            raise AssertionError('publisher helper mutation was accepted')
        print(json.dumps({{
            'frame': frame,
            'operation_fields': len(
                coordinator._REGISTERED_EXPERIMENT_OPERATION_FIELD_NAMES
            ),
            'checker': checker.__name__,
        }}, sort_keys=True))
        """
    )
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        (sys.executable, "-I", "-S", "-B", "-c", script),
        env=environment,
        capture_output=True,
        check=False,
        text=True,
        timeout=20.0,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result == {
        "checker": "require_parent_authority",
        "frame": ["1" * 40, "2" * 40, "3" * 64, "4" * 64],
        "operation_fields": 17,
    }


def test_registered_parent_factory_rejects_module_subclass_without_protocol() -> None:
    script = textwrap.dedent(
        f"""
        import importlib
        import json
        import sys

        sys.path.insert(0, {os.fspath(SOURCE_ROOT)!r})
        original_module_type = type(importlib)
        dictionary_reads = []

        class MutatedImportlibModule(original_module_type):
            def __getattribute__(self, name):
                if name == '__dict__':
                    dictionary_reads.append(name)
                return super().__getattribute__(name)

        importlib.__class__ = MutatedImportlibModule
        try:
            try:
                import falsewake.experiment_002_coordinator
            except RuntimeError as error:
                assert str(error) == (
                    'registered experiment module identities are invalid'
                )
            else:
                raise AssertionError(
                    'preloaded module subclass reached the parent factory'
                )
            assert dictionary_reads == []
        finally:
            importlib.__class__ = original_module_type
        print(json.dumps({{'dictionary_reads': dictionary_reads}}))
        """
    )
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        (sys.executable, "-I", "-S", "-B", "-c", script),
        env=environment,
        capture_output=True,
        check=False,
        text=True,
        timeout=20.0,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"dictionary_reads": []}


@pytest.mark.parametrize(
    "mutation_target",
    [
        "public",
        "recursive_capture",
        "required_seals_peer",
        "registration_verifier_peer",
    ],
)
def test_registered_parent_admission_rejects_in_place_factory_code_mutation(
    mutation_target: str,
) -> None:
    script = textwrap.dedent(
        f"""
        import sys

        sys.path.insert(0, {os.fspath(SOURCE_ROOT)!r})
        import falsewake.experiment_002_run_authority as authority

        state_type = authority._VerifiedState
        state = object.__new__(state_type)
        for name, value in (
            ('head_commit', '1' * 40),
            ('implementation_commit', '2' * 40),
            ('registration_sha256', '3' * 64),
            ('source_bundle_sha256', '4' * 64),
        ):
            state_type.__dict__[name].__set__(state, value)

        verification_calls = 0

        def executed_if_registration_check_is_late(value):
            raise AssertionError('mutated verified-state route executed')

        def verify_verified_run_registration(value):
            global verification_calls
            verification_calls += 1
            if type(value) is not authority.VerifiedRunRegistration:
                raise TypeError
            if (
                {mutation_target!r} == 'registration_verifier_peer'
                and verification_calls == 2
            ):
                authority._verified_state.__code__ = (
                    executed_if_registration_check_is_late.__code__
                )
            coordinator = sys.modules.get(
                'falsewake.experiment_002_coordinator'
            )
            if (
                coordinator is not None
                and {mutation_target!r} != 'required_seals_peer'
            ):
                if {mutation_target!r} == 'public':
                    target = coordinator.run_registered_experiment
                else:
                    target = coordinator._capture_recursive_function_integrity
                target.__code__ = target.__code__.replace(
                    co_name='tampered_in_place'
                )

        def reverify_verified_run_registration(value):
            return verify_verified_run_registration(value)

        def _verified_state(value):
            return state

        for route in (
            verify_verified_run_registration,
            reverify_verified_run_registration,
            _verified_state,
        ):
            route.__module__ = authority.__name__
        authority.verify_verified_run_registration = verify_verified_run_registration
        authority.reverify_verified_run_registration = (
            reverify_verified_run_registration
        )
        authority._verified_state = _verified_state
        if {mutation_target!r} == 'required_seals_peer':
            original_required_seals = authority._required_child_bundle_seals

            def _required_child_bundle_seals():
                authority._verified_state.__code__ = (
                    authority._verified_state.__code__.replace(
                        co_name='tampered_by_required_seals'
                    )
                )
                return original_required_seals()

            _required_child_bundle_seals.__module__ = authority.__name__
            authority._required_child_bundle_seals = _required_child_bundle_seals

        import falsewake.experiment_002_coordinator as coordinator

        registration = object.__new__(authority.VerifiedRunRegistration)
        public = coordinator.run_registered_experiment
        public_closure = dict(zip(
            public.__code__.co_freevars,
            public.__closure__,
            strict=True,
        ))
        parent_state = public_closure['state'].cell_contents
        state_descriptors = public_closure['typed_state_descriptors'].cell_contents
        state_descriptors[0].__set__(parent_state, True)
        state_descriptors[1].__set__(parent_state, True)
        claim = public_closure['claim_parent_operations'].cell_contents
        try:
            claim(registration)
        except coordinator.Experiment002CoordinatorError:
            raise SystemExit(0)
        raise SystemExit(91)
        """
    )
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        (sys.executable, "-I", "-S", "-B", "-c", script),
        env=environment,
        capture_output=True,
        check=False,
        text=True,
        timeout=20.0,
    )
    assert completed.returncode == 0, completed.stderr


@dataclass(slots=True)
class _RegisteredParentTrace:
    completed_status: str = "pass"
    trace: list[str] | None = None
    failures: dict[str, BaseException] | None = None
    published: list[object] | None = None
    failure_pairs: list[tuple[tuple[str, str, str, str], str, str]] | None = None
    selector_result: object = ((object(), object(), object()), 1, object())
    descriptor: int = 91
    frame_calls: int = 0
    close_calls: int = 0
    quiescence_calls: int = 0
    authority_broken: bool = False
    break_authority_during_create: bool = False
    break_authority_during_selector: bool = False
    restore_authority_during_close: bool = False
    publish_result: object = None
    close_result: object = None
    verify_completed_result: object = None
    cleanup_result: object = None
    quiescence_results: tuple[object, object] = (None, None)
    source_frame_failures: dict[int, BaseException] | None = None
    mutations: dict[str, Callable[[], None]] | None = None
    authority_calls: int = 0
    authority_calls_at_publish: int | None = None
    completed_verified: bool = False
    authority_route: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        if self.trace is None:
            self.trace = []
        if self.failures is None:
            self.failures = {}
        if self.published is None:
            self.published = []
        if self.failure_pairs is None:
            self.failure_pairs = []
        if self.source_frame_failures is None:
            self.source_frame_failures = {}
        if self.mutations is None:
            self.mutations = {}

    def _record(self, name: str) -> None:
        assert self.trace is not None
        self.trace.append(name)
        assert self.failures is not None
        failure = self.failures.get(name)
        if failure is not None:
            raise failure
        assert self.mutations is not None
        mutation = self.mutations.get(name)
        if mutation is not None:
            mutation()

    def require_authority(self) -> None:
        self.authority_calls += 1
        if self.authority_broken:
            raise RuntimeError("synthetic authority loss")

    def prepare(self) -> None:
        self._record("prepare")

    def cpu(self) -> tuple[int, int]:
        self._record("cpu")
        return (2, 7)

    def create(self, _registration: object, /) -> int:
        self._record("create")
        if self.break_authority_during_create:
            self.authority_broken = True
        return self.descriptor

    def source_frame(
        self,
        descriptor: int,
        /,
    ) -> coordinator._SourceBundleDescriptorFrame:
        assert descriptor == self.descriptor
        self.frame_calls += 1
        self._record("source_frame")
        assert self.source_frame_failures is not None
        failure = self.source_frame_failures.get(self.frame_calls)
        if failure is not None:
            raise failure
        return (
            descriptor,
            fcntl.FD_CLOEXEC,
            os.O_RDONLY,
            False,
            (1, 2, stat.S_IFREG | 0o400, 0, os.geteuid(), os.getegid(), 3, 4, 5),
            15,
            "/memfd:falsewake-exp002-child-bundle (deleted)",
            0,
        )

    def select(
        self,
        _registration: object,
        cpu_ids: tuple[int, int],
        descriptor: int,
        /,
    ) -> object:
        assert cpu_ids == (2, 7)
        assert descriptor == self.descriptor
        self._record("select")
        if self.break_authority_during_selector:
            self.authority_broken = True
            raise RuntimeError("synthetic selector boundary loss")
        return self.selector_result

    def close(self, descriptor: int, /) -> object:
        assert descriptor == self.descriptor
        self.close_calls += 1
        self._record("close")
        if self.restore_authority_during_close:
            self.authority_broken = False
        return self.close_result

    def build_completed(self, selector_result: object, /) -> object:
        assert selector_result is self.selector_result
        self._record("build_completed")
        return SimpleNamespace(status=self.completed_status)

    def verify_completed(self, evidence: object, /) -> object:
        assert cast(Any, evidence).status == self.completed_status
        self._record("verify_completed")
        self.completed_verified = True
        return self.verify_completed_result

    def quiescence(self) -> object:
        self.quiescence_calls += 1
        assert self.trace is not None
        name = "quiescence_closed" if "cleanup" in self.trace else "quiescence_active"
        self._record(name)
        return self.quiescence_results[self.quiescence_calls - 1]

    def cleanup(self) -> object:
        self._record("cleanup")
        return self.cleanup_result

    def registration_frame(
        self,
        _registration: object,
        /,
    ) -> tuple[str, str, str, str]:
        self._record("registration_frame")
        return (
            TEST_BINDING.head_commit,
            TEST_BINDING.implementation_commit,
            TEST_BINDING.registration_sha256,
            TEST_BINDING.source_bundle_sha256,
        )

    def build_failure(
        self,
        registration_frame: tuple[str, str, str, str],
        phase: str,
        code: str,
        /,
    ) -> object:
        self._record("build_failure")
        assert self.failure_pairs is not None
        self.failure_pairs.append((registration_frame, phase, code))
        return SimpleNamespace(status="execution_failure", phase=phase, code=code)

    def verify_failure(self, evidence: object, /) -> None:
        assert cast(Any, evidence).status == "execution_failure"
        self._record("verify_failure")

    def publish(self, _registration: object, evidence: object, /) -> object:
        self._record("publish")
        assert self.published is not None
        self.published.append(evidence)
        self.authority_calls_at_publish = self.authority_calls
        return self.publish_result

    def operations(self) -> coordinator._RegisteredExperimentOperations:
        harness = self

        def require_authority() -> None:
            harness.require_authority()

        authority_route = cast(FunctionType, require_authority)
        integrity_verifier = cast(
            FunctionType,
            coordinator._require_recursive_function_integrity_unchanged,
        )
        authority_integrity = coordinator._capture_recursive_function_integrity(
            (
                authority_route,
                integrity_verifier,
                cast(
                    FunctionType,
                    coordinator._call_registered_experiment_operation,
                ),
            )
        )
        self.authority_route = authority_route
        return coordinator._RegisteredExperimentOperations(
            prepare_staging=self.prepare,
            capture_cpu_ids=self.cpu,
            create_source_bundle=self.create,
            run_selection=self.select,
            build_completed_evidence=self.build_completed,
            verify_completed_evidence=self.verify_completed,
            require_quiescence=self.quiescence,
            cleanup_output_roots=self.cleanup,
            build_failure_evidence=self.build_failure,
            verify_failure_evidence=self.verify_failure,
            publish_final_evidence=self.publish,
            authority_integrity_verifier=integrity_verifier,
            authority_integrity=authority_integrity,
            require_authority=authority_route,
            registration_frame=self.registration_frame,
            source_bundle_frame=self.source_frame,
            close_source_bundle=self.close,
        )


def _admitted_registration_frame() -> tuple[str, str, str, str]:
    return (
        TEST_BINDING.head_commit,
        TEST_BINDING.implementation_commit,
        TEST_BINDING.registration_sha256,
        TEST_BINDING.source_bundle_sha256,
    )


@pytest.mark.parametrize("status", ["pass", "gate_failure"])
def test_registered_parent_pass_and_gate_failure_publish_once_and_return_none(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    status: str,
) -> None:
    harness = _RegisteredParentTrace(completed_status=status)
    assert (
        cast(Any, coordinator._run_registered_experiment_once)(
            synthetic_registration,
            _admitted_registration_frame(),
            harness.operations(),
        )
        is None
    )
    assert harness.trace == [
        "prepare",
        "cpu",
        "create",
        "source_frame",
        "source_frame",
        "select",
        "source_frame",
        "close",
        "build_completed",
        "verify_completed",
        "quiescence_active",
        "cleanup",
        "quiescence_closed",
        "registration_frame",
        "publish",
    ]
    assert harness.close_calls == 1
    assert harness.frame_calls == 3
    assert harness.quiescence_calls == 2
    assert harness.published is not None
    assert len(harness.published) == 1
    assert cast(Any, harness.published[0]).status == status
    assert harness.failure_pairs == []
    assert harness.authority_calls_at_publish == harness.authority_calls


@pytest.mark.parametrize(
    ("failed_stage", "expected_pair", "expected_prefix"),
    [
        (
            "prepare",
            ("parent_setup", "staging_prepare_failed"),
            ["prepare"],
        ),
        (
            "cpu",
            ("parent_setup", "cpu_affinity_capture_failed"),
            ["prepare", "cpu"],
        ),
        (
            "create",
            ("parent_setup", "source_bundle_create_failed"),
            ["prepare", "cpu", "create"],
        ),
        (
            "select",
            ("registered_execution", "seed_selection_failed"),
            [
                "prepare",
                "cpu",
                "create",
                "source_frame",
                "source_frame",
                "select",
                "source_frame",
                "close",
            ],
        ),
        (
            "close",
            ("registered_execution", "source_bundle_close_failed"),
            [
                "prepare",
                "cpu",
                "create",
                "source_frame",
                "source_frame",
                "select",
                "source_frame",
                "close",
            ],
        ),
        (
            "build_completed",
            ("completed_evidence", "completed_evidence_rejected"),
            [
                "prepare",
                "cpu",
                "create",
                "source_frame",
                "source_frame",
                "select",
                "source_frame",
                "close",
                "build_completed",
            ],
        ),
        (
            "verify_completed",
            ("completed_evidence", "completed_evidence_rejected"),
            [
                "prepare",
                "cpu",
                "create",
                "source_frame",
                "source_frame",
                "select",
                "source_frame",
                "close",
                "build_completed",
                "verify_completed",
            ],
        ),
        (
            "quiescence_active",
            ("parent_boundary", "pre_cleanup_quiescence_failed"),
            [
                "prepare",
                "cpu",
                "create",
                "source_frame",
                "source_frame",
                "select",
                "source_frame",
                "close",
                "build_completed",
                "verify_completed",
                "quiescence_active",
            ],
        ),
    ],
)
def test_registered_parent_publishes_only_closed_controlled_failure_pairs(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    failed_stage: str,
    expected_pair: tuple[str, str],
    expected_prefix: list[str],
) -> None:
    primary = RuntimeError("SECRET PRIMARY DETAIL")
    harness = _RegisteredParentTrace(failures={failed_stage: primary})
    with pytest.raises(
        coordinator.Experiment002CoordinatorError,
        match=rf"{expected_pair[0]}/{expected_pair[1]}",
    ) as captured:
        coordinator._run_registered_experiment_once(
            synthetic_registration,
            _admitted_registration_frame(),
            harness.operations(),
        )
    assert captured.value.__cause__ is primary
    assert cast(list[str], harness.trace)[: len(expected_prefix)] == expected_prefix
    assert cast(list[str], harness.trace)[-6:] == [
        "cleanup",
        "quiescence_closed",
        "registration_frame",
        "build_failure",
        "verify_failure",
        "publish",
    ]
    assert harness.failure_pairs == [(_admitted_registration_frame(), *expected_pair)]
    assert harness.published is not None
    assert len(harness.published) == 1
    published = harness.published[0]
    assert cast(Any, published).phase == expected_pair[0]
    assert cast(Any, published).code == expected_pair[1]
    assert "SECRET PRIMARY DETAIL" not in repr(vars(published))
    assert harness.authority_calls_at_publish == harness.authority_calls
    assert harness.close_calls == (
        1
        if failed_stage
        in {
            "select",
            "close",
            "build_completed",
            "verify_completed",
            "quiescence_active",
        }
        else 0
    )


def test_registered_parent_close_failure_overrides_selector_failure_once(
    synthetic_registration: coordinator.VerifiedRunRegistration,
) -> None:
    selector_failure = RuntimeError("selector secret")
    close_failure = OSError("close secret")
    harness = _RegisteredParentTrace(
        failures={"select": selector_failure, "close": close_failure}
    )
    with pytest.raises(
        coordinator.Experiment002CoordinatorError,
        match="registered_execution/source_bundle_close_failed",
    ) as captured:
        coordinator._run_registered_experiment_once(
            synthetic_registration,
            _admitted_registration_frame(),
            harness.operations(),
        )
    assert captured.value.__cause__ is close_failure
    assert harness.close_calls == 1
    assert harness.failure_pairs == [
        (
            _admitted_registration_frame(),
            "registered_execution",
            "source_bundle_close_failed",
        )
    ]


def test_registered_parent_non_none_completed_verifier_is_rejected(
    synthetic_registration: coordinator.VerifiedRunRegistration,
) -> None:
    harness = _RegisteredParentTrace(verify_completed_result=object())
    with pytest.raises(
        coordinator.Experiment002CoordinatorError,
        match="completed_evidence/completed_evidence_rejected",
    ):
        coordinator._run_registered_experiment_once(
            synthetic_registration,
            _admitted_registration_frame(),
            harness.operations(),
        )
    assert harness.failure_pairs == [
        (
            _admitted_registration_frame(),
            "completed_evidence",
            "completed_evidence_rejected",
        )
    ]
    assert harness.close_calls == 1
    assert "quiescence_active" not in cast(list[str], harness.trace)
    assert cast(list[str], harness.trace).count("quiescence_closed") == 1
    assert cast(list[str], harness.trace)[-6:] == [
        "cleanup",
        "quiescence_closed",
        "registration_frame",
        "build_failure",
        "verify_failure",
        "publish",
    ]


@pytest.mark.parametrize("frame_call", [2, 3])
def test_registered_parent_pre_and_post_selection_frame_rejection_close_once(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    frame_call: int,
) -> None:
    frame_failure = RuntimeError("source-frame secret")
    harness = _RegisteredParentTrace(source_frame_failures={frame_call: frame_failure})
    with pytest.raises(
        coordinator.Experiment002CoordinatorError,
        match="registered_execution/seed_selection_failed",
    ) as captured:
        coordinator._run_registered_experiment_once(
            synthetic_registration,
            _admitted_registration_frame(),
            harness.operations(),
        )
    assert captured.value.__cause__ is frame_failure
    assert harness.close_calls == 1
    assert harness.frame_calls == frame_call
    assert harness.failure_pairs == [
        (
            _admitted_registration_frame(),
            "registered_execution",
            "seed_selection_failed",
        )
    ]


def test_registered_parent_non_none_close_result_has_close_failure_precedence(
    synthetic_registration: coordinator.VerifiedRunRegistration,
) -> None:
    harness = _RegisteredParentTrace(close_result=object())
    with pytest.raises(
        coordinator.Experiment002CoordinatorError,
        match="registered_execution/source_bundle_close_failed",
    ):
        coordinator._run_registered_experiment_once(
            synthetic_registration,
            _admitted_registration_frame(),
            harness.operations(),
        )
    assert harness.close_calls == 1
    assert harness.failure_pairs == [
        (
            _admitted_registration_frame(),
            "registered_execution",
            "source_bundle_close_failed",
        )
    ]


def test_registered_parent_boundary_loss_never_becomes_reportable_on_close_error(
    synthetic_registration: coordinator.VerifiedRunRegistration,
) -> None:
    close_failure = OSError("close secret")
    harness = _RegisteredParentTrace(
        failures={"close": close_failure},
        break_authority_during_selector=True,
    )
    with pytest.raises(
        coordinator.Experiment002CoordinatorError,
        match="terminal cleanup boundary failed closed",
    ):
        coordinator._run_registered_experiment_once(
            synthetic_registration,
            _admitted_registration_frame(),
            harness.operations(),
        )
    assert harness.close_calls == 1
    assert harness.frame_calls == 2
    assert harness.failure_pairs == []
    assert harness.published == []
    assert "build_failure" not in cast(list[str], harness.trace)
    assert "publish" not in cast(list[str], harness.trace)


def test_registered_parent_transient_boundary_loss_stays_unreportable_after_close(
    synthetic_registration: coordinator.VerifiedRunRegistration,
) -> None:
    harness = _RegisteredParentTrace(
        break_authority_during_selector=True,
        restore_authority_during_close=True,
    )
    with pytest.raises(
        coordinator.Experiment002CoordinatorError,
        match="unreportable authority boundary",
    ):
        coordinator._run_registered_experiment_once(
            synthetic_registration,
            _admitted_registration_frame(),
            harness.operations(),
        )
    assert harness.close_calls == 1
    assert harness.failure_pairs == []
    assert harness.published == []
    assert cast(list[str], harness.trace)[-3:] == [
        "cleanup",
        "quiescence_closed",
        "registration_frame",
    ]


def test_registered_parent_claims_created_fd_before_post_creation_authority_loss(
    synthetic_registration: coordinator.VerifiedRunRegistration,
) -> None:
    harness = _RegisteredParentTrace(break_authority_during_create=True)
    with pytest.raises(
        coordinator.Experiment002CoordinatorError,
        match="terminal cleanup boundary failed closed",
    ):
        coordinator._run_registered_experiment_once(
            synthetic_registration,
            _admitted_registration_frame(),
            harness.operations(),
        )
    assert harness.trace == ["prepare", "cpu", "create", "close"]
    assert harness.close_calls == 1
    assert harness.frame_calls == 0
    assert harness.failure_pairs == []
    assert harness.published == []


@pytest.mark.parametrize(
    ("mutation_stage", "expected_frame_calls"),
    [
        ("create", 0),
        ("select", 2),
        ("close", 3),
    ],
)
def test_registered_parent_external_checker_anchor_closes_owned_fd_once(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    mutation_stage: str,
    expected_frame_calls: int,
) -> None:
    harness = _RegisteredParentTrace()
    operations = harness.operations()
    checker = cast(FunctionType, harness.authority_route)
    original_code = checker.__code__

    def compatible_noop() -> None:
        if harness.authority_broken:
            raise AssertionError

    assert compatible_noop.__closure__ is not None
    assert checker.__closure__ is not None
    assert len(compatible_noop.__closure__) == len(checker.__closure__)
    assert harness.mutations is not None
    harness.mutations[mutation_stage] = lambda: setattr(
        checker,
        "__code__",
        compatible_noop.__code__,
    )
    try:
        with pytest.raises(
            coordinator.Experiment002CoordinatorError,
            match="terminal cleanup boundary failed closed",
        ):
            coordinator._run_registered_experiment_once(
                synthetic_registration,
                _admitted_registration_frame(),
                operations,
            )
    finally:
        checker.__code__ = original_code
    assert harness.close_calls == 1
    assert harness.frame_calls == expected_frame_calls
    assert harness.failure_pairs == []
    assert harness.published == []
    assert "cleanup" not in cast(list[str], harness.trace)
    assert "quiescence_closed" not in cast(list[str], harness.trace)
    assert cast(list[str], harness.trace)[-1] == "close"


@pytest.mark.parametrize("target_name", ["integrity_verifier", "operation_caller"])
def test_registered_parent_close_precedes_external_anchor_rejection(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    target_name: str,
) -> None:
    harness = _RegisteredParentTrace()
    operations = harness.operations()
    if target_name == "integrity_verifier":
        target = cast(
            FunctionType,
            coordinator._require_recursive_function_integrity_unchanged,
        )

        def verifier_noop(_frame: object, /) -> None:
            return None

        replacement_code = verifier_noop.__code__
    else:
        target = cast(
            FunctionType,
            coordinator._call_registered_experiment_operation,
        )

        def caller_noop(*_arguments: object, **_keywords: object) -> None:
            return None

        replacement_code = caller_noop.__code__
    original_code = target.__code__
    assert harness.mutations is not None
    harness.mutations["close"] = lambda: setattr(
        target,
        "__code__",
        replacement_code,
    )
    try:
        with pytest.raises(
            coordinator.Experiment002CoordinatorError,
            match="terminal cleanup boundary failed closed",
        ):
            coordinator._run_registered_experiment_once(
                synthetic_registration,
                _admitted_registration_frame(),
                operations,
            )
    finally:
        target.__code__ = original_code
    assert cast(list[str], harness.trace)[-1] == "close"
    assert harness.close_calls == 1
    assert harness.failure_pairs == []
    assert harness.published == []


def test_registered_parent_creator_handoff_uses_preallocated_slot() -> None:
    source = inspect.getsource(coordinator._call_registered_experiment_operation)
    assert "_result_box[0] = result" in source
    assert "list_append" not in source


def test_registered_parent_rejects_non_none_completed_publisher_result_without_recheck(
    synthetic_registration: coordinator.VerifiedRunRegistration,
) -> None:
    unexpected = object()
    harness = _RegisteredParentTrace(publish_result=unexpected)
    with pytest.raises(
        coordinator.Experiment002CoordinatorError,
        match="final publication failed closed",
    ) as captured:
        coordinator._run_registered_experiment_once(
            synthetic_registration,
            _admitted_registration_frame(),
            harness.operations(),
        )
    assert type(captured.value.__cause__) is coordinator.Experiment002CoordinatorError
    assert "unexpected value" in str(captured.value.__cause__)
    assert harness.published is not None
    assert len(harness.published) == 1
    assert harness.authority_calls_at_publish == harness.authority_calls


def test_registered_parent_completed_publisher_exception_is_terminal_without_fallback(
    synthetic_registration: coordinator.VerifiedRunRegistration,
) -> None:
    publisher_failure = RuntimeError("publisher secret")
    harness = _RegisteredParentTrace(failures={"publish": publisher_failure})
    with pytest.raises(
        coordinator.Experiment002CoordinatorError,
        match="final publication failed closed",
    ) as captured:
        coordinator._run_registered_experiment_once(
            synthetic_registration,
            _admitted_registration_frame(),
            harness.operations(),
        )
    assert captured.value.__cause__ is publisher_failure
    assert cast(list[str], harness.trace).count("publish") == 1
    assert "build_failure" not in cast(list[str], harness.trace)
    assert "verify_failure" not in cast(list[str], harness.trace)
    assert harness.failure_pairs == []
    assert harness.published == []


@pytest.mark.parametrize(
    ("field_name", "unexpected"),
    [
        ("cleanup_result", object()),
        ("quiescence_results", (None, object())),
    ],
)
def test_registered_parent_non_none_cleanup_boundaries_prohibit_publication(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    field_name: str,
    unexpected: object,
) -> None:
    harness = _RegisteredParentTrace()
    setattr(harness, field_name, unexpected)
    with pytest.raises(
        coordinator.Experiment002CoordinatorError,
        match="terminal cleanup boundary failed closed",
    ):
        coordinator._run_registered_experiment_once(
            synthetic_registration,
            _admitted_registration_frame(),
            harness.operations(),
        )
    assert harness.published == []
    assert harness.failure_pairs == []


def test_registered_parent_never_truth_tests_falsey_postcleanup_failure(
    synthetic_registration: coordinator.VerifiedRunRegistration,
) -> None:
    class FalseyFailure(RuntimeError):
        def __bool__(self) -> bool:
            return False

    closed_failure = FalseyFailure("closed boundary secret")
    harness = _RegisteredParentTrace(
        failures={
            "cpu": RuntimeError("controlled primary"),
            "quiescence_closed": closed_failure,
        }
    )
    with pytest.raises(
        coordinator.Experiment002CoordinatorError,
        match="terminal cleanup boundary failed closed",
    ) as captured:
        coordinator._run_registered_experiment_once(
            synthetic_registration,
            _admitted_registration_frame(),
            harness.operations(),
        )
    assert captured.value.__cause__ is closed_failure
    assert harness.failure_pairs == []
    assert harness.published == []


@pytest.mark.parametrize(
    "failed_stage",
    [
        "cleanup",
        "quiescence_closed",
        "registration_frame",
        "build_failure",
        "verify_failure",
        "publish",
    ],
)
def test_registered_parent_terminal_failures_prohibit_any_second_publication(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    failed_stage: str,
) -> None:
    controlled = RuntimeError("controlled primary")
    terminal = RuntimeError("terminal primary")
    harness = _RegisteredParentTrace(
        failures={"cpu": controlled, failed_stage: terminal}
    )
    with pytest.raises(coordinator.Experiment002CoordinatorError):
        coordinator._run_registered_experiment_once(
            synthetic_registration,
            _admitted_registration_frame(),
            harness.operations(),
        )
    assert cast(list[str], harness.trace).count("cleanup") == 1
    assert cast(list[str], harness.trace).count("quiescence_closed") == 1
    assert cast(list[str], harness.trace).count("publish") <= 1
    if failed_stage in {
        "cleanup",
        "quiescence_closed",
        "registration_frame",
        "build_failure",
        "verify_failure",
    }:
        assert harness.published == []
    else:
        assert harness.published == []
    assert harness.close_calls == 0


def test_registered_parent_source_frame_rejection_still_closes_once(
    synthetic_registration: coordinator.VerifiedRunRegistration,
) -> None:
    inspection_failure = RuntimeError("inspection secret")
    harness = _RegisteredParentTrace(failures={"source_frame": inspection_failure})
    with pytest.raises(
        coordinator.Experiment002CoordinatorError,
        match="parent_setup/source_bundle_create_failed",
    ) as captured:
        coordinator._run_registered_experiment_once(
            synthetic_registration,
            _admitted_registration_frame(),
            harness.operations(),
        )
    assert captured.value.__cause__ is inspection_failure
    assert harness.close_calls == 1
    assert harness.frame_calls == 1
    assert harness.failure_pairs == [
        (
            _admitted_registration_frame(),
            "parent_setup",
            "source_bundle_create_failed",
        )
    ]


def test_public_parent_owner_thread_rejection_does_not_burn_attempt(
    synthetic_registration: coordinator.VerifiedRunRegistration,
) -> None:
    failures: list[BaseException] = []

    def call() -> None:
        try:
            coordinator.run_registered_experiment(synthetic_registration)
        except BaseException as error:
            failures.append(error)

    thread = threading.Thread(target=call)
    thread.start()
    thread.join()
    assert len(failures) == 1
    assert type(failures[0]) is coordinator.Experiment002CoordinatorError
    assert "caller thread changed" in str(failures[0])
    state = _registered_experiment_state()
    assert state.attempted is False
    assert state.in_flight is False
