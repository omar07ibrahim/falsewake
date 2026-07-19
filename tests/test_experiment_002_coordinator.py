from __future__ import annotations

import array
import ast
import copy
import errno
import fcntl
import importlib
import inspect
import itertools
import json
import os
import pickle
import socket
import subprocess
import sys
import textwrap
import warnings
import weakref
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
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
    registration = object.__new__(coordinator.VerifiedRunRegistration)

    def verify(value: coordinator.VerifiedRunRegistration) -> None:
        if type(value) is not coordinator.VerifiedRunRegistration:
            raise TypeError("not an exact synthetic registration")

    def binding(
        value: coordinator.VerifiedRunRegistration,
    ) -> coordinator._RegistrationBinding:
        verify(value)
        return TEST_BINDING

    monkeypatch.setattr(coordinator, "verify_verified_run_registration", verify)
    monkeypatch.setattr(coordinator, "_registration_binding", binding)
    return registration


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
    assert "numpy" not in source
    assert "torch" not in source
    assert "experiment_002_training" not in source


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

        registration = object.__new__(coordinator.VerifiedRunRegistration)
        binding = coordinator._RegistrationBinding(
            head_commit='1' * 40,
            implementation_commit='2' * 40,
            registration_sha256='3' * 64,
            source_bundle_sha256='4' * 64,
        )

        def verify_registration(value):
            if type(value) is not coordinator.VerifiedRunRegistration:
                raise TypeError('synthetic registration type changed')

        def registration_binding(value):
            verify_registration(value)
            return binding

        coordinator.verify_verified_run_registration = verify_registration
        coordinator._registration_binding = registration_binding
        parent_channel, child_channel = socket.socketpair(
            socket.AF_UNIX,
            socket.SOCK_SEQPACKET,
        )
        child_pid = os.fork()
        if child_pid == 0:
            try:
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


def test_public_parent_reverifies_then_stops_before_supervisor(
    synthetic_registration: coordinator.VerifiedRunRegistration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[coordinator.VerifiedRunRegistration] = []

    def verify(value: coordinator.VerifiedRunRegistration) -> None:
        calls.append(value)

    monkeypatch.setattr(coordinator, "verify_verified_run_registration", verify)
    with pytest.raises(
        coordinator.Experiment002CoordinatorError,
        match="supervisor and resource-control",
    ):
        coordinator.run_registered_experiment(synthetic_registration)
    assert calls == [synthetic_registration]
    with pytest.raises(TypeError):
        cast(Any, coordinator.run_registered_experiment)(
            registration=synthetic_registration
        )


def test_public_parent_rejects_forged_and_subclassed_registrations() -> None:
    class Subclass(authority.VerifiedRunRegistration):
        pass

    for value in (object(), object.__new__(Subclass)):
        with pytest.raises(TypeError, match="exact VerifiedRunRegistration"):
            coordinator.run_registered_experiment(cast(Any, value))


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
