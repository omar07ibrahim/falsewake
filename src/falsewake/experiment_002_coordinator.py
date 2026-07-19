"""Fail-closed process coordination for the registered Experiment 002 run.

The public parent entrypoint deliberately stops before spawning anything until
the separately reviewed supervisor and resource-control layer exists.  This
module already defines the activation boundary that layer must use: every
fresh child first obtains its own process-local run registration, then accepts
one kernel-credentialled ticket over Unix ``SOCK_SEQPACKET`` descriptor 3.

Only standard-library modules and the standard-library-only run authority are
imported here.  Numerical code is imported dynamically after a child activation
has been verified and claimed for its single dispatch.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import secrets
import socket
import struct
import threading
import weakref
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Final, NoReturn, Protocol, SupportsIndex, cast

from falsewake.experiment_002_run_authority import (
    VerifiedRunRegistration,
    verify_verified_run_registration,
)

__all__ = (
    "Experiment002CoordinatorError",
    "VerifiedChildActivation",
    "VerifiedRunRegistration",
    "run_registered_experiment",
    "verify_verified_child_activation",
)

REGISTERED_SEEDS: Final = (20_260_719, 20_260_720, 20_260_721)

_CHILD_CONTROL_FD: Final = 3
_SEED_ROLE: Final = "training_seed"
_RERUN_ROLE: Final = "selected_seed_rerun"
_WORKER_MODULE: Final = "falsewake.experiment_002_seed_worker"
_WORKER_FUNCTION: Final = "run_registered_seed_process"
_SCHEMA_VERSION: Final = 1
_FRAME_MAGIC: Final = b"FW2ACTV1"
_FRAME_HEADER: Final = struct.Struct(">8sI")
_CREDENTIALS: Final = struct.Struct("=3i")
_MAX_JSON_BYTES: Final = 3_072
_MAX_PACKET_BYTES: Final = _FRAME_HEADER.size + _MAX_JSON_BYTES
_PROTOCOL_TIMEOUT_SECONDS: Final = 30.0
_TICKET_DIGEST_DOMAIN: Final = b"falsewake-exp002-activation-ticket-v1\0"
_LOWER_HEX: Final = frozenset("0123456789abcdef")
_ACTIVATION_MARKER: Final = object()
_PHASE_ISSUED: Final = 1
_PHASE_DISPATCHED: Final = 2
_PHASE_COMPLETED: Final = 3


class Experiment002CoordinatorError(RuntimeError):
    """Experiment 002 process coordination failed closed."""


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
class VerifiedChildActivation:
    """Opaque proof of one child-local, kernel-authenticated activation."""

    def __init__(self) -> None:
        raise TypeError("child activations are issued by the coordinator")

    @property
    def role(self) -> str:
        """Return the exact registered child role."""

        return _activation_state(self).role

    @property
    def seed(self) -> int:
        """Return the registered seed assigned to this child."""

        return _activation_state(self).seed

    @property
    def ordinal(self) -> int:
        """Return the registered zero-based child execution ordinal."""

        return _activation_state(self).ordinal

    @property
    def child_pid(self) -> int:
        """Return the PID to which this activation is bound."""

        return _activation_state(self).child_pid

    def __copy__(self) -> NoReturn:
        raise TypeError("child activations cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        del memo
        raise TypeError("child activations cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("child activations cannot be serialized")

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        del protocol
        raise TypeError("child activations cannot be serialized")


@dataclass(frozen=True, slots=True)
class _RegistrationBinding:
    head_commit: str
    implementation_commit: str
    registration_sha256: str
    source_bundle_sha256: str


@dataclass(frozen=True, slots=True)
class _Assignment:
    role: str
    seed: int
    ordinal: int


@dataclass(frozen=True, slots=True)
class _PeerCredentials:
    pid: int
    uid: int
    gid: int


@dataclass(frozen=True, slots=True)
class _Ticket:
    assignment: _Assignment
    parent_pid: int
    child_pid: int
    nonce: str
    packet_sha256: str


@dataclass(frozen=True, slots=True)
class _ActivationState:
    marker: object
    registration: VerifiedRunRegistration
    process_id: int
    parent_pid: int
    child_pid: int
    role: str
    seed: int
    ordinal: int
    nonce: str
    ticket_sha256: str
    token: object
    phase: int


@dataclass(frozen=True, slots=True)
class _ActivationGuard:
    marker: object
    registration: VerifiedRunRegistration
    process_id: int
    parent_pid: int
    child_pid: int
    role: str
    seed: int
    ordinal: int
    nonce: str
    ticket_sha256: str
    token: object
    phase: int


@dataclass(frozen=True, slots=True)
class _ActivationAnchor:
    marker: object
    registration: VerifiedRunRegistration
    process_id: int
    parent_pid: int
    child_pid: int
    role: str
    seed: int
    ordinal: int
    nonce: str
    ticket_sha256: str
    token: object


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
class _AcceptedTicketReceipt:
    """Opaque proof that one exact ticket passed child-side validation."""

    def __init__(self) -> None:
        raise TypeError("accepted-ticket receipts are issued by the coordinator")


@dataclass(frozen=True, slots=True)
class _TrustedActivationAnchor:
    """Closure-owned issuance fact independent of mutable module registries."""

    activation_ref: weakref.ReferenceType[VerifiedChildActivation]
    registration: VerifiedRunRegistration
    ticket: _Ticket
    receipt: _AcceptedTicketReceipt
    receipt_token: object
    activation_token: object
    process_id: int
    marker: object


class _SeedWorker(Protocol):
    def run_registered_seed_process(
        self,
        registration: VerifiedRunRegistration,
        activation: VerifiedChildActivation,
        /,
    ) -> None:
        """Execute exactly one activated seed process."""


_ACTIVATIONS: weakref.WeakKeyDictionary[VerifiedChildActivation, _ActivationState] = (
    weakref.WeakKeyDictionary()
)
_ACTIVATION_GUARDS: weakref.WeakKeyDictionary[
    VerifiedChildActivation, _ActivationGuard
] = weakref.WeakKeyDictionary()
_ACTIVATION_ANCHORS: weakref.WeakKeyDictionary[
    VerifiedChildActivation, _ActivationAnchor
] = weakref.WeakKeyDictionary()
_ACTIVATION_PHASES: weakref.WeakKeyDictionary[VerifiedChildActivation, int] = (
    weakref.WeakKeyDictionary()
)
_DISPATCHED_ACTIVATIONS: weakref.WeakSet[VerifiedChildActivation] = weakref.WeakSet()
_COMPLETED_ACTIVATIONS: weakref.WeakSet[VerifiedChildActivation] = weakref.WeakSet()
_FAILED_ACTIVATIONS: weakref.WeakSet[VerifiedChildActivation] = weakref.WeakSet()
_ACCEPTED_TICKET_DIGESTS: set[str] = set()
_ACCEPTED_TICKET_NONCES: set[str] = set()
_ISSUED_TICKET_DIGESTS: set[str] = set()
_ISSUED_TICKET_NONCES: set[str] = set()
_PROTOCOL_LOCK = threading.RLock()


def _make_trusted_protocol_ledger() -> tuple[
    Callable[[str, str], None],
    Callable[[_Ticket], _AcceptedTicketReceipt],
    Callable[
        [
            _AcceptedTicketReceipt,
            _Ticket,
            VerifiedChildActivation,
            VerifiedRunRegistration,
            object,
        ],
        None,
    ],
    Callable[
        [VerifiedChildActivation, object],
        tuple[int | None, bool, bool],
    ],
    Callable[[VerifiedChildActivation, int, int], None],
    Callable[[VerifiedChildActivation], None],
]:
    """Build process-local append-only truth hidden from mutable state stores."""

    issued_nonces: set[str] = set()
    issued_digests: set[str] = set()
    accepted_nonces: set[str] = set()
    accepted_digests: set[str] = set()
    receipt_records: weakref.WeakKeyDictionary[
        _AcceptedTicketReceipt, tuple[_Ticket, object]
    ] = weakref.WeakKeyDictionary()
    consumed_receipts: weakref.WeakSet[_AcceptedTicketReceipt] = weakref.WeakSet()
    issuance_anchors: weakref.WeakKeyDictionary[
        VerifiedChildActivation, _TrustedActivationAnchor
    ] = weakref.WeakKeyDictionary()
    dispatched: weakref.WeakSet[VerifiedChildActivation] = weakref.WeakSet()
    completed: weakref.WeakSet[VerifiedChildActivation] = weakref.WeakSet()
    failed: weakref.WeakSet[VerifiedChildActivation] = weakref.WeakSet()
    ledger_lock = threading.RLock()

    def reserve_issued(nonce: str, digest: str) -> None:
        with ledger_lock:
            if nonce in issued_nonces or digest in issued_digests:
                raise Experiment002CoordinatorError(
                    "activation ticket was already issued"
                )
            issued_nonces.add(nonce)
            issued_digests.add(digest)

    def accept_ticket(ticket: _Ticket) -> _AcceptedTicketReceipt:
        with ledger_lock:
            if (
                ticket.nonce in accepted_nonces
                or ticket.packet_sha256 in accepted_digests
            ):
                raise Experiment002CoordinatorError("activation ticket was replayed")
            accepted_nonces.add(ticket.nonce)
            accepted_digests.add(ticket.packet_sha256)
            receipt = object.__new__(_AcceptedTicketReceipt)
            receipt_records[receipt] = (ticket, object())
            return receipt

    def record_issuance(
        receipt: _AcceptedTicketReceipt,
        ticket: _Ticket,
        activation: VerifiedChildActivation,
        registration: VerifiedRunRegistration,
        activation_token: object,
    ) -> None:
        with ledger_lock:
            if type(receipt) is not _AcceptedTicketReceipt:
                raise Experiment002CoordinatorError(
                    "activation lacks an exact accepted-ticket receipt"
                )
            receipt_record = receipt_records.get(receipt)
            if (
                receipt_record is None
                or receipt in consumed_receipts
                or receipt_record[0] != ticket
                or activation in issuance_anchors
            ):
                raise Experiment002CoordinatorError(
                    "accepted-ticket receipt is forged, mismatched, or consumed"
                )
            consumed_receipts.add(receipt)
            issuance_anchors[activation] = _TrustedActivationAnchor(
                activation_ref=weakref.ref(activation),
                registration=registration,
                ticket=ticket,
                receipt=receipt,
                receipt_token=receipt_record[1],
                activation_token=activation_token,
                process_id=os.getpid(),
                marker=_ACTIVATION_MARKER,
            )

    def receipt_linked(
        activation: VerifiedChildActivation,
        anchor: _TrustedActivationAnchor,
    ) -> bool:
        receipt_record = receipt_records.get(anchor.receipt)
        return (
            anchor.activation_ref() is activation
            and anchor.receipt in consumed_receipts
            and receipt_record is not None
            and receipt_record[0] == anchor.ticket
            and receipt_record[1] is anchor.receipt_token
        )

    def lifecycle_phase(activation: VerifiedChildActivation) -> int | None:
        was_dispatched = activation in dispatched
        was_completed = activation in completed
        if was_completed:
            return _PHASE_COMPLETED if was_dispatched else None
        if was_dispatched:
            return _PHASE_DISPATCHED
        return _PHASE_ISSUED

    def snapshot(
        activation: VerifiedChildActivation,
        state: object,
    ) -> tuple[int | None, bool, bool]:
        with ledger_lock:
            anchor = issuance_anchors.get(activation)
            is_failed = activation in failed
            if anchor is None:
                return None, is_failed, False
            linked = receipt_linked(activation, anchor)
            phase = lifecycle_phase(activation)
            if type(state) is not _ActivationState:
                return phase, is_failed, False
            try:
                _require_ticket(anchor.ticket)
            except (TypeError, Experiment002CoordinatorError):
                return phase, is_failed, False
            ticket = anchor.ticket
            matches = (
                anchor.registration is state.registration
                and anchor.process_id == os.getpid()
                and anchor.process_id == state.process_id
                and anchor.marker is _ACTIVATION_MARKER
                and anchor.marker is state.marker
                and type(anchor.receipt) is _AcceptedTicketReceipt
                and type(anchor.receipt_token) is object
                and type(anchor.activation_token) is object
                and anchor.activation_token is state.token
                and ticket.parent_pid == state.parent_pid
                and ticket.child_pid == state.child_pid
                and ticket.assignment.role == state.role
                and ticket.assignment.seed == state.seed
                and ticket.assignment.ordinal == state.ordinal
                and ticket.nonce == state.nonce
                and ticket.packet_sha256 == state.ticket_sha256
            )
            return phase, is_failed, linked and matches

    def advance(
        activation: VerifiedChildActivation,
        expected_phase: int,
        next_phase: int,
    ) -> None:
        with ledger_lock:
            anchor = issuance_anchors.get(activation)
            phase = lifecycle_phase(activation)
            if (
                anchor is None
                or activation in failed
                or not receipt_linked(activation, anchor)
                or phase != expected_phase
            ):
                failed.add(activation)
                raise Experiment002CoordinatorError(
                    "trusted activation lifecycle transition was replayed"
                )
            if expected_phase == _PHASE_ISSUED and next_phase == _PHASE_DISPATCHED:
                dispatched.add(activation)
                return
            if expected_phase == _PHASE_DISPATCHED and next_phase == _PHASE_COMPLETED:
                completed.add(activation)
                return
            failed.add(activation)
            raise Experiment002CoordinatorError(
                "trusted activation lifecycle transition is not registered"
            )

    def poison(activation: VerifiedChildActivation) -> None:
        if type(activation) is VerifiedChildActivation:
            with ledger_lock:
                failed.add(activation)

    return reserve_issued, accept_ticket, record_issuance, snapshot, advance, poison


(
    _TRUSTED_RESERVE_ISSUED,
    _TRUSTED_ACCEPT_TICKET,
    _TRUSTED_RECORD_ISSUANCE,
    _TRUSTED_VERIFY_ACTIVATION,
    _TRUSTED_ADVANCE_ACTIVATION,
    _TRUSTED_POISON_ACTIVATION,
) = _make_trusted_protocol_ledger()


def run_registered_experiment(registration: VerifiedRunRegistration, /) -> None:
    """Admit the parent capability, then stop before an unreviewed supervisor.

    There are intentionally no command, path, seed, callback, environment, or
    resource overrides.  The later supervisor implementation will replace only
    this terminal error and will use the private fixed protocol below.
    """

    _require_registration(registration)
    raise Experiment002CoordinatorError(
        "the registered supervisor and resource-control layer are not enabled"
    )


def verify_verified_child_activation(
    registration: VerifiedRunRegistration,
    activation: VerifiedChildActivation,
    /,
) -> None:
    """Reverify one process-local registration and its bound activation."""

    if type(registration) is not VerifiedRunRegistration:
        raise TypeError("registration must be an exact VerifiedRunRegistration")
    state = _activation_state(activation)
    if state.registration is not registration:
        _poison_activation(activation)
        raise Experiment002CoordinatorError(
            "child activation belongs to a different run registration"
        )
    try:
        verify_verified_run_registration(registration)
    except BaseException:
        _poison_activation(activation)
        raise


def _require_registration(registration: VerifiedRunRegistration) -> None:
    if type(registration) is not VerifiedRunRegistration:
        raise TypeError("registration must be an exact VerifiedRunRegistration")
    verify_verified_run_registration(registration)


def _registration_binding(
    registration: VerifiedRunRegistration,
) -> _RegistrationBinding:
    _require_registration(registration)
    binding = _RegistrationBinding(
        head_commit=registration.head_commit,
        implementation_commit=registration.implementation_commit,
        registration_sha256=registration.registration_sha256,
        source_bundle_sha256=registration.source_bundle_sha256,
    )
    _require_registration_binding(binding)
    return binding


def _require_registration_binding(binding: _RegistrationBinding) -> None:
    if type(binding) is not _RegistrationBinding:
        raise Experiment002CoordinatorError("registration binding has an invalid type")
    _require_lower_hex(binding.head_commit, 40, "HEAD commit")
    _require_lower_hex(binding.implementation_commit, 40, "implementation commit")
    _require_lower_hex(binding.registration_sha256, 64, "registration sha256")
    _require_lower_hex(binding.source_bundle_sha256, 64, "source bundle sha256")


def _assignment(role: object, seed: object) -> _Assignment:
    if type(role) is not str:
        raise TypeError("child role must be a string")
    if type(seed) is not int:
        raise TypeError("child seed must be an integer")
    if role == _SEED_ROLE:
        try:
            ordinal = REGISTERED_SEEDS.index(seed)
        except ValueError as error:
            raise Experiment002CoordinatorError(
                "training child seed is not one of the three registered seeds"
            ) from error
        return _Assignment(role=role, seed=seed, ordinal=ordinal)
    if role == _RERUN_ROLE and seed in REGISTERED_SEEDS:
        return _Assignment(role=role, seed=seed, ordinal=3)
    raise Experiment002CoordinatorError("child role and seed are not registered")


def _binding_document(binding: _RegistrationBinding) -> dict[str, object]:
    _require_registration_binding(binding)
    return {
        "head_commit": binding.head_commit,
        "implementation_commit": binding.implementation_commit,
        "registration_sha256": binding.registration_sha256,
        "source_bundle_sha256": binding.source_bundle_sha256,
    }


def _ticket_document(
    binding: _RegistrationBinding,
    assignment: _Assignment,
    *,
    parent_pid: int,
    child_pid: int,
    nonce: str,
) -> dict[str, object]:
    _require_assignment(assignment)
    _require_distinct_processes(parent_pid, child_pid)
    _require_lower_hex(nonce, 64, "activation nonce")
    return {
        "child_pid": child_pid,
        "kind": "activation_ticket",
        "nonce": nonce,
        "ordinal": assignment.ordinal,
        "parent_pid": parent_pid,
        "registration": _binding_document(binding),
        "role": assignment.role,
        "schema_version": _SCHEMA_VERSION,
        "seed": assignment.seed,
    }


def _ack_document(ticket: _Ticket) -> dict[str, object]:
    _require_ticket(ticket)
    return {
        "child_pid": ticket.child_pid,
        "kind": "activation_ack",
        "nonce": ticket.nonce,
        "ordinal": ticket.assignment.ordinal,
        "parent_pid": ticket.parent_pid,
        "role": ticket.assignment.role,
        "schema_version": _SCHEMA_VERSION,
        "seed": ticket.assignment.seed,
        "status": "accepted",
        "ticket_sha256": ticket.packet_sha256,
    }


def _encode_packet(document: dict[str, object]) -> bytes:
    if type(document) is not dict:
        raise TypeError("protocol document must be an exact dictionary")
    try:
        payload = (
            json.dumps(
                document,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise Experiment002CoordinatorError(
            "protocol document is not canonical ASCII JSON"
        ) from error
    if not payload or len(payload) > _MAX_JSON_BYTES:
        raise Experiment002CoordinatorError("protocol JSON size is invalid")
    return _FRAME_HEADER.pack(_FRAME_MAGIC, len(payload)) + payload


def _decode_packet(packet: bytes) -> dict[str, Any]:
    if type(packet) is not bytes:
        raise TypeError("protocol packet must be exact bytes")
    if not _FRAME_HEADER.size < len(packet) <= _MAX_PACKET_BYTES:
        raise Experiment002CoordinatorError("protocol packet size is invalid")
    magic, payload_length = _FRAME_HEADER.unpack(packet[: _FRAME_HEADER.size])
    payload = packet[_FRAME_HEADER.size :]
    if magic != _FRAME_MAGIC or payload_length != len(payload):
        raise Experiment002CoordinatorError("protocol frame header is invalid")
    try:
        text = payload.decode("ascii", errors="strict")
        parsed = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_number,
            parse_float=_reject_json_number,
        )
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as error:
        raise Experiment002CoordinatorError(
            "protocol payload is not exact JSON"
        ) from error
    if type(parsed) is not dict:
        raise Experiment002CoordinatorError("protocol payload must be an object")
    document = cast(dict[str, Any], parsed)
    if _encode_packet(cast(dict[str, object], document)) != packet:
        raise Experiment002CoordinatorError("protocol payload is not canonical")
    return document


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate protocol key: {key}")
        result[key] = value
    return result


def _reject_json_number(value: str) -> NoReturn:
    raise ValueError(f"non-integer protocol number is forbidden: {value}")


def _send_packet(channel: socket.socket, packet: bytes) -> None:
    _prepare_channel(channel)
    if type(packet) is not bytes or not 0 < len(packet) <= _MAX_PACKET_BYTES:
        raise Experiment002CoordinatorError("outbound protocol packet is invalid")
    ancillary = [
        (
            socket.SOL_SOCKET,
            socket.SCM_CREDENTIALS,
            _CREDENTIALS.pack(os.getpid(), os.getuid(), os.getgid()),
        )
    ]
    try:
        sent = channel.sendmsg([packet], ancillary)
    except (OSError, TimeoutError) as error:
        raise Experiment002CoordinatorError("protocol packet send failed") from error
    if sent != len(packet):
        raise Experiment002CoordinatorError("protocol packet send was incomplete")


def _receive_packet(channel: socket.socket) -> tuple[bytes, _PeerCredentials]:
    _prepare_channel(channel)
    ancillary_size = socket.CMSG_SPACE(_CREDENTIALS.size)
    try:
        packet, ancillary, flags, _address = channel.recvmsg(
            _MAX_PACKET_BYTES + 1,
            ancillary_size,
        )
    except (OSError, TimeoutError) as error:
        raise Experiment002CoordinatorError("protocol packet receive failed") from error
    if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
        raise Experiment002CoordinatorError(
            "protocol packet or credentials were truncated"
        )
    if not packet:
        raise Experiment002CoordinatorError(
            "protocol peer closed before sending a packet"
        )
    if len(packet) > _MAX_PACKET_BYTES:
        raise Experiment002CoordinatorError("protocol packet exceeds its maximum size")
    credentials_payloads = [
        payload
        for level, kind, payload in ancillary
        if level == socket.SOL_SOCKET and kind == socket.SCM_CREDENTIALS
    ]
    if len(credentials_payloads) != 1:
        raise Experiment002CoordinatorError(
            "protocol packet lacks exactly one SCM_CREDENTIALS record"
        )
    payload = credentials_payloads[0]
    if len(payload) != _CREDENTIALS.size:
        raise Experiment002CoordinatorError("SCM_CREDENTIALS has an invalid size")
    pid, uid, gid = _CREDENTIALS.unpack(payload)
    credentials = _PeerCredentials(pid=pid, uid=uid, gid=gid)
    _require_local_identity(credentials)
    return bytes(packet), credentials


def _prepare_channel(channel: socket.socket) -> None:
    if type(channel) is not socket.socket:
        raise TypeError("protocol channel must be an exact socket")
    if not hasattr(socket, "SCM_CREDENTIALS") or not hasattr(socket, "SO_PASSCRED"):
        raise Experiment002CoordinatorError("Linux SCM_CREDENTIALS support is required")
    if channel.family != socket.AF_UNIX:
        raise Experiment002CoordinatorError("protocol channel must use AF_UNIX")
    try:
        socket_type = channel.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE)
        channel.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
        channel.settimeout(_PROTOCOL_TIMEOUT_SECONDS)
    except OSError as error:
        raise Experiment002CoordinatorError("protocol channel setup failed") from error
    if socket_type != socket.SOCK_SEQPACKET:
        raise Experiment002CoordinatorError(
            "protocol channel must use Unix SOCK_SEQPACKET"
        )


def _require_local_identity(credentials: _PeerCredentials) -> None:
    if type(credentials) is not _PeerCredentials:
        raise Experiment002CoordinatorError("peer credentials have an invalid type")
    if (
        type(credentials.pid) is not int
        or credentials.pid < 1
        or credentials.uid != os.getuid()
        or credentials.gid != os.getgid()
    ):
        raise Experiment002CoordinatorError("protocol sender identity is invalid")


def _parent_activate_child(
    channel: socket.socket,
    registration: VerifiedRunRegistration,
    *,
    role: str,
    seed: int,
    child_pid: int,
) -> None:
    """Send one activation and authenticate the matching child acknowledgement."""

    binding = _registration_binding(registration)
    assignment = _assignment(role, seed)
    parent_pid = os.getpid()
    _require_distinct_processes(parent_pid, child_pid)
    nonce = secrets.token_hex(32)
    ticket_document = _ticket_document(
        binding,
        assignment,
        parent_pid=parent_pid,
        child_pid=child_pid,
        nonce=nonce,
    )
    packet = _encode_packet(ticket_document)
    packet_sha256 = _ticket_sha256(packet)
    _reserve_issued_ticket(nonce, packet_sha256)
    _send_packet(channel, packet)
    ack_packet, credentials = _receive_packet(channel)
    ack = _decode_packet(ack_packet)
    _validate_ack(
        ack,
        credentials,
        assignment=assignment,
        parent_pid=parent_pid,
        child_pid=child_pid,
        nonce=nonce,
        packet_sha256=packet_sha256,
    )
    _require_registration(registration)


def _receive_child_activation(
    channel: socket.socket,
    registration: VerifiedRunRegistration,
) -> VerifiedChildActivation:
    """Accept one parent ticket using a child-local run registration."""

    binding = _registration_binding(registration)
    packet, credentials = _receive_packet(channel)
    document = _decode_packet(packet)
    ticket = _validate_ticket(
        document,
        credentials,
        binding=binding,
        packet=packet,
    )
    receipt = _reserve_accepted_ticket(ticket)
    _require_registration(registration)
    activation = _issue_activation(registration, ticket, receipt)
    try:
        _send_packet(channel, _encode_packet(_ack_document(ticket)))
    except BaseException:
        _poison_activation(activation)
        raise
    return activation


def _receive_activation_from_fixed_fd(
    registration: VerifiedRunRegistration,
) -> VerifiedChildActivation:
    if type(registration) is not VerifiedRunRegistration:
        raise TypeError("registration must be an exact VerifiedRunRegistration")
    try:
        channel = socket.socket(fileno=_CHILD_CONTROL_FD)
    except OSError as error:
        raise Experiment002CoordinatorError(
            "registered child control descriptor 3 is unavailable"
        ) from error
    with channel:
        if channel.fileno() != _CHILD_CONTROL_FD:
            raise Experiment002CoordinatorError(
                "registered child control descriptor changed"
            )
        try:
            os.set_inheritable(_CHILD_CONTROL_FD, False)
        except OSError as error:
            raise Experiment002CoordinatorError(
                "child control descriptor could not be sealed"
            ) from error
        return _receive_child_activation(channel, registration)


def _run_registered_seed_child(registration: VerifiedRunRegistration, /) -> None:
    """Activate descriptor 3 and dispatch the future registered seed worker."""

    _require_registration(registration)
    activation = _receive_activation_from_fixed_fd(registration)
    _dispatch_registered_seed_process(registration, activation)


def _validate_ticket(
    document: dict[str, Any],
    credentials: _PeerCredentials,
    *,
    binding: _RegistrationBinding,
    packet: bytes,
) -> _Ticket:
    _require_exact_keys(
        document,
        {
            "child_pid",
            "kind",
            "nonce",
            "ordinal",
            "parent_pid",
            "registration",
            "role",
            "schema_version",
            "seed",
        },
        "activation ticket",
    )
    if document["kind"] != "activation_ticket" or type(document["kind"]) is not str:
        raise Experiment002CoordinatorError("activation ticket kind is invalid")
    _require_schema_version(document["schema_version"])
    parent_pid = _require_pid(document["parent_pid"], "ticket parent PID")
    child_pid = _require_pid(document["child_pid"], "ticket child PID")
    _require_distinct_processes(parent_pid, child_pid)
    if (
        credentials.pid != parent_pid
        or parent_pid != os.getppid()
        or child_pid != os.getpid()
    ):
        raise Experiment002CoordinatorError(
            "activation ticket sender or process binding is invalid"
        )
    assignment = _assignment(document["role"], document["seed"])
    if (
        type(document["ordinal"]) is not int
        or document["ordinal"] != assignment.ordinal
    ):
        raise Experiment002CoordinatorError("activation ticket ordinal is invalid")
    nonce = document["nonce"]
    _require_lower_hex(nonce, 64, "activation nonce")
    if _parse_binding(document["registration"]) != binding:
        raise Experiment002CoordinatorError(
            "activation ticket registration binding does not match this child"
        )
    return _Ticket(
        assignment=assignment,
        parent_pid=parent_pid,
        child_pid=child_pid,
        nonce=cast(str, nonce),
        packet_sha256=_ticket_sha256(packet),
    )


def _validate_ack(
    document: dict[str, Any],
    credentials: _PeerCredentials,
    *,
    assignment: _Assignment,
    parent_pid: int,
    child_pid: int,
    nonce: str,
    packet_sha256: str,
) -> None:
    _require_exact_keys(
        document,
        {
            "child_pid",
            "kind",
            "nonce",
            "ordinal",
            "parent_pid",
            "role",
            "schema_version",
            "seed",
            "status",
            "ticket_sha256",
        },
        "activation acknowledgement",
    )
    _require_schema_version(document["schema_version"])
    observed_parent_pid = _require_pid(document["parent_pid"], "ack parent PID")
    observed_child_pid = _require_pid(document["child_pid"], "ack child PID")
    observed_assignment = _assignment(document["role"], document["seed"])
    if type(document["ordinal"]) is not int:
        raise TypeError("ack ordinal must be an integer")
    if type(document["kind"]) is not str or type(document["status"]) is not str:
        raise Experiment002CoordinatorError(
            "activation acknowledgement text is invalid"
        )
    _require_lower_hex(document["nonce"], 64, "ack nonce")
    _require_lower_hex(document["ticket_sha256"], 64, "ack ticket sha256")
    expected = {
        "child_pid": child_pid,
        "kind": "activation_ack",
        "nonce": nonce,
        "ordinal": assignment.ordinal,
        "parent_pid": parent_pid,
        "role": assignment.role,
        "schema_version": _SCHEMA_VERSION,
        "seed": assignment.seed,
        "status": "accepted",
        "ticket_sha256": packet_sha256,
    }
    if (
        document != expected
        or observed_parent_pid != parent_pid
        or observed_child_pid != child_pid
        or observed_assignment != assignment
        or credentials.pid != child_pid
    ):
        raise Experiment002CoordinatorError(
            "activation acknowledgement does not match its ticket or child PID"
        )


def _parse_binding(value: object) -> _RegistrationBinding:
    if type(value) is not dict:
        raise Experiment002CoordinatorError("ticket registration must be an object")
    document = cast(dict[str, Any], value)
    _require_exact_keys(
        document,
        {
            "head_commit",
            "implementation_commit",
            "registration_sha256",
            "source_bundle_sha256",
        },
        "ticket registration",
    )
    binding = _RegistrationBinding(
        head_commit=document["head_commit"],
        implementation_commit=document["implementation_commit"],
        registration_sha256=document["registration_sha256"],
        source_bundle_sha256=document["source_bundle_sha256"],
    )
    _require_registration_binding(binding)
    return binding


def _require_exact_keys(
    document: dict[str, Any], expected: set[str], name: str
) -> None:
    if type(document) is not dict or set(document) != expected:
        raise Experiment002CoordinatorError(f"{name} fields are not exact")
    if any(type(key) is not str for key in document):
        raise Experiment002CoordinatorError(f"{name} keys must be strings")


def _require_schema_version(value: object) -> None:
    if type(value) is not int or value != _SCHEMA_VERSION:
        raise Experiment002CoordinatorError("protocol schema version is invalid")


def _require_pid(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise Experiment002CoordinatorError(f"{name} must be positive")
    return value


def _require_distinct_processes(parent_pid: object, child_pid: object) -> None:
    parent = _require_pid(parent_pid, "parent PID")
    child = _require_pid(child_pid, "child PID")
    if parent == child:
        raise Experiment002CoordinatorError("parent and child PIDs must be distinct")


def _require_lower_hex(value: object, length: int, name: str) -> None:
    if (
        type(value) is not str
        or len(value) != length
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise Experiment002CoordinatorError(f"{name} must be {length} lowercase hex")


def _require_assignment(assignment: _Assignment) -> None:
    if type(assignment) is not _Assignment:
        raise Experiment002CoordinatorError("child assignment has an invalid type")
    if _assignment(assignment.role, assignment.seed) != assignment:
        raise Experiment002CoordinatorError("child assignment is not registered")


def _require_ticket(ticket: _Ticket) -> None:
    if type(ticket) is not _Ticket:
        raise Experiment002CoordinatorError("activation ticket has an invalid type")
    _require_assignment(ticket.assignment)
    _require_distinct_processes(ticket.parent_pid, ticket.child_pid)
    _require_lower_hex(ticket.nonce, 64, "activation nonce")
    _require_lower_hex(ticket.packet_sha256, 64, "activation ticket sha256")


def _reserve_issued_ticket(
    nonce: str,
    packet_sha256: str,
    *,
    _trusted_reserve: Callable[[str, str], None] = _TRUSTED_RESERVE_ISSUED,
) -> None:
    _require_lower_hex(nonce, 64, "activation nonce")
    _require_lower_hex(packet_sha256, 64, "activation ticket sha256")
    with _PROTOCOL_LOCK:
        _trusted_reserve(nonce, packet_sha256)
        if nonce in _ISSUED_TICKET_NONCES or packet_sha256 in _ISSUED_TICKET_DIGESTS:
            raise Experiment002CoordinatorError("activation ticket was already issued")
        _ISSUED_TICKET_NONCES.add(nonce)
        _ISSUED_TICKET_DIGESTS.add(packet_sha256)


def _reserve_accepted_ticket(
    ticket: _Ticket,
    *,
    _trusted_accept: Callable[[_Ticket], _AcceptedTicketReceipt] = (
        _TRUSTED_ACCEPT_TICKET
    ),
) -> _AcceptedTicketReceipt:
    _require_ticket(ticket)
    with _PROTOCOL_LOCK:
        receipt = _trusted_accept(ticket)
        if (
            ticket.packet_sha256 in _ACCEPTED_TICKET_DIGESTS
            or ticket.nonce in _ACCEPTED_TICKET_NONCES
        ):
            raise Experiment002CoordinatorError("activation ticket was replayed")
        _ACCEPTED_TICKET_DIGESTS.add(ticket.packet_sha256)
        _ACCEPTED_TICKET_NONCES.add(ticket.nonce)
        return receipt


def _ticket_sha256(packet: bytes) -> str:
    if type(packet) is not bytes:
        raise TypeError("ticket packet must be exact bytes")
    return hashlib.sha256(_TICKET_DIGEST_DOMAIN + packet).hexdigest()


def _issue_activation(
    registration: VerifiedRunRegistration,
    ticket: _Ticket,
    receipt: _AcceptedTicketReceipt,
    *,
    _trusted_record: Callable[
        [
            _AcceptedTicketReceipt,
            _Ticket,
            VerifiedChildActivation,
            VerifiedRunRegistration,
            object,
        ],
        None,
    ] = _TRUSTED_RECORD_ISSUANCE,
    _trusted_poison: Callable[[VerifiedChildActivation], None] = (
        _TRUSTED_POISON_ACTIVATION
    ),
) -> VerifiedChildActivation:
    _require_ticket(ticket)
    if ticket.child_pid != os.getpid() or ticket.parent_pid != os.getppid():
        raise Experiment002CoordinatorError(
            "activation cannot be issued outside its ticketed child"
        )
    token = object()
    state = _ActivationState(
        marker=_ACTIVATION_MARKER,
        registration=registration,
        process_id=os.getpid(),
        parent_pid=ticket.parent_pid,
        child_pid=ticket.child_pid,
        role=ticket.assignment.role,
        seed=ticket.assignment.seed,
        ordinal=ticket.assignment.ordinal,
        nonce=ticket.nonce,
        ticket_sha256=ticket.packet_sha256,
        token=token,
        phase=_PHASE_ISSUED,
    )
    activation = object.__new__(VerifiedChildActivation)
    with _PROTOCOL_LOCK:
        try:
            _trusted_record(
                receipt,
                ticket,
                activation,
                registration,
                token,
            )
            _ACTIVATIONS[activation] = state
            _ACTIVATION_GUARDS[activation] = _guard_from_state(state)
            _ACTIVATION_ANCHORS[activation] = _anchor_from_state(state)
            _ACTIVATION_PHASES[activation] = _PHASE_ISSUED
        except BaseException:
            _trusted_poison(activation)
            _FAILED_ACTIVATIONS.add(activation)
            raise
    return activation


def _activation_state(
    activation: VerifiedChildActivation,
    *,
    _trusted_verify: Callable[
        [VerifiedChildActivation, object],
        tuple[int | None, bool, bool],
    ] = _TRUSTED_VERIFY_ACTIVATION,
    _trusted_poison: Callable[[VerifiedChildActivation], None] = (
        _TRUSTED_POISON_ACTIVATION
    ),
) -> _ActivationState:
    if type(activation) is not VerifiedChildActivation:
        raise TypeError("activation must be an exact VerifiedChildActivation")
    with _PROTOCOL_LOCK:
        state = _ACTIVATIONS.get(activation)
        guard = _ACTIVATION_GUARDS.get(activation)
        anchor = _ACTIVATION_ANCHORS.get(activation)
        phase = _ACTIVATION_PHASES.get(activation)
        was_dispatched = activation in _DISPATCHED_ACTIVATIONS
        was_completed = activation in _COMPLETED_ACTIVATIONS
        trusted_phase, trusted_failed, trusted_valid = _trusted_verify(
            activation,
            state,
        )
        if (
            state is None
            or guard is None
            or anchor is None
            or phase is None
            or activation in _FAILED_ACTIVATIONS
            or trusted_phase is None
            or trusted_failed
            or not trusted_valid
            or not _valid_activation_state(state)
            or guard != _guard_from_state(state)
            or anchor != _anchor_from_state(state)
            or phase != state.phase
            or trusted_phase != state.phase
            or not _monotonic_phase_matches(
                state.phase,
                was_dispatched=was_dispatched,
                was_completed=was_completed,
            )
            or state.phase not in {_PHASE_ISSUED, _PHASE_DISPATCHED}
        ):
            _trusted_poison(activation)
            _FAILED_ACTIVATIONS.add(activation)
            raise Experiment002CoordinatorError(
                "child activation is forged, stale, consumed, or corrupted"
            )
        return state


def _monotonic_phase_matches(
    phase: int,
    *,
    was_dispatched: bool,
    was_completed: bool,
) -> bool:
    if type(phase) is not int:
        return False
    if phase == _PHASE_ISSUED:
        return not was_dispatched and not was_completed
    if phase == _PHASE_DISPATCHED:
        return was_dispatched and not was_completed
    if phase == _PHASE_COMPLETED:
        return was_dispatched and was_completed
    return False


def _valid_activation_state(state: _ActivationState) -> bool:
    if type(state) is not _ActivationState:
        return False
    try:
        _require_pid(state.process_id, "activation process PID")
        _require_distinct_processes(state.parent_pid, state.child_pid)
        _require_assignment(_Assignment(state.role, state.seed, state.ordinal))
        _require_lower_hex(state.nonce, 64, "activation nonce")
        _require_lower_hex(state.ticket_sha256, 64, "activation ticket sha256")
    except (TypeError, Experiment002CoordinatorError):
        return False
    return (
        state.marker is _ACTIVATION_MARKER
        and type(state.registration) is VerifiedRunRegistration
        and state.process_id == os.getpid()
        and state.child_pid == os.getpid()
        and state.parent_pid == os.getppid()
        and type(state.token) is object
        and type(state.phase) is int
        and state.phase in {_PHASE_ISSUED, _PHASE_DISPATCHED, _PHASE_COMPLETED}
    )


def _guard_from_state(state: _ActivationState) -> _ActivationGuard:
    return _ActivationGuard(
        marker=state.marker,
        registration=state.registration,
        process_id=state.process_id,
        parent_pid=state.parent_pid,
        child_pid=state.child_pid,
        role=state.role,
        seed=state.seed,
        ordinal=state.ordinal,
        nonce=state.nonce,
        ticket_sha256=state.ticket_sha256,
        token=state.token,
        phase=state.phase,
    )


def _anchor_from_state(state: _ActivationState) -> _ActivationAnchor:
    return _ActivationAnchor(
        marker=state.marker,
        registration=state.registration,
        process_id=state.process_id,
        parent_pid=state.parent_pid,
        child_pid=state.child_pid,
        role=state.role,
        seed=state.seed,
        ordinal=state.ordinal,
        nonce=state.nonce,
        ticket_sha256=state.ticket_sha256,
        token=state.token,
    )


def _poison_activation(
    activation: VerifiedChildActivation,
    *,
    _trusted_poison: Callable[[VerifiedChildActivation], None] = (
        _TRUSTED_POISON_ACTIVATION
    ),
) -> None:
    if type(activation) is VerifiedChildActivation:
        with _PROTOCOL_LOCK:
            _trusted_poison(activation)
            _FAILED_ACTIVATIONS.add(activation)


def _transition_activation(
    activation: VerifiedChildActivation,
    *,
    expected_phase: int,
    next_phase: int,
    _trusted_advance: Callable[[VerifiedChildActivation, int, int], None] = (
        _TRUSTED_ADVANCE_ACTIVATION
    ),
    _trusted_poison: Callable[[VerifiedChildActivation], None] = (
        _TRUSTED_POISON_ACTIVATION
    ),
) -> _ActivationState:
    with _PROTOCOL_LOCK:
        state = _activation_state(activation)
        if (
            state.phase != expected_phase
            or _ACTIVATION_PHASES.get(activation) != expected_phase
        ):
            _trusted_poison(activation)
            _FAILED_ACTIVATIONS.add(activation)
            raise Experiment002CoordinatorError(
                "child activation transition was replayed"
            )
        if expected_phase == _PHASE_ISSUED and next_phase == _PHASE_DISPATCHED:
            if (
                activation in _DISPATCHED_ACTIVATIONS
                or activation in _COMPLETED_ACTIVATIONS
            ):
                _trusted_poison(activation)
                _FAILED_ACTIVATIONS.add(activation)
                raise Experiment002CoordinatorError(
                    "child activation dispatch was already attempted"
                )
            pass
        elif expected_phase == _PHASE_DISPATCHED and next_phase == _PHASE_COMPLETED:
            if (
                activation not in _DISPATCHED_ACTIVATIONS
                or activation in _COMPLETED_ACTIVATIONS
            ):
                _trusted_poison(activation)
                _FAILED_ACTIVATIONS.add(activation)
                raise Experiment002CoordinatorError(
                    "child activation completion was already attempted"
                )
            pass
        else:
            _trusted_poison(activation)
            _FAILED_ACTIVATIONS.add(activation)
            raise Experiment002CoordinatorError(
                "child activation transition is not registered"
            )
        try:
            # Advance closure-owned truth first.  Any partial later failure is
            # terminal because this monotonic fact cannot be rolled back by
            # replacing or clearing the mutable module registries.
            _trusted_advance(activation, expected_phase, next_phase)
            if next_phase == _PHASE_DISPATCHED:
                _DISPATCHED_ACTIVATIONS.add(activation)
            else:
                _COMPLETED_ACTIVATIONS.add(activation)
            updated = replace(state, phase=next_phase)
            _ACTIVATIONS[activation] = updated
            _ACTIVATION_GUARDS[activation] = _guard_from_state(updated)
            _ACTIVATION_PHASES[activation] = next_phase
            return updated
        except BaseException:
            _trusted_poison(activation)
            _FAILED_ACTIVATIONS.add(activation)
            raise


def _dispatch_registered_seed_process(
    registration: VerifiedRunRegistration,
    activation: VerifiedChildActivation,
) -> None:
    verify_verified_child_activation(registration, activation)
    _transition_activation(
        activation,
        expected_phase=_PHASE_ISSUED,
        next_phase=_PHASE_DISPATCHED,
    )
    try:
        worker = cast(_SeedWorker, importlib.import_module(_WORKER_MODULE))
        entrypoint = getattr(worker, _WORKER_FUNCTION, None)
        if not callable(entrypoint):
            raise Experiment002CoordinatorError(
                "registered seed worker entrypoint is not callable"
            )
        result = entrypoint(registration, activation)
        if result is not None:
            raise Experiment002CoordinatorError(
                "registered seed worker returned an unexpected value"
            )
        _transition_activation(
            activation,
            expected_phase=_PHASE_DISPATCHED,
            next_phase=_PHASE_COMPLETED,
        )
    except BaseException:
        _poison_activation(activation)
        raise


# Critical paths captured these closure callables in their defaults.  Remove
# the temporary module aliases so mutable module state cannot expose, replace,
# or invoke the closure-owned authority ledger directly.
del _TRUSTED_RESERVE_ISSUED
del _TRUSTED_ACCEPT_TICKET
del _TRUSTED_RECORD_ISSUANCE
del _TRUSTED_VERIFY_ACTIVATION
del _TRUSTED_ADVANCE_ACTIVATION
del _TRUSTED_POISON_ACTIVATION
