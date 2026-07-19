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

import errno
import fcntl
import hashlib
import importlib
import json
import math
import os
import secrets
import socket
import stat
import struct
import threading
import weakref
from collections.abc import Callable
from dataclasses import dataclass, replace
from fractions import Fraction
from types import FunctionType, MemberDescriptorType, ModuleType
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
_PROCESS_GUARD_MODULE: Final = "falsewake.experiment_002_process_guard"
_PROCESS_GUARD_GETTER: Final = "get_registered_child_process_guard"
_PROCESS_GUARD_VERIFIER: Final = "verify_verified_child_process_guard"
_SUPERVISOR_MODULE: Final = "falsewake.experiment_002_supervisor"
_SUPERVISE_CHILD_FUNCTION: Final = "_supervise_child"
_SUPERVISED_CHILD_RESULT_TYPE: Final = "_SupervisedChildResult"
_CHILD_RESULT_MODULE: Final = "falsewake.experiment_002_child_result"
_CHILD_RESULT_BINDING_TYPE: Final = "ChildResultBinding"
_VERIFIED_CHILD_RESULT_TYPE: Final = "VerifiedChildResult"
_CHILD_RESULT_LOADER: Final = "load_registered_child_result"
_CHILD_RESULT_VERIFIER: Final = "verify_verified_child_result"
_CHILD_RESULT_LOADER_FUNCTION: Final = "load_route"
_CHILD_RESULT_VERIFIER_FUNCTION: Final = "verify_route"
_SCHEMA_VERSION: Final = 1
_FRAME_MAGIC: Final = b"FW2ACTV1"
_FRAME_HEADER: Final = struct.Struct(">8sI")
_CREDENTIALS: Final = struct.Struct("=3i")
_RIGHT_DESCRIPTOR: Final = struct.Struct("=i")
_MAX_RIGHT_DESCRIPTORS: Final = 253
_RECEIVE_CLOEXEC: Final = int(socket.MSG_CMSG_CLOEXEC)
_ANCILLARY_BYTES: Final = socket.CMSG_SPACE(_CREDENTIALS.size) + socket.CMSG_SPACE(
    _RIGHT_DESCRIPTOR.size * _MAX_RIGHT_DESCRIPTORS
)
_MAX_JSON_BYTES: Final = 3_072
_MAX_PACKET_BYTES: Final = _FRAME_HEADER.size + _MAX_JSON_BYTES
_PROTOCOL_TIMEOUT_SECONDS: Final = 30.0
_CONTROL_STATUS_FLAGS: Final = os.O_RDWR | os.O_NONBLOCK
_TICKET_DIGEST_DOMAIN: Final = b"falsewake-exp002-activation-ticket-v1\0"
_LOWER_HEX: Final = frozenset("0123456789abcdef")
_ACTIVATION_MARKER: Final = object()
_PHASE_ISSUED: Final = 1
_PHASE_DISPATCHED: Final = 2
_PHASE_COMPLETED: Final = 3

type _ChildResultBindingFrame = tuple[str, int, int, str, str, str, str]
type _RegisteredAssignmentFrame = tuple[str, int, int]
type _RegistrationBindingFrame = tuple[str, str, str, str]
type _VerifiedChildResultFrame = tuple[
    str,
    int,
    int,
    str,
    str,
    str,
    str,
    int,
    int,
    int,
    str,
    str,
]


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
class _ReceivedPacket:
    packet: bytes
    credentials: _PeerCredentials
    address: object


@dataclass(frozen=True, slots=True)
class _ControlChannelFrame:
    channel: socket.socket
    descriptor: int
    stat_frame: tuple[int, ...]
    descriptor_flags: int
    status_flags: int
    socket_type: int
    passcred: int
    timeout: float
    local_name: str
    peer_name: str
    proc_target: str


class _ProcessGuardVerifier(Protocol):
    def __call__(self, guard: object, /) -> None:
        """Reverify the exact imported child process guard."""


@dataclass(frozen=True, slots=True)
class _ProcessGuardBinding:
    guard: object
    verifier: _ProcessGuardVerifier


class _VerifiedChildResultView(Protocol):
    @property
    def role(self) -> str: ...

    @property
    def ordinal(self) -> int: ...

    @property
    def seed(self) -> int: ...

    @property
    def registration_head_commit(self) -> str: ...

    @property
    def implementation_commit(self) -> str: ...

    @property
    def registration_sha256(self) -> str: ...

    @property
    def source_bundle_sha256(self) -> str: ...

    @property
    def winner_epoch(self) -> int: ...

    @property
    def winner_macro_f1(self) -> Fraction: ...

    @property
    def winner_validation_cross_entropy(self) -> float: ...

    @property
    def model_tensor_sha256(self) -> str: ...


type _ParentExecutionRoutes = tuple[
    ModuleType,
    ModuleType,
    FunctionType,
    type[object],
    type[object],
    type[object],
    FunctionType,
    FunctionType,
]
type _RegisteredParentChildResult = tuple[
    int,
    tuple[int, int],
    int,
    int,
    int,
    _VerifiedChildResultView,
]


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
    process_guard: object
    process_guard_verifier: _ProcessGuardVerifier
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
    process_guard: object
    process_guard_verifier: _ProcessGuardVerifier
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
    process_guard: object
    process_guard_verifier: _ProcessGuardVerifier
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
    ticket_frame: tuple[str, int, int, int, int, str, str]
    receipt: _AcceptedTicketReceipt
    receipt_token: object
    activation_token: object
    process_guard: object
    process_guard_verifier: _ProcessGuardVerifier
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
            object,
            _ProcessGuardVerifier,
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
        _AcceptedTicketReceipt,
        tuple[_Ticket, object, tuple[str, int, int, int, int, str, str]],
    ] = weakref.WeakKeyDictionary()
    consumed_receipts: weakref.WeakSet[_AcceptedTicketReceipt] = weakref.WeakSet()
    issuance_anchors: weakref.WeakKeyDictionary[
        VerifiedChildActivation, _TrustedActivationAnchor
    ] = weakref.WeakKeyDictionary()
    dispatched: weakref.WeakSet[VerifiedChildActivation] = weakref.WeakSet()
    completed: weakref.WeakSet[VerifiedChildActivation] = weakref.WeakSet()
    failed: weakref.WeakSet[VerifiedChildActivation] = weakref.WeakSet()
    ledger_lock = threading.RLock()

    def ticket_frame(
        ticket: _Ticket,
    ) -> tuple[str, int, int, int, int, str, str]:
        _require_ticket(ticket)
        return (
            ticket.assignment.role,
            ticket.assignment.seed,
            ticket.assignment.ordinal,
            ticket.parent_pid,
            ticket.child_pid,
            ticket.nonce,
            ticket.packet_sha256,
        )

    def exact_ticket_frames_match(
        first: object,
        second: object,
    ) -> bool:
        if (
            type(first) is not tuple
            or type(second) is not tuple
            or len(first) != 7
            or len(second) != 7
            or type(first[0]) is not str
            or type(second[0]) is not str
            or any(type(first[index]) is not int for index in range(1, 5))
            or any(type(second[index]) is not int for index in range(1, 5))
            or type(first[5]) is not str
            or type(second[5]) is not str
            or type(first[6]) is not str
            or type(second[6]) is not str
        ):
            return False
        return all(first[index] == second[index] for index in range(7))

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
            accepted_frame = ticket_frame(ticket)
            accepted_nonce = accepted_frame[5]
            accepted_digest = accepted_frame[6]
            if accepted_nonce in accepted_nonces or accepted_digest in accepted_digests:
                raise Experiment002CoordinatorError("activation ticket was replayed")
            accepted_nonces.add(accepted_nonce)
            accepted_digests.add(accepted_digest)
            receipt = object.__new__(_AcceptedTicketReceipt)
            receipt_records[receipt] = (ticket, object(), accepted_frame)
            return receipt

    def record_issuance(
        receipt: _AcceptedTicketReceipt,
        ticket: _Ticket,
        activation: VerifiedChildActivation,
        registration: VerifiedRunRegistration,
        activation_token: object,
        process_guard: object,
        process_guard_verifier: _ProcessGuardVerifier,
    ) -> None:
        with ledger_lock:
            if type(receipt) is not _AcceptedTicketReceipt:
                raise Experiment002CoordinatorError(
                    "activation lacks an exact accepted-ticket receipt"
                )
            receipt_record = receipt_records.get(receipt)
            try:
                current_frame = ticket_frame(ticket)
            except (TypeError, Experiment002CoordinatorError):
                current_frame = None
            if (
                receipt_record is None
                or receipt in consumed_receipts
                or receipt_record[0] is not ticket
                or not exact_ticket_frames_match(current_frame, receipt_record[2])
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
                ticket_frame=receipt_record[2],
                receipt=receipt,
                receipt_token=receipt_record[1],
                activation_token=activation_token,
                process_guard=process_guard,
                process_guard_verifier=process_guard_verifier,
                process_id=os.getpid(),
                marker=_ACTIVATION_MARKER,
            )

    def receipt_linked(
        activation: VerifiedChildActivation,
        anchor: _TrustedActivationAnchor,
        current_frame: object,
    ) -> bool:
        receipt_record = receipt_records.get(anchor.receipt)
        return (
            anchor.activation_ref() is activation
            and anchor.receipt in consumed_receipts
            and receipt_record is not None
            and receipt_record[0] is anchor.ticket
            and receipt_record[1] is anchor.receipt_token
            and exact_ticket_frames_match(receipt_record[2], anchor.ticket_frame)
            and exact_ticket_frames_match(current_frame, anchor.ticket_frame)
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
            phase = lifecycle_phase(activation)
            try:
                current_frame = ticket_frame(anchor.ticket)
            except (TypeError, Experiment002CoordinatorError):
                return phase, is_failed, False
            linked = receipt_linked(activation, anchor, current_frame)
            if type(state) is not _ActivationState:
                return phase, is_failed, False
            try:
                _require_ticket(anchor.ticket)
            except (TypeError, Experiment002CoordinatorError):
                return phase, is_failed, False
            trusted_ticket_frame = anchor.ticket_frame
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
                and anchor.process_guard is state.process_guard
                and anchor.process_guard_verifier is state.process_guard_verifier
                and exact_ticket_frames_match(
                    current_frame,
                    trusted_ticket_frame,
                )
                and trusted_ticket_frame[0] == state.role
                and trusted_ticket_frame[1] == state.seed
                and trusted_ticket_frame[2] == state.ordinal
                and trusted_ticket_frame[3] == state.parent_pid
                and trusted_ticket_frame[4] == state.child_pid
                and trusted_ticket_frame[5] == state.nonce
                and trusted_ticket_frame[6] == state.ticket_sha256
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
            try:
                current_frame = (
                    ticket_frame(anchor.ticket) if anchor is not None else None
                )
            except (TypeError, Experiment002CoordinatorError):
                current_frame = None
            if (
                anchor is None
                or activation in failed
                or not receipt_linked(activation, anchor, current_frame)
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
    try:
        if state.registration is not registration:
            raise Experiment002CoordinatorError(
                "child activation belongs to a different run registration"
            )
        verify_verified_run_registration(registration)
        _require_process_guard_binding(
            _ProcessGuardBinding(
                guard=state.process_guard,
                verifier=state.process_guard_verifier,
            )
        )
        final_state = _activation_state(activation)
        if final_state is not state:
            raise Experiment002CoordinatorError(
                "child activation state changed during verification"
            )
    except BaseException:
        _poison_activation(activation)
        raise


def _require_registration(registration: VerifiedRunRegistration) -> None:
    if type(registration) is not VerifiedRunRegistration:
        raise TypeError("registration must be an exact VerifiedRunRegistration")
    verify_verified_run_registration(registration)


def _require_process_guard_binding(binding: _ProcessGuardBinding) -> None:
    if (
        type(binding) is not _ProcessGuardBinding
        or binding.guard is None
        or type(binding.verifier) is not FunctionType
    ):
        raise Experiment002CoordinatorError(
            "child process-guard binding has an invalid exact type"
        )
    result = cast(Callable[[object], object], binding.verifier)(binding.guard)
    if result is not None:
        raise Experiment002CoordinatorError(
            "child process-guard verifier returned an unexpected value"
        )


def _claim_and_verify_child_process_guard() -> _ProcessGuardBinding:
    module = importlib.import_module(_PROCESS_GUARD_MODULE)
    if (
        type(module) is not ModuleType
        or type(module.__name__) is not str
        or module.__name__ != _PROCESS_GUARD_MODULE
    ):
        raise Experiment002CoordinatorError(
            "child process-guard module identity is invalid"
        )
    getter = getattr(module, _PROCESS_GUARD_GETTER, None)
    verifier = getattr(module, _PROCESS_GUARD_VERIFIER, None)
    if (
        type(getter) is not FunctionType
        or type(verifier) is not FunctionType
        or type(getter.__module__) is not str
        or getter.__module__ != _PROCESS_GUARD_MODULE
        or type(verifier.__module__) is not str
        or verifier.__module__ != _PROCESS_GUARD_MODULE
    ):
        raise Experiment002CoordinatorError(
            "child process-guard authority routes are invalid"
        )
    guard = cast(Callable[[], object], getter)()
    binding = _ProcessGuardBinding(
        guard=guard,
        verifier=cast(_ProcessGuardVerifier, verifier),
    )
    _require_process_guard_binding(binding)
    if (
        getattr(module, _PROCESS_GUARD_GETTER, None) is not getter
        or getattr(module, _PROCESS_GUARD_VERIFIER, None) is not verifier
    ):
        raise Experiment002CoordinatorError(
            "child process-guard authority routes changed during claim"
        )
    return binding


def _require_parent_module_identity(module: object, name: str) -> ModuleType:
    if (
        type(module) is not ModuleType
        or type(module.__name__) is not str
        or module.__name__ != name
    ):
        raise Experiment002CoordinatorError(
            "registered parent execution module identity is invalid"
        )
    return module


def _require_parent_function_identity(
    route: object,
    name: str,
    module_name: str,
) -> FunctionType:
    if (
        type(route) is not FunctionType
        or type(route.__name__) is not str
        or route.__name__ != name
        or type(route.__module__) is not str
        or route.__module__ != module_name
    ):
        raise Experiment002CoordinatorError(
            "registered parent execution function identity is invalid"
        )
    return route


def _require_parent_type_identity(
    value: object,
    name: str,
    module_name: str,
) -> type[object]:
    if (
        type(value) is not type
        or type(value.__name__) is not str
        or value.__name__ != name
        or type(value.__module__) is not str
        or value.__module__ != module_name
    ):
        raise Experiment002CoordinatorError(
            "registered parent execution type identity is invalid"
        )
    return value


def _require_parent_execution_routes_unchanged(
    routes: _ParentExecutionRoutes,
    /,
) -> None:
    if type(routes) is not tuple or len(routes) != 8:
        raise Experiment002CoordinatorError(
            "registered parent execution routes have an invalid exact type"
        )
    (
        claimed_supervisor_module,
        claimed_child_result_module,
        claimed_supervise_child,
        claimed_supervised_child_result_type,
        claimed_child_result_binding_type,
        claimed_verified_child_result_type,
        claimed_load_registered_child_result,
        claimed_verify_verified_child_result,
    ) = routes
    supervisor_module = _require_parent_module_identity(
        claimed_supervisor_module,
        _SUPERVISOR_MODULE,
    )
    child_result_module = _require_parent_module_identity(
        claimed_child_result_module,
        _CHILD_RESULT_MODULE,
    )
    supervise_child = _require_parent_function_identity(
        claimed_supervise_child,
        _SUPERVISE_CHILD_FUNCTION,
        _SUPERVISOR_MODULE,
    )
    supervised_child_result_type = _require_parent_type_identity(
        claimed_supervised_child_result_type,
        _SUPERVISED_CHILD_RESULT_TYPE,
        _SUPERVISOR_MODULE,
    )
    child_result_binding_type = _require_parent_type_identity(
        claimed_child_result_binding_type,
        _CHILD_RESULT_BINDING_TYPE,
        _CHILD_RESULT_MODULE,
    )
    verified_child_result_type = _require_parent_type_identity(
        claimed_verified_child_result_type,
        _VERIFIED_CHILD_RESULT_TYPE,
        _CHILD_RESULT_MODULE,
    )
    load_registered_child_result = _require_parent_function_identity(
        claimed_load_registered_child_result,
        _CHILD_RESULT_LOADER_FUNCTION,
        _CHILD_RESULT_MODULE,
    )
    verify_verified_child_result = _require_parent_function_identity(
        claimed_verify_verified_child_result,
        _CHILD_RESULT_VERIFIER_FUNCTION,
        _CHILD_RESULT_MODULE,
    )
    if (
        getattr(supervisor_module, _SUPERVISE_CHILD_FUNCTION, None)
        is not supervise_child
        or getattr(supervisor_module, _SUPERVISED_CHILD_RESULT_TYPE, None)
        is not supervised_child_result_type
        or getattr(child_result_module, _CHILD_RESULT_BINDING_TYPE, None)
        is not child_result_binding_type
        or getattr(child_result_module, _VERIFIED_CHILD_RESULT_TYPE, None)
        is not verified_child_result_type
        or getattr(child_result_module, _CHILD_RESULT_LOADER, None)
        is not load_registered_child_result
        or getattr(child_result_module, _CHILD_RESULT_VERIFIER, None)
        is not verify_verified_child_result
        or getattr(supervisor_module, _CHILD_RESULT_BINDING_TYPE, None)
        is not child_result_binding_type
        or getattr(supervisor_module, _VERIFIED_CHILD_RESULT_TYPE, None)
        is not verified_child_result_type
        or getattr(supervisor_module, _CHILD_RESULT_LOADER, None)
        is not load_registered_child_result
        or getattr(supervisor_module, _CHILD_RESULT_VERIFIER, None)
        is not verify_verified_child_result
    ):
        raise Experiment002CoordinatorError(
            "registered parent execution routes changed or disagree"
        )


def _claim_and_verify_parent_execution_routes() -> _ParentExecutionRoutes:
    supervisor_module = _require_parent_module_identity(
        importlib.import_module(_SUPERVISOR_MODULE),
        _SUPERVISOR_MODULE,
    )
    child_result_module = _require_parent_module_identity(
        importlib.import_module(_CHILD_RESULT_MODULE),
        _CHILD_RESULT_MODULE,
    )
    routes = (
        supervisor_module,
        child_result_module,
        _require_parent_function_identity(
            getattr(supervisor_module, _SUPERVISE_CHILD_FUNCTION, None),
            _SUPERVISE_CHILD_FUNCTION,
            _SUPERVISOR_MODULE,
        ),
        _require_parent_type_identity(
            getattr(supervisor_module, _SUPERVISED_CHILD_RESULT_TYPE, None),
            _SUPERVISED_CHILD_RESULT_TYPE,
            _SUPERVISOR_MODULE,
        ),
        _require_parent_type_identity(
            getattr(child_result_module, _CHILD_RESULT_BINDING_TYPE, None),
            _CHILD_RESULT_BINDING_TYPE,
            _CHILD_RESULT_MODULE,
        ),
        _require_parent_type_identity(
            getattr(child_result_module, _VERIFIED_CHILD_RESULT_TYPE, None),
            _VERIFIED_CHILD_RESULT_TYPE,
            _CHILD_RESULT_MODULE,
        ),
        _require_parent_function_identity(
            getattr(child_result_module, _CHILD_RESULT_LOADER, None),
            _CHILD_RESULT_LOADER_FUNCTION,
            _CHILD_RESULT_MODULE,
        ),
        _require_parent_function_identity(
            getattr(child_result_module, _CHILD_RESULT_VERIFIER, None),
            _CHILD_RESULT_VERIFIER_FUNCTION,
            _CHILD_RESULT_MODULE,
        ),
    )
    _require_parent_execution_routes_unchanged(routes)
    return routes


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


def _registration_bindings_match(
    first: _RegistrationBinding,
    second: _RegistrationBinding,
) -> bool:
    if (
        type(first) is not _RegistrationBinding
        or type(second) is not _RegistrationBinding
    ):
        return False
    try:
        _require_registration_binding(first)
        _require_registration_binding(second)
    except (TypeError, Experiment002CoordinatorError):
        return False
    return (
        first.head_commit == second.head_commit
        and first.implementation_commit == second.implementation_commit
        and first.registration_sha256 == second.registration_sha256
        and first.source_bundle_sha256 == second.source_bundle_sha256
    )


def _snapshot_registered_assignment(assignment: _Assignment, /) -> _Assignment:
    _require_assignment(assignment)
    first = (assignment.role, assignment.seed, assignment.ordinal)
    snapshot = _assignment(first[0], first[1])
    second = (assignment.role, assignment.seed, assignment.ordinal)
    if first != second or not _assignments_match(snapshot, assignment):
        raise Experiment002CoordinatorError(
            "child assignment changed while it was snapshotted"
        )
    return snapshot


def _snapshot_registered_cpu_ids(cpu_ids: tuple[int, int], /) -> tuple[int, int]:
    if (
        type(cpu_ids) is not tuple
        or len(cpu_ids) != 2
        or any(type(value) is not int or value < 0 for value in cpu_ids)
        or cpu_ids[0] >= cpu_ids[1]
    ):
        raise Experiment002CoordinatorError("captured CPU IDs are invalid")
    first = (cpu_ids[0], cpu_ids[1])
    second = (cpu_ids[0], cpu_ids[1])
    if first != second:
        raise Experiment002CoordinatorError(
            "captured CPU IDs changed while they were snapshotted"
        )
    return first


def _dynamic_child_result_binding_frame(
    binding: object,
    binding_type: type[object],
    /,
) -> _ChildResultBindingFrame:
    if type(binding) is not binding_type:
        raise Experiment002CoordinatorError(
            "registered child-result binding has an invalid exact type"
        )
    role = getattr(binding, "role", None)
    ordinal = getattr(binding, "ordinal", None)
    seed = getattr(binding, "seed", None)
    registration_head_commit = getattr(binding, "registration_head_commit", None)
    implementation_commit = getattr(binding, "implementation_commit", None)
    registration_sha256 = getattr(binding, "registration_sha256", None)
    source_bundle_sha256 = getattr(binding, "source_bundle_sha256", None)
    if (
        type(role) is not str
        or type(ordinal) is not int
        or type(seed) is not int
        or type(registration_head_commit) is not str
        or type(implementation_commit) is not str
        or type(registration_sha256) is not str
        or type(source_bundle_sha256) is not str
    ):
        raise Experiment002CoordinatorError(
            "registered child-result binding fields have invalid exact types"
        )
    return (
        role,
        ordinal,
        seed,
        registration_head_commit,
        implementation_commit,
        registration_sha256,
        source_bundle_sha256,
    )


def _build_registered_child_result_binding(
    routes: _ParentExecutionRoutes,
    assignment_frame: _RegisteredAssignmentFrame,
    registration_frame: _RegistrationBindingFrame,
    /,
) -> object:
    _require_parent_execution_routes_unchanged(routes)
    child_result_binding_type = routes[4]
    if (
        type(assignment_frame) is not tuple
        or len(assignment_frame) != 3
        or type(assignment_frame[0]) is not str
        or type(assignment_frame[1]) is not int
        or type(assignment_frame[2]) is not int
        or type(registration_frame) is not tuple
        or len(registration_frame) != 4
        or any(type(value) is not str for value in registration_frame)
    ):
        raise Experiment002CoordinatorError(
            "registered parent binding frames have invalid exact types"
        )
    expected = (
        assignment_frame[0],
        assignment_frame[2],
        assignment_frame[1],
        *registration_frame,
    )
    constructor = cast(Callable[..., object], child_result_binding_type)
    binding = constructor(
        role=expected[0],
        ordinal=expected[1],
        seed=expected[2],
        registration_head_commit=expected[3],
        implementation_commit=expected[4],
        registration_sha256=expected[5],
        source_bundle_sha256=expected[6],
    )
    first = _dynamic_child_result_binding_frame(
        binding,
        child_result_binding_type,
    )
    second = _dynamic_child_result_binding_frame(
        binding,
        child_result_binding_type,
    )
    if first != expected or second != first:
        raise Experiment002CoordinatorError(
            "registered child-result binding changed during construction"
        )
    _require_parent_execution_routes_unchanged(routes)
    return binding


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
    _prepare_send_channel(channel)
    _send_packet_prepared(channel, packet)


def _send_packet_prepared(channel: socket.socket, packet: bytes) -> None:
    _require_seqpacket_channel(channel)
    timeout = channel.gettimeout()
    if type(timeout) is not float or timeout != _PROTOCOL_TIMEOUT_SECONDS:
        raise Experiment002CoordinatorError(
            "outbound protocol channel timeout is not exact"
        )
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
    if type(sent) is not int or sent != len(packet):
        raise Experiment002CoordinatorError("protocol packet send was incomplete")


def _receive_packet_frame(
    channel: socket.socket,
    receive_flags: int,
) -> _ReceivedPacket:
    _require_prepared_receive_channel(channel)
    if type(receive_flags) is not int or receive_flags not in {
        0,
        int(socket.MSG_PEEK),
    }:
        raise Experiment002CoordinatorError(
            "protocol receive flags are not an exact fixed mode"
        )
    try:
        packet, ancillary, flags, address = channel.recvmsg(
            _MAX_PACKET_BYTES + 1,
            _ANCILLARY_BYTES,
            receive_flags | _RECEIVE_CLOEXEC,
        )
    except (OSError, TimeoutError) as error:
        raise Experiment002CoordinatorError("protocol packet receive failed") from error
    received_rights = _close_received_rights(ancillary)
    if (
        type(packet) is not bytes
        or type(ancillary) is not list
        or type(flags) is not int
    ):
        raise Experiment002CoordinatorError(
            "protocol receive result has invalid exact types"
        )
    if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
        raise Experiment002CoordinatorError(
            "protocol packet or credentials were truncated"
        )
    if flags != _RECEIVE_CLOEXEC:
        raise Experiment002CoordinatorError("protocol receive flags are invalid")
    if received_rights:
        raise Experiment002CoordinatorError(
            "protocol packet included forbidden SCM_RIGHTS descriptors"
        )
    if not packet:
        raise Experiment002CoordinatorError(
            "protocol peer closed before sending a packet"
        )
    if len(packet) > _MAX_PACKET_BYTES:
        raise Experiment002CoordinatorError("protocol packet exceeds its maximum size")
    if len(ancillary) != 1 or type(ancillary[0]) is not tuple:
        raise Experiment002CoordinatorError(
            "protocol packet lacks exactly one SCM_CREDENTIALS record"
        )
    record = ancillary[0]
    if len(record) != 3:
        raise Experiment002CoordinatorError(
            "protocol packet has a malformed credentials record"
        )
    level, kind, payload = record
    if (
        type(level) is not int
        or type(kind) is not int
        or type(payload) is not bytes
        or level != socket.SOL_SOCKET
        or kind != socket.SCM_CREDENTIALS
    ):
        raise Experiment002CoordinatorError(
            "protocol packet lacks exactly one SCM_CREDENTIALS record"
        )
    if len(payload) != _CREDENTIALS.size:
        raise Experiment002CoordinatorError("SCM_CREDENTIALS has an invalid size")
    pid, uid, gid = _CREDENTIALS.unpack(payload)
    credentials = _PeerCredentials(pid=pid, uid=uid, gid=gid)
    _require_local_identity(credentials)
    return _ReceivedPacket(
        packet=packet,
        credentials=credentials,
        address=address,
    )


def _receive_packet(channel: socket.socket) -> tuple[bytes, _PeerCredentials]:
    _prepare_channel(channel)
    received = _receive_packet_frame(channel, 0)
    return received.packet, received.credentials


def _peek_packet(channel: socket.socket) -> _ReceivedPacket:
    received = _receive_packet_frame(channel, int(socket.MSG_PEEK))
    _require_unnamed_received_packet(received)
    return received


def _consume_packet(channel: socket.socket) -> _ReceivedPacket:
    received = _receive_packet_frame(channel, 0)
    _require_unnamed_received_packet(received)
    return received


def _require_unnamed_received_packet(received: _ReceivedPacket) -> None:
    if type(received) is not _ReceivedPacket or received.address is not None:
        raise Experiment002CoordinatorError(
            "registered parent packet must have no Unix sender address"
        )


def _require_same_received_packet(
    peeked: _ReceivedPacket,
    consumed: _ReceivedPacket,
) -> None:
    if type(peeked) is not _ReceivedPacket or type(consumed) is not _ReceivedPacket:
        raise Experiment002CoordinatorError(
            "peeked protocol packet frames have invalid types"
        )
    _require_local_identity(peeked.credentials)
    _require_local_identity(consumed.credentials)
    if (
        type(peeked.packet) is not bytes
        or type(consumed.packet) is not bytes
        or peeked.packet != consumed.packet
        or peeked.address is not None
        or consumed.address is not None
        or peeked.credentials.pid != consumed.credentials.pid
        or peeked.credentials.uid != consumed.credentials.uid
        or peeked.credentials.gid != consumed.credentials.gid
    ):
        raise Experiment002CoordinatorError(
            "consumed protocol packet or credentials differ from MSG_PEEK"
        )


def _close_received_rights(ancillary: object) -> bool:
    """Close every delivered SCM_RIGHTS descriptor before rejecting its frame."""

    if type(ancillary) is not list:
        return False
    received_rights = False
    close_error: OSError | None = None
    for record in ancillary:
        if type(record) is not tuple or len(record) != 3:
            continue
        level, kind, payload = record
        if (
            type(level) is not int
            or type(kind) is not int
            or level != socket.SOL_SOCKET
            or kind != socket.SCM_RIGHTS
        ):
            continue
        received_rights = True
        if type(payload) is not bytes:
            continue
        complete_size = len(payload) - (len(payload) % _RIGHT_DESCRIPTOR.size)
        for offset in range(0, complete_size, _RIGHT_DESCRIPTOR.size):
            (descriptor,) = _RIGHT_DESCRIPTOR.unpack_from(payload, offset)
            try:
                os.close(descriptor)
            except OSError as error:
                if close_error is None:
                    close_error = error
    if close_error is not None:
        raise Experiment002CoordinatorError(
            "received SCM_RIGHTS descriptor could not be closed"
        ) from close_error
    return received_rights


def _require_seqpacket_channel(channel: socket.socket) -> None:
    if type(channel) is not socket.socket:
        raise TypeError("protocol channel must be an exact socket")
    if (
        not hasattr(socket, "SCM_CREDENTIALS")
        or not hasattr(socket, "SO_PASSCRED")
        or not hasattr(socket, "MSG_CMSG_CLOEXEC")
    ):
        raise Experiment002CoordinatorError("Linux SCM_CREDENTIALS support is required")
    if channel.family is not socket.AF_UNIX:
        raise Experiment002CoordinatorError("protocol channel must use AF_UNIX")
    try:
        socket_type = channel.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE)
    except OSError as error:
        raise Experiment002CoordinatorError(
            "protocol channel inspection failed"
        ) from error
    if type(socket_type) is not int or socket_type != socket.SOCK_SEQPACKET:
        raise Experiment002CoordinatorError(
            "protocol channel must use Unix SOCK_SEQPACKET"
        )


def _prepare_send_channel(channel: socket.socket) -> None:
    _require_seqpacket_channel(channel)
    try:
        channel.settimeout(_PROTOCOL_TIMEOUT_SECONDS)
    except OSError as error:
        raise Experiment002CoordinatorError("protocol channel setup failed") from error
    timeout = channel.gettimeout()
    if type(timeout) is not float or timeout != _PROTOCOL_TIMEOUT_SECONDS:
        raise Experiment002CoordinatorError(
            "outbound protocol channel timeout is not exact"
        )


def _prepare_channel(channel: socket.socket) -> None:
    _require_seqpacket_channel(channel)
    try:
        channel.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
        channel.settimeout(_PROTOCOL_TIMEOUT_SECONDS)
    except OSError as error:
        raise Experiment002CoordinatorError("protocol channel setup failed") from error
    _require_prepared_receive_channel(channel)


def _require_prepared_receive_channel(channel: socket.socket) -> None:
    _require_seqpacket_channel(channel)
    passcred = channel.getsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED)
    timeout = channel.gettimeout()
    if (
        type(passcred) is not int
        or passcred != 1
        or type(timeout) is not float
        or timeout != _PROTOCOL_TIMEOUT_SECONDS
    ):
        raise Experiment002CoordinatorError("protocol channel options are not exact")


def _socket_stat_frame(metadata: os.stat_result) -> tuple[int, ...]:
    if type(metadata) is not os.stat_result:
        raise Experiment002CoordinatorError(
            "registered child control descriptor stat type is invalid"
        )
    frame = (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_blocks,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )
    if any(type(value) is not int for value in frame):
        raise Experiment002CoordinatorError(
            "registered child control descriptor stat frame is invalid"
        )
    return frame


def _socket_proc_inode(proc_target: str) -> int:
    if (
        type(proc_target) is not str
        or not proc_target.startswith("socket:[")
        or not proc_target.endswith("]")
    ):
        raise Experiment002CoordinatorError(
            "registered child control descriptor proc target is invalid"
        )
    digits = proc_target[8:-1]
    if not digits or not digits.isascii() or not digits.isdecimal():
        raise Experiment002CoordinatorError(
            "registered child control descriptor proc inode is invalid"
        )
    return int(digits)


def _capture_fixed_control_channel_frame(
    channel: socket.socket,
) -> _ControlChannelFrame:
    _prepare_channel(channel)
    descriptor = channel.fileno()
    try:
        metadata = os.fstat(descriptor)
        descriptor_flags = fcntl.fcntl(descriptor, fcntl.F_GETFD)
        status_flags = fcntl.fcntl(descriptor, fcntl.F_GETFL)
        socket_type = channel.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE)
        passcred = channel.getsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED)
        timeout = channel.gettimeout()
        local_name = channel.getsockname()
        peer_name = channel.getpeername()
        proc_target = os.readlink(f"/proc/self/fd/{descriptor}")
        inheritable = os.get_inheritable(descriptor)
    except OSError as error:
        raise Experiment002CoordinatorError(
            "registered child control descriptor cannot be inspected"
        ) from error
    if (
        type(descriptor) is not int
        or descriptor != _CHILD_CONTROL_FD
        or not stat.S_ISSOCK(metadata.st_mode)
        or type(descriptor_flags) is not int
        or descriptor_flags != fcntl.FD_CLOEXEC
        or type(inheritable) is not bool
        or inheritable is not False
        or type(status_flags) is not int
        or status_flags != _CONTROL_STATUS_FLAGS
        or type(socket_type) is not int
        or socket_type != socket.SOCK_SEQPACKET
        or type(passcred) is not int
        or passcred != 1
        or type(timeout) is not float
        or timeout != _PROTOCOL_TIMEOUT_SECONDS
        or type(local_name) is not str
        or local_name != ""
        or type(peer_name) is not str
        or peer_name != ""
        or _socket_proc_inode(proc_target) != metadata.st_ino
    ):
        raise Experiment002CoordinatorError(
            "registered child control descriptor frame is invalid"
        )
    return _ControlChannelFrame(
        channel=channel,
        descriptor=descriptor,
        stat_frame=_socket_stat_frame(metadata),
        descriptor_flags=descriptor_flags,
        status_flags=status_flags,
        socket_type=socket_type,
        passcred=passcred,
        timeout=timeout,
        local_name=local_name,
        peer_name=peer_name,
        proc_target=proc_target,
    )


def _require_fixed_control_channel_frame(
    channel: socket.socket,
    frame: _ControlChannelFrame,
) -> None:
    if (
        type(channel) is not socket.socket
        or type(frame) is not _ControlChannelFrame
        or frame.channel is not channel
        or type(frame.descriptor) is not int
        or frame.descriptor != _CHILD_CONTROL_FD
    ):
        raise Experiment002CoordinatorError(
            "registered child control socket identity changed"
        )
    try:
        observed_descriptor = channel.fileno()
        metadata = os.fstat(frame.descriptor)
        descriptor_flags = fcntl.fcntl(frame.descriptor, fcntl.F_GETFD)
        status_flags = fcntl.fcntl(frame.descriptor, fcntl.F_GETFL)
        socket_type = channel.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE)
        passcred = channel.getsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED)
        timeout = channel.gettimeout()
        local_name = channel.getsockname()
        peer_name = channel.getpeername()
        proc_target = os.readlink(f"/proc/self/fd/{frame.descriptor}")
        inheritable = os.get_inheritable(frame.descriptor)
    except OSError as error:
        raise Experiment002CoordinatorError(
            "registered child control descriptor changed"
        ) from error
    if (
        type(observed_descriptor) is not int
        or observed_descriptor != frame.descriptor
        or type(frame.stat_frame) is not tuple
        or len(frame.stat_frame) != 10
        or any(type(value) is not int for value in frame.stat_frame)
        or _socket_stat_frame(metadata) != frame.stat_frame
        or type(frame.descriptor_flags) is not int
        or type(descriptor_flags) is not int
        or descriptor_flags != frame.descriptor_flags
        or descriptor_flags != fcntl.FD_CLOEXEC
        or type(inheritable) is not bool
        or inheritable is not False
        or type(frame.status_flags) is not int
        or type(status_flags) is not int
        or status_flags != frame.status_flags
        or status_flags != _CONTROL_STATUS_FLAGS
        or type(frame.socket_type) is not int
        or type(socket_type) is not int
        or socket_type != frame.socket_type
        or socket_type != socket.SOCK_SEQPACKET
        or type(frame.passcred) is not int
        or type(passcred) is not int
        or passcred != frame.passcred
        or passcred != 1
        or type(frame.timeout) is not float
        or type(timeout) is not float
        or timeout != frame.timeout
        or timeout != _PROTOCOL_TIMEOUT_SECONDS
        or type(frame.local_name) is not str
        or type(local_name) is not str
        or local_name != frame.local_name
        or local_name != ""
        or type(frame.peer_name) is not str
        or type(peer_name) is not str
        or peer_name != frame.peer_name
        or peer_name != ""
        or type(frame.proc_target) is not str
        or type(proc_target) is not str
        or proc_target != frame.proc_target
        or _socket_proc_inode(proc_target) != metadata.st_ino
        or channel.family is not socket.AF_UNIX
    ):
        raise Experiment002CoordinatorError(
            "registered child control descriptor changed"
        )


def _require_local_identity(credentials: _PeerCredentials) -> None:
    if type(credentials) is not _PeerCredentials:
        raise Experiment002CoordinatorError("peer credentials have an invalid type")
    if (
        type(credentials.pid) is not int
        or credentials.pid < 1
        or type(credentials.uid) is not int
        or credentials.uid < 0
        or type(credentials.gid) is not int
        or credentials.gid < 0
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


def _supervised_parent_child_frame(
    result: object,
    routes: _ParentExecutionRoutes,
    /,
) -> tuple[int, tuple[int, int], int, int, int, object]:
    supervised_child_result_type = routes[3]
    verified_child_result_type = routes[5]
    if type(result) is not supervised_child_result_type:
        raise Experiment002CoordinatorError(
            "supervisor returned an invalid exact child-result type"
        )
    pid = getattr(result, "pid", None)
    cpu_ids = getattr(result, "cpu_ids", None)
    elapsed_nanoseconds = getattr(result, "elapsed_nanoseconds", None)
    maximum_rss_bytes = getattr(result, "maximum_rss_bytes", None)
    output_and_scratch_bytes = getattr(result, "output_and_scratch_bytes", None)
    verified_child_result = getattr(result, "verified_child_result", None)
    if (
        type(pid) is not int
        or pid < 1
        or pid == os.getpid()
        or type(cpu_ids) is not tuple
        or len(cpu_ids) != 2
        or any(type(value) is not int or value < 0 for value in cpu_ids)
        or cpu_ids[0] >= cpu_ids[1]
        or type(elapsed_nanoseconds) is not int
        or elapsed_nanoseconds < 0
        or type(maximum_rss_bytes) is not int
        or maximum_rss_bytes < 0
        or type(output_and_scratch_bytes) is not int
        or output_and_scratch_bytes < 0
        or type(verified_child_result) is not verified_child_result_type
    ):
        raise Experiment002CoordinatorError(
            "supervised child result fields have invalid exact types or values"
        )
    return (
        pid,
        cpu_ids,
        elapsed_nanoseconds,
        maximum_rss_bytes,
        output_and_scratch_bytes,
        verified_child_result,
    )


def _make_verified_child_result_identity_frame() -> Callable[
    [object, _ParentExecutionRoutes],
    _VerifiedChildResultFrame,
]:
    exact_type = type
    read = getattr
    fraction_type = Fraction
    finite = math.isfinite
    sign = math.copysign
    binding_frame_route = _dynamic_child_result_binding_frame
    lower_hex = frozenset(_LOWER_HEX)
    error_type = Experiment002CoordinatorError

    def identity_frame(
        result: object,
        routes: _ParentExecutionRoutes,
        /,
    ) -> _VerifiedChildResultFrame:
        child_result_binding_type = routes[4]
        verified_child_result_type = routes[5]
        if exact_type(result) is not verified_child_result_type:
            raise error_type("verified child result has an invalid exact type")
        binding = read(result, "binding", None)
        binding_frame = binding_frame_route(
            binding,
            child_result_binding_type,
        )
        direct_frame = (
            read(result, "role", None),
            read(result, "ordinal", None),
            read(result, "seed", None),
            read(result, "registration_head_commit", None),
            read(result, "implementation_commit", None),
            read(result, "registration_sha256", None),
            read(result, "source_bundle_sha256", None),
        )
        if (
            exact_type(direct_frame[0]) is not str
            or exact_type(direct_frame[1]) is not int
            or exact_type(direct_frame[2]) is not int
            or exact_type(direct_frame[3]) is not str
            or exact_type(direct_frame[4]) is not str
            or exact_type(direct_frame[5]) is not str
            or exact_type(direct_frame[6]) is not str
            or direct_frame != binding_frame
        ):
            raise error_type("verified child result identity differs from its binding")
        winner_epoch = read(result, "winner_epoch", None)
        winner_macro_f1 = read(result, "winner_macro_f1", None)
        winner_validation_cross_entropy = read(
            result,
            "winner_validation_cross_entropy",
            None,
        )
        model_tensor_sha256 = read(result, "model_tensor_sha256", None)
        if (
            exact_type(winner_epoch) is not int
            or exact_type(winner_macro_f1) is not fraction_type
            or exact_type(winner_validation_cross_entropy) is not float
            or exact_type(model_tensor_sha256) is not str
        ):
            raise error_type(
                "verified child result rank fields have invalid exact types or values"
            )
        typed_winner_epoch = cast(int, winner_epoch)
        typed_winner_macro_f1 = cast(Fraction, winner_macro_f1)
        typed_winner_validation_cross_entropy = cast(
            float,
            winner_validation_cross_entropy,
        )
        typed_model_tensor_sha256 = cast(str, model_tensor_sha256)
        if (
            not 0 <= typed_winner_epoch < 30
            or not fraction_type(0, 1) <= typed_winner_macro_f1 <= fraction_type(1, 1)
            or not finite(typed_winner_validation_cross_entropy)
            or not typed_winner_validation_cross_entropy >= 0.0
            or sign(1.0, typed_winner_validation_cross_entropy) < 0.0
            or len(typed_model_tensor_sha256) != 64
            or any(
                character not in lower_hex for character in typed_model_tensor_sha256
            )
        ):
            raise error_type(
                "verified child result rank fields have invalid exact types or values"
            )
        return (
            *binding_frame,
            typed_winner_epoch,
            typed_winner_macro_f1.numerator,
            typed_winner_macro_f1.denominator,
            typed_winner_validation_cross_entropy.hex(),
            typed_model_tensor_sha256,
        )

    return identity_frame


_verified_child_result_identity_frame = _make_verified_child_result_identity_frame()
del _make_verified_child_result_identity_frame


type _ParentInputDescriptorGuard = Callable[[], None]
type _RegistrationBindingFramer = Callable[[object], _RegistrationBindingFrame]
type _ParentChildImplementation = Callable[
    [
        VerifiedRunRegistration,
        _RegisteredAssignmentFrame,
        tuple[int, int],
        int,
        _ParentInputDescriptorGuard,
        _RegistrationBindingFramer,
    ],
    _RegisteredParentChildResult,
]
type _ParentChildRoute = Callable[
    [VerifiedRunRegistration, _Assignment, tuple[int, int], int],
    _RegisteredParentChildResult,
]


def _bind_registered_parent_input_snapshot_authority(
    implementation: _ParentChildImplementation,
    /,
) -> _ParentChildRoute:
    exact_type = type
    type_cast = cast
    assignment_type = _Assignment
    role_descriptor = assignment_type.__dict__["role"]
    seed_descriptor = assignment_type.__dict__["seed"]
    ordinal_descriptor = assignment_type.__dict__["ordinal"]
    registration_binding_type = _RegistrationBinding
    head_commit_descriptor = registration_binding_type.__dict__["head_commit"]
    implementation_commit_descriptor = registration_binding_type.__dict__[
        "implementation_commit"
    ]
    registration_sha256_descriptor = registration_binding_type.__dict__[
        "registration_sha256"
    ]
    source_bundle_sha256_descriptor = registration_binding_type.__dict__[
        "source_bundle_sha256"
    ]
    if any(
        exact_type(descriptor) is not MemberDescriptorType
        for descriptor in (
            role_descriptor,
            seed_descriptor,
            ordinal_descriptor,
            head_commit_descriptor,
            implementation_commit_descriptor,
            registration_sha256_descriptor,
            source_bundle_sha256_descriptor,
        )
    ):
        raise RuntimeError("registered parent slot authority is unavailable")
    allocate = object.__new__
    error_type = Experiment002CoordinatorError
    base_exception_type = BaseException
    seed_role = _SEED_ROLE
    rerun_role = _RERUN_ROLE
    registered_seeds = (
        REGISTERED_SEEDS[0],
        REGISTERED_SEEDS[1],
        REGISTERED_SEEDS[2],
    )
    lower_hex = frozenset(_LOWER_HEX)

    def require_input_descriptors_unchanged() -> None:
        if (
            assignment_type.__dict__.get("role") is not role_descriptor
            or assignment_type.__dict__.get("seed") is not seed_descriptor
            or assignment_type.__dict__.get("ordinal") is not ordinal_descriptor
            or registration_binding_type.__dict__.get("head_commit")
            is not head_commit_descriptor
            or registration_binding_type.__dict__.get("implementation_commit")
            is not implementation_commit_descriptor
            or registration_binding_type.__dict__.get("registration_sha256")
            is not registration_sha256_descriptor
            or registration_binding_type.__dict__.get("source_bundle_sha256")
            is not source_bundle_sha256_descriptor
        ):
            raise error_type("registered parent input descriptors changed")

    def assignment_frame(value: object, /) -> tuple[str, int, int]:
        if exact_type(value) is not assignment_type:
            raise error_type("child assignment has an invalid exact type")
        role = role_descriptor.__get__(value, assignment_type)
        seed = seed_descriptor.__get__(value, assignment_type)
        ordinal = ordinal_descriptor.__get__(value, assignment_type)
        if (
            exact_type(role) is not str
            or exact_type(seed) is not int
            or exact_type(ordinal) is not int
        ):
            raise error_type("child assignment fields have invalid exact types")
        return role, seed, ordinal

    def snapshot_assignment(value: object, /) -> _Assignment:
        require_input_descriptors_unchanged()
        first = assignment_frame(value)
        role, seed, ordinal = first
        if role == seed_role:
            if seed == registered_seeds[0]:
                expected_ordinal = 0
            elif seed == registered_seeds[1]:
                expected_ordinal = 1
            elif seed == registered_seeds[2]:
                expected_ordinal = 2
            else:
                raise error_type("training child seed is not registered")
        elif role == rerun_role and seed in registered_seeds:
            expected_ordinal = 3
        else:
            raise error_type("child role and seed are not registered")
        if ordinal != expected_ordinal or assignment_frame(value) != first:
            raise error_type("child assignment changed while it was snapshotted")
        snapshot = allocate(assignment_type)
        role_descriptor.__set__(snapshot, role)
        seed_descriptor.__set__(snapshot, seed)
        ordinal_descriptor.__set__(snapshot, ordinal)
        if assignment_frame(snapshot) != first or assignment_frame(value) != first:
            raise error_type("child assignment changed while it was snapshotted")
        require_input_descriptors_unchanged()
        return snapshot

    def registration_binding_frame(
        value: object,
        /,
    ) -> _RegistrationBindingFrame:
        require_input_descriptors_unchanged()
        if exact_type(value) is not registration_binding_type:
            raise error_type("registration binding has an invalid exact type")

        def capture() -> _RegistrationBindingFrame:
            head_commit = head_commit_descriptor.__get__(
                value,
                registration_binding_type,
            )
            implementation_commit = implementation_commit_descriptor.__get__(
                value,
                registration_binding_type,
            )
            registration_sha256 = registration_sha256_descriptor.__get__(
                value,
                registration_binding_type,
            )
            source_bundle_sha256 = source_bundle_sha256_descriptor.__get__(
                value,
                registration_binding_type,
            )
            if (
                exact_type(head_commit) is not str
                or exact_type(implementation_commit) is not str
                or exact_type(registration_sha256) is not str
                or exact_type(source_bundle_sha256) is not str
            ):
                raise error_type("registration binding fields have invalid exact types")
            frame = (
                head_commit,
                implementation_commit,
                registration_sha256,
                source_bundle_sha256,
            )
            if (
                len(head_commit) != 40
                or len(implementation_commit) != 40
                or len(registration_sha256) != 64
                or len(source_bundle_sha256) != 64
                or any(
                    character not in lower_hex for field in frame for character in field
                )
            ):
                raise error_type("registration binding fields are not canonical")
            return frame

        first = capture()
        second = capture()
        require_input_descriptors_unchanged()
        if second != first:
            raise error_type("registration binding changed while it was snapshotted")
        return first

    def snapshot_cpu_ids(value: object, /) -> tuple[int, int]:
        if exact_type(value) is not tuple:
            raise error_type("captured CPU IDs are invalid")
        tuple_value = type_cast("tuple[object, ...]", value)
        if len(tuple_value) != 2:
            raise error_type("captured CPU IDs are invalid")
        left = tuple_value[0]
        right = tuple_value[1]
        if exact_type(left) is not int or exact_type(right) is not int:
            raise error_type("captured CPU IDs are invalid")
        typed_left = type_cast(int, left)
        typed_right = type_cast(int, right)
        if typed_left < 0 or typed_left >= typed_right:
            raise error_type("captured CPU IDs are invalid")
        snapshot = (typed_left, typed_right)
        if (tuple_value[0], tuple_value[1]) != snapshot:
            raise error_type("captured CPU IDs changed while they were snapshotted")
        return snapshot

    def _run_one_registered_parent_child(
        registration: VerifiedRunRegistration,
        assignment: _Assignment,
        cpu_ids: tuple[int, int],
        source_bundle_fd: int,
        /,
    ) -> _RegisteredParentChildResult:
        require_input_descriptors_unchanged()
        assignment_before = assignment_frame(assignment)
        assignment_snapshot = snapshot_assignment(assignment)
        assignment_snapshot_frame = assignment_frame(assignment_snapshot)
        if (
            assignment_snapshot_frame != assignment_before
            or assignment_frame(assignment) != assignment_before
        ):
            raise error_type("child assignment changed across its parent snapshot")
        cpu_ids_snapshot = snapshot_cpu_ids(cpu_ids)
        if exact_type(source_bundle_fd) is not int or source_bundle_fd < 0:
            raise error_type("source bundle descriptor has an invalid exact value")
        require_input_descriptors_unchanged()
        try:
            result = implementation(
                registration,
                assignment_snapshot_frame,
                cpu_ids_snapshot,
                source_bundle_fd,
                require_input_descriptors_unchanged,
                registration_binding_frame,
            )
        except base_exception_type as primary:
            try:
                require_input_descriptors_unchanged()
            except base_exception_type:
                raise error_type(
                    "registered parent execution failed after its input "
                    "descriptors changed"
                ) from primary
            raise
        require_input_descriptors_unchanged()
        return result

    return _run_one_registered_parent_child


@_bind_registered_parent_input_snapshot_authority
def _run_one_registered_parent_child(
    registration: VerifiedRunRegistration,
    assignment_frame: _RegisteredAssignmentFrame,
    cpu_ids: tuple[int, int],
    source_bundle_fd: int,
    require_input_descriptors_unchanged: _ParentInputDescriptorGuard,
    registration_binding_frame: _RegistrationBindingFramer,
    /,
) -> _RegisteredParentChildResult:
    """Run one fixed registered child while borrowing parent-owned resources."""

    require_registration = _require_registration
    registration_binding_route = _registration_binding
    claim_parent_execution_routes = _claim_and_verify_parent_execution_routes
    require_parent_execution_routes_unchanged = (
        _require_parent_execution_routes_unchanged
    )
    build_child_result_binding = _build_registered_child_result_binding
    parent_activate_child = _parent_activate_child
    supervised_parent_child_frame = _supervised_parent_child_frame
    verified_child_result_identity_frame = _verified_child_result_identity_frame
    dynamic_child_result_binding_frame = _dynamic_child_result_binding_frame
    fraction_type = Fraction
    math_module = math
    finite = math.isfinite
    sign = math.copysign
    lower_hex = _LOWER_HEX

    def require_coordinator_parent_routes_unchanged() -> None:
        if (
            _require_registration is not require_registration
            or _registration_binding is not registration_binding_route
            or _claim_and_verify_parent_execution_routes
            is not claim_parent_execution_routes
            or _require_parent_execution_routes_unchanged
            is not require_parent_execution_routes_unchanged
            or _build_registered_child_result_binding is not build_child_result_binding
            or _parent_activate_child is not parent_activate_child
            or _supervised_parent_child_frame is not supervised_parent_child_frame
            or _verified_child_result_identity_frame
            is not verified_child_result_identity_frame
            or _dynamic_child_result_binding_frame
            is not dynamic_child_result_binding_frame
            or Fraction is not fraction_type
            or math is not math_module
            or math.isfinite is not finite
            or math.copysign is not sign
            or _LOWER_HEX is not lower_hex
        ):
            raise Experiment002CoordinatorError(
                "coordinator parent execution routes changed"
            )

    require_input_descriptors_unchanged()
    require_registration(registration)
    if (
        type(assignment_frame) is not tuple
        or len(assignment_frame) != 3
        or type(assignment_frame[0]) is not str
        or type(assignment_frame[1]) is not int
        or type(assignment_frame[2]) is not int
    ):
        raise Experiment002CoordinatorError(
            "registered assignment frame has invalid exact types"
        )
    assignment_role, assignment_seed, assignment_ordinal = assignment_frame
    cpu_ids_snapshot = cpu_ids
    if type(source_bundle_fd) is not int or source_bundle_fd < 0:
        raise Experiment002CoordinatorError(
            "source bundle descriptor has an invalid exact value"
        )
    initial_registration_frame = registration_binding_frame(
        registration_binding_route(registration)
    )
    routes = claim_parent_execution_routes()
    (
        _supervisor_module,
        _child_result_module,
        supervise_child_route,
        _supervised_child_result_type,
        child_result_binding_type,
        _verified_child_result_type,
        _load_registered_child_result,
        verify_verified_child_result_route,
    ) = routes
    require_coordinator_parent_routes_unchanged()
    imported_registration_frame = registration_binding_frame(
        registration_binding_route(registration)
    )
    if initial_registration_frame != imported_registration_frame:
        raise Experiment002CoordinatorError(
            "run registration changed while parent routes were imported"
        )
    child_result_binding = build_child_result_binding(
        routes,
        assignment_frame,
        imported_registration_frame,
    )
    expected_child_result_binding_frame = (
        assignment_role,
        assignment_ordinal,
        assignment_seed,
        *initial_registration_frame,
    )
    first_child_result_binding_frame = dynamic_child_result_binding_frame(
        child_result_binding,
        child_result_binding_type,
    )
    second_child_result_binding_frame = dynamic_child_result_binding_frame(
        child_result_binding,
        child_result_binding_type,
    )
    if (
        first_child_result_binding_frame != expected_child_result_binding_frame
        or second_child_result_binding_frame != first_child_result_binding_frame
    ):
        raise Experiment002CoordinatorError(
            "registered child-result binding differs from parent primitive frames"
        )
    constructed_registration_frame = registration_binding_frame(
        registration_binding_route(registration)
    )
    if initial_registration_frame != constructed_registration_frame:
        raise Experiment002CoordinatorError(
            "run registration changed while the child binding was constructed"
        )

    def require_stable_parent_boundary() -> None:
        require_input_descriptors_unchanged()
        require_coordinator_parent_routes_unchanged()
        require_parent_execution_routes_unchanged(routes)
        require_registration(registration)
        current_registration_frame = registration_binding_frame(
            registration_binding_route(registration)
        )
        if initial_registration_frame != current_registration_frame:
            raise Experiment002CoordinatorError(
                "run registration changed across the parent execution boundary"
            )
        require_input_descriptors_unchanged()

    activation_lock = threading.Lock()
    accepting_activation = True
    activation_attempts = 0
    activated_pid: int | None = None

    def activate(channel: socket.socket, child_pid: int, /) -> None:
        nonlocal activated_pid, activation_attempts
        with activation_lock:
            if not accepting_activation:
                raise Experiment002CoordinatorError(
                    "supervisor activation callback is no longer accepting calls"
                )
            activation_attempts += 1
            if activation_attempts != 1:
                raise Experiment002CoordinatorError(
                    "supervisor attempted child activation more than once"
                )
            require_input_descriptors_unchanged()
            parent_activate_child(
                channel,
                registration,
                role=assignment_role,
                seed=assignment_seed,
                child_pid=child_pid,
            )
            require_input_descriptors_unchanged()
            if activated_pid is not None:
                raise Experiment002CoordinatorError(
                    "supervisor completed child activation more than once"
                )
            activated_pid = child_pid

    require_registration(registration)
    supervise_child = cast(
        Callable[
            [tuple[int, int], Callable[[socket.socket, int], None], int, object],
            object,
        ],
        supervise_child_route,
    )
    try:
        try:
            supervised = supervise_child(
                cpu_ids_snapshot,
                activate,
                source_bundle_fd,
                child_result_binding,
            )
        finally:
            with activation_lock:
                accepting_activation = False
    except BaseException as primary:
        try:
            require_stable_parent_boundary()
        except BaseException:
            raise Experiment002CoordinatorError(
                "child supervision failed after its parent boundary changed"
            ) from primary
        raise
    try:
        require_stable_parent_boundary()
        with activation_lock:
            completed_activation_attempts = activation_attempts
            completed_activation_pid = activated_pid
        if completed_activation_attempts != 1 or completed_activation_pid is None:
            raise Experiment002CoordinatorError(
                "supervisor returned without one completed child activation"
            )

        first = supervised_parent_child_frame(supervised, routes)
        second = supervised_parent_child_frame(supervised, routes)
        if first[:-1] != second[:-1] or first[-1] is not second[-1]:
            raise Experiment002CoordinatorError(
                "supervised child result changed while it was snapshotted"
            )
        if first[0] != completed_activation_pid or first[1] != cpu_ids_snapshot:
            raise Experiment002CoordinatorError(
                "supervised child result differs from its activation or CPU assignment"
            )
        verified_child_result = first[-1]
        verifier = cast(
            Callable[[object], object],
            verify_verified_child_result_route,
        )
        verification = verifier(verified_child_result)
        if verification is not None:
            raise Experiment002CoordinatorError(
                "verified child-result verifier returned an unexpected value"
            )
        expected_identity = (
            assignment_role,
            assignment_ordinal,
            assignment_seed,
            *initial_registration_frame,
        )
        first_identity = verified_child_result_identity_frame(
            verified_child_result,
            routes,
        )
        second_identity = verified_child_result_identity_frame(
            verified_child_result,
            routes,
        )
        if first_identity[:7] != expected_identity or second_identity != first_identity:
            raise Experiment002CoordinatorError(
                "verified child result differs from its registered assignment"
            )
        verification = verifier(verified_child_result)
        if verification is not None:
            raise Experiment002CoordinatorError(
                "verified child-result verifier returned an unexpected value"
            )

        third = supervised_parent_child_frame(supervised, routes)
        fourth = supervised_parent_child_frame(supervised, routes)
        if (
            third[:-1] != first[:-1]
            or third[-1] is not first[-1]
            or fourth[:-1] != third[:-1]
            or fourth[-1] is not third[-1]
        ):
            raise Experiment002CoordinatorError(
                "supervised child result changed after final verification"
            )
        third_identity = verified_child_result_identity_frame(
            verified_child_result,
            routes,
        )
        fourth_identity = verified_child_result_identity_frame(
            verified_child_result,
            routes,
        )
        if (
            third_identity != first_identity
            or fourth_identity != third_identity
            or third_identity[:7] != expected_identity
        ):
            raise Experiment002CoordinatorError(
                "verified child result changed after final verification"
            )
        require_stable_parent_boundary()
        registered_result = (
            third[0],
            third[1],
            third[2],
            third[3],
            third[4],
            cast(
                _VerifiedChildResultView,
                verified_child_result,
            ),
        )
        require_stable_parent_boundary()
    except BaseException as primary:
        try:
            require_stable_parent_boundary()
        except BaseException:
            raise Experiment002CoordinatorError(
                "child result validation failed after its registration or "
                "coordinator parent execution routes changed"
            ) from primary
        raise
    return registered_result


del _bind_registered_parent_input_snapshot_authority


def _make_guarded_registered_seed_child() -> Callable[[VerifiedRunRegistration], None]:
    """Capture the terminal one-attempt child route in an unresettable closure."""

    attempted = False
    attempt_lock = threading.Lock()

    def run_guarded_registered_seed_child(
        registration: VerifiedRunRegistration,
        /,
    ) -> None:
        nonlocal attempted

        with attempt_lock:
            if attempted:
                raise Experiment002CoordinatorError(
                    "registered child activation was already attempted"
                )
            attempted = True

        if type(registration) is not VerifiedRunRegistration:
            raise TypeError("registration must be an exact VerifiedRunRegistration")

        activation: VerifiedChildActivation | None = None
        try:
            try:
                channel = socket.socket(fileno=_CHILD_CONTROL_FD)
            except OSError as error:
                raise Experiment002CoordinatorError(
                    "registered child control descriptor 3 is unavailable"
                ) from error

            with channel:
                descriptor = channel.fileno()
                if type(descriptor) is not int or descriptor != _CHILD_CONTROL_FD:
                    raise Experiment002CoordinatorError(
                        "registered child control descriptor changed"
                    )
                channel_frame = _capture_fixed_control_channel_frame(channel)
                peeked = _peek_packet(channel)
                _require_fixed_control_channel_frame(channel, channel_frame)

                process_guard_binding = _claim_and_verify_child_process_guard()
                _require_fixed_control_channel_frame(channel, channel_frame)

                registration_binding = _registration_binding(registration)
                _require_registration(registration)
                _require_fixed_control_channel_frame(channel, channel_frame)

                ticket = _validate_ticket(
                    _decode_packet(peeked.packet),
                    peeked.credentials,
                    binding=registration_binding,
                    packet=peeked.packet,
                )
                _require_fixed_control_channel_frame(channel, channel_frame)

                consumed = _consume_packet(channel)
                _require_fixed_control_channel_frame(channel, channel_frame)
                _require_same_received_packet(peeked, consumed)
                _require_fixed_control_channel_frame(channel, channel_frame)

                _require_process_guard_binding(process_guard_binding)
                _require_fixed_control_channel_frame(channel, channel_frame)

                receipt = _reserve_accepted_ticket(ticket)
                _require_fixed_control_channel_frame(channel, channel_frame)
                activation = _issue_activation(
                    registration,
                    ticket,
                    receipt,
                    process_guard_binding,
                )
                _require_fixed_control_channel_frame(channel, channel_frame)
                verify_verified_child_activation(registration, activation)
                _require_fixed_control_channel_frame(channel, channel_frame)

                acknowledgement = _encode_packet(_ack_document(ticket))
                _require_fixed_control_channel_frame(channel, channel_frame)
                _send_packet_prepared(channel, acknowledgement)

            try:
                os.fstat(_CHILD_CONTROL_FD)
            except OSError as error:
                if error.errno != errno.EBADF:
                    raise Experiment002CoordinatorError(
                        "closed child control descriptor could not be verified"
                    ) from error
            else:
                raise Experiment002CoordinatorError(
                    "registered child control descriptor remained open"
                )
            if activation is None:
                raise Experiment002CoordinatorError(
                    "registered child activation was not issued"
                )
            _dispatch_registered_seed_process(registration, activation)
        except BaseException:
            if activation is not None:
                _poison_activation(activation)
            raise

    return run_guarded_registered_seed_child


_run_guarded_registered_seed_child = _make_guarded_registered_seed_child()
_run_registered_seed_child = _run_guarded_registered_seed_child
del _make_guarded_registered_seed_child


def _validate_ticket(
    document: dict[str, Any],
    credentials: _PeerCredentials,
    *,
    binding: _RegistrationBinding,
    packet: bytes,
) -> _Ticket:
    _require_local_identity(credentials)
    _require_registration_binding(binding)
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
    if type(document["kind"]) is not str or document["kind"] != "activation_ticket":
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
    parsed_binding = _parse_binding(document["registration"])
    if not _registration_bindings_match(parsed_binding, binding):
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
    _require_local_identity(credentials)
    _require_assignment(assignment)
    _require_distinct_processes(parent_pid, child_pid)
    _require_lower_hex(nonce, 64, "ack nonce")
    _require_lower_hex(packet_sha256, 64, "ack ticket sha256")
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
    if (
        document["child_pid"] != child_pid
        or document["kind"] != "activation_ack"
        or document["nonce"] != nonce
        or document["ordinal"] != assignment.ordinal
        or document["parent_pid"] != parent_pid
        or document["role"] != assignment.role
        or document["schema_version"] != _SCHEMA_VERSION
        or document["seed"] != assignment.seed
        or document["status"] != "accepted"
        or document["ticket_sha256"] != packet_sha256
        or observed_parent_pid != parent_pid
        or observed_child_pid != child_pid
        or not _assignments_match(observed_assignment, assignment)
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
    if type(document) is not dict:
        raise Experiment002CoordinatorError(f"{name} fields are not exact")
    if any(type(key) is not str for key in document):
        raise Experiment002CoordinatorError(f"{name} keys must be strings")
    if type(expected) is not set or any(type(key) is not str for key in expected):
        raise Experiment002CoordinatorError(f"{name} expected fields are invalid")
    if set(document) != expected:
        raise Experiment002CoordinatorError(f"{name} fields are not exact")


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
    if (
        type(assignment.role) is not str
        or type(assignment.seed) is not int
        or type(assignment.ordinal) is not int
    ):
        raise Experiment002CoordinatorError(
            "child assignment fields have invalid exact types"
        )
    expected = _assignment(assignment.role, assignment.seed)
    if not _assignments_match(assignment, expected):
        raise Experiment002CoordinatorError("child assignment is not registered")


def _assignments_match(first: _Assignment, second: _Assignment) -> bool:
    return (
        type(first) is _Assignment
        and type(second) is _Assignment
        and type(first.role) is str
        and type(second.role) is str
        and first.role == second.role
        and type(first.seed) is int
        and type(second.seed) is int
        and first.seed == second.seed
        and type(first.ordinal) is int
        and type(second.ordinal) is int
        and first.ordinal == second.ordinal
    )


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
    accepted_nonce = ticket.nonce
    accepted_digest = ticket.packet_sha256
    with _PROTOCOL_LOCK:
        receipt = _trusted_accept(ticket)
        _require_ticket(ticket)
        if (
            ticket.nonce != accepted_nonce
            or ticket.packet_sha256 != accepted_digest
            or accepted_digest in _ACCEPTED_TICKET_DIGESTS
            or accepted_nonce in _ACCEPTED_TICKET_NONCES
        ):
            raise Experiment002CoordinatorError(
                "activation ticket was replayed or changed during acceptance"
            )
        _ACCEPTED_TICKET_DIGESTS.add(accepted_digest)
        _ACCEPTED_TICKET_NONCES.add(accepted_nonce)
        return receipt


def _ticket_sha256(packet: bytes) -> str:
    if type(packet) is not bytes:
        raise TypeError("ticket packet must be exact bytes")
    return hashlib.sha256(_TICKET_DIGEST_DOMAIN + packet).hexdigest()


def _issue_activation(
    registration: VerifiedRunRegistration,
    ticket: _Ticket,
    receipt: _AcceptedTicketReceipt,
    process_guard_binding: _ProcessGuardBinding,
    *,
    _trusted_record: Callable[
        [
            _AcceptedTicketReceipt,
            _Ticket,
            VerifiedChildActivation,
            VerifiedRunRegistration,
            object,
            object,
            _ProcessGuardVerifier,
        ],
        None,
    ] = _TRUSTED_RECORD_ISSUANCE,
    _trusted_poison: Callable[[VerifiedChildActivation], None] = (
        _TRUSTED_POISON_ACTIVATION
    ),
) -> VerifiedChildActivation:
    _require_ticket(ticket)
    _require_process_guard_binding(process_guard_binding)
    _require_registration(registration)
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
        process_guard=process_guard_binding.guard,
        process_guard_verifier=process_guard_binding.verifier,
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
                process_guard_binding.guard,
                process_guard_binding.verifier,
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
        try:
            trusted_phase, trusted_failed, trusted_valid = _trusted_verify(
                activation,
                state,
            )
        except BaseException:
            _trusted_poison(activation)
            _FAILED_ACTIVATIONS.add(activation)
            raise
        if (
            type(state) is not _ActivationState
            or type(trusted_phase) is not int
            or type(trusted_failed) is not bool
            or type(trusted_valid) is not bool
            or type(phase) is not int
            or activation in _FAILED_ACTIVATIONS
            or trusted_failed
            or not trusted_valid
            or not _valid_activation_state(state)
            or not _activation_guard_matches_state(guard, state)
            or not _activation_anchor_matches_state(anchor, state)
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


def _valid_activation_state(state: object) -> bool:
    if type(state) is not _ActivationState:
        return False
    try:
        if (
            type(state.role) is not str
            or type(state.seed) is not int
            or type(state.ordinal) is not int
            or type(state.phase) is not int
            or type(state.registration) is not VerifiedRunRegistration
            or state.process_guard is None
            or type(state.process_guard_verifier) is not FunctionType
            or type(state.token) is not object
        ):
            return False
        _require_pid(state.process_id, "activation process PID")
        _require_distinct_processes(state.parent_pid, state.child_pid)
        _require_assignment(_Assignment(state.role, state.seed, state.ordinal))
        _require_lower_hex(state.nonce, 64, "activation nonce")
        _require_lower_hex(state.ticket_sha256, 64, "activation ticket sha256")
        _require_process_guard_binding(
            _ProcessGuardBinding(
                guard=state.process_guard,
                verifier=state.process_guard_verifier,
            )
        )
    except BaseException:
        return False
    return (
        state.marker is _ACTIVATION_MARKER
        and state.process_id == os.getpid()
        and state.child_pid == os.getpid()
        and state.parent_pid == os.getppid()
        and state.phase in {_PHASE_ISSUED, _PHASE_DISPATCHED, _PHASE_COMPLETED}
    )


def _activation_guard_matches_state(guard: object, state: object) -> bool:
    return (
        type(guard) is _ActivationGuard
        and type(state) is _ActivationState
        and guard.marker is state.marker
        and guard.marker is _ACTIVATION_MARKER
        and guard.registration is state.registration
        and type(guard.process_id) is int
        and type(state.process_id) is int
        and guard.process_id == state.process_id
        and type(guard.parent_pid) is int
        and type(state.parent_pid) is int
        and guard.parent_pid == state.parent_pid
        and type(guard.child_pid) is int
        and type(state.child_pid) is int
        and guard.child_pid == state.child_pid
        and type(guard.role) is str
        and type(state.role) is str
        and guard.role == state.role
        and type(guard.seed) is int
        and type(state.seed) is int
        and guard.seed == state.seed
        and type(guard.ordinal) is int
        and type(state.ordinal) is int
        and guard.ordinal == state.ordinal
        and type(guard.nonce) is str
        and type(state.nonce) is str
        and guard.nonce == state.nonce
        and type(guard.ticket_sha256) is str
        and type(state.ticket_sha256) is str
        and guard.ticket_sha256 == state.ticket_sha256
        and guard.process_guard is state.process_guard
        and guard.process_guard_verifier is state.process_guard_verifier
        and guard.token is state.token
        and type(guard.phase) is int
        and type(state.phase) is int
        and guard.phase == state.phase
    )


def _activation_anchor_matches_state(anchor: object, state: object) -> bool:
    return (
        type(anchor) is _ActivationAnchor
        and type(state) is _ActivationState
        and anchor.marker is state.marker
        and anchor.marker is _ACTIVATION_MARKER
        and anchor.registration is state.registration
        and type(anchor.process_id) is int
        and type(state.process_id) is int
        and anchor.process_id == state.process_id
        and type(anchor.parent_pid) is int
        and type(state.parent_pid) is int
        and anchor.parent_pid == state.parent_pid
        and type(anchor.child_pid) is int
        and type(state.child_pid) is int
        and anchor.child_pid == state.child_pid
        and type(anchor.role) is str
        and type(state.role) is str
        and anchor.role == state.role
        and type(anchor.seed) is int
        and type(state.seed) is int
        and anchor.seed == state.seed
        and type(anchor.ordinal) is int
        and type(state.ordinal) is int
        and anchor.ordinal == state.ordinal
        and type(anchor.nonce) is str
        and type(state.nonce) is str
        and anchor.nonce == state.nonce
        and type(anchor.ticket_sha256) is str
        and type(state.ticket_sha256) is str
        and anchor.ticket_sha256 == state.ticket_sha256
        and anchor.process_guard is state.process_guard
        and anchor.process_guard_verifier is state.process_guard_verifier
        and anchor.token is state.token
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
        process_guard=state.process_guard,
        process_guard_verifier=state.process_guard_verifier,
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
        process_guard=state.process_guard,
        process_guard_verifier=state.process_guard_verifier,
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
    if (
        type(expected_phase) is not int
        or type(next_phase) is not int
        or expected_phase not in {_PHASE_ISSUED, _PHASE_DISPATCHED}
        or next_phase not in {_PHASE_DISPATCHED, _PHASE_COMPLETED}
    ):
        _poison_activation(activation)
        raise Experiment002CoordinatorError(
            "child activation transition phases have invalid exact types"
        )
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
