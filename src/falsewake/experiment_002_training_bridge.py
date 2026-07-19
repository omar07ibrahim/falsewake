"""Process-local training-transition capabilities for Experiment 002.

The registered executor, population, and update trace exchange opaque, one-use
capabilities through this module.  Process tokens are authority only: none of
them are included in a registered population, update-trace, or history frame.
"""

from __future__ import annotations

import math
import struct
import threading
import weakref
from collections.abc import Container, Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from typing import Literal, NoReturn

import numpy as np

type TransitionPhase = Literal[
    "ISSUED", "TRACE_CONSUMED", "POPULATION_ACCEPTED", "FAILED"
]

_REGISTERED_ROUTE = object()
_REGISTERED_SEEDS = (20_260_719, 20_260_720, 20_260_721)
_REGISTERED_EPOCHS = 30
_UPDATES_PER_EPOCH = 313
_REGISTERED_BATCH_SIZE = 128
_REGISTERED_LAST_BATCH_SIZE = 91
_TOTAL_UPDATES = _REGISTERED_EPOCHS * _UPDATES_PER_EPOCH


class Experiment002TrainingBridgeError(ValueError):
    """An exact training authority crossed an invalid process boundary."""


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
class RegisteredExecutorSessionAuthority:
    """Issuer-only authority owned by one exact registered executor."""

    def __init__(self) -> None:
        raise TypeError("executor session authorities are issuer-only")

    def __copy__(self) -> NoReturn:
        raise TypeError("executor session authorities cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        del memo
        raise TypeError("executor session authorities cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("executor session authorities cannot be serialized")


@dataclass(frozen=True, slots=True)
class _ExecutorSessionSnapshot:
    session_token: object
    seed: int
    route_marker: object


@dataclass(frozen=True, slots=True)
class _ExecutorSessionIssuerAuthority:
    session_token: object
    seed: int
    route_marker: object
    authority_payload: bytes
    generation_anchor: _ExecutorGenerationAnchor


@dataclass(slots=True)
class _IssuedExecutorSessionState:
    snapshot: _ExecutorSessionSnapshot
    issuer: _ExecutorSessionIssuerAuthority
    next_optimizer_generation: int = 0
    active_transition_token: object | None = None
    failed: bool = False


@dataclass(frozen=True, slots=True)
class _ExecutorSessionLifecycleGuard:
    next_optimizer_generation: int
    active_transition_token: object | None
    failed: bool


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
class RegisteredOptimizerTransition:
    """Issuer-only proof of one exact optimizer generation transition."""

    def __init__(self) -> None:
        raise TypeError("optimizer transitions are issued by the exact executor")

    @property
    def phase(self) -> TransitionPhase:
        return _issued_transition_phase(self)

    def __copy__(self) -> NoReturn:
        raise TypeError("optimizer transitions cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        del memo
        raise TypeError("optimizer transitions cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("optimizer transitions cannot be serialized")


@dataclass(frozen=True, slots=True)
class _OptimizerTransitionSnapshot:
    executor_session_token: object
    epoch_session_token: object
    batch_token: object
    transition_token: object
    seed: int
    zero_based_epoch: int
    zero_based_global_update: int
    batch_size: int
    learning_rate: float
    learning_rate_bytes: bytes
    batch_mean_training_loss: np.float32
    batch_mean_training_loss_bytes: bytes
    returned_preclip_l2_norm: np.float32
    returned_preclip_l2_norm_bytes: bytes
    optimizer_generation_before: int
    optimizer_generation_after: int
    model_sha256_before: str
    model_sha256_after: str
    optimizer_sha256_before: str
    optimizer_sha256_after: str
    rng_sha256_before: str
    rng_sha256_after: str


@dataclass(frozen=True, slots=True, eq=False)
class _ExecutorGenerationAuthority:
    executor_authority: RegisteredExecutorSessionAuthority
    transition: RegisteredOptimizerTransition
    transition_token: object
    global_history_index: int
    previous_generation: _ExecutorGenerationAuthority | None
    generation_before: int
    generation_after: int
    model_sha256_before: str
    model_sha256_after: str
    optimizer_sha256_before: str
    optimizer_sha256_after: str
    rng_sha256_before: str
    rng_sha256_after: str


class _ExecutorGenerationAnchor:
    """Fixed-identity one-way head for one session's immutable generation chain."""

    __slots__ = ("_executor_authority", "_seed", "_latest_generation")
    _executor_authority: RegisteredExecutorSessionAuthority
    _seed: int
    _latest_generation: _ExecutorGenerationAuthority | None

    def __init__(
        self, executor_authority: RegisteredExecutorSessionAuthority, seed: int
    ) -> None:
        for attribute in self.__slots__:
            try:
                object.__getattribute__(self, attribute)
            except AttributeError:
                continue
            raise TypeError("executor generation anchors cannot be reinitialized")
        object.__setattr__(self, "_executor_authority", executor_authority)
        object.__setattr__(self, "_seed", seed)
        object.__setattr__(self, "_latest_generation", None)

    @property
    def executor_authority(self) -> RegisteredExecutorSessionAuthority:
        return self._executor_authority

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def generation_count(self) -> int:
        latest = self._latest_generation
        return 0 if latest is None else latest.generation_after

    @property
    def latest_generation(self) -> _ExecutorGenerationAuthority | None:
        return self._latest_generation

    def _append(self, generation: _ExecutorGenerationAuthority) -> None:
        latest = self._latest_generation
        if (
            type(generation) is not _ExecutorGenerationAuthority
            or generation.executor_authority is not self._executor_authority
            or generation.previous_generation is not latest
            or generation.generation_before != self.generation_count
            or generation.generation_after != self.generation_count + 1
            or generation not in _ISSUED_EXECUTOR_GENERATIONS
            or not 0
            <= generation.global_history_index
            < len(_EXECUTOR_GENERATION_HISTORY)
            or _EXECUTOR_GENERATION_HISTORY[generation.global_history_index]
            is not generation
        ):
            raise Experiment002TrainingBridgeError(
                "executor generation append is not monotonic"
            )
        object.__setattr__(self, "_latest_generation", generation)

    def __setattr__(self, name: str, value: object) -> NoReturn:
        del name, value
        raise TypeError("executor generation anchors are append-only")

    def __delattr__(self, name: str) -> NoReturn:
        del name
        raise TypeError("executor generation anchors are append-only")


@dataclass(frozen=True, slots=True)
class _TransitionIssuerAuthority:
    executor_authority: RegisteredExecutorSessionAuthority | None
    executor_session_token: object
    epoch_session_token: object
    batch_token: object
    transition_token: object
    route_marker: object | None
    authority_payload: bytes


@dataclass(frozen=True, slots=True, eq=False)
class _SyntheticTransitionAuthority:
    transition: RegisteredOptimizerTransition
    executor_session_token: object
    epoch_session_token: object
    batch_token: object
    transition_token: object
    route_marker: object | None


@dataclass(frozen=True, slots=True)
class _TraceMarkAuthority:
    receipt_token: object
    trace_session_token: object
    trace_record_index: int


@dataclass(slots=True)
class _IssuedTransitionState:
    snapshot: _OptimizerTransitionSnapshot
    issuer: _TransitionIssuerAuthority
    phase: TransitionPhase = "ISSUED"
    trace_authority: _TraceMarkAuthority | None = None
    trace_consumption: TraceConsumedTransition | None = None
    invalid: bool = False


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
class TraceConsumedTransition:
    """Issuer-only, one-use proof that one trace consumed one transition."""

    def __init__(self) -> None:
        raise TypeError("trace consumptions are issued by the registered trace")

    def __copy__(self) -> NoReturn:
        raise TypeError("trace consumptions cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        del memo
        raise TypeError("trace consumptions cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("trace consumptions cannot be serialized")


@dataclass(frozen=True, slots=True)
class _TraceConsumptionSnapshot:
    executor_session_token: object
    epoch_session_token: object
    batch_token: object
    transition_token: object
    receipt_token: object
    trace_session_token: object
    consumption_token: object
    zero_based_global_update: int
    trace_record_index: int


@dataclass(frozen=True, slots=True)
class _ConsumptionIssuerAuthority:
    transition: RegisteredOptimizerTransition
    executor_session_token: object
    epoch_session_token: object
    batch_token: object
    transition_token: object
    receipt_token: object
    trace_session_token: object
    consumption_token: object
    zero_based_global_update: int
    trace_record_index: int
    authority_payload: bytes


@dataclass(slots=True)
class _IssuedConsumptionState:
    snapshot: _TraceConsumptionSnapshot
    issuer: _ConsumptionIssuerAuthority
    used: bool = False
    failed: bool = False
    invalid: bool = False


_EXECUTOR_SESSIONS: weakref.WeakKeyDictionary[
    RegisteredExecutorSessionAuthority, _IssuedExecutorSessionState
] = weakref.WeakKeyDictionary()
_EXECUTOR_SESSION_GUARDS: weakref.WeakKeyDictionary[
    RegisteredExecutorSessionAuthority, _ExecutorSessionIssuerAuthority
] = weakref.WeakKeyDictionary()
_EXECUTOR_SESSION_LIFECYCLES: weakref.WeakKeyDictionary[
    RegisteredExecutorSessionAuthority, _ExecutorSessionLifecycleGuard
] = weakref.WeakKeyDictionary()
_EXECUTOR_GENERATION_LEDGERS: weakref.WeakKeyDictionary[
    RegisteredExecutorSessionAuthority, tuple[_ExecutorGenerationAuthority, ...]
] = weakref.WeakKeyDictionary()
# These individual capabilities are the monotonic authority, not another
# replaceable per-session lifecycle mirror. Registered training has a fixed
# maximum of 3 * 30 * 313 = 28,170 entries, so retaining them strongly for the
# process lifetime is deliberately bounded. Neither collection has a removal
# path; the independent set prevents a coherent rewrite of the ordered mirror.
_EXECUTOR_GENERATION_HISTORY: list[_ExecutorGenerationAuthority] = []
_ISSUED_EXECUTOR_GENERATIONS: set[_ExecutorGenerationAuthority] = set()
_EXECUTOR_GENERATION_ANCHORS: dict[
    RegisteredExecutorSessionAuthority, _ExecutorGenerationAnchor
] = {}
_EXECUTOR_GENERATION_ANCHOR_HISTORY: list[_ExecutorGenerationAnchor] = []
_ISSUED_EXECUTOR_GENERATION_ANCHORS: set[_ExecutorGenerationAnchor] = set()
_SYNTHETIC_TRANSITION_AUTHORITY_HISTORY: list[_SyntheticTransitionAuthority] = []
_ISSUED_SYNTHETIC_TRANSITION_AUTHORITIES: set[_SyntheticTransitionAuthority] = set()
_FAILED_EXECUTOR_SESSIONS: weakref.WeakSet[RegisteredExecutorSessionAuthority] = (
    weakref.WeakSet()
)
_ISSUED_REGISTERED_SEEDS: set[int] = set()
_ISSUED_TRANSITIONS: weakref.WeakKeyDictionary[
    RegisteredOptimizerTransition, _IssuedTransitionState
] = weakref.WeakKeyDictionary()
_TRANSITION_GUARDS: weakref.WeakKeyDictionary[
    RegisteredOptimizerTransition, _TransitionIssuerAuthority
] = weakref.WeakKeyDictionary()
_TRANSITION_PHASES: weakref.WeakKeyDictionary[
    RegisteredOptimizerTransition, TransitionPhase
] = weakref.WeakKeyDictionary()
_TRANSITION_CONSUMPTIONS: weakref.WeakKeyDictionary[
    RegisteredOptimizerTransition, TraceConsumedTransition
] = weakref.WeakKeyDictionary()
_TRANSITION_CLAIM_GUARDS: weakref.WeakKeyDictionary[
    RegisteredOptimizerTransition, TraceConsumedTransition
] = weakref.WeakKeyDictionary()
_CLAIMED_TRANSITIONS: weakref.WeakSet[RegisteredOptimizerTransition] = weakref.WeakSet()
_ACCEPTED_TRANSITIONS: weakref.WeakSet[RegisteredOptimizerTransition] = (
    weakref.WeakSet()
)
_FAILED_TRANSITIONS: weakref.WeakSet[RegisteredOptimizerTransition] = weakref.WeakSet()
_INVALID_TRANSITIONS: weakref.WeakSet[RegisteredOptimizerTransition] = weakref.WeakSet()
_ISSUED_CONSUMPTIONS: weakref.WeakKeyDictionary[
    TraceConsumedTransition, _IssuedConsumptionState
] = weakref.WeakKeyDictionary()
_CONSUMPTION_GUARDS: weakref.WeakKeyDictionary[
    TraceConsumedTransition, _ConsumptionIssuerAuthority
] = weakref.WeakKeyDictionary()
_CONSUMPTION_LIFECYCLES: weakref.WeakKeyDictionary[
    TraceConsumedTransition, tuple[bool, bool]
] = weakref.WeakKeyDictionary()
_FAILED_CONSUMPTIONS: weakref.WeakSet[TraceConsumedTransition] = weakref.WeakSet()
_USED_CONSUMPTIONS: weakref.WeakSet[TraceConsumedTransition] = weakref.WeakSet()
_INVALID_CONSUMPTIONS: weakref.WeakSet[TraceConsumedTransition] = weakref.WeakSet()
_BRIDGE_LOCK = threading.RLock()


def _safe_registry_get[K, V](registry: Mapping[K, V], key: K) -> V | None:
    try:
        return registry.get(key)
    except BaseException:
        return None


def _safe_registry_items[K, V](
    registry: Mapping[K, V],
) -> tuple[tuple[K, V], ...]:
    try:
        return tuple(registry.items())
    except BaseException:
        return ()


def _safe_registry_values[K, V](registry: Mapping[K, V]) -> tuple[V, ...]:
    try:
        return tuple(registry.values())
    except BaseException:
        return ()


def _safe_iterable[T](values: Iterable[T]) -> tuple[T, ...]:
    try:
        return tuple(values)
    except BaseException:
        return ()


def _safe_contains[T](values: Container[T], value: T) -> bool:
    try:
        return value in values
    except BaseException:
        return False


def _resilient_registry_lookup[K, V](
    registry: Mapping[K, V], key: K
) -> tuple[V | None, BaseException | None]:
    try:
        return registry.get(key), None
    except BaseException as error:
        for candidate, value in _safe_registry_items(registry):
            if candidate is key:
                return value, error
        return None, error


def _lookup_executor_session_or_fail_locked(
    authority: RegisteredExecutorSessionAuthority,
) -> _IssuedExecutorSessionState | None:
    state, lookup_error = _resilient_registry_lookup(_EXECUTOR_SESSIONS, authority)
    if lookup_error is not None:
        if (
            type(state) is _IssuedExecutorSessionState
            and _executor_session_state_matches_identity_locked(authority, state)
        ):
            _terminal_fail_executor_session_locked(authority, state)
        else:
            _invalidate_executor_authority_without_state_locked(authority)
        raise lookup_error
    if type(state) is not _IssuedExecutorSessionState:
        _invalidate_executor_authority_without_state_locked(authority)
        return None
    if not _executor_session_state_matches_identity_locked(authority, state):
        _invalidate_executor_authority_without_state_locked(authority)
        raise Experiment002TrainingBridgeError(
            "executor session primary state differs from independent authority"
        )
    return state


def _lookup_transition_or_fail_locked(
    transition: RegisteredOptimizerTransition,
) -> _IssuedTransitionState | None:
    state, lookup_error = _resilient_registry_lookup(_ISSUED_TRANSITIONS, transition)
    if lookup_error is not None:
        if type(
            state
        ) is _IssuedTransitionState and _transition_state_matches_identity_locked(
            transition, state
        ):
            _force_invalidate_transition_corruption_locked(transition, state)
        else:
            _poison_transition_identity_locked(transition)
        raise lookup_error
    if type(state) is not _IssuedTransitionState:
        _poison_transition_identity_locked(transition)
        return None
    if not _transition_state_matches_identity_locked(transition, state):
        _poison_transition_identity_locked(transition)
        raise Experiment002TrainingBridgeError(
            "optimizer transition primary state differs from independent authority"
        )
    return state


def _lookup_consumption_or_fail_locked(
    consumption: TraceConsumedTransition,
) -> _IssuedConsumptionState | None:
    state, lookup_error = _resilient_registry_lookup(_ISSUED_CONSUMPTIONS, consumption)
    if lookup_error is not None:
        if type(
            state
        ) is _IssuedConsumptionState and _consumption_state_matches_identity_locked(
            consumption, state
        ):
            _force_invalidate_consumption_corruption_locked(consumption, state)
        else:
            _poison_consumption_identity_locked(consumption)
        raise lookup_error
    if type(state) is not _IssuedConsumptionState:
        _poison_consumption_identity_locked(consumption)
        return None
    if not _consumption_state_matches_identity_locked(consumption, state):
        _poison_consumption_identity_locked(consumption)
        raise Experiment002TrainingBridgeError(
            "trace consumption primary state differs from independent authority"
        )
    return state


def _issue_registered_executor_session(
    seed: int,
) -> RegisteredExecutorSessionAuthority:
    """Private factory seam; issue one stateful registered executor authority."""

    _require_registered_seed(seed)
    result = object.__new__(RegisteredExecutorSessionAuthority)
    token = object()
    payload = struct.pack("<I", seed)
    snapshot = _ExecutorSessionSnapshot(token, seed, _REGISTERED_ROUTE)
    generation_anchor = _ExecutorGenerationAnchor(result, seed)
    issuer = _ExecutorSessionIssuerAuthority(
        token, seed, _REGISTERED_ROUTE, payload, generation_anchor
    )
    with _BRIDGE_LOCK:
        if (
            seed in _ISSUED_REGISTERED_SEEDS
            or any(
                type(anchor) is _ExecutorGenerationAnchor and anchor.seed == seed
                for anchor in _EXECUTOR_GENERATION_ANCHOR_HISTORY
            )
            or any(
                type(anchor) is _ExecutorGenerationAnchor and anchor.seed == seed
                for anchor in _ISSUED_EXECUTOR_GENERATION_ANCHORS
            )
        ):
            raise Experiment002TrainingBridgeError(
                "registered seed already has an executor session in this process"
            )
        _ISSUED_REGISTERED_SEEDS.add(seed)
        _EXECUTOR_SESSIONS[result] = _IssuedExecutorSessionState(snapshot, issuer)
        _EXECUTOR_SESSION_GUARDS[result] = replace(issuer)
        _EXECUTOR_SESSION_LIFECYCLES[result] = _ExecutorSessionLifecycleGuard(
            0, None, False
        )
        _EXECUTOR_GENERATION_LEDGERS[result] = ()
        _EXECUTOR_GENERATION_ANCHORS[result] = generation_anchor
        _ISSUED_EXECUTOR_GENERATION_ANCHORS.add(generation_anchor)
        _EXECUTOR_GENERATION_ANCHOR_HISTORY.append(generation_anchor)
    return result


def _executor_session_snapshot(
    authority: RegisteredExecutorSessionAuthority,
    *,
    require_live: bool = True,
) -> _ExecutorSessionSnapshot:
    """Return a checked copy of one exact executor-session binding."""

    if type(authority) is not RegisteredExecutorSessionAuthority:
        raise TypeError("authority must be a RegisteredExecutorSessionAuthority")
    if type(require_live) is not bool:
        raise TypeError("require_live must be a bool")
    with _BRIDGE_LOCK:
        state = _lookup_executor_session_or_fail_locked(authority)
        try:
            _validate_executor_session_state(authority, state)
        except BaseException:
            if type(state) is _IssuedExecutorSessionState:
                _terminal_fail_executor_session_locked(authority, state)
            raise
        assert state is not None
        if require_live and state.failed:
            raise Experiment002TrainingBridgeError(
                "executor session is terminally failed"
            )
        return replace(state.snapshot)


def _fail_registered_executor_session(
    authority: RegisteredExecutorSessionAuthority,
) -> None:
    """Terminally invalidate an exact executor session and active transition."""

    if type(authority) is not RegisteredExecutorSessionAuthority:
        raise TypeError("authority must be a RegisteredExecutorSessionAuthority")
    with _BRIDGE_LOCK:
        state = _lookup_executor_session_or_fail_locked(authority)
        if type(state) is not _IssuedExecutorSessionState:
            raise Experiment002TrainingBridgeError(
                "executor session authority was not issued by this process"
            )
        _terminal_fail_executor_session_locked(authority, state)


def _single_identity_anchor(
    candidates: list[_ExecutorGenerationAnchor],
) -> _ExecutorGenerationAnchor | None:
    unique: list[_ExecutorGenerationAnchor] = []
    for candidate in candidates:
        if not any(observed is candidate for observed in unique):
            unique.append(candidate)
    return unique[0] if len(unique) == 1 else None


def _trusted_session_anchor_locked(
    authority: RegisteredExecutorSessionAuthority,
    state: _IssuedExecutorSessionState | None,
) -> _ExecutorGenerationAnchor | None:
    candidates: list[_ExecutorGenerationAnchor] = []
    if type(state) is _IssuedExecutorSessionState:
        raw_anchor = state.issuer.generation_anchor
        if type(raw_anchor) is _ExecutorGenerationAnchor:
            candidates.append(raw_anchor)
    mapped_anchor = _safe_registry_get(_EXECUTOR_GENERATION_ANCHORS, authority)
    if type(mapped_anchor) is _ExecutorGenerationAnchor:
        candidates.append(mapped_anchor)
    for registry in (
        _safe_iterable(_EXECUTOR_GENERATION_ANCHOR_HISTORY),
        _safe_iterable(_ISSUED_EXECUTOR_GENERATION_ANCHORS),
    ):
        registry_anchor = _single_identity_anchor(
            [
                anchor
                for anchor in registry
                if type(anchor) is _ExecutorGenerationAnchor
                and anchor.executor_authority is authority
            ]
        )
        if registry_anchor is not None:
            candidates.append(registry_anchor)
    guard = _safe_registry_get(_EXECUTOR_SESSION_GUARDS, authority)
    if type(guard) is _ExecutorSessionIssuerAuthority:
        candidates.append(guard.generation_anchor)
    unique: list[_ExecutorGenerationAnchor] = []
    for candidate in candidates:
        if not any(observed is candidate for observed in unique):
            unique.append(candidate)
    supports = tuple(
        sum(observed is candidate for observed in candidates) for candidate in unique
    )
    if not supports or max(supports) < 2 or supports.count(max(supports)) != 1:
        return None
    winner = unique[supports.index(max(supports))]
    return winner if winner.executor_authority is authority else None


def _executor_session_state_matches_identity_locked(
    authority: RegisteredExecutorSessionAuthority,
    state: _IssuedExecutorSessionState,
) -> bool:
    issuer = state.issuer
    if type(issuer) is not _ExecutorSessionIssuerAuthority:
        return False
    anchor = issuer.generation_anchor
    if (
        type(anchor) is not _ExecutorGenerationAnchor
        or anchor.executor_authority is not authority
    ):
        return False
    supports = 0
    guard = _safe_registry_get(_EXECUTOR_SESSION_GUARDS, authority)
    if type(guard) is _ExecutorSessionIssuerAuthority and (
        _executor_session_issuer_matches_guard(issuer, guard)
    ):
        supports += 1
    if _safe_registry_get(_EXECUTOR_GENERATION_ANCHORS, authority) is anchor:
        supports += 1
    if any(
        observed is anchor
        for observed in _safe_iterable(_EXECUTOR_GENERATION_ANCHOR_HISTORY)
    ):
        supports += 1
    if _safe_contains(_ISSUED_EXECUTOR_GENERATION_ANCHORS, anchor):
        supports += 1
    return supports >= 2


def _invalidate_executor_authority_without_state_locked(
    authority: RegisteredExecutorSessionAuthority,
) -> None:
    """Persist failure by identity when the primary session state is unreadable."""

    try:
        anchor = _trusted_session_anchor_locked(authority, None)
    except BaseException:
        anchor = None
    if anchor is None:
        return
    with suppress(BaseException):
        _FAILED_EXECUTOR_SESSIONS.add(authority)
    latest = anchor.latest_generation
    if latest is None:
        return
    transition_state, _ = _resilient_registry_lookup(
        _ISSUED_TRANSITIONS, latest.transition
    )
    state_matches = type(
        transition_state
    ) is _IssuedTransitionState and _transition_state_matches_identity_locked(
        latest.transition, transition_state
    )
    observed_phase: TransitionPhase | None
    if state_matches:
        assert type(transition_state) is _IssuedTransitionState
        observed_phase = transition_state.phase
    else:
        observed_phase = _safe_registry_get(_TRANSITION_PHASES, latest.transition)
    if observed_phase in {"ISSUED", "TRACE_CONSUMED"}:
        if state_matches:
            assert type(transition_state) is _IssuedTransitionState
            _force_invalidate_transition_corruption_locked(
                latest.transition, transition_state
            )
        else:
            _poison_transition_identity_locked(latest.transition)


def _terminal_fail_executor_session_locked(
    authority: RegisteredExecutorSessionAuthority,
    state: _IssuedExecutorSessionState,
) -> None:
    """Make raw session state terminal before touching fallible mirrors."""

    state.failed = True
    state.active_transition_token = None
    with suppress(BaseException):
        anchor = _trusted_session_anchor_locked(authority, state)
        if anchor is not None:
            latest = anchor.latest_generation
            if latest is not None:
                transition = latest.transition
                transition_state, _ = _resilient_registry_lookup(
                    _ISSUED_TRANSITIONS, transition
                )
                state_matches = (
                    type(transition_state) is _IssuedTransitionState
                    and _transition_state_matches_identity_locked(
                        transition, transition_state
                    )
                )
                observed_phase: TransitionPhase | None
                if state_matches:
                    assert type(transition_state) is _IssuedTransitionState
                    observed_phase = transition_state.phase
                else:
                    observed_phase = _safe_registry_get(_TRANSITION_PHASES, transition)
                if observed_phase in {"ISSUED", "TRACE_CONSUMED"}:
                    if state_matches:
                        assert type(transition_state) is _IssuedTransitionState
                        _force_invalidate_transition_corruption_locked(
                            transition, transition_state
                        )
                    else:
                        _poison_transition_identity_locked(transition)
    with suppress(BaseException):
        _FAILED_EXECUTOR_SESSIONS.add(authority)
    with suppress(BaseException):
        _sync_executor_session_lifecycle_locked(authority, state)


def _require_registered_executor_generation(
    authority: RegisteredExecutorSessionAuthority,
    expected_generation: int,
) -> None:
    """Require exact optimizer continuity before an epoch is materialized."""

    _require_uint32(expected_generation, "expected_generation")
    with _BRIDGE_LOCK:
        state = _lookup_executor_session_or_fail_locked(authority)
        try:
            _validate_executor_session_state(authority, state)
        except BaseException:
            if type(state) is _IssuedExecutorSessionState:
                _terminal_fail_executor_session_locked(authority, state)
            raise
        assert state is not None
        if state.failed:
            raise Experiment002TrainingBridgeError(
                "executor session is terminally failed"
            )
        if (
            state.active_transition_token is not None
            or state.next_optimizer_generation != expected_generation
        ):
            raise Experiment002TrainingBridgeError(
                "executor optimizer generation differs from epoch boundary"
            )


def _issue_registered_optimizer_transition(
    *,
    executor_authority: RegisteredExecutorSessionAuthority,
    epoch_session_token: object,
    batch_token: object,
    zero_based_epoch: int,
    zero_based_global_update: int,
    batch_size: int,
    learning_rate: float,
    batch_mean_training_loss: np.float32,
    returned_preclip_l2_norm: np.float32,
    optimizer_generation_before: int,
    optimizer_generation_after: int,
    model_sha256_before: str,
    model_sha256_after: str,
    optimizer_sha256_before: str,
    optimizer_sha256_after: str,
    rng_sha256_before: str,
    rng_sha256_after: str,
) -> RegisteredOptimizerTransition:
    """Issue the only active transition for the executor's next generation."""

    if type(executor_authority) is not RegisteredExecutorSessionAuthority:
        raise TypeError(
            "executor_authority must be a RegisteredExecutorSessionAuthority"
        )
    with _BRIDGE_LOCK:
        session = _lookup_executor_session_or_fail_locked(executor_authority)
        try:
            _validate_executor_session_state(executor_authority, session)
        except BaseException:
            if type(session) is _IssuedExecutorSessionState:
                _terminal_fail_executor_session_locked(executor_authority, session)
            raise
        assert session is not None
        if session.failed:
            raise Experiment002TrainingBridgeError(
                "executor session is terminally failed"
            )
        if session.active_transition_token is not None:
            raise Experiment002TrainingBridgeError(
                "executor already has an active optimizer transition"
            )
        if optimizer_generation_before != session.next_optimizer_generation:
            _terminal_fail_executor_session_locked(executor_authority, session)
            raise Experiment002TrainingBridgeError(
                "optimizer generation differs from executor continuity"
            )
        try:
            result, state = _prepare_transition(
                executor_authority=executor_authority,
                executor_session_token=session.issuer.session_token,
                route_marker=_REGISTERED_ROUTE,
                seed=session.issuer.seed,
                epoch_session_token=epoch_session_token,
                batch_token=batch_token,
                zero_based_epoch=zero_based_epoch,
                zero_based_global_update=zero_based_global_update,
                batch_size=batch_size,
                learning_rate=learning_rate,
                batch_mean_training_loss=batch_mean_training_loss,
                returned_preclip_l2_norm=returned_preclip_l2_norm,
                optimizer_generation_before=optimizer_generation_before,
                optimizer_generation_after=optimizer_generation_after,
                model_sha256_before=model_sha256_before,
                model_sha256_after=model_sha256_after,
                optimizer_sha256_before=optimizer_sha256_before,
                optimizer_sha256_after=optimizer_sha256_after,
                rng_sha256_before=rng_sha256_before,
                rng_sha256_after=rng_sha256_after,
            )
        except BaseException:
            _terminal_fail_executor_session_locked(executor_authority, session)
            raise
        next_lifecycle = _ExecutorSessionLifecycleGuard(
            next_optimizer_generation=optimizer_generation_after,
            active_transition_token=state.issuer.transition_token,
            failed=False,
        )
        generation_ledger = _EXECUTOR_GENERATION_LEDGERS[executor_authority]
        generation_anchor = _EXECUTOR_GENERATION_ANCHORS[executor_authority]
        previous = generation_anchor.latest_generation
        if previous is not None and (
            previous.model_sha256_after != state.snapshot.model_sha256_before
            or previous.optimizer_sha256_after != state.snapshot.optimizer_sha256_before
            or previous.rng_sha256_after != state.snapshot.rng_sha256_before
        ):
            _terminal_fail_executor_session_locked(executor_authority, session)
            raise Experiment002TrainingBridgeError(
                "model, optimizer, or RNG digest continuity differs"
            )
        next_generation = _ExecutorGenerationAuthority(
            executor_authority=executor_authority,
            transition=result,
            transition_token=state.issuer.transition_token,
            global_history_index=len(_EXECUTOR_GENERATION_HISTORY),
            previous_generation=previous,
            generation_before=optimizer_generation_before,
            generation_after=optimizer_generation_after,
            model_sha256_before=state.snapshot.model_sha256_before,
            model_sha256_after=state.snapshot.model_sha256_after,
            optimizer_sha256_before=(state.snapshot.optimizer_sha256_before),
            optimizer_sha256_after=state.snapshot.optimizer_sha256_after,
            rng_sha256_before=state.snapshot.rng_sha256_before,
            rng_sha256_after=state.snapshot.rng_sha256_after,
        )
        next_generation_ledger = generation_ledger + (next_generation,)
        try:
            _ISSUED_TRANSITIONS[result] = state
            _TRANSITION_GUARDS[result] = replace(state.issuer)
            _TRANSITION_PHASES[result] = "ISSUED"
            _EXECUTOR_SESSION_LIFECYCLES[executor_authority] = next_lifecycle
            _EXECUTOR_GENERATION_LEDGERS[executor_authority] = next_generation_ledger
            # Append only after every replaceable mirror write succeeds. No
            # code path removes or rewrites an authoritative history entry.
            _ISSUED_EXECUTOR_GENERATIONS.add(next_generation)
            _EXECUTOR_GENERATION_HISTORY.append(next_generation)
            generation_anchor._append(next_generation)
        except BaseException:
            _terminal_fail_executor_session_locked(executor_authority, session)
            with suppress(BaseException):
                _ISSUED_TRANSITIONS.pop(result, None)
            with suppress(BaseException):
                _TRANSITION_GUARDS.pop(result, None)
            with suppress(BaseException):
                _TRANSITION_PHASES.pop(result, None)
            raise
        session.next_optimizer_generation = optimizer_generation_after
        session.active_transition_token = state.issuer.transition_token
        return result


def _issue_synthetic_optimizer_transition(
    *,
    seed: int,
    zero_based_epoch: int,
    zero_based_global_update: int,
    batch_size: int,
    learning_rate: float,
    batch_mean_training_loss: np.float32,
    returned_preclip_l2_norm: np.float32,
    optimizer_generation_before: int,
    optimizer_generation_after: int,
    model_sha256_before: str,
    model_sha256_after: str,
    optimizer_sha256_before: str,
    optimizer_sha256_after: str,
    rng_sha256_before: str,
    rng_sha256_after: str,
) -> RegisteredOptimizerTransition:
    """Issue an isolated non-registered transition for tiny executor tests."""

    result, state = _prepare_transition(
        executor_authority=None,
        executor_session_token=object(),
        route_marker=None,
        seed=seed,
        epoch_session_token=object(),
        batch_token=object(),
        zero_based_epoch=zero_based_epoch,
        zero_based_global_update=zero_based_global_update,
        batch_size=batch_size,
        learning_rate=learning_rate,
        batch_mean_training_loss=batch_mean_training_loss,
        returned_preclip_l2_norm=returned_preclip_l2_norm,
        optimizer_generation_before=optimizer_generation_before,
        optimizer_generation_after=optimizer_generation_after,
        model_sha256_before=model_sha256_before,
        model_sha256_after=model_sha256_after,
        optimizer_sha256_before=optimizer_sha256_before,
        optimizer_sha256_after=optimizer_sha256_after,
        rng_sha256_before=rng_sha256_before,
        rng_sha256_after=rng_sha256_after,
    )
    identity_authority = _SyntheticTransitionAuthority(
        transition=result,
        executor_session_token=state.issuer.executor_session_token,
        epoch_session_token=state.issuer.epoch_session_token,
        batch_token=state.issuer.batch_token,
        transition_token=state.issuer.transition_token,
        route_marker=state.issuer.route_marker,
    )
    with _BRIDGE_LOCK:
        try:
            _ISSUED_TRANSITIONS[result] = state
            _TRANSITION_GUARDS[result] = replace(state.issuer)
            _TRANSITION_PHASES[result] = "ISSUED"
            _ISSUED_SYNTHETIC_TRANSITION_AUTHORITIES.add(identity_authority)
            _SYNTHETIC_TRANSITION_AUTHORITY_HISTORY.append(identity_authority)
        except BaseException:
            state.invalid = True
            state.phase = "FAILED"
            with suppress(BaseException):
                _INVALID_TRANSITIONS.add(result)
            with suppress(BaseException):
                _FAILED_TRANSITIONS.add(result)
            raise
    return result


def _prepare_transition(
    *,
    executor_authority: RegisteredExecutorSessionAuthority | None,
    executor_session_token: object,
    route_marker: object | None,
    seed: int,
    epoch_session_token: object,
    batch_token: object,
    zero_based_epoch: int,
    zero_based_global_update: int,
    batch_size: int,
    learning_rate: float,
    batch_mean_training_loss: np.float32,
    returned_preclip_l2_norm: np.float32,
    optimizer_generation_before: int,
    optimizer_generation_after: int,
    model_sha256_before: str,
    model_sha256_after: str,
    optimizer_sha256_before: str,
    optimizer_sha256_after: str,
    rng_sha256_before: str,
    rng_sha256_after: str,
) -> tuple[RegisteredOptimizerTransition, _IssuedTransitionState]:
    transition_token = object()
    snapshot = _OptimizerTransitionSnapshot(
        executor_session_token=executor_session_token,
        epoch_session_token=epoch_session_token,
        batch_token=batch_token,
        transition_token=transition_token,
        seed=seed,
        zero_based_epoch=zero_based_epoch,
        zero_based_global_update=zero_based_global_update,
        batch_size=batch_size,
        learning_rate=learning_rate,
        learning_rate_bytes=struct.pack("<d", learning_rate),
        batch_mean_training_loss=batch_mean_training_loss,
        batch_mean_training_loss_bytes=struct.pack("<f", batch_mean_training_loss),
        returned_preclip_l2_norm=returned_preclip_l2_norm,
        returned_preclip_l2_norm_bytes=struct.pack("<f", returned_preclip_l2_norm),
        optimizer_generation_before=optimizer_generation_before,
        optimizer_generation_after=optimizer_generation_after,
        model_sha256_before=model_sha256_before,
        model_sha256_after=model_sha256_after,
        optimizer_sha256_before=optimizer_sha256_before,
        optimizer_sha256_after=optimizer_sha256_after,
        rng_sha256_before=rng_sha256_before,
        rng_sha256_after=rng_sha256_after,
    )
    _validate_transition_snapshot(snapshot, route_marker=route_marker)
    payload = _frame_transition_authority(snapshot, route_marker=route_marker)
    issuer = _TransitionIssuerAuthority(
        executor_authority=executor_authority,
        executor_session_token=executor_session_token,
        epoch_session_token=epoch_session_token,
        batch_token=batch_token,
        transition_token=transition_token,
        route_marker=route_marker,
        authority_payload=payload,
    )
    result = object.__new__(RegisteredOptimizerTransition)
    return result, _IssuedTransitionState(snapshot=snapshot, issuer=issuer)


def _transition_snapshot(
    transition: RegisteredOptimizerTransition,
    *,
    required_phase: TransitionPhase | None = None,
) -> _OptimizerTransitionSnapshot:
    """Return an immutable copy after validating issuer state and phase."""

    if type(transition) is not RegisteredOptimizerTransition:
        raise TypeError("transition must be a RegisteredOptimizerTransition")
    with _BRIDGE_LOCK:
        state = _lookup_transition_or_fail_locked(transition)
        _validate_transition_or_fail_locked(transition, state)
        assert state is not None
        if required_phase is not None and state.phase != required_phase:
            raise Experiment002TrainingBridgeError(
                f"transition phase must be {required_phase}, observed={state.phase}"
            )
        return replace(state.snapshot)


def _mark_transition_trace_consumed(
    transition: RegisteredOptimizerTransition,
    *,
    receipt_token: object,
    trace_session_token: object,
    trace_record_index: int,
) -> TraceConsumedTransition:
    """Atomically claim one registered transition for one ordered trace."""

    if type(transition) is not RegisteredOptimizerTransition:
        raise TypeError("transition must be a RegisteredOptimizerTransition")
    with _BRIDGE_LOCK:
        state = _lookup_transition_or_fail_locked(transition)
        _validate_transition_or_fail_locked(transition, state)
        assert state is not None
        if state.phase != "ISSUED":
            # A race loser cannot invalidate the winner's legitimate claim.
            raise Experiment002TrainingBridgeError(
                "optimizer transition was already consumed or failed"
            )
        try:
            if state.issuer.route_marker is not _REGISTERED_ROUTE:
                raise Experiment002TrainingBridgeError(
                    "synthetic transitions cannot enter a registered trace"
                )
            if type(receipt_token) is not object:
                raise TypeError("receipt_token must be an exact object")
            if type(trace_session_token) is not object:
                raise TypeError("trace_session_token must be an exact object")
            _require_uint32(trace_record_index, "trace_record_index")
            if trace_record_index != state.snapshot.zero_based_global_update:
                raise Experiment002TrainingBridgeError(
                    "trace record index differs from optimizer transition"
                )
            mark = _TraceMarkAuthority(
                receipt_token, trace_session_token, trace_record_index
            )
            consumption_token = object()
            snapshot = _TraceConsumptionSnapshot(
                executor_session_token=state.issuer.executor_session_token,
                epoch_session_token=state.issuer.epoch_session_token,
                batch_token=state.issuer.batch_token,
                transition_token=state.issuer.transition_token,
                receipt_token=receipt_token,
                trace_session_token=trace_session_token,
                consumption_token=consumption_token,
                zero_based_global_update=(state.snapshot.zero_based_global_update),
                trace_record_index=trace_record_index,
            )
            payload = _frame_consumption_authority(snapshot)
            issuer = _ConsumptionIssuerAuthority(
                transition=transition,
                executor_session_token=snapshot.executor_session_token,
                epoch_session_token=snapshot.epoch_session_token,
                batch_token=snapshot.batch_token,
                transition_token=snapshot.transition_token,
                receipt_token=snapshot.receipt_token,
                trace_session_token=snapshot.trace_session_token,
                consumption_token=snapshot.consumption_token,
                zero_based_global_update=snapshot.zero_based_global_update,
                trace_record_index=snapshot.trace_record_index,
                authority_payload=payload,
            )
            result = object.__new__(TraceConsumedTransition)
            consumption_state = _IssuedConsumptionState(snapshot, issuer)
        except BaseException:
            _terminal_fail_transition_locked(transition, state)
            raise
        try:
            _ISSUED_CONSUMPTIONS[result] = consumption_state
            _CONSUMPTION_GUARDS[result] = replace(issuer)
            _CONSUMPTION_LIFECYCLES[result] = (False, False)
            _TRANSITION_CONSUMPTIONS[transition] = result
            _TRANSITION_CLAIM_GUARDS[transition] = result
            _TRANSITION_PHASES[transition] = "TRACE_CONSUMED"
            _CLAIMED_TRANSITIONS.add(transition)
        except BaseException:
            _terminal_fail_transition_locked(transition, state)
            with suppress(BaseException):
                _ISSUED_CONSUMPTIONS.pop(result, None)
            with suppress(BaseException):
                _CONSUMPTION_GUARDS.pop(result, None)
            with suppress(BaseException):
                _CONSUMPTION_LIFECYCLES.pop(result, None)
            with suppress(BaseException):
                _TRANSITION_CONSUMPTIONS.pop(transition, None)
            with suppress(BaseException):
                _TRANSITION_CLAIM_GUARDS.pop(transition, None)
            raise
        # Every fallible registry write is complete; these assignments cannot
        # fail and expose the claim as one transition-state change.
        state.trace_authority = mark
        state.trace_consumption = result
        state.phase = "TRACE_CONSUMED"
        return result


def _accept_trace_consumed_transition(
    transition: RegisteredOptimizerTransition,
    consumption: TraceConsumedTransition,
    *,
    receipt_token: object,
    executor_authority: RegisteredExecutorSessionAuthority,
    epoch_session_token: object,
    batch_token: object,
) -> None:
    """Atomically spend the exact trace consumption at population acceptance."""

    if type(transition) is not RegisteredOptimizerTransition:
        raise TypeError("transition must be a RegisteredOptimizerTransition")
    if type(consumption) is not TraceConsumedTransition:
        raise TypeError("consumption must be a TraceConsumedTransition")
    with _BRIDGE_LOCK:
        transition_state = _lookup_transition_or_fail_locked(transition)
        _validate_transition_or_fail_locked(transition, transition_state)
        assert transition_state is not None
        if transition_state.phase != "TRACE_CONSUMED":
            # Likewise, an acceptance race loser must not poison a winner.
            raise Experiment002TrainingBridgeError(
                "optimizer transition is not trace-consumed"
            )
        guarded_consumption = _TRANSITION_CONSUMPTIONS.get(transition)
        if (
            guarded_consumption is not consumption
            or _TRANSITION_CLAIM_GUARDS.get(transition) is not consumption
            or transition_state.trace_consumption is not consumption
        ):
            raise Experiment002TrainingBridgeError(
                "trace consumption was not claimed by this transition"
            )
        consumption_state = _lookup_consumption_or_fail_locked(consumption)
        _validate_consumption_or_fail_locked(consumption, consumption_state)
        assert consumption_state is not None
        if consumption_state.used or consumption_state.failed:
            raise Experiment002TrainingBridgeError(
                "trace consumption was already spent or failed"
            )
        try:
            if type(executor_authority) is not RegisteredExecutorSessionAuthority:
                raise TypeError(
                    "executor_authority must be a RegisteredExecutorSessionAuthority"
                )
            session = _lookup_executor_session_or_fail_locked(executor_authority)
            _validate_executor_session_state(executor_authority, session)
            assert session is not None
            if session.failed:
                raise Experiment002TrainingBridgeError(
                    "executor session is terminally failed"
                )
            if type(receipt_token) is not object:
                raise TypeError("receipt_token must be an exact object")
            if type(epoch_session_token) is not object:
                raise TypeError("epoch_session_token must be an exact object")
            if type(batch_token) is not object:
                raise TypeError("batch_token must be an exact object")
            mark = transition_state.trace_authority
            snapshot = consumption_state.snapshot
            if (
                type(mark) is not _TraceMarkAuthority
                or transition_state.issuer.executor_authority is not executor_authority
                or transition_state.issuer.executor_session_token
                is not session.issuer.session_token
                or transition_state.issuer.epoch_session_token
                is not epoch_session_token
                or transition_state.issuer.batch_token is not batch_token
                or transition_state.issuer.transition_token
                is not snapshot.transition_token
                or mark.receipt_token is not receipt_token
                or mark.receipt_token is not snapshot.receipt_token
                or mark.trace_session_token is not snapshot.trace_session_token
                or mark.trace_record_index != snapshot.trace_record_index
                or transition_state.snapshot.zero_based_global_update
                != snapshot.zero_based_global_update
                or snapshot.zero_based_global_update != snapshot.trace_record_index
                or snapshot.executor_session_token is not session.issuer.session_token
                or snapshot.epoch_session_token is not epoch_session_token
                or snapshot.batch_token is not batch_token
                or session.active_transition_token
                is not transition_state.issuer.transition_token
            ):
                raise Experiment002TrainingBridgeError(
                    "trace consumption does not match optimizer transition"
                )
        except BaseException:
            consumption_state.failed = True
            _terminal_fail_transition_locked(transition, transition_state)
            raise
        next_session_lifecycle = _ExecutorSessionLifecycleGuard(
            next_optimizer_generation=session.next_optimizer_generation,
            active_transition_token=None,
            failed=False,
        )
        try:
            _USED_CONSUMPTIONS.add(consumption)
            _CONSUMPTION_LIFECYCLES[consumption] = (True, False)
            _TRANSITION_PHASES[transition] = "POPULATION_ACCEPTED"
            _EXECUTOR_SESSION_LIFECYCLES[executor_authority] = next_session_lifecycle
            _ACCEPTED_TRANSITIONS.add(transition)
        except BaseException:
            consumption_state.failed = True
            _terminal_fail_transition_locked(transition, transition_state)
            with suppress(BaseException):
                _USED_CONSUMPTIONS.discard(consumption)
            raise
        consumption_state.used = True
        transition_state.phase = "POPULATION_ACCEPTED"
        session.active_transition_token = None


def _fail_optimizer_transition(
    transition: RegisteredOptimizerTransition | None,
) -> None:
    """Terminally fail one issued transition unless acceptance already won."""

    if transition is None:
        return
    if type(transition) is not RegisteredOptimizerTransition:
        raise TypeError("transition must be a RegisteredOptimizerTransition")
    with _BRIDGE_LOCK:
        state = _lookup_transition_or_fail_locked(transition)
        _validate_transition_or_fail_locked(transition, state)
        assert state is not None
        if state.phase != "POPULATION_ACCEPTED":
            _terminal_fail_transition_locked(transition, state)


def _trace_consumption_snapshot(
    consumption: TraceConsumedTransition,
    *,
    required_used: bool | None = None,
) -> _TraceConsumptionSnapshot:
    """Return a checked copy of one trace-consumption authority."""

    if type(consumption) is not TraceConsumedTransition:
        raise TypeError("consumption must be a TraceConsumedTransition")
    if required_used is not None and type(required_used) is not bool:
        raise TypeError("required_used must be a bool or None")
    with _BRIDGE_LOCK:
        state = _lookup_consumption_or_fail_locked(consumption)
        _validate_consumption_or_fail_locked(consumption, state)
        assert state is not None
        if state.failed:
            raise Experiment002TrainingBridgeError(
                "trace consumption is terminally failed"
            )
        if required_used is not None and state.used is not required_used:
            raise Experiment002TrainingBridgeError(
                "trace consumption spent state differs"
            )
        return replace(state.snapshot)


def _issued_transition_phase(
    transition: RegisteredOptimizerTransition,
) -> TransitionPhase:
    if type(transition) is not RegisteredOptimizerTransition:
        raise TypeError("transition must be a RegisteredOptimizerTransition")
    with _BRIDGE_LOCK:
        state = _lookup_transition_or_fail_locked(transition)
        _validate_transition_or_fail_locked(transition, state)
        assert state is not None
        return state.phase


def _validate_transition_or_fail_locked(
    transition: RegisteredOptimizerTransition,
    state: _IssuedTransitionState | None,
) -> None:
    try:
        _validate_issued_transition_state(transition, state)
    except BaseException:
        if type(
            state
        ) is _IssuedTransitionState and _transition_state_matches_identity_locked(
            transition, state
        ):
            _force_invalidate_transition_corruption_locked(transition, state)
        else:
            _poison_transition_identity_locked(transition)
        raise


def _validate_consumption_or_fail_locked(
    consumption: TraceConsumedTransition,
    state: _IssuedConsumptionState | None,
) -> None:
    try:
        _validate_consumption_state(consumption, state)
    except BaseException:
        if type(
            state
        ) is _IssuedConsumptionState and _consumption_state_matches_identity_locked(
            consumption, state
        ):
            _force_invalidate_consumption_corruption_locked(consumption, state)
        else:
            _poison_consumption_identity_locked(consumption)
        raise


def _single_identity_authority(
    candidates: list[RegisteredExecutorSessionAuthority],
) -> RegisteredExecutorSessionAuthority | None:
    unique: list[RegisteredExecutorSessionAuthority] = []
    for candidate in candidates:
        if not any(observed is candidate for observed in unique):
            unique.append(candidate)
    return unique[0] if len(unique) == 1 else None


def _consensus_authorities(
    candidates: list[RegisteredExecutorSessionAuthority],
) -> tuple[RegisteredExecutorSessionAuthority, ...]:
    unique: list[RegisteredExecutorSessionAuthority] = []
    for candidate in candidates:
        if not any(observed is candidate for observed in unique):
            unique.append(candidate)
    supports = tuple(
        sum(observed is candidate for observed in candidates) for candidate in unique
    )
    if not supports or max(supports) < 2 or supports.count(max(supports)) != 1:
        return ()
    return (unique[supports.index(max(supports))],)


def _anchor_contains_transition(
    anchor: _ExecutorGenerationAnchor,
    transition: RegisteredOptimizerTransition,
) -> bool:
    current = anchor.latest_generation
    expected_after = anchor.generation_count
    while current is not None and expected_after > 0:
        if (
            current.executor_authority is not anchor.executor_authority
            or current.generation_after != expected_after
            or current.generation_before != expected_after - 1
        ):
            return False
        if current.transition is transition:
            return True
        current = current.previous_generation
        expected_after -= 1
    return False


def _trusted_transition_owners_locked(
    transition: RegisteredOptimizerTransition,
    state: _IssuedTransitionState | None,
) -> tuple[RegisteredExecutorSessionAuthority, ...]:
    owner_candidates: list[RegisteredExecutorSessionAuthority] = []

    list_owners = _single_identity_authority(
        [
            generation.executor_authority
            for generation in _safe_iterable(_EXECUTOR_GENERATION_HISTORY)
            if type(generation) is _ExecutorGenerationAuthority
            and generation.transition is transition
            and type(generation.executor_authority)
            is RegisteredExecutorSessionAuthority
        ]
    )
    if list_owners is not None:
        owner_candidates.append(list_owners)

    set_owner = _single_identity_authority(
        [
            generation.executor_authority
            for generation in _safe_iterable(_ISSUED_EXECUTOR_GENERATIONS)
            if type(generation) is _ExecutorGenerationAuthority
            and generation.transition is transition
            and type(generation.executor_authority)
            is RegisteredExecutorSessionAuthority
        ]
    )
    if set_owner is not None:
        owner_candidates.append(set_owner)

    anchor_registries = (
        _safe_iterable(_EXECUTOR_GENERATION_ANCHOR_HISTORY),
        _safe_iterable(_ISSUED_EXECUTOR_GENERATION_ANCHORS),
        _safe_registry_values(_EXECUTOR_GENERATION_ANCHORS),
    )
    anchors: list[_ExecutorGenerationAnchor] = []
    for registry in anchor_registries:
        for anchor in registry:
            if type(anchor) is _ExecutorGenerationAnchor and not any(
                observed is anchor for observed in anchors
            ):
                anchors.append(anchor)
    anchor_owner = _single_identity_authority(
        [
            anchor.executor_authority
            for anchor in anchors
            if sum(
                any(observed is anchor for observed in registry)
                for registry in anchor_registries
            )
            >= 2
            and _anchor_contains_transition(anchor, transition)
        ]
    )
    if anchor_owner is not None:
        owner_candidates.append(anchor_owner)

    if type(state) is _IssuedTransitionState:
        raw_owner = state.issuer.executor_authority
        if type(raw_owner) is RegisteredExecutorSessionAuthority:
            owner_candidates.append(raw_owner)
    guard = _safe_registry_get(_TRANSITION_GUARDS, transition)
    if (
        type(guard) is _TransitionIssuerAuthority
        and type(guard.executor_authority) is RegisteredExecutorSessionAuthority
    ):
        owner_candidates.append(guard.executor_authority)
    return _consensus_authorities(owner_candidates)


def _generation_matches_transition_state(
    generation: object,
    transition: RegisteredOptimizerTransition,
    state: _IssuedTransitionState,
) -> bool:
    issuer = state.issuer
    snapshot = state.snapshot
    return (
        type(generation) is _ExecutorGenerationAuthority
        and type(issuer) is _TransitionIssuerAuthority
        and type(snapshot) is _OptimizerTransitionSnapshot
        and generation.transition is transition
        and (
            generation.transition_token is issuer.transition_token
            or generation.transition_token is snapshot.transition_token
        )
    )


def _transition_issuer_identity_matches_guard(
    issuer: _TransitionIssuerAuthority,
    guard: _TransitionIssuerAuthority,
) -> bool:
    return (
        type(issuer) is _TransitionIssuerAuthority
        and type(guard) is _TransitionIssuerAuthority
        and issuer.executor_authority is guard.executor_authority
        and issuer.executor_session_token is guard.executor_session_token
        and issuer.epoch_session_token is guard.epoch_session_token
        and issuer.batch_token is guard.batch_token
        and issuer.transition_token is guard.transition_token
        and issuer.route_marker is guard.route_marker
    )


def _synthetic_authority_matches_transition_state(
    authority: object,
    transition: RegisteredOptimizerTransition,
    state: _IssuedTransitionState,
) -> bool:
    issuer = state.issuer
    return (
        type(authority) is _SyntheticTransitionAuthority
        and type(issuer) is _TransitionIssuerAuthority
        and authority.transition is transition
        and authority.executor_session_token is issuer.executor_session_token
        and authority.epoch_session_token is issuer.epoch_session_token
        and authority.batch_token is issuer.batch_token
        and authority.transition_token is issuer.transition_token
        and authority.route_marker is issuer.route_marker
    )


def _transition_identity_is_recognized_locked(
    transition: RegisteredOptimizerTransition,
) -> bool:
    if type(_safe_registry_get(_TRANSITION_GUARDS, transition)) is (
        _TransitionIssuerAuthority
    ):
        return True
    if any(
        type(authority) is _SyntheticTransitionAuthority
        and authority.transition is transition
        for authority in _safe_iterable(_SYNTHETIC_TRANSITION_AUTHORITY_HISTORY)
    ):
        return True
    if any(
        type(authority) is _SyntheticTransitionAuthority
        and authority.transition is transition
        for authority in _safe_iterable(_ISSUED_SYNTHETIC_TRANSITION_AUTHORITIES)
    ):
        return True
    return any(
        type(generation) is _ExecutorGenerationAuthority
        and generation.transition is transition
        for generation in _safe_iterable(_EXECUTOR_GENERATION_HISTORY)
    ) or any(
        type(generation) is _ExecutorGenerationAuthority
        and generation.transition is transition
        for generation in _safe_iterable(_ISSUED_EXECUTOR_GENERATIONS)
    )


def _transition_state_matches_identity_locked(
    transition: RegisteredOptimizerTransition,
    state: _IssuedTransitionState,
) -> bool:
    issuer = state.issuer
    if type(issuer) is not _TransitionIssuerAuthority:
        return False
    supports = 0
    guard = _safe_registry_get(_TRANSITION_GUARDS, transition)
    if type(guard) is _TransitionIssuerAuthority and (
        _transition_issuer_identity_matches_guard(issuer, guard)
    ):
        supports += 1
    if any(
        _generation_matches_transition_state(generation, transition, state)
        for generation in _safe_iterable(_EXECUTOR_GENERATION_HISTORY)
    ):
        supports += 1
    if any(
        _generation_matches_transition_state(generation, transition, state)
        for generation in _safe_iterable(_ISSUED_EXECUTOR_GENERATIONS)
    ):
        supports += 1
    if any(
        _synthetic_authority_matches_transition_state(authority, transition, state)
        for authority in _safe_iterable(_SYNTHETIC_TRANSITION_AUTHORITY_HISTORY)
    ):
        supports += 1
    if any(
        _synthetic_authority_matches_transition_state(authority, transition, state)
        for authority in _safe_iterable(_ISSUED_SYNTHETIC_TRANSITION_AUTHORITIES)
    ):
        supports += 1
    return supports >= 2


def _poison_transition_identity_locked(
    transition: RegisteredOptimizerTransition,
) -> None:
    """Invalidate a transition and its owner even if its primary row is unreadable."""

    transition_state, _ = _resilient_registry_lookup(_ISSUED_TRANSITIONS, transition)
    state_matches = type(
        transition_state
    ) is _IssuedTransitionState and _transition_state_matches_identity_locked(
        transition, transition_state
    )
    if not state_matches and not _transition_identity_is_recognized_locked(transition):
        return
    with suppress(BaseException):
        _INVALID_TRANSITIONS.add(transition)
    if state_matches:
        assert type(transition_state) is _IssuedTransitionState
        _force_invalidate_transition_corruption_locked(transition, transition_state)
        return
    try:
        authorities = _trusted_transition_owners_locked(transition, None)
    except BaseException:
        authorities = ()
    for authority in authorities:
        with suppress(BaseException):
            _FAILED_EXECUTOR_SESSIONS.add(authority)
        session, _ = _resilient_registry_lookup(_EXECUTOR_SESSIONS, authority)
        if (
            type(session) is _IssuedExecutorSessionState
            and _executor_session_state_matches_identity_locked(authority, session)
        ):
            if session.failed and session.active_transition_token is None:
                with suppress(BaseException):
                    _sync_executor_session_lifecycle_locked(authority, session)
            else:
                _terminal_fail_executor_session_locked(authority, session)
        else:
            _invalidate_executor_authority_without_state_locked(authority)


def _consensus_bound_consumption_locked(
    transition: RegisteredOptimizerTransition,
    state: _IssuedTransitionState,
) -> TraceConsumedTransition | None:
    candidates = [
        candidate
        for candidate in (
            state.trace_consumption,
            _safe_registry_get(_TRANSITION_CONSUMPTIONS, transition),
            _safe_registry_get(_TRANSITION_CLAIM_GUARDS, transition),
        )
        if type(candidate) is TraceConsumedTransition
    ]
    unique: list[TraceConsumedTransition] = []
    for candidate in candidates:
        if not any(observed is candidate for observed in unique):
            unique.append(candidate)
    for candidate in unique:
        consumption_state = _safe_registry_get(_ISSUED_CONSUMPTIONS, candidate)
        if (
            type(consumption_state) is _IssuedConsumptionState
            and _consumption_state_matches_identity_locked(candidate, consumption_state)
            and consumption_state.issuer.transition is transition
        ):
            candidates.append(candidate)
        guard = _safe_registry_get(_CONSUMPTION_GUARDS, candidate)
        if (
            type(guard) is _ConsumptionIssuerAuthority
            and guard.transition is transition
        ):
            candidates.append(candidate)
    supports = tuple(
        sum(observed is candidate for observed in candidates) for candidate in unique
    )
    if not supports or max(supports) < 2 or supports.count(max(supports)) != 1:
        return None
    return unique[supports.index(max(supports))]


def _force_invalidate_transition_corruption_locked(
    transition: RegisteredOptimizerTransition,
    state: _IssuedTransitionState,
) -> None:
    state.invalid = True
    accepted = state.phase == "POPULATION_ACCEPTED" or _safe_contains(
        _ACCEPTED_TRANSITIONS, transition
    )
    if accepted:
        state.phase = "POPULATION_ACCEPTED"
    try:
        guarded_consumption = _consensus_bound_consumption_locked(transition, state)
    except BaseException:
        guarded_consumption = None
    consumption_state = None
    if guarded_consumption is not None:
        consumption_state, _ = _resilient_registry_lookup(
            _ISSUED_CONSUMPTIONS, guarded_consumption
        )
    consumption_state_matches = (
        guarded_consumption is not None
        and type(consumption_state) is _IssuedConsumptionState
        and _consumption_state_matches_identity_locked(
            guarded_consumption, consumption_state
        )
    )
    if not accepted:
        state.phase = "FAILED"
    if consumption_state_matches:
        assert type(consumption_state) is _IssuedConsumptionState
        consumption_state.invalid = True
        committed_used = consumption_state.used or _safe_contains(
            _USED_CONSUMPTIONS, guarded_consumption
        )
        if committed_used:
            consumption_state.used = True
            consumption_state.failed = False
        else:
            consumption_state.failed = True
    with suppress(BaseException):
        _INVALID_TRANSITIONS.add(transition)
    if not accepted:
        with suppress(BaseException):
            _FAILED_TRANSITIONS.add(transition)
        with suppress(BaseException):
            _TRANSITION_PHASES[transition] = "FAILED"
    else:
        with suppress(BaseException):
            _TRANSITION_PHASES[transition] = "POPULATION_ACCEPTED"
    if guarded_consumption is not None:
        if consumption_state_matches or _consumption_identity_is_recognized_locked(
            guarded_consumption
        ):
            with suppress(BaseException):
                _INVALID_CONSUMPTIONS.add(guarded_consumption)
        if consumption_state_matches:
            assert type(consumption_state) is _IssuedConsumptionState
            if not consumption_state.used:
                with suppress(BaseException):
                    _CONSUMPTION_LIFECYCLES[guarded_consumption] = (False, True)
                with suppress(BaseException):
                    _FAILED_CONSUMPTIONS.add(guarded_consumption)
    try:
        authorities = _trusted_transition_owners_locked(transition, state)
    except BaseException:
        authorities = ()
    for authority in authorities:
        session, _ = _resilient_registry_lookup(_EXECUTOR_SESSIONS, authority)
        if (
            type(session) is _IssuedExecutorSessionState
            and _executor_session_state_matches_identity_locked(authority, session)
        ):
            if session.failed and session.active_transition_token is None:
                with suppress(BaseException):
                    _FAILED_EXECUTOR_SESSIONS.add(authority)
                with suppress(BaseException):
                    _sync_executor_session_lifecycle_locked(authority, session)
            else:
                _terminal_fail_executor_session_locked(authority, session)
        else:
            _invalidate_executor_authority_without_state_locked(authority)


def _single_identity_transition(
    candidates: list[RegisteredOptimizerTransition],
) -> RegisteredOptimizerTransition | None:
    unique: list[RegisteredOptimizerTransition] = []
    for candidate in candidates:
        if not any(observed is candidate for observed in unique):
            unique.append(candidate)
    return unique[0] if len(unique) == 1 else None


def _consensus_consumption_transitions_locked(
    consumption: TraceConsumedTransition,
    state: _IssuedConsumptionState | None,
) -> tuple[RegisteredOptimizerTransition, ...]:
    candidates: list[RegisteredOptimizerTransition] = []
    if type(state) is _IssuedConsumptionState:
        raw_transition = state.issuer.transition
        if type(raw_transition) is RegisteredOptimizerTransition:
            candidates.append(raw_transition)
    guard = _safe_registry_get(_CONSUMPTION_GUARDS, consumption)
    if type(guard) is _ConsumptionIssuerAuthority:
        candidates.append(guard.transition)
    for registry in (_TRANSITION_CONSUMPTIONS, _TRANSITION_CLAIM_GUARDS):
        reverse = _single_identity_transition(
            [
                transition
                for transition, observed in _safe_registry_items(registry)
                if observed is consumption
            ]
        )
        if reverse is not None:
            candidates.append(reverse)
    state_link = _single_identity_transition(
        [
            transition
            for transition, transition_state in _safe_registry_items(
                _ISSUED_TRANSITIONS
            )
            if type(transition_state) is _IssuedTransitionState
            and transition_state.trace_consumption is consumption
        ]
    )
    if state_link is not None:
        candidates.append(state_link)
    unique: list[RegisteredOptimizerTransition] = []
    for candidate in candidates:
        if not any(observed is candidate for observed in unique):
            unique.append(candidate)
    supports = tuple(
        sum(observed is candidate for observed in candidates) for candidate in unique
    )
    if not supports or max(supports) < 2 or supports.count(max(supports)) != 1:
        return ()
    return (unique[supports.index(max(supports))],)


def _consumption_identity_is_recognized_locked(
    consumption: TraceConsumedTransition,
) -> bool:
    if type(_safe_registry_get(_CONSUMPTION_GUARDS, consumption)) is (
        _ConsumptionIssuerAuthority
    ):
        return True
    if any(
        observed is consumption
        for registry in (_TRANSITION_CONSUMPTIONS, _TRANSITION_CLAIM_GUARDS)
        for observed in _safe_registry_values(registry)
    ):
        return True
    return any(
        type(state) is _IssuedTransitionState and state.trace_consumption is consumption
        for state in _safe_registry_values(_ISSUED_TRANSITIONS)
    )


def _consumption_state_matches_identity_locked(
    consumption: TraceConsumedTransition,
    state: _IssuedConsumptionState,
) -> bool:
    issuer = state.issuer
    if type(issuer) is not _ConsumptionIssuerAuthority:
        return False
    supports = 0
    guard = _safe_registry_get(_CONSUMPTION_GUARDS, consumption)
    if type(guard) is _ConsumptionIssuerAuthority and (
        _consumption_issuer_matches_guard(issuer, guard)
    ):
        supports += 1
    for registry in (_TRANSITION_CONSUMPTIONS, _TRANSITION_CLAIM_GUARDS):
        if _safe_registry_get(registry, issuer.transition) is consumption:
            supports += 1
    transition_state, _ = _resilient_registry_lookup(
        _ISSUED_TRANSITIONS, issuer.transition
    )
    if (
        type(transition_state) is _IssuedTransitionState
        and transition_state.trace_consumption is consumption
        and transition_state.issuer.transition_token is issuer.transition_token
    ):
        supports += 1
    return supports >= 2


def _poison_consumption_identity_locked(
    consumption: TraceConsumedTransition,
) -> None:
    """Invalidate a consumption and propagate through independent bindings."""

    consumption_state, _ = _resilient_registry_lookup(_ISSUED_CONSUMPTIONS, consumption)
    state_matches = type(
        consumption_state
    ) is _IssuedConsumptionState and _consumption_state_matches_identity_locked(
        consumption, consumption_state
    )
    if not state_matches and not _consumption_identity_is_recognized_locked(
        consumption
    ):
        return
    with suppress(BaseException):
        _INVALID_CONSUMPTIONS.add(consumption)
    if state_matches:
        assert type(consumption_state) is _IssuedConsumptionState
        _force_invalidate_consumption_corruption_locked(consumption, consumption_state)
        return
    try:
        transitions = _consensus_consumption_transitions_locked(consumption, None)
    except BaseException:
        transitions = ()
    for transition in transitions:
        _poison_transition_identity_locked(transition)


def _force_invalidate_consumption_corruption_locked(
    consumption: TraceConsumedTransition,
    state: _IssuedConsumptionState,
) -> None:
    state.invalid = True
    committed_used = state.used or _safe_contains(_USED_CONSUMPTIONS, consumption)
    if committed_used:
        state.used = True
        state.failed = False
    else:
        state.failed = True
    with suppress(BaseException):
        _INVALID_CONSUMPTIONS.add(consumption)
    if not committed_used:
        with suppress(BaseException):
            _CONSUMPTION_LIFECYCLES[consumption] = (False, True)
        with suppress(BaseException):
            _FAILED_CONSUMPTIONS.add(consumption)
    try:
        transitions = _consensus_consumption_transitions_locked(consumption, state)
    except BaseException:
        transitions = ()
    for transition in transitions:
        _poison_transition_identity_locked(transition)


def _terminal_fail_transition_locked(
    transition: RegisteredOptimizerTransition,
    state: _IssuedTransitionState,
) -> None:
    if state.phase == "POPULATION_ACCEPTED":
        return
    # Raw lifecycle assignments are deliberately first: even if an injected
    # registry write fails persistently, no retry can observe a live state.
    state.phase = "FAILED"
    consumption = state.trace_consumption
    consumption_state: _IssuedConsumptionState | None = None
    consumption_state_matches = False
    if consumption is not None:
        consumption_state, _ = _resilient_registry_lookup(
            _ISSUED_CONSUMPTIONS, consumption
        )
        consumption_state_matches = type(
            consumption_state
        ) is _IssuedConsumptionState and _consumption_state_matches_identity_locked(
            consumption, consumption_state
        )
        if consumption_state_matches:
            assert type(consumption_state) is _IssuedConsumptionState
            if not consumption_state.used:
                consumption_state.failed = True
        elif _consumption_identity_is_recognized_locked(consumption):
            with suppress(BaseException):
                _INVALID_CONSUMPTIONS.add(consumption)
    authority = state.issuer.executor_authority
    session: _IssuedExecutorSessionState | None = None
    session_state_matches = False
    if authority is not None:
        session, _ = _resilient_registry_lookup(_EXECUTOR_SESSIONS, authority)
        session_state_matches = (
            type(session) is _IssuedExecutorSessionState
            and _executor_session_state_matches_identity_locked(authority, session)
        )
        if session_state_matches:
            assert type(session) is _IssuedExecutorSessionState
            session.failed = True
            if session.active_transition_token is state.issuer.transition_token:
                session.active_transition_token = None
        else:
            _invalidate_executor_authority_without_state_locked(authority)
    with suppress(BaseException):
        _FAILED_TRANSITIONS.add(transition)
    with suppress(BaseException):
        _TRANSITION_PHASES[transition] = "FAILED"
    if consumption is not None and consumption_state_matches:
        with suppress(BaseException):
            _CONSUMPTION_LIFECYCLES[consumption] = (False, True)
        with suppress(BaseException):
            _FAILED_CONSUMPTIONS.add(consumption)
    if authority is not None and session_state_matches:
        with suppress(BaseException):
            _FAILED_EXECUTOR_SESSIONS.add(authority)
        assert type(session) is _IssuedExecutorSessionState
        with suppress(BaseException):
            _sync_executor_session_lifecycle_locked(authority, session)


def _sync_executor_session_lifecycle_locked(
    authority: RegisteredExecutorSessionAuthority,
    state: _IssuedExecutorSessionState,
) -> None:
    _EXECUTOR_SESSION_LIFECYCLES[authority] = _ExecutorSessionLifecycleGuard(
        next_optimizer_generation=state.next_optimizer_generation,
        active_transition_token=state.active_transition_token,
        failed=state.failed,
    )


def _executor_session_issuer_matches_guard(
    issuer: _ExecutorSessionIssuerAuthority,
    guard: _ExecutorSessionIssuerAuthority,
) -> bool:
    return (
        type(issuer) is _ExecutorSessionIssuerAuthority
        and type(guard) is _ExecutorSessionIssuerAuthority
        and issuer.session_token is guard.session_token
        and issuer.seed == guard.seed
        and issuer.route_marker is guard.route_marker
        and issuer.authority_payload == guard.authority_payload
        and issuer.generation_anchor is guard.generation_anchor
    )


def _transition_issuer_matches_guard(
    issuer: _TransitionIssuerAuthority,
    guard: _TransitionIssuerAuthority,
) -> bool:
    return (
        type(issuer) is _TransitionIssuerAuthority
        and type(guard) is _TransitionIssuerAuthority
        and issuer.executor_authority is guard.executor_authority
        and issuer.executor_session_token is guard.executor_session_token
        and issuer.epoch_session_token is guard.epoch_session_token
        and issuer.batch_token is guard.batch_token
        and issuer.transition_token is guard.transition_token
        and issuer.route_marker is guard.route_marker
        and issuer.authority_payload == guard.authority_payload
    )


def _consumption_issuer_matches_guard(
    issuer: _ConsumptionIssuerAuthority,
    guard: _ConsumptionIssuerAuthority,
) -> bool:
    return (
        type(issuer) is _ConsumptionIssuerAuthority
        and type(guard) is _ConsumptionIssuerAuthority
        and issuer.transition is guard.transition
        and issuer.executor_session_token is guard.executor_session_token
        and issuer.epoch_session_token is guard.epoch_session_token
        and issuer.batch_token is guard.batch_token
        and issuer.transition_token is guard.transition_token
        and issuer.receipt_token is guard.receipt_token
        and issuer.trace_session_token is guard.trace_session_token
        and issuer.consumption_token is guard.consumption_token
        and issuer.zero_based_global_update == guard.zero_based_global_update
        and issuer.trace_record_index == guard.trace_record_index
        and issuer.authority_payload == guard.authority_payload
    )


def _validate_executor_session_state(
    authority: RegisteredExecutorSessionAuthority,
    state: _IssuedExecutorSessionState | None,
) -> None:
    if type(state) is not _IssuedExecutorSessionState:
        raise Experiment002TrainingBridgeError(
            "executor session authority was not issued by this process"
        )
    snapshot = state.snapshot
    issuer = state.issuer
    guard = _EXECUTOR_SESSION_GUARDS.get(authority)
    lifecycle = _EXECUTOR_SESSION_LIFECYCLES.get(authority)
    generation_ledger = _EXECUTOR_GENERATION_LEDGERS.get(authority)
    generation_anchor = issuer.generation_anchor
    generation_count = (
        generation_anchor.generation_count
        if type(generation_anchor) is _ExecutorGenerationAnchor
        else -1
    )
    latest_generation = (
        generation_anchor.latest_generation
        if type(generation_anchor) is _ExecutorGenerationAnchor
        else None
    )
    if (
        type(guard) is not _ExecutorSessionIssuerAuthority
        or not _executor_session_issuer_matches_guard(issuer, guard)
        or type(lifecycle) is not _ExecutorSessionLifecycleGuard
        or type(generation_ledger) is not tuple
        or type(generation_anchor) is not _ExecutorGenerationAnchor
        or generation_anchor.executor_authority is not authority
        or generation_anchor.seed != issuer.seed
        or _EXECUTOR_GENERATION_ANCHORS.get(authority) is not generation_anchor
        or generation_anchor not in _ISSUED_EXECUTOR_GENERATION_ANCHORS
        or generation_anchor not in _EXECUTOR_GENERATION_ANCHOR_HISTORY
        or len(generation_ledger) != generation_count
        or (generation_count == 0 and latest_generation is not None)
        or (generation_count > 0 and generation_ledger[-1] is not latest_generation)
        or type(snapshot) is not _ExecutorSessionSnapshot
        or type(issuer) is not _ExecutorSessionIssuerAuthority
        or type(snapshot.session_token) is not object
        or snapshot.session_token is not issuer.session_token
        or type(snapshot.seed) is not int
        or snapshot.seed != issuer.seed
        or snapshot.route_marker is not _REGISTERED_ROUTE
        or snapshot.route_marker is not issuer.route_marker
        or type(issuer.authority_payload) is not bytes
        or issuer.authority_payload != struct.pack("<I", issuer.seed)
        or type(state.next_optimizer_generation) is not int
        or not 0 <= state.next_optimizer_generation <= _TOTAL_UPDATES
        or (
            state.active_transition_token is not None
            and type(state.active_transition_token) is not object
        )
        or type(state.failed) is not bool
        or (state.failed and state.active_transition_token is not None)
        or state.next_optimizer_generation != lifecycle.next_optimizer_generation
        or state.active_transition_token is not lifecycle.active_transition_token
        or state.failed is not lifecycle.failed
        or state.failed is not (authority in _FAILED_EXECUTOR_SESSIONS)
        or len(generation_ledger) != state.next_optimizer_generation
    ):
        raise Experiment002TrainingBridgeError(
            "executor session differs from issuer authority"
        )
    _require_registered_seed(issuer.seed)
    latest_state: _IssuedTransitionState | None = None
    if latest_generation is not None:
        generation = latest_generation
        expected_generation = state.next_optimizer_generation - 1
        if (
            type(generation) is not _ExecutorGenerationAuthority
            or generation.executor_authority is not authority
            or type(generation.transition) is not RegisteredOptimizerTransition
            or type(generation.transition_token) is not object
            or type(generation.global_history_index) is not int
            or not 0
            <= generation.global_history_index
            < len(_EXECUTOR_GENERATION_HISTORY)
            or _EXECUTOR_GENERATION_HISTORY[generation.global_history_index]
            is not generation
            or generation not in _ISSUED_EXECUTOR_GENERATIONS
            or generation.generation_before != expected_generation
            or generation.generation_after != expected_generation + 1
            or (expected_generation == 0 and generation.previous_generation is not None)
            or (
                expected_generation > 0
                and type(generation.previous_generation)
                is not _ExecutorGenerationAuthority
            )
            or not _is_sha256(generation.model_sha256_before)
            or not _is_sha256(generation.model_sha256_after)
            or not _is_sha256(generation.optimizer_sha256_before)
            or not _is_sha256(generation.optimizer_sha256_after)
            or not _is_sha256(generation.rng_sha256_before)
            or not _is_sha256(generation.rng_sha256_after)
        ):
            raise Experiment002TrainingBridgeError(
                "executor generation ledger is invalid"
            )
        latest_state = _ISSUED_TRANSITIONS.get(generation.transition)
        if (
            type(latest_state) is not _IssuedTransitionState
            or latest_state.invalid
            or latest_state.issuer.executor_authority is not authority
            or latest_state.issuer.executor_session_token is not issuer.session_token
            or latest_state.issuer.transition_token is not generation.transition_token
            or latest_state.snapshot.optimizer_generation_before != expected_generation
            or latest_state.snapshot.optimizer_generation_after
            != expected_generation + 1
            or latest_state.snapshot.zero_based_global_update != expected_generation
            or latest_state.snapshot.model_sha256_before
            != generation.model_sha256_before
            or latest_state.snapshot.model_sha256_after != generation.model_sha256_after
            or latest_state.snapshot.optimizer_sha256_before
            != generation.optimizer_sha256_before
            or latest_state.snapshot.optimizer_sha256_after
            != generation.optimizer_sha256_after
            or latest_state.snapshot.rng_sha256_before != generation.rng_sha256_before
            or latest_state.snapshot.rng_sha256_after != generation.rng_sha256_after
        ):
            raise Experiment002TrainingBridgeError(
                "executor generation differs from append-only ledger"
            )
        if expected_generation > 0:
            previous = generation.previous_generation
            if (
                type(previous) is not _ExecutorGenerationAuthority
                or previous.executor_authority is not authority
                or previous.generation_after != expected_generation
                or generation.model_sha256_before != previous.model_sha256_after
                or generation.optimizer_sha256_before != previous.optimizer_sha256_after
                or generation.rng_sha256_before != previous.rng_sha256_after
            ):
                raise Experiment002TrainingBridgeError(
                    "executor digest chain differs from append-only ledger"
                )
    if state.active_transition_token is None:
        active_differs = latest_state is not None and latest_state.phase in {
            "ISSUED",
            "TRACE_CONSUMED",
        }
    else:
        active_differs = (
            latest_generation is None
            or latest_state is None
            or latest_generation.transition_token is not state.active_transition_token
            or latest_state.issuer.transition_token is not state.active_transition_token
            or latest_state.phase not in {"ISSUED", "TRACE_CONSUMED"}
            or _TRANSITION_PHASES.get(latest_generation.transition)
            != latest_state.phase
            or latest_generation.transition in _ACCEPTED_TRANSITIONS
            or latest_generation.transition in _FAILED_TRANSITIONS
        )
    if active_differs:
        raise Experiment002TrainingBridgeError(
            "executor active transition differs from append-only authority"
        )


def _validate_issued_transition_state(
    transition: RegisteredOptimizerTransition,
    state: _IssuedTransitionState | None,
) -> None:
    if type(state) is not _IssuedTransitionState:
        raise Experiment002TrainingBridgeError(
            "optimizer transition was not issued by this process"
        )
    snapshot = state.snapshot
    issuer = state.issuer
    guard = _TRANSITION_GUARDS.get(transition)
    guarded_phase = _TRANSITION_PHASES.get(transition)
    guarded_consumption = _TRANSITION_CONSUMPTIONS.get(transition)
    claim_guard = _TRANSITION_CLAIM_GUARDS.get(transition)
    claimed = transition in _CLAIMED_TRANSITIONS
    accepted = transition in _ACCEPTED_TRANSITIONS
    failed = transition in _FAILED_TRANSITIONS
    if (
        type(guard) is not _TransitionIssuerAuthority
        or type(state.invalid) is not bool
        or state.invalid is not (transition in _INVALID_TRANSITIONS)
        or state.invalid
        or not _transition_issuer_matches_guard(issuer, guard)
        or guarded_phase
        not in {"ISSUED", "TRACE_CONSUMED", "POPULATION_ACCEPTED", "FAILED"}
        or state.phase != guarded_phase
        or type(issuer) is not _TransitionIssuerAuthority
        or type(issuer.executor_session_token) is not object
        or type(issuer.epoch_session_token) is not object
        or type(issuer.batch_token) is not object
        or type(issuer.transition_token) is not object
        or issuer.route_marker not in {None, _REGISTERED_ROUTE}
        or type(issuer.authority_payload) is not bytes
        or state.phase
        not in {"ISSUED", "TRACE_CONSUMED", "POPULATION_ACCEPTED", "FAILED"}
        or (state.phase == "ISSUED" and (claimed or accepted or failed))
        or (state.phase == "TRACE_CONSUMED" and (not claimed or accepted or failed))
        or (
            state.phase == "POPULATION_ACCEPTED"
            and (not claimed or not accepted or failed)
        )
        or (state.phase == "FAILED" and (not failed or accepted))
    ):
        raise Experiment002TrainingBridgeError(
            "optimizer transition issuer state is invalid"
        )
    _validate_transition_snapshot(snapshot, route_marker=issuer.route_marker)
    if (
        snapshot.executor_session_token is not issuer.executor_session_token
        or snapshot.epoch_session_token is not issuer.epoch_session_token
        or snapshot.batch_token is not issuer.batch_token
        or snapshot.transition_token is not issuer.transition_token
        or _frame_transition_authority(snapshot, route_marker=issuer.route_marker)
        != issuer.authority_payload
    ):
        raise Experiment002TrainingBridgeError(
            "optimizer transition differs from issuer authority"
        )
    if issuer.route_marker is _REGISTERED_ROUTE:
        if type(issuer.executor_authority) is not RegisteredExecutorSessionAuthority:
            raise Experiment002TrainingBridgeError(
                "registered transition lost executor authority"
            )
        session = _EXECUTOR_SESSIONS.get(issuer.executor_authority)
        if state.phase == "POPULATION_ACCEPTED":
            if type(session) is not _IssuedExecutorSessionState or not (
                _executor_session_state_matches_identity_locked(
                    issuer.executor_authority, session
                )
            ):
                raise Experiment002TrainingBridgeError(
                    "accepted transition lost executor identity"
                )
            session_guard = _safe_registry_get(
                _EXECUTOR_SESSION_GUARDS, issuer.executor_authority
            )
            anchor = _trusted_session_anchor_locked(issuer.executor_authority, session)
            if (
                type(session_guard) is not _ExecutorSessionIssuerAuthority
                or not _executor_session_issuer_matches_guard(
                    session.issuer, session_guard
                )
                or anchor is None
                or not _anchor_contains_transition(anchor, transition)
                or not any(
                    _generation_matches_transition_state(generation, transition, state)
                    for generation in _safe_iterable(_EXECUTOR_GENERATION_HISTORY)
                )
                or not any(
                    _generation_matches_transition_state(generation, transition, state)
                    for generation in _safe_iterable(_ISSUED_EXECUTOR_GENERATIONS)
                )
            ):
                raise Experiment002TrainingBridgeError(
                    "accepted transition lost immutable generation authority"
                )
        else:
            _validate_executor_session_state(issuer.executor_authority, session)
        assert type(session) is _IssuedExecutorSessionState
        if (
            session.issuer.session_token is not issuer.executor_session_token
            or session.issuer.seed != snapshot.seed
        ):
            raise Experiment002TrainingBridgeError(
                "registered transition executor binding differs"
            )
    elif issuer.executor_authority is not None:
        raise Experiment002TrainingBridgeError(
            "synthetic transition contains registered executor authority"
        )
    if state.phase == "ISSUED":
        if (
            state.trace_authority is not None
            or state.trace_consumption is not None
            or guarded_consumption is not None
        ):
            raise Experiment002TrainingBridgeError(
                "unconsumed transition contains trace authority"
            )
    elif state.phase in {"TRACE_CONSUMED", "POPULATION_ACCEPTED"}:
        mark = state.trace_authority
        if (
            type(mark) is not _TraceMarkAuthority
            or type(mark.receipt_token) is not object
            or type(mark.trace_session_token) is not object
            or type(mark.trace_record_index) is not int
            or mark.trace_record_index != snapshot.zero_based_global_update
            or type(state.trace_consumption) is not TraceConsumedTransition
            or state.trace_consumption is not guarded_consumption
            or state.trace_consumption is not claim_guard
        ):
            raise Experiment002TrainingBridgeError(
                "consumed transition trace authority is invalid"
            )
    elif state.trace_authority is None:
        if (
            state.trace_consumption is not None
            or guarded_consumption is not None
            or claim_guard is not None
            or claimed
        ):
            raise Experiment002TrainingBridgeError(
                "failed transition consumption authority is invalid"
            )
    else:
        mark = state.trace_authority
        if (
            type(mark) is not _TraceMarkAuthority
            or type(mark.receipt_token) is not object
            or type(mark.trace_session_token) is not object
            or mark.trace_record_index != snapshot.zero_based_global_update
            or type(state.trace_consumption) is not TraceConsumedTransition
            or state.trace_consumption is not guarded_consumption
            or state.trace_consumption is not claim_guard
        ):
            raise Experiment002TrainingBridgeError(
                "failed transition trace authority is invalid"
            )
    if guarded_consumption is not None:
        consumption_state = _ISSUED_CONSUMPTIONS.get(guarded_consumption)
        _validate_consumption_state(guarded_consumption, consumption_state)
        assert consumption_state is not None
        mark = state.trace_authority
        consumption_snapshot = consumption_state.snapshot
        if (
            type(mark) is not _TraceMarkAuthority
            or consumption_snapshot.executor_session_token
            is not issuer.executor_session_token
            or consumption_snapshot.epoch_session_token
            is not issuer.epoch_session_token
            or consumption_snapshot.batch_token is not issuer.batch_token
            or consumption_snapshot.transition_token is not issuer.transition_token
            or consumption_snapshot.receipt_token is not mark.receipt_token
            or consumption_snapshot.trace_session_token is not mark.trace_session_token
            or consumption_snapshot.trace_record_index != mark.trace_record_index
            or (
                state.phase == "TRACE_CONSUMED"
                and (consumption_state.used or consumption_state.failed)
            )
            or (
                state.phase == "POPULATION_ACCEPTED"
                and (not consumption_state.used or consumption_state.failed)
            )
            or (state.phase == "FAILED" and not consumption_state.failed)
        ):
            raise Experiment002TrainingBridgeError(
                "transition consumption differs from independent authority"
            )


def _validate_consumption_state(
    consumption: TraceConsumedTransition,
    state: _IssuedConsumptionState | None,
) -> None:
    if type(state) is not _IssuedConsumptionState:
        raise Experiment002TrainingBridgeError(
            "trace consumption was not issued by this process"
        )
    snapshot = state.snapshot
    issuer = state.issuer
    guard = _CONSUMPTION_GUARDS.get(consumption)
    lifecycle = _CONSUMPTION_LIFECYCLES.get(consumption)
    if (
        type(guard) is not _ConsumptionIssuerAuthority
        or type(state.invalid) is not bool
        or state.invalid is not (consumption in _INVALID_CONSUMPTIONS)
        or state.invalid
        or not _consumption_issuer_matches_guard(issuer, guard)
        or type(lifecycle) is not tuple
        or len(lifecycle) != 2
        or state.used is not lifecycle[0]
        or state.failed is not lifecycle[1]
        or state.failed is not (consumption in _FAILED_CONSUMPTIONS)
        or state.used is not (consumption in _USED_CONSUMPTIONS)
        or type(snapshot) is not _TraceConsumptionSnapshot
        or type(issuer) is not _ConsumptionIssuerAuthority
        or type(issuer.transition) is not RegisteredOptimizerTransition
        or type(state.used) is not bool
        or type(state.failed) is not bool
        or type(snapshot.executor_session_token) is not object
        or snapshot.executor_session_token is not issuer.executor_session_token
        or type(snapshot.epoch_session_token) is not object
        or snapshot.epoch_session_token is not issuer.epoch_session_token
        or type(snapshot.batch_token) is not object
        or snapshot.batch_token is not issuer.batch_token
        or type(snapshot.transition_token) is not object
        or snapshot.transition_token is not issuer.transition_token
        or type(snapshot.receipt_token) is not object
        or snapshot.receipt_token is not issuer.receipt_token
        or type(snapshot.trace_session_token) is not object
        or snapshot.trace_session_token is not issuer.trace_session_token
        or type(snapshot.consumption_token) is not object
        or snapshot.consumption_token is not issuer.consumption_token
        or snapshot.zero_based_global_update != issuer.zero_based_global_update
        or snapshot.trace_record_index != issuer.trace_record_index
        or type(issuer.authority_payload) is not bytes
        or _frame_consumption_authority(snapshot) != issuer.authority_payload
    ):
        raise Experiment002TrainingBridgeError(
            "trace consumption differs from issuer authority"
        )
    _require_uint32(snapshot.zero_based_global_update, "global_update")
    _require_uint32(snapshot.trace_record_index, "trace_record_index")
    transition_state = _ISSUED_TRANSITIONS.get(issuer.transition)
    if (
        type(transition_state) is not _IssuedTransitionState
        or transition_state.invalid
        or issuer.transition in _INVALID_TRANSITIONS
        or transition_state.issuer.transition_token is not issuer.transition_token
        or transition_state.trace_consumption is not consumption
        or _TRANSITION_CONSUMPTIONS.get(issuer.transition) is not consumption
        or _TRANSITION_CLAIM_GUARDS.get(issuer.transition) is not consumption
        or (
            not state.used
            and not state.failed
            and (
                transition_state.phase != "TRACE_CONSUMED"
                or _TRANSITION_PHASES.get(issuer.transition) != "TRACE_CONSUMED"
                or issuer.transition not in _CLAIMED_TRANSITIONS
                or issuer.transition in _ACCEPTED_TRANSITIONS
                or issuer.transition in _FAILED_TRANSITIONS
            )
        )
        or (
            state.used
            and (
                transition_state.phase != "POPULATION_ACCEPTED"
                or _TRANSITION_PHASES.get(issuer.transition) != "POPULATION_ACCEPTED"
                or issuer.transition not in _CLAIMED_TRANSITIONS
                or issuer.transition not in _ACCEPTED_TRANSITIONS
                or issuer.transition in _FAILED_TRANSITIONS
            )
        )
        or (
            state.failed
            and (
                transition_state.phase != "FAILED"
                or _TRANSITION_PHASES.get(issuer.transition) != "FAILED"
                or issuer.transition not in _FAILED_TRANSITIONS
                or issuer.transition in _ACCEPTED_TRANSITIONS
            )
        )
    ):
        raise Experiment002TrainingBridgeError(
            "trace consumption transition binding differs"
        )


def _validate_transition_snapshot(
    snapshot: _OptimizerTransitionSnapshot,
    *,
    route_marker: object | None,
) -> None:
    if (
        type(snapshot) is not _OptimizerTransitionSnapshot
        or type(snapshot.executor_session_token) is not object
        or type(snapshot.epoch_session_token) is not object
        or type(snapshot.batch_token) is not object
        or type(snapshot.transition_token) is not object
        or type(snapshot.seed) is not int
        or type(snapshot.zero_based_epoch) is not int
        or type(snapshot.zero_based_global_update) is not int
        or type(snapshot.batch_size) is not int
        or snapshot.batch_size < 1
        or type(snapshot.learning_rate) is not float
        or not math.isfinite(snapshot.learning_rate)
        or snapshot.learning_rate <= 0.0
        or type(snapshot.learning_rate_bytes) is not bytes
        or snapshot.learning_rate_bytes != struct.pack("<d", snapshot.learning_rate)
        or type(snapshot.batch_mean_training_loss) is not np.float32
        or not math.isfinite(float(snapshot.batch_mean_training_loss))
        or float(snapshot.batch_mean_training_loss) < 0.0
        or snapshot.batch_mean_training_loss_bytes
        != struct.pack("<f", snapshot.batch_mean_training_loss)
        or type(snapshot.returned_preclip_l2_norm) is not np.float32
        or not math.isfinite(float(snapshot.returned_preclip_l2_norm))
        or float(snapshot.returned_preclip_l2_norm) < 0.0
        or snapshot.returned_preclip_l2_norm_bytes
        != struct.pack("<f", snapshot.returned_preclip_l2_norm)
        or type(snapshot.optimizer_generation_before) is not int
        or type(snapshot.optimizer_generation_after) is not int
        or snapshot.optimizer_generation_after
        != snapshot.optimizer_generation_before + 1
        or snapshot.optimizer_generation_before != snapshot.zero_based_global_update
    ):
        raise Experiment002TrainingBridgeError(
            "optimizer transition snapshot is invalid"
        )
    for name, value in (
        ("seed", snapshot.seed),
        ("zero_based_epoch", snapshot.zero_based_epoch),
        ("zero_based_global_update", snapshot.zero_based_global_update),
        ("batch_size", snapshot.batch_size),
        ("optimizer_generation_before", snapshot.optimizer_generation_before),
        ("optimizer_generation_after", snapshot.optimizer_generation_after),
    ):
        _require_uint32(value, name)
    if route_marker is _REGISTERED_ROUTE:
        _require_registered_seed(snapshot.seed)
        if not 0 <= snapshot.zero_based_epoch < _REGISTERED_EPOCHS:
            raise Experiment002TrainingBridgeError(
                "registered transition epoch is outside registered range"
            )
        if snapshot.zero_based_global_update // _UPDATES_PER_EPOCH != (
            snapshot.zero_based_epoch
        ):
            raise Experiment002TrainingBridgeError(
                "registered transition update belongs to another epoch"
            )
        update_in_epoch = snapshot.zero_based_global_update % _UPDATES_PER_EPOCH
        expected_batch_size = (
            _REGISTERED_LAST_BATCH_SIZE
            if update_in_epoch == _UPDATES_PER_EPOCH - 1
            else _REGISTERED_BATCH_SIZE
        )
        if snapshot.batch_size != expected_batch_size:
            raise Experiment002TrainingBridgeError(
                "registered transition batch size differs"
            )
        expected_lr = _registered_learning_rate(snapshot.zero_based_global_update)
        if snapshot.learning_rate_bytes != struct.pack("<d", expected_lr):
            raise Experiment002TrainingBridgeError(
                "registered transition learning rate differs"
            )
    elif route_marker is not None:
        raise Experiment002TrainingBridgeError(
            "optimizer transition route marker is invalid"
        )
    for name, digest in (
        ("model_sha256_before", snapshot.model_sha256_before),
        ("model_sha256_after", snapshot.model_sha256_after),
        ("optimizer_sha256_before", snapshot.optimizer_sha256_before),
        ("optimizer_sha256_after", snapshot.optimizer_sha256_after),
        ("rng_sha256_before", snapshot.rng_sha256_before),
        ("rng_sha256_after", snapshot.rng_sha256_after),
    ):
        if not _is_sha256(digest):
            raise Experiment002TrainingBridgeError(f"{name} is not canonical SHA-256")


def _frame_transition_authority(
    snapshot: _OptimizerTransitionSnapshot,
    *,
    route_marker: object | None,
) -> bytes:
    _validate_transition_snapshot(snapshot, route_marker=route_marker)
    framed = bytearray(
        struct.pack(
            "<IIIIIdffII",
            snapshot.seed,
            snapshot.zero_based_epoch,
            snapshot.zero_based_global_update,
            snapshot.batch_size,
            snapshot.optimizer_generation_before,
            snapshot.learning_rate,
            snapshot.batch_mean_training_loss,
            snapshot.returned_preclip_l2_norm,
            snapshot.optimizer_generation_after,
            6,
        )
    )
    for digest in (
        snapshot.model_sha256_before,
        snapshot.model_sha256_after,
        snapshot.optimizer_sha256_before,
        snapshot.optimizer_sha256_after,
        snapshot.rng_sha256_before,
        snapshot.rng_sha256_after,
    ):
        framed.extend(bytes.fromhex(digest))
    return bytes(framed)


def _frame_consumption_authority(snapshot: _TraceConsumptionSnapshot) -> bytes:
    return struct.pack(
        "<II", snapshot.zero_based_global_update, snapshot.trace_record_index
    )


def _registered_learning_rate(global_update: int) -> float:
    _require_uint32(global_update, "global_update")
    if global_update >= _TOTAL_UPDATES:
        raise Experiment002TrainingBridgeError(
            "global_update is outside registered schedule"
        )
    if global_update < _UPDATES_PER_EPOCH:
        return 0.0003 + (0.003 - 0.0003) * global_update / (_UPDATES_PER_EPOCH - 1)
    return 0.00003 + 0.5 * (0.003 - 0.00003) * (
        1.0
        + math.cos(
            math.pi
            * (global_update - _UPDATES_PER_EPOCH)
            / (_TOTAL_UPDATES - 1 - _UPDATES_PER_EPOCH)
        )
    )


def _require_registered_seed(seed: int) -> None:
    _require_uint32(seed, "seed")
    if seed not in _REGISTERED_SEEDS:
        raise Experiment002TrainingBridgeError("seed is not registered")


def _require_uint32(value: object, name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if not 0 <= value <= 0xFFFF_FFFF:
        raise Experiment002TrainingBridgeError(f"{name} must fit UINT32LE")


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
