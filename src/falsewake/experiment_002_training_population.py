"""One-pass, provenance-bound training populations for Experiment 002.

The registered path deliberately materializes only the current canonical
batch.  Issuer-only capabilities and byte snapshots make the hand-off to the
trainer explicit: a batch is checked immediately before and after its update,
then an exact scalar receipt is accepted once and the batch storage is
released.

This module does not execute or observe an optimizer step.  The trusted trainer
must place the two batch checks on opposite sides of that step and bind the
resulting receipt to the separately verified optimizer/update trace; a receipt
alone is byte-exact input/scalar evidence, not proof that an update occurred.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import struct
import threading
import weakref
from contextlib import suppress
from dataclasses import dataclass, field
from dataclasses import replace as dataclass_replace
from typing import Final, Literal, NoReturn, Protocol

import numpy as np

from falsewake.experiment_002_data import (
    CLASS_ORDER,
    TRAIN_EPOCH_COUNT,
    TRAIN_EXAMPLE_COUNT,
    TRAINING_SEEDS,
    BackgroundSource,
    CommandExample,
    CommandSource,
    Experiment002Corpus,
    WindowExample,
    _require_registered_training_corpus,
    build_training_epoch,
)
from falsewake.experiment_002_normalization_artifact import (
    NORMALIZATION_ARTIFACT_BYTES,
    NormalizationArtifactIdentity,
    VerifiedNormalization,
    verify_registered_normalization,
)
from falsewake.experiment_002_pcm_cache import (
    REGISTERED_FILE_BYTES,
    Experiment002PCMCache,
    PCMCacheIdentity,
    PCMCacheSplitView,
    _IndexEntry,
    _ParsedIndex,
)
from falsewake.experiment_002_preprocessing import (
    MEL_BINS,
    TIME_FRAMES,
    Experiment002TrainingPreprocessor,
)
from falsewake.experiment_002_training_bridge import (
    RegisteredExecutorSessionAuthority,
    RegisteredOptimizerTransition,
    TraceConsumedTransition,
    _accept_trace_consumed_transition,
    _executor_session_snapshot,
    _fail_optimizer_transition,
    _fail_registered_executor_session,
    _OptimizerTransitionSnapshot,
    _require_registered_executor_generation,
    _trace_consumption_snapshot,
    _transition_snapshot,
)
from falsewake.features import FloatArray

TRAINING_BATCH_SIZE: Final = 128
REGISTERED_BATCH_COUNT: Final = 313
REGISTERED_LAST_BATCH_SIZE: Final = 91
TRAINING_POPULATION_DOMAIN: Final = b"falsewake-exp002-training-population-v1\0"
REGISTERED_PCM_CACHE_SHA256: Final = (
    "b56270a4b99d235c62c162545b139a79b7cf3dc157259a3bbf5678d8a451f653"
)
REGISTERED_NORMALIZATION_SHA256: Final = (
    "891900d4c36fa8a71ba429384f3f4ff594de7c81347ed846c0f790dc4579268e"
)

type TrainingState = Literal["OPEN", "BATCH_PENDING", "READY", "COMPLETE", "FAILED"]
type _Example = CommandExample | WindowExample

_REGISTERED_SOURCE_MARKER: Final = object()
_REGISTERED_EPOCH_MARKER: Final = object()
_UINT32_MAX: Final = (1 << 32) - 1
_UINT64_MAX: Final = (1 << 64) - 1
_RLOCK_TYPE: Final = type(threading.RLock())


class Experiment002TrainingPopulationError(ValueError):
    """Training input provenance or one-pass state violated the contract."""


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
class _RegisteredEpochConstructionAuthority:
    """One-use authority joining the checked registered begin path."""

    def __init__(self) -> None:
        raise TypeError("registered epoch construction authorities are issuer-only")


@dataclass(frozen=True, slots=True)
class _RegisteredEpochConstructionGuard:
    source: RegisteredTrainingInputSource
    executor_authority: RegisteredExecutorSessionAuthority
    executor_session_token: object
    seed: int
    zero_based_epoch: int


_REGISTERED_EPOCH_CONSTRUCTION_GUARDS: weakref.WeakKeyDictionary[
    _RegisteredEpochConstructionAuthority, _RegisteredEpochConstructionGuard
] = weakref.WeakKeyDictionary()
_CONSUMED_EPOCH_CONSTRUCTION_AUTHORITIES: weakref.WeakSet[
    _RegisteredEpochConstructionAuthority
] = weakref.WeakSet()
_REGISTERED_EPOCH_CONSTRUCTION_LOCK = threading.Lock()


def _require_positive_integer(value: object, name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise Experiment002TrainingPopulationError(f"{name} must be positive")


class _TrainingPreprocessor(Protocol):
    def command_model_input(
        self, example: CommandExample, *, seed: int, epoch: int
    ) -> FloatArray: ...

    def silence_model_input(
        self, example: WindowExample, *, seed: int, epoch: int
    ) -> FloatArray: ...


class _IncrementalDigest(Protocol):
    def update(self, payload: bytes | bytearray) -> None: ...

    def hexdigest(self) -> str: ...


@dataclass(frozen=True, slots=True)
class _TrainingLayout:
    example_count: int
    batch_size: int
    mel_bins: int = MEL_BINS
    time_frames: int = TIME_FRAMES

    def __post_init__(self) -> None:
        _require_positive_integer(self.example_count, "example_count")
        _require_positive_integer(self.batch_size, "batch_size")
        _require_positive_integer(self.mel_bins, "mel_bins")
        _require_positive_integer(self.time_frames, "time_frames")
        if self.example_count > _UINT32_MAX:
            raise Experiment002TrainingPopulationError("example_count exceeds UINT32LE")

    @property
    def batch_count(self) -> int:
        return (self.example_count + self.batch_size - 1) // self.batch_size

    @property
    def last_batch_size(self) -> int:
        remainder = self.example_count % self.batch_size
        return self.batch_size if remainder == 0 else remainder

    @property
    def feature_payload_bytes(self) -> int:
        return self.mel_bins * self.time_frames * np.dtype("<f4").itemsize


_REGISTERED_LAYOUT: Final = _TrainingLayout(
    example_count=TRAIN_EXAMPLE_COUNT,
    batch_size=TRAINING_BATCH_SIZE,
)


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
class RegisteredTrainingInputSource:
    """Issuer-only binding of the exact corpus, cache, and normalization."""

    def __init__(self) -> None:
        raise TypeError("registered training input sources are issued by this module")

    def begin_epoch(
        self, *, seed: int, zero_based_epoch: int
    ) -> RegisteredTrainingEpoch:
        """Reject direct registered use; the exact executor owns epoch plans."""

        del seed, zero_based_epoch
        raise Experiment002TrainingPopulationError(
            "registered training epochs are owned by the exact executor"
        )

    def __copy__(self) -> NoReturn:
        raise TypeError("registered training input sources cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        del memo
        raise TypeError("registered training input sources cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("registered training input sources cannot be serialized")


@dataclass(slots=True)
class _IssuedSourceState:
    corpus: Experiment002Corpus
    cache: PCMCacheSplitView
    normalization: VerifiedNormalization
    preprocessor: Experiment002TrainingPreprocessor
    corpus_snapshot: tuple[object, ...]
    cache_snapshot: tuple[object, ...]
    normalization_contents: bytes
    marker: object
    begun_contexts: set[tuple[int, int]] = field(default_factory=set)
    lock: threading.RLock = field(default_factory=threading.RLock)


@dataclass(frozen=True, slots=True)
class _SourceIssuanceGuard:
    corpus: Experiment002Corpus
    cache: PCMCacheSplitView
    normalization: VerifiedNormalization
    preprocessor: Experiment002TrainingPreprocessor
    corpus_snapshot: tuple[object, ...]
    cache_snapshot: tuple[object, ...]
    normalization_contents: bytes
    marker: object


_ISSUED_SOURCES: weakref.WeakKeyDictionary[
    RegisteredTrainingInputSource, _IssuedSourceState
] = weakref.WeakKeyDictionary()
_SOURCE_GUARDS: weakref.WeakKeyDictionary[
    RegisteredTrainingInputSource, _SourceIssuanceGuard
] = weakref.WeakKeyDictionary()
_SOURCE_CONTEXT_GUARDS: weakref.WeakKeyDictionary[
    RegisteredTrainingInputSource, frozenset[tuple[int, int]]
] = weakref.WeakKeyDictionary()
_ISSUED_SOURCES_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
class MaterializedTrainingBatch:
    """One issuer-bound owned batch, invalidated after receipt acceptance."""

    def __init__(self) -> None:
        raise TypeError("training batches are issued by a training epoch")

    @property
    def model_inputs(self) -> FloatArray:
        """Return the read-only owned float32 ``[B, 40, 98]`` array."""

        return _issued_batch_state(self).model_inputs

    @property
    def label_indices(self) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
        """Return the read-only owned int64 ``[B]`` label array."""

        return _issued_batch_state(self).label_indices

    @property
    def zero_based_global_update(self) -> int:
        return _issued_batch_state(self).global_update

    @property
    def batch_size(self) -> int:
        return _issued_batch_state(self).label_indices.shape[0]

    def __copy__(self) -> NoReturn:
        raise TypeError("training batches cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        del memo
        raise TypeError("training batches cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("training batches cannot be serialized")


@dataclass(frozen=True, slots=True)
class _IssuedBatchState:
    session_token: object
    batch_token: object
    batch_index: int
    global_update: int
    first_example: int
    examples: tuple[_Example, ...]
    example_snapshots: tuple[tuple[object, ...], ...]
    model_inputs: FloatArray
    label_indices: np.ndarray[tuple[int], np.dtype[np.int64]]
    model_inputs_sha256: str
    label_indices_sha256: str


@dataclass(frozen=True, slots=True)
class _BatchIssuanceGuard:
    session_token: object
    batch_token: object
    batch_index: int
    global_update: int
    first_example: int
    examples: tuple[_Example, ...]
    example_snapshots: tuple[tuple[object, ...], ...]
    model_inputs: FloatArray
    label_indices: np.ndarray[tuple[int], np.dtype[np.int64]]
    model_inputs_sha256: str
    label_indices_sha256: str


_ISSUED_BATCHES: weakref.WeakKeyDictionary[
    MaterializedTrainingBatch, _IssuedBatchState
] = weakref.WeakKeyDictionary()
_BATCH_GUARDS: weakref.WeakKeyDictionary[
    MaterializedTrainingBatch, _BatchIssuanceGuard
] = weakref.WeakKeyDictionary()
_ISSUED_BATCHES_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
class CompletedTrainingUpdate:
    """Issuer-only exact scalar receipt for one completed update."""

    def __init__(self) -> None:
        raise TypeError("completed training updates are issued by a training epoch")

    @property
    def zero_based_global_update(self) -> int:
        return _issued_receipt_state(self).snapshot.zero_based_global_update

    @property
    def batch_size(self) -> int:
        return _issued_receipt_state(self).snapshot.batch_size

    @property
    def learning_rate(self) -> float:
        return _issued_receipt_state(self).snapshot.learning_rate

    @property
    def batch_mean_training_loss(self) -> np.float32:
        return _issued_receipt_state(self).snapshot.batch_mean_training_loss

    @property
    def returned_preclip_l2_norm(self) -> np.float32:
        return _issued_receipt_state(self).snapshot.returned_preclip_l2_norm

    @property
    def accepted(self) -> bool:
        return _issued_receipt_state(self).accepted

    def __copy__(self) -> NoReturn:
        raise TypeError("completed training updates cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        del memo
        raise TypeError("completed training updates cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("completed training updates cannot be serialized")


@dataclass(frozen=True, slots=True)
class _CompletedTrainingUpdateSnapshot:
    session_token: object
    batch_token: object
    receipt_token: object
    optimizer_transition_token: object | None
    seed: int
    zero_based_epoch: int
    batch_index: int
    zero_based_global_update: int
    batch_size: int
    learning_rate: float
    learning_rate_bytes: bytes
    batch_mean_training_loss: np.float32
    batch_mean_training_loss_bytes: bytes
    returned_preclip_l2_norm: np.float32
    returned_preclip_l2_norm_bytes: bytes


@dataclass(slots=True)
class _IssuedReceiptState:
    snapshot: _CompletedTrainingUpdateSnapshot
    authority_payload: bytes
    session_token: object
    batch_token: object
    receipt_token: object
    optimizer_transition_token: object | None
    optimizer_transition: RegisteredOptimizerTransition | None
    accepted: bool = False


@dataclass(frozen=True, slots=True)
class _ReceiptIssuanceGuard:
    authority_payload: bytes
    session_token: object
    batch_token: object
    receipt_token: object
    optimizer_transition_token: object | None
    optimizer_transition: RegisteredOptimizerTransition | None


_ISSUED_RECEIPTS: weakref.WeakKeyDictionary[
    CompletedTrainingUpdate, _IssuedReceiptState
] = weakref.WeakKeyDictionary()
_RECEIPT_GUARDS: weakref.WeakKeyDictionary[
    CompletedTrainingUpdate, _ReceiptIssuanceGuard
] = weakref.WeakKeyDictionary()
_ACCEPTED_RECEIPTS: weakref.WeakSet[CompletedTrainingUpdate] = weakref.WeakSet()
_ISSUED_RECEIPTS_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class _AcceptedUpdateAuthority:
    payload: bytes
    session_token: object
    batch_token: object
    receipt_token: object
    optimizer_transition_token: object | None
    optimizer_transition: RegisteredOptimizerTransition | None
    trace_consumption: TraceConsumedTransition | None
    trace_consumption_token: object | None


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
class CompletedTrainingPopulation:
    """Issuer-only digest proving one epoch was consumed exactly once."""

    def __init__(self) -> None:
        raise TypeError("completed training populations are issued by an epoch")

    @property
    def seed(self) -> int:
        return _issued_population_state(self).snapshot.seed

    @property
    def zero_based_epoch(self) -> int:
        return _issued_population_state(self).snapshot.zero_based_epoch

    @property
    def sha256(self) -> str:
        return _issued_population_state(self).snapshot.sha256

    @property
    def training_population_digest(self) -> str:
        """Return the canonical training-population SHA-256 digest."""

        return _issued_population_state(self).snapshot.sha256

    @property
    def example_count(self) -> int:
        return _issued_population_state(self).snapshot.example_count

    @property
    def batch_count(self) -> int:
        return _issued_population_state(self).snapshot.batch_count

    def __copy__(self) -> NoReturn:
        raise TypeError("completed training populations cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        del memo
        raise TypeError("completed training populations cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("completed training populations cannot be serialized")


@dataclass(frozen=True, slots=True)
class _CompletedTrainingPopulationSnapshot:
    session_token: object
    seed: int
    zero_based_epoch: int
    sha256: str
    example_count: int
    batch_count: int
    updates: tuple[_CompletedTrainingUpdateSnapshot, ...]
    ordered_batch_tokens: tuple[object, ...]
    ordered_receipt_tokens: tuple[object, ...]
    ordered_optimizer_transition_tokens: tuple[object, ...]
    ordered_optimizer_transitions: tuple[RegisteredOptimizerTransition, ...]
    ordered_trace_consumptions: tuple[TraceConsumedTransition, ...]
    route_marker: object | None


@dataclass(frozen=True, slots=True)
class _IssuedPopulationState:
    snapshot: _CompletedTrainingPopulationSnapshot
    sha256: str
    seed: int
    zero_based_epoch: int
    example_count: int
    batch_count: int
    session_token: object
    update_authority_payloads: tuple[bytes, ...]
    ordered_batch_tokens: tuple[object, ...]
    ordered_receipt_tokens: tuple[object, ...]
    ordered_optimizer_transition_tokens: tuple[object, ...]
    ordered_optimizer_transitions: tuple[RegisteredOptimizerTransition, ...]
    ordered_trace_consumptions: tuple[TraceConsumedTransition, ...]
    route_marker: object | None


@dataclass(frozen=True, slots=True)
class _PopulationIssuanceGuard:
    sha256: str
    seed: int
    zero_based_epoch: int
    example_count: int
    batch_count: int
    session_token: object
    update_authority_payloads: tuple[bytes, ...]
    ordered_batch_tokens: tuple[object, ...]
    ordered_receipt_tokens: tuple[object, ...]
    ordered_optimizer_transition_tokens: tuple[object, ...]
    ordered_optimizer_transitions: tuple[RegisteredOptimizerTransition, ...]
    ordered_trace_consumptions: tuple[TraceConsumedTransition, ...]
    route_marker: object | None


_ISSUED_POPULATIONS: weakref.WeakKeyDictionary[
    CompletedTrainingPopulation, _IssuedPopulationState
] = weakref.WeakKeyDictionary()
_POPULATION_GUARDS: weakref.WeakKeyDictionary[
    CompletedTrainingPopulation, _PopulationIssuanceGuard
] = weakref.WeakKeyDictionary()
_ISSUED_POPULATIONS_LOCK = threading.Lock()


@dataclass(slots=True)
class _EpochState:
    seed: int
    zero_based_epoch: int
    layout: _TrainingLayout
    examples: tuple[_Example, ...]
    example_snapshots: tuple[tuple[object, ...], ...]
    preprocessor: _TrainingPreprocessor
    source: RegisteredTrainingInputSource | None
    route_marker: object | None
    session_token: object
    executor_authority: RegisteredExecutorSessionAuthority | None
    executor_session_token: object | None
    hasher: _IncrementalDigest
    phase: TrainingState = "OPEN"
    cursor: int = 0
    batch_index: int = 0
    current_batch: MaterializedTrainingBatch | None = None
    pending_verifications: int = 0
    ready_receipt: CompletedTrainingUpdate | None = None
    accepted_updates: list[_CompletedTrainingUpdateSnapshot] = field(
        default_factory=list
    )
    accepted_update_authorities: list[_AcceptedUpdateAuthority] = field(
        default_factory=list
    )
    active_executor_authority: RegisteredExecutorSessionAuthority | None = None
    active_transition: RegisteredOptimizerTransition | None = None
    active_consumption: TraceConsumedTransition | None = None
    active_verification_phase: str | None = None
    finished: bool = False
    lock: threading.RLock = field(default_factory=threading.RLock)


@dataclass(frozen=True, slots=True)
class _EpochIssuanceGuard:
    seed: int
    zero_based_epoch: int
    layout: _TrainingLayout
    examples: tuple[_Example, ...]
    example_snapshots: tuple[tuple[object, ...], ...]
    preprocessor: _TrainingPreprocessor
    source: RegisteredTrainingInputSource | None
    route_marker: object | None
    session_token: object
    executor_authority: RegisteredExecutorSessionAuthority | None
    executor_session_token: object | None
    hasher: _IncrementalDigest


@dataclass(frozen=True, slots=True)
class _EpochLifecycleGuard:
    phase: TrainingState
    cursor: int
    batch_index: int
    current_batch: MaterializedTrainingBatch | None
    pending_verifications: int
    ready_receipt: CompletedTrainingUpdate | None
    finished: bool


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
class RegisteredTrainingEpoch:
    """One canonical epoch with a strict one-pass batch/update state machine."""

    def __init__(self) -> None:
        raise TypeError("training epochs are issued by a training input source")

    @property
    def state(self) -> TrainingState:
        return _issued_epoch_state(self).phase

    @property
    def seed(self) -> int:
        return _issued_epoch_state(self).seed

    @property
    def zero_based_epoch(self) -> int:
        return _issued_epoch_state(self).zero_based_epoch

    def next_batch(self, global_update: int) -> MaterializedTrainingBatch:
        """Materialize the next owned canonical batch and no later examples."""

        state = _issued_epoch_state(self)
        with state.lock:
            try:
                _require_epoch_call_authority(state)
                _require_phase(state, "OPEN")
                _require_expected_global_update(state, global_update)
                _verify_epoch_backing(state, full=False)
                batch = _materialize_next_batch(self, state, global_update)
                state.current_batch = batch
                state.pending_verifications = 0
                state.phase = "BATCH_PENDING"
                _sync_epoch_lifecycle(self, state)
                return batch
            except BaseException:
                _terminal_fail_epoch_state(self, state)
                raise

    def verify_pending_batch(
        self,
        batch: MaterializedTrainingBatch,
        *,
        global_update: int,
    ) -> None:
        """Verify the same immutable batch once before and once after its step."""

        state = _issued_epoch_state(self)
        with state.lock:
            try:
                _require_epoch_call_authority(state)
                _require_phase(state, "BATCH_PENDING")
                _require_expected_global_update(state, global_update)
                if state.route_marker is _REGISTERED_EPOCH_MARKER:
                    expected_phase = (
                        "PRE" if state.pending_verifications == 0 else "POST"
                    )
                    if state.active_verification_phase != expected_phase:
                        raise Experiment002TrainingPopulationError(
                            "registered batch verification phase differs"
                        )
                if state.pending_verifications >= 2:
                    raise Experiment002TrainingPopulationError(
                        "pending batch was already verified before and after its step"
                    )
                _verify_pending_batch(state, batch)
                state.pending_verifications += 1
                _sync_epoch_lifecycle(self, state)
            except BaseException:
                _terminal_fail_epoch_state(self, state)
                raise

    def complete_update(
        self,
        batch: MaterializedTrainingBatch,
        *,
        learning_rate: float,
        batch_mean_training_loss: np.float32,
        returned_preclip_l2_norm: np.float32,
    ) -> CompletedTrainingUpdate:
        """Issue one byte-exact scalar receipt after both batch checks."""

        state = _issued_epoch_state(self)
        with state.lock:
            try:
                _require_epoch_call_authority(state)
                _require_phase(state, "BATCH_PENDING")
                if state.pending_verifications != 2:
                    raise Experiment002TrainingPopulationError(
                        "pending batch requires exactly two verifications"
                    )
                batch_state = _verify_pending_batch(state, batch)
                transition_token: object | None = None
                if state.route_marker is _REGISTERED_EPOCH_MARKER:
                    transition = state.active_transition
                    if type(transition) is not RegisteredOptimizerTransition:
                        raise Experiment002TrainingPopulationError(
                            "registered update requires an optimizer transition"
                        )
                    transition_snapshot = _transition_snapshot(
                        transition, required_phase="ISSUED"
                    )
                    _require_transition_matches_batch(
                        state,
                        batch_state,
                        transition_snapshot,
                        learning_rate=learning_rate,
                        batch_mean_training_loss=batch_mean_training_loss,
                        returned_preclip_l2_norm=returned_preclip_l2_norm,
                    )
                    transition_token = transition_snapshot.transition_token
                receipt = _issue_completed_update(
                    state,
                    batch_state,
                    learning_rate=learning_rate,
                    batch_mean_training_loss=batch_mean_training_loss,
                    returned_preclip_l2_norm=returned_preclip_l2_norm,
                    optimizer_transition_token=transition_token,
                    optimizer_transition=(
                        transition
                        if state.route_marker is _REGISTERED_EPOCH_MARKER
                        else None
                    ),
                )
                state.ready_receipt = receipt
                state.phase = "READY"
                _sync_epoch_lifecycle(self, state)
                return receipt
            except BaseException:
                _terminal_fail_epoch_state(self, state)
                raise

    def accept_completed_update(self, receipt: CompletedTrainingUpdate) -> None:
        """Accept the bound receipt once, invalidate and release its batch."""

        state = _issued_epoch_state(self)
        with state.lock:
            try:
                _require_epoch_call_authority(state)
                _require_phase(state, "READY")
                if type(receipt) is not CompletedTrainingUpdate:
                    raise TypeError("receipt must be a CompletedTrainingUpdate")
                if receipt is not state.ready_receipt:
                    raise Experiment002TrainingPopulationError(
                        "receipt does not belong to the pending batch"
                    )
                receipt_state = _issued_receipt_state(receipt)
                if receipt_state.accepted:
                    raise Experiment002TrainingPopulationError(
                        "completed update receipt was already accepted"
                    )
                batch = state.current_batch
                if batch is None:
                    raise Experiment002TrainingPopulationError(
                        "ready update has no retained batch"
                    )
                batch_state = _verify_pending_batch(state, batch)
                _verify_receipt_matches_batch(state, receipt_state, batch_state)
                with _ISSUED_EPOCHS_LOCK:
                    accepted_guard = _EPOCH_ACCEPTED_GUARDS.get(self)
                if (
                    type(accepted_guard) is not tuple
                    or tuple(state.accepted_update_authorities) != accepted_guard
                ):
                    raise Experiment002TrainingPopulationError(
                        "accepted update authority differs from issuance guard"
                    )
                if state.route_marker is _REGISTERED_EPOCH_MARKER:
                    transition = state.active_transition
                    consumption = state.active_consumption
                    if type(transition) is not RegisteredOptimizerTransition:
                        raise Experiment002TrainingPopulationError(
                            "registered acceptance lost its optimizer transition"
                        )
                    if type(consumption) is not TraceConsumedTransition:
                        raise Experiment002TrainingPopulationError(
                            "registered acceptance requires trace consumption"
                        )
                    _require_consumption_matches_receipt(
                        state,
                        receipt_state,
                        transition,
                        consumption,
                    )
                    consumption_snapshot = _trace_consumption_snapshot(
                        consumption, required_used=False
                    )
                    accepted_snapshot = _copy_update_snapshot(receipt_state.snapshot)
                    accepted_authority = _AcceptedUpdateAuthority(
                        payload=receipt_state.authority_payload,
                        session_token=receipt_state.session_token,
                        batch_token=receipt_state.batch_token,
                        receipt_token=receipt_state.receipt_token,
                        optimizer_transition_token=(
                            receipt_state.optimizer_transition_token
                        ),
                        optimizer_transition=transition,
                        trace_consumption=consumption,
                        trace_consumption_token=(
                            consumption_snapshot.consumption_token
                        ),
                    )
                    executor_authority = state.executor_authority
                    if (
                        type(executor_authority)
                        is not RegisteredExecutorSessionAuthority
                    ):
                        raise Experiment002TrainingPopulationError(
                            "registered epoch lost its executor authority"
                        )
                    next_accepted_guard = accepted_guard + (
                        dataclass_replace(accepted_authority),
                    )
                    _accept_trace_consumed_transition(
                        transition,
                        consumption,
                        receipt_token=receipt_state.receipt_token,
                        executor_authority=executor_authority,
                        epoch_session_token=state.session_token,
                        batch_token=batch_state.batch_token,
                    )
                else:
                    accepted_snapshot = _copy_update_snapshot(receipt_state.snapshot)
                    accepted_authority = _AcceptedUpdateAuthority(
                        payload=receipt_state.authority_payload,
                        session_token=receipt_state.session_token,
                        batch_token=receipt_state.batch_token,
                        receipt_token=receipt_state.receipt_token,
                        optimizer_transition_token=None,
                        optimizer_transition=None,
                        trace_consumption=None,
                        trace_consumption_token=None,
                    )
                    next_accepted_guard = accepted_guard + (
                        dataclass_replace(accepted_authority),
                    )
                with _ISSUED_RECEIPTS_LOCK:
                    receipt_state.accepted = True
                    _ACCEPTED_RECEIPTS.add(receipt)
                state.accepted_updates.append(accepted_snapshot)
                state.accepted_update_authorities.append(accepted_authority)
                with _ISSUED_EPOCHS_LOCK:
                    _EPOCH_ACCEPTED_GUARDS[self] = next_accepted_guard
                state.cursor += batch_state.label_indices.shape[0]
                state.batch_index += 1
                with _ISSUED_BATCHES_LOCK:
                    removed = _ISSUED_BATCHES.pop(batch, None)
                    removed_guard = _BATCH_GUARDS.pop(batch, None)
                if (
                    removed is not batch_state
                    or type(removed_guard) is not _BatchIssuanceGuard
                ):
                    raise Experiment002TrainingPopulationError(
                        "batch issuance changed while it was accepted"
                    )
                state.current_batch = None
                state.ready_receipt = None
                state.pending_verifications = 0
                state.phase = (
                    "COMPLETE" if state.cursor == state.layout.example_count else "OPEN"
                )
                _sync_epoch_lifecycle(self, state)
            except BaseException:
                _terminal_fail_epoch_state(self, state)
                raise

    def finish(self) -> CompletedTrainingPopulation:
        """Finalize the incremental population digest exactly once."""

        state = _issued_epoch_state(self)
        with state.lock:
            try:
                _require_epoch_call_authority(state)
                _require_phase(state, "COMPLETE")
                if state.finished:
                    raise Experiment002TrainingPopulationError(
                        "training population was already finalized"
                    )
                _verify_epoch_backing(state, full=True)
                if (
                    state.cursor != state.layout.example_count
                    or len(state.accepted_updates) != state.layout.batch_count
                ):
                    raise Experiment002TrainingPopulationError(
                        "completed population counts differ from its layout"
                    )
                if len(
                    state.accepted_update_authorities
                ) != state.layout.batch_count or any(
                    not _update_matches_accepted_authority(update, authority)
                    for update, authority in zip(
                        state.accepted_updates,
                        state.accepted_update_authorities,
                        strict=True,
                    )
                ):
                    raise Experiment002TrainingPopulationError(
                        "accepted update snapshots differ from issuer authority"
                    )
                _verify_epoch_hashers(self, state)
                result = _issue_completed_population(state)
                state.finished = True
                _sync_epoch_lifecycle(self, state)
                return result
            except BaseException:
                _terminal_fail_epoch_state(self, state)
                raise

    def __copy__(self) -> NoReturn:
        raise TypeError("training epochs cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        del memo
        raise TypeError("training epochs cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("training epochs cannot be serialized")


_ISSUED_EPOCHS: weakref.WeakKeyDictionary[RegisteredTrainingEpoch, _EpochState] = (
    weakref.WeakKeyDictionary()
)
_EPOCH_GUARDS: weakref.WeakKeyDictionary[
    RegisteredTrainingEpoch, _EpochIssuanceGuard
] = weakref.WeakKeyDictionary()
_EPOCH_ACCEPTED_GUARDS: weakref.WeakKeyDictionary[
    RegisteredTrainingEpoch, tuple[_AcceptedUpdateAuthority, ...]
] = weakref.WeakKeyDictionary()
_EPOCH_HASH_MIRRORS: weakref.WeakKeyDictionary[
    RegisteredTrainingEpoch, _IncrementalDigest
] = weakref.WeakKeyDictionary()
_EPOCH_LIFECYCLES: weakref.WeakKeyDictionary[
    RegisteredTrainingEpoch, _EpochLifecycleGuard
] = weakref.WeakKeyDictionary()
_ISSUED_EPOCHS_LOCK = threading.Lock()


def bind_registered_training_inputs(
    corpus: Experiment002Corpus,
    cache: PCMCacheSplitView,
    normalization: VerifiedNormalization,
) -> RegisteredTrainingInputSource:
    """Bind the exact live registered training provenance and preprocessor."""

    if type(corpus) is not Experiment002Corpus:
        raise TypeError("corpus must be an exact Experiment002Corpus")
    if type(cache) is not PCMCacheSplitView:
        raise TypeError("cache must be an exact PCMCacheSplitView")
    if type(normalization) is not VerifiedNormalization:
        raise TypeError("normalization must be an exact VerifiedNormalization")
    corpus_snapshot = _snapshot_registered_corpus(corpus)
    cache_snapshot = _snapshot_registered_cache(cache, corpus)
    normalization_contents = _snapshot_registered_normalization(normalization)
    preprocessor = Experiment002TrainingPreprocessor(corpus, cache, normalization)
    if type(preprocessor) is not Experiment002TrainingPreprocessor:
        raise TypeError("training preprocessor must have the exact registered type")
    if _snapshot_registered_corpus(corpus) != corpus_snapshot:
        raise Experiment002TrainingPopulationError(
            "registered corpus changed while training inputs were bound"
        )
    if _snapshot_registered_cache(cache, corpus) != cache_snapshot:
        raise Experiment002TrainingPopulationError(
            "registered cache changed while training inputs were bound"
        )
    if not hmac.compare_digest(
        _snapshot_registered_normalization(normalization),
        normalization_contents,
    ):
        raise Experiment002TrainingPopulationError(
            "registered normalization changed while training inputs were bound"
        )
    result = object.__new__(RegisteredTrainingInputSource)
    state = _IssuedSourceState(
        corpus=corpus,
        cache=cache,
        normalization=normalization,
        preprocessor=preprocessor,
        corpus_snapshot=corpus_snapshot,
        cache_snapshot=cache_snapshot,
        normalization_contents=normalization_contents,
        marker=_REGISTERED_SOURCE_MARKER,
    )
    with _ISSUED_SOURCES_LOCK:
        _ISSUED_SOURCES[result] = state
        _SOURCE_GUARDS[result] = _SourceIssuanceGuard(
            corpus=corpus,
            cache=cache,
            normalization=normalization,
            preprocessor=preprocessor,
            corpus_snapshot=tuple(corpus_snapshot),
            cache_snapshot=tuple(cache_snapshot),
            normalization_contents=bytes(normalization_contents),
            marker=_REGISTERED_SOURCE_MARKER,
        )
        _SOURCE_CONTEXT_GUARDS[result] = frozenset()
    return result


def verify_completed_registered_training_population(
    population: CompletedTrainingPopulation,
    *,
    seed: int,
    zero_based_epoch: int,
) -> None:
    """Require an intact final capability from the exact registered route."""

    _require_registered_context(seed, zero_based_epoch)
    snapshot = _completed_training_population_snapshot(population)
    if snapshot.route_marker is not _REGISTERED_EPOCH_MARKER:
        raise Experiment002TrainingPopulationError(
            "population was not completed through the registered route"
        )
    if snapshot.seed != seed or snapshot.zero_based_epoch != zero_based_epoch:
        raise Experiment002TrainingPopulationError(
            "population belongs to a different seed or epoch"
        )
    if (
        snapshot.example_count != TRAIN_EXAMPLE_COUNT
        or snapshot.batch_count != REGISTERED_BATCH_COUNT
        or len(snapshot.updates) != REGISTERED_BATCH_COUNT
    ):
        raise Experiment002TrainingPopulationError(
            "registered population counts differ"
        )


def _begin_registered_epoch_for_executor(
    source: RegisteredTrainingInputSource,
    *,
    executor_authority: RegisteredExecutorSessionAuthority,
    seed: int,
    zero_based_epoch: int,
) -> RegisteredTrainingEpoch:
    if type(source) is not RegisteredTrainingInputSource:
        raise TypeError("source must be a RegisteredTrainingInputSource")
    executor = _executor_session_snapshot(executor_authority)
    _require_registered_context(seed, zero_based_epoch)
    if executor.seed != seed:
        raise Experiment002TrainingPopulationError(
            "executor seed differs from the registered epoch"
        )
    _require_registered_executor_generation(
        executor_authority, zero_based_epoch * REGISTERED_BATCH_COUNT
    )
    source_state = _issued_source_state(source)
    with source_state.lock:
        _verify_registered_source_state(source_state)
        context = (seed, zero_based_epoch)
        if context in source_state.begun_contexts:
            raise Experiment002TrainingPopulationError(
                "this registered seed and epoch was already begun"
            )
        with _ISSUED_SOURCES_LOCK:
            guarded_contexts = _SOURCE_CONTEXT_GUARDS.get(source)
            if type(guarded_contexts) is not frozenset or guarded_contexts != frozenset(
                source_state.begun_contexts
            ):
                raise Experiment002TrainingPopulationError(
                    "registered source contexts differ from append-only authority"
                )
            next_guarded_contexts = guarded_contexts | {context}
        source_state.begun_contexts.add(context)
        with _ISSUED_SOURCES_LOCK:
            _SOURCE_CONTEXT_GUARDS[source] = next_guarded_contexts
        examples = build_training_epoch(
            source_state.corpus,
            seed=seed,
            epoch=zero_based_epoch,
        )
        _verify_registered_source_state(source_state)
        construction_authority = _issue_registered_epoch_construction(
            source=source,
            executor_authority=executor_authority,
            executor_session_token=executor.session_token,
            seed=seed,
            zero_based_epoch=zero_based_epoch,
        )
        return _construct_training_epoch(
            examples,
            source_state.preprocessor,
            seed=seed,
            zero_based_epoch=zero_based_epoch,
            layout=_REGISTERED_LAYOUT,
            source=source,
            route_marker=_REGISTERED_EPOCH_MARKER,
            executor_authority=executor_authority,
            executor_session_token=executor.session_token,
            construction_authority=construction_authority,
        )


def _issue_registered_epoch_construction(
    *,
    source: RegisteredTrainingInputSource,
    executor_authority: RegisteredExecutorSessionAuthority,
    executor_session_token: object,
    seed: int,
    zero_based_epoch: int,
) -> _RegisteredEpochConstructionAuthority:
    result = object.__new__(_RegisteredEpochConstructionAuthority)
    guard = _RegisteredEpochConstructionGuard(
        source=source,
        executor_authority=executor_authority,
        executor_session_token=executor_session_token,
        seed=seed,
        zero_based_epoch=zero_based_epoch,
    )
    with _REGISTERED_EPOCH_CONSTRUCTION_LOCK:
        _REGISTERED_EPOCH_CONSTRUCTION_GUARDS[result] = guard
    return result


def _consume_registered_epoch_construction(
    authority: _RegisteredEpochConstructionAuthority,
    *,
    source: RegisteredTrainingInputSource,
    executor_authority: RegisteredExecutorSessionAuthority,
    executor_session_token: object,
    seed: int,
    zero_based_epoch: int,
) -> None:
    if type(authority) is not _RegisteredEpochConstructionAuthority:
        raise TypeError("construction_authority must be an exact registered authority")
    with _REGISTERED_EPOCH_CONSTRUCTION_LOCK:
        guard = _REGISTERED_EPOCH_CONSTRUCTION_GUARDS.get(authority)
        if (
            type(guard) is not _RegisteredEpochConstructionGuard
            or authority in _CONSUMED_EPOCH_CONSTRUCTION_AUTHORITIES
            or guard.source is not source
            or guard.executor_authority is not executor_authority
            or guard.executor_session_token is not executor_session_token
            or guard.seed != seed
            or guard.zero_based_epoch != zero_based_epoch
        ):
            raise Experiment002TrainingPopulationError(
                "registered epoch construction authority differs"
            )
        _CONSUMED_EPOCH_CONSTRUCTION_AUTHORITIES.add(authority)


def _begin_training_epoch(
    examples: tuple[_Example, ...],
    preprocessor: _TrainingPreprocessor,
    *,
    seed: int,
    zero_based_epoch: int,
    layout: _TrainingLayout,
) -> RegisteredTrainingEpoch:
    """Private synthetic-layout seam, permanently outside registered authority."""

    return _construct_training_epoch(
        examples,
        preprocessor,
        seed=seed,
        zero_based_epoch=zero_based_epoch,
        layout=layout,
    )


def _construct_training_epoch(
    examples: tuple[_Example, ...],
    preprocessor: _TrainingPreprocessor,
    *,
    seed: int,
    zero_based_epoch: int,
    layout: _TrainingLayout,
    source: RegisteredTrainingInputSource | None = None,
    route_marker: object | None = None,
    executor_authority: RegisteredExecutorSessionAuthority | None = None,
    executor_session_token: object | None = None,
    construction_authority: _RegisteredEpochConstructionAuthority | None = None,
) -> RegisteredTrainingEpoch:
    """Private tiny-layout seam retaining the production state machine."""

    _require_uint32(seed, "seed")
    _require_uint32(zero_based_epoch, "zero_based_epoch")
    if type(layout) is not _TrainingLayout:
        raise TypeError("layout must be a _TrainingLayout")
    if type(examples) is not tuple:
        raise TypeError("examples must be an exact tuple")
    if len(examples) != layout.example_count:
        raise Experiment002TrainingPopulationError(
            "training plan count differs from its layout"
        )
    if not callable(getattr(preprocessor, "command_model_input", None)) or not callable(
        getattr(preprocessor, "silence_model_input", None)
    ):
        raise TypeError("preprocessor does not implement the training interface")
    if route_marker is None:
        if (
            source is not None
            or executor_authority is not None
            or executor_session_token is not None
            or construction_authority is not None
        ):
            raise Experiment002TrainingPopulationError(
                "an unregistered epoch cannot bind a registered source"
            )
    elif route_marker is not _REGISTERED_EPOCH_MARKER:
        raise Experiment002TrainingPopulationError("epoch route marker is invalid")
    elif (
        source is None
        or layout != _REGISTERED_LAYOUT
        or type(executor_authority) is not RegisteredExecutorSessionAuthority
        or type(executor_session_token) is not object
        or type(construction_authority) is not _RegisteredEpochConstructionAuthority
    ):
        raise Experiment002TrainingPopulationError(
            "registered epoch requires its source and exact layout"
        )
    if route_marker is _REGISTERED_EPOCH_MARKER:
        assert source is not None
        assert executor_authority is not None
        assert executor_session_token is not None
        assert construction_authority is not None
        _consume_registered_epoch_construction(
            construction_authority,
            source=source,
            executor_authority=executor_authority,
            executor_session_token=executor_session_token,
            seed=seed,
            zero_based_epoch=zero_based_epoch,
        )
    snapshots = _snapshot_plan(examples)
    hasher = hashlib.sha256()
    guard_hasher = hashlib.sha256()
    hasher.update(TRAINING_POPULATION_DOMAIN)
    guard_hasher.update(TRAINING_POPULATION_DOMAIN)
    context_frame = struct.pack("<III", seed, zero_based_epoch, layout.example_count)
    hasher.update(context_frame)
    guard_hasher.update(context_frame)
    result = object.__new__(RegisteredTrainingEpoch)
    state = _EpochState(
        seed=seed,
        zero_based_epoch=zero_based_epoch,
        layout=layout,
        examples=examples,
        example_snapshots=snapshots,
        preprocessor=preprocessor,
        source=source,
        route_marker=route_marker,
        session_token=object(),
        executor_authority=executor_authority,
        executor_session_token=executor_session_token,
        hasher=hasher,
    )
    with _ISSUED_EPOCHS_LOCK:
        _ISSUED_EPOCHS[result] = state
        _EPOCH_GUARDS[result] = _EpochIssuanceGuard(
            seed=seed,
            zero_based_epoch=zero_based_epoch,
            layout=dataclass_replace(layout),
            examples=tuple(examples),
            example_snapshots=tuple(snapshots),
            preprocessor=preprocessor,
            source=source,
            route_marker=route_marker,
            session_token=state.session_token,
            executor_authority=executor_authority,
            executor_session_token=executor_session_token,
            hasher=hasher,
        )
        _EPOCH_ACCEPTED_GUARDS[result] = ()
        _EPOCH_HASH_MIRRORS[result] = guard_hasher
        _EPOCH_LIFECYCLES[result] = _EpochLifecycleGuard(
            phase="OPEN",
            cursor=0,
            batch_index=0,
            current_batch=None,
            pending_verifications=0,
            ready_receipt=None,
            finished=False,
        )
    return result


@dataclass(frozen=True, slots=True)
class _RegisteredBatchBinding:
    executor_session_token: object
    epoch_session_token: object
    batch_token: object
    seed: int
    zero_based_epoch: int
    zero_based_global_update: int
    batch_size: int


def _executor_next_batch(
    epoch: RegisteredTrainingEpoch,
    executor_authority: RegisteredExecutorSessionAuthority,
    global_update: int,
) -> MaterializedTrainingBatch:
    state = _require_executor_epoch(epoch, executor_authority)
    with state.lock:
        state.active_executor_authority = executor_authority
        try:
            return epoch.next_batch(global_update)
        finally:
            state.active_executor_authority = None


def _executor_verify_pending_batch(
    epoch: RegisteredTrainingEpoch,
    executor_authority: RegisteredExecutorSessionAuthority,
    batch: MaterializedTrainingBatch,
    *,
    global_update: int,
    phase: Literal["PRE", "POST"],
) -> None:
    if phase not in {"PRE", "POST"}:
        raise Experiment002TrainingPopulationError(
            "registered verification phase must be PRE or POST"
        )
    state = _require_executor_epoch(epoch, executor_authority)
    with state.lock:
        state.active_executor_authority = executor_authority
        state.active_verification_phase = phase
        try:
            epoch.verify_pending_batch(batch, global_update=global_update)
        finally:
            state.active_verification_phase = None
            state.active_executor_authority = None


def _executor_batch_binding(
    epoch: RegisteredTrainingEpoch,
    executor_authority: RegisteredExecutorSessionAuthority,
    batch: MaterializedTrainingBatch,
) -> _RegisteredBatchBinding:
    state = _require_executor_epoch(epoch, executor_authority)
    with state.lock:
        if (
            state.phase != "BATCH_PENDING"
            or state.current_batch is not batch
            or state.pending_verifications != 2
            or state.active_verification_phase is not None
        ):
            _terminal_fail_epoch_state(epoch, state)
            raise Experiment002TrainingPopulationError(
                "executor batch binding requires exact completed POST verification"
            )
        batch_state = _verify_pending_batch(state, batch)
        return _RegisteredBatchBinding(
            executor_session_token=state.executor_session_token,
            epoch_session_token=state.session_token,
            batch_token=batch_state.batch_token,
            seed=state.seed,
            zero_based_epoch=state.zero_based_epoch,
            zero_based_global_update=batch_state.global_update,
            batch_size=batch_state.label_indices.shape[0],
        )


def _executor_complete_update(
    epoch: RegisteredTrainingEpoch,
    executor_authority: RegisteredExecutorSessionAuthority,
    batch: MaterializedTrainingBatch,
    transition: RegisteredOptimizerTransition,
) -> CompletedTrainingUpdate:
    state = _require_executor_epoch(epoch, executor_authority)
    with state.lock:
        try:
            transition_snapshot = _transition_snapshot(
                transition, required_phase="ISSUED"
            )
            batch_state = _issued_batch_state(batch)
            if (
                state.current_batch is not batch
                or transition_snapshot.executor_session_token
                is not state.executor_session_token
                or transition_snapshot.epoch_session_token is not state.session_token
                or transition_snapshot.batch_token is not batch_state.batch_token
                or transition_snapshot.seed != state.seed
                or transition_snapshot.zero_based_epoch != state.zero_based_epoch
                or transition_snapshot.zero_based_global_update
                != batch_state.global_update
                or transition_snapshot.batch_size != batch_state.label_indices.shape[0]
            ):
                raise Experiment002TrainingPopulationError(
                    "optimizer transition does not belong to this executor batch"
                )
        except BaseException:
            _terminal_fail_epoch_state(epoch, state)
            raise
        state.active_executor_authority = executor_authority
        state.active_transition = transition
        try:
            return epoch.complete_update(
                batch,
                learning_rate=transition_snapshot.learning_rate,
                batch_mean_training_loss=(transition_snapshot.batch_mean_training_loss),
                returned_preclip_l2_norm=(transition_snapshot.returned_preclip_l2_norm),
            )
        finally:
            state.active_transition = None
            state.active_executor_authority = None


def _executor_accept_completed_update(
    epoch: RegisteredTrainingEpoch,
    executor_authority: RegisteredExecutorSessionAuthority,
    receipt: CompletedTrainingUpdate,
    transition: RegisteredOptimizerTransition,
    consumption: TraceConsumedTransition,
) -> None:
    state = _require_executor_epoch(epoch, executor_authority)
    with state.lock:
        local_transition: RegisteredOptimizerTransition | None = None
        try:
            local_receipt = state.ready_receipt
            if type(local_receipt) is not CompletedTrainingUpdate:
                raise Experiment002TrainingPopulationError(
                    "executor epoch lost its local ready receipt"
                )
            receipt_state = _issued_receipt_state(local_receipt)
            local_transition = receipt_state.optimizer_transition
            if (
                type(local_transition) is not RegisteredOptimizerTransition
                or receipt_state.session_token is not state.session_token
                or receipt_state.snapshot.session_token is not state.session_token
                or receipt_state.batch_token is not receipt_state.snapshot.batch_token
                or receipt_state.optimizer_transition_token is None
            ):
                raise Experiment002TrainingPopulationError(
                    "receipt lost its local optimizer transition authority"
                )
            # Only the exact transition already bound into the local ready
            # receipt may become active. The caller-supplied candidate is never
            # adopted, so a foreign capability cannot be failed as collateral.
            state.active_transition = local_transition
            if receipt is not local_receipt:
                raise Experiment002TrainingPopulationError(
                    "receipt does not belong to this executor epoch"
                )
            transition_snapshot = _transition_snapshot(
                local_transition, required_phase="TRACE_CONSUMED"
            )
            consumption_snapshot = _trace_consumption_snapshot(
                consumption, required_used=False
            )
            if (
                transition is not local_transition
                or transition_snapshot.executor_session_token
                is not state.executor_session_token
                or transition_snapshot.epoch_session_token is not state.session_token
                or transition_snapshot.batch_token is not receipt_state.batch_token
                or transition_snapshot.transition_token
                is not receipt_state.optimizer_transition_token
                or consumption_snapshot.executor_session_token
                is not state.executor_session_token
                or consumption_snapshot.epoch_session_token is not state.session_token
                or consumption_snapshot.batch_token is not receipt_state.batch_token
                or consumption_snapshot.transition_token
                is not receipt_state.optimizer_transition_token
                or consumption_snapshot.receipt_token is not receipt_state.receipt_token
                or consumption_snapshot.zero_based_global_update
                != receipt_state.snapshot.zero_based_global_update
            ):
                raise Experiment002TrainingPopulationError(
                    "trace consumption does not belong to this executor receipt"
                )
        except BaseException:
            _terminal_fail_epoch_state(epoch, state)
            state.active_transition = None
            raise
        state.active_executor_authority = executor_authority
        state.active_consumption = consumption
        try:
            epoch.accept_completed_update(receipt)
        finally:
            state.active_consumption = None
            state.active_transition = None
            state.active_executor_authority = None


def _executor_finish_epoch(
    epoch: RegisteredTrainingEpoch,
    executor_authority: RegisteredExecutorSessionAuthority,
) -> CompletedTrainingPopulation:
    state = _require_executor_epoch(epoch, executor_authority)
    with state.lock:
        state.active_executor_authority = executor_authority
        try:
            return epoch.finish()
        finally:
            state.active_executor_authority = None


def _executor_fail_epoch(
    epoch: RegisteredTrainingEpoch | None,
    executor_authority: RegisteredExecutorSessionAuthority,
) -> None:
    if epoch is None:
        return
    state = _require_executor_epoch(epoch, executor_authority, require_live=False)
    with state.lock:
        _terminal_fail_epoch_state(epoch, state)
        state.active_executor_authority = None
        state.active_transition = None
        state.active_consumption = None
        state.active_verification_phase = None


def _require_executor_epoch(
    epoch: RegisteredTrainingEpoch,
    executor_authority: RegisteredExecutorSessionAuthority,
    *,
    require_live: bool = True,
) -> _EpochState:
    executor = _executor_session_snapshot(executor_authority, require_live=require_live)
    state = _issued_epoch_state(epoch)
    if (
        state.route_marker is not _REGISTERED_EPOCH_MARKER
        or state.executor_authority is not executor_authority
        or state.executor_session_token is not executor.session_token
        or state.seed != executor.seed
    ):
        raise Experiment002TrainingPopulationError(
            "training epoch does not belong to this exact executor"
        )
    return state


def _materialize_next_batch(
    epoch: RegisteredTrainingEpoch,
    state: _EpochState,
    global_update: int,
) -> MaterializedTrainingBatch:
    first = state.cursor
    stop = min(first + state.layout.batch_size, state.layout.example_count)
    examples = state.examples[first:stop]
    expected_snapshots = state.example_snapshots[first:stop]
    if _snapshot_plan(examples) != expected_snapshots:
        raise Experiment002TrainingPopulationError(
            "training plan changed before batch materialization"
        )
    size = stop - first
    model_inputs = np.empty(
        (size, state.layout.mel_bins, state.layout.time_frames),
        dtype=np.float32,
        order="C",
    )
    labels = np.empty(size, dtype=np.int64)
    framed = bytearray()
    for offset, example in enumerate(examples):
        if type(example) is CommandExample:
            produced = state.preprocessor.command_model_input(
                example,
                seed=state.seed,
                epoch=state.zero_based_epoch,
            )
        elif type(example) is WindowExample:
            produced = state.preprocessor.silence_model_input(
                example,
                seed=state.seed,
                epoch=state.zero_based_epoch,
            )
        else:
            raise TypeError("training plan contains an invalid example type")
        feature = _validated_feature(produced, state.layout)
        model_inputs[offset] = feature
        labels[offset] = example.label_index
        identity = example.identity
        payload = model_inputs[offset].astype("<f4", copy=False).tobytes(order="C")
        framed.extend(struct.pack("<I", len(identity)))
        framed.extend(identity)
        framed.extend(struct.pack("<B", example.label_index))
        framed.extend(struct.pack("<Q", len(payload)))
        framed.extend(payload)
    if _snapshot_plan(examples) != expected_snapshots:
        raise Experiment002TrainingPopulationError(
            "training plan changed during batch materialization"
        )
    model_inputs.setflags(write=False)
    labels.setflags(write=False)
    _validate_materialized_arrays(model_inputs, labels, examples, state.layout)
    model_sha256 = _array_sha256(model_inputs)
    labels_sha256 = _array_sha256(labels)
    _validate_materialized_arrays(model_inputs, labels, examples, state.layout)
    if model_sha256 != _array_sha256(model_inputs) or labels_sha256 != _array_sha256(
        labels
    ):
        raise Experiment002TrainingPopulationError(
            "materialized batch changed while it was issued"
        )
    state.hasher.update(framed)
    with _ISSUED_EPOCHS_LOCK:
        guard_hasher = _EPOCH_HASH_MIRRORS.get(epoch)
    if guard_hasher is None:
        raise Experiment002TrainingPopulationError(
            "training population digest mirror is missing"
        )
    guard_hasher.update(framed)
    _verify_epoch_hashers(epoch, state)
    batch = object.__new__(MaterializedTrainingBatch)
    batch_state = _IssuedBatchState(
        session_token=state.session_token,
        batch_token=object(),
        batch_index=state.batch_index,
        global_update=global_update,
        first_example=first,
        examples=examples,
        example_snapshots=expected_snapshots,
        model_inputs=model_inputs,
        label_indices=labels,
        model_inputs_sha256=model_sha256,
        label_indices_sha256=labels_sha256,
    )
    with _ISSUED_BATCHES_LOCK:
        _ISSUED_BATCHES[batch] = batch_state
        _BATCH_GUARDS[batch] = _BatchIssuanceGuard(
            session_token=batch_state.session_token,
            batch_token=batch_state.batch_token,
            batch_index=batch_state.batch_index,
            global_update=batch_state.global_update,
            first_example=batch_state.first_example,
            examples=tuple(batch_state.examples),
            example_snapshots=tuple(batch_state.example_snapshots),
            model_inputs=batch_state.model_inputs,
            label_indices=batch_state.label_indices,
            model_inputs_sha256=batch_state.model_inputs_sha256,
            label_indices_sha256=batch_state.label_indices_sha256,
        )
    return batch


def _verify_pending_batch(
    state: _EpochState,
    batch: MaterializedTrainingBatch,
) -> _IssuedBatchState:
    if type(batch) is not MaterializedTrainingBatch:
        raise TypeError("batch must be a MaterializedTrainingBatch")
    if batch is not state.current_batch:
        raise Experiment002TrainingPopulationError(
            "batch does not belong to this pending update"
        )
    batch_state = _issued_batch_state(batch)
    if (
        batch_state.session_token is not state.session_token
        or batch_state.batch_index != state.batch_index
        or batch_state.global_update
        != state.zero_based_epoch * state.layout.batch_count + state.batch_index
        or batch_state.first_example != state.cursor
    ):
        raise Experiment002TrainingPopulationError("pending batch context differs")
    _verify_epoch_backing(state, full=False)
    if _snapshot_plan(batch_state.examples) != batch_state.example_snapshots:
        raise Experiment002TrainingPopulationError("pending batch examples changed")
    _validate_materialized_arrays(
        batch_state.model_inputs,
        batch_state.label_indices,
        batch_state.examples,
        state.layout,
    )
    first_models = _array_sha256(batch_state.model_inputs)
    first_labels = _array_sha256(batch_state.label_indices)
    _validate_materialized_arrays(
        batch_state.model_inputs,
        batch_state.label_indices,
        batch_state.examples,
        state.layout,
    )
    if (
        first_models != _array_sha256(batch_state.model_inputs)
        or not hmac.compare_digest(first_models, batch_state.model_inputs_sha256)
        or first_labels != _array_sha256(batch_state.label_indices)
        or not hmac.compare_digest(first_labels, batch_state.label_indices_sha256)
    ):
        raise Experiment002TrainingPopulationError(
            "pending batch bytes changed after issuance"
        )
    return batch_state


def _issue_completed_update(
    state: _EpochState,
    batch: _IssuedBatchState,
    *,
    learning_rate: float,
    batch_mean_training_loss: np.float32,
    returned_preclip_l2_norm: np.float32,
    optimizer_transition_token: object | None = None,
    optimizer_transition: RegisteredOptimizerTransition | None = None,
) -> CompletedTrainingUpdate:
    if type(learning_rate) is not float:
        raise TypeError("learning_rate must be an exact float")
    if not math.isfinite(learning_rate) or learning_rate < 0.0:
        raise Experiment002TrainingPopulationError(
            "learning_rate must be finite and nonnegative"
        )
    loss = _require_finite_float32(batch_mean_training_loss, "batch_mean_training_loss")
    norm = _require_finite_float32(returned_preclip_l2_norm, "returned_preclip_l2_norm")
    snapshot = _CompletedTrainingUpdateSnapshot(
        session_token=state.session_token,
        batch_token=batch.batch_token,
        receipt_token=object(),
        optimizer_transition_token=optimizer_transition_token,
        seed=state.seed,
        zero_based_epoch=state.zero_based_epoch,
        batch_index=state.batch_index,
        zero_based_global_update=batch.global_update,
        batch_size=batch.label_indices.shape[0],
        learning_rate=learning_rate,
        learning_rate_bytes=struct.pack("<d", learning_rate),
        batch_mean_training_loss=loss,
        batch_mean_training_loss_bytes=struct.pack("<f", loss),
        returned_preclip_l2_norm=norm,
        returned_preclip_l2_norm_bytes=struct.pack("<f", norm),
    )
    receipt = object.__new__(CompletedTrainingUpdate)
    receipt_state = _IssuedReceiptState(
        snapshot=snapshot,
        authority_payload=_frame_update_snapshot(snapshot),
        session_token=snapshot.session_token,
        batch_token=snapshot.batch_token,
        receipt_token=snapshot.receipt_token,
        optimizer_transition_token=snapshot.optimizer_transition_token,
        optimizer_transition=optimizer_transition,
    )
    with _ISSUED_RECEIPTS_LOCK:
        _ISSUED_RECEIPTS[receipt] = receipt_state
        _RECEIPT_GUARDS[receipt] = _ReceiptIssuanceGuard(
            authority_payload=receipt_state.authority_payload,
            session_token=receipt_state.session_token,
            batch_token=receipt_state.batch_token,
            receipt_token=receipt_state.receipt_token,
            optimizer_transition_token=receipt_state.optimizer_transition_token,
            optimizer_transition=receipt_state.optimizer_transition,
        )
    return receipt


def _verify_receipt_matches_batch(
    epoch: _EpochState,
    receipt: _IssuedReceiptState,
    batch: _IssuedBatchState,
) -> None:
    _validate_receipt_state(receipt)
    snapshot = receipt.snapshot
    if (
        snapshot.session_token is not epoch.session_token
        or snapshot.batch_token is not batch.batch_token
        or snapshot.seed != epoch.seed
        or snapshot.zero_based_epoch != epoch.zero_based_epoch
        or snapshot.batch_index != epoch.batch_index
        or snapshot.zero_based_global_update != batch.global_update
        or snapshot.batch_size != batch.label_indices.shape[0]
        or snapshot.learning_rate_bytes != struct.pack("<d", snapshot.learning_rate)
        or snapshot.batch_mean_training_loss_bytes
        != struct.pack("<f", snapshot.batch_mean_training_loss)
        or snapshot.returned_preclip_l2_norm_bytes
        != struct.pack("<f", snapshot.returned_preclip_l2_norm)
    ):
        raise Experiment002TrainingPopulationError(
            "completed update receipt context or scalar bytes differ"
        )


def _issue_completed_population(
    state: _EpochState,
) -> CompletedTrainingPopulation:
    updates = tuple(_copy_update_snapshot(item) for item in state.accepted_updates)
    optimizer_transitions = tuple(
        authority.optimizer_transition
        for authority in state.accepted_update_authorities
        if authority.optimizer_transition is not None
    )
    trace_consumptions = tuple(
        authority.trace_consumption
        for authority in state.accepted_update_authorities
        if authority.trace_consumption is not None
    )
    snapshot = _CompletedTrainingPopulationSnapshot(
        session_token=state.session_token,
        seed=state.seed,
        zero_based_epoch=state.zero_based_epoch,
        sha256=state.hasher.hexdigest(),
        example_count=state.layout.example_count,
        batch_count=state.layout.batch_count,
        updates=updates,
        ordered_batch_tokens=tuple(item.batch_token for item in updates),
        ordered_receipt_tokens=tuple(item.receipt_token for item in updates),
        ordered_optimizer_transition_tokens=tuple(
            item.optimizer_transition_token
            for item in updates
            if item.optimizer_transition_token is not None
        ),
        ordered_optimizer_transitions=optimizer_transitions,
        ordered_trace_consumptions=trace_consumptions,
        route_marker=state.route_marker,
    )
    result = object.__new__(CompletedTrainingPopulation)
    issued_state = _IssuedPopulationState(
        snapshot=snapshot,
        sha256=snapshot.sha256,
        seed=snapshot.seed,
        zero_based_epoch=snapshot.zero_based_epoch,
        example_count=snapshot.example_count,
        batch_count=snapshot.batch_count,
        session_token=snapshot.session_token,
        update_authority_payloads=tuple(
            _frame_update_snapshot(item) for item in snapshot.updates
        ),
        ordered_batch_tokens=snapshot.ordered_batch_tokens,
        ordered_receipt_tokens=snapshot.ordered_receipt_tokens,
        ordered_optimizer_transition_tokens=(
            snapshot.ordered_optimizer_transition_tokens
        ),
        ordered_optimizer_transitions=snapshot.ordered_optimizer_transitions,
        ordered_trace_consumptions=snapshot.ordered_trace_consumptions,
        route_marker=snapshot.route_marker,
    )
    with _ISSUED_POPULATIONS_LOCK:
        _ISSUED_POPULATIONS[result] = issued_state
        _POPULATION_GUARDS[result] = _PopulationIssuanceGuard(
            sha256=issued_state.sha256,
            seed=issued_state.seed,
            zero_based_epoch=issued_state.zero_based_epoch,
            example_count=issued_state.example_count,
            batch_count=issued_state.batch_count,
            session_token=issued_state.session_token,
            update_authority_payloads=issued_state.update_authority_payloads,
            ordered_batch_tokens=issued_state.ordered_batch_tokens,
            ordered_receipt_tokens=issued_state.ordered_receipt_tokens,
            ordered_optimizer_transition_tokens=(
                issued_state.ordered_optimizer_transition_tokens
            ),
            ordered_optimizer_transitions=(issued_state.ordered_optimizer_transitions),
            ordered_trace_consumptions=issued_state.ordered_trace_consumptions,
            route_marker=issued_state.route_marker,
        )
    return result


def _completed_training_update_snapshot(
    receipt: CompletedTrainingUpdate,
) -> _CompletedTrainingUpdateSnapshot:
    """Return the immutable exact receipt snapshot for trusted evidence code."""

    return dataclass_replace(_issued_receipt_state(receipt).snapshot)


def _completed_training_update_token(receipt: CompletedTrainingUpdate) -> object:
    """Return the opaque receipt token for trusted cross-module binding."""

    return _issued_receipt_state(receipt).snapshot.receipt_token


def _completed_training_population_snapshot(
    population: CompletedTrainingPopulation,
) -> _CompletedTrainingPopulationSnapshot:
    """Return ordered update/token snapshots for trusted history code."""

    return dataclass_replace(_issued_population_state(population).snapshot)


def _verify_epoch_backing(state: _EpochState, *, full: bool) -> None:
    if full and _snapshot_plan(state.examples) != state.example_snapshots:
        raise Experiment002TrainingPopulationError("training epoch plan changed")
    if state.route_marker is _REGISTERED_EPOCH_MARKER:
        if state.source is None:
            raise Experiment002TrainingPopulationError(
                "registered epoch lost its source binding"
            )
        source_state = _issued_source_state(state.source)
        if full:
            _verify_registered_source_state(source_state)
        else:
            _verify_registered_source_fast(source_state)
        if source_state.preprocessor is not state.preprocessor:
            raise Experiment002TrainingPopulationError(
                "registered epoch preprocessor binding changed"
            )


def _verify_epoch_hashers(
    epoch: RegisteredTrainingEpoch,
    state: _EpochState,
) -> None:
    with _ISSUED_EPOCHS_LOCK:
        mirror_hasher = _EPOCH_HASH_MIRRORS.get(epoch)
    if mirror_hasher is None:
        raise Experiment002TrainingPopulationError(
            "training population digest mirror is missing"
        )
    first = state.hasher.hexdigest()
    mirror = mirror_hasher.hexdigest()
    if (
        not _is_sha256(first)
        or not _is_sha256(mirror)
        or not hmac.compare_digest(first, mirror)
        or first != state.hasher.hexdigest()
        or mirror != mirror_hasher.hexdigest()
    ):
        raise Experiment002TrainingPopulationError(
            "training population digest differs from its independent mirror"
        )


def _sync_epoch_lifecycle(
    epoch: RegisteredTrainingEpoch,
    state: _EpochState,
) -> None:
    lifecycle = _EpochLifecycleGuard(
        phase=state.phase,
        cursor=state.cursor,
        batch_index=state.batch_index,
        current_batch=state.current_batch,
        pending_verifications=state.pending_verifications,
        ready_receipt=state.ready_receipt,
        finished=state.finished,
    )
    with _ISSUED_EPOCHS_LOCK:
        _EPOCH_LIFECYCLES[epoch] = lifecycle


def _terminal_fail_epoch_state(
    epoch: RegisteredTrainingEpoch,
    state: _EpochState,
) -> None:
    state.phase = "FAILED"
    if state.route_marker is _REGISTERED_EPOCH_MARKER:
        transition = state.active_transition
        if transition is not None:
            with suppress(BaseException):
                _fail_optimizer_transition(transition)
        authority = state.executor_authority
        if authority is not None:
            with suppress(BaseException):
                _fail_registered_executor_session(authority)
    with suppress(BaseException):
        _sync_epoch_lifecycle(epoch, state)


def _verify_registered_source_state(state: _IssuedSourceState) -> None:
    if state.marker is not _REGISTERED_SOURCE_MARKER:
        raise Experiment002TrainingPopulationError(
            "training source route marker differs"
        )
    if _snapshot_registered_corpus(state.corpus) != state.corpus_snapshot:
        raise Experiment002TrainingPopulationError(
            "registered corpus changed after input binding"
        )
    if _snapshot_registered_cache(state.cache, state.corpus) != state.cache_snapshot:
        raise Experiment002TrainingPopulationError(
            "registered cache changed after input binding"
        )
    current_normalization = _snapshot_registered_normalization(state.normalization)
    if not hmac.compare_digest(current_normalization, state.normalization_contents):
        raise Experiment002TrainingPopulationError(
            "registered normalization changed after input binding"
        )
    _verify_registered_preprocessor(state, full=True)


def _verify_registered_source_fast(state: _IssuedSourceState) -> None:
    """Recheck live trust roots without rescanning the 84,843-row inventory."""

    if state.marker is not _REGISTERED_SOURCE_MARKER:
        raise Experiment002TrainingPopulationError(
            "training source route marker differs"
        )
    cache = state.cache
    if type(cache) is not PCMCacheSplitView or cache.split != "train":
        raise Experiment002TrainingPopulationError(
            "registered training cache view changed"
        )
    owner = cache._cache
    if type(owner) is not Experiment002PCMCache or owner.closed:
        raise Experiment002TrainingPopulationError(
            "registered training cache is no longer live"
        )
    if (
        len(state.cache_snapshot) < 2
        or type(state.cache_snapshot[1]) is not int
        or state.cache_snapshot[1] != id(owner)
    ):
        raise Experiment002TrainingPopulationError(
            "registered training cache owner binding changed"
        )
    identity = owner.identity
    if (
        type(identity) is not PCMCacheIdentity
        or identity.byte_count != REGISTERED_FILE_BYTES
        or not hmac.compare_digest(identity.sha256, REGISTERED_PCM_CACHE_SHA256)
    ):
        raise Experiment002TrainingPopulationError(
            "registered training cache identity changed"
        )
    current_normalization = _snapshot_registered_normalization(state.normalization)
    if not hmac.compare_digest(current_normalization, state.normalization_contents):
        raise Experiment002TrainingPopulationError(
            "registered normalization changed after input binding"
        )
    _verify_registered_preprocessor(state, full=False)


def _verify_registered_preprocessor(
    state: _IssuedSourceState,
    *,
    full: bool,
) -> None:
    preprocessor = state.preprocessor
    if (
        type(preprocessor) is not Experiment002TrainingPreprocessor
        or preprocessor._corpus is not state.corpus
        or preprocessor._cache is not state.cache
    ):
        raise Experiment002TrainingPopulationError(
            "registered training preprocessor provenance differs"
        )
    if full and (
        type(preprocessor._train_commands) is not frozenset
        or type(preprocessor._train_backgrounds) is not frozenset
        or {id(source) for source in preprocessor._train_commands}
        != {id(source) for source in state.corpus.train_commands}
        or {id(source) for source in preprocessor._train_backgrounds}
        != {id(source) for source in state.corpus.train_backgrounds}
    ):
        raise Experiment002TrainingPopulationError(
            "registered training preprocessor inventories differ"
        )
    expected_stats = state.normalization.stats
    for observed, expected in (
        (preprocessor._stats.means, expected_stats.means),
        (preprocessor._stats.standard_deviations, expected_stats.standard_deviations),
    ):
        if (
            type(observed) is not np.ndarray
            or observed.dtype != np.dtype(np.float32)
            or observed.shape != (MEL_BINS,)
            or not hmac.compare_digest(
                observed.astype("<f4", copy=False).tobytes(order="C"),
                expected.astype("<f4", copy=False).tobytes(order="C"),
            )
        ):
            raise Experiment002TrainingPopulationError(
                "registered training preprocessor statistics differ"
            )


def _snapshot_registered_corpus(corpus: Experiment002Corpus) -> tuple[object, ...]:
    _require_registered_training_corpus(corpus)
    if (
        type(corpus.train_commands) is not tuple
        or type(corpus.train_backgrounds) is not tuple
    ):
        raise TypeError("registered corpus inventories must be exact tuples")
    commands = tuple(
        _registered_command_snapshot(source) for source in corpus.train_commands
    )
    backgrounds = tuple(
        _registered_background_snapshot(source) for source in corpus.train_backgrounds
    )
    return (
        id(corpus),
        corpus.manifest_sha256,
        corpus.test_command_count,
        corpus._train_inventory_sha256,
        commands,
        backgrounds,
    )


def _snapshot_registered_cache(
    cache: PCMCacheSplitView,
    corpus: Experiment002Corpus,
) -> tuple[object, ...]:
    if type(cache) is not PCMCacheSplitView:
        raise TypeError("cache must be an exact PCMCacheSplitView")
    if cache.split != "train":
        raise Experiment002TrainingPopulationError(
            "training inputs require the training cache split"
        )
    owner = cache._cache
    if type(owner) is not Experiment002PCMCache:
        raise TypeError("cache owner must be an exact Experiment002PCMCache")
    if owner.closed:
        raise Experiment002TrainingPopulationError("training PCM cache is closed")
    identity = owner.identity
    if (
        type(identity) is not PCMCacheIdentity
        or identity.byte_count != REGISTERED_FILE_BYTES
        or not hmac.compare_digest(identity.sha256, REGISTERED_PCM_CACHE_SHA256)
    ):
        raise Experiment002TrainingPopulationError(
            "training PCM cache identity differs from the frozen artifact"
        )
    parsed = owner._parsed_index
    if type(parsed) is not _ParsedIndex:
        raise TypeError("cache parsed index must have the exact registered type")
    _require_cache_inventory_matches_corpus(parsed, corpus)
    train_commands = tuple(
        sorted(
            (
                _cache_source_snapshot(source),
                _cache_entry_snapshot(entry),
            )
            for source, entry in parsed.train_commands.items()
        )
    )
    train_backgrounds = tuple(
        sorted(
            (
                _cache_source_snapshot(source),
                _cache_entry_snapshot(entry),
            )
            for source, entry in parsed.train_backgrounds.items()
        )
    )
    return (
        id(cache),
        id(owner),
        identity.byte_count,
        identity.sha256,
        train_commands,
        train_backgrounds,
    )


def _cache_source_snapshot(
    source: CommandSource | BackgroundSource,
) -> tuple[object, ...]:
    if type(source) is CommandSource:
        return (
            "command",
            id(source),
            source.manifest_index,
            source.path,
            source.word,
            source.label,
            source.sample_count,
            source.sha256,
        )
    if type(source) is BackgroundSource:
        return (
            "background",
            id(source),
            source.path,
            source.sample_count,
            source.sha256,
        )
    raise TypeError("cache inventories must contain exact source identities")


def _cache_entry_snapshot(entry: _IndexEntry) -> tuple[object, ...]:
    if type(entry) is not _IndexEntry:
        raise TypeError("cache inventory contains an invalid entry")
    return (
        type(entry),
        entry.sample_count,
        entry.payload_offset,
        bytes(entry.pcm_sha256),
    )


def _registered_command_snapshot(source: CommandSource) -> tuple[object, ...]:
    if type(source) is not CommandSource:
        raise TypeError(
            "training command inventories require exact CommandSource values"
        )
    return (
        id(source),
        type(source),
        source.manifest_index,
        source.path,
        source.word,
        source.label,
        source.sample_count,
        source.sha256,
    )


def _registered_background_snapshot(source: BackgroundSource) -> tuple[object, ...]:
    if type(source) is not BackgroundSource:
        raise TypeError(
            "training background inventories require exact BackgroundSource values"
        )
    return (id(source), type(source), source.path, source.sample_count, source.sha256)


def _require_cache_inventory_matches_corpus(
    parsed: _ParsedIndex,
    corpus: Experiment002Corpus,
) -> None:
    command_entries = parsed.train_commands
    background_entries = parsed.train_backgrounds
    if type(command_entries) is not dict or type(background_entries) is not dict:
        raise TypeError("training cache inventories must be exact dictionaries")
    if {id(source) for source in command_entries} != {
        id(source) for source in corpus.train_commands
    }:
        raise Experiment002TrainingPopulationError(
            "training cache command inventory differs from the corpus"
        )
    if {id(source) for source in background_entries} != {
        id(source) for source in corpus.train_backgrounds
    }:
        raise Experiment002TrainingPopulationError(
            "training cache background inventory differs from the corpus"
        )
    for command_source, command_entry in command_entries.items():
        _require_cache_entry_matches_source(command_source, command_entry)
    for background_source, background_entry in background_entries.items():
        _require_cache_entry_matches_source(background_source, background_entry)


def _require_cache_entry_matches_source(
    source: CommandSource | BackgroundSource,
    entry: _IndexEntry,
) -> None:
    if type(entry) is not _IndexEntry:
        raise TypeError("training cache inventory contains an invalid entry")
    if (
        entry.sample_count != source.sample_count
        or type(entry.payload_offset) is not int
        or entry.payload_offset < 0
        or type(entry.sample_count) is not int
        or type(entry.pcm_sha256) is not bytes
        or len(entry.pcm_sha256) != 32
    ):
        raise Experiment002TrainingPopulationError(
            "training cache entry shape differs from its corpus source"
        )


def _snapshot_registered_normalization(
    normalization: VerifiedNormalization,
) -> bytes:
    verify_registered_normalization(normalization)
    identity = normalization.identity
    contents = normalization.contents
    if (
        type(identity) is not NormalizationArtifactIdentity
        or type(contents) is not bytes
        or identity.byte_count != NORMALIZATION_ARTIFACT_BYTES
        or len(contents) != NORMALIZATION_ARTIFACT_BYTES
        or not hmac.compare_digest(identity.sha256, REGISTERED_NORMALIZATION_SHA256)
        or not hmac.compare_digest(
            hashlib.sha256(contents).hexdigest(), REGISTERED_NORMALIZATION_SHA256
        )
    ):
        raise Experiment002TrainingPopulationError(
            "normalization capability differs from the frozen artifact"
        )
    return bytes(contents)


def _snapshot_plan(examples: tuple[_Example, ...]) -> tuple[tuple[object, ...], ...]:
    identities: set[bytes] = set()
    result: list[tuple[object, ...]] = []
    for example in examples:
        if type(example) is CommandExample:
            command_source = example.source
            snapshot: tuple[object, ...] = (
                "command",
                id(example),
                id(command_source),
                command_source.manifest_index,
                command_source.path,
                command_source.word,
                command_source.label,
                command_source.sample_count,
                command_source.sha256,
                example.label,
                example.label_index,
                example.identity,
            )
        elif type(example) is WindowExample:
            background_source = example.source
            snapshot = (
                "window",
                id(example),
                id(background_source),
                background_source.path,
                background_source.sample_count,
                background_source.sha256,
                example.start_sample,
                example.label,
                example.label_index,
                example.identity,
            )
        else:
            raise TypeError("training plan contains an invalid example type")
        if type(example.identity) is not bytes or not example.identity:
            raise TypeError("training example identity must be non-empty bytes")
        if len(example.identity) > _UINT32_MAX:
            raise Experiment002TrainingPopulationError(
                "training example identity exceeds UINT32LE"
            )
        if example.identity in identities:
            raise Experiment002TrainingPopulationError(
                "training plan repeats an example identity"
            )
        if type(example.label_index) is not int or not 0 <= example.label_index < len(
            CLASS_ORDER
        ):
            raise Experiment002TrainingPopulationError(
                "training example label index is invalid"
            )
        identities.add(example.identity)
        result.append(snapshot)
    return tuple(result)


def _validated_feature(feature: object, layout: _TrainingLayout) -> FloatArray:
    if type(feature) is not np.ndarray:
        raise TypeError("preprocessor feature must be an exact ndarray")
    if feature.dtype != np.dtype(np.float32) or feature.shape != (
        layout.mel_bins,
        layout.time_frames,
    ):
        raise Experiment002TrainingPopulationError(
            "preprocessor feature has an invalid dtype or shape"
        )
    if not feature.flags.c_contiguous or not np.all(np.isfinite(feature)):
        raise Experiment002TrainingPopulationError(
            "preprocessor feature must be C-contiguous and finite"
        )
    return feature


def _validate_materialized_arrays(
    model_inputs: FloatArray,
    labels: np.ndarray[tuple[int], np.dtype[np.int64]],
    examples: tuple[_Example, ...],
    layout: _TrainingLayout,
) -> None:
    size = len(examples)
    if (
        type(model_inputs) is not np.ndarray
        or model_inputs.dtype != np.dtype(np.float32)
        or model_inputs.shape != (size, layout.mel_bins, layout.time_frames)
        or not model_inputs.flags.c_contiguous
        or model_inputs.flags.writeable
        or not model_inputs.flags.owndata
        or not np.all(np.isfinite(model_inputs))
    ):
        raise Experiment002TrainingPopulationError(
            "materialized model inputs are not immutable owned float32"
        )
    if (
        type(labels) is not np.ndarray
        or labels.dtype != np.dtype(np.int64)
        or labels.shape != (size,)
        or not labels.flags.c_contiguous
        or labels.flags.writeable
        or not labels.flags.owndata
    ):
        raise Experiment002TrainingPopulationError(
            "materialized labels are not immutable owned int64"
        )
    expected = np.fromiter(
        (example.label_index for example in examples),
        dtype=np.int64,
        count=size,
    )
    if not np.array_equal(labels, expected):
        raise Experiment002TrainingPopulationError(
            "materialized labels differ from the epoch plan"
        )


def _require_expected_global_update(state: _EpochState, value: object) -> None:
    _require_uint32(value, "global_update")
    expected = state.zero_based_epoch * state.layout.batch_count + state.batch_index
    if value != expected:
        raise Experiment002TrainingPopulationError(
            f"global_update differs: expected={expected}, observed={value}"
        )


def _require_phase(state: _EpochState, expected: TrainingState) -> None:
    if state.phase != expected:
        raise Experiment002TrainingPopulationError(
            f"epoch state must be {expected}, observed={state.phase}"
        )


def _require_epoch_call_authority(state: _EpochState) -> None:
    if state.route_marker is _REGISTERED_EPOCH_MARKER and (
        type(state.executor_authority) is not RegisteredExecutorSessionAuthority
        or state.active_executor_authority is not state.executor_authority
        or type(state.executor_session_token) is not object
    ):
        raise Experiment002TrainingPopulationError(
            "registered epoch operations are owned by the exact executor"
        )
    if state.route_marker is _REGISTERED_EPOCH_MARKER:
        assert state.executor_authority is not None
        executor = _executor_session_snapshot(state.executor_authority)
        if (
            executor.session_token is not state.executor_session_token
            or executor.seed != state.seed
        ):
            raise Experiment002TrainingPopulationError(
                "registered epoch executor binding changed"
            )


def _require_transition_matches_batch(
    state: _EpochState,
    batch: _IssuedBatchState,
    transition: _OptimizerTransitionSnapshot,
    *,
    learning_rate: float,
    batch_mean_training_loss: np.float32,
    returned_preclip_l2_norm: np.float32,
) -> None:
    if (
        transition.executor_session_token is not state.executor_session_token
        or transition.epoch_session_token is not state.session_token
        or transition.batch_token is not batch.batch_token
        or transition.seed != state.seed
        or transition.zero_based_epoch != state.zero_based_epoch
        or transition.zero_based_global_update != batch.global_update
        or transition.batch_size != batch.label_indices.shape[0]
        or transition.learning_rate_bytes != struct.pack("<d", learning_rate)
        or transition.batch_mean_training_loss_bytes
        != struct.pack("<f", batch_mean_training_loss)
        or transition.returned_preclip_l2_norm_bytes
        != struct.pack("<f", returned_preclip_l2_norm)
    ):
        raise Experiment002TrainingPopulationError(
            "optimizer transition does not match the pending training batch"
        )


def _require_consumption_matches_receipt(
    state: _EpochState,
    receipt: _IssuedReceiptState,
    transition: RegisteredOptimizerTransition,
    consumption: TraceConsumedTransition,
) -> None:
    transition_snapshot = _transition_snapshot(
        transition, required_phase="TRACE_CONSUMED"
    )
    consumption_snapshot = _trace_consumption_snapshot(consumption, required_used=False)
    if (
        receipt.optimizer_transition is not transition
        or receipt.optimizer_transition_token
        is not transition_snapshot.transition_token
        or consumption_snapshot.executor_session_token
        is not state.executor_session_token
        or consumption_snapshot.epoch_session_token is not state.session_token
        or consumption_snapshot.batch_token is not receipt.batch_token
        or consumption_snapshot.transition_token
        is not receipt.optimizer_transition_token
        or consumption_snapshot.receipt_token is not receipt.receipt_token
        or consumption_snapshot.zero_based_global_update
        != receipt.snapshot.zero_based_global_update
        or consumption_snapshot.trace_record_index
        != receipt.snapshot.zero_based_global_update
    ):
        raise Experiment002TrainingPopulationError(
            "trace consumption does not match the completed update receipt"
        )


def _require_registered_context(seed: int, zero_based_epoch: int) -> None:
    _require_uint32(seed, "seed")
    _require_uint32(zero_based_epoch, "zero_based_epoch")
    if seed not in TRAINING_SEEDS:
        raise Experiment002TrainingPopulationError("seed is not registered")
    if zero_based_epoch >= TRAIN_EPOCH_COUNT:
        raise Experiment002TrainingPopulationError(
            "zero_based_epoch is outside the registered range"
        )


def _require_uint32(value: object, name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if not 0 <= value <= _UINT32_MAX:
        raise Experiment002TrainingPopulationError(f"{name} must fit UINT32LE")


def _require_finite_float32(value: object, name: str) -> np.float32:
    if type(value) is not np.float32:
        raise TypeError(f"{name} must be an exact numpy.float32")
    if not math.isfinite(float(value)):
        raise Experiment002TrainingPopulationError(f"{name} must be finite")
    return value


def _array_sha256(array: np.ndarray[tuple[int, ...], np.dtype[np.generic]]) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _issued_source_state(source: RegisteredTrainingInputSource) -> _IssuedSourceState:
    if type(source) is not RegisteredTrainingInputSource:
        raise TypeError("source must be a RegisteredTrainingInputSource")
    with _ISSUED_SOURCES_LOCK:
        try:
            state = _ISSUED_SOURCES[source]
            guard = _SOURCE_GUARDS[source]
            context_guard = _SOURCE_CONTEXT_GUARDS[source]
        except KeyError as error:
            raise Experiment002TrainingPopulationError(
                "training input source was not issued by this module"
            ) from error
    if (
        type(state) is not _IssuedSourceState
        or type(state.corpus_snapshot) is not tuple
        or type(state.cache_snapshot) is not tuple
        or type(state.normalization_contents) is not bytes
        or type(state.begun_contexts) is not set
        or type(state.lock) is not _RLOCK_TYPE
    ):
        raise Experiment002TrainingPopulationError(
            "training input source issuer state is invalid"
        )
    if (
        type(guard) is not _SourceIssuanceGuard
        or state.corpus is not guard.corpus
        or state.cache is not guard.cache
        or state.normalization is not guard.normalization
        or state.preprocessor is not guard.preprocessor
        or state.corpus_snapshot != guard.corpus_snapshot
        or state.cache_snapshot != guard.cache_snapshot
        or state.normalization_contents != guard.normalization_contents
        or state.marker is not guard.marker
        or type(context_guard) is not frozenset
        or frozenset(state.begun_contexts) != context_guard
    ):
        raise Experiment002TrainingPopulationError(
            "training input source differs from issuance guard"
        )
    return state


def _issued_epoch_state(epoch: RegisteredTrainingEpoch) -> _EpochState:
    if type(epoch) is not RegisteredTrainingEpoch:
        raise TypeError("epoch must be a RegisteredTrainingEpoch")
    with _ISSUED_EPOCHS_LOCK:
        try:
            state = _ISSUED_EPOCHS[epoch]
            guard = _EPOCH_GUARDS[epoch]
            accepted_guard = _EPOCH_ACCEPTED_GUARDS[epoch]
            mirror_hasher = _EPOCH_HASH_MIRRORS[epoch]
            lifecycle = _EPOCH_LIFECYCLES[epoch]
        except KeyError as error:
            raise Experiment002TrainingPopulationError(
                "training epoch was not issued by this module"
            ) from error
    if (
        type(state) is not _EpochState
        or type(state.seed) is not int
        or type(state.zero_based_epoch) is not int
        or type(state.layout) is not _TrainingLayout
        or type(state.examples) is not tuple
        or type(state.example_snapshots) is not tuple
        or type(state.session_token) is not object
        or (
            state.route_marker is _REGISTERED_EPOCH_MARKER
            and type(state.executor_authority) is not RegisteredExecutorSessionAuthority
        )
        or (
            state.route_marker is _REGISTERED_EPOCH_MARKER
            and type(state.executor_session_token) is not object
        )
        or (
            state.route_marker is None
            and (
                state.executor_authority is not None
                or state.executor_session_token is not None
            )
        )
        or type(state.phase) is not str
        or state.phase not in {"OPEN", "BATCH_PENDING", "READY", "COMPLETE", "FAILED"}
        or type(state.cursor) is not int
        or type(state.batch_index) is not int
        or type(state.pending_verifications) is not int
        or type(state.accepted_updates) is not list
        or type(state.accepted_update_authorities) is not list
        or any(
            type(authority) is not _AcceptedUpdateAuthority
            for authority in state.accepted_update_authorities
        )
        or len(state.accepted_updates) != len(state.accepted_update_authorities)
        or type(state.finished) is not bool
        or (
            state.active_executor_authority is not None
            and type(state.active_executor_authority)
            is not RegisteredExecutorSessionAuthority
        )
        or (
            state.active_transition is not None
            and type(state.active_transition) is not RegisteredOptimizerTransition
        )
        or (
            state.active_consumption is not None
            and type(state.active_consumption) is not TraceConsumedTransition
        )
        or state.active_verification_phase not in {None, "PRE", "POST"}
        or type(state.lock) is not _RLOCK_TYPE
    ):
        raise Experiment002TrainingPopulationError(
            "training epoch issuer state is invalid"
        )
    if (
        type(guard) is not _EpochIssuanceGuard
        or state.seed != guard.seed
        or state.zero_based_epoch != guard.zero_based_epoch
        or state.layout != guard.layout
        or state.examples != guard.examples
        or state.example_snapshots != guard.example_snapshots
        or state.preprocessor is not guard.preprocessor
        or state.source is not guard.source
        or state.route_marker is not guard.route_marker
        or state.session_token is not guard.session_token
        or state.executor_authority is not guard.executor_authority
        or state.executor_session_token is not guard.executor_session_token
        or state.hasher is not guard.hasher
        or type(accepted_guard) is not tuple
        or tuple(state.accepted_update_authorities) != accepted_guard
        or not callable(getattr(mirror_hasher, "update", None))
        or not callable(getattr(mirror_hasher, "hexdigest", None))
        or type(lifecycle) is not _EpochLifecycleGuard
        or state.phase != lifecycle.phase
        or state.cursor != lifecycle.cursor
        or state.batch_index != lifecycle.batch_index
        or state.current_batch is not lifecycle.current_batch
        or state.pending_verifications != lifecycle.pending_verifications
        or state.ready_receipt is not lifecycle.ready_receipt
        or state.finished is not lifecycle.finished
    ):
        raise Experiment002TrainingPopulationError(
            "training epoch differs from issuance guard"
        )
    return state


def _issued_batch_state(batch: MaterializedTrainingBatch) -> _IssuedBatchState:
    if type(batch) is not MaterializedTrainingBatch:
        raise TypeError("batch must be a MaterializedTrainingBatch")
    with _ISSUED_BATCHES_LOCK:
        try:
            state = _ISSUED_BATCHES[batch]
            guard = _BATCH_GUARDS[batch]
        except KeyError as error:
            raise Experiment002TrainingPopulationError(
                "training batch is not live or was not issued by this module"
            ) from error
    if (
        type(state) is not _IssuedBatchState
        or type(state.session_token) is not object
        or type(state.batch_token) is not object
        or type(state.batch_index) is not int
        or type(state.global_update) is not int
        or type(state.first_example) is not int
        or type(state.examples) is not tuple
        or type(state.example_snapshots) is not tuple
        or type(state.model_inputs) is not np.ndarray
        or type(state.label_indices) is not np.ndarray
        or not _is_sha256(state.model_inputs_sha256)
        or not _is_sha256(state.label_indices_sha256)
    ):
        raise Experiment002TrainingPopulationError(
            "training batch issuer state is invalid"
        )
    if (
        type(guard) is not _BatchIssuanceGuard
        or state.session_token is not guard.session_token
        or state.batch_token is not guard.batch_token
        or state.batch_index != guard.batch_index
        or state.global_update != guard.global_update
        or state.first_example != guard.first_example
        or state.examples != guard.examples
        or state.example_snapshots != guard.example_snapshots
        or state.model_inputs is not guard.model_inputs
        or state.label_indices is not guard.label_indices
        or state.model_inputs_sha256 != guard.model_inputs_sha256
        or state.label_indices_sha256 != guard.label_indices_sha256
    ):
        raise Experiment002TrainingPopulationError(
            "training batch differs from issuance guard"
        )
    return state


def _issued_receipt_state(receipt: CompletedTrainingUpdate) -> _IssuedReceiptState:
    if type(receipt) is not CompletedTrainingUpdate:
        raise TypeError("receipt must be a CompletedTrainingUpdate")
    with _ISSUED_RECEIPTS_LOCK:
        try:
            state = _ISSUED_RECEIPTS[receipt]
            guard = _RECEIPT_GUARDS[receipt]
        except KeyError as error:
            raise Experiment002TrainingPopulationError(
                "completed update was not issued by this module"
            ) from error
    _validate_receipt_state(state, receipt=receipt, guard=guard)
    return state


def _issued_population_state(
    population: CompletedTrainingPopulation,
) -> _IssuedPopulationState:
    if type(population) is not CompletedTrainingPopulation:
        raise TypeError("population must be a CompletedTrainingPopulation")
    with _ISSUED_POPULATIONS_LOCK:
        try:
            state = _ISSUED_POPULATIONS[population]
            guard = _POPULATION_GUARDS[population]
        except KeyError as error:
            raise Experiment002TrainingPopulationError(
                "completed population was not issued by this module"
            ) from error
    if (
        type(state) is not _IssuedPopulationState
        or not _is_sha256(state.sha256)
        or type(state.seed) is not int
        or type(state.zero_based_epoch) is not int
        or type(state.example_count) is not int
        or type(state.batch_count) is not int
        or type(state.session_token) is not object
        or type(state.update_authority_payloads) is not tuple
        or any(
            type(payload) is not bytes for payload in state.update_authority_payloads
        )
        or type(state.ordered_batch_tokens) is not tuple
        or type(state.ordered_receipt_tokens) is not tuple
        or type(state.ordered_optimizer_transition_tokens) is not tuple
        or type(state.ordered_optimizer_transitions) is not tuple
        or type(state.ordered_trace_consumptions) is not tuple
        or (
            state.route_marker is not None
            and state.route_marker is not _REGISTERED_EPOCH_MARKER
        )
    ):
        raise Experiment002TrainingPopulationError(
            "completed population issuer state is invalid"
        )
    _validate_population_snapshot(state.snapshot)
    snapshot = state.snapshot
    if (
        snapshot.sha256 != state.sha256
        or snapshot.seed != state.seed
        or snapshot.zero_based_epoch != state.zero_based_epoch
        or snapshot.example_count != state.example_count
        or snapshot.batch_count != state.batch_count
        or snapshot.session_token is not state.session_token
        or snapshot.ordered_batch_tokens != state.ordered_batch_tokens
        or snapshot.ordered_receipt_tokens != state.ordered_receipt_tokens
        or snapshot.ordered_optimizer_transition_tokens
        != state.ordered_optimizer_transition_tokens
        or snapshot.ordered_optimizer_transitions != state.ordered_optimizer_transitions
        or snapshot.ordered_trace_consumptions != state.ordered_trace_consumptions
        or snapshot.route_marker is not state.route_marker
        or tuple(_frame_update_snapshot(item) for item in snapshot.updates)
        != state.update_authority_payloads
    ):
        raise Experiment002TrainingPopulationError(
            "completed population differs from issuer authority"
        )
    if (
        type(guard) is not _PopulationIssuanceGuard
        or state.sha256 != guard.sha256
        or state.seed != guard.seed
        or state.zero_based_epoch != guard.zero_based_epoch
        or state.example_count != guard.example_count
        or state.batch_count != guard.batch_count
        or state.session_token is not guard.session_token
        or state.update_authority_payloads != guard.update_authority_payloads
        or state.ordered_batch_tokens != guard.ordered_batch_tokens
        or state.ordered_receipt_tokens != guard.ordered_receipt_tokens
        or state.ordered_optimizer_transition_tokens
        != guard.ordered_optimizer_transition_tokens
        or state.ordered_optimizer_transitions != guard.ordered_optimizer_transitions
        or state.ordered_trace_consumptions != guard.ordered_trace_consumptions
        or state.route_marker is not guard.route_marker
    ):
        raise Experiment002TrainingPopulationError(
            "completed population differs from issuance guard"
        )
    return state


def _validate_receipt_state(
    state: _IssuedReceiptState,
    *,
    receipt: CompletedTrainingUpdate | None = None,
    guard: _ReceiptIssuanceGuard | None = None,
) -> None:
    if (
        type(state) is not _IssuedReceiptState
        or type(state.authority_payload) is not bytes
        or type(state.session_token) is not object
        or type(state.batch_token) is not object
        or type(state.receipt_token) is not object
        or (
            state.optimizer_transition_token is not None
            and type(state.optimizer_transition_token) is not object
        )
        or (
            state.optimizer_transition is not None
            and type(state.optimizer_transition) is not RegisteredOptimizerTransition
        )
        or (
            (state.optimizer_transition_token is None)
            != (state.optimizer_transition is None)
        )
        or type(state.accepted) is not bool
    ):
        raise Experiment002TrainingPopulationError(
            "completed update issuer state is invalid"
        )
    _validate_update_snapshot(state.snapshot)
    if (
        state.snapshot.session_token is not state.session_token
        or state.snapshot.batch_token is not state.batch_token
        or state.snapshot.receipt_token is not state.receipt_token
        or state.snapshot.optimizer_transition_token
        is not state.optimizer_transition_token
        or _frame_update_snapshot(state.snapshot) != state.authority_payload
    ):
        raise Experiment002TrainingPopulationError(
            "completed update differs from issuer authority"
        )
    if guard is not None and (
        type(guard) is not _ReceiptIssuanceGuard
        or state.authority_payload != guard.authority_payload
        or state.session_token is not guard.session_token
        or state.batch_token is not guard.batch_token
        or state.receipt_token is not guard.receipt_token
        or state.optimizer_transition_token is not guard.optimizer_transition_token
        or state.optimizer_transition is not guard.optimizer_transition
    ):
        raise Experiment002TrainingPopulationError(
            "completed update differs from issuance guard"
        )
    if receipt is not None and state.accepted is not (receipt in _ACCEPTED_RECEIPTS):
        raise Experiment002TrainingPopulationError(
            "completed update acceptance differs from append-only authority"
        )
    if state.optimizer_transition is not None:
        transition = _transition_snapshot(state.optimizer_transition)
        if transition.transition_token is not state.optimizer_transition_token:
            raise Experiment002TrainingPopulationError(
                "completed update transition authority differs"
            )


def _frame_update_snapshot(snapshot: _CompletedTrainingUpdateSnapshot) -> bytes:
    _validate_update_snapshot(snapshot)
    return struct.pack(
        "<IIIIIdff",
        snapshot.seed,
        snapshot.zero_based_epoch,
        snapshot.batch_index,
        snapshot.zero_based_global_update,
        snapshot.batch_size,
        snapshot.learning_rate,
        snapshot.batch_mean_training_loss,
        snapshot.returned_preclip_l2_norm,
    )


def _copy_update_snapshot(
    snapshot: _CompletedTrainingUpdateSnapshot,
) -> _CompletedTrainingUpdateSnapshot:
    _validate_update_snapshot(snapshot)
    return _CompletedTrainingUpdateSnapshot(
        session_token=snapshot.session_token,
        batch_token=snapshot.batch_token,
        receipt_token=snapshot.receipt_token,
        optimizer_transition_token=snapshot.optimizer_transition_token,
        seed=snapshot.seed,
        zero_based_epoch=snapshot.zero_based_epoch,
        batch_index=snapshot.batch_index,
        zero_based_global_update=snapshot.zero_based_global_update,
        batch_size=snapshot.batch_size,
        learning_rate=snapshot.learning_rate,
        learning_rate_bytes=snapshot.learning_rate_bytes,
        batch_mean_training_loss=snapshot.batch_mean_training_loss,
        batch_mean_training_loss_bytes=snapshot.batch_mean_training_loss_bytes,
        returned_preclip_l2_norm=snapshot.returned_preclip_l2_norm,
        returned_preclip_l2_norm_bytes=snapshot.returned_preclip_l2_norm_bytes,
    )


def _update_matches_accepted_authority(
    snapshot: _CompletedTrainingUpdateSnapshot,
    authority: _AcceptedUpdateAuthority,
) -> bool:
    basic_match = (
        type(authority) is _AcceptedUpdateAuthority
        and type(authority.payload) is bytes
        and type(authority.session_token) is object
        and type(authority.batch_token) is object
        and type(authority.receipt_token) is object
        and (
            authority.optimizer_transition_token is None
            or type(authority.optimizer_transition_token) is object
        )
        and _frame_update_snapshot(snapshot) == authority.payload
        and snapshot.session_token is authority.session_token
        and snapshot.batch_token is authority.batch_token
        and snapshot.receipt_token is authority.receipt_token
        and snapshot.optimizer_transition_token is authority.optimizer_transition_token
    )
    if not basic_match:
        return False
    if authority.optimizer_transition_token is None:
        return (
            authority.optimizer_transition is None
            and authority.trace_consumption is None
            and authority.trace_consumption_token is None
        )
    if (
        type(authority.optimizer_transition) is not RegisteredOptimizerTransition
        or type(authority.trace_consumption) is not TraceConsumedTransition
        or type(authority.trace_consumption_token) is not object
    ):
        return False
    transition = _transition_snapshot(
        authority.optimizer_transition, required_phase="POPULATION_ACCEPTED"
    )
    consumption = _trace_consumption_snapshot(
        authority.trace_consumption, required_used=True
    )
    return (
        transition.transition_token is authority.optimizer_transition_token
        and consumption.transition_token is authority.optimizer_transition_token
        and consumption.consumption_token is authority.trace_consumption_token
        and consumption.receipt_token is authority.receipt_token
        and consumption.batch_token is authority.batch_token
        and consumption.zero_based_global_update == snapshot.zero_based_global_update
        and consumption.trace_record_index == snapshot.zero_based_global_update
    )


def _validate_update_snapshot(snapshot: _CompletedTrainingUpdateSnapshot) -> None:
    if (
        type(snapshot) is not _CompletedTrainingUpdateSnapshot
        or type(snapshot.session_token) is not object
        or type(snapshot.batch_token) is not object
        or type(snapshot.receipt_token) is not object
        or (
            snapshot.optimizer_transition_token is not None
            and type(snapshot.optimizer_transition_token) is not object
        )
        or type(snapshot.seed) is not int
        or type(snapshot.zero_based_epoch) is not int
        or type(snapshot.batch_index) is not int
        or type(snapshot.zero_based_global_update) is not int
        or type(snapshot.batch_size) is not int
        or snapshot.batch_size < 1
        or type(snapshot.learning_rate) is not float
        or not math.isfinite(snapshot.learning_rate)
        or snapshot.learning_rate < 0.0
        or type(snapshot.learning_rate_bytes) is not bytes
        or snapshot.learning_rate_bytes != struct.pack("<d", snapshot.learning_rate)
        or type(snapshot.batch_mean_training_loss) is not np.float32
        or not math.isfinite(float(snapshot.batch_mean_training_loss))
        or type(snapshot.batch_mean_training_loss_bytes) is not bytes
        or snapshot.batch_mean_training_loss_bytes
        != struct.pack("<f", snapshot.batch_mean_training_loss)
        or type(snapshot.returned_preclip_l2_norm) is not np.float32
        or not math.isfinite(float(snapshot.returned_preclip_l2_norm))
        or type(snapshot.returned_preclip_l2_norm_bytes) is not bytes
        or snapshot.returned_preclip_l2_norm_bytes
        != struct.pack("<f", snapshot.returned_preclip_l2_norm)
    ):
        raise Experiment002TrainingPopulationError(
            "completed update snapshot is invalid"
        )
    _require_uint32(snapshot.seed, "receipt seed")
    _require_uint32(snapshot.zero_based_epoch, "receipt epoch")
    _require_uint32(snapshot.batch_index, "receipt batch_index")
    _require_uint32(snapshot.zero_based_global_update, "receipt global_update")


def _validate_population_snapshot(
    snapshot: _CompletedTrainingPopulationSnapshot,
) -> None:
    if (
        type(snapshot) is not _CompletedTrainingPopulationSnapshot
        or type(snapshot.session_token) is not object
        or type(snapshot.seed) is not int
        or type(snapshot.zero_based_epoch) is not int
        or not _is_sha256(snapshot.sha256)
        or type(snapshot.example_count) is not int
        or snapshot.example_count < 1
        or type(snapshot.batch_count) is not int
        or snapshot.batch_count < 1
        or type(snapshot.updates) is not tuple
        or type(snapshot.ordered_batch_tokens) is not tuple
        or type(snapshot.ordered_receipt_tokens) is not tuple
        or type(snapshot.ordered_optimizer_transition_tokens) is not tuple
        or type(snapshot.ordered_optimizer_transitions) is not tuple
        or type(snapshot.ordered_trace_consumptions) is not tuple
        or (
            snapshot.route_marker is not None
            and snapshot.route_marker is not _REGISTERED_EPOCH_MARKER
        )
        or len(snapshot.updates) != snapshot.batch_count
        or len(snapshot.ordered_batch_tokens) != snapshot.batch_count
        or len(snapshot.ordered_receipt_tokens) != snapshot.batch_count
        or (
            snapshot.route_marker is _REGISTERED_EPOCH_MARKER
            and len(snapshot.ordered_optimizer_transition_tokens)
            != snapshot.batch_count
        )
        or (
            snapshot.route_marker is _REGISTERED_EPOCH_MARKER
            and len(snapshot.ordered_optimizer_transitions) != snapshot.batch_count
        )
        or (
            snapshot.route_marker is _REGISTERED_EPOCH_MARKER
            and len(snapshot.ordered_trace_consumptions) != snapshot.batch_count
        )
        or (
            snapshot.route_marker is None
            and (
                snapshot.ordered_optimizer_transition_tokens
                or snapshot.ordered_optimizer_transitions
                or snapshot.ordered_trace_consumptions
            )
        )
    ):
        raise Experiment002TrainingPopulationError(
            "completed population snapshot is invalid"
        )
    _require_uint32(snapshot.seed, "population seed")
    _require_uint32(snapshot.zero_based_epoch, "population epoch")
    total_examples = 0
    for batch_index, update in enumerate(snapshot.updates):
        _validate_update_snapshot(update)
        if (
            update.session_token is not snapshot.session_token
            or update.seed != snapshot.seed
            or update.zero_based_epoch != snapshot.zero_based_epoch
            or update.batch_index != batch_index
            or update.batch_token is not snapshot.ordered_batch_tokens[batch_index]
            or update.receipt_token is not snapshot.ordered_receipt_tokens[batch_index]
            or (
                snapshot.route_marker is _REGISTERED_EPOCH_MARKER
                and update.optimizer_transition_token
                is not snapshot.ordered_optimizer_transition_tokens[batch_index]
            )
        ):
            raise Experiment002TrainingPopulationError(
                "completed population update ordering differs"
            )
        if snapshot.route_marker is _REGISTERED_EPOCH_MARKER:
            transition = snapshot.ordered_optimizer_transitions[batch_index]
            consumption = snapshot.ordered_trace_consumptions[batch_index]
            transition_snapshot = _transition_snapshot(
                transition, required_phase="POPULATION_ACCEPTED"
            )
            consumption_snapshot = _trace_consumption_snapshot(
                consumption, required_used=True
            )
            if (
                transition_snapshot.transition_token
                is not update.optimizer_transition_token
                or consumption_snapshot.transition_token
                is not update.optimizer_transition_token
                or consumption_snapshot.receipt_token is not update.receipt_token
                or consumption_snapshot.trace_record_index
                != update.zero_based_global_update
            ):
                raise Experiment002TrainingPopulationError(
                    "completed population transition authority differs"
                )
        total_examples += update.batch_size
    if (
        total_examples != snapshot.example_count
        or len({id(token) for token in snapshot.ordered_batch_tokens})
        != snapshot.batch_count
        or len({id(token) for token in snapshot.ordered_receipt_tokens})
        != snapshot.batch_count
        or (
            snapshot.route_marker is _REGISTERED_EPOCH_MARKER
            and len(
                {id(token) for token in snapshot.ordered_optimizer_transition_tokens}
            )
            != snapshot.batch_count
        )
    ):
        raise Experiment002TrainingPopulationError(
            "completed population token or example counts differ"
        )


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
