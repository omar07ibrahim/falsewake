"""Fixed scratch-file transfer protocol for one Experiment 002 child result.

The three files in this protocol are deliberately noncanonical scratch.  A
child writes the canonical history and verified safetensors payload first, and
writes the canonical envelope last.  The parent recognizes a result only when
all three fixed names are the only directory entries and every byte has been
independently revalidated through descriptor-relative, no-follow I/O.

This module uses only the Python standard library.  It does not publish learned
artifacts and it accepts no filename, schema, digest-domain, or size overrides.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import stat
import struct
import threading
import weakref
from contextlib import suppress
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any, Final, Literal, NoReturn, Protocol, SupportsIndex, cast

__all__ = (
    "ChildResultBinding",
    "Experiment002ChildResultError",
    "VerifiedChildResult",
    "load_registered_child_result",
    "verify_verified_child_result",
    "write_registered_child_result",
)

type ChildRole = Literal["training_seed", "selected_seed_rerun"]

_SCHEMA_VERSION: Final = 1
_EXPERIMENT: Final = "002"
_REGISTERED_SEEDS: Final = (20_260_719, 20_260_720, 20_260_721)
_EPOCH_COUNT: Final = 30
_UPDATES_PER_EPOCH: Final = 313
_VALIDATION_EXAMPLE_COUNT: Final = 10_583
_VALIDATION_CLASS_SUPPORT: Final = (
    397,
    406,
    350,
    377,
    352,
    363,
    363,
    373,
    350,
    372,
    6_278,
    602,
)
_CLASS_COUNT: Final = len(_VALIDATION_CLASS_SUPPORT)
_HISTORY_DOMAIN: Final = b"falsewake-exp002-history-v1\0"
_MODEL_TENSOR_DOMAIN: Final = b"falsewake-exp002-model-tensors-v1\0"
_F32LE_TAG: Final = b"F32LE"
_ENVELOPE_DOMAIN: Final = b"falsewake-exp002-child-result-envelope-v1\0"
_HISTORY_FILENAME: Final = "experiment-002-history.json"
_SAFETENSORS_FILENAME: Final = "experiment-002-winner.safetensors"
_ENVELOPE_FILENAME: Final = "experiment-002-child-result.json"
_RESULT_FILENAMES: Final = frozenset(
    {_HISTORY_FILENAME, _SAFETENSORS_FILENAME, _ENVELOPE_FILENAME}
)
_SCRATCH_ROOT: Final = "/home/ubuntu/gitcode/.t"
_MAX_HISTORY_BYTES: Final = 1 << 20
_REGISTERED_SAFETENSORS_BYTES: Final = 99_776
_REGISTERED_SAFETENSORS_HEADER_BYTES: Final = 4_872
_REGISTERED_MODEL_PAYLOAD_BYTES: Final = 94_896
_REGISTERED_MODEL_TENSOR_COUNT: Final = 53
_REGISTERED_MODEL_VALUE_COUNT: Final = 23_724
_MAX_SAFETENSORS_BYTES: Final = _REGISTERED_SAFETENSORS_BYTES
_MAX_ENVELOPE_BYTES: Final = 64 << 10
_READ_CHUNK_BYTES: Final = 1 << 20
_LOWER_HEX: Final = frozenset("0123456789abcdef")
_PATH_TYPE: Final = type(Path())
_STATE_MARKER: Final = object()

_REGISTERED_MODEL_SPECS: Final = tuple(
    sorted(
        (
            *(
                item
                for block in range(8)
                for item in (
                    (f"blocks.{block}.depthwise.weight", (48, 1, 3)),
                    (f"blocks.{block}.depthwise_norm.norm.bias", (48,)),
                    (f"blocks.{block}.depthwise_norm.norm.weight", (48,)),
                    (f"blocks.{block}.pointwise.weight", (48, 48, 1)),
                    (f"blocks.{block}.pointwise_norm.norm.bias", (48,)),
                    (f"blocks.{block}.pointwise_norm.norm.weight", (48,)),
                )
            ),
            ("classifier.bias", (12,)),
            ("classifier.weight", (12, 48)),
            ("stem.weight", (48, 40, 1)),
            ("stem_norm.norm.bias", (48,)),
            ("stem_norm.norm.weight", (48,)),
        ),
        key=lambda item: item[0].encode("utf-8"),
    )
)

_TOP_HISTORY_KEYS: Final = frozenset(
    {
        "complete_update_trace_digest",
        "epochs",
        "experiment",
        "schema_version",
        "seed",
        "validation_input_digest",
    }
)
_EPOCH_KEYS: Final = frozenset(
    {
        "epoch_update_trace_digest",
        "first_global_update",
        "last_global_update_inclusive",
        "macro_f1_exact_denominator",
        "macro_f1_exact_numerator",
        "model_tensor_digest",
        "training_cross_entropy_float64_hex",
        "training_population_digest",
        "validation_confusion_matrix",
        "validation_cross_entropy_float64_hex",
        "validation_input_digest",
        "validation_prediction_digest",
        "zero_based_epoch",
    }
)


class Experiment002ChildResultError(ValueError):
    """A child-result binding, payload, or scratch directory failed closed."""


@dataclass(frozen=True, slots=True)
class ChildResultBinding:
    """Expected ticket and run-registration identity for one exact child."""

    role: ChildRole
    ordinal: int
    seed: int
    registration_head_commit: str
    implementation_commit: str
    registration_sha256: str
    source_bundle_sha256: str

    def __post_init__(self) -> None:
        _require_binding(self)


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
class VerifiedChildResult:
    """Opaque, parser-issued ownership of one fully verified child result."""

    def __init__(self) -> None:
        raise TypeError("verified child results are issued only by the parser")

    @property
    def binding(self) -> ChildResultBinding:
        return _bootstrap_verified_state(self).binding

    @property
    def role(self) -> str:
        return _bootstrap_verified_state(self).binding.role

    @property
    def ordinal(self) -> int:
        return _bootstrap_verified_state(self).binding.ordinal

    @property
    def seed(self) -> int:
        return _bootstrap_verified_state(self).binding.seed

    @property
    def registration_head_commit(self) -> str:
        return _bootstrap_verified_state(self).binding.registration_head_commit

    @property
    def implementation_commit(self) -> str:
        return _bootstrap_verified_state(self).binding.implementation_commit

    @property
    def registration_sha256(self) -> str:
        return _bootstrap_verified_state(self).binding.registration_sha256

    @property
    def source_bundle_sha256(self) -> str:
        return _bootstrap_verified_state(self).binding.source_bundle_sha256

    @property
    def history_sha256(self) -> str:
        return _bootstrap_verified_state(self).history_sha256

    @property
    def history_byte_count(self) -> int:
        return len(_bootstrap_verified_state(self).history_bytes)

    @property
    def safetensors_sha256(self) -> str:
        return _bootstrap_verified_state(self).safetensors_sha256

    @property
    def safetensors_byte_count(self) -> int:
        return len(_bootstrap_verified_state(self).safetensors_bytes)

    @property
    def winner_epoch(self) -> int:
        return _bootstrap_verified_state(self).history_summary.winner_epoch

    @property
    def winner_macro_f1(self) -> Fraction:
        summary = _bootstrap_verified_state(self).history_summary
        return Fraction(
            summary.winner_macro_f1_numerator,
            summary.winner_macro_f1_denominator,
        )

    @property
    def winner_validation_cross_entropy(self) -> float:
        summary = _bootstrap_verified_state(self).history_summary
        return float.fromhex(summary.winner_validation_cross_entropy_float64_hex)

    @property
    def model_tensor_sha256(self) -> str:
        return _bootstrap_verified_state(
            self
        ).history_summary.winner_model_tensor_sha256

    @property
    def envelope_sha256(self) -> str:
        """Return SHA-256 over the fixed envelope domain and canonical bytes."""

        return _bootstrap_verified_state(self).envelope_sha256

    @property
    def canonical_history_bytes(self) -> bytes:
        return _bootstrap_verified_state(self).history_bytes

    @property
    def safetensors_bytes(self) -> bytes:
        return _bootstrap_verified_state(self).safetensors_bytes

    @property
    def canonical_envelope_bytes(self) -> bytes:
        return _bootstrap_verified_state(self).envelope_bytes

    def __copy__(self) -> NoReturn:
        raise TypeError("verified child results cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        del memo
        raise TypeError("verified child results cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("verified child results cannot be serialized")

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        del protocol
        raise TypeError("verified child results cannot be serialized")


@dataclass(frozen=True, slots=True)
class _HistorySummary:
    seed: int
    winner_epoch: int
    winner_model_tensor_sha256: str
    winner_macro_f1_numerator: int
    winner_macro_f1_denominator: int
    winner_validation_cross_entropy_float64_hex: str


@dataclass(frozen=True, slots=True)
class _RankValue:
    zero_based_epoch: int
    macro_f1: Fraction
    validation_cross_entropy: float
    model_tensor_sha256: str


@dataclass(frozen=True, slots=True)
class _EnvelopeSummary:
    binding: ChildResultBinding
    history_byte_count: int
    history_sha256: str
    safetensors_byte_count: int
    safetensors_sha256: str
    winner_epoch: int
    model_tensor_sha256: str


@dataclass(frozen=True, slots=True)
class _VerifiedState:
    marker: object
    binding: ChildResultBinding
    history_bytes: bytes
    safetensors_bytes: bytes
    envelope_bytes: bytes
    history_sha256: str
    safetensors_sha256: str
    envelope_sha256: str
    history_summary: _HistorySummary


@dataclass(frozen=True, slots=True)
class _VerifiedGuard:
    marker: object
    binding: ChildResultBinding
    history_bytes: bytes
    safetensors_bytes: bytes
    envelope_bytes: bytes
    history_sha256: str
    safetensors_sha256: str
    envelope_sha256: str
    history_summary: _HistorySummary


def _bootstrap_verified_state(result: VerifiedChildResult, /) -> _VerifiedState:
    del result
    raise Experiment002ChildResultError(
        "verified result property routes were not closure-bound"
    )


@dataclass(frozen=True, slots=True)
class _OpenedFile:
    name: str
    descriptor: int
    before: os.stat_result


@dataclass(frozen=True, slots=True)
class _OwnedFile:
    name: str
    device: int
    inode: int


type _AuthorityFrame = tuple[object, ...]


class _ResultIssuer(Protocol):
    def __call__(
        self,
        *,
        binding: ChildResultBinding,
        history_bytes: bytes,
        safetensors_bytes: bytes,
        envelope_bytes: bytes,
        history_sha256: str,
        safetensors_sha256: str,
        history_summary: _HistorySummary,
    ) -> VerifiedChildResult: ...


class _ResultAuthorityValidator(Protocol):
    def __call__(
        self,
        result: VerifiedChildResult,
        state: _VerifiedState,
        guard: _VerifiedGuard,
        /,
    ) -> _AuthorityFrame | None: ...


class _ResultAuthorityReader(Protocol):
    def __call__(
        self,
        result: VerifiedChildResult,
        /,
    ) -> tuple[_VerifiedState, _VerifiedGuard]: ...


class _ResultAuthorityFailure(Protocol):
    def __call__(self, result: VerifiedChildResult, /) -> None: ...


class _ResultLoader(Protocol):
    def __call__(
        self,
        scratch_directory: Path,
        expected: ChildResultBinding,
        /,
    ) -> VerifiedChildResult: ...


class _ResultVerifier(Protocol):
    def __call__(self, result: VerifiedChildResult, /) -> None: ...


class _VerifiedStateSupplier(Protocol):
    def __call__(self, result: VerifiedChildResult, /) -> _VerifiedState: ...


_ISSUED: weakref.WeakKeyDictionary[VerifiedChildResult, _VerifiedState] = (
    weakref.WeakKeyDictionary()
)
_GUARDS: weakref.WeakKeyDictionary[VerifiedChildResult, _VerifiedGuard] = (
    weakref.WeakKeyDictionary()
)
_FAILED: weakref.WeakSet[VerifiedChildResult] = weakref.WeakSet()
_ISSUED_LOCK = threading.RLock()


def write_registered_child_result(
    scratch_directory: Path,
    binding: ChildResultBinding,
    *,
    history_json_bytes: bytes,
    history_sha256: str,
    safetensors_bytes: bytes,
    safetensors_sha256: str,
    winner_epoch: int,
    model_tensor_sha256: str,
) -> None:
    """Write two payloads and then the sole recognizing envelope.

    The directory must be a fresh, empty, mode-0700 directory below the fixed
    scratch root.  All arguments are verified before the first filesystem write.
    """

    _require_binding(binding)
    _require_exact_bytes(history_json_bytes, "history_json_bytes")
    _require_exact_bytes(safetensors_bytes, "safetensors_bytes")
    _require_payload_size(
        history_json_bytes,
        maximum=_MAX_HISTORY_BYTES,
        name="history JSON",
    )
    _require_payload_size(
        safetensors_bytes,
        maximum=_MAX_SAFETENSORS_BYTES,
        name="safetensors",
    )
    _require_sha256(history_sha256, "history_sha256")
    _require_sha256(safetensors_sha256, "safetensors_sha256")
    _require_sha256(model_tensor_sha256, "model_tensor_sha256")
    _require_exact_epoch(winner_epoch, "winner_epoch")

    parsed_history = _parse_history(history_json_bytes)
    observed_history_sha256 = _history_sha256(history_json_bytes)
    observed_safetensors_sha256 = hashlib.sha256(safetensors_bytes).hexdigest()
    observed_model_tensor_sha256 = _safetensors_model_tensor_sha256(safetensors_bytes)
    if (
        parsed_history.seed != binding.seed
        or parsed_history.winner_epoch != winner_epoch
        or parsed_history.winner_model_tensor_sha256 != model_tensor_sha256
        or observed_model_tensor_sha256 != model_tensor_sha256
        or observed_history_sha256 != history_sha256
        or observed_safetensors_sha256 != safetensors_sha256
    ):
        raise Experiment002ChildResultError(
            "child result payloads differ from their exact supplied bindings"
        )
    envelope = _canonical_envelope_bytes(
        binding=binding,
        history_byte_count=len(history_json_bytes),
        history_sha256=history_sha256,
        safetensors_byte_count=len(safetensors_bytes),
        safetensors_sha256=safetensors_sha256,
        winner_epoch=winner_epoch,
        model_tensor_sha256=model_tensor_sha256,
    )
    if len(envelope) > _MAX_ENVELOPE_BYTES:
        raise Experiment002ChildResultError("canonical envelope exceeds its bound")

    directory_fd = _open_scratch_directory(scratch_directory)
    owned: list[_OwnedFile] = []
    committed = False
    try:
        _lock_scratch_directory(directory_fd, exclusive=True)
        _require_directory_entries(directory_fd, frozenset())
        _write_new_file(
            directory_fd,
            _HISTORY_FILENAME,
            history_json_bytes,
            owned,
        )
        _write_new_file(
            directory_fd,
            _SAFETENSORS_FILENAME,
            safetensors_bytes,
            owned,
        )
        _retry_fsync(directory_fd)
        # The envelope is the commit marker and is intentionally written last.
        _write_new_file(directory_fd, _ENVELOPE_FILENAME, envelope, owned)
        _require_directory_entries(directory_fd, _RESULT_FILENAMES)
        _retry_fsync(directory_fd)
        committed = True
    except BaseException:
        committed = False
        # Unlink the commit marker first on every failure, including interrupts
        # after its file fsync or during the final directory fsync.
        for source in reversed(owned):
            while True:
                try:
                    _unlink_owned_file(directory_fd, source)
                    break
                except KeyboardInterrupt:
                    continue
        while True:
            try:
                with suppress(OSError):
                    _retry_fsync(directory_fd)
                break
            except KeyboardInterrupt:
                continue
        raise
    finally:
        try:
            os.close(directory_fd)
        except BaseException:
            # Once the exact three-file state is durably committed, an
            # ambiguous close must not turn success into a reported failure
            # that strands a recognizing envelope.
            if not committed:
                raise


def _load_registered_child_result_impl(
    scratch_directory: Path,
    expected: ChildResultBinding,
    issuer: _ResultIssuer,
    /,
) -> VerifiedChildResult:
    """Load one complete exact result through the closure-owned issuer."""

    _require_binding(expected)
    directory_fd = _open_scratch_directory(scratch_directory)
    opened: list[_OpenedFile] = []
    try:
        _lock_scratch_directory(directory_fd, exclusive=False)
        directory_before = os.fstat(directory_fd)
        _require_directory_metadata(directory_before)
        _require_directory_entries(directory_fd, _RESULT_FILENAMES)

        # Payload descriptors and bytes are acquired before the envelope.  The
        # envelope is therefore the last-recognized completion marker as well as
        # the last-written file on the producer route.
        history_file = _open_regular_at(
            directory_fd,
            _HISTORY_FILENAME,
            maximum=_MAX_HISTORY_BYTES,
        )
        opened.append(history_file)
        history_bytes = _read_stable_regular(
            history_file,
            maximum=_MAX_HISTORY_BYTES,
        )
        safetensors_file = _open_regular_at(
            directory_fd,
            _SAFETENSORS_FILENAME,
            maximum=_MAX_SAFETENSORS_BYTES,
        )
        opened.append(safetensors_file)
        safetensors_bytes = _read_stable_regular(
            safetensors_file,
            maximum=_MAX_SAFETENSORS_BYTES,
        )
        envelope_file = _open_regular_at(
            directory_fd,
            _ENVELOPE_FILENAME,
            maximum=_MAX_ENVELOPE_BYTES,
        )
        opened.append(envelope_file)
        _require_distinct_single_link_files(opened)
        envelope_bytes = _read_stable_regular(
            envelope_file,
            maximum=_MAX_ENVELOPE_BYTES,
        )

        envelope = _parse_envelope(envelope_bytes)
        history = _parse_history(history_bytes)
        history_sha256 = _history_sha256(history_bytes)
        safetensors_sha256 = hashlib.sha256(safetensors_bytes).hexdigest()
        model_tensor_sha256 = _safetensors_model_tensor_sha256(safetensors_bytes)
        if (
            envelope.binding != expected
            or history.seed != expected.seed
            or envelope.history_byte_count != len(history_bytes)
            or envelope.history_sha256 != history_sha256
            or envelope.safetensors_byte_count != len(safetensors_bytes)
            or envelope.safetensors_sha256 != safetensors_sha256
            or envelope.winner_epoch != history.winner_epoch
            or envelope.model_tensor_sha256 != history.winner_model_tensor_sha256
            or model_tensor_sha256 != history.winner_model_tensor_sha256
        ):
            raise Experiment002ChildResultError(
                "child result bytes do not match the expected envelope binding"
            )

        for source in opened:
            _require_name_still_matches(directory_fd, source)
        _require_directory_entries(directory_fd, _RESULT_FILENAMES)
        directory_after = os.fstat(directory_fd)
        if _stable_stat_frame(directory_before) != _stable_stat_frame(directory_after):
            raise Experiment002ChildResultError(
                "child result directory changed while it was verified"
            )
    finally:
        close_error: BaseException | None = None
        for source in reversed(opened):
            try:
                os.close(source.descriptor)
            except BaseException as error:
                if close_error is None:
                    close_error = error
        try:
            os.close(directory_fd)
        except BaseException as error:
            if close_error is None:
                close_error = error
        if close_error is not None:
            raise close_error

    # Authority is issued only after every descriptor has closed successfully.
    # A caller can therefore never observe a valid object for a load operation
    # that itself reported an I/O or cleanup failure.
    return issuer(
        binding=expected,
        history_bytes=history_bytes,
        safetensors_bytes=safetensors_bytes,
        envelope_bytes=envelope_bytes,
        history_sha256=history_sha256,
        safetensors_sha256=safetensors_sha256,
        history_summary=history,
    )


def _verify_verified_child_result_impl(
    result: VerifiedChildResult,
    reader: _ResultAuthorityReader,
    validator: _ResultAuthorityValidator,
    failure: _ResultAuthorityFailure,
    /,
) -> _AuthorityFrame:
    """Recompute every canonical byte, digest, rank, and immutable guard."""

    try:
        state, guard = reader(result)
        _require_binding(state.binding)
        history = _parse_history(state.history_bytes)
        envelope = _parse_envelope(state.envelope_bytes)
        model_tensor_sha256 = _safetensors_model_tensor_sha256(state.safetensors_bytes)
        history_sha256 = _history_sha256(state.history_bytes)
        safetensors_sha256 = hashlib.sha256(state.safetensors_bytes).hexdigest()
        envelope_sha256 = hashlib.sha256(
            _ENVELOPE_DOMAIN + state.envelope_bytes
        ).hexdigest()
        if (
            state.marker is not _STATE_MARKER
            or history != state.history_summary
            or envelope.binding != state.binding
            or envelope.history_byte_count != len(state.history_bytes)
            or envelope.history_sha256 != history_sha256
            or envelope.safetensors_byte_count != len(state.safetensors_bytes)
            or envelope.safetensors_sha256 != safetensors_sha256
            or envelope.winner_epoch != history.winner_epoch
            or envelope.model_tensor_sha256 != history.winner_model_tensor_sha256
            or model_tensor_sha256 != history.winner_model_tensor_sha256
            or state.history_sha256 != history_sha256
            or state.safetensors_sha256 != safetensors_sha256
            or state.envelope_sha256 != envelope_sha256
        ):
            raise Experiment002ChildResultError(
                "verified child result changed after parser issuance"
            )
        truth = validator(result, state, guard)
        if truth is None:
            raise Experiment002ChildResultError(
                "verified child result authority is missing or corrupted"
            )
        current_state, current_guard = reader(result)
        if current_state is not state or current_guard is not guard:
            raise Experiment002ChildResultError(
                "verified child result authority changed during verification"
            )
        return truth
    except BaseException:
        failure(result)
        raise


def _canonical_envelope_bytes(
    *,
    binding: ChildResultBinding,
    history_byte_count: int,
    history_sha256: str,
    safetensors_byte_count: int,
    safetensors_sha256: str,
    winner_epoch: int,
    model_tensor_sha256: str,
) -> bytes:
    _require_binding(binding)
    _require_positive_bounded_count(
        history_byte_count,
        maximum=_MAX_HISTORY_BYTES,
        name="history byte count",
    )
    _require_positive_bounded_count(
        safetensors_byte_count,
        maximum=_MAX_SAFETENSORS_BYTES,
        name="safetensors byte count",
    )
    _require_sha256(history_sha256, "history SHA-256")
    _require_sha256(safetensors_sha256, "safetensors SHA-256")
    _require_sha256(model_tensor_sha256, "model tensor SHA-256")
    _require_exact_epoch(winner_epoch, "winner epoch")
    document: dict[str, object] = {
        "experiment": _EXPERIMENT,
        "history": {
            "byte_count": history_byte_count,
            "domain_sha256": history_sha256,
            "filename": _HISTORY_FILENAME,
            "model_tensor_sha256": model_tensor_sha256,
            "winner_epoch": winner_epoch,
        },
        "ordinal": binding.ordinal,
        "registration": {
            "head_commit": binding.registration_head_commit,
            "implementation_commit": binding.implementation_commit,
            "registration_sha256": binding.registration_sha256,
            "source_bundle_sha256": binding.source_bundle_sha256,
        },
        "role": binding.role,
        "safetensors": {
            "byte_count": safetensors_byte_count,
            "filename": _SAFETENSORS_FILENAME,
            "sha256": safetensors_sha256,
        },
        "schema_version": _SCHEMA_VERSION,
        "seed": binding.seed,
    }
    return _canonical_json_bytes(document)


def _parse_envelope(raw: bytes) -> _EnvelopeSummary:
    document = _parse_canonical_object(
        raw,
        maximum=_MAX_ENVELOPE_BYTES,
        name="child result envelope",
    )
    _require_exact_keys(
        document,
        {
            "experiment",
            "history",
            "ordinal",
            "registration",
            "role",
            "safetensors",
            "schema_version",
            "seed",
        },
        "child result envelope",
    )
    _require_exact_scalar(document["schema_version"], 1, "envelope schema_version")
    _require_exact_scalar(document["experiment"], "002", "envelope experiment")
    registration = _require_exact_object(
        document["registration"],
        {
            "head_commit",
            "implementation_commit",
            "registration_sha256",
            "source_bundle_sha256",
        },
        "envelope registration",
    )
    binding = ChildResultBinding(
        role=_require_role(document["role"]),
        ordinal=_require_exact_int(document["ordinal"], "envelope ordinal"),
        seed=_require_exact_int(document["seed"], "envelope seed"),
        registration_head_commit=_require_hex_value(
            registration["head_commit"], 40, "registration head commit"
        ),
        implementation_commit=_require_hex_value(
            registration["implementation_commit"],
            40,
            "implementation commit",
        ),
        registration_sha256=_require_hex_value(
            registration["registration_sha256"],
            64,
            "registration SHA-256",
        ),
        source_bundle_sha256=_require_hex_value(
            registration["source_bundle_sha256"],
            64,
            "source bundle SHA-256",
        ),
    )
    history = _require_exact_object(
        document["history"],
        {
            "byte_count",
            "domain_sha256",
            "filename",
            "model_tensor_sha256",
            "winner_epoch",
        },
        "envelope history",
    )
    safetensors = _require_exact_object(
        document["safetensors"],
        {"byte_count", "filename", "sha256"},
        "envelope safetensors",
    )
    _require_exact_scalar(history["filename"], _HISTORY_FILENAME, "history filename")
    _require_exact_scalar(
        safetensors["filename"],
        _SAFETENSORS_FILENAME,
        "safetensors filename",
    )
    history_count = _require_exact_int(history["byte_count"], "history byte count")
    safetensors_count = _require_exact_int(
        safetensors["byte_count"], "safetensors byte count"
    )
    _require_positive_bounded_count(
        history_count,
        maximum=_MAX_HISTORY_BYTES,
        name="history byte count",
    )
    _require_positive_bounded_count(
        safetensors_count,
        maximum=_MAX_SAFETENSORS_BYTES,
        name="safetensors byte count",
    )
    return _EnvelopeSummary(
        binding=binding,
        history_byte_count=history_count,
        history_sha256=_require_hex_value(
            history["domain_sha256"], 64, "history domain SHA-256"
        ),
        safetensors_byte_count=safetensors_count,
        safetensors_sha256=_require_hex_value(
            safetensors["sha256"], 64, "safetensors SHA-256"
        ),
        winner_epoch=_require_epoch_value(history["winner_epoch"], "winner epoch"),
        model_tensor_sha256=_require_hex_value(
            history["model_tensor_sha256"], 64, "model tensor SHA-256"
        ),
    )


def _parse_history(raw: bytes) -> _HistorySummary:
    document = _parse_canonical_object(
        raw,
        maximum=_MAX_HISTORY_BYTES,
        name="registered history",
    )
    _require_exact_keys(document, set(_TOP_HISTORY_KEYS), "registered history")
    _require_exact_scalar(document["schema_version"], 1, "history schema_version")
    _require_exact_scalar(document["experiment"], "002", "history experiment")
    seed = _require_exact_int(document["seed"], "history seed")
    _require_registered_seed(seed)
    validation_input_digest = _require_hex_value(
        document["validation_input_digest"],
        64,
        "validation input digest",
    )
    _require_hex_value(
        document["complete_update_trace_digest"],
        64,
        "complete update trace digest",
    )
    epochs_value = document["epochs"]
    if type(epochs_value) is not list or len(epochs_value) != _EPOCH_COUNT:
        raise Experiment002ChildResultError(
            "registered history must contain exactly thirty epochs"
        )
    ranked: list[_RankValue] = []
    for expected_epoch, value in enumerate(cast(list[object], epochs_value)):
        epoch = _require_exact_object(value, set(_EPOCH_KEYS), "history epoch")
        zero_based_epoch = _require_exact_int(
            epoch["zero_based_epoch"], "zero_based_epoch"
        )
        if zero_based_epoch != expected_epoch:
            raise Experiment002ChildResultError(
                "registered history epochs are reordered or incomplete"
            )
        first_update = _require_exact_int(
            epoch["first_global_update"], "first_global_update"
        )
        last_update = _require_exact_int(
            epoch["last_global_update_inclusive"],
            "last_global_update_inclusive",
        )
        if (
            first_update != expected_epoch * _UPDATES_PER_EPOCH
            or last_update != (expected_epoch + 1) * _UPDATES_PER_EPOCH - 1
        ):
            raise Experiment002ChildResultError(
                "registered history update interval is invalid"
            )
        for key in (
            "training_population_digest",
            "epoch_update_trace_digest",
            "validation_prediction_digest",
            "model_tensor_digest",
        ):
            _require_hex_value(epoch[key], 64, key)
        epoch_validation_digest = _require_hex_value(
            epoch["validation_input_digest"],
            64,
            "epoch validation input digest",
        )
        if epoch_validation_digest != validation_input_digest:
            raise Experiment002ChildResultError(
                "epoch validation digest differs from the history binding"
            )
        _require_finite_nonnegative_float64_hex(
            epoch["training_cross_entropy_float64_hex"],
            "training cross entropy",
        )
        validation_cross_entropy = _require_finite_nonnegative_float64_hex(
            epoch["validation_cross_entropy_float64_hex"],
            "validation cross entropy",
        )
        confusion = _require_confusion_matrix(epoch["validation_confusion_matrix"])
        numerator = _require_exact_int(
            epoch["macro_f1_exact_numerator"], "macro-F1 numerator"
        )
        denominator = _require_exact_int(
            epoch["macro_f1_exact_denominator"], "macro-F1 denominator"
        )
        if numerator < 0 or denominator <= 0:
            raise Experiment002ChildResultError("macro-F1 fraction is invalid")
        macro_f1 = Fraction(numerator, denominator)
        if (
            macro_f1.numerator != numerator
            or macro_f1.denominator != denominator
            or macro_f1 != _macro_f1(confusion)
        ):
            raise Experiment002ChildResultError(
                "macro-F1 differs from its canonical confusion matrix"
            )
        ranked.append(
            _RankValue(
                zero_based_epoch=zero_based_epoch,
                macro_f1=macro_f1,
                validation_cross_entropy=validation_cross_entropy,
                model_tensor_sha256=cast(str, epoch["model_tensor_digest"]),
            )
        )
    winner = min(
        ranked,
        key=lambda item: (
            -item.macro_f1,
            item.validation_cross_entropy,
            item.zero_based_epoch,
        ),
    )
    return _HistorySummary(
        seed=seed,
        winner_epoch=winner.zero_based_epoch,
        winner_model_tensor_sha256=winner.model_tensor_sha256,
        winner_macro_f1_numerator=winner.macro_f1.numerator,
        winner_macro_f1_denominator=winner.macro_f1.denominator,
        winner_validation_cross_entropy_float64_hex=(
            winner.validation_cross_entropy.hex()
        ),
    )


def _safetensors_model_tensor_sha256(payload: bytes) -> str:
    """Independently parse the exact registered safetensors model payload."""

    _require_exact_bytes(payload, "safetensors payload")
    if len(payload) != _REGISTERED_SAFETENSORS_BYTES:
        raise Experiment002ChildResultError(
            "safetensors payload has the wrong registered byte count"
        )
    header_size = struct.unpack("<Q", payload[:8])[0]
    if header_size != _REGISTERED_SAFETENSORS_HEADER_BYTES:
        raise Experiment002ChildResultError(
            "safetensors header has the wrong registered byte count"
        )
    padded_header = payload[8 : 8 + header_size]
    header_bytes = padded_header.rstrip(b" ")
    padding = padded_header[len(header_bytes) :]
    if not header_bytes or len(padding) > 7 or padding != b" " * len(padding):
        raise Experiment002ChildResultError("safetensors header padding is invalid")
    try:
        decoded = header_bytes.decode("utf-8", errors="strict")
        parsed = json.loads(
            decoded,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_number,
            parse_float=_reject_json_number,
        )
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as error:
        raise Experiment002ChildResultError(
            "safetensors header is not exact JSON"
        ) from error
    if type(parsed) is not dict or "__metadata__" in parsed:
        raise Experiment002ChildResultError("safetensors metadata must be absent")
    header = cast(dict[str, Any], parsed)
    expected_names = tuple(name for name, _shape in _REGISTERED_MODEL_SPECS)
    if len(header) != _REGISTERED_MODEL_TENSOR_COUNT or tuple(header) != expected_names:
        raise Experiment002ChildResultError(
            "safetensors tensor names differ from the registered model"
        )

    expected_header: dict[str, object] = {}
    framed = bytearray(struct.pack("<I", _REGISTERED_MODEL_TENSOR_COUNT))
    data = payload[8 + header_size :]
    if len(data) != _REGISTERED_MODEL_PAYLOAD_BYTES:
        raise Experiment002ChildResultError(
            "safetensors data has the wrong registered byte count"
        )
    expected_offset = 0
    value_count = 0
    for name, shape in _REGISTERED_MODEL_SPECS:
        record_object = header[name]
        if type(record_object) is not dict:
            raise Experiment002ChildResultError(
                "safetensors tensor record is not an exact object"
            )
        record = cast(dict[str, Any], record_object)
        _require_exact_keys(
            record,
            {"data_offsets", "dtype", "shape"},
            "safetensors tensor record",
        )
        byte_count = 4 * math.prod(shape)
        expected_end = expected_offset + byte_count
        if (
            type(record["dtype"]) is not str
            or record["dtype"] != "F32"
            or type(record["shape"]) is not list
            or tuple(cast(list[object], record["shape"])) != shape
            or type(record["data_offsets"]) is not list
            or cast(list[object], record["data_offsets"])
            != [expected_offset, expected_end]
        ):
            raise Experiment002ChildResultError(
                "safetensors tensor dtype, shape, or offsets changed"
            )
        expected_header[name] = {
            "dtype": "F32",
            "shape": list(shape),
            "data_offsets": [expected_offset, expected_end],
        }
        raw_tensor = data[expected_offset:expected_end]
        if len(raw_tensor) != byte_count or any(
            not math.isfinite(value)
            for (value,) in struct.iter_unpack("<f", raw_tensor)
        ):
            raise Experiment002ChildResultError(
                "safetensors tensor payload is truncated or nonfinite"
            )
        name_bytes = name.encode("utf-8")
        framed.extend(struct.pack("<I", len(name_bytes)))
        framed.extend(name_bytes)
        framed.extend(struct.pack("<I", len(_F32LE_TAG)))
        framed.extend(_F32LE_TAG)
        framed.extend(struct.pack("<I", len(shape)))
        for dimension in shape:
            framed.extend(struct.pack("<Q", dimension))
        framed.extend(struct.pack("<Q", len(raw_tensor)))
        framed.extend(raw_tensor)
        value_count += math.prod(shape)
        expected_offset = expected_end
    expected_header_bytes = json.dumps(
        expected_header,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    expected_padding = b" " * (-len(expected_header_bytes) % 8)
    if (
        expected_offset != _REGISTERED_MODEL_PAYLOAD_BYTES
        or value_count != _REGISTERED_MODEL_VALUE_COUNT
        or expected_header_bytes + expected_padding != padded_header
        or len(payload) != 8 + header_size + expected_offset
    ):
        raise Experiment002ChildResultError(
            "safetensors header or contiguous data layout is not canonical"
        )
    return hashlib.sha256(_MODEL_TENSOR_DOMAIN + bytes(framed)).hexdigest()


def _macro_f1(confusion: tuple[tuple[int, ...], ...]) -> Fraction:
    total = Fraction(0, 1)
    for index in range(_CLASS_COUNT):
        true_positive = confusion[index][index]
        false_positive = sum(
            confusion[row][index] for row in range(_CLASS_COUNT) if row != index
        )
        false_negative = sum(
            confusion[index][column]
            for column in range(_CLASS_COUNT)
            if column != index
        )
        denominator = 2 * true_positive + false_positive + false_negative
        if denominator:
            total += Fraction(2 * true_positive, denominator)
    return total / _CLASS_COUNT


def _require_confusion_matrix(value: object) -> tuple[tuple[int, ...], ...]:
    if type(value) is not list or len(cast(list[object], value)) != _CLASS_COUNT:
        raise Experiment002ChildResultError(
            "validation confusion matrix must have twelve rows"
        )
    rows: list[tuple[int, ...]] = []
    for row_index, value_row in enumerate(cast(list[object], value)):
        if type(value_row) is not list or len(cast(list[object], value_row)) != (
            _CLASS_COUNT
        ):
            raise Experiment002ChildResultError(
                "validation confusion rows must have twelve cells"
            )
        row: list[int] = []
        for cell_value in cast(list[object], value_row):
            cell = _require_exact_int(cell_value, "validation confusion cell")
            if not 0 <= cell <= _VALIDATION_EXAMPLE_COUNT:
                raise Experiment002ChildResultError(
                    "validation confusion cell is outside the population"
                )
            row.append(cell)
        if sum(row) != _VALIDATION_CLASS_SUPPORT[row_index]:
            raise Experiment002ChildResultError(
                "validation confusion row differs from registered class support"
            )
        rows.append(tuple(row))
    if sum(sum(row) for row in rows) != _VALIDATION_EXAMPLE_COUNT:
        raise Experiment002ChildResultError(
            "validation confusion matrix has the wrong population"
        )
    return tuple(rows)


def _parse_canonical_object(
    raw: bytes,
    *,
    maximum: int,
    name: str,
) -> dict[str, Any]:
    _require_exact_bytes(raw, name)
    _require_payload_size(raw, maximum=maximum, name=name)
    try:
        text = raw.decode("ascii", errors="strict")
        parsed = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_number,
            parse_float=_reject_json_number,
        )
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as error:
        raise Experiment002ChildResultError(
            f"{name} is not exact canonical JSON"
        ) from error
    if type(parsed) is not dict:
        raise Experiment002ChildResultError(f"{name} must be a JSON object")
    document = cast(dict[str, Any], parsed)
    if _canonical_json_bytes(cast(dict[str, object], document)) != raw:
        raise Experiment002ChildResultError(f"{name} is not canonical ASCII JSON")
    return document


def _canonical_json_bytes(document: dict[str, object]) -> bytes:
    if type(document) is not dict:
        raise TypeError("canonical JSON document must be an exact dictionary")
    try:
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
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise Experiment002ChildResultError(
            "canonical JSON document contains an invalid value"
        ) from error


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_number(value: str) -> NoReturn:
    raise ValueError(f"non-integer JSON number is forbidden: {value}")


def _open_scratch_directory(path: Path) -> int:
    parts = _validated_scratch_path(path)
    _require_secure_filesystem_support()
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptor = os.open("/", flags)
    try:
        for part in parts:
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        _require_directory_metadata(os.fstat(descriptor))
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _validated_scratch_path(path: Path) -> tuple[str, ...]:
    if type(path) is not _PATH_TYPE:
        raise TypeError("scratch_directory must be an exact pathlib.Path")
    raw = os.fspath(path)
    try:
        raw.encode("ascii", errors="strict")
    except UnicodeEncodeError as error:
        raise Experiment002ChildResultError(
            "scratch directory must be an ASCII path"
        ) from error
    if (
        not raw.startswith(f"{_SCRATCH_ROOT}/")
        or raw.endswith("/")
        or "//" in raw
        or "\\" in raw
        or "\x00" in raw
        or any(part in {"", ".", ".."} for part in raw.split("/")[1:])
        or PurePosixPath(raw).as_posix() != raw
    ):
        raise Experiment002ChildResultError(
            "scratch directory is not a canonical child path below the fixed root"
        )
    return tuple(part for part in raw.split("/") if part)


def _require_directory_metadata(metadata: os.stat_result) -> None:
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise Experiment002ChildResultError(
            "child scratch directory must be an owned mode-0700 directory"
        )


def _require_directory_entries(descriptor: int, expected: frozenset[str]) -> None:
    try:
        entries = os.listdir(descriptor)
    except OSError as error:
        raise Experiment002ChildResultError(
            "child scratch directory cannot be enumerated"
        ) from error
    if (
        any(type(name) is not str for name in entries)
        or len(entries) != len(set(entries))
        or frozenset(entries) != expected
    ):
        raise Experiment002ChildResultError(
            "child scratch directory does not contain the exact fixed result names"
        )


def _lock_scratch_directory(descriptor: int, *, exclusive: bool) -> None:
    operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    try:
        fcntl.flock(descriptor, operation | fcntl.LOCK_NB)
    except (BlockingIOError, OSError) as error:
        raise Experiment002ChildResultError(
            "child scratch directory has a concurrent protocol owner"
        ) from error


def _write_new_file(
    directory_fd: int,
    name: str,
    payload: bytes,
    owned: list[_OwnedFile],
) -> None:
    if name not in _RESULT_FILENAMES:
        raise Experiment002ChildResultError("result filename is not registered")
    _require_exact_bytes(payload, f"{name} payload")
    if type(owned) is not list or any(
        type(source) is not _OwnedFile for source in owned
    ):
        raise TypeError("owned files must be an exact _OwnedFile list")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
    except OSError as error:
        raise Experiment002ChildResultError(
            f"cannot create fixed child result file: {name}"
        ) from error
    source: _OwnedFile | None = None
    try:
        os.fchmod(descriptor, 0o600)
        initial = os.fstat(descriptor)
        if (
            not stat.S_ISREG(initial.st_mode)
            or initial.st_uid != os.geteuid()
            or initial.st_nlink != 1
            or stat.S_IMODE(initial.st_mode) != 0o600
        ):
            raise Experiment002ChildResultError(
                f"new child result file has invalid initial metadata: {name}"
            )
        source = _OwnedFile(
            name=name,
            device=initial.st_dev,
            inode=initial.st_ino,
        )
        owned.append(source)
        offset = 0
        view = memoryview(payload)
        while offset < len(payload):
            try:
                written = os.pwrite(descriptor, view[offset:], offset)
            except InterruptedError:
                continue
            if written <= 0:
                raise Experiment002ChildResultError(
                    f"child result write made no progress: {name}"
                )
            offset += written
        _retry_fsync(descriptor)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or (metadata.st_dev, metadata.st_ino) != (source.device, source.inode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_size != len(payload)
        ):
            raise Experiment002ChildResultError(
                f"new child result file has invalid metadata: {name}"
            )
    except BaseException:
        if source is None:
            try:
                metadata = os.fstat(descriptor)
                source = _OwnedFile(name, metadata.st_dev, metadata.st_ino)
            except OSError:
                source = None
        if source is not None:
            while True:
                try:
                    _unlink_owned_file(directory_fd, source)
                    break
                except KeyboardInterrupt:
                    continue
        raise
    finally:
        os.close(descriptor)


def _open_regular_at(
    directory_fd: int,
    name: str,
    *,
    maximum: int,
) -> _OpenedFile:
    if name not in _RESULT_FILENAMES:
        raise Experiment002ChildResultError("result filename is not registered")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as error:
        raise Experiment002ChildResultError(
            f"cannot open fixed child result file: {name}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o600
            or not 0 < before.st_size <= maximum
        ):
            raise Experiment002ChildResultError(
                f"child result file metadata is invalid: {name}"
            )
        return _OpenedFile(name=name, descriptor=descriptor, before=before)
    except BaseException:
        os.close(descriptor)
        raise


def _read_stable_regular(source: _OpenedFile, *, maximum: int) -> bytes:
    if type(source) is not _OpenedFile:
        raise TypeError("source must be an exact opened-file record")
    size = source.before.st_size
    if not 0 < size <= maximum:
        raise Experiment002ChildResultError("child result byte count is invalid")
    payload = bytearray()
    offset = 0
    while offset < size:
        try:
            chunk = os.pread(
                source.descriptor,
                min(_READ_CHUNK_BYTES, size - offset),
                offset,
            )
        except InterruptedError:
            continue
        except OSError as error:
            raise Experiment002ChildResultError(
                f"cannot read fixed child result file: {source.name}"
            ) from error
        if not chunk:
            raise Experiment002ChildResultError(
                f"child result file was truncated while reading: {source.name}"
            )
        payload.extend(chunk)
        offset += len(chunk)
    trailing: bytes
    while True:
        try:
            trailing = os.pread(source.descriptor, 1, size)
            break
        except InterruptedError:
            continue
    if trailing:
        raise Experiment002ChildResultError(
            f"child result file grew while reading: {source.name}"
        )
    after = os.fstat(source.descriptor)
    if _stable_stat_frame(source.before) != _stable_stat_frame(after):
        raise Experiment002ChildResultError(
            f"child result file changed while reading: {source.name}"
        )
    return bytes(payload)


def _retry_fsync(descriptor: int) -> None:
    while True:
        try:
            os.fsync(descriptor)
            return
        except InterruptedError:
            continue


def _unlink_owned_file(directory_fd: int, source: _OwnedFile) -> None:
    if type(source) is not _OwnedFile:
        raise TypeError("cleanup source must be an exact _OwnedFile")
    while True:
        try:
            observed = os.stat(
                source.name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return
        except (InterruptedError, KeyboardInterrupt):
            continue
        if (observed.st_dev, observed.st_ino) != (source.device, source.inode):
            raise Experiment002ChildResultError(
                "child result cleanup refused a replaced file name"
            )
        try:
            os.unlink(source.name, dir_fd=directory_fd)
            return
        except FileNotFoundError:
            return
        except (InterruptedError, KeyboardInterrupt):
            continue


def _require_distinct_single_link_files(files: list[_OpenedFile]) -> None:
    identities: set[tuple[int, int]] = set()
    for source in files:
        metadata = source.before
        identity = (metadata.st_dev, metadata.st_ino)
        if metadata.st_nlink != 1 or identity in identities:
            raise Experiment002ChildResultError(
                "child result files are hard-linked or alias one inode"
            )
        identities.add(identity)


def _require_name_still_matches(directory_fd: int, source: _OpenedFile) -> None:
    try:
        observed = os.stat(source.name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as error:
        raise Experiment002ChildResultError(
            f"child result name changed during verification: {source.name}"
        ) from error
    if _stable_stat_frame(observed) != _stable_stat_frame(source.before):
        raise Experiment002ChildResultError(
            f"child result name no longer identifies its opened inode: {source.name}"
        )


def _stable_stat_frame(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _require_secure_filesystem_support() -> None:
    missing = [
        name
        for name in ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")
        if not hasattr(os, name)
    ]
    if (
        missing
        or os.open not in os.supports_dir_fd
        or os.stat not in os.supports_dir_fd
        or os.unlink not in os.supports_dir_fd
    ):
        detail = ", ".join(missing) if missing else "descriptor-relative calls"
        raise Experiment002ChildResultError(
            f"secure child-result filesystem support is unavailable: {detail}"
        )


def _require_binding(binding: ChildResultBinding) -> None:
    if type(binding) is not ChildResultBinding:
        raise TypeError("binding must be an exact ChildResultBinding")
    role = _require_role(binding.role)
    if type(binding.ordinal) is not int:
        raise TypeError("child ordinal must be an exact integer")
    if type(binding.seed) is not int:
        raise TypeError("child seed must be an exact integer")
    _require_registered_seed(binding.seed)
    if role == "training_seed":
        if not 0 <= binding.ordinal < len(_REGISTERED_SEEDS):
            raise Experiment002ChildResultError(
                "training child ordinal is outside the registered sequence"
            )
        if binding.seed != _REGISTERED_SEEDS[binding.ordinal]:
            raise Experiment002ChildResultError(
                "training child seed differs from its registered ordinal"
            )
    elif binding.ordinal != 3:
        raise Experiment002ChildResultError(
            "selected rerun must have registered ordinal three"
        )
    _require_lower_hex(
        binding.registration_head_commit,
        length=40,
        name="registration head commit",
    )
    _require_lower_hex(
        binding.implementation_commit,
        length=40,
        name="implementation commit",
    )
    if binding.registration_head_commit == binding.implementation_commit:
        raise Experiment002ChildResultError(
            "registration and implementation commits must differ"
        )
    _require_sha256(binding.registration_sha256, "registration_sha256")
    _require_sha256(binding.source_bundle_sha256, "source_bundle_sha256")


def _require_role(value: object) -> ChildRole:
    if type(value) is not str:
        raise TypeError("child role must be an exact string")
    if value not in {"training_seed", "selected_seed_rerun"}:
        raise Experiment002ChildResultError("child role is not registered")
    return cast(ChildRole, value)


def _require_registered_seed(seed: object) -> None:
    if type(seed) is not int:
        raise TypeError("seed must be an exact integer")
    if seed not in _REGISTERED_SEEDS:
        raise Experiment002ChildResultError("seed is not registered")


def _require_exact_epoch(value: object, name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if not 0 <= value < _EPOCH_COUNT:
        raise Experiment002ChildResultError(f"{name} is outside [0, 30)")


def _require_epoch_value(value: object, name: str) -> int:
    result = _require_exact_int(value, name)
    _require_exact_epoch(result, name)
    return result


def _require_finite_nonnegative_float64_hex(value: object, name: str) -> float:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    try:
        parsed = float.fromhex(value)
    except ValueError as error:
        raise Experiment002ChildResultError(f"{name} is not binary64 hex") from error
    if (
        not math.isfinite(parsed)
        or parsed < 0.0
        or math.copysign(1.0, parsed) < 0.0
        or parsed.hex() != value
        or value.lower() != value
    ):
        raise Experiment002ChildResultError(
            f"{name} is not canonical finite nonnegative binary64 hex"
        )
    return parsed


def _history_sha256(payload: bytes) -> str:
    return hashlib.sha256(_HISTORY_DOMAIN + payload).hexdigest()


def _require_exact_bytes(value: object, name: str) -> None:
    if type(value) is not bytes:
        raise TypeError(f"{name} must be exact bytes")


def _require_payload_size(payload: bytes, *, maximum: int, name: str) -> None:
    if not 0 < len(payload) <= maximum:
        raise Experiment002ChildResultError(f"{name} byte count is invalid")


def _require_positive_bounded_count(value: object, *, maximum: int, name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if not 0 < value <= maximum:
        raise Experiment002ChildResultError(f"{name} is outside its fixed bound")


def _require_exact_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    return value


def _require_exact_scalar(value: object, expected: object, name: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise Experiment002ChildResultError(f"{name} is not the fixed value")


def _require_exact_object(
    value: object,
    keys: set[str],
    name: str,
) -> dict[str, Any]:
    if type(value) is not dict:
        raise Experiment002ChildResultError(f"{name} must be an exact object")
    result = cast(dict[str, Any], value)
    _require_exact_keys(result, keys, name)
    return result


def _require_exact_keys(
    value: dict[str, Any],
    expected: set[str],
    name: str,
) -> None:
    if set(value) != expected or any(type(key) is not str for key in value):
        raise Experiment002ChildResultError(f"{name} has invalid keys")


def _require_hex_value(value: object, length: int, name: str) -> str:
    _require_lower_hex(value, length=length, name=name)
    return cast(str, value)


def _require_sha256(value: object, name: str) -> None:
    _require_lower_hex(value, length=64, name=name)


def _require_lower_hex(value: object, *, length: int, name: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    if len(value) != length or any(character not in _LOWER_HEX for character in value):
        raise Experiment002ChildResultError(
            f"{name} must be {length} lowercase hexadecimal characters"
        )


def _make_result_routes() -> tuple[
    _ResultLoader,
    _ResultVerifier,
    _VerifiedStateSupplier,
]:
    """Bind parser issuance and every later read to hidden immutable truth."""

    issued_cache = _ISSUED
    guard_cache = _GUARDS
    failed_cache = _FAILED
    cache_lock = _ISSUED_LOCK
    load_impl = _load_registered_child_result_impl
    verify_impl = _verify_verified_child_result_impl
    exact_type = type
    allocate = object.__new__
    assign = object.__setattr__
    result_type = VerifiedChildResult
    binding_type = ChildResultBinding
    summary_type = _HistorySummary
    state_type = _VerifiedState
    guard_type = _VerifiedGuard
    state_marker = _STATE_MARKER
    authority: weakref.WeakKeyDictionary[
        VerifiedChildResult,
        tuple[
            _VerifiedState,
            _VerifiedGuard,
            _AuthorityFrame,
        ],
    ] = weakref.WeakKeyDictionary()
    authority_failed: weakref.WeakSet[VerifiedChildResult] = weakref.WeakSet()
    authority_lock = threading.RLock()

    def frame(value: _VerifiedState | _VerifiedGuard) -> _AuthorityFrame:
        if exact_type(value) not in {state_type, guard_type}:
            raise TypeError("verified result authority frame has an invalid type")
        binding = value.binding
        summary = value.history_summary
        if (
            exact_type(binding) is not binding_type
            or exact_type(summary) is not summary_type
        ):
            raise Experiment002ChildResultError(
                "verified result authority frame contains aliased metadata"
            )
        typed_values = (
            binding.role,
            binding.ordinal,
            binding.seed,
            binding.registration_head_commit,
            binding.implementation_commit,
            binding.registration_sha256,
            binding.source_bundle_sha256,
            value.history_bytes,
            value.safetensors_bytes,
            value.envelope_bytes,
            value.history_sha256,
            value.safetensors_sha256,
            value.envelope_sha256,
            summary.seed,
            summary.winner_epoch,
            summary.winner_model_tensor_sha256,
            summary.winner_macro_f1_numerator,
            summary.winner_macro_f1_denominator,
            summary.winner_validation_cross_entropy_float64_hex,
        )
        return (
            value.marker,
            *((exact_type(item), item) for item in typed_values),
        )

    def copy_binding(source: ChildResultBinding) -> ChildResultBinding:
        if exact_type(source) is not binding_type:
            raise TypeError("binding must be an exact ChildResultBinding")
        result = allocate(binding_type)
        assign(result, "role", source.role)
        assign(result, "ordinal", source.ordinal)
        assign(result, "seed", source.seed)
        assign(
            result,
            "registration_head_commit",
            source.registration_head_commit,
        )
        assign(
            result,
            "implementation_commit",
            source.implementation_commit,
        )
        assign(
            result,
            "registration_sha256",
            source.registration_sha256,
        )
        assign(
            result,
            "source_bundle_sha256",
            source.source_bundle_sha256,
        )
        return result

    def copy_summary(source: _HistorySummary) -> _HistorySummary:
        if exact_type(source) is not summary_type:
            raise TypeError("history summary must be an exact _HistorySummary")
        result = allocate(summary_type)
        assign(result, "seed", source.seed)
        assign(result, "winner_epoch", source.winner_epoch)
        assign(
            result,
            "winner_model_tensor_sha256",
            source.winner_model_tensor_sha256,
        )
        assign(
            result,
            "winner_macro_f1_numerator",
            source.winner_macro_f1_numerator,
        )
        assign(
            result,
            "winner_macro_f1_denominator",
            source.winner_macro_f1_denominator,
        )
        assign(
            result,
            "winner_validation_cross_entropy_float64_hex",
            source.winner_validation_cross_entropy_float64_hex,
        )
        return result

    def materialize(truth: _AuthorityFrame) -> _VerifiedState:
        if (
            exact_type(truth) is not tuple
            or len(truth) != 20
            or truth[0] is not state_marker
        ):
            raise Experiment002ChildResultError(
                "verified result authority truth is malformed"
            )
        expected_types = (
            str,
            int,
            int,
            str,
            str,
            str,
            str,
            bytes,
            bytes,
            bytes,
            str,
            str,
            str,
            int,
            int,
            str,
            int,
            int,
            str,
        )
        values: list[object] = []
        for pair, expected_type in zip(truth[1:], expected_types, strict=True):
            if exact_type(pair) is not tuple:
                raise Experiment002ChildResultError(
                    "verified result authority truth is malformed"
                )
            typed_pair = cast(tuple[object, ...], pair)
            if (
                len(typed_pair) != 2
                or typed_pair[0] is not expected_type
                or exact_type(typed_pair[1]) is not expected_type
            ):
                raise Experiment002ChildResultError(
                    "verified result authority truth is malformed"
                )
            values.append(typed_pair[1])

        binding = allocate(binding_type)
        assign(binding, "role", values[0])
        assign(binding, "ordinal", values[1])
        assign(binding, "seed", values[2])
        assign(binding, "registration_head_commit", values[3])
        assign(binding, "implementation_commit", values[4])
        assign(binding, "registration_sha256", values[5])
        assign(binding, "source_bundle_sha256", values[6])
        summary = allocate(summary_type)
        assign(summary, "seed", values[13])
        assign(summary, "winner_epoch", values[14])
        assign(summary, "winner_model_tensor_sha256", values[15])
        assign(summary, "winner_macro_f1_numerator", values[16])
        assign(summary, "winner_macro_f1_denominator", values[17])
        assign(
            summary,
            "winner_validation_cross_entropy_float64_hex",
            values[18],
        )
        detached = allocate(state_type)
        assign(detached, "marker", truth[0])
        assign(detached, "binding", binding)
        assign(detached, "history_bytes", values[7])
        assign(detached, "safetensors_bytes", values[8])
        assign(detached, "envelope_bytes", values[9])
        assign(detached, "history_sha256", values[10])
        assign(detached, "safetensors_sha256", values[11])
        assign(detached, "envelope_sha256", values[12])
        assign(detached, "history_summary", summary)
        if frame(detached) != truth:
            raise Experiment002ChildResultError(
                "verified result authority truth failed reconstruction"
            )
        return detached

    def issuer(
        capture: list[VerifiedChildResult],
        /,
        *,
        binding: ChildResultBinding,
        history_bytes: bytes,
        safetensors_bytes: bytes,
        envelope_bytes: bytes,
        history_sha256: str,
        safetensors_sha256: str,
        history_summary: _HistorySummary,
    ) -> VerifiedChildResult:
        _require_binding(binding)
        binding_snapshot = copy_binding(binding)
        summary_snapshot = copy_summary(history_summary)
        state = allocate(state_type)
        assign(state, "marker", state_marker)
        assign(state, "binding", binding_snapshot)
        assign(state, "history_bytes", history_bytes)
        assign(state, "safetensors_bytes", safetensors_bytes)
        assign(state, "envelope_bytes", envelope_bytes)
        assign(state, "history_sha256", history_sha256)
        assign(state, "safetensors_sha256", safetensors_sha256)
        assign(
            state,
            "envelope_sha256",
            hashlib.sha256(_ENVELOPE_DOMAIN + envelope_bytes).hexdigest(),
        )
        assign(state, "history_summary", summary_snapshot)
        guard = allocate(guard_type)
        assign(guard, "marker", state.marker)
        assign(guard, "binding", copy_binding(binding_snapshot))
        assign(guard, "history_bytes", state.history_bytes)
        assign(guard, "safetensors_bytes", state.safetensors_bytes)
        assign(guard, "envelope_bytes", state.envelope_bytes)
        assign(guard, "history_sha256", state.history_sha256)
        assign(guard, "safetensors_sha256", state.safetensors_sha256)
        assign(guard, "envelope_sha256", state.envelope_sha256)
        assign(guard, "history_summary", copy_summary(summary_snapshot))
        truth = frame(state)
        if frame(guard) != truth:
            raise Experiment002ChildResultError(
                "verified result independent authority frames disagree"
            )
        result = allocate(result_type)
        capture.append(result)
        with authority_lock:
            try:
                if result in authority or result in authority_failed:
                    raise Experiment002ChildResultError(
                        "verified result authority issuance was replayed"
                    )
                authority[result] = (state, guard, truth)
                with cache_lock:
                    issued_cache[result] = state
                    guard_cache[result] = guard
            except BaseException:
                authority_failed.add(result)
                with cache_lock:
                    failed_cache.add(result)
                    issued_cache.pop(result, None)
                    guard_cache.pop(result, None)
                raise
        return result

    def reader(
        result: VerifiedChildResult,
        /,
    ) -> tuple[_VerifiedState, _VerifiedGuard]:
        if exact_type(result) is not result_type:
            raise TypeError("result must be an exact VerifiedChildResult")
        with cache_lock:
            state = issued_cache.get(result)
            guard = guard_cache.get(result)
            failed = result in failed_cache
        if (
            failed
            or exact_type(state) is not state_type
            or exact_type(guard) is not guard_type
        ):
            raise Experiment002ChildResultError(
                "verified child result was not issued or is terminally failed"
            )
        return cast(_VerifiedState, state), cast(_VerifiedGuard, guard)

    def validator(
        result: VerifiedChildResult,
        state: _VerifiedState,
        guard: _VerifiedGuard,
        /,
    ) -> _AuthorityFrame | None:
        try:
            state_frame = frame(state)
            guard_frame = frame(guard)
        except BaseException:
            return None
        with authority_lock:
            record = authority.get(result)
            valid = (
                result not in authority_failed
                and record is not None
                and record[0] is state
                and record[1] is guard
                and state_frame == record[2]
                and guard_frame == record[2]
            )
            return record[2] if valid and record is not None else None

    def failure(result: VerifiedChildResult, /) -> None:
        if exact_type(result) is not result_type:
            return
        with authority_lock:
            authority_failed.add(result)
            with cache_lock:
                failed_cache.add(result)

    def verified_state(
        result: VerifiedChildResult,
        /,
    ) -> _VerifiedState:
        try:
            return materialize(verify_impl(result, reader, validator, failure))
        except BaseException:
            failure(result)
            raise

    def load_route(
        scratch_directory: Path,
        expected: ChildResultBinding,
        /,
    ) -> VerifiedChildResult:
        captured: list[VerifiedChildResult] = []

        def transaction_issuer(
            *,
            binding: ChildResultBinding,
            history_bytes: bytes,
            safetensors_bytes: bytes,
            envelope_bytes: bytes,
            history_sha256: str,
            safetensors_sha256: str,
            history_summary: _HistorySummary,
        ) -> VerifiedChildResult:
            return issuer(
                captured,
                binding=binding,
                history_bytes=history_bytes,
                safetensors_bytes=safetensors_bytes,
                envelope_bytes=envelope_bytes,
                history_sha256=history_sha256,
                safetensors_sha256=safetensors_sha256,
                history_summary=history_summary,
            )

        try:
            result = load_impl(scratch_directory, expected, transaction_issuer)
            verified_state(result)
            return result
        except BaseException:
            for issued in captured:
                failure(issued)
            raise

    def verify_route(result: VerifiedChildResult, /) -> None:
        verified_state(result)

    return load_route, verify_route, verified_state


def _install_lexical_result_properties(
    supplier: _VerifiedStateSupplier,
) -> None:
    """Replace bootstrap fget bodies with routes that cannot be map-swapped."""

    exact_type = type
    fraction_type = Fraction
    float_type = float
    finite = math.isfinite
    sign = math.copysign
    summary_type = _HistorySummary
    error_type = Experiment002ChildResultError

    def binding(result: VerifiedChildResult) -> ChildResultBinding:
        return supplier(result).binding

    def role(result: VerifiedChildResult) -> str:
        return supplier(result).binding.role

    def ordinal(result: VerifiedChildResult) -> int:
        return supplier(result).binding.ordinal

    def seed(result: VerifiedChildResult) -> int:
        return supplier(result).binding.seed

    def registration_head_commit(result: VerifiedChildResult) -> str:
        return supplier(result).binding.registration_head_commit

    def implementation_commit(result: VerifiedChildResult) -> str:
        return supplier(result).binding.implementation_commit

    def registration_sha256(result: VerifiedChildResult) -> str:
        return supplier(result).binding.registration_sha256

    def source_bundle_sha256(result: VerifiedChildResult) -> str:
        return supplier(result).binding.source_bundle_sha256

    def history_sha256(result: VerifiedChildResult) -> str:
        return supplier(result).history_sha256

    def history_byte_count(result: VerifiedChildResult) -> int:
        return len(supplier(result).history_bytes)

    def safetensors_sha256(result: VerifiedChildResult) -> str:
        return supplier(result).safetensors_sha256

    def safetensors_byte_count(result: VerifiedChildResult) -> int:
        return len(supplier(result).safetensors_bytes)

    def winner_epoch(result: VerifiedChildResult) -> int:
        return supplier(result).history_summary.winner_epoch

    def winner_macro_f1(result: VerifiedChildResult) -> Fraction:
        summary = supplier(result).history_summary
        if exact_type(summary) is not summary_type:
            raise error_type("winner macro-F1 authority summary type changed")
        numerator = summary.winner_macro_f1_numerator
        denominator = summary.winner_macro_f1_denominator
        if (
            exact_type(numerator) is not int
            or exact_type(denominator) is not int
            or numerator < 0
            or denominator <= 0
            or numerator > denominator
        ):
            raise error_type("winner macro-F1 authority scalars changed")
        value = fraction_type(numerator, denominator)
        if (
            exact_type(value) is not fraction_type
            or value.numerator != numerator
            or value.denominator != denominator
        ):
            raise error_type("winner macro-F1 authority fraction is not reduced")
        return value

    def winner_validation_cross_entropy(result: VerifiedChildResult) -> float:
        summary = supplier(result).history_summary
        if exact_type(summary) is not summary_type:
            raise error_type(
                "winner validation cross-entropy authority summary type changed"
            )
        encoded = summary.winner_validation_cross_entropy_float64_hex
        if exact_type(encoded) is not str:
            raise error_type("winner validation cross-entropy authority scalar changed")
        try:
            value = float_type.fromhex(encoded)
        except (OverflowError, ValueError) as error:
            raise error_type(
                "winner validation cross-entropy authority scalar is invalid"
            ) from error
        if (
            exact_type(value) is not float_type
            or not finite(value)
            or value < 0.0
            or sign(1.0, value) < 0.0
            or value.hex() != encoded
            or encoded.lower() != encoded
        ):
            raise error_type(
                "winner validation cross-entropy authority scalar is not canonical"
            )
        return value

    def model_tensor_sha256(result: VerifiedChildResult) -> str:
        return supplier(result).history_summary.winner_model_tensor_sha256

    def envelope_sha256(result: VerifiedChildResult) -> str:
        return supplier(result).envelope_sha256

    def canonical_history_bytes(result: VerifiedChildResult) -> bytes:
        return supplier(result).history_bytes

    def safetensors_bytes(result: VerifiedChildResult) -> bytes:
        return supplier(result).safetensors_bytes

    def canonical_envelope_bytes(result: VerifiedChildResult) -> bytes:
        return supplier(result).envelope_bytes

    routes = {
        "binding": binding,
        "role": role,
        "ordinal": ordinal,
        "seed": seed,
        "registration_head_commit": registration_head_commit,
        "implementation_commit": implementation_commit,
        "registration_sha256": registration_sha256,
        "source_bundle_sha256": source_bundle_sha256,
        "history_sha256": history_sha256,
        "history_byte_count": history_byte_count,
        "safetensors_sha256": safetensors_sha256,
        "safetensors_byte_count": safetensors_byte_count,
        "winner_epoch": winner_epoch,
        "winner_macro_f1": winner_macro_f1,
        "winner_validation_cross_entropy": winner_validation_cross_entropy,
        "model_tensor_sha256": model_tensor_sha256,
        "envelope_sha256": envelope_sha256,
        "canonical_history_bytes": canonical_history_bytes,
        "safetensors_bytes": safetensors_bytes,
        "canonical_envelope_bytes": canonical_envelope_bytes,
    }
    for name, route in routes.items():
        setattr(VerifiedChildResult, name, property(route))


(
    load_registered_child_result,
    verify_verified_child_result,
    _LEXICAL_VERIFIED_STATE,
) = _make_result_routes()
_install_lexical_result_properties(_LEXICAL_VERIFIED_STATE)
del _LEXICAL_VERIFIED_STATE
del _install_lexical_result_properties
del _make_result_routes
