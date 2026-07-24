"""One-shot production worker for one registered Experiment 006 seed.

The coordinator imports this module only after it has authenticated and consumed
one child activation.  The entrypoint below deliberately exposes no paths,
epoch counts, callbacks, or retry controls: every input comes from the sealed
run authority and every numerical operation is delegated to an already-audited
registered capability.
"""

from __future__ import annotations

import hashlib
import os
import stat
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, Protocol, cast

from falsewake.experiment_002_child_result import (
    ChildResultBinding,
    write_registered_child_result,
)
from falsewake.experiment_002_data import load_registered_corpus
from falsewake.experiment_002_evidence import bind_registered_validation_inputs
from falsewake.experiment_002_normalization_artifact import (
    NormalizationArtifactIdentity,
    load_normalization_artifact,
)
from falsewake.experiment_002_pcm_cache import (
    Experiment002PCMCache,
    PCMCacheIdentity,
)
from falsewake.experiment_002_registered_checkpoint import (
    RegisteredSerializedCheckpoint,
    serialize_registered_checkpoint,
    verify_registered_serialized_checkpoint,
)
from falsewake.experiment_002_registered_evaluator import (
    RegisteredEvaluatedEpoch,
    evaluate_registered_epoch,
)
from falsewake.experiment_002_registered_executor import (
    advance_registered_training_executor,
    complete_registered_training_executor,
    create_registered_training_executor,
    execute_registered_training_epoch,
)
from falsewake.experiment_002_registered_history import (
    RegisteredCompletedTrainingHistory,
    consume_registered_evaluated_epoch,
    create_registered_training_history,
    verify_registered_completed_training_history,
)
from falsewake.experiment_002_training_population import (
    bind_registered_training_inputs,
)
from falsewake.experiment_002_validation import materialize_registered_validation
from falsewake.experiment_006_coordinator import (
    REGISTERED_SEEDS,
    VerifiedChildActivation,
    verify_verified_child_activation,
)
from falsewake.experiment_006_run_authority import (
    VerifiedRunRegistration,
    _registered_child_input_snapshot,
    _RegisteredChildInputSnapshot,
)

__all__ = (
    "Experiment006SeedWorkerError",
    "run_registered_seed_process",
)

type _ChildRole = Literal["training_seed", "selected_seed_rerun"]

_TRAINING_ROLE: Final = "training_seed"
_RERUN_ROLE: Final = "selected_seed_rerun"
_EPOCH_COUNT: Final = 30
_SCRATCH_ROOT: Final = Path("/home/ubuntu/gitcode/.t")
_PATH_TYPE: Final = type(Path())
_LOWER_HEX: Final = frozenset("0123456789abcdef")
_HISTORY_DOMAIN: Final = b"falsewake-exp002-history-v1\0"


class Experiment006SeedWorkerError(RuntimeError):
    """The registered seed process violated its fixed one-shot contract."""


class _HashResult(Protocol):
    def hexdigest(self) -> str: ...


@dataclass(frozen=True, slots=True)
class _ActivationBinding:
    role: str
    seed: int
    ordinal: int
    child_pid: int


@dataclass(frozen=True, slots=True)
class _TerminalPayload:
    history_json_bytes: bytes
    history_sha256: str
    safetensors_bytes: bytes
    safetensors_sha256: str
    winner_epoch: int
    model_tensor_sha256: str


def _require_registered_assignment(
    binding: _ActivationBinding,
    *,
    _binding_type: type[_ActivationBinding] = _ActivationBinding,
    _error_type: type[Experiment006SeedWorkerError] = Experiment006SeedWorkerError,
    _getpid: Callable[[], int] = os.getpid,
    _registered_seeds: tuple[int, int, int] = REGISTERED_SEEDS,
    _training_role: str = _TRAINING_ROLE,
    _rerun_role: str = _RERUN_ROLE,
    _exact_type: Callable[[object], type[object]] = type,
    _int_type: type[int] = int,
    _str_type: type[str] = str,
    _length: Callable[[Any], int] = len,
) -> None:
    if _exact_type(binding) is not _binding_type:
        raise _error_type("child activation binding has an invalid exact type")
    if (
        _exact_type(binding.role) is not _str_type
        or _exact_type(binding.seed) is not _int_type
        or _exact_type(binding.ordinal) is not _int_type
        or _exact_type(binding.child_pid) is not _int_type
    ):
        raise _error_type("child activation fields have invalid exact types")
    process_id = _getpid()
    if _exact_type(process_id) is not _int_type or process_id < 1:
        raise _error_type("worker process PID is invalid")
    if binding.child_pid != process_id:
        raise _error_type("child activation belongs to a different process")
    if binding.role == _training_role:
        if (
            binding.ordinal < 0
            or binding.ordinal >= _length(_registered_seeds)
            or binding.seed != _registered_seeds[binding.ordinal]
        ):
            raise _error_type(
                "training activation is not one of the three registered assignments"
            )
        return
    if (
        binding.role == _rerun_role
        and binding.ordinal == _length(_registered_seeds)
        and binding.seed in _registered_seeds
    ):
        return
    raise _error_type("child role, seed, and ordinal are not a registered assignment")


def _capture_activation_binding(
    activation: VerifiedChildActivation,
    *,
    _binding_type: type[_ActivationBinding] = _ActivationBinding,
    _require_assignment: Callable[[_ActivationBinding], None] = (
        _require_registered_assignment
    ),
) -> _ActivationBinding:
    binding = _binding_type(
        role=activation.role,
        seed=activation.seed,
        ordinal=activation.ordinal,
        child_pid=activation.child_pid,
    )
    _require_assignment(binding)
    return binding


def _bindings_match(
    first: _ActivationBinding,
    second: _ActivationBinding,
    *,
    _binding_type: type[_ActivationBinding] = _ActivationBinding,
    _exact_type: Callable[[object], type[object]] = type,
    _int_type: type[int] = int,
    _str_type: type[str] = str,
) -> bool:
    return (
        _exact_type(first) is _binding_type
        and _exact_type(second) is _binding_type
        and _exact_type(first.role) is _str_type
        and _exact_type(second.role) is _str_type
        and first.role == second.role
        and _exact_type(first.seed) is _int_type
        and _exact_type(second.seed) is _int_type
        and first.seed == second.seed
        and _exact_type(first.ordinal) is _int_type
        and _exact_type(second.ordinal) is _int_type
        and first.ordinal == second.ordinal
        and _exact_type(first.child_pid) is _int_type
        and _exact_type(second.child_pid) is _int_type
        and first.child_pid == second.child_pid
    )


def _verify_activation_boundary(
    registration: VerifiedRunRegistration,
    activation: VerifiedChildActivation,
    expected: _ActivationBinding | None,
    *,
    _verify_activation: Callable[
        [VerifiedRunRegistration, VerifiedChildActivation], object
    ] = verify_verified_child_activation,
    _capture_binding: Callable[[VerifiedChildActivation], _ActivationBinding] = (
        _capture_activation_binding
    ),
    _match_bindings: Callable[[_ActivationBinding, _ActivationBinding], bool] = (
        _bindings_match
    ),
    _error_type: type[Experiment006SeedWorkerError] = Experiment006SeedWorkerError,
) -> _ActivationBinding:
    result = _verify_activation(registration, activation)
    if result is not None:
        raise _error_type("child activation verifier returned an unexpected value")
    observed = _capture_binding(activation)
    if expected is not None and not _match_bindings(observed, expected):
        raise _error_type(
            "child activation assignment changed across a worker boundary"
        )
    return observed


def _initial_activation_binding(
    registration: VerifiedRunRegistration,
    activation: VerifiedChildActivation,
    *,
    _verify_boundary: Callable[
        [VerifiedRunRegistration, VerifiedChildActivation, _ActivationBinding | None],
        _ActivationBinding,
    ] = _verify_activation_boundary,
    _match_bindings: Callable[[_ActivationBinding, _ActivationBinding], bool] = (
        _bindings_match
    ),
    _error_type: type[Experiment006SeedWorkerError] = Experiment006SeedWorkerError,
) -> _ActivationBinding:
    first = _verify_boundary(registration, activation, None)
    second = _verify_boundary(registration, activation, first)
    if not _match_bindings(first, second):
        raise _error_type("child activation changed during initial worker admission")
    return first


def _require_child_input_snapshot(
    snapshot: _RegisteredChildInputSnapshot,
    *,
    _snapshot_type: type[_RegisteredChildInputSnapshot] = (
        _RegisteredChildInputSnapshot
    ),
    _path_type: type[Path] = _PATH_TYPE,
    _lower_hex: frozenset[str] = _LOWER_HEX,
    _error_type: type[Experiment006SeedWorkerError] = Experiment006SeedWorkerError,
    _exact_type: Callable[[object], type[object]] = type,
    _int_type: type[int] = int,
    _str_type: type[str] = str,
    _any: Callable[[Any], bool] = any,
    _length: Callable[[Any], int] = len,
    _set: Callable[[Any], set[object]] = set,
    _type_error: type[TypeError] = TypeError,
) -> None:
    if _exact_type(snapshot) is not _snapshot_type:
        raise _error_type("registered child input snapshot has an invalid exact type")
    path_fields = (
        snapshot.manifest_path,
        snapshot.pcm_cache_path,
        snapshot.normalization_path,
        snapshot.scratch_directory,
    )
    if _any(
        _exact_type(path) is not _path_type or not path.is_absolute()
        for path in path_fields
    ):
        raise _error_type("registered child input paths have invalid exact types")
    if _length(_set(path_fields)) != _length(path_fields):
        raise _error_type("registered child input paths are not distinct")
    for count_name, count_value in (
        ("PCM cache byte count", snapshot.pcm_cache_byte_count),
        ("normalization byte count", snapshot.normalization_byte_count),
    ):
        if _exact_type(count_value) is not _int_type:
            raise _type_error(f"{count_name} must be an exact integer")
        if count_value < 1:
            raise _error_type(f"{count_name} must be positive")
    for digest_name, digest_value in (
        ("PCM cache sha256", snapshot.pcm_cache_sha256),
        ("normalization sha256", snapshot.normalization_sha256),
    ):
        if (
            _exact_type(digest_value) is not _str_type
            or _length(digest_value) != 64
            or _any(character not in _lower_hex for character in digest_value)
        ):
            raise _error_type(
                f"{digest_name} must be 64 lowercase hexadecimal characters"
            )


def _child_input_snapshots_match(
    first: _RegisteredChildInputSnapshot,
    second: _RegisteredChildInputSnapshot,
    *,
    _snapshot_type: type[_RegisteredChildInputSnapshot] = (
        _RegisteredChildInputSnapshot
    ),
    _path_type: type[Path] = _PATH_TYPE,
    _require_snapshot: Callable[[_RegisteredChildInputSnapshot], None] = (
        _require_child_input_snapshot
    ),
    _error_type: type[Experiment006SeedWorkerError] = Experiment006SeedWorkerError,
    _exact_type: Callable[[object], type[object]] = type,
    _int_type: type[int] = int,
    _str_type: type[str] = str,
    _type_error: type[TypeError] = TypeError,
) -> bool:
    if (
        first is second
        or _exact_type(first) is not _snapshot_type
        or _exact_type(second) is not _snapshot_type
    ):
        return False
    try:
        _require_snapshot(first)
        _require_snapshot(second)
    except (_type_error, _error_type):
        return False
    return (
        _exact_type(first.manifest_path) is _path_type
        and _exact_type(second.manifest_path) is _path_type
        and first.manifest_path == second.manifest_path
        and _exact_type(first.pcm_cache_path) is _path_type
        and _exact_type(second.pcm_cache_path) is _path_type
        and first.pcm_cache_path == second.pcm_cache_path
        and _exact_type(first.pcm_cache_byte_count) is _int_type
        and _exact_type(second.pcm_cache_byte_count) is _int_type
        and first.pcm_cache_byte_count == second.pcm_cache_byte_count
        and _exact_type(first.pcm_cache_sha256) is _str_type
        and _exact_type(second.pcm_cache_sha256) is _str_type
        and first.pcm_cache_sha256 == second.pcm_cache_sha256
        and _exact_type(first.normalization_path) is _path_type
        and _exact_type(second.normalization_path) is _path_type
        and first.normalization_path == second.normalization_path
        and _exact_type(first.normalization_byte_count) is _int_type
        and _exact_type(second.normalization_byte_count) is _int_type
        and first.normalization_byte_count == second.normalization_byte_count
        and _exact_type(first.normalization_sha256) is _str_type
        and _exact_type(second.normalization_sha256) is _str_type
        and first.normalization_sha256 == second.normalization_sha256
        and _exact_type(first.scratch_directory) is _path_type
        and _exact_type(second.scratch_directory) is _path_type
        and first.scratch_directory == second.scratch_directory
    )


def _preflight_scratch_directory(
    path: Path,
    *,
    _require_stat: Callable[[os.stat_result], None] | None = None,
    _stat_frame: Callable[[os.stat_result], tuple[int, ...]] | None = None,
    _path_type: type[Path] = _PATH_TYPE,
    _scratch_root: Path = _SCRATCH_ROOT,
    _error_type: type[Experiment006SeedWorkerError] = Experiment006SeedWorkerError,
    _lstat: Callable[[Path], os.stat_result] = os.lstat,
    _open: Callable[[Path, int], int] = os.open,
    _fstat: Callable[[int], os.stat_result] = os.fstat,
    _listdir: Callable[[int], list[str]] = os.listdir,
    _close: Callable[[int], None] = os.close,
    _open_flags: int = (os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC),
    _exact_type: Callable[[object], type[object]] = type,
    _list_type: type[list[object]] = list,
    _str_type: type[str] = str,
    _any: Callable[[Any], bool] = any,
    _type_error: type[TypeError] = TypeError,
    _os_error: type[OSError] = OSError,
) -> None:
    if _require_stat is None:
        _require_stat = _require_owned_empty_directory_stat
    if _stat_frame is None:
        _stat_frame = _directory_stat_frame
    if _exact_type(path) is not _path_type:
        raise _type_error("scratch_directory must be an exact Path")
    if (
        not path.is_absolute()
        or path == _scratch_root
        or _scratch_root not in path.parents
    ):
        raise _error_type("scratch directory is outside the fixed scratch root")
    try:
        if path.resolve(strict=True) != path:
            raise _error_type(
                "scratch directory path is not canonical and symlink-free"
            )
        before_path = _lstat(path)
        descriptor = _open(path, _open_flags)
    except _error_type:
        raise
    except _os_error as error:
        raise _error_type(
            f"cannot open the registered scratch directory: {error}"
        ) from error
    try:
        before_descriptor = _fstat(descriptor)
        _require_stat(before_path)
        _require_stat(before_descriptor)
        if _stat_frame(before_path) != _stat_frame(before_descriptor):
            raise _error_type("scratch path changed while its descriptor was opened")
        entries = _listdir(descriptor)
        if _exact_type(entries) is not _list_type or _any(
            _exact_type(entry) is not _str_type for entry in entries
        ):
            raise _error_type("scratch directory listing has an invalid exact type")
        if entries:
            raise _error_type("registered scratch directory is not empty")
        final_descriptor = _fstat(descriptor)
        final_path = _lstat(path)
        _require_stat(final_descriptor)
        _require_stat(final_path)
        frame = _stat_frame(before_descriptor)
        if _stat_frame(final_descriptor) != frame or _stat_frame(final_path) != frame:
            raise _error_type("scratch directory changed during preflight")
    except _error_type:
        raise
    except _os_error as error:
        raise _error_type(
            f"cannot verify the registered scratch directory: {error}"
        ) from error
    finally:
        _close(descriptor)


def _require_owned_empty_directory_stat(
    value: os.stat_result,
    *,
    _stat_type: type[os.stat_result] = os.stat_result,
    _is_directory: Callable[[int], bool] = stat.S_ISDIR,
    _permission_bits: Callable[[int], int] = stat.S_IMODE,
    _getuid: Callable[[], int] = os.geteuid,
    _error_type: type[Experiment006SeedWorkerError] = Experiment006SeedWorkerError,
    _exact_type: Callable[[object], type[object]] = type,
    _int_type: type[int] = int,
) -> None:
    if _exact_type(value) is not _stat_type:
        raise _error_type("scratch directory stat has an invalid exact type")
    if (
        not _is_directory(value.st_mode)
        or _permission_bits(value.st_mode) != 0o700
        or _exact_type(value.st_uid) is not _int_type
        or value.st_uid != _getuid()
        or _exact_type(value.st_nlink) is not _int_type
        or value.st_nlink < 2
    ):
        raise _error_type("scratch directory is not an owned mode-0700 directory")


def _directory_stat_frame(
    value: os.stat_result,
    *,
    _error_type: type[Experiment006SeedWorkerError] = Experiment006SeedWorkerError,
    _exact_type: Callable[[object], type[object]] = type,
    _int_type: type[int] = int,
    _any: Callable[[Any], bool] = any,
) -> tuple[int, ...]:
    fields = (
        value.st_mode,
        value.st_ino,
        value.st_dev,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
    )
    if _any(_exact_type(field) is not _int_type for field in fields):
        raise _error_type("scratch directory stat fields have invalid exact types")
    return fields


def _terminal_payload(
    completed: RegisteredCompletedTrainingHistory,
    checkpoint: RegisteredSerializedCheckpoint,
    *,
    seed: int,
    _positive_int: Callable[[object, str], None] | None = None,
    _sha256: Callable[[object, str], None] | None = None,
    _completed_type: type[RegisteredCompletedTrainingHistory] = (
        RegisteredCompletedTrainingHistory
    ),
    _checkpoint_type: type[RegisteredSerializedCheckpoint] = (
        RegisteredSerializedCheckpoint
    ),
    _evaluated_type: type[RegisteredEvaluatedEpoch] = RegisteredEvaluatedEpoch,
    _epoch_count: int = _EPOCH_COUNT,
    _history_domain: bytes = _HISTORY_DOMAIN,
    _sha256_digest: Callable[[bytes], _HashResult] = hashlib.sha256,
    _terminal_type: type[_TerminalPayload] = _TerminalPayload,
    _error_type: type[Experiment006SeedWorkerError] = Experiment006SeedWorkerError,
    _exact_type: Callable[[object], type[object]] = type,
    _int_type: type[int] = int,
    _str_type: type[str] = str,
    _bytes_type: type[bytes] = bytes,
    _length: Callable[[Any], int] = len,
) -> _TerminalPayload:
    if _positive_int is None:
        _positive_int = _require_positive_int
    if _sha256 is None:
        _sha256 = _require_sha256
    if _exact_type(completed) is not _completed_type:
        raise _error_type("completed history has an invalid exact type")
    if _exact_type(checkpoint) is not _checkpoint_type:
        raise _error_type("serialized checkpoint has an invalid exact type")
    winner = completed.winner
    if _exact_type(winner) is not _evaluated_type:
        raise _error_type("history winner has an invalid exact type")

    history_bytes = completed.canonical_json_bytes
    history_sha256 = completed.history_sha256
    winner_seed = winner.seed
    winner_epoch = winner.zero_based_epoch
    winner_model_sha256 = winner.model_tensor_sha256
    checkpoint_seed = checkpoint.seed
    checkpoint_epoch = checkpoint.zero_based_epoch
    checkpoint_history_sha256 = checkpoint.history_sha256
    checkpoint_model_sha256 = checkpoint.model_tensor_sha256
    safetensors_bytes = checkpoint.safetensors_bytes
    safetensors_sha256 = checkpoint.safetensors_sha256
    safetensors_byte_count = checkpoint.safetensors_byte_count

    _positive_int(completed.epoch_count, "completed epoch count")
    if completed.epoch_count != _epoch_count:
        raise _error_type("completed history does not contain exactly thirty epochs")
    if _exact_type(history_bytes) is not _bytes_type or not history_bytes.endswith(
        b"\n"
    ):
        raise _error_type("canonical history bytes have an invalid exact form")
    _sha256(history_sha256, "history sha256")
    _sha256(winner_model_sha256, "winner model tensor sha256")
    _sha256(safetensors_sha256, "safetensors sha256")
    _positive_int(safetensors_byte_count, "safetensors byte count")
    if (
        _exact_type(seed) is not _int_type
        or _exact_type(completed.seed) is not _int_type
        or completed.seed != seed
        or _exact_type(winner_seed) is not _int_type
        or winner_seed != seed
        or _exact_type(winner_epoch) is not _int_type
        or not 0 <= winner_epoch < _epoch_count
        or _exact_type(checkpoint_seed) is not _int_type
        or checkpoint_seed != seed
        or _exact_type(checkpoint_epoch) is not _int_type
        or checkpoint_epoch != winner_epoch
        or _exact_type(checkpoint_history_sha256) is not _str_type
        or checkpoint_history_sha256 != history_sha256
        or _exact_type(checkpoint_model_sha256) is not _str_type
        or checkpoint_model_sha256 != winner_model_sha256
        or _exact_type(safetensors_bytes) is not _bytes_type
        or not safetensors_bytes
        or safetensors_byte_count != _length(safetensors_bytes)
        or _sha256_digest(_history_domain + history_bytes).hexdigest() != history_sha256
        or _sha256_digest(safetensors_bytes).hexdigest() != safetensors_sha256
    ):
        raise _error_type(
            "history and checkpoint terminal bindings do not agree exactly"
        )
    return _terminal_type(
        history_json_bytes=history_bytes,
        history_sha256=history_sha256,
        safetensors_bytes=safetensors_bytes,
        safetensors_sha256=safetensors_sha256,
        winner_epoch=winner_epoch,
        model_tensor_sha256=winner_model_sha256,
    )


def _child_result_binding(
    registration: VerifiedRunRegistration,
    activation: _ActivationBinding,
    *,
    _lower_hex: Callable[[object, int, str], None] | None = None,
    _sha256: Callable[[object, str], None] | None = None,
    _binding_type: type[ChildResultBinding] = ChildResultBinding,
    _training_role: str = _TRAINING_ROLE,
    _rerun_role: str = _RERUN_ROLE,
    _error_type: type[Experiment006SeedWorkerError] = Experiment006SeedWorkerError,
    _exact_type: Callable[[object], type[object]] = type,
) -> ChildResultBinding:
    if _lower_hex is None:
        _lower_hex = _require_lower_hex
    if _sha256 is None:
        _sha256 = _require_sha256
    head_commit = registration.head_commit
    implementation_commit = registration.implementation_commit
    registration_sha256 = registration.registration_sha256
    source_bundle_sha256 = registration.source_bundle_sha256
    _lower_hex(head_commit, 40, "registration HEAD commit")
    _lower_hex(
        implementation_commit,
        40,
        "registration implementation commit",
    )
    _sha256(registration_sha256, "registration sha256")
    _sha256(source_bundle_sha256, "source bundle sha256")
    role: _ChildRole
    if activation.role == _training_role:
        role = "training_seed"
    elif activation.role == _rerun_role:
        role = "selected_seed_rerun"
    else:
        raise _error_type("activation role is not registered")
    result = _binding_type(
        role=role,
        ordinal=activation.ordinal,
        seed=activation.seed,
        registration_head_commit=head_commit,
        implementation_commit=implementation_commit,
        registration_sha256=registration_sha256,
        source_bundle_sha256=source_bundle_sha256,
    )
    if _exact_type(result) is not _binding_type:
        raise _error_type("child result binding has an invalid exact type")
    return result


def _require_positive_int(
    value: object,
    name: str,
    *,
    _error_type: type[Experiment006SeedWorkerError] = Experiment006SeedWorkerError,
    _exact_type: Callable[[object], type[object]] = type,
    _int_type: type[int] = int,
    _type_error: type[TypeError] = TypeError,
    _cast: Callable[[Any, object], Any] = cast,
) -> None:
    if _exact_type(value) is not _int_type:
        raise _type_error(f"{name} must be an exact integer")
    integer = _cast(_int_type, value)
    if integer < 1:
        raise _error_type(f"{name} must be positive")


def _require_lower_hex(
    value: object,
    length: int,
    name: str,
    *,
    _lower_hex: frozenset[str] = _LOWER_HEX,
    _error_type: type[Experiment006SeedWorkerError] = Experiment006SeedWorkerError,
    _exact_type: Callable[[object], type[object]] = type,
    _str_type: type[str] = str,
    _length: Callable[[Any], int] = len,
    _any: Callable[[Any], bool] = any,
    _cast: Callable[[Any, object], Any] = cast,
) -> None:
    if _exact_type(value) is not _str_type or _length(value) != length:
        raise _error_type(f"{name} must be {length} lowercase hexadecimal characters")
    text = _cast(_str_type, value)
    if _any(character not in _lower_hex for character in text):
        raise _error_type(f"{name} must be {length} lowercase hexadecimal characters")


def _require_sha256(
    value: object,
    name: str,
    *,
    _lower_hex: Callable[[object, int, str], None] = _require_lower_hex,
) -> None:
    _lower_hex(value, 64, name)


def _make_registered_seed_process() -> Callable[
    [VerifiedRunRegistration, VerifiedChildActivation], None
]:
    """Capture the irreversible one-attempt worker latch outside module state."""

    worker_error = Experiment006SeedWorkerError
    exact_type = type
    argument_type_error = TypeError
    epoch_range = range
    get_process_id = os.getpid
    registration_type = VerifiedRunRegistration
    activation_type = VerifiedChildActivation
    normalization_identity_type = NormalizationArtifactIdentity
    pcm_cache_identity_type = PCMCacheIdentity
    pcm_cache_type = Experiment002PCMCache
    completed_history_type = RegisteredCompletedTrainingHistory
    epoch_count = _EPOCH_COUNT

    initial_activation = _initial_activation_binding
    verify_activation = _verify_activation_boundary
    snapshot_supplier = _registered_child_input_snapshot
    require_snapshot = _require_child_input_snapshot
    compare_snapshots = _child_input_snapshots_match
    preflight_impl = _preflight_scratch_directory
    require_scratch_stat = _require_owned_empty_directory_stat
    scratch_stat_frame = _directory_stat_frame
    terminal_impl = _terminal_payload
    positive_int = _require_positive_int
    sha256_validator = _require_sha256
    result_binding_impl = _child_result_binding
    lower_hex_validator = _require_lower_hex

    load_corpus = load_registered_corpus
    load_normalization = load_normalization_artifact
    open_cache = Experiment002PCMCache.open
    materialize_validation = materialize_registered_validation
    bind_validation = bind_registered_validation_inputs
    bind_training = bind_registered_training_inputs
    create_executor = create_registered_training_executor
    create_history = create_registered_training_history
    execute_epoch = execute_registered_training_epoch
    evaluate_epoch = evaluate_registered_epoch
    consume_epoch = consume_registered_evaluated_epoch
    advance_executor = advance_registered_training_executor
    complete_executor = complete_registered_training_executor
    verify_history = verify_registered_completed_training_history
    serialize_checkpoint = serialize_registered_checkpoint
    verify_checkpoint = verify_registered_serialized_checkpoint
    write_result_route = write_registered_child_result

    def preflight_scratch(path: Path) -> None:
        preflight_impl(
            path,
            _require_stat=require_scratch_stat,
            _stat_frame=scratch_stat_frame,
        )

    def capture_terminal(
        completed: RegisteredCompletedTrainingHistory,
        checkpoint: RegisteredSerializedCheckpoint,
        *,
        seed: int,
    ) -> _TerminalPayload:
        return terminal_impl(
            completed,
            checkpoint,
            seed=seed,
            _positive_int=positive_int,
            _sha256=sha256_validator,
        )

    def capture_result_binding(
        registration: VerifiedRunRegistration,
        binding: _ActivationBinding,
    ) -> ChildResultBinding:
        return result_binding_impl(
            registration,
            binding,
            _lower_hex=lower_hex_validator,
            _sha256=sha256_validator,
        )

    attempted = False
    worker_process_id = get_process_id()
    attempt_lock = threading.Lock()

    def run_registered_seed_process(
        registration: VerifiedRunRegistration,
        activation: VerifiedChildActivation,
        /,
    ) -> None:
        nonlocal attempted

        if get_process_id() != worker_process_id:
            raise worker_error(
                "registered seed worker cannot be inherited across a fork"
            )
        with attempt_lock:
            if get_process_id() != worker_process_id:
                raise worker_error(
                    "registered seed worker cannot be inherited across a fork"
                )
            if attempted:
                raise worker_error("registered seed process was already attempted")
            attempted = True

        if exact_type(registration) is not registration_type:
            raise argument_type_error(
                "registration must be an exact VerifiedRunRegistration"
            )
        if exact_type(activation) is not activation_type:
            raise argument_type_error(
                "activation must be an exact VerifiedChildActivation"
            )

        activation_binding = initial_activation(registration, activation)
        first_snapshot = snapshot_supplier(registration)
        require_snapshot(first_snapshot)
        verify_activation(registration, activation, activation_binding)
        preflight_scratch(first_snapshot.scratch_directory)
        verify_activation(registration, activation, activation_binding)

        corpus = load_corpus(first_snapshot.manifest_path)
        verify_activation(registration, activation, activation_binding)
        normalization_identity = normalization_identity_type(
            byte_count=first_snapshot.normalization_byte_count,
            sha256=first_snapshot.normalization_sha256,
        )
        pcm_cache_identity = pcm_cache_identity_type(
            byte_count=first_snapshot.pcm_cache_byte_count,
            sha256=first_snapshot.pcm_cache_sha256,
        )
        if exact_type(normalization_identity) is not normalization_identity_type:
            raise worker_error("normalization identity has an invalid exact type")
        if exact_type(pcm_cache_identity) is not pcm_cache_identity_type:
            raise worker_error("PCM cache identity has an invalid exact type")
        normalization = load_normalization(
            first_snapshot.normalization_path,
            normalization_identity,
        )
        verify_activation(registration, activation, activation_binding)

        opened_cache = open_cache(
            corpus,
            first_snapshot.pcm_cache_path,
            pcm_cache_identity,
        )
        if exact_type(opened_cache) is not pcm_cache_type:
            raise worker_error("opened PCM cache has an invalid exact type")
        with opened_cache as cache:
            if exact_type(cache) is not pcm_cache_type or cache is not opened_cache:
                raise worker_error(
                    "PCM cache context changed the opened cache identity"
                )
            verify_activation(registration, activation, activation_binding)
            validation_population = materialize_validation(
                corpus,
                cache.validation,
                normalization,
            )
            verify_activation(registration, activation, activation_binding)
            validation_inputs = bind_validation(
                corpus,
                validation_population,
            )
            verify_activation(registration, activation, activation_binding)
            training_inputs = bind_training(
                corpus,
                cache.training,
                normalization,
            )
            verify_activation(registration, activation, activation_binding)
            executor = create_executor(
                registration,
                training_inputs,
                validation_inputs,
                seed=activation_binding.seed,
            )
            verify_activation(registration, activation, activation_binding)
            history = create_history(
                registration,
                executor,
                validation_inputs,
                seed=activation_binding.seed,
            )
            verify_activation(registration, activation, activation_binding)

            completed: RegisteredCompletedTrainingHistory | None = None
            for zero_based_epoch in epoch_range(epoch_count):
                verify_activation(
                    registration,
                    activation,
                    activation_binding,
                )
                handoff = execute_epoch(executor)
                verify_activation(
                    registration,
                    activation,
                    activation_binding,
                )
                evaluated = evaluate_epoch(
                    registration,
                    handoff,
                    validation_inputs,
                )
                verify_activation(
                    registration,
                    activation,
                    activation_binding,
                )
                barrier = consume_epoch(history, evaluated)
                verify_activation(
                    registration,
                    activation,
                    activation_binding,
                )
                if zero_based_epoch < epoch_count - 1:
                    result = advance_executor(executor, barrier)
                    if result is not None:
                        raise worker_error(
                            "executor advance returned an unexpected value"
                        )
                else:
                    completed = complete_executor(
                        executor,
                        barrier,
                    )
                verify_activation(
                    registration,
                    activation,
                    activation_binding,
                )

            if exact_type(completed) is not completed_history_type:
                raise worker_error("epoch 29 did not return an exact completed history")
            assert completed is not None
            verified_history = verify_history(completed)
            if verified_history is not None:
                raise worker_error(
                    "completed history verifier returned an unexpected value"
                )
            verify_activation(registration, activation, activation_binding)
            checkpoint = serialize_checkpoint(completed)
            verify_activation(registration, activation, activation_binding)
            verified_checkpoint = verify_checkpoint(checkpoint)
            if verified_checkpoint is not None:
                raise worker_error("checkpoint verifier returned an unexpected value")
            verify_activation(registration, activation, activation_binding)
            terminal = capture_terminal(
                completed,
                checkpoint,
                seed=activation_binding.seed,
            )
            result_binding = capture_result_binding(registration, activation_binding)
            second_snapshot = snapshot_supplier(registration)
            require_snapshot(second_snapshot)
            if not compare_snapshots(first_snapshot, second_snapshot):
                raise worker_error(
                    "registered child inputs changed across the seed run"
                )
            verify_activation(registration, activation, activation_binding)
            write_result = write_result_route(
                second_snapshot.scratch_directory,
                result_binding,
                history_json_bytes=terminal.history_json_bytes,
                history_sha256=terminal.history_sha256,
                safetensors_bytes=terminal.safetensors_bytes,
                safetensors_sha256=terminal.safetensors_sha256,
                winner_epoch=terminal.winner_epoch,
                model_tensor_sha256=terminal.model_tensor_sha256,
            )
            verify_activation(registration, activation, activation_binding)
            if write_result is not None:
                raise worker_error("child result writer returned an unexpected value")
        return None

    return run_registered_seed_process


run_registered_seed_process = _make_registered_seed_process()
del _make_registered_seed_process
