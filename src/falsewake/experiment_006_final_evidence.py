"""Pure completed-run evidence for the registered Experiment 006 selector.

This module accepts only the immutable nested-tuple result produced by the
registered parent selector.  It deliberately imports no coordinator or training
module: all four child frames, histories, envelopes, and safetensors payloads are
independently framed with the Python standard library before completed evidence is
issued.

Only the deterministic completed outcomes live here.  Within a trusted CPython
process, issuance is deterministic and tamper-evident for isolated persistent
mutation checked by these routes.  It is not a security boundary against
coordinated arbitrary closure, ``gc``, ``ctypes``, debugger, or direct memory
rewriting.  The pure builder may be reused and called from multiple threads; the
publisher owns the later one-shot process boundary.  ``execution_failure`` and
filesystem publication are separate later boundaries.
"""

from __future__ import annotations

import builtins
import hashlib
import json
import math
import os
import struct
import threading
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from types import CodeType, FunctionType, MemberDescriptorType
from typing import Any, Final, Literal, NoReturn, SupportsIndex, cast

from falsewake import experiment_006_run_authority as _run_authority
from falsewake.experiment_006_run_authority import VerifiedRunRegistration

__all__ = (
    "ArtifactEvidence",
    "ArtifactKind",
    "ChildEvidence",
    "ClassRecallEvidence",
    "COMPLETED_PUBLICATION_BYTES_MAXIMUM",
    "CompletedStatus",
    "CompletedPublicationSnapshot",
    "Experiment006FinalEvidenceError",
    "FINAL_REPORT_BYTES_MAXIMUM",
    "FINAL_REPORT_PATH",
    "FinalCompletedEvidence",
    "GatesEvidence",
    "PublicationPayloadPlan",
    "RegistrationPublicationFrame",
    "RegistrationEvidence",
    "RerunEvidence",
    "SELECTED_RERUN_HISTORY_PATH",
    "SELECTED_SAFETENSORS_PATH",
    "SelectionEvidence",
    "TRAINING_HISTORY_PATHS",
    "build_completed_final_evidence",
    "canonical_completed_report_bytes",
    "snapshot_completed_publication",
    "verify_completed_final_evidence",
)

type CompletedStatus = Literal["pass", "gate_failure"]
type ChildRole = Literal["training_seed", "selected_seed_rerun"]
type ArtifactKind = Literal[
    "training_history", "selected_rerun_history", "selected_safetensors"
]
type _ConfusionMatrix = tuple[tuple[int, ...], ...]
type PublicationPayloadPlan = tuple[
    str,
    ArtifactKind,
    str,
    bytes,
    int,
    str,
    str | None,
    str | None,
    int,
]
type RegistrationPublicationFrame = tuple[str, str, str, str]
type CompletedPublicationSnapshot = tuple[
    CompletedStatus,
    RegistrationPublicationFrame,
    tuple[PublicationPayloadPlan, ...],
    str,
    str,
    bytes,
]

TRAINING_HISTORY_PATHS: Final = (
    "reports/experiment-006-seed-20260719-history.json",
    "reports/experiment-006-seed-20260720-history.json",
    "reports/experiment-006-seed-20260721-history.json",
)
SELECTED_RERUN_HISTORY_PATH: Final = (
    "reports/experiment-006-selected-rerun-history.json"
)
SELECTED_SAFETENSORS_PATH: Final = "models/experiment-006-selected.safetensors"
FINAL_REPORT_PATH: Final = "reports/experiment-006-training.json"
FINAL_REPORT_BYTES_MAXIMUM: Final = 256 << 10
COMPLETED_PUBLICATION_BYTES_MAXIMUM: Final = 8 << 20

_SCHEMA_VERSION: Final = 1
_EXPERIMENT: Final = "006"
_SCIENTIFIC_PROTOCOL: Final = "002"
_REGISTERED_SEEDS: Final = (20_260_719, 20_260_720, 20_260_721)
_SEED_ROLE: Final = "training_seed"
_RERUN_ROLE: Final = "selected_seed_rerun"
_EPOCH_COUNT: Final = 30
_UPDATES_PER_EPOCH: Final = 313
_HISTORY_DOMAIN: Final = b"falsewake-exp002-history-v1\0"
_ENVELOPE_DOMAIN: Final = b"falsewake-exp002-child-result-envelope-v1\0"
_MODEL_TENSOR_DOMAIN: Final = b"falsewake-exp002-model-tensors-v1\0"
_F32LE_TAG: Final = b"F32LE"
_HISTORY_FILENAME: Final = "experiment-002-history.json"
_SAFETENSORS_FILENAME: Final = "experiment-002-winner.safetensors"
_MAX_HISTORY_BYTES: Final = 1 << 20
_MAX_ENVELOPE_BYTES: Final = 64 << 10
_REGISTERED_SAFETENSORS_BYTES: Final = 99_776
_REGISTERED_SAFETENSORS_HEADER_BYTES: Final = 4_872
_REGISTERED_MODEL_PAYLOAD_BYTES: Final = 94_896
_REGISTERED_MODEL_TENSOR_COUNT: Final = 53
_REGISTERED_MODEL_VALUE_COUNT: Final = 23_724
_WALL_NANOSECONDS_MAXIMUM: Final = 21_600 * 1_000_000_000
_RSS_BYTES_MAXIMUM: Final = 8_589_934_592
_OUTPUT_BYTES_MAXIMUM: Final = 4_294_967_296
_UINT32_MAXIMUM: Final = (1 << 32) - 1
_ARTIFACT_MODE: Final = "0444"
_PREDECESSOR_EXECUTION_PROTOCOL_COMMIT: Final = (
    "0b6bf2cac2d4f6a04d6ced596bf8f835660eb072"
)
_PREDECESSOR_EXECUTION_PROTOCOL_SHA256: Final = (
    "824d1677cf8f75567cf1915381f0a9bbc778a74cba879ace80733c3ceb48e54d"
)
_PREDECESSOR_IMPLEMENTATION_COMMIT: Final = "04b634451d129faadadd921215123547e3467d65"
_PREDECESSOR_PROTOCOL_PROOF_SHA256: Final = (
    "cf6d4ee2fcba31de37774b591571abb2d86998cc0f2a3e99a40abe1af4af5414"
)
_PREDECESSOR_SHARED_AUTHORITY_SHA256: Final = (
    "937ad120dd5828cb4029cc1382df2a0a7c06aef14c1d37bfdb848d93993416af"
)
_PREDECESSOR_INCIDENT_COMMIT: Final = "594e2c491c057394f18860c8362a51190460645f"
_PREDECESSOR_INCIDENT_SHA256: Final = (
    "3de6fbbd6912bee47f3ef01f5371b57ae8d5942b6fc9915fc0ec243b9e931e3d"
)
_PREDECESSOR_PREFLIGHT_BOUNDARY_COMMIT: Final = (
    "9ae549ccd477b13b9531f7c44d854d34549fd9c3"
)
_PREDECESSOR_PREFLIGHT_BOUNDARY_PARENT: Final = (
    "1e20d6cf210a2d4a59cfc737a14eaccfc94a5b74"
)
_PREDECESSOR_CANONICAL_JSON: Final = (
    '{"attempt":{"canonical_marker":"/home/ubuntu/gitcode/.t/falsewake-experiment'
    '-005-attempt","canonical_marker_present":false,"optimizer_updates":0,"regist'
    'ered_attempt_consumed":false,"registered_authority_issuer_invoked":false,"re'
    'gistered_coordinator_invoked":false,"registered_invocation_count":0,"registe'
    'red_runner_invoked":false,"validation_examples":0},"execution_protocol":{"in'
    'troduction_commit":"0b6bf2cac2d4f6a04d6ced596bf8f835660eb072","mode":"100644'
    '","path":"configs/experiment-005-execution.json","sha256":"824d1677cf8f75567'
    'cf1915381f0a9bbc778a74cba879ace80733c3ceb48e54d"},"experiment":"005","implem'
    'entation":{"commit":"04b634451d129faadadd921215123547e3467d65","production_s'
    'ources":[{"path":"src/falsewake/experiment_005_coordinator.py","sha256":"4ad'
    '78de14ce12acdb63b4e103dd920ce878d7e60a06fa1c5893027d17401f812"},{"path":"src'
    '/falsewake/experiment_005_final_evidence.py","sha256":"1c99757fc9b78e612891a'
    '9d49e7dc40c2d0b90599a40927e4cd41dff781f355d"},{"path":"src/falsewake/experim'
    'ent_005_final_publication.py","sha256":"84f7aba57f8ec0c096e5f18a03a8118800f5'
    '3a9cfe4910c554bea677a9f79f07"},{"path":"src/falsewake/experiment_005_run_aut'
    'hority.py","sha256":"87d18e901b4b814ce31c7c17b60b6aa54f837edc84ad3bb5ee1af38'
    '6df006f5e"},{"path":"src/falsewake/experiment_005_runner.py","sha256":"eeaff'
    'ace36baaabba5e1479750674884a069b41ccaeeb9dee9ccbc4b179a5336"},{"path":"src/f'
    'alsewake/experiment_005_seed_worker.py","sha256":"0840a62f0f4327eec7649d17ad'
    '334edfa6941d6146bff6fe50e2d693f0b80f96"},{"path":"src/falsewake/experiment_0'
    '05_supervisor.py","sha256":"7c8d58566c2b5e904ebf7f1e5fa1ab3b1c79c4387b38f296'
    'b08ef5ce7e312e71"}],"protocol_proof":{"commit":"1e20d6cf210a2d4a59cfc737a14e'
    'accfc94a5b74","path":"tests/test_experiment_005_protocol.py","sha256":"cf6d4'
    'ee2fcba31de37774b591571abb2d86998cc0f2a3e99a40abe1af4af5414"},"shared_author'
    'ity":{"path":"src/falsewake/experiment_002_run_authority.py","sha256":"937ad'
    '120dd5828cb4029cc1382df2a0a7c06aef14c1d37bfdb848d93993416af"}},"incident":{"'
    'byte_count":6612,"commit":"594e2c491c057394f18860c8362a51190460645f","mode":'
    '"100644","parent":"9ae549ccd477b13b9531f7c44d854d34549fd9c3","path":"reports'
    '/experiment-005-preflight-incident.json","sha256":"3de6fbbd6912bee47f3ef01f5'
    '371b57ae8d5942b6fc9915fc0ec243b9e931e3d"},"managed_outputs_absent":["configs'
    '/experiment-005-run.json","models/experiment-005-selected.safetensors","repo'
    'rts/experiment-005-seed-20260719-history.json","reports/experiment-005-seed-'
    '20260720-history.json","reports/experiment-005-seed-20260721-history.json","'
    'reports/experiment-005-selected-rerun-history.json","reports/experiment-005-'
    'training.json"],"outcome":{"automatic_terminal_report_published":false,"chec'
    'kpoint_reusable":false,"code":"facade_normalization_recipe_cannot_express_tr'
    'uthful_profile_005","phase":"pre_registration_protocol_preflight","reason":"'
    "frozen_facade_normalization_cannot_express_the_truthful_profile_005_docstrin"
    'g","status":"preflight_rejected"},"preflight_boundary":{"commit":"9ae549ccd4'
    '77b13b9531f7c44d854d34549fd9c3","mode":"100644","parent":"1e20d6cf210a2d4a59'
    'cfc737a14eaccfc94a5b74","path":"docs/experiment-005.md","sha256":"c9852a028f'
    '572e33b2379b98786f5c72e41ddae1300f03742ad977947a1dd600"},"preflight_evidence'
    '":{"facade_candidate_normalized_sha256":"9778140580667e95e120a4a8472d8b79ffb'
    '7034c3886c208fc3fd3c4da966c86","facade_candidate_sha256":"87d18e901b4b814ce3'
    '1c7c17b60b6aa54f837edc84ad3bb5ee1af386df006f5e","facade_reference_sha256":"c'
    'acab13729f14bfdce10afa5b53b2a09116cf56abe21aee12ac9220a6ad2e5b5","normalized'
    '_added_lines":1,"normalized_deleted_lines":1,"registration_gate":"not_run"},'
    '"registration":{"path":"configs/experiment-005-run.json","present":false},"r'
    'euse_forbidden":true,"scientific_protocol":"002","terminal":true,"topology":'
    '{"implementation_commit":"04b634451d129faadadd921215123547e3467d65","impleme'
    'ntation_commit_is_ancestor_of_incident":true,"incident_commit":"594e2c491c05'
    '7394f18860c8362a51190460645f","incident_parent":"9ae549ccd477b13b9531f7c44d8'
    '54d34549fd9c3","outcome_report_path":"reports/experiment-005-training.json",'
    '"outcome_report_present":false,"preflight_boundary_commit":"9ae549ccd477b13b'
    '9531f7c44d854d34549fd9c3","preflight_boundary_parent":"1e20d6cf210a2d4a59cfc'
    '737a14eaccfc94a5b74","protocol_commit":"0b6bf2cac2d4f6a04d6ced596bf8f835660e'
    'b072","protocol_parent":"462aeba306a0612fd6d64884e323d72db3569a89","protocol'
    '_proof_commit":"1e20d6cf210a2d4a59cfc737a14eaccfc94a5b74","protocol_proof_pa'
    'rent":"04b634451d129faadadd921215123547e3467d65","registration_path":"config'
    's/experiment-005-run.json","registration_present":false}}'
)
_CLASS_ORDER: Final = (
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
    "unknown",
    "silence",
)
_CLASS_SUPPORT: Final = (
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
_TARGET_COUNT: Final = 10
_TARGET_POPULATION: Final = sum(_CLASS_SUPPORT[:_TARGET_COUNT])
_LOWER_HEX: Final = frozenset("0123456789abcdef")
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


class Experiment006FinalEvidenceError(ValueError):
    """A selector result or completed-evidence value failed closed."""


@dataclass(frozen=True, slots=True)
class RegistrationEvidence:
    """Exact source-bound registration shared by all four children."""

    head_commit: str
    implementation_commit: str
    registration_sha256: str
    source_bundle_sha256: str


@dataclass(frozen=True, slots=True)
class ChildEvidence:
    """Deterministic result and measured resources for one child."""

    role: ChildRole
    ordinal: int
    seed: int
    pid: int
    cpu_ids: tuple[int, int]
    elapsed_nanoseconds: int
    maximum_rss_bytes: int
    output_and_scratch_bytes: int
    history_byte_count: int
    history_sha256: str
    history_domain_sha256: str
    safetensors_byte_count: int
    safetensors_sha256: str
    envelope_byte_count: int
    envelope_domain_sha256: str
    winner_epoch: int
    winner_macro_f1_numerator: int
    winner_macro_f1_denominator: int
    winner_validation_cross_entropy_float64_hex: str
    model_tensor_sha256: str


@dataclass(frozen=True, slots=True)
class SelectionEvidence:
    """Independent reconstruction of the registered cross-seed rank."""

    selected_training_ordinal: int
    selected_seed: int
    selected_winner_epoch: int
    selected_macro_f1_numerator: int
    selected_macro_f1_denominator: int
    selected_validation_cross_entropy_float64_hex: str
    selected_model_tensor_sha256: str
    ranked_training_ordinals: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class RerunEvidence:
    """Explicit component equality between the selected seed and fresh rerun."""

    selected_training_ordinal: int
    rerun_ordinal: int
    selected_seed: int
    history_byte_identical: bool
    safetensors_byte_identical: bool
    validation_input_digest_identical: bool
    all_validation_prediction_digests_identical: bool
    all_model_tensor_digests_identical: bool
    selected_checkpoint_digest_identical: bool
    selected_rank_identical: bool


@dataclass(frozen=True, slots=True)
class ClassRecallEvidence:
    """One exact preregistered target-class recall gate."""

    class_index: int
    class_name: str
    correct: int
    support: int
    minimum_correct: int
    passed: bool


@dataclass(frozen=True, slots=True)
class GatesEvidence:
    """All exact validation gates recomputed from the rerun winner matrix."""

    target_accuracy_correct: int
    target_accuracy_total: int
    target_accuracy_minimum_correct: int
    target_accuracy_passed: bool
    target_recalls: tuple[
        ClassRecallEvidence,
        ClassRecallEvidence,
        ClassRecallEvidence,
        ClassRecallEvidence,
        ClassRecallEvidence,
        ClassRecallEvidence,
        ClassRecallEvidence,
        ClassRecallEvidence,
        ClassRecallEvidence,
        ClassRecallEvidence,
    ]
    macro_f1_numerator: int
    macro_f1_denominator: int
    macro_f1_minimum_numerator: int
    macro_f1_minimum_denominator: int
    macro_f1_passed: bool
    unknown_target_predictions: int
    unknown_total: int
    unknown_maximum_target_predictions: int
    unknown_target_rate_passed: bool
    silence_target_predictions: int
    silence_total: int
    silence_maximum_target_predictions: int
    silence_target_rate_passed: bool
    finite_contract_passed: bool
    all_passed: bool


@dataclass(frozen=True, slots=True)
class ArtifactEvidence:
    """One fixed completed-run payload retained for later publication."""

    path: str
    kind: ArtifactKind
    mode: str
    contents: bytes
    byte_count: int
    sha256: str
    domain_sha256: str | None
    model_tensor_sha256: str | None
    source_child_ordinal: int


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
class FinalCompletedEvidence:
    """Issuer-owned immutable evidence for pass or gate-failure completion."""

    schema_version: int
    experiment: str
    status: CompletedStatus
    registration: RegistrationEvidence
    children: tuple[ChildEvidence, ChildEvidence, ChildEvidence, ChildEvidence]
    selection: SelectionEvidence
    rerun: RerunEvidence
    gates: GatesEvidence
    artifacts: tuple[ArtifactEvidence, ...]
    report_path: str
    report_mode: str

    def __init__(self) -> None:
        raise TypeError("completed final evidence is issued only by the builder")

    def __copy__(self) -> NoReturn:
        raise TypeError("completed final evidence cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        del memo
        raise TypeError("completed final evidence cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("completed final evidence cannot be serialized")

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        del protocol
        raise TypeError("completed final evidence cannot be serialized")


@dataclass(frozen=True, slots=True)
class _ParsedEpoch:
    zero_based_epoch: int
    validation_confusion_matrix: _ConfusionMatrix
    validation_cross_entropy_float64_hex: str
    validation_cross_entropy: float
    macro_f1: Fraction
    validation_prediction_digest: str
    model_tensor_digest: str


@dataclass(frozen=True, slots=True)
class _ParsedHistory:
    raw_bytes: bytes
    seed: int
    validation_input_digest: str
    epochs: tuple[_ParsedEpoch, ...]
    winner: _ParsedEpoch
    raw_sha256: str
    domain_sha256: str


@dataclass(frozen=True, slots=True)
class _ParsedSnapshot:
    role: ChildRole
    ordinal: int
    seed: int
    registration: RegistrationEvidence
    history: _ParsedHistory
    safetensors_bytes: bytes
    safetensors_sha256: str
    envelope_byte_count: int
    envelope_domain_sha256: str
    winner_epoch: int
    winner_macro_f1_numerator: int
    winner_macro_f1_denominator: int
    winner_validation_cross_entropy_float64_hex: str
    model_tensor_sha256: str


@dataclass(frozen=True, slots=True)
class _ParsedChild:
    evidence: ChildEvidence
    snapshot: _ParsedSnapshot


@dataclass(frozen=True, slots=True)
class _CompletedFields:
    schema_version: int
    experiment: str
    status: CompletedStatus
    registration: RegistrationEvidence
    children: tuple[ChildEvidence, ChildEvidence, ChildEvidence, ChildEvidence]
    selection: SelectionEvidence
    rerun: RerunEvidence
    gates: GatesEvidence
    artifacts: tuple[ArtifactEvidence, ...]
    report_path: str
    report_mode: str


type _IssuedLedger = tuple[
    tuple[object, ...],
    bytes,
    CompletedPublicationSnapshot,
]
type _FunctionAuthority = tuple[
    str,
    FunctionType,
    CodeType,
    object,
    object,
    tuple[object, ...],
    tuple[object, ...],
]
type _NamespaceItems = tuple[tuple[str, object], ...]
type _JSONMethodAuthority = tuple[
    _FunctionAuthority,
    _NamespaceItems,
    dict[str, Any],
    _NamespaceItems,
    _NamespaceItems | None,
]
type _JSONClassAuthority = tuple[
    str,
    type[Any],
    _NamespaceItems,
    tuple[_JSONMethodAuthority, ...],
]


type _BuildRoute = Callable[
    [VerifiedRunRegistration, object],
    FinalCompletedEvidence,
]
type _VerifyRoute = Callable[[object], None]
type _ReportRoute = Callable[[object], bytes]
type _SnapshotRoute = Callable[[object], CompletedPublicationSnapshot]


def _fail(message: str) -> NoReturn:
    raise Experiment006FinalEvidenceError(message)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Experiment006FinalEvidenceError(
                "canonical JSON contains a duplicate key"
            )
        result[key] = value
    return result


def _reject_json_number(value: str) -> NoReturn:
    del value
    raise Experiment006FinalEvidenceError("canonical JSON contains a forbidden number")


def _canonical_json_bytes(document: object) -> bytes:
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
    except (TypeError, ValueError, UnicodeError) as error:
        raise Experiment006FinalEvidenceError(
            "completed evidence is not canonical JSON"
        ) from error


def _parse_canonical_object(raw: object, maximum: int, name: str) -> dict[str, Any]:
    if type(raw) is not bytes:
        raise TypeError(f"{name} must be exact bytes")
    payload = raw
    if not 0 < len(payload) <= maximum or not payload.endswith(b"\n"):
        _fail(f"{name} has an invalid byte count or terminator")
    try:
        text = payload.decode("utf-8", errors="strict")
        parsed = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_number,
            parse_float=_reject_json_number,
        )
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as error:
        raise Experiment006FinalEvidenceError(f"{name} is not exact JSON") from error
    if type(parsed) is not dict:
        _fail(f"{name} must be a JSON object")
    document = cast(dict[str, Any], parsed)
    if _canonical_json_bytes(document) != payload:
        _fail(f"{name} is not canonical compact ASCII JSON plus LF")
    return document


def _require_exact_keys(
    value: dict[str, Any], expected: frozenset[str], name: str
) -> None:
    if frozenset(value) != expected:
        _fail(f"{name} has unexpected or missing keys")


def _require_object(value: object, keys: frozenset[str], name: str) -> dict[str, Any]:
    if type(value) is not dict:
        _fail(f"{name} must be an exact object")
    result = cast(dict[str, Any], value)
    _require_exact_keys(result, keys, name)
    return result


def _require_int(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(f"{name} must be an exact integer at least {minimum}")
    return value


def _require_string(value: object, expected: str, name: str) -> None:
    if type(value) is not str or value != expected:
        _fail(f"{name} differs from its exact registered value")


def _require_hex(value: object, length: int, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != length
        or any(character not in _LOWER_HEX for character in value)
    ):
        _fail(f"{name} is not lowercase hexadecimal with length {length}")
    return value


def _require_float_hex(value: object, name: str) -> tuple[str, float]:
    if type(value) is not str:
        _fail(f"{name} must be an exact string")
    encoded = value
    try:
        parsed = float.fromhex(encoded)
    except ValueError as error:
        raise Experiment006FinalEvidenceError(f"{name} is not binary64 hex") from error
    if (
        not math.isfinite(parsed)
        or parsed < 0.0
        or math.copysign(1.0, parsed) < 0.0
        or parsed.hex() != encoded
        or encoded.lower() != encoded
    ):
        _fail(f"{name} is not canonical finite nonnegative binary64 hex")
    return encoded, parsed


def _macro_f1(confusion: _ConfusionMatrix) -> Fraction:
    total = Fraction(0, 1)
    for class_index in range(len(_CLASS_SUPPORT)):
        true_positive = confusion[class_index][class_index]
        false_positive = sum(
            confusion[row][class_index]
            for row in range(len(_CLASS_SUPPORT))
            if row != class_index
        )
        false_negative = _CLASS_SUPPORT[class_index] - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        total += (
            Fraction(0, 1)
            if denominator == 0
            else Fraction(2 * true_positive, denominator)
        )
    return total / len(_CLASS_SUPPORT)


def _parse_confusion(value: object) -> _ConfusionMatrix:
    if type(value) is not list or len(cast(list[object], value)) != len(_CLASS_SUPPORT):
        _fail("validation confusion matrix must have twelve rows")
    rows: list[tuple[int, ...]] = []
    for row_index, raw_row in enumerate(cast(list[object], value)):
        if type(raw_row) is not list or len(cast(list[object], raw_row)) != len(
            _CLASS_SUPPORT
        ):
            _fail("validation confusion matrix rows must have twelve cells")
        row = tuple(
            _require_int(cell, "validation confusion cell")
            for cell in cast(list[object], raw_row)
        )
        if sum(row) != _CLASS_SUPPORT[row_index]:
            _fail("validation confusion row differs from registered class support")
        rows.append(row)
    return tuple(rows)


def _parse_history(raw: object, expected_seed: int) -> _ParsedHistory:
    document = _parse_canonical_object(raw, _MAX_HISTORY_BYTES, "registered history")
    _require_exact_keys(document, _TOP_HISTORY_KEYS, "registered history")
    if _require_int(document["schema_version"], "history schema_version") != 1:
        _fail("history schema_version must be integer one")
    _require_string(
        document["experiment"],
        _SCIENTIFIC_PROTOCOL,
        "history scientific protocol",
    )
    seed = _require_int(document["seed"], "history seed")
    if seed != expected_seed or seed not in _REGISTERED_SEEDS:
        _fail("history seed differs from its registered child")
    validation_input_digest = _require_hex(
        document["validation_input_digest"], 64, "validation input digest"
    )
    _require_hex(
        document["complete_update_trace_digest"], 64, "complete update trace digest"
    )
    raw_epochs = document["epochs"]
    if (
        type(raw_epochs) is not list
        or len(cast(list[object], raw_epochs)) != _EPOCH_COUNT
    ):
        _fail("registered history must contain exactly thirty epochs")
    epochs: list[_ParsedEpoch] = []
    for expected_epoch, raw_epoch in enumerate(cast(list[object], raw_epochs)):
        epoch = _require_object(raw_epoch, _EPOCH_KEYS, "history epoch")
        if (
            _require_int(epoch["zero_based_epoch"], "zero_based_epoch")
            != expected_epoch
        ):
            _fail("registered history epochs are reordered or incomplete")
        if (
            _require_int(epoch["first_global_update"], "first_global_update")
            != expected_epoch * _UPDATES_PER_EPOCH
            or _require_int(
                epoch["last_global_update_inclusive"],
                "last_global_update_inclusive",
            )
            != (expected_epoch + 1) * _UPDATES_PER_EPOCH - 1
        ):
            _fail("registered history update interval is invalid")
        for key in (
            "training_population_digest",
            "epoch_update_trace_digest",
        ):
            _require_hex(epoch[key], 64, key)
        epoch_validation_digest = _require_hex(
            epoch["validation_input_digest"], 64, "epoch validation input digest"
        )
        if epoch_validation_digest != validation_input_digest:
            _fail("epoch validation digest differs from the history binding")
        _require_float_hex(
            epoch["training_cross_entropy_float64_hex"], "training cross entropy"
        )
        validation_ce_hex, validation_ce = _require_float_hex(
            epoch["validation_cross_entropy_float64_hex"],
            "validation cross entropy",
        )
        confusion = _parse_confusion(epoch["validation_confusion_matrix"])
        numerator = _require_int(
            epoch["macro_f1_exact_numerator"], "macro-F1 numerator"
        )
        denominator = _require_int(
            epoch["macro_f1_exact_denominator"], "macro-F1 denominator", minimum=1
        )
        macro_f1 = Fraction(numerator, denominator)
        if (
            macro_f1.numerator != numerator
            or macro_f1.denominator != denominator
            or macro_f1 != _macro_f1(confusion)
        ):
            _fail("macro-F1 differs from its canonical confusion matrix")
        prediction_digest = _require_hex(
            epoch["validation_prediction_digest"], 64, "validation prediction digest"
        )
        model_digest = _require_hex(
            epoch["model_tensor_digest"], 64, "model tensor digest"
        )
        epochs.append(
            _ParsedEpoch(
                zero_based_epoch=expected_epoch,
                validation_confusion_matrix=confusion,
                validation_cross_entropy_float64_hex=validation_ce_hex,
                validation_cross_entropy=validation_ce,
                macro_f1=macro_f1,
                validation_prediction_digest=prediction_digest,
                model_tensor_digest=model_digest,
            )
        )
    epoch_tuple = tuple(epochs)
    winner = min(
        epoch_tuple,
        key=lambda item: (
            -item.macro_f1,
            item.validation_cross_entropy,
            item.zero_based_epoch,
        ),
    )
    payload = cast(bytes, raw)
    return _ParsedHistory(
        raw_bytes=payload,
        seed=seed,
        validation_input_digest=validation_input_digest,
        epochs=epoch_tuple,
        winner=winner,
        raw_sha256=hashlib.sha256(payload).hexdigest(),
        domain_sha256=hashlib.sha256(_HISTORY_DOMAIN + payload).hexdigest(),
    )


def _parse_safetensors(payload: object) -> tuple[bytes, str, str]:
    if type(payload) is not bytes:
        raise TypeError("safetensors payload must be exact bytes")
    raw = payload
    if len(raw) != _REGISTERED_SAFETENSORS_BYTES:
        _fail("safetensors payload has the wrong registered byte count")
    header_size = struct.unpack("<Q", raw[:8])[0]
    if header_size != _REGISTERED_SAFETENSORS_HEADER_BYTES:
        _fail("safetensors header has the wrong registered byte count")
    padded_header = raw[8 : 8 + header_size]
    header_bytes = padded_header.rstrip(b" ")
    padding = padded_header[len(header_bytes) :]
    if not header_bytes or len(padding) > 7 or padding != b" " * len(padding):
        _fail("safetensors header padding is invalid")
    try:
        parsed = json.loads(
            header_bytes.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_number,
            parse_float=_reject_json_number,
        )
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as error:
        raise Experiment006FinalEvidenceError(
            "safetensors header is not exact JSON"
        ) from error
    if type(parsed) is not dict or "__metadata__" in parsed:
        _fail("safetensors metadata must be absent")
    header = cast(dict[str, Any], parsed)
    expected_names = tuple(name for name, _shape in _REGISTERED_MODEL_SPECS)
    if tuple(header) != expected_names or len(header) != _REGISTERED_MODEL_TENSOR_COUNT:
        _fail("safetensors tensor names differ from the registered model")
    data = raw[8 + header_size :]
    if len(data) != _REGISTERED_MODEL_PAYLOAD_BYTES:
        _fail("safetensors model payload has the wrong byte count")
    expected_header: dict[str, object] = {}
    framed = bytearray(struct.pack("<I", _REGISTERED_MODEL_TENSOR_COUNT))
    expected_offset = 0
    value_count = 0
    for name, shape in _REGISTERED_MODEL_SPECS:
        record = _require_object(
            header[name],
            frozenset({"data_offsets", "dtype", "shape"}),
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
            _fail("safetensors tensor dtype, shape, or offsets changed")
        raw_tensor = data[expected_offset:expected_end]
        if len(raw_tensor) != byte_count or any(
            not math.isfinite(value)
            for (value,) in struct.iter_unpack("<f", raw_tensor)
        ):
            _fail("safetensors tensor payload is truncated or nonfinite")
        expected_header[name] = {
            "dtype": "F32",
            "shape": list(shape),
            "data_offsets": [expected_offset, expected_end],
        }
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
        expected_offset = expected_end
        value_count += math.prod(shape)
    expected_header_bytes = json.dumps(
        expected_header,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    expected_padded_header = expected_header_bytes + b" " * (
        -len(expected_header_bytes) % 8
    )
    if (
        expected_offset != len(data)
        or value_count != _REGISTERED_MODEL_VALUE_COUNT
        or expected_padded_header != padded_header
    ):
        _fail("safetensors canonical header or model layout changed")
    return (
        raw,
        hashlib.sha256(raw).hexdigest(),
        hashlib.sha256(_MODEL_TENSOR_DOMAIN + bytes(framed)).hexdigest(),
    )


def _parse_envelope(
    raw: object,
    *,
    role: ChildRole,
    ordinal: int,
    seed: int,
    registration: RegistrationEvidence,
    history: _ParsedHistory,
    safetensors_sha256: str,
    model_tensor_sha256: str,
) -> tuple[int, str]:
    document = _parse_canonical_object(
        raw, _MAX_ENVELOPE_BYTES, "child result envelope"
    )
    _require_exact_keys(
        document,
        frozenset(
            {
                "experiment",
                "history",
                "ordinal",
                "registration",
                "role",
                "safetensors",
                "schema_version",
                "seed",
            }
        ),
        "child result envelope",
    )
    if _require_int(document["schema_version"], "envelope schema_version") != 1:
        _fail("envelope schema_version must be integer one")
    _require_string(
        document["experiment"],
        _SCIENTIFIC_PROTOCOL,
        "envelope scientific protocol",
    )
    _require_string(document["role"], role, "envelope role")
    if (
        _require_int(document["ordinal"], "envelope ordinal") != ordinal
        or _require_int(document["seed"], "envelope seed") != seed
    ):
        _fail("envelope child binding changed")
    registration_document = _require_object(
        document["registration"],
        frozenset(
            {
                "head_commit",
                "implementation_commit",
                "registration_sha256",
                "source_bundle_sha256",
            }
        ),
        "envelope registration",
    )
    if registration_document != {
        "head_commit": registration.head_commit,
        "implementation_commit": registration.implementation_commit,
        "registration_sha256": registration.registration_sha256,
        "source_bundle_sha256": registration.source_bundle_sha256,
    }:
        _fail("envelope registration binding changed")
    history_document = _require_object(
        document["history"],
        frozenset(
            {
                "byte_count",
                "domain_sha256",
                "filename",
                "model_tensor_sha256",
                "winner_epoch",
            }
        ),
        "envelope history",
    )
    safetensors_document = _require_object(
        document["safetensors"],
        frozenset({"byte_count", "filename", "sha256"}),
        "envelope safetensors",
    )
    if history_document != {
        "byte_count": len(history.raw_bytes),
        "domain_sha256": history.domain_sha256,
        "filename": _HISTORY_FILENAME,
        "model_tensor_sha256": model_tensor_sha256,
        "winner_epoch": history.winner.zero_based_epoch,
    } or safetensors_document != {
        "byte_count": _REGISTERED_SAFETENSORS_BYTES,
        "filename": _SAFETENSORS_FILENAME,
        "sha256": safetensors_sha256,
    }:
        _fail("envelope payload binding changed")
    payload = cast(bytes, raw)
    return len(payload), hashlib.sha256(_ENVELOPE_DOMAIN + payload).hexdigest()


def _parse_registration(fields: tuple[object, ...]) -> RegistrationEvidence:
    head = _require_hex(fields[3], 40, "registration head commit")
    implementation = _require_hex(fields[4], 40, "implementation commit")
    if head == implementation:
        _fail("registration and implementation commits must differ")
    return RegistrationEvidence(
        head_commit=head,
        implementation_commit=implementation,
        registration_sha256=_require_hex(fields[5], 64, "run registration SHA-256"),
        source_bundle_sha256=_require_hex(fields[6], 64, "source bundle SHA-256"),
    )


def _parse_snapshot(value: object) -> _ParsedSnapshot:
    if type(value) is not tuple or len(cast(tuple[object, ...], value)) != 21:
        _fail("child result snapshot has an invalid exact shape")
    fields = cast(tuple[object, ...], value)
    if type(fields[0]) is not str or fields[0] not in {_SEED_ROLE, _RERUN_ROLE}:
        _fail("child result snapshot role is not registered")
    role = cast(ChildRole, fields[0])
    ordinal = _require_int(fields[1], "child ordinal")
    seed = _require_int(fields[2], "child seed")
    if seed not in _REGISTERED_SEEDS:
        _fail("child seed is not registered")
    registration = _parse_registration(fields)
    history = _parse_history(fields[7], seed)
    if (
        _require_int(fields[8], "history byte count", minimum=1)
        != len(history.raw_bytes)
        or _require_hex(fields[9], 64, "history domain SHA-256")
        != history.domain_sha256
    ):
        _fail("snapshot history bytes or digest changed")
    safetensors_bytes, safetensors_sha256, model_tensor_sha256 = _parse_safetensors(
        fields[10]
    )
    if (
        _require_int(fields[11], "safetensors byte count", minimum=1)
        != len(safetensors_bytes)
        or _require_hex(fields[12], 64, "safetensors SHA-256") != safetensors_sha256
    ):
        _fail("snapshot safetensors bytes or digest changed")
    envelope_count, envelope_sha256 = _parse_envelope(
        fields[13],
        role=role,
        ordinal=ordinal,
        seed=seed,
        registration=registration,
        history=history,
        safetensors_sha256=safetensors_sha256,
        model_tensor_sha256=model_tensor_sha256,
    )
    if (
        _require_int(fields[14], "envelope byte count", minimum=1) != envelope_count
        or _require_hex(fields[15], 64, "envelope domain SHA-256") != envelope_sha256
    ):
        _fail("snapshot envelope bytes or digest changed")
    winner_epoch = _require_int(fields[16], "winner epoch")
    winner_numerator = _require_int(fields[17], "winner macro-F1 numerator")
    winner_denominator = _require_int(
        fields[18], "winner macro-F1 denominator", minimum=1
    )
    winner_ce_hex, _winner_ce = _require_float_hex(
        fields[19], "winner validation cross entropy"
    )
    snapshot_model_digest = _require_hex(fields[20], 64, "winner model tensor SHA-256")
    winner = history.winner
    if (
        winner_epoch != winner.zero_based_epoch
        or winner_numerator != winner.macro_f1.numerator
        or winner_denominator != winner.macro_f1.denominator
        or winner_ce_hex != winner.validation_cross_entropy_float64_hex
        or snapshot_model_digest != winner.model_tensor_digest
        or snapshot_model_digest != model_tensor_sha256
    ):
        _fail("snapshot winner differs from independently parsed payloads")
    return _ParsedSnapshot(
        role=role,
        ordinal=ordinal,
        seed=seed,
        registration=registration,
        history=history,
        safetensors_bytes=safetensors_bytes,
        safetensors_sha256=safetensors_sha256,
        envelope_byte_count=envelope_count,
        envelope_domain_sha256=envelope_sha256,
        winner_epoch=winner_epoch,
        winner_macro_f1_numerator=winner_numerator,
        winner_macro_f1_denominator=winner_denominator,
        winner_validation_cross_entropy_float64_hex=winner_ce_hex,
        model_tensor_sha256=snapshot_model_digest,
    )


def _parse_child(value: object) -> _ParsedChild:
    if type(value) is not tuple or len(cast(tuple[object, ...], value)) != 6:
        _fail("parent child result has an invalid exact shape")
    fields = cast(tuple[object, ...], value)
    pid = _require_int(fields[0], "child PID", minimum=1)
    if pid > _UINT32_MAXIMUM:
        _fail("child PID exceeds the unsigned 32-bit evidence ceiling")
    if pid == os.getpid():
        _fail("child PID equals the evidence parent process")
    if type(fields[1]) is not tuple or len(cast(tuple[object, ...], fields[1])) != 2:
        _fail("child CPU IDs have an invalid exact shape")
    cpu_fields = cast(tuple[object, ...], fields[1])
    cpu_ids = (
        _require_int(cpu_fields[0], "first child CPU ID"),
        _require_int(cpu_fields[1], "second child CPU ID"),
    )
    if cpu_ids[0] > _UINT32_MAXIMUM or cpu_ids[1] > _UINT32_MAXIMUM:
        _fail("child CPU ID exceeds the unsigned 32-bit evidence ceiling")
    if cpu_ids[0] >= cpu_ids[1]:
        _fail("child CPU IDs are not strictly increasing")
    elapsed = _require_int(fields[2], "child elapsed nanoseconds")
    maximum_rss = _require_int(fields[3], "child maximum RSS bytes")
    output_bytes = _require_int(fields[4], "child output and scratch bytes")
    if elapsed > _WALL_NANOSECONDS_MAXIMUM:
        _fail("child exceeded the registered wall-time budget")
    if maximum_rss > _RSS_BYTES_MAXIMUM:
        _fail("child exceeded the registered RSS budget")
    if output_bytes > _OUTPUT_BYTES_MAXIMUM:
        _fail("child exceeded the registered output budget")
    snapshot = _parse_snapshot(fields[5])
    history = snapshot.history
    evidence = ChildEvidence(
        role=snapshot.role,
        ordinal=snapshot.ordinal,
        seed=snapshot.seed,
        pid=pid,
        cpu_ids=cpu_ids,
        elapsed_nanoseconds=elapsed,
        maximum_rss_bytes=maximum_rss,
        output_and_scratch_bytes=output_bytes,
        history_byte_count=len(history.raw_bytes),
        history_sha256=history.raw_sha256,
        history_domain_sha256=history.domain_sha256,
        safetensors_byte_count=len(snapshot.safetensors_bytes),
        safetensors_sha256=snapshot.safetensors_sha256,
        envelope_byte_count=snapshot.envelope_byte_count,
        envelope_domain_sha256=snapshot.envelope_domain_sha256,
        winner_epoch=snapshot.winner_epoch,
        winner_macro_f1_numerator=snapshot.winner_macro_f1_numerator,
        winner_macro_f1_denominator=snapshot.winner_macro_f1_denominator,
        winner_validation_cross_entropy_float64_hex=(
            snapshot.winner_validation_cross_entropy_float64_hex
        ),
        model_tensor_sha256=snapshot.model_tensor_sha256,
    )
    return _ParsedChild(evidence=evidence, snapshot=snapshot)


def _rank_key(child: _ParsedChild) -> tuple[Fraction, float, int, int]:
    winner = child.snapshot.history.winner
    return (
        -winner.macro_f1,
        winner.validation_cross_entropy,
        winner.zero_based_epoch,
        child.snapshot.seed,
    )


def _require_rerun_equality(
    selected: _ParsedChild, rerun: _ParsedChild
) -> RerunEvidence:
    selected_snapshot = selected.snapshot
    rerun_snapshot = rerun.snapshot
    selected_history = selected_snapshot.history
    rerun_history = rerun_snapshot.history
    history_equal = selected_history.raw_bytes == rerun_history.raw_bytes
    safetensors_equal = (
        selected_snapshot.safetensors_bytes == rerun_snapshot.safetensors_bytes
        and selected_snapshot.safetensors_sha256 == rerun_snapshot.safetensors_sha256
    )
    validation_input_equal = (
        selected_history.validation_input_digest
        == rerun_history.validation_input_digest
    )
    predictions_equal = tuple(
        epoch.validation_prediction_digest for epoch in selected_history.epochs
    ) == tuple(epoch.validation_prediction_digest for epoch in rerun_history.epochs)
    model_digests_equal = tuple(
        epoch.model_tensor_digest for epoch in selected_history.epochs
    ) == tuple(epoch.model_tensor_digest for epoch in rerun_history.epochs)
    checkpoint_equal = (
        selected_snapshot.model_tensor_sha256 == rerun_snapshot.model_tensor_sha256
        and selected_history.winner.model_tensor_digest
        == rerun_history.winner.model_tensor_digest
    )
    rank_equal = (
        selected_snapshot.winner_epoch == rerun_snapshot.winner_epoch
        and selected_snapshot.winner_macro_f1_numerator
        == rerun_snapshot.winner_macro_f1_numerator
        and selected_snapshot.winner_macro_f1_denominator
        == rerun_snapshot.winner_macro_f1_denominator
        and selected_snapshot.winner_validation_cross_entropy_float64_hex
        == rerun_snapshot.winner_validation_cross_entropy_float64_hex
    )
    if not all(
        (
            history_equal,
            safetensors_equal,
            validation_input_equal,
            predictions_equal,
            model_digests_equal,
            checkpoint_equal,
            rank_equal,
        )
    ):
        _fail("selected-seed rerun differs from its registered training result")
    return RerunEvidence(
        selected_training_ordinal=selected_snapshot.ordinal,
        rerun_ordinal=rerun_snapshot.ordinal,
        selected_seed=selected_snapshot.seed,
        history_byte_identical=True,
        safetensors_byte_identical=True,
        validation_input_digest_identical=True,
        all_validation_prediction_digests_identical=True,
        all_model_tensor_digests_identical=True,
        selected_checkpoint_digest_identical=True,
        selected_rank_identical=True,
    )


def _target_accuracy_gate(correct: int, total: int, /) -> bool:
    return 20 * correct >= 17 * total


def _target_recall_gate(correct: int, support: int, /) -> bool:
    return 10 * correct >= 7 * support


def _macro_f1_gate(value: Fraction, /) -> bool:
    return value >= Fraction(4, 5)


def _unknown_target_rate_gate(target_predictions: int, total: int, /) -> bool:
    return 5 * target_predictions <= total


def _silence_target_rate_gate(target_predictions: int, total: int, /) -> bool:
    return 20 * target_predictions <= total


def _compute_gates(history: _ParsedHistory) -> GatesEvidence:
    winner = history.winner
    confusion = winner.validation_confusion_matrix
    target_accuracy_correct = sum(confusion[index][index] for index in range(10))
    target_accuracy_minimum = (17 * _TARGET_POPULATION + 19) // 20
    target_accuracy_passed = _target_accuracy_gate(
        target_accuracy_correct, _TARGET_POPULATION
    )
    recall_values: list[ClassRecallEvidence] = []
    for class_index in range(_TARGET_COUNT):
        support = _CLASS_SUPPORT[class_index]
        correct = confusion[class_index][class_index]
        minimum = (7 * support + 9) // 10
        recall_values.append(
            ClassRecallEvidence(
                class_index=class_index,
                class_name=_CLASS_ORDER[class_index],
                correct=correct,
                support=support,
                minimum_correct=minimum,
                passed=_target_recall_gate(correct, support),
            )
        )
    target_recalls = cast(
        "tuple[ClassRecallEvidence, ClassRecallEvidence, ClassRecallEvidence, "
        "ClassRecallEvidence, ClassRecallEvidence, ClassRecallEvidence, "
        "ClassRecallEvidence, ClassRecallEvidence, ClassRecallEvidence, "
        "ClassRecallEvidence]",
        tuple(recall_values),
    )
    macro_passed = _macro_f1_gate(winner.macro_f1)
    unknown_target_predictions = sum(confusion[10][:_TARGET_COUNT])
    silence_target_predictions = sum(confusion[11][:_TARGET_COUNT])
    unknown_passed = _unknown_target_rate_gate(
        unknown_target_predictions, _CLASS_SUPPORT[10]
    )
    silence_passed = _silence_target_rate_gate(
        silence_target_predictions, _CLASS_SUPPORT[11]
    )
    all_passed = all(
        (
            target_accuracy_passed,
            all(item.passed for item in target_recalls),
            macro_passed,
            unknown_passed,
            silence_passed,
        )
    )
    return GatesEvidence(
        target_accuracy_correct=target_accuracy_correct,
        target_accuracy_total=_TARGET_POPULATION,
        target_accuracy_minimum_correct=target_accuracy_minimum,
        target_accuracy_passed=target_accuracy_passed,
        target_recalls=target_recalls,
        macro_f1_numerator=winner.macro_f1.numerator,
        macro_f1_denominator=winner.macro_f1.denominator,
        macro_f1_minimum_numerator=4,
        macro_f1_minimum_denominator=5,
        macro_f1_passed=macro_passed,
        unknown_target_predictions=unknown_target_predictions,
        unknown_total=_CLASS_SUPPORT[10],
        unknown_maximum_target_predictions=_CLASS_SUPPORT[10] // 5,
        unknown_target_rate_passed=unknown_passed,
        silence_target_predictions=silence_target_predictions,
        silence_total=_CLASS_SUPPORT[11],
        silence_maximum_target_predictions=_CLASS_SUPPORT[11] // 20,
        silence_target_rate_passed=silence_passed,
        finite_contract_passed=True,
        all_passed=all_passed,
    )


def _history_artifact(
    child: _ParsedChild,
    path: str,
    kind: Literal["training_history", "selected_rerun_history"],
) -> ArtifactEvidence:
    history = child.snapshot.history
    return ArtifactEvidence(
        path=path,
        kind=kind,
        mode=_ARTIFACT_MODE,
        contents=history.raw_bytes,
        byte_count=len(history.raw_bytes),
        sha256=history.raw_sha256,
        domain_sha256=history.domain_sha256,
        model_tensor_sha256=None,
        source_child_ordinal=child.snapshot.ordinal,
    )


def _build_completed_fields(selector_result: object) -> _CompletedFields:
    if (
        type(selector_result) is not tuple
        or len(cast(tuple[object, ...], selector_result)) != 3
    ):
        _fail("registered selector result has an invalid exact shape")
    selector_fields = cast(tuple[object, ...], selector_result)
    raw_training = selector_fields[0]
    if (
        type(raw_training) is not tuple
        or len(cast(tuple[object, ...], raw_training)) != 3
    ):
        _fail("registered selector training results have an invalid exact shape")
    if type(selector_fields[1]) is not int:
        _fail("registered selected ordinal must be an exact integer")
    claimed_selected_ordinal = selector_fields[1]
    training = tuple(
        _parse_child(value) for value in cast(tuple[object, ...], raw_training)
    )
    rerun = _parse_child(selector_fields[2])
    all_children = (*training, rerun)
    cpu_ids = training[0].evidence.cpu_ids
    if any(child.evidence.cpu_ids != cpu_ids for child in all_children):
        _fail("registered children did not use the same captured CPU IDs")
    registration = training[0].snapshot.registration
    if any(child.snapshot.registration != registration for child in all_children):
        _fail("registered child source bindings differ")
    for ordinal, (child, seed) in enumerate(
        zip(training, _REGISTERED_SEEDS, strict=True)
    ):
        if (child.snapshot.role, child.snapshot.ordinal, child.snapshot.seed) != (
            _SEED_ROLE,
            ordinal,
            seed,
        ):
            _fail("training children differ from registered role/ordinal/seed order")
    ranking = tuple(sorted(range(3), key=lambda ordinal: _rank_key(training[ordinal])))
    selected_ordinal = ranking[0]
    if claimed_selected_ordinal != selected_ordinal:
        _fail("selector selected ordinal differs from the independent seed rank")
    selected = training[selected_ordinal]
    if (
        rerun.snapshot.role,
        rerun.snapshot.ordinal,
        rerun.snapshot.seed,
    ) != (_RERUN_ROLE, 3, selected.snapshot.seed):
        _fail("rerun differs from the selected registered role/ordinal/seed")
    rerun_evidence = _require_rerun_equality(selected, rerun)
    selected_snapshot = selected.snapshot
    selection = SelectionEvidence(
        selected_training_ordinal=selected_ordinal,
        selected_seed=selected_snapshot.seed,
        selected_winner_epoch=selected_snapshot.winner_epoch,
        selected_macro_f1_numerator=selected_snapshot.winner_macro_f1_numerator,
        selected_macro_f1_denominator=selected_snapshot.winner_macro_f1_denominator,
        selected_validation_cross_entropy_float64_hex=(
            selected_snapshot.winner_validation_cross_entropy_float64_hex
        ),
        selected_model_tensor_sha256=selected_snapshot.model_tensor_sha256,
        ranked_training_ordinals=cast(tuple[int, int, int], ranking),
    )
    gates = _compute_gates(rerun.snapshot.history)
    status: CompletedStatus = "pass" if gates.all_passed else "gate_failure"
    artifacts: list[ArtifactEvidence] = [
        _history_artifact(
            training[index], TRAINING_HISTORY_PATHS[index], "training_history"
        )
        for index in range(3)
    ]
    artifacts.append(
        _history_artifact(
            rerun,
            SELECTED_RERUN_HISTORY_PATH,
            "selected_rerun_history",
        )
    )
    if status == "pass":
        artifacts.append(
            ArtifactEvidence(
                path=SELECTED_SAFETENSORS_PATH,
                kind="selected_safetensors",
                mode=_ARTIFACT_MODE,
                contents=rerun.snapshot.safetensors_bytes,
                byte_count=len(rerun.snapshot.safetensors_bytes),
                sha256=rerun.snapshot.safetensors_sha256,
                domain_sha256=None,
                model_tensor_sha256=rerun.snapshot.model_tensor_sha256,
                source_child_ordinal=rerun.snapshot.ordinal,
            )
        )
    return _CompletedFields(
        schema_version=_SCHEMA_VERSION,
        experiment=_EXPERIMENT,
        status=status,
        registration=registration,
        children=cast(
            "tuple[ChildEvidence, ChildEvidence, ChildEvidence, ChildEvidence]",
            tuple(child.evidence for child in all_children),
        ),
        selection=selection,
        rerun=rerun_evidence,
        gates=gates,
        artifacts=tuple(artifacts),
        report_path=FINAL_REPORT_PATH,
        report_mode=_ARTIFACT_MODE,
    )


def _artifact_document(artifact: ArtifactEvidence) -> dict[str, object]:
    document: dict[str, object] = {
        "byte_count": artifact.byte_count,
        "kind": artifact.kind,
        "mode": artifact.mode,
        "path": artifact.path,
        "sha256": artifact.sha256,
        "source_child_ordinal": artifact.source_child_ordinal,
    }
    if artifact.domain_sha256 is not None:
        document["domain_sha256"] = artifact.domain_sha256
    if artifact.model_tensor_sha256 is not None:
        document["model_tensor_sha256"] = artifact.model_tensor_sha256
        document["tensor_count"] = _REGISTERED_MODEL_TENSOR_COUNT
        document["value_count"] = _REGISTERED_MODEL_VALUE_COUNT
    return document


def _child_document(child: ChildEvidence) -> dict[str, object]:
    return {
        "binding": {
            "ordinal": child.ordinal,
            "role": child.role,
            "seed": child.seed,
        },
        "envelope": {
            "byte_count": child.envelope_byte_count,
            "domain_sha256": child.envelope_domain_sha256,
        },
        "history": {
            "byte_count": child.history_byte_count,
            "domain_sha256": child.history_domain_sha256,
            "sha256": child.history_sha256,
        },
        "resources": {
            "cpu_ids": list(child.cpu_ids),
            "elapsed_nanoseconds": child.elapsed_nanoseconds,
            "maximum_rss_bytes": child.maximum_rss_bytes,
            "output_and_scratch_bytes": child.output_and_scratch_bytes,
            "pid": child.pid,
        },
        "safetensors": {
            "byte_count": child.safetensors_byte_count,
            "sha256": child.safetensors_sha256,
        },
        "winner": {
            "macro_f1_exact_denominator": child.winner_macro_f1_denominator,
            "macro_f1_exact_numerator": child.winner_macro_f1_numerator,
            "model_tensor_sha256": child.model_tensor_sha256,
            "validation_cross_entropy_float64_hex": (
                child.winner_validation_cross_entropy_float64_hex
            ),
            "zero_based_epoch": child.winner_epoch,
        },
    }


def _gates_document(gates: GatesEvidence) -> dict[str, object]:
    return {
        "all_passed": gates.all_passed,
        "finite_contract_passed": gates.finite_contract_passed,
        "macro_f1": {
            "denominator": gates.macro_f1_denominator,
            "minimum_denominator": gates.macro_f1_minimum_denominator,
            "minimum_numerator": gates.macro_f1_minimum_numerator,
            "numerator": gates.macro_f1_numerator,
            "passed": gates.macro_f1_passed,
        },
        "silence_target_rate": {
            "maximum_target_predictions": gates.silence_maximum_target_predictions,
            "passed": gates.silence_target_rate_passed,
            "target_predictions": gates.silence_target_predictions,
            "total": gates.silence_total,
        },
        "target_accuracy": {
            "correct": gates.target_accuracy_correct,
            "minimum_correct": gates.target_accuracy_minimum_correct,
            "passed": gates.target_accuracy_passed,
            "total": gates.target_accuracy_total,
        },
        "target_recalls": [
            {
                "class_index": item.class_index,
                "class_name": item.class_name,
                "correct": item.correct,
                "minimum_correct": item.minimum_correct,
                "passed": item.passed,
                "support": item.support,
            }
            for item in gates.target_recalls
        ],
        "unknown_target_rate": {
            "maximum_target_predictions": gates.unknown_maximum_target_predictions,
            "passed": gates.unknown_target_rate_passed,
            "target_predictions": gates.unknown_target_predictions,
            "total": gates.unknown_total,
        },
    }


def _predecessor_document() -> dict[str, object]:
    return cast(
        dict[str, object],
        json.loads(_PREDECESSOR_CANONICAL_JSON),
    )


def _require_publication_budgets(
    report_bytes: object,
    payload_byte_counts: object,
    report_maximum: object,
    aggregate_maximum: object,
    /,
) -> None:
    """Enforce publication byte ceilings; explicit limits form a boundary-test seam."""

    if type(report_bytes) is not bytes:
        raise TypeError("completed report must be exact bytes")
    if (
        type(payload_byte_counts) is not tuple
        or any(type(value) is not int or value < 0 for value in payload_byte_counts)
        or type(report_maximum) is not int
        or report_maximum < 0
        or type(aggregate_maximum) is not int
        or aggregate_maximum < 0
    ):
        raise TypeError("publication byte budgets require exact nonnegative integers")
    report_byte_count = len(report_bytes)
    if report_byte_count > report_maximum:
        _fail("completed report exceeds the publication report byte ceiling")
    if report_byte_count + sum(payload_byte_counts) > aggregate_maximum:
        _fail("completed payloads and report exceed the publication byte ceiling")


def _completed_report_bytes(fields: _CompletedFields) -> bytes:
    selection = fields.selection
    rerun = fields.rerun
    registration = fields.registration
    document: dict[str, object] = {
        "artifacts": [_artifact_document(artifact) for artifact in fields.artifacts],
        "checkpoint_reusable": fields.status == "pass",
        "children": [_child_document(child) for child in fields.children],
        "experiment": fields.experiment,
        "failure": None,
        "gates": _gates_document(fields.gates),
        "predecessor": _predecessor_document(),
        "publication": {
            "report_mode": fields.report_mode,
            "report_path": fields.report_path,
        },
        "registration": {
            "head_commit": registration.head_commit,
            "implementation_commit": registration.implementation_commit,
            "path": "configs/experiment-006-run.json",
            "registration_sha256": registration.registration_sha256,
            "source_bundle_sha256": registration.source_bundle_sha256,
        },
        "rerun": {
            "all_model_tensor_digests_identical": (
                rerun.all_model_tensor_digests_identical
            ),
            "all_validation_prediction_digests_identical": (
                rerun.all_validation_prediction_digests_identical
            ),
            "history_byte_identical": rerun.history_byte_identical,
            "rerun_ordinal": rerun.rerun_ordinal,
            "safetensors_byte_identical": rerun.safetensors_byte_identical,
            "selected_checkpoint_digest_identical": (
                rerun.selected_checkpoint_digest_identical
            ),
            "selected_rank_identical": rerun.selected_rank_identical,
            "selected_seed": rerun.selected_seed,
            "selected_training_ordinal": rerun.selected_training_ordinal,
            "validation_input_digest_identical": (
                rerun.validation_input_digest_identical
            ),
        },
        "schema_version": fields.schema_version,
        "scientific_protocol": _SCIENTIFIC_PROTOCOL,
        "selection": {
            "ranked_training_ordinals": list(selection.ranked_training_ordinals),
            "selected_macro_f1_denominator": (selection.selected_macro_f1_denominator),
            "selected_macro_f1_numerator": selection.selected_macro_f1_numerator,
            "selected_model_tensor_sha256": selection.selected_model_tensor_sha256,
            "selected_seed": selection.selected_seed,
            "selected_training_ordinal": selection.selected_training_ordinal,
            "selected_validation_cross_entropy_float64_hex": (
                selection.selected_validation_cross_entropy_float64_hex
            ),
            "selected_winner_epoch": selection.selected_winner_epoch,
        },
        "status": fields.status,
    }
    return _canonical_json_bytes(document)


def _make_completed_evidence_routes() -> tuple[
    _BuildRoute, _VerifyRoute, _ReportRoute, _SnapshotRoute
]:
    """Close exact descriptor and primitive-ledger authority over public routes."""

    exact_type = type
    length = len
    make_tuple = tuple
    make_set = set
    sort_values = sorted
    get_attribute = getattr
    any_values = any
    zip_values = zip
    object_new = object.__new__
    sha256_route = hashlib.sha256
    json_encoder_type = json.JSONEncoder
    json_decoder_type = json.JSONDecoder
    cast_route = cast
    bytes_type = bytes
    bool_type = bool
    int_type = int
    none_type = type(None)
    str_type = str
    tuple_type = tuple
    code_type = CodeType
    function_type = FunctionType
    member_descriptor_type = MemberDescriptorType
    completed_status_alias = CompletedStatus
    registration_frame_alias = RegistrationPublicationFrame
    completed_publication_alias = CompletedPublicationSnapshot
    evidence_type = FinalCompletedEvidence
    registration_type = RegistrationEvidence
    child_type = ChildEvidence
    selection_type = SelectionEvidence
    rerun_type = RerunEvidence
    recall_type = ClassRecallEvidence
    gates_type = GatesEvidence
    artifact_type = ArtifactEvidence
    parsed_epoch_type = _ParsedEpoch
    parsed_history_type = _ParsedHistory
    parsed_snapshot_type = _ParsedSnapshot
    parsed_child_type = _ParsedChild
    completed_fields_type = _CompletedFields
    verified_registration_type = VerifiedRunRegistration
    run_authority_module = _run_authority
    verified_state_route = _run_authority._verified_state
    verified_state_type = _run_authority._VerifiedState
    error_type = Experiment006FinalEvidenceError
    build_fields = _build_completed_fields
    report_builder = _completed_report_bytes
    budget_checker = _require_publication_budgets
    report_maximum = FINAL_REPORT_BYTES_MAXIMUM
    aggregate_maximum = COMPLETED_PUBLICATION_BYTES_MAXIMUM
    fixed_report_path = FINAL_REPORT_PATH
    fixed_report_mode = _ARTIFACT_MODE
    module_globals = globals()
    missing = object()
    empty_cell = object()
    verified_state_field_names = (
        "head_commit",
        "implementation_commit",
        "registration_sha256",
        "source_bundle_sha256",
    )
    verified_state_descriptors = make_tuple(
        verified_state_type.__dict__[name] for name in verified_state_field_names
    )
    if any_values(
        exact_type(descriptor) is not member_descriptor_type
        for descriptor in verified_state_descriptors
    ):
        raise RuntimeError("run-registration state descriptors are unavailable")
    verified_state_class_items = make_tuple(verified_state_type.__dict__.items())
    lock = threading.RLock()
    issued: weakref.WeakKeyDictionary[FinalCompletedEvidence, _IssuedLedger] = (
        weakref.WeakKeyDictionary()
    )

    def make_descriptor_authority(
        global_name: str,
        evidence_class: type[Any],
        field_names: tuple[str, ...],
        /,
    ) -> tuple[
        str,
        type[Any],
        tuple[str, ...],
        tuple[MemberDescriptorType, ...],
        tuple[tuple[str, object], ...],
    ]:
        descriptors = make_tuple(evidence_class.__dict__[name] for name in field_names)
        if any_values(
            exact_type(descriptor) is not member_descriptor_type
            for descriptor in descriptors
        ):
            raise RuntimeError(f"{global_name} field descriptors are unavailable")
        return (
            global_name,
            evidence_class,
            field_names,
            cast_route("tuple[MemberDescriptorType, ...]", descriptors),
            make_tuple(evidence_class.__dict__.items()),
        )

    registration_authority = make_descriptor_authority(
        "RegistrationEvidence",
        registration_type,
        (
            "head_commit",
            "implementation_commit",
            "registration_sha256",
            "source_bundle_sha256",
        ),
    )
    child_authority = make_descriptor_authority(
        "ChildEvidence",
        child_type,
        (
            "role",
            "ordinal",
            "seed",
            "pid",
            "cpu_ids",
            "elapsed_nanoseconds",
            "maximum_rss_bytes",
            "output_and_scratch_bytes",
            "history_byte_count",
            "history_sha256",
            "history_domain_sha256",
            "safetensors_byte_count",
            "safetensors_sha256",
            "envelope_byte_count",
            "envelope_domain_sha256",
            "winner_epoch",
            "winner_macro_f1_numerator",
            "winner_macro_f1_denominator",
            "winner_validation_cross_entropy_float64_hex",
            "model_tensor_sha256",
        ),
    )
    selection_authority = make_descriptor_authority(
        "SelectionEvidence",
        selection_type,
        (
            "selected_training_ordinal",
            "selected_seed",
            "selected_winner_epoch",
            "selected_macro_f1_numerator",
            "selected_macro_f1_denominator",
            "selected_validation_cross_entropy_float64_hex",
            "selected_model_tensor_sha256",
            "ranked_training_ordinals",
        ),
    )
    rerun_authority = make_descriptor_authority(
        "RerunEvidence",
        rerun_type,
        (
            "selected_training_ordinal",
            "rerun_ordinal",
            "selected_seed",
            "history_byte_identical",
            "safetensors_byte_identical",
            "validation_input_digest_identical",
            "all_validation_prediction_digests_identical",
            "all_model_tensor_digests_identical",
            "selected_checkpoint_digest_identical",
            "selected_rank_identical",
        ),
    )
    recall_authority = make_descriptor_authority(
        "ClassRecallEvidence",
        recall_type,
        (
            "class_index",
            "class_name",
            "correct",
            "support",
            "minimum_correct",
            "passed",
        ),
    )
    gates_authority = make_descriptor_authority(
        "GatesEvidence",
        gates_type,
        (
            "target_accuracy_correct",
            "target_accuracy_total",
            "target_accuracy_minimum_correct",
            "target_accuracy_passed",
            "target_recalls",
            "macro_f1_numerator",
            "macro_f1_denominator",
            "macro_f1_minimum_numerator",
            "macro_f1_minimum_denominator",
            "macro_f1_passed",
            "unknown_target_predictions",
            "unknown_total",
            "unknown_maximum_target_predictions",
            "unknown_target_rate_passed",
            "silence_target_predictions",
            "silence_total",
            "silence_maximum_target_predictions",
            "silence_target_rate_passed",
            "finite_contract_passed",
            "all_passed",
        ),
    )
    artifact_authority = make_descriptor_authority(
        "ArtifactEvidence",
        artifact_type,
        (
            "path",
            "kind",
            "mode",
            "contents",
            "byte_count",
            "sha256",
            "domain_sha256",
            "model_tensor_sha256",
            "source_child_ordinal",
        ),
    )
    final_authority = make_descriptor_authority(
        "FinalCompletedEvidence",
        evidence_type,
        (
            "schema_version",
            "experiment",
            "status",
            "registration",
            "children",
            "selection",
            "rerun",
            "gates",
            "artifacts",
            "report_path",
            "report_mode",
        ),
    )
    parsed_epoch_authority = make_descriptor_authority(
        "_ParsedEpoch",
        parsed_epoch_type,
        (
            "zero_based_epoch",
            "validation_confusion_matrix",
            "validation_cross_entropy_float64_hex",
            "validation_cross_entropy",
            "macro_f1",
            "validation_prediction_digest",
            "model_tensor_digest",
        ),
    )
    parsed_history_authority = make_descriptor_authority(
        "_ParsedHistory",
        parsed_history_type,
        (
            "raw_bytes",
            "seed",
            "validation_input_digest",
            "epochs",
            "winner",
            "raw_sha256",
            "domain_sha256",
        ),
    )
    parsed_snapshot_authority = make_descriptor_authority(
        "_ParsedSnapshot",
        parsed_snapshot_type,
        (
            "role",
            "ordinal",
            "seed",
            "registration",
            "history",
            "safetensors_bytes",
            "safetensors_sha256",
            "envelope_byte_count",
            "envelope_domain_sha256",
            "winner_epoch",
            "winner_macro_f1_numerator",
            "winner_macro_f1_denominator",
            "winner_validation_cross_entropy_float64_hex",
            "model_tensor_sha256",
        ),
    )
    parsed_child_authority = make_descriptor_authority(
        "_ParsedChild",
        parsed_child_type,
        ("evidence", "snapshot"),
    )
    completed_fields_authority = make_descriptor_authority(
        "_CompletedFields",
        completed_fields_type,
        (
            "schema_version",
            "experiment",
            "status",
            "registration",
            "children",
            "selection",
            "rerun",
            "gates",
            "artifacts",
            "report_path",
            "report_mode",
        ),
    )
    descriptor_authorities = (
        registration_authority,
        child_authority,
        selection_authority,
        rerun_authority,
        recall_authority,
        gates_authority,
        artifact_authority,
        final_authority,
        parsed_epoch_authority,
        parsed_history_authority,
        parsed_snapshot_authority,
        parsed_child_authority,
        completed_fields_authority,
    )

    def capture_function_authority(
        name: str, function: FunctionType, /
    ) -> _FunctionAuthority:
        closure = function.__closure__
        cells: tuple[object, ...] = () if closure is None else make_tuple(closure)
        contents: list[object] = []
        for cell in cells:
            try:
                contents.append(cell.cell_contents)  # type: ignore[attr-defined]
            except ValueError:
                contents.append(empty_cell)
        return (
            name,
            function,
            function.__code__,
            function.__defaults__,
            function.__kwdefaults__,
            cells,
            make_tuple(contents),
        )

    def capture_json_method_authority(
        name: str, function: FunctionType, /
    ) -> _JSONMethodAuthority:
        keyword_defaults = function.__kwdefaults__
        return (
            capture_function_authority(name, function),
            make_tuple(function.__dict__.items()),
            function.__globals__,
            make_tuple(function.__globals__.items()),
            (
                None
                if keyword_defaults is None
                else make_tuple(keyword_defaults.items())
            ),
        )

    def capture_json_class_authority(
        attribute_name: str, json_class: type[Any], /
    ) -> _JSONClassAuthority:
        class_items = make_tuple(json_class.__dict__.items())
        methods = make_tuple(
            capture_json_method_authority(
                f"json.{attribute_name}.{name}",
                cast_route(function_type, value),
            )
            for name, value in class_items
            if exact_type(value) is function_type
        )
        return attribute_name, json_class, class_items, methods

    json_class_authorities = (
        capture_json_class_authority("JSONEncoder", json_encoder_type),
        capture_json_class_authority("JSONDecoder", json_decoder_type),
    )

    helper_functions = make_tuple(
        (name, value)
        for name, value in module_globals.items()
        if name.startswith("_")
        and name != "_make_completed_evidence_routes"
        and exact_type(value) is function_type
    )
    helper_function_authorities = make_tuple(
        capture_function_authority(name, cast_route(function_type, function))
        for name, function in helper_functions
    )
    verified_state_route_authority = capture_function_authority(
        "experiment_006_run_authority._verified_state",
        cast_route(function_type, verified_state_route),
    )

    def collect_code_names(code: CodeType, names: set[str], /) -> None:
        names.update(code.co_names)
        for constant in code.co_consts:
            if exact_type(constant) is code_type:
                collect_code_names(cast_route(code_type, constant), names)

    referenced_global_names: set[str] = make_set()
    for _name, helper_function in helper_functions:
        collect_code_names(
            cast_route(function_type, helper_function).__code__, referenced_global_names
        )
    global_authorities = make_tuple(
        (name, module_globals[name])
        for name in sort_values(referenced_global_names)
        if name in module_globals
    ) + (
        ("FINAL_REPORT_BYTES_MAXIMUM", report_maximum),
        ("COMPLETED_PUBLICATION_BYTES_MAXIMUM", aggregate_maximum),
        ("FINAL_REPORT_PATH", fixed_report_path),
        ("_ARTIFACT_MODE", fixed_report_mode),
        ("VerifiedRunRegistration", verified_registration_type),
    )

    module_authorities = (
        ("_run_authority", run_authority_module),
        ("builtins", builtins),
        ("hashlib", hashlib),
        ("json", json),
        ("math", math),
        ("os", os),
        ("struct", struct),
        ("threading", threading),
        ("weakref", weakref),
    )
    module_attribute_authorities = (
        (hashlib, "sha256", sha256_route),
        (json, "JSONDecodeError", json.JSONDecodeError),
        (json, "JSONDecoder", json_decoder_type),
        (json, "JSONEncoder", json_encoder_type),
        (json, "dumps", json.dumps),
        (json, "loads", json.loads),
        (math, "copysign", math.copysign),
        (math, "isfinite", math.isfinite),
        (math, "prod", math.prod),
        (os, "getpid", os.getpid),
        (struct, "iter_unpack", struct.iter_unpack),
        (struct, "pack", struct.pack),
        (struct, "unpack", struct.unpack),
        (_run_authority, "_VerifiedState", verified_state_type),
        (_run_authority, "_verified_state", verified_state_route),
    )
    builtin_names = (
        "BaseException",
        "Exception",
        "IndexError",
        "RuntimeError",
        "TypeError",
        "UnicodeError",
        "ValueError",
        "all",
        "any",
        "bool",
        "bytes",
        "dict",
        "enumerate",
        "float",
        "frozenset",
        "getattr",
        "id",
        "int",
        "isinstance",
        "len",
        "list",
        "min",
        "object",
        "range",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "type",
        "zip",
    )
    builtin_authorities = make_tuple(
        (name, builtins.__dict__[name]) for name in builtin_names
    )
    route_function_authorities: tuple[
        _FunctionAuthority,
        ...,
    ] = ()
    internal_function_authorities: tuple[
        _FunctionAuthority,
        ...,
    ] = ()
    guard_authority_box: list[Any] = []

    def integrity_failure() -> NoReturn:
        raise error_type("completed evidence issuer integrity check failed")

    def function_authority_matches(
        authority: _FunctionAuthority,
        *,
        require_global: bool,
    ) -> bool:
        (
            name,
            function,
            code,
            defaults,
            keyword_defaults,
            cells,
            contents,
        ) = authority
        if (
            (require_global and module_globals.get(name, missing) is not function)
            or function.__code__ is not code
            or function.__defaults__ is not defaults
            or function.__kwdefaults__ is not keyword_defaults
        ):
            return False
        closure = function.__closure__
        current_cells: tuple[object, ...] = (
            () if closure is None else make_tuple(closure)
        )
        if length(current_cells) != length(cells) or any_values(
            current is not expected
            for current, expected in zip_values(current_cells, cells, strict=True)
        ):
            return False
        for cell, expected in zip_values(current_cells, contents, strict=True):
            try:
                current_content = cell.cell_contents  # type: ignore[attr-defined]
            except ValueError:
                current_content = empty_cell
            if current_content is not expected:
                return False
        return True

    integrity_failure_code = integrity_failure.__code__
    function_authority_matches_code = function_authority_matches.__code__

    def require_integrity() -> None:
        if (
            integrity_failure.__code__ is not integrity_failure_code
            or function_authority_matches.__code__
            is not function_authority_matches_code
        ):
            raise error_type("completed evidence issuer integrity check failed")
        for name, expected in global_authorities:
            if module_globals.get(name, missing) is not expected:
                integrity_failure()
        for name, expected in module_authorities:
            if module_globals.get(name, missing) is not expected:
                integrity_failure()
        for module, name, expected in module_attribute_authorities:
            if get_attribute(module, name, missing) is not expected:
                integrity_failure()
        for (
            attribute_name,
            json_class,
            class_items,
            method_authorities,
        ) in json_class_authorities:
            if get_attribute(json, attribute_name, missing) is not json_class:
                integrity_failure()
            class_dictionary = json_class.__dict__
            if length(class_dictionary) != length(class_items) or any_values(
                class_dictionary.get(name, missing) is not expected
                for name, expected in class_items
            ):
                integrity_failure()
            for (
                method_authority,
                method_items,
                method_globals,
                method_global_items,
                keyword_default_items,
            ) in method_authorities:
                method = method_authority[1]
                method_dictionary = method.__dict__
                if (
                    not function_authority_matches(
                        method_authority, require_global=False
                    )
                    or length(method_dictionary) != length(method_items)
                    or any_values(
                        method_dictionary.get(name, missing) is not expected
                        for name, expected in method_items
                    )
                    or length(method_globals) != length(method_global_items)
                    or any_values(
                        method_globals.get(name, missing) is not expected
                        for name, expected in method_global_items
                    )
                ):
                    integrity_failure()
                keyword_defaults = method.__kwdefaults__
                if keyword_default_items is None:
                    if keyword_defaults is not None:
                        integrity_failure()
                else:
                    if exact_type(keyword_defaults) is not dict:
                        integrity_failure()
                    exact_keyword_defaults = cast_route(
                        "dict[str, object]", keyword_defaults
                    )
                    if length(exact_keyword_defaults) != length(
                        keyword_default_items
                    ) or any_values(
                        exact_keyword_defaults.get(name, missing) is not expected
                        for name, expected in keyword_default_items
                    ):
                        integrity_failure()
        for name, expected in builtin_authorities:
            if (
                module_globals.get(name, missing) is not missing
                or builtins.__dict__.get(name, missing) is not expected
            ):
                integrity_failure()
        verified_state_dictionary = verified_state_type.__dict__
        if (
            length(verified_state_dictionary) != length(verified_state_class_items)
            or any_values(
                verified_state_dictionary.get(name, missing) is not expected
                for name, expected in verified_state_class_items
            )
            or any_values(
                exact_type(descriptor) is not member_descriptor_type
                or verified_state_dictionary.get(name, missing) is not descriptor
                for name, descriptor in zip_values(
                    verified_state_field_names,
                    verified_state_descriptors,
                    strict=True,
                )
            )
            or not function_authority_matches(
                verified_state_route_authority,
                require_global=False,
            )
        ):
            integrity_failure()
        for authority in descriptor_authorities:
            global_name, evidence_class, field_names, descriptors, class_items = (
                authority
            )
            if module_globals.get(global_name, missing) is not evidence_class:
                integrity_failure()
            class_dictionary = evidence_class.__dict__
            if length(class_dictionary) != length(class_items) or any_values(
                class_dictionary.get(name, missing) is not expected
                for name, expected in class_items
            ):
                integrity_failure()
            if any_values(
                exact_type(descriptor) is not member_descriptor_type
                or class_dictionary.get(name, missing) is not descriptor
                for name, descriptor in zip_values(
                    field_names, descriptors, strict=True
                )
            ):
                integrity_failure()
        if (
            any_values(
                not function_authority_matches(authority, require_global=True)
                for authority in helper_function_authorities
            )
            or any_values(
                not function_authority_matches(authority, require_global=True)
                for authority in route_function_authorities
            )
            or any_values(
                not function_authority_matches(authority, require_global=False)
                for authority in internal_function_authorities
            )
        ):
            integrity_failure()

    def read_fields(
        value: object,
        authority: tuple[
            str,
            type[Any],
            tuple[str, ...],
            tuple[MemberDescriptorType, ...],
            tuple[tuple[str, object], ...],
        ],
        /,
    ) -> tuple[object, ...]:
        global_name, evidence_class, _names, descriptors, _items = authority
        if exact_type(value) is not evidence_class:
            raise error_type(f"{global_name} has lost exact type authority")
        return make_tuple(
            descriptor.__get__(value, evidence_class) for descriptor in descriptors
        )

    def require_exact_values(
        values: tuple[object, ...],
        expected_types: tuple[type[Any], ...],
        name: str,
        /,
    ) -> None:
        if length(values) != length(expected_types) or any_values(
            exact_type(value) is not expected
            for value, expected in zip_values(values, expected_types, strict=True)
        ):
            raise error_type(f"{name} lost exact primitive authority")

    def exact_tuple(
        value: object, expected_length: int, name: str, /
    ) -> tuple[object, ...]:
        if exact_type(value) is not tuple_type:
            raise error_type(f"{name} lost exact tuple authority")
        tuple_value: tuple[object, ...] = cast_route("tuple[object, ...]", value)
        if length(tuple_value) != expected_length:
            raise error_type(f"{name} lost exact tuple authority")
        return tuple_value

    def frame_registration(value: object, /) -> tuple[object, ...]:
        fields = read_fields(value, registration_authority)
        require_exact_values(fields, (str_type,) * 4, "registration evidence")
        return fields

    def frame_child(value: object, /) -> tuple[object, ...]:
        fields = read_fields(value, child_authority)
        require_exact_values(
            fields[:4],
            (str_type, int_type, int_type, int_type),
            "child identity evidence",
        )
        cpu_ids = exact_tuple(fields[4], 2, "child CPU IDs")
        require_exact_values(cpu_ids, (int_type, int_type), "child CPU IDs")
        require_exact_values(
            fields[5:9], (int_type,) * 4, "child resource and history evidence"
        )
        require_exact_values(
            fields[9:11], (str_type, str_type), "child history digest evidence"
        )
        require_exact_values(
            fields[11:20],
            (
                int_type,
                str_type,
                int_type,
                str_type,
                int_type,
                int_type,
                int_type,
                str_type,
                str_type,
            ),
            "child payload and winner evidence",
        )
        return (*fields[:4], cpu_ids, *fields[5:])

    def frame_selection(value: object, /) -> tuple[object, ...]:
        fields = read_fields(value, selection_authority)
        require_exact_values(
            fields[:7],
            (
                int_type,
                int_type,
                int_type,
                int_type,
                int_type,
                str_type,
                str_type,
            ),
            "selection evidence",
        )
        ranking = exact_tuple(fields[7], 3, "selection ranking")
        require_exact_values(ranking, (int_type,) * 3, "selection ranking")
        return (*fields[:7], ranking)

    def frame_rerun(value: object, /) -> tuple[object, ...]:
        fields = read_fields(value, rerun_authority)
        require_exact_values(
            fields,
            (int_type,) * 3 + (bool_type,) * 7,
            "rerun evidence",
        )
        return fields

    def frame_recall(value: object, /) -> tuple[object, ...]:
        fields = read_fields(value, recall_authority)
        require_exact_values(
            fields,
            (int_type, str_type, int_type, int_type, int_type, bool_type),
            "class recall evidence",
        )
        return fields

    def frame_gates(value: object, /) -> tuple[object, ...]:
        fields = read_fields(value, gates_authority)
        require_exact_values(
            fields[:4],
            (int_type, int_type, int_type, bool_type),
            "target accuracy evidence",
        )
        recalls = exact_tuple(fields[4], 10, "target recall evidence")
        recall_frames = make_tuple(frame_recall(item) for item in recalls)
        require_exact_values(
            fields[5:10],
            (int_type, int_type, int_type, int_type, bool_type),
            "macro-F1 evidence",
        )
        require_exact_values(
            fields[10:14],
            (int_type, int_type, int_type, bool_type),
            "unknown rate evidence",
        )
        require_exact_values(
            fields[14:18],
            (int_type, int_type, int_type, bool_type),
            "silence rate evidence",
        )
        require_exact_values(
            fields[18:20], (bool_type, bool_type), "aggregate gate evidence"
        )
        return (*fields[:4], recall_frames, *fields[5:])

    def frame_artifact(value: object, /) -> tuple[object, ...]:
        fields = read_fields(value, artifact_authority)
        require_exact_values(
            fields[:6],
            (str_type, str_type, str_type, bytes_type, int_type, str_type),
            "artifact evidence",
        )
        if (
            fields[6] is not None
            and exact_type(fields[6]) is not str_type
            or fields[7] is not None
            and exact_type(fields[7]) is not str_type
            or exact_type(fields[8]) is not int_type
        ):
            raise error_type("artifact optional evidence lost primitive authority")
        contents = cast_route(bytes_type, fields[3])
        if (
            cast_route(int_type, fields[4]) != length(contents)
            or sha256_route(contents).hexdigest() != fields[5]
        ):
            raise error_type("artifact bytes changed after issuance")
        return fields

    def require_primitive_frame(value: object, /) -> None:
        value_type = exact_type(value)
        if value_type is tuple_type:
            for item in cast_route("tuple[object, ...]", value):
                require_primitive_frame(item)
        elif value is not None and value_type not in (
            bytes_type,
            bool_type,
            int_type,
            str_type,
            none_type,
        ):
            raise error_type("issuer authority frame contains a non-primitive value")

    def frame_final(value: object, /) -> tuple[object, ...]:
        fields = read_fields(value, final_authority)
        require_exact_values(
            fields[:3], (int_type, str_type, str_type), "final status evidence"
        )
        children = exact_tuple(fields[4], 4, "final child evidence")
        artifacts = fields[8]
        if exact_type(artifacts) is not tuple_type:
            raise error_type("final artifact evidence lost exact tuple authority")
        artifact_values: tuple[object, ...] = cast_route(
            "tuple[object, ...]", artifacts
        )
        if length(artifact_values) not in (4, 5):
            raise error_type("final artifact evidence lost exact tuple authority")
        framed = (
            fields[0],
            fields[1],
            fields[2],
            frame_registration(fields[3]),
            make_tuple(frame_child(child) for child in children),
            frame_selection(fields[5]),
            frame_rerun(fields[6]),
            frame_gates(fields[7]),
            make_tuple(frame_artifact(artifact) for artifact in artifact_values),
            fields[9],
            fields[10],
        )
        require_exact_values(
            framed[9:11], (str_type, str_type), "final publication evidence"
        )
        require_primitive_frame(framed)
        return framed

    def publication_snapshot_from_frame(
        frame: tuple[object, ...], report_bytes: bytes, /
    ) -> CompletedPublicationSnapshot:
        status = cast_route(completed_status_alias, frame[2])
        registration_frame = cast_route(registration_frame_alias, frame[3])
        artifact_frames = cast_route("tuple[PublicationPayloadPlan, ...]", frame[8])
        snapshot: CompletedPublicationSnapshot = (
            status,
            registration_frame,
            artifact_frames,
            cast_route(str_type, frame[9]),
            cast_route(str_type, frame[10]),
            report_bytes,
        )
        require_primitive_frame(cast_route("tuple[object, ...]", snapshot))
        return snapshot

    def require_snapshot_budgets(snapshot: CompletedPublicationSnapshot, /) -> None:
        (
            _status,
            _registration,
            plans,
            _report_path,
            _report_mode,
            report_bytes,
        ) = snapshot
        budget_checker(
            report_bytes,
            make_tuple(plan[4] for plan in plans),
            report_maximum,
            aggregate_maximum,
        )

    def revoke(evidence: FinalCompletedEvidence, /) -> None:
        with lock:
            issued.pop(evidence, None)

    def validate_issued_ledger(value: object, /) -> _IssuedLedger:
        values = exact_tuple(value, 3, "issued evidence ledger")
        authority_frame = exact_tuple(values[0], 11, "issued evidence authority frame")
        if exact_type(values[1]) is not bytes_type:
            raise error_type("issued evidence report lost exact bytes authority")
        publication_snapshot = exact_tuple(
            values[2], 6, "issued evidence publication snapshot"
        )
        require_primitive_frame(authority_frame)
        require_primitive_frame(publication_snapshot)
        return (
            authority_frame,
            cast_route(bytes_type, values[1]),
            cast_route(completed_publication_alias, publication_snapshot),
        )

    def verified_ledger(value: object, /) -> _IssuedLedger:
        if exact_type(value) is not evidence_type:
            raise TypeError("evidence must be an exact FinalCompletedEvidence")
        evidence = cast_route(evidence_type, value)
        with lock:
            stored = issued.get(evidence)
        if stored is None:
            raise error_type("completed final evidence was not issued by this builder")
        try:
            authority_frame, report_bytes, publication_snapshot = (
                validate_issued_ledger(stored)
            )
            current_frame = frame_final(evidence)
            current_snapshot = publication_snapshot_from_frame(
                current_frame, report_bytes
            )
            require_snapshot_budgets(current_snapshot)
        except BaseException:
            revoke(evidence)
            raise
        if current_frame != authority_frame or current_snapshot != publication_snapshot:
            revoke(evidence)
            raise error_type("completed final evidence changed after issuance")
        return authority_frame, report_bytes, publication_snapshot

    def guarded_verified_ledger(value: object, /) -> _IssuedLedger:
        require_integrity()
        try:
            ledger = verified_ledger(value)
        except BaseException:
            require_integrity()
            raise
        require_integrity()
        return ledger

    def verified_registration_evidence(
        registration: VerifiedRunRegistration,
        /,
    ) -> RegistrationEvidence:
        if exact_type(registration) is not verified_registration_type:
            raise TypeError("registration must be an exact VerifiedRunRegistration")
        state = verified_state_route(registration)
        if exact_type(state) is not verified_state_type:
            raise error_type("run-registration state type changed")
        fields = make_tuple(
            descriptor.__get__(state, verified_state_type)
            for descriptor in verified_state_descriptors
        )
        require_exact_values(
            fields,
            (str_type,) * 4,
            "live run-registration evidence",
        )
        return registration_type(
            head_commit=cast_route(str_type, fields[0]),
            implementation_commit=cast_route(str_type, fields[1]),
            registration_sha256=cast_route(str_type, fields[2]),
            source_bundle_sha256=cast_route(str_type, fields[3]),
        )

    def build_implementation(
        registration: VerifiedRunRegistration,
        selector_result: object,
        /,
    ) -> FinalCompletedEvidence:
        require_integrity()
        result: FinalCompletedEvidence | None = None
        try:
            live_registration = verified_registration_evidence(registration)
            require_integrity()
            fields = build_fields(selector_result)
            if fields.registration != live_registration:
                raise error_type(
                    "selector child registration differs from the live "
                    "Experiment 006 capability"
                )
            report_bytes = report_builder(fields)
            result = object_new(evidence_type)
            values: tuple[object, ...] = (
                fields.schema_version,
                fields.experiment,
                fields.status,
                fields.registration,
                fields.children,
                fields.selection,
                fields.rerun,
                fields.gates,
                fields.artifacts,
                fields.report_path,
                fields.report_mode,
            )
            final_descriptors = final_authority[3]
            for descriptor, field_value in zip_values(
                final_descriptors, values, strict=True
            ):
                descriptor.__set__(result, field_value)
            authority_frame = frame_final(result)
            publication_snapshot = publication_snapshot_from_frame(
                authority_frame, report_bytes
            )
            require_snapshot_budgets(publication_snapshot)
            ledger = validate_issued_ledger(
                (
                    authority_frame,
                    report_bytes,
                    publication_snapshot,
                )
            )
            require_integrity()
            with lock:
                issued[result] = ledger
            require_integrity()
        except BaseException:
            if result is not None:
                revoke(result)
            require_integrity()
            raise
        return result

    def make_verify_implementation(
        verified_route: Callable[[object], _IssuedLedger], /
    ) -> _VerifyRoute:
        def verify_implementation(evidence: object, /) -> None:
            verified_route(evidence)

        return verify_implementation

    def make_report_implementation(
        verified_route: Callable[[object], _IssuedLedger], /
    ) -> _ReportRoute:
        def report_implementation(evidence: object, /) -> bytes:
            ledger = verified_route(evidence)
            return ledger[1]

        return report_implementation

    def make_snapshot_implementation(
        verified_route: Callable[[object], _IssuedLedger], /
    ) -> _SnapshotRoute:
        def snapshot_implementation(
            evidence: object, /
        ) -> CompletedPublicationSnapshot:
            ledger = verified_route(evidence)
            return ledger[2]

        return snapshot_implementation

    verify_implementation = make_verify_implementation(guarded_verified_ledger)
    report_implementation = make_report_implementation(guarded_verified_ledger)
    snapshot_implementation = make_snapshot_implementation(guarded_verified_ledger)

    internal_functions = (
        ("read_fields", read_fields),
        ("require_exact_values", require_exact_values),
        ("exact_tuple", exact_tuple),
        ("frame_registration", frame_registration),
        ("frame_child", frame_child),
        ("frame_selection", frame_selection),
        ("frame_rerun", frame_rerun),
        ("frame_recall", frame_recall),
        ("frame_gates", frame_gates),
        ("frame_artifact", frame_artifact),
        ("require_primitive_frame", require_primitive_frame),
        ("frame_final", frame_final),
        ("publication_snapshot_from_frame", publication_snapshot_from_frame),
        ("require_snapshot_budgets", require_snapshot_budgets),
        ("revoke", revoke),
        ("validate_issued_ledger", validate_issued_ledger),
        ("verified_ledger", verified_ledger),
        ("guarded_verified_ledger", guarded_verified_ledger),
        ("verified_registration_evidence", verified_registration_evidence),
        ("build_implementation", build_implementation),
        ("verify_implementation", verify_implementation),
        ("report_implementation", report_implementation),
        ("snapshot_implementation", snapshot_implementation),
    )
    internal_function_authorities = make_tuple(
        capture_function_authority(name, cast_route(function_type, function))
        for name, function in internal_functions
    )

    def bind_build_route(
        fail_route: Callable[[], NoReturn],
        guard_authority: list[Any],
        guard_route: Callable[[], None],
        implementation: _BuildRoute,
        match_authority: Callable[..., bool],
        /,
    ) -> _BuildRoute:
        def build_completed_final_evidence(
            registration: VerifiedRunRegistration, selector_result: object, /
        ) -> FinalCompletedEvidence:
            try:
                expected_guard = guard_authority[0]
            except IndexError:
                fail_route()
            if guard_authority[1:] or not match_authority(
                expected_guard, require_global=False
            ):
                fail_route()
            guard_route()
            try:
                result = implementation(registration, selector_result)
            except BaseException:
                guard_route()
                raise
            guard_route()
            return result

        return build_completed_final_evidence

    def bind_verify_route(
        fail_route: Callable[[], NoReturn],
        guard_authority: list[Any],
        guard_route: Callable[[], None],
        implementation: _VerifyRoute,
        match_authority: Callable[..., bool],
        /,
    ) -> _VerifyRoute:
        def verify_completed_final_evidence(evidence: object, /) -> None:
            try:
                expected_guard = guard_authority[0]
            except IndexError:
                fail_route()
            if guard_authority[1:] or not match_authority(
                expected_guard, require_global=False
            ):
                fail_route()
            guard_route()
            try:
                implementation(evidence)
            except BaseException:
                guard_route()
                raise
            guard_route()

        return verify_completed_final_evidence

    def bind_report_route(
        fail_route: Callable[[], NoReturn],
        guard_authority: list[Any],
        guard_route: Callable[[], None],
        implementation: _ReportRoute,
        match_authority: Callable[..., bool],
        /,
    ) -> _ReportRoute:
        def canonical_completed_report_bytes(evidence: object, /) -> bytes:
            try:
                expected_guard = guard_authority[0]
            except IndexError:
                fail_route()
            if guard_authority[1:] or not match_authority(
                expected_guard, require_global=False
            ):
                fail_route()
            guard_route()
            try:
                result = implementation(evidence)
            except BaseException:
                guard_route()
                raise
            guard_route()
            return result

        return canonical_completed_report_bytes

    def bind_snapshot_route(
        fail_route: Callable[[], NoReturn],
        guard_authority: list[Any],
        guard_route: Callable[[], None],
        implementation: _SnapshotRoute,
        match_authority: Callable[..., bool],
        /,
    ) -> _SnapshotRoute:
        def snapshot_completed_publication(
            evidence: object, /
        ) -> CompletedPublicationSnapshot:
            try:
                expected_guard = guard_authority[0]
            except IndexError:
                fail_route()
            if guard_authority[1:] or not match_authority(
                expected_guard, require_global=False
            ):
                fail_route()
            guard_route()
            try:
                result = implementation(evidence)
            except BaseException:
                guard_route()
                raise
            guard_route()
            return result

        return snapshot_completed_publication

    build_completed_final_evidence = bind_build_route(
        integrity_failure,
        guard_authority_box,
        require_integrity,
        build_implementation,
        function_authority_matches,
    )
    verify_completed_final_evidence = bind_verify_route(
        integrity_failure,
        guard_authority_box,
        require_integrity,
        verify_implementation,
        function_authority_matches,
    )
    canonical_completed_report_bytes = bind_report_route(
        integrity_failure,
        guard_authority_box,
        require_integrity,
        report_implementation,
        function_authority_matches,
    )
    snapshot_completed_publication = bind_snapshot_route(
        integrity_failure,
        guard_authority_box,
        require_integrity,
        snapshot_implementation,
        function_authority_matches,
    )

    routes = (
        ("build_completed_final_evidence", build_completed_final_evidence),
        ("verify_completed_final_evidence", verify_completed_final_evidence),
        ("canonical_completed_report_bytes", canonical_completed_report_bytes),
        ("snapshot_completed_publication", snapshot_completed_publication),
    )
    route_function_authorities = make_tuple(
        capture_function_authority(name, cast_route(function_type, function))
        for name, function in routes
    )
    guard_authority_box.append(
        capture_function_authority(
            "require_integrity", cast_route(function_type, require_integrity)
        )
    )

    return (
        build_completed_final_evidence,
        verify_completed_final_evidence,
        canonical_completed_report_bytes,
        snapshot_completed_publication,
    )


(
    build_completed_final_evidence,
    verify_completed_final_evidence,
    canonical_completed_report_bytes,
    snapshot_completed_publication,
) = _make_completed_evidence_routes()
del _make_completed_evidence_routes
