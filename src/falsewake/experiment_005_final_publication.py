"""Fail-closed final evidence publication for registered Experiment 005.

The final report is the only completed-outcome commit marker.  Payload names are
linked first, in one fixed order, and the report is linked last.  A failure after
the first hard link may therefore leave an immutable prefix, but it can never
create a report that claims an incomplete set.  This module never removes,
renames, replaces, or changes the mode of a canonical destination.

The module is standard-library only.  In particular, it can perform the last
clean-worktree registration verification before the first destination-local
temporary makes the repository intentionally dirty.
"""

from __future__ import annotations

import builtins
import ctypes
import errno
import hashlib
import json
import os
import stat
import threading
from collections.abc import Callable
from dataclasses import dataclass
from types import CodeType, FunctionType, MemberDescriptorType
from typing import Any, Final, Literal, NoReturn, SupportsIndex, cast

from falsewake import experiment_005_final_evidence as _completed_evidence
from falsewake import experiment_005_run_authority as _run_authority
from falsewake.experiment_005_run_authority import (
    VerifiedRunRegistration,
    reverify_verified_run_registration,
)

__all__ = (
    "EXECUTION_FAILURE_PAIRS",
    "Experiment005FinalPublicationError",
    "FailureCode",
    "FailurePhase",
    "FinalExecutionFailureEvidence",
    "build_execution_failure_evidence",
    "publish_registered_final_evidence",
    "snapshot_execution_failure_publication",
    "verify_execution_failure_evidence",
)

type FailurePhase = Literal[
    "parent_setup",
    "registered_execution",
    "completed_evidence",
    "parent_boundary",
]
type FailureCode = Literal[
    "staging_prepare_failed",
    "cpu_affinity_capture_failed",
    "source_bundle_create_failed",
    "source_bundle_close_failed",
    "seed_selection_failed",
    "completed_evidence_rejected",
    "pre_cleanup_quiescence_failed",
]
type ExecutionFailurePair = tuple[FailurePhase, FailureCode]
type RegistrationPublicationFrame = tuple[str, str, str, str]
type PublicationPayloadPlan = tuple[
    str,
    str,
    str,
    bytes,
    int,
    str,
    str | None,
    str | None,
    int,
]
type FinalPublicationSnapshot = tuple[
    str,
    RegistrationPublicationFrame,
    tuple[PublicationPayloadPlan, ...],
    str,
    str,
    bytes,
]

EXECUTION_FAILURE_PAIRS: Final = frozenset(
    {
        ("parent_setup", "staging_prepare_failed"),
        ("parent_setup", "cpu_affinity_capture_failed"),
        ("parent_setup", "source_bundle_create_failed"),
        ("registered_execution", "source_bundle_close_failed"),
        ("registered_execution", "seed_selection_failed"),
        ("completed_evidence", "completed_evidence_rejected"),
        ("parent_boundary", "pre_cleanup_quiescence_failed"),
    }
)

_CANONICAL_REPOSITORY_ROOT: Final = "/home/ubuntu/gitcode/falsewake"
_REPORT_DIRECTORY: Final = "reports"
_MODEL_DIRECTORY: Final = "models"
_REPORT_PATH: Final = "reports/experiment-005-training.json"
_TRAINING_HISTORY_PATHS: Final = (
    "reports/experiment-005-seed-20260719-history.json",
    "reports/experiment-005-seed-20260720-history.json",
    "reports/experiment-005-seed-20260721-history.json",
)
_RERUN_HISTORY_PATH: Final = "reports/experiment-005-selected-rerun-history.json"
_MODEL_PATH: Final = "models/experiment-005-selected.safetensors"
_PAYLOAD_PATHS: Final = (*_TRAINING_HISTORY_PATHS, _RERUN_HISTORY_PATH, _MODEL_PATH)
_MANAGED_PATHS: Final = (*_PAYLOAD_PATHS, _REPORT_PATH)
_PASS_KINDS: Final = (
    "training_history",
    "training_history",
    "training_history",
    "selected_rerun_history",
    "selected_safetensors",
)
_GATE_KINDS: Final = _PASS_KINDS[:4]
_MODE_TEXT: Final = "0444"
_FINAL_MODE: Final = 0o444
_TEMPORARY_MODE: Final = 0o600
_REPORT_BYTES_MAXIMUM: Final = 256 << 10
_PUBLICATION_BYTES_MAXIMUM: Final = 8 << 20
_MODEL_TENSOR_COUNT: Final = 53
_MODEL_VALUE_COUNT: Final = 23_724
_TEMP_PREFIX: Final = ".falsewake-exp005-publication-"
_EXPERIMENT: Final = "005"
_SCIENTIFIC_PROTOCOL: Final = "002"
_PREDECESSOR_EXECUTION_PROTOCOL_COMMIT: Final = (
    "c7a8b3c493e05211ff7afa3a7abb974fc1c8b4e2"
)
_PREDECESSOR_EXECUTION_PROTOCOL_SHA256: Final = (
    "aba6c1eca84ad33c7751e4768a8ef630ba2c063720fd29938dd1b69abf945a46"
)
_PREDECESSOR_IMPLEMENTATION_COMMIT: Final = "f81cd142885068c27767d6d728da04248fb1a470"
_PREDECESSOR_PROTOCOL_PROOF_SHA256: Final = (
    "8539610615717b8b1fb5ff41876aef41a36e8ac7a8dbda4eeb7307174ae7a3a8"
)
_PREDECESSOR_SHARED_AUTHORITY_SHA256: Final = (
    "52ce6039345b2159a4bf83ea294af2c3211cdcfbe87e9bc806133c67647b1a47"
)
_PREDECESSOR_INCIDENT_COMMIT: Final = "462aeba306a0612fd6d64884e323d72db3569a89"
_PREDECESSOR_INCIDENT_SHA256: Final = (
    "d015b749acf6e98e87172372315c78ccfaad48b8ae6b3a29ae426890aa467f50"
)
_PREDECESSOR_PREFLIGHT_BOUNDARY_COMMIT: Final = (
    "55043086af773e613502f81685662e7fcc58f413"
)
_PREDECESSOR_PREFLIGHT_BOUNDARY_PARENT: Final = _PREDECESSOR_IMPLEMENTATION_COMMIT
_PREDECESSOR_CANONICAL_JSON: Final = (
    '{"attempt":{"canonical_marker":"/home/ubuntu/gitcode/.t/falsewake-experiment-004-attempt'
    '","canonical_marker_present":false,"optimizer_updates":0,"registered_attempt_consumed":f'
    'alse,"registered_authority_issuer_invoked":false,"registered_coordinator_invoked":false,'
    '"registered_invocation_count":0,"registered_runner_invoked":false,"validation_examples":'
    '0},"execution_protocol":{"introduction_commit":"c7a8b3c493e05211ff7afa3a7abb974fc1c8b4e2'
    '","mode":"100644","path":"configs/experiment-004-execution.json","sha256":"aba6c1eca84ad'
    '33c7751e4768a8ef630ba2c063720fd29938dd1b69abf945a46"},"experiment":"004","implementation'
    '":{"commit":"f81cd142885068c27767d6d728da04248fb1a470","production_sources":[{"path":"sr'
    'c/falsewake/experiment_004_coordinator.py","sha256":"6e4df88440c361376ce51a3574a16bbec79'
    'ecbc6ff4dd1fcf0cf8be86eac1daa"},{"path":"src/falsewake/experiment_004_final_evidence.py"'
    ',"sha256":"5fcff993e434c36d73c5e9b4f6d1e5954e8a54f2ce9fc17450ea615c5a4df5af"},{"path":"s'
    'rc/falsewake/experiment_004_final_publication.py","sha256":"b862490f15e88819e815cfe6d776'
    '9ff125e21e59cdde4642a4a700e5f2eaad16"},{"path":"src/falsewake/experiment_004_run_authori'
    'ty.py","sha256":"cacab13729f14bfdce10afa5b53b2a09116cf56abe21aee12ac9220a6ad2e5b5"},{"pa'
    'th":"src/falsewake/experiment_004_runner.py","sha256":"76f19ec33c10955fa50d26b8d35d56aeb'
    '15d712b72b22a17e88933f46d24accf"},{"path":"src/falsewake/experiment_004_seed_worker.py",'
    '"sha256":"40cc3f0681006b03eac03f4747c60feb7a0ab42c102772afd6b90ef809929361"},{"path":"sr'
    'c/falsewake/experiment_004_supervisor.py","sha256":"750371018aefd8dc3cdbffff20a60c378568'
    '0992b05d6246f052f37609a6c8c9"}],"protocol_proof":{"path":"tests/test_experiment_004_prot'
    'ocol.py","sha256":"8539610615717b8b1fb5ff41876aef41a36e8ac7a8dbda4eeb7307174ae7a3a8"},"s'
    'hared_authority":{"path":"src/falsewake/experiment_002_run_authority.py","sha256":"52ce6'
    '039345b2159a4bf83ea294af2c3211cdcfbe87e9bc806133c67647b1a47"}},"incident":{"commit":"462'
    'aeba306a0612fd6d64884e323d72db3569a89","mode":"100644","path":"reports/experiment-004-pr'
    'eflight-incident.json","sha256":"d015b749acf6e98e87172372315c78ccfaad48b8ae6b3a29ae42689'
    '0aa467f50"},"managed_outputs_absent":["configs/experiment-004-run.json","models/experime'
    'nt-004-selected.safetensors","reports/experiment-004-seed-20260719-history.json","report'
    's/experiment-004-seed-20260720-history.json","reports/experiment-004-seed-20260721-histo'
    'ry.json","reports/experiment-004-selected-rerun-history.json","reports/experiment-004-tr'
    'aining.json"],"outcome":{"automatic_terminal_report_published":false,"checkpoint_reusabl'
    'e":false,"code":"pre_registration_protocol_rejection","phase":"pre_registration_protocol'
    '_preflight","reason":"frozen_protocol_contract_is_internally_unsatisfiable","status":"pr'
    'eflight_rejected"},"registration":{"path":"configs/experiment-004-run.json","present":fa'
    'lse},"reuse_forbidden":true,"scientific_protocol":"002","terminal":true,"topology":{"imp'
    'lementation_commit":"f81cd142885068c27767d6d728da04248fb1a470","implementation_commit_is'
    '_ancestor_of_incident":true,"incident_commit":"462aeba306a0612fd6d64884e323d72db3569a89"'
    ',"incident_parent":"55043086af773e613502f81685662e7fcc58f413","outcome_report_path":"rep'
    'orts/experiment-004-training.json","outcome_report_present":false,"preflight_boundary_co'
    'mmit":"55043086af773e613502f81685662e7fcc58f413","preflight_boundary_parent":"f81cd14288'
    '5068c27767d6d728da04248fb1a470","registration_path":"configs/experiment-004-run.json","r'
    'egistration_present":false}}'
)
_LOWER_HEX: Final = frozenset("0123456789abcdef")
_OWNER_PROCESS: Final = os.getpid()
_AT_EMPTY_PATH: Final = 0x1000
_LIBC: Final = ctypes.CDLL(None, use_errno=True)
_LINKAT: Final = _LIBC.linkat
_LINKAT.argtypes = (
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.c_int,
)
_LINKAT.restype = ctypes.c_int


class Experiment005FinalPublicationError(ValueError):
    """Final evidence issuance or publication failed closed."""


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
class FinalExecutionFailureEvidence:
    """Issuer-owned evidence for one controlled pre-publication failure."""

    schema_version: int
    experiment: str
    status: Literal["execution_failure"]
    registration: RegistrationPublicationFrame
    phase: FailurePhase
    code: FailureCode
    report_path: str
    report_mode: str

    def __init__(self) -> None:
        raise TypeError("execution-failure evidence is issued only by its builder")

    def __copy__(self) -> NoReturn:
        raise TypeError("execution-failure evidence cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        del memo
        raise TypeError("execution-failure evidence cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("execution-failure evidence cannot be serialized")

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        del protocol
        raise TypeError("execution-failure evidence cannot be serialized")


@dataclass(frozen=True, slots=True)
class _FailureState:
    frame: tuple[object, ...]
    snapshot: FinalPublicationSnapshot


@dataclass(frozen=True, slots=True)
class _PublicationLayout:
    repository_root: str
    report_directory: str = _REPORT_DIRECTORY
    model_directory: str = _MODEL_DIRECTORY


@dataclass(slots=True)
class _PinnedDirectory:
    name: str
    descriptor: int
    identity: tuple[int, int]


@dataclass(slots=True)
class _PinnedLayout:
    layout: _PublicationLayout
    root: _PinnedDirectory
    reports: _PinnedDirectory
    models: _PinnedDirectory


@dataclass(frozen=True, slots=True)
class _FilePlan:
    path: str
    contents: bytes
    byte_count: int
    sha256: str
    slot: int


@dataclass(slots=True)
class _PreparedFile:
    plan: _FilePlan
    parent: _PinnedDirectory
    destination_leaf: str
    temporary_leaf: str | None
    descriptor: int
    identity: tuple[int, int]
    published: bool = False


@dataclass(slots=True)
class _CommitState:
    started: bool = False


type _FinalAuthorityCheck = Callable[[], None]
type _CompletedSnapshotRoute = Callable[[object], object]
type _FailureSnapshotRoute = Callable[[object], FinalPublicationSnapshot]
type _CompletedHistoryParser = Callable[[object, int], Any]
type _CompletedGatesComputer = Callable[[Any], Any]
type _CompletedGatesDocument = Callable[[Any], dict[str, object]]

_REGISTERED_SEEDS: Final = (20_260_719, 20_260_720, 20_260_721)
_SEED_ROLE: Final = "training_seed"
_RERUN_ROLE: Final = "selected_seed_rerun"
_EPOCH_COUNT: Final = 30
_REGISTERED_SAFETENSORS_BYTES: Final = 99_776
_CHILD_ENVELOPE_BYTES_MAXIMUM: Final = 64 << 10
_CHILD_WALL_NANOSECONDS_MAXIMUM: Final = 21_600 * 1_000_000_000
_CHILD_RSS_BYTES_MAXIMUM: Final = 8_589_934_592
_CHILD_OUTPUT_BYTES_MAXIMUM: Final = 4_294_967_296
_UINT32_MAXIMUM: Final = (1 << 32) - 1
_COMPLETED_HISTORY_PARSER: Final[_CompletedHistoryParser] = cast(
    _CompletedHistoryParser,
    _completed_evidence._parse_history,
)
_COMPLETED_GATES_COMPUTER: Final[_CompletedGatesComputer] = cast(
    _CompletedGatesComputer,
    _completed_evidence._compute_gates,
)
_COMPLETED_GATES_DOCUMENT: Final[_CompletedGatesDocument] = cast(
    _CompletedGatesDocument,
    _completed_evidence._gates_document,
)


def _fail(message: str) -> NoReturn:
    raise Experiment005FinalPublicationError(message)


def _require_hex(value: object, length: int, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != length
        or any(character not in _LOWER_HEX for character in value)
    ):
        _fail(f"{name} is not exact lowercase hexadecimal")
    return value


def _require_registration_frame(value: object) -> RegistrationPublicationFrame:
    if type(value) is not tuple or len(cast(tuple[object, ...], value)) != 4:
        _fail("registration publication frame has an invalid exact shape")
    fields = cast(tuple[object, object, object, object], value)
    head = _require_hex(fields[0], 40, "registration HEAD commit")
    implementation = _require_hex(fields[1], 40, "implementation commit")
    registration_sha256 = _require_hex(fields[2], 64, "registration digest")
    source_sha256 = _require_hex(fields[3], 64, "source-bundle digest")
    if head == implementation:
        _fail("registration commit must be later than its implementation commit")
    return head, implementation, registration_sha256, source_sha256


def _canonical_json_bytes(document: object) -> bytes:
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


def _predecessor_document() -> dict[str, object]:
    return {
        "attempt": {
            "canonical_marker": (
                "/home/ubuntu/gitcode/.t/falsewake-experiment-004-attempt"
            ),
            "canonical_marker_present": False,
            "optimizer_updates": 0,
            "registered_attempt_consumed": False,
            "registered_authority_issuer_invoked": False,
            "registered_coordinator_invoked": False,
            "registered_invocation_count": 0,
            "registered_runner_invoked": False,
            "validation_examples": 0,
        },
        "execution_protocol": {
            "introduction_commit": _PREDECESSOR_EXECUTION_PROTOCOL_COMMIT,
            "mode": "100644",
            "path": "configs/experiment-004-execution.json",
            "sha256": _PREDECESSOR_EXECUTION_PROTOCOL_SHA256,
        },
        "experiment": "004",
        "implementation": {
            "commit": _PREDECESSOR_IMPLEMENTATION_COMMIT,
            "production_sources": [
                {
                    "path": "src/falsewake/experiment_004_coordinator.py",
                    "sha256": (
                        "6e4df88440c361376ce51a3574a16bbec79ecbc6ff4dd1fcf0cf8be86eac1daa"
                    ),
                },
                {
                    "path": "src/falsewake/experiment_004_final_evidence.py",
                    "sha256": (
                        "5fcff993e434c36d73c5e9b4f6d1e5954e8a54f2ce9fc17450ea615c5a4df5af"
                    ),
                },
                {
                    "path": "src/falsewake/experiment_004_final_publication.py",
                    "sha256": (
                        "b862490f15e88819e815cfe6d7769ff125e21e59cdde4642a4a700e5f2eaad16"
                    ),
                },
                {
                    "path": "src/falsewake/experiment_004_run_authority.py",
                    "sha256": (
                        "cacab13729f14bfdce10afa5b53b2a09116cf56abe21aee12ac9220a6ad2e5b5"
                    ),
                },
                {
                    "path": "src/falsewake/experiment_004_runner.py",
                    "sha256": (
                        "76f19ec33c10955fa50d26b8d35d56aeb15d712b72b22a17e88933f46d24accf"
                    ),
                },
                {
                    "path": "src/falsewake/experiment_004_seed_worker.py",
                    "sha256": (
                        "40cc3f0681006b03eac03f4747c60feb7a0ab42c102772afd6b90ef809929361"
                    ),
                },
                {
                    "path": "src/falsewake/experiment_004_supervisor.py",
                    "sha256": (
                        "750371018aefd8dc3cdbffff20a60c3785680992b05d6246f052f37609a6c8c9"
                    ),
                },
            ],
            "protocol_proof": {
                "path": "tests/test_experiment_004_protocol.py",
                "sha256": _PREDECESSOR_PROTOCOL_PROOF_SHA256,
            },
            "shared_authority": {
                "path": "src/falsewake/experiment_002_run_authority.py",
                "sha256": _PREDECESSOR_SHARED_AUTHORITY_SHA256,
            },
        },
        "incident": {
            "commit": _PREDECESSOR_INCIDENT_COMMIT,
            "mode": "100644",
            "path": "reports/experiment-004-preflight-incident.json",
            "sha256": _PREDECESSOR_INCIDENT_SHA256,
        },
        "managed_outputs_absent": [
            "configs/experiment-004-run.json",
            "models/experiment-004-selected.safetensors",
            "reports/experiment-004-seed-20260719-history.json",
            "reports/experiment-004-seed-20260720-history.json",
            "reports/experiment-004-seed-20260721-history.json",
            "reports/experiment-004-selected-rerun-history.json",
            "reports/experiment-004-training.json",
        ],
        "outcome": {
            "automatic_terminal_report_published": False,
            "checkpoint_reusable": False,
            "code": "pre_registration_protocol_rejection",
            "phase": "pre_registration_protocol_preflight",
            "reason": "frozen_protocol_contract_is_internally_unsatisfiable",
            "status": "preflight_rejected",
        },
        "registration": {
            "path": "configs/experiment-004-run.json",
            "present": False,
        },
        "reuse_forbidden": True,
        "scientific_protocol": _SCIENTIFIC_PROTOCOL,
        "terminal": True,
        "topology": {
            "implementation_commit": _PREDECESSOR_IMPLEMENTATION_COMMIT,
            "implementation_commit_is_ancestor_of_incident": True,
            "incident_commit": _PREDECESSOR_INCIDENT_COMMIT,
            "incident_parent": _PREDECESSOR_PREFLIGHT_BOUNDARY_COMMIT,
            "outcome_report_path": "reports/experiment-004-training.json",
            "outcome_report_present": False,
            "preflight_boundary_commit": _PREDECESSOR_PREFLIGHT_BOUNDARY_COMMIT,
            "preflight_boundary_parent": _PREDECESSOR_PREFLIGHT_BOUNDARY_PARENT,
            "registration_path": "configs/experiment-004-run.json",
            "registration_present": False,
        },
    }


def _failure_report_bytes(
    registration: RegistrationPublicationFrame,
    phase: FailurePhase,
    code: FailureCode,
) -> bytes:
    """Build fixed canonical ASCII without a mutable JSON encoder dependency."""

    head, implementation, registration_sha256, source_sha256 = registration
    return (
        '{"artifacts":[],"checkpoint_reusable":false,"experiment":"005",'
        f'"failure":{{"code":"{code}","phase":"{phase}"}},'
        f'"predecessor":{_PREDECESSOR_CANONICAL_JSON},'
        f'"publication":{{"report_mode":"{_MODE_TEXT}",'
        f'"report_path":"{_REPORT_PATH}"}},'
        f'"registration":{{"head_commit":"{head}",'
        f'"implementation_commit":"{implementation}",'
        '"path":"configs/experiment-005-run.json",'
        f'"registration_sha256":"{registration_sha256}",'
        f'"source_bundle_sha256":"{source_sha256}"}},'
        '"schema_version":1,"scientific_protocol":"002",'
        '"status":"execution_failure"}\n'
    ).encode("ascii")


def _make_failure_evidence_routes() -> tuple[
    Callable[
        [RegistrationPublicationFrame, FailurePhase, FailureCode],
        FinalExecutionFailureEvidence,
    ],
    Callable[[object], None],
    _FailureSnapshotRoute,
]:
    evidence_type = FinalExecutionFailureEvidence
    failure_state_type = _FailureState
    exact_type = type
    string_type = str
    object_new = object.__new__
    error_type = Experiment005FinalPublicationError
    type_error = TypeError
    pair_authority = EXECUTION_FAILURE_PAIRS
    report_builder = _failure_report_bytes
    snapshot_type = tuple
    make_tuple = tuple
    sort_values = sorted
    any_values = any
    enumerate_values = enumerate
    make_zip = zip
    length = len
    cast_route = cast
    member_descriptor_type = MemberDescriptorType
    attribute_error_type = AttributeError
    module_globals = globals()
    fixed_report_path = _REPORT_PATH
    fixed_report_mode = _MODE_TEXT
    report_maximum = _REPORT_BYTES_MAXIMUM
    lower_hex = _LOWER_HEX

    def validate_registration_frame(
        value: object,
        /,
    ) -> RegistrationPublicationFrame:
        if (
            exact_type(value) is not snapshot_type
            or length(cast_route(tuple[object, ...], value)) != 4
        ):
            raise error_type(
                "registration publication frame has an invalid exact shape"
            )
        fields = cast_route(tuple[object, object, object, object], value)

        def require_hex(
            field: object,
            expected_length: int,
            name: str,
        ) -> str:
            if (
                exact_type(field) is not string_type
                or length(cast_route(str, field)) != expected_length
                or any_values(
                    character not in lower_hex for character in cast_route(str, field)
                )
            ):
                raise error_type(f"{name} is not exact lowercase hexadecimal")
            return cast_route(str, field)

        head = require_hex(fields[0], 40, "registration HEAD commit")
        implementation = require_hex(fields[1], 40, "implementation commit")
        registration_sha256 = require_hex(
            fields[2],
            64,
            "registration digest",
        )
        source_sha256 = require_hex(fields[3], 64, "source-bundle digest")
        if head == implementation:
            raise error_type(
                "registration commit must be later than its implementation commit"
            )
        return head, implementation, registration_sha256, source_sha256

    registration_validator = validate_registration_frame
    lock = threading.RLock()
    issued: dict[FinalExecutionFailureEvidence, _FailureState] = {}
    field_names = (
        "schema_version",
        "experiment",
        "status",
        "registration",
        "phase",
        "code",
        "report_path",
        "report_mode",
    )
    class_namespace = evidence_type.__dict__
    namespace_names = tuple(sorted(class_namespace))
    namespace_values = tuple(class_namespace[name] for name in namespace_names)
    descriptors = tuple(class_namespace[name] for name in field_names)
    state_namespace = failure_state_type.__dict__
    state_namespace_names = tuple(sorted(state_namespace))
    state_namespace_values = tuple(
        state_namespace[name] for name in state_namespace_names
    )
    state_frame_descriptor = state_namespace["frame"]
    state_snapshot_descriptor = state_namespace["snapshot"]
    if any(exact_type(value) is not MemberDescriptorType for value in descriptors):
        raise RuntimeError("execution-failure evidence slots are unavailable")
    if any(
        exact_type(value) is not MemberDescriptorType
        for value in (state_frame_descriptor, state_snapshot_descriptor)
    ):
        raise RuntimeError("execution-failure state slots are unavailable")

    helper_names = ("_failure_report_bytes",)
    helper_functions = (
        *make_tuple(module_globals[name] for name in helper_names),
        cast_route,
        registration_validator,
    )
    helper_frames = make_tuple(
        (
            function,
            function.__code__,
            function.__defaults__,
            (
                None
                if function.__defaults__ is None
                else make_tuple(function.__defaults__)
            ),
            function.__kwdefaults__,
            (
                None
                if function.__kwdefaults__ is None
                else make_tuple(function.__kwdefaults__.items())
            ),
            make_tuple(function.__closure__ or ()),
            make_tuple(cell.cell_contents for cell in (function.__closure__ or ())),
        )
        for function in helper_functions
    )
    global_names = (
        "Experiment005FinalPublicationError",
        "FinalExecutionFailureEvidence",
        "_FailureState",
        "EXECUTION_FAILURE_PAIRS",
        "_REPORT_PATH",
        "_MODE_TEXT",
        "_REPORT_BYTES_MAXIMUM",
        "_EXPERIMENT",
        "_PREDECESSOR_CANONICAL_JSON",
        "cast",
        "MemberDescriptorType",
        *helper_names,
    )
    global_values = make_tuple(module_globals[name] for name in global_names)

    def require_issuer_authority() -> None:
        current = evidence_type.__dict__
        current_state = failure_state_type.__dict__
        if (
            any_values(
                module_globals.get(name) is not global_values[index]
                for index, name in enumerate_values(global_names)
            )
            or make_tuple(sort_values(current)) != namespace_names
            or any_values(
                current[name] is not namespace_values[index]
                for index, name in enumerate_values(namespace_names)
            )
            or make_tuple(sort_values(current_state)) != state_namespace_names
            or any_values(
                current_state[name] is not state_namespace_values[index]
                for index, name in enumerate_values(state_namespace_names)
            )
        ):
            raise error_type("execution-failure evidence issuer authority changed")
        for (
            function,
            code,
            defaults,
            default_values,
            kwdefaults,
            kwdefault_values,
            closure,
            closure_values,
        ) in helper_frames:
            current_defaults = function.__defaults__
            current_kwdefaults = function.__kwdefaults__
            current_closure = make_tuple(function.__closure__ or ())
            if (
                function.__code__ is not code
                or current_defaults is not defaults
                or (
                    current_defaults is not None
                    and (
                        default_values is None
                        or length(current_defaults) != length(default_values)
                        or any_values(
                            value is not default_values[index]
                            for index, value in enumerate_values(current_defaults)
                        )
                    )
                )
                or current_kwdefaults is not kwdefaults
                or (
                    current_kwdefaults is not None
                    and (
                        kwdefault_values is None
                        or make_tuple(current_kwdefaults)
                        != make_tuple(name for name, _value in kwdefault_values)
                        or any_values(
                            current_kwdefaults[name] is not value
                            for name, value in kwdefault_values
                        )
                    )
                )
                or length(current_closure) != length(closure)
                or any_values(
                    cell is not closure[index]
                    or cell.cell_contents is not closure_values[index]
                    for index, cell in enumerate_values(current_closure)
                )
            ):
                raise error_type(
                    "execution-failure evidence issuer function authority changed"
                )

    def frame(value: object, /) -> tuple[object, ...]:
        require_issuer_authority()
        if exact_type(value) is not evidence_type:
            raise type_error("evidence must be an exact FinalExecutionFailureEvidence")
        try:
            return make_tuple(
                cast_route(member_descriptor_type, descriptor).__get__(
                    value, evidence_type
                )
                for descriptor in descriptors
            )
        except attribute_error_type as error:
            raise error_type(
                "execution-failure evidence lacks its exact issued fields"
            ) from error

    def validated_snapshot(value: object, /) -> FinalPublicationSnapshot:
        fields = frame(value)
        evidence = cast(FinalExecutionFailureEvidence, value)
        with lock:
            state = issued.get(evidence)
        if state is None or exact_type(state) is not failure_state_type:
            with lock:
                issued.pop(evidence, None)
            raise error_type(
                "execution-failure evidence was not issued or changed after issuance"
            )
        issued_frame = cast_route(
            member_descriptor_type, state_frame_descriptor
        ).__get__(state, failure_state_type)
        issued_snapshot = cast_route(
            FinalPublicationSnapshot,
            cast_route(member_descriptor_type, state_snapshot_descriptor).__get__(
                state, failure_state_type
            ),
        )
        if fields != issued_frame:
            with lock:
                issued.pop(evidence, None)
            raise error_type(
                "execution-failure evidence was not issued or changed after issuance"
            )
        if exact_type(issued_snapshot) is not snapshot_type or issued_snapshot != (
            "execution_failure",
            fields[3],
            (),
            fixed_report_path,
            fixed_report_mode,
            report_builder(
                cast_route(RegistrationPublicationFrame, fields[3]),
                cast_route(FailurePhase, fields[4]),
                cast_route(FailureCode, fields[5]),
            ),
        ):
            with lock:
                issued.pop(evidence, None)
            raise error_type("execution-failure publication snapshot changed")
        return cast(FinalPublicationSnapshot, issued_snapshot)

    def build_execution_failure_evidence(
        registration_frame: RegistrationPublicationFrame,
        phase: FailurePhase,
        code: FailureCode,
        /,
    ) -> FinalExecutionFailureEvidence:
        require_issuer_authority()
        registration = registration_validator(registration_frame)
        if exact_type(phase) is not string_type or exact_type(code) is not string_type:
            raise type_error("failure phase and code must be exact strings")
        pair: ExecutionFailurePair = (phase, code)
        if pair not in pair_authority:
            raise error_type("execution-failure phase and code are not registered")
        report = report_builder(registration, phase, code)
        if not 0 < length(report) <= report_maximum:
            raise error_type("execution-failure report exceeds its byte ceiling")
        result = object_new(evidence_type)
        values: tuple[object, ...] = (
            1,
            _EXPERIMENT,
            "execution_failure",
            registration,
            phase,
            code,
            fixed_report_path,
            fixed_report_mode,
        )
        for descriptor, value in make_zip(descriptors, values, strict=True):
            cast_route(member_descriptor_type, descriptor).__set__(result, value)
        snapshot: FinalPublicationSnapshot = (
            "execution_failure",
            registration,
            (),
            fixed_report_path,
            fixed_report_mode,
            report,
        )
        state = object_new(failure_state_type)
        cast_route(member_descriptor_type, state_frame_descriptor).__set__(
            state, values
        )
        cast_route(member_descriptor_type, state_snapshot_descriptor).__set__(
            state, snapshot
        )
        with lock:
            issued[result] = state
        if frame(result) != values:
            with lock:
                issued.pop(result, None)
            raise error_type("execution-failure evidence changed during issuance")
        return result

    def verify_execution_failure_evidence(evidence: object, /) -> None:
        validated_snapshot(evidence)

    def snapshot_execution_failure_publication(
        evidence: object, /
    ) -> FinalPublicationSnapshot:
        return validated_snapshot(evidence)

    return (
        build_execution_failure_evidence,
        verify_execution_failure_evidence,
        snapshot_execution_failure_publication,
    )


(
    build_execution_failure_evidence,
    verify_execution_failure_evidence,
    snapshot_execution_failure_publication,
) = _make_failure_evidence_routes()
del _make_failure_evidence_routes


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("final report contains a duplicate JSON key")
        result[key] = value
    return result


def _reject_json_number(value: str) -> NoReturn:
    del value
    _fail("final report contains a forbidden JSON number")


def _parse_canonical_report(raw: bytes) -> dict[str, Any]:
    if not 0 < len(raw) <= _REPORT_BYTES_MAXIMUM or not raw.endswith(b"\n"):
        _fail("final report has an invalid byte count or terminator")
    try:
        text = raw.decode("ascii", errors="strict")
        parsed = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_number,
            parse_float=_reject_json_number,
        )
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as error:
        raise Experiment005FinalPublicationError(
            "final report is not strict canonical JSON"
        ) from error
    if type(parsed) is not dict or _canonical_json_bytes(parsed) != raw:
        _fail("final report is not canonical compact ASCII JSON plus LF")
    return cast(dict[str, Any], parsed)


def _require_exact_object(
    value: object, keys: frozenset[str], name: str
) -> dict[str, Any]:
    if type(value) is not dict or frozenset(cast(dict[object, object], value)) != keys:
        _fail(f"{name} has unexpected or missing keys")
    return cast(dict[str, Any], value)


def _require_exact_json_value(
    observed: object,
    expected: object,
    name: str,
) -> None:
    """Compare JSON values without allowing bool/int equality aliases."""

    if type(observed) is not type(expected):
        _fail(f"{name} has an invalid exact JSON type")
    if type(expected) is dict:
        observed_object = cast(dict[object, object], observed)
        expected_object = cast(dict[object, object], expected)
        if tuple(observed_object) != tuple(expected_object):
            _fail(f"{name} has unexpected, missing, or reordered keys")
        for key, expected_value in expected_object.items():
            _require_exact_json_value(
                observed_object[key],
                expected_value,
                f"{name}.{key}",
            )
        return
    if type(expected) is list:
        observed_items = cast(list[object], observed)
        expected_items = cast(list[object], expected)
        if len(observed_items) != len(expected_items):
            _fail(f"{name} has an invalid exact array length")
        for index, (observed_item, expected_item) in enumerate(
            zip(observed_items, expected_items, strict=True)
        ):
            _require_exact_json_value(
                observed_item,
                expected_item,
                f"{name}[{index}]",
            )
        return
    if observed != expected:
        _fail(f"{name} differs from its recomputed value")


def _require_bounded_int(
    value: object,
    name: str,
    *,
    minimum: int = 0,
    maximum: int,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _fail(f"{name} is outside its exact integer bounds")
    return value


def _require_child_resources(
    value: object,
    expected_cpu_ids: tuple[int, int] | None,
) -> tuple[int, int]:
    resources = _require_exact_object(
        value,
        frozenset(
            {
                "cpu_ids",
                "elapsed_nanoseconds",
                "maximum_rss_bytes",
                "output_and_scratch_bytes",
                "pid",
            }
        ),
        "completed child resources",
    )
    raw_cpu_ids = resources["cpu_ids"]
    if type(raw_cpu_ids) is not list or len(cast(list[object], raw_cpu_ids)) != 2:
        _fail("completed child CPU IDs have an invalid exact shape")
    cpu_values = cast(list[object], raw_cpu_ids)
    cpu_ids = (
        _require_bounded_int(
            cpu_values[0],
            "completed child first CPU ID",
            maximum=_UINT32_MAXIMUM,
        ),
        _require_bounded_int(
            cpu_values[1],
            "completed child second CPU ID",
            maximum=_UINT32_MAXIMUM,
        ),
    )
    if cpu_ids[0] >= cpu_ids[1] or (
        expected_cpu_ids is not None and cpu_ids != expected_cpu_ids
    ):
        _fail("completed children do not share two increasing CPU IDs")
    _require_bounded_int(
        resources["pid"],
        "completed child PID",
        minimum=1,
        maximum=_UINT32_MAXIMUM,
    )
    _require_bounded_int(
        resources["elapsed_nanoseconds"],
        "completed child elapsed nanoseconds",
        maximum=_CHILD_WALL_NANOSECONDS_MAXIMUM,
    )
    _require_bounded_int(
        resources["maximum_rss_bytes"],
        "completed child maximum RSS bytes",
        maximum=_CHILD_RSS_BYTES_MAXIMUM,
    )
    _require_bounded_int(
        resources["output_and_scratch_bytes"],
        "completed child output bytes",
        maximum=_CHILD_OUTPUT_BYTES_MAXIMUM,
    )
    return cpu_ids


def _require_completed_report_semantics(
    document: dict[str, Any],
    status: str,
    plans: tuple[PublicationPayloadPlan, ...],
) -> None:
    """Recompute rank, rerun, and gates from the four published histories."""

    training_histories = tuple(
        _COMPLETED_HISTORY_PARSER(plans[index][3], seed)
        for index, seed in enumerate(_REGISTERED_SEEDS)
    )
    ranking = tuple(
        sorted(
            range(3),
            key=lambda index: (
                -training_histories[index].winner.macro_f1,
                training_histories[index].winner.validation_cross_entropy,
                training_histories[index].winner.zero_based_epoch,
                _REGISTERED_SEEDS[index],
            ),
        )
    )
    selected_ordinal = ranking[0]
    selected_seed = _REGISTERED_SEEDS[selected_ordinal]
    rerun_history = _COMPLETED_HISTORY_PARSER(plans[3][3], selected_seed)
    histories = (*training_histories, rerun_history)
    if (
        plans[selected_ordinal][3] != plans[3][3]
        or training_histories[selected_ordinal].raw_bytes != rerun_history.raw_bytes
    ):
        _fail("selected and rerun histories are not byte-identical")
    for index, history in enumerate(histories):
        if (
            history.raw_sha256 != plans[index][5]
            or history.domain_sha256 != plans[index][6]
        ):
            _fail("completed history payload differs from its parsed domain binding")

    raw_children = document["children"]
    if type(raw_children) is not list or len(cast(list[object], raw_children)) != 4:
        _fail("completed report must contain exactly four children")
    children = cast(list[object], raw_children)
    expected_cpu_ids: tuple[int, int] | None = None
    child_safetensors_sha256: list[str] = []
    for index, (raw_child, history) in enumerate(zip(children, histories, strict=True)):
        child = _require_exact_object(
            raw_child,
            frozenset(
                {
                    "binding",
                    "envelope",
                    "history",
                    "resources",
                    "safetensors",
                    "winner",
                }
            ),
            "completed child",
        )
        expected_binding = {
            "ordinal": index,
            "role": _SEED_ROLE if index < 3 else _RERUN_ROLE,
            "seed": _REGISTERED_SEEDS[index] if index < 3 else selected_seed,
        }
        _require_exact_json_value(
            child["binding"],
            expected_binding,
            "completed child binding",
        )
        expected_history = {
            "byte_count": plans[index][4],
            "domain_sha256": plans[index][6],
            "sha256": plans[index][5],
        }
        _require_exact_json_value(
            child["history"],
            expected_history,
            "completed child history",
        )
        winner = history.winner
        expected_winner = {
            "macro_f1_exact_denominator": winner.macro_f1.denominator,
            "macro_f1_exact_numerator": winner.macro_f1.numerator,
            "model_tensor_sha256": winner.model_tensor_digest,
            "validation_cross_entropy_float64_hex": (
                winner.validation_cross_entropy_float64_hex
            ),
            "zero_based_epoch": winner.zero_based_epoch,
        }
        _require_exact_json_value(
            child["winner"],
            expected_winner,
            "completed child winner",
        )
        envelope = _require_exact_object(
            child["envelope"],
            frozenset({"byte_count", "domain_sha256"}),
            "completed child envelope",
        )
        _require_bounded_int(
            envelope["byte_count"],
            "completed child envelope byte count",
            minimum=1,
            maximum=_CHILD_ENVELOPE_BYTES_MAXIMUM,
        )
        _require_hex(
            envelope["domain_sha256"],
            64,
            "completed child envelope domain digest",
        )
        safetensors = _require_exact_object(
            child["safetensors"],
            frozenset({"byte_count", "sha256"}),
            "completed child safetensors",
        )
        if (
            type(safetensors["byte_count"]) is not int
            or safetensors["byte_count"] != _REGISTERED_SAFETENSORS_BYTES
        ):
            _fail("completed child safetensors byte count changed")
        child_safetensors_sha256.append(
            _require_hex(
                safetensors["sha256"],
                64,
                "completed child safetensors digest",
            )
        )
        expected_cpu_ids = _require_child_resources(
            child["resources"],
            expected_cpu_ids,
        )

    selected_child = cast(dict[str, Any], children[selected_ordinal])
    rerun_child = cast(dict[str, Any], children[3])
    if (
        child_safetensors_sha256[selected_ordinal] != child_safetensors_sha256[3]
        or selected_child["winner"] != rerun_child["winner"]
    ):
        _fail("selected and rerun child payload bindings differ")

    selected_winner = training_histories[selected_ordinal].winner
    expected_selection = {
        "ranked_training_ordinals": list(ranking),
        "selected_macro_f1_denominator": selected_winner.macro_f1.denominator,
        "selected_macro_f1_numerator": selected_winner.macro_f1.numerator,
        "selected_model_tensor_sha256": selected_winner.model_tensor_digest,
        "selected_seed": selected_seed,
        "selected_training_ordinal": selected_ordinal,
        "selected_validation_cross_entropy_float64_hex": (
            selected_winner.validation_cross_entropy_float64_hex
        ),
        "selected_winner_epoch": selected_winner.zero_based_epoch,
    }
    _require_exact_json_value(
        document["selection"],
        expected_selection,
        "completed selection",
    )
    expected_rerun = {
        "all_model_tensor_digests_identical": True,
        "all_validation_prediction_digests_identical": True,
        "history_byte_identical": True,
        "rerun_ordinal": 3,
        "safetensors_byte_identical": True,
        "selected_checkpoint_digest_identical": True,
        "selected_rank_identical": True,
        "selected_seed": selected_seed,
        "selected_training_ordinal": selected_ordinal,
        "validation_input_digest_identical": True,
    }
    _require_exact_json_value(
        document["rerun"],
        expected_rerun,
        "completed rerun",
    )

    computed_gates = _COMPLETED_GATES_COMPUTER(rerun_history)
    expected_gates = _COMPLETED_GATES_DOCUMENT(computed_gates)
    _require_exact_json_value(
        document["gates"],
        expected_gates,
        "completed gates",
    )
    expected_status = "pass" if computed_gates.all_passed else "gate_failure"
    if status != expected_status:
        _fail("completed status differs from all recomputed gates")

    if status == "pass":
        model_plan = plans[4]
        if (
            model_plan[4] != _REGISTERED_SAFETENSORS_BYTES
            or model_plan[5] != child_safetensors_sha256[3]
            or model_plan[7] != rerun_history.winner.model_tensor_digest
        ):
            _fail("selected model artifact differs from the rerun winner")


def _require_report_registration(
    document: dict[str, Any], registration: RegistrationPublicationFrame
) -> None:
    observed = _require_exact_object(
        document.get("registration"),
        frozenset(
            {
                "head_commit",
                "implementation_commit",
                "path",
                "registration_sha256",
                "source_bundle_sha256",
            }
        ),
        "final report registration",
    )
    expected = {
        "head_commit": registration[0],
        "implementation_commit": registration[1],
        "path": "configs/experiment-005-run.json",
        "registration_sha256": registration[2],
        "source_bundle_sha256": registration[3],
    }
    if observed != expected:
        _fail("final report registration differs from its publication frame")


def _require_report_matches_snapshot(snapshot: FinalPublicationSnapshot) -> None:
    status, registration, plans, report_path, report_mode, report_bytes = snapshot
    document = _parse_canonical_report(report_bytes)
    if status == "execution_failure":
        expected_document_keys = frozenset(
            {
                "artifacts",
                "checkpoint_reusable",
                "experiment",
                "failure",
                "predecessor",
                "publication",
                "registration",
                "schema_version",
                "scientific_protocol",
                "status",
            }
        )
    else:
        expected_document_keys = frozenset(
            {
                "artifacts",
                "checkpoint_reusable",
                "children",
                "experiment",
                "failure",
                "gates",
                "predecessor",
                "publication",
                "registration",
                "rerun",
                "schema_version",
                "scientific_protocol",
                "selection",
                "status",
            }
        )
    document = _require_exact_object(document, expected_document_keys, "final report")
    if (
        type(document.get("schema_version")) is not int
        or document.get("schema_version") != 1
        or document.get("experiment") != _EXPERIMENT
        or document.get("scientific_protocol") != _SCIENTIFIC_PROTOCOL
        or document.get("status") != status
    ):
        _fail("final report identity or status differs from its snapshot")
    _require_exact_json_value(
        document.get("predecessor"),
        _predecessor_document(),
        "final report predecessor",
    )
    expected_checkpoint_reusable = status == "pass"
    if document.get("checkpoint_reusable") is not expected_checkpoint_reusable:
        _fail("final report checkpoint reuse differs from its status")
    publication = _require_exact_object(
        document.get("publication"),
        frozenset({"report_mode", "report_path"}),
        "final report publication",
    )
    if publication != {"report_mode": report_mode, "report_path": report_path}:
        _fail("final report publication fields differ from its snapshot")
    _require_report_registration(document, registration)
    artifacts = document.get("artifacts")
    if type(artifacts) is not list:
        _fail("final report artifacts must be an exact array")
    artifact_values = cast(list[object], artifacts)
    if len(artifact_values) != len(plans):
        _fail("final report artifact count differs from its snapshot")
    for value, plan in zip(artifact_values, plans, strict=True):
        required = {
            "byte_count": plan[4],
            "kind": plan[1],
            "mode": plan[2],
            "path": plan[0],
            "sha256": plan[5],
            "source_child_ordinal": plan[8],
        }
        expected_artifact_keys = set(required)
        if plan[6] is not None:
            expected_artifact_keys.add("domain_sha256")
        if plan[7] is not None:
            expected_artifact_keys.add("model_tensor_sha256")
            expected_artifact_keys.add("tensor_count")
            expected_artifact_keys.add("value_count")
        artifact = _require_exact_object(
            value,
            frozenset(expected_artifact_keys),
            "final report artifact",
        )
        if (
            type(artifact.get("byte_count")) is not int
            or type(artifact.get("source_child_ordinal")) is not int
            or any(artifact.get(key) != expected for key, expected in required.items())
        ):
            _fail("final report artifact differs from its payload plan")
        if plan[6] is not None and artifact.get("domain_sha256") != plan[6]:
            _fail("final report history domain digest differs")
        if plan[7] is not None and artifact.get("model_tensor_sha256") != plan[7]:
            _fail("final report model digest differs")
        if plan[7] is not None and (
            type(artifact.get("tensor_count")) is not int
            or artifact.get("tensor_count") != _MODEL_TENSOR_COUNT
            or type(artifact.get("value_count")) is not int
            or artifact.get("value_count") != _MODEL_VALUE_COUNT
        ):
            _fail("final report model tensor counts differ")
    if status == "execution_failure":
        failure = _require_exact_object(
            document.get("failure"),
            frozenset({"code", "phase"}),
            "execution-failure fields",
        )
        pair = (failure["phase"], failure["code"])
        if pair not in EXECUTION_FAILURE_PAIRS:
            _fail("execution-failure report contains an unregistered pair")
    else:
        if document.get("failure") is not None:
            _fail("completed final report unexpectedly carries failure fields")
        _require_completed_report_semantics(document, status, plans)


def _require_payload_plan(
    value: object,
    *,
    expected_path: str,
    expected_kind: str,
    expected_ordinal: int,
) -> PublicationPayloadPlan:
    if type(value) is not tuple or len(cast(tuple[object, ...], value)) != 9:
        _fail("publication payload plan has an invalid exact shape")
    plan = cast(PublicationPayloadPlan, value)
    path, kind, mode, contents, byte_count, sha256, domain, model, ordinal = plan
    if (
        type(path) is not str
        or path != expected_path
        or type(kind) is not str
        or kind != expected_kind
        or type(mode) is not str
        or mode != _MODE_TEXT
        or type(contents) is not bytes
        or type(byte_count) is not int
        or byte_count != len(contents)
        or byte_count < 1
        or type(sha256) is not str
        or hashlib.sha256(contents).hexdigest() != sha256
        or type(ordinal) is not int
        or ordinal != expected_ordinal
    ):
        _fail("publication payload plan differs from its fixed identity")
    _require_hex(sha256, 64, "payload digest")
    history = expected_kind in {"training_history", "selected_rerun_history"}
    if history:
        _require_hex(domain, 64, "history domain digest")
        if model is not None:
            _fail("history payload unexpectedly carries a model digest")
    else:
        if domain is not None:
            _fail("selected model unexpectedly carries a history domain digest")
        _require_hex(model, 64, "model tensor digest")
    return plan


def _require_publication_snapshot(value: object) -> FinalPublicationSnapshot:
    if type(value) is not tuple or len(cast(tuple[object, ...], value)) != 6:
        _fail("final publication snapshot has an invalid exact shape")
    fields = cast(tuple[object, object, object, object, object, object], value)
    status = fields[0]
    if type(status) is not str or status not in {
        "pass",
        "gate_failure",
        "execution_failure",
    }:
        _fail("final publication status is not registered")
    registration = _require_registration_frame(fields[1])
    if type(fields[2]) is not tuple:
        _fail("final publication payload plans are not an exact tuple")
    raw_plans = cast(tuple[object, ...], fields[2])
    if status == "pass":
        expected_paths: tuple[str, ...] = _PAYLOAD_PATHS
        expected_kinds: tuple[str, ...] = _PASS_KINDS
        expected_ordinals: tuple[int, ...] = (0, 1, 2, 3, 3)
    elif status == "gate_failure":
        expected_paths = _PAYLOAD_PATHS[:4]
        expected_kinds = _GATE_KINDS
        expected_ordinals = (0, 1, 2, 3)
    else:
        expected_paths = ()
        expected_kinds = ()
        expected_ordinals = ()
    if len(raw_plans) != len(expected_paths):
        _fail("final publication payload set differs from its status")
    plans = tuple(
        _require_payload_plan(
            raw,
            expected_path=expected_paths[index],
            expected_kind=expected_kinds[index],
            expected_ordinal=expected_ordinals[index],
        )
        for index, raw in enumerate(raw_plans)
    )
    report_path = fields[3]
    report_mode = fields[4]
    report_bytes = fields[5]
    if (
        type(report_path) is not str
        or report_path != _REPORT_PATH
        or type(report_mode) is not str
        or report_mode != _MODE_TEXT
        or type(report_bytes) is not bytes
        or not 0 < len(report_bytes) <= _REPORT_BYTES_MAXIMUM
        or len(report_bytes) + sum(plan[4] for plan in plans)
        > _PUBLICATION_BYTES_MAXIMUM
    ):
        _fail("final report path, mode, type, or byte budget is invalid")
    snapshot: FinalPublicationSnapshot = (
        status,
        registration,
        plans,
        report_path,
        report_mode,
        report_bytes,
    )
    _require_report_matches_snapshot(snapshot)
    return snapshot


def _required_os_constant(name: str) -> int:
    value = getattr(os, name, None)
    if type(value) is not int:
        _fail("secure filesystem constants are unavailable")
    return value


def _require_secure_filesystem_support() -> None:
    for name in ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK"):
        _required_os_constant(name)
    if not all(route in os.supports_dir_fd for route in (os.open, os.stat, os.unlink)):
        _fail("descriptor-relative filesystem operations are unavailable")


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | _required_os_constant("O_DIRECTORY")
        | _required_os_constant("O_NOFOLLOW")
        | _required_os_constant("O_CLOEXEC")
    )


def _identity(source: os.stat_result) -> tuple[int, int]:
    return source.st_dev, source.st_ino


def _open_directory(name: str, *, dir_fd: int | None = None) -> _PinnedDirectory:
    try:
        if dir_fd is None:
            descriptor = os.open(name, _directory_flags())
        else:
            descriptor = os.open(name, _directory_flags(), dir_fd=dir_fd)
    except OSError as error:
        raise Experiment005FinalPublicationError(
            "registered publication directory cannot be securely opened"
        ) from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISDIR(opened.st_mode):
            _fail("registered publication parent is not a directory")
        return _PinnedDirectory(
            name=name,
            descriptor=descriptor,
            identity=_identity(opened),
        )
    except BaseException:
        os.close(descriptor)
        raise


def _require_layout(layout: _PublicationLayout) -> None:
    if (
        type(layout) is not _PublicationLayout
        or type(layout.repository_root) is not str
        or not layout.repository_root.startswith("/")
        or layout.repository_root.rstrip("/") != layout.repository_root
        or type(layout.report_directory) is not str
        or layout.report_directory != _REPORT_DIRECTORY
        or type(layout.model_directory) is not str
        or layout.model_directory != _MODEL_DIRECTORY
    ):
        _fail("publication layout differs from its fixed directory shape")


def _pin_layout(layout: _PublicationLayout) -> _PinnedLayout:
    _require_layout(layout)
    _require_secure_filesystem_support()
    root = _open_directory(layout.repository_root)
    reports: _PinnedDirectory | None = None
    models: _PinnedDirectory | None = None
    try:
        reports = _open_directory(layout.report_directory, dir_fd=root.descriptor)
        models = _open_directory(layout.model_directory, dir_fd=root.descriptor)
        if reports.identity == models.identity:
            _fail("registered publication parents identify the same directory")
        result = _PinnedLayout(layout=layout, root=root, reports=reports, models=models)
        _require_pinned_layout(result)
        return result
    except BaseException:
        if models is not None:
            os.close(models.descriptor)
        if reports is not None:
            os.close(reports.descriptor)
        os.close(root.descriptor)
        raise


def _require_pinned_directory(directory: _PinnedDirectory) -> None:
    opened = os.fstat(directory.descriptor)
    if not stat.S_ISDIR(opened.st_mode) or _identity(opened) != directory.identity:
        _fail("pinned publication directory identity changed")


def _require_pinned_layout(pinned: _PinnedLayout) -> None:
    if type(pinned) is not _PinnedLayout:
        raise TypeError("pinned layout must be an exact _PinnedLayout")
    _require_pinned_directory(pinned.root)
    _require_pinned_directory(pinned.reports)
    _require_pinned_directory(pinned.models)
    try:
        root_named = os.stat(pinned.layout.repository_root, follow_symlinks=False)
        report_named = os.stat(
            pinned.layout.report_directory,
            dir_fd=pinned.root.descriptor,
            follow_symlinks=False,
        )
        model_named = os.stat(
            pinned.layout.model_directory,
            dir_fd=pinned.root.descriptor,
            follow_symlinks=False,
        )
    except OSError as error:
        raise Experiment005FinalPublicationError(
            "registered publication directory name cannot be verified"
        ) from error
    if (
        not stat.S_ISDIR(root_named.st_mode)
        or _identity(root_named) != pinned.root.identity
        or not stat.S_ISDIR(report_named.st_mode)
        or _identity(report_named) != pinned.reports.identity
        or not stat.S_ISDIR(model_named.st_mode)
        or _identity(model_named) != pinned.models.identity
    ):
        _fail("registered publication directory name changed")


def _close_pinned_layout(pinned: _PinnedLayout) -> None:
    failures: list[OSError] = []
    for directory in (pinned.models, pinned.reports, pinned.root):
        try:
            os.close(directory.descriptor)
        except OSError as error:
            failures.append(error)
    if failures:
        raise Experiment005FinalPublicationError(
            "registered publication directory close failed"
        ) from failures[0]


def _parent_and_leaf(pinned: _PinnedLayout, path: str) -> tuple[_PinnedDirectory, str]:
    if type(path) is not str or path not in _MANAGED_PATHS:
        _fail("publication path is not one fixed managed path")
    parent_name, separator, leaf = path.partition("/")
    if separator != "/" or not leaf or "/" in leaf or leaf in {".", ".."}:
        _fail("publication path has an invalid fixed leaf")
    if parent_name == _REPORT_DIRECTORY:
        return pinned.reports, leaf
    if parent_name == _MODEL_DIRECTORY:
        return pinned.models, leaf
    _fail("publication path has an invalid fixed parent")


def _leaf_exists(parent: _PinnedDirectory, leaf: str) -> bool:
    try:
        os.stat(leaf, dir_fd=parent.descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    except OSError as error:
        raise Experiment005FinalPublicationError(
            "managed publication leaf cannot be inspected"
        ) from error
    return True


def _require_fresh_namespace(pinned: _PinnedLayout) -> None:
    _require_pinned_layout(pinned)
    for path in _MANAGED_PATHS:
        parent, leaf = _parent_and_leaf(pinned, path)
        if _leaf_exists(parent, leaf):
            _fail("managed publication namespace is not fresh")
    for parent in (pinned.reports, pinned.models):
        try:
            entries = os.listdir(parent.descriptor)
        except OSError as error:
            raise Experiment005FinalPublicationError(
                "publication parent cannot be scanned for crash temporaries"
            ) from error
        if any(
            type(entry) is not str or entry.startswith(_TEMP_PREFIX)
            for entry in entries
        ):
            _fail("publication crash temporary or invalid directory entry exists")
    _require_pinned_layout(pinned)


def _file_plans(snapshot: FinalPublicationSnapshot) -> tuple[_FilePlan, ...]:
    _status, registration, payloads, report_path, _mode, report = snapshot
    del registration
    result = [
        _FilePlan(
            path=plan[0],
            contents=plan[3],
            byte_count=plan[4],
            sha256=plan[5],
            slot=_MANAGED_PATHS.index(plan[0]),
        )
        for plan in payloads
    ]
    result.append(
        _FilePlan(
            path=report_path,
            contents=report,
            byte_count=len(report),
            sha256=hashlib.sha256(report).hexdigest(),
            slot=_MANAGED_PATHS.index(report_path),
        )
    )
    return tuple(result)


def _temporary_leaf(registration_sha256: str, plan: _FilePlan) -> str:
    return f"{_TEMP_PREFIX}{registration_sha256}-{plan.slot:02d}.tmp"


def _write_all(descriptor: int, contents: bytes) -> None:
    view = memoryview(contents)
    offset = 0
    while offset < len(view):
        try:
            written = os.pwrite(descriptor, view[offset:], offset)
        except InterruptedError:
            continue
        if written <= 0:
            _fail("publication temporary write made invalid progress")
        offset += written


def _read_exact(descriptor: int, byte_count: int) -> bytes:
    result = bytearray()
    while len(result) < byte_count:
        try:
            chunk = os.pread(descriptor, byte_count - len(result), len(result))
        except InterruptedError:
            continue
        if not chunk:
            _fail("publication file is truncated")
        result.extend(chunk)
    return bytes(result)


def _stat_fingerprint(source: os.stat_result) -> tuple[int, ...]:
    return (
        source.st_dev,
        source.st_ino,
        source.st_mode,
        source.st_size,
        source.st_nlink,
        source.st_uid,
        source.st_mtime_ns,
        source.st_ctime_ns,
    )


def _require_file_stat(
    source: os.stat_result,
    plan: _FilePlan,
    *,
    links: int,
) -> None:
    if (
        not stat.S_ISREG(source.st_mode)
        or stat.S_IMODE(source.st_mode) != _FINAL_MODE
        or source.st_size != plan.byte_count
        or source.st_nlink != links
        or source.st_uid != os.geteuid()
    ):
        _fail("publication inode mode, size, owner, or link count differs")


def _verify_open_file(
    descriptor: int, plan: _FilePlan, *, links: int
) -> os.stat_result:
    before = os.fstat(descriptor)
    _require_file_stat(before, plan, links=links)
    observed = _read_exact(descriptor, plan.byte_count)
    if os.pread(descriptor, 1, plan.byte_count):
        _fail("publication file has trailing bytes")
    after = os.fstat(descriptor)
    if _stat_fingerprint(before) != _stat_fingerprint(after):
        _fail("publication inode changed while it was verified")
    if observed != plan.contents or hashlib.sha256(observed).hexdigest() != plan.sha256:
        _fail("publication file bytes or digest differ")
    return after


def _temporary_flags() -> int:
    return (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | _required_os_constant("O_NOFOLLOW")
        | _required_os_constant("O_CLOEXEC")
        | _required_os_constant("O_NONBLOCK")
    )


def _safe_unlink_owned_temporary(prepared: _PreparedFile) -> None:
    leaf = prepared.temporary_leaf
    if leaf is None:
        return
    try:
        opened = os.fstat(prepared.descriptor)
        named = os.stat(
            leaf,
            dir_fd=prepared.parent.descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    if (
        not stat.S_ISREG(opened.st_mode)
        or _identity(opened) != prepared.identity
        or _identity(named) != prepared.identity
        or opened.st_nlink != 1
        or named.st_nlink != 1
    ):
        _fail("publication temporary ownership changed before cleanup")
    os.unlink(leaf, dir_fd=prepared.parent.descriptor)
    prepared.temporary_leaf = None
    os.fsync(prepared.parent.descriptor)


def _prepare_file(
    pinned: _PinnedLayout,
    registration_sha256: str,
    plan: _FilePlan,
) -> _PreparedFile:
    parent, destination_leaf = _parent_and_leaf(pinned, plan.path)
    temporary_leaf = _temporary_leaf(registration_sha256, plan)
    descriptor = -1
    prepared: _PreparedFile | None = None
    try:
        descriptor = os.open(
            temporary_leaf,
            _temporary_flags(),
            _TEMPORARY_MODE,
            dir_fd=parent.descriptor,
        )
        created = os.fstat(descriptor)
        if (
            not stat.S_ISREG(created.st_mode)
            or created.st_nlink != 1
            or created.st_size != 0
            or created.st_uid != os.geteuid()
        ):
            _fail("new publication temporary is not one owned empty regular inode")
        os.fchmod(descriptor, _TEMPORARY_MODE)
        if stat.S_IMODE(os.fstat(descriptor).st_mode) != _TEMPORARY_MODE:
            _fail("publication temporary did not acquire mode 0600")
        prepared = _PreparedFile(
            plan=plan,
            parent=parent,
            destination_leaf=destination_leaf,
            temporary_leaf=temporary_leaf,
            descriptor=descriptor,
            identity=_identity(created),
        )
        _write_all(descriptor, plan.contents)
        os.ftruncate(descriptor, plan.byte_count)
        os.fsync(descriptor)
        os.fchmod(descriptor, _FINAL_MODE)
        os.fsync(descriptor)
        verified = _verify_open_file(descriptor, plan, links=1)
        if _identity(verified) != prepared.identity:
            _fail("publication temporary inode identity changed")
        named = os.stat(
            temporary_leaf,
            dir_fd=parent.descriptor,
            follow_symlinks=False,
        )
        if _identity(named) != prepared.identity:
            _fail("publication temporary name changed")
        return prepared
    except BaseException:
        cleanup_failure: BaseException | None = None
        if prepared is not None:
            try:
                _safe_unlink_owned_temporary(prepared)
            except BaseException as error:
                cleanup_failure = error
        elif descriptor >= 0:
            try:
                opened = os.fstat(descriptor)
                named = os.stat(
                    temporary_leaf,
                    dir_fd=parent.descriptor,
                    follow_symlinks=False,
                )
                if _identity(opened) == _identity(named) and opened.st_nlink == 1:
                    os.unlink(temporary_leaf, dir_fd=parent.descriptor)
                    os.fsync(parent.descriptor)
            except FileNotFoundError:
                pass
            except BaseException as error:
                cleanup_failure = error
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError as error:
                if cleanup_failure is None:
                    cleanup_failure = error
        if cleanup_failure is not None:
            raise Experiment005FinalPublicationError(
                "publication temporary preparation failed with cleanup errors"
            ) from cleanup_failure
        raise


def _commit_file(
    pinned: _PinnedLayout,
    prepared: _PreparedFile,
    state: _CommitState,
) -> None:
    _require_pinned_layout(pinned)
    leaf = prepared.temporary_leaf
    if leaf is None or prepared.published:
        _fail("publication temporary was already consumed")
    source = _verify_open_file(prepared.descriptor, prepared.plan, links=1)
    if _identity(source) != prepared.identity:
        _fail("publication source inode changed before hard link")
    try:
        _link_descriptor_no_overwrite(
            prepared.descriptor,
            prepared.parent.descriptor,
            prepared.destination_leaf,
        )
    except FileExistsError as error:
        raise Experiment005FinalPublicationError(
            "publication destination appeared before its no-overwrite commit"
        ) from error
    state.started = True
    source_named = os.stat(
        leaf,
        dir_fd=prepared.parent.descriptor,
        follow_symlinks=False,
    )
    destination = os.stat(
        prepared.destination_leaf,
        dir_fd=prepared.parent.descriptor,
        follow_symlinks=False,
    )
    opened = os.fstat(prepared.descriptor)
    if (
        _identity(source_named) != prepared.identity
        or _identity(destination) != prepared.identity
        or _identity(opened) != prepared.identity
    ):
        _fail("hard-linked publication names do not identify the built inode")
    _require_file_stat(source_named, prepared.plan, links=2)
    _require_file_stat(destination, prepared.plan, links=2)
    _require_file_stat(opened, prepared.plan, links=2)
    os.unlink(leaf, dir_fd=prepared.parent.descriptor)
    prepared.temporary_leaf = None
    os.fsync(prepared.parent.descriptor)
    opened_after = _verify_open_file(prepared.descriptor, prepared.plan, links=1)
    destination_after = os.stat(
        prepared.destination_leaf,
        dir_fd=prepared.parent.descriptor,
        follow_symlinks=False,
    )
    if (
        _identity(opened_after) != prepared.identity
        or _identity(destination_after) != prepared.identity
    ):
        _fail("published destination inode changed after temporary unlink")
    _require_file_stat(destination_after, prepared.plan, links=1)
    prepared.published = True
    _require_pinned_layout(pinned)


def _link_descriptor_no_overwrite(
    source_descriptor: int,
    destination_parent_descriptor: int,
    destination_leaf: str,
    /,
) -> None:
    """Hard-link the already-verified inode, never a replaceable source name."""

    if (
        type(source_descriptor) is not int
        or source_descriptor < 0
        or type(destination_parent_descriptor) is not int
        or destination_parent_descriptor < 0
        or type(destination_leaf) is not str
        or not destination_leaf
        or "/" in destination_leaf
        or "\0" in destination_leaf
    ):
        _fail("descriptor hard-link arguments are invalid")
    try:
        destination_bytes = destination_leaf.encode("ascii", errors="strict")
    except UnicodeError as error:
        raise Experiment005FinalPublicationError(
            "descriptor hard-link destination is not ASCII"
        ) from error
    ctypes.set_errno(0)
    result = _LINKAT(
        source_descriptor,
        b"",
        destination_parent_descriptor,
        destination_bytes,
        _AT_EMPTY_PATH,
    )
    if result == 0:
        return
    observed_errno = ctypes.get_errno()
    if observed_errno == errno.EEXIST:
        raise FileExistsError(
            observed_errno,
            "publication destination already exists",
            destination_leaf,
        )
    raise OSError(
        observed_errno,
        "descriptor-relative publication hard link failed",
        destination_leaf,
    )


def _destination_flags() -> int:
    return (
        os.O_RDONLY
        | _required_os_constant("O_NOFOLLOW")
        | _required_os_constant("O_CLOEXEC")
        | _required_os_constant("O_NONBLOCK")
    )


def _verify_published_destination(
    pinned: _PinnedLayout,
    prepared: _PreparedFile,
) -> None:
    _require_pinned_layout(pinned)
    if not prepared.published or prepared.temporary_leaf is not None:
        _fail("publication verification observed an incomplete file")
    try:
        descriptor = os.open(
            prepared.destination_leaf,
            _destination_flags(),
            dir_fd=prepared.parent.descriptor,
        )
    except OSError as error:
        raise Experiment005FinalPublicationError(
            "published destination cannot be securely reopened"
        ) from error
    try:
        verified = _verify_open_file(descriptor, prepared.plan, links=1)
        if _identity(verified) != prepared.identity:
            _fail("published destination no longer identifies its built inode")
    finally:
        os.close(descriptor)


def _require_report_commit_boundary(
    pinned: _PinnedLayout,
    prepared_files: tuple[_PreparedFile, ...],
    snapshot: FinalPublicationSnapshot,
) -> None:
    """Revalidate the complete payload prefix immediately before its marker."""

    _require_pinned_layout(pinned)
    if not prepared_files:
        _fail("final publication has no prepared report")
    report = prepared_files[-1]
    if (
        report.plan.path != snapshot[3]
        or report.published
        or report.temporary_leaf is None
    ):
        _fail("final report is not the one uncommitted publication suffix")
    for prepared in prepared_files[:-1]:
        _verify_published_destination(pinned, prepared)

    published_paths = {prepared.plan.path for prepared in prepared_files[:-1]}
    for path in _MANAGED_PATHS:
        if path not in published_paths:
            parent, leaf = _parent_and_leaf(pinned, path)
            if _leaf_exists(parent, leaf):
                _fail("managed destination appeared before final report commit")

    expected_temporary = report.temporary_leaf
    for parent in (pinned.reports, pinned.models):
        try:
            observed = {
                entry
                for entry in os.listdir(parent.descriptor)
                if entry.startswith(_TEMP_PREFIX)
            }
        except OSError as error:
            raise Experiment005FinalPublicationError(
                "publication parent cannot be scanned before final report commit"
            ) from error
        expected = (
            {expected_temporary}
            if parent.descriptor == report.parent.descriptor
            else set()
        )
        if observed != expected:
            _fail("publication temporary set changed before final report commit")

    source = _verify_open_file(report.descriptor, report.plan, links=1)
    try:
        named = os.stat(
            expected_temporary,
            dir_fd=report.parent.descriptor,
            follow_symlinks=False,
        )
    except OSError as error:
        raise Experiment005FinalPublicationError(
            "final report temporary cannot be securely reverified"
        ) from error
    if _identity(source) != report.identity or _identity(named) != report.identity:
        _fail("final report temporary identity changed before commit")
    _require_pinned_layout(pinned)


def _postverify(
    pinned: _PinnedLayout,
    prepared_files: tuple[_PreparedFile, ...],
    snapshot: FinalPublicationSnapshot,
) -> None:
    _require_pinned_layout(pinned)
    expected_paths = {prepared.plan.path for prepared in prepared_files}
    for prepared in prepared_files:
        _verify_published_destination(pinned, prepared)
    for path in _MANAGED_PATHS:
        if path not in expected_paths:
            parent, leaf = _parent_and_leaf(pinned, path)
            if _leaf_exists(parent, leaf):
                _fail("forbidden managed destination appeared during publication")
    for parent in (pinned.reports, pinned.models):
        entries = os.listdir(parent.descriptor)
        if any(entry.startswith(_TEMP_PREFIX) for entry in entries):
            _fail("publication crash temporary remains after completed commit")
    report_parent, report_leaf = _parent_and_leaf(pinned, snapshot[3])
    report = os.stat(
        report_leaf,
        dir_fd=report_parent.descriptor,
        follow_symlinks=False,
    )
    if not stat.S_ISREG(report.st_mode) or stat.S_IMODE(report.st_mode) != _FINAL_MODE:
        _fail("final report commit marker is not one immutable regular file")
    if not prepared_files or prepared_files[-1].plan.path != snapshot[3]:
        _fail("final report is not the last prepared publication file")
    _require_pinned_layout(pinned)
    _verify_published_destination(pinned, prepared_files[-1])


def _cleanup_prelink_temporaries(
    prepared_files: tuple[_PreparedFile, ...],
) -> None:
    failures: list[BaseException] = []
    parents: dict[int, _PinnedDirectory] = {}
    for prepared in prepared_files:
        parents[prepared.parent.descriptor] = prepared.parent
        try:
            _safe_unlink_owned_temporary(prepared)
        except BaseException as error:
            failures.append(error)
    for parent in parents.values():
        try:
            os.fsync(parent.descriptor)
        except BaseException as error:
            failures.append(error)
    if failures:
        raise Experiment005FinalPublicationError(
            "prelink publication temporary cleanup failed"
        ) from failures[0]


def _close_prepared_files(prepared_files: tuple[_PreparedFile, ...]) -> None:
    failures: list[OSError] = []
    for prepared in prepared_files:
        try:
            os.close(prepared.descriptor)
        except OSError as error:
            failures.append(error)
    if failures:
        raise Experiment005FinalPublicationError(
            "publication temporary descriptor close failed"
        ) from failures[0]


def _publish_snapshot(
    snapshot_value: object,
    layout: _PublicationLayout,
    final_authority_check: _FinalAuthorityCheck,
    /,
) -> None:
    """Private hermetic seam retaining the exact production filesystem protocol."""

    snapshot = _require_publication_snapshot(snapshot_value)
    if type(final_authority_check) is not FunctionType:
        raise TypeError("final authority check must be an exact function")
    pinned = _pin_layout(layout)
    prepared: tuple[_PreparedFile, ...] = ()
    commit_state = _CommitState()
    primary: BaseException | None = None
    try:
        _require_fresh_namespace(pinned)
        authority_result = final_authority_check()
        if authority_result is not None:
            _fail("final authority check returned an unexpected value")
        _require_fresh_namespace(pinned)
        registration_sha256 = snapshot[1][2]
        built: list[_PreparedFile] = []
        for plan in _file_plans(snapshot):
            built.append(_prepare_file(pinned, registration_sha256, plan))
            prepared = tuple(built)
        _require_pinned_layout(pinned)
        for index, item in enumerate(prepared):
            if index == len(prepared) - 1:
                _require_report_commit_boundary(pinned, prepared, snapshot)
            _commit_file(pinned, item, commit_state)
        _postverify(pinned, prepared, snapshot)
    except BaseException as error:
        primary = error
        if not commit_state.started:
            try:
                _cleanup_prelink_temporaries(prepared)
            except BaseException as cleanup_error:
                raise Experiment005FinalPublicationError(
                    "final publication failed before commit with cleanup errors"
                ) from cleanup_error
        raise
    finally:
        close_failure: BaseException | None = None
        try:
            _close_prepared_files(prepared)
        except BaseException as error:
            close_failure = error
        try:
            _close_pinned_layout(pinned)
        except BaseException as error:
            if close_failure is None:
                close_failure = error
        if close_failure is not None and primary is None:
            raise close_failure


@dataclass(frozen=True, slots=True)
class _FunctionAuthority:
    function: FunctionType
    code: CodeType
    defaults: tuple[object, ...] | None
    default_values: tuple[object, ...] | None
    kwdefaults: dict[str, object] | None
    kwdefault_values: tuple[tuple[str, object], ...] | None
    closure: tuple[tuple[object, object], ...]


def _function_authority(function: FunctionType) -> _FunctionAuthority:
    closure = function.__closure__ or ()
    defaults = function.__defaults__
    kwdefaults = function.__kwdefaults__
    return _FunctionAuthority(
        function=function,
        code=function.__code__,
        defaults=defaults,
        default_values=None if defaults is None else tuple(defaults),
        kwdefaults=kwdefaults,
        kwdefault_values=(None if kwdefaults is None else tuple(kwdefaults.items())),
        closure=tuple((cell, cell.cell_contents) for cell in closure),
    )


def _require_function_authority(authority: _FunctionAuthority) -> None:
    function = authority.function
    closure = function.__closure__ or ()
    defaults = function.__defaults__
    kwdefaults = function.__kwdefaults__
    if (
        type(function) is not FunctionType
        or function.__code__ is not authority.code
        or defaults is not authority.defaults
        or (
            defaults is not None
            and (
                authority.default_values is None
                or len(defaults) != len(authority.default_values)
                or any(
                    value is not authority.default_values[index]
                    for index, value in enumerate(defaults)
                )
            )
        )
        or kwdefaults is not authority.kwdefaults
        or (
            kwdefaults is not None
            and (
                authority.kwdefault_values is None
                or tuple(kwdefaults)
                != tuple(name for name, _value in authority.kwdefault_values)
                or any(
                    kwdefaults[name] is not value
                    for name, value in authority.kwdefault_values
                )
            )
        )
        or len(closure) != len(authority.closure)
        or any(
            cell is not authority.closure[index][0]
            or cell.cell_contents is not authority.closure[index][1]
            for index, cell in enumerate(closure)
        )
    ):
        _fail("registered publication function authority changed")


def _make_registered_publisher() -> Callable[[VerifiedRunRegistration, object], None]:
    registration_type = VerifiedRunRegistration
    completed_type = _completed_evidence.FinalCompletedEvidence
    failure_type = FinalExecutionFailureEvidence
    completed_snapshot = cast(
        _CompletedSnapshotRoute, _completed_evidence.snapshot_completed_publication
    )
    failure_snapshot = snapshot_execution_failure_publication
    reverify = reverify_verified_run_registration
    publish_core = _publish_snapshot
    layout = _PublicationLayout(_CANONICAL_REPOSITORY_ROOT)
    snapshot_validator = _require_publication_snapshot
    registration_validator = _require_registration_frame
    require_function_authority = _require_function_authority
    authority_checker_code = require_function_authority.__code__
    authority_checker_defaults = require_function_authority.__defaults__
    authority_checker_default_values = (
        None
        if authority_checker_defaults is None
        else tuple(authority_checker_defaults)
    )
    authority_checker_kwdefaults = require_function_authority.__kwdefaults__
    authority_checker_kwdefault_values = (
        None
        if authority_checker_kwdefaults is None
        else tuple(authority_checker_kwdefaults.items())
    )
    authority_checker_closure = tuple(require_function_authority.__closure__ or ())
    authority_checker_closure_values = tuple(
        cell.cell_contents for cell in authority_checker_closure
    )
    exact_type = type
    error_type = Experiment005FinalPublicationError
    module_globals = globals()
    os_module = os
    threading_module = threading
    ctypes_module = ctypes
    errno_module = errno
    hashlib_module = hashlib
    json_module = json
    stat_module = stat
    builtins_module = builtins
    run_authority_module = _run_authority
    tuple_route = tuple
    sorted_route = sorted
    any_route = any
    enumerate_route = enumerate
    getattr_route = getattr
    frozenset_route = frozenset
    type_error = TypeError
    owner_process = _OWNER_PROCESS
    owner_thread = threading_module.get_ident()
    if exact_type(owner_thread) is not int or owner_thread < 1:
        raise RuntimeError("registered final publisher thread identity is invalid")
    attempted = False
    in_flight = False
    lock = threading.Lock()
    route_names = (
        "snapshot_execution_failure_publication",
        "reverify_verified_run_registration",
        "_publish_snapshot",
    )
    routes = (failure_snapshot, reverify, publish_core)
    verified_state_route = _run_authority._verified_state

    def capture_closure_authorities(
        root: FunctionType,
    ) -> tuple[_FunctionAuthority, ...]:
        pending = [root]
        seen: set[int] = set()
        captured: list[_FunctionAuthority] = []
        while pending:
            function = pending.pop()
            identity = id(function)
            if identity in seen:
                continue
            seen.add(identity)
            captured.append(_function_authority(function))
            for cell in function.__closure__ or ():
                try:
                    value = cell.cell_contents
                except ValueError:
                    continue
                if exact_type(value) is FunctionType:
                    pending.append(cast(FunctionType, value))
        return tuple_route(captured)

    closure_route_authorities = tuple_route(
        authority
        for root in (
            cast(FunctionType, completed_snapshot),
            *(cast(FunctionType, route) for route in routes),
            cast(FunctionType, verified_state_route),
        )
        for authority in capture_closure_authorities(root)
    )
    verified_state_type = _run_authority._VerifiedState
    verified_state_namespace = verified_state_type.__dict__
    verified_state_namespace_names = tuple_route(sorted_route(verified_state_namespace))
    verified_state_namespace_values = tuple_route(
        verified_state_namespace[name] for name in verified_state_namespace_names
    )
    verified_state_field_names = (
        "head_commit",
        "implementation_commit",
        "registration_sha256",
        "source_bundle_sha256",
    )
    verified_state_descriptors = tuple_route(
        verified_state_namespace[name] for name in verified_state_field_names
    )
    if any_route(
        exact_type(value) is not MemberDescriptorType
        for value in verified_state_descriptors
    ):
        raise RuntimeError("run-registration state slots are unavailable")
    layout_type = _PublicationLayout
    layout_field_names = (
        "repository_root",
        "report_directory",
        "model_directory",
    )
    layout_descriptors = tuple_route(
        layout_type.__dict__[name] for name in layout_field_names
    )
    if any_route(
        exact_type(value) is not MemberDescriptorType for value in layout_descriptors
    ):
        raise RuntimeError("publication layout slots are unavailable")
    layout_frame = (
        _CANONICAL_REPOSITORY_ROOT,
        _REPORT_DIRECTORY,
        _MODEL_DIRECTORY,
    )
    critical_function_items = tuple_route(
        (name, value)
        for name, value in module_globals.items()
        if name != "_make_registered_publisher" and exact_type(value) is FunctionType
    )
    critical_function_authorities = tuple_route(
        _function_authority(cast(FunctionType, value))
        for _name, value in critical_function_items
    )
    constant_names = (
        "EXECUTION_FAILURE_PAIRS",
        "_CANONICAL_REPOSITORY_ROOT",
        "_REPORT_DIRECTORY",
        "_MODEL_DIRECTORY",
        "_REPORT_PATH",
        "_TRAINING_HISTORY_PATHS",
        "_RERUN_HISTORY_PATH",
        "_MODEL_PATH",
        "_PAYLOAD_PATHS",
        "_MANAGED_PATHS",
        "_PASS_KINDS",
        "_GATE_KINDS",
        "_MODE_TEXT",
        "_FINAL_MODE",
        "_TEMPORARY_MODE",
        "_REPORT_BYTES_MAXIMUM",
        "_PUBLICATION_BYTES_MAXIMUM",
        "_MODEL_TENSOR_COUNT",
        "_MODEL_VALUE_COUNT",
        "_TEMP_PREFIX",
        "_EXPERIMENT",
        "_SCIENTIFIC_PROTOCOL",
        "_PREDECESSOR_EXECUTION_PROTOCOL_COMMIT",
        "_PREDECESSOR_EXECUTION_PROTOCOL_SHA256",
        "_PREDECESSOR_IMPLEMENTATION_COMMIT",
        "_PREDECESSOR_PROTOCOL_PROOF_SHA256",
        "_PREDECESSOR_SHARED_AUTHORITY_SHA256",
        "_PREDECESSOR_INCIDENT_COMMIT",
        "_PREDECESSOR_INCIDENT_SHA256",
        "_PREDECESSOR_PREFLIGHT_BOUNDARY_COMMIT",
        "_PREDECESSOR_PREFLIGHT_BOUNDARY_PARENT",
        "_PREDECESSOR_CANONICAL_JSON",
        "_LOWER_HEX",
        "_REGISTERED_SEEDS",
        "_SEED_ROLE",
        "_RERUN_ROLE",
        "_EPOCH_COUNT",
        "_REGISTERED_SAFETENSORS_BYTES",
        "_CHILD_ENVELOPE_BYTES_MAXIMUM",
        "_CHILD_WALL_NANOSECONDS_MAXIMUM",
        "_CHILD_RSS_BYTES_MAXIMUM",
        "_CHILD_OUTPUT_BYTES_MAXIMUM",
        "_UINT32_MAXIMUM",
        "_OWNER_PROCESS",
        "_AT_EMPTY_PATH",
        "_LIBC",
        "_LINKAT",
    )
    constant_values = tuple_route(module_globals[name] for name in constant_names)
    class_names = (
        "Experiment005FinalPublicationError",
        "FinalExecutionFailureEvidence",
        "_FailureState",
        "_PublicationLayout",
        "_PinnedDirectory",
        "_PinnedLayout",
        "_FilePlan",
        "_PreparedFile",
        "_CommitState",
        "_FunctionAuthority",
    )
    classes = tuple_route(
        cast(type[object], module_globals[name]) for name in class_names
    )
    class_frames = tuple_route(
        (
            tuple_route(sorted_route(class_type.__dict__)),
            tuple_route(
                class_type.__dict__[name] for name in sorted_route(class_type.__dict__)
            ),
        )
        for class_type in classes
    )
    class_function_authorities = tuple_route(
        _function_authority(cast(FunctionType, value))
        for class_type in classes
        for value in class_type.__dict__.values()
        if exact_type(value) is FunctionType
    )
    os_route_names = (
        "close",
        "fchmod",
        "fstat",
        "fsync",
        "ftruncate",
        "geteuid",
        "getpid",
        "link",
        "listdir",
        "open",
        "pread",
        "pwrite",
        "stat",
        "unlink",
    )
    os_routes = tuple_route(getattr_route(os_module, name) for name in os_route_names)
    os_constant_names = (
        "O_CLOEXEC",
        "O_CREAT",
        "O_DIRECTORY",
        "O_EXCL",
        "O_NOFOLLOW",
        "O_NONBLOCK",
        "O_RDONLY",
        "O_RDWR",
    )
    os_constant_values = tuple_route(
        getattr_route(os_module, name) for name in os_constant_names
    )
    if any_route(
        exact_type(value) is not int or value < 0 for value in os_constant_values
    ):
        raise RuntimeError("secure publication OS constants are unavailable")
    hashlib_route_names = ("sha256",)
    hashlib_routes = tuple_route(
        getattr_route(hashlib_module, name) for name in hashlib_route_names
    )
    json_route_names = (
        "JSONDecodeError",
        "JSONDecoder",
        "JSONEncoder",
        "dumps",
        "loads",
    )
    json_routes = tuple_route(
        getattr_route(json_module, name) for name in json_route_names
    )
    json_scanner_module = cast(Any, getattr_route(json_module, "scanner"))
    json_dependency_modules = (
        json_module,
        json_module.decoder,
        json_module.encoder,
        json_scanner_module,
    )
    json_dependency_namespace_frames = tuple_route(
        (
            tuple_route(sorted_route(module.__dict__)),
            tuple_route(
                module.__dict__[name] for name in sorted_route(module.__dict__)
            ),
        )
        for module in json_dependency_modules
    )
    json_dependency_dict_frames = tuple_route(
        (
            value,
            tuple_route(value),
            tuple_route(value[key] for key in value),
        )
        for module in json_dependency_modules
        for name, value in module.__dict__.items()
        if name != "__builtins__" and exact_type(value) is dict
    )
    json_dependency_list_frames = tuple_route(
        (value, tuple_route(value))
        for module in json_dependency_modules
        for value in module.__dict__.values()
        if exact_type(value) is list
    )
    json_dependency_set_frames = tuple_route(
        (value, frozenset_route(value))
        for module in json_dependency_modules
        for value in module.__dict__.values()
        if exact_type(value) is set
    )
    json_dependency_function_authorities = tuple_route(
        _function_authority(cast(FunctionType, value))
        for module in json_dependency_modules
        for value in module.__dict__.values()
        if exact_type(value) is FunctionType
    )
    json_classes = (json_module.JSONDecoder, json_module.JSONEncoder)
    json_class_frames = tuple_route(
        (
            tuple_route(sorted_route(class_type.__dict__)),
            tuple_route(
                class_type.__dict__[name] for name in sorted_route(class_type.__dict__)
            ),
        )
        for class_type in json_classes
    )
    json_class_function_authorities = tuple_route(
        _function_authority(cast(FunctionType, value))
        for class_type in json_classes
        for value in class_type.__dict__.values()
        if exact_type(value) is FunctionType
    )
    stat_route_names = ("S_ISDIR", "S_ISREG", "S_IMODE")
    stat_routes = tuple_route(
        getattr_route(stat_module, name) for name in stat_route_names
    )
    ctypes_route_names = ("get_errno", "set_errno")
    ctypes_routes = tuple_route(
        getattr_route(ctypes_module, name) for name in ctypes_route_names
    )
    errno_eexist = errno_module.EEXIST
    builtins_namespace = builtins_module.__dict__
    builtins_namespace_names = tuple_route(sorted_route(builtins_namespace))
    builtins_namespace_values = tuple_route(
        builtins_namespace[name] for name in builtins_namespace_names
    )
    get_thread_ident = threading_module.get_ident
    supports_dir_fd = os_module.supports_dir_fd
    supports_dir_fd_frame = frozenset_route(supports_dir_fd)
    linkat_route = _LINKAT
    linkat_argtypes = tuple_route(_LINKAT.argtypes)
    linkat_restype = _LINKAT.restype
    linkat_errcheck = _LINKAT.errcheck
    linkat_flags = _LINKAT._flags_
    if linkat_errcheck is not None or exact_type(linkat_flags) is not int:
        raise RuntimeError("descriptor hard-link callable state is invalid")
    registration_namespace = registration_type.__dict__
    registration_namespace_names = tuple_route(sorted_route(registration_namespace))
    registration_namespace_values = tuple_route(
        registration_namespace[name] for name in registration_namespace_names
    )
    registration_property_names = (
        "head_commit",
        "implementation_commit",
        "registration_sha256",
        "source_bundle_sha256",
    )
    registration_properties = tuple_route(
        registration_namespace[name] for name in registration_property_names
    )
    if any_route(
        exact_type(value) is not property for value in registration_properties
    ):
        raise RuntimeError("run-registration publication properties are unavailable")
    registration_getters = tuple_route(
        cast(property, value).fget for value in registration_properties
    )
    if any_route(
        exact_type(value) is not FunctionType for value in registration_getters
    ):
        raise RuntimeError("run-registration publication getters are unavailable")
    registration_getter_authorities = tuple_route(
        _function_authority(cast(FunctionType, value)) for value in registration_getters
    )

    def require_routes() -> None:
        current_authority_defaults = require_function_authority.__defaults__
        current_authority_kwdefaults = require_function_authority.__kwdefaults__
        current_authority_closure = tuple_route(
            require_function_authority.__closure__ or ()
        )
        if (
            module_globals.get("os") is not os_module
            or module_globals.get("threading") is not threading_module
            or module_globals.get("ctypes") is not ctypes_module
            or module_globals.get("errno") is not errno_module
            or module_globals.get("hashlib") is not hashlib_module
            or module_globals.get("json") is not json_module
            or module_globals.get("stat") is not stat_module
            or module_globals.get("builtins") is not builtins_module
            or module_globals.get("_completed_evidence") is not _completed_evidence
            or module_globals.get("_run_authority") is not run_authority_module
            or require_function_authority.__code__ is not authority_checker_code
            or current_authority_defaults is not authority_checker_defaults
            or (
                current_authority_defaults is not None
                and (
                    authority_checker_default_values is None
                    or len(current_authority_defaults)
                    != len(authority_checker_default_values)
                    or any_route(
                        value is not authority_checker_default_values[index]
                        for index, value in enumerate_route(current_authority_defaults)
                    )
                )
            )
            or current_authority_kwdefaults is not authority_checker_kwdefaults
            or (
                current_authority_kwdefaults is not None
                and (
                    authority_checker_kwdefault_values is None
                    or tuple_route(current_authority_kwdefaults)
                    != tuple_route(
                        name for name, _value in authority_checker_kwdefault_values
                    )
                    or any_route(
                        current_authority_kwdefaults[name] is not value
                        for name, value in authority_checker_kwdefault_values
                    )
                )
            )
            or len(current_authority_closure) != len(authority_checker_closure)
            or any_route(
                cell is not authority_checker_closure[index]
                or cell.cell_contents is not authority_checker_closure_values[index]
                for index, cell in enumerate_route(current_authority_closure)
            )
            or run_authority_module._verified_state is not verified_state_route
            or run_authority_module._VerifiedState is not verified_state_type
            or exact_type(layout) is not layout_type
            or tuple_route(
                cast(MemberDescriptorType, descriptor).__get__(layout, layout_type)
                for descriptor in layout_descriptors
            )
            != layout_frame
            or tuple_route(sorted_route(verified_state_type.__dict__))
            != verified_state_namespace_names
            or any_route(
                verified_state_type.__dict__[name]
                is not verified_state_namespace_values[index]
                for index, name in enumerate_route(verified_state_namespace_names)
            )
            or threading_module.get_ident is not get_thread_ident
            or module_globals.get("_LINKAT") is not linkat_route
            or tuple_route(linkat_route.argtypes) != linkat_argtypes
            or linkat_route.restype is not linkat_restype
            or linkat_route.errcheck is not linkat_errcheck
            or exact_type(linkat_route._flags_) is not int
            or linkat_route._flags_ != linkat_flags
            or errno_module.EEXIST is not errno_eexist
            or module_globals.get("FinalExecutionFailureEvidence") is not failure_type
            or module_globals.get("VerifiedRunRegistration") is not registration_type
            or _completed_evidence.FinalCompletedEvidence is not completed_type
            or _completed_evidence.snapshot_completed_publication
            is not completed_snapshot
            or any_route(
                module_globals.get(name) is not routes[index]
                for index, name in enumerate_route(route_names)
            )
            or any_route(
                registration_type.__dict__.get(name)
                is not registration_properties[index]
                for index, name in enumerate_route(registration_property_names)
            )
            or tuple_route(sorted_route(registration_type.__dict__))
            != registration_namespace_names
            or any_route(
                registration_type.__dict__[name]
                is not registration_namespace_values[index]
                for index, name in enumerate_route(registration_namespace_names)
            )
            or any_route(
                module_globals.get(name) is not constant_values[index]
                for index, name in enumerate_route(constant_names)
            )
            or any_route(
                module_globals.get(name) is not classes[index]
                for index, name in enumerate_route(class_names)
            )
            or any_route(
                tuple_route(sorted_route(class_type.__dict__)) != class_frames[index][0]
                or any_route(
                    class_type.__dict__[name] is not class_frames[index][1][item]
                    for item, name in enumerate_route(sorted_route(class_type.__dict__))
                )
                for index, class_type in enumerate_route(classes)
            )
            or any_route(
                getattr_route(os_module, name, None) is not os_routes[index]
                for index, name in enumerate_route(os_route_names)
            )
            or any_route(
                exact_type(getattr_route(os_module, name, None)) is not int
                or getattr_route(os_module, name, None) != os_constant_values[index]
                for index, name in enumerate_route(os_constant_names)
            )
            or any_route(
                getattr_route(hashlib_module, name, None) is not hashlib_routes[index]
                for index, name in enumerate_route(hashlib_route_names)
            )
            or any_route(
                getattr_route(json_module, name, None) is not json_routes[index]
                for index, name in enumerate_route(json_route_names)
            )
            or any_route(
                tuple_route(sorted_route(module.__dict__))
                != json_dependency_namespace_frames[index][0]
                or any_route(
                    module.__dict__[name]
                    is not json_dependency_namespace_frames[index][1][item]
                    for item, name in enumerate_route(sorted_route(module.__dict__))
                )
                for index, module in enumerate_route(json_dependency_modules)
            )
            or any_route(
                tuple_route(value) != keys
                or any_route(
                    value[key] is not values[index]
                    for index, key in enumerate_route(keys)
                )
                for value, keys, values in json_dependency_dict_frames
            )
            or any_route(
                len(value) != len(values)
                or any_route(
                    item is not values[index] for index, item in enumerate_route(value)
                )
                for value, values in json_dependency_list_frames
            )
            or any_route(
                frozenset_route(value) != values
                for value, values in json_dependency_set_frames
            )
            or any_route(
                tuple_route(sorted_route(class_type.__dict__))
                != json_class_frames[index][0]
                or any_route(
                    class_type.__dict__[name] is not json_class_frames[index][1][item]
                    for item, name in enumerate_route(sorted_route(class_type.__dict__))
                )
                for index, class_type in enumerate_route(json_classes)
            )
            or any_route(
                getattr_route(stat_module, name, None) is not stat_routes[index]
                for index, name in enumerate_route(stat_route_names)
            )
            or any_route(
                getattr_route(ctypes_module, name, None) is not ctypes_routes[index]
                for index, name in enumerate_route(ctypes_route_names)
            )
            or tuple_route(sorted_route(builtins_module.__dict__))
            != builtins_namespace_names
            or any_route(
                builtins_module.__dict__[name] is not builtins_namespace_values[index]
                for index, name in enumerate_route(builtins_namespace_names)
            )
            or os_module.supports_dir_fd is not supports_dir_fd
            or frozenset_route(supports_dir_fd) != supports_dir_fd_frame
        ):
            raise error_type("registered final publication routes changed")
        for authority in closure_route_authorities:
            require_function_authority(authority)
        for authority in registration_getter_authorities:
            require_function_authority(authority)
        for authority in class_function_authorities:
            require_function_authority(authority)
        for authority in json_class_function_authorities:
            require_function_authority(authority)
        for authority in json_dependency_function_authorities:
            require_function_authority(authority)
        for index, (name, function) in enumerate(critical_function_items):
            if module_globals.get(name) is not function:
                raise error_type("registered final publication helpers changed")
            require_function_authority(critical_function_authorities[index])

    def registration_frame(
        registration: VerifiedRunRegistration,
    ) -> RegistrationPublicationFrame:
        state = verified_state_route(registration)
        if exact_type(state) is not verified_state_type:
            raise error_type("run-registration state type changed")
        fields = tuple_route(
            cast(MemberDescriptorType, descriptor).__get__(state, verified_state_type)
            for descriptor in verified_state_descriptors
        )
        return registration_validator(fields)

    def publish_registered_final_evidence(
        registration: VerifiedRunRegistration,
        evidence: object,
        /,
    ) -> None:
        nonlocal attempted, in_flight

        if exact_type(registration) is not registration_type:
            raise type_error("registration must be an exact VerifiedRunRegistration")
        if exact_type(evidence) not in {completed_type, failure_type}:
            raise type_error("evidence must be one exact final evidence type")
        require_routes()
        if os_module.getpid() != owner_process:
            raise error_type("registered final publisher was inherited by a fork")
        current_thread = threading_module.get_ident()
        if exact_type(current_thread) is not int or current_thread < 1:
            raise error_type("registered final publisher thread identity is invalid")
        if current_thread != owner_thread:
            raise error_type("registered final publisher thread changed")
        if not lock.acquire(blocking=False):
            raise error_type("registered final publication overlaps another call")
        try:
            if attempted:
                raise error_type("registered final publication was already attempted")
            attempted = True
            if in_flight:
                raise error_type("registered final publication is already in flight")
            in_flight = True
        finally:
            lock.release()

        try:
            require_routes()
            adapter = (
                completed_snapshot
                if exact_type(evidence) is completed_type
                else failure_snapshot
            )
            first = snapshot_validator(adapter(evidence))
            require_routes()

            def final_authority_check() -> None:
                require_routes()
                result = reverify(registration)
                if result is not None:
                    raise error_type(
                        "run-registration revalidation returned an unexpected value"
                    )
                observed_frame = registration_frame(registration)
                second = snapshot_validator(adapter(evidence))
                require_routes()
                if observed_frame != first[1] or second != first:
                    raise error_type(
                        "final evidence differs from the live run registration"
                    )

            publish_core(first, layout, final_authority_check)
            require_routes()
        except BaseException as primary:
            try:
                require_routes()
            except BaseException:
                raise error_type(
                    "registered final publication failed after its authority changed"
                ) from primary
            raise
        finally:
            with lock:
                in_flight = False

    return publish_registered_final_evidence


publish_registered_final_evidence = _make_registered_publisher()
del _make_registered_publisher


# Prevent mutable builtins aliases from silently changing exact-type and constructor
# behavior used by the issuer.  Keeping this module reference also makes the trust
# assumption explicit for source-bound review.
_BUILTINS_AUTHORITY: Final = builtins
