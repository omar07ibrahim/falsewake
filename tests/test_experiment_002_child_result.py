from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import os
import pickle
import shutil
import stat
import struct
import sys
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from types import FrameType
from typing import Any, cast

import pytest

import falsewake.experiment_002_child_result as child_result

_SCRATCH_ROOT = Path("/home/ubuntu/gitcode/.t")
_HISTORY_DOMAIN = b"falsewake-exp002-history-v1\0"
_ENVELOPE_DOMAIN = b"falsewake-exp002-child-result-envelope-v1\0"
_CLASS_SUPPORT = (397, 406, 350, 377, 352, 363, 363, 373, 350, 372, 6_278, 602)


@dataclass(frozen=True, slots=True)
class _Case:
    binding: child_result.ChildResultBinding
    history: bytes
    history_sha256: str
    safetensors: bytes
    safetensors_sha256: str
    winner_epoch: int
    model_tensor_sha256: str


@pytest.fixture
def scratch_directory() -> Iterator[Path]:
    path = _SCRATCH_ROOT / f"falsewake-child-result-test-{uuid.uuid4().hex}"
    path.mkdir(mode=0o700)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _canonical(document: dict[str, object]) -> bytes:
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


def _model_specs() -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple(
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


def _safetensors() -> tuple[bytes, str]:
    specs = _model_specs()
    header: dict[str, object] = {}
    data = bytearray()
    framed = bytearray(struct.pack("<I", len(specs)))
    offset = 0
    for tensor_index, (name, shape) in enumerate(specs):
        value_count = 1
        for dimension in shape:
            value_count *= dimension
        # Finite, deterministic little-endian float32 values with distinct
        # tensor populations but no dependency on Torch or NumPy.
        raw = struct.pack("<f", float(tensor_index) / 100.0) * value_count
        end = offset + len(raw)
        header[name] = {
            "dtype": "F32",
            "shape": list(shape),
            "data_offsets": [offset, end],
        }
        data.extend(raw)
        name_bytes = name.encode("utf-8")
        framed.extend(struct.pack("<I", len(name_bytes)))
        framed.extend(name_bytes)
        framed.extend(struct.pack("<I", 5))
        framed.extend(b"F32LE")
        framed.extend(struct.pack("<I", len(shape)))
        for dimension in shape:
            framed.extend(struct.pack("<Q", dimension))
        framed.extend(struct.pack("<Q", len(raw)))
        framed.extend(raw)
        offset = end
    header_bytes = json.dumps(
        header,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    padded = header_bytes + b" " * (-len(header_bytes) % 8)
    payload = struct.pack("<Q", len(padded)) + padded + bytes(data)
    assert len(specs) == 53
    assert len(data) == 94_896
    assert len(padded) == 4_872
    assert len(payload) == 99_776
    return payload, hashlib.sha256(
        b"falsewake-exp002-model-tensors-v1\0" + bytes(framed)
    ).hexdigest()


def _history(
    *,
    seed: int = 20_260_719,
    winner_epoch: int = 7,
    winner_model_sha256: str,
) -> tuple[bytes, str]:
    validation_input_digest = _sha("validation-inputs")
    confusion: list[list[int]] = []
    for index, support in enumerate(_CLASS_SUPPORT):
        row = [0] * len(_CLASS_SUPPORT)
        row[index] = support
        confusion.append(row)
    epochs: list[dict[str, object]] = []
    for epoch in range(30):
        model_sha256 = (
            winner_model_sha256 if epoch == winner_epoch else _sha(f"model-{epoch}")
        )
        if epoch == winner_epoch:
            winner_model_sha256 = model_sha256
        validation_cross_entropy = 0.25 if epoch == winner_epoch else 2.0 + epoch
        epochs.append(
            {
                "epoch_update_trace_digest": _sha(f"epoch-trace-{epoch}"),
                "first_global_update": epoch * 313,
                "last_global_update_inclusive": (epoch + 1) * 313 - 1,
                "macro_f1_exact_denominator": 1,
                "macro_f1_exact_numerator": 1,
                "model_tensor_digest": model_sha256,
                "training_cross_entropy_float64_hex": float(1.0 + epoch).hex(),
                "training_population_digest": _sha(f"population-{epoch}"),
                "validation_confusion_matrix": confusion,
                "validation_cross_entropy_float64_hex": (
                    validation_cross_entropy.hex()
                ),
                "validation_input_digest": validation_input_digest,
                "validation_prediction_digest": _sha(f"predictions-{epoch}"),
                "zero_based_epoch": epoch,
            }
        )
    history = _canonical(
        {
            "complete_update_trace_digest": _sha("complete-update-trace"),
            "epochs": epochs,
            "experiment": "002",
            "schema_version": 1,
            "seed": seed,
            "validation_input_digest": validation_input_digest,
        }
    )
    return history, winner_model_sha256


def _case(*, role: str = "training_seed", ordinal: int = 0) -> _Case:
    seed = 20_260_719 if ordinal in {0, 3} else 20_260_719 + ordinal
    binding = child_result.ChildResultBinding(
        role=cast(Any, role),
        ordinal=ordinal,
        seed=seed,
        registration_head_commit="a" * 40,
        implementation_commit="b" * 40,
        registration_sha256="c" * 64,
        source_bundle_sha256="d" * 64,
    )
    safetensors, model_tensor_sha256 = _safetensors()
    history, model_tensor_sha256 = _history(
        seed=seed,
        winner_model_sha256=model_tensor_sha256,
    )
    return _Case(
        binding=binding,
        history=history,
        history_sha256=hashlib.sha256(_HISTORY_DOMAIN + history).hexdigest(),
        safetensors=safetensors,
        safetensors_sha256=hashlib.sha256(safetensors).hexdigest(),
        winner_epoch=7,
        model_tensor_sha256=model_tensor_sha256,
    )


def _write(path: Path, case: _Case) -> None:
    child_result.write_registered_child_result(
        path,
        case.binding,
        history_json_bytes=case.history,
        history_sha256=case.history_sha256,
        safetensors_bytes=case.safetensors,
        safetensors_sha256=case.safetensors_sha256,
        winner_epoch=case.winner_epoch,
        model_tensor_sha256=case.model_tensor_sha256,
    )


def _envelope_path(path: Path) -> Path:
    return path / child_result._ENVELOPE_FILENAME


def _history_path(path: Path) -> Path:
    return path / child_result._HISTORY_FILENAME


def _safetensors_path(path: Path) -> Path:
    return path / child_result._SAFETENSORS_FILENAME


def _load_document(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="ascii"))
    assert type(parsed) is dict
    return cast(dict[str, Any], parsed)


def _rewrite_envelope(
    path: Path,
    mutation: Callable[[dict[str, Any]], None],
) -> None:
    envelope = _load_document(_envelope_path(path))
    mutation(envelope)
    _envelope_path(path).write_bytes(_canonical(cast(dict[str, object], envelope)))


def _rewrite_history_and_bind_envelope(path: Path, history: bytes) -> None:
    _history_path(path).write_bytes(history)

    def bind(envelope: dict[str, Any]) -> None:
        history_record = cast(dict[str, Any], envelope["history"])
        history_record["byte_count"] = len(history)
        history_record["domain_sha256"] = hashlib.sha256(
            _HISTORY_DOMAIN + history
        ).hexdigest()

    _rewrite_envelope(path, bind)


def test_public_surface_is_fixed_opaque_and_standard_library_only() -> None:
    writer_parameters = inspect.signature(
        child_result.write_registered_child_result
    ).parameters
    assert tuple(writer_parameters) == (
        "scratch_directory",
        "binding",
        "history_json_bytes",
        "history_sha256",
        "safetensors_bytes",
        "safetensors_sha256",
        "winner_epoch",
        "model_tensor_sha256",
    )
    loader_parameters = inspect.signature(
        child_result.load_registered_child_result
    ).parameters
    assert tuple(loader_parameters) == (
        "scratch_directory",
        "expected",
    )
    assert tuple(
        inspect.signature(child_result.verify_verified_child_result).parameters
    ) == ("result",)
    with pytest.raises(TypeError, match="issued only by the parser"):
        child_result.VerifiedChildResult()
    source = Path("src/falsewake/experiment_002_child_result.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.partition(".")[0])
    assert imported_roots <= {
        "__future__",
        "contextlib",
        "dataclasses",
        "fcntl",
        "fractions",
        "hashlib",
        "json",
        "math",
        "os",
        "pathlib",
        "stat",
        "struct",
        "threading",
        "typing",
        "weakref",
    }
    assert "os.link" not in source
    assert "os.rename" not in source
    assert "tempfile" not in source


@pytest.mark.parametrize(
    ("role", "ordinal"),
    [("training_seed", 0), ("selected_seed_rerun", 3)],
)
def test_canonical_roundtrip_owns_exact_bytes_and_digest_domains(
    scratch_directory: Path,
    role: str,
    ordinal: int,
) -> None:
    case = _case(role=role, ordinal=ordinal)
    _write(scratch_directory, case)

    observed = child_result.load_registered_child_result(
        scratch_directory, case.binding
    )

    assert observed.binding == case.binding
    assert observed.binding is not case.binding
    assert observed.role == role
    assert observed.ordinal == ordinal
    assert observed.seed == case.binding.seed
    assert observed.registration_head_commit == "a" * 40
    assert observed.implementation_commit == "b" * 40
    assert observed.registration_sha256 == "c" * 64
    assert observed.source_bundle_sha256 == "d" * 64
    assert observed.history_sha256 == case.history_sha256
    assert observed.history_byte_count == len(case.history)
    assert observed.safetensors_sha256 == case.safetensors_sha256
    assert observed.safetensors_byte_count == len(case.safetensors)
    assert observed.winner_epoch == case.winner_epoch
    assert observed.model_tensor_sha256 == case.model_tensor_sha256
    assert observed.canonical_history_bytes is observed.canonical_history_bytes
    assert observed.canonical_history_bytes == case.history
    assert observed.safetensors_bytes == case.safetensors
    assert (
        observed.envelope_sha256
        == hashlib.sha256(
            _ENVELOPE_DOMAIN + observed.canonical_envelope_bytes
        ).hexdigest()
    )
    child_result.verify_verified_child_result(observed)
    for copier in (copy.copy, copy.deepcopy, pickle.dumps):
        with pytest.raises(TypeError, match="cannot be copied|cannot be serialized"):
            copier(observed)

    assert set(os.listdir(scratch_directory)) == {
        child_result._HISTORY_FILENAME,
        child_result._SAFETENSORS_FILENAME,
        child_result._ENVELOPE_FILENAME,
    }
    for path in scratch_directory.iterdir():
        metadata = path.stat()
        assert stat.S_ISREG(metadata.st_mode)
        assert stat.S_IMODE(metadata.st_mode) == 0o600
        assert metadata.st_nlink == 1


def test_writer_validates_everything_before_first_filesystem_write(
    scratch_directory: Path,
) -> None:
    case = _case()
    invalid_cases = (
        replace(case, history_sha256="0" * 64),
        replace(case, safetensors_sha256="0" * 64),
        replace(case, winner_epoch=8),
        replace(case, model_tensor_sha256="0" * 64),
        replace(case, history=case.history[:-1]),
        replace(case, safetensors=b""),
    )
    for invalid in invalid_cases:
        with pytest.raises((TypeError, child_result.Experiment002ChildResultError)):
            _write(scratch_directory, invalid)
        assert os.listdir(scratch_directory) == []


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            {
                "role": "selected_seed_rerun",
                "ordinal": 0,
                "seed": 20_260_719,
            },
            "ordinal",
        ),
        (
            {"role": "training_seed", "ordinal": 1, "seed": 20_260_719},
            "seed differs",
        ),
        (
            {"role": "training_seed", "ordinal": 0, "seed": 123},
            "seed is not registered",
        ),
        (
            {
                "role": "training_seed",
                "ordinal": 0,
                "seed": 20_260_719,
                "registration_sha256": "G" * 64,
            },
            "lowercase hexadecimal",
        ),
    ],
)
def test_binding_rejects_wrong_role_ordinal_seed_and_digests(
    mutation: dict[str, object],
    match: str,
) -> None:
    fields: dict[str, object] = {
        "role": "training_seed",
        "ordinal": 0,
        "seed": 20_260_719,
        "registration_head_commit": "a" * 40,
        "implementation_commit": "b" * 40,
        "registration_sha256": "c" * 64,
        "source_bundle_sha256": "d" * 64,
    }
    fields.update(mutation)
    with pytest.raises(
        (TypeError, child_result.Experiment002ChildResultError),
        match=match,
    ):
        child_result.ChildResultBinding(**cast(Any, fields))


@pytest.mark.parametrize(
    "target",
    ["history", "safetensors", "envelope"],
)
def test_tamper_and_truncation_are_rejected(
    scratch_directory: Path,
    target: str,
) -> None:
    case = _case()
    _write(scratch_directory, case)
    path = {
        "history": _history_path(scratch_directory),
        "safetensors": _safetensors_path(scratch_directory),
        "envelope": _envelope_path(scratch_directory),
    }[target]
    payload = path.read_bytes()
    path.write_bytes(payload[:-1])
    with pytest.raises(child_result.Experiment002ChildResultError):
        child_result.load_registered_child_result(scratch_directory, case.binding)


@pytest.mark.parametrize("kind", ["reordered", "duplicate", "nonfinite"])
def test_noncanonical_reordered_duplicate_or_nonfinite_history_is_rejected(
    scratch_directory: Path,
    kind: str,
) -> None:
    case = _case()
    _write(scratch_directory, case)
    document = cast(dict[str, Any], json.loads(case.history))
    if kind == "reordered":
        reversed_document = dict(reversed(tuple(document.items())))
        history = (
            json.dumps(
                reversed_document,
                sort_keys=False,
                separators=(",", ":"),
                ensure_ascii=True,
            )
            + "\n"
        ).encode("ascii")
        assert history != case.history
    elif kind == "duplicate":
        history = case.history.replace(
            b'"experiment":"002"',
            b'"experiment":"002","experiment":"002"',
            1,
        )
    else:
        history = case.history.replace(
            b'"schema_version":1', b'"schema_version":NaN', 1
        )
    _rewrite_history_and_bind_envelope(scratch_directory, history)
    with pytest.raises(
        child_result.Experiment002ChildResultError,
        match="canonical|JSON|number",
    ):
        child_result.load_registered_child_result(scratch_directory, case.binding)


def test_duplicate_or_reordered_envelope_is_rejected(
    scratch_directory: Path,
) -> None:
    case = _case()
    _write(scratch_directory, case)
    envelope = _envelope_path(scratch_directory).read_bytes()
    duplicate = envelope.replace(
        b'"experiment":"002"',
        b'"experiment":"002","experiment":"002"',
        1,
    )
    _envelope_path(scratch_directory).write_bytes(duplicate)
    with pytest.raises(child_result.Experiment002ChildResultError, match="canonical"):
        child_result.load_registered_child_result(scratch_directory, case.binding)

    _envelope_path(scratch_directory).write_bytes(envelope)
    document = cast(dict[str, Any], json.loads(envelope))
    reversed_document = dict(reversed(tuple(document.items())))
    reordered = (
        json.dumps(reversed_document, sort_keys=False, separators=(",", ":")) + "\n"
    ).encode("ascii")
    assert reordered != envelope
    _envelope_path(scratch_directory).write_bytes(reordered)
    with pytest.raises(child_result.Experiment002ChildResultError, match="canonical"):
        child_result.load_registered_child_result(scratch_directory, case.binding)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("role", "selected_seed_rerun"),
        ("ordinal", 1),
        ("seed", 20_260_720),
        ("registration.head_commit", "e" * 40),
        ("registration.implementation_commit", "f" * 40),
        ("registration.registration_sha256", "1" * 64),
        ("registration.source_bundle_sha256", "2" * 64),
    ],
)
def test_loader_rejects_wrong_ticket_or_registration_binding(
    scratch_directory: Path,
    field: str,
    value: object,
) -> None:
    case = _case()
    _write(scratch_directory, case)

    def mutate(envelope: dict[str, Any]) -> None:
        if field.startswith("registration."):
            registration = cast(dict[str, Any], envelope["registration"])
            registration[field.partition(".")[2]] = value
        else:
            envelope[field] = value

    _rewrite_envelope(scratch_directory, mutate)
    with pytest.raises(child_result.Experiment002ChildResultError):
        child_result.load_registered_child_result(scratch_directory, case.binding)


def test_history_is_independently_ranked_and_binds_winner_model_digest(
    scratch_directory: Path,
) -> None:
    case = _case()
    document = cast(dict[str, Any], json.loads(case.history))
    epochs = cast(list[dict[str, Any]], document["epochs"])
    epochs[7]["validation_cross_entropy_float64_hex"] = (3.0).hex()
    epochs[11]["validation_cross_entropy_float64_hex"] = (0.125).hex()
    epochs[11]["model_tensor_digest"] = case.model_tensor_sha256
    history = _canonical(cast(dict[str, object], document))
    changed = replace(
        case,
        history=history,
        history_sha256=hashlib.sha256(_HISTORY_DOMAIN + history).hexdigest(),
        winner_epoch=11,
        model_tensor_sha256=case.model_tensor_sha256,
    )
    _write(scratch_directory, changed)
    result = child_result.load_registered_child_result(
        scratch_directory, changed.binding
    )
    assert result.winner_epoch == 11
    assert result.model_tensor_sha256 == epochs[11]["model_tensor_digest"]


def test_history_confusion_support_macro_fraction_and_epoch_order_are_checked(
    scratch_directory: Path,
) -> None:
    case = _case()
    mutations: tuple[Callable[[dict[str, Any]], None], ...] = (
        lambda document: cast(list[dict[str, Any]], document["epochs"])[0].__setitem__(
            "zero_based_epoch", 1
        ),
        lambda document: cast(list[dict[str, Any]], document["epochs"])[0].__setitem__(
            "macro_f1_exact_numerator", 0
        ),
        lambda document: cast(list[dict[str, Any]], document["epochs"])[0][
            "validation_confusion_matrix"
        ][0].__setitem__(0, 396),
    )
    for mutate in mutations:
        document = cast(dict[str, Any], json.loads(case.history))
        mutate(document)
        changed_history = _canonical(cast(dict[str, object], document))
        changed = replace(
            case,
            history=changed_history,
            history_sha256=hashlib.sha256(
                _HISTORY_DOMAIN + changed_history
            ).hexdigest(),
        )
        with pytest.raises(child_result.Experiment002ChildResultError):
            _write(scratch_directory, changed)
        assert os.listdir(scratch_directory) == []


def test_envelope_is_written_and_recognized_last(
    scratch_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case()
    events: list[str] = []
    original = child_result._write_new_file
    original_fsync = child_result._retry_fsync

    def record(
        directory_fd: int,
        name: str,
        payload: bytes,
        owned: list[child_result._OwnedFile],
    ) -> None:
        events.append(f"write:{name}")
        original(directory_fd, name, payload, owned)

    def record_fsync(descriptor: int) -> None:
        kind = "dir" if stat.S_ISDIR(os.fstat(descriptor).st_mode) else "file"
        events.append(f"fsync:{kind}")
        original_fsync(descriptor)

    monkeypatch.setattr(child_result, "_write_new_file", record)
    monkeypatch.setattr(child_result, "_retry_fsync", record_fsync)
    _write(scratch_directory, case)
    assert events == [
        f"write:{child_result._HISTORY_FILENAME}",
        "fsync:file",
        f"write:{child_result._SAFETENSORS_FILENAME}",
        "fsync:file",
        "fsync:dir",
        f"write:{child_result._ENVELOPE_FILENAME}",
        "fsync:file",
        "fsync:dir",
    ]
    _envelope_path(scratch_directory).unlink()
    with pytest.raises(child_result.Experiment002ChildResultError, match="exact fixed"):
        child_result.load_registered_child_result(scratch_directory, case.binding)


@pytest.mark.parametrize(
    "failure",
    ["after_envelope_write", "final_directory_check", "final_directory_fsync"],
)
def test_any_post_envelope_interrupt_removes_commit_marker_first(
    scratch_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    case = _case()
    if failure == "after_envelope_write":
        original_write = child_result._write_new_file

        def interrupt_after_write(
            directory_fd: int,
            name: str,
            payload: bytes,
            owned: list[child_result._OwnedFile],
        ) -> None:
            original_write(directory_fd, name, payload, owned)
            if name == child_result._ENVELOPE_FILENAME:
                raise KeyboardInterrupt("synthetic post-envelope interrupt")

        monkeypatch.setattr(child_result, "_write_new_file", interrupt_after_write)
    elif failure == "final_directory_check":
        original_entries = child_result._require_directory_entries
        interrupted = False

        def interrupt_final_check(
            descriptor: int,
            expected: frozenset[str],
        ) -> None:
            nonlocal interrupted
            if expected == child_result._RESULT_FILENAMES and not interrupted:
                interrupted = True
                raise KeyboardInterrupt("synthetic final-check interrupt")
            original_entries(descriptor, expected)

        monkeypatch.setattr(
            child_result,
            "_require_directory_entries",
            interrupt_final_check,
        )
    else:
        original_fsync = child_result._retry_fsync
        directory_calls = 0

        def interrupt_final_fsync(descriptor: int) -> None:
            nonlocal directory_calls
            if stat.S_ISDIR(os.fstat(descriptor).st_mode):
                directory_calls += 1
                if directory_calls == 2:
                    raise KeyboardInterrupt("synthetic final-fsync interrupt")
            original_fsync(descriptor)

        monkeypatch.setattr(child_result, "_retry_fsync", interrupt_final_fsync)

    with pytest.raises(KeyboardInterrupt):
        _write(scratch_directory, case)
    assert child_result._ENVELOPE_FILENAME not in os.listdir(scratch_directory)
    assert os.listdir(scratch_directory) == []


def test_semantic_safetensors_substitution_fails_even_with_updated_plain_hash(
    scratch_directory: Path,
) -> None:
    case = _case()
    _write(scratch_directory, case)
    path = _safetensors_path(scratch_directory)
    replacement = bytearray(path.read_bytes())
    replacement[-1] ^= 1
    path.write_bytes(replacement)

    def rebind_plain_hash(envelope: dict[str, Any]) -> None:
        safetensors = cast(dict[str, Any], envelope["safetensors"])
        safetensors["sha256"] = hashlib.sha256(replacement).hexdigest()

    _rewrite_envelope(scratch_directory, rebind_plain_hash)
    with pytest.raises(
        child_result.Experiment002ChildResultError,
        match="binding|model|tensor",
    ):
        child_result.load_registered_child_result(scratch_directory, case.binding)


def test_interrupted_pwrite_pread_and_fsync_are_retried(
    scratch_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case()
    original_pwrite = os.pwrite
    original_fsync = os.fsync
    pwrite_interrupted = False
    fsync_interrupted = False

    def interrupted_pwrite(
        descriptor: int,
        payload: bytes | memoryview,
        offset: int,
    ) -> int:
        nonlocal pwrite_interrupted
        if not pwrite_interrupted:
            pwrite_interrupted = True
            raise InterruptedError
        return original_pwrite(descriptor, payload, offset)

    def interrupted_fsync(descriptor: int) -> None:
        nonlocal fsync_interrupted
        if not fsync_interrupted:
            fsync_interrupted = True
            raise InterruptedError
        original_fsync(descriptor)

    monkeypatch.setattr(os, "pwrite", interrupted_pwrite)
    monkeypatch.setattr(os, "fsync", interrupted_fsync)
    _write(scratch_directory, case)
    assert pwrite_interrupted and fsync_interrupted

    original_pread = os.pread
    pread_interrupted = False

    def interrupted_pread(descriptor: int, length: int, offset: int) -> bytes:
        nonlocal pread_interrupted
        if not pread_interrupted:
            pread_interrupted = True
            raise InterruptedError
        return original_pread(descriptor, length, offset)

    monkeypatch.setattr(os, "pread", interrupted_pread)
    result = child_result.load_registered_child_result(
        scratch_directory,
        case.binding,
    )
    assert pread_interrupted
    assert result.model_tensor_sha256 == case.model_tensor_sha256


def test_extra_file_symlink_nonregular_and_hardlink_alias_are_rejected(
    scratch_directory: Path,
) -> None:
    case = _case()
    _write(scratch_directory, case)
    extra = scratch_directory / "unexpected"
    extra.write_bytes(b"extra")
    with pytest.raises(child_result.Experiment002ChildResultError, match="exact fixed"):
        child_result.load_registered_child_result(scratch_directory, case.binding)
    extra.unlink()

    history_path = _history_path(scratch_directory)
    saved_history = history_path.read_bytes()
    history_path.chmod(0o644)
    with pytest.raises(child_result.Experiment002ChildResultError, match="metadata"):
        child_result.load_registered_child_result(scratch_directory, case.binding)
    history_path.chmod(0o600)

    history_path.unlink()
    history_path.symlink_to(_envelope_path(scratch_directory).name)
    with pytest.raises(child_result.Experiment002ChildResultError):
        child_result.load_registered_child_result(scratch_directory, case.binding)
    history_path.unlink()
    history_path.write_bytes(saved_history)
    history_path.chmod(0o600)

    safetensors_path = _safetensors_path(scratch_directory)
    saved_safetensors = safetensors_path.read_bytes()
    safetensors_path.unlink()
    safetensors_path.mkdir()
    with pytest.raises(child_result.Experiment002ChildResultError):
        child_result.load_registered_child_result(scratch_directory, case.binding)
    safetensors_path.rmdir()
    safetensors_path.write_bytes(saved_safetensors)
    safetensors_path.chmod(0o600)

    safetensors_path.unlink()
    os.link(history_path, safetensors_path)
    with pytest.raises(
        child_result.Experiment002ChildResultError,
        match="metadata|hard",
    ):
        child_result.load_registered_child_result(scratch_directory, case.binding)


def test_fifo_is_rejected_without_blocking(scratch_directory: Path) -> None:
    case = _case()
    _write(scratch_directory, case)
    path = _safetensors_path(scratch_directory)
    path.unlink()
    os.mkfifo(path, 0o600)
    with pytest.raises(child_result.Experiment002ChildResultError, match="metadata"):
        child_result.load_registered_child_result(scratch_directory, case.binding)


def test_relative_outside_symlinked_and_wrong_mode_directories_are_rejected(
    scratch_directory: Path,
) -> None:
    case = _case()
    with pytest.raises(child_result.Experiment002ChildResultError, match="fixed root"):
        _write(Path("relative-scratch"), case)
    with pytest.raises(child_result.Experiment002ChildResultError, match="fixed root"):
        _write(Path("/tmp/outside-scratch"), case)

    linked = scratch_directory.with_name(f"{scratch_directory.name}-link")
    linked.symlink_to(scratch_directory, target_is_directory=True)
    try:
        with pytest.raises((OSError, child_result.Experiment002ChildResultError)):
            _write(linked, case)
    finally:
        linked.unlink()

    scratch_directory.chmod(0o755)
    try:
        with pytest.raises(child_result.Experiment002ChildResultError, match="0700"):
            _write(scratch_directory, case)
    finally:
        scratch_directory.chmod(0o700)


def test_unstable_file_read_is_detected(
    scratch_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case()
    _write(scratch_directory, case)
    target = _history_path(scratch_directory)
    target_inode = target.stat().st_ino
    original_pread = os.pread
    mutated = False

    def unstable_pread(descriptor: int, length: int, offset: int) -> bytes:
        nonlocal mutated
        payload = original_pread(descriptor, length, offset)
        if not mutated and os.fstat(descriptor).st_ino == target_inode and offset == 0:
            with target.open("ab") as stream:
                stream.write(b"x")
            mutated = True
        return payload

    monkeypatch.setattr(os, "pread", unstable_pread)
    with pytest.raises(
        child_result.Experiment002ChildResultError,
        match="grew|changed",
    ):
        child_result.load_registered_child_result(scratch_directory, case.binding)
    assert mutated


def test_oversized_sparse_payload_is_rejected_before_read(
    scratch_directory: Path,
) -> None:
    case = _case()
    _write(scratch_directory, case)
    with _safetensors_path(scratch_directory).open("r+b") as stream:
        stream.truncate(child_result._MAX_SAFETENSORS_BYTES + 1)
    with pytest.raises(child_result.Experiment002ChildResultError, match="metadata"):
        child_result.load_registered_child_result(scratch_directory, case.binding)


def test_mutated_issued_state_is_terminally_rejected(scratch_directory: Path) -> None:
    case = _case()
    _write(scratch_directory, case)
    result = child_result.load_registered_child_result(scratch_directory, case.binding)
    state = child_result._ISSUED[result]
    child_result._ISSUED[result] = replace(state, history_sha256="0" * 64)
    with pytest.raises(child_result.Experiment002ChildResultError, match="changed"):
        child_result.verify_verified_child_result(result)
    with pytest.raises(child_result.Experiment002ChildResultError, match="terminally"):
        _ = result.seed


def test_coherent_cache_replacement_cannot_replace_closure_truth(
    scratch_directory: Path,
) -> None:
    case = _case()
    _write(scratch_directory, case)
    result = child_result.load_registered_child_result(scratch_directory, case.binding)
    state = child_result._ISSUED[result]
    guard = child_result._GUARDS[result]
    cloned_state = replace(
        state,
        binding=replace(state.binding),
        history_summary=replace(state.history_summary),
    )
    cloned_guard = replace(
        guard,
        binding=replace(guard.binding),
        history_summary=replace(guard.history_summary),
    )
    child_result._ISSUED[result] = cloned_state
    child_result._GUARDS[result] = cloned_guard

    with pytest.raises(child_result.Experiment002ChildResultError, match="authority"):
        child_result.verify_verified_child_result(result)

    child_result._ISSUED[result] = state
    child_result._GUARDS[result] = guard
    with pytest.raises(child_result.Experiment002ChildResultError, match="terminally"):
        _ = result.seed


def test_cache_deletion_is_sticky_after_complete_restoration(
    scratch_directory: Path,
) -> None:
    case = _case()
    _write(scratch_directory, case)
    result = child_result.load_registered_child_result(scratch_directory, case.binding)
    state = child_result._ISSUED.pop(result)
    guard = child_result._GUARDS.pop(result)

    with pytest.raises(child_result.Experiment002ChildResultError, match="issued"):
        child_result.verify_verified_child_result(result)

    child_result._ISSUED[result] = state
    child_result._GUARDS[result] = guard
    with pytest.raises(child_result.Experiment002ChildResultError, match="terminally"):
        _ = result.seed


def test_authority_frames_distinguish_bool_from_equal_integer(
    scratch_directory: Path,
) -> None:
    base = _case()
    history, model_sha256 = _history(
        seed=base.binding.seed,
        winner_epoch=1,
        winner_model_sha256=base.model_tensor_sha256,
    )
    case = replace(
        base,
        history=history,
        history_sha256=hashlib.sha256(_HISTORY_DOMAIN + history).hexdigest(),
        winner_epoch=1,
        model_tensor_sha256=model_sha256,
    )
    _write(scratch_directory, case)
    result = child_result.load_registered_child_result(scratch_directory, case.binding)
    state = child_result._ISSUED[result]
    guard = child_result._GUARDS[result]
    object.__setattr__(state.history_summary, "winner_epoch", True)
    object.__setattr__(guard.history_summary, "winner_epoch", True)

    with pytest.raises(child_result.Experiment002ChildResultError, match="authority"):
        child_result.verify_verified_child_result(result)


def test_property_routes_are_lexically_bound_to_the_verified_state(
    scratch_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case()
    _write(scratch_directory, case)
    result = child_result.load_registered_child_result(scratch_directory, case.binding)

    def forbidden(_result: child_result.VerifiedChildResult, /) -> None:
        raise AssertionError("mutable public verifier route was used")

    monkeypatch.setattr(child_result, "verify_verified_child_result", forbidden)
    assert result.seed == case.binding.seed
    assert result.binding == case.binding


def test_authority_validation_uses_lexically_captured_exact_type(
    scratch_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case()
    _write(scratch_directory, case)
    result = child_result.load_registered_child_result(scratch_directory, case.binding)
    state = child_result._ISSUED[result]
    guard = child_result._GUARDS[result]
    original_type = type
    authority_value_observed = False

    def interposed_type(value: object) -> type[object]:
        nonlocal authority_value_observed
        if value is state or value is guard:
            authority_value_observed = True
        return original_type(value)

    monkeypatch.setattr(child_result, "type", interposed_type, raising=False)
    assert result.winner_epoch == case.winner_epoch
    assert not authority_value_observed
    child_result.verify_verified_child_result(result)


def test_property_returns_immutable_truth_if_caches_mutate_after_validation(
    scratch_directory: Path,
) -> None:
    case = _case()
    _write(scratch_directory, case)
    result = child_result.load_registered_child_result(scratch_directory, case.binding)
    state = child_result._ISSUED[result]
    guard = child_result._GUARDS[result]
    mutated = False

    def mutate_after_frames(
        frame: FrameType,
        event: str,
        _argument: object,
    ) -> Any:
        nonlocal mutated
        if (
            event == "line"
            and frame.f_code.co_name == "validator"
            and "state_frame" in frame.f_locals
            and "guard_frame" in frame.f_locals
        ):
            object.__setattr__(
                state.history_summary,
                "winner_epoch",
                case.winner_epoch + 1,
            )
            object.__setattr__(
                guard.history_summary,
                "winner_epoch",
                case.winner_epoch + 1,
            )
            mutated = True
            sys.settrace(None)
        return mutate_after_frames

    sys.settrace(mutate_after_frames)
    try:
        observed = result.winner_epoch
    finally:
        sys.settrace(None)
    assert mutated
    assert observed == case.winner_epoch
    with pytest.raises(child_result.Experiment002ChildResultError, match="changed"):
        _ = result.winner_epoch


def test_module_state_constructor_substitution_cannot_change_closure_truth(
    scratch_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case()
    _write(scratch_directory, case)

    def forbidden_state_constructor(**_values: object) -> object:
        raise AssertionError("mutable module state constructor was used")

    monkeypatch.setattr(child_result, "_VerifiedState", forbidden_state_constructor)
    result = child_result.load_registered_child_result(
        scratch_directory,
        case.binding,
    )
    assert result.winner_epoch == case.winner_epoch


def test_load_close_interrupt_precedes_authority_issuance(
    scratch_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case()
    _write(scratch_directory, case)
    original_close = os.close
    issued_before = set(child_result._ISSUED)
    issued_during_interrupt: list[child_result.VerifiedChildResult] = []
    interrupted = False

    def interrupt_first_regular_close(descriptor: int) -> None:
        nonlocal interrupted
        is_regular = stat.S_ISREG(os.fstat(descriptor).st_mode)
        original_close(descriptor)
        if is_regular and not interrupted:
            interrupted = True
            issued_during_interrupt.extend(set(child_result._ISSUED) - issued_before)
            raise KeyboardInterrupt("synthetic load close interrupt")

    monkeypatch.setattr(os, "close", interrupt_first_regular_close)
    with pytest.raises(KeyboardInterrupt, match="load close"):
        child_result.load_registered_child_result(scratch_directory, case.binding)
    assert interrupted
    assert issued_during_interrupt == []
    assert set(child_result._ISSUED) == issued_before


def test_post_issuance_load_failure_is_terminally_sticky(
    scratch_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case()
    _write(scratch_directory, case)
    original_parse = child_result._parse_history
    issued_before = set(child_result._ISSUED)
    captured: list[child_result.VerifiedChildResult] = []
    calls = 0

    def interrupt_verification(payload: bytes) -> child_result._HistorySummary:
        nonlocal calls
        calls += 1
        if calls == 2:
            captured.extend(set(child_result._ISSUED) - issued_before)
            raise KeyboardInterrupt("synthetic post-issuance interrupt")
        return original_parse(payload)

    monkeypatch.setattr(child_result, "_parse_history", interrupt_verification)
    with pytest.raises(KeyboardInterrupt, match="post-issuance"):
        child_result.load_registered_child_result(scratch_directory, case.binding)
    assert len(captured) == 1

    monkeypatch.setattr(child_result, "_parse_history", original_parse)
    with pytest.raises(child_result.Experiment002ChildResultError, match="terminally"):
        _ = captured[0].winner_epoch


def test_private_loader_cannot_issue_authority_through_a_substitute_issuer(
    scratch_directory: Path,
) -> None:
    case = _case()
    _write(scratch_directory, case)
    forged = object.__new__(child_result.VerifiedChildResult)

    def substitute_issuer(**_values: object) -> child_result.VerifiedChildResult:
        return forged

    assert not hasattr(child_result, "_issue_verified_result")
    observed = child_result._load_registered_child_result_impl(
        scratch_directory,
        case.binding,
        substitute_issuer,
    )
    assert observed is forged
    with pytest.raises(child_result.Experiment002ChildResultError, match="issued"):
        child_result.verify_verified_child_result(forged)


def test_concurrent_writers_cannot_unlink_the_winner_result(
    scratch_directory: Path,
) -> None:
    case = _case()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_write, scratch_directory, case) for _ in range(2)]
    outcomes: list[BaseException | None] = []
    for future in futures:
        try:
            future.result()
            outcomes.append(None)
        except BaseException as error:
            outcomes.append(error)
    assert sum(outcome is None for outcome in outcomes) == 1
    assert (
        sum(
            isinstance(outcome, child_result.Experiment002ChildResultError)
            for outcome in outcomes
        )
        == 1
    )
    verified = child_result.load_registered_child_result(
        scratch_directory,
        case.binding,
    )
    assert verified.history_sha256 == case.history_sha256


def test_nested_cleanup_interrupt_cannot_strand_the_commit_marker(
    scratch_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case()
    original_fsync = child_result._retry_fsync
    original_unlink = child_result._unlink_owned_file
    directory_fsyncs = 0
    interrupted_cleanup = False

    def interrupt_final_fsync(descriptor: int) -> None:
        nonlocal directory_fsyncs
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            directory_fsyncs += 1
            if directory_fsyncs == 2:
                raise KeyboardInterrupt("synthetic publication interrupt")
        original_fsync(descriptor)

    def interrupt_first_cleanup(
        directory_fd: int,
        source: child_result._OwnedFile,
    ) -> None:
        nonlocal interrupted_cleanup
        if not interrupted_cleanup:
            interrupted_cleanup = True
            raise KeyboardInterrupt("synthetic nested cleanup interrupt")
        original_unlink(directory_fd, source)

    monkeypatch.setattr(child_result, "_retry_fsync", interrupt_final_fsync)
    monkeypatch.setattr(child_result, "_unlink_owned_file", interrupt_first_cleanup)
    with pytest.raises(KeyboardInterrupt, match="publication"):
        _write(scratch_directory, case)
    assert interrupted_cleanup
    assert list(scratch_directory.iterdir()) == []


def test_post_commit_close_interrupt_is_committed_success(
    scratch_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case()
    original_close = os.close
    interrupted = False

    def interrupt_after_close(descriptor: int) -> None:
        nonlocal interrupted
        is_directory = stat.S_ISDIR(os.fstat(descriptor).st_mode)
        original_close(descriptor)
        if (
            is_directory
            and _envelope_path(scratch_directory).exists()
            and not interrupted
        ):
            interrupted = True
            raise KeyboardInterrupt("synthetic post-commit close interrupt")

    monkeypatch.setattr(os, "close", interrupt_after_close)
    _write(scratch_directory, case)
    assert interrupted
    verified = child_result.load_registered_child_result(
        scratch_directory,
        case.binding,
    )
    assert verified.safetensors_sha256 == case.safetensors_sha256
