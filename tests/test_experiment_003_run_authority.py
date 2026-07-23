from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import pytest

from falsewake import experiment_002_run_authority as engine
from falsewake import experiment_003_run_authority as authority

ROOT = Path(__file__).resolve().parents[1]
P2 = engine._EXPERIMENT_002_PROFILE
P3 = engine._EXPERIMENT_003_PROFILE


def _reset_issuer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine, "_ISSUED", weakref.WeakKeyDictionary())
    monkeypatch.setattr(engine, "_ISSUED_GUARDS", weakref.WeakKeyDictionary())
    monkeypatch.setattr(engine, "_FAILED", weakref.WeakSet())
    monkeypatch.setattr(engine, "_ISSUANCE_COMPLETE", False)


@pytest.fixture(autouse=True)
def reset_issuer(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_issuer(monkeypatch)


def _canonical(document: object) -> bytes:
    return (
        json.dumps(
            document,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _runtime(profile: engine._AuthorityProfile) -> dict[str, Any]:
    fields = {
        "numpy_version": "2.5.1",
        "python_executable": "/home/ubuntu/gitcode/.t/falsewake-venv/bin/python",
        "python_sys_path_roots": [
            "/usr/lib/python312.zip",
            "/usr/lib/python3.12",
            "/usr/lib/python3.12/lib-dynload",
            "/home/ubuntu/gitcode/.t/falsewake-venv/lib/python3.12/site-packages",
        ],
        "python_version": "3.12.3",
        "safetensors_version": "0.8.0",
        "torch_version": "2.13.0+cpu",
    }
    return {
        **fields,
        "fingerprint_sha256": engine._runtime_fingerprint_sha256(fields, profile),
    }


def _external_inputs(profile: engine._AuthorityProfile) -> dict[str, Any]:
    frozen = engine._expected_frozen_bindings(profile)
    inputs = cast(dict[str, Any], frozen["inputs"])
    artifacts = cast(dict[str, Any], frozen["artifacts"])
    pcm = cast(dict[str, Any], artifacts["pcm_cache"])
    return {
        "archive": {
            "path": "/home/ubuntu/gitcode/.t/authority-003/archive.tar.gz",
            "sha256": inputs["archive_sha256"],
        },
        "manifest": {
            "inventory_sha256": inputs["manifest_inventory_sha256"],
            "path": "/home/ubuntu/gitcode/.t/authority-003/manifest.jsonl",
            "record_count": inputs["manifest_record_count"],
            "sha256": inputs["manifest_sha256"],
        },
        "pcm_cache": {
            "byte_count": pcm["byte_count"],
            "path": "/home/ubuntu/gitcode/.t/authority-003/pcm.cache",
            "sha256": pcm["sha256"],
        },
    }


def _invocation(profile: engine._AuthorityProfile) -> dict[str, Any]:
    executable = "/home/ubuntu/gitcode/.t/falsewake-venv/bin/python"
    argv = [profile.runner_entrypoint]
    return {
        "argv": argv,
        "authority_process_role": "parent_bootstrap",
        "cwd": "/home/ubuntu/gitcode/falsewake",
        "environment": {
            "LANG": "C",
            "LC_ALL": "C",
            "MKL_DYNAMIC": "FALSE",
            "MKL_NUM_THREADS": "1",
            "OMP_DYNAMIC": "FALSE",
            "OMP_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "TMPDIR": profile.scratch_root,
            "TZ": "UTC",
        },
        "orig_argv": [executable, "-I", "-S", "-B", *argv],
        "python_flags": {
            "dont_write_bytecode": 1,
            "ignore_environment": 1,
            "isolated": 1,
            "no_site": 1,
            "no_user_site": 1,
        },
        "sys_path": [
            "/home/ubuntu/gitcode/falsewake/src",
            "/usr/lib/python312.zip",
            "/usr/lib/python3.12",
            "/usr/lib/python3.12/lib-dynload",
            "/home/ubuntu/gitcode/.t/falsewake-venv/lib/python3.12/site-packages",
        ],
    }


def _registration_material(
    profile: engine._AuthorityProfile,
) -> tuple[bytes, bytes, tuple[engine._FrozenBlob, ...]]:
    source_path = "src/falsewake/experiment_002_run_authority.py"
    source_payload = engine._source_bundle_payload(((source_path, b"shared-engine"),))
    source_sha256 = hashlib.sha256(
        profile.source_bundle_domain + source_payload
    ).hexdigest()
    document = {
        "experiment": profile.registration_experiment,
        "external_inputs": _external_inputs(profile),
        "frozen_bindings": engine._expected_frozen_bindings(profile),
        "implementation_commit": "1" * 40,
        "invocation": _invocation(profile),
        "runtime": _runtime(profile),
        "schema_version": 1,
        "source_bundle": {
            "import_roots": ["src/falsewake"],
            "paths": [source_path],
            "payload_byte_count": len(source_payload),
            "sha256": source_sha256,
        },
    }
    registration = _canonical(document)
    frozen_blobs = tuple(
        engine._FrozenBlob(path=path, payload=(ROOT / path).read_bytes())
        for path, _sha256 in sorted(
            engine._frozen_file_bindings(profile),
            key=lambda item: item[0].encode("utf-8"),
        )
    )
    return registration, source_payload, frozen_blobs


def _snapshot(
    profile: engine._AuthorityProfile,
    *,
    retained: bool = False,
) -> engine._RepositorySnapshot:
    if not retained:
        return engine._RepositorySnapshot(
            repository_root=ROOT,
            head_commit="2" * 40,
            implementation_commit="1" * 40,
            registration_sha256="3" * 64,
            source_bundle_sha256="4" * 64,
            source_paths=("src/falsewake/experiment_002_run_authority.py",),
            profile=profile,
        )
    registration, source_payload, frozen_blobs = _registration_material(profile)
    return engine._RepositorySnapshot(
        repository_root=ROOT,
        head_commit="2" * 40,
        implementation_commit="1" * 40,
        registration_sha256=hashlib.sha256(registration).hexdigest(),
        source_bundle_sha256=hashlib.sha256(
            profile.source_bundle_domain + source_payload
        ).hexdigest(),
        source_paths=("src/falsewake/experiment_002_run_authority.py",),
        registration_bytes=registration,
        source_bundle_payload=source_payload,
        frozen_blobs=frozen_blobs,
        profile=profile,
    )


def test_facade_has_one_shared_capability_and_no_profile_selector() -> None:
    assert authority.VerifiedRunRegistration is engine.VerifiedRunRegistration
    assert authority._VerifiedState is engine._VerifiedState
    assert (
        "profile"
        not in inspect.signature(
            authority.verify_and_issue_experiment_003_run_registration
        ).parameters
    )
    assert (
        tuple(
            inspect.signature(
                authority.verify_and_issue_experiment_003_run_registration
            ).parameters
        )
        == ()
    )
    assert (
        tuple(
            inspect.signature(
                authority._verify_and_issue_experiment_003_sealed_child_registration
            ).parameters
        )
        == ()
    )
    assert "_ISSUANCE_COMPLETE" not in authority.__dict__
    assert "_ISSUED" not in authority.__dict__
    assert "_ISSUED_GUARDS" not in authority.__dict__


def test_profiles_bind_every_execution_namespace_exactly() -> None:
    assert P2 is not P3
    assert dataclasses.astuple(P2) == (
        "002",
        "configs/experiment-002-run.json",
        b"falsewake-exp002-runtime-v1\0",
        b"FW2CHLD1",
        b"falsewake-exp002-source-bundle-v1\0",
        "falsewake-sealed://experiment-002/",
        "/memfd:falsewake-exp002-child-bundle (deleted)",
        "falsewake-exp002-child-bundle",
        b"FW2ACTV1",
        b"falsewake-exp002-activation-ticket-v1\0",
        "verify_and_issue_experiment_002_run_registration",
        "_verify_and_issue_experiment_002_sealed_child_registration",
        "src/falsewake/experiment_002_runner.py",
        "/home/ubuntu/gitcode/.t/falsewake-experiment-002-scratch",
    )
    assert dataclasses.astuple(P3) == (
        "003",
        "configs/experiment-003-run.json",
        b"falsewake-exp003-runtime-v1\0",
        b"FW3CHLD1",
        b"falsewake-exp003-source-bundle-v1\0",
        "falsewake-sealed://experiment-003/",
        "/memfd:falsewake-exp003-child-bundle (deleted)",
        "falsewake-exp003-child-bundle",
        b"FW3ACTV1",
        b"falsewake-exp003-activation-ticket-v1\0",
        "verify_and_issue_experiment_003_run_registration",
        "_verify_and_issue_experiment_003_sealed_child_registration",
        "src/falsewake/experiment_003_runner.py",
        "/home/ubuntu/gitcode/.t/falsewake-experiment-003-scratch",
    )
    assert engine._require_authority_profile(P2) is P2
    assert engine._require_authority_profile(P3) is P3


def test_registration_parser_is_fixed_to_one_profile() -> None:
    p2_raw, _source, _frozen = _registration_material(P2)
    p3_raw, _source, _frozen = _registration_material(P3)

    assert engine._parse_registration(p2_raw).profile is P2
    assert engine._parse_registration_for_profile(p3_raw, P3).profile is P3
    with pytest.raises(engine.Experiment002RunAuthorityError):
        engine._parse_registration_for_profile(p2_raw, P3)
    with pytest.raises(engine.Experiment002RunAuthorityError):
        engine._parse_registration_for_profile(p3_raw, P2)


def test_runtime_and_source_domains_are_not_cross_profile_compatible() -> None:
    fields = {
        "numpy_version": "2.5.1",
        "python_executable": "/home/ubuntu/gitcode/.t/falsewake-venv/bin/python",
        "python_sys_path_roots": [],
        "python_version": "3.12.3",
        "safetensors_version": "0.8.0",
        "torch_version": "2.13.0+cpu",
    }
    assert engine._runtime_fingerprint_sha256(
        fields, P2
    ) != engine._runtime_fingerprint_sha256(fields, P3)
    sources = (("src/falsewake/probe.py", b"probe"),)
    assert engine._source_bundle_digest(sources, P2) != engine._source_bundle_digest(
        sources, P3
    )


def test_child_bundle_magic_and_registration_are_cross_profile_rejected() -> None:
    p3_state = engine._state_from_snapshot(_snapshot(P3, retained=True))
    bundle = engine._child_bundle_bytes_from_state(p3_state)
    frame = engine._parse_experiment_003_child_bundle(bundle)

    assert frame.profile is P3
    assert frame.registration_bytes == p3_state.registration_bytes
    with pytest.raises(engine.Experiment002RunAuthorityError):
        engine._parse_experiment_002_child_bundle(bundle)

    wrong_magic = b"FW2CHLD1" + bundle[8:]
    with pytest.raises(engine.Experiment002RunAuthorityError):
        engine._parse_experiment_003_child_bundle(wrong_magic)
    with pytest.raises(engine.Experiment002RunAuthorityError):
        engine._parse_experiment_002_child_bundle(wrong_magic)


def test_sealed_origin_prefix_is_profile_bound() -> None:
    digest = "a" * 64
    origin = engine._sealed_authority_origin(digest, P3)

    assert origin.startswith("falsewake-sealed://experiment-003/")
    assert engine._sealed_origin_source_digest(origin, P3) == digest
    with pytest.raises(engine.Experiment002RunAuthorityError):
        engine._sealed_origin_source_digest(origin, P2)


def test_fixed_verifiers_mutually_reject_cross_profile_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine, "reverify_verified_run_registration", lambda _cap: None)
    p2_capability = engine._issue_controlled_snapshot_for_tests(_snapshot(P2))
    engine.verify_verified_run_registration(p2_capability)
    with pytest.raises(
        engine.Experiment002RunAuthorityError,
        match="different authority profile",
    ):
        authority.verify_verified_run_registration(p2_capability)

    _reset_issuer(monkeypatch)
    p3_capability = engine._issue_controlled_snapshot_for_tests(_snapshot(P3))
    authority.verify_verified_run_registration(p3_capability)
    with pytest.raises(
        engine.Experiment002RunAuthorityError,
        match="different authority profile",
    ):
        engine.verify_verified_run_registration(p3_capability)


def test_generic_reverify_dispatches_from_retained_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ProbeError(RuntimeError):
        pass

    observed: list[Path] = []

    def probe(path: Path, *, maximum_bytes: int) -> bytes:
        del maximum_bytes
        observed.append(path)
        raise ProbeError

    monkeypatch.setattr(engine, "_read_regular_file", probe)
    capability = engine._issue_controlled_snapshot_for_tests(_snapshot(P3))
    with pytest.raises(ProbeError):
        engine.reverify_verified_run_registration(capability)
    assert observed == [ROOT / "configs/experiment-003-run.json"]


@pytest.mark.parametrize("target", ["state", "guard"])
def test_profile_tamper_permanently_poisons_capability(
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    capability = engine._issue_controlled_snapshot_for_tests(_snapshot(P3))
    original_state = engine._ISSUED[capability]
    original_guard = engine._ISSUED_GUARDS[capability]
    if target == "state":
        engine._ISSUED[capability] = dataclasses.replace(
            original_state,
            profile=P2,
        )
    else:
        engine._ISSUED_GUARDS[capability] = dataclasses.replace(
            original_guard,
            profile=P2,
        )

    with pytest.raises(engine.Experiment002RunAuthorityError):
        engine._verified_state(capability)
    engine._ISSUED[capability] = original_state
    engine._ISSUED_GUARDS[capability] = original_guard
    with pytest.raises(engine.Experiment002RunAuthorityError):
        engine._verified_state(capability)
    assert capability in engine._FAILED


def test_shared_latch_allows_only_one_profile_under_concurrency() -> None:
    barrier = threading.Barrier(3)

    def issue(
        snapshot: engine._RepositorySnapshot,
    ) -> engine.VerifiedRunRegistration:
        barrier.wait()
        return engine._issue_controlled_snapshot_for_tests(snapshot)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(issue, _snapshot(P2)),
            executor.submit(issue, _snapshot(P3)),
        ]
        barrier.wait()
        outcomes: list[engine.VerifiedRunRegistration | BaseException] = []
        for future in futures:
            try:
                outcomes.append(future.result())
            except BaseException as error:
                outcomes.append(error)

    assert sum(type(value) is engine.VerifiedRunRegistration for value in outcomes) == 1
    assert (
        sum(type(value) is engine.Experiment002RunAuthorityError for value in outcomes)
        == 1
    )
    assert engine._ISSUANCE_COMPLETE is True


def test_p3_frozen_bindings_add_protocol_and_immutable_predecessor_only() -> None:
    p2 = engine._expected_frozen_bindings(P2)
    registered_p2 = json.loads(
        (ROOT / "configs" / "experiment-002-run.json").read_text(encoding="ascii")
    )["frozen_bindings"]
    assert p2 == registered_p2

    p3 = engine._expected_frozen_bindings(P3)
    assert {key for key in p3 if key not in p2} == {
        "execution_protocol",
        "predecessor",
    }
    assert p3["execution_protocol"] == {
        "introduction_commit": "e47bd581675abe22b529de7bcc825d39e84a00cd",
        "path": "configs/experiment-003-execution.json",
        "sha256": "3f48cb48f6c3a56284f272fa9e308ae62b3749df64cb7a581e770da8bdd8218a",
    }
    predecessor = cast(dict[str, Any], p3["predecessor"])
    assert predecessor["experiment"] == "002"
    assert predecessor["terminal"] is True
    assert predecessor["reuse_forbidden"] is True
    assert predecessor["scientific_protocol"] == "002"
