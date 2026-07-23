from __future__ import annotations

import hashlib
import json
import re
import stat
import subprocess
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs" / "experiment-003-execution.json"
RUN_CONFIG_PATH = ROOT / "configs" / "experiment-003-run.json"
REFERENCE_IMPLEMENTATION = "21bbcc56e898beb44edc8e3ff775fa9642f87577"
REPAIR_COMMIT = "650eefbf9f82b207ed58e3b1a1eac41197466b41"
EXPECTED_PRODUCTION_ADDITIONS = (
    "src/falsewake/experiment_003_run_authority.py",
    "src/falsewake/experiment_003_runner.py",
    "src/falsewake/experiment_003_supervisor.py",
    "src/falsewake/experiment_003_coordinator.py",
    "src/falsewake/experiment_003_seed_worker.py",
    "src/falsewake/experiment_003_final_evidence.py",
    "src/falsewake/experiment_003_final_publication.py",
)


def _reject_duplicate_key(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_float(value: str) -> None:
    raise ValueError(f"JSON floats are forbidden: {value}")


def _strict_json(payload: bytes) -> dict[str, Any]:
    assert not payload.startswith(b"\xef\xbb\xbf")
    text = payload.decode("ascii")
    document = json.loads(
        text,
        object_pairs_hook=_reject_duplicate_key,
        parse_float=_reject_float,
        parse_constant=_reject_float,
    )
    assert type(document) is dict
    return document


def _canonical(document: dict[str, Any], *, compact: bool = False) -> bytes:
    separators = (",", ":") if compact else None
    return (
        json.dumps(
            document,
            indent=None if compact else 2,
            sort_keys=True,
            separators=separators,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git(*arguments: str) -> bytes:
    try:
        result = subprocess.run(
            ("git", *arguments),
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        pytest.fail(f"Git repository evidence is unavailable: {error}")
    return result.stdout


def test_execution_protocol_is_canonical_and_fixed_profile() -> None:
    payload = PROTOCOL_PATH.read_bytes()
    document = _strict_json(payload)

    assert payload == _canonical(document)
    assert tuple(document) == (
        "attempt_contract",
        "authority_profile",
        "evidence_contract",
        "execution_delta",
        "experiment",
        "kind",
        "lifecycle",
        "namespace",
        "predecessor",
        "schema_version",
        "scientific_inheritance",
    )
    assert document["schema_version"] == 1
    assert document["experiment"] == "003"
    assert document["kind"] == "execution_only_recovery"

    authority = document["authority_profile"]
    assert authority == {
        "capability_state_profile": "003",
        "capability_type": (
            "falsewake.experiment_002_run_authority.VerifiedRunRegistration"
        ),
        "coordinator_profile_check_required": True,
        "cross_profile_acceptance": "forbidden",
        "cross_profile_rejections": [
            "experiment_002_capability_at_experiment_003_boundary",
            "experiment_003_capability_at_experiment_002_boundary",
        ],
        "facade": "src/falsewake/experiment_003_run_authority.py",
        "issuer_caller_supplied_values": [],
        "issuer_route": "verify_and_issue_experiment_003_run_registration",
        "logical_authority": "003",
        "process_issuance_latch": "shared_across_all_profiles",
        "profile_binding_fields": [
            "registration_experiment",
            "run_config_path",
            "runtime_fingerprint_domain",
            "child_bundle_magic",
            "source_bundle_domain",
            "sealed_origin_prefix",
            "activation_magic",
            "activation_ticket_domain",
            "issuer_route",
            "sealed_child_issuer_route",
        ],
        "publisher_profile_check_required": True,
        "reverify_dispatch": "retained_capability_state_profile_only",
        "sealed_child_issuer_route": (
            "_verify_and_issue_experiment_003_sealed_child_registration"
        ),
        "shared_engine": "src/falsewake/experiment_002_run_authority.py",
    }
    delta = document["execution_delta"]
    assert delta["scientific_protocol_change"] is False
    assert tuple(delta["production_additions"]) == EXPECTED_PRODUCTION_ADDITIONS
    assert delta["permitted_existing_source_changes"] == [
        {
            "path": "src/falsewake/experiment_002_run_authority.py",
            "scope": (
                "extend_in_place_as_a_profile_bound_shared_authority_engine_without_"
                "changing_"
                "scientific_inputs_or_registered_training_semantics"
            ),
        }
    ]
    parity = delta["procfs_repair"]["experiment_003_parity"]
    assert parity["normalized_source_equality_required"] is True
    assert parity["reference_path"] == "src/falsewake/experiment_002_supervisor.py"
    assert (
        parity["reference_sha256"]
        == "f441f2ead5566094364572138d69256a072165ffb2cc80a71340b4087ba5a8f0"
    )
    assert parity["forbidden_candidate_markers"] == [
        "Experiment 002",
        "Experiment002",
        "src/falsewake/experiment_002_runner.py",
        "falsewake-experiment-002-scratch",
        "falsewake-experiment-002-staging",
        "falsewake-experiment-002-launch",
        "falsewake-exp002-child-activation",
        "from falsewake.experiment_003_child_result import",
    ]
    assert parity["immutable_limits"] == {
        "activation_join_milliseconds_maximum": 35000,
        "activation_start_grace_milliseconds": 50,
        "cpu_count": 2,
        "output_bytes_maximum": 4294967296,
        "poll_interval_maximum_nanoseconds": 100000000,
        "poll_interval_nanoseconds": 25000000,
        "rss_bytes_maximum": 8589934592,
        "termination_nanoseconds_maximum": 10000000000,
        "thread_retire_nanoseconds_maximum": 100000000,
        "wall_nanoseconds_maximum": 21600000000000,
    }
    assert parity["normalization"] == {
        "application": (
            "replace_the_one_exact_runner_sha256_assignment_value_then_apply_each_"
            "literal_substitution_globally_in_list_order"
        ),
        "direction": ("experiment_003_candidate_to_repaired_experiment_002_reference"),
        "encoding": "UTF-8",
        "ordered_literal_substitutions": [
            {"candidate": "Experiment 003", "reference": "Experiment 002"},
            {"candidate": "Experiment003", "reference": "Experiment002"},
            {"candidate": "experiment_003", "reference": "experiment_002"},
            {"candidate": "experiment-003", "reference": "experiment-002"},
            {"candidate": "exp003", "reference": "exp002"},
        ],
        "runner_sha256_assignment": {
            "candidate_value": "one_exact_future_64_lowercase_hex_digest",
            "normalize_to_reference_value": (
                "3aaa25b54670c357d528887d582d39e5fb0b815edb86201a7eef793c92b4931d"
            ),
            "target": "_RUNNER_SHA256",
        },
    }
    assert parity["required_candidate_markers"] == [
        "Experiment 003",
        "Experiment003SupervisorError",
        "src/falsewake/experiment_003_runner.py",
        "falsewake-experiment-003-scratch",
        "falsewake-experiment-003-staging",
        "falsewake-experiment-003-launch",
        "falsewake-exp003-child-activation",
    ]
    assert parity["required_inherited_markers"] == [
        "from falsewake.experiment_002_child_result import"
    ]

    lifecycle = document["lifecycle"]
    assert lifecycle["attempt_limit"] == 1
    assert lifecycle["execution_config_authorizes_registered_training"] is False
    assert lifecycle["execution_protocol"] == {
        "committed_blob_verification_required": True,
        "introduction_commit_binding_required": True,
        "introduction_commit_must_be_direct_child_of": REPAIR_COMMIT,
        "introduction_commit_must_precede_implementation_commit": True,
        "path": "configs/experiment-003-execution.json",
        "run_config_binding_path": "frozen_bindings.execution_protocol",
        "run_config_sha256_binding_required": True,
    }
    assert lifecycle["pre_registration_observations"] == {
        "registered_optimizer_updates": 0,
        "registered_validation_examples": 0,
    }
    assert lifecycle["implementation_commit_must_descend_from"] == REPAIR_COMMIT
    assert lifecycle["implementation_commit_required_production_additions"] == (
        "execution_delta.production_additions"
    )
    assert lifecycle["implementation_commit_must_exclude"] == [
        "configs/experiment-003-run.json",
        "models/experiment-003-selected.safetensors",
        "reports/experiment-003-seed-20260719-history.json",
        "reports/experiment-003-seed-20260720-history.json",
        "reports/experiment-003-seed-20260721-history.json",
        "reports/experiment-003-selected-rerun-history.json",
        "reports/experiment-003-training.json",
    ]
    assert lifecycle["registration_commit"] == {
        "execution_protocol_binding": "lifecycle.execution_protocol",
        "only_change": "A configs/experiment-003-run.json",
        "parent": "implementation_commit",
        "required_mode": "100644",
    }
    assert lifecycle["retry_policy"] == (
        "any_terminal_attempt_requires_a_new_experiment_identifier"
    )
    assert lifecycle["terminal_execution_failure"] == {
        "artifacts": [],
        "checkpoint_reusable": False,
    }
    attempt = document["attempt_contract"]
    assert attempt["attempt_consumption_boundary"] == "before_parent_route_claim"
    assert attempt["cross_process_ownership"] == {
        "mechanism": "fresh_exclusive_staging_root",
        "stale_or_existing_namespace": "reject_without_removal_or_reuse",
    }
    assert attempt["registered_invocation_count"] == 1
    assert attempt["retry_allowed"] is False
    assert attempt["failure_is_terminal"] is True
    assert attempt["invoked_admission_failure_consumes_attempt"] is True
    assert attempt["preflight_rejection_consumes_registered_invocation"] is False
    assert attempt["reuse_predecessor_checkpoint"] is False
    assert attempt["child_plan"]["training"] == [
        {"ordinal": 0, "role": "training_seed", "seed": 20260719},
        {"ordinal": 1, "role": "training_seed", "seed": 20260720},
        {"ordinal": 2, "role": "training_seed", "seed": 20260721},
    ]
    assert attempt["child_plan"]["rerun"] == {
        "ordinal": 3,
        "role": "selected_seed_rerun",
        "seed_source": "selected_training_seed",
    }

    evidence = document["evidence_contract"]
    assert evidence["outer_experiment"] == "003"
    assert evidence["scientific_protocol"] == "002"
    assert evidence["predecessor_binding_required"] is True
    assert evidence["report_last"] is True
    assert evidence["report_mode"] == "0444"
    assert evidence["failure_pairs"] == [
        {"code": "staging_prepare_failed", "phase": "parent_setup"},
        {"code": "cpu_affinity_capture_failed", "phase": "parent_setup"},
        {"code": "source_bundle_create_failed", "phase": "parent_setup"},
        {
            "code": "source_bundle_close_failed",
            "phase": "registered_execution",
        },
        {"code": "seed_selection_failed", "phase": "registered_execution"},
        {
            "code": "completed_evidence_rejected",
            "phase": "completed_evidence",
        },
        {
            "code": "pre_cleanup_quiescence_failed",
            "phase": "parent_boundary",
        },
    ]
    histories = [
        "reports/experiment-003-seed-20260719-history.json",
        "reports/experiment-003-seed-20260720-history.json",
        "reports/experiment-003-seed-20260721-history.json",
        "reports/experiment-003-selected-rerun-history.json",
    ]
    assert evidence["statuses"]["execution_failure"] == {
        "artifact_paths": [],
        "checkpoint_reusable": False,
    }
    assert evidence["statuses"]["gate_failure"] == {
        "artifact_paths": histories,
        "checkpoint_reusable": False,
    }
    assert evidence["statuses"]["pass"] == {
        "artifact_paths": [
            *histories,
            "models/experiment-003-selected.safetensors",
        ],
        "checkpoint_reusable": True,
    }

    namespace = document["namespace"]
    assert namespace["authority_module"] == "falsewake.experiment_003_run_authority"
    assert namespace["child_control_fd"] == 3
    assert namespace["child_source_bundle_fd"] == 7
    assert namespace["run_config"] == "configs/experiment-003-run.json"
    assert namespace["runner"] == "src/falsewake/experiment_003_runner.py"
    assert (
        namespace["scratch_root"]
        == "/home/ubuntu/gitcode/.t/falsewake-experiment-003-scratch"
    )
    assert (
        namespace["staging_root"]
        == "/home/ubuntu/gitcode/.t/falsewake-experiment-003-staging"
    )
    assert (
        namespace["sealed_child_memfd_target"]
        == "/memfd:falsewake-exp003-child-bundle (deleted)"
    )
    assert namespace["temporary_publication_prefix"] == (
        ".falsewake-exp003-publication-"
    )
    assert namespace["managed_outputs"] == {
        "histories": histories[:3],
        "model": "models/experiment-003-selected.safetensors",
        "report": "reports/experiment-003-training.json",
        "selected_rerun_history": histories[3],
    }
    assert namespace["transport"] == {
        "activation_magic": "FW3ACTV1",
        "activation_ticket_domain": "falsewake-exp003-activation-ticket-v1\x00",
        "child_bundle_magic": "FW3CHLD1",
        "inherited_child_result_envelope_domain": (
            "falsewake-exp002-child-result-envelope-v1\x00"
        ),
        "inherited_process_guard_domain": (
            "falsewake-exp002-process-guard-filter-v1\x00"
        ),
        "runtime_fingerprint_domain": "falsewake-exp003-runtime-v1\x00",
        "sealed_origin_prefix": "falsewake-sealed://experiment-003",
        "source_bundle_domain": "falsewake-exp003-source-bundle-v1\x00",
    }


def test_predecessor_failure_and_incident_are_exact() -> None:
    protocol = _strict_json(PROTOCOL_PATH.read_bytes())
    predecessor = protocol["predecessor"]
    assert predecessor["experiment"] == "002"
    assert predecessor["reuse_forbidden"] is True
    assert predecessor["terminal"] is True
    assert predecessor["scientific_protocol"] == "002"

    registration = predecessor["registration"]
    registration_path = ROOT / registration["path"]
    assert _sha256(registration_path) == registration["sha256"]
    registration_blob = _git(
        "show",
        f"{registration['commit']}:{registration['path']}",
    )
    assert _payload_sha256(registration_blob) == registration["sha256"]
    assert registration_blob == registration_path.read_bytes()
    assert (
        registration["source_bundle_sha256"]
        == "840c70f6417245d0c3eb6237bfda9fc88161abcca07dc2c7325bc4287db0653b"
    )

    outcome = predecessor["outcome"]
    outcome_path = ROOT / outcome["path"]
    outcome_stat = outcome_path.lstat()
    assert stat.S_ISREG(outcome_stat.st_mode)
    assert stat.S_IMODE(outcome_stat.st_mode) == 0o444
    assert outcome_stat.st_nlink == 1
    assert _sha256(outcome_path) == outcome["sha256"]
    outcome_blob = _git("show", f"{outcome['commit']}:{outcome['path']}")
    assert _payload_sha256(outcome_blob) == outcome["sha256"]
    assert outcome_blob == outcome_path.read_bytes()
    outcome_document = _strict_json(outcome_path.read_bytes())
    assert outcome_path.read_bytes() == _canonical(outcome_document, compact=True)
    assert outcome_document["status"] == outcome["status"] == "execution_failure"
    assert outcome["code"] == "seed_selection_failed"
    assert outcome["phase"] == "registered_execution"
    assert outcome_document["failure"] == {
        "code": "seed_selection_failed",
        "phase": "registered_execution",
    }
    assert outcome_document["artifacts"] == []
    assert outcome_document["checkpoint_reusable"] is False

    incident = predecessor["incident"]
    incident_path = ROOT / incident["path"]
    assert _sha256(incident_path) == incident["sha256"]
    incident_blob = _git("show", f"{incident['commit']}:{incident['path']}")
    assert _payload_sha256(incident_blob) == incident["sha256"]
    assert incident_blob == incident_path.read_bytes()
    incident_document = _strict_json(incident_path.read_bytes())
    assert incident_path.read_bytes() == _canonical(incident_document)
    assert incident_document["experiment"] == "002"
    assert incident_document["outcome"]["status"] == "execution_failure"
    assert (
        incident_document["diagnosis"]["published_scientific_result_admitted"] is False
    )
    assert incident_document["remediation"]["next_experiment"] == "003"
    assert incident_document["remediation"]["reuse_experiment_002"] is False

    expected_chain = (
        (
            predecessor["registration"]["commit"],
            predecessor["implementation_commit"],
        ),
        (
            predecessor["outcome"]["commit"],
            predecessor["registration"]["commit"],
        ),
        (
            predecessor["incident"]["commit"],
            predecessor["outcome"]["commit"],
        ),
    )
    for commit, parent in expected_chain:
        fields = (
            _git("rev-list", "--parents", "-n", "1", commit)
            .decode("ascii")
            .strip()
            .split(" ")
        )
        assert fields == [commit, parent]

    repair = protocol["execution_delta"]["procfs_repair"]
    assert repair["affected_path"] == "src/falsewake/experiment_002_supervisor.py"
    assert repair["repair_commit"] == REPAIR_COMMIT
    failure_supervisor = _git(
        "show",
        f"{predecessor['implementation_commit']}:{repair['affected_path']}",
    )
    repaired_supervisor = _git(
        "show",
        f"{repair['repair_commit']}:{repair['affected_path']}",
    )
    assert _payload_sha256(failure_supervisor) == repair["failure_sha256"]
    assert _payload_sha256(repaired_supervisor) == repair["repair_sha256"]
    assert repaired_supervisor == (ROOT / repair["affected_path"]).read_bytes()

    repair_changes = _git(
        "diff-tree",
        "--no-commit-id",
        "--name-status",
        "-r",
        "-z",
        predecessor["outcome"]["commit"],
        repair["repair_commit"],
    )
    repair_fields = repair_changes.rstrip(b"\x00").split(b"\x00")
    assert tuple(zip(repair_fields[::2], repair_fields[1::2], strict=True)) == (
        (b"M", b"README.md"),
        (b"M", b"docs/experiment-002.md"),
        (b"A", b"reports/experiment-002-execution-incident.json"),
        (b"M", b"src/falsewake/experiment_002_supervisor.py"),
        (b"M", b"tests/test_experiment_002.py"),
        (b"M", b"tests/test_experiment_002_final_publication.py"),
        (b"M", b"tests/test_experiment_002_supervisor.py"),
    )


def test_frozen_science_and_training_invariants_match_their_sources() -> None:
    protocol = _strict_json(PROTOCOL_PATH.read_bytes())
    science = protocol["scientific_inheritance"]

    expected_frozen_paths = (
        "configs/experiment-002-training.json",
        "configs/experiment-002-numerics.json",
        "configs/experiment-002-trainer.json",
        "models/experiment-002-normalization.f32",
        "reports/experiment-002-normalization.json",
        "reports/experiment-002-pcm-cache.json",
        "reports/experiment-002-implementation-audit.json",
        "src/falsewake/causal_kws.py",
    )
    assert tuple(binding["path"] for binding in science["frozen_files"]) == (
        expected_frozen_paths
    )
    expected_frozen_sha256 = {
        "configs/experiment-002-training.json": (
            "a4a21ff66ef04bf15950853acda0897f24ed01eecf566c795a7330e69593d44a"
        ),
        "configs/experiment-002-numerics.json": (
            "d11fc50ba8f609551dbf88d7863a5b45cd138c6681da6c10ca4c0a9eaf8f2be9"
        ),
        "configs/experiment-002-trainer.json": (
            "768720d031a1f2f6d0a36466fd9b4fa095a2f1b9a6fbbc382d4af5fc51514b48"
        ),
        "models/experiment-002-normalization.f32": (
            "891900d4c36fa8a71ba429384f3f4ff594de7c81347ed846c0f790dc4579268e"
        ),
        "reports/experiment-002-normalization.json": (
            "2b95565fec7dc956a4b1f667638b87e3df64cfaaca3026017942a8057c5187a8"
        ),
        "reports/experiment-002-pcm-cache.json": (
            "a07b02a78bb6e0d2eef8548ec71a48e717f26c756548665f004b0ca1bf3e52ca"
        ),
        "reports/experiment-002-implementation-audit.json": (
            "0b8033e7f407550d1c8576d31485e5bbcf825592de0d3b86f42ce98c1fca7116"
        ),
        "src/falsewake/causal_kws.py": (
            "fa5d86aa218eaad7503b6e66ed9faa18e34e917696ccb98dc5ee4f1b2267c4a6"
        ),
    }
    for binding in science["frozen_files"]:
        path = ROOT / binding["path"]
        assert binding["sha256"] == expected_frozen_sha256[binding["path"]]
        assert _sha256(path) == binding["sha256"]
        if "byte_count" in binding:
            assert path.stat().st_size == binding["byte_count"]

    phase_1 = json.loads(
        (ROOT / "configs" / "experiment-002-training.json").read_text(encoding="ascii")
    )
    trainer = json.loads(
        (ROOT / "configs" / "experiment-002-trainer.json").read_text(encoding="ascii")
    )
    registered = science["registered_training"]
    assert registered == {
        "class_count": 12,
        "epochs_per_seed": 30,
        "parameter_tensor_count": 53,
        "parameter_value_count": 23724,
        "seeds": [20260719, 20260720, 20260721],
        "selected_seed_rerun_required": True,
        "total_updates_per_seed": 9390,
        "training_example_count_per_epoch": 40027,
        "updates_per_epoch": 313,
        "validation_example_count": 10583,
    }
    assert science["runtime"] == {
        "numpy_version": "2.5.1",
        "python_version": "3.12.3",
        "safetensors_version": "0.8.0",
        "torch_version": "2.13.0+cpu",
    }
    assert science["source_policy"] == {
        "baseline_commit": REFERENCE_IMPLEMENTATION,
        "immutable_paths": (
            "every_path_in_the_predecessor_source_bundle_except_execution_delta."
            "permitted_existing_source_changes_and_execution_delta.procfs_repair."
            "affected_path"
        ),
        "new_source_paths": "exactly_execution_delta.production_additions",
        "predecessor_source_bundle_sha256": (
            "840c70f6417245d0c3eb6237bfda9fc88161abcca07dc2c7325bc4287db0653b"
        ),
        "repaired_path": (
            "execution_delta.procfs_repair.affected_path_must_equal_execution_delta."
            "procfs_repair.repair_sha256"
        ),
    }
    assert phase_1["seeds"]["training"] == registered["seeds"]
    assert phase_1["training"]["epochs"] == registered["epochs_per_seed"]
    assert phase_1["training"]["updates_per_epoch"] == registered["updates_per_epoch"]
    assert phase_1["training"]["total_updates"] == registered["total_updates_per_seed"]
    assert trainer["training_execution"]["seed_order"] == registered["seeds"]
    assert trainer["training_execution"]["epochs"] == registered["epochs_per_seed"]
    assert (
        trainer["training_execution"]["updates_per_epoch"]
        == registered["updates_per_epoch"]
    )
    assert (
        trainer["training_execution"]["total_updates_per_seed"]
        == registered["total_updates_per_seed"]
    )

    predecessor_config = _strict_json(
        (ROOT / protocol["predecessor"]["registration"]["path"]).read_bytes()
    )
    external = science["external_inputs"]
    predecessor_external = predecessor_config["external_inputs"]
    assert external["archive_path"] == predecessor_external["archive"]["path"]
    assert external["archive_sha256"] == predecessor_external["archive"]["sha256"]
    assert external["manifest_path"] == predecessor_external["manifest"]["path"]
    assert (
        external["manifest_inventory_sha256"]
        == predecessor_external["manifest"]["inventory_sha256"]
    )
    assert (
        external["manifest_record_count"]
        == predecessor_external["manifest"]["record_count"]
    )
    assert external["manifest_sha256"] == predecessor_external["manifest"]["sha256"]
    assert external["pcm_cache_path"] == predecessor_external["pcm_cache"]["path"]
    assert (
        external["pcm_cache_byte_count"]
        == predecessor_external["pcm_cache"]["byte_count"]
    )
    assert external["pcm_cache_sha256"] == predecessor_external["pcm_cache"]["sha256"]
    predecessor_inputs = predecessor_config["frozen_bindings"]["inputs"]
    assert (
        external["train_command_inventory_sha256"]
        == predecessor_inputs["train_command_inventory_sha256"]
    )
    assert (
        external["validation_command_inventory_sha256"]
        == predecessor_inputs["validation_command_inventory_sha256"]
    )

    for path, expected_sha256 in science["source_blobs"].items():
        current = ROOT / path
        assert _sha256(current) == expected_sha256
        reference = _git("show", f"{REFERENCE_IMPLEMENTATION}:{path}")
        assert hashlib.sha256(reference).hexdigest() == expected_sha256
        assert current.read_bytes() == reference


def test_predecessor_source_paths_are_immutable_except_declared_execution_code() -> (
    None
):
    protocol = _strict_json(PROTOCOL_PATH.read_bytes())
    predecessor_config = _strict_json(
        (ROOT / protocol["predecessor"]["registration"]["path"]).read_bytes()
    )
    source_paths = predecessor_config["source_bundle"]["paths"]
    allowed = {
        binding["path"]
        for binding in protocol["execution_delta"]["permitted_existing_source_changes"]
    }
    repaired = protocol["execution_delta"]["procfs_repair"]
    repaired_path = repaired["affected_path"]

    assert repaired_path not in allowed
    assert _sha256(ROOT / repaired_path) == repaired["repair_sha256"]
    for path in source_paths:
        if path in allowed or path == repaired_path:
            continue
        current = (ROOT / path).read_bytes()
        reference = _git("show", f"{REFERENCE_IMPLEMENTATION}:{path}")
        assert current == reference, path


def test_registration_path_is_absent_or_has_a_config_only_introduction() -> None:
    if not RUN_CONFIG_PATH.exists():
        return

    relative = RUN_CONFIG_PATH.relative_to(ROOT).as_posix()
    commits = (
        _git(
            "log",
            "--reverse",
            "--diff-filter=A",
            "--format=%H",
            "--",
            relative,
        )
        .decode("ascii")
        .splitlines()
    )
    assert len(commits) == 1
    registration_commit = commits[0]
    parent_line = (
        _git(
            "rev-list",
            "--parents",
            "-n",
            "1",
            registration_commit,
        )
        .decode("ascii")
        .strip()
    )
    commit, implementation_commit = parent_line.split(" ")
    assert commit == registration_commit

    changes = _git(
        "diff-tree",
        "--no-commit-id",
        "--name-status",
        "-r",
        "-z",
        implementation_commit,
        registration_commit,
    )
    assert changes == b"A\x00configs/experiment-003-run.json\x00"
    mode_line = _git(
        "ls-tree",
        registration_commit,
        "--",
        relative,
    ).decode("ascii")
    assert mode_line.startswith("100644 blob ")
    run_payload = RUN_CONFIG_PATH.read_bytes()
    run_document = _strict_json(run_payload)
    assert run_payload == _canonical(run_document)
    committed_run_payload = _git(
        "show",
        f"{registration_commit}:{relative}",
    )
    assert committed_run_payload == run_payload
    assert run_document["experiment"] == "003"
    assert run_document["implementation_commit"] == implementation_commit
    protocol_commits = (
        _git(
            "log",
            "--reverse",
            "--diff-filter=A",
            "--format=%H",
            "--",
            PROTOCOL_PATH.relative_to(ROOT).as_posix(),
        )
        .decode("ascii")
        .splitlines()
    )
    assert len(protocol_commits) == 1
    _git(
        "merge-base",
        "--is-ancestor",
        protocol_commits[0],
        implementation_commit,
    )
    assert run_document["frozen_bindings"]["execution_protocol"] == {
        "introduction_commit": protocol_commits[0],
        "path": "configs/experiment-003-execution.json",
        "sha256": _sha256(PROTOCOL_PATH),
    }

    protocol = _strict_json(PROTOCOL_PATH.read_bytes())
    expected_source_changes = {
        binding["path"]
        for binding in protocol["execution_delta"]["permitted_existing_source_changes"]
    }
    expected_source_changes.update(protocol["execution_delta"]["production_additions"])
    source_changes = _git(
        "diff",
        "--name-only",
        "-z",
        REPAIR_COMMIT,
        implementation_commit,
        "--",
        "src/falsewake",
    )
    observed_source_changes = {
        path.decode("utf-8")
        for path in source_changes.rstrip(b"\x00").split(b"\x00")
        if path
    }
    assert observed_source_changes == expected_source_changes
    for path in protocol["execution_delta"]["production_additions"]:
        tree_line = _git("ls-tree", implementation_commit, "--", path).decode("ascii")
        assert tree_line.startswith("100644 blob ")
    for path, expected_sha256 in protocol["scientific_inheritance"][
        "source_blobs"
    ].items():
        implementation_payload = _git(
            "show",
            f"{implementation_commit}:{path}",
        )
        assert _payload_sha256(implementation_payload) == expected_sha256

    parity = protocol["execution_delta"]["procfs_repair"]["experiment_003_parity"]
    candidate_supervisor = _git(
        "show",
        f"{implementation_commit}:src/falsewake/experiment_003_supervisor.py",
    )
    candidate_runner = _git(
        "show",
        f"{implementation_commit}:src/falsewake/experiment_003_runner.py",
    )
    candidate_text = candidate_supervisor.decode(parity["normalization"]["encoding"])
    for marker in parity["required_candidate_markers"]:
        assert marker in candidate_text
    for marker in parity["required_inherited_markers"]:
        assert candidate_text.count(marker) == 1
    for forbidden_marker in parity["forbidden_candidate_markers"]:
        assert forbidden_marker not in candidate_text

    runner_sha256_matches = re.findall(
        r'^_RUNNER_SHA256: Final = \(\n    "([0-9a-f]{64})"\n\)$',
        candidate_text,
        flags=re.MULTILINE,
    )
    assert len(runner_sha256_matches) == 1
    candidate_runner_sha256 = runner_sha256_matches[0]
    assert candidate_runner_sha256 == _payload_sha256(candidate_runner)

    normalization = parity["normalization"]
    runner_assignment = normalization["runner_sha256_assignment"]
    candidate_assignment = (
        f'_RUNNER_SHA256: Final = (\n    "{candidate_runner_sha256}"\n)'
    )
    reference_assignment = (
        '_RUNNER_SHA256: Final = (\n    "'
        f'{runner_assignment["normalize_to_reference_value"]}"\n)'
    )
    assert candidate_text.count(candidate_assignment) == 1
    normalized_text = candidate_text.replace(
        candidate_assignment,
        reference_assignment,
        1,
    )
    for substitution in normalization["ordered_literal_substitutions"]:
        candidate_literal = substitution["candidate"]
        assert normalized_text.count(candidate_literal) >= 1
        normalized_text = normalized_text.replace(
            candidate_literal,
            substitution["reference"],
        )

    repaired_supervisor = _git(
        "show",
        (
            f"{protocol['execution_delta']['procfs_repair']['repair_commit']}:"
            f"{parity['reference_path']}"
        ),
    )
    assert _payload_sha256(repaired_supervisor) == parity["reference_sha256"]
    assert normalized_text.encode(normalization["encoding"]) == repaired_supervisor
    assert (
        _git(
            "ls-tree",
            implementation_commit,
            "--",
            "configs/experiment-003-run.json",
        )
        == b""
    )
    for path in protocol["lifecycle"]["implementation_commit_must_exclude"][1:]:
        assert _git("ls-tree", implementation_commit, "--", path) == b""


def test_protocol_freeze_commit_precedes_every_execution_source() -> None:
    relative = PROTOCOL_PATH.relative_to(ROOT).as_posix()
    commits = (
        _git(
            "log",
            "--reverse",
            "--diff-filter=A",
            "--format=%H",
            "--",
            relative,
        )
        .decode("ascii")
        .splitlines()
    )
    if not commits:
        return

    assert len(commits) == 1
    freeze_commit = commits[0]
    parent_fields = (
        _git("rev-list", "--parents", "-n", "1", freeze_commit)
        .decode("ascii")
        .strip()
        .split(" ")
    )
    assert parent_fields == [freeze_commit, REPAIR_COMMIT]
    changes = _git(
        "diff-tree",
        "--no-commit-id",
        "--name-status",
        "-r",
        "-z",
        REPAIR_COMMIT,
        freeze_commit,
    )
    assert changes == b"A\x00configs/experiment-003-execution.json\x00"
    mode_line = _git("ls-tree", freeze_commit, "--", relative).decode("ascii")
    assert mode_line.startswith("100644 blob ")
    freeze_payload = _git("show", f"{freeze_commit}:{relative}")
    current_payload = PROTOCOL_PATH.read_bytes()
    assert freeze_payload == current_payload
    assert freeze_payload == _canonical(_strict_json(freeze_payload))

    protocol = _strict_json(PROTOCOL_PATH.read_bytes())
    absent_paths = (
        *protocol["execution_delta"]["production_additions"],
        protocol["namespace"]["run_config"],
        *protocol["lifecycle"]["implementation_commit_must_exclude"][1:],
    )
    for path in absent_paths:
        assert _git("ls-tree", "-r", freeze_commit, "--", path) == b""


def test_documentation_forbids_a_retry_and_quiet_retune() -> None:
    documentation = (ROOT / "docs" / "experiment-003.md").read_text(encoding="utf-8")
    normalized = " ".join(documentation.split())
    assert "Running the Experiment 002 entrypoint again is forbidden." in documentation
    assert "This is not a scientific retune." in documentation
    assert 'scientific_protocol: "002"' in documentation
    assert "The registered runner has no retry" in documentation
    assert "A second run under this identifier is never allowed." in normalized
