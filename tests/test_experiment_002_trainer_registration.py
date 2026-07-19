from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NoReturn, cast

CONFIG_PATH = Path("configs/experiment-002-trainer.json")
DOCUMENT_PATH = Path("docs/experiment-002.md")
BASE_COMMIT = "a2118109e76c069fa6be2a0b3f557fc7e79d2cbc"
TRAINER_CONFIG_SHA256 = (
    "768720d031a1f2f6d0a36466fd9b4fa095a2f1b9a6fbbc382d4af5fc51514b48"
)

BOUND_SHA256 = {
    "configs/experiment-002-numerics.json": (
        "d11fc50ba8f609551dbf88d7863a5b45cd138c6681da6c10ca4c0a9eaf8f2be9"
    ),
    "configs/experiment-002-training.json": (
        "a4a21ff66ef04bf15950853acda0897f24ed01eecf566c795a7330e69593d44a"
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
    "src/falsewake/causal_kws.py": (
        "fa5d86aa218eaad7503b6e66ed9faa18e34e917696ccb98dc5ee4f1b2267c4a6"
    ),
}


def _reject_nonfinite(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _config() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(
            CONFIG_PATH.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite,
        ),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, nested in value.items():
            yield from _strings(key)
            yield from _strings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _strings(nested)


def test_trainer_registration_is_canonical_and_has_exact_sections() -> None:
    raw = CONFIG_PATH.read_text(encoding="utf-8")
    config = _config()

    assert _sha256(CONFIG_PATH) == TRAINER_CONFIG_SHA256
    assert (
        raw
        == json.dumps(
            config,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )
    assert set(config) == {
        "bindings",
        "checkpoint_selection",
        "collation",
        "digests",
        "experiment",
        "failure_semantics",
        "metrics",
        "optimizer_execution",
        "process_isolation",
        "registration",
        "runtime_execution",
        "schema_version",
        "state_finiteness",
        "training_execution",
        "training_history",
        "validation_execution",
    }
    assert config["schema_version"] == 1
    assert config["experiment"] == "002"
    assert all("/home/omar" not in value for value in _strings(config))


def test_all_existing_trust_roots_are_exact_and_reconciled() -> None:
    config = _config()
    bindings = config["bindings"]

    for path_text, expected in BOUND_SHA256.items():
        assert _sha256(Path(path_text)) == expected

    assert bindings["base_implementation_commit"] == BASE_COMMIT
    assert bindings["configs"] == {
        "numerics": {
            "path": "configs/experiment-002-numerics.json",
            "sha256": BOUND_SHA256["configs/experiment-002-numerics.json"],
        },
        "phase_1": {
            "path": "configs/experiment-002-training.json",
            "sha256": BOUND_SHA256["configs/experiment-002-training.json"],
        },
    }
    assert bindings["model"] == {
        "parameter_tensor_count": 53,
        "parameter_value_count": 23_724,
        "path": "src/falsewake/causal_kws.py",
        "sha256": BOUND_SHA256["src/falsewake/causal_kws.py"],
    }
    assert bindings["normalization"] == {
        "artifact_byte_count": 320,
        "artifact_path": "models/experiment-002-normalization.f32",
        "artifact_sha256": BOUND_SHA256["models/experiment-002-normalization.f32"],
        "report_path": "reports/experiment-002-normalization.json",
        "report_sha256": BOUND_SHA256["reports/experiment-002-normalization.json"],
    }

    pcm_report = json.loads(
        Path("reports/experiment-002-pcm-cache.json").read_text(encoding="utf-8")
    )
    normalization_report = json.loads(
        Path("reports/experiment-002-normalization.json").read_text(encoding="utf-8")
    )
    assert bindings["pcm_cache"] == {
        "artifact_byte_count": pcm_report["artifact"]["byte_count"],
        "artifact_sha256": pcm_report["artifact"]["sha256"],
        "report_path": "reports/experiment-002-pcm-cache.json",
        "report_sha256": BOUND_SHA256["reports/experiment-002-pcm-cache.json"],
    }
    assert bindings["data"] == {
        "archive_sha256": pcm_report["inputs"]["archive_sha256"],
        "manifest_inventory_sha256": pcm_report["inputs"]["manifest_inventory_sha256"],
        "manifest_record_count": 105_829,
        "manifest_sha256": pcm_report["inputs"]["manifest_sha256"],
        "train_command_inventory_sha256": pcm_report["inputs"][
            "train_command_inventory_sha256"
        ],
        "validation_command_inventory_sha256": pcm_report["inputs"][
            "validation_command_inventory_sha256"
        ],
    }
    assert normalization_report["artifact"]["byte_count"] == 320
    assert (
        normalization_report["artifact"]["sha256"]
        == bindings["normalization"]["artifact_sha256"]
    )


def test_registration_is_honest_and_cannot_authorize_training_alone() -> None:
    registration = _config()["registration"]

    assert registration["prior_cost_only_profiling"] == {
        "learned_artifacts": 0,
        "registered_training_optimizer_steps": 0,
        "training_epoch_plans_built": 1,
        "training_features_materialized": 128,
        "validation_examples_scored": 0,
        "zero_input_forward_backward_batches": 6,
    }
    assert registration["real_training_authorized_by_this_file_alone"] is False
    assert (
        "before_any_registered_Experiment_002_training_optimizer_step"
        in (registration["chronology"])
    )
    assert (
        "configs/experiment-002-run.json"
        in registration["required_later_run_registration"]
    )
    assert (
        "exact_implementation_commit" in registration["required_later_run_registration"]
    )
    assert (
        "no_optimizer_step_using_a_registered_Experiment_002"
        in registration["run_prohibition"]
    )
    assert "untrusted_digest_overrides" in registration["run_prohibition"]
    assert registration["synthetic_optimizer_contract_probes"]["steps"] == 2
    assert (
        "two_discarded_AdamW_steps"
        in registration["synthetic_optimizer_contract_probes"]["scope"]
    )


def test_training_and_validation_collation_are_fully_determined() -> None:
    config = _config()
    collation = config["collation"]
    training = collation["training"]
    validation = collation["validation"]
    execution = config["training_execution"]

    assert "no_torch_DataLoader" in collation["common"]
    assert "torch.from_numpy" in collation["torch_conversion"]
    assert training == {
        "batch_count_per_epoch": 313,
        "batch_size": 128,
        "drop_last": False,
        "features": (
            "materialize_each_canonical_batch_as_owned_C_contiguous_float32_[B,40,98]"
        ),
        "labels": ("materialize_each_canonical_batch_as_owned_C_contiguous_int64_[B]"),
        "last_batch_size": 91,
        "slices": "[0:128],[128:256],...,[39936:40027]",
    }
    assert validation["batch_count"] == math.ceil(10_583 / 128) == 83
    assert validation["last_batch_size"] == 87
    assert validation["slices"].endswith("[10496:10583]")
    assert execution["population_count_per_epoch"] == 40_027
    assert execution["updates_per_epoch"] == 313
    assert execution["total_updates_per_seed"] == 9_390
    assert execution["seed_order"] == [20_260_719, 20_260_720, 20_260_721]
    assert "frame_97_only" in execution["supervision"]
    assert "label_smoothing_0.05" in execution["loss"]

    registered_validation = config["validation_execution"]
    assert registered_validation["population_count"] == 10_583
    assert "before_torch.manual_seed" in registered_validation["materialization"]
    assert "exactly_once" in registered_validation["materialization"]
    assert "9981_validation_commands" in registered_validation["order"]
    assert "602_running_tap_windows" in registered_validation["order"]
    assert "all_30_epoch_evaluations" in registered_validation["reuse"]


def test_validation_arithmetic_selection_and_gates_are_exact() -> None:
    config = _config()
    metrics = config["metrics"]
    selection = config["checkpoint_selection"]

    assert metrics["support"] == [
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
    ]
    assert sum(metrics["support"][:10]) == metrics["target_population"] == 3_703
    assert sum(metrics["support"]) == 10_583
    assert "12_scalar_float64_additions" in metrics["validation_cross_entropy"]
    assert (
        "10583_values_with_one_scalar_float64_addition_each"
        in metrics["validation_cross_entropy"]
    )
    assert "no_label_smoothing_or_class_weights" in metrics["validation_cross_entropy"]
    assert "Fraction(2*TP,2*TP+FP+FN)" in metrics["macro_f1"]
    assert "zero_denominator_as_Fraction(0,1)" in metrics["macro_f1"]
    assert metrics["gates"] == {
        "every_target_recall": (
            "for_each_class_c_0_through_9_require_10*confusion[c,c]>=7*class_support[c]"
        ),
        "macro_f1": "require_exact_macro_F1_fraction>=4/5",
        "silence_target_rate": "require_20*sum(confusion[11,0:10])<=602",
        "target_accuracy": (
            "require_20*sum(confusion[c,c]_for_c_0_through_9)>=17*3703"
        ),
        "unknown_target_rate": "require_5*sum(confusion[10,0:10])<=6278",
    }
    assert selection["ranking"] == [
        "maximize_exact_12_class_macro_F1_fraction",
        (
            "minimize_finite_float64_validation_cross_entropy_with_no_"
            "tolerance_or_rounding"
        ),
        "prefer_lower_zero_based_epoch_index",
        "prefer_lower_training_seed",
    ]
    assert "gates_never_filter_or_reorder_candidates" in selection["gate_timing"]
    assert "fourth_fresh_execve_process" in selection["selected_rerun"]
    assert "byte_identical_canonical_history" in selection["selected_rerun_identity"]
    assert "immediately_after_each_epoch_validation" in selection["snapshot"]
    assert "owned_detached_CPU_C_contiguous_float32_clones" in selection["snapshot"]
    assert "before_any_later_training_call" in selection["snapshot"]
    assert (
        "publishes_selected_safetensors_only_if_all_gates_pass"
        in selection["publication"]
    )
    assert "failure_report_with_no_reusable_checkpoint" in selection["publication"]


def test_adamw_update_order_and_schedule_are_unambiguous() -> None:
    optimizer = _config()["optimizer_execution"]

    assert optimizer["adamw"] == {
        "amsgrad": False,
        "betas": [0.9, 0.999],
        "capturable": False,
        "differentiable": False,
        "eps": 1e-8,
        "foreach": False,
        "fused": False,
        "learning_rate_constructor_value": 0.003,
        "maximize": False,
        "weight_decay": 1e-4,
    }
    assert "53_unique_names" in optimizer["parameter_group"]
    assert "23724_values" in optimizer["parameter_group"]
    assert "pass_only_ordered_parameters" in optimizer["parameter_group"]
    assert "never_pass_name_parameter_pairs" in optimizer["parameter_group"]
    assert "exactly_the_keys_step_exp_avg_exp_avg_sq" in optimizer["optimizer_state"]
    assert "CPU_float32_scalar_tensor_shape_[]" in optimizer["optimizer_state"]
    assert optimizer["optimizer_class"].startswith("torch.optim.AdamW")
    assert "decoupled_weight_decay_is_True" in optimizer["parameter_group_runtime"]
    assert "error_if_nonfinite=True" in optimizer["clipping"]
    assert "foreach=False" in optimizer["clipping"]
    assert optimizer["zero_grad"].startswith("set_to_none_True_exactly_once")
    assert optimizer["schedule"] == {
        "cosine": (
            "for_u_313_through_9389_inclusive_compute_float64_lr=0.00003+0.5*"
            "(0.003-0.00003)*(1+math.cos(math.pi*(u-313)/(9389-313)))"
        ),
        "warmup": (
            "for_u_0_through_312_inclusive_compute_float64_lr=0.0003+"
            "(0.003-0.0003)*u/312"
        ),
    }

    order = optimizer["update_order"]
    assert order == [
        "optimizer.zero_grad(set_to_none=True)",
        "materialize_and_finite_check_the_registered_batch",
        "model.train_and_forward_stream_all_98_frames_from_a_fresh_zero_state",
        "finite_check_all_logits_explicit_next_state_and_scalar_loss",
        "loss.backward()",
        ("require_every_parameter_gradient_is_present_dense_CPU_float32_and_finite"),
        "call_the_registered_clip_grad_norm_exactly_once",
        ("require_the_returned_preclip_norm_and_every_postclip_gradient_are_finite"),
        (
            "compute_the_registered_float64_learning_rate_and_assign_that_same_"
            "Python_float_to_the_single_optimizer_group_immediately_before_step"
        ),
        "optimizer.step_without_a_closure",
        (
            "finite_check_all_parameters_and_complete_AdamW_state_and_require_"
            "step_counter_equals_zero_based_update_plus_one"
        ),
    ]
    assert order.index("loss.backward()") < next(
        index for index, value in enumerate(order) if "clip_grad_norm" in value
    )
    lr_index = next(
        index
        for index, value in enumerate(order)
        if "assign_that_same_Python_float" in value
    )
    assert order[lr_index + 1] == "optimizer.step_without_a_closure"

    def learning_rate(update: int) -> float:
        if update <= 312:
            return 0.0003 + (0.003 - 0.0003) * update / 312
        return 0.00003 + 0.5 * (0.003 - 0.00003) * (
            1 + math.cos(math.pi * (update - 313) / (9389 - 313))
        )

    assert learning_rate(0) == 0.0003
    assert learning_rate(312) == 0.003
    assert learning_rate(313) == 0.003
    assert learning_rate(9_389) == 0.00003


def test_digest_domains_framing_and_history_are_closed() -> None:
    config = _config()
    digests = config["digests"]
    domains = {
        digests["history"]["domain"],
        digests["model_tensors"]["domain"],
        digests["training_population"]["domain"],
        digests["update_trace"]["epoch_domain"],
        digests["update_trace"]["whole_domain"],
        digests["validation_inputs"]["domain"],
        digests["validation_predictions"]["domain"],
    }

    assert len(domains) == 7
    assert all(domain.endswith("\0") for domain in domains)
    assert "applies_only_to_model_tensors" in digests["custom_binary_common_encoding"]
    assert (
        "history_and_safetensors_use_only_their_separate"
        in digests["custom_binary_common_encoding"]
    )
    assert "SHA256_over_the_exact_ASCII_domain_bytes" in digests["digest_construction"]
    assert "followed_once" in digests["digest_construction"]
    assert (
        "state_dict_keys_exactly_equal_named_parameter_keys"
        in digests["model_tensors"]["requirements"]
    )
    assert "53_tensors_and_23724_values" in digests["model_tensors"]["requirements"]
    assert (
        "raw_C_order_little_endian_float32_[40,98]"
        in digests["validation_inputs"]["framing"]
    )
    assert "raw_example.identity_bytes" in digests["training_population"]["framing"]
    assert "raw_example.identity_bytes" in digests["validation_inputs"]["framing"]
    assert "raw_example.identity_bytes" in digests["validation_predictions"]["framing"]
    assert (
        "12_raw_little_endian_float32_frame_97_logits"
        in digests["validation_predictions"]["framing"]
    )
    assert "UINT32LE_313" in digests["update_trace"]["epoch_framing"]
    assert "UINT32LE_9390" in digests["update_trace"]["whole_framing"]
    assert digests["safetensors"]["metadata"] is None
    assert "called_twice" in digests["safetensors"]["serialization"]
    assert "no_overwrite_atomic_hard_link" in digests["safetensors"]["publication"]
    assert "chmod_the_temporary_inode_to_0444" in digests["safetensors"]["publication"]
    assert (
        "never_chmod_or_unlink_the_destination_name"
        in digests["safetensors"]["publication"]
    )
    assert "byte_identical" in digests["safetensors"]["winner_rerun"]
    assert "without_a_domain_prefix" in digests["history"]["file_bytes"]
    assert "followed_once" in digests["history"]["whole_history_sha256"]

    history = config["training_history"]
    assert history["epoch_record_fields"] == [
        "zero_based_epoch",
        "first_global_update",
        "last_global_update_inclusive",
        "training_population_digest",
        "training_cross_entropy_float64_hex",
        "epoch_update_trace_digest",
        "validation_input_digest",
        "validation_confusion_matrix",
        "validation_cross_entropy_float64_hex",
        "macro_f1_exact_numerator",
        "macro_f1_exact_denominator",
        "validation_prediction_digest",
        "model_tensor_digest",
    ]
    assert set(history["epoch_record_schema"]) == set(history["epoch_record_fields"])
    assert (
        "JSON_array_of_12_rows"
        in history["epoch_record_schema"]["validation_confusion_matrix"]
    )
    assert (
        "lowercase_ASCII_64_hex"
        in history["epoch_record_schema"]["model_tensor_digest"]
    )
    assert "complete_update_trace_digest" in history["structure"]
    assert set(history["excluded_nondeterministic_fields"]) == {
        "host_load",
        "peak_rss",
        "pid",
        "timestamps",
        "wall_time",
    }
    assert (
        "lowercase_CPython_float.hex_strings_only"
        in digests["history"]["floating_fields"]
    )


def test_process_runtime_finiteness_and_failure_boundaries_are_fail_closed() -> None:
    config = _config()
    process = config["process_isolation"]
    runtime = config["runtime_execution"]
    finiteness = config["state_finiteness"]
    failures = config["failure_semantics"]

    assert process["environment_before_any_NumPy_or_Torch_import"] == {
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
        "TZ": "UTC",
    }
    assert process["budgets_per_seed_process"] == {
        "logical_cpu_threads_maximum": 2,
        "output_and_scratch_bytes_maximum": 4_294_967_296,
        "peak_rss_bytes_maximum": 8_589_934_592,
        "wall_seconds_maximum": 21_600,
    }
    assert "two_lowest_sorted_logical_CPU_IDs" in process["cpu_affinity"]
    assert "exact_same_two_ID_set" in process["cpu_affinity"]
    assert "exactly_one_seed_or_rerun_child" in process["topology"]
    assert "four_independent_fresh_execve_processes" in process["creation"]
    assert "no_fork" in process["creation"]
    assert "no_dev_clean_audio" in process["access_boundary"]
    assert "no_test_clean_audio" in process["access_boundary"]
    assert "parsed_once_by_the_registered_corpus_verifier" in process["test_boundary"]
    assert "must_not_be_retained_returned" in process["test_boundary"]
    assert (
        "wait4_per_child_rusage.ru_maxrss" in process["resource_accounting"]["peak_rss"]
    )
    assert "time.monotonic_ns" in process["resource_accounting"]["wall"]
    assert "intervals_no_greater_than_100ms" in process["resource_enforcement"]
    assert "git_status_porcelain_v1_with_untracked_files_all" in process["worktree"]
    assert "only_bound_tracked_nonsymlink_files" in process["worktree"]
    assert "exact_argv_sanitized_environment" in process["source_verification"]

    assert runtime["libraries"] == {
        "numpy": "2.5.1",
        "python": "3.12.3",
        "safetensors": "0.8.0",
        "torch": "2.13.0+cpu",
    }
    assert "torch.manual_seed(seed)_exactly_once" in runtime["model_creation"]
    assert "immediately_before_CausalKWS()" in runtime["model_creation"]
    assert runtime["torch_setup_before_manual_seed"] == {
        "cuda_available_required": False,
        "default_device": "cpu",
        "default_dtype": "float32",
        "deterministic_algorithms": (
            "torch.use_deterministic_algorithms(True,warn_only=False)"
        ),
        "float32_matmul_precision": "highest",
        "flush_denormal": False,
        "inter_op_threads": 1,
        "intra_op_threads": 2,
        "mkldnn_enabled": False,
        "nnpack_enabled": False,
    }
    assert len(finiteness["every_training_batch"]) == 9
    assert len(finiteness["every_validation_batch"]) == 3
    assert (
        "unreturned_internal_transient_activations_are_not_hooked"
        in finiteness["scope"]
    )
    assert failures["nonfinite_policy"].startswith("abort_immediately")
    assert (
        "no_batch_epoch_seed_candidate_or_rerun_retry" in failures["exception_policy"]
    )
    assert "without_fallback" in failures["rerun_failure"]


def test_documentation_records_the_pre_result_barrier() -> None:
    document = DOCUMENT_PATH.read_text(encoding="utf-8")

    assert "## Frozen trainer execution addendum" in document
    assert "trainer execution registration" in document
    assert "zero registered training" in document
    assert "optimizer steps" in document
    assert "zero validation examples" in document
    assert "zero learned" in document
    assert "artifacts" in document
    assert "exactly two synthetic AdamW steps" in document
    assert "does not authorize training by itself" in document
    assert "second source-bound run registration" in document
    assert "real optimizer updates" in document
    assert "metrics remain prohibited" in document
