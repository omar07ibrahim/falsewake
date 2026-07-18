from __future__ import annotations

import hashlib
import json
import math
import struct
import tomllib
from pathlib import Path
from typing import Any, cast

CONFIG_PATH = Path("configs/experiment-002-training.json")
DOC_PATH = Path("docs/experiment-002.md")


def _config() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(CONFIG_PATH.read_text(encoding="utf-8")))


def _experiment_001_scorer_digest(paths: list[str]) -> str:
    digest = hashlib.sha256(b"falsewake-exp001-scorer-source-v1\0")
    for path_text in sorted(paths, key=lambda value: value.encode("utf-8")):
        path_bytes = path_text.encode("utf-8")
        contents = Path(path_text).read_bytes()
        digest.update(struct.pack("<I", len(path_bytes)))
        digest.update(path_bytes)
        digest.update(struct.pack("<Q", len(contents)))
        digest.update(contents)
    return digest.hexdigest()


def test_neural_toolchain_extras_are_separated_and_bounded() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]

    assert extras["export"] == ["onnx>=1.22,<2", "onnxscript>=0.7.1,<0.8"]
    assert extras["runtime"] == ["onnxruntime>=1.27,<2"]
    assert extras["train"] == ["safetensors>=0.8,<1", "torch>=2.13,<3"]

    runtime_lock = _config()["runtime_lock"]
    assert runtime_lock == {
        "numpy": "2.5.1",
        "onnx": "1.22.0",
        "onnxruntime": "1.27.0",
        "onnxscript": "0.7.1",
        "python": "3.12.3",
        "safetensors": "0.8.0",
        "torch": "2.13.0+cpu",
    }

    neural_packages = (
        "onnx",
        "onnxruntime",
        "onnxscript",
        "safetensors",
        "torch",
    )
    base_dependencies = pyproject["project"]["dependencies"]
    assert not any(
        dependency.startswith(neural_packages) for dependency in base_dependencies
    )


def test_neural_toolchain_instructions_are_fail_closed() -> None:
    documentation = DOC_PATH.read_text(encoding="utf-8")

    assert "https://download.pytorch.org/whl/cpu" in documentation
    assert '"torch==2.13.0+cpu"' in documentation
    assert '["runtime_lock"]' in documentation
    assert '"python": platform.python_version()' in documentation
    assert "importlib.metadata.version(package)" in documentation
    assert "torch.version.cuda is not None" in documentation
    assert "torch.cuda.is_available()" in documentation
    assert 'raise SystemExit("runtime lock mismatch:' in documentation


def test_training_preregistration_is_canonical_and_has_exact_sections() -> None:
    raw = CONFIG_PATH.read_text(encoding="utf-8")
    config = json.loads(raw)

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
        "augmentation",
        "benchmark",
        "class_order",
        "data",
        "deployment",
        "experiment",
        "frontend",
        "identity",
        "model",
        "normalization",
        "onnx",
        "optimizer",
        "phase_2_replay_registration",
        "phase_order",
        "reproducibility",
        "resource_budgets",
        "runtime_lock",
        "schema_version",
        "seeds",
        "streaming_evaluation",
        "training",
        "validation",
    }
    assert config["schema_version"] == 1
    assert config["identity"]["phase"].startswith(
        "phase_1_training_and_deployment_preregistration"
    )
    assert (
        "committed_before_the_experiment_002_model_implementation_commit"
        in config["identity"]["phase"]
    )


def test_predecessor_bytes_and_seven_file_digest_are_frozen() -> None:
    identity = _config()["identity"]
    frozen = identity["frozen_predecessor"]

    assert set(frozen) == {
        "experiment_000_portable_model_sha256",
        "experiment_001_config_sha256",
        "experiment_001_dev_replay_report_sha256",
        "experiment_001_scorer_source_digest",
        "experiment_001_scorer_source_files",
        "experiment_001_scorer_source_sha256",
        "experiment_001_selection_sha256",
    }
    assert frozen["experiment_001_config_sha256"] == (
        "7e27ba554ef357a2de4a929088a99efc20b9ee0285cb0c242a05474cc944b217"
    )
    config_001_digest = hashlib.sha256(
        Path("configs/experiment-001.json").read_bytes()
    ).hexdigest()
    assert config_001_digest == frozen["experiment_001_config_sha256"]
    assert frozen["experiment_000_portable_model_sha256"] == (
        "d5ac3579f539288c6ed7769894dd30ad2272480956b2f9b24b7e730baaeb0cd3"
    )
    assert (
        hashlib.sha256(
            Path("models/experiment-000-linear.json").read_bytes()
        ).hexdigest()
        == (frozen["experiment_000_portable_model_sha256"])
    )
    assert frozen["experiment_001_dev_replay_report_sha256"] == (
        "b8e30e26498af2600ece01f4cceb436e10a546b8390f039ca8a23cda45f00f2d"
    )
    assert (
        hashlib.sha256(
            Path("reports/experiment-001-dev-replay.json").read_bytes()
        ).hexdigest()
        == frozen["experiment_001_dev_replay_report_sha256"]
    )
    assert frozen["experiment_001_selection_sha256"] == (
        "1d44ae6ff06a5fab1567d0342299e293fe001b8c91f9972cfa8e79a2dabf2318"
    )
    assert (
        hashlib.sha256(
            Path("reports/experiment-001-selection.json").read_bytes()
        ).hexdigest()
        == frozen["experiment_001_selection_sha256"]
    )
    assert frozen["experiment_001_scorer_source_sha256"] == (
        "93bd11f50e11ed481f3f2306ef377a9dd106dd908853f7f9478b86bf5f62aeb9"
    )
    assert (
        _experiment_001_scorer_digest(frozen["experiment_001_scorer_source_files"])
        == frozen["experiment_001_scorer_source_sha256"]
    )


def test_registered_populations_balance_and_window_math_are_exact() -> None:
    data = _config()["data"]
    train = data["train_epoch"]
    validation = data["validation"]

    assert data["archive"] == {
        "bytes": 2_428_923_189,
        "filename": "speech_commands_v0.02.tar.gz",
        "sha256": ("af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58"),
    }
    assert data["manifest"] == {
        "inventory_sha256": (
            "c9596927b6de1aa9bb8174bea25f613ebb7bab9e671304c525961b66e0812c5c"
        ),
        "record_count": 105_829,
        "schema_version": 2,
        "sha256": ("d28e6993101bd6bc452033bcb7355b25e3b84ab5c51a1458cc097dc93c60f78b"),
        "source_lists": {
            "testing_sha256": (
                "2d17c6b3faf63be43eda93cfeb0c747cfd79b7b236282039dbac65a2cb5f1df5"
            ),
            "validation_sha256": (
                "5747407275538b4056e823982f0db1fc993776ab532048196a19be701bdc87d2"
            ),
        },
        "split_counts": {"test": 11_005, "train": 84_843, "validation": 9_981},
    }
    audit = json.loads(
        Path("reports/speech-commands-audit.json").read_text(encoding="utf-8")
    )
    assert audit["jsonl_sha256"] == data["manifest"]["sha256"]
    assert audit["inventory_sha256"] == data["manifest"]["inventory_sha256"]
    assert audit["source_files"] == {
        "testing_list_sha256": data["manifest"]["source_lists"]["testing_sha256"],
        "validation_list_sha256": data["manifest"]["source_lists"]["validation_sha256"],
    }
    assert train["targets"]["count"] == 30_769
    assert (
        sum(train["targets"]["support_by_class"].values()) == train["targets"]["count"]
    )
    assert train["unknown"]["count"] == 2 * 3_086 == 6_172
    assert train["unknown"]["source_clip_count"] == 84_843 - 30_769
    assert train["unknown"]["count"] == 22 * 247 + 3 * 246
    assert train["unknown"]["source_words"] == [
        "backward",
        "bed",
        "bird",
        "cat",
        "dog",
        "eight",
        "five",
        "follow",
        "forward",
        "four",
        "happy",
        "house",
        "learn",
        "marvin",
        "nine",
        "one",
        "seven",
        "sheila",
        "six",
        "three",
        "tree",
        "two",
        "visual",
        "wow",
        "zero",
    ]
    background_records = train["silence"]["background_records"]
    assert background_records == [
        {
            "path": "_background_noise_/doing_the_dishes.wav",
            "sample_count": 1_522_930,
            "sha256": (
                "099eafcd7c4c266612012b5622b97157042e5288acaf4cbf3dc475814bfcdaa8"
            ),
        },
        {
            "path": "_background_noise_/dude_miaowing.wav",
            "sample_count": 988_891,
            "sha256": (
                "1acd62f115d4c3f9daca9c5ec0c2e0c3e174a2a08be30200003a51de52b288ff"
            ),
        },
        {
            "path": "_background_noise_/exercise_bike.wav",
            "sample_count": 980_062,
            "sha256": (
                "e453813ed45b2f9f81d5600ec3ac2a5e22d3f2c947fe6f0a3ff1f3d6018c014d"
            ),
        },
        {
            "path": "_background_noise_/pink_noise.wav",
            "sample_count": 960_000,
            "sha256": (
                "b6e038c83fb342e39267d4fe69663f76ef0c1121ff60232e7b79171c00318cd6"
            ),
        },
    ]
    audited_train_backgrounds = [
        {key: record[key] for key in ("path", "sample_count", "sha256")}
        for record in audit["background"]
        if record["split"] == "train"
    ]
    assert audited_train_backgrounds == background_records
    assert (
        train["targets"]["count"]
        + train["unknown"]["count"]
        + train["silence"]["count"]
        == train["example_count"]
        == 40_027
    )
    assert validation["targets"]["count"] == 3_703
    assert (
        sum(validation["targets"]["support_by_class"].values())
        == validation["targets"]["count"]
    )
    silence = validation["silence"]
    assert silence["count"] == 602
    assert silence["sha256"] == (
        "c199c5fd61f5bf9fd57f1c346c8a51c0c035d63d176348dd9f0f7d671d7eb8eb"
    )
    audited_validation_background = next(
        record for record in audit["background"] if record["split"] == "validation"
    )
    assert silence["background_path"] == audited_validation_background["path"]
    assert silence["source_samples"] == audited_validation_background["sample_count"]
    assert silence["sha256"] == audited_validation_background["sha256"]
    assert (
        1
        + (silence["source_samples"] - silence["window_samples"])
        // silence["hop_samples"]
        == silence["count"]
    )
    assert (
        validation["targets"]["count"]
        + validation["unknown"]["count"]
        + silence["count"]
        == validation["example_count"]
        == 10_583
    )
    assert "forbidden" in data["test"]["audio_access"]


def test_causal_model_geometry_parameters_and_state_size_are_consistent() -> None:
    config = _config()
    frontend = config["frontend"]
    model = config["model"]
    dilations = model["blocks"]["dilations"]

    assert config["class_order"] == [
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
    ]
    assert model["blocks"]["count"] == 8
    assert dilations == [1, 2, 4, 8, 1, 2, 4, 8]
    assert model["initialization"] == {
        "convolutions": "kaiming_normal_fan_out_nonlinearity_relu",
        "layer_norm_bias": 0.0,
        "layer_norm_weight": 1.0,
        "linear_bias": 0.0,
        "linear_weight": "xavier_uniform",
    }
    assert (
        frontend["frame_count"]
        == 1
        + (frontend["clip_samples"] - frontend["window_samples"])
        // frontend["hop_samples"]
        == 98
    )
    encoder_receptive_frames = 1 + 2 * sum(dilations)
    pooled_receptive_frames = encoder_receptive_frames + model["pool"]["history_frames"]
    receptive_samples = (
        frontend["window_samples"]
        + (pooled_receptive_frames - 1) * frontend["hop_samples"]
    )
    assert model["receptive_field"] == {
        "encoder_frames": encoder_receptive_frames,
        "first_scored_frame_zero_based": pooled_receptive_frames - 1,
        "input_frames": pooled_receptive_frames,
        "samples": receptive_samples,
        "seconds": receptive_samples / frontend["sample_rate"],
        "statement": "995_ms_not_one_second",
    }
    stem_parameters = 40 * 48 + 2 * 48
    block_parameters = 48 * 3 + 2 * 48 + 48 * 48 + 2 * 48
    output_parameters = 48 * 12 + 12
    assert model["parameter_count"] == 23_724
    assert (
        stem_parameters + 8 * block_parameters + output_parameters
        == model["parameter_count"]
    )
    state_float_count = 48 * (sum(2 * dilation for dilation in dilations) + 37)
    assert model["state"]["bytes_per_batch_element_excluding_counter"] == 18_624
    assert (
        state_float_count * 4
        == model["state"]["bytes_per_batch_element_excluding_counter"]
    )
    assert model["input"] == {
        "dtype": "float32",
        "layout": "NCT",
        "shape": "[B,40,T]",
    }
    assert model["dense_output"].startswith("logits_float32_shape_[B,T,12]")
    assert model["state"]["block_history_shapes_in_order"] == [
        "[B,48,2]",
        "[B,48,4]",
        "[B,48,8]",
        "[B,48,16]",
        "[B,48,2]",
        "[B,48,4]",
        "[B,48,8]",
        "[B,48,16]",
    ]
    assert model["compute"] == {
        "dense_98_frame_call_MACs": 2_163_840,
        "dense_steady_10_frame_call_MACs": 220_800,
        "dense_steady_MACs_per_second_at_100_frames": 2_208_000,
        "formula": "21504_times_F_plus_576_times_S",
        "scope": (
            "neural_Conv_and_Linear_multiply_accumulates_only;excludes_"
            "LayerNorm_ReLU_residual_pool_softmax_and_frontend"
        ),
        "terms": (
            "stem_1920_times_F;8_blocks_each_(144_plus_2304)_times_F;head_"
            "576_times_S;dense_export_has_S_equal_F"
        ),
    }
    abi = model["streaming_abi"]
    assert abi["inputs_in_order"] == [
        "mel_frames:float32[B,40,T]",
        "dw_state_0:float32[B,48,2]",
        "dw_state_1:float32[B,48,4]",
        "dw_state_2:float32[B,48,8]",
        "dw_state_3:float32[B,48,16]",
        "dw_state_4:float32[B,48,2]",
        "dw_state_5:float32[B,48,4]",
        "dw_state_6:float32[B,48,8]",
        "dw_state_7:float32[B,48,16]",
        "pool_state:float32[B,48,37]",
        "frames_seen:int64[B]",
    ]
    assert abi["outputs_in_order"][0] == "logits:float32[B,T,12]"
    assert abi["outputs_in_order"][1:9] == [
        value.replace("dw_state", "next_dw_state")
        for value in abi["inputs_in_order"][1:9]
    ]
    assert abi["outputs_in_order"][-2:] == [
        "next_pool_state:float32[B,48,37]",
        "next_frames_seen:int64[B]",
    ]
    assert "only_batch_B" in abi["dynamic_axes"]
    assert "positive_zero" in abi["reset"]


def test_training_schedule_runtime_and_seed_selection_are_frozen() -> None:
    config = _config()
    training = config["training"]
    schedule = config["optimizer"]["schedule"]

    assert training["batch_size"] == 128
    assert training["updates_per_epoch"] == 313
    assert math.ceil(40_027 / training["batch_size"]) == training["updates_per_epoch"]
    assert training["total_updates"] == 9_390
    assert (
        training["epochs"] * training["updates_per_epoch"] == training["total_updates"]
    )
    assert training["amp"] is False
    assert training["cpu_only"] is True
    assert training["drop_last"] is False
    assert training["early_stopping"] is False
    assert training["last_batch_size"] == 91
    assert config["optimizer"]["adamw"] == {
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "foreach": False,
        "fused": False,
        "learning_rate": 0.003,
        "weight_decay": 1e-4,
    }
    assert config["optimizer"]["gradient_l2_clip"] == 5.0
    assert config["optimizer"]["parameter_groups"] == (
        "one_group_containing_every_trainable_parameter_including_LayerNorm_"
        "weights_LayerNorm_biases_and_classifier_bias"
    )
    assert training["loss"] == {
        "class_weights": "none",
        "label_smoothing": 0.05,
        "name": "cross_entropy",
        "reduction": "mean_over_current_batch",
    }
    assert training["forward"] == (
        "one_[B,40,98]_dense_forward_from_all_zero_block_pool_states_and_frames_"
        "seen_zero_per_independent_clip;all_98_logits_are_computed_so_Torch_"
        "dropout_consumption_is_fixed"
    )
    assert training["supervised_logit"] == (
        "logits[:,97,:]_only;startup_frames_0_through_96_are_excluded_from_the_loss"
    )
    assert training["checkpoint_selection"] == (
        "within_each_seed_maximize_12_class_validation_macro_F1_then_minimize_"
        "validation_cross_entropy_then_choose_earlier_epoch;across_seed_winners_"
        "apply_the_same_metric_order_then_earlier_epoch_then_lower_seed"
    )
    assert training["model_rng"] == (
        "torch_manual_seed(training_seed)_once_before_parameter_initialization;the_"
        "same_seeded_Torch_stream_drives_training_dropout_only_and_never_data_"
        "sampling_order_or_augmentation"
    )
    assert training["determinism"] == {
        "blas_threads": 1,
        "dataloader_workers": 0,
        "dtype": "float32",
        "inter_op_threads": 1,
        "intra_op_threads": 2,
        "mkldnn_enabled": False,
        "torch_deterministic_algorithms": True,
    }
    assert "u_0_through_312_inclusive" in schedule["warmup"]
    assert "u_313_through_9389_inclusive" in schedule["cosine"]
    assert config["seeds"] == {
        "benchmark": 20_260_723,
        "bootstrap": 20_260_724,
        "protocol": 20_260_718,
        "quantization": 20_260_722,
        "training": [20_260_719, 20_260_720, 20_260_721],
    }
    assert config["runtime_lock"] == {
        "numpy": "2.5.1",
        "onnx": "1.22.0",
        "onnxruntime": "1.27.0",
        "onnxscript": "0.7.1",
        "python": "3.12.3",
        "safetensors": "0.8.0",
        "torch": "2.13.0+cpu",
    }
    assert "identical" in config["reproducibility"]["selected_seed_rerun"]


def test_stateless_augmentation_and_validation_gates_are_exact() -> None:
    config = _config()
    augmentation = config["augmentation"]
    command = augmentation["command"]

    assert augmentation["global_data_rng"] == "forbidden"
    assert augmentation["draws"]["stateless_key"] == (
        "HMAC_SHA256_key_UINT64LE(seed)_message_ASCII_falsewake-exp002-v1_NUL_"
        "UINT32LE(epoch)_UINT32LE(identity_UTF8_byte_count)_identity_UTF8_"
        "UINT32LE(domain_ASCII_byte_count)_domain_ASCII_UINT64LE(counter)"
    )
    assert augmentation["draws"]["domains"] == {
        "benchmark_feature": "benchmark-feature",
        "benchmark_pcm": "benchmark-pcm",
        "calibration_rank_prefix": "calibration-rank-",
        "command_gain_db": "command-gain-db",
        "command_mel_start": "command-mel-start",
        "command_mel_width": "command-mel-width",
        "command_noise_apply": "command-noise-apply",
        "command_noise_snr_db": "command-noise-snr-db",
        "command_noise_window": "command-noise-window",
        "command_shift": "command-shift",
        "command_time_start": "command-time-start",
        "command_time_width": "command-time-width",
        "silence_gain_db": "silence-gain-db",
        "silence_mel_start": "silence-mel-start",
        "silence_mel_width": "silence-mel-width",
        "silence_time_start": "silence-time-start",
        "silence_time_width": "silence-time-width",
        "silence_window_rank": "silence-window-rank",
        "train_order": "train-order",
        "unknown_clip_rank": "unknown-clip-rank",
        "unknown_word_rank": "unknown-word-rank",
    }
    assert augmentation["draws"]["identity_encodings"] == {
        "command": (
            "UTF8(command)_NUL_manifest_path_UTF8_exactly_as_manifest_without_"
            "Unicode_normalization"
        ),
        "silence_or_noise_window": (
            "UTF8(window)_NUL_background_path_UTF8_exactly_as_audit_NUL_ASCII_"
            "canonical_unsigned_decimal_start_sample"
        ),
        "source_word": (
            "UTF8(word)_NUL_source_word_UTF8_exactly_as_manifest_without_Unicode_"
            "normalization"
        ),
    }
    assert augmentation["draws"]["integer"] == (
        "for_positive_bound_n_read_digest_word_x;reject_x_greater_than_or_equal_"
        "to_2^64_minus_(2^64_mod_n)_and_increment_counter;otherwise_return_x_mod_n"
    )
    assert augmentation["draws"]["rank"] == (
        "compare_all_32_HMAC_digest_bytes_as_unsigned_byte_strings_"
        "lexicographically_then_identity_UTF8_bytes"
    )
    assert augmentation["draws"]["real"] == (
        "u=float64(digest_word_right_shift_11)_divided_by_2^53;raw=float64(minimum)"
        "+float64(maximum_exclusive-minus-minimum)*u;return_min(raw,numpy_nextafter("
        "float64(maximum_exclusive),negative_infinity))_as_float64_so_the_maximum_"
        "remains_exclusive"
    )
    assert "right_zero_pad_to_exactly_16000" in command["base_waveform"]
    assert command["shift_samples"] == {
        "draw": "unbiased_integer_with_bound_3201_then_subtract_1600",
        "maximum_inclusive": 1_600,
        "minimum_inclusive": -1_600,
        "padding": "zero_fill_without_wrap",
        "positive_direction": (
            "positive_s_delays_audio_to_the_right_so_output[i]=input[i-s]_when_0_"
            "lte_i-s_lt_16000_else_positive_zero;negative_s_advances_left_by_minus_s"
        ),
    }
    assert command["gain_db"]["minimum_inclusive"] == -6.0
    assert command["gain_db"]["maximum_exclusive"] == 6.0
    assert command["noise_mix"]["probability"] == 0.8
    assert command["noise_mix"]["snr_db_minimum_inclusive"] == 0.0
    assert command["noise_mix"]["snr_db_maximum_exclusive"] == 20.0
    assert "full_duration_weighted_universe" in command["noise_mix"]["window_selection"]
    assert augmentation["feature_masks"]["fill_after_global_normalization"] == 0.0
    assert "0_through_10_inclusive" in augmentation["feature_masks"]["time_mask_width"]
    assert "0_through_4_inclusive" in augmentation["feature_masks"]["mel_mask_width"]
    assert config["validation"]["gates_before_any_development_replay"] == {
        "finite_all_metrics_logits_parameters_and_states": True,
        "maximum_silence_argmax_target_rate": 0.05,
        "maximum_unknown_argmax_target_rate": 0.2,
        "minimum_12_class_macro_f1": 0.8,
        "minimum_every_target_recall": 0.7,
        "minimum_target_accuracy": 0.85,
    }
    assert config["validation"]["order"] == (
        "all_9981_validation_command_rows_in_canonical_manifest_path_UTF8_byte_"
        "order_then_602_running_tap_silence_windows_in_ascending_start_sample_order"
    )
    assert config["validation"]["evaluation"].startswith(
        "after_each_completed_epoch_1_through_30_switch_to_eval"
    )


def test_export_quantization_and_benchmark_have_fail_closed_gates() -> None:
    config = _config()
    fp32 = config["onnx"]["fp32"]
    int8 = config["onnx"]["int8"]
    benchmark = config["benchmark"]

    assert fp32["opset"] == 18
    assert fp32["frames_seen_agreement"] == "exact_int64"
    assert "starting_with_Random" in fp32["forbidden_node_contract"]
    assert fp32["max_logit_absolute_delta"] == 1e-5
    assert fp32["max_state_absolute_delta"] == 1e-5
    assert fp32["size_limit_bytes"] == 256 * 1024
    assert set(fp32["partition_checks"]) == {
        "append_future_prefix_invariance_exact_prefix_logits_and_states_before_the_append",
        "full_sequence",
        "one_frame_chunks",
        "irregular_fixed_chunks_7_1_23_4_19_44_repeated_until_consumed",
    }
    assert {"BatchNormalization", "Dropout"}.issubset(fp32["forbidden_nodes"])
    assert int8["calibration"]["example_count"] == 1_536
    assert (
        int8["calibration"]["count_per_class"] * 12
        == int8["calibration"]["example_count"]
    )
    assert int8["calibration"]["source"] == "train_only"
    assert int8["calibration"]["candidate_universes"] == {
        "silence": (
            "every_unique_16000_sample_window_from_the_four_registered_training_"
            "backgrounds_at_each_integer_start_0_through_sample_count_minus_16000_"
            "inclusive"
        ),
        "target": (
            "for_each_target_class_every_training_command_clip_with_that_exact_label"
        ),
        "unknown": "all_54074_training_unknown_command_clips",
    }
    assert "no_shift_gain_noise_or_masks" in int8["calibration"]["preprocessing"]
    assert int8["calibration"]["reader_order"] == (
        "class_order_then_HMAC_rank_then_exact_identity_bytes"
    )
    assert "calibration-rank-<class_order_label>" in int8["calibration"]["sampling"]
    assert int8["comparison"] == {
        "argmax_agreement": (
            "fraction_of_the_10583_examples_where_INT8_and_FP32_ONNX_frame_97_"
            "argmax_are_identical_with_first_class_order_index_winning_exact_ties"
        ),
        "metric_deltas": (
            "target_accuracy_drop=FP32_minus_INT8;macro_F1_drop=FP32_minus_INT8;"
            "unknown_and_silence_target_rate_increase=INT8_minus_FP32"
        ),
        "population": (
            "the_exact_10583_example_validation_population_in_registered_"
            "validation_order_with_independent_zero_state_per_example"
        ),
        "probability_delta": (
            "stable_float64_softmax_separately_over_each_candidates_float32_frame_"
            "97_logits_then_maximum_absolute_difference_over_all_10583_times_12_"
            "example_class_probabilities"
        ),
        "reference": (
            "the_committed_eligible_FP32_ONNX_artifact_executed_with_"
            "CPUExecutionProvider_only"
        ),
    }
    assert int8["recipe_count"] == 1
    assert "Conv_Gemm_and_constant_B_MatMul_only" in int8["recipe"]
    assert int8["eligibility"] == {
        "all_metrics_logits_and_states_finite": True,
        "argmax_agreement_minimum": 0.99,
        "macro_f1_maximum_drop": 0.01,
        "maximum_probability_absolute_delta": 0.05,
        "maximum_silence_argmax_target_rate": 0.05,
        "maximum_unknown_argmax_target_rate": 0.2,
        "minimum_12_class_macro_f1": 0.8,
        "minimum_every_target_recall": 0.7,
        "minimum_target_accuracy": 0.85,
        "silence_target_rate_maximum_increase": 0.005,
        "size_limit_bytes": 192 * 1024,
        "target_accuracy_maximum_drop": 0.01,
        "unknown_target_rate_maximum_increase": 0.01,
    }
    recipe = int8["quantize_static_arguments"]
    assert recipe["quant_format"] == "QuantFormat.QDQ"
    assert recipe["activation_type"] == recipe["weight_type"] == "QuantType.QInt8"
    assert recipe["calibrate_method"] == "CalibrationMethod.MinMax"
    assert recipe["op_types_to_quantize"] == ["Conv", "Gemm", "MatMul"]
    assert recipe["nodes_to_quantize"] == []
    assert recipe["nodes_to_exclude"] == []
    assert recipe["per_channel"] is True
    assert recipe["reduce_range"] is False
    assert recipe["use_external_data_format"] is False
    assert recipe["calibration_providers"] == ["CPUExecutionProvider"]
    assert recipe["calibration_cache_path"] is None
    assert recipe["extra_options"] == {
        "ActivationSymmetric": False,
        "AddQDQPairToWeight": False,
        "CalibMovingAverage": False,
        "CalibStridedMinMax": None,
        "CalibTensorRangeSymmetric": False,
        "DedicatedQDQPair": False,
        "EnableSubgraph": False,
        "ForceQuantizeNoInputCheck": False,
        "MatMulConstBOnly": True,
        "MinimumRealRange": None,
        "OpTypesToExcludeOutputQuantization": [],
        "QDQDisableWeightAdjustForInt32Bias": False,
        "QDQKeepRemovableActivations": False,
        "QDQOpTypePerChannelSupportToAxis": {},
        "SmoothQuant": False,
        "UseQDQContribOps": False,
        "WeightSymmetric": True,
    }
    assert set(int8["stateful_parity"]["checks"]) == {
        "append_future_prefix_invariance",
        "full_sequence",
        "independent_state_reset",
        "irregular_fixed_chunks_7_1_23_4_19_44_repeated_until_consumed",
        "one_frame_chunks",
    }
    assert int8["stateful_parity"]["frames_seen"] == "exact_int64"
    assert int8["stateful_parity"]["maximum_partition_logit_absolute_delta"] == (1e-5)
    assert int8["stateful_parity"]["maximum_partition_state_absolute_delta"] == (1e-5)
    assert (
        "absolute_validation_relative_agreement_size_stateful_parity"
        in config["deployment"]["candidate_choice"]
    )
    assert "not_slower_than_FP32" in config["deployment"]["candidate_choice"]
    assert config["deployment"]["development_score_before_replay_registration"] == (
        "forbidden_for_both_FP32_and_INT8"
    )
    assert benchmark["warmup_iterations"] == 200
    assert benchmark["measured_iterations"] == 5_000
    assert benchmark["cold_input_frames"] == 98
    assert benchmark["cpu"] == "AMD_EPYC_7R13"
    assert benchmark["execution_provider"] == ("onnxruntime_CPUExecutionProvider_only")
    assert benchmark["inter_op_threads"] == benchmark["intra_op_threads"] == 1
    assert benchmark["mode"] == "ORT_SEQUENTIAL"
    assert benchmark["steady_input_frames"] == 10
    assert benchmark["measures"] == [
        "p50_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
        "state_cold_98_frame_latency_ms",
        "model_bytes",
        "state_bytes",
        "peak_RSS_bytes",
    ]
    assert benchmark["selected_model_steady_p95_limit_ms"] == 2.0
    assert benchmark["selected_frontend_plus_model_p95_limit_ms"] == 10.0
    assert benchmark["peak_rss_limit_bytes"] == 256 * 1024 * 1024
    assert benchmark["feature_tensor"]["shape"] == "[1,40,108]"
    assert benchmark["feature_tensor"]["dtype"] == "little_endian_float32"
    assert "seed_20260723" in benchmark["feature_tensor"]["construction"]
    assert benchmark["frontend_model_stream"]["sample_count"] == (
        (1 + 200 + 5_000) * 1_600
    )
    assert benchmark["frontend_model_stream"]["prime_samples"] == 1_600
    assert benchmark["frontend_model_stream"]["warmup_chunks"] == 200
    assert benchmark["frontend_model_stream"]["measured_chunks"] == 5_000
    assert benchmark["percentile_method"] == "numpy_percentile_method_higher"
    assert benchmark["session_options"] == {
        "enable_cpu_mem_arena": True,
        "enable_mem_pattern": True,
        "execution_mode": "ORT_SEQUENTIAL",
        "graph_optimization_level": "ORT_ENABLE_ALL",
        "inter_op_num_threads": 1,
        "intra_op_num_threads": 1,
        "providers": ["CPUExecutionProvider"],
    }
    assert "state_cold_does_not_mean_process_cold" in benchmark["state_cold"]
    assert "nonzero_output_state" in benchmark["steady_state"]
    assert set(benchmark) == {
        "cold_input_frames",
        "cpu",
        "execution_provider",
        "feature_tensor",
        "frontend_model_stream",
        "input_parity",
        "inter_op_threads",
        "intra_op_threads",
        "measured_iterations",
        "measures",
        "mode",
        "peak_rss_limit_bytes",
        "percentile_method",
        "process",
        "selected_frontend_plus_model_p95_limit_ms",
        "selected_model_steady_p95_limit_ms",
        "session_options",
        "state_cold",
        "steady_input_frames",
        "steady_state",
        "timing",
        "warmup_iterations",
    }
    assert benchmark["feature_tensor"]["construction"] == (
        "for_shape_[1,40,108]_in_C_order_use_HMAC_seed_20260723_epoch_0_identity_"
        "ASCII_benchmark-feature-tensor_domain_ASCII_benchmark-feature_counter_"
        "equal_to_scalar_index;unbiased_integer_0_through_65535_minus_32768_then_"
        "float32_divide_by_32768"
    )
    assert benchmark["frontend_model_stream"]["construction"] == (
        "generate_8321600_samples_with_HMAC_seed_20260723_epoch_0_identity_ASCII_"
        "benchmark-pcm-stream_domain_ASCII_benchmark-pcm_counter_equal_to_sample_"
        "index;unbiased_integer_0_through_65535_minus_32768_then_float32_divide_by_"
        "32768"
    )
    assert benchmark["frontend_model_stream"]["streaming"] == (
        "prime_frontend_and_model_once_outside_timing_then_carry_both_states;each_"
        "following_1600_sample_chunk_emits_exactly_10_frames;PCM_generation_is_"
        "excluded_but_frontend_NCT_copy_ORT_call_and_output_materialization_are_"
        "included"
    )
    assert benchmark["input_parity"] == (
        "FP32_and_INT8_use_byte_identical_feature_tensor_PCM_stream_initial_states_"
        "and_iteration_order"
    )
    assert benchmark["process"] == (
        "one_fresh_process_per_candidate_with_session_construction_before_timing_"
        "and_process_peak_RSS_including_Python_ONNX_Runtime_model_and_working_state"
    )
    assert benchmark["state_cold"] == (
        "use_feature_tensor_frames_0_through_97_with_positive_zero_float_histories_"
        "and_int64_zero_counter_for_every_call;session_is_warm_so_state_cold_does_"
        "not_mean_process_cold"
    )
    assert benchmark["steady_state"] == (
        "outside_timing_run_feature_tensor_frames_0_through_97_once_from_zero_state_"
        "then_reuse_that_nonzero_output_state_with_frames_98_through_107_for_every_"
        "10_frame_measured_call;ORT_inputs_are_not_mutated"
    )
    assert benchmark["timing"] == (
        "disable_Python_GC_during_loops;perf_counter_ns_immediately_around_each_"
        "call;all_inputs_and_microbenchmark_states_precomputed_outside_timing;"
        "materialize_every_output;session_creation_and_input_generation_excluded"
    )
    assert config["resource_budgets"] == {
        "cache_build": {
            "logical_CPU_threads_maximum": 2,
            "output_bytes_maximum": 4 * 1024**3,
            "peak_RSS_bytes_maximum": 4 * 1024**3,
            "scratch_disk_bytes_maximum": 8 * 1024**3,
            "wall_seconds_maximum": 7_200,
        },
        "export_and_quantization": {
            "logical_CPU_threads_maximum": 2,
            "output_and_scratch_disk_bytes_maximum": 2 * 1024**3,
            "peak_RSS_bytes_maximum": 4 * 1024**3,
            "wall_seconds_per_candidate_maximum": 1_800,
        },
        "scope": (
            "working_artifacts_and_scratch_only_excluding_the_immutable_source_archives"
        ),
        "training": {
            "logical_CPU_threads_maximum": 2,
            "output_and_scratch_disk_bytes_maximum": 4 * 1024**3,
            "peak_RSS_bytes_maximum": 8 * 1024**3,
            "wall_seconds_per_30_epoch_seed_maximum": 21_600,
        },
    }


def test_future_replay_is_iterative_registered_later_and_cannot_open_holdout() -> None:
    config = _config()
    identity = config["identity"]
    replay = config["streaming_evaluation"]
    gates = replay["development_gates"]

    assert "iterative_development" in identity["development_history"]
    for forbidden_claim in (
        "not_unseen",
        "not_blind",
        "not_holdout",
        "not_confirmatory",
    ):
        assert forbidden_claim in identity["development_history"]
    assert identity["librispeech"]["payload_for_training_or_calibration"] == (
        "forbidden"
    )
    assert identity["librispeech"]["test_clean_status"] == (
        "official_archive_absent_and_unread"
    )
    assert config["phase_order"].index(
        "freeze_and_commit_the_selected_deployment_and_its_complete_evidence"
    ) < config["phase_order"].index(
        "commit_a_separate_experiment_002_continuous_replay_config_before_the_first_experiment_002_dev_clean_score"
    )
    assert replay["argmax"] == (
        "numpy_argmax_over_the_registered_class_order_with_first_index_winning_"
        "exact_ties"
    )
    assert replay["threshold_grid"] == {
        "count": 1_001,
        "definition": "float64(integer_milli)_divided_by_float64(1000)",
        "start_milli": 0,
        "step_milli": 1,
        "stop_milli_inclusive": 1_000,
    }
    assert replay["event_rule"] == {
        "current_tick": (
            "must_have_target_argmax_and_its_float64_softmax_probability_greater_"
            "than_or_equal_to_threshold"
        ),
        "global_refractory": (
            "10_scoring_ticks_shared_by_all_targets;event_at_tick_k_allows_next_"
            "event_at_tick_k_plus_10"
        ),
        "history": "last_5_scoring_ticks_including_current_with_nonqualifying_sentinel",
        "reset": (
            "history_and_refractory_reset_at_every_utterance_or_independent_"
            "validation_stream_boundary"
        ),
        "same_target_vote": (
            "current_predicted_target_must_appear_as_a_qualifying_target_on_at_"
            "least_3_of_last_5_ticks"
        ),
        "threshold_state": (
            "independent_vote_history_and_refractory_state_for_each_threshold_milli"
        ),
    }
    assert gates == {
        "maximum_false_events_per_hour": 1.0,
        "maximum_positive_p95_latency_samples": 12_800,
        "maximum_silence_qualifying_target_rate": 0.05,
        "maximum_unknown_qualifying_target_rate": 0.05,
        "minimum_absolute_correct_detection_recall": 0.75,
        "minimum_conditional_correct_retention": 0.8,
        "minimum_every_target_recall": 0.6,
        "threshold_selection": "lowest_threshold_milli_meeting_every_gate",
    }
    assert (
        "never_use_it_for_threshold_or_deployment_selection"
        in replay["unsmoothed_diagnostic"]
    )
    assert replay["finite_gate"].startswith(
        "before_aggregation_require_every_frontend_value_ONNX_logit"
    )
    assert "no_threshold_can_pass_or_be_published" in replay["finite_gate"]
    assert (
        "strictly_greater_than_onset_sample"
        in replay["positive_validation"]["correct_detection"]
    )
    assert "numpy_percentile_method_higher" in replay["positive_validation"]["latency"]
    assert (
        "zero_or_nonfinite_denominator"
        in replay["positive_validation"]["conditional_retention"]
    )
    assert replay["validation_open_set"]["denominators"] == {
        "silence": 602,
        "unknown": 6_278,
    }
    assert (
        "actual_decoded_raw_unknown_clip"
        in replay["validation_open_set"]["unknown_stream"]
    )
    assert (
        "exact_registered_16000_sample_running_tap"
        in replay["validation_open_set"]["silence_stream"]
    )
    assert "no_access_after_pass" in replay["development_replay"]["test_clean"]
    assert (
        "forbidden_for_gradients_global_normalization"
        in config["validation"]["isolation"]
    )
    for forbidden_use in (
        "quantization_calibration",
        "augmentation_noise",
        "LayerNorm_fitting",
    ):
        assert forbidden_use in config["validation"]["isolation"]

    phase_2 = config["phase_2_replay_registration"]
    assert phase_2["known_development_identities"] == {
        "dev_archive_sha256": (
            "76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3"
        ),
        "dev_audit_report_sha256": (
            "810bbd4966d3ad5a24cd2bdd3bb1a8afb4b7325fc285e19f180fb7b26a9dbe74"
        ),
        "dev_manifest_sha256": (
            "6494fa36866b0c90eb12e8e1325339981fcd3cb5c61abd2d06dedd5d3ce6cff7"
        ),
    }
    assert phase_2["required_bindings"] == [
        "phase_1_training_config_sha256",
        "training_report_sha256",
        "selected_safetensors_sha256",
        "selected_canonical_tensor_digest",
        "normalization_stats_sha256",
        "fp32_onnx_sha256",
        "int8_attempt_report_sha256",
        "int8_onnx_sha256_or_JSON_null_if_no_artifact",
        "selected_deployment_representation_FP32_or_INT8",
        "selected_deployment_onnx_sha256",
        "export_inputs_and_config_sha256",
        "implementation_git_commit",
        "scorer_source_files",
        "scorer_source_digest_algorithm",
        "scorer_source_sha256",
        "runtime_identity_sha256",
        "benchmark_report_sha256",
        "speech_commands_manifest_sha256",
        "dev_archive_sha256",
        "dev_manifest_sha256",
        "dev_audit_report_sha256",
    ]


def test_normalization_uses_the_registered_sum_and_sumsq_path() -> None:
    normalization = _config()["normalization"]

    assert normalization["clip_count"] == 84_843
    assert normalization["frame_count_per_clip"] == 98
    assert normalization["total_frames"] == 84_843 * 98 == 8_314_614
    assert "canonical_manifest_order" in normalization["accumulation"]
    assert "float64_sum_and_sumsq" in normalization["accumulation"]
    assert "mean=sum/count" in normalization["accumulation"]
    assert normalization["output"].endswith("little_endian_binary32")
    assert normalization["application"] == (
        "load_little_endian_float32_mean_and_std;for_each_float32_log_mel_value_"
        "compute_float32_subtract_mean_then_float32_divide_by_std_with_no_epsilon_"
        "and_store_float32"
    )
    assert (
        "right_pad_with_positive_float32_zero"
        in normalization["waveform_preprocessing"]
    )
    assert normalization["zero_std"] == "error"


def test_documentation_states_the_honest_scope_and_key_limits() -> None:
    document = DOC_PATH.read_text(encoding="utf-8")

    assert "iterative development" in document
    assert "not a blind or confirmatory result" in document
    assert "23,724" in document
    assert "995 ms" in document
    assert "3-of-5" in document
    assert "2,163,840 MACs" in document
    assert "state-cold but session-warm" in document
    assert "single violation rejects the complete replay" in document
    assert "test-clean" in document
    assert "absent and unread" in document
