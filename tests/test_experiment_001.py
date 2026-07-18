from __future__ import annotations

import hashlib
import json
from pathlib import Path


def test_continuous_protocol_binds_the_published_linear_floor() -> None:
    config = json.loads(Path("configs/experiment-001.json").read_text(encoding="utf-8"))
    model = Path("models/experiment-000-linear.json")
    features = json.loads(
        Path("reports/experiment-000-features.json").read_text(encoding="utf-8")
    )
    experiment_000 = Path("configs/experiment-000.json")

    assert config["model"] == {
        "experiment_config_sha256": hashlib.sha256(
            experiment_000.read_bytes()
        ).hexdigest(),
        "features_sha256": features["features_sha256"],
        "portable_json_path": "models/experiment-000-linear.json",
        "portable_json_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
    }
    assert (
        config["positive_validation"]["examples_sha256"] == features["examples_sha256"]
    )


def test_continuous_thresholds_and_holdout_rule_are_fixed() -> None:
    config = json.loads(Path("configs/experiment-001.json").read_text(encoding="utf-8"))
    grid = config["scoring"]["threshold_grid"]
    assert grid == {
        "count": 1001,
        "definition": "float64(integer_milli)_divided_by_float64(1000)",
        "start_milli": 0,
        "step_milli": 1,
        "stop_milli_inclusive": 1000,
    }
    assert config["selection"] == {
        "exact_integer_gates": (
            "false_event_gate_is_events_times_57600000_lte_scored_exposure_samples;"
            "retention_gate_is_correct_accept_count_times_5_gte_baseline_correct_"
            "count_times_4"
        ),
        "false_event_rate_statistic": (
            "raw_event_count_divided_by_scored_exposure_hours_point_estimate_not_an_"
            "interval_bound"
        ),
        "if_multiple_pass": "lowest_threshold",
        "if_none_pass": (
            "reject_registered_threshold_grid_and_keep_test_clean_untouched"
        ),
        "max_dev_false_events_per_hour": 1.0,
        "min_validation_conditional_correct_retention": 0.8,
    }
    assert (
        config["negative_source"]["test_clean_access"]
        == "forbidden_until_one_configuration_is_selected"
    )
    assert config["negative_source"]["test_clean_official_identity"] == {
        "archive_bytes": 346_663_984,
        "archive_md5": "32fa31d27d2e1cad72775fee3f4849a9",
        "archive_sha256": (
            "39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23"
        ),
        "archive_url": "https://www.openslr.org/resources/12/test-clean.tar.gz",
        "dataset": "LibriSpeech test-clean",
        "identity_provenance": (
            "OpenSLR_12_md5sum_and_openslr_librispeech_asr_download_checksum_"
            "metadata_registered_before_archive_access"
        ),
    }
    verified = config["negative_source"]["archive_audit"]["verified_dev_clean_output"]
    assert verified == {
        "audit_report_sha256": (
            "810bbd4966d3ad5a24cd2bdd3bb1a8afb4b7325fc285e19f180fb7b26a9dbe74"
        ),
        "manifest_sha256": (
            "6494fa36866b0c90eb12e8e1325339981fcd3cb5c61abd2d06dedd5d3ce6cff7"
        ),
        "scored_exposure_samples": 308_310_400,
        "source_samples": 310_337_932,
        "speaker_count": 40,
        "utterance_count": 2_703,
    }
    firewall = config["selection_artifact"]
    assert firewall["enforcement_status"] == "implemented_fail_closed"
    assert firewall["acquisition_contract"] == (
        "after_final_artifact_HEAD_config_index_scorer_model_runtime_and_"
        "development_evidence_revalidation_create_one_anonymous_memfd_with_sealing_"
        "enabled;invoke_one_trusted_supplier_once_with_only_a_writable_binary_"
        "stream;fsync_then_apply_F_SEAL_WRITE_F_SEAL_GROW_F_SEAL_SHRINK_F_SEAL_"
        "SEAL;require_exact_registered_test_clean_size_MD5_and_SHA256;run_no_Git_"
        "after_supplier;return_the_same_sealed_inode_through_an_independent_read_"
        "only_open_file_description"
    )
    assert firewall["git_validation_contract"] == (
        "copy_only_physical_bounded_object_files_into_a_new_bare_snapshot;ignore_"
        "repository_local_config_info_alternates_replacements_and_lazy_fetch;use_"
        "only_non_worktree_plumbing_with_full_commit_IDs;capture_one_physical_"
        "bounded_index_and_require_every_stage_zero_mode_OID_path_entry_to_equal_"
        "the_recursive_artifact_HEAD_tree;reject_unmerged_or_sparse_entries;never_"
        "run_status_hooks_filters_or_other_worktree_commands"
    )
    assert firewall["trusted_git_path"] == ("reports/experiment-001-selection.json")
    assert firewall["schema_version"] == 1
    assert firewall["status_for_test_access"] == "pass"
    assert firewall["required_fields"] == [
        "schema_version",
        "status",
        "experiment_config_sha256",
        "scorer_source_sha256",
        "runtime_identity_sha256",
        "implementation_git_commit",
        "dev_archive_sha256",
        "dev_manifest_sha256",
        "dev_audit_report_sha256",
        "dev_replay_report_sha256",
        "selected_threshold_milli",
    ]
    assert firewall["status_values"] == ["pass", "reject"]
    assert firewall["selected_threshold_milli"].startswith(
        "JSON_integer_0_through_1000"
    )
    assert config["selection"]["false_event_rate_statistic"].endswith(
        "point_estimate_not_an_interval_bound"
    )


def test_continuous_audit_covers_every_payload_and_transcript() -> None:
    config = json.loads(Path("configs/experiment-001.json").read_text(encoding="utf-8"))
    audit = config["negative_source"]["archive_audit"]

    assert audit["canonical_order"] == "relative_POSIX_path_ascending"
    assert audit["regular_member_population"] == {
        "flac": "every_canonical_dev_clean_FLAC_exactly_once",
        "metadata_exact_paths": [
            "LibriSpeech/BOOKS.TXT",
            "LibriSpeech/CHAPTERS.TXT",
            "LibriSpeech/LICENSE.TXT",
            "LibriSpeech/README.TXT",
            "LibriSpeech/SPEAKERS.TXT",
        ],
        "other": "reject",
        "transcripts": (
            "exactly_one_canonical_<speaker>-<chapter>.trans.txt_for_each_and_only_"
            "each_represented_chapter"
        ),
    }
    assert "decoded_pcm16le_sha256" in audit["manifest_fields"]
    assert "transcript" in audit["manifest_fields"]
    assert "bijection" in audit["transcripts"]
    assert audit["inventory_domain_hex"] == {
        "decoded_pcm16le": (
            "66616c736577616b652d6578703030312d6c696272697370656563682d6465636f"
            "6465642d70636d31366c652d696e76656e746f72792d763100"
        ),
        "metadata": (
            "66616c736577616b652d6578703030312d6c696272697370656563682d6d657461"
            "646174612d696e76656e746f72792d763100"
        ),
        "raw_flac": (
            "66616c736577616b652d6578703030312d6c696272697370656563682d7261772d"
            "666c61632d696e76656e746f72792d763100"
        ),
        "transcript": (
            "66616c736577616b652d6578703030312d6c696272697370656563682d7472616e"
            "7363726970742d696e76656e746f72792d763100"
        ),
    }
    assert audit["output_files"] == {
        "audit_report": "dev-clean.audit.json",
        "manifest": "dev-clean.manifest.jsonl",
    }
    assert config["final_evaluation_if_selected"]["selection_inputs_forbidden"]
    assert config["negative_source"]["archive_sha256"] == (
        "76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3"
    )


def test_dev_clean_header_report_matches_the_registered_source() -> None:
    config = json.loads(
        Path("configs/experiment-001.json").read_text(encoding="utf-8")
    )["negative_source"]
    report = json.loads(
        Path("reports/dev-clean-header-inspection.json").read_text(encoding="utf-8")
    )

    assert report["archive_bytes"] == config["archive_bytes"]
    assert report["archive_md5"] == config["archive_md5"]
    assert report["archive_sha256"] == config["archive_sha256"]
    assert report["regular_count"] == (
        report["flac_count"] + report["transcript_count"] + report["metadata_count"]
    )
    assert report["flac_count"] == 2_703
    assert report["chapter_count"] == report["transcript_count"] == 97
    assert report["speaker_count"] == 40


def test_dev_clean_payload_report_is_bound_to_the_reproduced_manifest() -> None:
    config = json.loads(
        Path("configs/experiment-001.json").read_text(encoding="utf-8")
    )["negative_source"]
    report_path = Path("reports/dev-clean-audit.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    header = json.loads(
        Path("reports/dev-clean-header-inspection.json").read_text(encoding="utf-8")
    )

    assert hashlib.sha256(report_path.read_bytes()).hexdigest() == (
        "810bbd4966d3ad5a24cd2bdd3bb1a8afb4b7325fc285e19f180fb7b26a9dbe74"
    )
    assert report["archive_bytes"] == config["archive_bytes"]
    assert report["archive_md5"] == config["archive_md5"]
    assert report["archive_sha256"] == config["archive_sha256"]
    assert report["manifest_sha256"] == (
        "6494fa36866b0c90eb12e8e1325339981fcd3cb5c61abd2d06dedd5d3ce6cff7"
    )
    assert report["utterance_count"] == report["flac_count"] == 2_703
    assert report["chapter_count"] == report["transcript_count"] == 97
    assert report["speaker_count"] == header["speaker_count"] == 40
    assert report["metadata_count"] == header["metadata_count"] == 5
    assert report["sample_rate"] == 16_000
    assert report["source_duration_seconds"] == (
        report["source_samples"] / report["sample_rate"]
    )
    assert report["source_duration_hours"] == report["source_samples"] / 57_600_000
    for field in (
        "decoded_pcm16le_inventory_sha256",
        "metadata_inventory_sha256",
        "raw_flac_inventory_sha256",
        "transcript_inventory_sha256",
    ):
        assert len(bytes.fromhex(report[field])) == 32


def test_continuous_source_and_event_evidence_identities_are_bounded() -> None:
    config = json.loads(Path("configs/experiment-001.json").read_text(encoding="utf-8"))

    identity = config["implementation_identity"]
    source_digest = identity["scorer_source_sha256"]
    assert source_digest.startswith("SHA256(ASCII_falsewake-exp001-scorer-source-v1")
    assert "UINT32LE(path_UTF8_byte_count)" in source_digest
    assert identity["scorer_source_files"] == [
        "src/falsewake/baseline_data.py",
        "src/falsewake/continuous_replay.py",
        "src/falsewake/feature_matrix.py",
        "src/falsewake/features.py",
        "src/falsewake/holdout.py",
        "src/falsewake/librispeech.py",
        "src/falsewake/speech_commands.py",
    ]
    assert identity["runtime_versions"] == [
        "python",
        "numpy",
        "scipy",
        "soundfile",
        "libsndfile",
        "platform",
    ]
    examples = config["metrics"]["top_false_events_at_decision_thresholds"]
    assert examples["count_per_threshold"] == 50
    assert "retention_frontier" in examples["thresholds"]
    assert "negative_frontier" in examples["thresholds"]
    assert "selected_threshold_milli_if_pass" in examples["thresholds"]


def test_continuous_bootstrap_rng_stream_is_fixed_across_thresholds() -> None:
    config = json.loads(Path("configs/experiment-001.json").read_text(encoding="utf-8"))
    bootstrap = config["metrics"]["uncertainty"]["speaker_bootstrap_95_percent"]

    assert "ascending_integer_ID_order" in bootstrap
    assert "one_shared_int64_matrix" in bootstrap
    assert "size=(10000,S)" in bootstrap
    assert "used_for_every_threshold" in bootstrap


def test_continuous_event_state_and_exposure_are_unambiguous() -> None:
    config = json.loads(Path("configs/experiment-001.json").read_text(encoding="utf-8"))

    scoring = config["scoring"]
    assert scoring["initial_next_allowed_sample"] == 0
    assert scoring["refractory_samples"] == 16_000
    assert scoring["state"] == (
        "independent_for_each_threshold_and_reset_at_each_utterance"
    )
    assert scoring["argmax_tie_break"] == ("first_index_in_portable_model_class_order")
    assert config["windowing"]["window_count"].startswith("max(0,")
    assert "57600000" in config["windowing"]["scored_exposure_hours"]


def test_replay_report_and_positive_denominators_are_frozen() -> None:
    config = json.loads(Path("configs/experiment-001.json").read_text(encoding="utf-8"))
    positive = config["positive_validation"]
    report = config["replay_report"]

    assert positive["target_order"] == [
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
    ]
    assert sum(positive["target_support"]) == positive["target_example_count"] == 3_703
    assert (
        sum(positive["baseline_correct_by_target"])
        == (positive["baseline_correct_count"])
        == 2_088
    )
    assert positive["feature_cache"] == {
        "dtype": "float32",
        "experiment_config_sha256": (
            "023fdb4a61ae434b519bf5f20bc3be123d1ab2208202c7b9997e509edb461b98"
        ),
        "features_npy_sha256": (
            "3d7981af946263f6d857f5c19b28e4e39ef0594c46b42c5c818b8d47e7b98db0"
        ),
        "features_sha256": (
            "b46cdd047b52e7d7291b4551c905d4ce7bb058dbec0d519237549bdceddc3526"
        ),
        "manifest_sha256": (
            "d28e6993101bd6bc452033bcb7355b25e3b84ab5c51a1458cc097dc93c60f78b"
        ),
        "metadata_json_sha256": (
            "2c7ed6e406579b38bc4c240dc54851ec5b654d162e66291971e456b8a5309a47"
        ),
        "row_selection": (
            "build_sampling_plan_exact_order_then_rows_where_split_is_validation_"
            "and_label_is_in_target_order"
        ),
        "shape": [46_254, 80],
    }
    assert report["schema_version"] == 1
    assert report["threshold_axis"] == (
        "exactly_1001_rows_in_ascending_threshold_milli_0_through_1000"
    )
    assert report["threshold_row_fields"][:3] == [
        "threshold_milli",
        "dev_event_count",
        "dev_event_count_by_target",
    ]
    assert report["selection_fields"] == [
        "status",
        "selected_threshold_milli",
        "negative_frontier_milli",
        "retention_frontier_milli",
    ]
    assert (
        report["runtime_document"]["keys"]
        == config["implementation_identity"]["runtime_versions"]
    )
    assert "speaker_rows" in report["dev_fields"]
    assert "1001" in report["speaker_row_contract"]
    assert report["identity_fields"][-1] == ("positive_feature_matrix_semantic_sha256")
    assert "feature_cache.features_sha256" in report["identity_semantics"]
    assert "[lower,upper]" in report["interval_contract"]
    assert "baseline_correct_by_target" in report["positive_validation_fields"]
    assert report["top_false_events_fields"] == ["threshold_milli", "events"]
    assert (
        report["top_false_event_fields"]
        == config["metrics"]["top_false_events_at_decision_thresholds"]["fields"]
    )
