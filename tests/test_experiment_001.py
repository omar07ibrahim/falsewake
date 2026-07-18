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
    firewall = config["selection_artifact"]
    assert firewall["enforcement_status"] == ("registered_contract_not_yet_implemented")
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
