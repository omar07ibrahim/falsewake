from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import cast

import pytest

TOOLS_DIRECTORY = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIRECTORY))

import render_portfolio_visuals as visuals  # noqa: E402


def _mapping(value: object) -> dict[str, object]:
    assert type(value) is dict
    mapping = cast(dict[object, object], value)
    assert all(type(key) is str for key in mapping)
    return cast(dict[str, object], mapping)


def _list(value: object) -> list[object]:
    assert type(value) is list
    return cast(list[object], value)


def _load_provenance() -> tuple[bytes, dict[str, object]]:
    contents = (visuals.OUTPUT_DIRECTORY / "provenance.json").read_bytes()
    return contents, _mapping(json.loads(contents))


def _all_text_values(value: object) -> list[str]:
    if type(value) is str:
        return [value]
    if type(value) is list:
        return [
            text
            for item in cast(list[object], value)
            for text in _all_text_values(item)
        ]
    if type(value) is dict:
        return [
            text
            for item in cast(dict[object, object], value).values()
            for text in _all_text_values(item)
        ]
    return []


def test_exact_facts_are_extracted_from_allowlisted_evidence() -> None:
    evidence = visuals.extract_evidence()

    assert evidence.frontend == visuals.FrontendFacts(
        sample_rate=16_000,
        clip_samples=16_000,
        window_samples=400,
        hop_samples=160,
        fft_samples=512,
        mel_bins=40,
        frame_count=98,
        summary_values=80,
        replay_hop_samples=1_600,
        refractory_samples=16_000,
        threshold_count=1_001,
    )
    assert evidence.dataset.training_examples == 36_941
    assert evidence.dataset.validation_examples == 4_429
    assert evidence.dataset.test_examples == 4_884
    assert evidence.dataset.validation_role == "diagnostic_only_no_selection"
    assert evidence.dataset.headline_split == "test"
    assert evidence.dataset.dev_utterances == 2_703
    assert evidence.dataset.dev_speakers == 40
    assert evidence.dataset.dev_exposure_samples == 308_310_400
    assert evidence.dataset.positive_validation_examples == 3_703
    assert evidence.dataset.baseline_correct == 2_088
    assert evidence.lineage[1].boundary == "MEASURED · REJECTED"

    assert evidence.causal.mel_bins == 40
    assert evidence.causal.channels == 48
    assert evidence.causal.output_classes == 12
    assert evidence.causal.dilations == (1, 2, 4, 8, 1, 2, 4, 8)
    assert evidence.causal.tcn_receptive_frames == 61
    assert evidence.causal.model_receptive_frames == 98
    assert evidence.causal.state_values == 4_656
    assert evidence.causal.parameter_count == 23_724

    threshold = evidence.threshold
    assert len(threshold.points) == 1_001
    assert threshold.retention_frontier_milli == 395
    assert threshold.negative_frontier_milli == 991
    assert threshold.selection_status == "reject"
    assert threshold.points[395].retention == pytest.approx(1_674 / 2_088)
    assert threshold.points[395].false_events_per_hour == pytest.approx(
        10_147 * 57_600_000 / 308_310_400
    )
    assert threshold.points[991].retention == pytest.approx(3 / 2_088)
    assert threshold.points[991].false_events_per_hour == pytest.approx(
        4 * 57_600_000 / 308_310_400
    )

    unknown = evidence.unknown_words
    assert len(unknown) == 25
    assert sum(row.support for row in unknown) == 405
    assert sum(row.predicted_unknown_count for row in unknown) == 76
    assert (unknown[0].source_word, unknown[0].predicted_unknown_count) == (
        "two",
        18,
    )
    assert unknown[0].recall == pytest.approx(18 / 25)


def test_renderer_imports_no_registered_experiment_implementation() -> None:
    source = Path("tools/render_portfolio_visuals.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    project_imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith("falsewake")
    }
    assert project_imports == {"falsewake.result_plots"}
    assert not any(
        path.startswith("src/falsewake/experiment_")
        or path.startswith("tests/test_experiment_")
        or path.startswith("configs/experiment-")
        for path in visuals.SOURCE_ALLOWLIST
    )
    assert not any(
        ".t/" in path or path.startswith("/") for path in visuals.SOURCE_ALLOWLIST
    )

    modules_before = frozenset(sys.modules)
    visuals.build_bundle()
    forbidden_new_modules = sorted(
        module
        for module in frozenset(sys.modules) - modules_before
        if module.startswith("falsewake.experiment_")
    )
    assert forbidden_new_modules == []


def test_all_svg_assets_are_local_accessible_and_current() -> None:
    visuals.check_assets()
    observed = {
        path.name
        for path in visuals.OUTPUT_DIRECTORY.iterdir()
        if path.is_file() and path.suffix == ".svg"
    }
    assert observed == set(visuals.VISUAL_FILENAMES)
    for name in visuals.VISUAL_FILENAMES:
        contents = (visuals.OUTPUT_DIRECTORY / name).read_text(encoding="utf-8")
        assert '<title id="title">' in contents
        assert '<desc id="desc">' in contents
        assert 'role="img"' in contents
        assert 'aria-labelledby="title desc"' in contents
        assert "<image" not in contents
        assert "https://" not in contents
        assert "http://" not in contents.replace(
            'xmlns="http://www.w3.org/2000/svg"', ""
        )
        assert "file:" not in contents
        assert "/home/" not in contents
        assert all(line == line.rstrip() for line in contents.splitlines())


def test_provenance_is_canonical_current_and_contains_nonclaims() -> None:
    contents, document = _load_provenance()
    assert contents == (
        json.dumps(
            document,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )
    assert document["source_allowlist"] == sorted(visuals.SOURCE_ALLOWLIST)
    renderer = _mapping(document["renderer"])
    renderer_path = cast(str, renderer["path"])
    assert renderer_path == "tools/render_portfolio_visuals.py"
    assert renderer["sha256"] == hashlib.sha256(
        (visuals.REPOSITORY_ROOT / renderer_path).read_bytes()
    ).hexdigest()
    visual_records = _mapping(document["visuals"])
    assert set(visual_records) == set(visuals.VISUAL_FILENAMES)
    for name in visuals.VISUAL_FILENAMES:
        record = _mapping(visual_records[name])
        svg = (visuals.OUTPUT_DIRECTORY / name).read_bytes()
        assert record["output_sha256"] == hashlib.sha256(svg).hexdigest()
        nonclaims = _list(record["nonclaims"])
        assert nonclaims
        for raw_source in _list(record["source_records"]):
            source = _mapping(raw_source)
            relative_path = cast(str, source["path"])
            assert relative_path in visuals.SOURCE_ALLOWLIST
            assert (
                source["sha256"]
                == hashlib.sha256(
                    (visuals.REPOSITORY_ROOT / relative_path).read_bytes()
                ).hexdigest()
            )
    all_text = _all_text_values(document)
    assert not any(value.startswith("/") or "/home/" in value for value in all_text)
    lineage = _mapping(visual_records["experiment-lineage.svg"])
    assert "Experiment 006 root cause is undetermined from retained evidence." in _list(
        lineage["nonclaims"]
    )
    threshold = _mapping(visual_records["threshold-tradeoff.svg"])
    facts = _mapping(threshold["facts"])
    assert facts["false_events_per_hour_formula"] == (
        "dev_event_count * 57600000 / scored_exposure_samples"
    )
    assert facts["retention_formula"] == (
        "validation_correct_accept_count / baseline_correct_count"
    )


def test_two_complete_renders_are_byte_identical() -> None:
    first = visuals.build_bundle()
    second = visuals.build_bundle()

    assert first.visuals == second.visuals
    assert first.provenance == second.provenance


def test_existing_report_assets_are_regenerated_and_byte_current() -> None:
    _, document = _load_provenance()
    records = _mapping(document["existing_report_assets"])
    assert set(records) == {f"reports/{name}" for name in visuals.REPORT_ASSETS}
    for relative_path, raw_record in records.items():
        record = _mapping(raw_record)
        contents = (visuals.REPOSITORY_ROOT / relative_path).read_bytes()
        assert record["verified_byte_current"] is True
        assert record["sha256"] == hashlib.sha256(contents).hexdigest()
        sources = _list(record["source_records"])
        assert len(sources) == 3


def test_atomic_asset_writer_rejects_symlinks(tmp_path: Path) -> None:
    output = tmp_path / "evidence.svg"
    visuals._write_if_changed(output, b"first\n")
    assert output.read_bytes() == b"first\n"
    assert output.stat().st_mode & 0o777 == 0o644

    output.chmod(0o600)
    visuals._write_if_changed(output, b"first\n")
    assert output.stat().st_mode & 0o777 == 0o644

    visuals._write_if_changed(output, b"second\n")
    assert output.read_bytes() == b"second\n"
    assert not list(tmp_path.glob(".evidence.svg.*.tmp"))

    output.unlink()
    target = tmp_path / "outside.svg"
    target.write_bytes(b"outside\n")
    output.symlink_to(target)
    with pytest.raises(
        visuals.PortfolioVisualError, match="refusing to replace symlink"
    ):
        visuals._write_if_changed(output, b"forbidden\n")
    assert target.read_bytes() == b"outside\n"
