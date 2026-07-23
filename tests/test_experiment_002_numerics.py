from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
import pytest

from falsewake.experiment_002_frontend import complete_log_mel_frames
from falsewake.experiment_002_rng import (
    bernoulli,
    encode_command_identity,
    encode_window_identity,
    uniform_integer,
    uniform_real,
)

CONFIG_PATH = Path("configs/experiment-002-numerics.json")
PHASE_1_PATH = Path("configs/experiment-002-training.json")
DOC_PATH = Path("docs/experiment-002.md")
SEED = 20_260_719


class DrawIdentity(TypedDict):
    seed: int
    epoch: int
    identity: bytes


class FrontendProbe(TypedDict):
    different_float32_values: int
    legacy_shape: list[int]
    legacy_sha256: str
    maximum_absolute_difference: str
    streaming_shape: list[int]
    streaming_sha256: str
    value_count: int
    waveform_sha256: str


def _config() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(CONFIG_PATH.read_text(encoding="utf-8")))


def _frontend_probe(*, openblas_threads: int) -> FrontendProbe:
    script = textwrap.dedent(
        """
        import hashlib
        import json

        import numpy as np

        from falsewake.experiment_002_frontend import complete_log_mel_frames
        from falsewake.features import log_mel_spectrogram

        def sha256_float32(values):
            little_endian = np.ascontiguousarray(values).astype("<f4", copy=False)
            return hashlib.sha256(little_endian.tobytes(order="C")).hexdigest()

        sample_index = np.arange(16_000, dtype=np.float64)
        waveform = (
            0.31 * np.sin(2.0 * np.pi * 731.0 * sample_index / 16_000.0)
            + 0.17 * np.cos(2.0 * np.pi * 1_913.0 * sample_index / 16_000.0)
        ).astype(np.float32)
        streaming = complete_log_mel_frames(waveform)
        legacy = log_mel_spectrogram(waveform)
        difference = np.abs(
            streaming.astype(np.float64) - legacy.astype(np.float64)
        )
        print(
            json.dumps(
                {
                    "different_float32_values": int(
                        np.count_nonzero(streaming != legacy)
                    ),
                    "legacy_shape": list(legacy.shape),
                    "legacy_sha256": sha256_float32(legacy),
                    "maximum_absolute_difference": np.max(
                        difference
                    ).item().hex(),
                    "streaming_shape": list(streaming.shape),
                    "streaming_sha256": sha256_float32(streaming),
                    "value_count": int(streaming.size),
                    "waveform_sha256": sha256_float32(waveform),
                },
                sort_keys=True,
            )
        )
        """
    )
    environment = {
        "LANG": "C",
        "LC_ALL": "C",
        "MKL_DYNAMIC": "FALSE",
        "MKL_NUM_THREADS": "1",
        "NPY_DISABLE_CPU_FEATURES": "X86_V4",
        "OMP_DYNAMIC": "FALSE",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_CORETYPE": "Haswell",
        "OPENBLAS_NUM_THREADS": str(openblas_threads),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "TZ": "UTC",
    }
    completed = subprocess.run(
        (sys.executable, "-I", "-B", "-W", "error", "-c", script),
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        check=False,
        text=True,
        timeout=20.0,
    )
    assert completed.returncode == 0, completed.stderr
    return cast(FrontendProbe, json.loads(completed.stdout))


def _sha256_float32(values: np.ndarray[Any, Any]) -> str:
    little_endian = np.ascontiguousarray(values).astype("<f4", copy=False)
    return hashlib.sha256(little_endian.tobytes(order="C")).hexdigest()


def _float32_bits(value: np.float32) -> str:
    return f"0x{value.view(np.uint32).item():08x}"


def _rms_float64(waveform: np.ndarray[Any, Any]) -> np.float64:
    squares = np.square(waveform, dtype=np.float64)
    total = np.sum(squares, dtype=np.float64)
    return np.float64(np.sqrt(total / np.float64(16_000)))


def _mix_at_snr(
    command: np.ndarray[Any, Any],
    noise: np.ndarray[Any, Any],
    snr_db: float,
) -> tuple[np.ndarray[Any, Any], np.float64, np.float64, float, np.float32]:
    command_rms = _rms_float64(command)
    noise_rms = _rms_float64(noise)
    if noise_rms == np.float64(0.0):
        raise ValueError("noise RMS must be positive")
    ratio = 10.0 ** (snr_db / 20.0)
    scale64 = (
        0.0 if command_rms == np.float64(0.0) else command_rms / (noise_rms * ratio)
    )
    scale32 = np.float32(scale64)
    scaled_noise = np.multiply(noise, scale32, dtype=np.float32)
    mixed = np.add(command, scaled_noise, dtype=np.float32)
    if not np.all(np.isfinite(mixed)):
        raise ValueError("mix must be finite before clamping")
    return mixed, command_rms, noise_rms, scale64, scale32


def test_numerics_addendum_is_canonical_and_binds_pre_result_state() -> None:
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
        "authority",
        "feature_pipeline",
        "golden_vectors",
        "identity",
        "runtime",
        "schema_version",
    }
    assert config["schema_version"] == 1

    identity = config["identity"]
    assert identity["phase_1_training_config_sha256"] == (
        "a4a21ff66ef04bf15950853acda0897f24ed01eecf566c795a7330e69593d44a"
    )
    assert (
        hashlib.sha256(PHASE_1_PATH.read_bytes()).hexdigest()
        == (identity["phase_1_training_config_sha256"])
    )
    assert identity["interpretation_phase"] == (
        "committed_after_model_implementation_but_before_normalization_artifact_"
        "feature_cache_training_export_benchmark_or_experiment_002_metrics"
    )
    assert identity["purpose"] == (
        "resolve_pre_result_numeric_ambiguities_without_modifying_the_original_"
        "phase_1_preregistration"
    )

    expected_sources = {
        "src/falsewake/experiment_002_frontend.py": (
            "fdfa2534080cfbea2143c5b87905e1ce43be0d1e4aad4503063506d15e6c65ca"
        ),
        "src/falsewake/experiment_002_normalization.py": (
            "73bbe76aab35d352d2b702f67480f93645eb09cbd7627a9fafde86d3ecb69a81"
        ),
        "src/falsewake/experiment_002_rng.py": (
            "174e445b47d6da586ae92ea3c4a694966bb6e2acdc2be92f31c1a7023a0fcd4e"
        ),
    }
    assert identity["bound_sources"] == expected_sources
    for path, expected in expected_sources.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected


def test_streaming_frontend_is_the_only_experiment_002_feature_authority() -> None:
    authority = _config()["authority"]
    frontend = authority["frontend"]

    assert frontend["implementation"] == ("src/falsewake/experiment_002_frontend.py")
    assert frontend["output"] == "98_by_40_time_major_float32_complete_frames"
    assert frontend["scope"] == [
        "normalization_statistics",
        "training",
        "validation",
        "quantization_calibration",
        "streaming_replay",
        "frontend_plus_model_benchmark",
    ]
    assert authority["legacy_batch_frontend"]["role"].endswith(
        "not_an_experiment_002_feature_authority"
    )

    probe = authority["legacy_batch_frontend"]["runtime_probe"]
    frozen_probe = _frontend_probe(openblas_threads=4)
    registered_probe = _frontend_probe(openblas_threads=1)

    assert (
        frozen_probe["streaming_shape"]
        == frozen_probe["legacy_shape"]
        == registered_probe["streaming_shape"]
        == registered_probe["legacy_shape"]
        == [98, 40]
    )
    assert frozen_probe["value_count"] == probe["value_count"] == 3_920
    assert registered_probe["value_count"] == frozen_probe["value_count"]
    assert (
        frozen_probe["different_float32_values"]
        == (probe["different_float32_values"])
        == 72
    )
    assert (
        frozen_probe["maximum_absolute_difference"]
        == (probe["maximum_absolute_difference"])
        == "0x1.0000000000000p-19"
    )
    assert registered_probe["different_float32_values"] == 77
    assert (
        registered_probe["maximum_absolute_difference"]
        == frozen_probe["maximum_absolute_difference"]
    )
    assert (
        registered_probe["waveform_sha256"]
        == frozen_probe["waveform_sha256"]
        == "3d5bf9a94940531e9165744c5892eef9c9aa2afa89c434fb3da514f16ffc82b0"
    )
    assert (
        registered_probe["streaming_sha256"]
        == frozen_probe["streaming_sha256"]
        == "963e77d8c1e0cffdcea862f32447f8aeb47c78ede96d3d4059337b60b8aef443"
    )
    assert (
        registered_probe["legacy_sha256"]
        == "0be916f1b8b58434fa7394b86fec4bf914d7d790e2c09d34bb6bf21ebb97e057"
    )
    assert (
        frozen_probe["legacy_sha256"]
        == "55bebabfd55babac0cb46a9c308d0bec70a3146593bd477a1000893a41341bc8"
    )
    assert registered_probe["legacy_sha256"] != frozen_probe["legacy_sha256"]


def test_registered_command_pipeline_matches_every_numeric_golden() -> None:
    golden = _config()["golden_vectors"]["command"]
    identity = encode_command_identity("yes/alice_nohash_0.wav")
    common: DrawIdentity = {"seed": SEED, "epoch": 0, "identity": identity}

    sample_index = np.arange(15_997, dtype=np.int64)
    command_pcm16 = (((sample_index * 7_919 + 12_345) % 65_536) - 32_768).astype(
        np.int16
    )
    noise_index = np.arange(16_000, dtype=np.int64)
    noise_pcm16 = (((noise_index * 3_571 + 22_222) % 65_536) - 32_768).astype(np.int16)
    command = np.divide(
        command_pcm16.astype(np.float32),
        np.float32(32_768.0),
        dtype=np.float32,
    )
    noise = np.divide(
        noise_pcm16.astype(np.float32),
        np.float32(32_768.0),
        dtype=np.float32,
    )

    assert golden["pcm"] == {
        "generator": "int16(((i*7919+12345)%65536)-32768)",
        "sample_count": 15_997,
    }
    assert golden["noise"] == {
        "generator": "int16(((i*3571+22222)%65536)-32768)",
        "sample_count": 16_000,
    }
    padded = np.zeros(16_000, dtype=np.float32)
    padded[: command.size] = command
    assert (
        _sha256_float32(padded)
        == (golden["intermediates"]["padded_float32_sha256"])
        == "67bd4c468f075866c022cacc4bd2173fa41f0076592818f75b13afe55da5e9b1"
    )

    shift = uniform_integer(**common, domain="command-shift", bound=3_201) - 1_600
    assert shift == golden["draws"]["shift_samples"] == 698
    shifted = np.zeros(16_000, dtype=np.float32)
    shifted[shift:] = padded[: 16_000 - shift]
    assert (
        _sha256_float32(shifted)
        == (golden["intermediates"]["shifted_float32_sha256"])
        == "c2bf618daf0a8d6f18a24926fefc901b8555a3cce6b158bac3d55a1c2efa8b1e"
    )

    gain_db = uniform_real(
        **common,
        domain="command-gain-db",
        minimum=-6.0,
        maximum_exclusive=6.0,
    )
    gain_factor = np.float32(10.0 ** (gain_db / 20.0))
    assert gain_db.hex() == golden["draws"]["gain_db_float64_hex"]
    assert (
        _float32_bits(gain_factor)
        == (golden["draws"]["gain_factor_float32_bits"])
        == "0x3ff84b4b"
    )
    gained = np.multiply(shifted, gain_factor, dtype=np.float32)
    assert (
        _sha256_float32(gained)
        == (golden["intermediates"]["gained_float32_sha256"])
        == "56d15f0adbe96a2404092c9851c2cf0c909ad6c9f3ed2d4287886fb53da8405c"
    )

    apply_draw = uniform_real(
        **common,
        domain="command-noise-apply",
        minimum=0.0,
        maximum_exclusive=1.0,
    )
    assert apply_draw.hex() == golden["draws"]["noise_apply_float64_hex"]
    assert (
        bernoulli(**common, domain="command-noise-apply", probability=0.8)
        is golden["draws"]["noise_apply"]
        is True
    )

    universe_index = uniform_integer(
        **common,
        domain="command-noise-window",
        bound=4_387_887,
    )
    assert universe_index == golden["draws"]["noise_universe_index"] == 3_263_409
    train_backgrounds = (
        ("_background_noise_/doing_the_dishes.wav", 1_522_930),
        ("_background_noise_/dude_miaowing.wav", 988_891),
        ("_background_noise_/exercise_bike.wav", 980_062),
        ("_background_noise_/pink_noise.wav", 960_000),
    )
    remaining = universe_index
    selected: tuple[str, int] | None = None
    for path, sample_count in train_backgrounds:
        window_count = sample_count - 16_000 + 1
        if remaining < window_count:
            selected = path, remaining
            break
        remaining -= window_count
    assert (
        selected
        == (
            golden["draws"]["noise_window"]["path"],
            golden["draws"]["noise_window"]["start_sample"],
        )
        == ("_background_noise_/exercise_bike.wav", 783_586)
    )

    snr_db = uniform_real(
        **common,
        domain="command-noise-snr-db",
        minimum=0.0,
        maximum_exclusive=20.0,
    )
    assert snr_db.hex() == golden["draws"]["snr_db_float64_hex"]
    mixed, command_rms, noise_rms, scale64, scale32 = _mix_at_snr(gained, noise, snr_db)
    assert (
        command_rms.item().hex() == (golden["intermediates"]["command_rms_float64_hex"])
    )
    assert noise_rms.item().hex() == (golden["intermediates"]["noise_rms_float64_hex"])
    assert scale64.hex() == golden["intermediates"]["scale_float64_hex"]
    assert (
        _float32_bits(scale32)
        == (golden["intermediates"]["scale_float32_bits"])
        == "0x3f84605c"
    )

    clamped = np.clip(mixed, np.float32(-1.0), np.float32(1.0))
    assert clamped.dtype == np.dtype(np.float32)
    assert (
        _sha256_float32(clamped)
        == (golden["intermediates"]["clamped_float32_sha256"])
        == "8692de6bacedd9036a87198ca5ff79cb0c063b91faab7096278003f1a1814d97"
    )

    log_mel = complete_log_mel_frames(clamped)
    assert log_mel.shape == (98, 40)
    assert (
        _sha256_float32(log_mel)
        == golden["intermediates"]["streaming_frontend_98_by_40_float32_sha256"]
    )

    normalized = np.divide(
        np.subtract(log_mel, np.zeros(40, dtype=np.float32), dtype=np.float32),
        np.ones(40, dtype=np.float32),
        dtype=np.float32,
    )
    time_width = uniform_integer(**common, domain="command-time-width", bound=11)
    time_start = uniform_integer(
        **common,
        domain="command-time-start",
        bound=98 - time_width + 1,
    )
    mel_width = uniform_integer(**common, domain="command-mel-width", bound=5)
    mel_start = uniform_integer(
        **common,
        domain="command-mel-start",
        bound=40 - mel_width + 1,
    )
    assert (
        (time_start, time_width, mel_start, mel_width)
        == (
            golden["draws"]["time_mask_start"],
            golden["draws"]["time_mask_width"],
            golden["draws"]["mel_mask_start"],
            golden["draws"]["mel_mask_width"],
        )
        == (3, 8, 32, 1)
    )
    normalized[time_start : time_start + time_width, :] = np.float32(0.0)
    normalized[:, mel_start : mel_start + mel_width] = np.float32(0.0)
    assert not np.any(np.signbit(normalized[time_start : time_start + time_width, :]))
    assert not np.any(np.signbit(normalized[:, mel_start : mel_start + mel_width]))
    assert (
        _sha256_float32(normalized)
        == golden["intermediates"]["masked_identity_normalized_98_by_40_float32_sha256"]
    )

    model_input = np.ascontiguousarray(normalized.T, dtype=np.float32)
    assert model_input.shape == (40, 98)
    assert model_input.flags.c_contiguous
    assert (
        _sha256_float32(model_input)
        == golden["intermediates"]["transposed_C_contiguous_40_by_98_float32_sha256"]
    )


def test_zero_rms_and_nonfinite_mix_rules_are_fail_closed() -> None:
    zeros = np.zeros(16_000, dtype=np.float32)
    signal = np.ones(16_000, dtype=np.float32)

    with pytest.raises(ValueError, match="noise RMS"):
        _mix_at_snr(zeros, zeros, 10.0)

    mixed, command_rms, noise_rms, scale64, scale32 = _mix_at_snr(zeros, signal, 10.0)
    assert command_rms == np.float64(0.0)
    assert noise_rms == np.float64(1.0)
    assert scale64 == 0.0
    assert scale32.view(np.uint32).item() == 0
    assert not np.any(np.signbit(mixed))

    nonfinite = signal.copy()
    nonfinite[0] = np.float32(np.inf)
    with pytest.raises(ValueError, match="finite"):
        _mix_at_snr(nonfinite, signal, 10.0)


def test_edge_draw_goldens_fix_false_and_zero_width_control_flow() -> None:
    vectors = _config()["golden_vectors"]

    silence_identity = encode_window_identity(
        "_background_noise_/pink_noise.wav", 12_345
    )
    silence_common: DrawIdentity = {
        "seed": SEED,
        "epoch": 0,
        "identity": silence_identity,
    }
    silence_gain = uniform_real(
        **silence_common,
        domain="silence-gain-db",
        minimum=-6.0,
        maximum_exclusive=6.0,
    )
    silence_factor = np.float32(10.0 ** (silence_gain / 20.0))
    silence_time_width = uniform_integer(
        **silence_common, domain="silence-time-width", bound=11
    )
    silence_time_start = uniform_integer(
        **silence_common,
        domain="silence-time-start",
        bound=98 - silence_time_width + 1,
    )
    silence_mel_width = uniform_integer(
        **silence_common, domain="silence-mel-width", bound=5
    )
    silence_mel_start = uniform_integer(
        **silence_common,
        domain="silence-mel-start",
        bound=40 - silence_mel_width + 1,
    )
    silence = vectors["silence"]
    assert silence_gain.hex() == silence["gain_db_float64_hex"]
    assert _float32_bits(silence_factor) == silence["gain_factor_float32_bits"]
    assert (
        silence_time_start,
        silence_time_width,
        silence_mel_start,
        silence_mel_width,
    ) == (44, 7, 27, 3)

    zero_identity = encode_command_identity("yes/golden_nohash_54.wav")
    zero_common: DrawIdentity = {
        "seed": SEED,
        "epoch": 0,
        "identity": zero_identity,
    }
    zero_time_width = uniform_integer(
        **zero_common, domain="command-time-width", bound=11
    )
    zero_time_start = uniform_integer(
        **zero_common,
        domain="command-time-start",
        bound=98 - zero_time_width + 1,
    )
    zero_mel_width = uniform_integer(**zero_common, domain="command-mel-width", bound=5)
    zero_mel_start = uniform_integer(
        **zero_common,
        domain="command-mel-start",
        bound=40 - zero_mel_width + 1,
    )
    assert (
        (zero_time_start, zero_time_width, zero_mel_start, zero_mel_width)
        == (
            vectors["zero_width_masks"]["time_mask_start"],
            vectors["zero_width_masks"]["time_mask_width"],
            vectors["zero_width_masks"]["mel_mask_start"],
            vectors["zero_width_masks"]["mel_mask_width"],
        )
        == (20, 0, 39, 0)
    )

    false_identity = encode_command_identity("yes/golden_nohash_0.wav")
    false_common: DrawIdentity = {
        "seed": SEED,
        "epoch": 0,
        "identity": false_identity,
    }
    false_draw = uniform_real(
        **false_common,
        domain="command-noise-apply",
        minimum=0.0,
        maximum_exclusive=1.0,
    )
    assert false_draw.hex() == vectors["noise_not_applied"]["draw_float64_hex"]
    assert (
        bernoulli(
            **false_common,
            domain="command-noise-apply",
            probability=0.8,
        )
        is vectors["noise_not_applied"]["result"]
        is False
    )


def test_addendum_fixes_unaugmented_paths_and_is_documented_honestly() -> None:
    config = _config()
    unaugmented = config["feature_pipeline"]["unaugmented_paths"]
    assert unaugmented == {
        "quantization_calibration": (
            "no_shift_no_gain_no_noise_no_masks_and_no_augmentation_RNG"
        ),
        "training_normalization": (
            "all_registered_training_commands_no_shift_no_gain_no_noise_no_masks_"
            "and_no_augmentation_RNG"
        ),
        "validation": "no_shift_no_gain_no_noise_no_masks_and_no_augmentation_RNG",
    }
    assert config["feature_pipeline"]["masking"]["draw_order"].startswith(
        "width_then_start"
    )
    assert (
        "always_draw_start_even_when_width_is_zero"
        in (config["feature_pipeline"]["masking"]["time"])
    )
    assert config["runtime"] == {
        "floating_authority": (
            "CPython_3.12.3_float_operations_and_power_using_the_host_libm_plus_"
            "NumPy_2.5.1_explicit_dtype_operations_and_the_bound_source_bytes"
        ),
        "numpy": "2.5.1",
        "python": "3.12.3",
    }

    documentation = DOC_PATH.read_text(encoding="utf-8")
    normalized_documentation = " ".join(documentation.split())
    assert "configs/experiment-002-numerics.json" in documentation
    assert "after the model implementation" in normalized_documentation
    assert "before the normalization artifact" in normalized_documentation
    assert "streaming frontend is the numeric authority" in normalized_documentation
