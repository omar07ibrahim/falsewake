from __future__ import annotations

import ast
import copy
import hashlib
import math
import os
import pickle
import stat
import struct
import threading
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, cast

import numpy as np
import pytest

import falsewake.experiment_002_normalization_artifact as artifact
import falsewake.experiment_002_pcm_cache as pcm_cache
import falsewake.speech_commands_pcm as pcm
from falsewake.experiment_002_data import (
    BackgroundSource,
    CommandSource,
    Experiment002Corpus,
)
from falsewake.experiment_002_normalization import serialize_stats
from falsewake.experiment_002_pcm_cache import Experiment002PCMCache
from falsewake.experiment_002_preprocessing import unaugmented_log_mel
from falsewake.speech_commands_pcm import PCMSourceIdentity, VerifiedPCM16LE

ARTIFACT_BYTES: Final = 320
MANIFEST_INVENTORY_SHA256: Final = "2" * 64
ARCHIVE_SHA256: Final = "3" * 64


def _pcm(samples: list[int] | np.ndarray[tuple[int], np.dtype[np.int16]]) -> bytes:
    return np.asarray(samples, dtype="<i2").tobytes()


def _pattern(sample_count: int, *, token: int, step: int) -> bytes:
    values = (
        (np.arange(sample_count, dtype=np.int32) * step + token + 32_768) % 65_536
        - 32_768
    ).astype("<i2")
    values[0] = np.int16(token)
    if sample_count > 2:
        values[1] = np.int16(-32_768)
        values[2] = np.int16(32_767)
    return values.tobytes()


def _command(manifest_index: int, path: str, payload: bytes) -> CommandSource:
    word = path.split("/", maxsplit=1)[0]
    return CommandSource(
        manifest_index=manifest_index,
        path=path,
        word=word,
        label="unknown" if word == "cat" else word,
        sample_count=len(payload) // 2,
        sha256=hashlib.sha256(b"wav:" + path.encode()).hexdigest(),
    )


def _background(path: str, payload: bytes) -> BackgroundSource:
    return BackgroundSource(
        path=path,
        sample_count=len(payload) // 2,
        sha256=hashlib.sha256(b"wav:" + path.encode()).hexdigest(),
    )


class _TinyCorpus:
    def __init__(self) -> None:
        self.payloads: dict[str, bytes] = {
            "yes/z_nohash_0.wav": _pattern(7, token=303, step=11),
            "no/a_nohash_0.wav": _pattern(16_000, token=101, step=257),
            "cat/m_nohash_0.wav": _pattern(401, token=202, step=31),
            "up/validation_nohash_0.wav": _pattern(19, token=404, step=7),
            "_background_noise_/train.wav": _pattern(16_003, token=505, step=13),
            "_background_noise_/validation.wav": _pattern(16_005, token=606, step=17),
        }
        # Deliberately not manifest order: normalization must sort 1, 7, 12.
        train_commands = (
            _command(
                12,
                "yes/z_nohash_0.wav",
                self.payloads["yes/z_nohash_0.wav"],
            ),
            _command(
                1,
                "no/a_nohash_0.wav",
                self.payloads["no/a_nohash_0.wav"],
            ),
            _command(
                7,
                "cat/m_nohash_0.wav",
                self.payloads["cat/m_nohash_0.wav"],
            ),
        )
        validation_commands = (
            _command(
                20,
                "up/validation_nohash_0.wav",
                self.payloads["up/validation_nohash_0.wav"],
            ),
        )
        train_background = _background(
            "_background_noise_/train.wav",
            self.payloads["_background_noise_/train.wav"],
        )
        validation_background = _background(
            "_background_noise_/validation.wav",
            self.payloads["_background_noise_/validation.wav"],
        )
        self.corpus = Experiment002Corpus(
            train_commands=train_commands,
            validation_commands=validation_commands,
            train_backgrounds=(train_background,),
            validation_background=validation_background,
            # The firewall retains a count only; no test identity exists here.
            test_command_count=17,
            manifest_sha256="1" * 64,
        )
        self.loader_calls: list[tuple[str, str]] = []

    def loader_factory(self, dataset_root: Path) -> _FakeLoader:
        return _FakeLoader(dataset_root, self.payloads, self.loader_calls)


class _FakeLoader:
    def __init__(
        self,
        dataset_root: Path,
        payloads: Mapping[str, bytes],
        calls: list[tuple[str, str]],
    ) -> None:
        self.dataset_root = dataset_root
        self._payloads = payloads
        self._calls = calls
        self.closed = False

    def load_command(self, source: PCMSourceIdentity) -> VerifiedPCM16LE:
        return self._load("command", source)

    def load_background(self, source: PCMSourceIdentity) -> VerifiedPCM16LE:
        return self._load("background", source)

    def close(self) -> None:
        self.closed = True

    def _load(self, kind: str, source: PCMSourceIdentity) -> VerifiedPCM16LE:
        if self.closed:
            raise AssertionError("fake loader was closed")
        self._calls.append((kind, source.path))
        payload = self._payloads[source.path]
        return pcm._new_verified_pcm16le(
            path=source.path,
            sample_count=source.sample_count,
            sha256=source.sha256,
            payload=payload,
        )


@dataclass(slots=True)
class _OpenedTinyCache:
    synthetic: _TinyCorpus
    path: Path
    cache: Experiment002PCMCache


def _pcm_layout(synthetic: _TinyCorpus) -> pcm_cache._CacheLayout:
    return pcm_cache._CacheLayout.for_corpus(
        synthetic.corpus,
        manifest_inventory_sha256=MANIFEST_INVENTORY_SHA256,
        archive_sha256=ARCHIVE_SHA256,
    )


@pytest.fixture
def tiny_cache(tmp_path: Path) -> Iterator[_OpenedTinyCache]:
    synthetic = _TinyCorpus()
    path = tmp_path / "tiny-pcm.cache"
    identity = pcm_cache._build_pcm_cache(
        synthetic.corpus,
        tmp_path / "dataset",
        path,
        layout=_pcm_layout(synthetic),
        loader_factory=synthetic.loader_factory,
    )
    opened = Experiment002PCMCache._open(
        synthetic.corpus,
        path,
        identity,
        layout=_pcm_layout(synthetic),
    )
    try:
        yield _OpenedTinyCache(synthetic, path, opened)
    finally:
        opened.close()


def _normalization_layout(
    tiny: _OpenedTinyCache, *, frames_per_clip: int = 2
) -> artifact._NormalizationLayout:
    return artifact._NormalizationLayout.for_corpus(
        tiny.synthetic.corpus,
        frames_per_clip=frames_per_clip,
    )


def _artifact_identity(path: Path) -> artifact.NormalizationArtifactIdentity:
    contents = path.read_bytes()
    return artifact.NormalizationArtifactIdentity(
        byte_count=len(contents),
        sha256=hashlib.sha256(contents).hexdigest(),
    )


def _adversarial_frames(
    token: int,
) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
    rows = np.empty((2, 40), dtype=np.float32)
    offsets = np.arange(40, dtype=np.float32)
    if token == 101:
        rows[0] = np.float32(1.0e20)
        rows[1] = offsets + np.float32(1.25)
    elif token == 202:
        rows[0] = np.float32(-1.0e20)
        rows[1] = offsets + np.float32(2.5)
    elif token == 303:
        rows[0] = offsets + np.float32(3.75)
        rows[1] = offsets + np.float32(4.5)
    else:
        raise AssertionError(f"unexpected training token {token}")
    return rows


def _token(waveform: np.ndarray[tuple[int], np.dtype[np.float32]]) -> int:
    return int(round(float(waveform[0]) * 32_768.0))


def _independent_stats(
    matrices: list[np.ndarray[tuple[int, int], np.dtype[np.float32]]],
) -> bytes:
    sums = [0.0] * 40
    sums_sq = [0.0] * 40
    frame_count = 0
    for matrix in matrices:
        for frame in matrix:
            for mel_index in range(40):
                value = float(frame[mel_index])
                sums[mel_index] += value
                sums_sq[mel_index] += value * value
            frame_count += 1
    means64 = [value / frame_count for value in sums]
    deviations64 = [
        math.sqrt(max(0.0, sums_sq[index] / frame_count - mean * mean))
        for index, mean in enumerate(means64)
    ]
    return struct.pack(
        "<80f",
        *(np.float32(value) for value in (*means64, *deviations64)),
    )


def _compute_tiny(
    tiny: _OpenedTinyCache,
    *,
    feature_extractor: artifact._FeatureExtractor | None = None,
    layout: artifact._NormalizationLayout | None = None,
) -> artifact.VerifiedNormalization:
    return artifact._compute_normalization(
        tiny.synthetic.corpus,
        tiny.cache.training,
        layout=_normalization_layout(tiny) if layout is None else layout,
        feature_extractor=feature_extractor,
    )


@contextmanager
def _loaded_tiny(
    path: Path,
    expected: artifact.NormalizationArtifactIdentity,
    layout: artifact._NormalizationLayout,
) -> Iterator[artifact.VerifiedNormalization]:
    yield artifact._load_normalization_artifact(
        path,
        expected,
        layout=layout,
    )


def test_compute_uses_canonical_manifest_order_and_exact_scalar_float64_oracle(
    tiny_cache: _OpenedTinyCache,
) -> None:
    observed_tokens: list[int] = []

    def extract(
        waveform: np.ndarray[tuple[int], np.dtype[np.float32]],
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
        token = _token(waveform)
        observed_tokens.append(token)
        return _adversarial_frames(token)

    computed = _compute_tiny(tiny_cache, feature_extractor=extract)
    expected_matrices = [
        _adversarial_frames(101),
        _adversarial_frames(202),
        _adversarial_frames(303),
    ]
    expected_contents = _independent_stats(expected_matrices)

    assert observed_tokens == [101, 202, 303]
    assert computed.clip_count == 3
    assert computed.frame_count == 6
    assert len(computed.contents) == ARTIFACT_BYTES
    assert computed.contents == expected_contents
    assert serialize_stats(computed.stats) == expected_contents
    assert hashlib.sha256(expected_contents).hexdigest() == (
        "c36e74643bbc7097571e0a86a744f12acb83b661f495151c627e6ebea0d881c6"
    )
    assert computed.identity == artifact.NormalizationArtifactIdentity(
        byte_count=ARTIFACT_BYTES,
        sha256=hashlib.sha256(expected_contents).hexdigest(),
    )
    assert computed.stats.mel_bins == 40


def test_registered_constants_and_static_isolation_contract() -> None:
    assert artifact.MEL_BINS == 40
    assert artifact.FRAMES_PER_CLIP == 98
    assert artifact.REGISTERED_CLIP_COUNT == 84_843
    assert artifact.REGISTERED_FRAME_COUNT == 84_843 * 98 == 8_314_614
    assert artifact.NORMALIZATION_ARTIFACT_BYTES == ARTIFACT_BYTES

    module_path = Path(artifact.__file__)
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    imported_modules: set[str] = set()
    called_attributes: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name)
                imported_roots.add(alias.name.split(".", maxsplit=1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)
            imported_roots.add(node.module.split(".", maxsplit=1)[0])
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            called_attributes.add(node.func.attr)

    assert imported_roots.isdisjoint(
        {"glob", "mmap", "onnx", "onnxruntime", "random", "secrets", "torch"}
    )
    assert "falsewake.experiment_002_rng" not in imported_modules
    assert "falsewake.experiment_002_preprocessing" not in imported_modules
    assert called_attributes.isdisjoint(
        {"default_rng", "glob", "random", "rglob", "walk"}
    )
    assert "os.replace" not in source
    assert "white_noise.wav" not in source
    assert "test_commands" not in source


def test_authoritative_frontend_pcm_endpoints_padding_and_independent_oracle(
    tiny_cache: _OpenedTinyCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_sources: list[CommandSource] = []
    observed_waveforms: dict[int, np.ndarray[tuple[int], np.dtype[np.float32]]] = {}
    original_read = pcm_cache.PCMCacheSplitView.read_command_pcm16le

    def read_training_only(
        view: pcm_cache.PCMCacheSplitView,
        source: CommandSource,
    ) -> bytes:
        assert view.split == "train"
        assert source in tiny_cache.synthetic.corpus._train_command_sources
        observed_sources.append(source)
        return original_read(view, source)

    def forbid_background(
        view: pcm_cache.PCMCacheSplitView,
        source: BackgroundSource,
        start_sample: int,
    ) -> bytes:
        raise AssertionError(
            "normalization attempted a background read: "
            f"{view.split}/{source.path}/{start_sample}"
        )

    def extract(
        waveform: np.ndarray[tuple[int], np.dtype[np.float32]],
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
        observed_waveforms[_token(waveform)] = waveform.copy()
        return unaugmented_log_mel(waveform)

    monkeypatch.setattr(
        pcm_cache.PCMCacheSplitView,
        "read_command_pcm16le",
        read_training_only,
    )
    monkeypatch.setattr(
        pcm_cache.PCMCacheSplitView,
        "read_background_window_pcm16le",
        forbid_background,
    )
    layout = _normalization_layout(tiny_cache, frames_per_clip=98)
    computed = _compute_tiny(
        tiny_cache,
        feature_extractor=extract,
        layout=layout,
    )

    ordered = sorted(
        tiny_cache.synthetic.corpus.train_commands,
        key=lambda source: source.manifest_index,
    )
    assert observed_sources == ordered
    matrices = [
        unaugmented_log_mel(observed_waveforms[token]) for token in (101, 202, 303)
    ]
    assert computed.contents == _independent_stats(matrices)
    assert computed.frame_count == 3 * 98

    full = observed_waveforms[101]
    assert full.shape == (16_000,)
    assert full.dtype == np.dtype(np.float32)
    assert full.flags.c_contiguous
    assert full[0] == np.float32(101.0 / 32_768.0)
    assert full[1] == np.float32(-1.0)
    assert full[2] == np.float32(32_767.0 / 32_768.0)

    sample_counts = {101: 16_000, 202: 401, 303: 7}
    for token, sample_count in sample_counts.items():
        waveform = observed_waveforms[token]
        assert waveform.shape == (16_000,)
        if sample_count < 16_000:
            padding = waveform[sample_count:]
            assert np.all(padding == np.float32(0.0))
            assert not np.any(np.signbit(padding))


def test_two_computations_are_byte_identical_and_do_not_share_stats_arrays(
    tiny_cache: _OpenedTinyCache,
) -> None:
    def extract(
        waveform: np.ndarray[tuple[int], np.dtype[np.float32]],
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
        return _adversarial_frames(_token(waveform))

    first = _compute_tiny(tiny_cache, feature_extractor=extract)
    second = _compute_tiny(tiny_cache, feature_extractor=extract)

    assert first.contents == second.contents
    assert first.identity == second.identity
    assert first.clip_count == second.clip_count
    assert first.frame_count == second.frame_count
    assert not np.shares_memory(first.stats.means, second.stats.means)
    assert not np.shares_memory(
        first.stats.standard_deviations,
        second.stats.standard_deviations,
    )


def test_private_compute_requires_concrete_training_view_and_exact_layout(
    tiny_cache: _OpenedTinyCache,
) -> None:
    layout = _normalization_layout(tiny_cache)

    with pytest.raises(artifact.Experiment002NormalizationArtifactError, match="train"):
        artifact._compute_normalization(
            tiny_cache.synthetic.corpus,
            tiny_cache.cache.validation,
            layout=layout,
            feature_extractor=lambda waveform: _adversarial_frames(_token(waveform)),
        )
    with pytest.raises(TypeError, match="PCMCacheSplitView|cache|training"):
        artifact._compute_normalization(
            tiny_cache.synthetic.corpus,
            cast(pcm_cache.PCMCacheSplitView, object()),
            layout=layout,
            feature_extractor=lambda waveform: _adversarial_frames(_token(waveform)),
        )

    class SplitViewSubclass(pcm_cache.PCMCacheSplitView):
        pass

    with pytest.raises(TypeError, match="PCMCacheSplitView|cache|training"):
        artifact._compute_normalization(
            tiny_cache.synthetic.corpus,
            SplitViewSubclass(tiny_cache.cache, "train"),
            layout=layout,
            feature_extractor=lambda waveform: _adversarial_frames(_token(waveform)),
        )
    mismatched = replace(layout, clip_count=layout.clip_count + 1)
    with pytest.raises(
        artifact.Experiment002NormalizationArtifactError,
        match="clip|corpus|layout",
    ):
        artifact._compute_normalization(
            tiny_cache.synthetic.corpus,
            tiny_cache.cache.training,
            layout=mismatched,
            feature_extractor=lambda waveform: _adversarial_frames(_token(waveform)),
        )


@pytest.mark.parametrize(
    "invalid",
    [
        np.zeros((2, 40), dtype=np.float64),
        np.zeros((1, 40), dtype=np.float32),
        np.zeros((2, 39), dtype=np.float32),
        np.zeros((2, 40), dtype=np.float32)[:, ::-1],
        np.full((2, 40), np.nan, dtype=np.float32),
    ],
)
def test_feature_extractor_contract_fails_closed(
    tiny_cache: _OpenedTinyCache,
    invalid: np.ndarray[tuple[int, ...], np.dtype[np.floating]],
) -> None:
    def extract(
        waveform: np.ndarray[tuple[int], np.dtype[np.float32]],
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.floating]]:
        return invalid

    with pytest.raises(
        (TypeError, artifact.Experiment002NormalizationArtifactError),
        match="feature|float32|shape|contiguous|finite",
    ):
        _compute_tiny(
            tiny_cache,
            feature_extractor=cast(artifact._FeatureExtractor, extract),
        )


def test_zero_variance_from_extractor_is_rejected(
    tiny_cache: _OpenedTinyCache,
) -> None:
    constant = np.ones((2, 40), dtype=np.float32)
    with pytest.raises(ValueError, match="standard deviation is zero"):
        _compute_tiny(
            tiny_cache,
            feature_extractor=lambda waveform: constant,
        )


def test_public_compute_rejects_nonregistered_corpus_before_any_pcm_read(
    tiny_cache: _OpenedTinyCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> bytes:
        raise AssertionError("public corpus firewall did not run before PCM access")

    monkeypatch.setattr(
        pcm_cache.PCMCacheSplitView,
        "read_command_pcm16le",
        forbidden,
    )

    with pytest.raises(
        artifact.Experiment002NormalizationArtifactError,
        match="registered|corpus|population",
    ):
        artifact.compute_registered_normalization(
            tiny_cache.synthetic.corpus,
            tiny_cache.cache.training,
        )


def test_verified_capability_is_issuer_only_and_reparses_fresh_stats(
    tmp_path: Path,
    tiny_cache: _OpenedTinyCache,
) -> None:
    with pytest.raises(TypeError):
        artifact.VerifiedNormalization()

    computed = _compute_tiny(
        tiny_cache,
        feature_extractor=lambda waveform: _adversarial_frames(_token(waveform)),
    )
    original_contents = computed.contents
    original_identity = computed.identity
    exposed = computed.stats
    exposed.means.flags.writeable = True
    exposed.standard_deviations.flags.writeable = True
    exposed.means[:] = np.float32(-123.0)
    exposed.standard_deviations[:] = np.float32(999.0)

    assert computed.contents == original_contents
    assert computed.identity == original_identity
    assert serialize_stats(computed.stats) == original_contents
    assert not np.shares_memory(exposed.means, computed.stats.means)

    destination = tmp_path / "immutable.stats"
    published = artifact._publish_normalization_artifact(
        computed,
        destination,
        layout=_normalization_layout(tiny_cache),
    )
    assert published == original_identity
    assert destination.read_bytes() == original_contents


def test_publish_rejects_arbitrary_forged_object_before_filesystem_write(
    tmp_path: Path,
    tiny_cache: _OpenedTinyCache,
) -> None:
    destination = tmp_path / "forged.stats"
    with pytest.raises(TypeError, match="VerifiedNormalization|result"):
        artifact._publish_normalization_artifact(
            cast(artifact.VerifiedNormalization, object()),
            destination,
            layout=_normalization_layout(tiny_cache),
        )
    assert not destination.exists()


def test_verifier_rejects_copied_and_manually_forged_capabilities(
    tmp_path: Path,
    tiny_cache: _OpenedTinyCache,
) -> None:
    genuine = _compute_tiny(
        tiny_cache,
        feature_extractor=lambda waveform: _adversarial_frames(_token(waveform)),
    )
    copied = copy.copy(genuine)
    deep_copied = copy.deepcopy(genuine)
    unpickled = pickle.loads(pickle.dumps(genuine))
    forged = object.__new__(artifact.VerifiedNormalization)
    object.__setattr__(forged, "_contents", genuine.contents)
    object.__setattr__(forged, "_identity", genuine.identity)
    object.__setattr__(forged, "_clip_count", genuine.clip_count)
    object.__setattr__(forged, "_frame_count", genuine.frame_count)
    layout = _normalization_layout(tiny_cache)

    # The registry must use object identity: equal field values are not issuance.
    assert copied is not genuine
    for candidate in (copied, deep_copied, unpickled, forged):
        with pytest.raises(
            artifact.Experiment002NormalizationArtifactError,
            match="issued|capability|verified",
        ):
            artifact._revalidate_capability(candidate, layout)
        destination = tmp_path / f"forged-{id(candidate)}.stats"
        with pytest.raises(
            artifact.Experiment002NormalizationArtifactError,
            match="issued|capability|verified",
        ):
            artifact._publish_normalization_artifact(
                candidate,
                destination,
                layout=layout,
            )
        assert not destination.exists()

    # Genuine compute issuance still passes the same verifier and publisher.
    artifact._revalidate_capability(genuine, layout)
    genuine_path = tmp_path / "genuine.stats"
    assert _publish_tiny(genuine, genuine_path, tiny_cache) == genuine.identity
    loaded = artifact._load_normalization_artifact(
        genuine_path,
        genuine.identity,
        layout=layout,
    )
    artifact._revalidate_capability(loaded, layout)


def test_exact_type_guard_rejects_weak_registry_equality_collision(
    tmp_path: Path,
    tiny_cache: _OpenedTinyCache,
) -> None:
    genuine = _compute_tiny(
        tiny_cache,
        feature_extractor=lambda waveform: _adversarial_frames(_token(waveform)),
    )

    class CollidingSubclass(artifact.VerifiedNormalization):
        _target: artifact.VerifiedNormalization

        def __init__(self, target: artifact.VerifiedNormalization) -> None:
            object.__setattr__(self, "_target", target)
            object.__setattr__(self, "_contents", target.contents)
            object.__setattr__(self, "_identity", target.identity)
            object.__setattr__(self, "_clip_count", target.clip_count)
            object.__setattr__(self, "_frame_count", target.frame_count)

        def __hash__(self) -> int:
            return hash(self._target)

        def __eq__(self, other: object) -> bool:
            return other is self._target

    colliding = CollidingSubclass(genuine)
    layout = _normalization_layout(tiny_cache)

    with pytest.raises(TypeError, match="VerifiedNormalization"):
        artifact._revalidate_capability(colliding, layout)
    with pytest.raises(TypeError, match="VerifiedNormalization"):
        _ = colliding.stats
    destination = tmp_path / "colliding-subclass.stats"
    with pytest.raises(TypeError, match="VerifiedNormalization"):
        artifact._publish_normalization_artifact(
            colliding,
            destination,
            layout=layout,
        )
    assert not destination.exists()


def test_capability_revalidation_rejects_injected_integer_subclasses(
    tiny_cache: _OpenedTinyCache,
) -> None:
    class LyingInteger(int):
        def __ne__(self, other: object) -> bool:
            return False

    clip_mutated = _compute_tiny(
        tiny_cache,
        feature_extractor=lambda waveform: _adversarial_frames(_token(waveform)),
    )
    frame_mutated = _compute_tiny(
        tiny_cache,
        feature_extractor=lambda waveform: _adversarial_frames(_token(waveform)),
    )
    object.__setattr__(
        clip_mutated,
        "_clip_count",
        LyingInteger(clip_mutated.clip_count + 1),
    )
    object.__setattr__(
        frame_mutated,
        "_frame_count",
        LyingInteger(frame_mutated.frame_count + 1),
    )

    layout = _normalization_layout(tiny_cache)
    for candidate in (clip_mutated, frame_mutated):
        with pytest.raises(
            artifact.Experiment002NormalizationArtifactError,
            match="state changed",
        ):
            artifact._revalidate_capability(candidate, layout)


def test_public_verifier_accepts_only_genuine_registered_issuance(
    tiny_cache: _OpenedTinyCache,
) -> None:
    tiny = _compute_tiny(
        tiny_cache,
        feature_extractor=lambda waveform: _adversarial_frames(_token(waveform)),
    )
    registered = artifact._issue_verified_normalization(
        tiny.contents,
        tiny.identity,
        artifact._REGISTERED_LAYOUT,
    )
    artifact.verify_registered_normalization(registered)

    copied = copy.copy(registered)
    with pytest.raises(
        artifact.Experiment002NormalizationArtifactError,
        match="issued|capability|verified",
    ):
        artifact.verify_registered_normalization(copied)


@pytest.fixture
def verified(tiny_cache: _OpenedTinyCache) -> artifact.VerifiedNormalization:
    return _compute_tiny(
        tiny_cache,
        feature_extractor=lambda waveform: _adversarial_frames(_token(waveform)),
    )


def _publish_tiny(
    result: artifact.VerifiedNormalization,
    destination: Path,
    tiny: _OpenedTinyCache,
) -> artifact.NormalizationArtifactIdentity:
    return artifact._publish_normalization_artifact(
        result,
        destination,
        layout=_normalization_layout(tiny),
    )


def test_publish_and_load_preserve_exact_320_bytes_digest_counts_and_mode(
    tmp_path: Path,
    tiny_cache: _OpenedTinyCache,
    verified: artifact.VerifiedNormalization,
) -> None:
    first_path = tmp_path / "first.stats"
    second_path = tmp_path / "second.stats"

    first_identity = _publish_tiny(verified, first_path, tiny_cache)
    second_identity = _publish_tiny(verified, second_path, tiny_cache)

    assert first_identity == second_identity == verified.identity
    assert first_path.read_bytes() == second_path.read_bytes() == verified.contents
    assert len(verified.contents) == ARTIFACT_BYTES
    assert hashlib.sha256(verified.contents).hexdigest() == verified.identity.sha256
    assert stat.S_IMODE(first_path.stat().st_mode) & 0o222 == 0
    assert stat.S_IMODE(second_path.stat().st_mode) & 0o222 == 0

    loaded = artifact._load_normalization_artifact(
        first_path,
        first_identity,
        layout=_normalization_layout(tiny_cache),
    )
    assert loaded.contents == verified.contents
    assert loaded.identity == verified.identity
    assert loaded.clip_count == verified.clip_count == 3
    assert loaded.frame_count == verified.frame_count == 6
    assert serialize_stats(loaded.stats) == verified.contents


def test_load_requires_exact_external_digest_and_does_not_trust_file_contents(
    tmp_path: Path,
    tiny_cache: _OpenedTinyCache,
    verified: artifact.VerifiedNormalization,
) -> None:
    path = tmp_path / "external.stats"
    _publish_tiny(verified, path, tiny_cache)
    wrong = artifact.NormalizationArtifactIdentity(
        byte_count=ARTIFACT_BYTES,
        sha256="0" * 64 if verified.identity.sha256 != "0" * 64 else "1" * 64,
    )

    with pytest.raises(
        artifact.Experiment002NormalizationArtifactError,
        match="external|digest|identity",
    ):
        artifact._load_normalization_artifact(
            path,
            wrong,
            layout=_normalization_layout(tiny_cache),
        )

    path.chmod(0o600)
    corrupted = bytearray(path.read_bytes())
    corrupted[0] ^= 1
    path.write_bytes(corrupted)
    with pytest.raises(
        artifact.Experiment002NormalizationArtifactError,
        match="external|digest|identity",
    ):
        artifact._load_normalization_artifact(
            path,
            verified.identity,
            layout=_normalization_layout(tiny_cache),
        )


def test_external_identity_is_exact_type_snapshotted_and_never_retained(
    tmp_path: Path,
    tiny_cache: _OpenedTinyCache,
    verified: artifact.VerifiedNormalization,
) -> None:
    path = tmp_path / "identity-snapshot.stats"
    published = _publish_tiny(verified, path, tiny_cache)
    expected_sha256 = published.sha256

    class IdentitySubclass(artifact.NormalizationArtifactIdentity):
        pass

    subclassed = IdentitySubclass(
        byte_count=published.byte_count,
        sha256=published.sha256,
    )
    with pytest.raises(TypeError, match="NormalizationArtifactIdentity"):
        artifact._load_normalization_artifact(
            path,
            subclassed,
            layout=_normalization_layout(tiny_cache),
        )

    loaded = artifact._load_normalization_artifact(
        path,
        published,
        layout=_normalization_layout(tiny_cache),
    )
    object.__setattr__(published, "sha256", "f" * 64)
    assert loaded.identity.sha256 == expected_sha256
    artifact._revalidate_capability(loaded, _normalization_layout(tiny_cache))

    exposed = loaded.identity
    object.__setattr__(exposed, "sha256", "e" * 64)
    assert loaded.identity.sha256 == expected_sha256
    assert hashlib.sha256(loaded.contents).hexdigest() == expected_sha256
    artifact._revalidate_capability(loaded, _normalization_layout(tiny_cache))


@pytest.mark.parametrize(
    ("byte_count", "sha256", "exception"),
    [
        (-1, "0" * 64, artifact.Experiment002NormalizationArtifactError),
        (0, "0" * 64, artifact.Experiment002NormalizationArtifactError),
        (True, "0" * 64, TypeError),
        (
            0x1_0000_0000_0000_0000,
            "0" * 64,
            artifact.Experiment002NormalizationArtifactError,
        ),
        (ARTIFACT_BYTES, "0" * 63, artifact.Experiment002NormalizationArtifactError),
        (ARTIFACT_BYTES, "A" * 64, artifact.Experiment002NormalizationArtifactError),
    ],
)
def test_external_identity_fields_are_strict(
    byte_count: int,
    sha256: str,
    exception: type[BaseException],
) -> None:
    with pytest.raises(exception):
        artifact.NormalizationArtifactIdentity(byte_count, sha256)


def test_external_identity_rejects_scalar_subclasses_and_injected_fields() -> None:
    class IntegerSubclass(int):
        pass

    class StringSubclass(str):
        pass

    with pytest.raises(TypeError, match="byte_count.*integer"):
        artifact.NormalizationArtifactIdentity(
            byte_count=IntegerSubclass(ARTIFACT_BYTES),
            sha256="0" * 64,
        )
    with pytest.raises(TypeError, match="sha256.*string"):
        artifact.NormalizationArtifactIdentity(
            byte_count=ARTIFACT_BYTES,
            sha256=StringSubclass("0" * 64),
        )

    identity = artifact.NormalizationArtifactIdentity(
        byte_count=ARTIFACT_BYTES,
        sha256="0" * 64,
    )
    object.__setattr__(identity, "byte_count", IntegerSubclass(ARTIFACT_BYTES))
    with pytest.raises(TypeError, match="byte_count.*integer"):
        artifact._snapshot_identity(identity)


@pytest.mark.parametrize("claimed_bytes", [ARTIFACT_BYTES - 1, ARTIFACT_BYTES + 1])
def test_load_rejects_external_byte_count_before_open(
    tmp_path: Path,
    tiny_cache: _OpenedTinyCache,
    verified: artifact.VerifiedNormalization,
    claimed_bytes: int,
) -> None:
    path = tmp_path / "byte-count.stats"
    _publish_tiny(verified, path, tiny_cache)
    wrong = artifact.NormalizationArtifactIdentity(
        claimed_bytes,
        verified.identity.sha256,
    )

    with pytest.raises(
        artifact.Experiment002NormalizationArtifactError, match="external|byte"
    ):
        artifact._load_normalization_artifact(
            path,
            wrong,
            layout=_normalization_layout(tiny_cache),
        )


def test_load_handles_partial_pread_and_detects_inflight_mutation(
    tmp_path: Path,
    tiny_cache: _OpenedTinyCache,
    verified: artifact.VerifiedNormalization,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "partial-read.stats"
    _publish_tiny(verified, path, tiny_cache)
    original_pread = os.pread
    partial_calls = 0

    def partial_pread(descriptor: int, length: int, offset: int) -> bytes:
        nonlocal partial_calls
        if length > 1:
            partial_calls += 1
            length = max(1, length // 2)
        return original_pread(descriptor, length, offset)

    monkeypatch.setattr(os, "pread", partial_pread)
    loaded = artifact._load_normalization_artifact(
        path,
        verified.identity,
        layout=_normalization_layout(tiny_cache),
    )
    assert loaded.contents == verified.contents
    assert partial_calls > 0

    mutated = False
    issued_after_mutation = 0
    original_issue = artifact._issue_verified_normalization

    def record_issue(
        contents: bytes,
        identity: artifact.NormalizationArtifactIdentity,
        layout: artifact._NormalizationLayout,
    ) -> artifact.VerifiedNormalization:
        nonlocal issued_after_mutation
        issued_after_mutation += 1
        return original_issue(contents, identity, layout)

    def mutate_after_read(descriptor: int, length: int, offset: int) -> bytes:
        nonlocal mutated
        result = original_pread(descriptor, length, offset)
        if not mutated and offset == 0 and result:
            mutated = True
            status = path.stat()
            os.utime(
                path,
                ns=(status.st_atime_ns, status.st_mtime_ns + 1_000_000_000),
            )
        return result

    monkeypatch.setattr(os, "pread", mutate_after_read)
    monkeypatch.setattr(artifact, "_issue_verified_normalization", record_issue)
    with pytest.raises(
        artifact.Experiment002NormalizationArtifactError, match="changed"
    ):
        artifact._load_normalization_artifact(
            path,
            verified.identity,
            layout=_normalization_layout(tiny_cache),
        )
    assert mutated
    assert issued_after_mutation == 0


@pytest.mark.parametrize("change", ["truncate", "append"])
def test_load_rejects_truncation_and_trailing_bytes(
    tmp_path: Path,
    tiny_cache: _OpenedTinyCache,
    verified: artifact.VerifiedNormalization,
    change: str,
) -> None:
    path = tmp_path / "length.stats"
    contents = (
        verified.contents[:-1] if change == "truncate" else verified.contents + b"\0"
    )
    path.write_bytes(contents)

    with pytest.raises(
        artifact.Experiment002NormalizationArtifactError,
        match="320|byte|size|length",
    ):
        artifact._load_normalization_artifact(
            path,
            verified.identity,
            layout=_normalization_layout(tiny_cache),
        )


@pytest.mark.parametrize("kind", ["zero", "negative", "infinite", "nan"])
def test_load_rejects_invalid_statistics_even_with_matching_external_identity(
    tmp_path: Path,
    tiny_cache: _OpenedTinyCache,
    kind: str,
) -> None:
    values = np.concatenate(
        (np.zeros(40, dtype=np.float32), np.ones(40, dtype=np.float32))
    )
    if kind == "zero":
        values[40] = np.float32(0.0)
    elif kind == "negative":
        values[40] = np.float32(-1.0)
    elif kind == "infinite":
        values[40] = np.float32(np.inf)
    else:
        values[0] = np.float32(np.nan)
    path = tmp_path / f"invalid-{kind}.stats"
    path.write_bytes(values.astype("<f4", copy=False).tobytes())

    with pytest.raises(
        artifact.Experiment002NormalizationArtifactError,
        match="finite|positive|statistics|artifact",
    ):
        artifact._load_normalization_artifact(
            path,
            _artifact_identity(path),
            layout=_normalization_layout(tiny_cache),
        )


def test_load_rejects_symlink_fifo_and_directory_without_blocking(
    tmp_path: Path,
    tiny_cache: _OpenedTinyCache,
    verified: artifact.VerifiedNormalization,
) -> None:
    regular = tmp_path / "regular.stats"
    _publish_tiny(verified, regular, tiny_cache)
    symlink = tmp_path / "link.stats"
    symlink.symlink_to(regular)
    fifo = tmp_path / "fifo.stats"
    os.mkfifo(fifo)
    directory = tmp_path / "directory.stats"
    directory.mkdir()

    for candidate in (symlink, fifo, directory):
        with pytest.raises(artifact.Experiment002NormalizationArtifactError):
            artifact._load_normalization_artifact(
                candidate,
                verified.identity,
                layout=_normalization_layout(tiny_cache),
            )


def test_existing_destination_and_nonregular_names_are_never_overwritten(
    tmp_path: Path,
    tiny_cache: _OpenedTinyCache,
    verified: artifact.VerifiedNormalization,
) -> None:
    existing = tmp_path / "existing.stats"
    sentinel = b"owned by another process"
    existing.write_bytes(sentinel)
    target = tmp_path / "target"
    target.write_bytes(b"target")
    symlink = tmp_path / "symlink"
    symlink.symlink_to(target)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    directory = tmp_path / "directory"
    directory.mkdir()

    for destination in (existing, symlink, fifo, directory):
        with pytest.raises(
            artifact.Experiment002NormalizationArtifactError, match="exist"
        ):
            _publish_tiny(verified, destination, tiny_cache)

    assert existing.read_bytes() == sentinel
    assert target.read_bytes() == b"target"
    assert symlink.is_symlink()
    assert stat.S_ISFIFO(fifo.stat().st_mode)
    assert directory.is_dir()


def test_symlinked_parent_is_rejected_for_publish_and_load(
    tmp_path: Path,
    tiny_cache: _OpenedTinyCache,
    verified: artifact.VerifiedNormalization,
) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(
        artifact.Experiment002NormalizationArtifactError, match="parent|secure"
    ):
        _publish_tiny(verified, linked_parent / "artifact.stats", tiny_cache)
    assert list(real_parent.iterdir()) == []

    regular = real_parent / "artifact.stats"
    _publish_tiny(verified, regular, tiny_cache)
    with pytest.raises(
        artifact.Experiment002NormalizationArtifactError, match="parent|secure"
    ):
        artifact._load_normalization_artifact(
            linked_parent / "artifact.stats",
            verified.identity,
            layout=_normalization_layout(tiny_cache),
        )


def test_concurrent_no_overwrite_publish_has_one_winner(
    tmp_path: Path,
    tiny_cache: _OpenedTinyCache,
    verified: artifact.VerifiedNormalization,
) -> None:
    destination = tmp_path / "contended.stats"
    barrier = threading.Barrier(2)

    def publish() -> artifact.NormalizationArtifactIdentity | BaseException:
        barrier.wait(timeout=10)
        try:
            return _publish_tiny(verified, destination, tiny_cache)
        except BaseException as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: publish(), range(2)))

    identities = [
        result
        for result in results
        if isinstance(result, artifact.NormalizationArtifactIdentity)
    ]
    failures = [result for result in results if isinstance(result, BaseException)]
    assert identities == [verified.identity]
    assert len(failures) == 1
    assert isinstance(failures[0], artifact.Experiment002NormalizationArtifactError)
    assert destination.read_bytes() == verified.contents
    assert [path for path in tmp_path.iterdir() if path != tiny_cache.path] == [
        destination
    ]


def test_publish_handles_partial_pwrite_results(
    tmp_path: Path,
    tiny_cache: _OpenedTinyCache,
    verified: artifact.VerifiedNormalization,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "partial.stats"
    original_pwrite = os.pwrite
    partial_calls = 0

    def partial_pwrite(
        descriptor: int,
        data: bytes | bytearray | memoryview,
        offset: int,
    ) -> int:
        nonlocal partial_calls
        view = memoryview(data)
        if len(view) > 1:
            partial_calls += 1
            view = view[: max(1, len(view) // 2)]
        return original_pwrite(descriptor, view, offset)

    monkeypatch.setattr(os, "pwrite", partial_pwrite)

    identity = _publish_tiny(verified, destination, tiny_cache)

    assert partial_calls > 0
    assert identity == verified.identity
    assert destination.read_bytes() == verified.contents


@pytest.mark.parametrize("operation", ["pwrite", "fsync", "link"])
def test_prelink_publish_failures_clean_partial_files(
    tmp_path: Path,
    tiny_cache: _OpenedTinyCache,
    verified: artifact.VerifiedNormalization,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    destination = tmp_path / f"failed-{operation}.stats"
    if operation == "pwrite":

        def fail_pwrite(
            descriptor: int,
            data: bytes | bytearray | memoryview,
            offset: int,
        ) -> int:
            raise OSError("injected pwrite failure")

        monkeypatch.setattr(os, "pwrite", fail_pwrite)
    elif operation == "fsync":

        def fail_fsync(descriptor: int) -> None:
            raise OSError("injected file fsync failure")

        monkeypatch.setattr(os, "fsync", fail_fsync)
    else:

        def fail_link(*args: object, **kwargs: object) -> None:
            raise OSError("injected link failure")

        monkeypatch.setattr(os, "link", fail_link)

    with pytest.raises(artifact.Experiment002NormalizationArtifactError):
        _publish_tiny(verified, destination, tiny_cache)

    assert not destination.exists()
    assert [path for path in tmp_path.iterdir() if path != tiny_cache.path] == []


def test_parent_fsync_failure_preserves_committed_artifact_and_cleans_temp(
    tmp_path: Path,
    tiny_cache: _OpenedTinyCache,
    verified: artifact.VerifiedNormalization,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "parent-fsync.stats"
    original_fsync = os.fsync
    calls = 0

    def fail_parent_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("injected parent fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_parent_fsync)

    with pytest.raises(artifact.Experiment002NormalizationArtifactError):
        _publish_tiny(verified, destination, tiny_cache)

    assert calls >= 2
    assert destination.read_bytes() == verified.contents
    assert [path for path in tmp_path.iterdir() if path != tiny_cache.path] == [
        destination
    ]


def test_temporary_unlink_failure_after_link_preserves_committed_artifact(
    tmp_path: Path,
    tiny_cache: _OpenedTinyCache,
    verified: artifact.VerifiedNormalization,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "unlink-failure.stats"
    original_unlink = os.unlink
    failed_once = False

    def fail_temp_once(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        dir_fd: int | None = None,
    ) -> None:
        nonlocal failed_once
        if not failed_once and os.fsdecode(path).endswith(".tmp"):
            failed_once = True
            raise OSError("injected temporary unlink failure")
        original_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(os, "unlink", fail_temp_once)

    with pytest.raises(artifact.Experiment002NormalizationArtifactError):
        _publish_tiny(verified, destination, tiny_cache)

    assert failed_once
    assert destination.read_bytes() == verified.contents
    assert [path for path in tmp_path.iterdir() if path != tiny_cache.path] == [
        destination
    ]


def test_destination_replacement_after_link_is_detected_without_rollback(
    tmp_path: Path,
    tiny_cache: _OpenedTinyCache,
    verified: artifact.VerifiedNormalization,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "raced.stats"
    held = tmp_path / "attacker-held.stats"
    sentinel = b"concurrent replacement"
    original_link = os.link

    def replace_after_link(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        destination_name: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        original_link(
            source,
            destination_name,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )
        if dst_dir_fd is None:
            raise AssertionError("publisher did not pin the parent directory")
        os.rename(
            destination_name,
            held.name,
            src_dir_fd=dst_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )
        descriptor = os.open(
            destination.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o600,
            dir_fd=dst_dir_fd,
        )
        try:
            assert os.write(descriptor, sentinel) == len(sentinel)
        finally:
            os.close(descriptor)

    monkeypatch.setattr(os, "link", replace_after_link)

    with pytest.raises(
        artifact.Experiment002NormalizationArtifactError,
        match="published|inode|name",
    ):
        _publish_tiny(verified, destination, tiny_cache)

    assert destination.read_bytes() == sentinel
    assert held.read_bytes() == verified.contents
    assert sorted(
        path.name for path in tmp_path.iterdir() if path != tiny_cache.path
    ) == sorted([destination.name, held.name])
