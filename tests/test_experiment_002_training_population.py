from __future__ import annotations

import copy
import gc
import hashlib
import json
import pickle
import struct
import weakref
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import falsewake.experiment_002_training_population as training
from falsewake.experiment_002_data import (
    CLASS_ORDER,
    CommandExample,
    CommandSource,
)
from falsewake.experiment_002_pcm_cache import _IndexEntry
from falsewake.experiment_002_training_population import (
    CompletedTrainingPopulation,
    CompletedTrainingUpdate,
    Experiment002TrainingPopulationError,
    MaterializedTrainingBatch,
    RegisteredTrainingEpoch,
    RegisteredTrainingInputSource,
)
from falsewake.features import FloatArray


def _example(ordinal: int, *, label: str | None = None) -> CommandExample:
    selected_label = CLASS_ORDER[ordinal % 10] if label is None else label
    source = CommandSource(
        manifest_index=ordinal,
        path=f"{selected_label}/speaker_{ordinal:05d}_nohash_0.wav",
        word=selected_label,
        label=selected_label,
        sample_count=1,
        sha256=f"{ordinal % 16:x}" * 64,
    )
    return CommandExample(
        source=source,
        label=selected_label,
        label_index=CLASS_ORDER.index(selected_label),
        identity=source.identity,
    )


class _TinyPreprocessor:
    def __init__(
        self,
        shape: tuple[int, int],
        *,
        signed_zero: float = 0.0,
        mutate_on_call: CommandExample | None = None,
    ) -> None:
        self.shape = shape
        self.signed_zero = signed_zero
        self.mutate_on_call = mutate_on_call
        self.calls: list[bytes] = []

    def command_model_input(
        self,
        example: CommandExample,
        *,
        seed: int,
        epoch: int,
    ) -> FloatArray:
        self.calls.append(example.identity)
        values = np.arange(np.prod(self.shape), dtype=np.float32).reshape(self.shape)
        values = np.ascontiguousarray(
            values
            + np.float32(example.source.manifest_index)
            + np.float32(seed)
            + np.float32(epoch)
        )
        values[0, 0] = np.float32(self.signed_zero)
        if self.mutate_on_call is not None:
            object.__setattr__(self.mutate_on_call, "label_index", 9)
        return values

    def silence_model_input(
        self,
        example: object,
        *,
        seed: int,
        epoch: int,
    ) -> FloatArray:
        del example, seed, epoch
        raise AssertionError("tiny tests do not contain silence examples")


def _layout(
    example_count: int,
    *,
    batch_size: int = 2,
) -> training._TrainingLayout:
    return training._TrainingLayout(
        example_count=example_count,
        batch_size=batch_size,
        mel_bins=2,
        time_frames=3,
    )


def _epoch(
    examples: tuple[CommandExample, ...],
    *,
    batch_size: int = 2,
    seed: int = 7,
    zero_based_epoch: int = 0,
    preprocessor: _TinyPreprocessor | None = None,
) -> tuple[RegisteredTrainingEpoch, training._TrainingLayout, _TinyPreprocessor]:
    layout = _layout(len(examples), batch_size=batch_size)
    processor = (
        _TinyPreprocessor((layout.mel_bins, layout.time_frames))
        if preprocessor is None
        else preprocessor
    )
    epoch = training._begin_training_epoch(
        examples,
        processor,
        seed=seed,
        zero_based_epoch=zero_based_epoch,
        layout=layout,
    )
    return epoch, layout, processor


def _complete_pending_batch(
    epoch: RegisteredTrainingEpoch,
    batch: MaterializedTrainingBatch,
    global_update: int,
    *,
    learning_rate: float = 0.125,
    loss: np.float32 | None = None,
    norm: np.float32 | None = None,
) -> CompletedTrainingUpdate:
    epoch.verify_pending_batch(batch, global_update=global_update)
    epoch.verify_pending_batch(batch, global_update=global_update)
    receipt = epoch.complete_update(
        batch,
        learning_rate=learning_rate,
        batch_mean_training_loss=np.float32(1.25) if loss is None else loss,
        returned_preclip_l2_norm=np.float32(2.5) if norm is None else norm,
    )
    epoch.accept_completed_update(receipt)
    return receipt


def _consume(
    epoch: RegisteredTrainingEpoch,
    layout: training._TrainingLayout,
) -> tuple[CompletedTrainingPopulation, tuple[CompletedTrainingUpdate, ...]]:
    receipts: list[CompletedTrainingUpdate] = []
    first_update = epoch.zero_based_epoch * layout.batch_count
    for batch_index in range(layout.batch_count):
        global_update = first_update + batch_index
        batch = epoch.next_batch(global_update)
        receipts.append(
            _complete_pending_batch(
                epoch,
                batch,
                global_update,
                learning_rate=0.01 * (batch_index + 1),
                loss=np.float32(batch_index + 0.25),
                norm=np.float32(batch_index + 1.5),
            )
        )
    return epoch.finish(), tuple(receipts)


def _golden_digest(
    examples: tuple[CommandExample, ...],
    processor: _TinyPreprocessor,
    *,
    seed: int,
    zero_based_epoch: int,
) -> str:
    framed = bytearray(struct.pack("<III", seed, zero_based_epoch, len(examples)))
    for example in examples:
        feature = processor.command_model_input(
            example,
            seed=seed,
            epoch=zero_based_epoch,
        )
        payload = feature.astype("<f4", copy=False).tobytes(order="C")
        framed.extend(struct.pack("<I", len(example.identity)))
        framed.extend(example.identity)
        framed.extend(struct.pack("<B", example.label_index))
        framed.extend(struct.pack("<Q", len(payload)))
        framed.extend(payload)
    return hashlib.sha256(training.TRAINING_POPULATION_DOMAIN + framed).hexdigest()


def test_registered_constants_and_layout_reconcile_with_frozen_config() -> None:
    config = json.loads(
        Path("configs/experiment-002-trainer.json").read_text(encoding="utf-8")
    )
    collation = config["collation"]["training"]
    digest = config["digests"]["training_population"]

    assert digest["domain"].encode() == training.TRAINING_POPULATION_DOMAIN
    assert training.TRAINING_BATCH_SIZE == collation["batch_size"] == 128
    assert collation["batch_count_per_epoch"] == training.REGISTERED_BATCH_COUNT
    assert collation["last_batch_size"] == training.REGISTERED_LAST_BATCH_SIZE
    assert training._REGISTERED_LAYOUT.example_count == 40_027
    assert training._REGISTERED_LAYOUT.batch_count == 313
    assert training._REGISTERED_LAYOUT.last_batch_size == 91
    assert (
        config["bindings"]["pcm_cache"]["artifact_sha256"]
        == training.REGISTERED_PCM_CACHE_SHA256
    )
    assert (
        config["bindings"]["normalization"]["artifact_sha256"]
        == training.REGISTERED_NORMALIZATION_SHA256
    )


def test_incremental_digest_matches_independent_struct_golden() -> None:
    examples = tuple(_example(index) for index in range(5))
    epoch, layout, processor = _epoch(
        examples,
        batch_size=2,
        seed=13,
        zero_based_epoch=3,
    )

    completed, receipts = _consume(epoch, layout)

    golden_processor = _TinyPreprocessor((2, 3))
    assert completed.sha256 == _golden_digest(
        examples,
        golden_processor,
        seed=13,
        zero_based_epoch=3,
    )
    assert completed.example_count == 5
    assert completed.batch_count == 3
    assert [receipt.batch_size for receipt in receipts] == [2, 2, 1]
    assert [receipt.zero_based_global_update for receipt in receipts] == [9, 10, 11]
    assert processor.calls == [example.identity for example in examples]


def test_digest_is_batch_boundary_invariant_and_order_sensitive() -> None:
    examples = tuple(_example(index) for index in range(5))
    first_epoch, first_layout, _ = _epoch(examples, batch_size=2)
    second_epoch, second_layout, _ = _epoch(examples, batch_size=3)
    reversed_epoch, reversed_layout, _ = _epoch(tuple(reversed(examples)), batch_size=2)

    first, _ = _consume(first_epoch, first_layout)
    second, _ = _consume(second_epoch, second_layout)
    reversed_result, _ = _consume(reversed_epoch, reversed_layout)

    assert first.sha256 == second.sha256
    assert first.sha256 != reversed_result.sha256


def test_digest_preserves_float32_signed_zero() -> None:
    examples = (_example(0),)
    positive_processor = _TinyPreprocessor((2, 3), signed_zero=0.0)
    negative_processor = _TinyPreprocessor((2, 3), signed_zero=-0.0)
    positive_epoch, positive_layout, _ = _epoch(
        examples,
        preprocessor=positive_processor,
    )
    negative_epoch, negative_layout, _ = _epoch(
        examples,
        preprocessor=negative_processor,
    )

    positive, _ = _consume(positive_epoch, positive_layout)
    negative, _ = _consume(negative_epoch, negative_layout)

    assert positive.sha256 != negative.sha256


def test_batches_are_owned_read_only_and_only_one_batch_is_retained() -> None:
    examples = tuple(_example(index) for index in range(3))
    epoch, layout, _ = _epoch(examples, batch_size=2)
    batch = epoch.next_batch(0)

    assert batch.model_inputs.shape == (2, 2, 3)
    assert batch.model_inputs.dtype == np.float32
    assert batch.model_inputs.flags.c_contiguous
    assert batch.model_inputs.flags.owndata
    assert not batch.model_inputs.flags.writeable
    assert batch.label_indices.shape == (2,)
    assert batch.label_indices.dtype == np.int64
    assert batch.label_indices.flags.owndata
    assert not batch.label_indices.flags.writeable
    state = training._issued_epoch_state(epoch)
    assert state.current_batch is batch
    assert not hasattr(state, "model_inputs")

    _complete_pending_batch(epoch, batch, 0)
    assert state.current_batch is None
    with pytest.raises(Experiment002TrainingPopulationError, match="not live"):
        _ = batch.model_inputs

    final_batch = epoch.next_batch(1)
    assert final_batch.batch_size == layout.last_batch_size == 1


def test_accept_releases_batch_array_storage() -> None:
    epoch, _, _ = _epoch((_example(0),))
    batch = epoch.next_batch(0)
    model_inputs = batch.model_inputs
    labels = batch.label_indices
    model_ref = weakref.ref(model_inputs)
    labels_ref = weakref.ref(labels)
    del model_inputs, labels

    _complete_pending_batch(epoch, batch, 0)
    gc.collect()

    assert model_ref() is None
    assert labels_ref() is None


def test_pending_batch_detects_byte_and_writeability_mutation() -> None:
    epoch, _, _ = _epoch((_example(0),))
    batch = epoch.next_batch(0)
    batch.model_inputs.setflags(write=True)
    batch.model_inputs[0, 0, 0] = np.float32(99.0)

    with pytest.raises(Experiment002TrainingPopulationError, match="immutable"):
        epoch.verify_pending_batch(batch, global_update=0)
    assert epoch.state == "FAILED"


def test_plan_mutation_during_materialization_fails_closed() -> None:
    first = _example(0)
    second = _example(1)
    processor = _TinyPreprocessor((2, 3), mutate_on_call=second)
    epoch, _, _ = _epoch((first, second), preprocessor=processor)

    with pytest.raises(Experiment002TrainingPopulationError, match="plan changed"):
        epoch.next_batch(0)
    assert epoch.state == "FAILED"


def test_state_machine_requires_exact_global_update_and_two_checks() -> None:
    wrong_epoch, _, _ = _epoch((_example(0),), zero_based_epoch=2)
    with pytest.raises(Experiment002TrainingPopulationError, match="expected=2"):
        wrong_epoch.next_batch(1)
    assert wrong_epoch.state == "FAILED"

    epoch, _, _ = _epoch((_example(1),))
    batch = epoch.next_batch(0)
    epoch.verify_pending_batch(batch, global_update=0)
    with pytest.raises(Experiment002TrainingPopulationError, match="exactly two"):
        epoch.complete_update(
            batch,
            learning_rate=0.1,
            batch_mean_training_loss=np.float32(1.0),
            returned_preclip_l2_norm=np.float32(2.0),
        )
    assert epoch.state == "FAILED"


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("learning_rate", np.float64(0.1), TypeError),
        ("learning_rate", float("nan"), Experiment002TrainingPopulationError),
        ("batch_mean_training_loss", 1.0, TypeError),
        (
            "batch_mean_training_loss",
            np.float32(np.inf),
            Experiment002TrainingPopulationError,
        ),
        ("returned_preclip_l2_norm", np.float64(2.0), TypeError),
    ],
)
def test_receipt_scalars_require_exact_finite_types(
    field: str,
    value: object,
    error: type[Exception],
) -> None:
    epoch, _, _ = _epoch((_example(0),))
    batch = epoch.next_batch(0)
    epoch.verify_pending_batch(batch, global_update=0)
    epoch.verify_pending_batch(batch, global_update=0)
    arguments: dict[str, object] = {
        "learning_rate": 0.1,
        "batch_mean_training_loss": np.float32(1.0),
        "returned_preclip_l2_norm": np.float32(2.0),
    }
    arguments[field] = value

    with pytest.raises(error):
        epoch.complete_update(batch, **cast(Any, arguments))
    assert epoch.state == "FAILED"


def test_receipt_snapshots_preserve_scalar_bits_and_ordered_opaque_tokens() -> None:
    epoch, layout, _ = _epoch(tuple(_example(index) for index in range(3)))
    completed, receipts = _consume(epoch, layout)
    population_snapshot = training._completed_training_population_snapshot(completed)

    assert all(receipt.accepted for receipt in receipts)
    assert len(population_snapshot.updates) == 2
    assert population_snapshot.ordered_batch_tokens == tuple(
        item.batch_token for item in population_snapshot.updates
    )
    assert population_snapshot.ordered_receipt_tokens == tuple(
        training._completed_training_update_token(receipt) for receipt in receipts
    )
    assert len({id(token) for token in population_snapshot.ordered_batch_tokens}) == 2
    assert len({id(token) for token in population_snapshot.ordered_receipt_tokens}) == 2
    for receipt, snapshot in zip(receipts, population_snapshot.updates, strict=True):
        receipt_snapshot = training._completed_training_update_snapshot(receipt)
        assert receipt_snapshot is not snapshot
        assert training._frame_update_snapshot(receipt_snapshot) == (
            training._frame_update_snapshot(snapshot)
        )
        assert snapshot.learning_rate_bytes == struct.pack("<d", receipt.learning_rate)
        assert snapshot.batch_mean_training_loss_bytes == struct.pack(
            "<f", receipt.batch_mean_training_loss
        )
        assert snapshot.returned_preclip_l2_norm_bytes == struct.pack(
            "<f", receipt.returned_preclip_l2_norm
        )


def test_receipt_is_epoch_and_batch_bound_and_accepts_exactly_once() -> None:
    first_epoch, _, _ = _epoch((_example(0),))
    second_epoch, _, _ = _epoch((_example(1),))
    first_batch = first_epoch.next_batch(0)
    second_batch = second_epoch.next_batch(0)
    first_epoch.verify_pending_batch(first_batch, global_update=0)
    first_epoch.verify_pending_batch(first_batch, global_update=0)
    receipt = first_epoch.complete_update(
        first_batch,
        learning_rate=0.1,
        batch_mean_training_loss=np.float32(1.0),
        returned_preclip_l2_norm=np.float32(2.0),
    )
    second_epoch.verify_pending_batch(second_batch, global_update=0)
    second_epoch.verify_pending_batch(second_batch, global_update=0)
    second_epoch.complete_update(
        second_batch,
        learning_rate=0.1,
        batch_mean_training_loss=np.float32(1.0),
        returned_preclip_l2_norm=np.float32(2.0),
    )

    with pytest.raises(Experiment002TrainingPopulationError, match="pending batch"):
        second_epoch.accept_completed_update(receipt)
    assert second_epoch.state == "FAILED"

    first_epoch.accept_completed_update(receipt)
    assert receipt.accepted is True
    with pytest.raises(
        Experiment002TrainingPopulationError, match="state must be READY"
    ):
        first_epoch.accept_completed_update(receipt)
    assert first_epoch.state == "FAILED"
    del second_batch


def test_finish_requires_completion_and_is_single_use() -> None:
    incomplete, _, _ = _epoch((_example(0),))
    with pytest.raises(Experiment002TrainingPopulationError, match="COMPLETE"):
        incomplete.finish()
    assert incomplete.state == "FAILED"

    epoch, layout, _ = _epoch((_example(1),))
    completed, _ = _consume(epoch, layout)
    assert completed.example_count == 1
    with pytest.raises(Experiment002TrainingPopulationError, match="already finalized"):
        epoch.finish()
    assert epoch.state == "FAILED"


def test_issuer_only_values_reject_forgery_copy_and_pickle() -> None:
    epoch, layout, _ = _epoch((_example(0),))
    batch = epoch.next_batch(0)
    epoch.verify_pending_batch(batch, global_update=0)
    epoch.verify_pending_batch(batch, global_update=0)
    receipt = epoch.complete_update(
        batch,
        learning_rate=0.1,
        batch_mean_training_loss=np.float32(1.0),
        returned_preclip_l2_norm=np.float32(2.0),
    )
    epoch.accept_completed_update(receipt)
    completed = epoch.finish()

    for issued in (epoch, receipt, completed):
        with pytest.raises(TypeError, match="cannot be copied"):
            copy.copy(issued)
        with pytest.raises(TypeError, match="cannot be copied"):
            copy.deepcopy(issued)
        with pytest.raises(TypeError, match="cannot be serialized"):
            pickle.dumps(issued)

    forged_batch = object.__new__(MaterializedTrainingBatch)
    with pytest.raises(Experiment002TrainingPopulationError, match="not live"):
        _ = forged_batch.model_inputs
    forged_receipt = object.__new__(CompletedTrainingUpdate)
    with pytest.raises(Experiment002TrainingPopulationError, match="not issued"):
        _ = forged_receipt.learning_rate
    forged_population = object.__new__(CompletedTrainingPopulation)
    with pytest.raises(Experiment002TrainingPopulationError, match="not issued"):
        _ = forged_population.sha256
    forged_epoch = object.__new__(RegisteredTrainingEpoch)
    with pytest.raises(Experiment002TrainingPopulationError, match="not issued"):
        _ = forged_epoch.state
    forged_source = object.__new__(RegisteredTrainingInputSource)
    with pytest.raises(Experiment002TrainingPopulationError, match="not issued"):
        forged_source.begin_epoch(seed=20_260_719, zero_based_epoch=0)
    assert layout.batch_count == 1


def test_issuer_state_exact_type_mutations_fail_closed() -> None:
    batch_epoch, _, _ = _epoch((_example(0),))
    batch = batch_epoch.next_batch(0)
    batch_state = training._issued_batch_state(batch)
    object.__setattr__(batch_state, "batch_index", np.int64(0))
    with pytest.raises(Experiment002TrainingPopulationError, match="issuer state"):
        _ = batch.batch_size

    receipt_epoch, receipt_layout, _ = _epoch((_example(1),))
    _, receipts = _consume(receipt_epoch, receipt_layout)
    receipt_state = training._issued_receipt_state(receipts[0])
    object.__setattr__(
        receipt_state.snapshot,
        "learning_rate",
        np.float64(receipt_state.snapshot.learning_rate),
    )
    with pytest.raises(Experiment002TrainingPopulationError, match="snapshot"):
        _ = receipts[0].learning_rate

    population_epoch, population_layout, _ = _epoch((_example(2),))
    population, _ = _consume(population_epoch, population_layout)
    population_state = training._issued_population_state(population)
    string_subclass = type("StringSubclass", (str,), {})
    object.__setattr__(
        population_state.snapshot,
        "sha256",
        string_subclass(population_state.snapshot.sha256),
    )
    with pytest.raises(
        Experiment002TrainingPopulationError, match="snapshot|authority"
    ):
        _ = population.sha256

    epoch_state_epoch, _, _ = _epoch((_example(3),))
    epoch_state = training._issued_epoch_state(epoch_state_epoch)
    epoch_state.phase = cast(Any, type("State", (str,), {})("OPEN"))
    with pytest.raises(Experiment002TrainingPopulationError, match="issuer state"):
        _ = epoch_state_epoch.state


def test_consistent_receipt_and_population_snapshot_rewrites_fail_authority() -> None:
    epoch, _, _ = _epoch((_example(0),))
    batch = epoch.next_batch(0)
    epoch.verify_pending_batch(batch, global_update=0)
    epoch.verify_pending_batch(batch, global_update=0)
    receipt = epoch.complete_update(
        batch,
        learning_rate=0.1,
        batch_mean_training_loss=np.float32(1.0),
        returned_preclip_l2_norm=np.float32(2.0),
    )
    receipt_state = training._issued_receipt_state(receipt)
    object.__setattr__(
        receipt_state.snapshot,
        "batch_mean_training_loss",
        np.float32(99.0),
    )
    object.__setattr__(
        receipt_state.snapshot,
        "batch_mean_training_loss_bytes",
        struct.pack("<f", np.float32(99.0)),
    )
    with pytest.raises(Experiment002TrainingPopulationError, match="authority"):
        epoch.accept_completed_update(receipt)
    assert epoch.state == "FAILED"

    complete_epoch, complete_layout, _ = _epoch((_example(1),))
    population, _ = _consume(complete_epoch, complete_layout)
    population_state = training._issued_population_state(population)
    object.__setattr__(population_state.snapshot, "sha256", "0" * 64)
    with pytest.raises(Experiment002TrainingPopulationError, match="authority"):
        _ = population.sha256


def test_accepted_update_rewrite_cannot_change_finished_population() -> None:
    epoch, _, _ = _epoch((_example(0),))
    batch = epoch.next_batch(0)
    _complete_pending_batch(epoch, batch, 0)
    state = training._issued_epoch_state(epoch)
    accepted = state.accepted_updates[0]
    object.__setattr__(
        accepted,
        "batch_mean_training_loss",
        np.float32(99.0),
    )
    object.__setattr__(
        accepted,
        "batch_mean_training_loss_bytes",
        struct.pack("<f", np.float32(99.0)),
    )

    with pytest.raises(Experiment002TrainingPopulationError, match="authority"):
        epoch.finish()
    assert epoch.state == "FAILED"

    token_epoch, _, _ = _epoch((_example(1),))
    token_batch = token_epoch.next_batch(0)
    _complete_pending_batch(token_epoch, token_batch, 0)
    token_state = training._issued_epoch_state(token_epoch)
    object.__setattr__(token_state.accepted_updates[0], "batch_token", object())
    with pytest.raises(Experiment002TrainingPopulationError, match="authority"):
        token_epoch.finish()
    assert token_epoch.state == "FAILED"


def test_cache_pcm_digest_is_not_compared_with_full_wav_digest() -> None:
    source = _example(0).source
    entry = _IndexEntry(
        payload_offset=4_096,
        sample_count=source.sample_count,
        pcm_sha256=b"\xff" * 32,
    )

    training._require_cache_entry_matches_source(source, entry)
    assert entry.pcm_sha256 != bytes.fromhex(source.sha256)


def test_public_bind_and_begin_are_stubbed_before_full_plan_or_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCorpus:
        pass

    class FakeCache:
        pass

    class FakeNormalization:
        pass

    class FakePreprocessor:
        def __init__(
            self, corpus: object, cache: object, normalization: object
        ) -> None:
            self.arguments = (corpus, cache, normalization)

    corpus = FakeCorpus()
    cache = FakeCache()
    normalization = FakeNormalization()
    calls: list[tuple[Any, ...]] = []
    plan = cast(tuple[training._Example, ...], (object(),))
    sentinel = object()

    monkeypatch.setattr(training, "Experiment002Corpus", FakeCorpus)
    monkeypatch.setattr(training, "PCMCacheSplitView", FakeCache)
    monkeypatch.setattr(training, "VerifiedNormalization", FakeNormalization)
    monkeypatch.setattr(training, "Experiment002TrainingPreprocessor", FakePreprocessor)

    def snapshot_corpus(value: object) -> tuple[int]:
        calls.append(("corpus", value))
        return (id(value),)

    def snapshot_cache(value: object, bound_corpus: object) -> tuple[int]:
        calls.append(("cache", value, bound_corpus))
        return (id(value),)

    def snapshot_normalization(value: object) -> bytes:
        calls.append(("normalization", value))
        return b"n"

    monkeypatch.setattr(
        training,
        "_snapshot_registered_corpus",
        snapshot_corpus,
    )
    monkeypatch.setattr(
        training,
        "_snapshot_registered_cache",
        snapshot_cache,
    )
    monkeypatch.setattr(
        training,
        "_snapshot_registered_normalization",
        snapshot_normalization,
    )
    monkeypatch.setattr(
        training,
        "_verify_registered_source_state",
        lambda state: calls.append(("verify-source", state)),
    )

    def fake_build(value: object, *, seed: int, epoch: int) -> object:
        calls.append(("build", value, seed, epoch))
        return plan

    def fake_begin(*args: object, **kwargs: object) -> object:
        calls.append(("begin-private", args, kwargs))
        return sentinel

    monkeypatch.setattr(training, "build_training_epoch", fake_build)
    monkeypatch.setattr(training, "_begin_training_epoch", fake_begin)

    source = training.bind_registered_training_inputs(
        cast(Any, corpus), cast(Any, cache), cast(Any, normalization)
    )
    observed = source.begin_epoch(seed=20_260_719, zero_based_epoch=0)

    assert observed is sentinel
    source_state = training._issued_source_state(source)
    observed_preprocessor = cast(Any, source_state.preprocessor)
    assert type(observed_preprocessor) is FakePreprocessor
    assert observed_preprocessor.arguments == (corpus, cache, normalization)
    assert sum(item[0] == "build" for item in calls) == 1
    assert sum(item[0] == "begin-private" for item in calls) == 1
    assert not any(item[0] == "materialize" for item in calls)

    with pytest.raises(Experiment002TrainingPopulationError, match="already begun"):
        source.begin_epoch(seed=20_260_719, zero_based_epoch=0)


def test_public_binder_rejects_nonexact_types_without_touching_inputs() -> None:
    with pytest.raises(TypeError, match="exact Experiment002Corpus"):
        training.bind_registered_training_inputs(
            cast(Any, object()), cast(Any, object()), cast(Any, object())
        )
