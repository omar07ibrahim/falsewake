from __future__ import annotations

import ast
import hashlib
import json
import os
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

import numpy as np
import pytest

import falsewake.experiment_002_data as data
import falsewake.speech_commands_pcm as pcm
from falsewake.experiment_002_data import (
    BackgroundSource,
    CommandExample,
    CommandSource,
    Experiment002Corpus,
    TrainingWindowUniverse,
    WindowExample,
    build_validation_population,
    pad_command_waveform,
    select_command_noise_window,
    slice_background_window,
)
from falsewake.experiment_002_rng import (
    encode_command_identity,
    encode_window_identity,
)
from falsewake.speech_commands_pcm import PCMSourceIdentity, VerifiedPCM16LE

CLASS_ORDER = (
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
)
REGISTERED_MANIFEST_SHA256 = (
    "d28e6993101bd6bc452033bcb7355b25e3b84ab5c51a1458cc097dc93c60f78b"
)
UNKNOWN_WORDS = (
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
)
UNKNOWN_WORD_ORDER = (
    "follow",
    "eight",
    "four",
    "happy",
    "tree",
    "six",
    "bed",
    "zero",
    "marvin",
    "sheila",
    "one",
    "visual",
    "bird",
    "backward",
    "cat",
    "house",
    "forward",
    "seven",
    "two",
    "learn",
    "dog",
    "wow",
    "five",
    "three",
    "nine",
)
TRAIN_BACKGROUNDS = (
    BackgroundSource(
        path="_background_noise_/doing_the_dishes.wav",
        sample_count=1_522_930,
        sha256="099eafcd7c4c266612012b5622b97157042e5288acaf4cbf3dc475814bfcdaa8",
    ),
    BackgroundSource(
        path="_background_noise_/dude_miaowing.wav",
        sample_count=988_891,
        sha256="1acd62f115d4c3f9daca9c5ec0c2e0c3e174a2a08be30200003a51de52b288ff",
    ),
    BackgroundSource(
        path="_background_noise_/exercise_bike.wav",
        sample_count=980_062,
        sha256="e453813ed45b2f9f81d5600ec3ac2a5e22d3f2c947fe6f0a3ff1f3d6018c014d",
    ),
    BackgroundSource(
        path="_background_noise_/pink_noise.wav",
        sample_count=960_000,
        sha256="b6e038c83fb342e39267d4fe69663f76ef0c1121ff60232e7b79171c00318cd6",
    ),
)
VALIDATION_BACKGROUND = BackgroundSource(
    path="_background_noise_/running_tap.wav",
    sample_count=978_488,
    sha256="c199c5fd61f5bf9fd57f1c346c8a51c0c035d63d176348dd9f0f7d671d7eb8eb",
)


def _command(
    path: str,
    *,
    manifest_index: int = 0,
    label: str | None = None,
    sample_count: int = 16_000,
    sha256: str = "1" * 64,
) -> CommandSource:
    word = path.split("/", maxsplit=1)[0]
    return CommandSource(
        manifest_index=manifest_index,
        path=path,
        word=word,
        label=label if label is not None else word,
        sample_count=sample_count,
        sha256=sha256,
    )


def _command_example(source: CommandSource) -> CommandExample:
    return CommandExample(
        source=source,
        label=source.label,
        label_index=CLASS_ORDER.index(source.label),
        identity=source.identity,
    )


def _corpus(
    *,
    train_commands: tuple[CommandSource, ...] = (),
    validation_commands: tuple[CommandSource, ...] = (),
    train_backgrounds: tuple[BackgroundSource, ...] = TRAIN_BACKGROUNDS,
    validation_background: BackgroundSource = VALIDATION_BACKGROUND,
) -> Experiment002Corpus:
    return Experiment002Corpus(
        train_commands=train_commands,
        validation_commands=validation_commands,
        train_backgrounds=train_backgrounds,
        validation_background=validation_background,
        test_command_count=11_005,
        manifest_sha256=REGISTERED_MANIFEST_SHA256,
    )


def _snapshot(
    source: CommandSource | BackgroundSource, payload: bytes
) -> VerifiedPCM16LE:
    return pcm._new_verified_pcm16le(
        path=source.path,
        sample_count=source.sample_count,
        sha256=source.sha256,
        payload=payload,
    )


def _command_row(
    path: str,
    split: str,
    *,
    label: str | None = None,
    sample_count: int = 16_000,
    sha256: str = "1" * 64,
) -> dict[str, object]:
    word = path.split("/", maxsplit=1)[0]
    return {
        "kind": "command",
        "label": label if label is not None else word,
        "path": path,
        "sample_count": sample_count,
        "sha256": sha256,
        "speaker_id": Path(path).stem.split("_nohash_", maxsplit=1)[0],
        "split": split,
        "utterance_index": 0,
        "word": word,
    }


def _background_row(source: BackgroundSource, split: str) -> dict[str, object]:
    return {
        "kind": "background",
        "path": source.path,
        "sample_count": source.sample_count,
        "sha256": source.sha256,
        "split": split,
    }


def _synthetic_backgrounds() -> tuple[
    BackgroundSource, BackgroundSource, BackgroundSource
]:
    return (
        BackgroundSource(
            path="_background_noise_/room.wav",
            sample_count=16_003,
            sha256="a" * 64,
        ),
        BackgroundSource(
            path="_background_noise_/tap.wav",
            sample_count=17_600,
            sha256="b" * 64,
        ),
        BackgroundSource(
            path="_background_noise_/white.wav",
            sample_count=16_000,
            sha256="c" * 64,
        ),
    )


def _synthetic_rows() -> list[dict[str, object]]:
    train_background, validation_background, test_background = _synthetic_backgrounds()
    return [
        _command_row("cat/a_nohash_0.wav", "train", label="unknown"),
        _command_row("cat/b_nohash_0.wav", "validation", label="unknown"),
        _command_row("no/c_nohash_0.wav", "validation"),
        _command_row("up/d_nohash_0.wav", "test"),
        _command_row("yes/e_nohash_0.wav", "train"),
        _background_row(train_background, "train"),
        _background_row(validation_background, "validation"),
        _background_row(test_background, "test"),
    ]


def _jsonl(rows: Sequence[dict[str, object]]) -> bytes:
    return b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for row in rows
    )


def _synthetic_contract(contents: bytes) -> data._DataContract:
    train_background, validation_background, _ = _synthetic_backgrounds()
    return data._DataContract(
        manifest_sha256=hashlib.sha256(contents).hexdigest(),
        train_command_count=2,
        validation_command_count=2,
        test_command_count=1,
        train_target_support=(("yes", 1),),
        validation_target_support=(("no", 1),),
        train_unknown_count=1,
        validation_unknown_count=1,
        unknown_words=("cat",),
        backgrounds=(
            data._BackgroundExpectation(split="train", source=train_background),
            data._BackgroundExpectation(
                split="validation", source=validation_background
            ),
        ),
        test_background_count=1,
    )


def _load_synthetic(
    tmp_path: Path,
    *,
    rows: Sequence[dict[str, object]] | None = None,
    contents: bytes | None = None,
) -> Experiment002Corpus:
    if contents is None:
        contents = _jsonl(_synthetic_rows() if rows is None else rows)
    manifest = tmp_path / "speech-commands.jsonl"
    manifest.write_bytes(contents)
    return data._load_corpus(manifest, contract=_synthetic_contract(contents))


def test_synthetic_manifest_is_strictly_loaded_without_exposing_test_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbid_write(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"unexpected filesystem write: {args!r}, {kwargs!r}")

    monkeypatch.setattr(Path, "write_text", forbid_write)
    corpus = _load_synthetic(tmp_path)

    assert [source.path for source in corpus.train_commands] == [
        "cat/a_nohash_0.wav",
        "yes/e_nohash_0.wav",
    ]
    assert [source.path for source in corpus.validation_commands] == [
        "cat/b_nohash_0.wav",
        "no/c_nohash_0.wav",
    ]
    assert corpus.train_backgrounds == (_synthetic_backgrounds()[0],)
    assert corpus.validation_background == _synthetic_backgrounds()[1]
    assert corpus.test_command_count == 1
    assert "up/d_nohash_0.wav" not in repr(corpus)
    assert "_background_noise_/white.wav" not in repr(corpus)

    requested = [
        source.pcm_identity
        for source in (*corpus.train_commands, *corpus.validation_commands)
    ]
    requested.extend(
        PCMSourceIdentity(
            path=source.path,
            sample_count=source.sample_count,
            sha256=source.sha256,
        )
        for source in corpus.train_backgrounds
    )
    source = corpus.validation_background
    requested.append(
        PCMSourceIdentity(
            path=source.path,
            sample_count=source.sample_count,
            sha256=source.sha256,
        )
    )
    assert all(identity.path != "up/d_nohash_0.wav" for identity in requested)
    assert all(
        identity.path != "_background_noise_/white.wav" for identity in requested
    )


def test_public_loader_requires_the_exact_registered_manifest_digest(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "speech-commands.jsonl"
    manifest.write_bytes(_jsonl(_synthetic_rows()))

    with pytest.raises(data.Experiment002DataError, match="digest"):
        data.load_registered_corpus(manifest)


def test_registered_counts_supports_and_manifest_limit_are_exact() -> None:
    contract = data._REGISTERED_CONTRACT

    assert data.MANIFEST_SHA256 == REGISTERED_MANIFEST_SHA256
    assert data.MAX_MANIFEST_BYTES == 64 * 1024 * 1024
    assert data.TRAINING_SEEDS == (20_260_719, 20_260_720, 20_260_721)
    assert data.TRAIN_EPOCH_COUNT == 30
    assert data.TRAIN_COMMAND_INVENTORY_SHA256 == (
        "74c0e622b3b30df7600cd452f2d88ed62cabcbca040aba54a89f8fcf819c3546"
    )
    assert data.VALIDATION_COMMAND_INVENTORY_SHA256 == (
        "c5373d67a0bb97bb54ff0bd576ec7b089aa398a273e25115286a677ddb4140ce"
    )
    assert (
        contract.train_command_count,
        contract.validation_command_count,
        contract.test_command_count,
    ) == (84_843, 9_981, 11_005)
    assert dict(contract.train_target_support) == {
        "yes": 3_228,
        "no": 3_130,
        "up": 2_948,
        "down": 3_134,
        "left": 3_037,
        "right": 3_019,
        "on": 3_086,
        "off": 2_970,
        "stop": 3_111,
        "go": 3_106,
    }
    assert dict(contract.validation_target_support) == {
        "yes": 397,
        "no": 406,
        "up": 350,
        "down": 377,
        "left": 352,
        "right": 363,
        "on": 363,
        "off": 373,
        "stop": 350,
        "go": 372,
    }
    assert (contract.train_unknown_count, contract.validation_unknown_count) == (
        54_074,
        6_278,
    )
    assert contract.unknown_words == UNKNOWN_WORDS
    assert contract.test_background_count == 1
    assert all(expectation.split != "test" for expectation in contract.backgrounds)
    assert (
        data.TRAIN_TARGET_COUNT,
        data.TRAIN_UNKNOWN_COUNT,
        data.TRAIN_SILENCE_COUNT,
        data.TRAIN_EXAMPLE_COUNT,
    ) == (30_769, 6_172, 3_086, 40_027)
    assert (
        data.VALIDATION_TARGET_COUNT,
        data.VALIDATION_UNKNOWN_COUNT,
        data.VALIDATION_SILENCE_COUNT,
        data.VALIDATION_EXAMPLE_COUNT,
    ) == (3_703, 6_278, 602, 10_583)


def test_manifest_size_limit_fails_before_digest_or_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "speech-commands.jsonl"
    manifest.write_bytes(b"123456789")
    monkeypatch.setattr(data, "MAX_MANIFEST_BYTES", 8)

    with pytest.raises(data.Experiment002DataError, match="limit"):
        data._load_corpus(manifest, data._REGISTERED_CONTRACT)


def test_manifest_reader_rejects_symlinks_and_nonregular_files(tmp_path: Path) -> None:
    target = tmp_path / "target.jsonl"
    target.write_bytes(_jsonl(_synthetic_rows()))
    link = tmp_path / "link.jsonl"
    link.symlink_to(target)
    with pytest.raises(data.Experiment002DataError, match="securely open"):
        data._read_manifest(link)

    directory = tmp_path / "directory.jsonl"
    directory.mkdir()
    with pytest.raises(data.Experiment002DataError, match="regular file"):
        data._read_manifest(directory)

    fifo = tmp_path / "fifo.jsonl"
    os.mkfifo(fifo)
    with pytest.raises(data.Experiment002DataError, match="regular file"):
        data._read_manifest(fifo)


def test_manifest_digest_is_checked_before_json_decoding(tmp_path: Path) -> None:
    contents = b"not JSON\n"
    contract = _synthetic_contract(contents)
    contract = data._DataContract(
        manifest_sha256="0" * 64,
        train_command_count=contract.train_command_count,
        validation_command_count=contract.validation_command_count,
        test_command_count=contract.test_command_count,
        train_target_support=contract.train_target_support,
        validation_target_support=contract.validation_target_support,
        train_unknown_count=contract.train_unknown_count,
        validation_unknown_count=contract.validation_unknown_count,
        unknown_words=contract.unknown_words,
        backgrounds=contract.backgrounds,
        test_background_count=contract.test_background_count,
    )
    manifest = tmp_path / "speech-commands.jsonl"
    manifest.write_bytes(contents)

    with pytest.raises(data.Experiment002DataError, match="digest"):
        data._load_corpus(manifest, contract=contract)


@pytest.mark.parametrize(
    ("row_index", "mutation"),
    [
        (0, lambda row: row.pop("speaker_id")),
        (0, lambda row: row.__setitem__("extra", 1)),
        (0, lambda row: row.__setitem__("kind", "clip")),
        (0, lambda row: row.__setitem__("split", "development")),
        (0, lambda row: row.__setitem__("sample_count", True)),
        (0, lambda row: row.__setitem__("sample_count", 0)),
        (0, lambda row: row.__setitem__("sample_count", 16_001)),
        (0, lambda row: row.__setitem__("sha256", "A" * 64)),
        (0, lambda row: row.__setitem__("utterance_index", True)),
        (5, lambda row: row.pop("split")),
        (5, lambda row: row.__setitem__("extra", 1)),
        (5, lambda row: row.__setitem__("sample_count", 0)),
        (5, lambda row: row.__setitem__("sha256", "g" * 64)),
    ],
)
def test_manifest_rejects_non_exact_schemas_and_field_types(
    tmp_path: Path,
    row_index: int,
    mutation: Callable[[dict[str, object]], object],
) -> None:
    rows = _synthetic_rows()
    mutation(rows[row_index])

    with pytest.raises(data.Experiment002DataError):
        _load_synthetic(tmp_path, rows=rows)


@pytest.mark.parametrize(
    "path",
    [
        "",
        "/cat/a_nohash_0.wav",
        "../cat/a_nohash_0.wav",
        "./cat/a_nohash_0.wav",
        "cat//a_nohash_0.wav",
        "cat/a_nohash_0.wav/",
        "cat\\a_nohash_0.wav",
        "cat/a\0_nohash_0.wav",
        "cat/nested/a_nohash_0.wav",
        "cat/a_nohash_0.flac",
        "cat/bad-\ud800.wav",
    ],
)
def test_manifest_rejects_noncanonical_or_non_utf8_paths(
    tmp_path: Path, path: str
) -> None:
    rows = _synthetic_rows()
    rows[0]["path"] = path

    with pytest.raises(data.Experiment002DataError):
        _load_synthetic(tmp_path, rows=rows)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("word", "dog"),
        ("label", "cat"),
        ("label", "yes"),
    ],
)
def test_manifest_rejects_word_and_label_disagreement(
    tmp_path: Path, field: str, value: object
) -> None:
    rows = _synthetic_rows()
    rows[0][field] = value

    with pytest.raises(data.Experiment002DataError):
        _load_synthetic(tmp_path, rows=rows)


def test_manifest_rejects_duplicate_and_non_bytewise_ordered_command_paths(
    tmp_path: Path,
) -> None:
    duplicate = _synthetic_rows()
    duplicate[1]["path"] = duplicate[0]["path"]
    duplicate[1]["word"] = duplicate[0]["word"]
    duplicate[1]["label"] = "unknown"
    with pytest.raises(data.Experiment002DataError):
        _load_synthetic(tmp_path, rows=duplicate)

    unordered = _synthetic_rows()
    unordered[0], unordered[1] = unordered[1], unordered[0]
    with pytest.raises(data.Experiment002DataError, match="order"):
        _load_synthetic(tmp_path, rows=unordered)


def test_manifest_rejects_invalid_utf8_duplicate_json_keys_and_empty_lines(
    tmp_path: Path,
) -> None:
    invalid_documents = (
        b'\xff{"kind":"command"}\n',
        (
            b'{"kind":"command","kind":"background","label":"unknown",'
            b'"path":"cat/a_nohash_0.wav","sample_count":16000,'
            b'"sha256":"' + b"1" * 64 + b'","speaker_id":"a",'
            b'"split":"train","utterance_index":0,"word":"cat"}\n'
        ),
        _jsonl(_synthetic_rows()) + b"\n",
        _jsonl(_synthetic_rows()).replace(b"\n", b"\r\n"),
        b'{"kind":' + b"9" * 5_000 + b"}\n",
    )
    for index, contents in enumerate(invalid_documents):
        case = tmp_path / str(index)
        case.mkdir()
        with pytest.raises(data.Experiment002DataError):
            _load_synthetic(case, contents=contents)


def test_source_identities_are_exact_and_test_identities_have_no_corpus_field() -> None:
    command = _command("yes/alice_nohash_0.wav", manifest_index=17)
    background = TRAIN_BACKGROUNDS[0]
    corpus = _corpus(train_commands=(command,))

    assert command.identity == encode_command_identity(command.path)
    assert command.pcm_identity == PCMSourceIdentity(
        path=command.path,
        sample_count=command.sample_count,
        sha256=command.sha256,
    )
    window = background.window(123)
    assert window == WindowExample(
        source=background,
        start_sample=123,
        label="silence",
        label_index=11,
        identity=encode_window_identity(background.path, 123),
    )
    assert corpus.test_command_count == 11_005
    assert [field for field in corpus.__dataclass_fields__ if "test" in field] == [
        "test_command_count"
    ]
    with pytest.raises(data.Experiment002DataError, match="uint32"):
        _command("yes/overflow.wav", manifest_index=1 << 32)


def test_forged_corpus_cannot_claim_the_registered_manifest() -> None:
    train = _command("yes/forged_train.wav", manifest_index=0)
    validation = _command("no/forged_validation.wav", manifest_index=1)
    corpus = _corpus(train_commands=(train,), validation_commands=(validation,))

    assert corpus._train_inventory_sha256 == data._command_inventory_sha256((train,))
    assert corpus._validation_inventory_sha256 == data._command_inventory_sha256(
        (validation,)
    )
    assert corpus._train_inventory_sha256 != data.TRAIN_COMMAND_INVENTORY_SHA256
    assert (
        corpus._validation_inventory_sha256 != data.VALIDATION_COMMAND_INVENTORY_SHA256
    )
    with pytest.raises(data.Experiment002DataError, match="inventory"):
        data._require_registered_training_corpus(corpus)
    with pytest.raises(data.Experiment002DataError, match="inventory"):
        data._require_registered_window_universe(corpus)
    with pytest.raises(data.Experiment002DataError, match="inventory"):
        data._require_registered_validation_corpus(corpus)


def test_registered_training_window_universe_boundaries_are_exact() -> None:
    universe = TrainingWindowUniverse(TRAIN_BACKGROUNDS)

    assert universe.size == 4_387_887
    observed = {
        index: (universe.at(index).source.path, universe.at(index).start_sample)
        for index in (0, 1_506_930, 1_506_931, 4_387_886)
    }
    assert observed == {
        0: ("_background_noise_/doing_the_dishes.wav", 0),
        1_506_930: (
            "_background_noise_/doing_the_dishes.wav",
            1_506_930,
        ),
        1_506_931: ("_background_noise_/dude_miaowing.wav", 0),
        4_387_886: ("_background_noise_/pink_noise.wav", 944_000),
    }
    with pytest.raises(ValueError):
        universe.at(-1)
    with pytest.raises(IndexError):
        universe.at(universe.size)
    with pytest.raises(TypeError, match="BackgroundSource"):
        TrainingWindowUniverse(cast(Sequence[BackgroundSource], (object(),)))


def test_unknown_word_rank_and_balanced_clip_selection_are_exact() -> None:
    ranked_words = data._rank_unknown_words(
        reversed(UNKNOWN_WORDS),
        seed=20_260_719,
        epoch=0,
    )
    assert ranked_words == UNKNOWN_WORD_ORDER

    commands = tuple(
        _command(
            f"{word}/speaker-{clip_index:03d}.wav",
            manifest_index=word_index * 248 + clip_index,
            label="unknown",
            sha256=f"{word_index * 248 + clip_index:064x}",
        )
        for word_index, word in enumerate(UNKNOWN_WORDS)
        for clip_index in range(248)
    )
    selected = data._select_unknown_commands(
        reversed(commands),
        seed=20_260_719,
        epoch=0,
    )

    support = Counter(example.source.word for example in selected)
    assert len(selected) == 6_172
    assert len({example.identity for example in selected}) == 6_172
    assert support == {
        word: 247 if index < 22 else 246
        for index, word in enumerate(UNKNOWN_WORD_ORDER)
    }
    assert selected == data._select_unknown_commands(
        commands,
        seed=20_260_719,
        epoch=0,
    )
    assert {
        example.identity
        for example in data._select_unknown_commands(
            commands,
            seed=20_260_720,
            epoch=0,
        )
    } != {example.identity for example in selected}


def test_small_silence_universe_selection_is_unique_and_ties_use_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = BackgroundSource(
        path="_background_noise_/a.wav",
        sample_count=16_001,
        sha256="a" * 64,
    )
    second = BackgroundSource(
        path="_background_noise_/b.wav",
        sample_count=16_000,
        sha256="b" * 64,
    )
    forward = TrainingWindowUniverse((first, second))
    reverse = TrainingWindowUniverse((second, first))
    assert [example.identity for example in forward] == [
        example.identity for example in reverse
    ]
    assert forward.size == 3

    calls: list[tuple[bytes, str]] = []

    def tied_rank_key(
        *, seed: int, epoch: int, identity: bytes, domain: str
    ) -> tuple[bytes, bytes]:
        assert (seed, epoch) == (7, 3)
        calls.append((identity, domain))
        return b"\0" * 32, identity

    monkeypatch.setattr(data, "rank_key", tied_rank_key)
    selected = data._select_silence_windows(
        reverse,
        count=2,
        seed=7,
        epoch=3,
    )

    expected = tuple(sorted(forward, key=lambda example: example.identity)[:2])
    assert selected == expected
    assert len({example.identity for example in selected}) == 2
    assert calls == [(example.identity, "silence-window-rank") for example in reverse]


def test_final_order_is_stateless_and_independent_of_input_container_order() -> None:
    examples = (
        _command_example(_command("yes/a.wav", manifest_index=0)),
        _command_example(_command("no/b.wav", manifest_index=1)),
        TRAIN_BACKGROUNDS[0].window(0),
        TRAIN_BACKGROUNDS[1].window(10),
    )

    first = data._order_training_examples(examples, seed=20_260_719, epoch=0)
    repeated = data._order_training_examples(
        reversed(examples), seed=20_260_719, epoch=0
    )
    next_epoch = data._order_training_examples(examples, seed=20_260_719, epoch=1)

    assert first == repeated
    assert first != next_epoch
    assert {example.identity for example in first} == {
        example.identity for example in next_epoch
    }
    assert {
        example.identity for example in first if isinstance(example, CommandExample)
    } == {examples[0].identity, examples[1].identity}


def test_registered_command_noise_mapping_matches_the_golden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _command("yes/alice_nohash_0.wav")
    corpus = _corpus(train_commands=(command,))
    monkeypatch.setattr(data, "_require_registered_window_universe", lambda _: None)

    window = select_command_noise_window(
        corpus,
        command,
        seed=20_260_719,
        epoch=0,
    )

    assert (window.source.path, window.start_sample) == (
        "_background_noise_/exercise_bike.wav",
        783_586,
    )
    validation_only = _command("no/validation_nohash_0.wav", manifest_index=1)
    with pytest.raises(data.Experiment002DataError, match="training command"):
        select_command_noise_window(
            corpus,
            validation_only,
            seed=20_260_719,
            epoch=0,
        )
    with pytest.raises(data.Experiment002DataError, match="seed"):
        select_command_noise_window(corpus, command, seed=7, epoch=0)
    with pytest.raises(data.Experiment002DataError, match="epoch"):
        select_command_noise_window(
            corpus,
            command,
            seed=20_260_719,
            epoch=30,
        )


def test_validation_population_has_exact_support_order_and_full_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_support = {
        "yes": 397,
        "no": 406,
        "up": 350,
        "down": 377,
        "left": 352,
        "right": 363,
        "on": 363,
        "off": 373,
        "stop": 350,
        "go": 372,
    }
    commands: list[CommandSource] = []
    for label, count in target_support.items():
        commands.extend(
            _command(
                f"{label}/speaker-{index:05d}.wav",
                manifest_index=len(commands),
            )
            for index in range(count)
        )
    commands.extend(
        _command(
            f"cat/speaker-{index:05d}.wav",
            manifest_index=len(commands),
            label="unknown",
        )
        for index in range(6_278)
    )
    command_tuple = tuple(reversed(commands))
    corpus = _corpus(validation_commands=command_tuple)

    with pytest.raises(data.Experiment002DataError, match="inventory"):
        build_validation_population(corpus)
    monkeypatch.setattr(data, "_require_registered_validation_corpus", lambda _: None)

    population = build_validation_population(corpus)

    expected_commands = sorted(command_tuple, key=lambda source: source.path.encode())
    assert [example.identity for example in population[:9_981]] == [
        command.identity for command in expected_commands
    ]
    windows = cast(tuple[WindowExample, ...], population[9_981:])
    assert len(windows) == 602
    assert [window.start_sample for window in windows[:3]] == [0, 1_600, 3_200]
    assert windows[-1].start_sample == 961_600
    assert all(window.source == VALIDATION_BACKGROUND for window in windows)
    expected_support = {**target_support, "unknown": 6_278, "silence": 602}
    assert Counter(example.label for example in population) == expected_support
    assert len(population) == 10_583


def test_command_padding_preserves_exact_prefix_and_uses_positive_zero_tail() -> None:
    samples = np.asarray([-32_768, -1, 0, 1, 32_767], dtype=np.int16)
    source = _command(
        "yes/source.wav",
        sample_count=samples.size,
        sha256="a" * 64,
    )
    snapshot = _snapshot(source, samples.astype("<i2").tobytes())
    payload_before = snapshot.payload

    waveform = pad_command_waveform(_command_example(source), snapshot)

    expected_prefix = np.asarray(
        [-1.0, -1.0 / 32_768.0, 0.0, 1.0 / 32_768.0, 32_767.0 / 32_768.0],
        dtype=np.float32,
    )
    assert waveform.shape == (16_000,)
    assert waveform.dtype == np.dtype(np.float32)
    assert waveform.flags.c_contiguous and waveform.flags.owndata
    np.testing.assert_array_equal(waveform[: samples.size], expected_prefix)
    assert np.all(waveform[samples.size :].view("<u4") == 0)
    assert snapshot.payload == payload_before


def test_background_materialization_returns_owned_first_and_last_raw_slices() -> None:
    samples = np.arange(16_003, dtype=np.int16)
    source = BackgroundSource(
        path="_background_noise_/tiny.wav",
        sample_count=samples.size,
        sha256="b" * 64,
    )
    snapshot = _snapshot(source, samples.astype("<i2").tobytes())

    first = slice_background_window(source.window(0), snapshot)
    last = slice_background_window(source.window(3), snapshot)

    np.testing.assert_array_equal(
        first,
        samples[:16_000].astype(np.float32) / np.float32(32_768.0),
    )
    np.testing.assert_array_equal(
        last,
        samples[3:16_003].astype(np.float32) / np.float32(32_768.0),
    )
    assert first.flags.c_contiguous and first.flags.owndata
    assert last.flags.c_contiguous and last.flags.owndata
    assert not np.shares_memory(first, last)
    with pytest.raises(ValueError):
        source.window(-1)
    with pytest.raises(ValueError):
        source.window(4)


@pytest.mark.parametrize("field", ["path", "sample_count", "sha256"])
def test_materializers_reject_every_snapshot_identity_mismatch(field: str) -> None:
    command = _command(
        "yes/source.wav",
        sample_count=5,
        sha256="c" * 64,
    )
    background = BackgroundSource(
        path="_background_noise_/source.wav",
        sample_count=16_000,
        sha256="d" * 64,
    )
    command_values: dict[str, object] = {
        "path": command.path,
        "sample_count": command.sample_count,
        "sha256": command.sha256,
        "payload": b"\0\0" * command.sample_count,
    }
    background_values: dict[str, object] = {
        "path": background.path,
        "sample_count": background.sample_count,
        "sha256": background.sha256,
        "payload": b"\0\0" * background.sample_count,
    }
    if field == "path":
        command_values[field] = "yes/different.wav"
        background_values[field] = "_background_noise_/different.wav"
    elif field == "sample_count":
        command_values[field] = 4
        command_values["payload"] = b"\0\0" * 4
        background_values[field] = 16_001
        background_values["payload"] = b"\0\0" * 16_001
    else:
        command_values[field] = "e" * 64
        background_values[field] = "f" * 64
    command_snapshot = pcm._new_verified_pcm16le(
        path=cast(str, command_values["path"]),
        sample_count=cast(int, command_values["sample_count"]),
        sha256=cast(str, command_values["sha256"]),
        payload=cast(bytes, command_values["payload"]),
    )
    background_snapshot = pcm._new_verified_pcm16le(
        path=cast(str, background_values["path"]),
        sample_count=cast(int, background_values["sample_count"]),
        sha256=cast(str, background_values["sha256"]),
        payload=cast(bytes, background_values["payload"]),
    )

    with pytest.raises(ValueError, match="identity"):
        pad_command_waveform(_command_example(command), command_snapshot)
    with pytest.raises(ValueError, match="identity"):
        slice_background_window(background.window(0), background_snapshot)


def test_data_module_has_no_training_runtime_global_rng_or_write_dependency() -> None:
    source_path = Path("src/falsewake/experiment_002_data.py")
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(
                alias.name.split(".", maxsplit=1)[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module.split(".", maxsplit=1)[0])
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    forbidden_imports = {"librispeech", "onnx", "onnxruntime", "random", "torch"}
    assert imported.isdisjoint(forbidden_imports)
    assert called_attributes.isdisjoint(
        {"mkdir", "rename", "touch", "unlink", "write_bytes", "write_text"}
    )
    assert "numpy.random" not in source
    assert "np.random" not in source
    assert "_background_noise_/white_noise.wav" not in source


def test_frozen_data_predecessors_remain_byte_identical() -> None:
    expected = {
        "src/falsewake/feature_matrix.py": (
            "82ca61ebda6ec833d0047acb4a88091411efa16cbc5667ae7b16ca85d8842aff"
        ),
        "src/falsewake/features.py": (
            "3e7baf53c8dc772a68cd256ce6f4ca6428117c36f051c916f247dc409e515e3c"
        ),
        "src/falsewake/speech_commands.py": (
            "f80bd2bad56113d4adc168b04e1633ec22d8513afb973a4b5dc37d616901e234"
        ),
    }
    assert {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in expected
    } == expected
