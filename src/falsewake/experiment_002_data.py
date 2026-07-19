"""Registered data populations and PCM materialization for Experiment 002.

The public manifest loader is deliberately bound to the audited Speech Commands
JSONL digest.  It retains training and validation source identities only; test
identities are validated while parsing and then discarded behind a count-only
firewall.
"""

from __future__ import annotations

import hashlib
import heapq
import hmac
import json
import os
import stat
import struct
from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Final, Literal, NoReturn, cast

import numpy as np

from falsewake.experiment_002_rng import (
    UINT32_MAX,
    UINT64_MAX,
    encode_command_identity,
    encode_source_word_identity,
    encode_window_identity,
    rank_key,
    uniform_integer,
)
from falsewake.features import FloatArray, pcm16le_to_float32
from falsewake.speech_commands_pcm import (
    PCMSourceIdentity,
    VerifiedPCM16LE,
)

type Split = Literal["train", "validation", "test"]
type WindowLabel = Literal["silence"]

TARGET_WORDS: Final = (
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
)
CLASS_ORDER: Final = (*TARGET_WORDS, "unknown", "silence")
UNKNOWN_WORDS: Final = (
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

MANIFEST_SHA256: Final = (
    "d28e6993101bd6bc452033bcb7355b25e3b84ab5c51a1458cc097dc93c60f78b"
)
MAX_MANIFEST_BYTES: Final = 64 * 1024 * 1024
WINDOW_SAMPLES: Final = 16_000
VALIDATION_HOP_SAMPLES: Final = 1_600
TRAIN_TARGET_COUNT: Final = 30_769
TRAIN_UNKNOWN_SOURCE_COUNT: Final = 54_074
TRAIN_UNKNOWN_COUNT: Final = 6_172
TRAIN_SILENCE_COUNT: Final = 3_086
TRAIN_EXAMPLE_COUNT: Final = 40_027
VALIDATION_TARGET_COUNT: Final = 3_703
VALIDATION_UNKNOWN_COUNT: Final = 6_278
VALIDATION_SILENCE_COUNT: Final = 602
VALIDATION_EXAMPLE_COUNT: Final = 10_583
TRAIN_WINDOW_UNIVERSE_SIZE: Final = 4_387_887
TRAINING_SEEDS: Final = (20_260_719, 20_260_720, 20_260_721)
TRAIN_EPOCH_COUNT: Final = 30
TRAIN_COMMAND_INVENTORY_SHA256: Final = (
    "74c0e622b3b30df7600cd452f2d88ed62cabcbca040aba54a89f8fcf819c3546"
)
VALIDATION_COMMAND_INVENTORY_SHA256: Final = (
    "c5373d67a0bb97bb54ff0bd576ec7b089aa398a273e25115286a677ddb4140ce"
)
_COMMAND_INVENTORY_PREFIX: Final = b"falsewake-exp002-command-inventory-v1\0"

_COMMAND_KEYS: Final = frozenset(
    {
        "kind",
        "label",
        "path",
        "sample_count",
        "sha256",
        "speaker_id",
        "split",
        "utterance_index",
        "word",
    }
)
_BACKGROUND_KEYS: Final = frozenset({"kind", "path", "sample_count", "sha256", "split"})


class Experiment002DataError(ValueError):
    """The supplied metadata or PCM does not satisfy the registered protocol."""


@dataclass(frozen=True, slots=True)
class CommandSource:
    """One immutable training or validation command-source identity."""

    manifest_index: int
    path: str
    word: str
    label: str
    sample_count: int
    sha256: str

    def __post_init__(self) -> None:
        _require_nonnegative_integer("manifest_index", self.manifest_index)
        if self.manifest_index > UINT32_MAX:
            raise Experiment002DataError("manifest_index exceeds uint32")
        _validate_command_fields(
            path=self.path,
            word=self.word,
            label=self.label,
            sample_count=self.sample_count,
            sha256=self.sha256,
        )

    @property
    def identity(self) -> bytes:
        """Return the registered command HMAC identity."""

        return encode_command_identity(self.path)

    @property
    def pcm_identity(self) -> PCMSourceIdentity:
        """Return the full-WAV identity accepted by the secure PCM loader."""

        return PCMSourceIdentity(
            path=self.path,
            sample_count=self.sample_count,
            sha256=self.sha256,
        )


@dataclass(frozen=True, slots=True)
class BackgroundSource:
    """One immutable registered background-recording identity."""

    path: str
    sample_count: int
    sha256: str

    def __post_init__(self) -> None:
        parts = _canonical_path_parts(self.path)
        if parts[0] != "_background_noise_":
            raise Experiment002DataError("background path has an invalid directory")
        if not isinstance(self.sample_count, int) or isinstance(
            self.sample_count, bool
        ):
            raise TypeError("sample_count must be an integer")
        if self.sample_count < WINDOW_SAMPLES:
            raise Experiment002DataError(
                f"background must contain at least {WINDOW_SAMPLES} samples"
            )
        _validate_sha256(self.sha256)
        PCMSourceIdentity(self.path, self.sample_count, self.sha256)

    @property
    def pcm_identity(self) -> PCMSourceIdentity:
        """Return the full-WAV identity accepted by the secure PCM loader."""

        return PCMSourceIdentity(self.path, self.sample_count, self.sha256)

    def window(self, start_sample: int) -> WindowExample:
        """Create one full registered window beginning at ``start_sample``."""

        _require_nonnegative_integer("start_sample", start_sample)
        if start_sample + WINDOW_SAMPLES > self.sample_count:
            raise Experiment002DataError(
                f"background window exceeds {self.path!r}: start={start_sample}"
            )
        identity = encode_window_identity(self.path, start_sample)
        return WindowExample(
            source=self,
            start_sample=start_sample,
            label="silence",
            label_index=CLASS_ORDER.index("silence"),
            identity=identity,
        )


@dataclass(frozen=True, slots=True)
class CommandExample:
    """One labelled command example in a deterministic population plan."""

    source: CommandSource
    label: str
    label_index: int
    identity: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.source, CommandSource):
            raise TypeError("source must be a CommandSource")
        if self.label != self.source.label:
            raise Experiment002DataError(
                "command example label differs from its source"
            )
        if self.label not in CLASS_ORDER[:-1]:
            raise Experiment002DataError("command example has an invalid label")
        if self.label_index != CLASS_ORDER.index(self.label):
            raise Experiment002DataError("command example label_index is invalid")
        if self.identity != self.source.identity:
            raise Experiment002DataError("command example identity is invalid")


@dataclass(frozen=True, slots=True)
class WindowExample:
    """One labelled one-second window from a background recording."""

    source: BackgroundSource
    start_sample: int
    label: WindowLabel
    label_index: int
    identity: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.source, BackgroundSource):
            raise TypeError("source must be a BackgroundSource")
        _require_nonnegative_integer("start_sample", self.start_sample)
        if self.start_sample + WINDOW_SAMPLES > self.source.sample_count:
            raise Experiment002DataError("window falls outside its background source")
        if self.label != "silence":
            raise Experiment002DataError("background window label must be 'silence'")
        if self.label_index != CLASS_ORDER.index("silence"):
            raise Experiment002DataError("background window label_index is invalid")
        expected = encode_window_identity(self.source.path, self.start_sample)
        if self.identity != expected:
            raise Experiment002DataError("background window identity is invalid")


type Example = CommandExample | WindowExample


@dataclass(frozen=True, slots=True)
class Experiment002Corpus:
    """Metadata-only corpus with no Speech Commands test identities."""

    train_commands: tuple[CommandSource, ...]
    validation_commands: tuple[CommandSource, ...]
    train_backgrounds: tuple[BackgroundSource, ...]
    validation_background: BackgroundSource
    test_command_count: int
    manifest_sha256: str
    _train_command_sources: frozenset[CommandSource] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _train_inventory_sha256: str = field(init=False, repr=False, compare=False)
    _validation_inventory_sha256: str = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _require_tuple_of("train_commands", self.train_commands, CommandSource)
        _require_tuple_of(
            "validation_commands", self.validation_commands, CommandSource
        )
        _require_tuple_of("train_backgrounds", self.train_backgrounds, BackgroundSource)
        if not self.train_backgrounds:
            raise Experiment002DataError("train_backgrounds must not be empty")
        if not isinstance(self.validation_background, BackgroundSource):
            raise TypeError("validation_background must be a BackgroundSource")
        _require_nonnegative_integer("test_command_count", self.test_command_count)
        _validate_sha256(self.manifest_sha256)

        commands = (*self.train_commands, *self.validation_commands)
        command_paths = [source.path for source in commands]
        command_indices = [source.manifest_index for source in commands]
        if len(command_paths) != len(set(command_paths)):
            raise Experiment002DataError("corpus repeats a command path")
        if len(command_indices) != len(set(command_indices)):
            raise Experiment002DataError("corpus repeats a command manifest index")
        background_paths = [
            *(source.path for source in self.train_backgrounds),
            self.validation_background.path,
        ]
        if len(background_paths) != len(set(background_paths)):
            raise Experiment002DataError("corpus repeats a background path")
        if set(command_paths) & set(background_paths):
            raise Experiment002DataError("command and background paths overlap")
        object.__setattr__(
            self,
            "_train_command_sources",
            frozenset(self.train_commands),
        )
        object.__setattr__(
            self,
            "_train_inventory_sha256",
            _command_inventory_sha256(self.train_commands),
        )
        object.__setattr__(
            self,
            "_validation_inventory_sha256",
            _command_inventory_sha256(self.validation_commands),
        )


class TrainingWindowUniverse:
    """Lazy path-major universe of all training background windows."""

    def __init__(self, backgrounds: Sequence[BackgroundSource]) -> None:
        if not isinstance(backgrounds, Sequence) or isinstance(
            backgrounds, (str, bytes)
        ):
            raise TypeError("backgrounds must be a sequence")
        materialized = tuple(backgrounds)
        if not materialized:
            raise Experiment002DataError("backgrounds must not be empty")
        if any(not isinstance(source, BackgroundSource) for source in materialized):
            raise TypeError("every background must be a BackgroundSource")
        ordered = tuple(sorted(materialized, key=lambda source: _utf8_key(source.path)))
        paths = [source.path for source in ordered]
        if len(paths) != len(set(paths)):
            raise Experiment002DataError("background universe repeats a path")

        cumulative: list[int] = []
        size = 0
        for source in ordered:
            size += source.sample_count - WINDOW_SAMPLES + 1
            cumulative.append(size)
        self._backgrounds = ordered
        self._cumulative = tuple(cumulative)
        self._size = size

    @property
    def size(self) -> int:
        """Return the exact number of path/start pairs in the universe."""

        return self._size

    def at(self, index: int) -> WindowExample:
        """Map a flat duration-weighted index to its unique window."""

        _require_nonnegative_integer("index", index)
        if index >= self._size:
            raise IndexError(f"window index {index} is outside [0, {self._size})")
        source_index = bisect_right(self._cumulative, index)
        previous_end = 0 if source_index == 0 else self._cumulative[source_index - 1]
        return self._backgrounds[source_index].window(index - previous_end)

    def __iter__(self) -> Iterator[WindowExample]:
        for source in self._backgrounds:
            last_start = source.sample_count - WINDOW_SAMPLES
            for start_sample in range(last_start + 1):
                yield source.window(start_sample)


@dataclass(frozen=True, slots=True)
class _BackgroundExpectation:
    split: Split
    source: BackgroundSource


@dataclass(frozen=True, slots=True)
class _DataContract:
    manifest_sha256: str
    train_command_count: int
    validation_command_count: int
    test_command_count: int
    train_target_support: tuple[tuple[str, int], ...]
    validation_target_support: tuple[tuple[str, int], ...]
    train_unknown_count: int
    validation_unknown_count: int
    unknown_words: tuple[str, ...]
    backgrounds: tuple[_BackgroundExpectation, ...]
    test_background_count: int


def load_registered_corpus(manifest: Path) -> Experiment002Corpus:
    """Load the one registered manifest without retaining any test identity."""

    return _load_corpus(manifest, _REGISTERED_CONTRACT)


def _load_corpus(manifest: Path, contract: _DataContract) -> Experiment002Corpus:
    if not isinstance(contract, _DataContract):
        raise TypeError("contract must be a _DataContract")
    contents = _read_manifest(manifest)
    observed_sha256 = hashlib.sha256(contents).hexdigest()
    if not hmac.compare_digest(observed_sha256, contract.manifest_sha256):
        raise Experiment002DataError(
            "manifest digest differs from the registered contract: "
            f"observed={observed_sha256}"
        )

    train_commands: list[CommandSource] = []
    validation_commands: list[CommandSource] = []
    split_counts: Counter[str] = Counter()
    train_support: Counter[str] = Counter()
    validation_support: Counter[str] = Counter()
    train_unknown_words: Counter[str] = Counter()
    seen_paths: set[str] = set()
    speaker_splits: dict[str, Split] = {}
    observed_backgrounds: dict[str, tuple[Split, BackgroundSource]] = {}
    test_background_count = 0
    previous_command_key: bytes | None = None
    previous_background_key: bytes | None = None
    saw_background = False
    command_index = 0

    if b"\r" in contents:
        raise Experiment002DataError("manifest must use LF line endings only")
    if not contents.endswith(b"\n"):
        raise Experiment002DataError("manifest must end with one LF")
    raw_lines = contents[:-1].split(b"\n")
    if not raw_lines or any(not line for line in raw_lines):
        raise Experiment002DataError("manifest contains an empty row")

    for line_number, raw_line in enumerate(raw_lines, start=1):
        row = _parse_json_object(raw_line, line_number)
        kind = _string_field(row, "kind", line_number)
        if kind == "command":
            if saw_background:
                raise Experiment002DataError(
                    f"manifest line {line_number}: command follows a background row"
                )
            _require_exact_keys(row, _COMMAND_KEYS, line_number)
            parsed = _parse_command_row(row, line_number, contract.unknown_words)
            (
                path,
                split,
                word,
                label,
                speaker_id,
                sample_count,
                sha256,
            ) = parsed
            path_key = _utf8_key(path)
            if previous_command_key is not None and path_key <= previous_command_key:
                raise Experiment002DataError(
                    f"manifest line {line_number}: command paths are not strictly "
                    "UTF-8 byte ordered"
                )
            previous_command_key = path_key
            _add_unique_path(path, seen_paths, line_number)
            previous_split = speaker_splits.setdefault(speaker_id, split)
            if previous_split != split:
                raise Experiment002DataError(
                    f"manifest line {line_number}: speaker crosses splits"
                )

            split_counts[split] += 1
            if split == "train":
                train_support[label] += 1
                if label == "unknown":
                    train_unknown_words[word] += 1
                train_commands.append(
                    CommandSource(
                        manifest_index=command_index,
                        path=path,
                        word=word,
                        label=label,
                        sample_count=sample_count,
                        sha256=sha256,
                    )
                )
            elif split == "validation":
                validation_support[label] += 1
                validation_commands.append(
                    CommandSource(
                        manifest_index=command_index,
                        path=path,
                        word=word,
                        label=label,
                        sample_count=sample_count,
                        sha256=sha256,
                    )
                )
            command_index += 1
        elif kind == "background":
            saw_background = True
            _require_exact_keys(row, _BACKGROUND_KEYS, line_number)
            split, source = _parse_background_row(row, line_number)
            path_key = _utf8_key(source.path)
            if (
                previous_background_key is not None
                and path_key <= previous_background_key
            ):
                raise Experiment002DataError(
                    f"manifest line {line_number}: background paths are not strictly "
                    "UTF-8 byte ordered"
                )
            previous_background_key = path_key
            _add_unique_path(source.path, seen_paths, line_number)
            if split == "test":
                test_background_count += 1
            else:
                observed_backgrounds[source.path] = (split, source)
        else:
            raise Experiment002DataError(
                f"manifest line {line_number}: invalid kind {kind!r}"
            )

    expected_counts = {
        "train": contract.train_command_count,
        "validation": contract.validation_command_count,
        "test": contract.test_command_count,
    }
    if dict(split_counts) != expected_counts:
        raise Experiment002DataError(
            f"manifest command split counts differ: observed={dict(split_counts)}"
        )
    if command_index != sum(expected_counts.values()):
        raise Experiment002DataError("manifest command count differs")
    _require_support(
        "training target",
        train_support,
        contract.train_target_support,
        unknown_count=contract.train_unknown_count,
    )
    _require_support(
        "validation target",
        validation_support,
        contract.validation_target_support,
        unknown_count=contract.validation_unknown_count,
    )
    if set(train_unknown_words) != set(contract.unknown_words):
        raise Experiment002DataError("training unknown source-word inventory differs")
    _require_backgrounds(observed_backgrounds, contract.backgrounds)
    if test_background_count != contract.test_background_count:
        raise Experiment002DataError(
            "test background count differs from the registered contract"
        )

    train_backgrounds = tuple(
        expectation.source
        for expectation in contract.backgrounds
        if expectation.split == "train"
    )
    validation_backgrounds = tuple(
        expectation.source
        for expectation in contract.backgrounds
        if expectation.split == "validation"
    )
    if len(validation_backgrounds) != 1:
        raise Experiment002DataError(
            "contract must contain exactly one validation background"
        )
    return Experiment002Corpus(
        train_commands=tuple(train_commands),
        validation_commands=tuple(validation_commands),
        train_backgrounds=train_backgrounds,
        validation_background=validation_backgrounds[0],
        test_command_count=split_counts["test"],
        manifest_sha256=contract.manifest_sha256,
    )


def build_training_epoch(
    corpus: Experiment002Corpus, *, seed: int, epoch: int
) -> tuple[Example, ...]:
    """Build the exact 40,027-example registered training epoch."""

    _require_registered_training_corpus(corpus)
    _validate_registered_training_draw(seed, epoch)
    targets = [
        _command_example(source)
        for source in sorted(corpus.train_commands, key=_command_source_order)
        if source.label != "unknown"
    ]

    selected_unknown = list(
        _select_unknown_commands(corpus.train_commands, seed=seed, epoch=epoch)
    )

    universe = TrainingWindowUniverse(corpus.train_backgrounds)
    selected_silence = _select_silence_windows(
        universe,
        count=TRAIN_SILENCE_COUNT,
        seed=seed,
        epoch=epoch,
    )
    examples: list[Example] = [*targets, *selected_unknown, *selected_silence]
    if (
        len(targets) != TRAIN_TARGET_COUNT
        or len(selected_unknown) != TRAIN_UNKNOWN_COUNT
        or len(selected_silence) != TRAIN_SILENCE_COUNT
        or len(examples) != TRAIN_EXAMPLE_COUNT
    ):
        raise Experiment002DataError("training epoch population count differs")
    return _order_training_examples(examples, seed=seed, epoch=epoch)


def build_validation_population(
    corpus: Experiment002Corpus,
) -> tuple[Example, ...]:
    """Build the fixed validation commands followed by running-tap windows."""

    _require_registered_validation_corpus(corpus)
    commands = tuple(
        _command_example(source)
        for source in sorted(corpus.validation_commands, key=_command_source_order)
    )
    source = corpus.validation_background
    windows = tuple(
        source.window(start_sample)
        for start_sample in range(
            0,
            source.sample_count - WINDOW_SAMPLES + 1,
            VALIDATION_HOP_SAMPLES,
        )
    )
    population: tuple[Example, ...] = (*commands, *windows)
    if len(windows) != VALIDATION_SILENCE_COUNT or len(population) != (
        VALIDATION_EXAMPLE_COUNT
    ):
        raise Experiment002DataError("validation population count differs")
    return population


def select_command_noise_window(
    corpus: Experiment002Corpus,
    command: CommandSource,
    *,
    seed: int,
    epoch: int,
) -> WindowExample:
    """Map one training command draw into the full duration-weighted universe."""

    _require_registered_window_universe(corpus)
    if not isinstance(command, CommandSource):
        raise TypeError("command must be a CommandSource")
    if command not in corpus._train_command_sources:
        raise Experiment002DataError(
            "noise windows may be selected only for a corpus training command"
        )
    _validate_registered_training_draw(seed, epoch)
    universe = TrainingWindowUniverse(corpus.train_backgrounds)
    index = uniform_integer(
        seed=seed,
        epoch=epoch,
        identity=command.identity,
        domain="command-noise-window",
        bound=universe.size,
        counter=0,
    )
    return universe.at(index)


def pad_command_waveform(
    example: CommandExample, snapshot: VerifiedPCM16LE
) -> FloatArray:
    """Decode and right-pad one identity-bound command with positive float32 zero."""

    if not isinstance(example, CommandExample):
        raise TypeError("example must be a CommandExample")
    _require_matching_snapshot(example.source, snapshot)
    waveform = snapshot.to_float32()
    if waveform.shape != (example.source.sample_count,):
        raise Experiment002DataError("decoded command has an invalid shape")
    padded = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    padded[: waveform.size] = waveform
    return padded


def slice_background_window(
    example: WindowExample, snapshot: VerifiedPCM16LE
) -> FloatArray:
    """Decode only one identity-bound PCM window into an owned float32 array."""

    if not isinstance(example, WindowExample):
        raise TypeError("example must be a WindowExample")
    _require_matching_snapshot(example.source, snapshot)
    start_byte = 2 * example.start_sample
    end_byte = start_byte + 2 * WINDOW_SAMPLES
    if start_byte < 0 or end_byte > len(snapshot.payload):
        raise Experiment002DataError("background window falls outside PCM payload")
    waveform = pcm16le_to_float32(snapshot.payload[start_byte:end_byte])
    if waveform.dtype != np.dtype(np.float32) or waveform.shape != (WINDOW_SAMPLES,):
        raise Experiment002DataError("decoded background window has an invalid shape")
    return np.ascontiguousarray(waveform, dtype=np.float32)


def _rank_unknown_words(
    words: Iterable[str], *, seed: int, epoch: int
) -> tuple[str, ...]:
    _validate_seed_epoch(seed, epoch)
    materialized = tuple(words)
    if any(not isinstance(word, str) or not word for word in materialized):
        raise TypeError("unknown words must be non-empty strings")
    if len(materialized) != len(set(materialized)):
        raise Experiment002DataError(
            "unknown source-word inventory contains duplicates"
        )
    return tuple(
        sorted(
            materialized,
            key=lambda word: rank_key(
                seed=seed,
                epoch=epoch,
                identity=encode_source_word_identity(word),
                domain="unknown-word-rank",
            ),
        )
    )


def _select_unknown_commands(
    commands: Iterable[CommandSource], *, seed: int, epoch: int
) -> tuple[CommandExample, ...]:
    """Select the exact balanced 6,172-example unknown population."""

    _validate_seed_epoch(seed, epoch)
    unknown_by_word: dict[str, list[CommandSource]] = defaultdict(list)
    seen_paths: set[str] = set()
    for source in commands:
        if not isinstance(source, CommandSource):
            raise TypeError("commands must contain only CommandSource values")
        if source.label != "unknown":
            continue
        if source.path in seen_paths:
            raise Experiment002DataError("unknown command population repeats a path")
        seen_paths.add(source.path)
        unknown_by_word[source.word].append(source)
    if set(unknown_by_word) != set(UNKNOWN_WORDS):
        raise Experiment002DataError("unknown source-word inventory differs")

    ranked_words = _rank_unknown_words(unknown_by_word, seed=seed, epoch=epoch)
    selected: list[CommandExample] = []
    for word_index, word in enumerate(ranked_words):
        allocation = 247 if word_index < 22 else 246
        ranked_sources = sorted(
            unknown_by_word[word],
            key=lambda source: rank_key(
                seed=seed,
                epoch=epoch,
                identity=source.identity,
                domain="unknown-clip-rank",
            ),
        )
        if len(ranked_sources) < allocation:
            raise Experiment002DataError(
                f"unknown source word {word!r} has too few clips"
            )
        selected.extend(
            _command_example(source) for source in ranked_sources[:allocation]
        )
    if len(selected) != TRAIN_UNKNOWN_COUNT:
        raise Experiment002DataError("unknown training selection count differs")
    return tuple(selected)


def _order_training_examples(
    examples: Iterable[Example], *, seed: int, epoch: int
) -> tuple[Example, ...]:
    """Apply the registered digest/identity training order without shared RNG."""

    _validate_seed_epoch(seed, epoch)
    materialized = tuple(examples)
    if any(
        not isinstance(example, (CommandExample, WindowExample))
        for example in materialized
    ):
        raise TypeError("examples contain an invalid value")
    identities = [example.identity for example in materialized]
    if len(identities) != len(set(identities)):
        raise Experiment002DataError("training examples repeat an identity")
    return tuple(
        sorted(
            materialized,
            key=lambda example: rank_key(
                seed=seed,
                epoch=epoch,
                identity=example.identity,
                domain="train-order",
            ),
        )
    )


def _select_silence_windows(
    universe: TrainingWindowUniverse,
    *,
    count: int,
    seed: int,
    epoch: int,
) -> tuple[WindowExample, ...]:
    if not isinstance(universe, TrainingWindowUniverse):
        raise TypeError("universe must be a TrainingWindowUniverse")
    _require_nonnegative_integer("count", count)
    if count < 1 or count > universe.size:
        raise Experiment002DataError("silence selection count is outside the universe")
    _validate_seed_epoch(seed, epoch)
    selected = heapq.nsmallest(
        count,
        universe,
        key=lambda example: rank_key(
            seed=seed,
            epoch=epoch,
            identity=example.identity,
            domain="silence-window-rank",
        ),
    )
    return tuple(selected)


def _read_manifest(manifest: Path) -> bytes:
    if not isinstance(manifest, Path):
        raise TypeError("manifest must be a Path")
    required_flags = ("O_CLOEXEC", "O_NOFOLLOW", "O_NONBLOCK")
    if any(not hasattr(os, name) for name in required_flags):
        raise Experiment002DataError("secure manifest open flags are unavailable")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
    try:
        descriptor = os.open(manifest, flags)
    except (OSError, ValueError) as error:
        raise Experiment002DataError(
            f"cannot securely open manifest: {error}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise Experiment002DataError("manifest is not a regular file")
        if before.st_size > MAX_MANIFEST_BYTES:
            raise Experiment002DataError("manifest exceeds the 64 MiB limit")
        with os.fdopen(descriptor, "rb", closefd=True) as source:
            descriptor = -1
            contents = source.read(MAX_MANIFEST_BYTES + 1)
            after = os.fstat(source.fileno())
    except OSError as error:
        raise Experiment002DataError(f"cannot read manifest: {error}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(contents) > MAX_MANIFEST_BYTES:
        raise Experiment002DataError("manifest exceeds the 64 MiB limit while reading")
    if len(contents) != before.st_size or _stat_identity(before) != _stat_identity(
        after
    ):
        raise Experiment002DataError("manifest changed while reading")
    return contents


def _parse_json_object(raw_line: bytes, line_number: int) -> dict[str, object]:
    try:
        text = raw_line.decode("utf-8")
    except UnicodeDecodeError as error:
        raise Experiment002DataError(
            f"manifest line {line_number}: invalid UTF-8"
        ) from error
    try:
        value = cast(
            object,
            json.loads(
                text,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            ),
        )
    except (Experiment002DataError, ValueError) as error:
        raise Experiment002DataError(
            f"manifest line {line_number}: invalid JSON: {error}"
        ) from error
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise Experiment002DataError(
            f"manifest line {line_number}: row must be a JSON object"
        )
    return cast(dict[str, object], value)


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise Experiment002DataError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> NoReturn:
    raise Experiment002DataError(f"non-standard JSON constant {value!r}")


def _parse_command_row(
    row: dict[str, object], line_number: int, unknown_words: tuple[str, ...]
) -> tuple[str, Split, str, str, str, int, str]:
    path = _string_field(row, "path", line_number)
    parts = _canonical_path_parts(path)
    if parts[0] == "_background_noise_":
        raise Experiment002DataError(
            f"manifest line {line_number}: command uses the background directory"
        )
    split = _split_field(row, line_number)
    word = _string_field(row, "word", line_number)
    if word != parts[0]:
        raise Experiment002DataError(
            f"manifest line {line_number}: word differs from the path"
        )
    if word not in TARGET_WORDS and word not in unknown_words:
        raise Experiment002DataError(
            f"manifest line {line_number}: unregistered source word {word!r}"
        )
    label = _string_field(row, "label", line_number)
    expected_label = word if word in TARGET_WORDS else "unknown"
    if label != expected_label:
        raise Experiment002DataError(
            f"manifest line {line_number}: label differs from the registered mapping"
        )
    speaker_id = _string_field(row, "speaker_id", line_number)
    if not speaker_id or "\0" in speaker_id:
        raise Experiment002DataError(f"manifest line {line_number}: invalid speaker_id")
    try:
        speaker_id.encode("utf-8")
    except UnicodeEncodeError as error:
        raise Experiment002DataError(
            f"manifest line {line_number}: speaker_id is not valid UTF-8"
        ) from error
    utterance_index = _integer_field(row, "utterance_index", line_number)
    if utterance_index < 0:
        raise Experiment002DataError(
            f"manifest line {line_number}: utterance_index is negative"
        )
    expected_name = f"{speaker_id}_nohash_{utterance_index}.wav"
    if parts[1] != expected_name:
        raise Experiment002DataError(
            f"manifest line {line_number}: speaker or utterance differs from path"
        )
    sample_count = _integer_field(row, "sample_count", line_number)
    if not 1 <= sample_count <= WINDOW_SAMPLES:
        raise Experiment002DataError(
            f"manifest line {line_number}: command sample_count is outside [1, 16000]"
        )
    sha256 = _string_field(row, "sha256", line_number)
    _validate_sha256(sha256)
    return path, split, word, label, speaker_id, sample_count, sha256


def _parse_background_row(
    row: dict[str, object], line_number: int
) -> tuple[Split, BackgroundSource]:
    path = _string_field(row, "path", line_number)
    sample_count = _integer_field(row, "sample_count", line_number)
    sha256 = _string_field(row, "sha256", line_number)
    split = _split_field(row, line_number)
    try:
        source = BackgroundSource(path, sample_count, sha256)
    except (TypeError, ValueError) as error:
        raise Experiment002DataError(
            f"manifest line {line_number}: invalid background: {error}"
        ) from error
    return split, source


def _require_exact_keys(
    row: dict[str, object], expected: frozenset[str], line_number: int
) -> None:
    observed = frozenset(row)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise Experiment002DataError(
            f"manifest line {line_number}: row schema differs; "
            f"missing={missing}, extra={extra}"
        )


def _string_field(row: dict[str, object], key: str, line_number: int) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise Experiment002DataError(
            f"manifest line {line_number}: {key} must be a string"
        )
    return value


def _integer_field(row: dict[str, object], key: str, line_number: int) -> int:
    value = row.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise Experiment002DataError(
            f"manifest line {line_number}: {key} must be an integer"
        )
    return value


def _split_field(row: dict[str, object], line_number: int) -> Split:
    value = _string_field(row, "split", line_number)
    if value not in {"train", "validation", "test"}:
        raise Experiment002DataError(
            f"manifest line {line_number}: invalid split {value!r}"
        )
    return cast(Split, value)


def _canonical_path_parts(path: str) -> tuple[str, str]:
    if not isinstance(path, str):
        raise TypeError("path must be a string")
    try:
        path.encode("utf-8")
    except UnicodeEncodeError as error:
        raise Experiment002DataError("path must encode as valid UTF-8") from error
    parsed = PurePosixPath(path)
    if (
        not path
        or "\\" in path
        or "\0" in path
        or parsed.is_absolute()
        or len(parsed.parts) != 2
        or any(part in {"", ".", ".."} for part in parsed.parts)
        or parsed.as_posix() != path
        or parsed.suffix != ".wav"
    ):
        raise Experiment002DataError(f"path is not canonical: {path!r}")
    return parsed.parts[0], parsed.parts[1]


def _validate_command_fields(
    *, path: str, word: str, label: str, sample_count: int, sha256: str
) -> None:
    parts = _canonical_path_parts(path)
    if parts[0] == "_background_noise_":
        raise Experiment002DataError("command path uses the background directory")
    if not isinstance(word, str):
        raise TypeError("word must be a string")
    if word != parts[0]:
        raise Experiment002DataError("command word differs from its path")
    if word not in TARGET_WORDS and word not in UNKNOWN_WORDS:
        raise Experiment002DataError(f"unregistered command word {word!r}")
    expected_label = word if word in TARGET_WORDS else "unknown"
    if label != expected_label:
        raise Experiment002DataError("command label differs from its word")
    if not isinstance(sample_count, int) or isinstance(sample_count, bool):
        raise TypeError("sample_count must be an integer")
    if not 1 <= sample_count <= WINDOW_SAMPLES:
        raise Experiment002DataError("command sample_count is outside [1, 16000]")
    _validate_sha256(sha256)
    PCMSourceIdentity(path, sample_count, sha256)


def _validate_sha256(value: str) -> None:
    if not isinstance(value, str):
        raise TypeError("sha256 must be a string")
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise Experiment002DataError("sha256 must be a lowercase SHA-256 digest")


def _add_unique_path(path: str, seen: set[str], line_number: int) -> None:
    if path in seen:
        raise Experiment002DataError(
            f"manifest line {line_number}: duplicate path {path!r}"
        )
    seen.add(path)


def _require_support(
    name: str,
    observed: Counter[str],
    target_support: tuple[tuple[str, int], ...],
    *,
    unknown_count: int,
) -> None:
    expected = dict(target_support)
    expected["unknown"] = unknown_count
    if dict(observed) != expected:
        raise Experiment002DataError(
            f"{name} support differs: observed={dict(observed)}"
        )


def _require_backgrounds(
    observed: dict[str, tuple[Split, BackgroundSource]],
    expected: tuple[_BackgroundExpectation, ...],
) -> None:
    expected_by_path = {
        item.source.path: (item.split, item.source) for item in expected
    }
    if observed != expected_by_path:
        raise Experiment002DataError("background inventory or identity differs")


def _require_registered_training_corpus(corpus: Experiment002Corpus) -> None:
    if not isinstance(corpus, Experiment002Corpus):
        raise TypeError("corpus must be an Experiment002Corpus")
    if corpus.manifest_sha256 != MANIFEST_SHA256:
        raise Experiment002DataError("corpus is not bound to the registered manifest")
    if corpus.test_command_count != _REGISTERED_CONTRACT.test_command_count:
        raise Experiment002DataError("test command count differs")
    if corpus._train_inventory_sha256 != TRAIN_COMMAND_INVENTORY_SHA256:
        raise Experiment002DataError("training command inventory differs")
    expected_backgrounds = tuple(
        item.source
        for item in _REGISTERED_CONTRACT.backgrounds
        if item.split == "train"
    )
    if set(corpus.train_backgrounds) != set(expected_backgrounds):
        raise Experiment002DataError("training background identities differ")
    support = Counter(source.label for source in corpus.train_commands)
    expected_support = dict(_REGISTERED_CONTRACT.train_target_support)
    expected_support["unknown"] = TRAIN_UNKNOWN_SOURCE_COUNT
    if dict(support) != expected_support:
        raise Experiment002DataError("training command support differs")
    unknown_words = {
        source.word for source in corpus.train_commands if source.label == "unknown"
    }
    if unknown_words != set(UNKNOWN_WORDS):
        raise Experiment002DataError("training unknown source-word inventory differs")
    universe = TrainingWindowUniverse(corpus.train_backgrounds)
    if universe.size != TRAIN_WINDOW_UNIVERSE_SIZE:
        raise Experiment002DataError("training window universe size differs")


def _require_registered_window_universe(corpus: Experiment002Corpus) -> None:
    if not isinstance(corpus, Experiment002Corpus):
        raise TypeError("corpus must be an Experiment002Corpus")
    if corpus.manifest_sha256 != MANIFEST_SHA256:
        raise Experiment002DataError("corpus is not bound to the registered manifest")
    if corpus.test_command_count != _REGISTERED_CONTRACT.test_command_count:
        raise Experiment002DataError("test command count differs")
    if corpus._train_inventory_sha256 != TRAIN_COMMAND_INVENTORY_SHA256:
        raise Experiment002DataError("training command inventory differs")
    expected_backgrounds = tuple(
        item.source
        for item in _REGISTERED_CONTRACT.backgrounds
        if item.split == "train"
    )
    if set(corpus.train_backgrounds) != set(expected_backgrounds):
        raise Experiment002DataError("training background identities differ")
    if TrainingWindowUniverse(corpus.train_backgrounds).size != (
        TRAIN_WINDOW_UNIVERSE_SIZE
    ):
        raise Experiment002DataError("training window universe size differs")


def _require_registered_validation_corpus(corpus: Experiment002Corpus) -> None:
    if not isinstance(corpus, Experiment002Corpus):
        raise TypeError("corpus must be an Experiment002Corpus")
    if corpus.manifest_sha256 != MANIFEST_SHA256:
        raise Experiment002DataError("corpus is not bound to the registered manifest")
    if corpus.test_command_count != _REGISTERED_CONTRACT.test_command_count:
        raise Experiment002DataError("test command count differs")
    if corpus._validation_inventory_sha256 != (VALIDATION_COMMAND_INVENTORY_SHA256):
        raise Experiment002DataError("validation command inventory differs")
    expected_background = next(
        item.source
        for item in _REGISTERED_CONTRACT.backgrounds
        if item.split == "validation"
    )
    if corpus.validation_background != expected_background:
        raise Experiment002DataError("validation background identity differs")
    support = Counter(source.label for source in corpus.validation_commands)
    expected_support = dict(_REGISTERED_CONTRACT.validation_target_support)
    expected_support["unknown"] = VALIDATION_UNKNOWN_COUNT
    if dict(support) != expected_support:
        raise Experiment002DataError("validation command support differs")


def _command_example(source: CommandSource) -> CommandExample:
    return CommandExample(
        source=source,
        label=source.label,
        label_index=CLASS_ORDER.index(source.label),
        identity=source.identity,
    )


def _command_source_order(source: CommandSource) -> bytes:
    return _utf8_key(source.path)


def _command_inventory_sha256(sources: Iterable[CommandSource]) -> str:
    hasher = hashlib.sha256(_COMMAND_INVENTORY_PREFIX)
    ordered = sorted(sources, key=lambda source: source.manifest_index)
    for source in ordered:
        hasher.update(struct.pack("<I", source.manifest_index))
        for value in (source.path, source.word, source.label):
            encoded = value.encode("utf-8")
            hasher.update(struct.pack("<I", len(encoded)))
            hasher.update(encoded)
        hasher.update(struct.pack("<I", source.sample_count))
        hasher.update(bytes.fromhex(source.sha256))
    return hasher.hexdigest()


def _utf8_key(value: str) -> bytes:
    return value.encode("utf-8")


def _require_matching_snapshot(
    source: CommandSource | BackgroundSource, snapshot: VerifiedPCM16LE
) -> None:
    if not isinstance(snapshot, VerifiedPCM16LE):
        raise TypeError("snapshot must be a VerifiedPCM16LE")
    if (
        snapshot.path != source.path
        or snapshot.sample_count != source.sample_count
        or not hmac.compare_digest(snapshot.sha256, source.sha256)
    ):
        raise Experiment002DataError("PCM snapshot identity differs from the example")


def _require_tuple_of(name: str, values: object, expected_type: type[object]) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple")
    if any(not isinstance(value, expected_type) for value in values):
        raise TypeError(f"{name} contains an invalid value")


def _require_nonnegative_integer(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise Experiment002DataError(f"{name} must be nonnegative")


def _validate_seed_epoch(seed: int, epoch: int) -> None:
    _require_nonnegative_integer("seed", seed)
    _require_nonnegative_integer("epoch", epoch)
    if seed > UINT64_MAX:
        raise Experiment002DataError("seed exceeds uint64")
    if epoch > UINT32_MAX:
        raise Experiment002DataError("epoch exceeds uint32")


def _validate_registered_training_draw(seed: int, epoch: int) -> None:
    _validate_seed_epoch(seed, epoch)
    if seed not in TRAINING_SEEDS:
        raise Experiment002DataError("seed is not registered for training")
    if epoch >= TRAIN_EPOCH_COUNT:
        raise Experiment002DataError(
            f"training epoch must be within [0, {TRAIN_EPOCH_COUNT})"
        )


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


_REGISTERED_CONTRACT: Final = _DataContract(
    manifest_sha256=MANIFEST_SHA256,
    train_command_count=84_843,
    validation_command_count=9_981,
    test_command_count=11_005,
    train_target_support=(
        ("yes", 3_228),
        ("no", 3_130),
        ("up", 2_948),
        ("down", 3_134),
        ("left", 3_037),
        ("right", 3_019),
        ("on", 3_086),
        ("off", 2_970),
        ("stop", 3_111),
        ("go", 3_106),
    ),
    validation_target_support=(
        ("yes", 397),
        ("no", 406),
        ("up", 350),
        ("down", 377),
        ("left", 352),
        ("right", 363),
        ("on", 363),
        ("off", 373),
        ("stop", 350),
        ("go", 372),
    ),
    train_unknown_count=TRAIN_UNKNOWN_SOURCE_COUNT,
    validation_unknown_count=VALIDATION_UNKNOWN_COUNT,
    unknown_words=UNKNOWN_WORDS,
    backgrounds=(
        _BackgroundExpectation(
            "train",
            BackgroundSource(
                "_background_noise_/doing_the_dishes.wav",
                1_522_930,
                "099eafcd7c4c266612012b5622b97157042e5288acaf4cbf3dc475814bfcdaa8",
            ),
        ),
        _BackgroundExpectation(
            "train",
            BackgroundSource(
                "_background_noise_/dude_miaowing.wav",
                988_891,
                "1acd62f115d4c3f9daca9c5ec0c2e0c3e174a2a08be30200003a51de52b288ff",
            ),
        ),
        _BackgroundExpectation(
            "train",
            BackgroundSource(
                "_background_noise_/exercise_bike.wav",
                980_062,
                "e453813ed45b2f9f81d5600ec3ac2a5e22d3f2c947fe6f0a3ff1f3d6018c014d",
            ),
        ),
        _BackgroundExpectation(
            "train",
            BackgroundSource(
                "_background_noise_/pink_noise.wav",
                960_000,
                "b6e038c83fb342e39267d4fe69663f76ef0c1121ff60232e7b79171c00318cd6",
            ),
        ),
        _BackgroundExpectation(
            "validation",
            BackgroundSource(
                "_background_noise_/running_tap.wav",
                978_488,
                "c199c5fd61f5bf9fd57f1c346c8a51c0c035d63d176348dd9f0f7d671d7eb8eb",
            ),
        ),
    ),
    test_background_count=1,
)
