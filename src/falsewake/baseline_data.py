"""Deterministic clip sampling for experiment 000."""

from __future__ import annotations

import hashlib
import hmac
import json
import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

from falsewake.features import DEFAULT_FRONTEND
from falsewake.speech_commands import TARGET_WORDS, Split

EXPERIMENT_SEED = 20_260_718
MANIFEST_SHA256 = "d28e6993101bd6bc452033bcb7355b25e3b84ab5c51a1458cc097dc93c60f78b"
MAX_MANIFEST_BYTES = 64 * 1024 * 1024


class BaselineDataError(ValueError):
    """The audited corpus cannot support the registered sampling plan."""


@dataclass(frozen=True, slots=True)
class CommandSource:
    path: str
    split: Split
    label: str
    sample_count: int
    sha256: str


@dataclass(frozen=True, slots=True)
class BackgroundSource:
    path: str
    split: Split
    sample_count: int
    sha256: str


@dataclass(frozen=True, slots=True)
class AuditedCorpus:
    commands: tuple[CommandSource, ...]
    background: tuple[BackgroundSource, ...]
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class ClipExample:
    kind: Literal["command", "silence"]
    path: str
    split: Split
    label: str
    start_sample: int
    source_sample_count: int
    source_sha256: str


def _mapping(document: object, *, line_number: int) -> dict[str, object]:
    if type(document) is not dict:
        raise BaselineDataError(f"manifest line {line_number} is not an object")
    mapping = cast(dict[object, object], document)
    if not all(type(key) is str for key in mapping):
        raise BaselineDataError(f"manifest line {line_number} has a non-text key")
    return cast(dict[str, object], mapping)


def _text(row: dict[str, object], key: str, *, line_number: int) -> str:
    value = row.get(key)
    if type(value) is not str or not value:
        raise BaselineDataError(f"manifest line {line_number} has invalid {key}")
    return value


def _integer(row: dict[str, object], key: str, *, line_number: int) -> int:
    value = row.get(key)
    if type(value) is not int or value < 1:
        raise BaselineDataError(f"manifest line {line_number} has invalid {key}")
    return value


def _split(row: dict[str, object], *, line_number: int) -> Split:
    value = _text(row, "split", line_number=line_number)
    if value not in {"train", "validation", "test"}:
        raise BaselineDataError(f"manifest line {line_number} has invalid split")
    return cast(Split, value)


def load_audited_corpus(
    path: Path, *, expected_sha256: str = MANIFEST_SHA256
) -> AuditedCorpus:
    """Load only a byte-exact manifest produced by the source audit."""

    try:
        if path.stat().st_size > MAX_MANIFEST_BYTES:
            raise BaselineDataError("manifest exceeds the experiment size limit")
        contents = path.read_bytes()
    except OSError as error:
        raise BaselineDataError(f"cannot read manifest: {error}") from error
    observed_sha256 = hashlib.sha256(contents).hexdigest()
    if not hmac.compare_digest(observed_sha256, expected_sha256):
        raise BaselineDataError(
            f"manifest digest differs: expected={expected_sha256}, "
            f"observed={observed_sha256}"
        )

    commands: list[CommandSource] = []
    background: list[BackgroundSource] = []
    seen_paths: set[str] = set()
    for line_number, line in enumerate(contents.splitlines(), start=1):
        try:
            row = _mapping(json.loads(line), line_number=line_number)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise BaselineDataError(
                f"manifest line {line_number} is not valid JSON"
            ) from error
        kind = _text(row, "kind", line_number=line_number)
        source_path = _text(row, "path", line_number=line_number)
        if source_path in seen_paths:
            raise BaselineDataError(f"manifest repeats source path {source_path!r}")
        seen_paths.add(source_path)
        split = _split(row, line_number=line_number)
        sample_count = _integer(row, "sample_count", line_number=line_number)
        source_sha256 = _text(row, "sha256", line_number=line_number)
        if kind == "command":
            commands.append(
                CommandSource(
                    path=source_path,
                    split=split,
                    label=_text(row, "label", line_number=line_number),
                    sample_count=sample_count,
                    sha256=source_sha256,
                )
            )
        elif kind == "background":
            background.append(
                BackgroundSource(
                    path=source_path,
                    split=split,
                    sample_count=sample_count,
                    sha256=source_sha256,
                )
            )
        else:
            raise BaselineDataError(f"manifest line {line_number} has invalid kind")
    if not commands or not background:
        raise BaselineDataError("manifest omits command or background sources")
    return AuditedCorpus(
        commands=tuple(commands),
        background=tuple(background),
        manifest_sha256=observed_sha256,
    )


def _rank(seed: int, *parts: str) -> bytes:
    message = "\0".join((str(seed), *parts)).encode("utf-8")
    return hashlib.sha256(message).digest()


def build_sampling_plan(
    corpus: AuditedCorpus,
    *,
    seed: int = EXPERIMENT_SEED,
    target_words: Sequence[str] = TARGET_WORDS,
    clip_samples: int = DEFAULT_FRONTEND.clip_samples,
) -> tuple[ClipExample, ...]:
    """Keep all targets and add deterministic unknown and silence examples."""

    targets = tuple(target_words)
    if not targets or len(set(targets)) != len(targets):
        raise BaselineDataError("target_words must be non-empty and unique")
    examples: list[ClipExample] = []
    for split in ("train", "validation", "test"):
        split_commands = [record for record in corpus.commands if record.split == split]
        counts = Counter(
            record.label for record in split_commands if record.label in targets
        )
        if set(counts) != set(targets):
            raise BaselineDataError(f"{split} does not contain every target word")
        unknown_count = statistics.median_low(counts.values())
        selected_commands = [
            record for record in split_commands if record.label in targets
        ]
        unknown = [record for record in split_commands if record.label == "unknown"]
        if len(unknown) < unknown_count:
            raise BaselineDataError(f"{split} has too few unknown clips")
        selected_commands.extend(
            sorted(
                unknown,
                key=lambda record: (_rank(seed, "unknown", record.path), record.path),
            )[:unknown_count]
        )
        examples.extend(
            ClipExample(
                kind="command",
                path=record.path,
                split=record.split,
                label=record.label,
                start_sample=0,
                source_sample_count=record.sample_count,
                source_sha256=record.sha256,
            )
            for record in selected_commands
        )

        backgrounds = sorted(
            (record for record in corpus.background if record.split == split),
            key=lambda record: record.path,
        )
        if not backgrounds or any(
            record.sample_count < clip_samples for record in backgrounds
        ):
            raise BaselineDataError(f"{split} has no usable background recording")
        available_starts = sum(
            record.sample_count - clip_samples + 1 for record in backgrounds
        )
        if available_starts < unknown_count:
            raise BaselineDataError(f"{split} has too few distinct silence windows")
        used_silence: set[tuple[str, int]] = set()
        for index in range(unknown_count):
            selected: tuple[BackgroundSource, int] | None = None
            for nonce in range(10_000):
                digest = _rank(seed, "silence", split, str(index), str(nonce))
                background = backgrounds[
                    int.from_bytes(digest[:8], "big") % len(backgrounds)
                ]
                valid_starts = background.sample_count - clip_samples + 1
                start_sample = int.from_bytes(digest[8:16], "big") % valid_starts
                identity = (background.path, start_sample)
                if identity not in used_silence:
                    used_silence.add(identity)
                    selected = (background, start_sample)
                    break
            if selected is None:
                raise BaselineDataError(
                    f"cannot choose distinct {split} silence windows"
                )
            background, start_sample = selected
            examples.append(
                ClipExample(
                    kind="silence",
                    path=background.path,
                    split=split,
                    label="silence",
                    start_sample=start_sample,
                    source_sample_count=background.sample_count,
                    source_sha256=background.sha256,
                )
            )
    return tuple(
        sorted(
            examples,
            key=lambda example: (
                example.split,
                example.label,
                example.path,
                example.start_sample,
            ),
        )
    )


def sampling_summary(
    examples: Sequence[ClipExample], *, seed: int = EXPERIMENT_SEED
) -> dict[str, object]:
    """Return reviewable counts and a stable digest of the selected examples."""

    rows = b"".join(
        (
            json.dumps(asdict(example), sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        for example in examples
    )
    by_split: dict[str, dict[str, int]] = {}
    for split in ("train", "validation", "test"):
        counts = Counter(
            example.label for example in examples if example.split == split
        )
        by_split[split] = dict(sorted(counts.items()))
    return {
        "example_count": len(examples),
        "examples_sha256": hashlib.sha256(rows).hexdigest(),
        "label_counts_by_split": by_split,
        "seed": seed,
    }
