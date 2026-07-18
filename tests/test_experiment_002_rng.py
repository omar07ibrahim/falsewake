from __future__ import annotations

import ast
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict, cast

import pytest

import falsewake.experiment_002_rng as rng
from falsewake.experiment_002_rng import (
    REGISTERED_CALIBRATION_DOMAINS,
    REGISTERED_DOMAINS,
    UINT32_MAX,
    UINT64_MAX,
    bernoulli,
    digest_word,
    encode_command_identity,
    encode_source_word_identity,
    encode_window_identity,
    frame_message,
    rank_key,
    stateless_digest,
    uniform_integer,
    uniform_real,
)

SEED = 20_260_719
NON_ASCII_COMMAND = encode_command_identity("café/ёж_nohash_0.wav")
ASCII_COMMAND = encode_command_identity("yes/alice_nohash_0.wav")


class DrawIdentity(TypedDict):
    seed: int
    epoch: int
    identity: bytes


class DrawContext(DrawIdentity):
    domain: str


def test_rng_module_has_no_global_rng_or_numeric_runtime_dependency() -> None:
    source = Path("src/falsewake/experiment_002_rng.py").read_text(encoding="utf-8")
    imported_roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported_roots.update(
                alias.name.split(".", maxsplit=1)[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".", maxsplit=1)[0])

    assert imported_roots == {
        "__future__",
        "hashlib",
        "hmac",
        "math",
        "struct",
        "typing",
    }


def test_identity_encodings_are_exact_utf8_without_normalization() -> None:
    assert NON_ASCII_COMMAND.hex() == (
        "636f6d6d616e6400636166c3a92fd191d0b65f6e6f686173685f302e776176"
    )
    assert encode_window_identity(
        "_background_noise_/pink_noise.wav", 12_345
    ).hex() == (
        "77696e646f77005f6261636b67726f756e645f6e6f6973655f2f70696e6b5f"
        "6e6f6973652e776176003132333435"
    )
    assert encode_source_word_identity("marvin") == b"word\0marvin"

    composed = encode_command_identity("café/example.wav")
    decomposed = encode_command_identity("cafe\N{COMBINING ACUTE ACCENT}/example.wav")
    assert composed != decomposed


@pytest.mark.parametrize("component", ["", "contains\0nul"])
def test_invalid_identity_components_are_rejected(component: str) -> None:
    with pytest.raises(ValueError):
        encode_command_identity(component)
    with pytest.raises(ValueError):
        encode_source_word_identity(component)
    with pytest.raises(ValueError):
        encode_window_identity(component, 0)


def test_invalid_window_starts_are_rejected() -> None:
    for start_sample in (-1, UINT64_MAX + 1):
        with pytest.raises(ValueError):
            encode_window_identity("background.wav", start_sample)
    with pytest.raises(TypeError):
        encode_window_identity("background.wav", cast(int, True))
    with pytest.raises(TypeError):
        encode_command_identity(cast(str, 123))
    with pytest.raises(ValueError, match="valid UTF-8"):
        encode_command_identity("bad-\ud800-path")


def test_command_message_digest_and_word_match_independent_golden() -> None:
    message = frame_message(
        epoch=3,
        identity=NON_ASCII_COMMAND,
        domain="command-gain-db",
        counter=0,
    )
    assert len(message) == 86
    assert message.hex() == (
        "66616c736577616b652d6578703030322d763100030000001f000000636f6d6d"
        "616e6400636166c3a92fd191d0b65f6e6f686173685f302e7761760f00000063"
        "6f6d6d616e642d6761696e2d64620000000000000000"
    )
    digest = stateless_digest(
        seed=SEED,
        epoch=3,
        identity=NON_ASCII_COMMAND,
        domain="command-gain-db",
    )
    assert digest.hex() == (
        "35d5a6b1b8639bd44a4804b7f95ce6561b04e248b0b255ab0a6b7c4f44e59245"
    )
    assert (
        digest_word(
            seed=SEED,
            epoch=3,
            identity=NON_ASCII_COMMAND,
            domain="command-gain-db",
        )
        == 15_319_948_202_336_507_189
    )


def test_window_and_source_word_digests_match_independent_goldens() -> None:
    window = encode_window_identity("_background_noise_/pink_noise.wav", 12_345)
    assert stateless_digest(
        seed=SEED,
        epoch=7,
        identity=window,
        domain="silence-window-rank",
    ).hex() == ("1810de3e899b39c9dfcc6ad30c8933d0855f2eaa879e8c30645a70d221dd7a2b")
    assert stateless_digest(
        seed=SEED,
        epoch=0,
        identity=encode_source_word_identity("marvin"),
        domain="unknown-word-rank",
    ).hex() == ("467c3f4bd22ca747ce332e0a88d4f5b0fb8d35f82bdc45605a2b22d644701ec5")


def test_integer_rejection_increments_the_draw_counter() -> None:
    bound = (1 << 63) + 1
    counter_zero = digest_word(
        seed=SEED,
        epoch=3,
        identity=NON_ASCII_COMMAND,
        domain="command-shift",
        counter=0,
    )
    counter_one = digest_word(
        seed=SEED,
        epoch=3,
        identity=NON_ASCII_COMMAND,
        domain="command-shift",
        counter=1,
    )

    assert counter_zero == 14_002_292_622_351_677_442
    assert stateless_digest(
        seed=SEED,
        epoch=3,
        identity=NON_ASCII_COMMAND,
        domain="command-shift",
        counter=0,
    ).hex() == ("023858e4f32252c2c752bdda435e60c9b2c83705c4ef1342fcb778bff186d3df")
    assert frame_message(
        epoch=3,
        identity=NON_ASCII_COMMAND,
        domain="command-shift",
        counter=0,
    ).hex() == (
        "66616c736577616b652d6578703030322d763100030000001f000000636f6d6d"
        "616e6400636166c3a92fd191d0b65f6e6f686173685f302e7761760d00000063"
        "6f6d6d616e642d73686966740000000000000000"
    )
    assert counter_zero >= bound
    assert counter_one == 7_572_055_692_799_477_931
    assert stateless_digest(
        seed=SEED,
        epoch=3,
        identity=NON_ASCII_COMMAND,
        domain="command-shift",
        counter=1,
    ).hex() == ("ab88c9bc88581569547ea7bb738ae10ba848bd683350b706166146c3500e7283")
    assert frame_message(
        epoch=3,
        identity=NON_ASCII_COMMAND,
        domain="command-shift",
        counter=1,
    ).hex() == (
        "66616c736577616b652d6578703030322d763100030000001f000000636f6d6d"
        "616e6400636166c3a92fd191d0b65f6e6f686173685f302e7761760d00000063"
        "6f6d6d616e642d73686966740100000000000000"
    )
    assert counter_one < bound
    assert (
        uniform_integer(
            seed=SEED,
            epoch=3,
            identity=NON_ASCII_COMMAND,
            domain="command-shift",
            bound=bound,
        )
        == counter_one
    )


def test_registered_command_draws_match_golden_values() -> None:
    common: DrawIdentity = {"seed": SEED, "epoch": 0, "identity": ASCII_COMMAND}

    shift_draw = uniform_integer(**common, domain="command-shift", bound=3_201)
    assert shift_draw == 2_298
    assert shift_draw - 1_600 == 698
    gain = uniform_real(
        **common,
        domain="command-gain-db",
        minimum=-6.0,
        maximum_exclusive=6.0,
    )
    assert gain.hex() == "0x1.705407fc3f460p+2"
    assert (
        uniform_real(
            **common,
            domain="command-noise-apply",
            minimum=0.0,
            maximum_exclusive=1.0,
        )
        == 0.541094757941312
    )
    assert bernoulli(**common, domain="command-noise-apply", probability=0.8) is True
    assert (
        uniform_integer(
            **common,
            domain="command-noise-window",
            bound=4_387_887,
        )
        == 3_263_409
    )
    snr = uniform_real(
        **common,
        domain="command-noise-snr-db",
        minimum=0.0,
        maximum_exclusive=20.0,
    )
    assert snr.hex() == "0x1.513d21f53c3a6p+2"
    assert uniform_integer(**common, domain="command-time-width", bound=11) == 8
    assert uniform_integer(**common, domain="command-time-start", bound=91) == 3
    assert uniform_integer(**common, domain="command-mel-width", bound=5) == 1
    assert uniform_integer(**common, domain="command-mel-start", bound=40) == 32


def test_real_draw_matches_golden_and_keeps_maximum_exclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = uniform_real(
        seed=SEED,
        epoch=3,
        identity=NON_ASCII_COMMAND,
        domain="command-gain-db",
        minimum=-6.0,
        maximum_exclusive=6.0,
    )
    assert result.hex() == "0x1.fba4565429e90p+1"

    def maximum_word(**_: object) -> int:
        return UINT64_MAX

    monkeypatch.setattr(rng, "digest_word", maximum_word)
    clamped = uniform_real(
        seed=0,
        epoch=0,
        identity=b"benchmark-feature-tensor",
        domain="benchmark-feature",
        minimum=0.0,
        maximum_exclusive=1.0,
    )
    assert clamped == math.nextafter(1.0, -math.inf)
    assert clamped < 1.0

    call_with_counter = cast(Callable[..., float], uniform_real)
    with pytest.raises(TypeError, match="counter"):
        call_with_counter(
            seed=SEED,
            epoch=0,
            identity=ASCII_COMMAND,
            domain="command-gain-db",
            minimum=-6.0,
            maximum_exclusive=6.0,
            counter=1,
        )


def test_integer_and_probability_boundaries_fail_closed() -> None:
    common: DrawContext = {
        "seed": SEED,
        "epoch": 0,
        "identity": ASCII_COMMAND,
        "domain": "command-shift",
    }
    assert uniform_integer(**common, bound=1) == 0
    assert uniform_integer(**common, bound=1 << 64) == digest_word(**common)
    assert uniform_integer(**common, bound=1 << 64, counter=1) == digest_word(
        **common, counter=1
    )
    assert bernoulli(**common, probability=0.0) is False
    assert bernoulli(**common, probability=1.0) is True

    for bound in (0, -1, (1 << 64) + 1):
        with pytest.raises(ValueError):
            uniform_integer(**common, bound=bound)
    with pytest.raises(TypeError):
        uniform_integer(**common, bound=cast(int, True))
    for probability in (-0.01, 1.01, math.nan, math.inf):
        with pytest.raises(ValueError):
            bernoulli(**common, probability=probability)
    for bounds in ((0.0, 0.0), (1.0, -1.0), (0.0, math.inf)):
        with pytest.raises(ValueError):
            uniform_real(
                **common,
                minimum=bounds[0],
                maximum_exclusive=bounds[1],
            )
    with pytest.raises(TypeError):
        bernoulli(**common, probability=cast(float, True))
    with pytest.raises(TypeError):
        bernoulli(**common, probability=cast(float, "0.5"))
    with pytest.raises(TypeError):
        uniform_real(
            **common,
            minimum=cast(float, False),
            maximum_exclusive=1.0,
        )
    with pytest.raises(TypeError):
        uniform_real(
            **common,
            minimum=0.0,
            maximum_exclusive=cast(float, "1.0"),
        )


def test_rejection_counter_fails_closed_at_uint64_max(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def maximum_word(**_: object) -> int:
        return UINT64_MAX

    monkeypatch.setattr(rng, "digest_word", maximum_word)
    with pytest.raises(OverflowError, match="counter exhausted"):
        uniform_integer(
            seed=SEED,
            epoch=0,
            identity=ASCII_COMMAND,
            domain="command-shift",
            bound=(1 << 63) + 1,
            counter=UINT64_MAX,
        )


def test_unknown_word_rank_order_matches_registered_golden() -> None:
    words = [
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
    observed = sorted(
        words,
        key=lambda word: rank_key(
            seed=SEED,
            epoch=0,
            identity=encode_source_word_identity(word),
            domain="unknown-word-rank",
        ),
    )
    assert observed == [
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
    ]


def test_rank_tie_breaks_on_exact_identity_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def tied_digest(**_: object) -> bytes:
        return b"\x80" * 32

    monkeypatch.setattr(rng, "stateless_digest", tied_digest)
    identities = [b"word\0zero", b"word\0bed", b"word\0cat"]
    observed = sorted(
        identities,
        key=lambda identity: rank_key(
            seed=SEED,
            epoch=0,
            identity=identity,
            domain="unknown-word-rank",
        ),
    )
    assert observed == sorted(identities)


def test_rank_compares_digest_bytes_beyond_the_first_word(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identities = [b"word\0tail-late", b"word\0tail-early"]

    def same_word_different_tail(**arguments: object) -> bytes:
        identity = arguments["identity"]
        assert isinstance(identity, bytes)
        tail_byte = b"\x02" if identity.endswith(b"late") else b"\x01"
        return b"\x80" * 8 + tail_byte * 24

    monkeypatch.setattr(rng, "stateless_digest", same_word_different_tail)
    observed = sorted(
        identities,
        key=lambda identity: rank_key(
            seed=SEED,
            epoch=0,
            identity=identity,
            domain="unknown-word-rank",
        ),
    )
    assert observed == [b"word\0tail-early", b"word\0tail-late"]


def test_domains_are_bound_to_the_machine_readable_contract() -> None:
    config = json.loads(
        Path("configs/experiment-002-training.json").read_text(encoding="utf-8")
    )
    domain_config = config["augmentation"]["draws"]["domains"]
    assert domain_config["calibration_rank_prefix"] == "calibration-rank-"
    assert (
        frozenset(
            value
            for key, value in domain_config.items()
            if key != "calibration_rank_prefix"
        )
        == REGISTERED_DOMAINS
    )
    assert (
        frozenset(f"calibration-rank-{label}" for label in config["class_order"])
        == REGISTERED_CALIBRATION_DOMAINS
    )

    frame_message(
        epoch=0,
        identity=b"command\0yes/example.wav",
        domain="calibration-rank-yes",
        counter=0,
    )
    for domain in (
        "calibration-rank-",
        "calibration-rank-not-a-class",
        "calibration-rank-yes\0bad",
        "calibration-rank-yes ",
        "unknown-domain",
        "calibration-rank-é",
    ):
        with pytest.raises(ValueError):
            frame_message(
                epoch=0,
                identity=b"command\0yes/example.wav",
                domain=domain,
                counter=0,
            )
    with pytest.raises(TypeError):
        frame_message(
            epoch=0,
            identity=b"command\0yes/example.wav",
            domain=cast(str, b"train-order"),
            counter=0,
        )


def test_framing_rejects_invalid_types_ranges_and_identity_bytes() -> None:
    with pytest.raises(TypeError):
        stateless_digest(
            seed=cast(int, True),
            epoch=0,
            identity=b"command\0yes/example.wav",
            domain="train-order",
        )
    for seed in (-1, UINT64_MAX + 1):
        with pytest.raises(ValueError):
            stateless_digest(
                seed=seed,
                epoch=0,
                identity=b"command\0yes/example.wav",
                domain="train-order",
            )
    for epoch in (-1, UINT32_MAX + 1):
        with pytest.raises(ValueError):
            frame_message(
                epoch=epoch,
                identity=b"command\0yes/example.wav",
                domain="train-order",
                counter=0,
            )
    for counter in (-1, UINT64_MAX + 1):
        with pytest.raises(ValueError):
            frame_message(
                epoch=0,
                identity=b"command\0yes/example.wav",
                domain="train-order",
                counter=counter,
            )
    with pytest.raises(TypeError):
        frame_message(
            epoch=cast(int, True),
            identity=b"command\0yes/example.wav",
            domain="train-order",
            counter=0,
        )
    with pytest.raises(TypeError):
        frame_message(
            epoch=0,
            identity=b"command\0yes/example.wav",
            domain="train-order",
            counter=cast(int, True),
        )
    for identity in (b"", b"command\0invalid\xff"):
        with pytest.raises(ValueError):
            frame_message(
                epoch=0,
                identity=identity,
                domain="train-order",
                counter=0,
            )
    with pytest.raises(TypeError):
        frame_message(
            epoch=0,
            identity=cast(bytes, "command\0yes/example.wav"),
            domain="train-order",
            counter=0,
        )
