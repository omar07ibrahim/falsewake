"""Stateless HMAC-SHA256 draws registered for Experiment 002.

Every result is a pure function of a seed, epoch, exact UTF-8 identity, domain,
and scalar counter.  The module deliberately exposes the framing operation so
experiment evidence can bind bytes instead of relying on an implementation
description.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import struct
from typing import Final

PROTOCOL_PREFIX: Final = b"falsewake-exp002-v1\0"
UINT32_MAX: Final = (1 << 32) - 1
UINT64_MAX: Final = (1 << 64) - 1
UINT64_MODULUS: Final = 1 << 64
CALIBRATION_RANK_PREFIX: Final = "calibration-rank-"
CALIBRATION_CLASS_LABELS: Final = (
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
REGISTERED_CALIBRATION_DOMAINS: Final = frozenset(
    f"{CALIBRATION_RANK_PREFIX}{label}" for label in CALIBRATION_CLASS_LABELS
)
REGISTERED_DOMAINS: Final = frozenset(
    {
        "benchmark-feature",
        "benchmark-pcm",
        "command-gain-db",
        "command-mel-start",
        "command-mel-width",
        "command-noise-apply",
        "command-noise-snr-db",
        "command-noise-window",
        "command-shift",
        "command-time-start",
        "command-time-width",
        "silence-gain-db",
        "silence-mel-start",
        "silence-mel-width",
        "silence-time-start",
        "silence-time-width",
        "silence-window-rank",
        "train-order",
        "unknown-clip-rank",
        "unknown-word-rank",
    }
)


def encode_command_identity(manifest_path: str) -> bytes:
    """Return the exact identity for one command manifest row."""

    return b"command\0" + _encode_identity_component("manifest_path", manifest_path)


def encode_window_identity(background_path: str, start_sample: int) -> bytes:
    """Return the exact identity for one silence or noise window."""

    path = _encode_identity_component("background_path", background_path)
    _validate_unsigned("start_sample", start_sample, UINT64_MAX)
    return b"window\0" + path + b"\0" + str(start_sample).encode("ascii")


def encode_source_word_identity(source_word: str) -> bytes:
    """Return the exact identity for one unknown-word allocation."""

    return b"word\0" + _encode_identity_component("source_word", source_word)


def frame_message(
    *,
    epoch: int,
    identity: bytes,
    domain: str,
    counter: int,
) -> bytes:
    """Serialize the registered HMAC message exactly."""

    _validate_unsigned("epoch", epoch, UINT32_MAX)
    _validate_unsigned("counter", counter, UINT64_MAX)
    if not isinstance(identity, bytes):
        raise TypeError("identity must be bytes")
    if not identity:
        raise ValueError("identity must not be empty")
    try:
        identity.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("identity must be valid UTF-8 bytes") from error
    domain_bytes = _domain_bytes(domain)
    if len(identity) > UINT32_MAX:
        raise ValueError("identity UTF-8 encoding is too long")
    if len(domain_bytes) > UINT32_MAX:
        raise ValueError("domain ASCII encoding is too long")
    return b"".join(
        (
            PROTOCOL_PREFIX,
            struct.pack("<I", epoch),
            struct.pack("<I", len(identity)),
            identity,
            struct.pack("<I", len(domain_bytes)),
            domain_bytes,
            struct.pack("<Q", counter),
        )
    )


def stateless_digest(
    *,
    seed: int,
    epoch: int,
    identity: bytes,
    domain: str,
    counter: int = 0,
) -> bytes:
    """Return the registered 32-byte HMAC-SHA256 digest."""

    _validate_unsigned("seed", seed, UINT64_MAX)
    message = frame_message(
        epoch=epoch,
        identity=identity,
        domain=domain,
        counter=counter,
    )
    return hmac.new(struct.pack("<Q", seed), message, hashlib.sha256).digest()


def digest_word(
    *,
    seed: int,
    epoch: int,
    identity: bytes,
    domain: str,
    counter: int = 0,
) -> int:
    """Interpret the first eight digest bytes as one little-endian uint64."""

    digest = stateless_digest(
        seed=seed,
        epoch=epoch,
        identity=identity,
        domain=domain,
        counter=counter,
    )
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def uniform_integer(
    *,
    seed: int,
    epoch: int,
    identity: bytes,
    domain: str,
    bound: int,
    counter: int = 0,
) -> int:
    """Draw an unbiased integer in ``[0, bound)`` by rejection sampling."""

    _validate_unsigned("bound", bound, UINT64_MODULUS)
    if bound == 0:
        raise ValueError("bound must be positive")
    _validate_unsigned("counter", counter, UINT64_MAX)
    acceptance_limit = UINT64_MODULUS - (UINT64_MODULUS % bound)
    while True:
        word = digest_word(
            seed=seed,
            epoch=epoch,
            identity=identity,
            domain=domain,
            counter=counter,
        )
        if word < acceptance_limit:
            return word % bound
        if counter == UINT64_MAX:
            raise OverflowError("integer rejection counter exhausted uint64")
        counter += 1


def uniform_real(
    *,
    seed: int,
    epoch: int,
    identity: bytes,
    domain: str,
    minimum: float,
    maximum_exclusive: float,
) -> float:
    """Draw a float64 value in ``[minimum, maximum_exclusive)``."""

    lower = _finite_float("minimum", minimum)
    upper = _finite_float("maximum_exclusive", maximum_exclusive)
    if not lower < upper:
        raise ValueError("real bounds must be finite and strictly increasing")
    width = upper - lower
    if not math.isfinite(width):
        raise ValueError("real interval width must be finite")
    word = digest_word(
        seed=seed,
        epoch=epoch,
        identity=identity,
        domain=domain,
    )
    unit = float(word >> 11) / float(1 << 53)
    raw = lower + width * unit
    exclusive_ceiling = math.nextafter(upper, -math.inf)
    return min(raw, exclusive_ceiling)


def bernoulli(
    *,
    seed: int,
    epoch: int,
    identity: bytes,
    domain: str,
    probability: float,
) -> bool:
    """Draw the registered Bernoulli decision from a float64 unit draw."""

    threshold = _finite_float("probability", probability)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("probability must be finite and within [0, 1]")
    draw = uniform_real(
        seed=seed,
        epoch=epoch,
        identity=identity,
        domain=domain,
        minimum=0.0,
        maximum_exclusive=1.0,
    )
    return draw < threshold


def rank_key(
    *, seed: int, epoch: int, identity: bytes, domain: str
) -> tuple[bytes, bytes]:
    """Return the digest-first, identity-second registered ordering key."""

    digest = stateless_digest(
        seed=seed,
        epoch=epoch,
        identity=identity,
        domain=domain,
    )
    return digest, identity


def _encode_identity_component(name: str, value: str) -> bytes:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must not be empty")
    if "\0" in value:
        raise ValueError(f"{name} must not contain NUL")
    try:
        return value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} must encode as valid UTF-8") from error


def _domain_bytes(domain: str) -> bytes:
    if not isinstance(domain, str):
        raise TypeError("domain must be a string")
    registered = domain in REGISTERED_DOMAINS
    calibration = domain in REGISTERED_CALIBRATION_DOMAINS
    if not registered and not calibration:
        raise ValueError(f"unregistered Experiment 002 domain: {domain!r}")
    try:
        return domain.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError("domain must contain ASCII only") from error


def _validate_unsigned(name: str, value: int, maximum: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if not 0 <= value <= maximum:
        raise ValueError(f"{name} must be within [0, {maximum}]")


def _finite_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted
