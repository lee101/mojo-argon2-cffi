"""Low-level Argon2 KDF API matching argon2-cffi's public functions."""

from __future__ import annotations

import base64
import hmac

from enum import Enum
from typing import Literal

from ._lib import hash_raw as _mojo_hash_raw
from .exceptions import (
    HashingError,
    VerificationError,
    VerifyMismatchError,
)


ARGON2_VERSION = 19
_UINT32_MAX = (1 << 32) - 1


class Type(Enum):
    D = 0
    I = 1
    ID = 2


_TYPE_NAMES = {Type.D: "argon2d", Type.I: "argon2i", Type.ID: "argon2id"}
_NAME_TYPES = {value: key for key, value in _TYPE_NAMES.items()}


def _validate(
    secret: bytes,
    salt: bytes,
    time_cost: int,
    memory_cost: int,
    parallelism: int,
    hash_len: int,
    type: Type,
    version: int,
) -> None:
    if not isinstance(secret, bytes):
        raise TypeError("secret must be bytes")
    if not isinstance(salt, bytes):
        raise TypeError("salt must be bytes")
    if not isinstance(type, Type):
        raise TypeError("type must be a Type")
    for name, value in (
        ("time_cost", time_cost),
        ("memory_cost", memory_cost),
        ("parallelism", parallelism),
        ("hash_len", hash_len),
        ("version", version),
    ):
        if not isinstance(value, int):
            raise TypeError(f"{name} must be an int")
        if value > _UINT32_MAX:
            raise HashingError(f"{name} is too large")
    if len(secret) > _UINT32_MAX:
        raise HashingError("Secret is too long")
    if len(salt) > _UINT32_MAX:
        raise HashingError("Salt is too long")
    if len(salt) < 8:
        raise HashingError("Salt is too short")
    if time_cost < 1:
        raise HashingError("Time cost is too small")
    if parallelism < 1:
        raise HashingError("Too few lanes")
    if parallelism > 0xFFFFFF:
        raise HashingError("Too many lanes")
    if memory_cost < 8 * parallelism:
        raise HashingError("Memory cost is too small")
    if hash_len < 4:
        raise HashingError("Output is too short")
    if version not in (16, 19):
        raise HashingError("Unsupported version")


def hash_secret_raw(
    secret: bytes,
    salt: bytes,
    time_cost: int,
    memory_cost: int,
    parallelism: int,
    hash_len: int,
    type: Type,
    version: int = ARGON2_VERSION,
) -> bytes:
    _validate(
        secret, salt, time_cost, memory_cost, parallelism, hash_len, type, version
    )
    try:
        return _mojo_hash_raw(
            secret,
            salt,
            time_cost,
            memory_cost,
            parallelism,
            hash_len,
            type.value,
            version,
        )
    except (MemoryError, OSError, OverflowError, RuntimeError, ValueError) as exc:
        raise HashingError(str(exc)) from exc


def _b64(data: bytes) -> bytes:
    return base64.b64encode(data).rstrip(b"=")


def _decode(data: bytes) -> bytes:
    try:
        return base64.b64decode(data + b"=" * (-len(data) % 4), validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise VerificationError("Decoding failed") from exc


def hash_secret(
    secret: bytes,
    salt: bytes,
    time_cost: int,
    memory_cost: int,
    parallelism: int,
    hash_len: int,
    type: Type,
    version: int = ARGON2_VERSION,
) -> bytes:
    raw = hash_secret_raw(
        secret,
        salt,
        time_cost,
        memory_cost,
        parallelism,
        hash_len,
        type,
        version,
    )
    return (
        f"${_TYPE_NAMES[type]}$v={version}$m={memory_cost},t={time_cost},"
        f"p={parallelism}$"
    ).encode("ascii") + _b64(salt) + b"$" + _b64(raw)


def _parse(encoded: bytes) -> tuple[Type, int, int, int, int, bytes, bytes]:
    try:
        parts = encoded.split(b"$")
        if len(parts) == 5:
            parts.insert(2, b"v=18")
        if len(parts) != 6 or parts[0]:
            raise ValueError
        type_value = _NAME_TYPES[parts[1].decode("ascii")]
        version = int(parts[2].removeprefix(b"v="))
        values = dict(item.split(b"=", 1) for item in parts[3].split(b","))
        if set(values) != {b"m", b"t", b"p"}:
            raise ValueError
        memory_cost = int(values[b"m"])
        time_cost = int(values[b"t"])
        parallelism = int(values[b"p"])
        salt = _decode(parts[4])
        raw = _decode(parts[5])
        return (
            type_value,
            version,
            memory_cost,
            time_cost,
            parallelism,
            salt,
            raw,
        )
    except VerificationError:
        raise
    except Exception as exc:
        raise VerificationError("Decoding failed") from exc


def verify_secret(
    hash: bytes, secret: bytes, type: Type
) -> Literal[True]:
    if not isinstance(hash, bytes) or not isinstance(secret, bytes):
        raise TypeError("hash and secret must be bytes")
    if not isinstance(type, Type):
        raise TypeError("type must be a Type")
    (
        parsed_type,
        version,
        memory_cost,
        time_cost,
        parallelism,
        salt,
        expected,
    ) = _parse(hash)
    if parsed_type is not type:
        raise VerificationError("Decoding failed")
    try:
        actual = hash_secret_raw(
            secret,
            salt,
            time_cost,
            memory_cost,
            parallelism,
            len(expected),
            type,
            version,
        )
    except HashingError as exc:
        raise VerificationError(str(exc)) from exc
    if not hmac.compare_digest(actual, expected):
        raise VerifyMismatchError("The password does not match the supplied hash")
    return True


def core(context, type: int) -> int:
    raise NotImplementedError(
        "the mutable CFFI Argon2_Context API is outside this port's KDF surface"
    )


def error_to_str(error: int) -> str:
    return "OK" if error == 0 else f"Argon2 error {error}"
