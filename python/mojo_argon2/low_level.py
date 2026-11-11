"""Low-level Argon2 KDF API matching argon2-cffi's public functions."""

from __future__ import annotations

import base64
import hmac
import operator

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

# matches argon2_error_message() in upstream's argon2.c: the port reports the
# same text as argon2-cffi's low_level.error_to_str for every code upstream knows.
_ERROR_MESSAGES = {
    0: "OK",
    -1: "Output pointer is NULL",
    -2: "Output is too short",
    -3: "Output is too long",
    -4: "Password is too short",
    -5: "Password is too long",
    -6: "Salt is too short",
    -7: "Salt is too long",
    -8: "Associated data is too short",
    -9: "Associated data is too long",
    -10: "Secret is too short",
    -11: "Secret is too long",
    -12: "Time cost is too small",
    -13: "Time cost is too large",
    -14: "Memory cost is too small",
    -15: "Memory cost is too large",
    -16: "Too few lanes",
    -17: "Too many lanes",
    -18: "Password pointer is NULL, but password length is not 0",
    -19: "Salt pointer is NULL, but salt length is not 0",
    -20: "Secret pointer is NULL, but secret length is not 0",
    -21: "Associated data pointer is NULL, but ad length is not 0",
    -22: "Memory allocation error",
    -23: "The free memory callback is NULL",
    -24: "The allocate memory callback is NULL",
    -25: "Argon2_Context context is NULL",
    -26: "There is no such version of Argon2",
    -27: "Output pointer mismatch",
    -28: "Not enough threads",
    -29: "Too many threads",
    -30: "Missing arguments",
    -31: "Encoding failed",
    -32: "Decoding failed",
    -33: "Threading failure",
    -34: "Some of encoded parameters are too long or too short",
    -35: "The password does not match the supplied hash",
}


class Type(Enum):
    D = 0
    I = 1
    ID = 2


_TYPE_NAMES = {Type.D: "argon2d", Type.I: "argon2i", Type.ID: "argon2id"}
_NAME_TYPES = {value: key for key, value in _TYPE_NAMES.items()}


def _as_cffi_bytes(name: str, value):
    # upstream builds ffi.new("uint8_t[]", value); CFFI only takes bytes here
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        raise TypeError(
            "initializer for ctype 'uint8_t[]' must be a bytes or list or tuple, "
            "not str"
        )
    # list/tuple take CFFI's "new array length" path; bytearray, memoryview
    # and anything else are rejected by CFFI as an initializer type.
    raise TypeError(
        f"expected new array length or list/tuple/str, not {type(value).__name__}"
    )


def _validate(
    secret: bytes,
    salt: bytes,
    time_cost: int,
    memory_cost: int,
    parallelism: int,
    hash_len: int,
    type: Type,
    version: int,
    hash_len_checked: bool = False,
) -> None:
    """upstream validate_inputs()/unvalidated_args(), in the same order."""
    secret = _as_cffi_bytes("secret", secret)
    salt = _as_cffi_bytes("salt", salt)
    if not isinstance(type, Type):
        # upstream indexes Type.value directly, so a non-Type raises AttributeError
        raise AttributeError(f"'{type.__class__.__name__}' object has no value")
    # hash_len is only a CFFI buffer length (Py_ssize_t) on the raw path; it
    # is still converted here so the "< 4" check below sees an int.
    ints = {
        "time_cost": _cffi_uint(time_cost),
        "memory_cost": _cffi_uint(memory_cost),
        "parallelism": _cffi_uint(parallelism),
        "hash_len": _cffi_uint(hash_len),
        "version": _cffi_uint(version),
    }
    # upstream validate_inputs() runs unvalidated_args() first: the uint32
    # conversion overflowed before the Argon2-level range checks below.
    _check_uint32_range("time_cost", ints["time_cost"])
    _check_uint32_range("memory_cost", ints["memory_cost"])
    _check_uint32_range("parallelism", ints["parallelism"])
    if not hash_len_checked:
        _check_uint32_range("hash_len", ints["hash_len"])
    _check_uint32_range("version", ints["version"])
    if len(secret) > _UINT32_MAX:
        raise HashingError("Secret is too long")
    if len(salt) > _UINT32_MAX:
        raise HashingError("Salt is too long")
    if len(salt) < 8:
        raise HashingError("Salt is too short")
    if ints["time_cost"] < 1:
        raise HashingError("Time cost is too small")
    if ints["parallelism"] < 1:
        raise HashingError("Too few lanes")
    if ints["parallelism"] > 0xFFFFFF:
        raise HashingError("Too many lanes")
    if ints["memory_cost"] < 8 * ints["parallelism"]:
        raise HashingError("Memory cost is too small")
    if ints["hash_len"] < 4:
        raise HashingError("Output is too short")
    # argon2-cffi's argon2_hash accepts any version; only 16 gets the v1.0
    # fill_block indexing, every other value is used as given.


def _cffi_uint(value):
    """CFFI's conversion for one uint argument (bool and numpy ints pass)."""
    if value is None or isinstance(value, (str, bytes, bytearray, float)):
        raise TypeError("an integer is required")
    try:
        value = operator.index(value)
    except TypeError:
        raise TypeError("an integer is required") from None
    if value < 0:
        # argon2-cffi passes ints straight to CFFI, which refuses negatives.
        raise OverflowError("can't convert negative number to unsigned")
    return value


def _check_uint32_range(name, value):
    # the CFFI prototype uses 32-bit unsigned ints; CFFI raises on overflow.
    if value > _UINT32_MAX:
        raise OverflowError(f"integer {value} does not fit '32-bit unsigned int'")


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
    return _argon2_hash(
        secret,
        salt,
        time_cost,
        memory_cost,
        parallelism,
        hash_len,
        type,
        version,
        raw=True,
    )


def _argon2_hash(
    secret: bytes,
    salt: bytes,
    time_cost: int,
    memory_cost: int,
    parallelism: int,
    hash_len: int,
    type: Type,
    version: int,
    *,
    raw: bool,
) -> bytes:
    """upstream low_level.hash_secret_raw / hash_secret, in their own order."""
    secret = _as_cffi_bytes("secret", secret)
    salt = _as_cffi_bytes("salt", salt)
    if raw:
        # upstream calls ffi.new("uint8_t[]", hash_len) before argon2_hash:
        # CFFI converts the length (bool and numpy ints accepted) and takes any
        # non-negative value, so a float is CFFI's array-length TypeError.
        if not isinstance(hash_len, int):
            if hash_len is not None and not isinstance(
                hash_len, (str, bytes, bytearray)
            ):
                try:
                    hash_len = operator.index(hash_len)
                except TypeError:
                    raise TypeError(
                        "expected new array length or list/tuple/str, not "
                        + type(hash_len).__name__
                    ) from None
        if hash_len < 0:
            raise ValueError("negative array length")
    else:
        # upstream calls argon2_encodedlen() first, which narrows every cost to
        # uint32 and reads type.value (so a non-Type is AttributeError).
        if not isinstance(type, Type):
            raise AttributeError(f"'{type.__class__.__name__}' object has no value")
        for name, value in (
            ("time_cost", time_cost),
            ("memory_cost", memory_cost),
            ("parallelism", parallelism),
            ("hash_len", hash_len),
            ("version", version),
        ):
            _check_uint32_range(name, _cffi_uint(value))
    _validate(
        secret,
        salt,
        time_cost,
        memory_cost,
        parallelism,
        hash_len,
        type,
        version,
        hash_len_checked=True,
    )
    # argon2_hash takes hash_len as uint32 (only the bytes buffer uses
    # Py_ssize_t), so >uint32 is Argon2's "Output is too long", not overflow.
    if hash_len > _UINT32_MAX:
        raise HashingError("Output is too long")
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


def _from_base64_accepts(data: bytes) -> bool:
    """Literal port of encoding.c's from_base64() acceptance test.

    Returns False only for a 1-mod-4 tail or non-zero buffered bits; True
    otherwise.  ``acc`` is an ``unsigned`` (32-bit) accumulator, so for tags
    of 64+ bytes the shift itself discards its top bits, ``acc_len`` never
    builds up, and the buffered-bits test can no longer fail.
    """
    alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    acc = 0
    acc_len = 0
    for d in data:
        value = alphabet.find(bytes([d]))
        if value < 0:
            return True  # stops at the non-base64 char; the caller never gets there
        acc = ((acc << 6) + value) & 0xFFFFFFFF
        acc_len += 6
        if acc_len >= 8:
            acc_len -= 8
    if acc_len > 4 or (acc & ((1 << acc_len) - 1)) != 0:
        return False
    return True


def _to_base64_length(length: int, width: int) -> int:
    """Character count argon2.c's to_base64() writes for *length* bytes.

    The closing loop is ``while (i < len) and (written < dst_len)``: the final
    full 3-byte group is written whenever it is reached regardless of width,
    while the 1-2 byte tail adds ``leftover + 1`` chars only if 4 more fit.
    The result depends on the byte count alone, never on the byte values.
    """
    group = 4
    full = length // 3
    written = min(full, max(width, 0) // group) * group
    leftover = length % 3
    if leftover and written + 4 <= width:
        written += leftover + 1
    return written


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
    raw = _argon2_hash(
        secret,
        salt,
        time_cost,
        memory_cost,
        parallelism,
        hash_len,
        type,
        version,
        raw=False,
    )
    # bool is a valid int here; upstream's f-string prints it as 1, not True
    return (
        f"${_TYPE_NAMES[type]}$v={int(version)}$m={int(memory_cost)},"
        f"t={int(time_cost)},p={int(parallelism)}$"
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


_B64_ALPHABET = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"


def _b64_char_to_byte(c: int) -> int:
    """encoding.c b64_char_to_byte(): 0xFF for any non-Base64 character."""
    ch = bytes([c]) if 0 <= c <= 0xFF else b""
    value = _B64_ALPHABET.find(ch)
    return value if value >= 0 else 0xFF  # find() already yields the index


def _from_base64_run(data: bytes, start: int, max_len: int):
    """encoding.c from_base64(dst, &dst_len, src), including the 32-bit quirks.

    Returns (dst_len, end), ``end`` being the index of the first non-Base64
    character (C returns that pointer), or (None, end) on failure.

    C keeps ``unsigned acc`` and tests ``acc_len > 4 or (acc & ((1 << acc_len)
    - 1)) != 0``.  Replaying that literally is subtly wrong once many sextets
    have shifted through: the terminator itself (0xFF) and the high bits a
    real last character would leave cancel in that mask.  Measured against
    argon2-cffi 25.1.0 the surviving rule is:

    * a full group (4 chars) that ends in non-Base64 decodes fine whenever the
      characters before it decode to whole bytes -- the 2 leftover bits vanish;
    * the last Base64 character before the terminator must have its discarded
      bits clear, counting backwards from the last byte boundary.
    """
    length = 0
    acc = 0
    acc_len = 0
    pos = start
    while pos < len(data):
        d = _b64_char_to_byte(data[pos])
        if d == 0xFF:
            break
        pos += 1
        acc = ((acc << 6) + d) & 0xFFFFFFFF
        acc_len += 6
        if acc_len >= 8:
            acc_len -= 8
            if length >= max_len:
                return None, pos
            length += 1
    if acc_len > 4:
        return None, pos
    if acc_len in (2, 4) and _tail_is_partial(pos - start):
        last = _b64_char_to_byte(data[pos - 1])
        if last & ((1 << acc_len) - 1):
            return None, pos
        return length, pos
    if acc_len in (2, 4) and (acc & ((1 << acc_len) - 1)):
        return None, pos
    return length, pos


def _tail_is_partial(chars: int) -> bool:
    """Whether the last full 4-char group ended on a byte boundary.

    With ``acc_len`` bits still buffered the group was cut short only when
    the number of complete sextets, ``chars`` (the terminator is not one),
    is not a multiple of 4.
    """
    return chars % 4 != 0


def _decode_decimal(data: bytes, pos: int):
    """encoding.c decode_decimal(): minimal digits fitting unsigned long."""
    start = pos
    acc = 0
    while pos < len(data) and 0x30 <= data[pos] <= 0x39:
        acc = acc * 10 + (data[pos] - 0x30)
        if acc > 0xFFFFFFFFFFFFFFFF:
            return None, start
        pos += 1
    if pos == start:
        return None, start
    if pos - start > 1 and data[start] == 0x30:
        return None, start  # not minimal
    return acc, pos


def _decode_string(data: bytes, type: Type):
    """encoding.c decode_string() for the exact encoded string argon2_verify got.

    maxsaltlen/maxoutlen are both strlen(encoded) <= uint32, so no field can
    overflow the uint32 lengths.  Returns the decoded tuple, or the
    argon2_error_message() text that argon2_verify() would raise.
    """
    if len(data) > _UINT32_MAX:
        return "Decoding failed"
    max_field_len = len(data)
    pos = 0

    def take(literal: bytes) -> bool:
        nonlocal pos
        # peek first: the inner functions advance pos themselves
        end = pos + len(literal)
        if data[pos:end] == literal:
            pos = end
            return True
        return False

    if not take(b"$"):
        return "Decoding failed"
    if not take(_TYPE_NAMES[type].encode("ascii")):
        return "Decoding failed"

    version = 16  # ARGON2_VERSION_10 when "$v=" is omitted
    if take(b"$v="):
        version, pos = _decode_decimal(data, pos)
        if version is None or version > _UINT32_MAX:
            return "Decoding failed"

    costs = []
    for literal in (b"$m=", b",t=", b",p="):
        if not take(literal):
            return "Decoding failed"
        value, pos = _decode_decimal(data, pos)
        if value is None or value > _UINT32_MAX:
            return "Decoding failed"
        costs.append(value)
    memory_cost, time_cost, parallelism = costs

    if not take(b"$"):
        return "Decoding failed"
    salt_start = pos
    salt_len, salt_end = _from_base64_run(data, pos, max_field_len)
    if salt_len is None:
        return "Decoding failed"
    pos = salt_end
    if not take(b"$"):
        return "Decoding failed"
    out_start = pos
    out_len, out_end = _from_base64_run(data, pos, max_field_len)
    if out_len is None:
        return "Decoding failed"
    pos = out_end

    # decode_string() calls validate_inputs(ctx) here, ahead of the final
    # "no additional characters" check.
    if out_len < 4:
        return "Output is too short"
    if salt_len < 8:
        return "Salt is too short"
    if time_cost < 1:
        return "Time cost is too small"
    if parallelism < 1:
        return "Too few lanes"
    if parallelism > 0xFFFFFF:
        return "Too many lanes"
    if memory_cost < 8 * parallelism:
        return "Memory cost is too small"
    if time_cost > 0xFFFFFFFF:
        return "Time cost is too large"
    if pos != len(data):
        return "Decoding failed"
    return (
        version,
        memory_cost,
        time_cost,
        parallelism,
        _decode(data[salt_start:salt_end]),
        out_len,
        _decode(data[out_start:out_end]),
    )


def _cffi_bytes(value):
    # ffi.new("char[]", hash) / ffi.new("uint8_t[]", secret)
    if isinstance(value, bytes):
        return value
    raise TypeError(
        f"initializer for ctype 'char[]' must be a bytes or list or tuple, "
        f"not {type(value).__name__}"
    )


def verify_secret(
    hash: bytes, secret: bytes, type: Type
) -> Literal[True]:
    if not isinstance(type, Type):
        raise TypeError("type must be a Type")
    hash = _cffi_bytes(hash)
    secret = _cffi_bytes(secret)
    parsed = _decode_string(hash, type)
    if isinstance(parsed, str):
        raise VerificationError(parsed) from None
    version, memory_cost, time_cost, parallelism, salt, outlen, expected = parsed
    try:
        # argon2_verify_ctx() re-runs argon2_ctx on the decoded context, so
        # the hash is computed with the *decoded* outlen (equal to len(expected)
        # once from_base64 has consumed every character).
        actual = _argon2_hash(
            secret,
            salt,
            time_cost,
            memory_cost,
            parallelism,
            outlen,
            type,
            version,
            raw=True,
        )
    except HashingError as exc:
        raise VerificationError(str(exc)) from exc
    # argon2_verify_ctx compares all outlen bytes with argon2_compare
    if not hmac.compare_digest(actual, expected):
        raise VerifyMismatchError("The password does not match the supplied hash")
    return True


def error_to_str(error: int) -> str:
    # mirrors argon2_error_message() as surfaced by argon2-cffi's error_to_str;
    # the declared argument is a plain C int, so values outside int32 overflow.
    if error < -(2**31) or error > 2**31 - 1:
        raise OverflowError(f"integer {error} does not fit '32-bit int'")
    return _ERROR_MESSAGES.get(error, "Unknown error code")
