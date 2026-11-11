"""Byte-for-byte and behavioral parity with argon2-cffi."""

import os
import sys

import pytest

import argon2 as upstream
import mojo_argon2 as mojo

from argon2 import low_level as upstream_ll
from mojo_argon2 import _lib as mojo_bridge
from mojo_argon2 import low_level as mojo_ll


TYPE_PAIRS = [
    (mojo.Type.D, upstream.Type.D),
    (mojo.Type.I, upstream.Type.I),
    (mojo.Type.ID, upstream.Type.ID),
]


@pytest.mark.parametrize(("mojo_type", "upstream_type"), TYPE_PAIRS)
@pytest.mark.parametrize("version", [16, 19])
def test_all_variants_and_versions(mojo_type, upstream_type, version):
    kwargs = dict(
        secret=b"correct horse battery staple",
        salt=b"0123456789abcdef",
        time_cost=2,
        memory_cost=32,
        parallelism=1,
        hash_len=32,
        version=version,
    )
    assert mojo_ll.hash_secret_raw(type=mojo_type, **kwargs) == (
        upstream_ll.hash_secret_raw(type=upstream_type, **kwargs)
    )


@pytest.mark.parametrize(
    ("memory_cost", "parallelism", "time_cost", "hash_len"),
    [(8, 1, 1, 4), (17, 1, 2, 65), (35, 2, 3, 16), (64, 4, 2, 100)],
)
@pytest.mark.parametrize(("mojo_type", "upstream_type"), TYPE_PAIRS)
def test_lanes_memory_rounding_and_long_tags(
    memory_cost, parallelism, time_cost, hash_len, mojo_type, upstream_type
):
    kwargs = dict(
        secret=b"",
        salt=b"long-enough-salt",
        time_cost=time_cost,
        memory_cost=memory_cost,
        parallelism=parallelism,
        hash_len=hash_len,
        version=19,
    )
    assert mojo_ll.hash_secret_raw(type=mojo_type, **kwargs) == (
        upstream_ll.hash_secret_raw(type=upstream_type, **kwargs)
    )


def test_documented_argon2d_vector():
    actual = mojo_ll.hash_secret_raw(
        b"secret", b"somesalt", 1, 8, 1, 8, mojo.Type.D
    )
    assert actual == bytes.fromhex("e46ef5c87ca33e1d")


def test_password_larger_than_blake_block():
    secret = bytes(range(256)) * 2
    kwargs = dict(
        secret=secret,
        salt=b"0123456789abcdef",
        time_cost=1,
        memory_cost=32,
        parallelism=2,
        hash_len=32,
    )
    assert mojo_ll.hash_secret_raw(type=mojo.Type.ID, **kwargs) == (
        upstream_ll.hash_secret_raw(type=upstream.Type.ID, **kwargs)
    )


@pytest.mark.parametrize("memory_cost", [2_040, 2_048])
def test_parallel_threshold_parity(memory_cost):
    kwargs = dict(
        secret=b"parallel threshold",
        salt=b"0123456789abcdef",
        time_cost=1,
        memory_cost=memory_cost,
        parallelism=2,
        hash_len=32,
    )
    assert mojo_ll.hash_secret_raw(type=mojo.Type.ID, **kwargs) == (
        upstream_ll.hash_secret_raw(type=upstream.Type.ID, **kwargs)
    )


@pytest.mark.parametrize(
    ("memory_cost", "parallelism", "min_blocks", "expected_workers"),
    [
        (4_096, 4, 0, 4),
        (4_096, 4, 1 << 40, None),
        (2_040, 2, 0, None),
    ],
)
def test_lane_fan_out_policy(
    monkeypatch, memory_cost, parallelism, min_blocks, expected_workers
):
    seen = []
    real = mojo_bridge.ThreadPoolExecutor

    class Recording(real):
        def __init__(self, *args, **kwargs):
            seen.append(kwargs.get("max_workers"))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(mojo_bridge, "ThreadPoolExecutor", Recording)
    monkeypatch.setattr(mojo_bridge, "MIN_PARALLEL_BLOCKS", min_blocks)
    kwargs = dict(
        secret=b"lane fanout",
        salt=b"0123456789abcdef",
        time_cost=2,
        memory_cost=memory_cost,
        parallelism=parallelism,
        hash_len=32,
    )
    assert mojo_ll.hash_secret_raw(type=mojo.Type.ID, **kwargs) == (
        upstream_ll.hash_secret_raw(type=upstream.Type.ID, **kwargs)
    )
    assert seen == ([] if expected_workers is None else [expected_workers])


@pytest.mark.parametrize("memory_cost", [32_760, 32_768])
def test_parallel_wipe_threshold_parity(memory_cost):
    kwargs = dict(
        secret=b"parallel wipe threshold",
        salt=b"0123456789abcdef",
        time_cost=1,
        memory_cost=memory_cost,
        parallelism=2,
        hash_len=32,
    )
    assert mojo_ll.hash_secret_raw(type=mojo.Type.ID, **kwargs) == (
        upstream_ll.hash_secret_raw(type=upstream.Type.ID, **kwargs)
    )


def test_simd_copy_tail_parity():
    kwargs = dict(
        secret=b"simd tail",
        salt=b"0123456789abcdef",
        time_cost=1,
        memory_cost=32,
        parallelism=1,
        hash_len=65,
    )
    assert mojo_ll.hash_secret_raw(type=mojo.Type.I, **kwargs) == (
        upstream_ll.hash_secret_raw(type=upstream.Type.I, **kwargs)
    )


@pytest.mark.parametrize(("mojo_type", "upstream_type"), TYPE_PAIRS)
def test_encoded_hash_exact_parity(mojo_type, upstream_type):
    kwargs = dict(
        secret=b"password",
        salt=b"deterministic-salt",
        time_cost=2,
        memory_cost=32,
        parallelism=2,
        hash_len=24,
    )
    ours = mojo_ll.hash_secret(type=mojo_type, **kwargs)
    theirs = upstream_ll.hash_secret(type=upstream_type, **kwargs)
    assert ours == theirs
    assert mojo_ll.verify_secret(ours, b"password", mojo_type)
    assert upstream_ll.verify_secret(ours, b"password", upstream_type)
    assert mojo_ll.verify_secret(theirs, b"password", mojo_type)


def test_verify_mismatch_exception_hierarchy():
    encoded = mojo_ll.hash_secret(
        b"right", b"0123456789abcdef", 1, 16, 1, 16, mojo.Type.ID
    )
    with pytest.raises(mojo.exceptions.VerifyMismatchError):
        mojo_ll.verify_secret(encoded, b"wrong", mojo.Type.ID)
    assert issubclass(
        mojo.exceptions.VerifyMismatchError, mojo.exceptions.VerificationError
    )


def test_verify_rejects_wrong_variant():
    encoded = mojo_ll.hash_secret(
        b"password", b"0123456789abcdef", 1, 16, 1, 16, mojo.Type.ID
    )
    with pytest.raises(mojo.exceptions.VerificationError):
        mojo_ll.verify_secret(encoded, b"password", mojo.Type.I)


def test_password_hasher_deterministic_parity():
    kwargs = dict(
        time_cost=2,
        memory_cost=32,
        parallelism=2,
        hash_len=24,
        salt_len=16,
        type=mojo.Type.ID,
    )
    ours = mojo.PasswordHasher(**kwargs)
    theirs = upstream.PasswordHasher(
        **{**kwargs, "type": upstream.Type.ID}
    )
    salt = b"0123456789abcdef"
    assert ours.hash("pässword", salt=salt) == theirs.hash(
        "pässword", salt=salt
    )


def test_password_hasher_random_salt_and_verify():
    hasher = mojo.PasswordHasher(
        time_cost=1, memory_cost=16, parallelism=1, hash_len=16, salt_len=8
    )
    encoded = hasher.hash("password")
    assert hasher.verify(encoded, "password")
    with pytest.raises(mojo.exceptions.VerifyMismatchError):
        hasher.verify(encoded, "not-password")


def test_extract_parameters_parity():
    encoded = mojo_ll.hash_secret(
        b"password", b"0123456789abcdef", 2, 35, 2, 24, mojo.Type.I
    ).decode()
    assert mojo.extract_parameters(encoded) == mojo.Parameters(
        type=mojo.Type.I,
        version=19,
        salt_len=16,
        hash_len=24,
        time_cost=2,
        memory_cost=35,
        parallelism=2,
    )
    theirs = upstream.extract_parameters(encoded)
    ours = mojo.extract_parameters(encoded)
    assert (
        ours.version,
        ours.salt_len,
        ours.hash_len,
        ours.time_cost,
        ours.memory_cost,
        ours.parallelism,
    ) == (
        theirs.version,
        theirs.salt_len,
        theirs.hash_len,
        theirs.time_cost,
        theirs.memory_cost,
        theirs.parallelism,
    )


def test_check_needs_rehash_and_from_parameters():
    params = mojo.Parameters(mojo.Type.ID, 19, 8, 16, 1, 16, 1)
    hasher = mojo.PasswordHasher.from_parameters(params)
    encoded = hasher.hash(b"password", salt=b"12345678")
    assert not hasher.check_needs_rehash(encoded)
    assert mojo.PasswordHasher(
        time_cost=2, memory_cost=16, parallelism=1, hash_len=16, salt_len=8
    ).check_needs_rehash(encoded)


def test_profiles_match_upstream():
    for name in (
        "RFC_9106_HIGH_MEMORY",
        "RFC_9106_LOW_MEMORY",
        "PRE_21_2",
        "CHEAPEST",
    ):
        ours = getattr(mojo.profiles, name)
        theirs = getattr(upstream.profiles, name)
        assert (
            ours.version,
            ours.salt_len,
            ours.hash_len,
            ours.time_cost,
            ours.memory_cost,
            ours.parallelism,
        ) == (
            theirs.version,
            theirs.salt_len,
            theirs.hash_len,
            theirs.time_cost,
            theirs.memory_cost,
            theirs.parallelism,
        )


def test_legacy_helpers():
    encoded = mojo.hash_password(
        b"password",
        salt=b"0123456789abcdef",
        time_cost=1,
        memory_cost=16,
        parallelism=1,
        hash_len=16,
        type=mojo.Type.ID,
    )
    assert mojo.verify_password(encoded, b"password", mojo.Type.ID)
    assert mojo.hash_password_raw(
        b"password",
        salt=b"0123456789abcdef",
        time_cost=1,
        memory_cost=16,
        parallelism=1,
        hash_len=16,
        type=mojo.Type.ID,
    ) == upstream_ll.hash_secret_raw(
        b"password", b"0123456789abcdef", 1, 16, 1, 16, upstream.Type.ID
    )


def test_explicit_empty_salt_is_not_replaced():
    """Upstream `salt or os.urandom(...)`, so `salt=b""` is REPLACED, not rejected.

    _password_hasher.py: ``salt=salt or os.urandom(self.salt_len)``.  The old
    body here expected HashingError("Salt is too short"); upstream never
    reaches validation with an empty salt.  The property the name asserts
    (an explicitly passed *non-empty* salt is used verbatim) is kept below.
    """
    hasher = mojo.PasswordHasher(time_cost=1, memory_cost=8, parallelism=1)
    encoded = hasher.hash("password", salt=b"")
    assert hasher.verify(encoded, "password")
    assert mojo.extract_parameters(encoded).salt_len == hasher.salt_len

    fixed = hasher.hash("password", salt=b"01234567")
    assert mojo.extract_parameters(fixed).salt_len == 8
    assert hasher.hash("password", salt=b"01234567") == fixed

    with pytest.raises(mojo.exceptions.HashingError, match="Salt is too short"):
        mojo_ll.hash_secret_raw(b"password", b"", 1, 8, 1, 16, mojo.Type.ID)


def test_rejects_values_that_narrow_at_the_ffi():
    """Upstream CFFI overflows on any int that does not fit uint32.

    low_level.py passes ``time_cost`` straight to ``ffi.new``/lib calls, so
    1 << 32 raises OverflowError("integer ... does not fit '32-bit unsigned
    int'") before Argon2 runs.  The old body expected HashingError("time_cost
    is too large"), which upstream never raises for this input.
    """
    with pytest.raises(OverflowError, match="does not fit '32-bit unsigned int'"):
        mojo_ll.hash_secret_raw(
            b"password",
            b"01234567",
            1 << 32,
            8,
            1,
            16,
            mojo.Type.ID,
        )
    with pytest.raises(OverflowError, match="does not fit '32-bit unsigned int'"):
        mojo_ll.hash_secret_raw(
            b"password",
            b"01234567",
            1,
            1 << 32,
            1,
            16,
            mojo.Type.ID,
        )


def test_export_rejects_null_and_undersized_buffers():
    fn = mojo_bridge.lib().mojo_argon2_hash
    assert fn(*([0] * 17)) == -1

    import ctypes

    byte = ctypes.c_uint8(0)
    address = ctypes.addressof(byte)
    args = [
        address, 48, address, 0, address, 0, address, 0, address, 0,
        1, 8, 1, 4, mojo.Type.ID.value, 19, 0,
    ]
    assert fn(*args) == -1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"salt": b"short"}, "Salt is too short"),
        ({"time_cost": 0}, "Time cost is too small"),
        ({"memory_cost": 7}, "Memory cost is too small"),
        ({"hash_len": 3}, "Output is too short"),
        (
            {"parallelism": 0x1000000, "memory_cost": 0x8000000},
            "Too many lanes",
        ),
    ],
)
def test_invalid_parameters(kwargs, message):
    values = dict(
        secret=b"password",
        salt=b"01234567",
        time_cost=1,
        memory_cost=8,
        parallelism=1,
        hash_len=16,
        type=mojo.Type.ID,
    )
    values.update(kwargs)
    with pytest.raises(mojo.exceptions.HashingError, match=message):
        mojo_ll.hash_secret_raw(**values)


def test_invalid_encoded_hash():
    with pytest.raises(mojo.exceptions.InvalidHashError):
        mojo.extract_parameters("$not-argon2$")
    with pytest.raises(mojo.exceptions.InvalidHashError):
        mojo.PasswordHasher().verify("$not-argon2$", "password")


# ---------------------------------------------------------------------------
# Reference comparisons.
#
# Every public function is checked twice where possible:
#   * against the installed argon2-cffi package (live upstream), and
#   * against tests/upstream_fixtures.py, frozen outputs produced by
#     argon2-cffi 25.1.0 itself, so parity is still proven without upstream.
# Comparisons use an explicit tolerance, even though Argon2 output is exact:
#     assert bytes_parity(...) is True  and  max |diff| <= BYTE_TOLERANCE == 0
# ---------------------------------------------------------------------------

import math
import random
import re

import numpy as np

from argon2 import _password_hasher as upstream_hasher_module
from argon2 import _utils as upstream_utils

from tests import upstream_fixtures as fixtures


BYTE_TOLERANCE = 0  # byte-for-byte: the only tolerance Argon2 permits
INT_TOLERANCE = 0  # costs/lengths are exact integers upstream


def counts_within_tolerance(diffs):
    """Integer parameters (lanes, lengths, costs) must match upstream exactly.

    Upstream rounds memory with integer arithmetic, so there is no legitimate
    rounding slack: the tolerance is 0, stated rather than assumed.
    """
    return max(diffs, default=0) <= INT_TOLERANCE


def bytes_within_tolerance(ours, theirs, tolerance=BYTE_TOLERANCE):
    if len(ours) != len(theirs):
        return False
    diff = np.frombuffer(ours, dtype=np.uint8).astype(np.int16) - np.frombuffer(
        theirs, dtype=np.uint8
    ).astype(np.int16)
    return int(np.abs(diff).max(initial=0)) <= tolerance


def _comparable(value):
    """Strip module identity from results shared with upstream.

    `mojo_argon2.Type`/`Parameters` and `argon2.Type`/`Parameters` are distinct
    classes, so compare their fields instead of their identities.
    """
    if hasattr(value, "__dataclass_fields__"):
        return type(value).__name__, tuple(
            _comparable(getattr(value, field)) for field in value.__dataclass_fields__
        )
    if hasattr(value, "name") and hasattr(value, "value"):
        return type(value).__name__, value.value
    return value


def _outcome(fn, *args, **kwargs):
    try:
        return ("ok", _comparable(fn(*args, **kwargs)))
    except Exception as exc:  # noqa: BLE001 - the class is what we compare
        # mojo_argon2 and argon2-cffi define parallel exception hierarchies,
        # so a matching class is one with the same name, not the same object.
        return ("raised", type(exc).__name__)


def _upstream_version_number():
    """Upstream's ARGON2_VERSION_NUMBER from the vendored _argon2_cffi_bindings."""
    import _argon2_cffi_bindings

    return int(_argon2_cffi_bindings.lib.ARGON2_VERSION_NUMBER)


# --- low_level.hash_secret_raw -------------------------------------------


@pytest.mark.parametrize(
    (
        "secret",
        "salt",
        "time_cost",
        "memory_cost",
        "parallelism",
        "hash_len",
        "type_value",
        "version",
        "expected_hex",
    ),
    fixtures.RAW_VECTORS,
)
def test_hash_secret_raw_matches_frozen_upstream_fixture(
    secret,
    salt,
    time_cost,
    memory_cost,
    parallelism,
    hash_len,
    type_value,
    version,
    expected_hex,
):
    expected = bytes.fromhex(expected_hex)
    actual = mojo_ll.hash_secret_raw(
        secret,
        salt,
        time_cost,
        memory_cost,
        parallelism,
        hash_len,
        mojo.Type(type_value),
        version,
    )
    assert len(actual) == hash_len
    assert bytes_within_tolerance(actual, expected)
    assert actual == upstream_ll.hash_secret_raw(
        secret,
        salt,
        time_cost,
        memory_cost,
        parallelism,
        hash_len,
        upstream.Type(type_value),
        version,
    )


@pytest.mark.parametrize("version", [16, 17, 18, 19, 20, 21, 31, 255])
@pytest.mark.parametrize(("mojo_type", "upstream_type"), TYPE_PAIRS)
def test_any_version_upstream_accepts(version, mojo_type, upstream_type):
    kwargs = dict(
        secret=b"version field",
        salt=b"0123456789abcdef",
        time_cost=2,
        memory_cost=32,
        parallelism=2,
        hash_len=32,
    )
    ours = mojo_ll.hash_secret_raw(type=mojo_type, version=version, **kwargs)
    theirs = upstream_ll.hash_secret_raw(
        type=upstream_type, version=version, **kwargs
    )
    assert bytes_within_tolerance(ours, theirs)


def test_default_version_matches_upstream_constant():
    assert mojo_ll.ARGON2_VERSION == _upstream_version_number()
    assert mojo_ll.ARGON2_VERSION == upstream_ll.ARGON2_VERSION


# --- low_level.hash_secret ----------------------------------------------


@pytest.mark.parametrize(
    (
        "secret",
        "salt",
        "time_cost",
        "memory_cost",
        "parallelism",
        "hash_len",
        "type_value",
        "expected",
    ),
    fixtures.ENCODED_VECTORS,
)
def test_hash_secret_matches_frozen_upstream_fixture(
    secret, salt, time_cost, memory_cost, parallelism, hash_len, type_value, expected
):
    actual = mojo_ll.hash_secret(
        secret,
        salt,
        time_cost,
        memory_cost,
        parallelism,
        hash_len,
        mojo.Type(type_value),
    )
    assert actual == expected
    assert bytes_within_tolerance(actual, expected)
    assert actual == upstream_ll.hash_secret(
        secret,
        salt,
        time_cost,
        memory_cost,
        parallelism,
        hash_len,
        upstream.Type(type_value),
    )


# --- low_level.verify_secret ---------------------------------------------


def test_verify_secret_accepts_upstream_output():
    encoded = upstream_ll.hash_secret(
        b"password", b"01234567", 1, 8, 1, 16, upstream.Type.ID
    )
    assert mojo_ll.verify_secret(encoded, b"password", mojo.Type.ID) is True
    assert upstream_ll.verify_secret(encoded, b"password", upstream.Type.ID) is True


def _one_byte_mutations(encoded):
    yield from (encoded[:i] for i in range(len(encoded) + 1))
    for i in range(len(encoded)):
        yield encoded[:i] + b"x" + encoded[i + 1 :]


def test_verify_secret_matches_upstream_on_every_one_byte_mutation():
    encoded = upstream_ll.hash_secret(
        b"password", b"01234567", 1, 8, 1, 16, upstream.Type.ID
    )
    mutations = list(_one_byte_mutations(encoded))
    assert len(mutations) > 50
    for candidate in mutations:
        ours = _outcome(mojo_ll.verify_secret, candidate, b"password", mojo.Type.ID)
        theirs = _outcome(
            upstream_ll.verify_secret, candidate, b"password", upstream.Type.ID
        )
        assert ours == theirs, candidate


def test_verify_secret_rejects_non_canonical_base64_like_upstream():
    encoded = upstream_ll.hash_secret(
        b"password", b"01234567", 1, 8, 1, 16, upstream.Type.ID
    )
    head = encoded.rsplit(b"$", 1)[0]
    # argon2.c's from_base64 rejects a final group with non-zero discard bits.
    for tail in (b"xOlTsn", b"xOlTsnS", b"xOlTsnSovD"):
        candidate = head + b"$" + tail
        with pytest.raises(mojo.exceptions.VerificationError, match="Decoding failed"):
            mojo_ll.verify_secret(candidate, b"password", mojo.Type.ID)
        with pytest.raises(upstream.exceptions.VerificationError, match="Decoding failed"):
            upstream_ll.verify_secret(candidate, b"password", upstream.Type.ID)


# --- low_level.error_to_str ---------------------------------------------


@pytest.mark.parametrize("code", sorted(fixtures.ERROR_STRINGS))
def test_error_to_str_matches_frozen_upstream_fixture(code):
    expected = fixtures.ERROR_STRINGS[code]
    assert mojo_ll.error_to_str(code) == expected
    assert mojo_ll.error_to_str(code) == upstream_ll.error_to_str(code)


@pytest.mark.parametrize("code", [1, 99, -36, -100, -(2**31) + 1])
def test_error_to_str_unknown_codes(code):
    assert mojo_ll.error_to_str(code) == "Unknown error code"
    assert mojo_ll.error_to_str(code) == upstream_ll.error_to_str(code)


@pytest.mark.parametrize("code", [2**31, -(2**31) - 1, 2**40])
def test_error_to_str_out_of_int32_range_overflows(code):
    with pytest.raises(OverflowError):
        mojo_ll.error_to_str(code)
    with pytest.raises(OverflowError):
        upstream_ll.error_to_str(code)


# --- PasswordHasher -----------------------------------------------------


def test_password_hasher_hash_matches_upstream_bit_for_bit():
    salt = b"0123456789abcdef"
    for type_value in (mojo.Type.D, mojo.Type.I, mojo.Type.ID):
        ours = mojo.PasswordHasher(
            time_cost=2,
            memory_cost=32,
            parallelism=2,
            hash_len=24,
            salt_len=len(salt),
            type=type_value,
        ).hash("pässword", salt=salt)
        theirs = upstream.PasswordHasher(
            time_cost=2,
            memory_cost=32,
            parallelism=2,
            hash_len=24,
            salt_len=len(salt),
            type=upstream.Type(type_value.value),
        ).hash("pässword", salt=salt)
        assert ours.encode() == theirs.encode()
        assert bytes_within_tolerance(ours.encode(), theirs.encode())


def test_password_hasher_random_salt_matches_upstream_patch(monkeypatch):
    def fake_urandom(n):
        return bytes(range(n))

    monkeypatch.setattr(os, "urandom", fake_urandom)
    for module, name in ((mojo, "mojo"), (upstream, "upstream")):
        hasher = module.PasswordHasher(
            time_cost=1, memory_cost=8, parallelism=1, hash_len=16, salt_len=16
        )
        if name == "mojo":
            ours = hasher.hash("password")
        else:
            theirs = hasher.hash("password")
    assert ours == theirs


def test_password_hasher_falsy_salt_uses_a_random_salt():
    hasher = mojo.PasswordHasher(
        time_cost=1, memory_cost=8, parallelism=1, hash_len=16, salt_len=16
    )
    encoded = mojo.extract_parameters(hasher.hash("password", salt=b""))
    assert encoded.salt_len == 16
    assert mojo.PasswordHasher(
        time_cost=1, memory_cost=8, parallelism=1, hash_len=16, salt_len=16
    ).verify(hasher.hash("password", salt=b""), "password")


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(salt_len=3),
        dict(time_cost=0),
        dict(memory_cost=1),
        dict(parallelism=0),
        dict(hash_len=1),
    ],
)
def test_password_hasher_constructor_defers_validation_like_upstream(kwargs):
    # upstream validates types only in __init__; numeric ranges surface at hash().
    upstream_hasher = upstream.PasswordHasher(**kwargs)
    mojo_hasher = mojo.PasswordHasher(**kwargs)
    for attr in (
        "time_cost",
        "memory_cost",
        "parallelism",
        "hash_len",
        "salt_len",
    ):
        assert counts_within_tolerance(
            [abs(getattr(mojo_hasher, attr) - getattr(upstream_hasher, attr))]
        )


def test_password_hasher_type_error_matches_upstream():
    with pytest.raises(TypeError):
        mojo.PasswordHasher(time_cost=1.5)
    with pytest.raises(TypeError):
        upstream.PasswordHasher(time_cost=1.5)
    assert upstream_hasher_module._check_types(time_cost=(1.5, int)) is not None


def test_password_hasher_properties_match_upstream():
    kwargs = dict(
        time_cost=3, memory_cost=65_536, parallelism=4, hash_len=32, salt_len=16
    )
    ours = mojo.PasswordHasher(**kwargs)
    theirs = upstream.PasswordHasher(**kwargs)
    for attr in ("time_cost", "memory_cost", "parallelism", "hash_len", "salt_len"):
        assert counts_within_tolerance([abs(getattr(ours, attr) - getattr(theirs, attr))])


def test_default_parameters_match_upstream():
    module = sys.modules["mojo_argon2"]
    upstream_module = upstream
    for attr in (
        "DEFAULT_TIME_COST",
        "DEFAULT_MEMORY_COST",
        "DEFAULT_PARALLELISM",
        "DEFAULT_HASH_LENGTH",
        "DEFAULT_RANDOM_SALT_LENGTH",
    ):
        assert counts_within_tolerance(
            [abs(getattr(module, attr) - getattr(upstream_module, attr))]
        )


def test_from_parameters_matches_upstream():
    params = mojo.Parameters(mojo.Type.I, 19, 8, 16, 2, 16, 1)
    upstream_params = upstream.Parameters(upstream.Type.I, 19, 8, 16, 2, 16, 1)
    ours = mojo.PasswordHasher.from_parameters(params)
    theirs = upstream.PasswordHasher.from_parameters(upstream_params)
    for attr in ("time_cost", "memory_cost", "parallelism", "hash_len", "salt_len"):
        assert counts_within_tolerance([abs(getattr(ours, attr) - getattr(theirs, attr))])


@pytest.mark.parametrize(
    "hash_input",
    ["garbage", b"garbage", "$argon2d$x", "$argon2i$v=19$m=8,t=1,p=1$", ""],
)
def test_password_hasher_verify_error_classes_match_upstream(hash_input):
    ours = _outcome(
        mojo.PasswordHasher(time_cost=1, memory_cost=8, parallelism=1).verify,
        hash_input,
        "password",
    )
    theirs = _outcome(
        upstream.PasswordHasher(time_cost=1, memory_cost=8, parallelism=1).verify,
        hash_input,
        "password",
    )
    assert ours == theirs


@pytest.mark.parametrize("password", [bytearray(b"pw"), None, 42])
def test_password_hasher_verify_non_str_password_matches_upstream(password):
    # verify() only reaches _ensure_bytes after the hash header is recognised,
    # so a well-formed hash is what puts a non-str password on upstream's
    # AttributeError path (a bad header raises InvalidHashError first).
    encoded = upstream_ll.hash_secret(
        b"pw", b"01234567", 1, 8, 1, 8, upstream.Type.ID
    )
    ours = _outcome(
        mojo.PasswordHasher(time_cost=1, memory_cost=8, parallelism=1).verify,
        encoded,
        password,
    )
    theirs = _outcome(
        upstream.PasswordHasher(time_cost=1, memory_cost=8, parallelism=1).verify,
        encoded,
        password,
    )
    assert ours[0] == theirs[0] == "raised"
    # upstream calls `.encode` on anything that is not bytes -> AttributeError
    assert ours == ("raised", "AttributeError")
    assert theirs == ("raised", "AttributeError")


def test_check_needs_rehash_matches_upstream():
    hasher_kwargs = dict(time_cost=1, memory_cost=8, parallelism=1, hash_len=16)
    mojo_hasher = mojo.PasswordHasher(**hasher_kwargs)
    upstream_hasher = upstream.PasswordHasher(**hasher_kwargs)
    for time_cost in (1, 2):
        for type_value in (mojo.Type.D, mojo.Type.I, mojo.Type.ID):
            encoded = mojo_ll.hash_secret(
                b"pw", b"01234567", time_cost, 8, 1, 16, type_value
            )
            assert mojo_hasher.check_needs_rehash(encoded) == (
                upstream_hasher.check_needs_rehash(encoded)
            )
            assert mojo_hasher.check_needs_rehash(encoded.decode()) == (
                upstream_hasher.check_needs_rehash(encoded.decode())
            )


# --- extract_parameters -------------------------------------------------


_EXTRACT_CASES = [
    "$argon2id$v=19$m=32,t=2,p=2$MDEyMzQ1Njc4OWFiY2RlZg$Gc2A3YzW1gc0ZzoW2hm8ynhVWDRYUZO4",
    "$argon2i$m=8,t=1,p=1$c2FsdA$c2FsdB",  # legacy v1.2 form
    "$argon2id$v=19$m=8,t=1,p=1$!!!!$AAAAAAAAAAAAAAAAAAAAAA",  # not base64
    "$argon2d$v=19$m=8,t=1,p=1$$",  # empty salt and hash
    "$argon2id$v=19$m=8,t=1,p=1$abc$abc",  # lengths that cannot be base64
]


def _parameter_fields(params):
    return (
        params.version,
        params.salt_len,
        params.hash_len,
        params.time_cost,
        params.memory_cost,
        params.parallelism,
    )


@pytest.mark.parametrize("encoded", _EXTRACT_CASES)
def test_extract_parameters_matches_upstream(encoded):
    ours = mojo.extract_parameters(encoded)
    theirs = upstream.extract_parameters(encoded)
    assert ours.type.name == theirs.type.name
    for actual, expected in zip(_parameter_fields(ours), _parameter_fields(theirs)):
        assert counts_within_tolerance([abs(actual - expected)])
    assert _decoded_str_len_matches_upstream(encoded)


def _decoded_str_len_matches_upstream(encoded):
    parts = encoded.split("$")
    if len(parts) == 5:
        parts.insert(2, "v=18")
    return len(parts) == 6 and all(
        _upstream_decoded_len(len(part)) == mojo._utils._decoded_len(len(part))
        for part in parts[4:]
    )


def _upstream_decoded_len(length):
    return upstream_utils._decoded_str_len(length)


@pytest.mark.parametrize(
    "encoded",
    [
        "$not-argon2$",
        "",
        "argon2id$v=19$m=8,t=1,p=1$c2FsdA$c2FsdB",
        "$argon2id$v=19$m=8,t=1$c2FsdA$c2FsdB",
        "$argon2id$v=19$m=8,t=1,p=1,x=1$c2FsdA$c2FsdB",
        "$argon2id$v=19$m=a,t=1,p=1$c2FsdA$c2FsdB",
        "$argon2x$v=19$m=8,t=1,p=1$c2FsdA$c2FsdB",
    ],
)
def test_extract_parameters_error_classes_match_upstream(encoded):
    assert _outcome(mojo.extract_parameters, encoded) == _outcome(
        upstream.extract_parameters, encoded
    )


# --- legacy helpers -----------------------------------------------------


def test_legacy_helpers_match_upstream():
    kwargs = dict(
        salt=b"0123456789abcdef",
        time_cost=1,
        memory_cost=16,
        parallelism=1,
        hash_len=16,
    )
    for type_value in (mojo.Type.D, mojo.Type.I, mojo.Type.ID):
        ours = mojo.hash_password(b"pw", type=type_value, **kwargs)
        theirs = upstream.hash_password(
            b"pw", type=upstream.Type(type_value.value), **kwargs
        )
        assert ours == theirs
        assert bytes_within_tolerance(ours, theirs)
        ours_raw = mojo.hash_password_raw(b"pw", type=type_value, **kwargs)
        theirs_raw = upstream.hash_password_raw(
            b"pw", type=upstream.Type(type_value.value), **kwargs
        )
        assert ours_raw == theirs_raw
        assert mojo.verify_password(ours, b"pw", type_value) == (
            upstream.verify_password(ours, b"pw", type_value)
        )
        assert _outcome(mojo.verify_password, ours, b"nope", type_value) == _outcome(
            upstream.verify_password, ours, b"nope", type_value
        )


# --- degenerate inputs --------------------------------------------------
#
# Argon2 takes bytes and scalars, not arrays.  "Shapes" here are the byte
# lengths of the argument buffers: empty, one element (one byte), duplicates
# (repeated bytes), and mismatched shapes (salt or output lengths that do not
# match upstream's requirements).  Non-integral and non-finite values stand in
# for NaN/inf.  Each must raise the same class or return the same value.

_DEGENERATE_BASE = dict(
    secret=b"p",
    salt=b"01234567",
    time_cost=1,
    memory_cost=8,
    parallelism=1,
    hash_len=4,
)

_DEGENERATE = [
    pytest.param(dict(secret=b""), id="empty-secret"),
    pytest.param(dict(secret=b"p"), id="one-byte-secret"),
    pytest.param(dict(secret=b"a" * 129), id="duplicated-secret-over-blake-block"),
    pytest.param(dict(hash_len=4), id="shortest-tag"),
    pytest.param(dict(hash_len=1024), id="longest-auto-tag"),
    pytest.param(dict(salt=b""), id="empty-salt"),
    pytest.param(dict(salt=b"1234567"), id="one-byte-short-salt"),
    pytest.param(dict(salt=b"aaaaaaaa"), id="duplicated-salt"),
    pytest.param(dict(salt=b"0123456789abcdef" * 8), id="long-salt"),
    pytest.param(dict(hash_len=3), id="tag-one-below-minimum"),
    pytest.param(dict(time_cost=0), id="zero-time"),
    pytest.param(dict(time_cost=-1), id="negative-time"),
    pytest.param(dict(memory_cost=7), id="memory-below-one-page"),
    pytest.param(dict(memory_cost=0), id="zero-memory"),
    pytest.param(dict(parallelism=0), id="zero-lanes"),
    pytest.param(dict(parallelism=2, memory_cost=15), id="mlocks-lane-mismatch"),
    pytest.param(dict(parallelism=2, memory_cost=17), id="memory-rounds-to-lanes"),
    pytest.param(dict(parallelism=0x1000000, memory_cost=8), id="too-many-lanes"),
    pytest.param(dict(version=0), id="version-zero"),
    # No 2**31 cost/size cases here: upstream starts a genuine multi-GiB or
    # 2^31-iteration KDF instead of raising.  Uint32 range is covered by
    # _DEGENERATE_UINT32 and test_error_to_str_out_of_int32_range_overflows.
    pytest.param(dict(time_cost=2**32), id="time-over-uint32"),
    pytest.param(dict(memory_cost=2**32), id="memory-over-uint32"),
    pytest.param(dict(parallelism=2**32), id="lanes-over-uint32"),
    pytest.param(dict(hash_len=2**32), id="hash-len-over-uint32"),
    pytest.param(dict(version=2**32), id="version-over-uint32"),
    pytest.param(dict(version=-1), id="version-negative"),
    pytest.param(dict(memory_cost=-1), id="memory-negative"),
    pytest.param(dict(hash_len=-1), id="hash-len-negative"),
]


@pytest.mark.parametrize(("mojo_type", "upstream_type"), TYPE_PAIRS)
@pytest.mark.parametrize("override", _DEGENERATE)
def test_degenerate_hash_secret_raw_matches_upstream(
    override, mojo_type, upstream_type
):
    kwargs = {**_DEGENERATE_BASE, **override}
    ours = _outcome(mojo_ll.hash_secret_raw, type=mojo_type, **kwargs)
    theirs = _outcome(upstream_ll.hash_secret_raw, type=upstream_type, **kwargs)
    assert ours == theirs
    if ours[0] == "ok":
        assert bytes_within_tolerance(ours[1], theirs[1])


@pytest.mark.parametrize("bad", ["1234567", 12345678, None, bytearray(b"01234567")])
@pytest.mark.parametrize("position", ["secret", "salt"])
def test_degenerate_hash_secret_raw_non_bytes_matches_upstream(bad, position):
    kwargs = {**_DEGENERATE_BASE, position: bad}
    ours = _outcome(
        mojo_ll.hash_secret_raw, type=mojo.Type.ID, **kwargs
    )
    theirs = _outcome(
        upstream_ll.hash_secret_raw, type=upstream.Type.ID, **kwargs
    )
    assert ours == theirs


@pytest.mark.parametrize("bad_type", [0, 1, 2, "argon2id", None])
def test_degenerate_hash_secret_raw_type_argument(bad_type):
    ours = _outcome(
        mojo_ll.hash_secret_raw, type=bad_type, **_DEGENERATE_BASE
    )
    theirs = _outcome(
        upstream_ll.hash_secret_raw, type=bad_type, **_DEGENERATE_BASE
    )
    assert ours == theirs


@pytest.mark.parametrize(
    "bad",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        -(2**31) - 1,
        2**32,
        True,
        1.0,
        np.float64(3.0),
        np.int64(3),
    ],
)
@pytest.mark.parametrize(
    "position", ["time_cost", "memory_cost", "parallelism", "hash_len", "version"]
)
def test_degenerate_non_finite_and_non_int_parameters(bad, position):
    kwargs = {**_DEGENERATE_BASE, position: bad}
    ours = _outcome(
        mojo_ll.hash_secret, type=mojo.Type.ID, **kwargs
    )
    theirs = _outcome(
        upstream_ll.hash_secret, type=upstream.Type.ID, **kwargs
    )
    assert ours == theirs


@pytest.mark.parametrize(
    "bad",
    [float("nan"), float("inf"), -(2**31) - 1, 2**32],
)
def test_degenerate_password_hasher_parameters_match_upstream(bad):
    for field in ("time_cost", "memory_cost", "parallelism", "hash_len", "salt_len"):
        ours = _outcome(lambda: mojo.PasswordHasher(**{field: bad})._parameters)
        theirs = _outcome(lambda: upstream.PasswordHasher(**{field: bad})._parameters)
        # NaN/inf are stored verbatim (no __init__ range check upstream) and
        # only fail later inside hash(); compare the stored fields, not ids.
        assert ours[0] == theirs[0], field
        if ours[0] == "ok":
            assert ours[1][0] == "Parameters" == theirs[1][0]
            assert len(ours[1][1]) == len(theirs[1][1]) == 7


@pytest.mark.parametrize(
    "duplicate",
    [
        "$argon2id$v=19$m=8,t=1,p=1,p=1$c2FsdA$c2FsdB",
        "$argon2id$v=19$m=8,m=8,t=1,p=1$c2FsdA$c2FsdB",
        "$argon2id$v=19$v=19,m=8,t=1,p=1$c2FsdA$c2FsdB",
    ],
)
def test_duplicate_parameter_keys_match_upstream(duplicate):
    # upstream builds a dict comprehension, so a repeated key silently wins.
    assert _outcome(mojo.extract_parameters, duplicate) == _outcome(
        upstream.extract_parameters, duplicate
    )
    assert _outcome(mojo.extract_parameters, duplicate)[0] == "ok"


def test_empty_encoded_hash_verify_matches_upstream():
    for encoded in (b"", ""):
        for module, hasher_cls in (
            (mojo, mojo.PasswordHasher),
            (upstream, upstream.PasswordHasher),
        ):
            outcome = _outcome(
                hasher_cls(time_cost=1, memory_cost=8, parallelism=1).verify,
                encoded,
                "pw",
            )
            if module is mojo:
                ours = outcome
            else:
                assert ours == outcome
