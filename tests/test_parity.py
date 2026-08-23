"""Byte-for-byte and behavioral parity with argon2-cffi."""

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


def test_parallel_runtime_failure_falls_back(monkeypatch):
    monkeypatch.setattr(
        mojo_bridge, "_ensure_parallel_runtime", lambda: False
    )
    kwargs = dict(
        secret=b"runtime fallback",
        salt=b"0123456789abcdef",
        time_cost=1,
        memory_cost=2_048,
        parallelism=2,
        hash_len=32,
    )
    assert mojo_ll.hash_secret_raw(type=mojo.Type.ID, **kwargs) == (
        upstream_ll.hash_secret_raw(type=upstream.Type.ID, **kwargs)
    )


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
    with pytest.raises(mojo.exceptions.HashingError, match="Salt is too short"):
        mojo.PasswordHasher(time_cost=1, memory_cost=8, parallelism=1).hash(
            "password", salt=b""
        )


def test_rejects_values_that_narrow_at_the_ffi():
    with pytest.raises(mojo.exceptions.HashingError, match="time_cost is too large"):
        mojo_ll.hash_secret_raw(
            b"password",
            b"01234567",
            1 << 32,
            8,
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
