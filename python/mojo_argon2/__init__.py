"""A Mojo implementation of argon2-cffi's KDF-facing API."""

from . import exceptions, low_level, profiles
from ._password_hasher import (
    DEFAULT_HASH_LENGTH,
    DEFAULT_MEMORY_COST,
    DEFAULT_PARALLELISM,
    DEFAULT_RANDOM_SALT_LENGTH,
    DEFAULT_TIME_COST,
    PasswordHasher,
)
from ._utils import Parameters, extract_parameters
from .low_level import Type


def hash_password(
    password: bytes,
    salt: bytes | None = None,
    time_cost: int = DEFAULT_TIME_COST,
    memory_cost: int = DEFAULT_MEMORY_COST,
    parallelism: int = DEFAULT_PARALLELISM,
    hash_len: int = DEFAULT_HASH_LENGTH,
    type: Type = Type.I,
) -> bytes:
    import os

    return low_level.hash_secret(
        password,
        salt if salt is not None else os.urandom(DEFAULT_RANDOM_SALT_LENGTH),
        time_cost,
        memory_cost,
        parallelism,
        hash_len,
        type,
    )


def hash_password_raw(
    password: bytes,
    salt: bytes | None = None,
    time_cost: int = DEFAULT_TIME_COST,
    memory_cost: int = DEFAULT_MEMORY_COST,
    parallelism: int = DEFAULT_PARALLELISM,
    hash_len: int = DEFAULT_HASH_LENGTH,
    type: Type = Type.I,
) -> bytes:
    import os

    return low_level.hash_secret_raw(
        password,
        salt if salt is not None else os.urandom(DEFAULT_RANDOM_SALT_LENGTH),
        time_cost,
        memory_cost,
        parallelism,
        hash_len,
        type,
    )


def verify_password(hash: bytes, password: bytes, type: Type = Type.I) -> bool:
    return low_level.verify_secret(hash, password, type)


__all__ = [
    "DEFAULT_HASH_LENGTH",
    "DEFAULT_MEMORY_COST",
    "DEFAULT_PARALLELISM",
    "DEFAULT_RANDOM_SALT_LENGTH",
    "DEFAULT_TIME_COST",
    "Parameters",
    "PasswordHasher",
    "Type",
    "exceptions",
    "extract_parameters",
    "hash_password",
    "hash_password_raw",
    "low_level",
    "profiles",
    "verify_password",
]
