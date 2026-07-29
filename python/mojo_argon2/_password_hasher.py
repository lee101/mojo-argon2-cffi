"""High-level password hashing API compatible with argon2-cffi."""

from __future__ import annotations

import os

from ._utils import Parameters, extract_parameters
from .exceptions import InvalidHashError
from .low_level import Type, hash_secret, verify_secret
from .profiles import get_default_parameters


_DEFAULT = get_default_parameters()
DEFAULT_RANDOM_SALT_LENGTH = _DEFAULT.salt_len
DEFAULT_HASH_LENGTH = _DEFAULT.hash_len
DEFAULT_TIME_COST = _DEFAULT.time_cost
DEFAULT_MEMORY_COST = _DEFAULT.memory_cost
DEFAULT_PARALLELISM = _DEFAULT.parallelism


class PasswordHasher:
    __slots__ = ("_parameters", "encoding")

    def __init__(
        self,
        time_cost: int = DEFAULT_TIME_COST,
        memory_cost: int = DEFAULT_MEMORY_COST,
        parallelism: int = DEFAULT_PARALLELISM,
        hash_len: int = DEFAULT_HASH_LENGTH,
        salt_len: int = DEFAULT_RANDOM_SALT_LENGTH,
        encoding: str = "utf-8",
        type: Type = Type.ID,
    ):
        values = {
            "time_cost": time_cost,
            "memory_cost": memory_cost,
            "parallelism": parallelism,
            "hash_len": hash_len,
            "salt_len": salt_len,
        }
        for name, value in values.items():
            if not isinstance(value, int):
                raise TypeError(f"'{name}' must be an int")
        if not isinstance(encoding, str):
            raise TypeError("'encoding' must be a str")
        if not isinstance(type, Type):
            raise TypeError("'type' must be a Type")
        if salt_len < 8:
            raise ValueError("salt_len must be at least 8")
        self._parameters = Parameters(
            type, 19, salt_len, hash_len, time_cost, memory_cost, parallelism
        )
        self.encoding = encoding

    @classmethod
    def from_parameters(cls, params: Parameters) -> PasswordHasher:
        return cls(
            params.time_cost,
            params.memory_cost,
            params.parallelism,
            params.hash_len,
            params.salt_len,
            type=params.type,
        )

    @property
    def time_cost(self) -> int:
        return self._parameters.time_cost

    @property
    def memory_cost(self) -> int:
        return self._parameters.memory_cost

    @property
    def parallelism(self) -> int:
        return self._parameters.parallelism

    @property
    def hash_len(self) -> int:
        return self._parameters.hash_len

    @property
    def salt_len(self) -> int:
        return self._parameters.salt_len

    @property
    def type(self) -> Type:
        return self._parameters.type

    def hash(self, password: str | bytes, *, salt: bytes | None = None) -> str:
        password_bytes = (
            password if isinstance(password, bytes) else password.encode(self.encoding)
        )
        if not isinstance(password_bytes, bytes):
            raise TypeError("password must be str or bytes")
        if salt is not None and not isinstance(salt, bytes):
            raise TypeError("salt must be bytes")
        return hash_secret(
            password_bytes,
            salt if salt is not None else os.urandom(self.salt_len),
            self.time_cost,
            self.memory_cost,
            self.parallelism,
            self.hash_len,
            self.type,
        ).decode("ascii")

    def verify(self, hash: str | bytes, password: str | bytes) -> bool:
        encoded = hash.encode("ascii") if isinstance(hash, str) else hash
        if not isinstance(encoded, bytes):
            raise TypeError("hash must be str or bytes")
        secret = password.encode(self.encoding) if isinstance(password, str) else password
        if not isinstance(secret, bytes):
            raise TypeError("password must be str or bytes")
        headers = {
            b"$argon2d$": Type.D,
            b"$argon2i$": Type.I,
            b"$argon2id": Type.ID,
        }
        try:
            type_value = headers[encoded[:9]]
        except KeyError:
            raise InvalidHashError from None
        return verify_secret(encoded, secret, type_value)

    def check_needs_rehash(self, hash: str | bytes) -> bool:
        value = hash.decode("ascii") if isinstance(hash, bytes) else hash
        return self._parameters != extract_parameters(value)
