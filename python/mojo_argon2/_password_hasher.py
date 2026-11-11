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


def _ensure_bytes(s: bytes | str, encoding: str) -> bytes:
    # same contract as upstream's _password_hasher._ensure_bytes
    if isinstance(s, bytes):
        return s
    return s.encode(encoding)


def _check_types(**kw) -> str | None:
    # upstream _utils._check_types: collects every violation into one message
    errors = []
    for name, (value, expected) in kw.items():
        if not isinstance(value, expected):
            if isinstance(expected, tuple):
                types = ", or ".join(t.__name__ for t in expected)
            else:
                types = expected.__name__
            errors.append(f"'{name}' must be a {types} (got {type(value).__name__})")

    if errors != []:
        return ", ".join(errors) + "."

    return None


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
        # upstream validates by isinstance only here: bool passes, float does not.
        e = _check_types(
            time_cost=(time_cost, int),
            memory_cost=(memory_cost, int),
            parallelism=(parallelism, int),
            hash_len=(hash_len, int),
            salt_len=(salt_len, int),
            encoding=(encoding, str),
            type=(type, Type),
        )
        if e:
            raise TypeError(e)
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
        # upstream uses `salt or os.urandom(...)`, so b"" also gets a random salt
        return hash_secret(
            _ensure_bytes(password, self.encoding),
            salt or os.urandom(self.salt_len),
            self.time_cost,
            self.memory_cost,
            self.parallelism,
            self.hash_len,
            self.type,
        ).decode("ascii")

    def verify(self, hash: str | bytes, password: str | bytes) -> bool:
        encoded = _ensure_bytes(hash, "ascii")
        secret = _ensure_bytes(password, self.encoding)
        headers = {
            b"$argon2d$": Type.D,
            b"$argon2i$": Type.I,
            b"$argon2id": Type.ID,
        }
        try:
            type_value = headers[encoded[:9]]
        except LookupError:
            raise InvalidHashError from None
        return verify_secret(encoded, secret, type_value)

    def check_needs_rehash(self, hash: str | bytes) -> bool:
        value = hash.decode("ascii") if isinstance(hash, bytes) else hash
        return self._parameters != extract_parameters(value)
