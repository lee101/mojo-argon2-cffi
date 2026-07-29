"""Parameter representation and encoded-hash inspection."""

from __future__ import annotations

import base64

from dataclasses import dataclass

from .exceptions import InvalidHashError
from .low_level import Type


@dataclass(slots=True)
class Parameters:
    type: Type
    version: int
    salt_len: int
    hash_len: int
    time_cost: int
    memory_cost: int
    parallelism: int


_NAME_TO_TYPE = {"argon2id": Type.ID, "argon2i": Type.I, "argon2d": Type.D}


def _decoded_len(value: str) -> int:
    try:
        return len(base64.b64decode(value + "=" * (-len(value) % 4), validate=True))
    except Exception as exc:
        raise InvalidHashError from exc


def extract_parameters(hash: str) -> Parameters:
    if not isinstance(hash, str):
        raise TypeError("hash must be str")
    parts = hash.split("$")
    if len(parts) == 5:
        parts.insert(2, "v=18")
    if len(parts) != 6 or parts[0]:
        raise InvalidHashError
    try:
        type_value = _NAME_TO_TYPE[parts[1]]
        fields = [parts[2], *parts[3].split(",")]
        values = dict(field.split("=", 1) for field in fields)
        if set(values) != {"v", "m", "t", "p"}:
            raise ValueError
        return Parameters(
            type=type_value,
            version=int(values["v"]),
            salt_len=_decoded_len(parts[4]),
            hash_len=_decoded_len(parts[5]),
            time_cost=int(values["t"]),
            memory_cost=int(values["m"]),
            parallelism=int(values["p"]),
        )
    except InvalidHashError:
        raise
    except Exception as exc:
        raise InvalidHashError from exc
