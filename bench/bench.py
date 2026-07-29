"""Benchmark the Mojo KDF against argon2-cffi on identical inputs."""

from __future__ import annotations

import math
import os
import platform
import time

from argon2 import low_level as upstream
from mojo_argon2 import low_level as mojo


CASES = [
    ("Argon2id, 64 MiB, t=3, p=4", 65_536, 3, 4, mojo.Type.ID, upstream.Type.ID),
    ("Argon2id, 16 MiB, t=2, p=1", 16_384, 2, 1, mojo.Type.ID, upstream.Type.ID),
    ("Argon2i, 4 MiB, t=3, p=1", 4_096, 3, 1, mojo.Type.I, upstream.Type.I),
]


def timeit(fn, repeat: int = 3) -> float:
    best = math.inf
    for _ in range(repeat):
        started = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - started)
    return best


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="ascii") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main() -> None:
    secret = b"correct horse battery staple"
    salt = b"0123456789abcdef"
    print(f"Machine: {cpu_name()}; {platform.system()} {platform.machine()}")
    print()
    print("| case | Mojo | argon2-cffi | upstream / Mojo |")
    print("|---|---:|---:|---:|")
    for name, memory, passes, lanes, mojo_type, upstream_type in CASES:
        kwargs = dict(
            secret=secret,
            salt=salt,
            time_cost=passes,
            memory_cost=memory,
            parallelism=lanes,
            hash_len=32,
        )
        ours = lambda: mojo.hash_secret_raw(type=mojo_type, **kwargs)
        theirs = lambda: upstream.hash_secret_raw(type=upstream_type, **kwargs)
        if ours() != theirs():
            raise RuntimeError(f"parity failed before timing {name}")
        mojo_seconds = timeit(ours)
        upstream_seconds = timeit(theirs)
        ratio = upstream_seconds / mojo_seconds
        print(
            f"| {name} | {mojo_seconds * 1000:.2f} ms | "
            f"{upstream_seconds * 1000:.2f} ms | {ratio:.2f}x |"
        )


if __name__ == "__main__":
    main()
