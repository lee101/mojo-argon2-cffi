"""ctypes bridge to the compiled Mojo Argon2 implementation."""

from __future__ import annotations

import ctypes
import os
import struct
from concurrent.futures import ThreadPoolExecutor

import numpy as np


WORK_WORDS = 1024
SEGMENT_BLOCKS_THRESHOLD = 256
# Measured on this box: the lane fan-out only repays the pool once the fill
# runs to several hundred thousand blocks.  Below that it is 0.8-0.9x, so the
# default keeps small hashes on the single-call kernel.
MIN_PARALLEL_BLOCKS = 1 << 19
MAX_LANE_WORKERS = 16


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_ARGON2_LIB") or os.path.join(
    ROOT, "dist", "libmojo-argon2-cffi.so"
)

_SIGMA = np.array(
    [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3,
        11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4,
        7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8,
        9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13,
        2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9,
        12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11,
        13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10,
        6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5,
        10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3,
    ],
    dtype=np.uint8,
)

_lib: ctypes.CDLL | None = None

_PHASE_ARITY = {
    "mojo_argon2_hash": 17,
    "mojo_argon2_begin": 7,
    "mojo_argon2_fill": 11,
    "mojo_argon2_finish": 8,
}


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        if not os.path.exists(LIB):
            raise RuntimeError("compiled library missing; run `pixi run build`")
        _lib = ctypes.CDLL(LIB)
        for name, arity in _PHASE_ARITY.items():
            entry = getattr(_lib, name)
            entry.argtypes = [ctypes.c_int64] * arity
            entry.restype = ctypes.c_int64
    return _lib


def _fill_lane(part):
    (
        memory_addr,
        work_addr,
        memory_blocks,
        time_cost,
        parallelism,
        type_id,
        version,
        pass_number,
        slice_number,
        lane,
    ) = part
    rv = lib().mojo_argon2_fill(
        memory_addr,
        work_addr,
        memory_blocks,
        time_cost,
        parallelism,
        type_id,
        version,
        pass_number,
        slice_number,
        lane,
        WORK_WORDS,
    )
    if rv != 0:
        raise RuntimeError(f"Mojo Argon2 lane fill returned error {rv}")


def _check(rv, what):
    if rv != 0:
        raise RuntimeError(f"Mojo Argon2 {what} returned error {rv}")


def hash_raw(
    secret: bytes,
    salt: bytes,
    time_cost: int,
    memory_cost: int,
    parallelism: int,
    hash_len: int,
    type_id: int,
    version: int,
) -> bytes:
    initial = bytearray(40 + len(secret) + len(salt))
    struct.pack_into(
        "<7I",
        initial,
        0,
        parallelism,
        hash_len,
        memory_cost,
        time_cost,
        version,
        type_id,
        len(secret),
    )
    offset = 28
    initial[offset : offset + len(secret)] = secret
    offset += len(secret)
    struct.pack_into("<I", initial, offset, len(salt))
    offset += 4
    initial[offset : offset + len(salt)] = salt
    struct.pack_into("<II", initial, offset + len(salt), 0, 0)
    initial_buf = np.frombuffer(initial, dtype=np.uint8)
    try:
        memory_blocks = 4 * parallelism * (memory_cost // (4 * parallelism))
        memory = np.empty(memory_blocks * 128, dtype=np.uint64)
        segment_blocks = memory_blocks // (parallelism * 4)
        threaded = (
            parallelism > 1
            and segment_blocks >= SEGMENT_BLOCKS_THRESHOLD
            and memory_blocks * time_cost >= MIN_PARALLEL_BLOCKS
        )
        work_words = WORK_WORDS * parallelism if threaded else WORK_WORDS
        work = np.empty(work_words, dtype=np.uint64)
        result = np.empty(hash_len, dtype=np.uint8)
        if not threaded:
            _check(
                lib().mojo_argon2_hash(
                    initial_buf.ctypes.data,
                    initial_buf.size,
                    memory.ctypes.data,
                    memory.size,
                    work.ctypes.data,
                    work.size,
                    result.ctypes.data,
                    result.size,
                    _SIGMA.ctypes.data,
                    _SIGMA.size,
                    time_cost,
                    memory_cost,
                    parallelism,
                    hash_len,
                    type_id,
                    version,
                    0,
                ),
                "kernel",
            )
            return result.tobytes()
        _check(
            lib().mojo_argon2_begin(
                initial_buf.ctypes.data,
                initial_buf.size,
                memory.ctypes.data,
                memory_blocks,
                work.ctypes.data,
                _SIGMA.ctypes.data,
                parallelism,
            ),
            "begin",
        )
        # RFC 9106 fills a pass slice by slice and a pass reads the previous
        # pass, so the rounds must drain in order; only the lanes inside a
        # (pass, slice) round are independent.
        with ThreadPoolExecutor(
            max_workers=min(parallelism, MAX_LANE_WORKERS)
        ) as pool:
            for pass_number in range(time_cost):
                for slice_number in range(4):
                    parts = [
                        (
                            memory.ctypes.data,
                            work.ctypes.data,
                            memory_blocks,
                            time_cost,
                            parallelism,
                            type_id,
                            version,
                            pass_number,
                            slice_number,
                            lane,
                        )
                        for lane in range(parallelism)
                    ]
                    for _ in pool.map(_fill_lane, parts):
                        pass
        _check(
            lib().mojo_argon2_finish(
                memory.ctypes.data,
                memory_blocks,
                work.ctypes.data,
                result.ctypes.data,
                hash_len,
                _SIGMA.ctypes.data,
                parallelism,
                work.size,
            ),
            "finish",
        )
        return result.tobytes()
    finally:
        initial_buf.fill(0)
