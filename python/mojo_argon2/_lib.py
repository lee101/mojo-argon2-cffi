"""ctypes bridge to the compiled Mojo Argon2 implementation."""

from __future__ import annotations

import ctypes
import os
import struct
import sys

import numpy as np


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
_runtime: ctypes.CDLL | None = None
_runtime_ready = False


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        if not os.path.exists(LIB):
            raise RuntimeError("compiled library missing; run `pixi run build`")
        _lib = ctypes.CDLL(LIB)
        fn = _lib.mojo_argon2_hash
        fn.argtypes = [ctypes.c_int64] * 17
        fn.restype = ctypes.c_int64
    return _lib


def _ensure_parallel_runtime() -> bool:
    global _runtime, _runtime_ready
    if _runtime_ready:
        return True
    try:
        runtime_path = os.path.join(
            sys.prefix, "lib", "libKGENCompilerRTShared.so"
        )
        _runtime = ctypes.CDLL(runtime_path)
        initialize = _runtime.KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice
        initialize.argtypes = []
        initialize.restype = ctypes.c_void_p
        _runtime_ready = bool(initialize())
    except (AttributeError, OSError):
        _runtime = None
        _runtime_ready = False
    return _runtime_ready


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
    initial = bytearray(
        struct.pack(
            "<7I",
            parallelism,
            hash_len,
            memory_cost,
            time_cost,
            version,
            type_id,
            len(secret),
        )
        + secret
        + struct.pack("<I", len(salt))
        + salt
        + struct.pack("<II", 0, 0)
    )
    initial_buf = np.frombuffer(initial, dtype=np.uint8)
    try:
        memory_blocks = 4 * parallelism * (memory_cost // (4 * parallelism))
        memory = np.empty(memory_blocks * 128, dtype=np.uint64)
        segment_blocks = memory_blocks // (parallelism * 4)
        use_threads = (
            parallelism > 1
            and segment_blocks >= 256
            and _ensure_parallel_runtime()
        )
        work_words = 1024 * parallelism if use_threads else 1024
        work = np.empty(work_words, dtype=np.uint64)
        result = np.empty(hash_len, dtype=np.uint8)
        rv = lib().mojo_argon2_hash(
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
            use_threads,
        )
        if rv != 0:
            raise RuntimeError(f"Mojo Argon2 kernel returned error {rv}")
        return result.tobytes()
    finally:
        initial_buf.fill(0)
