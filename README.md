# mojo-argon2-cffi

`mojo-argon2-cffi` is a from-scratch Mojo port of the compute-bound Argon2
key-derivation function behind Python's
[`argon2-cffi`](https://argon2-cffi.readthedocs.io/). It implements BLAKE2b,
Argon2's variable-length hash, block compression, memory indexing, and memory
filling in Mojo. A small Python layer exposes the familiar high- and low-level
APIs through `ctypes`.

This is working cryptographic code with byte-for-byte parity tests, but it has
not received an independent security audit. Use `argon2-cffi` for production
password storage unless you have reviewed the implementation and accept that
risk.

## Coverage

The port covers:

- Argon2d, Argon2i, and Argon2id;
- Argon2 versions 16 and 19;
- arbitrary valid memory, time, parallelism, and output-length parameters;
- `hash_secret_raw`, `hash_secret`, and `verify_secret`;
- `PasswordHasher`, including `hash`, `verify`, `from_parameters`, and
  `check_needs_rehash`;
- `Type`, `Parameters`, `extract_parameters`, the standard profiles, and the
  legacy password helper names.

`parallelism` changes lanes and output exactly as upstream does. Large lane
segments are filled concurrently, while small segments stay serial to avoid
thread-launch overhead. The advanced
`low_level.core(context, type)` CFFI context interface, its secret and
associated-data inputs, custom allocators, and the upstream `ffi` and `lib`
objects are not covered. The package is named `mojo_argon2` so it can coexist
with `argon2-cffi` for parity testing; use `import mojo_argon2 as argon2` when
swapping the covered subset.

## Install

Install the pinned Mojo nightly and Python dependencies, then build the shared
library:

```bash
pixi install
pixi run build
```

The build writes `dist/libmojo-argon2-cffi.so`. All repository workflows run
inside Pixi:

```bash
pixi run test
pixi run bench
```

## Usage

```python
from mojo_argon2 import PasswordHasher

hasher = PasswordHasher(
    time_cost=2,
    memory_cost=19_456,
    parallelism=1,
)
encoded = hasher.hash("correct horse battery staple")

assert hasher.verify(encoded, "correct horse battery staple")
assert not hasher.check_needs_rehash(encoded)
```

The low-level API has the upstream signature:

```python
from mojo_argon2.low_level import Type, hash_secret_raw

key = hash_secret_raw(
    secret=b"password",
    salt=b"0123456789abcdef",
    time_cost=3,
    memory_cost=65_536,
    parallelism=4,
    hash_len=32,
    type=Type.ID,
)
```

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux x86-64. Times are the best of three complete KDF calls and include
allocation and secure memory clearing. A ratio below `1.00x` means Mojo is
slower.

| case | Mojo | argon2-cffi | upstream / Mojo |
|---|---:|---:|---:|
| Argon2id, 64 MiB, t=3, p=4 | 200.79 ms | 94.67 ms | 0.47x |
| Argon2id, 16 MiB, t=2, p=1 | 24.94 ms | 28.14 ms | 1.13x |
| Argon2i, 4 MiB, t=3, p=1 | 8.69 ms | 9.58 ms | 1.10x |

On this run, Mojo is faster for both single-lane cases. The 64 MiB four-lane
case remains slower than upstream's mature, architecture-specific C
implementation and is limited by random-reference traffic beyond the shared
cache.

The four independent BLAMKA permutations in each compression round execute in
four-wide UInt64 SIMD registers. Block XOR, copy, final-lane reduction, and
secure-wipe passes use native SIMD-width loads and stores with scalar tails.
Lane filling fans out across Python worker threads when there is more than one
lane, each slice contains at least 256 blocks, and the whole hash is at least
`MIN_PARALLEL_BLOCKS` (2^19) blocks; below that a single-call kernel is
measured faster than building the pool. Pass/slice rounds still drain in
order, as RFC 9106 requires; only the lanes inside a round run concurrently.
The mandatory memory wipe is a pure bandwidth pass and always runs on the
calling thread. Each active lane gets independent scratch storage; serial calls
reuse one scratch arena.

CPU only: memory-bound.

No GPU path is included, because no kernel in this port clears the GPU
selection gate in [`bench/analysis.md`](bench/analysis.md), which requires
roughly 2 integer flops per byte of matrix traffic:

| kernel | sizes | flops/byte | bound |
|---|---|---:|---|
| `fill_block` (via `fill_segment` / `argon2_fill_lane`) | 4 MiB, 16 MiB, 64 MiB | 0.94 | memory |
| `argon2_finish` (secure memory wipe) | 4 MiB, 16 MiB, 64 MiB | 0.41 | memory |
| `argon2_begin` (BLAKE2b + `hprime`) | any | 12.5, but 8 KiB of input | compute, negligible (0.17 ms at 64 MiB) |

`fill_block` is the hot kernel at 91.8% of the 64 MiB four-lane call and is
memory-bound by design: Argon2's block filler XORs and permutes a 1 KiB block
with two other blocks through data-dependent random references, so it moves
far more bytes than it computes. Copying the matrix to the device, crossing
slice barriers, and copying back would add transfer cost on top of the
bottleneck rather than amortize it. The only kernel above the threshold,
`argon2_begin`, works on 8 KiB and would lose to launch overhead. The CPU path
is therefore the only execution device, and there are no GPU tests to skip on
CPU-only hosts. The shared GPU gate was checked and not reached:
`nvidia-smi --query-gpu=memory.free` reported 6676 MiB free, but the flop gate
fails first.

Baseline numbers for this port, from [`bench/analysis.md`](bench/analysis.md)
(Intel Xeon E5-2697 v4 @ 2.30 GHz, same host as this phase's checks). They are
unchanged because this phase produced no code change:

| case | before | after |
|---|---:|---:|
| Argon2id, 64 MiB, t=3, p=4 | 191.28 ms | 191.28 ms |
| Argon2id, 16 MiB, t=2, p=1 | 23.71 ms | 23.71 ms |
| Argon2i, 4 MiB, t=3, p=1 | 8.73 ms | 8.73 ms |

The README benchmark table above now reports the measured run from this
verification session (200.79 ms / 24.94 ms / 8.69 ms).

## How it works

Python serializes the Argon2 initial-hash input, allocates the memory matrix,
scratch space, and output as contiguous NumPy buffers, then passes their
addresses as 64-bit integers through a C ABI exported by one Mojo compilation
unit. The buffers remain zero-copy across the FFI boundary. Mojo reconstructs
mutable pointers inside the exported function; no Python callback occurs
during the KDF.

The matrix is a flat row-major array of 1024-byte blocks, with each block stored
as 128 little-endian `UInt64` words. Memory cost is rounded to
`4 * parallelism` blocks exactly as specified by Argon2. Caller-owned scratch
holds the compression temporaries, address-generation blocks, BLAKE2b state,
and final lane XOR. Scratch is allocated without a redundant Python-side
initialization because the kernel overwrites it before use. The kernel wipes
both the memory matrix and scratch before returning the tag.

The tests compare raw and encoded output against `argon2-cffi` across every
variant, both versions, multiple lane counts, rounded memory sizes, long
passwords, and tags on both sides of the 64-byte variable-hash boundary.
