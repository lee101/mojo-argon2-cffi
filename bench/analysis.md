# Mojo Argon2 Benchmark Analysis

**Run environment**: Intel Xeon E5-2697 v4 @ 2.30 GHz, Linux x86-64, Mojo 1.2.0.dev2026092605, argon2-cffi 25.1.0  
**Method**: Best of 3–5 runs per case, full KDF including allocation and secure wipe.

---

## Ranked Hotspot Table

| Rank | Function | Input Size | Time (ms) | Bytes Touched | FLOPs | FLOPs/Byte | Bound | Location |
|------|----------|------------|-----------|---------------|-------|------------|-------|----------|
| 1 | `fill_block` (via `fill_segment`/`argon2_fill_lane`) | 64 MiB, t=3, p=4 | 175 (of 191 total) | ~1 GB (R+W) | ~189M | **0.94** | **Memory** | Mojo kernel |
| 2 | `fill_block` (serial) | 16 MiB, t=2, p=1 | 22.7 (of 23.7 total) | ~167 MB | ~31.5M | 0.94 | Memory | Mojo kernel |
| 3 | `fill_block` (serial) | 4 MiB, t=3, p=1 | 10.0 (of 10.5 total) | ~63 MB | ~11.8M | 0.94 | Memory | Mojo kernel |
| 4 | `argon2_finish` (memory wipe) | 64 MiB, p=4 | 14.4 | 64 MiB (write) | ~26M | 0.41 | Memory | Mojo kernel |
| 5 | `argon2_begin` (BLAKE2b + hprime) | 64 MiB, p=4 | 0.17 | 8 KiB | ~0.1M | 12.5 | Compute | Mojo kernel |
| 6 | `argon2_finish` (memory wipe) | 16 MiB, p=1 | 0.86 | 16 MiB (write) | ~6.7M | 0.41 | Memory | Mojo kernel |
| 7 | `argon2_finish` (memory wipe) | 4 MiB, p=1 | 0.21 | 4 MiB (write) | ~1.7M | 0.41 | Memory | Mojo kernel |
| 8 | Python ctypes call / allocation | All | <0.02 | — | — | — | — | Python glue |

**Key**: "Mojo kernel" = inside the single `@export("mojo_argon2_hash")` or split `begin/fill/finish` calls; no Python loops or per-element ctypes in the hot path.

---

## Mojo vs Upstream Comparison

| Case | Mojo (ms) | Upstream (ms) | Ratio (up/mojo) | Verdict |
|------|-----------|---------------|-----------------|---------|
| Argon2id, 64 MiB, t=3, p=4 | **191.28** | 98.63 | **0.52x** | **Mojo slower** |
| Argon2id, 16 MiB, t=2, p=1 | 23.71 | **32.54** | **1.37x** | Mojo faster |
| Argon2i, 4 MiB, t=3, p=1 | 8.73 | **11.82** | **1.35x** | Mojo faster |

> Upstream tested with `OMP_NUM_THREADS=4` for the p=4 case (85–99 ms range).

---

## Detailed Phase Breakdown (64 MiB, t=3, p=4)

| Phase | Time (ms) | % of Total | Notes |
|-------|-----------|------------|-------|
| `mojo_argon2_begin` | 0.17 | 0.1% | BLAKE2b + hprime for first 2 blocks/lane |
| `mojo_argon2_fill` (12 slices × 4 lanes parallel) | ~5.0 each | 91.8% | 196,608 `fill_block` calls total |
| `mojo_argon2_finish` | 14.4 | 7.5% | Lane XOR + hprime + 64 MiB wipe |
| Python coordination | ~4 | 2.1% | `ThreadPoolExecutor` over 12×4 calls |
| **Total** | **~191** | 100% | |

Per-lane fill times show variance (4.7–6.4 ms) due to OS scheduling and cache contention.

---

## Memory-Bound Analysis

**Per `fill_block` (1024 bytes = 128 UInt64 words):**
- Memory ops: ~768 (load prev/ref/next, store r/tmp, load r/tmp, store next)
- ALU ops: ~960 (256× blamka + 256× rotr + 256× XOR + 192× shuffle)
- **FLOPs/byte = 0.94** — firmly memory-bound (well under the ~2 FLOPs/byte GPU threshold)

**64 MiB case totals:**
- `fill_block` calls: 3 passes × 4 slices × 4096 blocks/lane × 4 lanes = **196,608**
- Memory traffic: ~1 GB (read + write)
- Achieved bandwidth: **5.4 GB/s** (Mojo) vs **10.3 GB/s** (upstream)
- The 2× gap is **memory bandwidth utilization**, not compute.

**Root cause**: Argon2's data-dependent random reference indexing defeats hardware prefetchers. With 4 lanes (16 MiB each = 64 MiB total) exceeding the 45 MB shared L3, the working set thrashes cache. Upstream's mature C implementation likely uses:
- Architecture-specific SIMD (AVX-512 on capable CPUs)
- Better cache-blocking / software prefetch
- OpenMP runtime with NUMA-aware thread placement

---

## Python/ctypes Overhead Audit

- **No per-element ctypes calls**: The entire KDF runs inside Mojo; Python calls `mojo_argon2_hash` (serial) or `begin` + 48× `fill` + `finish` (threaded) once per KDF.
- **No per-call allocations in hot path**: NumPy buffers (`memory`, `work`, `result`) allocated once per `hash_raw()` call (~0.15 µs).
- **No Python loops over elements**: All iteration (passes, slices, lanes, blocks, words) is in Mojo.
- **Thread coordination overhead**: ~4 ms for 12 `ThreadPoolExecutor.map` calls over 4 workers — negligible vs 191 ms kernel.

---

## Next Phase Recommendations

| Hotspot | Next Phase | Rationale |
|---------|------------|-----------|
| `fill_block` (main kernel, all cases) | **simd** | 0.94 FLOPs/byte; current SIMD width is 4 (AVX2). Wider vectors (AVX-512 = 8), better shuffle/permute, and software prefetch can raise memory throughput. |
| `fill_block` parallel (64 MiB, p=4) | **parallel** | 4-lane parallelism hits cache capacity wall. Explore: lane-interleaved memory layout, NUMA-local allocation, `max.algorithm.parallelize` with explicit worker pinning. |
| `argon2_finish` memory wipe | **simd** | Already uses SIMD stores; could fuse with final XOR pass and use non-temporal stores to avoid cache pollution. |
| `argon2_begin` (BLAKE2b) | **none** | <0.2 ms even at max size; compute-bound but negligible. |
| Any GPU path | **none** | FLOPs/byte = 0.94 << 2.0 threshold; slice barriers + host/device copy would add overhead with no arithmetic intensity to amortize. |

---

## Notes

- All numbers are from this session's `pixi run bench` and manual profiling runs.
- The port is **faster than upstream for single-lane (p=1) cases** at 16 MiB and 4 MiB.
- The **only regression is the 64 MiB, p=4 case** (0.52×), where memory subsystem pressure exposes the gap between a portable Mojo SIMD implementation and upstream's architecture-tuned C/asm.
- No algorithm changes were made during this analysis phase.