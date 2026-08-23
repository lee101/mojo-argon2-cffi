"""Argon2d, Argon2i, and Argon2id with BLAKE2b and the RFC 9106 indexing rule."""

from max.algorithm import parallelize
from std.runtime import initialize_runtime
from std.sys.info import simd_width_of

comptime U8Ptr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime U64Ptr = UnsafePointer[UInt64, AnyOrigin[mut=True]]

comptime R_OFF = 0
comptime TMP_OFF = 128
comptime ADDR_OFF = 256
comptime INPUT_OFF = 384
comptime ZERO_OFF = 512
comptime FINAL_OFF = 640
comptime BLAKE_OFF = 768
comptime SEED_OFF = 832
comptime DIGEST_OFF = 848
comptime WORK_WORDS = 1024
comptime PARALLEL_SEGMENT_BLOCKS = 256
comptime PARALLEL_WIPE_BLOCKS = 32768


def rotr(x: UInt64, n: Int) -> UInt64:
    return (x >> UInt64(n)) | (x << UInt64(64 - n))


def blake_g(
    work: U64Ptr,
    v: Int,
    m: Int,
    a: Int,
    b: Int,
    c: Int,
    d: Int,
    x: Int,
    y: Int,
):
    work[v + a] = work[v + a] + work[v + b] + work[m + x]
    work[v + d] = rotr(work[v + d] ^ work[v + a], 32)
    work[v + c] = work[v + c] + work[v + d]
    work[v + b] = rotr(work[v + b] ^ work[v + c], 24)
    work[v + a] = work[v + a] + work[v + b] + work[m + y]
    work[v + d] = rotr(work[v + d] ^ work[v + a], 16)
    work[v + c] = work[v + c] + work[v + d]
    work[v + b] = rotr(work[v + b] ^ work[v + c], 63)


def blake2b(
    src: U8Ptr,
    src_len: Int,
    dst: U8Ptr,
    dst_len: Int,
    prefix: Bool,
    prefix_value: Int,
    sigma: U8Ptr,
    work: U64Ptr,
):
    var m = BLAKE_OFF
    var v = BLAKE_OFF + 16
    var h = BLAKE_OFF + 32
    work[h] = UInt64(0x6A09E667F3BCC908) ^ UInt64(0x01010000 + dst_len)
    work[h + 1] = UInt64(0xBB67AE8584CAA73B)
    work[h + 2] = UInt64(0x3C6EF372FE94F82B)
    work[h + 3] = UInt64(0xA54FF53A5F1D36F1)
    work[h + 4] = UInt64(0x510E527FADE682D1)
    work[h + 5] = UInt64(0x9B05688C2B3E6C1F)
    work[h + 6] = UInt64(0x1F83D9ABFB41BD6B)
    work[h + 7] = UInt64(0x5BE0CD19137E2179)

    var prefix_len = 4 if prefix else 0
    var total = src_len + prefix_len
    var blocks = (total + 127) // 128
    if blocks == 0:
        blocks = 1
    for block in range(blocks):
        for i in range(16):
            work[m + i] = UInt64(0)
        for j in range(128):
            var pos = block * 128 + j
            if pos < total:
                var byte = UInt8(0)
                if pos < prefix_len:
                    byte = UInt8(
                        (prefix_value >> ((pos & 3) * 8)) & 255
                    )
                else:
                    byte = src[pos - prefix_len]
                var shift = UInt64((j & 7) * 8)
                work[m + (j >> 3)] |= UInt64(byte) << shift

        for i in range(8):
            work[v + i] = work[h + i]
        work[v + 8] = UInt64(0x6A09E667F3BCC908)
        work[v + 9] = UInt64(0xBB67AE8584CAA73B)
        work[v + 10] = UInt64(0x3C6EF372FE94F82B)
        work[v + 11] = UInt64(0xA54FF53A5F1D36F1)
        var count = total
        if (block + 1) * 128 < count:
            count = (block + 1) * 128
        work[v + 12] = UInt64(0x510E527FADE682D1) ^ UInt64(count)
        work[v + 13] = UInt64(0x9B05688C2B3E6C1F)
        work[v + 14] = UInt64(0x1F83D9ABFB41BD6B)
        work[v + 15] = UInt64(0x5BE0CD19137E2179)
        if block == blocks - 1:
            work[v + 14] = ~work[v + 14]

        for r in range(12):
            var s = r * 16
            blake_g(work, v, m, 0, 4, 8, 12, Int(sigma[s]), Int(sigma[s + 1]))
            blake_g(work, v, m, 1, 5, 9, 13, Int(sigma[s + 2]), Int(sigma[s + 3]))
            blake_g(work, v, m, 2, 6, 10, 14, Int(sigma[s + 4]), Int(sigma[s + 5]))
            blake_g(work, v, m, 3, 7, 11, 15, Int(sigma[s + 6]), Int(sigma[s + 7]))
            blake_g(work, v, m, 0, 5, 10, 15, Int(sigma[s + 8]), Int(sigma[s + 9]))
            blake_g(work, v, m, 1, 6, 11, 12, Int(sigma[s + 10]), Int(sigma[s + 11]))
            blake_g(work, v, m, 2, 7, 8, 13, Int(sigma[s + 12]), Int(sigma[s + 13]))
            blake_g(work, v, m, 3, 4, 9, 14, Int(sigma[s + 14]), Int(sigma[s + 15]))
        for i in range(8):
            work[h + i] ^= work[v + i] ^ work[v + i + 8]

    for i in range(dst_len):
        dst[i] = UInt8(
            (work[h + (i >> 3)] >> UInt64((i & 7) * 8)) & UInt64(255)
        )


def copy_bytes(src: U8Ptr, dst: U8Ptr, n: Int):
    comptime W = simd_width_of[DType.float64]()
    var i = 0
    while i + W <= n:
        dst.store(i, src.load[width=W](i))
        i += W
    while i < n:
        dst[i] = src[i]
        i += 1


def hprime(
    src: U8Ptr,
    src_len: Int,
    dst: U8Ptr,
    dst_len: Int,
    digest: U8Ptr,
    sigma: U8Ptr,
    work: U64Ptr,
):
    if dst_len <= 64:
        blake2b(src, src_len, dst, dst_len, True, dst_len, sigma, work)
        return
    blake2b(src, src_len, digest, 64, True, dst_len, sigma, work)
    copy_bytes(digest, dst, 32)
    var produced = 32
    var remaining = dst_len - 32
    while remaining > 64:
        blake2b(digest, 64, digest, 64, False, 0, sigma, work)
        copy_bytes(digest, dst + produced, 32)
        produced += 32
        remaining -= 32
    blake2b(digest, 64, digest, remaining, False, 0, sigma, work)
    copy_bytes(digest, dst + produced, remaining)


def store32(dst: U8Ptr, offset: Int, value: Int):
    for i in range(4):
        dst[offset + i] = UInt8((value >> (i * 8)) & 255)


def blamka(x: UInt64, y: UInt64) -> UInt64:
    var low_x = x & UInt64(0xFFFFFFFF)
    var low_y = y & UInt64(0xFFFFFFFF)
    return x + y + UInt64(2) * low_x * low_y


def blamka4(
    x: SIMD[DType.uint64, 4], y: SIMD[DType.uint64, 4]
) -> SIMD[DType.uint64, 4]:
    var low_x = x & UInt64(0xFFFFFFFF)
    var low_y = y & UInt64(0xFFFFFFFF)
    return x + y + UInt64(2) * low_x * low_y


def rotr4(x: SIMD[DType.uint64, 4], n: Int) -> SIMD[DType.uint64, 4]:
    return (x >> UInt64(n)) | (x << UInt64(64 - n))


def blamka_g(block: U64Ptr, a: Int, b: Int, c: Int, d: Int):
    block[a] = blamka(block[a], block[b])
    block[d] = rotr(block[d] ^ block[a], 32)
    block[c] = blamka(block[c], block[d])
    block[b] = rotr(block[b] ^ block[c], 24)
    block[a] = blamka(block[a], block[b])
    block[d] = rotr(block[d] ^ block[a], 16)
    block[c] = blamka(block[c], block[d])
    block[b] = rotr(block[b] ^ block[c], 63)


def round16(
    block: U64Ptr,
    x0: Int,
    x1: Int,
    x2: Int,
    x3: Int,
    x4: Int,
    x5: Int,
    x6: Int,
    x7: Int,
    x8: Int,
    x9: Int,
    x10: Int,
    x11: Int,
    x12: Int,
    x13: Int,
    x14: Int,
    x15: Int,
):
    comptime W = simd_width_of[DType.float64]()
    comptime if W >= 4:
        var a = SIMD[DType.uint64, 4](
            block[x0], block[x1], block[x2], block[x3]
        )
        var b = SIMD[DType.uint64, 4](
            block[x4], block[x5], block[x6], block[x7]
        )
        var c = SIMD[DType.uint64, 4](
            block[x8], block[x9], block[x10], block[x11]
        )
        var d = SIMD[DType.uint64, 4](
            block[x12], block[x13], block[x14], block[x15]
        )
        a = blamka4(a, b)
        d = rotr4(d ^ a, 32)
        c = blamka4(c, d)
        b = rotr4(b ^ c, 24)
        a = blamka4(a, b)
        d = rotr4(d ^ a, 16)
        c = blamka4(c, d)
        b = rotr4(b ^ c, 63)
        b = b.shuffle[1, 2, 3, 0]()
        c = c.shuffle[2, 3, 0, 1]()
        d = d.shuffle[3, 0, 1, 2]()
        a = blamka4(a, b)
        d = rotr4(d ^ a, 32)
        c = blamka4(c, d)
        b = rotr4(b ^ c, 24)
        a = blamka4(a, b)
        d = rotr4(d ^ a, 16)
        c = blamka4(c, d)
        b = rotr4(b ^ c, 63)
        b = b.shuffle[3, 0, 1, 2]()
        c = c.shuffle[2, 3, 0, 1]()
        d = d.shuffle[1, 2, 3, 0]()
        block[x0], block[x1], block[x2], block[x3] = (
            a[0], a[1], a[2], a[3]
        )
        block[x4], block[x5], block[x6], block[x7] = (
            b[0], b[1], b[2], b[3]
        )
        block[x8], block[x9], block[x10], block[x11] = (
            c[0], c[1], c[2], c[3]
        )
        block[x12], block[x13], block[x14], block[x15] = (
            d[0], d[1], d[2], d[3]
        )
    else:
        blamka_g(block, x0, x4, x8, x12)
        blamka_g(block, x1, x5, x9, x13)
        blamka_g(block, x2, x6, x10, x14)
        blamka_g(block, x3, x7, x11, x15)
        blamka_g(block, x0, x5, x10, x15)
        blamka_g(block, x1, x6, x11, x12)
        blamka_g(block, x2, x7, x8, x13)
        blamka_g(block, x3, x4, x9, x14)


def fill_block(
    prev: U64Ptr,
    reference: U64Ptr,
    next: U64Ptr,
    scratch: U64Ptr,
    with_xor: Bool,
):
    var r = scratch + R_OFF
    var tmp = scratch + TMP_OFF
    comptime W = simd_width_of[DType.float64]()
    var word = 0
    if with_xor:
        while word + W <= 128:
            var mixed = (
                prev.load[width=W](word)
                ^ reference.load[width=W](word)
            )
            r.store(word, mixed)
            tmp.store(word, mixed ^ next.load[width=W](word))
            word += W
    else:
        while word + W <= 128:
            var mixed = (
                prev.load[width=W](word)
                ^ reference.load[width=W](word)
            )
            r.store(word, mixed)
            tmp.store(word, mixed)
            word += W
    while word < 128:
        r[word] = prev[word] ^ reference[word]
        tmp[word] = r[word]
        if with_xor:
            tmp[word] ^= next[word]
        word += 1
    for i in range(8):
        var j = i * 16
        round16(
            r,
            j, j + 1, j + 2, j + 3, j + 4, j + 5, j + 6, j + 7,
            j + 8, j + 9, j + 10, j + 11, j + 12, j + 13, j + 14, j + 15,
        )
    for i in range(8):
        var j = i * 2
        round16(
            r,
            j, j + 1, j + 16, j + 17, j + 32, j + 33, j + 48, j + 49,
            j + 64, j + 65, j + 80, j + 81, j + 96, j + 97, j + 112, j + 113,
        )
    word = 0
    while word + W <= 128:
        next.store(
            word,
            tmp.load[width=W](word) ^ r.load[width=W](word),
        )
        word += W
    while word < 128:
        next[word] = tmp[word] ^ r[word]
        word += 1


def next_addresses(scratch: U64Ptr):
    var addr = scratch + ADDR_OFF
    var input_block = scratch + INPUT_OFF
    var zero = scratch + ZERO_OFF
    input_block[6] += UInt64(1)
    fill_block(zero, input_block, addr, scratch, False)
    fill_block(zero, addr, addr, scratch, False)


def reference_index(
    pass_number: Int,
    slice_number: Int,
    index: Int,
    segment_length: Int,
    lane_length: Int,
    same_lane: Bool,
    j1: UInt64,
) -> Int:
    var area: Int
    if pass_number == 0:
        if slice_number == 0:
            area = index - 1
        elif same_lane:
            area = slice_number * segment_length + index - 1
        else:
            area = slice_number * segment_length
            if index == 0:
                area -= 1
    elif same_lane:
        area = lane_length - segment_length + index - 1
    else:
        area = lane_length - segment_length
        if index == 0:
            area -= 1
    var relative = j1 * j1
    relative >>= UInt64(32)
    relative = UInt64(area - 1) - (
        (UInt64(area) * relative) >> UInt64(32)
    )
    var start = 0
    if pass_number != 0 and slice_number != 3:
        start = (slice_number + 1) * segment_length
    return (start + Int(relative)) % lane_length


def fill_segment(
    memory: U64Ptr,
    scratch: U64Ptr,
    memory_blocks: Int,
    passes: Int,
    lanes: Int,
    type_id: Int,
    version: Int,
    pass_number: Int,
    slice_number: Int,
    lane: Int,
):
    var segment_length = memory_blocks // (lanes * 4)
    var lane_length = segment_length * 4
    var independent = type_id == 1 or (
        type_id == 2 and pass_number == 0 and slice_number < 2
    )
    if independent:
        comptime W = simd_width_of[DType.float64]()
        var zeroes = SIMD[DType.uint64, W](0)
        var i = 0
        while i + W <= 128:
            scratch.store(ADDR_OFF + i, zeroes)
            scratch.store(INPUT_OFF + i, zeroes)
            scratch.store(ZERO_OFF + i, zeroes)
            i += W
        while i < 128:
            scratch[ADDR_OFF + i] = UInt64(0)
            scratch[INPUT_OFF + i] = UInt64(0)
            scratch[ZERO_OFF + i] = UInt64(0)
            i += 1
        scratch[INPUT_OFF] = UInt64(pass_number)
        scratch[INPUT_OFF + 1] = UInt64(lane)
        scratch[INPUT_OFF + 2] = UInt64(slice_number)
        scratch[INPUT_OFF + 3] = UInt64(memory_blocks)
        scratch[INPUT_OFF + 4] = UInt64(passes)
        scratch[INPUT_OFF + 5] = UInt64(type_id)

    var start = 0
    if pass_number == 0 and slice_number == 0:
        start = 2
        if independent:
            next_addresses(scratch)

    for index in range(start, segment_length):
        var curr_rel = slice_number * segment_length + index
        var prev_rel = curr_rel - 1
        if curr_rel == 0:
            prev_rel = lane_length - 1
        var prev_block = lane * lane_length + prev_rel
        var pseudo = UInt64(0)
        if independent:
            if index % 128 == 0:
                next_addresses(scratch)
            pseudo = scratch[ADDR_OFF + index % 128]
        else:
            pseudo = memory[prev_block * 128]
        var ref_lane = Int(
            (pseudo >> UInt64(32)) % UInt64(lanes)
        )
        if pass_number == 0 and slice_number == 0:
            ref_lane = lane
        var ref_index = reference_index(
            pass_number,
            slice_number,
            index,
            segment_length,
            lane_length,
            ref_lane == lane,
            pseudo & UInt64(0xFFFFFFFF),
        )
        var curr_block = lane * lane_length + curr_rel
        fill_block(
            memory + prev_block * 128,
            memory + (ref_lane * lane_length + ref_index) * 128,
            memory + curr_block * 128,
            scratch,
            version != 16 and pass_number != 0,
        )


def argon2_hash(
    initial: U8Ptr,
    initial_len: Int,
    memory: U64Ptr,
    memory_bytes: U8Ptr,
    scratch: U64Ptr,
    scratch_bytes: U8Ptr,
    dst: U8Ptr,
    sigma: U8Ptr,
    time_cost: Int,
    memory_cost: Int,
    parallelism: Int,
    hash_len: Int,
    type_id: Int,
    version: Int,
    use_threads: Bool,
):
    if use_threads:
        initialize_runtime()
    var memory_blocks = 4 * parallelism * (
        memory_cost // (4 * parallelism)
    )
    var lane_length = memory_blocks // parallelism
    var seed = scratch_bytes + SEED_OFF * 8
    var digest = scratch_bytes + DIGEST_OFF * 8
    blake2b(initial, initial_len, seed, 64, False, 0, sigma, scratch)
    for lane in range(parallelism):
        store32(seed, 64, 0)
        store32(seed, 68, lane)
        hprime(
            seed, 72, memory_bytes + (lane * lane_length) * 1024,
            1024, digest, sigma, scratch,
        )
        store32(seed, 64, 1)
        hprime(
            seed, 72, memory_bytes + (lane * lane_length + 1) * 1024,
            1024, digest, sigma, scratch,
        )

    for pass_number in range(time_cost):
        for slice_number in range(4):
            if (
                use_threads
                and
                parallelism > 1
                and memory_blocks // (parallelism * 4)
                >= PARALLEL_SEGMENT_BLOCKS
            ):
                @__parameter
                @__copy_capture(
                    memory,
                    scratch,
                    memory_blocks,
                    time_cost,
                    parallelism,
                    type_id,
                    version,
                    pass_number,
                    slice_number,
                )
                def fill_lane(lane: Int):
                    fill_segment(
                        memory,
                        scratch + lane * WORK_WORDS,
                        memory_blocks,
                        time_cost,
                        parallelism,
                        type_id,
                        version,
                        pass_number,
                        slice_number,
                        lane,
                    )

                parallelize[fill_lane](parallelism, parallelism)
            else:
                for lane in range(parallelism):
                    fill_segment(
                        memory,
                        scratch,
                        memory_blocks,
                        time_cost,
                        parallelism,
                        type_id,
                        version,
                        pass_number,
                        slice_number,
                        lane,
                    )

    var final_block = scratch + FINAL_OFF
    var last = lane_length - 1
    comptime W = simd_width_of[DType.float64]()
    var word = 0
    while word + W <= 128:
        final_block.store(
            word,
            memory.load[width=W](last * 128 + word),
        )
        word += W
    while word < 128:
        final_block[word] = memory[last * 128 + word]
        word += 1
    for lane in range(1, parallelism):
        var block = (lane * lane_length + last) * 128
        word = 0
        while word + W <= 128:
            final_block.store(
                word,
                final_block.load[width=W](word)
                ^ memory.load[width=W](block + word),
            )
            word += W
        while word < 128:
            final_block[word] ^= memory[block + word]
            word += 1
    hprime(
        scratch_bytes + FINAL_OFF * 8, 1024, dst, hash_len,
        digest, sigma, scratch,
    )
    var zeroes = SIMD[DType.uint64, W](0)
    var memory_words = memory_blocks * 128
    if use_threads and memory_blocks >= PARALLEL_WIPE_BLOCKS:
        @__parameter
        @__copy_capture(memory, memory_words, parallelism)
        def wipe_lane(lane: Int):
            comptime WIPE_W = simd_width_of[DType.float64]()
            var lane_words = memory_words // parallelism
            var start = lane * lane_words
            var stop = start + lane_words
            var wipe_zeroes = SIMD[DType.uint64, WIPE_W](0)
            while start + WIPE_W <= stop:
                memory.store(start, wipe_zeroes)
                start += WIPE_W
            while start < stop:
                memory[start] = UInt64(0)
                start += 1

        parallelize[wipe_lane](parallelism, parallelism)
    else:
        word = 0
        while word + W <= memory_words:
            memory.store(word, zeroes)
            word += W
        while word < memory_words:
            memory[word] = UInt64(0)
            word += 1
    word = 0
    var work_words = WORK_WORDS
    if use_threads:
        work_words *= parallelism
    while word + W <= work_words:
        scratch.store(word, zeroes)
        word += W
    while word < work_words:
        scratch[word] = UInt64(0)
        word += 1


@export("mojo_argon2_hash")
def mojo_argon2_hash(
    initial_addr: Int,
    initial_len: Int,
    memory_addr: Int,
    memory_words: Int,
    work_addr: Int,
    work_words: Int,
    dst_addr: Int,
    dst_len: Int,
    sigma_addr: Int,
    sigma_len: Int,
    time_cost: Int,
    memory_cost: Int,
    parallelism: Int,
    hash_len: Int,
    type_id: Int,
    version: Int,
    use_threads: Int,
) abi("C") -> Int:
    if (
        initial_addr == 0
        or initial_len < 48
        or memory_addr == 0
        or work_addr == 0
        or dst_addr == 0
        or sigma_addr == 0
        or time_cost < 1
        or parallelism < 1
        or parallelism > 0xFFFFFF
        or memory_cost < 8 * parallelism
        or hash_len < 4
        or time_cost > 0xFFFFFFFF
        or memory_cost > 0xFFFFFFFF
        or hash_len > 0xFFFFFFFF
        or type_id < 0
        or type_id > 2
        or (version != 16 and version != 19)
        or (use_threads != 0 and use_threads != 1)
    ):
        return -1
    var memory_blocks = 4 * parallelism * (
        memory_cost // (4 * parallelism)
    )
    var required_work_words = WORK_WORDS
    if use_threads != 0:
        required_work_words *= parallelism
    if (
        memory_words < memory_blocks * 128
        or work_words < required_work_words
        or dst_len < hash_len
        or sigma_len < 192
    ):
        return -1
    var initial = U8Ptr(unsafe_from_address=initial_addr)
    var memory = U64Ptr(unsafe_from_address=memory_addr)
    var memory_bytes = U8Ptr(unsafe_from_address=memory_addr)
    var scratch = U64Ptr(unsafe_from_address=work_addr)
    var scratch_bytes = U8Ptr(unsafe_from_address=work_addr)
    var dst = U8Ptr(unsafe_from_address=dst_addr)
    var sigma = U8Ptr(unsafe_from_address=sigma_addr)
    argon2_hash(
        initial,
        initial_len,
        memory,
        memory_bytes,
        scratch,
        scratch_bytes,
        dst,
        sigma,
        time_cost,
        memory_cost,
        parallelism,
        hash_len,
        type_id,
        version,
        use_threads != 0,
    )
    return 0
