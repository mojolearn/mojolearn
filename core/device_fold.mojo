# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Device folds that replace "download, then loop over n on the host".

Lane cgr4-download-loop (2026-10-03). Every helper here reads a device
buffer once, in parallel, and leaves at most `SCAN_BLOCKS` (512) partials
for the host to combine: a fixed, size-capped fold, never a pass over n.
The launch shape is `core/device_scan.mojo`'s (256 threads, one block per
256 elements, at most 512 blocks, grid-stride above that).

INTEGER FOLDS (`device_sum_i32`, `device_count_nonzero_i32`,
`device_count_equal_i32`, `device_first_nonneg_i32`) are exact and
order-free: they return the value the host loop they replace returned, on
every vendor, so they change no bits.

THE FLOAT SUM (`device_sum_f32_fixed`) has ONE fold order, a pure function
of n, the same on NVIDIA, AMD, Apple and the host column:

  1. blocks = min(ceil(n / 256), 512); stride = blocks * 256.
  2. thread t of block b adds, in ascending k, the elements
     `b * 256 + t + k * stride` into a register that starts at 0.0.
  3. the block folds its 256 registers with the halving tree
     `red[t] += red[t + s]`, s = 128, 64, ..., 1.
  4. the host adds the block partials in ascending block order onto 0.0.

`host_sum_f32_fixed` is that order on the host, for the CPU-only column.
Every add is flushed (`checks.numerics.ftz`), so a subnormal is zero on
every vendor; no multiply appears, so no fused multiply-add can change a bit.
"""

from std.gpu import block_idx, grid_dim, thread_idx
from std.memory import bitcast, stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import ftz
from core.device_scan import SCAN_TPB, SCAN_BLOCKS, NONFINITE_NONE
from core.fast_radix_sort import fast_radix_sort_pairs_u32, frs_counts_len, frs_exclusive_scan, frs_scan_blocks

comptime _I32P = MutPointer[Int32, MutAnyOrigin]
comptime _I64P = MutPointer[Int64, MutAnyOrigin]
comptime _F32P = MutPointer[Float32, MutAnyOrigin]
comptime _U32P = MutPointer[UInt32, MutAnyOrigin]


def fold_blocks(n: Int) -> Int:
    """Blocks for `n` elements: one per `SCAN_TPB`, capped at `SCAN_BLOCKS`."""
    var blocks = (n + SCAN_TPB - 1) // SCAN_TPB
    if blocks > SCAN_BLOCKS:
        blocks = SCAN_BLOCKS
    if blocks < 1:
        blocks = 1
    return blocks


# ------------------------------------------------------------- kernels ----


def _sum_i32_kernel(part: _I64P, buf: _I32P, n_in: Int32, mode: Int32, ref_v: Int32):
    """One Int64 partial per block. mode 0: the sum of the values; mode 1:
    the count of nonzero values; mode 2: the count of values equal to
    `ref_v`. Integer, so exact in any order."""
    var n = Int(n_in)
    var red = stack_allocation[SCAN_TPB, Scalar[DType.int64], address_space = AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var stride = Int(grid_dim.x) * SCAN_TPB
    var i = Int(block_idx.x) * SCAN_TPB + tid
    var acc = Int64(0)
    while i < n:
        var v = buf.unsafe_load(i)
        if mode == 0:
            acc += Int64(v)
        elif mode == 1:
            if v != Int32(0):
                acc += 1
        else:
            if v == ref_v:
                acc += 1
        i += stride
    red.unsafe_store(tid, acc)
    barrier()
    var active = SCAN_TPB // 2
    while active > 0:
        if tid < active:
            red.unsafe_store(tid, red.unsafe_load(tid) + red.unsafe_load(tid + active))
        barrier()
        active = active // 2
    if tid == 0:
        part.unsafe_store(Int(block_idx.x), red.unsafe_load(0))


def _first_nonneg_i32_kernel(part: _I32P, buf: _I32P, n_in: Int32):
    """One partial per block: the smallest index with `buf[i] >= 0`, or
    `NONFINITE_NONE`. An integer minimum, so the first index everywhere."""
    var n = Int(n_in)
    var red = stack_allocation[SCAN_TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var stride = Int(grid_dim.x) * SCAN_TPB
    var i = Int(block_idx.x) * SCAN_TPB + tid
    var best = NONFINITE_NONE
    while i < n:
        if buf.unsafe_load(i) >= Int32(0):
            best = Int32(i)
            break
        i += stride
    red.unsafe_store(tid, best)
    barrier()
    var active = SCAN_TPB // 2
    while active > 0:
        if tid < active:
            var o = red.unsafe_load(tid + active)
            if o < red.unsafe_load(tid):
                red.unsafe_store(tid, o)
        barrier()
        active = active // 2
    if tid == 0:
        part.unsafe_store(Int(block_idx.x), red.unsafe_load(0))


def _sum_f32_kernel(part: _F32P, buf: _F32P, n_in: Int32):
    """Steps 2 and 3 of the module docstring's fold."""
    var n = Int(n_in)
    var red = stack_allocation[SCAN_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var stride = Int(grid_dim.x) * SCAN_TPB
    var i = Int(block_idx.x) * SCAN_TPB + tid
    var acc = Float32(0.0)
    while i < n:
        acc = ftz(acc + buf.unsafe_load(i))
        i += stride
    red.unsafe_store(tid, acc)
    barrier()
    var active = SCAN_TPB // 2
    while active > 0:
        if tid < active:
            red.unsafe_store(tid, ftz(red.unsafe_load(tid) + red.unsafe_load(tid + active)))
        barrier()
        active = active // 2
    if tid == 0:
        part.unsafe_store(Int(block_idx.x), red.unsafe_load(0))


def _mark_kernel(flags: _I32P, keys: _I32P, n_in: Int32, mask: Int32, m_in: Int32):
    """flags[keys[i] & mask] = 1 for every i (keys outside [0, m) skipped).
    Every writer stores the same 1, so the race is benign and exact."""
    var i = Int(block_idx.x) * SCAN_TPB + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    var key = Int(keys.unsafe_load(i) & mask)
    if key >= 0 and key < Int(m_in):
        flags.unsafe_store(key, Int32(1))


def _zero_i32_kernel(buf: _I32P, n_in: Int32):
    var i = Int(block_idx.x) * SCAN_TPB + Int(thread_idx.x)
    if i < Int(n_in):
        buf.unsafe_store(i, Int32(0))


def _eq_flag_kernel(src: _I32P, n_in: Int32, value: Int32, flag: _I32P, scan: _I32P):
    var i = Int(block_idx.x) * SCAN_TPB + Int(thread_idx.x)
    if i < Int(n_in):
        var f = Int32(1) if src.unsafe_load(i) == value else Int32(0)
        flag.unsafe_store(i, f)
        scan.unsafe_store(i, f)


def _emit_rows_kernel(flag: _I32P, scan: _I32P, n_in: Int32, out: _I32P):
    var i = Int(block_idx.x) * SCAN_TPB + Int(thread_idx.x)
    if i < Int(n_in) and flag.unsafe_load(i) != Int32(0):
        out.unsafe_store(Int(scan.unsafe_load(i)), Int32(i))


def _key_i32_kernel(src: _I32P, n_in: Int32, keys: _U32P, vals: _U32P):
    """The order-preserving u32 key of an int32 (sign bit flipped)."""
    var i = Int(block_idx.x) * SCAN_TPB + Int(thread_idx.x)
    if i < Int(n_in):
        keys.unsafe_store(i, bitcast[DType.uint32](src.unsafe_load(i)) ^ UInt32(0x80000000))
        vals.unsafe_store(i, UInt32(i))


def _boundary_kernel(keys: _U32P, n_in: Int32, flag: _I32P, scan: _I32P):
    var i = Int(block_idx.x) * SCAN_TPB + Int(thread_idx.x)
    if i < Int(n_in):
        var f = Int32(1)
        if i > 0 and keys.unsafe_load(i) == keys.unsafe_load(i - 1):
            f = Int32(0)
        flag.unsafe_store(i, f)
        scan.unsafe_store(i, f)


def _emit_keys_kernel(keys: _U32P, flag: _I32P, scan: _I32P, n_in: Int32, out: _I32P):
    var i = Int(block_idx.x) * SCAN_TPB + Int(thread_idx.x)
    if i < Int(n_in) and flag.unsafe_load(i) != Int32(0):
        out.unsafe_store(
            Int(scan.unsafe_load(i)), bitcast[DType.int32](keys.unsafe_load(i) ^ UInt32(0x80000000))
        )


def _scan_input_kernel(src: _I32P, dst: _I32P, n_in: Int32):
    """dst[i] = src[i] for i < n and dst[n] = 0 (src may be dst)."""
    var i = Int(block_idx.x) * SCAN_TPB + Int(thread_idx.x)
    var n = Int(n_in)
    if i < n:
        dst.unsafe_store(i, src.unsafe_load(i))
    elif i == n:
        dst.unsafe_store(n, Int32(0))


comptime COLMEAN_BLOCKS = 256
"""`device_column_means32`'s grid cap."""


def _colsum32_partial_kernel(x: _F32P, part: _F32P, n_in: Int32, k_in: Int32):
    """Block b, thread t = 32 * p + f: column f summed over rows
    `r = (b + G * j) * 8 + p` (G = grid size), ascending j; the block's 8
    row-phases folded ascending into `part[b * 32 + f]`."""
    var n = Int(n_in)
    var k = Int(k_in)
    var tid = Int(thread_idx.x)
    var f = tid % 32
    var p = tid // 32
    var g = Int(grid_dim.x)
    var sh = stack_allocation[256, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var s = Float32(0.0)
    if f < k:
        var r = Int(block_idx.x) * 8 + p
        while r < n:
            s = s + x.unsafe_load(r * k + f)
            r += g * 8
    sh.unsafe_store(tid, s)
    barrier()
    if tid < 32:
        var t = Float32(0.0)
        for q in range(8):
            t = t + sh.unsafe_load(q * 32 + tid)
        part.unsafe_store(Int(block_idx.x) * 32 + tid, t)


def _colmean32_fold_kernel(part: _F32P, mean: _F32P, blocks_in: Int32, n_in: Int32, k_in: Int32):
    """Thread f < k: the block partials of column f added ascending, over n."""
    var f = Int(thread_idx.x)
    if f < Int(k_in):
        var t = Float32(0.0)
        for b in range(Int(blocks_in)):
            t = t + part.unsafe_load(b * 32 + f)
        mean.unsafe_store(f, t / Float32(Int(n_in)))


# ------------------------------------------------------------ host side ----


def _fold_i64(ctx: DeviceContext, mut part: DeviceBuffer[DType.int64], blocks: Int) raises -> Int64:
    var host = ctx.enqueue_create_host_buffer[DType.int64](blocks)
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=part)
    # the host reads `host` next: the one wait this fold needs
    ctx.synchronize()
    var total = Int64(0)
    for b in range(blocks):
        total += host.unsafe_ptr().unsafe_load(b)
    _ = host^
    return total


def _i32_fold(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.int32], n: Int, mode: Int32, ref_v: Int32
) raises -> Int64:
    if n <= 0:
        return Int64(0)
    var blocks = fold_blocks(n)
    var part = ctx.enqueue_create_buffer[DType.int64](blocks)
    ctx.enqueue_function[_sum_i32_kernel](
        part.unsafe_ptr(), buf.unsafe_ptr(), Int32(n), mode, ref_v,
        grid_dim=(blocks, 1, 1), block_dim=(SCAN_TPB, 1, 1),
    )
    var total = _fold_i64(ctx, part, blocks)
    _ = part^
    return total


def device_sum_i32(ctx: DeviceContext, mut buf: DeviceBuffer[DType.int32], n: Int) raises -> Int64:
    """The exact Int64 sum of `buf[0:n]`."""
    return _i32_fold(ctx, buf, n, Int32(0), Int32(0))


def device_count_nonzero_i32(ctx: DeviceContext, mut buf: DeviceBuffer[DType.int32], n: Int) raises -> Int:
    """How many of `buf[0:n]` are nonzero."""
    return Int(_i32_fold(ctx, buf, n, Int32(1), Int32(0)))


def device_count_equal_i32(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.int32], n: Int, value: Int32
) raises -> Int:
    """How many of `buf[0:n]` equal `value`."""
    return Int(_i32_fold(ctx, buf, n, Int32(2), value))


def device_first_nonneg_i32(ctx: DeviceContext, mut buf: DeviceBuffer[DType.int32], n: Int) raises -> Int:
    """The first index `i < n` with `buf[i] >= 0`, or -1."""
    if n <= 0:
        return -1
    var blocks = fold_blocks(n)
    var part = ctx.enqueue_create_buffer[DType.int32](blocks)
    ctx.enqueue_function[_first_nonneg_i32_kernel](
        part.unsafe_ptr(), buf.unsafe_ptr(), Int32(n),
        grid_dim=(blocks, 1, 1), block_dim=(SCAN_TPB, 1, 1),
    )
    var host = ctx.enqueue_create_host_buffer[DType.int32](blocks)
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=part)
    ctx.synchronize()
    var best = NONFINITE_NONE
    for b in range(blocks):
        var v = host.unsafe_ptr().unsafe_load(b)
        if v < best:
            best = v
    _ = host^
    _ = part^
    return -1 if best == NONFINITE_NONE else Int(best)


def device_count_distinct_keys(
    ctx: DeviceContext, mut keys: DeviceBuffer[DType.int32], n: Int, m: Int, mask: Int32
) raises -> Int:
    """How many distinct values `keys[i] & mask` take over `i < n`, for keys
    known to lie in [0, m): one flag per key value, set in parallel, then
    counted. Exact."""
    if n <= 0 or m <= 0:
        return 0
    var flags = ctx.enqueue_create_buffer[DType.int32](m)
    ctx.enqueue_function[_zero_i32_kernel](
        flags.unsafe_ptr(), Int32(m),
        grid_dim=((m + SCAN_TPB - 1) // SCAN_TPB, 1, 1), block_dim=(SCAN_TPB, 1, 1),
    )
    ctx.enqueue_function[_mark_kernel](
        flags.unsafe_ptr(), keys.unsafe_ptr(), Int32(n), mask, Int32(m),
        grid_dim=((n + SCAN_TPB - 1) // SCAN_TPB, 1, 1), block_dim=(SCAN_TPB, 1, 1),
    )
    var c = device_count_nonzero_i32(ctx, flags, m)
    _ = flags^
    return c


def device_sum_f32_fixed(ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int) raises -> Float32:
    """The module docstring's fixed-order Float32 sum of `buf[0:n]`."""
    if n <= 0:
        return Float32(0.0)
    var blocks = fold_blocks(n)
    var part = ctx.enqueue_create_buffer[DType.float32](blocks)
    ctx.enqueue_function[_sum_f32_kernel](
        part.unsafe_ptr(), buf.unsafe_ptr(), Int32(n),
        grid_dim=(blocks, 1, 1), block_dim=(SCAN_TPB, 1, 1),
    )
    var host = ctx.enqueue_create_host_buffer[DType.float32](blocks)
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=part)
    ctx.synchronize()
    var total = Float32(0.0)
    for b in range(blocks):
        total = ftz(total + host.unsafe_ptr().unsafe_load(b))
    _ = host^
    _ = part^
    return total


def host_sum_f32_fixed(x: List[Float32], n: Int) -> Float32:
    """`device_sum_f32_fixed`'s fold order on the host (the CPU-only column)."""
    if n <= 0:
        return Float32(0.0)
    var blocks = fold_blocks(n)
    var stride = blocks * SCAN_TPB
    var red = List[Float32](length=SCAN_TPB, fill=Float32(0.0))
    var total = Float32(0.0)
    for b in range(blocks):
        for t in range(SCAN_TPB):
            var acc = Float32(0.0)
            var i = b * SCAN_TPB + t
            while i < n:
                acc = ftz(acc + x[i])
                i += stride
            red[t] = acc
        var active = SCAN_TPB // 2
        while active > 0:
            for t in range(active):
                red[t] = ftz(red[t] + red[t + active])
            active = active // 2
        total = ftz(total + red[0])
    return total


def _grid(n: Int) -> Int:
    return (n + SCAN_TPB - 1) // SCAN_TPB


def device_compact_equal_i32(
    ctx: DeviceContext,
    mut src: DeviceBuffer[DType.int32],
    n: Int,
    value: Int32,
    mut out_rows: DeviceBuffer[DType.int32],
) raises -> Int:
    """Writes, ascending, every index `i < n` with `src[i] == value` into
    `out_rows` (which holds at least n slots) and returns how many: a
    stream compaction by an exclusive scan, so the order is the host
    loop's. Exact."""
    if n <= 0:
        return 0
    var flag = ctx.enqueue_create_buffer[DType.int32](n)
    var scan = ctx.enqueue_create_buffer[DType.int32](n)
    var bsum = ctx.enqueue_create_buffer[DType.int32](frs_scan_blocks(n))
    ctx.enqueue_function[_eq_flag_kernel](
        src.unsafe_ptr(), Int32(n), value, flag.unsafe_ptr(), scan.unsafe_ptr(),
        grid_dim=(_grid(n), 1, 1), block_dim=(SCAN_TPB, 1, 1),
    )
    frs_exclusive_scan(ctx, scan, n, bsum)
    ctx.enqueue_function[_emit_rows_kernel](
        flag.unsafe_ptr(), scan.unsafe_ptr(), Int32(n), out_rows.unsafe_ptr(),
        grid_dim=(_grid(n), 1, 1), block_dim=(SCAN_TPB, 1, 1),
    )
    var c = device_count_nonzero_i32(ctx, flag, n)
    _ = bsum^
    _ = scan^
    _ = flag^
    return c


def device_sorted_unique_i32(
    ctx: DeviceContext, mut src: DeviceBuffer[DType.int32], n: Int
) raises -> List[Int32]:
    """The sorted distinct values of `src[0:n]`: a stable device radix sort
    of the order-preserving keys, a boundary flag, an exclusive scan and an
    emit; only the distinct values (the answer) are downloaded. Exact."""
    var out = List[Int32]()
    if n <= 0:
        return out^
    var keys = ctx.enqueue_create_buffer[DType.uint32](n)
    var vals = ctx.enqueue_create_buffer[DType.uint32](n)
    var tk = ctx.enqueue_create_buffer[DType.uint32](n)
    var tv = ctx.enqueue_create_buffer[DType.uint32](n)
    var cnt = ctx.enqueue_create_buffer[DType.int32](frs_counts_len(n))
    var flag = ctx.enqueue_create_buffer[DType.int32](n)
    var scan = ctx.enqueue_create_buffer[DType.int32](n)
    var bsum = ctx.enqueue_create_buffer[DType.int32](frs_scan_blocks(n))
    var uniq = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_function[_key_i32_kernel](
        src.unsafe_ptr(), Int32(n), keys.unsafe_ptr(), vals.unsafe_ptr(),
        grid_dim=(_grid(n), 1, 1), block_dim=(SCAN_TPB, 1, 1),
    )
    fast_radix_sort_pairs_u32(ctx, n, keys, vals, tk, tv, cnt)
    ctx.enqueue_function[_boundary_kernel](
        keys.unsafe_ptr(), Int32(n), flag.unsafe_ptr(), scan.unsafe_ptr(),
        grid_dim=(_grid(n), 1, 1), block_dim=(SCAN_TPB, 1, 1),
    )
    frs_exclusive_scan(ctx, scan, n, bsum)
    ctx.enqueue_function[_emit_keys_kernel](
        keys.unsafe_ptr(), flag.unsafe_ptr(), scan.unsafe_ptr(), Int32(n), uniq.unsafe_ptr(),
        grid_dim=(_grid(n), 1, 1), block_dim=(SCAN_TPB, 1, 1),
    )
    var k = device_count_nonzero_i32(ctx, flag, n)
    var host = ctx.enqueue_create_host_buffer[DType.int32](k)
    var head = uniq.create_sub_buffer[DType.int32](0, k)
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=head)
    ctx.synchronize()
    out = List[Int32](capacity=k)
    for j in range(k):
        out.append(host.unsafe_ptr().unsafe_load(j))
    _ = head^
    _ = host^
    _ = uniq^
    _ = bsum^
    _ = scan^
    _ = flag^
    _ = cnt^
    _ = tv^
    _ = tk^
    _ = vals^
    _ = keys^
    return out^


def device_exclusive_scan_total(
    ctx: DeviceContext, src: MutPointer[Int32, MutAnyOrigin], mut out: DeviceBuffer[DType.int32], n: Int
) raises:
    """out[0 .. n) = the exclusive scan of src[0 .. n), out[n] = the total,
    over the whole device (`frs_exclusive_scan`). Int32 adds with the usual
    wrap, so every value is the one a one-block scan gives. `src` may be
    `out`'s own pointer (an in-place scan)."""
    if n < 0:
        return
    ctx.enqueue_function[_scan_input_kernel](
        src, out.unsafe_ptr(), Int32(n),
        grid_dim=(_grid(n + 1), 1, 1), block_dim=(SCAN_TPB, 1, 1),
    )
    var bsum = ctx.enqueue_create_buffer[DType.int32](frs_scan_blocks(n + 1))
    frs_exclusive_scan(ctx, out, n + 1, bsum)
    _ = bsum^


def device_column_means32(
    ctx: DeviceContext, x: MutPointer[Float32, MutAnyOrigin], n: Int, k: Int,
    mean: MutPointer[Float32, MutAnyOrigin],
) raises:
    """`mean[f]` = the mean of column f of the row-major n x k matrix x, for
    k <= 32, over the whole device: per-block partial column sums of a
    grid-strided row slice (at most `COLMEAN_BLOCKS` blocks), then one warp
    adding the partials in ascending block order. A fixed order (a pure
    function of n and k)."""
    if n <= 0 or k <= 0:
        return
    if k > 32:
        raise Error("device_column_means32: k must be at most 32")
    var blocks = (n + 2047) // 2048
    if blocks > COLMEAN_BLOCKS:
        blocks = COLMEAN_BLOCKS
    var part = ctx.enqueue_create_buffer[DType.float32](blocks * 32)
    ctx.enqueue_function[_colsum32_partial_kernel](
        x, part.unsafe_ptr(), Int32(n), Int32(k),
        grid_dim=(blocks, 1, 1), block_dim=(256, 1, 1),
    )
    ctx.enqueue_function[_colmean32_fold_kernel](
        part.unsafe_ptr(), mean, Int32(blocks), Int32(n), Int32(k),
        grid_dim=(1, 1, 1), block_dim=(32, 1, 1),
    )
    _ = part^
