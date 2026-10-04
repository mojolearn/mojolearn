# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The weighted bootstrap on the device, in exact integers
(fam2-forests, 2026-10-04, `IDN_RF_WEIGHTED_BOOTSTRAP_DEVICE`).

Before (DEVIATION 305/306): `RowSampler.prepare_weights` built a Float64
CDF on the host and every tree drew `n_sampled_rows` Float64 uniforms on
the host, binary-searched the host CDF, uploaded the rows and drained the
queue. Metal has no Float64, so that arm could never move as written.

Now, IDENTICAL on every vendor and on the host column
(`ensemble/host/rf_oracle.mojo: host_weight_qcdf` / `host_sampled_rows`):

  quantum  each Float32 weight becomes an unsigned integer by an exact
           power-of-two scale taken from the LARGEST weight's exponent:
           the largest weight lands in [2^30, 2^31), every other weight is
           its own significand shifted by the exponent difference (floor),
           and a nonzero weight that would floor to 0 is clamped to 1, so a
           row with any weight keeps a nonzero chance. Integer shifts on
           the bit pattern only: no float operation, same on every device.
  cdf      the UInt64 inclusive prefix sum of the quanta (at most 2^31
           rows x 2^31, inside 64 bits), by tile scans of 256: each thread
           sums its tile's prefix, tile totals are scanned the same way
           three more levels up, and one pass adds the offsets. Integer
           adds, so the scan shape cannot move a bit.
  draw     per tree, one thread per sampled row: PCG (the quantile
           sampler's `PCGenerator`, subsequence = the sample index, seed =
           the tree's hashed seed) draws an integer in [0, total) by
           Lemire's bounded draw and `upper_bound`s it in the CDF. A row's
           probability is its quantum over the total, i.e. its weight to
           within one quantum of the largest weight's 2^-30.

No per-tree upload and no per-tree synchronize. THE DRAWN ROWS CHANGE
(a new stream replaces Philox `uniform<double>`), on all four columns
together. `-D MOJOLEARN_IDN_RF_WEIGHTED_BOOTSTRAP_DEVICE_OFF` (or
`MOJOLEARN_IDN_ALL_OFF`) restores the host Float64 arm everywhere.
"""
from std.gpu import block_dim, block_idx, thread_idx
from std.math import ceildiv
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.launch_clock import log_launch_ctx
from ensemble.decisiontree.batched_levelalgo.quantiles import (
    PCGenerator,
    custom_next_uniform_int_u64,
)

comptime IDN_RF_WEIGHTED_BOOTSTRAP_DEVICE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not (
        is_defined["MOJOLEARN_IDN_RF_WEIGHTED_BOOTSTRAP_DEVICE_OFF"]()
        or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
    )
)

#: elements per scan / max tile, and threads per block
comptime WB_TILE = 256


@always_inline
def weight_quantum(bits: UInt32, max_bits: UInt32) -> UInt64:
    """The integer weight of a non-negative Float32 given by its bit
    pattern, on the scale that puts the largest weight (`max_bits`, sign
    cleared) in [2^30, 2^31). Zero stays zero; any other weight is at
    least 1. `host_weight_quantum` in `ensemble/host/rf_oracle.mojo` is
    this function, statement for statement."""
    if (bits & UInt32(0x7FFFFFFF)) == UInt32(0):
        return UInt64(0)
    var e = Int((bits >> 23) & UInt32(0xFF))
    var sig = UInt64(Int(bits & UInt32(0x7FFFFF)))
    if e == 0:
        # subnormal: the exponent of e == 1, no hidden bit
        e = 1
    else:
        sig = sig | UInt64(0x800000)
    var em = Int((max_bits >> 23) & UInt32(0xFF))
    if em == 0:
        em = 1
    var d = e - em + 7
    var q = UInt64(0)
    if d >= 0:
        q = sig << UInt64(d)
    elif d > -24:
        q = sig >> UInt64(-d)
    if q == UInt64(0):
        q = UInt64(1)
    return q


def wb_max_tile_kernel(
    src: MutPointer[UInt32, MutAnyOrigin],
    n_src: Int32,
    dst: MutPointer[UInt32, MutAnyOrigin],
):
    """`dst[t]` = the largest of `src`'s tile `t` (256 entries), sign bit
    cleared. Non-negative Float32 bit patterns order as their values, so
    this is the max weight; applied level after level it is the global
    max. One thread per tile."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var start = t * WB_TILE
    if start >= Int(n_src):
        return
    var stop = start + WB_TILE
    if stop > Int(n_src):
        stop = Int(n_src)
    var best = UInt32(0)
    var j = start
    while j < stop:
        var v = src.unsafe_load(j) & UInt32(0x7FFFFFFF)
        if v > best:
            best = v
        j += 1
    dst.unsafe_store(t, best)


def wb_quantum_tile_scan_kernel(
    wbits: MutPointer[UInt32, MutAnyOrigin],
    max_bits: MutPointer[UInt32, MutAnyOrigin],
    n: Int32,
    cdf: MutPointer[UInt64, MutAnyOrigin],
    tot: MutPointer[UInt64, MutAnyOrigin],
):
    """Level 0: `cdf[i]` = the sum of the quanta of row `i`'s tile up to
    and including `i`; the tile's last thread also stores the tile total.
    One thread per row, at most 256 loads each, no shared memory."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n):
        return
    var mb = max_bits.unsafe_load(0)
    var t = i // WB_TILE
    var j = t * WB_TILE
    var s = UInt64(0)
    while j <= i:
        s += weight_quantum(wbits.unsafe_load(j), mb)
        j += 1
    cdf.unsafe_store(i, s)
    if i == Int(n) - 1 or (i + 1) % WB_TILE == 0:
        tot.unsafe_store(t, s)


def wb_u64_tile_scan_kernel(
    src: MutPointer[UInt64, MutAnyOrigin],
    n_src: Int32,
    incl: MutPointer[UInt64, MutAnyOrigin],
    tot: MutPointer[UInt64, MutAnyOrigin],
):
    """Upper levels: the same tile scan over the level below's totals."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_src):
        return
    var t = i // WB_TILE
    var j = t * WB_TILE
    var s = UInt64(0)
    while j <= i:
        s += src.unsafe_load(j)
        j += 1
    incl.unsafe_store(i, s)
    if i == Int(n_src) - 1 or (i + 1) % WB_TILE == 0:
        tot.unsafe_store(t, s)


def wb_finish_kernel(
    cdf: MutPointer[UInt64, MutAnyOrigin],
    incl1: MutPointer[UInt64, MutAnyOrigin],
    incl2: MutPointer[UInt64, MutAnyOrigin],
    incl3: MutPointer[UInt64, MutAnyOrigin],
    n: Int32,
):
    """`cdf[i]` += everything before row `i`'s tile: the tiles before it
    in its group (level 1), the groups before that group in its
    super-group (level 2), and the super-groups before it (level 3).
    256^4 rows is past any Int32 row count, so three levels close it."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n):
        return
    var off = UInt64(0)
    var t = i // WB_TILE
    if t % WB_TILE != 0:
        off += incl1.unsafe_load(t - 1)
    t = t // WB_TILE
    if t % WB_TILE != 0:
        off += incl2.unsafe_load(t - 1)
    t = t // WB_TILE
    if t % WB_TILE != 0:
        off += incl3.unsafe_load(t - 1)
    cdf.unsafe_store(i, cdf.unsafe_load(i) + off)


def weighted_bootstrap_rows_kernel(
    rows: MutPointer[Int32, MutAnyOrigin],
    cdf: MutPointer[UInt64, MutAnyOrigin],
    n_sampled: Int32,
    n_rows: Int32,
    seed_bits: Int32,
):
    """One thread per sampled row: an integer in [0, total) from the PCG
    stream (seed = the tree's hashed 32-bit seed, subsequence = the sample
    index), then `upper_bound` over the integer CDF -- the first row whose
    cumulative quantum is STRICTLY GREATER than the draw, so a zero-weight
    row (CDF equal to its predecessor's) is never drawn."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_sampled):
        return
    var total = cdf.unsafe_load(Int(n_rows) - 1)
    # the seed travels as an Int32 bit pattern; the mask undoes the sign
    # extension of the widening
    var sb = seed_bits
    var seed = UInt64(Int(sb) & 0xFFFFFFFF)
    var gen = PCGenerator.init_pcg(seed, UInt64(i), UInt64(0))
    var u = custom_next_uniform_int_u64(gen, UInt64(0), total)
    var lo = 0
    var hi = Int(n_rows)
    while lo < hi:
        var mid = (lo + hi) // 2
        if cdf.unsafe_load(mid) <= u:
            lo = mid + 1
        else:
            hi = mid
    if lo > Int(n_rows) - 1:
        lo = Int(n_rows) - 1
    rows.unsafe_store(i, Int32(lo))


def build_weight_cdf_device(
    ctx: DeviceContext,
    mut wbits: DeviceBuffer[DType.uint32],
    n_rows: Int,
) raises -> DeviceBuffer[DType.uint64]:
    """The UInt64 inclusive CDF of the quanta of `wbits` (the weights'
    Float32 bit patterns, already on the device). Enqueues everything and
    drains once, so every scratch buffer below outlives its launches."""
    var n1 = ceildiv(n_rows, WB_TILE)
    var n2 = ceildiv(n1, WB_TILE)
    var n3 = ceildiv(n2, WB_TILE)
    var n4 = ceildiv(n3, WB_TILE)

    # the largest weight, level by level
    var m1 = ctx.enqueue_create_buffer[DType.uint32](n1)
    var m2 = ctx.enqueue_create_buffer[DType.uint32](n2)
    var m3 = ctx.enqueue_create_buffer[DType.uint32](n3)
    var m4 = ctx.enqueue_create_buffer[DType.uint32](n4)
    log_launch_ctx(ctx, "wboot_max_l0")
    ctx.enqueue_function[wb_max_tile_kernel](
        wbits.unsafe_ptr(), Int32(n_rows), m1.unsafe_ptr(),
        grid_dim=ceildiv(n1, WB_TILE), block_dim=WB_TILE,
    )
    log_launch_ctx(ctx, "wboot_max_l1")
    ctx.enqueue_function[wb_max_tile_kernel](
        m1.unsafe_ptr(), Int32(n1), m2.unsafe_ptr(),
        grid_dim=ceildiv(n2, WB_TILE), block_dim=WB_TILE,
    )
    log_launch_ctx(ctx, "wboot_max_l2")
    ctx.enqueue_function[wb_max_tile_kernel](
        m2.unsafe_ptr(), Int32(n2), m3.unsafe_ptr(),
        grid_dim=ceildiv(n3, WB_TILE), block_dim=WB_TILE,
    )
    log_launch_ctx(ctx, "wboot_max_l3")
    ctx.enqueue_function[wb_max_tile_kernel](
        m3.unsafe_ptr(), Int32(n3), m4.unsafe_ptr(),
        grid_dim=ceildiv(n4, WB_TILE), block_dim=WB_TILE,
    )
    # n4 is 1 for every Int32 row count (256^4 > 2^31), so m4[0] is the max.

    var cdf = ctx.enqueue_create_buffer[DType.uint64](n_rows)
    var tot0 = ctx.enqueue_create_buffer[DType.uint64](n1)
    var incl1 = ctx.enqueue_create_buffer[DType.uint64](n1)
    var tot1 = ctx.enqueue_create_buffer[DType.uint64](n2)
    var incl2 = ctx.enqueue_create_buffer[DType.uint64](n2)
    var tot2 = ctx.enqueue_create_buffer[DType.uint64](n3)
    var incl3 = ctx.enqueue_create_buffer[DType.uint64](n3)
    var tot3 = ctx.enqueue_create_buffer[DType.uint64](n4)
    log_launch_ctx(ctx, "wboot_scan_l0")
    ctx.enqueue_function[wb_quantum_tile_scan_kernel](
        wbits.unsafe_ptr(), m4.unsafe_ptr(), Int32(n_rows),
        cdf.unsafe_ptr(), tot0.unsafe_ptr(),
        grid_dim=ceildiv(n_rows, WB_TILE), block_dim=WB_TILE,
    )
    log_launch_ctx(ctx, "wboot_scan_l1")
    ctx.enqueue_function[wb_u64_tile_scan_kernel](
        tot0.unsafe_ptr(), Int32(n1), incl1.unsafe_ptr(), tot1.unsafe_ptr(),
        grid_dim=ceildiv(n1, WB_TILE), block_dim=WB_TILE,
    )
    log_launch_ctx(ctx, "wboot_scan_l2")
    ctx.enqueue_function[wb_u64_tile_scan_kernel](
        tot1.unsafe_ptr(), Int32(n2), incl2.unsafe_ptr(), tot2.unsafe_ptr(),
        grid_dim=ceildiv(n2, WB_TILE), block_dim=WB_TILE,
    )
    log_launch_ctx(ctx, "wboot_scan_l3")
    ctx.enqueue_function[wb_u64_tile_scan_kernel](
        tot2.unsafe_ptr(), Int32(n3), incl3.unsafe_ptr(), tot3.unsafe_ptr(),
        grid_dim=ceildiv(n3, WB_TILE), block_dim=WB_TILE,
    )
    log_launch_ctx(ctx, "wboot_scan_finish")
    ctx.enqueue_function[wb_finish_kernel](
        cdf.unsafe_ptr(), incl1.unsafe_ptr(), incl2.unsafe_ptr(),
        incl3.unsafe_ptr(), Int32(n_rows),
        grid_dim=ceildiv(n_rows, WB_TILE), block_dim=WB_TILE,
    )
    # Once per forest. The scratch buffers were handed to kernels as raw
    # pointers, so they must be named AFTER the drain (a Mojo local dies at
    # its last named use).
    ctx.synchronize()
    _ = m1^
    _ = m2^
    _ = m3^
    _ = m4^
    _ = tot0^
    _ = incl1^
    _ = tot1^
    _ = incl2^
    _ = tot2^
    _ = incl3^
    _ = tot3^
    return cdf^


def launch_weighted_bootstrap_rows(
    ctx: DeviceContext,
    mut rows: DeviceBuffer[DType.int32],
    mut cdf: DeviceBuffer[DType.uint64],
    n_sampled: Int,
    n_rows: Int,
    tree_seed: UInt32,
) raises:
    """One tree's weighted bootstrap rows; no upload, no drain. `rows`
    and `cdf` are the sampler's fields, alive past `fit_forest`'s final
    synchronize."""
    if n_sampled <= 0:
        return
    var s = Int(tree_seed)
    if s >= 2147483648:
        s -= 4294967296
    log_launch_ctx(ctx, "wboot_rows")
    ctx.enqueue_function[weighted_bootstrap_rows_kernel](
        rows.unsafe_ptr(),
        cdf.unsafe_ptr(),
        Int32(n_sampled),
        Int32(n_rows),
        Int32(s),
        grid_dim=ceildiv(n_sampled, WB_TILE),
        block_dim=WB_TILE,
    )
