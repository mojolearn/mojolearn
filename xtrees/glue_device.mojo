# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The wrapper glue of python/mojolearn/_expansion_trees.py on the device
(lane apple-fast-py2mojo-trees, 2026-10-03: "everything is supposed to be in
mojo"). OneVsRest's 0/1 targets, MultiOutputClassifier's predict columns
stacked into the (n, n_outputs) answer and a binary predict_proba's
(1 - p, p) rows left their Python row loops. Each is one grid-wide launch
over the rows; `xtrees/api.mojo` routes every GPU build here and the CPU
column (`TARGET_COLUMN == COLUMN_CPU`) to the serial loops of
`xtrees/ops.mojo`.

BITS. `indicator_codes` and `stack_w64` are compares and word copies.
`binary_proba` widens a binary32 exactly (`sf64_from_f32`'s rule, NaN keeps
its payload) and forms 1 - p with `sf64_sub` (correctly rounded binary64 on
every vendor, the Apple GPU has no float64): the IEEE `1.0 - float(p)` the
Python row loop computed, so every vendor and the host agree word for word.
A NaN p gives the quieted NaN operand (x86 and ARM `1.0 - nan`).
"""
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from checks.soft_f64 import SF64_ONE, sf64_sub
from xtrees.ops_device import OPS_TPB, _blocks, _ctx


def widen_f32_word(b32: UInt32) -> UInt64:
    """The exact binary32 -> binary64 widening as words; a NaN keeps its sign
    and payload (the hardware conversion's rule)."""
    var b = UInt64(b32)
    var s = b >> 31
    var e = Int((b >> 23) & UInt64(0xFF))
    var f = b & UInt64(0x7FFFFF)
    if e == 0xFF:
        return (s << 63) | (UInt64(0x7FF) << 52) | (f << 29)
    if e == 0:
        if f == 0:
            return s << 63
        var lead = 0
        var t = f
        while t > 1:
            t >>= 1
            lead += 1
        var frac = (f << UInt64(52 - lead)) & UInt64(0x000FFFFFFFFFFFFF)
        return (s << 63) | (UInt64(lead - 149 + 1023) << 52) | frac
    return (s << 63) | (UInt64(e + 896) << 52) | (f << 29)


def one_minus_word(p: UInt64) -> UInt64:
    """1.0 - p as binary64 words; a NaN p returns p quieted."""
    if (p & UInt64(0x7FF0000000000000)) == UInt64(0x7FF0000000000000) and (p & UInt64(0x000FFFFFFFFFFFFF)) != 0:
        return p | UInt64(0x0008000000000000)
    return sf64_sub(SF64_ONE, p)


def indicator_kernel(
    codes: MutPointer[Int32, MutAnyOrigin], n: Int64, cls: Int64,
    out_i: MutPointer[Int32, MutAnyOrigin], out_f: MutPointer[UInt64, MutAnyOrigin], want_f: Int32,
):
    """out_i[r] = codes[r] == cls (and out_f 1.0 / 0.0 words), one thread per row."""
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while r < Int(n):
        var hit = Int64(codes.unsafe_load(r)) == cls
        out_i.unsafe_store(r, Int32(1) if hit else Int32(0))
        if want_f != 0:
            out_f.unsafe_store(r, SF64_ONE if hit else UInt64(0))
        r += stride


def indicator_codes_device(
    codes: MutPointer[Int32, MutUntrackedOrigin], n: Int, cls: Int,
    out_i: MutPointer[Int32, MutUntrackedOrigin], out_f: MutPointer[Float64, MutUntrackedOrigin], want_f: Bool,
) raises:
    """`ops.indicator_codes` on the device."""
    if n <= 0:
        return
    var ctx = _ctx()
    var d_codes = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_buf=d_codes, src_ptr=codes)
    var d_i = ctx.enqueue_create_buffer[DType.int32](n)
    var d_f = ctx.enqueue_create_buffer[DType.uint64](n if want_f else 1)
    ctx.enqueue_function[indicator_kernel](
        d_codes.unsafe_ptr(), Int64(n), Int64(cls), d_i.unsafe_ptr(), d_f.unsafe_ptr(),
        Int32(1) if want_f else Int32(0),
        grid_dim=_blocks(n), block_dim=OPS_TPB,
    )
    ctx.enqueue_copy(dst_ptr=out_i, src_buf=d_i)
    if want_f:
        ctx.enqueue_copy(dst_ptr=out_f.bitcast[UInt64](), src_buf=d_f)
    ctx.synchronize()
    _ = d_codes^
    _ = d_i^
    _ = d_f^


def stack_w64_kernel(
    dst: MutPointer[UInt64, MutAnyOrigin], src: MutPointer[UInt64, MutAnyOrigin], n: Int64, m: Int64,
):
    """dst[r * m + j] = src[j * n + r] (column j of the answer from its own
    contiguous column), one thread per cell (a word copy)."""
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    var total = Int(n) * Int(m)
    while c < total:
        var r = c // Int(m)
        var j = c - r * Int(m)
        dst.unsafe_store(c, src.unsafe_load(j * Int(n) + r))
        c += stride


def stack_w64_device(cols: List[Int], n: Int, dst: MutPointer[UInt64, MutUntrackedOrigin]) raises:
    """dst (n x m, row-major 8-byte words) with column j = the n words at
    address cols[j]."""
    var m = len(cols)
    if n <= 0 or m <= 0:
        return
    var ctx = _ctx()
    var d_src = ctx.enqueue_create_buffer[DType.uint64](n * m)
    for j in range(m):
        var view = d_src.create_sub_buffer[DType.uint64](j * n, n)
        ctx.enqueue_copy(dst_buf=view, src_ptr=MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=cols[j]))
    var d_dst = ctx.enqueue_create_buffer[DType.uint64](n * m)
    ctx.enqueue_function[stack_w64_kernel](
        d_dst.unsafe_ptr(), d_src.unsafe_ptr(), Int64(n), Int64(m),
        grid_dim=_blocks(n * m), block_dim=OPS_TPB,
    )
    ctx.enqueue_copy(dst_ptr=dst, src_buf=d_dst)
    ctx.synchronize()
    _ = d_src^
    _ = d_dst^


def binary_proba_kernel(
    p: MutPointer[UInt32, MutAnyOrigin], n: Int64, res: MutPointer[UInt64, MutAnyOrigin],
):
    """res[2 r + 1] = p[r] widened, res[2 r] = 1 - that, one thread per row."""
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while r < Int(n):
        var w = widen_f32_word(p.unsafe_load(r))
        res.unsafe_store(2 * r + 1, w)
        res.unsafe_store(2 * r, one_minus_word(w))
        r += stride


def binary_proba_device(
    p: MutPointer[Float32, MutUntrackedOrigin], n: Int, res: MutPointer[Float64, MutUntrackedOrigin],
) raises:
    """`ops.binary_proba` on the device."""
    if n <= 0:
        return
    var ctx = _ctx()
    var d_p = ctx.enqueue_create_buffer[DType.uint32](n)
    ctx.enqueue_copy(dst_buf=d_p, src_ptr=p.bitcast[UInt32]())
    var d_out = ctx.enqueue_create_buffer[DType.uint64](2 * n)
    ctx.enqueue_function[binary_proba_kernel](
        d_p.unsafe_ptr(), Int64(n), d_out.unsafe_ptr(),
        grid_dim=_blocks(n), block_dim=OPS_TPB,
    )
    ctx.enqueue_copy(dst_ptr=res.bitcast[UInt64](), src_buf=d_out)
    ctx.synchronize()
    _ = d_p^
    _ = d_out^
