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
from std.atomic import Atomic
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


# ------------------------------------------------ DART / RTE bookkeeping --
# lane apple-fast-py2mojo-trees: DART's class counts, its column-sampled
# trees' colid remap, its binary predict codes, TreeExplainer's DART leaf
# spread and RandomTreesEmbedding's leaf numbering. Integers, compares and
# word copies only: every vendor and the host column agree.


def class_counts_kernel(y: MutPointer[Float32, MutAnyOrigin], n: Int64, k: Int64,
                        counts: MutPointer[Int32, MutAnyOrigin], bad: MutPointer[Int32, MutAnyOrigin]):
    """counts[int(y[r])] += 1 (integer atomics); a code outside [0, k) or not
    an integer sets bad[0]."""
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while r < Int(n):
        var v = y.unsafe_load(r)
        var c = Int(v) if (v >= 0.0 and v < Float32(Int(k))) else -1
        if c < 0 or Float32(c) != v:
            bad.unsafe_store(0, Int32(1))
        else:
            _ = Atomic.fetch_add(counts.unsafe_offset(c), Int32(1))
        r += stride


def class_counts_device(y: MutPointer[Float32, MutUntrackedOrigin], n: Int, k: Int,
                        counts: MutPointer[Int32, MutUntrackedOrigin]) raises:
    var ctx = _ctx()
    var d_y = ctx.enqueue_create_buffer[DType.float32](max(1, n))
    if n > 0:
        ctx.enqueue_copy(dst_buf=d_y, src_ptr=y)
    var d_c = ctx.enqueue_create_buffer[DType.int32](k)
    d_c.enqueue_fill(Int32(0))
    var d_bad = ctx.enqueue_create_buffer[DType.int32](1)
    d_bad.enqueue_fill(Int32(0))
    ctx.enqueue_function[class_counts_kernel](
        d_y.unsafe_ptr(), Int64(n), Int64(k), d_c.unsafe_ptr(), d_bad.unsafe_ptr(),
        grid_dim=_blocks(n), block_dim=OPS_TPB,
    )
    var h_bad = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_buf=h_bad, src_buf=d_bad)
    ctx.enqueue_copy(dst_ptr=counts, src_buf=d_c)
    ctx.synchronize()
    var is_bad = h_bad.unsafe_ptr().unsafe_load(0) != Int32(0)
    _ = d_y^
    _ = d_c^
    _ = d_bad^
    _ = h_bad^
    if is_bad:
        raise Error("x_trees class_counts: a class code outside [0, n_classes)")


def remap_cols_kernel(colid: MutPointer[Int32, MutAnyOrigin], nn: Int64, cols: MutPointer[Int32, MutAnyOrigin],
                      m: Int64, bad: MutPointer[Int32, MutAnyOrigin]):
    """colid[g] = cols[colid[g]] for a split node (colid >= 0); a leaf (< 0)
    keeps its word. A column outside [0, m) sets bad[0]."""
    var g = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while g < Int(nn):
        var v = Int(colid.unsafe_load(g))
        if v >= 0:
            if v >= Int(m):
                bad.unsafe_store(0, Int32(1))
            else:
                colid.unsafe_store(g, cols.unsafe_load(v))
        g += stride


def remap_cols_device(colid: MutPointer[Int32, MutUntrackedOrigin], nn: Int,
                      cols: MutPointer[Int32, MutUntrackedOrigin], m: Int) raises:
    var ctx = _ctx()
    var d_id = ctx.enqueue_create_buffer[DType.int32](nn)
    ctx.enqueue_copy(dst_buf=d_id, src_ptr=colid)
    var d_cols = ctx.enqueue_create_buffer[DType.int32](max(1, m))
    if m > 0:
        ctx.enqueue_copy(dst_buf=d_cols, src_ptr=cols)
    var d_bad = ctx.enqueue_create_buffer[DType.int32](1)
    d_bad.enqueue_fill(Int32(0))
    ctx.enqueue_function[remap_cols_kernel](
        d_id.unsafe_ptr(), Int64(nn), d_cols.unsafe_ptr(), Int64(m), d_bad.unsafe_ptr(),
        grid_dim=_blocks(nn), block_dim=OPS_TPB,
    )
    var h_bad = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_buf=h_bad, src_buf=d_bad)
    ctx.synchronize()
    var is_bad = h_bad.unsafe_ptr().unsafe_load(0) != Int32(0)
    if not is_bad:
        ctx.enqueue_copy(dst_ptr=colid, src_buf=d_id)
        ctx.synchronize()
    _ = d_id^
    _ = d_cols^
    _ = d_bad^
    _ = h_bad^
    if is_bad:
        raise Error("x_trees remap_cols: a split column outside the sampled columns")


def positive_codes_kernel(x: MutPointer[UInt64, MutAnyOrigin], n: Int64, codes: MutPointer[Int32, MutAnyOrigin]):
    """codes[r] = 1 if x[r] > 0 else 0 (binary64 words; a NaN and -0.0 give 0)."""
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while r < Int(n):
        var w = x.unsafe_load(r)
        var is_nan = (w & UInt64(0x7FF0000000000000)) == UInt64(0x7FF0000000000000) and (w & UInt64(0x000FFFFFFFFFFFFF)) != 0
        var pos = (w >> 63) == 0 and (w & UInt64(0x7FFFFFFFFFFFFFFF)) != 0 and not is_nan
        codes.unsafe_store(r, Int32(1) if pos else Int32(0))
        r += stride


def positive_codes_device(x: MutPointer[Float64, MutUntrackedOrigin], n: Int,
                          codes: MutPointer[Int32, MutUntrackedOrigin]) raises:
    var ctx = _ctx()
    var d_x = ctx.enqueue_create_buffer[DType.uint64](n)
    ctx.enqueue_copy(dst_buf=d_x, src_ptr=x.bitcast[UInt64]())
    var d_c = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_function[positive_codes_kernel](
        d_x.unsafe_ptr(), Int64(n), d_c.unsafe_ptr(), grid_dim=_blocks(n), block_dim=OPS_TPB,
    )
    ctx.enqueue_copy(dst_ptr=codes, src_buf=d_c)
    ctx.synchronize()
    _ = d_x^
    _ = d_c^


def spread_leaves_kernel(vals: MutPointer[UInt32, MutAnyOrigin], offs: MutPointer[Int32, MutAnyOrigin], t: Int64,
                         nn: Int64, k: Int64, dst: MutPointer[UInt32, MutAnyOrigin]):
    """dst[g * k + c] = vals[g] when c == (tree of node g) mod k, else +0.0;
    one thread per cell, the tree found by binary search over offs."""
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    var kk = Int(k)
    while cell < Int(nn) * kk:
        var g = cell // kk
        var c = cell - g * kk
        var lo = 0
        var hi = Int(t)
        while hi - lo > 1:
            var mid = (lo + hi) // 2
            if Int(offs.unsafe_load(mid)) <= g:
                lo = mid
            else:
                hi = mid
        dst.unsafe_store(cell, vals.unsafe_load(g) if c == lo % kk else UInt32(0))
        cell += stride


def spread_leaves_device(vals: MutPointer[Float32, MutUntrackedOrigin], offs: MutPointer[Int32, MutUntrackedOrigin],
                         t: Int, nn: Int, k: Int, dst: MutPointer[Float32, MutUntrackedOrigin]) raises:
    var ctx = _ctx()
    var d_v = ctx.enqueue_create_buffer[DType.uint32](nn)
    ctx.enqueue_copy(dst_buf=d_v, src_ptr=vals.bitcast[UInt32]())
    var d_o = ctx.enqueue_create_buffer[DType.int32](t + 1)
    ctx.enqueue_copy(dst_buf=d_o, src_ptr=offs)
    var d_d = ctx.enqueue_create_buffer[DType.uint32](nn * k)
    ctx.enqueue_function[spread_leaves_kernel](
        d_v.unsafe_ptr(), d_o.unsafe_ptr(), Int64(t), Int64(nn), Int64(k), d_d.unsafe_ptr(),
        grid_dim=_blocks(nn * k), block_dim=OPS_TPB,
    )
    ctx.enqueue_copy(dst_ptr=dst.bitcast[UInt32](), src_buf=d_d)
    ctx.synchronize()
    _ = d_v^
    _ = d_o^
    _ = d_d^


# ---------------------------------------------------------------------------
# cpu2-l5-trees (2026-10-04): the forests' class_weight row expansion
# (python/mojolearn/randomforest.py `_class_weight_rows`) left the host
# helpers `bincount_i64` and `gather_rows_bytes`: the per-class counts are
# integer atomics over int32 codes, the per-row weight is the class's
# float32 word copied (as a 32-bit word, so no device flushes a subnormal).
# The host column runs `code_counts_host` / `class_rows_host`, the same
# integers and the same words.
# ---------------------------------------------------------------------------
def code_counts_kernel(codes: MutPointer[Int32, MutAnyOrigin], n: Int64, k: Int64,
                       counts: MutPointer[Int32, MutAnyOrigin], bad: MutPointer[Int32, MutAnyOrigin]):
    """counts[codes[r]] += 1 (integer atomics: order-free); a code outside
    [0, k) sets bad[0]."""
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while r < Int(n):
        var c = Int(codes.unsafe_load(r))
        if c < 0 or c >= Int(k):
            bad.unsafe_store(0, Int32(1))
        else:
            _ = Atomic.fetch_add(counts.unsafe_offset(c), Int32(1))
        r += stride


def class_rows_kernel(codes: MutPointer[Int32, MutAnyOrigin], n: Int64, k: Int64,
                      values: MutPointer[UInt32, MutAnyOrigin], dst: MutPointer[UInt32, MutAnyOrigin],
                      bad: MutPointer[Int32, MutAnyOrigin]):
    """dst[r] = values[codes[r]] (a word copy); a code outside [0, k) sets
    bad[0] and writes nothing for that row."""
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while r < Int(n):
        var c = Int(codes.unsafe_load(r))
        if c < 0 or c >= Int(k):
            bad.unsafe_store(0, Int32(1))
        else:
            dst.unsafe_store(r, values.unsafe_load(c))
        r += stride


def code_counts_device(codes: MutPointer[Int32, MutUntrackedOrigin], n: Int, k: Int,
                       counts: MutPointer[Int32, MutUntrackedOrigin]) raises:
    """counts (int32, k) = the rows of each int32 class code, on the device."""
    var ctx = _ctx()
    var d_y = ctx.enqueue_create_buffer[DType.int32](max(1, n))
    if n > 0:
        ctx.enqueue_copy(dst_buf=d_y, src_ptr=codes)
    var d_c = ctx.enqueue_create_buffer[DType.int32](k)
    d_c.enqueue_fill(Int32(0))
    var d_bad = ctx.enqueue_create_buffer[DType.int32](1)
    d_bad.enqueue_fill(Int32(0))
    ctx.enqueue_function[code_counts_kernel](
        d_y.unsafe_ptr(), Int64(n), Int64(k), d_c.unsafe_ptr(), d_bad.unsafe_ptr(),
        grid_dim=_blocks(n), block_dim=OPS_TPB,
    )
    var h_bad = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_buf=h_bad, src_buf=d_bad)
    ctx.enqueue_copy(dst_ptr=counts, src_buf=d_c)
    ctx.synchronize()
    var is_bad = h_bad.unsafe_ptr().unsafe_load(0) != Int32(0)
    _ = d_y^
    _ = d_c^
    _ = d_bad^
    _ = h_bad^
    if is_bad:
        raise Error("x_trees code_counts: a class code outside [0, n_classes)")


def class_rows_device(codes: MutPointer[Int32, MutUntrackedOrigin], n: Int, k: Int,
                      values: MutPointer[Float32, MutUntrackedOrigin], dst: MutPointer[Float32, MutUntrackedOrigin]) raises:
    """dst (float32, n) = values (float32, k) at each row's int32 code, on
    the device; nothing written to `dst` on a refusal."""
    if n <= 0:
        return
    var ctx = _ctx()
    var d_y = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_buf=d_y, src_ptr=codes)
    var d_v = ctx.enqueue_create_buffer[DType.uint32](k)
    ctx.enqueue_copy(dst_buf=d_v, src_ptr=values.bitcast[UInt32]())
    var d_d = ctx.enqueue_create_buffer[DType.uint32](n)
    var d_bad = ctx.enqueue_create_buffer[DType.int32](1)
    d_bad.enqueue_fill(Int32(0))
    ctx.enqueue_function[class_rows_kernel](
        d_y.unsafe_ptr(), Int64(n), Int64(k), d_v.unsafe_ptr(), d_d.unsafe_ptr(), d_bad.unsafe_ptr(),
        grid_dim=_blocks(n), block_dim=OPS_TPB,
    )
    var h_bad = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_buf=h_bad, src_buf=d_bad)
    ctx.synchronize()
    var is_bad = h_bad.unsafe_ptr().unsafe_load(0) != Int32(0)
    if not is_bad:
        ctx.enqueue_copy(dst_ptr=dst.bitcast[UInt32](), src_buf=d_d)
        ctx.synchronize()
    _ = d_y^
    _ = d_v^
    _ = d_d^
    _ = d_bad^
    _ = h_bad^
    if is_bad:
        raise Error("x_trees class_rows: a class code outside [0, n_classes)")


def code_counts_host(codes: MutPointer[Int32, MutUntrackedOrigin], n: Int, k: Int,
                     counts: MutPointer[Int32, MutUntrackedOrigin]) raises:
    """The host column's `code_counts_device`: the same integers."""
    for c in range(k):
        counts[unsafe_offset=c] = Int32(0)
    for r in range(n):
        var c = Int(codes[unsafe_offset=r])
        if c < 0 or c >= k:
            raise Error("x_trees code_counts: a class code outside [0, n_classes)")
        counts[unsafe_offset=c] = counts[unsafe_offset=c] + Int32(1)


def class_rows_host(codes: MutPointer[Int32, MutUntrackedOrigin], n: Int, k: Int,
                    values: MutPointer[Float32, MutUntrackedOrigin], dst: MutPointer[Float32, MutUntrackedOrigin]) raises:
    """The host column's `class_rows_device`: the same words, the same
    refusal (checked before any row is written)."""
    for r in range(n):
        var c = Int(codes[unsafe_offset=r])
        if c < 0 or c >= k:
            raise Error("x_trees class_rows: a class code outside [0, n_classes)")
    var vw = values.bitcast[UInt32]()
    var dw = dst.bitcast[UInt32]()
    for r in range(n):
        dw[unsafe_offset=r] = vw[unsafe_offset=Int(codes[unsafe_offset=r])]
