# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Device forms of the category dictionary ops (lane classical-encoders,
2026-10-07, IDENTICAL, every vendor). Integer moves and counts only: the
outputs are exact functions of the input words, so NVIDIA, AMD, Apple and the
host column (the units in x_prep/prims.mojo) write the same words.

`unique_inverse` (C08 routes, q = [S, n, d, U, CNT, X, CODES]): replaces the
unit's one thread per column doing n binary searches.
  1. Every (column, row) loads the key of canon(ftz(X[i, c])) and its row i.
  2. A stable LSD radix sort of the keys that carries the row index
     (x_prep/dradix.mojo's chunked counting sort; C07 sets its digit bits and
     chunk rows). Equal keys keep row order, so the order is fixed.
  3. Run scan: every (column, chunk) counts the runs that open in it; every
     column turns its chunk counts into exclusive offsets (CNT = the total);
     every (column, chunk) writes the first word of each run at U[c*n + k]
     and scatters code k-1 to CODES[row*d + c] for every sorted position.
  No binary search; O(n*d) parallel work over (column, chunk) tasks.

`unique_cols` (C08_UNIQUE_SCAN, q = [S, n, d, U, CNT]): steps 3 without the
codes, over the already sorted words S, for every column in one launch each.

A run opens at sorted position i when i == 0 or key(i) != key(i-1) (`key`
equality, as unique_cols_unit). The radix keys are a bijection of the words,
so key equality there is word equality, and canon words make it `key` equality.
"""
from std.gpu import block_idx, block_dim, thread_idx
from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import ftz
from x_prep.common import FP, canon, key, st
from x_prep.dradix import (
    RUP, RBS, RWS, RBITS, RBINS, RPASSES, radix_key, radix_word, radix_chunks, radix_hist_kernel, radix_hsum_kernel,
    radix_hstart_kernel,
)

#: sorted positions one run-scan task walks (a version constant: it fixes no
#: bit, only how many (column, chunk) tasks the scan launches)
comptime RUN_ROWS = 2048


def dict_load_kernel(f: FP, w: RUP, KA: Int32, IA: Int32, X: Int32, n: Int32, d: Int32, total: Int32):
    """t = c * n + i: w[KA + t] = the key of canon(ftz(X[i, c])), w[IA + t] = i."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= Int(total):
        return
    var nn = Int(n)
    var c = t // nn
    var i = t - c * nn
    var v = canon(ftz(f.unsafe_load(Int(X) + i * Int(d) + c)))
    w[Int(KA) + t] = radix_key(bitcast[DType.uint32](v))
    w[Int(IA) + t] = UInt32(i)


def dict_scatter_kernel(w: RUP, src: Int32, dst: Int32, isrc: Int32, idst: Int32, H: Int32, START: Int32, n: Int32,
                        ch_n: Int32, shift: Int32, total: Int32):
    """radix_scatter_kernel that moves each key's row index with it."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= Int(total):
        return
    var nn = Int(n)
    var chn = Int(ch_n)
    var c = t // chn
    var ch = t - c * chn
    var hb = Int(H) + t * RBINS
    var sb = Int(START) + c * RBINS
    for b in range(RBINS):
        w[hb + b] = w[hb + b] + w[sb + b]
    var cs = (nn + chn - 1) // chn
    var lo = min(ch * cs, nn)
    var hi = min(lo + cs, nn)
    var base = Int(src) + c * nn
    var ibase = Int(isrc) + c * nn
    var out = Int(dst) + c * nn
    var iout = Int(idst) + c * nn
    var sh = UInt32(Int(shift))
    for i in range(lo, hi):
        var v = w[base + i]
        var dg = Int((v >> sh) & UInt32(RBINS - 1))
        var k = Int(w[hb + dg])
        w[out + k] = v
        w[iout + k] = w[ibase + i]
        w[hb + dg] = UInt32(k + 1)


@always_inline
def _key_at[FROM_KEYS: Bool](f: FP, w: RUP, src: Int, at: Int) -> UInt32:
    comptime if FROM_KEYS:
        return w[src + at]
    else:
        return key(f.unsafe_load(src + at))


def runs_count_kernel[FROM_KEYS: Bool](f: FP, w: RUP, src: Int32, n: Int32, ch_n: Int32, RC: Int32, total: Int32):
    """t = c * CH + ch: w[RC + t] = how many runs open in chunk ch of sorted
    column c (keys in w, or words in f at src)."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= Int(total):
        return
    var nn = Int(n)
    var chn = Int(ch_n)
    var c = t // chn
    var ch = t - c * chn
    var cs = (nn + chn - 1) // chn
    var lo = min(ch * cs, nn)
    var hi = min(lo + cs, nn)
    var base = Int(src) + c * nn
    var k = 0
    var last = UInt32(0)
    if lo > 0 and lo < hi:
        last = _key_at[FROM_KEYS](f, w, base, lo - 1)
    for i in range(lo, hi):
        var kv = _key_at[FROM_KEYS](f, w, base, i)
        if i == 0 or kv != last:
            k += 1
        last = kv
    w[Int(RC) + t] = UInt32(k)


def runs_offset_kernel(f: FP, w: RUP, RC: Int32, RO: Int32, ch_n: Int32, CNT: Int32, total: Int32):
    """t = column c: w[RO + c*CH + ch] = the runs of column c before chunk ch
    (in chunk order, a fixed order); CNT[c] = the run count as a float."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= Int(total):
        return
    var chn = Int(ch_n)
    var acc = UInt32(0)
    for ch in range(chn):
        var at = t * chn + ch
        var v = w[Int(RC) + at]
        w[Int(RO) + at] = acc
        acc = acc + v
    st(f, Int(CNT) + t, Float32(Int(acc)))


def runs_write_kernel[FROM_KEYS: Bool](f: FP, w: RUP, src: Int32, isrc: Int32, n: Int32, d: Int32, ch_n: Int32,
                                       RO: Int32, U: Int32, CODES: Int32, total: Int32):
    """t = c * CH + ch: the first word of every run opening in the chunk at
    U[c*n + k] (k = its run number); with CODES >= 0 (FROM_KEYS only), every
    sorted position's run number into CODES[row * d + c], row = its index."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= Int(total):
        return
    var nn = Int(n)
    var chn = Int(ch_n)
    var c = t // chn
    var ch = t - c * chn
    var cs = (nn + chn - 1) // chn
    var lo = min(ch * cs, nn)
    var hi = min(lo + cs, nn)
    var base = Int(src) + c * nn
    var ub = Int(U) + c * nn
    var k = Int(w[Int(RO) + t])
    var last = UInt32(0)
    if lo > 0 and lo < hi:
        last = _key_at[FROM_KEYS](f, w, base, lo - 1)
    for i in range(lo, hi):
        var kv = _key_at[FROM_KEYS](f, w, base, i)
        if i == 0 or kv != last:
            comptime if FROM_KEYS:
                f.unsafe_store(ub + k, bitcast[DType.float32](radix_word(kv)))
            else:
                f.unsafe_store(ub + k, f.unsafe_load(base + i))
            k += 1
        last = kv
        comptime if FROM_KEYS:
            if Int(CODES) >= 0:
                var row = Int(w[Int(isrc) + c * nn + i])
                st(f, Int(CODES) + row * Int(d) + c, Float32(k - 1))


def _rb(total: Int) -> Int:
    return (total + RBS - 1) // RBS


def unique_runs_scratch_words(n: Int, cols: Int) -> Int:
    """Scratch of `unique_runs_device`: chunk counts and offsets."""
    return 2 * max(cols, 0) * radix_chunks(n, RUN_ROWS)


def dict_inverse_scratch_words(n: Int, cols: Int, chunk_rows: Int) -> Int:
    """Scratch of `dict_inverse_device`: two key and two index blocks, the
    radix chunk counts, digit totals and starts, then the run scan's."""
    var c = max(cols, 0)
    return (4 * c * n + c * radix_chunks(n, chunk_rows) * RBINS + 2 * c * RBINS
            + unique_runs_scratch_words(n, cols))


def _runs_device[FROM_KEYS: Bool](ctx: DeviceContext, mut df: DeviceBuffer[DType.float32],
                                  mut dw: DeviceBuffer[DType.uint32], cols: Int, src: Int, isrc: Int, n: Int, d: Int,
                                  U: Int, CNT: Int, CODES: Int, rc_at: Int) raises:
    var f = df.unsafe_ptr()
    var w = dw.unsafe_ptr()
    var chn = radix_chunks(n, RUN_ROWS)
    var units = cols * chn
    var ro_at = rc_at + units
    ctx.enqueue_function[runs_count_kernel[FROM_KEYS]](
        f, w, Int32(src), Int32(n), Int32(chn), Int32(rc_at), Int32(units), grid_dim=_rb(units), block_dim=RBS,
    )
    ctx.enqueue_function[runs_offset_kernel](
        f, w, Int32(rc_at), Int32(ro_at), Int32(chn), Int32(CNT), Int32(cols), grid_dim=_rb(cols), block_dim=RBS,
    )
    ctx.enqueue_function[runs_write_kernel[FROM_KEYS]](
        f, w, Int32(src), Int32(isrc), Int32(n), Int32(d), Int32(chn), Int32(ro_at), Int32(U), Int32(CODES),
        Int32(units), grid_dim=_rb(units), block_dim=RBS,
    )


def unique_runs_device(ctx: DeviceContext, mut df: DeviceBuffer[DType.float32], mut dw: DeviceBuffer[DType.uint32],
                       cols: Int, S: Int, n: Int, U: Int, CNT: Int) raises:
    """`unique_cols` (q = [S, n, d, U, CNT], one unit a column) as a chunked
    run scan of all columns at once; scratch `unique_runs_scratch_words`."""
    if n <= 0 or cols <= 0:
        return
    if unique_runs_scratch_words(n, cols) > 2 ** 31 - 1 or cols * n > 2 ** 31 - 1:
        raise Error("x_prep: unique_cols too large for the device run scan")
    _runs_device[False](ctx, df, dw, cols, S, -1, n, 0, U, CNT, -1, 0)


def dict_inverse_device(ctx: DeviceContext, mut df: DeviceBuffer[DType.float32], mut dw: DeviceBuffer[DType.uint32],
                        cols: Int, X: Int, n: Int, d: Int, U: Int, CNT: Int, CODES: Int, chunk_rows: Int) raises:
    """`unique_inverse` (q = [S, n, d, U, CNT, X, CODES]) without S: the
    index radix sort of X's columns, then the run scan with the codes."""
    comptime assert RPASSES % 2 == 0, "the radix pass count must be even"
    if n <= 0 or cols <= 0:
        return
    if dict_inverse_scratch_words(n, cols, chunk_rows) > 2 ** 31 - 1:
        raise Error("x_prep: unique_inverse too large for the device sort")
    var f = df.unsafe_ptr()
    var w = dw.unsafe_ptr()
    var tot = cols * n
    var chn = radix_chunks(n, chunk_rows)
    var ka = 0
    var kb = tot
    var ia = 2 * tot
    var ib = 3 * tot
    var h_at = 4 * tot
    var tot_at = h_at + cols * chn * RBINS
    var st_at = tot_at + cols * RBINS
    var rc_at = st_at + cols * RBINS
    ctx.enqueue_function[dict_load_kernel](
        f, w, Int32(ka), Int32(ia), Int32(X), Int32(n), Int32(d), Int32(tot),
        grid_dim=(tot + RWS - 1) // RWS, block_dim=RWS,
    )
    var src = ka
    var dst = kb
    var isrc = ia
    var idst = ib
    var units = cols * chn
    for ps in range(RPASSES):
        var shift = ps * RBITS
        ctx.enqueue_function[radix_hist_kernel](
            w, Int32(src), Int32(h_at), Int32(n), Int32(chn), Int32(shift), Int32(units),
            grid_dim=_rb(units), block_dim=RBS,
        )
        ctx.enqueue_function[radix_hsum_kernel](
            w, Int32(h_at), Int32(tot_at), Int32(chn), Int32(cols * RBINS), grid_dim=_rb(cols * RBINS), block_dim=RBS,
        )
        ctx.enqueue_function[radix_hstart_kernel](
            w, Int32(tot_at), Int32(st_at), Int32(cols), grid_dim=_rb(cols), block_dim=RBS,
        )
        ctx.enqueue_function[dict_scatter_kernel](
            w, Int32(src), Int32(dst), Int32(isrc), Int32(idst), Int32(h_at), Int32(st_at), Int32(n), Int32(chn),
            Int32(shift), Int32(units), grid_dim=_rb(units), block_dim=RBS,
        )
        var tmp = src
        src = dst
        dst = tmp
        var itmp = isrc
        isrc = idst
        idst = itmp
    # RPASSES is even: the sorted keys and their rows are back in the first blocks
    _runs_device[True](ctx, df, dw, cols, src, isrc, n, d, U, CNT, CODES, rc_at)
