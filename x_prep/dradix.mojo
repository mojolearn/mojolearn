# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The radix form of the `sort_cols` stage (lane prep-apple3, 2026-09-28).
FAST on the Apple GPU, and (K1, lane ml-cluster-nbrs 2026-10-04) IDENTICAL on
every vendor: x_prep/device.mojo gates every call on RADIX_SORT. The sort has
one answer, so IDENTICAL keeps its bits (the host heapsort is unchanged).
`-D MOJOLEARN_IDN_XPREP_RADIX_OFF` (or MOJOLEARN_IDN_ALL_OFF) restores the
bitonic sort under IDENTICAL. RBS = 64 assumes no wave width.

A sort has ONE answer under a total order on the words (x_prep/dsort.mojo),
so this sort writes the words the bitonic sort and the host heapsort write,
bit for bit. The order is `word_order` (x_prep/common.mojo): `key` first,
the raw bits second. `radix_key` is a bijection of the 32-bit words that is
monotone in that order (the non-NaN keys moved down by 0x007FFFFF, which
leaves the 2^24 - 2 NaN words their own slots above +inf, in raw bit order),
so an unsigned sort of the keys is the sort of the words and `radix_word`
gives each word back.

The sort is a least-significant-digit radix sort, RBITS bits per pass, each
pass a stable counting sort by CHUNKS of consecutive positions (the scheme
of TargetEncoder's parallel buckets, x_prep/target.mojo te_hist ..
te_hscatter): every (column, chunk) counts its digits, every (column, digit)
turns its chunk counts into the keys of that digit before each chunk, every
column lays out its digit starts, and every (column, chunk) places its keys
in position order. No arithmetic on the values, only moves of words.

Measured on the M4 (request 1790626651574): the bitonic sort of 16 columns
of 1M rows is 0.44 to 0.48 s (41 passes over the block), the largest FAST
phase of every estimator that sorts.
"""
from std.gpu import block_idx, block_dim, thread_idx
from std.memory import bitcast
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import ftz, GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from x_prep.common import FP, canon

comptime RUP = MutPointer[UInt32, MutAnyOrigin]
#: K1 (IDENTICAL, every vendor): the radix sort of sort_cols, same words as the bitonic sort
comptime IDN_XPREP_RADIX = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not (
    is_defined["MOJOLEARN_IDN_XPREP_RADIX_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
#: FAST on Apple, IDENTICAL everywhere unless IDN_XPREP_RADIX is off
comptime RADIX_SORT = (GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL and has_apple_gpu_accelerator()) or IDN_XPREP_RADIX
comptime RBITS = 8
comptime RBINS = 256
comptime RPASSES = 4
#: threads per block of every launch here
comptime RBS = 64
#: threads per block of the load and the store (one thread per word)
comptime RWS = 256
#: below this many rows the bitonic sort runs (3 launches at one tile)
comptime RADIX_MIN_ROWS = 8192


@always_inline
def radix_key(b: UInt32) -> UInt32:
    """Monotone in `word_order`, one to one: -inf -> 0, -0.0 -> 0x7F800000,
    +0.0 -> 0x7F800001, +inf -> 0xFF000001, then the NaN words with the sign
    clear in bit order (0xFF000002 ..), then the NaN words with it set."""
    var mag = b & UInt32(0x7FFFFFFF)
    var neg = (b & UInt32(0x80000000)) != UInt32(0)
    if mag > UInt32(0x7F800000):
        if neg:
            return b
        return b + UInt32(0x7F800001)
    if neg:
        return (~b) - UInt32(0x007FFFFF)
    return (b | UInt32(0x80000000)) - UInt32(0x007FFFFF)


@always_inline
def radix_word(u: UInt32) -> UInt32:
    """The word whose `radix_key` is u."""
    if u >= UInt32(0xFF800001):
        return u
    if u >= UInt32(0xFF000002):
        return u - UInt32(0x7F800001)
    var k = u + UInt32(0x007FFFFF)
    if k >= UInt32(0x80000000):
        return k & UInt32(0x7FFFFFFF)
    return ~k


def radix_load_kernel(f: FP, w: RUP, X: Int32, n: Int32, d: Int32, cn: Int32, total: Int32):
    """t = c * n + i: w[t] = the key of the flushed (and with cn, canonical)
    word of X[i, c] (the unit's own load: ftz, then canon)."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= Int(total):
        return
    var nn = Int(n)
    var c = t // nn
    var i = t - c * nn
    var v = ftz(f.unsafe_load(Int(X) + i * Int(d) + c))
    if cn != Int32(0):
        v = canon(v)
    w[t] = radix_key(bitcast[DType.uint32](v))


def radix_store_kernel(f: FP, w: RUP, S: Int32, total: Int32):
    """t = c * n + i: S[t] = the word of key w[t], moved as bits."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= Int(total):
        return
    var fu = f.bitcast[UInt32]()
    fu[Int(S) + t] = radix_word(w[t])


def radix_hist_kernel(w: RUP, src: Int32, H: Int32, n: Int32, ch_n: Int32, shift: Int32, total: Int32):
    """t = c * CH + ch: w[H + t*RBINS + b] = how many keys of chunk ch of
    column c have digit b."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= Int(total):
        return
    var nn = Int(n)
    var chn = Int(ch_n)
    var c = t // chn
    var ch = t - c * chn
    var hb = Int(H) + t * RBINS
    for b in range(RBINS):
        w[hb + b] = UInt32(0)
    var cs = (nn + chn - 1) // chn
    var lo = min(ch * cs, nn)
    var hi = min(lo + cs, nn)
    var base = Int(src) + c * nn
    var sh = UInt32(Int(shift))
    for i in range(lo, hi):
        var dg = Int((w[base + i] >> sh) & UInt32(0xFF))
        w[hb + dg] = w[hb + dg] + UInt32(1)


def radix_hsum_kernel(w: RUP, H: Int32, TOT: Int32, ch_n: Int32, total: Int32):
    """t = c * RBINS + b: digit b's chunk counts become the keys of b before
    each chunk; w[TOT + t] = b's total in column c."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= Int(total):
        return
    var chn = Int(ch_n)
    var c = t // RBINS
    var b = t - c * RBINS
    var run = UInt32(0)
    for ch in range(chn):
        var at = Int(H) + (c * chn + ch) * RBINS + b
        var v = w[at]
        w[at] = run
        run = run + v
    w[Int(TOT) + t] = run


def radix_hstart_kernel(w: RUP, TOT: Int32, START: Int32, total: Int32):
    """t = column c: w[START + c*RBINS + b] = the keys of column c with a
    digit below b."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= Int(total):
        return
    var acc = UInt32(0)
    for b in range(RBINS):
        var v = w[Int(TOT) + t * RBINS + b]
        w[Int(START) + t * RBINS + b] = acc
        acc = acc + v


def radix_scatter_kernel(w: RUP, src: Int32, dst: Int32, H: Int32, START: Int32, n: Int32, ch_n: Int32,
                         shift: Int32, total: Int32):
    """t = c * CH + ch: the keys of chunk ch in position order, each at its
    digit's start plus the keys of that digit before it (a stable counting
    sort: equal digits keep their order)."""
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
    var out = Int(dst) + c * nn
    var sh = UInt32(Int(shift))
    for i in range(lo, hi):
        var v = w[base + i]
        var dg = Int((v >> sh) & UInt32(0xFF))
        var k = w[hb + dg]
        w[out + Int(k)] = v
        w[hb + dg] = k + UInt32(1)


def _rblocks(total: Int) -> Int:
    return (total + RBS - 1) // RBS


def _wblocks(total: Int) -> Int:
    return (total + RWS - 1) // RWS


def radix_chunks(n: Int, chunk_rows: Int) -> Int:
    """Chunks per column: consecutive ranges of about chunk_rows positions."""
    var cr = max(chunk_rows, 1)
    return max(1, (n + cr - 1) // cr)


def radix_scratch_words(n: Int, cols: Int, chunk_rows: Int) -> Int:
    """The scratch words `radix_sort_cols_device` needs: two key blocks, the
    chunk counts, the digit totals and the digit starts."""
    var c = max(cols, 0)
    return 2 * c * n + c * radix_chunks(n, chunk_rows) * RBINS + 2 * c * RBINS


def radix_sort_cols_device(ctx: DeviceContext, mut df: DeviceBuffer[DType.float32],
                           mut dw: DeviceBuffer[DType.uint32], cols: Int, X: Int, n: Int, d: Int, S: Int,
                           cn: Int, chunk_rows: Int) raises:
    """Enqueue the sort of columns 0 .. cols-1 of X[n, d] into S[c*n : c*n+n]
    (`sort_cols_device`'s contract) through the scratch `w` of at least
    radix_scratch_words(n, cols, chunk_rows) words."""
    if n <= 0 or cols <= 0:
        return
    var f = df.unsafe_ptr()
    var w = dw.unsafe_ptr()
    var tot = cols * n
    var chn = radix_chunks(n, chunk_rows)
    if radix_scratch_words(n, cols, chunk_rows) > 2 ** 31 - 1:
        raise Error("x_prep: sort_cols too large for the device sort")
    var a_at = 0
    var b_at = tot
    var h_at = 2 * tot
    var tot_at = h_at + cols * chn * RBINS
    var st_at = tot_at + cols * RBINS
    ctx.enqueue_function[radix_load_kernel](
        f, w, Int32(X), Int32(n), Int32(d), Int32(cn), Int32(tot), grid_dim=_wblocks(tot), block_dim=RWS,
    )
    var src = a_at
    var dst = b_at
    var units = cols * chn
    for ps in range(RPASSES):
        var shift = ps * RBITS
        ctx.enqueue_function[radix_hist_kernel](
            w, Int32(src), Int32(h_at), Int32(n), Int32(chn), Int32(shift), Int32(units),
            grid_dim=_rblocks(units), block_dim=RBS,
        )
        ctx.enqueue_function[radix_hsum_kernel](
            w, Int32(h_at), Int32(tot_at), Int32(chn), Int32(cols * RBINS),
            grid_dim=_rblocks(cols * RBINS), block_dim=RBS,
        )
        ctx.enqueue_function[radix_hstart_kernel](
            w, Int32(tot_at), Int32(st_at), Int32(cols), grid_dim=_rblocks(cols), block_dim=RBS,
        )
        ctx.enqueue_function[radix_scatter_kernel](
            w, Int32(src), Int32(dst), Int32(h_at), Int32(st_at), Int32(n), Int32(chn), Int32(shift),
            Int32(units), grid_dim=_rblocks(units), block_dim=RBS,
        )
        var tmp = src
        src = dst
        dst = tmp
    # RPASSES is even: the sorted keys are back in the first block
    ctx.enqueue_function[radix_store_kernel](
        f, w, Int32(S), Int32(tot), grid_dim=_wblocks(tot), block_dim=RWS,
    )
