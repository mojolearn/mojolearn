# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Parallel forms of the units whose answer is a COUNT or a MOVE of words
(lane prep-apple3, 2026-09-28). FAST on the Apple GPU only: x_prep/device.mojo
gates every launch on FAST_EXACT, so the IDENTICAL binding and the other
vendors never compile one.

Each unit below walks n rows on ONE thread per column (or per class and
category), so a stage of 9 units keeps 9 GPU threads busy. An integer count
is the same integer in any order, and the distinct words of a sorted column
are the same words whoever finds them, so these kernels write the words the
units write, bit for bit (no float arithmetic on the data):

  count_neg    a threadgroup per column, a tree of integer counts
  unique_cols  by chunks of consecutive positions: the run starts of every
               chunk counted, the counts turned into offsets, the run starts
               written at their offsets
  cat_counts   (unweighted) one pass over the rows by chunks into a
               (class, category) table per column and chunk, then the sum
               over the chunks; the unit tests every row once per class and
               category

`ii_gram_sym_fast_kernel` is x_prep/fastred.mojo's `ii_gram_fast_kernel` on
the pairs a <= b only, each sum written to G[a, b] and G[b, a]: the product
of two floats does not depend on their order, so the two sums the full
kernel folds are the same word and the FAST bits do not move.

`te_global_fast_kernel` is a FAST fold (the bits may change): the target's
mean and variance over a fold's rows by a threadgroup tree, as
x_prep/fastred.mojo folds the columns. It is held to the paired quality rule
(bench/x_prep_quality.py, target-encoder).
"""
from std.gpu import block_idx, block_dim, thread_idx
from std.memory import stack_allocation
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from x_prep.common import FP, IP, p, ld, raw, st, key, is_nan
from x_prep.prims import add, sub, mul, div, logf
from x_prep.transform import PT_STATE, pt_finish, log1pf

comptime XUP = MutPointer[UInt32, MutAnyOrigin]
#: FAST on Apple only
comptime FAST_EXACT = GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL and has_apple_gpu_accelerator()
#: threads per group of the threadgroup kernels
comptime XTG = 256
#: threads per block of the chunk kernels
comptime XBS = 64
#: positions per chunk
comptime XCHUNK = 4096
#: below this many rows the units run
comptime EXACT_MIN_ROWS = 65536
#: the largest table (words) `cat_counts` keeps
comptime CAT_TABLE_MAX = 1 << 26
#: the most groups a column's FAST fold is split into (the combining tree has XTG leaves)
comptime FOLD_GROUPS_MAX = 256


def fold_scratch_words(cols: Int, groups: Int) -> Int:
    """The scratch words of a FAST fold by groups: three sums and one sum of
    squared deviations per (column, group), three words per column."""
    return max(cols, 0) * (groups * 4 + 3)


def exact_chunks(n: Int) -> Int:
    return max(1, (n + XCHUNK - 1) // XCHUNK)


def cat_table_words(n: Int, d: Int, K: Int, cmax: Int) -> Int:
    """The scratch words of `cat_counts`' tables, or 0 when they are too
    large (the units run)."""
    var words = d * exact_chunks(n) * K * cmax
    if K <= 0 or cmax <= 0 or d <= 0 or words > CAT_TABLE_MAX:
        return 0
    return words


def count_neg_fast_kernel(f: FP, q: IP):
    """`count_neg_unit` for column block_idx.x (q = [CODES, n, d, OUT]):
    every thread counts a strided share of the rows, a tree adds the
    counts."""
    var c = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var n = p(q, 1)
    var d = p(q, 2)
    var C = p(q, 0)
    var sh = stack_allocation[XTG, Int32, address_space = AddressSpace.SHARED]()
    var k = 0
    for i in range(tid, n, XTG):
        if ld(f, C + i * d + c) < Float32(0):
            k += 1
    sh[tid] = Int32(k)
    barrier()
    var w = XTG // 2
    while w >= 1:
        if tid < w:
            sh[tid] = sh[tid] + sh[tid + w]
        barrier()
        w //= 2
    if tid == 0:
        st(f, p(q, 3) + c, Float32(Int(sh[0])))


def uniq_count_kernel(f: FP, q: IP, w: XUP, ch_n: Int32, total: Int32):
    """q = [S, n, d, U, CNT]; t = c * CH + ch: w[t] = the positions of chunk
    ch of sorted column c that start a run of equal keys."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= Int(total):
        return
    var n = p(q, 1)
    var chn = Int(ch_n)
    var c = t // chn
    var ch = t - c * chn
    var cs = (n + chn - 1) // chn
    var lo = min(ch * cs, n)
    var hi = min(lo + cs, n)
    var Sc = p(q, 0) + c * n
    var k = 0
    var last = UInt32(0)
    if lo > 0:
        last = key(raw(f, Sc + lo - 1))
    for i in range(lo, hi):
        var kv = key(raw(f, Sc + i))
        if i == 0 or kv != last:
            k += 1
        last = kv
    w[t] = UInt32(k)


def uniq_prefix_kernel(f: FP, q: IP, w: XUP, ch_n: Int32, total: Int32):
    """t = column c: w[c * CH + ch] becomes the run starts before chunk ch;
    CNT[c] = the column's distinct count."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= Int(total):
        return
    var chn = Int(ch_n)
    var run = 0
    for ch in range(chn):
        var v = Int(w[t * chn + ch])
        w[t * chn + ch] = UInt32(run)
        run += v
    st(f, p(q, 4) + t, Float32(run))


def uniq_write_kernel(f: FP, q: IP, w: XUP, ch_n: Int32, total: Int32):
    """t = c * CH + ch: the first word of every run that starts in chunk ch,
    at U[c*n + the run starts before it]."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= Int(total):
        return
    var n = p(q, 1)
    var chn = Int(ch_n)
    var c = t // chn
    var ch = t - c * chn
    var cs = (n + chn - 1) // chn
    var lo = min(ch * cs, n)
    var hi = min(lo + cs, n)
    var Sc = p(q, 0) + c * n
    var U = p(q, 3) + c * n
    var k = Int(w[t])
    var last = UInt32(0)
    if lo > 0:
        last = key(raw(f, Sc + lo - 1))
    for i in range(lo, hi):
        var v = raw(f, Sc + i)
        var kv = key(v)
        if i == 0 or kv != last:
            f.unsafe_store(U + k, v)
            k += 1
        last = kv


def cat_hist_kernel(f: FP, q: IP, w: XUP, ch_n: Int32, total: Int32):
    """q = [X, n, d, Y, K, NCAT, CMAX, W, OUT]; t = j * CH + ch:
    w[t*K*CMAX + k*CMAX + v] = the rows of chunk ch of class k whose feature
    j equals v (the unit's own tests: Int of the flushed words)."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= Int(total):
        return
    var n = p(q, 1)
    var d = p(q, 2)
    var K = p(q, 4)
    var cmax = p(q, 6)
    var chn = Int(ch_n)
    var j = t // chn
    var ch = t - j * chn
    var bins = K * cmax
    var hb = t * bins
    for b in range(bins):
        w[hb + b] = UInt32(0)
    var cs = (n + chn - 1) // chn
    var lo = min(ch * cs, n)
    var hi = min(lo + cs, n)
    var Y = p(q, 3)
    var X = p(q, 0) + j
    for i in range(lo, hi):
        var k = Int(ld(f, Y + i))
        var v = Int(ld(f, X + i * d))
        if k >= 0 and k < K and v >= 0 and v < cmax:
            var at = hb + k * cmax + v
            w[at] = w[at] + UInt32(1)


def cat_sum_kernel(f: FP, q: IP, w: XUP, ch_n: Int32, total: Int32):
    """t = (j*K + k)*CMAX + v: OUT[t] = the chunk counts added (slots
    v >= NCAT[j] are left as they are, as the unit leaves them)."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= Int(total):
        return
    var K = p(q, 4)
    var cmax = p(q, 6)
    var v = t % cmax
    var jk = t // cmax
    var k = jk % K
    var j = jk // K
    if v >= Int(ld(f, p(q, 5) + j)):
        return
    var chn = Int(ch_n)
    var bins = K * cmax
    var m = 0
    for ch in range(chn):
        m += Int(w[(j * chn + ch) * bins + k * cmax + v])
    st(f, p(q, 8) + t, Float32(m))


def te_global_fast_kernel(f: FP, q: IP):
    """`te_global_unit` for unit block_idx.x (q = [Y, n, T, FOLD, META];
    t = fi*T + tt): the count and the sum of target column tt over the rows
    outside fold fi by a tree, then the squared deviations by a tree."""
    var t = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var n = p(q, 1)
    var T = p(q, 2)
    var Y = p(q, 0)
    var FO = p(q, 3)
    var fi = t // T
    var tt = t % T
    var sh_s = stack_allocation[XTG, Float32, address_space = AddressSpace.SHARED]()
    var sh_c = stack_allocation[XTG, Int32, address_space = AddressSpace.SHARED]()
    var s = Float32(0)
    var cnt = 0
    for i in range(tid, n, XTG):
        if Int(ld(f, FO + i)) == fi:
            continue
        s = add(s, ld(f, Y + i * T + tt))
        cnt += 1
    sh_s[tid] = s
    sh_c[tid] = Int32(cnt)
    barrier()
    var w = XTG // 2
    while w >= 1:
        if tid < w:
            sh_s[tid] = add(sh_s[tid], sh_s[tid + w])
            sh_c[tid] = sh_c[tid] + sh_c[tid + w]
        barrier()
        w //= 2
    var total = Int(sh_c[0])
    var mean = Float32(0)
    if total > 0:
        mean = div(sh_s[0], Float32(total))
    barrier()
    var ss = Float32(0)
    if total > 0:
        for i in range(tid, n, XTG):
            if Int(ld(f, FO + i)) == fi:
                continue
            var e = sub(ld(f, Y + i * T + tt), mean)
            ss = add(ss, mul(e, e))
    sh_s[tid] = ss
    barrier()
    var w2 = XTG // 2
    while w2 >= 1:
        if tid < w2:
            sh_s[tid] = add(sh_s[tid], sh_s[tid + w2])
        barrier()
        w2 //= 2
    if tid == 0:
        var var_ = Float32(0)
        if total > 0:
            var_ = div(sh_s[0], Float32(total))
        st(f, p(q, 4) + 2 * t, mean)
        st(f, p(q, 4) + 2 * t + 1, var_)


def ii_gram_sym_fast_kernel(f: FP, q: IP):
    """`ii_gram_fast_kernel` (q = [X, n, d, MASK, j, MEANS, G, FLAG];
    t = block_idx.x = a*d + b) for a <= b, written to both halves."""
    var t = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    if f[p(q, 7)] != Float32(0):
        return
    var X = p(q, 0)
    var nn = p(q, 1)
    var dd = p(q, 2)
    var M = p(q, 3)
    var j = p(q, 4)
    var a = t // dd
    var b = t % dd
    if a > b:
        return
    var ma = f[p(q, 5) + a]
    var mb = f[p(q, 5) + b]
    var sh_s = stack_allocation[XTG, Float32, address_space = AddressSpace.SHARED]()
    var s = Float32(0)
    for i in range(tid, nn, XTG):
        if f[M + i * dd + j] != Float32(0):
            continue
        s = add(s, mul(sub(f[X + i * dd + a], ma), sub(f[X + i * dd + b], mb)))
    sh_s[tid] = s
    barrier()
    var w = XTG // 2
    while w >= 1:
        if tid < w:
            sh_s[tid] = add(sh_s[tid], sh_s[tid + w])
        barrier()
        w //= 2
    if tid == 0:
        f[p(q, 6) + t] = sh_s[0]
        f[p(q, 6) + b * dd + a] = sh_s[0]


# ---------------------------------------------------------------- FAST folds by groups
# x_prep/fastred.mojo folds a column with ONE threadgroup (TGR threads), so a
# stage of 16 columns keeps 4096 GPU threads busy (measured on the M3 Ultra,
# request 1790627886703: 5 ms a fold of 16M words, where a kernel with one
# thread per word takes 1.5 ms). Here a column is folded by G groups: group g
# takes the rows g*XTG + tid, then every G*XTG-th, a tree adds its threads,
# and a second launch adds the G group sums by a tree. A FAST fold: the bits
# may change (the tree has another shape), held to the paired quality rule.


@always_inline
def _pt_state(f: FP, q: IP, c: Int) -> Int:
    return p(q, 6) + c * PT_STATE


def pt_fold_sum_kernel(f: FP, q: IP, w: XUP, groups: Int32):
    """`pt_fold_fast_kernel`'s first pass for column c, group g (block
    c*G + g): w[(c*G + g)*3 ..] = the count, the sum of T and (K = 0) the sum
    of J over the group's rows (float words)."""
    var G = Int(groups)
    var blk = Int(block_idx.x)
    var c = blk // G
    var g = blk - c * G
    var tid = Int(thread_idx.x)
    var X = p(q, 0)
    var nn = p(q, 1)
    var dd = p(q, 2)
    var method = p(q, 3)
    var T = p(q, 4)
    var K = p(q, 5)
    var S = _pt_state(f, q, c)
    if f[S + 7] != Float32(0):
        return
    var first = K == 0
    var sh_s = stack_allocation[XTG, Float32, address_space = AddressSpace.SHARED]()
    var sh_c = stack_allocation[XTG, Float32, address_space = AddressSpace.SHARED]()
    var sh_j = stack_allocation[XTG, Float32, address_space = AddressSpace.SHARED]()
    var cnt = Float32(0)
    var sm = Float32(0)
    var sj = Float32(0)
    for i in range(g * XTG + tid, nn, G * XTG):
        var x = f[X + i * dd + c]
        if is_nan(x):
            continue
        cnt += 1
        sm = add(sm, f[T + c * nn + i])
        if first:
            if method == 1:
                sj = add(sj, logf(x))
            elif x >= Float32(0):
                sj = add(sj, log1pf(x))
            else:
                sj = sub(sj, log1pf(sub(Float32(0), x)))
    sh_s[tid] = sm
    sh_c[tid] = cnt
    sh_j[tid] = sj
    barrier()
    var h = XTG // 2
    while h >= 1:
        if tid < h:
            sh_s[tid] = add(sh_s[tid], sh_s[tid + h])
            sh_c[tid] = sh_c[tid] + sh_c[tid + h]
            sh_j[tid] = add(sh_j[tid], sh_j[tid + h])
        barrier()
        h //= 2
    if tid == 0:
        var pw = w.bitcast[Float32]()
        pw[blk * 3] = sh_c[0]
        pw[blk * 3 + 1] = sh_s[0]
        pw[blk * 3 + 2] = sh_j[0]


def pt_fold_mean_kernel(f: FP, q: IP, w: XUP, groups: Int32, cols: Int32):
    """Column c = block: the G group sums added by a tree; w[MEANS + 3c ..] =
    the count, the mean of T and the sum of J (MEANS = cols*G*4)."""
    var G = Int(groups)
    var c = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var S = _pt_state(f, q, c)
    if f[S + 7] != Float32(0):
        return
    var first = p(q, 5) == 0
    var pw = w.bitcast[Float32]()
    var sh_s = stack_allocation[XTG, Float32, address_space = AddressSpace.SHARED]()
    var sh_c = stack_allocation[XTG, Float32, address_space = AddressSpace.SHARED]()
    var sh_j = stack_allocation[XTG, Float32, address_space = AddressSpace.SHARED]()
    var cnt = Float32(0)
    var sm = Float32(0)
    var sj = Float32(0)
    if tid < G:
        cnt = pw[(c * G + tid) * 3]
        sm = pw[(c * G + tid) * 3 + 1]
        sj = pw[(c * G + tid) * 3 + 2]
    sh_s[tid] = sm
    sh_c[tid] = cnt
    sh_j[tid] = sj
    barrier()
    var h = XTG // 2
    while h >= 1:
        if tid < h:
            sh_s[tid] = add(sh_s[tid], sh_s[tid + h])
            sh_c[tid] = sh_c[tid] + sh_c[tid + h]
            sh_j[tid] = add(sh_j[tid], sh_j[tid + h])
        barrier()
        h //= 2
    if tid == 0:
        var total = sh_c[0]
        var mean = Float32(0)
        if total > 0:
            mean = div(sh_s[0], total)
        var M = Int(cols) * G * 4 + c * 3
        pw[M] = total
        pw[M + 1] = mean
        pw[M + 2] = sh_j[0] if first else f[S + 8]


def pt_fold_dev_kernel(f: FP, q: IP, w: XUP, groups: Int32, cols: Int32):
    """Column c, group g: w[cols*G*3 + c*G + g] = the group's squared
    deviations of T from the column's mean."""
    var G = Int(groups)
    var blk = Int(block_idx.x)
    var c = blk // G
    var g = blk - c * G
    var tid = Int(thread_idx.x)
    var nn = p(q, 1)
    var T = p(q, 4)
    var S = _pt_state(f, q, c)
    if f[S + 7] != Float32(0):
        return
    var pw = w.bitcast[Float32]()
    var M = Int(cols) * G * 4 + c * 3
    var total = pw[M]
    var mean = pw[M + 1]
    var sh_s = stack_allocation[XTG, Float32, address_space = AddressSpace.SHARED]()
    var ss = Float32(0)
    if total > 0:
        for i in range(g * XTG + tid, nn, G * XTG):
            var tv = f[T + c * nn + i]
            if is_nan(tv):
                continue
            var e = sub(tv, mean)
            ss = add(ss, mul(e, e))
    sh_s[tid] = ss
    barrier()
    var h = XTG // 2
    while h >= 1:
        if tid < h:
            sh_s[tid] = add(sh_s[tid], sh_s[tid + h])
        barrier()
        h //= 2
    if tid == 0:
        pw[Int(cols) * G * 3 + blk] = sh_s[0]


def pt_fold_end_kernel(f: FP, q: IP, w: XUP, groups: Int32, cols: Int32):
    """Column c = block: the G groups' squared deviations added by a tree,
    then `pt_finish` (the step of the search) on thread 0."""
    var G = Int(groups)
    var c = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var S = _pt_state(f, q, c)
    if f[S + 7] != Float32(0):
        return
    var first = p(q, 5) == 0
    var pw = w.bitcast[Float32]()
    var sh_s = stack_allocation[XTG, Float32, address_space = AddressSpace.SHARED]()
    var ss = Float32(0)
    if tid < G:
        ss = pw[Int(cols) * G * 3 + c * G + tid]
    sh_s[tid] = ss
    barrier()
    var h = XTG // 2
    while h >= 1:
        if tid < h:
            sh_s[tid] = add(sh_s[tid], sh_s[tid + h])
        barrier()
        h //= 2
    if tid == 0:
        var M = Int(cols) * G * 4 + c * 3
        var sjt = pw[M + 2]
        if first:
            f[S + 8] = sjt
        pt_finish(c, f, q, Int(pw[M]), sjt, sh_s[0])
