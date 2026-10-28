# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Typed host-side helpers of the trees entry points (`providers/python/xtrees_api_pyapi.mojo`):
the host column's cv folds, the scatter refusal, KernelExplainer's coalition
schedule, the class priors and DART's per-tree bookkeeping. Every argument is
a typed pointer or scalar; the PythonObject unpacking stays in the shell."""

from std.math import ceil
from std.memory import bitcast

from core.hostptr import i32_ptr
from checks.numerics import identical_log64

comptime _TWO53 = Float64(9007199254740992.0)


@always_inline
def _dart_thr(v: Float64) -> Int64:
    """ceil(v * 2^53): `u < v` for a counter draw u = m / 2^53 is exactly
    `m < ceil(v * 2^53)` (0 for v <= 0 or NaN, 2^53 for v >= 1)."""
    if not v > 0.0:
        return 0
    if v >= 1.0:
        return Int64(1) << 53
    return Int64(Int(ceil(v * _TWO53)))



def host_folds(
    codes: MutPointer[Int32, MutUntrackedOrigin], n: Int, k: Int, n_splits: Int,
    rows: MutPointer[Int32, MutUntrackedOrigin], counts: MutPointer[Int32, MutUntrackedOrigin],
) -> Int:
    """The host column's `device_folds` (lane cgr4-py-compute: the same law
    and the same output layout, so a CPU-only install no longer runs the
    Python fold routines): fold id per row (KFold when k == 0, else
    StratifiedKFold over first-seen classes), then per fold the rows
    outside it ascending, then inside it ascending; counts[f] = fold f's
    size, counts[n_splits] = the status."""
    var fold = List[Int](length=n, fill=0)
    var sizes = List[Int](length=n_splits, fill=0)
    if k == 0:
        var off = 0
        for f in range(n_splits):
            var size = n // n_splits + (1 if f < n % n_splits else 0)
            for r in range(off, off + size):
                fold[r] = f
            sizes[f] = size
            off += size
    else:
        var cnt = List[Int](length=k, fill=0)
        var order = List[Int]()
        for r in range(n):
            var c = Int(codes[r])
            if c < 0 or c >= k:
                counts[n_splits] = 2
                return 2
            if cnt[c] == 0:
                order.append(c)
            cnt[c] += 1
        var mx = 0
        for c in range(k):
            mx = max(mx, cnt[c])
        if n_splits > mx:
            counts[n_splits] = 1
            return 1
        # alloc[c * n_splits + f]: positions p of class c's block [s, s + cnt)
        # of the sorted labels with p mod n_splits == f
        var alloc = List[Int](length=k * n_splits, fill=0)
        var start = 0
        for j in range(len(order)):
            var c = order[j]
            var e = start + cnt[c]
            for f in range(n_splits):
                alloc[c * n_splits + f] = (e - 1 - f) // n_splits - (start - 1 - f) // n_splits
            start = e
        var cur = List[Int](length=k, fill=0)
        for r in range(n):
            var c = Int(codes[r])
            var f = cur[c]
            while alloc[c * n_splits + f] == 0:
                f += 1
            alloc[c * n_splits + f] -= 1
            cur[c] = f
            fold[r] = f
            sizes[f] += 1
    for f in range(n_splits):
        var base = f * n
        var a = 0
        var b = n - sizes[f]
        for r in range(n):
            if fold[r] == f:
                rows[base + b] = Int32(r)
                b += 1
            else:
                rows[base + a] = Int32(r)
                a += 1
        counts[f] = Int32(sizes[f])
    counts[n_splits] = 0
    return 0


def scatter_rows_check(rp: MutPointer[Int32, MutUntrackedOrigin], m: Int, n_dst: Int) raises:
    """The host column's scatter refusal: every target row in [0, n_dst)."""
    for r in range(m):
        var i = Int(rp[unsafe_offset=r])
        if i < 0 or i >= n_dst:
            raise Error("x_trees_scatter: row out of range")


def _kshap_binom(M: Int, r: Int) -> Float64:
    """C(M, r) in float64 by the multiplicative recurrence; every step is an
    exact integer while C(M, r) * M < 2^53, which holds for every size the
    schedule enumerates in full (nsub <= nsamples)."""
    var c = Float64(1)
    for i in range(r):
        c = c * Float64(M - i) / Float64(i + 1)
    return c


def kshap_schedule(
    M: Int, nsamples: Int, off: MutPointer[Int64, MutUntrackedOrigin],
    sw: MutPointer[Float64, MutUntrackedOrigin], cdf: MutPointer[Float64, MutUntrackedOrigin],
    res: MutPointer[Int64, MutUntrackedOrigin],
) -> Int:
    """KernelExplainer's coalition schedule over subset sizes (M > 1):
    size offsets, size weights and the sampling cdf in `off`, `sw`, `cdf`;
    res = [m, nfixed, nfull, npaired, L, wrand_bits]. Returns nfull. The
    sums run in index order (one fixed fold, the same on every column)."""
    var nss = M // 2
    var npaired = (M - 1) // 2
    var wv = List[Float64](length=nss, fill=0.0)
    var tot = Float64(0)
    for i in range(1, nss + 1):
        var w = Float64(M - 1) / Float64(i * (M - i))
        if i - 1 < npaired:
            w *= 2.0
        wv[i - 1] = w
        tot += w
    for i in range(nss):
        wv[i] = wv[i] / tot
    var rem = wv.copy()
    var nfull = 0
    var left = nsamples
    var acc = 0
    off[0] = 0
    for size in range(1, nss + 1):
        var b = _kshap_binom(M, size)
        var nsub = b * (2.0 if size <= npaired else 1.0)
        if Float64(left) * rem[size - 1] / nsub >= 1.0 - 1e-8:
            var nsub_i = Int(nsub)
            nfull += 1
            left -= nsub_i
            if rem[size - 1] < 1.0:
                var r0 = rem[size - 1]
                for j in range(nss):
                    rem[j] = rem[j] / (1.0 - r0)
            var w = wv[size - 1] / b
            if size <= npaired:
                w /= 2.0
            acc += nsub_i
            off[size] = Int64(acc)
            sw[size - 1] = w
        else:
            break
    var nfixed = acc
    var samples_left = nsamples - nfixed
    var L = 0
    var wrand = Float64(0)
    if nfull != nss and samples_left > 0:
        var t = Float64(0)
        for i in range(nfull, nss):
            t += wv[i] / 2.0 if i < npaired else wv[i]
        var run = Float64(0)
        for i in range(nfull, nss):
            run += (wv[i] / 2.0 if i < npaired else wv[i]) / t
            cdf[L] = run
            L += 1
        var tail = Float64(0)
        for i in range(nfull, nss):
            tail += wv[i]
        wrand = tail / Float64(samples_left)
    else:
        samples_left = 0
    res[0] = Int64(nfixed + samples_left)
    res[1] = Int64(nfixed)
    res[2] = Int64(nfull)
    res[3] = Int64(npaired)
    res[4] = Int64(L)
    res[5] = bitcast[DType.int64](wrand)
    return nfull


def class_priors(
    cp: MutPointer[Int32, MutUntrackedOrigin], op: MutPointer[Float64, MutUntrackedOrigin], n: Int, k: Int
):
    """priors[c] = log(max(1e-15, counts[c] / n)), the pinned binary64 log."""
    for c in range(k):  # small-loop(k: class count): one prior per class
        var q = Float64(Int(cp[c])) / Float64(n)
        op[c] = identical_log64(q if q > 1e-15 else 1e-15)


def dart_thresholds(
    wp: MutPointer[Float64, MutUntrackedOrigin], cp: MutPointer[Float64, MutUntrackedOrigin],
    c32: MutPointer[Float32, MutUntrackedOrigin], tp: MutPointer[Int64, MutUntrackedOrigin],
    sum_w: Float64, t: Int, k: Int, uniform: Bool, max_drop: Int, rate: Float64,
):
    """One DART round's drop thresholds (int64, t) and float32 coefficients
    (t * k) from the tree weights, the coefficients and the weight sum."""
    var r = rate
    if not uniform:
        var inv_avg = Float64(t) / sum_w if sum_w > 0 else 0.0
        if max_drop > 0 and sum_w > 0:
            var cap = Float64(max_drop) * inv_avg / sum_w
            if cap < r:
                r = cap
        for i in range(t):  # small-loop(t: trees so far): one drop threshold per tree
            tp[i] = _dart_thr(r * wp[i] * inv_avg)
    else:
        if max_drop > 0:
            var cap = Float64(max_drop) / Float64(t)
            if cap < r:
                r = cap
        var u = _dart_thr(r)
        for i in range(t):  # small-loop(t: trees so far): one drop threshold per tree
            tp[i] = u
    for j in range(t * k):  # small-loop(t: trees so far, times k classes): one float32 coefficient per member
        c32[j] = Float32(cp[j])


def dart_rescale(
    wp: MutPointer[Float64, MutUntrackedOrigin], cp: MutPointer[Float64, MutUntrackedOrigin],
    sp: MutPointer[Float64, MutUntrackedOrigin], flags_addr: Int, t: Int, k: Int, uniform: Bool, xgb: Bool,
    l: Float64,
) raises -> Tuple[Int, Float64, Float64]:
    """The round's drop count from the step's flags (int32, t), then the
    shrink, the dropped trees' rescale (ascending tree order, the weight
    sum sp[0] updated per dropped tree) and the new round's weight and
    coefficients. Returns (k_d, shrink, factor)."""
    var nd = 0
    if t > 0:
        var fp = i32_ptr(flags_addr)
        for i in range(t):  # small-loop(t: trees so far): counts the dropped trees
            if fp[i] != 0:
                nd += 1
    var kd = Float64(nd)
    var shrink: Float64
    var factor: Float64
    var wdiv: Float64
    if not xgb:
        shrink = l / (1.0 + kd)
        factor = kd / (kd + 1.0)
        wdiv = 1.0 / (kd + 1.0)
    else:
        shrink = l if nd == 0 else l / (l + kd)
        factor = kd / (kd + l)
        wdiv = 1.0 / (kd + l)
    var sum_w = sp[0]
    if nd > 0:
        var fp = i32_ptr(flags_addr)
        for i in range(t):  # small-loop(t: trees so far): rescales each dropped tree in ascending order
            if fp[i] == 0:
                continue
            for c in range(k):  # small-loop(k: class count): one member per class
                cp[i * k + c] *= factor
            if not uniform:
                sum_w -= wp[i] * wdiv
                wp[i] *= factor
    wp[t] = shrink
    for c in range(k):  # small-loop(k: class count): the new round's member coefficients
        cp[t * k + c] = shrink
    sum_w += shrink
    sp[0] = sum_w
    return (nd, shrink, factor)
