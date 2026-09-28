# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE CURVE EPILOGUE ON THE HOST (lane metrics-apple2, 2026-09-28).

DEVIATION 6106 puts what follows the O(n) device work in IEEE binary64 on
the host: for the ROC and precision-recall scores that is O(n) too (one
term per curve point), and in Python it made a Python float per point,
several times over (the largest cost left in roc_auc_score, the curves and
average_precision_score on the Apple board). These functions are the SAME
binary64 operations, in the same order, as the Python they stand in for in
python/mojolearn/_expansion_metrics.py (`_drop_collinear`, `_trapezoid`,
`_binary_auc`, `_binary_ap`, `roc_curve`), run over the Float32 curve words
in the caller's arena:

- + - / are correctly rounded binary64 operations, as in Python; every
  product is `pinned_mul_f64` (never fused into a neighboring add or
  subtract, whatever the build's contraction mode), as Python never fuses;
- the Float32 words widen exactly to binary64, as Python's float() does;
- `fsum` is CPython's `math.fsum` (Modules/mathmodule.c, Shewchuk's
  partials with the half-even fix-up), step for step, and returns the
  correctly rounded sum, which is unique, so it is `math.fsum`'s value;
- the collinear-drop rule compares the same binary64 differences.

A case the Python path handles by a warning or an exception (an empty
class, a zero denominator) is never sent here: the Python checks it first,
and any doubt raises so the caller falls back to its Python path.
"""
from std.memory import bitcast, memcpy
from checks.numerics import pinned_mul_f64
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count, host_predict_chunk
from x_metrics.common import FP, IP

#: lane metrics-apple3: the least rows x columns (row_sum_range) or rows
#: (expected_mi) before a host epilogue wakes the pool.
comptime EPI_TASK_WORK = 65536
#: The most partials one task hands back (nonoverlapping binary64 values:
#: at most about 40 can exist); a task that would need more reports it and
#: the caller takes the sequential walk.
comptime EPI_PARTIALS = 64
#: `encode_small_i64` answers labels that span fewer than this many values
comptime ENC_SPAN = 65536


@always_inline
def _w(a: Int, i: Int) -> Float64:
    """Float32 word i of the arena at address a, widened exactly."""
    return Float64(FP(unsafe_from_address=a).unsafe_load(i))


@always_inline
def _word(a: Int, i: Int) -> Int32:
    return bitcast[DType.int32](FP(unsafe_from_address=a).unsafe_load(i))


@no_inline
def fsum(vals: List[Float64]) -> Float64:
    """CPython's math.fsum over finite values (and the module's `_fsum`
    wrapper: a zero sum is +0.0)."""
    var p = List[Float64]()
    for k in range(len(vals)):
        var x = vals[k]
        var i = 0
        for j in range(len(p)):
            var y = p[j]
            if abs(x) < abs(y):
                var t = x
                x = y
                y = t
            var hi = x + y
            var yr = hi - x
            var lo = y - yr
            if lo != 0.0:
                p[i] = lo
                i += 1
            x = hi
        while len(p) > i:
            _ = p.pop()
        if x != 0.0:
            p.append(x)
    var n = len(p)
    var hi: Float64 = 0.0
    var lo: Float64 = 0.0
    if n > 0:
        n -= 1
        hi = p[n]
        while n > 0:
            var x = hi
            n -= 1
            var y = p[n]
            hi = x + y
            var yr = hi - x
            lo = y - yr
            if lo != 0.0:
                break
        if n > 0 and ((lo < 0.0 and p[n - 1] < 0.0) or (lo > 0.0 and p[n - 1] > 0.0)):
            var y = lo * 2.0
            var x = hi + y
            var yr = x - hi
            if y == yr:
                hi = x
    if hi == 0.0:
        return 0.0
    return hi


def fsum_partials(vals: List[Float64], mut p: List[Float64]):
    """`fsum`'s first loop (CPython's math.fsum partials), continued over
    the partials already in `p`: afterwards the exact sum of `p` is the
    exact sum of what it held plus every value of `vals` (lane
    metrics-apple3)."""
    for k in range(len(vals)):
        var x = vals[k]
        var i = 0
        for j in range(len(p)):
            var y = p[j]
            if abs(x) < abs(y):
                var t = x
                x = y
                y = t
            var hi = x + y
            var yr = hi - x
            var lo = y - yr
            if lo != 0.0:
                p[i] = lo
                i += 1
            x = hi
        while len(p) > i:
            _ = p.pop()
        if x != 0.0:
            p.append(x)


def _trapezoid(x: List[Float64], y: List[Float64], L: Int) -> Float64:
    """fsum of ((x[i] - x[i-1]) * (y[i] + y[i-1])) / 2 over the first L."""
    var terms = List[Float64](capacity=max(L - 1, 0))
    for i in range(1, L):
        terms.append(pinned_mul_f64(x[i] - x[i - 1], y[i] + y[i - 1]) / 2.0)
    return fsum(terms)


def kept(a: Int, fps: Int, tps: Int, keep: Int, c: Int, drop: Bool) -> List[Int]:
    """The curve points `_drop_collinear` keeps: all of them when not
    `drop` or c <= 2; the device's flags (`keep` >= 0, an unweighted curve);
    else the first, the last and every point where either binary64 step
    changes. keep == -2: the device already dropped them (a compacted
    curve), every point is kept."""
    var out = List[Int](capacity=c)
    if not drop or c <= 2 or keep == -2:
        for i in range(c):
            out.append(i)
        return out^
    if keep >= 0:
        for i in range(c):
            if _word(a, keep + i) != 0:
                out.append(i)
        return out^
    out.append(0)
    for i in range(1, c - 1):
        var f0 = _w(a, fps + i - 1)
        var f1 = _w(a, fps + i)
        var f2 = _w(a, fps + i + 1)
        var t0 = _w(a, tps + i - 1)
        var t1 = _w(a, tps + i)
        var t2 = _w(a, tps + i + 1)
        if (f2 - f1) != (f1 - f0) or (t2 - t1) != (t1 - t0):
            out.append(i)
    out.append(c - 1)
    return out^


def binary_auc(a: Int, fps: Int, tps: Int, keep: Int, c: Int, max_fpr: Float64) raises -> Float64:
    """`_binary_auc` for fps[c-1] > 0 and tps[c-1] > 0 (the caller checks);
    max_fpr < 0 means None (or 1)."""
    var F = _w(a, fps + c - 1)
    var T = _w(a, tps + c - 1)
    if not (F > 0.0 and T > 0.0):
        raise Error("x_metrics epilogue: an empty class goes the Python way")
    var ks = kept(a, fps, tps, keep, c, True)
    var m = len(ks)
    var fpr = List[Float64](capacity=m + 1)
    var tpr = List[Float64](capacity=m + 1)
    fpr.append(0.0)
    tpr.append(0.0)
    for j in range(m):
        fpr.append(_w(a, fps + ks[j]) / F)
        tpr.append(_w(a, tps + ks[j]) / T)
    if max_fpr < 0.0:
        return _trapezoid(fpr, tpr, m + 1)
    # bisect_right(fpr, max_fpr)
    var lo = 0
    var hi = m + 1
    while lo < hi:
        var mid = (lo + hi) // 2
        if max_fpr < fpr[mid]:
            hi = mid
        else:
            lo = mid + 1
    var stop = lo
    if stop < 1 or stop > m:
        raise Error("x_metrics epilogue: max_fpr outside the curve goes the Python way")
    var x0 = fpr[stop - 1]
    var x1 = fpr[stop]
    var y0 = tpr[stop - 1]
    var y1 = tpr[stop]
    var yi = y0
    if x1 != x0:
        yi = y0 + pinned_mul_f64(max_fpr - x0, y1 - y0) / (x1 - x0)
    fpr[stop] = max_fpr
    tpr[stop] = yi
    var part = _trapezoid(fpr, tpr, stop + 1)
    var min_area = pinned_mul_f64(pinned_mul_f64(0.5, max_fpr), max_fpr)
    return pinned_mul_f64(0.5, 1.0 + (part - min_area) / (max_fpr - min_area))


def binary_ap(a: Int, fps: Int, tps: Int, c: Int) raises -> Float64:
    """`_binary_ap` for tps[c-1] != 0 (the caller checks): max(0.0, fsum
    of (r - r_prev) * (t / (t + f))), r = t / T, r_prev of the first 0.0."""
    var T = _w(a, tps + c - 1)
    if not (T != 0.0):
        raise Error("x_metrics epilogue: no positives goes the Python way")
    var terms = List[Float64](capacity=c)
    var rp: Float64 = 0.0
    for i in range(c):
        var t = _w(a, tps + i)
        var d = t + _w(a, fps + i)
        if d == 0.0:
            raise Error("x_metrics epilogue: a zero denominator goes the Python way")
        var r = t / T
        terms.append(pinned_mul_f64(r - rp, t / d))
        rp = r
    var s = fsum(terms)
    if s > 0.0:
        return s
    return 0.0


def roc_arrays(a: Int, fps: Int, tps: Int, thr: Int, keep: Int, c: Int, drop: Bool,
               out_fpr: Int, out_tpr: Int, out_thr: Int) raises -> Int:
    """`roc_curve`'s three Float64 arrays for fps[c-1] > 0 and tps[c-1] > 0
    (the caller checks): the kept points after a leading (0, 0, +inf).
    Returns their length."""
    var F = _w(a, fps + c - 1)
    var T = _w(a, tps + c - 1)
    if not (F > 0.0 and T > 0.0):
        raise Error("x_metrics epilogue: an empty class goes the Python way")
    var ks = kept(a, fps, tps, keep, c, drop)
    var pf = MutPointer[Float64, MutAnyOrigin](unsafe_from_address=out_fpr)
    var pt = MutPointer[Float64, MutAnyOrigin](unsafe_from_address=out_tpr)
    var ph = MutPointer[Float64, MutAnyOrigin](unsafe_from_address=out_thr)
    pf[0] = 0.0
    pt[0] = 0.0
    ph[0] = bitcast[DType.float64](UInt64(0x7FF0000000000000))
    for j in range(len(ks)):
        pf[j + 1] = _w(a, fps + ks[j]) / F
        pt[j + 1] = _w(a, tps + ks[j]) / T
        ph[j + 1] = _w(a, thr + ks[j])
    return len(ks) + 1


# ---------------------------------------------------------------------------
# The expected mutual information (adjusted_mutual_info_score)
# ---------------------------------------------------------------------------

@always_inline
def _fm(a: Float64, b: Float64, c: Float64) -> Float64:
    from std.math import fma
    return fma(a, b, c)


@always_inline
def _mul(a: Float64, b: Float64) -> Float64:
    return pinned_mul_f64(a, b)


def _log_fraction(x_in: Float64, mut e: Int, mut frac: Float64) -> Float64:
    """packaging/portable_math/portable_math.c `log_fraction`, operation
    for operation (its products unfused, its fm sites fma)."""
    var xv = x_in
    var u = bitcast[DType.uint64](xv)
    e = 0
    if (u >> 52) == UInt64(0):
        xv = _mul(xv, 18014398509481984.0)
        u = bitcast[DType.uint64](xv)
        e = -54
    e += Int((u >> 52) & UInt64(0x7FF)) - 1022
    var m = bitcast[DType.float64]((u & UInt64(0x000FFFFFFFFFFFFF)) | UInt64(0x3FE0000000000000))
    var z: Float64
    var y: Float64
    var x: Float64
    if e > 2 or e < -2:
        if m < 0.70710678118654752440:
            e -= 1
            z = m - 0.5
            y = _fm(0.5, z, 0.5)
        else:
            z = m - 0.5
            z = z - 0.5
            y = _fm(0.5, m, 0.5)
        x = z / y
        z = _mul(x, x)
        var r = _fm(-7.89580278884799154124e-1, z, 1.63866645699558079767e1)
        r = _fm(r, z, -6.41409952958715622951e1)
        var q = z + -3.56722798256324312549e1
        q = _fm(q, z, 3.12093766372244180303e2)
        q = _fm(q, z, -7.69691943550460008604e2)
        y = _mul(x, _mul(z, r) / q)
    else:
        if m < 0.70710678118654752440:
            e -= 1
            x = _fm(2.0, m, -1.0)
        else:
            x = m - 1.0
        z = _mul(x, x)
        var p = _fm(1.01875663804580931796e-4, x, 4.97494994976747001425e-1)
        p = _fm(p, x, 4.70579119878881725854e0)
        p = _fm(p, x, 1.44989225341610930846e1)
        p = _fm(p, x, 1.79368678507819816313e1)
        p = _fm(p, x, 7.70838733755885391666e0)
        var q = x + 1.12873587189167450590e1
        q = _fm(q, x, 4.52279145837532221105e1)
        q = _fm(q, x, 8.29875266912776603211e1)
        q = _fm(q, x, 7.11544750618563894466e1)
        q = _fm(q, x, 2.31251620126765340583e1)
        y = _mul(x, _mul(z, p) / q)
    frac = x
    return y


def portable_log_c(xv: Float64) -> Float64:
    """packaging/portable_math/portable_math.c `mojolearn_log` (what
    `mojolearn._portable_math.log` calls), operation for operation, for a
    finite positive xv (the only kind the callers below pass)."""
    var u = bitcast[DType.uint64](xv)
    var raw_e = 0
    if (u >> 52) == UInt64(0):
        u = bitcast[DType.uint64](_mul(xv, 18014398509481984.0))
        raw_e = -54
    raw_e += Int((u >> 52) & UInt64(0x7FF)) - 1022
    var e = 0
    var x: Float64 = 0.0
    var y = _log_fraction(xv, e, x)
    y = _fm(Float64(e), -2.121944400546905827679e-4, y)
    if not (raw_e > 2 or raw_e < -2):
        y = _fm(_mul(x, x), -0.5, y)
    y = y + x
    return _fm(Float64(e), 0.693359375, y)


def _emi_cell(a: Int, b: Int, n: Int, mut terms: List[Float64]):
    """One (a, b) cell of `expected_mi`: its terms appended to `terms`, in
    ascending nij (the statements of the walk below, unchanged)."""
    var n_f64 = Float64(n)
    var la = portable_log_c(Float64(a))
    var lb = portable_log_c(Float64(b))
    var lo = max(0, a + b - n)
    var hi = min(a, b)
    var mode = min(max(((a + 1) * (b + 1)) // (n + 2), lo), hi)
    var up = List[Float64]()
    up.append(1.0)
    var x = mode
    var v: Float64 = 1.0
    while x < hi:
        v = _mul(v, Float64((a - x) * (b - x))) / Float64((x + 1) * (n - a - b + x + 1))
        if v == 0.0:
            break
        up.append(v)
        x += 1
    var down = List[Float64]()
    x = mode
    v = 1.0
    while x > lo:
        v = _mul(v, Float64(x * (n - a - b + x))) / Float64((a - x + 1) * (b - x + 1))
        if v == 0.0:
            break
        down.append(v)
        x -= 1
    var zs = List[Float64](capacity=len(up) + len(down))
    for k in range(len(up)):
        zs.append(up[k])
    for k in range(len(down)):
        zs.append(down[k])
    var z = fsum(zs)
    var first = mode - len(down)
    var nd = len(down)
    for k in range(nd + len(up)):
        var nij = first + k
        var u = down[nd - 1 - k] if k < nd else up[k - nd]
        if nij < 1:
            continue
        var pr = u / z
        if pr == 0.0:
            continue
        var q = Float64(nij) / n_f64
        var d = (portable_log_c(Float64(n * nij)) - la) - lb
        terms.append(_mul(_mul(q, d), pr))


def expected_mi(a_addr: Int, na: Int, b_addr: Int, nb: Int, n: Int) raises -> Float64:
    """python/mojolearn/_expansion_metrics.py `_expected_mi` for na, nb >= 2
    (Int64 class counts at the two addresses): the same integer bounds and
    mode, the same binary64 ratio walks from the mode (each stops at the
    first exact 0), the same pmf normalization by fsum, the same terms
    (nij / n) * (log(n nij) - log a - log b) * pr, fsum-ed. Python's
    int -> float conversions and int / int division are correctly rounded,
    as Float64(Int) and Float64 division are for these magnitudes.

    THREADS (lane metrics-apple3). A cell's terms depend on its own (a, b)
    alone, and `fsum` returns the correctly rounded EXACT sum of its
    values, which does not depend on their order or grouping. So the cells
    run as host tasks (`host_parallelize`: the caller's floating-point
    environment, so a walk passes through the same subnormals), each task
    reduces its cells' terms to math.fsum's partials (an exact expansion
    of their sum), and the partials of every task are fsum-ed: the
    correctly rounded sum of the same exact number. Below EPI_TASK_WORK
    rows, or with one task, the cells run in order on the caller."""
    var A = MutPointer[Int64, MutAnyOrigin](unsafe_from_address=a_addr)
    var B = MutPointer[Int64, MutAnyOrigin](unsafe_from_address=b_addr)
    if n <= 0 or n >= (1 << 31):
        raise Error("x_metrics epilogue: expected MI size goes the Python way")
    var cells = na * nb
    var tasks = 1
    if n >= EPI_TASK_WORK and cells > 1:
        tasks = min(4 * host_predict_task_count(cells), cells)
        if host_predict_task_count(cells) <= 1:
            tasks = 1
    if tasks <= 1:
        var terms = List[Float64]()
        for i in range(na):
            for j in range(nb):
                _emi_cell(Int(A[i]), Int(B[j]), n, terms)
        return fsum(terms)
    var chunk = host_predict_chunk(cells, tasks)
    var parts = List[Float64](length=tasks * EPI_PARTIALS, fill=0.0)
    var counts = List[Int](length=tasks, fill=0)
    var pp = MutPointer[Float64, MutAnyOrigin](unsafe_from_address=Int(parts.unsafe_ptr()))
    var cp = MutPointer[Int, MutAnyOrigin](unsafe_from_address=Int(counts.unsafe_ptr()))

    def _cells(t: Int) {imm A, imm B, imm pp, imm cp, imm n, imm nb, imm cells, imm chunk}:
        var c0 = t * chunk
        var c1 = min(c0 + chunk, cells)
        var mine = List[Float64]()
        for c in range(c0, c1):
            _emi_cell(Int(A[c // nb]), Int(B[c % nb]), n, mine)
        var part = List[Float64]()
        fsum_partials(mine, part)
        if len(part) > EPI_PARTIALS:
            cp[t] = -1
            return
        for q in range(len(part)):
            pp[t * EPI_PARTIALS + q] = part[q]
        cp[t] = len(part)

    host_parallelize(_cells, tasks)
    var all_parts = List[Float64]()
    var ok = True
    for t in range(tasks):
        var got = counts[t]
        if got < 0:
            ok = False
            break
        for q in range(got):
            all_parts.append(parts[t * EPI_PARTIALS + q])
    if not ok:
        var again = List[Float64]()
        for i in range(na):
            for j in range(nb):
                _emi_cell(Int(A[i]), Int(B[j]), n, again)
        return fsum(again)
    return fsum(all_parts)


@always_inline
def _row_fsum(S: FP, base: Int, k: Int, mut row: List[Float64]) -> Float64:
    """`fsum` of the k Float32 words at S[base], widened (lane
    metrics-apple3). The running binary64 sum is carried with Fast2Sum's
    error term (the larger magnitude first, as `fsum` orders each pair):
    while every error is exactly 0 the running sum IS the exact sum, and
    the exact sum, being representable, is the correctly rounded sum fsum
    returns (+0.0 for a zero sum). The first nonzero error sends the row
    to `fsum` itself."""
    var s: Float64 = 0.0
    var exact = True
    for c in range(k):
        var x = s
        var y = Float64(S.unsafe_load(base + c))
        if abs(x) < abs(y):
            var t = x
            x = y
            y = t
        var hi = x + y
        var yr = hi - x
        var lo = y - yr
        if lo != 0.0:
            exact = False
            break
        s = hi
    if exact:
        if s == 0.0:
            return 0.0
        return s
    for c in range(k):
        row[c] = Float64(S.unsafe_load(base + c))
    return fsum(row)


def row_sum_range(s_addr: Int, n: Int, k: Int, out_addr: Int):
    """`_rows_sum_to_one`'s two numbers: the largest and the smallest row
    `math.fsum` of the n x k finite Float32 scores (row major) at s_addr,
    at out_addr[0] and [1]. Each row's sum is `fsum` above (a zero sum is
    +0.0 where math.fsum may give -0.0; the caller takes |s - 1|).

    lane metrics-apple3: each row's sum by `_row_fsum` (the same value),
    and the rows as contiguous host tasks (a largest and a smallest value
    do not depend on the order the rows are visited in)."""
    var S = FP(unsafe_from_address=s_addr)
    var out = MutPointer[Float64, MutAnyOrigin](unsafe_from_address=out_addr)
    if n <= 0:
        out[0] = 0.0
        out[1] = 0.0
        return
    var tasks = 1
    if n * k >= 2 * EPI_TASK_WORK:
        tasks = max(min(host_predict_task_count(n), (n * k) // EPI_TASK_WORK), 1)
    var chunk = host_predict_chunk(n, tasks)
    var his = List[Float64](length=tasks, fill=0.0)
    var los = List[Float64](length=tasks, fill=0.0)
    var seen = List[Int](length=tasks, fill=0)
    var hp = MutPointer[Float64, MutAnyOrigin](unsafe_from_address=Int(his.unsafe_ptr()))
    var lp = MutPointer[Float64, MutAnyOrigin](unsafe_from_address=Int(los.unsafe_ptr()))
    var sp = MutPointer[Int, MutAnyOrigin](unsafe_from_address=Int(seen.unsafe_ptr()))

    def _rows(t: Int) {imm S, imm hp, imm lp, imm sp, imm n, imm k, imm chunk}:
        var r0 = t * chunk
        var r1 = min(r0 + chunk, n)
        if r1 <= r0:
            return
        var row = List[Float64](length=k, fill=0.0)
        var top: Float64 = 0.0
        var low: Float64 = 0.0
        for r in range(r0, r1):
            var v = _row_fsum(S, r * k, k, row)
            if r == r0 or v > top:
                top = v
            if r == r0 or v < low:
                low = v
        hp[t] = top
        lp[t] = low
        sp[t] = 1

    if tasks == 1:
        _rows(0)
    else:
        host_parallelize(_rows, tasks)
    var best_hi: Float64 = 0.0
    var best_lo: Float64 = 0.0
    var first = True
    for t in range(tasks):
        if seen[t] == 0:
            continue
        if first or his[t] > best_hi:
            best_hi = his[t]
        if first or los[t] < best_lo:
            best_lo = los[t]
        first = False
    out[0] = best_hi
    out[1] = best_lo


def scatter_rows(src_addr: Int, dst_addr: Int, idx_addr: Int, n: Int, n_dst: Int, row_bytes: Int) raises:
    """dst row idx[j] = src row j for j < n, rows of `row_bytes` bytes, idx
    Int64 (lane metrics-apple3: cross_val_predict puts each fold's
    predictions back in row order). An index outside [0, n_dst) raises
    before any write. A byte copy: no value is computed."""
    if n < 0 or n_dst < 0 or row_bytes < 1:
        raise Error("x_metrics scatter_rows: invalid sizes")
    var I = MutPointer[Int64, MutAnyOrigin](unsafe_from_address=idx_addr)
    for j in range(n):
        var r = Int(I[j])
        if r < 0 or r >= n_dst:
            raise Error("x_metrics scatter_rows: index out of range")
    var sp = MutPointer[UInt8, MutAnyOrigin](unsafe_from_address=src_addr)
    var dp = MutPointer[UInt8, MutAnyOrigin](unsafe_from_address=dst_addr)
    for j in range(n):
        memcpy(dest=dp + Int(I[j]) * row_bytes, src=sp + j * row_bytes, count=row_bytes)


# ---------------------------------------------------------------------------
# lane py-misc-metrics (2026-09-28): the rest of DEVIATION 6106's O(n) and
# O(cells) epilogues. Each is the Python it stands in for in
# python/mojolearn/_expansion_metrics.py, operation for operation, in the
# same order: + - / correctly rounded, every product `pinned_mul_f64`,
# `fsum_strict` for `_fsum` (CPython's math.fsum, raising wherever the
# Python would leave math.fsum: a non-finite term, an intermediate
# overflow, a non-finite sum), `portable_log_c` for `_portable_math.log`,
# IEEE sqrt for `_portable_math.sqrt`. Any case the Python handles by a
# warning, an exception or a non-finite value raises here, and the caller
# falls back to its Python, which decides it.
# ---------------------------------------------------------------------------

@always_inline
def _finite(x: Float64) -> Bool:
    return (x - x) == 0.0


def fsum_strict(vals: List[Float64]) raises -> Float64:
    """`_fsum` where it is CPython's math.fsum: every term finite, no
    partial overflows, a finite sum (a zero sum is +0.0); anything else
    raises (the Python then takes its portable route)."""
    for k in range(len(vals)):
        if not _finite(vals[k]):
            raise Error("x_metrics epilogue: a non-finite fsum term goes the Python way")
    # math.fsum's intermediate-overflow test: a partial sum that is not
    # finite. With finite terms the running magnitude is bounded by the
    # sum of |terms|; when that bound is finite no partial can overflow.
    var bound: Float64 = 0.0
    for k in range(len(vals)):
        bound = bound + abs(vals[k])
    if not (bound < 8.98846567431158e307):  # 2^1023: no partial can reach the overflow threshold
        raise Error("x_metrics epilogue: a possible fsum overflow goes the Python way")
    var s = fsum(vals)
    if not _finite(s):
        raise Error("x_metrics epilogue: a non-finite sum goes the Python way")
    return s


@always_inline
def _dp(p: Int) -> MutPointer[Float64, MutAnyOrigin]:
    return MutPointer[Float64, MutAnyOrigin](unsafe_from_address=p)


@always_inline
def _qp(p: Int) -> MutPointer[Int64, MutAnyOrigin]:
    return MutPointer[Int64, MutAnyOrigin](unsafe_from_address=p)


def _pr_keep(a: Int, tps: Int, c: Int, drop: Bool) -> List[Int]:
    """precision_recall_curve / det_curve's drop_intermediate rule: the
    first, the last and every i with tps[i] != tps[i-1] or tps[i+1] !=
    tps[i] (applied only when drop and c > 2)."""
    var out = List[Int](capacity=c)
    if not drop or c <= 2:
        for i in range(c):
            out.append(i)
        return out^
    out.append(0)
    for i in range(1, c - 1):
        if _w(a, tps + i) != _w(a, tps + i - 1) or _w(a, tps + i + 1) != _w(a, tps + i):
            out.append(i)
    out.append(c - 1)
    return out^


def pr_arrays(a: Int, fps: Int, tps: Int, thr: Int, c: Int, drop: Bool,
              out_prec: Int, out_rec: Int, out_thr: Int) raises -> Int:
    """`precision_recall_curve_options` for tps[c-1] != 0 (the caller
    checks): precision (m + 1, the last 1.0), recall (m + 1, the last 0.0),
    thresholds (m), each reversed, m = the kept points. Returns m."""
    if c <= 0:
        raise Error("x_metrics epilogue: an empty curve goes the Python way")
    var T = _w(a, tps + c - 1)
    if not (T != 0.0):
        raise Error("x_metrics epilogue: no positives goes the Python way")
    var ks = _pr_keep(a, tps, c, drop)
    var m = len(ks)
    var pp = _dp(out_prec)
    var pr = _dp(out_rec)
    var ph = _dp(out_thr)
    for j in range(m):
        var i = ks[j]
        var t = _w(a, tps + i)
        var d = t + _w(a, fps + i)
        var r = m - 1 - j
        pp[r] = t / d if d != 0.0 else 0.0
        pr[r] = t / T
        ph[r] = _w(a, thr + i)
    pp[m] = 1.0
    pr[m] = 0.0
    return m


def _bisect_right(v: List[Float64], x: Float64) -> Int:
    var lo = 0
    var hi = len(v)
    while lo < hi:
        var mid = (lo + hi) // 2
        if x < v[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _bisect_left(v: List[Float64], x: Float64) -> Int:
    var lo = 0
    var hi = len(v)
    while lo < hi:
        var mid = (lo + hi) // 2
        if v[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


def det_arrays(a: Int, fps: Int, tps: Int, thr: Int, c: Int, drop: Bool,
               out_fpr: Int, out_fnr: Int, out_thr: Int) raises -> Int:
    """`det_curve` after its two-class check: the kept points behind a
    leading (0, 0, +inf), fns = T - tps, the slice [first, last) of
    bisect_right(fps, fps[0]) - 1 and bisect_left(tps, T) + 1, fpr = fps /
    F and fnr = fns / T over it, each reversed. F and T must be nonzero (a
    zero goes the Python way, which raises or warns). Returns the length."""
    if c <= 0:
        raise Error("x_metrics epilogue: an empty curve goes the Python way")
    var ks = _pr_keep(a, tps, c, drop)
    var m = len(ks) + 1
    var f = List[Float64](capacity=m)
    var t = List[Float64](capacity=m)
    f.append(0.0)
    t.append(0.0)
    for j in range(len(ks)):
        f.append(_w(a, fps + ks[j]))
        t.append(_w(a, tps + ks[j]))
    var T = t[m - 1]
    var F = f[m - 1]
    if not (T != 0.0 and F != 0.0):
        raise Error("x_metrics epilogue: an empty class goes the Python way")
    var right = _bisect_right(f, f[0])
    var first = right - 1 if right > 0 else 0
    var last = min(_bisect_left(t, T) + 1, m)
    var L = last - first if last > first else 0
    var pf = _dp(out_fpr)
    var pn = _dp(out_fnr)
    var ph = _dp(out_thr)
    for q in range(L):
        var i = first + q
        var r = L - 1 - q
        pf[r] = f[i] / F
        pn[r] = (T - t[i]) / T
        if i == 0:
            ph[r] = bitcast[DType.float64](UInt64(0x7FF0000000000000))
        else:
            ph[r] = _w(a, thr + ks[i - 1])
    return L


def ndcg_mean(a: Int, gain: Int, ideal: Int, n: Int, w_addr: Int) raises -> Float64:
    """`ndcg_score`'s epilogue: per = g / i (0.0 where i == 0); unweighted
    fsum(per) / n, weighted fsum(per * w) / fsum(w), w the Float32 weights
    at w_addr (0: none) widened exactly."""
    var per = List[Float64](capacity=n)
    for r in range(n):
        var g = _w(a, gain + r)
        var i = _w(a, ideal + r)
        per.append(g / i if i != 0.0 else 0.0)
    if w_addr == 0:
        return fsum_strict(per) / Float64(n)
    var W = FP(unsafe_from_address=w_addr)
    var wl = List[Float64](capacity=n)
    for r in range(n):
        wl.append(Float64(W.unsafe_load(r)))
    var terms = List[Float64](capacity=n)
    for r in range(n):
        terms.append(pinned_mul_f64(per[r], wl[r]))
    var den = fsum_strict(wl)
    if den == 0.0:
        raise Error("x_metrics epilogue: a zero weight total goes the Python way")
    return fsum_strict(terms) / den


def class_sums(codes_addr: Int, w_addr: Int, n: Int, k: Int, out_addr: Int) raises:
    """`_class_weights` / `_ovr`'s weighted support: per[c] += 1 (w_addr 0)
    or += w[r] (the Float32 weight widened), binary64, in row order; a code
    outside [0, k) raises (Python would index from the end or fail)."""
    var C = IP(unsafe_from_address=codes_addr)
    var out = _dp(out_addr)
    for c in range(k):
        out[c] = 0.0
    var W = FP(unsafe_from_address=w_addr if w_addr != 0 else codes_addr)
    for r in range(n):
        var c = Int(C.unsafe_load(r))
        if c < 0 or c >= k:
            raise Error("x_metrics epilogue: a class code outside [0, k) goes the Python way")
        if w_addr == 0:
            out[c] = out[c] + 1.0
        else:
            out[c] = out[c] + Float64(W.unsafe_load(r))


def auc_xy(x_addr: Int, y_addr: Int, n: Int) raises -> Float64:
    """public `auc` over n >= 2 binary64 points: dx, the direction (-1 when
    every dx <= 0 and one < 0; a mixed x raises for the Python's message),
    direction * trapezoid (fsum of (dx) * (y[i] + y[i-1]) / 2)."""
    var X = _dp(x_addr)
    var Y = _dp(y_addr)
    if n < 2:
        raise Error("x_metrics epilogue: fewer than 2 points go the Python way")
    var neg = False
    var pos = False
    for i in range(1, n):
        var d = X[i] - X[i - 1]
        if d < 0.0:
            neg = True
        elif d > 0.0:
            pos = True
        elif not (d <= 0.0):
            pos = True  # NaN: not <= 0
    var direction: Float64 = 1.0
    if neg:
        if pos:
            raise Error("x_metrics epilogue: a non-monotonic x goes the Python way")
        direction = -1.0
    var terms = List[Float64](capacity=n - 1)
    for i in range(1, n):
        terms.append(pinned_mul_f64(X[i] - X[i - 1], Y[i] + Y[i - 1]) / 2.0)
    return direction * fsum_strict(terms)


def mi_contingency(c_addr: Int, ka: Int, kb: Int) raises -> Float64:
    """`_mi_from_contingency` of the ka x kb Int64 counts (row major) at
    c_addr: the same terms nm * (log v - log total) + nm * log_outer, with
    log_outer = (-log(pi pj) + log(sum pi)) + log(sum pj) (the two
    loop-invariant logs hoisted: the same values), |t| < eps to 0.0,
    max(fsum, 0.0)."""
    var C = _qp(c_addr)
    var pi = List[Int](length=ka, fill=0)
    var pj = List[Int](length=kb, fill=0)
    var total = 0
    for i in range(ka):
        for j in range(kb):
            var v = Int(C[i * kb + j])
            if v < 0:
                raise Error("x_metrics epilogue: a negative count goes the Python way")
            pi[i] += v
            pj[j] += v
            total += v
    if ka == 1 or kb == 1:
        return 0.0
    if total <= 0 or total >= (1 << 31):
        raise Error("x_metrics epilogue: contingency size goes the Python way")
    var ft = Float64(total)
    var lt = portable_log_c(ft)
    var lsi = portable_log_c(ft)   # log(sum(pi)); sum(pi) == total
    var lsj = portable_log_c(ft)   # log(sum(pj))
    var terms = List[Float64]()
    for i in range(ka):
        for j in range(kb):
            var v = Int(C[i * kb + j])
            if v == 0:
                continue
            var nm = Float64(v) / ft
            var log_outer = (-portable_log_c(Float64(pi[i] * pj[j])) + lsi) + lsj
            var t = pinned_mul_f64(nm, portable_log_c(Float64(v)) - lt) + pinned_mul_f64(nm, log_outer)
            terms.append(0.0 if abs(t) < 2.220446049250313e-16 else t)
    var s = fsum_strict(terms)
    if s > 0.0:
        return s
    return 0.0


def _cents(a: Int, sums: Int, counts_addr: Int, k: Int, d: Int) raises -> List[Float64]:
    """cents[i*d + c] = sums[i*d + c] / counts[i] (Float32 sums widened,
    Int64 counts, exact below 2^53)."""
    var Q = _qp(counts_addr)
    var out = List[Float64](capacity=k * d)
    for i in range(k):
        var cnt = Int(Q[i])
        if cnt <= 0 or cnt >= (1 << 53):
            raise Error("x_metrics epilogue: a cluster count goes the Python way")
        var fc = Float64(cnt)
        for c in range(d):
            out.append(_w(a, sums + i * d + c) / fc)
    return out^


def centroids_f32(a: Int, sums: Int, counts_addr: Int, k: Int, d: Int, out_addr: Int) raises:
    """`_row_dists`'s centroid words: Float32(cents) (round to nearest even,
    `_f32`)."""
    var cs = _cents(a, sums, counts_addr, k, d)
    var O = FP(unsafe_from_address=out_addr)
    for q in range(k * d):
        O.unsafe_store(q, Float32(cs[q]))


def ch_extra(a: Int, sums: Int, gsum: Int, counts_addr: Int, k: Int, d: Int, n: Int) raises -> Float64:
    """calinski_harabasz_score's between-cluster dispersion: fsum over i of
    counts[i] * fsum over c of (cents - mean)^2 (the square a product),
    mean = gsum / n."""
    var cs = _cents(a, sums, counts_addr, k, d)
    var Q = _qp(counts_addr)
    var fnn = Float64(n)
    var mean = List[Float64](capacity=d)
    for c in range(d):
        mean.append(_w(a, gsum + c) / fnn)
    var outer = List[Float64](capacity=k)
    var inner = List[Float64](length=d, fill=0.0)
    for i in range(k):
        for c in range(d):
            var e = cs[i * d + c] - mean[c]
            inner[c] = pinned_mul_f64(e, e)
        outer.append(pinned_mul_f64(Float64(Int(Q[i])), fsum_strict(inner)))
    return fsum_strict(outer)


def db_score(a: Int, sums: Int, counts_addr: Int, k: Int, d: Int, per_addr: Int) raises -> Float64:
    """davies_bouldin_score after its row distances (per_addr: k binary64
    per-cluster sums): intra = per / counts, dist = sqrt(fsum of squared
    centroid differences), 0.0 when every intra or every dist is within
    1e-8 of 0, else fsum over i of max_j (intra_i + intra_j) / dist_ij
    (inf for a zero dist), over k."""
    from std.math import sqrt
    var cs = _cents(a, sums, counts_addr, k, d)
    var Q = _qp(counts_addr)
    var P = _dp(per_addr)
    var intra = List[Float64](capacity=k)
    for i in range(k):
        intra.append(P[i] / Float64(Int(Q[i])))
    var dist = List[Float64](capacity=k * k)
    var inner = List[Float64](length=d, fill=0.0)
    for i in range(k):
        for j in range(k):
            for c in range(d):
                var e = cs[i * d + c] - cs[j * d + c]
                inner[c] = pinned_mul_f64(e, e)
            dist.append(sqrt(fsum_strict(inner)))
    var all_i = True
    for i in range(k):
        if not (abs(intra[i]) <= 1e-8):
            all_i = False
    var all_d = True
    for q in range(k * k):
        if not (abs(dist[q]) <= 1e-8):
            all_d = False
    if all_i or all_d:
        return 0.0
    var inf = bitcast[DType.float64](UInt64(0x7FF0000000000000))
    var scores = List[Float64](capacity=k)
    for i in range(k):
        var best = -inf
        for j in range(k):
            var dd = dist[i * k + j]
            var den = dd if dd != 0.0 else inf
            var v = (intra[i] + intra[j]) / den
            if v > best:   # Python's max(best, v): v only when it is larger
                best = v
        scores.append(best)
    return fsum_strict(scores) / Float64(k)


# ---------------------------------------------------------------------------
# lane metrics-apple3 (2026-09-28): integer label plumbing. No float is read
# or written: integer compares and byte stores only.
# ---------------------------------------------------------------------------

def encode_small_i64(src_addr: Int, n: Int, classes_addr: Int, max_classes: Int, codes_addr: Int) raises -> Int:
    """The ORDER RULE's encoder (`bindings/hotpath_helpers.mojo`
    `_encode_labels`) for n Int64 labels that span fewer than ENC_SPAN
    values: the distinct labels ascending at classes_addr (Int64), each
    row's index into them at codes_addr (Int32). Returns the class count;
    -1 when more than `max_classes` distinct labels were seen (nothing the
    caller may read was written); -3 when the labels span ENC_SPAN values or
    more (the caller takes the core encoder). The classes of a set of
    integers and each label's rank among them do not depend on the order
    the rows are visited in, so the rows run as host tasks: the span, a
    seen-byte per value per task, then one table load per row."""
    if n < 1 or max_classes < 1:
        raise Error("x_metrics encode_small: n and max_classes must be positive")
    var src = MutPointer[Int64, MutAnyOrigin](unsafe_from_address=src_addr)
    var cls = MutPointer[Int64, MutAnyOrigin](unsafe_from_address=classes_addr)
    var codes = MutPointer[Int32, MutAnyOrigin](unsafe_from_address=codes_addr)
    var tasks = 1
    if n >= 2 * EPI_TASK_WORK:
        tasks = max(min(host_predict_task_count(n), n // EPI_TASK_WORK), 1)
    var chunk = host_predict_chunk(n, tasks)
    var mins = List[Int](length=tasks, fill=0)
    var maxs = List[Int](length=tasks, fill=0)
    var used = List[Int](length=tasks, fill=0)
    var minp = MutPointer[Int, MutAnyOrigin](unsafe_from_address=Int(mins.unsafe_ptr()))
    var maxp = MutPointer[Int, MutAnyOrigin](unsafe_from_address=Int(maxs.unsafe_ptr()))
    var usedp = MutPointer[Int, MutAnyOrigin](unsafe_from_address=Int(used.unsafe_ptr()))

    def _span(t: Int) {imm src, imm minp, imm maxp, imm usedp, imm n, imm chunk}:
        var r0 = t * chunk
        var r1 = min(r0 + chunk, n)
        if r1 <= r0:
            return
        var least = Int(src[r0])
        var most = least
        for r in range(r0 + 1, r1):
            var v = Int(src[r])
            if v < least:
                least = v
            if v > most:
                most = v
        minp[t] = least
        maxp[t] = most
        usedp[t] = 1

    if tasks == 1:
        _span(0)
    else:
        host_parallelize(_span, tasks)
    var lo = 0
    var hi = 0
    var first = True
    for t in range(tasks):
        if used[t] == 0:
            continue
        if first or mins[t] < lo:
            lo = mins[t]
        if first or maxs[t] > hi:
            hi = maxs[t]
        first = False
    var bound = 1 << 40
    if first or lo <= -bound or hi >= bound or hi - lo >= ENC_SPAN:
        return -3
    var span = hi - lo + 1
    var seen = List[UInt8](length=tasks * span, fill=UInt8(0))
    var seenp = MutPointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(seen.unsafe_ptr()))

    def _mark(t: Int) {imm src, imm seenp, imm n, imm chunk, imm span, imm lo}:
        var r0 = t * chunk
        var r1 = min(r0 + chunk, n)
        for r in range(r0, r1):
            seenp[t * span + (Int(src[r]) - lo)] = UInt8(1)

    if tasks == 1:
        _mark(0)
    else:
        host_parallelize(_mark, tasks)
    var table = List[Int32](length=span, fill=Int32(-1))
    var k = 0
    for v in range(span):
        var found = False
        for t in range(tasks):
            if seen[t * span + v] != UInt8(0):
                found = True
                break
        if found:
            if k == max_classes:
                return -1
            cls[k] = Int64(lo + v)
            table[v] = Int32(k)
            k += 1
    var tablep = MutPointer[Int32, MutAnyOrigin](unsafe_from_address=Int(table.unsafe_ptr()))

    def _code(t: Int) {imm src, imm codes, imm tablep, imm n, imm chunk, imm lo}:
        var r0 = t * chunk
        var r1 = min(r0 + chunk, n)
        for r in range(r0, r1):
            codes[r] = tablep[Int(src[r]) - lo]

    if tasks == 1:
        _code(0)
    else:
        host_parallelize(_code, tasks)
    _ = len(table)
    _ = len(seen)
    return k


def first_rows_i32(codes_addr: Int, n: Int, k: Int, out_addr: Int) raises:
    """out[c] (Int64) = the first row whose Int32 code is c, -1 when no row
    has it; a code outside [0, k) raises. StratifiedKFold's first-seen class
    order read from sorted codes."""
    if n < 0 or k < 1:
        raise Error("x_metrics first_rows: invalid sizes")
    var C = MutPointer[Int32, MutAnyOrigin](unsafe_from_address=codes_addr)
    var out = MutPointer[Int64, MutAnyOrigin](unsafe_from_address=out_addr)
    for c in range(k):
        out[c] = Int64(-1)
    var left = k
    for r in range(n):
        var c = Int(C[r])
        if c < 0 or c >= k:
            raise Error("x_metrics first_rows: a class code outside [0, k)")
        if Int(out[c]) < 0:
            out[c] = Int64(r)
            left -= 1
            if left == 0:
                break
    # the rows after the last first row still have to be valid codes
    for r in range(n):
        var c = Int(C[r])
        if c < 0 or c >= k:
            raise Error("x_metrics first_rows: a class code outside [0, k)")


def ovo_pair(codes_addr: Int, s_addr: Int, n: Int, k: Int, a: Int, b: Int,
             out_sa: Int, out_sb: Int, out_fa: Int, out_fb: Int, m: Int) raises -> Int:
    """roc_auc_score one-vs-one, the pair (a, b): the rows whose Int32 code
    is a or b, in ascending row order; for each, its score in column a and
    in column b of the n x k row-major Float32 scores (the words copied as
    they are) and the 0/1 Int32 flags code == a and code == b. `m` is the
    length of the four outputs: a row past it raises before it is written.
    Returns the rows written. A byte selection: no value is computed."""
    if n < 0 or k < 1 or a < 0 or a >= k or b < 0 or b >= k or m < 0:
        raise Error("x_metrics ovo_pair: invalid sizes")
    var C = MutPointer[Int32, MutAnyOrigin](unsafe_from_address=codes_addr)
    var S = FP(unsafe_from_address=s_addr)
    var sa = FP(unsafe_from_address=out_sa)
    var sb = FP(unsafe_from_address=out_sb)
    var fa = MutPointer[Int32, MutAnyOrigin](unsafe_from_address=out_fa)
    var fb = MutPointer[Int32, MutAnyOrigin](unsafe_from_address=out_fb)
    var at = 0
    for r in range(n):
        var c = Int(C[r])
        if c != a and c != b:
            continue
        if at >= m:
            raise Error("x_metrics ovo_pair: more rows than the caller counted")
        sa.unsafe_store(at, S.unsafe_load(r * k + a))
        sb.unsafe_store(at, S.unsafe_load(r * k + b))
        fa[at] = Int32(1) if c == a else Int32(0)
        fb[at] = Int32(1) if c == b else Int32(0)
        at += 1
    return at
