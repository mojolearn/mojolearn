# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE CURVE EPILOGUE ON THE HOST (lane metrics-apple2, 2026-09-28).

DEVIATION 6106 puts what follows the O(n) device work in IEEE binary64 on
the host: for the ROC and precision-recall scores that is O(n) too (one
term per curve point), and in Python it made a Python float per point,
several times over. The ROC AUC and average precision SUMS left this file
(lane cgr2-metrics-shap, 2026-10-03): they are device folds now
(x_metrics/par.mojo curve_fold_unit); what stays here formats the curves a
caller receives as Float64 arrays (one division per returned point). These functions are the SAME
binary64 operations, in the same order, as the Python they stand in for in
python/mojolearn/_expansion_metrics.py (`_drop_collinear`, `roc_curve`),
run over the Float32 curve words
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
from x_metrics.common import FP, IP

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
    as Float64(Int) and Float64 division are for these magnitudes."""
    var A = MutPointer[Int64, MutAnyOrigin](unsafe_from_address=a_addr)
    var B = MutPointer[Int64, MutAnyOrigin](unsafe_from_address=b_addr)
    if n <= 0 or n >= (1 << 31):
        raise Error("x_metrics epilogue: expected MI size goes the Python way")
    var n_f64 = Float64(n)
    var terms = List[Float64]()
    for i in range(na):
        var a = Int(A[i])
        var la = portable_log_c(Float64(a))
        for j in range(nb):
            var b = Int(B[j])
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
    return fsum(terms)


def expected_mi_tasks(a_addr: Int, na: Int, b_addr: Int, nb: Int, n: Int) raises -> Float64:
    """`expected_mi` cell by cell through `_emi_cell` (lane metrics-apple3's
    entry, kept for the host binding's `x_metrics_expected_mi_tasks`). The
    cells' terms are fsum-ed once, the correctly rounded exact sum, so the
    value is `expected_mi`'s. cpu-gpu-cleanup c-metrics-prep: the host
    thread pool is gone from this GPU-binding module (the Python caller's
    opt-in MOJOLEARN_MSEL3 arm is deleted)."""
    var A = MutPointer[Int64, MutAnyOrigin](unsafe_from_address=a_addr)
    var B = MutPointer[Int64, MutAnyOrigin](unsafe_from_address=b_addr)
    if n <= 0 or n >= (1 << 31):
        raise Error("x_metrics epilogue: expected MI size goes the Python way")
    var terms = List[Float64]()
    for i in range(na):
        for j in range(nb):
            _emi_cell(Int(A[i]), Int(B[j]), n, terms)
    return fsum(terms)


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
    +0.0 where math.fsum may give -0.0; the caller takes |s - 1|)."""
    var S = FP(unsafe_from_address=s_addr)
    var out = MutPointer[Float64, MutAnyOrigin](unsafe_from_address=out_addr)
    var row = List[Float64](length=k, fill=0.0)
    var hi: Float64 = 0.0
    var lo: Float64 = 0.0
    for r in range(n):
        for c in range(k):
            row[c] = Float64(S.unsafe_load(r * k + c))
        var v = fsum(row)
        if r == 0 or v > hi:
            hi = v
        if r == 0 or v < lo:
            lo = v
    out[0] = hi
    out[1] = lo


def row_sum_range_tasks(s_addr: Int, n: Int, k: Int, out_addr: Int):
    """`row_sum_range` by `_row_fsum` (lane metrics-apple3's entry, kept for
    the host binding's `x_metrics_row_sum_range_tasks`): the largest and the
    smallest row `math.fsum` of the n x k finite Float32 scores (row major)
    at s_addr, at out_addr[0] and [1] (a zero sum is +0.0 where math.fsum
    may give -0.0; the caller takes |s - 1|). cpu-gpu-cleanup
    c-metrics-prep: no host thread pool in this GPU-binding module."""
    var S = FP(unsafe_from_address=s_addr)
    var out = MutPointer[Float64, MutAnyOrigin](unsafe_from_address=out_addr)
    if n <= 0:
        out[0] = 0.0
        out[1] = 0.0
        return
    var row = List[Float64](length=k, fill=0.0)
    var top: Float64 = 0.0
    var low: Float64 = 0.0
    for r in range(n):
        var v = _row_fsum(S, r * k, k, row)
        if r == 0 or v > top:
            top = v
        if r == 0 or v < low:
            low = v
    out[0] = top
    out[1] = low


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
    more (the caller takes the core encoder): the span, a seen-byte per
    value, then one table load per row. cpu-gpu-cleanup c-metrics-prep: no
    host thread pool in this GPU-binding module (the Python caller's opt-in
    MOJOLEARN_MSEL3 arm is deleted; the host binding keeps the entry)."""
    if n < 1 or max_classes < 1:
        raise Error("x_metrics encode_small: n and max_classes must be positive")
    var src = MutPointer[Int64, MutAnyOrigin](unsafe_from_address=src_addr)
    var cls = MutPointer[Int64, MutAnyOrigin](unsafe_from_address=classes_addr)
    var codes = MutPointer[Int32, MutAnyOrigin](unsafe_from_address=codes_addr)
    var lo = Int(src[0])
    var hi = lo
    for r in range(1, n):
        var v = Int(src[r])
        if v < lo:
            lo = v
        if v > hi:
            hi = v
    var bound = 1 << 40
    if lo <= -bound or hi >= bound or hi - lo >= ENC_SPAN:
        return -3
    var span = hi - lo + 1
    var seen = List[UInt8](length=span, fill=UInt8(0))
    for r in range(n):
        seen[Int(src[r]) - lo] = UInt8(1)
    var table = List[Int32](length=span, fill=Int32(-1))
    var k = 0
    for v in range(span):
        if seen[v] != UInt8(0):
            if k == max_classes:
                return -1
            cls[k] = Int64(lo + v)
            table[v] = Int32(k)
            k += 1
    for r in range(n):
        codes[r] = table[Int(src[r]) - lo]
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
