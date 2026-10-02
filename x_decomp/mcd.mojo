# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""MinCovDet's fast_mcd driver in Mojo (lane/py-decomp-nbrs, 2026-09-28).

Before this file, `_expansion_decomp.MinCovDet._fast_mcd` drove the search
from Python: one C-step at a time, 12 to 40 kit calls each, Python argsorts
over n and `take_rows` gathers (48 s at 20k x 8 measured, about 40 minutes
at 1M). This is the SAME search, statement for statement, with the SAME
cells in the SAME order: every kit call Python made (`ew`, `gemm`,
`colsum`, `rowsum`, `eigh`, `lu`, `rand`) is the same executor call here,
with the same operands, broadcast modes and float32 scalars. What moved is
control flow and data movement only:

- the argsorts. Python sorted by `(value, index)` tuples; here the value
  is mapped to its monotone uint32 image (-0.0 and +0.0 made one value, as
  Python's `==` makes them) and packed above the index, so the order is
  the same total order. A permutation is a full sort of those keys; the
  "h smallest, then sorted by index" selections are a quickselect of the
  h-th key and a scan in index order, the same SET of rows in the same
  order. A NaN distance or draw is REFUSED (Python's tuple sort of a NaN
  is not a total order; no finite input reaches one).
- `take_rows`, the support mask and the distance scatter: exact copies.
- the Python float scalars: `1.0 / r` (IEEE double), the pinvh cutoff
  `(wmax * n) * eps` in double then rounded once to float32, exactly as
  `_f32` and the binding boundary round them.

The CPU column runs this file on `x_decomp/kit.mojo`; the GPU binding runs
x_decomp/kit_device.mojo's `fast_mcd_dev`, the same search with every
matrix resident on the device.
"""
from std.memory import bitcast
from std.builtin.sort import sort

from x_decomp.cells import F32Ptr, I32Ptr
from x_decomp.exec_trait import Exec
from x_decomp.kit import Kit, Mat, OP_ABS, OP_LOGS, OP_MUL, OP_RECIP, OP_SCALE, OP_SUB, take_rows

#: `_expansion_decomp._F32_EPS` and the `logs` floor `_slogdet` passes
comptime _F32_EPS: Float64 = 1.1920928955078125e-07
comptime _FLT_MIN: Float64 = 1.1754943508222875e-38


def _neg_inf() -> Float64:
    return bitcast[DType.float64](UInt64(0xFFF0000000000000))


def _pos_inf() -> Float64:
    return bitcast[DType.float64](UInt64(0x7FF0000000000000))


def _key(v: Float32, i: Int) raises -> UInt64:
    """(v, i) as one uint64 in Python's tuple order: v's monotone image
    above the index. -0.0 is +0.0 (Python's `==`); NaN is refused."""
    if v != v:
        raise Error("x_decomp MinCovDet: a NaN distance or draw has no order (refused)")
    var w = v
    if w == Float32(0):
        w = Float32(0)
    var ub = bitcast[DType.uint32](w)
    var tw = ub ^ UInt32(0xFFFFFFFF) if (ub >> 31) == 1 else ub | UInt32(0x80000000)
    return (UInt64(tw) << 32) | UInt64(i)


def argsort_values(v: Mat) raises -> List[Int]:
    """`sorted(range(n), key=lambda i: (v[i], i))`."""
    var n = v.n()
    var keys = List[UInt64](capacity=n)
    for i in range(n):
        keys.append(_key(v.d[i], i))
    sort(keys)
    var out = List[Int](capacity=n)
    for a in range(n):
        out.append(Int(keys[a] & UInt64(0xFFFFFFFF)))
    return out^


def smallest_sorted(v: Mat, h: Int) raises -> List[Int]:
    """`sorted(sorted(range(n), key=lambda i: (v[i], i))[:h])`: the h rows
    with the smallest keys, in index order."""
    var n = v.n()
    if h <= 0:
        return List[Int]()
    if h >= n:
        for i in range(n):
            _ = _key(v.d[i], i)          # the NaN refusal, as the sort would meet it
        var all = List[Int](capacity=n)
        for i in range(n):
            all.append(i)
        return all^
    var keys = List[UInt64](capacity=n)
    for i in range(n):
        keys.append(_key(v.d[i], i))
    var kth = _quickselect(keys, h - 1)
    var out = List[Int](capacity=h)
    for i in range(n):
        if _key(v.d[i], i) <= kth:
            out.append(i)
    if len(out) != h:
        raise Error("x_decomp MinCovDet: selection count differs from h")
    return out^


def _quickselect(mut keys: List[UInt64], k: Int) -> UInt64:
    """The k-th smallest (0-based) of distinct keys (Hoare partition,
    median of three). Reorders `keys`."""
    var lo = 0
    var hi = len(keys) - 1
    while lo < hi:
        var a = keys[lo]
        var b = keys[(lo + hi) // 2]
        var c = keys[hi]
        var piv: UInt64
        if (a <= b and b <= c) or (c <= b and b <= a):
            piv = b
        elif (b <= a and a <= c) or (c <= a and a <= b):
            piv = a
        else:
            piv = c
        var i = lo
        var j = hi
        while i <= j:
            while keys[i] < piv:
                i += 1
            while keys[j] > piv:
                j -= 1
            if i <= j:
                var t = keys[i]
                keys[i] = keys[j]
                keys[j] = t
                i += 1
                j -= 1
        if k <= j:
            hi = j
        elif k >= i:
            lo = i
        else:
            break
    return keys[k]


struct Est(Copyable, Movable):
    """One C-step's answer: (location, covariance, log det, support rows,
    distances)."""
    var loc: Mat
    var cov: Mat
    var det: Float64
    var sel: List[Int]
    var dist: Mat

    def __init__(out self, var loc: Mat, var cov: Mat, det: Float64, var sel: List[Int], var dist: Mat):
        self.loc = loc^
        self.cov = cov^
        self.det = det
        self.sel = sel^
        self.dist = dist^


def _order_by_det(est: List[Est], keep: Int) raises -> List[Int]:
    """`sorted(range(len(est)), key=lambda j: (est[j].det, j))[:keep]`.
    A det is a float32 value or -inf, so its float32 image orders it."""
    var keys = List[UInt64](capacity=len(est))
    for j in range(len(est)):
        keys.append(_key(Float32(est[j].det), j))
    sort(keys)
    var out = List[Int]()
    for a in range(min(keep, len(keys))):
        out.append(Int(keys[a] & UInt64(0xFFFFFFFF)))
    return out^


struct Mcd[E: Exec]:
    var k: Kit[Self.E]
    var seed: Int
    var draws: Int

    def __init__(out self, seed: Int):
        self.k = Kit[Self.E]()
        self.seed = seed
        self.draws = 0

    # ---- the composites of _expansion_decomp.py
    def colmean(self, A: Mat) raises -> Mat:
        return self.k.colmean(A)

    def emp_cov(self, Xs: Mat) raises -> Mat:
        """`_emp_cov(k, Xs)` (assume_centered=False)."""
        var Xc = self.k.ew2(OP_SUB, Xs, self.colmean(Xs))
        return self.k.ew1(OP_SCALE, self.k.mm(Xc, Xc, True, False), 1.0 / Float64(Xs.r))

    def mahal(self, X: Mat, loc: Mat, P: Mat) raises -> Mat:
        var Xc = self.k.ew2(OP_SUB, X, loc)
        return self.k.rowsum(self.k.ew2(OP_MUL, self.k.mm(Xc, P, False, False), Xc))

    def pinvh(self, A: Mat) raises -> Mat:
        """`_pinvh`: V diag(1/w) V^T over |w| > max|w| * n * float32 eps."""
        var n = A.r
        var w = Mat(1, n)
        var V = Mat(n, n)
        self.k.eigh(A, w, V)
        var wmax: Float64 = 0.0
        if n > 0:
            wmax = abs(Float64(w.d[0]))
            for j in range(1, n):
                var a = abs(Float64(w.d[j]))
                if a > wmax:                  # Python max(): replace on >
                    wmax = a
        var cut = Float64(Float32((wmax * Float64(n)) * _F32_EPS))
        var keep = self.k.ew1(OP_RECIP, w, 0.0)
        var inv = Mat(1, n)
        for j in range(n):
            inv.d[j] = keep.d[j] if abs(Float64(w.d[j])) > cut else Float32(0)
        return self.k.mm(self.k.ew2(OP_MUL, V, inv), V, False, True)

    def fast_logdet(self, A: Mat) raises -> Float64:
        """`_fast_logdet` over `_slogdet` (getrf; logs summed as `total`)."""
        var n = A.r
        var lu = A.copy()
        var piv = List[Int32](length=max(n, 1), fill=Int32(0))
        var info = Mat(1, 1)
        self.k.lu(lu, piv, info)
        var diag = Mat(1, n)
        for i in range(n):
            diag.d[i] = lu.d[i * n + i]
        for i in range(n):
            if diag.d[i] == Float32(0):
                return _neg_inf()             # sign 0 -> -inf
        var neg = 0
        for i in range(n):
            if diag.d[i] < Float32(0):
                neg += 1
        for i in range(n):
            if Int(piv[i]) != i:
                neg += 1
        var t = self.k.total((self.k.ew1(OP_LOGS, self.k.ew1(OP_ABS, diag, 0.0), _FLT_MIN)))
        if neg % 2 != 0:
            return _neg_inf()
        return Float64(t.d[0])

    # ---- randomness
    def perm(mut self, n: Int) raises -> List[Int]:
        """`MinCovDet._perm`: a sort of the next Philox stream's draws."""
        self.draws += 1
        return argsort_values(self.k.rand(1, n, self.seed, 1000 + self.draws, 0))

    def perm_head_sorted(mut self, n: Int, h: Int) raises -> List[Int]:
        """`sorted(self._perm(n)[:h])`."""
        self.draws += 1
        return smallest_sorted(self.k.rand(1, n, self.seed, 1000 + self.draws, 0), h)

    # ---- the C-step
    def c_step(
        mut self, X: Mat, h: Int, n_iter: Int, has_init: Bool, loc0: Mat, cov0: Mat, want_dist: Bool
    ) raises -> Est:
        var iters = n_iter
        var dist = Mat(0, 0)
        var sel: List[Int]
        if not has_init:
            sel = self.perm_head_sorted(X.r, h)
        else:
            var P0 = self.pinvh(cov0)
            dist = self.mahal(X, loc0, P0)
            sel = smallest_sorted(dist, h)
        var Xs = take_rows(X, sel)
        var loc = self.colmean(Xs)
        var cov = self.emp_cov(Xs)
        var det = self.fast_logdet(cov)
        var P = Mat(0, 0)
        var has_p = False
        if det == _neg_inf():
            P = self.pinvh(cov)
            has_p = True
        var prev_det = _pos_inf()
        var have_prev = False
        var prev_loc = Mat(0, 0)
        var prev_cov = Mat(0, 0)
        var prev_sel = List[Int]()
        while det < prev_det and iters > 0 and det != _neg_inf():
            prev_loc = loc.copy()
            prev_cov = cov.copy()
            prev_sel = sel.copy()
            have_prev = True
            prev_det = det
            P = self.pinvh(cov)
            has_p = True
            dist = self.mahal(X, loc, P)
            sel = smallest_sorted(dist, h)
            Xs = take_rows(X, sel)
            loc = self.colmean(Xs)
            cov = self.emp_cov(Xs)
            det = self.fast_logdet(cov)
            iters -= 1
        if not has_p:
            # Python's final `_mahal(k, X, loc, None)` fails here (a +inf or
            # NaN log det at the first step); so does this.
            raise Error("x_decomp MinCovDet: the first C-step's log determinant is not finite")
        # sklearn's checks in its order, the last one that fires wins
        var use_prev = have_prev and det > prev_det
        if iters == 0:
            use_prev = False
        if use_prev:
            return Est(prev_loc^, prev_cov^, prev_det, prev_sel^, dist^)
        var final = Mat(0, 0)
        if want_dist:
            final = self.mahal(X, loc, P)
        return Est(loc^, cov^, det, sel^, final^)

    def select_random(mut self, X: Mat, h: Int, trials: Int, keep: Int, n_iter: Int, want_dist: Bool) raises -> List[Est]:
        var est = List[Est]()
        var none = Mat(0, 0)
        for _ in range(trials):
            est.append(self.c_step(X, h, n_iter, False, none, none, want_dist))
        return Self._top(est, keep)

    def select_init(
        mut self, X: Mat, h: Int, inits: List[Est], keep: Int, n_iter: Int, want_dist: Bool
    ) raises -> List[Est]:
        var est = List[Est]()
        for t in range(len(inits)):
            est.append(self.c_step(X, h, n_iter, True, inits[t].loc, inits[t].cov, want_dist))
        return Self._top(est, keep)

    @staticmethod
    def _top(est: List[Est], keep: Int) raises -> List[Est]:
        var order = _order_by_det(est, keep)
        var out = List[Est]()
        for a in range(len(order)):
            out.append(est[order[a]].copy())
        return out^


def fast_mcd[E: Exec](
    X: Mat, p: List[Int], loc_out: F32Ptr, cov_out: F32Ptr, sup_out: I32Ptr, dist_out: F32Ptr,
) raises:
    """`MinCovDet._fast_mcd` for p >= 2. `p` = [n, n_features, h, seed,
    n_sub, n_ss, h_sub, n_trials, n_m, h_m, n_best_m], every integer the
    Python caller computed from n, p and h (the same float expressions it
    always used). Writes location (p), covariance (p x p), the support mask
    (n int32, 0/1) and the distances (n)."""
    var n = p[0]
    var d = p[1]
    var h = p[2]
    var run = Mcd[E](p[3])
    var best: Est
    var support = List[Int32](length=n, fill=Int32(0))
    var dist = List[Float32](length=n, fill=Float32(0))
    if n > 500:
        var n_sub = p[4]
        var n_ss = p[5]
        var h_sub = p[6]
        var n_trials = p[7]
        var n_m = p[8]
        var h_m = p[9]
        var n_best_m = p[10]
        var shuf = run.perm(n)
        var pool = List[Est]()
        for i in range(n_sub):
            var rows = List[Int](capacity=n_ss)
            for a in range(i * n_ss, (i + 1) * n_ss):
                rows.append(shuf[a])
            var cur = take_rows(X, rows)
            var got = run.select_random(cur, h_sub, n_trials, 10, 2, False)
            for e in range(len(got)):
                pool.append(got[e].copy())
        var selection_all = run.perm(n)
        var selection = List[Int](capacity=n_m)
        for a in range(n_m):
            selection.append(selection_all[a])
        var Xm = take_rows(X, selection)
        var merged = run.select_init(Xm, h_m, pool, n_best_m, 30, n < 1500)
        if n < 1500:
            ref m0 = merged[0]
            for a in range(len(selection)):
                dist[selection[a]] = m0.dist.d[a]
            for a in range(len(m0.sel)):
                support[selection[m0.sel[a]]] = Int32(1)
            _write(m0.loc, m0.cov, support, dist, d, loc_out, cov_out, sup_out, dist_out)
            return
        var full = run.select_init(X, h, merged, 1, 30, True)
        best = full[0].copy()
    else:
        var first = run.select_random(X, h, 30, 10, 2, False)
        var full = run.select_init(X, h, first, 1, 30, True)
        best = full[0].copy()
    for a in range(len(best.sel)):
        support[best.sel[a]] = Int32(1)
    for i in range(n):
        dist[i] = best.dist.d[i]
    _write(best.loc, best.cov, support, dist, d, loc_out, cov_out, sup_out, dist_out)


def _write(
    loc: Mat, cov: Mat, support: List[Int32], dist: List[Float32], d: Int, loc_out: F32Ptr,
    cov_out: F32Ptr, sup_out: I32Ptr, dist_out: F32Ptr,
):
    for j in range(d):
        loc_out.unsafe_store(j, loc.d[j])
    for j in range(d * d):
        cov_out.unsafe_store(j, cov.d[j])
    for i in range(len(support)):
        sup_out.unsafe_store(i, support[i])
        dist_out.unsafe_store(i, dist[i])
