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
from std.sys.compile import is_defined
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.accelerator_target import TARGET_HAS_APPLE_GPU

from x_decomp.cells import F32Ptr, I32Ptr
from x_decomp.exec_trait import Exec
from x_decomp.kit import Kit, Mat, OP_ABS, OP_LOGS, OP_MUL, OP_RECIP, OP_SCALE, OP_SUB, take_rows

# lane apple-holes (2026-10-12), IDENTICAL: the C-step's log determinant and
# Mahalanobis distances by the blocked Cholesky (cholesky/checks/potrf.mojo
# `potrf_lower`, profile mojolearn.identical.cholesky.fp32.v1; `chol_logdet`;
# L^{-1} by `trsm_lower` against I; d = rowsum((Xc L^{-T})^2)) instead of a
# partial-pivot LU for the determinant and a round-robin Jacobi eigh for
# pinvh on every C-step. Cause: at p = 220 the Jacobi is ~219 rounds x 2
# launches a sweep plus the LU's ~5 launches a column, thousands of launches
# per C-step; on Apple (per-launch cost ~20 us) MinCovDet identical took
# 225 s at 2,000 rows and hit the board's race timeout at 100,000. The
# Cholesky is ~10 launches a 32-column panel. A covariance whose Cholesky
# stops (not positive definite) keeps the old rule: log det -inf, pinvh by
# eigh. Semantics: pinvh dropped eigen-directions below max|w| p eps32 (an
# absolute cut that removes small-scale features of a badly scaled
# covariance); the Cholesky keeps them, as scikit-learn's float64 pinvh does.
# Bits change (all columns: this host twin runs chol_oracle's spelling of the
# same profile). No size rule.
#
# DEFAULT OFF since lane restore-classical-measured-routes (2026-11-13): this
# switch went in (2cb89eb1b9) as the IDENTICAL default with its NV + AMD A/B
# owed and never run. The board on record: min-cov-det taxi fit AMD 2116 ms
# (0.8.25, LU + Jacobi pinvh C-step) -> 9852 ms (0.8.42 a6b5f76cb, after
# this switch), NVIDIA 2871 -> 3420 ms; elliptic-envelope taxi AMD 2141 ->
# 9760 ms; istella went the other way (AMD 268.9 s @1658ba796 -> 243.4 s
# @a6b5f76cb, NVIDIA 186.2 s @12412eca4 -> 168.8 s @4c75b4144). One cell
# each, other changes in between (history-audit hist-classical rank 2). The
# measured LU + round-robin Jacobi pinvh C-step (0.8.25 .. 0.8.41) is the
# default again; `-D MOJOLEARN_IDN_MCD_CHOL` is the Cholesky C-step, the B
# arm of the owed A/B (NV + AMD, min-cov-det and elliptic-envelope taxi AND
# istella, plus one non-board shape; owed row mcd-chol in
# bench/results/paper_compare/owed-pre0867.json). Flipping the default
# changes MinCovDet / EllipticEnvelope IDENTICAL bits on every column
# together (host, Apple, NVIDIA, AMD). The `_OFF` define is gone (it would
# select the default). A size rule only if each arm wins on a range of
# shapes (no dimension targeting).
# TOMBSTONE: MOJOLEARN_IDN_MCD_CHOL (opt-in arm: the Cholesky C-step, EXP:581 RUN OWED, no A/B recorded) deleted 2026-11-15 by lane/expdel2-17-decomposition; code recoverable at c387c0b2c1.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_MCD_CHOL.patch; record in docs/TOMBSTONES.md.
# RESTORED 2026-11-15 by lane/expdel2-restore-owed: owed B arm (owed-pre0867.json), under four weeks old; the deletion was premature.
# (MOJOLEARN_IDN_ALL_OFF stays deleted, so its term is not restored.)
# A/B 2026-11-18 (ab1, owed row mcd-chol, NVIDIA L40S host 37f71a6102af, driver 595.112.07, wheel d652a4b0a1,
# base and arm back to back in one job, one run per arm): the Cholesky arm is FASTER, arm/base min-cov-det taxi
# fit 6199 -> 3422 ms (0.55), elliptic-envelope taxi fit 6188 -> 3386 (0.55), min-cov-det istella 125.6 ->
# 107.5 s (0.86), elliptic-envelope istella 127.5 -> 106.8 s (0.84), elliptic-envelope istella infer 25.83 ->
# 20.26 ms (0.78), taxi infer 1.188 -> 1.229 (1.03). QUALITY IS LOWER on one cell: elliptic-envelope taxi support
# Jaccard vs scikit-learn 0.9636 (LU) -> 0.9258 (Cholesky; fraction flagged 0.10237 -> 0.10209); istella equal
# (same digest c8f7d3738967a583, Jaccard 0.0622 both); min-cov-det records no quality metric (n_features only).
# The Cholesky taxi output is 0.8.55's exactly (digest 8c1b478088f596bf, Jaccard 0.9258, stored 0.8.55 board of
# the 2026-11-17 full board). Hashes differ on 3 of 6 cells (the taxi cells). HELD, NOT FLIPPED: the owed row
# promotes B only with Jaccard not lower, and AGENTS.md never trades quality for speed; a decision for Andrew.
comptime IDN_MCD_CHOL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_IDN_MCD_CHOL"]()

# Experiment MOJOLEARN_MCD_FAST_LOCKSTEP (lane fx-board-covariance-nb, 2026-11-19; decided by A/B round 3, below: owed row
# mcd-lockstep in bench/results/paper_compare/owed-pre0867.json, docs/apple-fast/EXPERIMENTS.md). IDENTICAL, GPU
# columns only; the host column and every word are unchanged (no bit change on any column).
# Cause it addresses: the resident search (x_decomp/kit_device.mojo `DMcd.select_random` / `select_init`) runs its
# candidates ONE AT A TIME, and every C-step of every candidate waits once for its log determinant and runs its
# narrow pinvh as a one-block Jacobi with a batch of one, so the device works on one d x d problem at a time: about
# 1,650 log-determinant waits and 1,100 single-block eigen solves a fit whatever the row count (the plan's 500 random
# trials x up to 3, 50 merged candidates, 10 final ones). That is the cost the Cholesky C-step (IDN_MCD_CHOL above)
# removes by changing the math (and the support: Jaccard 0.9636 -> 0.9258 on its A/B). This arm keeps the LU + Jacobi
# pinvh math and the words and changes only the ORDER OF LAUNCHES: the candidates of one `select_random` /
# `select_init` call take each C-step together. Per round: one batched Jacobi launch over every still-running
# candidate's covariance (`DevExec._eigh_batch_enq`, one block a candidate: the same kernel and words as the batch of
# one), each candidate's distances, selection, gather, mean, covariance and LU enqueued, then ONE wait for all their
# log determinants. Candidates are independent (a trial's draws are its own Philox stream, numbered in trial order
# as before), so each candidate's values are the serial loop's values. Applies up to XD_EIGH_ONE_BLOCK_N columns (the
# one-block solve's bound: above it the Jacobi is multi-block and already fills the device); `select_init_best` (ten
# candidates, support rows and distances kept whole) stays the serial loop.
# Estimate (not measured): taxi-like fits (d 11-16, positive definite) lose most of the 2.8 s that separates the two
# C-steps on the L40S (ab1: 6199 vs 3422 ms) plus the per-C-step waits of the other half. Time beside the board rows:
# a neighbouring shape (300,000 x 16 rows of random float32 with 5 percent outliers) and a non-board dataset
# (covtype's 10 continuous columns). Apple: a round enqueues up to trials x about 60 launches before its wait; the
# Apple column is not timed (AGENTS.md part 8) and must be smoke-run before any promotion.
# A/B round 3, 2026-11-20 (lane fx-verdicts5; NVIDIA L40S host 2ef132a6de04, wheel 28a780604c and its B set, base and
# arm back to back in one pod job, tag nvc-ab-20261120T063324Z-c630ad, workers bound to the GPU's NUMA node 0; one
# scored run per arm; digests and quality equal on all 6 cells). Base -> arm ms:
# min-cov-det taxi fit 6,009.9 -> 4,269.0 (0.710), elliptic-envelope taxi fit 6,015 -> 4,299 (0.715; Jaccard vs
# scikit-learn 0.9636 both, unlike the Cholesky arm), taxi infer 0.863 -> 0.896 (1.04, median of 20, spread
# 0.85..1.78 / 0.84..2.65: noise, the arm does not touch predict); istella (d = 220, above the one-block bound, so
# the serial loop on both arms) fit 122,670 -> 122,650 (1.000) and 122,770 -> 122,910 (1.001), infer 1.006.
# PROMOTED on the NVIDIA and AMD columns: DEFAULT ON there since 2026-11-20 (-D MOJOLEARN_MCD_FAST_LOCKSTEP_OFF
# builds the serial candidate loop). The Apple column keeps the serial loop (a round enqueues up to trials x about
# 60 launches before its wait and was never run on Metal); there the old name MOJOLEARN_MCD_FAST_LOCKSTEP still
# opts in, for the owed smoke. The words are the same in either order, so the columns agree whichever runs.
# Not timed: the neighbouring shape and the non-board dataset named above.
comptime MCD_LOCKSTEP = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not is_defined["MOJOLEARN_MCD_FAST_LOCKSTEP_OFF"]()
    and (not TARGET_HAS_APPLE_GPU or is_defined["MOJOLEARN_MCD_FAST_LOCKSTEP"]())
    and not IDN_MCD_CHOL
)

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
    distances). `owed`: the final distances are not computed yet
    (`Mcd.finish` runs them from P, the precision the loop last used
    (L^{-1} when `p_chol`), or from pinvh(cov) when P is empty)."""
    var loc: Mat
    var cov: Mat
    var det: Float64
    var sel: List[Int]
    var dist: Mat
    var P: Mat
    var p_chol: Bool
    var owed: Bool

    def __init__(
        out self, var loc: Mat, var cov: Mat, det: Float64, var sel: List[Int], var dist: Mat, var P: Mat,
        p_chol: Bool, owed: Bool,
    ):
        self.loc = loc^
        self.cov = cov^
        self.det = det
        self.sel = sel^
        self.dist = dist^
        self.P = P^
        self.p_chol = p_chol
        self.owed = owed


struct Chol(Movable):
    """IDN_MCD_CHOL: a covariance's Cholesky log determinant (-inf when the
    factorization stopped) and L^{-1} (empty then)."""
    var det: Float64
    var li: Mat

    def __init__(out self, det: Float64, var li: Mat):
        self.det = det
        self.li = li^


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

    def emp_cov_at(self, Xs: Mat, loc: Mat) raises -> Mat:
        # TOMBSTONE: MOJOLEARN_CLASSICAL_C23_MCD (its use here: the centered-Gram covariance arm) deleted 2026-11-15 by lane/expdel2-17-decomposition (owner lane/expdel2-06-base); code recoverable at 7b5151da9b.
        # Restore: git apply experiments/removed/MOJOLEARN_CLASSICAL_C23_MCD.expdel2-17-decomposition.patch.
        # TOMBSTONE: MOJOLEARN_C57_CANDIDATE_STATE (its use here: the covariance centred at the candidate's mean) deleted 2026-11-15 by lane/expdel2-17-decomposition (owner lane/expdel2-07-base); code recoverable at 3b6b933bcc.
        # Restore: git apply experiments/removed/MOJOLEARN_C57_CANDIDATE_STATE.expdel2-17-decomposition.patch.
        return self.emp_cov(Xs)

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

    # ---- IDN_MCD_CHOL: the Cholesky C-step's pieces (host column)
    def chol(self, A: Mat) raises -> Chol:
        """`potrf_lower` (+0.0 ridge, as `add_jitter`), `chol_logdet` and
        L^{-1} = `trsm_lower(L, I)` through the executor (`Exec.chol_inv`)."""
        var n = A.r
        var li = Mat(n, n)
        var st = Mat(1, 2)
        self.k.chol_inv(A, li, st)
        if Int(st.d[0]) != 0:
            return Chol(_neg_inf(), Mat(0, 0))
        return Chol(Float64(st.d[1]), li^)

    def mahal_chol(self, X: Mat, loc: Mat, Li: Mat) raises -> Mat:
        """rowsum((Xc L^{-T})^2): the distances under P = L^{-T} L^{-1}."""
        var Xc = self.k.ew2(OP_SUB, X, loc)
        var Y = self.k.mm(Xc, Li, False, True)
        return self.k.rowsum(self.k.ew2(OP_MUL, Y, Y))

    def dist_from(self, X: Mat, loc: Mat, cov: Mat) raises -> Mat:
        """The distances under cov: its Cholesky when it factors, pinvh
        otherwise."""
        var f = self.chol(cov)
        if f.det != _neg_inf():
            return self.mahal_chol(X, loc, f.li)
        return self.mahal(X, loc, self.pinvh(cov))

    def finish(self, X: Mat, mut e: Est) raises:
        """A C-step's final distances, owed since `c_step` (lane
        mcd-per-step-cost, 2026-11-14; x_decomp/kit_device.mojo
        `DMcd.finish` is the GPU twin): the same calls on the same operands
        as before, run only for a candidate whose distances are read."""
        if not e.owed:
            return
        if e.P.r == 0:
            e.dist = self.mahal(X, e.loc, self.pinvh(e.cov))
        elif e.p_chol:
            e.dist = self.mahal_chol(X, e.loc, e.P)
        else:
            e.dist = self.mahal(X, e.loc, e.P)
        e.P = Mat(0, 0)
        e.owed = False

    def c_step_chol(
        mut self, X: Mat, h: Int, n_iter: Int, has_init: Bool, loc0: Mat, cov0: Mat
    ) raises -> Est:
        """`c_step` with the Cholesky's log determinant and distances
        (IDN_MCD_CHOL): the same control flow, statement for statement; the
        final distances owed (`finish`)."""
        var iters = n_iter
        var dist = Mat(0, 0)
        var sel: List[Int]
        if not has_init:
            sel = self.perm_head_sorted(X.r, h)
        else:
            dist = self.dist_from(X, loc0, cov0)
            sel = smallest_sorted(dist, h)
        var Xs = take_rows(X, sel)
        var loc = self.colmean(Xs)
        var cov = self.emp_cov_at(Xs, loc)
        var f = self.chol(cov)
        var det = f.det
        # P empty with has_p: a -inf start, pinvh(cov) owed to `finish`
        var P = Mat(0, 0)
        var p_chol = False
        var has_p = False
        if det == _neg_inf():
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
            P = f.li.copy()                   # det is finite: cov factored
            p_chol = True
            has_p = True
            dist = self.mahal_chol(X, loc, P)
            sel = smallest_sorted(dist, h)
            Xs = take_rows(X, sel)
            loc = self.colmean(Xs)
            cov = self.emp_cov_at(Xs, loc)
            f = self.chol(cov)
            det = f.det
            iters -= 1
        if not has_p:
            raise Error("x_decomp MinCovDet: the first C-step's log determinant is not finite")
        var use_prev = have_prev and det > prev_det
        if iters == 0:
            use_prev = False
        if use_prev:
            return Est(prev_loc^, prev_cov^, prev_det, prev_sel^, dist^, Mat(0, 0), False, False)
        return Est(loc^, cov^, det, sel^, Mat(0, 0), P^, p_chol, True)

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
        mut self, X: Mat, h: Int, n_iter: Int, has_init: Bool, loc0: Mat, cov0: Mat
    ) raises -> Est:
        """One C-step search from a random subset or an initial estimate;
        the final distances owed to `finish` (lane mcd-per-step-cost: a -inf
        start's pinvh(cov) and the final distances were computed for every
        candidate and read for the kept one only)."""
        comptime if IDN_MCD_CHOL:
            return self.c_step_chol(X, h, n_iter, has_init, loc0, cov0)
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
        var cov = self.emp_cov_at(Xs, loc)
        var det = self.fast_logdet(cov)
        # P empty with has_p: a -inf start, pinvh(cov) owed to `finish`
        var P = Mat(0, 0)
        var has_p = False
        if det == _neg_inf():
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
            cov = self.emp_cov_at(Xs, loc)
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
            return Est(prev_loc^, prev_cov^, prev_det, prev_sel^, dist^, Mat(0, 0), False, False)
        return Est(loc^, cov^, det, sel^, Mat(0, 0), P^, False, True)

    def select_random(mut self, X: Mat, h: Int, trials: Int, keep: Int, n_iter: Int, want_dist: Bool) raises -> List[Est]:
        var est = List[Est]()
        var none = Mat(0, 0)
        for _ in range(trials):
            est.append(self.c_step(X, h, n_iter, False, none, none))
        return self._kept(X, est, keep, want_dist)

    def select_init(
        mut self, X: Mat, h: Int, inits: List[Est], keep: Int, n_iter: Int, want_dist: Bool
    ) raises -> List[Est]:
        var est = List[Est]()
        for t in range(len(inits)):
            est.append(self.c_step(X, h, n_iter, True, inits[t].loc, inits[t].cov))
        return self._kept(X, est, keep, want_dist)

    def _kept(self, X: Mat, est: List[Est], keep: Int, want_dist: Bool) raises -> List[Est]:
        """`_top`, then the kept candidates' final distances when read."""
        var out = Self._top(est, keep)
        if want_dist:
            for a in range(len(out)):
                self.finish(X, out[a])
        return out^

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
