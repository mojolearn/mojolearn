# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The decomp lane's Python entry points, written ONCE and instantiated for
both executors: `bindings/_mojolearn_x_decomp.mojo` registers `*_py[DevExec]`,
`bindings/_mojolearn_x_decomp_host.mojo` registers `*_py[HostExec]` under the
same names, so the GPU binding and the CPU host binding share the address
contract by construction. Every address is a host buffer the caller owns."""
from std.memory import bitcast
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.sys.compile import is_defined

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, NUMERIC_FAST
from x_decomp.cells import F32Ptr, I32Ptr
from x_decomp.exec_trait import Exec
from x_decomp.kit import mat_from
from x_decomp.mcd import fast_mcd
from x_decomp.lda_online import lda_online_pass
from x_decomp.moves import (
    F64Ptr, PY2MOJO_DECOMP, accuracy, argmin_all, argsort_f32, dsum_sq_host, gather, iso_order, move_host, order_f,
    scatter, select_smallest, sign_labels, topn_desc, triu_nonzero,
)
from x_decomp.tsqr_core import TS_MAX_N
from x_decomp.pca_mle import mle_scratch_words, pca_mle_rank_host
from x_decomp.api_pyapi import _d, _f, _i, _n


def _fp(a: Int) raises -> F32Ptr:
    """A float32 host buffer from its address (the typed `_f`)."""
    if a == 0:
        raise Error("x_decomp: null float32 buffer address")
    return F32Ptr(unsafe_from_address=a)


def _ip(a: Int) raises -> I32Ptr:
    """An int32 host buffer from its address (the typed `_i`)."""
    if a == 0:
        raise Error("x_decomp: null int32 buffer address")
    return I32Ptr(unsafe_from_address=a)


def _dp(a: Int) raises -> F64Ptr:
    """A float64 host buffer from its address (the typed `_d`)."""
    if a == 0:
        raise Error("x_decomp: null float64 buffer address")
    return F64Ptr(unsafe_from_address=a)


def gemm_run[E: Exec](pa: F32Ptr, pb: F32Ptr, pc: F32Ptr, m: Int, k: Int, n: Int, ta: Bool, tb: Bool) raises -> Int:
    """`gemm_py`'s typed entry: the Int32 bound, then Exec.gemm; returns m n."""
    if m * n > 2147483647 or m * k > 2147483647 or k * n > 2147483647:
        raise Error("x_decomp: gemm exceeds the Int32 index bound")
    E.gemm(pa, pb, pc, m, k, n, ta, tb)
    return m * n


def ew_run[E: Exec](
    op: Int, pa: F32Ptr, pb: F32Ptr, lb: Int, bm: Int, pc: F32Ptr, lc: Int, cm: Int, po: F32Ptr, count: Int, d: Int,
    sv: Float32,
) raises:
    """`ew_py`'s typed entry: the row-width check, then Exec.ew."""
    if d <= 0:
        raise Error("x_decomp: ew needs a positive row width")
    E.ew(op, pa, pb, lb, bm, pc, lc, cm, po, count, d, sv)


def sqdist_run[E: Exec](
    pa: F32Ptr, pb: F32Ptr, po: F32Ptr, na: Int, nb: Int, d: Int, kind: Int, pw: Float32
) raises -> Int:
    """`sqdist_py`'s typed entry: the Int32 bound, then Exec.sqdist; returns na nb."""
    if na * nb > 2147483647:
        raise Error("x_decomp: sqdist exceeds the Int32 index bound")
    E.sqdist(pa, pb, po, na, nb, d, kind, pw)
    return na * nb


def rand_run[E: Exec](po: F32Ptr, count: Int, seed: Int, stream: Int, kind: Int) raises:
    """`rand_py`'s typed entry: seed and stream taken mod 2^32, then Exec.rand."""
    var s = UInt32(seed & 0xFFFFFFFF)
    var st = UInt32(stream & 0xFFFFFFFF)
    E.rand(po, count, s, st, kind)


def trisolve_run[E: Exec](pl: F32Ptr, pi: F32Ptr, ps: F32Ptr, pd: F32Ptr, n: Int, nrhs: Int, trans: Int) raises:
    """`trisolve_py`'s typed entry: `trisolve_serial` (x_decomp/cells.mojo)."""
    if n >= 1 << 24:
        raise Error("x_decomp: trisolve row numbers exceed float32's exact integers")
    E.trisolve(pl, pi, ps, pd, n, nrhs, trans)


def knn_select_run[E: Exec](pd: F32Ptr, ps: F32Ptr, pi: F32Ptr, n: Int, m: Int, k: Int, ex: Int) raises:
    """`knn_select_py`'s typed entry: `knn_select_row` for every row."""
    if m >= 1 << 24:
        raise Error("x_decomp: knn_select columns exceed float32's exact integers")
    E.knn_select(pd, ps, pi, n, m, k, ex)


def lu_solve_run[E: Exec](pl: F32Ptr, pp: I32Ptr, pb: F32Ptr, n: Int, nrhs: Int, trans: Int) raises:
    """`lu_solve_py`'s typed entry: the exact-integer check, then Exec.lu_solve."""
    if n >= 1 << 24:
        raise Error("x_decomp: lu_solve row numbers exceed float32's exact integers")
    E.lu_solve(pl, pp, pb, n, nrhs, trans)


def eigh_run[E: Exec](pa: F32Ptr, pw: F32Ptr, pv: F32Ptr, n: Int, uplo: Int) raises:
    """`eigh_py`'s typed entry. uplo: numpy's UPLO, 1 the lower triangle, 2
    the upper, 0 the whole matrix; mirrored by the executor (on the device)."""
    if n <= 0:
        raise Error("x_decomp: eigh needs n >= 1")
    if uplo < 0 or uplo > 2:
        raise Error("x_decomp: eigh uplo is 0, 1 (L) or 2 (U)")
    E.eigh(pa, pw, pv, n, uplo)


def lle_apply_run[E: Exec](
    pw: F32Ptr, pi: F32Ptr, pe: F32Ptr, po: F32Ptr, nq: Int, nf: Int, nn: Int, nc: Int
) raises:
    """`lle_apply_py`'s typed entry: LLE transform's out (nq x nc) = W E[idx]."""
    if nq * nc > 2147483647 or nf * nc > 2147483647 or nq * nn > 2147483647:
        raise Error("x_decomp: lle_apply exceeds the Int32 index bound")
    E.lle_apply(pw, pi, pe, po, nq, nf, nn, nc)


def lle_local_run[E: Exec](
    px: F32Ptr, pi: F32Ptr, pb: F32Ptr, method: Int, n: Int, d: Int, nn: Int, nc: Int, tol: Float32
) raises:
    """`lle_local_py`'s typed entry: LocallyLinearEmbedding's stacked factor
    (x_decomp/lle_local.mojo) after its shape checks."""
    if method < 0 or method > 2 or n < 1 or d < 1 or nn < 1 or nc < 1 or (method == 1 and nn - 1 - nc < 1):
        raise Error("x_decomp: lle_local needs method 0..2, n, d, n_neighbors, n_components >= 1 (hessian: n_neighbors > n_components + 1)")
    if n * nn * n > 2147483647 or n * nn * nn > 2147483647 or n >= 16777216:
        raise Error("x_decomp: lle_local exceeds the Int32 index bound")
    E.lle_local(px, pi, pb, method, n, d, nn, nc, tol)


def eigh_batch_run[E: Exec](pa: F32Ptr, pw: F32Ptr, pv: F32Ptr, batch: Int, n: Int) raises:
    """`eigh_batch_py`'s typed entry: `batch` n x n symmetric problems
    (x_decomp/rr_batch.mojo) after the shape checks."""
    if n <= 0 or batch < 0:
        raise Error("x_decomp: eigh_batch needs n >= 1, batch >= 0")
    if batch * n * n > 2147483647:
        raise Error("x_decomp: eigh_batch exceeds the Int32 index bound")
    E.eigh_batch(pa, pw, pv, batch, n)


def svd_run[E: Exec](pa: F32Ptr, ps: F32Ptr, pv: F32Ptr, m: Int, n: Int) raises:
    """`svd_py`'s typed entry: a tall matrix (m >= n >= 1), then Exec.svd."""
    if n <= 0 or m < n:
        raise Error("x_decomp: svd needs m >= n >= 1 (a tall matrix)")
    E.svd(pa, m, n, ps, pv)


def lasso_rows_run[E: Exec](
    pg: F32Ptr, pq: F32Ptr, pw: F32Ptr, pi: F32Ptr, n: Int, k: Int, max_iter: Int, positive: Bool, alpha: Float32,
    tol: Float32,
) raises:
    """`lasso_rows_py`'s typed entry: the n k scratch, then Exec.lasso_rows."""
    var h = List[Float32](length=n * k if n * k > 0 else 1, fill=Float32(0))
    var ph = F32Ptr(unsafe_from_address=Int(h.unsafe_ptr()))
    E.lasso_rows(pg, pq, pw, ph, pi, n, k, alpha, max_iter, tol, positive)
    _ = h^


def lu_aux_run[E: Exec](
    pl: F32Ptr, pv: I32Ptr, p1: F32Ptr, p2: F32Ptr, pd: F32Ptr, ps: F32Ptr, n: Int, clamp: Int
) raises:
    """`lu_aux_py`'s typed entry: 1 <= n < 2^24, then Exec.lu_aux."""
    if n < 1 or n >= 16777216:
        raise Error("x_decomp: lu_aux needs 1 <= n < 2^24")
    E.lu_aux(pl, pv, p1, p2, pd, ps, n, clamp)


def lars_rows_run[E: Exec](pg: F32Ptr, pq: F32Ptr, pw: F32Ptr, pn: F32Ptr, n: Int, k: Int, m: Int, nnz: Int) raises:
    """`lars_rows_py`'s typed entry: the Int32 bound, then Exec.lars_rows."""
    if n * (k * k + 7 * k) > 2147483647:
        raise Error("x_decomp: lars_rows exceeds the Int32 index bound")
    E.lars_rows(pg, pq, pw, pn, n, k, m, nnz)


def omp_rows_run[E: Exec](pg: F32Ptr, pq: F32Ptr, pw: F32Ptr, pn: F32Ptr, n: Int, k: Int, nnz: Int) raises:
    """`omp_rows_py`'s typed entry: the n (k^2 + 3 k) scratch, then Exec.omp_rows."""
    var per = k * k + 3 * k
    var s = List[Float32](length=n * per if n * per > 0 else 1, fill=Float32(0))
    var ps = F32Ptr(unsafe_from_address=Int(s.unsafe_ptr()))
    E.omp_rows(pg, pq, pw, ps, pn, n, k, nnz)
    _ = s^


def rand_gamma_run[E: Exec](po: F32Ptr, count: Int, seed: Int, stream: Int, a: Float32) raises:
    """`rand_gamma_py`'s typed entry: seed and stream mod 2^32, shape >= 1."""
    var s = UInt32(seed & 0xFFFFFFFF)
    var st = UInt32(stream & 0xFFFFFFFF)
    if not (a >= Float32(1)):
        raise Error("x_decomp: the gamma sampler takes shape >= 1")
    E.rand_gamma(po, count, s, st, a)


def lda_rows_run[E: Exec](
    px: F32Ptr, pw: F32Ptr, pd: F32Ptr, pe: F32Ptr, pi: F32Ptr, n: Int, k: Int, v: Int, max_iter: Int,
    prior: Float32, tol: Float32,
) raises:
    """`lda_rows_py`'s typed entry: the n (v + k) scratch, then Exec.lda_rows."""
    var s = List[Float32](length=n * (v + k) if n > 0 else 1, fill=Float32(0))
    var ps = F32Ptr(unsafe_from_address=Int(s.unsafe_ptr()))
    E.lda_rows(px, pw, pd, pe, ps, pi, n, k, v, prior, max_iter, tol)
    _ = s^


def absmax_sign_run[E: Exec](pa: F32Ptr, pd: F32Ptr, n: Int, d: Int, by_col: Bool) raises -> Int:
    """`absmax_sign_py`'s typed entry; returns d by column, else n."""
    E.absmax_sign(pa, pd, n, d, by_col)
    return d if by_col else n


def qr_r_run[E: Exec](pa: F32Ptr, pr: F32Ptr, m: Int, n: Int) raises:
    """`qr_r_py`'s typed entry: m >= n >= 1, then Exec.qr_r."""
    if n <= 0 or m < n:
        raise Error("x_decomp: qr_r needs m >= n >= 1")
    E.qr_r(pa, m, n, pr)


def tsqr_r_run[E: Exec](pa: F32Ptr, pb: F32Ptr, pr: F32Ptr, m: Int, d: Int, nrhs: Int, keep: Bool) raises -> Int:
    """`tsqr_r_py`'s typed entry: the shape checks, then Exec.tsqr_factor;
    returns n = d + nrhs."""
    var n = d + nrhs
    if d < 1 or n > TS_MAX_N or m < n:
        raise Error("x_decomp: tsqr_r needs 1 <= d, d + nrhs <= " + String(TS_MAX_N) + " and m >= d + nrhs")
    if m * n > 2147483647:
        raise Error("x_decomp: tsqr_r exceeds the Int32 index bound")
    E.tsqr_factor(pa, pb, pr, m, d, nrhs, keep)
    return n


def r_signs_run(pr: F32Ptr, pc: F32Ptr, n: Int):
    """Lane py-runtime (2026-10-05): R's diagonal made non-negative for
    `linalg.qr` (it was Python in `_linalg_impl._qr_tsqr`). r is n x n row
    major and is rewritten: every row j whose R[j, j] < 0 is negated (an
    exact sign flip; -0.0 and NaN count as non-negative, as the Python `<`
    did). c (n x n) gets the diagonal of the signs, zeros elsewhere: Q C is
    then the matching Q. Host Mojo on every column: n^2 sign moves
    after the TSQR, no arithmetic."""
    for j in range(n):  # small-loop(n: R order, the feature count): one sign per R row
        var neg = pr[j * n + j] < 0.0
        for t in range(n):  # small-loop(n: R order, the feature count): one row of R and of C
            if neg:
                pr[j * n + t] = -pr[j * n + t]
            pc[j * n + t] = Float32(0.0)
        pc[j * n + j] = Float32(-1.0) if neg else Float32(1.0)


def rank_above_run(sv: Int, n: Int, rtol: Float64) raises -> Int:
    """Lane py-runtime round 2: how many of the n float32 singular values at
    address sv (descending, s[0] the largest) are > 0 and > s[0] * rtol, the
    product and the comparisons in binary64 as Python's
    `sum(1 for v in S.s if v > 0.0 and v > s0 * rtol)` made them
    (`_linalg_impl._svd_tall`). sv is read only when n > 0."""
    if n == 0:
        return 0
    var ps = _fp(sv)
    var t = Float64(ps[0]) * rtol
    var r = 0
    for i in range(n):  # small-loop(n: singular value count): one comparison per singular value
        var v = Float64(ps[i])
        if v > 0.0 and v > t:
            r += 1
    return r

# lane idn-dense-linalg (2026-10-04): the one-entry routes Python takes when
# `idn_flags_py` says so (both bindings; the same words as the calls they
# replace). -D MOJOLEARN_IDN_LU_GESV_OFF clears its bit, and Python keeps
# the old call sequence. IDENTICAL builds only: a FAST build's bits are 0
# and its routes are as they were.
# Tried 2026-10-08 (MOJOLEARN_IDN_OLS_ONE_ENTRY_OFF, the ols_one_entry "off" arm, run ge123e6f9): the old four-crossing
# OLS sequence (_column_means + _center + _ols_tsqr) is NV/AMD 2.97x/1.74x SLOWER istella, 6.17x/3.83x taxi, r2 SAME, so
# the one-entry route is the only IDENTICAL OLS route (MOJOLEARN_IDN_ALL_OFF no longer clears it); the _OFF define is
# refused. Recoverable at main 42d1e42c6; row in docs/apple-fast/EXPERIMENTS.md.
comptime _API_IDN = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
comptime IDN_OLS_ONE_ENTRY = _API_IDN
# TOMBSTONE: MOJOLEARN_IDN_OLS_R_TRISOLVE (loser: noise, ols taxi/istella avg ~0.98x, bits changed) deleted 2026-10-11 by c14beeda5; code recoverable at c14beeda5^.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_OLS_R_TRISOLVE.patch
comptime IDN_LU_GESV = _API_IDN and not (is_defined["MOJOLEARN_IDN_LU_GESV_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())
# lane fam-decomp (2026-10-04): LLE builds F0 = [F^ | u] in three cells and
# takes F^ X as F0 [X; 0] on every column of an IDENTICAL build (device and
# host binding alike, so the four agree), instead of downloading F, slicing
# its columns in Python and uploading F^ again.
# -D MOJOLEARN_IDN_LLE_DEV_F0_OFF clears the bit: Python keeps the host
# F.cols + _hstack route.
comptime IDN_LLE_DEV_F0 = _API_IDN and not (is_defined["MOJOLEARN_IDN_LLE_DEV_F0_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())
# lane fam-decomp (2026-10-04): the kit's `rand` draws into a device matrix
# on the GPU binding (`dev_rand_py`, the kernel `DevExec.rand` launches: the
# same Philox words) and downloads it only if Python reads it, instead of
# downloading every draw and uploading it again for the first product (NMF's
# n x k init, randomized_svd's d x l test matrix, ALS's factors, MDS's
# starts). -D MOJOLEARN_IDN_RAND_RESIDENT_OFF clears the bit.
comptime IDN_RAND_RESIDENT = _API_IDN and not (is_defined["MOJOLEARN_IDN_RAND_RESIDENT_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())
# lane fam2-decomp (2026-10-04): a device-resident matrix's transpose, column
# selection and row range stay on the device (`x_decomp_dev_move`: TRANSPOSE,
# the new TAKE_COLS, TAKE_ROWS; one thread a moved value) when the result
# has at least 16,384 values, instead of downloading the operand (which then
# lives on the host) and uploading the result at its next product: NMF's
# W^T and H^T, FactorAnalysis, PLS, the svd tail's column selections, LLE.
# Copies only: the same words. -D MOJOLEARN_IDN_RES_MOVES_OFF clears the bit.
comptime IDN_RES_MOVES = _API_IDN and not (is_defined["MOJOLEARN_IDN_RES_MOVES_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())
# lane fix-d1-decomp (2026-10-04, audit F7/B7): the decomp Python leftovers.
# IDN_NMF_L2_CELL (bit 5): NMF's coordinate descent adds l2 to H H^T's
# diagonal with one `axpy` cell against the identity mask on EVERY column of
# an IDENTICAL build (device and host binding alike), instead of a host copy
# of H H^T and a Python loop. Bits move on all four together: l2 is rounded
# to float32 and added in the cell's fused step, and an off-diagonal -0.0
# becomes +0.0. -D MOJOLEARN_IDN_NMF_L2_CELL_OFF keeps the Python loop.
comptime IDN_NMF_L2_CELL = _API_IDN and not (is_defined["MOJOLEARN_IDN_NMF_L2_CELL_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())
# IDN_ICA_LIM_DEV (bit 6): FastICA's parallel convergence scalar
# max |abs(diag(W1 W^T)) - 1| is reduced on the device (`x_decomp_dev_maxabs`)
# and one word comes down, instead of k words and a Python max. A max is
# exact: the same value. -D MOJOLEARN_IDN_ICA_LIM_DEV_OFF.
comptime IDN_ICA_LIM_DEV = _API_IDN and not (is_defined["MOJOLEARN_IDN_ICA_LIM_DEV_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())
# IDN_POLAR_MAX_DEV (bit 7): varimax/quartimax `_polar`'s max |A| (it picks
# an exact power-of-two scale) is reduced on the device, one word down,
# instead of the k x k matrix and a Python max. The same value.
# -D MOJOLEARN_IDN_POLAR_MAX_DEV_OFF.
comptime IDN_POLAR_MAX_DEV = _API_IDN and not (is_defined["MOJOLEARN_IDN_POLAR_MAX_DEV_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())
# IDN_LLE_PAD_DEV (bit 8): LLE's shift-invert loop pads its resident X
# ((n - 1) x p) with a zero row on the device (PLACE_COLS + FILL0 moves),
# instead of downloading X and uploading [X; 0] twice per iteration. Copies
# only: the same words. -D MOJOLEARN_IDN_LLE_PAD_DEV_OFF.
comptime IDN_LLE_PAD_DEV = _API_IDN and not (is_defined["MOJOLEARN_IDN_LLE_PAD_DEV_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())
#: the binding exports `x_decomp_dev_maxabs` (GPU binding) when a user is on
comptime IDN_DEV_MAXABS = IDN_ICA_LIM_DEV or IDN_POLAR_MAX_DEV


def idn_flags_py() raises -> PythonObject:
    """Bit 0: LinearRegression takes `ols_tsqr_r_py`; bit 1: solve takes
    `lu_gesv_py`; bit 2: LLE builds F0 in cells (IDN_LLE_DEV_F0); bit 3: the
    kit's `rand` stays on the device (IDN_RAND_RESIDENT); bit 4: resident
    transpose / take_cols / rows (IDN_RES_MOVES); bit 5: NMF l2 by a cell
    (IDN_NMF_L2_CELL); bit 6: FastICA's limit on the device
    (IDN_ICA_LIM_DEV); bit 7: `_polar`'s max on the device
    (IDN_POLAR_MAX_DEV); bit 8: LLE's zero-row pad on the device
    (IDN_LLE_PAD_DEV); bit 9 was IDN_OLS_R_TRISOLVE (deleted 2026-10-11)."""
    var bits = 0
    comptime if IDN_NMF_L2_CELL:
        bits |= 32
    comptime if IDN_ICA_LIM_DEV:
        bits |= 64
    comptime if IDN_POLAR_MAX_DEV:
        bits |= 128
    comptime if IDN_LLE_PAD_DEV:
        bits |= 256
    comptime if IDN_RES_MOVES:
        bits |= 16
    comptime if IDN_RAND_RESIDENT:
        bits |= 8
    comptime if IDN_LLE_DEV_F0:
        bits |= 4
    comptime if IDN_OLS_ONE_ENTRY:
        bits |= 1
    comptime if IDN_LU_GESV:
        bits |= 2
    return PythonObject(bits)


def ols_tsqr_r_py[E: Exec](
    a: PythonObject, b: PythonObject, r: PythonObject, mu: PythonObject, ymean: PythonObject, p: PythonObject
) raises -> PythonObject:
    """r ((d + 1) x (d + 1)) = R of the blocked TSQR of [a - mu | b - ymean]
    (a m x d, b m), mu (float32 [d]) and ymean (float64 [1]) written: X and
    y cross to the device once (Exec.ols_tsqr_factor). p = [m, d]."""
    var m = _n(p, 0)
    var d = _n(p, 1)
    var n = d + 1
    if d < 1 or n > TS_MAX_N or m < n:
        raise Error("x_decomp: ols_tsqr_r needs 1 <= d, d + 1 <= " + String(TS_MAX_N) + " and m >= d + 1")
    if m * n > 2147483647:
        raise Error("x_decomp: ols_tsqr_r exceeds the Int32 index bound")
    var pa = _f(a)
    var pb = _f(b)
    var pr = _f(r)
    var pm = _f(mu)
    var ya = Int(py=ymean)
    if ya == 0:
        raise Error("x_decomp: null float64 buffer address")
    var py = MutPointer[UInt64, MutAnyOrigin](unsafe_from_address=ya)
    with GILReleased(Python()):
        E.ols_tsqr_factor(pa, pb, pr, pm, py, m, d)
    return PythonObject(n)


# TOMBSTONE: MOJOLEARN_IDN_OLS_R_TRISOLVE (loser: noise, ols taxi/istella avg ~0.98x, bits changed) deleted 2026-10-11 by c14beeda5; code recoverable at c14beeda5^.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_OLS_R_TRISOLVE.patch


def lu_gesv_py[E: Exec](a: PythonObject, b: PythonObject, info: PythonObject, p: PythonObject) raises -> PythonObject:
    """b (n x nrhs) = the solution of a x = b through `lu` then `lu_solve`
    with the factor resident (Exec.lu_gesv); info (1) is `lu`'s. p = [n, nrhs]."""
    var n = _n(p, 0)
    var nrhs = _n(p, 1)
    if n >= 1 << 24:
        raise Error("x_decomp: lu_gesv row numbers exceed float32's exact integers")
    var pa = _f(a)
    var pb = _f(b)
    var pi = _f(info)
    with GILReleased(Python()):
        E.lu_gesv(pa, pb, pi, n, nrhs)
    return PythonObject(n)


def tsqr_q_py[E: Exec](c: PythonObject, q: PythonObject, p: PythonObject) raises -> PythonObject:
    """q (m x k) = Q c (c n x k) for the factorization `tsqr_r_py` kept,
    which is then released; p = [m, n, k], k == 0 releases it only."""
    var m = _n(p, 0)
    var n = _n(p, 1)
    var k = _n(p, 2)
    if k > 0 and m * k > 2147483647:
        raise Error("x_decomp: tsqr_q exceeds the Int32 index bound")
    var pc = _f(c)
    var pq = _f(q)
    with GILReleased(Python()):
        E.tsqr_apply(pc, pq, m, n, k)
    return PythonObject(k)


def geqrf_py[E: Exec](a: PythonObject, tau: PythonObject, p: PythonObject) raises -> PythonObject:
    """In place: a (m x n, row major) becomes geqrf's factored form, tau
    (min(m, n)) its scalars."""
    var m = _n(p, 0)
    var n = _n(p, 1)
    if m <= 0 or n <= 0:
        raise Error("x_decomp: geqrf needs m, n >= 1")
    var pa = _f(a)
    var pt = _f(tau)
    with GILReleased(Python()):
        E.geqrf(pa, pt, m, n)
    return PythonObject(m if m < n else n)


def orgqr_py[E: Exec](h: PythonObject, tau: PythonObject, q: PythonObject, p: PythonObject) raises -> PythonObject:
    """q (m x qc) = the first qc columns of H_0 ... H_{kk-1} from geqrf's
    factored h (m x n) and tau (kk)."""
    var m = _n(p, 0)
    var n = _n(p, 1)
    var kk = _n(p, 2)
    var qc = _n(p, 3)
    if m <= 0 or n <= 0 or kk > m or kk > n or qc > m or qc <= 0:
        raise Error("x_decomp: orgqr needs 1 <= qc <= m and kk <= min(m, n)")
    var ph = _f(h)
    var pt = _f(tau)
    var pq = _f(q)
    with GILReleased(Python()):
        E.orgqr(ph, pt, pq, m, n, kk, qc)
    return PythonObject(qc)


def als_cg_rows_py[E: Exec](
    c: PythonObject, y: PythonObject, yty: PythonObject, x: PythonObject, steps: PythonObject, p: PythonObject,
    reg: PythonObject,
) raises -> PythonObject:
    """implicit's conjugate-gradient half-sweep: x (n x f) is the start and
    the result."""
    var n = _n(p, 0)
    var m = _n(p, 1)
    var f = _n(p, 2)
    var cg = _n(p, 3)
    var r = Float32(Float64(py=reg))
    var pc = _f(c)
    var py_ = _f(y)
    var pg = _f(yty)
    var px = _f(x)
    var ps = _f(steps)
    with GILReleased(Python()):
        E.als_cg_rows(pc, py_, pg, px, ps, n, m, f, r, cg)
    return PythonObject(n)


def mcd_py[E: Exec](
    x: PythonObject, loc: PythonObject, cov: PythonObject, sup: PythonObject, dist: PythonObject,
    p: PythonObject,
) raises -> PythonObject:
    """MinCovDet's fast_mcd (x_decomp/mcd.mojo): x (n x d) in; location
    (d), covariance (d x d), support (n int32 0/1) and distances (n) out.
    p = [n, d, h, seed, n_sub, n_ss, h_sub, n_trials, n_m, h_m, n_best_m]."""
    var q = List[Int]()
    for i in range(11):
        q.append(Int(py=p[i]))
    var n = q[0]
    var d = q[1]
    if n < 1 or d < 2 or n * d > 2147483647 or q[2] < 1 or q[2] > n:
        raise Error("x_decomp: mcd needs n >= 1, d >= 2 and 1 <= h <= n")
    if n > 500 and (q[4] < 1 or q[4] * q[5] > n or q[8] > n or q[8] < 1 or q[10] < 1):
        raise Error("x_decomp: mcd subset plan out of range")
    var px = _f(x)
    var pl = _f(loc)
    var pc = _f(cov)
    var ps = _i(sup)
    var pd = _f(dist)
    with GILReleased(Python()):
        var X = mat_from(px, n, d)
        fast_mcd[E](X, q, pl, pc, ps, pd)
    return PythonObject(n)


def lda_online_py[E: Exec](
    x: PythonObject, comps: PythonObject, exp_dir: PythonObject, p: PythonObject, f: PythonObject
) raises -> PythonObject:
    """One online pass of LatentDirichletAllocation (x_decomp/lda_online.mojo)
    over x (n x v): comps and exp_dir (nc x v) updated in place.
    p = [n, v, nc, batch_size, max_doc_update_iter, seed, draw, n_batch_iter];
    f = [doc_topic_prior, topic_word_prior, learning_offset, learning_decay,
    mean_change_tol, total_samples]. Returns (draw, n_batch_iter)."""
    var n = _n(p, 0)
    var v = _n(p, 1)
    var nc = _n(p, 2)
    var bs = _n(p, 3)
    var mdi = _n(p, 4)
    var seed = Int(py=p[5])
    var draw = Int(py=p[6])
    var nbi = Int(py=p[7])
    if bs < 1 or v < 1 or nc < 1 or n * v > 2147483647 or nc * v > 2147483647:
        raise Error("x_decomp: lda_online shape out of range")
    var fv = List[Float64]()
    for i in range(6):
        fv.append(Float64(py=f[i]))
    var px = _f(x)
    var pc = _f(comps)
    var pe = _f(exp_dir)
    with GILReleased(Python()):
        var X = mat_from(px, n, v)
        var C = mat_from(pc, nc, v)
        var ED = mat_from(pe, nc, v)
        lda_online_pass[E](X, C, ED, bs, mdi, seed, draw, nbi, fv[0], fv[1], fv[2], fv[3], fv[4], fv[5])
        for i in range(nc * v):
            pc.unsafe_store(i, C.d[i])
            pe.unsafe_store(i, ED.d[i])
    return Python.tuple(draw, nbi)


def gather_py(src: PythonObject, idx: PythonObject, m: PythonObject, dst: PythonObject) raises -> PythonObject:
    """dst[a] = src[idx[a]] for a < m (exact copies; x_decomp/moves.mojo)."""
    var n = Int(py=m)
    if n < 0:
        raise Error("x_decomp: negative gather count")
    var ps = _f(src)
    var pi = _i(idx)
    var pd = _f(dst)
    with GILReleased(Python()):
        gather(ps, pi, n, pd)
    return PythonObject(n)


def scatter_py(dst: PythonObject, idx: PythonObject, m: PythonObject, src: PythonObject) raises -> PythonObject:
    """dst[idx[a]] = src[a] for a < m, in order (exact copies)."""
    var n = Int(py=m)
    if n < 0:
        raise Error("x_decomp: negative scatter count")
    var pd = _f(dst)
    var pi = _i(idx)
    var ps = _f(src)
    with GILReleased(Python()):
        scatter(pd, pi, n, ps)
    return PythonObject(n)


def triu_nonzero_py(dis: PythonObject, n: PythonObject, pos: PythonObject, mir: PythonObject) raises -> PythonObject:
    """The row-major positions of the nonzero strict upper triangle of an
    n x n matrix (and their mirrors); returns their count."""
    var nn = Int(py=n)
    if nn < 0 or nn * nn > 2147483647:
        raise Error("x_decomp: triu_nonzero exceeds the Int32 index bound")
    var pd = _f(dis)
    var pp = _i(pos)
    var pm = _i(mir)
    var m = 0
    with GILReleased(Python()):
        m = triu_nonzero(pd, nn, pp, pm)
    return PythonObject(m)


def argsort_f32_py(x: PythonObject, m: PythonObject, order: PythonObject) raises -> PythonObject:
    """The stable order of x (m float32 values); NaN refused."""
    var n = Int(py=m)
    var px = _f(x)
    var po = _i(order)
    argsort_f32(px, n, po)
    return PythonObject(n)


def iso_order_py(
    x: PythonObject, y: PythonObject, xorder: PythonObject, m: PythonObject, order: PythonObject
) raises -> PythonObject:
    """The stable order by (x, y) given the stable order by x; NaN refused."""
    var n = Int(py=m)
    var px = _f(x)
    var py_ = _f(y)
    var pxo = _i(xorder)
    var po = _i(order)
    iso_order(px, py_, pxo, n, po)
    return PythonObject(n)


# ---- lane apple-fast-py2mojo-decomp (2026-10-03): host buffers, both bindings
def py2mojo_py() raises -> PythonObject:
    """1: the Mojo data path; 0: built with -D MOJOLEARN_PY2MOJO_decomp_OFF (the A/B arm)."""
    return PythonObject(1 if PY2MOJO_DECOMP else 0)


def move_py(src: PythonObject, idx: PythonObject, dst: PythonObject, p: PythonObject) raises -> PythonObject:
    """x_decomp/moves.mojo `move` on host buffers; p = [op, count, a1, a2, a3,
    ist, ioff, nsrc, ndst, nidx], every index checked."""
    var op = Int(py=p[0])
    var count = _n(p, 1)
    var nsrc = _n(p, 7)
    var ndst = _n(p, 8)
    var ps = _f(src)
    var pi = _f(idx)
    var pd = _f(dst)
    var a1 = _n(p, 2)
    var a2 = _n(p, 3)
    var a3 = _n(p, 4)
    var ist = _n(p, 5)
    var ioff = _n(p, 6)
    var nidx = _n(p, 9)
    if op == 0 and count > 0 and (count - 1) // max(a1, 1) * ist + ioff >= nidx:
        raise Error("x_decomp: take_rows reads past its index list")
    with GILReleased(Python()):
        move_host(op, ps, pi, pd, count, a1, a2, a3, ist, ioff, nsrc, ndst)
    return PythonObject(count)


def dsum_sq_py(x: PythonObject, m: PythonObject) raises -> PythonObject:
    """The binary64 sum of the squares of m float32 values in the device
    form's chunked order (`dsum_sq_host`): the host column."""
    var n = Int(py=m)
    if n <= 0:
        return PythonObject(Float64(0))
    var px = _f(x)
    var t = UInt64(0)
    with GILReleased(Python()):
        t = dsum_sq_host(px, n)
    return PythonObject(bitcast[DType.float64](t))


def order_f_py(x: PythonObject, m: PythonObject, dst: PythonObject) raises -> PythonObject:
    """The stable order of x (m float32 values) as exact floats; NaN refused."""
    var n = Int(py=m)
    if n > 0:
        order_f(_f(x), n, _f(dst))
    return PythonObject(n)


def select_smallest_py(x: PythonObject, p: PythonObject, sel: PythonObject, mask: PythonObject) raises -> PythonObject:
    """p = [m, h]: the h smallest of x by (value, index), ascending by index,
    as exact floats, and the int32 membership of every row."""
    var m = _n(p, 0)
    var h = _n(p, 1)
    if m > 0:
        select_smallest(_f(x), m, h, _f(sel), _i(mask))
    return PythonObject(h)


def argmin_all_py(x: PythonObject, m: PythonObject, dst: PythonObject) raises -> PythonObject:
    """The positions (exact floats) of every value equal to the minimum; their count."""
    var n = Int(py=m)
    if n <= 0:
        return PythonObject(0)
    return PythonObject(argmin_all(_f(x), n, _f(dst)))


def sign_labels_py(x: PythonObject, m: PythonObject, dst: PythonObject) raises -> PythonObject:
    """int32 1 where x >= 0, else -1."""
    var n = Int(py=m)
    if n > 0:
        sign_labels(_f(x), n, _i(dst))
    return PythonObject(n)


def accuracy_py(y: PythonObject, pred: PythonObject, w: PythonObject, m: PythonObject) raises -> PythonObject:
    """(matches, total): float64 labels against int32 predictions; w = 0
    for unweighted, else a float64 weight address (sums in order)."""
    var n = Int(py=m)
    var weighted = Int(py=w) != 0
    if n <= 0:
        return Python.tuple(Float64(0), Float64(0))
    var pw = _d(w) if weighted else _d(y)
    var r = accuracy(_d(y), _i(pred), pw, weighted, n)
    return Python.tuple(r[0], r[1])


def pca_mle_rank_host_py(sp: PythonObject, p: PythonObject) raises -> PythonObject:
    """The host column of `x_decomp_dev_pca_mle_rank` (x_decomp/pca_mle.mojo):
    sp = the address of d binary64 spectrum values, p = [d, n_samples]; the
    Minka MLE rank (0 when d < 2)."""
    var d = _n(p, 0)
    var n = _n(p, 1)
    if d < 2:
        return PythonObject(0)
    var scratch = List[Float32](length=mle_scratch_words(d), fill=Float32(0))
    var r = pca_mle_rank_host(_f(sp), d, n, F32Ptr(unsafe_from_address=Int(scratch.unsafe_ptr())))
    _ = scratch^
    return PythonObject(r)


def topn_desc_py(x: PythonObject, p: PythonObject, skip: PythonObject, dst: PythonObject) raises -> PythonObject:
    """p = [m, n]: the n best positions by (-x, index), skipping the nonzero
    entries of the float32 row at `skip` (0: none); their count."""
    var m = _n(p, 0)
    var n = _n(p, 1)
    if m == 0 or n == 0:
        return PythonObject(0)
    var has = Int(py=skip) != 0
    var px = _f(x)
    var ps = _f(skip) if has else px
    return PythonObject(topn_desc(px, m, ps, has, n, _i(dst)))


def numeric_mode_py() raises -> PythonObject:
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))


def vendor_py[E: Exec]() raises -> PythonObject:
    return PythonObject(E.vendor())


def fast_defines_py() raises -> PythonObject:
    """The recovered Apple FAST switches the Python linalg doors read
    (lane/apple-fast-rec-decomp, 2026-10-04; from
    lane/apple-fast-decomp-linalg@74d52352b), comma-joined: the build's
    switches. Both (MOJOLEARN_QR_FAST_DEV, MOJOLEARN_SVD_FAST_CHOLQR) were
    deleted 2026-10-09 (lane/owed-deletions-D1), so this is always empty;
    the binding stays for `_kit_fast_define` in
    python/mojolearn/_expansion_decomp.py."""
    # TOMBSTONE: MOJOLEARN_QR_FAST_DEV (DROPPED-slower) deleted 2026-10-09 on lane/owed-deletions-D1; code
    # recoverable at b639a2bd2. Restore: git apply experiments/removed/MOJOLEARN_QR_FAST_DEV.patch.
    # TOMBSTONE: MOJOLEARN_SVD_FAST_CHOLQR (DROPPED-slower) deleted 2026-10-09 on lane/owed-deletions-D1; code
    # recoverable at b639a2bd2. Restore: git apply experiments/removed/MOJOLEARN_SVD_FAST_CHOLQR.patch.
    return PythonObject(String(""))
