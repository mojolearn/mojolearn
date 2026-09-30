# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE NEIGHBORS EXPANSION LANE'S ONE SOURCE (algorithm expansion lane 3:
neighbors + kernel).

Every primitive the lane's estimators run is an ITEM FUNCTION here: the work
of one output element (or one row, or the whole of a sequential solve),
written once. `x_neighbors/device_ops.mojo` calls it from a GPU kernel, one
thread per item; `x_neighbors/host_ops.mojo` calls it from a plain host loop.
Same statements, same order, so the CPU column and every GPU column compute
the same bits (IDENTITY_PATHS.md "The rule"):

  * every fold runs in ascending index order inside one item (no atomics, no
    tree reductions whose shape could depend on a launch);
  * every product-plus-sum is `identical_mul_add` (the pinned fused spelling),
    every bare product `identical_mul`, every division `identical_div`, every
    transcendental the portable spelling in checks/numerics.mojo;
  * every stored intermediate goes through `ftz`;
  * ties break by the lowest index;
  * no float64 anywhere in this file.

Nothing here imports a GPU module, so the CPU-only host binding compiles it.
"""
from checks.numerics import (
    ftz,
    identical_mul_add,
    identical_mul,
    identical_div,
    identical_exp,
    identical_log,
    identical_sqrt,
    identical_tanh,
    identical_cos,
    identical_sin,
)

from std.memory import bitcast

comptime FP = MutPointer[Float32, MutAnyOrigin]
comptime IP = MutPointer[Int32, MutAnyOrigin]


@always_inline
def _sub(a: Float32, b: Float32) -> Float32:
    return ftz(ftz(a) - ftz(b))


@always_inline
def _add(a: Float32, b: Float32) -> Float32:
    return ftz(ftz(a) + ftz(b))


# ------------------------------------------------------------------ distances
# DEVIATION 5206 (IDENTITY_PATHS row 120)
def sqdist_item(t: Int, x: FP, y: FP, res: FP, n: Int, m: Int, d: Int):
    """Squared euclidean distance of x row i and y row j, t = i*m + j,
    features folded in ascending order through the pinned fma."""
    var i = t // m
    var j = t - i * m
    var acc = Float32(0)
    for f in range(d):
        var df = _sub(x.unsafe_load(i * d + f), y.unsafe_load(j * d + f))
        acc = ftz(identical_mul_add(df, df, acc))
    res.unsafe_store(t, acc)


# DEVIATION 5206 (row 120)
def nan_sqdist_item(t: Int, x: FP, y: FP, res: FP, n: Int, m: Int, d: Int):
    """sklearn `nan_euclidean_distances(squared=True)`: the sum over the
    coordinates present in BOTH rows, divided by the present count and
    multiplied by the feature count (their order: `/= present`, `*= d`).
    No coordinate in common is stored as -1 (their NaN; a computed NaN never
    reaches an output here, Clause B), which a caller reads as "no distance"."""
    var i = t // m
    var j = t - i * m
    var acc = Float32(0)
    var present = 0
    for f in range(d):
        var a = x.unsafe_load(i * d + f)
        var b = y.unsafe_load(j * d + f)
        if a != a or b != b:
            continue
        present += 1
        var df = _sub(a, b)
        acc = ftz(identical_mul_add(df, df, acc))
    if present == 0:
        res.unsafe_store(t, Float32(-1))
        return
    var r = ftz(identical_div(acc, Float32(present)))
    res.unsafe_store(t, ftz(identical_mul(r, Float32(d))))


# ------------------------------------------------------------------ kernels
comptime K_LINEAR = 0
comptime K_POLY = 1
comptime K_RBF = 2
comptime K_SIGMOID = 3
comptime K_LAPLACIAN = 4
comptime K_COSINE = 5
comptime K_CHI2 = 6
comptime K_ADDITIVE_CHI2 = 7


# DEVIATION 5208 (row 122)
def kernel_item(
    t: Int, x: FP, y: FP, res: FP, n: Int, m: Int, d: Int,
    kind: Int, gamma: Float32, coef0: Float32, degree: Int,
):
    """One kernel matrix entry K(x_i, y_j), t = i*m + j. sklearn
    `metrics/pairwise.py` (linear_kernel, polynomial_kernel, rbf_kernel,
    sigmoid_kernel, laplacian_kernel, cosine_similarity, chi2_kernel,
    additive_chi2_kernel). The polynomial degree is an integer power by
    repeated pinned products, ascending."""
    var i = t // m
    var j = t - i * m
    var r = Float32(0)
    if kind == K_LINEAR or kind == K_POLY or kind == K_SIGMOID:
        var acc = Float32(0)
        for f in range(d):
            acc = ftz(identical_mul_add(ftz(x.unsafe_load(i * d + f)), ftz(y.unsafe_load(j * d + f)), acc))
        if kind == K_LINEAR:
            r = acc
        else:
            var z = ftz(identical_mul_add(gamma, acc, coef0))
            if kind == K_SIGMOID:
                r = ftz(identical_tanh(z))
            else:
                var p = Float32(1)
                for _ in range(degree):
                    p = ftz(identical_mul(p, z))
                r = p
    elif kind == K_RBF:
        var acc = Float32(0)
        for f in range(d):
            var df = _sub(x.unsafe_load(i * d + f), y.unsafe_load(j * d + f))
            acc = ftz(identical_mul_add(df, df, acc))
        r = ftz(identical_exp(ftz(identical_mul(-gamma, acc))))
    elif kind == K_LAPLACIAN:
        var acc = Float32(0)
        for f in range(d):
            acc = _add(acc, abs(_sub(x.unsafe_load(i * d + f), y.unsafe_load(j * d + f))))
        r = ftz(identical_exp(ftz(identical_mul(-gamma, acc))))
    elif kind == K_COSINE:
        var dot = Float32(0)
        var nx = Float32(0)
        var ny = Float32(0)
        for f in range(d):
            var a = ftz(x.unsafe_load(i * d + f))
            var b = ftz(y.unsafe_load(j * d + f))
            dot = ftz(identical_mul_add(a, b, dot))
            nx = ftz(identical_mul_add(a, a, nx))
            ny = ftz(identical_mul_add(b, b, ny))
        # sklearn normalize(): a zero-norm row stays zero, so its similarity is 0
        if nx == Float32(0) or ny == Float32(0):
            r = Float32(0)
        else:
            var den = ftz(identical_mul(ftz(identical_sqrt(nx)), ftz(identical_sqrt(ny))))
            r = ftz(identical_div(dot, den))
    else:
        # chi2 / additive chi2: -sum (x-y)^2 / (x+y) over the terms with x+y != 0
        var acc = Float32(0)
        for f in range(d):
            var a = ftz(x.unsafe_load(i * d + f))
            var b = ftz(y.unsafe_load(j * d + f))
            var den = _add(a, b)
            if den != Float32(0):
                var df = _sub(a, b)
                acc = _add(acc, ftz(identical_div(ftz(identical_mul(df, df)), den)))
        if kind == K_ADDITIVE_CHI2:
            r = -acc
        else:
            r = ftz(identical_exp(ftz(identical_mul(-gamma, acc))))
    res.unsafe_store(t, r)


# ------------------------------------------------------------------ dense algebra
# DEVIATION 5209 (row 123)
def matmul_item(t: Int, a: FP, b: FP, res: FP, n: Int, k: Int, m: Int):
    """C = A (n x k) B (k x m), t = i*m + j, ascending p, pinned fma."""
    var i = t // m
    var j = t - i * m
    var acc = Float32(0)
    for p in range(k):
        acc = ftz(identical_mul_add(ftz(a.unsafe_load(i * k + p)), ftz(b.unsafe_load(p * m + j)), acc))
    res.unsafe_store(t, acc)


# DEVIATION 5209 (row 123)
def rowsum_item(t: Int, a: FP, res: FP, n: Int, m: Int):
    var acc = Float32(0)
    for j in range(m):
        acc = _add(acc, a.unsafe_load(t * m + j))
    res.unsafe_store(t, acc)


# DEVIATION 5209 (row 123)
def colsum_item(t: Int, a: FP, res: FP, n: Int, m: Int):
    var acc = Float32(0)
    for i in range(n):
        acc = _add(acc, a.unsafe_load(i * m + t))
    res.unsafe_store(t, acc)


comptime U_EXP = 0
comptime U_LOG = 1
comptime U_SQRT = 2
comptime U_TANH = 3
comptime U_COS = 4
comptime U_SIN = 5
comptime U_IDENTITY = 6
comptime U_RECIP = 7


def unary_item(t: Int, x: FP, res: FP, count: Int, op: Int, a: Float32, b: Float32):
    """res = f(a*x + b), the affine step through the pinned fma."""
    var z = ftz(identical_mul_add(a, ftz(x.unsafe_load(t)), b))
    var r: Float32
    if op == U_EXP:
        r = identical_exp(z)
    elif op == U_LOG:
        r = identical_log(z)
    elif op == U_SQRT:
        r = identical_sqrt(z)
    elif op == U_TANH:
        r = identical_tanh(z)
    elif op == U_COS:
        r = identical_cos(z)
    elif op == U_SIN:
        r = identical_sin(z)
    elif op == U_RECIP:
        r = identical_div(Float32(1), z)
    else:
        r = z
    res.unsafe_store(t, ftz(r))


# ------------------------------------------------------------------ selection
# DEVIATION 5207 (row 121)
def knn_select_item(t: Int, dmat: FP, dist: FP, idx: IP, n: Int, m: Int, k: Int, exclude_self: Int):
    """Row t's k smallest entries of the n x m distance matrix, ascending by
    (value, column): a candidate enters only when STRICTLY smaller than the
    entry it passes, so an equal value keeps the lower column first.
    exclude_self != 0 skips column t (a query against its own training set).
    Unfilled slots (m too small) hold +inf and index -1."""
    var inf = bitcast[DType.float32](UInt32(0x7F800000))
    for s in range(k):
        dist.unsafe_store(t * k + s, inf)
        idx.unsafe_store(t * k + s, Int32(-1))
    for j in range(m):
        if exclude_self != 0 and j == t:
            continue
        var v = dmat.unsafe_load(t * m + j)
        if not (v < dist.unsafe_load(t * k + k - 1)):
            continue
        var s = k - 1
        while s > 0 and v < dist.unsafe_load(t * k + s - 1):
            dist.unsafe_store(t * k + s, dist.unsafe_load(t * k + s - 1))
            idx.unsafe_store(t * k + s, idx.unsafe_load(t * k + s - 1))
            s -= 1
        dist.unsafe_store(t * k + s, v)
        idx.unsafe_store(t * k + s, Int32(j))


# DEVIATION 5206 + 5207 (rows 120, 121)
def knn_sq_item(
    t: Int, x: FP, y: FP, dist: FP, idx: IP,
    n: Int, m: Int, d: Int, k: Int, exclude_self: Int,
):
    """`sqdist_item` fused into `knn_select_item` for x row t: the k smallest
    squared euclidean distances to the rows of y, ascending by (value,
    column). Each candidate's value is `sqdist_item`'s statements for cell
    (t, j) (features ascending, the pinned fma, ftz), and the columns are
    offered in ascending order to `knn_select_item`'s strict `<` insertion, so
    the answer is the one the two ops return through an n x m matrix, bit for
    bit; the matrix is never written. `worst` holds the last slot's value
    (the one the insertion test reads) in a register."""
    var inf = bitcast[DType.float32](UInt32(0x7F800000))
    for s in range(k):
        dist.unsafe_store(t * k + s, inf)
        idx.unsafe_store(t * k + s, Int32(-1))
    var worst = inf
    for j in range(m):
        if exclude_self != 0 and j == t:
            continue
        var acc = Float32(0)
        for f in range(d):
            var df = _sub(x.unsafe_load(t * d + f), y.unsafe_load(j * d + f))
            acc = ftz(identical_mul_add(df, df, acc))
        var v = acc
        if not (v < worst):
            continue
        var s = k - 1
        while s > 0 and v < dist.unsafe_load(t * k + s - 1):
            dist.unsafe_store(t * k + s, dist.unsafe_load(t * k + s - 1))
            idx.unsafe_store(t * k + s, idx.unsafe_load(t * k + s - 1))
            s -= 1
        dist.unsafe_store(t * k + s, v)
        idx.unsafe_store(t * k + s, Int32(j))
        worst = dist.unsafe_load(t * k + k - 1)


# DEVIATION 5209 (row 123)
def group_mean_item(t: Int, x: FP, labels: IP, res: FP, n: Int, d: Int, n_groups: Int):
    """Mean of the rows labelled g, feature f, t = g*d + f: rows in ascending
    order, one division. An empty group stores 0."""
    var g = t // d
    var f = t - g * d
    var acc = Float32(0)
    var cnt = 0
    for i in range(n):
        if Int(labels.unsafe_load(i)) == g:
            acc = _add(acc, x.unsafe_load(i * d + f))
            cnt += 1
    if cnt == 0:
        res.unsafe_store(t, Float32(0))
    else:
        res.unsafe_store(t, ftz(identical_div(acc, Float32(cnt))))


# ------------------------------------------------------------------ one-class SVM
comptime SMO_TAU = Float32(1e-12)


# DEVIATION 5200 (row 125)
@always_inline
def ocsvm_g0(q: FP, alpha: FP, n: Int, i: Int) -> Float32:
    """The initial gradient of sample i: sum_j Q[i, j] alpha_j over the
    nonzero alpha, j ascending."""
    var acc = Float32(0)
    for j in range(n):
        var a = alpha.unsafe_load(j)
        if a != Float32(0):
            acc = ftz(identical_mul_add(ftz(q.unsafe_load(i * n + j)), a, acc))
    return acc


@always_inline
def ocsvm_obj(gmax: Float32, gjv: Float32, qdi: Float32, qjj: Float32, qij: Float32) -> Float32:
    """WSS3's second-order objective of pair (i, j), for grad_diff > 0."""
    var grad_diff = _add(gmax, gjv)
    var two_q = ftz(identical_mul(Float32(2), qij))
    var quad = _sub(_add(qdi, qjj), two_q)
    var num = ftz(identical_mul(grad_diff, grad_diff))
    if quad > Float32(0):
        return -ftz(identical_div(num, quad))
    return -ftz(identical_div(num, SMO_TAU))


@always_inline
def ocsvm_update(q: FP, cv: FP, alpha: FP, g: FP, n: Int, i: Int, j: Int) -> Tuple[Float32, Float32]:
    """The two-variable step on (i, j): stores alpha_i, alpha_j and returns
    their changes (dai, daj)."""
    var old_ai = alpha.unsafe_load(i)
    var old_aj = alpha.unsafe_load(j)
    var quad = _sub(_add(q.unsafe_load(i * n + i), q.unsafe_load(j * n + j)), ftz(identical_mul(Float32(2), q.unsafe_load(i * n + j))))
    if quad <= Float32(0):
        quad = SMO_TAU
    var delta = ftz(identical_div(_sub(g.unsafe_load(i), g.unsafe_load(j)), quad))
    var ci = cv.unsafe_load(i)
    var cj = cv.unsafe_load(j)
    var total = _add(old_ai, old_aj)
    var ai = _sub(old_ai, delta)
    var aj = _add(old_aj, delta)
    if total > ci:
        if ai > ci:
            ai = ci
            aj = _sub(total, ci)
    else:
        if aj < Float32(0):
            aj = Float32(0)
            ai = total
    if total > cj:
        if aj > cj:
            aj = cj
            ai = _sub(total, cj)
    else:
        if ai < Float32(0):
            ai = Float32(0)
            aj = total
    alpha.unsafe_store(i, ai)
    alpha.unsafe_store(j, aj)
    return (_sub(ai, old_ai), _sub(aj, old_aj))


@always_inline
def ocsvm_g_step(q: FP, g: FP, n: Int, i: Int, j: Int, dai: Float32, daj: Float32, k: Int):
    """Gradient entry k after the step: Q[i, k] dai, then Q[j, k] daj."""
    var gk = g.unsafe_load(k)
    gk = ftz(identical_mul_add(ftz(q.unsafe_load(i * n + k)), dai, gk))
    gk = ftz(identical_mul_add(ftz(q.unsafe_load(j * n + k)), daj, gk))
    g.unsafe_store(k, gk)


@always_inline
def ocsvm_rho(g: FP, alpha: FP, cv: FP, n: Int) -> Float32:
    """libsvm's calculate_rho, y = +1 throughout; samples ascending."""
    var neg_inf = bitcast[DType.float32](UInt32(0xFF800000))
    var pos_inf = bitcast[DType.float32](UInt32(0x7F800000))
    var ub = pos_inf
    var lb = neg_inf
    var nr_free = 0
    var sum_free = Float32(0)
    for i in range(n):
        var yg = g.unsafe_load(i)
        var a = alpha.unsafe_load(i)
        if a >= cv.unsafe_load(i):
            if yg > lb:
                lb = yg
        elif a <= Float32(0):
            if yg < ub:
                ub = yg
        else:
            nr_free += 1
            sum_free = _add(sum_free, yg)
    if nr_free > 0:
        return ftz(identical_div(sum_free, Float32(nr_free)))
    return ftz(identical_mul(_add(ub, lb), Float32(0.5)))


def ocsvm_smo_item(t: Int, q: FP, cv: FP, alpha: FP, g: FP, info: FP, iters: IP, n: Int, eps: Float32, max_iter: Int):
    """libsvm's `Solver::Solve` for the one-class problem (sklearn
    `svm/src/libsvm/svm.cpp`: `solve_one_class`, `Solver::Solve`,
    `select_working_set` (WSS3, second order), `calculate_rho`), with every
    y = +1 and p = 0, no shrinking, SEQUENTIAL in one item. `cv` holds each
    sample's upper bound C_i (sklearn's sample_weight; all ones without it).
    `alpha` arrives holding libsvm's initial point (C_i while nu * sum(C)
    lasts, then the remainder); `q` is the n x n kernel matrix. Float32 with
    the pinned spellings where libsvm computes in double (DEVIATION 5200).
    Ties in the working-set scans resolve as libsvm's `>=` / `<=` dres: the
    LAST index of equal gradient wins. info[0] = rho. The GPU column runs the
    same helpers with the scans spread over a threadgroup
    (`x_neighbors/block_ops.mojo::ocsvm_smo_block`)."""
    var neg_inf = bitcast[DType.float32](UInt32(0xFF800000))
    var pos_inf = bitcast[DType.float32](UInt32(0x7F800000))
    for i in range(n):
        g.unsafe_store(i, ocsvm_g0(q, alpha, n, i))
    var it = 0
    while it < max_iter:
        var gmax = neg_inf
        var gi = -1
        for t in range(n):
            if alpha.unsafe_load(t) < cv.unsafe_load(t):
                var ng = -g.unsafe_load(t)
                if ng >= gmax:
                    gmax = ng
                    gi = t
        var gmax2 = neg_inf
        var gj = -1
        var obj_min = pos_inf
        if gi >= 0:
            var qdi = q.unsafe_load(gi * n + gi)
            for j in range(n):
                if alpha.unsafe_load(j) > Float32(0):
                    var gjv = g.unsafe_load(j)
                    var grad_diff = _add(gmax, gjv)
                    if gjv >= gmax2:
                        gmax2 = gjv
                    if grad_diff > Float32(0):
                        var obj = ocsvm_obj(gmax, gjv, qdi, q.unsafe_load(j * n + j), q.unsafe_load(gi * n + j))
                        if obj <= obj_min:
                            gj = j
                            obj_min = obj
        if gi < 0 or gj < 0 or _add(gmax, gmax2) < eps:
            break
        it += 1
        var d = ocsvm_update(q, cv, alpha, g, n, gi, gj)
        for k in range(n):
            ocsvm_g_step(q, g, n, gi, gj, d[0], d[1], k)
    info.unsafe_store(0, ocsvm_rho(g, alpha, cv, n))
    iters.unsafe_store(0, Int32(it))


# ------------------------------------------------------------------ gathers and small reductions
def take_rows_item(t: Int, src: FP, rows: IP, res: FP, n_out: Int, d: Int, n_src: Int):
    """res[r, f] = src[rows[r], f], t = r*d + f (a pure copy)."""
    var r = t // d
    var f = t - r * d
    res.unsafe_store(t, src.unsafe_load(Int(rows.unsafe_load(r)) * d + f))


def take_cols_item(t: Int, src: FP, cols: IP, res: FP, n: Int, src_c: Int, c: Int):
    """res[i, k] = src[i, cols[k]], t = i*c + k (a pure copy)."""
    var i = t // c
    var k = t - i * c
    res.unsafe_store(t, src.unsafe_load(i * src_c + Int(cols.unsafe_load(k))))


# DEVIATION 5209 (row 123)
def variance_item(t: Int, x: FP, res: FP, count: Int):
    """numpy's `X.var()` over every element, ONE item: the mean by an
    ascending fold and one division, then the ascending fold of squared
    deviations and one division (population variance, ddof = 0)."""
    var acc = Float32(0)
    for i in range(count):
        acc = _add(acc, x.unsafe_load(i))
    var mean = ftz(identical_div(acc, Float32(count)))
    var ss = Float32(0)
    for i in range(count):
        var df = _sub(x.unsafe_load(i), mean)
        ss = ftz(identical_mul_add(df, df, ss))
    res.unsafe_store(0, ftz(identical_div(ss, Float32(count))))


# ------------------------------------------------------------------ LocalOutlierFactor
# DEVIATION 5210 (row 124)
def lof_lrd_item(t: Int, dist: FP, idx: IP, fit_dist: FP, lrd: FP, k: Int, n: Int, n_fit: Int):
    """sklearn `_lof.py` `_local_reachability_density`: reach = max(dist,
    the neighbor's own k-distance), lrd = 1 / (mean(reach) + 1e-10); the
    k reach distances folded ascending by neighbor rank."""
    var acc = Float32(0)
    for j in range(k):
        var nb = Int(idx.unsafe_load(t * k + j))
        var kd = fit_dist.unsafe_load(nb * k + k - 1)
        var dv = dist.unsafe_load(t * k + j)
        acc = _add(acc, dv if dv > kd else kd)
    var mean = ftz(identical_div(acc, Float32(k)))
    lrd.unsafe_store(t, ftz(identical_div(Float32(1), _add(mean, Float32(1e-10)))))


# DEVIATION 5210 (row 124)
def lof_score_item(t: Int, idx: IP, fit_lrd: FP, lrd: FP, score: FP, k: Int, n: Int, n_fit: Int):
    """-mean(lrd[neighbors] / lrd[t]): the ratios folded ascending by rank."""
    var own = lrd.unsafe_load(t)
    var acc = Float32(0)
    for j in range(k):
        acc = _add(acc, ftz(identical_div(fit_lrd.unsafe_load(Int(idx.unsafe_load(t * k + j))), own)))
    score.unsafe_store(t, -ftz(identical_div(acc, Float32(k))))


# ------------------------------------------------------------------ distances, L1
# DEVIATION 5206 (row 120)
def l1dist_item(t: Int, x: FP, y: FP, res: FP, n: Int, m: Int, d: Int):
    var i = t // m
    var j = t - i * m
    var acc = Float32(0)
    for f in range(d):
        acc = _add(acc, abs(_sub(x.unsafe_load(i * d + f), y.unsafe_load(j * d + f))))
    res.unsafe_store(t, acc)


# ------------------------------------------------------------------ KernelPCA
# DEVIATION 5202 (row 126)
def kpca_center_item(t: Int, k: FP, fit_cols: FP, pred_rows: FP, fit_all: FP, res: FP, n: Int, m: Int):
    """sklearn KernelCenterer: K - K_fit_rows_[j] - K_pred_cols[i] + K_fit_all_,
    in their order (two subtractions, then the addition)."""
    var i = t // m
    var j = t - i * m
    var v = _sub(k.unsafe_load(t), fit_cols.unsafe_load(j))
    v = _sub(v, pred_rows.unsafe_load(i))
    res.unsafe_store(t, _add(v, fit_all.unsafe_load(0)))


def scale_div_item(t: Int, x: FP, res: FP, count: Int, s: Float32):
    res.unsafe_store(t, ftz(identical_div(ftz(x.unsafe_load(t)), s)))


# DEVIATION 5202 (row 126)
def svd_flip_item(t: Int, v: FP, n: Int, c: Int):
    """sklearn `svd_flip(u, None)` on column t of the n x c matrix: the FIRST
    row of largest |value| decides, and a negative one flips the column."""
    var best = Float32(-1)
    var at = 0
    for i in range(n):
        var a = abs(v.unsafe_load(i * c + t))
        if a > best:
            best = a
            at = i
    if v.unsafe_load(at * c + t) < Float32(0):
        for i in range(n):
            v.unsafe_store(i * c + t, -v.unsafe_load(i * c + t))


def kpca_alpha_scale_item(t: Int, v: FP, w: FP, res: FP, n: Int, c: Int, divide: Int):
    """divide = 0: v * sqrt(lambda) (fit_transform); divide = 1: v / sqrt(lambda)
    (the transform's scaled alphas). A non-positive eigenvalue gives 0 (their
    `non_zeros` mask after `_check_psd_eigenvalues` zeroed it)."""
    var col = t - (t // c) * c
    var lam = w.unsafe_load(col)
    if not (lam > Float32(0)):
        res.unsafe_store(t, Float32(0))
        return
    var s = ftz(identical_sqrt(lam))
    var x = ftz(v.unsafe_load(t))
    if divide != 0:
        res.unsafe_store(t, ftz(identical_div(x, s)))
    else:
        res.unsafe_store(t, ftz(identical_mul(x, s)))


# ------------------------------------------------------------------ NearestCentroid
# DEVIATION 5212 (row 126)
def nc_std_item(t: Int, x: FP, lab: IP, cent: FP, std: FP, n: Int, d: Int, n_classes: Int):
    """`within_class_std_dev_[f]` = sqrt(sum_i (x_if - centroid_{y_i f})^2 / (n - C)),
    rows ascending. n == C stores 0 (theirs divides by zero)."""
    var f = t
    var ss = Float32(0)
    for i in range(n):
        var df = _sub(x.unsafe_load(i * d + f), cent.unsafe_load(Int(lab.unsafe_load(i)) * d + f))
        ss = ftz(identical_mul_add(df, df, ss))
    if n - n_classes <= 0:
        std.unsafe_store(f, Float32(0))
        return
    std.unsafe_store(f, ftz(identical_sqrt(ftz(identical_div(ss, Float32(n - n_classes))))))


# DEVIATION 5212 / 5201 (row 126)
def nc_shrink_item(
    t: Int, x: FP, cent: FP, nk: FP, std: FP, res: FP, devs: FP,
    n: Int, d: Int, n_classes: Int, do_shrink: Int, med: Float32, shrink: Float32,
):
    """sklearn's shrunken centroid for class k, feature f (t = k*d + f): the
    dataset centroid (ascending rows, one division), m = sqrt(1/n_k - 1/n),
    s = std + median(std), deviation = (centroid - dataset centroid) / (m*s),
    soft-thresholded by `shrink`, centroid = dataset centroid + m*s*deviation.
    Without shrinking the centroid is returned unchanged. `devs` is sklearn's
    `deviations_` (soft-thresholded only when shrinking). DEVIATION 5201: m*s
    == 0 gives deviation 0."""
    var k = t // d
    var f = t - k * d
    var c = cent.unsafe_load(t)
    var acc = Float32(0)
    for i in range(n):
        acc = _add(acc, x.unsafe_load(i * d + f))
    var dsc = ftz(identical_div(acc, Float32(n)))
    var m = ftz(identical_sqrt(_sub(ftz(identical_div(Float32(1), nk.unsafe_load(k))), ftz(identical_div(Float32(1), Float32(n))))))
    var s = _add(std.unsafe_load(f), med)
    var ms = ftz(identical_mul(m, s))
    var dev = Float32(0)
    if ms != Float32(0):
        dev = ftz(identical_div(_sub(c, dsc), ms))
    if do_shrink == 0:
        devs.unsafe_store(t, dev)
        res.unsafe_store(t, c)
        return
    var mag = _sub(abs(dev), shrink)
    if mag < Float32(0):
        mag = Float32(0)
    if dev < Float32(0):
        dev = -mag
    elif dev > Float32(0):
        dev = mag
    else:
        dev = Float32(0)
    devs.unsafe_store(t, dev)
    res.unsafe_store(t, _add(dsc, ftz(identical_mul(ms, dev))))


# DEVIATION 5212 (row 126)
def nc_decision_item(t: Int, q: FP, cent: FP, std: FP, prior: FP, res: FP, n: Int, d: Int, n_classes: Int):
    """The discriminant -||x/sigma - c/sigma||^2 + 2 log(prior_k) (features
    with sigma == 0 left unscaled), t = i*C + k; the distance is square-rooted
    and squared again, as their pairwise_distances then `**= 2`."""
    var i = t // n_classes
    var k = t - i * n_classes
    var acc = Float32(0)
    for f in range(d):
        var a = ftz(q.unsafe_load(i * d + f))
        var b = ftz(cent.unsafe_load(k * d + f))
        var sg = std.unsafe_load(f)
        if sg != Float32(0):
            a = ftz(identical_div(a, sg))
            b = ftz(identical_div(b, sg))
        var df = _sub(a, b)
        acc = ftz(identical_mul_add(df, df, acc))
    var dist = ftz(identical_sqrt(acc))
    var d2 = ftz(identical_mul(dist, dist))
    var lp = ftz(identical_mul(Float32(2), ftz(identical_log(prior.unsafe_load(k)))))
    res.unsafe_store(t, _add(-d2, lp))


# DEVIATION 5209 (row 123)
def softmax_item(t: Int, x: FP, res: FP, n: Int, c: Int):
    """Row t: exp(x - max) / sum, the sum ascending."""
    var mx = x.unsafe_load(t * c)
    for j in range(1, c):
        var v = x.unsafe_load(t * c + j)
        if v > mx:
            mx = v
    var acc = Float32(0)
    for j in range(c):
        var e = ftz(identical_exp(_sub(x.unsafe_load(t * c + j), mx)))
        res.unsafe_store(t * c + j, e)
        acc = _add(acc, e)
    for j in range(c):
        res.unsafe_store(t * c + j, ftz(identical_div(res.unsafe_load(t * c + j), acc)))


# DEVIATION 5209 (row 123)
def log_softmax_item(t: Int, x: FP, res: FP, n: Int, c: Int):
    """Row t: sklearn's predict_log_proba, (x - max) - log(sum(exp(x - max))),
    the sum ascending."""
    var mx = x.unsafe_load(t * c)
    for j in range(1, c):
        var v = x.unsafe_load(t * c + j)
        if v > mx:
            mx = v
    var acc = Float32(0)
    for j in range(c):
        var l = _sub(x.unsafe_load(t * c + j), mx)
        res.unsafe_store(t * c + j, l)
        acc = _add(acc, ftz(identical_exp(l)))
    var ls = ftz(identical_log(acc))
    for j in range(c):
        res.unsafe_store(t * c + j, _sub(res.unsafe_load(t * c + j), ls))


# ------------------------------------------------------------------ kernel approximation
# DEVIATION 5203 (row 127)
def pcs_item(
    t: Int, x: FP, hidx: IP, hbit: IP, res: FP, scr: FP,
    n: Int, d_in: Int, nf: Int, nc: Int, degree: Int, gamma: Float32, coef0: Float32,
):
    """sklearn PolynomialCountSketch.transform for row t: X_gamma = sqrt(gamma) x
    (and a last feature sqrt(coef0) when nf = d_in + 1), one count sketch per
    degree (features ascending, the +/-1 bit applied exactly), and their
    CIRCULAR CONVOLUTION, which is what their fft / product / real(ifft)
    computes, here as the direct sum (ascending shift index) with no FFT
    (DEVIATION 5203). `scr` holds 2*nc floats per row."""
    var sg = ftz(identical_sqrt(gamma))
    var sc = ftz(identical_sqrt(coef0))
    var acc_row = scr + t * 2 * nc          # the running product
    var sk = scr + t * 2 * nc + nc          # this degree's sketch
    for p in range(degree):
        for h in range(nc):
            sk.unsafe_store(h, Float32(0))
        for j in range(nf):
            var v: Float32
            if j < d_in:
                v = ftz(identical_mul(sg, ftz(x.unsafe_load(t * d_in + j))))
            else:
                v = sc
            if Int(hbit.unsafe_load(p * nf + j)) < 0:
                v = -v
            var h = Int(hidx.unsafe_load(p * nf + j))
            sk.unsafe_store(h, _add(sk.unsafe_load(h), v))
        if p == 0:
            for h in range(nc):
                acc_row.unsafe_store(h, sk.unsafe_load(h))
        else:
            for h in range(nc):
                var s = Float32(0)
                for a in range(nc):
                    var b = h - a
                    if b < 0:
                        b += nc
                    s = ftz(identical_mul_add(acc_row.unsafe_load(a), sk.unsafe_load(b), s))
                res.unsafe_store(t * nc + h, s)
            for h in range(nc):
                acc_row.unsafe_store(h, res.unsafe_load(t * nc + h))
    for h in range(nc):
        res.unsafe_store(t * nc + h, acc_row.unsafe_load(h))


def pcs_sketch_item(
    t: Int, x: FP, hidx: IP, hbit: IP, sk: FP,
    n: Int, d_in: Int, nf: Int, nc: Int, degree: Int, gamma: Float32, coef0: Float32,
):
    """`pcs_item`'s count sketch of degree p = t % degree for row t // degree,
    into sk[t * nc ..]: the same statements, features ascending."""
    var r = t // degree
    var p = t - r * degree
    var sg = ftz(identical_sqrt(gamma))
    var sc = ftz(identical_sqrt(coef0))
    var o = sk + t * nc
    for h in range(nc):
        o.unsafe_store(h, Float32(0))
    for j in range(nf):
        var v: Float32
        if j < d_in:
            v = ftz(identical_mul(sg, ftz(x.unsafe_load(r * d_in + j))))
        else:
            v = sc
        if Int(hbit.unsafe_load(p * nf + j)) < 0:
            v = -v
        var h = Int(hidx.unsafe_load(p * nf + j))
        o.unsafe_store(h, _add(o.unsafe_load(h), v))


def pcs_conv_item(t: Int, acc: FP, sk: FP, res: FP, n: Int, nc: Int, degree: Int, p: Int):
    """One cell (row t // nc, component t % nc) of `pcs_item`'s circular
    convolution of the running product `acc` (n x nc) with the row's degree-p
    sketch (sk, rows of degree * nc): the same fold, a ascending."""
    var r = t // nc
    var h = t - r * nc
    var ar = acc + r * nc
    var sr = sk + (r * degree + p) * nc
    var s = Float32(0)
    for a in range(nc):
        var b = h - a
        if b < 0:
            b += nc
        s = ftz(identical_mul_add(ar.unsafe_load(a), sr.unsafe_load(b), s))
    res.unsafe_store(t, s)


def pcs_copy0_item(t: Int, sk: FP, res: FP, n: Int, nc: Int, degree: Int):
    """res row r = the row's degree-0 sketch (`pcs_item`'s p == 0 copy)."""
    var r = t // nc
    var h = t - r * nc
    res.unsafe_store(t, sk.unsafe_load(r * degree * nc + h))


comptime PI_F32 = Float32(3.14159265358979323846)


def _coshf(z: Float32) -> Float32:
    return ftz(identical_mul(Float32(0.5), _add(ftz(identical_exp(z)), ftz(identical_exp(-z)))))


# DEVIATION 5213 (row 127)
def achi2_item(t: Int, x: FP, res: FP, n: Int, d: Int, steps: Int, interval: Float32):
    """sklearn AdditiveChi2Sampler._transform_dense for one input cell (t =
    i*d + f): sqrt(x L) into block 0, and for j = 1..steps-1
    sqrt(2 x L / cosh(pi j L)) * cos / sin(j L log x) into blocks 2j-1, 2j.
    A zero input writes zeros everywhere, as their `non_zero` mask."""
    var i = t // d
    var f = t - i * d
    var w = d * (2 * steps - 1)
    var xv = ftz(x.unsafe_load(t))
    if xv == Float32(0):
        for b in range(2 * steps - 1):
            res.unsafe_store(i * w + b * d + f, Float32(0))
        return
    res.unsafe_store(i * w + f, ftz(identical_sqrt(ftz(identical_mul(xv, interval)))))
    var log_step = ftz(identical_mul(interval, ftz(identical_log(xv))))
    var step = ftz(identical_mul(ftz(identical_mul(Float32(2), xv)), interval))
    for j in range(1, steps):
        var ch = _coshf(ftz(identical_mul(ftz(identical_mul(PI_F32, Float32(j))), interval)))
        var factor = ftz(identical_sqrt(ftz(identical_div(step, ch))))
        var arg = ftz(identical_mul(Float32(j), log_step))
        res.unsafe_store(i * w + (2 * j - 1) * d + f, ftz(identical_mul(factor, ftz(identical_cos(arg)))))
        res.unsafe_store(i * w + (2 * j) * d + f, ftz(identical_mul(factor, ftz(identical_sin(arg)))))


# DEVIATION 5213 (row 127)
def skew_weights_item(t: Int, z: FP, res: FP, count: Int):
    """SkewedChi2Sampler's inverse sech CDF: (1/pi) log(tan(z)), z = pi/2 u."""
    var zv = ftz(z.unsafe_load(t))
    var tn = ftz(identical_div(ftz(identical_sin(zv)), ftz(identical_cos(zv))))
    res.unsafe_store(t, ftz(identical_mul(ftz(identical_div(Float32(1), PI_F32)), ftz(identical_log(tn)))))


# DEVIATION 5213 (row 127)
def skew_transform_item(t: Int, lx: FP, w: FP, off: FP, res: FP, n: Int, d: Int, nc: Int):
    """cos(log(X + skewedness) @ W + offset) * sqrt(2) / sqrt(n_components),
    t = i*nc + c; the log was taken by the unary op, features ascending."""
    var i = t // nc
    var c = t - i * nc
    var acc = Float32(0)
    for f in range(d):
        acc = ftz(identical_mul_add(ftz(lx.unsafe_load(i * d + f)), ftz(w.unsafe_load(f * nc + c)), acc))
    var p = _add(acc, off.unsafe_load(c))
    var scale = ftz(identical_div(ftz(identical_sqrt(Float32(2))), ftz(identical_sqrt(Float32(nc)))))
    res.unsafe_store(t, ftz(identical_mul(ftz(identical_cos(p)), scale)))


# ------------------------------------------------------------------ label propagation / spreading
# DEVIATION 5209 (row 123)
def absdiff_sum_item(t: Int, a: FP, b: FP, res: FP, count: Int):
    """sum |a - b| over every element, ascending, ONE item."""
    var acc = Float32(0)
    for i in range(count):
        acc = _add(acc, abs(_sub(a.unsafe_load(i), b.unsafe_load(i))))
    res.unsafe_store(0, acc)


# DEVIATION 5209 (row 123)
def row_normalize_item(t: Int, a: FP, res: FP, n: Int, m: Int):
    """a / rowsum (a zero row sum divides by 1, as their `normalizer == 0`)."""
    var s = Float32(0)
    for j in range(m):
        s = _add(s, a.unsafe_load(t * m + j))
    if s == Float32(0):
        s = Float32(1)
    for j in range(m):
        res.unsafe_store(t * m + j, ftz(identical_div(ftz(a.unsafe_load(t * m + j)), s)))


# DEVIATION 5214 (row 128)
def lp_clamp_item(t: Int, ld: FP, ystatic: FP, unlabeled: IP, res: FP, n: Int, c: Int):
    """LabelPropagation's step after the product: normalize the row, then a
    labeled row takes its static distribution back."""
    if Int(unlabeled.unsafe_load(t)) == 0:
        for j in range(c):
            res.unsafe_store(t * c + j, ystatic.unsafe_load(t * c + j))
        return
    row_normalize_item(t, ld, res, n, c)


# DEVIATION 5214 (row 128)
def ls_clamp_item(t: Int, ld: FP, ystatic: FP, res: FP, count: Int, alpha: Float32):
    """LabelSpreading's clamp: alpha * ld + y_static (their multiply, then add)."""
    res.unsafe_store(t, _add(ftz(identical_mul(alpha, ftz(ld.unsafe_load(t)))), ystatic.unsafe_load(t)))


# DEVIATION 5214 (row 128)
def ls_laplacian_item(t: Int, a: FP, res: FP, n: Int):
    """-csgraph.laplacian(A, normed=True) with the diagonal zeroed (sklearn
    LabelSpreading._build_graph): degrees are IN-degrees (column sums, scipy's
    default) with the diagonal excluded, w = sqrt(degree) (1 where the degree
    is 0), entry A_ij / w_j / w_i in scipy's order. t = i*n + j."""
    var i = t // n
    var j = t - i * n
    if i == j:
        res.unsafe_store(t, Float32(0))
        return
    var di = Float32(0)
    var dj = Float32(0)
    for k in range(n):
        if k != i:
            di = _add(di, a.unsafe_load(k * n + i))
        if k != j:
            dj = _add(dj, a.unsafe_load(k * n + j))
    var wi = ftz(identical_sqrt(di)) if di != Float32(0) else Float32(1)
    var wj = ftz(identical_sqrt(dj)) if dj != Float32(0) else Float32(1)
    var v = ftz(identical_div(ftz(a.unsafe_load(t)), wj))
    res.unsafe_store(t, ftz(identical_div(v, wi)))


def col_degree_item(t: Int, a: FP, res: FP, n: Int):
    """`ls_laplacian_item`'s in-degree of node t: column t of A summed over
    the rows k != t, ascending, the same `_add` chain."""
    var dt = Float32(0)
    for k in range(n):
        if k != t:
            dt = _add(dt, a.unsafe_load(k * n + t))
    res.unsafe_store(t, dt)


def ls_laplacian_deg_item(t: Int, a: FP, deg: FP, res: FP, n: Int):
    """`ls_laplacian_item` with the two degrees read from `col_degree_item`'s
    output instead of refolded per cell (O(n^2) instead of O(n^3); every
    stored value the same)."""
    var i = t // n
    var j = t - i * n
    if i == j:
        res.unsafe_store(t, Float32(0))
        return
    var di = deg.unsafe_load(i)
    var dj = deg.unsafe_load(j)
    var wi = ftz(identical_sqrt(di)) if di != Float32(0) else Float32(1)
    var wj = ftz(identical_sqrt(dj)) if dj != Float32(0) else Float32(1)
    var v = ftz(identical_div(ftz(a.unsafe_load(t)), wj))
    res.unsafe_store(t, ftz(identical_div(v, wi)))


def row_all_zero_item(t: Int, a: IP, res: IP, n: Int, m: Int):
    """1 when every entry of row t of the float32 matrix (read as its bit
    patterns) is +0.0 or -0.0, i.e. Python's `all(v == 0 for v in row)`
    (a NaN is not zero), else 0."""
    var z = 1
    for j in range(m):
        var b = bitcast[DType.uint32](a.unsafe_load(t * m + j)) & UInt32(0x7FFFFFFF)
        if b != UInt32(0):
            z = 0
            break
    res.unsafe_store(t, Int32(z))


def knn_graph_item(t: Int, idx: IP, res: FP, n: Int, m: Int, k: Int):
    """Row t of the connectivity graph: 1 at each of the row's k neighbors."""
    for j in range(m):
        res.unsafe_store(t * m + j, Float32(0))
    for s in range(k):
        var j = Int(idx.unsafe_load(t * k + s))
        if j >= 0:
            res.unsafe_store(t * m + j, Float32(1))


# ------------------------------------------------------------------ KNNImputer
# DEVIATION 5215 (row 121)
@always_inline
def knn_impute_finish(
    t: Int, fx: FP, bd: FP, bi: IP, res: FP, m: Int, d: Int, k: Int, weights: Int, n_donors: Int,
):
    """`knn_impute_item`'s tail after the donor scan: the fallback column
    mean or the (weighted) donor mean, stored at cell t."""
    var c = t - (t // d) * d
    var kk = k if k < n_donors else n_donors
    var found = 0
    for s in range(kk):
        if Int(bi.unsafe_load(s)) >= 0:
            found += 1
    if found == 0:
        var acc = Float32(0)
        var cnt = 0
        for j in range(m):
            var dv = fx.unsafe_load(j * d + c)
            if dv == dv:
                acc = _add(acc, dv)
                cnt += 1
        res.unsafe_store(t, ftz(identical_div(acc, Float32(cnt))) if cnt > 0 else Float32(0))
        return
    var any_zero = False
    for s in range(found):
        if bd.unsafe_load(s) == Float32(0):
            any_zero = True
    var num = Float32(0)
    var den = Float32(0)
    for s in range(found):
        var w = Float32(1)
        if weights == 1:
            if any_zero:
                w = Float32(1) if bd.unsafe_load(s) == Float32(0) else Float32(0)
            else:
                w = ftz(identical_div(Float32(1), bd.unsafe_load(s)))
        var val = fx.unsafe_load(Int(bi.unsafe_load(s)) * d + c)
        num = ftz(identical_mul_add(ftz(val), w, num))
        den = _add(den, w)
    res.unsafe_store(t, ftz(identical_div(num, den)))


def knn_impute_item(
    t: Int, x: FP, fx: FP, best_d: FP, best_i: IP, res: FP,
    n: Int, m: Int, d: Int, k: Int, weights: Int,
):
    """sklearn KNNImputer.transform for cell t = r*d + c: a present value is
    copied; a missing one is the (weighted) mean of column c over the k
    nearest donors (fit rows with column c present) by nan_euclidean
    distance, nearest first and the lower donor index on a tie; donors at no
    finite distance are skipped (their weight 0). No donor at a finite
    distance: the masked column mean of the fit data. weights: 0 uniform,
    1 distance (1/d; any zero distance: the zero-distance donors only).
    best_d / best_i are k slots of scratch per cell."""
    var r = t // d
    var c = t - r * d
    var v = x.unsafe_load(t)
    if v == v:
        res.unsafe_store(t, v)
        return
    var inf = bitcast[DType.float32](UInt32(0x7F800000))
    var bd = best_d + t * k
    var bi = best_i + t * k
    for s in range(k):
        bd.unsafe_store(s, inf)
        bi.unsafe_store(s, Int32(-1))
    var n_donors = 0
    for j in range(m):
        var dv = fx.unsafe_load(j * d + c)
        if dv != dv:
            continue
        n_donors += 1
        var acc = Float32(0)
        var present = 0
        for f in range(d):
            var a = x.unsafe_load(r * d + f)
            var b = fx.unsafe_load(j * d + f)
            if a != a or b != b:
                continue
            present += 1
            var df = _sub(a, b)
            acc = ftz(identical_mul_add(df, df, acc))
        if present == 0:
            continue
        var sq = ftz(identical_mul(ftz(identical_div(acc, Float32(present))), Float32(d)))
        var dist = ftz(identical_sqrt(sq))
        if not (dist < bd.unsafe_load(k - 1)):
            continue
        var s = k - 1
        while s > 0 and dist < bd.unsafe_load(s - 1):
            bd.unsafe_store(s, bd.unsafe_load(s - 1))
            bi.unsafe_store(s, bi.unsafe_load(s - 1))
            s -= 1
        bd.unsafe_store(s, dist)
        bi.unsafe_store(s, Int32(j))
    knn_impute_finish(t, fx, bd, bi, res, m, d, k, weights, n_donors)


def knn_impute_cell_item(
    t: Int, cells: IP, x: FP, fx: FP, best_d: FP, best_i: IP, res: FP,
    n: Int, m: Int, d: Int, k: Int, weights: Int, nc: Int,
):
    """`knn_impute_item` for the t-th MISSING cell of a compact list (cell
    ids ascending): the same statements for that cell, so a GPU thread per
    missing cell instead of one per cell, most of which return at once. The
    caller seeds `res` with x, which is what the item stores for a present
    cell."""
    knn_impute_item(Int(cells.unsafe_load(t)), x, fx, best_d, best_i, res, n, m, d, k, weights)


# ------------------------------------------------------------------ graphs
# DEVIATION 5216 (row 129)
def pagerank_step_item(t: Int, q: FP, x: FP, p: FP, dw: FP, dangling: IP, res: FP, n: Int, alpha: Float32):
    """One power-iteration step for node t (networkx `_pagerank_scipy`;
    cuGraph cpp/src/link_analysis/pagerank_impl.cuh): alpha * (x @ Q +
    sum(x[dangling]) * dw) + (1 - alpha) * p, both folds ascending by node;
    dw is networkx's `dangling` weights (the personalization when not given)."""
    var acc = Float32(0)
    for i in range(n):
        acc = ftz(identical_mul_add(ftz(x.unsafe_load(i)), ftz(q.unsafe_load(i * n + t)), acc))
    var dsum = Float32(0)
    for i in range(n):
        if Int(dangling.unsafe_load(i)) != 0:
            dsum = _add(dsum, x.unsafe_load(i))
    var pt = ftz(p.unsafe_load(t))
    var inner = ftz(identical_mul_add(dsum, ftz(dw.unsafe_load(t)), acc))
    var teleport = ftz(identical_mul(_sub(Float32(1), alpha), pt))
    res.unsafe_store(t, ftz(identical_mul_add(alpha, inner, teleport)))


# DEVIATION 5217 (row 129)
def cc_step_item(t: Int, a: FP, lab: IP, res: IP, n: Int):
    """Weak connectivity as a product (DBSCAN's weak_cc; cuGraph
    weakly_connected_components_impl.cuh): the smallest label among the node
    and its neighbors in either direction. Integers only."""
    var best = lab.unsafe_load(t)
    for j in range(n):
        if a.unsafe_load(t * n + j) != Float32(0) or a.unsafe_load(j * n + t) != Float32(0):
            var l = lab.unsafe_load(j)
            if l < best:
                best = l
    res.unsafe_store(t, best)


def _louvain_modularity(w: FP, comm: IP, n: Int, m: Float32, resolution: Float32, tot: FP, inner: FP) -> Float32:
    """networkx `modularity` on the dense symmetric graph w (self-loops on the
    diagonal, each edge counted once): sum_c L_c / m - resolution (deg_c / 2m)^2,
    communities folded in ascending id. tot / inner are n floats of scratch."""
    for c in range(n):
        tot.unsafe_store(c, Float32(0))
        inner.unsafe_store(c, Float32(0))
    for u in range(n):
        var cu = Int(comm.unsafe_load(u))
        var deg = Float32(0)
        for v in range(n):
            var wv = w.unsafe_load(u * n + v)
            deg = _add(deg, wv)
            if v == u:
                deg = _add(deg, wv)
                inner.unsafe_store(cu, _add(inner.unsafe_load(cu), wv))
            elif v > u and Int(comm.unsafe_load(v)) == cu:
                inner.unsafe_store(cu, _add(inner.unsafe_load(cu), wv))
        tot.unsafe_store(cu, _add(tot.unsafe_load(cu), deg))
    var q = Float32(0)
    var two_m = ftz(identical_mul(Float32(2), m))
    for c in range(n):
        var lc = ftz(identical_div(inner.unsafe_load(c), m))
        var fr = ftz(identical_div(tot.unsafe_load(c), two_m))
        q = _add(q, _sub(lc, ftz(identical_mul(resolution, ftz(identical_mul(fr, fr))))))
    return q


# DEVIATION 5204 (row 129)
def louvain_item(
    t: Int, a: FP, labels: IP, info: FP, w: FP, w2: FP, comm: IP, node_of: IP, deg: FP, stot: FP, k2c: FP, tmp: FP,
    n: Int, max_level: Int, resolution: Float32, threshold: Float32,
):
    """Louvain community detection, sequential, ONE item (networkx
    `louvain_partitions` / `_one_level` / `_gen_graph`; cuGraph
    cpp/src/community/louvain_impl.cuh is the parallel, order-dependent
    reference). PINNED ORDER (DEVIATION 5204): nodes are visited in ascending
    id (networkx shuffles them by `seed`), candidate communities are scanned
    in ascending id and a move needs a STRICTLY larger gain, so equal gains
    go to the lowest community id. Float32 folds in ascending order. `a` is
    the n x n symmetric weight matrix (diagonal = self-loops). labels[u]
    (original node u) ends as its community, numbered by first appearance.
    info = [modularity, levels]."""
    # m = total edge weight, each undirected edge once, self-loops once
    var m = Float32(0)
    for u in range(n):
        for v in range(u, n):
            m = _add(m, a.unsafe_load(u * n + v))
    var two_m2 = ftz(identical_mul(Float32(2), ftz(identical_mul(m, m))))
    var nn = n
    for u in range(n):
        labels.unsafe_store(u, Int32(u))
        for v in range(n):
            w.unsafe_store(u * n + v, a.unsafe_load(u * n + v))
    for u in range(n):
        comm.unsafe_store(u, Int32(u))
    var levels = 0
    var mod = _louvain_modularity(w, comm, nn, m, resolution, tmp, k2c)
    while max_level <= 0 or levels < max_level:
        # ---- networkx `_one_level` on the current graph w (nn nodes) ----
        for u in range(nn):
            comm.unsafe_store(u, Int32(u))
            var dg = Float32(0)
            for v in range(nn):
                dg = _add(dg, w.unsafe_load(u * nn + v))
            dg = _add(dg, w.unsafe_load(u * nn + u))
            deg.unsafe_store(u, dg)
            stot.unsafe_store(u, dg)
        var improvement = False
        var moves = 1
        while moves > 0:
            moves = 0
            for u in range(nn):
                var cu = Int(comm.unsafe_load(u))
                for c in range(nn):
                    k2c.unsafe_store(c, Float32(0))
                for v in range(nn):
                    if v != u:
                        var wv = w.unsafe_load(u * nn + v)
                        if wv != Float32(0):
                            var cv = Int(comm.unsafe_load(v))
                            k2c.unsafe_store(cv, _add(k2c.unsafe_load(cv), wv))
                var du = deg.unsafe_load(u)
                stot.unsafe_store(cu, _sub(stot.unsafe_load(cu), du))
                var remove_cost = _add(
                    -ftz(identical_div(k2c.unsafe_load(cu), m)),
                    ftz(identical_div(ftz(identical_mul(resolution, ftz(identical_mul(stot.unsafe_load(cu), du)))), two_m2)),
                )
                var best = cu
                var best_gain = Float32(0)
                for c in range(nn):
                    var kc = k2c.unsafe_load(c)
                    if kc == Float32(0):
                        continue
                    var gain = _sub(
                        _add(remove_cost, ftz(identical_div(kc, m))),
                        ftz(identical_div(ftz(identical_mul(resolution, ftz(identical_mul(stot.unsafe_load(c), du)))), two_m2)),
                    )
                    if gain > best_gain:
                        best_gain = gain
                        best = c
                stot.unsafe_store(best, _add(stot.unsafe_load(best), du))
                if best != cu:
                    comm.unsafe_store(u, Int32(best))
                    moves += 1
                    improvement = True
        if levels > 0 and not improvement:
            break
        # renumber by ascending old community id (their filter(len, partition))
        for c in range(nn):
            node_of.unsafe_store(c, Int32(-1))
        var nc = 0
        for c in range(nn):
            var used = False
            for u in range(nn):
                if Int(comm.unsafe_load(u)) == c:
                    used = True
                    break
            if used:
                node_of.unsafe_store(c, Int32(nc))
                nc += 1
        for u in range(nn):
            comm.unsafe_store(u, node_of.unsafe_load(Int(comm.unsafe_load(u))))
        for u in range(n):
            labels.unsafe_store(u, comm.unsafe_load(Int(labels.unsafe_load(u))))
        levels += 1
        var new_mod = _louvain_modularity(w, comm, nn, m, resolution, tmp, k2c)
        if not (_sub(new_mod, mod) > threshold):
            break
        mod = new_mod
        # aggregate (their _gen_graph): W'[c,d] = sum of w[u,v] over u in c, v in d,
        # each undirected edge once; an edge inside c becomes c's self-loop
        for i in range(nc * nc):
            w2.unsafe_store(i, Float32(0))
        for u in range(nn):
            var cu2 = Int(comm.unsafe_load(u))
            for v in range(u, nn):
                var wv = w.unsafe_load(u * nn + v)
                if wv == Float32(0):
                    continue
                var cv2 = Int(comm.unsafe_load(v))
                w2.unsafe_store(cu2 * nc + cv2, _add(w2.unsafe_load(cu2 * nc + cv2), wv))
                if cu2 != cv2:
                    w2.unsafe_store(cv2 * nc + cu2, _add(w2.unsafe_load(cv2 * nc + cu2), wv))
        for i in range(nc * nc):
            w.unsafe_store(i, w2.unsafe_load(i))
        nn = nc
        for u in range(nn):
            comm.unsafe_store(u, Int32(u))
    info.unsafe_store(0, _louvain_modularity(a, labels, n, m, resolution, tmp, k2c))
    info.unsafe_store(1, Float32(levels))


# ------------------------------------------------------------------ SVGP
# DEVIATION 5205 (row 129)
def _chol_inplace(a: FP, m: Int) -> Bool:
    """Lower Cholesky of the m x m row-major a, in place (upper triangle
    zeroed), columns left to right, each fold ascending. False when a pivot
    is not positive."""
    for j in range(m):
        var s = a.unsafe_load(j * m + j)
        for k in range(j):
            var l = a.unsafe_load(j * m + k)
            s = ftz(identical_mul_add(-l, l, s))
        if not (s > Float32(0)):
            return False
        var d = ftz(identical_sqrt(s))
        a.unsafe_store(j * m + j, d)
        for i in range(j + 1, m):
            var t = a.unsafe_load(i * m + j)
            for k in range(j):
                t = ftz(identical_mul_add(-a.unsafe_load(i * m + k), a.unsafe_load(j * m + k), t))
            a.unsafe_store(i * m + j, ftz(identical_div(t, d)))
        for i in range(j):
            a.unsafe_store(i * m + j, Float32(0))
    return True


def _chol_solve(l: FP, m: Int, b: FP, x: FP):
    """x = (L L^T)^-1 b: forward then back substitution, ascending folds."""
    for i in range(m):
        var s = b.unsafe_load(i)
        for k in range(i):
            s = ftz(identical_mul_add(-l.unsafe_load(i * m + k), x.unsafe_load(k), s))
        x.unsafe_store(i, ftz(identical_div(s, l.unsafe_load(i * m + i))))
    for ii in range(m):
        var i = m - 1 - ii
        var s = x.unsafe_load(i)
        for k in range(i + 1, m):
            s = ftz(identical_mul_add(-l.unsafe_load(k * m + i), x.unsafe_load(k), s))
        x.unsafe_store(i, ftz(identical_div(s, l.unsafe_load(i * m + i))))


def _log_diag_sum(l: FP, m: Int) -> Float32:
    var s = Float32(0)
    for i in range(m):
        s = _add(s, ftz(identical_log(l.unsafe_load(i * m + i))))
    return s


# DEVIATION 5205 (row 129)
def svgp_item(
    t: Int, kuu: FP, bmat: FP, b: FP, y: FP, alpha: FP, cmat: FP, qmu: FP, qsqrt: FP, info: FP,
    luu: FP, ls: FP, e: FP, col: FP,
    m: Int, n: Int, noise: Float32, jitter: Float32, kdiag: Float32,
):
    """The SVGP with a Gaussian likelihood at its OPTIMAL variational
    distribution (Titsias 2009; GPflow gpflow/models/svgp.py `elbo` reaches
    this bound at its optimum q), ONE sequential item on the m x m system:
    Sigma = Kuu + jitter I + B / noise with B = Kuf Kfu; alpha = Sigma^-1 b /
    noise with b = Kuf y (the predictive mean is K*u alpha); C = Kuu^-1 -
    Sigma^-1 (the predictive variance is k** - K*u C Ku*); q_mu = Kuu alpha,
    q_sqrt = chol(Kuu Sigma^-1 Kuu); info = [elbo, ok]. Cholesky, the
    triangular solves and every fold in ascending order."""
    for i in range(m * m):
        luu.unsafe_store(i, kuu.unsafe_load(i))
    for i in range(m):
        luu.unsafe_store(i * m + i, _add(luu.unsafe_load(i * m + i), jitter))
    for i in range(m * m):
        ls.unsafe_store(i, _add(luu.unsafe_load(i), ftz(identical_div(bmat.unsafe_load(i), noise))))
    var ok1 = _chol_inplace(luu, m)
    var ok2 = _chol_inplace(ls, m)
    if not (ok1 and ok2):
        info.unsafe_store(0, Float32(0))
        info.unsafe_store(1, Float32(0))
        return
    # alpha = Sigma^-1 b / noise
    _chol_solve(ls, m, b, alpha)
    for i in range(m):
        alpha.unsafe_store(i, ftz(identical_div(alpha.unsafe_load(i), noise)))
    # C = Kuu^-1 - Sigma^-1, column by column; e / col are m floats of scratch
    for j in range(m):
        for i in range(m):
            e.unsafe_store(i, Float32(1) if i == j else Float32(0))
        _chol_solve(luu, m, e, col)
        for i in range(m):
            cmat.unsafe_store(i * m + j, col.unsafe_load(i))
        _chol_solve(ls, m, e, col)
        for i in range(m):
            cmat.unsafe_store(i * m + j, _sub(cmat.unsafe_load(i * m + j), col.unsafe_load(i)))
    # q_mu = Kuu alpha (the jittered Kuu, as the solves)
    for i in range(m):
        var s = Float32(0)
        for k in range(m):
            var kv = kuu.unsafe_load(i * m + k)
            if i == k:
                kv = _add(kv, jitter)
            s = ftz(identical_mul_add(kv, alpha.unsafe_load(k), s))
        qmu.unsafe_store(i, s)
    # S = Kuu Sigma^-1 Kuu, then its Cholesky into q_sqrt
    for j in range(m):
        for i in range(m):
            var kv = kuu.unsafe_load(i * m + j)
            if i == j:
                kv = _add(kv, jitter)
            e.unsafe_store(i, kv)
        _chol_solve(ls, m, e, col)
        for i in range(m):
            var s = Float32(0)
            for k in range(m):
                var kv = kuu.unsafe_load(i * m + k)
                if i == k:
                    kv = _add(kv, jitter)
                s = ftz(identical_mul_add(kv, col.unsafe_load(k), s))
            qsqrt.unsafe_store(i * m + j, s)
    var ok3 = _chol_inplace(qsqrt, m)
    # the collapsed bound
    var yty = Float32(0)
    for i in range(n):
        var yv = y.unsafe_load(i)
        yty = ftz(identical_mul_add(yv, yv, yty))
    _chol_solve(ls, m, b, col)
    var bsb = Float32(0)
    for i in range(m):
        bsb = ftz(identical_mul_add(b.unsafe_load(i), col.unsafe_load(i), bsb))
    var quad = _sub(ftz(identical_div(yty, noise)), ftz(identical_div(bsb, ftz(identical_mul(noise, noise)))))
    var logdet = _add(
        ftz(identical_mul(Float32(2), _sub(_log_diag_sum(ls, m), _log_diag_sum(luu, m)))),
        ftz(identical_mul(Float32(n), ftz(identical_log(noise)))),
    )
    # tr(Kuu^-1 B): column solves against B
    var trq = Float32(0)
    for j in range(m):
        for i in range(m):
            e.unsafe_store(i, bmat.unsafe_load(i * m + j))
        _chol_solve(luu, m, e, col)
        trq = _add(trq, col.unsafe_load(j))
    var trace_term = ftz(identical_div(_sub(ftz(identical_mul(Float32(n), kdiag)), trq), noise))
    var log2pi = Float32(1.8378770664093453)
    var elbo = -ftz(identical_mul(Float32(0.5), _add(_add(ftz(identical_mul(Float32(n), log2pi)), logdet), _add(quad, trace_term))))
    info.unsafe_store(0, elbo)
    info.unsafe_store(1, Float32(1) if ok3 else Float32(0))


# DEVIATION 5209 (row 123), lane/py-dn-kern 2026-09-28
def matmul_tn_acc_item(t: Int, a: FP, b: FP, res: FP, rows: Int, n: Int, m: Int):
    """res (n x m) continued by A^T B over `rows` rows (A rows x n, B rows x m),
    t = i*m + j: `matmul_item`'s fold of (A^T)[i, p] B[p, j], p ascending,
    carried on from res[t] (0 before the first rows). A fold cut into
    consecutive row ranges and carried through float32 res is the same
    sequence of pinned fmas as the uncut fold, so the bits are those of
    `matmul(A^T, B)` with A^T the transposed copy."""
    var i = t // m
    var j = t - i * m
    var acc = res.unsafe_load(t)
    for p in range(rows):
        acc = ftz(identical_mul_add(ftz(a.unsafe_load(p * n + i)), ftz(b.unsafe_load(p * m + j)), acc))
    res.unsafe_store(t, acc)


# DEVIATION 5205 (row 129)
def svgp_var_item(t: Int, ksu: FP, cmat: FP, res: FP, n: Int, m: Int, kdiag: Float32):
    """Predictive variance of f at row t: k** - K*u C Ku*, the inner fold
    per row of C ascending, then the outer ascending."""
    var acc = Float32(0)
    for i in range(m):
        var s = Float32(0)
        for j in range(m):
            s = ftz(identical_mul_add(cmat.unsafe_load(i * m + j), ksu.unsafe_load(t * m + j), s))
        acc = ftz(identical_mul_add(ksu.unsafe_load(t * m + i), s, acc))
    res.unsafe_store(t, _sub(kdiag, acc))
