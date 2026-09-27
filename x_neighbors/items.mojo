# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE NEIGHBORS EXPANSION LANE'S ONE SOURCE (docs/lanes/ALGORITHM_EXPANSION_BRIEFS.md,
"Lane 3: neighbors + kernel").

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
def matmul_item(t: Int, a: FP, b: FP, res: FP, n: Int, k: Int, m: Int):
    """C = A (n x k) B (k x m), t = i*m + j, ascending p, pinned fma."""
    var i = t // m
    var j = t - i * m
    var acc = Float32(0)
    for p in range(k):
        acc = ftz(identical_mul_add(ftz(a.unsafe_load(i * k + p)), ftz(b.unsafe_load(p * m + j)), acc))
    res.unsafe_store(t, acc)


def rowsum_item(t: Int, a: FP, res: FP, n: Int, m: Int):
    var acc = Float32(0)
    for j in range(m):
        acc = _add(acc, a.unsafe_load(t * m + j))
    res.unsafe_store(t, acc)


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


def ocsvm_smo_item(t: Int, q: FP, alpha: FP, g: FP, info: FP, iters: IP, n: Int, eps: Float32, max_iter: Int):
    """libsvm's `Solver::Solve` for the one-class problem (sklearn
    `svm/src/libsvm/svm.cpp`: `solve_one_class`, `Solver::Solve`,
    `select_working_set` (WSS3, second order), `calculate_rho`), with every
    y = +1, p = 0 and C = 1, no shrinking, SEQUENTIAL in one item. `alpha`
    arrives holding libsvm's initial point (the first floor(nu*l) ones and
    the fractional remainder); `q` is the n x n kernel matrix. Float32 with
    the pinned spellings where libsvm computes in double (DEVIATION 5200).
    Ties in the working-set scans resolve as libsvm's `>=` / `<=` dres: the
    LAST index of equal gradient wins. info[0] = rho."""
    var neg_inf = bitcast[DType.float32](UInt32(0xFF800000))
    var pos_inf = bitcast[DType.float32](UInt32(0x7F800000))
    for i in range(n):
        var acc = Float32(0)
        for j in range(n):
            var a = alpha.unsafe_load(j)
            if a != Float32(0):
                acc = ftz(identical_mul_add(ftz(q.unsafe_load(i * n + j)), a, acc))
        g.unsafe_store(i, acc)
    var it = 0
    while it < max_iter:
        var gmax = neg_inf
        var gi = -1
        for t in range(n):
            if alpha.unsafe_load(t) < Float32(1):
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
                        var two_q = ftz(identical_mul(Float32(2), q.unsafe_load(gi * n + j)))
                        var quad = _sub(_add(qdi, q.unsafe_load(j * n + j)), two_q)
                        var num = ftz(identical_mul(grad_diff, grad_diff))
                        var obj: Float32
                        if quad > Float32(0):
                            obj = -ftz(identical_div(num, quad))
                        else:
                            obj = -ftz(identical_div(num, SMO_TAU))
                        if obj <= obj_min:
                            gj = j
                            obj_min = obj
        if gi < 0 or gj < 0 or _add(gmax, gmax2) < eps:
            break
        it += 1
        var i = gi
        var j = gj
        var old_ai = alpha.unsafe_load(i)
        var old_aj = alpha.unsafe_load(j)
        var quad = _sub(_add(q.unsafe_load(i * n + i), q.unsafe_load(j * n + j)), ftz(identical_mul(Float32(2), q.unsafe_load(i * n + j))))
        if quad <= Float32(0):
            quad = SMO_TAU
        var delta = ftz(identical_div(_sub(g.unsafe_load(i), g.unsafe_load(j)), quad))
        var total = _add(old_ai, old_aj)
        var ai = _sub(old_ai, delta)
        var aj = _add(old_aj, delta)
        if total > Float32(1):
            if ai > Float32(1):
                ai = Float32(1)
                aj = _sub(total, Float32(1))
        else:
            if aj < Float32(0):
                aj = Float32(0)
                ai = total
        if total > Float32(1):
            if aj > Float32(1):
                aj = Float32(1)
                ai = _sub(total, Float32(1))
        else:
            if ai < Float32(0):
                ai = Float32(0)
                aj = total
        alpha.unsafe_store(i, ai)
        alpha.unsafe_store(j, aj)
        var dai = _sub(ai, old_ai)
        var daj = _sub(aj, old_aj)
        for k in range(n):
            var gk = g.unsafe_load(k)
            gk = ftz(identical_mul_add(ftz(q.unsafe_load(i * n + k)), dai, gk))
            gk = ftz(identical_mul_add(ftz(q.unsafe_load(j * n + k)), daj, gk))
            g.unsafe_store(k, gk)
    # calculate_rho, y = +1 throughout
    var ub = pos_inf
    var lb = neg_inf
    var nr_free = 0
    var sum_free = Float32(0)
    for i in range(n):
        var yg = g.unsafe_load(i)
        var a = alpha.unsafe_load(i)
        if a >= Float32(1):
            if yg > lb:
                lb = yg
        elif a <= Float32(0):
            if yg < ub:
                ub = yg
        else:
            nr_free += 1
            sum_free = _add(sum_free, yg)
    var rho: Float32
    if nr_free > 0:
        rho = ftz(identical_div(sum_free, Float32(nr_free)))
    else:
        rho = ftz(identical_mul(_add(ub, lb), Float32(0.5)))
    info.unsafe_store(0, rho)
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


def lof_score_item(t: Int, idx: IP, fit_lrd: FP, lrd: FP, score: FP, k: Int, n: Int, n_fit: Int):
    """-mean(lrd[neighbors] / lrd[t]): the ratios folded ascending by rank."""
    var own = lrd.unsafe_load(t)
    var acc = Float32(0)
    for j in range(k):
        acc = _add(acc, ftz(identical_div(fit_lrd.unsafe_load(Int(idx.unsafe_load(t * k + j))), own)))
    score.unsafe_store(t, -ftz(identical_div(acc, Float32(k))))


# ------------------------------------------------------------------ distances, L1
def l1dist_item(t: Int, x: FP, y: FP, res: FP, n: Int, m: Int, d: Int):
    var i = t // m
    var j = t - i * m
    var acc = Float32(0)
    for f in range(d):
        acc = _add(acc, abs(_sub(x.unsafe_load(i * d + f), y.unsafe_load(j * d + f))))
    res.unsafe_store(t, acc)


# ------------------------------------------------------------------ KernelPCA
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


def nc_shrink_item(
    t: Int, x: FP, cent: FP, nk: FP, std: FP, res: FP,
    n: Int, d: Int, n_classes: Int, do_shrink: Int, med: Float32, shrink: Float32,
):
    """sklearn's shrunken centroid for class k, feature f (t = k*d + f): the
    dataset centroid (ascending rows, one division), m = sqrt(1/n_k - 1/n),
    s = std + median(std), deviation = (centroid - dataset centroid) / (m*s),
    soft-thresholded by `shrink`, centroid = dataset centroid + m*s*deviation.
    Without shrinking the centroid is returned unchanged. DEVIATION 5201: m*s
    == 0 gives deviation 0."""
    var k = t // d
    var f = t - k * d
    var c = cent.unsafe_load(t)
    if do_shrink == 0:
        res.unsafe_store(t, c)
        return
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
    var mag = _sub(abs(dev), shrink)
    if mag < Float32(0):
        mag = Float32(0)
    if dev < Float32(0):
        dev = -mag
    elif dev > Float32(0):
        dev = mag
    else:
        dev = Float32(0)
    res.unsafe_store(t, _add(dsc, ftz(identical_mul(ms, dev))))


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


# ------------------------------------------------------------------ kernel approximation
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


comptime PI_F32 = Float32(3.14159265358979323846)


def _coshf(z: Float32) -> Float32:
    return ftz(identical_mul(Float32(0.5), _add(ftz(identical_exp(z)), ftz(identical_exp(-z)))))


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


def skew_weights_item(t: Int, z: FP, res: FP, count: Int):
    """SkewedChi2Sampler's inverse sech CDF: (1/pi) log(tan(z)), z = pi/2 u."""
    var zv = ftz(z.unsafe_load(t))
    var tn = ftz(identical_div(ftz(identical_sin(zv)), ftz(identical_cos(zv))))
    res.unsafe_store(t, ftz(identical_mul(ftz(identical_div(Float32(1), PI_F32)), ftz(identical_log(tn)))))


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
def absdiff_sum_item(t: Int, a: FP, b: FP, res: FP, count: Int):
    """sum |a - b| over every element, ascending, ONE item."""
    var acc = Float32(0)
    for i in range(count):
        acc = _add(acc, abs(_sub(a.unsafe_load(i), b.unsafe_load(i))))
    res.unsafe_store(0, acc)


def row_normalize_item(t: Int, a: FP, res: FP, n: Int, m: Int):
    """a / rowsum (a zero row sum divides by 1, as their `normalizer == 0`)."""
    var s = Float32(0)
    for j in range(m):
        s = _add(s, a.unsafe_load(t * m + j))
    if s == Float32(0):
        s = Float32(1)
    for j in range(m):
        res.unsafe_store(t * m + j, ftz(identical_div(ftz(a.unsafe_load(t * m + j)), s)))


def lp_clamp_item(t: Int, ld: FP, ystatic: FP, unlabeled: IP, res: FP, n: Int, c: Int):
    """LabelPropagation's step after the product: normalize the row, then a
    labeled row takes its static distribution back."""
    if Int(unlabeled.unsafe_load(t)) == 0:
        for j in range(c):
            res.unsafe_store(t * c + j, ystatic.unsafe_load(t * c + j))
        return
    row_normalize_item(t, ld, res, n, c)


def ls_clamp_item(t: Int, ld: FP, ystatic: FP, res: FP, count: Int, alpha: Float32):
    """LabelSpreading's clamp: alpha * ld + y_static (their multiply, then add)."""
    res.unsafe_store(t, _add(ftz(identical_mul(alpha, ftz(ld.unsafe_load(t)))), ystatic.unsafe_load(t)))


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


def knn_graph_item(t: Int, idx: IP, res: FP, n: Int, m: Int, k: Int):
    """Row t of the connectivity graph: 1 at each of the row's k neighbors."""
    for j in range(m):
        res.unsafe_store(t * m + j, Float32(0))
    for s in range(k):
        var j = Int(idx.unsafe_load(t * k + s))
        if j >= 0:
            res.unsafe_store(t * m + j, Float32(1))


# ------------------------------------------------------------------ KNNImputer
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

