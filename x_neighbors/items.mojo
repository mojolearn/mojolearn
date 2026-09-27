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
