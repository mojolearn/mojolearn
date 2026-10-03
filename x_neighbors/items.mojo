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
from std.sys.compile import is_defined
from std.sys.info import is_gpu

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
def ocsvm_pair(
    q: FP, cv: FP, n: Int, i: Int, j: Int, old_ai: Float32, old_aj: Float32, g_i: Float32, g_j: Float32,
) -> Tuple[Float32, Float32]:
    """The two-variable step on (i, j) from alpha_i, alpha_j and gradients
    g_i, g_j: the new (alpha_i, alpha_j). Pure, so every GPU block of the
    step launch computes it from the values the selection carried."""
    var quad = _sub(_add(q.unsafe_load(i * n + i), q.unsafe_load(j * n + j)), ftz(identical_mul(Float32(2), q.unsafe_load(i * n + j))))
    if quad <= Float32(0):
        quad = SMO_TAU
    var delta = ftz(identical_div(_sub(g_i, g_j), quad))
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
    return (ai, aj)


@always_inline
def ocsvm_update(q: FP, cv: FP, alpha: FP, g: FP, n: Int, i: Int, j: Int) -> Tuple[Float32, Float32]:
    """The two-variable step on (i, j): stores alpha_i, alpha_j and returns
    their changes (dai, daj)."""
    var old_ai = alpha.unsafe_load(i)
    var old_aj = alpha.unsafe_load(j)
    var a = ocsvm_pair(q, cv, n, i, j, old_ai, old_aj, g.unsafe_load(i), g.unsafe_load(j))
    alpha.unsafe_store(i, a[0])
    alpha.unsafe_store(j, a[1])
    return (_sub(a[0], old_ai), _sub(a[1], old_aj))


@always_inline
def ocsvm_g_step(q: FP, g: FP, n: Int, i: Int, j: Int, dai: Float32, daj: Float32, k: Int):
    """Gradient entry k after the step: Q[i, k] dai, then Q[j, k] daj."""
    var gk = g.unsafe_load(k)
    gk = ftz(identical_mul_add(ftz(q.unsafe_load(i * n + k)), dai, gk))
    gk = ftz(identical_mul_add(ftz(q.unsafe_load(j * n + k)), daj, gk))
    g.unsafe_store(k, gk)


@always_inline
def ocsvm_rho_part(g: FP, alpha: FP, cv: FP, n: Int, b: Int) -> Tuple[Float32, Float32, Float32, Int]:
    """calculate_rho over block b (samples [b * XN_FOLD_BLOCK, ...), ascending):
    (the free gradients summed from zero, lb, ub, the free count)."""
    var neg_inf = bitcast[DType.float32](UInt32(0xFF800000))
    var pos_inf = bitcast[DType.float32](UInt32(0x7F800000))
    var ub = pos_inf
    var lb = neg_inf
    var nr_free = 0
    var sum_free = Float32(0)
    var lo = b * XN_FOLD_BLOCK
    var hi = min(lo + XN_FOLD_BLOCK, n)
    for i in range(lo, hi):
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
    return (sum_free, lb, ub, nr_free)


@always_inline
def ocsvm_rho_from(sum_free: Float32, lb: Float32, ub: Float32, nr_free: Int) -> Float32:
    if nr_free > 0:
        return ftz(identical_div(sum_free, Float32(nr_free)))
    return ftz(identical_mul(_add(ub, lb), Float32(0.5)))


@always_inline
def ocsvm_rho(g: FP, alpha: FP, cv: FP, n: Int) -> Float32:
    """libsvm's calculate_rho, y = +1 throughout, as a blocked fold: block
    partials (ocsvm_rho_part), then the partials ascending (the free sum
    from zero; lb / ub by the scan's strict `>` / `<`, which a fold of the
    block results in order reproduces exactly). The GPU runs the same
    partials as items (ocsvm_rho_part_item, ocsvm_rho_fin_item)."""
    var neg_inf = bitcast[DType.float32](UInt32(0xFF800000))
    var pos_inf = bitcast[DType.float32](UInt32(0x7F800000))
    var ub = pos_inf
    var lb = neg_inf
    var nr_free = 0
    var sum_free = Float32(0)
    for b in range(xn_fold_blocks(n)):
        var p = ocsvm_rho_part(g, alpha, cv, n, b)
        sum_free = _add(sum_free, p[0])
        if p[1] > lb:
            lb = p[1]
        if p[2] < ub:
            ub = p[2]
        nr_free += p[3]
    return ocsvm_rho_from(sum_free, lb, ub, nr_free)


def ocsvm_rho_part_item(t: Int, g: FP, alpha: FP, cv: FP, pf: FP, pc: IP, n: Int):
    """Block t's calculate_rho partial: pf[3t..3t+2] = (sum, lb, ub), pc[t] = count."""
    var p = ocsvm_rho_part(g, alpha, cv, n, t)
    pf.unsafe_store(3 * t, p[0])
    pf.unsafe_store(3 * t + 1, p[1])
    pf.unsafe_store(3 * t + 2, p[2])
    pc.unsafe_store(t, Int32(p[3]))


def ocsvm_rho_fin_item(t: Int, pf: FP, pc: IP, info: FP, n: Int):
    """ONE item: ocsvm_rho's fold of the block partials, ascending."""
    var neg_inf = bitcast[DType.float32](UInt32(0xFF800000))
    var pos_inf = bitcast[DType.float32](UInt32(0x7F800000))
    var ub = pos_inf
    var lb = neg_inf
    var nr_free = 0
    var sum_free = Float32(0)
    for b in range(xn_fold_blocks(n)):
        sum_free = _add(sum_free, pf.unsafe_load(3 * b))
        var plb = pf.unsafe_load(3 * b + 1)
        var pub = pf.unsafe_load(3 * b + 2)
        if plb > lb:
            lb = plb
        if pub < ub:
            ub = pub
        nr_free += Int(pc.unsafe_load(b))
    info.unsafe_store(0, ocsvm_rho_from(sum_free, lb, ub, nr_free))


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
    same helpers with every scan and update over the grid
    (`x_neighbors/ocsvm_dev.mojo`)."""
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
comptime XN_FOLD_BLOCK = 2048
"""The fixed block of the lane's blocked folds (variance, the absolute-
difference sum, SVGP's y^T y): elements [t * XN_FOLD_BLOCK, (t + 1) *
XN_FOLD_BLOCK) fold from zero in ascending order into partial t (one item per
block, the device's full width), then the partials fold from zero in
ascending order (one item). The host runs the same items, so the order is
the device's by construction."""


@always_inline
def xn_fold_blocks(count: Int) -> Int:
    """The number of XN_FOLD_BLOCK blocks (partials) over `count` elements."""
    return (count + XN_FOLD_BLOCK - 1) // XN_FOLD_BLOCK if count > 0 else 0


@always_inline
def _fold_parts(part: FP, nb: Int) -> Float32:
    """The partials folded from zero, ascending."""
    var acc = Float32(0)
    for i in range(nb):
        acc = _add(acc, part.unsafe_load(i))
    return acc


#: lane apple-fast-purity (2026-10-03): the absolute-difference sum's
#: partials fold in a fixed tree, so the device folds them with one block of
#: XN_TREE threads instead of one thread (`-D MOJOLEARN_PURITY_4_OFF`: the
#: ascending chain). Slot s folds partials s, s + XN_TREE, ... from zero,
#: ascending; then halving: slot s += slot s + h for h = XN_TREE / 2 .. 1.
#: The host column runs the same steps, so every column has the same word.
comptime XN_TREE = 256
comptime XN_TREE_ON = not is_defined["MOJOLEARN_PURITY_4_OFF"]()


@always_inline
def xn_tree_slot(part: FP, s: Int, nb: Int) -> Float32:
    """Slot s of the tree fold: partials s, s + XN_TREE, ... ascending from zero."""
    var acc = Float32(0)
    var i = s
    while i < nb:
        acc = _add(acc, part.unsafe_load(i))
        i += XN_TREE
    return acc


@always_inline
def xn_tree_fold(part: FP, nb: Int) -> Float32:
    """The tree fold on one thread (the host column; the device's
    `xn_tree_fold_kernel` runs the same steps across a block)."""
    var sl = InlineArray[Float32, XN_TREE](fill=Float32(0))
    for s in range(XN_TREE):
        sl[s] = xn_tree_slot(part, s, nb)
    var h = XN_TREE // 2
    while h > 0:
        for s in range(h):
            sl[s] = _add(sl[s], sl[s + h])
        h //= 2
    return sl[0]


def variance_part_item(t: Int, x: FP, res: FP, part: FP, count: Int):
    """numpy's `X.var()` over every element (population variance, ddof = 0),
    stage 1 of 4: block t's ascending sum from zero into part[t]."""
    var lo = t * XN_FOLD_BLOCK
    var hi = min(lo + XN_FOLD_BLOCK, count)
    var acc = Float32(0)
    for i in range(lo, hi):
        acc = _add(acc, x.unsafe_load(i))
    part.unsafe_store(t, acc)


def variance_mean_item(t: Int, x: FP, res: FP, part: FP, count: Int):
    """Stage 2 of 4, ONE item: the partials folded ascending, one division;
    the mean is parked in res[0] for stage 3."""
    var acc = _fold_parts(part, xn_fold_blocks(count))
    res.unsafe_store(0, ftz(identical_div(acc, Float32(count))))


def variance_ss_part_item(t: Int, x: FP, res: FP, part: FP, count: Int):
    """Stage 3 of 4: block t's ascending fold of squared deviations from the
    mean (res[0]) from zero into part[t]."""
    var mean = res.unsafe_load(0)
    var lo = t * XN_FOLD_BLOCK
    var hi = min(lo + XN_FOLD_BLOCK, count)
    var ss = Float32(0)
    for i in range(lo, hi):
        var df = _sub(x.unsafe_load(i), mean)
        ss = ftz(identical_mul_add(df, df, ss))
    part.unsafe_store(t, ss)


def variance_fin_item(t: Int, x: FP, res: FP, part: FP, count: Int):
    """Stage 4 of 4, ONE item: the partials folded ascending, one division."""
    var ss = _fold_parts(part, xn_fold_blocks(count))
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


# lane/neural-pass95 (2026-10-01): NearestCentroid's fit statistics for
# feature f in one item, so the fit uploads X once and runs on the device:
# the class means (group_mean_item's chain for every class, rows
# ascending, one division by the class count), the within-class std
# (nc_std_item's chain) and the dataset centroid nc_shrink_item summed
# again for every class (its chain, once). The same statements, the same
# words. On the GPU column group_mean and nc_std ran on the host
# (HOST_RUN), each a walk over all of X at a stride of d words: 5.2 s of
# a 5.7 s istella fit on the M4.
def nc_stats_item(t: Int, x: FP, lab: IP, nk: FP, cent: FP, std: FP, dsc: FP, n: Int, d: Int, n_classes: Int):
    # on a device each pass loads NC_U rows ahead of its chain (a thread then
    # waits on memory once a block, not once a row); the adds keep their order
    comptime U = 32 if is_gpu() else 1
    var f = t
    for g in range(n_classes):
        var acc = Float32(0)
        var cnt = 0
        var i0 = 0
        while i0 + U <= n:
            var xs = SIMD[DType.float32, U]()
            var ls = SIMD[DType.int32, U]()
            comptime for u in range(U):
                xs[u] = x.unsafe_load((i0 + u) * d + f)
                ls[u] = lab.unsafe_load(i0 + u)
            comptime for u in range(U):
                if Int(ls[u]) == g:
                    acc = _add(acc, xs[u])
                    cnt += 1
            i0 += U
        for i in range(i0, n):
            if Int(lab.unsafe_load(i)) == g:
                acc = _add(acc, x.unsafe_load(i * d + f))
                cnt += 1
        if cnt == 0:
            cent.unsafe_store(g * d + f, Float32(0))
        else:
            cent.unsafe_store(g * d + f, ftz(identical_div(acc, Float32(cnt))))
    var ss = Float32(0)
    var all_ = Float32(0)
    var i0 = 0
    while i0 + U <= n:
        var xs = SIMD[DType.float32, U]()
        var cs = SIMD[DType.float32, U]()
        comptime for u in range(U):
            xs[u] = x.unsafe_load((i0 + u) * d + f)
            cs[u] = cent.unsafe_load(Int(lab.unsafe_load(i0 + u)) * d + f)
        comptime for u in range(U):
            var df = _sub(xs[u], cs[u])
            ss = ftz(identical_mul_add(df, df, ss))
            all_ = _add(all_, xs[u])
        i0 += U
    for i in range(i0, n):
        var xv = x.unsafe_load(i * d + f)
        var df = _sub(xv, cent.unsafe_load(Int(lab.unsafe_load(i)) * d + f))
        ss = ftz(identical_mul_add(df, df, ss))
        all_ = _add(all_, xv)
    if n - n_classes <= 0:
        std.unsafe_store(f, Float32(0))
    else:
        std.unsafe_store(f, ftz(identical_sqrt(ftz(identical_div(ss, Float32(n - n_classes))))))
    dsc.unsafe_store(f, ftz(identical_div(all_, Float32(n))))


def nc_shrink_d_item(
    t: Int, dsc_in: FP, cent: FP, nk: FP, std: FP, res: FP, devs: FP,
    n: Int, d: Int, n_classes: Int, do_shrink: Int, med: Float32, shrink: Float32,
):
    """nc_shrink_item with the dataset centroid given (nc_stats_item's)."""
    var k = t // d
    var f = t - k * d
    var c = cent.unsafe_load(t)
    var dsc = dsc_in.unsafe_load(f)
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
@always_inline
def _absdiff_block(a: FP, b: FP, t: Int, count: Int) -> Float32:
    """Block t of sum |a - b|, ascending from zero."""
    var lo = t * XN_FOLD_BLOCK
    var hi = min(lo + XN_FOLD_BLOCK, count)
    var acc = Float32(0)
    for i in range(lo, hi):
        acc = _add(acc, abs(_sub(a.unsafe_load(i), b.unsafe_load(i))))
    return acc


def absdiff_part_item(t: Int, a: FP, b: FP, res: FP, part: FP, count: Int):
    """sum |a - b| over every element, stage 1 of 2: block t into part[t]."""
    part.unsafe_store(t, _absdiff_block(a, b, t, count))


def absdiff_fin_item(t: Int, a: FP, b: FP, res: FP, part: FP, count: Int):
    """Stage 2 of 2, ONE item: the partials folded (the XN_TREE order;
    ascending under -D MOJOLEARN_PURITY_4_OFF)."""
    comptime if XN_TREE_ON:
        res.unsafe_store(0, xn_tree_fold(part, xn_fold_blocks(count)))
    else:
        res.unsafe_store(0, _fold_parts(part, xn_fold_blocks(count)))


def absdiff_sum_item(t: Int, a: FP, b: FP, res: FP, count: Int):
    """sum |a - b| over every element, ONE host item in the blocked order of
    `absdiff_part_item` + `absdiff_fin_item` (the resident label-propagation
    loops' convergence test, the same bits as the `absdiff_sum` op)."""
    comptime if XN_TREE_ON:
        var nb = xn_fold_blocks(count)
        var parts = List[Float32](length=max(nb, 1), fill=Float32(0))
        var pp = FP(unsafe_from_address=Int(parts.unsafe_ptr()))
        for blk in range(nb):
            pp.unsafe_store(blk, _absdiff_block(a, b, blk, count))
        res.unsafe_store(0, xn_tree_fold(pp, nb))
        _ = parts^
    else:
        var acc = Float32(0)
        for blk in range(xn_fold_blocks(count)):
            acc = _add(acc, _absdiff_block(a, b, blk, count))
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
def graph_symmetry_row_item(t: Int, a: FP, flags: IP, rf: IP, n: Int):
    """The adjacency checks the graph estimators' wrappers made in Python
    over `A.tolist()` (lane neural-pass14; at the board's 20,000 nodes that
    was a 400-million-cell Python scan), stage 1 of 2, one item per row u =
    t: rf[2u] = 1 when some cell `a[u, v]` with `u < v` is not equal to
    `a[v, u]` (a NaN is not equal to itself, as in Python), rf[2u + 1] = 1
    when some cell of row u is nonzero. Integers out, no arithmetic."""
    var asym = Int32(0)
    var nonzero = Int32(0)
    for v in range(n):
        var x = a.unsafe_load(t * n + v)
        if x != Float32(0):
            nonzero = Int32(1)
        if v > t and x != a.unsafe_load(v * n + t):
            asym = Int32(1)
    rf.unsafe_store(2 * t, asym)
    rf.unsafe_store(2 * t + 1, nonzero)


def graph_symmetry_fin_item(t: Int, a: FP, flags: IP, rf: IP, n: Int):
    """Stage 2 of 2, ONE item: the rows' flags OR-ed, rows ascending
    (integers: the same flags as the one-item scan)."""
    var asym = Int32(0)
    var nonzero = Int32(0)
    for u in range(n):
        if rf.unsafe_load(2 * u) != Int32(0):
            asym = Int32(1)
        if rf.unsafe_load(2 * u + 1) != Int32(0):
            nonzero = Int32(1)
    flags.unsafe_store(0, asym)
    flags.unsafe_store(1, nonzero)


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


def row_argmax_item(t: Int, a: FP, res: IP, n: Int, m: Int):
    """Row t's first maximum (lane apple-fast-purity): `_argmax` in
    _expansion_neighbors.py, strictly greater replaces, so ties and a
    leading NaN keep the earlier column. Exact: no arithmetic."""
    var best = a.unsafe_load(t * m)
    var at = 0
    for j in range(1, m):
        var v = a.unsafe_load(t * m + j)
        if v > best:
            best = v
            at = j
    res.unsafe_store(t, Int32(at))
