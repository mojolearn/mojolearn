"""The host oracles for KDE: a float32 serial replay and a float64 reference.

NOT A PORT. cuML ships one backend and checks `score_samples` against
scikit-learn to a tolerance (`tests/test_kernel_density.py`); it has no
bit-level oracle because it needs none. We ship three backends from one
source, so the device arm is gated BIT FOR BIT under IDENTICAL against the
float32 replay below, and the replay is gated against the float64 reference
to a tolerance. Both are written FIRST and gated FIRST (COMMON_BRIEF 6).

TWO ORACLES, TWO JOBS
---------------------
`oracle_score_samples`   float32, SERIAL, ASCENDING, through the same
                         helpers the device uses (`identical_mul_add`,
                         `ftz`, `identical_exp/log/sqrt/cos`), every formula
                         spelled here a SECOND time rather than imported
                         from `kde/ported/` -- so the gate compares two
                         spellings of one arithmetic, not a function against
                         itself. Also returns every stage (dists, logk,
                         rowmax, logsumexp, scores) so a mismatch has an
                         address.
`reference_score_samples_f64`
                         float64, scikit-learn's SEMANTICS (`-inf` outside
                         a compact kernel's support, `_binary_tree.pxi:
                         377-414` and `_log_kernel_norm` at `:448-475`),
                         `std.math` on the host. Its job is tolerance sanity
                         (DEVIATION 600's cost) and the closed-form norms.

THE ONE PLACE THE FLOAT32 ORACLE MIRRORS A DEVICE SHAPE
-------------------------------------------------------
`sqeuclidean` goes through `core/row_norms.mojo::row_norm_kernel`, whose
fold is `pinned_block_sum[NORM_TPB]` -- a `NORM_TPB`-lane strided partial
per lane then a halving tree, NOT a serial sum. The oracle replays THAT
shape for the two norms (`_host_row_norm_halving`), exactly as
`glm/mojo_only/ridge_check.mojo::_host_halving_xty` does for `A^T b`,
because the norms are the k-NN lane's and this lane calls rather than
re-spells them. Everything else in this file is serial ascending. Under
FAST that kernel is `block.sum` (the library's shape) and the sqeuclidean
comparison is a REPORT.
"""

from std.math import cos, exp, lgamma, log, pi, sqrt
from std.memory import bitcast

from core.row_norms import NORM_TPB
from kde.ported.distance.distance_ops import (
    DIST_L1,
    DIST_L2_EXPANDED,
    DIST_L2_SQRT_UNEXPANDED,
    DIST_LINF,
)
from kde.ported.neighbors.kernel_density import (
    KDE_KERNEL_COSINE,
    KDE_KERNEL_EPANECHNIKOV,
    KDE_KERNEL_EXPONENTIAL,
    KDE_KERNEL_GAUSSIAN,
    KDE_KERNEL_LINEAR,
    KDE_KERNEL_TOPHAT,
    log_kernel_norm,
)
from mojo_only.numerics import (
    ftz,
    identical_cos,
    identical_exp,
    identical_log,
    identical_mul_add,
    identical_sqrt,
)


comptime ORACLE_FLOAT32_MIN_BITS: UInt32 = 0xFF7FFFFF
comptime ORACLE_LOG_FLOOR = Float32(1e-30)


@fieldwise_init
struct KdeOracleStages(Movable):
    """Every stage of one `score_samples`, float32, host-computed."""

    var dists: List[Float32]
    var logk: List[Float32]
    var rowmax: List[Float32]
    var logsumexp: List[Float32]
    var scores: List[Float32]
    var log_sw: Float32
    var norm: Float32


def _host_row_norm_halving(a: List[Float32], row: Int, d: Int) -> Float32:
    """`row_norm_kernel` at `take_sqrt = 0`, replayed: NORM_TPB strided lane
    partials (`acc = ftz(fma(v, v, acc))`), then the halving tree of
    `pinned_block_sum`, then `ftz` of the total."""
    var red = List[Float32]()
    for t in range(NORM_TPB):
        var acc = Float32(0.0)
        var col = t
        while col < d:
            var v = ftz(a[row * d + col])
            acc = ftz(identical_mul_add(v, v, acc))
            col += NORM_TPB
        red.append(acc)
    var step = NORM_TPB // 2
    while step > 0:
        for t in range(step):
            red[t] = red[t] + red[t + step]
        step //= 2
    return ftz(red[0])


def oracle_distance(
    query: List[Float32],
    train: List[Float32],
    q: Int,
    j: Int,
    d: Int,
    metric: Int,
    q_norm: Float32,
    t_norm: Float32,
) -> Float32:
    """One cell of the distance matrix, the feature axis ascending in one
    serial fold. The expanded arm takes the two norms from the caller (they
    are per-row, computed once)."""
    var acc = Float32(0.0)
    if metric == DIST_L2_SQRT_UNEXPANDED:
        for f in range(d):
            var diff = ftz(ftz(query[q * d + f]) - ftz(train[j * d + f]))
            acc = ftz(identical_mul_add(diff, diff, acc))
        return ftz(identical_sqrt(acc))
    if metric == DIST_L1:
        for f in range(d):
            acc = ftz(acc + abs(ftz(ftz(query[q * d + f]) - ftz(train[j * d + f]))))
        return acc
    if metric == DIST_LINF:
        for f in range(d):
            var diff = abs(ftz(ftz(query[q * d + f]) - ftz(train[j * d + f])))
            if diff > acc:
                acc = diff
        return acc
    # DIST_L2_EXPANDED: the pinned tile's arithmetic, `is_sqrt = 0`.
    for f in range(d):
        acc = ftz(
            identical_mul_add(ftz(query[q * d + f]), ftz(train[j * d + f]), acc)
        )
    var dist = ftz(identical_mul_add(Float32(-2.0), acc, ftz(q_norm + t_norm)))
    if dist <= Float32(0.0):
        dist = Float32(0.0)
    return dist


def oracle_log_kernel(x: Float32, h: Float32, kernel: Int) -> Float32:
    """The six cupy log-kernels, spelled a second time (see the header).
    `fmin` is `np.finfo(float32).min`; `0.0 * fmin` is `-0.0`."""
    var fmin = bitcast[DType.float32](ORACLE_FLOAT32_MIN_BITS)
    if kernel == KDE_KERNEL_GAUSSIAN:
        var num = -ftz(x * x)
        var den = ftz(ftz(Float32(2.0) * h) * h)
        return ftz(num / den)
    if kernel == KDE_KERNEL_EXPONENTIAL:
        return ftz((-x) / h)
    if kernel == KDE_KERNEL_TOPHAT:
        if x >= h:
            return fmin
        return Float32(0.0) * fmin
    # the three log-of-clamped kernels
    if x >= h:
        return fmin
    var z: Float32
    if kernel == KDE_KERNEL_EPANECHNIKOV:
        var hsq = ftz(h * h)
        z = ftz(Float32(1.0) - ftz(ftz(x * x) / hsq))
    elif kernel == KDE_KERNEL_LINEAR:
        z = ftz(Float32(1.0) - ftz(x / h))
    else:
        var arg = ftz(ftz(Float32(1.5707963267948966) * x) / h)
        z = ftz(identical_cos(arg))
    if z < ORACLE_LOG_FLOOR:
        z = ORACLE_LOG_FLOOR
    return ftz(identical_log(z))


def oracle_logsumexp_row(
    logk: List[Float32], base: Int, n_train: Int
) -> Tuple[Float32, Float32]:
    """Their numba kernel, serial ascending: `(rowmax, log(sum) + max)`."""
    var max_exp = logk[base]
    for j in range(1, n_train):
        if logk[base + j] > max_exp:
            max_exp = logk[base + j]
    var s = Float32(0.0)
    for j in range(n_train):
        s = ftz(s + ftz(identical_exp(ftz(logk[base + j] - max_exp))))
    return (max_exp, ftz(identical_log(s) + max_exp))


def oracle_naive_log_sum_row(
    logk: List[Float32], base: Int, n_train: Int
) -> Float32:
    """The UNSHIFTED `log(sum_j exp(v_j))`, serial ascending. Not what
    anyone ships; it exists so `check_kde_logsumexp_beats_naive` can show a
    row where this underflows to `log(0) = -inf` and the shifted form does
    not."""
    var s = Float32(0.0)
    for j in range(n_train):
        s = ftz(s + ftz(identical_exp(logk[base + j])))
    return ftz(identical_log(s))


def oracle_score_samples(
    train: List[Float32],
    query: List[Float32],
    weights: List[Float32],
    has_weights: Bool,
    n_train: Int,
    n_query: Int,
    d: Int,
    h: Float32,
    kernel: Int,
    metric: Int,
) raises -> KdeOracleStages:
    """`score_samples` on the host, float32, every stage serial ascending."""
    var cells = n_query * n_train
    var dists = List[Float32](capacity=cells)
    var logk = List[Float32](capacity=cells)
    var rowmax = List[Float32](capacity=n_query)
    var lse = List[Float32](capacity=n_query)
    var scores = List[Float32](capacity=n_query)

    var q_norms = List[Float32]()
    var t_norms = List[Float32]()
    if metric == DIST_L2_EXPANDED:
        for q in range(n_query):
            q_norms.append(_host_row_norm_halving(query, q, d))
        for j in range(n_train):
            t_norms.append(_host_row_norm_halving(train, j, d))

    var logw = List[Float32]()
    if has_weights:
        for j in range(n_train):
            logw.append(ftz(identical_log(ftz(weights[j]))))

    for q in range(n_query):
        for j in range(n_train):
            var qn = Float32(0.0)
            var tn = Float32(0.0)
            if metric == DIST_L2_EXPANDED:
                qn = q_norms[q]
                tn = t_norms[j]
            var dist = oracle_distance(query, train, q, j, d, metric, qn, tn)
            dists.append(dist)
            var v = oracle_log_kernel(ftz(dist), h, kernel)
            if has_weights:
                v = ftz(v + logw[j])
            logk.append(v)

    var sum_w = Float32(0.0)
    if has_weights:
        for j in range(n_train):
            sum_w = ftz(sum_w + weights[j])
    else:
        sum_w = Float32(n_train)
    var log_sw = ftz(identical_log(sum_w))
    var norm = log_kernel_norm(kernel, h, d)

    for q in range(n_query):
        var mm = oracle_logsumexp_row(logk, q * n_train, n_train)
        rowmax.append(mm[0])
        lse.append(mm[1])
        var a = ftz(mm[1] - log_sw)
        scores.append(ftz(a - norm))

    return KdeOracleStages(dists^, logk^, rowmax^, lse^, scores^, log_sw, norm)


# ---------------------------------------------------------------------------
# The float64 reference, scikit-learn semantics.
# ---------------------------------------------------------------------------


def _neg_inf64() -> Float64:
    return bitcast[DType.float64](UInt64(0xFFF0000000000000))


def reference_log_kernel_f64(x: Float64, h: Float64, kernel: Int) -> Float64:
    """`_binary_tree.pxi:377-414`: `-inf` outside the support, no floor."""
    if kernel == KDE_KERNEL_GAUSSIAN:
        return -0.5 * (x * x) / (h * h)
    if kernel == KDE_KERNEL_EXPONENTIAL:
        return -x / h
    if x >= h:
        return _neg_inf64()
    if kernel == KDE_KERNEL_TOPHAT:
        return 0.0
    if kernel == KDE_KERNEL_EPANECHNIKOV:
        return log(1.0 - (x * x) / (h * h))
    if kernel == KDE_KERNEL_LINEAR:
        return log(1.0 - x / h)
    return log(cos(0.5 * Float64(pi) * x / h))


def _log_vn64(n: Int) -> Float64:
    return 0.5 * Float64(n) * log(Float64(pi)) - lgamma(0.5 * Float64(n) + 1.0)


def _log_sn64(n: Int) -> Float64:
    return log(2.0 * Float64(pi)) + _log_vn64(n - 1)


def reference_log_kernel_norm_f64(kernel: Int, h: Float64, d: Int) raises -> Float64:
    """`_log_kernel_norm(h, d, kernel)` (`_binary_tree.pxi:448-475`), the
    value scikit-learn ADDS: `-factor - d * log(h)`. cuML subtracts the
    negative of this; the two agree term for term -- EXCEPT the cosine
    kernel at even d, where both are wrong and this reference is not
    (DEVIATION 602)."""
    var dd = Float64(d)
    var factor: Float64
    if kernel == KDE_KERNEL_GAUSSIAN:
        factor = 0.5 * dd * log(2.0 * Float64(pi))
    elif kernel == KDE_KERNEL_TOPHAT:
        factor = _log_vn64(d)
    elif kernel == KDE_KERNEL_EPANECHNIKOV:
        factor = _log_vn64(d) + log(2.0 / (dd + 2.0))
    elif kernel == KDE_KERNEL_EXPONENTIAL:
        factor = _log_sn64(d - 1) + lgamma(dd)
    elif kernel == KDE_KERNEL_LINEAR:
        factor = _log_vn64(d) - log(dd + 1.0)
    elif kernel == KDE_KERNEL_COSINE:
        # DEVIATION 602: the TRUE radial integral `I_{d-1}` by its
        # recurrence, not scikit-learn's loop (`:465-470`), which drops
        # `I_1`'s second term for even d (NaN at d = 4). The reference is
        # the mathematics, not the upstream defect.
        var two_over_pi = 2.0 / Float64(pi)
        var c = two_over_pi * two_over_pi
        var n = d - 1
        var acc: Float64
        var m: Int
        if n % 2 == 0:
            acc = two_over_pi
            m = 2
        else:
            acc = two_over_pi - c
            m = 3
        while m <= n:
            acc = two_over_pi - Float64(m * (m - 1)) * c * acc
            m += 2
        factor = log(acc) + _log_sn64(d - 1)
    else:
        raise Error("Kernel code not recognized")
    return -factor - dd * log(h)


def reference_distance_f64(
    query: List[Float32], train: List[Float32], q: Int, j: Int, d: Int, metric: Int
) -> Float64:
    """The metric as mathematics in float64: `sqeuclidean` is the plain
    sum of squared differences (no expansion), the others as named."""
    var acc = 0.0
    for f in range(d):
        var diff = Float64(query[q * d + f]) - Float64(train[j * d + f])
        if metric == DIST_L1:
            acc += abs(diff)
        elif metric == DIST_LINF:
            if abs(diff) > acc:
                acc = abs(diff)
        else:
            acc += diff * diff
    if metric == DIST_L2_SQRT_UNEXPANDED:
        return sqrt(acc)
    return acc


def reference_score_samples_f64(
    train: List[Float32],
    query: List[Float32],
    weights: List[Float32],
    has_weights: Bool,
    n_train: Int,
    n_query: Int,
    d: Int,
    h: Float64,
    kernel: Int,
    metric: Int,
) raises -> List[Float64]:
    """scikit-learn's `score_samples` semantics in float64: `logsumexp_j
    (log K(dist_qj) + log w_j) - log(sum w) + log_kernel_norm`. A query with
    no training point in its support is `-inf`."""
    var out = List[Float64]()
    var sum_w = 0.0
    if has_weights:
        for j in range(n_train):
            sum_w += Float64(weights[j])
    else:
        sum_w = Float64(n_train)
    var knorm = reference_log_kernel_norm_f64(kernel, h, d)
    var neg_inf = _neg_inf64()
    for q in range(n_query):
        var vals = List[Float64]()
        var mx = neg_inf
        for j in range(n_train):
            var v = reference_log_kernel_f64(
                reference_distance_f64(query, train, q, j, d, metric), h, kernel
            )
            if has_weights:
                v += log(Float64(weights[j]))
            vals.append(v)
            if v > mx:
                mx = v
        if mx == neg_inf:
            out.append(neg_inf)
            continue
        var s = 0.0
        for j in range(n_train):
            if vals[j] != neg_inf:
                s += exp(vals[j] - mx)
        out.append(log(s) + mx - log(sum_w) + knorm)
    return out^
