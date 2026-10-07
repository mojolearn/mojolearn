# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into a CPU host binding (python/mojolearn/host_surface.py names which); product, not only a check.
"""The host oracles for KDE: a float32 serial replay and a float64 reference.

NO REFERENCE FILE. cuML ships one backend and checks `score_samples` against
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
                         from `kde/impl/` -- so the gate compares two
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
`glm/checks/ridge_check.mojo::_host_halving_xty` does for `A^T b`,
because the norms are the k-NN lane's and this lane calls rather than
re-spells them. Everything else in this file is serial ascending. Under
FAST that kernel is `block.sum` (the library's shape) and the sqeuclidean
comparison is a REPORT.
"""

from experiments.classical_identical_ideas.stats_controls import C52_PAIR
from experiments.classical_identical_ideas.graph_controls import KDE_DIRECT_DISTANCE
from core.classical_distance import direct_squared_distance, direct_distance_step
from kde.pair_lse import pair_row
from std.math import cos, exp, lgamma, log, pi, sqrt
from std.memory import bitcast
from std.sys.compile import is_defined

from core.host_parallel import host_parallelize

from core.host_predict_threads import (
    HostF32Ptr,
    host_list_ptr,
    host_predict_chunk,
)
from core.row_norms import NORM_TPB
from kde.impl.distance.distance_ops import (
    DIST_COSINE_EXPANDED,
    DIST_L1,
    DIST_L2_EXPANDED,
    DIST_L2_SQRT_UNEXPANDED,
    DIST_LINF,
    DIST_LP_UNEXPANDED,
)
from kde.impl.neighbors.kernel_density import (
    KDE_KERNEL_COSINE,
    KDE_KERNEL_EPANECHNIKOV,
    KDE_KERNEL_EXPONENTIAL,
    KDE_KERNEL_GAUSSIAN,
    KDE_KERNEL_LINEAR,
    KDE_KERNEL_TOPHAT,
    kde_chunk_lse_metric_applies,
    kde_chunk_rows_for,
    log_kernel_norm,
)
from checks.numerics import (
    ftz,
    identical_cos,
    identical_div,
    identical_exp,
    identical_log,
    identical_mul,
    identical_mul_add,
    identical_mul_add_simd,
    identical_pow,
    identical_sqrt,
)
from core.host_simd_identical import expf_v, ftz_v, logf_v, powf_v


#: THE NEGATIVE CONTROL OF THE CPU IDENTITY GATE (the CPU training lane,
#: 2026-09-13, brief section 3.4). `-D MOJOLEARN_HOST_SABOTAGE=1` makes
#: `oracle_logsumexp_row` sum the shifted exponentials DESCENDING instead of
#: ascending, from one extra unit instead of from zero, so the estimators
#: host binding built with it computes a different fold and every kde lane
#: must read DIVERGENT against the GPU columns. The extra unit is there for
#: the tophat kernel (lane kde-tophat-sqeuclidean, 2026-09-14): its shifted
#: exponentials are exactly 0 or 1, so a descending sum of them is the same
#: integer and the descending walk alone left all nine fixtures IDENTICAL
#: under sabotage (measured on the M4's CPU column, one core). Passed by the host build scripts only; a host binding that
#: carries it says so through `<prefix>_sabotage()` and is refused outside
#: the gate (`python/mojolearn/_backend.py::load_host_module`).
comptime KDE_ORACLE_HOST_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()

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
    """`_host_row_norm_halving_ptr` over a List, the checks' door."""
    return _host_row_norm_halving_ptr(host_list_ptr(a), row, d)


def _host_row_norm_halving_ptr(a: HostF32Ptr, row: Int, d: Int) -> Float32:
    """`row_norm_kernel` at `take_sqrt = 0`, replayed: NORM_TPB strided lane
    partials (`acc = ftz(fma(v, v, acc))`), then the halving tree of
    `pinned_block_sum`, then `ftz` of the total."""
    var red = List[Float32]()
    for t in range(NORM_TPB):
        var acc = Float32(0.0)
        var col = t
        while col < d:
            var v = ftz(a.unsafe_load(row * d + col))
            acc = ftz(identical_mul_add(v, v, acc))
            col += NORM_TPB
        red.append(acc)
    var step = NORM_TPB // 2
    while step > 0:
        for t in range(step):
            red[t] = red[t] + red[t + step]
        step //= 2
    return ftz(red[0])


def _host_row_norm_halving_sqrt(a: List[Float32], row: Int, d: Int) -> Float32:
    """`cosine_row_norm_kernel` replayed: the SAME halving fold as above,
    then the zero clamp and `identical_sqrt`. The TRUE L2 norm, which is
    what `rowNorm<L2Norm, true>(..., raft::sqrt_op{})` gives cosine
    (`distance.cuh:215-216`) and NOT the squared norm the expanded L2 arm
    takes."""
    return _host_row_norm_halving_sqrt_ptr(host_list_ptr(a), row, d)


def _host_row_norm_halving_sqrt_ptr(a: HostF32Ptr, row: Int, d: Int) -> Float32:
    """`_host_row_norm_halving_sqrt` over the caller's memory."""
    var total = _host_row_norm_halving_ptr(a, row, d)
    if total <= Float32(0.0):
        total = Float32(0.0)
    return ftz(identical_sqrt(total))


def oracle_distance(
    query: List[Float32],
    train: List[Float32],
    q: Int,
    j: Int,
    d: Int,
    metric: Int,
    q_norm: Float32,
    t_norm: Float32,
    metric_arg: Float32 = Float32(2.0),
) -> Float32:
    """`oracle_distance_ptr` over two Lists, the checks' door."""
    return oracle_distance_ptr(
        host_list_ptr(query), host_list_ptr(train), q, j, d, metric, q_norm,
        t_norm, metric_arg,
    )


@always_inline
def oracle_distance_ptr(
    query: HostF32Ptr,
    train: HostF32Ptr,
    q: Int,
    j: Int,
    d: Int,
    metric: Int,
    q_norm: Float32,
    t_norm: Float32,
    metric_arg: Float32 = Float32(2.0),
) -> Float32:
    """One cell of the distance matrix, the feature axis ascending in one
    serial fold. The two expanded arms (sqeuclidean, cosine) take the two
    norms from the caller (they are per-row, computed once); cosine's are
    the SQRT'd norms, sqeuclidean's are the squared ones.

    A SECOND SPELLING of `metric_distance_kernel`, not an import of it.
    That is this file's whole contract (see its header): if the device
    kernel and this function ever disagree, one of them is wrong and the
    gate says which cell."""
    comptime if KDE_DIRECT_DISTANCE:
        if metric == DIST_L2_EXPANDED or metric == DIST_L2_SQRT_UNEXPANDED:
            var direct = direct_squared_distance(query+q*d, train+j*d, d)
            return ftz(identical_sqrt(direct)) if metric == DIST_L2_SQRT_UNEXPANDED else direct
    var acc = Float32(0.0)
    if metric == DIST_COSINE_EXPANDED:
        # `cosine.cuh:68` core, then `:86` epilog.
        for f in range(d):
            acc = ftz(
                identical_mul_add(
                    ftz(query.unsafe_load(q * d + f)), ftz(train.unsafe_load(j * d + f)), acc
                )
            )
        var denom = ftz(identical_mul(q_norm, t_norm))
        return ftz(Float32(1.0) - ftz(identical_div(acc, denom)))
    if metric == DIST_LP_UNEXPANDED:
        # `lp_unexp.cuh:56-57` core, then `:67` and `:72` epilog.
        for f in range(d):
            var diff = abs(ftz(ftz(query.unsafe_load(q * d + f)) - ftz(train.unsafe_load(j * d + f))))
            acc = ftz(acc + ftz(identical_pow(diff, metric_arg)))
        var one_over_p = ftz(identical_div(Float32(1.0), metric_arg))
        return ftz(identical_pow(acc, one_over_p))
    if metric == DIST_L2_SQRT_UNEXPANDED:
        for f in range(d):
            var diff = ftz(ftz(query.unsafe_load(q * d + f)) - ftz(train.unsafe_load(j * d + f)))
            acc = ftz(identical_mul_add(diff, diff, acc))
        return ftz(identical_sqrt(acc))
    if metric == DIST_L1:
        for f in range(d):
            acc = ftz(acc + abs(ftz(ftz(query.unsafe_load(q * d + f)) - ftz(train.unsafe_load(j * d + f)))))
        return acc
    if metric == DIST_LINF:
        for f in range(d):
            var diff = abs(ftz(ftz(query.unsafe_load(q * d + f)) - ftz(train.unsafe_load(j * d + f))))
            # row 39: the device's strict `>` over abs() candidates, seeded
            # +0.0; a tie is the same bits either way (distance_ops.mojo).
            if diff > acc:
                acc = diff
        return acc
    # DIST_L2_EXPANDED: the pinned tile's arithmetic, `is_sqrt = 0`.
    for f in range(d):
        acc = ftz(
            identical_mul_add(ftz(query.unsafe_load(q * d + f)), ftz(train.unsafe_load(j * d + f)), acc)
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
    """Their numba kernel, serial ascending: `(rowmax, log(sum) + max)`.
    Row 39: strict `>` from `j = 0`, the lower index wins a tie of `-0.0`
    and `+0.0` (the device's rule, `logsumexp_kernel`). DEVIATION 603: a
    row of all `-inf` is `-inf`, not `exp(NaN)`."""
    comptime if C52_PAIR:
        return pair_row(host_list_ptr(logk) + base, n_train, kde_chunk_rows_for(n_train))
    var max_exp = logk[base]
    for j in range(1, n_train):
        if logk[base + j] > max_exp:
            max_exp = logk[base + j]
    if max_exp == bitcast[DType.float32](UInt32(0xFF800000)):
        return (max_exp, max_exp)
    var s = Float32(0.0)
    comptime if KDE_ORACLE_HOST_SABOTAGE:
        # THE SABOTAGE ARM: the same row, summed DESCENDING from one extra
        # unit. Wrong on purpose; see KDE_ORACLE_HOST_SABOTAGE.
        s = Float32(1.0)
        for jj in range(n_train):
            var j = n_train - 1 - jj
            s = ftz(s + ftz(identical_exp(ftz(logk[base + j] - max_exp))))
    else:
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
    metric_arg: Float32 = Float32(2.0),
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
    elif metric == DIST_COSINE_EXPANDED:
        # THE SQRT'D NORM, and getting this line wrong is the sabotage
        # `check_cosine_norm_flag_is_load_bearing` performs on purpose.
        for q in range(n_query):
            q_norms.append(_host_row_norm_halving_sqrt(query, q, d))
        for j in range(n_train):
            t_norms.append(_host_row_norm_halving_sqrt(train, j, d))

    var logw = List[Float32]()
    if has_weights:
        for j in range(n_train):
            logw.append(ftz(identical_log(ftz(weights[j]))))

    for q in range(n_query):
        for j in range(n_train):
            var qn = Float32(0.0)
            var tn = Float32(0.0)
            if metric == DIST_L2_EXPANDED or metric == DIST_COSINE_EXPANDED:
                qn = q_norms[q]
                tn = t_norms[j]
            var dist = oracle_distance(
                query, train, q, j, d, metric, qn, tn, metric_arg
            )
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
        var mm: Tuple[Float32, Float32]
        if kde_chunk_lse_metric_applies(metric):
            # lane/fam-neighbors: the device's chunked fold for the tiled metrics
            mm = _kde_lse_row_chunked(host_list_ptr(logk) + q * n_train, n_train)
        else:
            mm = oracle_logsumexp_row(logk, q * n_train, n_train)
        rowmax.append(mm[0])
        lse.append(mm[1])
        var a = ftz(mm[1] - log_sw)
        scores.append(ftz(a - norm))

    return KdeOracleStages(dists^, logk^, rowmax^, lse^, scores^, log_sw, norm)


#: THE BLOCK ENGINE (lane neighbors-cpu, 2026-09-28): the scoring path of
#: `oracle_score_samples_into` computes KDE_W training columns of KDE_QB
#: query rows at once, one SIMD lane per CELL. Lane `l` of a register is the
#: scalar statement sequence of `oracle_distance_ptr` for cell (q, j0 + l)
#: (the feature axis ascending, each operand flushed; the training operand
#: flushed once when packed), then `oracle_log_kernel` and the weight's
#: log, lane by lane; `ftz_v` / `logf_v` / `expf_v` are measured equal to
#: `ftz` / `portable_logf` / `portable_expf` on all 2^32 words
#: (core/host_simd_identical_check.mojo), `identical_div` is `ftz(ftz(a) /
#: ftz(b))` (portable_divf), and `identical_sqrt`, `identical_pow` and
#: `identical_cos` run per lane through the scalar seam. The row's
#: log-sum-exp keeps `oracle_logsumexp_row` exactly: the strict-`>` max scan
#: ascending, then the shifted exponentials (computed W at a time) summed
#: one by one in the same order, the sabotage arm's descending walk from
#: one unit included.
comptime KDE_W = 8
comptime KDE_QB = 4
comptime KdeV = SIMD[DType.float32, KDE_W]


@always_inline
def _kde_step[M: Int](acc: KdeV, qv: Float32, t: KdeV, metric_arg: Float32) -> KdeV:
    """One feature step of W cells of `oracle_distance_ptr`, the metric
    fixed at compile time (see `_kde_tile`)."""
    comptime if KDE_DIRECT_DISTANCE and (M == DIST_L2_EXPANDED or M == DIST_L2_SQRT_UNEXPANDED):
        return direct_distance_step[KDE_W](acc, KdeV(qv), t)
    comptime if M == DIST_COSINE_EXPANDED or M == DIST_L2_EXPANDED:
        return ftz_v[KDE_W](identical_mul_add_simd[KDE_W](KdeV(qv), t, acc))
    elif M == DIST_L2_SQRT_UNEXPANDED:
        var diff = ftz_v[KDE_W](KdeV(qv) - t)
        return ftz_v[KDE_W](identical_mul_add_simd[KDE_W](diff, diff, acc))
    elif M == DIST_L1:
        return ftz_v[KDE_W](acc + abs(ftz_v[KDE_W](KdeV(qv) - t)))
    elif M == DIST_LINF:
        var diff = abs(ftz_v[KDE_W](KdeV(qv) - t))
        return diff.gt(acc).select(diff, acc)
    else:
        var diff = abs(ftz_v[KDE_W](KdeV(qv) - t))
        return ftz_v[KDE_W](acc + ftz_v[KDE_W](powf_v[KDE_W](diff, metric_arg)))


def _kde_tile_m[M: Int](
    qbp: HostF32Ptr, pb: HostF32Ptr, d: Int, metric_arg: Float32, res: HostF32Ptr,
):
    """KDE_QB query rows (at `qbp`, row r at `r * d`) x KDE_W training
    columns (the packed block `pb`), raw accumulators into `res`."""
    var a0 = KdeV(0.0)
    var a1 = KdeV(0.0)
    var a2 = KdeV(0.0)
    var a3 = KdeV(0.0)
    for f in range(d):
        var tv = pb.unsafe_load[width=KDE_W](f * KDE_W)
        a0 = _kde_step[M](a0, qbp.unsafe_load(f), tv, metric_arg)
        a1 = _kde_step[M](a1, qbp.unsafe_load(d + f), tv, metric_arg)
        a2 = _kde_step[M](a2, qbp.unsafe_load(2 * d + f), tv, metric_arg)
        a3 = _kde_step[M](a3, qbp.unsafe_load(3 * d + f), tv, metric_arg)
    res.unsafe_store[width=KDE_W](0, a0)
    res.unsafe_store[width=KDE_W](KDE_W, a1)
    res.unsafe_store[width=KDE_W](2 * KDE_W, a2)
    res.unsafe_store[width=KDE_W](3 * KDE_W, a3)


def _kde_tile(
    qbp: HostF32Ptr, pb: HostF32Ptr, d: Int, metric: Int, metric_arg: Float32,
    res: HostF32Ptr,
):
    if metric == DIST_COSINE_EXPANDED or metric == DIST_L2_EXPANDED:
        _kde_tile_m[DIST_L2_EXPANDED](qbp, pb, d, metric_arg, res)
    elif metric == DIST_L2_SQRT_UNEXPANDED:
        _kde_tile_m[DIST_L2_SQRT_UNEXPANDED](qbp, pb, d, metric_arg, res)
    elif metric == DIST_L1:
        _kde_tile_m[DIST_L1](qbp, pb, d, metric_arg, res)
    elif metric == DIST_LINF:
        _kde_tile_m[DIST_LINF](qbp, pb, d, metric_arg, res)
    else:
        _kde_tile_m[DIST_LP_UNEXPANDED](qbp, pb, d, metric_arg, res)


@always_inline
def _kde_epilogue(acc: KdeV, qn: Float32, tn: KdeV, metric: Int, metric_arg: Float32) -> KdeV:
    """The epilogues of `oracle_distance_ptr`, W cells of one query row."""
    comptime if KDE_DIRECT_DISTANCE:
        if metric == DIST_L2_EXPANDED:
            return acc
    if metric == DIST_COSINE_EXPANDED:
        var denom = ftz_v[KDE_W](KdeV(qn) * tn)
        var ratio = ftz_v[KDE_W](ftz_v[KDE_W](acc) / ftz_v[KDE_W](denom))
        return ftz_v[KDE_W](KdeV(1.0) - ftz_v[KDE_W](ratio))
    if metric == DIST_LP_UNEXPANDED:
        var one_over_p = ftz(identical_div(Float32(1.0), metric_arg))
        return ftz_v[KDE_W](powf_v[KDE_W](acc, one_over_p))
    if metric == DIST_L2_SQRT_UNEXPANDED:
        var out = acc
        comptime for l in range(KDE_W):
            out[l] = ftz(identical_sqrt(acc[l]))
        return out
    if metric == DIST_L1 or metric == DIST_LINF:
        return acc
    var dist = ftz_v[KDE_W](identical_mul_add_simd[KDE_W](
        KdeV(-2.0), acc, ftz_v[KDE_W](KdeV(qn) + tn)
    ))
    return dist.le(KdeV(0.0)).select(KdeV(0.0), dist)


@always_inline
def _kde_log_kernel_v(x: KdeV, h: Float32, kernel: Int) -> KdeV:
    """`oracle_log_kernel`, lane by lane."""
    var fmin = KdeV(bitcast[DType.float32](ORACLE_FLOAT32_MIN_BITS))
    if kernel == KDE_KERNEL_GAUSSIAN:
        var num = -ftz_v[KDE_W](x * x)
        var den = ftz(ftz(Float32(2.0) * h) * h)
        return ftz_v[KDE_W](num / KdeV(den))
    if kernel == KDE_KERNEL_EXPONENTIAL:
        return ftz_v[KDE_W]((-x) / KdeV(h))
    var outside = x.ge(KdeV(h))
    if kernel == KDE_KERNEL_TOPHAT:
        return outside.select(fmin, KdeV(0.0) * fmin)
    var z: KdeV
    if kernel == KDE_KERNEL_EPANECHNIKOV:
        var hsq = ftz(h * h)
        z = ftz_v[KDE_W](KdeV(1.0) - ftz_v[KDE_W](ftz_v[KDE_W](x * x) / KdeV(hsq)))
    elif kernel == KDE_KERNEL_LINEAR:
        z = ftz_v[KDE_W](KdeV(1.0) - ftz_v[KDE_W](x / KdeV(h)))
    else:
        var arg = ftz_v[KDE_W](ftz_v[KDE_W](KdeV(Float32(1.5707963267948966)) * x) / KdeV(h))
        z = arg
        comptime for l in range(KDE_W):
            z[l] = ftz(identical_cos(arg[l]))
    z = z.lt(KdeV(ORACLE_LOG_FLOOR)).select(KdeV(ORACLE_LOG_FLOOR), z)
    return outside.select(fmin, ftz_v[KDE_W](logf_v[KDE_W](z)))


def _kde_lse_row(row: HostF32Ptr, n_train: Int) -> Tuple[Float32, Float32]:
    """`oracle_logsumexp_row` over one row buffer (see the block engine)."""
    comptime if C52_PAIR:
        # pair_row reads the caller-owned buffer through its shared raw ABI.
        var raw = MutPointer[Float32, MutAnyOrigin](unsafe_from_address=Int(row))
        return pair_row(raw, n_train, kde_chunk_rows_for(n_train))
    var max_exp = row.unsafe_load(0)
    for j in range(1, n_train):
        var v = row.unsafe_load(j)
        if v > max_exp:
            max_exp = v
    if max_exp == bitcast[DType.float32](UInt32(0xFF800000)):
        return (max_exp, max_exp)
    var nb = (n_train + KDE_W - 1) // KDE_W
    var s = Float32(0.0)
    comptime if KDE_ORACLE_HOST_SABOTAGE:
        s = Float32(1.0)
        for bb in range(nb):
            var b = nb - 1 - bb
            var e = ftz_v[KDE_W](expf_v[KDE_W](ftz_v[KDE_W](
                row.unsafe_load[width=KDE_W](b * KDE_W) - KdeV(max_exp)
            )))
            for ll in range(KDE_W):
                var l = KDE_W - 1 - ll
                if b * KDE_W + l < n_train:
                    s = ftz(s + e[l])
    else:
        for b in range(nb):
            var e = ftz_v[KDE_W](expf_v[KDE_W](ftz_v[KDE_W](
                row.unsafe_load[width=KDE_W](b * KDE_W) - KdeV(max_exp)
            )))
            var cnt = min(KDE_W, n_train - b * KDE_W)
            for l in range(cnt):
                s = ftz(s + e[l])
    return (max_exp, ftz(identical_log(s) + max_exp))


def _kde_lse_row_chunked(row: HostF32Ptr, n_train: Int) -> Tuple[Float32, Float32]:
    """The device's chunked log-sum-exp (lane/fam-neighbors, 2026-10-04;
    `kde_chunk_lse_kernel` then `kde_chunk_lse_reduce_kernel` in
    kde/impl/neighbors/kernel_density.mojo) over one row buffer: per chunk
    of `kde_chunk_rows_for(n_train)` cells, ascending, the pair (m, s) with
    the rescale on a new strict max; then the chunk maxima's max and the
    chunk sums scaled to it, chunks ascending. Returns `(rowmax, lse)`."""
    comptime if C52_PAIR:
        var raw = MutPointer[Float32, MutAnyOrigin](unsafe_from_address=Int(row))
        return pair_row(raw, n_train, kde_chunk_rows_for(n_train))
    var neg_inf = bitcast[DType.float32](UInt32(0xFF800000))
    var chunk_rows = kde_chunk_rows_for(n_train)
    var n_chunks = (n_train + chunk_rows - 1) // chunk_rows
    var pm = List[Float32](length=n_chunks, fill=neg_inf)
    var ps = List[Float32](length=n_chunks, fill=Float32(0.0))
    for c in range(n_chunks):
        var j_end = min((c + 1) * chunk_rows, n_train)
        var m = neg_inf
        var s = Float32(0.0)
        for j in range(c * chunk_rows, j_end):
            var v = row.unsafe_load(j)
            if v > m:
                if m == neg_inf:
                    s = Float32(1.0)
                else:
                    s = ftz(identical_mul_add(s, ftz(identical_exp(ftz(m - v))), Float32(1.0)))
                m = v
            elif v != neg_inf:
                s = ftz(s + ftz(identical_exp(ftz(v - m))))
        pm[c] = m
        ps[c] = s
    var mx = pm[0]
    for c in range(1, n_chunks):
        if pm[c] > mx:
            mx = pm[c]
    if mx == neg_inf:
        return (mx, mx)
    var tot = Float32(0.0)
    comptime if KDE_ORACLE_HOST_SABOTAGE:
        # THE SABOTAGE ARM: one extra unit; wrong on purpose.
        tot = Float32(1.0)
    for c in range(n_chunks):
        if pm[c] != neg_inf:
            tot = ftz(identical_mul_add(ps[c], ftz(identical_exp(ftz(pm[c] - mx))), tot))
    return (mx, ftz(identical_log(tot) + mx))


def oracle_score_samples_into(
    train: HostF32Ptr,
    query: HostF32Ptr,
    weights: List[Float32],
    has_weights: Bool,
    n_train: Int,
    n_query: Int,
    d: Int,
    h: Float32,
    kernel: Int,
    metric: Int,
    scores: HostF32Ptr,
    tasks: Int,
    metric_arg: Float32 = Float32(2.0),
) raises:
    """`oracle_score_samples` over the caller's memory, the scores only
    (lane/infer-speed-classical, 2026-09-17, DEVIATION 2920): the two norm
    vectors, the log weights, `log_sw` and the kernel norm first, in that
    function's order; then the query rows split into at most `tasks`
    contiguous ranges (`core/host_predict_threads.mojo`). A task keeps one
    `n_train` row of log kernels and, per query row, spells that
    function's row statements in its order: every training row's
    `oracle_distance_ptr`, `oracle_log_kernel(ftz(dist), h, kernel)`, the
    weight's log added, then `oracle_logsumexp_row` over the row and the
    two subtractions. A row reads the inputs and its own row only, so the
    split moves no bit; the stages the checks read are not materialized
    (a 2,000 x 100,000 score held two 800 MB stage matrices)."""
    var q_norms = List[Float32](length=n_query, fill=Float32(0.0))
    var t_norms = List[Float32](length=n_train, fill=Float32(0.0))
    var use_norms = metric == DIST_L2_EXPANDED or metric == DIST_COSINE_EXPANDED
    if metric == DIST_L2_EXPANDED:
        for q in range(n_query):
            q_norms[q] = _host_row_norm_halving_ptr(query, q, d)
        for j in range(n_train):
            t_norms[j] = _host_row_norm_halving_ptr(train, j, d)
    elif metric == DIST_COSINE_EXPANDED:
        for q in range(n_query):
            q_norms[q] = _host_row_norm_halving_sqrt_ptr(query, q, d)
        for j in range(n_train):
            t_norms[j] = _host_row_norm_halving_sqrt_ptr(train, j, d)

    var logw = List[Float32](length=n_train, fill=Float32(0.0))
    if has_weights:
        for j in range(n_train):
            logw[j] = ftz(identical_log(ftz(weights[j])))

    var sum_w = Float32(0.0)
    if has_weights:
        for j in range(n_train):
            sum_w = ftz(sum_w + weights[j])
    else:
        sum_w = Float32(n_train)
    var log_sw = ftz(identical_log(sum_w))
    var norm = log_kernel_norm(kernel, h, d)

    var t = tasks
    if t < 1:
        t = 1
    if t > n_query:
        t = n_query
    var chunk = host_predict_chunk(n_query, t)
    var qnp = host_list_ptr(q_norms)
    var failed = List[Int](length=t, fill=0)
    var fp = failed.unsafe_ptr()

    # The training operand, flushed, feature-major in blocks of KDE_W
    # columns (block jb, feature f, lane l at (jb*d + f)*W + l), its norms
    # and log weights padded to whole blocks. Read-only for every task.
    var nbk = (n_train + KDE_W - 1) // KDE_W
    var pan = List[Float32](length=nbk * d * KDE_W, fill=Float32(0.0))
    var tnpad = List[Float32](length=nbk * KDE_W, fill=Float32(0.0))
    var lwpad = List[Float32](length=nbk * KDE_W, fill=Float32(0.0))
    for j in range(n_train):
        var jb = j // KDE_W
        var l = j % KDE_W
        for f in range(d):
            pan[(jb * d + f) * KDE_W + l] = ftz(train.unsafe_load(j * d + f))
        tnpad[j] = t_norms[j]
        lwpad[j] = logw[j]
    var panp = host_list_ptr(pan)
    var tnpp = host_list_ptr(tnpad)
    var lwpp = host_list_ptr(lwpad)

    def _rows(c: Int) {imm query, imm qnp, imm panp, imm tnpp, imm lwpp, imm scores, imm fp, imm chunk, imm n_query, imm n_train, imm nbk, imm d, imm h, imm kernel, imm metric, imm metric_arg, imm has_weights, imm log_sw, imm norm}:
        try:
            var rowbuf = List[Float32](length=KDE_QB * nbk * KDE_W, fill=Float32(0.0))
            var qbuf = List[Float32](length=KDE_QB * d, fill=Float32(0.0))
            var tl = List[Float32](length=KDE_QB * KDE_W, fill=Float32(0.0))
            var rbp = host_list_ptr(rowbuf)
            var qbp = host_list_ptr(qbuf)
            var tlp = host_list_ptr(tl)
            var rstride = nbk * KDE_W
            var lo = c * chunk
            var hi = min(lo + chunk, n_query)
            var q0 = lo
            while q0 < hi:
                var nq = min(KDE_QB, hi - q0)
                for r in range(KDE_QB):
                    var src = q0 + (r if r < nq else 0)
                    for f in range(d):
                        qbp.unsafe_store(r * d + f, ftz(query.unsafe_load(src * d + f)))
                for jb in range(nbk):
                    var pb = panp + jb * d * KDE_W
                    _kde_tile(qbp, pb, d, metric, metric_arg, tlp)
                    var tn = tnpp.unsafe_load[width=KDE_W](jb * KDE_W)
                    var lw = lwpp.unsafe_load[width=KDE_W](jb * KDE_W)
                    for r in range(nq):
                        var acc = tlp.unsafe_load[width=KDE_W](r * KDE_W)
                        var dist = _kde_epilogue(acc, qnp.unsafe_load(q0 + r), tn, metric, metric_arg)
                        var v = _kde_log_kernel_v(ftz_v[KDE_W](dist), h, kernel)
                        if has_weights:
                            v = ftz_v[KDE_W](v + lw)
                        rbp.unsafe_store(r * rstride + jb * KDE_W, v)
                for r in range(nq):
                    var mm: Tuple[Float32, Float32]
                    if kde_chunk_lse_metric_applies(metric):
                        mm = _kde_lse_row_chunked(rbp + r * rstride, n_train)
                    else:
                        mm = _kde_lse_row(rbp + r * rstride, n_train)
                    var a = ftz(mm[1] - log_sw)
                    scores.unsafe_store(q0 + r, ftz(a - norm))
                q0 += KDE_QB
            _ = rowbuf^
            _ = qbuf^
            _ = tl^
        except:
            fp.unsafe_store(c, 1)

    if t == 1:
        _rows(0)
    else:
        host_parallelize(_rows, t)
    _ = q_norms^
    _ = t_norms^
    _ = logw^
    _ = pan^
    _ = tnpad^
    _ = lwpad^
    for c in range(t):
        if failed[c] != 0:
            raise Error("kde host: score row chunk " + String(c) + " raised")
    _ = failed^


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
        # the mathematics, not the reference defect.
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
    query: List[Float32],
    train: List[Float32],
    q: Int,
    j: Int,
    d: Int,
    metric: Int,
    metric_arg: Float64 = 2.0,
) -> Float64:
    """The metric as MATHEMATICS in float64, independently of how any
    kernel computes it: `sqeuclidean` is the plain sum of squared
    differences (no expansion), `cosine` is `1 - dot/(|x| |y|)` with the
    norms taken directly (no expansion either), `minkowski` is
    `(sum |x-y|^p)^(1/p)` through the HOST libm `pow`, not through
    `exp(p log)`.

    Independence is the point. The float32 path builds cosine out of an
    expanded dot product and Minkowski out of `identical_pow`; if this
    reference reused either construction it would agree with a wrong
    kernel. The host `pow` here is a THIRD arithmetic and that is
    deliberate."""
    var acc = 0.0
    if metric == DIST_COSINE_EXPANDED:
        var dot = 0.0
        var qn = 0.0
        var tn = 0.0
        for f in range(d):
            var a = Float64(query[q * d + f])
            var b = Float64(train[j * d + f])
            dot += a * b
            qn += a * a
            tn += b * b
        return 1.0 - dot / (sqrt(qn) * sqrt(tn))
    if metric == DIST_LP_UNEXPANDED:
        for f in range(d):
            var diff = abs(
                Float64(query[q * d + f]) - Float64(train[j * d + f])
            )
            acc += diff**metric_arg
        return acc ** (1.0 / metric_arg)
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
    metric_arg: Float64 = 2.0,
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
                reference_distance_f64(
                    query, train, q, j, d, metric, metric_arg
                ),
                h,
                kernel,
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
