# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE CLUSTER LANE'S SHARED ARITHMETIC (algorithm expansion, lane/algos-cluster).

Every per-element body of the lane lives here ONCE. A device kernel
(`x_cluster/device_ops.mojo`) calls a body with its thread index; the host
binding (`x_cluster/host/host_ops.mojo`) calls the SAME body in a plain loop
over the same indices. So the two columns run one source, and IDENTICAL rests
on the numerics leaves alone: every operand through `ftz`, every product
`identical_mul` (no contraction into a neighboring add on any target), every
quotient `identical_div`, every root `identical_sqrt`, every fold in ascending
index order and every argmin/argmax with the lowest index on a tie.

Nothing here imports `std.gpu` or `max.gpu`: this file is host-safe.

`REV` is the host sabotage arm (`-D MOJOLEARN_HOST_SABOTAGE=1`, read by
`x_cluster/host/host_ops.mojo`): it walks the feature fold of a distance
descending, which moves the low bits of every distance.
"""
from std.memory import bitcast

from std.bit import count_leading_zeros

from checks.numerics import ftz, identical_cos, identical_div, identical_exp, identical_log, identical_mul, identical_mul64, identical_pow, identical_sqrt

comptime FPtr = MutPointer[Float32, MutAnyOrigin]
comptime IPtr = MutPointer[Int32, MutAnyOrigin]


# DEVIATION 5100 (fold order: features ascending) and 5101 (contraction:
# the product pinned, never fused into the chain). IDENTITY_PATHS row 110;
# check x_cluster/checks/dist_check.mojo.
@always_inline
def sq_dist_rows[REV: Bool = False](a: FPtr, i: Int, b: FPtr, j: Int, d: Int) -> Float32:
    """Squared euclidean distance of row `i` of `a` to row `j` of `b`, by
    DIFFERENCE (never the expanded norm form), one ascending chain over the
    features: `acc = ftz(acc + ftz(t * t))`, `t = ftz(ftz(a) - ftz(b))`."""
    var acc = Float32(0)
    for q in range(d):
        var f = d - 1 - q if REV else q
        var t = ftz(ftz(a[i * d + f]) - ftz(b[j * d + f]))
        acc = ftz(acc + ftz(identical_mul(t, t)))
    return acc


@always_inline
def sqdist_cell[REV: Bool = False](a: FPtr, na: Int, b: FPtr, nb: Int, d: Int, dst: FPtr, cell: Int):
    """out[i * nb + j] = sq_dist_rows(a, i, b, j)."""
    var i = cell // nb
    var j = cell - i * nb
    dst[cell] = sq_dist_rows[REV](a, i, b, j, d)


# DEVIATION 5102 (argmin tie: the lowest index). Row 111; nearest_check.
@always_inline
def nearest_row[REV: Bool = False](
    a: FPtr, b: FPtr, nb: Int, d: Int, labels: IPtr, dist: FPtr, i: Int
):
    """Nearest row of `b` to row `i` of `a`: the lowest squared distance under
    a strict `<`, so a tie keeps the LOWEST index."""
    var best = sq_dist_rows[REV](a, i, b, 0, d)
    var bi = 0
    for j in range(1, nb):
        var v = sq_dist_rows[REV](a, i, b, j, d)
        if v < best:
            best = v
            bi = j
    labels[i] = Int32(bi)
    dist[i] = best


@always_inline
def sqrt_cell(x: FPtr, i: Int):
    """x[i] = identical_sqrt(max(x[i], 0))."""
    var v = x[i]
    x[i] = identical_sqrt(v if v > Float32(0) else Float32(0))


# DEVIATION 5103 (the order statistic by a bisection on the bits: no order
# of the row can move it). Row 112; kth_check.
@always_inline
def kth_smallest_row(m: FPtr, n_cols: Int, k: Int, dst: FPtr, row: Int):
    """The k-th smallest (1-based) of row `row` of a NON-NEGATIVE matrix, by a
    bisection on the float bits (monotone for +0 .. +inf): the smallest bit
    pattern `v` with `count(x <= v) >= k`. Exact, and no order of the row can
    move it. A -0.0 reads as +0.0."""
    var lo = UInt32(0)
    var hi = UInt32(0x7F800000)
    while lo < hi:
        var mid = lo + (hi - lo) // 2
        var c = 0
        for j in range(n_cols):
            var bits = bitcast[DType.uint32](m[row * n_cols + j]) & UInt32(0x7FFFFFFF)
            if bits <= mid:
                c += 1
        if c >= k:
            hi = mid
        else:
            lo = mid + 1
    dst[row] = bitcast[DType.float32](lo)


# DEVIATION 5104 (the flat-kernel fold over the rows ascending, one quotient
# per feature, the rooted shift test). Row 113; meanshift_check.
@always_inline
def meanshift_seed[REV: Bool = False](
    x: FPtr, n: Int, d: Int, bw: Float32, stop: Float32, max_iter: Int,
    centers: FPtr, scratch: FPtr, intensity: IPtr, iters: IPtr, s: Int,
):
    """sklearn `_mean_shift_single_seed` (cluster/_mean_shift.py:97) for seed
    `s`, whose start is already in `centers[s]`: the flat kernel (every point
    with `sqrt(d2) <= bw`), the mean by one ascending fold over the points and
    ONE quotient per feature, the shift `sqrt(sum (new - old)^2)`, stop at
    `shift <= stop` or `completed == max_iter`. `scratch[s]` holds the sums."""
    var completed = 0
    var within = 0
    while True:
        within = 0
        for f in range(d):
            scratch[s * d + f] = Float32(0)
        for pp in range(n):
            var p = n - 1 - pp if REV else pp
            var dd = identical_sqrt(sq_dist_rows[REV](centers, s, x, p, d))
            if dd <= bw:
                within += 1
                for f in range(d):
                    scratch[s * d + f] = ftz(scratch[s * d + f] + ftz(x[p * d + f]))
        if within == 0:
            break
        var shift2 = Float32(0)
        var cnt = Float32(within)
        for f in range(d):
            var m = ftz(identical_div(scratch[s * d + f], cnt))
            var t = ftz(m - centers[s * d + f])
            shift2 = ftz(shift2 + ftz(identical_mul(t, t)))
            centers[s * d + f] = m
        if identical_sqrt(shift2) <= stop or completed == max_iter:
            break
        completed += 1
    intensity[s] = Int32(within)
    iters[s] = Int32(completed)


# DEVIATION 5122 (the AffinityPropagation tie noise by its COUNTER: draw j of
# the lane's splitmix64 stream is mix(seed + j * gamma), and the float32 of
# its unit double is rounded from the integer, so a cell needs no stream
# position and no Float64). Row 202; ap_noise_check.
comptime SPLITMIX_GAMMA = UInt64(0x9E3779B97F4A7C15)


@always_inline
def splitmix_at(seed: UInt64, j: UInt64) -> UInt64:
    """Draw `j` (1-based) of `SplitMix64(seed)`: its state after j steps is
    seed + j * gamma (wrapping), and `next` mixes that state."""
    var z = seed + j * SPLITMIX_GAMMA
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


@always_inline
def unit_f32[TRUNC: Bool = False](v: UInt64) -> Float32:
    """`Float32(SplitMix64.unit())` for the draw `v`, without a Float64
    (Apple GPUs have none): the unit double is the 53-bit integer
    `v >> 11` times 2^-53, EXACT, so its float32 is that integer rounded to
    24 significant bits, to nearest, a tie to even, then scaled by an exact
    power of two (the result is a normal float32 or 0)."""
    var m = v >> 11
    if m == UInt64(0):
        return Float32(0)
    var p = 64 - Int(count_leading_zeros(m))
    var e = -53
    if p > 24:
        var sh = UInt64(p - 24)
        var q = m >> sh
        comptime if not TRUNC:
            var rem = m & ((UInt64(1) << sh) - UInt64(1))
            var half = UInt64(1) << (sh - UInt64(1))
            if rem > half or (rem == half and (q & UInt64(1)) == UInt64(1)):
                q += UInt64(1)
        m = q
        e += Int(sh)
    return Float32(UInt32(m)) * bitcast[DType.float32](UInt32(127 + e) << 23)


@always_inline
def ap_noise_cell[TRUNC: Bool = False](s_m: FPtr, seed: UInt64, t: Int):
    """sklearn's tie noise on cell `t` of S (`S += (eps * S + tiny * 100) *
    z`), z the Box-Muller normal of draws 2t + 1 (u1) and 2t + 2 (u2): the
    cosine branch, u1 in (0, 1], the portable log, cos and sqrt."""
    var u1 = Float32(1) - unit_f32[TRUNC](splitmix_at(seed, UInt64(2 * t + 1)))
    if u1 <= Float32(0):
        u1 = Float32(1.1754944e-38)
    var u2 = unit_f32[TRUNC](splitmix_at(seed, UInt64(2 * t + 2)))
    var rad = identical_sqrt(ftz(identical_mul(Float32(-2), identical_log(u1))))
    var ang = identical_mul(Float32(6.2831855), u2)
    var z = ftz(identical_mul(rad, identical_cos(ang)))
    var scale = ftz(ftz(identical_mul(Float32(1.1920929e-07), s_m[t])) + Float32(1.1754944e-36))
    s_m[t] = ftz(s_m[t] + ftz(identical_mul(scale, z)))


# DEVIATION 5105 (damping as two pinned products and one add). Row 114;
# ap_check.
@always_inline
def ap_responsibility_row(s_m: FPtr, a_m: FPtr, r_m: FPtr, n: Int, damping: Float32, i: Int):
    """sklearn `_affinity_propagation` (cluster/_affinity_propagation.py:
    94-110), row `i`: `AS = A + S`, its max (lowest index on a tie) and the
    second max, `R_new = S - max` (`S - second` at the argmax), then
    `R = R * damping + R_new * (1 - damping)`."""
    var one_minus = ftz(Float32(1) - damping)
    var first = ftz(a_m[i * n] + s_m[i * n])
    var arg = 0
    for k in range(1, n):
        var v = ftz(a_m[i * n + k] + s_m[i * n + k])
        if v > first:
            first = v
            arg = k
    var second = Float32(-3.4028234663852886e38)
    var have_second = False
    for k in range(n):
        if k == arg:
            continue
        var v = ftz(a_m[i * n + k] + s_m[i * n + k])
        if not have_second or v > second:
            second = v
            have_second = True
    for k in range(n):
        var sub = second if k == arg else first
        var new = ftz(s_m[i * n + k] - sub)
        var old = r_m[i * n + k]
        r_m[i * n + k] = ftz(ftz(identical_mul(old, damping)) + ftz(identical_mul(new, one_minus)))


# DEVIATION 5106 (the availability column fold ascending, the clamp at 0
# off the diagonal). Row 115; ap_check.
@always_inline
def ap_availability_col[REV: Bool = False](r_m: FPtr, a_m: FPtr, n: Int, damping: Float32, k: Int):
    """sklearn `_affinity_propagation` (:112-122), column `k`: `Rp =
    max(R, 0)` off the diagonal and `R[k, k]` on it, `colsum` by one
    ascending fold, `A_new[i, k] = min(colsum - Rp[i, k], 0)` off the
    diagonal and `colsum - Rp[k, k]` on it, then the same damping."""
    var one_minus = ftz(Float32(1) - damping)
    var colsum = Float32(0)
    for ii in range(n):
        var i = n - 1 - ii if REV else ii
        var v = r_m[i * n + k]
        var rp = v if (i == k or v > Float32(0)) else Float32(0)
        colsum = ftz(colsum + rp)
    for i in range(n):
        var v = r_m[i * n + k]
        var rp = v if (i == k or v > Float32(0)) else Float32(0)
        var new = ftz(colsum - rp)
        if i != k and new > Float32(0):
            new = Float32(0)
        var old = a_m[i * n + k]
        a_m[i * n + k] = ftz(ftz(identical_mul(old, damping)) + ftz(identical_mul(new, one_minus)))


@always_inline
def ap_exemplar_cell(a_m: FPtr, r_m: FPtr, n: Int, e: IPtr, i: Int):
    """e[i] = (A[i, i] + R[i, i] > 0), sklearn's `E`."""
    e[i] = Int32(1) if ftz(a_m[i * n + i] + r_m[i * n + i]) > Float32(0) else Int32(0)


comptime METRIC_EUCLIDEAN = 0
comptime METRIC_MANHATTAN = 1
comptime METRIC_CHEBYSHEV = 2
comptime METRIC_MINKOWSKI = 3
comptime METRIC_COSINE = 4


# DEVIATION 5111 (the non-euclidean metrics: every fold over the features
# ascending, the portable pow and sqrt, cosine's zero-norm row at distance 1,
# every distance clamped at +0). OPTICS's metric option; pdist_check.
@always_inline
def pdist_cell[REV: Bool = False](a: FPtr, na: Int, b: FPtr, nb: Int, d: Int, metric: Int, p: Float32, dst: FPtr, cell: Int):
    var i = cell // nb
    var j = cell - i * nb
    var v = Float32(0)
    if metric == METRIC_EUCLIDEAN:
        v = identical_sqrt(sq_dist_rows[REV](a, i, b, j, d))
    elif metric == METRIC_MANHATTAN:
        for ff in range(d):
            var f = d - 1 - ff if REV else ff
            v = ftz(v + abs(ftz(ftz(a[i * d + f]) - ftz(b[j * d + f]))))
    elif metric == METRIC_CHEBYSHEV:
        for f in range(d):
            var t = abs(ftz(ftz(a[i * d + f]) - ftz(b[j * d + f])))
            if t > v:
                v = t
    elif metric == METRIC_MINKOWSKI:
        for f in range(d):
            v = ftz(v + ftz(identical_pow(abs(ftz(ftz(a[i * d + f]) - ftz(b[j * d + f]))), p)))
        v = ftz(identical_pow(v, ftz(identical_div(Float32(1), p))))
    else:
        var dot = Float32(0)
        var na2 = Float32(0)
        var nb2 = Float32(0)
        for f in range(d):
            var x = ftz(a[i * d + f])
            var y = ftz(b[j * d + f])
            dot = ftz(dot + ftz(identical_mul(x, y)))
            na2 = ftz(na2 + ftz(identical_mul(x, x)))
            nb2 = ftz(nb2 + ftz(identical_mul(y, y)))
        if na2 == Float32(0) or nb2 == Float32(0):
            v = Float32(1)
        else:
            var den = ftz(identical_mul(identical_sqrt(na2), identical_sqrt(nb2)))
            v = ftz(Float32(1) - ftz(identical_div(dot, den)))
    dst[cell] = v if v > Float32(0) else Float32(0)


# ------------------------------------------------ Gaussian mixture bodies
# DEVIATION 5108 (the Mahalanobis fold: difference first, `a` then `j`
# ascending, pinned products). Row 117; gauss_check.
@always_inline
def gauss_q_cell[REV: Bool = False](x: FPtr, d: Int, means: FPtr, pchol: FPtr, kc: Int, dst: FPtr, cell: Int):
    """dst[i, k] = || (x_i - mu_k) P_k ||^2, P_k the UPPER-triangular
    precision Cholesky (d x d, row-major): y_j = sum_{a <= j} (x_a - mu_a)
    P[a, j] in ascending a, then the ascending sum of y_j^2. sklearn's
    `_estimate_log_gaussian_prob` ('full') forms `X @ P - mu @ P`; the
    difference first is ours."""
    var i = cell // kc
    var k = cell - i * kc
    var acc = Float32(0)
    for j in range(d):
        var y = Float32(0)
        for aa in range(j + 1):
            var a = j - aa if REV else aa
            var diff = ftz(ftz(x[i * d + a]) - ftz(means[k * d + a]))
            y = ftz(y + ftz(identical_mul(diff, pchol[k * d * d + a * d + j])))
        acc = ftz(acc + ftz(identical_mul(y, y)))
    dst[cell] = acc


# DEVIATION 5109 (the E-step log-sum-exp: the first max, the ascending sum
# of the portable exp, the portable log). Row 118; gauss_check.
@always_inline
def resp_row(q: FPtr, c: FPtr, kc: Int, lpn: FPtr, i: Int):
    """Row i of the E-step: v_k = c_k - q_ik / 2 (the weighted log
    probability), the row max (the first on a tie), `lse = max +
    log(sum exp(v - max))` over ascending k, then q_ik <- v_k - lse (the log
    responsibility) and lpn[i] = lse."""
    var mx = Float32(0)
    for k in range(kc):
        var v = ftz(c[k] - ftz(identical_mul(Float32(0.5), q[i * kc + k])))
        q[i * kc + k] = v
        if k == 0 or v > mx:
            mx = v
    var s = Float32(0)
    for k in range(kc):
        s = ftz(s + ftz(identical_exp(ftz(q[i * kc + k] - mx))))
    var lse = ftz(mx + ftz(identical_log(s)))
    for k in range(kc):
        q[i * kc + k] = ftz(q[i * kc + k] - lse)
    lpn[i] = lse


@always_inline
def exp_cell(src: FPtr, dst: FPtr, t: Int):
    dst[t] = ftz(identical_exp(src[t]))


# DEVIATION 5110 (the M-step moments nk, means, covariances: every fold over
# the rows ascending, one quotient). Row 119; moments_check. The per-row
# steps and the finals below are the ONE spelling: the cells here (the host
# column) and the device's tiled kernel (`device_ops._moments_kernel`,
# DEVIATION 5121) both call them, in the same row order.
@always_inline
def chain_add(acc: Float32, t: Float32) -> Float32:
    """One link of every moments chain: the addend onto the running sum."""
    return ftz(acc + t)


@always_inline
def nk_step(acc: Float32, r: Float32) -> Float32:
    return chain_add(acc, r)


@always_inline
def nk_final(acc: Float32) -> Float32:
    """+ 10 * FLT_EPSILON (sklearn, float32 input)."""
    return ftz(acc + Float32(1.1920929e-06))


@always_inline
def xk_term(r: Float32, xa: Float32) -> Float32:
    """One row's addend of a mean chain (independent of the chain, so a
    kernel may form several ahead of the adds without moving a bit)."""
    return ftz(identical_mul(r, ftz(xa)))


@always_inline
def xk_step(acc: Float32, r: Float32, xa: Float32) -> Float32:
    return chain_add(acc, xk_term(r, xa))


@always_inline
def mean_final(acc: Float32, nkv: Float32) -> Float32:
    return ftz(identical_div(acc, nkv))


@always_inline
def cov_term(r: Float32, xa: Float32, xb: Float32, ma: Float32, mb: Float32) -> Float32:
    """One row's addend of a covariance chain (independent of the chain)."""
    var da = ftz(ftz(xa) - ma)
    var db = ftz(ftz(xb) - mb)
    return ftz(identical_mul(r, ftz(identical_mul(da, db))))


@always_inline
def cov_step(acc: Float32, r: Float32, xa: Float32, xb: Float32, ma: Float32, mb: Float32) -> Float32:
    return chain_add(acc, cov_term(r, xa, xb, ma, mb))


@always_inline
def cov_final(acc: Float32, nkv: Float32, reg: Float32, diag: Bool) -> Float32:
    var v = ftz(identical_div(acc, nkv))
    if diag:
        v = ftz(v + reg)
    return v


@always_inline
def nk_cell(resp: FPtr, n: Int, kc: Int, dst: FPtr, k: Int):
    """nk = sum_i resp[i, k] + 10 * FLT_EPSILON (sklearn, float32 input)."""
    var acc = Float32(0)
    for i in range(n):
        acc = nk_step(acc, resp[i * kc + k])
    dst[k] = nk_final(acc)


@always_inline
def xk_cell(resp: FPtr, x: FPtr, n: Int, d: Int, kc: Int, nk: FPtr, dst: FPtr, cell: Int):
    """means[k, a] = sum_i resp[i, k] x[i, a] / nk[k]."""
    var k = cell // d
    var a = cell - k * d
    var acc = Float32(0)
    for i in range(n):
        acc = xk_step(acc, resp[i * kc + k], x[i * d + a])
    dst[cell] = mean_final(acc, nk[k])


@always_inline
def cov_cell(
    resp: FPtr, x: FPtr, n: Int, d: Int, kc: Int, means: FPtr, nk: FPtr, reg: Float32, dst: FPtr, cell: Int
):
    """cov[k, a, b] = sum_i resp[i, k] (x_ia - m_ka)(x_ib - m_kb) / nk[k],
    + reg on the diagonal (sklearn `_estimate_gaussian_covariances_full`)."""
    var k = cell // (d * d)
    var r = cell - k * d * d
    var a = r // d
    var b = r - a * d
    var acc = Float32(0)
    for i in range(n):
        acc = cov_step(acc, resp[i * kc + k], x[i * d + a], x[i * d + b], means[k * d + a], means[k * d + b])
    dst[cell] = cov_final(acc, nk[k], reg, a == b)


# ------------------------------------------------------------------ host RNG
@fieldwise_init
struct SplitMix64(Copyable, Movable):
    """splitmix64 (Steele, Lea, Flood 2014): the lane's one host stream. Host
    code, run the same way in both bindings."""

    var state: UInt64

    def next(mut self) -> UInt64:
        self.state = self.state + UInt64(0x9E3779B97F4A7C15)
        var z = self.state
        z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
        return z ^ (z >> 31)

    def below(mut self, n: Int) -> Int:
        """An index in [0, n): `u % n` (n is far below 2^32 here)."""
        return Int(self.next() % UInt64(n))

    def unit(mut self) -> Float64:
        """A double in [0, 1): the top 53 bits times 2^-53."""
        return Float64(self.next() >> 11) * Float64(1.1102230246251565e-16)


# DEVIATION 5107 (the LEFT child on an exact tie). Row 116; descend_check.
@always_inline
def tree_descend[REV: Bool = False](
    x: FPtr, d: Int, centers: FPtr, nodes: IPtr, labels: IPtr, i: Int
):
    """BisectingKMeans `_predict_recursive` (sklearn cluster/_bisect_k_means.py
    :490-540) for row `i`: from the root (node 0) take the nearer of the two
    child centers, the LEFT one on a tie (`_labels_inertia`'s argmin), until a
    leaf. `nodes` is (left, right, label) per node, left == -1 at a leaf."""
    var node = 0
    while nodes[node * 3] >= 0:
        var l = Int(nodes[node * 3])
        var r = Int(nodes[node * 3 + 1])
        var dl = sq_dist_rows[REV](x, i, centers, l, d)
        var dr = sq_dist_rows[REV](x, i, centers, r, d)
        node = r if dr < dl else l
    labels[i] = nodes[node * 3 + 2]


# ------------------------------------------------ agglomerative (Lance-Williams)
comptime LINK_WARD = 0
comptime LINK_COMPLETE = 1
comptime LINK_AVERAGE = 2
comptime LINK_SINGLE = 3


# DEVIATION 5117 (the Lance-Williams update of agglomerative linkage: Float64
# arithmetic from the Float32 matrix; ward as three pinned products summed
# LEFT TO RIGHT, minus last, then ONE quotient by the three sizes summed
# left to right; average as two pinned products, one add, one quotient;
# complete/single exact max/min; a result below zero clamped to +0; the
# Float32 rounding then flushed). Row 120; agglo_check.
@always_inline
def lance_williams(
    linkage: Int, dak: Float32, dbk: Float32, dab: Float32, na: Float64, nb: Float64, nk: Float64,
    has_a: Bool, has_b: Bool,
) -> Float32:
    """The dissimilarity of the merged cluster (a u b) to cluster k.

    ward (on SQUARED euclidean dissimilarities, scipy `_hierarchy_distance_
    update.pxi::_ward` squared out): ((na + nk) dak + (nb + nk) dbk - nk dab)
    / (na + nb + nk), always from both (the squared matrix is complete).
    complete / average / single: max, the size-weighted mean
    (na dak + nb dbk) / (na + nb), min. Under a connectivity graph only the
    pairs that are EDGES exist (scikit-learn `_hierarchical_fast.max_merge` /
    `average_merge`): an edge from one side alone keeps its value."""
    if linkage == LINK_WARD:
        var t1 = identical_mul64(na + nk, Float64(dak))
        var t2 = identical_mul64(nb + nk, Float64(dbk))
        var t3 = identical_mul64(nk, Float64(dab))
        var w = ((t1 + t2) - t3) / ((na + nb) + nk)
        return ftz(Float32(w)) if w > Float64(0) else Float32(0)
    if not has_b:
        return dak
    if not has_a:
        return dbk
    if linkage == LINK_COMPLETE:
        return dak if dak >= dbk else dbk
    if linkage == LINK_SINGLE:
        return dak if dak <= dbk else dbk
    var v = (identical_mul64(na, Float64(dak)) + identical_mul64(nb, Float64(dbk))) / (na + nb)
    return ftz(Float32(v))
