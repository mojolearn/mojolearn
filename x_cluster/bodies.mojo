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

from checks.numerics import ftz, identical_div, identical_mul, identical_sqrt

comptime FPtr = MutPointer[Float32, MutAnyOrigin]
comptime IPtr = MutPointer[Int32, MutAnyOrigin]


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
        for p in range(n):
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


@always_inline
def ap_availability_col(r_m: FPtr, a_m: FPtr, n: Int, damping: Float32, k: Int):
    """sklearn `_affinity_propagation` (:112-122), column `k`: `Rp =
    max(R, 0)` off the diagonal and `R[k, k]` on it, `colsum` by one
    ascending fold, `A_new[i, k] = min(colsum - Rp[i, k], 0)` off the
    diagonal and `colsum - Rp[k, k]` on it, then the same damping."""
    var one_minus = ftz(Float32(1) - damping)
    var colsum = Float32(0)
    for i in range(n):
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
