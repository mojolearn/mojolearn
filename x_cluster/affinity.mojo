# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""AffinityPropagation (lane/algos-cluster). Reference: scikit-learn
`sklearn/cluster/_affinity_propagation.py` (`_affinity_propagation` :34-165,
`AffinityPropagation.fit` :500-550, `predict`).

S is `-(squared euclidean distance)` from the device (or the caller's
precomputed matrix); the default preference is the median of S, taken as the
device's exact order statistic of the n^2 distances. THE TIE NOISE is
sklearn's formula `S += (eps * S + tiny * 100) * z`, z standard normal, but
z comes from the lane's seeded splitmix64 stream through a Box-Muller with
the portable log, cos and sqrt, not NumPy's generator (NOT_IMPLEMENTED.tsv);
cell t reads draws 2t + 1 and 2t + 2 by their counter, on the device
(`bodies.ap_noise_cell`, DEVIATION 5122).
Each iteration is two device kernels over the resident S, R and A (a row per
thread for the responsibilities, a column per thread for the availabilities,
every fold ascending, the lowest index on an argmax tie) and one for the
exemplar flags; the convergence window, the exemplar refinement and the
labels are the reference's host logic from one source."""
from checks.numerics import ftz, identical_mul
from x_cluster.ops import ClusterOps


comptime AP_BATCH = 16
"""AffinityPropagation iterations enqueued per host read of the stop state."""


def affinity_fit[O: ClusterOps](
    mut ops: O, x: List[Float32], n: Int, d: Int, precomputed: Bool, pref_mode: Int,
    pref_scalar: Float32, pref_array: List[Float32], damping: Float32, max_iter: Int,
    conv_iter: Int, seed: UInt64,
    mut centers_idx: List[Int32], mut labels: List[Int32], mut n_iter: Int, mut affinity: List[Float32],
    mut ar_diag: List[Float32],
) raises:
    """`ar_diag` returns the final diagonals of A then R (2n floats; empty on
    the equal-similarities shortcut): the continuous state the exemplar
    choice reads, kept so a verifier lane can see the message arithmetic and
    not only the discrete labels it settles into."""
    var s_m: List[Float32]
    if precomputed:
        s_m = x.copy()
    else:
        var xs = ops.put(x)
        var dm = ops.zeros(n * n)
        ops.sqdist(xs, n, xs, n, d, dm)
        s_m = ops.get(dm, n * n)
        for t in range(n * n):
            s_m[t] = -s_m[t]
    affinity = s_m.copy()
    # the preference
    var pref = List[Float32](length=n, fill=pref_scalar)
    if pref_mode == 0:
        # np.median(S): the mean of the two middle values of n^2 (one when odd)
        var m = n * n
        var neg = List[Float32](capacity=m)
        var nonneg = True
        for t in range(m):
            neg.append(-s_m[t])
            if not (neg[t] >= Float32(0)):
                nonneg = False
        var med: Float32
        if nonneg:
            var ds = ops.put(neg)
            var ks = ops.zeros(1)
            ops.kth(ds, 1, m, m // 2 + 1, ks)
            var hi = ops.get(ks, 1)[0]
            if m % 2 == 1:
                med = -hi
            else:
                ops.kth(ds, 1, m, m // 2, ks)
                var lo = ops.get(ks, 1)[0]
                med = -ftz(identical_mul(ftz(lo + hi), Float32(0.5)))
        else:
            # a precomputed S with a positive entry: the two middle order
            # statistics by a host heap sort (one source in both bindings)
            var v = s_m.copy()
            _heap_sort(v)
            if m % 2 == 1:
                med = v[m // 2]
            else:
                med = ftz(identical_mul(ftz(v[m // 2 - 1] + v[m // 2]), Float32(0.5)))
        for i in range(n):
            pref[i] = med
    elif pref_mode == 2:
        for i in range(n):
            pref[i] = pref_array[i]
    # _equal_similarities_and_preferences
    var all_equal = True
    var first_off = Float32(0)
    var have = False
    for i in range(n):
        for j in range(n):
            if i != j:
                if not have:
                    first_off = s_m[i * n + j]
                    have = True
                elif s_m[i * n + j] != first_off:
                    all_equal = False
    for i in range(1, n):
        if pref[i] != pref[0]:
            all_equal = False
    if n == 1 or all_equal:
        n_iter = 0
        centers_idx = List[Int32]()
        labels = List[Int32]()
        if pref[0] > s_m[n - 1]:
            for i in range(n):
                centers_idx.append(Int32(i))
                labels.append(Int32(i))
        else:
            centers_idx.append(Int32(0))
            for _i in range(n):
                labels.append(Int32(0))
        return
    for i in range(n):
        s_m[i * n + i] = pref[i]
    # the tie noise, row-major order of the draws (cell t reads draws 2t + 1
    # and 2t + 2 of the seeded stream, `bodies.ap_noise_cell`, DEVIATION
    # 5122), on the column that holds S; the host keeps a copy for the
    # exemplar refinement below
    var ss = ops.put(s_m)
    ops.ap_noise(ss, n * n, seed)
    s_m = ops.get(ss, n * n)
    var a_s = ops.zeros(n * n)
    var r_s = ops.zeros(n * n)
    var e_s = ops.zeros_i(n)
    # the convergence window lives beside the matrices (`ops.ap_iterate`):
    # AP_BATCH iterations per host read of the stop state, the iteration
    # that converges freezing everything after it, so `it`, `e`, A and R are
    # the one-read-per-iteration loop's (lane/cluster-apple: on Metal each
    # read was a drain with pending work)
    var ring_s = ops.zeros_i(n * conv_iter if n * conv_iter > 0 else 1)
    var st_s = ops.zeros_i(2)
    var it = 0
    var never_converged = True
    while it < max_iter:
        var count = max_iter - it
        if count > AP_BATCH:
            count = AP_BATCH
        ops.ap_iterate(ss, a_s, r_s, e_s, ring_s, st_s, n, damping, conv_iter, it, count)
        var st = ops.get_i(st_s, 2)
        if st[0] != Int32(0):
            it = Int(st[1])
            never_converged = False
            break
        it += count
    var e = ops.get_i(e_s, n)
    var a_fin = ops.get(a_s, n * n)
    var r_fin = ops.get(r_s, n * n)
    ar_diag = List[Float32](capacity=2 * n)
    for i in range(n):
        ar_diag.append(a_fin[i * n + i])
    for i in range(n):
        ar_diag.append(r_fin[i * n + i])
    if never_converged:
        it = max_iter - 1
    n_iter = it + 1
    var ex = List[Int]()
    for i in range(n):
        if e[i] != 0:
            ex.append(i)
    var K = len(ex)
    labels = List[Int32]()
    centers_idx = List[Int32]()
    if K == 0:
        for _i in range(n):
            labels.append(Int32(-1))
        return
    var c = _argmax_cols(s_m, n, ex)
    for k in range(K):
        c[ex[k]] = k
    for k in range(K):
        var ii = List[Int]()
        for i in range(n):
            if c[i] == k:
                ii.append(i)
        var best = 0
        var best_v = Float32(0)
        for jj in range(len(ii)):
            var acc = Float32(0)
            for q in range(len(ii)):
                acc = ftz(acc + s_m[ii[q] * n + ii[jj]])
            if jj == 0 or acc > best_v:
                best_v = acc
                best = jj
        ex[k] = ii[best]
    c = _argmax_cols(s_m, n, ex)
    for k in range(K):
        c[ex[k]] = k
    # labels = I[c]; centers = unique(labels); labels = searchsorted(centers, labels)
    var is_center = List[Bool](length=n, fill=False)
    for i in range(n):
        is_center[ex[c[i]]] = True
    var rank = List[Int](length=n, fill=-1)
    for i in range(n):
        if is_center[i]:
            rank[i] = len(centers_idx)
            centers_idx.append(Int32(i))
    for i in range(n):
        labels.append(Int32(rank[ex[c[i]]]))


def _argmax_cols(s_m: List[Float32], n: Int, cols: List[Int]) -> List[Int]:
    """argmax over `cols` of each row of S, the lowest position on a tie."""
    var out = List[Int](capacity=n)
    for i in range(n):
        var best = 0
        var bv = s_m[i * n + cols[0]]
        for q in range(1, len(cols)):
            var v = s_m[i * n + cols[q]]
            if v > bv:
                bv = v
                best = q
        out.append(best)
    return out^


def _heap_sort(mut v: List[Float32]):
    """Ascending, in place (a -0.0 and a +0.0 compare equal; the median of
    them is the same either way)."""
    var n = len(v)

    def sift(mut a: List[Float32], start: Int, end: Int):
        var root = start
        while 2 * root + 1 <= end:
            var child = 2 * root + 1
            if child + 1 <= end and a[child] < a[child + 1]:
                child += 1
            if a[root] < a[child]:
                var t = a[root]
                a[root] = a[child]
                a[child] = t
                root = child
            else:
                return

    var start = (n - 2) // 2
    while start >= 0:
        sift(v, start, n - 1)
        start -= 1
    var end = n - 1
    while end > 0:
        var t = v[0]
        v[0] = v[end]
        v[end] = t
        end -= 1
        sift(v, 0, end)
