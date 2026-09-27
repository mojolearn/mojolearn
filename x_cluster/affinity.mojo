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
the portable log, cos and sqrt, not NumPy's generator (NOT_IMPLEMENTED.tsv).
Each iteration is two device kernels over the resident S, R and A (a row per
thread for the responsibilities, a column per thread for the availabilities,
every fold ascending, the lowest index on an argmax tie) and one for the
exemplar flags; the convergence window, the exemplar refinement and the
labels are the reference's host logic from one source."""
from checks.numerics import ftz, identical_cos, identical_log, identical_mul, identical_sqrt
from x_cluster.bodies import SplitMix64
from x_cluster.ops import ClusterOps


def _std_normal(mut rng: SplitMix64) -> Float32:
    """Box-Muller, the cosine branch: u1 in (0, 1], u2 in [0, 1)."""
    var u1 = Float32(1) - Float32(rng.unit())
    if u1 <= Float32(0):
        u1 = Float32(1.1754944e-38)
    var u2 = Float32(rng.unit())
    var rad = identical_sqrt(ftz(identical_mul(Float32(-2), identical_log(u1))))
    var ang = identical_mul(Float32(6.2831855), u2)
    return ftz(identical_mul(rad, identical_cos(ang)))


def affinity_fit[O: ClusterOps](
    mut ops: O, x: List[Float32], n: Int, d: Int, precomputed: Bool, pref_mode: Int,
    pref_scalar: Float32, pref_array: List[Float32], damping: Float32, max_iter: Int,
    conv_iter: Int, seed: UInt64,
    mut centers_idx: List[Int32], mut labels: List[Int32], mut n_iter: Int, mut affinity: List[Float32],
) raises:
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
            raise Error("AffinityPropagation: the default (median) preference needs a similarity matrix with no positive entry; pass preference=")
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
    # the tie noise, row-major order of the draws
    var rng = SplitMix64(seed)
    var eps32 = Float32(1.1920929e-07)
    var tiny100 = Float32(1.1754944e-36)
    for t in range(n * n):
        var z = _std_normal(rng)
        var scale = ftz(ftz(identical_mul(eps32, s_m[t])) + tiny100)
        s_m[t] = ftz(s_m[t] + ftz(identical_mul(scale, z)))
    var ss = ops.put(s_m)
    var a_s = ops.zeros(n * n)
    var r_s = ops.zeros(n * n)
    var e_s = ops.zeros_i(n)
    var ring = List[Int32](length=n * conv_iter, fill=Int32(0))
    var e = List[Int32]()
    var it = 0
    var never_converged = True
    while it < max_iter:
        ops.ap_r(ss, a_s, r_s, n, damping)
        ops.ap_a(r_s, a_s, n, damping)
        ops.ap_e(a_s, r_s, n, e_s)
        e = ops.get_i(e_s, n)
        var K = 0
        for i in range(n):
            ring[i * conv_iter + it % conv_iter] = e[i]
            K += Int(e[i])
        if it >= conv_iter:
            var settled = 0
            for i in range(n):
                var se = 0
                for c in range(conv_iter):
                    se += Int(ring[i * conv_iter + c])
                if se == conv_iter or se == 0:
                    settled += 1
            if settled == n and K > 0:
                never_converged = False
                break
        it += 1
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
