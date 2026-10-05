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
exemplar flags; the convergence window (one flag read an iteration), the
exemplar refinement and the labels are device primitives too (lane
cgr2-cluster; the host column runs the reference's loops)."""
from std.sys.compile import is_defined

from checks.kernel_matrix import COLUMN_APPLE, TARGET_COLUMN
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, ftz, identical_mul
from x_cluster.ops import ClusterOps
from x_cluster.optics import dist_slot

# Lane cluster-apple3, FAST and the GPU binding only, OPT-IN while unproven.
# `-D MOJOLEARN_AP_EXACT=1`: the same values by less work (the median by the
# grid-wide radix select straight from the device's distances, the two final
# diagonals gathered on the device, the equal-similarities scan stopped at
# its first difference); since lane cgr2-cluster every build takes these
# (the radix median, the device diagonals, the device equality test), so the
# define changes nothing. `-D MOJOLEARN_AP_SPLIT=1`: the availability column
# sums folded over row slices (bits move; the paired quality check).
comptime AP_EXACT = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and is_defined["MOJOLEARN_AP_EXACT"]()
# AP_SPLIT is the FAST + Apple DEFAULT since the M3 A/B (lane
# apple-fast-cluster2: affinity-prop istella 394.8 -> 264.8 ms, n=1, quality
# identical); off with `-D MOJOLEARN_AP_SPLIT_OFF`. The old
# `-D MOJOLEARN_AP_SPLIT=1` is harmless; other FAST columns stay as before.
comptime AP_SPLIT = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and TARGET_COLUMN == COLUMN_APPLE and not is_defined["MOJOLEARN_AP_SPLIT_OFF"]()


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
    var m = n * n
    # S on the device: -(squared distances), or the caller's matrix
    var ss: Int
    var neg: Int
    if precomputed:
        ss = ops.put(x)
        affinity = x.copy()
        neg = ops.alloc(m)
        ops.negate(ss, neg, m)
    else:
        var xs = ops.put(x)
        neg = dist_slot(ops, m)
        ops.sqdist(xs, n, xs, n, d, neg)
        ss = ops.alloc(m)
        ops.negate(neg, ss, m)
        affinity = ops.get(ss, m)
    # the preference
    var pref = List[Float32](length=n, fill=pref_scalar)
    if pref_mode == 0:
        # np.median(S): the mean of the two middle values of n^2 (one when
        # odd), exact order statistics by the device's radix select
        var med: Float32
        if ops.check_nonneg(neg, m):
            var hi = ops.kth_flat(neg, m, m // 2 + 1)
            if m % 2 == 1:
                med = -hi
            else:
                var lo = ops.kth_flat(neg, m, m // 2)
                med = -ftz(identical_mul(ftz(lo + hi), Float32(0.5)))
        else:
            # a precomputed S with a positive entry: each order statistic of
            # S from the side of zero it falls on
            var cneg = ops.count_neg(ss, m)
            var side = ops.alloc(m)
            var hi = _kth_signed(ops, ss, m, m // 2 + 1, cneg, side)
            if m % 2 == 1:
                med = hi
            else:
                var lo = _kth_signed(ops, ss, m, m // 2, cneg, side)
                med = ftz(identical_mul(ftz(lo + hi), Float32(0.5)))
        for i in range(n):
            pref[i] = med
    elif pref_mode == 2:
        for i in range(n):
            pref[i] = pref_array[i]
    var ps = ops.put(pref)
    # _equal_similarities_and_preferences
    if n == 1 or ops.ap_equal(ss, ps, n):
        n_iter = 0
        centers_idx = List[Int32]()
        labels = List[Int32]()
        var row0 = ops.get(ss, n)
        if pref[0] > row0[n - 1]:
            for i in range(n):
                centers_idx.append(Int32(i))
                labels.append(Int32(i))
        else:
            centers_idx.append(Int32(0))
            for _i in range(n):
                labels.append(Int32(0))
        return
    ops.set_diag(ss, ps, n)
    # the tie noise, row-major order of the draws (cell t reads draws 2t + 1
    # and 2t + 2 of the seeded stream, `bodies.ap_noise_cell`, DEVIATION
    # 5122), on the device; S stays resident for the exemplar refinement
    ops.ap_noise(ss, m, seed)
    var a_s = ops.zeros(m)
    var r_s = ops.zeros(m)
    var e_s = ops.zeros_i(n)
    var ring = ops.zeros_i(n * conv_iter)
    var it = 0
    var never_converged = True
    var split = False
    comptime if AP_SPLIT:
        split = ops.fast_device()
    # fam2-cluster: the device column runs the whole loop with the window
    # decided on the device (`ap_loop`; -1: this column does not take it)
    var taken = -1
    if not split:
        taken = ops.ap_loop(ss, a_s, r_s, e_s, ring, n, damping, max_iter, conv_iter)
    if taken >= 0:
        it = taken
        if taken < max_iter:
            never_converged = False
    else:
        while it < max_iter:
            ops.ap_r(ss, a_s, r_s, n, damping)
            if split:
                ops.ap_a_split(r_s, a_s, n, damping)
            else:
                ops.ap_a(r_s, a_s, n, damping)
            ops.ap_e(a_s, r_s, n, e_s)
            # the convergence window on the device; one flag read per iteration
            if ops.ap_conv(e_s, ring, n, conv_iter, it):
                never_converged = False
                break
            it += 1
    ar_diag = List[Float32](capacity=2 * n)
    var a_d = ops.get_diag(a_s, n)
    var r_d = ops.get_diag(r_s, n)
    for i in range(n):
        ar_diag.append(a_d[i])
    for i in range(n):
        ar_diag.append(r_d[i])
    if never_converged:
        it = max_iter - 1
    n_iter = it + 1
    # the exemplar refinement and the labels on the device
    var cen_s = ops.zeros_i(n)
    var lab_s = ops.zeros_i(n)
    var nc = ops.ap_exemplars(ss, e_s, n, cen_s, lab_s)
    labels = ops.get_i(lab_s, n)
    centers_idx = List[Int32]()
    if nc > 0:
        centers_idx = ops.get_i(cen_s, nc)


def _kth_signed[O: ClusterOps](mut ops: O, s: Int, m: Int, k: Int, cneg: Int, side: Int) raises -> Float32:
    """The k-th smallest (1-based) of m values of either sign: among the
    negatives the (cneg - k + 1)-th smallest magnitude, negated; else the
    (k - cneg)-th smallest of the non-negatives (`kth_flat` on one side, the
    other side +inf)."""
    if k <= cneg:
        ops.sign_side(s, m, True, side)
        return -ops.kth_flat(side, m, cneg - k + 1)
    ops.sign_side(s, m, False, side)
    return ops.kth_flat(side, m, k - cneg)
