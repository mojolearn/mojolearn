# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""BisectingKMeans (lane/algos-cluster). Reference: scikit-learn
`sklearn/cluster/_bisect_k_means.py` (`_BisectingTree` :20-80, `_bisect`
:300-360, `fit` :362-450, `_predict_recursive` :490-540).

The data are centered on their column means (sklearn's `_X_mean`); from the
one-leaf tree, `n_clusters - 1` times: the leaf with the highest score (the
first in depth-first left-to-right order on a tie, a strict `>`) is split by
2-means, `n_init` restarts keeping `inertia < best * (1 - 1e-6)`. THE 2-MEANS
IS THE REPOSITORY'S K-MEANS through `ClusterOps.kmeans` (cuVS's Lloyd,
`cluster/estimator.mojo::kmeans_fit` on the device and its host oracle on the
CPU), so its tolerance is cuVS's centroid-shift rule, not sklearn's
variance-scaled one (NOT_IMPLEMENTED.tsv). Each split's restart seed is the
next draw of the lane's splitmix64 stream. The scores are the per-child
inertia ('biggest_inertia') or size ('largest_cluster'). Leaves in
depth-first order are the labels; `predict` descends the tree on the device
(`bodies.tree_descend`)."""
from std.sys.compile import is_defined

from checks.kernel_matrix import COLUMN_APPLE, TARGET_COLUMN
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, ftz
from cluster.impl.kmeans_params import INIT_KMEANS_PLUS_PLUS, INIT_RANDOM
from x_cluster.bodies import SplitMix64
from x_cluster.common import gather_rows
from x_cluster.ops import ClusterOps

# Lane cluster2 (lane/apple-fast-cluster2, 2026-10-02), FAST + Apple, the
# GPU binding only, the build define `-D MOJOLEARN_BISECT_FAST_RESIDENT=1`
# (BISECT_FAST_RESIDENT, default off):
# the centered data and the raw data uploaded ONCE and kept resident. Cause:
# every split re-uploaded its m x d subset for the child scores
# (`ops.put(sub)` below, after `ops.kmeans` had uploaded the same rows; an
# upload of a million floats or more drains the stream), and the final
# inertia uploaded all of x again (`ops.put(x)`). With the switch the scores
# read the resident rows by index (`ops.sqdist_rows`: m ints up, not m d
# floats) and the inertia reuses the resident x. Same bits. What stays:
# the gather and the upload inside `ops.kmeans` (cluster/estimator.mojo
# kmeans_fit takes host rows), listed in docs/apple-fast/ab/cluster2.md.
comptime XC2_FAST = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and TARGET_COLUMN == COLUMN_APPLE
comptime BISECT_FAST_RESIDENT = XC2_FAST and is_defined["MOJOLEARN_BISECT_FAST_RESIDENT"]()


struct BisectTree(Movable):
    """Nodes in creation order; node 0 is the root. `centers` are UNCENTERED
    (the data mean added back), as sklearn compares against in predict."""

    var centers: List[Float32]
    var left: List[Int]
    var right: List[Int]
    var label: List[Int]
    var score: List[Float64]
    var rows: List[List[Int]]

    def __init__(out self):
        self.centers = List[Float32]()
        self.left = List[Int]()
        self.right = List[Int]()
        self.label = List[Int]()
        self.score = List[Float64]()
        self.rows = List[List[Int]]()

    def add(mut self, center: List[Float32], score: Float64, var rows: List[Int]) -> Int:
        for v in center:
            self.centers.append(v)
        self.left.append(-1)
        self.right.append(-1)
        self.label.append(-1)
        self.score.append(score)
        self.rows.append(rows^)
        return len(self.left) - 1

    def leaves(self) -> List[Int]:
        """Depth-first, left before right (`iter_leaves`)."""
        var out = List[Int]()
        var stack = List[Int]()
        stack.append(0)
        while len(stack) > 0:
            var t = stack.pop()
            if self.left[t] < 0:
                out.append(t)
            else:
                stack.append(self.right[t])
                stack.append(self.left[t])
        return out^


def bisect_fit[O: ClusterOps](
    mut ops: O, x: List[Float32], n: Int, d: Int, k: Int, n_init: Int, init: Int,
    max_iter: Int, tol: Float64, seed: UInt64, largest_cluster: Bool,
    mut tree: BisectTree, mut labels: List[Int32], mut centers: List[Float32],
    weights: List[Float32] = List[Float32](),
) raises -> Float64:
    """Returns the inertia against the leaf centers. `weights` (empty: unit)
    are sklearn's sample_weight: the data mean stays unweighted (sklearn's
    `_X_mean`), each 2-means is weighted, and so are the inertia scores and
    the inertia; 'largest_cluster' counts rows."""
    from checks.numerics import identical_mul64

    var weighted = len(weights) > 0
    if k < 1 or k > n:
        raise Error("BisectingKMeans: n_samples=" + String(n) + " should be >= n_clusters=" + String(k))
    # the column means: one ascending Float64 chain per column, the chains
    # advanced row by row (lane/neural-pass108: one pass over the rows in
    # memory order, not d strided passes; each chain's adds are unchanged)
    var acc = List[Float64](length=d, fill=Float64(0))
    for r in range(n):
        var row = r * d
        for f in range(d):
            acc[f] = acc[f] + Float64(x[row + f])
    var mean = List[Float32](capacity=d)
    for f in range(d):
        mean.append(Float32(acc[f] / Float64(n)))
    var xc = List[Float32](length=n * d, fill=Float32(0))
    for r in range(n):
        var row = r * d
        for f in range(d):
            xc[row + f] = ftz(ftz(x[row + f]) - ftz(mean[f]))
    var resident = False
    comptime if BISECT_FAST_RESIDENT:
        resident = ops.fast_device()
    var xcs = -1
    var xs_full = -1
    if resident:
        xcs = ops.put(xc)
        xs_full = ops.put(x)
    var rng = SplitMix64(seed)
    var all_rows = List[Int](capacity=n)
    for r in range(n):
        all_rows.append(r)
    var root_center = List[Float32](length=d, fill=Float32(0))
    _ = tree.add(root_center, Float64(0), all_rows^)
    var kinit = INIT_RANDOM if init == INIT_RANDOM else INIT_KMEANS_PLUS_PLUS
    var xc_s = -1
    for _split in range(k - 1):
        var leaves = tree.leaves()
        var pick = leaves[0]
        for t in range(1, len(leaves)):
            if tree.score[leaves[t]] > tree.score[pick]:
                pick = leaves[t]
        var rows = tree.rows[pick].copy()
        var m = len(rows)
        if m < 2:
            raise Error("BisectingKMeans: a cluster of " + String(m) + " sample cannot be bisected")
        var sub_w = List[Float32]()
        if weighted:
            for r in rows:
                sub_w.append(weights[r])
        var best_c = List[Float32]()
        var best_l = List[Int32]()
        var best_inertia = Float64(0)
        var sub_s: Int
        if weighted:
            var sub = gather_rows(xc, d, rows)
            for it in range(n_init):
                var c = List[Float32]()
                var l = List[Int32]()
                var inertia = ops.kmeans(sub, m, d, 2, max_iter, tol, rng.next() >> 1, 1, kinit, c, l, sub_w)
                if it == 0 or inertia < best_inertia * (1 - 1e-6):
                    best_inertia = inertia
                    best_c = c^
                    best_l = l^
            sub_s = ops.put(sub)
        else:
            # the cluster's rows gathered on the device from the one upload
            # of the centered data (lane/neural-pass108: a host gather and two
            # uploads per split were 6 s of a 13 s istella 1M x 220 fit)
            if xc_s < 0:
                xc_s = ops.put(xc)
            var idx = List[Int32](capacity=m)
            for r in rows:
                idx.append(Int32(r))
            var idx_s = ops.put_i(idx)
            sub_s = ops.empty(m * d)
            ops.gather_rows(xc_s, d, idx_s, m, sub_s)
            for it in range(n_init):
                var c = List[Float32]()
                var l = List[Int32]()
                var inertia = ops.kmeans_rows(sub_s, xc, rows, d, 2, max_iter, tol, rng.next() >> 1, 1, kinit, c, l)
                if it == 0 or inertia < best_inertia * (1 - 1e-6):
                    best_inertia = inertia
                    best_c = c^
                    best_l = l^
        # per-child scores and rows
        var cs = ops.put(best_c)
        var ds = ops.zeros(m * 2)
        if resident:
            var rows32 = List[Int32](capacity=m)
            for r in rows:
                rows32.append(Int32(r))
            var rs_ = ops.put_i(rows32)
            ops.sqdist_rows(xcs, rs_, m, cs, 2, d, ds)
        else:
            var sub_s = ops.put(sub)
            ops.sqdist(sub_s, m, cs, 2, d, ds)
        var dd = ops.get(ds, m * 2)
        var sc = List[Float64](length=2, fill=Float64(0))
        var child_rows = List[List[Int]]()
        child_rows.append(List[Int]())
        child_rows.append(List[Int]())
        for t in range(m):
            var j = Int(best_l[t])
            child_rows[j].append(rows[t])
            if largest_cluster:
                sc[j] = sc[j] + 1
            elif weighted:
                sc[j] = sc[j] + identical_mul64(Float64(sub_w[t]), Float64(dd[t * 2 + j]))
            else:
                sc[j] = sc[j] + Float64(dd[t * 2 + j])
        var ids = List[Int]()
        for j in range(2):
            var cen = List[Float32](capacity=d)
            for f in range(d):
                cen.append(ftz(ftz(best_c[j * d + f]) + ftz(mean[f])))
            ids.append(tree.add(cen, sc[j], child_rows[j].copy()))
        ops.shrink(sub_s)
        tree.left[pick] = ids[0]
        tree.right[pick] = ids[1]
        tree.rows[pick] = List[Int]()
    var leaves = tree.leaves()
    labels = List[Int32](length=n, fill=Int32(-1))
    centers = List[Float32](capacity=k * d)
    for i in range(len(leaves)):
        var t = leaves[i]
        tree.label[t] = i
        for r in tree.rows[t]:
            labels[r] = Int32(i)
        for f in range(d):
            centers.append(tree.centers[t * d + f])
    # inertia against the (uncentered) leaf centers, one Float64 chain
    var xs = xs_full if resident else ops.put(x)
    var cs = ops.put(centers)
    var ds = ops.zeros(n * k)
    ops.sqdist(xs, n, cs, k, d, ds)
    var dd = ops.get(ds, n * k)
    var inertia = Float64(0)
    for r in range(n):
        if weighted:
            inertia = inertia + identical_mul64(Float64(weights[r]), Float64(dd[r * k + Int(labels[r])]))
        else:
            inertia = inertia + Float64(dd[r * k + Int(labels[r])])
    return inertia


def bisect_predict[O: ClusterOps](
    mut ops: O, x: List[Float32], n: Int, d: Int, tree_centers: List[Float32], nodes: List[Int32]
) raises -> List[Int32]:
    var xs = ops.put(x)
    var cs = ops.put(tree_centers)
    var ns = ops.put_i(nodes)
    var ls = ops.zeros_i(n)
    ops.descend(xs, n, d, cs, ns, ls)
    return ops.get_i(ls, n)
