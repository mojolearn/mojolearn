# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""OPTICS (lane/algos-cluster). Reference: scikit-learn
`sklearn/cluster/_optics.py` (`compute_optics_graph` :459-670,
`_set_reach_dist` :672-725, `cluster_optics_dbscan` :727-770,
`cluster_optics_xi` :811-920, `_extend_region`, `_update_filter_sdas`,
`_correct_predecessor`, `_xi_cluster` :1021-1175, `_extract_xi_labels`).

The n x n euclidean distances and the core distances (the `min_samples`-th
smallest of each row, the row itself included: `ops.kth`)
are the device's, and so is THE ORDERING (lane cgr2-cluster,
`ClusterOps.optics_order`), one step after another as the reference's: the
next point is the unprocessed one with the lowest reachability, THE LOWEST
INDEX ON A TIE (`np.argmin` over the unprocessed indices in order), a
min-reduction over the rows; its unprocessed neighbors within `max_eps` get
`max(dist, core)` when that is strictly lower, one thread a row. The dbscan
extraction is the device's (`optics_dbscan`). So is the xi extraction
(lane cgr4-device-optim-optics, `ops.optics_xi`): the reference's
steep-region walk as parallel scans, sparse tables and pointer doubling over
the plot, no serial walk (`x_cluster/optics_xi_cells.mojo` has the
formulation; xc in Float32 with exact product tests, a bit change against the
old Float64 ratios at ties only). sklearn's `np.around(...,
decimals=precision)` of the core and reach distances is not carried
(NOT_IMPLEMENTED.tsv)."""
from checks.kernel_matrix import COLUMN_APPLE, TARGET_COLUMN
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from x_cluster.ops import ClusterOps

# TOMBSTONE: MOJOLEARN_OPTICS_SIMD (unmeasured FAST opt-in arm: the ordering loop's row walks as host vectors, `_order_simd`) deleted 2026-11-15 by lane/expdel2-15-cluster; code recoverable at c699edc9ce.
# Restore: git apply experiments/removed/MOJOLEARN_OPTICS_SIMD.patch; record in docs/TOMBSTONES.md.
# TOMBSTONE: MOJOLEARN_XC_ALLOC (unmeasured FAST opt-in arm: a distance slot not zeroed before its kernel fills it) deleted 2026-11-15 by lane/expdel2-15-cluster; code recoverable at 545ad31a18.
# Restore: git apply experiments/removed/MOJOLEARN_XC_ALLOC.patch; record in docs/TOMBSTONES.md.
# The opt-in host distance rows (`-D MOJOLEARN_OPTICS_HOSTROWS`) were removed
# (hr-optin-flags): the loop reads the device's n x n distances.
# Lane cluster2 (lane/apple-fast-cluster2, 2026-10-02), FAST + Apple, the
# GPU binding only, OPTICS_FAST_DEVICE_ORDER (DEFAULT ON since the M3 A/B:
# optics istella 432.7 -> 260.7 ms, n=1, quality identical; the old `-D`
# name is harmless):
# the ordering loop as n + 1 grid launches (`ops.optics_order_fast`,
# device_ops.mojo) in place of main's 2n (`ops.optics_order`).
# Each step folds the previous launch's partial keys itself, so the
# separate per-step partial-scan launch goes. Same picks, same reachability
# and predecessors (compares and selects only).
comptime XC2_FAST = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and TARGET_COLUMN == COLUMN_APPLE
# TOMBSTONE: MOJOLEARN_OPTICS_FAST_DEVICE_ORDER_OFF (old path beside the FAST+Apple default, EXP:516; non-owner site) deleted 2026-11-15 by lane/expdel2-15-cluster; code recoverable at 75a1700222.
# Restore: git apply experiments/removed/MOJOLEARN_OPTICS_FAST_DEVICE_ORDER_OFF.expdel2-15-cluster.patch; record in docs/TOMBSTONES.md.
comptime OPTICS_FAST_DEVICE_ORDER = XC2_FAST


def dist_slot[O: ClusterOps](mut ops: O, n: Int) raises -> Int:
    """The slot of an n-value matrix a distance kernel is about to fill."""
    # TOMBSTONE: MOJOLEARN_XC_ALLOC (unmeasured FAST opt-in arm: a distance slot not zeroed before its kernel fills it) deleted 2026-11-15 by lane/expdel2-15-cluster; code recoverable at 545ad31a18.
    # Restore: git apply experiments/removed/MOJOLEARN_XC_ALLOC.patch; record in docs/TOMBSTONES.md.
    return ops.zeros(n)


def optics_graph[O: ClusterOps](
    mut ops: O, x: List[Float32], n: Int, d: Int, min_samples: Int, max_eps: Float32,
    mut ordering: List[Int], mut core: List[Float32], mut reach: List[Float32], mut pred: List[Int],
    mut slots: List[Int], metric: Int = -1, p: Float32 = Float32(2), x_slot: Int = -1,
) raises:
    """metric -1: euclidean through the squared distance (the recorded
    default); 0-4 the `bodies.pdist_cell` metrics; 5 precomputed (`x` is the
    n x n distance matrix, negatives refused). `slots` gets the int slot of
    the ordering, the float slot of the reachability and the int slot of the
    predecessors (resident for `optics_xi_ops`)."""
    var inf = Float32.MAX * Float32(2)
    # `x_slot`: X already in a float slot of `ops` (THE FOLD X WORD, lane vdf-scope-cluster6); the slot is this
    # call's own copy, so the precomputed flush may write it.
    var xs = ops.put(x) if x_slot < 0 else x_slot
    var dm: Int
    if metric == 5:
        if not ops.check_nonneg(xs, n * n):
            raise Error("OPTICS: a precomputed distance matrix must be non-negative")
        # Flushed once on the device, as every computed distance already is:
        # the ordering compares reachabilities as floats, and the Apple GPU
        # reads a subnormal operand as zero where every other target does
        # not (x-cluster-optics-metrics/denormal pre_order, 0.8.36 record).
        ops.flush(xs, n * n)
        dm = xs
    elif metric >= 0:
        dm = dist_slot(ops, n * n)
        ops.pdist(xs, n, xs, n, d, metric, p, dm)
    else:
        dm = dist_slot(ops, n * n)
        ops.sqdist(xs, n, xs, n, d, dm)
        ops.sqrt(dm, n * n)
    var cs = ops.zeros(n)
    ops.kth(dm, n, n, min_samples, cs)
    comptime if OPTICS_FAST_DEVICE_ORDER:
        if ops.fast_device():
            # lane cluster2: the device's raw core distances (`cs`), the
            # kernel's test `core <= max_eps and core != inf` is the clamp below
            var fos = ops.zeros_i(n)
            var frs = ops.zeros(n)
            var fps = ops.zeros_i(n)
            var fqs = ops.zeros_i(n)
            ops.optics_order_fast(dm, cs, n, max_eps, fos, frs, fps, fqs)
            var o32 = List[Int32]()
            ops.get_if(fos, n, frs, n, o32, reach)
            core = ops.get(cs, n)
            for i in range(n):
                if core[i] > max_eps:
                    core[i] = inf
            var p32 = ops.get_i(fps, n)
            ordering = List[Int](capacity=n)
            pred = List[Int](capacity=n)
            for i in range(n):
                ordering.append(Int(o32[i]))
                pred.append(Int(p32[i]))
            slots = [fos, frs, fps]
            return
    # TOMBSTONE: MOJOLEARN_OPTICS_SIMD (unmeasured FAST opt-in arm: the ordering loop's row walks as host vectors, `_order_simd`) deleted 2026-11-15 by lane/expdel2-15-cluster; code recoverable at c699edc9ce.
    # Restore: git apply experiments/removed/MOJOLEARN_OPTICS_SIMD.patch; record in docs/TOMBSTONES.md.
    # THE ORDERING ON THE DEVICE (lane cgr2-cluster): the distances stay
    # resident; each step is a min-reduction of (reachability, index) over
    # the unprocessed rows and one parallel relaxation (`ops.optics_order`;
    # the host column walks the same steps).
    var os_ = ops.zeros_i(n)
    var rs = ops.zeros(n)
    var ps = ops.zeros_i(n)
    ops.optics_order(dm, cs, n, max_eps, os_, rs, ps)
    var oi = List[Int32]()
    var pi = List[Int32]()
    ops.get_if(os_, n, rs, n, oi, reach)
    var g = ops.gets([cs], [n])
    core = g[0].copy()
    pi = ops.get_i(ps, n)
    ordering = List[Int](capacity=n)
    pred = List[Int](capacity=n)
    for q in range(n):
        ordering.append(Int(oi[q]))
        pred.append(Int(pi[q]))
    slots = [os_, rs, ps]


def optics_dbscan_ops[O: ClusterOps](
    mut ops: O, ordering: List[Int], reach: List[Float32], core: List[Float32], eps: Float32
) raises -> List[Int32]:
    """`cluster_optics_dbscan` by the device (a flag per ordered row and a
    prefix count; `ops.optics_dbscan`)."""
    var n = len(core)
    var oi = List[Int32](capacity=n)
    for q in range(n):
        oi.append(Int32(ordering[q]))
    var os_ = ops.put_i(oi)
    var rs = ops.put(reach)
    var cs = ops.put(core)
    var ls = ops.zeros_i(n)
    ops.optics_dbscan(os_, rs, cs, n, eps, ls)
    return ops.get_i(ls, n)


def optics_xi_ops[O: ClusterOps](
    mut ops: O, slots: List[Int], n: Int, xi: Float64, min_samples: Int, min_cluster_size: Int,
    predecessor_correction: Bool, mut labels: List[Int32],
) raises -> List[Int32]:
    """`cluster_optics_xi` by the device (`ops.optics_xi`; the host column
    runs the same cells): the clusters (start, end) flattened, in the
    reference's order, and the labels. xc = 1 - xi, rounded to Float32 once."""
    var xc = Float32(Float64(1) - xi)
    var ls = ops.zeros_i(n)
    var cl = ops.optics_xi(slots[0], slots[1], slots[2], n, xc, min_samples, min_cluster_size, predecessor_correction, ls)
    labels = ops.get_i(ls, n)
    return cl^
