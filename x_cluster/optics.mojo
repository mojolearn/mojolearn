# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""OPTICS (lane/algos-cluster). Reference: scikit-learn
`sklearn/cluster/_optics.py` (`compute_optics_graph` :459-670,
`_set_reach_dist` :672-725, `cluster_optics_dbscan` :727-770,
`cluster_optics_xi` :811-920, `_extend_region`, `_update_filter_sdas`,
`_correct_predecessor`, `_xi_cluster` :1021-1175, `_extract_xi_labels`).

The n x n euclidean distances and the core distances (the `min_samples`-th
smallest of each row, the row itself included: `bodies.kth_smallest_row`)
are the device's, and so is THE ORDERING (lane cgr2-cluster,
`ClusterOps.optics_order`), one step after another as the reference's: the
next point is the unprocessed one with the lowest reachability, THE LOWEST
INDEX ON A TIE (`np.argmin` over the unprocessed indices in order), a
min-reduction over the rows; its unprocessed neighbors within `max_eps` get
`max(dist, core)` when that is strictly lower, one thread a row. The dbscan
extraction is the device's (`optics_dbscan`). The xi extraction is the
reference's sequential steep-region walk (a data-dependent state machine over
the n-length plot, run once on the fitted reachability the fit returns),
the ratios in Float64 from the Float32 plot. sklearn's `np.around(...,
decimals=precision)` of the core and reach distances is not carried
(NOT_IMPLEMENTED.tsv)."""
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, identical_mul64
from x_cluster.bodies import FPtr, IPtr
from x_cluster.ops import ClusterOps

# Lane cluster-apple3, FAST only, OPT-IN while unproven.
# `-D MOJOLEARN_OPTICS_SIMD=1`: the ordering loop's two row walks as vector
# min, compare and select (the same point and the same updates at every
# step, so the same ordering, reachability and predecessors).
# `-D MOJOLEARN_XC_ALLOC=1`: a slot a distance kernel fills completely is
# not zeroed first (`ops.alloc`; OPTICS, MeanShift, AffinityPropagation,
# agglomerative).
comptime OPTICS_SIMD = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and is_defined["MOJOLEARN_OPTICS_SIMD"]()
comptime XC_ALLOC = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and is_defined["MOJOLEARN_XC_ALLOC"]()
# The opt-in host distance rows (`-D MOJOLEARN_OPTICS_HOSTROWS`) were removed
# (hr-optin-flags): the loop reads the device's n x n distances.
comptime _OW = 8

# lane/apple-fast-optics2 (2026-10-03), FAST on Apple ONLY, every switch a
# build define, default OFF; IDENTICAL, the host binding and a FAST build
# without a define compile main's code (docs/apple-fast/notes/optics2.md
# has the profile, docs/apple-fast/ab/optics2.md the experiments). The
# ordering is `DeviceOps.optics_fast` (x_cluster/optics_fast.mojo kernels):
# `MOJOLEARN_OPTICS_STEP_BATCH`: one threadgroup runs 512 ordering steps per
#   launch (n/512 launches instead of 2n);
# `MOJOLEARN_OPTICS_FRONTIER_DEVICE`: one fused launch per step (relax and
#   emit the next step's block minima; STEP_BATCH wins when both are set);
# `MOJOLEARN_OPTICS_CORE_SQ`: no sqrt pass over the n x n matrix (kth on the
#   squared cells, sqrt of the n core values, sqrt at use in the relaxation);
# `MOJOLEARN_OPTICS_LIVEBUF`: no memset of kernel-filled slots, the four
#   outputs in two paired buffers, one readback synchronize;
# `MOJOLEARN_OPTICS2_ALL`: STEP_BATCH + CORE_SQ + LIVEBUF.
# Same picks (lowest reachability, lowest index on a tie), same relaxation,
# same words as main's `optics_order` and the host column.
comptime OPTICS_FAST_APPLE = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
comptime OPTICS2_ALL = OPTICS_FAST_APPLE and is_defined["MOJOLEARN_OPTICS2_ALL"]()
comptime OPTICS_STEP_BATCH = OPTICS_FAST_APPLE and (is_defined["MOJOLEARN_OPTICS_STEP_BATCH"]() or OPTICS2_ALL)
comptime OPTICS_FRONTIER_DEVICE = (
    OPTICS_FAST_APPLE and is_defined["MOJOLEARN_OPTICS_FRONTIER_DEVICE"]() and not OPTICS_STEP_BATCH
)
comptime OPTICS_CORE_SQ = OPTICS_FAST_APPLE and (is_defined["MOJOLEARN_OPTICS_CORE_SQ"]() or OPTICS2_ALL)
comptime OPTICS_LIVEBUF = OPTICS_FAST_APPLE and (is_defined["MOJOLEARN_OPTICS_LIVEBUF"]() or OPTICS2_ALL)
comptime OPTICS_FAST_ANY = OPTICS_STEP_BATCH or OPTICS_FRONTIER_DEVICE or OPTICS_CORE_SQ or OPTICS_LIVEBUF


def dist_slot[O: ClusterOps](mut ops: O, n: Int) raises -> Int:
    """The slot of an n-value matrix a distance kernel is about to fill."""
    comptime if XC_ALLOC or OPTICS_LIVEBUF:
        if ops.fast_device():
            return ops.alloc(n)
    return ops.zeros(n)


@always_inline
def _relax(
    dd: SIMD[DType.float32, _OW], j: Int, cpv: SIMD[DType.float32, _OW], ev: SIMD[DType.float32, _OW],
    zero: SIMD[DType.float32, _OW], pv: SIMD[DType.int32, _OW], rp: FPtr, mp: FPtr, qp: IPtr,
):
    """Rows j .. j + 7 against the point's distances `dd`: an unprocessed row
    within `max_eps` takes `max(dd, core)` when that is strictly lower."""
    var rd = dd.gt(cpv).select(dd, cpv)
    var rr = rp.load[width=_OW](j)
    var take = dd.le(ev) & rd.lt(rr) & mp.load[width=_OW](j).lt(zero)
    (rp + j).store(take.select(rd, rr))
    (qp + j).store(take.select(pv, qp.load[width=_OW](j)))


@always_inline
def _relax1(dd: Float32, j: Int, cp: Float32, max_eps: Float32, point: Int, rp: FPtr, mp: FPtr, qp: IPtr):
    if mp[j] < Float32(0) and dd <= max_eps:
        var rd = dd if dd > cp else cp
        if rd < rp[j]:
            rp[j] = rd
            qp[j] = Int32(point)


def _order_simd(
    dist: List[Float32], core: List[Float32], n: Int, max_eps: Float32,
    mut ordering: List[Int], mut reach: List[Float32], mut pred: List[Int],
):
    """The ordering loop of `optics_graph`, the same decisions by vectors.

    `pm[j]` is -inf while row j is unprocessed and +inf after, so
    `max(reach, pm)` is the row's reachability or +inf, and its minimum `m`
    is the loop's `best`. The point is the FIRST unprocessed row whose
    reachability equals `m` (every unprocessed row when m is +inf), which is
    what the scalar walk's strict `<` keeps. No reachability is a NaN: a
    value enters `reach` only through `dd <= max_eps`."""
    var inf = Float32.MAX * Float32(2)
    reach = List[Float32](length=n, fill=inf)
    var pm = List[Float32](length=n, fill=-inf)
    var pr = List[Int32](length=n, fill=Int32(-1))
    ordering = List[Int](capacity=n)
    var rp: FPtr = reach.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var mp: FPtr = pm.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var qp: IPtr = pr.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var dp = dist.unsafe_ptr()
    var zero = SIMD[DType.float32, _OW](0)
    var ev = SIMD[DType.float32, _OW](max_eps)
    for _step in range(n):
        var vmin = SIMD[DType.float32, _OW](inf)
        var j = 0
        while j + _OW <= n:
            vmin = min(vmin, max(rp.load[width=_OW](j), mp.load[width=_OW](j)))
            j += _OW
        var m = vmin.reduce_min()
        while j < n:
            var v = rp[j] if mp[j] < Float32(0) else inf
            if v < m:
                m = v
            j += 1
        var mv = SIMD[DType.float32, _OW](m)
        j = 0
        while j + _OW <= n:
            var hit = rp.load[width=_OW](j).eq(mv) & mp.load[width=_OW](j).lt(zero)
            if hit.reduce_or():
                break
            j += _OW
        var point = -1
        while j < n:
            if mp[j] < Float32(0) and rp[j] == m:
                point = j
                break
            j += 1
        if point < 0:
            # unreachable while no reachability is a NaN; never index with -1
            for q in range(n):
                if mp[q] < Float32(0):
                    point = q
                    break
        if point < 0:
            break
        mp[point] = inf
        ordering.append(point)
        if core[point] != inf:
            var cp = core[point]
            var cpv = SIMD[DType.float32, _OW](cp)
            var pv = SIMD[DType.int32, _OW](Int32(point))
            var row = dp + point * n
            j = 0
            while j + _OW <= n:
                _relax(row.load[width=_OW](j), j, cpv, ev, zero, pv, rp, mp, qp)
                j += _OW
            while j < n:
                _relax1(row[j], j, cp, max_eps, point, rp, mp, qp)
                j += 1
    pred = List[Int](capacity=n)
    for i in range(n):
        pred.append(Int(pr[i]))
    _ = pm^
    _ = pr^


def optics_graph[O: ClusterOps](
    mut ops: O, x: List[Float32], n: Int, d: Int, min_samples: Int, max_eps: Float32,
    mut ordering: List[Int], mut core: List[Float32], mut reach: List[Float32], mut pred: List[Int],
    metric: Int = -1, p: Float32 = Float32(2),
) raises:
    """metric -1: euclidean through the squared distance (the recorded
    default); 0-4 the `bodies.pdist_cell` metrics; 5 precomputed (`x` is the
    n x n distance matrix, negatives refused)."""
    var inf = Float32.MAX * Float32(2)
    var xs = ops.put(x)
    var dm: Int
    if metric == 5:
        if not ops.check_nonneg(xs, n * n):
            raise Error("OPTICS: a precomputed distance matrix must be non-negative")
        dm = xs
    elif metric >= 0:
        dm = dist_slot(ops, n * n)
        ops.pdist(xs, n, xs, n, d, metric, p, dm)
    else:
        dm = dist_slot(ops, n * n)
        ops.sqdist(xs, n, xs, n, d, dm)
        comptime if not OPTICS_CORE_SQ:
            ops.sqrt(dm, n * n)
        # OPTICS_CORE_SQ: the squared cells stay; `kth` below reads them (sqrt
        # is monotone: the k-th smallest root is the root of the k-th smallest
        # square), the n core values are rooted, the relaxation roots at use
    var cs: Int
    comptime if OPTICS_LIVEBUF:
        cs = ops.alloc(n)  # `kth` writes every row
    else:
        cs = ops.zeros(n)
    ops.kth(dm, n, n, min_samples, cs)
    comptime if OPTICS_CORE_SQ:
        if metric < 0:
            ops.sqrt(cs, n)
    comptime if OPTICS_FAST_ANY:
        var oi2 = List[Int32]()
        var pi2 = List[Int32]()
        if ops.optics_fast(dm, cs, n, max_eps, OPTICS_CORE_SQ and metric < 0, oi2, reach, core, pi2):
            ordering = List[Int](capacity=n)
            pred = List[Int](capacity=n)
            for q in range(n):
                ordering.append(Int(oi2[q]))
                pred.append(Int(pi2[q]))
            return
    comptime if OPTICS_SIMD:
        if n >= _OW:
            core = ops.get(cs, n)
            for i in range(n):
                if core[i] > max_eps:
                    core[i] = inf
            var dist = ops.get(dm, n * n)
            _order_simd(dist, core, n, max_eps, ordering, reach, pred)
            return
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


def _extend_region(steep: List[Bool], xward: List[Bool], start: Int, min_samples: Int) -> Int:
    var n = len(steep)
    var non_xward = 0
    var index = start
    var end = start
    while index < n:
        if steep[index]:
            non_xward = 0
            end = index
        elif not xward[index]:
            non_xward += 1
            if non_xward > min_samples:
                break
        else:
            return end
        index += 1
    return end


@fieldwise_init
struct _Sda(Copyable, Movable):
    var start: Int
    var end: Int
    var mib: Float64


def _update_filter_sdas(sdas: List[_Sda], mib: Float64, xc: Float64, plot: List[Float64]) -> List[_Sda]:
    var out = List[_Sda]()
    if mib == Float64.MAX * 2:
        return out^
    for s in sdas:
        if mib <= identical_mul64(plot[s.start], xc):
            var t = s.copy()
            if mib > t.mib:
                t.mib = mib
            out.append(t^)
    return out^


def optics_xi_clusters(
    reach: List[Float32], pred: List[Int], ordering: List[Int], xi: Float64,
    min_samples: Int, min_cluster_size: Int, predecessor_correction: Bool,
) -> List[Int]:
    """`_xi_cluster`: the (start, end) pairs, flattened, in the reference's order."""
    var n = len(ordering)
    var inf = Float64.MAX * 2
    var plot = List[Float64](capacity=n + 1)
    var pplot = List[Int](capacity=n)
    for q in range(n):
        var r = reach[ordering[q]]
        plot.append(inf if r == Float32.MAX * Float32(2) else Float64(r))
        pplot.append(pred[ordering[q]])
    plot.append(inf)
    var xc = 1 - xi
    var inv_xc = 1 / xc
    var steep_up = List[Bool](capacity=n)
    var steep_down = List[Bool](capacity=n)
    var down = List[Bool](capacity=n)
    var up = List[Bool](capacity=n)
    for q in range(n):
        var ratio = plot[q] / plot[q + 1]  # inf / inf is a NaN here, which compares false
        steep_up.append(ratio <= xc)
        steep_down.append(ratio >= inv_xc)
        down.append(ratio > 1)
        up.append(ratio < 1)
    var sdas = List[_Sda]()
    var clusters = List[Int]()
    var index = 0
    var mib = Float64(0)
    for steep_index in range(n):
        if not (steep_up[steep_index] or steep_down[steep_index]):
            continue
        if steep_index < index:
            continue
        for q in range(index, steep_index + 1):
            if plot[q] > mib:
                mib = plot[q]
        if steep_down[steep_index]:
            sdas = _update_filter_sdas(sdas, mib, xc, plot)
            var d_end = _extend_region(steep_down, up, steep_index, min_samples)
            sdas.append(_Sda(steep_index, d_end, Float64(0)))
            index = d_end + 1
            mib = plot[index]
        else:
            sdas = _update_filter_sdas(sdas, mib, xc, plot)
            var u_start = steep_index
            var u_end = _extend_region(steep_up, down, u_start, min_samples)
            index = u_end + 1
            mib = plot[index]
            var u_clusters = List[Int]()
            for s in sdas:
                var c_start = s.start
                var c_end = u_end
                if identical_mul64(plot[c_end + 1], xc) < s.mib:
                    continue
                var d_max = plot[s.start]
                if identical_mul64(d_max, xc) >= plot[c_end + 1]:
                    while plot[c_start + 1] > plot[c_end + 1] and c_start < s.end:
                        c_start += 1
                elif identical_mul64(plot[c_end + 1], xc) >= d_max:
                    while plot[c_end - 1] > d_max and c_end > u_start:
                        c_end -= 1
                if predecessor_correction:
                    var ok = False
                    while c_start < c_end:
                        if plot[c_start] > plot[c_end]:
                            ok = True
                            break
                        var p_e = pplot[c_end]
                        var hit = False
                        for i in range(c_start, c_end):
                            if p_e == ordering[i]:
                                hit = True
                                break
                        if hit:
                            ok = True
                            break
                        c_end -= 1
                    if not ok:
                        continue
                if c_end - c_start + 1 < min_cluster_size:
                    continue
                if c_start > s.end:
                    continue
                if c_end < u_start:
                    continue
                u_clusters.append(c_start)
                u_clusters.append(c_end)
            var m = len(u_clusters) // 2
            for q in range(m):
                clusters.append(u_clusters[(m - 1 - q) * 2])
                clusters.append(u_clusters[(m - 1 - q) * 2 + 1])
    return clusters^


def optics_xi_labels(ordering: List[Int], clusters: List[Int]) -> List[Int32]:
    """`_extract_xi_labels`."""
    var n = len(ordering)
    var lab = List[Int32](length=n, fill=Int32(-1))
    var label = 0
    for c in range(len(clusters) // 2):
        var a = clusters[c * 2]
        var b = clusters[c * 2 + 1]
        var free = True
        for q in range(a, b + 1):
            if lab[q] != Int32(-1):
                free = False
                break
        if free:
            for q in range(a, b + 1):
                lab[q] = Int32(label)
            label += 1
    var out = List[Int32](length=n, fill=Int32(-1))
    for q in range(n):
        out[ordering[q]] = lab[q]
    return out^
