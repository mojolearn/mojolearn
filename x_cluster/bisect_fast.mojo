# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""BisectingKMeans, FAST on Apple: X uploaded from the caller's array and
centered on the device (lane/apple-fast-gap-clus3, 2026-10-03). Switch:
`BISECT_FAST_ZEROCOPY` below, default on in FAST on
Apple (`-D MOJOLEARN_BISECT_FAST_ZEROCOPY_OFF` turns it off); taken by bindings/_mojolearn_x_cluster.mojo for
ENTRY_BISECT. IDENTICAL compiles none of this.

Cause (x_cluster/bisect.mojo `bisect_fit`, board shape Istella 1M x 220
float32, 880 MB): before the first split the host
  * copies X into a fresh list (the binding's `read_f32`: ~21 ms per 64 MB
    of first-touch page faults on Apple, memory metal-transfer-costs-on-apple),
  * folds the column means in one serial pass over 220M values,
  * builds the centered copy `xc` (a second fresh 880 MB list) in another
    serial pass, and uploads it (`ops.put(xc)`, a fresh 880 MB buffer);
and after the last split it uploads X AGAIN (`ops.put(x)`), runs an n x k
`sqdist` and downloads it (32 MB) to fold the inertia on the host.

Here X goes from the caller's pointer into a pooled device buffer (the MBK
pool pattern, lane/apple-fast-gap-cls2), the column means are a grid of
(column, 1024-row chunk) partials folded per column in chunk order, the
centering is a thread per value in place, and the splits read that buffer
(`ops.gather_rows`, as `bisect_fit`'s unweighted path does). The inertia is
the sum of the leaf scores ('biggest_inertia': each leaf's score is its
rows' squared distances to its centered center, the same distances the
final pass took against the uncentered centers), so the second upload, the
n x k pass and its download are gone. Unit weights and 'biggest_inertia'
only; anything else returns False and the binding takes the copying path.
FAST promises quality, not bits: the means are float32 chunk sums (the host
folds float64), so the centered words, and the inertia's order, differ.
"""
from std.gpu import block_idx, thread_idx
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceBuffer

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, ftz, identical_div
from cluster.impl.kmeans_params import INIT_KMEANS_PLUS_PLUS, INIT_RANDOM
from core.device_pool import pool_give, pool_take
from x_cluster.bisect import BisectTree
from x_cluster.bodies import FPtr, SplitMix64
from x_cluster.device_ops import DeviceOps
from x_cluster.out import ClusterOut

comptime BISECT_FAST_ZEROCOPY = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_BISECT_FAST_ZEROCOPY_OFF"]()
)  # FAST + Apple default since the M3 A/B clus3-bisect-zc-istella (n=1):
# bisecting-kmeans istella 2033 -> 848 ms (-58.3%), silhouette .1183 identical;
# -D MOJOLEARN_BISECT_FAST_ZEROCOPY_OFF turns it off (the old -D name is harmless)

comptime XPtr = MutPointer[Float32, MutUntrackedOrigin]
comptime BFZ_TPB = 256
comptime BFZ_CH = 1024
"""Rows per column-sum chunk."""
comptime BFZ_FOLD_TPB = 64
comptime BFZ_CENTER_GRID = 65535


def _bfz_colsum_kernel(x: FPtr, n: Int32, d: Int32, part: FPtr):
    """grid (ceil(d / BFZ_TPB), chunks): thread (f, c) sums rows
    [c * BFZ_CH, ...) of column f in row order (adjacent threads adjacent
    columns: the reads of a row are coalesced)."""
    var f = Int(block_idx.x) * BFZ_TPB + Int(thread_idx.x)
    var c = Int(block_idx.y)
    var D = Int(d)
    if f < D:
        var r0 = c * BFZ_CH
        var r1 = r0 + BFZ_CH
        if r1 > Int(n):
            r1 = Int(n)
        var acc = Float32(0)
        for r in range(r0, r1):
            acc = ftz(acc + ftz(x[r * D + f]))
        part[c * D + f] = acc


def _bfz_mean_kernel(part: FPtr, nch: Int32, d: Int32, n: Int32, mean: FPtr):
    """A thread per column: the chunk partials folded in chunk order, / n."""
    var f = Int(block_idx.x) * BFZ_FOLD_TPB + Int(thread_idx.x)
    var D = Int(d)
    if f < D:
        var acc = Float32(0)
        for c in range(Int(nch)):
            acc = ftz(acc + part[c * D + f])
        mean[f] = ftz(identical_div(acc, Float32(Int(n))))


def _bfz_center_kernel(x: FPtr, total: Int32, d: Int32, mean: FPtr):
    """x[i] = x[i] - mean[i % d] in place, a grid-stride loop over the values
    (`bisect_fit`'s ftz(ftz(x) - ftz(mean)))."""
    var D = Int(d)
    var stride = BFZ_CENTER_GRID * BFZ_TPB
    var i = Int(block_idx.x) * BFZ_TPB + Int(thread_idx.x)
    var T = Int(total)
    while i < T:
        x[i] = ftz(ftz(x[i]) - ftz(mean[i % D]))
        i += stride


def bisect_entry_ptr(
    mut ops: DeviceOps, xp: XPtr, nx: Int, has_w: Bool, ip: List[Int], fp: List[Float64],
    mut out: ClusterOut,
) raises -> Bool:
    """`entries.bisect_entry` with X read from `xp` (nx values, alive for the
    call). False (nothing done) outside the switch, for weights, for
    'largest_cluster' or a shape mismatch: the binding then copies X."""
    comptime if BISECT_FAST_ZEROCOPY:
        var n = ip[0]
        var d = ip[1]
        var k = ip[2]
        var n_init = ip[3]
        var init = ip[4]
        var max_iter = ip[5]
        var seed = UInt64(ip[6])
        var largest_cluster = ip[7] != 0
        var tol = fp[0]
        if has_w or largest_cluster or n * d != nx or n < 1 or d < 1 or nx > 2147483647:
            return False
        if k < 1 or k > n:
            raise Error("BisectingKMeans: n_samples=" + String(n) + " should be >= n_clusters=" + String(k))
        # X from the caller's pointer into a pooled buffer, centered in place
        var xbuf = pool_take["MojoXClusterClus3BisectX"](ops.ctx, nx)
        ops.ctx.enqueue_copy(dst_buf=xbuf, src_ptr=xp)
        ops.f.append(xbuf^)
        var xs = len(ops.f) - 1
        var nch = (n + BFZ_CH - 1) // BFZ_CH
        var part = ops.empty(nch * d)
        var mean_s = ops.empty(d)
        ops.ctx.enqueue_function[_bfz_colsum_kernel](
            ops._fp(xs), Int32(n), Int32(d), ops._fp(part),
            grid_dim=((d + BFZ_TPB - 1) // BFZ_TPB, nch), block_dim=BFZ_TPB,
        )
        ops.ctx.enqueue_function[_bfz_mean_kernel](
            ops._fp(part), Int32(nch), Int32(d), Int32(n), ops._fp(mean_s),
            grid_dim=(d + BFZ_FOLD_TPB - 1) // BFZ_FOLD_TPB, block_dim=BFZ_FOLD_TPB,
        )
        var cgrid = (nx + BFZ_TPB - 1) // BFZ_TPB
        if cgrid > BFZ_CENTER_GRID:
            cgrid = BFZ_CENTER_GRID
        ops.ctx.enqueue_function[_bfz_center_kernel](
            ops._fp(xs), Int32(nx), Int32(d), ops._fp(mean_s), grid_dim=cgrid, block_dim=BFZ_TPB,
        )
        var mean = ops.get(mean_s, d)
        ops.shrink(part)

        # bisect_fit's split loop, unweighted path, on the resident centered X
        var tree = BisectTree()
        var all_rows = List[Int](capacity=n)
        for r in range(n):
            all_rows.append(r)
        var root_center = List[Float32](length=d, fill=Float32(0))
        _ = tree.add(root_center, Float64(0), all_rows^)
        var kinit = INIT_RANDOM if init == INIT_RANDOM else INIT_KMEANS_PLUS_PLUS
        var rng = SplitMix64(seed)
        var host_dummy = List[Float32](length=1, fill=Float32(0))  # kmeans_fit_rows does not read it
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
            var idx = List[Int32](capacity=m)
            for r in rows:
                idx.append(Int32(r))
            var idx_s = ops.put_i(idx)
            var sub_s = ops.empty(m * d)
            ops.gather_rows(xs, d, idx_s, m, sub_s)
            var best_c = List[Float32]()
            var best_l = List[Int32]()
            var best_inertia = Float64(0)
            for it in range(n_init):
                var c = List[Float32]()
                var l = List[Int32]()
                var inertia = ops.kmeans_rows(
                    sub_s, host_dummy, rows, d, 2, max_iter, tol, rng.next() >> 1, 1, kinit, c, l
                )
                if it == 0 or inertia < best_inertia * (1 - 1e-6):
                    best_inertia = inertia
                    best_c = c^
                    best_l = l^
            var cs = ops.put(best_c)
            var ds = ops.zeros(m * 2)
            ops.sqdist(sub_s, m, cs, 2, d, ds)
            var dd = ops.get(ds, m * 2)
            var sc = List[Float64](length=2, fill=Float64(0))
            var child_rows = List[List[Int]]()
            child_rows.append(List[Int]())
            child_rows.append(List[Int]())
            for t in range(m):
                var j = Int(best_l[t])
                child_rows[j].append(rows[t])
                sc[j] = sc[j] + Float64(dd[t * 2 + j])
            var ids = List[Int]()
            for j in range(2):
                var cen = List[Float32](capacity=d)
                for f in range(d):
                    cen.append(ftz(ftz(best_c[j * d + f]) + ftz(mean[f])))
                ids.append(tree.add(cen, sc[j], child_rows[j].copy()))
            ops.shrink(sub_s)
            ops.shrink(ds)
            tree.left[pick] = ids[0]
            tree.right[pick] = ids[1]
            tree.rows[pick] = List[Int]()
        # every launch that reads X has been drained by the last `get`
        ops.ctx.synchronize()
        pool_give["MojoXClusterClus3BisectX"](ops.f.pop(xs))
        _ = host_dummy^

        var leaves = tree.leaves()
        var labels = List[Int32](length=n, fill=Int32(-1))
        var centers = List[Float32](capacity=k * d)
        var inertia = Float64(0)
        for i in range(len(leaves)):
            var t = leaves[i]
            tree.label[t] = i
            inertia = inertia + tree.score[t]
            for r in tree.rows[t]:
                labels[r] = Int32(i)
            for f in range(d):
                centers.append(tree.centers[t * d + f])
        var nodes = List[Int32]()
        for t in range(len(tree.left)):
            nodes.append(Int32(tree.left[t]))
            nodes.append(Int32(tree.right[t]))
            nodes.append(Int32(tree.label[t]))
        out.f.append(centers^)
        out.f.append(tree.centers.copy())
        out.i.append(labels^)
        out.i.append(nodes^)
        out.s.append(inertia)
        return True
    return False
