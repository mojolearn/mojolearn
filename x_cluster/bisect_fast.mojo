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
from x_cluster.bodies import FPtr, IPtr, SplitMix64
from x_cluster.device_ops import DeviceOps
from x_cluster.out import ClusterOut
from x_cluster.post_bodies import FM_VAL

comptime BISECT_FAST_ZEROCOPY = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_BISECT_FAST_ZEROCOPY_OFF"]()
)  # FAST + Apple default since the M3 A/B clus3-bisect-zc-istella (n=1):
# bisecting-kmeans istella 2033 -> 848 ms (-58.3%), silhouette .1183 identical;
# -D MOJOLEARN_BISECT_FAST_ZEROCOPY_OFF turns it off (the old -D name is harmless)

comptime XPtr = MutPointer[Float32, MutUntrackedOrigin]
comptime UPtr32 = MutPointer[UInt32, MutAnyOrigin]
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


def _bfz_flag_kernel(node: IPtr, n: Int32, pick: Int32, flags: IPtr):
    """flags[r] = 1 when row r is in leaf `pick`."""
    var r = Int(block_idx.x) * BFZ_TPB + Int(thread_idx.x)
    if r < Int(n):
        flags[r] = Int32(1) if node[r] == pick else Int32(0)


def _bfz_ids_kernel(flags: IPtr, pos: IPtr, n: Int32, ids: IPtr):
    """ids[pos[r]] = r for the flagged rows (`pos` the exclusive scan): the
    leaf's rows in row order."""
    var r = Int(block_idx.x) * BFZ_TPB + Int(thread_idx.x)
    if r < Int(n) and flags[r] != Int32(0):
        ids[Int(pos[r])] = Int32(r)


def _bfz_split_kernel(
    ids: IPtr, lab: UPtr32, m: Int32, node: IPtr, lid: Int32, rid: Int32, dd: FPtr, e0: FPtr, e1: FPtr, one: IPtr,
):
    """Row t of the split (row ids[t] of X) to child `lid` (label 0) or
    `rid` (label 1); e_j[t] its squared distance to center j when its label
    is j, else 0 (the children's scores are the folds of e0 / e1); one[t]
    the label-1 flag (its scan total is the right child's size)."""
    var t = Int(block_idx.x) * BFZ_TPB + Int(thread_idx.x)
    if t < Int(m):
        var j = Int(lab[t])
        node[Int(ids[t])] = rid if j == 1 else lid
        e0[t] = dd[t * 2] if j == 0 else Float32(0)
        e1[t] = dd[t * 2 + 1] if j == 1 else Float32(0)
        one[t] = Int32(j)


def _bfz_uncenter_kernel(c: FPtr, mean: FPtr, d: Int32, total: Int32, dst: FPtr):
    """dst[q] = ftz(ftz(c[q]) + ftz(mean[q % d])) (`bisect_fit`'s uncentered
    child centers), a thread per word."""
    var q = Int(block_idx.x) * BFZ_TPB + Int(thread_idx.x)
    if q < Int(total):
        dst[q] = ftz(ftz(c[q]) + ftz(mean[q % Int(d)]))


def _bfz_label_kernel(node: IPtr, n: Int32, lt: IPtr, labels: IPtr):
    """labels[r] = the leaf index of row r's leaf."""
    var r = Int(block_idx.x) * BFZ_TPB + Int(thread_idx.x)
    if r < Int(n):
        labels[r] = lt[Int(node[r])]


def _release_scan(mut ops: DeviceOps, prefix: Int) raises:
    """Shrink the int slots of one drained `DeviceOps._scan` (its partials,
    prefix and total slots are consecutive, prefix in the middle)."""
    for q in range(prefix - 1, prefix + 2):  # small-loop(prefix: three scan slots): fixed three slot releases
        ops.i[q] = ops.ctx.enqueue_create_buffer[DType.int32](1)


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
        ops.shrink(part)

        # bisect_fit's split loop, unweighted path, on the resident centered
        # X. Lane cpu3-core: row membership stays on the device (`node_s`,
        # each row's leaf), so a split's rows are a device compaction (in
        # row order, as the host lists were), the 2-means labels stay on the
        # device, the children's scores are float-float folds and the
        # uncentered centers a device add; the host keeps the tree's k-sized
        # metadata only.
        var tree = BisectTree()
        var sizes = List[Int]()
        var nmax = 2 * k - 1
        var cent_all = ops.zeros(nmax * d)  # every node's uncentered center (root: zeros, as bisect_fit)
        var node_s = ops.zeros_i(n)  # every row starts in the root
        _ = tree.add(List[Float32](), Float64(0), List[Int]())
        sizes.append(n)
        var kinit = INIT_RANDOM if init == INIT_RANDOM else INIT_KMEANS_PLUS_PLUS
        var rng = SplitMix64(seed)
        var flags = ops.zeros_i(n)
        var ids_s = ops.zeros_i(n)
        var e0 = ops.zeros(n)
        var e1 = ops.zeros(n)
        var one_s = ops.zeros_i(n)
        var ngrid = (n + BFZ_TPB - 1) // BFZ_TPB
        for _split in range(k - 1):  # small-loop(k: bisections): k - 1 splits, each launches device steps
            var leaves = tree.leaves()
            var pick = leaves[0]
            for t in range(1, len(leaves)):  # small-loop(leaves: tree leaves): at most k leaves of tree metadata
                if tree.score[leaves[t]] > tree.score[pick]:
                    pick = leaves[t]
            var m = sizes[pick]
            if m < 2:
                raise Error("BisectingKMeans: a cluster of " + String(m) + " sample cannot be bisected")
            # the picked leaf's rows, ascending, compacted on the device
            ops.ctx.enqueue_function[_bfz_flag_kernel](
                ops._ip(node_s), Int32(n), Int32(pick), ops._ip(flags), grid_dim=ngrid, block_dim=BFZ_TPB,
            )
            var sc_pos = ops._scan(ops._ip(flags), n)
            ops.ctx.enqueue_function[_bfz_ids_kernel](
                ops._ip(flags), ops._ip(sc_pos[0]), Int32(n), ops._ip(ids_s), grid_dim=ngrid, block_dim=BFZ_TPB,
            )
            var sub_s = ops.empty(m * d)
            ops.gather_rows(xs, d, ids_s, m, sub_s)
            var best_c = List[Float32]()
            var best_lab = ops.ctx.enqueue_create_buffer[DType.uint32](m)
            var best_inertia = Float64(0)
            for it in range(n_init):  # small-loop(n_init: restarts): each restart is a device fit
                var c = List[Float32]()
                var lab = ops.ctx.enqueue_create_buffer[DType.uint32](m)
                var inertia = ops.kmeans_sub(sub_s, m, d, 2, max_iter, tol, rng.next() >> 1, 1, kinit, c, lab)
                if it == 0 or inertia < best_inertia * (1 - 1e-6):
                    best_inertia = inertia
                    best_c = c^
                    best_lab = lab^
            var cs = ops.put(best_c)
            var ds = ops.zeros(m * 2)
            ops.sqdist(sub_s, m, cs, 2, d, ds)
            var lid = len(tree.left)
            var rid = lid + 1
            # each row to its child, the children's score terms and sizes
            var plab = best_lab.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
            ops.ctx.enqueue_function[_bfz_split_kernel](
                ops._ip(ids_s), plab, Int32(m), ops._ip(node_s), Int32(lid), Int32(rid), ops._fp(ds),
                ops._fp(e0), ops._fp(e1), ops._ip(one_s), grid_dim=(m + BFZ_TPB - 1) // BFZ_TPB, block_dim=BFZ_TPB,
            )
            ops.ctx.enqueue_function[_bfz_uncenter_kernel](
                ops._fp(cs), ops._fp(mean_s), Int32(d), Int32(2 * d), ops._fp(cent_all) + lid * d,
                grid_dim=(2 * d + BFZ_TPB - 1) // BFZ_TPB, block_dim=BFZ_TPB,
            )
            var sc0 = ops.sum_ff(e0, -1, -1, m, FM_VAL)
            var sc1 = ops.sum_ff(e1, -1, -1, m, FM_VAL)
            var n1s = ops._scan(ops._ip(one_s), m)
            var n1 = ops._int1(n1s[1])
            _ = best_lab^  # the split kernel above has run (the reads synchronized)
            # the two scans' slots (partials, prefix, total: allocated in that
            # order by `_scan`) are drained; release them so a fit holds O(n)
            _release_scan(ops, sc_pos[0])
            _release_scan(ops, n1s[0])
            _ = tree.add(List[Float32](), sc0, List[Int]())
            _ = tree.add(List[Float32](), sc1, List[Int]())
            sizes.append(m - n1)
            sizes.append(n1)
            ops.shrink(sub_s)
            ops.shrink(ds)
            tree.left[pick] = lid
            tree.right[pick] = rid
        var leaves = tree.leaves()
        var nl = len(leaves)
        var nn = len(tree.left)
        var inertia = Float64(0)
        var lt = List[Int32](length=nn, fill=Int32(-1))
        var lv = List[Int32](capacity=nl)
        for i in range(nl):  # small-loop(nl: tree leaves): at most k leaves of tree metadata
            var t = leaves[i]
            tree.label[t] = i
            lt[t] = Int32(i)
            lv.append(Int32(t))
            inertia = inertia + tree.score[t]
        # labels: each row's leaf to its leaf index; centers: the leaves'
        # rows of `cent_all`, gathered on the device
        var lt_s = ops.put_i(lt)
        var lab_s = ops.zeros_i(n)
        ops.ctx.enqueue_function[_bfz_label_kernel](
            ops._ip(node_s), Int32(n), ops._ip(lt_s), ops._ip(lab_s), grid_dim=ngrid, block_dim=BFZ_TPB,
        )
        var lv_s = ops.put_i(lv)
        var cl_s = ops.empty(nl * d)
        ops.gather_rows(cent_all, d, lv_s, nl, cl_s)
        var labels = ops.get_i(lab_s, n)
        var centers = ops.get(cl_s, nl * d)
        tree.centers = ops.get(cent_all, nn * d)
        # Removing X shifts every later float-buffer slot. Keep it in the
        # slot table until the final center readbacks have used their saved
        # indices; those reads also drain all launches that reference X.
        pool_give["MojoXClusterClus3BisectX"](ops.f.pop(xs))
        var nodes = List[Int32](capacity=3 * nn)
        for t in range(nn):  # small-loop(nn: tree nodes): 2k - 1 nodes of tree metadata
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
