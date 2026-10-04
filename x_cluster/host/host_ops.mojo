# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into the CPU host binding _mojolearn_x_cluster_host.
"""THE CLUSTER LANE'S HOST COLUMN (lane/algos-cluster): `ClusterOps` on the
CPU. HOST ONLY: no `std.gpu`, no `max.gpu`, no DeviceContext. Each primitive
walks the SAME `x_cluster/bodies.mojo` body as the device kernel, over the
same indices, so the arithmetic of the two columns is one source.

THE NEGATIVE CONTROL. `-D MOJOLEARN_HOST_SABOTAGE=1` walks every distance's
feature fold descending (`bodies.sq_dist_rows[REV=True]`), which moves the low
bits of distances and so the fitted centers.

THE THREAD SPLIT (lane cluster-cpu, 2026-09-28). Every primitive walks
independent indices: each index writes only its own output cells and reads
inputs no other index writes in the same primitive. `host_cells` splits the
index range into contiguous tasks (`cluster/host/host_cells.mojo`: the
caller's floating-point environment, DEVIATION 5900). An index's arithmetic is the serial walk's, in
the same order, so the bits are the same at every thread count; it is not a
numeric row."""
from std.memory import bitcast, memcpy
from std.sys.compile import is_defined

from checks.numerics import ftz, identical_div, identical_mul
from x_cluster.minibatch_cells import mb_center_update
from x_cluster.optics_xi_cells import optics_xi_host
from cluster.host.host_cells import ftz_v, host_cells, mul_v
from x_cluster.host.moments_gemm import gemm_fold_cov, gemm_fold_means

from x_cluster.bodies import (
    FPtr,
    IPtr,
    cov_cell,
    argmax_row,
    exp_cell,
    gauss_q_cell,
    nk_cell,
    pdist_cell,
    resp_row,
    xk_cell,
    ap_availability_col,
    ap_exemplar_cell,
    ap_noise_cell,
    ap_responsibility_row,
    kth_smallest_row,
    meanshift_seed,
    nearest_row,
    SplitMix64,
    sqdist_cell,
    sqrt_cell,
    flush_cell,
    tree_descend,
    ward_nn_row,
)
from cluster.host.kmeans_oracle import host_kmeans_fit
from cluster.impl.kmeans_params import METRIC_L2_EXPANDED
from x_cluster.ops import ClusterOps
from x_cluster.bgmm_device import bgmm_host_step
from x_cluster.post_bodies import (
    FM_MIN,
    FM_PROD,
    FM_VAL,
    FM_WMIN,
    FOLD_CHUNK,
    bin_key,
    bin_value,
    center_greater,
    ff_chunk_host,
    ff_fold_host,
    ff_of_f64,
    ff_to_f64,
    first_equal_cell,
    kpp_search_cell,
    optics_relax_cell,
    rand_resp_row,
)

comptime X_CLUSTER_HOST_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()

#: Cells per vector in the host moments.
comptime MOMENTS_W = 8
#: Covariance rows per host task.
comptime COV_AB = 4
#: Rows per vector in the host Mahalanobis squares.
comptime GAUSS_W = 8
#: Columns per vector in the host availability update.
comptime AP_W = 8
#: Pairs per vector in the host squared distances.
comptime DIST_W = 8
#: Floats of padding around a task's scratch block (two 64-byte lines).
comptime SCRATCH_PAD = 32



def _feature_major(pb: FPtr, nb: Int, d: Int) -> List[Float32]:
    """`bt[f * (nb + DIST_W) + j] = ftz(b[j, f])`, zero-padded by DIST_W."""
    var mp = nb + DIST_W
    var bt = List[Float32](length=d * mp if d > 0 else 1, fill=Float32(0))
    for j in range(nb):
        for f in range(d):
            bt[f * mp + j] = ftz(pb[j * d + f])
    return bt^


@always_inline
def _sq_dists8(pa: FPtr, i: Int, d: Int, btp: FPtr, mp: Int, j0: Int) -> SIMD[DType.float32, DIST_W]:
    """`bodies.sq_dist_rows(a, i, b, j, d)` for j = j0 .. j0+7, one per lane:
    `t = ftz(ftz(a) - ftz(b))`, `acc = ftz(acc + ftz(t * t))` (the pinned
    product), features ascending."""
    var acc = SIMD[DType.float32, DIST_W](0)
    for f in range(d):
        var av = SIMD[DType.float32, DIST_W](ftz(pa[i * d + f]))
        var t = ftz_v[DIST_W](av - (btp + f * mp + j0).load[width=DIST_W]())
        acc = ftz_v[DIST_W](acc + ftz_v[DIST_W](mul_v[DIST_W](t, t)))
    return acc


def _select_u32(mut a: List[UInt32], kk: Int) -> UInt32:
    """The kk-th smallest (0-based) of `a` (reordered in place): quickselect
    with a middle pivot and a three-way partition."""
    var lo = 0
    var hi = len(a) - 1
    while lo < hi:
        var pivot = a[lo + (hi - lo) // 2]
        var lt = lo
        var i = lo
        var gt = hi
        while i <= gt:
            var v = a[i]
            if v < pivot:
                a[i] = a[lt]
                a[lt] = v
                lt += 1
                i += 1
            elif v > pivot:
                a[i] = a[gt]
                a[gt] = v
                gt -= 1
            else:
                i += 1
        if kk < lt:
            hi = lt - 1
        elif kk > gt:
            lo = gt + 1
        else:
            return pivot
    return a[lo]


struct HostOps(ClusterOps):
    var f: List[List[Float32]]
    var i: List[List[Int32]]

    def __init__(out self):
        self.f = List[List[Float32]]()
        self.i = List[List[Int32]]()

    def _fp(mut self, slot: Int) -> FPtr:
        return self.f[slot].unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()

    def _ip(mut self, slot: Int) -> IPtr:
        return self.i[slot].unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()

    def put(mut self, v: List[Float32]) raises -> Int:
        var c = v.copy()
        if len(c) == 0:
            c.append(Float32(0))
        self.f.append(c^)
        return len(self.f) - 1

    def put_i(mut self, v: List[Int32]) raises -> Int:
        var c = v.copy()
        if len(c) == 0:
            c.append(Int32(0))
        self.i.append(c^)
        return len(self.i) - 1

    def zeros(mut self, n: Int) raises -> Int:
        return self.put(List[Float32](length=n, fill=Float32(0)))

    def zeros_i(mut self, n: Int) raises -> Int:
        return self.put_i(List[Int32](length=n, fill=Int32(0)))

    def get(mut self, slot: Int, n: Int) raises -> List[Float32]:
        if n > len(self.f[slot]):
            raise Error("x_cluster host: get of " + String(n) + " values from a slot of " + String(len(self.f[slot])))
        var out = List[Float32](length=n, fill=Float32(0))
        memcpy(dest=out.unsafe_ptr(), src=self.f[slot].unsafe_ptr(), count=n)
        return out^

    def get_i(mut self, slot: Int, n: Int) raises -> List[Int32]:
        if n > len(self.i[slot]):
            raise Error("x_cluster host: get_i of " + String(n) + " values from a slot of " + String(len(self.i[slot])))
        var out = List[Int32](length=n, fill=Int32(0))
        memcpy(dest=out.unsafe_ptr(), src=self.i[slot].unsafe_ptr(), count=n)
        return out^

    def gets(mut self, slots: List[Int], ns: List[Int]) raises -> List[List[Float32]]:
        var outs = List[List[Float32]](capacity=len(slots))
        for q in range(len(slots)):
            outs.append(self.get(slots[q], ns[q]))
        return outs^

    def get_if(
        mut self, islot: Int, ni: Int, fslot: Int, nf: Int, mut oi: List[Int32], mut of: List[Float32]
    ) raises:
        oi = self.get_i(islot, ni)
        of = self.get(fslot, nf)

    def set(mut self, slot: Int, v: List[Float32]) raises:
        if len(v) > len(self.f[slot]):
            raise Error("x_cluster host: set of " + String(len(v)) + " values into a slot of " + String(len(self.f[slot])))
        memcpy(dest=self._fp(slot), src=v.unsafe_ptr(), count=len(v))

    def sqdist(mut self, a: Int, na: Int, b: Int, nb: Int, d: Int, dst: Int) raises:
        var pa = self._fp(a)
        var pb = self._fp(b)
        var po = self._fp(dst)

        comptime if X_CLUSTER_HOST_SABOTAGE:
            def body(t: Int) {imm pa, imm pb, imm po, imm na, imm nb, imm d}:
                sqdist_cell[X_CLUSTER_HOST_SABOTAGE](pa, na, pb, nb, d, po, t)

            host_cells(body, na * nb, 3 * d)
            return
        # Rows of `a` as tasks; DIST_W columns of `b` per vector (b
        # feature-major, flushed once): lane j is cell (i, j)'s own chain
        # (`bodies.sq_dist_rows`).
        var bt = _feature_major(pb, nb, d)
        var btp = bt.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var mp = nb + DIST_W

        def row(i: Int) {imm pa, imm btp, imm po, imm nb, imm d, imm mp}:
            var j0 = 0
            while j0 < nb:
                var acc = _sq_dists8(pa, i, d, btp, mp, j0)
                var lim = min(DIST_W, nb - j0)
                for l in range(lim):
                    po[i * nb + j0 + l] = acc[l]
                j0 += DIST_W

        host_cells(row, na, 3 * d * nb)
        _ = bt^

    def nearest(mut self, a: Int, na: Int, b: Int, nb: Int, d: Int, labels: Int, dist: Int) raises:
        var pa = self._fp(a)
        var pb = self._fp(b)
        var pl = self._ip(labels)
        var pd = self._fp(dist)

        comptime if X_CLUSTER_HOST_SABOTAGE:
            def body(t: Int) {imm pa, imm pb, imm pl, imm pd, imm nb, imm d}:
                nearest_row[X_CLUSTER_HOST_SABOTAGE](pa, pb, nb, d, pl, pd, t)

            host_cells(body, na, 3 * d * nb)
            return
        # `bodies.nearest_row` with DIST_W candidate rows per vector: the
        # distances are each pair's own chain, then the argmin walks the
        # candidates in order under the strict `<` (the lowest index on a tie).
        var bt = _feature_major(pb, nb, d)
        var btp = bt.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var mp = nb + DIST_W

        def row(i: Int) {imm pa, imm btp, imm pl, imm pd, imm nb, imm d, imm mp}:
            var best = Float32(0)
            var bi = 0
            var j0 = 0
            while j0 < nb:
                var acc = _sq_dists8(pa, i, d, btp, mp, j0)
                var lim = min(DIST_W, nb - j0)
                for l in range(lim):
                    var j = j0 + l
                    if j == 0 or acc[l] < best:
                        best = acc[l]
                        bi = j
                j0 += DIST_W
            pl[i] = Int32(bi)
            pd[i] = best

        host_cells(row, na, 3 * d * nb)
        _ = bt^

    def sqrt(mut self, x: Int, n: Int) raises:
        var px = self._fp(x)

        def body(t: Int) {imm px}:
            sqrt_cell(px, t)

        host_cells(body, n, 4)

    def flush(mut self, x: Int, n: Int) raises:
        var px = self._fp(x)

        def body(t: Int) {imm px}:
            flush_cell(px, t)

        host_cells(body, n, 1)

    def kth(mut self, m: Int, n_rows: Int, n_cols: Int, k: Int, dst: Int) raises:
        var pm = self._fp(m)
        var po = self._fp(dst)

        if k < 1 or k > n_cols:
            def body(t: Int) {imm pm, imm po, imm n_cols, imm k}:
                kth_smallest_row(pm, n_cols, k, po, t)

            host_cells(body, n_rows, 32 * n_cols)
            return

        # The k-th smallest is ONE value, so any exact selection returns it:
        # the host selects over the row's magnitude bits (-0.0 as +0.0, the
        # body's order) instead of the body's 31 counting passes; a value
        # past +inf (a NaN) reads as +inf, where the body's bisection stops.
        def sel(t: Int) {imm pm, imm po, imm n_cols, imm k}:
            var buf = List[UInt32](length=n_cols, fill=UInt32(0))
            for j in range(n_cols):
                buf[j] = bitcast[DType.uint32](pm[t * n_cols + j]) & UInt32(0x7FFFFFFF)
            var v = _select_u32(buf, k - 1)
            if v > UInt32(0x7F800000):
                v = UInt32(0x7F800000)
            po[t] = bitcast[DType.float32](v)

        host_cells(sel, n_rows, 4 * n_cols)

    def meanshift(
        mut self, x: Int, n: Int, d: Int, bw: Float32, stop: Float32, max_iter: Int,
        centers: Int, ns: Int, scratch: Int, intensity: Int, iters: Int,
    ) raises:
        var px = self._fp(x)
        var pc = self._fp(centers)
        var ps = self._fp(scratch)
        var pi = self._ip(intensity)
        var pt = self._ip(iters)

        def body(t: Int) {imm px, imm pc, imm ps, imm pi, imm pt, imm n, imm d, imm bw, imm stop, imm max_iter}:
            meanshift_seed[X_CLUSTER_HOST_SABOTAGE](px, n, d, bw, stop, max_iter, pc, ps, pi, pt, t)

        host_cells(body, ns, 3 * n * d)

    def ap_r(mut self, s: Int, a: Int, r: Int, n: Int, damping: Float32) raises:
        var ps = self._fp(s)
        var pa = self._fp(a)
        var pr = self._fp(r)

        def body(t: Int) {imm ps, imm pa, imm pr, imm n, imm damping}:
            ap_responsibility_row(ps, pa, pr, n, damping, t)

        host_cells(body, n, 6 * n)

    def ap_a(mut self, r: Int, a: Int, n: Int, damping: Float32) raises:
        var pr = self._fp(r)
        var pa = self._fp(a)

        comptime if X_CLUSTER_HOST_SABOTAGE:
            def body(t: Int) {imm pr, imm pa, imm n, imm damping}:
                ap_availability_col[X_CLUSTER_HOST_SABOTAGE](pr, pa, n, damping, t)

            host_cells(body, n, 6 * n)
            return

        # EIGHT COLUMNS PER VECTOR: lane l is column k0 + l, its own chains
        # (`bodies.ap_availability_col`: colsum over i ascending, then each
        # cell's update), read along contiguous rows instead of one strided
        # walk per column. A ragged last group runs the body.
        var n_groups = n // AP_W

        def group(g: Int) {imm pr, imm pa, imm n, imm damping}:
            var k0 = g * AP_W
            var one_minus = ftz(Float32(1) - damping)
            var dv = SIMD[DType.float32, AP_W](damping)
            var om = SIMD[DType.float32, AP_W](one_minus)
            var zero = SIMD[DType.float32, AP_W](0)
            var kidx = SIMD[DType.int64, AP_W](0)
            comptime for l in range(AP_W):
                kidx[l] = Int64(k0 + l)
            var colsum = SIMD[DType.float32, AP_W](0)
            for i in range(n):
                var v = (pr + i * n + k0).load[width=AP_W]()
                var diag = kidx.eq(SIMD[DType.int64, AP_W](Int64(i)))
                var rp = (diag | v.gt(zero)).select(v, zero)
                colsum = ftz_v[AP_W](colsum + rp)
            for i in range(n):
                var v = (pr + i * n + k0).load[width=AP_W]()
                var diag = kidx.eq(SIMD[DType.int64, AP_W](Int64(i)))
                var rp = (diag | v.gt(zero)).select(v, zero)
                var nw = ftz_v[AP_W](colsum - rp)
                nw = ((~diag) & nw.gt(zero)).select(zero, nw)
                var old = (pa + i * n + k0).load[width=AP_W]()
                (pa + i * n + k0).store(ftz_v[AP_W](ftz_v[AP_W](mul_v[AP_W](old, dv)) + ftz_v[AP_W](mul_v[AP_W](nw, om))))

        host_cells(group, n_groups, 12 * n * AP_W)
        for t in range(n_groups * AP_W, n):
            ap_availability_col[X_CLUSTER_HOST_SABOTAGE](pr, pa, n, damping, t)

    def ap_noise(mut self, s: Int, m: Int, seed: UInt64) raises:
        var ps = self._fp(s)
        for t in range(m):
            ap_noise_cell(ps, seed, t)

    def ap_e(mut self, a: Int, r: Int, n: Int, e: Int) raises:
        var pa = self._fp(a)
        var pr = self._fp(r)
        var pe = self._ip(e)

        def body(t: Int) {imm pa, imm pr, imm pe, imm n}:
            ap_exemplar_cell(pa, pr, n, pe, t)

        host_cells(body, n, 2)

    def descend(mut self, x: Int, n: Int, d: Int, centers: Int, nodes: Int, labels: Int) raises:
        var px = self._fp(x)
        var pc = self._fp(centers)
        var pn = self._ip(nodes)
        var pl = self._ip(labels)

        def body(t: Int) {imm px, imm pc, imm pn, imm pl, imm d}:
            tree_descend[X_CLUSTER_HOST_SABOTAGE](px, d, pc, pn, pl, t)

        host_cells(body, n, 24 * d)

    def kmeans(
        mut self, x: List[Float32], n: Int, d: Int, k: Int, max_iter: Int, tol: Float64,
        seed: UInt64, n_init: Int, init: Int, mut centers: List[Float32], mut labels: List[Int32],
        weights: List[Float32] = List[Float32](),
    ) raises -> Float64:
        centers = List[Float32](length=k * d, fill=Float32(0))
        var lab = List[UInt32](length=n, fill=UInt32(0))
        var r = host_kmeans_fit(
            x, n, d, k, centers, lab, weights, len(weights), max_iter, tol, seed, n_init, init,
            METRIC_L2_EXPANDED,
        )
        labels = List[Int32](capacity=n)
        for t in range(n):
            labels.append(Int32(lab[t]))
        return r.inertia

    def gather_rows(mut self, src: Int, d: Int, idx: Int, m: Int, dst: Int) raises:
        var ps = self._fp(src)
        var pi = self._ip(idx)
        var pd = self._fp(dst)
        for t in range(m):
            var r = Int(pi[t])
            for f in range(d):
                pd[t * d + f] = ps[r * d + f]

    def kmeans_rows(
        mut self, sub: Int, x: List[Float32], rows: List[Int], d: Int, k: Int, max_iter: Int,
        tol: Float64, seed: UInt64, n_init: Int, init: Int, mut centers: List[Float32],
        mut labels: List[Int32],
    ) raises -> Float64:
        var g = List[Float32](length=len(rows) * d, fill=Float32(0))
        for t in range(len(rows)):
            memcpy(dest=g.unsafe_ptr() + t * d, src=x.unsafe_ptr() + rows[t] * d, count=d)
        return self.kmeans(g, len(rows), d, k, max_iter, tol, seed, n_init, init, centers, labels)

    def shrink(mut self, slot: Int) raises:
        self.f[slot] = List[Float32](length=1, fill=Float32(0))

    def empty(mut self, n: Int) raises -> Int:
        return self.zeros(n)

    def gauss_q(mut self, x: Int, n: Int, d: Int, means: Int, pchol: Int, kc: Int, dst: Int) raises:
        var px = self._fp(x)
        var pm = self._fp(means)
        var pp = self._fp(pchol)
        var pd = self._fp(dst)

        comptime if X_CLUSTER_HOST_SABOTAGE:
            def body(t: Int) {imm px, imm pm, imm pp, imm pd, imm d, imm kc}:
                gauss_q_cell[X_CLUSTER_HOST_SABOTAGE](px, d, pm, pp, kc, pd, t)

            host_cells(body, n * kc, 2 * d * d)
            return

        # EIGHT ROWS PER VECTOR: lane l is cell (i0 + l, k), its own chains
        # (`bodies.gauss_q_cell`: y_j over a ascending, then the sum of y_j^2
        # over j ascending, the same flushes and pinned products). A block's
        # differences are formed once per component and reused across j.
        var n_blocks = n // GAUSS_W

        def block(bi: Int) {imm px, imm pm, imm pp, imm pd, imm d, imm kc}:
            var i0 = bi * GAUSS_W
            # padded per-task scratch (no cache line shared with another task)
            var dscr = List[SIMD[DType.float32, GAUSS_W]](length=d + 8, fill=SIMD[DType.float32, GAUSS_W](0))
            var diff = dscr.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]() + 4
            for k in range(kc):
                for a in range(d):
                    var xv = SIMD[DType.float32, GAUSS_W](0)
                    comptime for l in range(GAUSS_W):
                        xv[l] = px[(i0 + l) * d + a]
                    diff[a] = ftz_v[GAUSS_W](ftz_v[GAUSS_W](xv) - SIMD[DType.float32, GAUSS_W](ftz(pm[k * d + a])))
                var acc = SIMD[DType.float32, GAUSS_W](0)
                for j in range(d):
                    var y = SIMD[DType.float32, GAUSS_W](0)
                    for a in range(j + 1):
                        var pv = SIMD[DType.float32, GAUSS_W](pp[k * d * d + a * d + j])
                        y = ftz_v[GAUSS_W](y + ftz_v[GAUSS_W](mul_v[GAUSS_W](diff[a], pv)))
                    acc = ftz_v[GAUSS_W](acc + ftz_v[GAUSS_W](mul_v[GAUSS_W](y, y)))
                comptime for l in range(GAUSS_W):
                    pd[(i0 + l) * kc + k] = acc[l]
            _ = dscr^

        host_cells(block, n_blocks, kc * d * d * 2)
        for t in range(n_blocks * GAUSS_W * kc, n * kc):
            gauss_q_cell[X_CLUSTER_HOST_SABOTAGE](px, d, pm, pp, kc, pd, t)

    def resp(mut self, q: Int, c: Int, n: Int, kc: Int, lpn: Int) raises:
        var pq = self._fp(q)
        var pc = self._fp(c)
        var pl = self._fp(lpn)

        def body(t: Int) {imm pq, imm pc, imm pl, imm kc}:
            resp_row(pq, pc, kc, pl, t)

        host_cells(body, n, 40 * kc)

    def exp(mut self, src: Int, dst: Int, n: Int) raises:
        var ps = self._fp(src)
        var pd = self._fp(dst)

        def body(t: Int) {imm ps, imm pd}:
            exp_cell(ps, pd, t)

        host_cells(body, n, 30)

    def argmax_rows(mut self, src: Int, n: Int, kc: Int, labels: Int) raises:
        var ps = self._fp(src)
        var pl = self._ip(labels)

        def body(t: Int) {imm ps, imm pl, imm kc}:
            argmax_row(ps, kc, pl, t)

        if n > 0 and kc > 0:
            host_cells(body, n, kc)

    def moments(
        mut self, resp: Int, x: Int, n: Int, d: Int, kc: Int, reg: Float32, nk: Int, means: Int, cov: Int
    ) raises:
        var pr = self._fp(resp)
        var px = self._fp(x)
        var pn = self._fp(nk)
        var pm = self._fp(means)
        var pc = self._fp(cov)

        def nk_body(t: Int) {imm pr, imm pn, imm n, imm kc}:
            nk_cell(pr, n, kc, pn, t)

        host_cells(nk_body, kc, 2 * n)

        # DEVIATION 5110 (revised 2026-09-29): means and covariances fold the
        # sample axis through the identical GEMM (`x_cluster/host/
        # moments_gemm.mojo`), the device's `identical_gemm_into` bit for bit.
        var nkl = List[Float32](length=kc, fill=Float32(0))
        for k in range(kc):
            nkl[k] = pn[k]
        var ml = gemm_fold_means(self.f[resp], self.f[x], nkl, n, d, kc)
        for t in range(kc * d):
            pm[t] = ml[t]
        var cl = gemm_fold_cov(self.f[resp], self.f[x], ml, nkl, n, d, kc, reg)
        for t in range(kc * d * d):
            pc[t] = cl[t]

    def pdist(
        mut self, a: Int, na: Int, b: Int, nb: Int, d: Int, metric: Int, p: Float32, dst: Int
    ) raises:
        var pa = self._fp(a)
        var pb = self._fp(b)
        var pd = self._fp(dst)

        def body(t: Int) {imm pa, imm pb, imm pd, imm na, imm nb, imm d, imm metric, imm p}:
            pdist_cell[X_CLUSTER_HOST_SABOTAGE](pa, na, pb, nb, d, metric, p, pd, t)

        host_cells(body, na * nb, 4 * d)

    def fast_device(self) -> Bool:
        return False

    def ward_nn(mut self, c: Int, sz: Int, l: Int, d: Int, nn: Int, md: Int) raises:
        var pc = self._fp(c)
        var ps = self._fp(sz)
        var pn = self._ip(nn)
        var pm = self._fp(md)
        for p in range(l):
            ward_nn_row(pc, ps, l, d, pn, pm, p)

    def kth_flat(mut self, m: Int, n: Int, k: Int) raises -> Float32:
        var dst = self.zeros(1)
        self.kth(m, 1, n, k, dst)
        return self.f[dst][0]

    def get_diag(mut self, slot: Int, n: Int) raises -> List[Float32]:
        if n * n > len(self.f[slot]):
            raise Error("x_cluster host: get_diag of " + String(n) + " from a slot of " + String(len(self.f[slot])))
        var out = List[Float32](capacity=n)
        for i in range(n):
            out.append(self.f[slot][i * n + i])
        return out^

    def ap_a_split(mut self, r: Int, a: Int, n: Int, damping: Float32) raises:
        # the host column never takes the FAST device paths (`fast_device`)
        self.ap_a(r, a, n, damping)

    def dot_groups(mut self, a: Int, b: Int, n: Int, g: Int, parts: Int) raises:
        var pa = self._fp(a)
        var pb = self._fp(b)
        var po = self._fp(parts)
        var q = 0
        var t0 = 0
        while t0 < n:
            var t1 = t0 + g
            if t1 > n:
                t1 = n
            var acc = Float32(0)
            for t in range(t0, t1):
                acc = acc + pa[t] * pb[t]
            po[q] = acc
            q += 1
            t0 = t1

    def alloc(mut self, n: Int) raises -> Int:
        return self.zeros(n)

    def agglo_on_device(self) -> Bool:
        return False

    def agglo_mirror(mut self, x: Int, n: Int, dst: Int) raises:
        raise Error("x_cluster: agglo_mirror is the GPU column's (the host column runs agglo_tree's loop)")

    def agglo_merge(
        mut self, dm: Int, adj: Int, n: Int, linkage: Int, n_merges: Int, mut children: List[Int32],
        mut dist: List[Float32],
    ) raises:
        raise Error("x_cluster: agglo_merge is the GPU column's (the host column runs agglo_tree's loop)")

    def agglo_connect(
        mut self, edges: Int, n_edges: Int, n: Int, dm: Int, linkage: Int, adj: Int, edge_mode: Int
    ) raises -> Int:
        raise Error("x_cluster: agglo_connect is the GPU column's (the host column runs agglo_tree's loop)")

    def tree_parent(mut self, children: Int, n: Int, m: Int, parent: Int) raises:
        var pc = self._ip(children)
        var pp = self._ip(parent)
        for j in range(n + m):
            pp[j] = Int32(j)
        for t in range(m):
            pp[Int(pc[2 * t])] = Int32(n + t)
            pp[Int(pc[2 * t + 1])] = Int32(n + t)

    def tree_roots(mut self, parent: Int, total: Int, rank1: Int) raises -> Int:
        var pp = self._ip(parent)
        var pr = self._ip(rank1)
        var c = 0
        for j in range(total):
            if Int(pp[j]) == j:
                c += 1
                pr[j] = Int32(c)
            else:
                pr[j] = 0
        return c

    def tree_scatter(mut self, nodes: Int, c: Int, rank1: Int) raises:
        var pn = self._ip(nodes)
        var pr = self._ip(rank1)
        for i in range(c):
            pr[Int(pn[i])] = Int32(i + 1)

    def tree_leaf_label(mut self, parent: Int, rank1: Int, n: Int, labels: Int) raises:
        var pp = self._ip(parent)
        var pr = self._ip(rank1)
        var pl = self._ip(labels)
        for t in range(n):
            var v = t
            var lab = Int32(-1)
            while True:
                if pr[v] != 0:
                    lab = pr[v] - 1
                    break
                var p = Int(pp[v])
                if p == v:
                    break
                v = p
            pl[t] = lab

    def count_ge(mut self, x: Int, n: Int, thr: Float32) raises -> Int:
        var p = self._fp(x)
        var c = 0
        for t in range(n):
            if p[t] >= thr:
                c += 1
        return c

    def estep(
        mut self, x: Int, n: Int, d: Int, means: Int, pchol: Int, c: Int, kc: Int, q: Int, r: Int, lpn: Int
    ) raises:
        self.gauss_q(x, n, d, means, pchol, kc, q)
        self.resp(q, c, n, kc, lpn)
        self.exp(q, r, n * kc)

    def optics_order_fast(
        mut self, dm: Int, core: Int, n: Int, max_eps: Float32, ordering: Int, reach: Int, pred: Int, proc: Int
    ) raises:
        # the host column never takes the FAST device paths (`fast_device`):
        # the driver's serial loop over the slots, for the trait's sake
        var inf = Float32.MAX * Float32(2)
        var pd = self._fp(dm)
        var pcore = self._fp(core)
        var po = self._ip(ordering)
        var pr = self._fp(reach)
        var pp = self._ip(pred)
        var pq = self._ip(proc)
        for j in range(n):
            pr[j] = inf
            pp[j] = Int32(-1)
            pq[j] = Int32(0)
        for step in range(n):
            var point = -1
            var best = inf
            for j in range(n):
                if pq[j] != Int32(0):
                    continue
                if point < 0 or pr[j] < best:
                    point = j
                    best = pr[j]
            pq[point] = Int32(1)
            po[step] = Int32(point)
            var cp = pcore[point]
            if cp <= max_eps and cp != inf:
                for o in range(n):
                    if pq[o] != Int32(0):
                        continue
                    var dd = pd[point * n + o]
                    if not (dd <= max_eps):
                        continue
                    var rd = dd if dd > cp else cp
                    if rd < pr[o]:
                        pr[o] = rd
                        pp[o] = Int32(point)

    def minibatch_fast(
        mut self, xs: Int, n: Int, d: Int, k: Int, batch: Int, n_steps: Int, max_no_improvement: Int,
        ratio: Float64, seed: UInt64, mut rng: SplitMix64, mut c: List[Float32], mut w: List[Float32],
        mut steps_done: Int,
    ) raises -> Bool:
        # the host column never takes the FAST device paths (`fast_device`)
        return False

    def set_i(mut self, slot: Int, v: List[Int32]) raises:
        if len(v) > len(self.i[slot]):
            raise Error("x_cluster host: set_i of " + String(len(v)) + " values into a slot of " + String(len(self.i[slot])))
        memcpy(dest=self._ip(slot), src=v.unsafe_ptr(), count=len(v))

    def mb_update(mut self, b: Int, batch: Int, labels: Int, c: Int, w: Int, k: Int, d: Int) raises:
        var pb = self._fp(b)
        var pl = self._ip(labels)
        var pc = self._fp(c)
        var pw = self._fp(w)
        for j in range(k):
            mb_center_update(pb, batch, pl, pc, pw, j, d)

    def mb_assign(mut self, src: Int, d: Int, idx: Int, m: Int, c: Int, k: Int, labels: Int, dist: Int, dst: Int) raises:
        self.gather_rows(src, d, idx, m, dst)
        self.nearest(dst, m, c, k, d, labels, dist)

    # ------------------------------------------------------------------
    # lane cgr2-cluster: the post-processing primitives, host column
    def _fp_or(mut self, slot: Int) -> FPtr:
        return self._fp(slot if slot >= 0 else 0)

    def check_nonneg(mut self, x: Int, n: Int) raises -> Bool:
        var p = self._fp(x)
        for t in range(n):
            if not (p[t] >= Float32(0)):
                return False
        return True

    def optics_order(
        mut self, dm: Int, core: Int, n: Int, max_eps: Float32, ordering: Int, reach: Int, pred: Int
    ) raises:
        var inf = Float32.MAX * Float32(2)
        var pd = self._fp(dm)
        var pc = self._fp(core)
        var po = self._ip(ordering)
        var pr = self._fp(reach)
        var pp = self._ip(pred)
        var done = List[Int32](length=n, fill=Int32(0))
        var pdone = done.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        for i in range(n):
            if pc[i] > max_eps:
                pc[i] = inf
            pr[i] = inf
            pp[i] = Int32(-1)
        for step in range(n):
            var point = -1
            var best = inf
            for i in range(n):
                if pdone[i] != 0:
                    continue
                if point < 0 or pr[i] < best:
                    point = i
                    best = pr[i]
            pdone[point] = 1
            po[step] = Int32(point)
            var cp = pc[point]
            if cp != inf:
                for o in range(n):
                    optics_relax_cell(pd, n, point, cp, max_eps, pdone, pr, pp, o)
        _ = done^

    def optics_fast(
        mut self, dm: Int, core: Int, n: Int, max_eps: Float32, sq: Bool,
        mut ordering: List[Int32], mut reach: List[Float32], mut core_out: List[Float32], mut pred: List[Int32],
    ) raises -> Bool:
        # the host column never takes the FAST device paths (`fast_device`)
        return False

    def optics_dbscan(mut self, ordering: Int, reach: Int, core: Int, n: Int, eps: Float32, labels: Int) raises:
        var po = self._ip(ordering)
        var pr = self._fp(reach)
        var pc = self._fp(core)
        var pl = self._ip(labels)
        var c = -1
        for q in range(n):
            var p = Int(po[q])
            if pr[p] > eps and pc[p] <= eps:
                c += 1
            pl[p] = Int32(c)
        for p in range(n):
            if pr[p] > eps and not (pc[p] <= eps):
                pl[p] = Int32(-1)

    def optics_xi(
        mut self, ordering: Int, reach: Int, pred: Int, n: Int, xc: Float32, min_samples: Int,
        min_cluster_size: Int, predecessor_correction: Bool, labels: Int,
    ) raises -> List[Int32]:
        return optics_xi_host(
            self._ip(ordering), self._fp(reach), self._ip(pred), n, xc, min_samples, min_cluster_size,
            predecessor_correction, self._ip(labels),
        )

    def sum_ff(mut self, a: Int, b: Int, c: Int, n: Int, mode: Int) raises -> Float64:
        return ff_to_f64(ff_fold_host(mode, self._fp(a), self._fp_or(b), self._fp_or(c), n))

    def fold_into(mut self, a: Int, b: Int, c: Int, n: Int, mode: Int, dst: Int) raises:
        var v = ff_fold_host(mode, self._fp(a), self._fp_or(b), self._fp_or(c), n)
        var po = self._fp(dst)
        po[0] = v.hi
        po[1] = v.lo

    def bgmm_step(
        mut self, step: Int, kc: Int, d: Int, cfg: Int, aux: Int, w: Int, p1: Int, p2: Int, p3: Int
    ) raises:
        bgmm_host_step(
            step, self._fp(w), self._fp(p1 if p1 >= 0 else w), self._fp(p2 if p2 >= 0 else w),
            self._fp(p3 if p3 >= 0 else w), kc, d, cfg, aux,
        )

    def bin_seeds(mut self, x: Int, n: Int, d: Int, bin_size: Float32, min_bin_freq: Int, dst: Int) raises -> Int:
        var px = self._fp(x)
        var keys = List[Float32](length=n * d if n * d > 0 else 1, fill=Float32(0))
        var pk = keys.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        for t in range(n * d):
            pk[t] = bin_key(px[t], bin_size)
        var none = List[Int32](length=1, fill=Int32(0))
        var pn = none.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var cnt = List[Int](length=n, fill=0)
        for r in range(n):
            cnt[Int(first_equal_cell(pk, pn, False, d, r))] += 1
        var pd = self._fp(dst)
        var kept = 0
        for u in range(n):
            if cnt[u] > 0 and cnt[u] >= min_bin_freq:
                for f in range(d):
                    pd[kept * d + f] = bin_value(pk[u * d + f], bin_size)
                kept += 1
        _ = keys^
        _ = none^
        return kept

    def ms_unique(
        mut self, centers: Int, inten: Int, iters: Int, ns: Int, d: Int, dst: Int, mut n_iter: Int
    ) raises -> Int:
        var pc = self._fp(centers)
        var pi = self._ip(inten)
        var pt = self._ip(iters)
        n_iter = 0
        for s in range(ns):
            if Int(pt[s]) > n_iter:
                n_iter = Int(pt[s])
        var rep = List[Int32](length=ns, fill=Int32(-1))
        var rv = List[Int32](length=ns, fill=Int32(0))
        var prv = rv.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        for s in range(ns):
            rep[s] = first_equal_cell(pc, pi, True, d, s)
            if rep[s] >= 0:
                prv[Int(rep[s])] = pi[s]
        var pd = self._fp(dst)
        var m = 0
        for u in range(ns):
            if Int(rep[u]) != u:
                continue
            m += 1
            var rank = 0
            for v in range(ns):
                if v != u and Int(rep[v]) == v and center_greater(pc, prv, v, u, d):
                    rank += 1
            for f in range(d):
                pd[rank * d + f] = pc[u * d + f]
        _ = rv^
        return m

    def ms_suppress(mut self, sorted: Int, dd: Int, m: Int, d: Int, bw: Float32, dst: Int) raises -> Int:
        var ps = self._fp(sorted)
        var pdd = self._fp(dd)
        var unique = List[Bool](length=m, fill=True)
        for i in range(m):
            if unique[i]:
                for j in range(m):
                    if pdd[i * m + j] <= bw:
                        unique[j] = False
                unique[i] = True
        var po = self._fp(dst)
        var kc = 0
        for i in range(m):
            if unique[i]:
                for f in range(d):
                    po[kc * d + f] = ps[i * d + f]
                kc += 1
        return kc

    def ms_noise(mut self, labels: Int, dist: Int, n: Int, bw: Float32) raises:
        var pl = self._ip(labels)
        var pd = self._fp(dist)
        for r in range(n):
            if not (pd[r] <= bw):
                pl[r] = Int32(-1)

    def negate(mut self, src: Int, dst: Int, n: Int) raises:
        var ps = self._fp(src)
        var pd = self._fp(dst)
        for t in range(n):
            pd[t] = -ps[t]

    def count_neg(mut self, x: Int, n: Int) raises -> Int:
        var p = self._fp(x)
        var c = 0
        for t in range(n):
            if p[t] < Float32(0):
                c += 1
        return c

    def sign_side(mut self, src: Int, n: Int, neg: Bool, dst: Int) raises:
        var inf = Float32.MAX * Float32(2)
        var ps = self._fp(src)
        var pd = self._fp(dst)
        for t in range(n):
            var v = ps[t]
            if neg:
                pd[t] = -v if v < Float32(0) else inf
            else:
                pd[t] = inf if v < Float32(0) else v

    def ap_equal(mut self, s: Int, pref: Int, n: Int) raises -> Bool:
        var ps = self._fp(s)
        var pp = self._fp(pref)
        if n >= 2:
            var first = ps[1]
            for i in range(n):
                for j in range(n):
                    if i != j and ps[i * n + j] != first:
                        return False
        for i in range(1, n):
            if pp[i] != pp[0]:
                return False
        return True

    def set_diag(mut self, s: Int, v: Int, n: Int) raises:
        var ps = self._fp(s)
        var pv = self._fp(v)
        for i in range(n):
            ps[i * n + i] = pv[i]

    def ap_conv(mut self, e: Int, ring: Int, n: Int, conv_iter: Int, it: Int) raises -> Bool:
        var pe = self._ip(e)
        var pr = self._ip(ring)
        var K = 0
        for i in range(n):
            pr[i * conv_iter + it % conv_iter] = pe[i]
            K += Int(pe[i])
        if it < conv_iter:
            return False
        for i in range(n):
            var se = 0
            for c in range(conv_iter):
                se += Int(pr[i * conv_iter + c])
            if se != conv_iter and se != 0:
                return False
        return K > 0

    def ap_exemplars(mut self, s: Int, e: Int, n: Int, centers: Int, labels: Int) raises -> Int:
        var ps = self._fp(s)
        var pe = self._ip(e)
        var pcen = self._ip(centers)
        var plab = self._ip(labels)
        var ex = List[Int]()
        for i in range(n):
            if pe[i] != 0:
                ex.append(i)
        var K = len(ex)
        if K == 0:
            for i in range(n):
                plab[i] = Int32(-1)
            return 0
        var c = _ap_argmax_cols(ps, n, ex)
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
                    acc = ftz(acc + ps[ii[q] * n + ii[jj]])
                if jj == 0 or acc > best_v:
                    best_v = acc
                    best = jj
            ex[k] = ii[best]
        c = _ap_argmax_cols(ps, n, ex)
        for k in range(K):
            c[ex[k]] = k
        var is_center = List[Bool](length=n, fill=False)
        for i in range(n):
            is_center[ex[c[i]]] = True
        var rank = List[Int](length=n, fill=-1)
        var nc = 0
        for i in range(n):
            if is_center[i]:
                rank[i] = nc
                pcen[nc] = Int32(i)
                nc += 1
        for i in range(n):
            plab[i] = Int32(rank[ex[c[i]]])
        return nc

    def onehot(mut self, idx: Int, m: Int, kc: Int, by_row: Bool, dst: Int) raises:
        var pi = self._ip(idx)
        var pd = self._fp(dst)
        for q in range(m):
            if by_row:
                pd[q * kc + Int(pi[q])] = Float32(1)
            else:
                pd[Int(pi[q]) * kc + q] = Float32(1)

    def rand_resp(mut self, dst: Int, n: Int, kc: Int, state: UInt64) raises:
        var pd = self._fp(dst)
        for i in range(n):
            rand_resp_row(state, kc, pd, i)

    def kpp_search(mut self, closest: Int, w: Int, m: Int, vs: List[Float64], ids: Int) raises:
        var pc = self._fp(closest)
        var pw = self._fp_or(w)
        var mode = FM_PROD if w >= 0 else FM_VAL
        var nch = (m + FOLD_CHUNK - 1) // FOLD_CHUNK
        var th = List[Float32](length=nch, fill=Float32(0))
        var tl = List[Float32](length=nch, fill=Float32(0))
        for q in range(nch):
            var v = ff_chunk_host(mode, pc, pw, pc, q * FOLD_CHUNK, min(m, (q + 1) * FOLD_CHUNK))
            th[q] = v.hi
            tl[q] = v.lo
        var pth = th.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var ptl = tl.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var pi = self._ip(ids)
        for t in range(len(vs)):
            var v = ff_of_f64(vs[t])
            pi[t] = Int32(kpp_search_cell(mode, pc, pw, pth, ptl, m, v.hi, v.lo))
        _ = th^
        _ = tl^

    def kpp_pots(mut self, dc: Int, closest: Int, w: Int, nt: Int, m: Int) raises -> List[Float64]:
        var pc = self._fp(closest)
        var pw = self._fp_or(w)
        var out = List[Float64](capacity=nt)
        for t in range(nt):
            var pdc = self._fp(dc) + t * m
            out.append(ff_to_f64(ff_fold_host(FM_WMIN if w >= 0 else FM_MIN, pdc, pc, pw, m)))
        return out^

    def kpp_take(mut self, dc: Int, closest: Int, best: Int, m: Int) raises:
        var pc = self._fp(closest)
        var pdc = self._fp(dc) + best * m
        for j in range(m):
            var v = pdc[j]
            if v < pc[j]:
                pc[j] = v


def _ap_argmax_cols(s_m: FPtr, n: Int, cols: List[Int]) -> List[Int]:
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
