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
from cluster.host.host_cells import ftz_v, host_cells, mul_v

from x_cluster.bodies import (
    FPtr,
    IPtr,
    cov_cell,
    exp_cell,
    gauss_q_cell,
    nk_cell,
    pdist_cell,
    resp_row,
    xk_cell,
    ap_availability_col,
    ap_exemplar_cell,
    ap_responsibility_row,
    kth_smallest_row,
    meanshift_seed,
    nearest_row,
    sqdist_cell,
    sqrt_cell,
    tree_descend,
)
from cluster.host.kmeans_oracle import host_kmeans_fit
from cluster.impl.kmeans_params import METRIC_L2_EXPANDED
from x_cluster.ops import ClusterOps

comptime X_CLUSTER_HOST_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()

#: Cells per vector in the host moments.
comptime MOMENTS_W = 8
#: Covariance rows per host task.
comptime COV_AB = 4
#: Rows per vector in the host Mahalanobis squares.
comptime GAUSS_W = 8
#: Columns per vector in the host availability update.
comptime AP_W = 8



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

    def set(mut self, slot: Int, v: List[Float32]) raises:
        if len(v) > len(self.f[slot]):
            raise Error("x_cluster host: set of " + String(len(v)) + " values into a slot of " + String(len(self.f[slot])))
        memcpy(dest=self._fp(slot), src=v.unsafe_ptr(), count=len(v))

    def sqdist(mut self, a: Int, na: Int, b: Int, nb: Int, d: Int, dst: Int) raises:
        var pa = self._fp(a)
        var pb = self._fp(b)
        var po = self._fp(dst)

        def body(t: Int) {imm pa, imm pb, imm po, imm na, imm nb, imm d}:
            sqdist_cell[X_CLUSTER_HOST_SABOTAGE](pa, na, pb, nb, d, po, t)

        host_cells(body, na * nb, 3 * d)

    def nearest(mut self, a: Int, na: Int, b: Int, nb: Int, d: Int, labels: Int, dist: Int) raises:
        var pa = self._fp(a)
        var pb = self._fp(b)
        var pl = self._ip(labels)
        var pd = self._fp(dist)

        def body(t: Int) {imm pa, imm pb, imm pl, imm pd, imm nb, imm d}:
            nearest_row[X_CLUSTER_HOST_SABOTAGE](pa, pb, nb, d, pl, pd, t)

        host_cells(body, na, 3 * d * nb)

    def sqrt(mut self, x: Int, n: Int) raises:
        var px = self._fp(x)

        def body(t: Int) {imm px}:
            sqrt_cell(px, t)

        host_cells(body, n, 4)

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
            var diff = List[SIMD[DType.float32, GAUSS_W]](length=d, fill=SIMD[DType.float32, GAUSS_W](0))
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

        # The host's moments walk the rows once per output ROW of cells and
        # carry MOMENTS_W neighbouring cells in the lanes of one vector: each
        # lane is its cell's own chain (`bodies.xk_cell` / `bodies.cov_cell`,
        # rows ascending, the same flushes and pinned products), read from
        # contiguous memory instead of one strided walk per cell.
        def xk_task(k: Int) {imm pr, imm px, imm pn, imm pm, imm n, imm d, imm kc}:
            var a0 = 0
            while a0 + MOMENTS_W <= d:
                var acc = SIMD[DType.float32, MOMENTS_W](0)
                for i in range(n):
                    var r = SIMD[DType.float32, MOMENTS_W](pr[i * kc + k])
                    var xv = ftz_v[MOMENTS_W]((px + i * d + a0).load[width=MOMENTS_W]())
                    acc = ftz_v[MOMENTS_W](acc + ftz_v[MOMENTS_W](mul_v[MOMENTS_W](r, xv)))
                comptime for l in range(MOMENTS_W):
                    pm[k * d + a0 + l] = ftz(identical_div(acc[l], pn[k]))
                a0 += MOMENTS_W
            for a in range(a0, d):
                xk_cell(pr, px, n, d, kc, pn, pm, k * d + a)

        host_cells(xk_task, kc, 4 * n * d)

        # A task is one component and COV_AB rows a of the covariance: every
        # row i is read once for COV_AB x d cells (each its own chain,
        # `bodies.cov_cell`), the column differences formed once per row.
        var a_blocks = (d + COV_AB - 1) // COV_AB
        var n_vec = d // MOMENTS_W

        def cov_task(t: Int) {imm pr, imm px, imm pm, imm pn, imm pc, imm n, imm d, imm kc, imm reg, imm a_blocks, imm n_vec}:
            var k = t // a_blocks
            var a0 = (t - k * a_blocks) * COV_AB
            var a1 = min(a0 + COV_AB, d)
            var na = a1 - a0
            var accv = List[SIMD[DType.float32, MOMENTS_W]](length=na * n_vec if na * n_vec > 0 else 1, fill=SIMD[DType.float32, MOMENTS_W](0))
            var tail = d - n_vec * MOMENTS_W
            var accs = List[Float32](length=na * tail if na * tail > 0 else 1, fill=Float32(0))
            var dbv = List[SIMD[DType.float32, MOMENTS_W]](length=n_vec if n_vec > 0 else 1, fill=SIMD[DType.float32, MOMENTS_W](0))
            var dbs = List[Float32](length=tail if tail > 0 else 1, fill=Float32(0))
            for i in range(n):
                var r = pr[i * kc + k]
                var rv = SIMD[DType.float32, MOMENTS_W](r)
                var xrow = px + i * d
                for q in range(n_vec):
                    var b0 = q * MOMENTS_W
                    dbv[q] = ftz_v[MOMENTS_W](ftz_v[MOMENTS_W]((xrow + b0).load[width=MOMENTS_W]()) - (pm + k * d + b0).load[width=MOMENTS_W]())
                for q in range(tail):
                    var b = n_vec * MOMENTS_W + q
                    dbs[q] = ftz(ftz(xrow[b]) - pm[k * d + b])
                for aa in range(na):
                    var a = a0 + aa
                    var da = ftz(ftz(xrow[a]) - pm[k * d + a])
                    var dav = SIMD[DType.float32, MOMENTS_W](da)
                    for q in range(n_vec):
                        var prod = ftz_v[MOMENTS_W](mul_v[MOMENTS_W](dav, dbv[q]))
                        accv[aa * n_vec + q] = ftz_v[MOMENTS_W](accv[aa * n_vec + q] + ftz_v[MOMENTS_W](mul_v[MOMENTS_W](rv, prod)))
                    for q in range(tail):
                        accs[aa * tail + q] = ftz(accs[aa * tail + q] + ftz(identical_mul(r, ftz(identical_mul(da, dbs[q])))))
            for aa in range(na):
                var a = a0 + aa
                for b in range(d):
                    var acc: Float32
                    if b < n_vec * MOMENTS_W:
                        acc = accv[aa * n_vec + b // MOMENTS_W][b % MOMENTS_W]
                    else:
                        acc = accs[aa * tail + b - n_vec * MOMENTS_W]
                    var v = ftz(identical_div(acc, pn[k]))
                    if a == b:
                        v = ftz(v + reg)
                    pc[k * d * d + a * d + b] = v

        host_cells(cov_task, kc * a_blocks, 7 * n * d * COV_AB)

    def pdist(
        mut self, a: Int, na: Int, b: Int, nb: Int, d: Int, metric: Int, p: Float32, dst: Int
    ) raises:
        var pa = self._fp(a)
        var pb = self._fp(b)
        var pd = self._fp(dst)

        def body(t: Int) {imm pa, imm pb, imm pd, imm na, imm nb, imm d, imm metric, imm p}:
            pdist_cell[X_CLUSTER_HOST_SABOTAGE](pa, na, pb, nb, d, metric, p, pd, t)

        host_cells(body, na * nb, 4 * d)
