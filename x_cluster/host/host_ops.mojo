# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into the CPU host binding _mojolearn_x_cluster_host.
"""THE CLUSTER LANE'S HOST COLUMN (lane/algos-cluster): `ClusterOps` on the
CPU. HOST ONLY: no `std.gpu`, no `max.gpu`, no DeviceContext. Each primitive
walks the SAME `x_cluster/bodies.mojo` body as the device kernel, over the
same indices, so the arithmetic of the two columns is one source.

THE NEGATIVE CONTROL. `-D MOJOLEARN_HOST_SABOTAGE=1` walks every distance's
feature fold descending (`bodies.sq_dist_rows[REV=True]`), which moves the low
bits of distances and so the fitted centers."""
from std.sys.compile import is_defined

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
    ap_noise_cell,
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
        var out = List[Float32](capacity=n)
        for t in range(n):
            out.append(self.f[slot][t])
        return out^

    def get_i(mut self, slot: Int, n: Int) raises -> List[Int32]:
        var out = List[Int32](capacity=n)
        for t in range(n):
            out.append(self.i[slot][t])
        return out^

    def set(mut self, slot: Int, v: List[Float32]) raises:
        for t in range(len(v)):
            self.f[slot][t] = v[t]

    def sqdist(mut self, a: Int, na: Int, b: Int, nb: Int, d: Int, dst: Int) raises:
        var pa = self._fp(a)
        var pb = self._fp(b)
        var po = self._fp(dst)
        for t in range(na * nb):
            sqdist_cell[X_CLUSTER_HOST_SABOTAGE](pa, na, pb, nb, d, po, t)

    def nearest(mut self, a: Int, na: Int, b: Int, nb: Int, d: Int, labels: Int, dist: Int) raises:
        var pa = self._fp(a)
        var pb = self._fp(b)
        var pl = self._ip(labels)
        var pd = self._fp(dist)
        for t in range(na):
            nearest_row[X_CLUSTER_HOST_SABOTAGE](pa, pb, nb, d, pl, pd, t)

    def sqrt(mut self, x: Int, n: Int) raises:
        var px = self._fp(x)
        for t in range(n):
            sqrt_cell(px, t)

    def kth(mut self, m: Int, n_rows: Int, n_cols: Int, k: Int, dst: Int) raises:
        var pm = self._fp(m)
        var po = self._fp(dst)
        for t in range(n_rows):
            kth_smallest_row(pm, n_cols, k, po, t)

    def meanshift(
        mut self, x: Int, n: Int, d: Int, bw: Float32, stop: Float32, max_iter: Int,
        centers: Int, ns: Int, scratch: Int, intensity: Int, iters: Int,
    ) raises:
        var px = self._fp(x)
        var pc = self._fp(centers)
        var ps = self._fp(scratch)
        var pi = self._ip(intensity)
        var pt = self._ip(iters)
        for t in range(ns):
            meanshift_seed[X_CLUSTER_HOST_SABOTAGE](px, n, d, bw, stop, max_iter, pc, ps, pi, pt, t)

    def ap_r(mut self, s: Int, a: Int, r: Int, n: Int, damping: Float32) raises:
        var ps = self._fp(s)
        var pa = self._fp(a)
        var pr = self._fp(r)
        for t in range(n):
            ap_responsibility_row(ps, pa, pr, n, damping, t)

    def ap_a(mut self, r: Int, a: Int, n: Int, damping: Float32) raises:
        var pr = self._fp(r)
        var pa = self._fp(a)
        for t in range(n):
            ap_availability_col[X_CLUSTER_HOST_SABOTAGE](pr, pa, n, damping, t)

    def ap_noise(mut self, s: Int, m: Int, seed: UInt64) raises:
        var ps = self._fp(s)
        for t in range(m):
            ap_noise_cell(ps, seed, t)

    def ap_e(mut self, a: Int, r: Int, n: Int, e: Int) raises:
        var pa = self._fp(a)
        var pr = self._fp(r)
        var pe = self._ip(e)
        for t in range(n):
            ap_exemplar_cell(pa, pr, n, pe, t)

    def descend(mut self, x: Int, n: Int, d: Int, centers: Int, nodes: Int, labels: Int) raises:
        var px = self._fp(x)
        var pc = self._fp(centers)
        var pn = self._ip(nodes)
        var pl = self._ip(labels)
        for t in range(n):
            tree_descend[X_CLUSTER_HOST_SABOTAGE](px, d, pc, pn, pl, t)

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
        for t in range(n * kc):
            gauss_q_cell[X_CLUSTER_HOST_SABOTAGE](px, d, pm, pp, kc, pd, t)

    def resp(mut self, q: Int, c: Int, n: Int, kc: Int, lpn: Int) raises:
        var pq = self._fp(q)
        var pc = self._fp(c)
        var pl = self._fp(lpn)
        for t in range(n):
            resp_row(pq, pc, kc, pl, t)

    def exp(mut self, src: Int, dst: Int, n: Int) raises:
        var ps = self._fp(src)
        var pd = self._fp(dst)
        for t in range(n):
            exp_cell(ps, pd, t)

    def moments(
        mut self, resp: Int, x: Int, n: Int, d: Int, kc: Int, reg: Float32, nk: Int, means: Int, cov: Int
    ) raises:
        var pr = self._fp(resp)
        var px = self._fp(x)
        var pn = self._fp(nk)
        var pm = self._fp(means)
        var pc = self._fp(cov)
        for t in range(kc):
            nk_cell(pr, n, kc, pn, t)
        for t in range(kc * d):
            xk_cell(pr, px, n, d, kc, pn, pm, t)
        for t in range(kc * d * d):
            cov_cell(pr, px, n, d, kc, pm, pn, reg, pc, t)

    def pdist(
        mut self, a: Int, na: Int, b: Int, nb: Int, d: Int, metric: Int, p: Float32, dst: Int
    ) raises:
        var pa = self._fp(a)
        var pb = self._fp(b)
        var pd = self._fp(dst)
        for t in range(na * nb):
            pdist_cell[X_CLUSTER_HOST_SABOTAGE](pa, na, pb, nb, d, metric, p, pd, t)
