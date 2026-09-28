# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE CLUSTER LANE'S TWO-COLUMN SEAM (lane/algos-cluster).

Every fit in `x_cluster/` is ONE generic driver over this trait. The GPU
binding instantiates it with `x_cluster.device_ops.DeviceOps` (buffers on the
device, each primitive one kernel); the CPU host binding with
`x_cluster.host.host_ops.HostOps` (the same bodies in plain loops). The driver
itself (sampling, sorting, the sequential loops) is host code compiled once per
binding from the same source. Arrays live in numbered SLOTS so the device keeps
them resident between primitives.
"""


trait ClusterOps(Movable):
    def put(mut self, v: List[Float32]) raises -> Int:
        """A new float slot holding `v`."""
        ...

    def put_i(mut self, v: List[Int32]) raises -> Int:
        ...

    def zeros(mut self, n: Int) raises -> Int:
        ...

    def zeros_i(mut self, n: Int) raises -> Int:
        ...

    def get(mut self, slot: Int, n: Int) raises -> List[Float32]:
        """The first `n` values of a float slot."""
        ...

    def get_i(mut self, slot: Int, n: Int) raises -> List[Int32]:
        ...

    def gets(mut self, slots: List[Int], ns: List[Int]) raises -> List[List[Float32]]:
        """The first ns[q] values of each float slot slots[q], read together
        (the device waits once)."""
        ...

    def get_if(
        mut self, islot: Int, ni: Int, fslot: Int, nf: Int, mut oi: List[Int32], mut of: List[Float32]
    ) raises:
        """`get_i(islot, ni)` and `get(fslot, nf)` read together."""
        ...

    def set(mut self, slot: Int, v: List[Float32]) raises:
        """Overwrite the first len(v) values of a float slot."""
        ...

    def sqdist(mut self, a: Int, na: Int, b: Int, nb: Int, d: Int, dst: Int) raises:
        """out[i, j] = squared distance of a[i] and b[j] (`bodies.sqdist_cell`)."""
        ...

    def nearest(mut self, a: Int, na: Int, b: Int, nb: Int, d: Int, labels: Int, dist: Int) raises:
        """labels[i], dist[i] = nearest row of b to a[i] (`bodies.nearest_row`)."""
        ...

    def sqrt(mut self, x: Int, n: Int) raises:
        ...

    def kth(mut self, m: Int, n_rows: Int, n_cols: Int, k: Int, dst: Int) raises:
        """out[r] = k-th smallest of row r (`bodies.kth_smallest_row`)."""
        ...

    def meanshift(
        mut self, x: Int, n: Int, d: Int, bw: Float32, stop: Float32, max_iter: Int,
        centers: Int, ns: Int, scratch: Int, intensity: Int, iters: Int,
    ) raises:
        ...

    def ap_r(mut self, s: Int, a: Int, r: Int, n: Int, damping: Float32) raises:
        ...

    def ap_a(mut self, r: Int, a: Int, n: Int, damping: Float32) raises:
        ...

    def ap_e(mut self, a: Int, r: Int, n: Int, e: Int) raises:
        """e[i] = A[i, i] + R[i, i] > 0 (`bodies.ap_exemplar_cell`)."""
        ...

    def descend(mut self, x: Int, n: Int, d: Int, centers: Int, nodes: Int, labels: Int) raises:
        """labels[i] = the leaf row i reaches (`bodies.tree_descend`)."""
        ...

    def kmeans(
        mut self, x: List[Float32], n: Int, d: Int, k: Int, max_iter: Int, tol: Float64,
        seed: UInt64, n_init: Int, init: Int, mut centers: List[Float32], mut labels: List[Int32],
        weights: List[Float32] = List[Float32](),
    ) raises -> Float64:
        """The repository's k-means (cuVS's fit_predict): `cluster/estimator.mojo::
        kmeans_fit` on the device, `cluster/host/kmeans_oracle.mojo::
        host_kmeans_fit` on the host, the pair the kmeans identity lanes hold
        bit for bit. Returns the inertia; `centers` k x d and `labels` n."""
        ...

    def gauss_q(mut self, x: Int, n: Int, d: Int, means: Int, pchol: Int, kc: Int, dst: Int) raises:
        """dst (n x kc) = the Mahalanobis squares (`bodies.gauss_q_cell`)."""
        ...

    def resp(mut self, q: Int, c: Int, n: Int, kc: Int, lpn: Int) raises:
        """In place: q -> log responsibilities; lpn = the row log-sum-exp
        (`bodies.resp_row`)."""
        ...

    def exp(mut self, src: Int, dst: Int, n: Int) raises:
        ...

    def moments(
        mut self, resp: Int, x: Int, n: Int, d: Int, kc: Int, reg: Float32, nk: Int, means: Int, cov: Int
    ) raises:
        """nk (kc), means (kc x d), cov (kc x d x d) of `resp` (n x kc)."""
        ...

    def pdist(
        mut self, a: Int, na: Int, b: Int, nb: Int, d: Int, metric: Int, p: Float32, dst: Int
    ) raises:
        """dst (na x nb) = the metric's distances (`bodies.pdist_cell`)."""
        ...
