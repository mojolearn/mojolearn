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
from x_cluster.bodies import SplitMix64


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

    def flush(mut self, x: Int, n: Int) raises:
        """x[i] = ftz(x[i]) for the first n values (`bodies.flush_cell`)."""
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

    def ap_noise(mut self, s: Int, m: Int, seed: UInt64) raises:
        """The tie noise on the first m cells of S (`bodies.ap_noise_cell`)."""
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

    def gather_rows(mut self, src: Int, d: Int, idx: Int, m: Int, dst: Int) raises:
        """dst[t * d + f] = src[idx[t] * d + f], t < m (lane/neural-pass108)."""
        ...

    def kmeans_rows(
        mut self, sub: Int, x: List[Float32], rows: List[Int], d: Int, k: Int, max_iter: Int,
        tol: Float64, seed: UInt64, n_init: Int, init: Int, mut centers: List[Float32],
        mut labels: List[Int32],
    ) raises -> Float64:
        """`kmeans` (unit weights) of the rows `rows` of the host matrix `x`
        in that order, whose gathered copy is the slot `sub` (the device
        fits it in place; the host gathers `x`). The same words as `kmeans`
        on the gathered list (lane/neural-pass108)."""
        ...

    def shrink(mut self, slot: Int) raises:
        """Releases a float slot's storage (its index stays valid, one word)."""
        ...

    def empty(mut self, n: Int) raises -> Int:
        """A float slot of n words with no defined contents (the caller
        writes every word before reading one)."""
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

    def argmax_rows(mut self, src: Int, n: Int, kc: Int, labels: Int) raises:
        """labels (int, n) = each row's `bodies.argmax_row` of src (n x kc)."""
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

    def fast_device(self) -> Bool:
        """True on the one column that takes the FAST device-round paths
        (lane cluster-apple3): the GPU binding built FAST. The host column
        and every IDENTICAL build answer False."""
        ...

    def ward_nn(mut self, c: Int, sz: Int, l: Int, d: Int, nn: Int, md: Int) raises:
        """FAST ward: nn[p], md[p] = the cluster q != p among the first `l`
        (centroids c, l x d; sizes sz) at the lowest `bodies.ward_cell`, the
        lowest q on a tie."""
        ...

    def kth_flat(mut self, m: Int, n: Int, k: Int) raises -> Float32:
        """The k-th smallest (1-based) of the first `n` values of slot `m`
        (non-negative), the value `kth` gives for one row of n columns. On
        the device a radix select over every block of the grid (lane
        cluster-apple3: `kth` runs a row on ONE block)."""
        ...

    def get_diag(mut self, slot: Int, n: Int) raises -> List[Float32]:
        """The n diagonal values of an n x n float slot."""
        ...

    def ap_a_split(mut self, r: Int, a: Int, n: Int, damping: Float32) raises:
        """FAST: `ap_a` with every column sum folded over row slices (another
        summation order than `bodies.ap_availability_col`)."""
        ...

    def dot_groups(mut self, a: Int, b: Int, n: Int, g: Int, parts: Int) raises:
        """FAST: parts[q] = the sum of a[t] * b[t] over the q-th run of `g`
        consecutive cells of the first n (ceil(n / g) values)."""
        ...

    def alloc(mut self, n: Int) raises -> Int:
        """A new float slot of `n` values that a primitive is about to fill
        completely (`sqdist`, `pdist`): on the device NOT initialized, where
        `zeros` writes n zeros first; on the host `zeros`."""
        ...

    def estep(
        mut self, x: Int, n: Int, d: Int, means: Int, pchol: Int, c: Int, kc: Int, q: Int, r: Int, lpn: Int
    ) raises:
        """`gauss_q`, `resp` and `exp` of one E-step as ONE primitive: row i's
        kc Mahalanobis squares, its log-sum-exp and its responsibilities by
        the same bodies in the same order (the same values)."""
        ...

    def optics_order_fast(
        mut self, dm: Int, core: Int, n: Int, max_eps: Float32, ordering: Int, reach: Int, pred: Int, proc: Int
    ) raises:
        """FAST (lane cluster2): the OPTICS ordering loop over the resident
        n x n distances `dm` and core distances `core`: ordering (n ints),
        reachability (n floats, +inf unreached), predecessor (n ints, -1
        none); `proc` n ints of scratch. The host loop's picks and updates."""
        ...

    def minibatch_fast(
        mut self, xs: Int, n: Int, d: Int, k: Int, batch: Int, n_steps: Int, max_no_improvement: Int,
        ratio: Float64, seed: UInt64, mut rng: SplitMix64, mut c: List[Float32], mut w: List[Float32],
        mut steps_done: Int,
    ) raises -> Bool:
        """FAST on Apple (lane/apple-fast-cluster): `minibatch_fit`'s step
        loop resident on the device (x_cluster/minibatch_fast.mojo); `c`,
        `w` in and out. False when the column does not take it (the host,
        every IDENTICAL build, a shape past its caps): the caller runs the
        step loop."""
        ...

    def set_i(mut self, slot: Int, v: List[Int32]) raises:
        """Writes v into the first len(v) words of the int slot (lane/neural-pass133)."""
        ...

    def mb_update(mut self, b: Int, batch: Int, labels: Int, c: Int, w: Int, k: Int, d: Int) raises:
        """MiniBatchKMeans' `update_center_dense` with unit weights for every
        center, in place: the centers `c` (k x d) and counts `w` (k) from the
        batch rows `b` (batch x d) and their labels, each center's chain the
        host loop's (`c * w`, `+ x` in batch order, `w += wsum`, `* (1 / w)`),
        a center without rows untouched (lane/neural-pass133)."""
        ...

    def mb_assign(mut self, src: Int, d: Int, idx: Int, m: Int, c: Int, k: Int, labels: Int, dist: Int, dst: Int) raises:
        """`gather_rows(src, d, idx, m, dst)` then `nearest(dst, m, c, k, d,
        labels, dist)`: the device fuses them into one launch (the same words;
        lane/neural-pass133)."""
        ...

    def mb_draw(mut self, idx: Int, m: Int, n: Int, state: UInt64) raises:
        """idx[t] = draw t + 1 of the splitmix64 stream whose state is
        `state`, `% n`, t < m: `SplitMix64.below(n)` m times, by counter
        (fam2-cluster)."""
        ...

    def fold_at(mut self, a: Int, n: Int, mode: Int, dst: Int, off: Int, th: Int, tl: Int) raises:
        """`fold_into` of the one slot `a` (FM_VAL, FM_SQRT: the modes that
        read `a` alone) left in dst[off], dst[off + 1]; `th`, `tl` float
        slots of `minibatch.mb_fold_scratch(n)` words the device's levels
        use (the host ignores them)."""
        ...

    def copy_at(mut self, src: Int, n: Int, dst: Int, off: Int) raises:
        """dst[off + t] = src[t], t < n (float slots)."""
        ...

    def dist_sel(mut self, a: Int, n: Int, c: Int, d: Int, lab: Int, j: Int, dst: Int) raises:
        """dst[t] = `bodies.sq_dist_rows` of row t of `a` to row lab[t] of
        `c` where j < 0 or lab[t] == j, else 0 (t < n; `lab` an int slot):
        each row's distance to its OWN center (fam2-cluster)."""
        ...

    def agglo_on_device(self) -> Bool:
        """True on the GPU column: `agglo_merge` runs the unconstrained
        agglomerative merge loop on the device (lane hr2-mds-agglo). The
        host column answers False and `agglo.agglo_tree` runs its loop."""
        ...

    def agglo_mirror(mut self, x: Int, n: Int, dst: Int) raises:
        """dst (n x n) = the precomputed matrix x's UPPER triangle mirrored
        below the diagonal, a zero diagonal; raises unless every upper value
        is finite and non-negative."""
        ...

    def agglo_merge(
        mut self, dm: Int, adj: Int, n: Int, linkage: Int, n_merges: Int, mut children: List[Int32],
        mut dist: List[Float32],
    ) raises:
        """The first n_merges merges of the unconstrained agglomerative loop
        (`agglo.agglo_tree`'s order: the live row with the lowest nearest
        value, the lowest row on a tie; its nearest partner, the lowest
        column on a tie; Lance-Williams by `bodies.lance_williams`) on the
        n x n dissimilarity slot `dm`, which it overwrites. `adj` >= 0: the
        int slot (n x n, 0/1) of a connectivity graph from `agglo_connect`
        (the constrained loop: only connected pairs merge; the
        Lance-Williams value of a pair neither child touched is kept unless
        ward; the merged cluster takes the union of the edges)."""
        ...

    # ------------------------------------------------------------------
    # THE n-SIZED POST-PROCESSING (lane cgr2-cluster, 2026-10-03): the parts
    # of OPTICS, MeanShift, AffinityPropagation, the mixtures and k-means++
    # that ran on the host between device calls. Each is one primitive; the
    # host column runs the same decisions in loops (the bodies in
    # `x_cluster/post_bodies.mojo`), sums are its float-float fold.
    def agglo_connect(
        mut self, edges: Int, n_edges: Int, n: Int, dm: Int, linkage: Int, adj: Int, edge_mode: Int
    ) raises -> Int:
        """GPU column: adj (int slot, n x n) = the connectivity graph of the
        n_edges (row, col) float pairs in `edges` (edge_mode 0; 1: `edges` is
        the dense n x n matrix, n_edges = n * n; 2: the COO rows, columns and
        values concatenated, n_edges entries each; a nonzero entry is an edge),
        symmetrized, the diagonal
        dropped; with several components each pair of components joined at
        its closest pair (`agglo.agglo_tree`'s rule); returns the number of
        components. The host column runs agglo_tree's loop."""
        ...

    # ------------------------------------------------------------------
    # THE TREE CUT (lane apple-fast-py2mojo-cluster): the labels of an
    # agglomerative tree, which `_hierarchy_impl.py` computed in Python.
    def tree_parent(mut self, children: Int, n: Int, m: Int, parent: Int) raises:
        """Int slot `parent` (n + m) = each node's parent, a root its own:
        merge t of the int slot `children` (m x 2) is node n + t."""
        ...

    def tree_roots(mut self, parent: Int, total: Int, rank1: Int) raises -> Int:
        """Int slot `rank1` (total) = 1 + the number of roots below j for
        every root j (parent[j] == j), 0 elsewhere; returns the root count."""
        ...

    def tree_scatter(mut self, nodes: Int, c: Int, rank1: Int) raises:
        """rank1[nodes[i]] = i + 1 for the c ids of the int slot `nodes`."""
        ...

    def tree_leaf_label(mut self, parent: Int, rank1: Int, n: Int, labels: Int) raises:
        """Int slot `labels` (n): leaf t's first ancestor-or-self v with
        rank1[v] != 0 gives rank1[v] - 1."""
        ...

    def count_ge(mut self, x: Int, n: Int, thr: Float32) raises -> Int:
        """The number of the first n values that are >= thr."""
        ...

    def check_nonneg(mut self, x: Int, n: Int) raises -> Bool:
        """Every one of the first n values is >= 0 (a NaN is not)."""
        ...

    def optics_order(
        mut self, dm: Int, core: Int, n: Int, max_eps: Float32, ordering: Int, reach: Int, pred: Int
    ) raises:
        """`core` > max_eps becomes +inf in place; then the OPTICS ordering
        over the n x n distances `dm`: each step the unprocessed row of the
        lowest reachability (the lowest index on a tie), whose unprocessed
        neighbours take `post_bodies.optics_relax_cell`. Int slots
        `ordering`, `pred`; float slot `reach`."""
        ...

    def optics_dbscan(mut self, ordering: Int, reach: Int, core: Int, n: Int, eps: Float32, labels: Int) raises:
        """`cluster_optics_dbscan` into the int slot `labels`."""
        ...

    def optics_xi(
        mut self, ordering: Int, reach: Int, pred: Int, n: Int, xc: Float32, min_samples: Int,
        min_cluster_size: Int, predecessor_correction: Bool, labels: Int,
    ) raises -> List[Int32]:
        """`_xi_cluster` + `_extract_xi_labels` over the fitted ordering,
        reachability and predecessors (slots): labels (point order) into the
        int slot `labels`; returns the clusters (start, end) flattened
        (`x_cluster/optics_xi_cells.mojo`, both columns)."""
        ...

    def sum_ff(mut self, a: Int, b: Int, c: Int, n: Int, mode: Int) raises -> Float64:
        """The float-float fold (`post_bodies`) of n elements of `mode` over
        slots a, b, c (-1 when unused), as a double."""
        ...

    def fold_into(mut self, a: Int, b: Int, c: Int, n: Int, mode: Int, dst: Int) raises:
        """`sum_ff`'s fold left where it is: (hi, lo) into dst[0], dst[1]."""
        ...

    def bgmm_step(
        mut self, step: Int, kc: Int, d: Int, cfg: Int, aux: Int, w: Int, p1: Int, p2: Int, p3: Int
    ) raises:
        """One step of the mixtures' k-sized work on the workspace slot w
        (`x_cluster/bgmm_device.mojo`; p1..p3 -1 when unused)."""
        ...

    def bin_seeds(mut self, x: Int, n: Int, d: Int, bin_size: Float32, min_bin_freq: Int, dst: Int) raises -> Int:
        """sklearn `get_bin_seeds`: the kept bins (first-seen order) scaled
        back into `dst` (n x d); returns their count (n: use the rows)."""
        ...

    def ms_unique(
        mut self, centers: Int, inten: Int, iters: Int, ns: Int, d: Int, dst: Int, mut n_iter: Int
    ) raises -> Int:
        """MeanShift: the distinct centers with a nonzero intensity (the last
        seed's intensity each), sorted by (intensity, coordinates) descending,
        into `dst`; returns their count; n_iter = the most iterations."""
        ...

    def ms_suppress(mut self, sorted: Int, dd: Int, m: Int, d: Int, bw: Float32, dst: Int) raises -> Int:
        """MeanShift's radius suppression over the m sorted centers (their
        distances `dd`, m x m): the kept ones in order into `dst`; their count."""
        ...

    def ms_noise(mut self, labels: Int, dist: Int, n: Int, bw: Float32) raises:
        """labels[r] = -1 where not dist[r] <= bw."""
        ...

    def negate(mut self, src: Int, dst: Int, n: Int) raises:
        ...

    def count_neg(mut self, x: Int, n: Int) raises -> Int:
        """How many of the first n values are < 0."""
        ...

    def sign_side(mut self, src: Int, n: Int, neg: Bool, dst: Int) raises:
        """neg: dst = -v where v < 0, else +inf; not neg: dst = v where
        v >= 0, else +inf (the inputs of `kth_flat` on one side of zero)."""
        ...

    def ap_equal(mut self, s: Int, pref: Int, n: Int) raises -> Bool:
        """Every off-diagonal value of S equals the first one and every
        preference equals the first."""
        ...

    def set_diag(mut self, s: Int, v: Int, n: Int) raises:
        ...

    def ap_conv(mut self, e: Int, ring: Int, n: Int, conv_iter: Int, it: Int) raises -> Bool:
        """The convergence window: e into column it % conv_iter of ring
        (n x conv_iter); True when it >= conv_iter, every row's window is
        all ones or all zeros, and some e is 1."""
        ...

    def ap_loop(
        mut self, s: Int, a: Int, r: Int, e: Int, ring: Int, n: Int, damping: Float32, max_iter: Int,
        conv_iter: Int,
    ) raises -> Int:
        """The whole message loop (`ap_r`, `ap_a`, `ap_e`, `ap_conv` per
        iteration) with the convergence window decided where the data is
        (fam2-cluster): returns the iteration it converged at (the `it` of
        `affinity_fit`'s break), `max_iter` when it never did, or -1 when the
        column does not take it (the host, FAST, the `_OFF` define): the
        caller runs the loop."""
        ...

    def ap_exemplars(mut self, s: Int, e: Int, n: Int, centers: Int, labels: Int) raises -> Int:
        """AffinityPropagation's exemplar refinement and labels from the
        flags `e`; returns the number of centers (0: every label -1)."""
        ...

    def onehot(mut self, idx: Int, m: Int, kc: Int, by_row: Bool, dst: Int) raises:
        """by_row: dst[q * kc + idx[q]] = 1 (labels); else dst[idx[q] * kc
        + q] = 1 (picks); q < m; the rest of dst untouched."""
        ...

    def rand_resp(mut self, dst: Int, n: Int, kc: Int, state: UInt64) raises:
        """`post_bodies.rand_resp_row` for every row."""
        ...

    def kpp_search(mut self, closest: Int, w: Int, m: Int, vs: List[Float64], ids: Int) raises:
        """ids[t] = `post_bodies.kpp_search_cell` of vs[t] over the running
        table of closest (times w when w >= 0)."""
        ...

    def kpp_pots(mut self, dc: Int, closest: Int, w: Int, nt: Int, m: Int) raises -> List[Float64]:
        """Each candidate row t of dc (nt x m): the fold of min(dc, closest)
        (times w when w >= 0)."""
        ...

    def kpp_take(mut self, dc: Int, closest: Int, best: Int, m: Int) raises:
        """closest = min(closest, dc row best)."""
        ...
