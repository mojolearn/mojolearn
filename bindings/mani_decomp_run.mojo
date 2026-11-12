# SPDX-License-Identifier: Apache-2.0
"""RUN piece of the decomp lane's manifold estimators (lane entangled-mkit,
2026-11-12): ClassicalMDS, MDS, Isomap and LocallyLinearEmbedding as typed
steps on the typed kit (bindings/mani_kit_run.mojo `MRun`), statement for
statement the steps bindings/mani_decomp_native.mojo ran on the
Python-holding MKit: the same x_decomp operations, operands, float32 scalars
and order, the Lanczos batches, the shift-invert LLE solve and SMACOF with the
same float64 stopping tests. The door (bindings/mani_decomp_native.mojo)
unpacks the settings and inputs into the typed values below, releases the GIL
around each step and packs the results; nothing here holds or calls Python.
"""
from std.math import sqrt
from bindings.mani_kit_run import (
    MRun, MK_F32P, MK_F64P,
    OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_SQRT, OP_SQ, OP_SCALE, OP_FMA, OP_ADDS, OP_GTS, OP_MIN, OP_LE,
    SEL_MAXABS, SEL_MAX, SEL_MIN,
)
from x_decomp.descriptor_schema import XD_DEV_DOWNLOAD
from x_decomp.descriptor_mani_schema import (
    XD_LANCZOS, XD_EIGH_TRIDIAG, XD_DEV_LANCZOS, XD_DEV_EIGH_TRIDIAG, XD_LLE_ITERATE, XD_MDS_FIT,
)

comptime _LANCZOS_MIN_N = 200
comptime _LANCZOS_MAX_NC = 10
comptime _LANCZOS_TOL = 1e-7
comptime _LANCZOS_MAX_M = 600
comptime _LANCZOS_DEV_MAX_FLOATS = 1 << 26
comptime _LLE_ITER_MIN_N = 200
comptime _LLE_ITER_MAX_K = 10
comptime _LLE_SUBSPACE_TOL = 1e-5
comptime _LLE_STALL_TOL = 1e-2
comptime _LLE_NULL_GUARD = 1e-3
comptime _LLE_NULL_FLOOR = 8.0
comptime _F32_EPS = 1.1920928955078125e-07

# `_metric_spec` outcomes the door resolves (the kinds 0 .. 4 of pdist, or
# the two named routes that never went through `_metric_spec`)
comptime MANI_METRIC_PRECOMPUTED = -2
comptime MANI_METRIC_EUCLIDEAN = -1

# LLE methods
comptime LLE_STANDARD = 0
comptime LLE_LTSA = 1
comptime LLE_HESSIAN = 2
comptime LLE_MODIFIED = 3

# MDS starts
comptime MDS_INIT_ARRAY = 0
comptime MDS_INIT_CLASSICAL = 1
comptime MDS_INIT_RANDOM = 2


def mani_finite(v: Float64) -> Bool:
    """`math.isfinite(v)`."""
    return v - v == 0.0


def mani_col_range(a: Int, b: Int) -> List[Int]:
    var idx = List[Int]()
    for j in range(a, b):  # native data loop (not a driver loop), no CPython call: (columns: a column range): bounded by the caller
        idx.append(j)
    return idx^


def mani_desc_range(start: Int, count: Int) -> List[Int]:
    """`list(range(start, start - count, -1))`."""
    var idx = List[Int]()
    for t in range(count):  # driver-loop: native Mojo values, no CPython call: (components: the selected positions): bounded by n_components
        idx.append(start - t)
    return idx^


def mani_dist(mut run: MRun, k: Int, a: Int, b: Int, kind: Int, pw: Float64, same: Bool) raises -> Int:
    """`_dist(k, A, B, kind, pw, same)`."""
    if kind == 0:
        var sq = run.sqdist(k, a, b)
        return run.ew(k, OP_SQRT, sq)
    var d = run.pdist(k, a, b, kind, pw)
    if same and kind == 4:
        run.fill0(k, d, 0, run.r(b) + 1, min(run.r(a), run.r(b)))
    return d


def mani_knn(mut run: MRun, k: Int, q: Int, x: Int, nn: Int, exclude_self: Bool, kind: Int,
             pw: Float64) raises -> Tuple[Int, Int]:
    """`_knn_mats(k, Q, X, nn, exclude_self, kind, pw)`."""
    var d: Int
    if kind == 0:
        d = run.sqdist(k, q, x)
    else:
        d = mani_dist(run, k, q, x, kind, pw, exclude_self)
    return run.graph_knn(k, d, nn, exclude_self)


def mani_center(mut run: MRun, k: Int, kk: Int) raises -> Tuple[Int, Int, Int]:
    """`_center_kernel(k, K)`: (Kc, column means 1 x n, total mean 1 x 1)."""
    var n = run.r(kk)
    var col = run.ew(k, OP_SCALE, run.colsum(k, kk), s=1.0 / Float64(n))
    var allm = run.ew(k, OP_SCALE, run.total(k, col), s=1.0 / Float64(n))
    var row = run.ew(k, OP_SCALE, run.rowsum(k, kk), s=1.0 / Float64(run.c(kk)))
    var t1 = run.ew(k, OP_SUB, kk, col)
    var t2 = run.ew(k, OP_SUB, t1, row)
    var kc = run.ew(k, OP_ADD, t2, allm)
    return (kc, col, allm)


def mani_kdot(mut run: MRun, k: Int, a: Int, b: Int) raises -> Float64:
    """`_kdot(k, a, b)`: float(a^T b)."""
    var p = run.mm(k, a, b, True, False)
    return run.at(p, 0)


def mani_lanczos_top(mut run: MRun, k: Int, a: Int, nc: Int) raises -> Tuple[Bool, Int, Int]:
    """`_lanczos_top(k, A, nc)`: (found, values, vectors); not found when the
    basis cap is reached unconverged (the caller then runs the exact solve)."""
    var n = run.r(a)
    var cap = min(n, _LANCZOS_MAX_M)
    if (cap + 1) * n > _LANCZOS_DEV_MAX_FLOATS:
        return Tuple[Bool, Int, Int](False, -1, -1)
    var q = run.ew(k, OP_ADDS, run.rand(k, n, 1, 0x1A2C05, 91, 0), s=-0.5)
    q = run.ew(k, OP_SCALE, q, s=1.0 / sqrt(mani_kdot(run, k, q, q)))
    # the batch's alphas and betas: a host store, made once per solve
    var ab = run.zeros(1, 2 * cap)
    var resident = run.images[k].resident
    var qd: Int
    var abd = -1
    var aid = -1
    if resident:
        qd = run.dout(k, cap + 1, n)
        run.place_rows(k, qd, run.reshape(q, 1, n), 0)
        abd = run.dout(k, 1, 2 * cap)
        aid = run.device(k, a)
    else:
        qd = run.zeros(cap + 1, n)
        run.place_rows(k, qd, run.reshape(q, 1, n), 0)
    var alphas = List[Float64]()
    var betas = List[Float64]()
    var m = min(n, max(2 * nc + 1, 20))
    var j = 0
    var stop = False
    var th = -1
    var yv = -1
    var top = List[Int]()
    while True:  # driver-loop: over Lanczos basis growth rounds (at most log2(_LANCZOS_MAX_M / 20) + 1)
        if j < m and not stop:
            # `_lanczos_batch`: steps j .. m - 1, then their alphas and betas
            # up to the first breakdown
            if resident:
                _ = run.dcall(k, XD_DEV_LANCZOS, [aid, run.did_of(qd), run.did_of(abd), n, j, m, cap])
                _ = run.hcall(k, XD_DEV_DOWNLOAD, [run.addr(ab)], [run.did_of(abd), 2 * cap])
            else:
                var pa = run.addr(a)
                _ = run.hcall(k, XD_LANCZOS, [pa, run.addr(qd), run.addr(ab)], [n, j, m, cap])
            var abp = run.ptr(ab)
            var reached = m
            var broke = False
            for jj in range(j, m):  # driver-loop: over this batch's Lanczos steps (at most _LANCZOS_MAX_M)
                var av = Float64(abp[jj])
                var bv = Float64(abp[cap + jj])
                alphas.append(av)
                betas.append(bv)
                if bv <= 1e-30 * max(1.0, abs(av)) or jj + 1 == n:
                    reached = jj + 1
                    broke = True
                    break
            j = reached
            stop = broke
        # The projected solve is the executor op `eigh_op_tridiag` on the
        # alphas / betas the Lanczos step wrote (x_decomp/lanczos_dev.mojo
        # `dev_eigh_tridiag_run`, w and z left on the device), the host
        # column its twin on `ab` (x_decomp/lanczos_host.mojo).
        if resident:
            th = run.dout(k, 1, j)
            yv = run.dout(k, j, j)
            _ = run.dcall(k, XD_DEV_EIGH_TRIDIAG, [run.did_of(abd), run.did_of(th), run.did_of(yv), j, cap])
        else:
            th = run.zeros(1, j)
            yv = run.zeros(j, j)
            _ = run.hcall(k, XD_EIGH_TRIDIAG, [run.addr(ab), run.addr(th), run.addr(yv)], [j, cap])
        top = List[Int]()
        var stop_at = max(j - 1 - nc, -1)
        var t = j - 1
        while t > stop_at:  # driver-loop: native Mojo values, no CPython call: (components: the wanted Ritz positions): bounded by n_components
            top.append(t)
            t -= 1
        if len(top) < nc:
            return Tuple[Bool, Int, Int](False, -1, -1)
        var big = run.word(k, run.reduce(k, run.take_cols(th, top), SEL_MAXABS), 0)
        if big == 0.0:
            big = 1.0
        var last = run.take_cols(run.rows(yv, j - 1, j), top)
        var res = run.word(k, run.reduce(k, run.ew(k, OP_SCALE, last, s=betas[j - 1]), SEL_MAXABS), 0)
        if stop or res <= _LANCZOS_TOL * big:
            break
        if m >= cap:
            return Tuple[Bool, Int, Int](False, -1, -1)
        m = min(min(n, 2 * m), _LANCZOS_MAX_M)
    var yt = run.take_cols(yv, top)
    var qm: Int
    if run.on_dev(qd):
        qm = run.add(-1, j, n, run.mats[qd].device, run.mats[qd].image)
    else:
        qm = run.rows(qd, 0, j)
    var v = run.mm(k, qm, yt, True, False)
    return (True, run.take_cols(th, top), v)


def mani_top_eig(mut run: MRun, k: Int, a: Int, nc: Int, topk: Bool) raises -> Tuple[Int, Int]:
    """`_top_eig(k, A, nc, topk)`: the nc largest eigenpairs (descending),
    vectors in columns, each column signed by svd_flip(u_based_decision=True)."""
    var n = run.r(a)
    var found = False
    var w = -1
    var v = -1
    if topk and n > _LANCZOS_MIN_N and nc < _LANCZOS_MAX_NC:
        var got = mani_lanczos_top(run, k, a, nc)
        found = got[0]
        w = got[1]
        v = got[2]
    if not found:
        var eig = run.eigh(k, a)
        var order = mani_desc_range(n - 1, nc)
        w = run.take_cols(eig[0], order)
        v = run.take_cols(eig[1], order)
    var signs = run.absmax_signs(k, v, True)
    return (w, run.ew(k, OP_MUL, v, signs))


# ---------------------------------------------------------------- ClassicalMDS
def cmds_core(mut run: MRun, k: Int, m: Int, metric: Int, pw: Float64, nc: Int) raises -> Tuple[Int, Int, Int]:
    """`ClassicalMDS.fit`'s steps: (the dissimilarity matrix, eigenvalues
    1 x nc, embedding_m_). `metric` is MANI_METRIC_PRECOMPUTED,
    MANI_METRIC_EUCLIDEAN or a `_metric_spec` kind (with `pw`)."""
    var d2: Int
    var dissimilarity: Int
    if metric == MANI_METRIC_PRECOMPUTED:
        d2 = run.ew(k, OP_SQ, m)
        dissimilarity = m
    elif metric == MANI_METRIC_EUCLIDEAN:
        d2 = run.sqdist(k, m, m)
        dissimilarity = run.ew(k, OP_SQRT, d2)
    else:
        var dm = mani_dist(run, k, m, m, metric, pw, True)
        d2 = run.ew(k, OP_SQ, dm)
        dissimilarity = dm
    var centered = mani_center(run, k, run.ew(k, OP_SCALE, d2, s=-0.5))
    var eig = mani_top_eig(run, k, centered[0], nc, True)
    var w = eig[0]
    var v = eig[1]
    var embedding = run.ew(k, OP_MUL, v, run.ew(k, OP_SQRT, w))
    return (dissimilarity, w, embedding)


# ---------------------------------------------------------------- MDS
struct MdsPlan(Copyable, Movable):
    """MDS.fit_transform's settings as UNPACK resolved them."""
    var metric: Int
    var pw: Float64
    var metric_mds: Bool
    var norm: Bool
    var nc: Int
    var seed: Int  # `int(seed) & 0xFFFFFFFF`
    var init: Int  # MDS_INIT_*
    var restarts: Int
    var max_iter: Int
    var eps: Float64
    # the admitted `init` array (MDS_INIT_ARRAY), kept alive by the door
    var init_address: Int
    var init_rows: Int
    var init_cols: Int
    var init_ndim: Int

    def __init__(out self):
        self.metric = MANI_METRIC_EUCLIDEAN
        self.pw = 2.0
        self.metric_mds = True
        self.norm = False
        self.nc = 2
        self.seed = 0
        self.init = MDS_INIT_RANDOM
        self.restarts = 1
        self.max_iter = 300
        self.eps = 1e-3
        self.init_address = 0
        self.init_rows = 0
        self.init_cols = 0
        self.init_ndim = 0


def mds_dissimilarity_run(mut run: MRun, k: Int, m: Int, plan: MdsPlan) raises -> Int:
    """MDS.fit_transform's dissimilarity matrix."""
    if plan.metric == MANI_METRIC_PRECOMPUTED:
        return m
    if plan.metric == MANI_METRIC_EUCLIDEAN:
        return run.ew(k, OP_SQRT, run.sqdist(k, m, m))
    return mani_dist(run, k, m, m, plan.metric, plan.pw, True)


def mds_fit_run(mut run: MRun, k: Int, dis: Int, plan: MdsPlan) raises -> Tuple[Int, Float64, Int]:
    """MDS.fit_transform's starts and SMACOF: (y, stress, n_iter)."""
    var n = run.r(dis)
    var nc = plan.nc
    var starts = List[Int]()
    if plan.init == MDS_INIT_ARRAY:
        starts.append(run.host_snapshot(plan.init_address, plan.init_rows, plan.init_cols, plan.init_ndim, "init"))
    elif plan.init == MDS_INIT_CLASSICAL:
        # ClassicalMDS(nc, metric="precomputed", numeric_mode=self.numeric_mode_).fit(Dis.out()).embedding_m_
        var got = cmds_core(run, k, run.copy(dis), MANI_METRIC_PRECOMPUTED, 0.0, nc)
        _ = run.addr(got[2])  # the inner fit's embedding_ (it reads the start back to the host)
        starts.append(got[2])
    else:
        for r in range(plan.restarts):  # driver-loop: over random restarts (n_init)
            starts.append(run.rand(k, n, nc, plan.seed, 70 + r, 0))
    if not plan.metric_mds and n * n > 2147483647:
        raise Error("MDS(metric_mds=False): " + String(n) + " rows exceed the 32-bit pair index (n * n <= 2**31 - 1)")
    var s = run.vstack(k, starts) if len(starts) > 1 else starts[0]
    var y = run.zeros(n, nc)
    var pd = run.addr(dis)
    var ps = run.addr(s)
    var stress = List[Float64](length=1, fill=0.0)
    var floats: List[Float64] = [plan.eps]
    var n_iter = run.call(k, XD_MDS_FIT, [pd, ps, run.addr(y), Int(stress.unsafe_ptr())],
                          [n, nc, len(starts), plan.max_iter, 1 if plan.metric_mds else 0, 1 if plan.norm else 0],
                          floats)
    return (y, stress[0], n_iter)


# ---------------------------------------------------------------- Isomap
struct IsoPlan(Copyable, Movable):
    var use_radius: Bool
    var radius: Float64  # `_f32(float(radius))`
    var nn: Int
    var kind: Int
    var pw: Float64
    var topk: Bool
    var nc: Int

    def __init__(out self):
        self.use_radius = False
        self.radius = 0.0
        self.nn = 5
        self.kind = 0
        self.pw = 2.0
        self.topk = True
        self.nc = 2


struct IsoFit(Movable):
    """Isomap.fit's matrices (arena indices) and the component count."""
    var d: Int
    var col: Int
    var allm: Int
    var w: Int
    var v: Int
    var em: Int
    var kc: Int
    var iv: Int
    var components: Int

    def __init__(out self):
        self.d = -1
        self.col = -1
        self.allm = -1
        self.w = -1
        self.v = -1
        self.em = -1
        self.kc = -1
        self.iv = -1
        self.components = 0


def iso_graph_run(mut run: MRun, k: Int, m: Int, plan: IsoPlan) raises -> Tuple[Int, Int]:
    """`Isomap._graph`: (the dense neighbor graph, its component count)."""
    var n = run.r(m)
    var wg: Int
    if plan.use_radius:
        var d = mani_dist(run, k, m, m, plan.kind, plan.pw, True)
        wg = run.graph_radius(k, d, plan.radius)
    else:
        var nn = plan.nn
        if nn >= n:
            # scikit-learn's kneighbors refusal (the query point itself is
            # one of the n_neighbors + 1 it asks for)
            raise Error("Expected n_neighbors <= n_samples_fit, but n_neighbors = " + String(nn + 1)
                        + ", n_samples_fit = " + String(n) + ", n_samples = " + String(n))
        var got = mani_knn(run, k, m, m, nn, True, plan.kind, plan.pw)
        var idx = got[0]
        var sq = got[1]
        if plan.kind == 0:
            sq = run.ew(k, OP_SQRT, sq)
        wg = run.graph_knn_dense(k, idx, sq, n)
    var comp = run.graph_components(k, wg)
    var count = comp[1]
    if count > 1:
        # the door warns (UserWarning) after this step, before publishing
        var d = mani_dist(run, k, m, m, plan.kind, plan.pw, True)
        run.graph_join(k, wg, d, comp[0], count)
    return (wg, count)


def iso_fit_run(mut run: MRun, k: Int, m: Int, plan: IsoPlan) raises -> IsoFit:
    var out = IsoFit()
    var graph = iso_graph_run(run, k, m, plan)
    out.components = graph[1]
    out.d = run.dijkstra(k, graph[0])
    var g = run.ew(k, OP_SCALE, run.ew(k, OP_SQ, out.d), s=-0.5)
    var centered = mani_center(run, k, g)
    out.kc = centered[0]
    out.col = centered[1]
    out.allm = centered[2]
    var eig = mani_top_eig(run, k, out.kc, plan.nc, plan.topk)
    out.w = eig[0]
    out.v = eig[1]
    out.em = run.ew(k, OP_MUL, eig[1], run.ew(k, OP_SQRT, eig[0]))
    # transform's eigenvectors / sqrt(eigenvalues) (row N295), made once here
    out.iv = run.ew(k, OP_DIV, eig[1], run.ew(k, OP_SQRT, eig[0]))
    return out^


struct IsoState(Copyable, Movable):
    """Isomap.transform's fitted matrices (arena indices; `iv` -1 for a model
    fitted before `_iso_v_m_` was kept) and settings."""
    var fit_x: Int
    var d: Int
    var col: Int
    var allm: Int
    var iv: Int
    var w: Int
    var v: Int
    var knn: Int  # n_neighbors, -1 for a radius model
    var radius: Float64
    var radius_text: String
    var kind: Int
    var pw: Float64

    def __init__(out self):
        self.fit_x = -1
        self.d = -1
        self.col = -1
        self.allm = -1
        self.iv = -1
        self.w = -1
        self.v = -1
        self.knn = -1
        self.radius = 0.0
        self.radius_text = String("")
        self.kind = 0
        self.pw = 2.0


def iso_transform_run(mut run: MRun, k: Int, q: Int, st: IsoState) raises -> Int:
    """`Isomap.transform(X)`: the embedding of q (an arena matrix)."""
    var n = run.r(st.fit_x)
    var g = -1
    if st.knn < 0:
        # `_radius_geodesic`
        var r = st.radius
        var dq = mani_dist(run, k, q, st.fit_x, st.kind, st.pw, False)
        var within = run.ew(k, OP_LE, dq, run.of([r], 1, 1))
        var hits = run.ew(k, OP_GTS, run.rowsum(k, within), s=0.5)
        var qr = run.r(q)
        if qr > 0 and run.at(run.total(k, hits), 0) != Float64(qr):
            var hp = run.ptr(hits)
            var first = 0
            for i in range(run.n(hits)):  # cpu-route: the refusal names the first query row without a training row in reach
                if Float64(hp[i]) == 0.0:
                    first = i
                    break
            raise Error("Isomap.transform: query row " + String(first) + " has no training row within radius "
                        + st.radius_text)
        g = run.graph_radius_geo(k, dq, st.d, r)
    else:
        var nn = st.knn
        # the neighbour matrices as they come (indices as exact floats),
        # each neighbour's rows of D gathered on the device
        var got = mani_knn(run, k, q, st.fit_x, nn, False, st.kind, st.pw)
        var im = got[0]
        var sq = got[1]
        if st.kind == 0:
            sq = run.ew(k, OP_SQRT, sq)
        # A resident image makes the whole gather-and-min in one native
        # operation (row N295): the same launches, in the same order.
        var fused = run.iso_gather_min(k, st.d, im, sq, run.r(q), nn)
        if fused >= 0:
            g = fused
        else:
            var running = -1
            for a in range(nn):  # driver-loop: over neighbor ranks (n_neighbors), one gather-and-min step each (host image)
                var rows = run.take_rows_at(k, st.d, im, run.r(q), nn, a, 0)
                var cand = run.ew(k, OP_ADD, rows, run.take_cols(sq, mani_col_range(a, a + 1)))
                if running < 0:
                    running = cand
                else:
                    running = run.ew(k, OP_MIN, running, cand)
            if running >= 0:
                g = running
    if g < 0:
        raise Error("RuntimeError: Isomap.transform: no neighbor rank to gather")
    g = run.ew(k, OP_SCALE, run.ew(k, OP_SQ, g), s=-0.5)
    var row = run.ew(k, OP_SCALE, run.rowsum(k, g), s=1.0 / Float64(n))
    var t1 = run.ew(k, OP_SUB, g, st.col)
    var t2 = run.ew(k, OP_SUB, t1, row)
    var kc = run.ew(k, OP_ADD, t2, st.allm)
    # eigenvectors / sqrt(eigenvalues), kept by fit (models fitted before it
    # was kept make it here, as every transform did)
    var v = st.iv
    if v < 0:
        v = run.ew(k, OP_DIV, st.v, run.ew(k, OP_SQRT, st.w))
    return run.mm(k, kc, v, False, False)


def iso_reconstruction_error_run(mut run: MRun, k: Int, kc: Int, w: Int) raises -> Float64:
    """`Isomap.reconstruction_error()`: a difference of two nearly equal sums,
    accumulated in binary64 (`_dsum_sq`)."""
    var t = run.dsum_sq(k, kc) - run.dsum_sq(k, w)
    if t > 0:
        return sqrt(t) / Float64(run.r(kc))
    return 0.0


# ---------------------------------------------------------------- LocallyLinearEmbedding
struct LlePlan(Copyable, Movable):
    var method: Int  # LLE_*
    var nn: Int
    var nc: Int
    var reg: Float64
    var tol: Float64  # hessian_tol / modified_tol for those methods
    var auto_solver: Bool  # eigen_solver == 'auto'
    var max_iter: Int
    var seed: Int  # `int(_seed_of(random_state)) & 0xFFFFFFFF`

    def __init__(out self):
        self.method = LLE_STANDARD
        self.nn = 5
        self.nc = 2
        self.reg = 1e-3
        self.tol = 0.0
        self.auto_solver = True
        self.max_iter = 100
        self.seed = 0


struct LleFit(Movable):
    """LocallyLinearEmbedding.fit's outcome. `refusal` 1: the subspace
    iteration did not settle (`change` its last subspace change); 2: it
    returned columns of squared norm `norms` (not unit). The door words the
    refusal (it formats Python floats)."""
    var em: Int
    var reconstruction_error: Float64
    var refusal: Int
    var change: Float64
    var norms: List[Float64]

    def __init__(out self):
        self.em = -1
        self.reconstruction_error = 0.0
        self.refusal = 0
        self.change = 0.0
        self.norms = List[Float64]()


def lle_smallest_run(mut run: MRun, k: Int, f: Int, nc: Int, max_iter: Int, seed: Int,
                     mut outcome: LleFit) raises -> Tuple[Bool, Int, Int]:
    """`_lle_smallest(k, F, nc, max_iter, seed)`: (found, embedding, sv) by
    the shift-invert subspace iteration on the deflated F0; not found sends
    the caller to the full SVD. A refusal is left in `outcome`."""
    var n = run.c(f)
    if run.r(f) != n:
        return Tuple[Bool, Int, Int](False, -1, -1)
    var n1 = n - 1
    var p = min(n1, max(2 * nc + 1, 20))
    if p <= nc:
        return Tuple[Bool, Int, Int](False, -1, -1)
    var rn = 1.0 / sqrt(Float64(n))
    var hv = List[Float64](length=n, fill=rn)
    hv[n - 1] = rn - 1.0
    var coef = 2.0 / (Float64(n - 1) * (rn * rn) + (rn - 1.0) * (rn - 1.0))
    var h = run.of(hv, n, 1)
    var un = run.fill(rn, n, 1)
    var hrow = run.fill(rn, 1, n1)
    var fh = run.mm(k, f, h, False, False)
    var hfull_v = List[Float64](length=n, fill=rn)
    hfull_v[n - 1] = 0.0
    var mask_v = List[Float64](length=n, fill=1.0)
    mask_v[n - 1] = 0.0
    var last_v = List[Float64](length=n, fill=0.0)
    last_v[n - 1] = rn
    var hfull = run.of(hfull_v, 1, n)
    var mask = run.of(mask_v, 1, n)
    var last = run.of(last_v, 1, n)
    var dd = run.ew(k, OP_SUB, f, run.ew(k, OP_SCALE, run.mm(k, fh, hfull, False, False), s=coef))
    var f0 = run.ew(k, OP_FMA, dd, mask, last)
    var fu = run.mm(k, f, un, False, False)
    var g = sqrt(run.dsum_sq(k, fu))
    var tot = run.at(run.total(k, run.ew(k, OP_SQ, f)), 0)
    # max(tot, 0.0): Python's order (tot unless 0.0 is larger)
    var kept = Float64(0.0) if 0.0 > tot else tot
    var rms = sqrt(kept / Float64(n))
    if not (g <= _LLE_NULL_GUARD * rms):
        return Tuple[Bool, Int, Int](False, -1, -1)
    var floor = _LLE_NULL_FLOOR * _F32_EPS * rms
    var fac = run.lu(k, f0)
    var lu = fac[0]
    var aux = run.lu_aux(k, lu, fac[1], True)
    var big = aux[0][0]
    var pm = aux[2]
    var im = aux[3]
    if not (big > 0.0 and mani_finite(big)):
        return Tuple[Bool, Int, Int](False, -1, -1)
    var en = run.zeros(n, 1)
    run.ptr(en)[n - 1] = 1.0
    var z = run.trisolve(k, lu, im, en, 1)
    var zn = sqrt(run.dsum_sq(k, z))
    if not zn > 0.0 or not mani_finite(zn):
        return Tuple[Bool, Int, Int](False, -1, -1)
    z = run.ew(k, OP_SCALE, z, s=1.0 / zn)
    # `_lle_iterate`
    var xm = run.zeros(n1, p)
    var ym = run.zeros(n1, nc)
    var sm = run.zeros(1, p)
    var change = List[Float64](length=1, fill=0.0)
    var tols: List[Float64] = [floor, _LLE_SUBSPACE_TOL, _LLE_STALL_TOL]
    var pf = run.addr(f0)
    var pl = run.addr(lu)
    var pp = run.addr(pm)
    var pi = run.addr(im)
    var pz = run.addr(z)
    var status = run.call(k, XD_LLE_ITERATE, [pf, pl, pp, pi, pz, run.addr(xm), run.addr(ym), run.addr(sm),
                                              Int(change.unsafe_ptr())], [n, n1, nc, p, max(1, max_iter), seed], tols)
    if status == 2:
        outcome.refusal = 1
        outcome.change = change[0]
        return Tuple[Bool, Int, Int](False, -1, -1)
    var sv = run.take_cols(sm, mani_desc_range(p - 1, nc))
    # back to R^n: H [Y; 0] = [Y; 0] - coef h (h^T [Y; 0])
    var t = run.mm(k, hrow, ym, False, False)
    var full = run.pad_zero_row(k, ym)
    var v = run.ew(k, OP_SUB, full, run.ew(k, OP_SCALE, run.mm(k, h, t, False, False), s=coef))
    var sq = run.colsum(k, run.ew(k, OP_SQ, v))
    var lo = run.word(k, run.reduce(k, sq, SEL_MIN), 0)
    var hi = run.word(k, run.reduce(k, sq, SEL_MAX), 0)
    if not (0.9 <= lo and hi <= 1.1) or not mani_finite(run.word(k, run.reduce(k, sv, SEL_MAXABS), 0)):
        var sp = run.ptr(sq)
        for i in range(run.n(sq)):  # driver-loop: over (components: the refusal lists each column's squared norm): bounded by n_components
            outcome.norms.append(Float64(sp[i]))
        outcome.refusal = 2
        return Tuple[Bool, Int, Int](False, -1, -1)
    var signs = run.absmax_signs(k, v, True)
    return (True, run.ew(k, OP_MUL, v, signs), sv)


def lle_fit_run(mut run: MRun, k: Int, hk: Int, m: Int, plan: LlePlan) raises -> LleFit:
    """`LocallyLinearEmbedding.fit(X)` after X is admitted (`m`) and the
    door's n_neighbors / method refusals."""
    var out = LleFit()
    var n = run.r(m)
    # n_neighbors / method refusals: the door's UNPACK, after X
    var nn = plan.nn
    var nc = plan.nc
    var knn = mani_knn(run, k, m, m, nn, True, 0, 2.0)
    var idm = knn[0]
    var iw: Int
    if plan.method == LLE_STANDARD:
        var wb = run.barycenter(k, m, m, idm, plan.reg)
        iw = run.graph_lle_iw(k, idm, wb, n)
    elif plan.method == LLE_LTSA:
        iw = run.lle_local(k, m, idm, 0, nn, nc, 0.0)
    elif plan.method == LLE_HESSIAN:
        iw = run.lle_local(k, m, idm, 1, nn, nc, plan.tol)
    else:
        iw = run.lle_local(k, m, idm, 2, nn, nc, plan.tol)
    var found = False
    var em = -1
    var sv = -1
    if plan.method == LLE_STANDARD and plan.auto_solver and n > _LLE_ITER_MIN_N and nc + 1 < _LLE_ITER_MAX_K:
        run.enable_lle_precision(k)
        var got = lle_smallest_run(run, k, iw, nc, plan.max_iter, plan.seed, out)
        if out.refusal != 0:
            return out^
        found = got[0]
        em = got[1]
        sv = got[2]
    if not found:
        var dec = run.svd(k, iw, hk)
        var rows = mani_desc_range(n - 2, nc)
        em = run.t(run.take_rows_list(dec[1], rows, hk), hk)
        sv = run.take_cols(dec[0], rows)
    out.em = em
    out.reconstruction_error = run.dsum_sq(k, sv)
    return out^


def lle_transform_run(mut run: MRun, k: Int, q: Int, fit_x: Int, embedding: Int, nn: Int, reg: Float64) raises -> Int:
    """`LocallyLinearEmbedding.transform(X)`."""
    var knn = mani_knn(run, k, q, fit_x, nn, False, 0, 2.0)
    var idm = knn[0]
    var wb = run.barycenter(k, q, fit_x, idm, reg)
    return run.lle_apply(k, wb, idm, embedding)
