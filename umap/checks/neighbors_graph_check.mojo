# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`mojolearn.neighbors_graph` (lane sc-ann-1-umap) and `UMAP.graph_` (lane
sc-leftovers-1, built when read) on hand-sized fixtures, under IDENTICAL.
Written by lane sc-leftovers-1, 2026-11-28; NOT RUN by the lane.

  host (CPU column, `run_host_checks`):
    * LINE: the four points 0, 1, 3, 7 on a line, n_neighbors 3 (self and two
      others). By hand, the distances CSR is
        indptr [0, 2, 4, 6, 8], indices [1, 2, 0, 2, 1, 0, 2, 1],
        data [1, 3, 1, 2, 2, 3, 4, 6]
      (exact: the squared distances are perfect squares). The connectivities
      join 0-1, 0-2, 1-2, 1-3, 2-3 (columns ascending: rows [1, 2], [0, 2, 3],
      [0, 1, 3], [1, 2]); each nearest-neighbor membership is exp(0) = 1, so
      0-1 and 2-3 (and 1-2, whose 2 -> 1 side is 1) have weight 1 within
      1e-6; every other weight is in (0, 1); the matrix is symmetric bit for bit.
    * FIT GRAPH: on 48 clustered rows, n_neighbors 6, the graph
      `host_umap_fit_graph` builds (UMAP_OP_FIT_GRAPH, what `graph_` reads)
      equals the CSR the fit publishes when asked (`host_umap_fit_transform`
      with its graph outputs) word for word, unsupervised and with a
      categorical target, and the unsupervised one equals the
      connectivities of `host_umap_neighbors_graph`.
    * IVF: 400 clustered rows, 8 lists, 2 probes, n_neighbors 6: every
      distances row holds 5 other rows, no self and no repeat, ascending.
  device (`run_checks`, a GPU column): `umap_neighbors_graph_device` equals
    the host words (both CSRs), exact and IVF; `umap_fit_graph` equals
    `host_umap_fit_graph`, unsupervised and categorical.

    tools/with_identical_mode.sh pixi run mojo run -I . checks/native_tests/umap_neighbors_graph_host_check.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . checks/native_tests/umap_neighbors_graph_check.mojo"""

from max.gpu.host import DeviceContext
from std.memory import bitcast
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from spectral.checks.device_io import upload_f32
from umap.params import UMAPParams, UMAP_NEIGHBORS_IVF
from umap.graph_out import UmapFitOut, umap_graph_capacity
from umap.host.umap_oracle import host_umap_neighbors_graph, host_umap_fit_graph, host_umap_fit_transform
from umap.sparse_estimator import umap_neighbors_graph_device, umap_fit_graph
from x_ann.checks.ann_check_fixtures import report


struct NgCsr(Movable):
    """A CSR the checks fill: n + 1 int64 offsets, `cap` int32 columns and float32 values."""
    var indptr: List[Int64]
    var indices: List[Int32]
    var data: List[Float32]

    def __init__(out self, n: Int, cap: Int):
        self.indptr = List[Int64](length=n + 1, fill=Int64(0))
        self.indices = List[Int32](length=max(cap, 1), fill=Int32(0))
        self.data = List[Float32](length=max(cap, 1), fill=Float32(0.0))

    def dst(self) -> UmapFitOut:
        """The three addresses as a fitted-graph destination (no curve)."""
        return UmapFitOut(Int(self.indptr.unsafe_ptr()), Int(self.indices.unsafe_ptr()), Int(self.data.unsafe_ptr()),
                          0)

    def nnz(self) -> Int:
        return Int(self.indptr[len(self.indptr) - 1])


def _same_csr(a: NgCsr, b: NgCsr) -> Bool:
    if len(a.indptr) != len(b.indptr):
        return False
    for i in range(len(a.indptr)):
        if a.indptr[i] != b.indptr[i]:
            return False
    var m = a.nnz()
    if m != b.nnz() or m > len(a.indices) or m > len(b.indices):
        return False
    for e in range(m):
        if a.indices[e] != b.indices[e]:
            return False
        if bitcast[DType.uint32](a.data[e]) != bitcast[DType.uint32](b.data[e]):
            return False
    return True


def ng_fixture_clusters(n: Int, d: Int) -> List[Float32]:
    """Four clusters 50 apart on the first feature, a small deterministic spread inside each."""
    var x = List[Float32](capacity=n * d)
    for i in range(n):
        var c = (i * 4) // n
        for f in range(d):
            var h = ((i * 2654435761 + f * 40503) >> 7) % 1000
            var v = Float32(h) / Float32(1000.0)
            if f == 0:
                v += Float32(50 * c)
            x.append(v)
    return x^


def _labels(n: Int) -> List[Float32]:
    """A categorical target: three labels and one unknown (-1) every seventh row."""
    var t = List[Float32](length=n, fill=Float32(0.0))
    for i in range(n):
        t[i] = Float32(-1.0) if i % 7 == 3 else Float32((i * 3) // n)
    return t^


def _weight_at(c: NgCsr, r: Int, col: Int) -> Float32:
    for e in range(Int(c.indptr[r]), Int(c.indptr[r + 1])):
        if Int(c.indices[e]) == col:
            return c.data[e]
    return Float32(-1.0)


def _line_checks(mut failed: Int) raises:
    var x: List[Float32] = [0.0, 1.0, 3.0, 7.0]
    var n = 4
    var k = 3
    var params = UMAPParams(n_neighbors=k, n_components=2)
    var dist = NgCsr(n, n * (k - 1))
    var conn = NgCsr(n, umap_graph_capacity(n, k, 0, k))
    var edges = host_umap_neighbors_graph(x, n, 1, params, Int(dist.indptr.unsafe_ptr()),
                                          Int(dist.indices.unsafe_ptr()), Int(dist.data.unsafe_ptr()), conn.dst())
    var want_ptr: List[Int] = [0, 2, 4, 6, 8]
    var want_idx: List[Int] = [1, 2, 0, 2, 1, 0, 2, 1]
    var want_d: List[Float32] = [1.0, 3.0, 1.0, 2.0, 2.0, 3.0, 4.0, 6.0]
    var ok = True
    for i in range(n + 1):
        ok = ok and Int(dist.indptr[i]) == want_ptr[i]
    for e in range(8):
        ok = ok and Int(dist.indices[e]) == want_idx[e] and dist.data[e] == want_d[e]
    report("neighbors_graph.line.distances_by_hand", ok, failed)
    var cptr: List[Int] = [0, 2, 5, 8, 10]
    var cidx: List[Int] = [1, 2, 0, 2, 3, 0, 1, 3, 1, 2]
    var pattern = edges == 10 and conn.nnz() == 10
    for i in range(n + 1):
        pattern = pattern and Int(conn.indptr[i]) == cptr[i]
    for e in range(10):
        pattern = pattern and Int(conn.indices[e]) == cidx[e]
    report("neighbors_graph.line.connectivities_pattern", pattern, failed)
    var ones = True
    for pair in range(3):
        var a = pair
        var w = _weight_at(conn, a, a + 1)
        ones = ones and w > Float32(1.0 - 1e-6) and w <= Float32(1.0)
    var inside = True
    var others: List[Int] = [0, 2, 1, 3]
    for q in range(2):
        var w = _weight_at(conn, others[2 * q], others[2 * q + 1])
        inside = inside and w > Float32(0.0) and w < Float32(1.0)
    report("neighbors_graph.line.nearest_memberships_one", ones, failed)
    report("neighbors_graph.line.other_weights_inside", inside, failed)
    var sym = True
    for r in range(n):
        for e in range(Int(conn.indptr[r]), Int(conn.indptr[r + 1])):
            var back = _weight_at(conn, Int(conn.indices[e]), r)
            sym = sym and bitcast[DType.uint32](back) == bitcast[DType.uint32](conn.data[e])
    report("neighbors_graph.line.symmetric_bits", sym, failed)
    _ = x^


def _fit_graph_checks(mut failed: Int) raises:
    var n = 48
    var d = 3
    var k = 6
    var x = ng_fixture_clusters(n, d)
    var labels = _labels(n)
    for supervised in range(2):
        var params = UMAPParams(n_neighbors=k, n_components=2, n_epochs=4, random_seed=UInt64(3))
        var target = labels.copy() if supervised == 1 else List[Float32]()
        var kind = 1 if supervised == 1 else 0
        var cap = umap_graph_capacity(n, k, kind, k)
        var fitted = NgCsr(n, cap)
        var ab = List[Float32](length=2, fill=Float32(0.0))
        var fo = fitted.dst()
        _ = host_umap_fit_transform(x, n, d, params, List[Float32](), target, kind, 1, 0, Float32(0.5),
                                    fo.indptr, fo.indices, fo.data, Int(ab.unsafe_ptr()))
        var built = NgCsr(n, cap)
        var edges = host_umap_fit_graph(x, n, d, params, target, kind, 1, 0, Float32(0.5), built.dst())
        var tag = String("categorical") if supervised == 1 else String("unsupervised")
        report("fit_graph." + tag + ".is_the_fit_graph", edges == built.nnz() and _same_csr(built, fitted), failed)
        if supervised == 0:
            var dist = NgCsr(n, n * (k - 1))
            var conn = NgCsr(n, cap)
            _ = host_umap_neighbors_graph(x, n, d, UMAPParams(n_neighbors=k, n_components=2),
                                          Int(dist.indptr.unsafe_ptr()), Int(dist.indices.unsafe_ptr()),
                                          Int(dist.data.unsafe_ptr()), conn.dst())
            report("fit_graph.unsupervised.is_neighbors_graph", _same_csr(built, conn), failed)
        _ = ab^
    _ = x^


def _ivf_checks(mut failed: Int) raises:
    var n = 400
    var d = 5
    var k = 6
    var x = ng_fixture_clusters(n, d)
    var params = UMAPParams(n_neighbors=k, n_components=2, neighbors=UMAP_NEIGHBORS_IVF, ivf_lists=8,
                            ivf_probes=2, ivf_seed=17)
    var dist = NgCsr(n, n * (k - 1))
    var conn = NgCsr(n, umap_graph_capacity(n, k, 0, k))
    _ = host_umap_neighbors_graph(x, n, d, params, Int(dist.indptr.unsafe_ptr()), Int(dist.indices.unsafe_ptr()),
                                  Int(dist.data.unsafe_ptr()), conn.dst())
    var ok = dist.nnz() == n * (k - 1)
    for r in range(n):
        for s in range(k - 1):
            var e = r * (k - 1) + s
            var j = Int(dist.indices[e])
            ok = ok and j >= 0 and j < n and j != r
            for t in range(s):
                ok = ok and Int(dist.indices[r * (k - 1) + t]) != j
            if s > 0:
                ok = ok and dist.data[e - 1] <= dist.data[e]
    report("neighbors_graph.ivf.valid_rows", ok, failed)
    _ = x^


def run_host_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL, "run under tools/with_identical_mode.sh"
    var failed = 0
    _line_checks(failed)
    _fit_graph_checks(failed)
    _ivf_checks(failed)
    if failed > 0:
        raise Error("neighbors_graph_check (host): FAILED " + String(failed))
    print("neighbors_graph_check (host): ALL OK")


def run_checks() raises:
    """The device column against the host column, word for word."""
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL, "run under tools/with_identical_mode.sh"
    var failed = 0
    var ctx = DeviceContext()
    var n = 400
    var d = 5
    var k = 6
    var x = ng_fixture_clusters(n, d)
    for route in range(2):
        var params = UMAPParams(n_neighbors=k, n_components=2)
        if route == 1:
            params = UMAPParams(n_neighbors=k, n_components=2, neighbors=UMAP_NEIGHBORS_IVF, ivf_lists=8,
                                ivf_probes=2, ivf_seed=17)
        var cap = umap_graph_capacity(n, k, 0, k)
        var hd = NgCsr(n, n * (k - 1))
        var hc = NgCsr(n, cap)
        var gd = NgCsr(n, n * (k - 1))
        var gc = NgCsr(n, cap)
        var he = host_umap_neighbors_graph(x, n, d, params, Int(hd.indptr.unsafe_ptr()), Int(hd.indices.unsafe_ptr()),
                                           Int(hd.data.unsafe_ptr()), hc.dst())
        var d_x = upload_f32(ctx, x)
        var ge = umap_neighbors_graph_device(ctx, d_x, n, d, params, Int(gd.indptr.unsafe_ptr()),
                                             Int(gd.indices.unsafe_ptr()), Int(gd.data.unsafe_ptr()), gc.dst())
        ctx.synchronize()
        _ = d_x^
        var tag = String("ivf") if route == 1 else String("exact")
        report("neighbors_graph." + tag + ".device_is_host", he == ge and _same_csr(gd, hd) and _same_csr(gc, hc),
               failed)
    var labels = _labels(n)
    for supervised in range(2):
        var params = UMAPParams(n_neighbors=k, n_components=2, random_seed=UInt64(3))
        var target = labels.copy() if supervised == 1 else List[Float32]()
        var kind = 1 if supervised == 1 else 0
        var cap = umap_graph_capacity(n, k, kind, k)
        var hb = NgCsr(n, cap)
        var gb = NgCsr(n, cap)
        var he = host_umap_fit_graph(x, n, d, params, target, kind, 1, 0, Float32(0.5), hb.dst())
        var ge = umap_fit_graph(ctx, x, n, d, params, target, kind, 1, 0, Float32(0.5), gb.dst())
        ctx.synchronize()
        var tag = String("categorical") if supervised == 1 else String("unsupervised")
        report("fit_graph." + tag + ".device_is_host", he == ge and _same_csr(gb, hb), failed)
    _ = x^
    if failed > 0:
        raise Error("neighbors_graph_check (device): FAILED " + String(failed))
    print("neighbors_graph_check (device): ALL OK")
