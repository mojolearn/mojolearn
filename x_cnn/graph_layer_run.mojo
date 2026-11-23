# SPDX-License-Identifier: Apache-2.0
"""GCNConv / SAGEConv graph build and host-form backward as typed RUN entries
over integer and float32 buffers (lane py-msel-prep-cnn, 2026-11-14).

What bindings/xcnn_graph_native_door.mojo composed over Python edge arrays:
the (2, E) edge index as int32 source and target rows, the two CSR views
(`x_cnn_csr_build`: rows = targets for the propagation, rows = sources for
its transpose), GCN's self loops, per-entry weights gathered into the forward
view and the symmetric normalization, and the host-form backward tapes of
both layers (the transposed values, the scratch the tapes write, the address
words). The entry calls, their argument words, the order and the buffer
sizes are the door's: no bit moves. Two copies the door made are gone: the
CSR order is built straight into its published buffer (it was built into a
max(nnz, 1) buffer and sliced), and the backward scratch (transposed values,
the forward's recorded h and output, ones, the tape's temporaries) is owned
here as native storage instead of Python arrays.

`GraphCalls` is what a caller supplies: storage for the published buffers
(handles, so the door can publish Python arrays and a native check native
lists) and the x_cnn entry operations (the door releases the GIL around each,
as it did; `GraphTableCalls` calls the table entry directly). No Python
object is read, built or called here.
"""
from std.memory import unsafe_memcpy
from core.entry_table import EntryRef
from x_cnn.graph_run import (
    graph_run_csr_words, graph_run_edges, graph_run_check_edge_weight, graph_run_ones, graph_run_slice,
    graph_run_csr_build, graph_run_gather, graph_run_vals_t, graph_run_gcn_loops, graph_run_gcn_norm,
    graph_run_gcn_backward, graph_run_sage_backward, graph_run_gemm, graph_run_spmm, graph_run_add,
    graph_run_bias_rows, graph_run_linear_forward, graph_run_relu_forward, graph_run_graph_op,
)


trait GraphCalls:
    """Published storage and the x_cnn entries the graph RUN calls."""

    def alloc_i32(mut self, count: Int) raises -> Int:
        """A new zero-or-uninitialized int32 buffer of `count` words; its handle."""
        ...

    def alloc_f32(mut self, count: Int) raises -> Int:
        """A new float32 buffer of `count` words; its handle."""
        ...

    def address(self, handle: Int) raises -> Int:
        ...

    def csr_build(self, rows: Int, cols: Int, csr: Int, order: Int, n: Int, nnz: Int) raises:
        ...

    def gather(self, values: Int, order: Int, dst: Int, size: Int, n: Int) raises:
        ...

    def vals_t(self, vals_f: Int, order_f: Int, order_t: Int, dst: Int, nnz: Int) raises:
        ...

    def gcn_loops(self, src: Int, dst: Int, w: Int, src_out: Int, dst_out: Int, w_out: Int, n: Int, edges: Int,
                  improved: Bool) raises -> Int:
        ...

    def gcn_norm(self, wf: Int, vals: Int, csr: Int, n: Int, nnz: Int, graph_route: Bool) raises:
        ...

    def gcn_backward(self, addrs: List[Int], n: Int, d: Int, f: Int, nnz: Int, bias: Bool) raises:
        """The host-form GCN tape (13 address words)."""
        ...

    def sage_backward(self, addrs: List[Int], n: Int, d: Int, f: Int, nnz: Int, kind: Int, norm: Bool,
                      root: Bool, project: Bool) raises:
        """The host-form SAGE tape (33 address words)."""
        ...


@fieldwise_init
struct GraphEdges(ImplicitlyCopyable, Movable):
    """Handles of the int32 source and target rows of `count` edges."""

    var src: Int
    var dst: Int
    var count: Int


@fieldwise_init
struct GraphViews(ImplicitlyCopyable, Movable):
    """Handles of a graph's published buffers: the edge rows (nnz each), the
    forward and transposed CSR views (graph_run_csr_words(n, nnz) each) and
    their entry orders (nnz each)."""

    var n: Int
    var nnz: Int
    var src: Int
    var dst: Int
    var csr_f: Int
    var order_f: Int
    var csr_t: Int
    var order_t: Int


@fieldwise_init
struct GcnGraph(ImplicitlyCopyable, Movable):
    """A GCN graph and the handle of its nnz normalized (or gathered) values."""

    var views: GraphViews
    var vals: Int


def graph_edges_run[C: GraphCalls](mut calls: C, base: Int, row_step: Int, edge_step: Int, count: Int, item: Int,
                                   kind: Int, little: Bool, n: Int) raises -> GraphEdges:
    """The (2, E) edge index at `base` (byte strides, item width and cast
    kind of its dtype) as int32 rows, with the node range refusal."""
    var src = calls.alloc_i32(count)
    var dst = calls.alloc_i32(count)
    graph_run_edges(base, row_step, edge_step, count, item, kind, little, n, calls.address(src), calls.address(dst))
    return GraphEdges(src, dst, count)


def _view[C: GraphCalls](mut calls: C, rows: Int, cols: Int, n: Int, nnz: Int) raises -> Tuple[Int, Int]:
    """One CSR view and its order (built into the published order buffer;
    an empty graph's order gets one scratch word)."""
    var csr = calls.alloc_i32(graph_run_csr_words(n, nnz))
    var order = calls.alloc_i32(nnz)
    var scratch = List[Int32](length=1, fill=0)
    var order_addr = calls.address(order) if nnz > 0 else Int(scratch.unsafe_ptr())
    calls.csr_build(rows, cols, calls.address(csr), order_addr, n, nnz)
    _ = scratch^
    return (csr, order)


def graph_views_run[C: GraphCalls](mut calls: C, src: Int, dst: Int, n: Int, nnz: Int) raises -> GraphViews:
    """`_Graph(src, dst, n)`: both CSR views of the edge rows `src`, `dst`
    (handles of nnz int32 words each)."""
    var s = calls.address(src)
    var d = calls.address(dst)
    var f = _view(calls, d, s, n, nnz)
    var t = _view(calls, s, d, n, nnz)
    return GraphViews(n, nnz, src, dst, f[0], f[1], t[0], t[1])


def gcn_graph_run[C: GraphCalls](mut calls: C, edges: GraphEdges, n: Int, weight: Int, weight_count: Int,
                                 normalize: Bool, add_self_loops: Bool, improved: Bool,
                                 graph_route: Bool) raises -> GcnGraph:
    """`GCNConv._graph(n, edge_index, edge_weight)`: the edge weights (ones
    when `weight` is 0, else a copy of the `weight_count` float32 values at
    `weight`), the self loops when normalizing with them, both CSR views, the
    weights gathered into the forward view, and the normalization."""
    var e = edges.count
    var w: Int
    var w_count: Int
    if weight == 0:
        w = calls.alloc_f32(e)
        w_count = e
        graph_run_ones(calls.address(w), e)
    else:
        w = calls.alloc_f32(weight_count)
        w_count = weight_count
        graph_run_slice(calls.address(w), weight, 0, weight_count)
    graph_run_check_edge_weight(w_count, e)
    var src = edges.src
    var dst = edges.dst
    if normalize and add_self_loops:
        var t = e + n
        var so = calls.alloc_i32(t)
        var dso = calls.alloc_i32(t)
        var wo = calls.alloc_f32(t)
        var k = calls.gcn_loops(calls.address(src), calls.address(dst), calls.address(w), calls.address(so),
                                calls.address(dso), calls.address(wo), n, e, improved)
        src = calls.alloc_i32(k + n)
        dst = calls.alloc_i32(k + n)
        var kept_w = calls.alloc_f32(k + n)
        graph_run_slice(calls.address(src), calls.address(so), 0, k + n)
        graph_run_slice(calls.address(dst), calls.address(dso), 0, k + n)
        graph_run_slice(calls.address(kept_w), calls.address(wo), 0, k + n)
        w = kept_w
        w_count = k + n
    var nnz = w_count if normalize and add_self_loops else e
    var views = graph_views_run(calls, src, dst, n, nnz)
    var wf = calls.alloc_f32(nnz)
    calls.gather(calls.address(w), calls.address(views.order_f), calls.address(wf), w_count, nnz)
    var vals = wf
    if normalize:
        vals = calls.alloc_f32(nnz)
        calls.gcn_norm(calls.address(wf), calls.address(vals), calls.address(views.csr_f), n, nnz, graph_route)
    return GcnGraph(views, vals)


comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]


def gcn_forward_host_run(entry: EntryRef, x: Int, weight: Int, bias: Int, vals: Int, csr_f: Int, dst: Int, n: Int,
                         f: Int, d: Int, nnz: Int) raises:
    """GCNConv.forward on the host form (lane bm-trees, 2026-11-22; the
    composition bindings/xcnn_graph_native_door.mojo `xcnn_gcn_forward` ran
    over Python arrays, call for call): `h = x @ weight^T` (gemm op 1),
    `y = A_hat h` (spmm mode 0 over the normalized values `vals` on the
    forward CSR view), then `+ bias` broadcast over the rows when `bias` is
    not 0. `x` is (n, d), `weight` (f, d), `dst` (n, f), all C-ordered
    Float32. The door (with the GIL released) and a native check call it."""
    var h = List[Float32](length=n * f, fill=0)
    graph_run_gemm(entry, x, weight, Int(h.unsafe_ptr()), n, f, d, 1)
    if bias == 0:
        graph_run_spmm(entry, vals, Int(h.unsafe_ptr()), dst, csr_f, n, f, nnz, 0)
        _ = h^
        return
    var spread = List[Float32](length=n * f, fill=0)
    graph_run_spmm(entry, vals, Int(h.unsafe_ptr()), Int(spread.unsafe_ptr()), csr_f, n, f, nnz, 0)
    var rows = List[Float32](length=n * f, fill=0)
    graph_run_bias_rows(bias, Int(rows.unsafe_ptr()), n, f)
    graph_run_add(entry, Int(spread.unsafe_ptr()), Int(rows.unsafe_ptr()), dst, n * f)
    _ = h^
    _ = spread^
    _ = rows^


def _ones(count: Int) -> List[Float32]:
    return List[Float32](length=count, fill=1)


def gcn_backward_host_run[C: GraphCalls](calls: C, csr_t: Int, order_f: Int, order_t: Int, vals: Int, nnz: Int,
                                         n: Int, d: Int, f: Int, g: Int, x: Int, w: Int, bias: Bool, b: Int,
                                         gw: Int, gb: Int, dx: Int) raises:
    """GCNConv's host-form backward: the transposed values, then the tape
    over g (n, f), the forward input x (n, d), weight_ w (f, d) and bias_ b
    (f; 0 without a bias), writing gw (f, d), gb (f) and dx (n, d). The
    forward's h and output (recorded as the tape's values, never read) get
    disjoint halves of one scratch block (the tape refuses an output that
    overlaps a recorded value)."""
    var vt = List[Float32](length=max(nnz, 1), fill=0)
    calls.vals_t(vals, order_f, order_t, Int(vt.unsafe_ptr()), nnz)
    var dh = List[Float32](length=n * f, fill=0)
    var hv = List[Float32](length=2 * n * f, fill=0)
    var h_addr = Int(hv.unsafe_ptr())
    var ones = _ones(n if bias else 0)
    var addrs: List[Int] = [
        g, x, h_addr, h_addr + n * f * 4, w, gw, dx, Int(dh.unsafe_ptr()), Int(vt.unsafe_ptr()), csr_t,
        b if bias else 0, gb if bias else 0, Int(ones.unsafe_ptr()) if bias else 0,
    ]
    calls.gcn_backward(addrs, n, d, f, nnz, bias)
    _ = vt^
    _ = dh^
    _ = hv^
    _ = ones^


@fieldwise_init
struct SageBackwardHost(Copyable, Movable):
    """The buffers of SAGEConv's host-form backward, by address (0 when the
    configuration has none): inputs g (n, f), x and the aggregate agg (n, d),
    lin_l's weight and bias; the normalization's y, pre (n, f), den (n) and
    nograph (n + 1, int32); the max aggregation's aux; the root weight wr;
    the projection's source features s, its pre-ReLU rows relu_x and its
    linear wp, bp. Outputs dw (f, d), db (f), dx (n, d), gwr (f, d), dwp
    (d, d), dbp (d)."""

    var g: Int
    var x: Int
    var agg: Int
    var lin_w: Int
    var lin_b: Int
    var dw: Int
    var db: Int
    var dx: Int
    var y: Int
    var pre: Int
    var den: Int
    var nograph: Int
    var max_aux: Int
    var wr: Int
    var gwr: Int
    var s: Int
    var relu_x: Int
    var wp: Int
    var bp: Int
    var dwp: Int
    var dbp: Int


def sage_backward_host_run[C: GraphCalls](calls: C, csr_f: Int, csr_t: Int, nnz: Int, n: Int, d: Int, f: Int,
                                          kind: Int, norm: Bool, root: Bool, project: Bool,
                                          io: SageBackwardHost) raises:
    """SAGEConv's host-form backward tape (autograd/graph_tape.mojo
    `sage_cfg`: 33 address words, ten integers), its scratch owned here."""
    var dagg = List[Float32](length=n * d, fill=0)
    var ones = _ones(max(nnz, 1))
    var gp = List[Float32](length=n * f, fill=0)   # the normalization's gradient, or the recorded output
    var dsc = List[Float32](length=n * d if kind == 1 else 0, fill=0)
    var roots = List[Float32](length=3 * n * d if root else 0, fill=0)
    var proj = List[Float32](length=2 * n * d if project else 0, fill=0)
    var gp_addr = Int(gp.unsafe_ptr())
    var r0 = Int(roots.unsafe_ptr())
    var p0 = Int(proj.unsafe_ptr())
    var block = n * d * 4
    var ones_addr = Int(ones.unsafe_ptr())
    var addrs: List[Int] = [
        io.g, io.x, io.agg, io.y if norm else gp_addr, io.lin_w, io.lin_b, io.dw, io.db, Int(dagg.unsafe_ptr()),
        io.dx, r0 + block if root else 0, ones_addr, csr_f, ones_addr, csr_t, io.max_aux if kind == 2 else 0,
        Int(dsc.unsafe_ptr()) if kind == 1 else 0, io.wr if root else 0, io.gwr if root else 0,
        r0 if root else 0, io.pre if norm else 0, gp_addr if norm else 0, io.den if norm else 0,
        # the host form's ten extra words
        io.s if project else 0, io.relu_x if project else 0, io.wp if project else 0, io.bp if project else 0,
        io.dwp if project else 0, io.dbp if project else 0, p0 if project else 0, p0 + block if project else 0,
        io.nograph if norm else 0, r0 + 2 * block if root else 0,
    ]
    calls.sage_backward(addrs, n, d, f, nnz, kind, norm, root, project)
    _ = dagg^
    _ = ones^
    _ = gp^
    _ = dsc^
    _ = roots^
    _ = proj^


# ---------------------------------------------------------------- SAGEConv host form: forward, saved state, backward
# Lane bm-rest-graph (2026-11-23). What bindings/xcnn_graph_native_door.mojo `xcnn_sage_forward` composed over
# Python arrays (the projection's linear and ReLU, the max graph op or the spmm over ones, lin_l's linear, the root
# gemm and add, the normalization graph op), call for call with the same entries, words and order, and the buffers
# its `_sage_backward_host` read back from the layer's fields (`_xs`, `_max_aux`, `_pre`, `_den`, `_nograph`, `_y`
# and the `_x` of lin_l, lin and the ReLU) now live in one `SageSaved`; the layer holds it through one native owner
# (bindings/xcnn_sage_saved_owner_pack.mojo). The forward's intermediates are native storage; the outputs go
# straight into the caller's `dst` (no copy: the last step of each configuration writes it).


@fieldwise_init
struct SageShape(ImplicitlyCopyable, Movable):
    """One SAGEConv host-form call: n nodes, d in and f out channels, nnz
    edges, the aggregation word (graph_run_aggr_kind: 0 sum / add, 1 mean,
    2 max) and the normalize, root_weight and project flags."""

    var n: Int
    var d: Int
    var f: Int
    var nnz: Int
    var kind: Int
    var norm: Bool
    var root: Bool
    var project: Bool


@fieldwise_init
struct SageEntries(ImplicitlyCopyable, Movable):
    """The resolved x_cnn entries of the layer (aggregation, root gemm and
    add, normalization), of lin_l, of the projection's linear and of its
    ReLU (the door resolved each module's own route, as before; a native
    caller passes one entry four times)."""

    var layer: EntryRef
    var lin_l: EntryRef
    var lin: EntryRef
    var relu: EntryRef


@fieldwise_init
struct SageWeights(ImplicitlyCopyable, Movable):
    """Addresses of the parameters: lin_l's weight (f, d) and bias (f), the
    root weight (f, d; 0 without root_weight) and the projection's weight
    (d, d) and bias (d; 0 without project). Float32, C-ordered."""

    var wl: Int
    var bl: Int
    var wr: Int
    var wp: Int
    var bp: Int


@fieldwise_init
struct SageGrads(ImplicitlyCopyable, Movable):
    """Addresses the backward writes: lin_l's dw (f, d) and db (f), dx
    (n, d), the root gradient gwr (f, d; 0 without root_weight) and the
    projection's dwp (d, d) and dbp (d; 0 without project)."""

    var dw: Int
    var db: Int
    var dx: Int
    var gwr: Int
    var dwp: Int
    var dbp: Int


struct SageSaved(Movable):
    """The buffers SAGEConv's host-form forward saves for its backward:
    the projection's pre-ReLU rows hp and output xs (n, d; project only),
    the aggregate agg (n, d), the max aggregation's aux (2 n d; max only),
    the normalization's pre (n, f), den (n) and nograph (n + 1 zero
    offsets; normalize only), and the address of the normalized output y
    (the caller's `dst`, kept alive by whoever holds this state; 0 without
    normalize)."""

    var shape: SageShape
    var hp: List[Float32]
    var xs: List[Float32]
    var agg: List[Float32]
    var aux: List[Float32]
    var pre: List[Float32]
    var den: List[Float32]
    var nograph: List[Int32]
    var y: Int

    def __init__(out self, shape: SageShape):
        var nd = shape.n * shape.d
        var nf = shape.n * shape.f
        self.shape = shape
        self.hp = List[Float32](length=max(nd, 1) if shape.project else 0, fill=0)
        self.xs = List[Float32](length=max(nd, 1) if shape.project else 0, fill=0)
        self.agg = List[Float32](length=max(nd, 1), fill=0)
        self.aux = List[Float32](length=max(2 * nd, 1) if shape.kind == 2 else 0, fill=0)
        self.pre = List[Float32](length=max(nf, 1) if shape.norm else 0, fill=0)
        self.den = List[Float32](length=max(shape.n, 1) if shape.norm else 0, fill=0)
        self.nograph = List[Int32](length=shape.n + 1 if shape.norm else 0, fill=0)
        self.y = 0


def sage_forward_host_run(entries: SageEntries, x: Int, csr_f: Int, weights: SageWeights, dst: Int,
                          mut saved: SageSaved) raises:
    """SAGEConv.forward on the host form over `x` (n, d) and the forward CSR
    view `csr_f`, writing the output (n, f) to `dst` and the backward's
    buffers to `saved` (built by `SageSaved(shape)`). The door (with the GIL
    released) and a native check call it."""
    var s = saved.shape
    var n = s.n
    var d = s.d
    var f = s.f
    var src = x
    if s.project:
        # `_Linear.forward(x)` then `_ReLU.forward`: xs = relu(x W_p^T + b_p)
        graph_run_linear_forward(entries.lin, x, weights.wp, weights.bp, Int(saved.hp.unsafe_ptr()), n, d, d)
        graph_run_relu_forward(entries.relu, Int(saved.hp.unsafe_ptr()), Int(saved.xs.unsafe_ptr()), n * d)
        src = Int(saved.xs.unsafe_ptr())
    var agg = Int(saved.agg.unsafe_ptr())
    if s.kind == 2:
        graph_run_graph_op(entries.layer, src, src, Int(saved.aux.unsafe_ptr()), agg, csr_f, n, d, s.nnz, 0)
    else:
        var ones = _ones(max(s.nnz, 1))
        graph_run_spmm(entries.layer, Int(ones.unsafe_ptr()), src, agg, csr_f, n, d, s.nnz, 1 if s.kind == 1 else 0)
        _ = ones^
    # lin_l's output: the returned rows when it is the last step, the
    # normalization's input without a root, else scratch for the root add
    var ol = List[Float32](length=n * f if s.root else 0, fill=0)
    var lin_dst: Int
    if s.root:
        lin_dst = Int(ol.unsafe_ptr())
    elif s.norm:
        lin_dst = Int(saved.pre.unsafe_ptr())
    else:
        lin_dst = dst
    graph_run_linear_forward(entries.lin_l, agg, weights.wl, weights.bl, lin_dst, n, d, f)
    var rows = List[Float32](length=n * f if s.root else 0, fill=0)
    if s.root:
        graph_run_gemm(entries.layer, x, weights.wr, Int(rows.unsafe_ptr()), n, f, d, 1)
        var sum_dst = Int(saved.pre.unsafe_ptr()) if s.norm else dst
        graph_run_add(entries.layer, Int(ol.unsafe_ptr()), Int(rows.unsafe_ptr()), sum_dst, n * f)
    if s.norm:
        var pre = Int(saved.pre.unsafe_ptr())
        graph_run_graph_op(entries.layer, pre, pre, Int(saved.den.unsafe_ptr()), dst,
                           Int(saved.nograph.unsafe_ptr()), n, f, 0, 2)
        saved.y = dst
    _ = ol^
    _ = rows^


def sage_backward_saved_run[C: GraphCalls](calls: C, csr_f: Int, csr_t: Int, saved: SageSaved, x: Int, g: Int,
                                           weights: SageWeights, grads: SageGrads) raises:
    """SAGEConv's host-form backward over the forward's saved state: the
    upstream gradient g (n, f), the forward input x (n, d), the current
    parameters, writing `grads` (`sage_backward_host_run`, the tape)."""
    var s = saved.shape
    var io = SageBackwardHost(
        g, x, Int(saved.agg.unsafe_ptr()), weights.wl, weights.bl, grads.dw, grads.db, grads.dx,
        saved.y if s.norm else 0, Int(saved.pre.unsafe_ptr()) if s.norm else 0,
        Int(saved.den.unsafe_ptr()) if s.norm else 0, Int(saved.nograph.unsafe_ptr()) if s.norm else 0,
        Int(saved.aux.unsafe_ptr()) if s.kind == 2 else 0, weights.wr if s.root else 0,
        grads.gwr if s.root else 0, Int(saved.xs.unsafe_ptr()) if s.project else 0,
        Int(saved.hp.unsafe_ptr()) if s.project else 0, weights.wp if s.project else 0,
        weights.bp if s.project else 0, grads.dwp if s.project else 0, grads.dbp if s.project else 0)
    sage_backward_host_run(calls, csr_f, csr_t, s.nnz, s.n, s.d, s.f, s.kind, s.norm, s.root, s.project, io)


# ---------------------------------------------------------------- the native calls (checks)

struct GraphTableCalls(GraphCalls, Movable):
    """`GraphCalls` over one resolved x_cnn table entry, with native storage
    (handle 2*i an int32 buffer, 2*i + 1 a float32 buffer)."""

    var entry: EntryRef
    var i32: List[List[Int32]]
    var f32: List[List[Float32]]

    def __init__(out self, entry: EntryRef):
        self.entry = entry
        self.i32 = List[List[Int32]]()
        self.f32 = List[List[Float32]]()

    def alloc_i32(mut self, count: Int) raises -> Int:
        self.i32.append(List[Int32](length=max(count, 1), fill=0))
        return 2 * (len(self.i32) - 1)

    def alloc_f32(mut self, count: Int) raises -> Int:
        self.f32.append(List[Float32](length=max(count, 1), fill=0))
        return 2 * (len(self.f32) - 1) + 1

    def address(self, handle: Int) raises -> Int:
        if handle % 2 == 0:
            return Int(self.i32[handle // 2].unsafe_ptr())
        return Int(self.f32[handle // 2].unsafe_ptr())

    def csr_build(self, rows: Int, cols: Int, csr: Int, order: Int, n: Int, nnz: Int) raises:
        graph_run_csr_build(self.entry, rows, cols, csr, order, n, nnz)

    def gather(self, values: Int, order: Int, dst: Int, size: Int, n: Int) raises:
        graph_run_gather(self.entry, values, order, dst, size, n)

    def vals_t(self, vals_f: Int, order_f: Int, order_t: Int, dst: Int, nnz: Int) raises:
        graph_run_vals_t(self.entry, vals_f, order_f, order_t, dst, nnz)

    def gcn_loops(self, src: Int, dst: Int, w: Int, src_out: Int, dst_out: Int, w_out: Int, n: Int, edges: Int,
                  improved: Bool) raises -> Int:
        return graph_run_gcn_loops(self.entry, src, dst, w, src_out, dst_out, w_out, n, edges, improved)

    def gcn_norm(self, wf: Int, vals: Int, csr: Int, n: Int, nnz: Int, graph_route: Bool) raises:
        graph_run_gcn_norm(self.entry, wf, vals, csr, n, nnz, graph_route)

    def gcn_backward(self, addrs: List[Int], n: Int, d: Int, f: Int, nnz: Int, bias: Bool) raises:
        graph_run_gcn_backward(self.entry, addrs, n, d, f, nnz, bias, False, 0, True)

    def sage_backward(self, addrs: List[Int], n: Int, d: Int, f: Int, nnz: Int, kind: Int, norm: Bool,
                      root: Bool, project: Bool) raises:
        graph_run_sage_backward(self.entry, addrs, n, d, f, nnz, kind, norm, root, False, True, project)
