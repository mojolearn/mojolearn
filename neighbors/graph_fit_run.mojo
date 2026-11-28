# SPDX-License-Identifier: Apache-2.0
"""PageRank, Louvain, Leiden, connected_components, the shared-nearest-neighbor graph and PAGA RUN: typed requests in, typed
results out.

The boundary (bindings/neighbors_graph_native_door.mojo) admits the public
arguments into these requests, releases the GIL once for each function here
and publishes the results. Nothing in this module imports Python. Every
address in a request is admitted storage the boundary keeps alive through
the call; the restart vectors, the info words and the labels before their
renumbering are native scratch of this module. The operations, their words
and their order are the public method's (the requests and checks are
neighbors/neighbors_graph_run.mojo's).
"""
from std.memory import unsafe_memcpy
from core.neighbors_native_driver import XnDriver
from neighbors.xn_request_run import xn_request_run, xn_scratch_f32, xn_scratch_i32
from neighbors.neighbors_graph_run import (
    xn_relabel_request, xn_relabel_decode, xn_fill_request, xn_iota_request, xn_unit_request, xn_unit_check,
    xn_pagerank_csr_request, xn_pagerank_dense_request, xn_pagerank_decode, xn_cc_csr_request, xn_cc_dense_request,
    xn_cc_csr_components, xn_cc_relabel_check, xn_louvain_csr_request, xn_louvain_request,
    xn_graph_symmetry_request, xn_louvain_symmetry_check, xn_louvain_device_message, xn_louvain_relabel_check,
    xn_louvain_decode, xn_leiden_csr_request, xn_leiden_request, xn_leiden_symmetry_check,
    xn_leiden_device_message, xn_leiden_relabel_check, xn_i32_word, xn_snn_request, xn_snn_room,
    xn_graph_device_message, xn_paga_request, xn_paga_csr_request,
)

comptime _GRAPH_I32P = MutPointer[Int32, MutUntrackedOrigin]

#: Capability bit 2 of the image: it carries the `unit_ff` operation.
comptime XN_CAPABILITY_UNIT = 4


def _f32_at(values: List[Float32]) -> Int:
    return Int(values.unsafe_ptr())


def _i32_at(values: List[Int32]) -> Int:
    return Int(values.unsafe_ptr())


def xn_capabilities_run(driver: XnDriver) raises -> Int:
    """The image's capability word (the `capabilities` operation)."""
    var a: Array[Int, 0] = []
    var p: Array[Int, 0] = []
    var f: Array[Float64, 0] = []
    return driver.run("capabilities", a, p, f)


# ---------------------------------------------------------------- PageRank


def pagerank_unit_run(driver: XnDriver, v: Int, dst: Int, n: Int, what: String) raises:
    """`PageRank._unit(v, n, what)`: a caller's length-n non-negative vector
    divided by its sum on the device, with its refusal."""
    var info = xn_scratch_i32(2)
    _ = xn_request_run(driver, xn_unit_request(v, dst, _i32_at(info), n))
    xn_unit_check(_i32_at(info), what)
    _ = info^


@fieldwise_init
struct PageRankRequest(Copyable, Movable):
    # `csr`: the adjacency as int32 `indptr` / `indices` (`nnz` entries) and
    # float32 `vals` (0: no weights); else `adjacency` is the dense float32
    # n x n matrix. `x` (n float32) is the published result and the start
    # vector. `personalization`, `dangling`, `nstart`: the caller's admitted
    # float32 vectors (0: uniform; a missing `dangling` follows the
    # personalization).
    var csr: Bool
    var indptr: Int
    var indices: Int
    var vals: Int
    var nnz: Int
    var adjacency: Int
    var x: Int
    var n: Int
    var personalization: Int
    var dangling: Int
    var nstart: Int
    var max_iter: Int
    var tol: Float64
    var binary: Bool
    var alpha: Float64


def pagerank_fit_run(driver: XnDriver, request: PageRankRequest) raises -> Tuple[Int, Bool]:
    """The power iteration as one resident operation. Returns (iterations,
    converged)."""
    var n = request.n
    var p_uniform = request.personalization == 0
    var x_uniform = request.nstart == 0
    var p = xn_scratch_f32(n)
    var own_dw = xn_scratch_f32(0 if request.dangling == 0 else n)
    if not p_uniform:
        pagerank_unit_run(driver, request.personalization, _f32_at(p), n, "personalization")
    elif not request.csr and n != 0:
        # `uniform()`: [1/n] * n; the CSR operation takes the uniform flag instead
        _ = xn_request_run(driver, xn_fill_request(_f32_at(p), n))
    var dw = _f32_at(p)
    if request.dangling != 0:
        dw = _f32_at(own_dw)
        pagerank_unit_run(driver, request.dangling, dw, n, "dangling")
    if not x_uniform:
        pagerank_unit_run(driver, request.nstart, request.x, n, "nstart")
    elif not request.csr and n != 0:
        _ = xn_request_run(driver, xn_fill_request(request.x, n))
    var info = xn_scratch_i32(2)
    if request.csr:
        var has_vals = request.vals != 0
        var spare = xn_scratch_f32(1)
        var vals = request.vals if has_vals else _f32_at(spare)
        _ = xn_request_run(driver, xn_pagerank_csr_request(request.indptr, request.indices, vals, request.x,
                                                           _f32_at(p), dw, _i32_at(info), n, request.nnz, has_vals,
                                                           request.max_iter, request.tol, request.binary, x_uniform,
                                                           p_uniform, request.dangling == 0 and p_uniform,
                                                           request.alpha))
        _ = spare^
    else:
        _ = xn_request_run(driver, xn_pagerank_dense_request(request.adjacency, request.x, _f32_at(p), dw,
                                                             _i32_at(info), n, request.max_iter, request.tol,
                                                             request.binary, request.alpha))
    var got = xn_pagerank_decode(_i32_at(info))
    _ = info^
    _ = p^
    _ = own_dw^
    return got


# ---------------------------------------------------------------- connected components


@fieldwise_init
struct ComponentsRequest(Copyable, Movable):
    # `csr`: the adjacency as int32 `indptr` / `indices` (`nnz` entries);
    # else `adjacency` is the dense float32 n x n matrix. `labels` (n int32)
    # receives the component labels, renumbered by first occurrence.
    # `scope_block`: 0, or the door's `resident()` scope block for the dense
    # adjacency (bindings/function_scope_door.mojo `ScopeBlock`; lane
    # gd-resident-2): the GPU image reads A from its store in place.
    var csr: Bool
    var indptr: Int
    var indices: Int
    var nnz: Int
    var adjacency: Int
    var labels: Int
    var n: Int
    var scope_block: Int


def components_run(driver: XnDriver, request: ComponentsRequest) raises -> Int:
    """The label iteration: ONE operation on either column (lane vdf-misc2,
    2026-11-24) that starts from the identity labels, runs the rounds,
    relabels the components in order of first appearance and counts them
    (info[1] = count + 1). Before, three operations, each a round trip of
    the n labels through the host on a device image: p2m_iota, the rounds,
    then p2m_relabel into scratch copied back here. Returns the component
    count."""
    var n = request.n
    var info = xn_scratch_i32(2)
    if request.csr:
        _ = xn_request_run(driver, xn_cc_csr_request(request.indptr, request.indices, request.labels, _i32_at(info),
                                                     n, request.nnz))
    else:
        _ = xn_request_run(driver, xn_cc_dense_request(request.adjacency, request.labels, _i32_at(info), n,
                                                       request.scope_block))
    var components = xn_cc_csr_components(_i32_at(info))
    _ = info^
    xn_cc_relabel_check(components >= 0 or n == 0)
    return max(components, 0)


# ---------------------------------------------------------------- Louvain


@fieldwise_init
struct LouvainRequest(Copyable, Movable):
    # `csr`: the adjacency as int32 `indptr` / `indices` (`nnz` entries) and
    # float32 `vals` (0: no weights); else `adjacency` is the dense float32
    # n x n matrix. `labels` (max(n, 1) int32) receives the community labels
    # renumbered by first occurrence. `max_level` 0: no limit.
    var csr: Bool
    var indptr: Int
    var indices: Int
    var vals: Int
    var nnz: Int
    var adjacency: Int
    var labels: Int
    var n: Int
    var max_level: Int
    var resolution: Float64
    var threshold: Float64


struct LouvainOutcome(Copyable, Movable):
    var communities: Int
    var modularity: Float64
    var levels: Float64

    def __init__(out self):
        self.communities = 0
        self.modularity = 0.0
        self.levels = 0.0


def louvain_fit_run(driver: XnDriver, request: LouvainRequest) raises -> LouvainOutcome:
    """The aggregation levels as one resident operation, then the
    renumbering. The image's own checks keep their words, as ValueErrors
    from "Louvain:" on."""
    var outcome = LouvainOutcome()
    var n = request.n
    var raw = xn_scratch_i32(n)
    var info = xn_scratch_f32(2)
    if request.csr:
        var has_vals = request.vals != 0
        var spare = xn_scratch_f32(1)
        var vals = request.vals if has_vals else _f32_at(spare)
        try:
            _ = xn_request_run(driver, xn_louvain_csr_request(request.indptr, request.indices, vals, _i32_at(raw),
                                                              _f32_at(info), n, request.nnz, has_vals,
                                                              request.max_level, request.resolution,
                                                              request.threshold))
        except error:
            var refusal = xn_louvain_device_message(String(error))
            if refusal.byte_length() == 0:
                raise error^
            raise Error(refusal)
        _ = spare^
    else:
        # the symmetry and no-edge checks run natively (`graph_symmetry`)
        var flags = xn_scratch_i32(2)
        _ = xn_request_run(driver, xn_graph_symmetry_request(request.adjacency, _i32_at(flags), n))
        xn_louvain_symmetry_check(_i32_at(flags))
        _ = flags^
        _ = xn_request_run(driver, xn_louvain_request(request.adjacency, _i32_at(raw), _f32_at(info), n,
                                                      request.max_level, request.resolution, request.threshold))
    var rinfo = xn_scratch_i32(2)
    _ = xn_request_run(driver, xn_relabel_request(_i32_at(raw), request.labels, _i32_at(rinfo), n))
    var got = xn_relabel_decode(_i32_at(rinfo))
    xn_louvain_relabel_check(got[1])
    var decoded = xn_louvain_decode(_f32_at(info))
    outcome.communities = got[0]
    outcome.modularity = decoded[0]
    outcome.levels = decoded[1]
    _ = raw^
    _ = info^
    _ = rinfo^
    return outcome^


# ---------------------------------------------------------------- Leiden


@fieldwise_init
struct LeidenRequest(Copyable, Movable):
    # The adjacency as for LouvainRequest. `n_iterations` positive, or -1
    # (until stable); `quality` XN_LEIDEN_MODULARITY / XN_LEIDEN_CPM; `seed`
    # the colouring's random_state.
    var csr: Bool
    var indptr: Int
    var indices: Int
    var vals: Int
    var nnz: Int
    var adjacency: Int
    var labels: Int
    var n: Int
    var n_iterations: Int
    var quality: Int
    var seed: Int
    var resolution: Float64


struct LeidenOutcome(Copyable, Movable):
    var communities: Int
    var quality: Float64
    var iterations: Float64

    def __init__(out self):
        self.communities = 0
        self.quality = 0.0
        self.iterations = 0.0


def leiden_fit_run(driver: XnDriver, request: LeidenRequest) raises -> LeidenOutcome:
    """Leiden's iterations as one resident operation, then the renumbering
    (Louvain's `p2m_relabel`). The image's own checks keep their words, as
    ValueErrors from "Leiden:" on."""
    var outcome = LeidenOutcome()
    var n = request.n
    var raw = xn_scratch_i32(n)
    var info = xn_scratch_f32(2)
    if request.csr:
        var has_vals = request.vals != 0
        var spare = xn_scratch_f32(1)
        var vals = request.vals if has_vals else _f32_at(spare)
        try:
            _ = xn_request_run(driver, xn_leiden_csr_request(request.indptr, request.indices, vals, _i32_at(raw),
                                                             _f32_at(info), n, request.nnz, has_vals,
                                                             request.n_iterations, request.quality, request.seed,
                                                             request.resolution))
        except error:
            var refusal = xn_leiden_device_message(String(error))
            if refusal.byte_length() == 0:
                raise error^
            raise Error(refusal)
        _ = spare^
    else:
        var flags = xn_scratch_i32(2)
        _ = xn_request_run(driver, xn_graph_symmetry_request(request.adjacency, _i32_at(flags), n))
        xn_leiden_symmetry_check(_i32_at(flags))
        _ = flags^
        try:
            _ = xn_request_run(driver, xn_leiden_request(request.adjacency, _i32_at(raw), _f32_at(info), n,
                                                         request.n_iterations, request.quality, request.seed,
                                                         request.resolution))
        except error:
            var refusal = xn_leiden_device_message(String(error))
            if refusal.byte_length() == 0:
                raise error^
            raise Error(refusal)
    var rinfo = xn_scratch_i32(2)
    _ = xn_request_run(driver, xn_relabel_request(_i32_at(raw), request.labels, _i32_at(rinfo), n))
    var got = xn_relabel_decode(_i32_at(rinfo))
    xn_leiden_relabel_check(got[1])
    var decoded = xn_louvain_decode(_f32_at(info))
    outcome.communities = got[0]
    outcome.quality = decoded[0]
    outcome.iterations = decoded[1]
    _ = raw^
    _ = info^
    _ = rinfo^
    return outcome^


# ---------------------------------------------------------------- shared-nearest-neighbor graph


@fieldwise_init
struct SnnRequest(Copyable, Movable):
    # The neighbor lists: int32 `indptr` (n + 1) and `indices` (nnz), or,
    # when `k_uniform` > 0, `indices` an (n, k_uniform) int32 array and
    # `indptr` unused. `out_indptr` (n + 1 int32) is the published row
    # offsets; `prune` Seurat's prune.SNN.
    var indptr: Int
    var indices: Int
    var nnz: Int
    var k_uniform: Int
    var out_indptr: Int
    var n: Int
    var prune: Float64


struct SnnOutcome(Movable):
    """The result's columns and weights (native scratch, `nnz` of them
    filled) for PACK to publish."""
    var indices: List[Int32]
    var data: List[Float32]
    var nnz: Int

    def __init__(out self):
        self.indices = List[Int32]()
        self.data = List[Float32]()
        self.nnz = 0


def snn_run(driver: XnDriver, request: SnnRequest) raises -> SnnOutcome:
    """The `snn` operation with a first room of `xn_snn_room` entries; when
    the result is larger the image has counted it (info[0]) and wrote
    nothing, and the second run has exactly that room. The image's own
    checks keep their words, as ValueErrors from "shared_neighbors_graph:"
    on."""
    var out = SnnOutcome()
    var info = xn_scratch_i32(2)
    var spare = xn_scratch_i32(1)
    var indptr = request.indptr if request.k_uniform == 0 else _i32_at(spare)
    var cap = xn_snn_room(request.n, request.nnz)
    var done = False
    for _ in range(2):  # driver-loop: the first room, then the room the image counted
        out.indices = List[Int32](unsafe_uninit_length=max(cap, 1))
        out.data = List[Float32](unsafe_uninit_length=max(cap, 1))
        try:
            _ = xn_request_run(driver, xn_snn_request(indptr, request.indices, request.out_indptr,
                                                      _i32_at(out.indices), _f32_at(out.data), _i32_at(info),
                                                      request.n, request.nnz, request.k_uniform, cap,
                                                      request.prune))
        except error:
            var refusal = xn_graph_device_message(String(error), "shared_neighbors_graph:")
            if refusal.byte_length() == 0:
                raise error^
            raise Error(refusal)
        var total = xn_i32_word(_i32_at(info), 0)
        if xn_i32_word(_i32_at(info), 1) == 0:
            out.nnz = total
            done = True
            break
        cap = total
    _ = spare^
    _ = info^
    if not done:
        raise Error("RuntimeError: shared_neighbors_graph: the image refused the room it counted")
    return out^


# ---------------------------------------------------------------- PAGA


@fieldwise_init
struct PagaRequest(Copyable, Movable):
    # The graph as for LouvainRequest (no weights: every stored edge counts
    # once); `labels` the int32 group of each of the n nodes.
    var csr: Bool
    var indptr: Int
    var indices: Int
    var nnz: Int
    var adjacency: Int
    var labels: Int
    var n: Int


struct PagaOutcome(Movable):
    """c groups; the c x c connectivities and tree (row-major float32) and
    the group sizes, native scratch for PACK to publish."""
    var groups: Int
    var connectivities: List[Float32]
    var tree: List[Float32]
    var sizes: List[Int32]

    def __init__(out self):
        self.groups = 0
        self.connectivities = List[Float32]()
        self.tree = List[Float32]()
        self.sizes = List[Int32]()


def _paga_call(driver: XnDriver, request: PagaRequest, csr: Bool, indptr: Int, indices: Int, nnz: Int, cap: Int,
               conn: Int, tree: Int, sizes: Int, info: Int) raises:
    """One `paga` / `paga_csr` operation; the image's checks as ValueErrors
    from "PAGA:" on."""
    try:
        if csr:
            _ = xn_request_run(driver, xn_paga_csr_request(indptr, indices, request.labels, conn, tree, sizes, info,
                                                           request.n, nnz, cap))
        else:
            _ = xn_request_run(driver, xn_paga_request(request.adjacency, request.labels, conn, tree, sizes, info,
                                                       request.n, cap))
    except error:
        var refusal = xn_graph_device_message(String(error), "PAGA:")
        if refusal.byte_length() == 0:
            raise error^
        raise Error(refusal)


def paga_run(driver: XnDriver, request: PagaRequest) raises -> PagaOutcome:
    """Two operations: the sizing one (the CSR entry with no room and no
    edges: the labels go up, the group count comes back, no graph is
    touched), then the whole PAGA with c x c room on the caller's form."""
    var out = PagaOutcome()
    var info = xn_scratch_i32(2)
    var spare = xn_scratch_i32(1)
    var spare_f = xn_scratch_f32(1)
    _paga_call(driver, request, True, _i32_at(spare), _i32_at(spare), 0, 0, _f32_at(spare_f), _f32_at(spare_f),
               _i32_at(spare), _i32_at(info))
    var c = xn_i32_word(_i32_at(info), 0)
    if c > request.n:
        # c x c words of state: groups are numbered 0 .. c - 1, at most one per node
        raise Error("ValueError: PAGA: group labels must be numbered from 0 (largest label "
                    + String(c - 1) + " for " + String(request.n) + " nodes)")
    out.groups = c
    out.connectivities = List[Float32](unsafe_uninit_length=max(c * c, 1))
    out.tree = List[Float32](unsafe_uninit_length=max(c * c, 1))
    out.sizes = List[Int32](unsafe_uninit_length=max(c, 1))
    var indices = request.indices if request.nnz > 0 else _i32_at(spare)
    _paga_call(driver, request, request.csr, request.indptr, indices, request.nnz, c * c, _f32_at(out.connectivities),
               _f32_at(out.tree), _i32_at(out.sizes), _i32_at(info))
    if xn_i32_word(_i32_at(info), 1) != 0 or xn_i32_word(_i32_at(info), 0) != c:
        raise Error("RuntimeError: PAGA: the group count changed between the two operations")
    _ = spare^
    _ = spare_f^
    _ = info^
    return out^
