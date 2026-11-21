# SPDX-License-Identifier: Apache-2.0
"""PageRank, Louvain and connected_components RUN: typed requests in, typed
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
    xn_louvain_decode,
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
    var csr: Bool
    var indptr: Int
    var indices: Int
    var nnz: Int
    var adjacency: Int
    var labels: Int
    var n: Int


def components_run(driver: XnDriver, request: ComponentsRequest) raises -> Int:
    """The label iteration, then the renumbering when the operation did not
    already count the components. Returns the component count."""
    var n = request.n
    if n > 0:
        _ = xn_request_run(driver, xn_iota_request(request.labels, n))
    if request.csr:
        var info = xn_scratch_i32(2)
        _ = xn_request_run(driver, xn_cc_csr_request(request.indptr, request.indices, request.labels, _i32_at(info),
                                                     n, request.nnz))
        var components = xn_cc_csr_components(_i32_at(info))
        _ = info^
        if components >= 0:
            return components
    else:
        var info = xn_scratch_i32(1)
        _ = xn_request_run(driver, xn_cc_dense_request(request.adjacency, request.labels, _i32_at(info), n))
        _ = info^
    if n == 0:
        return 0
    var renumbered = xn_scratch_i32(n)
    var rinfo = xn_scratch_i32(2)
    _ = xn_request_run(driver, xn_relabel_request(request.labels, _i32_at(renumbered), _i32_at(rinfo), n))
    var got = xn_relabel_decode(_i32_at(rinfo))
    xn_cc_relabel_check(got[1])
    unsafe_memcpy(dest=_GRAPH_I32P(unsafe_from_address=request.labels),
                  src=_GRAPH_I32P(unsafe_from_address=_i32_at(renumbered)), count=n)
    _ = renumbered^
    _ = rinfo^
    return got[0]


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
    return outcome
