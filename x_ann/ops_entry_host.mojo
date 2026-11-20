# SPDX-License-Identifier: Apache-2.0
"""The CPU column of the ANN operations table
(core/neighbors_entry_table_ann.mojo): the typed bodies of the host binding's
Python exports (bindings/_mojolearn_x_ann_host.mojo `x_ann_ivf_pq_search`,
`x_ann_ivf_sq_search`, `x_ann_ivf_rabitq_search`, `x_ann_cagra_search`,
`x_ann_filter_topk`, `x_ann_tsne_fit`), which read and write the caller's
arrays in place. The exports UNPACK their lists, release the GIL and call the
same `*_run` body the table entry calls: the same checks, the same kernels
and the same words. The host binding has no resident index, so the resident
operations are refused here, as the absent exports were. The table entry is
called with the GIL already released by its caller; nothing here touches
Python.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.hostptr import f32_ptr, i32_ptr, read_f32
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot
from core.host_finite_scan import host_all_finite_f32, host_finite_status
from core.neighbors_entry_table_ann import (
    ANN_OP_IVF_PQ_SEARCH, ANN_OP_IVF_SQ_SEARCH, ANN_OP_IVF_RABITQ_SEARCH, ANN_OP_CAGRA_SEARCH,
    ANN_OP_FILTER_TOPK, ANN_OP_TSNE_FIT, ann_ops_check_spans, ann_span_list,
)
from x_ann.abi import check_search, out_f32_to
from x_ann.abi_shell_impl import cagra_check_graph
from x_ann.filter_topk import filter_topk_host
from x_ann.tsne_init import TSNE_INIT_GIVEN, TSNE_INIT_RANDOM, tsne_first_components, tsne_init_y0
from x_ann.ivf_pq_core import F32P, I32P
from x_ann.host.cagra_host import cagra_search_host
from x_ann.host.tsne_host import tsne_fit_host
from x_ann.host.ivf_pq_host import ivf_pq_search_host, ivf_sq_search_host, ivf_rabitq_search_host


def _pf(address: Int) raises -> F32P:
    # The exports' `ptr_f32`: the caller's float32 array itself (no copy).
    return rebind[F32P](f32_ptr(address))


def _pi(address: Int) raises -> I32P:
    return rebind[I32P](i32_ptr(address))


def ann_query_status_host(p: List[Int], at: Int, queries: Int, n: Int) raises -> Bool:
    """The CPU column's answer to a search's optional status word `p[at]`
    (lane a8b-device-x-neighbors-rest; the GPU binding tests the queries
    behind their upload): one parallel SIMD scan of the `n` query words,
    written to the word (0 finite, 1 not). True when the search may run: no
    word was asked, or the words are finite (a flagged search is skipped and
    the owner refuses)."""
    if len(p) <= at:
        return True
    var address = p[at]
    if address == 0:
        return True
    if address < 0 or address % 4 != 0:
        raise Error("x_ann: the status address must be a positive int32-aligned address")
    # parallel: host_finite_status splits the words across the host's cores
    var word = host_finite_status(queries, n)
    MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=address).unsafe_store(0, word)
    return word == 0


def ann_ivf_pq_search_host_run(a: List[Int], p: List[Int]) raises -> Int:
    """`x_ann_ivf_pq_search` on the host (the device body's words)."""
    var dim = p[1]
    var n_lists = p[2]
    var pq_dim = p[3]
    var pq_bits = p[4]
    var m = p[5]
    var k = p[6]
    var n_probes = p[7]
    check_search(n_lists, m, k, n_probes)
    var centers = _pf(a[0])
    var offsets = _pi(a[1])
    var list_indices = _pi(a[2])
    var cb = _pf(a[3])
    var codes = _pi(a[4])
    var queries = _pf(a[5])
    var od = _pf(a[6])
    var oi = _pi(a[7])
    var on = _pi(a[8])
    var mask = _pi(a[9])
    if not ann_query_status_host(p, 8, a[5], m * dim):
        return m
    ivf_pq_search_host(centers, offsets, list_indices, cb, codes, mask, n_lists, dim, pq_dim, pq_bits,
                       queries, m, k, n_probes, od, oi, on)
    return m


def ann_ivf_sq_search_host_run(a: List[Int], p: List[Int]) raises -> Int:
    """`x_ann_ivf_sq_search` on the host."""
    var dim = p[1]
    var n_lists = p[2]
    var m = p[3]
    var k = p[4]
    var n_probes = p[5]
    check_search(n_lists, m, k, n_probes)
    var centers = _pf(a[0])
    var offsets = _pi(a[1])
    var list_indices = _pi(a[2])
    var vmin = _pf(a[3])
    var delta = _pf(a[4])
    var codes = _pi(a[5])
    var mask = _pi(a[6])
    var queries = _pf(a[7])
    var od = _pf(a[8])
    var oi = _pi(a[9])
    var on = _pi(a[10])
    if not ann_query_status_host(p, 6, a[7], m * dim):
        return m
    ivf_sq_search_host(centers, offsets, list_indices, vmin, delta, codes, mask, n_lists, dim, queries, m, k,
                       n_probes, od, oi, on)
    return m


def ann_ivf_rabitq_search_host_run(a: List[Int], p: List[Int]) raises -> Int:
    """`x_ann_ivf_rabitq_search` on the host."""
    var dim = p[1]
    var n_lists = p[2]
    var seed = p[3]
    var m = p[4]
    var k = p[5]
    var n_probes = p[6]
    check_search(n_lists, m, k, n_probes)
    var centers = _pf(a[0])
    var offsets = _pi(a[1])
    var list_indices = _pi(a[2])
    var codes = _pi(a[3])
    var norms = _pf(a[4])
    var ips = _pf(a[5])
    var mask = _pi(a[6])
    var queries = _pf(a[7])
    var od = _pf(a[8])
    var oi = _pi(a[9])
    var on = _pi(a[10])
    if not ann_query_status_host(p, 7, a[7], m * dim):
        return m
    ivf_rabitq_search_host(centers, offsets, list_indices, codes, norms, ips, mask, n_lists, dim, seed,
                           queries, m, k, n_probes, od, oi, on)
    return m


def ann_cagra_search_host_run(a: List[Int], p: List[Int]) raises -> Int:
    """`x_ann_cagra_search` on the host."""
    var n = p[0]
    var d = p[1]
    var deg = p[2]
    var m = p[3]
    var k = p[4]
    var L = p[5]
    var width = p[6]
    var max_iter = p[7]
    var n_seeds = p[8]
    var rs = p[9] if len(p) > 9 else 0
    if rs < 0:
        raise Error("CAGRA search: rs must be >= 0")
    if m <= 0 or k <= 0 or L < k or width < 1 or max_iter < 1 or n_seeds < 1 or n_seeds > n:
        raise Error("CAGRA search: need k >= 1, itopk_size >= k, search_width >= 1, max_iterations >= 1, 1 <= n_seeds <= n")
    var x = _pf(a[0])
    var g = _pi(a[1])
    cagra_check_graph(g, n, deg)
    var q = _pf(a[2])
    var od = _pf(a[3])
    var oi = _pi(a[4])
    if not ann_query_status_host(p, 10, a[2], m * d):
        return m
    cagra_search_host(x, n, d, g, deg, q, m, k, L, width, max_iter, n_seeds, od, oi, rs)
    return m


def ann_filter_topk_host_run(a: List[Int], p: List[Int]) raises -> Int:
    """`x_ann_filter_topk` on the host."""
    var m = p[0]
    var n = p[1]
    var L = p[2]
    var k = p[3]
    if m <= 0 or n <= 0 or k <= 0 or L < k:
        raise Error("CAGRA filtered search: need m, n, k >= 1 and itopk_size >= k")
    filter_topk_host(_pf(a[0]), _pi(a[1]), _pi(a[2]), m, n, L, k, _pf(a[3]), _pi(a[4]))
    return m


def ann_tsne_fit_host_run(a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    """`x_ann_tsne_fit` on the host (the device body's words): the finite
    flag scans the host's own copy of X (the parallel host scan); a flagged
    X is not fit and status 1 comes back."""
    var n = p[0]
    var d = p[1]
    var max_iter = p[2]
    var exploration = p[3]
    var perplexity = Float32(f[0])
    var exaggeration = Float32(f[1])
    var lr = Float32(f[2])
    var post_factor = Float32(f[3])
    var x = read_f32(a[0], n * d)
    var mode = p[4] if len(p) > 4 else TSNE_INIT_GIVEN
    var seed = UInt64(0)
    if len(p) > 6:
        seed = (UInt64(p[6]) << 32) | UInt64(p[5])
    var k = p[7] if len(p) > 7 else 2
    var y0 = tsne_init_y0(
        mode, read_f32(a[1], n * k) if mode != TSNE_INIT_RANDOM else List[Float32](), n, seed, k
    )
    var test = len(p) > 8 and p[8] != 0
    if test and not host_all_finite_f32(
            MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(x.unsafe_ptr())), n * d):
        _ = x^  # the scan read it through its address; keep it alive past the scan
        return 1
    var y = List[Float32]()
    var kl = Float32(0.0)
    tsne_fit_host(x, n, d, y0, perplexity, exaggeration, lr, post_factor, max_iter, exploration, y, kl)
    var kl_words: List[Float32] = [kl]
    out_f32_to(tsne_first_components(y, n, k), a[2])
    out_f32_to(kl_words, a[3])
    return 0


def ann_ops_host_run(operation: Int, pa: MutPointer[Int, MutUntrackedOrigin], na: Int,
                     pp: MutPointer[Int, MutUntrackedOrigin], np: Int,
                     pf: MutPointer[Float64, MutUntrackedOrigin], nf: Int) raises -> Int:
    var a = ann_span_list(pa, 0, na)
    var p = ann_span_list(pp, 0, np)
    if operation == ANN_OP_IVF_PQ_SEARCH:
        return ann_ivf_pq_search_host_run(a, p)
    if operation == ANN_OP_IVF_SQ_SEARCH:
        return ann_ivf_sq_search_host_run(a, p)
    if operation == ANN_OP_IVF_RABITQ_SEARCH:
        return ann_ivf_rabitq_search_host_run(a, p)
    if operation == ANN_OP_CAGRA_SEARCH:
        return ann_cagra_search_host_run(a, p)
    if operation == ANN_OP_FILTER_TOPK:
        return ann_filter_topk_host_run(a, p)
    if operation == ANN_OP_TSNE_FIT:
        var f = List[Float64](capacity=nf)
        for i in range(nf):  # small-loop(no: t-SNE's four float settings): metadata, not data
            f.append(pf[i])
        return ann_tsne_fit_host_run(a, p, f)
    raise Error("ann ops: operation " + String(operation)
                + " needs a resident index, which the CPU column does not keep")


def ann_ops_host_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        ann_ops_check_spans(operation, aa, na, pp, np, ff, nf, result)
        var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
        var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
        var f = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
        var r = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)
        r[] = ann_ops_host_run(operation, a, na, p, np, f, nf)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct AnnOpsHostAPI(Defaultable, Movable):
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(ann_ops_host_v1)


comptime _OPS_SLOT = "MojolearnAnnOpsHostIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnAnnOpsHostFastV1"
# The publisher identity of this image's table (its entry list's address).
comptime ANN_OPS_HOST_API = SharedSlot[AnnOpsHostAPI, _OPS_SLOT]
