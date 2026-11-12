# SPDX-License-Identifier: Apache-2.0
"""The device column of the ANN operations table
(core/neighbors_entry_table_ann.mojo): the typed bodies of the GPU binding's
Python exports (bindings/_mojolearn_x_ann.mojo `x_ann_ivf_pq_search`,
`x_ann_ivf_sq_search`, `x_ann_ivf_rabitq_search`, `x_ann_cagra_search`,
`x_ann_filter_topk`, `x_ann_tsne_fit`) and the resident index operations
(x_ann/resident.mojo). The exports UNPACK their lists, release the GIL and
call the same `*_run` body the table entry calls: the same checks, the same
copies, the same kernels and the same words, so the bits are unchanged. The
table entry is called with the GIL already released by its caller (the
resident operations with it held, see the table's module note); nothing
here touches Python.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.hostptr import read_f32, read_i32
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot
from core.abs_sum_blocked import device_any_index_out_of_range
from core.neighbors_entry_table_ann import (
    ANN_OP_INDEX_PREPARE, ANN_OP_INDEX_SEARCH, ANN_OP_INDEX_RELEASE, ANN_OP_INDEX_EXPORT_CODES,
    ANN_OP_IVF_PQ_SEARCH, ANN_OP_IVF_SQ_SEARCH, ANN_OP_IVF_RABITQ_SEARCH, ANN_OP_CAGRA_SEARCH,
    ANN_OP_FILTER_TOPK, ANN_OP_TSNE_FIT, ann_ops_check_spans, ann_span_list,
)
from x_ann.abi import check_search, out_f32_to, out_i32_to
from x_ann.abi_shell_impl import rabitq_padded_dim
from x_ann.filter_topk import filter_topk_device
from x_ann.tsne_init import TSNE_INIT_GIVEN, TSNE_INIT_RANDOM, tsne_first_components, tsne_init_y0
from x_ann.tsne_device import tsne_fit_device
from x_ann.ivf_pq_core import pq_len_of
from x_ann.cagra_device import cagra_search_on
from x_ann.device_ctx import x_ann_ctx
from x_ann.io import upload_f32, upload_i32
from x_ann.ivf_pq_device import ivf_pq_search_device, ivf_sq_search_device, ivf_rabitq_search_device
from x_ann.resident import x_ann_index_export_codes, x_ann_index_prepare, x_ann_index_release, x_ann_index_search


def _opt(p: List[Int], at: Int, default: Int) -> Int:
    # The exports' `p_int(params, at) if len(params) > at else default`.
    return p[at] if len(p) > at else default


def ann_ivf_pq_search_device_run(a: List[Int], p: List[Int]) raises -> Int:
    """`x_ann_ivf_pq_search`: a centers, offsets, list_indices, codebooks,
    codes, queries, out_d, out_i, out_n, mask; p n, dim, n_lists, pq_dim,
    pq_bits, m, k, n_probes[, status]. Returns m."""
    var n = p[0]
    var dim = p[1]
    var n_lists = p[2]
    var pq_dim = p[3]
    var pq_bits = p[4]
    var m = p[5]
    var k = p[6]
    var n_probes = p[7]
    var status = _opt(p, 8, 0)
    check_search(n_lists, m, k, n_probes)
    var pq_len = pq_len_of(dim, pq_dim)
    var centers = read_f32(a[0], n_lists * dim)
    var offsets = read_i32(a[1], n_lists + 1)
    var list_indices = read_i32(a[2], n)
    var cb = read_f32(a[3], pq_dim * (1 << pq_bits) * pq_len)
    var codes = read_i32(a[4], n * pq_dim)
    var queries = read_f32(a[5], m * dim)
    var mask = read_i32(a[9], n)
    var od = List[Float32]()
    var oi = List[Int32]()
    var on = List[Int32]()
    ivf_pq_search_device(centers, offsets, list_indices, cb, codes, mask, n_lists, dim, pq_dim, pq_bits,
                         queries, m, k, n_probes, od, oi, on, status)
    out_f32_to(od, a[6])
    out_i32_to(oi, a[7])
    out_i32_to(on, a[8])
    return m


def ann_ivf_sq_search_device_run(a: List[Int], p: List[Int]) raises -> Int:
    """`x_ann_ivf_sq_search`: a centers, offsets, list_indices, vmin, delta,
    codes, mask, queries, out_d, out_i, out_n; p n, dim, n_lists, m, k,
    n_probes[, status]. Returns m."""
    var n = p[0]
    var dim = p[1]
    var n_lists = p[2]
    var m = p[3]
    var k = p[4]
    var n_probes = p[5]
    var status = _opt(p, 6, 0)
    check_search(n_lists, m, k, n_probes)
    var centers = read_f32(a[0], n_lists * dim)
    var offsets = read_i32(a[1], n_lists + 1)
    var list_indices = read_i32(a[2], n)
    var vmin = read_f32(a[3], dim)
    var delta = read_f32(a[4], dim)
    var codes = read_i32(a[5], n * dim)
    var mask = read_i32(a[6], n)
    var queries = read_f32(a[7], m * dim)
    var od = List[Float32]()
    var oi = List[Int32]()
    var on = List[Int32]()
    ivf_sq_search_device(centers, offsets, list_indices, vmin, delta, codes, mask, n_lists, dim, queries, m, k,
                         n_probes, od, oi, on, status)
    out_f32_to(od, a[8])
    out_i32_to(oi, a[9])
    out_i32_to(on, a[10])
    return m


def ann_ivf_rabitq_search_device_run(a: List[Int], p: List[Int]) raises -> Int:
    """`x_ann_ivf_rabitq_search`: a centers, offsets, list_indices, codes,
    norms, ips, mask, queries, out_d, out_i, out_n; p n, dim, n_lists, seed,
    m, k, n_probes[, status]. Returns m."""
    var n = p[0]
    var dim = p[1]
    var n_lists = p[2]
    var seed = p[3]
    var m = p[4]
    var k = p[5]
    var n_probes = p[6]
    var status = _opt(p, 7, 0)
    check_search(n_lists, m, k, n_probes)
    var D = rabitq_padded_dim(dim)
    var words = (D + 31) // 32
    var centers = read_f32(a[0], n_lists * dim)
    var offsets = read_i32(a[1], n_lists + 1)
    var list_indices = read_i32(a[2], n)
    var codes = read_i32(a[3], n * words)
    var norms = read_f32(a[4], n)
    var ips = read_f32(a[5], n)
    var mask = read_i32(a[6], n)
    var queries = read_f32(a[7], m * dim)
    var od = List[Float32]()
    var oi = List[Int32]()
    var on = List[Int32]()
    ivf_rabitq_search_device(centers, offsets, list_indices, codes, norms, ips, mask, n_lists, dim, seed,
                             queries, m, k, n_probes, od, oi, on, status)
    out_f32_to(od, a[8])
    out_i32_to(oi, a[9])
    out_i32_to(on, a[10])
    return m


def ann_cagra_search_device_run(a: List[Int], p: List[Int]) raises -> Int:
    """`x_ann_cagra_search`: a x, graph, queries, out_d, out_i; p n, d,
    graph_degree, m, k, itopk_size, search_width, max_iterations,
    n_seeds[, rs[, status]]. Returns m."""
    var n = p[0]
    var d = p[1]
    var deg = p[2]
    var m = p[3]
    var k = p[4]
    var L = p[5]
    var width = p[6]
    var max_iter = p[7]
    var n_seeds = p[8]
    var rs = _opt(p, 9, 0)
    var status = _opt(p, 10, 0)
    if rs < 0:
        raise Error("CAGRA search: rs must be >= 0")
    if m <= 0 or k <= 0 or L < k or width < 1 or max_iter < 1 or n_seeds < 1 or n_seeds > n:
        raise Error("CAGRA search: need k >= 1, itopk_size >= k, search_width >= 1, max_iterations >= 1, 1 <= n_seeds <= n")
    var x = read_f32(a[0], n * d)
    var g = read_i32(a[1], n * deg)
    var q = read_f32(a[2], m * d)
    var od = List[Float32]()
    var oi = List[Int32]()
    # cpu3-bindings: `cagra_search_device`'s uploads, then the graph's
    # row-id range refusal as one device pass over the uploaded graph
    # (one Int32 back) instead of a host walk over n * deg ids
    var ctx = x_ann_ctx()
    var dx = upload_f32(ctx, x)
    var dg = upload_i32(ctx, g)
    if device_any_index_out_of_range(ctx, dg, n * deg, n):
        raise Error("CAGRA search: the graph names a row outside the dataset")
    cagra_search_on(
        ctx, dx.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), n, d,
        dg.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), deg, q, m, k, L, width,
        max_iter, n_seeds, od, oi, rs, status,
    )
    _ = dg^
    _ = dx^
    _ = ctx^
    out_f32_to(od, a[3])
    out_i32_to(oi, a[4])
    return m


def ann_filter_topk_device_run(a: List[Int], p: List[Int]) raises -> Int:
    """`x_ann_filter_topk` (x_ann/filter_topk.mojo): a itopk distances,
    itopk ids, keep, out_d, out_i; p m, n, L, k. Returns m."""
    var m = p[0]
    var n = p[1]
    var L = p[2]
    var k = p[3]
    if m <= 0 or n <= 0 or k <= 0 or L < k:
        raise Error("CAGRA filtered search: need m, n, k >= 1 and itopk_size >= k")
    var bd = read_f32(a[0], m * L)
    var bi = read_i32(a[1], m * L)
    var keep = read_i32(a[2], n)
    var od = List[Float32]()
    var oi = List[Int32]()
    filter_topk_device(bd, bi, keep, m, n, L, k, od, oi)
    out_f32_to(od, a[3])
    out_i32_to(oi, a[4])
    return m


def ann_tsne_fit_device_run(a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    """`x_ann_tsne_fit`: a x, y0, y_out, kl_out; p n, d, max_iter,
    exploration[, init mode[, seed low, seed high[, n_components[, finite
    flag]]]] (the export's params[8:] follow p[3]); f perplexity,
    early_exaggeration, learning_rate, post factor. Returns X's finite status
    word (0 without the flag); a flagged X writes neither output."""
    var n = p[0]
    var d = p[1]
    var max_iter = p[2]
    var exploration = p[3]
    var perplexity = Float32(f[0])
    var exaggeration = Float32(f[1])
    var lr = Float32(f[2])
    var post_factor = Float32(f[3])
    var x = read_f32(a[0], n * d)
    var mode = _opt(p, 4, TSNE_INIT_GIVEN)
    var seed = UInt64(0)
    if len(p) > 6:
        seed = (UInt64(p[6]) << 32) | UInt64(p[5])
    var k = _opt(p, 7, 2)
    var y0 = tsne_init_y0(
        mode, read_f32(a[1], n * k) if mode != TSNE_INIT_RANDOM else List[Float32](), n, seed, k
    )
    var test = len(p) > 8 and p[8] != 0
    var word = List[Int32](length=1, fill=Int32(0))
    var word_addr = Int(word.unsafe_ptr()) if test else 0
    var y = List[Float32]()
    var kl = Float32(0.0)
    tsne_fit_device(x, n, d, y0, perplexity, exaggeration, lr, post_factor, max_iter, exploration, y, kl,
                    word_addr)
    var status = Int(word[0])
    _ = word^  # the device wrote it through word_addr; keep it alive past the read
    if test and status != 0:
        return status
    var kl_words: List[Float32] = [kl]
    out_f32_to(tsne_first_components(y, n, k), a[2])
    out_f32_to(kl_words, a[3])
    return 0


def ann_ops_device_run(operation: Int, pa: MutPointer[Int, MutUntrackedOrigin], na: Int,
                       pp: MutPointer[Int, MutUntrackedOrigin], np: Int,
                       pf: MutPointer[Float64, MutUntrackedOrigin], nf: Int) raises -> Int:
    var a = ann_span_list(pa, 0, na)
    if operation == ANN_OP_INDEX_PREPARE:
        return x_ann_index_prepare(pp[0], a, ann_span_list(pp, 1, np - 1))
    if operation == ANN_OP_INDEX_SEARCH:
        return x_ann_index_search(pp[0], a, ann_span_list(pp, 1, np - 1))
    if operation == ANN_OP_INDEX_RELEASE:
        x_ann_index_release(pp[0])
        return 0
    if operation == ANN_OP_INDEX_EXPORT_CODES:
        x_ann_index_export_codes(pp[0], a[0], pp[1])
        return 0
    var p = ann_span_list(pp, 0, np)
    if operation == ANN_OP_IVF_PQ_SEARCH:
        return ann_ivf_pq_search_device_run(a, p)
    if operation == ANN_OP_IVF_SQ_SEARCH:
        return ann_ivf_sq_search_device_run(a, p)
    if operation == ANN_OP_IVF_RABITQ_SEARCH:
        return ann_ivf_rabitq_search_device_run(a, p)
    if operation == ANN_OP_CAGRA_SEARCH:
        return ann_cagra_search_device_run(a, p)
    if operation == ANN_OP_FILTER_TOPK:
        return ann_filter_topk_device_run(a, p)
    # ANN_OP_TSNE_FIT (ann_ops_check_spans refused any other id)
    var f = List[Float64](capacity=nf)
    for i in range(nf):  # small-loop(no: t-SNE's four float settings): metadata, not data
        f.append(pf[i])
    return ann_tsne_fit_device_run(a, p, f)


def ann_ops_device_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        ann_ops_check_spans(operation, aa, na, pp, np, ff, nf, result)
        var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
        var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
        var f = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
        var r = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)
        r[] = ann_ops_device_run(operation, a, na, p, np, f, nf)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct AnnOpsDeviceAPI(Defaultable, Movable):
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(ann_ops_device_v1)


comptime _OPS_SLOT = "MojolearnAnnOpsDeviceIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnAnnOpsDeviceFastV1"
# The publisher identity of this image's table (its entry list's address).
comptime ANN_OPS_DEVICE_API = SharedSlot[AnnOpsDeviceAPI, _OPS_SLOT]
