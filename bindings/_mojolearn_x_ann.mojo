# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE ANN LANE'S GPU BINDING (lane/algos-ann): IVF-PQ, t-SNE, CAGRA and the
IVF quantization arms. Every entry takes (addrs, params) lists; the orders
are written in x_ann/abi.mojo's callers below and in
python/mojolearn/_expansion_ann.py. `bindings/_mojolearn_x_ann_host.mojo`
exports the same names with the same contract on the CPU."""
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from x_ann.filter_topk import filter_topk_device
from x_ann.tsne_init import TSNE_INIT_GIVEN, TSNE_INIT_RANDOM, X_ANN_PY2MOJO, tsne_init_y0
from checks.vendor import COMPILED_VENDOR
from checks.numerics import GLOBAL_NUMERIC_MODE
from x_ann.abi import a_int, check_search, in_f32, in_i32, out_f32, out_i32, p_int
from x_ann.switches import ANN3_DIRECT_OUT
from x_ann.ivf_pq_core import pq_len_of
from x_ann.stage_timer import AnnStages
from x_ann.cagra_device import cagra_build_device, cagra_search_device, cagra_search_on
from x_ann.device_ctx import x_ann_ctx
from x_ann.io import upload_f32, upload_i32
from core.abs_sum_blocked import device_any_index_out_of_range
from x_ann.tsne_device import tsne_fit_device
from x_ann.resident import x_ann_index_prepare_binding, x_ann_index_release_binding, x_ann_index_search_binding
from x_ann.ivf_pq_device import ivf_pq_build_device, ivf_pq_search_device, ivf_sq_build_device, ivf_sq_search_device, refine_device, ivf_rabitq_build_device, ivf_rabitq_search_device
from x_ann.ivf_pq_device import refine_device_team
from x_ann.vsearch_fast import IVF_REFINE_TEAM


def ivf_pq_build_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """addrs: x, centers, offsets, list_indices, codebooks, codes.
    params: n, dim, n_lists, kmeans_n_iters, seed, pq_dim, pq_bits, pq_kmeans_n_iters."""
    var n = p_int(params, 0)
    var dim = p_int(params, 1)
    var bst = AnnStages("ivf_pq_binding")
    var x = in_f32(addrs, 0, n * dim)
    bst.host("copy_in")
    var n_lists = p_int(params, 2)
    var iters = p_int(params, 3)
    var seed = p_int(params, 4)
    var pq_dim = p_int(params, 5)
    var pq_bits = p_int(params, 6)
    var pq_iters = p_int(params, 7)
    # lane ann-apple3, behind `ANN3_DIRECT_OUT`: the codes go from the device
    # straight into the caller's array (`index.codes` is then empty and the
    # copy below moves nothing)
    var codes_addr = 0
    comptime if ANN3_DIRECT_OUT:
        codes_addr = a_int(addrs, 5)
    with GILReleased(Python()):
        var index = ivf_pq_build_device(x, n, dim, n_lists, iters, seed, pq_dim, pq_bits, pq_iters, codes_addr)
        bst.host("build")
        out_f32(index.centers, addrs, 1)
        out_i32(index.offsets, addrs, 2)
        out_i32(index.list_indices, addrs, 3)
        out_f32(index.codebooks, addrs, 4)
        out_i32(index.codes, addrs, 5)
        bst.host("copy_out")
    return PythonObject(n)


def ivf_pq_search_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """addrs: centers, offsets, list_indices, codebooks, codes, queries, out_d, out_i, out_n, mask
    (int32 per row, 0 removes the row: the sample filter).
    params: n, dim, n_lists, pq_dim, pq_bits, m, k, n_probes."""
    var n = p_int(params, 0)
    var dim = p_int(params, 1)
    var n_lists = p_int(params, 2)
    var pq_dim = p_int(params, 3)
    var pq_bits = p_int(params, 4)
    var m = p_int(params, 5)
    var k = p_int(params, 6)
    var n_probes = p_int(params, 7)
    check_search(n_lists, m, k, n_probes)
    var pq_len = pq_len_of(dim, pq_dim)
    var centers = in_f32(addrs, 0, n_lists * dim)
    var offsets = in_i32(addrs, 1, n_lists + 1)
    var list_indices = in_i32(addrs, 2, n)
    var cb = in_f32(addrs, 3, pq_dim * (1 << pq_bits) * pq_len)
    var codes = in_i32(addrs, 4, n * pq_dim)
    var queries = in_f32(addrs, 5, m * dim)
    var mask = in_i32(addrs, 9, n)
    var od = List[Float32]()
    var oi = List[Int32]()
    var on = List[Int32]()
    with GILReleased(Python()):
        ivf_pq_search_device(centers, offsets, list_indices, cb, codes, mask, n_lists, dim, pq_dim, pq_bits,
                             queries, m, k, n_probes, od, oi, on)
    out_f32(od, addrs, 6)
    out_i32(oi, addrs, 7)
    out_i32(on, addrs, 8)
    return PythonObject(m)


def tsne_fit_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """addrs: x, y0, y_out, kl_out (one float32).
    params: n, d, max_iter, exploration_iters, perplexity, early_exaggeration, learning_rate
    [, init mode, random_state low 32 bits, high 32 bits] (lane apple-fast-py2mojo-cluster: mode 1
    random, 2 the PCA embedding in y0 scaled, x_ann/tsne_init.mojo; 0 or absent y0 as given)."""
    var n = p_int(params, 0)
    var d = p_int(params, 1)
    var max_iter = p_int(params, 2)
    var exploration = p_int(params, 3)
    var perplexity = Float32(Float64(py=params[4]))
    var exaggeration = Float32(Float64(py=params[5]))
    var lr = Float32(Float64(py=params[6]))
    var x = in_f32(addrs, 0, n * d)
    var mode = p_int(params, 7) if len(params) > 7 else TSNE_INIT_GIVEN
    var seed = UInt64(0)
    if len(params) > 9:
        seed = (UInt64(p_int(params, 9)) << 32) | UInt64(p_int(params, 8))
    var y0 = tsne_init_y0(
        mode, in_f32(addrs, 1, n * 2) if mode != TSNE_INIT_RANDOM else List[Float32](), n, seed
    )
    var y = List[Float32]()
    var kl = Float32(0.0)
    with GILReleased(Python()):
        tsne_fit_device(x, n, d, y0, perplexity, exaggeration, lr, max_iter, exploration, y, kl)
    out_f32(y, addrs, 2)
    out_f32([kl], addrs, 3)
    return PythonObject(n)


def cagra_build_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """addrs: x, graph_out. params: n, d, intermediate_graph_degree, graph_degree."""
    var n = p_int(params, 0)
    var d = p_int(params, 1)
    var kdeg = p_int(params, 2)
    var deg = p_int(params, 3)
    if n < 2 or d <= 0 or kdeg < 1 or kdeg > n - 1 or deg < 1 or deg > kdeg:
        raise Error("CAGRA: need 1 <= graph_degree <= intermediate_graph_degree <= n - 1")
    var x = in_f32(addrs, 0, n * d)
    var g = List[Int32]()
    with GILReleased(Python()):
        g = cagra_build_device(x, n, d, kdeg, deg)
    out_i32(g, addrs, 1)
    return PythonObject(n)


def cagra_search_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """addrs: x, graph, queries, out_d, out_i.
    params: n, d, graph_degree, m, k, itopk_size, search_width, max_iterations, n_seeds[, rs]
    (rs: 0 the evenly spaced seeds, > 0 random_state's, x_ann/cagra_core.mojo cg_seed_node)."""
    var n = p_int(params, 0)
    var d = p_int(params, 1)
    var deg = p_int(params, 2)
    var m = p_int(params, 3)
    var k = p_int(params, 4)
    var L = p_int(params, 5)
    var width = p_int(params, 6)
    var max_iter = p_int(params, 7)
    var n_seeds = p_int(params, 8)
    var rs = p_int(params, 9) if len(params) > 9 else 0
    if rs < 0:
        raise Error("CAGRA search: rs must be >= 0")
    if m <= 0 or k <= 0 or L < k or width < 1 or max_iter < 1 or n_seeds < 1 or n_seeds > n:
        raise Error("CAGRA search: need k >= 1, itopk_size >= k, search_width >= 1, max_iterations >= 1, 1 <= n_seeds <= n")
    var x = in_f32(addrs, 0, n * d)
    var g = in_i32(addrs, 1, n * deg)
    var q = in_f32(addrs, 2, m * d)
    var od = List[Float32]()
    var oi = List[Int32]()
    with GILReleased(Python()):
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
            max_iter, n_seeds, od, oi, rs,
        )
        _ = dg^
        _ = dx^
        _ = ctx^
    out_f32(od, addrs, 3)
    out_i32(oi, addrs, 4)
    return PythonObject(m)


def ivf_sq_build_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """addrs: x, centers, offsets, list_indices, vmin, delta, codes.
    params: n, dim, n_lists, kmeans_n_iters, seed."""
    var n = p_int(params, 0)
    var dim = p_int(params, 1)
    var n_lists = p_int(params, 2)
    var iters = p_int(params, 3)
    var seed = p_int(params, 4)
    var bst = AnnStages("ivf_sq_binding")
    var x = in_f32(addrs, 0, n * dim)
    bst.host("copy_in")
    var centers = List[Float32]()
    var offsets = List[Int32]()
    var list_indices = List[Int32]()
    var vmin = List[Float32]()
    var delta = List[Float32]()
    var codes = List[Int32]()
    var codes_addr = 0
    comptime if ANN3_DIRECT_OUT:
        codes_addr = a_int(addrs, 6)
    with GILReleased(Python()):
        ivf_sq_build_device(x, n, dim, n_lists, iters, seed, centers, offsets, list_indices, vmin, delta, codes,
                            codes_addr)
    bst.host("build")
    out_f32(centers, addrs, 1)
    out_i32(offsets, addrs, 2)
    out_i32(list_indices, addrs, 3)
    out_f32(vmin, addrs, 4)
    out_f32(delta, addrs, 5)
    out_i32(codes, addrs, 6)
    bst.host("copy_out")
    return PythonObject(n)


def ivf_sq_search_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """addrs: centers, offsets, list_indices, vmin, delta, codes, mask, queries, out_d, out_i, out_n.
    params: n, dim, n_lists, m, k, n_probes."""
    var n = p_int(params, 0)
    var dim = p_int(params, 1)
    var n_lists = p_int(params, 2)
    var m = p_int(params, 3)
    var k = p_int(params, 4)
    var n_probes = p_int(params, 5)
    check_search(n_lists, m, k, n_probes)
    var centers = in_f32(addrs, 0, n_lists * dim)
    var offsets = in_i32(addrs, 1, n_lists + 1)
    var list_indices = in_i32(addrs, 2, n)
    var vmin = in_f32(addrs, 3, dim)
    var delta = in_f32(addrs, 4, dim)
    var codes = in_i32(addrs, 5, n * dim)
    var mask = in_i32(addrs, 6, n)
    var queries = in_f32(addrs, 7, m * dim)
    var od = List[Float32]()
    var oi = List[Int32]()
    var on = List[Int32]()
    with GILReleased(Python()):
        ivf_sq_search_device(centers, offsets, list_indices, vmin, delta, codes, mask, n_lists, dim, queries, m, k,
                             n_probes, od, oi, on)
    out_f32(od, addrs, 8)
    out_i32(oi, addrs, 9)
    out_i32(on, addrs, 10)
    return PythonObject(m)


def refine_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """addrs: dataset, queries, candidates (int32 m x k0, < 0 = padding), out_d, out_i.
    params: n, d, m, k0, k[, root] (root 1: the euclidean metric's square
    roots of the kept distances, lane apple-fast-py2mojo-cluster)."""
    var n = p_int(params, 0)
    var d = p_int(params, 1)
    var m = p_int(params, 2)
    var k0 = p_int(params, 3)
    var k = p_int(params, 4)
    var root = len(params) > 5 and p_int(params, 5) != 0
    if n <= 0 or d <= 0 or m <= 0 or k0 <= 0 or k <= 0 or k > k0:
        raise Error("refine: need positive shapes and 1 <= k <= n_candidates")
    comptime if IVF_REFINE_TEAM:
        # lane af-vsearch, FAST on Apple, OPT-IN: the dataset goes up from the
        # caller's array (no List copy of n x d first); the array outlives the call
        var x_addr = a_int(addrs, 0)
        var tq = in_f32(addrs, 1, m * d)
        var tcand = in_i32(addrs, 2, m * k0)
        var tod = List[Float32]()
        var toi = List[Int32]()
        with GILReleased(Python()):
            refine_device_team(x_addr, n, d, tq, m, tcand, k0, k, tod, toi, root)
        out_f32(tod, addrs, 3)
        out_i32(toi, addrs, 4)
        return PythonObject(m)
    var x = in_f32(addrs, 0, n * d)
    var q = in_f32(addrs, 1, m * d)
    var cand = in_i32(addrs, 2, m * k0)
    var od = List[Float32]()
    var oi = List[Int32]()
    with GILReleased(Python()):
        refine_device(x, n, d, q, m, cand, k0, k, od, oi, root)
    out_f32(od, addrs, 3)
    out_i32(oi, addrs, 4)
    return PythonObject(m)


def ivf_rabitq_build_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """addrs: x, centers, offsets, list_indices, codes (n x words int32 bit words), norms, ips.
    params: n, dim, n_lists, kmeans_n_iters, seed."""
    var n = p_int(params, 0)
    var dim = p_int(params, 1)
    var n_lists = p_int(params, 2)
    var iters = p_int(params, 3)
    var seed = p_int(params, 4)
    var bst = AnnStages("ivf_rabitq_binding")
    var x = in_f32(addrs, 0, n * dim)
    bst.host("copy_in")
    var centers = List[Float32]()
    var offsets = List[Int32]()
    var list_indices = List[Int32]()
    var codes = List[Int32]()
    var norms = List[Float32]()
    var ips = List[Float32]()
    with GILReleased(Python()):
        ivf_rabitq_build_device(x, n, dim, n_lists, iters, seed, centers, offsets, list_indices, codes, norms, ips)
    bst.host("build")
    out_f32(centers, addrs, 1)
    out_i32(offsets, addrs, 2)
    out_i32(list_indices, addrs, 3)
    out_i32(codes, addrs, 4)
    out_f32(norms, addrs, 5)
    out_f32(ips, addrs, 6)
    bst.host("copy_out")
    return PythonObject(n)


def ivf_rabitq_search_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """addrs: centers, offsets, list_indices, codes, norms, ips, mask, queries, out_d, out_i, out_n.
    params: n, dim, n_lists, seed, m, k, n_probes."""
    var n = p_int(params, 0)
    var dim = p_int(params, 1)
    var n_lists = p_int(params, 2)
    var seed = p_int(params, 3)
    var m = p_int(params, 4)
    var k = p_int(params, 5)
    var n_probes = p_int(params, 6)
    check_search(n_lists, m, k, n_probes)
    var D = 1
    while D < dim:
        D *= 2
    var words = (D + 31) // 32
    var centers = in_f32(addrs, 0, n_lists * dim)
    var offsets = in_i32(addrs, 1, n_lists + 1)
    var list_indices = in_i32(addrs, 2, n)
    var codes = in_i32(addrs, 3, n * words)
    var norms = in_f32(addrs, 4, n)
    var ips = in_f32(addrs, 5, n)
    var mask = in_i32(addrs, 6, n)
    var queries = in_f32(addrs, 7, m * dim)
    var od = List[Float32]()
    var oi = List[Int32]()
    var on = List[Int32]()
    with GILReleased(Python()):
        ivf_rabitq_search_device(centers, offsets, list_indices, codes, norms, ips, mask, n_lists, dim, seed,
                                 queries, m, k, n_probes, od, oi, on)
    out_f32(od, addrs, 8)
    out_i32(oi, addrs, 9)
    out_i32(on, addrs, 10)
    return PythonObject(m)


def filter_topk_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """CAGRA's filtered compaction (x_ann/filter_topk.mojo, lane
    apple-fast-py2mojo-cluster). addrs: itopk distances (m x L), itopk ids
    (m x L int32), keep (n int32), out_d, out_i (m x k). params: m, n, L, k."""
    var m = p_int(params, 0)
    var n = p_int(params, 1)
    var L = p_int(params, 2)
    var k = p_int(params, 3)
    if m <= 0 or n <= 0 or k <= 0 or L < k:
        raise Error("CAGRA filtered search: need m, n, k >= 1 and itopk_size >= k")
    var bd = in_f32(addrs, 0, m * L)
    var bi = in_i32(addrs, 1, m * L)
    var keep = in_i32(addrs, 2, n)
    var od = List[Float32]()
    var oi = List[Int32]()
    with GILReleased(Python()):
        filter_topk_device(bd, bi, keep, m, n, L, k, od, oi)
    out_f32(od, addrs, 3)
    out_i32(oi, addrs, 4)
    return PythonObject(m)


def py2mojo_binding() raises -> PythonObject:
    """1: TSNE's start, refine's euclidean roots and CAGRA's filtered
    compaction run in this binding (lane apple-fast-py2mojo-cluster); 0 under
    `-D MOJOLEARN_PY2MOJO_cluster_OFF`, and `_expansion_ann.py` runs them."""
    return PythonObject(1 if X_ANN_PY2MOJO else 0)


def numeric_mode_binding() raises -> PythonObject:
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))


def vendor_binding() raises -> PythonObject:
    return PythonObject(String(COMPILED_VENDOR))


@export
def PyInit__mojolearn_x_ann() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_x_ann")
        m.def_function[ivf_pq_build_binding]("x_ann_ivf_pq_build")
        m.def_function[ivf_pq_search_binding]("x_ann_ivf_pq_search")
        m.def_function[tsne_fit_binding]("x_ann_tsne_fit")
        m.def_function[cagra_build_binding]("x_ann_cagra_build")
        m.def_function[cagra_search_binding]("x_ann_cagra_search")
        m.def_function[ivf_sq_build_binding]("x_ann_ivf_sq_build")
        m.def_function[ivf_sq_search_binding]("x_ann_ivf_sq_search")
        m.def_function[refine_binding]("x_ann_refine")
        m.def_function[py2mojo_binding]("x_ann_py2mojo")
        m.def_function[filter_topk_binding]("x_ann_filter_topk")
        m.def_function[ivf_rabitq_build_binding]("x_ann_ivf_rabitq_build")
        m.def_function[ivf_rabitq_search_binding]("x_ann_ivf_rabitq_search")
        m.def_function[x_ann_index_prepare_binding]("x_ann_index_prepare")
        m.def_function[x_ann_index_search_binding]("x_ann_index_search")
        m.def_function[x_ann_index_release_binding]("x_ann_index_release")
        m.def_function[numeric_mode_binding]("x_ann_numeric_mode")
        m.def_function[vendor_binding]("x_ann_vendor")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_ann: ", e))
