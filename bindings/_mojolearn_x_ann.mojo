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
from checks.vendor import COMPILED_VENDOR
from checks.numerics import GLOBAL_NUMERIC_MODE
from x_ann.abi import check_search, in_f32, in_i32, out_f32, out_i32, p_int
from x_ann.ivf_pq_core import pq_len_of
from x_ann.cagra_device import cagra_build_device, cagra_search_device
from x_ann.tsne_device import tsne_fit_device
from x_ann.ivf_pq_device import ivf_pq_build_device, ivf_pq_search_device, ivf_sq_build_device, ivf_sq_search_device, refine_device, ivf_rabitq_build_device, ivf_rabitq_search_device


def ivf_pq_build_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """addrs: x, centers, offsets, list_indices, codebooks, codes.
    params: n, dim, n_lists, kmeans_n_iters, seed, pq_dim, pq_bits, pq_kmeans_n_iters."""
    var n = p_int(params, 0)
    var dim = p_int(params, 1)
    var x = in_f32(addrs, 0, n * dim)
    var n_lists = p_int(params, 2)
    var iters = p_int(params, 3)
    var seed = p_int(params, 4)
    var pq_dim = p_int(params, 5)
    var pq_bits = p_int(params, 6)
    var pq_iters = p_int(params, 7)
    with GILReleased(Python()):
        var index = ivf_pq_build_device(x, n, dim, n_lists, iters, seed, pq_dim, pq_bits, pq_iters)
        out_f32(index.centers, addrs, 1)
        out_i32(index.offsets, addrs, 2)
        out_i32(index.list_indices, addrs, 3)
        out_f32(index.codebooks, addrs, 4)
        out_i32(index.codes, addrs, 5)
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
    """addrs: x, y0 (n x n_components), y_out, kl_out (one float32), n_iter_out (one int32).
    params: n, d, max_iter, exploration_iters, perplexity, early_exaggeration, learning_rate,
    n_components, exact (0/1), n_iter_without_progress, min_grad_norm."""
    var n = p_int(params, 0)
    var d = p_int(params, 1)
    var max_iter = p_int(params, 2)
    var exploration = p_int(params, 3)
    var perplexity = Float32(Float64(py=params[4]))
    var exaggeration = Float32(Float64(py=params[5]))
    var lr = Float32(Float64(py=params[6]))
    var nc = p_int(params, 7)
    var exact = p_int(params, 8) != 0
    var patience = p_int(params, 9)
    var min_grad_norm = Float32(Float64(py=params[10]))
    var x = in_f32(addrs, 0, n * d)
    var y0 = in_f32(addrs, 1, n * nc)
    var y = List[Float32]()
    var kl = Float32(0.0)
    var n_iter = 0
    with GILReleased(Python()):
        tsne_fit_device(x, n, d, nc, y0, perplexity, exaggeration, lr, max_iter, exploration, exact, patience, min_grad_norm,
            y, kl, n_iter)
    out_f32(y, addrs, 2)
    out_f32([kl], addrs, 3)
    out_i32([Int32(n_iter)], addrs, 4)
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
    params: n, d, graph_degree, m, k, itopk_size, search_width, max_iterations, n_seeds."""
    var n = p_int(params, 0)
    var d = p_int(params, 1)
    var deg = p_int(params, 2)
    var m = p_int(params, 3)
    var k = p_int(params, 4)
    var L = p_int(params, 5)
    var width = p_int(params, 6)
    var max_iter = p_int(params, 7)
    var n_seeds = p_int(params, 8)
    if m <= 0 or k <= 0 or L < k or width < 1 or max_iter < 1 or n_seeds < 1 or n_seeds > n:
        raise Error("CAGRA search: need k >= 1, itopk_size >= k, search_width >= 1, max_iterations >= 1, 1 <= n_seeds <= n")
    var x = in_f32(addrs, 0, n * d)
    var g = in_i32(addrs, 1, n * deg)
    for e in range(n * deg):
        if Int(g[e]) < 0 or Int(g[e]) >= n:
            raise Error("CAGRA search: the graph names a row outside the dataset")
    var q = in_f32(addrs, 2, m * d)
    var od = List[Float32]()
    var oi = List[Int32]()
    with GILReleased(Python()):
        cagra_search_device(x, n, d, g, deg, q, m, k, L, width, max_iter, n_seeds, od, oi)
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
    var x = in_f32(addrs, 0, n * dim)
    var centers = List[Float32]()
    var offsets = List[Int32]()
    var list_indices = List[Int32]()
    var vmin = List[Float32]()
    var delta = List[Float32]()
    var codes = List[Int32]()
    with GILReleased(Python()):
        ivf_sq_build_device(x, n, dim, n_lists, iters, seed, centers, offsets, list_indices, vmin, delta, codes)
    out_f32(centers, addrs, 1)
    out_i32(offsets, addrs, 2)
    out_i32(list_indices, addrs, 3)
    out_f32(vmin, addrs, 4)
    out_f32(delta, addrs, 5)
    out_i32(codes, addrs, 6)
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
    params: n, d, m, k0, k."""
    var n = p_int(params, 0)
    var d = p_int(params, 1)
    var m = p_int(params, 2)
    var k0 = p_int(params, 3)
    var k = p_int(params, 4)
    if n <= 0 or d <= 0 or m <= 0 or k0 <= 0 or k <= 0 or k > k0:
        raise Error("refine: need positive shapes and 1 <= k <= n_candidates")
    var x = in_f32(addrs, 0, n * d)
    var q = in_f32(addrs, 1, m * d)
    var cand = in_i32(addrs, 2, m * k0)
    var od = List[Float32]()
    var oi = List[Int32]()
    with GILReleased(Python()):
        refine_device(x, n, d, q, m, cand, k0, k, od, oi)
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
    var x = in_f32(addrs, 0, n * dim)
    var centers = List[Float32]()
    var offsets = List[Int32]()
    var list_indices = List[Int32]()
    var codes = List[Int32]()
    var norms = List[Float32]()
    var ips = List[Float32]()
    with GILReleased(Python()):
        ivf_rabitq_build_device(x, n, dim, n_lists, iters, seed, centers, offsets, list_indices, codes, norms, ips)
    out_f32(centers, addrs, 1)
    out_i32(offsets, addrs, 2)
    out_i32(list_indices, addrs, 3)
    out_i32(codes, addrs, 4)
    out_f32(norms, addrs, 5)
    out_f32(ips, addrs, 6)
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
        m.def_function[ivf_rabitq_build_binding]("x_ann_ivf_rabitq_build")
        m.def_function[ivf_rabitq_search_binding]("x_ann_ivf_rabitq_search")
        m.def_function[numeric_mode_binding]("x_ann_numeric_mode")
        m.def_function[vendor_binding]("x_ann_vendor")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_ann: ", e))
