# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE ANN LANE'S CPU (HOST) BINDING (lane/algos-ann): IVF-PQ, t-SNE, CAGRA and the
IVF quantization arms. Every entry takes (addrs, params) lists; the orders
are written in x_ann/abi.mojo's callers below and in
python/mojolearn/_expansion_ann.py. `bindings/_mojolearn_x_ann_host.mojo`
exports the same names with the same contract on the CPU."""
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from checks.numerics import GLOBAL_NUMERIC_MODE
from x_ann.abi import check_search, in_f32, in_i32, out_f32, out_i32, p_int
from x_ann.ivf_pq_core import pq_len_of
from x_ann.host.cagra_host import cagra_build_host, cagra_search_host
from x_ann.host.tsne_host import tsne_fit_host
from x_ann.host.ivf_pq_host import X_ANN_HOST_SABOTAGE, ivf_pq_build_host, ivf_pq_search_host
from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN, column_name


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
        var index = ivf_pq_build_host(x, n, dim, n_lists, iters, seed, pq_dim, pq_bits, pq_iters)
        out_f32(index.centers, addrs, 1)
        out_i32(index.offsets, addrs, 2)
        out_i32(index.list_indices, addrs, 3)
        out_f32(index.codebooks, addrs, 4)
        out_i32(index.codes, addrs, 5)
    return PythonObject(n)


def ivf_pq_search_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """addrs: centers, offsets, list_indices, codebooks, codes, queries, out_d, out_i, out_n.
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
    var od = List[Float32]()
    var oi = List[Int32]()
    var on = List[Int32]()
    with GILReleased(Python()):
        ivf_pq_search_host(centers, offsets, list_indices, cb, codes, n_lists, dim, pq_dim, pq_bits,
                             queries, m, k, n_probes, od, oi, on)
    out_f32(od, addrs, 6)
    out_i32(oi, addrs, 7)
    out_i32(on, addrs, 8)
    return PythonObject(m)


def tsne_fit_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """addrs: x, y0, y_out, kl_out (one float32).
    params: n, d, max_iter, exploration_iters, perplexity, early_exaggeration, learning_rate."""
    var n = p_int(params, 0)
    var d = p_int(params, 1)
    var max_iter = p_int(params, 2)
    var exploration = p_int(params, 3)
    var perplexity = Float32(Float64(py=params[4]))
    var exaggeration = Float32(Float64(py=params[5]))
    var lr = Float32(Float64(py=params[6]))
    var x = in_f32(addrs, 0, n * d)
    var y0 = in_f32(addrs, 1, n * 2)
    var y = List[Float32]()
    var kl = Float32(0.0)
    with GILReleased(Python()):
        tsne_fit_host(x, n, d, y0, perplexity, exaggeration, lr, max_iter, exploration, y, kl)
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
        g = cagra_build_host(x, n, d, kdeg, deg)
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
        cagra_search_host(x, n, d, g, deg, q, m, k, L, width, max_iter, n_seeds, od, oi)
    out_f32(od, addrs, 3)
    out_i32(oi, addrs, 4)
    return PythonObject(m)


def numeric_mode_binding() raises -> PythonObject:
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))


def vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def host_column_binding() raises -> PythonObject:
    comptime assert TARGET_COLUMN == COLUMN_CPU, "x_ann host: pass -D MOJOLEARN_COLUMN_CPU"
    return PythonObject(column_name(TARGET_COLUMN))


def host_sabotage_binding() raises -> PythonObject:
    return PythonObject(X_ANN_HOST_SABOTAGE)


@export
def PyInit__mojolearn_x_ann_host() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_x_ann_host")
        m.def_function[ivf_pq_build_binding]("x_ann_ivf_pq_build")
        m.def_function[ivf_pq_search_binding]("x_ann_ivf_pq_search")
        m.def_function[tsne_fit_binding]("x_ann_tsne_fit")
        m.def_function[cagra_build_binding]("x_ann_cagra_build")
        m.def_function[cagra_search_binding]("x_ann_cagra_search")
        m.def_function[numeric_mode_binding]("x_ann_numeric_mode")
        m.def_function[vendor_binding]("x_ann_vendor")
        m.def_function[numeric_mode_binding]("x_ann_host_numeric_mode")
        m.def_function[vendor_binding]("x_ann_host_vendor")
        m.def_function[host_column_binding]("x_ann_host_column")
        m.def_function[host_sabotage_binding]("x_ann_host_sabotage")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_ann_host: ", e))
