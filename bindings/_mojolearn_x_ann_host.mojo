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
from x_ann.abi import check_search, in_f32, in_i32, out_f32, out_i32, p_int, ptr_f32, ptr_i32
from x_ann.ivf_pq_core import pq_len_of
from x_ann.host.cagra_host import cagra_build_host, cagra_search_host
from x_ann.host.tsne_host import tsne_fit_host
from x_ann.host.ivf_pq_host import X_ANN_HOST_SABOTAGE, ivf_pq_build_host, ivf_pq_search_host, ivf_sq_build_host, ivf_sq_search_host, refine_host, ivf_rabitq_build_host, ivf_rabitq_search_host
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
    _ = n
    _ = pq_len
    var centers = ptr_f32(addrs, 0)
    var offsets = ptr_i32(addrs, 1)
    var list_indices = ptr_i32(addrs, 2)
    var cb = ptr_f32(addrs, 3)
    var codes = ptr_i32(addrs, 4)
    var queries = ptr_f32(addrs, 5)
    var od = ptr_f32(addrs, 6)
    var oi = ptr_i32(addrs, 7)
    var on = ptr_i32(addrs, 8)
    var mask = ptr_i32(addrs, 9)
    with GILReleased(Python()):
        ivf_pq_search_host(centers, offsets, list_indices, cb, codes, mask, n_lists, dim, pq_dim, pq_bits,
                           queries, m, k, n_probes, od, oi, on)
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
    var x = ptr_f32(addrs, 0)
    var g = List[Int32]()
    with GILReleased(Python()):
        g = cagra_build_host(x, n, d, kdeg, deg)
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
    var x = ptr_f32(addrs, 0)
    var g = ptr_i32(addrs, 1)
    for e in range(n * deg):
        if Int(g.unsafe_load(e)) < 0 or Int(g.unsafe_load(e)) >= n:
            raise Error("CAGRA search: the graph names a row outside the dataset")
    var q = ptr_f32(addrs, 2)
    var od = ptr_f32(addrs, 3)
    var oi = ptr_i32(addrs, 4)
    with GILReleased(Python()):
        cagra_search_host(x, n, d, g, deg, q, m, k, L, width, max_iter, n_seeds, od, oi, rs)
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
        ivf_sq_build_host(x, n, dim, n_lists, iters, seed, centers, offsets, list_indices, vmin, delta, codes)
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
    _ = n
    var centers = ptr_f32(addrs, 0)
    var offsets = ptr_i32(addrs, 1)
    var list_indices = ptr_i32(addrs, 2)
    var vmin = ptr_f32(addrs, 3)
    var delta = ptr_f32(addrs, 4)
    var codes = ptr_i32(addrs, 5)
    var mask = ptr_i32(addrs, 6)
    var queries = ptr_f32(addrs, 7)
    var od = ptr_f32(addrs, 8)
    var oi = ptr_i32(addrs, 9)
    var on = ptr_i32(addrs, 10)
    with GILReleased(Python()):
        ivf_sq_search_host(centers, offsets, list_indices, vmin, delta, codes, mask, n_lists, dim, queries, m, k,
                           n_probes, od, oi, on)
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
    var x = ptr_f32(addrs, 0)
    var q = ptr_f32(addrs, 1)
    var cand = ptr_i32(addrs, 2)
    var od = ptr_f32(addrs, 3)
    var oi = ptr_i32(addrs, 4)
    with GILReleased(Python()):
        refine_host(x, n, d, q, m, cand, k0, k, od, oi)
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
        ivf_rabitq_build_host(x, n, dim, n_lists, iters, seed, centers, offsets, list_indices, codes, norms, ips)
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
    _ = n
    _ = words
    var centers = ptr_f32(addrs, 0)
    var offsets = ptr_i32(addrs, 1)
    var list_indices = ptr_i32(addrs, 2)
    var codes = ptr_i32(addrs, 3)
    var norms = ptr_f32(addrs, 4)
    var ips = ptr_f32(addrs, 5)
    var mask = ptr_i32(addrs, 6)
    var queries = ptr_f32(addrs, 7)
    var od = ptr_f32(addrs, 8)
    var oi = ptr_i32(addrs, 9)
    var on = ptr_i32(addrs, 10)
    with GILReleased(Python()):
        ivf_rabitq_search_host(centers, offsets, list_indices, codes, norms, ips, mask, n_lists, dim, seed,
                               queries, m, k, n_probes, od, oi, on)
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
        m.def_function[ivf_sq_build_binding]("x_ann_ivf_sq_build")
        m.def_function[ivf_sq_search_binding]("x_ann_ivf_sq_search")
        m.def_function[refine_binding]("x_ann_refine")
        m.def_function[ivf_rabitq_build_binding]("x_ann_ivf_rabitq_build")
        m.def_function[ivf_rabitq_search_binding]("x_ann_ivf_rabitq_search")
        m.def_function[numeric_mode_binding]("x_ann_numeric_mode")
        m.def_function[vendor_binding]("x_ann_vendor")
        m.def_function[numeric_mode_binding]("x_ann_host_numeric_mode")
        m.def_function[vendor_binding]("x_ann_host_vendor")
        m.def_function[host_column_binding]("x_ann_host_column")
        m.def_function[host_sabotage_binding]("x_ann_host_sabotage")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_ann_host: ", e))
