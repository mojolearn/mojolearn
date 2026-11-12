# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE ANN LANE'S CPU (HOST) BINDING (lane/algos-ann): IVF-PQ, t-SNE, CAGRA and the
IVF quantization arms. Every entry takes (addrs, params) lists; the orders
are written in providers/python/x_ann_abi_pyapi.mojo's callers below and in
python/mojolearn/_expansion_ann.py. `bindings/_mojolearn_x_ann_host.mojo`
exports the same names with the same contract on the CPU."""
from core.experiment_guards_cluster import CLUSTER_GUARDS_OK
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from checks.numerics import GLOBAL_NUMERIC_MODE
from providers.python.x_ann_abi_pyapi import a_int, in_f32, out_f32, out_i32, p_int, ptr_f32, ptr_i32, py_int_list
from x_ann.abi import out_f32_to, out_i32_to
from x_ann.host.cagra_host import cagra_build_host
from x_ann.host.ivf_pq_host import X_ANN_HOST_SABOTAGE, ivf_pq_build_host, ivf_sq_build_host, refine_host, ivf_rabitq_build_host
from x_ann.ops_entry_host import (
    ANN_OPS_HOST_API, ann_ops_host_v1, ann_ivf_pq_search_host_run, ann_ivf_sq_search_host_run,
    ann_ivf_rabitq_search_host_run, ann_cagra_search_host_run, ann_filter_topk_host_run, ann_tsne_fit_host_run,
)
from core.neighbors_entry_table import NeighborsEntryKey
from core.neighbors_entry_table_ann import ann_ops_publish
from bindings.py_balanced import py_setattr
from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN, column_name
from core.host_finite_scan import host_finite_status


def _query_status(params: PythonObject, at: Int, queries: Int, n: Int,
                  other: Int = 0, n_other: Int = 0) raises -> Bool:
    """The CPU column's answer to a search's optional status word `params[at]`
    (lane a8b-device-x-neighbors-rest; the GPU binding tests the queries
    behind their upload): one parallel SIMD scan of the `n` query words,
    written to the word (0 finite, 1 not); `other` / `n_other`, when given,
    a second span in the same word (refine's dataset). True when the search
    may run: no word was asked, or the words are finite (a flagged search is
    skipped and the door refuses)."""
    if len(params) <= at:
        return True
    var address = p_int(params, at)
    if address == 0:
        return True
    if address < 0 or address % 4 != 0:
        raise Error("x_ann: the status address must be a positive int32-aligned address")
    # parallel: host_finite_status splits the words across the host's cores
    var word = host_finite_status(queries, n)
    if word == 0 and n_other > 0:
        word = host_finite_status(other, n_other)
    MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=address).unsafe_store(0, word)
    return word == 0


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
    # Admit Python addresses while holding the GIL; released work uses native values.
    var centers_addr = a_int(addrs, 1)
    var offsets_addr = a_int(addrs, 2)
    var indices_addr = a_int(addrs, 3)
    var codebooks_addr = a_int(addrs, 4)
    var codes_addr = a_int(addrs, 5)
    with GILReleased(Python()):
        var index = ivf_pq_build_host(x, n, dim, n_lists, iters, seed, pq_dim, pq_bits, pq_iters)
        out_f32_to(index.centers, centers_addr)
        out_i32_to(index.offsets, offsets_addr)
        out_i32_to(index.list_indices, indices_addr)
        out_f32_to(index.codebooks, codebooks_addr)
        out_i32_to(index.codes, codes_addr)
    return PythonObject(n)


def ivf_pq_search_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """addrs: centers, offsets, list_indices, codebooks, codes, queries, out_d, out_i, out_n, mask
    (int32 per row, 0 removes the row: the sample filter).
    params: n, dim, n_lists, pq_dim, pq_bits, m, k, n_probes[, status] (status: the host
    address of the int32 word that receives the queries' finite status).
    UNPACK and PACK only: the body is x_ann/ops_entry_host.mojo `ann_ivf_pq_search_host_run`,
    which the ANN operations table (core/neighbors_entry_table_ann.mojo) also
    calls. The export remains for python/mojolearn/_expansion_ann.py."""
    var a = py_int_list(addrs)
    var p = py_int_list(params)
    var m = 0
    with GILReleased(Python()):
        m = ann_ivf_pq_search_host_run(a, p)
    return PythonObject(m)


def tsne_fit_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """addrs: x, y0, y_out, kl_out (one float32).
    params: n, d, max_iter, exploration_iters, perplexity, early_exaggeration, learning_rate,
    post_exaggeration_learning_rate_factor [, init mode, random_state low 32 bits, high 32 bits]
    (lane apple-fast-py2mojo-cluster: mode 1 random, 2 the PCA embedding in y0 scaled,
    x_ann/tsne_init.mojo; 0 or absent y0 as given). release-0845: the factor is params[7] and the
    init tail moved from 7..9 to 8..10. Lane sk-kernels: an optional params[11], n_components
    (1 or 2, default 2); at 1, y0 and y_out hold n floats (`tsne_init_y0`). Lane a8d: an
    optional params[12], the finite flag: 1 asks for X's finite status word; the return is
    then [n, status word] and a flagged X writes neither output (the door refuses).
    UNPACK and PACK only: the body is x_ann/ops_entry_host.mojo `ann_tsne_fit_host_run`, which the ANN
    operations table (core/neighbors_entry_table_ann.mojo) also calls with
    the four float words apart (f) and params[8:] after params[3] (p). The
    export remains for python/mojolearn/_expansion_ann.py."""
    var a = py_int_list(addrs)
    var p = List[Int]()
    for i in range(len(params)):  # small-loop(no: one call's parameter words): metadata, not data
        if i < 4 or i > 7:
            p.append(Int(py=params[i]))
    var f: List[Float64] = [Float64(py=params[4]), Float64(py=params[5]), Float64(py=params[6]),
                            Float64(py=params[7])]
    var test = len(params) > 12 and p_int(params, 12) != 0
    var status = 0
    with GILReleased(Python()):
        status = ann_tsne_fit_host_run(a, p, f)
    if test:
        return Python.list(PythonObject(p[0]), PythonObject(status))
    return PythonObject(p[0])


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
    params: n, d, graph_degree, m, k, itopk_size, search_width, max_iterations, n_seeds[, rs[, status]]
    (rs: 0 the evenly spaced seeds, > 0 random_state's, x_ann/cagra_core.mojo cg_seed_node;
    status: the host address of the int32 word that receives the queries' finite status).
    UNPACK and PACK only: the body is x_ann/ops_entry_host.mojo `ann_cagra_search_host_run`,
    which the ANN operations table (core/neighbors_entry_table_ann.mojo) also
    calls. The export remains for python/mojolearn/_expansion_ann.py."""
    var a = py_int_list(addrs)
    var p = py_int_list(params)
    var m = 0
    with GILReleased(Python()):
        m = ann_cagra_search_host_run(a, p)
    return PythonObject(m)


def ivf_sq_build_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """addrs: x, centers, offsets, list_indices, vmin, delta, codes.
    params: n, dim, n_lists, kmeans_n_iters, seed[, train_rows_per_list] (release-0845:
    IVFSQIndex max_train_points_per_cluster; 0 or absent keeps the default training rows)."""
    var n = p_int(params, 0)
    var dim = p_int(params, 1)
    var n_lists = p_int(params, 2)
    var iters = p_int(params, 3)
    var seed = p_int(params, 4)
    var train_rows_per_list = p_int(params, 5) if len(params) > 5 else 0
    var x = in_f32(addrs, 0, n * dim)
    var centers = List[Float32]()
    var offsets = List[Int32]()
    var list_indices = List[Int32]()
    var vmin = List[Float32]()
    var delta = List[Float32]()
    var codes = List[Int32]()
    with GILReleased(Python()):
        ivf_sq_build_host(
            x, n, dim, n_lists, iters, seed, centers, offsets, list_indices, vmin, delta, codes,
            train_rows_per_list=train_rows_per_list,
        )
    out_f32(centers, addrs, 1)
    out_i32(offsets, addrs, 2)
    out_i32(list_indices, addrs, 3)
    out_f32(vmin, addrs, 4)
    out_f32(delta, addrs, 5)
    out_i32(codes, addrs, 6)
    return PythonObject(n)


def ivf_sq_search_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """addrs: centers, offsets, list_indices, vmin, delta, codes, mask, queries, out_d, out_i, out_n.
    params: n, dim, n_lists, m, k, n_probes[, status] (the queries' finite status word).
    UNPACK and PACK only: the body is x_ann/ops_entry_host.mojo `ann_ivf_sq_search_host_run`,
    which the ANN operations table (core/neighbors_entry_table_ann.mojo) also
    calls. The export remains for python/mojolearn/_expansion_ann.py."""
    var a = py_int_list(addrs)
    var p = py_int_list(params)
    var m = 0
    with GILReleased(Python()):
        m = ann_ivf_sq_search_host_run(a, p)
    return PythonObject(m)


def refine_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """addrs: dataset, queries, candidates (int32 m x k0, < 0 = padding), out_d, out_i.
    params: n, d, m, k0, k[, root[, status]] (root 1: the euclidean metric's square
    roots of the kept distances, lane apple-fast-py2mojo-cluster; status: the
    dataset's and queries' finite status word)."""
    var n = p_int(params, 0)
    var d = p_int(params, 1)
    var m = p_int(params, 2)
    var k0 = p_int(params, 3)
    var k = p_int(params, 4)
    var root = len(params) > 5 and p_int(params, 5) != 0
    if n <= 0 or d <= 0 or m <= 0 or k0 <= 0 or k <= 0 or k > k0:
        raise Error("refine: need positive shapes and 1 <= k <= n_candidates")
    var x = ptr_f32(addrs, 0)
    var q = ptr_f32(addrs, 1)
    var cand = ptr_i32(addrs, 2)
    var od = ptr_f32(addrs, 3)
    var oi = ptr_i32(addrs, 4)
    if not _query_status(params, 6, a_int(addrs, 1), m * d, a_int(addrs, 0), n * d):
        return PythonObject(m)
    with GILReleased(Python()):
        refine_host(x, n, d, q, m, cand, k0, k, od, oi, root)
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
    params: n, dim, n_lists, seed, m, k, n_probes[, status] (the queries' finite status word).
    UNPACK and PACK only: the body is x_ann/ops_entry_host.mojo `ann_ivf_rabitq_search_host_run`,
    which the ANN operations table (core/neighbors_entry_table_ann.mojo) also
    calls. The export remains for python/mojolearn/_expansion_ann.py."""
    var a = py_int_list(addrs)
    var p = py_int_list(params)
    var m = 0
    with GILReleased(Python()):
        m = ann_ivf_rabitq_search_host_run(a, p)
    return PythonObject(m)


def filter_topk_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """CAGRA's filtered compaction (x_ann/filter_topk.mojo, lane
    apple-fast-py2mojo-cluster). addrs: itopk distances (m x L), itopk ids
    (m x L int32), keep (n int32), out_d, out_i (m x k). params: m, n, L, k.
    UNPACK and PACK only: the body is x_ann/ops_entry_host.mojo `ann_filter_topk_host_run`,
    which the ANN operations table (core/neighbors_entry_table_ann.mojo) also
    calls. The export remains for python/mojolearn/_expansion_ann.py."""
    var a = py_int_list(addrs)
    var p = py_int_list(params)
    var m = 0
    with GILReleased(Python()):
        m = ann_filter_topk_host_run(a, p)
    return PythonObject(m)


# TOMBSTONE: x_ann_py2mojo (dead native entry: a constant A/B probe nothing reads) deleted 2026-10-26 by f9b52032a; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_x_ann.patch


def numeric_mode_binding() raises -> PythonObject:
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))


def vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def host_column_binding() raises -> PythonObject:
    comptime assert TARGET_COLUMN == COLUMN_CPU, "x_ann host: pass -D MOJOLEARN_COLUMN_CPU"
    return PythonObject(column_name(TARGET_COLUMN))


def host_sabotage_binding() raises -> PythonObject:
    return PythonObject(X_ANN_HOST_SABOTAGE)


def _publish_ann_ops(module: PythonObject) raises:
    # The CPU column of the ANN operations table
    # (core/neighbors_entry_table_ann.mojo): the ANN search and TSNE owners
    # call this entry with plain values and read the key from the module
    # attribute `x_ann_native_key`. The Python exports remain for
    # python/mojolearn/_expansion_ann.py.
    var api = ANN_OPS_HOST_API.get_or_create_ptr()
    ann_ops_publish(NeighborsEntryKey(Int(GLOBAL_NUMERIC_MODE), 0), Int(api[].entries.unsafe_ptr()),
                    ann_ops_host_v1)
    py_setattr(module, "x_ann_native_key", Python.tuple(PythonObject(Int(GLOBAL_NUMERIC_MODE)), PythonObject(0)))


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_x_ann_host.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).
def build__mojolearn_x_ann_host() -> PythonObject:
    comptime assert CLUSTER_GUARDS_OK, "invalid integrated experiment configuration (cluster guards)"
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
        # TOMBSTONE: x_ann_py2mojo (dead native entry: a constant A/B probe nothing reads) deleted 2026-10-26 by f9b52032a; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_x_ann.patch
        m.def_function[filter_topk_binding]("x_ann_filter_topk")
        m.def_function[ivf_rabitq_build_binding]("x_ann_ivf_rabitq_build")
        m.def_function[ivf_rabitq_search_binding]("x_ann_ivf_rabitq_search")
        m.def_function[numeric_mode_binding]("x_ann_numeric_mode")
        m.def_function[vendor_binding]("x_ann_vendor")
        m.def_function[numeric_mode_binding]("x_ann_host_numeric_mode")
        m.def_function[vendor_binding]("x_ann_host_vendor")
        m.def_function[host_column_binding]("x_ann_host_column")
        m.def_function[host_sabotage_binding]("x_ann_host_sabotage")
        var published_module = m.finalize()
        _publish_ann_ops(published_module)
        return published_module
    except e:
        abort(String("failed to create _mojolearn_x_ann_host: ", e))
