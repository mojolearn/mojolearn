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
from providers.python.x_ann_abi_pyapi_unpack import a_int, p_int, ptr_f32, ptr_i32, py_int_list
from x_ann.host.ivf_pq_host import X_ANN_HOST_SABOTAGE, refine_host
from x_ann.ops_entry_host_fit import (
    ann_ivf_pq_build_host_run, ann_ivf_sq_build_host_run, ann_ivf_rabitq_build_host_run,
    ann_cagra_build_host_run,
)
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
    params: n, dim, n_lists, kmeans_n_iters, seed, pq_dim, pq_bits, pq_kmeans_n_iters.
    UNPACK and PACK only (lane dw-neighbors, 2026-11-21): the body is
    x_ann/ops_entry_host_fit.mojo `ann_ivf_pq_build_host_run`, which the ANN operations table
    (core/neighbors_entry_table_ann.mojo) also calls."""
    # Admit Python addresses while holding the GIL; released work uses native values.
    var a = py_int_list(addrs)
    var p = py_int_list(params)
    var n = 0
    with GILReleased(Python()):
        n = ann_ivf_pq_build_host_run(a, p)
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


def cagra_build_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """addrs: x, graph_out. params: n, d, intermediate_graph_degree, graph_degree.
    UNPACK and PACK only (lane dw-neighbors, 2026-11-21): the body is
    x_ann/ops_entry_host_fit.mojo `ann_cagra_build_host_run`, which the ANN operations table
    (core/neighbors_entry_table_ann.mojo) also calls."""
    # Admit Python addresses while holding the GIL; released work uses native values.
    var a = py_int_list(addrs)
    var p = py_int_list(params)
    var n = 0
    with GILReleased(Python()):
        n = ann_cagra_build_host_run(a, p)
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
    IVFSQIndex max_train_points_per_cluster; 0 or absent keeps the default training rows).
    UNPACK and PACK only (lane dw-neighbors, 2026-11-21): the body is
    x_ann/ops_entry_host_fit.mojo `ann_ivf_sq_build_host_run`, which the ANN operations table
    (core/neighbors_entry_table_ann.mojo) also calls."""
    # Admit Python addresses while holding the GIL; released work uses native values.
    var a = py_int_list(addrs)
    var p = py_int_list(params)
    var n = 0
    with GILReleased(Python()):
        n = ann_ivf_sq_build_host_run(a, p)
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
    params: n, dim, n_lists, kmeans_n_iters, seed.
    UNPACK and PACK only (lane dw-neighbors, 2026-11-21): the body is
    x_ann/ops_entry_host_fit.mojo `ann_ivf_rabitq_build_host_run`, which the ANN operations table
    (core/neighbors_entry_table_ann.mojo) also calls."""
    # Admit Python addresses while holding the GIL; released work uses native values.
    var a = py_int_list(addrs)
    var p = py_int_list(params)
    var n = 0
    with GILReleased(Python()):
        n = ann_ivf_rabitq_build_host_run(a, p)
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
        m.def_function[cagra_build_binding]("x_ann_cagra_build")
        m.def_function[cagra_search_binding]("x_ann_cagra_search")
        m.def_function[ivf_sq_build_binding]("x_ann_ivf_sq_build")
        m.def_function[ivf_sq_search_binding]("x_ann_ivf_sq_search")
        m.def_function[refine_binding]("x_ann_refine")
        # TOMBSTONE: x_ann_py2mojo (dead native entry: a constant A/B probe nothing reads) deleted 2026-10-26 by f9b52032a; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_x_ann.patch
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
