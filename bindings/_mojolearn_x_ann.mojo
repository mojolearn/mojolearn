# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE ANN LANE'S GPU BINDING (lane/algos-ann): IVF-PQ, t-SNE, CAGRA and the
IVF quantization arms. Every entry takes (addrs, params) lists; the orders
are written in providers/python/x_ann_abi_pyapi.mojo's callers below and in
python/mojolearn/_expansion_ann.py. `bindings/_mojolearn_x_ann_host.mojo`
exports the same names with the same contract on the CPU."""
from core.experiment_guards_cluster import CLUSTER_GUARDS_OK
from bindings.device_plan_binding_pack import device_plan_enter_binding, device_plan_exit_binding, device_plan_log_binding
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from checks.vendor import COMPILED_VENDOR
from checks.numerics import GLOBAL_NUMERIC_MODE
from x_ann.mojolearn_x_ann_run import ivf_pq_build_run, ivf_pq_build_resident_run
from providers.python.x_ann_abi_pyapi_unpack import a_int, in_f32, in_i32, out_f32, out_i32, p_int, py_int_list
from x_ann.switches import ANN3_DIRECT_OUT, IVF_PQ_ONE_UPLOAD, IVF_PQ_RESIDENT_FIT
from ivf.impl.neighbors.ivf_flat.ivf_flat_build import IVF_BUILD_FROM_POINTER
from x_ann.ivf_pq_core import IvfPqIndex
from x_ann.stage_timer import AnnStages
from x_ann.cagra_device import cagra_build_device
from x_ann.device_ctx import x_ann_ctx
from providers.python.x_ann_resident_pyapi_pack import x_ann_index_prepare_binding, x_ann_index_release_binding, x_ann_index_search_binding
from providers.python.x_ann_resident_pyapi_pack import x_ann_index_export_codes_binding
from bindings.archive_x_ann_entry import publish_archive_x_ann
from x_ann.resident import x_ann_register_pq_built
from x_ann.ivf_pq_device import ivf_sq_build_device, refine_device, ivf_rabitq_build_device
from x_ann.ops_entry_device import (
    ANN_OPS_DEVICE_API, ann_ops_device_v1, ann_ivf_pq_search_device_run, ann_ivf_sq_search_device_run,
    ann_ivf_rabitq_search_device_run, ann_cagra_search_device_run, ann_filter_topk_device_run,
    ann_tsne_fit_device_run,
)
from core.neighbors_entry_table import NeighborsEntryKey, neighbors_vendor_word
from core.neighbors_entry_table_ann import ann_ops_publish
from bindings.py_balanced import py_setattr
from x_ann.ivf_pq_device import IvfPqDevice
# TOMBSTONE: MOJOLEARN_IVF_REFINE_TEAM (DROPPED-noise) deleted 2026-10-09 by lane/owed-deletions-D3; code recoverable at b639a2bd2.
# Tried: ivf refine with one threadgroup per query and the dataset uploaded from the caller's array (refine_team_kernel, refine_device_team); ivf-refine taxi 1,208.3 -> 1,201.8 ms (-0.5%, noise).
# Restore: git apply experiments/removed/MOJOLEARN_IVF_REFINE_TEAM.patch; record in docs/TOMBSTONES.md.


def ivf_pq_build_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """addrs: x, centers, offsets, list_indices, codebooks, codes.
    params: n, dim, n_lists, kmeans_n_iters, seed, pq_dim, pq_bits, pq_kmeans_n_iters."""
    var n = p_int(params, 0)
    var dim = p_int(params, 1)
    var bst = AnnStages("ivf_pq_binding")
    # lane fg-ivf A3: where the coarse build reads the caller's rows in place
    # (IVF_PQ_ONE_UPLOAD with IVF_BUILD_FROM_POINTER) the n x dim copy is gone
    var x_addr = a_int(addrs, 0)
    var x = List[Float32]()
    comptime if not (IVF_PQ_ONE_UPLOAD and IVF_BUILD_FROM_POINTER):
        x = in_f32(addrs, 0, n * dim)
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
    # p3-neighbors-cluster: the output addresses are Python items, so they are
    # read here with the GIL held; the released block below touches no
    # PythonObject (the host binding's pattern).
    var centers_out = a_int(addrs, 1)
    var offsets_out = a_int(addrs, 2)
    var indices_out = a_int(addrs, 3)
    var codebooks_out = a_int(addrs, 4)
    var codes_out = a_int(addrs, 5)
    with GILReleased(Python()):
        ivf_pq_build_run(
            x, n, dim, n_lists, iters, seed, pq_dim, pq_bits, pq_iters, codes_addr, x_addr,
            bst, centers_out, offsets_out, indices_out, codebooks_out, codes_out,
        )
    return PythonObject(n)


def ivf_pq_build_resident_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """lane fg-ivf A7 (`IVF_PQ_RESIDENT_FIT`, registered only when it holds):
    `ivf_pq_build_binding` with the fitted index LEFT ON THE DEVICE as an
    `x_ann/resident.mojo` handle (returned); the codes are not downloaded
    (`x_ann_index_export_codes` writes them on first read of `codes_`).
    addrs: x, centers, offsets, list_indices, codebooks.
    params: n, dim, n_lists, kmeans_n_iters, seed, pq_dim, pq_bits, pq_kmeans_n_iters."""
    if len(addrs) != 5 or len(params) != 8:
        raise Error("x_ann_ivf_pq_build_resident: need 5 addresses and 8 parameters")
    var n = p_int(params, 0)
    var dim = p_int(params, 1)
    var bst = AnnStages("ivf_pq_binding_resident")
    var x_addr = a_int(addrs, 0)
    var x = List[Float32]()
    comptime if not (IVF_PQ_ONE_UPLOAD and IVF_BUILD_FROM_POINTER):
        x = in_f32(addrs, 0, n * dim)
    bst.host("copy_in")
    var n_lists = p_int(params, 2)
    var iters = p_int(params, 3)
    var seed = p_int(params, 4)
    var pq_dim = p_int(params, 5)
    var pq_bits = p_int(params, 6)
    var pq_iters = p_int(params, 7)
    var ctx = x_ann_ctx()
    var dev = IvfPqDevice(ctx)
    # a placeholder the build below replaces (assigned inside the GIL release)
    var index = IvfPqIndex(
        0, 0, 0, 0, 0, 0, List[Float32](), List[Int32](), List[Int32](), List[Float32](), List[Int32]()
    )
    # p3-neighbors-cluster: output addresses read with the GIL held.
    var centers_out = a_int(addrs, 1)
    var offsets_out = a_int(addrs, 2)
    var indices_out = a_int(addrs, 3)
    var codebooks_out = a_int(addrs, 4)
    with GILReleased(Python()):
        ivf_pq_build_resident_run(
            x, x_addr, n, dim, n_lists, iters, seed, pq_dim, pq_bits, pq_iters, dev,
            index, bst, centers_out, offsets_out, indices_out, codebooks_out,
        )
    _ = x^
    var handle = x_ann_register_pq_built(index^, dev^)
    bst.host("register")
    _ = ctx^
    return PythonObject(handle)


def ivf_pq_search_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """addrs: centers, offsets, list_indices, codebooks, codes, queries, out_d, out_i, out_n, mask
    (int32 per row, 0 removes the row: the sample filter).
    params: n, dim, n_lists, pq_dim, pq_bits, m, k, n_probes[, status] (status: the host
    address of the int32 word that receives the queries' finite status).
    UNPACK and PACK only: the body is x_ann/ops_entry_device.mojo `ann_ivf_pq_search_device_run`,
    which the ANN operations table (core/neighbors_entry_table_ann.mojo) also
    calls. The export remains for python/mojolearn/_expansion_ann.py."""
    var a = py_int_list(addrs)
    var p = py_int_list(params)
    var m = 0
    with GILReleased(Python()):
        m = ann_ivf_pq_search_device_run(a, p)
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
    UNPACK and PACK only: the body is x_ann/ops_entry_device.mojo `ann_tsne_fit_device_run`, which the ANN
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
        status = ann_tsne_fit_device_run(a, p, f)
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
    var x = in_f32(addrs, 0, n * d)
    var g = List[Int32]()
    with GILReleased(Python()):
        g = cagra_build_device(x, n, d, kdeg, deg)
    out_i32(g, addrs, 1)
    return PythonObject(n)


def cagra_search_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """addrs: x, graph, queries, out_d, out_i.
    params: n, d, graph_degree, m, k, itopk_size, search_width, max_iterations, n_seeds[, rs[, status]]
    (rs: 0 the evenly spaced seeds, > 0 random_state's, x_ann/cagra_core.mojo cg_seed_node;
    status: the host address of the int32 word that receives the queries' finite status).
    UNPACK and PACK only: the body is x_ann/ops_entry_device.mojo `ann_cagra_search_device_run`,
    which the ANN operations table (core/neighbors_entry_table_ann.mojo) also
    calls. The export remains for python/mojolearn/_expansion_ann.py."""
    var a = py_int_list(addrs)
    var p = py_int_list(params)
    var m = 0
    with GILReleased(Python()):
        m = ann_cagra_search_device_run(a, p)
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
                            codes_addr, train_rows_per_list=train_rows_per_list)
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
    params: n, dim, n_lists, m, k, n_probes[, status] (the queries' finite status word).
    UNPACK and PACK only: the body is x_ann/ops_entry_device.mojo `ann_ivf_sq_search_device_run`,
    which the ANN operations table (core/neighbors_entry_table_ann.mojo) also
    calls. The export remains for python/mojolearn/_expansion_ann.py."""
    var a = py_int_list(addrs)
    var p = py_int_list(params)
    var m = 0
    with GILReleased(Python()):
        m = ann_ivf_sq_search_device_run(a, p)
    return PythonObject(m)


def refine_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """addrs: dataset, queries, candidates (int32 m x k0, < 0 = padding), out_d, out_i.
    params: n, d, m, k0, k[, root[, status]] (root 1: the euclidean metric's square
    roots of the kept distances, lane apple-fast-py2mojo-cluster; status: the
    host address of the int32 word that receives the dataset's and queries'
    finite status)."""
    var n = p_int(params, 0)
    var d = p_int(params, 1)
    var m = p_int(params, 2)
    var k0 = p_int(params, 3)
    var k = p_int(params, 4)
    var root = len(params) > 5 and p_int(params, 5) != 0
    var status = p_int(params, 6) if len(params) > 6 else 0
    if n <= 0 or d <= 0 or m <= 0 or k0 <= 0 or k <= 0 or k > k0:
        raise Error("refine: need positive shapes and 1 <= k <= n_candidates")
    var x = in_f32(addrs, 0, n * d)
    var q = in_f32(addrs, 1, m * d)
    var cand = in_i32(addrs, 2, m * k0)
    var od = List[Float32]()
    var oi = List[Int32]()
    with GILReleased(Python()):
        refine_device(x, n, d, q, m, cand, k0, k, od, oi, root, status)
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
    params: n, dim, n_lists, seed, m, k, n_probes[, status] (the queries' finite status word).
    UNPACK and PACK only: the body is x_ann/ops_entry_device.mojo `ann_ivf_rabitq_search_device_run`,
    which the ANN operations table (core/neighbors_entry_table_ann.mojo) also
    calls. The export remains for python/mojolearn/_expansion_ann.py."""
    var a = py_int_list(addrs)
    var p = py_int_list(params)
    var m = 0
    with GILReleased(Python()):
        m = ann_ivf_rabitq_search_device_run(a, p)
    return PythonObject(m)


def filter_topk_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """CAGRA's filtered compaction (x_ann/filter_topk.mojo, lane
    apple-fast-py2mojo-cluster). addrs: itopk distances (m x L), itopk ids
    (m x L int32), keep (n int32), out_d, out_i (m x k). params: m, n, L, k.
    UNPACK and PACK only: the body is x_ann/ops_entry_device.mojo `ann_filter_topk_device_run`,
    which the ANN operations table (core/neighbors_entry_table_ann.mojo) also
    calls. The export remains for python/mojolearn/_expansion_ann.py."""
    var a = py_int_list(addrs)
    var p = py_int_list(params)
    var m = 0
    with GILReleased(Python()):
        m = ann_filter_topk_device_run(a, p)
    return PythonObject(m)


# TOMBSTONE: x_ann_py2mojo (dead native entry: a constant A/B probe nothing reads) deleted 2026-10-26 by f9b52032a; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_x_ann.patch


def numeric_mode_binding() raises -> PythonObject:
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))


def vendor_binding() raises -> PythonObject:
    return PythonObject(String(COMPILED_VENDOR))


def _publish_ann_ops(module: PythonObject) raises:
    # The device column of the ANN operations table
    # (core/neighbors_entry_table_ann.mojo): the ANN search owner
    # (bindings/neighbors_ann_query_native.mojo) and the TSNE owner
    # (bindings/mani_tsne_native.mojo) call this entry with plain values and
    # read the key from the module attribute `x_ann_native_key`. The Python
    # exports remain for python/mojolearn/_expansion_ann.py. A build with no
    # GPU vendor ("none") publishes nothing and records no key.
    var word = neighbors_vendor_word(COMPILED_VENDOR)
    if word < 0:
        return
    var api = ANN_OPS_DEVICE_API.get_or_create_ptr()
    ann_ops_publish(NeighborsEntryKey(Int(GLOBAL_NUMERIC_MODE), word), Int(api[].entries.unsafe_ptr()),
                    ann_ops_device_v1)
    py_setattr(module, "x_ann_native_key", Python.tuple(PythonObject(Int(GLOBAL_NUMERIC_MODE)), PythonObject(word)))


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_x_ann.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).
def build__mojolearn_x_ann() -> PythonObject:
    comptime assert CLUSTER_GUARDS_OK, "invalid integrated experiment configuration (cluster guards)"
    try:
        var m = PythonModuleBuilder("_mojolearn_x_ann")
        m.def_function[ivf_pq_build_binding]("x_ann_ivf_pq_build")
        m.def_function[device_plan_enter_binding]("device_plan_enter")
        m.def_function[device_plan_exit_binding]("device_plan_exit")
        m.def_function[device_plan_log_binding]("device_plan_log")
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
        m.def_function[x_ann_index_prepare_binding]("x_ann_index_prepare")
        m.def_function[x_ann_index_search_binding]("x_ann_index_search")
        m.def_function[x_ann_index_release_binding]("x_ann_index_release")
        comptime if IVF_PQ_RESIDENT_FIT:
            # lane fg-ivf A7: the resident IVF-PQ fit and its codes export
            # (absent under -D MOJOLEARN_IVF_PQ_RESIDENT_FIT_OFF: fit takes
            # x_ann_ivf_pq_build and the first search prepares the handle)
            m.def_function[ivf_pq_build_resident_binding]("x_ann_ivf_pq_build_resident")
            m.def_function[x_ann_index_export_codes_binding]("x_ann_index_export_codes")
        m.def_function[numeric_mode_binding]("x_ann_numeric_mode")
        m.def_function[vendor_binding]("x_ann_vendor")
        var published_module = m.finalize()
        _publish_ann_ops(published_module)
        comptime if IVF_PQ_RESIDENT_FIT:
            # The archive export table: the archive writer saves a resident
            # IVF-PQ index's codes through this entry natively.
            publish_archive_x_ann()
        return published_module
    except e:
        abort(String("failed to create _mojolearn_x_ann: ", e))
