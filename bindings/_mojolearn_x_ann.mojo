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
from providers.python.x_ann_abi_pyapi_unpack import a_int, in_f32, p_int, py_int_list
from x_ann.switches import IVF_PQ_ONE_UPLOAD, IVF_PQ_RESIDENT_FIT
from ivf.impl.neighbors.ivf_flat.ivf_flat_build import IVF_BUILD_FROM_POINTER
from x_ann.ivf_pq_core import IvfPqIndex
from x_ann.stage_timer import AnnStages
from x_ann.cagra_device import cagra_build_device_ptr
from x_ann.device_ctx import x_ann_ctx
from providers.python.x_ann_resident_pyapi_pack import x_ann_index_release_binding
from providers.python.x_ann_resident_pyapi_pack import x_ann_index_export_codes_binding
from bindings.archive_x_ann_entry import publish_archive_x_ann
from x_ann.resident import x_ann_register_pq_built
from x_ann.ivf_pq_device import ivf_sq_build_device_ptr, refine_device_ptr, ivf_rabitq_build_device_ptr
from core.hostptr import f32_ptr, i32_ptr
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
    # The codes go from the device straight into the caller's array (lane
    # fx-ann-fit-nocopy, 2026-11-20: the default; before, a result list the
    # size of the codes and a copy out of it). The opt-in `ANN3_DIRECT_OUT`
    # (lane ann-apple3) now only chooses the copy inside the build
    # (x_ann/ivf_pq_device.mojo): one raw copy into the pageable array
    # in place of the pooled pinned stages.
    # TOMBSTONE: MOJOLEARN_ANN3_DIRECT_OUT (unmeasured opt-in arm: codes written by the device into the caller's array) deleted 2026-11-15 by lane/expdel2-01-base; code recoverable at 418bb3c6c3.
    # Restore: git apply experiments/removed/MOJOLEARN_ANN3_DIRECT_OUT.expdel2-01-base.patch
    # RESTORED 2026-11-15 by lane/restore-apple-fast-classical: Apple fast is paused, not abandoned (Andrew); opt-in, default off, Apple fast only.
    var codes_addr = a_int(addrs, 5)
    if codes_addr == 0:
        raise Error("mojolearn: null int32 buffer address")
    # p3-neighbors-cluster: the output addresses are Python items, so they are
    # read here with the GIL held; the released block below touches no
    # PythonObject (the host binding's pattern).
    var centers_out = a_int(addrs, 1)
    var offsets_out = a_int(addrs, 2)
    var indices_out = a_int(addrs, 3)
    var codebooks_out = a_int(addrs, 4)
    with GILReleased(Python()):
        ivf_pq_build_run(
            x, n, dim, n_lists, iters, seed, pq_dim, pq_bits, pq_iters, codes_addr, x_addr,
            bst, centers_out, offsets_out, indices_out, codebooks_out,
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


def cagra_build_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """addrs: x, graph_out. params: n, d, intermediate_graph_degree, graph_degree."""
    var n = p_int(params, 0)
    var d = p_int(params, 1)
    var kdeg = p_int(params, 2)
    var deg = p_int(params, 3)
    if n < 2 or d <= 0 or kdeg < 1 or kdeg > n - 1 or deg < 1 or deg > kdeg:
        raise Error("CAGRA: need 1 <= graph_degree <= intermediate_graph_degree <= n - 1")
    # The caller's arrays themselves (lane fx-ann-fit-nocopy, 2026-11-20):
    # the rows are uploaded from where they are and the graph is downloaded
    # where the caller wants it. Gone: a list copy of the n x d rows and a
    # graph list with a copy out of it. The door keeps both arrays alive
    # through this call.
    var xp = f32_ptr(a_int(addrs, 0))
    var gp = i32_ptr(a_int(addrs, 1))
    with GILReleased(Python()):
        cagra_build_device_ptr(xp, n, d, kdeg, deg, gp)
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
    # The caller's arrays themselves (lane fx-ann-fit-nocopy, 2026-11-20).
    # Where the coarse build reads the caller's rows in place
    # (IVF_PQ_ONE_UPLOAD with IVF_BUILD_FROM_POINTER, as IVF-PQ's fit) the
    # n x dim list copy is gone; the six results are written where the
    # caller wants them (before: six lists and a copy out of each, the
    # n x dim codes among them). The door keeps all seven arrays alive
    # through this call.
    # TOMBSTONE: MOJOLEARN_ANN3_DIRECT_OUT (unmeasured opt-in arm: codes written by the device into the caller's array) deleted 2026-11-15 by lane/expdel2-01-base; code recoverable at 418bb3c6c3.
    # Restore: git apply experiments/removed/MOJOLEARN_ANN3_DIRECT_OUT.expdel2-01-base.patch
    # RESTORED 2026-11-15 by lane/restore-apple-fast-classical: Apple fast is paused, not abandoned (Andrew); opt-in, default off, Apple fast only.
    # (2026-11-20, lane fx-ann-fit-nocopy: the arm's read moved into x_ann/ivf_pq_device.mojo `ivf_sq_build_device_ptr`.)
    var x_addr = a_int(addrs, 0)
    if x_addr == 0:
        raise Error("mojolearn: null float32 buffer address")
    var x = List[Float32]()
    comptime if not (IVF_PQ_ONE_UPLOAD and IVF_BUILD_FROM_POINTER):
        x = in_f32(addrs, 0, n * dim)
    bst.host("copy_in")
    var centers_p = f32_ptr(a_int(addrs, 1))
    var offsets_p = i32_ptr(a_int(addrs, 2))
    var indices_p = i32_ptr(a_int(addrs, 3))
    var vmin_p = f32_ptr(a_int(addrs, 4))
    var delta_p = f32_ptr(a_int(addrs, 5))
    var codes_p = i32_ptr(a_int(addrs, 6))
    with GILReleased(Python()):
        ivf_sq_build_device_ptr(x, x_addr, n, dim, n_lists, iters, seed, centers_p, offsets_p, indices_p,
                                vmin_p, delta_p, codes_p, train_rows_per_list)
    bst.host("build")
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
    # The caller's arrays themselves (lane fx-board-modelsel-metrics,
    # 2026-11-19): uploaded from and downloaded to where they are. Gone: a
    # list copy of the dataset, the queries and the candidates per call, and
    # two result lists with a copy out of each. The door keeps all five
    # arrays alive through this call.
    var xp = f32_ptr(a_int(addrs, 0))
    var qp = f32_ptr(a_int(addrs, 1))
    var candp = i32_ptr(a_int(addrs, 2))
    var odp = f32_ptr(a_int(addrs, 3))
    var oip = i32_ptr(a_int(addrs, 4))
    with GILReleased(Python()):
        refine_device_ptr(xp, n, d, qp, m, candp, k0, k, odp, oip, root, status)
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
    # The caller's arrays themselves (lane fx-ann-fit-nocopy, 2026-11-20;
    # `ivf_sq_build_binding`'s form): no list copy of the rows where the
    # coarse build reads them in place, and the six results written where
    # the caller wants them.
    var x_addr = a_int(addrs, 0)
    if x_addr == 0:
        raise Error("mojolearn: null float32 buffer address")
    var x = List[Float32]()
    comptime if not (IVF_PQ_ONE_UPLOAD and IVF_BUILD_FROM_POINTER):
        x = in_f32(addrs, 0, n * dim)
    bst.host("copy_in")
    var centers_p = f32_ptr(a_int(addrs, 1))
    var offsets_p = i32_ptr(a_int(addrs, 2))
    var indices_p = i32_ptr(a_int(addrs, 3))
    var codes_p = i32_ptr(a_int(addrs, 4))
    var norms_p = f32_ptr(a_int(addrs, 5))
    var ips_p = f32_ptr(a_int(addrs, 6))
    with GILReleased(Python()):
        ivf_rabitq_build_device_ptr(x, x_addr, n, dim, n_lists, iters, seed, centers_p, offsets_p, indices_p,
                                    codes_p, norms_p, ips_p)
    bst.host("build")
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
        m.def_function[cagra_build_binding]("x_ann_cagra_build")
        m.def_function[cagra_search_binding]("x_ann_cagra_search")
        m.def_function[ivf_sq_build_binding]("x_ann_ivf_sq_build")
        m.def_function[ivf_sq_search_binding]("x_ann_ivf_sq_search")
        m.def_function[refine_binding]("x_ann_refine")
        # TOMBSTONE: x_ann_py2mojo (dead native entry: a constant A/B probe nothing reads) deleted 2026-10-26 by f9b52032a; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_x_ann.patch
        m.def_function[ivf_rabitq_build_binding]("x_ann_ivf_rabitq_build")
        m.def_function[ivf_rabitq_search_binding]("x_ann_ivf_rabitq_search")
        m.def_function[x_ann_index_release_binding]("x_ann_index_release")
        comptime if IVF_PQ_RESIDENT_FIT:
            # lane fg-ivf A7: the resident IVF-PQ fit and its codes export
            # (absent under FAST: fit takes x_ann_ivf_pq_build and the first
            # search prepares the handle)
            # TOMBSTONE: MOJOLEARN_IVF_PQ_RESIDENT_FIT_OFF (old path; one-vendor promotion, grid fg-ivf.json A7: the other vendor's A/B is owed) deleted 2026-11-15 by lane/expdel2-01-base; code recoverable at 699d5e93a1.
            # Restore: git apply experiments/removed/MOJOLEARN_IVF_PQ_RESIDENT_FIT_OFF.expdel2-01-base.patch
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
