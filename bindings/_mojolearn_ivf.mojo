# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPython boundary for the IVF-FLAT lane (`mojolearn.IVFIndex`), exposed 2026-09-14.

`ivf/checks/ivf_check.mojo` (the layout sabotage, the large-k and
large-probe checks) reads ALL OK at IDENTICAL on the Apple M4, an NVIDIA
H100 and an AMD MI300X with one card, 1e7c1702, on all three
(bench/results/ivf_embed_km_legs_2026-09-14/README.md). The binding is in
`python/mojolearn/_backend.py`'s `_MODULES` and `_build_script` and in the
four packaging lists `packaging/check_ext_lists.py` holds to it;
`python/mojolearn/_ivf_impl.py` is the Python half and the identity_break
lane is `ivf`.

The build and the search are two entries (below). `n_probes` has no default at this boundary (policy 1) and
`n_probes > n_lists` RAISES rather than clamps (policy 2), both on the
Mojo host.

THE ABI IS THE GP'S: two length-checked lists, orders written out below
and mirrored in `_ivf_impl.py`.
"""

from core.experiment_guards_cluster import CLUSTER_GUARDS_OK
from bindings.device_plan_binding import device_plan_enter_binding, device_plan_exit_binding, device_plan_log_binding
from std.os import abort
from std.sys.compile import is_defined
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.parallel_ivf_entry_device import (
    ivf_finalize_distances_device_run, ivf_flat_index_export_device_run, publish_parallel_ivf_device,
)
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from bindings.ivf_index_arrays import (
    ivf_arrays_extent,
    ivf_build_extents,
    ivf_extend_count,
    ivf_read_index_arrays,
    ivf_read_resident_filter,
    ivf_read_search_filter,
    ivf_search_extents,
)
from x_ann.stage_timer import AnnStages
from ivf.resident import ivf_resident_check, ivf_resident_n_rows, ivf_resident_release
from ivf.ivf_index_arrays_run import ivf_arrays_check_cells
# The RUN bodies of these doors, on the binding's one process-lifetime
# DeviceContext (lane rs-decomposition, 2026-11-12).
from bindings.ivf_device_run import (
    ivf_device_extend_run,
    ivf_device_prepare_run,
    ivf_device_resident_search_run,
    ivf_device_search_run,
    ivf_flat_build_resident_run,
    ivf_flat_build_run,
)


def ivf_numeric_mode_binding() raises -> PythonObject:
    """THE BUILD'S TIER as the `NUMERIC_*` code: 0 FAST, 1 IDENTICAL."""
    return PythonObject(GLOBAL_NUMERIC_MODE)


def ivf_vendor_binding() raises -> PythonObject:
    """'metal', 'cuda', 'hip' or 'none', from `checks/vendor.mojo`."""
    return PythonObject(String(COMPILED_VENDOR))


# TOMBSTONE: ivf_flat_build_and_search (dead native entry: the retired one-call door; IVFIndex fits and searches through ivf_flat_build / ivf_flat_search) deleted 2026-10-26 by e760c256b; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_ivf.patch


# ===========================================================================
# BUILD AND SEARCH AS TWO CALLS (lane/inference-embedding-ivf-cholesky,
# 2026-09-15). The index crosses to Python as five arrays
# (`bindings/ivf_index_arrays.mojo`), so `IVFIndex.fit` builds once,
# `IVFIndex.save` writes the arrays and a CPU loads them. The build is
# `ivf_flat_build_host` and the search `ivf_flat_search_host`, the two halves
# `ivf_flat_build_and_search_host` calls, over host lists the upload copies,
# so the split changes no bit. Each call writes its own identity card (policy
# 3's caveat).
# ===========================================================================


def ivf_flat_build_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`ivf_flat::build`, the index written to the caller's five buffers.
    Returns 0. See `bindings/ivf_index_arrays.mojo` for the lists."""
    var ext = ivf_build_extents(addrs, params, String("ivf_flat_build"))
    ivf_flat_build_run(
        Int(py=addrs[0]), Int(py=addrs[1]), Int(py=addrs[2]), Int(py=addrs[3]), Int(py=addrs[4]), Int(py=addrs[5]),
        ext[0], ext[1], ext[2], ext[3], ext[4], ext[5],
    )
    return PythonObject(0)


def _ivf_search_arrays(
    addrs: PythonObject, params: PythonObject, partial_storage: Bool
) raises -> PythonObject:
    """`ivf_flat::search` over a built index handed back as five arrays and
    admitted by `ivf_validate_index_arrays`. Returns 0."""
    var arrays = ivf_read_index_arrays(
        addrs, params, String("ivf_flat_search"), partial_storage=partial_storage, allow_filter=not partial_storage
    )
    var keep = ivf_read_search_filter(addrs, arrays.n_rows)
    var ext = ivf_search_extents(params)
    ivf_device_search_run(
        arrays^, keep, Int(py=addrs[5]), ext[0], ext[1], ext[2], partial_storage,
        Int(py=addrs[6]), Int(py=addrs[7]), Int(py=addrs[8]),
    )
    return PythonObject(0)


def ivf_flat_search_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    return _ivf_search_arrays(addrs, params, False)


def ivf_flat_partial_search_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """Disjoint storage with global centers/probes; squared output, valid count=min(k,candidates)."""
    return _ivf_search_arrays(addrs, params, True)


def ivf_finalize_distances_binding(address: PythonObject, count: PythonObject, metric: PythonObject) raises -> PythonObject:
    """`postprocess_distances` over the merged distances, on the device
    (bindings/parallel_ivf_entry_device.mojo `ivf_finalize_distances_device_run`,
    which the distributed IVF owner reaches natively)."""
    ivf_finalize_distances_device_run(Int(py=address), Int(py=count), Int(py=metric))
    return PythonObject(0)


# `ivf_merge_shards` and `ivf_shard_plan` have no Python door any more (lane
# calltable-parallel-archive, 2026-11-11): their one caller, the distributed
# IVF owner, reaches the device bodies through this image's parallel IVF
# table entry (bindings/parallel_ivf_entry_device.mojo).


# ===========================================================================
# THE RESIDENT INDEX (lane/py-dn-ann, 2026-09-28; ivf/resident.mojo, the
# closure of DEVIATION 1804). Three doors, the contract in
# bindings/ivf_index_arrays.mojo: prepare admits and uploads the index once,
# search uploads only its queries, release drops it.
# ===========================================================================


def ivf_flat_index_prepare_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    var partial = Int(py=params[4]) != 0
    var bst = AnnStages("ivf_flat_prepare")
    var arrays = ivf_read_index_arrays(
        addrs, params, String("ivf_flat_search"), 5, 5, partial_storage=partial
    )
    bst.host("read_admit")
    return PythonObject(ivf_device_prepare_run(arrays^, partial, bst))


# ===========================================================================
# THE RESIDENT BUILD (lane gap-ivf, 2026-10-08, docs/plans/gaps-2026-10-08.md
# section 6 item 1). `IVFIndex.fit` builds and KEEPS the index on the device:
# the list vectors and the device CSR never cross to numpy inside the fit, and
# the first search does not re-admit and re-upload them. The four small arrays
# (centres, their norms, offsets, carried ids) are written for the Python
# object; `list_data_` is made on first read by `ivf_flat_index_export`
# (save, pickle, extend, clone, shard split: never inside fit or search).
#
# `ivf_flat_build_resident(addrs, params)` -> handle:
#     addrs   0 x                n * dim float32, read
#             1 centers_out      n_lists * dim float32, WRITTEN
#             2 center_norms_out n_lists float32, WRITTEN (squared)
#             3 offsets_out      n_lists + 1 int32, WRITTEN
#             4 indices_out      n int32, WRITTEN (original row ids)
#     params  0 n, 1 dim, 2 n_lists, 3 kmeans_n_iters, 4 metric, 5 seed
# `ivf_flat_index_export(handle, addr, count)`: list_data (n * dim float32)
#     WRITTEN at addr; count must be n * dim.
# The bits are `ivf_flat_build`'s: the same build statements over the same
# rows, the same layout kernels; only where the vectors live differs.
# ===========================================================================


def ivf_flat_build_resident_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    if len(addrs) != 5:
        raise Error(
            "ivf_flat_build_resident: addrs must contain 5 addresses (x, centers_out,"
            " center_norms_out, offsets_out, indices_out), got " + String(len(addrs))
        )
    if len(params) != 6:
        raise Error(
            "ivf_flat_build_resident: params must contain 6 values (n, dim, n_lists,"
            " kmeans_n_iters, metric, seed), got " + String(len(params))
        )
    var n = ivf_arrays_extent(params[0], String("n"))
    var dim = ivf_arrays_extent(params[1], String("dim"))
    var n_lists = ivf_arrays_extent(params[2], String("n_lists"))
    ivf_arrays_check_cells(n, dim, String("n * dim"))
    ivf_arrays_check_cells(n_lists, dim, String("n_lists * dim"))
    var iters = Int(py=params[3])
    var metric = Int(py=params[4])
    var seed = UInt64(Int(py=params[5]))
    var handle = ivf_flat_build_resident_run(
        Int(py=addrs[0]), Int(py=addrs[1]), Int(py=addrs[2]), Int(py=addrs[3]), Int(py=addrs[4]),
        n, dim, n_lists, iters, metric, seed,
    )
    return PythonObject(handle)


def ivf_flat_index_export_binding(handle: PythonObject, address: PythonObject, count: PythonObject) raises -> PythonObject:
    # The archive writer reaches the same body through the archive export
    # table (bindings/parallel_ivf_entry_device.mojo); this door stays for
    # the Python and neighbor-query callers.
    ivf_flat_index_export_device_run(Int(py=handle), Int(py=address), Int(py=count))
    return PythonObject(0)


def ivf_flat_index_search_binding(
    handle: PythonObject, addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    var h = Int(py=handle)
    if len(params) != 8:
        raise Error("ivf_flat_index_search: params must contain 8 values (n, dim, n_lists, metric, m, k, n_probes, partial)")
    var n = Int(py=params[0])
    var dim = Int(py=params[1])
    ivf_resident_check(h, n, dim, Int(py=params[2]), Int(py=params[3]), Int(py=params[7]) != 0)
    var ext = ivf_search_extents(params)
    var queries_addr = Int(py=addrs[0])
    var keep = ivf_read_resident_filter(addrs, ivf_resident_n_rows(h))
    ivf_device_resident_search_run(
        h, queries_addr, dim, ext[0], ext[1], ext[2], keep, Int(py=addrs[1]), Int(py=addrs[2]), Int(py=addrs[3])
    )
    return PythonObject(0)


def ivf_flat_index_release_binding(handle: PythonObject) raises -> PythonObject:
    ivf_resident_release(Int(py=handle))
    return PythonObject(0)


def ivf_flat_extend_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`ivf_flat::extend` over a built index handed back as five arrays
    (stage 2 of lane/inference-embedding-ivf-cholesky, 2026-09-15). Returns 0.
    See `bindings/ivf_index_arrays.mojo` for the lists."""
    var arrays = ivf_read_index_arrays(addrs, params, String("ivf_flat_extend"), 10, 5)
    var n_new = ivf_extend_count(params, arrays.n_rows, arrays.dim)
    ivf_device_extend_run(
        arrays^, Int(py=addrs[5]), n_new, Int(py=addrs[6]), Int(py=addrs[7]), Int(py=addrs[8]), Int(py=addrs[9])
    )
    return PythonObject(0)


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_ivf.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).
def build__mojolearn_ivf() -> PythonObject:
    comptime assert CLUSTER_GUARDS_OK, "invalid integrated experiment configuration (cluster guards)"
    try:
        var m = PythonModuleBuilder("_mojolearn_ivf")
        m.def_function[ivf_vendor_binding]("ivf_vendor")
        m.def_function[device_plan_enter_binding]("device_plan_enter")
        m.def_function[device_plan_exit_binding]("device_plan_exit")
        m.def_function[device_plan_log_binding]("device_plan_log")
        m.def_function[ivf_numeric_mode_binding]("ivf_numeric_mode")
        # TOMBSTONE: MOJOLEARN_IVF_FAST_BALANCED_TASKS (never measured, deleted) deleted 2026-11-09 by 0c3399062b; code
        # recoverable at f59d913b0b (with its MOJOLEARN_IVF_FAST_BALANCED_AUDIT export ivf_fast_balanced_hits).
        # Restore: git apply experiments/removed/MOJOLEARN_IVF_FAST_BALANCED_TASKS.patch
        # TOMBSTONE: ivf_flat_build_and_search (dead native entry: the retired one-call door; IVFIndex fits and searches through ivf_flat_build / ivf_flat_search) deleted 2026-10-26 by e760c256b; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_ivf.patch
        m.def_function[ivf_flat_build_binding]("ivf_flat_build")
        m.def_function[ivf_flat_search_binding]("ivf_flat_search")
        m.def_function[ivf_flat_partial_search_binding]("ivf_flat_partial_search")
        m.def_function[ivf_finalize_distances_binding]("ivf_finalize_distances")
        m.def_function[ivf_flat_extend_binding]("ivf_flat_extend")
        m.def_function[ivf_flat_index_prepare_binding]("ivf_flat_index_prepare")
        m.def_function[ivf_flat_index_search_binding]("ivf_flat_index_search")
        m.def_function[ivf_flat_index_release_binding]("ivf_flat_index_release")
        m.def_function[ivf_flat_build_resident_binding]("ivf_flat_build_resident")
        m.def_function[ivf_flat_index_export_binding]("ivf_flat_index_export")
        var published_module = m.finalize()
        # The parallel IVF and archive export tables: the distributed IVF
        # owner and the archive writer call these entries natively.
        publish_parallel_ivf_device()
        return published_module
    except e:
        abort(String("failed to create _mojolearn_ivf: ", e))

