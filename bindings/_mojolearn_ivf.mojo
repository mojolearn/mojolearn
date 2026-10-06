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

ONE ENTRY, ONE CARD. `ivf_flat_build_and_search_host` is the entry every
gate uses (the estimator's policy 3): a build and a search under one
identity card. The index itself does not cross, so there is no model
column; `n_probes` has no default at this boundary (policy 1) and
`n_probes > n_lists` RAISES rather than clamps (policy 2), both on the
Mojo host.

THE ABI IS THE GP'S: two length-checked lists, orders written out below
and mirrored in `_ivf_impl.py`.
"""

from std.os import abort
from std.sys.compile import is_defined
from std.memory import memcpy
from bindings.hostptr import f32_ptr, i32_ptr, copy_f32, read_f32
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from max.gpu.host import DeviceContext
from core.neural_context import process_ctx
from checks.numerics import GLOBAL_NUMERIC_MODE as _DEVCTX_MODE, NUMERIC_IDENTICAL as _DEVCTX_IDENTICAL

#: This binding's ONE process-lifetime DeviceContext (core/neural_context.mojo,
#: lane/devctx-lifetime): a context per call exhausts Metal command queues.
comptime _DEVCTX_SLOT = "MojoIvfContextIdentical" if _DEVCTX_MODE == _DEVCTX_IDENTICAL else "MojoIvfContextFast"


from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from bindings.ivf_index_arrays import (
    ivf_build_extents,
    ivf_extend_count,
    ivf_write_extended_arrays,
    ivf_read_index_arrays,
    ivf_read_search_filter,
    ivf_search_extents,
    ivf_write_index_arrays,
    ivf_write_search_result,
)
from ivf.estimator import (
    ivf_flat_build_and_search_host,
    ivf_flat_build_host,
    ivf_flat_extend_host,
    ivf_flat_search_host,
)
from ivf.impl.neighbors.ivf_flat.ivf_flat_index import IvfFlatIndex
from ivf.impl.neighbors.ivf_flat.ivf_flat_search import ivf_fast_balanced_hits
from x_ann.stage_timer import AnnStages
from x_ann.switches import ANN3_PREPARE
from ivf.resident import (
    ivf_resident_check,
    ivf_resident_n_rows,
    ivf_resident_prepare,
    ivf_resident_release,
    ivf_resident_search,
)
from bindings.ivf_index_arrays import ivf_read_resident_filter, ivf_write_resident_result
from core.shard_merge_device import device_ivf_merge_shards, device_ivf_shard_plan, device_root_f32


def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    return f32_ptr(addr)


def _i32_ptr(addr: Int) raises -> MutPointer[Int32, MutUntrackedOrigin]:
    return i32_ptr(addr)


def ivf_numeric_mode_binding() raises -> PythonObject:
    """THE BUILD'S TIER as the `NUMERIC_*` code: 0 FAST, 1 IDENTICAL, 2
    DETERMINISTIC."""
    return PythonObject(GLOBAL_NUMERIC_MODE)


def ivf_vendor_binding() raises -> PythonObject:
    """'metal', 'cuda', 'hip' or 'none', from `checks/vendor.mojo`."""
    return PythonObject(String(COMPILED_VENDOR))


def _ivf_run(
    x: List[Float32],
    n: Int,
    dim: Int,
    n_lists: Int,
    queries: List[Float32],
    m: Int,
    k: Int,
    n_probes: Int,
    kmeans_n_iters: Int,
    metric: Int,
    seed: UInt64,
    dp: MutPointer[Float32, MutUntrackedOrigin],
    ip: MutPointer[Int32, MutUntrackedOrigin],
    cp: MutPointer[Int32, MutUntrackedOrigin],
) raises:
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var r = ivf_flat_build_and_search_host(
        ctx,
        x,
        n,
        dim,
        n_lists,
        queries,
        m,
        k,
        n_probes,
        kmeans_n_iters,
        metric,
        seed,
    )
    ctx.synchronize()
    copy_f32(r.distances.unsafe_ptr(), dp, m * k)
    # ORIGINAL row ids, UInt32 on the Mojo side and below 2**31 by
    # construction (n <= 46340 on every lane in this tree), so the int32
    # bytes are the same bytes.
    # cpu2-l6-bindings: one byte copy instead of a per-element conversion
    memcpy(
        dest=ip.bitcast[UInt32](), src=r.indices.unsafe_ptr(), count=m * k
    )
    comptime if is_defined["MOJOLEARN_IVF_BINDING_SABOTAGE"]():
        # NEVER SHIPPED. Swaps the first two neighbor ids of query 0, the
        # tie-class corruption a wrong (distance, id) order would produce, so
        # the identity_break `ivf` lane is SEEN TO FAIL on its idx part.
        if m * k >= 2:
            var a = ip.unsafe_load(0)
            ip.unsafe_store(0, ip.unsafe_load(1))
            ip.unsafe_store(1, a)
    for i in range(m):
        cp.unsafe_store(i, r.n_candidates[i])
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^


def ivf_flat_build_and_search_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`ivf_flat::build` then `ivf_flat::search` under ONE identity card
    (`ivf_flat_build_and_search_host`, policy 3). Returns 0.

    `addrs`, in this exact order:

        0  x               n * dim float32, row-major, read
        1  queries         m * dim float32, row-major, read
        2  dist_out        m * k float32, WRITTEN
        3  idx_out         m * k int32, WRITTEN (original row ids)
        4  cand_out        m int32, WRITTEN (candidates examined per query)

    `params`, in this exact order:

        0  n
        1  dim
        2  m               n_queries
        3  k
        4  n_lists
        5  n_probes        REQUIRED, no default (policy 1); > n_lists RAISES
        6  kmeans_n_iters  cuVS's 20
        7  metric          0 L2Expanded (squared), 1 L2SqrtExpanded
        8  seed
    """
    if len(addrs) != 5:
        raise Error(
            "ivf_flat_build_and_search: addrs must contain 5 addresses (x,"
            " queries, dist_out, idx_out, cand_out), got "
            + String(len(addrs))
        )
    if len(params) != 9:
        raise Error(
            "ivf_flat_build_and_search: params must contain 9 values (n, dim,"
            " m, k, n_lists, n_probes, kmeans_n_iters, metric, seed), got "
            + String(len(params))
        )
    var n = Int(py=params[0])
    var dim = Int(py=params[1])
    var m = Int(py=params[2])
    var k = Int(py=params[3])
    var n_lists = Int(py=params[4])
    var n_probes = Int(py=params[5])
    var kmeans_n_iters = Int(py=params[6])
    var metric = Int(py=params[7])
    var seed = UInt64(Int(py=params[8]))
    var x = read_f32(Int(py=addrs[0]), max(0, n * dim))
    var queries = read_f32(Int(py=addrs[1]), max(0, m * dim))
    var dp = _f32_ptr(Int(py=addrs[2]))
    var ip = _i32_ptr(Int(py=addrs[3]))
    var cp = _i32_ptr(Int(py=addrs[4]))
    with GILReleased(Python()):
        _ivf_run(
            x,
            n,
            dim,
            n_lists,
            queries,
            m,
            k,
            n_probes,
            kmeans_n_iters,
            metric,
            seed,
            dp,
            ip,
            cp,
        )
    return PythonObject(0)


# ===========================================================================
# BUILD AND SEARCH AS TWO CALLS (lane/inference-embedding-ivf-cholesky,
# 2026-09-15). The index crosses to Python as five arrays
# (`bindings/ivf_index_arrays.mojo`), so `IVFIndex.fit` builds once,
# `IVFIndex.save` writes the arrays and a CPU loads them. The build is
# `ivf_flat_build_host` and the search `ivf_flat_search_host`, the two halves
# `ivf_flat_build_and_search_host` calls, over host lists the upload copies,
# so the split changes no bit. Each call writes its own identity card (policy
# 3's caveat); the one-card entry above stays for the trace gates.
# ===========================================================================


def _ivf_build_run(
    x: List[Float32], n: Int, dim: Int, n_lists: Int, iters: Int, metric: Int,
    seed: UInt64,
) raises -> IvfFlatIndex:
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var index = ivf_flat_build_host(ctx, x, n, dim, n_lists, iters, metric, seed)
    ctx.synchronize()
    # DEVIATION 1946: the context dies LAST.
    _ = ctx^
    return index^


def ivf_flat_build_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`ivf_flat::build`, the index written to the caller's five buffers.
    Returns 0. See `bindings/ivf_index_arrays.mojo` for the lists."""
    var ext = ivf_build_extents(addrs, params, String("ivf_flat_build"))
    var n = ext[0]
    var dim = ext[1]
    var n_lists = ext[2]
    var bst = AnnStages("ivf_flat_binding")
    var x = read_f32(Int(py=addrs[0]), n * dim)
    bst.host("copy_in")
    var index = _ivf_build_run(x, n, dim, n_lists, ext[3], ext[4], ext[5])
    bst.host("build")
    ivf_write_index_arrays(
        addrs, index.n_rows, index.dim, index.n_lists, index.centers,
        index.center_norms, index.list_offsets, index.list_indices,
        index.list_data,
    )
    bst.host("copy_out")
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
    var m = ext[0]
    var k = ext[1]
    var n_probes = ext[2]
    var queries = read_f32(Int(py=addrs[5]), m * arrays.dim)
    # `labels` is build workspace the search never reads (cpu2-l6-bindings:
    # no longer restored by a host scatter over every row; left empty).
    var labels = List[UInt32]()
    var index = IvfFlatIndex(
        arrays.n_lists, arrays.dim, arrays.n_rows, arrays.metric,
        arrays.centers.copy(), arrays.center_norms.copy(), arrays.offsets.copy(),
        arrays.list_indices.copy(), arrays.list_data.copy(), labels^,
    )
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var r = ivf_flat_search_host(ctx, index, queries, m, k, n_probes, partial_storage=partial_storage, keep=keep)
    ctx.synchronize()
    ivf_write_search_result(addrs, r.distances, r.indices, r.n_candidates, m, k)
    _ = r^
    _ = index^
    _ = ctx^
    return PythonObject(0)


def ivf_flat_search_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    return _ivf_search_arrays(addrs, params, False)


def ivf_flat_partial_search_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """Disjoint storage with global centers/probes; squared output, valid count=min(k,candidates)."""
    return _ivf_search_arrays(addrs, params, True)


def ivf_finalize_distances_binding(address: PythonObject, count: PythonObject, metric: PythonObject) raises -> PythonObject:
    """`postprocess_distances` over the merged distances, on the device
    (lane cpu4-python: `ftz(identical_sqrt(d))` per word, the host
    statement; the host walk stays the host bindings')."""
    from ivf.impl.neighbors.ivf_common import postprocess_distances_is_identity
    var n = Int(py=count)
    if n < 0:
        raise Error("distance count must be nonnegative")
    if postprocess_distances_is_identity(Int(py=metric)) or n == 0:
        return PythonObject(0)
    if Int(py=address) == 0:
        raise Error("ivf_finalize_distances: null buffer address")
    var ctx = process_ctx[_DEVCTX_SLOT]()
    device_root_f32(ctx, Int(py=address), n)
    return PythonObject(0)


def ivf_merge_shards_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """`ivf_merge_shards` (bindings/ivf_index_arrays.mojo's contract) on the
    device (lane cpu4-python, core/shard_merge_device.mojo): [status, row]."""
    var shards = Int(py=params[0])
    var m = Int(py=params[1])
    var k = Int(py=params[2])
    if shards < 1 or m < 0 or k < 1:
        raise Error("ivf_merge_shards: need at least one shard and k >= 1")
    if len(addrs) != 3 + 4 * shards or len(params) != 3 + shards:
        raise Error("ivf_merge_shards: address or parameter count differs from the shard count")
    var d_addrs = List[Int]()
    var i_addrs = List[Int]()
    var c_addrs = List[Int]()
    var m_addrs = List[Int]()
    var sizes = List[Int]()
    for s in range(shards):  # small-loop(shards: one shard per device): reads the shard addresses, launch arguments only
        d_addrs.append(Int(py=addrs[3 + 4 * s]))
        i_addrs.append(Int(py=addrs[4 + 4 * s]))
        c_addrs.append(Int(py=addrs[5 + 4 * s]))
        m_addrs.append(Int(py=addrs[6 + 4 * s]))
        sizes.append(Int(py=params[3 + s]))
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var r = device_ivf_merge_shards(ctx, m, k, d_addrs, i_addrs, c_addrs, m_addrs, sizes,
                                    Int(py=addrs[0]), Int(py=addrs[1]), Int(py=addrs[2]))
    var out = Python.list()
    out.append(PythonObject(Int(r[0])))
    out.append(PythonObject(Int(r[1])))
    return out


def ivf_shard_plan_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """`ivf_shard_plan` (bindings/ivf_index_arrays.mojo's contract) on the
    device (lane cpu4-python, core/shard_merge_device.mojo)."""
    var n = Int(py=params[0])
    var n_lists = Int(py=params[1])
    var shards = Int(py=params[2])
    if n < 0 or n_lists < 1 or shards < 1 or len(addrs) != 5:
        raise Error("ivf_shard_plan: needs 5 addresses, n >= 0, n_lists >= 1 and shards >= 1")
    var ctx = process_ctx[_DEVCTX_SLOT]()
    return PythonObject(device_ivf_shard_plan(
        ctx, Int(py=addrs[0]), Int(py=addrs[1]), Int(py=addrs[2]), Int(py=addrs[3]), Int(py=addrs[4]),
        n, n_lists, shards))


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
    # cpu2-l6-bindings: the resident index never reads `labels` (build
    # workspace), so no host scatter over every row restores it
    var labels = List[UInt32]()
    # lane ann-apple3, behind `ANN3_PREPARE`: the admitted arrays move into
    # the index (copied otherwise, the n_rows x dim list data among them)
    var centers = List[Float32]()
    var center_norms = List[Float32]()
    var offsets = List[Int32]()
    var list_indices = List[UInt32]()
    var list_data = List[Float32]()
    comptime if ANN3_PREPARE:
        swap(centers, arrays.centers)
        swap(center_norms, arrays.center_norms)
        swap(offsets, arrays.offsets)
        swap(list_indices, arrays.list_indices)
        swap(list_data, arrays.list_data)
    else:
        centers = arrays.centers.copy()
        center_norms = arrays.center_norms.copy()
        offsets = arrays.offsets.copy()
        list_indices = arrays.list_indices.copy()
        list_data = arrays.list_data.copy()
    var index = IvfFlatIndex(
        arrays.n_lists, arrays.dim, arrays.n_rows, arrays.metric,
        centers^, center_norms^, offsets^, list_indices^, list_data^, labels^,
    )
    bst.host("index_copy")
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var handle = ivf_resident_prepare(ctx, index^, partial)
    bst.host("device_prepare")
    _ = ctx^
    return PythonObject(handle)


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
    var m = ext[0]
    var k = ext[1]
    var queries = read_f32(Int(py=addrs[0]), m * dim)
    var keep = ivf_read_resident_filter(addrs, ivf_resident_n_rows(h))
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var r = ivf_resident_search(ctx, h, queries, m, k, ext[2], keep)
    ctx.synchronize()
    ivf_write_resident_result(addrs, r.distances, r.indices, r.n_candidates, m, k)
    _ = r^
    _ = ctx^
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
    var new_x = read_f32(Int(py=addrs[5]), n_new * arrays.dim)
    # cpu2-l6-bindings: the extension reads only the NEW rows' labels, so
    # the old assignment is not restored by a host scatter (left empty);
    # `ivf_flat_extend_host` appends the new labels after it.
    var labels = List[UInt32]()
    var index = IvfFlatIndex(
        arrays.n_lists, arrays.dim, arrays.n_rows, arrays.metric,
        arrays.centers.copy(), arrays.center_norms.copy(), arrays.offsets.copy(),
        arrays.list_indices.copy(), arrays.list_data.copy(), labels^,
    )
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var out = ivf_flat_extend_host(ctx, index, new_x, n_new)
    ctx.synchronize()
    # the new rows' labels: one memcpy of the tail (no per-row host loop)
    var new_labels = List[UInt32](length=max(n_new, 0), fill=UInt32(0))
    var first_new = len(out.labels) - n_new
    if n_new > 0:
        memcpy(dest=new_labels.unsafe_ptr(), src=out.labels.unsafe_ptr() + first_new, count=n_new)
    ivf_write_extended_arrays(
        addrs, out.n_rows, out.dim, out.n_lists, out.list_offsets,
        out.list_indices, out.list_data, new_labels, n_new,
    )
    _ = out^
    _ = index^
    _ = ctx^
    return PythonObject(0)


@export
def PyInit__mojolearn_ivf() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_ivf")
        m.def_function[ivf_vendor_binding]("ivf_vendor")
        m.def_function[ivf_numeric_mode_binding]("ivf_numeric_mode")
        # NEVER RUN — PENDING MEASUREMENT. New candidate remains opt-in/default OFF.
        comptime if is_defined["MOJOLEARN_IVF_FAST_BALANCED_AUDIT"]():
            m.def_function[ivf_fast_balanced_hits_binding]("ivf_fast_balanced_hits")
        m.def_function[ivf_flat_build_and_search_binding](
            "ivf_flat_build_and_search"
        )
        m.def_function[ivf_flat_build_binding]("ivf_flat_build")
        m.def_function[ivf_flat_search_binding]("ivf_flat_search")
        m.def_function[ivf_flat_partial_search_binding]("ivf_flat_partial_search")
        m.def_function[ivf_finalize_distances_binding]("ivf_finalize_distances")
        m.def_function[ivf_flat_extend_binding]("ivf_flat_extend")
        m.def_function[ivf_flat_index_prepare_binding]("ivf_flat_index_prepare")
        m.def_function[ivf_flat_index_search_binding]("ivf_flat_index_search")
        m.def_function[ivf_flat_index_release_binding]("ivf_flat_index_release")
        m.def_function[ivf_merge_shards_binding]("ivf_merge_shards")
        m.def_function[ivf_shard_plan_binding]("ivf_shard_plan")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_ivf: ", e))

def ivf_fast_balanced_hits_binding() raises -> PythonObject:
    return PythonObject(ivf_fast_balanced_hits())
