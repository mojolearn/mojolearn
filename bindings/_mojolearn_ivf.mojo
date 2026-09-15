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
from bindings.hostptr import f32_ptr, i32_ptr, copy_f32, read_f32
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from max.gpu.host import DeviceContext

from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from bindings.ivf_index_arrays import (
    ivf_build_extents,
    ivf_extend_count,
    ivf_write_extended_arrays,
    ivf_read_index_arrays,
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
    var ctx = DeviceContext()
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
    for i in range(m * k):
        ip.unsafe_store(i, Int32(Int(r.indices[i])))
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
    var ctx = DeviceContext()
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
    var x = read_f32(Int(py=addrs[0]), n * dim)
    var index = _ivf_build_run(x, n, dim, n_lists, ext[3], ext[4], ext[5])
    ivf_write_index_arrays(
        addrs, index.n_rows, index.dim, index.n_lists, index.centers,
        index.center_norms, index.list_offsets, index.list_indices,
        index.list_data,
    )
    return PythonObject(0)


def ivf_flat_search_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`ivf_flat::search` over a built index handed back as five arrays and
    admitted by `ivf_validate_index_arrays`. Returns 0."""
    var arrays = ivf_read_index_arrays(addrs, params, String("ivf_flat_search"))
    var ext = ivf_search_extents(params)
    var m = ext[0]
    var k = ext[1]
    var n_probes = ext[2]
    var queries = read_f32(Int(py=addrs[5]), m * arrays.dim)
    # `labels` is build workspace the search never reads; it is restored
    # from the carried ids so the struct holds the assignment it describes.
    var labels = List[UInt32](length=arrays.n_rows, fill=UInt32(0))
    for l in range(arrays.n_lists):
        for s in range(Int(arrays.offsets[l]), Int(arrays.offsets[l + 1])):
            labels[Int(arrays.list_indices[s])] = UInt32(l)
    var index = IvfFlatIndex(
        arrays.n_lists, arrays.dim, arrays.n_rows, arrays.metric,
        arrays.centers.copy(), arrays.center_norms.copy(), arrays.offsets.copy(),
        arrays.list_indices.copy(), arrays.list_data.copy(), labels^,
    )
    var ctx = DeviceContext()
    var r = ivf_flat_search_host(ctx, index, queries, m, k, n_probes)
    ctx.synchronize()
    ivf_write_search_result(addrs, r.distances, r.indices, r.n_candidates, m, k)
    _ = r^
    _ = index^
    _ = ctx^
    return PythonObject(0)


def _labels_from_arrays(offsets: List[Int32], list_indices: List[UInt32], n_lists: Int, n_rows: Int) -> List[UInt32]:
    """The assignment an admitted index describes, from its carried ids."""
    var labels = List[UInt32](length=n_rows, fill=UInt32(0))
    for l in range(n_lists):
        for s in range(Int(offsets[l]), Int(offsets[l + 1])):
            labels[Int(list_indices[s])] = UInt32(l)
    return labels^


def ivf_flat_extend_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`ivf_flat::extend` over a built index handed back as five arrays
    (stage 2 of lane/inference-embedding-ivf-cholesky, 2026-09-15). Returns 0.
    See `bindings/ivf_index_arrays.mojo` for the lists."""
    var arrays = ivf_read_index_arrays(addrs, params, String("ivf_flat_extend"), 10, 5)
    var n_new = ivf_extend_count(params, arrays.n_rows, arrays.dim)
    var new_x = read_f32(Int(py=addrs[5]), n_new * arrays.dim)
    var labels = _labels_from_arrays(arrays.offsets, arrays.list_indices, arrays.n_lists, arrays.n_rows)
    var index = IvfFlatIndex(
        arrays.n_lists, arrays.dim, arrays.n_rows, arrays.metric,
        arrays.centers.copy(), arrays.center_norms.copy(), arrays.offsets.copy(),
        arrays.list_indices.copy(), arrays.list_data.copy(), labels^,
    )
    var ctx = DeviceContext()
    var out = ivf_flat_extend_host(ctx, index, new_x, n_new)
    ctx.synchronize()
    var new_labels = List[UInt32](capacity=n_new)
    for j in range(n_new):
        new_labels.append(out.labels[arrays.n_rows + j])
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
        m.def_function[ivf_flat_build_and_search_binding](
            "ivf_flat_build_and_search"
        )
        m.def_function[ivf_flat_build_binding]("ivf_flat_build")
        m.def_function[ivf_flat_search_binding]("ivf_flat_search")
        m.def_function[ivf_flat_extend_binding]("ivf_flat_extend")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_ivf: ", e))
