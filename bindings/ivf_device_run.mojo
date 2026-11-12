# SPDX-License-Identifier: Apache-2.0
"""RUN bodies of the IVF-Flat GPU binding (bindings/_mojolearn_ivf.mojo): the
build, the array search, the extend, the resident prepare, build and search,
over typed addresses and extents on the binding's one process-lifetime
DeviceContext. Typed Mojo only, no Python import (tools/python_free_dirs.sh).
The doors keep the list-count checks, the extent admission and the reads of
the caller's address words; the statements here are the former door
statements after those reads, unchanged, so the bits are the same (lane
rs-decomposition, 2026-11-12)."""
from std.memory import unsafe_memcpy
from core.hostptr import f32_ptr, i32_ptr, read_f32
from core.neural_context import process_ctx
from bindings.parallel_ivf_entry_device import IVF_DEVCTX_SLOT
from ivf.estimator import (
    ivf_flat_build_host,
    ivf_flat_build_resident_host,
    ivf_flat_extend_host,
    ivf_flat_search_host,
)
from ivf.impl.neighbors.ivf_flat.ivf_flat_index import IvfFlatIndex
from ivf.impl.neighbors.ivf_flat.ivf_flat_build import IVF_BUILD_FROM_POINTER, IvfFlatBuildDevice
from ivf.resident import ivf_resident_prepare, ivf_resident_register_built, ivf_resident_search
from ivf.ivf_index_arrays_run import (
    IvfIndexArrays,
    ivf_write_extended_arrays_run,
    ivf_write_index_arrays_run,
    ivf_write_search_result_run,
)
from x_ann.stage_timer import AnnStages
from x_ann.switches import ANN3_PREPARE

#: This binding's ONE process-lifetime DeviceContext (core/neural_context.mojo,
#: lane/devctx-lifetime): a context per call exhausts Metal command queues.
#: Defined once beside the IVF table entries that share it.
comptime _DEVCTX_SLOT = IVF_DEVCTX_SLOT


def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    return f32_ptr(addr)


def _i32_ptr(addr: Int) raises -> MutPointer[Int32, MutUntrackedOrigin]:
    return i32_ptr(addr)


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


def ivf_flat_build_run(
    x_addr: Int, centers_addr: Int, center_norms_addr: Int, offsets_addr: Int, indices_addr: Int,
    list_data_addr: Int, n: Int, dim: Int, n_lists: Int, iters: Int, metric: Int, seed: UInt64,
) raises:
    """`ivf_flat::build` from the rows at `x_addr`, the index written to the
    five output buffers (`ivf_flat_build_binding`)."""
    var bst = AnnStages("ivf_flat_binding")
    var x = read_f32(x_addr, n * dim)
    bst.host("copy_in")
    var index = _ivf_build_run(x, n, dim, n_lists, iters, metric, seed)
    bst.host("build")
    ivf_write_index_arrays_run(
        centers_addr, center_norms_addr, offsets_addr, indices_addr, list_data_addr,
        index.n_rows, index.dim, index.n_lists, index.centers,
        index.center_norms, index.list_offsets, index.list_indices,
        index.list_data,
    )
    bst.host("copy_out")


def ivf_device_search_run(
    var arrays: IvfIndexArrays, keep: List[Int32], queries_addr: Int, m: Int, k: Int, n_probes: Int,
    partial_storage: Bool, dist_addr: Int, idx_addr: Int, cand_addr: Int,
) raises:
    """`ivf_flat::search` over an admitted index (`_ivf_search_arrays`)."""
    var queries = read_f32(queries_addr, m * arrays.dim)
    # `labels` is build workspace the search never reads (cpu2-l6-bindings:
    # no longer restored by a host scatter over every row; left empty).
    var labels = List[UInt32]()
    # p3-neighbors-cluster: the admitted arrays move into the index (as the
    # host binding does); they were copied a second time on every call.
    # compile-fix-1: each list moves out and its field is refilled empty, so
    # `arrays` stays whole and is destroyed as one value (moving fields out of
    # the middle of it does not compile); still no copy of the index arrays.
    var taken_centers = arrays.centers^
    arrays.centers = List[Float32]()
    var taken_center_norms = arrays.center_norms^
    arrays.center_norms = List[Float32]()
    var taken_offsets = arrays.offsets^
    arrays.offsets = List[Int32]()
    var taken_list_indices = arrays.list_indices^
    arrays.list_indices = List[UInt32]()
    var taken_list_data = arrays.list_data^
    arrays.list_data = List[Float32]()
    var index = IvfFlatIndex(
        arrays.n_lists, arrays.dim, arrays.n_rows, arrays.metric,
        taken_centers^, taken_center_norms^, taken_offsets^,
        taken_list_indices^, taken_list_data^, labels^,
    )
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var r = ivf_flat_search_host(ctx, index, queries, m, k, n_probes, partial_storage=partial_storage, keep=keep)
    ctx.synchronize()
    ivf_write_search_result_run(dist_addr, idx_addr, cand_addr, r.distances, r.indices, r.n_candidates, m, k)
    _ = r^
    _ = index^
    _ = ctx^


def ivf_device_prepare_run(var arrays: IvfIndexArrays, partial: Bool, mut bst: AnnStages) raises -> Int:
    """The resident index from an admitted index; its handle
    (`ivf_flat_index_prepare_binding`)."""
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
    return handle


def ivf_flat_build_resident_run(
    x_addr: Int, centers_addr: Int, center_norms_addr: Int, offsets_addr: Int, indices_addr: Int,
    n: Int, dim: Int, n_lists: Int, iters: Int, metric: Int, seed: UInt64,
) raises -> Int:
    """The resident build and its four small arrays; the handle
    (`ivf_flat_build_resident_binding`)."""
    var bst = AnnStages("ivf_flat_build_resident")
    # IDENTICAL reads the rows straight from the caller's buffer (no host
    # copy); every other build copies them in as `ivf_flat_build` does
    var x = List[Float32]()
    comptime if not IVF_BUILD_FROM_POINTER:
        x = read_f32(x_addr, n * dim)
    bst.host("copy_in")
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var dev = IvfFlatBuildDevice(ctx)
    var index = ivf_flat_build_resident_host(ctx, x, x_addr, n, dim, n_lists, iters, metric, seed, dev)
    ctx.synchronize()
    _ = x^
    bst.host("build")
    # the four small arrays: one memcpy each (the ids cross as int32: below
    # 2^31 by the extent check, so the uint32 words are the int32 words)
    if len(index.centers) != n_lists * dim or len(index.center_norms) != n_lists:
        raise Error("ivf_flat_build_resident: the built centres disagree with n_lists and dim")
    if len(index.list_offsets) != n_lists + 1 or len(index.list_indices) != n:
        raise Error("ivf_flat_build_resident: the built lists disagree with n and n_lists")
    unsafe_memcpy(dest=_f32_ptr(centers_addr), src=index.centers.unsafe_ptr(), count=n_lists * dim)
    unsafe_memcpy(dest=_f32_ptr(center_norms_addr), src=index.center_norms.unsafe_ptr(), count=n_lists)
    unsafe_memcpy(dest=_i32_ptr(offsets_addr), src=index.list_offsets.unsafe_ptr(), count=n_lists + 1)
    if n > 0:
        unsafe_memcpy(dest=_i32_ptr(indices_addr).bitcast[UInt32](), src=index.list_indices.unsafe_ptr(), count=n)
    var handle = ivf_resident_register_built(ctx, index^, dev^)
    bst.host("register")
    _ = ctx^
    return handle


def ivf_device_resident_search_run(
    h: Int, queries_addr: Int, dim: Int, m: Int, k: Int, n_probes: Int, keep: List[Int32],
    dist_addr: Int, idx_addr: Int, cand_addr: Int,
) raises:
    """A search of the resident index `h` (`ivf_flat_index_search_binding`)."""
    var queries = read_f32(queries_addr, m * dim)
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var r = ivf_resident_search(ctx, h, queries, m, k, n_probes, keep)
    ctx.synchronize()
    ivf_write_search_result_run(dist_addr, idx_addr, cand_addr, r.distances, r.indices, r.n_candidates, m, k)
    _ = r^
    _ = ctx^


def ivf_device_extend_run(
    var arrays: IvfIndexArrays, new_x_addr: Int, n_new: Int, offsets_addr: Int, indices_addr: Int,
    list_data_addr: Int, labels_addr: Int,
) raises:
    """`ivf_flat::extend` over an admitted index, the extended lists written
    to the four output buffers (`ivf_flat_extend_binding`)."""
    var new_x = read_f32(new_x_addr, n_new * arrays.dim)
    # cpu2-l6-bindings: the extension reads only the NEW rows' labels, so
    # the old assignment is not restored by a host scatter (left empty);
    # `ivf_flat_extend_host` appends the new labels after it.
    var labels = List[UInt32]()
    # p3-neighbors-cluster: the admitted arrays move into the index (as the
    # host binding does); they were copied a second time on every call.
    # compile-fix-1: each list moves out and its field is refilled empty, so
    # `arrays` stays whole and is destroyed as one value (moving fields out of
    # the middle of it does not compile); still no copy of the index arrays.
    var taken_centers = arrays.centers^
    arrays.centers = List[Float32]()
    var taken_center_norms = arrays.center_norms^
    arrays.center_norms = List[Float32]()
    var taken_offsets = arrays.offsets^
    arrays.offsets = List[Int32]()
    var taken_list_indices = arrays.list_indices^
    arrays.list_indices = List[UInt32]()
    var taken_list_data = arrays.list_data^
    arrays.list_data = List[Float32]()
    var index = IvfFlatIndex(
        arrays.n_lists, arrays.dim, arrays.n_rows, arrays.metric,
        taken_centers^, taken_center_norms^, taken_offsets^,
        taken_list_indices^, taken_list_data^, labels^,
    )
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var out = ivf_flat_extend_host(ctx, index, new_x, n_new)
    ctx.synchronize()
    # the new rows' labels: one memcpy of the tail (no per-row host loop)
    var new_labels = List[UInt32](length=max(n_new, 0), fill=UInt32(0))
    var first_new = len(out.labels) - n_new
    if n_new > 0:
        unsafe_memcpy(dest=new_labels.unsafe_ptr(), src=out.labels.unsafe_ptr() + first_new, count=n_new)
    ivf_write_extended_arrays_run(
        offsets_addr, indices_addr, list_data_addr, labels_addr, out.n_rows, out.dim, out.n_lists,
        out.list_offsets, out.list_indices, out.list_data, new_labels, n_new,
    )
    _ = out^
    _ = index^
    _ = ctx^
