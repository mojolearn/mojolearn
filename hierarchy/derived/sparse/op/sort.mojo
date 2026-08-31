# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`coo_sort_by_weight`, from RAFT, with the order made total.

PORT OF `raft/cpp/include/raft/sparse/op/detail/sort.h:94-102`
(`coo_sort_by_weight`), RAFT `661a3b8`. The call site is
`cuvs/cpp/src/cluster/detail/mst.cuh:337-338`, right before the edges are
copied to the host for `build_dendrogram_host`.

WHERE IT RUNS. Theirs is `thrust::sort_by_key` on the device; ours sorts
on the HOST, because the next consumer of the sorted list is
`build_dendrogram_host`'s `raft::update_host` (`agglomerative.cuh:122-124`)
and the list is `m - 1` edges long. The device list is rewritten in the
sorted order afterwards so the device-side artifact is the same sorted
list theirs leaves behind. This moves WHERE the sort happens, not what is
sorted or how the result is ordered -- except for the order among ties,
which is the deviation below.

======================================================================
DEVIATION BLOCK -- DEVIATION 621. THE MST SORT IS BY (weight, min(u,v),
max(u,v)), A TOTAL ORDER; THEIRS IS BY WEIGHT ALONE AND UNSTABLE.
======================================================================

WHAT THEIRS DOES. `thrust::sort_by_key(t_data, t_data + nnz, zip(rows,
cols))` (`sort.h:101`): keys are the weights, the payload is the (row,
col) pair. `thrust::sort_by_key` is NOT stable (Thrust documents
`stable_sort_by_key` separately), so two MST edges of EQUAL weight come
out in an order the sort implementation chooses -- radix pass order, block
shape, CUB version -- and the dendrogram (`build_dendrogram_host`,
`agglomerative.cuh:134-150`) walks the sorted list IN ORDER, so a swap of
two equal-weight edges is a swap of two merge rows in `children`, and
when the swap straddles the `n_clusters` cut, a different partition.

HOW IT COULD PASS UNNOTICED ON THEIR SIDE. Under DEVIATION 620's
alteration every MST weight is DISTINCT in the altered space, but
`temp_weights` carries the ORIGINAL float out (`mst_kernels.cuh:148`) and
the sort keys on the original, so equal original weights are ties again
here even on their side. With duplicate points (weight 0) or grid data
this is the common case, not the corner.

WHAT OURS DOES. The sort key is `pack_edge_key(weight_order_key(w),
min(u,v), max(u,v))`, the same total order the MST itself used, so the
sorted list is a pure function of the edge SET. Two distinct MST edges
never compare equal, so stability is moot, and the sort is a merge sort
(deterministic, host). `linkage_check.mojo`'s `LINK_SAB_SORT_WEIGHT_ONLY`
sorts by weight with ties in reverse discovery order -- an order Thrust is
permitted to return -- and the dendrogram gate fails on the equal-distance
fixture; that is the measurement.
======================================================================
"""

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from hierarchy.original.edge_order import (
    LINK_SAB_NONE,
    LINK_SAB_SORT_WEIGHT_ONLY,
    edge_hi,
    edge_lo,
    pack_edge_key,
    weight_order_key,
)


def merge_sort_u64_with_index(
    mut keys: List[UInt64], mut idx: List[Int]
):
    """Bottom-up merge sort of `keys` carrying `idx` along. Stable, so a
    caller who packs a non-total key still gets a defined (discovery)
    order among ties."""
    var n = len(keys)
    if n < 2:
        return
    var tk = List[UInt64](capacity=n)
    var ti = List[Int](capacity=n)
    for _ in range(n):
        tk.append(UInt64(0))
        ti.append(0)
    var width = 1
    while width < n:
        var lo = 0
        while lo < n:
            var mid = lo + width
            if mid > n:
                mid = n
            var hi = lo + 2 * width
            if hi > n:
                hi = n
            var i = lo
            var j = mid
            var k = lo
            while i < mid and j < hi:
                if keys[j] < keys[i]:
                    tk[k] = keys[j]
                    ti[k] = idx[j]
                    j += 1
                else:
                    tk[k] = keys[i]
                    ti[k] = idx[i]
                    i += 1
                k += 1
            while i < mid:
                tk[k] = keys[i]
                ti[k] = idx[i]
                i += 1
                k += 1
            while j < hi:
                tk[k] = keys[j]
                ti[k] = idx[j]
                j += 1
                k += 1
            lo += 2 * width
        for t in range(n):
            keys[t] = tk[t]
            idx[t] = ti[t]
        width *= 2


def coo_sort_by_weight(
    ctx: DeviceContext,
    mut rows: DeviceBuffer[DType.int32],
    mut cols: DeviceBuffer[DType.int32],
    mut data: DeviceBuffer[DType.float32],
    nnz: Int,
    sabotage: Int32 = LINK_SAB_NONE,
) raises:
    """`sort.h:94-102`, in place on the three device arrays, total order
    (DEVIATION 621). `sabotage == LINK_SAB_SORT_WEIGHT_ONLY` keys on the
    weight alone with ties in REVERSE discovery order, for the check."""
    if nnz <= 1:
        return
    var h_rows = ctx.enqueue_create_host_buffer[DType.int32](nnz)
    var h_cols = ctx.enqueue_create_host_buffer[DType.int32](nnz)
    var h_data = ctx.enqueue_create_host_buffer[DType.float32](nnz)
    ctx.synchronize()
    var v_rows = rows.create_sub_buffer[DType.int32](0, nnz)
    var v_cols = cols.create_sub_buffer[DType.int32](0, nnz)
    var v_data = data.create_sub_buffer[DType.float32](0, nnz)
    ctx.enqueue_copy(dst_ptr=h_rows.unsafe_ptr(), src_buf=v_rows)
    ctx.enqueue_copy(dst_ptr=h_cols.unsafe_ptr(), src_buf=v_cols)
    ctx.enqueue_copy(dst_ptr=h_data.unsafe_ptr(), src_buf=v_data)
    ctx.synchronize()

    var keys = List[UInt64](capacity=nnz)
    var idx = List[Int](capacity=nnz)
    for i in range(nnz):
        var u = h_rows.unsafe_ptr().unsafe_load(i)
        var v = h_cols.unsafe_ptr().unsafe_load(i)
        var wk = weight_order_key(h_data.unsafe_ptr().unsafe_load(i))
        if sabotage == LINK_SAB_SORT_WEIGHT_ONLY:
            # weight only; reverse discovery order among ties
            keys.append(pack_edge_key(wk, Int32(0), Int32(0)) | UInt64(nnz - 1 - i))
        else:
            keys.append(pack_edge_key(wk, edge_lo(u, v), edge_hi(u, v)))
        idx.append(i)
    merge_sort_u64_with_index(keys, idx)

    var s_rows = ctx.enqueue_create_host_buffer[DType.int32](nnz)
    var s_cols = ctx.enqueue_create_host_buffer[DType.int32](nnz)
    var s_data = ctx.enqueue_create_host_buffer[DType.float32](nnz)
    ctx.synchronize()
    for k in range(nnz):
        var i = idx[k]
        s_rows.unsafe_ptr().unsafe_store(k, h_rows.unsafe_ptr().unsafe_load(i))
        s_cols.unsafe_ptr().unsafe_store(k, h_cols.unsafe_ptr().unsafe_load(i))
        s_data.unsafe_ptr().unsafe_store(k, h_data.unsafe_ptr().unsafe_load(i))
    ctx.enqueue_copy(dst_buf=v_rows, src_ptr=s_rows.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=v_cols, src_ptr=s_cols.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=v_data, src_ptr=s_data.unsafe_ptr())
    ctx.synchronize()
    _ = h_rows^
    _ = h_cols^
    _ = h_data^
    _ = s_rows^
    _ = s_cols^
    _ = s_data^
