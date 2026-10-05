# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`coo_sort_by_weight`, from RAFT, with the order made total.

Reference: `coo_sort_by_weight`, `raft/cpp/include/raft/sparse/op/detail/sort.h:94-102`
(RAFT `661a3b8`). The call site is
`cuvs/cpp/src/cluster/detail/mst.cuh:337-338`, right before the edges are
copied to the host for `build_dendrogram_host`.

WHERE IT RUNS. The reference runs `thrust::sort_by_key` on the device; this implementation sorts
on the HOST, because the next consumer of the sorted list is
`build_dendrogram_host`'s `raft::update_host` (`agglomerative.cuh:122-124`)
and the list is `m - 1` edges long. The device list is rewritten in the
sorted order afterwards so the device-side artifact is the same sorted
list the reference leaves behind. This moves WHERE the sort happens, not what is
sorted or how the result is ordered -- except for the order among ties,
which is the deviation below.

======================================================================
DEVIATION BLOCK -- DEVIATION 621. THE MST SORT IS BY (weight, min(u,v),
max(u,v)), A TOTAL ORDER; THE REFERENCE SORTS BY WEIGHT ALONE AND UNSTABLE.
======================================================================

WHAT THE REFERENCE DOES. `thrust::sort_by_key(t_data, t_data + nnz, zip(rows,
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

WHAT THIS IMPLEMENTATION DOES. The sort key is `pack_edge_key(weight_order_key(w),
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

from hierarchy.checks.edge_order import (
    LINK_SAB_NONE,
    LINK_SAB_SORT_WEIGHT_ONLY,
    edge_hi,
    edge_lo,
    pack_edge_key,
    weight_order_key,
)


def merge_sort_u64_with_index_host(
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


# `coo_sort_by_weight` runs on the device since lane hr2-mds-agglo
# (`hierarchy/impl/cluster/detail/dendrogram_device.mojo`
# `coo_sort_by_weight_device`): the same keys, ranked in parallel.
