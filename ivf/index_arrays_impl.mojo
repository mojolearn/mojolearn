# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The typed bodies behind `bindings/ivf_index_arrays.mojo`'s Python doors
(lane thin-trees-metrics, 2026-10-21: moved word for word out of that shell,
so a Python-free caller reaches the same source; no Python here). The
address and parameter contracts are that file's."""
from std.memory import memcpy

from core.hostptr import f32_ptr, i32_ptr, read_f32
from ivf.impl.neighbors.ivf_common import postprocess_distances
from x_ann.switches import ANN3_HOST_PASSES


def ivf_store_search_result(
    dp: MutPointer[Float32, MutUntrackedOrigin],
    ip: MutPointer[Int32, MutUntrackedOrigin],
    cp: MutPointer[Int32, MutUntrackedOrigin],
    distances: List[Float32],
    indices: List[UInt32],
    n_candidates: List[Int32],
    m: Int,
    k: Int,
):
    """Distances, ORIGINAL row ids as int32 (below 2^31 by the row bound)
    and the candidate count per query."""
    for i in range(m * k):
        dp.unsafe_store(i, distances[i])
        ip.unsafe_store(i, Int32(Int(indices[i])))
    for i in range(m):
        cp.unsafe_store(i, n_candidates[i])



def ivf_store_ids_counts(
    ip: MutPointer[Int32, MutUntrackedOrigin],
    cp: MutPointer[Int32, MutUntrackedOrigin],
    indices: List[UInt32],
    n_candidates: List[Int32],
    m: Int,
    k: Int,
):
    """ORIGINAL row ids as int32 and the candidate count per query (the
    host build-and-search door's tail after its distance copy)."""
    for i in range(m * k):
        ip.unsafe_store(i, Int32(Int(indices[i])))
    for i in range(m):
        cp.unsafe_store(i, n_candidates[i])

def ivf_store_index_arrays(
    cp: MutPointer[Float32, MutUntrackedOrigin],
    np_: MutPointer[Float32, MutUntrackedOrigin],
    op: MutPointer[Int32, MutUntrackedOrigin],
    ip: MutPointer[Int32, MutUntrackedOrigin],
    lp: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    dim: Int,
    n_lists: Int,
    centers: List[Float32],
    center_norms: List[Float32],
    offsets: List[Int32],
    list_indices: List[UInt32],
    list_data: List[Float32],
) raises:
    """`ivf_flat_build`'s five outputs from a built index."""
    for i in range(n_lists * dim):
        cp.unsafe_store(i, centers[i])
    for i in range(n_lists):
        np_.unsafe_store(i, center_norms[i])
    for i in range(n_lists + 1):
        op.unsafe_store(i, offsets[i])
    for i in range(n_rows):
        ip.unsafe_store(i, Int32(Int(list_indices[i])))
    # lane ann-apple3, behind `ANN3_HOST_PASSES`: the n_rows x dim vectors
    # in one memcpy (the same words)
    if len(list_data) != n_rows * dim:
        raise Error("ivf_flat_build: the built index holds no list data")
    comptime if ANN3_HOST_PASSES:
        memcpy(dest=lp, src=list_data.unsafe_ptr(), count=n_rows * dim)
    else:
        for i in range(n_rows * dim):
            lp.unsafe_store(i, list_data[i])


def ivf_store_extended_arrays(
    op: MutPointer[Int32, MutUntrackedOrigin],
    ip: MutPointer[Int32, MutUntrackedOrigin],
    lp: MutPointer[Float32, MutUntrackedOrigin],
    bp: MutPointer[Int32, MutUntrackedOrigin],
    n_total: Int,
    dim: Int,
    n_lists: Int,
    offsets: List[Int32],
    list_indices: List[UInt32],
    list_data: List[Float32],
    new_labels: List[UInt32],
    n_new: Int,
):
    """`ivf_flat_extend`'s four outputs from the extended index."""
    for i in range(n_lists + 1):
        op.unsafe_store(i, offsets[i])
    for i in range(n_total):
        ip.unsafe_store(i, Int32(Int(list_indices[i])))
    for i in range(n_total * dim):
        lp.unsafe_store(i, list_data[i])
    for j in range(n_new):
        bp.unsafe_store(j, Int32(Int(new_labels[j])))


def ivf_merge_shards_host(
    od: MutPointer[Float32, MutUntrackedOrigin],
    oi: MutPointer[Int32, MutUntrackedOrigin],
    oc: MutPointer[Int32, MutUntrackedOrigin],
    sd: List[Int],
    si: List[Int],
    sc: List[Int],
    sm: List[Int],
    sn: List[Int],
    shards: Int,
    m: Int,
    k: Int,
) raises -> Tuple[Int, Int]:
    """`ivf_merge_shards`: the shards' candidates merged per query row by
    (float32 distance, original id), the first k written; returns
    (status, bad_row) as the shell's contract says."""
    var cd = List[Float32](capacity=shards * k)
    var ci = List[Int32](capacity=shards * k)
    var status = 0
    var bad_row = 0
    for row in range(m):
        cd.clear()
        ci.clear()
        var total = 0
        for s in range(shards):
            var n = Int(i32_ptr(sc[s]).unsafe_load(row))
            if n < 0 or n > sn[s]:
                status = 1
                break
            total += n
            var dp = f32_ptr(sd[s])
            var ip = i32_ptr(si[s])
            var mp = i32_ptr(sm[s])
            for j in range(min(k, n)):
                var local = Int(ip.unsafe_load(row * k + j))
                if local < 0 or local >= sn[s]:
                    status = 2
                    break
                var d = dp.unsafe_load(row * k + j)
                if d != d:
                    status = 4
                    break
                var id = mp.unsafe_load(local)
                # insertion into the (distance, id) order
                var at = len(cd)
                cd.append(d)
                ci.append(id)
                while at > 0 and (cd[at - 1] > d or (cd[at - 1] == d and ci[at - 1] > id)):
                    cd[at] = cd[at - 1]
                    ci[at] = ci[at - 1]
                    at -= 1
                cd[at] = d
                ci[at] = id
            if status != 0:
                break
        if status == 0 and total < k:
            status = 3
        if status != 0:
            bad_row = row
            break
        for j in range(k):
            od.unsafe_store(row * k + j, cd[j])
            oi.unsafe_store(row * k + j, ci[j])
        oc.unsafe_store(row, Int32(total))
    return (status, bad_row)


def ivf_shard_plan_host(
    ind: MutPointer[Int32, MutUntrackedOrigin],
    off: MutPointer[Int32, MutUntrackedOrigin],
    soff: MutPointer[Int32, MutUntrackedOrigin],
    loc: MutPointer[Int32, MutUntrackedOrigin],
    mp: MutPointer[Int32, MutUntrackedOrigin],
    n: Int,
    n_lists: Int,
    shards: Int,
) raises -> Int:
    """`ivf_shard_plan`'s validation and row split; returns the status
    (0 planned, 1 the ids are not a permutation, 2 the offsets are not
    0 .. n nondecreasing)."""
    var L1 = n_lists + 1
    if Int(off.unsafe_load(0)) != 0 or Int(off.unsafe_load(n_lists)) != n:
        return 2
    for j in range(n_lists):
        if off.unsafe_load(j) > off.unsafe_load(j + 1):
            return 2
    # owner shard of every original id; a repeat or an id out of range is
    # not a permutation
    var owner = List[Int32](length=n, fill=Int32(-1))
    for p in range(shards):
        var lo = p * n // shards
        var hi = (p + 1) * n // shards
        for i in range(lo, hi):
            var v = Int(ind.unsafe_load(i))
            if v < 0 or v >= n or owner[v] != -1:
                return 1
            owner[v] = Int32(p)
    # ids ascending: each shard's map and every id's rank in its shard
    var cnt = List[Int](length=shards, fill=0)
    var rank = List[Int32](length=n, fill=Int32(0))
    for v in range(n):
        var p = Int(owner[v])
        var r = cnt[p]
        mp.unsafe_store(p * n // shards + r, Int32(v))
        rank[v] = Int32(r)
        cnt[p] = r + 1
    for i in range(n):
        loc.unsafe_store(i, rank[Int(ind.unsafe_load(i))])
    for p in range(shards):
        var lo = p * n // shards
        var hi = (p + 1) * n // shards
        for j in range(L1):
            var o = Int(off.unsafe_load(j))
            soff.unsafe_store(p * L1 + j, Int32(max(0, min(o, hi) - lo)))
    return 0


def ivf_finalize_distances_host(address: Int, n: Int, metric: Int) raises:
    """`postprocess_distances` applied in place to `n` float32 at
    `address` (`bindings/ivf_host_search.mojo::ivf_finalize_distances`)."""
    var distances = read_f32(address, n)
    postprocess_distances(distances, metric)
    var dst = f32_ptr(address)
    for i in range(n):
        dst.unsafe_store(i, distances[i])
