# SPDX-License-Identifier: Apache-2.0
"""RUN pieces of the IVF-Flat five-buffer contract (bindings/ivf_index_arrays.mojo):
the admitted index record, the cell bound, the reads and writes of the index,
search and extend buffers at plain addresses, and the host merge and shard
plan of the parallel IVF table. Typed Mojo only, no Python import
(tools/python_free_dirs.sh). The door reads the address and parameter words
from the caller's lists and checks their counts; these bodies are the former
statements after those reads, unchanged, so the bits are the same (lane
rs-decomposition, 2026-11-12)."""
from core.hostptr import f32_ptr, i32_ptr, u32_ptr, read_f32, read_i32, list_u32
from ivf.impl.neighbors.ivf_flat.ivf_flat_index import ivf_validate_index_arrays
from ivf.index_arrays_impl import (
    ivf_merge_shards_host,
    ivf_shard_plan_host,
    ivf_store_extended_arrays,
    ivf_store_index_arrays,
    ivf_store_search_result,
)


#: Extents far from any Int edge: rows and lists below 2^31 (the ids cross
#: as int32), `n * dim` and `m * k` below 2^40.
comptime IVF_ARRAYS_MAX_ROWS = 2147483647
comptime IVF_ARRAYS_MAX_CELLS = 1099511627776


@fieldwise_init
struct IvfIndexArrays(Movable):
    """A built index read from the caller's five buffers and admitted."""

    var n_lists: Int
    var dim: Int
    var n_rows: Int
    var metric: Int
    var centers: List[Float32]
    var center_norms: List[Float32]
    var offsets: List[Int32]
    var list_indices: List[UInt32]
    var list_data: List[Float32]


def ivf_arrays_check_cells(a: Int, b: Int, what: String) raises:
    if a > 0 and b > IVF_ARRAYS_MAX_CELLS // a:
        raise Error("ivf_flat: " + what + " exceeds 2^40 cells")


def ivf_read_index_arrays_run(
    centers_addr: Int,
    center_norms_addr: Int,
    offsets_addr: Int,
    indices_addr: Int,
    list_data_addr: Int,
    n_lists: Int,
    dim: Int,
    n: Int,
    metric: Int,
    partial_storage: Bool,
) raises -> IvfIndexArrays:
    """The RUN half of `ivf_read_index_arrays`: the five index buffers at
    their addresses, read and admitted."""
    var centers = read_f32(centers_addr, n_lists * dim)
    var center_norms = read_f32(center_norms_addr, n_lists)
    var offsets = read_i32(offsets_addr, n_lists + 1)
    var list_data = read_f32(list_data_addr, n * dim)
    # cpu3-bindings: the int32 ids' bits in one memcpy (no per-slot host
    # loop). A negative id reads as >= 2^31 > n, so the id range check of
    # `ivf_validate_index_arrays` below refuses it by name.
    var list_indices = list_u32(u32_ptr(indices_addr), n)
    ivf_validate_index_arrays(
        n_lists, dim, n, metric, centers, center_norms, offsets, list_indices,
        list_data, partial_storage,
    )
    return IvfIndexArrays(
        n_lists, dim, n, metric, centers^, center_norms^, offsets^,
        list_indices^, list_data^,
    )


def ivf_write_search_result_run(
    dist_addr: Int,
    idx_addr: Int,
    cand_addr: Int,
    distances: List[Float32],
    indices: List[UInt32],
    n_candidates: List[Int32],
    m: Int,
    k: Int,
) raises:
    """The RUN half of `ivf_write_search_result`: distances, ORIGINAL row
    ids as int32 and the candidate count per query, at their addresses."""
    var dp = f32_ptr(dist_addr)
    var ip = i32_ptr(idx_addr)
    var cp = i32_ptr(cand_addr)
    ivf_store_search_result(dp, ip, cp, distances, indices, n_candidates, m, k)


def ivf_write_index_arrays_run(
    centers_addr: Int,
    center_norms_addr: Int,
    offsets_addr: Int,
    indices_addr: Int,
    list_data_addr: Int,
    n_rows: Int,
    dim: Int,
    n_lists: Int,
    centers: List[Float32],
    center_norms: List[Float32],
    offsets: List[Int32],
    list_indices: List[UInt32],
    list_data: List[Float32],
) raises:
    """The RUN half of `ivf_write_index_arrays`: a built index written to
    the five output buffers at their addresses."""
    var cp = f32_ptr(centers_addr)
    var np_ = f32_ptr(center_norms_addr)
    var op = i32_ptr(offsets_addr)
    var ip = i32_ptr(indices_addr)
    var lp = f32_ptr(list_data_addr)
    ivf_store_index_arrays(
        cp, np_, op, ip, lp, n_rows, dim, n_lists, centers, center_norms,
        offsets, list_indices, list_data,
    )


def ivf_write_extended_arrays_run(
    offsets_addr: Int,
    indices_addr: Int,
    list_data_addr: Int,
    labels_addr: Int,
    n_total: Int,
    dim: Int,
    n_lists: Int,
    offsets: List[Int32],
    list_indices: List[UInt32],
    list_data: List[Float32],
    new_labels: List[UInt32],
    n_new: Int,
) raises:
    """The RUN half of `ivf_write_extended_arrays`: the extended index
    written to the four output buffers at their addresses."""
    var op = i32_ptr(offsets_addr)
    var ip = i32_ptr(indices_addr)
    var lp = f32_ptr(list_data_addr)
    var bp = i32_ptr(labels_addr)
    ivf_store_extended_arrays(
        op, ip, lp, bp, n_total, dim, n_lists, offsets, list_indices,
        list_data, new_labels, n_new,
    )



# ===========================================================================
# THE DISTRIBUTED MERGE (lane/py-dn-ann, 2026-09-28). `DistributedIVFIndex.
# search` merged its shards' candidates in Python, one `Array` scalar index
# per candidate (about 40 s per 1M queries at two shards and k = 8). This is
# the same merge statement for statement, host code in every IVF binding:
#
#   for each query row: for each shard in order, its count `n` (0 <= n <=
#   the shard's row count, else status 1), then its first min(k, n)
#   candidates, each local id in [0, the shard's row count) (else status 2)
#   mapped to its ORIGINAL id; the candidates ordered by (float32 distance,
#   original id), where -0.0 equals +0.0 as a float compare says, and the
#   first k written; a row whose total count is below k short-fills its
#   remaining slots with (+inf, -1), the plain search's kOutOfBoundsRecord. A NaN
#   distance is refused (status 4): Python's tuple sort had no defined
#   order for it. Original ids are distinct across disjoint shards, so the
#   order is total and any correct sort gives the same answer.
#
#   addrs   0 dist_out (m * k float32), 1 ids_out (m * k int32),
#           2 counts_out (m int32), then per shard s: 3 + 4s distances
#           (m * k float32), 4 + 4s local ids (m * k int32), 5 + 4s counts
#           (m int32), 6 + 4s the local-to-original map (int32)
#   params  0 shards, 1 m, 2 k, then 3 + s the shard's row count
#   returns [status, row] (status 0: merged)
# ===========================================================================


def ivf_merge_shards_host_run(aa: Int, pp: Int) raises -> Int:
    """The host merge over the PI_MERGE_SHARDS layout of
    core/parallel_entry_table.mojo (the addrs/params above as plain words;
    spans checked by the caller): the status. The former Python door
    `ivf_merge_shards` had no caller left once the distributed IVF owner read
    the parallel IVF table (lane calltable-parallel-archive, 2026-11-11)."""
    var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
    var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
    var shards = p[0]
    var m = p[1]
    var k = p[2]
    if shards < 1 or m < 0 or k < 1:
        raise Error("ivf_merge_shards: need at least one shard and k >= 1")
    var sd = List[Int]()
    var si = List[Int]()
    var sc = List[Int]()
    var sm = List[Int]()
    var sn = List[Int]()
    for s in range(shards):  # small-loop(shards: four address words and one row count per shard)
        sd.append(a[3 + 4 * s])
        si.append(a[4 + 4 * s])
        sc.append(a[5 + 4 * s])
        sm.append(a[6 + 4 * s])
        sn.append(p[3 + s])
    var merged = ivf_merge_shards_host(f32_ptr(a[0]), i32_ptr(a[1]), i32_ptr(a[2]), sd, si, sc, sm, sn, shards, m, k)
    return merged[0]


# ===========================================================================
# ivf_shard_plan (pyglue-sweep, 2026-10-03): `DistributedIVFIndex.from_index`'s
# validation and row split, which was Python over every stored row (a sort of
# the original ids, a dict of local ids, a list per shard). Shard p holds the
# stored rows [lo_p, hi_p), lo_p = p * n // shards.
#   addrs   0 indices (n int32, read: original row id of each stored row)
#           1 offsets (n_lists + 1 int32, read)
#           2 shard_offsets_out (shards * (n_lists + 1) int32): offsets
#             clipped to the shard, max(0, min(o, hi) - lo)
#           3 local_out (n int32): each stored row's rank among its shard's
#             original ids (the shard's list_indices_)
#           4 map_out (n int32): shard p's original ids ascending at [lo_p, hi_p)
#   params  0 n, 1 n_lists, 2 shards
#   returns status: 0 planned, 1 the ids are not a permutation of 0..n-1,
#           2 the offsets are not 0 .. n nondecreasing
# ===========================================================================


def ivf_shard_plan_host_run(ind_addr: Int, off_addr: Int, soff_addr: Int, loc_addr: Int, map_addr: Int,
                            n: Int, n_lists: Int, shards: Int) raises -> Int:
    """The host plan (contract above). The former Python door
    `ivf_shard_plan` had no caller left once the distributed IVF owner read
    the parallel IVF table (lane calltable-parallel-archive, 2026-11-11)."""
    if n < 0 or n_lists < 1 or shards < 1:
        raise Error("ivf_shard_plan: needs 5 addresses, n >= 0, n_lists >= 1 and shards >= 1")
    return ivf_shard_plan_host(i32_ptr(ind_addr), i32_ptr(off_addr), i32_ptr(soff_addr), i32_ptr(loc_addr),
                               i32_ptr(map_addr), n, n_lists, shards)
