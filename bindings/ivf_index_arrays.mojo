# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The IVF-Flat index as five caller buffers, shared by three bindings
(lane/inference-embedding-ivf-cholesky, 2026-09-15).

`bindings/_mojolearn_ivf.mojo` (the GPU binding), `bindings/_mojolearn_ivf_host.mojo`
(the internal reference binding, build included) and
`bindings/_mojolearn_ivf_search_host.mojo` (the inference binding the wheels
ship, no build) read and write a built index through here, so the three
agree on one contract. Not a binding itself: it registers nothing, and the
host surface tests glob only `_mojolearn_*_host.mojo`. Host code only: no
device import.

THE CONTRACT, mirrored in `python/mojolearn/_ivf_impl.py`.

`ivf_flat_build(addrs, params)`:

    addrs   0 x                n * dim float32, read
            1 centers_out      n_lists * dim float32, WRITTEN
            2 center_norms_out n_lists float32, WRITTEN (squared)
            3 offsets_out      n_lists + 1 int32, WRITTEN
            4 indices_out      n int32, WRITTEN (original row ids)
            5 list_data_out    n * dim float32, WRITTEN
    params  0 n, 1 dim, 2 n_lists, 3 kmeans_n_iters, 4 metric, 5 seed

`ivf_flat_search(addrs, params)`:

    addrs   0 centers, 1 center_norms, 2 offsets, 3 indices, 4 list_data
            (read, the build's five outputs), 5 queries (m * dim float32,
            read), 6 dist_out (m * k float32), 7 idx_out (m * k int32),
            8 cand_out (m int32), WRITTEN
    params  0 n, 1 dim, 2 n_lists, 3 metric, 4 m, 5 k, 6 n_probes

`ivf_flat_extend(addrs, params)` (2026-09-15, stage 2 of the same lane):

    addrs   0 to 4 the index as for search (read), 5 new_x (n_new * dim
            float32, read), 6 offsets_out (n_lists + 1 int32), 7 indices_out
            (n + n_new int32), 8 list_data_out ((n + n_new) * dim float32),
            9 labels_out (n_new int32, the list each new row went to), WRITTEN
    params  0 n, 1 dim, 2 n_lists, 3 metric, 4 n_new

The centres and their norms do not change under extend and are not written.
The index arrays are refused by name unless `ivf_validate_index_arrays`
admits them, before any search or extend statement runs.
"""
from std.memory import memcpy
from std.python import Python, PythonObject

from bindings.hostptr import f32_ptr, i32_ptr, u32_ptr, read_f32, read_i32, list_u32
from ivf.impl.neighbors.ivf_flat.ivf_flat_index import ivf_validate_index_arrays
from x_ann.switches import ANN3_HOST_PASSES

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


def ivf_arrays_extent(value: PythonObject, name: String) raises -> Int:
    """A non-negative integer parameter within the row bound, by name."""
    var type_name = String(py=value.__class__.__name__)
    if type_name == "bool" or type_name == "bool_":
        raise Error("ivf_flat: " + name + " must be an integer, not a boolean")
    var v = Int(py=value)
    if v < 0 or v > IVF_ARRAYS_MAX_ROWS:
        raise Error(
            "ivf_flat: " + name + " must be in [0, 2^31), got " + String(v)
        )
    return v


def ivf_arrays_check_cells(a: Int, b: Int, what: String) raises:
    if a > 0 and b > IVF_ARRAYS_MAX_CELLS // a:
        raise Error("ivf_flat: " + what + " exceeds 2^40 cells")


def ivf_read_index_arrays(
    addrs: PythonObject,
    params: PythonObject,
    what: String,
    n_addrs: Int = 9,
    n_params: Int = 7,
    partial_storage: Bool = False,
    allow_filter: Bool = False,
) raises -> IvfIndexArrays:
    """Addresses 0 to 4 and params 0 to 3 of `ivf_flat_search` (9 addresses,
    7 params; with `allow_filter`, a 10th address is the sample filter,
    `ivf_read_search_filter`) or `ivf_flat_extend` (10 and 5), read and
    admitted (module docstring)."""
    if len(addrs) != n_addrs and not (allow_filter and len(addrs) == n_addrs + 1):
        raise Error(
            what + ": addrs must contain " + String(n_addrs) + " addresses"
            " (see bindings/ivf_index_arrays.mojo), got " + String(len(addrs))
        )
    if len(params) != n_params:
        raise Error(
            what + ": params must contain " + String(n_params) + " values"
            " (see bindings/ivf_index_arrays.mojo), got " + String(len(params))
        )
    var n = ivf_arrays_extent(params[0], String("n"))
    var dim = ivf_arrays_extent(params[1], String("dim"))
    var n_lists = ivf_arrays_extent(params[2], String("n_lists"))
    var metric = Int(py=params[3])
    ivf_arrays_check_cells(n, dim, String("n * dim"))
    ivf_arrays_check_cells(n_lists, dim, String("n_lists * dim"))
    if n < 1 or dim < 1 or n_lists < 1:
        raise Error(
            what + ": n, dim and n_lists must all be at least 1, got n="
            + String(n) + " dim=" + String(dim) + " n_lists=" + String(n_lists)
        )
    var centers = read_f32(Int(py=addrs[0]), n_lists * dim)
    var center_norms = read_f32(Int(py=addrs[1]), n_lists)
    var offsets = read_i32(Int(py=addrs[2]), n_lists + 1)
    var list_data = read_f32(Int(py=addrs[4]), n * dim)
    # cpu3-bindings: the int32 ids' bits in one memcpy (no per-slot host
    # loop). A negative id reads as >= 2^31 > n, so the id range check of
    # `ivf_validate_index_arrays` below refuses it by name.
    var list_indices = list_u32(u32_ptr(Int(py=addrs[3])), n)
    ivf_validate_index_arrays(
        n_lists, dim, n, metric, centers, center_norms, offsets, list_indices,
        list_data, partial_storage,
    )
    return IvfIndexArrays(
        n_lists, dim, n, metric, centers^, center_norms^, offsets^,
        list_indices^, list_data^,
    )


def ivf_read_search_filter(addrs: PythonObject, n_rows: Int) raises -> List[Int32]:
    """`ivf_flat_search`'s optional address 9: the sample filter, one int32
    per ORIGINAL row id (0 removes the row; DEVIATION 5863). Absent (9
    addresses) is no filter and returns an empty list."""
    if len(addrs) < 10:
        return List[Int32]()
    return read_i32(Int(py=addrs[9]), n_rows)


def ivf_search_extents(params: PythonObject) raises -> Tuple[Int, Int, Int]:
    """`ivf_flat_search`'s params 4 to 6: (m, k, n_probes)."""
    var m = ivf_arrays_extent(params[4], String("m"))
    var k = ivf_arrays_extent(params[5], String("k"))
    var n_probes = ivf_arrays_extent(params[6], String("n_probes"))
    ivf_arrays_check_cells(m, k, String("m * k"))
    return (m, k, n_probes)


def ivf_write_search_result(
    addrs: PythonObject,
    distances: List[Float32],
    indices: List[UInt32],
    n_candidates: List[Int32],
    m: Int,
    k: Int,
) raises:
    """Addresses 6 to 8: distances, ORIGINAL row ids as int32 (below 2^31
    by the row bound) and the candidate count per query."""
    var dp = f32_ptr(Int(py=addrs[6]))
    var ip = i32_ptr(Int(py=addrs[7]))
    var cp = i32_ptr(Int(py=addrs[8]))
    for i in range(m * k):
        dp.unsafe_store(i, distances[i])
        ip.unsafe_store(i, Int32(Int(indices[i])))
    for i in range(m):
        cp.unsafe_store(i, n_candidates[i])


def ivf_build_extents(
    addrs: PythonObject, params: PythonObject, what: String
) raises -> Tuple[Int, Int, Int, Int, Int, UInt64]:
    """`ivf_flat_build`'s params: (n, dim, n_lists, kmeans_n_iters, metric,
    seed), with the address count checked."""
    if len(addrs) != 6:
        raise Error(
            what + ": addrs must contain 6 addresses (x, centers_out,"
            " center_norms_out, offsets_out, indices_out, list_data_out), got "
            + String(len(addrs))
        )
    if len(params) != 6:
        raise Error(
            what + ": params must contain 6 values (n, dim, n_lists,"
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
    return (n, dim, n_lists, iters, metric, seed)


def ivf_write_index_arrays(
    addrs: PythonObject,
    n_rows: Int,
    dim: Int,
    n_lists: Int,
    centers: List[Float32],
    center_norms: List[Float32],
    offsets: List[Int32],
    list_indices: List[UInt32],
    list_data: List[Float32],
) raises:
    """Addresses 1 to 5 of `ivf_flat_build`, from a built index."""
    var cp = f32_ptr(Int(py=addrs[1]))
    var np_ = f32_ptr(Int(py=addrs[2]))
    var op = i32_ptr(Int(py=addrs[3]))
    var ip = i32_ptr(Int(py=addrs[4]))
    var lp = f32_ptr(Int(py=addrs[5]))
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


def ivf_extend_count(params: PythonObject, n_rows: Int, dim: Int) raises -> Int:
    """`ivf_flat_extend`'s param 4, `n_new`: at least one row, the extended
    row count below 2^31 (the ids cross as int32) and `(n + n_new) * dim`
    below 2^40."""
    var n_new = ivf_arrays_extent(params[4], String("n_new"))
    if n_new < 1:
        raise Error("ivf_flat_extend: n_new must be at least 1, got " + String(n_new))
    if n_new > IVF_ARRAYS_MAX_ROWS - n_rows:
        raise Error(
            "ivf_flat_extend: n + n_new = " + String(n_rows) + " + " + String(n_new)
            + " would carry an id past 2^31 - 1"
        )
    ivf_arrays_check_cells(n_rows + n_new, dim, String("(n + n_new) * dim"))
    return n_new


def ivf_write_extended_arrays(
    addrs: PythonObject,
    n_total: Int,
    dim: Int,
    n_lists: Int,
    offsets: List[Int32],
    list_indices: List[UInt32],
    list_data: List[Float32],
    new_labels: List[UInt32],
    n_new: Int,
) raises:
    """Addresses 6 to 9 of `ivf_flat_extend`, from the extended index."""
    var op = i32_ptr(Int(py=addrs[6]))
    var ip = i32_ptr(Int(py=addrs[7]))
    var lp = f32_ptr(Int(py=addrs[8]))
    var bp = i32_ptr(Int(py=addrs[9]))
    for i in range(n_lists + 1):
        op.unsafe_store(i, offsets[i])
    for i in range(n_total):
        ip.unsafe_store(i, Int32(Int(list_indices[i])))
    for i in range(n_total * dim):
        lp.unsafe_store(i, list_data[i])
    for j in range(n_new):
        bp.unsafe_store(j, Int32(Int(new_labels[j])))


# ===========================================================================
# THE RESIDENT DOORS' ADDRESSES (lane/py-dn-ann, 2026-09-28; ivf/resident.mojo
# on the GPU, bindings/ivf_host_search.mojo on the host).
#
# `ivf_flat_index_prepare(addrs, params) -> handle`:
#     addrs   0 to 4 the index as for search (read, admitted once)
#     params  0 n, 1 dim, 2 n_lists, 3 metric, 4 partial_storage (0 or 1)
# `ivf_flat_index_search(handle, addrs, params)`:
#     addrs   0 queries (m * dim float32, read), 1 dist_out (m * k float32),
#             2 idx_out (m * k int32), 3 cand_out (m int32), WRITTEN,
#             optional 4 the sample filter (n int32, read)
#     params  0 n, 1 dim, 2 n_lists, 3 metric (must be the handle's),
#             4 m, 5 k, 6 n_probes, 7 partial_storage (must be the handle's)
# `ivf_flat_index_release(handle)`.
# ===========================================================================


def ivf_read_resident_filter(addrs: PythonObject, n_rows: Int) raises -> List[Int32]:
    """`ivf_flat_index_search`'s optional address 4 (DEVIATION 5863); absent
    is no filter and returns an empty list."""
    if len(addrs) < 5:
        return List[Int32]()
    return read_i32(Int(py=addrs[4]), n_rows)


def ivf_write_resident_result(
    addrs: PythonObject,
    distances: List[Float32],
    indices: List[UInt32],
    n_candidates: List[Int32],
    m: Int,
    k: Int,
) raises:
    """`ivf_flat_index_search`'s addresses 1 to 3, as `ivf_write_search_result`."""
    var dp = f32_ptr(Int(py=addrs[1]))
    var ip = i32_ptr(Int(py=addrs[2]))
    var cp = i32_ptr(Int(py=addrs[3]))
    for i in range(m * k):
        dp.unsafe_store(i, distances[i])
        ip.unsafe_store(i, Int32(Int(indices[i])))
    for i in range(m):
        cp.unsafe_store(i, n_candidates[i])


# ===========================================================================
# THE DISTRIBUTED MERGE (lane/py-dn-ann, 2026-09-28). `DistributedIVFIndex.
# search` merged its shards' candidates in Python, one `Array` scalar index
# per candidate (about 40 s per 1M queries at two shards and k = 8). This is
# the same merge statement for statement, host code in every IVF binding:
#
#   for each query row: for each shard in order, its count `n` (0 <= n <=
#   the shard's row count, else status 1), then its first min(k, n)
#   candidates, each local id in [0, the shard's row count) (else status 2)
#   mapped to its ORIGINAL id; the total count at least k (else status 3);
#   the candidates ordered by (float32 distance, original id), where -0.0
#   equals +0.0 as a float compare says, and the first k written. A NaN
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


def ivf_merge_shards_host_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    var shards = Int(py=params[0])
    var m = Int(py=params[1])
    var k = Int(py=params[2])
    if shards < 1 or m < 0 or k < 1:
        raise Error("ivf_merge_shards: need at least one shard and k >= 1")
    if len(addrs) != 3 + 4 * shards or len(params) != 3 + shards:
        raise Error("ivf_merge_shards: address or parameter count differs from the shard count")
    var od = f32_ptr(Int(py=addrs[0]))
    var oi = i32_ptr(Int(py=addrs[1]))
    var oc = i32_ptr(Int(py=addrs[2]))
    var sd = List[Int]()
    var si = List[Int]()
    var sc = List[Int]()
    var sm = List[Int]()
    var sn = List[Int]()
    for s in range(shards):
        sd.append(Int(py=addrs[3 + 4 * s]))
        si.append(Int(py=addrs[4 + 4 * s]))
        sc.append(Int(py=addrs[5 + 4 * s]))
        sm.append(Int(py=addrs[6 + 4 * s]))
        sn.append(Int(py=params[3 + s]))
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
    var out = Python.list()
    out.append(PythonObject(status))
    out.append(PythonObject(bad_row))
    return out


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


def ivf_shard_plan_host_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    var n = Int(py=params[0])
    var n_lists = Int(py=params[1])
    var shards = Int(py=params[2])
    if n < 0 or n_lists < 1 or shards < 1 or len(addrs) != 5:
        raise Error("ivf_shard_plan: needs 5 addresses, n >= 0, n_lists >= 1 and shards >= 1")
    var ind = i32_ptr(Int(py=addrs[0]))
    var off = i32_ptr(Int(py=addrs[1]))
    var soff = i32_ptr(Int(py=addrs[2]))
    var loc = i32_ptr(Int(py=addrs[3]))
    var mp = i32_ptr(Int(py=addrs[4]))
    var L1 = n_lists + 1
    if Int(off.unsafe_load(0)) != 0 or Int(off.unsafe_load(n_lists)) != n:
        return PythonObject(2)
    for j in range(n_lists):
        if off.unsafe_load(j) > off.unsafe_load(j + 1):
            return PythonObject(2)
    # owner shard of every original id; a repeat or an id out of range is
    # not a permutation
    var owner = List[Int32](length=n, fill=Int32(-1))
    for p in range(shards):
        var lo = p * n // shards
        var hi = (p + 1) * n // shards
        for i in range(lo, hi):
            var v = Int(ind.unsafe_load(i))
            if v < 0 or v >= n or owner[v] != -1:
                return PythonObject(1)
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
    return PythonObject(0)
