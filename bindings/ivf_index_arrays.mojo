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

The index arrays are refused by name unless `ivf_validate_index_arrays`
admits them, before any search statement runs.
"""
from std.python import PythonObject

from bindings.hostptr import f32_ptr, i32_ptr, read_f32, read_i32
from ivf.impl.neighbors.ivf_flat.ivf_flat_index import ivf_validate_index_arrays

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
    addrs: PythonObject, params: PythonObject, what: String
) raises -> IvfIndexArrays:
    """`ivf_flat_search`'s addresses 0 to 4 and params 0 to 3, read and
    admitted (module docstring)."""
    if len(addrs) != 9:
        raise Error(
            what + ": addrs must contain 9 addresses (centers, center_norms,"
            " offsets, indices, list_data, queries, dist_out, idx_out,"
            " cand_out), got " + String(len(addrs))
        )
    if len(params) != 7:
        raise Error(
            what + ": params must contain 7 values (n, dim, n_lists, metric,"
            " m, k, n_probes), got " + String(len(params))
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
    var ids = read_i32(Int(py=addrs[3]), n)
    var list_data = read_f32(Int(py=addrs[4]), n * dim)
    var list_indices = List[UInt32](capacity=n)
    for s in range(n):
        var id = Int(ids[s])
        if id < 0:
            raise Error(
                what + ": slot " + String(s) + " carries a negative row id "
                + String(id)
            )
        list_indices.append(UInt32(id))
    ivf_validate_index_arrays(
        n_lists, dim, n, metric, centers, center_norms, offsets, list_indices,
        list_data,
    )
    return IvfIndexArrays(
        n_lists, dim, n, metric, centers^, center_norms^, offsets^,
        list_indices^, list_data^,
    )


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
    for i in range(n_rows * dim):
        lp.unsafe_store(i, list_data[i])
