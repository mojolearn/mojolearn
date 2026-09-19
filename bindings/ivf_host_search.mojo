# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`ivf_flat_search` on the host, shared by two bindings
(lane/inference-embedding-ivf-cholesky, 2026-09-15).

`bindings/_mojolearn_ivf_host.mojo` (the internal reference binding, build
included) and `bindings/_mojolearn_ivf_search_host.mojo` (the inference
binding the wheels ship, no build) both register `ivf_flat_search` from
here, so the two binaries answer a saved index through the same source.
Not a binding itself: it registers nothing, and the host surface tests glob
only `_mojolearn_*_host.mojo`. The contract is
`bindings/ivf_index_arrays.mojo`'s; the arithmetic is
`ivf/host/ivf_host.mojo::host_ivf_search`. Since stage 2 of the same lane
both also register `ivf_flat_extend` from here (`host_ivf_extend`).

Since lane/laneless-public-classes (2026-09-19) both also register
`ivf_flat_partial_search` and `ivf_finalize_distances`, the GPU binding's
two remaining search names (`bindings/_mojolearn_ivf.mojo:275,280`). They
are what `python/mojolearn/parallel_ivf.py::DistributedIVFIndex` calls in
its workers, so with them the driver's Python partition, its local-id maps
and its global merge run on a CPU-only install exactly as they run on a
GPU, and the `par-ivf` lane of tools/identity_break.py has a CPU route.
"""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from bindings.hostptr import f32_ptr, read_f32
from bindings.ivf_index_arrays import (
    ivf_extend_count,
    ivf_read_index_arrays,
    ivf_write_extended_arrays,
    ivf_search_extents,
    ivf_write_search_result,
)
from ivf.host.ivf_host import IvfHostIndex, host_ivf_extend, host_ivf_search
from ivf.impl.neighbors.ivf_common import postprocess_distances


def _ivf_host_search_arrays(
    addrs: PythonObject, params: PythonObject, partial_storage: Bool
) raises -> PythonObject:
    """`ivf_flat::search` over a built index, restated on the host. Returns
    0. See `bindings/ivf_index_arrays.mojo` for the lists. `partial_storage`
    is the disjoint-shard arm `host_ivf_search`'s docstring describes; the
    GPU binding routes its two names through the same one statement
    (`bindings/_mojolearn_ivf.mojo::_ivf_search_arrays`)."""
    var arrays = ivf_read_index_arrays(
        addrs, params, String("ivf_flat_search"), partial_storage=partial_storage
    )
    var ext = ivf_search_extents(params)
    var m = ext[0]
    var k = ext[1]
    var n_probes = ext[2]
    var queries = read_f32(Int(py=addrs[5]), m * arrays.dim)
    var index = IvfHostIndex(
        arrays.n_lists, arrays.dim, arrays.n_rows, arrays.metric,
        arrays.centers.copy(), arrays.center_norms.copy(), arrays.offsets.copy(),
        arrays.list_indices.copy(), arrays.list_data.copy(),
    )
    var dist = List[Float32]()
    var idx = List[UInt32]()
    var cand = List[Int32]()
    with GILReleased(Python()):
        var r = host_ivf_search(index, queries, m, k, n_probes, partial_storage)
        dist = r.distances.copy()
        idx = r.indices.copy()
        cand = r.n_candidates.copy()
    ivf_write_search_result(addrs, dist, idx, cand, m, k)
    return PythonObject(0)


def ivf_flat_search_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`ivf_flat::search` over a whole built index."""
    return _ivf_host_search_arrays(addrs, params, False)


def ivf_flat_partial_search_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """Disjoint storage with global centers/probes; squared output, valid
    count=min(k,candidates). The GPU binding's docstring, same contract."""
    return _ivf_host_search_arrays(addrs, params, True)


def ivf_finalize_distances_binding(
    address: PythonObject, count: PythonObject, metric: PythonObject
) raises -> PythonObject:
    """`postprocess_distances` applied in place to `count` float32 at
    `address`: the Euclidean root the partial search withheld, taken ONCE
    by the driver after the global order is fixed on the squared keys.
    `bindings/_mojolearn_ivf.mojo:280` is the original, statement for
    statement."""
    var n = Int(py=count)
    if n < 0:
        raise Error("distance count must be nonnegative")
    var distances = read_f32(Int(py=address), n)
    postprocess_distances(distances, Int(py=metric))
    var dst = f32_ptr(Int(py=address))
    for i in range(n):
        dst.unsafe_store(i, distances[i])
    return PythonObject(0)


def ivf_flat_extend_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`ivf_flat::extend` over a built index, restated on the host. Returns
    0. See `bindings/ivf_index_arrays.mojo` for the lists."""
    var arrays = ivf_read_index_arrays(addrs, params, String("ivf_flat_extend"), 10, 5)
    var n_new = ivf_extend_count(params, arrays.n_rows, arrays.dim)
    var new_x = read_f32(Int(py=addrs[5]), n_new * arrays.dim)
    var index = IvfHostIndex(
        arrays.n_lists, arrays.dim, arrays.n_rows, arrays.metric,
        arrays.centers.copy(), arrays.center_norms.copy(), arrays.offsets.copy(),
        arrays.list_indices.copy(), arrays.list_data.copy(),
    )
    var labels = List[UInt32]()
    var out = host_ivf_extend(index, new_x, n_new, labels)
    ivf_write_extended_arrays(
        addrs, out.n_rows, out.dim, out.n_lists, out.offsets, out.list_indices,
        out.list_data, labels, n_new,
    )
    return PythonObject(0)
