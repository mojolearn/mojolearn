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
`ivf/host/ivf_host.mojo::host_ivf_search`.
"""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from bindings.hostptr import read_f32
from bindings.ivf_index_arrays import (
    ivf_read_index_arrays,
    ivf_search_extents,
    ivf_write_search_result,
)
from ivf.host.ivf_host import IvfHostIndex, host_ivf_search


def ivf_flat_search_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`ivf_flat::search` over a built index, restated on the host. Returns
    0. See `bindings/ivf_index_arrays.mojo` for the lists."""
    var arrays = ivf_read_index_arrays(addrs, params, String("ivf_flat_search"))
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
        var r = host_ivf_search(index, queries, m, k, n_probes)
        dist = r.distances.copy()
        idx = r.indices.copy()
        cand = r.n_candidates.copy()
    ivf_write_search_result(addrs, dist, idx, cand, m, k)
    return PythonObject(0)
