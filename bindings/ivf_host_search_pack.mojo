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
# The PACK half of the former `bindings/ivf_host_search.mojo` (fence-flip split, 2026-11-12);
# the UNPACK half is `bindings/ivf_host_search_unpack.mojo`.
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from core.hostptr import read_f32
from bindings.ivf_index_arrays_unpack import (
    ivf_read_resident_filter,
    ivf_write_resident_result,
    ivf_extend_count,
    ivf_read_index_arrays,
    ivf_read_search_filter,
    ivf_search_extents,
    ivf_write_search_result,
)
from ivf.index_arrays_impl import ivf_finalize_distances_host
from ivf.ivf_host_search_run import (
    ivf_host_extend_run,
    ivf_host_index_of,
    ivf_host_register,
    ivf_host_release,
    ivf_host_search_run,
    _host_ivf_view,
)


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
    ivf_finalize_distances_host(Int(py=address), n, Int(py=metric))
    return PythonObject(0)


def ivf_flat_extend_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`ivf_flat::extend` over a built index, restated on the host. Returns
    0. See `bindings/ivf_index_arrays.mojo` for the lists."""
    var arrays = ivf_read_index_arrays(addrs, params, String("ivf_flat_extend"), 10, 5)
    var n_new = ivf_extend_count(params, arrays.n_rows, arrays.dim)
    ivf_host_extend_run(
        arrays^, Int(py=addrs[5]), n_new, Int(py=addrs[6]), Int(py=addrs[7]), Int(py=addrs[8]), Int(py=addrs[9])
    )
    return PythonObject(0)


# ===========================================================================
# THE RESIDENT DOORS ON THE HOST (lane/py-dn-ann, 2026-09-28): the GPU
# binding's three names (ivf/resident.mojo), the contract in
# bindings/ivf_index_arrays.mojo. A host search read the five arrays into
# lists, admitted them and copied them into an `IvfHostIndex` on every call;
# a handle admits and copies once and later searches read it, so the same
# `host_ivf_search` runs over the same bytes.
# ===========================================================================


def ivf_flat_index_prepare_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    var partial = Int(py=params[4]) != 0
    var arrays = ivf_read_index_arrays(
        addrs, params, String("ivf_flat_search"), 5, 5, partial_storage=partial
    )
    return PythonObject(ivf_host_register(arrays^, partial))


def ivf_flat_index_release_binding(handle: PythonObject) raises -> PythonObject:
    ivf_host_release(Int(py=handle))
    return PythonObject(0)
