# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for the `_mojolearn_ivf` family: IVFIndex, cuVS `ivf_flat`
build plus search (lane/cpu-training-embedding-ivf, 2026-09-15; the ivf and
ivf-euclidean lanes of tools/identity_break.py).

HOST ONLY. No DeviceContext, no kernel launch, no GPU. The arithmetic is
`ivf/host/ivf_host.mojo`, `ivf/estimator.mojo::ivf_flat_build_and_search_host`
with every device launch restated on the host (the quantizer through
`cluster/host/kmeans_oracle.mojo`); that file's header names every original.

THE EXPORTED NAMES ARE THE GPU BINDING'S NAMES (`bindings/_mojolearn_ivf.mojo`):
`ivf_flat_build` and `ivf_flat_search` with the SAME address and `params`
lists, plus the read-backs `ivf_numeric_mode` (1) and `ivf_vendor` ("cpu"), so
`python/mojolearn/_ivf_impl.py` runs unchanged on a CPU-only install through
`_backend._HOST_MODULES` (`"_mojolearn_ivf": "_mojolearn_ivf_host"`). The
GPU binding exports nothing else.
"""
from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

from core.hostptr import read_f32
from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN, column_name
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from bindings.ivf_index_arrays import ivf_merge_shards_host_binding, ivf_shard_plan_host_binding
from bindings.ivf_host_search import (
    ivf_flat_index_prepare_binding,
    ivf_flat_index_release_binding,
    ivf_flat_index_search_binding,
    ivf_finalize_distances_binding,
    ivf_flat_extend_binding,
    ivf_flat_partial_search_binding,
    ivf_flat_search_binding,
)
from bindings.ivf_index_arrays import ivf_build_extents, ivf_write_index_arrays
from ivf.host.ivf_host import IVF_HOST_SABOTAGE, host_ivf_build


def ivf_host_numeric_mode_binding() raises -> PythonObject:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL, (
        "ivf host: this binding restates the IDENTICAL arms only;"
        " bindings/build_host_family.sh passes -D MOJOLEARN_NUMERIC_IDENTICAL=1"
    )
    return PythonObject(GLOBAL_NUMERIC_MODE)


def ivf_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def ivf_host_column_binding() raises -> PythonObject:
    """`column_name(TARGET_COLUMN)`, the comptime assert's witness: "cpu".
    THE COLUMN IS THE CPU COLUMN, OR THIS DOES NOT BUILD; the assert lives
    in a function body PyInit registers, so it is compiled in every build.
    `bindings/build_host_family.sh` passes -D MOJOLEARN_COLUMN_CPU."""
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "ivf host: this binding compiles the CPU column only; pass"
        " -D MOJOLEARN_COLUMN_CPU (bindings/build_ivf_host.sh does)"
    )
    return PythonObject(column_name(TARGET_COLUMN))


def ivf_host_sabotage_binding() raises -> PythonObject:
    """Whether this binary walks every candidate distance's feature axis
    descending and moves the bits of every returned distance on purpose
    (-D MOJOLEARN_HOST_SABOTAGE=1, the gate's negative control)."""
    return PythonObject(IVF_HOST_SABOTAGE)


# The GPU binding's names, same contract.


def ivf_numeric_mode_binding() raises -> PythonObject:
    """THE BUILD'S TIER as the `NUMERIC_*` code: always 1 here."""
    return PythonObject(GLOBAL_NUMERIC_MODE)


def ivf_vendor_binding() raises -> PythonObject:
    """"cpu", the answer `_backend.read_vendor` expects from a host binding."""
    return PythonObject(String("cpu"))


# TOMBSTONE: ivf_flat_build_and_search (dead native entry: the retired one-call door; IVFIndex fits and searches through ivf_flat_build / ivf_flat_search) deleted 2026-10-26 by @@SHA@@; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_ivf.patch


def ivf_flat_build_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`ivf_flat::build` restated, the index written to the caller's five
    buffers (`bindings/ivf_index_arrays.mojo`; lane/inference-embedding-ivf-
    cholesky, 2026-09-15). Returns 0. Reference only: the shipped search
    binding carries no build."""
    var ext = ivf_build_extents(addrs, params, String("ivf_flat_build"))
    var n = ext[0]
    var dim = ext[1]
    var n_lists = ext[2]
    var x = read_f32(Int(py=addrs[0]), n * dim)
    var index = host_ivf_build(x, n, dim, n_lists, ext[3], ext[4], ext[5])
    ivf_write_index_arrays(
        addrs, index.n_rows, index.dim, index.n_lists, index.centers,
        index.center_norms, index.offsets, index.list_indices, index.list_data,
    )
    return PythonObject(0)


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_ivf_host.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).
def build__mojolearn_ivf_host() -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_ivf_host")
        m.def_function[ivf_host_numeric_mode_binding]("ivf_host_numeric_mode")
        m.def_function[ivf_host_vendor_binding]("ivf_host_vendor")
        m.def_function[ivf_host_column_binding]("ivf_host_column")
        m.def_function[ivf_host_sabotage_binding]("ivf_host_sabotage")
        m.def_function[ivf_vendor_binding]("ivf_vendor")
        m.def_function[ivf_numeric_mode_binding]("ivf_numeric_mode")
        # TOMBSTONE: ivf_flat_build_and_search (dead native entry: the retired one-call door; IVFIndex fits and searches through ivf_flat_build / ivf_flat_search) deleted 2026-10-26 by @@SHA@@; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_ivf.patch
        m.def_function[ivf_flat_build_binding]("ivf_flat_build")
        m.def_function[ivf_flat_search_binding]("ivf_flat_search")
        m.def_function[ivf_flat_extend_binding]("ivf_flat_extend")
        m.def_function[ivf_flat_partial_search_binding]("ivf_flat_partial_search")
        m.def_function[ivf_finalize_distances_binding]("ivf_finalize_distances")
        m.def_function[ivf_flat_index_prepare_binding]("ivf_flat_index_prepare")
        m.def_function[ivf_flat_index_search_binding]("ivf_flat_index_search")
        m.def_function[ivf_flat_index_release_binding]("ivf_flat_index_release")
        m.def_function[ivf_merge_shards_host_binding]("ivf_merge_shards")
        m.def_function[ivf_shard_plan_host_binding]("ivf_shard_plan")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_ivf_host: ", e))
