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
`ivf_flat_build_and_search` with the SAME address and `params` lists, plus
the read-backs `ivf_numeric_mode` (1) and `ivf_vendor` ("cpu"), so
`python/mojolearn/_ivf_impl.py` runs unchanged on a CPU-only install through
`_backend._HOST_MODULES` (`"_mojolearn_ivf": "_mojolearn_ivf_host"`). The
GPU binding exports nothing else.
"""
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from bindings.hostptr import copy_f32, f32_ptr, i32_ptr, read_f32
from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN, column_name
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from ivf.host.ivf_host import IVF_HOST_SABOTAGE, host_ivf_build_and_search


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
    descending on purpose (-D MOJOLEARN_HOST_SABOTAGE=1, the gate's negative
    control)."""
    return PythonObject(IVF_HOST_SABOTAGE)


# The GPU binding's names, same contract.


def ivf_numeric_mode_binding() raises -> PythonObject:
    """THE BUILD'S TIER as the `NUMERIC_*` code: always 1 here."""
    return PythonObject(GLOBAL_NUMERIC_MODE)


def ivf_vendor_binding() raises -> PythonObject:
    """"cpu", the answer `_backend.read_vendor` expects from a host binding."""
    return PythonObject(String("cpu"))


def ivf_flat_build_and_search_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`ivf_flat::build` then `ivf_flat::search`, restated. Returns 0.

    `addrs` = [x (n * dim f32, read), queries (m * dim f32, read), dist_out
    (m * k f32, WRITTEN), idx_out (m * k i32, WRITTEN), cand_out (m i32,
    WRITTEN)]; `params` = [n, dim, m, k, n_lists, n_probes, kmeans_n_iters,
    metric (0 L2Expanded, 1 L2SqrtExpanded), seed]. The GPU binding's order."""
    if len(addrs) != 5:
        raise Error(
            "ivf_flat_build_and_search: addrs must contain 5 addresses (x,"
            " queries, dist_out, idx_out, cand_out), got "
            + String(len(addrs))
        )
    if len(params) != 9:
        raise Error(
            "ivf_flat_build_and_search: params must contain 9 values (n, dim,"
            " m, k, n_lists, n_probes, kmeans_n_iters, metric, seed), got "
            + String(len(params))
        )
    var n = Int(py=params[0])
    var dim = Int(py=params[1])
    var m = Int(py=params[2])
    var k = Int(py=params[3])
    var n_lists = Int(py=params[4])
    var n_probes = Int(py=params[5])
    var kmeans_n_iters = Int(py=params[6])
    var metric = Int(py=params[7])
    var seed = UInt64(Int(py=params[8]))
    var x = read_f32(Int(py=addrs[0]), max(0, n * dim))
    var queries = read_f32(Int(py=addrs[1]), max(0, m * dim))
    var dp = f32_ptr(Int(py=addrs[2]))
    var ip = i32_ptr(Int(py=addrs[3]))
    var cp = i32_ptr(Int(py=addrs[4]))
    with GILReleased(Python()):
        var r = host_ivf_build_and_search(
            x, n, dim, n_lists, queries, m, k, n_probes, kmeans_n_iters, metric, seed,
        )
        copy_f32(r.distances.unsafe_ptr(), dp, m * k)
        for i in range(m * k):
            ip.unsafe_store(i, Int32(Int(r.indices[i])))
        for i in range(m):
            cp.unsafe_store(i, r.n_candidates[i])
    return PythonObject(0)


@export
def PyInit__mojolearn_ivf_host() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_ivf_host")
        m.def_function[ivf_host_numeric_mode_binding]("ivf_host_numeric_mode")
        m.def_function[ivf_host_vendor_binding]("ivf_host_vendor")
        m.def_function[ivf_host_column_binding]("ivf_host_column")
        m.def_function[ivf_host_sabotage_binding]("ivf_host_sabotage")
        m.def_function[ivf_vendor_binding]("ivf_vendor")
        m.def_function[ivf_numeric_mode_binding]("ivf_numeric_mode")
        m.def_function[ivf_flat_build_and_search_binding]("ivf_flat_build_and_search")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_ivf_host: ", e))
