# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for the `_mojolearn_hdbscan` family: HDBSCAN (CPU training for
the workstream D estimators, 2026-09-15; brief
docs/lanes/BRIEF_cpu_training_2026-09-13.md sections 1.1 and 3.2).

HOST ONLY. No DeviceContext, no kernel launch. The fit is
`hdbscan/host/hdbscan_host_oracle.mojo`, the device path of
`hdbscan/estimator.mojo::hdbscan_fit_host` restated on the host: the k-NN
core distances, the dense mutual reachability graph, the Boruvka MST with
its round count, the dendrogram, the condensed tree, the stabilities, the
selection and the labels. So `labels_`, `core_distances_` and the four
counts are meant to be the GPU columns' bytes.

THE EXPORTED NAMES ARE THE GPU BINDING'S NAMES, with the GPU binding's
address and params contract word for word (`bindings/_mojolearn_hdbscan.mojo`,
mirrored in `python/mojolearn/hdbscan.py`), so `HDBSCAN` runs unchanged on a
CPU-only install through `_backend._HOST_MODULES`
(`"_mojolearn_hdbscan": "_mojolearn_hdbscan_host"`): `hdbscan_fit`
(4 addresses, 10 params), `hdbscan_vendor` answering "cpu" and
`hdbscan_numeric_mode`. ABSENT, and so refused BY NAME through
`_HostBinding`: `hdbscan_rows_parallel_available`, the multi-GPU driver's
probe.

The sabotage arm (`hdbscan_host_sabotage`) is
`hdbscan/host/hdbscan_host_oracle.mojo::HDBH_HOST_SABOTAGE`
(`-D MOJOLEARN_HOST_SABOTAGE=1`): every core distance is read one neighbor
early.
"""
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from bindings.hostptr import f32_ptr, i32_ptr, read_f32
from checks.kernel_matrix import (
    COLUMN_CPU,
    TARGET_COLUMN,
    column_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE
from hdbscan.host.hdbscan_host_oracle import HDBH_HOST_SABOTAGE, hdbh_fit


def hdbscan_host_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def hdbscan_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def hdbscan_host_column_binding() raises -> PythonObject:
    """`column_name(TARGET_COLUMN)`, the comptime assert's witness: "cpu"."""
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "hdbscan host: this binding compiles the CPU column only; pass"
        " -D MOJOLEARN_COLUMN_CPU (bindings/build_hdbscan_host.sh does)"
    )
    return PythonObject(column_name(TARGET_COLUMN))


def hdbscan_host_sabotage_binding() raises -> PythonObject:
    """Whether this binary reads every core distance one neighbor early on
    purpose (-D MOJOLEARN_HOST_SABOTAGE=1, the gate's negative control)."""
    return PythonObject(HDBH_HOST_SABOTAGE)


def hdbscan_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def hdbscan_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def hdbscan_fit_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`HDBSCAN(...).fit(X)` on the host. Returns the cluster count.
    `addrs`: 0 x, 1 labels_out (n int32), 2 core_dists_out (n float32),
    3 info_out (4 int32: n_clusters, n_outliers, Boruvka rounds,
    n_condensed_clusters). `params`: 0 n, 1 d, 2 min_samples,
    3 min_cluster_size, 4 max_cluster_size, 5 alpha, 6 allow_single_cluster,
    7 cluster_selection_method, 8 cluster_selection_epsilon, 9 metric."""
    if len(addrs) != 4:
        raise Error(
            "hdbscan_fit: addrs must contain 4 addresses (x, labels_out,"
            " core_dists_out, info_out), got "
            + String(len(addrs))
        )
    if len(params) != 10:
        raise Error(
            "hdbscan_fit: params must contain 10 values (n, d, min_samples,"
            " min_cluster_size, max_cluster_size, alpha,"
            " allow_single_cluster, cluster_selection_method,"
            " cluster_selection_epsilon, metric), got "
            + String(len(params))
        )
    var lp = i32_ptr(Int(py=addrs[1]))
    var cp = f32_ptr(Int(py=addrs[2]))
    var ip = i32_ptr(Int(py=addrs[3]))
    var n = Int(py=params[0])
    var d = Int(py=params[1])
    var min_samples = Int(py=params[2])
    var min_cluster_size = Int(py=params[3])
    var max_cluster_size = Int(py=params[4])
    var alpha = Float32(Float64(py=params[5]))
    var allow_single = Int(py=params[6]) != 0
    var method = Int(py=params[7])
    var eps = Float32(Float64(py=params[8]))
    var metric = Int(py=params[9])
    var x = read_f32(Int(py=addrs[0]), max(0, n * d))
    var n_clusters = 0
    with GILReleased(Python()):
        var out = hdbh_fit(
            x, n, d, min_samples, min_cluster_size, max_cluster_size, alpha,
            allow_single, method, eps, metric,
        )
        for i in range(n):
            lp.unsafe_store(i, out.labels[i])
            cp.unsafe_store(i, out.core_dists[i])
        ip.unsafe_store(0, Int32(out.n_clusters))
        ip.unsafe_store(1, Int32(out.n_outliers))
        ip.unsafe_store(2, Int32(out.n_boruvka_rounds))
        ip.unsafe_store(3, Int32(out.n_condensed_clusters))
        n_clusters = out.n_clusters
        _ = out^
    _ = x^
    return PythonObject(n_clusters)


@export
def PyInit__mojolearn_hdbscan_host() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_hdbscan_host")
        module.def_function[hdbscan_host_numeric_mode_binding]("hdbscan_host_numeric_mode")
        module.def_function[hdbscan_host_vendor_binding]("hdbscan_host_vendor")
        module.def_function[hdbscan_host_column_binding]("hdbscan_host_column")
        module.def_function[hdbscan_host_sabotage_binding]("hdbscan_host_sabotage")
        module.def_function[hdbscan_vendor_binding]("hdbscan_vendor")
        module.def_function[hdbscan_numeric_mode_binding]("hdbscan_numeric_mode")
        module.def_function[hdbscan_fit_binding]("hdbscan_fit")
        return module.finalize()
    except e:
        abort(String("failed to create _mojolearn_hdbscan_host: ", e))
