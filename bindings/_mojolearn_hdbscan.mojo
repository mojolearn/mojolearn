# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPython boundary for the HDBSCAN lane (workstream D, 2026-09-14).

A separate extension module, for `bindings/_mojolearn_gp.mojo`'s reason.
`hdbscan/estimator.mojo::hdbscan_fit_host` is reached and nothing is
re-decided: its header lists every refusal and where each is raised
(metric, build algorithm, selection method and epsilon, min_samples,
min_cluster_size, alpha, the row bounds, a non-finite cell, and
`probabilities_`, DEVIATION 1610). This file refuses a null address and a
list of the wrong length and nothing else.

THE ABI IS THE GP'S: two length-checked lists, orders written out below
and mirrored in `python/mojolearn/hdbscan.py`. The fit writes the labels,
the core distances and four integers (`info_ptr`) into caller-sized
buffers and returns the cluster count. There is no model to carry: HDBSCAN
is transductive, as `DBSCAN` is, and the harness's model column is
`n/a:no-save`.

THE GIL is released around the device call, and nothing inside the
`GILReleased` block touches a `PythonObject`.
"""

from std.os import abort
from bindings.hostptr import f32_ptr, i32_ptr
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from max.gpu.host import DeviceContext

from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from hdbscan.estimator import hdbscan_fit_host


def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    return f32_ptr(addr)


def _i32_ptr(addr: Int) raises -> MutPointer[Int32, MutUntrackedOrigin]:
    return i32_ptr(addr)


def hdbscan_numeric_mode_binding() raises -> PythonObject:
    """THE BUILD'S TIER as the `NUMERIC_*` code: 0 FAST, 1 IDENTICAL, 2
    DETERMINISTIC."""
    return PythonObject(GLOBAL_NUMERIC_MODE)


def hdbscan_vendor_binding() raises -> PythonObject:
    """'metal', 'cuda', 'hip' or 'none', from `checks/vendor.mojo`."""
    return PythonObject(String(COMPILED_VENDOR))


def hdbscan_rows_parallel_available() raises -> PythonObject:
    """1: the core-distance k-NN reads MOJOLEARN_NEIGHBORS_DEVICE_COUNT
    (neighbors/impl/multi_gpu.mojo) and the dense pairwise distance matrix
    reads MOJOLEARN_HIERARCHY_DEVICE_COUNT (hierarchy/impl/cluster/detail/
    multi_gpu.mojo); parallel_classical.fit_hdbscan sets both in its worker."""
    return PythonObject(1)


def hdbscan_parallel_available() raises -> PythonObject:
    return PythonObject(1)


def _hdbscan_fit_run(
    xp: MutPointer[Float32, MutUntrackedOrigin],
    lp: MutPointer[Int32, MutUntrackedOrigin],
    cp: MutPointer[Float32, MutUntrackedOrigin],
    ip: MutPointer[Int32, MutUntrackedOrigin],
    n: Int,
    d: Int,
    min_samples: Int,
    min_cluster_size: Int,
    max_cluster_size: Int,
    alpha: Float32,
    allow_single_cluster: Bool,
    cluster_selection_method: Int,
    cluster_selection_epsilon: Float32,
    metric: Int,
) raises -> Int:
    var ctx = DeviceContext()
    var n_clusters = hdbscan_fit_host(
        ctx,
        xp,
        lp,
        cp,
        ip,
        n,
        d,
        min_samples,
        min_cluster_size,
        max_cluster_size,
        alpha,
        allow_single_cluster,
        cluster_selection_method,
        cluster_selection_epsilon,
        metric,
    )
    ctx.synchronize()
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^
    return n_clusters


def hdbscan_fit_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`HDBSCAN(...).fit(X)` (`hdbscan_fit_host`, cuML `runner.h:152-234`).
    Returns the cluster count.

    `addrs`, in this exact order:

        0  x                n * d float32, row-major, read
        1  labels_out       n int32, WRITTEN (0 .. n_clusters-1, -1 noise)
        2  core_dists_out   n float32, WRITTEN (distance to the
                             min_samples-th neighbor excluding self)
        3  info_out         4 int32, WRITTEN: n_clusters, n_outliers,
                             Boruvka round count, n_condensed_clusters

    `params`, in this exact order:

        0  n
        1  d
        2  min_samples
        3  min_cluster_size
        4  max_cluster_size            (0 means no cap)
        5  alpha                       (float)
        6  allow_single_cluster        (0/1)
        7  cluster_selection_method    (0 eom, 1 leaf)
        8  cluster_selection_epsilon   (float; only 0.0 is implemented,
                                        refused by name otherwise)
        9  metric                      (1 L2SqrtExpanded, the only one)
    """
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
    var xp = _f32_ptr(Int(py=addrs[0]))
    var lp = _i32_ptr(Int(py=addrs[1]))
    var cp = _f32_ptr(Int(py=addrs[2]))
    var ip = _i32_ptr(Int(py=addrs[3]))
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
    var n_clusters = 0
    with GILReleased(Python()):
        n_clusters = _hdbscan_fit_run(
            xp,
            lp,
            cp,
            ip,
            n,
            d,
            min_samples,
            min_cluster_size,
            max_cluster_size,
            alpha,
            allow_single,
            method,
            eps,
            metric,
        )
    return PythonObject(n_clusters)


@export
def PyInit__mojolearn_hdbscan() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_hdbscan")
        m.def_function[hdbscan_parallel_available]("hdbscan_parallel_available")
        m.def_function[hdbscan_rows_parallel_available]("hdbscan_rows_parallel_available")
        m.def_function[hdbscan_vendor_binding]("hdbscan_vendor")
        m.def_function[hdbscan_numeric_mode_binding]("hdbscan_numeric_mode")
        m.def_function[hdbscan_fit_binding]("hdbscan_fit")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_hdbscan: ", e))
