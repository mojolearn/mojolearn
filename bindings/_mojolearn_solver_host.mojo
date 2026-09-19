# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for the `_mojolearn_solver` family, coordinate descent today
(the CPU training lane, phase 1, 2026-09-13; brief
docs/lanes/BRIEF_cpu_training_2026-09-13.md sections 1.1 lasso, elasticnet
and 3.2).

HOST ONLY. No DeviceContext, no kernel launch, no GPU. The fit is
`solver/host/cd_oracle.mojo::cd_oracle_fit` at `profile=True`, "cdFit on
the host, stage for stage", the arm `check_cd_device_equals_oracle` holds
the device to bit for bit under IDENTICAL; every row-length reduction in
it is `gemm_oracle_cell` at the contract's leaf size. The predict is
`linearRegH`'s IDENTICAL arm restated over the same oracle
(`solver/impl/functions/linear_reg.mojo:62-63` is `identical_gemm(pred, x,
coef, n_rows, 1, n_cols, OP_TN)`, so here `gemm_oracle(x, coef, OP_TN,
n_rows, 1, n_cols)`, then `add_scalar_kernel`'s `ftz(v + s)` when the
intercept is not zero). The guards are the GPU entry's, in the GPU entry's
order and words (`solver/estimator.mojo::cd_fit_host` then
`solver/impl/cd.mojo::cd_fit_traced`; `cd_predict_host` then
`cd_predict`), restated because that file also defines the kernels.

THE EXPORTED NAMES ARE THE GPU BINDING'S NAMES for the fits this covers, so
`python/mojolearn/_solver_impl.py::ElasticNet` and `Lasso` run unchanged on
a CPU-only install through `_backend._HOST_MODULES` (`"_mojolearn_solver":
"_mojolearn_solver_host"`): `cd_fit` and `cd_predict` with the SAME address
contract (`x` COLUMN-MAJOR `n_rows x n_cols`; the nine-value and
three-value params lists mirrored word for word in `_solver_impl.py`), and
`solver_vendor` answering "cpu".

`linkage_fit` (agglomerative, 2026-09-14; brief section 1.1 agglomerative)
is `hierarchy/checks/linkage_oracle.mojo`'s whole fit under the GPU
binding's address contract: `host_pinned_distance_matrix` (the IDENTICAL
tile's arithmetic, `core/row_norms` fold then the `-2 dot + (n_i + n_j)`
epilogue through `identical_mul_add` / `ftz` / `identical_sqrt`),
`host_kruskal` (Kruskal under the SAME total order `(weight_order_key, lo,
hi)` the device's Boruvka uses; the MST is unique under a total order, so
the edge set is the device's), `host_dendrogram` (the `children` rows over
a union-find) and `host_extract_flattened_clusters` (cuVS's cut, serial).
The guards are the device path's in the device path's order and words
(`linkage_fit_host`, `cuvs single_linkage`, `get_distance_graph`,
`pairwise_distances`). ONE ATTRIBUTE IS DEVICE-ONLY: `info[0]`, the
Boruvka round count `n_boruvka_rounds_`, is a pass count of an algorithm
the host does not run, so this binding writes -1 there; no identity_break
cell hashes it (the train column is `labels_`), and `_hierarchy_impl.py`
publishes what it reads. `info[1]`, `n_connected_components`, is 1, the
literal `single_linkage.mojo` returns on the pairwise arm.
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
from checks.numerics import GLOBAL_NUMERIC_MODE, ftz
from gemm.host.identical_gemm import (
    GEMM_ORACLE_HOST_SABOTAGE,
    OP_TN,
    gemm_oracle,
)
from hierarchy.checks.linkage_oracle import (
    host_dendrogram,
    host_extract_flattened_clusters,
    host_kruskal,
    host_pinned_distance_matrix,
)
from hierarchy.impl.cluster.detail.connectivities import (
    DISTANCE_L2_EXPANDED,
    DISTANCE_L2_SQRT_EXPANDED,
    PAIRWISE_MAX_ROWS,
)
from solver.host.cd_oracle import cd_oracle_fit


def _index(value: PythonObject) raises -> Int:
    var type_name = String(py=value.__class__.__name__)
    if type_name == "bool" or type_name == "bool_":
        raise Error("solver host: integers expected, not booleans")
    var operator_module = Python.import_module("operator")
    return Int(py=operator_module.index(value))


def solver_host_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def solver_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def solver_host_column_binding() raises -> PythonObject:
    """`column_name(TARGET_COLUMN)`, the comptime assert's witness: "cpu".
    THE COLUMN IS THE CPU COLUMN, OR THIS DOES NOT BUILD; the assert lives
    in a function body PyInit registers, so it is compiled in every build.
    `bindings/build_solver_host.sh` passes -D MOJOLEARN_COLUMN_CPU."""
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "solver host: this binding compiles the CPU column only; pass"
        " -D MOJOLEARN_COLUMN_CPU (bindings/build_solver_host.sh does)"
    )
    return PythonObject(column_name(TARGET_COLUMN))


# There is no `solver_host_detected_column` read-back, for the reason 8d16ce2f
# removed it from the forest and byte LM host bindings: the detected column
# folds to the GPU of the machine that ran the build, so its name would land
# in the vendor-neutral binary. The comptime assert above is the check.


def solver_host_sabotage_binding() raises -> PythonObject:
    """Whether this binary walks every GEMM leaf descending on purpose
    (-D MOJOLEARN_HOST_SABOTAGE=1, the gate's negative control; every
    reduction of the CD oracle is a gemm_oracle_cell, so the arm reaches
    this family through gemm/host/gemm_oracle.mojo)."""
    return PythonObject(GEMM_ORACLE_HOST_SABOTAGE)


# The GPU binding's names, same contract.


def solver_vendor_binding() raises -> PythonObject:
    """"cpu". On a CPU-only install `_backend.vendor()` is "cpu" and the
    read-back cross-check expects that string from every host binding."""
    return PythonObject(String("cpu"))


def cd_fit_binding(
    x_addr: PythonObject,
    y_addr: PythonObject,
    coef_addr: PythonObject,
    info_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`cdFit` for Lasso / ElasticNet on the host by `cd_oracle_fit`.
    Writes `n_cols` float32 coefficients to `coef_addr` and the intercept
    to `info_addr[0]`. Returns `n_iter`, the epochs run.

    `x_addr` is COLUMN-MAJOR `n_rows x n_cols` float32. `params` is, in
    this exact order (mirrored in `python/mojolearn/_solver_impl.py` and in
    the GPU binding):

        0  n_rows
        1  n_cols
        2  fit_intercept      (0/1)
        3  max_iter           (cdFit's `epochs`)
        4  alpha              (float)
        5  l1_ratio           (float; Lasso passes 1.0)
        6  tol                (float)
        7  shuffle            (0/1; 1 is REFUSED BY NAME, as on the device)
        8  has_sample_weight  (0/1; 1 is REFUSED BY NAME, as on the device)
    """
    if len(params) != 9:
        raise Error(
            "cd_fit: params must contain 9 values, got " + String(len(params))
        )
    var x_address = _index(x_addr)
    var y_address = _index(y_addr)
    var cp = f32_ptr(_index(coef_addr))
    var ip = f32_ptr(_index(info_addr))
    var n_rows = _index(params[0])
    var n_cols = _index(params[1])
    var fit_intercept = _index(params[2]) != 0
    var epochs = _index(params[3])
    var alpha = Float32(Float64(py=params[4]))
    var l1_ratio = Float32(Float64(py=params[5]))
    var tol = Float32(Float64(py=params[6]))
    var shuffle = _index(params[7]) != 0
    var has_sw = _index(params[8]) != 0
    var n_iter = 0
    with GILReleased(Python()):
        # `cd_fit_host`'s guards, then `cd_fit_traced`'s, in their order and
        # words, before anything is read.
        if n_rows < 1 or n_cols < 1:
            raise Error(
                "cd_fit_host needs n_rows and n_cols >= 1: got "
                + String(n_rows) + ", " + String(n_cols)
            )
        if epochs < 0:
            raise Error(
                "cd_fit_host: max_iter cannot be negative, got " + String(epochs)
            )
        if n_cols <= 0:
            raise Error(
                "Parameter n_cols: number of columns cannot be less than one"
            )
        if n_rows <= 1:
            raise Error("Parameter n_rows: number of rows cannot be less than two")
        if has_sw:
            raise Error(
                "Parameter sample_weight: REFUSED BY NAME. cd.cuh:136-163 and"
                " :240-251 (the weighted preprocess, the sqrt-weight scaling of"
                " input and labels, and their undo) are not implemented"
            )
        if shuffle:
            raise Error(
                "Parameter shuffle: REFUSED BY NAME (cuML selection='random')."
                " std::shuffle's algorithm is unspecified by the C++ standard,"
                " so cuML's permutation is not a pure function of its seed; only"
                " the cyclic order (shuffle=false, selection='cyclic') is implemented."
                " See solver/impl/shuffle.mojo"
            )
        if alpha < Float32(0.0):
            raise Error("Expected alpha >= 0, got " + String(alpha))
        if l1_ratio < Float32(0.0) or l1_ratio > Float32(1.0):
            raise Error(
                "Expected 0.0 <= l1_ratio <= 1.0, got " + String(l1_ratio)
            )
        if alpha != alpha or alpha - alpha != Float32(0.0):
            raise Error(
                "Parameter alpha: must be a finite number, got " + String(alpha)
            )
        if l1_ratio != l1_ratio:
            raise Error("Parameter l1_ratio: must not be NaN")
        if tol != tol:
            raise Error("Parameter tol: must not be NaN")
        var x = read_f32(x_address, n_rows * n_cols)
        var y = read_f32(y_address, n_rows)
        # THE ONE CALL THAT COMPUTES ANYTHING. `coef` starts at zero inside
        # the oracle, as `cd_fit_host` zeroes it on the device.
        var out = cd_oracle_fit(
            x, y, n_rows, n_cols, fit_intercept, epochs, alpha, l1_ratio, tol, True
        )
        if len(out.coef) != n_cols:
            raise Error("cd_fit: the host oracle returned coef of an unexpected length; nothing written")
        for j in range(n_cols):
            cp[j] = out.coef[j]
        ip[0] = out.intercept
        n_iter = out.n_iter
    return PythonObject(n_iter)


def cd_predict_binding(
    x_addr: PythonObject,
    coef_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`cdPredict` -> `linearRegH` on the host: `pred = X coef + intercept`
    by `gemm_oracle` at OP_TN over the column-major design, then the
    scalar add through `ftz`. `x_addr` is COLUMN-MAJOR `n_rows x n_cols`.
    Writes `n_rows` float32 to `out_addr`.

    `params` is, in this exact order (mirrored in
    `python/mojolearn/_solver_impl.py` and in the GPU binding):

        0  n_rows
        1  n_cols
        2  intercept  (float)
    """
    if len(params) != 3:
        raise Error(
            "cd_predict: params must contain n_rows, n_cols, intercept"
        )
    var x_address = _index(x_addr)
    var coef_address = _index(coef_addr)
    var op = f32_ptr(_index(out_addr))
    var n_rows = _index(params[0])
    var n_cols = _index(params[1])
    var intercept = Float32(Float64(py=params[2]))
    with GILReleased(Python()):
        # `cd_predict_host`'s guard, then `cd_predict`'s, in their words.
        if n_rows < 1 or n_cols < 1:
            raise Error(
                "cd_predict_host needs n_rows and n_cols >= 1: got "
                + String(n_rows) + ", " + String(n_cols)
            )
        if n_cols <= 0:
            raise Error(
                "Parameter n_cols: number of columns cannot be less than one"
            )
        if n_rows <= 1:
            raise Error("Parameter n_rows: number of rows cannot be less than two")
        var x = read_f32(x_address, n_rows * n_cols)
        var coef = read_f32(coef_address, n_cols)
        # `identical_gemm(ctx, pred, x, coef, n_rows, 1, n_cols, OP_TN)`:
        # A is k x m row-major (the column-major design read as
        # n_cols x n_rows), B is k x 1.
        var pred = gemm_oracle(x, coef, OP_TN, n_rows, 1, n_cols)
        for i in range(n_rows):
            var v = pred[i]
            if intercept != Float32(0.0):
                v = ftz(v + intercept)
            op[i] = v
    return PythonObject(0)


def linkage_fit_binding(
    x_addr: PythonObject,
    children_addr: PythonObject,
    labels_addr: PythonObject,
    info_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Single-linkage agglomerative clustering on the host, the GPU
    binding's contract (`bindings/_mojolearn_solver.mojo::linkage_fit_binding`):
    writes `(n_rows - 1) * 2` int32 to `children_addr`, `n_rows` int32 to
    `labels_addr`, two int32 to `info_addr` (`[0]` the Boruvka round count,
    -1 HERE because the host runs Kruskal, see the module docstring; `[1]`
    `n_connected_components`, 1), and returns `info[0]`.

    `x_addr` is ROW-MAJOR `n_rows x n_cols` float32. `params` is, in this
    exact order (mirrored in `python/mojolearn/_hierarchy_impl.py`):

        0  n_rows
        1  n_cols
        2  n_clusters
        3  metric    (1 = L2SqrtExpanded, 0 = L2Expanded; every other code
                      is REFUSED BY NAME as on the device)
        4  use_knn   (0/1; 1 is REFUSED BY NAME as on the device)
    """
    if len(params) != 5:
        raise Error(
            "linkage_fit: params must contain 5 values, got "
            + String(len(params))
        )
    var x_address = _index(x_addr)
    var chp = i32_ptr(_index(children_addr))
    var lp = i32_ptr(_index(labels_addr))
    var ip = i32_ptr(_index(info_addr))
    var n_rows = _index(params[0])
    var n_cols = _index(params[1])
    var n_clusters = _index(params[2])
    var metric = _index(params[3])
    var use_knn = _index(params[4]) != 0
    with GILReleased(Python()):
        # `linkage_fit_host`'s guards, then `cuvs_single_linkage`'s, then
        # `get_distance_graph`'s, then `pairwise_distances`'s, in the order
        # the device path reaches them and in their words.
        if n_rows < 2:
            raise Error(
                "linkage_fit_host needs n_rows >= 2, got " + String(n_rows)
            )
        if n_cols < 1:
            raise Error(
                "linkage_fit_host needs n_cols >= 1, got " + String(n_cols)
            )
        if n_clusters > n_rows:
            raise Error(
                "hierarchy.single_linkage: n_clusters must be less than or equal"
                " to the number of data points (n_clusters=" + String(n_clusters)
                + ", n_rows=" + String(n_rows) + ")"
            )
        if n_clusters < 1:
            raise Error(
                "hierarchy.single_linkage: n_clusters=" + String(n_clusters)
                + " < 1 refused by name (their extract_flattened_clusters would"
                " index children at a negative offset)"
            )
        if use_knn:
            raise Error(
                "hierarchy.get_distance_graph: Linkage::KNN_GRAPH (connectivity="
                "'knn', c=15) refused by name: the knn-graph"
                " connectivity (connectivities.cuh:60-108, knn_graph.cuh) and"
                " the cross-component connection it needs (mst.cuh:75-123,"
                " cross_component_nn.cuh) are rung 2 and not implemented;"
                " use connectivity='pairwise'"
            )
        if n_rows > PAIRWISE_MAX_ROWS:
            raise Error(
                "hierarchy.pairwise_distances: n_rows=" + String(n_rows)
                + " > " + String(PAIRWISE_MAX_ROWS)
                + "; their `int nnz = m * m` (connectivities.cuh:145) overflows"
                " and the dense connectivity matrix is refused by name"
            )
        if metric != DISTANCE_L2_SQRT_EXPANDED and metric != DISTANCE_L2_EXPANDED:
            raise Error(
                "hierarchy.pairwise_distances: metric=" + String(metric)
                + " refused by name; only L2SqrtExpanded (1, cuML's 'euclidean'/"
                "'l2') and L2Expanded (0) are implemented (pairwise_distance_kmeans"
                " raises on every other metric too, kmeans_common.cuh:320)"
            )
        var x = read_f32(x_address, n_rows * n_cols)
        # THE FIT: the oracle's four stages, in the oracle's order.
        var dists = host_pinned_distance_matrix(
            x, n_rows, n_cols, metric == DISTANCE_L2_SQRT_EXPANDED
        )
        var mst = host_kruskal(dists, n_rows)
        if len(mst[0]) != n_rows - 1:
            raise Error(
                "linkage_fit: the host MST has " + String(len(mst[0]))
                + " edges, not n_rows - 1; nothing written"
            )
        var children = host_dendrogram(mst[0], mst[1], n_rows)
        var labels = host_extract_flattened_clusters(children, n_clusters, n_rows)
        if len(children) != (n_rows - 1) * 2 or len(labels) != n_rows:
            raise Error("linkage_fit: the host dendrogram or labels have an unexpected length; nothing written")
        for i in range((n_rows - 1) * 2):
            chp[i] = children[i]
        for i in range(n_rows):
            lp[i] = labels[i]
        ip[0] = Int32(-1)
        ip[1] = Int32(1)
    return PythonObject(-1)


@export
def PyInit__mojolearn_solver_host() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_solver_host")
        module.def_function[solver_host_numeric_mode_binding]("solver_host_numeric_mode")
        module.def_function[solver_host_vendor_binding]("solver_host_vendor")
        module.def_function[solver_host_column_binding]("solver_host_column")
        module.def_function[solver_host_sabotage_binding]("solver_host_sabotage")
        module.def_function[solver_vendor_binding]("solver_vendor")
        module.def_function[cd_fit_binding]("cd_fit")
        module.def_function[cd_predict_binding]("cd_predict")
        module.def_function[linkage_fit_binding]("linkage_fit")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_solver_host: ", error))
