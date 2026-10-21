# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for the `_mojolearn_solver` family, coordinate descent today
(the CPU training lane, phase 1, 2026-09-13).

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
Since 2026-09-28 (lane cluster-cpu) the matrix, Kruskal and the dendrogram
run as `hierarchy/host/linkage_host.mojo` (Prim's walk under the same total
order, threaded, O(m) memory): the same edges in the same order and the
same `children` rows, checked against the oracle by
`hierarchy/checks/linkage_host_check.mojo`.
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
from bindings.py_balanced import py_call
from core.mojolearn_solver_host_impl import linkage_fit_host_into, cd_predict_host_into, cd_fit_host_into
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from core.hostptr import f32_ptr, i32_ptr, read_f32
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
from hierarchy.checks.linkage_oracle import host_extract_flattened_clusters
from hierarchy.host.linkage_host import host_dendrogram_fast, host_prim_mst
from hierarchy.impl.cluster.detail.connectivities import (
    DISTANCE_L2_EXPANDED,
    DISTANCE_L2_SQRT_EXPANDED,
    PAIRWISE_MAX_ROWS,
)
from solver.host.cd_oracle import CdOracleResult, cd_oracle_fit, cd_oracle_fit_weighted
from solver.impl.cd_gram_rule import cd_idn_gram_shape
from std.os import getenv


def _index(value: PythonObject) raises -> Int:
    var type_name = String(py=value.__class__.__name__)
    if type_name == "bool" or type_name == "bool_":
        raise Error("solver host: integers expected, not booleans")
    var operator_module = Python.import_module("operator")
    return Int(py=py_call(operator_module.index, value))


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
        8  has_sample_weight  (0/1; 1 runs `cd_oracle_fit_weighted`, the
                              device's weighted fit, and needs slot 10)
        9  row_major          (0/1, optional): `x_addr` is C-ORDER; the GPU
                              binding transposes it on the device, this one
                              reads it into the same column-major words
                              before the oracle (the refcol host smoke at
                              811275d5b refused every Lasso/ElasticNet fit
                              here: Python passes 10 values since
                              lane/gap-nv-classical2)
       10  sample_weight address (optional; lane sk-kernels): n_rows
                              float32 weights, read only when slot 8 is 1
    """
    if len(params) < 9 or len(params) > 11:
        raise Error(
            "cd_fit: params must contain 9 to 11 values, got " + String(len(params))
        )
    var row_major = False
    if len(params) >= 10:
        row_major = _index(params[9]) != 0
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
    var sw_address = 0
    if has_sw:
        if len(params) != 11:
            raise Error("cd_fit: has_sample_weight needs the weight address as params[10]")
        sw_address = _index(params[10])
    var n_iter = 0
    with GILReleased(Python()):
        n_iter = cd_fit_host_into(
            x_address, y_address, sw_address, cp, ip, n_rows, n_cols, fit_intercept, epochs,
            alpha, l1_ratio, tol, shuffle, has_sw, row_major,
        )
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
        cd_predict_host_into(x_address, coef_address, op, n_rows, n_cols, intercept)
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
        linkage_fit_host_into(x_address, chp, lp, ip, n_rows, n_cols, n_clusters, metric, use_knn)
    return PythonObject(-1)


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_solver_host.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).
def build__mojolearn_solver_host() -> PythonObject:
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
