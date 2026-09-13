# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for the `_mojolearn_solver` family, coordinate descent today
(the CPU training lane, phase 1, 2026-09-13; brief
docs/lanes/BRIEF_cpu_training_2026-09-13.md sections 1.1 lasso, elasticnet
and 3.2).

HOST ONLY. No DeviceContext, no kernel launch, no GPU. The fit is
`solver/checks/cd_oracle.mojo::cd_oracle_fit` at `profile=True`, "cdFit on
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
`solver_vendor` answering "cpu". `linkage_fit` (agglomerative) is absent
until its lane lands and refuses BY NAME through `_HostBinding`.
"""
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from bindings.hostptr import f32_ptr, read_f32
from checks.kernel_matrix import (
    COLUMN_CPU,
    DETECTED_COLUMN,
    TARGET_COLUMN,
    column_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, ftz
from gemm.checks.gemm_oracle import (
    GEMM_ORACLE_HOST_SABOTAGE,
    OP_TN,
    gemm_oracle,
)
from solver.checks.cd_oracle import cd_oracle_fit


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


def solver_host_detected_column_binding() raises -> PythonObject:
    """`column_name(DETECTED_COLUMN)`: what the accelerator predicates fold
    to in THIS build, with no define."""
    return PythonObject(column_name(DETECTED_COLUMN))


def solver_host_sabotage_binding() raises -> PythonObject:
    """Whether this binary walks every GEMM leaf descending on purpose
    (-D MOJOLEARN_HOST_SABOTAGE=1, the gate's negative control; every
    reduction of the CD oracle is a gemm_oracle_cell, so the arm reaches
    this family through gemm/checks/gemm_oracle.mojo)."""
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


@export
def PyInit__mojolearn_solver_host() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_solver_host")
        module.def_function[solver_host_numeric_mode_binding]("solver_host_numeric_mode")
        module.def_function[solver_host_vendor_binding]("solver_host_vendor")
        module.def_function[solver_host_column_binding]("solver_host_column")
        module.def_function[solver_host_detected_column_binding]("solver_host_detected_column")
        module.def_function[solver_host_sabotage_binding]("solver_host_sabotage")
        module.def_function[solver_vendor_binding]("solver_vendor")
        module.def_function[cd_fit_binding]("cd_fit")
        module.def_function[cd_predict_binding]("cd_predict")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_solver_host: ", error))
