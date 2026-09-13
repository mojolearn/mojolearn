# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU inference binding for saved RandomForest and ExtraTrees models.

HOST ONLY. No DeviceContext, no kernel, no GPU, and nothing imported from
`ensemble/randomforest.mojo` or `extratrees/impl/randomforest/randomforest.mojo`,
which import the device side. The per-tree walks come from the files the GPU
bindings compile; the forest loops around them are `core/forest_host_predict.mojo`,
which says line by line what it MIRRORS. Every entry validates the counts and
the spans before it dereferences an address, copies the inputs into owned
Lists, computes, and only then writes the caller's output.

Address contract, shared by the three predict entries. `addresses` is
`[offsets i32 (n_trees + 1), colid i32 (n_nodes), quesval f32 (n_nodes),
left_child i32 (n_nodes), leaves f32 (n_nodes * num_outputs), x f32
(n_rows * n_cols, ROW-major), out f32 (n_rows * num_outputs)]` and `params`
is `[n_rows, n_cols, n_trees, num_outputs, n_nodes]`. `n_nodes` is what the
GPU bindings do not take; it lets a file read off disk be refused instead of
read past. Each entry returns the rows written.
"""
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from bindings.host_helpers import (
    all_finite_f32_binding,
    all_finite_f64_binding,
    argmax_rows_f32_binding,
    argmax_rows_f64_binding,
    cast_f64_to_f32_binding,
    gather_f64_binding,
    gather_i64_binding,
)
from bindings.hostptr import f32_ptr, i32_ptr, read_f32
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.forest_host_predict import (
    FOREST_HOST_SABOTAGE,
    et_host_predict,
    et_host_trees,
    rf_host_predict,
    rf_host_trees,
)


comptime FAMILY_RF = 0
comptime FAMILY_ET = 1
#: Rows per call, so `n_rows * n_cols` stays far from any Int edge.
comptime FOREST_HOST_MAX_ROWS = 1073741824


def _index(value: PythonObject) raises -> Int:
    var type_name = String(py=value.__class__.__name__)
    if type_name == "bool" or type_name == "bool_":
        raise Error("forest host: integers expected, not booleans")
    var operator_module = Python.import_module("operator")
    return Int(py=operator_module.index(value))


def forest_host_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def forest_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def forest_host_sabotage_binding() raises -> PythonObject:
    """Whether this binary divides the vote by the wrong count on purpose."""
    return PythonObject(FOREST_HOST_SABOTAGE)


def _predict(
    addresses: PythonObject, params: PythonObject, family: Int, entry: String
) raises -> PythonObject:
    if len(addresses) != 7 or len(params) != 5:
        raise Error(entry + ": expected 7 addresses and 5 params")
    var n_rows = _index(params[0])
    var n_cols = _index(params[1])
    var n_trees = _index(params[2])
    var num_outputs = _index(params[3])
    var n_nodes = _index(params[4])
    if n_rows <= 0 or n_rows > FOREST_HOST_MAX_ROWS:
        raise Error(entry + ": n_rows must be in [1, 2^30]")
    if n_cols <= 0 or n_cols > FOREST_HOST_MAX_ROWS:
        raise Error(entry + ": n_cols must be in [1, 2^30]")
    if n_trees <= 0 or n_trees > 2147483646:
        raise Error(entry + ": n_trees must be in [1, 2^31 - 2]")
    if num_outputs <= 0 or n_nodes <= 0 or n_nodes > 2147483647 // num_outputs:
        raise Error(entry + ": num_outputs and n_nodes must be positive and n_nodes * num_outputs must fit int32")
    if n_nodes < n_trees:
        raise Error(entry + ": fewer nodes than trees")
    var offsets_p = i32_ptr(_index(addresses[0]))
    var colid_p = i32_ptr(_index(addresses[1]))
    var quesval_p = f32_ptr(_index(addresses[2]))
    var left_p = i32_ptr(_index(addresses[3]))
    var leaves_p = f32_ptr(_index(addresses[4]))
    var x_addr = _index(addresses[5])
    var op = f32_ptr(_index(addresses[6]))
    if Int(offsets_p[0]) != 0 or Int(offsets_p[n_trees]) != n_nodes:
        raise Error(entry + ": tree_offsets must start at 0 and end at n_nodes")

    var wrote = 0
    with GILReleased(Python()):
        var rows = read_f32(x_addr, n_rows * n_cols)
        var out = List[Float32](length=n_rows * num_outputs, fill=Float32(0.0))
        if family == FAMILY_RF:
            var trees = rf_host_trees(
                offsets_p, colid_p, quesval_p, left_p, leaves_p,
                n_trees, n_nodes, n_cols, num_outputs,
            )
            rf_host_predict(trees, rows, n_rows, n_cols, n_trees, num_outputs, out)
        else:
            var trees = et_host_trees(
                offsets_p, colid_p, quesval_p, left_p, leaves_p,
                n_trees, n_nodes, n_cols, num_outputs,
            )
            et_host_predict(trees, rows, n_rows, n_cols, n_trees, num_outputs, out)
        for i in range(n_rows * num_outputs):
            op[i] = out[i]
        wrote = n_rows
    return PythonObject(wrote)


def forest_host_rf_predict_proba_binding(
    addresses: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`rf_predict_proba`'s answer on the host: the divided vote, `num_outputs >= 2`.
    The argmax is the Python layer's, as it is for the GPU binding."""
    if _index(params[3]) < 2:
        raise Error("forest_host_rf_predict_proba: num_outputs must be >= 2")
    return _predict(addresses, params, FAMILY_RF, "forest_host_rf_predict_proba")


def forest_host_rf_predict_reg_binding(
    addresses: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`rf_predict_reg`'s answer on the host: the one-output vote, `num_outputs == 1`."""
    if _index(params[3]) != 1:
        raise Error("forest_host_rf_predict_reg: num_outputs must be 1")
    return _predict(addresses, params, FAMILY_RF, "forest_host_rf_predict_reg")


def forest_host_et_predict_binding(
    addresses: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`et_predict`'s answer on the host: the divided vote for classifier and
    regressor alike, `num_outputs >= 1`."""
    return _predict(addresses, params, FAMILY_ET, "forest_host_et_predict")


@export
def PyInit__mojolearn_forest_host() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_forest_host")
        module.def_function[forest_host_numeric_mode_binding]("forest_host_numeric_mode")
        module.def_function[forest_host_vendor_binding]("forest_host_vendor")
        module.def_function[forest_host_sabotage_binding]("forest_host_sabotage")
        module.def_function[forest_host_rf_predict_proba_binding]("forest_host_rf_predict_proba")
        module.def_function[forest_host_rf_predict_reg_binding]("forest_host_rf_predict_reg")
        module.def_function[forest_host_et_predict_binding]("forest_host_et_predict")
        module.def_function[all_finite_f32_binding]("all_finite_f32")
        module.def_function[all_finite_f64_binding]("all_finite_f64")
        module.def_function[cast_f64_to_f32_binding]("cast_f64_to_f32")
        module.def_function[argmax_rows_f32_binding]("argmax_rows_f32")
        module.def_function[argmax_rows_f64_binding]("argmax_rows_f64")
        module.def_function[gather_i64_binding]("gather_i64")
        module.def_function[gather_f64_binding]("gather_f64")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_forest_host: ", error))
