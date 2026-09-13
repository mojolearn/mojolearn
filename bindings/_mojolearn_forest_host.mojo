# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU inference binding for saved RandomForest, ExtraTrees and
GradientBoosting models.

HOST ONLY. No DeviceContext, no kernel, no GPU, and nothing imported from
`ensemble/randomforest.mojo`, `extratrees/impl/randomforest/randomforest.mojo`
or `gbdt/train.mojo`, which import the device side. The forest per-tree walks
come from the files the GPU bindings compile; the forest loops around them
are `core/forest_host_predict.mojo`, and the GBDT quantize, walk and leaf sum
are `core/gbdt_host_predict.mojo`, each saying line by line what it MIRRORS.
Every entry validates the counts and the spans before it dereferences an
address, copies the inputs into owned Lists, computes, and only then writes
the caller's output.

The GBDT entry (`forest_host_gbdt_predict`, the GBDT host lane, 2026-09-13)
takes the model as FLAT ARRAYS parsed from the saved model text by
`python/mojolearn/_gbdt_host.py`, not as the text the GPU binding takes,
because the text parser lives in a module that imports the device side.
Its address contract is `[border_offsets i32 (n_features + 1), borders f32,
fold_counts i32 (n_features), one_hot i32 (n_features), nan_treatment i32
(n_features), tree_offsets i32 (n_trees + 1), split_feature i32 (n_splits),
split_bin i32 (n_splits), split_take_bin i32 (n_splits), node_left i32
(n_splits), node_right i32 (n_splits), leaf_offsets i32 (n_trees + 1),
leaves f32 (n_leaf_values), x f32 (n_rows * n_cols, COLUMN-major, as
`gbdt_predict` takes it), out f32 (n_rows * dim, ROW-major)]`, `params` is
`[n_rows, n_cols, n_trees, dim, non_symmetric, n_splits, n_leaf_values,
n_borders]`, and `bias` is the model's float64 bias. It returns the rows
written. `forest_host_gbdt_sigmoid` is `gbdt_sigmoid`'s body
(`bindings/_mojolearn_gbdt.mojo:133-149`), the Logloss link the GPU binding
already computes on the host.

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
from bindings.hostptr import f32_ptr, f64_ptr, i32_ptr, read_f32
from checks.kernel_matrix import (
    COLUMN_CPU,
    DETECTED_COLUMN,
    TARGET_COLUMN,
    column_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, identical_exp64
from core.forest_host_predict import (
    FOREST_HOST_SABOTAGE,
    et_host_predict,
    et_host_trees,
    rf_host_predict,
    rf_host_trees,
)
from core.gbdt_host_predict import gbdt_host_predict


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


def forest_host_column_binding() raises -> PythonObject:
    """`column_name(TARGET_COLUMN)`, the comptime assert's witness: "cpu".

    THE COLUMN IS THE CPU COLUMN, OR THIS DOES NOT BUILD (the CPU training
    lane, 2026-09-13). Until COLUMN_CPU existed a host build fell through to
    COLUMN_APPLE, and a matched hash under that fallthrough was luck for the
    rows the byte LM reaches and would have been Apple's kNN repairs for a
    host fit. `bindings/build_forest_host.sh` passes -D MOJOLEARN_COLUMN_CPU;
    a build that reaches this file any other way stops here with the column
    it got. The assert lives in this function because Mojo takes a
    `comptime assert` inside a function body only, and PyInit registers this
    function, so it is compiled in every build of the module."""
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "forest host: this binding compiles the CPU column only; pass"
        " -D MOJOLEARN_COLUMN_CPU (bindings/build_forest_host.sh does)"
    )
    return PythonObject(column_name(TARGET_COLUMN))


def forest_host_detected_column_binding() raises -> PythonObject:
    """`column_name(DETECTED_COLUMN)`: what the accelerator predicates fold to
    in THIS build, with no define. "cpu" on a build with no accelerator target;
    a vendor's name means the predicate answered for the host machine."""
    return PythonObject(column_name(DETECTED_COLUMN))


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


def forest_host_gbdt_predict_binding(
    addresses: PythonObject, params: PythonObject, bias: PythonObject
) raises -> PythonObject:
    """`gbdt_predict` / `gbdt_predict_multi` at `PREDICT_RAW` on the host:
    the raw approxes, `n_rows * dim` float32, ROW-major. The link (the
    Logloss sigmoid) is the Python layer's call to `forest_host_gbdt_sigmoid`,
    as it is `gbdt_sigmoid` for the GPU binding."""
    var entry = String("forest_host_gbdt_predict")
    if len(addresses) != 15 or len(params) != 8:
        raise Error(entry + ": expected 15 addresses and 8 params")
    var n_rows = _index(params[0])
    var n_cols = _index(params[1])
    var n_trees = _index(params[2])
    var dim = _index(params[3])
    var non_symmetric = _index(params[4]) != 0
    var n_splits = _index(params[5])
    var n_leaf_values = _index(params[6])
    var n_borders = _index(params[7])
    if n_rows <= 0 or n_rows > FOREST_HOST_MAX_ROWS:
        raise Error(entry + ": n_rows must be in [1, 2^30]")
    if n_cols <= 0 or n_cols > FOREST_HOST_MAX_ROWS:
        raise Error(entry + ": n_cols must be in [1, 2^30]")
    if n_trees < 0 or n_trees > 2147483646:
        raise Error(entry + ": n_trees must be in [0, 2^31 - 2]")
    if dim <= 0 or dim > 65536:
        raise Error(entry + ": dim must be in [1, 65536]")
    if n_splits < 0 or n_leaf_values < 0 or n_borders < 0:
        raise Error(entry + ": counts must be non-negative")
    if n_splits > 2147483647 or n_leaf_values > 2147483647 or n_borders > 2147483647:
        raise Error(entry + ": counts must fit int32")
    var bias_value = Float64(py=bias)
    var border_offsets_p = i32_ptr(_index(addresses[0]))
    var borders_p = f32_ptr(_index(addresses[1]))
    var fold_counts_p = i32_ptr(_index(addresses[2]))
    var one_hot_p = i32_ptr(_index(addresses[3]))
    var nan_p = i32_ptr(_index(addresses[4]))
    var tree_offsets_p = i32_ptr(_index(addresses[5]))
    var split_feature_p = i32_ptr(_index(addresses[6]))
    var split_bin_p = i32_ptr(_index(addresses[7]))
    var split_take_bin_p = i32_ptr(_index(addresses[8]))
    var node_left_p = i32_ptr(_index(addresses[9]))
    var node_right_p = i32_ptr(_index(addresses[10]))
    var leaf_offsets_p = i32_ptr(_index(addresses[11]))
    var leaves_p = f32_ptr(_index(addresses[12]))
    var x_addr = _index(addresses[13])
    var op = f32_ptr(_index(addresses[14]))
    if Int(border_offsets_p[0]) != 0 or Int(border_offsets_p[n_cols]) != n_borders:
        raise Error(entry + ": border_offsets must start at 0 and end at n_borders")

    var wrote = 0
    with GILReleased(Python()):
        var rows = read_f32(x_addr, n_rows * n_cols)
        var out = List[Float32](length=n_rows * dim, fill=Float32(0.0))
        gbdt_host_predict(
            rows, n_rows, n_cols,
            border_offsets_p, borders_p, fold_counts_p, one_hot_p, nan_p,
            tree_offsets_p, split_feature_p, split_bin_p, split_take_bin_p,
            node_left_p, node_right_p, leaf_offsets_p, leaves_p,
            n_trees, dim, non_symmetric, n_splits, n_leaf_values, bias_value,
            out,
        )
        for i in range(n_rows * dim):
            op[i] = out[i]
        wrote = n_rows
    return PythonObject(wrote)


def forest_host_gbdt_sigmoid_binding(
    raw_addr: PythonObject, out_addr: PythonObject, n: PythonObject
) raises -> PythonObject:
    """`gbdt_sigmoid` (`bindings/_mojolearn_gbdt.mojo:133-149`), the same
    body: `out[i] = 1 / (1 + exp(-raw[i]))` in double through
    `identical_exp64`, `n` values, both buffers float64."""
    var rp = f64_ptr(_index(raw_addr))
    var op = f64_ptr(_index(out_addr))
    var count = _index(n)
    if count < 0:
        raise Error("forest_host_gbdt_sigmoid: n must be non-negative")
    for i in range(count):
        var r = rp.unsafe_load(i)
        op.unsafe_store(i, 1.0 / (1.0 + identical_exp64(-r)))
    return PythonObject(count)


@export
def PyInit__mojolearn_forest_host() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_forest_host")
        module.def_function[forest_host_numeric_mode_binding]("forest_host_numeric_mode")
        module.def_function[forest_host_vendor_binding]("forest_host_vendor")
        module.def_function[forest_host_column_binding]("forest_host_column")
        module.def_function[forest_host_detected_column_binding]("forest_host_detected_column")
        module.def_function[forest_host_sabotage_binding]("forest_host_sabotage")
        module.def_function[forest_host_rf_predict_proba_binding]("forest_host_rf_predict_proba")
        module.def_function[forest_host_rf_predict_reg_binding]("forest_host_rf_predict_reg")
        module.def_function[forest_host_et_predict_binding]("forest_host_et_predict")
        module.def_function[forest_host_gbdt_predict_binding]("forest_host_gbdt_predict")
        module.def_function[forest_host_gbdt_sigmoid_binding]("forest_host_gbdt_sigmoid")
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
