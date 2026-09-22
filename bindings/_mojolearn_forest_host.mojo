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
written. `forest_host_gbdt_expand_ctr` (lane/inference-gbdt-ctr-tables,
2026-09-15) turns the raw input columns of a model with CTR tables or tensor
CTRs into its model columns first, through `core/gbdt_host_ctr.mojo`; its
address contract is `[x f32 (n_rows * n_raw, COLUMN-major), ctr_ints i32,
ctr_floats f32, tensor_ints i32, tensor_floats f32, counts i32,
border_offsets i32 (n_cols + 1), borders f32, one_hot i32 (n_cols), out f32
(n_rows * n_cols, COLUMN-major)]` and `params` is `[n_rows, n_raw, n_cols,
n_ctr_tables, len(ctr_ints), len(ctr_floats), n_tensor_tables,
len(tensor_ints), len(tensor_floats), len(counts), n_borders]`.
`forest_host_gbdt_sigmoid` is `gbdt_sigmoid`'s body
(`bindings/_mojolearn_gbdt.mojo:133-149`), the Logloss link the GPU binding
already computes on the host.

Address contract, shared by the three predict entries. `addresses` is
`[offsets i32 (n_trees + 1), colid i32 (n_nodes), quesval f32 (n_nodes),
left_child i32 (n_nodes), leaves f32 (n_nodes * num_outputs), x f32
(n_rows * n_cols, ROW-major), out f32 (n_rows * num_outputs)]` and `params`
is `[n_rows, n_cols, n_trees, num_outputs, n_nodes]`. `n_nodes` is what the
GPU bindings do not take; it lets a file read off disk be refused instead of
read past. Each entry returns the rows written.

THE `parallel_groves` ENGINE (lane/forest-groves-cpu-and-speed, 2026-09-17).
`forest_host_groves_prepare(addresses, params, family)` takes the five model
addresses above (no x, no out), `params = [n_trees, n_cols, num_outputs,
n_nodes]` and `family` 0 for RandomForest (the input flushed, `RF_INPUT`) or
1 for ExtraTrees, validates the graph as the GPU resident snapshot does
(`core/forest_host_groves.mojo::HostGroveForest`) and returns a handle;
`forest_host_groves_predict(handle, x_addr, out_addr, params, family)` with
`params = [n_rows, n_cols, num_outputs]` runs the host grove engine (32
fixed lanes, the 16/8/4/2/1 fold, DEVIATION 2960 threads) and returns the
rows written; `forest_host_groves_release(handle, family)` drops the
snapshot. The three hold the GIL, as the GPU binding's resident entries do,
so a handle is never released under a running prediction.
`forest_host_groves_sabotage` reads back DEVIATION 2961's define.
"""
from std.ffi import _Global
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
from bindings.hostptr import f32_ptr, f64_ptr, i32_ptr, read_f32, read_i32
from checks.kernel_matrix import (
    COLUMN_CPU,
    TARGET_COLUMN,
    column_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, identical_exp64
from core.forest_host_groves import (
    FOREST_GROVES_SABOTAGE,
    HostGroveForest,
    HostGroveRegistry,
)
from core.forest_host_predict import (
    FOREST_HOST_SABOTAGE,
    et_host_predict,
    et_host_trees,
    rf_host_predict,
    rf_host_trees,
)
from core.gbdt_host_ctr import GBDT_CTR_HOST_SABOTAGE, gbdt_host_expand_ctr
from core.gbdt_host_predict import gbdt_host_predict


comptime FAMILY_RF = 0
comptime FAMILY_ET = 1

# The host grove registries of THIS binding, named apart from the rf and
# trees host families' ("MojoRFResidentForestHost",
# "MojoETResidentForestHost" in bindings/forest_host_groves_binding.mojo),
# so two host bindings loaded into one process never share a handle space.
comptime RF_GROVES = _Global[
    StorageType=HostGroveRegistry,
    name="MojoForestHostRFGroves",
    init_fn=HostGroveRegistry.__init__,
]
comptime ET_GROVES = _Global[
    StorageType=HostGroveRegistry,
    name="MojoForestHostETGroves",
    init_fn=HostGroveRegistry.__init__,
]
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


# There is no `forest_host_detected_column` read-back. The 2026-09-13 phase 0
# witness returned the name of the kernel matrix's detected column, which folds
# to the GPU of the machine that ran the build, so the 0.8.5 CPU training
# binding built on the NVIDIA legs carried the string "nvidia" and the copy
# from the AMD leg "amd" (43 bytes apart: that string, its length, its symbol
# name and the build id) and packaging/linux/pack_wheel.py refused the wheel.
# A vendor-neutral binary must not depend on the builder's accelerator; the
# comptime assert above is the load-bearing check and needs no such witness.


def forest_host_sabotage_binding() raises -> PythonObject:
    """Whether this binary carries ANY of its sabotage arms on purpose.

    BOTH arms, not just `FOREST_HOST_SABOTAGE`. The CTR arm
    (`-D MOJOLEARN_GBDT_CTR_HOST_SABOTAGE=1`, `core/gbdt_host_ctr.mojo`) is
    also exported on its own as `forest_host_gbdt_ctr_sabotage`, and until
    2026-09-16 this read-back ignored it. A binding built with the CTR define
    therefore recorded `sabotage: false` in a column's `host.families`, so the
    column could not witness its own arm, and `_backend`'s
    MOJOLEARN_HOST_ALLOW_SABOTAGE guard never fired for it either. The
    committed column
    `bench/results/identity_break/2026-09-15_gbdt-ctr-tables/cpu-x86-ctr-sabotage.json`
    is classified as a PRODUCTION column by any reader that trusts this flag,
    which is how the sabotage audit found it. The arm itself is
    real and was watched to fail; what was missing is the read-back, and a
    column that misreports which binary produced it undermines every verdict
    read from it. `python/mojolearn/_forest_host.py` already ORs the two when
    it decides whether to refuse the load; this makes the column say the same.
    """
    return PythonObject(FOREST_HOST_SABOTAGE or GBDT_CTR_HOST_SABOTAGE or FOREST_GROVES_SABOTAGE)


def forest_host_groves_sabotage_binding() raises -> PythonObject:
    """Whether this binary folds the 32 grove lanes in lane order on purpose
    (DEVIATION 2961, `-D MOJOLEARN_FOREST_GROVES_SABOTAGE=1`)."""
    return PythonObject(FOREST_GROVES_SABOTAGE)


def _family(value: PythonObject, entry: String) raises -> Int:
    var family = _index(value)
    if family != FAMILY_RF and family != FAMILY_ET:
        raise Error(entry + ": family must be 0 (RandomForest) or 1 (ExtraTrees)")
    return family


def forest_host_groves_prepare_binding(
    addresses: PythonObject, params: PythonObject, family_arg: PythonObject
) raises -> PythonObject:
    """The host grove snapshot of a saved forest: the five model arrays
    copied and validated once, a handle back."""
    var entry = String("forest_host_groves_prepare")
    var family = _family(family_arg, entry)
    if len(addresses) != 5 or len(params) != 4:
        raise Error(entry + ": expected 5 addresses and 4 params")
    var n_trees = _index(params[0])
    var n_cols = _index(params[1])
    var num_outputs = _index(params[2])
    var n_nodes = _index(params[3])
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
    if Int(offsets_p[0]) != 0 or Int(offsets_p[n_trees]) != n_nodes:
        raise Error(entry + ": tree_offsets must start at 0 and end at n_nodes")
    var offsets = read_i32(_index(addresses[0]), n_trees + 1)
    var columns = read_i32(_index(addresses[1]), n_nodes)
    var thresholds = read_f32(_index(addresses[2]), n_nodes)
    var left = read_i32(_index(addresses[3]), n_nodes)
    var leaves = read_f32(_index(addresses[4]), n_nodes * num_outputs)
    _ = colid_p
    _ = quesval_p
    _ = left_p
    _ = leaves_p
    var model = HostGroveForest(
        offsets^, columns^, thresholds^, left^, leaves^, n_cols, num_outputs
    )
    if family == FAMILY_RF:
        return PythonObject(RF_GROVES.get_or_create_ptr()[].prepare(model^))
    return PythonObject(ET_GROVES.get_or_create_ptr()[].prepare(model^))


def forest_host_groves_predict_binding(
    handle: PythonObject, x_addr: PythonObject, out_addr: PythonObject,
    params: PythonObject, family_arg: PythonObject,
) raises -> PythonObject:
    """The grove engine's answer for `n_rows` ROW-major rows: the divided
    fold, `n_rows * num_outputs` float32 into `out`. Returns the rows."""
    var entry = String("forest_host_groves_predict")
    var family = _family(family_arg, entry)
    if len(params) != 3:
        raise Error(entry + ": expected 3 params [n_rows, n_cols, num_outputs]")
    var n_rows = _index(params[0])
    var n_cols = _index(params[1])
    var num_outputs = _index(params[2])
    if n_rows <= 0 or n_rows > FOREST_HOST_MAX_ROWS:
        raise Error(entry + ": n_rows must be in [1, 2^30]")
    if n_cols <= 0 or n_cols > FOREST_HOST_MAX_ROWS or num_outputs <= 0:
        raise Error(entry + ": n_cols and num_outputs must be positive")
    if n_rows > 2147483647 // n_cols or n_rows > 2147483647 // num_outputs:
        raise Error(entry + ": n_rows * n_cols and n_rows * num_outputs must fit int32")
    var xp = f32_ptr(_index(x_addr))
    var op = f32_ptr(_index(out_addr))
    var id = _index(handle)
    var state = RF_GROVES.get_or_create_ptr()
    if family == FAMILY_ET:
        state = ET_GROVES.get_or_create_ptr()
    if id not in state[].entries:
        raise Error(entry + ": unknown or released host grove forest handle")
    if family == FAMILY_RF:
        state[].entries[id].predict_into[True](xp, op, n_rows, n_cols, num_outputs)
    else:
        state[].entries[id].predict_into[False](xp, op, n_rows, n_cols, num_outputs)
    return PythonObject(n_rows)


def forest_host_groves_release_binding(
    handle: PythonObject, family_arg: PythonObject
) raises -> PythonObject:
    var family = _family(family_arg, String("forest_host_groves_release"))
    if family == FAMILY_RF:
        RF_GROVES.get_or_create_ptr()[].release(_index(handle))
    else:
        ET_GROVES.get_or_create_ptr()[].release(_index(handle))
    return PythonObject(None)


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
    if len(addresses) != 15 or (len(params) != 8 and len(params) != 9):
        raise Error(entry + ": expected 15 addresses and 8 or 9 params")
    var n_rows = _index(params[0])
    var n_cols = _index(params[1])
    var n_trees = _index(params[2])
    var dim = _index(params[3])
    var non_symmetric = _index(params[4]) != 0
    var n_splits = _index(params[5])
    var n_leaf_values = _index(params[6])
    var n_borders = _index(params[7])
    var row_major = len(params) == 9 and _index(params[8]) != 0
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
            out, row_major=row_major,
        )
        for i in range(n_rows * dim):
            op[i] = out[i]
        wrote = n_rows
    return PythonObject(wrote)


def forest_host_gbdt_expand_ctr_binding(
    addresses: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`predict_floats`'s CTR step on the host: raw columns to model
    columns. Returns the columns written."""
    var entry = String("forest_host_gbdt_expand_ctr")
    if len(addresses) != 10 or len(params) != 11:
        raise Error(entry + ": expected 10 addresses and 11 params")
    var n_rows = _index(params[0])
    var n_raw = _index(params[1])
    var n_cols = _index(params[2])
    var n_ctr = _index(params[3])
    var n_ctr_ints = _index(params[4])
    var n_ctr_floats = _index(params[5])
    var n_tensor = _index(params[6])
    var n_tensor_ints = _index(params[7])
    var n_tensor_floats = _index(params[8])
    var n_counts = _index(params[9])
    var n_borders = _index(params[10])
    if n_rows <= 0 or n_rows > FOREST_HOST_MAX_ROWS:
        raise Error(entry + ": n_rows must be in [1, 2^30]")
    if n_raw <= 0 or n_cols < n_raw or n_cols > 65536:
        raise Error(entry + ": n_raw must be positive and n_cols in [n_raw, 65536]")
    if n_rows > 2147483647 // n_cols:
        raise Error(entry + ": n_rows * n_cols must fit int32")
    for i in range(3, 11):
        if _index(params[i]) < 0 or _index(params[i]) > 2147483647:
            raise Error(entry + ": counts must be non-negative and fit int32")
    var wrote = 0
    var x_addr = _index(addresses[0])
    var a1 = _index(addresses[1])
    var a2 = _index(addresses[2])
    var a3 = _index(addresses[3])
    var a4 = _index(addresses[4])
    var a5 = _index(addresses[5])
    var a6 = _index(addresses[6])
    var a7 = _index(addresses[7])
    var a8 = _index(addresses[8])
    var op = f32_ptr(_index(addresses[9]))
    with GILReleased(Python()):
        var x = read_f32(x_addr, n_rows * n_raw)
        var ctr_ints = read_i32(a1, n_ctr_ints)
        var ctr_floats = read_f32(a2, n_ctr_floats)
        var tensor_ints = read_i32(a3, n_tensor_ints)
        var tensor_floats = read_f32(a4, n_tensor_floats)
        var counts = read_i32(a5, n_counts)
        var border_offsets = read_i32(a6, n_cols + 1)
        var borders = read_f32(a7, n_borders)
        var one_hot = read_i32(a8, n_cols)
        var out = gbdt_host_expand_ctr(
            x, n_rows, n_cols, ctr_ints, ctr_floats, n_ctr, tensor_ints,
            tensor_floats, n_tensor, counts, border_offsets, borders, one_hot,
        )
        if len(out) != n_rows * n_cols:
            raise Error(entry + ": the expansion wrote " + String(len(out)) + " values")
        for i in range(n_rows * n_cols):
            op[i] = out[i]
        wrote = n_cols
    return PythonObject(wrote)


def forest_host_gbdt_ctr_sabotage_binding() raises -> PythonObject:
    """Whether this binary rotates every CTR table's counts on purpose."""
    return PythonObject(GBDT_CTR_HOST_SABOTAGE)


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


def forest_host_gbdt_sigmoid_pair_binding(
    raw_addr: PythonObject, out_addr: PythonObject, n: PythonObject
) raises -> PythonObject:
    """DEVIATION 2902 (lane/infer-speed-trees, 2026-09-17): the two
    probability columns of a Logloss or CrossEntropy `predict_proba`,
    `out[2 * i] = 1 - p` and `out[2 * i + 1] = p` with
    `p = 1 / (1 + exp(-raw[i]))` through `identical_exp64`, `n` rows, both
    buffers float64 and the caller's. `p` is `forest_host_gbdt_sigmoid`'s
    value and `1.0 - p` is the one IEEE double subtraction the Python
    layer computed per row (DEVIATION 2333, retired by this entry point
    where the binary carries it); a subtraction has no fusion partner and
    no association, so the column's bits are the Python column's. Under
    the gate's sabotage build the two columns are written swapped, so a
    column that could not tell would read IDENTICAL and be caught."""
    var rp = f64_ptr(_index(raw_addr))
    var op = f64_ptr(_index(out_addr))
    var count = _index(n)
    if count < 0:
        raise Error("forest_host_gbdt_sigmoid_pair: n must be non-negative")
    for i in range(count):
        var r = rp.unsafe_load(i)
        var p = 1.0 / (1.0 + identical_exp64(-r))
        comptime if FOREST_HOST_SABOTAGE:
            op.unsafe_store(2 * i, p)
            op.unsafe_store(2 * i + 1, 1.0 - p)
        else:
            op.unsafe_store(2 * i, 1.0 - p)
            op.unsafe_store(2 * i + 1, p)
    return PythonObject(count)


@export
def PyInit__mojolearn_forest_host() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_forest_host")
        module.def_function[forest_host_numeric_mode_binding]("forest_host_numeric_mode")
        module.def_function[forest_host_vendor_binding]("forest_host_vendor")
        module.def_function[forest_host_column_binding]("forest_host_column")
        module.def_function[forest_host_sabotage_binding]("forest_host_sabotage")
        module.def_function[forest_host_rf_predict_proba_binding]("forest_host_rf_predict_proba")
        module.def_function[forest_host_rf_predict_reg_binding]("forest_host_rf_predict_reg")
        module.def_function[forest_host_et_predict_binding]("forest_host_et_predict")
        module.def_function[forest_host_groves_sabotage_binding]("forest_host_groves_sabotage")
        module.def_function[forest_host_groves_prepare_binding]("forest_host_groves_prepare")
        module.def_function[forest_host_groves_predict_binding]("forest_host_groves_predict")
        module.def_function[forest_host_groves_release_binding]("forest_host_groves_release")
        module.def_function[forest_host_gbdt_predict_binding]("forest_host_gbdt_predict")
        module.def_function[forest_host_gbdt_sigmoid_binding]("forest_host_gbdt_sigmoid")
        module.def_function[forest_host_gbdt_sigmoid_pair_binding]("forest_host_gbdt_sigmoid_pair")
        module.def_function[forest_host_gbdt_expand_ctr_binding]("forest_host_gbdt_expand_ctr")
        module.def_function[forest_host_gbdt_ctr_sabotage_binding]("forest_host_gbdt_ctr_sabotage")
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
