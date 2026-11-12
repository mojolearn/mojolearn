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
`forest_host_gbdt_sigmoid_pair` is `gbdt_sigmoid_pair`'s body
(`bindings/_mojolearn_gbdt.mojo`), the Logloss link of both probability
columns.

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
from core.experiment_guards_trees import TREES_GUARDS_OK
from std.os import abort
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call as _balanced_call, py_type as _exact_type
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
from checks.kernel_matrix import (
    COLUMN_CPU,
    TARGET_COLUMN,
    column_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, identical_exp64
from core.forest_host_groves import FOREST_GROVES_SABOTAGE
from core.forest_host_predict import (
    FOREST_HOST_SABOTAGE,
    et_host_predict,
    et_host_trees,
    rf_host_predict,
    rf_host_trees,
)
from core.gbdt_host_ctr import GBDT_CTR_HOST_SABOTAGE, gbdt_host_expand_ctr
from core.gbdt_host_predict import gbdt_host_predict
from core.host_entry_table_forest import (
    forest_host_et_predict_run,
    forest_host_gbdt_expand_ctr_run,
    forest_host_gbdt_predict_run,
    forest_host_gbdt_sigmoid_pair_run,
    forest_host_groves_predict_run,
    forest_host_groves_prepare_run,
    forest_host_groves_release_run,
    forest_host_rf_predict_proba_run,
    forest_host_rf_predict_reg_run,
    publish_forest_host_entries,
)


def _index(value: PythonObject) raises -> Int:
    # An exact int (every address, count and slot the wrappers pass) is its
    # own operator.index: read it directly (p3-trees: two attribute reads, a
    # class-name string and a module lookup per integer argument before).
    if _exact_type(value) is _exact_type(PythonObject(0)):
        return Int(py=value)
    var type_name = String(py=value.__class__.__name__)
    if type_name == "bool" or type_name == "bool_":
        raise Error("forest host: integers expected, not booleans")
    var operator_module = native_loaded_module("operator")
    # operator.index through the balanced call: PythonObject.__call__ keeps one
    # reference on each argument (bindings/py_balanced.mojo), which leaked every
    # buffer address passed to a host entry.
    return Int(py=_balanced_call(operator_module.index, value))


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


def _words(values: PythonObject) raises -> List[Int]:
    """UNPACK: the exact-int address or parameter words of one export."""
    var words = List[Int](capacity=len(values))
    for i in range(len(values)):  # driver-loop: the fixed address/parameter words of one call
        words.append(_index(values[i]))
    return words^


# The bodies of the entries below are the typed runs of
# core/host_entry_table_forest.mojo, which the native forest host entry
# (published at module init) runs too. These Python exports only unpack and
# pack; they remain for the Python callers (python/mojolearn/host_surface.py,
# the legacy reference tests and the identity tools). The saved-model owner
# (bindings/host_tree_native.mojo) calls the native entry instead.


def forest_host_groves_prepare_binding(
    addresses: PythonObject, params: PythonObject, family_arg: PythonObject
) raises -> PythonObject:
    """The host grove snapshot of a saved forest: the five model arrays
    copied and validated once, a handle back."""
    var family = _index(family_arg)
    return PythonObject(forest_host_groves_prepare_run(_words(addresses), _words(params), family))


def forest_host_groves_predict_binding(
    handle: PythonObject, x_addr: PythonObject, out_addr: PythonObject,
    params: PythonObject, family_arg: PythonObject,
) raises -> PythonObject:
    """The grove engine's answer for `n_rows` ROW-major rows: the divided
    fold, `n_rows * num_outputs` float32 into `out`. Returns the rows."""
    var family = _index(family_arg)
    var words = _words(params)
    return PythonObject(forest_host_groves_predict_run(_index(handle), _index(x_addr), _index(out_addr), words, family))


def forest_host_groves_release_binding(
    handle: PythonObject, family_arg: PythonObject
) raises -> PythonObject:
    var family = _index(family_arg)
    forest_host_groves_release_run(_index(handle), family)
    return PythonObject(None)


def forest_host_rf_predict_proba_binding(
    addresses: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`rf_predict_proba`'s answer on the host: the divided vote, `num_outputs >= 2`.
    The argmax is the Python layer's, as it is for the GPU binding."""
    var a = _words(addresses)
    var p = _words(params)
    var wrote = 0
    with GILReleased(Python()):
        wrote = forest_host_rf_predict_proba_run(a, p)
    return PythonObject(wrote)


def forest_host_rf_predict_reg_binding(
    addresses: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`rf_predict_reg`'s answer on the host: the one-output vote, `num_outputs == 1`."""
    var a = _words(addresses)
    var p = _words(params)
    var wrote = 0
    with GILReleased(Python()):
        wrote = forest_host_rf_predict_reg_run(a, p)
    return PythonObject(wrote)


def forest_host_et_predict_binding(
    addresses: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`et_predict`'s answer on the host: the divided vote for classifier and
    regressor alike, `num_outputs >= 1`."""
    var a = _words(addresses)
    var p = _words(params)
    var wrote = 0
    with GILReleased(Python()):
        wrote = forest_host_et_predict_run(a, p)
    return PythonObject(wrote)


def forest_host_gbdt_predict_binding(
    addresses: PythonObject, params: PythonObject, bias: PythonObject
) raises -> PythonObject:
    """`gbdt_predict` / `gbdt_predict_multi` at `PREDICT_RAW` on the host:
    the raw approxes, `n_rows * dim` float32, ROW-major. The link (the
    Logloss sigmoid) is the Python layer's call to
    `forest_host_gbdt_sigmoid_pair`, as it is `gbdt_sigmoid_pair` for the GPU binding."""
    var a = _words(addresses)
    var p = _words(params)
    var bias_value = Float64(py=bias)
    var wrote = 0
    with GILReleased(Python()):
        wrote = forest_host_gbdt_predict_run(a, p, bias_value)
    return PythonObject(wrote)


def forest_host_gbdt_expand_ctr_binding(
    addresses: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`predict_floats`'s CTR step on the host: raw columns to model
    columns. Returns the columns written."""
    var a = _words(addresses)
    var p = _words(params)
    var wrote = 0
    with GILReleased(Python()):
        wrote = forest_host_gbdt_expand_ctr_run(a, p)
    return PythonObject(wrote)


def forest_host_gbdt_ctr_sabotage_binding() raises -> PythonObject:
    """Whether this binary rotates every CTR table's counts on purpose."""
    return PythonObject(GBDT_CTR_HOST_SABOTAGE)


# TOMBSTONE: forest_host_gbdt_sigmoid (dead native entry: no live caller; ensemble.py calls forest_host_gbdt_sigmoid_pair) deleted 2026-10-26 by 65616e765; code recoverable at 485ba9cc1.
# Restore: git apply experiments/removed/forest_host_gbdt_sigmoid.patch


def forest_host_gbdt_sigmoid_pair_binding(
    raw_addr: PythonObject, out_addr: PythonObject, n: PythonObject
) raises -> PythonObject:
    """DEVIATION 2902 (lane/infer-speed-trees, 2026-09-17): the two
    probability columns of a Logloss or CrossEntropy `predict_proba`,
    `out[2 * i] = 1 - p` and `out[2 * i + 1] = p` with
    `p = 1 / (1 + exp(-raw[i]))` through `identical_exp64`, `n` rows, both
    buffers float64 and the caller's. `p` is that double and `1.0 - p` is the one IEEE double subtraction the Python
    layer computed per row (DEVIATION 2333, retired by this entry point
    where the binary carries it); a subtraction has no fusion partner and
    no association, so the column's bits are the Python column's. Under
    the gate's sabotage build the two columns are written swapped, so a
    column that could not tell would read IDENTICAL and be caught."""
    var raw = _index(raw_addr)
    var dst = _index(out_addr)
    return PythonObject(forest_host_gbdt_sigmoid_pair_run(raw, dst, _index(n)))


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_forest_host.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).
def build__mojolearn_forest_host() -> PythonObject:
    comptime assert TREES_GUARDS_OK, "invalid integrated experiment configuration (trees guards)"
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
        # TOMBSTONE: forest_host_gbdt_sigmoid (dead native entry: no live caller) deleted 2026-10-26 by 65616e765; code recoverable at 485ba9cc1.
        # Restore: git apply experiments/removed/forest_host_gbdt_sigmoid.patch
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
        var published_module = module.finalize()
        # Native publication of the same runs (core/host_entry_table.mojo),
        # after the module is ready; the Python exports above remain for the
        # Python callers named beside them.
        publish_forest_host_entries()
        return published_module
    except error:
        abort(String("failed to create _mojolearn_forest_host: ", error))
