# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for the `_mojolearn_trees` family, the ExtraTrees classifier
and regressor (the CPU training lane, phase 1, et-clf and et-reg,
2026-09-14).

HOST ONLY. No DeviceContext, no kernel launch, no GPU. The fits are
`extratrees/estimator.mojo::fit_extra_trees_classifier_host_exact` and
`fit_extra_trees_regressor_host_exact`, the DEVICE trainer restated on the
host (`train_tree_exact`, `batched_levelalgo/builder.mojo`: the score
kernel's own sequential oracle per (node, feature), the exact `Int64` keys
the device reduces on, the keyed tie order, the device's rescue, and the
leaf kernel's arithmetic over the device's label plane), so the five model
arrays are the GPU columns' bytes, quantized regression leaves included.
The predict is `core/forest_host_predict.mojo::et_host_predict`, the walk
`bindings/_mojolearn_forest_host.mojo` runs and the forest host gate holds
to the recorded Metal predictions on seven CPUs, exported here under the
GPU binding's `et_predict` name and contract so a loaded-back model
predicts through it on a CPU-only install (that is the model column's
RELOAD check).

THE EXPORTED NAMES ARE THE GPU BINDING'S NAMES for what this covers, so
`python/mojolearn/extratrees.py` runs unchanged on a CPU-only install
through `_backend._HOST_MODULES` (`"_mojolearn_trees":
"_mojolearn_trees_host"`): `et_classifier_fit`, `et_regressor_fit`, their
`_export` and `_rowmajor` and `_rowmajor_export` forms (the 22-slot params
list of `bindings/_mojolearn_trees.mojo`, mirrored in `extratrees.py`; the
`_export` forms return the owned-handle descriptor `_forest_protocol.py`
consumes through `forest_export` and `forest_export_release`; the
`_rowmajor` forms take a ROW-major X and transpose it on the host, the
same column-major bytes DEVIATION 2637's pinned stage produces),
`forest_export`, `forest_export_legacy`, `forest_export_release`,
`et_predict`, `trees_vendor` answering "cpu" and `trees_numeric_mode`.
Slot 20 (`device`) is accepted at 1, the only value the wrapper sends: the
GPU binding's "Extra Trees training is GPU-only" refusal is the one line
this binding drops (brief section 3.2). `inference_engine='parallel_groves'`
predicts through the resident entries, and since 2026-09-15
(et-reg-bootstrap-parallel) `forest_prepare_gpu`,
`forest_predict_resident_reuse_gpu` and `forest_release_gpu` are exported
here under those names over `core/forest_host_groves.mojo`, the grove
kernels restated on the host (`RF_INPUT=False`: no input flush). The
non-resident `et_predict_gpu_parallel`, the pool entries and the other
resident comparison arms stay absent and refuse BY NAME through
`_HostBinding`. Best-first growth (`max_leaf_nodes`, DEVIATION 466) fits
through `train_tree_exact_bestfirst` (et-clf-entropy-bestfirst, the same
day).

The sabotage arm (`-D MOJOLEARN_HOST_SABOTAGE=1`, `trees_host_sabotage`)
is `extratrees/checks/pcg_rng.mojo::PCG_HOST_SABOTAGE`: one extra draw
before every threshold, so every forest this binary fits differs.
"""
from core.experiment_guards_trees import TREES_GUARDS_OK
from core.weight_compact_host import (
    host_gather_colmajor,
    host_gather_f32,
    host_nonzero_weight_rows,
)
from std.os import abort
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call as _balanced_call, py_type as _exact_type
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from bindings.forest_export_binding import (
    ForestExportRegistry,
    copy_forest_export_leaves,
    forest_export_py_list_f64,
    forest_export_py_list_i64,
    validate_forest_export_destinations,
)
from core.mojolearn_forest_host_impl import forest_host_vote_into
from core.mojolearn_rf_host_impl import rf_host_col_major
from core.mojolearn_trees_host_impl import et_export_into, et_forest_lists
from core.hostptr import f32_ptr, i32_ptr, read_f32
from checks.kernel_matrix import (
    COLUMN_CPU,
    TARGET_COLUMN,
    column_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.forest_host_predict import et_host_predict, et_host_trees
from bindings.forest_host_strict_binding import (
    forest_strict_prepare_host_binding, forest_strict_predict_host_binding, forest_strict_release_host_binding,
    forest_strict_predict_host_native,
)
from core.forest_host_strict_run import forest_strict_prepare_host_native, forest_strict_release_host_native
from bindings.forest_host_groves_binding import (
    forest_predict_resident_host_binding,
    forest_prepare_host_binding,
    forest_release_host_binding,
)
from core.forest_host_groves_run import (
    forest_prepare_host_native, forest_predict_resident_host_native, forest_release_host_native,
)
from bindings.forest_fit_boundary import forest_fit_args_py
from core.forest_fit_request import ForestFitArgs, XF_CAP_STRICT, forest_export_verify_lists
from core.forest_entry import ForestImage, forest_entry_dispatch, forest_entry_publish
from core.forest_entry_table import FOREST_TABLE_ET
from core.native_descriptor_abi import NativeDescriptorError
from extratrees.checks.pcg_rng import PCG_HOST_SABOTAGE
from extratrees.estimator import (
    ExtraTreesConfig,
    FitResult,
)
from extratrees.host_estimator import (
    fit_extra_trees_classifier_host_exact,
    fit_extra_trees_regressor_host_exact,
)
from extratrees.impl.decisiontree.decisiontree import (
    CRITERION_ENTROPY,
    CRITERION_GINI,
    CRITERION_GAMMA,
    CRITERION_INVERSE_GAUSSIAN,
    CRITERION_MSE,
    CRITERION_POISSON,
)
from core.shared_slot import SharedSlot


comptime N_FIT_PARAMS = 22
"""`bindings/_mojolearn_trees.mojo::N_FIT_PARAMS`, the same 22 slots in the
same order (that docstring is the contract; slot 20 is `device`, 21 the
criterion code)."""

comptime ET_HOST_EXPORTS = SharedSlot[ForestExportRegistry[FitResult], "MojolearnETFitExportHostV1"]


def _index(value: PythonObject) raises -> Int:
    # An exact int (every address, count and slot the wrappers pass) is its
    # own operator.index: read it directly (p3-trees: two attribute reads, a
    # class-name string and a module lookup per integer argument before).
    if _exact_type(value) is _exact_type(PythonObject(0)):
        return Int(py=value)
    var type_name = String(py=value.__class__.__name__)
    if type_name == "bool" or type_name == "bool_":
        raise Error("trees host: integers expected, not booleans")
    var operator_module = native_loaded_module("operator")
    # operator.index through the balanced call: PythonObject.__call__ keeps one
    # reference on each argument (bindings/py_balanced.mojo), which leaked every
    # buffer address passed to a host entry.
    return Int(py=_balanced_call(operator_module.index, value))


def trees_host_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def trees_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def trees_host_column_binding() raises -> PythonObject:
    """`column_name(TARGET_COLUMN)`, the comptime assert's witness: "cpu".
    THE COLUMN IS THE CPU COLUMN, OR THIS DOES NOT BUILD; the assert lives
    in a function body PyInit registers, so it is compiled in every build.
    `bindings/build_host_family.sh` passes -D MOJOLEARN_COLUMN_CPU."""
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "trees host: this binding compiles the CPU column only; pass"
        " -D MOJOLEARN_COLUMN_CPU (bindings/build_trees_host.sh does)"
    )
    return PythonObject(column_name(TARGET_COLUMN))


# There is no `trees_host_detected_column` read-back, for the reason 8d16ce2f
# removed it from the forest and byte LM host bindings: the detected column
# folds to the GPU of the machine that ran the build. The comptime assert
# above is the check.


def trees_host_sabotage_binding() raises -> PythonObject:
    """Whether this binary burns one extra RNG draw before every split
    threshold on purpose (-D MOJOLEARN_HOST_SABOTAGE=1, the gate's negative
    control; `extratrees/checks/pcg_rng.mojo::PCG_HOST_SABOTAGE`)."""
    return PythonObject(PCG_HOST_SABOTAGE)


# The GPU binding's names, same contract.


def trees_vendor_binding() raises -> PythonObject:
    """"cpu". On a CPU-only install `_backend.vendor()` is "cpu" and the
    read-back cross-check expects that string from every host binding."""
    return PythonObject(String("cpu"))


def trees_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))


def _config_from(
    params: ForestFitArgs, base: ExtraTreesConfig
) raises -> ExtraTreesConfig:
    """`bindings/_mojolearn_trees.mojo::_config_from`, the same slots written
    OVER `base` in the same order (DEVIATION 458's lesson: every slot
    overwrites its field after the base is taken)."""
    var config = base.copy()
    config.n_estimators = Int32(params.i(3))
    config.max_depth = Int32(params.i(4))
    config.min_samples_split = Int32(params.i(5))
    config.min_samples_leaf = Int32(params.i(6))
    config.min_weight_fraction_leaf = params.f(7)
    config.max_features_spec = params.i(8)
    config.max_features_fraction = params.f(9)
    config.min_impurity_decrease = Float32(params.f(10))
    config.bootstrap = params.i(11) != 0
    config.oob_score = params.i(12) != 0
    config.random_state = UInt64(params.i(13))
    config.warm_start = params.i(14) != 0
    config.ccp_alpha = params.f(15)
    config.has_class_weight = params.i(16) != 0
    config.has_monotonic_cst = params.i(17) != 0
    config.max_samples = params.i(18)
    config.max_leaf_nodes = Int32(params.i(19))
    config.criterion = Int32(params.i(21))
    return config^


def _read_x_col_major(
    x_addr: Int, n_rows: Int, n_features: Int, row_major: Bool
) raises -> List[Float32]:
    """The design as the trainer reads it, COLUMN-major. A ROW-major X (the
    `_rowmajor` entries, DEVIATION 2637) is transposed here on the host:
    the same bytes the GPU binding's pinned stage holds, moved and folded
    through nothing."""
    var flat = read_f32(x_addr, n_rows * n_features)
    return rf_host_col_major(flat^, n_rows, n_features, row_major)


def _forest_out(result: FitResult) raises -> PythonObject:
    """`bindings/_mojolearn_trees.mojo::_forest_out`, the same six lists."""
    var offsets = List[Int64]()
    var colid = List[Int64]()
    var quesval = List[Float64]()
    var left_child = List[Int64]()
    var leaves = List[Float64]()
    et_forest_lists(result, offsets, colid, quesval, left_child, leaves)
    var meta = Python.list()
    meta.append(PythonObject(Int(result.forest.n_trees)))
    meta.append(PythonObject(Int(result.forest.num_outputs)))
    meta.append(PythonObject(1 if result.depth_cap_bound else 0))
    meta.append(PythonObject(Int(result.plan.params.max_depth)))
    meta.append(PythonObject(result.plan.max_features_count))
    meta.append(PythonObject(Int(result.plan.n_sampled_rows)))
    var out = Python.list()
    out.append(forest_export_py_list_i64(offsets))
    out.append(forest_export_py_list_i64(colid))
    out.append(forest_export_py_list_f64(quesval))
    out.append(forest_export_py_list_i64(left_child))
    out.append(forest_export_py_list_f64(leaves))
    out.append(meta)
    return out


def _retain_et_handle(var result: FitResult) raises -> Int:
    """`bindings/_mojolearn_trees.mojo::_retain_et_export`: the fitted
    forest kept under a handle until `forest_export_release`."""
    var trees = len(result.forest.trees)
    var nodes = 0
    for tree in result.forest.trees:  # driver-loop: over fitted trees, one node count each (export shape bookkeeping)
        nodes += tree.num_nodes()
        if len(tree.vector_leaf) != tree.num_nodes() * Int(result.forest.num_outputs):
            raise Error("fitted ET leaf storage differs from export dimensions")
    var meta: List[Int64] = [
        Int64(result.forest.n_trees),
        Int64(result.forest.num_outputs),
        Int64(1 if result.depth_cap_bound else 0),
        Int64(result.plan.params.max_depth),
        Int64(result.plan.max_features_count),
        Int64(result.plan.n_sampled_rows),
    ]
    var outputs = Int(result.forest.num_outputs)
    if trees != Int(result.forest.n_trees):
        raise Error("fitted ET tree metadata differs from export dimensions")
    return ET_HOST_EXPORTS.get_or_create_ptr()[].insert_native(
        result^, trees, nodes, outputs, meta^
    )


def et_legacy_meta(result: FitResult) -> List[Int64]:
    """The six metadata words `_forest_out` reports (and the export keeps)."""
    var meta: List[Int64] = [Int64(result.forest.n_trees), Int64(result.forest.num_outputs),
        Int64(1 if result.depth_cap_bound else 0), Int64(result.plan.params.max_depth),
        Int64(result.plan.max_features_count), Int64(result.plan.n_sampled_rows)]
    return meta^


def _et_fit[CLASSIFIER: Bool, ROWMAJOR: Bool](params: ForestFitArgs) raises -> Int:
    """`_et_classifier_fit` / `_et_regressor_fit` of the GPU binding: the
    same slot checks in the same words, then the host fit. `tree_start` is
    the GPU binding's global tree ID offset (the shard fits below). Returns
    the export handle (lane calltable-forest)."""
    var tree_start = params.tree_start
    var weights_addr = params.weights
    comptime entry = "et_classifier_fit" if CLASSIFIER else "et_regressor_fit"
    if len(params) != N_FIT_PARAMS:
        raise Error(
            entry + ": params must hold "
            + String(N_FIT_PARAMS)
            + " values, got "
            + String(len(params))
        )
    var n_rows = params.i(0)
    var n_features = params.i(1)
    var n_classes = params.i(2)
    comptime if not CLASSIFIER:
        if n_classes != 0:
            raise Error("et_regressor_fit: n_classes (slot 2) must be 0")
    if params.f(20) != Float64(1):
        raise Error(
            entry + ": device (slot 20) must be 1, the value the wrapper"
            " sends; the host binding fits the same forest the GPU fits"
        )
    var config: ExtraTreesConfig
    comptime if CLASSIFIER:
        config = _config_from(params, ExtraTreesConfig())
        if config.criterion != CRITERION_GINI and config.criterion != CRITERION_ENTROPY:
            raise Error(
                "et_classifier_fit: criterion (slot 21) must be GINI (0) or"
                " ENTROPY (1); got " + String(config.criterion)
            )
    else:
        config = _config_from(params, ExtraTreesConfig().for_regression())
        if config.criterion != CRITERION_MSE and config.criterion != CRITERION_POISSON and (
        config.criterion != CRITERION_GAMMA and config.criterion != CRITERION_INVERSE_GAUSSIAN
    ):
            raise Error(
                "et_regressor_fit: criterion (slot 21) must be MSE (2), POISSON (4), GAMMA (5) or INVERSE_GAUSSIAN (6); got "
                + String(config.criterion)
            )
    if n_rows < 1 or n_features < 1:
        raise Error(entry + ": n_rows and n_features must be >= 1")
    var x_address = params.x
    var y_address = params.y
    var result: FitResult
    with GILReleased(Python()):
        var x = _read_x_col_major(x_address, n_rows, n_features, ROWMAJOR)
        var y = read_f32(y_address, n_rows)
        # lane sk-kernels / sk-final: the per-row weights (0: none)
        var w = List[Float32]()
        if weights_addr != 0:
            w = read_f32(weights_addr, n_rows)
        # lane sk-weq-trees: without bootstrap the zero-weight rows leave
        # the fit, the device's `DeviceDataset.drop_zero_weight_rows` (same
        # rows, same ascending order); the plan then sees the kept count, as
        # the device fit's does (`device_count_nonzero_weights`)
        var n_fit = n_rows
        if len(w) > 0 and not config.bootstrap:
            var rows = host_nonzero_weight_rows(w)
            if len(rows) > 0 and len(rows) < n_rows:
                x = host_gather_colmajor(x, n_rows, n_features, rows)
                y = host_gather_f32(y, rows)
                w = host_gather_f32(w, rows)
                n_fit = len(rows)
        comptime if CLASSIFIER:
            result = fit_extra_trees_classifier_host_exact(
                x, y, Int32(n_fit), Int32(n_features), Int32(n_classes), config,
                tree_start, w^,
            )
        else:
            result = fit_extra_trees_regressor_host_exact(
                x, y, Int32(n_fit), Int32(n_features), config, tree_start,
                w^,
            )
    return _retain_et_handle(result^)


def _et_fit_native[CLASSIFIER: Bool, ROWMAJOR: Bool](args: ForestFitArgs) raises -> Int:
    """One fit (the forest entry's XF_FIT): the export handle."""
    return _et_fit[CLASSIFIER, ROWMAJOR](args)


def _kept_forest_out(handle: Int) raises -> PythonObject:
    # The kept fit's diagnostic lists. A helper of its own so the Dict
    # lookup (DictKeyError) and the list build (Error) both raise Error to
    # the caller's try/finally, which cannot mix the two error types.
    var registry = ET_HOST_EXPORTS.get_or_create_ptr()
    return _forest_out(registry[].entries[handle].model)


def _et_handle_py(handle: Int, export: Bool) raises -> PythonObject:
    """PACK of a kept fit for the Python entries: the export descriptor
    (`*_export`), or the diagnostic lists of the kept fit, released (the
    plain entries; the lists `_forest_out` built from the fit)."""
    var registry = ET_HOST_EXPORTS.get_or_create_ptr()
    if export:
        try:
            return registry[].descriptor_py(handle)
        except error:
            registry[].release(handle)
            raise error^
    var out = PythonObject(None)
    try:
        out = _kept_forest_out(handle)
    finally:
        registry[].release(handle)
    return out


def _et_fit_py[CLASSIFIER: Bool, ROWMAJOR: Bool](
    x_addr: PythonObject, y_addr: PythonObject, params: PythonObject, export: Bool,
    weights: Int = 0, tree_start: Int = 0,
) raises -> PythonObject:
    """The Python fit entries: UNPACK (`forest_fit_args_py`), the native
    fit, PACK (`_et_handle_py`)."""
    var args = forest_fit_args_py(x_addr, y_addr, params, PythonObject(None))
    args.weights = weights
    args.tree_start = tree_start
    return _et_handle_py(_et_fit_native[CLASSIFIER, ROWMAJOR](args), export)


def et_classifier_fit_binding[EXPORT: Bool = False](
    x_addr: PythonObject, y_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    return _et_fit_py[True, False](x_addr, y_addr, params, EXPORT)


def et_regressor_fit_binding[EXPORT: Bool = False](
    x_addr: PythonObject, y_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    return _et_fit_py[False, False](x_addr, y_addr, params, EXPORT)


def _et_export_into_native(id: Int, dst: List[Int], trees: Int, nodes: Int, outputs: Int) raises:
    """XF_EXPORT_INTO: the retained forest into the caller's five arrays."""
    var registry = ET_HOST_EXPORTS.get_or_create_ptr()
    registry[].validate(id, trees, nodes, outputs)
    validate_forest_export_destinations(dst[0], dst[1], dst[2], dst[3], dst[4])
    var op = i32_ptr(dst[0])
    var cp = i32_ptr(dst[1])
    var tp = f32_ptr(dst[2])
    var lp = i32_ptr(dst[3])
    et_export_into(registry[].entries[id].model, op, cp, tp, lp, dst[4], registry[].entries[id].outputs)


def _et_export_verify_native(id: Int, dst: List[Int]) raises -> Int:
    """XF_EXPORT_VERIFY (MOJOLEARN_FOREST_EXPORT=verify): the diagnostic
    lists `forest_export_legacy` returns, against the exported arrays."""
    var registry = ET_HOST_EXPORTS.get_or_create_ptr()
    if id not in registry[].entries:
        raise Error("unknown or released fitted forest export handle")
    ref entry = registry[].entries[id]
    var legacy = et_legacy_meta(entry.model)
    if len(legacy) != len(entry.meta):
        return -1
    for k in range(len(legacy)):  # small-loop(meta: six export metadata words)
        if legacy[k] != entry.meta[k]:
            return -1
    var offsets = List[Int64]()
    var colid = List[Int64]()
    var quesval = List[Float64]()
    var left_child = List[Int64]()
    var leaves = List[Float64]()
    et_forest_lists(entry.model, offsets, colid, quesval, left_child, leaves)
    return forest_export_verify_lists(offsets, colid, quesval, left_child, leaves, dst)


def et_forest_export_binding(
    handle: PythonObject,
    offsets: PythonObject,
    columns: PythonObject,
    thresholds: PythonObject,
    left: PythonObject,
    leaves: PythonObject,
    counts: PythonObject,
) raises -> PythonObject:
    """`forest_export` (UNPACK of `_et_export_into_native`)."""
    if len(counts) != 3:
        raise Error("forest_export requires trees, nodes, outputs capacities")
    var dst: List[Int] = [_index(offsets), _index(columns), _index(thresholds), _index(left), _index(leaves)]
    _et_export_into_native(_index(handle), dst, _index(counts[0]), _index(counts[1]), _index(counts[2]))
    return PythonObject(None)


def et_forest_export_legacy_binding(handle: PythonObject) raises -> PythonObject:
    var registry = ET_HOST_EXPORTS.get_or_create_ptr()
    var id = _index(handle)
    if id not in registry[].entries:
        raise Error("unknown or released fitted forest export handle")
    return _forest_out(registry[].entries[id].model)


def et_forest_export_release_binding(handle: PythonObject) raises -> PythonObject:
    ET_HOST_EXPORTS.get_or_create_ptr()[].release(_index(handle))
    return PythonObject(None)


def et_predict_native(model: List[Int], x_address: Int, out_addr: Int, n_rows: Int, n_features: Int,
                      n_trees: Int, num_outputs: Int) raises -> Int:
    """`et_predict`'s contract (XF_PREDICT): the forest's averaged vote per
    row of a ROW-major `x`, `out` is `n_rows * num_outputs` float32; returns
    rows written. The walk is `core/forest_host_predict.mojo`'s, which
    rebuilds the trees from the five arrays and refuses a malformed node
    instead of reading past it."""
    if n_trees < 1 or num_outputs < 1:
        raise Error("et_predict: n_trees and num_outputs must be >= 1")
    if n_rows < 0 or n_features < 1:
        raise Error("et_predict: n_rows must be >= 0 and n_features >= 1")
    var offsets_p = i32_ptr(model[0])
    var colid_p = i32_ptr(model[1])
    var quesval_p = f32_ptr(model[2])
    var left_p = i32_ptr(model[3])
    var leaves_p = f32_ptr(model[4])
    var op = f32_ptr(out_addr)
    var n_nodes = Int(offsets_p[n_trees])
    if Int(offsets_p[0]) != 0 or n_nodes < n_trees:
        raise Error("et_predict: tree_offsets must start at 0 and hold at least one node per tree")
    if n_rows == 0:
        return 0
    with GILReleased(Python()):
        var rows = read_f32(x_address, n_rows * n_features)
        forest_host_vote_into(
            False, rows, offsets_p, colid_p, quesval_p, left_p,
            leaves_p, op, n_rows, n_features, n_trees, num_outputs, n_nodes,
        )
    return n_rows


def et_predict_binding(
    offsets_addr: PythonObject,
    colid_addr: PythonObject,
    quesval_addr: PythonObject,
    left_child_addr: PythonObject,
    leaves_addr: PythonObject,
    x_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`et_predict` (UNPACK of `et_predict_native`): `params` is `[n_rows,
    n_features, n_trees, num_outputs]`."""
    if len(params) != 4:
        raise Error(
            "et_predict: params must hold [n_rows, n_features, n_trees,"
            " num_outputs], got "
            + String(len(params))
        )
    var model: List[Int] = [_index(offsets_addr), _index(colid_addr), _index(quesval_addr),
                            _index(left_child_addr), _index(leaves_addr)]
    return PythonObject(et_predict_native(model, _index(x_addr), _index(out_addr), _index(params[0]),
                                          _index(params[1]), _index(params[2]), _index(params[3])))


def resident_prepare_binding(
    offsets_addr: PythonObject, colid_addr: PythonObject,
    quesval_addr: PythonObject, left_child_addr: PythonObject,
    leaves_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """`forest_prepare_gpu`: the resident parallel_groves snapshot, on the host."""
    return forest_prepare_host_binding[False](
        offsets_addr, colid_addr, quesval_addr, left_child_addr, leaves_addr, params
    )


def resident_predict_binding(
    handle: PythonObject, x_addr: PythonObject, out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`forest_predict_resident_reuse_gpu`: the grove reduction, on the host."""
    return forest_predict_resident_host_binding[False](handle, x_addr, out_addr, params)


def resident_release_binding(handle: PythonObject) raises -> PythonObject:
    """`forest_release_gpu`."""
    return forest_release_host_binding[False](handle)


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_trees_host.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).
struct EtHostForest(ForestImage):
    """This image's forest entry operations (core/forest_entry.mojo; lane
    calltable-forest). The host column scans X itself, keeps its trees in
    the strict snapshot and has no ordered resident word or resident
    labels."""

    @staticmethod
    def caps() -> Int:
        return XF_CAP_STRICT

    @staticmethod
    def dt_default_bins() -> Int:
        return 128

    @staticmethod
    def fit(classifier: Bool, row_major: Bool, args: ForestFitArgs) raises -> Int:
        if classifier:
            if row_major:
                return _et_fit[True, True](args)
            return _et_fit[True, False](args)
        if row_major:
            return _et_fit[False, True](args)
        return _et_fit[False, False](args)

    @staticmethod
    def fit_dart(args: ForestFitArgs) raises -> Int:
        raise Error("RuntimeError: the extra-trees binding has no DART member fit")

    @staticmethod
    def export_sizes(handle: Int, out_addr: Int) raises:
        ET_HOST_EXPORTS.get_or_create_ptr()[].sizes_into(handle, out_addr)

    @staticmethod
    def export_into(handle: Int, dst: List[Int], trees: Int, nodes: Int, outputs: Int) raises:
        _et_export_into_native(handle, dst, trees, nodes, outputs)

    @staticmethod
    def export_verify(handle: Int, dst: List[Int]) raises -> Int:
        return _et_export_verify_native(handle, dst)

    @staticmethod
    def export_release(handle: Int) raises:
        ET_HOST_EXPORTS.get_or_create_ptr()[].release(handle)

    @staticmethod
    def predict(model: List[Int], x: Int, out_addr: Int, rows: Int, cols: Int, trees: Int, outputs: Int,
                regression: Bool) raises -> Int:
        return et_predict_native(model, x, out_addr, rows, cols, trees, outputs)

    @staticmethod
    def resident_prepare(model: List[Int], features: Int, trees: Int, outputs: Int, ordered: Int) raises -> Int:
        return forest_prepare_host_native[False](model, features, trees, outputs, ordered)

    @staticmethod
    def resident_predict(handle: Int, x: Int, out_addr: Int, rows: Int, cols: Int, outputs: Int) raises -> Int:
        return forest_predict_resident_host_native[False](handle, x, out_addr, rows, cols, outputs)

    @staticmethod
    def resident_labels(handle: Int, x: Int, out_addr: Int, rows: Int, cols: Int, outputs: Int) raises -> Int:
        raise Error("RuntimeError: this forest binding has no resident label prediction")

    @staticmethod
    def resident_release(handle: Int) raises:
        forest_release_host_native[False](handle)

    @staticmethod
    def strict_prepare(model: List[Int], features: Int, trees: Int, outputs: Int) raises -> Int:
        return forest_strict_prepare_host_native[False](model, features, trees, outputs)

    @staticmethod
    def strict_predict(handle: Int, x: Int, out_addr: Int, rows: Int, cols: Int, outputs: Int) raises -> Int:
        return forest_strict_predict_host_native[False](handle, x, out_addr, rows, cols, outputs)

    @staticmethod
    def strict_release(handle: Int) raises:
        forest_strict_release_host_native[False](handle)


def et_host_forest_entry_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """This image's published forest entry (core/forest_entry_table.mojo)."""
    return forest_entry_dispatch[EtHostForest](operation, aa, na, pp, np, ff, nf, result, context, sink)


def build__mojolearn_trees_host() -> PythonObject:
    comptime assert TREES_GUARDS_OK, "invalid integrated experiment configuration (trees guards)"
    try:
        var module = PythonModuleBuilder("_mojolearn_trees_host")
        # The native forest entry (core/forest_entry_table.mojo) under the cpu key.
        forest_entry_publish["MojolearnForestEntryOwnerEtHostV1"](FOREST_TABLE_ET, "cpu", et_host_forest_entry_v1)
        module.def_function[trees_host_numeric_mode_binding]("trees_host_numeric_mode")
        module.def_function[trees_host_vendor_binding]("trees_host_vendor")
        module.def_function[trees_host_column_binding]("trees_host_column")
        module.def_function[trees_host_sabotage_binding]("trees_host_sabotage")
        module.def_function[trees_vendor_binding]("trees_vendor")
        module.def_function[trees_numeric_mode_binding]("trees_numeric_mode")
        module.def_function[et_classifier_fit_binding[False]]("et_classifier_fit")
        module.def_function[et_classifier_fit_binding[True]]("et_classifier_fit_export")
        # TOMBSTONE: the legacy-arm rf/et fits and rf_regressor_fit_session_export (dead native entries: reached only by the deleted MOJOLEARN_FOREST_EXPORT=legacy arm / no caller) deleted 2026-10-26 by f515bbcb9; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_trees.patch
        # TOMBSTONE: TREES_LEGACY_PYTHON_ENTRIES (DELETED: no caller; the door fits through the forest entry table) deleted 2026-11-12 by round2-fixes-2; code recoverable at 8c8e62f56b.
        # Restore: git apply experiments/removed/TREES_LEGACY_PYTHON_ENTRIES.patch
        # TOMBSTONE: the legacy-arm rf/et fits and rf_regressor_fit_session_export (dead native entries: reached only by the deleted MOJOLEARN_FOREST_EXPORT=legacy arm / no caller) deleted 2026-10-26 by f515bbcb9; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_trees.patch
        module.def_function[et_regressor_fit_binding[False]]("et_regressor_fit")
        module.def_function[et_regressor_fit_binding[True]]("et_regressor_fit_export")
        # TOMBSTONE: the legacy-arm rf/et fits and rf_regressor_fit_session_export (dead native entries: reached only by the deleted MOJOLEARN_FOREST_EXPORT=legacy arm / no caller) deleted 2026-10-26 by f515bbcb9; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_trees.patch
        module.def_function[et_forest_export_binding]("forest_export")
        module.def_function[et_forest_export_legacy_binding]("forest_export_legacy")
        module.def_function[et_forest_export_release_binding]("forest_export_release")
        # TOMBSTONE: the legacy-arm rf/et fits and rf_regressor_fit_session_export (dead native entries: reached only by the deleted MOJOLEARN_FOREST_EXPORT=legacy arm / no caller) deleted 2026-10-26 by f515bbcb9; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_trees.patch
        module.def_function[et_predict_binding]("et_predict")
        module.def_function[resident_prepare_binding]("forest_prepare_gpu")
        module.def_function[resident_predict_binding]("forest_predict_resident_reuse_gpu")
        module.def_function[resident_release_binding]("forest_release_gpu")
        # The strict snapshot of the sequential engine (N044, ne1-a-trees).
        module.def_function[forest_strict_prepare_host_binding[False]]("forest_strict_prepare")
        module.def_function[forest_strict_predict_host_binding[False]]("forest_strict_predict")
        module.def_function[forest_strict_release_host_binding[False]]("forest_strict_release")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_trees_host: ", error))
