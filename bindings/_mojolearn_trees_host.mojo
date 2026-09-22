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
from std.ffi import _Global
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from bindings.forest_export_binding import (
    ForestExportRegistry,
    copy_forest_export_leaves,
    validate_forest_export_destinations,
)
from bindings.hostptr import f32_ptr, i32_ptr, read_f32
from checks.kernel_matrix import (
    COLUMN_CPU,
    TARGET_COLUMN,
    column_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.forest_host_predict import et_host_predict, et_host_trees
from bindings.forest_host_groves_binding import (
    forest_predict_resident_host_binding,
    forest_prepare_host_binding,
    forest_release_host_binding,
)
from extratrees.checks.pcg_rng import PCG_HOST_SABOTAGE
from extratrees.estimator import (
    ExtraTreesConfig,
    FitResult,
    fit_extra_trees_classifier_host_exact,
    fit_extra_trees_regressor_host_exact,
)
from extratrees.impl.decisiontree.decisiontree import (
    CRITERION_ENTROPY,
    CRITERION_GINI,
    CRITERION_MSE,
)


comptime N_FIT_PARAMS = 22
"""`bindings/_mojolearn_trees.mojo::N_FIT_PARAMS`, the same 22 slots in the
same order (that docstring is the contract; slot 20 is `device`, 21 the
criterion code)."""

comptime ET_HOST_EXPORTS = _Global[
    StorageType=ForestExportRegistry[FitResult],
    name="MojoETFitExportHost",
    init_fn=ForestExportRegistry[FitResult].__init__,
]


def _index(value: PythonObject) raises -> Int:
    var type_name = String(py=value.__class__.__name__)
    if type_name == "bool" or type_name == "bool_":
        raise Error("trees host: integers expected, not booleans")
    var operator_module = Python.import_module("operator")
    return Int(py=operator_module.index(value))


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
    params: PythonObject, base: ExtraTreesConfig
) raises -> ExtraTreesConfig:
    """`bindings/_mojolearn_trees.mojo::_config_from`, the same slots written
    OVER `base` in the same order (DEVIATION 458's lesson: every slot
    overwrites its field after the base is taken)."""
    var config = base.copy()
    config.n_estimators = Int32(_index(params[3]))
    config.max_depth = Int32(_index(params[4]))
    config.min_samples_split = Int32(_index(params[5]))
    config.min_samples_leaf = Int32(_index(params[6]))
    config.min_weight_fraction_leaf = Float64(py=params[7])
    config.max_features_spec = _index(params[8])
    config.max_features_fraction = Float64(py=params[9])
    config.min_impurity_decrease = Float32(Float64(py=params[10]))
    config.bootstrap = _index(params[11]) != 0
    config.oob_score = _index(params[12]) != 0
    config.random_state = UInt64(_index(params[13]))
    config.warm_start = _index(params[14]) != 0
    config.ccp_alpha = Float64(py=params[15])
    config.has_class_weight = _index(params[16]) != 0
    config.has_monotonic_cst = _index(params[17]) != 0
    config.max_samples = _index(params[18])
    config.max_leaf_nodes = Int32(_index(params[19]))
    config.criterion = Int32(_index(params[21]))
    return config^


def _read_x_col_major(
    x_addr: Int, n_rows: Int, n_features: Int, row_major: Bool
) raises -> List[Float32]:
    """The design as the trainer reads it, COLUMN-major. A ROW-major X (the
    `_rowmajor` entries, DEVIATION 2637) is transposed here on the host:
    the same bytes the GPU binding's pinned stage holds, moved and folded
    through nothing."""
    var flat = read_f32(x_addr, n_rows * n_features)
    if not row_major:
        return flat^
    var out = List[Float32](length=n_rows * n_features, fill=Float32(0.0))
    for r in range(n_rows):
        for c in range(n_features):
            out[c * n_rows + r] = flat[r * n_features + c]
    return out^


def _forest_out(result: FitResult) raises -> PythonObject:
    """`bindings/_mojolearn_trees.mojo::_forest_out`, the same six lists."""
    var offsets = Python.list()
    var colid = Python.list()
    var quesval = Python.list()
    var left_child = Python.list()
    var leaves = Python.list()
    var total = 0
    offsets.append(PythonObject(0))
    for t in range(len(result.forest.trees)):
        var n = result.forest.trees[t].num_nodes()
        total += n
        offsets.append(PythonObject(total))
        for i in range(n):
            var node = result.forest.trees[t].sparsetree[i]
            colid.append(PythonObject(Int(node.colid)))
            quesval.append(PythonObject(Float64(node.quesval)))
            left_child.append(PythonObject(Int(node.left_child_id)))
        for i in range(len(result.forest.trees[t].vector_leaf)):
            leaves.append(
                PythonObject(Float64(result.forest.trees[t].vector_leaf[i]))
            )
    var meta = Python.list()
    meta.append(PythonObject(Int(result.forest.n_trees)))
    meta.append(PythonObject(Int(result.forest.num_outputs)))
    meta.append(PythonObject(1 if result.depth_cap_bound else 0))
    meta.append(PythonObject(Int(result.plan.params.max_depth)))
    meta.append(PythonObject(result.plan.max_features_count))
    meta.append(PythonObject(Int(result.plan.n_sampled_rows)))
    var out = Python.list()
    out.append(offsets)
    out.append(colid)
    out.append(quesval)
    out.append(left_child)
    out.append(leaves)
    out.append(meta)
    return out


def _retain_et_export(var result: FitResult) raises -> PythonObject:
    """`bindings/_mojolearn_trees.mojo::_retain_et_export`: the fitted
    forest kept under a handle until `forest_export_release`."""
    var trees = len(result.forest.trees)
    var nodes = 0
    for tree in result.forest.trees:
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
    return ET_HOST_EXPORTS.get_or_create_ptr()[].insert(
        result^, trees, nodes, outputs, meta^
    )


def _et_fit[
    CLASSIFIER: Bool, EXPORT: Bool, ROWMAJOR: Bool
](
    x_addr: PythonObject, y_addr: PythonObject, params: PythonObject,
    tree_start: Int = 0,
) raises -> PythonObject:
    """`_et_classifier_fit` / `_et_regressor_fit` of the GPU binding: the
    same slot checks in the same words, then the host fit. `tree_start` is
    the GPU binding's global tree ID offset (the shard fits below)."""
    comptime entry = "et_classifier_fit" if CLASSIFIER else "et_regressor_fit"
    if len(params) != N_FIT_PARAMS:
        raise Error(
            entry + ": params must hold "
            + String(N_FIT_PARAMS)
            + " values, got "
            + String(len(params))
        )
    var n_rows = _index(params[0])
    var n_features = _index(params[1])
    var n_classes = _index(params[2])
    comptime if not CLASSIFIER:
        if n_classes != 0:
            raise Error("et_regressor_fit: n_classes (slot 2) must be 0")
    if Float64(py=params[20]) != Float64(1):
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
        if config.criterion != CRITERION_MSE:
            raise Error(
                "et_regressor_fit: criterion (slot 21) must be MSE (2); got "
                + String(config.criterion)
            )
    if n_rows < 1 or n_features < 1:
        raise Error(entry + ": n_rows and n_features must be >= 1")
    var x_address = _index(x_addr)
    var y_address = _index(y_addr)
    var result: FitResult
    with GILReleased(Python()):
        var x = _read_x_col_major(x_address, n_rows, n_features, ROWMAJOR)
        var y = read_f32(y_address, n_rows)
        comptime if CLASSIFIER:
            result = fit_extra_trees_classifier_host_exact(
                x, y, Int32(n_rows), Int32(n_features), Int32(n_classes), config,
                tree_start,
            )
        else:
            result = fit_extra_trees_regressor_host_exact(
                x, y, Int32(n_rows), Int32(n_features), config, tree_start
            )
    comptime if EXPORT:
        return _retain_et_export(result^)
    else:
        return _forest_out(result)


def et_classifier_fit_binding(
    x_addr: PythonObject, y_addr: PythonObject, params: PythonObject
) raises -> PythonObject:
    return _et_fit[True, False, False](x_addr, y_addr, params)


def et_classifier_fit_export_binding(
    x_addr: PythonObject, y_addr: PythonObject, params: PythonObject
) raises -> PythonObject:
    return _et_fit[True, True, False](x_addr, y_addr, params)


def et_classifier_fit_rowmajor_binding(
    x_addr: PythonObject, y_addr: PythonObject, params: PythonObject
) raises -> PythonObject:
    return _et_fit[True, False, True](x_addr, y_addr, params)


def et_classifier_fit_rowmajor_export_binding(
    x_addr: PythonObject, y_addr: PythonObject, params: PythonObject
) raises -> PythonObject:
    return _et_fit[True, True, True](x_addr, y_addr, params)


def et_regressor_fit_binding(
    x_addr: PythonObject, y_addr: PythonObject, params: PythonObject
) raises -> PythonObject:
    return _et_fit[False, False, False](x_addr, y_addr, params)


def et_regressor_fit_export_binding(
    x_addr: PythonObject, y_addr: PythonObject, params: PythonObject
) raises -> PythonObject:
    return _et_fit[False, True, False](x_addr, y_addr, params)


def et_regressor_fit_rowmajor_binding(
    x_addr: PythonObject, y_addr: PythonObject, params: PythonObject
) raises -> PythonObject:
    return _et_fit[False, False, True](x_addr, y_addr, params)


def et_regressor_fit_rowmajor_export_binding(
    x_addr: PythonObject, y_addr: PythonObject, params: PythonObject
) raises -> PythonObject:
    return _et_fit[False, True, True](x_addr, y_addr, params)


def et_classifier_fit_shard_binding(
    x_addr: PythonObject, y_addr: PythonObject, params: PythonObject,
    tree_start: PythonObject,
) raises -> PythonObject:
    """`bindings/_mojolearn_trees.mojo::et_classifier_fit_shard_binding`:
    the column-major fit of trees `tree_start .. tree_start + n_estimators`
    of the whole forest (`parallel_ensemble.fit_forest`'s shard;
    lane/cpu-training-par-wave2, 2026-09-15)."""
    return _et_fit[True, False, False](x_addr, y_addr, params, _index(tree_start))


def et_regressor_fit_shard_binding(
    x_addr: PythonObject, y_addr: PythonObject, params: PythonObject,
    tree_start: PythonObject,
) raises -> PythonObject:
    """`bindings/_mojolearn_trees.mojo::et_regressor_fit_shard_binding`."""
    return _et_fit[False, False, False](x_addr, y_addr, params, _index(tree_start))


def et_forest_export_binding(
    handle: PythonObject,
    offsets: PythonObject,
    columns: PythonObject,
    thresholds: PythonObject,
    left: PythonObject,
    leaves: PythonObject,
    counts: PythonObject,
) raises -> PythonObject:
    """`bindings/_mojolearn_trees.mojo::et_forest_export_binding`: the
    retained forest into the caller's five arrays."""
    if len(counts) != 3:
        raise Error("forest_export requires trees, nodes, outputs capacities")
    var id = _index(handle)
    var registry = ET_HOST_EXPORTS.get_or_create_ptr()
    registry[].validate(id, _index(counts[0]), _index(counts[1]), _index(counts[2]))
    validate_forest_export_destinations(
        _index(offsets), _index(columns), _index(thresholds), _index(left), _index(leaves)
    )
    var op = i32_ptr(_index(offsets))
    var cp = i32_ptr(_index(columns))
    var tp = f32_ptr(_index(thresholds))
    var lp = i32_ptr(_index(left))
    var total = 0
    op[0] = 0
    ref model = registry[].entries[id].model
    for t in range(len(model.forest.trees)):
        ref tree = model.forest.trees[t]
        for i in range(tree.num_nodes()):
            ref node = tree.sparsetree[i]
            cp[total + i] = node.colid
            tp[total + i] = node.quesval
            lp[total + i] = node.left_child_id
        copy_forest_export_leaves(
            tree.vector_leaf, _index(leaves), total * registry[].entries[id].outputs
        )
        total += tree.num_nodes()
        op[t + 1] = Int32(total)
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
    """`et_predict`'s contract (`bindings/_mojolearn_trees.mojo`): the
    forest's averaged vote per row of a ROW-major `x`, `out` is
    `n_rows * num_outputs` float32, `params` is `[n_rows, n_features,
    n_trees, num_outputs]`; returns rows written. The walk is
    `core/forest_host_predict.mojo`'s (`et_host_trees`, `et_host_predict`),
    the forest host binding's, which rebuilds the trees from the five
    arrays and refuses a malformed node instead of reading past it."""
    if len(params) != 4:
        raise Error(
            "et_predict: params must hold [n_rows, n_features, n_trees,"
            " num_outputs], got "
            + String(len(params))
        )
    var n_rows = _index(params[0])
    var n_features = _index(params[1])
    var n_trees = _index(params[2])
    var num_outputs = _index(params[3])
    if n_trees < 1 or num_outputs < 1:
        raise Error("et_predict: n_trees and num_outputs must be >= 1")
    if n_rows < 0 or n_features < 1:
        raise Error("et_predict: n_rows must be >= 0 and n_features >= 1")
    var offsets_p = i32_ptr(_index(offsets_addr))
    var colid_p = i32_ptr(_index(colid_addr))
    var quesval_p = f32_ptr(_index(quesval_addr))
    var left_p = i32_ptr(_index(left_child_addr))
    var leaves_p = f32_ptr(_index(leaves_addr))
    var x_address = _index(x_addr)
    var op = f32_ptr(_index(out_addr))
    var n_nodes = Int(offsets_p[n_trees])
    if Int(offsets_p[0]) != 0 or n_nodes < n_trees:
        raise Error("et_predict: tree_offsets must start at 0 and hold at least one node per tree")
    var wrote = 0
    if n_rows == 0:
        return PythonObject(0)
    with GILReleased(Python()):
        var rows = read_f32(x_address, n_rows * n_features)
        var out = List[Float32](length=n_rows * num_outputs, fill=Float32(0.0))
        var trees = et_host_trees(
            offsets_p, colid_p, quesval_p, left_p, leaves_p,
            n_trees, n_nodes, n_features, num_outputs,
        )
        et_host_predict(trees, rows, n_rows, n_features, n_trees, num_outputs, out)
        for i in range(n_rows * num_outputs):
            op[i] = out[i]
        wrote = n_rows
    return PythonObject(wrote)


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


@export
def PyInit__mojolearn_trees_host() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_trees_host")
        module.def_function[trees_host_numeric_mode_binding]("trees_host_numeric_mode")
        module.def_function[trees_host_vendor_binding]("trees_host_vendor")
        module.def_function[trees_host_column_binding]("trees_host_column")
        module.def_function[trees_host_sabotage_binding]("trees_host_sabotage")
        module.def_function[trees_vendor_binding]("trees_vendor")
        module.def_function[trees_numeric_mode_binding]("trees_numeric_mode")
        module.def_function[et_classifier_fit_binding]("et_classifier_fit")
        module.def_function[et_classifier_fit_export_binding]("et_classifier_fit_export")
        module.def_function[et_classifier_fit_rowmajor_binding]("et_classifier_fit_rowmajor")
        module.def_function[et_classifier_fit_rowmajor_export_binding]("et_classifier_fit_rowmajor_export")
        module.def_function[et_regressor_fit_binding]("et_regressor_fit")
        module.def_function[et_regressor_fit_export_binding]("et_regressor_fit_export")
        module.def_function[et_regressor_fit_rowmajor_binding]("et_regressor_fit_rowmajor")
        module.def_function[et_regressor_fit_rowmajor_export_binding]("et_regressor_fit_rowmajor_export")
        module.def_function[et_forest_export_binding]("forest_export")
        module.def_function[et_forest_export_legacy_binding]("forest_export_legacy")
        module.def_function[et_forest_export_release_binding]("forest_export_release")
        module.def_function[et_classifier_fit_shard_binding]("et_classifier_fit_shard")
        module.def_function[et_regressor_fit_shard_binding]("et_regressor_fit_shard")
        module.def_function[et_predict_binding]("et_predict")
        module.def_function[resident_prepare_binding]("forest_prepare_gpu")
        module.def_function[resident_predict_binding]("forest_predict_resident_reuse_gpu")
        module.def_function[resident_release_binding]("forest_release_gpu")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_trees_host: ", error))
