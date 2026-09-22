# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for the `_mojolearn_rf` family, the RandomForest classifier and
regressor (workstream E batch 3, rf-clf and rf-reg, 2026-09-14).

HOST ONLY. No DeviceContext, no kernel launch, no GPU. The fit is
`ensemble/host/rf_oracle.mojo::rf_host_fit`, the device trainer
(`ensemble/randomforest.mojo::fit_forest` and the batched-level builder)
restated on the host from its kernels, so the five model arrays are meant to
be the GPU columns' bytes. The predict entries are
`core/forest_host_predict.mojo::rf_host_trees` and `rf_host_predict`, the
walk the forest host binding runs and the forest host gate holds to the
recorded GPU predictions on seven CPUs, exported here under the GPU
binding's `rf_predict_proba` and `rf_predict_reg` names and address
contract, so a fitted or loaded model predicts through them on a CPU-only
install (the infer column and the model column's reload check).

THE EXPORTED NAMES ARE THE GPU BINDING'S NAMES for what this covers, so
`python/mojolearn/randomforest.py` runs unchanged on a CPU-only install
through `_backend._HOST_MODULES` (`"_mojolearn_rf": "_mojolearn_rf_host"`):
`rf_classifier_fit`, `rf_regressor_fit`, their `_export`, `_rowmajor` and
`_rowmajor_export` forms (the 16-slot params list and the criterion
argument of `bindings/_mojolearn_rf.mojo`), `forest_export`,
`forest_export_legacy`, `forest_export_release`, `rf_predict_proba`,
`rf_predict_reg`, `rf_vendor` answering "cpu" and `rf_numeric_mode`.
ADDED 2026-09-15 (rf-reg-poisson, rf-reg-gamma-ig,
rf-clf-balanced-parallel): the POISSON, GAMMA and INVERSE_GAUSSIAN criteria
fit through the oracle; `rf_classifier_fit_weighted` and its `_export` form
take the per-row class weights (the weighted BOOTSTRAP; weights without a
bootstrap reach the weighted objective, which the oracle refuses by name);
and `forest_prepare_gpu`, `forest_predict_resident_reuse_gpu` and
`forest_release_gpu`, the resident `parallel_groves` entries, predict over
`core/forest_host_groves.mojo` (`RF_INPUT=True`: the input flushed).
ABSENT, and so refused BY NAME through `_HostBinding`: the global tree-ID
shard fits (`rf_*_fit_shard`, the multi-GPU driver's), the non-resident
`rf_predict_*_gpu_parallel` and the pool and comparison entries.

WHY A FAMILY OF ITS OWN AND NOT THE FOREST HOST BINDING. The batch 3 brief
named `bindings/_mojolearn_forest_host.mojo` as the home. That binding is
loaded BY PATH and exports `forest_host_*` names under a different address
contract, and `_backend` deliberately does not route `_mojolearn_rf` to it
(its `_HOST_MODULES` comment says why). Routing the RandomForest classes on
a CPU-only install needs a binding that exports the GPU binding's own names,
which is exactly the shape the Extra Trees lane took with the `trees`
family, so this is the `rf` family, gated by the routed set's
`-D MOJOLEARN_HOST_SABOTAGE=1`.

The sabotage arm (`rf_host_sabotage`) is
`ensemble/host/rf_oracle.mojo::RF_ORACLE_HOST_SABOTAGE`: every bootstrap row
is drawn from the next Philox subsequence, so every forest this binary fits
differs.
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
from bindings.hostptr import f32_ptr, i32_ptr, read_f32, read_i32
from checks.fixed_point import choose_scale
from checks.kernel_matrix import (
    COLUMN_CPU,
    TARGET_COLUMN,
    column_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.forest_host_predict import rf_host_predict, rf_host_trees
from bindings.forest_host_groves_binding import (
    forest_predict_resident_host_binding,
    forest_prepare_host_binding,
    forest_release_host_binding,
)
from ensemble.host.rf_oracle import (
    RF_ENTROPY,
    RF_GAMMA,
    RF_GINI,
    RF_INVERSE_GAUSSIAN,
    RF_MSE,
    RF_ORACLE_HOST_SABOTAGE,
    RF_POISSON,
    RfHostForest,
    RfHostParams,
    rf_host_fit,
)


comptime N_RF_FIT_PARAMS = 16
"""`bindings/_mojolearn_rf.mojo::N_RF_FIT_PARAMS`, the same 16 slots in the
same order (that docstring is the contract)."""

comptime RF_HOST_EXPORTS = _Global[
    StorageType=ForestExportRegistry[RfHostForest],
    name="MojoRFFitExportHost",
    init_fn=ForestExportRegistry[RfHostForest].__init__,
]


def _index(value: PythonObject) raises -> Int:
    var type_name = String(py=value.__class__.__name__)
    if type_name == "bool" or type_name == "bool_":
        raise Error("rf host: integers expected, not booleans")
    var operator_module = Python.import_module("operator")
    return Int(py=operator_module.index(value))


def rf_host_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def rf_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def rf_host_column_binding() raises -> PythonObject:
    """`column_name(TARGET_COLUMN)`, the comptime assert's witness: "cpu".
    THE COLUMN IS THE CPU COLUMN, OR THIS DOES NOT BUILD; the assert lives
    in a function body PyInit registers, so it is compiled in every build.
    `bindings/build_host_family.sh` passes -D MOJOLEARN_COLUMN_CPU."""
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "rf host: this binding compiles the CPU column only; pass"
        " -D MOJOLEARN_COLUMN_CPU (bindings/build_rf_host.sh does)"
    )
    return PythonObject(column_name(TARGET_COLUMN))


# There is no `rf_host_detected_column` read-back, for the reason 8d16ce2f
# removed it from the forest and byte LM host bindings: the detected column
# folds to the GPU of the machine that ran the build. The comptime assert
# above is the check.


def rf_host_sabotage_binding() raises -> PythonObject:
    """Whether this binary draws every bootstrap row from the next Philox
    subsequence on purpose (-D MOJOLEARN_HOST_SABOTAGE=1, the gate's
    negative control; `ensemble/host/rf_oracle.mojo::RF_ORACLE_HOST_SABOTAGE`)."""
    return PythonObject(RF_ORACLE_HOST_SABOTAGE)


# The GPU binding's names, same contract.


def rf_vendor_binding() raises -> PythonObject:
    """"cpu". On a CPU-only install `_backend.vendor()` is "cpu" and the
    read-back cross-check expects that string from every host binding."""
    return PythonObject(String("cpu"))


def rf_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))


def _check_criterion(who: String, criterion: Int, classification: Bool) raises:
    """`bindings/_mojolearn_rf.mojo::_check_criterion` (DEVIATION 407), the
    same accepted sets."""
    if classification:
        if criterion == RF_GINI or criterion == RF_ENTROPY:
            return
        raise Error(
            who + ": split criterion " + String(criterion)
            + " has no arm in this objective's GainPerSplit"
            " (objectives.mojo, DEVIATION 407); accepted: GINI/ENTROPY"
        )
    if (
        criterion == RF_MSE
        or criterion == RF_POISSON
        or criterion == RF_GAMMA
        or criterion == RF_INVERSE_GAUSSIAN
    ):
        return
    raise Error(
        who + ": split criterion " + String(criterion)
        + " has no arm in this objective's GainPerSplit"
        " (objectives.mojo, DEVIATION 407); accepted:"
        " MSE/POISSON/GAMMA/INVERSE_GAUSSIAN"
    )


def _params_from(params: PythonObject, criterion: Int) raises -> RfHostParams:
    """`bindings/_mojolearn_rf.mojo::_rf_params_from`, slots 3-15, the same
    narrowings (`Int32(Int(...))`, `Float32(Float64(...))`)."""
    return RfHostParams(
        n_trees=Int(Int32(_index(params[3]))),
        max_depth=Int(Int32(_index(params[4]))),
        max_leaves=Int(Int32(_index(params[5]))),
        max_features=Float32(Float64(py=params[6])),
        max_n_bins=Int(Int32(_index(params[7]))),
        min_samples_leaf=Int(Int32(_index(params[8]))),
        min_samples_split=Int(Int32(_index(params[9]))),
        min_impurity_decrease=Float32(Float64(py=params[10])),
        bootstrap=_index(params[11]) != 0,
        max_samples=Float32(Float64(py=params[12])),
        seed=UInt64(_index(params[13])),
        n_streams=Int(Int32(_index(params[14]))),
        max_batch_size=Int(Int32(_index(params[15]))),
        criterion=criterion,
    )


def _read_x_col_major(
    x_addr: Int, n_rows: Int, n_cols: Int, row_major: Bool
) raises -> List[Float32]:
    """The design as the trainer reads it, COLUMN-major. A ROW-major X (the
    `_rowmajor` entries, DEVIATION 2637) is transposed here: the same
    column-major bytes the GPU binding's pinned stage holds."""
    var flat = read_f32(x_addr, n_rows * n_cols)
    if not row_major:
        return flat^
    var out = List[Float32](length=n_rows * n_cols, fill=Float32(0.0))
    for r in range(n_rows):
        for c in range(n_cols):
            out[c * n_rows + r] = flat[r * n_cols + c]
    return out^


def _forest_out(forest: RfHostForest) raises -> PythonObject:
    """`bindings/_mojolearn_rf.mojo::_forest_out_trees`, the same six lists."""
    var offsets = Python.list()
    var colid = Python.list()
    var quesval = Python.list()
    var left_child = Python.list()
    var leaves = Python.list()
    for t in range(len(forest.offsets)):
        offsets.append(PythonObject(Int(forest.offsets[t])))
    for i in range(forest.n_nodes()):
        colid.append(PythonObject(Int(forest.colid[i])))
        quesval.append(PythonObject(Float64(forest.quesval[i])))
        left_child.append(PythonObject(Int(forest.left_child[i])))
    for i in range(len(forest.leaves)):
        leaves.append(PythonObject(Float64(forest.leaves[i])))
    var meta = Python.list()
    meta.append(PythonObject(forest.n_trees))
    meta.append(PythonObject(forest.num_outputs))
    var out = Python.list()
    out.append(offsets)
    out.append(colid)
    out.append(quesval)
    out.append(left_child)
    out.append(leaves)
    out.append(meta)
    return out


def _retain_rf_export(var forest: RfHostForest) raises -> PythonObject:
    """`bindings/_mojolearn_rf.mojo::_retain_rf_export`: the fitted forest kept
    under a handle until `forest_export_release`."""
    var trees = forest.n_trees
    var nodes = forest.n_nodes()
    var outputs = forest.num_outputs
    if trees < 1:
        raise Error("cannot export an empty fitted RF")
    if len(forest.leaves) != nodes * outputs:
        raise Error("fitted RF leaf storage differs from export dimensions")
    var meta: List[Int64] = [Int64(trees), Int64(outputs)]
    return RF_HOST_EXPORTS.get_or_create_ptr()[].insert(
        forest^, trees, nodes, outputs, meta^
    )


def _rf_fit[
    CLASSIFIER: Bool, EXPORT: Bool, ROWMAJOR: Bool
](
    x_addr: PythonObject,
    y_addr: PythonObject,
    params: PythonObject,
    criterion: PythonObject,
    weights_addr: Int = 0,
    tree_start: Int = 0,
) raises -> PythonObject:
    """`_rf_classifier_fit` / `_rf_regressor_fit` of the GPU binding: the same
    slot checks in the same words, then the host fit. `weights_addr` is the
    weighted classifier's Float32 row weights (0: none); `tree_start` the
    GPU binding's global tree ID offset (the shard fits below)."""
    comptime entry = "rf_classifier_fit" if CLASSIFIER else "rf_regressor_fit"
    if len(params) != N_RF_FIT_PARAMS:
        raise Error(
            entry + ": params must hold " + String(N_RF_FIT_PARAMS)
            + " values, got " + String(len(params))
        )
    var n_rows = _index(params[0])
    var n_cols = _index(params[1])
    var n_classes = _index(params[2])
    comptime if CLASSIFIER:
        if n_classes < 2:
            raise Error("rf_classifier_fit: n_classes must be >= 2")
    else:
        if n_classes != 0:
            raise Error("rf_regressor_fit: n_classes (slot 2) must be 0")
    if n_rows < 1 or n_cols < 1:
        raise Error(entry + ": n_rows and n_cols must be >= 1")
    var crit = _index(criterion)
    _check_criterion(entry, crit, CLASSIFIER)
    var p = _params_from(params, crit)
    # `bindings/_mojolearn_rf.mojo:363-380`: the weights' checks in their
    # words; all-unit weights fit unweighted.
    var weights = List[Float32]()
    if weights_addr != 0:
        var wp = f32_ptr(weights_addr)
        var total = Float64(0)
        var all_unit = True
        for i in range(n_rows):
            var w = wp[i]
            if not (w >= 0 and w <= Float32(3.4028234663852886e38)):
                raise Error("class weights must be finite and nonnegative")
            weights.append(w)
            total += Float64(w)
            all_unit = all_unit and w == Float32(1)
        if total <= 0:
            raise Error("class weights must have positive total")
        if all_unit:
            weights = List[Float32]()
    var x_address = _index(x_addr)
    var y_address = _index(y_addr)
    var forest: RfHostForest
    with GILReleased(Python()):
        var x = _read_x_col_major(x_address, n_rows, n_cols, ROWMAJOR)
        comptime if CLASSIFIER:
            var y = read_i32(y_address, n_rows)
            forest = rf_host_fit(
                x^, y, List[Float32](), n_rows, n_cols, n_classes, True, p,
                Float32(1.0), tree_start, weights,
            )
        else:
            var y = read_f32(y_address, n_rows)
            # `bindings/_mojolearn_rf.mojo:594-600`: the label plane's
            # fixed-point scale from the sum of label magnitudes, in their
            # Float64 order.
            var mag = Float64(0.0)
            for i in range(n_rows):
                var v = Float64(y[i])
                mag += v if v >= 0.0 else -v
            var scale = Float32(choose_scale(mag, n_rows))
            forest = rf_host_fit(
                x^, List[Int32](), y, n_rows, n_cols, 1, False, p, scale,
                tree_start,
            )
    comptime if EXPORT:
        return _retain_rf_export(forest^)
    else:
        return _forest_out(forest)


def rf_classifier_fit_binding(
    x_addr: PythonObject, y_addr: PythonObject,
    params: PythonObject, criterion: PythonObject,
) raises -> PythonObject:
    return _rf_fit[True, False, False](x_addr, y_addr, params, criterion)


def rf_classifier_fit_export_binding(
    x_addr: PythonObject, y_addr: PythonObject,
    params: PythonObject, criterion: PythonObject,
) raises -> PythonObject:
    return _rf_fit[True, True, False](x_addr, y_addr, params, criterion)


def rf_classifier_fit_rowmajor_binding(
    x_addr: PythonObject, y_addr: PythonObject,
    params: PythonObject, criterion: PythonObject,
) raises -> PythonObject:
    return _rf_fit[True, False, True](x_addr, y_addr, params, criterion)


def rf_classifier_fit_rowmajor_export_binding(
    x_addr: PythonObject, y_addr: PythonObject,
    params: PythonObject, criterion: PythonObject,
) raises -> PythonObject:
    return _rf_fit[True, True, True](x_addr, y_addr, params, criterion)


def _rf_classifier_fit_weighted[EXPORT: Bool](
    x_addr: PythonObject, y_addr: PythonObject,
    params: PythonObject, criterion: PythonObject, weights_addr: PythonObject,
) raises -> PythonObject:
    """`bindings/_mojolearn_rf.mojo::rf_classifier_fit_weighted_binding`."""
    var address = _index(weights_addr)
    if address == 0:
        raise Error("weighted RF requires a nonzero Float32 weight pointer")
    return _rf_fit[True, EXPORT, False](x_addr, y_addr, params, criterion, address)


def rf_classifier_fit_weighted_binding(
    x_addr: PythonObject, y_addr: PythonObject,
    params: PythonObject, criterion: PythonObject, weights_addr: PythonObject,
) raises -> PythonObject:
    return _rf_classifier_fit_weighted[False](x_addr, y_addr, params, criterion, weights_addr)


def rf_classifier_fit_weighted_export_binding(
    x_addr: PythonObject, y_addr: PythonObject,
    params: PythonObject, criterion: PythonObject, weights_addr: PythonObject,
) raises -> PythonObject:
    return _rf_classifier_fit_weighted[True](x_addr, y_addr, params, criterion, weights_addr)


def rf_regressor_fit_binding(
    x_addr: PythonObject, y_addr: PythonObject,
    params: PythonObject, criterion: PythonObject,
) raises -> PythonObject:
    return _rf_fit[False, False, False](x_addr, y_addr, params, criterion)


def rf_regressor_fit_export_binding(
    x_addr: PythonObject, y_addr: PythonObject,
    params: PythonObject, criterion: PythonObject,
) raises -> PythonObject:
    return _rf_fit[False, True, False](x_addr, y_addr, params, criterion)


def rf_regressor_fit_rowmajor_binding(
    x_addr: PythonObject, y_addr: PythonObject,
    params: PythonObject, criterion: PythonObject,
) raises -> PythonObject:
    return _rf_fit[False, False, True](x_addr, y_addr, params, criterion)


def rf_regressor_fit_rowmajor_export_binding(
    x_addr: PythonObject, y_addr: PythonObject,
    params: PythonObject, criterion: PythonObject,
) raises -> PythonObject:
    return _rf_fit[False, True, True](x_addr, y_addr, params, criterion)


def rf_classifier_fit_shard_binding(
    x_addr: PythonObject, y_addr: PythonObject,
    params: PythonObject, criterion: PythonObject, tree_start: PythonObject,
    weights_addr: PythonObject,
) raises -> PythonObject:
    """`bindings/_mojolearn_rf.mojo::rf_classifier_fit_shard_binding`: the
    column-major fit of trees `tree_start .. tree_start + n_estimators` of
    the whole forest, weighted when `weights_addr` is nonzero
    (`parallel_ensemble.fit_forest`'s shard; lane/cpu-training-par-wave2,
    2026-09-15)."""
    return _rf_fit[True, False, False](
        x_addr, y_addr, params, criterion, _index(weights_addr), _index(tree_start)
    )


def rf_regressor_fit_shard_binding(
    x_addr: PythonObject, y_addr: PythonObject,
    params: PythonObject, criterion: PythonObject, tree_start: PythonObject,
) raises -> PythonObject:
    """`bindings/_mojolearn_rf.mojo::rf_regressor_fit_shard_binding`."""
    return _rf_fit[False, False, False](
        x_addr, y_addr, params, criterion, 0, _index(tree_start)
    )


def rf_forest_export_binding(
    handle: PythonObject,
    offsets: PythonObject,
    columns: PythonObject,
    thresholds: PythonObject,
    left: PythonObject,
    leaves: PythonObject,
    counts: PythonObject,
) raises -> PythonObject:
    """`bindings/_mojolearn_rf.mojo::rf_forest_export_binding`: the retained
    forest into the caller's five arrays."""
    if len(counts) != 3:
        raise Error("forest_export requires trees, nodes, outputs capacities")
    var id = _index(handle)
    var registry = RF_HOST_EXPORTS.get_or_create_ptr()
    registry[].validate(id, _index(counts[0]), _index(counts[1]), _index(counts[2]))
    validate_forest_export_destinations(
        _index(offsets), _index(columns), _index(thresholds), _index(left), _index(leaves)
    )
    var op = i32_ptr(_index(offsets))
    var cp = i32_ptr(_index(columns))
    var tp = f32_ptr(_index(thresholds))
    var lp = i32_ptr(_index(left))
    ref model = registry[].entries[id].model
    for t in range(len(model.offsets)):
        op[t] = model.offsets[t]
    for i in range(model.n_nodes()):
        cp[i] = model.colid[i]
        tp[i] = model.quesval[i]
        lp[i] = model.left_child[i]
    copy_forest_export_leaves(model.leaves, _index(leaves), 0)
    return PythonObject(None)


def rf_forest_export_legacy_binding(handle: PythonObject) raises -> PythonObject:
    var registry = RF_HOST_EXPORTS.get_or_create_ptr()
    var id = _index(handle)
    if id not in registry[].entries:
        raise Error("unknown or released fitted forest export handle")
    return _forest_out(registry[].entries[id].model)


def rf_forest_export_release_binding(handle: PythonObject) raises -> PythonObject:
    RF_HOST_EXPORTS.get_or_create_ptr()[].release(_index(handle))
    return PythonObject(None)


def _rf_predict(
    offsets_addr: PythonObject,
    colid_addr: PythonObject,
    quesval_addr: PythonObject,
    left_child_addr: PythonObject,
    leaves_addr: PythonObject,
    x_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
    entry: String,
    classifier: Bool,
) raises -> PythonObject:
    """`rf_predict_proba` / `rf_predict_reg`'s contract
    (`bindings/_mojolearn_rf.mojo:696-817`): ROW-major `x`, `params` is
    `[n_rows, n_cols, n_trees, num_outputs]`, returns rows written. The
    walk is `core/forest_host_predict.mojo`'s, which rebuilds the trees
    from the five arrays and refuses a malformed node instead of reading
    past it; `n_nodes` is read off the offsets."""
    if len(params) != 4:
        raise Error(
            entry + ": params must hold [n_rows, n_cols, n_trees,"
            " num_outputs], got " + String(len(params))
        )
    var n_rows = _index(params[0])
    var n_cols = _index(params[1])
    var n_trees = _index(params[2])
    var num_outputs = _index(params[3])
    if classifier:
        if n_trees < 1 or num_outputs < 2:
            raise Error("rf_predict_proba: n_trees must be >= 1 and num_outputs >= 2")
    else:
        if n_trees < 1 or num_outputs != 1:
            raise Error("rf_predict_reg: n_trees must be >= 1 and num_outputs must be 1")
    if n_rows < 0 or n_cols < 1:
        raise Error(entry + ": n_rows must be >= 0 and n_cols >= 1")
    var offsets_p = i32_ptr(_index(offsets_addr))
    var colid_p = i32_ptr(_index(colid_addr))
    var quesval_p = f32_ptr(_index(quesval_addr))
    var left_p = i32_ptr(_index(left_child_addr))
    var leaves_p = f32_ptr(_index(leaves_addr))
    var x_address = _index(x_addr)
    var op = f32_ptr(_index(out_addr))
    var n_nodes = Int(offsets_p[n_trees])
    if Int(offsets_p[0]) != 0 or n_nodes < n_trees:
        raise Error(entry + ": tree_offsets must start at 0 and hold at least one node per tree")
    if n_rows == 0:
        return PythonObject(0)
    var wrote = 0
    with GILReleased(Python()):
        var rows = read_f32(x_address, n_rows * n_cols)
        var out = List[Float32](length=n_rows * num_outputs, fill=Float32(0.0))
        var trees = rf_host_trees(
            offsets_p, colid_p, quesval_p, left_p, leaves_p,
            n_trees, n_nodes, n_cols, num_outputs,
        )
        rf_host_predict(trees, rows, n_rows, n_cols, n_trees, num_outputs, out)
        for i in range(n_rows * num_outputs):
            op[i] = out[i]
        wrote = n_rows
    return PythonObject(wrote)


def rf_predict_proba_binding(
    offsets_addr: PythonObject, colid_addr: PythonObject,
    quesval_addr: PythonObject, left_child_addr: PythonObject,
    leaves_addr: PythonObject, x_addr: PythonObject,
    out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    return _rf_predict(
        offsets_addr, colid_addr, quesval_addr, left_child_addr, leaves_addr,
        x_addr, out_addr, params, String("rf_predict_proba"), True,
    )


def rf_predict_reg_binding(
    offsets_addr: PythonObject, colid_addr: PythonObject,
    quesval_addr: PythonObject, left_child_addr: PythonObject,
    leaves_addr: PythonObject, x_addr: PythonObject,
    out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    return _rf_predict(
        offsets_addr, colid_addr, quesval_addr, left_child_addr, leaves_addr,
        x_addr, out_addr, params, String("rf_predict_reg"), False,
    )


def resident_prepare_binding(
    offsets_addr: PythonObject, colid_addr: PythonObject,
    quesval_addr: PythonObject, left_child_addr: PythonObject,
    leaves_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """`forest_prepare_gpu`: the resident parallel_groves snapshot, on the host."""
    return forest_prepare_host_binding[True](
        offsets_addr, colid_addr, quesval_addr, left_child_addr, leaves_addr, params
    )


def resident_predict_binding(
    handle: PythonObject, x_addr: PythonObject, out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`forest_predict_resident_reuse_gpu`: the grove reduction, on the host."""
    return forest_predict_resident_host_binding[True](handle, x_addr, out_addr, params)


def resident_release_binding(handle: PythonObject) raises -> PythonObject:
    """`forest_release_gpu`."""
    return forest_release_host_binding[True](handle)


@export
def PyInit__mojolearn_rf_host() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_rf_host")
        module.def_function[rf_host_numeric_mode_binding]("rf_host_numeric_mode")
        module.def_function[rf_host_vendor_binding]("rf_host_vendor")
        module.def_function[rf_host_column_binding]("rf_host_column")
        module.def_function[rf_host_sabotage_binding]("rf_host_sabotage")
        module.def_function[rf_vendor_binding]("rf_vendor")
        module.def_function[rf_numeric_mode_binding]("rf_numeric_mode")
        module.def_function[rf_classifier_fit_binding]("rf_classifier_fit")
        module.def_function[rf_classifier_fit_export_binding]("rf_classifier_fit_export")
        module.def_function[rf_classifier_fit_rowmajor_binding]("rf_classifier_fit_rowmajor")
        module.def_function[rf_classifier_fit_rowmajor_export_binding]("rf_classifier_fit_rowmajor_export")
        module.def_function[rf_regressor_fit_binding]("rf_regressor_fit")
        module.def_function[rf_regressor_fit_export_binding]("rf_regressor_fit_export")
        module.def_function[rf_regressor_fit_rowmajor_binding]("rf_regressor_fit_rowmajor")
        module.def_function[rf_regressor_fit_rowmajor_export_binding]("rf_regressor_fit_rowmajor_export")
        module.def_function[rf_classifier_fit_shard_binding]("rf_classifier_fit_shard")
        module.def_function[rf_regressor_fit_shard_binding]("rf_regressor_fit_shard")
        module.def_function[rf_forest_export_binding]("forest_export")
        module.def_function[rf_forest_export_legacy_binding]("forest_export_legacy")
        module.def_function[rf_forest_export_release_binding]("forest_export_release")
        module.def_function[rf_predict_proba_binding]("rf_predict_proba")
        module.def_function[rf_predict_reg_binding]("rf_predict_reg")
        module.def_function[rf_classifier_fit_weighted_binding]("rf_classifier_fit_weighted")
        module.def_function[rf_classifier_fit_weighted_export_binding]("rf_classifier_fit_weighted_export")
        module.def_function[resident_prepare_binding]("forest_prepare_gpu")
        module.def_function[resident_predict_binding]("forest_predict_resident_reuse_gpu")
        module.def_function[resident_release_binding]("forest_release_gpu")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_rf_host: ", error))
