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
take the per-row class weights (the weighted BOOTSTRAP, and since
2026-09-27 weights without a bootstrap, the weighted objective);
and `forest_prepare_gpu`, `forest_predict_resident_reuse_gpu` and
`forest_release_gpu`, the resident `parallel_groves` entries, predict over
`core/forest_host_groves.mojo` (`RF_INPUT=True`: the input flushed).
ADDED 2026-10-02 (lane/fix-dart-host): the data session entries
`rf_data_session_open`, `rf_data_session_close`,
`rf_regressor_fit_session_export`, `rf_regressor_fit_session_rows_export`
and `rf_classifier_fit_weighted_session_export`, which DART opens by default
(e3372c826); each member fit is the plain entries' body on the session's X.
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
from core.abs_sum_blocked_host import host_abs_sum_blocked
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
from ensemble.host_layout import has_nan_f32_threaded
from ensemble.nan_refusal import RF_NAN_REFUSAL
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


def _read_weights(weights_addr: Int, n_rows: Int) raises -> List[Float32]:
    """The weighted classifier's Float32 row weights (0: none), with the GPU
    binding's checks in its words (`bindings/_mojolearn_rf.mojo:363-380`);
    all-unit weights fit unweighted (an empty list)."""
    var weights = List[Float32]()
    if weights_addr == 0:
        return weights^
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
        return List[Float32]()
    return weights^


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
    var weights = _read_weights(weights_addr, n_rows)
    var x_address = _index(x_addr)
    var y_address = _index(y_addr)
    var forest: RfHostForest
    with GILReleased(Python()):
        var x = _read_x_col_major(x_address, n_rows, n_cols, ROWMAJOR)
        forest = _rf_fit_colmajor[CLASSIFIER](
            x^, y_address, n_rows, n_cols, n_classes, p, weights, tree_start
        )
    comptime if EXPORT:
        return _retain_rf_export(forest^)
    else:
        return _forest_out(forest)


def _rf_fit_colmajor[CLASSIFIER: Bool](
    var x: List[Float32],
    y_address: Int,
    n_rows: Int,
    n_cols: Int,
    n_classes: Int,
    p: RfHostParams,
    weights: List[Float32],
    tree_start: Int,
) raises -> RfHostForest:
    """The host fit on a COLUMN-major X already read: the one body `_rf_fit`
    and the data session entries below share, so a session member's forest
    is the forest the plain entry returns on the same X."""
    var forest: RfHostForest
    comptime if CLASSIFIER:
        var y = read_i32(y_address, n_rows)
        forest = rf_host_fit(
            x^, y, List[Float32](), n_rows, n_cols, n_classes, True, p,
            Float32(1.0), tree_start, weights,
        )
    else:
        var y = read_f32(y_address, n_rows)
        # `bindings/_mojolearn_rf.mojo`: the label plane's fixed-point
        # scale from the sum of label magnitudes, in the blocked binary64
        # order the device uses (`core/abs_sum_blocked`, cpu2-l6-bindings).
        var mag = host_abs_sum_blocked(Int(y.unsafe_ptr()), n_rows)
        var scale = Float32(choose_scale(mag, n_rows))
        forest = rf_host_fit(
            x^, List[Int32](), y, n_rows, n_cols, 1, False, p, scale,
            tree_start,
        )
    return forest^


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


# ---------------------------------------------------------------------------
# THE DATA SESSION, on the host (lane/fix-dart-host, 2026-10-02). The GPU
# binding's `rf_data_session_open` (bindings/_mojolearn_rf.mojo, trees-apple3)
# stages X once for the member fits of one boosted ensemble; DART opens it by
# default since e3372c826 (python/mojolearn/_expansion_trees.py `_DARTBase`).
# Its member fits are the plain fit on the staged bytes, so a member's forest
# is the forest `rf_*_fit_export` returns. Here the session holds the
# COLUMN-major X `_read_x_col_major` reads, and every member fit is
# `_rf_fit_colmajor` on a copy of it: the body the plain entries run, so the
# host column computes the bits it computes without a session. The same
# names, params and refusals as the GPU binding. One thread uses a session at
# a time.
struct RfHostSession(Movable):
    var id: Int
    var x: List[Float32]
    var n_rows: Int
    var n_cols: Int

    def __init__(out self, id: Int, var x: List[Float32], n_rows: Int, n_cols: Int):
        self.id = id
        self.x = x^
        self.n_rows = n_rows
        self.n_cols = n_cols


struct RfHostSessionRegistry(Defaultable, Movable):
    var sessions: List[RfHostSession]
    var next_id: Int

    def __init__(out self):
        self.sessions = List[RfHostSession]()
        self.next_id = 1

    def find(self, id: Int) raises -> Int:
        for i in range(len(self.sessions)):
            if self.sessions[i].id == id:
                return i
        raise Error("unknown or closed forest data session handle")


comptime RF_HOST_SESSIONS = _Global[
    StorageType=RfHostSessionRegistry,
    name="MojoRFDataSessionHost",
    init_fn=RfHostSessionRegistry.__init__,
]


def rf_data_session_open_binding(
    x_addr: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`bindings/_mojolearn_rf.mojo::rf_data_session_open_binding`: `params`
    is [n_rows, n_cols, row_major, share_tables]; `x` is float32, ROW-major
    when `row_major` is 1 and COLUMN-major otherwise, borrowed through this
    call only. Returns the session handle."""
    if len(params) != 4:
        raise Error("rf_data_session_open: params must hold 4 values")
    var n_rows = _index(params[0])
    var n_cols = _index(params[1])
    var row_major = _index(params[2]) != 0
    var share_tables = _index(params[3]) != 0
    if n_rows <= 0 or n_cols <= 0:
        raise Error("rf_data_session_open: invalid shape")
    # A host binding is IDENTICAL only, and an IDENTICAL member draws its
    # own quantile sample (the GPU binding's IDENTICAL refusal).
    if share_tables:
        raise Error(
            "rf_data_session_open: shared quantile tables are a FAST"
            " option; an IDENTICAL member draws its own sample"
        )
    var x_address = _index(x_addr)
    if has_nan_f32_threaded(f32_ptr(x_address), n_rows * n_cols):
        raise Error("rf_data_session_open: " + RF_NAN_REFUSAL)
    var x = List[Float32]()
    with GILReleased(Python()):
        x = _read_x_col_major(x_address, n_rows, n_cols, row_major)
    var reg = RF_HOST_SESSIONS.get_or_create_ptr()
    var id = reg[].next_id
    reg[].next_id += 1
    reg[].sessions.append(RfHostSession(id, x^, n_rows, n_cols))
    return PythonObject(id)


def rf_data_session_close_binding(handle: PythonObject) raises -> PythonObject:
    var reg = RF_HOST_SESSIONS.get_or_create_ptr()
    var si = reg[].find(_index(handle))
    _ = reg[].sessions.pop(si)
    return PythonObject(None)


def _session_params[CLASSIFIER: Bool](
    entry: String, params: PythonObject, criterion: PythonObject
) raises -> RfHostParams:
    """The GPU session entries' slot checks, in their words."""
    if len(params) != N_RF_FIT_PARAMS:
        raise Error(
            entry + ": params must hold " + String(N_RF_FIT_PARAMS)
            + " values, got " + String(len(params))
        )
    comptime if CLASSIFIER:
        if _index(params[2]) < 2:
            raise Error(entry + ": n_classes must be >= 2")
    else:
        if _index(params[2]) != 0:
            raise Error(entry + ": n_classes (slot 2) must be 0")
    var crit = _index(criterion)
    _check_criterion(entry, crit, CLASSIFIER)
    return _params_from(params, crit)


def _session_x(entry: String, session_id: Int, n_rows: Int, n_cols: Int) raises -> List[Float32]:
    """A copy of the session's column-major X (the fit consumes its X), after
    `bindings/_mojolearn_rf.mojo::_session_shape_check`."""
    var reg = RF_HOST_SESSIONS.get_or_create_ptr()
    var si = reg[].find(session_id)
    var sr = reg[].sessions[si].n_rows
    var sc = reg[].sessions[si].n_cols
    if sr != n_rows or sc != n_cols:
        raise Error(
            entry + ": params name " + String(n_rows) + " x " + String(n_cols)
            + ", the session holds " + String(sr) + " x " + String(sc)
        )
    return reg[].sessions[si].x.copy()


def rf_regressor_fit_session_binding(
    handle: PythonObject, y_addr: PythonObject,
    params: PythonObject, criterion: PythonObject,
) raises -> PythonObject:
    """`rf_regressor_fit_export` on the session's X."""
    comptime entry = "rf_regressor_fit_session"
    var p = _session_params[False](entry, params, criterion)
    var n_rows = _index(params[0])
    var n_cols = _index(params[1])
    var x = _session_x(entry, _index(handle), n_rows, n_cols)
    var y_address = _index(y_addr)
    var forest: RfHostForest
    with GILReleased(Python()):
        forest = _rf_fit_colmajor[False](
            x^, y_address, n_rows, n_cols, 0, p, List[Float32](), 0
        )
    return _retain_rf_export(forest^)


def rf_regressor_fit_session_rows_binding(
    handle: PythonObject, rows_addr: PythonObject, y_addr: PythonObject,
    params: PythonObject, criterion: PythonObject,
) raises -> PythonObject:
    """`rf_regressor_fit_export` on the rows `rows` (int32 row ids of the
    session's X, `params[0]` of them, repeats allowed), `y` their labels in
    that order: the column-major gather of those rows (the bytes the GPU
    binding's `launch_gather_rows_colmajor` writes), then the plain fit."""
    comptime entry = "rf_regressor_fit_session_rows"
    var p = _session_params[False](entry, params, criterion)
    var n_rows = _index(params[0])
    var n_cols = _index(params[1])
    if n_rows <= 0:
        raise Error(entry + ": no rows")
    var rp = i32_ptr(_index(rows_addr))
    var reg = RF_HOST_SESSIONS.get_or_create_ptr()
    var si = reg[].find(_index(handle))
    var src_rows = reg[].sessions[si].n_rows
    if reg[].sessions[si].n_cols != n_cols:
        raise Error(entry + ": the session holds another column count")
    for i in range(n_rows):
        var r = Int(rp[i])
        if r < 0 or r >= src_rows:
            raise Error(entry + ": a row id is outside the session's X")
    var x = List[Float32](length=n_rows * n_cols, fill=Float32(0.0))
    for c in range(n_cols):
        for i in range(n_rows):
            x[c * n_rows + i] = reg[].sessions[si].x[c * src_rows + Int(rp[i])]
    var y_address = _index(y_addr)
    var forest: RfHostForest
    with GILReleased(Python()):
        forest = _rf_fit_colmajor[False](
            x^, y_address, n_rows, n_cols, 0, p, List[Float32](), 0
        )
    return _retain_rf_export(forest^)


def rf_classifier_fit_weighted_session_binding(
    handle: PythonObject, y_addr: PythonObject,
    params: PythonObject, criterion: PythonObject, weights_addr: PythonObject,
) raises -> PythonObject:
    """`rf_classifier_fit_weighted_export` on the session's X."""
    comptime entry = "rf_classifier_fit_weighted_session"
    var p = _session_params[True](entry, params, criterion)
    var n_rows = _index(params[0])
    var n_cols = _index(params[1])
    var n_classes = _index(params[2])
    var w_address = _index(weights_addr)
    if w_address == 0:
        raise Error("weighted RF requires a nonzero Float32 weight pointer")
    var weights = _read_weights(w_address, n_rows)
    var x = _session_x(entry, _index(handle), n_rows, n_cols)
    var y_address = _index(y_addr)
    var forest: RfHostForest
    with GILReleased(Python()):
        forest = _rf_fit_colmajor[True](
            x^, y_address, n_rows, n_cols, n_classes, p, weights, 0
        )
    return _retain_rf_export(forest^)


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
        module.def_function[rf_data_session_open_binding]("rf_data_session_open")
        module.def_function[rf_data_session_close_binding]("rf_data_session_close")
        module.def_function[rf_regressor_fit_session_binding]("rf_regressor_fit_session_export")
        module.def_function[rf_regressor_fit_session_rows_binding]("rf_regressor_fit_session_rows_export")
        module.def_function[rf_classifier_fit_weighted_session_binding]("rf_classifier_fit_weighted_session_export")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_rf_host: ", error))
