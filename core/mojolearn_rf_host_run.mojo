# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RUN of `bindings/_mojolearn_rf_host.mojo` (wave 8 run split): the host
RandomForest fit, its data sessions, exports and the sequential predict as
typed Mojo. Imports no Python. The binding keeps UNPACK, PACK and the GIL:
it releases the GIL around `_rf_fit_forest`, `_rf_predict_walk` and the
`_rf_fit_colmajor` member fits, and holds it for the registry steps (the
session and export registries are serialized by the GIL)."""
from ensemble.host.importance import rf_host_importance
from ensemble.host.oob import rf_host_oob

from core.mojolearn_forest_host_impl import forest_host_vote_into
from core.mojolearn_rf_host_impl import (
    RfHostSession,
    RfHostSessionRegistry,
    _apply_min_weight_fraction,
    rf_host_col_major,
    rf_host_dart_member_xy,
    rf_host_export_into,
    rf_host_forest_lists,
    rf_host_session_rows_x,
    rf_host_weights_from,
)
from core.hostptr import f32_ptr, i32_ptr, read_f32, read_i32
from core.weight_compact_host import (
    host_all_unit,
    host_gather_colmajor,
    host_gather_f32,
    host_gather_i32,
    host_nonzero_weight_rows,
)
from checks.fixed_point import choose_scale
from core.abs_sum_blocked_host import host_abs_sum_blocked
from checks.kernel_matrix import (
    COLUMN_CPU,
    TARGET_COLUMN,
    column_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.forest_host_predict import rf_host_predict, rf_host_trees
from core.forest_fit_request import ForestFitArgs, XF_CAP_STRICT, XF_CAP_DART, forest_export_verify_lists
from xtrees.dart_host import DART_HOST, DART_HOST_SESSIONS
from ensemble.nan_refusal import RF_NONFINITE_REFUSAL
from core.host_finite_scan import host_all_finite_f32
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
    n_sampled_rows_host,
    rf_host_fit,
)
from core.shared_slot import SharedSlot
from core.forest_export_run import ForestExportRegistry
from core.forest_export_binding_impl import validate_forest_export_destinations


comptime N_RF_FIT_PARAMS = 16
"""`bindings/_mojolearn_rf.mojo::N_RF_FIT_PARAMS`, the same 16 slots in the
same order (that docstring is the contract)."""


comptime RF_HOST_EXPORTS = SharedSlot[ForestExportRegistry[RfHostForest], "MojolearnRFFitExportHostV1"]


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


def _params_from(params: ForestFitArgs, criterion: Int) raises -> RfHostParams:
    """`bindings/_mojolearn_rf.mojo::_rf_params_from`, slots 3-15, the same
    narrowings (`Int32(Int(...))`, `Float32(Float64(...))`)."""
    return RfHostParams(
        n_trees=Int(Int32(params.i(3))),
        max_depth=Int(Int32(params.i(4))),
        max_leaves=Int(Int32(params.i(5))),
        max_features=Float32(params.f(6)),
        max_n_bins=Int(Int32(params.i(7))),
        min_samples_leaf=Int(Int32(params.i(8))),
        min_samples_split=Int(Int32(params.i(9))),
        min_impurity_decrease=Float32(params.f(10)),
        bootstrap=params.i(11) != 0,
        max_samples=Float32(params.f(12)),
        seed=UInt64(params.i(13)),
        n_streams=Int(Int32(params.i(14))),
        max_batch_size=Int(Int32(params.i(15))),
        criterion=criterion,
    )


def _read_x_col_major(
    x_addr: Int, n_rows: Int, n_cols: Int, row_major: Bool
) raises -> List[Float32]:
    """The design as the trainer reads it, COLUMN-major. A ROW-major X (the
    `_rowmajor` entries, DEVIATION 2637) is transposed here: the same
    column-major bytes the GPU binding's pinned stage holds."""
    var flat = read_f32(x_addr, n_rows * n_cols)
    return rf_host_col_major(flat^, n_rows, n_cols, row_major)


def _retain_rf_handle(var forest: RfHostForest) raises -> Int:
    """`bindings/_mojolearn_rf.mojo::_retain_rf_export`: the fitted forest kept
    under a handle until `forest_export_release`."""
    var trees = forest.n_trees
    var nodes = forest.n_nodes()
    var outputs = forest.num_outputs
    if trees < 1:
        raise Error("cannot export an empty fitted RF")
    if len(forest.leaves) != nodes * outputs:
        raise Error("fitted RF leaf storage differs from export dimensions")
    var meta: List[Int64] = [Int64(trees), Int64(outputs), Int64(forest.notices)]
    return RF_HOST_EXPORTS.get_or_create_ptr()[].insert_native(
        forest^, trees, nodes, outputs, meta^
    )


def _params_len_ok(params: ForestFitArgs) raises -> Bool:
    """`bindings/_mojolearn_rf.mojo::_rf_params_len_ok`: 16 slots, or 17
    with the optional `min_weight_fraction_leaf`."""
    var n = len(params)
    return n == N_RF_FIT_PARAMS or n == N_RF_FIT_PARAMS + 1


def _min_weight_fraction(params: ForestFitArgs) raises -> Float64:
    """`bindings/_mojolearn_rf.mojo::_rf_min_weight_fraction`, its words."""
    if len(params) <= N_RF_FIT_PARAMS:
        return 0.0
    var f = params.f(N_RF_FIT_PARAMS)
    if not (f >= 0.0 and f <= 0.5):
        raise Error(
            "The 'min_weight_fraction_leaf' parameter must be a float in the"
            " range [0.0, 0.5]. Got " + String(f) + " instead."
        )
    return f


def _read_weights(weights_addr: Int, n_rows: Int, noun: String = "class") raises -> List[Float32]:
    """The weighted fit's Float32 row weights (0: none), with the GPU
    binding's checks in its words (`bindings/_mojolearn_rf.mojo:363-380`;
    `noun` "sample" for the regressor's, as its device scan says);
    all-unit weights fit unweighted (an empty list)."""
    if weights_addr == 0:
        return List[Float32]()
    return rf_host_weights_from(f32_ptr(weights_addr), n_rows, noun)


def _host_drop_zero_weight_rows(
    mut x: List[Float32],
    mut weights: List[Float32],
    mut y_kept: List[Int32],
    y_address: Int,
    n_rows: Int,
    n_cols: Int,
    bootstrap: Bool,
) raises -> Int:
    """`bindings/_mojolearn_rf.mojo::_rf_drop_zero_weight_rows` on the host
    column (lane sk-weq-trees): a weighted classifier fit without bootstrap
    keeps only its nonzero-weight rows (scikit-learn's splitter), in the
    device's ascending order. `x` (column-major) and `weights` become the
    kept rows', `y_kept` the kept labels; all-1.0 kept weights become the
    empty list (the count objective), as the device rescan does. Returns
    the kept row count (n_rows, nothing touched, when no weight is zero)."""
    if len(weights) == 0 or bootstrap:
        return n_rows
    var rows = host_nonzero_weight_rows(weights)
    if len(rows) >= n_rows or len(rows) == 0:
        return n_rows
    x = host_gather_colmajor(x, n_rows, n_cols, rows)
    y_kept = host_gather_i32(read_i32(y_address, n_rows), rows)
    weights = host_gather_f32(weights, rows)
    if host_all_unit(weights):
        weights = List[Float32]()
    return len(rows)


def _host_drop_zero_weight_rows_reg(
    mut x: List[Float32],
    mut weights: List[Float32],
    mut y_kept: List[Float32],
    y_address: Int,
    n_rows: Int,
    n_cols: Int,
    bootstrap: Bool,
) raises -> Int:
    """`_host_drop_zero_weight_rows` for the weighted regressor (lane
    sk-last): the device's `_rf_drop_zero_weight_rows[RLT]` on the host
    column, the float32 targets gathered with the kept rows."""
    if len(weights) == 0 or bootstrap:
        return n_rows
    var rows = host_nonzero_weight_rows(weights)
    if len(rows) >= n_rows or len(rows) == 0:
        return n_rows
    x = host_gather_colmajor(x, n_rows, n_cols, rows)
    y_kept = host_gather_f32(read_f32(y_address, n_rows), rows)
    weights = host_gather_f32(weights, rows)
    if host_all_unit(weights):
        weights = List[Float32]()
    return len(rows)


def _rf_fit_colmajor[CLASSIFIER: Bool](
    var x: List[Float32],
    y_address: Int,
    n_rows: Int,
    n_cols: Int,
    n_classes: Int,
    p: RfHostParams,
    weights: List[Float32],
    tree_start: Int,
    retain_importance: Bool = False,
    min_weight_leaf: Float32 = Float32(0),
    weight_max: Float64 = 0,
) raises -> RfHostForest:
    """The host fit on a COLUMN-major X already read: the one body `_rf_fit`
    and the data session entries below share, so a session member's forest
    is the forest the plain entry returns on the same X."""
    var forest: RfHostForest
    comptime if CLASSIFIER:
        var y = read_i32(y_address, n_rows)
        forest = rf_host_fit(
            x^, y, List[Float32](), n_rows, n_cols, n_classes, True, p,
            Float32(1.0), tree_start, weights, retain_importance,
            min_weight_leaf, notices_to_stderr=False,
        )
    else:
        var y = read_f32(y_address, n_rows)
        # `bindings/_mojolearn_rf.mojo`: the label plane's fixed-point
        # scale from the sum of label magnitudes, in the blocked binary64
        # order the device uses (`core/abs_sum_blocked`, cpu2-l6-bindings).
        var mag = host_abs_sum_blocked(Int(y.unsafe_ptr()), n_rows)
        var scale = Float32(choose_scale(mag, n_rows))
        if len(weights) > 0 and not p.bootstrap:
            # the weighted objective (lane sk-last; `rf_regressor_fit_weighted`):
            # `label_sum` holds label * weight, its scale sized by
            # sum |y| * max w, as the device binding chooses it
            scale = Float32(choose_scale(mag * max(weight_max, Float64(0)), n_rows))
        forest = rf_host_fit(
            x^, List[Int32](), y, n_rows, n_cols, 1, False, p, scale,
            tree_start, weights, retain_importance=retain_importance,
            min_weight_leaf=min_weight_leaf, notices_to_stderr=False,
        )
    return forest^


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
comptime RF_HOST_SESSIONS = SharedSlot[RfHostSessionRegistry, "MojolearnRFDataSessionHostV1"]


def _session_params[CLASSIFIER: Bool](
    entry: String, params: ForestFitArgs
) raises -> RfHostParams:
    """The GPU session entries' slot checks, in their words."""
    if not _params_len_ok(params):
        raise Error(
            entry + ": params must hold " + String(N_RF_FIT_PARAMS)
            + " values (17 with min_weight_fraction_leaf), got "
            + String(len(params))
        )
    comptime if CLASSIFIER:
        if params.i(2) < 2:
            raise Error(entry + ": n_classes must be >= 2")
    else:
        if params.i(2) != 0:
            raise Error(entry + ": n_classes (slot 2) must be 0")
    var crit = params.criterion
    _check_criterion(entry, crit, CLASSIFIER)
    var p = _params_from(params, crit)
    comptime if not CLASSIFIER:
        # the regressor sessions fit unweighted: slot 16 is a count bound
        # (the weighted classifier session applies it with its weights)
        _ = _apply_min_weight_fraction(
            p, _min_weight_fraction(params), List[Float32](), params.i(0)
        )
    return p


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


def _rf_export_into_native(id: Int, dst: List[Int], trees: Int, nodes: Int, outputs: Int) raises:
    """XF_EXPORT_INTO: the retained forest into the caller's five arrays."""
    var registry = RF_HOST_EXPORTS.get_or_create_ptr()
    registry[].validate(id, trees, nodes, outputs)
    validate_forest_export_destinations(dst[0], dst[1], dst[2], dst[3], dst[4])
    var op = i32_ptr(dst[0])
    var cp = i32_ptr(dst[1])
    var tp = f32_ptr(dst[2])
    var lp = i32_ptr(dst[3])
    rf_host_export_into(registry[].entries[id].model, op, cp, tp, lp, dst[4])


def _rf_export_verify_native(id: Int, dst: List[Int]) raises -> Int:
    """XF_EXPORT_VERIFY (MOJOLEARN_FOREST_EXPORT=verify): the diagnostic
    lists `forest_export_legacy` returns, against the exported arrays."""
    var registry = RF_HOST_EXPORTS.get_or_create_ptr()
    if id not in registry[].entries:
        raise Error("unknown or released fitted forest export handle")
    ref entry = registry[].entries[id]
    if (len(entry.meta) != 3 or Int(entry.meta[0]) != entry.model.n_trees
            or Int(entry.meta[1]) != entry.model.num_outputs or Int(entry.meta[2]) != entry.model.notices):
        return -1
    var offsets = List[Int64]()
    var colid = List[Int64]()
    var quesval = List[Float64]()
    var left_child = List[Int64]()
    var leaves = List[Float64]()
    rf_host_forest_lists(entry.model, offsets, colid, quesval, left_child, leaves)
    return forest_export_verify_lists(offsets, colid, quesval, left_child, leaves, dst)


def _rf_fit_forest[CLASSIFIER: Bool, ROWMAJOR: Bool](params: ForestFitArgs) raises -> RfHostForest:
    """The typed body of `_rf_fit` (wave 8 run split), run by the
    binding with the GIL released: the slot checks, the fit, the OOB and
    importance outputs. The binding keeps the fit (`_retain_rf_handle`,
    under the GIL)."""
    comptime entry = "rf_classifier_fit" if CLASSIFIER else "rf_regressor_fit"
    var weights_addr = params.weights
    var tree_start = params.tree_start
    var oob_addr = params.oob
    var oob_score_addr = params.score
    var importance_addr = params.importance
    var weight_max = params.weight_max
    if (oob_addr == 0) != (oob_score_addr == 0):
        raise Error("RF OOB output and score must be requested together")
    if not _params_len_ok(params):
        raise Error(
            entry + ": params must hold " + String(N_RF_FIT_PARAMS)
            + " values (17 with min_weight_fraction_leaf), got "
            + String(len(params))
        )
    var n_rows = params.i(0)
    var n_cols = params.i(1)
    var n_classes = params.i(2)
    comptime if CLASSIFIER:
        if n_classes < 2:
            raise Error("rf_classifier_fit: n_classes must be >= 2")
    else:
        if n_classes != 0:
            raise Error("rf_regressor_fit: n_classes (slot 2) must be 0")
    if n_rows < 1 or n_cols < 1:
        raise Error(entry + ": n_rows and n_cols must be >= 1")
    var crit = params.criterion
    _check_criterion(entry, crit, CLASSIFIER)
    var p = _params_from(params, crit)
    if oob_addr != 0 and (oob_score_addr == 0 or not p.bootstrap or tree_start != 0):
        raise Error("RF OOB requires bootstrap, score storage, and an unsharded forest")
    # `bindings/_mojolearn_rf.mojo:363-380`: the weights' checks in their
    # words; all-unit weights fit unweighted.
    var weights = _read_weights(weights_addr, n_rows, String("class") if CLASSIFIER else String("sample"))
    var x_address = params.x
    var y_address = params.y
    # sk-weq-trees: a weighted classifier fit without bootstrap drops its
    # zero-weight rows (`_host_drop_zero_weight_rows`); n_fit is the kept count
    var x = _read_x_col_major(x_address, n_rows, n_cols, ROWMAJOR)
    var y_kept = List[Int32]()
    var y_kept_f = List[Float32]()
    var n_fit = n_rows
    comptime if CLASSIFIER:
        n_fit = _host_drop_zero_weight_rows(
            x, weights, y_kept, y_address, n_rows, n_cols, p.bootstrap
        )
    else:
        n_fit = _host_drop_zero_weight_rows_reg(
            x, weights, y_kept_f, y_address, n_rows, n_cols, p.bootstrap
        )
    var y_fit = y_address
    if n_fit != n_rows:
        comptime if CLASSIFIER:
            y_fit = Int(y_kept.unsafe_ptr())
        else:
            y_fit = Int(y_kept_f.unsafe_ptr())
    var mwl = _apply_min_weight_fraction(p, _min_weight_fraction(params), weights, n_fit)
    var forest: RfHostForest
    forest = _rf_fit_colmajor[CLASSIFIER](
        x^, y_fit, n_fit, n_cols, n_classes, p, weights, tree_start, importance_addr != 0,
        mwl, weight_max,
    )
    if oob_addr != 0:
        rf_host_oob[CLASSIFIER](forest,p,f32_ptr(x_address),y_address,weights,n_rows,n_cols,oob_addr,oob_score_addr)
    _ = y_kept^
    _ = y_kept_f^
    if importance_addr != 0:
        rf_host_importance(forest,n_cols,importance_addr)
    return forest^


def _rf_predict_walk(model: List[Int], x_address: Int, out_addr: Int, n_rows: Int, n_cols: Int, n_trees: Int,
                       num_outputs: Int, classifier: Bool) raises -> Int:
    """The typed body of `_rf_predict_native` (wave 8 run split),
    run by the binding with the GIL released. `rf_predict_proba` / `rf_predict_reg`'s contract (XF_PREDICT): ROW-major
    `x`, returns rows written. The walk is `core/forest_host_predict.mojo`'s,
    which rebuilds the trees from the five arrays and refuses a malformed
    node instead of reading past it; `n_nodes` is read off the offsets."""
    var entry = String("rf_predict_proba") if classifier else String("rf_predict_reg")
    if classifier:
        if n_trees < 1 or num_outputs < 2:
            raise Error("rf_predict_proba: n_trees must be >= 1 and num_outputs >= 2")
    else:
        if n_trees < 1 or num_outputs != 1:
            raise Error("rf_predict_reg: n_trees must be >= 1 and num_outputs must be 1")
    if n_rows < 0 or n_cols < 1:
        raise Error(entry + ": n_rows must be >= 0 and n_cols >= 1")
    var offsets_p = i32_ptr(model[0])
    var colid_p = i32_ptr(model[1])
    var quesval_p = f32_ptr(model[2])
    var left_p = i32_ptr(model[3])
    var leaves_p = f32_ptr(model[4])
    var op = f32_ptr(out_addr)
    var n_nodes = Int(offsets_p[n_trees])
    if Int(offsets_p[0]) != 0 or n_nodes < n_trees:
        raise Error(entry + ": tree_offsets must start at 0 and hold at least one node per tree")
    if n_rows == 0:
        return 0
    var rows = read_f32(x_address, n_rows * n_cols)
    forest_host_vote_into(
        True, rows, offsets_p, colid_p, quesval_p, left_p,
        leaves_p, op, n_rows, n_cols, n_trees, num_outputs, n_nodes,
    )
    return n_rows
