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
    RfHostSessionLease,
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
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock
from core.forest_export_run import ForestExportRegistry
from core.forest_export_binding_impl import validate_forest_export_destinations
from core.forest_fit_request import XF_CAP_SESSION
from core.forest_entry import ForestImage, forest_entry_dispatch, forest_entry_publish
from core.forest_entry_table import FOREST_TABLE_RF
from core.native_descriptor_abi import NativeDescriptorError
from core.forest_host_strict_run import (
    forest_host_strict_predict, forest_strict_prepare_host_native, forest_strict_release_host_native,
)
from core.forest_host_groves_run import (
    forest_prepare_host_native, forest_predict_resident_host_native, forest_release_host_native,
)


comptime N_RF_FIT_PARAMS = 16
"""`bindings/_mojolearn_rf.mojo::N_RF_FIT_PARAMS`, the same 16 slots in the
same order (that docstring is the contract)."""


# V2 (lane fx-gil, 2026-11-25): the registry gained its lock word.
# V3 (lane pv-race, 2026-11-22): the shared table moved behind a pointer (core/forest_export_run.mojo).
comptime RF_HOST_EXPORTS = SharedSlot[ForestExportRegistry[RfHostForest], "MojolearnRFFitExportHostV4", 3]


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
# a time; different sessions run in parallel on threads that hold no GIL
# (lane mf-ada2: the registry's lock word, the `rf_host_session_*` steps).
# Layout version 2 (the lock word, the leased X).
comptime RF_HOST_SESSIONS = SharedSlot[RfHostSessionRegistry, "MojolearnRFDataSessionHostV2", 2]


def _host_sessions_word() raises -> MutPointer[Int64, MutUntrackedOrigin]:
    var reg = RF_HOST_SESSIONS.get_or_create_ptr()
    return MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(reg[].lock.unsafe_ptr()))


def _host_session_index(id: Int) raises -> Int:
    """The index of session `id`; the caller holds the lock."""
    var reg = RF_HOST_SESSIONS.get_or_create_ptr()
    for i in range(len(reg[].sessions)):  # small-loop(sessions: open data sessions of this process): handle lookup, not data
        if reg[].sessions[i].id == id:
            return i
    raise Error("unknown or closed forest data session handle")


def rf_host_session_keep(var x: List[Float32], n_rows: Int, n_cols: Int) raises -> Int:
    """A session over the column-major X `x`, kept under a new handle."""
    var reg = RF_HOST_SESSIONS.get_or_create_ptr()
    var word = _host_sessions_word()
    shared_slot_lock(word)
    var id = reg[].next_id
    reg[].next_id += 1
    reg[].sessions.append(RfHostSession(id, x^, n_rows, n_cols))
    shared_slot_unlock(word)
    return id


def rf_host_session_lease(id: Int) raises -> RfHostSessionLease:
    """Session `id`'s X (shared) and shape, under the lock."""
    var reg = RF_HOST_SESSIONS.get_or_create_ptr()
    var word = _host_sessions_word()
    shared_slot_lock(word)
    try:
        ref s = reg[].sessions[_host_session_index(id)]
        return RfHostSessionLease(s.x, s.n_rows, s.n_cols)
    finally:
        shared_slot_unlock(word)


def rf_host_session_retire(id: Int) raises:
    """Session `id` leaves the registry; its X is freed when the last lease
    lets go."""
    var reg = RF_HOST_SESSIONS.get_or_create_ptr()
    var word = _host_sessions_word()
    shared_slot_lock(word)
    try:
        _ = reg[].sessions.pop(_host_session_index(id))
    finally:
        shared_slot_unlock(word)


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
    var lease = rf_host_session_lease(session_id)
    var sr = lease.n_rows
    var sc = lease.n_cols
    if sr != n_rows or sc != n_cols:
        raise Error(
            entry + ": params name " + String(n_rows) + " x " + String(n_cols)
            + ", the session holds " + String(sr) + " x " + String(sc)
        )
    # the copy reads the leased X with no lock
    return lease.x[].copy()


def _rf_export_into_native(id: Int, dst: List[Int], trees: Int, nodes: Int, outputs: Int) raises:
    """XF_EXPORT_INTO: the retained forest into the caller's five arrays."""
    var registry = RF_HOST_EXPORTS.get_or_create_ptr()
    registry[].validate(id, trees, nodes, outputs)
    validate_forest_export_destinations(dst[0], dst[1], dst[2], dst[3], dst[4])
    var op = i32_ptr(dst[0])
    var cp = i32_ptr(dst[1])
    var tp = f32_ptr(dst[2])
    var lp = i32_ptr(dst[3])
    # THE LOCK (core/forest_export_run.mojo): the entry is taken out, read with
    # no lock held and put back on every path.
    var entry = registry[].take(id)
    var failed = False
    var words = String("")
    try:
        rf_host_export_into(entry.model, op, cp, tp, lp, dst[4])
    except problem:
        failed = True
        words = String(problem)
    registry[].put_back(id, entry^)
    if failed:
        raise Error(words)


def _rf_export_verify_native(id: Int, dst: List[Int]) raises -> Int:
    """XF_EXPORT_VERIFY (MOJOLEARN_FOREST_EXPORT=verify): the diagnostic
    lists `forest_export_legacy` returns, against the exported arrays."""
    var registry = RF_HOST_EXPORTS.get_or_create_ptr()
    # THE LOCK (core/forest_export_run.mojo): the entry is taken out, read with
    # no lock held and put back on every path.
    var entry = registry[].take(id)
    var same = (len(entry.meta) == 3 and Int(entry.meta[0]) == entry.model.n_trees
                and Int(entry.meta[1]) == entry.model.num_outputs and Int(entry.meta[2]) == entry.model.notices)
    var offsets = List[Int64]()
    var colid = List[Int64]()
    var quesval = List[Float64]()
    var left_child = List[Int64]()
    var leaves = List[Float64]()
    var failed = False
    var words = String("")
    if same:
        try:
            rf_host_forest_lists(entry.model, offsets, colid, quesval, left_child, leaves)
        except problem:
            failed = True
            words = String(problem)
    registry[].put_back(id, entry^)
    if failed:
        raise Error(words)
    if not same:
        return -1
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


# ------------------------------------------------- the image (lane bm-trees)
# The host RF image's forest entry operations moved here from the binding
# (lane bm-trees, 2026-11-22), so a Mojo check publishes the same image
# (`publish_rf_host_forest`) and fits a DART member, a plain fit or a strict
# predict through the forest entry with no binding loaded. ONE BODY, TWO
# ENTRIES, as core/mojolearn_et_host_run.mojo: every body here runs bare;
# the binding's entry releases the GIL around the long operations
# (`forest_host_entry_releases`). The data sessions live here too (lane
# mf-ada2, 2026-11-23): their registry has its own lock word
# (`RF_HOST_SESSIONS`), so their bodies run bare like the fits, and a typed
# RUN with no GIL (xtrees/trees_ada_fit_run.mojo) or a Mojo check opens and
# fits sessions through the same image.
# TOMBSTONE: RfHostSessionOps / RfHostNoSessions / the binding's RfHostPySessions (the image's session
# operations as a parameter: the GIL-holding binding bodies, or a refusal for a caller with no Python) deleted
# 2026-11-23 by lane mf-ada2: the bodies need no GIL. Not an experiment arm.


def _rf_session_open_native(x_address: Int, n_rows: Int, n_cols: Int, row_major: Bool,
                            share_tables: Bool) raises -> Int:
    """XF_SESSION_OPEN (`bindings/_mojolearn_rf.mojo` `_rf_session_open_native`):
    `x` is float32, ROW-major when `row_major` and COLUMN-major otherwise,
    borrowed through this call only. Returns the session handle."""
    if n_rows <= 0 or n_cols <= 0:
        raise Error("rf_data_session_open: invalid shape")
    # A host binding is IDENTICAL only, and an IDENTICAL member draws its
    # own quantile sample (the GPU binding's IDENTICAL refusal).
    if share_tables:
        raise Error(
            "rf_data_session_open: shared quantile tables are a FAST"
            " option; an IDENTICAL member draws its own sample"
        )
    # Lane a8c-device-x-linear-prep-rfe: NaN AND infinity (the GPU binding's
    # device scan of its upload), so the boost meta-estimators' own host
    # scan before the open is deleted; the host column tests the words it
    # reads, once, in parallel SIMD tasks (core/host_finite_scan.mojo).
    if not host_all_finite_f32(f32_ptr(x_address), n_rows * n_cols):
        raise Error("rf_data_session_open: " + RF_NONFINITE_REFUSAL)
    var x = _read_x_col_major(x_address, n_rows, n_cols, row_major)
    return rf_host_session_keep(x^, n_rows, n_cols)


def _rf_session_close_native(handle: Int) raises:
    """XF_SESSION_CLOSE."""
    rf_host_session_retire(handle)


def _rf_fit_session_rows_native(params: ForestFitArgs) raises -> Int:
    """XF_FIT_SESSION, regressor: the regressor fit on the rows `params.x`
    (int32 row ids of the session's X, `params[0]` of them, repeats
    allowed), `params.y` their labels in that order: the column-major gather
    of those rows (the bytes the GPU binding's `launch_gather_rows_colmajor`
    writes), then the plain fit. Returns the export handle."""
    comptime entry = "rf_regressor_fit_session_rows"
    var p = _session_params[False](entry, params)
    var n_rows = params.i(0)
    var n_cols = params.i(1)
    if n_rows <= 0:
        raise Error(entry + ": no rows")
    var rp = i32_ptr(params.x)
    # the X leased under the registry's lock, gathered with no lock
    var lease = rf_host_session_lease(params.session)
    var src_rows = lease.n_rows
    if lease.n_cols != n_cols:
        raise Error(entry + ": the session holds another column count")
    var x = rf_host_session_rows_x(
        entry, lease.x, src_rows, rp, n_rows, n_cols
    )
    _ = lease^
    var y_address = params.y
    var forest = _rf_fit_colmajor[False](
        x^, y_address, n_rows, n_cols, 0, p, List[Float32](), 0
    )
    return _retain_rf_handle(forest^)


def _rf_fit_weighted_session_native(params: ForestFitArgs) raises -> Int:
    """XF_FIT_SESSION, classifier: the weighted classifier fit on the
    session's X. Returns the export handle."""
    comptime entry = "rf_classifier_fit_weighted_session"
    var p = _session_params[True](entry, params)
    var n_rows = params.i(0)
    var n_cols = params.i(1)
    var n_classes = params.i(2)
    var w_address = params.weights
    if w_address == 0:
        raise Error("weighted RF requires a nonzero Float32 weight pointer")
    var weights = _read_weights(w_address, n_rows)
    var x = _session_x(entry, params.session, n_rows, n_cols)
    var y_address = params.y
    # sk-weq-trees: the zero-weight rows leave this member's fit
    var y_kept = List[Int32]()
    var n_fit = _host_drop_zero_weight_rows(
        x, weights, y_kept, y_address, n_rows, n_cols, p.bootstrap
    )
    var y_fit = y_address if n_fit == n_rows else Int(y_kept.unsafe_ptr())
    var mwl = _apply_min_weight_fraction(p, _min_weight_fraction(params), weights, n_fit)
    var forest = _rf_fit_colmajor[True](
        x^, y_fit, n_fit, n_cols, n_classes, p, weights, 0,
        min_weight_leaf=mwl,
    )
    _ = y_kept^
    return _retain_rf_handle(forest^)



def rf_host_fit_handle[CLASSIFIER: Bool, ROWMAJOR: Bool](params: ForestFitArgs) raises -> Int:
    """XF_FIT: `_rf_fit_forest`, kept in RF_HOST_EXPORTS; the export handle."""
    return _retain_rf_handle(_rf_fit_forest[CLASSIFIER, ROWMAJOR](params))


def rf_host_fit_dart(params: ForestFitArgs) raises -> Int:
    """XF_FIT_DART, the host twin of `bindings/_mojolearn_rf.mojo::rf_regressor_fit_dart_binding`
    (box-run-2-dart-host, 2026-10-05): the DART member fit on the HOST DART
    session (xtrees/dart_host.mojo `DART_HOST_SESSIONS`): X is the session's
    row-major n x d array, the target its k x n `target` plane. The gathers
    are `ensemble/dart_member.mojo`'s, element for element: X column-major
    `x[j * m + i] = X[rows[i] * d + cols[j]]` (identity rows / columns when
    the address is 0) and `y[i] = target[c * n + rows[i]]` (the whole class
    plane when rows is 0). Then `_rf_fit_colmajor[False]`, the body the plain
    host regressor fit runs, so the member forest's words are the device
    member's. Same params, checks and export descriptor as the device
    entry. No GIL (see above)."""
    comptime entry = "rf_regressor_fit_dart"
    comptime if DART_HOST:
        var p = _session_params[False](entry, params)
        var n_rows = params.i(0)
        var n_cols = params.i(1)
        var hid = params.session
        var c = params.cls
        var rows_a = params.x
        var cols_a = params.y
        # held for this member fit (the registry's lock word covers the
        # lookup only; lane fx-gil)
        var dreg = DART_HOST_SESSIONS.get_or_create_ptr()
        var dsess = dreg[].lease(hid)
        var n = dsess[].n
        var d = dsess[].d
        var k = dsess[].k
        if c < 0 or c >= k:
            raise Error(entry + ": class out of range")
        if n_rows <= 0 or n_cols <= 0:
            raise Error(entry + ": invalid shape")
        if (rows_a == 0 and n_rows != n) or (cols_a == 0 and n_cols != d):
            raise Error(
                entry + ": params name a shape the DART session does not hold"
            )
        if cols_a != 0 and n_cols > d:
            raise Error(entry + ": more columns than X holds")
        # the row and column ids, checked against the session (the device
        # entry trusts its callers; a host read out of range is not a word
        # the device writes, so refuse it instead)
        var x = List[Float32]()
        var y = List[Float32]()
        rf_host_dart_member_xy(
            entry, rows_a, cols_a, n_rows, n_cols, n, d, c,
            dsess[].x_addr, dsess[].target, x, y,
        )
        var y_address = Int(y.unsafe_ptr())
        var forest = _rf_fit_colmajor[False](
            x^, y_address, n_rows, n_cols, 0, p, List[Float32](), 0
        )
        # y is read through its address inside the fit: keep it alive past it
        _ = y^
        return _retain_rf_handle(forest^)
    else:
        raise Error(
            entry + ": built without the host DART round"
            " (MOJOLEARN_IDN_DART_DEVICE_OFF)"
        )


struct RfHostForestImage(ForestImage):
    """This image's forest entry operations (core/forest_entry.mojo; lane
    calltable-forest). The host column scans X itself (no device scan), keeps
    its trees in the strict snapshot and has no ordered resident word or
    resident labels."""

    @staticmethod
    def caps() -> Int:
        comptime if DART_HOST:
            return XF_CAP_STRICT | XF_CAP_DART | XF_CAP_SESSION
        else:
            return XF_CAP_STRICT | XF_CAP_SESSION

    @staticmethod
    def dt_default_bins() -> Int:
        # this image exports no `rf_dt_default_bins`: the door's default
        return 128

    @staticmethod
    def fit(classifier: Bool, row_major: Bool, args: ForestFitArgs) raises -> Int:
        if classifier:
            if row_major:
                return rf_host_fit_handle[True, True](args)
            return rf_host_fit_handle[True, False](args)
        if row_major:
            return rf_host_fit_handle[False, True](args)
        return rf_host_fit_handle[False, False](args)

    @staticmethod
    def fit_dart(args: ForestFitArgs) raises -> Int:
        return rf_host_fit_dart(args)

    @staticmethod
    def export_sizes(handle: Int, out_addr: Int) raises:
        RF_HOST_EXPORTS.get_or_create_ptr()[].sizes_into(handle, out_addr)

    @staticmethod
    def export_into(handle: Int, dst: List[Int], trees: Int, nodes: Int, outputs: Int) raises:
        _rf_export_into_native(handle, dst, trees, nodes, outputs)

    @staticmethod
    def export_verify(handle: Int, dst: List[Int]) raises -> Int:
        return _rf_export_verify_native(handle, dst)

    @staticmethod
    def export_release(handle: Int) raises:
        RF_HOST_EXPORTS.get_or_create_ptr()[].release(handle)

    @staticmethod
    def predict(model: List[Int], x: Int, out_addr: Int, rows: Int, cols: Int, trees: Int, outputs: Int,
                regression: Bool, free: Bool) raises -> Int:
        return _rf_predict_walk(model, x, out_addr, rows, cols, trees, outputs, not regression)

    @staticmethod
    def resident_prepare(model: List[Int], features: Int, trees: Int, outputs: Int, ordered: Int) raises -> Int:
        return forest_prepare_host_native[True](model, features, trees, outputs, ordered)

    @staticmethod
    def resident_predict(handle: Int, x: Int, out_addr: Int, rows: Int, cols: Int, outputs: Int) raises -> Int:
        return forest_predict_resident_host_native[True](handle, x, out_addr, rows, cols, outputs)

    @staticmethod
    def resident_labels(handle: Int, x: Int, out_addr: Int, rows: Int, cols: Int, outputs: Int) raises -> Int:
        raise Error("RuntimeError: this forest binding has no resident label prediction")

    @staticmethod
    def resident_release(handle: Int) raises:
        forest_release_host_native[True](handle)

    @staticmethod
    def strict_prepare(model: List[Int], features: Int, trees: Int, outputs: Int) raises -> Int:
        return forest_strict_prepare_host_native[True](model, features, trees, outputs)

    @staticmethod
    def strict_predict(handle: Int, x: Int, out_addr: Int, rows: Int, cols: Int, outputs: Int,
                       free: Bool) raises -> Int:
        return forest_host_strict_predict[True](handle, x, out_addr, rows, cols, outputs)

    @staticmethod
    def strict_release(handle: Int) raises:
        forest_strict_release_host_native[True](handle)

    @staticmethod
    def session_open(x: Int, rows: Int, cols: Int, row_major: Bool, share_tables: Bool) raises -> Int:
        return _rf_session_open_native(x, rows, cols, row_major, share_tables)

    @staticmethod
    def session_close(handle: Int) raises:
        _rf_session_close_native(handle)

    @staticmethod
    def fit_session(classifier: Bool, args: ForestFitArgs) raises -> Int:
        if classifier:
            return _rf_fit_weighted_session_native(args)
        return _rf_fit_session_rows_native(args)

    @staticmethod
    def ada_open(handle: Int, w: Int, y: Int, cw: Int, n: Int, k: Int) raises:
        raise Error("RuntimeError: the forest host binding has no AdaBoost device rounds")

    @staticmethod
    def fit_ada_round(classifier: Bool, args: ForestFitArgs) raises -> Int:
        raise Error("RuntimeError: the forest host binding has no AdaBoost device rounds")


def rf_host_forest_entry_native(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The image's forest entry for a caller that holds no GIL and no Python
    (a Mojo check; the binding publishes `rf_host_forest_entry_v1`, which
    releases the GIL around the same dispatch for a door)."""
    return forest_entry_dispatch[RfHostForestImage](
        operation, aa, na, pp, np, ff, nf, result, context, sink
    )


def publish_rf_host_forest() raises:
    """Publish `rf_host_forest_entry_native` under the RF table's cpu key, for
    a process with no `_mojolearn_rf_host` loaded (a Mojo check)."""
    forest_entry_publish["MojolearnForestEntryOwnerRfHostNativeV1"](FOREST_TABLE_RF, "cpu", rf_host_forest_entry_native)
