# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RUN of `bindings/_mojolearn_rf.mojo` (wave 8 run split): the RandomForest
fits, data sessions, AdaBoost rounds, exports and the sequential predict as
typed Mojo. Imports no Python. The binding keeps UNPACK (the Python
arguments into `ForestFitArgs` and typed words), PACK (descriptors and
diagnostic lists) and the GIL: it releases the GIL around each device body
here (`*_forest`, `*_stage`, `*_round`, `*_fit` bodies) exactly where it
released it before the split, and holds it for the registry steps (the
session and export registries are serialized by the GIL)."""
from std.memory import unsafe_memcpy
from core.hostptr import copy_f32, list_f32, list_i32
from ensemble.device_layout import device_has_nan_f32, upload_forest_x
from ensemble.dart_member import launch_dart_member_x, launch_dart_member_y
from ensemble.ada_rounds import (
    AdaState,
    ada_member_weights,
    ada_predict_codes,
    ada_predict_values,
    ada_rescale,
    ada_upload_nodes,
)
from xtrees.ops_device import weighted_sample_on
from xtrees.ops_device_boost import r2_step_on, samme_step_on
from xtrees.dart_device import DART_SESSIONS, _DART_CTX
from ensemble.nan_refusal import RF_NAN_REFUSAL, RF_NONFINITE_REFUSAL

from ensemble.device_finite import FOREST_DEVICE_FINITE, ForestFiniteScan

from core.forest_fit_request import (
    ForestFitArgs, XF_CAP_DEVICE_SCAN, XF_CAP_DART, forest_export_verify_lists,
)
from core.forest_inference_model import (
    resident_prepare, resident_predict_into, resident_release,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from std.sys.compile import is_defined
from core.accelerator_target import TARGET_HAS_APPLE_GPU

from max.gpu.host import DeviceBuffer, DeviceContext
from core.neural_context import process_ctx
from checks.numerics import GLOBAL_NUMERIC_MODE as _DEVCTX_MODE, NUMERIC_IDENTICAL as _DEVCTX_IDENTICAL

#: This binding's ONE process-lifetime DeviceContext (core/neural_context.mojo,
#: lane/devctx-lifetime): a context per call exhausts Metal command queues.
comptime _DEVCTX_SLOT = "MojolearnRfContextIdenticalV3" if _DEVCTX_MODE == _DEVCTX_IDENTICAL else "MojolearnRfContextFastV3"


from ensemble.decisiontree.batched_levelalgo.bins import (
    BinScales,
    ClassificationBin,
    WeightedClassificationBin,
    RegressionBin,
    WeightedRegressionBin,
)
from checks.fixed_point import choose_scale
from core.abs_sum_blocked import (
    device_abs_sum_blocked,
    device_any_index_out_of_range,
)
from core.weight_scan_device import device_scan_weights
from core.weight_compact_device import (
    device_gather_rows,
    device_gather_rows_colmajor,
    device_nonzero_weight_rows,
)
from ensemble.decisiontree.batched_levelalgo.objectives import (
    ClassificationObjectiveFunction,
    RegressionObjectiveFunction,
)
from ensemble.instruments import StageTimes
from ensemble.decisiontree.decisiontree import (
    ENTROPY,
    GAMMA,
    GINI,
    INVERSE_GAUSSIAN,
    MSE,
    POISSON,
    DecisionTreeParams,
    TreeMetaDataNode,
    criterion_name,
)
from ensemble.flatnode import SparseTreeNode
from ensemble.randomforest import (
    CLASSIFICATION,
    REGRESSION,
    RF_params,
    RandomForest,
    RandomForestMetaData,
    ForestPrep,
    fit_forest,
    fit_forest_prepared,
    launch_gather_rows_colmajor,
    n_sampled_rows_for,
)
from core.shared_slot import SharedSlot

comptime DT = DType.float32
comptime CLT = DType.int32
comptime RLT = DType.float32
comptime ClsObj = ClassificationObjectiveFunction[DT, CLT, ClassificationBin]
comptime WeightedClsObj = ClassificationObjectiveFunction[
    DT, CLT, WeightedClassificationBin
]
comptime RegObj = RegressionObjectiveFunction[DT, RLT, RegressionBin]
comptime WeightedRegObj = RegressionObjectiveFunction[DT, RLT, WeightedRegressionBin]
from core.forest_export_run import ForestExportRegistry
from core.forest_export_binding_impl import validate_forest_export_destinations
from core.mojolearn_rf_impl import rf_export_trees_into, rf_flatten_trees


comptime N_RF_FIT_PARAMS = 16
"""`params` for both fit entry points, in this exact order -- the wrapper
names the same order in the same words:

     0  n_rows
     1  n_cols
     2  n_classes             (classifier only; MUST be 0 for the regressor)
     3  n_trees
     4  max_depth
     5  max_leaves            (-1 = unlimited, cuML's own sentinel)
     6  max_features_fraction (float in (0, 1]; the WRAPPER resolves
                               'sqrt'/'log2'/counts to a fraction, exactly
                               as cuML's python layer does)
     7  max_n_bins
     8  min_samples_leaf
     9  min_samples_split
    10  min_impurity_decrease (float)
    11  bootstrap             (0/1)
    12  max_samples           (float; ignored and reset to 1.0 by the implementation
                               when bootstrap is 0, `randomforest.cuh:304`)
    13  seed
    14  n_streams             (cuML's python default is 4; DEVIATION 117)
    15  max_batch_size        (cuML default 4096)
    16  min_weight_fraction_leaf  OPTIONAL (float in [0, 0.5]; scikit-
                               learn's; a 16-slot call is 0). Read by
                               `rf_classifier_fit*` (session included)
                               and `rf_regressor_fit*`; see
                               `_rf_min_weight_fraction` below.

The split criterion is NOT a slot: it crosses as its own integer argument
(`criterion`, the `CRITERION` enumerator value from `decisiontree.mojo`)
so the two fit entry points can gate it by LABEL TYPE before the engine
sees it -- see DEVIATION 407 at `_check_criterion` below.
"""

# DEVIATION 407 (2026-08-23, RF lane). THE CRITERION SELECTOR CROSSES THE
# PYTHON BOUNDARY.
# THEIRS: `randomforest_common.pyx:104-151` maps the strings
# 'gini'/'entropy'/'mse'/'poisson'/'gamma'/'inverse_gaussian' (and the
# digits '0'..'7') onto `CRITERION`, refuses MAE by NotImplementedError, and
# passes the enumerator down as `split_criterion`; nothing in their python
# layer checks that the enumerator suits the LABEL TYPE. The C++ `default:`
# arm of `GainPerSplit` (`objectives.cuh:132-136`, `:331-338`) returns
# `-max()` for every candidate, so a regression enumerator handed to the
# classifier (or vice versa) fits a forest of STUMPS without a word.
# OURS: the wrapper maps sklearn-shaped names onto the same enumerators
# (`python/mojolearn/randomforest.py::_CLS_CRITERIA` / `_REG_CRITERIA`)
# and each fit binding REFUSES BY NAME an enumerator its objective has no
# arm for -- a silent stump is the "reached but inert" failure class
# `ensemble/checks/criteria_check.mojo` arm A/B exists to catch, and
# the Python surface must not reopen it. Until this deviation the binding
# hardcoded GINI / MSE and the wrapper refused every other name; that text
# is gone.


def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null float32 buffer address")
    return MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=addr)


def _i32_ptr(addr: Int) raises -> MutPointer[Int32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null int32 buffer address")
    return MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=addr)


def _cls_criteria() -> List[Int]:
    """The enumerators `ClassificationObjectiveFunction.GainPerSplit` has
    an arm for (`objectives.cuh:132-136`)."""
    return [GINI, ENTROPY]


def _reg_criteria() -> List[Int]:
    """The enumerators `RegressionObjectiveFunction.GainPerSplit` has an
    arm for (`objectives.cuh:331-338`); MAE is enumerated but armless,
    theirs and ours."""
    return [MSE, POISSON, GAMMA, INVERSE_GAUSSIAN]


def _check_criterion(
    who: String, criterion: Int, allowed: List[Int]
) raises:
    """DEVIATION 407: refuse an enumerator the objective has no arm for."""
    for i in range(len(allowed)):  # small-loop(allowed: at most four criterion enumerators): parameter validation, not data
        if allowed[i] == criterion:
            return
    var names = String("")
    for i in range(len(allowed)):  # small-loop(allowed: at most four criterion enumerators): error message names, not data
        if i > 0:
            names += "/"
        names += criterion_name(allowed[i])
    raise Error(
        who
        + ": split criterion "
        + criterion_name(criterion)
        + " ("
        + String(criterion)
        + ") has no arm in this objective's GainPerSplit"
        + " (objectives.mojo, DEVIATION 407); accepted: "
        + names
    )


# TOMBSTONE: MOJOLEARN_RF_FAST_BATCH16K (DROPPED-noise) deleted 2026-10-03 by 288809ef8; code recoverable at 288809ef8^.
# Restore: git apply experiments/removed/MOJOLEARN_RF_FAST_BATCH16K.patch; record in docs/TOMBSTONES.md.
#: FAST on Apple (trees-apple3): a wider node batch. A tree deeper than 12
#: levels has levels of more than 4096 nodes, and every batch of a level is
#: one more round of launches, uploads and a host wait for nodes of a few
#: rows each (M4 Pro, taxi, 100 trees of depth 16: 32.8 node batches and
#: 71.5 histogram rounds per tree). The queue is first in, first out and a
#: node's split reads only its own rows and its own feature sample, so the
#: batch width changes which launch a node rides, not its split. Only the
#: default width (4096) is widened, only for trees without a leaf budget
#: (max_leaves -1) that may grow past 12 levels, and only as far as `RF_FAST_BATCH_BYTES` of histogram
#: workspace per stream allows. OPT-IN until its A/B passes: `-D
#: MOJOLEARN_RF_FAST_BATCH32K` (the 16K arm was DROPPED-noise,
#: lane/apple-fast-trees2 @ bfd1d7cc6).
comptime RF_FAST_BATCH = 0 if not (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and TARGET_HAS_APPLE_GPU
) else (
    32768 if is_defined["MOJOLEARN_RF_FAST_BATCH32K"]() else 0
)


comptime RF_FAST_BATCH_BYTES = 536870912


comptime RF_CUML_DEFAULT_BATCH = 4096


def _rf_batch_size(params: ForestFitArgs) raises -> Int:
    """Slot 15, widened under `RF_FAST_BATCH` (see its note)."""
    var batch = params.i(15)
    comptime if RF_FAST_BATCH > 0:
        var depth = params.i(4)
        if (
            batch == RF_CUML_DEFAULT_BATCH
            and params.i(5) == -1
            and (depth > 12 or depth < 1)
        ):
            var n_cols = params.i(1)
            var outputs = params.i(2)
            if outputs < 1:
                outputs = 1
            var cols = Int(params.f(6) * Float64(n_cols)) + 1
            if cols > n_cols:
                cols = n_cols
            if cols > 40:
                cols = 40
            # 8 bytes covers both bin types
            var per_node = params.i(7) * outputs * cols * 8
            var wide = RF_FAST_BATCH
            if per_node > 0 and wide * per_node > RF_FAST_BATCH_BYTES:
                wide = RF_FAST_BATCH_BYTES // per_node
            if wide > batch:
                batch = wide
    return batch


def _rf_params_len_ok(params: ForestFitArgs) raises -> Bool:
    """16 slots, or 17 with the optional `min_weight_fraction_leaf`."""
    var n = len(params)
    return n == N_RF_FIT_PARAMS or n == N_RF_FIT_PARAMS + 1


def _rf_min_weight_fraction(params: ForestFitArgs) raises -> Float64:
    """Slot 16, scikit-learn's `min_weight_fraction_leaf` (lane
    sk-kernels): 0 for a 16-slot call; refused outside [0, 0.5] in
    scikit-learn's words (its `_parameter_constraints`)."""
    if len(params) <= N_RF_FIT_PARAMS:
        return 0.0
    var f = params.f(N_RF_FIT_PARAMS)
    if not (f >= 0.0 and f <= 0.5):
        raise Error(
            "The 'min_weight_fraction_leaf' parameter must be a float in the"
            " range [0.0, 0.5]. Got " + String(f) + " instead."
        )
    return f


def _rf_fold_min_weight_count(
    mut rf_params: RF_params, frac: Float64, n_rows: Int
):
    """scikit-learn's `min_weight_leaf = frac * total weight` on a COUNT
    objective (no row weights reach the bins: unweighted, all-unit, or
    weighted bootstrap, where the weights drive the draw and every drawn
    row counts once). A tree's total weight is then its row count -- the
    bootstrap draw `n_sampled_rows_for` sizes, else n_rows -- and an
    integer child count c satisfies c >= frac * total exactly when
    c >= ceil(frac * total), so the bound is `min_samples_leaf` raised to
    that ceiling. frac == 0 leaves the params, and every bit, unchanged."""
    if not (frac > 0.0):
        return
    var n_tree = n_sampled_rows_for(
        rf_params.bootstrap,
        rf_params.max_samples if rf_params.bootstrap else Float32(1.0),
        n_rows,
    )
    var bound = frac * Float64(n_tree)
    var need = Int(bound)
    if Float64(need) < bound:
        need += 1
    if need > Int(rf_params.tree_params.min_samples_leaf):
        rf_params.tree_params.min_samples_leaf = Int32(need)


def _rf_min_weight_leaf(frac: Float64, weight_total: Float64) -> Float32:
    """scikit-learn's `min_weight_leaf` on a WEIGHTED objective (no
    bootstrap, non-unit row weights): `frac * total weight`, read by the
    weighted bins' `GainPerSplit` (`ObjectiveLike.SetMinWeightLeaf`)."""
    if not (frac > 0.0):
        return Float32(0)
    return Float32(frac * weight_total)


def _rf_drop_zero_weight_rows[LT: DType](
    ctx: DeviceContext,
    mut dx: DeviceBuffer[DT],
    mut dy: DeviceBuffer[LT],
    mut dsw: DeviceBuffer[DT],
    mut weights: List[Float32],
    n_rows: Int,
    n_cols: Int,
) raises -> Int:
    """lane sk-weq-trees: scikit-learn's tree splitter keeps only the rows
    whose sample_weight is not zero (`_splitter.pyx`, `Splitter.init`), so a
    zero-weight row must shape nothing -- not the quantile edges (computed
    from every row of `dx`), not a node's row count. Called on a weighted
    fit without bootstrap whose weights are not all 1.0: the kept rows are
    found and gathered ON THE DEVICE (`core/weight_compact_device`) into new
    column-major X, labels and weights, which replace the caller's; the
    host weight list becomes the kept weights (one bulk copy). Returns the
    kept row count; n_rows (and every buffer untouched) when no weight is
    zero, so such a fit keeps every bit."""
    var rows = ctx.enqueue_create_buffer[DType.int32](n_rows)
    var kept = device_nonzero_weight_rows(ctx, dsw, n_rows, rows)
    if kept >= n_rows or kept <= 0:
        _ = rows^
        return n_rows
    var cx = device_gather_rows_colmajor(ctx, dx, n_rows, n_cols, rows, kept)
    var cy = device_gather_rows[LT](ctx, dy, rows, kept)
    var cw = device_gather_rows[DT](ctx, dsw, rows, kept)
    weights = List[Float32](length=kept, fill=Float32(0))
    ctx.enqueue_copy(dst_ptr=weights.unsafe_ptr(), src_buf=cw)
    ctx.synchronize()
    _ = rows^
    dx = cx^
    dy = cy^
    dsw = cw^
    return kept


def _rf_params_from(params: ForestFitArgs, criterion: Int) raises -> RF_params:
    """Slots 3-15 into `RF_params`, read under the GIL."""
    return RF_params(
        n_trees=Int32(params.i(3)),
        bootstrap=params.i(11) != 0,
        max_samples=Float32(params.f(12)),
        seed=UInt64(params.i(13)),
        n_streams=Int32(params.i(14)),
        tree_params=DecisionTreeParams(
            max_depth=Int32(params.i(4)),
            max_leaves=Int32(params.i(5)),
            max_features=Float32(params.f(6)),
            max_n_bins=Int32(params.i(7)),
            min_samples_leaf=Int32(params.i(8)),
            min_samples_split=Int32(params.i(9)),
            split_criterion=criterion,
            min_impurity_decrease=Float32(params.f(10)),
            max_batch_size=Int32(_rf_batch_size(params)),
        ),
    )


comptime RFExportTrees = List[TreeMetaDataNode[DT]]


comptime RF_EXPORTS = SharedSlot[ForestExportRegistry[RFExportTrees], ("MojolearnRFFitExportIdenticalV1" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnRFFitExportFastV1")]


def _retain_rf_handle(var trees: RFExportTrees, notices: Int) raises -> Int:
    if not len(trees):
        raise Error("cannot export an empty fitted RF")
    var nodes = 0
    var outputs = Int(trees[0].num_outputs)
    var n_fitted = len(trees)
    for t in range(n_fitted):  # small-loop(n_fitted: fitted trees, one node count each): export shape bookkeeping, no row or node data
        var tree_nodes = len(trees[t].sparsetree)
        nodes += tree_nodes
        if Int(trees[t].num_outputs) != outputs or len(trees[t].vector_leaf) != tree_nodes * outputs:
            raise Error("fitted RF leaf storage differs from export dimensions")
    var count = len(trees)
    # The third word is the fit's notice mask (`ensemble/rf_notices.mojo`):
    # the public fit raises those notices as Python warnings.
    var meta: List[Int64] = [Int64(count), Int64(outputs), Int64(notices)]
    return RF_EXPORTS.get_or_create_ptr()[].insert_native(trees^, count, nodes, outputs, meta^)


def _rf_export_into_native(id: Int, dst: List[Int], trees: Int, nodes: Int, outputs: Int) raises:
    """XF_EXPORT_INTO: the kept fit into the caller's five arrays."""
    var registry = RF_EXPORTS.get_or_create_ptr()
    registry[].validate(id, trees, nodes, outputs)
    validate_forest_export_destinations(dst[0], dst[1], dst[2], dst[3], dst[4])
    var op = _i32_ptr(dst[0])
    var cp = _i32_ptr(dst[1])
    var tp = _f32_ptr(dst[2])
    var lp = _i32_ptr(dst[3])
    rf_export_trees_into(
        registry[].entries[id].model, op, cp, tp, lp, dst[4],
        registry[].entries[id].outputs,
    )


def _rf_export_verify_native(id: Int, dst: List[Int]) raises -> Int:
    """XF_EXPORT_VERIFY (MOJOLEARN_FOREST_EXPORT=verify): the diagnostic
    lists `forest_export_legacy` returns, against the exported arrays."""
    var registry = RF_EXPORTS.get_or_create_ptr()
    if id not in registry[].entries:
        raise Error("unknown or released fitted forest export handle")
    ref entry = registry[].entries[id]
    var legacy_outputs = Int(entry.model[0].num_outputs) if len(entry.model) > 0 else 1
    if (len(entry.meta) != 3 or Int(entry.meta[0]) != len(entry.model)
            or Int(entry.meta[1]) != legacy_outputs):
        return -1
    var offsets = List[Int64]()
    var colid = List[Int64]()
    var quesval = List[Float64]()
    var left_child = List[Int64]()
    var leaves = List[Float64]()
    rf_flatten_trees(entry.model, offsets, colid, quesval, left_child, leaves)
    return forest_export_verify_lists(offsets, colid, quesval, left_child, leaves, dst)


# ---------------------------------------------------------------------------
# THE DATA SESSION (trees-apple3). A boosted ensemble (DART, AdaBoost) fits
# one small forest per member on the SAME X: each member's fit scanned X for
# NaN, copied it into a pinned stage and uploaded it again (M4 Pro, 1M x 16:
# about 10 ms of a 30 ms member fit). A session does that once. Its member
# fits are `fit_forest_prepared` on the staged X: the same kernels on the
# same bytes, so a member's forest is the forest `rf_*_fit` returns.
# `share_tables` also keeps the first member's quantile table and binned
# dataset for the later members. Their quantile SAMPLE is then the first
# member's (the sample is drawn with the fit's seed), so the later members'
# bins differ from their own fits': a FAST-only option with a quality check.
# One thread uses a session at a time.
struct RfDataSession(Movable):
    var id: Int
    var dx: DeviceBuffer[DT]
    var n_rows: Int
    var n_cols: Int
    var host_x: Int
    var share_tables: Bool
    var prep: List[ForestPrep]
    # AdaBoost's resident state (`rf_ada_open`), at most one; dropped with
    # the session (lane n5-prophet-adaboost-loops)
    var ada: List[AdaState]

    def __init__(
        out self,
        id: Int,
        var dx: DeviceBuffer[DT],
        n_rows: Int,
        n_cols: Int,
        host_x: Int,
        share_tables: Bool,
    ):
        self.id = id
        self.dx = dx^
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.host_x = host_x
        self.share_tables = share_tables
        self.prep = List[ForestPrep]()
        self.ada = List[AdaState]()


struct RfSessionRegistry(Defaultable, Movable):
    var sessions: List[RfDataSession]
    var next_id: Int

    def __init__(out self):
        self.sessions = List[RfDataSession]()
        self.next_id = 1

    def find(self, id: Int) raises -> Int:
        for i in range(len(self.sessions)):  # small-loop(sessions: open data sessions of this process): handle lookup, not data
            if self.sessions[i].id == id:
                return i
        raise Error("unknown or closed forest data session handle")


comptime RF_SESSIONS = SharedSlot[RfSessionRegistry, ("MojolearnRFDataSessionIdenticalV1" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnRFDataSessionFastV1")]


def _session_shape_check(
    name: String, session_rows: Int, session_cols: Int,
    n_rows: Int, n_cols: Int,
) raises:
    if session_rows != n_rows or session_cols != n_cols:
        raise Error(
            name
            + ": params name "
            + String(n_rows)
            + " x "
            + String(n_cols)
            + ", the session holds "
            + String(session_rows)
            + " x "
            + String(session_cols)
        )


def _rf_session_restore_prep(session_id: Int, mut prep: List[ForestPrep]) raises:
    """Hand a member fit's swapped-out prep back to its session, if the
    session is still open (a closed session simply drops it)."""
    var back = RF_SESSIONS.get_or_create_ptr()
    for i in range(len(back[].sessions)):  # small-loop(sessions: open data sessions of this process): handle lookup, not data
        if back[].sessions[i].id == session_id:
            swap(prep, back[].sessions[i].prep)
            return


def _u64_ptr(addr: Int) raises -> MutPointer[UInt64, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null float64 buffer address")
    return MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=addr)


def _f64_ptr(addr: Int) raises -> MutPointer[Float64, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null float64 buffer address")
    return MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=addr)


def _ada_state_of(session_id: Int, k_needed: Bool, name: String) raises -> AdaState:
    """A copy of the session's AdaBoost state (buffer handles: the device
    memory is shared), taken under the GIL (hazard B)."""
    var reg = RF_SESSIONS.get_or_create_ptr()
    var si = reg[].find(session_id)
    if len(reg[].sessions[si].ada) != 1:
        raise Error(name + ": the session holds no AdaBoost state (rf_ada_open first)")
    ref st = reg[].sessions[si].ada[0]
    if (st.k >= 2) != k_needed:
        raise Error(name + ": the session's AdaBoost state is the other estimator's")
    return AdaState(st.n, st.k, st.w.copy(), st.codes.copy(), st.values.copy(), st.cw.copy(), st.has_cw)


def _default_rf_params(n_trees: Int) raises -> RF_params:
    """A benign `RF_params` for the predict-side metadata: predict reads the
    forest's trees, not these knobs, but the structs require one."""
    return RF_params(
        n_trees=Int32(n_trees),
        bootstrap=True,
        max_samples=Float32(1.0),
        seed=UInt64(0),
        n_streams=Int32(4),
        tree_params=DecisionTreeParams(
            max_depth=Int32(16),
            max_leaves=Int32(-1),
            max_features=Float32(1.0),
            max_n_bins=Int32(128),
            min_samples_leaf=Int32(1),
            min_samples_split=Int32(2),
            split_criterion=GINI,
            min_impurity_decrease=Float32(0.0),
            max_batch_size=Int32(4096),
        ),
    )


def _rf_predict_native(model: List[Int], x_addr: Int, out_addr: Int, n_rows: Int, n_cols: Int, n_trees: Int,
                       num_outputs: Int, regression: Bool) raises -> Int:
    """The sequential vote (XF_PREDICT; lane calltable-forest): classifier
    probabilities through the IMPLEMENTED `predict_proba` (`predict` stopped
    one line early, per its docstring), or with `regression` the regressor's
    values through the IMPLEMENTED `RandomForest.predict` (output 0 of a
    one-output vote). Model arrays are int32/float32 as `_forest_out_trees`
    laid them out; `x` is ROW-major; `out` receives n_rows * num_outputs
    float32. The wrapper's argmax over the probabilities IS cuML's
    `predict`. Returns rows written."""
    var name = String("rf_predict_reg") if regression else String("rf_predict_proba")
    if regression:
        if n_trees < 1 or num_outputs != 1:
            raise Error("rf_predict_reg: n_trees must be >= 1 and num_outputs must be 1")
    elif n_trees < 1 or num_outputs < 2:
        raise Error("rf_predict_proba: n_trees must be >= 1 and num_outputs >= 2")
    var offsets_p = _i32_ptr(model[0])
    var colid_p = _i32_ptr(model[1])
    var quesval_p = _f32_ptr(model[2])
    var left_p = _i32_ptr(model[3])
    var leaves_p = _f32_ptr(model[4])
    _ = _f32_ptr(x_addr)
    _ = _f32_ptr(out_addr)
    # cpu-gpu-cleanup t-forest: the `sequential` engine runs on the device.
    # The host walk (`core/forest_host_predict.mojo`'s `rf_host_predict`,
    # DEVIATION 2900) is the CPU-only install's (`_mojolearn_rf_host`); here
    # the same arithmetic -- zero, add every tree's leaf in increasing tree
    # order, divide by `n_trees` -- is `forest_ordered_kernel`, one thread
    # per (row, output), through a one-call ordered snapshot.
    if n_rows == 0:
        return 0
    return _rf_predict_ordered(
        name, offsets_p, colid_p, quesval_p, left_p, leaves_p,
        x_addr, out_addr, n_rows, n_cols, n_trees, num_outputs,
    )


def _rf_predict_ordered(
    name: String,
    offsets_p: MutPointer[Int32, MutUntrackedOrigin],
    colid_p: MutPointer[Int32, MutUntrackedOrigin],
    quesval_p: MutPointer[Float32, MutUntrackedOrigin],
    left_p: MutPointer[Int32, MutUntrackedOrigin],
    leaves_p: MutPointer[Float32, MutUntrackedOrigin],
    x_address: Int,
    out_address: Int,
    n_rows: Int,
    n_cols: Int,
    n_trees: Int,
    num_outputs: Int,
) raises -> Int:
    """The `sequential` engine on the device: the flat model goes into an
    ordered resident snapshot (`forest_ordered_kernel`: increasing-tree
    association, the host walk's fold), the borrowed rows are predicted into
    the borrowed output, and the snapshot is released. The GIL stays held,
    as `forest_inference_binding` requires for the registry."""
    var n_nodes = Int(offsets_p[n_trees])
    if Int(offsets_p[0]) != 0 or n_nodes < n_trees:
        raise Error(name + ": tree_offsets must start at 0 and hold at least one node per tree")
    if n_nodes > 2147483647 // num_outputs:
        raise Error(name + ": node/output count exceeds Int32")
    # one memcpy per array (no per-node host loop); the snapshot uploads once
    var offsets = list_i32(offsets_p, n_trees + 1)
    var columns = list_i32(colid_p, n_nodes)
    var thresholds = list_f32(quesval_p, n_nodes)
    var left = list_i32(left_p, n_nodes)
    var leaves = list_f32(leaves_p, n_nodes * num_outputs)
    var handle = resident_prepare[True](
        offsets, columns, thresholds, left, leaves, n_cols, num_outputs, True
    )
    try:
        resident_predict_into[True](
            handle,
            _f32_ptr(x_address).unsafe_origin_cast[MutAnyOrigin](),
            _f32_ptr(out_address).unsafe_origin_cast[MutAnyOrigin](),
            n_rows, n_cols, num_outputs, True,
        )
    except e:
        resident_release[True](handle)
        raise e
    resident_release[True](handle)
    return n_rows


def _rf_weighted_session_fit(
    session_dx: DeviceBuffer[DT],
    yp: MutPointer[Int32, MutUntrackedOrigin],
    var weights: List[Float32],
    n_rows: Int,
    n_cols: Int,
    n_classes: Int,
    var rf_params: RF_params,
    mw_frac: Float64,
    mut prep: List[ForestPrep],
    share_tables: Bool,
    host_x: Int,
    mut weight_total: Float64,
) raises -> RandomForestMetaData[DT, CLT]:
    """The GIL-released body of `rf_classifier_fit_weighted_session_binding`
    over the session X handle the binding took under the GIL."""
    var forest: RandomForestMetaData[DT, CLT]
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var dxv = session_dx.create_sub_buffer[DT](
        0, n_rows * n_cols
    )
    var hy = ctx.enqueue_create_host_buffer[CLT](n_rows)
    ctx.synchronize()
    unsafe_memcpy(dest=hy.unsafe_ptr(), src=yp, count=n_rows)
    var dy = ctx.enqueue_create_buffer[CLT](n_rows)
    ctx.enqueue_copy(dst_buf=dy, src_ptr=hy.unsafe_ptr())
    var dsw = ctx.enqueue_create_buffer[DT](max(1, len(weights)))
    ctx.synchronize()
    ctx.enqueue_copy(dst_buf=dsw, src_ptr=weights.unsafe_ptr())
    var wscan = device_scan_weights(ctx, dsw, n_rows)
    if wscan.bad:
        raise Error("class weights must be finite and nonnegative")
    if wscan.total <= 0:
        raise Error("class weights must have positive total")
    # sk-weq-trees: zero-weight rows leave the fit (no bootstrap), as in
    # `_rf_classifier_fit`. The member then fits its OWN compacted X
    # with its own quantile table (`fit_forest`), never the session's
    # shared one (built from every row); a member without a zero weight
    # keeps the session path and every bit.
    var n_fit = n_rows
    if not rf_params.bootstrap and not wscan.all_unit:
        n_fit = _rf_drop_zero_weight_rows[CLT](
            ctx, dxv, dy, dsw, weights, n_rows, n_cols
        )
        if n_fit < n_rows:
            wscan = device_scan_weights(ctx, dsw, n_fit)
    weight_total = wscan.total
    if wscan.all_unit:
        weights = List[Float32]()
    if n_fit < n_rows:
        if len(weights) > 0:
            var cscale = choose_scale(weight_total, n_fit)
            if cscale < Float64(1.1754943508222875e-38) or cscale > Float64(3.4028234663852886e38):
                raise Error("class weights exceed Float32 fixed-point scale range")
            forest = fit_forest[WeightedClsObj](
                ctx, dxv, dy, dsw, n_fit, n_cols, n_classes, rf_params,
                BinScales(Float32(1), Float32(cscale)),
                sample_weight_host=weights, host_x_addr=0,
                min_weight_leaf=_rf_min_weight_leaf(mw_frac, weight_total),
                notices_to_stderr=False,
            )
        else:
            _rf_fold_min_weight_count(rf_params, mw_frac, n_fit)
            forest = fit_forest[ClsObj](
                ctx, dxv, dy, dsw, n_fit, n_cols, n_classes, rf_params,
                sample_weight_host=weights, host_x_addr=0,
                notices_to_stderr=False,
            )
    elif len(weights) > 0 and not rf_params.bootstrap:
        var scale = choose_scale(weight_total, n_rows)
        if scale < Float64(1.1754943508222875e-38) or scale > Float64(3.4028234663852886e38):
            raise Error("class weights exceed Float32 fixed-point scale range")
        var scales = BinScales(Float32(1), Float32(scale))
        forest = fit_forest_prepared[WeightedClsObj](
            ctx, dxv, dy, dsw, n_rows, n_cols, n_classes, rf_params,
            prep, share_tables, scales, sample_weight_host=weights,
            host_x_addr=host_x,
            min_weight_leaf=_rf_min_weight_leaf(mw_frac, weight_total),
            notices_to_stderr=False,
        )
    else:
        _rf_fold_min_weight_count(rf_params, mw_frac, n_rows)
        forest = fit_forest_prepared[ClsObj](
            ctx, dxv, dy, dsw, n_rows, n_cols, n_classes, rf_params,
            prep, share_tables, sample_weight_host=weights,
            host_x_addr=host_x,
            notices_to_stderr=False,
        )
    ctx.synchronize()
    _ = dxv^
    _ = dy^
    _ = dsw^
    _ = hy^
    _ = ctx^
    _ = weights^
    return forest^


def _rf_ada_classifier_round(
    session_dx: DeviceBuffer[DT],
    state: AdaState,
    n_rows: Int,
    n_cols: Int,
    n_classes: Int,
    var rf_params: RF_params,
    mw_frac: Float64,
    mut prep: List[ForestPrep],
    share_tables: Bool,
    host_x: Int,
    lr: Float64,
    last: Bool,
    stats: MutPointer[Float64, MutUntrackedOrigin],
) raises -> RandomForestMetaData[DT, CLT]:
    """The GIL-released body of `rf_classifier_fit_ada_round_binding`: the
    branches of `_rf_weighted_session_fit` with the weights built on the
    device, then the member's codes and the SAMME step."""
    var forest: RandomForestMetaData[DT, CLT]
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var dxv = session_dx.create_sub_buffer[DT](0, n_rows * n_cols)
    var dfull = session_dx.create_sub_buffer[DT](0, n_rows * n_cols)
    # the fit owns its labels (a zero-weight compaction replaces them)
    var dy = ctx.enqueue_create_buffer[CLT](n_rows)
    ctx.enqueue_copy(dst_buf=dy, src_buf=state.codes)
    var dsw = ctx.enqueue_create_buffer[DT](n_rows)
    ada_member_weights(ctx, state, dsw)
    var wscan = device_scan_weights(ctx, dsw, n_rows)
    if wscan.bad:
        raise Error("class weights must be finite and nonnegative")
    if wscan.total <= 0:
        raise Error("class weights must have positive total")
    # sk-weq-trees, as in `_rf_weighted_session_fit`: zero-weight rows
    # leave a fit without bootstrap; the kept weights then come home
    # once (the compaction's copy), the only host weights of a round.
    var weights = List[Float32]()
    var n_fit = n_rows
    if not rf_params.bootstrap and not wscan.all_unit:
        n_fit = _rf_drop_zero_weight_rows[CLT](
            ctx, dxv, dy, dsw, weights, n_rows, n_cols
        )
        if n_fit < n_rows:
            wscan = device_scan_weights(ctx, dsw, n_fit)
    var weight_total = wscan.total
    if wscan.all_unit:
        weights = List[Float32]()
    if n_fit < n_rows:
        if len(weights) > 0:
            var cscale = choose_scale(weight_total, n_fit)
            if cscale < Float64(1.1754943508222875e-38) or cscale > Float64(3.4028234663852886e38):
                raise Error("class weights exceed Float32 fixed-point scale range")
            forest = fit_forest[WeightedClsObj](
                ctx, dxv, dy, dsw, n_fit, n_cols, n_classes, rf_params,
                BinScales(Float32(1), Float32(cscale)),
                sample_weight_host=weights, host_x_addr=0,
                min_weight_leaf=_rf_min_weight_leaf(mw_frac, weight_total),
                notices_to_stderr=False,
            )
        else:
            _rf_fold_min_weight_count(rf_params, mw_frac, n_fit)
            forest = fit_forest[ClsObj](
                ctx, dxv, dy, dsw, n_fit, n_cols, n_classes, rf_params,
                sample_weight_host=weights, host_x_addr=0,
                notices_to_stderr=False,
            )
    elif not wscan.all_unit and not rf_params.bootstrap:
        var scale = choose_scale(weight_total, n_rows)
        if scale < Float64(1.1754943508222875e-38) or scale > Float64(3.4028234663852886e38):
            raise Error("class weights exceed Float32 fixed-point scale range")
        forest = fit_forest_prepared[WeightedClsObj](
            ctx, dxv, dy, dsw, n_rows, n_cols, n_classes, rf_params,
            prep, share_tables, BinScales(Float32(1), Float32(scale)),
            host_x_addr=host_x,
            min_weight_leaf=_rf_min_weight_leaf(mw_frac, weight_total),
            notices_to_stderr=False, weights_on_device=True,
        )
    else:
        _rf_fold_min_weight_count(rf_params, mw_frac, n_rows)
        forest = fit_forest_prepared[ClsObj](
            ctx, dxv, dy, dsw, n_rows, n_cols, n_classes, rf_params,
            prep, share_tables, host_x_addr=host_x,
            notices_to_stderr=False, weights_on_device=not wscan.all_unit,
        )
    # the member's codes of every training row, then the SAMME step on
    # the resident weights
    var nodes = ada_upload_nodes(ctx, forest.trees)
    var dpred = ctx.enqueue_create_buffer[DType.int32](n_rows)
    ada_predict_codes(ctx, nodes, dfull, n_rows, dpred)
    var moved = samme_step_on(ctx, state.w, dpred, state.codes, n_rows, n_classes, lr, last, stats)
    var total = stats[unsafe_offset=3]
    if moved and total > 0:
        ada_rescale(ctx, state, total)
    ctx.synchronize()
    _ = nodes^
    _ = dpred^
    _ = dxv^
    _ = dfull^
    _ = dy^
    _ = dsw^
    _ = weights^
    _ = ctx^
    return forest^


def _rf_classifier_fit_forest[ROWMAJOR: Bool = False](params: ForestFitArgs, mut bt: StageTimes) raises -> RandomForestMetaData[DT, CLT]:
    """The typed body of `_rf_classifier_fit` (wave 8 run split): the checks,
    the uploads and the fit, run by the binding with the GIL released;
    the auxiliary outputs are written here. The binding keeps the fit
    (`_rf_keep_forest`, under the GIL)."""
    var weights_addr = params.weights
    var tree_start = params.tree_start
    var oob_addr = params.oob
    var oob_score_addr = params.score
    var importance_addr = params.importance
    if (oob_addr == 0) != (oob_score_addr == 0):
        raise Error("RF OOB output and score must be requested together")
    if not _rf_params_len_ok(params):
        raise Error(
            "rf_classifier_fit: params must hold "
            + String(N_RF_FIT_PARAMS)
            + " values (17 with min_weight_fraction_leaf), got "
            + String(len(params))
        )
    var n_rows = params.i(0)
    var n_cols = params.i(1)
    var n_classes = params.i(2)
    if n_classes < 2:
        raise Error("rf_classifier_fit: n_classes must be >= 2")
    var xp = _f32_ptr(params.x)
    var yp = _i32_ptr(params.y)
    var crit = params.criterion
    _check_criterion("rf_classifier_fit", crit, _cls_criteria())
    var rf_params = _rf_params_from(params, crit)
    var mw_frac = _rf_min_weight_fraction(params)

    # cpu3-bindings: the weights are one memcpy here; their refusals, the
    # all-ones test and the total run on the device after the upload
    # (`core/weight_scan_device`), not as a host walk over every row.
    var weights = List[Float32]()
    var weight_total = Float64(0)
    var weighted = weights_addr != 0
    if weighted:
        weights = list_f32(_f32_ptr(weights_addr), n_rows)
    var forest: RandomForestMetaData[DT, CLT]
    # DEVIATION 2510 -- the binding's own stage table (MOJOLEARN_STAGE_TIMES=1
    # only): what this entry point spends OUTSIDE `fit_forest`'s fit_total,
    # host-stamped, so the Python-side residual can be split. No arithmetic.
    var t_s = bt.start()
    var ctx = process_ctx[_DEVCTX_SLOT]()
    bt.stop_host("bind_ctx_create", t_s)
    t_s = bt.start()
    var hy = ctx.enqueue_create_host_buffer[CLT](n_rows)
    ctx.synchronize()
    bt.stop_host("bind_pinned_alloc", t_s)
    t_s = bt.start()
    # DEVIATION 2481: bulk typed copies preserve every input bit while
    # avoiding scalar stores into pinned memory.
    unsafe_memcpy(dest=hy.unsafe_ptr(), src=yp, count=n_rows)
    bt.stop_host("bind_host_copy", t_s)
    t_s = bt.start()
    # cpu-gpu-cleanup t-forest: X goes to the device as the caller's
    # bytes and is transposed there when the caller lent its ROW-major
    # C-order block (ROWMAJOR); the NaN refusal is a device scan
    # (`ensemble/device_layout.mojo`). No host pass over X: the threaded
    # pinned-stage transpose and host NaN scan (DEVIATION 2637) are gone.
    # The builder's host view of X (`host_x_addr`, DEVIATION 2484) is the
    # caller's column-major block when it lent one, else none.
    var dx = upload_forest_x(ctx, xp, n_rows, n_cols, ROWMAJOR)
    # FAST on Apple (lane apple-fast-rfet-scan): the NaN-and-inf device
    # scan below (ensemble/device_finite.mojo) stands in for this NaN scan.
    comptime if not FOREST_DEVICE_FINITE:
        if device_has_nan_f32(ctx, dx, n_rows * n_cols):
            raise Error("rf_classifier_fit: " + RF_NAN_REFUSAL)
    var host_x = 0 if ROWMAJOR else Int(xp)
    # sk-weq-trees: the rows the fit sees (fewer when zero weights drop)
    var n_fit = n_rows
    var dy = ctx.enqueue_create_buffer[CLT](n_rows)
    ctx.enqueue_copy(dst_buf=dy, src_ptr=hy.unsafe_ptr())
    # The host weights drive sampling; non-bootstrap objectives read the
    # device weights at original row IDs. Keep both alive through fitting.
    var dsw = ctx.enqueue_create_buffer[DT](max(1, len(weights)))
    comptime if FOREST_DEVICE_FINITE:
        var fscan = ForestFiniteScan(ctx)
        fscan.enqueue(ctx, dx, n_rows * n_cols)
        ctx.synchronize()
        fscan.refuse_if_bad("rf_classifier_fit: ")
    else:
        ctx.synchronize()
    bt.stop_host("bind_h2d", t_s)
    if weighted:
        ctx.enqueue_copy(dst_buf=dsw, src_ptr=weights.unsafe_ptr())
        var wscan = device_scan_weights(ctx, dsw, n_rows)
        if wscan.bad:
            raise Error("class weights must be finite and nonnegative")
        if wscan.total <= 0:
            raise Error("class weights must have positive total")
        # sk-weq-trees: zero-weight rows leave the fit (no bootstrap);
        # the kept weights are rescanned (all 1.0 after the drop: the
        # count objective, the fit on the repeated rows)
        if not rf_params.bootstrap and not wscan.all_unit:
            n_fit = _rf_drop_zero_weight_rows[CLT](
                ctx, dx, dy, dsw, weights, n_rows, n_cols
            )
            if n_fit < n_rows:
                host_x = 0
                wscan = device_scan_weights(ctx, dsw, n_fit)
        weight_total = wscan.total
        if wscan.all_unit:
            weights = List[Float32]()
    # cuML randomforest.cuh dispatches weighted objectives only when
    # bootstrap is disabled: sampling already applies bootstrap weights.
    if len(weights) > 0 and not rf_params.bootstrap:
        var scale = choose_scale(weight_total, n_fit)
        if scale < Float64(1.1754943508222875e-38) or scale > Float64(3.4028234663852886e38):
            raise Error("class weights exceed Float32 fixed-point scale range")
        var scales = BinScales(Float32(1), Float32(scale))
        forest = fit_forest[WeightedClsObj](
            ctx, dx, dy, dsw, n_fit, n_cols, n_classes, rf_params,
            scales, sample_weight_host=weights, host_x_addr=host_x, tree_start=tree_start, oob_score=oob_addr != 0, feature_importances=importance_addr != 0,
            min_weight_leaf=_rf_min_weight_leaf(mw_frac, weight_total),
            notices_to_stderr=False,
        )
    else:
        _rf_fold_min_weight_count(rf_params, mw_frac, n_fit)
        forest = fit_forest[ClsObj](
            ctx, dx, dy, dsw, n_fit, n_cols, n_classes, rf_params,
            sample_weight_host=weights, host_x_addr=host_x, tree_start=tree_start, oob_score=oob_addr != 0, feature_importances=importance_addr != 0,
            notices_to_stderr=False,
        )
    t_s = bt.start()
    ctx.synchronize()
    _ = dx^
    _ = dy^
    _ = dsw^
    _ = hy^
    bt.stop_host("bind_release_buffers", t_s)
    t_s = bt.start()
    # DEVIATION 1946: THE CONTEXT DIES LAST. Mojo destroys a value at its
    # LAST USE, so without this line `ctx`'s last use is the
    # `synchronize()` above and the five buffers -- two of them PINNED
    # HOST allocations -- are freed against a context that is already
    # gone. `ensemble/checks/rf_ctx_order_probe.mojo` is that ordering
    # in 60 lines with no Python. It is the same class as DEVIATION 1944
    # (a device buffer freed against a context that is not the live one),
    # and it is why the pure-Mojo probe passed where this binding hung:
    # `rf_ctx_probe.mojo::one_fit` takes `ctx` as a BORROWED argument, so
    # the caller's frame keeps it alive past every release.
    _ = ctx^
    bt.stop_host("bind_release_ctx", t_s)
    _ = weights^
    if oob_addr != 0:
        if oob_score_addr == 0 or not forest.has_oob:
            raise Error("RF OOB export requires output and score buffers")
        # T15 native public output boundary: borrowed result buffers, no Python
        # rows/labels/reductions. The forest owns state through this bulk copy.
        # NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
        var oob_out = MutPointer[Float64,MutUntrackedOrigin](unsafe_from_address=oob_addr)
        unsafe_memcpy(dest=oob_out,src=forest.oob_decision_function_.unsafe_ptr(),count=len(forest.oob_decision_function_))
        MutPointer[Float64,MutUntrackedOrigin](unsafe_from_address=oob_score_addr)[unsafe_offset=0] = forest.oob_score_
    if importance_addr != 0:
        if len(forest.feature_importances_) != n_cols:
            raise Error("RF feature importance output does not match feature count")
        unsafe_memcpy(dest=_f32_ptr(importance_addr),src=forest.feature_importances_.unsafe_ptr(),count=n_cols)
    return forest^


def _rf_regressor_fit_forest[ROWMAJOR: Bool = False](params: ForestFitArgs, mut bt: StageTimes) raises -> RandomForestMetaData[DT, RLT]:
    """The typed body of `_rf_regressor_fit` (wave 8 run split): the checks,
    the uploads and the fit, run by the binding with the GIL released;
    the auxiliary outputs are written here. The binding keeps the fit
    (`_rf_keep_forest`, under the GIL)."""
    var tree_start = params.tree_start
    var oob_addr = params.oob
    var oob_score_addr = params.score
    var importance_addr = params.importance
    var weights_addr = params.weights
    var weight_max = params.weight_max
    if (oob_addr == 0) != (oob_score_addr == 0):
        raise Error("RF OOB output and score must be requested together")
    if not _rf_params_len_ok(params):
        raise Error(
            "rf_regressor_fit: params must hold "
            + String(N_RF_FIT_PARAMS)
            + " values (17 with min_weight_fraction_leaf), got "
            + String(len(params))
        )
    if params.i(2) != 0:
        raise Error("rf_regressor_fit: n_classes (slot 2) must be 0")
    var n_rows = params.i(0)
    var n_cols = params.i(1)
    var xp = _f32_ptr(params.x)
    var yp = _f32_ptr(params.y)
    var crit = params.criterion
    _check_criterion("rf_regressor_fit", crit, _reg_criteria())
    var rf_params = _rf_params_from(params, crit)
    var mw_frac = _rf_min_weight_fraction(params)
    # sk-edges: the row weights of `rf_regressor_fit_weighted` (0: none), one
    # memcpy as in the classifier; their checks run on the device below.
    var weights = List[Float32]()
    var weight_total = Float64(0)
    var weighted = weights_addr != 0
    if weighted:
        weights = list_f32(_f32_ptr(weights_addr), n_rows)

    var forest: RandomForestMetaData[DT, RLT]
    # DEVIATION 2510 -- as in the classifier: the binding's own host stamps.
    var t_s = bt.start()
    var ctx = process_ctx[_DEVCTX_SLOT]()
    bt.stop_host("bind_ctx_create", t_s)
    t_s = bt.start()
    var hy = ctx.enqueue_create_host_buffer[RLT](n_rows)
    ctx.synchronize()
    bt.stop_host("bind_pinned_alloc", t_s)
    t_s = bt.start()
    # DEVIATION 2481: bulk typed copies preserve every input bit while
    # avoiding scalar stores into pinned memory.
    copy_f32(yp, hy.unsafe_ptr(), n_rows)
    bt.stop_host("bind_host_copy", t_s)
    t_s = bt.start()
    # cpu-gpu-cleanup t-forest, as in the classifier: device upload,
    # device transpose, device NaN scan.
    var dx = upload_forest_x(ctx, xp, n_rows, n_cols, ROWMAJOR)
    # FAST on Apple (lane apple-fast-rfet-scan): the NaN-and-inf device
    # scan below (ensemble/device_finite.mojo) stands in for this NaN scan.
    comptime if not FOREST_DEVICE_FINITE:
        if device_has_nan_f32(ctx, dx, n_rows * n_cols):
            raise Error("rf_regressor_fit: " + RF_NAN_REFUSAL)
    var host_x = 0 if ROWMAJOR else Int(xp)
    # sk-weq-trees: the rows the fit sees (fewer when zero weights drop)
    var n_fit = n_rows
    var dy = ctx.enqueue_create_buffer[RLT](n_rows)
    ctx.enqueue_copy(dst_buf=dy, src_ptr=hy.unsafe_ptr())
    var dsw = ctx.enqueue_create_buffer[DT](max(1, len(weights)))
    comptime if FOREST_DEVICE_FINITE:
        var fscan = ForestFiniteScan(ctx)
        fscan.enqueue(ctx, dx, n_rows * n_cols)
        ctx.synchronize()
        fscan.refuse_if_bad("rf_regressor_fit: ")
    else:
        ctx.synchronize()
    bt.stop_host("bind_h2d", t_s)
    if weighted:
        # the classifier's weight checks and all-unit test, on the device
        ctx.enqueue_copy(dst_buf=dsw, src_ptr=weights.unsafe_ptr())
        var wscan = device_scan_weights(ctx, dsw, n_rows)
        if wscan.bad:
            raise Error("sample weights must be finite and nonnegative")
        if wscan.total <= 0:
            raise Error("sample weights must have positive total")
        # sk-weq-trees: zero-weight rows leave the fit, as in the
        # classifier (`_rf_drop_zero_weight_rows`)
        if not rf_params.bootstrap and not wscan.all_unit:
            n_fit = _rf_drop_zero_weight_rows[RLT](
                ctx, dx, dy, dsw, weights, n_rows, n_cols
            )
            if n_fit < n_rows:
                host_x = 0
                wscan = device_scan_weights(ctx, dsw, n_fit)
        weight_total = wscan.total
        if wscan.all_unit:
            weights = List[Float32]()
    # THE LABEL SCALE IS NOT OPTIONAL. `RegressionBin` accumulates
    # `label_sum` in fixed point through `BinScales.label_scale`
    # (DEVIATION 101b), and the host chooses the scale once per fit
    # from the sum of label magnitudes (`fixed_point.choose_scale`).
    # The unit scale TRUNCATES every |label| < 1 to zero, which fits a
    # forest of zero-leaved stumps -- the build gate caught exactly
    # that. The weight plane stays 1.0: unweighted `Weight()` is a
    # count, which is exact.
    # cpu2-l6-bindings: the label magnitude on the device, in the fixed
    # blocked order the host column restates (`core/abs_sum_blocked`)
    var mag = device_abs_sum_blocked(ctx, dy, n_fit)
    var scales = BinScales(
        Float32(choose_scale(mag, n_fit)), Float32(1.0)
    )
    # `n_unique_labels` is 1 for regression, exactly what
    # `rf_regressor_fit`'s cuML counterpart passes.
    if len(weights) > 0 and not rf_params.bootstrap:
        # sk-edges: the weighted objective (cuML dispatches it only
        # without bootstrap, as for the classifier). `label_sum` holds
        # label * weight, so its scale bounds sum |y| * max w, which is
        # >= sum |y * w| for non-negative weights; the weight plane takes
        # the classifier's scale from the weight total.
        var lscale = choose_scale(mag * max(weight_max, Float64(0)), n_fit)
        var wscale = choose_scale(weight_total, n_fit)
        if wscale < Float64(1.1754943508222875e-38) or wscale > Float64(3.4028234663852886e38):
            raise Error("sample weights exceed Float32 fixed-point scale range")
        forest = fit_forest[WeightedRegObj](
            ctx, dx, dy, dsw, n_fit, n_cols, 1, rf_params,
            BinScales(Float32(lscale), Float32(wscale)), sample_weight_host=weights,
            host_x_addr=host_x, tree_start=tree_start, oob_score=oob_addr != 0, feature_importances=importance_addr != 0,
            min_weight_leaf=_rf_min_weight_leaf(mw_frac, weight_total),
            notices_to_stderr=False,
        )
    else:
        _rf_fold_min_weight_count(rf_params, mw_frac, n_fit)
        forest = fit_forest[RegObj](
            ctx, dx, dy, dsw, n_fit, n_cols, 1, rf_params, scales,
            sample_weight_host=weights,
            host_x_addr=host_x, tree_start=tree_start, oob_score=oob_addr != 0, feature_importances=importance_addr != 0,
            notices_to_stderr=False,
        )
    t_s = bt.start()
    ctx.synchronize()
    _ = dx^
    _ = dy^
    _ = dsw^
    _ = hy^
    bt.stop_host("bind_release_buffers", t_s)
    t_s = bt.start()
    # DEVIATION 1946, as in `rf_classifier_fit_binding`: the context
    # outlives every buffer created on it.
    _ = ctx^
    bt.stop_host("bind_release_ctx", t_s)
    if oob_addr != 0:
        if oob_score_addr == 0 or not forest.has_oob:
            raise Error("RF OOB export requires output and score buffers")
        # T15 native public output boundary: borrowed result buffers, no Python
        # rows/labels/reductions. The forest owns state through this bulk copy.
        # NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
        var oob_out = MutPointer[Float64,MutUntrackedOrigin](unsafe_from_address=oob_addr)
        unsafe_memcpy(dest=oob_out,src=forest.oob_prediction_.unsafe_ptr(),count=len(forest.oob_prediction_))
        MutPointer[Float64,MutUntrackedOrigin](unsafe_from_address=oob_score_addr)[unsafe_offset=0] = forest.oob_score_
    if importance_addr != 0:
        if len(forest.feature_importances_) != n_cols:
            raise Error("RF feature importance output does not match feature count")
        unsafe_memcpy(dest=_f32_ptr(importance_addr),src=forest.feature_importances_.unsafe_ptr(),count=n_cols)
    return forest^


def _rf_keep_forest[LT: DType](mut forest: RandomForestMetaData[DT, LT], mut bt: StageTimes, t_bind: Int) raises -> Int:
    """The tail both fits shared (wave 8 run split): move the fitted trees
    into the export registry and close the binding's stage table. The
    binding calls it holding the GIL, which serializes the registry."""
    var t_x = bt.start()
    var export_trees = forest.trees^
    forest.trees = RFExportTrees()
    var handle = _retain_rf_handle(export_trees^, forest.notices)
    bt.stop_host("bind_export_retain", t_x)
    bt.stop_host("binding_total", t_bind)
    bt.report()
    return handle


def _rf_dart_member_forest(
    mut dart_x: DeviceBuffer[DT],
    mut dart_target: DeviceBuffer[RLT],
    rows_a: Int,
    cols_a: Int,
    n_rows: Int,
    n_cols: Int,
    n: Int,
    d: Int,
    c: Int,
    mut rf_params: RF_params,
) raises -> RandomForestMetaData[DT, RLT]:
    """The GIL-released body of `_rf_regressor_fit_dart` (wave 8 run split):
    the member's X and target on the DART context and its fit, over the
    session buffer handles the binding took under the GIL."""
    var forest: RandomForestMetaData[DT, RLT]
    var ctx = process_ctx[_DART_CTX]()
    var drows = ctx.enqueue_create_buffer[DType.int32](
        n_rows if rows_a != 0 else 1
    )
    var dcols = ctx.enqueue_create_buffer[DType.int32](
        n_cols if cols_a != 0 else 1
    )
    if rows_a != 0:
        ctx.enqueue_copy(dst_buf=drows, src_ptr=_i32_ptr(rows_a))
    if cols_a != 0:
        ctx.enqueue_copy(dst_buf=dcols, src_ptr=_i32_ptr(cols_a))
    var dx = ctx.enqueue_create_buffer[DT](n_rows * n_cols)
    launch_dart_member_x(
        ctx, dart_x, drows, dcols, dx, d, n_rows, n_cols,
        rows_a != 0, cols_a != 0,
    )
    var dy: DeviceBuffer[RLT]
    if rows_a != 0:
        dy = ctx.enqueue_create_buffer[RLT](n_rows)
        var tsub = dart_target.create_sub_buffer[RLT](
            c * n, n
        )
        launch_dart_member_y(ctx, tsub, drows, dy, n_rows)
        ctx.synchronize()
        _ = tsub^
    else:
        # a device copy of the class plane: the fit owns its labels
        dy = ctx.enqueue_create_buffer[RLT](n_rows)
        var tall = dart_target.create_sub_buffer[RLT](
            c * n, n
        )
        ctx.enqueue_copy(dst_buf=dy, src_buf=tall)
        ctx.synchronize()
        _ = tall^
    var dsw = ctx.enqueue_create_buffer[DT](1)
    # the label scale, as `_rf_regressor_fit` chooses it, on the device
    var mag = device_abs_sum_blocked(ctx, dy, n_rows)
    var scales = BinScales(
        Float32(choose_scale(mag, n_rows)), Float32(1.0)
    )
    var prep = List[ForestPrep]()
    forest = fit_forest_prepared[RegObj](
        ctx, dx, dy, dsw, n_rows, n_cols, 1, rf_params, prep, False,
        scales,
        notices_to_stderr=False,
    )
    ctx.synchronize()
    _ = dx^
    _ = dy^
    _ = dsw^
    _ = drows^
    _ = dcols^
    _ = prep^
    _ = ctx^
    return forest^


def _rf_session_rows_forest(
    mut session_dx: DeviceBuffer[DT],
    rp: MutPointer[Int32, MutUntrackedOrigin],
    yp: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    n_cols: Int,
    src_rows: Int,
    mut rf_params: RF_params,
) raises -> RandomForestMetaData[DT, RLT]:
    """The GIL-released body of `rf_regressor_fit_session_rows_binding`
    (wave 8 run split): the device gather of the rows and the member fit,
    over the session X handle the binding took under the GIL."""
    var forest: RandomForestMetaData[DT, RLT]
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var hrows = ctx.enqueue_create_host_buffer[CLT](n_rows)
    var hy = ctx.enqueue_create_host_buffer[RLT](n_rows)
    ctx.synchronize()
    unsafe_memcpy(dest=hrows.unsafe_ptr(), src=rp, count=n_rows)
    copy_f32(yp, hy.unsafe_ptr(), n_rows)
    var drows = ctx.enqueue_create_buffer[CLT](n_rows)
    ctx.enqueue_copy(dst_buf=drows, src_ptr=hrows.unsafe_ptr())
    # cpu2-l6-bindings: the row-id range refusal on the device, before
    # the gather reads through the ids (one Int32 read back)
    if device_any_index_out_of_range(ctx, drows, n_rows, src_rows):
        raise Error("rf_regressor_fit_session_rows: a row id is outside the session's X")
    var dy = ctx.enqueue_create_buffer[RLT](n_rows)
    ctx.enqueue_copy(dst_buf=dy, src_ptr=hy.unsafe_ptr())
    var dsw = ctx.enqueue_create_buffer[DT](1)
    var dxg = ctx.enqueue_create_buffer[DT](n_rows * n_cols)
    launch_gather_rows_colmajor(
        ctx,
        session_dx,
        drows,
        dxg,
        src_rows,
        n_rows,
        n_cols,
    )
    # cpu2-l6-bindings: the label magnitude on the device, in the fixed
    # blocked order the host column restates (`core/abs_sum_blocked`)
    var mag = device_abs_sum_blocked(ctx, dy, n_rows)
    var scales = BinScales(
        Float32(choose_scale(mag, n_rows)), Float32(1.0)
    )
    forest = fit_forest[RegObj](
        ctx, dxg, dy, dsw, n_rows, n_cols, 1, rf_params, scales,
        notices_to_stderr=False,
    )
    ctx.synchronize()
    _ = dxg^
    _ = drows^
    _ = dy^
    _ = dsw^
    _ = hrows^
    _ = hy^
    _ = ctx^
    return forest^


def _rf_data_session_stage(
    xp: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    n_cols: Int,
    row_major: Bool,
    share_tables: Bool,
    mut made: List[RfDataSession],
) raises -> Bool:
    """The GIL-released body of `rf_data_session_open_binding` (wave 8 run
    split): the upload and the finite scan; the staged session goes into
    `made` unless a cell is not finite (returns True then)."""
    var bad = False
    var ctx = process_ctx[_DEVCTX_SLOT]()
    # cpu-gpu-cleanup t-forest: the caller's bytes go to the device as
    # they are, the transpose and the finite scan run there. The session
    # keeps no host copy of X (`host_x` 0: the borrow ends with this
    # call), so a member's OOB pass reads the device plane.
    var dx = upload_forest_x(ctx, xp, n_rows, n_cols, row_major)
    # Lane a8c-device-x-linear-prep-rfe: the scan tests NaN AND infinity
    # (ensemble/device_finite.mojo, bits of the exponent), so the boost
    # meta-estimators' host scan of X before the open
    # (bindings/trees_public_boost.mojo, TREES_NONFINITE) is deleted:
    # this one device pass over the upload the session makes anyway is
    # the only test. One synchronize, as the NaN scan had.
    var scan = ForestFiniteScan(ctx)
    scan.enqueue(ctx, dx, n_rows * n_cols)
    ctx.synchronize()
    bad = scan.hflag.unsafe_ptr().unsafe_load(0) != 0
    _ = scan^
    if bad:
        _ = dx^
    else:
        made.append(
            RfDataSession(0, dx^, n_rows, n_cols, 0, share_tables)
        )
    _ = ctx^
    return bad


def _rf_ada_open_stage(
    wp: MutPointer[UInt64, MutUntrackedOrigin],
    ya: Int,
    ca: Int,
    n: Int,
    k: Int,
    mut made: List[AdaState],
) raises:
    """The GIL-released body of `rf_ada_open_binding` (wave 8 run split):
    the AdaBoost state's device buffers, into `made`."""
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var dw = ctx.enqueue_create_buffer[DType.uint64](n)
    ctx.enqueue_copy(dst_buf=dw, src_ptr=wp)
    var dc = ctx.enqueue_create_buffer[DType.int32](n if k > 0 else 1)
    var dv = ctx.enqueue_create_buffer[DType.float32](n if k == 0 else 1)
    if k > 0:
        ctx.enqueue_copy(dst_buf=dc, src_ptr=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=ya))
    else:
        ctx.enqueue_copy(dst_buf=dv, src_ptr=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=ya))
    var dcw = ctx.enqueue_create_buffer[DType.float32](n if ca != 0 else 1)
    if ca != 0:
        ctx.enqueue_copy(dst_buf=dcw, src_ptr=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=ca))
    # the borrows end with this call
    ctx.synchronize()
    made.append(AdaState(n, k, dw^, dc^, dv^, dcw^, ca != 0))
    _ = ctx^


def _rf_ada_regressor_round(
    mut session_dx: DeviceBuffer[DT],
    mut state: AdaState,
    n_rows: Int,
    n_cols: Int,
    mut rf_params: RF_params,
    seed: Int,
    round: Int,
    loss: Int,
    lr: Float64,
    last: Bool,
    stats: MutPointer[Float64, MutUntrackedOrigin],
) raises -> RandomForestMetaData[DT, RLT]:
    """The GIL-released body of `rf_regressor_fit_ada_round_binding` (wave 8
    run split): the weighted draw, the gather, the member fit, its
    predictions of the training X and the R2 step, over the session X
    handle and AdaBoost state the binding took under the GIL."""
    var forest: RandomForestMetaData[DT, RLT]
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var drows = ctx.enqueue_create_buffer[CLT](n_rows)
    weighted_sample_on(ctx, state.w, n_rows, drows, n_rows, seed, round)
    var dxg = ctx.enqueue_create_buffer[DT](n_rows * n_cols)
    launch_gather_rows_colmajor(ctx, session_dx, drows, dxg, n_rows, n_rows, n_cols)
    var dy = ctx.enqueue_create_buffer[RLT](n_rows)
    var dvals = state.values.copy()
    launch_dart_member_y(ctx, dvals, drows, dy, n_rows)
    var dsw = ctx.enqueue_create_buffer[DT](1)
    # the label scale, as `_rf_regressor_fit` chooses it, on the device
    var mag = device_abs_sum_blocked(ctx, dy, n_rows)
    var scales = BinScales(Float32(choose_scale(mag, n_rows)), Float32(1.0))
    forest = fit_forest[RegObj](
        ctx, dxg, dy, dsw, n_rows, n_cols, 1, rf_params, scales,
        notices_to_stderr=False,
    )
    var nodes = ada_upload_nodes(ctx, forest.trees)
    var dpred = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var dfull = session_dx.create_sub_buffer[DT](0, n_rows * n_cols)
    ada_predict_values(ctx, nodes, dfull, n_rows, dpred)
    var moved = r2_step_on(ctx, state.w, dpred, state.values, n_rows, loss, lr, last, stats)
    var total = stats[unsafe_offset=3]
    if moved and total > 0:
        ada_rescale(ctx, state, total)
    ctx.synchronize()
    _ = nodes^
    _ = dpred^
    _ = dfull^
    _ = dxg^
    _ = drows^
    _ = dy^
    _ = dvals^
    _ = dsw^
    _ = ctx^
    return forest^


def _rf_data_session_close(id: Int) raises:
    """`rf_data_session_close` (wave 8 run split): drop the session after
    the context drains. The binding calls it holding the GIL."""
    var reg = RF_SESSIONS.get_or_create_ptr()
    var si = reg[].find(id)
    var session = reg[].sessions.pop(si)
    var ctx = process_ctx[_DEVCTX_SLOT]()
    ctx.synchronize()
    _ = session^
    _ = ctx^
