# SPDX-License-Identifier: Apache-2.0
"""RUN of the Extra Trees GPU binding (bindings/_mojolearn_trees.mojo): the
fit's slot checks, config and device fit, the forest predict through the
implemented traversal, and the borrowed-buffer helpers. Typed Mojo values
and addresses only, no Python import (tools/python_free_dirs.sh). The door
releases the GIL around each RUN fit call and keeps the export registry
(ET_EXPORTS: its `ForestExportRegistry` lives in the Python-importing
bindings/forest_export_binding.mojo) and the PACK. Lane rs-boosting-trees
(2026-11-12): moved word for word; the fit's GIL release is hoisted to the
door around the whole RUN call (the label copy and the slot checks now run
with the GIL released too). Same operands in the same order, so the bits
are unchanged."""
from core.hostptr import list_f32, list_i32
from core.forest_fit_request import ForestFitArgs
from core.neural_context import process_ctx
from checks.numerics import GLOBAL_NUMERIC_MODE as _DEVCTX_MODE, NUMERIC_IDENTICAL as _DEVCTX_IDENTICAL
from ensemble.instruments import StageTimes
from extratrees.estimator import (
    ExtraTreesConfig,
    FitResult,
    fit_extra_trees_classifier_device,
    fit_extra_trees_regressor_device,
)
from extratrees.impl.decisiontree.decisiontree import (
    CRITERION_ENTROPY,
    CRITERION_GINI,
    CRITERION_GAMMA,
    CRITERION_INVERSE_GAUSSIAN,
    CRITERION_MSE,
    CRITERION_POISSON,
)
from core.forest_inference_model import (
    resident_prepare, resident_predict_into, resident_release,
)


comptime N_FIT_PARAMS = 22
"""`params` for both fit entry points, in this exact order -- the wrapper
names the same order in the same words, because a silent reordering here is a
wrong answer rather than a failure:

     0  n_rows
     1  n_features
     2  n_classes        (classifier only; MUST be 0 for the regressor)
     3  n_estimators
     4  max_depth        (-1 = sklearn's None)
     5  min_samples_split
     6  min_samples_leaf
     7  min_weight_fraction_leaf  (float)
     8  max_features_spec         (positive count, 0 = fraction at slot 9,
                                   -1 sqrt, -2 log2, -3 all -- the
                                   estimator.mojo sentinels)
     9  max_features_fraction     (float)
    10  min_impurity_decrease     (float)
    11  bootstrap        (0/1)
    12  oob_score        (0/1)
    13  random_state
    14  warm_start       (0/1)
    15  ccp_alpha        (float)
    16  has_class_weight (0/1)
    17  has_monotonic_cst(0/1)
    18  max_samples      (sklearn's max_samples RESOLVED TO A COUNT by the
                          wrapper: 0 = None; honoured with bootstrap=1,
                          refused by name otherwise -- DEVIATION 460. This
                          slot was a 0/1 `max_samples_set` flag while
                          bootstrap was refused)
    19  max_leaf_nodes   (-1 = sklearn's None)
    20  device           (must be 1: GPU; CPU training retired)
    21  criterion        (the `decisiontree.mojo` CRITERION_* code:
                          0 GINI, 1 ENTROPY for the classifier; 2 MSE for
                          the regressor -- DEVIATION 459)

Every refused sklearn parameter RIDES THROUGH so the refusal fires in
`refuse_unported` by name, in one place, rather than being re-decided at this
boundary. The criterion RIDES AS SLOT 21 since DEVIATION 459 (it did
not while each entry point had exactly one criterion): `et_classifier_fit`
admits GINI and ENTROPY, `et_regressor_fit` admits MSE, and the OTHER
criteria are refused by name in the WRAPPER, which cites the same
NOT_IMPLEMENTED.tsv rows validity_check does.
"""


#: This binding's ONE process-lifetime DeviceContext (core/neural_context.mojo,
#: lane/devctx-lifetime): a context per call exhausts Metal command queues.
comptime _DEVCTX_SLOT = "MojolearnTreesContextIdenticalV3" if _DEVCTX_MODE == _DEVCTX_IDENTICAL else "MojolearnTreesContextFastV3"


def et_legacy_meta(result: FitResult) -> List[Int64]:
    """The six metadata words `_forest_out` reports (and the export keeps)."""
    var meta: List[Int64] = [Int64(result.forest.n_trees), Int64(result.forest.num_outputs),
        Int64(1 if result.depth_cap_bound else 0), Int64(result.plan.params.max_depth),
        Int64(result.plan.max_features_count), Int64(result.plan.n_sampled_rows)]
    return meta^


def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null float32 buffer address")
    return MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=addr)


def _i32_ptr(addr: Int) raises -> MutPointer[Int32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null int32 buffer address")
    return MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=addr)


def _copy_f32(addr: Int, n: Int) raises -> List[Float32]:
    """Borrowed host memory into an owned List, read while the GIL-holding
    caller keeps the array alive (the `_arrays.py` contract)."""
    # one memcpy (no per-element host loop)
    return list_f32(_f32_ptr(addr), n)


def _config_from(
    params: ForestFitArgs, base: ExtraTreesConfig
) raises -> ExtraTreesConfig:
    """Slots 3-19 and 21 written OVER `base`. Read under the GIL.

    `base` carries the defaults the slots then overwrite; since DEVIATION
    459 the criterion is slot 21 and `base`'s criterion is only a default
    the slot replaces. `et_classifier_fit` passes `ExtraTreesConfig()` and
    `et_regressor_fit` passes `ExtraTreesConfig().for_regression()`. Every
    slot overwrites its field AFTER the base is taken,
    so no default can shadow what the caller sent -- DEVIATION 458 is what
    happened when the regressor applied `for_regression()` the other way round
    and its `max_features_spec = ALL` default overwrote slots 8-9 of every
    regressor fit.
    """
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
    # slot 21, DEVIATION 459: the criterion code, written over the base's
    # default like every other slot.
    config.criterion = Int32(params.i(21))
    return config^


def _et_classifier_fit_run[ROWMAJOR: Bool = False](
    params: ForestFitArgs, mut times: StageTimes, mut total_start: Int
) raises -> FitResult:
    """Fit `ExtraTreesClassifier`. `x` is COLUMN-major float32
    (n_rows * n_features), `y` is float32 class CODES in [0, n_classes).
    See `N_FIT_PARAMS`. Returns the fit; the door `_et_classifier_fit`
    calls this with the GIL released and keeps the fit in ET_EXPORTS."""
    if len(params) != N_FIT_PARAMS:
        raise Error(
            "et_classifier_fit: params must hold "
            + String(N_FIT_PARAMS)
            + " values, got "
            + String(len(params))
        )
    var n_rows = params.i(0)
    var n_features = params.i(1)
    var n_classes = params.i(2)
    if params.f(20) != Float64(1):
        raise Error("Extra Trees training is GPU-only; device (slot 20) must be 1")
    var config = _config_from(params, ExtraTreesConfig())
    if config.criterion != CRITERION_GINI and config.criterion != CRITERION_ENTROPY:
        raise Error(
            "et_classifier_fit: criterion (slot 21) must be GINI (0) or"
            " ENTROPY (1); got " + String(config.criterion)
        )
    # DEVIATION 2480: optional host boundary attribution, no added drains.
    total_start = times.start()
    var stamp = times.start()
    # DEVIATION 2481: caller keeps the immutable feature buffer alive through
    # this synchronous fit; only labels still require owned staging.
    var x_pointer = params.x
    _ = _f32_ptr(x_pointer)
    var x = List[Float32]()
    times.stop_host("boundary_x_borrow", stamp)
    stamp = times.start()
    var y = _copy_f32(params.y, n_rows)
    times.stop_host("boundary_y_list", stamp)
    stamp = times.start()

    var result: FitResult
    var ctx = process_ctx[_DEVCTX_SLOT]()
    result = fit_extra_trees_classifier_device(
        ctx, x, y, Int32(n_rows), Int32(n_features), Int32(n_classes),
        config, x_addr=x_pointer, x_row_major=ROWMAJOR, tree_start=params.tree_start,
        weights_addr=params.weights,
    )
    ctx.synchronize()
    times.stop_host("boundary_device_fit_and_context", stamp)
    return result^


def _et_regressor_fit_run[ROWMAJOR: Bool = False](
    params: ForestFitArgs, mut times: StageTimes, mut total_start: Int
) raises -> FitResult:
    """Fit `ExtraTreesRegressor`. Same contract; slot 2 MUST be 0.
    `weights_addr` (lane sk-final, 0: none) as `et_regressor_fit_weighted`.
    The door `_et_regressor_fit` calls this with the GIL released."""
    if len(params) != N_FIT_PARAMS:
        raise Error(
            "et_regressor_fit: params must hold "
            + String(N_FIT_PARAMS)
            + " values, got "
            + String(len(params))
        )
    if params.i(2) != 0:
        raise Error("et_regressor_fit: n_classes (slot 2) must be 0")
    var n_rows = params.i(0)
    var n_features = params.i(1)
    if params.f(20) != Float64(1):
        raise Error("Extra Trees training is GPU-only; device (slot 20) must be 1")
    var config = _config_from(params, ExtraTreesConfig().for_regression())
    if config.criterion != CRITERION_MSE and config.criterion != CRITERION_POISSON and (
        config.criterion != CRITERION_GAMMA and config.criterion != CRITERION_INVERSE_GAUSSIAN
    ):
        raise Error(
            "et_regressor_fit: criterion (slot 21) must be MSE (2), POISSON (4), GAMMA (5) or INVERSE_GAUSSIAN (6); got "
            + String(config.criterion)
        )
    # DEVIATION 2480: optional host boundary attribution, no added drains.
    total_start = times.start()
    var stamp = times.start()
    # DEVIATION 2481: caller keeps the immutable feature buffer alive through
    # this synchronous fit; only labels still require owned staging.
    var x_pointer = params.x
    _ = _f32_ptr(x_pointer)
    var x = List[Float32]()
    times.stop_host("boundary_x_borrow", stamp)
    stamp = times.start()
    var y = _copy_f32(params.y, n_rows)
    times.stop_host("boundary_y_list", stamp)
    stamp = times.start()

    var result: FitResult
    var ctx = process_ctx[_DEVCTX_SLOT]()
    result = fit_extra_trees_regressor_device(
        ctx, x, y, Int32(n_rows), Int32(n_features), config,
        x_addr=x_pointer, x_row_major=ROWMAJOR, tree_start=params.tree_start,
        weights_addr=params.weights,
    )
    ctx.synchronize()
    times.stop_host("boundary_device_fit_and_context", stamp)
    return result^


def et_predict_native(model: List[Int], x_address: Int, out_address: Int, n_rows: Int, n_features: Int,
                      n_trees: Int, num_outputs: Int) raises -> Int:
    """The forest's averaged vote per row, through the IMPLEMENTED traversal
    (XF_PREDICT).

    Model arrays are int32/float32 as `_forest_out` laid them out. `x` here
    is ROW-major (the traversal reads `row[offset + colid]`). `out` is
    n_rows * num_outputs float32 and receives `forest_vote_host`'s average --
    per-class probabilities for the classifier (argmax is the wrapper's,
    exactly as `RandomForest::predict` argmaxes over `predict_proba`), the
    mean prediction for the regressor. Returns rows written.
    """
    if n_trees < 1 or num_outputs < 1:
        raise Error("et_predict: n_trees and num_outputs must be >= 1")

    var offsets_p = _i32_ptr(model[0])
    var colid_p = _i32_ptr(model[1])
    var quesval_p = _f32_ptr(model[2])
    var left_p = _i32_ptr(model[3])
    var leaves_p = _f32_ptr(model[4])
    var xp = _f32_ptr(x_address)
    var op = _f32_ptr(out_address)

    # cpu-gpu-cleanup t-gbdt: the `sequential` engine runs on the device.
    # The host walk (`core/forest_host_predict.mojo`'s `et_host_predict`,
    # DEVIATION 2900) is the CPU-only install's (`_mojolearn_trees_host`);
    # here `forest_vote_host`'s arithmetic -- zero, add every tree's leaf in
    # increasing tree order, divide by `Float32(n_trees)` -- is
    # `forest_ordered_kernel`, one thread per (row, output), through a
    # one-call ordered snapshot. The GIL stays held for the registry.
    if n_rows == 0:
        return 0
    var n_nodes = Int(offsets_p[n_trees])
    if Int(offsets_p[0]) != 0 or n_nodes < n_trees:
        raise Error("et_predict: tree_offsets must start at 0 and hold at least one node per tree")
    if n_nodes > 2147483647 // num_outputs:
        raise Error("et_predict: node/output count exceeds Int32")
    # one memcpy per array (no per-node host loop); the snapshot uploads once
    var offsets = list_i32(offsets_p, n_trees + 1)
    var columns = list_i32(colid_p, n_nodes)
    var thresholds = list_f32(quesval_p, n_nodes)
    var left = list_i32(left_p, n_nodes)
    var leaves = list_f32(leaves_p, n_nodes * num_outputs)
    var handle = resident_prepare[False](
        offsets, columns, thresholds, left, leaves, n_features, num_outputs, True
    )
    try:
        resident_predict_into[False](
            handle,
            xp.unsafe_origin_cast[MutAnyOrigin](),
            op.unsafe_origin_cast[MutAnyOrigin](),
            n_rows, n_features, num_outputs, True,
        )
    except e:
        resident_release[False](handle)
        raise e
    resident_release[False](handle)
    return n_rows
