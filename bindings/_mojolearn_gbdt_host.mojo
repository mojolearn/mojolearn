# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for the `_mojolearn_gbdt` family, GradientBoosting on the
gbdt-symmetric, gbdt-rmse, gbdt-depthwise and gbdt-lossguide lanes
(workstream E batch 3, 2026-09-14) and the gbdt-nan-modes, gbdt-adapter-clf,
gbdt-adapter-reg, gbdt-parametric-losses, gbdt-exact-mae,
gbdt-lossguide-newtoncosine, gbdt-multiclass and gbdt-onevsall lanes
(lane/cpu-training-gbdt-losses, 2026-09-15: the pointwise losses, the Exact
leaves and the row bootstraps through `gbdt/host/gbdt_oracle_losses.mojo`,
the Lossguide searcher options through `gbdt/host/gbdt_oracle_depthwise.mojo`,
MultiClass and MultiClassOneVsAll through
`gbdt/host/gbdt_oracle_multiclass.mojo`. `loss="RMSE"` fits through
`gbdt/host/gbdt_oracle_rmse.mojo::gbdt_rmse_host_fit`.

HOST ONLY. No DeviceContext, no kernel launch, no GPU. The fit is
`gbdt/host/gbdt_oracle.mojo::gbdt_host_fit` for SymmetricTree and
`gbdt/host/gbdt_oracle_depthwise.mojo::gbdt_host_fit_non_symmetric` for
Depthwise and Lossguide (the `ntree` / `node` model shape), the device trainer
(`gbdt/train.mojo::train`, `gbdt/methods/doc_parallel_boosting.mojo::
fit_with_test`, the greedy symmetric searcher and the Newton leaf estimator)
restated on the host, so the model text is meant to be the GPU columns'
bytes. The predict entry is `core/gbdt_host_predict.mojo::gbdt_host_predict`,
the walk the forest host binding runs and the forest host gate holds to the
recorded GPU predictions, over the model text parsed here.

THE EXPORTED NAMES ARE THE GPU BINDING'S NAMES for what this covers, so
`python/mojolearn/ensemble.py` runs unchanged on a CPU-only install through
`_backend._HOST_MODULES` (`"_mojolearn_gbdt": "_mojolearn_gbdt_host"`):
`gbdt_fit` (the same eight arguments, the same 35-slot params list with its
counted class-weight tail and optional float tail, the same four strings,
the same five-element return), `gbdt_predict` and `gbdt_model_dim` (the
model text in, as `bindings/_mojolearn_gbdt.mojo:410-450`), `gbdt_sigmoid_pair`
(its body), `gbdt_binary_probabilities` and `gbdt_binary_classes` (the
classifier adapter's transforms, `bindings/_mojolearn_gbdt.mojo:107-130`
over `gbdt/binary_prediction.mojo`'s kernel, restated per element),
`gbdt_predict_multi` (a multi-dimensional model's raw, softmax and sigmoid
columns), `gbdt_vendor` answering "cpu" and `gbdt_numeric_mode`.

`gbdt_fit_ordered_rmse` (the gbdt-ordered-rmse lane, through
`gbdt/host/gbdt_oracle_ordered.mojo`) and `gbdt_fit_two_level_feature_freq`
(the gbdt-feature-freq lane, through `gbdt/host/gbdt_oracle_feature_freq.mojo`)
take the GPU binding's params and return its model text
(lane/cpu-training-gbdt-ordered, 2026-09-15); both refuse `sample_weight` by
name. `gbdt_fit` with `use_pointwise_searcher=True` trains the
gbdt-pointwise-l2-bayesian-eval lane (`_gbdt_fit_pointwise_arm`,
`gbdt/host/gbdt_oracle_pointwise.mojo`) and one-hot categorical columns train
under SymmetricTree with Logloss (gbdt-categorical-ctr,
`gbdt/host/gbdt_oracle_onehot.mojo`).

AN EVAL SET, THE OVERFITTING DETECTOR AND `use_best_model` on the PLAIN
SymmetricTree Logloss fit (lane/close-no-cpu-path-gbdt, 2026-09-20, the
gbdt-symmetric-eval lane), through `gbdt/host/gbdt_oracle_eval.mojo`:
their `testCursor` (`_apply_last_tree_to_test`), the held-out curve
(`_test_loss` through the same Logloss kernel the learn curve uses), the
shared detector and `ShrinkToBestIteration`. An eval set does not reach the
learn cursor, the borders, the splits or the leaves on this path, so the
model text of a fit WITH one is the model text of the same fit without one
and what the arm adds is `test_loss_curve_`, `best_iteration_` and
`stopped_early_`. Every OTHER Plain arm (RMSE, the pointwise losses,
MultiClass, Depthwise, Lossguide) still refuses an eval set BY NAME: their
test cursor runs the same four pieces over a different loss kernel or a
different model shape, and neither is restated.

ABSENT, and so refused BY NAME through `_HostBinding`: `gbdt_per_round_paths`
and the two `*_parallel_available` probes.

NaN IN X. The oracle's grid places the NaN border (`nan_mode` Min or Max,
`_calc_quantization_phase_b`) and `_binarize_columns` substitutes the value
before binning, which is the device fit's order, so an X carrying NaN trains
through the same entries on SymmetricTree with Logloss (the gbdt-nan-modes
lane). Under RMSE or a non-symmetric policy NaN is unmeasured and still
refused by name.

REFUSED BY NAME INSIDE `gbdt_fit`, with the sentence
`cpu_identity_gate_check.py` requires ("no CPU implementation of"), every
parameter value the oracle does not restate: see `_refuse` and the oracle's
module docstring. Every other GBDT lane of tools/identity_break.py therefore
reads REFUSED on a CPU-only install, as it did before this binding existed.

The sabotage arm (`gbdt_host_sabotage`) is
`gbdt/host/gbdt_oracle.mojo::GBDT_ORACLE_HOST_SABOTAGE`: the Newton walker's
Hessian regularizer is one larger, so every leaf of every tree moves; on
the RMSE arm, which runs no walker, the searcher's leaf regularizer
(`gbdt/host/gbdt_oracle_rmse.mojo::_rmse_leaf_value`) is one larger; the
ordered arm's Newton Hessian regularizer and the FeatureFreq leaf value's l2
are one larger.
"""
from std.math import isfinite
from std.memory import bitcast
from std.os import abort
from std.python import Python, PythonObject

#: their Quantile / MAE `delta` default (`optimal_const_for_loss.h:198`),
#: the value `gbdt/metrics/optimal_const_for_loss.mojo` passes
comptime QUANTILE_CONST_DELTA = 1e-6
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from core.hostptr import f32_ptr, f64_ptr, read_f32, u32_ptr
from bindings.py_lists import py_list_f64
# The typed bodies (lane thin-trees-metrics, 2026-10-21): the model text
# reader, the predict walks and links, the binary transforms, and the scans
# and reads of the fit's tails, moved word for word out of this shell.
from gbdt.host.gbdt_host_entries_impl import (
    _HostModelArrays,
    _parse_model,
    _refuse_predict,
    gbdt_host_binary_class_row_weights,
    gbdt_host_binary_margins_check,
    gbdt_host_binary_prediction_into,
    gbdt_host_eval_const,
    gbdt_host_group_sizes,
    gbdt_host_has_nan_f32,
    gbdt_host_has_nan_ptr,
    gbdt_host_load_sources,
    gbdt_host_load_u32,
    gbdt_host_predict_into,
    gbdt_host_predict_multi_into,
    gbdt_host_read_pairs,
    gbdt_host_sigmoid_pair_into,
)
from checks.kernel_matrix import (
    COLUMN_CPU,
    TARGET_COLUMN,
    column_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, ftz, identical_exp64, identical_sigmoid
from checks.soft_f64 import (
    SF64_ONE,
    SF64_ZERO,
    sf64_add,
    sf64_div,
    sf64_exp,
    sf64_from_f32,
    sf64_ftz,
    sf64_gt,
    sf64_neg,
    sf64_sigmoid_f32,
    sf64_sigmoid_f64,
    sf64_sub,
    sf64_to_f32,
)
from core.gbdt_host_predict import GBDT_HOST_SABOTAGE, gbdt_host_predict
from gbdt.data.quantization import (
    NAN_TREATMENT_AS_FALSE,
    NAN_TREATMENT_AS_IS,
    NAN_TREATMENT_AS_TRUE,
)
from gbdt.host.gbdt_oracle_eval import GbdtHostEval, gbdt_eval_want_best_model
from gbdt.host.gbdt_oracle import (
    GBDT_BOOT_BAYESIAN,
    GBDT_BOOT_BERNOULLI,
    GBDT_BOOT_POISSON,
    GBDT_LOGLOSS_NEWTON_ITERATIONS,
    GBDT_ORACLE_HOST_SABOTAGE,
    GbdtHostParams,
    gbdt_host_fit,
    gbdt_host_fit_eval,
    gbdt_host_model_text,
)
from gbdt.host.gbdt_oracle_losses import (
    GBDT_LEAF_EXACT,
    GBDT_LEAF_GRADIENT,
    GBDT_LEAF_NEWTON,
    GBDT_OBJ_CROSSENTROPY,
    GBDT_OBJ_EXPECTILE,
    GBDT_OBJ_HUBER,
    GBDT_OBJ_LOGLINQUANTILE,
    GBDT_OBJ_LOGLOSS,
    GBDT_OBJ_LQ,
    GBDT_OBJ_MAE,
    GBDT_OBJ_MAPE,
    GBDT_OBJ_POISSON,
    GBDT_OBJ_PAIR_LOGIT,
    GBDT_OBJ_QUANTILE,
    GBDT_OBJ_QUERY_RMSE,
    GBDT_OBJ_RMSE,
    GBDT_OBJ_TWEEDIE,
    GBDT_OBJ_YETI_RANK,
    GbdtHostLoss,
    gbdt_losses_host_fit,
)
from gbdt.host.gbdt_oracle_multiclass import (
    GBDT_OBJ_MULTICLASS,
    GBDT_OBJ_MULTICLASS_OVA,
    GBDT_OBJ_MULTIRMSE,
    gbdt_multi_host_fit,
    gbdt_multi_host_model_text,
)
from gbdt.host.gbdt_oracle_rmse import (
    GbdtRmseHostFit,
    _rmse_starting_approx,
    gbdt_rmse_host_fit,
    gbdt_rmse_host_model_text,
    gbdt_text_with_bias,
)
from gbdt.host.gbdt_oracle_depthwise import (
    GBDT_HOST_GROW_LOSSGUIDE,
    GBDT_HOST_SCORE_NEWTON_COSINE,
    GBDT_HOST_SCORE_NEWTON_L2,
    GbdtHostTreeParams,
    gbdt_host_fit_non_symmetric,
    gbdt_host_ns_model_text,
)
from gbdt.options.data_processing_options import nan_mode_from_name
from gbdt.options.overfitting_detector_options import (
    load_overfitting_detector_options,
)
from gbdt.overfitting_detector.overfitting_detector import OD_NONE
from gbdt.grid_creator.binarization import (
    BORDER_TYPE_GREEDY_LOG_SUM,
    border_type_from_name,
)
from gbdt.metrics.sample_quantile import (
    calculate_optimal_const_approx_for_mape,
    calculate_weighted_target_quantile,
)
from gbdt.host.gbdt_oracle_feature_freq import gbdt_feature_freq_host_fit
from gbdt.host.gbdt_oracle_ordered import (
    GbdtOrderedHostEval,
    GbdtOrderedHostOptions,
    gbdt_ordered_host_fit,
    gbdt_ordered_rmse_host_fit,
)
from gbdt.data.ordered_plan import (
    ORDERED_MIN_FOLD_SIZE,
    ordered_permutation_block_size,
)
from gbdt.host.gbdt_oracle_pointwise import gbdt_pointwise_host_fit
from gbdt.host.gbdt_oracle_ctr import (
    gbdt_ctr_column_kinds,
    gbdt_ctr_host_fit,
    gbdt_ctr_host_model_text,
    gbdt_has_ctr_columns,
)
from gbdt.host.gbdt_oracle_onehot import (
    gbdt_host_model_text_one_hot,
    gbdt_resolve_one_hot,
)


#: `SCORE_FUNCTION_COSINE` and `LEAF_ESTIMATION_NEWTON`
#: (`gbdt/options/catboost_options.mojo:130`, `:270`), the codes the wrapper
#: sends in slots 10 and 17.
comptime GBDT_HOST_SCORE_COSINE = 1
comptime GBDT_HOST_LEAF_NEWTON = 1
#: `SCORE_FUNCTION_L2` (`gbdt/options/catboost_options.mojo:135`).
comptime GBDT_HOST_SCORE_L2 = 6
#: `DEFAULT_TARGET_BORDER` (`gbdt/options/loss_description.mojo:72`).
comptime GBDT_HOST_DEFAULT_BORDER = Float32(0.5)
#: The deepest tree the host binding grows; the model text reader refuses
#: past 31 and `1 << depth` leaves are materialized per tree.
comptime GBDT_HOST_MAX_DEPTH = 16


def gbdt_host_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def gbdt_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def gbdt_host_column_binding() raises -> PythonObject:
    """`column_name(TARGET_COLUMN)`, the comptime assert's witness: "cpu".
    THE COLUMN IS THE CPU COLUMN, OR THIS DOES NOT BUILD; the assert lives in
    a function body PyInit registers, so it is compiled in every build.
    `bindings/build_host_family.sh` passes -D MOJOLEARN_COLUMN_CPU."""
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "gbdt host: this binding compiles the CPU column only; pass"
        " -D MOJOLEARN_COLUMN_CPU (bindings/build_gbdt_host.sh does)"
    )
    return PythonObject(column_name(TARGET_COLUMN))


# There is no `gbdt_host_detected_column` read-back, for the reason 8d16ce2f
# removed it from the forest and byte LM host bindings: the detected column
# folds to the GPU of the machine that ran the build. The comptime assert
# above is the check.


def gbdt_host_sabotage_binding() raises -> PythonObject:
    """Whether this binary adds 1.0 to the Newton walker's Hessian
    regularizer and to the RMSE arm's leaf regularizer on purpose
    (-D MOJOLEARN_HOST_SABOTAGE=1, the gate's negative control;
    `gbdt/host/gbdt_oracle.mojo::GBDT_ORACLE_HOST_SABOTAGE`)."""
    return PythonObject(GBDT_ORACLE_HOST_SABOTAGE)


# The GPU binding's names, same contract.


def gbdt_vendor_binding() raises -> PythonObject:
    """"cpu". On a CPU-only install `_backend.vendor()` is "cpu" and the
    read-back cross-check expects that string from every host binding."""
    return PythonObject(String("cpu"))


def gbdt_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def _refuse(what: String) raises:
    """The by-name refusal. The sentence starts with the phrase the CPU
    identity gate's column check keys on, so a lane this binding does not
    cover reads REFUSED and never a hash."""
    raise Error(
        "no CPU implementation of _mojolearn_gbdt.gbdt_fit for " + what
        + "; the gbdt host binding trains the declared GBDT lanes"
        " only (SymmetricTree with Cosine and the Logloss, RMSE, pointwise"
        " and multiclass losses, Depthwise with Logloss and Cosine,"
        " Lossguide with Logloss and NewtonL2 or NewtonCosine; no sample"
        " weights and no CTR categoricals anywhere; an eval set and the"
        " overfitting detector on SymmetricTree with Logloss and on the"
        " Ordered and pointwise-searcher arms only; one-hot categorical"
        " columns under SymmetricTree with Logloss), see"
        " gbdt/host/gbdt_oracle.mojo and the gbdt/host oracles beside it"
    )


def _pointwise_objective(loss: String) -> Int:
    """`objective_from_name` (`gbdt/targets/kernel/pointwise_targets.mojo:
    83-123`) for the losses `gbdt_oracle_losses.mojo` restates; -1 else."""
    if loss == String("CrossEntropy"):
        return GBDT_OBJ_CROSSENTROPY
    if loss == String("Quantile"):
        return GBDT_OBJ_QUANTILE
    if loss == String("MAE"):
        return GBDT_OBJ_MAE
    if loss == String("LogLinQuantile"):
        return GBDT_OBJ_LOGLINQUANTILE
    if loss == String("MAPE"):
        return GBDT_OBJ_MAPE
    if loss == String("Poisson"):
        return GBDT_OBJ_POISSON
    if loss == String("Lq"):
        return GBDT_OBJ_LQ
    if loss == String("Expectile"):
        return GBDT_OBJ_EXPECTILE
    if loss == String("Tweedie"):
        return GBDT_OBJ_TWEEDIE
    if loss == String("Huber"):
        return GBDT_OBJ_HUBER
    if loss == String("QueryRMSE"):
        return GBDT_OBJ_QUERY_RMSE
    if loss == String("PairLogit"):
        return GBDT_OBJ_PAIR_LOGIT
    if loss == String("YetiRank"):
        return GBDT_OBJ_YETI_RANK
    return -1


def _resolve_pointwise_loss(
    objective: Int,
    name: String,
    loss_alpha: Float32,
    loss_q: Float32,
    loss_delta: Float32,
    loss_variance_power: Float32,
    method_override: Int,
    iterations_override: Int,
    bootstrap_type: String,
    subsample: Float32,
    bagging_temperature: Float32 = Float32(1.0),
) raises -> GbdtHostLoss:
    """What `train` resolves for these losses, restated because the option
    modules import a kernel module: `make_loss_description` and `validate`
    (`gbdt/options/loss_description.mojo:110-219`), `kernel_alpha` and
    `get_alpha`, `get_estimation_method_defaults`, `use_exact_leaves`,
    `set_leaves_estimation_default` and `ensure_newton_is_available`
    (`gbdt/options/catboost_options.mojo:1247-1499`), and the bootstrap
    parameter (`gbdt/train.mojo:1660-1690`, `DEFAULT_SUBSAMPLE` 0.66)."""
    var has_alpha = loss_alpha >= Float32(0.0)
    var estimator_alpha = loss_alpha if has_alpha else Float32(0.5)
    if objective == GBDT_OBJ_LQ and not (loss_q >= Float32(0.0)):
        raise Error("Param q is mandatory for Lq loss")
    if objective == GBDT_OBJ_HUBER and not (loss_delta >= Float32(0.0)):
        raise Error("For Huber delta parameter is mandatory")
    if objective == GBDT_OBJ_TWEEDIE and not (loss_variance_power >= Float32(0.0)):
        raise Error("For Tweedie variance_power parameter is mandatory")
    if objective == GBDT_OBJ_EXPECTILE and not has_alpha:
        raise Error("Param alpha is mandatory for expectile loss")
    var kernel_alpha = Float32(0.0)
    if objective == GBDT_OBJ_MAE:
        kernel_alpha = Float32(0.5)
    elif objective == GBDT_OBJ_LQ:
        kernel_alpha = loss_q
    elif objective == GBDT_OBJ_HUBER:
        kernel_alpha = loss_delta
    elif objective == GBDT_OBJ_TWEEDIE:
        kernel_alpha = loss_variance_power
    elif objective == GBDT_OBJ_QUANTILE or objective == GBDT_OBJ_LOGLINQUANTILE or objective == GBDT_OBJ_EXPECTILE:
        kernel_alpha = estimator_alpha
    var method = GBDT_LEAF_NEWTON
    var newton = 1
    var gradient = 1
    if objective == GBDT_OBJ_LQ:
        if loss_q < Float32(2.0):
            method = GBDT_LEAF_GRADIENT
    elif objective == GBDT_OBJ_MAE or objective == GBDT_OBJ_MAPE or objective == GBDT_OBJ_QUANTILE or objective == GBDT_OBJ_LOGLINQUANTILE:
        method = GBDT_LEAF_GRADIENT
    elif objective == GBDT_OBJ_EXPECTILE:
        newton = 5
        gradient = 10
    elif objective == GBDT_OBJ_POISSON:
        newton = 10
    elif objective == GBDT_OBJ_CROSSENTROPY:
        newton = 10
        gradient = 40
    elif objective == GBDT_OBJ_PAIR_LOGIT:
        # `GetEstimationMethodDefaults`' PairLogit case
        # (`catboost_options.cpp:120-125`), as `gbdt/options/catboost_options.mojo`
        newton = 10
        gradient = 40
    elif objective == GBDT_OBJ_TWEEDIE:
        newton = 20
        gradient = 20
    if objective == GBDT_OBJ_MAE or objective == GBDT_OBJ_MAPE or objective == GBDT_OBJ_QUANTILE:
        method = GBDT_LEAF_EXACT
        newton = 1
        gradient = 1
    if method_override >= 0:
        if method_override != method and objective == GBDT_OBJ_YETI_RANK:
            # `catboost_options.cpp:307-308`, their message verbatim
            raise Error(
                "At the moment, in the YetiRank mode, changing the"
                " leaf_estimation_method parameter is prohibited."
            )
        method = method_override
    var iterations = 1
    if method == GBDT_LEAF_NEWTON:
        iterations = newton
    elif method == GBDT_LEAF_GRADIENT:
        iterations = gradient
    if iterations_override >= 0:
        iterations = iterations_override
    if method == GBDT_LEAF_NEWTON:
        if objective == GBDT_OBJ_QUANTILE or objective == GBDT_OBJ_MAE or objective == GBDT_OBJ_LOGLINQUANTILE or objective == GBDT_OBJ_MAPE:
            raise Error("Newton leaves estimation method is not supported for " + name + " loss function")
        if objective == GBDT_OBJ_LQ and loss_q < Float32(2.0):
            raise Error("Newton leaves estimation method is not supported for Lq loss function with q < 2")
    if method == GBDT_LEAF_EXACT and not (
        objective == GBDT_OBJ_MAE or objective == GBDT_OBJ_MAPE or objective == GBDT_OBJ_QUANTILE
    ):
        raise Error("Only MAPE, MAE and Quantile are supported for Exact leaves estimation on GPU")
    var boot_kind = -1
    var boot_param = Float32(0.0)
    if bootstrap_type == String("Bernoulli"):
        boot_kind = GBDT_BOOT_BERNOULLI
        boot_param = subsample if subsample >= Float32(0.0) else Float32(0.66)
    elif bootstrap_type == String("Poisson"):
        boot_kind = GBDT_BOOT_POISSON
        boot_param = subsample if subsample >= Float32(0.0) else Float32(0.66)
    elif bootstrap_type == String("Bayesian"):
        if subsample >= Float32(0.0):
            raise Error(
                "Error: default bootstrap type (bayesian) doesn't support"
                " 'subsample' option"
            )
        boot_kind = GBDT_BOOT_BAYESIAN
        boot_param = bagging_temperature
    return GbdtHostLoss(
        objective, kernel_alpha, estimator_alpha, method, iterations,
        boot_kind, boot_param, Float32(0.5),
    )


def _refuse_pointwise(what: String) raises:
    """The pointwise arm's by-name refusal, with the gate's sentence."""
    raise Error(
        "no CPU implementation of _mojolearn_gbdt.gbdt_fit for " + what
        + " under use_pointwise_searcher=True; the gbdt host binding trains"
        " the gbdt-pointwise-l2-bayesian-eval lane only on this arm"
        " (SymmetricTree, Logloss, L2 scores, the Bayesian bootstrap,"
        " boost_from_average=True, sample_weight, an eval set with the Iter"
        " detector, Newton leaves), see gbdt/host/gbdt_oracle_pointwise.mojo"
    )


def _gbdt_fit_pointwise_arm(
    x_address: Int,
    y_address: Int,
    w_address: Int,
    ex_address: Int,
    ey_address: Int,
    params: PythonObject,
    strs: PythonObject,
) raises -> PythonObject:
    """`gbdt_fit` with `use_pointwise_searcher=True`: the slots were checked
    by the caller; this reads them again, refuses by name every value the
    pointwise oracle does not restate, applies `train`'s own raises on the
    rest, and fits through `gbdt_pointwise_host_fit`."""
    var n_class_weights = Int(py=params[34])
    var fixed_and_weights = 35 + n_class_weights
    var min_split_gain = Float64(-1)
    if len(params) >= fixed_and_weights + 1:
        min_split_gain = Float64(py=params[fixed_and_weights])
    var min_child_hessian = Float64(-1)
    if len(params) >= fixed_and_weights + 2:
        min_child_hessian = Float64(py=params[fixed_and_weights + 1])
    var feature_fraction = Float64(1)
    if len(params) == fixed_and_weights + 3:
        feature_fraction = Float64(py=params[fixed_and_weights + 2])
    var n_rows = Int(py=params[0])
    var n_features = Int(py=params[1])
    var n_weights = Int(py=params[2])
    var n_flags = Int(py=params[3])
    var border_count = Int(py=params[4])
    var n_estimators = Int(py=params[5])
    var max_depth = Int(py=params[6])
    var learning_rate = Float32(Float64(py=params[7]))
    var l2_leaf_reg = Float32(Float64(py=params[8]))
    var random_seed = UInt64(Int(py=params[9]))
    var score_function = Int(py=params[10])
    var loss = String(py=strs[0])
    var loss_border = Float32(Float64(py=params[15]))
    var leaf_iterations = Int(py=params[16])
    var leaf_method = Int(py=params[17])
    var bagging_temperature = Float32(Float64(py=params[18]))
    var subsample = Float32(Float64(py=params[19]))
    var bootstrap_type = String(py=strs[1])
    var n_eval_rows = Int(py=params[20])
    var od_type = String(py=strs[2])
    var od_pvalue = Float64(py=params[21])
    var od_wait = Int(py=params[22])
    var use_best_model = Int(py=params[23])
    var best_model_min_trees = Int(py=params[24])
    var nan_mode_name = String(py=strs[3])
    var random_strength = Float32(Float64(py=params[25]))
    var border_build_max_samples = Int(py=params[27])
    var boost_from_average = Int(py=params[30])
    var grow_code = Int(py=params[31])
    var max_leaves = Int(py=params[32])
    var min_data_in_leaf = Int(py=params[33])

    if loss != String("Logloss"):
        _refuse_pointwise("loss='" + loss + "'")
    if grow_code != 0:
        _refuse_pointwise("grow_policy code " + String(grow_code))
    if score_function != GBDT_HOST_SCORE_L2:
        _refuse_pointwise("score_function code " + String(score_function) + " (only L2)")
    if leaf_method != -1 and leaf_method != GBDT_HOST_LEAF_NEWTON:
        _refuse_pointwise("leaf_estimation_method code " + String(leaf_method) + " (only Newton)")
    if bootstrap_type != String("Bayesian"):
        _refuse_pointwise("bootstrap_type='" + bootstrap_type + "' (only Bayesian)")
    if n_weights == 0:
        _refuse_pointwise("a fit without sample_weight")
    if n_class_weights != 0:
        _refuse_pointwise("class_weights")
    if n_flags != 0:
        _refuse_pointwise("cat_features or one_hot_features")
    if n_eval_rows == 0:
        _refuse_pointwise("a fit without eval_set")
    if od_type != String("Iter") or od_pvalue >= 0.0 or od_wait < 0:
        _refuse_pointwise("the overfitting detector other than od_type='Iter' with od_wait")
    if random_strength != Float32(0.0):
        _refuse_pointwise("random_strength=" + String(random_strength))
    if boost_from_average != 1:
        _refuse_pointwise("boost_from_average other than True")
    if feature_fraction != 1.0:
        _refuse_pointwise("feature_fraction=" + String(feature_fraction))
    if border_count < 1 or border_count > 255:
        _refuse_pointwise("border_count=" + String(border_count) + " (1 to 255)")
    if max_depth < 1 or max_depth > GBDT_HOST_MAX_DEPTH:
        _refuse_pointwise("max_depth=" + String(max_depth) + " (1 to 16)")
    # `train`'s own raises (`gbdt/train.mojo:944-1760`)
    if min_split_gain >= 0:
        raise Error("min_split_gain requires Depthwise or Lossguide")
    if min_child_hessian >= 0:
        raise Error("min_child_hessian requires Depthwise or Lossguide")
    if min_data_in_leaf != 1:
        raise Error(
            "min_data_in_leaf=" + String(min_data_in_leaf) + " does nothing"
            " under grow_policy=SymmetricTree"
        )
    if max_leaves >= 0 and max_leaves != (1 << max_depth):
        raise Error("max_leaves option works only with lossguide tree growing (catboost_options.cpp:998)")
    if subsample >= Float32(0.0):
        raise Error(
            "Error: default bootstrap type (bayesian) doesn't support"
            " 'subsample' option"
        )
    if use_best_model != -1 and use_best_model != 0 and use_best_model != 1:
        raise Error("use_best_model must be -1 (unset), 0 or 1, got " + String(use_best_model))
    if best_model_min_trees < 1:
        raise Error("best_model_min_trees must be at least 1, got " + String(best_model_min_trees))
    var nan_mode = nan_mode_from_name(nan_mode_name)
    var iterations = GBDT_LOGLOSS_NEWTON_ITERATIONS
    if leaf_iterations >= 0:
        iterations = leaf_iterations
    var border = GBDT_HOST_DEFAULT_BORDER
    if loss_border >= Float32(0.0):
        border = loss_border

    var has_nan = False
    var text = String("")
    var losses = List[Float64]()
    var test_losses = List[Float64]()
    var best_iteration = 0
    var stopped_early = False
    with GILReleased(Python()):
        var x = read_f32(x_address, n_rows * n_features)
        var ex = read_f32(ex_address, n_eval_rows * n_features)
        has_nan = gbdt_host_has_nan_f32(x) or gbdt_host_has_nan_f32(ex)
        if not has_nan:
            var y = read_f32(y_address, n_rows)
            var w = read_f32(w_address, n_rows)
            var ey = read_f32(ey_address, n_eval_rows)
            var fit = gbdt_pointwise_host_fit(
                x, y, w, n_rows, n_features, ex, ey, n_eval_rows,
                border_count, border_build_max_samples, n_estimators,
                max_depth, learning_rate, l2_leaf_reg, random_seed, nan_mode,
                border, iterations, bagging_temperature, od_wait,
                use_best_model, best_model_min_trees,
            )
            text = fit.text
            losses = fit.losses.copy()
            test_losses = fit.test_losses.copy()
            best_iteration = fit.best_iteration
            stopped_early = fit.stopped_early
    if has_nan:
        _refuse_pointwise("an X carrying NaN")
    var learn = py_list_f64(losses)
    var test = py_list_f64(test_losses)
    var out = Python.list()
    out.append(PythonObject(text))
    out.append(PythonObject(best_iteration))
    out.append(PythonObject(stopped_early))
    out.append(learn)
    out.append(test)
    return out


def _refuse_ordered_host(what: String) raises:
    raise Error(
        "no CPU implementation of _mojolearn_gbdt.gbdt_fit for " + what
        + " with boosting_type='Ordered'; the gbdt host binding trains Ordered"
        " boosting at unit weights on numeric columns with RMSE, Logloss,"
        " CrossEntropy and the pointwise losses (see"
        " gbdt/host/gbdt_oracle_ordered.mojo::gbdt_ordered_host_fit)"
    )


def _gbdt_fit_ordered_arm(
    x_address: Int,
    y_address: Int,
    flags_address: Int,
    eval_x_address: Int,
    eval_y_address: Int,
    params: PythonObject,
    strs: PythonObject,
    border_type: Int,
) raises -> PythonObject:
    """`gbdt_fit` with `boosting_type='Ordered'`: `train`'s Ordered refusals
    in its words, the host binding's own (what the oracle does not restate),
    the loss resolved as `train` resolves it under Ordered (their GPU
    `useExact` needs Plain, `catboost_options.cpp:290-293`, so MAE, MAPE and
    Quantile keep Gradient), then `gbdt_ordered_host_fit`."""
    var n_class_weights = Int(py=params[34])
    var fixed_and_weights = 35 + n_class_weights
    var n_rows = Int(py=params[0])
    var n_features = Int(py=params[1])
    var n_weights = Int(py=params[2])
    var n_flags = Int(py=params[3])
    var border_count = Int(py=params[4])
    var n_estimators = Int(py=params[5])
    var max_depth = Int(py=params[6])
    var learning_rate = Float32(Float64(py=params[7]))
    var l2_leaf_reg = Float32(Float64(py=params[8]))
    var random_seed = UInt64(Int(py=params[9]))
    var score_function = Int(py=params[10])
    var loss_border = Float32(Float64(py=params[15]))
    var leaf_iterations = Int(py=params[16])
    var leaf_method = Int(py=params[17])
    var bagging_temperature = Float32(Float64(py=params[18]))
    var subsample = Float32(Float64(py=params[19]))
    var n_eval_rows = Int(py=params[20])
    var od_pvalue = Float64(py=params[21])
    var od_wait = Int(py=params[22])
    var random_strength = Float32(Float64(py=params[25]))
    var use_pointwise = Int(py=params[26]) != 0
    var border_build_max_samples = Int(py=params[27])
    var permutation_count = Int(py=params[28])
    var boost_from_average = Int(py=params[30])
    var grow_code = Int(py=params[31])
    var feature_fraction = Float64(1)
    if len(params) >= fixed_and_weights + 3:
        feature_fraction = Float64(py=params[fixed_and_weights + 2])
    var loss = String(py=strs[0])
    var bootstrap_type = String(py=strs[1])
    var od_type = String(py=strs[2])
    var nan_mode = nan_mode_from_name(String(py=strs[3]))
    var fold_len_multiplier = bitcast[DType.float64](UInt64(Int(String(py=strs[6]))))
    var fold_permutation_block = Int(String(py=strs[7]))

    # ---- `train`'s Ordered refusals (`gbdt/train.mojo`), their words ----
    if grow_code != 0:
        raise Error(
            "Ordered boosting is not supported for nonsymmetric trees."
            " (catboost_options.cpp:757-759)"
        )
    if loss == String("MultiRMSE"):
        # `train`'s words (`gbdt/train.mojo`)
        raise Error(
            "boosting_type='Ordered' with loss='MultiRMSE' is not carried"
            " here: the Ordered arm (gbdt/methods/ordered_boosting.mojo)"
            " is one-dimensional"
        )
    if loss == String("MultiClass") or loss == String("MultiClassOneVsAll"):
        raise Error(
            "On GPU loss " + loss + " can't be used with ordered boosting"
            " (catboost_options.cpp:949-967: their GPU trains this loss"
            " doc-parallel and Plain only)"
        )
    if score_function != GBDT_HOST_SCORE_COSINE and score_function != GBDT_HOST_SCORE_NEWTON_COSINE:
        raise Error(
            "Score function can't be used with ordered boosting"
            " (catboost_options.cpp:972-978); Cosine and NewtonCosine are"
            " the two with an ordered kernel"
        )
    if loss == String("QueryRMSE") or loss == String("PairLogit") or loss == String("YetiRank"):
        raise Error(
            "boosting_type='Ordered' with loss='" + loss + "' is not"
            " implemented here: the reference's folds follow the query"
            " grouping (dynamic_boosting.h:189-223), which this Ordered arm"
            " does not restate"
        )
    if use_pointwise:
        raise Error(
            "use_pointwise_searcher selects the doc-parallel Plain searcher;"
            " an Ordered fit always runs the feature-parallel fold searcher"
        )
    if feature_fraction != 1.0:
        raise Error(
            "boosting_type='Ordered' with feature_fraction < 1 is not"
            " implemented here"
        )
    if leaf_method == GBDT_LEAF_EXACT:
        raise Error(
            "Exact leaf estimation method don't work with ordered boosting"
            " on GPU (catboost_options.cpp:346-350)"
        )
    # ---- what this oracle does not restate, by name ----
    if n_weights != 0:
        _refuse_ordered_host("sample_weight")
    if n_class_weights != 0:
        _refuse_ordered_host("class_weights")
    if n_flags != 0:
        _refuse_ordered_host("cat_features or one_hot_features")
    # the detector as `gbdt_fit` resolves it, through the same host function
    # (`load_overfitting_detector_options`, their `Load`)
    var use_best_model = Int(py=params[23])
    var best_model_min_trees = Int(py=params[24])
    var od = load_overfitting_detector_options(od_type, od_pvalue, od_wait)
    var od_kind = od.od_type
    if use_best_model == 1 and n_eval_rows == 0:
        raise Error(
            "use_best_model=1 needs an eval set: pass eval_x_colmajor and"
            " eval_y, or leave it unset."
        )
    if best_model_min_trees < 1:
        raise Error(
            "best_model_min_trees must be at least 1, got "
            + String(best_model_min_trees)
        )
    if od_kind != OD_NONE and n_eval_rows == 0:
        raise Error(
            "od_type is set but there is no held-out set. Stopping on the"
            " LEARN loss would stop on a curve that falls almost by"
            " construction; their own detector is inert without a test set"
            " (overfitting_detector.cpp:122-124) and this refuses rather"
            " than silently never firing."
        )
    var ordered_bfa_losses = (
        loss == String("RMSE") or loss == String("MAE")
        or loss == String("Quantile") or loss == String("MAPE")
    )
    if boost_from_average == 1 and not ordered_bfa_losses:
        _refuse_ordered_host(
            "boost_from_average=True outside RMSE, MAE, Quantile and MAPE"
        )
    if border_count < 1 or border_count > 255:
        _refuse_ordered_host("border_count=" + String(border_count) + " (1 to 255)")

    # ---- the loss, as `train` resolves it under Ordered ----
    var hloss: GbdtHostLoss
    var border = GBDT_HOST_DEFAULT_BORDER
    if loss_border >= Float32(0.0):
        border = loss_border
    if loss == String("RMSE") or loss == String("Logloss"):
        var objective = GBDT_OBJ_RMSE if loss == String("RMSE") else GBDT_OBJ_LOGLOSS
        var method = GBDT_LEAF_NEWTON if leaf_method == -1 else leaf_method
        var iterations: Int
        if method == GBDT_LEAF_NEWTON:
            iterations = 1 if objective == GBDT_OBJ_RMSE else GBDT_LOGLOSS_NEWTON_ITERATIONS
        elif method == GBDT_LEAF_GRADIENT:
            iterations = 1 if objective == GBDT_OBJ_RMSE else 40
        else:
            _refuse_ordered_host("leaf_estimation_method code " + String(method))
            iterations = 1
        if leaf_iterations >= 0:
            iterations = leaf_iterations
        # `kernel_alpha` carries Logloss's border, as `_loss_row` reads it
        hloss = GbdtHostLoss(
            objective, border if objective == GBDT_OBJ_LOGLOSS else Float32(0.0),
            Float32(0.5), method, iterations, -1, Float32(0.0), border,
        )
    else:
        var pw_objective = _pointwise_objective(loss)
        if pw_objective < 0:
            _refuse_ordered_host("loss='" + loss + "'")
        hloss = _resolve_pointwise_loss(
            pw_objective, loss,
            Float32(Float64(py=params[11])), Float32(Float64(py=params[12])),
            Float32(Float64(py=params[13])), Float32(Float64(py=params[14])),
            leaf_method, leaf_iterations, String(""), Float32(-1.0),
        )
        if hloss.method == GBDT_LEAF_EXACT:
            # the loss's own default under Ordered: Gradient, one iteration
            # (`GetEstimationMethodDefaults`, `catboost_options.cpp:113-124`)
            hloss.method = GBDT_LEAF_GRADIENT
            hloss.iterations = leaf_iterations if leaf_iterations >= 0 else 1
        if pw_objective == GBDT_OBJ_CROSSENTROPY:
            hloss.kernel_alpha = border

    # ---- the bootstrap (`train`'s resolution, `gbdt/train.mojo`) ----
    var boot_kind = -1
    var boot_param = Float32(0.0)
    if bootstrap_type == String("Bayesian"):
        if subsample >= Float32(0.0):
            raise Error(
                "Error: default bootstrap type (bayesian) doesn't support"
                " 'subsample' option"
            )
        boot_kind = 0
        boot_param = bagging_temperature
    elif bootstrap_type == String("Bernoulli"):
        boot_kind = 1
        boot_param = subsample if subsample >= Float32(0.0) else Float32(0.66)
    elif bootstrap_type == String("Poisson"):
        boot_kind = 2
        boot_param = subsample if subsample >= Float32(0.0) else Float32(0.66)
    elif bootstrap_type != String("") and bootstrap_type != String("No"):
        raise Error(
            "unknown bootstrap_type '" + bootstrap_type
            + "': Bayesian, Bernoulli, Poisson, No"
        )

    var perm_count = permutation_count if permutation_count != -1 else 4
    if perm_count < 1:
        raise Error("Permutation count should be positive (boosting_options.cpp:67)")
    # `AdjustBoostFromAverageDefaultValue`: unset is True for RMSE, MAE,
    # Quantile and MAPE (`gbdt/train.mojo`)
    var bfa = boost_from_average == 1 or (
        boost_from_average == -1 and ordered_bfa_losses
    )
    var p = GbdtHostParams(
        border_count, border_build_max_samples, n_estimators, max_depth,
        learning_rate, l2_leaf_reg, random_seed, nan_mode, border,
        hloss.iterations, border_type,
    )
    var opts = GbdtOrderedHostOptions(
        score_function == GBDT_HOST_SCORE_NEWTON_COSINE, random_strength,
        boot_kind, boot_param, perm_count, fold_len_multiplier,
        ordered_permutation_block_size(n_rows, fold_permutation_block),
        ORDERED_MIN_FOLD_SIZE, bfa,
    )
    var text = String("")
    var losses = List[Float64]()
    var test_losses = List[Float64]()
    var best = 0
    var stopped = False
    with GILReleased(Python()):
        var x = read_f32(x_address, n_rows * n_features)
        var y = read_f32(y_address, n_rows)
        var ex = List[Float32]()
        var ey = List[Float32]()
        if n_eval_rows > 0:
            ex = read_f32(eval_x_address, n_eval_rows * n_features)
            ey = read_f32(eval_y_address, n_eval_rows)
        # `UpdateUseBestModel` (`options_helper.cpp:100-113`), as `train`
        var eval_const = gbdt_host_eval_const(ey, n_eval_rows)
        var want_best = use_best_model
        if want_best == -1:
            want_best = 1 if (n_eval_rows > 0 and not eval_const) else 0
        var ev = GbdtOrderedHostEval(
            ex^, ey^, n_eval_rows, od_kind, od.auto_stop_p_value,
            od.iterations_wait, want_best, best_model_min_trees,
        )
        var fit = gbdt_ordered_host_fit(x, y, n_rows, n_features, p, hloss, opts, ev)
        text = fit.text
        losses = fit.losses.copy()
        test_losses = fit.test_losses.copy()
        best = fit.best_iteration
        stopped = fit.stopped_early
    _ = flags_address
    var learn = py_list_f64(losses)
    var test = py_list_f64(test_losses)
    var out = Python.list()
    out.append(PythonObject(text))
    out.append(PythonObject(best))
    out.append(PythonObject(stopped))
    out.append(learn)
    out.append(test)
    return out


def _gbdt_host_eval_arm(
    eval_x_address: Int,
    eval_y_address: Int,
    n_eval_rows: Int,
    n_features: Int,
    od_kind: Int,
    od_pvalue: Float64,
    od_wait: Int,
    use_best_model: Int,
    best_model_min_trees: Int,
) raises -> GbdtHostEval:
    """The held-out pool `gbdt_host_fit_eval` reads, with `use_best_model`
    resolved by their `UpdateUseBestModel` rule from the held-out target
    (`options_helper.cpp:100-113`), exactly as the Ordered arm resolves it.
    CALL IT WITH THE GIL RELEASED: it reads the two addresses directly."""
    if n_eval_rows <= 0:
        return GbdtHostEval.none()
    var ex = read_f32(eval_x_address, n_eval_rows * n_features)
    var ey = read_f32(eval_y_address, n_eval_rows)
    var want_best = gbdt_eval_want_best_model(use_best_model, ey, n_eval_rows)
    return GbdtHostEval(
        ex^, ey^, n_eval_rows, od_kind, od_pvalue, od_wait, want_best,
        best_model_min_trees,
    )


def gbdt_fit_binding(
    x_addr: PythonObject,
    y_addr: PythonObject,
    weights_addr: PythonObject,
    cat_flags_addr: PythonObject,
    eval_x_addr: PythonObject,
    eval_y_addr: PythonObject,
    params: PythonObject,
    strs: PythonObject,
) raises -> PythonObject:
    """`gbdt_fit_binding` (`bindings/_mojolearn_gbdt.mojo:169-407`): the same
    slot checks in the same words, then the refusals of what the host fit
    does not restate, then the host fit and the model text. Returns
    `[model_text, best_iteration, stopped_early, learn_losses,
    test_losses]`."""
    if len(params) < 35:
        raise Error(
            "gbdt_fit: params must hold at least 35 values, got "
            + String(len(params))
        )
    var n_class_weights = Int(py=params[34])
    if n_class_weights < 0:
        raise Error(
            "gbdt_fit: n_class_weights must not be negative, got "
            + String(n_class_weights)
        )
    var fixed_and_weights = 35 + n_class_weights
    if len(params) != fixed_and_weights and len(params) != fixed_and_weights + 1 and len(params) != fixed_and_weights + 2 and len(params) != fixed_and_weights + 3 and len(params) != fixed_and_weights + 4 and len(params) != fixed_and_weights + 5 and len(params) != fixed_and_weights + 8:
        raise Error(
            "gbdt_fit: params must hold 35 + n_class_weights, optionally min_split_gain, min_child_hessian, then feature_fraction, then the group sizes address and group count, then the pairs address, pair count and pair weights address values ("
            + String(35 + n_class_weights)
            + ") values, got "
            + String(len(params))
        )
    if len(strs) != 4 and len(strs) != 5 and len(strs) != 8:
        raise Error(
            "gbdt_fit: strs must hold [loss, bootstrap_type, od_type,"
            " nan_mode], optionally feature_border_type, optionally then"
            " boosting_type, the fold_len_multiplier's float64 bits and"
            " fold_permutation_block, got " + String(len(strs))
        )
    # `feature_border_type`, the optional fifth string (lane/catboost-parity);
    # the seven types are the device fit's own host functions
    # (`calc_quantization` / `select_borders`), restated nowhere
    var border_type = BORDER_TYPE_GREEDY_LOG_SUM
    if len(strs) >= 5:
        border_type = border_type_from_name(String(py=strs[4]))
    # the Ordered tail (lane/catboost-parity): its own arm, before any of
    # the Plain arms' refusals
    if len(strs) == 8 and String(py=strs[5]) == String("Ordered"):
        return _gbdt_fit_ordered_arm(
            Int(py=x_addr), Int(py=y_addr), Int(py=cat_flags_addr),
            Int(py=eval_x_addr), Int(py=eval_y_addr), params, strs,
            border_type,
        )
    if len(strs) == 8 and String(py=strs[5]) != String("Plain"):
        raise Error(
            "gbdt_fit: boosting_type must be 'Plain' or 'Ordered', got '"
            + String(py=strs[5]) + "'"
        )
    var xp = f32_ptr(Int(py=x_addr))
    var yp = f32_ptr(Int(py=y_addr))
    _ = f32_ptr(Int(py=weights_addr))
    var cp = u32_ptr(Int(py=cat_flags_addr))
    _ = f32_ptr(Int(py=eval_x_addr))
    _ = f32_ptr(Int(py=eval_y_addr))

    var n_rows = Int(py=params[0])
    var n_features = Int(py=params[1])
    var n_weights = Int(py=params[2])
    var n_flags = Int(py=params[3])
    var class_weights = List[Float32]()
    for i in range(n_class_weights):  # small-loop(class weight words of params)
        class_weights.append(Float32(Float64(py=params[35 + i])))
    var grow_code = Int(py=params[31])
    if grow_code != 0 and grow_code != 1 and grow_code != 2:
        raise Error(
            "gbdt_fit: grow_policy code must be 0 (SymmetricTree), 1"
            " (Depthwise) or 2 (Lossguide), got " + String(grow_code)
        )
    var min_split_gain = Float64(-1)
    if len(params) >= fixed_and_weights + 1:
        min_split_gain = Float64(py=params[fixed_and_weights])
    var min_child_hessian = Float64(-1)
    if len(params) >= fixed_and_weights + 2:
        min_child_hessian = Float64(py=params[fixed_and_weights + 1])
    var feature_fraction = Float64(1)
    if len(params) >= fixed_and_weights + 3:
        feature_fraction = Float64(py=params[fixed_and_weights + 2])
    # MultiRMSE's target dimension, the one value of the +4 tail
    # (`bindings/_mojolearn_gbdt.mojo`'s), in its words
    var target_dim = 1
    if len(params) == fixed_and_weights + 4:
        target_dim = Int(py=params[fixed_and_weights + 3])
        if target_dim < 2:
            raise Error(
                "gbdt_fit: the target-dimension tail needs a dimension >= 2,"
                " got " + String(target_dim)
            )
    # the pool's grouping (`bindings/_mojolearn_gbdt.mojo`'s group tail),
    # checked, refused and resolved in the words and by the rule
    # `gbdt/train.mojo::train` uses, so the CPU column refuses exactly where
    # the device column does and QueryRMSE sees the same queries
    var tail_loss = String(py=strs[0])
    var tail_rows = Int(py=params[0])
    var host_group_sizes = List[Int]()
    if len(params) >= fixed_and_weights + 5:
        var n_groups = Int(py=params[fixed_and_weights + 4])
        if n_groups < 1:
            raise Error(
                "gbdt_fit: the group tail needs a positive group count, got "
                + String(n_groups)
            )
        var gp = u32_ptr(Int(py=params[fixed_and_weights + 3]))
        host_group_sizes = gbdt_host_group_sizes(gp, n_groups, tail_rows)
        if tail_loss != String("QueryRMSE") and tail_loss != String("PairLogit") and tail_loss != String("YetiRank"):
            raise Error(
                "group_id is read only by the querywise and pairwise losses"
                " (QueryRMSE, PairLogit and YetiRank are trained here;"
                " QuerySoftMax and QueryCrossEntropy are not implemented);"
                " loss='" + tail_loss + "' does not use it, so it is refused by"
                " name rather than carried and ignored"
            )
    # the PairLogit pairs tail, read with the GIL held; a count of -1 means
    # generate them inside the fit, as `gbdt/train.mojo::train` does
    var host_pair_winners = List[UInt32]()
    var host_pair_losers = List[UInt32]()
    var host_pair_weights = List[Float32]()
    if len(params) == fixed_and_weights + 8:
        var n_pairs = Int(py=params[fixed_and_weights + 6])
        if n_pairs != -1 and n_pairs < 1:
            raise Error(
                "gbdt_fit: the pairs tail needs a positive pair count or -1,"
                " got " + String(n_pairs)
            )
        if n_pairs > 0:
            var pp = u32_ptr(Int(py=params[fixed_and_weights + 5]))
            var pw = f32_ptr(Int(py=params[fixed_and_weights + 7]))
            gbdt_host_read_pairs(
                pp, pw, n_pairs, host_pair_winners, host_pair_losers, host_pair_weights
            )
    if len(host_pair_winners) > 0 and tail_loss != String("PairLogit"):
        raise Error(
            "pairs are read only by loss='PairLogit' here; loss='" + tail_loss
            + "' does not use them"
        )
    if tail_loss == String("PairLogit") and len(host_group_sizes) == 0:
        if len(host_pair_winners) > 0:
            raise Error(
                "pairs without group_id are not implemented here: the"
                " reference regroups and reorders the whole pool by the"
                " pairs' connected components (data_providers.cpp:857-872,"
                " 922-942)"
            )
        raise Error("Cannot generate pairs for data without groups")
    if tail_loss == String("QueryRMSE") or tail_loss == String("PairLogit") or tail_loss == String("YetiRank"):
        # `TDocParallelSplit` (`gpu_data/doc_parallel_dataset.h:26-38`): the
        # pool's queries only with fewer groups than rows, otherwise every
        # row a query of one (`TWithoutQueriesGrouping`)
        if not (len(host_group_sizes) > 0 and len(host_group_sizes) < tail_rows):
            host_group_sizes = List[Int](length=tail_rows, fill=1)

    var border_count = Int(py=params[4])
    var n_estimators = Int(py=params[5])
    var max_depth = Int(py=params[6])
    var learning_rate = Float32(Float64(py=params[7]))
    var l2_leaf_reg = Float32(Float64(py=params[8]))
    var random_seed = UInt64(Int(py=params[9]))
    var score_function = Int(py=params[10])
    var loss = String(py=strs[0])
    if loss == String("YetiRank") and l2_leaf_reg == Float32(0.0):
        # `catboost_options.cpp:357-359`, as `gbdt/train.mojo::train` applies it
        l2_leaf_reg = Float32(1e-20)
    var loss_border = Float32(Float64(py=params[15]))
    var leaf_iterations = Int(py=params[16])
    var leaf_method = Int(py=params[17])
    var bootstrap_type = String(py=strs[1])
    var n_eval_rows = Int(py=params[20])
    var od_type = String(py=strs[2])
    var od_pvalue = Float64(py=params[21])
    var od_wait = Int(py=params[22])
    var use_best_model = Int(py=params[23])
    var best_model_min_trees = Int(py=params[24])
    var nan_mode_name = String(py=strs[3])
    var random_strength = Float32(Float64(py=params[25]))
    var use_pointwise = Int(py=params[26]) != 0
    var border_build_max_samples = Int(py=params[27])
    var permutation_count = Int(py=params[28])
    var ctr_permutation_id = Int(py=params[29])
    var boost_from_average = Int(py=params[30])
    var max_leaves = Int(py=params[32])
    var min_data_in_leaf = Int(py=params[33])

    # `gbdt_fit`'s own checks (`gbdt/estimator.mojo:446-464`)
    if n_rows <= 0:
        raise Error("gbdt_fit: n_rows must be positive")
    if n_features <= 0:
        raise Error("gbdt_fit: n_features must be positive")
    if n_weights != 0 and n_weights != n_rows:
        raise Error(
            "gbdt_fit: n_weights must be 0 or n_rows, got " + String(n_weights)
        )
    if n_flags != 0 and n_flags != n_features:
        raise Error(
            "gbdt_fit: n_flags must be 0 or n_features, got " + String(n_flags)
        )
    if n_eval_rows < 0:
        raise Error(
            "gbdt_fit: n_eval_rows must not be negative, got "
            + String(n_eval_rows)
        )

    # ---- what the host fit does not restate, refused by name ----
    var is_rmse = loss == String("RMSE")
    # the pointwise losses of gbdt/host/gbdt_oracle_losses.mojo
    var pw_objective = _pointwise_objective(loss)
    var is_pointwise = pw_objective >= 0
    # the multi-output losses of gbdt/host/gbdt_oracle_multiclass.mojo
    # MultiRMSE rides the same oracle (lane/algos-trees, 2026-09-27)
    var is_multi_rmse = loss == String("MultiRMSE")
    var is_multi = (
        loss == String("MultiClass") or loss == String("MultiClassOneVsAll")
        or is_multi_rmse
    )
    if target_dim != 1 and not is_multi_rmse:
        raise Error(
            "a multi-dimensional target (target_dim=" + String(target_dim)
            + ") is read only by loss='MultiRMSE'; loss='" + loss
            + "' takes one target per row"
        )
    if is_multi_rmse and target_dim < 2:
        # `train`'s words (`multiclass_targets.h:167`)
        raise Error(
            "Only one class found, can't learn multiclass objective"
            " (MultiRMSE needs a target dimension >= 2,"
            " multiclass_targets.h:167); got target_dim="
            + String(target_dim)
        )
    if loss != String("Logloss") and not is_rmse and not is_pointwise and not is_multi:
        _refuse("loss='" + loss + "'")
    if is_multi and grow_code != 0:
        raise Error(
            "Error: optimization scheme is not supported for GPU learning"
            " Loss=MultiClass;OptimizationScheme="
            + String("Depthwise" if grow_code == 1 else "Lossguide")
            + " (their TGpuTrainerFactory has no non-symmetric"
            " multiclass trainer, multiclass.cpp:5-14, train.cpp:279)"
        )
    if is_multi and leaf_method != -1 and leaf_method != GBDT_HOST_LEAF_NEWTON:
        _refuse("leaf_estimation_method code " + String(leaf_method) + " under loss='" + loss + "' (only Newton)")
    if is_multi and leaf_iterations >= 0 and leaf_iterations != 1:
        _refuse("leaf_estimation_iterations=" + String(leaf_iterations) + " under loss='" + loss + "' (only 1)")
    if is_pointwise and grow_code != 0:
        _refuse("loss='" + loss + "' under grow_policy code " + String(grow_code) + " (Depthwise or Lossguide)")
    if is_rmse and leaf_iterations >= 0 and leaf_iterations != 1:
        _refuse(
            "leaf_estimation_iterations=" + String(leaf_iterations)
            + " under loss='RMSE' (only 1, the searcher's own leaves of"
            " DEVIATION 64; the RMSE Newton walker is not restated)"
        )
    if use_pointwise and border_type != BORDER_TYPE_GREEDY_LOG_SUM:
        _refuse("feature_border_type under use_pointwise_searcher")
    if use_pointwise:
        # the gbdt-pointwise-l2-bayesian-eval lane, its own arm and refusals
        # (`_gbdt_fit_pointwise_arm`, gbdt/host/gbdt_oracle_pointwise.mojo)
        return _gbdt_fit_pointwise_arm(
            Int(py=x_addr), Int(py=y_addr), Int(py=weights_addr),
            Int(py=eval_x_addr), Int(py=eval_y_addr), params, strs,
        )
    # the score function each covered lane runs: Cosine under SymmetricTree
    # and Depthwise, NewtonL2 under Lossguide (the policy defaults)
    # and NewtonCosine under Lossguide (gbdt-lossguide-newtoncosine), where
    # the searcher knobs of that lane are restated as well
    var lossguide_knobs = grow_code == GBDT_HOST_GROW_LOSSGUIDE and loss == String("Logloss")
    # the symmetric Logloss, RMSE and multiclass fits' stochastic arm, CatBoost's GPU
    # defaults (Bayesian bootstrap, random_strength 1):
    # gbdt_oracle.mojo::gbdt_host_fit and gbdt_oracle_rmse.mojo::
    # gbdt_rmse_host_fit; numeric columns only, as the one-hot arm is
    # measured without it
    var symmetric_stochastic = (
        grow_code == 0 and (loss == String("Logloss") or is_rmse or is_multi)
        and not is_pointwise and n_flags == 0
    )
    # the non-symmetric driver's stochastic arm, Logloss and RMSE:
    # gbdt_oracle_depthwise.mojo::gbdt_host_fit_non_symmetric reads the
    # bootstrap kind and the strength as options under both policies
    var ns_stochastic = grow_code != 0 and (loss == String("Logloss") or is_rmse)
    # The shared loss oracle applies row bootstrap AFTER full grouped
    # gradients, matching querywise_targets_impl.h::StochasticDer (162-188).
    # Its score-noise implementation is shared by all these objectives.
    var pw_stochastic = is_pointwise and grow_code == 0
    if grow_code == GBDT_HOST_GROW_LOSSGUIDE:
        if score_function != GBDT_HOST_SCORE_NEWTON_L2 and score_function != GBDT_HOST_SCORE_NEWTON_COSINE:
            _refuse("score_function code " + String(score_function) + " under Lossguide (only NewtonL2 and NewtonCosine)")
    elif score_function != GBDT_HOST_SCORE_COSINE:
        _refuse("score_function code " + String(score_function) + " (only Cosine)")
    if grow_code == 1:
        if min_split_gain >= 0:
            _refuse("min_split_gain=" + String(min_split_gain) + " under Depthwise")
        if min_child_hessian >= 0:
            _refuse("min_child_hessian=" + String(min_child_hessian) + " under Depthwise")
        if min_data_in_leaf != 1:
            _refuse("min_data_in_leaf=" + String(min_data_in_leaf) + " under Depthwise")
    var leaf_gradient_ok = lossguide_knobs and leaf_method == GBDT_LEAF_GRADIENT
    if not is_pointwise and leaf_method != -1 and leaf_method != GBDT_HOST_LEAF_NEWTON and not leaf_gradient_ok:
        _refuse("leaf_estimation_method code " + String(leaf_method) + " (only Newton, or Gradient under Lossguide)")
    if is_pointwise and leaf_method != -1 and leaf_method != GBDT_LEAF_GRADIENT and leaf_method != GBDT_LEAF_NEWTON and leaf_method != GBDT_LEAF_EXACT:
        _refuse("leaf_estimation_method code " + String(leaf_method) + " (Gradient, Newton or Exact)")
    if bootstrap_type != String("") and bootstrap_type != String("No"):
        var pw_boot = is_pointwise and (
            bootstrap_type == String("Poisson") or bootstrap_type == String("Bernoulli")
        ) or pw_stochastic and bootstrap_type == String("Bayesian")
        var lg_boot = ns_stochastic and (
            bootstrap_type == String("Bayesian")
            or bootstrap_type == String("Bernoulli")
        )
        var sym_boot = symmetric_stochastic and (
            bootstrap_type == String("Bayesian")
            or bootstrap_type == String("Bernoulli")
            or bootstrap_type == String("Poisson")
        )
        if not pw_boot and not lg_boot and not sym_boot:
            _refuse("bootstrap_type='" + bootstrap_type + "' under loss='" + loss + "'")
    if n_weights != 0:
        _refuse("sample_weight")
    # binary class weights (lane/fix-cpu-column2): Logloss on a plain
    # SymmetricTree, Depthwise or Lossguide fit, `train`'s weight column
    # `class_weights[y > 0.5]` threaded through the search planes, the leaf
    # estimation and the learn loss (gbdt/host/gbdt_oracle.mojo)
    var binary_class_weights = (
        n_class_weights != 0 and loss == String("Logloss") and n_flags == 0
    )
    if n_class_weights != 0 and (not is_multi or is_multi_rmse) and not binary_class_weights:
        _refuse(
            "class_weights outside MultiClass, MultiClassOneVsAll and Logloss"
            " without categorical columns"
        )
    if binary_class_weights and n_class_weights != 2:
        # `train`'s words (gbdt/train.mojo)
        raise Error(
            "class_weights takes 2 entries for loss 'Logloss', got "
            + String(n_class_weights)
        )
    if n_flags != 0 and (is_rmse or grow_code != 0 or is_pointwise or is_multi):
        _refuse("cat_features or one_hot_features outside SymmetricTree with Logloss")
    # the HELD-OUT arm (lane/close-no-cpu-path-gbdt, 2026-09-20): their
    # `testCursor`, held-out curve, overfitting detector and
    # `ShrinkToBestIteration`, restated in gbdt/host/gbdt_oracle_eval.mojo
    # for the SymmetricTree Logloss fit of gbdt/host/gbdt_oracle.mojo and
    # for that fit ONLY. Every other Plain arm still refuses by name: their
    # test cursor runs the same four pieces over a different loss kernel and
    # a different model shape, and neither is restated.
    var eval_arm_ok = (
        grow_code == 0 and loss == String("Logloss") and not is_pointwise
        and not is_multi and not is_rmse
    )
    if n_eval_rows != 0 and not eval_arm_ok:
        _refuse(
            "eval_set outside SymmetricTree with Logloss (the held-out arm"
            " of gbdt/host/gbdt_oracle_eval.mojo covers that fit only)"
        )
    var od = load_overfitting_detector_options(od_type, od_pvalue, od_wait)
    var od_kind = od.od_type
    if od_kind != OD_NONE and not eval_arm_ok:
        _refuse(
            "the overfitting detector (od_type, od_pvalue, od_wait) outside"
            " SymmetricTree with Logloss"
        )
    if od_kind != OD_NONE and n_eval_rows == 0:
        raise Error(
            "od_type is set but there is no held-out set. Stopping on the"
            " LEARN loss would stop on a curve that falls almost by"
            " construction; their own detector is inert without a test set"
            " (overfitting_detector.cpp:122-124) and this refuses rather"
            " than silently never firing."
        )
    if random_strength != Float32(0.0) and not ns_stochastic and not symmetric_stochastic and not pw_stochastic:
        _refuse(
            "random_strength=" + String(random_strength)
            + " under loss='" + loss + "'"
        )
    # the quantile family's constant (`sample_quantile.mojo`), shared with
    # the device fit, lane/catboost-parity
    var quantile_family = (
        pw_objective == GBDT_OBJ_MAE
        or pw_objective == GBDT_OBJ_QUANTILE
        or pw_objective == GBDT_OBJ_MAPE
    )
    if boost_from_average == 1 and not is_rmse and not quantile_family:
        _refuse("boost_from_average=True")
    if feature_fraction != 1.0 and not lossguide_knobs:
        _refuse("feature_fraction=" + String(feature_fraction) + " outside Lossguide with Logloss")
    if not (feature_fraction > 0.0) or feature_fraction > 1.0:
        raise Error("feature_fraction must be finite and in (0, 1]")
    if border_count < 1 or border_count > 255:
        _refuse("border_count=" + String(border_count) + " (1 to 255)")
    if max_depth < 0 or max_depth > GBDT_HOST_MAX_DEPTH:
        _refuse("max_depth=" + String(max_depth) + " (0 to 16)")

    # ---- `train`'s own raises on the rest (`gbdt/train.mojo:944-1149`,
    # `:1580-1605`, `:1733-1758`) ----
    # `child_hessian_threshold` (`gbdt/options/child_hessian.mojo:8-14`)
    if not isfinite(min_child_hessian) or (
        min_child_hessian < 0 and min_child_hessian != -1
    ) or min_child_hessian > Float64(Float32.MAX_FINITE):
        raise Error(
            "min_child_hessian must be -1 (disabled) or finite nonnegative"
            " and <= Float32.MAX_FINITE"
        )
    if min_child_hessian >= 0 and grow_code == 0:
        raise Error("min_child_hessian requires Depthwise or Lossguide")
    if min_child_hessian >= 0 and score_function != GBDT_HOST_SCORE_NEWTON_COSINE and score_function != GBDT_HOST_SCORE_NEWTON_L2:
        raise Error("min_child_hessian requires NewtonL2 or NewtonCosine; first-order scores store weights, not Hessians")
    # `child_hessian_threshold`'s round-up (`child_hessian.mojo:17-25`)
    var child_hessian = Float32(-1.0)
    if min_child_hessian >= 0:
        child_hessian = Float32(min_child_hessian)
        if Float64(child_hessian) < min_child_hessian:
            child_hessian = bitcast[DType.float32](bitcast[DType.uint32](child_hessian) + UInt32(1))
    if not isfinite(min_split_gain) or (
        min_split_gain < 0 and min_split_gain != -1
    ):
        raise Error("min_split_gain must be -1 (disabled) or finite and nonnegative")
    if min_split_gain >= 0 and grow_code == 0:
        raise Error("min_split_gain requires Depthwise or Lossguide")
    if min_data_in_leaf != 1 and grow_code == 0:
        raise Error(
            "min_data_in_leaf=" + String(min_data_in_leaf) + " does nothing"
            " under grow_policy=SymmetricTree: CatBoost guards its leaf-size"
            " test with `Policy != SymmetricTree`"
            " (greedy_search_helper.cpp:685) and discards the value; refused"
            " here rather than accepted and ignored. It is live under"
            " Depthwise and Lossguide."
        )
    # `fit_with_test`'s `ns_max_leaves` (`doc_parallel_boosting.mojo:
    # 1093-1113`) and `train`'s cap (`gbdt/train.mojo:968-983`)
    var ns_max_leaves = 1 << max_depth
    if grow_code != GBDT_HOST_GROW_LOSSGUIDE:
        if max_leaves >= 0 and max_leaves != (1 << max_depth):
            raise Error(
                "max_leaves option works only with lossguide tree growing"
                " (catboost_options.cpp:998): under grow_policy="
                + String("SymmetricTree" if grow_code == 0 else "Depthwise")
                + " CatBoost pins it to 1 << depth == "
                + String(1 << max_depth) + ", got " + String(max_leaves)
            )
    else:
        ns_max_leaves = max_leaves if max_leaves >= 0 else 31
        if ns_max_leaves > 65536:
            raise Error(
                "Maximum leaves count for Lossguide grow policy is 65536, got "
                + String(ns_max_leaves)
            )
        if ns_max_leaves < 2:
            raise Error(
                "max_leaves must be at least 2 under Lossguide, got "
                + String(ns_max_leaves)
            )
    var perm_count = permutation_count
    if perm_count == -1:
        perm_count = 4
    if perm_count < 1:
        raise Error(
            "Permutation count should be positive, got " + String(perm_count)
        )
    # ONE permutation unless a categorical column builds CTRs (`train`'s
    # Plain collapse, `gbdt/train.mojo:1433-1434`); the CTR arm below
    # (gbdt/host/gbdt_oracle_ctr.mojo) resolves the count and checks the
    # estimation permutation itself, because only the columns can say
    perm_count = 1
    var est_perm = ctr_permutation_id
    if est_perm == -1:
        est_perm = perm_count - 1
    if n_flags == 0 and (est_perm < 0 or est_perm >= perm_count):
        raise Error(
            "ctr_estimation_permutation_id " + String(est_perm)
            + " is outside the " + String(perm_count)
            + " permutations this fit builds"
        )
    if boost_from_average != -1 and boost_from_average != 0 and boost_from_average != 1:
        raise Error(
            "boost_from_average must be -1 (their data-dependent"
            " default), 0 or 1; got " + String(boost_from_average)
        )
    if use_best_model == 1 and n_eval_rows == 0:
        raise Error(
            "use_best_model=1 needs an eval set: pass eval_x_colmajor"
            " and eval_y, or leave it unset."
        )
    elif use_best_model != 0 and use_best_model != -1 and use_best_model != 1:
        raise Error(
            "use_best_model must be -1 (unset), 0 or 1, got "
            + String(use_best_model)
        )
    if best_model_min_trees < 1:
        raise Error(
            "best_model_min_trees must be at least 1, got "
            + String(best_model_min_trees)
        )
    var nan_mode = nan_mode_from_name(nan_mode_name)

    var iterations = GBDT_LOGLOSS_NEWTON_ITERATIONS
    if is_rmse:
        iterations = 1
    if leaf_iterations >= 0:
        iterations = leaf_iterations
    var border = GBDT_HOST_DEFAULT_BORDER
    if loss_border >= Float32(0.0):
        border = loss_border

    var p = GbdtHostParams(
        border_count, border_build_max_samples, n_estimators, max_depth,
        learning_rate, l2_leaf_reg, random_seed, nan_mode, border, iterations,
        border_type,
    )
    var pw_loss = GbdtHostLoss(-1, Float32(0), Float32(0), -1, -1, -1, Float32(0), border)
    # the non-symmetric fit's estimator and bootstrap: Logloss, Newton at the
    # resolved count or Gradient at 40 unless overridden
    # (`catboost_options.mojo:1304-1308`), Bernoulli at `subsample` or 0.66
    var ns_method = GBDT_LEAF_NEWTON
    var ns_iterations = iterations
    if leaf_method == GBDT_LEAF_GRADIENT:
        ns_method = GBDT_LEAF_GRADIENT
        ns_iterations = leaf_iterations if leaf_iterations >= 0 else 40
    var ns_boot = -1
    var ns_boot_param = Float32(0.0)
    if bootstrap_type == String("Bernoulli"):
        ns_boot = GBDT_BOOT_BERNOULLI
        var subsample = Float32(Float64(py=params[19]))
        ns_boot_param = subsample if subsample >= Float32(0.0) else Float32(0.66)
    elif bootstrap_type == String("Bayesian") and grow_code != 0:
        if Float32(Float64(py=params[19])) >= Float32(0.0):
            raise Error(
                "Error: default bootstrap type (bayesian) doesn't support"
                " 'subsample' option"
            )
        ns_boot = GBDT_BOOT_BAYESIAN
        ns_boot_param = Float32(Float64(py=params[18]))
    var ns_loss = GbdtHostLoss(
        GBDT_OBJ_RMSE if is_rmse else GBDT_OBJ_LOGLOSS, border, Float32(0.5),
        ns_method, ns_iterations, ns_boot, ns_boot_param, border,
    )
    if is_pointwise:
        pw_loss = _resolve_pointwise_loss(
            pw_objective, loss,
            Float32(Float64(py=params[11])), Float32(Float64(py=params[12])),
            Float32(Float64(py=params[13])), Float32(Float64(py=params[14])),
            leaf_method, leaf_iterations, bootstrap_type,
            Float32(Float64(py=params[19])), Float32(Float64(py=params[18])),
        )
    # the symmetric stochastic arm's bootstrap, `train`'s resolution
    # (`gbdt/train.mojo`), the Ordered arm's words above
    var sym_boot_kind = -1
    var sym_boot_param = Float32(1.0)
    if symmetric_stochastic:
        var sym_subsample = Float32(Float64(py=params[19]))
        if bootstrap_type == String("Bayesian"):
            if sym_subsample >= Float32(0.0):
                raise Error(
                    "Error: default bootstrap type (bayesian) doesn't support"
                    " 'subsample' option"
                )
            sym_boot_kind = GBDT_BOOT_BAYESIAN
            sym_boot_param = Float32(Float64(py=params[18]))
        elif bootstrap_type == String("Bernoulli"):
            sym_boot_kind = GBDT_BOOT_BERNOULLI
            sym_boot_param = sym_subsample if sym_subsample >= Float32(0.0) else Float32(0.66)
        elif bootstrap_type == String("Poisson"):
            sym_boot_kind = GBDT_BOOT_POISSON
            sym_boot_param = sym_subsample if sym_subsample >= Float32(0.0) else Float32(0.66)
    var flags = gbdt_host_load_u32(cp, n_flags)
    var x_address = Int(py=x_addr)
    var y_address = Int(py=y_addr)
    _ = xp
    _ = yp
    var text = String("")
    var losses = List[Float64]()
    var best_iteration = 0
    var stopped_early = False
    var has_nan = False
    if is_rmse or grow_code != 0 or is_pointwise or is_multi or len(flags) != 0:
        # NaN is measured on the symmetric Logloss fit without categorical
        # columns only (gbdt-nan-modes)
        var xs = f32_ptr(x_address)
        has_nan = gbdt_host_has_nan_ptr(xs, n_rows * n_features)
        if not has_nan and n_eval_rows > 0:
            var exs = f32_ptr(Int(py=eval_x_addr))
            has_nan = gbdt_host_has_nan_ptr(exs, n_eval_rows * n_features)
    if has_nan:
        _refuse(
            "an X carrying NaN under loss='" + loss + "' and grow_policy code "
            + String(grow_code) + " (NaN is measured on SymmetricTree with"
            " Logloss only, the gbdt-nan-modes lane)"
        )
    var eval_x_address = Int(py=eval_x_addr)
    var eval_y_address = Int(py=eval_y_addr)
    var test_losses = List[Float64]()
    with GILReleased(Python()):
        var x = read_f32(x_address, n_rows * n_features)
        # MultiRMSE's `target_dim` dim-major planes; one per row otherwise
        var y = read_f32(y_address, n_rows * target_dim)
        # `train`'s weight column for binary class weights: `1.0 *
        # class_weights[cls]`, cls the binarized target (y > 0.5)
        var row_weights = List[Float32]()
        if binary_class_weights:
            row_weights = gbdt_host_binary_class_row_weights(y, n_rows, class_weights)
        if is_multi:
            # gbdt/host/gbdt_oracle_multiclass.mojo
            var multi_objective = GBDT_OBJ_MULTICLASS_OVA
            if loss == String("MultiClass"):
                multi_objective = GBDT_OBJ_MULTICLASS
            elif is_multi_rmse:
                multi_objective = GBDT_OBJ_MULTIRMSE
            var multi_model = gbdt_multi_host_fit(
                x, y, n_rows, n_features, p,
                multi_objective,
                class_weights, sym_boot_kind, sym_boot_param, random_strength,
                target_dim,
            )
            text = gbdt_multi_host_model_text(multi_model)
            losses = multi_model.losses.copy()
        elif is_pointwise:
            # gbdt/host/gbdt_oracle_losses.mojo
            # `AdjustBoostFromAverageDefaultValue`: unset is True for MAE,
            # Quantile and MAPE (`gbdt/train.mojo`); the constant is
            # `calc_one_dimensional_optimum_const_approx`'s, the same host
            # code the device fit calls
            var pw_start = Float64(0.0)
            if quantile_family and boost_from_average != 0:
                if pw_objective == GBDT_OBJ_MAPE:
                    pw_start = Float64(
                        calculate_optimal_const_approx_for_mape(
                            y, List[Float32](), False
                        )
                    )
                else:
                    pw_start = Float64(
                        calculate_weighted_target_quantile(
                            y, List[Float32](), False,
                            0.5 if pw_objective == GBDT_OBJ_MAE else Float64(
                                pw_loss.estimator_alpha
                            ),
                            QUANTILE_CONST_DELTA,
                        )
                    )
            var pw_model = gbdt_losses_host_fit(
                x, y, n_rows, n_features, p, pw_loss, host_group_sizes,
                host_pair_winners, host_pair_losers, host_pair_weights,
                pw_start, random_strength,
            )
            losses = pw_model.losses.copy()
            best_iteration = pw_model.best_iteration
            stopped_early = pw_model.stopped_early
            text = gbdt_rmse_host_model_text(GbdtRmseHostFit(pw_model^, pw_start))
        elif is_rmse and grow_code == 0:
            # `AdjustBoostFromAverageDefaultValue` (`gbdt/train.mojo:
            # 1597-1600`): unset is True for RMSE
            var fit = gbdt_rmse_host_fit(
                x, y, n_rows, n_features, p, boost_from_average != 0,
                sym_boot_kind, sym_boot_param, random_strength,
            )
            text = gbdt_rmse_host_model_text(fit)
            losses = fit.model.losses.copy()
            best_iteration = fit.model.best_iteration
            stopped_early = fit.model.stopped_early
        elif grow_code == 0 and len(flags) != 0 and gbdt_has_ctr_columns(
            gbdt_ctr_column_kinds(flags, x, n_rows, n_features)
        ):
            # the CTR categorical arm (gbdt-categorical-ctr-tables):
            # gbdt/host/gbdt_oracle_ctr.mojo, every permutation
            # Full CPU binding compiled at 2d5c5081a83e, 2026-10-28:
            # x86-64-v3/O0, 32 jobs, exit 0. Runtime/identity are pending.
            # Receipt: docs/plans/gbdt-cpu-ctr-eval-20261028-compile.json.
            # The owner expands eval rows with its final learn-only tables
            # and reuses the existing held-out boost/detector/shrink arm.
            var ev = _gbdt_host_eval_arm(
                eval_x_address, eval_y_address, n_eval_rows, n_features,
                od_kind, od.auto_stop_p_value, od.iterations_wait,
                use_best_model, best_model_min_trees,
            )
            var ctr_fit = gbdt_ctr_host_fit(
                x, y, n_rows, n_features, p, flags, permutation_count,
                ctr_permutation_id, ev^,
            )
            text = gbdt_ctr_host_model_text(ctr_fit)
            losses = ctr_fit.fit.model.losses.copy()
            best_iteration = ctr_fit.fit.eval.best_iteration
            stopped_early = ctr_fit.fit.eval.stopped_early
            test_losses = ctr_fit.fit.eval.test_losses.copy()
        elif grow_code == 0 and len(flags) != 0:
            # the one-hot categorical arm (gbdt-categorical-ctr):
            # gbdt/host/gbdt_oracle_onehot.mojo
            if est_perm < 0 or est_perm >= perm_count:
                raise Error(
                    "ctr_estimation_permutation_id " + String(est_perm)
                    + " is outside the " + String(perm_count)
                    + " permutations this fit builds"
                )
            var one_hot = gbdt_resolve_one_hot(flags, x, n_rows, n_features)
            var ev = _gbdt_host_eval_arm(
                eval_x_address, eval_y_address, n_eval_rows, n_features,
                od_kind, od.auto_stop_p_value, od.iterations_wait,
                use_best_model, best_model_min_trees,
            )
            var r = gbdt_host_fit_eval(
                x, y, n_rows, n_features, p, one_hot, -1, Float32(1.0),
                Float32(0.0), ev^,
            )
            text = gbdt_host_model_text_one_hot(r.model, one_hot)
            losses = r.model.losses.copy()
            best_iteration = r.eval.best_iteration
            stopped_early = r.eval.stopped_early
            test_losses = r.eval.test_losses.copy()
        elif grow_code == 0:
            var ev = _gbdt_host_eval_arm(
                eval_x_address, eval_y_address, n_eval_rows, n_features,
                od_kind, od.auto_stop_p_value, od.iterations_wait,
                use_best_model, best_model_min_trees,
            )
            var r = gbdt_host_fit_eval(
                x, y, n_rows, n_features, p, List[Bool](),
                sym_boot_kind, sym_boot_param, random_strength, ev^,
                row_weights,
            )
            text = gbdt_host_model_text(r.model)
            losses = r.model.losses.copy()
            best_iteration = r.eval.best_iteration
            stopped_early = r.eval.stopped_early
            test_losses = r.eval.test_losses.copy()
        else:
            # Depthwise and Lossguide: gbdt/host/gbdt_oracle_depthwise.mojo
            var tp = GbdtHostTreeParams(
                p, grow_code, ns_max_leaves, Float64(min_data_in_leaf),
                score_function, child_hessian, min_split_gain,
                random_strength, feature_fraction, ns_loss,
            )
            # RMSE's starting constant, unset resolving True
            # (`gbdt/train.mojo:1597-1600`), written as the model's bias
            var ns_start = Float64(0.0)
            if is_rmse and boost_from_average != 0:
                ns_start = _rmse_starting_approx(y, n_rows)
            var ns_model = gbdt_host_fit_non_symmetric(
                x, y, n_rows, n_features, tp, ns_start, row_weights
            )
            text = gbdt_text_with_bias(
                gbdt_host_ns_model_text(ns_model), len(ns_model.losses), ns_start
            )
            losses = ns_model.losses.copy()

    var learn = py_list_f64(losses)
    var test = py_list_f64(test_losses)
    var out = Python.list()
    out.append(PythonObject(text))
    out.append(PythonObject(best_iteration))
    out.append(PythonObject(stopped_early))
    out.append(learn)
    out.append(test)
    return out


# ===========================================================================
# THE MODEL TEXT, READ (the records `gbdt_host_model_text` and the GPU
# binding's `model_text` write; `gbdt/models/model_text.mojo:706` is the
# reader this follows, for the oblivious float-only shape)
# ===========================================================================


def gbdt_predict_binding(
    model: PythonObject,
    x_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`gbdt_predict_binding` (`bindings/_mojolearn_gbdt.mojo:410-439`): RAW
    approxes for a one-dimensional model, `params` is `[n_rows]`, the
    feature count comes from the model, returns rows written."""
    if len(params) != 1:
        raise Error(
            "gbdt_predict: params must hold [n_rows], got "
            + String(len(params))
        )
    var text = String(py=model)
    var x_address = Int(py=x_addr)
    var op = f32_ptr(Int(py=out_addr))
    _ = f32_ptr(x_address)
    var n_rows = Int(py=params[0])
    if n_rows <= 0:
        raise Error("gbdt_predict: n_rows must be positive")
    var m = _parse_model(text)
    if m.dim != 1:
        raise Error(
            "predict_floats is one-dimensional; this model has"
            " approx_dim " + String(m.dim)
            + ". Use predict_multi_floats, which returns"
            " [row * approx_dim + dim]."
        )
    var n_features = len(m.fold_counts)
    var n_trees = len(m.tree_offsets) - 1
    with GILReleased(Python()):
        gbdt_host_predict_into(m^, x_address, n_rows, n_features, n_trees, op)
    return PythonObject(n_rows)


def gbdt_predict_multi_binding(
    model: PythonObject,
    x_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`gbdt_predict_multi_binding` (`bindings/_mojolearn_gbdt.mojo:453-485`)
    over `gbdt/estimator.mojo::gbdt_predict_multi`: `params` is
    `[n_rows, mode]`, mode 0 RAW (`dim` columns), 1 SOFTMAX (`dim + 1`,
    `gbdt/train.mojo::multiclass_probabilities`), 2 SIGMOID (`dim`,
    `one_vs_all_probabilities`), both transforms in double through
    `identical_exp64`, row-major out. Returns the width written."""
    if len(params) != 2:
        raise Error(
            "gbdt_predict_multi: params must hold [n_rows,"
            " as_probabilities], got " + String(len(params))
        )
    var text = String(py=model)
    var x_address = Int(py=x_addr)
    var op = f32_ptr(Int(py=out_addr))
    _ = f32_ptr(x_address)
    var n_rows = Int(py=params[0])
    var mode = Int(py=params[1])
    if n_rows <= 0:
        raise Error("gbdt_predict_multi: n_rows must be positive")
    var m = _parse_model(text)
    var dim = m.dim
    var n_features = len(m.fold_counts)
    var n_trees = len(m.tree_offsets) - 1
    if mode != 0 and dim < 2:
        raise Error(
            "gbdt_predict_multi: a probability mode needs a"
            " multi-dimensional model; this one has dim " + String(dim)
            + ". A two-class problem's link is the sigmoid, which"
            " Logloss's own predict_proba applies."
        )
    if mode != 0 and mode != 1 and mode != 2:
        raise Error("gbdt_predict_multi: unknown mode " + String(mode))
    var width = dim
    if mode == 1:
        width = dim + 1
    with GILReleased(Python()):
        gbdt_host_predict_multi_into(m^, x_address, n_rows, n_features, n_trees, dim, mode, op)
    return PythonObject(width)


def gbdt_model_dim_binding(model: PythonObject) raises -> PythonObject:
    """`gbdt_model_dim` (`bindings/_mojolearn_gbdt.mojo:442-450`): 1 for an
    empty ensemble, else the trees' common dim."""
    return PythonObject(_parse_model(String(py=model)).dim)


def gbdt_fit_two_level_feature_freq_binding(
    x_addr: PythonObject,
    y_addr: PythonObject,
    weights_addr: PythonObject,
    sources_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`gbdt_fit_two_level_feature_freq_binding`
    (`bindings/_mojolearn_gbdt.mojo:499-532`): the same seven params in the
    same order, `[n_rows, n_features, n_weights, n_sources, learning_rate,
    l2_leaf_reg, random_seed]`, the model text back, through
    `gbdt/host/gbdt_oracle_feature_freq.mojo::gbdt_feature_freq_host_fit`
    (the gbdt-feature-freq lane). `sample_weight` refuses by name. The seed
    is read and, as on the device (`gbdt/estimator.mojo:220-226` hands it to
    a searcher that draws no noise at `score_std_dev` 0), reaches no bit."""
    if len(params) != 7:
        raise Error("two-level FeatureFreq params must have seven values")
    var x_address = Int(py=x_addr)
    var y_address = Int(py=y_addr)
    _ = f32_ptr(x_address)
    _ = f32_ptr(y_address)
    _ = f32_ptr(Int(py=weights_addr))
    var sp = u32_ptr(Int(py=sources_addr))
    var n_rows = Int(py=params[0])
    var n_features = Int(py=params[1])
    var n_weights = Int(py=params[2])
    var n_sources = Int(py=params[3])
    var learning_rate = Float32(Float64(py=params[4]))
    var l2_leaf_reg = Float32(Float64(py=params[5]))
    _ = UInt64(Int(py=params[6]))
    if n_rows < 1 or n_features < 2 or n_sources < 2:
        raise Error("two-level FeatureFreq fit needs rows and two sources")
    if n_weights != 0 and n_weights != n_rows:
        raise Error("two-level FeatureFreq weights must be empty or per-row")
    if n_weights != 0:
        raise Error(
            "no CPU implementation of"
            " _mojolearn_gbdt.gbdt_fit_two_level_feature_freq for"
            " sample_weight; the gbdt host binding trains the"
            " gbdt-feature-freq lane only (unit weights), see"
            " gbdt/host/gbdt_oracle_feature_freq.mojo"
        )
    var sources = gbdt_host_load_sources(sp, n_sources)
    var text = String("")
    with GILReleased(Python()):
        var x = read_f32(x_address, n_rows * n_features)
        var y = read_f32(y_address, n_rows)
        text = gbdt_feature_freq_host_fit(
            x, y, n_rows, n_features, sources, learning_rate, l2_leaf_reg
        )
    return PythonObject(text)


def gbdt_fit_ordered_rmse_binding(
    x_addr: PythonObject, y_addr: PythonObject,
    weights_addr: PythonObject, permutation_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`gbdt_fit_ordered_rmse_binding` (`bindings/_mojolearn_gbdt.mojo:
    535-591`): the same nine params in the same order, `[n_rows,
    n_features, n_weights, n_permutation, n_estimators, max_depth,
    border_count, learning_rate, l2_leaf_reg]`, the model text back, through
    `gbdt/host/gbdt_oracle_ordered.mojo::gbdt_ordered_rmse_host_fit` (the
    gbdt-ordered-rmse lane). `sample_weight` refuses by name."""
    if len(params) != 9:
        raise Error("ordered RMSE params must have nine values")
    var n_rows = Int(py=params[0])
    var n_features = Int(py=params[1])
    var n_weights = Int(py=params[2])
    var n_permutation = Int(py=params[3])
    var n_estimators = Int(py=params[4])
    var max_depth = Int(py=params[5])
    var border_count = Int(py=params[6])
    var learning_rate = Float32(Float64(py=params[7]))
    var l2_leaf_reg = Float32(Float64(py=params[8]))
    if n_rows < 4 or n_features < 1 or n_permutation != n_rows:
        raise Error("ordered RMSE requires >=4 rows, features and a full permutation")
    if n_weights != 0 and n_weights != n_rows:
        raise Error("ordered RMSE sample weight shape mismatch")
    if n_weights != 0:
        raise Error(
            "no CPU implementation of _mojolearn_gbdt.gbdt_fit_ordered_rmse"
            " for sample_weight; the gbdt host binding trains the"
            " gbdt-ordered-rmse lane only (unit weights), see"
            " gbdt/host/gbdt_oracle_ordered.mojo"
        )
    var x_address = Int(py=x_addr)
    var y_address = Int(py=y_addr)
    _ = f32_ptr(x_address)
    _ = f32_ptr(y_address)
    _ = f32_ptr(Int(py=weights_addr))
    var pp = u32_ptr(Int(py=permutation_addr))
    var permutation = gbdt_host_load_u32(pp, n_rows)
    var text = String("")
    with GILReleased(Python()):
        var x = read_f32(x_address, n_rows * n_features)
        var y = read_f32(y_address, n_rows)
        text = gbdt_ordered_rmse_host_fit(
            x, y, n_rows, n_features, permutation, n_estimators, max_depth,
            border_count, learning_rate, l2_leaf_reg,
        )
    return PythonObject(text)


# TOMBSTONE: gbdt_sigmoid (dead native entry: no live caller) deleted 2026-10-26 by 7c1201602; code recoverable at 485ba9cc1.
# Restore: git apply experiments/removed/gbdt_sigmoid.patch


def gbdt_sigmoid_pair_binding(
    raw_addr: PythonObject, out_addr: PythonObject, n: PythonObject
) raises -> PythonObject:
    """`gbdt_sigmoid_pair_binding` (`bindings/_mojolearn_gbdt.mojo`, DEVIATION
    2902), its body: `out[2 * i] = 1 - p`, `out[2 * i + 1] = p` with `p`
    exactly `1 / (1 + identical_exp64(-raw))`, `n` rows, both buffers float64. The
    `1.0 - p` is the one IEEE double subtraction the Python layer computed
    per row. Under the forest walk's sabotage define
    (`MOJOLEARN_FOREST_HOST_SABOTAGE`, the lane's negative control) the two
    columns are written swapped."""
    var rp = f64_ptr(Int(py=raw_addr))
    var op = f64_ptr(Int(py=out_addr))
    var count = Int(py=n)
    if count < 0:
        raise Error("gbdt_sigmoid_pair: n must be non-negative")
    gbdt_host_sigmoid_pair_into(rp, op, count)
    return PythonObject(count)


def gbdt_binary_prediction_binding[probabilities: Bool, dtype: DType](
    raw_addr: PythonObject, out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """`gbdt_binary_prediction_binding` (`bindings/_mojolearn_gbdt.mojo:
    107-130`), the GradientBoostingClassifier adapter's two transforms, with
    the same checks in the same words. The body is
    `gbdt/binary_prediction.mojo::binary_prediction_kernel` restated per
    element on the host: the probability pair
    `[ftz(1 - p), p]` with `p = ftz(identical_sigmoid(ftz(margin)))`, and the
    class code from the margin's sign and magnitude bits (strict raw > 0,
    so a positive subnormal margin is class one on every column)."""
    if len(params) != 1:
        raise Error("binary prediction: params must contain n")
    var n = Int(py=params[0])
    if n <= 0 or n > 2147483647:
        raise Error("binary prediction: positive n<=Int32.max required")
    var rp = f32_ptr(Int(py=raw_addr))
    var address = Int(py=out_addr)
    if address == 0:
        raise Error("binary prediction: null output")
    var op = MutPointer[Scalar[dtype], MutUntrackedOrigin](unsafe_from_address=address)
    gbdt_host_binary_margins_check(rp, n)
    var count = 2 * n if probabilities else n
    with GILReleased(Python()):
        gbdt_host_binary_prediction_into[probabilities, dtype](rp, op, n)
    return PythonObject(count)


def gbdt_binary_probabilities_binding(
    raw_addr: PythonObject, out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """`[1 - p, p]` per margin, Float32 (see `gbdt_binary_prediction_binding`)."""
    return gbdt_binary_prediction_binding[True, DType.float32](raw_addr, out_addr, params)


def gbdt_binary_classes_binding(
    raw_addr: PythonObject, out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """The Int32 class code per margin (see `gbdt_binary_prediction_binding`)."""
    return gbdt_binary_prediction_binding[False, DType.int32](raw_addr, out_addr, params)


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_gbdt_host.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).
def build__mojolearn_gbdt_host() -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_gbdt_host")
        module.def_function[gbdt_host_numeric_mode_binding]("gbdt_host_numeric_mode")
        module.def_function[gbdt_host_vendor_binding]("gbdt_host_vendor")
        module.def_function[gbdt_host_column_binding]("gbdt_host_column")
        module.def_function[gbdt_host_sabotage_binding]("gbdt_host_sabotage")
        module.def_function[gbdt_vendor_binding]("gbdt_vendor")
        module.def_function[gbdt_numeric_mode_binding]("gbdt_numeric_mode")
        module.def_function[gbdt_fit_binding]("gbdt_fit")
        module.def_function[gbdt_predict_binding]("gbdt_predict")
        module.def_function[gbdt_predict_multi_binding]("gbdt_predict_multi")
        module.def_function[gbdt_model_dim_binding]("gbdt_model_dim")
        # TOMBSTONE: gbdt_sigmoid (dead native entry: no live caller) deleted 2026-10-26 by 7c1201602; code recoverable at 485ba9cc1.
        # Restore: git apply experiments/removed/gbdt_sigmoid.patch
        module.def_function[gbdt_sigmoid_pair_binding]("gbdt_sigmoid_pair")
        module.def_function[gbdt_binary_probabilities_binding]("gbdt_binary_probabilities")
        module.def_function[gbdt_binary_classes_binding]("gbdt_binary_classes")
        module.def_function[gbdt_fit_two_level_feature_freq_binding]("gbdt_fit_two_level_feature_freq")
        module.def_function[gbdt_fit_ordered_rmse_binding]("gbdt_fit_ordered_rmse")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_gbdt_host: ", error))
