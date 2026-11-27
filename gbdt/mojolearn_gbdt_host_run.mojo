# SPDX-License-Identifier: Apache-2.0
"""RUN of the GBDT host binding (bindings/_mojolearn_gbdt_host.mojo): the
fit's slot checks, refusals, loss resolution and every host fit arm, and the
typed bodies of the multi-dimensional predict, the two-level FeatureFreq fit
and the ordered RMSE fit. Typed Mojo values and addresses only, no Python
import (tools/python_free_dirs.sh). The door UNPACKs the Python arguments,
releases the GIL around each RUN call and PACKs the result (lane
rs-boosting-trees, 2026-11-12: moved word for word; the GIL releases that
sat inside the fit and its arms are hoisted to the door around the whole
RUN call, so the GIL is released as before or longer). Same operands in the
same order, so the bits are unchanged."""
from std.math import isfinite
from std.memory import bitcast


from core.hostptr import f32_ptr, read_f32, u32_ptr
from gbdt.fit_request import GbdtFitArgs, GbdtFitOutcome
from core.fit_loop_clock import FitLoop
# The typed bodies (lane thin-trees-metrics, 2026-10-21): the model text
# reader, the predict walks and links, the binary transforms, and the scans
# and reads of the fit's tails, moved word for word out of this shell.
from gbdt.host.gbdt_host_entries_impl import (
    _parse_model,
    gbdt_host_binary_row_weights,
    gbdt_host_eval_const,
    gbdt_host_group_sizes,
    gbdt_host_load_sources,
    gbdt_host_load_u32,
    gbdt_host_model_dim,
    gbdt_host_predict_ctr_into,
    gbdt_host_predict_multi_into,
    gbdt_host_read_pairs,
    gbdt_host_text_has_ctr,
)
from gbdt.host.gbdt_oracle_eval import GbdtHostEval, gbdt_eval_want_best_model
from gbdt.host.gbdt_oracle import (
    GBDT_BOOST_LOGLOSS,
    GBDT_BOOST_RMSE,
    GBDT_BOOT_BAYESIAN,
    GBDT_BOOT_BERNOULLI,
    GBDT_BOOT_POISSON,
    GBDT_LOGLOSS_NEWTON_ITERATIONS,
    GbdtHostFitWithEval,
    GbdtHostModel,
    GbdtHostParams,
    gbdt_host_fit,
    gbdt_host_fit_eval,
    gbdt_host_model_text,
)
from gbdt.host.gbdt_oracle_pointwise_rank import gbdt_ranking_pointwise_host_fit
from gbdt.host.gbdt_oracle_losses import (
    GBDT_LEAF_EXACT,
    GBDT_LEAF_GRADIENT,
    GBDT_LEAF_NEWTON,
    GBDT_LEAF_SIMPLE,
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
    gbdt_multi_host_model_text_ctr,
)
from gbdt.host.gbdt_oracle_rmse import (
    GbdtRmseHostFit,
    _rmse_starting_approx,
    gbdt_rmse_host_model_text,
    gbdt_text_with_bias,
)
from gbdt.host.gbdt_oracle_depthwise import (
    GBDT_HOST_GROW_LOSSGUIDE,
    GBDT_HOST_SCORE_NEWTON_COSINE,
    GBDT_HOST_SCORE_NEWTON_L2,
    GbdtHostTreeParams,
    gbdt_ctr_host_fit_non_symmetric,
    gbdt_ctr_host_ns_model_text,
    gbdt_host_fit_non_symmetric,
    gbdt_host_ns_model_text,
)
from gbdt.options.data_processing_options import nan_mode_from_name
from gbdt.options.catboost_options import (
    check_tree_limits,
    pointwise_policy_check,
    querywise_pointwise_leaf_check,
    simple_leaf_iterations_check,
    simple_leaves_check,
)
from gbdt.options.child_hessian import (
    check_child_hessian_objective,
    child_hessian_threshold,
)
from gbdt.metrics.optimal_const_for_loss import (
    calc_one_dimensional_optimum_const_approx,
)
from gbdt.options.overfitting_detector_options import (
    load_overfitting_detector_options,
)
from gbdt.overfitting_detector.overfitting_detector import OD_NONE
from gbdt.grid_creator.binarization import (
    BORDER_TYPE_GREEDY_LOG_SUM,
    border_type_from_name,
)
from gbdt.metrics.optimal_const_for_loss import (
    calc_one_dimensional_optimum_const_approx,
)
from gbdt.metrics.sample_quantile import (
    calculate_optimal_const_approx_for_mape,
    calculate_weighted_target_quantile,
    host_weighted_l2,
)
from gbdt.metrics.optimal_const_for_loss import (
    calc_one_dimensional_optimum_const_approx,
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
    GbdtCtrHostFit,
    gbdt_ctr_column_kinds,
    gbdt_ctr_host_fit,
    gbdt_ctr_host_model_text,
    gbdt_has_ctr_columns,
)
from gbdt.host.gbdt_oracle_onehot import (
    gbdt_host_model_text_one_hot,
    gbdt_resolve_one_hot,
)


#: their Quantile / MAE `delta` default (`optimal_const_for_loss.h:198`),
#: the value `gbdt/metrics/optimal_const_for_loss.mojo` passes
comptime QUANTILE_CONST_DELTA = 1e-6


#: `SCORE_FUNCTION_COSINE` and `LEAF_ESTIMATION_NEWTON`
#: (`gbdt/options/catboost_options.mojo:130`, `:270`), the codes the wrapper
#: sends in slots 10 and 17.
comptime GBDT_HOST_SCORE_COSINE = 1



#: `SCORE_FUNCTION_L2` (`gbdt/options/catboost_options.mojo:135`).
comptime GBDT_HOST_SCORE_L2 = 6


#: `DEFAULT_TARGET_BORDER` (`gbdt/options/loss_description.mojo:72`).
comptime GBDT_HOST_DEFAULT_BORDER = Float32(0.5)


# lane gbdt-cpu-rest-4-tail (2026-11-23): `GBDT_HOST_MAX_DEPTH`, the
# host's own depth cap, is gone: the depth and border-count limits are
# `train`'s (`gbdt/options/catboost_options.mojo::check_tree_limits`, one
# body for both columns)


def _fit_outcome(var text: String, best_iteration: Int, stopped_early: Bool, var losses: List[Float64],
                 var test_losses: List[Float64], fit_loop: FitLoop) -> GbdtFitOutcome:
    """The host fit's outcome with its boosting loop's record (`fit_loop_`,
    lane ab-loop-mode: the rounds and the time the fit's own two clock
    reads bracketed)."""
    var outcome = GbdtFitOutcome(text^, best_iteration, stopped_early, losses^, test_losses^)
    outcome.fit_loop = fit_loop.copy()
    return outcome^


def _fit_model(mut fit: GbdtHostFitWithEval) -> GbdtHostModel:
    """The ensemble of a fit with its held-out report, swapped out for an empty model so the pair stays
    destructible (AGENTS.md part 6; its eval fields are read before)."""
    var model = GbdtHostModel(
        List[Int](), List[List[Float32]](), List[Int](), List[Int](), List[Int](), List[Int](), List[Int](),
        List[Float32](), List[Float64](), 0, False,
    )
    swap(model, fit.model)
    return model^


# lane gbdt-cpu-rest-4-simple (2026-11-23): `_refuse`, the by-name
# refusal ("no CPU implementation of _mojolearn_gbdt.gbdt_fit for ..."), is
# deleted with its last caller: every combination it named is trained or
# refused on both columns in the device's words (python/mojolearn/
# host_surface.py `NO_CPU_PATH` is deleted).


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


# Lane gbdt-cpu-rest-4-pwl (2026-11-23): `_refuse_pointwise`, the pointwise
# arm's by-name refusal, is deleted with its last callers: every value it
# refused is trained (the one-output losses, border types, detectors,
# boost_from_average resolved, categorical columns, depth 0) or refused on
# both columns in the device's words (grow policies, the multi-output
# losses, feature combinations).


def _gbdt_fit_pointwise_arm(
    params: GbdtFitArgs,
    border_type: Int,
    feature_fraction: Float64,
    min_split_gain: Float64,
    min_child_hessian: Float64,
) raises -> GbdtFitOutcome:
    """`gbdt_fit` with `use_pointwise_searcher=True` on a one-output loss
    (the grouped losses take `gbdt_ranking_pointwise_host_fit`): the slots
    were checked by the caller; this reads them again, applies the
    device's own refusals in its words and order (`gbdt/train.mojo::train`,
    `fit_with_test`), resolves the loss, the weight column, the starting
    constant and the held-out arm as `train` does, and fits through
    `gbdt_pointwise_host_fit`.

    Lane gbdt-cpu-rest-4-pwl (2026-11-23): the whole of the device's
    pointwise arm. Trained: CrossEntropy, MAE, Quantile, MAPE,
    LogLinQuantile, Poisson, Lq, Expectile, Tweedie and Huber beside Logloss
    and RMSE, each at its leaf methods (Exact where `train` resolves it),
    bootstraps, sample and class weights; `boost_from_average` True, False
    or unset (`AdjustBoostFromAverageDefaultValue`); every
    `feature_border_type`; an eval set under any detector (Iter, IncToDec,
    or none); one-hot and CTR columns; `max_depth` 0. Refused in the
    device's words: Depthwise and Lossguide (`train`), feature combinations
    (`fit_with_test`), `boost_from_average=True` outside its list (`train`).
    MultiClass, MultiClassOneVsAll and MultiRMSE take the multi-output
    fit (`gbdt_multi_host_fit(use_pointwise_searcher=True)`)."""
    var n_class_weights = params.i(35)
    var n_rows = params.i(0)
    var n_features = params.i(1)
    var n_weights = params.i(2)
    var n_flags = params.i(3)
    var border_count = params.i(4)
    var n_estimators = params.i(5)
    var max_depth = params.i(6)
    var learning_rate = Float32(params.f(7))
    var l2_leaf_reg = Float32(params.f(8))
    var random_seed = UInt64(params.i(9))
    var score_function = params.i(10)
    var loss = params.strs[0]
    var loss_border = Float32(params.f(15))
    var leaf_iterations = params.i(16)
    var leaf_method = params.i(17)
    var bagging_temperature = Float32(params.f(18))
    var subsample = Float32(params.f(19))
    var bootstrap_type = params.strs[1]
    var n_eval_rows = params.i(20)
    var od_type = params.strs[2]
    var od_pvalue = params.f(21)
    var od_wait = params.i(22)
    var use_best_model = params.i(23)
    var best_model_min_trees = params.i(24)
    var nan_mode_name = params.strs[3]
    var random_strength = Float32(params.f(25))
    var border_build_max_samples = params.i(27)
    var permutation_count = params.i(28)
    var ctr_permutation_id = params.i(29)
    var boost_from_average = params.i(30)
    var max_leaves = params.i(32)
    var min_data_in_leaf = params.i(33)
    var max_ctr_complexity = params.i(34)
    # (lane gbdt-cpu-rest-4-tail, 2026-11-23: the policy code this arm's
    # checks read was never declared here; the caller's slot 31)
    var grow_code = params.i(31)

    # ---- `train`'s option checks, its words and its order
    # (`gbdt/train.mojo`, the grow policy block) ----
    if feature_fraction < 1.0 and n_flags != 0:
        raise Error("feature_fraction<1 supports numeric features only; categorical/CTR input refused")
    # lane gbdt-cpu-rest-4-tail (2026-11-23): `train`'s border-count and
    # depth limits in its words and place, one body (`check_tree_limits`)
    check_tree_limits(border_count, max_depth, grow_code == GBDT_HOST_GROW_LOSSGUIDE)
    if not isfinite(min_child_hessian) or (
        min_child_hessian < 0 and min_child_hessian != -1
    ) or min_child_hessian > Float64(Float32.MAX_FINITE):
        raise Error(
            "min_child_hessian must be -1 (disabled) or finite nonnegative"
            " and <= Float32.MAX_FINITE"
        )
    if min_child_hessian >= 0 and grow_code == 0:
        raise Error("min_child_hessian requires Depthwise or Lossguide")
    if not isfinite(min_split_gain) or (min_split_gain < 0 and min_split_gain != -1):
        raise Error("min_split_gain must be -1 (disabled) or finite and nonnegative")
    if grow_code == 0 and min_split_gain >= 0:
        raise Error("min_split_gain requires Depthwise or Lossguide")
    if grow_code == 0 and min_data_in_leaf != 1:
        raise Error(
            "min_data_in_leaf=" + String(min_data_in_leaf) + " does nothing"
            " under grow_policy=SymmetricTree: CatBoost guards its leaf-size"
            " test with `Policy != SymmetricTree`"
            " (greedy_search_helper.cpp:685) and discards the value; refused"
            " here rather than accepted and ignored. It is live under"
            " Depthwise and Lossguide."
        )
    if grow_code != GBDT_HOST_GROW_LOSSGUIDE and max_leaves >= 0 and max_leaves != (1 << max_depth):
        raise Error(
            "max_leaves option works only with lossguide tree growing"
            " (catboost_options.cpp:998): under grow_policy="
            + String("SymmetricTree" if grow_code == 0 else "Depthwise")
            + " CatBoost pins it to 1 << depth == "
            + String(1 << max_depth) + ", got " + String(max_leaves)
        )
    if grow_code == GBDT_HOST_GROW_LOSSGUIDE and max_leaves >= 0 and max_leaves > 65536:
        raise Error(
            "Maximum leaves count for Lossguide grow policy is 65536; got "
            + String(max_leaves)
        )
    # `train`'s body (gbdt/options/catboost_options.mojo), both columns
    pointwise_policy_check(True, grow_code)
    # ---- the loss: the one-output losses this searcher trains; the
    # multi-output ones refused in `fit_with_test`'s words ----
    var pw_is_rmse = loss == String("RMSE")
    var objective = _pointwise_objective(loss)
    if loss == String("Logloss"):
        objective = GBDT_OBJ_LOGLOSS
    elif pw_is_rmse:
        objective = GBDT_OBJ_RMSE
    # lane gbdt-pw-multiclass (2026-11-23): MultiClass, MultiClassOneVsAll and
    # MultiRMSE never reach this arm: the caller sends them to the multi-output fit
    # with the doc-parallel searcher (`gbdt_multi_host_fit(
    # use_pointwise_searcher=True)`), as the device trains them now
    if objective < 0 or objective == GBDT_OBJ_QUERY_RMSE or objective == GBDT_OBJ_PAIR_LOGIT or objective == GBDT_OBJ_YETI_RANK:
        raise Error(
            "the pointwise searcher's one-output arm takes Logloss, RMSE,"
            " CrossEntropy and the pointwise losses; got loss='" + loss + "'"
        )
    # lane gbdt-rest-3 (2026-11-23): the four score functions the public
    # API admits (SolarL2, LOOL2 and SatL2 are refused by the public API
    # itself, python/mojolearn/ensemble.py `_UNSUPPORTED_SCORE_FUNCTIONS`),
    # Gradient leaves, every bootstrap the device's pointwise arm draws and
    # any random_strength (gbdt_oracle_pointwise.mojo)
    var pw_second_order = (
        score_function == GBDT_HOST_SCORE_NEWTON_L2
        or score_function == GBDT_HOST_SCORE_NEWTON_COSINE
    )
    var pw_cosine = (
        score_function == GBDT_HOST_SCORE_COSINE
        or score_function == GBDT_HOST_SCORE_NEWTON_COSINE
    )
    if score_function != GBDT_HOST_SCORE_L2 and not pw_second_order and not pw_cosine:
        # lane gbdt-cpu-rest-4 (2026-11-23): the device kernel's words, both columns
        raise Error(
            "score_function has no calcer here; only Cosine, NewtonCosine,"
            " L2 and NewtonL2 are implemented"
        )
    if (pw_is_rmse or objective == GBDT_OBJ_LOGLOSS) and leaf_method == GBDT_LEAF_EXACT:
        # `train`'s words (`use_exact_leaves`), both columns
        raise Error("Only MAPE, MAE and Quantile are supported for Exact leaves estimation on GPU")
    if leaf_method != -1 and leaf_method != GBDT_LEAF_NEWTON and leaf_method != GBDT_LEAF_GRADIENT and leaf_method != GBDT_LEAF_EXACT:
        # lane gbdt-cpu-rest-4 (2026-11-23): an unknown code never reaches
        # the walker, the device's words, both columns (Simple is refused
        # before this arm by `simple_leaves_check`, lane
        # gbdt-cpu-rest-4-simple, 2026-11-23)
        raise Error(
            "WriteSecondDerivatives: only Newton and Gradient reach"
            " the walker; Exact replaces it"
        )
    var pw_boot_kind = -1
    var pw_boot_param = Float32(0.0)
    if bootstrap_type == String("Bayesian"):
        if subsample >= Float32(0.0):
            raise Error(
                "Error: default bootstrap type (bayesian) doesn't support"
                " 'subsample' option"
            )
        pw_boot_kind = GBDT_BOOT_BAYESIAN
        pw_boot_param = bagging_temperature
    elif bootstrap_type == String("Bernoulli"):
        pw_boot_kind = GBDT_BOOT_BERNOULLI
        pw_boot_param = subsample if subsample >= Float32(0.0) else Float32(0.66)
    elif bootstrap_type == String("Poisson"):
        pw_boot_kind = GBDT_BOOT_POISSON
        pw_boot_param = subsample if subsample >= Float32(0.0) else Float32(0.66)
    elif bootstrap_type == String("MVS"):
        # `train`'s words (gbdt/train.mojo), both columns
        raise Error(
            "MVS is not reachable from their GPU oblivious searcher"
            " (weak_objective_impl.h:30 asserts it away)"
        )
    elif bootstrap_type != String("No") and bootstrap_type != String(""):
        raise Error(
            "unknown bootstrap_type '" + bootstrap_type
            + "': Bayesian, Bernoulli, Poisson, No"
        )
    # lane gbdt-one-body (2026-11-23): class weights train on this
    # searcher as on the device: `train`'s one weight column (the sample
    # word times `class_weights[y > 0.5]`, `class_weight_body`), the
    # weighted arm on (`fit_with_test`'s `use_class_weights or
    # use_sample_weight`). `train`'s checks, its words and its order; lane
    # gbdt-cpu-rest-4-pwl: on every one-output loss but RMSE, as `train`
    if n_class_weights != 0 and pw_is_rmse:
        raise Error(
            "class weights take effect only with a classification loss,"
            " their option check's words (catboost_options.cpp:617)"
        )
    if n_class_weights != 0 and n_class_weights != 2:
        raise Error(
            "class_weights takes 2 entries for loss '" + loss + "', got "
            + String(n_class_weights)
        )
    # the overfitting detector as `gbdt_fit` loads it (`TOverfittingDetector
    # Options::Load`, gbdt/estimator.mojo), the device's detector options
    var od = load_overfitting_detector_options(od_type, od_pvalue, od_wait)
    if n_eval_rows == 0:
        if od.od_type != OD_NONE:
            raise Error(
                # `train`'s words (gbdt/train.mojo), both columns
                "od_type='" + od_type + "' needs an eval set: pass"
                " eval_x_colmajor and eval_y. Stopping on the learn loss"
                " would stop on a curve that falls by construction."
            )
        if use_best_model == 1:
            raise Error(
                "use_best_model=1 needs an eval set: pass eval_x_colmajor"
                " and eval_y, or leave it unset."
            )
    if use_best_model != -1 and use_best_model != 0 and use_best_model != 1:
        raise Error("use_best_model must be -1 (unset), 0 or 1, got " + String(use_best_model))
    if best_model_min_trees < 1:
        raise Error("best_model_min_trees must be at least 1, got " + String(best_model_min_trees))
    # `AdjustBoostFromAverageDefaultValue` (`gbdt/train.mojo`), its words:
    # set is kept (True only on their list), unset is True for RMSE, MAE,
    # Quantile and MAPE
    if boost_from_average != -1 and boost_from_average != 0 and boost_from_average != 1:
        raise Error(
            "boost_from_average must be -1 (their data-dependent"
            " default), 0 or 1; got " + String(boost_from_average)
        )
    var bfa_listed = (
        objective == GBDT_OBJ_RMSE or objective == GBDT_OBJ_LOGLOSS
        or objective == GBDT_OBJ_CROSSENTROPY or objective == GBDT_OBJ_QUANTILE
        or objective == GBDT_OBJ_MAE or objective == GBDT_OBJ_MAPE
    )
    if boost_from_average == 1 and not bfa_listed:
        raise Error(
            "You can use boost_from_average only for these loss"
            " functions now: RMSE, Logloss, CrossEntropy, Quantile, MAE,"
            " MAPE (catboost_options.cpp:705-709; their MultiQuantile"
            " and RMSPE are not trained here, and MultiRMSE's"
            " per-dimension start is refused above)."
        )
    var bfa = boost_from_average == 1 or (
        boost_from_average == -1 and (
            objective == GBDT_OBJ_RMSE or objective == GBDT_OBJ_QUANTILE
            or objective == GBDT_OBJ_MAE or objective == GBDT_OBJ_MAPE
        )
    )
    # lane gbdt-rest-3 (2026-11-23): feature_fraction is the device's
    # per-tree sample on this searcher (`_pw_sample_tree`)
    if not (feature_fraction > 0.0) or feature_fraction > 1.0:
        raise Error("feature_fraction must be finite and in (0, 1]")
    # lane gbdt-cpu-rest-4-tail (2026-11-23): a negative depth is the
    # searcher's refusal (`create_subsets`); the limits above are `train`'s
    if max_depth < 0:
        raise Error(
            String("maxDepth must be non-negative, got ") + String(max_depth)
        )
    if max_ctr_complexity < 1:
        # `TCatFeatureParams.check`'s words (`train` raises them)
        raise Error(
            "max_ctr_complexity must be at least 1; got "
            + String(max_ctr_complexity)
        )
    var perm_count = permutation_count
    if perm_count == -1:
        perm_count = 4
    if perm_count < 1:
        raise Error(
            "Permutation count should be positive, got " + String(perm_count)
        )
    var nan_mode = nan_mode_from_name(nan_mode_name)

    # ---- the loss as `train` resolves it on this arm ----
    var pw_loss: GbdtHostLoss
    var border = GBDT_HOST_DEFAULT_BORDER
    if loss_border >= Float32(0.0):
        border = loss_border
    if objective == GBDT_OBJ_LOGLOSS or pw_is_rmse:
        # `get_estimation_method_defaults`: Logloss Newton 10 / Gradient 40,
        # RMSE 1 / 1 (gbdt/options/catboost_options.mojo); Logloss's border
        # in the kernel alpha
        var pw_method = GBDT_LEAF_NEWTON if leaf_method == -1 else leaf_method
        var iterations = GBDT_LOGLOSS_NEWTON_ITERATIONS
        if pw_method == GBDT_LEAF_GRADIENT:
            iterations = 40
        if pw_is_rmse:
            iterations = 1
        if leaf_iterations >= 0:
            iterations = leaf_iterations
        pw_loss = GbdtHostLoss(
            objective, border, Float32(0.5), pw_method, iterations,
            pw_boot_kind, pw_boot_param, border,
        )
    else:
        # the pointwise losses: their parameters, leaf defaults (Exact for
        # MAE, MAPE and Quantile on this Plain arm) and refusals, as the
        # greedy arm resolves them
        pw_loss = _resolve_pointwise_loss(
            objective, loss,
            Float32(params.f(11)), Float32(params.f(12)),
            Float32(params.f(13)), Float32(params.f(14)),
            leaf_method, leaf_iterations, bootstrap_type,
            subsample, bagging_temperature,
        )

    var x = read_f32(params.x, n_rows * n_features)
    var y = read_f32(params.y, n_rows)
    var flags = List[UInt32]()
    if n_flags != 0:
        flags = gbdt_host_load_u32(u32_ptr(params.cat_flags), n_flags)
    # `train`'s one weight column: the sample weights times the class
    # weights, a negative weight refused at its first row (`train`'s
    # words); empty is the unweighted arm. One parallel pass, once per fit.
    var w = List[Float32]()
    if n_class_weights != 0:
        var class_weights = List[Float32]()
        for i in range(n_class_weights):  # small-loop(class weight words of params)
            class_weights.append(Float32(params.f(36 + i)))
        var sample_weights = List[Float32]()
        if n_weights != 0:
            sample_weights = read_f32(params.weights, n_rows)
        w = gbdt_host_binary_row_weights(y, n_rows, class_weights, sample_weights)
    elif n_weights != 0:
        w = gbdt_host_binary_row_weights(
            y, n_rows, List[Float32](), read_f32(params.weights, n_rows)
        )
    # THE WEIGHTED L2 (lane w3f2-gbdtw, 2026-11-24): `train`'s
    # `device_weighted_l2` over the same column, `host_weighted_l2` here
    # (the same fold, once per fit); empty `w` keeps `l2_leaf_reg`
    if loss != String("PairLogit"):
        l2_leaf_reg = host_weighted_l2(l2_leaf_reg, w, n_rows)
    # the starting constant (`calc_one_dimensional_optimum_const_approx`,
    # the one statement of every arm the device's `optimum_const_approx_
    # device` folds on the device)
    var start = Float64(0.0)
    if bfa:
        start = calc_one_dimensional_optimum_const_approx(
            objective, y, w, len(w) > 0, Float64(pw_loss.estimator_alpha)
        )
    var p = GbdtHostParams(
        border_count, border_build_max_samples, n_estimators, max_depth,
        learning_rate, l2_leaf_reg, random_seed, nan_mode, border,
        pw_loss.iterations, border_type,
    )
    var ev = _gbdt_host_eval_arm(
        params.eval_x, params.eval_y, n_eval_rows, n_features,
        od.od_type, od.auto_stop_p_value, od.iterations_wait,
        use_best_model, best_model_min_trees,
    )
    var fit = gbdt_pointwise_host_fit(
        x, y, w, n_rows, n_features, p, pw_loss, ev, start, flags,
        permutation_count, ctr_permutation_id, pw_second_order, pw_cosine,
        random_strength, feature_fraction, max_ctr_complexity,
    )
    var text = fit.text.copy()
    var losses = fit.losses.copy()
    var test_losses = fit.test_losses.copy()
    return _fit_outcome(text^, fit.best_iteration, fit.stopped_early, losses^, test_losses^, fit.fit_loop)


# lane gbdt-cpu-rest-4-simple (2026-11-23): `_refuse_ordered_host`, the
# Ordered arm's by-name refusal, is deleted: it had no caller left.


def _gbdt_fit_ordered_arm(
    x_address: Int,
    y_address: Int,
    flags_address: Int,
    eval_x_address: Int,
    eval_y_address: Int,
    params: GbdtFitArgs,
    border_type: Int,
) raises -> GbdtFitOutcome:
    """`gbdt_fit` with `boosting_type='Ordered'`: `train`'s Ordered refusals
    in its words, the host binding's own (what the oracle does not restate),
    the loss resolved as `train` resolves it under Ordered (their GPU
    `useExact` needs Plain, `catboost_options.cpp:290-293`, so MAE, MAPE and
    Quantile keep Gradient), then `gbdt_ordered_host_fit`."""
    var n_class_weights = params.i(35)
    var fixed_and_weights = 36 + n_class_weights
    var n_rows = params.i(0)
    var n_features = params.i(1)
    var n_weights = params.i(2)
    var n_flags = params.i(3)
    var border_count = params.i(4)
    var n_estimators = params.i(5)
    var max_depth = params.i(6)
    var learning_rate = Float32(params.f(7))
    var l2_leaf_reg = Float32(params.f(8))
    var random_seed = UInt64(params.i(9))
    var score_function = params.i(10)
    var loss_border = Float32(params.f(15))
    var leaf_iterations = params.i(16)
    var leaf_method = params.i(17)
    var bagging_temperature = Float32(params.f(18))
    var subsample = Float32(params.f(19))
    var n_eval_rows = params.i(20)
    var od_pvalue = params.f(21)
    var od_wait = params.i(22)
    var random_strength = Float32(params.f(25))
    var use_pointwise = params.i(26) != 0
    var border_build_max_samples = params.i(27)
    var permutation_count = params.i(28)
    var boost_from_average = params.i(30)
    var grow_code = params.i(31)
    var feature_fraction = Float64(1)
    if len(params) >= fixed_and_weights + 3:
        feature_fraction = params.f(fixed_and_weights + 2)
    var loss = params.strs[0]
    var bootstrap_type = params.strs[1]
    var od_type = params.strs[2]
    var nan_mode = nan_mode_from_name(params.strs[3])
    var fold_len_multiplier = bitcast[DType.float64](UInt64(Int(params.strs[6])))
    var fold_permutation_block = Int(params.strs[7])

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
    # ---- the row weights (lane gbdt-ordered-weights, 2026-11-23) ----
    # sample weights and class weights train under Ordered as on the
    # device: `train`'s one `weights` column (the sample word times
    # `class_weights[y > 0.5]`), threaded by `gbdt_ordered_host_fit` through
    # the fold planes, the per-fold and estimation leaves, the starting
    # average and the learn loss. `gbdt_fit`'s and `train`'s checks, their
    # words (this arm runs before the Plain arms' slot checks).
    if n_weights != 0 and n_weights != n_rows:
        raise Error(
            "gbdt_fit: n_weights must be 0 or n_rows, got " + String(n_weights)
        )
    if n_class_weights != 0 and loss == String("RMSE"):
        raise Error(
            "class weights take effect only with a classification loss,"
            " their option check's words (catboost_options.cpp:617)"
        )
    if n_class_weights != 0 and n_class_weights != 2:
        raise Error(
            "class_weights takes 2 entries for loss '" + loss + "', got "
            + String(n_class_weights)
        )
    # CATEGORICAL COLUMNS (lane cat-ctr, 2026-11-21): the Ordered oracle
    # takes the flags and builds single-column CTRs over its fold
    # permutations, as `train` does. What stays refused, by name: an
    # estimation permutation that is not the last (`train`'s own refusal,
    # its words). A one-hot column is fitted (the fold searcher's take-bin
    # arm) and feature combinations are grown by the level driver, as on
    # the device.
    var ctr_permutation_id = params.i(29)
    var max_ctr_complexity = params.i(34)
    if max_ctr_complexity < 1:
        raise Error(
            "max_ctr_complexity must be at least 1; got "
            + String(max_ctr_complexity)
        )
    if n_flags != 0 and n_flags != n_features:
        raise Error("cat flags must be one per feature")
    # the detector as `gbdt_fit` resolves it, through the same host function
    # (`load_overfitting_detector_options`, their `Load`)
    var use_best_model = params.i(23)
    var best_model_min_trees = params.i(24)
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
            # `train`'s words (gbdt/train.mojo), both columns
            "od_type='" + od_type + "' needs an eval set: pass"
            " eval_x_colmajor and eval_y. Stopping on the learn loss"
            " would stop on a curve that falls by construction."
        )
    var ordered_bfa_losses = (
        loss == String("RMSE") or loss == String("MAE")
        or loss == String("Quantile") or loss == String("MAPE")
    )
    # lane gbdt-cpu-rest-4-sym (2026-11-23): boost_from_average=True
    # trains Logloss and CrossEntropy under Ordered too (their log-odds
    # start, `gbdt_ordered_host_fit`'s starting point); every loss outside
    # `train`'s list refuses on both columns in its words
    if boost_from_average == 1 and not ordered_bfa_losses and loss != String("Logloss") and loss != String("CrossEntropy"):
        raise Error(
            "You can use boost_from_average only for these loss"
            " functions now: RMSE, Logloss, CrossEntropy, Quantile, MAE,"
            " MAPE (catboost_options.cpp:705-709; their MultiQuantile"
            " and RMSPE are not trained here, and MultiRMSE's"
            " per-dimension start is refused above)."
        )
    # lane gbdt-cpu-rest-4-tail (2026-11-23): `train`'s limits in its
    # words (`check_tree_limits`; Ordered is SymmetricTree), the depth range
    # below 1 is `fit_ordered`'s (`gbdt_ordered_host_fit`)
    check_tree_limits(border_count, max_depth, False)
    # lane gbdt-cpu-rest-4-simple (2026-11-23): Simple's admission, the
    # device's (`train`, right after `set_leaves_estimation_default`): one
    # iteration, then refused under Ordered, whose every tree is estimated
    # (the device reached "a walk reached no device walker" at the first
    # tree; this arm refused by name in the walker's words). YetiRank is
    # refused first on both columns, by its own words.
    if loss != String("YetiRank"):
        simple_leaf_iterations_check(
            leaf_method, leaf_iterations if leaf_iterations >= 0 else 1
        )
        simple_leaves_check(leaf_method, False, True, False, use_pointwise)

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
        elif method == GBDT_LEAF_EXACT:
            # `set_leaves_estimation_default`'s words, both columns (lane
            # gbdt-cpu-rest-4, 2026-11-23)
            raise Error(
                "Exact leaf estimation method don't work with ordered boosting"
                " on GPU (catboost_options.cpp:346-350)"
            )
        else:
            # an unknown code never reaches the walker, the device's words,
            # both columns (lane gbdt-cpu-rest-4; Simple is refused above,
            # lane gbdt-cpu-rest-4-simple)
            raise Error(
                "WriteSecondDerivatives: only Newton and Gradient reach"
                " the walker; Exact replaces it"
            )
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
            # `objective_from_name`'s words, both columns (lane gbdt-cpu-rest-4)
            raise Error(
                "unknown loss '" + loss + "': this implementation trains RMSE, Logloss,"
                " CrossEntropy, Quantile, MAE, LogLinQuantile, MAPE, Poisson, Lq,"
                " Expectile, Tweedie, Huber, MultiClass, MultiClassOneVsAll,"
                " QueryRMSE, PairLogit, YetiRank and MultiRMSE"
            )
        if leaf_method == GBDT_LEAF_EXACT:
            # an explicit Exact under Ordered: `set_leaves_estimation_default`'s
            # words, both columns (lane gbdt-cpu-rest-4, 2026-11-23); only the
            # loss's own Exact default falls back to Gradient below
            raise Error(
                "Exact leaf estimation method don't work with ordered boosting"
                " on GPU (catboost_options.cpp:346-350)"
            )
        hloss = _resolve_pointwise_loss(
            pw_objective, loss,
            Float32(params.f(11)), Float32(params.f(12)),
            Float32(params.f(13)), Float32(params.f(14)),
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
    if n_flags != 0 and ctr_permutation_id != -1 and ctr_permutation_id != perm_count - 1:
        raise Error(
            "ctr_estimation_permutation_id " + String(ctr_permutation_id)
            + " with boosting_type='Ordered': an Ordered fit estimates on"
            " its last permutation (" + String(perm_count - 1)
            + "); leave it unset"
        )
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
    var x = read_f32(x_address, n_rows * n_features)
    var y = read_f32(y_address, n_rows)
    # `train`'s weight column: empty (the unit weight) without sample or
    # class weights; a negative sample weight refused at its first row, in
    # `train`'s words (`gbdt_host_binary_row_weights`)
    var row_weights = List[Float32]()
    if n_weights != 0 or n_class_weights != 0:
        var class_weights = List[Float32]()
        for i in range(n_class_weights):  # small-loop(class weight words of params)
            class_weights.append(Float32(params.f(36 + i)))
        var sample_weights = List[Float32]()
        if n_weights != 0:
            sample_weights = read_f32(params.weights, n_weights)
        row_weights = gbdt_host_binary_row_weights(
            y, n_rows, class_weights, sample_weights
        )
    # THE WEIGHTED L2 (lane w3f2-gbdtw, 2026-11-24): `train`'s
    # `device_weighted_l2` over the same column, once per fit; the
    # unweighted fit keeps it exactly
    if loss != String("PairLogit"):
        p.l2_leaf_reg = host_weighted_l2(p.l2_leaf_reg, row_weights, n_rows)
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
    var flags = List[UInt32]()
    if n_flags != 0:
        flags = gbdt_host_load_u32(u32_ptr(flags_address), n_flags)
    # feature combinations (`max_ctr_complexity` above 1) are grown by the
    # Ordered fit's level driver, as on the device
    var fit = gbdt_ordered_host_fit(
        x, y, n_rows, n_features, p, hloss, opts, ev, flags,
        max_ctr_complexity, row_weights,
    )
    text = fit.text
    losses = fit.losses.copy()
    test_losses = fit.test_losses.copy()
    best = fit.best_iteration
    stopped = fit.stopped_early
    return _fit_outcome(text^, best, stopped, losses^, test_losses^, fit.fit_loop)


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


def gbdt_fit_native_run(params: GbdtFitArgs) raises -> GbdtFitOutcome:
    """`gbdt_fit_binding` (`bindings/_mojolearn_gbdt.mojo:169-407`): the same
    slot checks in the same words, then the refusals of what the host fit
    does not restate, then the host fit and the model text. Returns
    `[model_text, best_iteration, stopped_early, learn_losses,
    test_losses]`."""
    if len(params) < 36:
        raise Error(
            "gbdt_fit: params must hold at least 36 values, got "
            + String(len(params))
        )
    var n_class_weights = params.i(35)
    if n_class_weights < 0:
        raise Error(
            "gbdt_fit: n_class_weights must not be negative, got "
            + String(n_class_weights)
        )
    var fixed_and_weights = 36 + n_class_weights
    if len(params) != fixed_and_weights and len(params) != fixed_and_weights + 1 and len(params) != fixed_and_weights + 2 and len(params) != fixed_and_weights + 3 and len(params) != fixed_and_weights + 4 and len(params) != fixed_and_weights + 5 and len(params) != fixed_and_weights + 7 and len(params) != fixed_and_weights + 8 and len(params) != fixed_and_weights + 10:
        raise Error(
            "gbdt_fit: params must hold 36 + n_class_weights, optionally min_split_gain, min_child_hessian, then feature_fraction, then the group sizes address and group count, then the pairs address, pair count and pair weights address values ("
            + String(36 + n_class_weights)
            + ") values, got "
            + String(len(params))
        )
    if len(params.strs) != 4 and len(params.strs) != 5 and len(params.strs) != 8:
        raise Error(
            "gbdt_fit: strs must hold [loss, bootstrap_type, od_type,"
            " nan_mode], optionally feature_border_type, optionally then"
            " boosting_type, the fold_len_multiplier's float64 bits and"
            " fold_permutation_block, got " + String(len(params.strs))
        )
    # `feature_border_type`, the optional fifth string (lane/catboost-parity);
    # the seven types are the device fit's own host functions
    # (`calc_quantization` / `select_borders`), restated nowhere
    var border_type = BORDER_TYPE_GREEDY_LOG_SUM
    if len(params.strs) >= 5:
        border_type = border_type_from_name(params.strs[4])
    # the Ordered tail (lane/catboost-parity): its own arm, before any of
    # the Plain arms' refusals
    if len(params.strs) == 8 and params.strs[5] == String("Ordered"):
        return _gbdt_fit_ordered_arm(
            params.x, params.y, params.cat_flags,
            params.eval_x, params.eval_y, params,
            border_type,
        )
    if len(params.strs) == 8 and params.strs[5] != String("Plain"):
        raise Error(
            "gbdt_fit: boosting_type must be 'Plain' or 'Ordered', got '"
            + params.strs[5] + "'"
        )
    var xp = f32_ptr(params.x)
    var yp = f32_ptr(params.y)
    var weights_address = params.weights
    _ = f32_ptr(weights_address)
    var cp = u32_ptr(params.cat_flags)
    _ = f32_ptr(params.eval_x)
    _ = f32_ptr(params.eval_y)

    var n_rows = params.i(0)
    var n_features = params.i(1)
    var n_weights = params.i(2)
    var n_flags = params.i(3)
    var class_weights = List[Float32]()
    for i in range(n_class_weights):  # small-loop(class weight words of params)
        class_weights.append(Float32(params.f(36 + i)))
    var grow_code = params.i(31)
    if grow_code != 0 and grow_code != 1 and grow_code != 2:
        raise Error(
            "gbdt_fit: grow_policy code must be 0 (SymmetricTree), 1"
            " (Depthwise) or 2 (Lossguide), got " + String(grow_code)
        )
    # lane gbdt-cpu-rest-4-ns (2026-11-23): not a CPU gap; `train` refuses
    # it first on every column, through the one body (lane
    # gbdt-pw-multiclass: `pointwise_policy_check`)
    pointwise_policy_check(params.i(26) != 0, grow_code)
    var min_split_gain = Float64(-1)
    if len(params) >= fixed_and_weights + 1:
        min_split_gain = params.f(fixed_and_weights)
    var min_child_hessian = Float64(-1)
    if len(params) >= fixed_and_weights + 2:
        min_child_hessian = params.f(fixed_and_weights + 1)
    var feature_fraction = Float64(1)
    if len(params) >= fixed_and_weights + 3:
        feature_fraction = params.f(fixed_and_weights + 2)
    # MultiRMSE's target dimension, the one value of the +4 tail
    # (`bindings/_mojolearn_gbdt.mojo`'s), in its words
    var target_dim = 1
    if len(params) == fixed_and_weights + 4:
        target_dim = params.i(fixed_and_weights + 3)
        if target_dim < 2:
            raise Error(
                "gbdt_fit: the target-dimension tail needs a dimension >= 2,"
                " got " + String(target_dim)
            )
    # the pool's grouping (`bindings/_mojolearn_gbdt.mojo`'s group tail),
    # checked, refused and resolved in the words and by the rule
    # `gbdt/train.mojo::train` uses, so the CPU column refuses exactly where
    # the device column does and QueryRMSE sees the same queries
    var eval_group_sizes = List[Int]()
    if len(params) == fixed_and_weights + 7 or len(params) == fixed_and_weights + 10:
        var count = params.i(len(params) - 1)
        if count < 1 or count > params.i(20) or params.i(len(params) - 2) == 0:
            raise Error("gbdt_fit: invalid held-out query count or group buffer")
        if params.strs[0] != "QueryRMSE" and params.strs[0] != "PairLogit" and params.strs[0] != "YetiRank":
            raise Error("gbdt_fit: only grouped ranking objectives take eval_group_id")
        if (len(params) == fixed_and_weights + 10) != (params.strs[0] == "PairLogit"):
            raise Error("gbdt_fit: invalid held-out group and training pair layout")
        eval_group_sizes = gbdt_host_group_sizes(u32_ptr(params.i(len(params) - 2)), count, params.i(20))
    var tail_loss = params.strs[0]
    var tail_rows = params.i(0)
    var host_group_sizes = List[Int]()
    if len(params) >= fixed_and_weights + 5:
        var n_groups = params.i(fixed_and_weights + 4)
        if n_groups < 1:
            raise Error(
                "gbdt_fit: the group tail needs a positive group count, got "
                + String(n_groups)
            )
        var gp = u32_ptr(params.i(fixed_and_weights + 3))
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
    if len(params) == fixed_and_weights + 8 or len(params) == fixed_and_weights + 10:
        var n_pairs = params.i(fixed_and_weights + 6)
        if n_pairs != -1 and n_pairs < 1:
            raise Error(
                "gbdt_fit: the pairs tail needs a positive pair count or -1,"
                " got " + String(n_pairs)
            )
        if n_pairs > 0:
            var pp = u32_ptr(params.i(fixed_and_weights + 5))
            var pw = f32_ptr(params.i(fixed_and_weights + 7))
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

    var border_count = params.i(4)
    var n_estimators = params.i(5)
    var max_depth = params.i(6)
    var learning_rate = Float32(params.f(7))
    var l2_leaf_reg = Float32(params.f(8))
    var random_seed = UInt64(params.i(9))
    var score_function = params.i(10)
    var loss = params.strs[0]
    if loss == String("YetiRank") and l2_leaf_reg == Float32(0.0):
        # `catboost_options.cpp:357-359`, as `gbdt/train.mojo::train` applies it
        l2_leaf_reg = Float32(1e-20)
    var loss_border = Float32(params.f(15))
    var leaf_iterations = params.i(16)
    var leaf_method = params.i(17)
    var bootstrap_type = params.strs[1]
    var n_eval_rows = params.i(20)
    var od_type = params.strs[2]
    var od_pvalue = params.f(21)
    var od_wait = params.i(22)
    var use_best_model = params.i(23)
    var best_model_min_trees = params.i(24)
    var nan_mode_name = params.strs[3]
    var random_strength = Float32(params.f(25))
    var use_pointwise = params.i(26) != 0
    var border_build_max_samples = params.i(27)
    var permutation_count = params.i(28)
    var ctr_permutation_id = params.i(29)
    var boost_from_average = params.i(30)
    var max_leaves = params.i(32)
    var min_data_in_leaf = params.i(33)
    # feature combinations (tree CTRs); read by the CTR categorical arm
    # only, as on the device, where a fit without CTR columns builds none
    var max_ctr_complexity = params.i(34)
    if max_ctr_complexity < 1:
        # `TCatFeatureParams.check`'s words (`train` raises them)
        raise Error(
            "max_ctr_complexity must be at least 1; got "
            + String(max_ctr_complexity)
        )

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
        # lane gbdt-cpu-rest-4 (2026-11-23): `objective_from_name`'s words
        # (gbdt/targets/kernel/pointwise_targets.mojo, a GPU module, so the
        # sentence is carried here), both columns
        raise Error(
            "unknown loss '" + loss + "': this implementation trains RMSE, Logloss,"
            " CrossEntropy, Quantile, MAE, LogLinQuantile, MAPE, Poisson, Lq,"
            " Expectile, Tweedie, Huber, MultiClass, MultiClassOneVsAll,"
            " QueryRMSE, PairLogit, YetiRank and MultiRMSE"
        )
    # lane gbdt-cpu-rest-4-pwl (2026-11-23): the pointwise searcher on a
    # one-output (or a multi-output) loss takes its own arm BEFORE the
    # greedy arms' refusals, which are not its refusals: the device's own
    # checks in the device's order are the arm's (`_gbdt_fit_pointwise_arm`,
    # gbdt/host/gbdt_oracle_pointwise.mojo); the grouped losses keep the
    # grouped pointwise searcher below
    var pw_grouped_loss = (
        pw_objective == GBDT_OBJ_QUERY_RMSE
        or pw_objective == GBDT_OBJ_PAIR_LOGIT
        or pw_objective == GBDT_OBJ_YETI_RANK
    )
    # lane gbdt-cpu-rest-4-simple (2026-11-23): Simple's admission, the
    # device's (`train` calls the same two bodies right after
    # `set_leaves_estimation_default`): one iteration, then the greedy
    # SymmetricTree searcher's leaves on the one-output losses only. YetiRank
    # is refused first on both columns ("changing the leaf_estimation_method
    # parameter is prohibited", `_resolve_pointwise_loss`), so it skips here.
    if loss != String("YetiRank"):
        simple_leaf_iterations_check(
            leaf_method, leaf_iterations if leaf_iterations >= 0 else 1
        )
        simple_leaves_check(
            leaf_method, is_multi, False, grow_code != 0, use_pointwise
        )
    # lane gbdt-pw-multiclass (2026-11-23): MultiClass and MultiClassOneVsAll
    # on the pointwise searcher take the multi-output fit below (the greedy
    # arm's checks are `train`'s and hold for both searchers), whose search
    # sums every approx dimension's score
    var pw_multi_class = (
        loss == String("MultiClass") or loss == String("MultiClassOneVsAll")
        or loss == String("MultiRMSE")
    )
    if use_pointwise and not pw_grouped_loss and not pw_multi_class:
        return _gbdt_fit_pointwise_arm(
            params, border_type, feature_fraction, min_split_gain,
            min_child_hessian,
        )
    if is_multi and grow_code != 0:
        raise Error(
            "Error: optimization scheme is not supported for GPU learning"
            " Loss=MultiClass;OptimizationScheme="
            + String("Depthwise" if grow_code == 1 else "Lossguide")
            + " (their TGpuTrainerFactory has no non-symmetric"
            " multiclass trainer, multiclass.cpp:5-14, train.cpp:279)"
        )
    # lane gbdt-cpu-rest-4-sym (2026-11-23): Exact leaves on the
    # multi-output losses refuse on both columns in `fit_with_test`'s words
    # (MultiRMSE's, then the exact estimator's for the two multiclass
    # losses, `compute_exact_approx`)
    if is_multi_rmse and leaf_method == GBDT_LEAF_EXACT:
        raise Error(
            "Exact leaves estimation is one-dimensional"
            " (ComputeExactValue, permutation_der_calcer.h:98-110);"
            " MultiRMSE does not reach it"
        )
    if is_multi and leaf_method == GBDT_LEAF_EXACT:
        raise Error("Only MAPE, MAE and Quantile are supported for Exact leaves estimation on GPU")
    # lane gbdt-cpu-rest-4-tail (2026-11-23): the multi-output losses walk
    # their leaves under Newton and Gradient at any iteration count
    # (gbdt_oracle_multiclass.mojo `_estimate_multi`, the device's
    # `device_multi_walk_estimate`); Simple and unknown codes refuse below
    # in the device walker's words, so the two by-name refusals are gone
    # lane gbdt-cpu-rest (2026-11-23): the pointwise losses grow
    # Depthwise and Lossguide trees through the non-symmetric driver
    # (gbdt_oracle_depthwise.mojo); the grouped ones are SymmetricTree only
    # on both columns, in `train`'s words
    var pw_grouped = (
        pw_objective == GBDT_OBJ_QUERY_RMSE
        or pw_objective == GBDT_OBJ_PAIR_LOGIT
        or pw_objective == GBDT_OBJ_YETI_RANK
    )
    if pw_grouped and grow_code != 0:
        raise Error(
            "loss='" + loss + "' is implemented on SymmetricTree only here;"
            " grow_policy=" + String("Depthwise" if grow_code == 1 else "Lossguide")
            + " is refused (the reference registers it,"
            " querywise_non_symmetric.cpp:5-14)"
        )
    # lane gbdt-cpu-rest-4-ns / -sym (2026-11-23): RMSE leaves at any
    # iteration count on every greedy searcher (the non-symmetric estimator,
    # the symmetric loop's walker); the pointwise searchers took their arms
    # above
    # lane gbdt-cpu-rest-4-pwl (2026-11-23): every feature_border_type on
    # both pointwise searchers (their grids take `params.border_type`, the
    # device's own border search), so the by-name refusal is gone
    # CPU binding compilation passed on Apple M1 at O0 (2026-10-28);
    # runtime/identity remain pending. Existing ranking CPU derivatives are
    # shared; this owner performs the distinct doc-parallel tree search.
    var ranking_pointwise = use_pointwise and (
        pw_objective == GBDT_OBJ_QUERY_RMSE
        or pw_objective == GBDT_OBJ_PAIR_LOGIT
        or pw_objective == GBDT_OBJ_YETI_RANK
    )
    # The grouped pointwise search's leaf method, admitted where the device
    # fit admits it, in its words (gbdt/options/catboost_options.mojo).
    querywise_pointwise_leaf_check(use_pointwise, ranking_pointwise, GBDT_LEAF_NEWTON if leaf_method == -1 else leaf_method)
    # (lane gbdt-cpu-rest-4-ns, 2026-11-23: `lossguide_knobs`, the one
    # Lossguide Logloss configuration whose Gradient leaves were admitted,
    # is gone: both policies admit every leaf method and score below)
    # the symmetric Logloss, RMSE and multiclass fits' stochastic arm, CatBoost's GPU
    # defaults (Bayesian bootstrap, random_strength 1):
    # gbdt_oracle.mojo::gbdt_host_boost, which the numeric, one-hot and CTR
    # arms all run (lane pv-gbdt-cpu-cat, 2026-11-22: categorical columns
    # no longer refuse it; the multiclass arm refuses them below)
    var symmetric_stochastic = (
        grow_code == 0 and (loss == String("Logloss") or is_rmse or is_multi)
        and not is_pointwise
    )
    if grow_code != 0:
        # lane gbdt-cpu-rest-4-ns (2026-11-23): both policies score every
        # calcer the device's leafwise kernels have (Cosine, NewtonCosine,
        # L2, NewtonL2; leafwise_score_body.mojo, the kernels' own
        # statements); the rest in the device searcher's words
        # (`structure_searcher_options.check`), so not a CPU gap
        if score_function != GBDT_HOST_SCORE_NEWTON_L2 and score_function != GBDT_HOST_SCORE_NEWTON_COSINE and score_function != GBDT_HOST_SCORE_COSINE and score_function != GBDT_HOST_SCORE_L2:
            raise Error(
                "score_function=" + String(score_function)
                + " has no calcer in this implementation; only Cosine, NewtonCosine,"
                " L2 and NewtonL2 are implemented (SolarL2, SatL2 and LOOL2 are"
                " theirs at compute_scores.cu:509-546 and are not written)"
            )
    elif ranking_pointwise:
        if (
            score_function != GBDT_HOST_SCORE_COSINE
            and score_function != GBDT_HOST_SCORE_NEWTON_COSINE
            and score_function != GBDT_HOST_SCORE_L2
            and score_function != GBDT_HOST_SCORE_NEWTON_L2
        ):
            # the device kernel's words, both columns (lane gbdt-cpu-rest-4)
            raise Error(
                "score_function has no calcer here; only Cosine, NewtonCosine,"
                " L2 and NewtonL2 are implemented"
            )
    elif grow_code == 0:
        # lane gbdt-cpu-rest-4-sym (2026-11-23): the greedy SymmetricTree
        # searcher scores Cosine, NewtonCosine, L2 and NewtonL2 on every
        # loss (`gbdt_oracle.mojo::_greedy_gains`, the device's
        # `enqueue_score`; the Newton scores read the second-order search
        # planes); the multi-output losses refuse the Newton scores in
        # `fit_with_test`'s words. Any other code is refused in the device
        # kernel's words.
        if (
            score_function != GBDT_HOST_SCORE_COSINE
            and score_function != GBDT_HOST_SCORE_NEWTON_COSINE
            and score_function != GBDT_HOST_SCORE_L2
            and score_function != GBDT_HOST_SCORE_NEWTON_L2
        ):
            raise Error(
                "score_function has no calcer here; only Cosine, NewtonCosine,"
                " L2 and NewtonL2 are implemented"
            )
        if is_multi and (
            score_function == GBDT_HOST_SCORE_NEWTON_COSINE
            or score_function == GBDT_HOST_SCORE_NEWTON_L2
        ):
            raise Error(
                "MultiClass loss doesn't support second derivatives in tree"
                " structure search currently (their CB_ENSURE,"
                " `multiclass_targets.cpp:27`)"
            )
    # lane gbdt-cpu-rest-4-ns (2026-11-23): Depthwise's min_split_gain,
    # min_child_hessian and min_data_in_leaf are the one non-symmetric
    # driver's (gbdt_oracle_depthwise.mojo `_grow_non_symmetric_tree`: the
    # split-gain threshold, the child-Hessian rejection and terminal leaf,
    # the leaf-size test), as the device's `fit_non_symmetric_tree` runs
    # them under both policies; their checks are `train`'s below
    if grow_code != 0:
        # the non-symmetric estimator (`_estimate_and_apply` on every tree):
        # Newton and Gradient walk, Exact is the quantile family's only;
        # Simple was refused above (`simple_leaves_check`), so only an
        # unknown code reaches the second refusal
        if leaf_method == GBDT_LEAF_EXACT and not is_pointwise:
            raise Error("Only MAPE, MAE and Quantile are supported for Exact leaves estimation on GPU")
        if leaf_method != -1 and leaf_method != GBDT_LEAF_GRADIENT and leaf_method != GBDT_LEAF_NEWTON and leaf_method != GBDT_LEAF_EXACT:
            raise Error(
                "WriteSecondDerivatives: only Newton and Gradient reach"
                " the walker; Exact replaces it"
            )
        # `train`'s bootstrap resolution, its words for the two it refuses
        if bootstrap_type == String("MVS"):
            raise Error(
                "MVS is not reachable from their GPU oblivious searcher"
                " (weak_objective_impl.h:30 asserts it away)"
            )
        if (
            bootstrap_type != String("") and bootstrap_type != String("No")
            and bootstrap_type != String("Bayesian")
            and bootstrap_type != String("Bernoulli")
            and bootstrap_type != String("Poisson")
        ):
            raise Error(
                "unknown bootstrap_type '" + bootstrap_type
                + "': Bayesian, Bernoulli, Poisson, No"
            )
    # lane gbdt-cpu-rest-4-sym (2026-11-23): Gradient leaves on the greedy
    # SymmetricTree Logloss and RMSE fits (`gbdt_host_boost`'s walker,
    # `_second_derivatives`); Exact there refuses on both columns in the
    # exact estimator's words (`compute_exact_approx`); an unknown code
    # never reaches the walker, the device's words as above
    # lane gbdt-cpu-rest-4-simple (2026-11-23): Simple trains here, the
    # searcher's own leaves (`gbdt_host_boost`, `gbdt_losses_host_fit`; the
    # multi-output losses were refused above), so it is no longer refused
    var sym_logloss_rmse = grow_code == 0 and not is_pointwise and not is_multi
    if sym_logloss_rmse and leaf_method == GBDT_LEAF_EXACT:
        raise Error("Only MAPE, MAE and Quantile are supported for Exact leaves estimation on GPU")
    if grow_code == 0 and leaf_method != -1 and leaf_method != GBDT_LEAF_GRADIENT and leaf_method != GBDT_LEAF_NEWTON and leaf_method != GBDT_LEAF_EXACT and leaf_method != GBDT_LEAF_SIMPLE:
        raise Error(
            "WriteSecondDerivatives: only Newton and Gradient reach"
            " the walker; Exact replaces it"
        )
    # lane gbdt-cpu-rest-4-tail (2026-11-23): Gradient leaves on the
    # multi-output losses too (the multi walk), so every code that passes
    # the walker check above trains and the by-name refusal is gone
    # lane gbdt-cpu-rest-4-sym (2026-11-23): under SymmetricTree every
    # loss draws Bayesian, Bernoulli and Poisson; MVS and an unknown name
    # are refused on both columns in `train`'s words
    if grow_code == 0 and bootstrap_type == String("MVS"):
        raise Error(
            "MVS is not reachable from their GPU oblivious searcher"
            " (weak_objective_impl.h:30 asserts it away)"
        )
    if grow_code == 0 and bootstrap_type != String("") and bootstrap_type != String("No") and bootstrap_type != String("Bayesian") and bootstrap_type != String("Bernoulli") and bootstrap_type != String("Poisson"):
        raise Error(
            "unknown bootstrap_type '" + bootstrap_type
            + "': Bayesian, Bernoulli, Poisson, No"
        )
    # lane gbdt-cpu-rest-4 (2026-11-23): every policy and loss that reaches
    # here draws Bayesian, Bernoulli and Poisson (the symmetric loop, the
    # non-symmetric driver and the losses fit), so the by-name bootstrap
    # refusal that followed had no case left and is gone
    # Plain numeric Logloss already threads row weights through search,
    # leaf estimation and learn loss. Sample weights use that same owner;
    # class weights multiply them as in train.mojo.
    var binary_sample_weights = (
        n_weights != 0 and loss == String("Logloss")
    )
    # the device's one `weights` column for every other numeric fit (lane
    # gbdt-gaps, 2026-11-23): RMSE on both tree shapes, Logloss on
    # Depthwise and Lossguide under every leaf method and bootstrap, the
    # pointwise losses and YetiRank / PairLogit on the greedy symmetric
    # searcher (gbdt/host/gbdt_oracle_losses.mojo threads them through the
    # search pass, the leaf estimate and the learn loss). Refused still:
    # categorical columns (the CTR tables' weighted counts are not
    # restated), the multi-output losses, QueryRMSE (its weighted query
    # means), the pointwise searcher's grouped arm, and Exact leaves (the
    # weighted quantile), each by name below or in its owner.
    # lane gbdt-cpu-rest (2026-11-23): QueryRMSE's weighted query means
    # (gbdt/host/gbdt_oracle_query.mojo, the device's bodies in
    # gbdt/targets/query_bodies.mojo) and the grouped pointwise searcher
    # (gbdt_oracle_pointwise_rank.mojo) read them as the greedy grouped fit
    # and the multi-output losses (gbdt_oracle_multiclass.mojo: the
    # sample weight times the class weight, `train`'s one product)
    # and categorical columns (the one-hot and CTR arms run the same boost
    # loop with the same row weights; the device builds its CTR tables over
    # trivial weights, so the tables do not read them)
    var sample_weights_ok = n_weights != 0 and (
        loss == String("Logloss") or is_rmse or is_pointwise or is_multi
    )
    # (lane gbdt-cpu-rest-4, 2026-11-23: the by-name refusal here had no case
    # left: every loss that reaches here reads sample weights)
    # lane gbdt-cpu-rest-4-ns (2026-11-23): on Depthwise and Lossguide the
    # pointwise losses read class weights as `train` folds them for every
    # loss but RMSE and the grouped ones (two slots, `y > 0.5`,
    # class_weight_body.mojo through gbdt_host_binary_row_weights); RMSE is
    # refused there in `train`'s words
    if n_class_weights != 0 and grow_code != 0 and is_rmse:
        raise Error(
            "class weights take effect only with a classification loss,"
            " their option check's words (catboost_options.cpp:617)"
        )
    var binary_class_weights = n_class_weights != 0 and (
        loss == String("Logloss")
        or (grow_code != 0 and is_pointwise and not pw_grouped)
    )
    # lane gbdt-cpu-rest-4-sym (2026-11-23): under SymmetricTree the class
    # weights fold into the one weight column for the pointwise losses too
    # (`train`'s `fold_class_weights` at the binarized target `y > 0.5`,
    # the same `class_weight_body` the host column calls); RMSE, the grouped
    # losses and MultiRMSE refuse them on both columns in `train`'s words
    if grow_code == 0 and n_class_weights != 0:
        if is_multi_rmse:
            raise Error("class_weights do not apply to loss='MultiRMSE'")
        if pw_grouped:
            raise Error("class_weights do not apply to loss='" + loss + "'")
        if is_rmse:
            raise Error(
                "class weights take effect only with a classification loss,"
                " their option check's words (catboost_options.cpp:617)"
            )
        if is_pointwise:
            binary_class_weights = True
    # (lane gbdt-cpu-rest-4, 2026-11-23: the by-name refusal here had no case
    # left: RMSE, MultiRMSE and the grouped losses refuse above in `train`'s
    # words; every other loss folds them)
    if binary_class_weights and n_class_weights != 2:
        # `train`'s words (gbdt/train.mojo)
        raise Error(
            "class_weights takes 2 entries for loss '" + loss + "', got "
            + String(n_class_weights)
        )
    # lane gbdt-cpu-rest-4-cat (2026-11-23): what `train` refuses with a
    # categorical or one-hot column, in its words and its order (the
    # MultiRMSE check, then the grouped losses' cat_features loop, then
    # their one_hot loop); every other loss trains them below
    var flags = gbdt_host_load_u32(cp, n_flags)
    var any_cat_flag = False
    var any_one_hot_flag = False
    for f in range(n_flags):  # small-loop(n_flags: per-feature option flags): option validation, reads no data
        if (flags[f] & UInt32(1)) != UInt32(0):
            any_cat_flag = True
        if (flags[f] & UInt32(2)) != UInt32(0):
            any_one_hot_flag = True
    if is_multi_rmse and any_cat_flag:
        raise Error(
            "loss='MultiRMSE' with cat_features is not carried here:"
            " the CTR target binarization reads one target per row"
        )
    # lane gbdt-cpu-rest-4-cat (2026-11-23): every other categorical or
    # one-hot fit trains on the host now, through the one categorical
    # prelude (gbdt_oracle_ctr.mojo::gbdt_ctr_host_prelude): SymmetricTree
    # with the pointwise losses (gbdt_oracle_losses.mojo) and with
    # MultiClass, MultiClassOneVsAll and MultiRMSE's one-hot columns
    # (gbdt_oracle_multiclass.mojo), every permutation and the tree CTRs;
    # Depthwise and Lossguide with Logloss, RMSE and the pointwise losses
    # (gbdt_oracle_depthwise.mojo, sub-lane gbdt-cpu-rest-4-cat-ns). The
    # by-name refusal is deleted.
    # the HELD-OUT arm (lane/close-no-cpu-path-gbdt, 2026-09-20): their
    # `testCursor`, held-out curve, overfitting detector and
    # `ShrinkToBestIteration`, restated in gbdt/host/gbdt_oracle_eval.mojo
    # for the SymmetricTree Logloss fit of gbdt/host/gbdt_oracle.mojo and
    # for that fit ONLY. Every other Plain arm still refuses by name: their
    # test cursor runs the same four pieces over a different loss kernel and
    # a different model shape, and neither is restated.
    # lane gbdt-gaps (2026-11-23): the same arm on the RMSE fit of that loop
    # (numeric, one-hot and CTR columns) and on the non-symmetric driver
    # (gbdt_oracle_depthwise.mojo, Logloss and RMSE)
    # lane gbdt-cpu-rest (2026-11-23): and the pointwise losses on the
    # non-symmetric driver, whose held-out arm reads `_loss_value`
    # lane gbdt-cpu-rest-2 (2026-11-23): and the symmetric pointwise losses
    # (gbdt_oracle_losses.mojo::gbdt_losses_host_fit, the held-out arm of
    # gbdt_host_boost); the grouped losses refuse an eval set on both
    # columns, in `train`'s words
    if is_multi_rmse and n_eval_rows != 0:
        raise Error(
            "loss='MultiRMSE' with eval_set is not carried here: the"
            " held-out arm's target is one value per row"
        )
    if pw_grouped and n_eval_rows != 0 and len(eval_group_sizes) == 0:
        raise Error("grouped validation needs one held-out query id per eval row")
    # (lane gbdt-cpu-rest-4, 2026-11-23: the by-name refusal here had no case
    # left: an eval set on MultiRMSE and the grouped losses refuses above in
    # `train`'s words; every other loss has its held-out arm)
    var od = load_overfitting_detector_options(od_type, od_pvalue, od_wait)
    var od_kind = od.od_type
    # (lane gbdt-cpu-rest-4, 2026-11-23: the by-name refusal here had no case
    # left: the detector needs an eval set (refused below without one) and every
    # fit that takes one has its held-out arm)
    if od_kind != OD_NONE and n_eval_rows == 0:
        raise Error(
            # `train`'s words (gbdt/train.mojo), both columns
            "od_type='" + od_type + "' needs an eval set: pass"
            " eval_x_colmajor and eval_y. Stopping on the learn loss"
            " would stop on a curve that falls by construction."
        )
    # (lane gbdt-cpu-rest-4, 2026-11-23: the by-name random_strength
    # refusal had no case left: every loss and policy that reaches here
    # carries the score noise)
    # the quantile family's constant (`sample_quantile.mojo`), shared with
    # the device fit, lane/catboost-parity
    var quantile_family = (
        pw_objective == GBDT_OBJ_MAE
        or pw_objective == GBDT_OBJ_QUANTILE
        or pw_objective == GBDT_OBJ_MAPE
    )
    # lane gbdt-cpu-rest-4-ns and gbdt-cpu-rest-4-sym (2026-11-23):
    # boost_from_average=True trains Logloss and CrossEntropy as well on
    # every policy (their log-odds start,
    # `calc_one_dimensional_optimum_const_approx`, the host arm of the
    # device's `optimum_const_approx_device`); every other loss refuses on
    # both columns in `train`'s words
    if boost_from_average == 1:
        if is_multi_rmse:
            raise Error(
                "boost_from_average=True with loss MultiRMSE is not carried here:"
                " its StartingPoint is per-dimension"
                " (optimal_const_for_loss.h:230-239) and this model's bias is one"
                " value"
            )
        if not (
            is_rmse or quantile_family or loss == String("Logloss")
            or pw_objective == GBDT_OBJ_CROSSENTROPY
        ):
            raise Error(
                "You can use boost_from_average only for these loss"
                " functions now: RMSE, Logloss, CrossEntropy, Quantile, MAE,"
                " MAPE (catboost_options.cpp:705-709; their MultiQuantile"
                " and RMSPE are not trained here, and MultiRMSE's"
                " per-dimension start is refused above)."
            )
    # lane gbdt-cpu-rest (2026-11-23): the non-symmetric driver samples
    # each tree's features for every loss and both policies
    # (`_sample_tree_folds`, the device's `feature_sampling.mojo`); the
    # symmetric loop does not restate the sampling
    if feature_fraction < 1.0 and any_cat_flag:
        # `train`'s words (gbdt/train.mojo), its cat_features loop first
        raise Error("feature_fraction<1 supports numeric features only; categorical/CTR input refused")
    if feature_fraction < 1.0 and any_one_hot_flag:
        # and its one_hot loop (lane gbdt-cpu-rest-4-cat: this was the
        # categorical sentence for a one-hot column)
        raise Error("feature_fraction<1 supports numeric features only; one_hot input refused")
    # and the numeric symmetric Logloss / RMSE loop (gbdt_host_boost, the
    # same two helpers), and the symmetric pointwise losses, grouped ones
    # included (gbdt_losses_host_fit, the same statements; lane
    # gbdt-cpu-rest-2)
    # lane gbdt-rest-3 (2026-11-23): and the grouped pointwise searcher
    # (gbdt_oracle_pointwise_rank.mojo `_pw_sample_tree`)
    if not (feature_fraction > 0.0) or feature_fraction > 1.0:
        raise Error("feature_fraction must be finite and in (0, 1]")
    # lane gbdt-cpu-rest-4-tail (2026-11-23): `train`'s limits in its
    # words, one body (`check_tree_limits`); a negative depth keeps the
    # searcher's own refusal (`create_subsets`)
    check_tree_limits(border_count, max_depth, grow_code == GBDT_HOST_GROW_LOSSGUIDE)
    if max_depth < 0 and grow_code == 0:
        raise Error(
            String("maxDepth must be non-negative, got ") + String(max_depth)
        )

    # ---- `train`'s own raises on the rest (`gbdt/train.mojo:944-1149`,
    # `:1580-1605`, `:1733-1758`) ----
    # `child_hessian_threshold` and `check_child_hessian_objective`
    # (`gbdt/options/child_hessian.mojo`), the device's own checks and
    # round-up, called (lane gbdt-cpu-rest-4-ns, 2026-11-23: this was an
    # inline restatement without the objective check, so a pointwise loss
    # outside RMSE / Logloss / CrossEntropy trained here and refused there)
    var child_hessian = child_hessian_threshold(min_child_hessian, grow_code, score_function)
    var ch_objective = -1
    if loss == String("Logloss"):
        ch_objective = GBDT_OBJ_LOGLOSS
    elif is_rmse:
        ch_objective = GBDT_OBJ_RMSE
    elif is_pointwise:
        ch_objective = pw_objective
    check_child_hessian_objective(min_child_hessian, ch_objective)
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
    # (lane gbdt-cpu-rest-4-tail, 2026-11-23: `1 << max_depth` only where
    # the device takes it, so Lossguide's uncapped depth never shifts)
    var ns_max_leaves: Int
    if grow_code != GBDT_HOST_GROW_LOSSGUIDE:
        ns_max_leaves = 1 << max_depth
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
            # `train`'s words (gbdt/train.mojo)
            raise Error(
                "Maximum leaves count for Lossguide grow policy is 65536; got "
                + String(ns_max_leaves)
            )
    # `fit_with_test`'s non-symmetric refusals and the searcher's
    # `check` (`structure_searcher_options.mojo`), in their words, under
    # both policies (lane gbdt-cpu-rest-4-ns, 2026-11-23: Depthwise at
    # depth 0 reached the host fit's own sentence)
    if grow_code != 0 and ns_max_leaves < 2:
        raise Error(
            "max_leaves must be at least 2 under "
            + String("Depthwise" if grow_code == 1 else "Lossguide")
            + ", got " + String(ns_max_leaves)
        )
    if grow_code != 0 and min_data_in_leaf < 0:
        raise Error(
            "min_data_in_leaf must be non-negative, got "
            + String(min_data_in_leaf)
        )
    if grow_code != 0 and max_depth < 1:
        raise Error("max_depth must be at least 1, got " + String(max_depth))
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
    # the greedy SymmetricTree Logloss / RMSE leaf method and its count, as
    # `set_leaves_estimation_default` resolves them (Gradient: 40 for
    # Logloss, 1 for RMSE, unless overridden; lane gbdt-cpu-rest-4-sym)
    var sym_leaf_method = GBDT_LEAF_NEWTON
    if sym_logloss_rmse and leaf_method == GBDT_LEAF_GRADIENT:
        sym_leaf_method = GBDT_LEAF_GRADIENT
        iterations = leaf_iterations if leaf_iterations >= 0 else (1 if is_rmse else 40)
    elif sym_logloss_rmse and leaf_method == GBDT_LEAF_SIMPLE:
        # lane gbdt-cpu-rest-4-simple (2026-11-23): the searcher's own
        # leaves, one iteration (`set_leaves_estimation_default`)
        sym_leaf_method = GBDT_LEAF_SIMPLE
        iterations = 1
    # the multi-output losses' walker, as `set_leaves_estimation_default`
    # resolves it (`get_estimation_method_defaults`: Newton 1 on all three,
    # Gradient 10 on MultiClass / MultiClassOneVsAll and 1 on MultiRMSE),
    # `leaf_estimation_iterations` winning when given (lane
    # gbdt-cpu-rest-4-tail, 2026-11-23)
    var multi_leaf_method = GBDT_LEAF_NEWTON
    var multi_iterations = 1
    if is_multi and leaf_method == GBDT_LEAF_GRADIENT:
        multi_leaf_method = GBDT_LEAF_GRADIENT
        multi_iterations = 1 if is_multi_rmse else 10
    if is_multi and leaf_iterations >= 0:
        multi_iterations = leaf_iterations
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
        # `get_estimation_method_defaults`' Gradient count: 40 for
        # Logloss, 1 for RMSE (lane gbdt-cpu-rest-4-ns, 2026-11-23: RMSE
        # Gradient leaves are admitted on both policies now)
        ns_iterations = leaf_iterations if leaf_iterations >= 0 else (1 if is_rmse else 40)
    var ns_boot = -1
    var ns_boot_param = Float32(0.0)
    if bootstrap_type == String("Bernoulli"):
        ns_boot = GBDT_BOOT_BERNOULLI
        var subsample = Float32(params.f(19))
        ns_boot_param = subsample if subsample >= Float32(0.0) else Float32(0.66)
    elif bootstrap_type == String("Poisson") and grow_code != 0:
        # `train`'s Poisson: `subsample` or DEFAULT_SUBSAMPLE 0.66
        ns_boot = GBDT_BOOT_POISSON
        var subsample_p = Float32(params.f(19))
        ns_boot_param = subsample_p if subsample_p >= Float32(0.0) else Float32(0.66)
    elif bootstrap_type == String("Bayesian") and grow_code != 0:
        if Float32(params.f(19)) >= Float32(0.0):
            raise Error(
                "Error: default bootstrap type (bayesian) doesn't support"
                " 'subsample' option"
            )
        ns_boot = GBDT_BOOT_BAYESIAN
        ns_boot_param = Float32(params.f(18))
    var ns_loss = GbdtHostLoss(
        GBDT_OBJ_RMSE if is_rmse else GBDT_OBJ_LOGLOSS, border, Float32(0.5),
        ns_method, ns_iterations, ns_boot, ns_boot_param, border,
    )
    if is_pointwise:
        pw_loss = _resolve_pointwise_loss(
            pw_objective, loss,
            Float32(params.f(11)), Float32(params.f(12)),
            Float32(params.f(13)), Float32(params.f(14)),
            leaf_method, leaf_iterations, bootstrap_type,
            Float32(params.f(19)), Float32(params.f(18)),
        )
    # the symmetric stochastic arm's bootstrap, `train`'s resolution
    # (`gbdt/train.mojo`), the Ordered arm's words above
    var sym_boot_kind = -1
    var sym_boot_param = Float32(1.0)
    if symmetric_stochastic:
        var sym_subsample = Float32(params.f(19))
        if bootstrap_type == String("Bayesian"):
            if sym_subsample >= Float32(0.0):
                raise Error(
                    "Error: default bootstrap type (bayesian) doesn't support"
                    " 'subsample' option"
                )
            sym_boot_kind = GBDT_BOOT_BAYESIAN
            sym_boot_param = Float32(params.f(18))
        elif bootstrap_type == String("Bernoulli"):
            sym_boot_kind = GBDT_BOOT_BERNOULLI
            sym_boot_param = sym_subsample if sym_subsample >= Float32(0.0) else Float32(0.66)
        elif bootstrap_type == String("Poisson"):
            sym_boot_kind = GBDT_BOOT_POISSON
            sym_boot_param = sym_subsample if sym_subsample >= Float32(0.0) else Float32(0.66)
    var x_address = params.x
    var y_address = params.y
    _ = xp
    _ = yp
    var text = String("")
    var losses = List[Float64]()
    var best_iteration = 0
    var stopped_early = False
    # the fit's boosting loop (`fit_loop_`, lane ab-loop-mode): each arm
    # copies its fit's record beside its best iteration
    var fit_loop = FitLoop.empty()
    # lane gbdt-cpu-rest-2 (2026-11-23): a NaN in a FLOAT column beside
    # categorical ones is binned as on the numeric arms (the one grid and
    # binarizer, nan_mode and the held-out "no NaNs in the learn dataset"
    # refusal inside); a NaN in a cat_features column is refused on both
    # columns by `dense_category_code`'s words ("is not finite"). Lane
    # gbdt-rest-3 (2026-11-23): a NaN in a one_hot_features column that is
    # not categorical is not refused here by name any more: the one-hot
    # scan skips it as the device's does (`gbdt_onehot_max_code`), the
    # column stays AsIs, and the binarize refuses it on both columns in
    # `train`'s words ("There are NaNs in feature number N but there were
    # no NaNs in the learn dataset").
    var eval_x_address = params.eval_x
    var eval_y_address = params.eval_y
    var test_losses = List[Float64]()
    var x = read_f32(x_address, n_rows * n_features)
    # MultiRMSE's `target_dim` dim-major planes; one per row otherwise
    var y = read_f32(y_address, n_rows * target_dim)
    # Read and fold row weights natively before entering the existing
    # weighted search/leaf/loss owners. An empty list keeps the original
    # unweighted arm when neither weighting option was requested.
    var row_weights = List[Float32]()
    if binary_class_weights or binary_sample_weights:
        row_weights = gbdt_host_binary_row_weights(
            y, n_rows, class_weights, read_f32(weights_address, n_weights)
        )
    elif sample_weights_ok:
        # the sample word as given (no class weights apply), negative
        # weights refused at the first such row, as `train` does
        row_weights = gbdt_host_binary_row_weights(
            y, n_rows, List[Float32](), read_f32(weights_address, n_weights)
        )
    # THE WEIGHTED L2 (lane w3f2-gbdtw, 2026-11-24): `train`'s
    # `device_weighted_l2` over the same column, once per fit, before any
    # searcher or leaf reads `p`; the multi-output losses fold their class
    # weights inside `gbdt_multi_host_fit` and scale there; PairLogit's
    # column is its pair sums and keeps `l2`, as on the device
    if not is_multi and loss != String("PairLogit"):
        p.l2_leaf_reg = host_weighted_l2(p.l2_leaf_reg, row_weights, n_rows)
    # the symmetric loop's objective and starting constant
    # (gbdt_oracle.mojo::gbdt_host_boost): RMSE's
    # `AdjustBoostFromAverageDefaultValue` (`gbdt/train.mojo:1597-1600`),
    # unset True, written as the model's `bias`; Logloss starts at zero
    var boost_objective: Int = GBDT_BOOST_LOGLOSS
    if is_rmse:
        boost_objective = GBDT_BOOST_RMSE
    var boost_start = Float64(0.0)
    if is_rmse and grow_code == 0 and boost_from_average != 0:
        boost_start = _rmse_starting_approx(y, n_rows, row_weights)
    elif loss == String("Logloss") and grow_code == 0 and boost_from_average == 1:
        # lane gbdt-cpu-rest-4-sym (2026-11-23): their log-odds start, the
        # host arm of the device's `optimum_const_approx_device` (the
        # weighted mean target, then `-log(1 / x - 1)`), written as the
        # model's `bias` and the start of every cursor
        boost_start = calc_one_dimensional_optimum_const_approx(
            GBDT_OBJ_LOGLOSS, y, row_weights, len(row_weights) > 0
        )
    if is_multi:
        # gbdt/host/gbdt_oracle_multiclass.mojo
        var multi_objective = GBDT_OBJ_MULTICLASS_OVA
        if loss == String("MultiClass"):
            multi_objective = GBDT_OBJ_MULTICLASS
        elif is_multi_rmse:
            multi_objective = GBDT_OBJ_MULTIRMSE
        # the held-out arm and the feature sample (lane gbdt-cpu-rest-2)
        var multi_ev = _gbdt_host_eval_arm(
            eval_x_address, eval_y_address, n_eval_rows, n_features,
            od_kind, od.auto_stop_p_value, od.iterations_wait,
            use_best_model, best_model_min_trees,
        )
        # lane gbdt-cpu-rest-4-cat (2026-11-23): categorical and one-hot
        # columns (`flags`), `train`'s CTR options with them
        var multi_fit = gbdt_multi_host_fit(
            x, y, n_rows, n_features, p,
            multi_objective,
            class_weights, multi_ev^, sym_boot_kind, sym_boot_param,
            random_strength, target_dim, row_weights, feature_fraction,
            score_function,
            flags, permutation_count, ctr_permutation_id, max_ctr_complexity,
            multi_leaf_method, multi_iterations,
            use_pointwise_searcher=use_pointwise,
        )
        if len(flags) != 0:
            text = gbdt_multi_host_model_text_ctr(multi_fit)
        else:
            text = gbdt_multi_host_model_text(multi_fit.model)
        losses = multi_fit.model.losses.copy()
        best_iteration = multi_fit.eval.best_iteration
        fit_loop = multi_fit.fit_loop.copy()
        stopped_early = multi_fit.eval.stopped_early
        test_losses = multi_fit.eval.test_losses.copy()
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
                        y, row_weights, len(row_weights) > 0
                    )
                )
            else:
                pw_start = Float64(
                    calculate_weighted_target_quantile(
                        y, row_weights, len(row_weights) > 0,
                        0.5 if pw_objective == GBDT_OBJ_MAE else Float64(
                            pw_loss.estimator_alpha
                        ),
                        QUANTILE_CONST_DELTA,
                    )
                )
        elif pw_objective == GBDT_OBJ_CROSSENTROPY and boost_from_average == 1:
            # lane gbdt-cpu-rest-4-ns / -sym (2026-11-23): CrossEntropy's
            # log-odds start on every policy, the arm Logloss takes
            pw_start = calc_one_dimensional_optimum_const_approx(
                GBDT_OBJ_CROSSENTROPY, y, row_weights, len(row_weights) > 0
            )
        if grow_code != 0:
            # Depthwise and Lossguide: the non-symmetric driver over the
            # pointwise loss (gbdt/host/gbdt_oracle_depthwise.mojo)
            var pw_tp = GbdtHostTreeParams(
                p, grow_code, ns_max_leaves, Float64(min_data_in_leaf),
                score_function, child_hessian, min_split_gain,
                random_strength, feature_fraction, pw_loss,
            )
            var pw_ev = _gbdt_host_eval_arm(
                eval_x_address, eval_y_address, n_eval_rows, n_features,
                od_kind, od.auto_stop_p_value, od.iterations_wait,
                use_best_model, best_model_min_trees,
            )
            if len(flags) != 0:
                # lane gbdt-cpu-rest-4-cat-ns (2026-11-23): cat_features and
                # one_hot_features columns on the non-symmetric driver, the
                # symmetric categorical fit's prelude and every permutation
                # (gbdt_oracle_depthwise.mojo::gbdt_ctr_host_fit_non_symmetric)
                var pw_cat = gbdt_ctr_host_fit_non_symmetric(
                    x, y, n_rows, n_features, pw_tp, flags, permutation_count,
                    ctr_permutation_id, pw_ev^, max_ctr_complexity, pw_start,
                    row_weights,
                )
                text = gbdt_text_with_bias(
                    gbdt_ctr_host_ns_model_text(pw_cat),
                    len(pw_cat.model.losses), pw_start,
                )
                losses = pw_cat.model.losses.copy()
                best_iteration = pw_cat.model.best_iteration
                fit_loop = pw_cat.model.fit_loop.copy()
                stopped_early = pw_cat.model.stopped_early
                test_losses = pw_cat.model.test_losses.copy()
                return _fit_outcome(text^, best_iteration, stopped_early, losses^, test_losses^, fit_loop)
            var pw_ns = gbdt_host_fit_non_symmetric(
                x, y, n_rows, n_features, pw_tp, pw_ev^, pw_start, row_weights
            )
            text = gbdt_text_with_bias(
                gbdt_host_ns_model_text(pw_ns), len(pw_ns.losses), pw_start
            )
            losses = pw_ns.losses.copy()
            best_iteration = pw_ns.best_iteration
            fit_loop = pw_ns.fit_loop.copy()
            stopped_early = pw_ns.stopped_early
            test_losses = pw_ns.test_losses.copy()
            return _fit_outcome(text^, best_iteration, stopped_early, losses^, test_losses^, fit_loop)
        var pw_sym_ev = _gbdt_host_eval_arm(
            eval_x_address, eval_y_address, n_eval_rows, n_features, od_kind,
            od.auto_stop_p_value, od.iterations_wait, use_best_model, best_model_min_trees,
        )
        var pw_fit: GbdtCtrHostFit
        if ranking_pointwise:
            pw_fit = gbdt_ranking_pointwise_host_fit(
                x, y, n_rows, n_features, p, pw_loss, host_group_sizes,
                host_pair_winners, host_pair_losers, host_pair_weights,
                score_function == GBDT_HOST_SCORE_NEWTON_COSINE or score_function == GBDT_HOST_SCORE_NEWTON_L2,
                score_function == GBDT_HOST_SCORE_COSINE or score_function == GBDT_HOST_SCORE_NEWTON_COSINE,
                random_strength, row_weights, feature_fraction, pw_sym_ev^, eval_group_sizes,
                flags, permutation_count, ctr_permutation_id, max_ctr_complexity,
            )
        else:
            pw_fit = gbdt_losses_host_fit(
                x, y, n_rows, n_features, p, pw_loss, pw_sym_ev^,
                host_group_sizes, host_pair_winners, host_pair_losers, host_pair_weights,
                pw_start, random_strength, row_weights, feature_fraction, score_function,
                flags, permutation_count, ctr_permutation_id, max_ctr_complexity, eval_group_sizes,
            )
        losses = pw_fit.fit.model.losses.copy()
        best_iteration = pw_fit.fit.eval.best_iteration
        fit_loop = pw_fit.fit.model.fit_loop.copy()
        stopped_early = pw_fit.fit.eval.stopped_early
        test_losses = pw_fit.fit.eval.test_losses.copy()
        if len(flags) != 0:
            text = gbdt_text_with_bias(gbdt_ctr_host_model_text(pw_fit), len(losses), pw_start)
        else:
            text = gbdt_rmse_host_model_text(GbdtRmseHostFit(_fit_model(pw_fit.fit), pw_start))
        return _fit_outcome(text^, best_iteration, stopped_early, losses^, test_losses^, fit_loop)
    elif is_rmse and grow_code == 0 and len(flags) == 0:
        # the numeric RMSE fit: the one symmetric loop at GBDT_BOOST_RMSE
        # (gbdt_oracle.mojo::gbdt_host_boost; the twin loop
        # gbdt_rmse_host_fit is deleted, lane pv-gbdt-cpu-cat)
        var ev = _gbdt_host_eval_arm(
            eval_x_address, eval_y_address, n_eval_rows, n_features,
            od_kind, od.auto_stop_p_value, od.iterations_wait,
            use_best_model, best_model_min_trees,
        )
        var r = gbdt_host_fit_eval(
            x, y, n_rows, n_features, p, List[Bool](),
            sym_boot_kind, sym_boot_param, random_strength,
            ev^, row_weights, GBDT_BOOST_RMSE, boost_start, feature_fraction,
            score_function, sym_leaf_method,
        )
        losses = r.model.losses.copy()
        best_iteration = r.eval.best_iteration
        fit_loop = r.model.fit_loop.copy()
        stopped_early = r.eval.stopped_early
        test_losses = r.eval.test_losses.copy()
        text = gbdt_text_with_bias(
            gbdt_host_model_text(r.model), len(r.model.losses), boost_start
        )
    elif grow_code == 0 and len(flags) != 0 and gbdt_has_ctr_columns(
        gbdt_ctr_column_kinds(flags, x, n_rows, n_features)
    ):
        # the CTR categorical arm (gbdt-categorical-ctr-tables):
        # gbdt/host/gbdt_oracle_ctr.mojo, every permutation
        # Full CPU binding compiled at 2d5c5081a83e, 2026-10-28:
        # x86-64-v3/O0, 32 jobs, exit 0. Runtime/identity are pending.
        # Receipt: plan gbdt-cpu-ctr-eval-20261028-compile (git history).
        # The owner expands eval rows with its final learn-only tables
        # and reuses the existing held-out boost/detector/shrink arm.
        var ev = _gbdt_host_eval_arm(
            eval_x_address, eval_y_address, n_eval_rows, n_features,
            od_kind, od.auto_stop_p_value, od.iterations_wait,
            use_best_model, best_model_min_trees,
        )
        var ctr_fit = gbdt_ctr_host_fit(
            x, y, n_rows, n_features, p, flags, permutation_count,
            ctr_permutation_id, ev^, max_ctr_complexity,
            sym_boot_kind, sym_boot_param, random_strength, boost_objective,
            boost_start, row_weights, score_function, sym_leaf_method,
        )
        # the `bias` record a seeded RMSE fit adds (absent at zero bits)
        text = gbdt_text_with_bias(
            gbdt_ctr_host_model_text(ctr_fit),
            len(ctr_fit.fit.model.losses), boost_start,
        )
        losses = ctr_fit.fit.model.losses.copy()
        best_iteration = ctr_fit.fit.eval.best_iteration
        fit_loop = ctr_fit.fit.model.fit_loop.copy()
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
            x, y, n_rows, n_features, p, one_hot, sym_boot_kind,
            sym_boot_param, random_strength, ev^, row_weights,
            boost_objective, boost_start, Float64(1.0), score_function,
            sym_leaf_method,
        )
        text = gbdt_text_with_bias(
            gbdt_host_model_text_one_hot(r.model, one_hot),
            len(r.model.losses), boost_start,
        )
        losses = r.model.losses.copy()
        best_iteration = r.eval.best_iteration
        fit_loop = r.model.fit_loop.copy()
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
            row_weights, GBDT_BOOST_LOGLOSS, boost_start, feature_fraction,
            score_function, sym_leaf_method,
        )
        # the `bias` record a boost_from_average start adds (absent at zero
        # bits, so the unseeded text is unchanged)
        text = gbdt_text_with_bias(
            gbdt_host_model_text(r.model), len(r.model.losses), boost_start
        )
        losses = r.model.losses.copy()
        best_iteration = r.eval.best_iteration
        fit_loop = r.model.fit_loop.copy()
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
            ns_start = _rmse_starting_approx(y, n_rows, row_weights)
        elif boost_from_average == 1:
            # Logloss (lane gbdt-cpu-rest-4-ns, 2026-11-23): the device's
            # own constant, the log-odds of the weighted mean target
            ns_start = calc_one_dimensional_optimum_const_approx(
                GBDT_OBJ_LOGLOSS, y, row_weights, len(row_weights) > 0
            )
        var ev = _gbdt_host_eval_arm(
            eval_x_address, eval_y_address, n_eval_rows, n_features,
            od_kind, od.auto_stop_p_value, od.iterations_wait,
            use_best_model, best_model_min_trees,
        )
        if len(flags) != 0:
            # lane gbdt-cpu-rest-4-cat-ns (2026-11-23): Logloss and RMSE with
            # cat_features and one_hot_features columns on the non-symmetric
            # driver (gbdt_oracle_depthwise.mojo::
            # gbdt_ctr_host_fit_non_symmetric)
            var cat_ns = gbdt_ctr_host_fit_non_symmetric(
                x, y, n_rows, n_features, tp, flags, permutation_count,
                ctr_permutation_id, ev^, max_ctr_complexity, ns_start,
                row_weights,
            )
            text = gbdt_text_with_bias(
                gbdt_ctr_host_ns_model_text(cat_ns), len(cat_ns.model.losses),
                ns_start,
            )
            losses = cat_ns.model.losses.copy()
            best_iteration = cat_ns.model.best_iteration
            fit_loop = cat_ns.model.fit_loop.copy()
            stopped_early = cat_ns.model.stopped_early
            test_losses = cat_ns.model.test_losses.copy()
        else:
            var ns_model = gbdt_host_fit_non_symmetric(
                x, y, n_rows, n_features, tp, ev^, ns_start, row_weights
            )
            text = gbdt_text_with_bias(
                gbdt_host_ns_model_text(ns_model), len(ns_model.losses), ns_start
            )
            losses = ns_model.losses.copy()
            best_iteration = ns_model.best_iteration
            fit_loop = ns_model.fit_loop.copy()
            stopped_early = ns_model.stopped_early
            test_losses = ns_model.test_losses.copy()

    return _fit_outcome(text^, best_iteration, stopped_early, losses^, test_losses^, fit_loop)


def gbdt_predict_multi_run(
    text: String,
    x_address: Int,
    n_rows: Int,
    mode: Int,
    op: MutPointer[Float32, MutUntrackedOrigin],
) raises -> Int:
    """The typed body of `gbdt_predict_multi_binding`: parse the model, check
    the mode against its dimension, write the rows. Returns the width
    written. The door calls it with the GIL released."""
    if n_rows <= 0:
        raise Error("gbdt_predict_multi: n_rows must be positive")
    if gbdt_host_text_has_ctr(text):
        # a CTR model (lane gbdt-units-6): the expansion and walk of
        # `gbdt_host_predict_ctr_into`, the same mode checks
        var cdim = gbdt_host_model_dim(text)
        if mode != 0 and cdim < 2:
            raise Error(
                "gbdt_predict_multi: a probability mode needs a"
                " multi-dimensional model; this one has dim " + String(cdim)
                + ". A two-class problem's link is the sigmoid, which"
                " Logloss's own predict_proba applies."
            )
        _ = gbdt_host_predict_ctr_into(text, x_address, n_rows, mode, op)
        return cdim + 1 if mode == 1 else cdim
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
    gbdt_host_predict_multi_into(m^, x_address, n_rows, n_features, n_trees, dim, mode, op)
    return width


def gbdt_fit_two_level_feature_freq_run(
    x_address: Int,
    y_address: Int,
    sp: MutPointer[UInt32, MutUntrackedOrigin],
    n_rows: Int,
    n_features: Int,
    n_weights: Int,
    n_sources: Int,
    learning_rate: Float32,
    l2_leaf_reg: Float32,
    # the sample weights' address (read when `n_weights` is n_rows; lane
    # gbdt-cpu-rest, 2026-11-23, the device's `has_w` arm)
    weights_address: Int = 0,
) raises -> String:
    """The typed body of `gbdt_fit_two_level_feature_freq_binding`: its
    checks and refusal, the source columns and the host fit. The door calls
    it with the GIL released."""
    if n_rows < 1 or n_features < 2 or n_sources < 2:
        raise Error("two-level FeatureFreq fit needs rows and two sources")
    if n_weights != 0 and n_weights != n_rows:
        raise Error("two-level FeatureFreq weights must be empty or per-row")
    var sources = gbdt_host_load_sources(sp, n_sources)
    var x = read_f32(x_address, n_rows * n_features)
    var y = read_f32(y_address, n_rows)
    var w = List[Float32]()
    if n_weights != 0:
        w = read_f32(weights_address, n_rows)
    # THE WEIGHTED L2 (lane w3f2-gbdtw, 2026-11-24): the device fit's
    # `device_weighted_l2` (`fit_two_level_feature_freq_tree`), the same
    # fold here, once per fit; unweighted, `l2_leaf_reg` exactly
    return gbdt_feature_freq_host_fit(
        x, y, n_rows, n_features, sources, learning_rate,
        host_weighted_l2(l2_leaf_reg, w, n_rows), w,
    )


def gbdt_fit_ordered_rmse_run(
    x_address: Int,
    y_address: Int,
    weights_address: Int,
    permutation_address: Int,
    n_rows: Int,
    n_features: Int,
    n_weights: Int,
    n_permutation: Int,
    n_estimators: Int,
    max_depth: Int,
    border_count: Int,
    learning_rate: Float32,
    l2_leaf_reg: Float32,
) raises -> String:
    """The typed body of `gbdt_fit_ordered_rmse_binding`: its checks and
    refusal, the permutation and the host fit. The door calls it with the
    GIL released."""
    if n_rows < 4 or n_features < 1 or n_permutation != n_rows:
        raise Error("ordered RMSE requires >=4 rows, features and a full permutation")
    if n_weights != 0 and n_weights != n_rows:
        raise Error("ordered RMSE sample weight shape mismatch")
    _ = f32_ptr(x_address)
    _ = f32_ptr(y_address)
    _ = f32_ptr(weights_address)
    var pp = u32_ptr(permutation_address)
    var permutation = gbdt_host_load_u32(pp, n_rows)
    var x = read_f32(x_address, n_rows * n_features)
    var y = read_f32(y_address, n_rows)
    var w = List[Float32]()
    if n_weights != 0:
        w = read_f32(weights_address, n_rows)
    # THE WEIGHTED L2 (lane w3f2-gbdtw, 2026-11-24): the device fit's
    # `device_weighted_l2` (`fit_ordered_rmse`), the same fold here, once
    # per fit; unweighted, `l2_leaf_reg` exactly
    return gbdt_ordered_rmse_host_fit(
        x, y, n_rows, n_features, permutation, n_estimators, max_depth,
        border_count, learning_rate, host_weighted_l2(l2_leaf_reg, w, n_rows), w,
    )
