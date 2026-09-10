# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Reusable numeric training pool for native GPU parameter searches.

The pool owns its device context, frozen quantization grid, compressed index,
labels and weights. Each fit starts a fresh model; changing its random seed
changes training randomness, not the pool's already chosen borders. This is
an explicit dataset, not an implicit cache keyed by a mutable caller pointer.
Categorical CTRs, evaluation pools and Python handles are not exposed here.
"""
from gbdt.options.child_hessian import child_hessian_threshold, check_child_hessian_objective
from max.gpu.host import DeviceBuffer, DeviceContext
from std.math import isfinite
from core.identity_trace import IdentityTrace
from gbdt.train import (
    TrainedModel, _quantize_training_columns, _build_cindex_from_columns,
)
from gbdt.ctrs.ctr_binarization import TBinarizationOptions
from gbdt.models.ctr_value_table import TCtrValueTable
from gbdt.models.tensor_ctr_value_table import TTensorCtrRegistry
from gbdt.methods.doc_parallel_boosting import TAdditiveModel, fit_with_test
from gbdt.options.loss_description import make_loss_description
from gbdt.options.catboost_options import (
    GROW_SYMMETRIC, GROW_LOSSGUIDE, SCORE_FUNCTION_COSINE,
    grow_policy_from_name, set_leaves_estimation_default,
)
from gbdt.targets.kernel.pointwise_targets import OBJECTIVE_RMSE
from gbdt.gpu_util.kernel.bootstrap import (
    BOOTSTRAP_KERNEL_BAYESIAN, BOOTSTRAP_KERNEL_BERNOULLI,
    BOOTSTRAP_KERNEL_POISSON,
)


@fieldwise_init
struct PreparedNumericDataset(Movable):
    """An owned, fixed numeric dataset. Fits on one pool are sequential.

    Treat fields as read-only; mutating device buffers or grid metadata
    invalidates the dataset. Keep the pool alive until a fit returns.
    """
    var ctx: DeviceContext
    var n_rows: Int
    var n_features: Int
    var borders: List[List[Float32]]
    var fold_counts: List[Int]
    var nan_treatment: List[Int]
    var cindex: DeviceBuffer[DType.uint32]
    var targets: DeviceBuffer[DType.float32]
    var weights: DeviceBuffer[DType.float32]
    var has_weights: Bool

    def fit(
        mut self,
        n_estimators: Int = 100,
        max_depth: Int = 6,
        learning_rate: Float32 = Float32(0.03),
        l2_leaf_reg: Float32 = Float32(3),
        grow_policy: String = "SymmetricTree",
        max_leaves: Int = -1,
        min_data_in_leaf: Int = 1,
        min_split_gain: Float64 = -1.0,
        loss: String = "RMSE",
        random_seed: UInt64 = UInt64(0),
        score_function: Int = SCORE_FUNCTION_COSINE,
        leaf_estimation_iterations: Int = -1,
        leaf_estimation_method: Int = -1,
        bootstrap_type: Int = -1,
        bootstrap_param: Float32 = Float32(1),
        min_child_hessian: Float64 = -1.0,
    ) raises -> TrainedModel:
        """Fresh RMSE/Logloss/CrossEntropy model using the frozen pool.

        Bootstrap constants are the native BOOTSTRAP_KERNEL_* values; -1
        disables sampling. The parameter is Bayesian temperature, Bernoulli
        subsample or Poisson rate (lambda), as in the native kernels.
        RMSE uses boost_from_average, as train() does.
        Quantization options belong to preparation and cannot change here.
        """
        if loss != "RMSE" and loss != "Logloss" and loss != "CrossEntropy":
            raise Error("prepared numeric fits support RMSE, Logloss and CrossEntropy")
        if n_estimators < 1 or max_depth < 1 or max_depth > 16:
            raise Error("prepared fit requires positive iterations and depth in 1..16")
        if not isfinite(learning_rate) or learning_rate <= 0:
            raise Error("learning_rate must be finite and positive")
        if not isfinite(l2_leaf_reg) or l2_leaf_reg < 0:
            raise Error("l2_leaf_reg must be finite and nonnegative")
        if not isfinite(bootstrap_param):
            raise Error("bootstrap_param must be finite")
        if bootstrap_type == BOOTSTRAP_KERNEL_BAYESIAN:
            if bootstrap_param < 0:
                raise Error("Bayesian temperature must be nonnegative")
        elif bootstrap_type == BOOTSTRAP_KERNEL_BERNOULLI:
            if bootstrap_param <= 0 or bootstrap_param > 1:
                raise Error("Bernoulli subsample must be in (0, 1]")
        elif bootstrap_type == BOOTSTRAP_KERNEL_POISSON:
            if bootstrap_param <= 0:
                raise Error("Poisson rate must be positive")
        elif bootstrap_type != -1:
            raise Error("unknown bootstrap type")
        var policy = grow_policy_from_name(grow_policy)
        if min_data_in_leaf < 1:
            raise Error("min_data_in_leaf must be positive")
        if policy == GROW_SYMMETRIC and min_data_in_leaf != 1:
            raise Error("min_data_in_leaf requires Depthwise or Lossguide")
        if policy != GROW_LOSSGUIDE and max_leaves >= 0 and max_leaves != (1 << max_depth):
            raise Error("max_leaves works only with Lossguide")
        if max_leaves == 0 or max_leaves > 65536 or max_leaves < -1:
            raise Error("max_leaves must be -1 or in 1..65536")
        if not isfinite(min_split_gain) or (min_split_gain < 0 and min_split_gain != -1):
            raise Error("min_split_gain must be -1 or finite and nonnegative")
        if policy == GROW_SYMMETRIC and min_split_gain >= 0:
            raise Error("min_split_gain requires Depthwise or Lossguide")
        _ = child_hessian_threshold(min_child_hessian, policy, score_function)
        var desc = make_loss_description(loss)
        check_child_hessian_objective(min_child_hessian, desc.loss_function)
        var estimation = set_leaves_estimation_default(
            desc, method_override=leaf_estimation_method,
            iterations_override=leaf_estimation_iterations,
        )
        var model = TAdditiveModel()
        var trace = IdentityTrace()
        var result = fit_with_test(
            model, self.ctx, self.n_rows, self.fold_counts, max_depth,
            self.cindex, self.targets, self.weights, self.has_weights,
            n_estimators, trace, learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg, random_seed=random_seed,
            score_function=score_function, objective=desc.loss_function,
            logloss_border=desc.get_logloss_border(),
            leaf_estimation_iterations=estimation.iterations,
            leaf_estimation_method=estimation.method,
            alpha=desc.kernel_alpha(), estimator_alpha=desc.get_alpha(),
            boost_from_average=desc.loss_function == OBJECTIVE_RMSE,
            bootstrap_type=bootstrap_type, bootstrap_param=bootstrap_param,
            grow_policy=policy, max_leaves=max_leaves,
            min_data_in_leaf=min_data_in_leaf,
            min_split_gain=min_split_gain,
            min_child_hessian=min_child_hessian,
        )
        var flags = List[Bool]()
        for _ in range(self.n_features):
            flags.append(False)
        return TrainedModel(
            model^, self.fold_counts.copy(), flags^,
            self.borders.copy(), self.nan_treatment.copy(),
            result.learn_losses.copy(), result.test_losses.copy(),
            result.best_iteration, result.stopped_early, 0,
            List[TCtrValueTable](), TTensorCtrRegistry(self.n_features),
        )


def prepare_numeric_dataset(
    ctx: DeviceContext,
    x_colmajor: List[Float32],
    y: List[Float32],
    n_rows: Int,
    n_features: Int,
    border_count: Int = 128,
    border_build_max_samples: Int = 200_000,
    random_seed: UInt64 = UInt64(0),
    nan_mode: String = "Min",
    sample_weight: List[Float32] = List[Float32](),
) raises -> PreparedNumericDataset:
    """Quantize and transfer once, using train()'s exact grid builder.

    The returned pool owns all retained data; subsequent caller mutations
    cannot change it. The provided context is retained by a shared handle.
    """
    if n_rows < 1 or n_features < 1 or len(x_colmajor) != n_rows * n_features:
        raise Error("prepared numeric dataset shape mismatch")
    if len(y) != n_rows or (len(sample_weight) != 0 and len(sample_weight) != n_rows):
        raise Error("prepared numeric dataset label/weight size mismatch")
    if border_count < 1 or border_count > 254 or border_build_max_samples < 0:
        raise Error("border_count must be in 1..254; sample cap must be nonnegative")
    var columns = List[List[Float32]]()
    var flags = List[Bool]()
    var no_ctr = List[Int]()
    for f in range(n_features):
        var col = List[Float32]()
        for r in range(n_rows):
            col.append(x_colmajor[f * n_rows + r])
        columns.append(col^)
        flags.append(False)
        no_ctr.append(-1)
    var grid = _quantize_training_columns(
        ctx, columns, flags, no_ctr, no_ctr,
        List[List[List[Float32]]](), List[TBinarizationOptions](),
        n_rows, border_count, border_build_max_samples, random_seed, nan_mode,
    )
    var ci = _build_cindex_from_columns(ctx, columns, n_rows, grid[0], grid[1], grid[2])
    var targets = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var weights = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var ht = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
    var hw = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
    var weighted = len(sample_weight) > 0
    var positive_weight = False
    for r in range(n_rows):
        if not isfinite(y[r]):
            raise Error("prepared dataset labels must be finite")
        var w = sample_weight[r] if weighted else Float32(1)
        if not isfinite(w) or w < 0:
            raise Error("prepared dataset weights must be finite and nonnegative")
        positive_weight |= w > 0
        ht[r] = y[r]
        hw[r] = w
    if not positive_weight:
        raise Error("prepared dataset requires positive total weight")
    ctx.enqueue_copy(dst_buf=targets, src_ptr=ht.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=weights, src_ptr=hw.unsafe_ptr())
    ctx.synchronize()
    _ = ht^
    _ = hw^
    return PreparedNumericDataset(
        ctx.copy(), n_rows, n_features, grid[0].copy(), grid[1].copy(),
        grid[2].copy(), ci^, targets^, weights^, weighted,
    )
