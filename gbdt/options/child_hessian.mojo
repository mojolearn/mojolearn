# SPDX-License-Identifier: Apache-2.0
"""Validation for the optional scalar Newton child-Hessian growth guard."""
from std.math import isfinite
from std.memory import bitcast
from gbdt.options.catboost_options import GROW_SYMMETRIC, SCORE_FUNCTION_NEWTON_COSINE, SCORE_FUNCTION_NEWTON_L2
from gbdt.targets.kernel.pointwise_targets import OBJECTIVE_RMSE, OBJECTIVE_LOGLOSS, OBJECTIVE_CROSSENTROPY

def child_hessian_threshold(value: Float64, policy: Int, score: Int) raises -> Float32:
    if not isfinite(value) or (value < 0 and value != -1) or value > Float64(Float32.MAX_FINITE):
        raise Error("min_child_hessian must be -1 (disabled) or finite nonnegative and <= Float32.MAX_FINITE")
    if value < 0:
        return -1.0
    if policy == GROW_SYMMETRIC:
        raise Error("min_child_hessian requires Depthwise or Lossguide")
    if score != SCORE_FUNCTION_NEWTON_COSINE and score != SCORE_FUNCTION_NEWTON_L2:
        raise Error("min_child_hessian requires NewtonL2 or NewtonCosine; first-order scores store weights, not Hessians")
    if value == 0:
        return Float32(0)
    # The accumulated score plane is Float32. Round the comparison threshold
    # UP so comparing that plane implements its comparison with the user's
    # Float64 bound, including bounds lying between adjacent Float32 values.
    var result = Float32(value)
    if Float64(result) < value:
        result = bitcast[DType.float32](bitcast[DType.uint32](result) + UInt32(1))
    return result

def check_child_hessian_objective(value: Float64, objective: Int) raises:
    if value >= 0 and objective != OBJECTIVE_RMSE and objective != OBJECTIVE_LOGLOSS and objective != OBJECTIVE_CROSSENTROPY:
        raise Error("min_child_hessian currently supports RMSE, Logloss and CrossEntropy only")
