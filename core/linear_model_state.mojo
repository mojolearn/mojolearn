# SPDX-License-Identifier: Apache-2.0
"""Typed state consumed by the linear estimator's native operations.

The boundary retains owners of exposed coefficient/mean buffers. These native
descriptors borrow their storage and never carry or inspect a Python object.
"""
from core.linear_boundary_types import LinearPredictRequest


struct LinearModelState(Copyable, Movable, Defaultable):
    var coef: Int
    var coef_count: Int
    var features: Int
    var intercept: Float64
    var mean: Int
    var ymean: Float64
    var has_coef: Bool
    var fitted: Bool
    var fitting: Bool
    var input_copied: Bool

    def __init__(out self):
        self.coef = 0
        self.coef_count = 0
        self.features = 0
        self.intercept = 0.0
        self.mean = 0
        self.ymean = 0.0
        self.has_coef = False
        self.fitted = False
        self.fitting = False
        self.input_copied = False


def linear_state_require_coef(state: LinearModelState) raises:
    if not state.has_coef:
        raise Error("ValueError: call fit before predict")


def linear_state_check_width(state: LinearModelState, cols: Int, estimator: String) raises:
    if cols != state.features:
        raise Error("X has " + String(cols) + " features, but " + estimator + " is expecting "
            + String(state.features) + " features as input.")


def linear_state_check_coefficients(state: LinearModelState, cols: Int) raises:
    if state.coef_count != cols:
        raise Error("coef_ length differs from fitted feature count")


def linear_state_predict_request(state: LinearModelState, x: Int, dst: Int,
    rows: Int, cols: Int, wide: Bool, estimator: String) raises -> LinearPredictRequest:
    linear_state_require_coef(state)
    linear_state_check_width(state, cols, estimator)
    linear_state_check_coefficients(state, cols)
    if rows <= 0 or cols <= 0:
        raise Error("linear prediction: rows and cols must be positive")
    return LinearPredictRequest(x, state.coef, dst, rows, cols, state.intercept, wide)


def linear_state_is_fitted(state: LinearModelState, public_attribute_present: Bool) -> Bool:
    return state.fitted or state.fitting or public_attribute_present


def linear_ridge_parameter_status(alpha: Float64, solver: String, normalize: Bool) -> Int:
    if alpha < 0.0:
        return 1
    if solver != "auto" and solver != "eig" and solver != "svd" and solver != "cd":
        return 2
    if solver == "svd" or solver == "cd":
        return 3
    if normalize:
        return 4
    return 0


def linear_ridge_parameter_error(status: Int, alpha_display: String, solver_display: String) raises:
    if status == 1:
        raise Error("alpha must be non-negative, got " + alpha_display)
    if status == 2:
        raise Error("TypeError: solver " + solver_display + " is not supported")
    if status == 3:
        raise Error("NotImplementedError: mojolearn Ridge: solver=" + solver_display + " is not implemented "
            + "(ridgeSVD is raft::linalg::svdQR -> cuSOLVER gesvd; 'cd' is cuml/solvers/cd.pyx); "
            + "solver='eig' (cuML's 'auto') is the implemented arm. See glm/NOT_IMPLEMENTED.tsv")
    if status == 4:
        raise Error("NotImplementedError: mojolearn Ridge: normalize is not implemented "
            + "(preprocess.cuh:76-108, the meanvar arm of preProcessData; glm/NOT_IMPLEMENTED.tsv)")
