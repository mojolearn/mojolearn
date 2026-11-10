# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of Mojolearn.
"""Typed linear fitted-state and parameter contracts, without Python.

The owner stays in each parallel fixture while copied native metadata and
prediction requests borrow its buffers. Public attribute/protocol admission
remains covered by the Python boundary suites; these tests invoke no hooks.
"""
from core.host_parallel import host_parallelize
from core.linear_model_state import (
    LinearModelState, linear_state_predict_request, linear_state_is_fitted,
    linear_ridge_parameter_status, linear_ridge_parameter_error,
)


def _require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _request[WIDE: Bool](fault: Int) raises:
    comptime DT = DType.float64 if WIDE else DType.float32
    var coef: List[Scalar[DT]] = [1.25, -2.5, 3.5]
    var x = List[Scalar[DT]](length=6, fill=1.0)
    var dst = List[Scalar[DT]](length=2, fill=-99.0)
    var state = LinearModelState()
    state.coef = Int(coef.unsafe_ptr())
    state.coef_count = 3
    state.features = 3
    state.intercept = 1.0000000000009095
    state.has_coef = True
    state.fitted = True
    var rows = 2
    var cols = 3
    var expected = String("")
    if fault == 1 or fault == 9:
        state.has_coef = False
        expected = "call fit before predict"
        if fault == 9:
            cols = 4
    elif fault == 2 or fault == 8:
        cols = 4
        expected = "expecting 3 features"
        if fault == 8:
            rows = 0
    elif fault == 3:
        state.coef_count = 2
        expected = "coef_ length differs"
    elif fault == 4 or fault == 5:
        rows = 0 if fault == 4 else -1
        expected = "rows and cols must be positive"
    elif fault == 6 or fault == 7:
        cols = 0 if fault == 6 else -1
        state.features = cols
        state.coef_count = cols
        expected = "rows and cols must be positive"
    # Copying metadata does not copy or consume the caller's buffers.
    var snapshot = state
    state.features = 99
    var refused = False
    try:
        var request = linear_state_predict_request(snapshot, Int(x.unsafe_ptr()), Int(dst.unsafe_ptr()),
            rows, cols, WIDE, "native fixture")
        _require(request.rows == 2 and request.cols == 3 and request.wide == WIDE, "prediction dimensions/dtype")
        _require(request.x == Int(x.unsafe_ptr()) and request.dst == Int(dst.unsafe_ptr()), "prediction borrows caller IO")
        _require(request.coef == Int(coef.unsafe_ptr()), "prediction borrows coefficients")
        _require(request.intercept == 1.0000000000009095, "native state preserves Float64 intercept")
        coef[1] = -7.0
        var borrowed = MutPointer[Scalar[DT], MutUntrackedOrigin](unsafe_from_address=request.coef)
        _require(borrowed[1] == -7.0, "copied state refers to the retained coefficient owner")
    except problem:
        _require(expected.byte_length() > 0 and String(problem).find(expected) >= 0, "unexpected state refusal")
        refused = True
    _require(refused == (fault != 0), "native fitted state admission")
    _require(dst[0] == -99.0 and dst[1] == -99.0, "request construction does not execute prediction")
    _require(state.features == 99, "state copy owns independent metadata")
    _ = coef
    _ = x
    _ = dst
    _ = snapshot


def _fitted(unit: Int) raises:
    var expected: List[Bool] = [False, True, True, True, True, True, True, True]
    var flags = unit % 8
    var state = LinearModelState()
    state.fitted = (flags & 1) != 0
    state.fitting = (flags & 2) != 0
    var public_attribute_present = (flags & 4) != 0
    # Input ownership and a native coefficient-presence marker alone do not
    # replace fitted/in-progress/public-attribute state in this API query.
    state.input_copied = unit >= 8
    state.has_coef = unit >= 8
    _require(linear_state_is_fitted(state, public_attribute_present) == expected[flags],
             "fitted/in-progress/public attribute contract")


def _parameters(unit: Int) raises:
    # Explicit accepted and refused public combinations, including refusal
    # precedence when more than one argument is unsupported.
    var alphas: List[Float64] = [0, 2.5, 0, 0.125, -0.25, -3, 0, 0, 0, 0, 0, 0, 1, 1]
    var solvers: List[String] = ["auto", "auto", "eig", "eig", "auto", "mystery",
        "mystery", "mystery", "svd", "cd", "svd", "cd", "auto", "eig"]
    var normalize: List[Bool] = [False, False, False, False, False, True, False,
        True, False, False, True, True, True, True]
    var expected: List[Int] = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4]
    var status = linear_ridge_parameter_status(alphas[unit], solvers[unit], normalize[unit])
    _require(status == expected[unit], "Ridge parameter admission/refusal priority")
    var refused = False
    try:
        linear_ridge_parameter_error(status, "fixture-alpha", "fixture-solver")
    except problem:
        var message = String(problem)
        if status == 1:
            _require(message.find("alpha must be non-negative, got fixture-alpha") >= 0, "negative alpha explanation")
        elif status == 2:
            _require(message.find("TypeError: solver fixture-solver is not supported") >= 0, "unsupported solver explanation")
        elif status == 3:
            _require(message.find("NotImplementedError:") >= 0 and message.find("solver=fixture-solver") >= 0,
                     "unimplemented solver explanation")
        elif status == 4:
            _require(message.find("NotImplementedError:") >= 0 and message.find("normalize is not implemented") >= 0,
                     "unimplemented normalization explanation")
        else:
            raise Error("accepted Ridge parameters raised")
        refused = True
    _require(refused == (status != 0), "parameter refusal must become a native error")


def main() raises:
    comptime FIXTURES = 50
    var failed = List[Int](length=FIXTURES, fill=0)
    var failures_ptr = failed.unsafe_ptr()
    def task(unit: Int) {imm failures_ptr}:
        try:
            if unit < 10:
                _request[False](unit)
            elif unit < 20:
                _request[True](unit - 10)
            elif unit < 36:
                _fitted(unit - 20)
            else:
                _parameters(unit - 36)
        except problem:
            print("FAIL linear_model_state fixture", unit, String(problem))
            failures_ptr.unsafe_store(unit, 1)
    host_parallelize(task, FIXTURES)
    var failures = 0
    for value in failed:
        failures += value
    _require(failures == 0, "linear model state failed " + String(failures) + " of " + String(FIXTURES) + " fixtures")
    print("PASS linear_model_state:", FIXTURES, "native fixtures, parallel")
