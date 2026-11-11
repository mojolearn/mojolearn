# SPDX-License-Identifier: Apache-2.0
"""The ridge multi table (core/linear_entry_table_multi.mojo), reached
natively (lane calltable-base-helpers, 2026-11-11).

A caller refuses before publication and `ridge_multi_available` answers
False. On a build that carries the one-program ridge
(`glm/impl/ridge_multi.mojo::MULTIOUT_RIDGE`, the FAST Apple column) the
estimators binding's publication (`publish_ridge_multi`) is made and each
operation runs through `core/linear_multi_calls.mojo` with literal expected
values; on every other build the publication publishes nothing and the
table stays unavailable, which is what MultiOutputRegressor reads as "one
member per column". The device operations share the process's one device
context, so the cases run one after another. Run in a fresh process.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from core.linear_entry_table import LinearEntryKey
from core.linear_multi_calls import (
    ridge_multi_available, rm_col_sums, rm_means_finish, rm_center, rm_fit, rm_predict,
)
from core.linear_multi_device import publish_ridge_multi
from glm.impl.ridge_multi import MULTIOUT_RIDGE

comptime RIDGE_MULTI_CASES = 5


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _key() -> LinearEntryKey:
    var vendor = (1 if COMPILED_VENDOR == "metal" else 2 if COMPILED_VENDOR == "cuda"
                  else 3 if COMPILED_VENDOR == "hip" else 0)
    return LinearEntryKey(Int(GLOBAL_NUMERIC_MODE), vendor)


def precondition_unpublished() raises:
    require(not ridge_multi_available(_key()), "an unpublished ridge multi table reads as available")
    var x: List[Float32] = [1.0]
    var sums: List[Float64] = [0.0]
    try:
        rm_col_sums(_key(), Int(x.unsafe_ptr()), Int(sums.unsafe_ptr()), 1, 1)
    except problem:
        require("provider is not published" in String(problem), "unexpected refusal: " + String(problem))
        _ = x
        _ = sums
        return
    raise Error("an unpublished ridge multi table did not refuse")


def case(unit: Int) raises:
    var k = _key()
    if unit == 0:
        var x: List[Float32] = [1.0, 2.0, 3.0, 4.0]
        var sums: List[Float64] = [0.0, 0.0]
        rm_col_sums(k, Int(x.unsafe_ptr()), Int(sums.unsafe_ptr()), 2, 2)
        require(sums[0] == 4.0 and sums[1] == 6.0, "lm_col_sums")
        _ = x
        _ = sums
    elif unit == 1:
        var sums: List[Float64] = [4.0, 6.0]
        var m64: List[Float64] = [0.0, 0.0]
        var m32: List[Float32] = [0.0, 0.0]
        rm_means_finish(k, Int(sums.unsafe_ptr()), Int(m64.unsafe_ptr()), Int(m32.unsafe_ptr()), 2, 2, False, 0.0)
        require(m64[0] == 2.0 and m64[1] == 3.0 and m32[0] == 2.0 and m32[1] == 3.0, "lm_means_finish")
        _ = sums
        _ = m64
        _ = m32
    elif unit == 2:
        var x: List[Float32] = [1.0, 2.0, 3.0, 4.0]
        var mu: List[Float32] = [2.0, 3.0]
        var dst: List[Float32] = [9.0, 9.0, 9.0, 9.0]
        rm_center(k, Int(x.unsafe_ptr()), Int(mu.unsafe_ptr()), Int(dst.unsafe_ptr()), 2, 2)
        require(dst[0] == -1.0 and dst[1] == -1.0 and dst[2] == 1.0 and dst[3] == 1.0, "lm_center")
        _ = x
        _ = mu
        _ = dst
    elif unit == 3:
        # Centered X [[-1, 0], [1, 0]], Y [[1], [5]]: ymean 3, coef
        # (X'X + 2 I)^-1 X'y_c = [4 / 4, 0 / 2] = [1, 0], intercept
        # 3 - (2 * 1 + 5 * 0) = 1 with xmean [2, 5].
        var x: List[Float32] = [-1.0, 0.0, 1.0, 0.0]
        var y: List[Float32] = [1.0, 5.0]
        var coef: List[Float32] = [9.0, 9.0]
        var ymean: List[Float32] = [0.0]
        var xmean: List[Float32] = [2.0, 5.0]
        var icpt: List[Float32] = [0.0]
        rm_fit(k, Int(x.unsafe_ptr()), Int(y.unsafe_ptr()), Int(coef.unsafe_ptr()), Int(ymean.unsafe_ptr()),
               2, 2, 1, 2.0, True, Int(xmean.unsafe_ptr()), Int(icpt.unsafe_ptr()))
        require(ymean[0] == 3.0 and abs(coef[0] - 1.0) < 1e-5 and abs(coef[1]) < 1e-5
                and abs(icpt[0] - 1.0) < 1e-5, "ridge_fit_multi")
        _ = x
        _ = y
        _ = coef
        _ = ymean
        _ = xmean
        _ = icpt
    elif unit == 4:
        var x: List[Float32] = [0.0, 0.0, 1.0, 0.0]
        var coef: List[Float32] = [1.0, 0.0]
        var icpt: List[Float32] = [1.0]
        var dst: List[Float32] = [9.0, 9.0]
        rm_predict(k, Int(x.unsafe_ptr()), Int(coef.unsafe_ptr()), Int(icpt.unsafe_ptr()), Int(dst.unsafe_ptr()),
                   2, 2, 1)
        require(abs(dst[0] - 1.0) < 1e-6 and abs(dst[1] - 2.0) < 1e-6, "ridge_predict_multi")
        _ = x
        _ = coef
        _ = icpt
        _ = dst
    else:
        raise Error("no ridge multi case " + String(unit))


def run_checks() raises:
    precondition_unpublished()
    # The publication _mojolearn_estimators makes at module init.
    publish_ridge_multi()
    comptime if MULTIOUT_RIDGE:
        require(ridge_multi_available(_key()), "the published ridge multi table reads as unavailable")
        var failed = 0
        for unit in range(RIDGE_MULTI_CASES):  # driver-loop: device cases share one device context
            try:
                case(unit)
            except problem:
                failed += 1
                print("FAIL ridge multi entry case", unit, String(problem))
        require(failed == 0, "ridge multi entry cases failed: " + String(failed))
        print("PASS ridge multi entry", RIDGE_MULTI_CASES)
    else:
        require(not ridge_multi_available(_key()), "a build without MULTIOUT_RIDGE published the table")
        print("PASS ridge multi entry (not carried by this build: unpublished)")
