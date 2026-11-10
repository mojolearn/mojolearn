# SPDX-License-Identifier: Apache-2.0
"""LinearRegression fit orchestration over native buffers and solver entries."""
from core.hostptr import f32_ptr
from core.linear_fit_types import (
    LinearFitRequest, LinearFitResult, LinearOlsPlan, LF_FLAGS, LF_MODE,
    LF_PREP, LF_OLS, LF_OLS_RESIDENT, LF_HAS_OLS_RESIDENT, LF_PAIRED_MEANS,
)
from core.linear_tsqr_run import linear_tsqr_solve
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call
from glm.impl.lm_finish import lm_intercept
from x_decomp.descriptor_schema import XD_MODE, XD_VENDOR, XD_FLAGS, XD_TSQR_R, XD_OLS_TSQR_R


def linear_ols_plan(entry: NativeDescriptorCall, rows: Int, cols: Int, estimator: String) raises -> LinearOlsPlan:
    if rows < 2:
        raise Error(estimator + " cannot be fit on n_samples=" + String(rows) + ": Found array with " + String(rows)
            + " sample(s) (shape=(" + String(rows) + ", " + String(cols) + ")) while a minimum of 2 is required.")
    var flags = native_descriptor_call(entry, LF_FLAGS, [], [], [])
    # This is TSQR's factorization width limit, not a benchmark shape.
    var tsqr = cols >= 1 and cols + 1 <= 512 and rows >= cols + 1 and not (flags & LF_HAS_OLS_RESIDENT)
    var mode = native_descriptor_call(entry, LF_MODE, [], [], []) if tsqr else 0
    return LinearOlsPlan(flags, tsqr, mode)


def linear_ols_run(entry: NativeDescriptorCall, decomp: NativeDescriptorCall,
    request: LinearFitRequest, plan: LinearOlsPlan, vendor: Int) raises -> LinearFitResult:
    var rows = request.rows
    var cols = request.cols
    var center = request.intercept
    var weighted = request.weights != 0
    var ymean = List[Float64](length=1, fill=0.0)
    var ignored_mean = List[Float32](length=cols if not center else 0, fill=0.0)
    var mean_address = request.mean if center else Int(ignored_mean.unsafe_ptr())
    var result = LinearFitResult()
    var done = False
    try:
        if plan.flags & LF_HAS_OLS_RESIDENT and not weighted:
            _ = native_descriptor_call(entry, LF_OLS_RESIDENT,
                [request.x, request.y, request.coef, mean_address, Int(ymean.unsafe_ptr())],
                [rows, cols, Int(center)], [0.0])
            done = True
        if plan.tsqr:
            if native_descriptor_call(decomp, XD_VENDOR, [], [], []) != vendor:
                raise Error("RuntimeError: TSQR vendor differs from selected execution")
            if native_descriptor_call(decomp, XD_MODE, [], [], []) != plan.numeric_mode:
                raise Error("RuntimeError: TSQR numeric mode differs from selected execution")
        if not done and plan.tsqr and center and not weighted:
            if native_descriptor_call(decomp, XD_FLAGS, [], [], []) & 1:
                var ra = List[Float32](length=(cols + 1) * (cols + 1), fill=0.0)
                try:
                    _ = native_descriptor_call(decomp, XD_OLS_TSQR_R,
                        [request.x, request.y, Int(ra.unsafe_ptr()), request.mean, Int(ymean.unsafe_ptr())],
                        [rows, cols], [])
                    linear_tsqr_solve(Int(ra.unsafe_ptr()), cols, request.coef, decomp)
                finally:
                    _ = ra
                done = True
        if not done:
            var prepare = center or weighted
            var work_x = List[Float32](unsafe_uninit_length=rows * cols if prepare else 0)
            var work_y = List[Float32](unsafe_uninit_length=rows if prepare else 0)
            var xm64 = List[Float64](length=cols if prepare else 0, fill=0.0)
            var ym32 = List[Float32](length=1 if prepare else 0, fill=0.0)
            var x_address = request.x
            var y_address = request.y
            try:
                if prepare:
                    x_address = Int(work_x.unsafe_ptr())
                    y_address = Int(work_y.unsafe_ptr())
                    var mode = 2 if weighted else (1 if plan.flags & LF_PAIRED_MEANS else 0)
                    _ = native_descriptor_call(entry, LF_PREP,
                        [request.x, request.y, request.weights, x_address, y_address,
                         Int(xm64.unsafe_ptr()), mean_address, Int(ymean.unsafe_ptr()), Int(ym32.unsafe_ptr())],
                        [rows, cols, Int(center), mode], [])
                if plan.tsqr:
                    var ra = List[Float32](length=(cols + 1) * (cols + 1), fill=0.0)
                    try:
                        _ = native_descriptor_call(decomp, XD_TSQR_R,
                            [x_address, y_address, Int(ra.unsafe_ptr())], [rows, cols, 1, 0], [])
                        linear_tsqr_solve(Int(ra.unsafe_ptr()), cols, request.coef, decomp)
                    finally:
                        _ = ra
                else:
                    _ = native_descriptor_call(entry, LF_OLS,
                        [x_address, y_address, request.coef], [rows, cols], [])
            finally:
                _ = work_x
                _ = work_y
                _ = xm64
                _ = ym32
        if center:
            result.ymean = ymean[0]
            result.intercept = lm_intercept(f32_ptr(request.mean), f32_ptr(request.coef), cols, result.ymean)
    finally:
        _ = ymean
        _ = ignored_mean
    return result
