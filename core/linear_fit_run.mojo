# SPDX-License-Identifier: Apache-2.0
"""Linear fit RUN owns scratch, route selection and intermediate results.

The Python boundary lends admitted buffers and retains their owners. Every
callee here accepts native values; the measured kernels remain unchanged.
"""
from std.memory import unsafe_memcpy
from core.hostptr import f32_ptr
from core.linear_fit_types import (
    LinearFitRequest, LinearFitResult, LF_FLAGS, LF_GRAM, LF_RIDGE_RESIDENT,
    LF_PREP, LF_RIDGE, LF_HAS_GRAM, LF_HAS_RIDGE_RESIDENT, LF_PAIRED_MEANS,
)
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call
from glm.impl.lm_finish import lm_intercept


def linear_ridge_run(entry: NativeDescriptorCall, weighted_entry: NativeDescriptorCall,
                     request: LinearFitRequest) raises -> LinearFitResult:
    var rows = request.rows
    var cols = request.cols
    if rows <= 0 or cols <= 0:
        raise Error("Ridge: rows and cols must be positive")
    if request.alpha < 0.0:
        raise Error("alpha must be non-negative, got " + String(request.alpha))
    if not request.x or not request.y or not request.coef or not request.mean:
        raise Error("Ridge: null fit buffer")
    var result = LinearFitResult()
    if request.weights or cols == 1:
        var has_weights = request.weights != 0
        var targets = List[Float32](unsafe_uninit_length=rows * 2 if has_weights else 0)
        var y_address = request.y
        if has_weights:
            unsafe_memcpy(dest=targets.unsafe_ptr(), src=f32_ptr(request.y), count=rows)
            unsafe_memcpy(dest=targets.unsafe_ptr() + rows, src=f32_ptr(request.weights), count=rows)
            y_address = Int(targets.unsafe_ptr())
        var values = List[Float32](length=cols + 5, fill=0.0)
        try:
            _ = native_descriptor_call(weighted_entry, 0,
                [request.x, y_address, Int(values.unsafe_ptr())],
                [rows, cols, Int(request.intercept), Int(has_weights)], [request.alpha])
            if values[cols + 4] == Float32(2.0):
                raise Error("mojolearn Ridge: X'X + alpha I is singular even in float-float; rescale X or use a larger alpha")
            unsafe_memcpy(dest=f32_ptr(request.coef), src=values.unsafe_ptr(), count=cols)
            if request.intercept:
                result.intercept = Float64(values[cols])
        finally:
            _ = targets
            _ = values
        return result^

    var flags = native_descriptor_call(entry, LF_FLAGS, List[Int](), List[Int](), List[Float64]())
    var ymean = List[Float64](length=1, fill=0.0)
    # A no-intercept fit publishes zero means, even if a resident kernel
    # writes its temporary means. Keep those discarded words native.
    var ignored_mean = List[Float32](length=cols if not request.intercept else 0, fill=0.0)
    var mean_address = request.mean if request.intercept else Int(ignored_mean.unsafe_ptr())
    var done = False
    try:
        var addresses: List[Int] = [request.x, request.y, request.coef, mean_address, Int(ymean.unsafe_ptr())]
        var ints: List[Int] = [rows, cols, Int(request.intercept)]
        var floats: List[Float64] = [request.alpha]
        if flags & LF_HAS_GRAM:
            done = native_descriptor_call(entry, LF_GRAM, addresses, ints, floats) == 0
        if not done and flags & LF_HAS_RIDGE_RESIDENT:
            _ = native_descriptor_call(entry, LF_RIDGE_RESIDENT, addresses, ints, floats)
            done = True
        if not done:
            var work_x = List[Float32](unsafe_uninit_length=rows * cols if request.intercept else 0)
            var work_y = List[Float32](unsafe_uninit_length=rows if request.intercept else 0)
            var xm64 = List[Float64](unsafe_uninit_length=cols if request.intercept else 0)
            var ym32 = List[Float32](unsafe_uninit_length=1 if request.intercept else 0)
            var x_address = request.x
            var y_address = request.y
            try:
                if request.intercept:
                    x_address = Int(work_x.unsafe_ptr())
                    y_address = Int(work_y.unsafe_ptr())
                    _ = native_descriptor_call(entry, LF_PREP,
                        [request.x, request.y, 0, x_address, y_address,
                         Int(xm64.unsafe_ptr()), request.mean, Int(ymean.unsafe_ptr()), Int(ym32.unsafe_ptr())],
                        [rows, cols, 1, 1 if flags & LF_PAIRED_MEANS else 0], List[Float64]())
                _ = native_descriptor_call(entry, LF_RIDGE,
                    [x_address, y_address, request.coef], [rows, cols], [request.alpha])
            finally:
                _ = work_x
                _ = work_y
                _ = xm64
                _ = ym32
        if request.intercept:
            result.ymean = ymean[0]
            result.intercept = lm_intercept(f32_ptr(request.mean), f32_ptr(request.coef), cols, result.ymean)
    finally:
        _ = ymean
        _ = ignored_mean
    return result^
