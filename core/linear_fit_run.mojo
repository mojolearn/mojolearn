# SPDX-License-Identifier: Apache-2.0
"""Linear fit RUN owns scratch, route selection and intermediate results.

The Python boundary lends admitted buffers and retains their owners. Every
callee here accepts native values; the measured kernels remain unchanged.
"""
from std.memory import unsafe_memcpy
from core.hostptr import f32_ptr
from core.linear_fit_types import (
    LinearFitRequest, LinearFitResult, LF_FLAGS, LF_GRAM, LF_RIDGE_RESIDENT,
    LF_PREP, LF_RIDGE, LF_HAS_GRAM, LF_HAS_RIDGE_RESIDENT, LF_PAIRED_MEANS, ridge_fit_result_words,
)
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call
from glm.impl.lm_finish import lm_intercept


def linear_prep_x_out(request: LinearFitRequest, host_work: Int) raises -> Int:
    """LF_PREP's prepared-X output address: the host scratch for a host X;
    for THE FOLD X WORD (`request.x < 0`, lane vdf-scope-linear4) the
    request's `prep_slot` Int64, which LF_PREP fills with the fold word of
    the device slot that holds the prepared X (glm/impl/ols_prep_device.mojo)."""
    if request.x >= 0:
        return host_work
    if request.prep_slot == 0:
        raise Error("RuntimeError: a fold X's prepared X needs the member's prep_slot word")
    MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=request.prep_slot)[] = Int64(0)
    return request.prep_slot


def linear_prep_x_in(request: LinearFitRequest, out_address: Int) raises -> Int:
    """The X address the solve entry reads after LF_PREP: the host scratch,
    or the prepared slot's fold word (negative) LF_PREP wrote."""
    if request.x >= 0:
        return out_address
    var word = Int(MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=out_address)[])
    if word >= 0:
        raise Error("RuntimeError: LF_PREP left no prepared X slot for a fold X")
    return word


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
    # THE FOLD X WORD (core/fold_x_device.mojo): a negative X names a device fold
    # slot. Lane vdf-scope-linear4: every route below reads it on the device:
    # the weighted / one-column solve (x_linear/ridge_descriptor_device.mojo,
    # `fit_device` ALGO_RIDGE uploads through fold_x_fill), LF_GRAM, the
    # resident Ridge, and LF_PREP + LF_RIDGE (the prepared X stays a device
    # slot, `linear_prep_x_out`; ridge_fit_host reads it through fold_x_fill).
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
            var work_x = List[Float32](unsafe_uninit_length=rows * cols if request.intercept and request.x >= 0 else 0)
            var work_y = List[Float32](unsafe_uninit_length=rows if request.intercept else 0)
            var xm64 = List[Float64](unsafe_uninit_length=cols if request.intercept else 0)
            var ym32 = List[Float32](unsafe_uninit_length=1 if request.intercept else 0)
            var x_address = request.x
            var y_address = request.y
            try:
                if request.intercept:
                    x_address = linear_prep_x_out(request, Int(work_x.unsafe_ptr()))
                    y_address = Int(work_y.unsafe_ptr())
                    _ = native_descriptor_call(entry, LF_PREP,
                        [request.x, request.y, 0, x_address, y_address,
                         Int(xm64.unsafe_ptr()), request.mean, Int(ymean.unsafe_ptr()), Int(ym32.unsafe_ptr())],
                        [rows, cols, 1, 1 if flags & LF_PAIRED_MEANS else 0], List[Float64]())
                    x_address = linear_prep_x_in(request, x_address)
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


def linear_ridge_targets_run(weighted_entry: NativeDescriptorCall, request: LinearFitRequest, targets: Int,
                             intercepts: Int) raises:
    """Ridge on `targets` targets at once (lane w3f3-k; scikit-learn's Ridge
    takes a 2-D y): `request.y` is rows x targets float32, row-major; the
    targets x cols coefficients go to `request.coef` (row t is target t's)
    and the targets' float64 intercepts to the words at `intercepts`. ONE
    solve with every target as a right-hand side: the estimator's Cholesky
    route (the one-target weighted / one-feature fit above, `weighted_entry`)
    forms the centered Gram and factors it once, then solves each target's
    X'y against the same factor (x_linear/ridge.mojo `ridge_fit`, T
    targets), with or without sample weights. `request.mean` is untouched
    (zeros), as on that route."""
    var rows = request.rows
    var cols = request.cols
    if rows <= 0 or cols <= 0 or targets < 1:
        raise Error("Ridge: rows, cols and targets must be positive")
    if request.alpha < 0.0:
        raise Error("alpha must be non-negative, got " + String(request.alpha))
    if not request.x or not request.y or not request.coef or not intercepts:
        raise Error("Ridge: null fit buffer")
    var has_weights = request.weights != 0
    var words = rows * targets
    var joined = List[Float32](unsafe_uninit_length=words + rows if has_weights else 0)
    var y_address = request.y
    if has_weights:
        unsafe_memcpy(dest=joined.unsafe_ptr(), src=f32_ptr(request.y), count=words)
        unsafe_memcpy(dest=joined.unsafe_ptr() + words, src=f32_ptr(request.weights), count=rows)
        y_address = Int(joined.unsafe_ptr())
    var width = ridge_fit_result_words(targets, cols)
    var values = List[Float32](length=width, fill=0.0)
    try:
        _ = native_descriptor_call(weighted_entry, 0,
            [request.x, y_address, Int(values.unsafe_ptr())],
            [rows, cols, Int(request.intercept), Int(has_weights), targets], [request.alpha])
        if values[width - 1] == Float32(2.0):
            raise Error("mojolearn Ridge: X'X + alpha I is singular even in float-float; rescale X or use a larger alpha")
        unsafe_memcpy(dest=f32_ptr(request.coef), src=values.unsafe_ptr(), count=targets * cols)
        var icpt = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=intercepts)
        for t in range(targets):  # small-loop(the targets' intercept words)
            icpt[t] = Float64(values[targets * cols + t]) if request.intercept else 0.0
    finally:
        _ = joined
        _ = values
