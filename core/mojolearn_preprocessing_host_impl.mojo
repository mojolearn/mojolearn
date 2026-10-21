# SPDX-License-Identifier: Apache-2.0
"""Typed host scaler checks and bodies moved word for word from bindings/_mojolearn_preprocessing_host.mojo."""
from std.math import isfinite
from core.hostptr import copy_f32, f32_ptr, read_f32
from preprocessing.host.scaler_oracle import (
    host_minmax_fit,
    host_minmax_transform,
    host_standard_fit,
    host_standard_transform,
)


def ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("preprocessing: null Float32 pointer")
    return f32_ptr(addr)


def load(addr: Int, n: Int) raises -> List[Float32]:
    _ = ptr(addr)  # Preserve this surface's null-pointer refusal.
    return read_f32(addr, max(0, n))


# `preprocessing/estimator.mojo`'s validation, verbatim.


def validate_dimensions(n: Int, d: Int, lower: Float32, upper: Float32) raises:
    if n <= 0 or d <= 0 or d > 2147483647 or n > 2147483647 // d:
        raise Error("MinMaxScaler: positive dimensions with n*d<=Int32.max required")
    if not isfinite(lower) or not isfinite(upper) or lower >= upper:
        raise Error("MinMaxScaler: finite increasing Float32 feature range required")


def finite_values(values: List[Float32]) raises:
    for value in values:
        if not isfinite(value):
            raise Error("MinMaxScaler: nonfinite input or Float32 arithmetic overflow")


def validate_standard(n: Int, d: Int, with_mean: Int, with_std: Int) raises:
    if n <= 0 or d <= 0 or d > 2147483647 or n > 2147483647 // d:
        raise Error("StandardScaler: positive dimensions with n*d<=Int32.max required")
    if with_mean < 0 or with_mean > 1 or with_std < 0 or with_std > 1:
        raise Error("StandardScaler: flags must be 0 or 1")


def standard_finite(values: List[Float32]) raises:
    for value in values:
        if not isfinite(value):
            raise Error("StandardScaler: nonfinite input or Float32 arithmetic overflow")


def minmax_fit_host_into(
    x: List[Float32], output: MutPointer[Float32, MutUntrackedOrigin], n: Int, d: Int, lower: Float32, upper: Float32,
) raises:
    if len(x) < n * d:
        raise Error("MinMaxScaler: short input")
    finite_values(x)
    var result = host_minmax_fit(x, n, d, lower, upper)
    finite_values(result)
    for c in range(d):
        if result[3 * d + c] <= 0:
            raise Error("MinMaxScaler: Float32 scale underflow")
    copy_f32(result.unsafe_ptr(), output, 5 * d)


def minmax_transform_host_into(
    x: List[Float32], scale: List[Float32], offset: List[Float32], output: MutPointer[Float32, MutUntrackedOrigin],
    n: Int, d: Int, inverse: Int, clip: Int, lower: Float32, upper: Float32,
) raises:
    if len(x) < n * d or len(scale) < d or len(offset) < d or inverse < 0 or inverse > 1 or clip < 0 or clip > 1:
        raise Error("MinMaxScaler: invalid transform parameters")
    finite_values(x)
    finite_values(scale)
    finite_values(offset)
    for c in range(d):
        if scale[c] <= 0:
            raise Error("MinMaxScaler: scale must be positive")
    var result = host_minmax_transform(x, scale, offset, n, d, inverse, clip, lower, upper)
    finite_values(result)
    copy_f32(result.unsafe_ptr(), output, n * d)


def standard_fit_host_into(
    x: List[Float32], output: MutPointer[Float32, MutUntrackedOrigin], n: Int, d: Int, with_mean: Int, with_std: Int,
) raises:
    if len(x) < n * d:
        raise Error("StandardScaler: short input")
    standard_finite(x)
    var result = host_standard_fit(x, n, d, with_mean, with_std)
    standard_finite(result)
    for c in range(d):
        if result[d + c] < 0 or result[2 * d + c] <= 0:
            raise Error("StandardScaler: invalid variance or scale")
    copy_f32(result.unsafe_ptr(), output, 3 * d)


def standard_transform_host_into(
    x: List[Float32], mean: List[Float32], scale: List[Float32], output: MutPointer[Float32, MutUntrackedOrigin],
    n: Int, d: Int, inverse: Int, with_mean: Int, with_std: Int,
) raises:
    if len(x) < n * d or len(mean) < d or len(scale) < d or inverse < 0 or inverse > 1:
        raise Error("StandardScaler: invalid transform parameters")
    standard_finite(x)
    if with_mean != 0:
        standard_finite(mean)
    if with_std != 0:
        standard_finite(scale)
        for c in range(d):
            if scale[c] <= 0:
                raise Error("StandardScaler: scale must be positive")
    var result = host_standard_transform(x, mean, scale, n, d, inverse, with_mean, with_std)
    standard_finite(result)
    copy_f32(result.unsafe_ptr(), output, n * d)
