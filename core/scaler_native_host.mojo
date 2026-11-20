# SPDX-License-Identifier: Apache-2.0
"""Adapters to existing scaler kernels, preserving the provider image."""
from core.scaler_native_api import ScalerNativeKernels, ScalerFitRequest, ScalerTransformRequest, MinMaxFitRequest, MinMaxTransformRequest
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.mojolearn_preprocessing_host_impl import load, ptr, validate_standard, standard_fit_host_into, standard_transform_host_into
from core.mojolearn_preprocessing_host_impl import validate_dimensions, minmax_fit_host_into, minmax_transform_host_into

struct ScalerHostKernels(ScalerNativeKernels):
    @staticmethod
    def mode() raises -> Int:
        return Int(GLOBAL_NUMERIC_MODE)

    @staticmethod
    def direct() raises -> Int:
        return 0

    @staticmethod
    def fit(r: ScalerFitRequest) raises -> Int:
        validate_standard(r.rows, r.features, Int(r.with_mean), Int(r.with_std))
        if r.direct:
            raise Error("scaler native host: direct route unavailable")
        var x = load(r.source, r.rows * r.features)
        standard_fit_host_into(x, ptr(r.dst), r.rows, r.features, Int(r.with_mean), Int(r.with_std))
        _ = x
        return 1

    @staticmethod
    def transform(r: ScalerTransformRequest) raises -> Int:
        validate_standard(r.rows, r.features, Int(r.with_mean), Int(r.with_std))
        if r.direct:
            raise Error("scaler native host: direct route unavailable")
        var x = load(r.source, r.rows * r.features)
        var mean = load(r.mean, r.features)
        var scale = load(r.scale, r.features)
        standard_transform_host_into(x, mean, scale, ptr(r.dst), r.rows, r.features, Int(r.inverse), Int(r.with_mean), Int(r.with_std))
        _ = x
        _ = mean
        _ = scale
        return 1

    # MinMax (lane scaler-typed-core): the host binding's minmax_fit /
    # minmax_transform bodies (preprocessing/host/scaler_oracle.mojo, the same
    # fold order as the device kernels). The host has no direct route.
    @staticmethod
    def minmax_fit(r: MinMaxFitRequest) raises -> Int:
        validate_dimensions(r.rows, r.features, r.lower, r.upper)
        if r.direct:
            raise Error("scaler native host: direct route unavailable")
        var x = load(r.source, r.rows * r.features)
        minmax_fit_host_into(x, ptr(r.dst), r.rows, r.features, r.lower, r.upper)
        _ = x
        return 1

    @staticmethod
    def minmax_transform(r: MinMaxTransformRequest) raises -> Int:
        validate_dimensions(r.rows, r.features, r.lower, r.upper)
        if r.direct:
            raise Error("scaler native host: direct route unavailable")
        var x = load(r.source, r.rows * r.features)
        var scale = load(r.scale, r.features)
        var offset = load(r.offset, r.features)
        minmax_transform_host_into(x, scale, offset, ptr(r.dst), r.rows, r.features, Int(r.inverse), Int(r.clip), r.lower, r.upper)
        _ = x
        _ = scale
        _ = offset
        return 1
