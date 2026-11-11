# SPDX-License-Identifier: Apache-2.0
"""Adapters to existing scaler kernels, preserving the provider image."""
from core.scaler_native_api import ScalerNativeKernels, ScalerFitRequest, ScalerTransformRequest
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.mojolearn_preprocessing_host_impl import load, ptr, validate_standard, standard_fit_host_into, standard_transform_host_into

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
