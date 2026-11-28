# SPDX-License-Identifier: Apache-2.0
"""Adapters to existing scaler kernels, preserving the provider image."""
from core.scaler_native_api import ScalerNativeKernels, ScalerFitRequest, ScalerTransformRequest, MinMaxFitRequest, MinMaxTransformRequest, ScalerFitTransformRequest
from checks.numerics import GLOBAL_NUMERIC_MODE
from preprocessing.estimator import validate_standard, standard_fit_direct, standard_fit_refusing, standard_transform_direct, standard_transform_refusing
from preprocessing.estimator import validate_dimensions, minmax_fit_direct, minmax_fit_refusing, minmax_transform_direct, minmax_transform_refusing
from preprocessing.estimator import standard_fit_transform_direct, minmax_fit_transform_direct
from core.hostptr import f32_ptr
from preprocessing.prep_member_store import prep_slot_fit, prep_slot_transform

struct ScalerDeviceKernels(ScalerNativeKernels):
    @staticmethod
    def mode() raises -> Int:
        return Int(GLOBAL_NUMERIC_MODE)

    @staticmethod
    def direct() raises -> Int:
        return 1

    @staticmethod
    def fit(r: ScalerFitRequest) raises -> Int:
        validate_standard(r.rows, r.features, Int(r.with_mean), Int(r.with_std))
        var x = f32_ptr(r.source)
        var dst = f32_ptr(r.dst)
        if r.direct:
            return standard_fit_direct(x, r.rows, r.features, Int(r.with_mean), Int(r.with_std), dst)
        standard_fit_refusing(x, r.rows, r.features, Int(r.with_mean), Int(r.with_std), dst)
        return 1

    @staticmethod
    def transform(r: ScalerTransformRequest) raises -> Int:
        validate_standard(r.rows, r.features, Int(r.with_mean), Int(r.with_std))
        var x = f32_ptr(r.source)
        var mean = f32_ptr(r.mean)
        var scale = f32_ptr(r.scale)
        var dst = f32_ptr(r.dst)
        if r.direct:
            return standard_transform_direct(x, mean, scale, dst, r.rows, r.features, Int(r.inverse), Int(r.with_mean), Int(r.with_std))
        standard_transform_refusing(x, mean, scale, dst, r.rows, r.features, Int(r.inverse), Int(r.with_mean), Int(r.with_std))
        return 1

    # MinMax (lane scaler-typed-core): the device binding's minmax_fit /
    # minmax_fit_direct / minmax_transform / minmax_transform_direct bodies,
    # the same kernels on the same words; direct returns their 1 / 0 / -1.
    @staticmethod
    def minmax_fit(r: MinMaxFitRequest) raises -> Int:
        validate_dimensions(r.rows, r.features, r.lower, r.upper)
        var x = f32_ptr(r.source)
        var dst = f32_ptr(r.dst)
        if r.direct:
            return minmax_fit_direct(x, r.rows, r.features, r.lower, r.upper, dst)
        minmax_fit_refusing(x, r.rows, r.features, r.lower, r.upper, dst)
        return 1

    @staticmethod
    def minmax_transform(r: MinMaxTransformRequest) raises -> Int:
        validate_dimensions(r.rows, r.features, r.lower, r.upper)
        var x = f32_ptr(r.source)
        var scale = f32_ptr(r.scale)
        var offset = f32_ptr(r.offset)
        var dst = f32_ptr(r.dst)
        if r.direct:
            return minmax_transform_direct(x, scale, offset, dst, r.rows, r.features, Int(r.inverse), Int(r.clip), r.lower, r.upper)
        minmax_transform_refusing(x, scale, offset, dst, r.rows, r.features, Int(r.inverse), Int(r.clip), r.lower, r.upper)
        return 1

    # The combined fit_transform (one upload of X): the bodies of the device
    # binding's `standard_fit_transform` / `minmax_fit_transform`, the same
    # kernels on the same words; 1 / 0 / -1 as those return.
    @staticmethod
    def standard_fit_transform(r: ScalerFitTransformRequest) raises -> Int:
        return standard_fit_transform_direct(f32_ptr(r.source), r.rows, r.features, Int(r.with_mean),
                                             Int(r.with_std), f32_ptr(r.stats), f32_ptr(r.dst))

    @staticmethod
    def minmax_fit_transform(r: ScalerFitTransformRequest) raises -> Int:
        return minmax_fit_transform_direct(f32_ptr(r.source), r.rows, r.features, r.lower, r.upper, Int(r.clip),
                                           f32_ptr(r.stats), f32_ptr(r.dst))

    # Lane vdf-scope-neural2: THE RESIDENT ROUTE (core/scaler_native_api.mojo
    # SCALER_OP_FIT_SLOT / TRANSFORM_SLOT): X a slot of the image's
    # preprocessing member store, read in place by the direct bodies.
    @staticmethod
    def fit_slot(slot: Int, n: Int, d: Int, minmax: Bool, with_mean: Int, with_std: Int, lower: Float32,
                 upper: Float32, dst: Int) raises -> Int:
        return prep_slot_fit(slot, n, d, minmax, with_mean, with_std, lower, upper, dst)

    @staticmethod
    def transform_slot(slot: Int, n: Int, d: Int, minmax: Bool, inverse: Int, with_mean: Int, with_std: Int,
                       clip: Int, lower: Float32, upper: Float32, first: Int, second: Int, dst: Int) raises -> Int:
        return prep_slot_transform(slot, n, d, minmax, inverse, with_mean, with_std, clip, lower, upper, first,
                                   second, dst)
