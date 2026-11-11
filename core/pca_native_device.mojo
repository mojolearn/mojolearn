# SPDX-License-Identifier: Apache-2.0
"""Typed PCA device calls preserving exact, dense, randomized and fused routes."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.vendor import COMPILED_VENDOR
from core.shared_slot import SharedSlot
from core.neural_context import process_ctx
from core.hostptr import f32_ptr
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.estimators_impl import pca_whiten_overlap_check
from decomposition.estimator import (
    pca_transform_host, inverse_transform_host, pca_whiten_transform_host, pca_whiten_inverse_transform_host,
)
from decomposition.impl.pca_owner import pca_fit_owned, pca_randomized_owned
from core.pca_native_checks import pca_whiten_dimensions, pca_whiten_device_pointer
from core.pca_native_api import (
    PCA_NATIVE_ABI, PCA_CAP_EXACT, PCA_CAP_RANDOMIZED, PCA_CAP_FULL, PCA_CAP_WHITEN,
    PCANativeKernels, PCAFitRequest, PCAFitResult, PCAQueryRequest, PCAQueryResult,
    pca_native_dispatch, pca_vendor_code,
)

comptime PCA_CONTEXT = "MojolearnEstimatorsContextIdenticalV3" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnEstimatorsContextFastV3"


struct PCADeviceKernels(PCANativeKernels):
    @staticmethod
    def mode() -> Int:
        return Int(GLOBAL_NUMERIC_MODE)

    @staticmethod
    def vendor() -> Int:
        return pca_vendor_code(String(COMPILED_VENDOR))

    @staticmethod
    def capabilities() -> Int:
        return PCA_CAP_EXACT | PCA_CAP_RANDOMIZED | PCA_CAP_FULL | PCA_CAP_WHITEN

    @staticmethod
    def fit(r: PCAFitRequest) raises -> PCAFitResult:
        if r.fold.active():
            raise Error("PCA estimator provider does not own resident folds")
        var xp = f32_ptr(r.source)
        var cp = f32_ptr(r.components)
        var mp = f32_ptr(r.mean)
        var ep = f32_ptr(r.variance)
        var rp = f32_ptr(r.ratio)
        var sp = f32_ptr(r.singular)
        var op = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=r.projection)
        var word = List[Int32](length=1, fill=0)
        var word_address = Int(word.unsafe_ptr()) if r.test_finite else 0
        var keep = r.count
        var noise = Float64(0)
        var ctx = process_ctx[PCA_CONTEXT]()
        if r.randomized:
            noise = pca_randomized_owned(ctx, xp, cp, mp, ep, rp, sp,
                r.rows, r.features, r.count, r.oversamples, r.iterations, r.auto_iterations, r.seed,
                op, r.projection != 0, r.whiten, word_address)
        else:
            var fitted = pca_fit_owned(ctx, xp, cp, mp, ep, rp, sp,
                r.rows, r.features, r.count, r.dense, r.keep_mode, r.fraction,
                op, r.projection != 0, r.whiten, word_address)
            keep = fitted[0]
            noise = fitted[1]
        return PCAFitResult(keep, noise, Int(word[0]), 0)

    @staticmethod
    def query(r: PCAQueryRequest) raises -> PCAQueryResult:
        if r.fold.active():
            raise Error("PCA estimator provider does not own resident folds")
        if r.whiten:
            pca_whiten_dimensions(r)
            var inputs = r.rows * (r.count if r.inverse else r.features)
            var outputs = r.rows * (r.features if r.inverse else r.count)
            var xp = pca_whiten_device_pointer(r.source, inputs)
            var mp = pca_whiten_device_pointer(r.mean, r.features)
            var cp = pca_whiten_device_pointer(r.components, r.count * r.features)
            var sp = pca_whiten_device_pointer(r.singular, r.count)
            var dst = pca_whiten_device_pointer(r.dst, outputs)
            pca_whiten_overlap_check(r.source, r.mean, r.components, r.singular, r.dst,
                inputs, r.features, r.count, outputs)
            var ctx = process_ctx[PCA_CONTEXT]()
            if r.inverse:
                pca_whiten_inverse_transform_host(ctx, xp, cp, sp, mp, dst,
                    r.rows, r.features, r.count, r.fit_rows, device_checks=True)
            else:
                pca_whiten_transform_host(ctx, xp, mp, cp, sp, dst,
                    r.rows, r.features, r.count, r.fit_rows, device_checks=True)
            ctx.synchronize()
            return PCAQueryResult(0, 0)
        var xp = f32_ptr(r.source)
        # Preserve each entry's pointer-admission order.
        if r.inverse:
            var cp = f32_ptr(r.components)
            var mp = f32_ptr(r.mean)
            var dst = f32_ptr(r.dst)
            var ctx = process_ctx[PCA_CONTEXT]()
            inverse_transform_host(ctx, xp, cp, mp, dst, r.rows, r.features, r.count, True)
            ctx.synchronize()
            return PCAQueryResult(0, 0)
        var mp = f32_ptr(r.mean)
        var cp = f32_ptr(r.components)
        var dst = f32_ptr(r.dst)
        var ctx = process_ctx[PCA_CONTEXT]()
        return PCAQueryResult(Int(pca_transform_host(ctx, xp, mp, cp, dst, r.rows, r.features, r.count)), 0)


def pca_device_native_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    return pca_native_dispatch[PCADeviceKernels](operation, aa, na, pp, np, ff, nf, result, context, sink)


struct PCADeviceAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(PCA_NATIVE_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(pca_device_native_v1)


comptime PCA_API_SLOT = "MojolearnPCADeviceIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnPCADeviceFastV1"
comptime PCA_DEVICE_API = SharedSlot[PCADeviceAPI, PCA_API_SLOT]
