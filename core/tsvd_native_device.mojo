# SPDX-License-Identifier: Apache-2.0
"""TruncatedSVD estimator and resident requests preserve existing device calls."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.vendor import COMPILED_VENDOR
from core.shared_slot import SharedSlot
from core.neural_context import process_ctx
from core.hostptr import f32_ptr
from core.fitted_resident_store import fitted_lease_close
from decomposition.estimator import tsvd_fit_host, tsvd_fit_explained_host, tsvd_explained_host, tsvd_transform_host, inverse_transform_host
from decomposition.tsvd_resident_state import (
    tsvd_resident_store, tsvd_resident_lease_open, tsvd_transform_resident_device, tsvd_inverse_resident_device,
)
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.pca_native_api import pca_vendor_code
from core.tsvd_native_api import (
    TSVD_NATIVE_ABI, TSVD_CAP_FUSED, TSVD_CAP_RESIDENT, TSVD_GRAM, TSVD_FUSED, TSVD_EXPLAINED,
    TSVDNativeKernels, TSVDFitRequest, TSVDFitResult, TSVDQueryRequest, tsvd_native_dispatch,
)

comptime TSVD_CONTEXT = "MojolearnEstimatorsContextIdenticalV3" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnEstimatorsContextFastV3"


struct TSVDDeviceKernels(TSVDNativeKernels):
    @staticmethod
    def mode() -> Int:
        return Int(GLOBAL_NUMERIC_MODE)

    @staticmethod
    def vendor() -> Int:
        return pca_vendor_code(String(COMPILED_VENDOR))

    @staticmethod
    def capabilities() -> Int:
        return TSVD_CAP_FUSED | TSVD_CAP_RESIDENT

    @staticmethod
    def qfix() -> Int:
        return 0

    @staticmethod
    def fit(r: TSVDFitRequest) raises -> TSVDFitResult:
        if r.route != TSVD_GRAM and r.route != TSVD_FUSED and r.route != TSVD_EXPLAINED:
            raise Error("TruncatedSVD estimator does not provide this fit route")
        var xp = f32_ptr(r.source)
        var cp = f32_ptr(r.components)
        if r.route == TSVD_EXPLAINED:
            var ep = f32_ptr(r.variance)
            var rp = f32_ptr(r.ratio)
            var ctx = process_ctx[TSVD_CONTEXT]()
            tsvd_explained_host(ctx, xp, cp, ep, rp, r.rows, r.features, r.count)
            ctx.synchronize()
            return TSVDFitResult(0, 0)
        var sp = f32_ptr(r.singular)
        if r.route == TSVD_FUSED:
            var ep = f32_ptr(r.variance)
            var rp = f32_ptr(r.ratio)
            var word = List[Int32](length=1, fill=0)
            var ctx = process_ctx[TSVD_CONTEXT]()
            tsvd_fit_explained_host(ctx, xp, cp, sp, ep, rp, r.rows, r.features, r.count,
                Int(word.unsafe_ptr()) if r.test_finite else 0)
            ctx.synchronize()
            return TSVDFitResult(Int(word[0]), 0)
        var word = List[Int32](length=1, fill=0)
        var ctx = process_ctx[TSVD_CONTEXT]()
        tsvd_fit_host(ctx, xp, cp, sp, r.rows, r.features, r.count, Int(word.unsafe_ptr()) if r.test_finite else 0)
        ctx.synchronize()
        return TSVDFitResult(Int(word[0]), 0)

    @staticmethod
    def query(r: TSVDQueryRequest) raises -> Int:
        if r.resident:
            if r.lease < 0:
                var name = String("tsvd_inverse_transform_resident") if r.inverse else String("tsvd_transform_resident")
                raise Error(name + ": the lease must be 0 or a lease, got " + String(r.lease))
            var xp = f32_ptr(r.source)
            var dst = f32_ptr(r.dst)
            var ctx = process_ctx[TSVD_CONTEXT]()
            if r.inverse:
                tsvd_inverse_resident_device(ctx, r.lease, xp, r.components, dst, r.rows, r.features, r.count)
                ctx.synchronize()
                return 0
            var word = List[Int32](length=1, fill=0)
            tsvd_transform_resident_device(ctx, r.lease, xp, r.components, dst,
                r.rows, r.features, r.count, Int(word.unsafe_ptr()) if r.test_finite else 0)
            ctx.synchronize()
            return Int(word[0])
        var xp = f32_ptr(r.source)
        var cp = f32_ptr(r.components)
        var dst = f32_ptr(r.dst)
        var ctx = process_ctx[TSVD_CONTEXT]()
        if r.inverse:
            inverse_transform_host(ctx, xp, cp, cp, dst, r.rows, r.features, r.count, False)
            ctx.synchronize()
            return 0
        var word = List[Int32](length=1, fill=0)
        tsvd_transform_host(ctx, xp, cp, dst, r.rows, r.features, r.count, Int(word.unsafe_ptr()) if r.test_finite else 0)
        ctx.synchronize()
        return Int(word[0])

    @staticmethod
    def open() raises -> Int:
        return tsvd_resident_lease_open()

    @staticmethod
    def close(lease: Int) raises:
        fitted_lease_close(tsvd_resident_store(), lease)


def tsvd_device_native_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    return tsvd_native_dispatch[TSVDDeviceKernels](operation, aa, na, pp, np, ff, nf, result, context, sink)


struct TSVDDeviceAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(TSVD_NATIVE_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(tsvd_device_native_v1)


comptime TSVD_API_SLOT = "MojolearnTSVDDeviceIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnTSVDDeviceFastV1"
comptime TSVD_DEVICE_API = SharedSlot[TSVDDeviceAPI, TSVD_API_SLOT]
