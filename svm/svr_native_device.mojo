# SPDX-License-Identifier: Apache-2.0
"""Typed requests around the existing SVR device-X, short and resident routes."""
from std.memory import unsafe_memcpy
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.hostptr import f32_ptr, f64_ptr, i32_ptr, read_f32, copy_f32
from core.device_query_x import QueryFold
from core.fitted_resident_store import fitted_lease_close
from core.shared_slot import SharedSlot
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.svr_native_api import SVR_NATIVE_ABI, SvrNativeKernels, SvrFitRequest, SvrQueryRequest, svr_native_dispatch
from svm.estimator import svr_fit_host_borrowed, svr_predict_host
from svm.svm_resident_state import svr_predict_resident_host, svm_resident_lease_open, svm_resident_store


struct SvrDeviceKernels(SvrNativeKernels):
    @staticmethod
    def device() -> Bool:
        return True

    @staticmethod
    def fit(r: SvrFitRequest) raises -> Int:
        var fold = QueryFold(r.base_id, r.base_rows, r.index_id)
        var source = 0 if fold.active() else r.x
        if r.candidate:
            # The existing candidate's short ABI admits its host X pointer
            # before output pointers; the default device-X entry does not.
            source = Int(f32_ptr(r.x))
        var yp = f32_ptr(r.target)
        var dp = f32_ptr(r.dual)
        var sip = i32_ptr(r.support)
        var smp = f32_ptr(r.vectors)
        var ip = f64_ptr(r.info)
        if r.rows <= 0 or r.features <= 0:
            raise Error("svr_fit: n_rows and n_features must both be positive")
        var target = read_f32(Int(yp), max(0, r.rows))
        var bounds = List[Float32]()
        if r.bounds != 0:
            bounds = read_f32(r.bounds, r.rows)
        var fitted = svr_fit_host_borrowed(source, fold, target, r.rows, r.features,
            r.kernel, r.gamma, r.c, r.epsilon, r.tolerance, r.max_iter,
            r.nochange_steps, bounds, r.degree, r.coef0, r.scale_gamma)
        _ = target
        _ = bounds
        if r.candidate and fitted.finite_word != 0:
            raise Error("Input X contains NaN or infinity (SVR).")
        copy_f32(fitted.dual_coefs.unsafe_ptr(), dp, fitted.n_support)
        unsafe_memcpy(dest=sip, src=fitted.support_idx.unsafe_ptr(), count=fitted.n_support)
        copy_f32(fitted.support_matrix.unsafe_ptr(), smp, fitted.n_support * r.features)
        ip[0] = Float64(fitted.b)
        ip[1] = Float64(fitted.n_support)
        ip[2] = Float64(fitted.n_iter)
        if r.device_form:
            ip[3] = Float64(fitted.finite_word)
            ip[4] = Float64(fitted.range_word)
            ip[5] = fitted.gamma
        elif fitted.finite_word != 0:
            raise Error("Input X contains NaN or infinity (SVR).")
        var count = fitted.n_support
        _ = fitted
        return count

    @staticmethod
    def query(r: SvrQueryRequest, resident: Bool) raises -> Int:
        if resident:
            if r.rows <= 0 or r.features <= 0:
                raise Error("svr_predict: n_rows and n_features must both be positive")
            if r.supports < 0:
                raise Error("svr_predict: n_support cannot be negative")
            var words = svr_predict_resident_host(r.lease, r.slot, r.x, r.rows, r.features,
                r.vectors if r.supports > 0 and r.kernel != 4 else 0,
                r.dual if r.supports > 0 else 0, r.supports, Float32(r.bias),
                r.kernel, r.gamma, r.cache_mib, r.degree, r.coef0, r.dst,
                QueryFold(r.base_id, r.base_rows, r.index_id))
            if r.status != 0:
                var status = i32_ptr(r.status)
                status[0] = words[0]
                status[1] = words[1]
            return r.rows
        var xp = f32_ptr(r.x)
        var output = f32_ptr(r.dst)
        if r.rows <= 0 or r.features <= 0:
            raise Error("svr_predict: n_rows and n_features must both be positive")
        if r.supports < 0:
            raise Error("svr_predict: n_support cannot be negative")
        var source = read_f32(Int(xp), max(0, r.rows * r.features))
        var dual = List[Float32]()
        var vectors = List[Float32]()
        if r.supports > 0:
            var dp = f32_ptr(r.dual)
            var smp = f32_ptr(r.vectors)
            dual = read_f32(Int(dp), max(0, r.supports))
            if r.kernel != 4:
                vectors = read_f32(Int(smp), max(0, r.supports * r.features))
        var result = svr_predict_host(source, r.rows, r.features, vectors, dual, r.supports,
            Float32(r.bias), r.kernel, r.gamma, r.cache_mib, r.degree, r.coef0)
        copy_f32(result.unsafe_ptr(), output, r.rows)
        _ = source
        _ = dual
        _ = vectors
        _ = result
        return r.rows

    @staticmethod
    def lease_open() raises -> Int:
        return svm_resident_lease_open()

    @staticmethod
    def lease_close(handle: Int) raises:
        fitted_lease_close(svm_resident_store(), handle)


def svr_device_native_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    return svr_native_dispatch[SvrDeviceKernels](operation, aa, na, pp, np, ff, nf, result, context, sink)


struct SvrDeviceAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(SVR_NATIVE_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(svr_device_native_v1)


comptime _SLOT = "MojolearnSvrDeviceIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnSvrDeviceFastV1"
comptime SVR_DEVICE_API = SharedSlot[SvrDeviceAPI, _SLOT]
