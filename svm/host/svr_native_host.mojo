# SPDX-License-Identifier: Apache-2.0
"""Typed SVR requests over the existing CPU SMO and prediction owners."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.hostptr import f32_ptr, f64_ptr, i32_ptr, read_f32
from core.shared_slot import SharedSlot
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.svr_native_api import SVR_NATIVE_ABI, SvrNativeKernels, SvrFitRequest, SvrQueryRequest, svr_native_dispatch
from svm.host.svm_host_impl import svr_fit_host, svr_predict_host


struct SvrHostKernels(SvrNativeKernels):
    @staticmethod
    def device() -> Bool:
        return False

    @staticmethod
    def fit(r: SvrFitRequest) raises -> Int:
        var xp = f32_ptr(r.x)
        var dp = f32_ptr(r.dual)
        var sip = i32_ptr(r.support)
        var smp = f32_ptr(r.vectors)
        var ip = f64_ptr(r.info)
        if r.rows <= 0 or r.features <= 0:
            raise Error("svr_fit: n_rows and n_features must both be positive")
        var targets = read_f32(r.target, max(0, r.rows))
        var bounds = List[Float32]()
        if r.bounds != 0:
            bounds = read_f32(r.bounds, r.rows)
        var count = svr_fit_host(xp, targets, bounds, dp, sip, smp, ip, r.rows,
            r.features, r.kernel, r.gamma, r.c, r.epsilon, r.tolerance,
            r.max_iter, r.nochange_steps, r.degree, r.coef0)
        _ = targets
        _ = bounds
        return count

    @staticmethod
    def query(r: SvrQueryRequest, resident: Bool) raises -> Int:
        if resident:
            raise Error("SVR native ABI: the host column has no resident query route")
        var output = f32_ptr(r.dst)
        if r.rows <= 0 or r.features <= 0:
            raise Error("svr_predict: n_rows and n_features must both be positive")
        if r.supports < 0:
            raise Error("svr_predict: n_support cannot be negative")
        svr_predict_host(r.x, output, r.rows, r.features, r.supports, Float32(r.bias),
            r.kernel, r.gamma, r.cache_mib, r.degree, r.coef0,
            r.dual if r.supports > 0 else 0, r.vectors if r.supports > 0 else 0)
        return r.rows

    @staticmethod
    def lease_open() raises -> Int:
        raise Error("SVR native ABI: the host column has no resident lease")

    @staticmethod
    def lease_close(handle: Int) raises:
        raise Error("SVR native ABI: the host column has no resident lease")


def svr_host_native_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    return svr_native_dispatch[SvrHostKernels](operation, aa, na, pp, np, ff, nf, result, context, sink)


struct SvrHostAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(SVR_NATIVE_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(svr_host_native_v1)


comptime _SLOT = "MojolearnSvrHostIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnSvrHostFastV1"
comptime SVR_HOST_API = SharedSlot[SvrHostAPI, _SLOT]
