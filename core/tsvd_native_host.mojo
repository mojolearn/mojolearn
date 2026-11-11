# SPDX-License-Identifier: Apache-2.0
"""Native TruncatedSVD estimator requests over the existing host kernels."""
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.shared_slot import SharedSlot
from core.hostptr import f32_ptr, read_f32
from core.host_finite_scan import host_finite_status
from core.host_predict_threads import host_predict_task_count
from core.classical_host_predict import host_tsvd_transform_into, host_inverse_transform_into
from core.estimators_host_impl import _positive, host_pca_validate_first, tsvd_fit_host_run, tsvd_fit_explained_host_run
from decomposition.host.pca_oracle import host_tsvd_explained
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.tsvd_native_api import (
    TSVD_NATIVE_ABI, TSVD_CAP_FUSED, TSVD_CAP_HOST, TSVD_GRAM, TSVD_FUSED, TSVD_EXPLAINED,
    TSVDNativeKernels, TSVDFitRequest, TSVDFitResult, TSVDQueryRequest, tsvd_native_dispatch,
)


struct TSVDHostKernels(TSVDNativeKernels):
    @staticmethod
    def mode() -> Int:
        return Int(GLOBAL_NUMERIC_MODE)

    @staticmethod
    def vendor() -> Int:
        return 1

    @staticmethod
    def capabilities() -> Int:
        return TSVD_CAP_FUSED | TSVD_CAP_HOST

    @staticmethod
    def qfix() -> Int:
        return 0

    @staticmethod
    def fit(r: TSVDFitRequest) raises -> TSVDFitResult:
        if r.route == TSVD_EXPLAINED:
            var ep = f32_ptr(r.variance)
            var rp = f32_ptr(r.ratio)
            host_pca_validate_first(r.rows, r.features, r.count)
            host_tsvd_explained(f32_ptr(r.source), f32_ptr(r.components), ep, rp, r.rows, r.features, r.count)
            return TSVDFitResult(0, 0)
        if r.route != TSVD_GRAM and r.route != TSVD_FUSED:
            raise Error("TruncatedSVD host estimator does not provide this fit route")
        var cp = f32_ptr(r.components)
        var sp = f32_ptr(r.singular)
        # Admit fused output pointers before the finite test, as before.
        if r.route == TSVD_FUSED:
            var ep = f32_ptr(r.variance)
            var rp = f32_ptr(r.ratio)
            var finite = 0
            if r.test_finite and r.rows >= 1 and r.features >= 1:
                finite = Int(host_finite_status(r.source, r.rows * r.features))
            if finite == 0:
                tsvd_fit_explained_host_run(r.source, r.components, cp, sp, ep, rp, r.rows, r.features, r.count)
            return TSVDFitResult(finite, 0)
        var finite = 0
        if r.test_finite and r.rows >= 1 and r.features >= 1:
            finite = Int(host_finite_status(r.source, r.rows * r.features))
        if finite == 0:
            tsvd_fit_host_run(r.source, cp, sp, r.rows, r.features, r.count)
        return TSVDFitResult(finite, 0)

    @staticmethod
    def query(r: TSVDQueryRequest) raises -> Int:
        if r.resident:
            raise Error("TruncatedSVD host estimator does not own resident leases")
        var dst = f32_ptr(r.dst)
        _positive(r.rows, "n_rows")
        _positive(r.features, "n_features")
        _positive(r.count, "n_components")
        if r.inverse:
            var components = read_f32(r.components, r.count * r.features)
            var mean = List[Float32]()
            host_inverse_transform_into(f32_ptr(r.source), components, mean, dst,
                r.rows, r.features, r.count, False, host_predict_task_count(r.rows))
            return 0
        var finite = Int(host_finite_status(r.source, r.rows * r.features)) if r.test_finite else 0
        if finite == 0:
            host_tsvd_transform_into(f32_ptr(r.source), f32_ptr(r.components), dst,
                r.rows, r.features, r.count, host_predict_task_count(r.rows))
        return finite

    @staticmethod
    def open() raises -> Int:
        raise Error("TruncatedSVD host estimator does not own resident leases")

    @staticmethod
    def close(lease: Int) raises:
        raise Error("TruncatedSVD host estimator does not own resident leases")


def tsvd_host_native_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    return tsvd_native_dispatch[TSVDHostKernels](operation, aa, na, pp, np, ff, nf, result, context, sink)


struct TSVDHostAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(TSVD_NATIVE_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(tsvd_host_native_v1)


comptime TSVD_HOST_API = SharedSlot[TSVDHostAPI, "MojolearnTSVDHostNativeV1"]
