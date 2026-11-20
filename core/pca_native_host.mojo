# SPDX-License-Identifier: Apache-2.0
"""Typed PCA host calls over the existing host implementations."""
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.shared_slot import SharedSlot
from core.hostptr import f32_ptr, read_f32
from core.host_finite_scan import host_finite_status
from core.host_predict_threads import host_predict_task_count
from core.classical_host_predict import host_pca_transform_into, host_inverse_transform_into
from core.estimators_host_impl import (
    _positive, pca_fit_owned_host_run, pca_randomized_owned_host_run,
    pca_whiten_check_host, pca_whiten_host_run,
)
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.pca_native_checks import pca_whiten_dimensions
from core.pca_native_api import (
    PCA_NATIVE_ABI, PCA_CAP_EXACT, PCA_CAP_RANDOMIZED, PCA_CAP_FULL, PCA_CAP_WHITEN, PCA_CAP_HOST,
    PCANativeKernels, PCAFitRequest, PCAFitResult, PCAQueryRequest, PCAQueryResult, pca_native_dispatch,
)


struct PCAHostKernels(PCANativeKernels):
    @staticmethod
    def mode() -> Int:
        return Int(GLOBAL_NUMERIC_MODE)

    @staticmethod
    def vendor() -> Int:
        return 1

    @staticmethod
    def capabilities() -> Int:
        return PCA_CAP_EXACT | PCA_CAP_RANDOMIZED | PCA_CAP_FULL | PCA_CAP_WHITEN | PCA_CAP_HOST

    @staticmethod
    def fit(r: PCAFitRequest) raises -> PCAFitResult:
        if r.fold.active():
            raise Error("PCA host provider does not own resident folds")
        var cp = f32_ptr(r.components)
        var mp = f32_ptr(r.mean)
        var ep = f32_ptr(r.variance)
        var rp = f32_ptr(r.ratio)
        var sp = f32_ptr(r.singular)
        var op = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=r.projection)
        var keep = r.count if r.randomized else 0
        var noise = Float64(0)
        var finite = 0
        if r.randomized:
            var word = List[Int32](length=1, fill=0)
            noise = pca_randomized_owned_host_run(r.source, cp, mp, ep, rp, sp,
                r.rows, r.features, r.count, r.oversamples, r.iterations, r.auto_iterations, r.seed,
                op, r.projection != 0, r.whiten, Int(word.unsafe_ptr()) if r.test_finite else 0)
            finite = Int(word[0])
        else:
            if r.test_finite and r.rows >= 1 and r.features >= 1:
                finite = Int(host_finite_status(r.source, r.rows * r.features))
            if finite == 0:
                var fitted = pca_fit_owned_host_run(r.source, cp, mp, ep, rp, sp,
                    r.rows, r.features, r.count, r.dense, r.keep_mode, r.fraction,
                    op, r.projection != 0, r.whiten)
                keep = fitted[0]
                noise = fitted[1]
        return PCAFitResult(keep, noise, finite, 0)

    @staticmethod
    def query(r: PCAQueryRequest) raises -> PCAQueryResult:
        if r.fold.active():
            raise Error("PCA host provider does not own resident folds")
        if r.whiten:
            pca_whiten_dimensions(r)
            var inputs = r.rows * (r.count if r.inverse else r.features)
            var outputs = r.rows * (r.features if r.inverse else r.count)
            pca_whiten_check_host(r.source, r.mean, r.components, r.singular, r.dst,
                inputs, r.features, r.count, outputs)
            var dst = f32_ptr(r.dst)
            pca_whiten_host_run(r.source, r.mean, r.components, r.singular, dst,
                r.rows, r.features, r.count, r.fit_rows, inputs, outputs, r.inverse)
            return PCAQueryResult(0, 0)
        var dst = f32_ptr(r.dst)
        _positive(r.rows, "n_rows")
        _positive(r.features, "n_features")
        _positive(r.count, "n_components")
        if r.inverse:
            # Preserve the existing host inverse's two fitted-state copies.
            var components = read_f32(r.components, r.count * r.features)
            var mean = read_f32(r.mean, r.features)
            host_inverse_transform_into(f32_ptr(r.source), components, mean, dst,
                r.rows, r.features, r.count, True, host_predict_task_count(r.rows))
            return PCAQueryResult(0, 0)
        var finite = Int(host_finite_status(r.source, r.rows * r.features))
        if finite == 0:
            host_pca_transform_into(f32_ptr(r.source), f32_ptr(r.mean), f32_ptr(r.components), dst,
                r.rows, r.features, r.count, host_predict_task_count(r.rows))
        return PCAQueryResult(finite, 0)


def pca_host_native_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    return pca_native_dispatch[PCAHostKernels](operation, aa, na, pp, np, ff, nf, result, context, sink)


struct PCAHostAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(PCA_NATIVE_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(pca_host_native_v1)


comptime PCA_HOST_API = SharedSlot[PCAHostAPI, "MojolearnPCAHostNativeV1"]
