# SPDX-License-Identifier: Apache-2.0
"""Native KMeans CPU plumbing, retaining the existing host operations."""
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.native_descriptor_abi import NativeDescriptorError
from core.kmeans_native_api import (
    KMeansNativeKernels, KMeansFitRequest, KMeansFitResult, KMeansQueryRequest,
    KMeansQueryResult, kmeans_native_dispatch, KMEANS_CAP_HOST,
)
from core.hostptr import f32_ptr, u32_ptr
from core.host_finite_scan import host_all_finite_f32
from core.core_host_impl import kmeans_fit_host, kmeans_predict_host, kmeans_transform_host


struct KMeansHostKernels(KMeansNativeKernels):
    @staticmethod
    def mode() -> Int:
        return Int(GLOBAL_NUMERIC_MODE)

    @staticmethod
    def capabilities() -> Int:
        return KMEANS_CAP_HOST

    @staticmethod
    def fit(r: KMeansFitRequest) raises -> KMeansFitResult:
        if r.fold.active() or r.distances != 0:
            raise Error("KMeans native ABI: host fit has no resident fold or fused transform")
        var labels = u32_ptr(r.labels)
        var status = 0
        if r.test_finite and r.rows >= 1 and r.features >= 1:
            status = 0 if host_all_finite_f32(f32_ptr(r.source), r.rows * r.features) else 1
        if status != 0:
            return KMeansFitResult(0.0, 0, 0.0, 0.0, status, 0)
        var answer = kmeans_fit_host(r.source, r.centers, labels, r.weights,
            r.rows, r.features, r.clusters, r.weight_count, r.max_iter,
            r.tolerance, UInt64(r.seed), r.n_init, r.init, r.metric, r.oversampling)
        return KMeansFitResult(answer[0], answer[1], answer[2], answer[3], 0, 0)

    @staticmethod
    def query(r: KMeansQueryRequest) raises -> KMeansQueryResult:
        if r.fold.active() or r.handle >= 0:
            raise Error("KMeans native ABI: host query has no resident fold or center handle")
        # The output pointer is admitted before the deferred finite scan.
        if r.transform:
            _ = f32_ptr(r.dst)
        else:
            _ = u32_ptr(r.dst)
        var status = 0
        if r.test_finite and r.rows >= 1 and r.features >= 1:
            status = 0 if host_all_finite_f32(f32_ptr(r.source), r.rows * r.features) else 1
        if status == 0:
            if r.transform:
                kmeans_transform_host(r.source, r.centers, f32_ptr(r.dst),
                    r.rows, r.features, r.clusters, r.metric)
            else:
                kmeans_predict_host(r.source, r.centers, u32_ptr(r.dst),
                    r.rows, r.features, r.clusters, r.metric)
        return KMeansQueryResult(status, 0)

    @staticmethod
    def prepare(centers: Int, clusters: Int, features: Int) raises -> Int:
        raise Error("KMeans native ABI: host provider has no resident center registry")

    @staticmethod
    def release(handle: Int) raises:
        raise Error("KMeans native ABI: host provider has no resident center registry")


def kmeans_host_native_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    return kmeans_native_dispatch[KMeansHostKernels](operation, aa, na, pp, np, ff, nf, result, context, sink)
