# SPDX-License-Identifier: Apache-2.0
"""KernelDensity typed device provider preserving resident and direct routes."""
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from core.pca_native_api import pca_vendor_code
from core.hostptr import f32_ptr, read_f32
from core.kde_native_api import KDENativeKernels, KDEPrepareRequest, KDEQueryRequest, KDEResidentQuery, KDE_CAP_QUERY, KDE_CAP_RESIDENT, KDE_CAP_IMMUTABLE
from kde.estimator import kde_score_samples_host_ptr
from kde.resident_fit import kde_fit_prepare, kde_fit_release, kde_score_samples_resident, KDE_FAST_IMMUTABLE_FIT


struct KDEDeviceKernels(KDENativeKernels):
    @staticmethod
    def mode() -> Int:
        return Int(GLOBAL_NUMERIC_MODE)

    @staticmethod
    def vendor() -> Int:
        return pca_vendor_code(String(COMPILED_VENDOR))

    @staticmethod
    def capabilities() -> Int:
        return KDE_CAP_QUERY | KDE_CAP_RESIDENT | (KDE_CAP_IMMUTABLE if KDE_FAST_IMMUTABLE_FIT else 0)

    @staticmethod
    def prepare(r: KDEPrepareRequest) raises -> Int:
        var train = f32_ptr(r.train)
        var bandwidth = Float32(r.bandwidth)
        var weights = List[Float32]()
        if r.has_weights:
            var pointer = f32_ptr(r.weights)
            weights = read_f32(Int(pointer), max(0, r.rows))
        return kde_fit_prepare(train, r.rows, r.features, bandwidth, r.kernel, r.metric,
            weights, r.has_weights, r.finite_tested)

    @staticmethod
    def release(handle: Int) raises:
        kde_fit_release(handle)

    @staticmethod
    def query(r: KDEQueryRequest) raises -> Float64:
        if r.want_total:
            raise Error("kde_score_samples: want_total is served by kde_score_samples_resident on this binding")
        var train = f32_ptr(r.train)
        var query = f32_ptr(r.query)
        var dst = f32_ptr(r.dst)
        var bandwidth = Float32(r.bandwidth)
        var weights = List[Float32]()
        if r.has_weights:
            var pointer = f32_ptr(r.weights)
            weights = read_f32(Int(pointer), max(0, r.train_rows))
        kde_score_samples_host_ptr(train, r.train_rows, query, r.query_rows, r.features,
            bandwidth, r.kernel, r.metric, weights, r.has_weights, dst)
        return 0.0

    @staticmethod
    def resident(r: KDEResidentQuery) raises -> Float64:
        if r.status < 0 or r.status % 4 != 0:
            raise Error("kde_score_samples_resident: the status address must be a positive int32-aligned address")
        var query = f32_ptr(r.query)
        var dst = f32_ptr(r.dst)
        var bandwidth = Float32(r.bandwidth)
        var total = kde_score_samples_resident(r.handle, query, r.rows, r.features,
            bandwidth, r.kernel, r.metric, dst, want_total=r.want_total, status_addr=r.status)
        return Float64(total)


from core.shared_slot import SharedSlot
from core.kde_native_api import KDEProviderAPI

comptime _KDE_DEVICE_NAME = "MojolearnKDEDeviceIdenticalV1" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnKDEDeviceFastV1"
comptime KDE_DEVICE_API = SharedSlot[KDEProviderAPI[KDEDeviceKernels], _KDE_DEVICE_NAME]
