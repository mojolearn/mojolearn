# SPDX-License-Identifier: Apache-2.0
"""KernelDensity typed device provider preserving resident and direct routes."""
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from core.pca_native_api import pca_vendor_code
from core.hostptr import f32_ptr, read_f32
from core.kde_native_api import KDENativeKernels, KDEPrepareRequest, KDEQueryRequest, KDEResidentQuery, KDE_CAP_QUERY, KDE_CAP_RESIDENT
from core.kde_native_api import KDE_CAP_DEV_STORE
from core.estimators_store import estimators_store_buffer
from kde.estimator import kde_score_samples_host_ptr
from kde.resident_fit import kde_fit_prepare, kde_fit_release, kde_score_samples_resident
from kde.resident_fit import kde_fit_prepare_device, kde_score_samples_resident_device


struct KDEDeviceKernels(KDENativeKernels):
    @staticmethod
    def mode() -> Int:
        return Int(GLOBAL_NUMERIC_MODE)

    @staticmethod
    def vendor() -> Int:
        return pca_vendor_code(String(COMPILED_VENDOR))

    @staticmethod
    def capabilities() -> Int:
        # TOMBSTONE: MOJOLEARN_KDE_FAST_IMMUTABLE_FIT (opt-in arm, no A/B recorded) deleted 2026-11-15 by lane/expdel2-04-base; code recoverable at 6b64b91b3f.
        # Restore: git apply experiments/removed/MOJOLEARN_KDE_FAST_IMMUTABLE_FIT.expdel2-04-base.patch; record in docs/TOMBSTONES.md.
        # KDE_CAP_DEV_STORE (lane vdf-scope-cluster2b): the image's resident store
        # (core/estimators_store.mojo) and the two resident-store forms below.
        return KDE_CAP_QUERY | KDE_CAP_RESIDENT | KDE_CAP_DEV_STORE

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
    def prepare_store(r: KDEPrepareRequest) raises -> Int:
        """`prepare` over training rows in slot `r.train` of the resident store (a device-to-device copy)."""
        var weights = List[Float32]()
        if r.has_weights:
            weights = read_f32(r.weights, max(0, r.rows))
        var train = estimators_store_buffer(r.train, r.rows * r.features)
        var handle = kde_fit_prepare_device(train, r.rows, r.features, Float32(r.bandwidth), r.kernel, r.metric,
            weights, r.has_weights)
        _ = train^
        return handle

    @staticmethod
    def resident_store(r: KDEResidentQuery) raises -> Float64:
        """`resident` over queries in slot `r.query` of the resident store (no upload)."""
        if r.status < 0 or r.status % 4 != 0:
            raise Error("kde_score_samples_resident: the status address must be a positive int32-aligned address")
        var query = estimators_store_buffer(r.query, r.rows * r.features)
        var total = kde_score_samples_resident_device(r.handle, query, r.rows, r.features, Float32(r.bandwidth),
            r.kernel, r.metric, f32_ptr(r.dst), want_total=r.want_total, status_addr=r.status)
        _ = query^
        return Float64(total)

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
