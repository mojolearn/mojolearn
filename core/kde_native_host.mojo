# SPDX-License-Identifier: Apache-2.0
"""KernelDensity typed host provider over the unchanged host executor."""
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.hostptr import f32_ptr, read_f32
from core.estimators_host_impl import kde_score_samples_host_run
from core.kde_native_api import KDENativeKernels, KDEPrepareRequest, KDEQueryRequest, KDEResidentQuery, KDE_CAP_QUERY


struct KDEHostKernels(KDENativeKernels):
    @staticmethod
    def mode() -> Int:
        return Int(GLOBAL_NUMERIC_MODE)

    @staticmethod
    def vendor() -> Int:
        return 1

    @staticmethod
    def capabilities() -> Int:
        return KDE_CAP_QUERY

    @staticmethod
    def prepare(request: KDEPrepareRequest) raises -> Int:
        raise Error("KDE host provider does not retain a device fit")

    @staticmethod
    def release(handle: Int) raises:
        raise Error("KDE host provider does not own device handles")

    @staticmethod
    def query(r: KDEQueryRequest) raises -> Float64:
        var train = f32_ptr(r.train)
        var query = f32_ptr(r.query)
        var dst = f32_ptr(r.dst)
        var bandwidth = Float32(r.bandwidth)
        var weights = List[Float32]()
        if r.has_weights:
            weights = read_f32(r.weights, max(0, r.train_rows))
        return kde_score_samples_host_run(train, query, dst, weights, r.has_weights,
            r.train_rows, r.query_rows, r.features, bandwidth, r.kernel, r.metric, r.want_total)

    @staticmethod
    def resident(request: KDEResidentQuery) raises -> Float64:
        raise Error("KDE host provider does not retain a device fit")


from core.shared_slot import SharedSlot
from core.kde_native_api import KDEProviderAPI

comptime _KDE_HOST_NAME = "MojolearnKDEHostIdenticalV1" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnKDEHostFastV1"
comptime KDE_HOST_API = SharedSlot[KDEProviderAPI[KDEHostKernels], _KDE_HOST_NAME]
