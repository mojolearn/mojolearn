# SPDX-License-Identifier: Apache-2.0
"""Typed DBSCAN host provider; the existing host algorithms are unchanged."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.hostptr import i32_ptr
from core.shared_slot import SharedSlot
from core.dbscan_native_api import (
    DBSCANNativeKernels, DBSCANProviderAPI, DBSCANFitRequest, DBSCANCoreRequest, DBSCANGatherRequest,
    DBSCANReferenceRequest, DBSCANQueryRequest, DBSCANResidentQuery,
)
from core.estimators_host_impl import dbscan_fit_host_run, labeled_reference_predict_host_run
from core.py2mojo_cluster_est_impl import dbscan_core_count, dbscan_core_gather


struct DBSCANHostKernels(DBSCANNativeKernels):
    @staticmethod
    def mode() -> Int:
        return Int(GLOBAL_NUMERIC_MODE)

    @staticmethod
    def vendor() -> Int:
        return 1

    @staticmethod
    def capabilities() -> Int:
        return 0

    @staticmethod
    def fit(r: DBSCANFitRequest) raises -> Int:
        var labels = i32_ptr(r.labels)
        if r.budget_mb != 0:
            raise Error("mojolearn: no CPU implementation of DBSCAN.fit with"
                + " max_mbytes_per_batch yet; the host runs one batch of every row"
                + " and does not size a device it does not have")
        return dbscan_fit_host_run(r.x, labels, r.weights, r.core, r.rows, r.features,
            r.eps, r.min_samples, r.max_iter, r.method, r.metric)

    @staticmethod
    def core_count(r: DBSCANCoreRequest) raises -> Int:
        return dbscan_core_count(MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=r.core), r.rows)

    @staticmethod
    def core_gather(r: DBSCANGatherRequest) raises -> Int:
        if r.count > 0:
            dbscan_core_gather(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=r.x),
                MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=r.labels),
                MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=r.core), r.rows, r.features,
                MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=r.indices),
                MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=r.components),
                MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=r.core_labels))
        return 0

    @staticmethod
    def prepare(r: DBSCANReferenceRequest) raises -> Int:
        raise Error("DBSCAN host provider does not retain device reference sets")

    @staticmethod
    def release(handle: Int) raises -> Int:
        raise Error("DBSCAN host provider does not own device reference handles")

    @staticmethod
    def query(r: DBSCANQueryRequest) raises -> Int:
        var dst = i32_ptr(r.dst)
        var chosen = i32_ptr(r.chosen)
        labeled_reference_predict_host_run(r.refs, r.keys, r.labels, r.query, dst, chosen,
            r.rows, r.queries, r.features, r.metric, r.eps, r.has_threshold)
        return 0

    @staticmethod
    def resident(r: DBSCANResidentQuery) raises -> Int:
        raise Error("DBSCAN host provider does not retain device reference sets")


comptime _DBSCAN_HOST_NAME = "MojolearnDBSCANHostIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnDBSCANHostFastV1"
comptime DBSCAN_HOST_API = SharedSlot[DBSCANProviderAPI[DBSCANHostKernels], _DBSCAN_HOST_NAME]
