# SPDX-License-Identifier: Apache-2.0
"""Typed DBSCAN device provider over the existing fit/gather/reference kernels."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.vendor import COMPILED_VENDOR
from core.pca_native_api import pca_vendor_code
from core.neural_context import process_ctx
from core.hostptr import f32_ptr, i32_ptr
from core.shared_slot import SharedSlot
from core.dbscan_native_api import (
    DBSCANNativeKernels, DBSCANProviderAPI, DBSCANFitRequest, DBSCANCoreRequest, DBSCANGatherRequest,
    DBSCANReferenceRequest, DBSCANQueryRequest, DBSCANResidentQuery, DBSCAN_CAP_RESIDENT, DBSCAN_CAP_DEV_STORE,
)
from dbscan.estimator import dbscan_fit, dbscan_fit_resident
from dbscan.core_device_impl import dbscan_core_count, dbscan_core_gather, dbscan_core_gather_resident
from core.estimators_store import estimators_store_buffer
from core.labeled_reference_predict import labeled_reference_predict
from core.labeled_reference_resident import labeled_reference_prepare, labeled_reference_release, labeled_reference_predict_resident

# Exactly the estimator image's existing owning context/root policy.
comptime _DEVCTX_SLOT = "MojolearnEstimatorsContextIdenticalV3" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnEstimatorsContextFastV3"


struct DBSCANDeviceKernels(DBSCANNativeKernels):
    @staticmethod
    def mode() -> Int:
        return Int(GLOBAL_NUMERIC_MODE)

    @staticmethod
    def vendor() -> Int:
        return pca_vendor_code(String(COMPILED_VENDOR))

    @staticmethod
    def capabilities() -> Int:
        # DBSCAN_CAP_DEV_STORE (lane vdf-scope-cluster4): the image's resident store
        # (core/estimators_store.mojo, shared with KernelDensity) and the two store forms below.
        return DBSCAN_CAP_RESIDENT | DBSCAN_CAP_DEV_STORE

    @staticmethod
    def fit(r: DBSCANFitRequest) raises -> Int:
        var x = f32_ptr(r.x)
        var labels = i32_ptr(r.labels)
        var ctx = process_ctx[_DEVCTX_SLOT]()
        var passes = dbscan_fit(ctx, x, r.rows, r.features, r.eps, r.min_samples, labels,
            r.budget_mb, r.max_iter, r.method, r.metric, r.weights, r.core)
        ctx.synchronize()
        return passes

    @staticmethod
    def fit_store(r: DBSCANFitRequest) raises -> Int:
        """`fit` over the rows in slot `r.x` of the resident store (no upload; cosine scales a device copy)."""
        var labels = i32_ptr(r.labels)
        var ctx = process_ctx[_DEVCTX_SLOT]()
        var rows = estimators_store_buffer(r.x, r.rows * r.features)
        var passes = dbscan_fit_resident(ctx, rows, r.rows, r.features, r.eps, r.min_samples, labels,
            r.budget_mb, r.max_iter, r.method, r.metric, r.weights, r.core)
        ctx.synchronize()
        _ = rows^
        return passes

    @staticmethod
    def core_gather_store(r: DBSCANGatherRequest) raises -> Int:
        """`core_gather` reading X from slot `r.x` of the resident store (no upload)."""
        var ctx = process_ctx[_DEVCTX_SLOT]()
        if r.rows < 0 or r.features < 1:
            raise Error("dbscan_core_arrays: n >= 0 and d >= 1")
        var rows = estimators_store_buffer(r.x, max(1, r.rows * r.features))
        dbscan_core_gather_resident(ctx, rows, r.labels, r.core, r.rows, r.features, r.count,
            r.indices, r.components, r.core_labels)
        _ = rows^
        return 0

    @staticmethod
    def core_count(r: DBSCANCoreRequest) raises -> Int:
        var ctx = process_ctx[_DEVCTX_SLOT]()
        if r.rows < 0 or r.features < 1 or (r.rows > 0 and r.features > 2147483000 // r.rows):
            raise Error("dbscan_core_arrays: n >= 0, d >= 1 and n * d within Int32")
        if r.rows == 0:
            return 0
        if r.x == 0 or r.labels == 0 or r.core == 0:
            raise Error("dbscan_core_arrays: null buffer address")
        return dbscan_core_count(ctx, r.core, r.rows)

    @staticmethod
    def core_gather(r: DBSCANGatherRequest) raises -> Int:
        var ctx = process_ctx[_DEVCTX_SLOT]()
        dbscan_core_gather(ctx, r.x, r.labels, r.core, r.rows, r.features, r.count,
            r.indices, r.components, r.core_labels)
        return 0

    @staticmethod
    def prepare(r: DBSCANReferenceRequest) raises -> Int:
        var refs = f32_ptr(r.refs)
        var keys = i32_ptr(r.keys)
        var labels = i32_ptr(r.labels)
        return labeled_reference_prepare(refs, r.rows, keys, labels, r.features, r.metric)

    @staticmethod
    def release(handle: Int) raises -> Int:
        labeled_reference_release(handle)
        return 0

    @staticmethod
    def query(r: DBSCANQueryRequest) raises -> Int:
        var refs = f32_ptr(r.refs)
        var keys = i32_ptr(r.keys)
        var labels = i32_ptr(r.labels)
        var query = f32_ptr(r.query)
        var dst = i32_ptr(r.dst)
        var chosen = i32_ptr(r.chosen)
        var ctx = process_ctx[_DEVCTX_SLOT]()
        labeled_reference_predict(ctx, refs, r.rows, keys, labels, query, r.queries, r.features,
            r.metric, r.eps, r.has_threshold, dst, chosen)
        ctx.synchronize()
        return 0

    @staticmethod
    def resident(r: DBSCANResidentQuery) raises -> Int:
        var query = f32_ptr(r.query)
        var dst = i32_ptr(r.dst)
        labeled_reference_predict_resident(r.handle, query, r.queries, r.features,
            r.metric, r.eps, r.has_threshold, dst)
        return 0

    @staticmethod
    def resident_store(r: DBSCANResidentQuery) raises -> Int:
        """`resident` over the query rows in slot `r.query` of the resident store (one device copy into the
        reference set's context, no upload; lane vdf-scope-cluster6)."""
        var dst = i32_ptr(r.dst)
        var rows = estimators_store_buffer(r.query, max(1, r.queries * r.features))
        # The host query pointer is not read on the store form: the destination's address stands in for it.
        labeled_reference_predict_resident(r.handle, f32_ptr(r.dst), r.queries, r.features,
            r.metric, r.eps, r.has_threshold, dst, Optional(rows))
        _ = rows^
        return 0


comptime _DBSCAN_DEVICE_NAME = "MojolearnDBSCANDeviceIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnDBSCANDeviceFastV1"
comptime DBSCAN_DEVICE_API = SharedSlot[DBSCANProviderAPI[DBSCANDeviceKernels], _DBSCAN_DEVICE_NAME]
