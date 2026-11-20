# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The model-selection helpers of the GPU base binding (`bindings/_mojolearn.mojo`)
on the device (lane cpu2-l4-modelsel, 2026-10-04; `core/msel_device.mojo`).

The core host binding exports the same names and signatures from
`bindings/msel_host.mojo` (the host column, and the CPU-only install's route).
Every IDENTICAL and FAST GPU build takes these; there is no host fallback in
this binding.

    msel_put(addr, nbytes) -> id          resident copy of host bytes
    msel_alloc(nbytes) -> id              resident zero bytes
    msel_read(id, dst_addr, nbytes)       download
    msel_free(id) / msel_live() -> count
    msel_take_rows(src_id, n_src, row_bytes, idx_id, n_idx, dst_addr)
    msel_scatter_rows(dst_id, n_dst, dst_code, src_addr, src_code, n_src, width, idx_id)
    msel_proba_column(src_addr, code, n, k, col, dst_addr)
    msel_rebase_offsets_i32(raw_addr, n_raw, lens_addr, parts, n_out, dst_addr)
"""

from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.msel_device import device_take_rows
from core.neural_context import process_ctx
from core.msel_device_x import msel_fold_entry_context, msel_store, MSEL_CONTEXT_SLOT
# lane calltable-msel: the store, scatter, column, split-table and
# group-fold bodies are the runs the model-selection helper table publishes
# (core/msel_helpers_device.mojo); these exports unpack and call them.
from core.msel_helpers_device import (
    msel_store_alloc_device_run, msel_store_read_device_run, msel_store_free_device_run,
    msel_scatter_rows_device_run, msel_proba_column_device_run, msel_split_table_device_run,
    msel_group_fold_perm_device_run, msel_store_free_bytes_device_run,
)
from decomposition.pca_fold import pca_transform_fold, tsvd_fit_fold
from core.pca_native_fold import PCA_FOLD_API
from core.tsvd_native_fold import TSVD_FOLD_API
from glm.logistic_fold_native import LOGISTIC_FOLD_API, logistic_fold_query
from core.logistic_native_api import QnQueryRequest, QnFoldStore, QN_NATIVE_DECISION, QN_NATIVE_BINARY, QN_NATIVE_PROBA
from providers.python.neural_descriptor_pyapi_door import neural_descriptor_capsule
from cluster.kmeans_fold import kmeans_query_fold
from core.msel_device_run import _mdx_f32_ptr, _mdx_u32_ptr, kmeans_fit_fold_run, pca_fit_fold_run

from bindings.msel_device_pack import (
    _kmeans_query_fold_call,
)


def tsvd_fold_native_api_v1_binding() raises -> PythonObject:
    var api = TSVD_FOLD_API.get_or_create_ptr()
    return neural_descriptor_capsule(api[].name, api[].entries)


def pca_fold_native_api_v1_binding() raises -> PythonObject:
    var api = PCA_FOLD_API.get_or_create_ptr()
    return neural_descriptor_capsule(api[].name, api[].entries)


def logistic_fold_native_api_v1_binding() raises -> PythonObject:
    var api = LOGISTIC_FOLD_API.get_or_create_ptr()
    return neural_descriptor_capsule(api[].name, api[].entries)


# qn_fit_fold_binding (the `qn_fit_fold` Python export) was removed by lane
# calltable-solver-estimators (2026-11-11): its one caller, QNRegressor's
# owner (bindings/qn_native.mojo), calls the same body (glm/logistic_fold_native.mojo
# logistic_fold_v1) through the native estimators table
# (core/estimators_entry_table.mojo ESTIMATORS_TABLE_QN_FOLD).


def kmeans_fit_fold_binding(
    ids: PythonObject,
    out_centroids_addr: PythonObject,
    out_labels_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`kmeans_fit` with unit weights over a model-selection fold resident
    on the device (lane a7d-device-x-decomp-cluster; cluster/kmeans_fold.mojo):
    `ids` is [base store id, base rows, index store id] in model selection's
    store (bindings/device_rows_door.mojo door_rows_store); `params` is
    `kmeans_fit_binding`'s first 11 values with n_weights 0 (n_samples,
    n_features, n_clusters, 0, max_iter, tol, seed, n_init, init, metric,
    oversampling_factor). Returns [inertia, n_iter, sum_scale,
    weight_scale, finite word, range word]; a nonzero word means the fold
    was not fit and the door refuses."""
    if len(ids) != 3 or len(params) != 11:
        raise Error("kmeans_fit_fold: ids must hold 3 values and params 11")
    if Int(py=params[3]) != 0:
        raise Error("kmeans_fit_fold: a fold fit takes unit weights (n_weights 0)")
    var base_id = Int(py=ids[0])
    var base_rows = Int(py=ids[1])
    var index_id = Int(py=ids[2])
    var cp = _mdx_f32_ptr(Int(py=out_centroids_addr))
    var lp = _mdx_u32_ptr(Int(py=out_labels_addr))
    var ns = Int(py=params[0])
    var nf = Int(py=params[1])
    var nc = Int(py=params[2])
    var mi = Int(py=params[4])
    var tl = Float64(py=params[5])
    var sd = UInt64(Int(py=params[6]))
    var ninit = Int(py=params[7])
    var ii = Int(py=params[8])
    var mm = Int(py=params[9])
    var ovs = Float64(py=params[10])
    var inertia = Float64(0.0)
    var n_iter = 0
    var sum_scale = Float64(0.0)
    var weight_scale = Float64(0.0)
    var finite = Int32(0)
    var in_range = Int32(0)
    with GILReleased(Python()):
        var r = kmeans_fit_fold_run(base_id, base_rows, index_id, ns, nf, nc, cp, lp,
                                    mi, tl, sd, ninit, ii, mm, ovs)
        inertia = r.fit.inertia
        n_iter = r.fit.n_iter
        sum_scale = r.fit.sum_scale
        weight_scale = r.fit.weight_scale
        finite = r.finite
        in_range = r.in_range
    return Python.list(PythonObject(inertia), PythonObject(n_iter), PythonObject(sum_scale),
                       PythonObject(weight_scale), PythonObject(Int(finite)), PythonObject(Int(in_range)))


def kmeans_predict_fold_binding(
    ids: PythonObject, centroids_addr: PythonObject, out_labels_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """`kmeans_predict` over a model-selection fold resident on the device
    (lane a7d): `ids` as `kmeans_fit_fold_binding`'s, `params` [n_samples,
    n_features, n_clusters, metric]. Returns [finite word, range word]."""
    return _kmeans_query_fold_call(ids, centroids_addr, out_labels_addr, params, False)


def kmeans_transform_fold_binding(
    ids: PythonObject, centroids_addr: PythonObject, out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """`kmeans_transform` over a model-selection fold resident on the device
    (lane a7d): as `kmeans_predict_fold_binding`, `out_addr` n_samples x
    n_clusters float32."""
    return _kmeans_query_fold_call(ids, centroids_addr, out_addr, params, True)
