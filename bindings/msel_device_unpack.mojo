# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The model-selection helpers of the GPU base binding (`bindings/_mojolearn.mojo`)
on the device (lane cpu2-l4-modelsel, 2026-10-04; `core/msel_device.mojo`).

The core host binding exports the same names and signatures from
`bindings/msel_host.mojo` (the host column, and the CPU-only install's route).
Every IDENTICAL and FAST GPU build takes these; there is no host fallback in
this binding.

    msel_alloc(nbytes) -> id              resident zero bytes
    msel_read(id, dst_addr, nbytes)       download
    msel_live() -> count                  (live store ids, for the resident tests)
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
    msel_group_fold_perm_device_run,
)
from decomposition.pca_fold import pca_transform_fold, tsvd_fit_fold
from core.pca_native_fold import PCA_FOLD_API
from core.tsvd_native_fold import TSVD_FOLD_API
from glm.logistic_fold_native import LOGISTIC_FOLD_API, logistic_fold_query
from core.logistic_native_api import QnQueryRequest, QnFoldStore, QN_NATIVE_DECISION, QN_NATIVE_BINARY, QN_NATIVE_PROBA
from providers.python.neural_descriptor_pyapi_door import neural_descriptor_capsule
from core.msel_device_run import _mdx_f32_ptr, pca_fit_fold_run


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
