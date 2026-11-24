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

# The PACK half of the former `bindings/msel_device.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/msel_device_unpack.mojo`.
from std.python import PythonObject

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.msel_device_x import msel_store
# lane calltable-msel: the store, scatter, column, split-table and
# group-fold bodies are the runs the model-selection helper table publishes
# (core/msel_helpers_device.mojo); these exports unpack and call them.
from core.msel_helpers_device import (
    msel_store_alloc_device_run, msel_store_read_device_run,
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

# Deleted (arena unit E, 2026-11-21): `msel_put` and
# `msel_take_rows` had no caller since the resident owner calls the
# model-selection helper table as typed RUN (core/msel_entry_table.mojo);
# `msel_free` deleted too (lane fx-leftovers item 9, 2026-11-21): it was only
# the resident tests' fault injection; the close / release failure order is
# checked natively (checks/native_tests/arena_state_suite.mojo).
def msel_live_binding() raises -> PythonObject:
    return PythonObject(msel_store()[].live_count())


