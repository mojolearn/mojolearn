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

# The PACK half of the former `bindings/msel_device.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/msel_device_unpack.mojo`.
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

#: The store's own context slot (core/msel_device_x.mojo MSEL_CONTEXT_SLOT,
#: the base binding's `_DEVCTX_SLOT`): one definition, so every reader of the
#: store's buffers runs on the context that made them. The store itself is
#: `msel_store()` there, one process-wide SharedSlot with a layout word.
comptime _MSEL_SLOT = MSEL_CONTEXT_SLOT


def msel_put_binding(addr: PythonObject, nbytes: PythonObject) raises -> PythonObject:
    var a = Int(py=addr)
    var n = Int(py=nbytes)
    var ctx = process_ctx[_MSEL_SLOT]()
    var id: Int
    with GILReleased(Python()):
        id = msel_store()[].put(ctx, a, n)
    return PythonObject(id)


def msel_free_binding(id: PythonObject) raises -> PythonObject:
    """`core/msel_helpers_device.mojo::msel_store_free_device_run`."""
    var i = Int(py=id)
    var status = 0
    with GILReleased(Python()):
        status = msel_store_free_device_run(i)
    return PythonObject(status)


def msel_live_binding() raises -> PythonObject:
    return PythonObject(msel_store()[].live)


def msel_take_rows_binding(
    src_id: PythonObject, n_src: PythonObject, row_bytes: PythonObject,
    idx_id: PythonObject, n_idx: PythonObject, dst_addr: PythonObject,
) raises -> PythonObject:
    """Rows idx of the resident source into host memory (a fold's rows)."""
    var s = Int(py=src_id)
    var ns = Int(py=n_src)
    var rb = Int(py=row_bytes)
    var ix = Int(py=idx_id)
    var ni = Int(py=n_idx)
    var d = Int(py=dst_addr)
    if ns < 0 or rb < 0 or ni < 0:
        raise Error("msel_take_rows: dimensions must be non-negative")
    var ctx = process_ctx[_MSEL_SLOT]()
    var ok = True
    with GILReleased(Python()):
        ok = device_take_rows(ctx, msel_store()[], s, ns, rb, ix, ni, d)
    if not ok:
        raise Error("msel_take_rows: row index out of bounds")
    return PythonObject(0)


# Lane a6: the device-X entries of family and classical estimators live in
# this binding (the identical GPU base binding). The placement is a choice,
# not a need: the fold store is one SharedSlot per process by name, so an
# entry in any extension module reaches it through `msel_fold_store()` and
# runs on `msel_fold_entry_context()` (core/msel_device_x.mojo; lane a8d: the
# build's own context, so a `fast` build's entry runs on the `fast` context).
def _kmeans_query_fold_call(ids: PythonObject, centroids_addr: PythonObject, out_addr: PythonObject,
                            params: PythonObject, transform: Bool) raises -> PythonObject:
    if len(ids) != 3 or len(params) != 4:
        raise Error("kmeans fold query: ids and params must hold 3 and 4 values")
    var base_id = Int(py=ids[0])
    var base_rows = Int(py=ids[1])
    var index_id = Int(py=ids[2])
    var cp = _mdx_f32_ptr(Int(py=centroids_addr))
    var out = Int(py=out_addr)
    _ = _mdx_u32_ptr(out)
    var ns = Int(py=params[0])
    var nf = Int(py=params[1])
    var nc = Int(py=params[2])
    var mm = Int(py=params[3])
    var words: Tuple[Int32, Int32]
    with GILReleased(Python()):
        var ctx = msel_fold_entry_context()
        words = kmeans_query_fold(ctx, base_id, base_rows, index_id, ns, nf, nc, cp, out, mm, transform)
    return Python.list(Int(words[0]), Int(words[1]))
