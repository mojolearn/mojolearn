# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RUN pieces of `bindings/_mojolearn_estimators.mojo`: the binding's device
context slot, the typed address casts, the PCA whitening dimension checks and
device call, and the native entry-table publications at module init. Typed
Mojo only, no Python import (tools/python_free_dirs.sh). The door keeps the
argument reads and releases the GIL around `_pca_whiten_device` exactly where
it did before; the bodies are moved unchanged, so the bits are unchanged
(lane rs-metrics-estimators, 2026-11-12)."""
from core.linear_entry_table import LinearEntryKey, linear_table_publish, LINEAR_TABLE_ESTIMATORS
from core.linear_fit_types import LINEAR_FIT_ABI, LINEAR_FIT_OPERATION_COUNT
from core.linear_boundary_types import LINEAR_PREDICT_ABI
from glm.linear_fit_device import linear_fit_device_v1, LINEAR_FIT_DEVICE_API
from glm.linear_predict_device import linear_predict_device_v1
from glm.logistic_native_device import LOGISTIC_DEVICE_API, qn_device_native_v1
from core.estimators_entry_table import EstimatorsEntryKey, estimators_table_publish, estimators_vendor_word, ESTIMATORS_TABLE_QN
from core.hostptr import f32_ptr, f64_ptr, i32_ptr
from core.neural_context import process_ctx
from decomposition.estimator import pca_whiten_inverse_transform_host, pca_whiten_transform_host
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.numerics import GLOBAL_NUMERIC_MODE as _DEVCTX_MODE, NUMERIC_IDENTICAL as _DEVCTX_IDENTICAL
from checks.vendor import COMPILED_VENDOR

#: This binding's ONE process-lifetime DeviceContext (core/neural_context.mojo,
#: lane/devctx-lifetime): a context per call exhausts Metal command queues.
comptime _DEVCTX_SLOT = "MojolearnEstimatorsContextIdenticalV3" if _DEVCTX_MODE == _DEVCTX_IDENTICAL else "MojolearnEstimatorsContextFastV3"


def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    return f32_ptr(addr)


def _i32_ptr(addr: Int) raises -> MutPointer[Int32, MutUntrackedOrigin]:
    return i32_ptr(addr)


def _f64_ptr(addr: Int) raises -> MutPointer[Float64, MutUntrackedOrigin]:
    return f64_ptr(addr)


def _pca_whiten_pointer(
    address: Int, count: Int,
) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    if address <= 0 or address % 4 != 0:
        raise Error("PCA whitening requires positive aligned FP32 pointers")
    if address > 9223372036854775807 - count * 4:
        raise Error("PCA whitening pointer span overflows signed Int")
    return _f32_ptr(address)


def _pca_whiten_dims(nr: Int, nf: Int, nc: Int, nfit: Int) raises:
    if nr < 1 or nf < 1 or nc < 1 or nc > nf or nfit < 2:
        # one feature admitted with the one-column PCA fit (lane sk-kernels)
        raise Error("PCA whitening requires positive rows/features/components and fit_rows>=2")
    if nr > 2147483647 or nf > 2147483647 or nc > 2147483647 or nfit > 2147483647:
        raise Error("PCA whitening dimensions exceed Int32")
    if nr > 2147483647 // nf or nc > 2147483647 // nf:
        raise Error("PCA whitening matrix cell count exceeds Int32")


def _pca_whiten_device(
    xp: MutPointer[Float32, MutUntrackedOrigin],
    mp: MutPointer[Float32, MutUntrackedOrigin],
    cp: MutPointer[Float32, MutUntrackedOrigin],
    sp: MutPointer[Float32, MutUntrackedOrigin],
    op: MutPointer[Float32, MutUntrackedOrigin],
    nr: Int, nf: Int, nc: Int, nfit: Int, inverse: Bool,
) raises:
    # Called with the GIL released by the door (bindings/_mojolearn_estimators.mojo _pca_whiten_apply).
    var ctx = process_ctx[_DEVCTX_SLOT]()
    if inverse:
        pca_whiten_inverse_transform_host(
            ctx, xp, cp, sp, mp, op, nr, nf, nc, nfit, device_checks=True
        )
    else:
        pca_whiten_transform_host(
            ctx, xp, mp, cp, sp, op, nr, nf, nc, nfit, device_checks=True
        )
    ctx.synchronize()


def _publish_linear_entry_table() raises:
    # Native publication only; the existing capsule ABI remains for clients
    # not yet converted. The original descriptor owner has image lifetime.
    var api = LINEAR_FIT_DEVICE_API.get_or_create_ptr()
    var key = LinearEntryKey(GLOBAL_NUMERIC_MODE, (1 if COMPILED_VENDOR == "metal" else 2 if COMPILED_VENDOR == "cuda" else 3 if COMPILED_VENDOR == "hip" else -1))
    linear_table_publish[LINEAR_TABLE_ESTIMATORS](key,
        Int(api[].entries.unsafe_ptr()), [String(LINEAR_FIT_ABI), String(LINEAR_PREDICT_ABI)], [LINEAR_FIT_OPERATION_COUNT, 1], [linear_fit_device_v1, linear_predict_device_v1])


def _publish_estimators_qn_table() raises:
    # The QN entry of the estimators table (core/estimators_entry_table.mojo):
    # QNRegressor's owner (bindings/qn_native.mojo) calls it instead of the
    # `qn_fit` / `qn_decision_function` Python exports, which remain for the
    # other doors and the Python tests that call the binding directly. A
    # build with no GPU vendor ("none") publishes nothing.
    var word = estimators_vendor_word(String(COMPILED_VENDOR))
    if word < 0:
        return
    var api = LOGISTIC_DEVICE_API.get_or_create_ptr()
    estimators_table_publish[ESTIMATORS_TABLE_QN](EstimatorsEntryKey(Int(GLOBAL_NUMERIC_MODE), word),
        Int(api[].entries.unsafe_ptr()), qn_device_native_v1)
