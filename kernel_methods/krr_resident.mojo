# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The device-resident KernelRidge fit (lane migrate-classical-b2).

`KernelRidge.predict` used to upload `X_fit_` (n x d) and `dual_coef_`
(n x t) on every call. The first predict after a fit (or a load) now
uploads them ONCE through `kernel_ridge_resident_prepare`; the handle lives
in the model's `_resident_holder` with the binding that minted it
(bindings/kernel_methods_native.mojo), and every later predict reads the
device copies (`kernel_ridge_predict_resident`). Released by a refit, a
change of either fitted array, or the native finalizer
(bindings/host_archive_lifetime.mojo "entry" plans).

The registry is `neighbors/resident_index.mojo`'s (DEVIATION 2921): one
process-wide `_Global` keyed by an increasing handle, the caller holding the
GIL across prepare, predict and release; an entry owns its context and
buffers, the buffers destroyed first (DEVIATION 1946).

WHAT DOES NOT CHANGE. The upload is `_upload_ptr`, the pointer route's
(KRR_IDN_PTR_IN), and the predict is `kernel_ridge_predict_dev`, the body
`kernel_ridge_predict_ptr_into` runs after its own uploads of the same
bytes: the same launches in the same order, so the bits are the same.
"""
from std.ffi import _Global

from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.identity_trace import IdentityTrace
from svm.impl.svm_parameter import KernelParams
from kernel_methods.estimator import (
    _family_ctx,
    _upload_checked,
    _upload_ptr,
    kernel_ridge_predict_dev,
)


struct ResidentKernelRidge(Movable):
    """One fit's `X_fit_` (n x d) and `dual_coef_` (n x t) on the device."""

    var ctx: DeviceContext
    var x_fit: DeviceBuffer[DType.float32]
    var dual: DeviceBuffer[DType.float32]
    var n: Int
    var d: Int
    var t: Int

    def __init__(out self, xfit_addr: Int, dual_addr: Int, n: Int, d: Int, t: Int) raises:
        if xfit_addr == 0 or dual_addr == 0:
            raise Error("kernel_ridge_resident_prepare: null X_fit or dual address")
        if n <= 0 or d <= 0 or t <= 0:
            raise Error(
                "kernel_ridge_predict: the model needs positive n, d and n_targets, got "
                + String(n) + ", " + String(d) + ", " + String(t)
            )
        var ctx = _family_ctx()
        var x_fit = _upload_ptr(ctx, xfit_addr, n * d)
        var dual = _upload_ptr(ctx, dual_addr, n * t)
        self.n = n
        self.d = d
        self.t = t
        self.x_fit = x_fit^
        self.dual = dual^
        self.ctx = ctx^

    def __deinit__(deinit self):
        # The buffers before the context they were created on (DEVIATION 1946).
        _ = self.dual^
        _ = self.x_fit^
        _ = self.ctx^


struct KernelRidgeResidentRegistry(Movable):
    var entries: Dict[Int, ResidentKernelRidge]
    var next_id: Int

    def __init__(out self):
        self.entries = Dict[Int, ResidentKernelRidge]()
        self.next_id = 1


comptime KRR_RESIDENT_REGISTRY = _Global[
    StorageType=KernelRidgeResidentRegistry,
    name=(
        "MojoKernelRidgeResidentIdentical" if GLOBAL_NUMERIC_MODE == 1 else
        "MojoKernelRidgeResidentFast"
    ),
    init_fn=KernelRidgeResidentRegistry.__init__,
]


def kernel_ridge_resident_prepare(xfit_addr: Int, dual_addr: Int, n: Int, d: Int, t: Int) raises -> Int:
    """Upload the fitted arrays once; the handle every later predict names."""
    var state = KRR_RESIDENT_REGISTRY.get_or_create_ptr()
    var entry = ResidentKernelRidge(xfit_addr, dual_addr, n, d, t)
    if state[].next_id == 9223372036854775807:
        raise Error("resident KernelRidge handle space exhausted")
    var handle = state[].next_id
    state[].next_id += 1
    state[].entries[handle] = entry^
    return handle


def kernel_ridge_resident_release(handle: Int) raises:
    """Drop the device copies; a released handle is refused by every later call."""
    var state = KRR_RESIDENT_REGISTRY.get_or_create_ptr()
    if handle not in state[].entries:
        raise Error("unknown or released resident KernelRidge handle")
    var released = state[].entries.pop(handle)
    _ = released^


def kernel_ridge_predict_resident[out_origin: MutOrigin, //](
    handle: Int,
    xnew_addr: Int,
    n: Int,
    d: Int,
    t: Int,
    kp: KernelParams,
    n_query: Int,
    output: MutPointer[Float32, out_origin],
    mut trace: IdentityTrace,
) raises:
    """`kernel_ridge_predict_ptr_into` over the handle's fitted arrays: the
    query checked and uploaded (`_upload_checked`), then
    `kernel_ridge_predict_dev`."""
    if xnew_addr == 0:
        raise Error("kernel_ridge_predict: null X_fit, dual or X address")
    if n <= 0 or d <= 0 or t <= 0:
        raise Error(
            "kernel_ridge_predict: the model needs positive n, d and n_targets, got "
            + String(n) + ", " + String(d) + ", " + String(t)
        )
    var state = KRR_RESIDENT_REGISTRY.get_or_create_ptr()
    if handle not in state[].entries:
        raise Error("unknown or released resident KernelRidge handle")
    ref entry = state[].entries[handle]
    if entry.n != n or entry.d != d or entry.t != t:
        raise Error(
            "kernel_ridge_predict_resident: the handle holds a " + String(entry.n) + " x " + String(entry.d)
            + " fit with " + String(entry.t) + " targets, the call names " + String(n) + " x "
            + String(d) + " with " + String(t)
        )
    var dq = _upload_checked(entry.ctx, xnew_addr, n_query, d, "predict X")
    kernel_ridge_predict_dev(entry.ctx, dq, entry.x_fit, entry.dual, n, d, t, kp, n_query, output, trace)
    _ = dq^
