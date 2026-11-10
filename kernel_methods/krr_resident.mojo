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
process-wide `_Global` keyed by an increasing handle; an entry owns its
context and buffers, the buffers destroyed first (DEVIATION 1946).

THREADS (lane p3-svm-gp-kernel, hazard class B). Prepare and predict run
with the GIL released (bindings/_mojolearn_kernel_methods.mojo and the
descriptor route), so a first predict of another model can insert (a
rehash moves entries) or a refit can release a handle while a predict
reads its entry. Every registry read and write now takes the registry's
lock word for one dictionary step only (the pattern of
cluster/resident_centers.mojo): prepare uploads outside the lock and then
inserts, release pops under the lock and destroys outside it, and predict
copies the entry's context and buffer HANDLES (reference counted, the
memory is shared) under the lock, so a release while a predict runs drops
only the registry's references.

WHAT DOES NOT CHANGE. The upload is `_upload_ptr`, the pointer route's
(KRR_IDN_PTR_IN), and the predict is `kernel_ridge_predict_dev`, the body
`kernel_ridge_predict_ptr_into` runs after its own uploads of the same
bytes: the same launches in the same order, so the bits are the same.
"""
from std.atomic import Atomic, Ordering

from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.identity_trace import IdentityTrace
from core.device_query_x import DeviceQueryX, QueryFold
from svm.impl.svm_parameter import KernelParams
from kernel_methods.estimator import (
    _family_ctx,
    _upload_checked,
    _upload_ptr,
    kernel_ridge_predict_dev,
)
from core.shared_slot import SharedSlot


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


struct KernelRidgeResidentRegistry(Defaultable, Movable):
    var entries: Dict[Int, ResidentKernelRidge]
    var next_id: Int
    # One lock word: prepare and predict run with the GIL released (see the module note).
    var lock: List[Int64]

    def __init__(out self):
        self.entries = Dict[Int, ResidentKernelRidge]()
        self.next_id = 1
        self.lock = List[Int64](length=1, fill=Int64(0))


comptime KRR_RESIDENT_REGISTRY = SharedSlot[KernelRidgeResidentRegistry, ( "MojolearnKernelRidgeResidentIdenticalV1" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnKernelRidgeResidentFastV1" )]


def _krr_registry() raises -> UnsafePointer[KernelRidgeResidentRegistry, MutAnyOrigin]:
    return KRR_RESIDENT_REGISTRY.get_or_create_ptr().unsafe_origin_cast[MutAnyOrigin]()


def _krr_lock(state: UnsafePointer[KernelRidgeResidentRegistry, MutAnyOrigin]) raises:
    var word = rebind[MutPointer[Int64, MutUntrackedOrigin]](state[].lock.unsafe_ptr())
    while True:  # driver-loop: spin on the registry's lock word (held for one dictionary step only)
        var expected = Int64(0)
        if Atomic.compare_exchange[
            success_ordering=Ordering.SEQUENTIAL,
            failure_ordering=Ordering.RELAXED,
        ](word, expected, Int64(1)):
            return


def _krr_unlock(state: UnsafePointer[KernelRidgeResidentRegistry, MutAnyOrigin]):
    var word = rebind[MutPointer[Int64, MutUntrackedOrigin]](state[].lock.unsafe_ptr())
    Atomic.store[ordering=Ordering.RELEASE](word, Int64(0))


struct _KrrResidentView(Movable):
    """A predict's own copies of one entry's context and buffer HANDLES
    (reference counted, the memory is shared) and its shape words."""

    var ctx: DeviceContext
    var x_fit: DeviceBuffer[DType.float32]
    var dual: DeviceBuffer[DType.float32]
    var n: Int
    var d: Int
    var t: Int

    def __init__(out self, var ctx: DeviceContext, var x_fit: DeviceBuffer[DType.float32],
                 var dual: DeviceBuffer[DType.float32], n: Int, d: Int, t: Int):
        self.ctx = ctx^
        self.x_fit = x_fit^
        self.dual = dual^
        self.n = n
        self.d = d
        self.t = t

    def __deinit__(deinit self):
        # The buffers before the context they were created on (DEVIATION 1946).
        _ = self.dual^
        _ = self.x_fit^
        _ = self.ctx^


def kernel_ridge_resident_prepare(xfit_addr: Int, dual_addr: Int, n: Int, d: Int, t: Int) raises -> Int:
    """Upload the fitted arrays once; the handle every later predict names."""
    var state = _krr_registry()
    # The upload runs outside the lock; only the dictionary step is locked.
    var entry = ResidentKernelRidge(xfit_addr, dual_addr, n, d, t)
    _krr_lock(state)
    try:
        if state[].next_id == 9223372036854775807:
            raise Error("resident KernelRidge handle space exhausted")
        var handle = state[].next_id
        state[].next_id += 1
        state[].entries[handle] = entry^
        _krr_unlock(state)
        return handle
    except error:
        _krr_unlock(state)
        raise error^


def _krr_locked_pop(state: UnsafePointer[KernelRidgeResidentRegistry, MutAnyOrigin],
                    handle: Int) raises -> ResidentKernelRidge:
    _krr_lock(state)
    try:
        if handle not in state[].entries:
            raise Error("unknown or released resident KernelRidge handle")
        var popped = state[].entries.pop(handle)
        _krr_unlock(state)
        return popped^
    except error:
        _krr_unlock(state)
        raise error^


def kernel_ridge_resident_release(handle: Int) raises:
    """Drop the device copies; a released handle is refused by every later call."""
    var state = _krr_registry()
    var released = _krr_locked_pop(state, handle)
    # Destroyed outside the lock: a predict still running holds its own
    # handle copies of the context and the buffers.
    _ = released^


def _krr_locked_view(state: UnsafePointer[KernelRidgeResidentRegistry, MutAnyOrigin],
                     handle: Int) raises -> _KrrResidentView:
    _krr_lock(state)
    try:
        if handle not in state[].entries:
            raise Error("unknown or released resident KernelRidge handle")
        ref entry = state[].entries[handle]
        var view = _KrrResidentView(entry.ctx.copy(), entry.x_fit.copy(), entry.dual.copy(),
                                    entry.n, entry.d, entry.t)
        _krr_unlock(state)
        return view^
    except error:
        _krr_unlock(state)
        raise error^


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
) raises -> Tuple[Int32, Int32]:
    """`kernel_ridge_predict_ptr_into` over the handle's fitted arrays: the
    query uploaded with its finite status (core/device_query_x.mojo, lane
    a7a), then `kernel_ridge_predict_dev`. Returns the query's (finite word,
    range word); the door refuses a flagged query. A cross-validation fold
    takes `kernel_ridge_predict_ptr_into` (its context is the fold store's,
    not the handle's)."""
    if xnew_addr == 0:
        raise Error("kernel_ridge_predict: null X_fit, dual or X address")
    if n <= 0 or d <= 0 or t <= 0:
        raise Error(
            "kernel_ridge_predict: the model needs positive n, d and n_targets, got "
            + String(n) + ", " + String(d) + ", " + String(t)
        )
    var state = _krr_registry()
    var view = _krr_locked_view(state, handle)
    if view.n != n or view.d != d or view.t != t:
        raise Error(
            "kernel_ridge_predict_resident: the handle holds a " + String(view.n) + " x " + String(view.d)
            + " fit with " + String(view.t) + " targets, the call names " + String(n) + " x "
            + String(d) + " with " + String(t)
        )
    # TOMBSTONE: _upload_checked on the predict query (synchronising scan, DEVIATION 1686 text) removed 2026-11-09 by lane a7a-device-x-svm-gp-krr; recoverable at 5b5490960d (docs/TOMBSTONES.md).
    var qx = DeviceQueryX(view.ctx, xnew_addr, n_query, d, QueryFold.none())
    # kernel_ridge_predict_dev synchronises before its read-back: the status
    # words copied after the query are valid below.
    kernel_ridge_predict_dev(view.ctx, qx.x, view.x_fit, view.dual, n, d, t, kp, n_query, output, trace)
    var words = (qx.finite_word(), qx.range_word())
    _ = qx^
    _ = view^
    return words
