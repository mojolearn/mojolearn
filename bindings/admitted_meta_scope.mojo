# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Holding a meta-estimator's admitted matrices for its members (lane
# s1-shared-state-hardening-and-selector, S4; design note
# ~/mojolearn-evidence/design-shared-process-state.md section 2: the
# admitted registry is one per process, so a member door in ANY binding
# reuses what the meta door holds).
#
#   admitted_hold_array_f32(array, scan) -> Bool
#                     hold a 2-D non-empty float32 C-order mojolearn Array in
#                     the calling thread's open scope; `scan` runs the one
#                     host finite scan first (an unknown matrix), without it
#                     the caller vouches the words are finite (rows or
#                     columns gathered out of a matrix already admitted
#                     finite). Returns True when the array is held.
#   admitted_hold_array_f32_deferred(array) -> Bool
#                     hold it with its finite test DEFERRED (lane s2): no
#                     host pass now; a member whose kernel entry tests the
#                     words (defer_finite) never needs one, and the first
#                     member that needs a host verdict runs it once
#   admitted_held_verify(index) -> Bool
#                     the host verdict of a held row: True at once when it
#                     was tested; else the one scan of the oldest untested
#                     matrix on its parent chain (fold rows gathered out of a
#                     finite X are finite), recorded on every row it covers
#
# The caller passes only exact mojolearn Arrays (its own context's
# `array_class` test): an ndarray or a DeviceRows handle is never held here,
# so its member door admits it as before. A non-finite matrix is never held:
# each member refuses or accepts it as its own door does (a forest with a
# missing-value arm keeps accepting it). Numerics: none.
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from bindings.admitted_registry import (
    admitted_hold, admitted_hold_deferred, admitted_is_held, admitted_scope_open, admitted_held_index,
    admitted_held_finite, admitted_held_snapshot, admitted_held_apply,
)
from core.admitted_held_plan import HeldVerification, admitted_holdable, admitted_held_scan, admitted_held_verify_native


def _admitted_hold_array(array: PythonObject, scan: Bool, deferred: Bool) raises -> Bool:
    if not admitted_scope_open():
        return False
    if admitted_is_held(array):
        return True
    var rank = Int(py=array.ndim)
    var rows = Int(py=array.shape[0]) if rank > 0 else 1
    var cols = Int(py=array.shape[1]) if rank > 1 else 1
    if not admitted_holdable(rank, String(py=array.dtype), String(py=array.order), rows, cols):
        return False
    var address = Int(py=array._addr)
    var finite = True
    if scan:
        try:
            with GILReleased(Python()):
                finite = admitted_held_scan(address, rows, cols)
        finally:
            _ = array
    if not finite:
        return False
    if deferred:
        admitted_hold_deferred(array, address, rows, cols, PythonObject(None))
    else:
        admitted_hold(array, address, rows, cols)
    return True


def admitted_hold_array_f32(array: PythonObject, scan: Bool) raises -> Bool:
    return _admitted_hold_array(array, scan, False)


def admitted_hold_array_f32_deferred(array: PythonObject) raises -> Bool:
    return _admitted_hold_array(array, False, True)


def admitted_held_verify(index: Int) raises -> Bool:
    if admitted_held_finite(index):
        return True
    var snapshot = admitted_held_snapshot(index)
    var verdict = HeldVerification(False, List[Int]())
    try:
        with GILReleased(Python()):
            verdict = admitted_held_verify_native(snapshot.buffers, snapshot.index)
        admitted_held_apply(snapshot, verdict.proved)
        return verdict.finite
    finally:
        _ = snapshot
        _ = verdict
