# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Holding a meta-estimator's admitted matrices for its members (lane
# s1-shared-state-hardening-and-selector, S4; design note
# ~/mojolearn-evidence/design-shared-process-state.md section 2: the
# admitted registry is one per process, so a member door in ANY binding
# reuses what the meta door holds).
#
#   admitted_hold_array_f32_deferred(array) -> Bool
#                     hold a 2-D non-empty float32 C-order mojolearn Array in
#                     the calling thread's open scope with its finite test
#                     DEFERRED (lane s2): no
#                     host pass now; a member whose kernel entry tests the
#                     words (defer_finite) never needs one, and the first
#                     member that needs a host verdict runs it once
#   admitted_held_verify(index) -> Bool
#                     the host verdict of a held row: True at once when it
#                     was tested; else the one scan of the oldest untested
#                     matrix on its parent chain (fold rows gathered out of a
#                     finite X are finite), recorded on every row it covers
#
# REMOVED (lane mf-rfe7, 2026-11-23): `admitted_hold_array_f32(array, scan)` and its scanned or vouched,
# undeferred hold; its one caller, RFE's door rounds (each round's block held while the estimator's own fit
# door read it), is deleted.
#
# The caller passes only exact mojolearn Arrays (its own context's
# `array_class` test): an ndarray or a DeviceRows handle is never held here,
# so its member door admits it as before. A non-finite matrix is never held:
# each member refuses or accepts it as its own door does (a forest with a
# missing-value arm keeps accepting it). Numerics: none.
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from bindings.py_admitted_registry import (
    admitted_hold_deferred, admitted_is_held, admitted_scope_open, admitted_held_index,
    admitted_held_finite, admitted_held_snapshot, admitted_held_apply,
)
from core.admitted_held_plan import HeldVerification, admitted_holdable, admitted_held_verify_native


def admitted_hold_array_f32_deferred(array: PythonObject) raises -> Bool:
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
    admitted_hold_deferred(array, address, rows, cols, PythonObject(None))
    return True


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
