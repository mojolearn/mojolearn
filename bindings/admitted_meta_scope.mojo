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
from std.python import PythonObject

from bindings.admitted_registry import (
    admitted_hold, admitted_hold_deferred, admitted_is_held, admitted_scope_open, admitted_held_index,
    admitted_held_finite, admitted_held_parent, admitted_held_set_finite, admitted_held_address,
    admitted_held_rows, admitted_held_cols,
)
from core.host_finite_scan import host_all_finite_f32


def admitted_hold_array_f32(array: PythonObject, scan: Bool) raises -> Bool:
    """Hold `array` (an exact mojolearn Array) for the members of the open
    scope when it is 2-D, non-empty, float32 and C-ordered (and finite,
    after one scan when `scan`). True when it is held afterwards."""
    if not admitted_scope_open():
        return False
    if admitted_is_held(array):
        return True
    if Int(py=array.ndim) != 2 or String(py=array.dtype) != "<f4" or String(py=array.order) != "C":
        return False
    var rows = Int(py=array.shape[0])
    var cols = Int(py=array.shape[1])
    if rows < 1 or cols < 1:
        return False
    var address = Int(py=array._addr)
    if scan and not host_all_finite_f32(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=address),
                                        rows * cols):
        return False
    admitted_hold(array, address, rows, cols)
    return True


def admitted_hold_array_f32_deferred(array: PythonObject) raises -> Bool:
    """Hold `array` (an exact mojolearn Array) for the members of the open
    scope when it is 2-D, non-empty, float32 and C-ordered, with no host
    scan: its finite test is deferred to the member that reads it
    (`admitted_held_verify`, or the member's kernel entry). True when it is
    held afterwards."""
    if not admitted_scope_open():
        return False
    if admitted_is_held(array):
        return True
    if Int(py=array.ndim) != 2 or String(py=array.dtype) != "<f4" or String(py=array.order) != "C":
        return False
    var rows = Int(py=array.shape[0])
    var cols = Int(py=array.shape[1])
    if rows < 1 or cols < 1:
        return False
    admitted_hold_deferred(array, Int(py=array._addr), rows, cols, PythonObject(None))
    return True


def admitted_held_verify(index: Int) raises -> Bool:
    """The host finite verdict of held row `index`, run at most once per
    matrix: a row gathered out of a held parent takes the parent's verdict
    (the parent is scanned instead of the row when neither was tested); a
    row with no held parent is scanned itself. A finite verdict is recorded
    on every row of the chain. False: the words (or the parent's) hold a NaN
    or an infinity; the caller then admits the row as an unknown matrix."""
    if admitted_held_finite(index):
        return True
    # the chain from the row up to its oldest held, untested ancestor
    var chain = List[Int]()
    chain.append(index)
    var top = index
    while True:  # driver-loop: over a held row's parent chain (fold rows -> X, a few)
        var parent = admitted_held_parent(top)
        if parent is None:
            break
        var p = admitted_held_index(parent)
        if p < 0:
            break
        if admitted_held_finite(p):
            for i in range(len(chain)):  # driver-loop: over the chain's rows
                admitted_held_set_finite(chain[i])
            return True
        chain.append(p)
        top = p
    var address = admitted_held_address(top)
    var n = admitted_held_rows(top) * admitted_held_cols(top)
    if not host_all_finite_f32(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=address), n):
        if top == index:
            return False
        # a non-finite parent says nothing of these rows: test the row itself
        var own = admitted_held_address(index)
        var own_n = admitted_held_rows(index) * admitted_held_cols(index)
        if not host_all_finite_f32(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=own), own_n):
            return False
        admitted_held_set_finite(index)
        return True
    for i in range(len(chain)):  # driver-loop: over the chain's rows
        admitted_held_set_finite(chain[i])
    return True
