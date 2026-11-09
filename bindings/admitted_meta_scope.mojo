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
#
# The caller passes only exact mojolearn Arrays (its own context's
# `array_class` test): an ndarray or a DeviceRows handle is never held here,
# so its member door admits it as before. A non-finite matrix is never held:
# each member refuses or accepts it as its own door does (a forest with a
# missing-value arm keeps accepting it). Numerics: none.
from std.python import PythonObject

from bindings.admitted_registry import admitted_hold, admitted_is_held, admitted_scope_open
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
