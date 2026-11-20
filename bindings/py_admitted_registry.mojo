# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Door-side pins of the "already admitted in this operation" registry that
# `admit_x` consults (UNPACK and PACK only). The typed table (identity words,
# addresses, shapes, threads, verdicts, scope depths) is RUN in
# bindings/admitted_registry_run.mojo (split from bindings/admitted_registry.mojo,
# lane resident-pin-tables, 2026-11-12). This store holds, in lists aligned
# row for row with that table, the held matrices and their parents, strongly,
# so an identity word compared there cannot be reused while its row is listed.
#
#   admitted_scope_begin() -> mark / admitted_scope_end(mark)
#                     the hold scope of one meta-estimator operation (end it
#                     in a `finally`)
#   admitted_hold(array, address, rows, cols)
#                     hold an admitted finite float32 C-order matrix until
#                     the open scope ends (outside a scope: nothing held)
#   admitted_hold_deferred(array, address, rows, cols, parent)
#                     hold one whose finite test is DEFERRED; `parent`: the
#                     held matrix these rows were gathered out of, or None
#   admitted_hold_rows_of(parent, array, address, rows, cols)
#                     hold rows gathered out of a held `parent` with the
#                     parent's verdict
#   admitted_scope_open() -> Bool
#   admitted_held_index(source) -> Int / admitted_is_held(source) -> Bool
#   admitted_held_address / _rows / _cols / _finite / _set_finite(index)
#                     (typed, re-exported from the RUN module)
#   admitted_held_snapshot(index) / admitted_held_apply(snapshot, proved)
#
# This file imports nothing of the admission path, so admitted_input.mojo
# and any meta-estimator door can import it without a cycle. Every call runs
# under the GIL, which serialises the table with these pins.
from std.python import PythonObject

from core.shared_slot import SharedSlot
from core.admitted_held_plan import HeldBufferDescriptor
from bindings.admitted_registry_run import (
    admitted_run_scope_begin, admitted_run_scope_end, admitted_run_hold_row, admitted_run_scope_open,
    admitted_run_held_index, admitted_run_snapshot, admitted_run_apply,
    admitted_held_address, admitted_held_rows, admitted_held_cols, admitted_held_finite,
    admitted_held_set_finite,
)


struct AdmittedPins(Defaultable, Movable):
    # aligned row for row with bindings/admitted_registry_run.mojo AdmittedTable
    var items: List[PythonObject]
    var parents: List[PythonObject]

    def __init__(out self):
        self.items = List[PythonObject]()
        self.parents = List[PythonObject]()


comptime _ADMITTED_PINS = SharedSlot[AdmittedPins, "MojolearnAdmittedPinsV1", 1]


def _pins() raises -> UnsafePointer[AdmittedPins, MutAnyOrigin]:
    return _ADMITTED_PINS.get_or_create_ptr().unsafe_origin_cast[MutAnyOrigin]()


def _word(value: PythonObject) -> Int:
    return 0 if value is None else Int(value._obj_ptr)


def admitted_scope_begin() raises -> Int:
    """Open a hold scope on the calling thread; returns the mark
    `admitted_scope_end` takes."""
    return admitted_run_scope_begin()


def admitted_scope_end(mark: Int) raises:
    """Close the scope `mark` opened: drop every entry the calling thread
    held inside it; other threads' entries stay."""
    var drop = admitted_run_scope_end(mark)
    if len(drop) == 0:
        return
    # Hold the references while the pin lists shrink, so no destructor runs
    # while they are being changed.
    var dropped = List[PythonObject]()
    var p = _pins()
    for j in range(len(drop)):  # small-loop(drop: this thread's rows of the closing scope, a few, descending)
        dropped.append(p[].items.pop(drop[j]))
        dropped.append(p[].parents.pop(drop[j]))
    _ = dropped^


def _hold_row(array: PythonObject, address: Int, rows: Int, cols: Int, verdict: Int, parent: PythonObject) raises:
    if admitted_run_hold_row(Int(array._obj_ptr), address, rows, cols, verdict, _word(parent)):
        var p = _pins()
        p[].items.append(array)
        p[].parents.append(parent)


def admitted_hold(array: PythonObject, address: Int, rows: Int, cols: Int) raises:
    """Hold `array` (admitted finite: rows x cols float32 C-order words at
    `address`) until the calling thread's open scope ends, so member doors
    reuse its admission. Outside a scope nothing is held."""
    _hold_row(array, address, rows, cols, 1, PythonObject(None))


def admitted_hold_deferred(array: PythonObject, address: Int, rows: Int, cols: Int, parent: PythonObject) raises:
    """Hold `array` (rows x cols float32 C-order words at `address`, NOT yet
    tested finite) until the calling thread's open scope ends. `parent` is
    the held matrix the rows were gathered out of (its verdict covers them),
    or None."""
    _hold_row(array, address, rows, cols, 0, parent)


def admitted_hold_rows_of(parent: PythonObject, array: PythonObject, address: Int, rows: Int, cols: Int) raises:
    """Hold rows gathered out of the held matrix `parent` with its verdict:
    finite when `parent` was tested finite, else deferred and linked to it.
    Nothing is held when `parent` is not held by the calling thread."""
    var k = admitted_held_index(parent)
    if k < 0:
        return
    if admitted_held_finite(k):
        _hold_row(array, address, rows, cols, 1, PythonObject(None))
    else:
        _hold_row(array, address, rows, cols, 0, parent)


def admitted_scope_open() raises -> Bool:
    """True inside a hold scope of the calling thread."""
    return admitted_run_scope_open()


def admitted_held_index(source: PythonObject) raises -> Int:
    """The registry row the calling thread holds `source` in (identity), or
    -1."""
    return admitted_run_held_index(Int(source._obj_ptr))


def admitted_is_held(source: PythonObject) raises -> Bool:
    return admitted_held_index(source) >= 0


struct AdmittedHeldSnapshot(Movable):
    var buffers: List[HeldBufferDescriptor]
    var owners: List[PythonObject]
    var index: Int

    def __init__(out self, var buffers: List[HeldBufferDescriptor], var owners: List[PythonObject], index: Int):
        self.buffers = buffers^
        self.owners = owners^
        self.index = index


def admitted_held_snapshot(index: Int) raises -> AdmittedHeldSnapshot:
    """UNPACK one calling-thread registry snapshot and pin its buffer owners.

    Native RUN never follows Python parent objects or borrows registry Lists
    that another thread can compact while the GIL is released.
    """
    var buffers = List[HeldBufferDescriptor]()
    var positions = List[Int]()
    var selected = admitted_run_snapshot(index, buffers, positions)
    var owners = List[PythonObject](capacity=len(positions))
    var p = _pins()
    for i in range(len(positions)):  # unpack: the calling thread's retained buffer owners
        owners.append(p[].items[positions[i]])
    return AdmittedHeldSnapshot(buffers^, owners^, selected)


def admitted_held_apply(snapshot: AdmittedHeldSnapshot, proved: List[Int]) raises:
    """PACK finite verdicts after RUN; another thread may have compacted rows.
    The caller's snapshot keeps its owners pinned through this call, so
    their identity words still name them."""
    admitted_run_apply(snapshot.buffers, proved)
