# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# A meta-estimator's X admitted once for all its members (lane A1
# admit-trees-labels, plan docs/plans/EXECUTOR_CONSOLIDATION_AND_CLEANUP.md
# section 4).
#
#   admitted_scope_begin() -> mark / admitted_scope_end(mark)
#                     the hold scope of one meta-estimator operation
#   admitted_hold(array, address, rows, cols)
#                     hold an admitted finite float32 C-order matrix
#   admitted_is_held(array) -> Bool
#   admitted_held_x(source, est, context, admitted) -> AdmittedX
#                     a held matrix as it is, or bindings/admitted_input.mojo
#                     `admit_x_unrecorded` for anything else
#
# The tree doors (forest, IsolationForest) read X through
# `admitted_held_x`. Doors in other bindings still call admit_x directly
# (bindings/admitted_input.mojo belongs to lane A5); the one-line hook there
# (`_admitted_x` consulting `admitted_held_index`) is owed to that lane.
from std.ffi import _Global
from std.python import PythonObject
from bindings.admitted_input import AdmittedX, admit_x_unrecorded


# ------------------------------------------------- admitted once per call
#
# Lane A1 admit-trees-labels (plan section 4, meta-estimators): a
# meta-estimator (Stacking, Voting, OneVsRest, MultiOutput,
# CalibratedClassifierCV) admits X once and hands the same float32 C-order
# matrix to every member's public fit and predict. Each member door used to
# admit it again (its view, its checks and one finiteness scan per member,
# per fold). The meta-estimator's operation now opens a scope
# (`admitted_scope_begin` ... `admitted_scope_end` in a `finally`), holds
# the matrices it admitted inside it (`admitted_hold`), and the doors that
# read X through `admitted_held_x` return a held matrix as it is: no view, no
# check, no scan.
#
# A matrix is held only inside an open scope and only after it was
# admitted finite, 2-D, non-empty and float32 C-order; entries are strong
# references (no address reuse while held), matched by identity, and the
# scope's end drops every entry made inside it, so nothing outlives the
# call that admitted it. Any other object is admitted as before. A binding
# that does not see the holder's registry (another extension module) admits
# the matrix as before: the same verdict, one more scan. Numerics: none.


struct AdmittedHeld(Defaultable, Movable):
    var items: List[PythonObject]
    var addresses: List[Int]
    var rows: List[Int]
    var cols: List[Int]
    var depth: Int

    def __init__(out self):
        self.items = List[PythonObject]()
        self.addresses = List[Int]()
        self.rows = List[Int]()
        self.cols = List[Int]()
        self.depth = 0


comptime ADMITTED_HELD = _Global[StorageType=AdmittedHeld, name="MojolearnAdmittedHeldV1",
                                 init_fn=AdmittedHeld.__init__]


def admitted_scope_begin() raises -> Int:
    """Open a hold scope; returns the mark `admitted_scope_end` takes."""
    var g = ADMITTED_HELD.get_or_create_ptr()
    g[].depth += 1
    return len(g[].items)


def admitted_scope_end(mark: Int) raises:
    """Close the scope `mark` opened: drop every entry held inside it."""
    var g = ADMITTED_HELD.get_or_create_ptr()
    # Hold the references while the registry shrinks, so no destructor runs
    # while it is being changed.
    var dropped = List[PythonObject]()
    while len(g[].items) > mark:  # small-loop(items: the held matrices of the running calls, a few)
        dropped.append(g[].items.pop())
        _ = g[].addresses.pop()
        _ = g[].rows.pop()
        _ = g[].cols.pop()
    if g[].depth > 0:
        g[].depth -= 1
    _ = dropped^


def admitted_hold(array: PythonObject, address: Int, rows: Int, cols: Int) raises:
    """Hold `array` (admitted finite: rows x cols float32 C-order words at
    `address`) until the open scope ends, so member doors reuse its
    admission. Outside a scope nothing is held."""
    var g = ADMITTED_HELD.get_or_create_ptr()
    if g[].depth <= 0:
        return
    g[].items.append(array)
    g[].addresses.append(address)
    g[].rows.append(rows)
    g[].cols.append(cols)


def admitted_scope_open() raises -> Bool:
    """True inside a hold scope."""
    return ADMITTED_HELD.get_or_create_ptr()[].depth > 0


def admitted_is_held(array: PythonObject) raises -> Bool:
    return admitted_held_index(array) >= 0


def admitted_held_index(source: PythonObject) raises -> Int:
    """The registry row holding `source` (identity), or -1."""
    var g = ADMITTED_HELD.get_or_create_ptr()
    for k in range(len(g[].items)):  # small-loop(items: the held matrices of the running calls, a few)
        if g[].items[k] is source:
            return k
    return -1


def admitted_held_x(source: PythonObject, est: String, context: PythonObject,
                    admitted: Bool = False) raises -> AdmittedX:
    """X for a door: a matrix the running meta-estimator holds, as it is
    (borrowed, already refused-checked and finite), else the one admission
    (`admit_x_unrecorded`, the same refusals and words)."""
    var held = admitted_held_index(source)
    if held >= 0:
        var g = ADMITTED_HELD.get_or_create_ptr()
        return AdmittedX(source, source, g[].addresses[held], g[].rows[held], g[].cols[held], 2, True)
    return admit_x_unrecorded(source, est, context, admitted)
