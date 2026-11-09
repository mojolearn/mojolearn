# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# The "already admitted in this operation" registry that `admit_x`
# consults (lane a5-admit-core-followup, 2026-11-08, for lane A1
# admit-trees-labels; plan docs/plans/EXECUTOR_CONSOLIDATION_AND_CLEANUP.md
# section 4, meta-estimators).
#
#   admitted_scope_begin() -> mark / admitted_scope_end(mark)
#                     the hold scope of one meta-estimator operation (end it
#                     in a `finally`)
#   admitted_hold(array, address, rows, cols)
#                     hold an admitted finite float32 C-order matrix until
#                     the open scope ends (outside a scope: nothing held)
#   admitted_hold_deferred(array, address, rows, cols, parent)
#                     hold one whose finite test is DEFERRED (lane s2): a
#                     member whose kernel entry tests the words it uploads
#                     (defer_finite) reads it with no host pass; the first
#                     member that needs a host verdict gets it once
#                     (bindings/admitted_meta_scope.mojo admitted_held_verify)
#                     and it is recorded on the row. `parent`: the held
#                     matrix these rows were gathered out of, or None
#   admitted_hold_rows_of(parent, array, address, rows, cols)
#                     hold rows gathered out of a held `parent` with the
#                     parent's verdict (finite, or deferred with the link)
#   admitted_held_finite(index) / admitted_held_parent(index) /
#   admitted_held_set_finite(index)
#                     the row's recorded verdict, its parent, and the record
#   admitted_scope_open() -> Bool
#   admitted_held_index(source) -> Int
#                     the registry row holding `source` (identity), or -1
#   admitted_is_held(source) -> Bool
#   admitted_held_address / _rows / _cols(index)
#                     the held matrix's words and shape
#
# `_admitted_x` (bindings/admitted_input.mojo) returns a held matrix as it is
# (borrowed, finite: no view, no check, no scan) for a C-order float32
# request; every door that reads X through admit_x gets the hook. This file
# imports nothing of the admission path, so admitted_input.mojo and any
# meta-estimator door can import it without a cycle.
#
# A matrix is held only inside an open scope and only after it was admitted
# finite, 2-D, non-empty and float32 C-order; entries are strong references
# (no address reuse while held), matched by identity AND by the calling
# thread, and the scope's end drops every entry the calling thread made
# inside it, so nothing outlives the call that admitted it and one thread's
# scope never ends another's entries (lane s1, design note hazard 3).
#
# ONE REGISTRY PER PROCESS. A `_Global` is one slot per process per name
# (core/shared_slot.mojo): every extension module of every wheel that
# imports this file reads the same registry, so a member door in any binding
# reuses the admission a meta-estimator door in any other binding held. The
# name is not tier-keyed on purpose: the held verdict ("finite float32
# C-order") is the same in FAST and IDENTICAL. Every call runs under the GIL
# (the rows are PythonObjects), which serialises the mutations; the layout
# word refuses a registry created by a binary of another layout.
# Numerics: none.
#
# Moved from lane A1's bindings/admitted_held.mojo (branch
# lane/a1-admit-trees-labels 076abee5ea) word for word, under a new global
# name so the two definitions never share storage while both exist.
from std.ffi import external_call
from std.python import PythonObject

from core.shared_slot import SharedSlot

#: Layout version of `AdmittedRegistry` (core/shared_slot.mojo rule). V2: the
#: layout word, the per-row thread and the per-thread depth. V3 (lane s2):
#: the per-row finite verdict and parent (deferred holds).
comptime _ADMITTED_LAYOUT = 3
comptime _ADMITTED_NAME = "MojolearnAdmittedRegistryV3"


struct AdmittedRegistry(Defaultable, Movable):
    var items: List[PythonObject]
    var addresses: List[Int]
    var rows: List[Int]
    var cols: List[Int]
    # the thread that held each row
    var threads: List[Int]
    # 1: tested finite; 0: deferred (not tested on the host yet)
    var verdicts: List[Int]
    # the held matrix each row was gathered out of, or None
    var parents: List[PythonObject]
    # the open-scope depth of each thread that has one open
    var depth_threads: List[Int]
    var depths: List[Int]

    def __init__(out self):
        self.items = List[PythonObject]()
        self.addresses = List[Int]()
        self.rows = List[Int]()
        self.cols = List[Int]()
        self.threads = List[Int]()
        self.verdicts = List[Int]()
        self.parents = List[PythonObject]()
        self.depth_threads = List[Int]()
        self.depths = List[Int]()


comptime ADMITTED_REGISTRY = SharedSlot[AdmittedRegistry, _ADMITTED_NAME, _ADMITTED_LAYOUT]


def _registry() raises -> UnsafePointer[AdmittedRegistry, MutAnyOrigin]:
    var g = ADMITTED_REGISTRY.get_or_create_ptr().unsafe_origin_cast[MutAnyOrigin]()
    return g


def _thread() -> Int:
    # pthread_t: an unsigned long on Linux, a pointer on macOS; either way a
    # nonzero word naming the live calling thread (core/device_plan.mojo).
    return Int(external_call["pthread_self", UInt]())


def _depth_index(g: UnsafePointer[AdmittedRegistry, MutAnyOrigin], me: Int) -> Int:
    for i in range(len(g[].depth_threads)):  # small-loop(depth_threads: threads with an open scope, a few)
        if g[].depth_threads[i] == me:
            return i
    return -1


def _mine(g: UnsafePointer[AdmittedRegistry, MutAnyOrigin], me: Int) -> Int:
    var n = 0
    for k in range(len(g[].threads)):  # small-loop(threads: the held matrices of the running calls, a few)
        if g[].threads[k] == me:
            n += 1
    return n


def admitted_scope_begin() raises -> Int:
    """Open a hold scope on the calling thread; returns the mark
    `admitted_scope_end` takes (the thread's held count)."""
    var g = _registry()
    var me = _thread()
    var d = _depth_index(g, me)
    if d < 0:
        g[].depth_threads.append(me)
        g[].depths.append(1)
    else:
        g[].depths[d] += 1
    return _mine(g, me)


def admitted_scope_end(mark: Int) raises:
    """Close the scope `mark` opened: drop every entry the calling thread
    held inside it; other threads' entries stay."""
    var g = _registry()
    var me = _thread()
    # Hold the references while the registry shrinks, so no destructor runs
    # while it is being changed.
    var dropped = List[PythonObject]()
    var drop = List[Int]()
    var seen = 0
    for k in range(len(g[].threads)):  # small-loop(threads: the held matrices of the running calls, a few)
        if g[].threads[k] == me:
            if seen >= mark:
                drop.append(k)
            seen += 1
    var j = len(drop) - 1
    while j >= 0:  # small-loop(drop: this thread's rows of the closing scope, a few)
        var k = drop[j]
        dropped.append(g[].items.pop(k))
        _ = g[].addresses.pop(k)
        _ = g[].rows.pop(k)
        _ = g[].cols.pop(k)
        _ = g[].threads.pop(k)
        _ = g[].verdicts.pop(k)
        dropped.append(g[].parents.pop(k))
        j -= 1
    var d = _depth_index(g, me)
    if d >= 0:
        g[].depths[d] -= 1
        if g[].depths[d] <= 0:
            _ = g[].depths.pop(d)
            _ = g[].depth_threads.pop(d)
    _ = dropped^


def _hold_row(array: PythonObject, address: Int, rows: Int, cols: Int, verdict: Int, parent: PythonObject) raises:
    var g = _registry()
    var me = _thread()
    if _depth_index(g, me) < 0:
        return
    g[].items.append(array)
    g[].addresses.append(address)
    g[].rows.append(rows)
    g[].cols.append(cols)
    g[].threads.append(me)
    g[].verdicts.append(verdict)
    g[].parents.append(parent)


def admitted_hold(array: PythonObject, address: Int, rows: Int, cols: Int) raises:
    """Hold `array` (admitted finite: rows x cols float32 C-order words at
    `address`) until the calling thread's open scope ends, so member doors
    reuse its admission. Outside a scope nothing is held."""
    _hold_row(array, address, rows, cols, 1, PythonObject(None))


def admitted_hold_deferred(array: PythonObject, address: Int, rows: Int, cols: Int, parent: PythonObject) raises:
    """Hold `array` (rows x cols float32 C-order words at `address`, NOT yet
    tested finite) until the calling thread's open scope ends. A member door
    that defers its finite test to its kernel entry reads it with no host
    pass; one that needs a host verdict gets it once from
    `admitted_held_verify` (bindings/admitted_meta_scope.mojo). `parent` is
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
    if _registry()[].verdicts[k] == 1:
        _hold_row(array, address, rows, cols, 1, PythonObject(None))
    else:
        _hold_row(array, address, rows, cols, 0, parent)


def admitted_scope_open() raises -> Bool:
    """True inside a hold scope of the calling thread."""
    var g = _registry()
    return _depth_index(g, _thread()) >= 0


def admitted_held_index(source: PythonObject) raises -> Int:
    """The registry row the calling thread holds `source` in (identity), or
    -1. Outside every scope the registry is empty, so this is one load and a
    compare."""
    var g = _registry()
    if len(g[].items) == 0:
        return -1
    var me = _thread()
    for k in range(len(g[].items)):  # small-loop(items: the held matrices of the running calls, a few)
        if g[].threads[k] == me and g[].items[k] is source:
            return k
    return -1


def admitted_is_held(source: PythonObject) raises -> Bool:
    return admitted_held_index(source) >= 0


def admitted_held_address(index: Int) raises -> Int:
    return _registry()[].addresses[index]


def admitted_held_rows(index: Int) raises -> Int:
    return _registry()[].rows[index]


def admitted_held_cols(index: Int) raises -> Int:
    return _registry()[].cols[index]


def admitted_held_finite(index: Int) raises -> Bool:
    """True when the held row was tested finite (not deferred)."""
    return _registry()[].verdicts[index] == 1


def admitted_held_parent(index: Int) raises -> PythonObject:
    """The held matrix the row was gathered out of, or None."""
    return _registry()[].parents[index]


def admitted_held_set_finite(index: Int) raises:
    """Record that the held row's words were tested finite."""
    _registry()[].verdicts[index] = 1
