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
# (no address reuse while held), matched by identity, and the scope's end
# drops every entry made inside it, so nothing outlives the call that
# admitted it. A binding that does not see this registry (another extension
# module: `_Global` storage is per loaded module) admits the matrix as
# before: the same verdict, one more scan. Numerics: none.
#
# Moved from lane A1's bindings/admitted_held.mojo (branch
# lane/a1-admit-trees-labels 076abee5ea) word for word, under a new global
# name so the two definitions never share storage while both exist.
from std.ffi import _Global
from std.python import PythonObject


struct AdmittedRegistry(Defaultable, Movable):
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


comptime ADMITTED_REGISTRY = _Global[StorageType=AdmittedRegistry, name="MojolearnAdmittedRegistryV1",
                                     init_fn=AdmittedRegistry.__init__]


def admitted_scope_begin() raises -> Int:
    """Open a hold scope; returns the mark `admitted_scope_end` takes."""
    var g = ADMITTED_REGISTRY.get_or_create_ptr()
    g[].depth += 1
    return len(g[].items)


def admitted_scope_end(mark: Int) raises:
    """Close the scope `mark` opened: drop every entry held inside it."""
    var g = ADMITTED_REGISTRY.get_or_create_ptr()
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
    var g = ADMITTED_REGISTRY.get_or_create_ptr()
    if g[].depth <= 0:
        return
    g[].items.append(array)
    g[].addresses.append(address)
    g[].rows.append(rows)
    g[].cols.append(cols)


def admitted_scope_open() raises -> Bool:
    """True inside a hold scope."""
    return ADMITTED_REGISTRY.get_or_create_ptr()[].depth > 0


def admitted_held_index(source: PythonObject) raises -> Int:
    """The registry row holding `source` (identity), or -1. Outside a scope
    the registry is empty, so this is one load and a compare."""
    var g = ADMITTED_REGISTRY.get_or_create_ptr()
    for k in range(len(g[].items)):  # small-loop(items: the held matrices of the running calls, a few)
        if g[].items[k] is source:
            return k
    return -1


def admitted_is_held(source: PythonObject) raises -> Bool:
    return admitted_held_index(source) >= 0


def admitted_held_address(index: Int) raises -> Int:
    return ADMITTED_REGISTRY.get_or_create_ptr()[].addresses[index]


def admitted_held_rows(index: Int) raises -> Int:
    return ADMITTED_REGISTRY.get_or_create_ptr()[].rows[index]


def admitted_held_cols(index: Int) raises -> Int:
    return ADMITTED_REGISTRY.get_or_create_ptr()[].cols[index]
