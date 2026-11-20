# SPDX-License-Identifier: Apache-2.0
"""The vendor and numeric-mode check of a connected entries table, once per
table (rows N313, N317 of docs/plans/NATIVE_ENTRY_POINTS_AFTER_0866.md).

The decomposition owners (TruncatedSVD, KernelPCA, ...) checked, on every
call, that the binding behind a connected entries table reports the selected
vendor and the table's numeric mode: two compiled calls, two String builds
and two compares per query. A compiled binding's vendor and mode words are
fixed for the life of its module, so the answer can change only when the
table itself is replaced or the selected vendor object in the connected
context is. `entries_verified` answers from a record kept in the owner's
own connected context (a dict built once at import, never shared between
owners): `context[slot][mode] = (entries, vendor)`, compared by identity.
`entries_mark_verified` writes the record after the caller's full check
passed. Nothing here computes on data or calls into the binding.

The record holds the table and the vendor object strongly, so neither
address can be reused under it while it is kept.
"""
from std.python import Python, PythonObject
from bindings.py_balanced import py_get, py_setitem


def entries_verified(context: PythonObject, slot: String, mode: PythonObject,
                     entries: PythonObject) raises -> Bool:
    """True when `entries` (the table connected for `mode`) passed the full
    vendor and mode check under the context's current `vendor` object."""
    var records = py_get(context, PythonObject(slot))
    if records is None:
        return False
    var held = py_get(records, mode)
    if held is None:
        return False
    return held[0] is entries and held[1] is py_get(context, "vendor")


def entries_mark_verified(context: PythonObject, slot: String, mode: PythonObject,
                          entries: PythonObject) raises:
    """Record that `entries` passed the full check (call it only after the
    check, never instead of it)."""
    var records = py_get(context, PythonObject(slot))
    if records is None:
        records = Python.dict()
        py_setitem(context, PythonObject(slot), records)
    py_setitem(records, mode, Python.tuple(entries, py_get(context, "vendor")))
