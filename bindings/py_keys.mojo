# SPDX-License-Identifier: Apache-2.0
# Held name keys and integer operation codes for the boundary files (lane
# fx-boundary-core, 2026-11-19; docs/plans/SPEED_AND_CLEANUP_2026-11-18.md
# part 5, design A of ideas-boundary).
#
# WHAT IT REPLACES. A boundary read written `fields["alpha"]`,
# `py_get(context, "ndarray")` or `py_setitem(fields, "coef_", value)` builds
# a Python str from the literal, hashes it, uses it once and frees it, on
# every call. A door written `var operation = String(py=call[1])` followed by
# `if operation == "fit": ... elif operation == "predict": ...` copies the
# name into a Mojo String and compares it against every earlier name of the
# chain, on every call.
#
# THE ONE FORM (every family, every lane):
#   * One table per family, `bindings/<family>_keys_unpack.mojo`: a struct
#     with one `PythonObject` field per name the family reads or writes
#     (`k_<name>`: a typo is a compile error, where a mistyped literal was a
#     KeyError at run time), one `ops` dictionary, one `comptime` integer per
#     public operation, a `SharedSlot`, a fill function and the accessor
#     `<family>_keys()`. The fill runs once per process, at the family's
#     first call; every later call finds the keys by pointer.
#   * A function that reads by name fetches the table once, first line,
#     outside every loop (`var nk = tsa_keys()`), the form `cpython_table()`
#     already has, and writes `py_item(fields, nk[].k_alpha)`,
#     `py_get(context, nk[].k_ndarray)`, `py_setitem(fields, nk[].k_coef_,
#     value)`. The dictionary stays the one source of truth: the same
#     `PyObject_GetItem` / `PyDict_GetItemWithError` / `PyObject_SetItem` runs
#     with the same key value, so a missing key, a foreign mapping or a
#     user's attribute assignment behaves exactly as before.
#   * A public C entry resolves its operation once: `var op =
#     py_op_code(nk[].ops, call[1])` (one dictionary lookup with the caller's
#     own str object, whose hash CPython caches), and dispatches on
#     `op == TSA_OP_HW_FIT`. An unknown name is `-1`; only that path builds
#     the name String, for the unchanged error message.
#
# COST (AGENTS.md "No refactor adds cost on a hot path"). Per function with
# by-name reads: one slot fetch and one flag test replace one str build, one
# hash and one free PER READ. Nothing is allocated per call. No table is
# walked: every key is a field at a fixed offset, every operation one lookup.
# A function with a single by-name read on a per-item path takes the table
# from its caller as an argument instead of fetching it per item.
#
# The keys are interned (`sys.intern`), so a lookup in a dictionary whose
# keys are interned too (instance attributes, keys written as literals in
# Python source) ends on the pointer compare. An interned str created here is
# kept for the life of the process, as the slot is.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_setitem


def py_key(name: String) raises -> PythonObject:
    """The interned Python str of `name`. Fill time only (once per process
    and name), never on a call path."""
    return py_call(native_loaded_module("sys").intern, PythonObject(name))


def py_op_add(ops: PythonObject, name: String, code: Int) raises:
    """Record `name -> code` in a family's operation dictionary. Fill time
    only."""
    py_setitem(ops, py_key(name), PythonObject(code))


def py_op_code(ops: PythonObject, operation: PythonObject) raises -> Int:
    """The integer code of a public operation name, or -1 for a name the
    family does not have. `operation` is the caller's own str object: no
    String is built and no chain is walked. Any hashable object that is not a
    known name is -1 (the door then raises its unchanged "unknown operation"
    error); an unhashable one raises its TypeError."""
    ref cpy = Python().cpython()
    var found = cpy.PyDict_GetItemWithError(ops._obj_ptr, operation._obj_ptr)
    if not found:
        if cpy.PyErr_Occurred():
            raise cpy.unsafe_get_error()
        return -1
    return Int(py=PythonObject(from_borrowed=found))
