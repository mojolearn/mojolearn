# SPDX-License-Identifier: Apache-2.0
"""Shared pieces for registering a Mojo-owned Python type once per process.

Mojo keeps one Python type object per Mojo type id per interpreter. A host
binding can be initialized twice in one process under two module names (the
package loads `mojolearn._host.<member>` at connection time; a plain import of
`mojolearn.host.<member>` initializes the same file again through the merged
finder), and a second `add_type` of the same type aborts PyInit. Each
registrar looks the type up first, creates it only on the documented absent
answer, and otherwise attaches the existing type object to the new module, so
allocation, downcast and destruction use one class. Same pattern as
bindings/native_identity_type.mojo. Module setup runs with the GIL; this never
runs in an execution loop.
"""
from std.python import PythonObject
from bindings.py_balanced import py_setattr


def py_type_absent(error: Error, type_name: String) -> Bool:
    # Only the documented absent-registration answer authorizes creation.
    # stdlib1.0 source: std/python/bindings.mojo, lookup_py_type_object.
    var absent = String("No Python type object registered for Mojo type with name: ") + type_name
    return String(error) == absent


def py_type_attach(module: PythonObject, var name: String, existing: PythonObject) raises:
    # Retain the canonical type itself on this module under its usual name.
    py_setattr(module, name^, existing)
