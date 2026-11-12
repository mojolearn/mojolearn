# SPDX-License-Identifier: Apache-2.0
"""Exact compiled owner identities and native inspection of their retained pins.

Only audited compiled type registrars call register_native_owner, using the
canonical result of lookup_py_type_object after the module is finalized. Nothing
here is exported to Python. Registration and admission hold the GIL.
"""
# The PACK half of the former `bindings/native_owner_registry.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/native_owner_registry_unpack.mojo`.
from std.python import PythonObject
from std.python.bindings import lookup_py_type_object
from std.python._cpython import PyObjectPtr
from core.shared_slot import SharedSlot


def borrowed_native_owner(address: Int) raises -> PythonObject:
    # Callers retain and admit this non-null PyObject address before inspection.
    # PyObjectPtr is an opaque SDK wrapper, constructed by its supported upcast;
    # it is not itself a Pointer with an unsafe_from_address initializer.
    var raw = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address)
    return PythonObject(from_borrowed=PyObjectPtr(upcast_from=Optional(raw)))
