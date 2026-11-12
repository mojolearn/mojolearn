# SPDX-License-Identifier: Apache-2.0
"""Connect the one canonical identity-key Python type to each native owner.

Mojo's supported type registry is per interpreter, not per extension module.
Separate host families and merged modules must therefore share the registered
class. Module setup runs with the GIL; this never runs in an execution loop.
"""
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder, lookup_py_type_object
from bindings.py_type_registry import py_type_absent
from bindings.native_identity_key import NativeIdentityKey
from bindings.py_balanced import py_setattr


def native_identity_type(mut module: PythonModuleBuilder) raises:
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[NativeIdentityKey]()
    except error:
        # Only the documented absent-registration case authorizes creation.
        # Keep every other lookup error and all creation/attachment errors.
        if not py_type_absent(error):
            raise error
        _ = module.add_type[NativeIdentityKey]("_NativeIdentityKey")
        return
    # Retain the canonical type itself, so allocation, downcast and destruction
    # use the same native class regardless of which family initialized first.
    py_setattr(module.module, "_NativeIdentityKey", existing)
