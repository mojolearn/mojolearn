# SPDX-License-Identifier: Apache-2.0
# Python type of the Samba program handle owner (lane samba-sessions-native,
# 2026-11-12). Module setup only, GIL held.
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder, lookup_py_type_object
from bindings.py_type_registry import py_type_absent, py_type_attach
from core.samba_program_handle import SambaProgramHandle


def samba_program_handle_type(mut module: PythonModuleBuilder) raises:
    """Register `_SambaProgramHandle` on `module` (the one owner:
    `_mojolearn_runtime_neural_host`, the image that runs the Samba public
    doors). A second load in the same interpreter attaches the class the
    first one built (the causal_session_handles_type pattern)."""
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[SambaProgramHandle]()
    except error:
        if not py_type_absent(error):
            raise error
        _ = module.add_type[SambaProgramHandle]("_SambaProgramHandle")
        return
    py_type_attach(module.module, "_SambaProgramHandle", existing)
