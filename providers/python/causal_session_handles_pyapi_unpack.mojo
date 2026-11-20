# SPDX-License-Identifier: Apache-2.0
# Python type of the CausalLM native session-handle owner (lane
# causal-lm-sessions-native, 2026-11-11). Module setup only, GIL held.
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder, lookup_py_type_object
from bindings.py_type_registry import py_type_absent, py_type_attach
from core.causal_session_handles import CausalSessionHandles


def causal_session_handles_type(mut module: PythonModuleBuilder) raises:
    """Register `_CausalSessionHandles` on `module`. Mojo's Python type
    registry is per interpreter; the transformer and Mamba bindings both
    own CausalLM handles: the first module to load builds the type, every
    later one attaches that same class (the causal_program_owner_type
    pattern)."""
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[CausalSessionHandles]()
    except error:
        if not py_type_absent(error):
            raise error
        _ = module.add_type[CausalSessionHandles]("_CausalSessionHandles")
        return
    py_type_attach(module.module, "_CausalSessionHandles", existing)
