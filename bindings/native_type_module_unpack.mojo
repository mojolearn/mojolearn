# SPDX-License-Identifier: Apache-2.0
"""Native class-module metadata without metaclass attribute callbacks."""
from std.python import PythonObject
from bindings.native_loaded_module_door import NativeModules
from bindings.mode_metadata_native_door import _class_fields
from bindings.py_balanced import py_call
from bindings.native_keys_unpack import native_keys
from bindings.py_balanced import py_item


def native_type_module(cls: PythonObject) raises -> PythonObject:
    var nk = native_keys()
    var mods = NativeModules()
    # The native module getter probes the type namespace. Admit every key
    # before that probe, reusing the existing watched class-key admission.
    _ = _class_fields(mods, cls)
    var b = mods.module("builtins")
    # type's native getset handles builtin and heap types itself. Calling it
    # directly bypasses a Python metaclass __getattribute__ or descriptor.
    return py_call(py_item(b.type.__dict__, nk[].k___module__).__get__, cls)
