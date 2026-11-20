# SPDX-License-Identifier: Apache-2.0
# Native mode selection over import-time exact-module connections. No loader,
# proxy __getattr__, model __getattribute__, or Python callback runs here.
from std.python import Python, PythonObject
from bindings.py_balanced import py_get, py_type
from bindings.native_loaded_module_door import native_loaded_module, _loaded_fields


from bindings.spectral_umap_routes_pack import (
    spectral_umap_optional_entry,
)


def spectral_umap_entry(module: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    var entry = spectral_umap_optional_entry(module, name, context)
    if entry is None:
        raise Error("ImportError: mojolearn: compiled entry is unavailable: " + name)
    return entry
