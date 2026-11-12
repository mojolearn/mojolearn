# SPDX-License-Identifier: Apache-2.0
# Native mode selection over import-time exact-module connections. No loader,
# proxy __getattr__, model __getattribute__, or Python callback runs here.
# The PACK half of the former `bindings/spectral_umap_routes.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/spectral_umap_routes_unpack.mojo`.
from std.python import Python, PythonObject
from bindings.py_balanced import py_get, py_type
from bindings.native_loaded_module_door import native_loaded_module, _loaded_fields


def spectral_umap_route(context: PythonObject, group: String, requested: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var mode = context["backend_fields"]["_DEFAULT_MODE"] if requested is None else requested
    if py_type(mode) is not b.str:
        raise Error("numeric_mode must be fast, identical or None")
    # Exact str methods are native builtins; reject protocol objects first.
    mode = mode.strip().lower()
    if String(mode) != "identical" and String(mode) != "fast":
        raise Error("numeric_mode must be fast, identical or None")
    var module = context["family_spectral_umap_bindings"][PythonObject(group)][mode]
    if py_type(module) is b.str:
        raise Error(String(module))
    if py_type(module) is not py_type(b):
        raise Error("TypeError: spectral/UMAP binding must be an exact module")
    return module


def spectral_umap_optional_entry(module: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    if py_type(module) is not py_type(native_loaded_module("builtins")):
        raise Error("TypeError: spectral/UMAP binding must be an exact module")
    var entry = py_get(_loaded_fields(module), PythonObject(name))
    if entry is not None and py_type(entry) is not context["family_builtin_type"]:
        raise Error("TypeError: mojolearn: " + name + " must be a compiled entry")
    return entry
