# SPDX-License-Identifier: Apache-2.0
# Canonical split descriptor JSON. Mojo owns traversal; the connected CPython
# C encoder supplies JSON's exact ASCII string escaping, including surrogates.
# The PACK half of the former `bindings/msel_json.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/msel_json_unpack.mojo`.
from std.math import isfinite
from bindings.native_loaded_module_door import native_loaded_module
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get
from bindings.msel_inputs_unpack import msel_admit


from bindings.msel_json_unpack import (
    _text,
)


def msel_canonical(value: PythonObject, context: PythonObject) raises -> PythonObject:
    msel_admit(value, context)
    var b = native_loaded_module("builtins")
    if py_type(context["family_msel_json_string"]) is not py_type(b.len):
        raise Error("TypeError: split descriptor string encoding requires the connected C entry")
    return py_call(PythonObject(_text(value, context, b, 0)).encode, "ascii")
