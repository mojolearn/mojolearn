# SPDX-License-Identifier: Apache-2.0
# PACK of PyTorch's `parameters()` / `named_parameters()` result forms over a
# class's `{name: array}` mapping (lane api-torch; split out of
# bindings/torch_parameters_pack.mojo in round 3 so that the CNN layers and
# SambaStack import the form without that module's CausalLM door). Cold API
# call: no loop or repeated call reaches it.
from std.python import PythonObject
from bindings.py_balanced import py_call
from bindings.native_loaded_module_door import native_loaded_module

comptime TORCH_FORM_MAPPING = 0
comptime TORCH_FORM_PARAMETERS = 1
comptime TORCH_FORM_NAMED_PARAMETERS = 2


def torch_parameter_form(named: PythonObject, form: Int) raises -> PythonObject:
    """`named` ({name: array}) as `form` asks: the mapping itself, an
    iterator of its arrays, or an iterator of its (name, array) items."""
    if form == TORCH_FORM_MAPPING:
        return named
    var b = native_loaded_module("builtins")
    if form == TORCH_FORM_PARAMETERS:
        return py_call(b.iter, py_call(b.list, named.values()))
    if form == TORCH_FORM_NAMED_PARAMETERS:
        return py_call(b.iter, py_call(b.list, named.items()))
    raise Error("parameters: unknown result form " + String(form))
