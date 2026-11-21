# SPDX-License-Identifier: Apache-2.0
# PACK of PyTorch's `parameters()` / `named_parameters()` result forms over a
# class's existing `{name: array}` entry (lane api-torch round 2, 2026-11-21;
# orchestrator decision, Andrew told): `parameters()` is an iterator of
# arrays, `named_parameters()` an iterator of `(name, array)` pairs, as torch
# returns them, on every neural class. The arrays are the entry's own (views
# or copies as the entry documents them): nothing is copied here. Cold API
# call: no loop or repeated call reaches it.
from std.python import PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call
from bindings.native_loaded_module_door import native_loaded_module
from bindings.causal_lm_public_door import causal_public_c
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class

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


def causal_lm_parameters_form_c[form: Int](py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    # CausalLM.parameters() / named_parameters(): the existing entry
    # (`causal_lm_parameters`, causal_public_c[13], lane causal-doors' door,
    # unchanged) then its result in torch's form. A raised entry returns its
    # null with the Python error set, untouched.
    var raw = causal_public_c[13](py_self, args_ptr)
    if not raw:
        return raw
    try:
        return torch_parameter_form(PythonObject(from_owned=raw), form).steal_data()
    except error:
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
