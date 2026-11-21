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
from bindings.torch_parameter_form_pack import torch_parameter_form, TORCH_FORM_MAPPING
from bindings.causal_lm_public_door import causal_public_c
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class


def causal_lm_parameters_form_c[form: Int](py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    # CausalLM.parameters() / named_parameters(): the existing entry
    # (`causal_lm_parameters`, causal_public_c[13], lane causal-doors' door,
    # unchanged) then its result in torch's form. A raised entry returns its
    # null with the Python error set, untouched. state_dict() (the mapping
    # form) takes the copying entry, causal_public_c[18], which leaves the
    # model unexposed and its resident session kept (lane fx-leftovers
    # item 1, 2026-11-21; it was the exposing entry 13 for every form).
    var raw = PyObjectPtr()
    comptime if form == TORCH_FORM_MAPPING:
        raw = causal_public_c[18](py_self, args_ptr)
    else:
        raw = causal_public_c[13](py_self, args_ptr)
    if not raw:
        return raw
    try:
        return torch_parameter_form(PythonObject(from_owned=raw), form).steal_data()
    except error:
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
