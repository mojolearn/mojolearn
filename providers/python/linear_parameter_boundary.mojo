# SPDX-License-Identifier: Apache-2.0
"""Unpack Ridge options; native state code owns every option decision."""
from std.python import PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type
from core.linear_model_state import linear_ridge_parameter_status, linear_ridge_parameter_error


def classical_ridge_parameters(fields: PythonObject) raises:
    var b = native_loaded_module("builtins")
    var alpha_object = fields["alpha"]
    # A string is not a numeric alpha; Float64's string parser would admit
    # values the public comparison contract refuses.
    if py_type(alpha_object) is b.str:
        raise Error("TypeError: '<' not supported between instances of 'str' and 'float'")
    var alpha = Float64(0.0)
    try:
        alpha = Float64(py=alpha_object)
    except problem:
        raise Error("TypeError: " + String(problem))
    var status = linear_ridge_parameter_status(alpha, "", False)
    if status == 1:
        linear_ridge_parameter_error(status, String(py_call(b.str, alpha_object)), "")
    var solver = fields["solver"]
    var name = String(solver) if py_type(solver) is b.str else String("")
    status = linear_ridge_parameter_status(alpha, name, False)
    if status != 0:
        linear_ridge_parameter_error(status, "", String(py_call(b.repr, solver)))
    status = linear_ridge_parameter_status(alpha, name, Bool(py=fields["normalize"]))
    linear_ridge_parameter_error(status, "", "")
