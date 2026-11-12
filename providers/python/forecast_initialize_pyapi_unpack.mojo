# SPDX-License-Identifier: Apache-2.0
# Constructor admission, normalization and state commit for forecast families.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_setitem, py_contains
from bindings.sequential_admission import _one_of
from providers.python.autoarima_lifecycle_pyapi import _fields, _integer
from providers.python.forecast_owner_pyapi_unpack import forecast_commit


def _scalar(value: PythonObject, context: PythonObject) raises:
    var cls = py_type(value)
    if value is None or cls is py_type(PythonObject("")) or cls is py_type(PythonObject(True)) or cls is py_type(PythonObject(0)) or cls is py_type(PythonObject(0.0)) or _one_of(cls, context["scalar_types"]):
        return
    raise Error("TypeError: forecast constructor settings require native scalar values")
