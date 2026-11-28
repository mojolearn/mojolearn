# SPDX-License-Identifier: Apache-2.0
# Prophet constructor and custom-seasonality operations, including admission
# and owner release. The public shells perform one compiled binding call.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from bindings.sequential_admission_door import _one_of
from providers.python.forecast_initialize_pyapi_unpack import _scalar
from bindings.prophet_keys_unpack import prophet_keys
from bindings.py_balanced import py_item


def prophet_scalar(value: PythonObject, context: PythonObject) raises:
    var nk = prophet_keys()
    if _one_of(py_type(value), py_item(context, nk[].k_xseq_prophet_scalar_inputs)):
        return
    _scalar(value, py_item(context, nk[].k_xseq_lifecycle_context))


def prophet_text(value: PythonObject, context: PythonObject) raises -> Bool:
    var nk = prophet_keys()
    return py_type(value) is py_type(PythonObject("")) or py_type(value) is py_item(context, nk[].k_xseq_prophet_string_type)
