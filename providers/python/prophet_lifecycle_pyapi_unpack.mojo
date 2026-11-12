# SPDX-License-Identifier: Apache-2.0
# Prophet constructor and custom-seasonality operations, including admission
# and owner release. The public shells perform one compiled binding call.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module, NativeModules
from bindings.native_class_admission_door import class_record, class_admit
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from bindings.classical_pandas_door import classical_admit_value
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.sequential_admission_door import _one_of
from providers.python.autoarima_lifecycle_pyapi_door import _fields
from providers.python.forecast_initialize_pyapi_unpack import _scalar
from providers.python.forecast_owner_pyapi_unpack import forecast_commit


def prophet_scalar(value: PythonObject, context: PythonObject) raises:
    if _one_of(py_type(value), context["xseq_prophet_scalar_inputs"]):
        return
    _scalar(value, context["xseq_lifecycle_context"])


def prophet_text(value: PythonObject, context: PythonObject) raises -> Bool:
    return py_type(value) is py_type(PythonObject("")) or py_type(value) is context["xseq_prophet_string_type"]
