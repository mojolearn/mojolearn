# SPDX-License-Identifier: Apache-2.0
# Explicit CPU KDE keeps its existing serial kernel and fixed sum order;
# query preparation, stored-buffer ownership and result publication are native.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_contains
from bindings.classical_attributes_door import classical_getattribute
from bindings.classical_pandas_door import _pd_type_fields
from bindings.native_numpy_metadata_unpack import native_numpy_dtype
from bindings.classical_arrays_door import classical_input, classical_address, classical_is_array, classical_is_f32, classical_c_order, classical_kind
from bindings.host_array_native_pack import host_archive_typed
from bindings.host_inference_scalars_door import host_inference_scalar, host_inference_primitive_number
from bindings.host_inference_features_unpack import host_inference_feature_names
from bindings.host_kde_bandwidth_native_unpack import kde_bandwidth_code, kde_native_bandwidth
from bindings.host_kernel_inference_native_unpack import _kernel_entry, _kernel_buffer
from bindings.host_svc_native_door import _svc_input
from bindings.host_iforest_native_door import host_iforest_alias_types


from bindings.host_kde_native_pack import (
    _kde_option_value,
)


def host_kde_admit_state(fields: PythonObject, context: PythonObject) raises:
    var names = Python.tuple("kernel", "metric")
    for i in range(len(names)):  # driver-loop: fixed KDE kernel/metric names
        if py_contains(fields, names[i]):
            _ = _kde_option_value(fields[names[i]], context)
    var bandwidth = py_get(fields, "bandwidth")
    if bandwidth is not None and kde_bandwidth_code(bandwidth, context) == "":
        _ = _kde_option_value(host_inference_primitive_number(bandwidth, context), context, True)
    var buffers = Python.tuple("_x", "_w")
    for i in range(len(buffers)):  # driver-loop: fixed saved KDE buffers
        var value = py_get(fields, buffers[i])
        if py_type(value) is py_get(context, "ndarray") and classical_kind(value, context) == "O":
            raise Error("TypeError: KDE saved buffers require native numeric storage")
    # An exact string was accepted by the old score's float(bandwidth).
    # Its native float conversion happens below; custom conversion hooks
    # never enter that constructor.
