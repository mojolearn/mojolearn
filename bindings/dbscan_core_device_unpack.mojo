# bindings/dbscan_core_device.mojo -- the Python door of the device DBSCAN
# core-sample arrays. The kernels and typed bodies live in
# dbscan/core_device_impl.mojo (2026-10-21).
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call
from max.gpu.host import DeviceContext
from dbscan.core_device_impl import dbscan_core_count, dbscan_core_gather


def _array_addr(a: PythonObject) raises -> Int:
    return Int(py=a.buffer_info()[0])
