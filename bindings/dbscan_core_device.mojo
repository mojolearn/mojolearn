# bindings/dbscan_core_device.mojo -- the Python door of the device DBSCAN
# core-sample arrays. The kernels and typed bodies live in
# dbscan/core_device_impl.mojo (2026-10-21).
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call
from max.gpu.host import DeviceContext
from dbscan.core_device_impl import dbscan_core_count, dbscan_core_gather


def _new_array(arr: PythonObject, code: StaticString, nbytes: Int) raises -> PythonObject:
    """An `array.array(code)` of `nbytes` zero bytes."""
    # p3-neighbors-cluster: ONE allocation of the final array (a one-item
    # array repeated), not a zero bytes object copied into a second buffer.
    var one = py_call(arr.array, PythonObject(code), Python.list(PythonObject(0)))
    var item = Int(py=one.itemsize)
    return py_call(one.__mul__, PythonObject(nbytes // item))


def _array_addr(a: PythonObject) raises -> Int:
    return Int(py=a.buffer_info()[0])

def dbscan_core_arrays_device(
    ctx: DeviceContext, x_addr: PythonObject, labels_addr: PythonObject,
    core_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """`dbscan_core_arrays` (params [n, d]) on the device: [idx int32 (m),
    components float32 (m, d), labels int32 (m)] as `array.array`s."""
    if len(params) != 2:
        raise Error("dbscan_core_arrays: params must be [n, d], got " + String(len(params)) + " values")
    var n = Int(py=params[0])
    var d = Int(py=params[1])
    if n < 0 or d < 1 or (n > 0 and d > 2147483000 // n):
        raise Error("dbscan_core_arrays: n >= 0, d >= 1 and n * d within Int32")
    var arr = native_loaded_module("array")
    if n == 0:
        var out0 = Python.list()
        out0.append(_new_array(arr, "i", 0))
        out0.append(_new_array(arr, "f", 0))
        out0.append(_new_array(arr, "i", 0))
        return out0
    var xa = Int(py=x_addr)
    var la = Int(py=labels_addr)
    var ca = Int(py=core_addr)
    if xa == 0 or la == 0 or ca == 0:
        raise Error("dbscan_core_arrays: null buffer address")
    var m = dbscan_core_count(ctx, ca, n)
    var idx = _new_array(arr, "i", 4 * m)
    var comp = _new_array(arr, "f", 4 * m * d)
    var lab = _new_array(arr, "i", 4 * m)
    dbscan_core_gather(ctx, xa, la, ca, n, d, m, _array_addr(idx), _array_addr(comp), _array_addr(lab))
    var out = Python.list()
    out.append(idx)
    out.append(comp)
    out.append(lab)
    return out
