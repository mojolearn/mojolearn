# SPDX-License-Identifier: Apache-2.0
# Python doors for the ONNX model reader (import_reader.mojo): each converts
# its arguments and the reader's typed results, nothing else.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call
from autograd.import_reader import onnx_read, onnx_info, onnx_buffers, onnx_close


def onnx_read_py(path: PythonObject) raises -> PythonObject:
    """Read the ONNX model at `path`; returns its handle (>= 1)."""
    return PythonObject(onnx_read(String(py=path)))


def onnx_info_py(h: PythonObject) raises -> PythonObject:
    """[words, strings, floats] of handle h (the layout in import_reader.mojo's docstring)."""
    var info = onnx_info(Int(py=h))
    var w = Python.list()
    for i in range(len(info.words)):  # driver-loop: over the graph's words, converted; cpu-route: model file parsing conversion
        _ = py_call(w.append, PythonObject(info.words[i]))
    var strs = Python.list()
    for i in range(len(info.strs)):  # driver-loop: over the graph's strings, converted; cpu-route: model file parsing conversion
        _ = py_call(strs.append, PythonObject(info.strs[i]))
    var floats = Python.list()
    for i in range(len(info.floats)):  # driver-loop: over attribute floats, converted; cpu-route: model file parsing conversion
        _ = py_call(floats.append, PythonObject(info.floats[i]))
    var out = Python.list()
    _ = py_call(out.append, w)
    _ = py_call(out.append, strs)
    _ = py_call(out.append, floats)
    return out


def onnx_buffers_py(h: PythonObject) raises -> PythonObject:
    """The address of each buffer of handle h (0 for an empty one), valid until onnx_close."""
    var addrs = onnx_buffers(Int(py=h))
    var out = Python.list()
    for i in range(len(addrs)):  # driver-loop: over the model's files, converted
        _ = py_call(out.append, PythonObject(addrs[i]))
    return out


def onnx_close_py(h: PythonObject) raises -> PythonObject:
    """Release handle h."""
    onnx_close(Int(py=h))
    return PythonObject(0)
