# SPDX-License-Identifier: Apache-2.0
# Python door for ONNX shape resolution (import_resolve.mojo): it converts the
# shapes argument and the typed result, nothing else.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call
from autograd.import_resolve import onnx_resolve


def onnx_resolve_py(h: PythonObject, names: PythonObject, flat: PythonObject) raises -> PythonObject:
    """[words, strings]: the resolved shapes of handle h (layout in import_resolve.mojo's docstring)."""
    var gnames = List[String]()
    for i in range(len(names)):  # driver-loop: over the shapes argument, converted; cpu-route: model file parsing conversion
        gnames.append(String(py=names[i]))
    var gflat = List[Int]()
    for i in range(len(flat)):  # driver-loop: over the shapes argument, converted; cpu-route: model file parsing conversion
        gflat.append(Int(py=flat[i]))
    var r = onnx_resolve(Int(py=h), gnames, gflat)
    var w = Python.list()
    for i in range(len(r.words)):  # driver-loop: over the resolution's words, converted; cpu-route: model file parsing conversion
        _ = py_call(w.append, PythonObject(r.words[i]))
    var strs = Python.list()
    for i in range(len(r.strs)):  # driver-loop: over the graph's strings, converted; cpu-route: model file parsing conversion
        _ = py_call(strs.append, PythonObject(r.strs[i]))
    var out = Python.list()
    _ = py_call(out.append, w)
    _ = py_call(out.append, strs)
    return out
