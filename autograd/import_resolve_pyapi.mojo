# SPDX-License-Identifier: Apache-2.0
# Python door for ONNX shape resolution (import_resolve.mojo): it converts the
# shapes argument and the typed result, nothing else.
from std.python import Python, PythonObject
from autograd.import_resolve import onnx_resolve


def onnx_resolve_py(h: PythonObject, names: PythonObject, flat: PythonObject) raises -> PythonObject:
    """[words, strings]: the resolved shapes of handle h (layout in import_resolve.mojo's docstring)."""
    var gnames = List[String]()
    for i in range(len(names)):  # cpu-route: model file parsing (the shapes argument, converted)
        gnames.append(String(py=names[i]))
    var gflat = List[Int]()
    for i in range(len(flat)):  # cpu-route: model file parsing (the shapes argument, converted)
        gflat.append(Int(py=flat[i]))
    var r = onnx_resolve(Int(py=h), gnames, gflat)
    var w = Python.list()
    for i in range(len(r.words)):  # cpu-route: model file parsing (the resolution's words, converted)
        w.append(PythonObject(r.words[i]))
    var strs = Python.list()
    for i in range(len(r.strs)):  # cpu-route: model file parsing (the graph's strings, converted)
        strs.append(PythonObject(r.strs[i]))
    var out = Python.list()
    out.append(w)
    out.append(strs)
    return out
