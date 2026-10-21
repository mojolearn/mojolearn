# SPDX-License-Identifier: Apache-2.0
# Python doors for the import lowering (import_lower.mojo): each converts its
# arguments and the typed result to the established lists, nothing else.
from std.python import Python, PythonObject
from sequence.exec_trait import Exec
from autograd.import_lower import (
    IMPORT_BAD_VALUE, IMPORT_OK, IMPORT_PROBLEMS, _Found, onnx_from_words, onnx_import,
)
from autograd.pyapi import program_info


def _int_list(p: PythonObject) raises -> List[Int]:
    var out = List[Int]()
    for i in range(len(p)):  # cpu-route: model graph lowered on the host (argument words, converted)
        out.append(Int(py=p[i]))
    return out^


def _str_list(p: PythonObject) raises -> List[String]:
    var out = List[String]()
    for i in range(len(p)):  # cpu-route: model graph lowered on the host (argument strings, converted)
        out.append(String(py=p[i]))
    return out^


def _problems_out(found: List[_Found]) raises -> PythonObject:
    var out = Python.list()
    for i in range(len(found)):  # cpu-route: model graph lowered on the host (one per problem)
        var p = Python.list()
        p.append(PythonObject(found[i].name))
        if found[i].has_op:
            p.append(PythonObject(found[i].op))
        else:
            p.append(PythonObject(None))
        p.append(PythonObject(found[i].why))
        out.append(p)
    return out


def _dims_py(v: List[Int]) raises -> PythonObject:
    var d = Python.list()
    for i in range(len(v)):  # cpu-route: model graph lowered on the host (one per axis)
        d.append(PythonObject(v[i]))
    return d


def onnx_import_py[E: Exec & Movable & Deinitable, KIND: Int](
    h: PythonObject, sid: PythonObject, names: PythonObject, flat: PythonObject, addrs: PythonObject,
    source: PythonObject,
) raises -> PythonObject:
    """Import graph `h` into inference session `sid` (import_lower.mojo's
    docstring). Returns [status, ...]."""
    var r = onnx_import[E, KIND](Int(py=h), Int(py=sid), _str_list(names), _int_list(flat),
                                 _int_list(addrs), String(py=source))
    var res = Python.list()
    if r.status == IMPORT_BAD_VALUE:
        res.append(PythonObject(IMPORT_BAD_VALUE))
        res.append(PythonObject(r.message))
        return res
    if r.status == IMPORT_PROBLEMS:
        res.append(PythonObject(IMPORT_PROBLEMS))
        res.append(_problems_out(r.problems))
        return res
    var in_meta = Python.list()
    for q in range(len(r.in_names)):  # cpu-route: model graph lowered on the host (graph inputs)
        var m = Python.list()
        m.append(PythonObject(r.in_names[q]))
        m.append(_dims_py(r.in_dims[q]))
        m.append(PythonObject(r.in_dtypes[q]))
        in_meta.append(m)
    var out_meta = Python.list()
    for i in range(len(r.out_names)):  # cpu-route: model graph lowered on the host (graph outputs)
        var m = Python.list()
        m.append(PythonObject(r.out_names[i]))
        m.append(_dims_py(r.out_dims[i]))
        m.append(PythonObject(r.out_int32[i]))
        out_meta.append(m)
    res.append(PythonObject(IMPORT_OK))
    res.append(program_info(r.pid, r.prog))
    res.append(in_meta)
    res.append(out_meta)
    return res


def onnx_from_words_py(words: PythonObject, strings: PythonObject, floats: PythonObject,
                       addrs: PythonObject) raises -> PythonObject:
    """A graph handle from the `onnx_info` word layout, its strings, its
    floats and the address of each buffer. Returns the handle (>= 1)."""
    var strs = _str_list(strings)
    var fl = List[Float64]()
    for i in range(len(floats)):  # cpu-route: model graph description (attribute numbers, converted)
        fl.append(Float64(py=floats[i]))
    var bufs = _int_list(addrs)
    return PythonObject(onnx_from_words(_int_list(words), strs, fl, bufs))
