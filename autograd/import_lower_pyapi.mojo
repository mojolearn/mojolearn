# SPDX-License-Identifier: Apache-2.0
# Python doors for the import lowering (import_lower.mojo): each converts its
# arguments and the typed result to the established lists, nothing else.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call
from sequence.exec_trait import Exec
from autograd.import_lower import (
    IMPORT_BAD_VALUE, IMPORT_OK, IMPORT_PROBLEMS, IMPORT_READ_FAILED, OnnxImport, _Found,
    onnx_import, onnx_import_file,
)
from autograd.pyapi import program_info


def _int_list(p: PythonObject) raises -> List[Int]:
    var out = List[Int]()
    for i in range(len(p)):  # driver-loop: over argument words, converted; cpu-route: model graph lowered on the host conversion
        out.append(Int(py=p[i]))
    return out^


def _str_list(p: PythonObject) raises -> List[String]:
    var out = List[String]()
    for i in range(len(p)):  # driver-loop: over argument strings, converted; cpu-route: model graph lowered on the host conversion
        out.append(String(py=p[i]))
    return out^


def _problems_out(found: List[_Found]) raises -> PythonObject:
    var out = Python.list()
    for i in range(len(found)):  # driver-loop: over the reported problems; cpu-route: model graph lowered on the host conversion
        var p = Python.list()
        _ = py_call(p.append, PythonObject(found[i].name))
        if found[i].has_op:
            _ = py_call(p.append, PythonObject(found[i].op))
        else:
            _ = py_call(p.append, PythonObject(None))
        _ = py_call(p.append, PythonObject(found[i].why))
        _ = py_call(out.append, p)
    return out


def _dims_py(v: List[Int]) raises -> PythonObject:
    var d = Python.list()
    for i in range(len(v)):  # driver-loop: over the axes of one shape; cpu-route: model graph lowered on the host conversion
        _ = py_call(d.append, PythonObject(v[i]))
    return d


def onnx_import_py[E: Exec & Movable & Deinitable, KIND: Int](
    h: PythonObject, sid: PythonObject, names: PythonObject, flat: PythonObject, addrs: PythonObject,
    source: PythonObject,
) raises -> PythonObject:
    """Import graph `h` into inference session `sid` (import_lower.mojo's
    docstring). Returns [status, ...]."""
    var r = onnx_import[E, KIND](Int(py=h), Int(py=sid), _str_list(names), _int_list(flat),
                                 _int_list(addrs), String(py=source))
    return _import_out(r)


def onnx_import_file_py[E: Exec & Movable & Deinitable, KIND: Int](
    path: PythonObject, sid: PythonObject, names: PythonObject, flat: PythonObject, source: PythonObject,
) raises -> PythonObject:
    """Read the ONNX model at `path` and import it into inference session
    `sid` (import_lower.mojo onnx_import_file). Returns [status, ...]."""
    var r = onnx_import_file[E, KIND](String(py=path), Int(py=sid), _str_list(names), _int_list(flat),
                                      String(py=source))
    return _import_out(r)


def _import_out(r: OnnxImport) raises -> PythonObject:
    """[status, ...] of an import (import_lower.mojo's docstring)."""
    var res = Python.list()
    if r.status == IMPORT_READ_FAILED:
        _ = py_call(res.append, PythonObject(IMPORT_READ_FAILED))
        _ = py_call(res.append, PythonObject(r.read_status))
        _ = py_call(res.append, PythonObject(r.message))
        _ = py_call(res.append, PythonObject(1 if r.missing else 0))
        return res
    if r.status == IMPORT_BAD_VALUE:
        _ = py_call(res.append, PythonObject(IMPORT_BAD_VALUE))
        _ = py_call(res.append, PythonObject(r.message))
        return res
    if r.status == IMPORT_PROBLEMS:
        _ = py_call(res.append, PythonObject(IMPORT_PROBLEMS))
        _ = py_call(res.append, _problems_out(r.problems))
        return res
    var in_meta = Python.list()
    for q in range(len(r.in_names)):  # driver-loop: over graph inputs; cpu-route: model graph lowered on the host conversion
        var m = Python.list()
        _ = py_call(m.append, PythonObject(r.in_names[q]))
        _ = py_call(m.append, _dims_py(r.in_dims[q]))
        _ = py_call(m.append, PythonObject(r.in_dtypes[q]))
        _ = py_call(in_meta.append, m)
    var out_meta = Python.list()
    for i in range(len(r.out_names)):  # driver-loop: over graph outputs; cpu-route: model graph lowered on the host conversion
        var m = Python.list()
        _ = py_call(m.append, PythonObject(r.out_names[i]))
        _ = py_call(m.append, _dims_py(r.out_dims[i]))
        _ = py_call(m.append, PythonObject(r.out_int32[i]))
        _ = py_call(out_meta.append, m)
    _ = py_call(res.append, PythonObject(IMPORT_OK))
    _ = py_call(res.append, program_info(r.pid, r.prog))
    _ = py_call(res.append, in_meta)
    _ = py_call(res.append, out_meta)
    return res
