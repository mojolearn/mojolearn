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
from providers.python.autograd_pyapi_door import program_info


from providers.python.autograd_import_lower_pyapi_pack import (
    _import_out,
)


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
