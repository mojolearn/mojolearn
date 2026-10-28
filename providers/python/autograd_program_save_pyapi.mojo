# SPDX-License-Identifier: Apache-2.0
# Provider boundary for native checkpoints and verification sidecars. Admit
# the exact owner and passive metadata before reading any user-set field.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_fields, py_get
from bindings.native_loaded_module import native_loaded_module
from sequence.exec_trait import Exec
from autograd.ops_api import ops_session, ops_session_registry_token
from autograd.program_save import program_save_verified
from bindings.native_json import native_path
from providers.python.autograd_import_metadata_pyapi import _fields
from providers.python.autograd_session_frontend_pyapi import ops_session_fields
from providers.python.autograd_program_arrays_pyapi import program_admission_context
from providers.python.neural_descriptor_pyapi import neural_descriptor_resource_handle


def _save_fields(program: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var layouts = context["layouts"]
    if py_type(layouts) is not b.tuple:
        raise Error("TypeError: Program save requires its connected native layouts")
    for i in range(len(layouts)):  # small-loop(layouts: connected Program classes): exact owner metadata only
        var layout = layouts[i]
        if py_type(layout) is not b.dict:
            raise Error("TypeError: Program save requires a native layout dictionary")
        if py_type(program) is layout["cls"]:
            return py_call(b.dict, _fields(program, layout))
    raise Error("TypeError: save requires an exact Program or ImportedProgram")


def _names(fields: PythonObject, attribute: String, count: Int) raises -> List[String]:
    # Preserve unnamed/count-mismatch fallback without running hasattr,
    # descriptors, len/index hooks, or arbitrary name string conversions.
    var out = List[String]()
    var b = native_loaded_module("builtins")
    var entries = py_get(fields, attribute)
    if entries is None:
        return out^
    if py_type(entries) is not b.list and py_type(entries) is not b.tuple:
        raise Error("TypeError: program names require an exact list or tuple")
    if len(entries) != count:
        return out^
    for k in range(count):  # driver-loop: over admitted input/output name descriptors
        var row = entries[k]
        if (py_type(row) is not b.list and py_type(row) is not b.tuple) or len(row) == 0:
            raise Error("TypeError: each program name requires a native descriptor tuple or list")
        if py_type(row[0]) is not b.str:
            raise Error("TypeError: program names require exact native strings")
        out.append(String(py=row[0]))
    return out^


def _save_provenance(context: PythonObject) raises -> Tuple[String, String]:
    var b = native_loaded_module("builtins")
    var version = context["version"]
    var platform = context["platform"]
    if py_type(version) is not b.str or py_type(platform) is not b.str:
        raise Error("TypeError: checkpoint provenance requires exact native strings")
    return (String(py=version), String(py=platform))


def ops_program_save_native_py[E: Exec & Movable & Deinitable, KIND: Int](
    program: PythonObject, path: PythonObject,
) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var context = program_admission_context()["program_save_context"]
    if py_type(context) is not b.dict:
        raise Error("TypeError: Program save requires its native admission context")
    var fields = _save_fields(program, context)
    var state = ops_session_fields(fields["_s"])
    if state["_id"] is None:
        raise Error("ValueError: this session is closed")
    if py_type(state["_id"]) is not b.int or py_type(fields["_id"]) is not b.int:
        raise Error("TypeError: program and Session handles require exact native integers")
    var sid = Int(py=state["_id"])
    var pid = Int(py=fields["_id"])
    var scope: List[Int] = [ops_session_registry_token(), KIND]
    if neural_descriptor_resource_handle(state["_lifetime"], scope) != sid:
        raise Error("TypeError: Program save belongs to another native Session owner")
    var s = ops_session[E, KIND](sid)
    var p = s[].program(pid)
    var normalized = native_path(path)
    var input_names = _names(fields, "inputs", p.n_in)
    var output_names = _names(fields, "outputs", len(p.outs))
    var provenance = _save_provenance(context)
    program_save_verified[E, KIND](sid, pid, String(py=normalized), input_names,
                                  output_names, provenance[0], provenance[1])
    _ = fields
    _ = state
    return PythonObject(None)
