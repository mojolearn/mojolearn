# SPDX-License-Identifier: Apache-2.0
# Python door for the native checkpoint and verification-sidecar writer
# (program_save.mojo): it converts the program object's names and the
# install's version and platform strings, nothing else.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call
from sequence.exec_trait import Exec
from autograd.ops_api import ops_session
from autograd.program_save import program_save_verified
from bindings.native_json import native_path


def _names(owner: PythonObject, attribute: String, count: Int) raises -> List[String]:
    """The first item of each (name, ...) entry of owner.<attribute>, when
    it holds exactly `count` entries; empty otherwise."""
    var out = List[String]()
    var b = Python.import_module("builtins")
    if not Bool(py=py_call(b.hasattr, owner, attribute)):
        return out^
    var entries = py_call(b.getattr, owner, attribute)
    if len(entries) != count:
        return out^
    for k in range(count):  # small-loop(entries: one name per program input or output): argument parsing, no data
        out.append(String(entries[k][0]))
    return out^


def ops_program_save_native_py[E: Exec & Movable & Deinitable, KIND: Int](
    program: PythonObject, path: PythonObject,
) raises -> PythonObject:
    var sid = Int(py=program._s._id)
    var pid = Int(py=program._id)
    var s = ops_session[E, KIND](sid)
    var p = s[].program(pid)
    var normalized = native_path(path)
    var package = Python.import_module("mojolearn._version")
    var uname = Python.import_module("os").uname()
    program_save_verified[E, KIND](
        sid, pid, String(py=normalized), _names(program, "inputs", p.n_in), _names(program, "outputs", len(p.outs)),
        String(package.__version__), String(uname[0]) + " " + String(uname[4]),
    )
    return PythonObject(None)
