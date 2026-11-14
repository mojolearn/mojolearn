# SPDX-License-Identifier: Apache-2.0
# One native public call; route to the exact Session provider image retaining
# the captured Program dictionary and original owner for the complete call
# (a direct typed call when this image owns the registry: lane
# restore-executor-fast-run 2026-11-13, see the Session request PACK file).
from std.python import Python, PythonObject
from bindings.py_balanced import py_type, py_setitem
from bindings.native_loaded_module_door import native_loaded_module
from sequence.exec_trait import Exec
from providers.python.autograd_program_owner_pyapi_door import program_owner_fields
from providers.python.autograd_session_public_request_pyapi_pack import ops_session_public_call_py
from providers.python.autograd_session_frontend_pyapi_door import ops_session_fields
from providers.python.autograd_program_arrays_pyapi_door import program_numeric_scalar
from providers.python.autograd_program_pyapi_door import _program_run_fixed, PROGRAM_RUN_ENTRY
from providers.python.neural_descriptor_pyapi_door import NativeDescriptorResource
from autograd.ops_api import ops_session_registry_token


# Lane fix-executor-session-rows (2026-11-13): Program.run(X) on the fixed
# route is one binding call straight to `_program_run_fixed`
# (`OpsSession.run_program`: retained scratch, cached replay, each output
# downloaded into its unfilled result array), as 0.8.55's
# `ops_program_run` was. Before this every run first went program door ->
# snapshot dict + request tuple -> Session door (session admission, the
# public-provider checks) -> operation-name lookup and a second handle
# admission -> `ops_program_public_execute_py` (the Program owner admitted
# a second time, its snapshot keys listed and walked, the flags admitted)
# -> `ops_program_execute_py`. Board cell program-run NV 1.04 ms (0.8.55)
# -> 1.46 (ba08dac006, after the restore lane put the run back on
# run_program). The checks that decide anything are kept, once: the exact
# Program and Session owners (program_owner_fields, ops_session_fields),
# the native flag scalars, the inputs container, the session resource in
# this registry image and executor kind, and its handle equal to `_id`.
# Anything else (a flag set, a closed session, a freed program, a session
# of another image) returns 0 here and takes the full route, which keeps
# every refusal and its message. B arm: -D MOJOLEARN_PROGRAM_RUN_ENTRY
# (owed row restore-executor-program-run) runs the full route onto
# program_execute_run, as before the restore lane. Same replay, same
# kernels: no bit change.
def _fixed_run_session[KIND: Int](session: PythonObject, args: PythonObject) raises -> Int:
    """The session handle when this Program.run can go straight to the
    fixed route in this image, else 0."""
    for i in range(1, 4):  # small-loop(3: native execution flags)
        if not program_numeric_scalar(args[i]) or Bool(py=args[i]):
            return 0
    var arrays_type = py_type(args[0])
    if arrays_type is not py_type(Python.tuple()) and arrays_type is not py_type(Python.list()):
        return 0
    var session_fields = ops_session_fields(session)
    var sid = session_fields["_id"]
    if sid is None or py_type(sid) is not py_type(PythonObject(0)):
        return 0
    var lifetime = session_fields["_lifetime"]
    var resource = lifetime.downcast_value_ptr[NativeDescriptorResource]()
    if (len(resource[].context) != 2 or resource[].context[0] != ops_session_registry_token()
            or resource[].context[1] != KIND):
        return 0
    var handle = resource[].handle
    _ = lifetime
    if handle == 0 or handle != Int(py=sid):
        return 0
    return handle


def ops_program_public_call_py[E: Exec & Movable & Deinitable, KIND: Int](
    owner: PythonObject, operation: PythonObject, args: PythonObject,
) raises -> PythonObject:
    if py_type(operation) is not py_type(PythonObject("")) or py_type(args) is not py_type(Python.tuple()):
        raise Error("TypeError: Program requires an exact operation string and argument tuple")
    var fields = program_owner_fields(owner)
    var pid = fields["_id"]
    if pid is not None and py_type(pid) is not py_type(PythonObject(0)):
        raise Error("TypeError: Program handle requires an exact native integer")
    comptime if not PROGRAM_RUN_ENTRY:
        if pid is not None and len(args) == 4 and String(py=operation) == "run":
            var program_id = Int(py=pid)
            var session = fields["_s"]
            var sid = _fixed_run_session[KIND](session, args)
            if sid != 0 and program_id >= 1:
                var answer = _program_run_fixed[E, KIND](sid, program_id, sid, args[0])
                _ = session  # the Session owner stays pinned through the run
                _ = fields
                return answer
    # Do not copy unrelated user fields: destroying a snapshot after execution
    # must not release application finalizers that another thread removed.
    var snapshot = Python.dict()
    py_setitem(snapshot, "_s", fields["_s"])
    py_setitem(snapshot, "_id", pid)
    return ops_session_public_call_py[E, KIND](snapshot["_s"], PythonObject("program_public"),
                                               Python.tuple(owner, snapshot, fields, operation, args))
