# SPDX-License-Identifier: Apache-2.0
# One native public call; route to the exact Session provider image retaining
# the captured Program dictionary and original owner for the complete call
# (a direct typed call when this image owns the registry: lane
# restore-executor-fast-run 2026-11-13, see the Session request PACK file).
from std.python import Python, PythonObject
from bindings.py_balanced import py_item, py_type, py_setitem
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_cpython_table import cpython_table
from sequence.exec_trait import Exec
from bindings.autograd_keys_unpack import autograd_keys
from providers.python.autograd_program_owner_pyapi_door import program_owner_fields
from providers.python.autograd_session_public_request_pyapi_pack import ops_session_public_call_py
from providers.python.autograd_session_frontend_pyapi_door import ops_session_fields
from providers.python.autograd_program_arrays_pyapi_door import program_numeric_scalar, program_builtin_types
from providers.python.autograd_program_pyapi_door import _program_run_fixed
from providers.python.neural_descriptor_pyapi_door import NativeDescriptorResource
from providers.python.autograd_program_run_route_pyapi_door import program_run_route, program_run_route_record
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
# every refusal and its message. Same replay, same kernels: no bit change.
# (The B arm, the full route onto program_execute_run for every run, lost
# A/B abn9 and is deleted: tombstones at the two doors below.)
# TOMBSTONE: MOJOLEARN_PROGRAM_RUN_ENTRY (unmeasured B arm: the former program_execute_run chain; EXP:2394 RESTORED the run_program route as default) deleted 2026-11-15 by lane/expdel2-08-base; code recoverable at a435c824c4.
# Restore: git apply experiments/removed/MOJOLEARN_PROGRAM_RUN_ENTRY.patch; record in docs/TOMBSTONES.md.
# RESTORED 2026-11-15 by lane/expdel2-restore-owed: owed B arm (owed-pre0867.json), under four weeks old; the deletion was premature.
# Hot-path audit 2026-11-13 (BLOCK pack:47): the exact-type checks below
# compare against type objects the caller already holds (`tuple_type` is the
# class of the exact `args` tuple, `int_type` the class of the exact `_id`
# integer), so no Python list or tuple is built per run to find them.
# `Program.run(*arrays)` always passes a tuple; a list (only a direct
# binding call can pass one) takes the full route, which admits it.
def _fixed_run_session[KIND: Int](
    session: PythonObject, args: PythonObject,
    tuple_type: PythonObject, int_type: PythonObject,
) raises -> Int:
    """The session handle when this Program.run can go straight to the
    fixed route in this image, else 0."""
    for i in range(1, 4):  # small-loop(3: native execution flags)
        if not program_numeric_scalar(args[i]) or Bool(py=args[i]):
            return 0
    if py_type(args[0]) is not tuple_type:
        return 0
    var session_fields = ops_session_fields(session)
    var nk = autograd_keys()
    var sid = py_item(session_fields, nk[].k__id)
    if sid is None or py_type(sid) is not int_type:
        return 0
    var lifetime = py_item(session_fields, nk[].k__lifetime)
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
    # Lane fix-wave2-leftovers: the exact builtin types come from the state
    # connected at Session setup (was: an empty tuple, a "" str and a 0 int
    # built per call to read their types).
    var types = program_builtin_types()
    var tuple_type = types[].tuple_type
    if py_type(operation) is not types[].str_type or py_type(args) is not tuple_type:
        raise Error("TypeError: Program requires an exact operation string and argument tuple")
    # TOMBSTONE: MOJOLEARN_PROGRAM_RUN_ENTRY (B arm loses: every fixed-shape Program.run on program_execute_run; A/B abn9 NVIDIA L40S, program-run 0.551 -> 0.603 ms, 1.10 arm/base) deleted 2026-11-21 by lane/fx-verdicts7; code recoverable at 519388d23a.
    # Restore: git apply experiments/removed/MOJOLEARN_PROGRAM_RUN_ENTRY.fx-verdicts7.patch; record in docs/TOMBSTONES.md.
    # Lane fix-program-run-per-call (2026-11-14): a run the full checks
    # below already admitted for this Program and Session, with neither
    # owner's dictionary nor class changed since, goes straight to the
    # fixed route (autograd_program_run_route_pyapi_door.mojo). Profile
    # r5 program-run: program_owner_fields 22% + ops_session_fields of
    # the in-package samples were these per-run Python reads.
    if len(args) == 4 and String(py=operation) == "run":
        # The exact args tuple keeps its item alive: a borrowed read.
        var inputs = PythonObject(from_borrowed=cpython_table()[].PyTuple_GetItem(args._obj_ptr, 0))
        if py_type(inputs) is tuple_type:
            var kept = program_run_route[KIND](owner, args)
            if kept.sid != 0:
                var answer = _program_run_fixed[E, KIND](kept.sid, kept.pid, kept.sid, inputs)
                _ = owner  # the Program (and through it the Session) stays pinned through the run
                return answer
    var fields = program_owner_fields(owner)
    # Held keys: `_id` and `_s` were a Python str built per read.
    var nk = autograd_keys()
    var pid = py_item(fields, nk[].k__id)
    var int_type = types[].int_type
    if pid is not None and py_type(pid) is not int_type:
        raise Error("TypeError: Program handle requires an exact native integer")
    # TOMBSTONE: MOJOLEARN_PROGRAM_RUN_ENTRY (B arm loses: every fixed-shape Program.run on program_execute_run; A/B abn9 NVIDIA L40S, program-run 0.551 -> 0.603 ms, 1.10 arm/base) deleted 2026-11-21 by lane/fx-verdicts7; code recoverable at 519388d23a.
    # Restore: git apply experiments/removed/MOJOLEARN_PROGRAM_RUN_ENTRY.fx-verdicts7.patch; record in docs/TOMBSTONES.md.
    if pid is not None and len(args) == 4 and String(py=operation) == "run":
        var program_id = Int(py=pid)
        var session = py_item(fields, nk[].k__s)
        var sid = _fixed_run_session[KIND](session, args, tuple_type, int_type)
        if sid != 0 and program_id >= 1:
            program_run_route_record[KIND](owner, fields, session, sid, program_id)
            var answer = _program_run_fixed[E, KIND](sid, program_id, sid, args[0])
            _ = session  # the Session owner stays pinned through the run
            _ = fields
            return answer
    # Do not copy unrelated user fields: destroying a snapshot after execution
    # must not release application finalizers that another thread removed.
    var snapshot = Python.dict()
    var owning = py_item(fields, nk[].k__s)
    py_setitem(snapshot, nk[].k__s, owning)
    py_setitem(snapshot, nk[].k__id, pid)
    return ops_session_public_call_py[E, KIND](owning, PythonObject("program_public"),
                                               Python.tuple(owner, snapshot, fields, operation, args))
