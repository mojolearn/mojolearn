# SPDX-License-Identifier: Apache-2.0
# The owning Session image completes the whole public Program operation.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_setitem
from bindings.native_loaded_module import native_loaded_module
from bindings.native_json import native_path
from sequence.exec_trait import Exec
from autograd.ops_api import ops_session
from autograd.control_checkpoint import control_save
from providers.python.autograd_program_owner_pyapi import program_owner_fields, program_native_sequence, program_release_fields
from providers.python.autograd_program_arrays_pyapi import program_numeric_scalar, program_admit_value, uint64_input
from providers.python.autograd_program_pyapi import ops_program_execute_py, ops_program_inspect_py, ops_program_fit_native_py, _integer
from providers.python.autograd_program_control_pyapi import ops_program_control_py
from providers.python.autograd_program_control_grad_pyapi import ops_program_control_vjp_py
from providers.python.autograd_program_save_pyapi import ops_program_save_fields_py
from providers.python.autograd_owned_program_public_pyapi import ops_owned_program_create_request_py


def _option(value: PythonObject) raises:
    if not program_numeric_scalar(value):
        raise Error("TypeError: Program flags require native boolean or numeric scalars")


def _component(owner: PythonObject, session: PythonObject) raises -> PythonObject:
    var fields = program_owner_fields(owner)
    if fields["_s"] is not session:
        raise Error("ValueError: control and training components must belong to the same session")
    return PythonObject(_integer(fields["_id"], "program", 1))


def _arguments(args: PythonObject, count: Int) raises:
    if py_type(args) is not py_type(Python.tuple()) or len(args) != count:
        raise Error("TypeError: invalid native Program operation arguments")


def ops_program_public_execute_py[E: Exec & Movable & Deinitable, KIND: Int](
    session: PythonObject, state: PythonObject, request: PythonObject,
) raises -> PythonObject:
    _arguments(request, 5)
    var owner = request[0]
    var fields = request[1]
    var output_fields = request[2]
    if py_type(fields) is not py_type(Python.dict()) or py_type(output_fields) is not py_type(Python.dict()):
        raise Error("TypeError: Program requires native captured storage")
    if output_fields is not program_owner_fields(owner):
        raise Error("TypeError: Program output storage differs from its admitted owner")
    var keys = py_call(native_loaded_module("builtins").list, fields)
    for i in range(len(keys)):  # driver-loop: direct native callers cannot supply key comparison callbacks
        if py_type(keys[i]) is not py_type(PythonObject("")):
            raise Error("TypeError: Program snapshot requires native string keys")
    if fields["_id"] is not output_fields["_id"]:
        raise Error("TypeError: Program snapshot differs from its admitted handle")
    if fields["_id"] is not None and py_type(fields["_id"]) is not py_type(PythonObject(0)):
        raise Error("TypeError: Program handle requires an exact native integer")
    if py_type(request[3]) is not py_type(PythonObject("")):
        raise Error("TypeError: Program requires an exact operation string")
    var name = String(py=request[3])
    var args = request[4]
    if fields["_s"] is not session or output_fields["_s"] is not session:
        raise Error("ValueError: Program belongs to another Session")
    if name == "free":
        _arguments(args, 0)
        if state["_id"] is not None and fields["_id"] is not None:
            var s = ops_session[E, KIND](Int(py=state["_id"]))
            s[].program_free(_integer(fields["_id"], "program", 1))
        py_setitem(output_fields, "_id", PythonObject(None))
        return PythonObject(None)
    if state["_id"] is None:
        raise Error("ValueError: this session is closed")
    var sid = state["_id"]
    var pid = PythonObject(_integer(fields["_id"], "program", 1))
    if name == "partition" or name == "distribute":
        _arguments(args, 2)
        return ops_owned_program_create_request_py(state, Int(py=pid), name, args)
    if name == "run" or name == "profile" or name == "run_dynamic":
        _arguments(args, 4)
        program_native_sequence(args[0])
        for i in range(1, 4):  # small-loop(3: native execution flags)
            _option(args[i])
        return ops_program_execute_py[E, KIND](sid, pid, sid, args[0], args[1], args[2], args[3])
    if name == "inspect":
        _arguments(args, 1)
        _option(args[0])
        return ops_program_inspect_py[E, KIND](sid, pid, args[0])
    if name == "run_if" or name == "while_loop" or name == "vjp_if" or name == "vjp_while":
        _arguments(args, 7)
        var condition = _component(args[0], session)
        if (name == "run_if" or name == "vjp_if") and args[1] is None:
            raise Error("TypeError: a conditional Program operation requires an otherwise Program")
        var other = PythonObject(0)
        if args[1] is not None:
            other = _component(args[1], session)
        program_native_sequence(args[2])
        var limit = PythonObject(_integer(args[3], "max_iterations", 0))
        _option(args[4])
        if name == "vjp_if" or name == "vjp_while":
            # Type admission precedes the VJP's len/index of multi-output seeds
            # and handle selections; buffer metadata is captured by its input
            # owner once, when it is consumed.
            program_admit_value(args[5])
            program_admit_value(args[6])
            return ops_program_control_vjp_py[E, KIND](sid, pid,
                Python.tuple(sid, condition, sid, other, args[5], args[6]), args[2], limit)
        return ops_program_control_py[E, KIND](sid, pid, condition, other, sid, sid, args[2], limit, args[4])
    if name == "fit" or name == "fit_if":
        _arguments(args, 5)
        var condition = 0
        var otherwise = 0
        if name == "fit_if":
            condition = Int(py=_component(args[0], session))
            otherwise = Int(py=_component(args[1], session))
        program_native_sequence(args[2])
        var options = args[4]
        _arguments(options, 8)
        program_admit_value(options)
        if py_type(options[2]) is not py_type(PythonObject("")):
            raise Error("TypeError: Program loss requires an exact native string")
        _option(options[5])
        _option(options[7])
        var replaced = Python.tuple("loss_curve_", "n_iter_", "branch_counts_") if name == "fit_if" else Python.tuple("loss_curve_", "n_iter_")
        program_release_fields(output_fields, replaced)
        var result = ops_program_fit_native_py[E, KIND](sid, pid, args[2], args[3],
            options[0], options[1], options[2], options[3], options[4], options[5], options[6], options[7],
            condition, otherwise)
        py_setitem(output_fields, "loss_curve_", result[0])
        py_setitem(output_fields, "n_iter_", result[1])
        if name == "fit_if":
            py_setitem(output_fields, "branch_counts_", result[2])
        return owner
    if name == "save":
        _arguments(args, 1)
        return ops_program_save_fields_py[E, KIND](fields, state, args[0], output_fields)
    if name == "save_control":
        _arguments(args, 6)
        var condition = Int(py=_component(args[1], session))
        var loop = args[2] is None
        var other = 0
        var limit = 0
        if loop:
            limit = _integer(args[3], "max_iterations", 0)
        else:
            other = Int(py=_component(args[2], session))
        var optimizer = 0 if args[4] is None else _integer(args[4], "optimizer", 1)
        var seed = uint64_input(args[5], "seed")
        var path = native_path(args[0])
        return PythonObject(control_save[E, KIND](Int(py=sid), Int(py=sid), Int(py=pid),
            Int(py=sid), condition, loop, 0 if loop else Int(py=sid), other,
            args[3] is not None, limit, String(py=path), optimizer, seed))
    raise Error("ValueError: unsupported native Program operation " + name)
