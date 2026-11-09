# SPDX-License-Identifier: Apache-2.0
# Python door for explicit control flow (program_control.mojo): argument
# unpacking, input conversion and ndarray results around the typed entries.
from std.python import Python, PythonObject
from bindings.py_balanced import py_setitem
from sequence.exec_trait import Exec
from providers.python.autograd_program_pyapi import program_inputs, program_input, _dict, _result, _integer
from providers.python.autograd_program_arrays_pyapi import host_arrays_py, host_array_py
from autograd.program_control import control_prepare, control_run
from autograd.program_io import ProgramInput
from autograd.ops_api import ops_session
from bindings.py_balanced import py_call
from bindings.native_loaded_module import native_loaded_module


def ops_program_control_py[E: Exec & Movable & Deinitable, KIND: Int](
    sid: PythonObject, body_id: PythonObject, condition_id: PythonObject,
    other_id: PythonObject, condition_sid: PythonObject, other_sid: PythonObject, arrays: PythonObject, limit: PythonObject, optimize: PythonObject,
    check_every: Int = 1, iteration_input: Bool = False,
) raises -> PythonObject:
    # other_id >= 1 selects if/else; other_id == 0 selects a bounded while.
    # `check_every` and `iteration_input` are the bounded loop's
    # (autograd/program_loop.mojo); the if/else ignores them.
    var plan = control_prepare[E, KIND](
        Int(py=sid), Int(py=body_id), Int(py=condition_id), Int(py=other_id),
        Int(py=condition_sid), Int(py=other_sid), _integer(limit, "max_iterations", 0), Bool(py=optimize),
        check_every, iteration_input,
    )
    var inputs: List[ProgramInput]
    if plan.is_loop and plan.iteration_input:
        # The caller passes the carried state only; the iteration register
        # (the body's last input) lives on the device.
        var state_view = plan.body.copy()
        state_view.n_in -= 1
        _ = state_view.in_kinds.pop()
        _ = state_view.in_shapes.pop()
        inputs = program_inputs(state_view, arrays, True)
    else:
        inputs = program_inputs(plan.body, arrays, True)
    var r = control_run[E, KIND](Int(py=sid), plan, inputs^)
    var result = host_arrays_py(r.outputs)
    if not plan.is_loop:
        return _result(result)
    var report = _dict()
    py_setitem(report, "outputs", _result(result))
    py_setitem(report, "iterations", r.iterations)
    py_setitem(report, "stopped", r.stopped)
    return report


def ops_program_decode_py[E: Exec & Movable & Deinitable, KIND: Int](
    sid: PythonObject, body_id: PythonObject, arrays: PythonObject, n: PythonObject, unroll: Int,
    iteration_input: Bool, state: PythonObject,
) raises -> PythonObject:
    """`Program.decode` (lane x1-executor-state-and-step-ops): `state` (None
    or a dict of session state handle -> array) is written into the
    session's state tensors, then the body runs `n` times as the bounded
    loop with no condition and `unroll` iterations between two status
    reads; every committed iteration's state 0 (the next tokens) is
    recorded on the device and returned as `tokens`."""
    var session_id = Int(py=sid)
    if state is not PythonObject(None):
        var s = ops_session[E, KIND](session_id)
        var items = py_call(state.items)
        var pairs = py_call(native_loaded_module("builtins").list, items)
        for i in range(len(pairs)):  # driver-loop: over the caller's state tensors
            var h = _integer(pairs[i][0], "state handle", 1)
            if not s[].is_state(h):
                raise Error("ValueError: decode: handle " + String(h) + " is not a state tensor of this session")
            var input = program_input(pairs[i][1], s[].is_integer(h))
            s[].load_into(h, input.src)
            _ = input^  # keep the converted values alive through the upload
    var plan = control_prepare[E, KIND](
        session_id, Int(py=body_id), 0, 0, session_id, session_id, _integer(n, "n", 0), False,
        unroll, iteration_input, True,
    )
    var inputs: List[ProgramInput]
    if plan.iteration_input:
        var state_view = plan.body.copy()
        state_view.n_in -= 1
        _ = state_view.in_kinds.pop()
        _ = state_view.in_shapes.pop()
        inputs = program_inputs(state_view, arrays, True)
    else:
        inputs = program_inputs(plan.body, arrays, True)
    var r = control_run[E, KIND](session_id, plan, inputs^)
    var report = _dict()
    py_setitem(report, "outputs", _result(host_arrays_py(r.outputs)))
    py_setitem(report, "tokens", host_array_py(r.history[0]))
    py_setitem(report, "iterations", r.iterations)
    return report
