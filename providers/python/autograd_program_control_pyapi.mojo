# SPDX-License-Identifier: Apache-2.0
# Python door for explicit control flow (program_control.mojo): argument
# unpacking, input conversion and ndarray results around the typed entries.
from std.python import Python, PythonObject
from bindings.py_balanced import py_setitem
from sequence.exec_trait import Exec
from providers.python.autograd_program_pyapi import program_inputs, _dict, _result, _integer
from providers.python.autograd_program_arrays_pyapi import host_arrays_py
from autograd.program_control import control_prepare, control_run
from autograd.program_io import ProgramInput


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
