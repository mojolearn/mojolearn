# SPDX-License-Identifier: Apache-2.0
# Python door for explicit control flow (program_control.mojo): argument
# unpacking, input upload and ndarray results around the native driver.
from std.python import Python, PythonObject
from std.memory import bitcast
from sequence.exec_trait import Exec
from sequence.ops import FP, Args, OP_COPY
from autograd.ops_api import OpsSession, ops_session
from autograd.program import OpsProgram, PROGRAM_IN_IDS
from autograd.program_optimize import program_optimize
from autograd.program_runtime import parameter_buffers, replay_values
from autograd.program_pyapi import program_inputs, output_array, _dict, _result, _integer
from autograd.cast_ops import upload_cast, upload_cast_i32
from autograd.rankn import ShapeN
from autograd.tape import Tape
from autograd.program_control import _signature, _execute, _predicate, _copy_state


def ops_program_control_py[E: Exec & Movable & Deinitable, KIND: Int](
    sid: PythonObject, body_id: PythonObject, condition_id: PythonObject,
    other_id: PythonObject, condition_sid: PythonObject, other_sid: PythonObject, arrays: PythonObject, limit: PythonObject, optimize: PythonObject,
) raises -> PythonObject:
    # other_id >= 1 selects if/else; other_id == 0 selects a bounded while.
    if Int(py=condition_sid) != Int(py=sid) or Int(py=other_sid) != Int(py=sid):
        raise Error("control-flow programs must belong to the same session")
    var s = ops_session[E, KIND](Int(py=sid))
    if s[].tape.is_recording():
        raise Error("control flow requires an inference session")
    var body = s[].program(Int(py=body_id))
    var condition = s[].program(Int(py=condition_id))
    var other = body.copy()
    var is_loop = Int(py=other_id) == 0
    var maximum = _integer(limit, "max_iterations", 0)
    if not is_loop:
        other = s[].program(Int(py=other_id))
        _signature(body, other)
    elif maximum < 0:
        raise Error("while_loop max_iterations must be nonnegative")
    _signature(body, condition)
    if len(condition.outs) != 1:
        raise Error("control-flow condition must return one scalar")
    if is_loop:
        if len(body.outs) != body.n_in:
            raise Error("while_loop body must return one tensor per input")
        for k in range(body.n_in):  # driver-loop: over state input kinds
            if body.in_kinds[k] == PROGRAM_IN_IDS:
                raise Error("ValueError: while_loop state inputs must hold floating-point values")
        for k in range(len(body.outs)):  # driver-loop: over int32-word outputs cannot become float state
            var value_ref = body.outs[k]
            if value_ref >= body.n_in:
                var node = value_ref - body.n_in
                if body.code[node] == 72 and body.ints[body.i_at[node] + 2] == 0:
                    raise Error("ValueError: while_loop state outputs must hold floating-point values")
    if Bool(py=optimize):
        var body_plan = program_optimize(body)
        var condition_plan = program_optimize(condition)
        var other_plan = program_optimize(other)
        body = body_plan.program.copy()
        condition = condition_plan.program.copy()
        other = other_plan.program.copy()
    var inputs = program_inputs(body, arrays, True)
    var buffers = List[FP]()
    var shapes = List[ShapeN]()
    try:
        for k in range(len(inputs)):  # driver-loop: over initial state upload
            var sh = inputs[k].shape
            var buf = s[].ex.alloc(sh.count())
            buffers.append(buf)
            shapes.append(sh)
            if body.in_kinds[k] == PROGRAM_IN_IDS:
                upload_cast_i32(s[].ex, buf, inputs[k].src, "control-flow ids")
            else:
                upload_cast(s[].ex, buf, inputs[k].src)
    except err:
        for k in range(len(buffers)):  # driver-loop: over input owners
            s[].ex.free(buffers[k])
        raise err
    _ = inputs^  # retain scalar sequence owners until all input uploads finish
    var held = Tape()
    swap(held, s[].prog_tape)
    swap(held, s[].tape)
    var temps = List[Int]()
    var result = Python.list()
    var iterations = 0
    var stopped = False
    try:
        s[].tape.no_grad()
        if is_loop:
            while True:  # native control-flow driver, bounded by the explicit iteration limit
                if not _predicate(s[], condition, buffers, shapes, temps):
                    stopped = True
                    break
                if iterations == maximum:
                    break
                var outputs = _execute(s[], body, buffers, shapes, temps)
                var next_state = _copy_state(s[], body, outputs)
                for k in range(len(buffers)):  # driver-loop: over previous state owners
                    s[].ex.free(buffers[k])
                buffers = next_state[0].copy()
                shapes = next_state[1].copy()
                iterations += 1
            for k in range(len(buffers)):  # driver-loop: over final state download
                var a = output_array(shapes[k])
                s[].ex.download(FP(unsafe_from_address=Int(py=a.__array_interface__["data"][0])), buffers[k], shapes[k].count())
                result.append(a)
        else:
            var take_body = _predicate(s[], condition, buffers, shapes, temps)
            var chosen = body.copy() if take_body else other.copy()
            var outputs = _execute(s[], chosen, buffers, shapes, temps)
            for k in range(len(outputs)):  # driver-loop: over selected branch outputs
                var a = output_array(s[].shapes[s[]._slot(outputs[k])])
                s[].download(outputs[k], FP(unsafe_from_address=Int(py=a.__array_interface__["data"][0])))
                result.append(a)
    except err:
        s[]._drop_temps(temps)
        swap(held, s[].tape)
        swap(held, s[].prog_tape)
        for k in range(len(buffers)):  # driver-loop: over current state owners
            s[].ex.free(buffers[k])
        var message = String(err)
        var context = "program control flow after " + String(iterations) + " iterations: "
        if message.startswith("ValueError: "):
            raise Error("ValueError: " + context + message[byte=11:])
        raise Error(context + message)
    s[]._drop_temps(temps)
    swap(held, s[].tape)
    swap(held, s[].prog_tape)
    for k in range(len(buffers)):  # driver-loop: over current state owners
        s[].ex.free(buffers[k])
    if not is_loop:
        return _result(result)
    var report = _dict()
    report["outputs"] = _result(result)
    report["iterations"] = iterations
    report["stopped"] = stopped
    return report
