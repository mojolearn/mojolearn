# SPDX-License-Identifier: Apache-2.0
# Explicit data-dependent control flow over captured graphs. Every branch,
# predicate read, loop iteration and state copy is driven by compiled Mojo.
# No graph format change: component Programs still save/load independently.
from std.python import Python, PythonObject
from std.memory import bitcast
from sequence.exec_trait import Exec
from sequence.ops import FP, Args, OP_COPY
from autograd.ops_api import OpsSession, ops_session
from autograd.program import OpsProgram, PROGRAM_IN_IDS
from autograd.program_tools import program_cse
from autograd.program_runtime import parameter_buffers, replay_values
from autograd.program_pyapi import program_inputs, output_array, _dict, _result
from autograd.cast_ops import upload_cast, upload_cast_i32
from autograd.rankn import ShapeN
from autograd.tape import Tape


def _signature(base: OpsProgram, other: OpsProgram) raises:
    if other.n_in != base.n_in:
        raise Error("control-flow programs must have the same input count")
    for k in range(base.n_in):  # driver-loop: input metadata
        if other.in_kinds[k] != base.in_kinds[k] or other.in_shapes[k].ndim != base.in_shapes[k].ndim:
            raise Error("control-flow programs must have the same input ranks and kinds")


def _execute[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], p: OpsProgram, buffers: List[FP], shapes: List[ShapeN], mut temps: List[Int],
) raises -> List[Int]:
    s._drop_temps(temps)
    temps.clear()
    s.tape.reset()
    var params = parameter_buffers(s, p)
    var flags = List[Bool](length=len(params), fill=False)
    var phs = List[Int]()
    return replay_values(s, p, buffers, shapes, params, flags, temps, phs)


def _predicate[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], p: OpsProgram, buffers: List[FP], shapes: List[ShapeN], mut temps: List[Int],
) raises -> Bool:
    var outputs = _execute(s, p, buffers, shapes, temps)
    if len(outputs) != 1 or s.shapes[s._slot(outputs[0])].count() != 1:
        raise Error("control-flow condition must return one scalar")
    var value = s.scalar(outputs[0])
    if (bitcast[DType.uint32](value) & UInt32(0x7FFFFFFF)) >= UInt32(0x7F800000):
        raise Error("control-flow condition must be finite")
    return value != 0


def _copy_state[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], body: OpsProgram, outputs: List[Int],
) raises -> Tuple[List[FP], List[ShapeN]]:
    var buffers = List[FP]()
    var shapes = List[ShapeN]()
    try:
        for k in range(len(outputs)):  # driver-loop: loop-carried state tensors
            var slot = s._slot(outputs[k])
            var sh = s.shapes[slot]
            if sh.ndim != body.in_shapes[k].ndim:
                raise Error("while_loop body must preserve each state tensor's rank")
            var t = s.tensors[slot]
            var src = s.tape.value_of(s.ex, t)
            var dst = s.ex.alloc(sh.count())
            buffers.append(dst)
            shapes.append(sh)
            var args = Args()
            args.p0 = src
            args.p1 = dst
            s.ex.launch[OP_COPY](args, sh.count())
        # Reusing scratch after this point cannot overwrite a state copy.
        s.ex.sync()
    except err:
        for k in range(len(buffers)):  # driver-loop: acquired state owners
            s.ex.free(buffers[k])
        raise err
    return (buffers^, shapes^)


def ops_program_control_py[E: Exec & Movable & Deinitable, KIND: Int](
    sid: PythonObject, body_id: PythonObject, condition_id: PythonObject,
    other_id: PythonObject, arrays: PythonObject, limit: PythonObject, optimize: PythonObject,
) raises -> PythonObject:
    # other_id >= 1 selects if/else; other_id == 0 selects a bounded while.
    var s = ops_session[E, KIND](Int(py=sid))
    if s[].tape.is_recording():
        raise Error("control flow requires an inference session")
    var body = s[].program(Int(py=body_id))
    var condition = s[].program(Int(py=condition_id))
    var other = body.copy()
    var is_loop = Int(py=other_id) == 0
    var maximum = Int(py=limit)
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
        for k in range(body.n_in):  # driver-loop: state input kinds
            if body.in_kinds[k] == PROGRAM_IN_IDS:
                raise Error("while_loop state inputs must hold floating-point values")
        for k in range(len(body.outs)):  # driver-loop: int32-word outputs cannot become float state
            var ref = body.outs[k]
            if ref >= body.n_in:
                var node = ref - body.n_in
                if body.code[node] == 72 and body.ints[body.i_at[node] + 2] == 0:
                    raise Error("while_loop state outputs must hold floating-point values")
    if Bool(py=optimize):
        body = program_cse(body)
        condition = program_cse(condition)
        other = program_cse(other)
    var inputs = program_inputs(body, arrays, True)
    var buffers = List[FP]()
    var shapes = List[ShapeN]()
    try:
        for k in range(len(inputs)):  # driver-loop: initial state upload
            var sh = inputs[k].shape
            var buf = s[].ex.alloc(sh.count())
            buffers.append(buf)
            shapes.append(sh)
            if body.in_kinds[k] == PROGRAM_IN_IDS:
                upload_cast_i32(s[].ex, buf, inputs[k].src, "control-flow ids")
            else:
                upload_cast(s[].ex, buf, inputs[k].src)
    except err:
        for k in range(len(buffers)):  # driver-loop: input owners
            s[].ex.free(buffers[k])
        raise err
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
                for k in range(len(buffers)):  # driver-loop: previous state owners
                    s[].ex.free(buffers[k])
                buffers = next_state[0].copy()
                shapes = next_state[1].copy()
                iterations += 1
            for k in range(len(buffers)):  # driver-loop: final state download
                var a = output_array(shapes[k])
                s[].ex.download(FP(unsafe_from_address=Int(py=a.ctypes.data)), buffers[k], shapes[k].count())
                result.append(a)
        else:
            var take_body = _predicate(s[], condition, buffers, shapes, temps)
            var chosen = body.copy() if take_body else other.copy()
            var outputs = _execute(s[], chosen, buffers, shapes, temps)
            for k in range(len(outputs)):  # driver-loop: selected branch outputs
                var a = output_array(s[].shapes[s[]._slot(outputs[k])])
                s[].download(outputs[k], FP(unsafe_from_address=Int(py=a.ctypes.data)))
                result.append(a)
    except err:
        s[]._drop_temps(temps)
        swap(held, s[].tape)
        swap(held, s[].prog_tape)
        for k in range(len(buffers)):  # driver-loop: current state owners
            s[].ex.free(buffers[k])
        raise Error("program control flow after " + String(iterations) + " iterations: " + String(err))
    s[]._drop_temps(temps)
    swap(held, s[].tape)
    swap(held, s[].prog_tape)
    for k in range(len(buffers)):  # driver-loop: current state owners
        s[].ex.free(buffers[k])
    if not is_loop:
        return _result(result)
    var report = _dict()
    report.__setitem__("outputs", _result(result))
    report.__setitem__("iterations", iterations)
    report.__setitem__("stopped", stopped)
    return report
