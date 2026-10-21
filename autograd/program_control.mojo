# SPDX-License-Identifier: Apache-2.0
# Explicit data-dependent control flow over captured graphs. Every branch,
# predicate read, loop iteration and state copy is driven by compiled Mojo.
# No graph format change: component Programs still save/load independently.
# The Python entry point is program_control_pyapi.mojo.
from std.memory import bitcast
from sequence.exec_trait import Exec
from sequence.ops import FP, Args, OP_COPY
from autograd.ops_api import OpsSession, ops_session
from autograd.program import OpsProgram, PROGRAM_IN_IDS
from autograd.program_optimize import program_optimize
from autograd.program_runtime import parameter_buffers, replay_values
from autograd.cast_ops import upload_cast, upload_cast_i32
from autograd.rankn import ShapeN
from autograd.tape import Tape
from autograd.program_io import ProgramInput, HostArray


def _signature(base: OpsProgram, other: OpsProgram) raises:
    if other.n_in != base.n_in:
        raise Error("control-flow programs must have the same input count")
    for k in range(base.n_in):  # driver-loop: over input metadata
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
        for k in range(len(outputs)):  # driver-loop: over loop-carried state tensors
            var slot = s._slot(outputs[k])
            var sh = s.shapes[slot]
            if s.int_words[slot]:
                raise Error("ValueError: while_loop state outputs must hold floating-point values")
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
        for k in range(len(buffers)):  # driver-loop: over acquired state owners
            s.ex.free(buffers[k])
        raise err
    return (buffers^, shapes^)


# ---------------------------------------------------------------- entries
# The typed entries behind program_control_pyapi.mojo: control_prepare
# checks and optimizes the component programs, the door converts the input
# arrays against the prepared body, and control_run uploads, drives the
# branch or the bounded loop and downloads the results to the host.
struct ControlPlan(Movable):
    var body: OpsProgram
    var condition: OpsProgram
    var other: OpsProgram
    var is_loop: Bool
    var maximum: Int

    def __init__(out self, var body: OpsProgram, var condition: OpsProgram, var other: OpsProgram,
                 is_loop: Bool, maximum: Int):
        self.body = body^
        self.condition = condition^
        self.other = other^
        self.is_loop = is_loop
        self.maximum = maximum


struct ControlResult(Movable):
    var outputs: List[HostArray]
    var iterations: Int
    var stopped: Bool

    def __init__(out self):
        self.outputs = List[HostArray]()
        self.iterations = 0
        self.stopped = False


def control_prepare[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, body_id: Int, condition_id: Int, other_id: Int, condition_sid: Int, other_sid: Int,
    maximum: Int, optimize: Bool,
) raises -> ControlPlan:
    # other_id >= 1 selects if/else; other_id == 0 selects a bounded while.
    if condition_sid != sid or other_sid != sid:
        raise Error("control-flow programs must belong to the same session")
    var s = ops_session[E, KIND](sid)
    if s[].tape.is_recording():
        raise Error("control flow requires an inference session")
    var body = s[].program(body_id)
    var condition = s[].program(condition_id)
    var other = body.copy()
    var is_loop = other_id == 0
    if not is_loop:
        other = s[].program(other_id)
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
    if optimize:
        var body_plan = program_optimize(body)
        var condition_plan = program_optimize(condition)
        var other_plan = program_optimize(other)
        body = body_plan.program.copy()
        condition = condition_plan.program.copy()
        other = other_plan.program.copy()
    return ControlPlan(body^, condition^, other^, is_loop, maximum)


def control_run[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, plan: ControlPlan, var inputs: List[ProgramInput],
) raises -> ControlResult:
    var s = ops_session[E, KIND](sid)
    var body = plan.body.copy()
    var condition = plan.condition.copy()
    var other = plan.other.copy()
    var is_loop = plan.is_loop
    var maximum = plan.maximum
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
    var result = ControlResult()
    var iterations = 0
    var stopped = False
    try:
        s[].tape.no_grad()
        if is_loop:
            while True:  # driver-loop: over control-flow iterations, bounded by the explicit iteration limit
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
                var a = HostArray(shapes[k])
                s[].ex.download(FP(unsafe_from_address=Int(a.data.unsafe_ptr())), buffers[k], shapes[k].count())
                result.outputs.append(a^)
        else:
            var take_body = _predicate(s[], condition, buffers, shapes, temps)
            var chosen = body.copy() if take_body else other.copy()
            var outputs = _execute(s[], chosen, buffers, shapes, temps)
            for k in range(len(outputs)):  # driver-loop: over selected branch outputs
                var a = HostArray(s[].shapes[s[]._slot(outputs[k])])
                s[].download(outputs[k], FP(unsafe_from_address=Int(a.data.unsafe_ptr())))
                result.outputs.append(a^)
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
    result.iterations = iterations
    result.stopped = stopped
    return result^
