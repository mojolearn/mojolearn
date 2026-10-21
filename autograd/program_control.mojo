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
