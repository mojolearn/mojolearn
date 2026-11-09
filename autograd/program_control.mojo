# SPDX-License-Identifier: Apache-2.0
# Explicit data-dependent control flow over captured graphs. Every branch,
# predicate read, loop iteration and state copy is driven by compiled Mojo.
# No graph format change: component Programs still save/load independently.
# The Python entry point is program_control_pyapi.mojo.
from std.memory import bitcast
from sequence.exec_trait import Exec
from sequence.ops import FP, Args, OP_COPY
from autograd.program_loop import (
    LOOP_DONE, LOOP_ITER, LOOP_STOPPED, LOOP_BAD, LOOP_WORDS, LOOP_MAX_EXACT, EXECUTOR_TIMERS,
    OP_LOOP_TICK, OP_LOOP_COMMIT, OP_LOOP_ADVANCE, OP_LOOP_APPEND, OP_LOOP_ITER_WORD, LoopTimers, loop_check_every,
)
from autograd.ops_api import OpsSession, ops_session
from autograd.program import OpsProgram, PROGRAM_IN_IDS
from autograd.program_optimize import program_optimize
from autograd.program_runtime import parameter_buffers, replay_values_prepared
from autograd.control_replay_metadata import ControlReplay
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
    mut s: OpsSession[E], p: OpsProgram, mut replay: ControlReplay,
    buffers: List[FP], shapes: List[ShapeN], mut temps: List[Int],
) raises -> List[Int]:
    s._drop_temps(temps)
    temps.clear()
    s.tape.reset()
    var params = parameter_buffers(s, p)
    if not replay.ready:
        replay.no_grad = List[Bool](length=len(params), fill=False)
        replay.prepare(p)
    replay.parameters.clear()
    return replay_values_prepared(s, p, replay.graph, buffers, shapes, params,
                                  replay.no_grad, temps, replay.parameters)


def _predicate[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], p: OpsProgram, mut replay: ControlReplay,
    buffers: List[FP], shapes: List[ShapeN], mut temps: List[Int],
) raises -> Bool:
    var outputs = _execute(s, p, replay, buffers, shapes, temps)
    if len(outputs) != 1 or s.shapes[s._slot(outputs[0])].count() != 1:
        raise Error("control-flow condition must return one scalar")
    var value = s.scalar(outputs[0])
    if (bitcast[DType.uint32](value) & UInt32(0x7FFFFFFF)) >= UInt32(0x7F800000):
        raise Error("control-flow condition must be finite")
    return value != 0


# TOMBSTONE: _copy_state (REFACTORED: per-iteration copy of the carried state into fresh buffers plus a drain) deleted 2026-11-08 by lane/fix-d3-executor-loops; code recoverable at 22ac89b54c.
# The bounded loop now carries its state in place (`_commit_state`, OP_LOOP_COMMIT) and stops on the device (OP_LOOP_TICK).
# Restore: git apply experiments/removed/EXECUTOR_LOOP_DRIVER_LOOPS.patch; record in docs/TOMBSTONES.md.


def _lap[E: Exec & Movable & Deinitable](mut s: OpsSession[E], mut timers: LoopTimers, name: String) raises:
    """Per-op timer mark (`MOJOLEARN_EXECUTOR_TIMERS`): drain the executor and
    charge the time since the last mark to `name`. Compiled out when off."""
    comptime if EXECUTOR_TIMERS:
        s.ex.sync()
        timers.lap(name)


def _free_loop_owners[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], extras: List[FP], scratch: List[FP], scratch_n: List[Int],
):
    """Return the loop's register block and staging buffers (quiet: this runs
    on the error path too)."""
    for k in range(len(scratch_n)):  # driver-loop: over loop-carried state tensors (staging slots)
        if scratch_n[k] > 0:
            try:
                s.ex.free(scratch[k])
            except:
                pass
    for k in range(len(extras)):  # driver-loop: over the loop's register blocks
        try:
            s.ex.free(extras[k])
        except:
            pass


def _overlaps(addr: Int, n: Int, base: Int, m: Int) -> Bool:
    """[addr, addr + n) and [base, base + m) (float32 words) share a word."""
    return addr < base + 4 * m and base < addr + 4 * n


def _commit_state[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], body: OpsProgram, outputs: List[Int], words: FP,
    mut buffers: List[FP], mut shapes: List[ShapeN], mut scratch: List[FP], mut scratch_n: List[Int],
    gated_only: Bool,
) raises:
    """Write the body's outputs into the carried state buffers in place, each
    behind the register block's gate (OP_LOOP_COMMIT, a bit copy). An output
    that reads another state buffer is staged first, so no commit overwrites
    a value a later commit still reads. With `gated_only` (several iterations
    between flag reads) every output must keep its state tensor's shape;
    otherwise (the gate is known open) a resized output takes a new buffer."""
    var n_state = len(outputs)
    var srcs = List[FP]()
    var out_shapes = List[ShapeN]()
    for k in range(n_state):  # driver-loop: over loop-carried state tensors
        var slot = s._slot(outputs[k])
        var sh = s.shapes[slot]
        # lane x1: int32-word state (token ids, positions) is carried like
        # float state; each output keeps its state input's kind.
        if s.int_words[slot] != (body.in_kinds[k] == PROGRAM_IN_IDS):
            raise Error("ValueError: while_loop state output " + String(k) + " must hold the kind of its state input"
                        + " (integer ids for an ids input, floating-point values otherwise)")
        if sh.ndim != body.in_shapes[k].ndim:
            raise Error("while_loop body must preserve each state tensor's rank")
        if gated_only and not sh.same(shapes[k]):
            raise Error("ValueError: while_loop with check_every above 1 needs a body that keeps each state tensor's shape")
        srcs.append(s.tape.value_of(s.ex, s.tensors[slot]))
        out_shapes.append(sh)
    # Stage the outputs that read a state buffer other than their own (or a
    # shifted view of their own): every staging copy is enqueued before the
    # first commit.
    for k in range(n_state):  # driver-loop: over loop-carried state tensors
        var addr = Int(srcs[k])
        var n = out_shapes[k].count()
        var staged = False
        for j in range(n_state):  # driver-loop: over loop-carried state tensors (alias test, addresses only)
            if j == k and addr == Int(buffers[k]):
                continue
            if _overlaps(addr, n, Int(buffers[j]), shapes[j].count()):
                staged = True
        if staged:
            # One staging buffer per state slot, made at its first use and
            # kept for the loop (`scratch_n[k]` words; 0 = none yet).
            if scratch_n[k] < n:
                if scratch_n[k] > 0:
                    s.ex.free(scratch[k])
                scratch[k] = s.ex.alloc(n)
                scratch_n[k] = n
            var a = Args()
            a.p0 = srcs[k]
            a.p1 = scratch[k]
            s.ex.launch[OP_COPY](a, n)
            srcs[k] = scratch[k]
    for k in range(n_state):  # driver-loop: over loop-carried state tensors
        if Int(srcs[k]) == Int(buffers[k]):
            if not gated_only:
                shapes[k] = out_shapes[k]
            continue
        var n = out_shapes[k].count()
        if n != shapes[k].count():
            # Only reached with the gate known open (gated_only is False).
            var fresh = s.ex.alloc(n)
            var c = Args()
            c.p0 = srcs[k]
            c.p1 = fresh
            s.ex.launch[OP_COPY](c, n)
            s.ex.free(buffers[k])
            buffers[k] = fresh
            shapes[k] = out_shapes[k]
            continue
        var a = Args()
        a.p0 = srcs[k]
        a.p1 = buffers[k]
        a.p2 = words
        s.ex.launch[OP_LOOP_COMMIT](a, n)
        if not gated_only:
            shapes[k] = out_shapes[k]


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
    #: iterations between two host reads of the loop's register block
    var check_every: Int
    #: the body and the condition take the iteration register as a last input
    var iteration_input: Bool
    #: lane x1: False for a loop with no condition program (it runs to the
    #: bound: a decode of a fixed token count)
    var has_condition: Bool
    #: lane x1: record state 0 after every committed iteration (the tokens
    #: of a decode loop), returned as ControlResult.history
    var history: Bool

    def __init__(out self, var body: OpsProgram, var condition: OpsProgram, var other: OpsProgram,
                 is_loop: Bool, maximum: Int, check_every: Int = 1, iteration_input: Bool = False,
                 has_condition: Bool = True, history: Bool = False):
        self.body = body^
        self.condition = condition^
        self.other = other^
        self.is_loop = is_loop
        self.maximum = maximum
        self.check_every = check_every
        self.iteration_input = iteration_input
        self.has_condition = has_condition
        self.history = history


struct ControlResult(Movable):
    var outputs: List[HostArray]
    var iterations: Int
    var stopped: Bool
    #: ControlPlan.history: (iterations, state 0's element count), one row a
    #: committed iteration; empty otherwise
    var history: List[HostArray]

    def __init__(out self):
        self.outputs = List[HostArray]()
        self.iterations = 0
        self.stopped = False
        self.history = List[HostArray]()


def control_prepare[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, body_id: Int, condition_id: Int, other_id: Int, condition_sid: Int, other_sid: Int,
    maximum: Int, optimize: Bool, check_every: Int = 1, iteration_input: Bool = False, history: Bool = False,
) raises -> ControlPlan:
    # other_id >= 1 selects if/else; other_id == 0 selects a bounded while.
    # condition_id == 0 (a loop only, lane x1): no condition, the loop runs
    # to max_iterations (Program.decode).
    if condition_sid != sid or other_sid != sid:
        raise Error("control-flow programs must belong to the same session")
    var s = ops_session[E, KIND](sid)
    if s[].tape.is_recording():
        raise Error("control flow requires an inference session")
    var body = s[].program(body_id)
    var is_loop = other_id == 0
    var has_condition = condition_id != 0
    if not has_condition and not is_loop:
        raise Error("ValueError: run_if needs a condition program")
    if history and not is_loop:
        raise Error("ValueError: a token record needs a loop")
    var condition = s[].program(condition_id) if has_condition else body.copy()
    var other = body.copy()
    if not is_loop:
        other = s[].program(other_id)
        _signature(body, other)
    elif maximum < 0:
        raise Error("while_loop max_iterations must be nonnegative")
    if has_condition:
        _signature(body, condition)
        if len(condition.outs) != 1:
            raise Error("control-flow condition must return one scalar")
    var every = 1
    if is_loop:
        every = loop_check_every(check_every, "while_loop")
        if maximum > LOOP_MAX_EXACT:
            raise Error("ValueError: while_loop max_iterations must be at most " + String(LOOP_MAX_EXACT)
                        + ", got " + String(maximum))
        var n_state = body.n_in - (1 if iteration_input else 0)
        if iteration_input:
            # The register is one value: a float (exact below 2**24) or, for
            # an ids input (lane x1), an int32 word `take`/`index` can read.
            if body.n_in < 1 or body.in_shapes[n_state].count() != 1:
                raise Error("ValueError: while_loop with iteration=True needs a last body input of one value")
        if len(body.outs) != n_state:
            raise Error("while_loop body must return one tensor per input")
        if history and n_state < 1:
            raise Error("ValueError: a token record needs at least one state input")
        # lane x1: int32-word state is carried as it is; each output must
        # keep its input's kind (an int32 argmax cannot become float state,
        # nor a float value ids state). The rest is checked at each commit.
        for k in range(len(body.outs)):  # driver-loop: over int32-word argmax outputs and their state kinds
            var value_ref = body.outs[k]
            if value_ref >= body.n_in:
                var node = value_ref - body.n_in
                var int_argmax = body.code[node] == 72 and body.ints[body.i_at[node] + 2] == 0
                if int_argmax and body.in_kinds[k] != PROGRAM_IN_IDS:
                    raise Error("ValueError: while_loop state output " + String(k)
                                + " holds integer ids; its state input must be ids")
    if optimize:
        var body_plan = program_optimize(body)
        var condition_plan = program_optimize(condition)
        var other_plan = program_optimize(other)
        body = body_plan.program.copy()
        condition = condition_plan.program.copy()
        other = other_plan.program.copy()
    return ControlPlan(body^, condition^, other^, is_loop, maximum, every, iteration_input, has_condition, history)


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
    # Owners are local to this admitted control operation. Preparation remains
    # lazy until a component executes; only its native node metadata is reused.
    var body_replay = ControlReplay()
    var condition_replay = ControlReplay()
    var other_replay = ControlReplay()
    var result = ControlResult()
    var iterations = 0
    var stopped = False
    # The loop's register block and staging slots (freed with the state).
    var extras = List[FP]()
    var scratch = List[FP]()
    var scratch_n = List[Int]()
    try:
        s[].tape.no_grad()
        if is_loop:
            # Lane fix-d3-executor-loops (2026-11-08): the executor's bounded
            # loop (autograd/program_loop.mojo). The carried state stays in
            # the uploaded buffers and is updated in place (OP_LOOP_COMMIT);
            # the condition goes to the register block on the device
            # (OP_LOOP_TICK) and the iteration register counts committed
            # iterations (OP_LOOP_ADVANCE). The host reads the block once
            # every `check_every` iterations, after a tick and before its
            # body, so with check_every = 1 no body runs after the stop and
            # the loop drains once per iteration (the former loop drained
            # twice: the condition read and the state copy). Iterations
            # enqueued past the stop leave the state as the stopping
            # iteration left it (closed gate), so the outputs, `iterations`
            # and `stopped` do not depend on `check_every`.
            var every = plan.check_every
            var n_state = len(buffers)
            var words = s[].ex.alloc(LOOP_WORDS)
            extras.append(words)
            for _k in range(n_state):  # driver-loop: over loop-carried state tensors (staging slots)
                scratch.append(words)
                scratch_n.append(0)
            var mirror = List[Float32](length=LOOP_WORDS, fill=Float32(0))
            var mirror_p = FP(unsafe_from_address=Int(mirror.unsafe_ptr()))
            var register = FP(unsafe_from_address=Int(words) + 4 * LOOP_ITER)
            # lane x1: an ids iteration input reads an int32 copy of the
            # register (OP_LOOP_ITER_WORD before each iteration's condition).
            var iter_words = plan.iteration_input and body.in_kinds[n_state] == PROGRAM_IN_IDS
            if iter_words:
                register = s[].ex.alloc(1)
                extras.append(register)
            # lane x1: a loop with no condition ticks on a constant 1.
            var one = words
            if not plan.has_condition:
                one = s[].ex.alloc(1)
                extras.append(one)
                var one_host = List[Float32](length=1, fill=Float32(1.0))
                s[].ex.upload(one, FP(unsafe_from_address=Int(one_host.unsafe_ptr())), 1)
                _ = len(one_host)  # keep-alive past the staging copy
            # lane x1: the record of state 0 after every committed iteration.
            var record = words
            var per_token = shapes[0].count() if n_state > 0 else 0
            if plan.history and maximum > 0:
                record = s[].ex.alloc(maximum * per_token)
                extras.append(record)
            var timers = LoopTimers("while_loop")
            var done = False
            while not done:  # driver-loop: over flag-read chunks of the bounded loop, bounded by max_iterations
                var j = 0
                while j < every:  # driver-loop: over loop iterations enqueued between two flag reads
                    var in_buffers = buffers.copy()
                    var in_shapes = shapes.copy()
                    if plan.iteration_input:
                        if iter_words:
                            var iw = Args()
                            iw.p0 = words
                            iw.p1 = register
                            s[].ex.launch[OP_LOOP_ITER_WORD](iw, 1)
                        in_buffers.append(register)
                        in_shapes.append(body.in_shapes[n_state])
                    var tick = Args()
                    tick.p0 = words
                    tick.p1 = one
                    tick.i0 = maximum
                    if plan.has_condition:
                        var cout = _execute(s[], condition, condition_replay, in_buffers, in_shapes, temps)
                        if len(cout) != 1 or s[].shapes[s[]._slot(cout[0])].count() != 1:
                            raise Error("control-flow condition must return one scalar")
                        tick.p1 = s[].tape.value_of(s[].ex, s[].tensors[s[]._slot(cout[0])])
                    s[].ex.launch[OP_LOOP_TICK](tick, 1)
                    _lap(s[], timers, "condition")
                    j += 1
                    if j == every:
                        s[].ex.download(mirror_p, words, LOOP_WORDS)
                        iterations = Int(mirror[LOOP_ITER])
                        _lap(s[], timers, "read_flag")
                        if mirror[LOOP_DONE] != Float32(0.0):
                            done = True
                            break
                    var outputs = _execute(s[], body, body_replay, in_buffers, in_shapes, temps)
                    _lap(s[], timers, "body")
                    # The gate is known open right after a read; between
                    # reads it is the device's, so shapes must hold.
                    _commit_state(s[], body, outputs, words, buffers, shapes, scratch, scratch_n, j != every)
                    if plan.history and maximum > 0:
                        if shapes[0].count() != per_token:
                            raise Error("ValueError: a token record needs a body that keeps state 0's shape")
                        var rec = Args()
                        rec.p0 = buffers[0]
                        rec.p1 = record
                        rec.p2 = words
                        rec.i0 = per_token
                        s[].ex.launch[OP_LOOP_APPEND](rec, per_token)
                    var advance = Args()
                    advance.p0 = words
                    s[].ex.launch[OP_LOOP_ADVANCE](advance, 1)
                    _lap(s[], timers, "commit")
            if mirror[LOOP_BAD] != Float32(0.0):
                raise Error("control-flow condition must be finite")
            stopped = mirror[LOOP_STOPPED] != Float32(0.0)
            timers.report()
            for k in range(len(buffers)):  # driver-loop: over final state download
                var a = HostArray(shapes[k])
                a.int_words = body.in_kinds[k] == PROGRAM_IN_IDS
                s[].ex.download(FP(unsafe_from_address=Int(a.data.unsafe_ptr())), buffers[k], shapes[k].count())
                result.outputs.append(a^)
            if plan.history:
                var rows = iterations if maximum > 0 else 0
                var h = HostArray(ShapeN.matrix(rows, per_token))
                h.int_words = body.in_kinds[0] == PROGRAM_IN_IDS
                if rows > 0:
                    s[].ex.download(FP(unsafe_from_address=Int(h.data.unsafe_ptr())), record, rows * per_token)
                result.history.append(h^)
        else:
            var take_body = _predicate(s[], condition, condition_replay, buffers, shapes, temps)
            var outputs = _execute(s[], body, body_replay, buffers, shapes, temps) if take_body else _execute(
                s[], other, other_replay, buffers, shapes, temps)
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
        _free_loop_owners(s[], extras, scratch, scratch_n)
        var message = String(err)
        var context = "program control flow after " + String(iterations) + " iterations: "
        if message.startswith("ValueError: "):
            raise Error("ValueError: " + context + message[byte=12:])
        raise Error(context + message)
    s[]._drop_temps(temps)
    swap(held, s[].tape)
    swap(held, s[].prog_tape)
    for k in range(len(buffers)):  # driver-loop: over current state owners
        s[].ex.free(buffers[k])
    _free_loop_owners(s[], extras, scratch, scratch_n)
    result.iterations = iterations
    result.stopped = stopped
    return result^
