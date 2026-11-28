# SPDX-License-Identifier: Apache-2.0
# The executor's bounded loop: the vocabulary shared by the program loop
# (`autograd/program_control.mojo`, over any `Exec`) and the device loop of
# compiled step ops (`autograd/program_loop_device.mojo`, over one
# DeviceContext). Lane fix-d3-executor-loops (2026-11-08).
#
# A bounded loop here has four parts, and none of them is a family's:
#   1. carried state that lives in fixed buffers and is updated in place by
#      the loop's own ops (no per-iteration allocation, copy into a fresh
#      buffer, or host sync);
#   2. an iteration register on the device, readable by the body as a value;
#   3. the stopping criterion as a device value: the device compares, sets a
#      done word, and freezes the state; the host reads the register block
#      once every `check_every` iterations (and at the bound);
#   4. step ops: a body is a list of ops enqueued on the device stream in a
#      fixed order, gated on the done word, so iterations enqueued past the
#      stop leave every buffer as the stopping iteration left it.
#
# The register block (`LOOP_WORDS` words). The device loop of step ops keeps
# it in Int32 words (the gate its step kernels read); the program loop keeps
# it in float32 program words (program values are float32; every count used
# here is below 2**24, so it is exact).
#   [LOOP_DONE]       nonzero once the loop has stopped (criterion met,
#                     condition false, or the bound reached)
#   [LOOP_STOP_ITER]  the 1-based iteration at which the criterion held
#   [LOOP_ITER]       the iteration register: iterations committed so far
#   [LOOP_STOPPED]    program loop: the condition read false
#   [LOOP_BAD]        program loop: the condition was not finite
#   [LOOP_GATE]       program loop: nonzero when this iteration's state is
#                     committed (set by the tick, read by the commits)
#   [LOOP_FIT_BEST]   Program.fit: the best epoch loss so far (lane x1b; word 7,
#                     after core/step_gate.mojo STEP_FAIL = word 6)
#   [LOOP_FIT_STALL]  Program.fit: epochs since the loss last improved
#
# Lane x1b: every write an iteration makes is behind the gate, including
# the declared state a body writes in place (a block step op's state, kv and
# words): the loop shadows it (OP_LOOP_SHADOW), so a body needs no gate of
# its own and no body op changes. A step op of the device loop below takes
# the gate itself (core/step_gate.mojo).
#
# Per-op timers: `-D MOJOLEARN_EXECUTOR_TIMERS` (default off). Each loop
# drains the stream around every op it times (so the timing is relative and
# the loop is slower), accumulates host nanoseconds per op name and prints
# one line per op when the loop ends:
#   EXECUTOR_TIMER loop=<label> op=<name> calls=<n> total_ms=<ms> mean_us=<us>
# Off, every timer call is compiled out. The per-op-code launch profile of a
# program body is the existing `MOJOLEARN_AG_PROFILE` (sequence/launch_profile.mojo).
from std.memory import bitcast
from std.sys.compile import is_defined
from std.time import perf_counter_ns
from sequence.ops import Args

# The register block words, the chunk rule and LoopTimers live in
# core/bounded_loop.mojo (moved byte for byte by lane thin-edges-1,
# 2026-11-28) and are exported from here under the same names.
from core.bounded_loop import (
    LOOP_DONE, LOOP_STOP_ITER, LOOP_ITER, LOOP_STOPPED, LOOP_BAD, LOOP_GATE, LOOP_WORDS,
    LOOP_FIT_BEST, LOOP_FIT_STALL, LOOP_MAX_EXACT, EXECUTOR_TIMERS,
    LOOP_CHECK_AUTO, LOOP_AUTO_CHUNK, LOOP_DRAIN_FLOATS, FIT_STEPS_PER_READ,
    loop_check_every, loop_auto_chunk, fit_epochs_per_read, LoopTimers,
)


# ---------------------------------------------------------------- program loop ops
# The program loop's three element ops over float32 program words, codes
# OP_LOOP_FIRST..OP_LOOP_LAST (1800..1809); sequence/dispatch.mojo routes
# and fences them. They run on every executor (one GPU thread per element on
# the device, an ascending loop on the host) with the same body, so the
# loop's control words agree on every column.
comptime OP_LOOP_TICK = 1800
comptime OP_LOOP_COMMIT = 1801
comptime OP_LOOP_ADVANCE = 1802
# lane x1-executor-state-and-step-ops: the decode loop's token record and the
# iteration register as an int32 word.
comptime OP_LOOP_APPEND = 1803
comptime OP_LOOP_ITER_WORD = 1804
# lane x1b-executor-gated-state-chunked-loop: the gated shadow of state a
# body writes in place, and Program.fit's epoch gate and stopping criterion.
comptime OP_LOOP_SHADOW = 1805
comptime OP_LOOP_OPEN = 1806
comptime OP_LOOP_FIT_TICK = 1807
comptime OP_LOOP_FIRST = OP_LOOP_TICK
comptime OP_LOOP_LAST = 1809


def op_loop_tick(t: Int, a: Args):
    """One element. p0: the register block (float32 words); p1: the
    condition's one value; i0: the iteration bound. Closes the gate, then,
    while the loop runs: a non-finite condition stops it with LOOP_BAD, a
    zero condition with LOOP_STOPPED, a reached bound stops it; otherwise the
    gate opens and this iteration's body results are committed. The order of
    the tests is the host loop's: the condition first, then the bound."""
    var w = a.p0
    w.unsafe_store(LOOP_GATE, Float32(0.0))
    if w.unsafe_load(LOOP_DONE) != Float32(0.0):
        return
    var c = a.p1.unsafe_load(0)
    if (bitcast[DType.uint32](c) & UInt32(0x7FFFFFFF)) >= UInt32(0x7F800000):
        w.unsafe_store(LOOP_BAD, Float32(1.0))
        w.unsafe_store(LOOP_DONE, Float32(1.0))
        return
    if c == Float32(0.0):
        w.unsafe_store(LOOP_STOPPED, Float32(1.0))
        w.unsafe_store(LOOP_DONE, Float32(1.0))
        return
    if w.unsafe_load(LOOP_ITER) >= Float32(a.i0):
        w.unsafe_store(LOOP_DONE, Float32(1.0))
        return
    w.unsafe_store(LOOP_GATE, Float32(1.0))


def op_loop_commit(t: Int, a: Args):
    """p1[t] = p0[t] when the gate of register block p2 is open: one carried
    state tensor updated in place with this iteration's value (a bit copy:
    a float32 load and store moves no arithmetic, so int32-word state such as
    token ids is carried word for word)."""
    if a.p2.unsafe_load(LOOP_GATE) != Float32(0.0):
        a.p1.unsafe_store(t, a.p0.unsafe_load(t))


def op_loop_advance(t: Int, a: Args):
    """One element: the iteration register of block p0 counts a committed
    iteration (exact below 2**24, the bound the loop admits)."""
    var w = a.p0
    if w.unsafe_load(LOOP_GATE) != Float32(0.0):
        w.unsafe_store(LOOP_ITER, w.unsafe_load(LOOP_ITER) + Float32(1.0))


def op_loop_append(t: Int, a: Args):
    """p1[r * i0 + t] = p0[t] when the gate of register block p2 is open,
    r = the iteration register (iterations committed before this one): the
    committed state 0 (a decode loop's token ids, i0 words a token) appended
    to the record of every iteration, a bit copy. Launched before the
    iteration register advances; r < the bound the record was sized for."""
    var w = a.p2
    if w.unsafe_load(LOOP_GATE) != Float32(0.0):
        var r = Int(w.unsafe_load(LOOP_ITER))
        a.p1.unsafe_store(r * a.i0 + t, a.p0.unsafe_load(t))


def op_loop_iter_word(t: Int, a: Args):
    """One element: p1[0] = the iteration register of block p0 as an int32
    word (the body's iteration input when it is an ids input: `take` and
    `index` read it as a position)."""
    var r = Int32(Int(a.p0.unsafe_load(LOOP_ITER)))
    a.p1.unsafe_store(0, bitcast[DType.float32](r))


def op_loop_shadow(t: Int, a: Args):
    """The gated shadow of one declared state tensor a body writes in place
    (a block step op's state, kv or words). p0: the live tensor, p1: its
    shadow, p2: the register block. Launched after every body: when the
    iteration commits (gate open) the shadow takes the new value; when it
    does not (an iteration enqueued past the stop) the live tensor gets the
    committed value back. A bit copy either way (int32 words move word for
    word), so the state after the loop is the stopping iteration's for every
    chunk size."""
    if a.p2.unsafe_load(LOOP_GATE) != Float32(0.0):
        a.p1.unsafe_store(t, a.p0.unsafe_load(t))
    else:
        a.p0.unsafe_store(t, a.p1.unsafe_load(t))


def op_loop_open(t: Int, a: Args):
    """One element: Program.fit's epoch gate. p0: the register block; the
    gate opens while the loop runs (the done word is 0) and closes after the
    stop, so the epoch's commits, shadows and criterion do nothing."""
    var w = a.p0
    if w.unsafe_load(LOOP_DONE) == Float32(0.0):
        w.unsafe_store(LOOP_GATE, Float32(1.0))
    else:
        w.unsafe_store(LOOP_GATE, Float32(0.0))


def op_loop_fit_tick(t: Int, a: Args):
    """One element, after an epoch: p0 the register block, p1 the epoch's
    mean loss, f0 `tol`, i0 `n_iter_no_change` (0: no criterion). While the
    gate is open the iteration register counts the committed epoch; with a
    criterion, scikit-learn's SGD rule: an epoch whose loss is above the best
    so far minus `tol` adds one to the stall count (otherwise it is reset),
    the best is kept, and `n_iter_no_change` stalled epochs in a row stop the
    loop after this epoch (LOOP_STOPPED, LOOP_STOP_ITER, the done word). The
    first epoch compares against +inf. Float32 compares only: the same bit on
    every vendor and on the host."""
    var w = a.p0
    if w.unsafe_load(LOOP_GATE) == Float32(0.0):
        return
    var it = w.unsafe_load(LOOP_ITER) + Float32(1.0)
    w.unsafe_store(LOOP_ITER, it)
    if a.i0 <= 0:
        return
    var loss = a.p1.unsafe_load(0)
    var best = w.unsafe_load(LOOP_FIT_BEST)
    if it == Float32(1.0):
        best = bitcast[DType.float32](UInt32(0x7F800000))
    var stall = w.unsafe_load(LOOP_FIT_STALL)
    if loss > best - a.f0:
        stall = stall + Float32(1.0)
    else:
        stall = Float32(0.0)
    if loss < best:
        best = loss
    w.unsafe_store(LOOP_FIT_BEST, best)
    w.unsafe_store(LOOP_FIT_STALL, stall)
    if stall >= Float32(a.i0):
        w.unsafe_store(LOOP_STOPPED, Float32(1.0))
        w.unsafe_store(LOOP_STOP_ITER, it)
        w.unsafe_store(LOOP_DONE, Float32(1.0))


def apply_loop[OP: Int](t: Int, a: Args):
    comptime if OP == OP_LOOP_TICK:
        op_loop_tick(t, a)
    elif OP == OP_LOOP_COMMIT:
        op_loop_commit(t, a)
    elif OP == OP_LOOP_ADVANCE:
        op_loop_advance(t, a)
    elif OP == OP_LOOP_APPEND:
        op_loop_append(t, a)
    elif OP == OP_LOOP_ITER_WORD:
        op_loop_iter_word(t, a)
    elif OP == OP_LOOP_SHADOW:
        op_loop_shadow(t, a)
    elif OP == OP_LOOP_OPEN:
        op_loop_open(t, a)
    elif OP == OP_LOOP_FIT_TICK:
        op_loop_fit_tick(t, a)
