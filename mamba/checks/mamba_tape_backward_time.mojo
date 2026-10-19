# SPDX-License-Identifier: Apache-2.0
"""Times the hand-chained Mamba-1 / Mamba-2 prefill backward against the tape
assembly (autograd/rules_mamba.mojo), one build, one run per arm.

For the largest fixture case of each family (Mamba-1 comp_b2_l257_d8,
Mamba-2 m2_statepass_b2_l770_d64), runs one warm-up call and then N timed
calls of the hand backward and N of the tape backward, and N of a 4-block
tape stack (no hand path exists for a stack). Prints

    TIME m1.hand median_ms=<x> n=<N>
    TIME m1.tape median_ms=<x> n=<N>
    TIME m1.stack4.tape median_ms=<x> n=<N>
    (the same for m2)

N comes from MOJOLEARN_MAMBA_TAPE_TIME (default 10). Both arms take host
lists and return host lists, so each timed call includes the same uploads
and downloads in both arms; the device is synchronized before each clock
reading. With MOJOLEARN_AG_PROFILE=1 each arm also prints
`LAUNCHES <arm> <n>` for its last timed call: the launches that go through
DeviceExec. The hand path enqueues on the context directly, so it prints
n/a; the tape's custom rules launch the hand kernels on the context too, so
its count covers the tape's own DeviceExec launches only.

The gradients are not compared here (mamba_tape_backward_check.mojo does
that); the first word of each arm's x gradient is printed so a run that
returned nothing shows.

    MOJOLEARN_MAMBA_TAPE_TIME=20 mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . \
        mamba/checks/mamba_tape_backward_time.mojo
"""
from std.builtin.sort import sort
from std.os import getenv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from mamba.checks.mamba_fixture import MambaWeights, corpus_case, corpus_case_weights, corpus_case_x
from mamba.checks.mamba2_fixture import Mamba2Weights, m2_case_weights, m2_case_x, m2_corpus_case
from mamba.impl.modeling.modeling_mamba_prefill_backward import mamba1_prefill_backward
from mamba.impl.modules.mamba2_prefill_backward import mamba2_prefill_backward
from autograd.rules_mamba import (
    mamba1_tape_backward,
    mamba1_tape_backward_stack,
    mamba2_tape_backward,
    mamba2_tape_backward_stack,
)
from sequence.launch_profile import launch_profile_level, launch_profile_reset, launch_profile_snapshot

#: the largest case of each fixture table (by B * L * d_model)
comptime M1_CASE = 13
comptime M2_CASE = 9
comptime STACK_BLOCKS = 4


def timed_calls() raises -> Int:
    var s = String(getenv("MOJOLEARN_MAMBA_TAPE_TIME"))
    if s == "":
        return 10
    var n = Int(atol(s))
    if n < 1:
        raise Error("MOJOLEARN_MAMBA_TAPE_TIME must be >= 1, got " + s)
    return n


def cotangent(n: Int) -> List[Float32]:
    var c = List[Float32]()
    for i in range(n):
        c.append(Float32((i * 37 + 11) % 31 - 15) * Float32(0.0625))
    return c^


def report(name: String, mut ns: List[Int]):
    """Prints the median of the timed calls (the warm-up is not in `ns`)."""
    sort(ns)
    var k = len(ns)
    var med = Float64(ns[k // 2])
    if k % 2 == 0:
        med = (Float64(ns[k // 2 - 1]) + Float64(ns[k // 2])) / 2.0
    print("TIME " + name + " median_ms=" + String(med / 1000000.0) + " n=" + String(k))


def profile_start(prof: Bool, last: Bool) raises:
    if prof and last:
        launch_profile_reset()


def profile_report(prof: Bool, last: Bool, arm: String, counted: Bool) raises:
    if not (prof and last):
        return
    if not counted:
        print("LAUNCHES " + arm + " n/a")
        return
    var p = launch_profile_snapshot()
    var total = 0
    for i in range(len(p.ops)):  # driver-loop: over profiled operation codes
        total += p.op_launches[i]
    # The Mamba tape issues no DeviceExec launch of its own (its rule copies
    # are skipped when source and destination are the same buffer, which
    # they are since the pass that matched the hand path's launches); every
    # kernel it runs is a hand kernel enqueued on the context, which the
    # profile does not count. 0 here is that, not a missed profile.
    print("LAUNCHES " + arm + " " + String(total) + " (DeviceExec only; context-enqueued hand kernels not counted)")


def main() raises:
    var n = timed_calls()
    var prof = launch_profile_level() > 0
    var ctx = DeviceContext()

    # ---------------------------------------------------------------- Mamba-1
    var c = corpus_case(M1_CASE)
    var w = corpus_case_weights(M1_CASE)
    var x = corpus_case_x(M1_CASE)
    var dy = cotangent(len(x))
    var stack = List[MambaWeights]()
    for _ in range(STACK_BLOCKS):  # driver-loop: over stack blocks
        stack.append(w.copy())
    print("CASE m1", c.name, "b", c.b, "l", c.l, "d_model", c.d_model, "blocks", STACK_BLOCKS)
    var hand = List[Int]()
    var tape = List[Int]()
    var stk = List[Int]()
    for k in range(n + 1):  # driver-loop: over timed calls, call 0 the warm-up
        var last = k == n
        ctx.synchronize()
        profile_start(prof, last)
        var t0 = Int(perf_counter_ns())
        var h = mamba1_prefill_backward(w, x, dy, c.b, c.l, ctx.copy())
        ctx.synchronize()
        var t1 = Int(perf_counter_ns())
        profile_report(prof, last, "m1.hand", False)
        profile_start(prof, last)
        var t = mamba1_tape_backward(w, x, dy, c.b, c.l, ctx.copy())
        ctx.synchronize()
        var t2 = Int(perf_counter_ns())
        profile_report(prof, last, "m1.tape", True)
        profile_start(prof, last)
        var s = mamba1_tape_backward_stack(stack, x, dy, c.b, c.l, ctx.copy())
        ctx.synchronize()
        var t3 = Int(perf_counter_ns())
        profile_report(prof, last, "m1.stack4.tape", True)
        if k > 0:
            hand.append(t1 - t0)
            tape.append(t2 - t1)
            stk.append(t3 - t2)
        if last:
            print("FIRST m1 hand.x", h.x[0], "tape.x", t.x[0], "stack4.x", s[0].x[0])
    report("m1.hand", hand)
    report("m1.tape", tape)
    report("m1.stack4.tape", stk)
    print("TIME m1.stack4.hand n/a (no hand stack path)")

    # ---------------------------------------------------------------- Mamba-2
    var f = m2_corpus_case(M2_CASE)
    var w2 = m2_case_weights(M2_CASE)
    var x2 = m2_case_x(M2_CASE)
    var dy2 = cotangent(len(x2))
    var stack2 = List[Mamba2Weights]()
    for _ in range(STACK_BLOCKS):  # driver-loop: over stack blocks
        stack2.append(w2.copy())
    print("CASE m2", f.name, "b", f.b, "l", f.l, "d_model", f.d_model, "blocks", STACK_BLOCKS)
    var hand2 = List[Int]()
    var tape2 = List[Int]()
    var stk2 = List[Int]()
    for k in range(n + 1):  # driver-loop: over timed calls, call 0 the warm-up
        var last = k == n
        ctx.synchronize()
        profile_start(prof, last)
        var t0 = Int(perf_counter_ns())
        var h = mamba2_prefill_backward(w2, x2, dy2, f.b, f.l, f.dt_lo, f.dt_hi, ctx.copy())
        ctx.synchronize()
        var t1 = Int(perf_counter_ns())
        profile_report(prof, last, "m2.hand", False)
        profile_start(prof, last)
        var t = mamba2_tape_backward(w2, x2, dy2, f.b, f.l, f.dt_lo, f.dt_hi, ctx.copy())
        ctx.synchronize()
        var t2 = Int(perf_counter_ns())
        profile_report(prof, last, "m2.tape", True)
        profile_start(prof, last)
        var s = mamba2_tape_backward_stack(stack2, x2, dy2, f.b, f.l, f.dt_lo, f.dt_hi, ctx.copy())
        ctx.synchronize()
        var t3 = Int(perf_counter_ns())
        profile_report(prof, last, "m2.stack4.tape", True)
        if k > 0:
            hand2.append(t1 - t0)
            tape2.append(t2 - t1)
            stk2.append(t3 - t2)
        if last:
            print("FIRST m2 hand.x", h.x[0], "tape.x", t.x[0], "stack4.x", s[0].x[0])
    report("m2.hand", hand2)
    report("m2.tape", tape2)
    report("m2.stack4.tape", stk2)
    print("TIME m2.stack4.hand n/a (no hand stack path)")
    print("MAMBA_TAPE_TIME_DONE n=" + String(n))
