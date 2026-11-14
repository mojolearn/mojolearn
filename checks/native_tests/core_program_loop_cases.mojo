# SPDX-License-Identifier: Apache-2.0
"""Program.while_loop as the executor's bounded loop, moved from
test_program_loop.py (2026-11-14), through the typed entries the public door
calls (autograd/program_control.mojo control_prepare / control_run, with the
door's words: no other branch, optimize off, check_every None as 0 = the
loop's chunk rule).

  0..5  check_every in {None, 1, 2, 3, 8, 64}: x = [[0]], condition
        x < 5, body x + 1. For (limit, iterations, stopped) in (10, 5, yes),
        (3, 3, no), (0, 0, no), (5, 5, yes) the default loop gives that count
        and stop flag, and the loop with check_every gives the same count,
        flag and output words.
  6, 7  check_every 1 and 4: a body that swaps its two state tensors commits
        both from one iteration (3 iterations, not stopped, swapped).
  8, 9  check_every 1 and 5: the iteration register is a program value
        (x accumulates 0 + 1 + 2 + 3 = 6, stopped at i == 4 after 4).
The resizing body and the refusals stay in the Python test.
"""
from autograd.ops_api import OpsSession, ops_session
from autograd.program_control import control_prepare, control_run, ControlResult
from autograd.autograd_program_control_run import control_state_view
from autograd.program_io import ProgramInput
from autograd.rankn import ShapeN
from core.cast_in import CastIn
from sequence.exec_trait import Exec
from checks.native_tests.ops_higher_cases import require, tensor, op2, total, same_words
from checks.native_tests.core_ops_common import dims2, opi

comptime CORE_PROGRAM_LOOP_FIXTURES = 10

comptime _ADD = 2
comptime _FULL = 30
comptime _COMPARE = 69
comptime _LT = 2


def _full[E: Exec & Movable & Deinitable](mut s: OpsSession[E], value: Float32) raises -> Int:
    var fs: List[Float32] = [value]
    return opi(s, _FULL, List[Int](), dims2(1, 1), fs)


def _less[E: Exec & Movable & Deinitable](mut s: OpsSession[E], a: Int, b: Int) raises -> Int:
    var hs: List[Int] = [a, b]
    var kind: List[Int] = [_LT]
    return opi(s, _COMPARE, hs, kind, List[Float32]())


def _loop[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, body: Int, condition: Int, maximum: Int, every: Int, iteration: Bool,
    arrays: List[List[Float32]], cols: List[Int],
) raises -> ControlResult:
    var plan = control_prepare[E, KIND](sid, body, condition, 0, sid, sid, maximum, False, every, iteration)
    var inputs = List[ProgramInput]()
    for k in range(len(arrays)):
        var rows = len(arrays[k]) // cols[k]
        inputs.append(ProgramInput(CastIn.f32_c(Int(arrays[k].unsafe_ptr()), rows, cols[k]),
                                   ShapeN.of(dims2(rows, cols[k])), List[Float64](), List[Int64]()))
    var r = control_run[E, KIND](sid, plan, inputs^)
    _ = arrays
    return r^


def run_core_program_loop_fixture[E: Exec & Movable & Deinitable, KIND: Int](sid: Int, fixture_id: Int) raises:
    var s = ops_session[E, KIND](sid)
    if fixture_id < 6:
        var everies: List[Int] = [0, 1, 2, 3, 8, 64]
        var every = everies[fixture_id]
        var a: List[Float32] = [0.0]
        var x = tensor(s[], a, dims2(1, 1))
        var cond_out: List[Int] = [_less(s[], x, _full(s[], 5.0))]
        var ins: List[Int] = [x]
        var condition = s[].capture(ins, cond_out)
        var body_out: List[Int] = [op2(s[], _ADD, x, _full(s[], 1.0))]
        var body = s[].capture(ins, body_out)
        var limits: List[Int] = [10, 3, 0, 5]
        var counts: List[Int] = [5, 3, 0, 5]
        var stops: List[Bool] = [True, False, False, True]
        var arrays: List[List[Float32]] = [a.copy()]
        var cols: List[Int] = [1]
        for c in range(4):
            var ref_ = _loop[E, KIND](sid, body, condition, limits[c], 0, False, arrays, cols)
            var got = _loop[E, KIND](sid, body, condition, limits[c], every, False, arrays, cols)
            var tag = "check_every " + String(every) + " limit " + String(limits[c])
            require(ref_.iterations == counts[c] and ref_.stopped == stops[c], tag + ": the default loop's count or stop flag")
            require(got.iterations == ref_.iterations and got.stopped == ref_.stopped, tag + ": count or stop flag differs")
            same_words(got.outputs[0].data, ref_.outputs[0].data, tag + ": outputs differ")
        return
    if fixture_id < 8:
        var every = 1 if fixture_id == 6 else 4
        var a: List[Float32] = [1.0, 2.0]
        var b: List[Float32] = [10.0, 20.0]
        var x = tensor(s[], a, dims2(1, 2))
        var y = tensor(s[], b, dims2(1, 2))
        var ins: List[Int] = [x, y]
        var cond_out: List[Int] = [_less(s[], total(s[], x), _full(s[], 1.0e9))]
        var condition = s[].capture(ins, cond_out)
        var swapped: List[Int] = [y, x]
        var body = s[].capture(ins, swapped)
        var arrays: List[List[Float32]] = [a.copy(), b.copy()]
        var cols: List[Int] = [2, 2]
        var r = _loop[E, KIND](sid, body, condition, 3, every, False, arrays, cols)
        require(r.iterations == 3 and not r.stopped, "swap: 3 iterations, not stopped")
        same_words(r.outputs[0].data, b, "swap: the first state is the second input")
        same_words(r.outputs[1].data, a, "swap: the second state is the first input")
        return
    require(fixture_id < CORE_PROGRAM_LOOP_FIXTURES, "unknown program loop fixture")
    var every = 1 if fixture_id == 8 else 5
    var a: List[Float32] = [0.0]
    var x = tensor(s[], a, dims2(1, 1))
    var i = tensor(s[], a, dims2(1, 1))
    var ins: List[Int] = [x, i]
    var cond_out: List[Int] = [_less(s[], i, _full(s[], 4.0))]
    var condition = s[].capture(ins, cond_out)
    var body_out: List[Int] = [op2(s[], _ADD, x, i)]
    var body = s[].capture(ins, body_out)
    var arrays: List[List[Float32]] = [a.copy()]
    var cols: List[Int] = [1]
    var r = _loop[E, KIND](sid, body, condition, 100, every, True, arrays, cols)
    require(r.iterations == 4 and r.stopped, "iteration register: 4 iterations, stopped")
    var six: List[Float32] = [6.0]
    same_words(r.outputs[0].data, six, "iteration register: x = 0 + 1 + 2 + 3")
