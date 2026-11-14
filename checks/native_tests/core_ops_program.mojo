# SPDX-License-Identifier: Apache-2.0
"""Captured programs equal the eager session word for word, moved from
test_ops_program.py (2026-11-14): the mixture classifier's predict path and
the Shapes section's attention product.

For each fixture (the former `_check`): on one inference session build the
composition on inputs A, read the eager outputs, capture inputs -> outputs,
check the program's input and output shapes, reset the session, run the
program on A twice and on new inputs B. The first run equals the eager
outputs word for word, the second equals the first, and the run on B equals
an eager build on B in a second session word for word.

  0 mixture predict: x (128, 4); softmax(gate-weighted sum of two linear
    experts) and the softmax gate, with fixed 0.1-scaled parameters.
  1 attention: q, kv (2, 4, 6, 4); scale(q @ kv^T, 0.5), softmax, @ kv,
    permute (0, 2, 1, 3), reshape (2, 6, 16); outputs the merged tensor, its
    two halves along axis 2 concatenated in swapped order, and its max over
    axis 1.

The NumPy draws are core_ops_common's seeded values of the same shapes. The
quickstart, conv, ids-input, saved-program, guide-block, direct-route and
refusal arms stay in the Python test.
"""
from autograd.ops_api import OpsSession
from autograd.rankn import ShapeN
from core.cast_in import CastIn
from sequence.exec_trait import Exec
from sequence.ops import FP
from checks.native_tests.ops_higher import tensor, op1, op2, scale, values, same_words
from checks.native_tests.core_ops_common import normal, dims2, dims3, dims4, opi
from core.host_parallel import host_parallelize
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


comptime CORE_OPS_PROGRAM_FIXTURES = 2

comptime _MATMUL = 1
comptime _ADD = 2
comptime _MUL = 4
comptime _TRANSPOSE = 13
comptime _SOFTMAX = 17
comptime _LINEAR = 19
comptime _RESHAPE_N = 33
comptime _PERMUTE = 34
comptime _MAX = 37
comptime _CONCAT = 39
comptime _NARROW = 40
comptime _ACT_NONE = 0
comptime _ACT_SOFTMAX = 4


def _count(dims: List[Int]) -> Int:
    var n = 1
    for d in dims:
        n *= d
    return n


def _scaled(values_in: List[Float32], factor: Float32) -> List[Float32]:
    var out = List[Float32](length=len(values_in), fill=0)
    for i in range(len(values_in)):
        out[i] = factor * values_in[i]
    return out^


@fieldwise_init
struct Built(Movable):
    var ins: List[Int]
    var outs: List[Int]
    var out_dims: List[List[Int]]


def _linear[E: Exec & Movable & Deinitable](mut s: OpsSession[E], x: Int, w: Int, b: Int, act: Int) raises -> Int:
    var hs: List[Int] = [x, w, b]
    var ints: List[Int] = [act]
    return opi(s, _LINEAR, hs, ints, List[Float32]())


def _mixture[E: Exec & Movable & Deinitable](mut s: OpsSession[E], arrays: List[List[Float32]]) raises -> Built:
    var x = tensor(s, arrays[0], dims2(128, 4))
    var w0 = tensor(s, _scaled(normal(8, 30), 0.1), dims2(4, 2))
    var b0 = tensor(s, _scaled(normal(2, 31), 0.1), dims2(1, 2))
    var w1 = tensor(s, _scaled(normal(8, 32), 0.1), dims2(4, 2))
    var b1 = tensor(s, _scaled(normal(2, 33), 0.1), dims2(1, 2))
    var wg = tensor(s, _scaled(normal(8, 34), 0.1), dims2(4, 2))
    var bg = tensor(s, _scaled(normal(2, 35), 0.1), dims2(1, 2))
    var e0: List[Float32] = [1.0, 0.0]
    var e1: List[Float32] = [0.0, 1.0]
    var z0 = _linear(s, x, w0, b0, _ACT_NONE)
    var z1 = _linear(s, x, w1, b1, _ACT_NONE)
    var gate = _linear(s, x, wg, bg, _ACT_SOFTMAX)
    var logits = op2(s, _ADD,
                     op2(s, _MUL, op2(s, _MATMUL, gate, tensor(s, e0, dims2(2, 1))), z0),
                     op2(s, _MUL, op2(s, _MATMUL, gate, tensor(s, e1, dims2(2, 1))), z1))
    var ins: List[Int] = [x]
    var outs: List[Int] = [op1(s, _SOFTMAX, logits), gate]
    var dims: List[List[Int]] = [dims2(128, 2), dims2(128, 2)]
    return Built(ins^, outs^, dims^)


def _attention[E: Exec & Movable & Deinitable](mut s: OpsSession[E], arrays: List[List[Float32]]) raises -> Built:
    var q = tensor(s, arrays[0], dims4(2, 4, 6, 4))
    var kv = tensor(s, arrays[1], dims4(2, 4, 6, 4))
    var scores = scale(s, op2(s, _MATMUL, q, op1(s, _TRANSPOSE, kv)), 0.5)
    var out = op2(s, _MATMUL, op1(s, _SOFTMAX, scores), kv)
    var one: List[Int] = [out]
    var axes: List[Int] = [0, 0, 2, 1, 3]
    var permuted = opi(s, _PERMUTE, one, axes, List[Float32]())
    var p1: List[Int] = [permuted]
    var shape: List[Int] = [0, 2, 6, 16]
    var merged = opi(s, _RESHAPE_N, p1, shape, List[Float32]())
    # Session.split(merged, 2, axis=2): operation 40 with [axis, section, 1, count]
    var m1: List[Int] = [merged]
    var first_half: List[Int] = [2, 0, 1, 2]
    var second_half: List[Int] = [2, 1, 1, 2]
    var piece0 = opi(s, _NARROW, m1, first_half, List[Float32]())
    var piece1 = opi(s, _NARROW, m1, second_half, List[Float32]())
    var pieces: List[Int] = [piece1, piece0]
    var axis2: List[Int] = [2]
    var swapped = opi(s, _CONCAT, pieces, axis2, List[Float32]())
    var max_words: List[Int] = [-1, 1]
    var top = opi(s, _MAX, m1, max_words, List[Float32]())
    var ins: List[Int] = [q, kv]
    var outs: List[Int] = [merged, swapped, top]
    var dims: List[List[Int]] = [dims3(2, 6, 16), dims3(2, 6, 16), dims2(2, 16)]
    return Built(ins^, outs^, dims^)


def _build[E: Exec & Movable & Deinitable](mut s: OpsSession[E], fixture_id: Int, arrays: List[List[Float32]]) raises -> Built:
    if fixture_id == 0:
        return _mixture(s, arrays)
    return _attention(s, arrays)


def _inputs(fixture_id: Int, seed: Int) -> List[List[Float32]]:
    if fixture_id == 0:
        var out: List[List[Float32]] = [normal(512, seed)]
        return out^
    var out: List[List[Float32]] = [normal(192, seed), normal(192, seed + 1)]
    return out^


def _in_cols(fixture_id: Int) -> Int:
    return 4


def _run[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], pid: Int, fixture_id: Int, arrays: List[List[Float32]], out_dims: List[List[Int]]
) raises -> List[List[Float32]]:
    var srcs = List[CastIn]()
    var cols = _in_cols(fixture_id)
    for k in range(len(arrays)):
        srcs.append(CastIn.f32_c(Int(arrays[k].unsafe_ptr()), len(arrays[k]) // cols, cols))
    var results = List[List[Float32]]()
    for k in range(len(out_dims)):
        results.append(List[Float32](length=_count(out_dims[k]), fill=Float32(-7.5)))
    var dsts = List[FP]()
    for k in range(len(results)):
        dsts.append(FP(unsafe_from_address=Int(results[k].unsafe_ptr())))
    s.run_program(pid, srcs, dsts)
    _ = arrays
    return results^


def _same_shape_list(a: ShapeN, dims: List[Int]) raises -> Bool:
    return a.same(ShapeN.of(dims))


def run_core_ops_program_fixture[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], mut again: OpsSession[E], fixture_id: Int
) raises:
    require(fixture_id >= 0 and fixture_id < CORE_OPS_PROGRAM_FIXTURES, "unknown program fixture")
    var a = _inputs(fixture_id, 40 + 10 * fixture_id)
    var b = _inputs(fixture_id, 45 + 10 * fixture_id)
    var built = _build(s, fixture_id, a)
    var eager = List[List[Float32]]()
    for k in range(len(built.outs)):
        eager.append(values(s, built.outs[k], built.out_dims[k]))
    var pid = s.capture(built.ins, built.outs)
    var p = s.program(pid)
    require(len(p.in_shapes) == len(built.ins), "program input count")
    for k in range(len(built.ins)):
        require(p.in_shapes[k].same(s.shape_nd(built.ins[k])), "program input shape " + String(k))
    require(len(p.out_shapes) == len(built.outs), "program output count")
    for k in range(len(built.outs)):
        require(_same_shape_list(p.out_shapes[k], built.out_dims[k]), "program output shape " + String(k))
    s.reset()
    var got = _run(s, pid, fixture_id, a, built.out_dims)
    var second = _run(s, pid, fixture_id, a, built.out_dims)
    var moved = _run(s, pid, fixture_id, b, built.out_dims)
    for k in range(len(eager)):
        same_words(got[k], eager[k], "fixture " + String(fixture_id) + ": the program differs from the eager session, output " + String(k))
        same_words(second[k], got[k], "fixture " + String(fixture_id) + ": a second run differs, output " + String(k))
    var built2 = _build(again, fixture_id, b)
    for k in range(len(built2.outs)):
        same_words(moved[k], values(again, built2.outs[k], built2.out_dims[k]),
                   "fixture " + String(fixture_id) + ": the run on new inputs differs from the eager session, output " + String(k))


def core_ops_program_records(fixture_id: Int) -> Bool:
    """Programs run in an inference session (Session(record=False))."""
    return False


# ---- The host column (was core_ops_program_cpu_suite.mojo) ----
# Captured programs against the eager session: each fixture on its own two host sessions, the fixtures run
# across host cores with a private failure slot each.
def _run_cpu(mut report: Report) raises:
    var cases = report.cases(CORE_OPS_PROGRAM_FIXTURES)
    def task(fixture_id: Int) {imm cases}:
        try:
            var record = core_ops_program_records(fixture_id)
            var s = OpsSession[HostExec](HostExec(), record)
            var again = OpsSession[HostExec](HostExec(), record)
            try:
                run_core_ops_program_fixture(s, again, fixture_id)
            except error:
                s.close()
                again.close()
                raise error
            s.close()
            again.close()
        except error:
            cases.fail(fixture_id, error)
    report.parallel(task, CORE_OPS_PROGRAM_FIXTURES)


# ---- The device column (was core_ops_program_gpu_suite.mojo) ----
# Captured programs against the eager session on the device, in order.
#
# DeviceExec shares sequence_ctx() and its pool, so independent concurrent
# sessions would race that context's ownership (as in ops_higher_gpu); the
# fixtures run one after another and each kernel keeps its own parallelism.
def _run_gpu(mut report: Report) raises:
    var cases = report.cases(CORE_OPS_PROGRAM_FIXTURES)
    for fixture_id in range(CORE_OPS_PROGRAM_FIXTURES):
        try:
            var record = core_ops_program_records(fixture_id)
            var s = OpsSession[DeviceExec](DeviceExec(), record)
            var again = OpsSession[DeviceExec](DeviceExec(), record)
            try:
                run_core_ops_program_fixture(s, again, fixture_id)
            except error:
                s.close()
                again.close()
                raise error
            s.close()
            again.close()
        except error:
            cases.fail(fixture_id, error)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if CPU_COLUMN_BUILD:
            _run_cpu(report)
        else:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("core_ops_program", cpu=True, gpu=True, serial="", run=run_cases)
