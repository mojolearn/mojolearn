# SPDX-License-Identifier: Apache-2.0
"""Native partitioned programs on one device, moved from
test_partitioned_program.py (2026-11-14), through the typed entries the
public door calls (autograd/partition_entries.mojo partition_open_native,
partition_run_native, partition_state_native, partition_residency_native,
PartitionedProgram.fit, partition_close_native) with the door's words
(devices from MOJOLEARN_DISTRIBUTED_PROGRAM_DEVICES on a GPU, else (0,); no selected parameters, seed 0, loss 'mse' = 1, delta 1.0).

  0 column partition: x (3, 1) @ w (1, 4) from zeros, SGD lr 0.125, max(2, devices) stages,
    2 tensor shards, 3 microbatches; one epoch of x = 1, y = 2 (batch 5, no
    shuffle): the loss curve is [4], both column shards hold 0.125 (atol
    2e-8), run(x) is 0.125, one partitioned product, 4 parameter elements, 2
    owners (distinct physical owners, none holding all 4, on several devices).
  1 pipeline: tanh(x @ w + b) @ v, SGD lr 0.0625, 2 stages, 1 shard, 1
    microbatch: run(x) against a float64 forward (rtol 2e-6, atol 2e-7) and
    one epoch of batch 3 against the float64 dense parameter gradient step.
  2 shared parameter: y = x w + x w at w = 0, MSE to 2, SGD lr 1/16: w
    becomes 0.5 exactly (the alias's gradient is not lost) and run gives 1.
  3 Adam moments: a 3-epoch program fit (batch 4, no shuffle) then a
    partition (2 stages, 2 shards, 1 microbatch) starts at epoch 3, step 6
    with nonzero first moments; one more partitioned epoch equals one more
    program epoch word for word.
The multi-device pipeline overlap, the checkpoint/resume with dropout and
the constructor/checkpoint refusals stay in the Python test.
"""
from std.math import tanh
from autograd.ops_api import OpsSession, ops_session
from autograd.optimizer_requests import OpsOptimizerRequest
from autograd.partition_entries import (
    partition, partition_open_native, partition_run_native, partition_state_native,
    partition_residency_native, partition_close_native,
)
from autograd.program_entries import program_fit_parameters, program_fit_native
from autograd.program_io import ProgramInput
from autograd.rankn import ShapeN
from core.cast_in import CastIn
from sequence.exec_trait import Exec
from sequence.ops import FP
from checks.native_tests.ops_higher_cases import require, tensor, op1, op2, values, same_words
from checks.native_tests.core_ops_common import wide, allclose, exact, dims2, opi

comptime CORE_PARTITIONED_PROGRAM_FIXTURES = 4

comptime _MATMUL = 1
comptime _ADD = 2
comptime _TANH = 8
comptime _LINEAR = 19
comptime _LOSS_MSE = 1


def _sgd(lr: Float32) -> OpsOptimizerRequest:
    return OpsOptimizerRequest(0, False, 0, 1, lr, Float32(0.0), Float32(0.9), Float32(0.999), Float32(1e-8),
                               Float32(0.0), Float32(0.1), Float32(0.0))


def _adam(lr: Float32) -> OpsOptimizerRequest:
    return OpsOptimizerRequest(1, False, 0, 1, lr, Float32(0.0), Float32(0.9), Float32(0.999), Float32(1e-8),
                               Float32(0.0), Float32(0.1), Float32(0.0))


def _fill(n: Int, v: Float32) -> List[Float32]:
    return List[Float32](length=n, fill=v)


def _arange(n: Int, divisor: Float32) -> List[Float32]:
    var out = List[Float32](length=n, fill=0)
    for i in range(n):
        out[i] = Float32(i) / divisor
    return out^


def _open[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, pid: Int, devices: List[Int], stages: Int, shards: Int, micro: Int, opt: Int, var executors: List[E]
) raises -> Int:
    return partition_open_native[E, KIND](sid, pid, devices, stages, shards, micro, opt, List[Int](), 0, executors^)


def _fit[E: Exec & Movable & Deinitable, KIND: Int](
    id: Int, x: List[Float32], xc: Int, y: List[Float32], yc: Int, epochs: Int, batch: Int
) raises -> List[Float32]:
    var rows = len(x) // xc
    var srcs: List[CastIn] = [CastIn.f32_c(Int(x.unsafe_ptr()), rows, xc)]
    var shapes: List[ShapeN] = [ShapeN.of(dims2(rows, xc))]
    var target = CastIn.f32_c(Int(y.unsafe_ptr()), len(y) // yc, yc)
    var curve = List[Float32](length=epochs, fill=Float32(-1))
    var d = partition[E, KIND](id)
    d[].fit(srcs, shapes, target, _LOSS_MSE, Float32(1.0), epochs, batch, False, FP(unsafe_from_address=Int(curve.unsafe_ptr())))
    _ = x
    _ = y
    return curve^


def _run[E: Exec & Movable & Deinitable, KIND: Int](id: Int, x: List[Float32], xc: Int) raises -> List[Float32]:
    var rows = len(x) // xc
    var srcs: List[CastIn] = [CastIn.f32_c(Int(x.unsafe_ptr()), rows, xc)]
    var shapes: List[ShapeN] = [ShapeN.of(dims2(rows, xc))]
    var g = partition_run_native[E, KIND](id, srcs, shapes)
    _ = x
    return g.words.copy()


def _program_fit[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, pid: Int, opt: Int, x: List[Float32], y: List[Float32], epochs: Int, batch: Int
) raises:
    var params = program_fit_parameters[E, KIND](sid, pid, 0, False, List[Int]())
    var owners = List[ProgramInput]()
    owners.append(ProgramInput(CastIn.f32_c(Int(x.unsafe_ptr()), len(x), 1), ShapeN.of(dims2(len(x), 1)), List[Float64](), List[Int64]()))
    var targets = List[ProgramInput]()
    targets.append(ProgramInput(CastIn.f32_c(Int(y.unsafe_ptr()), len(y) // 4, 4), ShapeN.of(dims2(len(y) // 4, 4)), List[Float64](), List[Int64]()))
    var curve = List[Float32](length=epochs, fill=Float32(-1))
    _ = program_fit_native[E, KIND](sid, pid, owners^, targets^, _LOSS_MSE, Float32(1.0), opt, params, epochs, batch, False,
                                    UInt64(0), FP(unsafe_from_address=Int(curve.unsafe_ptr())), 0, 0)
    _ = x
    _ = y


def _linear_program[E: Exec & Movable & Deinitable](mut s: OpsSession[E]) raises -> Tuple[Int, Int]:
    """(program, w): x (3, 1) @ w (1, 4) from zeros."""
    var x = tensor(s, _fill(3, 0.0), dims2(3, 1))
    var w = tensor(s, _fill(4, 0.0), dims2(1, 4))
    var ins: List[Int] = [x]
    var outs: List[Int] = [op2(s, _MATMUL, x, w)]
    return (s.capture(ins, outs), w)


def run_core_partitioned_program_fixture[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, fixture_id: Int, devices: List[Int], var executors: List[E]
) raises:
    """`devices` and `executors`: one per physical device (the suite's
    MOJOLEARN_DISTRIBUTED_PROGRAM_DEVICES, default (0,)); fixture 3 takes
    the first only, as the former test's partition((0,), ...)."""
    var s = ops_session[E, KIND](sid)
    var stages = max(2, len(devices))
    if fixture_id == 0:
        var lp = _linear_program(s[])
        var opt = s[].optimizer(_sgd(0.125))
        var id = _open[E, KIND](sid, lp[0], devices, stages, 2, 3, opt, executors^)
        try:
            var x = _fill(5, 1.0)
            var curve = _fit[E, KIND](id, x, 1, _fill(20, 2.0), 4, 1, 5)
            var four: List[Float32] = [4.0]
            exact(curve, four, "the column partition's loss curve")
            var st = partition_state_native[E, KIND](id)
            require(len(st.arrays[0]) == 2, "two column shards")
            for k in range(2):
                var want = List[Float64](length=len(st.arrays[0][k].data), fill=0.125)
                allclose(st.arrays[0][k].data, want, 0.0, 2e-8, "column shard " + String(k))
            allclose(_run[E, KIND](id, x, 1), List[Float64](length=20, fill=0.125), 0.0, 2e-8, "run after the step")
            var r = partition_residency_native[E, KIND](id)
            require(r.partitioned_products == 1, "one partitioned product")
            var total = 0
            for c in r.counts:
                total += c
            require(total == 4, "four parameter elements")
            require(len(r.owners) == 2, "two parameter owners")
            if len(devices) > 1:
                require(r.owners[0] != r.owners[1], "the two shards have distinct physical owners")
                var most = 0
                for c in r.counts:
                    most = max(most, c)
                require(most < 4, "no rank holds every parameter element")
        except error:
            partition_close_native[E, KIND](id)
            raise error
        partition_close_native[E, KIND](id)
        return
    if fixture_id == 1:
        var xv: List[Float32] = [0.5, -0.25, 0.125, 0.75, -0.5, 1.0]
        var wv = _arange(8, 32.0)
        var bv = _fill(4, 0.125)
        var vv = _arange(8, 64.0)
        var yv = _fill(6, 0.25)
        var xh = tensor(s[], _fill(6, 0.0), dims2(3, 2))
        var w = tensor(s[], wv, dims2(2, 4))
        var b = tensor(s[], bv, dims2(1, 4))
        var v = tensor(s[], vv, dims2(4, 2))
        var hs: List[Int] = [xh, w, b]
        var act: List[Int] = [0]
        var hidden = op1(s[], _TANH, opi(s[], _LINEAR, hs, act, List[Float32]()))
        var ins: List[Int] = [xh]
        var outs: List[Int] = [op2(s[], _MATMUL, hidden, v)]
        var pid = s[].capture(ins, outs)
        var opt = s[].optimizer(_sgd(0.0625))
        var id = _open[E, KIND](sid, pid, devices, stages, 1, 1, opt, executors^)
        try:
            # float64 forward and the dense gradient step
            var x = wide(xv)
            var h = List[Float64](length=12, fill=0)
            var pred = List[Float64](length=6, fill=0)
            for i in range(3):
                for j in range(4):
                    var z = Float64(bv[j])
                    for k in range(2):
                        z += x[2 * i + k] * Float64(wv[4 * k + j])
                    h[4 * i + j] = tanh(z)
                for o in range(2):
                    var p = Float64(0)
                    for j in range(4):
                        p += h[4 * i + j] * Float64(vv[2 * j + o])
                    pred[2 * i + o] = p
            allclose(_run[E, KIND](id, xv, 2), pred, 2e-6, 2e-7, "pipeline forward")
            _ = _fit[E, KIND](id, xv, 2, yv, 2, 1, 3)
            var go = List[Float64](length=6, fill=0)
            for i in range(6):
                go[i] = (2.0 / 6.0) * (pred[i] - Float64(yv[i]))
            var gv = List[Float64](length=8, fill=0)
            var gh = List[Float64](length=12, fill=0)
            for i in range(3):
                for j in range(4):
                    var t = Float64(0)
                    for o in range(2):
                        gv[2 * j + o] += h[4 * i + j] * go[2 * i + o]
                        t += go[2 * i + o] * Float64(vv[2 * j + o])
                    gh[4 * i + j] = t * (1.0 - h[4 * i + j] * h[4 * i + j])
            var nw = List[Float64](length=8, fill=0)
            var nb = List[Float64](length=4, fill=0)
            var nv = List[Float64](length=8, fill=0)
            for k in range(2):
                for j in range(4):
                    var g = Float64(0)
                    for i in range(3):
                        g += x[2 * i + k] * gh[4 * i + j]
                    nw[4 * k + j] = Float64(wv[4 * k + j]) - 0.0625 * g
            for j in range(4):
                var g = Float64(0)
                for i in range(3):
                    g += gh[4 * i + j]
                nb[j] = Float64(bv[j]) - 0.0625 * g
            for i in range(8):
                nv[i] = Float64(vv[i]) - 0.0625 * gv[i]
            var st = partition_state_native[E, KIND](id)
            require(len(st.arrays[0]) == 3, "three parameters")
            allclose(st.arrays[0][0].data, nw, 2e-6, 2e-7, "w after one step")
            allclose(st.arrays[0][1].data, nb, 2e-6, 2e-7, "b after one step")
            allclose(st.arrays[0][2].data, nv, 2e-6, 2e-7, "v after one step")
        except error:
            partition_close_native[E, KIND](id)
            raise error
        partition_close_native[E, KIND](id)
        return
    if fixture_id == 2:
        var a = tensor(s[], _fill(3, 0.0), dims2(3, 1))
        var w = tensor(s[], _fill(1, 0.0), dims2(1, 1))
        var h = op2(s[], _MATMUL, a, w)
        var ins: List[Int] = [a]
        var outs: List[Int] = [op2(s[], _ADD, h, h)]
        var pid = s[].capture(ins, outs)
        var opt = s[].optimizer(_sgd(0.0625))
        var id = _open[E, KIND](sid, pid, devices, stages, 2, 2, opt, executors^)
        try:
            var x = _fill(4, 1.0)
            _ = _fit[E, KIND](id, x, 1, _fill(4, 2.0), 1, 1, 4)
            var st = partition_state_native[E, KIND](id)
            var half: List[Float32] = [0.5]
            exact(st.arrays[0][0].data, half, "the shared parameter's step")
            exact(_run[E, KIND](id, x, 1), _fill(4, 1.0), "run after the step")
        except error:
            partition_close_native[E, KIND](id)
            raise error
        partition_close_native[E, KIND](id)
        return
    require(fixture_id == 3, "unknown partitioned fixture")
    var lp = _linear_program(s[])
    var opt = s[].optimizer(_adam(0.01))
    var x = _fill(8, 1.0)
    var y = _fill(32, 2.0)
    _program_fit[E, KIND](sid, lp[0], opt, x, y, 3, 4)
    var first: List[Int] = [devices[0]]
    var one = List[E]()
    one.append(executors.pop(0))
    var id = _open[E, KIND](sid, lp[0], first, 2, 2, 1, opt, one^)
    var actual = List[Float32]()
    try:
        var st = partition_state_native[E, KIND](id)
        require(st.epoch == 3 and st.step == 6, "the partition starts at epoch 3, step 6")
        var moved = False
        for k in range(len(st.arrays[1])):
            for v in st.arrays[1][k].data:
                moved = moved or v != 0
        require(moved, "the source Adam first moments moved with the parameters")
        _ = _fit[E, KIND](id, x, 1, y, 4, 1, 4)
        var after = partition_state_native[E, KIND](id)
        for k in range(len(after.arrays[0])):
            for v in after.arrays[0][k].data:
                actual.append(v)
    except error:
        partition_close_native[E, KIND](id)
        raise error
    partition_close_native[E, KIND](id)
    _program_fit[E, KIND](sid, lp[0], opt, x, y, 1, 4)
    same_words(actual, values(s[], lp[1], dims2(1, 4)), "one more partitioned epoch equals one more program epoch")
