# SPDX-License-Identifier: Apache-2.0
"""Distributed and partitioned captured programs: independent gradients,
replica snapshots, optimizer migration and checkpoint continuation (moved
from python/mojolearn/tests/test_distributed_program.py and
test_distributed_program_boundaries.py, lane tmig-training-2, 2026-11-15),
through the typed entries the public doors call
(autograd/distributed_entries.mojo distributed_open_native, DistributedProgram.fit,
distributed_run_native, distributed_state_native, distributed_save_native,
distributed_load_native, distributed_ready, distributed_close_native; the
partition_* twins in autograd/partition_entries.mojo) with the doors' words
(loss 'mse' = 1, delta 1.0, no selected parameters).

Devices: (0,) on the host column; on the device column
MOJOLEARN_DISTRIBUTED_PROGRAM_DEVICES (default (0,)), as the former tests.

  0, 1  SGD analytic (logical_shards 1 and 3, raised to the device count):
        x (3, 1) @ w (1, 1) from zeros, lr 1/8, one epoch of x = 1, y = 2
        (batch 5, no shuffle): the loss curve is [4], w is 0.5, run(x) is
        0.5, step 1, epoch 1, and the session's own w is still 0.
  2     optimizer migration: a 3-epoch Adam (lr 0.01) program fit (batch 4,
        no shuffle), then distribute((0,), 1 shard): epoch 3, step 6, the
        same parameter bits; one more distributed epoch equals one more
        program epoch word for word.
  3-10  detached paths (route distribute, then partition; graphs mixed_shared,
        mixed_distinct, detached, copied): one SGD step (lr 1/16) of MSE to
        zero at x = [[1], [2]], w = 0.25 gives the analytic parameters with no
        derivative through detach or copy.
  11,12 zero-gradient AdamW (lr 1/16, weight decay 1/8; route distribute,
        then partition): three epochs only decay the parameter (three dyadic
        float32 updates), moments stay zero, step = epoch = 3; a checkpoint
        after epoch 1 restores that state word for word and continues to the
        same final state; the handle refuses use after close.

The Adam/dropout checkpoint resume, the physical multi-GPU topology replay,
the refusals and the damaged checkpoint stay in the Python test; the audited
Python-frame checks (audit_native_call) stay at the boundary.

Serial: sessions, distributed programs and partitions live in process-wide
tables (autograd/ops_api.mojo _OPS_SESSIONS, distributed_entries.mojo
_DISTRIBUTED, partition_entries.mojo _PARTITIONS) whose open/close append to
and pop from unlocked lists. Each program runs its own ranks in parallel.
"""
from std.ffi import external_call
from std.memory import bitcast
from std.os import getenv, remove
from autograd.ops_api import (
    OpsSession, ops_session, ops_session_open, ops_session_close, OPS_HOST, OPS_DEVICE,
    OPS_MATMUL, OPS_ADD, OPS_DETACH, OPS_COPY,
)
from autograd.optimizer_requests import OpsOptimizerRequest
from autograd.distributed_entries import (
    distributed,
    distributed_open_native,
    distributed_load_native,
    distributed_run_native,
    distributed_state_native,
    distributed_save_native,
    distributed_ready,
    distributed_close_native,
)
from autograd.partition_entries import (
    partition,
    partition_open_native,
    partition_load_native,
    partition_state_native,
    partition_save_native,
    partition_close_native,
    partition_ready,
)
from autograd.program_entries import program_fit_parameters, program_fit_native
from autograd.program_io import ProgramInput
from autograd.rankn import ShapeN
from core.cast_in import CastIn
from sequence.exec_trait import Exec
from sequence.ops import FP
from checks.native_tests.ops_higher import tensor, op1, op2, values
from checks.native_tests.core_ops_common import dims2
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from sequence.x_sequence_impl import device_executors
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


comptime DISTRIBUTED_FIXTURES = 13
comptime _LOSS_MSE = 1
comptime _SERIAL = "sessions, distributed programs and partitions live in unlocked process-wide tables"


def _optimizer(kind: Int, lr: Float32, weight_decay: Float32) -> OpsOptimizerRequest:
    """`Session.optimizer(kind, lr, weight_decay=...)` with the public defaults
    (momentum 0, betas 0.9/0.999, eps 1e-8, constant schedule)."""
    return OpsOptimizerRequest(kind, False, 0, 1, lr, Float32(0.0), Float32(0.9), Float32(0.999), Float32(1e-8),
                               weight_decay, Float32(0.1), Float32(0.0))


def _fill(n: Int, v: Float32) -> List[Float32]:
    return List[Float32](length=n, fill=v)


def _words(values: List[Float32]) -> List[UInt32]:
    var out = List[UInt32](length=len(values), fill=0)
    for i in range(len(values)):  # small-loop(check comparison): fixture-sized, no product work
        out[i] = bitcast[DType.uint32](values[i])
    return out^


def _same(a: List[Float32], b: List[Float32], what: String) raises:
    require(len(a) == len(b), what + ": lengths differ")
    var wa = _words(a)
    var wb = _words(b)
    for i in range(len(a)):  # small-loop(check comparison): fixture-sized, no product work
        require(wa[i] == wb[i], what + ": word " + String(i) + " differs")


def _exactly(a: List[Float32], want: Float32, what: String) raises:
    for i in range(len(a)):  # small-loop(check comparison): fixture-sized, no product work
        require(bitcast[DType.uint32](a[i]) == bitcast[DType.uint32](want),
                what + ": element " + String(i) + " is " + String(a[i]) + ", expected " + String(want))


def _route_is_partition(unit: Int) -> Bool:
    if unit >= 3 and unit <= 10:
        return unit >= 7
    return unit == 12


def _checkpoint_path(unit: Int) -> String:
    var base = getenv("TMPDIR", "/tmp")
    return (base + "/mojolearn-check-distributed-" + String(external_call["getpid", Int32]()) + "-"
            + String(unit) + ".mjl")


def _open[E: Exec & Movable & Deinitable, KIND: Int](
    partitioned: Bool, sid: Int, pid: Int, devices: List[Int], logical: Int, opt: Int, seed: Int, var executors: List[E]
) raises -> Int:
    if partitioned:
        return partition_open_native[E, KIND](sid, pid, devices, max(2, len(devices)), 1, 2, opt, List[Int](), seed,
                                               executors^)
    return distributed_open_native[E, KIND](sid, pid, devices, logical, opt, List[Int](), seed, executors^)


def _close[E: Exec & Movable & Deinitable, KIND: Int](partitioned: Bool, id: Int) raises:
    if partitioned:
        partition_close_native[E, KIND](id)
    else:
        distributed_close_native[E, KIND](id)


def _fit[E: Exec & Movable & Deinitable, KIND: Int](
    partitioned: Bool, id: Int, x: List[Float32], y: List[Float32], epochs: Int, batch: Int
) raises -> List[Float32]:
    """One-column x and y, no shuffle; returns this call's loss curve."""
    var rows = len(x)
    var srcs: List[CastIn] = [CastIn.f32_c(Int(x.unsafe_ptr()), rows, 1)]
    var shapes: List[ShapeN] = [ShapeN.of(dims2(rows, 1))]
    var target = CastIn.f32_c(Int(y.unsafe_ptr()), len(y), 1)
    var curve = List[Float32](length=epochs, fill=Float32(-1))
    if partitioned:
        var d = partition[E, KIND](id)
        d[].fit(srcs, shapes, target, _LOSS_MSE, Float32(1.0), epochs, batch, False,
                FP(unsafe_from_address=Int(curve.unsafe_ptr())))
    else:
        var d = distributed[E, KIND](id)
        d[].fit(srcs, shapes, target, _LOSS_MSE, Float32(1.0), epochs, batch, False,
                FP(unsafe_from_address=Int(curve.unsafe_ptr())))
    _ = x
    _ = y
    return curve^


struct _State(Movable):
    """Parameters, first and second moments (flat per parameter) and the
    counters, from either route."""
    var parameters: List[List[Float32]]
    var first: List[List[Float32]]
    var second: List[List[Float32]]
    var epoch: Int
    var step: Int
    var seed: UInt64

    def __init__(out self):
        self.parameters = List[List[Float32]]()
        self.first = List[List[Float32]]()
        self.second = List[List[Float32]]()
        self.epoch = 0
        self.step = 0
        self.seed = 0


def _state[E: Exec & Movable & Deinitable, KIND: Int](partitioned: Bool, id: Int) raises -> _State:
    var out = _State()
    if partitioned:
        var st = partition_state_native[E, KIND](id)
        for j in range(len(st.arrays[0])):  # driver-loop: over returned parameter tensors
            out.parameters.append(st.arrays[0][j].data.copy())
            out.first.append(st.arrays[1][j].data.copy())
            out.second.append(st.arrays[2][j].data.copy())
        out.epoch = st.epoch
        out.step = st.step
        out.seed = st.seed
    else:
        var st = distributed_state_native[E, KIND](id)
        for j in range(len(st.parameters)):  # driver-loop: over returned parameter tensors
            out.parameters.append(st.parameters[j].data.copy())
            out.first.append(st.first_moments[j].data.copy())
            out.second.append(st.second_moments[j].data.copy())
        out.epoch = st.epoch
        out.step = st.step
        out.seed = st.seed
    return out^


def _same_state(a: _State, b: _State, what: String) raises:
    require(a.epoch == b.epoch and a.step == b.step and a.seed == b.seed, what + ": counters differ")
    require(len(a.parameters) == len(b.parameters), what + ": parameter counts differ")
    for j in range(len(a.parameters)):  # driver-loop: over parameter tensors
        _same(a.parameters[j], b.parameters[j], what + " parameter " + String(j))
        _same(a.first[j], b.first[j], what + " first moment " + String(j))
        _same(a.second[j], b.second[j], what + " second moment " + String(j))


def _program_fit[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, pid: Int, opt: Int, x: List[Float32], y: List[Float32], epochs: Int, batch: Int
) raises:
    var params = program_fit_parameters[E, KIND](sid, pid, 0, False, List[Int]())
    var owners = List[ProgramInput]()
    owners.append(ProgramInput(CastIn.f32_c(Int(x.unsafe_ptr()), len(x), 1), ShapeN.of(dims2(len(x), 1)),
                               List[Float64](), List[Int64]()))
    var targets = List[ProgramInput]()
    targets.append(ProgramInput(CastIn.f32_c(Int(y.unsafe_ptr()), len(y), 1), ShapeN.of(dims2(len(y), 1)),
                                List[Float64](), List[Int64]()))
    var curve = List[Float32](length=epochs, fill=Float32(-1))
    _ = program_fit_native[E, KIND](sid, pid, owners^, targets^, _LOSS_MSE, Float32(1.0), opt, params, epochs, batch,
                                    False, UInt64(0), FP(unsafe_from_address=Int(curve.unsafe_ptr())), 0, 0)
    # the fit wrote the curve through its address: keep it (and the inputs) alive past the call
    _ = curve^
    _ = x
    _ = y


def run_distributed_fixture[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, unit: Int, devices: List[Int], var executors: List[E], var single: List[E]
) raises:
    """`executors`: one per entry of `devices`; `single`: one more executor
    for device 0 (the migration's distribute((0,)) and the checkpoint load)."""
    var s = ops_session[E, KIND](sid)
    if unit <= 1:
        var xh = tensor(s[], _fill(3, 0.0), dims2(3, 1))
        var w = tensor(s[], _fill(1, 0.0), dims2(1, 1))
        var ins: List[Int] = [xh]
        var outs: List[Int] = [op2(s[], OPS_MATMUL, xh, w)]
        var pid = s[].capture(ins, outs)
        var opt = s[].optimizer(_optimizer(0, 0.125, 0.0))
        var shards = 1 if unit == 0 else 3
        var id = _open[E, KIND](False, sid, pid, devices, max(shards, len(devices)), opt, 0, executors^)
        try:
            var x = _fill(5, 1.0)
            var curve = _fit[E, KIND](False, id, x, _fill(5, 2.0), 1, 5)
            require(len(curve) == 1, "one epoch loss")
            _exactly(curve, 4.0, "the analytic loss curve")
            var st = _state[E, KIND](False, id)
            require(len(st.parameters) == 1, "one parameter")
            _exactly(st.parameters[0], 0.5, "w after one SGD step")
            require(st.step == 1 and st.epoch == 1, "step 1, epoch 1")
            var srcs: List[CastIn] = [CastIn.f32_c(Int(x.unsafe_ptr()), 5, 1)]
            var shapes: List[ShapeN] = [ShapeN.of(dims2(5, 1))]
            var ran = distributed_run_native[E, KIND](id, srcs, shapes)
            _ = x
            require(len(ran.data) == 5, "run gives five rows")
            _exactly(ran.data, 0.5, "run after the step")
            _exactly(values(s[], w, dims2(1, 1)), 0.0, "the session's own w is never mutated")
        except error:
            distributed_close_native[E, KIND](id)
            raise error
        distributed_close_native[E, KIND](id)
        return
    if unit == 2:
        var xh = tensor(s[], _fill(3, 0.0), dims2(3, 1))
        var w = tensor(s[], _fill(1, 0.0), dims2(1, 1))
        var ins: List[Int] = [xh]
        var outs: List[Int] = [op2(s[], OPS_MATMUL, xh, w)]
        var pid = s[].capture(ins, outs)
        var opt = s[].optimizer(_optimizer(1, 0.01, 0.0))
        var x = _fill(8, 1.0)
        var y = _fill(8, 2.0)
        _program_fit[E, KIND](sid, pid, opt, x, y, 3, 4)
        var before = values(s[], w, dims2(1, 1))
        var first: List[Int] = [0]
        var id = _open[E, KIND](False, sid, pid, first, 1, opt, 0, single^)
        var actual = List[Float32]()
        try:
            var st = _state[E, KIND](False, id)
            require(st.step == 6 and st.epoch == 3, "the distributed copy starts at epoch 3, step 6")
            _same(st.parameters[0], before, "the migrated parameter")
            _ = _fit[E, KIND](False, id, x, y, 1, 4)
            actual = _state[E, KIND](False, id).parameters[0].copy()
        except error:
            distributed_close_native[E, KIND](id)
            raise error
        distributed_close_native[E, KIND](id)
        _program_fit[E, KIND](sid, pid, opt, x, y, 1, 4)
        _same(actual, values(s[], w, dims2(1, 1)), "one more distributed epoch equals one more program epoch")
        return
    var partitioned = _route_is_partition(unit)
    var data: List[Float32] = [1.0, 2.0]
    var target = _fill(2, 0.0)
    if unit <= 10:
        var graph = (unit - 3) % 4
        var xh = tensor(s[], data, dims2(2, 1))
        var w = tensor(s[], _fill(1, 0.25), dims2(1, 1))
        var product = op2(s[], OPS_MATMUL, xh, w)
        var out: Int
        var expected = List[Float32]()
        if graph == 0:
            # mixed_shared: mean(2 * (0.5 x) x) = 2.5, with no derivative through detach
            out = op2(s[], OPS_ADD, product, op1(s[], OPS_DETACH, product))
            expected.append(Float32(0.25) - Float32(0.0625 * 2.5))
        elif graph == 1:
            # mixed_distinct
            var v = tensor(s[], _fill(1, 0.5), dims2(1, 1))
            out = op2(s[], OPS_ADD, product, op1(s[], OPS_DETACH, op2(s[], OPS_MATMUL, xh, v)))
            expected.append(Float32(0.25) - Float32(0.0625 * 3.75))
            expected.append(Float32(0.5))
        elif graph == 2:
            out = op1(s[], OPS_DETACH, product)
            expected.append(Float32(0.25))
        else:
            out = op1(s[], OPS_COPY, product)
            expected.append(Float32(0.25))
        var ins: List[Int] = [xh]
        var outs: List[Int] = [out]
        var pid = s[].capture(ins, outs)
        var opt = s[].optimizer(_optimizer(0, 0.0625, 0.0))
        var id = _open[E, KIND](partitioned, sid, pid, devices, max(2, len(devices)), opt, 91, executors^)
        try:
            distributed_ready_or_partition[E, KIND](partitioned, id)
            _ = _fit[E, KIND](partitioned, id, data, target, 1, 2)
            var st = _state[E, KIND](partitioned, id)
            require(st.step == 1 and st.epoch == 1, "step = epoch = 1")
            require(len(st.parameters) == len(expected), "the parameter count")
            for j in range(len(expected)):  # driver-loop: over the captured parameters
                require(len(st.parameters[j]) == 1, "a (1, 1) parameter")
                _exactly(st.parameters[j], expected[j], "detached-path parameter " + String(j))
        except error:
            _close[E, KIND](partitioned, id)
            raise error
        _close[E, KIND](partitioned, id)
        return
    require(unit <= 12, "unknown distributed fixture")
    # zero-gradient AdamW: three dyadic decays with exact float32 oracles
    var expected = Float32(0.25)
    for _ in range(3):  # small-loop(3: oracle updates): the analytic sequence
        expected = expected - Float32(0.0625) * (Float32(0.125) * expected)
    var xh = tensor(s[], data, dims2(2, 1))
    var w = tensor(s[], _fill(1, 0.25), dims2(1, 1))
    var ins: List[Int] = [xh]
    var outs: List[Int] = [op1(s[], OPS_DETACH, op2(s[], OPS_MATMUL, xh, w))]
    var pid = s[].capture(ins, outs)
    var opt = s[].optimizer(_optimizer(2, 0.0625, 0.125))
    var path = _checkpoint_path(unit)
    var id = _open[E, KIND](partitioned, sid, pid, devices, max(2, len(devices)), opt, 91, executors^)
    var saved = _State()
    var final = _State()
    try:
        distributed_ready_or_partition[E, KIND](partitioned, id)
        _ = _fit[E, KIND](partitioned, id, data, target, 1, 2)
        saved = _state[E, KIND](partitioned, id)
        if partitioned:
            _ = partition_save_native[E, KIND](id, path)
        else:
            _ = distributed_save_native[E, KIND](id, path)
        _ = _fit[E, KIND](partitioned, id, data, target, 2, 2)
        final = _state[E, KIND](partitioned, id)
        require(len(final.parameters) == 1, "one parameter")
        _exactly(final.parameters[0], expected, "AdamW decay only")
        _exactly(final.first[0], 0.0, "first moments stay zero")
        _exactly(final.second[0], 0.0, "second moments stay zero")
        require(final.step == 3 and final.epoch == 3, "step = epoch = 3")
    except error:
        _close[E, KIND](partitioned, id)
        try:
            remove(path)
        except:
            pass
        raise error
    _close[E, KIND](partitioned, id)
    var zero: List[Int] = [0]
    var restored = 0
    try:
        if partitioned:
            restored = partition_load_native[E, KIND](path, zero, single^)
        else:
            restored = distributed_load_native[E, KIND](path, zero, single^)
    except error:
        try:
            remove(path)
        except:
            pass
        raise error
    try:
        remove(path)
    except:
        pass
    try:
        distributed_ready_or_partition[E, KIND](partitioned, restored)
        _same_state(_state[E, KIND](partitioned, restored), saved, "the restored checkpoint")
        _ = _fit[E, KIND](partitioned, restored, data, target, 2, 2)
        _same_state(_state[E, KIND](partitioned, restored), final, "the restored run's continuation")
    except error:
        _close[E, KIND](partitioned, restored)
        raise error
    _close[E, KIND](partitioned, restored)
    var refused = False
    try:
        distributed_ready_or_partition[E, KIND](partitioned, restored)
    except:
        refused = True
    require(refused, "a closed handle is refused")


def distributed_ready_or_partition[E: Exec & Movable & Deinitable, KIND: Int](partitioned: Bool, id: Int) raises:
    """The `__enter__` check: the handle is open and usable."""
    if partitioned:
        partition_ready[E, KIND](id)
    else:
        distributed_ready[E, KIND](id)


# ---- The host column ----
def _run_cpu(mut report: Report) raises:
    var cases = report.cases(DISTRIBUTED_FIXTURES)
    def task(unit: Int) {imm cases}:
        try:
            var sid = ops_session_open[HostExec, OPS_HOST](HostExec(), False)
            try:
                var executors = List[HostExec]()
                executors.append(HostExec())
                var single = List[HostExec]()
                single.append(HostExec())
                var devices: List[Int] = [0]
                run_distributed_fixture[HostExec, OPS_HOST](sid, unit, devices, executors^, single^)
            except error:
                ops_session_close[HostExec, OPS_HOST](sid)
                raise error
            ops_session_close[HostExec, OPS_HOST](sid)
        except error:
            cases.fail(unit, error)
    report.parallel(task, DISTRIBUTED_FIXTURES)


# ---- The device column ----
def _devices() raises -> List[Int]:
    """MOJOLEARN_DISTRIBUTED_PROGRAM_DEVICES (comma-separated physical ids), default (0,)."""
    var text = getenv("MOJOLEARN_DISTRIBUTED_PROGRAM_DEVICES", "0")
    var out = List[Int]()
    for part in text.split(","):
        out.append(Int(String(part)))
    return out^


def _run_gpu(mut report: Report) raises:
    var cases = report.cases(DISTRIBUTED_FIXTURES)
    def task(unit: Int) {imm cases}:
        try:
            var sid = ops_session_open[DeviceExec, OPS_DEVICE](DeviceExec(), False)
            try:
                var devices = _devices()
                var executors = device_executors(devices)
                var zero: List[Int] = [0]
                var single = device_executors(zero)
                run_distributed_fixture[DeviceExec, OPS_DEVICE](sid, unit, devices, executors^, single^)
            except error:
                ops_session_close[DeviceExec, OPS_DEVICE](sid)
                raise error
            ops_session_close[DeviceExec, OPS_DEVICE](sid)
        except error:
            cases.fail(unit, error)
    report.parallel(task, DISTRIBUTED_FIXTURES)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if CPU_COLUMN_BUILD:
            _run_cpu(report)
        else:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("training_distributed_program", cpu=True, gpu=True, serial=_SERIAL, run=run_cases)
