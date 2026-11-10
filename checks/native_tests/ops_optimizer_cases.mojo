# SPDX-License-Identifier: Apache-2.0
"""Native session optimizer math, control state and checkpoint continuation.

Exact session/model-fit parity and exact checkpoint continuation preserve the
former checks. Independent Float64 equations additionally check the answer;
same-function agreement alone is not a numerical oracle. No identity digests.
"""
from std.memory import bitcast
from autograd.ops_api import OpsSession, OPS_LINEAR, OPS_MSE, OPS_MATMUL, OPS_SUM
from autograd.losses import REDUCTION_MEAN
from autograd.model import FitConfig, LOSS_MSE, TARGET_DENSE, parse_model, model_fit
from sequence.recurrent import OptConfig
from sequence.exec_trait import Exec
from sequence.ops import OPT_SGD
from core.cast_in import CastIn
from core.hostptr import f32_ptr
from core.host_parallel import host_parallelize
from checks.native_tests.ops_optimizer_oracle import (
    OPT_N, OPT_D, OPT_H, OPT_PARAMS, OPT_EPOCHS,
    optimizer_ints, optimizer_floats, optimizer_data, optimizer_oracle,
)

comptime OPS_OPTIMIZER_FIXTURES = 10


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _download[E: Exec & Movable & Deinitable](mut s: OpsSession[E], h: Int, count: Int) raises -> List[Float32]:
    var values = List[Float32](length=count, fill=0)
    s.download(h, f32_ptr(Int(values.unsafe_ptr())))
    return values^


def _upload[E: Exec & Movable & Deinitable](mut s: OpsSession[E], values: List[Float32], rows: Int, cols: Int, grad: Bool = False, offset: Int = 0) raises -> Int:
    var handle = s.upload(CastIn.f32_c(Int(values.unsafe_ptr()) + 4 * offset, rows, cols), grad)
    _ = values
    return handle


def _parameters[E: Exec & Movable & Deinitable](mut s: OpsSession[E], flat: List[Float32]) raises -> List[Int]:
    var result = List[Int]()
    result.append(_upload(s, flat, OPT_D, OPT_H, True))
    result.append(_upload(s, flat, 1, OPT_H, True, 15))
    result.append(_upload(s, flat, OPT_H, 1, True, 20))
    result.append(_upload(s, flat, 1, 1, True, 25))
    return result^


def _loss[E: Exec & Movable & Deinitable](mut s: OpsSession[E], params: List[Int], x: Int, y: Int) raises -> Int:
    var hidden = s.run_op(OPS_LINEAR, [x, params[0], params[1]], [1], [])
    var prediction = s.run_op(OPS_LINEAR, [hidden, params[2], params[3]], [0], [])
    return s.run_op(OPS_MSE, [prediction, y], [REDUCTION_MEAN], [])


def _steps[E: Exec & Movable & Deinitable](mut s: OpsSession[E], opt: Int, params: List[Int], x: Int, y: Int, steps: Int, advance_schedule: Bool) raises:
    for epoch in range(steps):  # dependent training/optimizer recurrence
        s.backward(_loss(s, params, x, y), False)
        s.step(opt, params)
        if advance_schedule:
            s.scheduler_step(opt)
        s.reset()


def _flat[E: Exec & Movable & Deinitable](mut s: OpsSession[E], params: List[Int]) raises -> List[Float32]:
    var result = List[Float32](length=OPT_PARAMS, fill=0)
    var offsets: List[Int] = [0, 15, 20, 25, 26]
    for parameter in range(4):  # driver-loop: over four owned parameter tensors
        var values = _download(s, params[parameter], offsets[parameter + 1] - offsets[parameter])
        for j in range(len(values)):
            result[offsets[parameter] + j] = values[j]
    return result^


def _check_parameters(actual: List[Float32], wanted: List[Float64]) raises:
    for i in range(OPT_PARAMS):
        # Additional Float64 reference: six float32 forward/backward/update
        # rounds introduce rounding. The original exact parity remains below.
        require(abs(Float64(actual[i]) - wanted[i]) <= 2e-6 + 1e-5 * abs(wanted[i]), "independent optimizer parameter " + String(i))


def _check_state[E: Exec & Movable & Deinitable](mut s: OpsSession[E], opt: Int,
    params: List[Int], wanted_m: List[Float64], wanted_v: List[Float64], steps: Int) raises:
    var offsets: List[Int] = [0, 15, 20, 25, 26]
    for parameter in range(4):  # driver-loop: over four optimizer state records
        var state = s.optimizer_state(opt, params[parameter])
        require(state[0] == steps, "native optimizer step count")
        var count = offsets[parameter + 1] - offsets[parameter]
        if state[4] != 0:
            var first = _download(s, state[4], count)
            for j in range(count):
                var expected = wanted_m[offsets[parameter] + j]
                require(abs(Float64(first[j]) - expected) <= 2e-8 + 2e-4 * abs(expected), "independent first-moment/momentum state")
        if state[5] != 0:
            var second = _download(s, state[5], count)
            for j in range(count):
                var expected = wanted_v[offsets[parameter] + j]
                require(abs(Float64(second[j]) - expected) <= 1e-12 + 2e-4 * abs(expected), "independent second-moment state")
        require(state[6] == 0, "unexpected third state for SGD/Adam/AdamW")


def _sequential[E: Exec](mut ex: E, fixture_id: Int) raises -> List[Float32]:
    var data = optimizer_data()
    var params = data[2].copy()
    var program: List[Int] = [1, OPT_D, 3, 1, OPT_H, 1, 0, 2, 0, 0, 0, 1, 1, 1, 0]
    var table: List[Float32] = [0, 0, 0]
    var spec = parse_model(program, table)
    var ints = optimizer_ints(fixture_id)
    var fs = optimizer_floats(fixture_id)
    var opt = OptConfig(ints[0], ints[1], fs[1] if ints[0] == OPT_SGD else fs[2],
        Float32(0) if ints[0] == OPT_SGD else fs[3], fs[4], fs[5], Float32(0))
    var cfg = FitConfig(LOSS_MSE, TARGET_DENSE, opt, fs[0], ints[2], ints[3], fs[6],
        OPT_N, OPT_EPOCHS, False, UInt64(0), 0, 0, 1, False)
    var curve = List[Float32](length=OPT_EPOCHS, fill=0)
    var xin = CastIn.f32_c(Int(data[0].unsafe_ptr()), OPT_N, OPT_D)
    var yin = CastIn.f32_c(Int(data[1].unsafe_ptr()), OPT_N, 1)
    require(model_fit(ex, spec, cfg, xin, yin, OPT_N, f32_ptr(Int(params.unsafe_ptr())),
        f32_ptr(Int(curve.unsafe_ptr()))) == OPT_EPOCHS, "native model_fit epoch count")
    ex.sync()
    _ = data
    _ = curve
    return params^


def _training[E: Exec & Movable & Deinitable](mut s: OpsSession[E], mut ex: E, fixture_id: Int) raises:
    var data = optimizer_data()
    var params = _parameters(s, data[2])
    var x = _upload(s, data[0], OPT_N, OPT_D)
    var y = _upload(s, data[1], OPT_N, 1)
    var opt = s.optimizer(optimizer_ints(fixture_id), optimizer_floats(fixture_id))
    _steps(s, opt, params, x, y, OPT_EPOCHS, True)
    var actual = _flat(s, params)
    var oracle = optimizer_oracle(fixture_id, OPT_EPOCHS)
    _check_parameters(actual, oracle[0])
    _check_state(s, opt, params, oracle[1], oracle[2], OPT_EPOCHS)
    var reference = _sequential(ex, fixture_id)
    var changed = False
    for i in range(OPT_PARAMS):
        require(bitcast[DType.uint32](actual[i]) == bitcast[DType.uint32](reference[i]), "session/model_fit orchestration words")
        changed = changed or bitcast[DType.uint32](actual[i]) != bitcast[DType.uint32](data[2][i])
    require(changed, "optimizer did not update parameters")


@fieldwise_init
struct SavedParameter(Copyable, Movable):
    var weights: List[Float32]
    var first: List[Float32]
    var second: List[Float32]
    var step: Int
    var pw1: Float32
    var pw2: Float32
    var mu: Float32


def _resume[E: Exec & Movable & Deinitable](mut s: OpsSession[E], mut restored: OpsSession[E], fixture_id: Int) raises:
    var data = optimizer_data()
    var x = _upload(s, data[0], OPT_N, OPT_D)
    var y = _upload(s, data[1], OPT_N, 1)
    var baseline = _parameters(s, data[2])
    var baseline_opt = s.optimizer(optimizer_ints(fixture_id), optimizer_floats(fixture_id))
    _steps(s, baseline_opt, baseline, x, y, 5, False)
    var straight = _flat(s, baseline)
    var params = _parameters(s, data[2])
    var opt = s.optimizer(optimizer_ints(fixture_id), optimizer_floats(fixture_id))
    _steps(s, opt, params, x, y, 3, False)
    var offsets: List[Int] = [0, 15, 20, 25, 26]
    var rows: List[Int] = [OPT_D, 1, OPT_H, 1]
    var cols: List[Int] = [OPT_H, OPT_H, 1, 1]
    var saved = List[SavedParameter]()
    for parameter in range(4):  # driver-loop: over checkpoint parameter records
        var snapshot = s.optimizer_state(opt, params[parameter])
        require(snapshot[0] == 3 and snapshot[6] == 0, "checkpoint state extent/count")
        var count = offsets[parameter + 1] - offsets[parameter]
        var w = _download(s, params[parameter], count)
        var first = List[Float32]()
        var second = List[Float32]()
        if snapshot[4] != 0:
            first = _download(s, snapshot[4], count)
        if snapshot[5] != 0:
            second = _download(s, snapshot[5], count)
        saved.append(SavedParameter(w^, first^, second^, snapshot[0], snapshot[1], snapshot[2], snapshot[3]))
    # Drop ephemeral snapshot tensors before restoration. SavedParameter owns
    # native copies, so reset cannot invalidate the checkpoint storage.
    s.reset()
    var resumed_params = List[Int]()
    var checkpoint_first = List[Int]()
    var checkpoint_second = List[Int]()
    var resumed_opt = restored.optimizer(optimizer_ints(fixture_id), optimizer_floats(fixture_id))
    for parameter in range(4):  # driver-loop: over checkpoint parameter records
        var h = _upload(restored, saved[parameter].weights, rows[parameter], cols[parameter], True)
        resumed_params.append(h)
        var first = _upload(restored, saved[parameter].first, rows[parameter], cols[parameter]) if len(saved[parameter].first) else 0
        var second = _upload(restored, saved[parameter].second, rows[parameter], cols[parameter]) if len(saved[parameter].second) else 0
        checkpoint_first.append(first)
        checkpoint_second.append(second)
        restored.load_optimizer_state(resumed_opt, h, [saved[parameter].step, first, second, 0],
            [saved[parameter].pw1, saved[parameter].pw2, saved[parameter].mu])
    var rx = _upload(restored, data[0], OPT_N, OPT_D)
    var ry = _upload(restored, data[1], OPT_N, 1)
    _steps(restored, resumed_opt, resumed_params, rx, ry, 2, False)
    var actual = _flat(restored, resumed_params)
    var oracle = optimizer_oracle(fixture_id, 5)
    _check_parameters(actual, oracle[0])
    _check_state(restored, resumed_opt, resumed_params, oracle[1], oracle[2], 5)
    for i in range(OPT_PARAMS):
        require(bitcast[DType.uint32](actual[i]) == bitcast[DType.uint32](straight[i]), "native checkpoint continuation differs")
    # The uploaded checkpoint tensors must remain unchanged while restored
    # optimizer state advances: load_optimizer_state owns its own copies.
    for parameter in range(4):  # driver-loop: over saved checkpoint tensors
        var count = offsets[parameter + 1] - offsets[parameter]
        if checkpoint_first[parameter] != 0:
            var original = _download(restored, checkpoint_first[parameter], count)
            for j in range(count):
                require(bitcast[DType.uint32](original[j]) == bitcast[DType.uint32](saved[parameter].first[j]), "resume aliased first checkpoint buffer")
        if checkpoint_second[parameter] != 0:
            var original = _download(restored, checkpoint_second[parameter], count)
            for j in range(count):
                require(bitcast[DType.uint32](original[j]) == bitcast[DType.uint32](saved[parameter].second[j]), "resume aliased second checkpoint buffer")
        require(saved[parameter].step == 3, "resume mutated saved step")
    _ = saved


def _schedule[E: Exec & Movable & Deinitable](mut s: OpsSession[E]) raises:
    var fs = optimizer_floats(0)
    fs[0] = 0.1
    var opt = s.optimizer(optimizer_ints(0), fs)
    var expected: List[Float32] = [0.1, 0.1, 0.05, 0.05, 0.025]
    for epoch in range(5):  # dependent scheduler state progression
        require(bitcast[DType.uint32](s.learning_rate(opt)) == bitcast[DType.uint32](expected[epoch]), "literal step schedule rate")
        s.scheduler_step(opt)


def _zero_grad[E: Exec & Movable & Deinitable](mut s: OpsSession[E]) raises:
    var data = optimizer_data()
    var w = _upload(s, data[2], OPT_D, OPT_H, True)
    var x = _upload(s, data[0], OPT_N, OPT_D)
    var targets = List[Float32](length=OPT_N * OPT_H, fill=0)
    var targetp = targets.unsafe_ptr()
    var yp = data[1].unsafe_ptr()
    def target_task(row: Int) {imm targetp, imm yp}:
        for col in range(OPT_H):
            targetp.unsafe_store(row * OPT_H + col, yp.unsafe_load(row))
    host_parallelize(target_task, OPT_N)
    var y = _upload(s, targets, OPT_N, OPT_H)
    var product = s.run_op(OPS_MATMUL, [x, w], [], [])
    s.backward(s.run_op(OPS_MSE, [product, y], [REDUCTION_MEAN], []), False)
    var plain = List[Float32](length=OPT_D * OPT_H, fill=0)
    s.gradient_into(w, f32_ptr(Int(plain.unsafe_ptr())))
    s.zero_grad()
    var again = List[Float32](length=OPT_D * OPT_H, fill=0)
    var refused = False
    try:
        s.gradient_into(w, f32_ptr(Int(again.unsafe_ptr())))
    except:
        refused = True
    require(refused, "zero_grad retained a gradient")
    product = s.run_op(OPS_MATMUL, [x, w], [], [])
    s.backward(s.run_op(OPS_MSE, [product, y], [REDUCTION_MEAN], []), True)
    s.gradient_into(w, f32_ptr(Int(again.unsafe_ptr())))
    var expected_values = List[Float64](length=OPT_D * OPT_H, fill=0)
    var ep = expected_values.unsafe_ptr()
    var xp = data[0].unsafe_ptr()
    var pp = data[2].unsafe_ptr()
    def gradient_task(i: Int) {imm ep, imm xp, imm yp, imm pp}:
        var column = i // OPT_H
        var h = i % OPT_H
        var expected = Float64(0)
        for row in range(OPT_N):  # one analytic parameter-gradient reduction
            var prediction = Float64(0)
            for col in range(OPT_D):
                prediction += Float64(xp.unsafe_load(row * OPT_D + col)) * Float64(pp.unsafe_load(col * OPT_H + h))
            expected += Float64(xp.unsafe_load(row * OPT_D + column)) * (prediction - Float64(yp.unsafe_load(row)))
        ep.unsafe_store(i, expected * (2.0 / Float64(OPT_N * OPT_H)))
    host_parallelize(gradient_task, OPT_D * OPT_H)
    for i in range(OPT_D * OPT_H):
        require(bitcast[DType.uint32](plain[i]) == bitcast[DType.uint32](again[i]), "zero_grad plus accumulate changed gradient words")
        require(abs(Float64(plain[i]) - expected_values[i]) <= 1e-7 + 1e-5 * abs(expected_values[i]), "independent accumulated MSE gradient")
    _ = targets
    _ = data
    _ = plain
    _ = again


def _refusals[E: Exec & Movable & Deinitable](mut s: OpsSession[E], mut unrecorded: OpsSession[E]) raises:
    var ones: List[Float32] = [1, 1, 1, 1]
    var w = _upload(s, ones, 2, 2, True)
    var c = _upload(s, ones, 2, 2)
    var opt = s.optimizer(optimizer_ints(2), optimizer_floats(2))
    var refused = False
    try:
        s.step(opt, [w])
    except:
        refused = True
    require(refused, "step without backward accepted")
    var product = s.run_op(OPS_MATMUL, [c, w], [], [])
    s.backward(s.run_op(OPS_SUM, [product], [], []), False)
    for fixture_id in range(3):  # dependent refusal attempts on unchanged optimizer state
        refused = False
        try:
            if fixture_id == 0:
                s.step(opt, [c])
            elif fixture_id == 1:
                s.step(opt, [w, w])
            else:
                s.step(opt + 1, [w])
        except:
            refused = True
        require(refused and s.optimizer_state(opt, w)[0] == 0, "invalid step advanced optimizer state")
    var before = _download(s, w, 4)
    for value in before:
        require(value == 1, "refused step changed parameter")
    s.step(opt, [w])
    require(s.optimizer_state(opt, w)[0] == 1, "successful step count")
    refused = False
    try:
        unrecorded.zero_grad()
    except:
        refused = True
    require(refused, "non-recording session accepted zero_grad")


def run_ops_optimizer_fixture[E: Exec & Movable & Deinitable](mut s: OpsSession[E],
    mut restored: OpsSession[E], mut model_ex: E, fixture_id: Int) raises:
    if fixture_id < 5:
        _training(s, model_ex, fixture_id)
    elif fixture_id < 7:
        _resume(s, restored, fixture_id)
    elif fixture_id == 7:
        _schedule(s)
    elif fixture_id == 8:
        _zero_grad(s)
    else:
        _refusals(s, restored)
