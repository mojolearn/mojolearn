# SPDX-License-Identifier: Apache-2.0
"""Independent Float64 MLP gradients and SGD/Adam/AdamW state equations.

Only operation ids are imported from the product. No product loss, gradient,
optimizer, schedule or numerical helper computes an expected answer. Native
dyadic fixtures keep both active and inactive ReLU units away from the kink.
Rows and parameter cells are independent host-pool units; epochs depend on
the preceding parameter/moment state and therefore remain ordered.
"""
from std.math import sqrt
from core.host_parallel import host_parallelize
from sequence.ops import OPT_SGD, OPT_ADAM, OPT_ADAMW
from autograd.model import SCHED_CONSTANT, SCHED_STEP, SCHED_EXPONENTIAL

comptime OPT_N = 24
comptime OPT_D = 3
comptime OPT_H = 5
comptime OPT_PARAMS = 26
comptime OPT_EPOCHS = 6


def optimizer_ints(fixture_id: Int) -> List[Int]:
    var result: List[Int] = [OPT_ADAM, 0, SCHED_CONSTANT, 1]
    if fixture_id == 0:
        result = [OPT_SGD, 0, SCHED_STEP, 2]
    elif fixture_id == 1:
        result = [OPT_SGD, 1, SCHED_CONSTANT, 1]
    elif fixture_id == 3:
        result = [OPT_ADAM, 0, SCHED_EXPONENTIAL, 1]
    elif fixture_id == 4:
        result = [OPT_ADAMW, 0, SCHED_CONSTANT, 1]
    elif fixture_id == 6:
        result = [OPT_SGD, 0, SCHED_CONSTANT, 1]
    return result^


def optimizer_floats(fixture_id: Int) -> List[Float32]:
    # lr, momentum, beta1, beta2, eps, weight_decay, gamma
    var result: List[Float32] = [0.01, 0, 0.9, 0.999, 1e-8, 0, 1]
    if fixture_id == 0:
        result[0] = 0.05
        result[6] = 0.5
    elif fixture_id == 1 or fixture_id == 6:
        result[0] = 0.05
        result[1] = 0.9
        result[5] = Float32(0.01) if fixture_id == 1 else Float32(0)
    elif fixture_id == 3:
        result[0] = 0.02
        result[5] = 0.001
        result[6] = 0.9
    elif fixture_id == 4:
        result[2] = 0.8
        result[3] = 0.99
        result[5] = 0.05
    return result^


def optimizer_data() -> Tuple[List[Float32], List[Float32], List[Float32]]:
    var x = List[Float32](length=OPT_N * OPT_D, fill=0)
    var y = List[Float32](length=OPT_N, fill=0)
    var xp = x.unsafe_ptr()
    var yp = y.unsafe_ptr()
    def row_task(row: Int) {imm xp, imm yp}:
        for col in range(OPT_D):
            xp.unsafe_store(row * OPT_D + col, Float32((row * 5 + col * 3) % 17 - 8) * 0.03125)
        yp.unsafe_store(row, Float32((row * 7) % 11 - 5) * 0.0625)
    host_parallelize(row_task, OPT_N)
    var params = List[Float32](length=OPT_PARAMS, fill=0)
    for i in range(OPT_D * OPT_H):
        params[i] = Float32((i * 7) % 13 - 6) * 0.015625
    var biases: List[Float32] = [-0.5, 0.5, -0.5, 0.5, 0.25]
    var weights: List[Float32] = [0.125, -0.125, 0.25, -0.25, 0.0625]
    for j in range(OPT_H):
        params[15 + j] = biases[j]
        params[20 + j] = weights[j]
    params[25] = 0.125
    return (x^, y^, params^)


def oracle_gradients(x: List[Float32], y: List[Float32], params: List[Float64]) -> List[Float64]:
    var partial = List[Float64](length=OPT_N * OPT_PARAMS, fill=0)
    var xp = x.unsafe_ptr()
    var yp = y.unsafe_ptr()
    var pp = params.unsafe_ptr()
    var gp = partial.unsafe_ptr()
    def row_task(row: Int) {imm xp, imm yp, imm pp, imm gp}:
        var hidden = List[Float64](length=OPT_H, fill=0)
        var prediction = pp.unsafe_load(25)
        for h in range(OPT_H):
            var value = pp.unsafe_load(15 + h)
            for col in range(OPT_D):
                value += Float64(xp.unsafe_load(row * OPT_D + col)) * pp.unsafe_load(col * OPT_H + h)
            hidden[h] = max(Float64(0), value)
            prediction += hidden[h] * pp.unsafe_load(20 + h)
        var delta = 2.0 * (prediction - Float64(yp.unsafe_load(row))) / Float64(OPT_N)
        gp.unsafe_store(row * OPT_PARAMS + 25, delta)
        for h in range(OPT_H):
            gp.unsafe_store(row * OPT_PARAMS + 20 + h, delta * hidden[h])
            var hidden_delta = delta * pp.unsafe_load(20 + h) if hidden[h] > 0 else Float64(0)
            gp.unsafe_store(row * OPT_PARAMS + 15 + h, hidden_delta)
            for col in range(OPT_D):
                gp.unsafe_store(row * OPT_PARAMS + col * OPT_H + h, Float64(xp.unsafe_load(row * OPT_D + col)) * hidden_delta)
    host_parallelize(row_task, OPT_N)
    var result = List[Float64](length=OPT_PARAMS, fill=0)
    var rp = result.unsafe_ptr()
    def parameter_task(parameter: Int) {imm gp, imm rp}:
        var value = Float64(0)
        for row in range(OPT_N):
            value += gp.unsafe_load(row * OPT_PARAMS + parameter)
        rp.unsafe_store(parameter, value)
    host_parallelize(parameter_task, OPT_PARAMS)
    _ = partial
    _ = x
    _ = y
    _ = params
    return result^


def _power(base: Float64, exponent: Int) -> Float64:
    var result = Float64(1)
    for i in range(exponent):
        result *= base
    return result


def optimizer_oracle(fixture_id: Int, epochs: Int) -> Tuple[List[Float64], List[Float64], List[Float64]]:
    var data = optimizer_data()
    var params = List[Float64](length=OPT_PARAMS, fill=0)
    for i in range(OPT_PARAMS):
        params[i] = Float64(data[2][i])
    var moment = List[Float64](length=OPT_PARAMS, fill=0)
    var second = List[Float64](length=OPT_PARAMS, fill=0)
    var ints = optimizer_ints(fixture_id)
    var fs = optimizer_floats(fixture_id)
    var kind = ints[0]
    var nesterov = ints[1] != 0
    var momentum = Float64(fs[1])
    var beta1 = Float64(fs[2])
    var beta2 = Float64(fs[3])
    var eps = Float64(fs[4])
    var decay = Float64(fs[5])
    var pp = params.unsafe_ptr()
    var mp = moment.unsafe_ptr()
    var vp = second.unsafe_ptr()
    for epoch in range(epochs):  # dependent optimizer state recurrence
        var gradients = oracle_gradients(data[0], data[1], params)
        var gp = gradients.unsafe_ptr()
        var exponent = 0 if ints[2] == SCHED_CONSTANT else (epoch // ints[3] if ints[2] == SCHED_STEP else epoch)
        var lr = Float64(fs[0]) * _power(Float64(fs[6]), exponent)
        var bc1 = 1.0 - _power(beta1, epoch + 1)
        var bc2 = 1.0 - _power(beta2, epoch + 1)
        def update(parameter: Int) {imm pp, imm mp, imm vp, imm gp, imm kind, imm nesterov, imm momentum, imm beta1, imm beta2, imm eps, imm decay, imm lr, imm bc1, imm bc2, imm epoch}:
            var p = pp.unsafe_load(parameter)
            var g = gp.unsafe_load(parameter)
            if kind == OPT_SGD:
                g += decay * p
                if momentum != 0:
                    var m = g if epoch == 0 else momentum * mp.unsafe_load(parameter) + g
                    mp.unsafe_store(parameter, m)
                    g = g + momentum * m if nesterov else m
                p -= lr * g
            else:
                if kind == OPT_ADAMW:
                    p *= 1.0 - lr * decay
                else:
                    g += decay * p
                var m = beta1 * mp.unsafe_load(parameter) + (1.0 - beta1) * g
                var v = beta2 * vp.unsafe_load(parameter) + (1.0 - beta2) * g * g
                mp.unsafe_store(parameter, m)
                vp.unsafe_store(parameter, v)
                p -= lr * (m / bc1) / (sqrt(v / bc2) + eps)
            pp.unsafe_store(parameter, p)
        host_parallelize(update, OPT_PARAMS)
        _ = gradients
    _ = data
    return (params^, moment^, second^)
