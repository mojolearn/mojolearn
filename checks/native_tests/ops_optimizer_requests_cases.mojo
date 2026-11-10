# SPDX-License-Identifier: Apache-2.0
"""Typed optimizer request/state contracts, independent constant-gradient math.

Depends on producer e0af392644. Previous numerical fixtures keep their packed
native interfaces; this separate batch exercises the canonical typed path.
"""
from std.memory import bitcast
from autograd.ops_api import OpsSession, OPS_MATMUL, OPS_SUM
from autograd.optimizer_requests import (
    OpsOptimizerRequest, OpsOptimizerRestoreRequest, OpsOptimizerStateValue,
    ops_optimizer_codes, ops_optimizer_request, ops_optimizer_restore_request,
)
from core.cast_in import CastIn
from core.hostptr import f32_ptr
from sequence.exec_trait import Exec

comptime OPS_REQUEST_FIXTURES = 17
comptime OPS_REQUEST_MATH_FIXTURES = 3


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def upload[E: Exec & Movable & Deinitable](mut s: OpsSession[E], values: List[Float32], grad: Bool = False) raises -> Int:
    var h = s.upload(CastIn.f32_c(Int(values.unsafe_ptr()), 2, 2), grad)
    _ = values
    return h


def download[E: Exec & Movable & Deinitable](mut s: OpsSession[E], h: Int) raises -> List[Float32]:
    var values = List[Float32](length=4, fill=0)
    s.download(h, f32_ptr(Int(values.unsafe_ptr())))
    return values^


def one_step[E: Exec & Movable & Deinitable](mut s: OpsSession[E], opt: Int, param: Int, identity: Int) raises:
    # sum(I @ parameter) has derivative one in every parameter cell.
    var product = s.run_op(OPS_MATMUL, [identity, param], [], [])
    s.backward(s.run_op(OPS_SUM, [product], [-1], []), False)
    s.step(opt, [param])


def check_state[E: Exec & Movable & Deinitable](mut s: OpsSession[E], state: OpsOptimizerStateValue, kind: Int, step: Int) raises:
    require(state.kind == kind and state.step == step, "typed state kind/step")
    require(state.first != 0 and state.third == 0, "typed state buffer fields")
    require(state.mu_product == 1, "typed unused mu product")
    var first = download(s, state.first)
    var moment = (Float32(1) if step == 1 else Float32(1.25)) if kind == 0 else (Float32(0.5) if step == 1 else Float32(0.75))
    for value in first:
        require(value == moment, "literal typed first-moment state")
    var power = Float32(1) if kind == 0 else (Float32(0.5) if step == 1 else Float32(0.25))
    require(state.beta1_power == power and state.beta2_power == power, "literal typed bias powers")
    if kind == 0:
        require(state.second == 0, "SGD unexpectedly owns a second moment")
    else:
        require(state.second != 0, "Adam typed second-moment handle")
        var second = download(s, state.second)
        for value in second:
            require(value == moment, "literal typed second-moment state")


def check_parameters(actual: List[Float32], kind: Int) raises:
    var original: List[Float64] = [1, -2, 0.5, -0.25]
    for index in range(4):
        # SGD: first buffer1, then .5*1 + (1-.25)*1 =1.25.
        # Adam: corrected m=v=1, eps=.25 => each update .125/1.25=.1.
        # AdamW additionally multiplies p by 1-.125*.25 before each update.
        var expected = original[index] - 0.28125
        if kind == 1:
            expected = original[index] - 0.2
        elif kind == 2:
            expected = (original[index] * 0.96875 - 0.1) * 0.96875 - 0.1
        if kind == 0:
            require(bitcast[DType.uint32](actual[index]) == bitcast[DType.uint32](Float32(expected)), "exact dampened SGD result")
        else:
            require(abs(Float64(actual[index]) - expected) <= 1e-6, "closed-form typed Adam/AdamW result")


def run_typed_optimizer[E: Exec & Movable & Deinitable](mut s: OpsSession[E], mut restored: OpsSession[E], kind: Int) raises:
    var original: List[Float32] = [1, -2, 0.5, -0.25]
    var identity_values: List[Float32] = [1, 0, 0, 1]
    var request = OpsOptimizerRequest(kind, False, 0, 1, 0.125,
        Float32(0.5) if kind == 0 else Float32(0), 0.5, 0.5, 0.25,
        Float32(0.25) if kind == 2 else Float32(0), 1,
        Float32(0.25) if kind == 0 else Float32(0))
    var param = upload(s, original, True)
    var identity = upload(s, identity_values)
    var opt = s.optimizer(request)
    one_step(s, opt, param, identity)
    var snapshot = s.optimizer_state_value(opt, param)
    check_state(s, snapshot, kind, 1)
    var weights = download(s, param)
    var first = download(s, snapshot.first)
    var second = List[Float32](length=4, fill=0)
    if kind != 0:
        second = download(s, snapshot.second)
    s.reset()
    one_step(s, opt, param, identity)
    var straight = download(s, param)
    check_parameters(straight, kind)
    var current = s.optimizer_state_value(opt, param)
    check_state(s, current, kind, 2)

    var rp = upload(restored, weights, True)
    var ri = upload(restored, identity_values)
    var ro = restored.optimizer(request)
    var h1 = upload(restored, first)
    var h2 = upload(restored, second) if kind != 0 else 0
    var restore = OpsOptimizerRestoreRequest(snapshot.step, h1, h2, 0,
        snapshot.beta1_power, snapshot.beta2_power, snapshot.mu_product)
    restored.load_optimizer_state(ro, rp, restore)
    current = restored.optimizer_state_value(ro, rp)
    check_state(restored, current, kind, 1)
    one_step(restored, ro, rp, ri)
    check_parameters(download(restored, rp), kind)
    current = restored.optimizer_state_value(ro, rp)
    check_state(restored, current, kind, 2)
    var source = download(restored, h1)
    for index in range(4):
        require(source[index] == first[index], "typed restore aliased checkpoint first buffer")
    if kind != 0:
        source = download(restored, h2)
        for index in range(4):
            require(source[index] == second[index], "typed restore aliased checkpoint second buffer")
    require(restore.step == 1, "typed restore request mutated")


def run_request_metadata(fixture_id: Int) raises:
    if fixture_id < 6:
        var names: List[String] = ["sgd", "adam", "adamw"]
        var schedules: List[String] = ["step", "constant", "exponential"]
        var expected: List[Int] = [1, 0, 2]
        var index = fixture_id - 3
        var codes = ops_optimizer_codes(names[index], schedules[index])
        require(codes[0] == index and codes[1] == expected[index], "typed optimizer name codes")
    elif fixture_id < 8:
        var refused = False
        try:
            _ = ops_optimizer_codes("lion" if fixture_id == 6 else "adam", "constant" if fixture_id == 6 else "callback")
        except:
            refused = True
        require(refused, "unknown optimizer/schedule code accepted")
    elif fixture_id < 10:
        var ints: List[Int] = [2, 7, 1, 3]
        var values: List[Float32] = [0.125, 0.5, 0.75, 0.875, 0.25, 0.0625, 0.5]
        if fixture_id == 9:
            values.append(0.25)
        var request = ops_optimizer_request(ints, values)
        require(request.kind == 2 and request.nesterov and request.schedule == 1 and request.step_size == 3, "packed typed integer fields")
        require(request.lr == 0.125 and request.momentum == 0.5 and request.beta1 == 0.75 and request.beta2 == 0.875, "packed typed optimizer fields")
        require(request.eps == 0.25 and request.weight_decay == 0.0625 and request.gamma == 0.5, "packed typed tail fields")
        require(request.dampening == (Float32(0.25) if fixture_id == 9 else Float32(0)), "packed optional dampening")
    elif fixture_id < 13:
        var ints = List[Int](length=3 if fixture_id == 10 else 4, fill=0)
        var values = List[Float32](length=6 if fixture_id == 11 else (9 if fixture_id == 12 else 7), fill=0)
        var refused = False
        try:
            _ = ops_optimizer_request(ints, values)
        except:
            refused = True
        require(refused, "packed config length accepted")
    elif fixture_id == 13:
        var restore = ops_optimizer_restore_request([7, 11, 13, 17], [0.5, 0.25, 0.125])
        require(restore.step == 7 and restore.first == 11 and restore.second == 13 and restore.third == 17, "packed restore integer fields")
        require(restore.beta1_power == 0.5 and restore.beta2_power == 0.25 and restore.mu_product == 0.125, "packed restore scalar fields")
    else:
        var ints = List[Int](length=3 if fixture_id == 14 else 4, fill=0)
        var values = List[Float32](length=2 if fixture_id == 15 else (4 if fixture_id == 16 else 3), fill=0)
        var refused = False
        try:
            _ = ops_optimizer_restore_request(ints, values)
        except:
            refused = True
        require(refused, "packed restore length accepted")
