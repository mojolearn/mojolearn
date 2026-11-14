# SPDX-License-Identifier: Apache-2.0
"""MoEBlock forward against a float64 restatement of Mixtral's sparse block,
and the routing tie rule (ported from
python/mojolearn/tests/test_x_sequence_moe.py `test_forward_and_ties`).

Fixture 0: 20 tokens of width 8, 4 experts of intermediate 12, top 2,
renormalised, the router scaled by 30 so the routing is sharp. The output
of `sequence/ops_impl.mojo::moe_forward_impl` must equal, within atol 2e-5,
sum over the top-2 experts (softmax order, ties to the lower index) of
w_e * down_e (silu(g) * u), [g; u] = gate_up_e x, w the renormalised
probabilities, all in float64. Fixture 1: 3 experts, top 1, a zero router
(every probability ties): every token selects expert 0.

Inputs and weights are hashed (the original drew them from NumPy and the
block's seeded constructor; its reference never depended on either).
"""
from std.math import exp, sqrt
from core.host_parallel import host_parallelize
from sequence.exec import Exec
from sequence.ops_impl import moe_forward_impl
from checks.native_tests.sequence_port_common import require, allclose, u01, normal

comptime MOE_FIXTURES = 2
comptime MOE_T = 20
comptime MOE_D = 8


def hashed_uniform(count: Int, salt: Int, bound: Float64) -> List[Float32]:
    var values = List[Float32](length=count, fill=0.0)
    var vp = values.unsafe_ptr()
    def cell(i: Int) {imm vp, imm salt, imm bound}:
        vp.unsafe_store(i, Float32((2.0 * u01(salt, i) - 1.0) * bound))
    host_parallelize(cell, count)
    return values^


def hashed_normal(count: Int, salt: Int) -> List[Float32]:
    var values = List[Float32](length=count, fill=0.0)
    var vp = values.unsafe_ptr()
    def cell(i: Int) {imm vp, imm salt}:
        vp.unsafe_store(i, Float32(normal(salt, i)))
    host_parallelize(cell, count)
    return values^


def silu(v: Float64) -> Float64:
    return v / (1.0 + exp(-v))


def run_moe_fixture[E: Exec](mut ex: E, unit: Int) raises:
    var F = 12 if unit == 0 else 4
    var En = 4 if unit == 0 else 3
    var k = 2 if unit == 0 else 1
    var x = hashed_normal(MOE_T * MOE_D, 0x30E1)
    # fixture 0: the router scaled by 30 (sharp routing); fixture 1: the zero
    # router, every probability tied
    var router = hashed_uniform(En * MOE_D, 0x30E2, 30.0 / sqrt(Float64(MOE_D))) if unit == 0 else List[Float32](
        length=En * MOE_D, fill=0.0)
    var gate_up = hashed_uniform(En * 2 * F * MOE_D, 0x30E3, 1.0 / sqrt(Float64(MOE_D)))
    var down = hashed_uniform(En * MOE_D * F, 0x30E4, 1.0 / sqrt(Float64(F)))
    var y = List[Float32](length=MOE_T * MOE_D, fill=0.0)
    var logits = List[Float32](length=MOE_T * En, fill=0.0)
    var selected = List[Float32](length=MOE_T * k, fill=-1.0)
    var weights = List[Float32](length=MOE_T * k, fill=0.0)
    var addresses: List[Int] = [Int(x.unsafe_ptr()), Int(router.unsafe_ptr()), Int(gate_up.unsafe_ptr()),
                                Int(down.unsafe_ptr()), Int(y.unsafe_ptr()), Int(logits.unsafe_ptr()),
                                Int(selected.unsafe_ptr()), Int(weights.unsafe_ptr())]
    _ = moe_forward_impl(ex, addresses, [MOE_T, MOE_D, F, En, k, 1])
    var failures = List[Int](length=MOE_T, fill=0)
    var fp = failures.unsafe_ptr()
    if unit == 1:
        var sp = selected.unsafe_ptr()
        def tie(t: Int) {imm fp, imm sp}:
            if sp[t] != 0.0:
                fp.unsafe_store(t, 1)
        host_parallelize(tie, MOE_T)
        for t in range(MOE_T):
            require(failures[t] == 0, "MoE tie: token " + String(t) + " did not select expert 0")
    else:
        var xp = x.unsafe_ptr()
        var rp = router.unsafe_ptr()
        var gp = gate_up.unsafe_ptr()
        var dp = down.unsafe_ptr()
        var yp = y.unsafe_ptr()
        def token(t: Int) {imm fp, imm xp, imm rp, imm gp, imm dp, imm yp, imm F, imm En}:
            var p = List[Float64](length=En, fill=0.0)
            var top = Float64(-1e300)
            for e in range(En):
                var z = 0.0
                for c in range(MOE_D):
                    z += Float64(xp[t * MOE_D + c]) * Float64(rp[e * MOE_D + c])
                p[e] = z
                top = max(top, z)
            var total = 0.0
            for e in range(En):
                p[e] = exp(p[e] - top)
                total += p[e]
            # the two largest, a stable descending order: ties to the lower index
            var first = 0
            for e in range(1, En):
                if p[e] > p[first]:
                    first = e
            var second = 1 if first == 0 else 0
            for e in range(En):
                if e != first and p[e] > p[second]:
                    second = e
            var norm = p[first] + p[second]
            var ref = List[Float64](length=MOE_D, fill=0.0)
            for pick in range(2):
                var e = first if pick == 0 else second
                var w = p[e] / norm
                var hidden = List[Float64](length=F, fill=0.0)
                for j in range(F):
                    var g = 0.0
                    var u = 0.0
                    for c in range(MOE_D):
                        g += Float64(gp[(e * 2 * F + j) * MOE_D + c]) * Float64(xp[t * MOE_D + c])
                        u += Float64(gp[(e * 2 * F + F + j) * MOE_D + c]) * Float64(xp[t * MOE_D + c])
                    hidden[j] = silu(g) * u
                for r in range(MOE_D):
                    var acc = 0.0
                    for j in range(F):
                        acc += Float64(dp[(e * MOE_D + r) * F + j]) * hidden[j]
                    ref[r] += w * acc
            for r in range(MOE_D):
                if not allclose(Float64(yp[t * MOE_D + r]), ref[r], 0.0, 2e-5):
                    fp.unsafe_store(t, 1)
        host_parallelize(token, MOE_T)
        for t in range(MOE_T):
            require(failures[t] == 0, "MoE forward: token " + String(t) + " differs from the float64 block (atol 2e-5)")
    _ = x^
    _ = router^
    _ = gate_up^
    _ = down^
    _ = y^
    _ = logits^
    _ = selected^
    _ = weights^
