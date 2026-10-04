# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Opt-in parallel PT score search, using float-float centered moments.

The NLL derivative is n * Cov(y, dy/dlambda) / Var(y) - sum(J).
Unlike differences of rounded NLL values, its zero is first order in the
lambda error. All observations and reductions stay on the GPU.
"""
from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from x_linear.ff import FF, ff_of, ff_add, ff_add_f, ff_sub, ff_mul, ff_mul_f, ff_div, ff_f32
from x_prep.common import FP, IP, p, is_nan
from x_prep.prims import expf
from x_prep.transform import power_log, PT_STATE

comptime SCORE_WORDS = 12
comptime SCORE_TGR = 256


@always_inline
def score_power(lg: Float32, nonneg: Bool, lam: Float32, method: Int) -> Tuple[Float32, Float32]:
    var positive = method == 1 or nonneg
    var a = lam if positive else Float32(2) - lam
    var z = a * lg
    var y = Float32(0)
    var dy = Float32(0)
    if abs(z) < Float32(0.5):
        # expm1(z)/z and its derivative: avoid cancellation at lambda 0/2.
        var term = Float32(1)
        var ratio = Float32(1)
        var deriv = Float32(0.5)
        var dt = Float32(0.5)
        for k in range(1, 12):
            term *= z / Float32(k + 1)
            ratio += term
            dt *= z * Float32(k + 1) / (Float32(k) * Float32(k + 2))
            deriv += dt
        y = lg * ratio
        dy = lg * lg * deriv
    else:
        var e = expf(z)
        y = (e - Float32(1)) / a
        dy = ((z - Float32(1)) * e + Float32(1)) / (a * a)
    return (y if positive else -y, dy)


@always_inline
def score_merge(mut a: InlineArray[FF, 6], b: InlineArray[FF, 6]):
    if b[0].hi == Float32(0):
        return
    if a[0].hi == Float32(0):
        a = b
        return
    var n = ff_add(a[0], b[0])
    var delta = ff_sub(b[2], a[2])
    var dd = ff_sub(b[3], a[3])
    var ratio = ff_div(b[0], n)
    var weight = ff_mul(a[0], ratio)
    a[4] = ff_add(ff_add(a[4], b[4]), ff_mul(ff_mul(delta, delta), weight))
    a[5] = ff_add(ff_add(a[5], b[5]), ff_mul(ff_mul(delta, dd), weight))
    a[2] = ff_add(a[2], ff_mul(delta, ratio))
    a[3] = ff_add(a[3], ff_mul(dd, ratio))
    a[1] = ff_add(a[1], b[1])
    a[0] = n


def score_tile(f: FP, pp: FP, q: IP, rows: Int32, tpb: Int32):
    var c = Int(block_idx.y) * Int(tpb) + Int(thread_idx.x)
    var d = p(q, 2)
    if c >= d:
        return
    var S = p(q, 6) + c * PT_STATE
    if f[S + 7] != Float32(0):
        return
    var chunk = Int(block_idx.x)
    var a = InlineArray[FF, 6](fill=ff_of(Float32(0)))
    var lam = f[p(q, 7) + c]
    for i in range(chunk * Int(rows), min((chunk + 1) * Int(rows), p(q, 1))):
        var x = f[p(q, 0) + i * d + c]
        if is_nan(x):
            continue
        var lg = power_log(x, p(q, 3))
        var yd = score_power(lg, x >= Float32(0), lam, p(q, 3))
        var y = ff_of(yd[0])
        var dy = ff_of(yd[1])
        a[0] = ff_add_f(a[0], Float32(1))
        var delta = ff_sub(y, a[2])
        var dd = ff_sub(dy, a[3])
        a[2] = ff_add(a[2], ff_div(delta, a[0]))
        a[3] = ff_add(a[3], ff_div(dd, a[0]))
        a[4] = ff_add(a[4], ff_mul(delta, ff_sub(y, a[2])))
        a[5] = ff_add(a[5], ff_mul(delta, ff_sub(dy, a[3])))
        a[1] = ff_add_f(a[1], lg if p(q, 3) == 1 or x >= Float32(0) else -lg)
    var o = (chunk * d + c) * SCORE_WORDS
    for k in range(6):
        pp[o + 2 * k] = a[k].hi
        pp[o + 2 * k + 1] = a[k].lo


def score_finish(f: FP, pp: FP, q: IP, chunks: Int32):
    var c = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var S = p(q, 6) + c * PT_STATE
    if f[S + 7] != Float32(0):
        return
    var sh = stack_allocation[SCORE_TGR * SCORE_WORDS, Float32, address_space=AddressSpace.SHARED]()
    var a = InlineArray[FF, 6](fill=ff_of(Float32(0)))
    for j in range(tid, Int(chunks), SCORE_TGR):
        var b = InlineArray[FF, 6](fill=ff_of(Float32(0)))
        var o = (j * p(q, 2) + c) * SCORE_WORDS
        for k in range(6):
            b[k] = FF(pp[o + 2 * k], pp[o + 2 * k + 1])
        score_merge(a, b)
    for k in range(6):
        sh[tid * SCORE_WORDS + 2 * k] = a[k].hi
        sh[tid * SCORE_WORDS + 2 * k + 1] = a[k].lo
    barrier()
    var w = SCORE_TGR // 2
    while w > 0:
        if tid < w:
            var b = InlineArray[FF, 6](fill=ff_of(Float32(0)))
            for k in range(6):
                var o = (tid + w) * SCORE_WORDS + 2 * k
                b[k] = FF(sh[o], sh[o + 1])
            score_merge(a, b)
            for k in range(6):
                sh[tid * SCORE_WORDS + 2 * k] = a[k].hi
                sh[tid * SCORE_WORDS + 2 * k + 1] = a[k].lo
        barrier()
        w //= 2
    if tid == 0:
        var lam = f[p(q, 7) + c]
        # No division by variance: sign of n*cov - J*M2 is the score sign.
        var score = ff_sub(ff_mul(a[0], a[5]), ff_mul(a[1], a[4]))
        if ff_f32(score) > Float32(0):
            f[S + 1] = lam
        elif ff_f32(score) < Float32(0):
            f[S] = lam
        elif ff_f32(score) == Float32(0):
            f[S] = lam
            f[S + 1] = lam
        # NaN score leaves the bracket unchanged; quality gate must reject
        # nonfinite outputs/poor objective on overflow-stress fixtures.
        var next_lam = (f[S] + f[S + 1]) * Float32(0.5)
        f[p(q, 7) + c] = next_lam
        f[p(q, 8) + c] = next_lam
