# SPDX-License-Identifier: Apache-2.0
"""The Samba training surface's schedules and position-keyed random stream
(ported from python/mojolearn/tests/test_samba_surface.py: the numerical
halves of `arm_schedule` and `arm_rng`).

Schedules, through the typed construction the public schedules use
(`training/optimizer_schedule_plan.mojo::optimizer_warmup_schedule`) and
`sequence/lr_table.mojo::lr_spec_value` (host scalar code on every column):
- fixture 0, constant (peak 1e-3, warmup 4): lr(4) is float32(1e-3), lr(1)
  is the exact rational peak / 4 rounded once, lr(100) == lr(4), step 0 is
  refused (LR_ONE_BASED);
- fixture 1, linear (1e-3, warmup 2, total 10, min 1e-4): lr(10) is
  float32(1e-4) and lr(11) == lr(10);
- fixture 2, cosine (same): lr(6) is the exact midpoint (peak + min) / 2
  rounded once; t = 3..9 within 1e-6 relative of the float64 cos form;
  non-increasing over t = 2..11.

The stream, through the training host image's entry
(`core/mojolearn_training_host_impl.mojo::neural_rng_host_into`, the
Philox layout of `core/philox_neural.mojo`):
- fixture 3, uniform (seed 12345): 1000 draws in [0, 1); a second call gives
  the same bits; stream 1 differs; the 300 draws at offset 700 equal the
  whole stream's elements 700..;
- fixture 4, normal(0, 1) (seed 7): 20000 draws, all finite, mean and
  standard deviation within 0.03 of 0 and 1;
- fixture 5, dropout p = 0.25 (seed 99) of a hashed 4 x 256 x: keeps 65 to 85
  percent; kept cells (y != 0) are x * scale bit for bit; the backward's
  mask equals the forward's and its kept cells are dy * scale; a replay gives the same
  bits; rows 2..3 drawn at offset 512 equal the unsplit rows; p = 1 is
  refused.

The Generator's counter and state_dict, kaiming_uniform's bound, the
refusal messages, the schedule pins (reported, OWED) and the config round
trip stay in the Python gate with its stack arms.
"""
from std.math import cos, isfinite, sqrt
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.philox_neural import NRNG_UNIFORM, NRNG_NORMAL, NRNG_DROPOUT_FORWARD, NRNG_DROPOUT_BACKWARD
from core.mojolearn_training_host_impl import neural_rng_host_into
from sequence.lr_table import LrSpec, lr_spec_value, LR_ONE_BASED
from training.optimizer_schedule_plan import optimizer_warmup_schedule
from checks.native_tests.sequence_port_common import require, u01, same_bits, count_failures

comptime SAMBA_SURFACE_FIXTURES = 6


def lr(spec: LrSpec, t: Int) raises -> Float32:
    var result = lr_spec_value(spec, t)
    require(result.status == 0, "schedule step " + String(t) + " refused with status " + String(result.status))
    return result.value


def schedule_fixture(unit: Int) raises:
    var peak = Float64(Float32(1e-3))
    var low = Float64(Float32(1e-4))
    if unit == 0:
        var c = optimizer_warmup_schedule(0, 1e-3, 0.0, 4, 0, False)
        require(lr(c, 4) == Float32(1e-3), "constant: reaches peak at warmup end")
        require(lr(c, 1) == Float32(peak / 4.0), "constant: warmup step 1 is peak / 4 (exact, rounded once)")
        require(lr(c, 100) == lr(c, 4), "constant: flat after warmup")
        require(lr_spec_value(c, 0).status == LR_ONE_BASED, "constant: step 0 is not refused")
    elif unit == 1:
        var l = optimizer_warmup_schedule(1, 1e-3, 1e-4, 2, 10, True)
        require(lr(l, 10) == Float32(1e-4) and lr(l, 11) == lr(l, 10), "linear: reaches min_lr at total and holds")
    else:
        var s = optimizer_warmup_schedule(2, 1e-3, 1e-4, 2, 10, True)
        # peak / 2 + min / 2 is exact in binary64 (two float32s a few binades apart)
        require(lr(s, 6) == Float32(peak / 2.0 + low / 2.0), "cosine: p = 1/2 is the exact midpoint rounded once")
        var worst = 0.0
        for t in range(3, 10):  # seven steps of one schedule
            var p = Float64(t - 2) / 8.0
            var reference = 1e-4 + (1e-3 - 1e-4) * (1.0 + cos(3.141592653589793 * p)) / 2.0
            worst = max(worst, abs(Float64(lr(s, t)) - reference) / reference)
        require(worst < 1e-6, "cosine: worst relative distance " + String(worst) + " from the float64 cos form")
        for t in range(2, 11):
            require(lr(s, t) >= lr(s, t + 1), "cosine: increases after step " + String(t))


def draw(n: Int, offset: Int, seed: Int, stream: Int, kind: Int, a: Float32, b: Float32,
         input: List[Float32]) raises -> List[Float32]:
    var values = List[Float32](length=n, fill=-7.0)
    require(neural_rng_host_into(Int(values.unsafe_ptr()), Int(input.unsafe_ptr()), n, offset, seed, 0, stream, kind, a, b)
            == n, "neural rng count")
    return values^


def differing(a: List[Float32], b: List[Float32], start: Int, count: Int) -> Int:
    """Words of a[start:start + count] that differ from b[0:count]."""
    var flags = List[Int](length=count, fill=0)
    var fp = flags.unsafe_ptr()
    var ap = a.unsafe_ptr()
    var bp = b.unsafe_ptr()
    def word(i: Int) {imm fp, imm ap, imm bp, imm start}:
        if not same_bits(ap[start + i], bp[i]):
            fp.unsafe_store(i, 1)
    host_parallelize(word, count)
    return count_failures(flags)


def stream_fixture(unit: Int) raises:
    var none = List[Float32](length=1, fill=0.0)
    if unit == 3:
        var u = draw(1000, 0, 12345, 0, NRNG_UNIFORM, 0.0, 1.0, none)
        var bad = List[Int](length=1000, fill=0)
        var bp = bad.unsafe_ptr()
        var up = u.unsafe_ptr()
        def in_range(i: Int) {imm bp, imm up}:
            if not (up[i] >= 0.0 and up[i] < 1.0):
                bp.unsafe_store(i, 1)
        host_parallelize(in_range, 1000)
        require(count_failures(bad) == 0, "uniform draws outside [0, 1)")
        require(differing(draw(1000, 0, 12345, 0, NRNG_UNIFORM, 0.0, 1.0, none), u, 0, 1000) == 0,
                "uniform: the same seed and stream give other bits")
        require(differing(draw(1000, 0, 12345, 1, NRNG_UNIFORM, 0.0, 1.0, none), u, 0, 1000) > 0,
                "uniform: the next stream equals the first")
        require(differing(u, draw(300, 700, 12345, 0, NRNG_UNIFORM, 0.0, 1.0, none), 700, 300) == 0,
                "uniform: the slice at offset 700 differs from the whole stream")
        _ = u^
        _ = bad^
    elif unit == 4:
        var n = 20000
        var z = draw(n, 0, 7, 0, NRNG_NORMAL, 0.0, 1.0, none)
        var blocks = 20
        var sums = List[Float64](length=2 * blocks, fill=0.0)
        var finite = List[Int](length=blocks, fill=0)
        var sp = sums.unsafe_ptr()
        var fp = finite.unsafe_ptr()
        var zp = z.unsafe_ptr()
        def block(k: Int) {imm sp, imm fp, imm zp, imm n, imm blocks}:
            var s = 0.0
            var q = 0.0
            for i in range(k * (n // blocks), (k + 1) * (n // blocks)):
                if not isfinite(zp[i]):
                    fp.unsafe_store(k, 1)
                s += Float64(zp[i])
                q += Float64(zp[i]) * Float64(zp[i])
            sp.unsafe_store(2 * k, s)
            sp.unsafe_store(2 * k + 1, q)
        host_parallelize(block, blocks)
        require(count_failures(finite) == 0, "normal draws are not all finite")
        var s = 0.0
        var q = 0.0
        for k in range(blocks):  # twenty block partials
            s += sums[2 * k]
            q += sums[2 * k + 1]
        var mean = s / Float64(n)
        var sd = sqrt(q / Float64(n) - mean * mean)
        require(abs(mean) < 0.03 and abs(sd - 1.0) < 0.03, "normal(0, 1): mean " + String(mean) + " sd " + String(sd))
        _ = z^
        _ = sums^
        _ = finite^
    else:
        var count = 4 * 256
        var x = List[Float32](length=count, fill=0.0)
        var dy = List[Float32](length=count, fill=0.0)
        var xp = x.unsafe_ptr()
        var gp = dy.unsafe_ptr()
        def fill(i: Int) {imm xp, imm gp}:
            xp.unsafe_store(i, Float32(2.0 * u01(0x5A31, i) - 1.0))
            gp.unsafe_store(i, Float32(2.0 * u01(0x5A32, i) - 1.0))
        host_parallelize(fill, count)
        var p = Float32(0.25)
        var scale = Float32(1.0) / (Float32(1.0) - p)
        var y = draw(count, 0, 99, 0, NRNG_DROPOUT_FORWARD, p, scale, x)
        var dx = draw(count, 0, 99, 0, NRNG_DROPOUT_BACKWARD, p, scale, dy)
        var flags = List[Int](length=count, fill=0)
        var kept = List[Int](length=count, fill=0)
        var fp = flags.unsafe_ptr()
        var kp = kept.unsafe_ptr()
        var yp = y.unsafe_ptr()
        var dp = dx.unsafe_ptr()
        def cell(i: Int) {imm fp, imm kp, imm xp, imm gp, imm yp, imm dp, imm scale}:
            # the original's mask: kept = y != 0
            if yp[i] != 0.0:
                kp.unsafe_store(i, 1)
                if not same_bits(yp[i], xp[i] * scale):
                    fp.unsafe_store(i, 1)
                elif not same_bits(dp[i], gp[i] * scale):
                    fp.unsafe_store(i, 2)
            elif dp[i] != 0.0:
                fp.unsafe_store(i, 3)
        host_parallelize(cell, count)
        var frac = Float64(count_failures(kept)) / Float64(count)
        require(frac > 0.65 and frac < 0.85, "dropout p = 0.25 kept " + String(frac))
        for i in range(count):
            require(flags[i] != 1, "dropout: a kept cell is not x * scale at " + String(i))
            require(flags[i] != 2, "dropout backward: a kept cell is not dy * scale at " + String(i))
            require(flags[i] != 3, "dropout backward mask differs from the forward mask at " + String(i))
        require(differing(draw(count, 0, 99, 0, NRNG_DROPOUT_FORWARD, p, scale, x), y, 0, count) == 0,
                "dropout does not replay from the same seed and stream")
        var tail = List[Float32](length=512, fill=0.0)
        var tp = tail.unsafe_ptr()
        def copy_tail(i: Int) {imm tp, imm xp}:
            tp.unsafe_store(i, xp[512 + i])
        host_parallelize(copy_tail, 512)
        require(differing(y, draw(512, 512, 99, 0, NRNG_DROPOUT_FORWARD, p, scale, tail), 512, 512) == 0,
                "dropout: a microbatch at its token offset does not draw the unsplit coins")
        var refused = False
        try:
            _ = draw(count, 0, 99, 0, NRNG_DROPOUT_FORWARD, 1.0, 1.0, x)
        except:
            refused = True
        require(refused, "dropout p = 1 is not refused")
        _ = x^
        _ = dy^
        _ = y^
        _ = dx^
        _ = flags^
        _ = kept^
        _ = tail^
    _ = none^


def run_samba_surface_fixture(unit: Int) raises:
    if unit < 3:
        schedule_fixture(unit)
    else:
        stream_fixture(unit)
