# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The step glue arms (DEVIATIONS 2645 to 2648) against the shipped step,
BIT FOR BIT, with reach by sabotage. Brief
docs/lanes/BRIEF_step_glue_2026-09-11.md section 6.

    pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 \\
        -D MOJOLEARN_STEP_GLUE_TRIAL=1 -I . \\
        training/checks/step_glue_check.mojo -o /tmp/step-glue-check

Without -D MOJOLEARN_STEP_GLUE_TRIAL=1 every arm runs the shipped path, so
`main` FAILS at once naming the define: a green run is always a run of the
arms.

CLAUSES (each prints PASS or FAIL lines; the last line is the verdict):
  (a) NAMES, host only: the 16 valid arm words and their names are
      inverses; eight invalid spellings raise.
  (b) RMSNorm FORWARD (`llama_rms_norm`) under shipped against rows16,
      rows8, rows4 and optskip_noshadow_rows16 on (m, dm) = (45, 24),
      (300, 64), (2048, 768), hashed operands with +0, -0, subnormals,
      tiny and large values: `out` and `sumsq` bit-equal. REACH: under
      MOJOLEARN_STEP_GLUE_ARM_SABOTAGE=1 the rows launch covers
      floor(m / R) blocks, so at m = 45 the first moved element of `out`
      must be exactly floor(45 / R) * R * dm; the shipped launch under the
      same variable must move nothing.
  (c) RMSNorm BACKWARD (`bwd_rms_norm`): `dot`, `dx`, `dW` the same way;
      REACH: the first moved element of `dot` is floor(45 / R) * R.
  (d) UPDATE: `identical_optimizer_step` against `byte_glue_update_launch`
      in place (the shipped kernel) and out of place (the transcribed
      kernel) on 1,000 elements over three tensors, gradients that overflow
      `g * g` to infinity, subnormal and zero gradients: param, m, v
      bit-equal, and the out-of-place launch leaves its inputs untouched.
      REACH: under the sabotage the out-of-place launch covers 3 blocks of
      256, so its first moved element is exactly 768.
  (e) STEP END TO END at the default ByteConfig (34,944 parameters): three
      resident steps under shipped and every arm, loss, gradient, param, m
      and v after every step bit-equal. VACUOUS if param did not move.
  (f) REFUSAL END TO END, two admitted configurations that must raise: an
      update that overflows param (caught by validate_after, rolled back
      AFTER noshadow's swap) and an lr whose step size overflows. Every arm
      must print the same message as shipped, restore param, m, v, keep
      completed_steps 0 and healthy, and a second rollback returns False.
  (g) ROLLBACK AFTER SUCCESS: one good step, then `byte_rollback` returns
      True and the pre-step bits come back, in every arm.

Small shapes only (the eager attention path at head_dim 8). The target
shape, the fused attention kernels and the price are the H100 leg's
(`tools/step_glue_leg.sh`).
"""

from std.memory import bitcast
from std.os import setenv
from max.gpu.host import DeviceContext

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.step_glue import (
    STEP_GLUE_ALL_BITS,
    step_glue_arm_name,
    step_glue_arm_parse,
    step_glue_arm_valid,
    step_glue_rows_of,
    step_glue_trial_build,
)
from training.checks.train_loop import _upload, _zeros, download_f32
from training.checks.optimizer import (
    SAB_CHUNKS,
    identical_optimizer_step,
    identical_optimizer_workspace_floats,
)
from training.checks.optimizer_oracle import OPT_ADAMW, OptimizerConfig
from training.byte_lm import (
    ByteTrainer,
    byte_glue_update_launch,
    byte_offsets,
    byte_rollback,
    byte_train_step_resident,
)
from training.byte_lm_config import ByteConfig
from transformer.impl.llama.modeling_llama import llama_rms_norm
from transformer.checks.transformer_backward import bwd_rms_norm

comptime ARM_ENV = "MOJOLEARN_STEP_GLUE_ARM"
comptime SAB_ENV = "MOJOLEARN_STEP_GLUE_ARM_SABOTAGE"
comptime NORM_EPS: Float32 = 1e-6
comptime REACH_M = 45
comptime UPDATE_TPB = 256


# ===========================================================================
# HELPERS
# ===========================================================================


def _set_arm(name: String, sabotage: Bool) raises:
    """The two variables every launcher reads per call (trial build)."""
    if not setenv(ARM_ENV, name, True):
        raise Error("step_glue_check: setenv " + String(ARM_ENV) + " failed")
    var sab = String("")
    if sabotage:
        sab = String("1")
    if not setenv(SAB_ENV, sab, True):
        raise Error("step_glue_check: setenv " + String(SAB_ENV) + " failed")


def _mix(seed: UInt64) -> UInt64:
    """splitmix64's finalizer."""
    var z = seed + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def _unit(seed: UInt64, i: Int) -> Float32:
    """A hashed value in [-1, 1) from 24 bits, exact in Float32."""
    var h = _mix(seed * UInt64(1000003) + UInt64(i))
    var q = Int(h >> 40)
    return Float32(q - 8388608) / Float32(8388608.0)


def _operands(seed: UInt64, n: Int, scale: Float32, specials: Bool) -> List[Float32]:
    """Hashed values times `scale`; with `specials`, every 97 elements plant
    +0, -0, two subnormals, a large value and a tiny one at fixed offsets
    that avoid the reach indices of clause (b)."""
    var out = List[Float32]()
    for i in range(n):
        var v = _unit(seed, i) * scale
        if specials:
            var k = i % 97
            if k == 11:
                v = Float32(0.0)
            elif k == 23:
                v = bitcast[DType.float32](UInt32(0x80000000))
            elif k == 37:
                v = bitcast[DType.float32](UInt32(0x00000100))
            elif k == 53:
                v = bitcast[DType.float32](UInt32(0x80000100))
            elif k == 71:
                v = Float32(3.0e18)
            elif k == 88:
                v = Float32(1.0e-20)
        out.append(v)
    return out^


def _weights(seed: UInt64, n: Int) -> List[Float32]:
    """Norm weights in [0.5, 1.5)."""
    var out = List[Float32]()
    for i in range(n):
        out.append(_unit(seed, i) * Float32(0.5) + Float32(1.0))
    return out^


def _append(mut dst: List[Float32], src: List[Float32]):
    for i in range(len(src)):
        dst.append(src[i])


def _first_diff(a: List[Float32], b: List[Float32]) -> Int:
    """The first index whose BITS differ, -1 when none, -2 on a length
    mismatch."""
    if len(a) != len(b):
        return -2
    for i in range(len(a)):
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            return i
    return -1


def _fail(clause: String, what: String) -> Int:
    print("FAIL " + clause + ": " + what)
    return 1


# ===========================================================================
# (a) NAMES
# ===========================================================================


def clause_names() raises -> Int:
    var fails = 0
    var count = 0
    for arm in range(STEP_GLUE_ALL_BITS + 1):
        if not step_glue_arm_valid(arm):
            continue
        count += 1
        var name = step_glue_arm_name(arm)
        if step_glue_arm_parse(name) != arm:
            fails += _fail("names", name + " does not read back as " + String(arm))
    if count != 16:
        fails += _fail("names", String(count) + " valid words, expected 16")
    if step_glue_arm_parse(String("")) != 0:
        fails += _fail("names", "empty is not shipped")
    var bad: List[String] = [
        "noshadow_optskip", "rows16_rows8", "optskip_optskip", "rows32",
        "bogus", "shipped_rows16", "rows16_optskip", "bits4",
    ]
    for i in range(len(bad)):
        var raised = False
        try:
            _ = step_glue_arm_parse(bad[i])
        except err:
            raised = True
        if not raised:
            fails += _fail("names", "'" + bad[i] + "' was accepted")
    if fails == 0:
        print("PASS names: 16 words read back, 8 invalid spellings refused")
    return fails


# ===========================================================================
# (b) RMSNorm FORWARD
# ===========================================================================


def _norm_forward(ctx: DeviceContext, x_host: List[Float32], w_host: List[Float32],
                  m: Int, dm: Int, arm: String, sabotage: Bool) raises -> List[Float32]:
    """`out` then `sumsq`, on fresh zeroed buffers."""
    _set_arm(arm, sabotage)
    var x = _upload(ctx, x_host)
    var w = _upload(ctx, w_host)
    var sumsq = _zeros(ctx, m)
    var out = _zeros(ctx, m * dm)
    llama_rms_norm(ctx, sumsq, out, x, w, m, dm, NORM_EPS)
    ctx.synchronize()
    var result = download_f32(ctx, out, m * dm)
    _append(result, download_f32(ctx, sumsq, m))
    _ = x
    _ = w
    _set_arm(String(""), False)
    return result^


def _rows_arms() -> List[String]:
    return ["rows16", "rows8", "rows4", "optskip_noshadow_rows16"]


def clause_norm_forward(ctx: DeviceContext) raises -> Int:
    var fails = 0
    var shapes: List[Int] = [REACH_M, 24, 300, 64, 2048, 768]
    var arms = _rows_arms()
    for si in range(3):
        var m = shapes[2 * si]
        var dm = shapes[2 * si + 1]
        var x = _operands(UInt64(101 + si), m * dm, Float32(2.0), True)
        var w = _weights(UInt64(201 + si), dm)
        var base = _norm_forward(ctx, x, w, m, dm, String("shipped"), False)
        for a in range(len(arms)):
            var got = _norm_forward(ctx, x, w, m, dm, arms[a], False)
            var d = _first_diff(base, got)
            if d != -1:
                fails += _fail("norm_forward", arms[a] + " at (" + String(m) + ", " + String(dm)
                    + ") first differing element " + String(d))
        if m == REACH_M:
            var shipped_sab = _norm_forward(ctx, x, w, m, dm, String("shipped"), True)
            if _first_diff(base, shipped_sab) != -1:
                fails += _fail("norm_forward", "the sabotage moved the SHIPPED launch")
            for a in range(len(arms)):
                var rows = step_glue_rows_of(step_glue_arm_parse(arms[a]))
                var expect = (m // rows) * rows * dm
                var sab = _norm_forward(ctx, x, w, m, dm, arms[a], True)
                var d = _first_diff(base, sab)
                if d != expect:
                    fails += _fail("norm_forward", "REACH " + arms[a] + " first moved element "
                        + String(d) + ", expected " + String(expect))
                else:
                    print("REACH norm_forward " + arms[a] + " threads_per_block=" + String(rows)
                        + " first_moved_row=" + String(d // dm))
    if fails == 0:
        print("PASS norm_forward: 3 shapes x 4 arms bit-equal, reach named on every arm")
    return fails


# ===========================================================================
# (c) RMSNorm BACKWARD
# ===========================================================================


def _norm_backward(ctx: DeviceContext, x_host: List[Float32], w_host: List[Float32],
                   dy_host: List[Float32], sumsq_host: List[Float32], m: Int, dm: Int,
                   arm: String, sabotage: Bool) raises -> List[Float32]:
    """`dot`, then `dx`, then `dW`, on fresh zeroed buffers."""
    _set_arm(arm, sabotage)
    var x = _upload(ctx, x_host)
    var w = _upload(ctx, w_host)
    var dy = _upload(ctx, dy_host)
    var sumsq = _upload(ctx, sumsq_host)
    var ones_host = List[Float32]()
    for _ in range(m):
        ones_host.append(Float32(1.0))
    var ones = _upload(ctx, ones_host)
    var dot = _zeros(ctx, m)
    var dx = _zeros(ctx, m * dm)
    var dw = _zeros(ctx, dm)
    var dh = _zeros(ctx, m * dm)
    var dprod = _zeros(ctx, m * dm)
    var rstd = _zeros(ctx, m)
    var dvcoef = _zeros(ctx, m)
    bwd_rms_norm(ctx, dot, dx, dw, dh, dprod, rstd, dvcoef, ones, dy, x, w, sumsq, m, dm, NORM_EPS)
    ctx.synchronize()
    var result = download_f32(ctx, dot, m)
    _append(result, download_f32(ctx, dx, m * dm))
    _append(result, download_f32(ctx, dw, dm))
    _ = x
    _ = w
    _ = dy
    _ = sumsq
    _ = ones
    _ = dh
    _ = dprod
    _ = rstd
    _ = dvcoef
    _set_arm(String(""), False)
    return result^


def clause_norm_backward(ctx: DeviceContext) raises -> Int:
    var fails = 0
    var shapes: List[Int] = [REACH_M, 24, 300, 64, 2048, 768]
    var arms = _rows_arms()
    for si in range(3):
        var m = shapes[2 * si]
        var dm = shapes[2 * si + 1]
        var x = _operands(UInt64(301 + si), m * dm, Float32(2.0), True)
        var w = _weights(UInt64(401 + si), dm)
        var dy = _operands(UInt64(501 + si), m * dm, Float32(0.5), True)
        # The saved sum of squares is the shipped forward's.
        var fwd = _norm_forward(ctx, x, w, m, dm, String("shipped"), False)
        var sumsq = List[Float32]()
        for i in range(m):
            sumsq.append(fwd[m * dm + i])
        var base = _norm_backward(ctx, x, w, dy, sumsq, m, dm, String("shipped"), False)
        for a in range(len(arms)):
            var got = _norm_backward(ctx, x, w, dy, sumsq, m, dm, arms[a], False)
            var d = _first_diff(base, got)
            if d != -1:
                fails += _fail("norm_backward", arms[a] + " at (" + String(m) + ", " + String(dm)
                    + ") first differing element " + String(d))
        if m == REACH_M:
            var shipped_sab = _norm_backward(ctx, x, w, dy, sumsq, m, dm, String("shipped"), True)
            if _first_diff(base, shipped_sab) != -1:
                fails += _fail("norm_backward", "the sabotage moved the SHIPPED launch")
            for a in range(len(arms)):
                var rows = step_glue_rows_of(step_glue_arm_parse(arms[a]))
                var expect = (m // rows) * rows
                var sab = _norm_backward(ctx, x, w, dy, sumsq, m, dm, arms[a], True)
                var d = _first_diff(base, sab)
                if d != expect:
                    fails += _fail("norm_backward", "REACH " + arms[a] + " first moved dot row "
                        + String(d) + ", expected " + String(expect))
                else:
                    print("REACH norm_backward " + arms[a] + " threads_per_block=" + String(rows)
                        + " first_moved_row=" + String(d))
    if fails == 0:
        print("PASS norm_backward: 3 shapes x 4 arms bit-equal, reach named on every arm")
    return fails


# ===========================================================================
# (d) UPDATE
# ===========================================================================


def _opt_cfg(lr: Float32, beta1: Float32, beta2: Float32, wd: Float32) -> OptimizerConfig:
    return OptimizerConfig(OPT_ADAMW, lr, beta1, beta2, Float32(1e-8), wd,
                           Float32(0.0), Float32(0.0), False, Float32(0.0))


def _update_grad(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        var v = _unit(UInt64(601), i) * Float32(0.01)
        var k = i % 50
        if k == 7:
            v = Float32(3.0e20)  # g * g overflows: v and its root are infinite
        elif k == 19:
            v = Float32(1.0e-25)  # g * g is subnormal before the flush
        elif k == 31:
            v = Float32(0.0)
        out.append(v)
    return out^


def _update_v(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        var v = _unit(UInt64(602), i)
        if v < Float32(0.0):
            v = -v
        if i % 50 == 13:
            v = Float32(0.0)
        out.append(v * Float32(1.0e-4))
    return out^


def _update_shipped(ctx: DeviceContext, p_h: List[Float32], g_h: List[Float32],
                    m_h: List[Float32], v_h: List[Float32], offsets: List[Int],
                    cfg: OptimizerConfig, t: Int) raises -> List[Float32]:
    var n = offsets[len(offsets) - 1]
    var j = len(offsets) - 1
    var p = _upload(ctx, p_h)
    var g = _upload(ctx, g_h)
    var m = _upload(ctx, m_h)
    var v = _upload(ctx, v_h)
    var denom = _zeros(ctx, n)
    var q = _zeros(ctx, n)
    var sumsq = _zeros(ctx, j)
    var norms = _zeros(ctx, j)
    var total = _zeros(ctx, 1)
    var out2 = _zeros(ctx, 2)
    var ws = _zeros(ctx, identical_optimizer_workspace_floats(offsets))
    var sabp = _zeros(ctx, SAB_CHUNKS)
    var flags = List[Bool]()
    for _ in range(j):
        flags.append(False)
    identical_optimizer_step(ctx, p, g, m, v, denom, q, sumsq, norms, total, out2,
        ws, sabp, flags, offsets, cfg, t)
    var result = download_f32(ctx, p, n)
    _append(result, download_f32(ctx, m, n))
    _append(result, download_f32(ctx, v, n))
    _ = g
    return result^


def _update_glue(ctx: DeviceContext, p_h: List[Float32], g_h: List[Float32],
                 m_h: List[Float32], v_h: List[Float32], n: Int, cfg: OptimizerConfig,
                 t: Int, out_of_place: Bool, sabotage: Bool) raises -> List[Float32]:
    """The new param, m, v, then the INPUT param buffer as it is after the
    launch (the update for in place; untouched for out of place)."""
    _set_arm(String("shipped"), sabotage)
    var p = _upload(ctx, p_h)
    var g = _upload(ctx, g_h)
    var m = _upload(ctx, m_h)
    var v = _upload(ctx, v_h)
    var po = _zeros(ctx, n)
    var mo = _zeros(ctx, n)
    var vo = _zeros(ctx, n)
    var denom = _zeros(ctx, n)
    var q = _zeros(ctx, n)
    byte_glue_update_launch(ctx, p, g, m, v, po, mo, vo, denom, q, n, cfg, t, out_of_place)
    var result = List[Float32]()
    if out_of_place:
        _append(result, download_f32(ctx, po, n))
        _append(result, download_f32(ctx, mo, n))
        _append(result, download_f32(ctx, vo, n))
    else:
        _append(result, download_f32(ctx, p, n))
        _append(result, download_f32(ctx, m, n))
        _append(result, download_f32(ctx, v, n))
    _append(result, download_f32(ctx, p, n))
    _ = g
    _set_arm(String(""), False)
    return result^


def clause_update(ctx: DeviceContext) raises -> Int:
    var fails = 0
    var offsets: List[Int] = [0, 300, 700, 1000]
    var n = 1000
    var p_h = _operands(UInt64(603), n, Float32(2.0), False)
    var g_h = _update_grad(n)
    var m_h = _operands(UInt64(604), n, Float32(1.0e-3), False)
    var v_h = _update_v(n)
    var cfg = _opt_cfg(Float32(1.0e-3), Float32(0.9), Float32(0.999), Float32(0.01))
    var t = 3
    var base = _update_shipped(ctx, p_h, g_h, m_h, v_h, offsets, cfg, t)
    var saw_inf = False
    for i in range(n, 3 * n):
        if (bitcast[DType.uint32](base[i]) & UInt32(0x7FFFFFFF)) == UInt32(0x7F800000):
            saw_inf = True
    if not saw_inf:
        fails += _fail("update", "VACUOUS: no infinite moment in the shipped output")
    for mode in range(2):
        var oop = mode == 1
        var label = String("noshadow (out of place)") if oop else String("optskip (in place)")
        var got = _update_glue(ctx, p_h, g_h, m_h, v_h, n, cfg, t, oop, False)
        var state = List[Float32]()
        for i in range(3 * n):
            state.append(got[i])
        var d = _first_diff(base, state)
        if d != -1:
            fails += _fail("update", label + " first differing element " + String(d))
        if oop:
            var input_after = List[Float32]()
            for i in range(3 * n, 4 * n):
                input_after.append(got[i])
            if _first_diff(p_h, input_after) != -1:
                fails += _fail("update", "the out-of-place launch wrote its input param")
    var sab = _update_glue(ctx, p_h, g_h, m_h, v_h, n, cfg, t, True, True)
    var sab_state = List[Float32]()
    for i in range(3 * n):
        sab_state.append(sab[i])
    var expect = (n // UPDATE_TPB) * UPDATE_TPB
    var d = _first_diff(base, sab_state)
    if d != expect:
        fails += _fail("update", "REACH out-of-place first moved element " + String(d)
            + ", expected " + String(expect))
    else:
        print("REACH update noshadow first_moved_element=" + String(d))
    var sab_in = _update_glue(ctx, p_h, g_h, m_h, v_h, n, cfg, t, False, True)
    var sab_in_state = List[Float32]()
    for i in range(3 * n):
        sab_in_state.append(sab_in[i])
    if _first_diff(base, sab_in_state) != -1:
        fails += _fail("update", "the sabotage moved the in-place (shipped kernel) launch")
    if fails == 0:
        print("PASS update: in place and out of place bit-equal to identical_optimizer_step"
            + " (infinite moments included), input untouched, reach named")
    return fails


# ===========================================================================
# (e), (f), (g) THE STEP END TO END
# ===========================================================================


def _trainer_arms() -> List[String]:
    return [
        "optskip", "noshadow", "optskip_noshadow", "rows16", "rows8", "rows4",
        "optskip_noshadow_rows16", "optskip_noshadow_rows8",
    ]


def _initial_params(config: ByteConfig) raises -> List[Float32]:
    var offsets = byte_offsets(config)
    var n_tensors = config.n_tensors()
    var out = List[Float32]()
    for j in range(n_tensors):
        # Norm weights (block tensors 0 and 5) near 1, everything else small.
        var is_norm = j >= 1 and j < n_tensors - 1 and ((j - 1) % 9 == 0 or (j - 1) % 9 == 5)
        for i in range(offsets[j], offsets[j + 1]):
            var v = _unit(UInt64(777), i) * Float32(0.02)
            if is_norm:
                v = v + Float32(1.0)
            out.append(v)
    return out^


def _ids(config: ByteConfig, step: Int) -> List[Int32]:
    var out = List[Int32]()
    var count = config.batch * (config.length + 1)
    for i in range(count):
        var h = _mix(UInt64(5000 + step) * UInt64(65537) + UInt64(i))
        out.append(Int32(Int(h % UInt64(config.vocab_size))))
    return out^


def _new_trainer(ctx: DeviceContext, cfg: OptimizerConfig) raises -> ByteTrainer:
    var config = ByteConfig()
    var n = config.n_total()
    var p0 = _initial_params(config)
    var zeros = List[Float32]()
    for _ in range(n):
        zeros.append(Float32(0.0))
    var flags = List[Bool]()
    for _ in range(config.n_tensors()):
        flags.append(False)
    return ByteTrainer(ctx, p0, zeros, zeros, flags, 0, cfg, config)


def _state(ctx: DeviceContext, mut tr: ByteTrainer) raises -> List[Float32]:
    var n = tr.config.n_total()
    var out = download_f32(ctx, tr.buffers.param, n)
    _append(out, download_f32(ctx, tr.buffers.m_state, n))
    _append(out, download_f32(ctx, tr.buffers.v_state, n))
    return out^


def _run_steps(ctx: DeviceContext, arm: String, cfg: OptimizerConfig, steps: Int) raises -> List[Float32]:
    """Loss, gradient, param, m, v after every step."""
    _set_arm(arm, False)
    var tr = _new_trainer(ctx, cfg)
    var config = ByteConfig()
    var n = config.n_total()
    var record = List[Float32]()
    for s in range(steps):
        var r = byte_train_step_resident(ctx, tr, _ids(config, s))
        record.append(r.loss)
        _append(record, download_f32(ctx, tr.buffers.grad, n))
        _append(record, _state(ctx, tr))
    _set_arm(String(""), False)
    return record^


def clause_step(ctx: DeviceContext) raises -> Int:
    var fails = 0
    var cfg = _opt_cfg(Float32(1.0e-3), Float32(0.9), Float32(0.999), Float32(0.01))
    var base = _run_steps(ctx, String("shipped"), cfg, 3)
    var config = ByteConfig()
    var n = config.n_total()
    var p0 = _initial_params(config)
    var p_last = List[Float32]()
    var per_step = 1 + 4 * n
    for i in range(n):
        p_last.append(base[2 * per_step + 1 + n + i])
    if _first_diff(p0, p_last) == -1:
        fails += _fail("step", "VACUOUS: three shipped steps did not move a parameter")
    var arms = _trainer_arms()
    for a in range(len(arms)):
        var got = _run_steps(ctx, arms[a], cfg, 3)
        var d = _first_diff(base, got)
        if d != -1:
            var where = String("element ") + String(d)
            if d >= 0:
                where = where + " (step " + String(d // per_step + 1) + ", offset "
                    + String(d % per_step) + " of loss 1, grad n, param n, m n, v n)"
            fails += _fail("step", arms[a] + " differs at " + where)
    if fails == 0:
        print("PASS step: 3 resident steps x " + String(len(arms))
            + " arms bit-equal to shipped (loss, gradient, param, m, v)")
    return fails


@fieldwise_init
struct RefusalOutcome(Copyable, Movable):
    """What one refused step leaves: its message, whether param, m and v are
    the pre-step bits, the step count, health, and a second rollback."""

    var raised: Bool
    var msg: String
    var restored: Bool
    var completed: Int
    var healthy: Bool
    var second_rollback: Bool

    def describe(self) -> String:
        return (String("raised=") + String(self.raised) + " msg='" + self.msg + "' restored="
            + String(self.restored) + " completed=" + String(self.completed) + " healthy="
            + String(self.healthy) + " second_rollback=" + String(self.second_rollback))

    def same_as(self, other: RefusalOutcome) -> Bool:
        return (self.raised == other.raised and self.msg == other.msg
            and self.restored == other.restored and self.completed == other.completed
            and self.healthy == other.healthy and self.second_rollback == other.second_rollback)

    def clean(self) -> Bool:
        return (self.raised and self.restored and self.completed == 0 and self.healthy
            and not self.second_rollback)


def _refusal_outcome(ctx: DeviceContext, arm: String, cfg: OptimizerConfig) raises -> RefusalOutcome:
    _set_arm(arm, False)
    var tr = _new_trainer(ctx, cfg)
    var config = ByteConfig()
    var before = _state(ctx, tr)
    var raised = False
    var msg = String("")
    try:
        _ = byte_train_step_resident(ctx, tr, _ids(config, 0))
    except err:
        raised = True
        msg = String(err)
    var after = _state(ctx, tr)
    var restored = _first_diff(before, after) == -1
    var second = byte_rollback(ctx, tr)
    _set_arm(String(""), False)
    return RefusalOutcome(raised, msg, restored, tr.completed_steps, tr.healthy, second)


def clause_refusal(ctx: DeviceContext) raises -> Int:
    var fails = 0
    var cfgs = List[OptimizerConfig]()
    # validate_after: decoupled decay of 1 takes a norm weight near 1 to about
    # -3.4e38 and the step of about lr overflows it (param nonfinite), so the
    # rollback runs AFTER noshadow's swap.
    cfgs.append(_opt_cfg(Float32(3.4e38), Float32(0.0), Float32(0.0), Float32(1.0)))
    # the step size lr / (1 - beta1) overflows.
    cfgs.append(_opt_cfg(Float32(3.0e38), Float32(0.9), Float32(0.999), Float32(0.0)))
    var arms = _trainer_arms()
    for c in range(len(cfgs)):
        var base = _refusal_outcome(ctx, String("shipped"), cfgs[c])
        print("REFUSAL case " + String(c) + " shipped: " + base.describe())
        if not base.clean():
            fails += _fail("refusal", "case " + String(c) + " shipped outcome is not a clean refusal: "
                + base.describe())
        for a in range(len(arms)):
            var got = _refusal_outcome(ctx, arms[a], cfgs[c])
            if not got.same_as(base):
                fails += _fail("refusal", "case " + String(c) + " " + arms[a] + ": " + got.describe())
    if fails == 0:
        print("PASS refusal: 2 cases x " + String(len(arms))
            + " arms raise the shipped message and restore the shipped state")
    return fails


def _rollback_outcome(ctx: DeviceContext, arm: String, cfg: OptimizerConfig) raises -> String:
    _set_arm(arm, False)
    var tr = _new_trainer(ctx, cfg)
    var config = ByteConfig()
    var before = _state(ctx, tr)
    _ = byte_train_step_resident(ctx, tr, _ids(config, 0))
    var moved = _first_diff(before, _state(ctx, tr)) != -1
    var rolled = byte_rollback(ctx, tr)
    var restored = _first_diff(before, _state(ctx, tr)) == -1
    _set_arm(String(""), False)
    return ("moved=" + String(moved) + " rolled=" + String(rolled) + " restored="
        + String(restored) + " completed=" + String(tr.completed_steps))


def clause_rollback(ctx: DeviceContext) raises -> Int:
    var fails = 0
    var cfg = _opt_cfg(Float32(1.0e-3), Float32(0.9), Float32(0.999), Float32(0.01))
    var expect = String("moved=True rolled=True restored=True completed=0")
    var arms: List[String] = ["shipped"]
    var more = _trainer_arms()
    for a in range(len(more)):
        arms.append(more[a])
    for a in range(len(arms)):
        var got = _rollback_outcome(ctx, arms[a], cfg)
        if got != expect:
            fails += _fail("rollback", arms[a] + ": " + got)
    if fails == 0:
        print("PASS rollback: one step then byte_rollback restores the pre-step bits in "
            + String(len(arms)) + " arms")
    return fails


def main() raises:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("step_glue_check: requires -D MOJOLEARN_NUMERIC_IDENTICAL=1")
    if not step_glue_trial_build():
        print("step_glue_check: FAIL, built without -D MOJOLEARN_STEP_GLUE_TRIAL=1:"
            + " every arm would run the shipped path, so no clause would be a run of the arms")
        raise Error("step_glue_check: requires -D MOJOLEARN_STEP_GLUE_TRIAL=1")
    var fails = clause_names()
    var ctx = DeviceContext()
    fails += clause_norm_forward(ctx)
    fails += clause_norm_backward(ctx)
    fails += clause_update(ctx)
    fails += clause_step(ctx)
    fails += clause_refusal(ctx)
    fails += clause_rollback(ctx)
    if fails != 0:
        print("step_glue_check: FAIL (" + String(fails) + " failures)")
        raise Error("step_glue_check: FAIL")
    print("step_glue_check: PASS (names, norm forward and backward with reach, update with reach,"
        + " step, refusal, rollback)")
