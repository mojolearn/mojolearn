# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE maximize= SEAM CHECK (DEVIATION 6200, IDENTITY_PATHS row 200).

    tools/with_identical_mode.sh pixi run mojo run -I . training/checks/maximize_check.mojo

The production sign flip, `training/maximize.mojo::maximize_negate`, is what
both optimizer bindings run on the gradient when `maximize=True`. This check
holds it against a restatement written here (`oracle_negate`, the sign bit
XORed through `bitcast`, i.e. torch's `-grads[i]`), through the normative
step `optimizer_step_oracle` for SGD (momentum + Nesterov + coupled decay,
the copy arm on step 1), Adam (coupled decay) and AdamW, with and without
the clip, BIT FOR BIT on every parameter, moment and clipped gradient.

THE FIXTURE MUST SEPARATE. `-g` and `0.0 - g` differ only at `g = +0.0`. The
fixture plants +0.0 and -0.0 gradient cells and -0.0 parameters, and before
trusting anything the check runs the SGD step on both spellings and exits 2
(VACUOUS) if they agree.

CARD. `MOJOLEARN_MAXIMIZE_CARD=<path>` writes the negated gradient and the
step's parameters per case through `core.identity_trace.IdentityTrace`, so
`tools/identity_trace_diff.py` localizes a cross-box difference to the flip
(`<case>.neg_grad`) or to the step after it (`<case>.param`).

Exit 0 PASS, 1 FAIL (naming the case and stage), 2 VACUOUS.
Sabotage arm: training/checks/sabotage/maximize_6200_subtract_from_zero.patch.
"""

from std.memory import bitcast
from std.os import getenv
from std.sys import exit

from checks.fixture_rng import u01_triple
from core.identity_trace import IdentityTrace
from training.maximize import maximize_negate
from training.checks.optimizer_contract import OPT_ADAM, OPT_ADAMW, OPT_SGD, OptimizerConfig
from training.checks.optimizer_oracle import optimizer_step_oracle


def oracle_negate(x: Float32) -> Float32:
    """The restatement: torch's `-g`, the IEEE sign bit flipped."""
    return bitcast[DType.float32](bitcast[DType.uint32](x) ^ UInt32(0x80000000))


def subtract_negate(x: Float32) -> Float32:
    """The OTHER legal spelling, `0.0 - g`, for the separation guard only."""
    return Float32(0.0) - x


def fixture(seed: Int, n: Int, lo: Float32, hi: Float32, plant: Int) -> List[Float32]:
    """`n` floats on [lo, hi) from checks/fixture_rng.mojo's `u01_triple`; `plant` 1 puts +0.0, -0.0, +0.0 in
    each tensor's first three cells (a gradient), 2 puts -0.0 there (a
    parameter). Tensors are 8, 5 and 3 long."""
    var out = List[Float32](length=n, fill=Float32(0.0))
    for i in range(n):
        out[i] = lo + (hi - lo) * Float32(u01_triple(i, seed, 6200))
    var starts: List[Int] = [0, 8, 13]
    for k in range(3):
        var b = starts[k]
        if plant == 1:
            out[b] = Float32(0.0)
            out[b + 1] = Float32(-0.0)
            out[b + 2] = Float32(0.0)
        elif plant == 2:
            out[b] = Float32(-0.0)
            out[b + 1] = Float32(-0.0)
            out[b + 2] = Float32(-0.0)
    return out^


def bits_equal(a: List[Float32], b: List[Float32]) -> Int:
    """-1 when bitwise equal, else the first differing index."""
    if len(a) != len(b):
        return 0
    for i in range(len(a)):
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            return i
    return -1


struct Run(Movable):
    var param: List[Float32]
    var grad: List[Float32]
    var m: List[Float32]
    var v: List[Float32]

    def __init__(out self, var param: List[Float32], var grad: List[Float32], var m: List[Float32], var v: List[Float32]):
        self.param = param^
        self.grad = grad^
        self.m = m^
        self.v = v^


def run_steps(cfg: OptimizerConfig, spelling: Int, mut trace: IdentityTrace, tag: String) raises -> Run:
    """Two steps. `spelling` 0: the production `maximize_negate`; 1: the
    restatement; 2: the subtraction spelling."""
    var n = 16
    var offsets: List[Int] = [0, 8, 13, 16]
    var param = fixture(11, n, -0.5, 0.5, 2)
    var m = List[Float32](length=n, fill=Float32(0.0))
    var v = List[Float32](length=n, fill=Float32(0.0))
    var init: List[Bool] = [False, False, False]
    var grad = List[Float32]()
    for t in range(1, 3):
        var g = fixture(100 + t, n, -1.0, 1.0, 1)
        grad = List[Float32](length=n, fill=Float32(0.0))
        for i in range(n):
            if spelling == 0:
                grad[i] = maximize_negate(g[i])
            elif spelling == 1:
                grad[i] = oracle_negate(g[i])
            else:
                grad[i] = subtract_negate(g[i])
        trace.record_list_f32(tag + ".t" + String(t) + ".neg_grad", grad)
        _ = optimizer_step_oracle(param, grad, m, v, init, offsets, cfg, t)
        trace.record_list_f32(tag + ".t" + String(t) + ".param", param)
    return Run(param^, grad^, m^, v^)


def main() raises:
    var card = getenv("MOJOLEARN_MAXIMIZE_CARD")
    var trace = IdentityTrace.to_path(card) if card != "" else IdentityTrace.disabled()
    var none = IdentityTrace.disabled()
    var sgd = OptimizerConfig(OPT_SGD, 1e-2, 0.9, 0.999, 1e-8, 0.01, 0.9, 0.0, True, 0.0)

    # THE FIXTURE MUST SEPARATE the two spellings on SGD's copy arm.
    var a = run_steps(sgd.copy(), 1, none, "sep.a")
    var b = run_steps(sgd.copy(), 2, none, "sep.b")
    if bits_equal(a.param, b.param) < 0 and bits_equal(a.m, b.m) < 0:
        print("maximize_check: VACUOUS: the fixture does not separate -g from 0.0 - g")
        exit(2)
    print("maximize_check: fixture separates -g from 0.0 - g (param first diff at", bits_equal(a.param, b.param), ")")

    var cfgs = List[OptimizerConfig]()
    var names = List[String]()
    for clip in range(2):
        var mn = Float32(1.0) if clip == 1 else Float32(0.0)
        cfgs.append(OptimizerConfig(OPT_SGD, 1e-2, 0.9, 0.999, 1e-8, 0.01, 0.9, 0.0, True, mn))
        names.append("sgd_nesterov_decay_clip" + String(clip))
        cfgs.append(OptimizerConfig(OPT_ADAM, 1e-3, 0.9, 0.999, 1e-8, 0.01, 0.0, 0.0, False, mn))
        names.append("adam_decay_clip" + String(clip))
        cfgs.append(OptimizerConfig(OPT_ADAMW, 1e-3, 0.9, 0.999, 1e-8, 0.01, 0.0, 0.0, False, mn))
        names.append("adamw_clip" + String(clip))
    var fails = 0
    for c in range(len(cfgs)):
        var got = run_steps(cfgs[c].copy(), 0, trace, names[c])
        var want = run_steps(cfgs[c].copy(), 1, none, names[c] + ".oracle")
        var stages: List[String] = ["param", "grad", "m", "v"]
        var d0 = bits_equal(got.param, want.param)
        var d1 = bits_equal(got.grad, want.grad)
        var d2 = bits_equal(got.m, want.m)
        var d3 = bits_equal(got.v, want.v)
        var ds: List[Int] = [d0, d1, d2, d3]
        var ok = True
        for k in range(4):
            if ds[k] >= 0:
                print("FAIL maximize seam (DEVIATION 6200):", names[c], stages[k], "first differs at", ds[k])
                ok = False
                fails += 1
        if ok:
            print("OK  ", names[c], ": production flip == -g restatement, bitwise")
    if fails:
        print("maximize_check: FAIL")
        exit(1)
    print("maximize_check: PASS")
