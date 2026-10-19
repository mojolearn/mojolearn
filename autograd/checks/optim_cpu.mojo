# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# autograd/optim.mojo against a float64 host oracle written here (lane
# ag-optim, 2026-10-19): SGD, SGD with momentum (dampening), Nesterov, Adam,
# AdamW, and the global-norm clip under Adam and under momentum SGD, six
# steps each over 37 parameters, plus the learning-rate schedules. Each
# case prints PASS with an FNV-1a digest of the executor's float32 result
# words; run_optim_cases is generic over Exec so a GPU driver can print the
# same digests for the identity comparison.
#
# Build and run (orchestrator): mojo build -I . autograd/checks/optim_cpu.mojo -o optim_cpu && ./optim_cpu
from std.math import cos, sqrt
from std.memory import bitcast
from sequence.exec import HostExec
from sequence.exec_trait import Exec
from sequence.ops import FP, OPT_ADAM, OPT_ADAMW, OPT_SGD
from autograd.optim import (
    LrSchedule,
    OptimizerConfig,
    OptimizerState,
    SCHED_CONSTANT,
    SCHED_COSINE,
    SCHED_STEP,
    optimizer_config,
    scheduled_lr,
    step_lr,
)

comptime N_PARAMS = 37
comptime N_STEPS = 6


def _ptr(mut values: List[Float32]) -> FP:
    return FP(unsafe_from_address=Int(values.unsafe_ptr()))


def _grad(i: Int, t: Int) -> Float32:
    return Float32((i * 7 + t * 3) % 11 - 5) / Float32(8)


def _param0(i: Int) -> Float32:
    return Float32((i * 5) % 13 - 6) / Float32(16)


def _oracle(cfg: OptimizerConfig, lrs: List[Float64]) -> List[Float64]:
    """torch.optim's SGD / Adam / AdamW and clip_grad_norm_ in float64."""
    var n = N_PARAMS
    var p = List[Float64](length=n, fill=0.0)
    var buf = List[Float64](length=n, fill=0.0)
    var m = List[Float64](length=n, fill=0.0)
    var v = List[Float64](length=n, fill=0.0)
    for i in range(n):
        p[i] = Float64(_param0(i))
    var mu = Float64(cfg.momentum)
    var damp = Float64(cfg.dampening)
    var wd = Float64(cfg.weight_decay)
    var b1 = Float64(cfg.beta1)
    var b2 = Float64(cfg.beta2)
    var eps = Float64(cfg.eps)
    for t in range(1, N_STEPS + 1):
        var lr = lrs[t - 1]
        var g = List[Float64](length=n, fill=0.0)
        for i in range(n):
            g[i] = Float64(_grad(i, t))
        if cfg.clip > Float32(0.0):
            var ss = 0.0
            for i in range(n):
                ss += g[i] * g[i]
            var coef = Float64(cfg.clip) / (sqrt(ss) + 1.0e-6)
            if coef > 1.0:
                coef = 1.0
            for i in range(n):
                g[i] *= coef
        for i in range(n):
            var gi = g[i]
            if cfg.kind == OPT_SGD:
                gi += wd * p[i]
                if mu != 0.0:
                    if t == 1:
                        buf[i] = gi
                    else:
                        buf[i] = mu * buf[i] + (1.0 - damp) * gi
                    if cfg.nesterov:
                        gi = gi + mu * buf[i]
                    else:
                        gi = buf[i]
                p[i] -= lr * gi
            else:
                if cfg.kind == OPT_ADAMW:
                    p[i] *= 1.0 - lr * wd
                else:
                    gi += wd * p[i]
                m[i] = b1 * m[i] + (1.0 - b1) * gi
                v[i] = b2 * v[i] + (1.0 - b2) * gi * gi
                var bc1 = 1.0 - b1 ** Float64(t)
                var bc2 = 1.0 - b2 ** Float64(t)
                p[i] -= (lr / bc1) * m[i] / (sqrt(v[i]) / sqrt(bc2) + eps)
    return p^


def _digest(values: List[Float32]) -> UInt64:
    var d = UInt64(1469598103934665603)
    for i in range(len(values)):
        d = (d ^ UInt64(bitcast[DType.uint32](values[i]))) * UInt64(1099511628211)
    return d


def _case[E: Exec](mut ex: E, name: String, cfg: OptimizerConfig, sched: LrSchedule) raises:
    var n = N_PARAMS
    var hp = List[Float32](length=n, fill=0.0)
    var hg = List[Float32](length=n, fill=0.0)
    for i in range(n):
        hp[i] = _param0(i)
    var P = ex.alloc(n)
    var G = ex.alloc(n)
    ex.upload(P, _ptr(hp), n)
    var state = OptimizerState(ex, cfg, n)
    var lrs = List[Float64]()
    for t in range(1, N_STEPS + 1):
        for i in range(n):
            hg[i] = _grad(i, t)
        ex.upload(G, _ptr(hg), n)
        var lr = scheduled_lr(sched, cfg.lr, t)
        lrs.append(Float64(lr))
        step_lr(ex, cfg, state, P, G, n, t, lr)
    ex.sync()
    ex.download(_ptr(hp), P, n)
    state.release(ex)
    var want = _oracle(cfg, lrs)
    for i in range(n):
        var got = Float64(hp[i])
        var tol = 2.0e-5 + 2.0e-5 * abs(want[i])
        if not (abs(got - want[i]) <= tol):
            print("FAIL", name, "index", i, "got", got, "want", want[i])
            raise Error(name + ": optimizer step differs from the float64 oracle")
    print("PASS", name, "params", n, "steps", N_STEPS, "digest", _digest(hp))
    _ = hg^


def _schedule_checks() raises:
    """scheduled_lr against the float64 formulas."""
    var base = Float32(0.1)
    var cases = 0
    for t in range(1, 21):
        var st = LrSchedule(SCHED_STEP, 20, 0, 5, Float32(0.5), Float32(0.0))
        var want = Float64(Float32(0.1)) * Float64(0.5) ** Float64((t - 1) // 5)
        var cs = LrSchedule(SCHED_COSINE, 20, 4, 1, Float32(1.0), Float32(0.001))
        var done = Float64(min(t - 1, 20))
        var b64 = Float64(Float32(0.1))
        var m64 = Float64(Float32(0.001))
        var wc = m64 + 0.5 * (b64 - m64) * (1.0 + cos(3.14159265358979 * done / 20.0))
        if t <= 4:
            wc = wc * Float64(t) / 4.0
        var cn = LrSchedule(SCHED_CONSTANT, 1, 0, 1, Float32(1.0), Float32(0.0))
        var pairs: List[Float64] = [Float64(scheduled_lr(st, base, t)), want,
                                    Float64(scheduled_lr(cs, base, t)), wc,
                                    Float64(scheduled_lr(cn, base, t)), Float64(base)]
        for k in range(3):
            var got = pairs[2 * k]
            var w = pairs[2 * k + 1]
            if not (abs(got - w) <= 1.0e-6 + 1.0e-5 * abs(w)):
                print("FAIL schedule", k, "t", t, "got", got, "want", w)
                raise Error("scheduled_lr differs from its formula")
            cases += 1
    print("PASS schedules", cases)


def run_optim_cases[E: Exec](mut ex: E) raises:
    var const_s = LrSchedule(SCHED_CONSTANT, 1, 0, 1, Float32(1.0), Float32(0.0))
    var cos_s = LrSchedule(SCHED_COSINE, N_STEPS, 2, 1, Float32(1.0), Float32(0.001))

    var sgd = optimizer_config(OPT_SGD, Float32(0.1))
    sgd.weight_decay = Float32(0.01)
    _case(ex, "sgd", sgd, const_s)

    var mom = optimizer_config(OPT_SGD, Float32(0.05))
    mom.momentum = Float32(0.9)
    mom.dampening = Float32(0.1)
    _case(ex, "sgd_momentum", mom, const_s)

    var nes = optimizer_config(OPT_SGD, Float32(0.05))
    nes.momentum = Float32(0.9)
    nes.nesterov = True
    nes.weight_decay = Float32(0.01)
    _case(ex, "sgd_nesterov", nes, cos_s)

    var adam = optimizer_config(OPT_ADAM, Float32(0.01))
    adam.weight_decay = Float32(0.01)
    _case(ex, "adam", adam, const_s)

    var adamw = optimizer_config(OPT_ADAMW, Float32(0.01))
    adamw.weight_decay = Float32(0.05)
    _case(ex, "adamw", adamw, cos_s)

    var adam_clip = optimizer_config(OPT_ADAM, Float32(0.01))
    adam_clip.clip = Float32(0.5)
    _case(ex, "adam_clip", adam_clip, const_s)

    var mom_clip = optimizer_config(OPT_SGD, Float32(0.05))
    mom_clip.momentum = Float32(0.9)
    mom_clip.clip = Float32(0.5)
    _case(ex, "sgd_momentum_clip", mom_clip, const_s)

    _schedule_checks()
    print("PASS optim_cases 7")


def main() raises:
    var ex = HostExec()
    run_optim_cases(ex)
