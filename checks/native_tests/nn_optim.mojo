# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""The optimizers and learning-rate schedules of Sequential.compile (lane nn-optim, 2026-11-23;
autograd/nn_optim.mojo).

- Steps: rmsprop (plain, weight decay, momentum, centered), adagrad (lr_decay, initial accumulator, weight
  decay) and lamb (global clip, weight decay; trust_clip with always_adapt; no clip) run five steps on a
  two-tensor parameter vector through `nn_opt_step` (the fit's own step) against a float64 reference written
  here from torch.optim.RMSprop / Adagrad and timm's Lamb: parameters and both state buffers within 3e-5
  relative (+1e-6 absolute). An optimizer has no parameters of its own, so there is no gradient to probe:
  the reference of its update is the check.
- Schedules: every value a fit uses against a float64 reference written here from torch's MultiStepLR,
  CosineAnnealingLR, CosineAnnealingWarmRestarts, OneCycleLR (cos, linear, three_phase) and
  ReduceLROnPlateau, and the linear warmup-then-decay rule, 1e-6 relative.
- Fits: a 4-16-1 tanh network on a smooth regression, one fit per optimizer and per schedule: the epoch loss
  must fall below a quarter of the first epoch's and every loss, parameter and prediction must be finite;
  fit(6) + fit(6, resume) must give the bits of fit(12) (lamb, centered rmsprop, adagrad, one_cycle,
  warm restarts, plateau), and so must fit(6), save_checkpoint, load and resume(6) (lamb, plateau).
- Batch-norm state: a model whose first layer is a batch norm holds the same running statistics, word for
  word, after a fit with lamb (weight decay 0.5), rmsprop or adagrad as after one with plain sgd: the
  statistics depend on the input only, so no optimizer moved them.
Columns: the CPU column runs everything on HostExec, independent units across host cores. The device column
also runs the steps and the fits on DeviceExec and requires the host column's words.
Launches a step (autograd/nn_optim.mojo): rmsprop / adagrad 1; lamb 7 with the clip and weight decay, 4
without the clip, 5 without weight decay and always_adapt off (one fill instead of the 2 trust launches).
"""
from std.math import cos, sqrt
from std.memory import bitcast
from std.testing import TestSuite
from std.time import perf_counter_ns
from sequence.exec_trait import Exec
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from sequence.ops import FP, OPT_SGD
from sequence.recurrent import OptConfig, OptState
from autograd.model import OUT_RAW, SCHED_EXPONENTIAL, epoch_lr
from autograd.nn_optim import (
    NN_LAMB_BIAS_CORRECTION, NN_LAMB_CLIP, NN_LAMB_GRAD_AVERAGING, NN_OPT_ADAGRAD, NN_OPT_LAMB, NN_OPT_RMSPROP,
    NN_SCHED_ARGS, NN_SCHED_COSINE, NN_SCHED_LINEAR_WARMUP, NN_SCHED_MULTI_STEP, NN_SCHED_ONE_CYCLE,
    NN_SCHED_PLATEAU, NN_SCHED_WARM_RESTARTS, NnLamb, NnPlateau, nn_epoch_lr, nn_multi_step_count, nn_opt_step,
    nn_step_lrs,
)
from checks.native_tests.core_ops_common import normal
from checks.native_tests.training_sequential_common import SeqModel, L_LINEAR, L_TANH, L_FLATTEN
from checks.native_tests.testsuite_cases import CPU_COLUMN_BUILD
from core.host_parallel import host_parallelize

comptime _PI = 3.141592653589793
comptime L_BATCH_NORM = 46


def _fail(message: String) raises:
    print("FAIL NN_OPTIM", message)
    raise Error(message)


def _close(what: String, got: Float64, want: Float64, rel: Float64, absol: Float64 = 0.0) raises:
    if not (got == got) or abs(got - want) > rel * abs(want) + absol:
        _fail(what + ": got " + String(got) + " want " + String(want))


def _same_words(what: String, a: List[Float32], b: List[Float32]) raises:
    if len(a) != len(b):
        _fail(what + ": " + String(len(a)) + " words on the host column, " + String(len(b)) + " on the device")
    for i in range(len(a)):  # small-loop(compared words)
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            _fail(what + ": word " + String(i) + " differs between the host column and the device")


# ---------------------------------------------------------------- steps
comptime _N = 37
comptime _SPLIT = 20
comptime _STEPS = 5


@fieldwise_init
struct _StepCase(Copyable, Movable):
    var name: String
    var cfg: OptConfig
    var lr: Float32


def _step_cases() -> List[_StepCase]:
    var c = List[_StepCase]()
    # OptConfig(kind, flags, f1, f2, eps, wd, f7): rmsprop f2 alpha, f7 momentum, bit0 centered
    c.append(_StepCase("rmsprop", OptConfig(NN_OPT_RMSPROP, 0, 0.0, 0.99, 1e-8, 0.0, 0.0), 0.01))
    c.append(_StepCase("rmsprop wd momentum", OptConfig(NN_OPT_RMSPROP, 0, 0.0, 0.9, 1e-8, 0.01, 0.5), 0.01))
    c.append(_StepCase("rmsprop centered", OptConfig(NN_OPT_RMSPROP, 1, 0.0, 0.9, 1e-6, 0.0, 0.0), 0.01))
    # adagrad f1 lr_decay, f2 initial accumulator
    c.append(_StepCase("adagrad", OptConfig(NN_OPT_ADAGRAD, 0, 0.1, 0.1, 1e-10, 0.01, 0.0), 0.1))
    # lamb f1 b1, f2 b2, f7 max_grad_norm
    var base = NN_LAMB_GRAD_AVERAGING | NN_LAMB_BIAS_CORRECTION
    c.append(_StepCase("lamb clip wd", OptConfig(NN_OPT_LAMB, base | NN_LAMB_CLIP, 0.9, 0.999, 1e-6, 0.01, 1.0), 0.01))
    c.append(_StepCase("lamb trust_clip always_adapt", OptConfig(NN_OPT_LAMB, base | 3, 0.9, 0.99, 1e-6, 0.0, 0.0), 0.02))
    c.append(_StepCase("lamb no clip", OptConfig(NN_OPT_LAMB, base, 0.8, 0.95, 1e-6, 0.1, 0.0), 0.01))
    return c^


def _grad(k: Int, i: Int) -> Float64:
    """Step k's gradient of parameter i: fixed, mixed signs, a few large."""
    var v = Float64(((i * 7 + k * 13) % 23) - 11) / 8.0
    if (i + k) % 9 == 0:
        v *= 6.0
    return v


def _reference(tc: _StepCase, mut p: List[Float64], mut s1: List[Float64], mut s2: List[Float64]):
    """torch.optim.RMSprop / Adagrad and timm Lamb in float64, _STEPS steps (s1, s2 the saved buffers:
    rmsprop buf / grad average and square average; adagrad -, sum; lamb exp_avg, exp_avg_sq)."""
    var cfg = tc.cfg
    var lr = Float64(tc.lr)
    var f1 = Float64(cfg.f1)
    var f2 = Float64(cfg.f2)
    var eps = Float64(cfg.eps)
    var wd = Float64(cfg.wd)
    var f7 = Float64(cfg.f7)
    if cfg.kind == NN_OPT_ADAGRAD:
        for i in range(_N):
            s2[i] = f2
    for k in range(_STEPS):
        var t = k + 1
        if cfg.kind == NN_OPT_RMSPROP:
            for i in range(_N):
                var g = _grad(k, i) + wd * p[i]
                s2[i] = f2 * s2[i] + (1.0 - f2) * g * g
                var avg: Float64
                if (cfg.flags & 1) != 0:
                    s1[i] = f2 * s1[i] + (1.0 - f2) * g
                    avg = sqrt(max(s2[i] - s1[i] * s1[i], 0.0)) + eps
                else:
                    avg = sqrt(s2[i]) + eps
                if f7 > 0.0:
                    s1[i] = f7 * s1[i] + g / avg
                    p[i] -= lr * s1[i]
                else:
                    p[i] -= lr * g / avg
        elif cfg.kind == NN_OPT_ADAGRAD:
            var clr = lr / (1.0 + Float64(t - 1) * f1)
            for i in range(_N):
                var g = _grad(k, i) + wd * p[i]
                s2[i] += g * g
                p[i] -= clr * g / (sqrt(s2[i]) + eps)
        else:
            var gn = Float64(0)
            for i in range(_N):
                gn += _grad(k, i) * _grad(k, i)
            gn = sqrt(gn)
            var clip = 1.0
            if (cfg.flags & NN_LAMB_CLIP) != 0 and gn > f7:
                clip = gn / f7
            var bc1 = 1.0 - _powi(f1, t)
            var bc2 = 1.0 - _powi(f2, t)
            var u = List[Float64](length=_N, fill=0)
            for i in range(_N):
                var g = _grad(k, i) / clip
                s1[i] = f1 * s1[i] + (1.0 - f1) * g
                s2[i] = f2 * s2[i] + (1.0 - f2) * g * g
                u[i] = (s1[i] / bc1) / (sqrt(s2[i]) / sqrt(bc2) + eps)
                if wd != 0.0:
                    u[i] += wd * p[i]
            var bounds: List[Int] = [0, _SPLIT, _N]
            for tt in range(2):
                var trust = 1.0
                if wd != 0.0 or (cfg.flags & 2) != 0:
                    var wn = Float64(0)
                    var un = Float64(0)
                    for i in range(bounds[tt], bounds[tt + 1]):
                        wn += p[i] * p[i]
                        un += u[i] * u[i]
                    wn = sqrt(wn)
                    un = sqrt(un)
                    if wn > 0.0 and un > 0.0:
                        trust = wn / un
                    if (cfg.flags & 1) != 0:
                        trust = min(trust, 1.0)
                for i in range(bounds[tt], bounds[tt + 1]):
                    p[i] -= lr * trust * u[i]


def run_step_case[E: Exec](mut ex: E, tc: _StepCase) raises -> List[Float32]:
    """_STEPS steps of nn_opt_step from p = normal(_N) over tensors [0, _SPLIT), [_SPLIT, _N); checks the
    float64 reference and returns (p, s1, s2) as words."""
    var p0 = normal(_N, 31)
    var P = ex.alloc(_N)
    var Gr = ex.alloc(_N)
    var s1 = ex.alloc(_N)
    var s2 = ex.alloc(_N)
    var s3 = ex.alloc(_N)
    var zero = List[Float32](length=_N, fill=Float32(0))
    var init2 = List[Float32](length=_N, fill=tc.cfg.f2 if tc.cfg.kind == NN_OPT_ADAGRAD else Float32(0))
    ex.upload(P, FP(unsafe_from_address=Int(p0.unsafe_ptr())), _N)
    ex.upload(s1, FP(unsafe_from_address=Int(zero.unsafe_ptr())), _N)
    ex.upload(s2, FP(unsafe_from_address=Int(init2.unsafe_ptr())), _N)
    ex.upload(s3, FP(unsafe_from_address=Int(zero.unsafe_ptr())), _N)
    var offsets: List[Int] = [0, _SPLIT, _N]
    var lamb = NnLamb(ex, tc.cfg, offsets, _N)
    var st = OptState()
    var g = List[Float32](length=_N, fill=Float32(0))
    for k in range(_STEPS):  # driver-loop: over optimizer steps
        for i in range(_N):  # small-loop(fixture gradient)
            g[i] = Float32(_grad(k, i))
        ex.upload(Gr, FP(unsafe_from_address=Int(g.unsafe_ptr())), _N)
        nn_opt_step(ex, tc.cfg, st, k + 1, tc.lr, P, Gr, s1, s2, s3, _N, lamb)
        ex.sync()
    var words = List[Float32](length=3 * _N, fill=Float32(0))
    ex.download(FP(unsafe_from_address=Int(words.unsafe_ptr())), P, _N)
    ex.download(FP(unsafe_from_address=Int(words.unsafe_ptr())) + _N, s1, _N)
    ex.download(FP(unsafe_from_address=Int(words.unsafe_ptr())) + 2 * _N, s2, _N)
    ex.sync()
    lamb.release(ex)
    _ = g^
    _ = zero^
    _ = init2^
    var rp = List[Float64](length=_N, fill=0)
    for i in range(_N):  # small-loop(fixture)
        rp[i] = Float64(p0[i])
    var r1 = List[Float64](length=_N, fill=0)
    var r2 = List[Float64](length=_N, fill=0)
    _reference(tc, rp, r1, r2)
    for i in range(_N):  # small-loop(compared words)
        _close(tc.name + " param " + String(i), Float64(words[i]), rp[i], 3e-5, 1e-6)
        _close(tc.name + " state1 " + String(i), Float64(words[_N + i]), r1[i], 3e-5, 1e-6)
        _close(tc.name + " state2 " + String(i), Float64(words[2 * _N + i]), r2[i], 3e-5, 1e-6)
    var moved = False
    for i in range(_N):  # small-loop(compared words)
        if words[i] != p0[i]:
            moved = True
    if not moved:
        _fail(tc.name + ": no parameter moved")
    return words^


# ---------------------------------------------------------------- schedules
def _torch_onecycle(t: Int, total: Int, lr: Float64, pct: Float64, div: Float64, fdiv: Float64, linear: Bool,
                    three: Bool) -> Float64:
    """torch OneCycleLR.get_lr at step_num = t - 1."""
    var init = lr / div
    var low = init / fdiv
    var ends = List[Float64]()
    var a = List[Float64]()
    var b = List[Float64]()
    ends.append(pct * Float64(total) - 1.0)
    a.append(init)
    b.append(lr)
    if three:
        ends.append(2.0 * pct * Float64(total) - 2.0)
        a.append(lr)
        b.append(init)
        ends.append(Float64(total - 1))
        a.append(init)
        b.append(low)
    else:
        ends.append(Float64(total - 1))
        a.append(lr)
        b.append(low)
    var s = Float64(t - 1)
    var start = Float64(0)
    for i in range(len(ends)):
        if s <= ends[i] or i == len(ends) - 1:
            var q = (s - start) / (ends[i] - start)
            if linear:
                return (b[i] - a[i]) * q + a[i]
            return b[i] + (a[i] - b[i]) / 2.0 * (cos(_PI * q) + 1.0)
        start = ends[i]
    return 0.0


def _args(*values: Float64) -> List[Float64]:
    var a = List[Float64](length=NN_SCHED_ARGS, fill=0.0)
    for k in range(len(values)):  # small-loop(at most 8 argument words)
        a[k] = values[k]
    return a^


def _powi(b: Float64, k: Int) -> Float64:
    var r = Float64(1)
    for _ in range(k):  # small-loop(exponent)
        r *= b
    return r


def check_schedules() raises -> Int:
    var lr = Float32(0.1)
    var lr64 = Float64(lr)
    var n = 0
    var t_start = perf_counter_ns()
    print("NN_OPTIM schedule timing mark 0 ms", Float64(perf_counter_ns() - t_start) / 1e6)
    # multi_step: milestones 2, 5, 6; gamma 0.5 (the fit's form: epoch_lr's exponential at k)
    var ms = _args(2, 5, 6, -1, -1, -1, -1, -1)
    for e in range(9):
        var k = nn_multi_step_count(ms, e)
        var got = epoch_lr(lr, SCHED_EXPONENTIAL, 1, Float32(0.5), k)
        var want = lr64 * _powi(0.5, (1 if e >= 2 else 0) + (1 if e >= 5 else 0) + (1 if e >= 6 else 0))
        _close("multi_step epoch " + String(e), Float64(got), want, 1e-6)
        n += 1
    print("NN_OPTIM schedule timing mark 1 ms", Float64(perf_counter_ns() - t_start) / 1e6)
    # cosine: T_max 7, eta_min 0.01; past T_max the rate stays at eta_min
    var ca = _args(7, 0.01)
    for e in range(10):
        var want = 0.01 + (lr64 - 0.01) * (1.0 + cos(_PI * Float64(min(e, 7)) / 7.0)) / 2.0
        _close("cosine epoch " + String(e), Float64(nn_epoch_lr(NN_SCHED_COSINE, lr, ca, e)), want, 1e-6)
        n += 1
    print("NN_OPTIM schedule timing mark 2 ms", Float64(perf_counter_ns() - t_start) / 1e6)
    # warm restarts: T_0 3, T_mult 2 (cycles of 3, 6, 12), eta_min 0.001
    var wa = _args(3, 0.001, 2)
    for e in range(22):
        var cur = e
        var period = 3
        while cur >= period:
            cur -= period
            period *= 2
        var want = 0.001 + (lr64 - 0.001) * (1.0 + cos(_PI * Float64(cur) / Float64(period))) / 2.0
        _close("warm restarts epoch " + String(e), Float64(nn_epoch_lr(NN_SCHED_WARM_RESTARTS, lr, wa, e)), want, 1e-6)
        n += 1
    print("NN_OPTIM schedule timing mark 3 ms", Float64(perf_counter_ns() - t_start) / 1e6)
    # linear warmup 4 then decay to 0 at 15
    var la = _args(4, 15)
    var lv = nn_step_lrs(NN_SCHED_LINEAR_WARMUP, lr, la, 1, 18)
    for t in range(1, 19):
        var want = lr64 * Float64(t) / 4.0 if t <= 4 else (lr64 * (1.0 - Float64(t - 4) / 11.0) if t < 15 else 0.0)
        _close("linear_warmup step " + String(t), Float64(lv[t - 1]), want, 1e-6, 1e-9)
        n += 1
    print("NN_OPTIM schedule timing mark 4 ms", Float64(perf_counter_ns() - t_start) / 1e6)
    # one_cycle: cos, linear, three_phase; total 20; a window starting mid-run (the resumed fit's)
    for variant in range(3):
        var oa = _args(20, 0.3, 25.0, 1e4, 1.0 if variant == 1 else 0.0, 1.0 if variant == 2 else 0.0)
        var ov = nn_step_lrs(NN_SCHED_ONE_CYCLE, lr, oa, 1, 20)
        var tail = nn_step_lrs(NN_SCHED_ONE_CYCLE, lr, oa, 8, 13)
        for t in range(1, 21):
            var want = _torch_onecycle(t, 20, lr64, 0.3, 25.0, 1e4, variant == 1, variant == 2)
            _close("one_cycle " + String(variant) + " step " + String(t), Float64(ov[t - 1]), want, 1e-6, 1e-12)
            if t >= 8 and ov[t - 1] != tail[t - 8]:
                _fail("one_cycle: the resumed window's step " + String(t) + " differs")
            n += 1
    print("NN_OPTIM schedule timing mark 5 ms", Float64(perf_counter_ns() - t_start) / 1e6)
    var refused = False
    try:
        _ = nn_step_lrs(NN_SCHED_ONE_CYCLE, lr, _args(20, 0.3, 25.0, 1e4, 0, 0), 15, 7)
    except:
        refused = True
    if not refused:
        _fail("one_cycle: a fit past total_steps was not refused")
    print("NN_OPTIM schedule timing mark 6 ms", Float64(perf_counter_ns() - t_start) / 1e6)
    # reduce_on_plateau: factor 0.5, patience 1, threshold 0.1 (rel), cooldown 1, min_lr 0.02
    var pa = _args(0.5, 1, 0.1, 1, 0.02, 1e-8)
    var metrics: List[Float64] = [1.0, 0.95, 0.94, 0.93, 0.5, 0.49, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43]
    var pl = NnPlateau(lr64)
    var r_lr = lr64
    var r_best = Float64(1.0e308) * 10.0
    var r_bad = 0
    var r_cool = 0
    for e in range(len(metrics)):
        pl.epoch_end(metrics[e], pa)
        var m = metrics[e]
        if m < r_best * (1.0 - 0.1):
            r_best = m
            r_bad = 0
        else:
            r_bad += 1
        if r_cool > 0:
            r_cool -= 1
            r_bad = 0
        if r_bad > 1:
            var nl = max(r_lr * 0.5, 0.02)
            if r_lr - nl > 1e-8:
                r_lr = nl
            r_cool = 1
            r_bad = 0
        if pl.lr != r_lr or pl.bad != r_bad or pl.cool != r_cool:
            _fail("reduce_on_plateau epoch " + String(e) + ": lr " + String(pl.lr) + " want " + String(r_lr))
        n += 1
    if not (r_lr < lr64):
        _fail("reduce_on_plateau: the fixture never reduced the rate")
    return n


# ---------------------------------------------------------------- fits
comptime _FN = 96
comptime _FD = 4


def _fit_data() -> Tuple[List[Float32], List[Float32]]:
    var x = normal(_FN * _FD, 7)
    var y = List[Float32](length=_FN, fill=Float32(0))
    for r in range(_FN):  # small-loop(fixture rows)
        var a = Float64(x[r * _FD])
        var b = Float64(x[r * _FD + 1])
        var c = Float64(x[r * _FD + 2])
        y[r] = Float32(0.6 * a - 0.4 * b * c + 0.3 * Float64(x[r * _FD + 3]))
    return (x^, y^)


@fieldwise_init
struct _FitCase(Copyable, Movable):
    var name: String
    var opt: Int
    var flags: Int
    var lr: Float64
    var f1: Float64
    var f2: Float64
    var eps: Float64
    var wd: Float64
    var f7: Float64
    var sched: Int
    var gamma: Float64
    var args: List[Float64]
    var resume: Bool      # also check fit(3) + resume(3) == fit(6)
    var checkpoint: Bool  # also check fit(3), save, load, resume(3) == fit(6)


def _fit_cases() -> List[_FitCase]:
    var c = List[_FitCase]()
    var none = List[Float64]()
    var lb = NN_LAMB_GRAD_AVERAGING | NN_LAMB_BIAS_CORRECTION
    c.append(_FitCase("rmsprop", NN_OPT_RMSPROP, 0, 0.01, 0.0, 0.99, 1e-8, 0.0, 0.0, 0, 0.1, none.copy(), False, False))
    c.append(_FitCase("rmsprop momentum", NN_OPT_RMSPROP, 0, 0.005, 0.0, 0.99, 1e-8, 0.0, 0.5, 0, 0.1, none.copy(), False, False))
    c.append(_FitCase("rmsprop centered", NN_OPT_RMSPROP, 1, 0.01, 0.0, 0.99, 1e-8, 0.0, 0.0, 0, 0.1, none.copy(), True, False))
    c.append(_FitCase("adagrad", NN_OPT_ADAGRAD, 0, 0.1, 0.01, 0.1, 1e-10, 0.0, 0.0, 0, 0.1, none.copy(), True, False))
    c.append(_FitCase("lamb", NN_OPT_LAMB, lb | NN_LAMB_CLIP, 0.05, 0.9, 0.999, 1e-6, 0.01, 1.0, 0, 0.1, none.copy(), True, True))
    # schedules on adam (opt 1), lr 0.01
    c.append(_FitCase("multi_step", 1, 0, 0.02, 0.9, 0.999, 1e-8, 0.0, 0.0, NN_SCHED_MULTI_STEP, 0.5, _args(5, 9, -1, -1, -1, -1, -1, -1), False, False))
    c.append(_FitCase("cosine", 1, 0, 0.02, 0.9, 0.999, 1e-8, 0.0, 0.0, NN_SCHED_COSINE, 0.1, _args(12, 0.001), False, False))
    c.append(_FitCase("warm_restarts", 1, 0, 0.02, 0.9, 0.999, 1e-8, 0.0, 0.0, NN_SCHED_WARM_RESTARTS, 0.1, _args(3, 0.001, 2), True, False))
    c.append(_FitCase("linear_warmup", 1, 0, 0.02, 0.9, 0.999, 1e-8, 0.0, 0.0, NN_SCHED_LINEAR_WARMUP, 0.1, _args(4, 36), False, False))
    c.append(_FitCase("one_cycle", 1, 0, 0.03, 0.9, 0.999, 1e-8, 0.0, 0.0, NN_SCHED_ONE_CYCLE, 0.1, _args(36, 0.3, 25.0, 1e4, 0, 0), True, False))
    c.append(_FitCase("reduce_on_plateau", 1, 0, 0.02, 0.9, 0.999, 1e-8, 0.0, 0.0, NN_SCHED_PLATEAU, 0.1, _args(0.5, 0, 0.01, 0, 0.0, 1e-8), True, True))
    return c^


def _model(tc: _FitCase) -> SeqModel:
    var m = SeqModel()
    m.linear(16)
    m.act(L_TANH)
    m.linear(1)
    m.opt = tc.opt
    m.flags = tc.flags
    m.lr = tc.lr
    m.f1 = tc.f1
    m.f2 = tc.f2
    m.eps = tc.eps
    m.weight_decay = tc.wd
    m.f7 = tc.f7
    m.sched = tc.sched
    m.gamma = tc.gamma
    m.sched_args = tc.args.copy()
    return m^


def _finite_words(what: String, w: List[Float32]) raises:
    for i in range(len(w)):  # small-loop(checked words)
        var v = w[i]
        if v != v or abs(v) > Float32(1e30):
            _fail(what + ": word " + String(i) + " is not finite")


def _same(what: String, a: List[Float32], b: List[Float32]) raises:
    if len(a) != len(b):
        _fail(what + ": lengths differ")
    for i in range(len(a)):  # small-loop(compared words)
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            _fail(what + ": word " + String(i) + " differs")


def run_fit_case[E: Exec](mut ex: E, tc: _FitCase, tag: String) raises -> List[Float32]:
    var data = _fit_data()
    var x = data[0].copy()
    var y = data[1].copy()
    var m = _model(tc)
    _ = m.fit(ex, x, _FN, _FD, y, 1, 0, 12, 32, 1, 3)
    var curve = m.losses()
    _finite_words(tc.name + " losses", curve)
    _finite_words(tc.name + " params", m.params)
    if not (curve[len(curve) - 1] < Float32(0.25) * curve[0]):
        _fail(tc.name + ": the loss went from " + String(curve[0]) + " to " + String(curve[len(curve) - 1])
              + ", not below a quarter")
    var pred = m.predict(ex, x, _FN, _FD, OUT_RAW)
    _finite_words(tc.name + " predictions", pred)
    if tc.resume:
        var r = _model(tc)
        _ = r.fit(ex, x, _FN, _FD, y, 1, 0, 6, 32, 1, 3)
        _ = r.fit(ex, x, _FN, _FD, y, 1, 0, 6, -1, -1, 0, True)
        _same(tc.name + ": fit(6) + resume(6) against fit(12)", r.params, m.params)
        _same(tc.name + ": resumed optimizer state", r.m, m.m)
        _same(tc.name + ": resumed optimizer state 2", r.v, m.v)
    if tc.checkpoint:
        var r = _model(tc)
        _ = r.fit(ex, x, _FN, _FD, y, 1, 0, 6, 32, 1, 3)
        var file = String("/tmp/nn_optim_") + tag + "_" + String(tc.opt) + "_" + String(tc.sched) + ".ckpt"
        _ = r.save(file, _FD, False)
        var back = _model(tc)
        back.sched_args = List[Float64]()  # the file must carry the schedule's words
        back.f7 = 0.0                      # and f7
        back.load(file, _FD)
        _ = back.fit(ex, x, _FN, _FD, y, 1, 0, 6, -1, -1, 0, True)
        _same(tc.name + ": checkpoint + resume(6) against fit(12)", back.params, m.params)
    print("NN_OPTIM fit", tc.name, "loss", curve[0], "->", curve[len(curve) - 1])
    var words = m.params.copy()
    for i in range(len(pred)):  # small-loop(predictions)
        words.append(pred[i])
    _ = x^
    _ = y^
    return words^


def run_bn_state[E: Exec](mut ex: E, opt: Int, flags: Int, f1: Float64, f2: Float64, wd: Float64) raises -> List[Float32]:
    """A batch norm first, then flatten and linear(1): two one-batch epochs; returns the running statistics."""
    var C = 2
    var L = 8
    var rows = 12
    var x = normal(rows * C * L, 11)
    var y = normal(rows, 12)
    var m = SeqModel()
    m.sample = [C, L]
    m.layer(L_BATCH_NORM, C, Int(bitcast[DType.int32](Float32(0.1))), 1, 0, 0, 0, 1e-5)
    m.layer(L_FLATTEN)
    m.layer(L_LINEAR, 1, 1)
    m.opt = opt
    m.flags = flags
    m.lr = 0.05
    m.f1 = f1
    m.f2 = f2
    m.eps = 1e-6
    m.weight_decay = wd
    m.f7 = 1.0 if opt == NN_OPT_LAMB else 0.0
    _ = m.fit(ex, x, rows, C * L, y, 1, 0, 2, rows, 0, 5)
    var stats = List[Float32]()
    for i in range(2 * C, 4 * C):  # small-loop(running mean and var words)
        stats.append(m.params[i])
    if m.params[0] == Float32(1):
        _fail("batch-norm state: the weight did not train")
    _ = x^
    _ = y^
    return stats^


# ---------------------------------------------------------------- tests
def test_nn_optim_steps() raises:
    var cases = _step_cases()
    var n = len(cases)
    var failed = List[Int](length=n, fill=0)
    var host_words = List[List[Float32]](length=n, fill=List[Float32]())
    var fp = failed.unsafe_ptr()
    var wp = host_words.unsafe_ptr()
    var cp = cases.unsafe_ptr()
    def task(unit: Int) {imm fp, imm wp, imm cp}:
        try:
            var host = HostExec()
            wp[unit] = run_step_case(host, cp[unit])
        except error:
            print("FAIL NN_OPTIM step", unit, error)
            fp[unit] = 1
    host_parallelize(task, n)
    for u in range(n):  # small-loop(failure flags)
        if failed[u] != 0:
            _fail("step tc " + String(u) + " failed on the host column")
    comptime if not CPU_COLUMN_BUILD:
        for k in range(n):  # driver-loop: over the step cases on the one device
            var dev = DeviceExec()
            _same_words(cases[k].name, host_words[k], run_step_case(dev, cases[k]))
    _ = failed^
    _ = host_words^
    _ = cases^
    print("PASS NN_OPTIM steps", n)


def test_nn_optim_schedules() raises:
    var n = check_schedules()
    print("PASS NN_OPTIM schedules", n, "values")


def test_nn_optim_fits() raises:
    var cases = _fit_cases()
    var n = len(cases)
    var failed = List[Int](length=n, fill=0)
    var host_words = List[List[Float32]](length=n, fill=List[Float32]())
    var fp = failed.unsafe_ptr()
    var wp = host_words.unsafe_ptr()
    var cp = cases.unsafe_ptr()
    def task(unit: Int) {imm fp, imm wp, imm cp}:
        try:
            var host = HostExec()
            wp[unit] = run_fit_case(host, cp[unit], "host")
        except error:
            print("FAIL NN_OPTIM fit", unit, error)
            fp[unit] = 1
    host_parallelize(task, n)
    for u in range(n):  # small-loop(failure flags)
        if failed[u] != 0:
            _fail("fit tc " + String(u) + " failed on the host column")
    comptime if not CPU_COLUMN_BUILD:
        for k in range(n):  # driver-loop: over the fit cases on the one device
            var dev = DeviceExec()
            _same_words(cases[k].name, host_words[k], run_fit_case(dev, cases[k], "device"))
    _ = failed^
    _ = host_words^
    _ = cases^
    print("PASS NN_OPTIM fits", n)


def test_nn_optim_batch_norm_state() raises:
    var host = HostExec()
    var plain = run_bn_state(host, OPT_SGD, 0, 0.0, 0.0, 0.0)
    var lb = NN_LAMB_GRAD_AVERAGING | NN_LAMB_BIAS_CORRECTION | NN_LAMB_CLIP
    _same("batch-norm state under lamb", run_bn_state(host, NN_OPT_LAMB, lb, 0.9, 0.999, 0.5), plain)
    _same("batch-norm state under rmsprop", run_bn_state(host, NN_OPT_RMSPROP, 1, 0.0, 0.9, 0.1), plain)
    _same("batch-norm state under adagrad", run_bn_state(host, NN_OPT_ADAGRAD, 0, 0.0, 0.1, 0.1), plain)
    print("PASS NN_OPTIM batch-norm state untouched by lamb, rmsprop, adagrad")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
