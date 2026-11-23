# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The optimizers and learning-rate schedules `Sequential.compile` offers
beyond sgd / momentum / adam / adamw and constant / step / exponential
(lane nn-optim, 2026-11-23).

No new update or schedule math is written here. Every optimizer step is
the product's existing body:

  rmsprop, adagrad  sequence/ops.mojo op_opt (torch.optim.RMSprop and
                    Adagrad), through sequence/recurrent.mojo opt_step: ONE
                    OP_OPT launch over the flat parameter vector a step, the
                    host scalars (Adagrad's lr / (1 + (t - 1) lr_decay))
                    from opt_scalars, once a step.
  lamb              sequence/lamb_core.mojo lamb_core (timm's Lamb, the
                    public LAMB optimizer's body): per step, with the global
                    gradient clip on, 3 launches for the clip coefficient,
                    1 for the moments and the update direction, 2 for the
                    per-tensor trust ratios (only when weight_decay != 0 or
                    always_adapt) and 1 that updates every tensor: 7 at most,
                    nothing read back. Its buffers and its tensor table are
                    allocated and uploaded ONCE per fit (`NnLamb`).

and every schedule value is the product's existing schedule body, one host
scalar a step or an epoch (never per parameter):

  multi_step        autograd/model.mojo epoch_lr: lr gamma^k, k the
                    milestones passed (the exponential form at epoch k).
  cosine,           sequence/lr_table.mojo lr_spec_value (LR_COSINE, the
  cosine_warm_      public WarmupCosineLR's exact float32): eta_min +
  restarts          (lr - eta_min)(1 + cos(pi T_cur / T)) / 2.
  linear_warmup     lr_table LR_LINEAR (WarmupLinearLR), every step's value
                    filled ONCE per fit.
  one_cycle         lr_table LR_ONECYCLE (OneCycleLR), every step's value
                    filled ONCE per fit.
  reduce_on_        torch.optim.lr_scheduler.ReduceLROnPlateau (mode 'min',
  plateau           threshold_mode 'rel') on the monitored loss the fit
                    already computes each epoch (the validation loss with
                    validation_data, else the epoch's training loss): one
                    float64 rule an epoch on the host, no extra pass.

The schedule's eight argument words (`NN_SCHED_ARGS`, compile's
`sched_args`) by schedule:
  multi_step        the milestones (epochs, rising; -1 unused), up to 8
  cosine            [T_max, eta_min]
  cosine_warm_..    [T_0, eta_min, T_mult]
  linear_warmup     [warmup_steps, total_steps]
  one_cycle         [total_steps, pct_start, div_factor, final_div_factor,
                     linear (1) or cos (0), three_phase]
  reduce_on_plateau [factor, patience, threshold, cooldown, min_lr, eps]

The optimizer words (OptConfig): rmsprop f2 alpha, f7 momentum, flags bit0
centered; adagrad f1 lr_decay, f2 initial_accumulator_value; lamb f1 beta1,
f2 beta2, f7 max_grad_norm (0: no clip), flags timm's (bit0 trust_clip,
bit1 always_adapt, bit2 grad_averaging, bit3 bias_correction, bit4 clip).

This module imports nothing of autograd/model.mojo (which imports it).
"""
from std.memory import bitcast
from checks.numerics import ftz, identical_mul, pinned_mul_f64
from sequence.exec_trait import Exec
from sequence.ops import FP, OPT_ADAGRAD, OPT_ADAM, OPT_ADAMW, OPT_RMSPROP, OPT_SGD, dummy_ptr
from sequence.recurrent import OptConfig, OptState, opt_step
from sequence.lamb_core import lamb_core, lamb_table
from sequence.lr_table import LR_COSINE, LR_LINEAR, LR_ONECYCLE, LrSpec, lr_spec_fill, lr_spec_value

# ---------------------------------------------------------------- codes (this lane's group)
#: Sequential optimizer codes (python/mojolearn/_x_sequence_model.py _OPT): rmsprop and adagrad are
#: op_opt's own kinds; lamb runs lamb_core (no op_opt kind; 10 is clear of sequence/ops.mojo OPT_* 0..9)
comptime NN_OPT_RMSPROP = OPT_RMSPROP
comptime NN_OPT_ADAGRAD = OPT_ADAGRAD
comptime NN_OPT_LAMB = 10

#: Sequential schedule codes (_SCHED); 0..2 (constant, step, exponential) are autograd/model.mojo's
comptime NN_SCHED_MULTI_STEP = 3
comptime NN_SCHED_COSINE = 4
comptime NN_SCHED_WARM_RESTARTS = 5
comptime NN_SCHED_LINEAR_WARMUP = 6
comptime NN_SCHED_ONE_CYCLE = 7
comptime NN_SCHED_PLATEAU = 8
comptime NN_SCHED_FIRST = NN_SCHED_MULTI_STEP
comptime NN_SCHED_LAST = NN_SCHED_PLATEAU

#: argument words of a schedule (compile's sched_args, the fit's float tail, the checkpoint record)
comptime NN_SCHED_ARGS = 8

#: LAMB flag bits (sequence/ops_impl.mojo lamb_step_impl)
comptime NN_LAMB_TRUST_CLIP = 1
comptime NN_LAMB_ALWAYS_ADAPT = 2
comptime NN_LAMB_GRAD_AVERAGING = 4
comptime NN_LAMB_BIAS_CORRECTION = 8
comptime NN_LAMB_CLIP = 16


def nn_sched_per_step(sched: Int) -> Bool:
    """Whether the schedule's value changes every optimizer step (else every epoch)."""
    return sched == NN_SCHED_LINEAR_WARMUP or sched == NN_SCHED_ONE_CYCLE


def _whole(v: Float64) -> Bool:
    return v == v and Float64(Int(v)) == v


def _finite(v: Float64) -> Bool:
    return v == v and abs(v) <= 1.7976931348623157e308


# ---------------------------------------------------------------- admission (compile and fit call this one body)
def nn_optim_admit(
    kind: Int, flags: Int, f1: Float64, f2: Float64, eps: Float64, wd: Float64, f7: Float64,
    sched: Int, args: List[Float64],
) raises:
    """The refusals of this lane's optimizers and schedules, in words. The
    compile door and check_fit (autograd/model.mojo) both call it."""
    if not (_finite(f1) and _finite(f2) and _finite(eps) and _finite(wd) and _finite(f7)):
        raise Error("ValueError: the optimizer's hyperparameters must be finite numbers")
    if kind == NN_OPT_RMSPROP:
        if f2 < 0.0 or f2 >= 1.0:
            raise Error("ValueError: rmsprop's alpha must be in [0, 1)")
        if f7 < 0.0:
            raise Error("ValueError: rmsprop's momentum must be >= 0")
        if eps < 0.0 or wd < 0.0:
            raise Error("ValueError: rmsprop's eps and weight_decay must be >= 0")
        if (flags & 1) != 0 and f7 > 0.0:
            raise Error("ValueError: rmsprop with centered=True and momentum > 0 is not available yet "
                        + "(its third state buffer is not saved); use one of the two")
    elif kind == NN_OPT_ADAGRAD:
        if f1 < 0.0:
            raise Error("ValueError: adagrad's lr_decay must be >= 0")
        if f2 < 0.0:
            raise Error("ValueError: adagrad's initial_accumulator_value must be >= 0")
        if eps < 0.0 or wd < 0.0:
            raise Error("ValueError: adagrad's eps and weight_decay must be >= 0")
    elif kind == NN_OPT_LAMB:
        if f1 < 0.0 or f1 >= 1.0 or f2 < 0.0 or f2 >= 1.0:
            raise Error("ValueError: lamb's beta1 and beta2 must be in [0, 1)")
        if eps < 0.0 or wd < 0.0:
            raise Error("ValueError: lamb's eps and weight_decay must be >= 0")
        if f7 < 0.0:
            raise Error("ValueError: lamb's max_grad_norm must be >= 0 (None: no clip)")
    elif kind != OPT_SGD and kind != OPT_ADAM and kind != OPT_ADAMW:
        raise Error("ValueError: unknown optimizer code " + String(kind))
    if sched < NN_SCHED_FIRST:
        return
    if sched > NN_SCHED_LAST:
        raise Error("ValueError: unknown learning-rate schedule code " + String(sched))
    if len(args) != NN_SCHED_ARGS:
        raise Error("ValueError: the schedule needs its " + String(NN_SCHED_ARGS) + " argument words")
    for i in range(NN_SCHED_ARGS):  # small-loop(8: the schedule's argument words)
        if not _finite(args[i]):
            raise Error("ValueError: the schedule's arguments must be finite numbers")
    if sched == NN_SCHED_MULTI_STEP:
        var last = Float64(0)
        var used = 0
        for i in range(NN_SCHED_ARGS):  # small-loop(8 milestone words)
            var m = args[i]
            if m < 0.0:
                for j in range(i, NN_SCHED_ARGS):  # small-loop(8 milestone words)
                    if args[j] >= 0.0:
                        raise Error("ValueError: milestones must be listed first, then unused words")
                break
            if not _whole(m) or m < 1.0 or (used > 0 and m <= last):
                raise Error("ValueError: milestones must be rising whole epochs >= 1")
            last = m
            used += 1
        if used == 0:
            raise Error("ValueError: multi_step needs at least one milestone")
    elif sched == NN_SCHED_COSINE:
        if not _whole(args[0]) or args[0] < 1.0:
            raise Error("ValueError: cosine needs T_max, a whole number of epochs >= 1")
        if args[1] < 0.0:
            raise Error("ValueError: eta_min must be >= 0")
    elif sched == NN_SCHED_WARM_RESTARTS:
        if not _whole(args[0]) or args[0] < 1.0:
            raise Error("ValueError: cosine_warm_restarts needs T_0, a whole number of epochs >= 1")
        if not _whole(args[2]) or args[2] < 1.0:
            raise Error("ValueError: T_mult must be a whole number >= 1")
        if args[1] < 0.0:
            raise Error("ValueError: eta_min must be >= 0")
    elif sched == NN_SCHED_LINEAR_WARMUP:
        if not _whole(args[1]) or args[1] < 1.0:
            raise Error("ValueError: linear_warmup needs total_steps, a whole number of steps >= 1")
        if not _whole(args[0]) or args[0] < 0.0 or args[0] >= args[1]:
            raise Error("ValueError: warmup_steps must be a whole number from 0 to total_steps - 1")
    elif sched == NN_SCHED_ONE_CYCLE:
        if not _whole(args[0]) or args[0] < 1.0:
            raise Error("ValueError: one_cycle needs total_steps, a whole number of steps >= 1")
        if args[1] <= 0.0 or args[1] >= 1.0:
            raise Error("ValueError: pct_start must be in (0, 1)")
        if args[2] <= 0.0 or args[3] <= 0.0:
            raise Error("ValueError: div_factor and final_div_factor must be > 0")
        if (args[4] != 0.0 and args[4] != 1.0) or (args[5] != 0.0 and args[5] != 1.0):
            raise Error("ValueError: anneal_strategy must be 'cos' or 'linear' and three_phase a bool")
    elif sched == NN_SCHED_PLATEAU:
        if args[0] <= 0.0 or args[0] >= 1.0:
            raise Error("ValueError: reduce_on_plateau's factor must be in (0, 1)")
        if not _whole(args[1]) or args[1] < 0.0 or not _whole(args[3]) or args[3] < 0.0:
            raise Error("ValueError: patience and cooldown must be whole numbers >= 0")
        if args[2] < 0.0 or args[4] < 0.0 or args[5] < 0.0:
            raise Error("ValueError: threshold, min_lr and eps must be >= 0")


# ---------------------------------------------------------------- schedules
def _cosine_value(lr: Float32, eta_min: Float64, done: Int, period: Int) raises -> Float32:
    """eta_min + (lr - eta_min)(1 + cos(pi done / period)) / 2 (done = 0: lr), lr_table's exact float32."""
    if done <= 0:
        return lr
    var r = lr_spec_value(LrSpec(LR_COSINE, 0, period, 0, Float64(lr), eta_min, 0.0, 0.0), done)
    if r.status != 0:
        raise Error("model fit: the cosine schedule's value is out of float32 range (status "
                    + String(r.status) + ")")
    return r.value


def nn_multi_step_count(args: List[Float64], epoch: Int) -> Int:
    """The milestones passed by zero-based epoch `epoch` (torch MultiStepLR: lr gamma^k)."""
    var k = 0
    for i in range(NN_SCHED_ARGS):  # small-loop(8 milestone words)
        if args[i] >= 0.0 and Int(args[i]) <= epoch:
            k += 1
    return k


def nn_epoch_lr(sched: Int, lr: Float32, args: List[Float64], epoch: Int) raises -> Float32:
    """The per-epoch value of cosine and cosine_warm_restarts at zero-based
    epoch `epoch` (multi_step and plateau are autograd/model.mojo's)."""
    if sched == NN_SCHED_COSINE:
        # torch CosineAnnealingLR's closed form; past T_max the rate stays at eta_min
        return _cosine_value(lr, args[1], epoch, Int(args[0]))
    if sched == NN_SCHED_WARM_RESTARTS:
        # torch CosineAnnealingWarmRestarts at whole epochs: T_cur and T_i of the epoch's cycle
        var period = Int(args[0])
        var mult = Int(args[2])
        var cur = epoch
        while cur >= period:  # driver-loop: over the restarts before this epoch (logarithmic for T_mult > 1)
            cur -= period
            period *= mult
        return _cosine_value(lr, args[1], cur, period)
    return lr


def nn_step_lrs(sched: Int, lr: Float32, args: List[Float64], first: Int, count: Int) raises -> List[Float32]:
    """The rates of one-based optimizer steps first .. first + count - 1 of a
    per-step schedule, filled ONCE per fit (lr_table's lr_spec_fill)."""
    var rates = List[Float32](length=max(count, 1), fill=lr)
    if count < 1:
        return rates^
    var spec: LrSpec
    if sched == NN_SCHED_LINEAR_WARMUP:
        spec = LrSpec(LR_LINEAR, Int(args[0]), Int(args[1]), 0, Float64(lr), 0.0, 0.0, 0.0)
    else:
        var total = Int(args[0])
        if first + count - 1 > total:
            raise Error("ValueError: model fit: one_cycle's total_steps is " + String(total)
                        + " and this fit runs optimizer steps " + String(first) + " to "
                        + String(first + count - 1))
        spec = LrSpec(LR_ONECYCLE, 1 if args[4] != 0.0 else 0, 1 if args[5] != 0.0 else 0, total,
                      Float64(lr), args[2], args[3], args[1])
    var r = lr_spec_fill(spec, first, count, rates.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]())
    if r[0] != 0:
        raise Error("model fit: the schedule's value at step " + String(first + r[1])
                    + " is out of float32 range (status " + String(r[0]) + ")")
    return rates^


struct NnPlateau(Copyable, Movable):
    """torch ReduceLROnPlateau's state (mode 'min', threshold_mode 'rel'):
    the current rate (float64, as torch keeps it), the best metric, the bad
    epochs and the cooldown counter. Four words of the fit state."""
    var lr: Float64
    var best: Float64
    var bad: Int
    var cool: Int

    def __init__(out self, lr: Float64):
        self.lr = lr
        self.best = Float64(1.0e308) * Float64(10.0)  # +inf, torch's mode_worse
        self.bad = 0
        self.cool = 0

    def epoch_end(mut self, metric: Float64, args: List[Float64]):
        """torch's step(metrics): improve test rel(min) a < best (1 - threshold);
        the cooldown zeroes the bad count; past patience the rate becomes
        max(lr factor, min_lr) when that changes it by more than eps."""
        if metric < pinned_mul_f64(self.best, 1.0 - args[2]):
            self.best = metric
            self.bad = 0
        else:
            self.bad += 1
        if self.cool > 0:
            self.cool -= 1
            self.bad = 0
        if self.bad > Int(args[1]):
            var new_lr = max(pinned_mul_f64(self.lr, args[0]), args[4])
            if self.lr - new_lr > args[5]:
                self.lr = new_lr
            self.cool = Int(args[3])
            self.bad = 0


# ---------------------------------------------------------------- the step
struct NnLamb(Movable):
    """LAMB's buffers and tensor table, allocated and uploaded once per fit
    (sequence/ops_impl.mojo lamb_step_impl's, per call there). Empty for
    every other optimizer."""
    var on: Bool
    var nt: Int
    var nb: Int
    var tab: List[Float32]
    var U: FP
    var TAB: FP
    var partsA: FP
    var partsB: FP
    var nrm: FP
    var ratio: FP
    var scal: FP

    def __init__[E: Exec](out self, mut ex: E, cfg: OptConfig, offsets: List[Int], n: Int) raises:
        self.on = cfg.kind == NN_OPT_LAMB
        self.nt = 0
        self.nb = 0
        self.tab = List[Float32]()
        self.U = dummy_ptr()
        self.TAB = dummy_ptr()
        self.partsA = dummy_ptr()
        self.partsB = dummy_ptr()
        self.nrm = dummy_ptr()
        self.ratio = dummy_ptr()
        self.scal = dummy_ptr()
        if not self.on:
            return
        # the parameter tensors' offsets, rising strictly from 0 to n (the checkpoint layout's tensors,
        # autograd/model_checkpoint.mojo model_layout; a model the layout does not cover is one tensor)
        var offs = List[Int]()
        offs.append(0)
        for k in range(1, len(offsets)):  # driver-loop: over parameter tensors (description metadata)
            if offsets[k] > offs[len(offs) - 1] and offsets[k] <= n:
                offs.append(offsets[k])
        if offs[len(offs) - 1] < n:
            offs.append(n)
        self.nt = len(offs) - 1
        self.nb = lamb_table(offs, self.tab)
        self.TAB = ex.alloc(len(self.tab))
        ex.upload(self.TAB, FP(unsafe_from_address=Int(self.tab.unsafe_ptr())), len(self.tab))
        self.U = ex.alloc(n)
        self.partsA = ex.alloc(self.nb)
        self.partsB = ex.alloc(self.nb)
        self.nrm = ex.alloc(self.nt)
        self.ratio = ex.alloc(self.nt)
        self.scal = ex.alloc(1)
        var f = List[Float32](length=1, fill=Float32(1.0))
        ex.upload(self.scal, FP(unsafe_from_address=Int(f.unsafe_ptr())), 1)
        ex.sync()
        _ = f^

    def release[E: Exec](mut self, mut ex: E) raises:
        if not self.on:
            return
        ex.sync()
        ex.free(self.U)
        ex.free(self.TAB)
        ex.free(self.partsA)
        ex.free(self.partsB)
        ex.free(self.nrm)
        ex.free(self.ratio)
        ex.free(self.scal)
        self.on = False


def nn_opt_step[E: Exec](
    mut ex: E, cfg: OptConfig, mut st: OptState, t: Int, lr: Float32,
    P: FP, Gr: FP, s1: FP, s2: FP, s3: FP, n: Int, lamb: NnLamb,
) raises:
    """One optimizer step of a Sequential fit: lamb_core for LAMB, else opt_step's
    one OP_OPT launch. The two saved buffers are s1 and s2 (optimizer_m_,
    optimizer_v_): a centered RMSprop without momentum keeps its gradient
    average (op_opt's p4) in s1, its unread momentum slot pointing at s3."""
    if cfg.kind == NN_OPT_LAMB:
        var bc1 = Float32(1.0)
        var bc2 = Float32(1.0)
        if (cfg.flags & NN_LAMB_BIAS_CORRECTION) != 0:
            # lamb_bias's products, one a step, carried in the fit state (FS_PW1, FS_PW2)
            st.pw1 = ftz(identical_mul(st.pw1, cfg.f1))
            st.pw2 = ftz(identical_mul(st.pw2, cfg.f2))
            bc1 = Float32(1.0) - st.pw1
            bc2 = Float32(1.0) - st.pw2
        lamb_core(ex, P, Gr, s1, s2, lamb.U, lamb.TAB, lamb.partsA, lamb.partsB, lamb.nrm, lamb.ratio, lamb.scal,
                  n, lamb.nt, lamb.nb, cfg.flags, lr, cfg.f1, cfg.f2, cfg.eps, cfg.wd, cfg.f7, bc1, bc2)
    elif cfg.kind == NN_OPT_RMSPROP and (cfg.flags & 1) != 0:
        opt_step(ex, cfg, st, t, lr, P, Gr, s3, s2, s1, n)
    else:
        opt_step(ex, cfg, st, t, lr, P, Gr, s1, s2, s3, n)
