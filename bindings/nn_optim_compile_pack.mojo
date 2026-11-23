# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""UNPACK of `Sequential.compile`'s optimizer and schedule arguments (lane
nn-optim, 2026-11-23): the optimizer words (flags, f1, f2, eps,
weight_decay, f7) of every optimizer and the schedule's NN_SCHED_ARGS
argument words (autograd/nn_optim.mojo), each argument read once here at
compile and admitted by the one native admission body (nn_optim_admit)
that the fit calls again. Called by bindings/sequential_description_pack.mojo
sequential_compile; nothing here runs in a fit.

compile's positional options (python/mojolearn/_x_sequence_model.py):
  0 loss, 1 optimizer, 2 lr, 3 momentum, 4 nesterov, 5 beta1, 6 beta2,
  7 eps, 8 weight_decay, 9 schedule, 10 step_size, 11 gamma,
  12 huber_delta, 13 tol, 14 n_iter_no_change,
  15 reduction, 16 pos_weight, 17 beta, 18 blank, 19 zero_infinity
  (lanes nn-losses / nn-losses-2: read by sequential_compile),
  20 alpha, 21 centered, 22 lr_decay, 23 initial_accumulator_value,
  24 max_grad_norm, 25 trust_clip, 26 always_adapt,
  27 milestones, 28 T_max, 29 eta_min, 30 T_0, 31 T_mult,
  32 warmup_steps, 33 total_steps, 34 pct_start, 35 anneal_strategy,
  36 div_factor, 37 final_div_factor, 38 three_phase,
  39 factor, 40 patience, 41 threshold, 42 cooldown, 43 min_lr,
  44 cycle_momentum (one_cycle)
  (lane nn-integrate, 2026-11-23: the five loss options sit before the optimizer's).
"""
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_setitem
from bindings.sequential_keys_unpack import sequential_keys
from autograd.nn_optim import (
    NN_LAMB_BIAS_CORRECTION, NN_LAMB_CLIP, NN_LAMB_GRAD_AVERAGING, NN_SCHED_ARGS, NN_SCHED_COSINE,
    NN_SCHED_LINEAR_WARMUP, NN_SCHED_MULTI_STEP, NN_SCHED_ONE_CYCLE, NN_SCHED_PLATEAU, NN_SCHED_WARM_RESTARTS,
    nn_optim_admit,
)

comptime NN_COMPILE_OPTIONS = 45


def _f(b: PythonObject, v: PythonObject) raises -> Float64:
    return Float64(py=py_call(b.float, v))


def _or(b: PythonObject, v: PythonObject, default: Float64) raises -> Float64:
    return default if v is None else _f(b, v)


def _need(b: PythonObject, v: PythonObject, what: String) raises -> Float64:
    if v is None:
        raise Error("ValueError: " + what)
    return _f(b, v)


def nn_optim_compile_words(
    dst: PythonObject, options: PythonObject, optimizer: String, schedule_code: Int, b: PythonObject,
) raises:
    """Set dst's flags, f1, f2, eps, weight_decay, f7 and sched_args from
    compile's options, with each optimizer's PyTorch defaults for the
    arguments left None (momentum 0.9 for 'momentum', 0 for rmsprop; eps
    1e-8, adagrad 1e-10, lamb 1e-6; weight_decay 0, lamb 0.01), and admit
    them (autograd/nn_optim.mojo nn_optim_admit)."""
    if len(options) != NN_COMPILE_OPTIONS:
        raise Error("TypeError: compile takes " + String(NN_COMPILE_OPTIONS) + " option values")
    var nk = sequential_keys()
    var flags = 0
    var f1 = Float64(0.0)
    var f2 = Float64(0.0)
    var f7 = Float64(0.0)
    var eps = _or(b, options[7], 1e-8)
    var wd = _or(b, options[8], 0.0)
    var kind = 0
    if optimizer == "adam" or optimizer == "adamw":
        kind = 1 if optimizer == "adam" else 2
        f1 = _f(b, options[5])
        f2 = _f(b, options[6])
    elif optimizer == "momentum":
        flags = 1 if Bool(py=options[4]) else 0
        f1 = _or(b, options[3], 0.9)
    elif optimizer == "rmsprop":
        kind = 3
        flags = 1 if Bool(py=options[21]) else 0
        f2 = _f(b, options[20])
        f7 = _or(b, options[3], 0.0)
    elif optimizer == "adagrad":
        kind = 4
        eps = _or(b, options[7], 1e-10)
        f1 = _f(b, options[22])
        f2 = _f(b, options[23])
    elif optimizer == "lamb":
        kind = 10
        eps = _or(b, options[7], 1e-6)
        wd = _or(b, options[8], 0.01)
        f1 = _f(b, options[5])
        f2 = _f(b, options[6])
        flags = NN_LAMB_GRAD_AVERAGING | NN_LAMB_BIAS_CORRECTION
        if Bool(py=options[25]):
            flags |= 1
        if Bool(py=options[26]):
            flags |= 2
        if options[24] is not None:
            f7 = _f(b, options[24])
            if f7 > 0.0:
                flags |= NN_LAMB_CLIP
    var args = List[Float64](length=NN_SCHED_ARGS, fill=0.0)
    if schedule_code == NN_SCHED_MULTI_STEP:
        var ms = options[27]
        if ms is None or len(ms) < 1 or len(ms) > NN_SCHED_ARGS:
            raise Error("ValueError: multi_step needs milestones: 1 to 8 rising epochs")
        for k in range(NN_SCHED_ARGS):  # small-loop(8 milestone words)
            args[k] = _f(b, ms[k]) if k < len(ms) else -1.0
    elif schedule_code == NN_SCHED_COSINE:
        args[0] = _need(b, options[28], "cosine needs T_max (epochs)")
        args[1] = _f(b, options[29])
    elif schedule_code == NN_SCHED_WARM_RESTARTS:
        args[0] = _need(b, options[30], "cosine_warm_restarts needs T_0 (epochs)")
        args[1] = _f(b, options[29])
        args[2] = _f(b, options[31])
    elif schedule_code == NN_SCHED_LINEAR_WARMUP:
        args[0] = _f(b, options[32])
        args[1] = _need(b, options[33], "linear_warmup needs total_steps (optimizer steps)")
    elif schedule_code == NN_SCHED_ONE_CYCLE:
        # lane nn-integrate (2026-11-23): PyTorch's OneCycleLR also cycles the momentum (or Adam's beta1)
        # between 0.85 and 0.95 by default; this schedule cycles the learning rate only, so the default is
        # refused in words rather than giving different results silently
        if Bool(py=options[44]):
            raise Error("ValueError: one_cycle with cycle_momentum=True (PyTorch's default) is not available: "
                        + "this schedule cycles the learning rate only; pass cycle_momentum=False")
        var anneal = String(py=py_call(b.str, options[35]))
        if anneal != "cos" and anneal != "linear":
            raise Error("ValueError: anneal_strategy must be 'cos' or 'linear'")
        args[0] = _need(b, options[33], "one_cycle needs total_steps (optimizer steps)")
        args[1] = _f(b, options[34])
        args[2] = _f(b, options[36])
        args[3] = _f(b, options[37])
        args[4] = 1.0 if anneal == "linear" else 0.0
        args[5] = 1.0 if Bool(py=options[38]) else 0.0
    elif schedule_code == NN_SCHED_PLATEAU:
        args[0] = _f(b, options[39])
        args[1] = _f(b, options[40])
        args[2] = _f(b, options[41])
        args[3] = _f(b, options[42])
        args[4] = _f(b, options[43])
        args[5] = 1e-8
    nn_optim_admit(kind, flags, f1, f2, eps, wd, f7, schedule_code, args)
    py_setitem(dst, nk[].k_flags, PythonObject(flags))
    py_setitem(dst, nk[].k_f1, PythonObject(f1))
    py_setitem(dst, nk[].k_f2, PythonObject(f2))
    py_setitem(dst, nk[].k_eps, PythonObject(eps))
    py_setitem(dst, nk[].k_weight_decay, PythonObject(wd))
    py_setitem(dst, nk[].k_f7, PythonObject(f7))
    var words = Python.list()
    for k in range(NN_SCHED_ARGS):  # small-loop(8: the schedule's argument words)
        _ = py_call(words.append, PythonObject(args[k]))
    py_setitem(dst, nk[].k_sched_args, py_call(b.tuple, words))
