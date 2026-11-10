# SPDX-License-Identifier: Apache-2.0
"""Typed public schedule preparation around the unchanged native LR kernels.

One schedule configuration is a fixed set of scalar words. Series execution
retains lr_spec_fill's existing block/exact operations; this module changes no
rounding, folding, table dispatch or kernel. Python owners stay in the boundary.
"""
from std.math import isfinite
from sequence.lr_table import LrSpec, F32P, LR_BEYOND, LR_ONE_BASED, lr_spec_fill


def optimizer_schedule_config_name(kind: Int) raises -> String:
    if kind == 0:
        return "constant"
    if kind == 1:
        return "linear"
    if kind == 2:
        return "cosine"
    raise Error("IndexError: tuple index out of range")


def optimizer_schedule_config_kind(name: String) raises -> Int:
    if name == "constant":
        return 0
    if name == "linear":
        return 1
    if name == "cosine":
        return 2
    raise Error("KeyError: unknown native schedule kind")


def optimizer_warmup_schedule(kind: Int, peak: Float64, low: Float64,
    warmup: Int, total: Int, has_total: Bool) raises -> LrSpec:
    var rounded_peak = Float64(Float32(peak))
    var rounded_low = Float64(Float32(low))
    if not isfinite(rounded_peak) or rounded_peak < 0:
        raise Error("ValueError: peak_lr must be finite and >= 0")
    if not isfinite(rounded_low) or rounded_low < 0:
        raise Error("ValueError: min_lr must be finite and >= 0")
    if warmup < 0:
        raise Error("ValueError: warmup_steps must be >= 0")
    if has_total and total < max(1, warmup):
        raise Error("ValueError: total_steps must be >= max(1, warmup_steps)")
    return LrSpec(kind, warmup, total if has_total else -1, 0, rounded_peak, rounded_low, 0.0, 0.0)


def optimizer_decay_schedule(kind: Int, base: Float64, step: Int, gamma: Float64,
    label: String) raises -> LrSpec:
    if step < 1:
        raise Error("ValueError: StepLR: step_size must be >= 1")
    if not isfinite(base) or base < 0:
        raise Error("ValueError: " + label + "base_lr must be finite and >= 0")
    if not isfinite(gamma) or gamma < 0:
        raise Error("ValueError: " + label + "gamma must be finite and >= 0")
    return LrSpec(20, step, 0, 0, base, gamma, 0.0, 0.0)


def optimizer_cycle_schedule(maximum: Float64, total: Int, pct: Float64,
    anneal: String, divide: Float64, final_divide: Float64) raises -> LrSpec:
    if total < 1:
        raise Error("ValueError: OneCycleLR: total_steps must be >= 1")
    if not (pct >= 0.0 and pct <= 1.0):
        raise Error("ValueError: OneCycleLR: pct_start must lie in [0, 1]")
    if anneal != "cos" and anneal != "linear":
        raise Error("ValueError: OneCycleLR: anneal_strategy must be 'cos' or 'linear'")
    if not isfinite(maximum) or not isfinite(divide) or not isfinite(final_divide):
        raise Error("ValueError: OneCycleLR: max_lr and division factors must be finite")
    if divide == 0 or final_divide == 0:
        raise Error("ZeroDivisionError: OneCycleLR: division factors must be nonzero")
    return LrSpec(10, 1 if anneal == "linear" else 0, 0, total, maximum, divide, final_divide, pct)


@fieldwise_init
struct OptimizerDecayReadback(Copyable, ImplicitlyCopyable, Movable):
    var base: Float64
    var gamma: Float64

    def spec(self, step: Int) -> LrSpec:
        return LrSpec(20, step, 0, 0, self.base, self.gamma, 0.0, 0.0)


def optimizer_decay_readback(base: Float64, gamma: Float64) raises -> OptimizerDecayReadback:
    # Readback permits rebound public fields exactly as before: construction
    # refuses negative values, while this operation refuses only non-finite.
    if not isfinite(base) or not isfinite(gamma):
        raise Error("ValueError: base_lr and gamma must be finite")
    return OptimizerDecayReadback(base, gamma)


def optimizer_schedule_refusal(status: Int, t: Int, style: Int, total_steps: Int) -> Error:
    if style == 2 and status == LR_BEYOND:
        return Error("ValueError: OneCycleLR: step " + String(t) + " is beyond total_steps " + String(total_steps)
                     + " + 1 (torch refuses it too)")
    if status == LR_ONE_BASED:
        if style == 0:
            return Error("ValueError: mojolearn schedule: step t is ONE-BASED, got " + String(t))
        return Error("ValueError: schedule step t is ONE-BASED")
    if status == 1:
        return Error("OverflowError: mojolearn: schedule value overflows float32")
    if status == 2:
        return Error("ArithmeticError: mojolearn: the exact schedule value straddles a float32 rounding "
                     "boundary within 1e-60; refusing to guess a bit")
    if status == 3:
        return Error("ValueError: mojolearn: cosine progress must be in [0, 1]")
    return Error("KeyError: " + String(status))


@fieldwise_init
struct OptimizerScheduleEvaluation(Copyable, ImplicitlyCopyable, Movable):
    var spec: LrSpec
    var first: Int
    var count: Int
    var style: Int

    def fill(self, destination: F32P) raises:
        var status = lr_spec_fill(self.spec, self.first, self.count, destination)
        if status[0] != 0:
            raise optimizer_schedule_refusal(status[0], self.first + status[1], self.style, self.spec.c)


def optimizer_schedule_evaluation(kind: Int, spec: LrSpec, requested: Int,
    series: Bool) -> OptimizerScheduleEvaluation:
    return OptimizerScheduleEvaluation(spec, 1 if series else requested,
        max(requested, 0) if series else 1, 0 if kind < 3 else (1 if kind < 5 else 2))
