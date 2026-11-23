# SPDX-License-Identifier: Apache-2.0
# RUN of the regression metrics (bindings/metrics_reg_door.mojo): the typed
# `_Reg` program builder and the post-run checks. No Python import
# (tools/python_free_dirs.sh).
#
# Lane metric-programs-2 (2026-11-12): `_Reg` held the validated y / p / w
# Arrays, the custom multioutput weights Array and a `_Prog` with a Python
# opcode snapshot. It is now typed: the shape (n, d), the multioutput rule,
# the custom weights as binary64 values read once at UNPACK, and the
# MetricProgramPlan (stage records through the typed op table). The door
# pins y, p and w in the order `program` declares them as inputs. Same
# arena layout, same records, same parameter words: the bits do not move.
from std.memory import bitcast
from core.metrics_program_plan import MetricProgramPlan, metric_term
from core.host_finite_scan import host_all_finite_f32
from bindings.arena_prog_run import MetricArenaOwner, MetricRoute, metric_inputs_run

comptime _NONE = -1
# reg_epi (op 60, x_metrics/reg_epi.mojo): kinds, MEAN modes, AVG modes.
comptime _RE_MEAN = 0
comptime _RE_PCT = 1
comptime _RE_ASM = 2
comptime _RE_AVG = 3
comptime _RE_SIGN = 4
comptime _RE_DIV = 1
comptime _RE_ROOT = 2
comptime _RE_UNIFORM = 1
comptime _RE_CUSTOM = 2
comptime _RE_VARIANCE = 3
# flag_scan tests (x_metrics/tail.mojo).
comptime _SCAN_LOG_DOMAIN = 0


struct _Reg(Movable):
    """`_expansion_metrics._Reg`: the validated (n, D) float32 targets'
    shape, whether weights are present, the multioutput rule and the program
    being built. The caller buffers are the door's pins (y, p, then w)."""
    var n: Int
    var d: Int
    var has_w: Bool
    var mo: String               # "raw_values", "uniform_average", "variance_weighted" or "custom"
    var custom: List[Float64]    # the custom weights (mo == "custom"), binary64
    var plan: MetricProgramPlan
    var y_off: Int
    var p_off: Int
    var s_off: Int
    var w_off: Int
    var key: Int
    var sw: Int

    def __init__(out self, n: Int, d: Int, has_w: Bool, mo: String, var custom: List[Float64]):
        self.n = n
        self.d = d
        self.has_w = has_w
        self.mo = String(mo)
        self.custom = custom^
        self.plan = MetricProgramPlan()
        self.y_off = _NONE
        self.p_off = _NONE
        self.s_off = _NONE
        self.w_off = _NONE
        self.key = _NONE
        self.sw = _NONE

    def group(mut self, key: Int, n: Int, m: Int, values: Int, vstride: Int, weights: Int, width: Int) raises -> Int:
        """`_group`: the stable counting sort and the PairSums; returns OUT."""
        return self.plan.group(key, n, m, values, vstride, weights, width)[1]

    def program(mut self, scalar: Float64) raises:
        """`_Reg.program`: y, p, the kind's scalar (Float32), the weights,
        the one-group key and the weight total. Inputs y, p (n * d words)
        and w (n words) are the door's pins, in that order."""
        self.y_off = self.plan.input(self.n * self.d)
        self.p_off = self.plan.input(self.n * self.d)
        self.s_off = self.plan.input_f32(scalar)
        if self.has_w:
            self.w_off = self.plan.input(self.n)
        self.key = self.plan.scratch(self.n)
        self.plan.stage_op("pair_key", self.n, [0, 0, self.key, 1, 2])
        if self.has_w:
            self.sw = self.group(self.key, self.n, 1, _NONE, 1, self.w_off, 1)

    def term(mut self, kind: Int, v: Int, p: Int, broadcast: Bool) raises -> Int:
        var out = self.plan.scratch(self.n * self.d)
        self.plan.stage_op("reg_term", self.n * self.d, [v, p, out, self.d, kind, self.s_off, 1 if broadcast else 0])
        return out

    def sums(mut self, v: Int) raises -> Int:
        return self.group(self.key, self.n, 1, v, self.d, self.w_off, self.d)

    def fin(mut self, sums: Int, mode: Int) raises -> Int:
        """reg_epi MEAN without the Float32 column: dst."""
        var dst = self.plan.alloc(2 * self.d)
        self.plan.stage_op("reg_epi", self.d, [_RE_MEAN, self.d, sums, self.sw, self.n, dst, _NONE, mode])
        return dst

    def fin32(mut self, sums: Int, mode: Int) raises -> Int:
        """reg_epi MEAN with the Float32 broadcast column: d32."""
        var dst = self.plan.alloc(2 * self.d)
        var d32 = self.plan.scratch(self.d)
        self.plan.stage_op("reg_epi", self.d, [_RE_MEAN, self.d, sums, self.sw, self.n, dst, d32, mode])
        return d32

    def stage_mean(mut self, kind: Int, mode: Int, v: Int, p: Int, broadcast: Bool) raises -> Int:
        var terms = self.term(kind, v, p, broadcast)
        var sums = self.sums(terms)
        return self.fin(sums, mode)

    def percentile(mut self, v: Int, rank: Float64, f32: Bool) raises -> Int:
        """`_Reg.percentile` (average=True): dst, or d32 with f32."""
        var n = self.n
        var d = self.d
        var order = self.plan.scratch(n * d)
        self.plan.stage_op("col_sort", d, [v, n, d, order])
        var r = self.plan.input_f32(rank)
        var out = self.plan.scratch(d)
        var cdf = self.plan.scratch(n * d)
        self.plan.stage_op("wpercentile", d, [v, n, d, order, self.w_off, r, 1, out, cdf])
        var dst = self.plan.alloc(2 * d)
        var d32 = self.plan.scratch(d) if f32 else _NONE
        self.plan.stage_op("reg_epi", d, [_RE_PCT, d, out, cdf, n, dst, d32])
        return d32 if f32 else dst

    def scan_sign(mut self, v: Int) raises -> Int:
        var flags = self.plan.alloc(2)
        self.plan.stage_op("reg_epi", self.n * self.d, [_RE_SIGN, self.n * self.d, v, flags])
        return self.plan.want(flags, 2, -1, 1)

    def scan_log_domain(mut self) raises -> Int:
        var total = self.n * self.d
        var flag = self.plan.alloc(1)
        if total:
            self.plan.stage_op("flag_scan", total, [self.y_off, total, _SCAN_LOG_DOMAIN, flag])
        self.plan.stage_op("flag_scan", total, [self.p_off, total, _SCAN_LOG_DOMAIN, flag])
        return self.plan.want(flag, 1, -1, 1)

    def average(mut self, vals: Int, nan_rule: Bool, variance: Int) raises -> Int:
        """`_Reg.average`: the values themselves (raw_values) or one reg_epi AVG unit."""
        var d = self.d
        if self.mo == "raw_values":
            return self.plan.want(vals, 2 * d, -1, 1)
        var mode = _RE_UNIFORM
        var w = _NONE
        if self.mo == "variance_weighted":
            mode = _RE_VARIANCE
            w = variance
        elif self.mo == "custom":
            mode = _RE_CUSTOM
            # `_words64`: the binary64 weights as their Int32 word pairs, low
            # first (program-owned words, read once at UNPACK)
            w = self.plan.input_f64s(self.custom)
        var dst = self.plan.alloc(2)
        self.plan.stage_op("reg_epi", 1, [_RE_AVG, d, vals, mode, w, dst, 1 if nan_rule else 0])
        return self.plan.want(dst, 2, -1, 1)

    def assemble(mut self, num: Int, den: Int, force_finite: Bool, average: Bool) raises -> Int:
        """`_assemble`: the per-output `1 - num / den` and their average."""
        var scores = self.plan.alloc(2 * self.d)
        self.plan.stage_op("reg_epi", self.d, [_RE_ASM, self.d, num, den, 1 if force_finite else 0, scores])
        if average:
            return self.average(scores, True, den)
        return scores


def reg_raw_values(r: _Reg, out_buf: Int, arena: MetricArenaOwner) -> List[Float64]:
    """The per-output binary64 values of a raw_values result (low word first)."""
    var values = List[Float64](capacity=r.d)
    for i in range(r.d):  # driver-loop: over the metric's outputs
        values.append(arena.f64(out_buf + 2 * i))
    return values^


def tweedie_check(flags_y: Int, flags_p: Int, power: Float64, shown: String, arena: MetricArenaOwner) raises:
    """`_tweedie_domain`'s check, after the run."""
    var msg = "Mean Tweedie deviance error with power=" + shown + " can only be used on "
    var y_neg = arena.i32(flags_y) != 0
    var y_nonpos = arena.i32(flags_y + 1) != 0
    var p_nonpos = arena.i32(flags_p + 1) != 0
    if power < 0:
        if p_nonpos:
            raise Error(msg + "strictly positive y_pred.")
    elif power >= 1 and power < 2:
        if y_neg or p_nonpos:
            raise Error(msg + "non-negative y and strictly positive y_pred.")
    elif power >= 2:
        if y_nonpos or p_nonpos:
            raise Error(msg + "strictly positive y and y_pred.")


# ---------------------------------------------------------------- the typed builders
# Lane mf-metric4 (2026-11-23): each regression metric's program assembly,
# moved word for word out of bindings/metrics_reg_door.mojo (`_mean_error`,
# `_median_absolute_error`, `_max_error`, `_explained_variance`, `_tweedie`,
# `_d2_pinball`, `_r2_options`): typed words in, the program out. The door's
# UNPACK reads the arrays and options once and calls them; a typed RUN (the
# search scorers, core/msel_scorer_run.mojo) calls the same builders over its
# own buffers. One body per metric: the same stages in the same order, the
# same arena layout, the same declared outputs, so the bits do not move.


def reg_mean_error_stages(mut r: _Reg, kind: Int, root: Bool, scalar: Float64,
                          log_domain: Bool) raises -> Tuple[Int, Int]:
    """`_mean_error`'s program: (the result offset, the log-domain flag or
    `_NONE`). `kind` is the `metric_term` code."""
    r.program(scalar)
    var flag = _NONE
    if log_domain:
        flag = r.scan_log_domain()
    var mode = _RE_DIV | (_RE_ROOT if root else 0)
    var y_off = r.y_off
    var p_off = r.p_off
    var mean = r.stage_mean(kind, mode, y_off, p_off, False)
    return (r.average(mean, False, _NONE), flag)


def reg_median_stages(mut r: _Reg, abs_term: Int) raises -> Int:
    """`_median_absolute_error`'s program: the result offset (`abs_term` is
    `metric_term("abs")`)."""
    r.program(0.0)
    var y_off = r.y_off
    var p_off = r.p_off
    var t = r.term(abs_term, y_off, p_off, False)
    var median = r.percentile(t, 50.0, False)
    return r.average(median, False, _NONE)


def reg_max_error_stages(mut plan: MetricProgramPlan, n: Int, abs_term: Int) raises -> Int:
    """`_max_error`'s program over its two inputs y and p (n words each, in
    that order): the Float32 result offset."""
    var y = plan.input(n)
    var p = plan.input(n)
    var t = plan.scratch(n)
    plan.stage_op("reg_term", n, [y, p, t, 1, abs_term, y, 0])
    var out = plan.want(plan.alloc(1), 1, -1, 1)
    plan.stage_op("col_max", 1, [t, n, 1, out])
    return out


def reg_explained_variance_stages(mut r: _Reg, sq: Int, diff: Int, force_finite: Bool) raises -> Int:
    """`_explained_variance`'s program: the result offset."""
    r.program(0.0)
    var y_off = r.y_off
    var p_off = r.p_off
    # `_diff_and_y_means`, then `_centered` twice.
    var d = r.term(diff, y_off, p_off, False)
    var diff_sums = r.sums(d)
    var diff_mean = r.fin32(diff_sums, _RE_DIV)
    var y_sums = r.sums(y_off)
    var y_mean = r.fin32(y_sums, _RE_DIV)
    var num_terms = r.term(sq, d, diff_mean, True)
    var num_sums = r.sums(num_terms)
    var num = r.fin(num_sums, _RE_DIV)
    var den_terms = r.term(sq, y_off, y_mean, True)
    var den_sums = r.sums(den_terms)
    var den = r.fin(den_sums, _RE_DIV)
    return r.assemble(num, den, force_finite, True)


@fieldwise_init
struct RegTweedie(ImplicitlyCopyable, Movable):
    """`_tweedie`'s program offsets: the result, the y and p sign flags
    (`_NONE` at power 0)."""
    var out: Int
    var fy: Int
    var fp: Int


def reg_tweedie_stages(mut r: _Reg, tweedie: Int, power: Float64, d2: Bool, caller: String) raises -> RegTweedie:
    """`_tweedie`'s program from `r.program(power)` on, with its power
    refusal in the door's words."""
    r.program(power)
    var y_off = r.y_off
    var p_off = r.p_off
    # `_tweedie_domain`: the power refusal now, the sign scans staged.
    if not (power <= 0 or power >= 1):
        raise Error("mojolearn " + caller + ": power in (0, 1) is not a Tweedie distribution")
    var fy = _NONE
    var fp = _NONE
    if power != 0:
        fy = r.scan_sign(y_off)
        fp = r.scan_sign(p_off)
    var out: Int
    if d2:
        var num = r.stage_mean(tweedie, _RE_DIV, y_off, p_off, False)
        var y_sums = r.sums(y_off)
        var y_avg = r.fin32(y_sums, _RE_DIV)
        var den = r.stage_mean(tweedie, _RE_DIV, y_off, y_avg, True)
        out = r.assemble(num, den, False, False)
    else:
        out = r.stage_mean(tweedie, _RE_DIV, y_off, p_off, False)
    _ = r.plan.want(out, 2, -1, 1)
    return RegTweedie(out, fy, fp)


def reg_d2_pinball_stages(mut r: _Reg, pinball: Int, alpha: Float64) raises -> Int:
    """`_d2_pinball`'s program (n >= 2): the result offset."""
    r.program(alpha)
    var y_off = r.y_off
    var p_off = r.p_off
    var num = r.stage_mean(pinball, _RE_DIV, y_off, p_off, False)
    var quant = r.percentile(y_off, alpha * 100, True)
    var den = r.stage_mean(pinball, _RE_DIV, y_off, quant, True)
    return r.assemble(num, den, True, True)


def reg_r2_stages(mut r: _Reg, sq: Int, force_finite: Bool) raises -> Int:
    """`_r2_options`' program (n >= 2): SS_res over SS_tot, the result offset."""
    r.program(0.0)
    # `_r2_parts`: the y mean (Float32 column), SS_res, SS_tot.
    var y_off = r.y_off
    var p_off = r.p_off
    var y_sums = r.sums(y_off)
    var y_mean = r.fin32(y_sums, _RE_DIV)
    var res_terms = r.term(sq, y_off, p_off, False)
    var res_sums = r.sums(res_terms)
    var num = r.fin(res_sums, 0)
    var tot_terms = r.term(sq, y_off, y_mean, True)
    var tot_sums = r.sums(tot_terms)
    var den = r.fin(tot_sums, 0)
    return r.assemble(num, den, force_finite, True)


# ---------------------------------------------------------------- one metric over typed buffers
#: `reg_metric_run`'s metrics (the search scorers' regression family).
comptime REG_R2 = 0
comptime REG_MSE = 1
comptime REG_MAE = 2
comptime REG_RMSE = 3
comptime REG_MSLE = 4
comptime REG_RMSLE = 5
comptime REG_MAPE = 6
comptime REG_MEDAE = 7
comptime REG_EXPLAINED_VARIANCE = 8
comptime REG_MAX_ERROR = 9
comptime REG_POISSON = 10
comptime REG_GAMMA = 11
comptime REG_D2_ABSOLUTE = 12


def reg_metric_caller(metric: Int) -> String:
    """The public function name each metric's refusals carry."""
    if metric == REG_R2:
        return "r2_score"
    if metric == REG_MSE:
        return "mean_squared_error"
    if metric == REG_MAE:
        return "mean_absolute_error"
    if metric == REG_RMSE:
        return "root_mean_squared_error"
    if metric == REG_MSLE:
        return "mean_squared_log_error"
    if metric == REG_RMSLE:
        return "root_mean_squared_log_error"
    if metric == REG_MAPE:
        return "mean_absolute_percentage_error"
    if metric == REG_MEDAE:
        return "median_absolute_error"
    if metric == REG_EXPLAINED_VARIANCE:
        return "explained_variance_score"
    if metric == REG_MAX_ERROR:
        return "max_error"
    if metric == REG_D2_ABSOLUTE:
        return "d2_pinball_score"
    return "mean_tweedie_deviance"


def reg_metric_run(route: MetricRoute, metric: Int, y: Int, p: Int, n: Int) raises -> Float64:
    """THE TYPED RUN of one single-output, unweighted regression metric with
    its default options over n float32 targets at `y` and n float32
    predictions at `p` (the caller keeps both alive): the door's refusals
    that read the values (non-finite words, the log and Tweedie domains), the
    builder's program through `metric_inputs_run`, the value. A metric of
    fewer than two samples that the door answers NaN with a warning answers
    NaN (the warning is the door's). No Python."""
    var caller = reg_metric_caller(metric)
    if n == 0:
        raise Error("mojolearn " + caller + ": y_true must be a nonempty (n,) or (n, n_outputs) array")
    if not host_all_finite_f32(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=y), n):
        raise Error("mojolearn " + caller + ": y_true contains NaN or infinity")
    if not host_all_finite_f32(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=p), n):
        raise Error("mojolearn " + caller + ": y_pred contains NaN or infinity")
    var addresses: List[Int] = [y, p]
    var sizes: List[Int] = [n, n]
    if metric == REG_MAX_ERROR:
        var plan = MetricProgramPlan()
        var out = reg_max_error_stages(plan, n, metric_term("abs"))
        var arena = metric_inputs_run(route, plan, addresses, sizes)
        return arena.f32(out)
    if (metric == REG_R2 or metric == REG_D2_ABSOLUTE) and n < 2:
        return _reg_nan()
    var r = _Reg(n, 1, False, "uniform_average", List[Float64]())
    var out = _NONE
    var flag = _NONE
    var tw = RegTweedie(_NONE, _NONE, _NONE)
    var power = 0.0
    if metric == REG_R2:
        out = reg_r2_stages(r, metric_term("sq"), True)
    elif metric == REG_MSE or metric == REG_RMSE:
        out = reg_mean_error_stages(r, metric_term("sq"), metric == REG_RMSE, 0.0, False)[0]
    elif metric == REG_MAE:
        out = reg_mean_error_stages(r, metric_term("abs"), False, 0.0, False)[0]
    elif metric == REG_MSLE or metric == REG_RMSLE:
        var got = reg_mean_error_stages(r, metric_term("sqlog"), metric == REG_RMSLE, 0.0, True)
        out = got[0]
        flag = got[1]
    elif metric == REG_MAPE:
        out = reg_mean_error_stages(r, metric_term("ape"), False, 0.0, False)[0]
    elif metric == REG_MEDAE:
        out = reg_median_stages(r, metric_term("abs"))
    elif metric == REG_EXPLAINED_VARIANCE:
        out = reg_explained_variance_stages(r, metric_term("sq"), metric_term("diff"), True)
    elif metric == REG_POISSON or metric == REG_GAMMA:
        power = 1.0 if metric == REG_POISSON else 2.0
        tw = reg_tweedie_stages(r, metric_term("tweedie"), power, False, caller)
        out = tw.out
    elif metric == REG_D2_ABSOLUTE:
        out = reg_d2_pinball_stages(r, metric_term("pinball"), 0.5)
    else:
        raise Error("RuntimeError: unknown regression metric " + String(metric))
    var arena = metric_inputs_run(route, r.plan, addresses, sizes)
    if flag != _NONE and arena.i32(flag) != 0:
        var shown = String("Mean Squared Logarithmic Error") if metric == REG_MSLE else String(
            "Root Mean Squared Logarithmic Error")
        raise Error(shown + " cannot be used when targets contain values less than or equal to -1.")
    if tw.fy != _NONE:
        tweedie_check(tw.fy, tw.fp, power, "1" if metric == REG_POISSON else "2", arena)
    var value = arena.f64(out)
    _ = r^
    return value


def _reg_nan() -> Float64:
    return bitcast[DType.float64](UInt64(0x7FF8000000000000))
