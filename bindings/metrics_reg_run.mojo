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
from core.metrics_program_plan import MetricProgramPlan
from bindings.arena_prog_run import MetricArenaOwner

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
