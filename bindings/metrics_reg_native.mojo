# SPDX-License-Identifier: Apache-2.0
# The regression metrics of python/mojolearn/_expansion_metrics.py, one
# native call each: mean_squared_log_error, root_mean_squared_log_error,
# mean_absolute_percentage_error, mean_pinball_loss, median_absolute_error,
# max_error, explained_variance_score, mean_tweedie_deviance,
# mean_poisson_deviance, mean_gamma_deviance, d2_tweedie_score,
# d2_pinball_score and d2_absolute_error_score.
#
# This owner is the Python `_Reg` and `_Prog` it replaces: the same input
# refusals (wording and order), then the same x_metrics PROGRAM, built here
# with the same arena layout, the same stage records in the same order and
# the same parameter words, and run by the shared program runner
# (bindings/arena_prog_native.mojo `_metrics_run`, the one native call the
# Python `_run` made). The result words are read back from the arena as the
# Python `words(out, 2, "d")` read them. Same program, same kernels: the
# bits do not move.
from std.memory import memcpy, bitcast
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setattr, py_setitem, py_fields
from bindings.native_buffer_owner import native_buffer_view, native_storage_allocate
from bindings.classical_arrays import classical_address, classical_is_array, classical_dtype_str
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_validate import classical_view
from bindings.arena_prog_native import _metrics_run
from bindings.family_common import (
    FAMILY_F32P, FAMILY_F64P, family_repr, family_str, family_isinstance, family_call, family_call_kw,
    family_array, family_shape_repr, family_cast_view, family_cast_view_f64, family_min,
)
from core.family_cast_impl import family_all_finite_f32, family_max_f32

comptime _NONE = -1
comptime _PARAMS = 14
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


struct _Prog(Movable):
    """`_expansion_metrics._Prog`: an arena layout, the inputs copied into
    it, the stage records and the declared outputs, as the Python lists the
    shared runner reads."""
    var size: Int
    var inputs: PythonObject
    var stages: PythonObject
    var skip: PythonObject
    var outs: PythonObject
    var ops: PythonObject

    def __init__(out self, ops: PythonObject) raises:
        self.size = 0
        self.inputs = Python.list()
        self.stages = Python.list()
        self.skip = Python.list()
        self.outs = PythonObject(None)
        self.ops = ops

    def alloc(mut self, n: Int) raises -> Int:
        var off = self.size
        self.size += max(n, 0)
        if self.size > 2147483647:
            raise Error("mojolearn metrics: the problem exceeds the Int32 arena bound")
        return off

    def put(mut self, array: PythonObject) raises -> Int:
        """A float32 C-ordered array -> its offset."""
        var count = Int(py=array.size)
        var off = self.alloc(count)
        _ = py_call(self.inputs.append, Python.tuple(PythonObject(off), array))
        _ = py_call(self.skip.append, Python.tuple(PythonObject(off), PythonObject(off + count)))
        return off

    def put_i32(mut self, array: PythonObject) raises -> Int:
        """int32 words (read with ldi), copied in as their bits."""
        return self.put(array)

    def scratch(mut self, n: Int) raises -> Int:
        var off = self.alloc(n)
        if n > 0:
            _ = py_call(self.skip.append, Python.tuple(PythonObject(off), PythonObject(off + n)))
        return off

    def stage(mut self, op: String, total: Int, params: List[Int]) raises:
        if len(params) > _PARAMS:
            raise Error("ValueError: x_metrics: too many stage parameters")
        var record = Python.list()
        _ = py_call(record.append, self.ops[PythonObject(op)])
        _ = py_call(record.append, PythonObject(total))
        for i in range(len(params)):  # small-loop(stage words: at most 14 parameters)
            _ = py_call(record.append, PythonObject(params[i]))
        for _ in range(_PARAMS - len(params)):  # small-loop(stage words: the zero padding of one record)
            _ = py_call(record.append, PythonObject(0))
        _ = py_call(self.stages.append, record)

    def want(mut self, off: Int, n: Int) raises -> Int:
        if self.outs is None:
            self.outs = Python.list()
        if n > 0:
            _ = py_call(self.outs.append, Python.tuple(PythonObject(off), PythonObject(off + n), PythonObject(-1), PythonObject(1)))
        return off


def _f32_scalar_array(value: Float64, context: PythonObject) raises -> PythonObject:
    """`Array.from_list([_f32(value)], "<f4")`."""
    var one = family_array(Python.tuple(1), "<f4", context)
    FAMILY_F32P(unsafe_from_address=Int(py=classical_address(one)))[0] = Float32(value)
    return one


struct _Reg(Movable):
    """`_expansion_metrics._Reg`: validated (n, D) float32 targets, the
    weights, the multioutput rule, and the program being built."""
    var y: PythonObject
    var p: PythonObject
    var w: PythonObject
    var n: Int
    var d: Int
    var mo: String               # "raw_values", "uniform_average", "variance_weighted" or "custom"
    var custom: PythonObject     # the custom weights as an <f8 Array (mo == "custom")
    var prog: _Prog
    var y_off: Int
    var p_off: Int
    var s_off: Int
    var w_off: Int
    var key: Int
    var sw: Int

    def __init__(out self, y: PythonObject, p: PythonObject, n: Int, d: Int, w: PythonObject, mo: String,
                 custom: PythonObject, ops: PythonObject) raises:
        self.y = y
        self.p = p
        self.w = w
        self.n = n
        self.d = d
        self.mo = mo
        self.custom = custom
        self.prog = _Prog(ops)
        self.y_off = _NONE
        self.p_off = _NONE
        self.s_off = _NONE
        self.w_off = _NONE
        self.key = _NONE
        self.sw = _NONE

    def group(mut self, key: Int, n: Int, m: Int, values: Int, vstride: Int, weights: Int, width: Int) raises -> Int:
        """`_group`: the stable counting sort and the PairSums; returns OUT."""
        var off = self.prog.alloc(m + 1)
        var order = self.prog.scratch(n)
        var out = self.prog.alloc(m * width)
        self.prog.stage("group_sort", 1, [key, n, m, off, order])
        if m * width:
            self.prog.stage("group_sum", m * width, [off, order, values, vstride, weights, out, width])
        return out

    def program(mut self, scalar: Float64, context: PythonObject) raises:
        """`_Reg.program`: y, p, the kind's scalar (Float32), the weights,
        the one-group key and the weight total."""
        self.y_off = self.prog.put(self.y)
        self.p_off = self.prog.put(self.p)
        self.s_off = self.prog.put(_f32_scalar_array(scalar, context))
        if self.w is not None:
            self.w_off = self.prog.put(self.w)
        self.key = self.prog.scratch(self.n)
        self.prog.stage("pair_key", self.n, [0, 0, self.key, 1, 2])
        if self.w is not None:
            self.sw = self.group(self.key, self.n, 1, _NONE, 1, self.w_off, 1)

    def term(mut self, kind: Int, v: Int, p: Int, broadcast: Bool) raises -> Int:
        var out = self.prog.scratch(self.n * self.d)
        self.prog.stage("reg_term", self.n * self.d, [v, p, out, self.d, kind, self.s_off, 1 if broadcast else 0])
        return out

    def sums(mut self, v: Int) raises -> Int:
        return self.group(self.key, self.n, 1, v, self.d, self.w_off, self.d)

    def fin(mut self, sums: Int, mode: Int) raises -> Int:
        """reg_epi MEAN without the Float32 column: dst."""
        var dst = self.prog.alloc(2 * self.d)
        self.prog.stage("reg_epi", self.d, [_RE_MEAN, self.d, sums, self.sw, self.n, dst, _NONE, mode])
        return dst

    def fin32(mut self, sums: Int, mode: Int) raises -> Int:
        """reg_epi MEAN with the Float32 broadcast column: d32."""
        var dst = self.prog.alloc(2 * self.d)
        var d32 = self.prog.scratch(self.d)
        self.prog.stage("reg_epi", self.d, [_RE_MEAN, self.d, sums, self.sw, self.n, dst, d32, mode])
        return d32

    def stage_mean(mut self, kind: Int, mode: Int, v: Int, p: Int, broadcast: Bool) raises -> Int:
        return self.fin(self.sums(self.term(kind, v, p, broadcast)), mode)

    def percentile(mut self, v: Int, rank: Float64, f32: Bool, context: PythonObject) raises -> Int:
        """`_Reg.percentile` (average=True): dst, or d32 with f32."""
        var n = self.n
        var d = self.d
        var order = self.prog.scratch(n * d)
        self.prog.stage("col_sort", d, [v, n, d, order])
        var r = self.prog.put(_f32_scalar_array(rank, context))
        var out = self.prog.scratch(d)
        var cdf = self.prog.scratch(n * d)
        self.prog.stage("wpercentile", d, [v, n, d, order, self.w_off, r, 1, out, cdf])
        var dst = self.prog.alloc(2 * d)
        var d32 = self.prog.scratch(d) if f32 else _NONE
        self.prog.stage("reg_epi", d, [_RE_PCT, d, out, cdf, n, dst, d32])
        return d32 if f32 else dst

    def scan_sign(mut self, v: Int) raises -> Int:
        var flags = self.prog.alloc(2)
        self.prog.stage("reg_epi", self.n * self.d, [_RE_SIGN, self.n * self.d, v, flags])
        return self.prog.want(flags, 2)

    def scan_log_domain(mut self) raises -> Int:
        var total = self.n * self.d
        var flag = self.prog.alloc(1)
        if total:
            self.prog.stage("flag_scan", total, [self.y_off, total, _SCAN_LOG_DOMAIN, flag])
        self.prog.stage("flag_scan", total, [self.p_off, total, _SCAN_LOG_DOMAIN, flag])
        return self.prog.want(flag, 1)

    def average(mut self, vals: Int, nan_rule: Bool, variance: Int, context: PythonObject) raises -> Int:
        """`_Reg.average`: the values themselves (raw_values) or one reg_epi AVG unit."""
        var d = self.d
        if self.mo == "raw_values":
            return self.prog.want(vals, 2 * d)
        var mode = _RE_UNIFORM
        var w = _NONE
        if self.mo == "variance_weighted":
            mode = _RE_VARIANCE
            w = variance
        elif self.mo == "custom":
            mode = _RE_CUSTOM
            # `_words64`: the binary64 weights as their Int32 word pairs, low first.
            var words = family_array(Python.tuple(2 * d), "<i4", context)
            memcpy(dest=FAMILY_F64P(unsafe_from_address=Int(py=classical_address(words))),
                   src=FAMILY_F64P(unsafe_from_address=Int(py=classical_address(self.custom))), count=d)
            w = self.prog.put_i32(words)
        var dst = self.prog.alloc(2)
        self.prog.stage("reg_epi", 1, [_RE_AVG, d, vals, mode, w, dst, 1 if nan_rule else 0])
        return self.prog.want(dst, 2)

    def assemble(mut self, num: Int, den: Int, force_finite: Bool, average: Bool, context: PythonObject) raises -> Int:
        """`_assemble`: the per-output `1 - num / den` and their average."""
        var scores = self.prog.alloc(2 * self.d)
        self.prog.stage("reg_epi", self.d, [_RE_ASM, self.d, num, den, 1 if force_finite else 0, scores])
        if average:
            return self.average(scores, True, den, context)
        return scores


def _reg_run(prog: _Prog, mode: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_run(prog, numeric_mode)`: one `_Prog` object handed to the shared
    runner (the call the Python `_run` made); returns the arena store."""
    var prog_context = context["family_metrics_prog_context"]
    classical_connect_numpy(prog_context)
    var program = native_storage_allocate(prog_context["prog_class"])
    py_setattr(program, "size", PythonObject(prog.size))
    py_setattr(program, "_inputs", prog.inputs)
    py_setattr(program, "_stages", prog.stages)
    py_setattr(program, "arena", PythonObject(None))
    py_setattr(program, "_outs", prog.outs)
    py_setattr(program, "_skip", prog.skip)
    py_setattr(program, "rows", PythonObject(None))
    _ = _metrics_run(program, mode, prog_context)
    return py_fields(program)["arena"]


def _arena_base(arena: PythonObject) raises -> Int:
    return Int(py=py_call(arena.buffer_info)[0])


def _arena_double(arena: PythonObject, off: Int) raises -> Float64:
    """The binary64 value whose words are arena[off], arena[off + 1], low first."""
    var words = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=_arena_base(arena) + 4 * off)
    return bitcast[DType.float64](UInt64(words[0]) | (UInt64(words[1]) << 32))


def _arena_int(arena: PythonObject, off: Int) raises -> Int:
    return Int(MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=_arena_base(arena) + 4 * off)[0])


def _reg_result(r: _Reg, out: Int, arena: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_Reg.result`'s answer: a float, or the per-output float64 Array for raw_values."""
    if r.mo == "raw_values":
        var values = family_array(Python.tuple(r.d), "<f8", context)
        var dst = FAMILY_F64P(unsafe_from_address=Int(py=classical_address(values)))
        for i in range(r.d):  # driver-loop: over the metric's outputs
            dst[i] = _arena_double(arena, out + 2 * i)
        return values
    return PythonObject(_arena_double(arena, out))


# ---------------------------------------------------------------- validation


def _reg_target(source: PythonObject, name: String, caller: String, context: PythonObject) raises -> PythonObject:
    """One `_Reg` target: `materialize_f32_lists`, the float32 dtype rule,
    (n,) as (n, 1), the nonempty rank rule, C order, finite. Returns
    (array, n, n_outputs)."""
    var b = Python.import_module("builtins")
    var view = classical_view(source, context, name)
    var typestr = classical_dtype_str(view, context)
    if (typestr == "<f8" or typestr == "<i8") and not classical_is_array(source, context):
        var listed = py_type(source) is b.list or py_type(source) is b.tuple
        var buffered = True
        if not listed:
            try:
                _ = native_buffer_view(source)
            except:
                buffered = False
        if listed or not buffered:
            view = family_cast_view(view, name, context)
            typestr = "<f4"
    if typestr != "<f4":
        raise Error("TypeError: mojolearn " + caller + ": " + name + " must have dtype float32; cast explicitly before scoring")
    var nd = Int(py=view.ndim)
    var rows = Int(py=view.shape[0]) if nd >= 1 else 0
    var cols = 1
    if nd == 2:
        cols = Int(py=view.shape[1])
    if (nd != 1 and nd != 2) or rows == 0 or cols == 0:
        raise Error("mojolearn " + caller + ": " + name + " must be a nonempty (n,) or (n, n_outputs) array")
    # (n,) reads as (n, 1): the same C-ordered words.
    var flat = family_cast_view(view, name, context)
    if not family_all_finite_f32(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(flat))), rows * cols):
        raise Error("mojolearn " + caller + ": " + name + " contains NaN or infinity")
    _ = view^
    return Python.tuple(flat, PythonObject(rows), PythonObject(cols))


def _reg_weights(source: PythonObject, n: Int, caller: String, context: PythonObject) raises -> PythonObject:
    """`_metrics_impl._sample_weight_f32`: 1-D, n entries, finite and
    nonnegative, a positive total; None stays None."""
    if source is None:
        return PythonObject(None)
    var view = classical_view(source, context, "sample_weight")
    var nd = Int(py=view.ndim)
    if nd != 1:
        raise Error("mojolearn " + caller + ": sample_weight must be 1-D, got shape " + family_shape_repr(view))
    var count = Int(py=view.shape[0])
    if count != n:
        raise Error("mojolearn " + caller + ": sample_weight has " + String(count) + " entries for " + String(n) + " samples")
    var w = family_cast_view(view, "sample_weight", context)
    var words = FAMILY_F32P(unsafe_from_address=Int(py=classical_address(w)))
    if not family_all_finite_f32(words, n) or family_min(w) < 0:
        raise Error("mojolearn " + caller + ": sample_weight must have finite nonnegative entries")
    if not family_max_f32(words, n) > 0:
        raise Error("mojolearn " + caller + ": sample_weight must have positive total weight")
    _ = view^
    return w


def _reg_new(args: PythonObject, multioutput: PythonObject, caller: String, variance_ok: Bool,
             context: PythonObject) raises -> _Reg:
    """`_Reg(y_true, y_pred, sample_weight, multioutput, caller)` from
    args = (y_true, y_pred, sample_weight, ...)."""
    var b = Python.import_module("builtins")
    var y = _reg_target(args[0], "y_true", caller, context)
    var p = _reg_target(args[1], "y_pred", caller, context)
    var n = Int(py=y[1])
    var d = Int(py=y[2])
    if n != Int(py=p[1]) or d != Int(py=p[2]):
        raise Error("mojolearn " + caller + ": y_true and y_pred have different shapes ("
                    + String(n) + ", " + String(d) + ") and (" + String(Int(py=p[1])) + ", "
                    + String(Int(py=p[2])) + ")")
    var w = _reg_weights(args[2], n, caller, context)
    var mo = String("uniform_average")
    var custom = PythonObject(None)
    if family_isinstance(multioutput, b.str):
        var allowed = Python.tuple(PythonObject("raw_values"), PythonObject("uniform_average"))
        if variance_ok:
            allowed = Python.tuple(PythonObject("raw_values"), PythonObject("uniform_average"), PythonObject("variance_weighted"))
        if not py_contains(allowed, multioutput):
            raise Error("Allowed 'multioutput' string values are " + family_repr(allowed)
                        + ". You provided multioutput=" + family_repr(multioutput))
        mo = String(multioutput)
    elif multioutput is not None:
        # `[float(v) for v in flatten_mo(multioutput)]`, as binary64.
        var view = classical_view(multioutput, context, "multioutput")
        var values = family_cast_view_f64(view, "multioutput", context)
        var count = Int(py=values.size)
        if d == 1:
            raise Error("Custom weights are useful only in multi-output cases.")
        if count != d:
            raise Error("There must be equally many custom weights (" + String(count) + ") as outputs (" + String(d) + ").")
        mo = "custom"
        custom = values
    return _Reg(y[0], p[0], n, d, w, mo, custom, context["family_metrics_ops"])


def _metric_real(value: PythonObject, context: PythonObject) raises -> Bool:
    """A real number that is not a bool (`is_bool` or not numbers.Real)."""
    var b = Python.import_module("builtins")
    if family_isinstance(value, b.bool) or String(py_type(value).__name__) == "bool_":
        return False
    return family_isinstance(value, context["family_metrics_real"])


def _check_alpha(alpha: PythonObject, context: PythonObject) raises -> Float64:
    """`_expansion_metrics._check_alpha`: a real number in [0, 1]."""
    var b = Python.import_module("builtins")
    if not _metric_real(alpha, context):
        raise Error("alpha must be a real number in [0, 1]")
    var value = Float64(py=family_call(b.float, alpha))
    if not (0 <= value and value <= 1):
        raise Error("alpha must be a real number in [0, 1]")
    return value


def _undefined(message: String, context: PythonObject) raises:
    """`_undefined_warning`: scikit-learn's UndefinedMetricWarning when the
    caller imported sklearn.exceptions, else mojolearn's; attributed to the
    metric's caller."""
    var b = Python.import_module("builtins")
    var category = context["family_metrics_undefined"]
    var sk = py_get(Python.import_module("sys").modules, PythonObject("sklearn.exceptions"), PythonObject(None))
    if sk is not None:
        category = py_call(b.getattr, sk, PythonObject("UndefinedMetricWarning"), category)
    var keywords = Python.dict()
    py_setitem(keywords, "stacklevel", PythonObject(2))
    _ = family_call_kw(Python.import_module("warnings").warn, keywords, PythonObject(message), category)


# ---------------------------------------------------------------- the metrics


def _mean_error(args: PythonObject, kind: String, caller: String, root: Bool, scalar: Float64,
                log_domain: String, context: PythonObject) raises -> PythonObject:
    """`_mean_error`: the per-output (root) mean of the term, averaged, in
    one program; args = (y_true, y_pred, sample_weight, multioutput, mode)."""
    var r = _reg_new(args, args[3], caller, False, context)
    var terms = context["family_metrics_terms"]
    r.program(scalar, context)
    var flag = _NONE
    if log_domain.byte_length() > 0:
        flag = r.scan_log_domain()
    var mode = _RE_DIV | (_RE_ROOT if root else 0)
    var out = r.average(r.stage_mean(Int(py=terms[PythonObject(kind)]), mode, r.y_off, r.p_off, False), False, _NONE, context)
    var arena = _reg_run(r.prog, args[4], context)
    if flag != _NONE and _arena_int(arena, flag) != 0:
        raise Error(log_domain + " cannot be used when targets contain values less than or equal to -1.")
    return _reg_result(r, out, arena, context)


def _median_absolute_error(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """args = (y_true, y_pred, sample_weight, multioutput, mode)."""
    var r = _reg_new(args, args[3], "median_absolute_error", False, context)
    var terms = context["family_metrics_terms"]
    r.program(0.0, context)
    var t = r.term(Int(py=terms["abs"]), r.y_off, r.p_off, False)
    var out = r.average(r.percentile(t, 50.0, False, context), False, _NONE, context)
    var arena = _reg_run(r.prog, args[4], context)
    return _reg_result(r, out, arena, context)


def _max_error(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """args = (y_true, y_pred, mode): the largest |y - p| (single output)."""
    var r = _reg_new(Python.tuple(args[0], args[1], PythonObject(None)), PythonObject("uniform_average"),
                     "max_error", False, context)
    if r.d != 1:
        raise Error("Multioutput not supported in max_error")
    var terms = context["family_metrics_terms"]
    var prog = _Prog(context["family_metrics_ops"])
    var y = prog.put(r.y)
    var p = prog.put(r.p)
    var t = prog.scratch(r.n)
    prog.stage("reg_term", r.n, [y, p, t, 1, Int(py=terms["abs"]), y, 0])
    var out = prog.want(prog.alloc(1), 1)
    prog.stage("col_max", 1, [t, r.n, 1, out])
    var arena = _reg_run(prog, args[2], context)
    var word = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=_arena_base(arena) + 4 * out)[0]
    return PythonObject(Float64(word))


def _explained_variance(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """args = (y_true, y_pred, sample_weight, multioutput, force_finite, mode)."""
    var b = Python.import_module("builtins")
    var r = _reg_new(args, args[3], "explained_variance_score", True, context)
    var terms = context["family_metrics_terms"]
    var sq = Int(py=terms["sq"])
    r.program(0.0, context)
    # `_diff_and_y_means`, then `_centered` twice.
    var diff = r.term(Int(py=terms["diff"]), r.y_off, r.p_off, False)
    var diff_mean = r.fin32(r.sums(diff), _RE_DIV)
    var y_mean = r.fin32(r.sums(r.y_off), _RE_DIV)
    var num = r.fin(r.sums(r.term(sq, diff, diff_mean, True)), _RE_DIV)
    var den = r.fin(r.sums(r.term(sq, r.y_off, y_mean, True)), _RE_DIV)
    var out = r.assemble(num, den, Bool(py=py_call(b.bool, args[4])), True, context)
    var arena = _reg_run(r.prog, args[5], context)
    return _reg_result(r, out, arena, context)


def _tweedie_power(power: PythonObject, context: PythonObject) raises -> Float64:
    if not _metric_real(power, context):
        raise Error("power must be a real number")
    return Float64(py=family_call(Python.import_module("builtins").float, power))


def _tweedie_check(flags_y: Int, flags_p: Int, power: Float64, shown: String, arena: PythonObject) raises:
    """`_tweedie_domain`'s check, after the run."""
    var msg = "Mean Tweedie deviance error with power=" + shown + " can only be used on "
    var y_neg = _arena_int(arena, flags_y) != 0
    var y_nonpos = _arena_int(arena, flags_y + 1) != 0
    var p_nonpos = _arena_int(arena, flags_p + 1) != 0
    if power < 0:
        if p_nonpos:
            raise Error(msg + "strictly positive y_pred.")
    elif 1 <= power and power < 2:
        if y_neg or p_nonpos:
            raise Error(msg + "non-negative y and strictly positive y_pred.")
    elif power >= 2:
        if y_nonpos or p_nonpos:
            raise Error(msg + "strictly positive y and y_pred.")


def _tweedie(args: PythonObject, power_obj: PythonObject, caller: String, d2: Bool,
             context: PythonObject) raises -> PythonObject:
    """mean_tweedie_deviance (d2 False) or d2_tweedie_score (d2 True);
    args = (y_true, y_pred, sample_weight, mode)."""
    var power = _tweedie_power(power_obj, context)
    var r = _reg_new(Python.tuple(args[0], args[1], args[2]), PythonObject("uniform_average"), caller, False, context)
    if r.d != 1:
        raise Error("Multioutput not supported in " + caller)
    if d2 and r.n < 2:
        _undefined("D^2 score is not well-defined with less than two samples.", context)
        return py_call(Python.import_module("builtins").float, PythonObject("nan"))
    var terms = context["family_metrics_terms"]
    var tweedie = Int(py=terms["tweedie"])
    r.program(power, context)
    # `_tweedie_domain`: the power refusal now, the sign scans staged.
    if not (power <= 0 or power >= 1):
        raise Error("mojolearn " + caller + ": power in (0, 1) is not a Tweedie distribution")
    var fy = _NONE
    var fp = _NONE
    if power != 0:
        fy = r.scan_sign(r.y_off)
        fp = r.scan_sign(r.p_off)
    var out: Int
    if d2:
        var num = r.stage_mean(tweedie, _RE_DIV, r.y_off, r.p_off, False)
        var y_avg = r.fin32(r.sums(r.y_off), _RE_DIV)
        var den = r.stage_mean(tweedie, _RE_DIV, r.y_off, y_avg, True)
        out = r.assemble(num, den, False, False, context)
    else:
        out = r.stage_mean(tweedie, _RE_DIV, r.y_off, r.p_off, False)
    _ = r.prog.want(out, 2)
    var arena = _reg_run(r.prog, args[3], context)
    if fy != _NONE:
        _tweedie_check(fy, fp, power, family_str(power_obj), arena)
    return PythonObject(_arena_double(arena, out))


def _d2_pinball(args: PythonObject, alpha_obj: PythonObject, context: PythonObject) raises -> PythonObject:
    """d2_pinball_score; args = (y_true, y_pred, sample_weight, multioutput, mode)."""
    var alpha = _check_alpha(alpha_obj, context)
    var r = _reg_new(args, args[3], "d2_pinball_score", False, context)
    if r.n < 2:
        _undefined("D^2 score is not well-defined with less than two samples.", context)
        return py_call(Python.import_module("builtins").float, PythonObject("nan"))
    var terms = context["family_metrics_terms"]
    var pinball = Int(py=terms["pinball"])
    r.program(alpha, context)
    var num = r.stage_mean(pinball, _RE_DIV, r.y_off, r.p_off, False)
    var quant = r.percentile(r.y_off, alpha * 100, True, context)
    var den = r.stage_mean(pinball, _RE_DIV, r.y_off, quant, True)
    var out = r.assemble(num, den, True, True, context)
    var arena = _reg_run(r.prog, args[4], context)
    return _reg_result(r, out, arena, context)


def _r2_options(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`r2_score_options`: SS_res over SS_tot per output, one program;
    args = (y_true, y_pred, sample_weight, multioutput, force_finite, mode)."""
    var b = Python.import_module("builtins")
    var r = _reg_new(args, args[3], "r2_score", True, context)
    if r.n < 2:
        _undefined("R^2 score is not well-defined with less than two samples.", context)
        return py_call(b.float, PythonObject("nan"))
    var sq = Int(py=context["family_metrics_terms"]["sq"])
    r.program(0.0, context)
    # `_r2_parts`: the y mean (Float32 column), SS_res, SS_tot.
    var y_mean = r.fin32(r.sums(r.y_off), _RE_DIV)
    var num = r.fin(r.sums(r.term(sq, r.y_off, r.p_off, False)), 0)
    var den = r.fin(r.sums(r.term(sq, r.y_off, y_mean, True)), 0)
    var out = r.assemble(num, den, Bool(py=py_call(b.bool, args[4])), True, context)
    var arena = _reg_run(r.prog, args[5], context)
    return _reg_result(r, out, arena, context)


def metrics_reg_operation(operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """The regression metrics; each shell passes its arguments in its
    signature's order (see python/mojolearn/_expansion_metrics.py)."""
    classical_connect_numpy(context)
    if operation == "family_metric_regression_error":
        # args = (name, y_true, y_pred, sample_weight, multioutput, mode):
        # mean_squared_error / mean_absolute_error / root_mean_squared_error
        # with weights, multioutput or 2-D targets.
        var name = String(args[0])
        var rest = Python.tuple(args[1], args[2], args[3], args[4], args[5])
        if name == "mean_squared_error":
            return _mean_error(rest, "sq", name, False, 0.0, "", context)
        if name == "mean_absolute_error":
            return _mean_error(rest, "abs", name, False, 0.0, "", context)
        if name == "root_mean_squared_error":
            return _mean_error(rest, "sq", name, True, 0.0, "", context)
        raise Error("RuntimeError: unknown regression error metric " + family_repr(args[0]))
    if operation == "family_metric_r2_options":
        return _r2_options(args, context)
    if operation == "family_metric_msle":
        return _mean_error(args, "sqlog", "mean_squared_log_error", False, 0.0, "Mean Squared Logarithmic Error", context)
    if operation == "family_metric_rmsle":
        return _mean_error(args, "sqlog", "root_mean_squared_log_error", True, 0.0, "Root Mean Squared Logarithmic Error", context)
    if operation == "family_metric_mape":
        return _mean_error(args, "ape", "mean_absolute_percentage_error", False, 0.0, "", context)
    if operation == "family_metric_pinball":
        # args = (y_true, y_pred, sample_weight, multioutput, mode, alpha)
        var alpha = _check_alpha(args[5], context)
        return _mean_error(args, "pinball", "mean_pinball_loss", False, alpha, "", context)
    if operation == "family_metric_medae":
        return _median_absolute_error(args, context)
    if operation == "family_metric_max_error":
        return _max_error(args, context)
    if operation == "family_metric_explained_variance":
        return _explained_variance(args, context)
    if operation == "family_metric_tweedie":
        # args = (y_true, y_pred, sample_weight, mode, power)
        return _tweedie(args, args[4], "mean_tweedie_deviance", False, context)
    if operation == "family_metric_poisson":
        return _tweedie(args, PythonObject(1), "mean_tweedie_deviance", False, context)
    if operation == "family_metric_gamma":
        return _tweedie(args, PythonObject(2), "mean_tweedie_deviance", False, context)
    if operation == "family_metric_d2_tweedie":
        return _tweedie(args, args[4], "d2_tweedie_score", True, context)
    if operation == "family_metric_d2_pinball":
        # args = (y_true, y_pred, sample_weight, multioutput, mode, alpha)
        return _d2_pinball(args, args[5], context)
    if operation == "family_metric_d2_absolute":
        return _d2_pinball(args, PythonObject(0.5), context)
    raise Error("RuntimeError: unknown regression metric " + operation)
