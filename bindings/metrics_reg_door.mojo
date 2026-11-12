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
#
# Lane metric-programs-2 (2026-11-12): renamed from metrics_reg_native.mojo.
# This is the door: UNPACK (the refusals, the admitted y / p / w arrays and
# the custom weights read once), the typed program (`_Reg` in
# bindings/metrics_reg_run.mojo, stage records through the typed op and term
# tables), the run with y / p / w as the only pins, and PACK.
from std.memory import unsafe_memcpy, bitcast
from bindings.admitted_family_door import admit_family, admit_family_f64
from bindings.native_loaded_module_door import native_loaded_module
from std.python import Python, PythonObject
from providers.python.metrics_vector_input_pyapi_unpack import metric_vector_exact
from providers.python.metrics_matrix_input_pyapi_unpack import metric_sample_weights
from providers.python.metrics_matrix_input_pyapi_pack import metric_reg_target
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.native_buffer_owner_pack import native_buffer_view
from bindings.classical_arrays_door import classical_address, classical_is_array, classical_dtype_str
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.classical_pandas_door import classical_admit_value
from bindings.metrics_native_support_door import metric_is_real, metric_is_bool, metric_float, metric_arguments, metric_undefined
from bindings.classical_validate_unpack import classical_view
from bindings.arena_prog_run import MetricArenaOwner
from bindings.metrics_program_door import MetricDoorProg, metric_door_run, metric_plan_run
from bindings.metrics_program_pack import metric_f64_array
from bindings.metrics_reg_run import (
    _Reg, _NONE, _RE_DIV, _RE_ROOT, reg_raw_values, tweedie_check,
)
from core.metrics_program_plan import metric_term
from bindings.family_common_unpack import (
    FAMILY_F32P,
    FAMILY_F64P,
    family_repr,
    family_str,
    family_isinstance,
    family_shape_repr,
    family_min,
)
from bindings.family_common_pack import (
    family_call,
    family_call_kw,
    family_array,
)
from core.family_cast_impl import family_max_f32
from core.host_finite_scan import host_all_finite_f32

# TOMBSTONE: `_Prog` (the third copy of the metric program builder, with a per-program Python opcode
# snapshot), the Python-holding `_Reg`, `_reg_run` and the `_arena_*` readers deleted 2026-11-12 by lane
# metric-programs-2: one door program (bindings/metrics_program_door.mojo `MetricDoorProg`), a typed `_Reg`
# (bindings/metrics_reg_run.mojo) and the typed reads on `MetricArenaOwner`; code recoverable at 4f5a5ae6d8.


def _reg_result(r: _Reg, out_buf: Int, arena: MetricArenaOwner, context: PythonObject) raises -> PythonObject:
    """PACK `_Reg.result`'s answer: a float, or the per-output float64 Array for raw_values."""
    if r.mo == "raw_values":
        return metric_f64_array(reg_raw_values(r, out_buf, arena), context)
    return PythonObject(arena.f64(out_buf))


# ---------------------------------------------------------------- validation


def _reg_target(source: PythonObject, name: String, caller: String, context: PythonObject) raises -> PythonObject:
    """One `_Reg` target: `materialize_f32_lists`, the float32 dtype rule,
    (n,) as (n, 1), the nonempty rank rule, C order, finite. Returns
    (array, n, n_outputs)."""
    if metric_vector_exact(source, context):
        return metric_reg_target(source, name, caller, context)
    var b = native_loaded_module("builtins")
    var view = classical_view(source, context, name)
    var typestr = classical_dtype_str(view, context)
    var cast = False
    if (typestr == "<f8" or typestr == "<i8") and not classical_is_array(source, context):
        var listed = py_type(source) is b.list or py_type(source) is b.tuple
        var buffered = True
        if not listed:
            try:
                _ = native_buffer_view(source)
            except:
                buffered = False
        if listed or not buffered:
            view = admit_family(view, -1, name, context, admitted=True, empty=True).array
            typestr = "<f4"
            cast = True
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
    # p3-labels-metrics: a list already cast above is the fresh C-order
    # float32 array this cast would return; it is not cast twice.
    var flat = view if cast else admit_family(view, -1, name, context, admitted=True, empty=True).array
    if not host_all_finite_f32(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(flat))), rows * cols):
        raise Error("mojolearn " + caller + ": " + name + " contains NaN or infinity")
    _ = flat  # keep the Python object alive through the native calls above
    _ = view^
    return Python.tuple(flat, PythonObject(rows), PythonObject(cols))


def _reg_weights(source: PythonObject, n: Int, caller: String, context: PythonObject) raises -> PythonObject:
    """`_metrics_impl._sample_weight_f32`: 1-D, n entries, finite and
    nonnegative, a positive total; None stays None."""
    if source is None:
        return PythonObject(None)
    if metric_vector_exact(source, context):
        return metric_sample_weights(source, n, caller, context)
    var view: PythonObject
    try:
        view = classical_view(source, context, "sample_weight")
    except error:
        # an empty or ragged list is refused by the sequence reader; the
        # refusal names the argument, as `_sample_weight_f32` did
        var text = String(error)
        if text.startswith("TypeError: "):
            raise error^
        raise Error("mojolearn " + caller + ": sample_weight: " + text)
    var nd = Int(py=view.ndim)
    if nd != 1:
        raise Error("mojolearn " + caller + ": sample_weight must be 1-D, got shape " + family_shape_repr(view))
    var count = Int(py=view.shape[0])
    if count != n:
        raise Error("mojolearn " + caller + ": sample_weight has " + String(count) + " entries for " + String(n) + " samples")
    var w = admit_family(view, -1, "sample_weight", context, admitted=True, empty=True).array
    var words = FAMILY_F32P(unsafe_from_address=Int(py=classical_address(w)))
    _ = w  # keep the Python object alive through the native call above
    if not host_all_finite_f32(words, n) or family_min(w) < 0:
        raise Error("mojolearn " + caller + ": sample_weight must have finite nonnegative entries")
    if not family_max_f32(words, n) > 0:
        raise Error("mojolearn " + caller + ": sample_weight must have positive total weight")
    _ = w  # keep the Python object alive through the native calls above
    _ = view^
    return w


def _reg_new(args: PythonObject, multioutput: PythonObject, caller: String, variance_ok: Bool,
             mut pins: List[PythonObject], context: PythonObject) raises -> _Reg:
    """UNPACK `_Reg(y_true, y_pred, sample_weight, multioutput, caller)` from
    args = (y_true, y_pred, sample_weight, ...): the typed `_Reg`, with the
    admitted y, p (and w) appended to `pins` in the program's input order."""
    var b = native_loaded_module("builtins")
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
    var custom = List[Float64]()
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
        var values = admit_family_f64(view, "multioutput", context)
        var count = Int(py=values.size)
        if d == 1:
            raise Error("Custom weights are useful only in multi-output cases.")
        if count != d:
            raise Error("There must be equally many custom weights (" + String(count) + ") as outputs (" + String(d) + ").")
        mo = "custom"
        # the custom weights read once (one buffer read): the program's own words
        var src = FAMILY_F64P(unsafe_from_address=Int(py=classical_address(values)))
        custom = List[Float64](capacity=d)
        for i in range(d):  # driver-loop: over the metric's outputs
            custom.append(src[i])
        _ = values  # keep the Python object alive through the native reads above
    pins.append(y[0])
    pins.append(p[0])
    if w is not None:
        pins.append(w)
    return _Reg(n, d, w is not None, mo, custom^)


def _metric_real(value: PythonObject, context: PythonObject) raises -> Bool:
    return metric_is_real(value, context) and not metric_is_bool(value, context)


def _check_alpha(alpha: PythonObject, context: PythonObject) raises -> Float64:
    """`_expansion_metrics._check_alpha`: a real number in [0, 1]."""
    var b = native_loaded_module("builtins")
    if not _metric_real(alpha, context):
        raise Error("alpha must be a real number in [0, 1]")
    var value = metric_float(alpha, context)
    if not (value >= 0 and value <= 1):
        raise Error("alpha must be a real number in [0, 1]")
    return value


def _undefined(message: String, context: PythonObject) raises:
    metric_undefined(message, context)


def _mean_error(args: PythonObject, kind: String, caller: String, root: Bool, scalar: Float64,
                log_domain: String, context: PythonObject) raises -> PythonObject:
    """`_mean_error`: the per-output (root) mean of the term, averaged, in
    one program; args = (y_true, y_pred, sample_weight, multioutput, mode)."""
    var pins = List[PythonObject]()
    var r = _reg_new(args, args[3], caller, False, pins, context)
    r.program(scalar)
    var flag = _NONE
    if log_domain.byte_length() > 0:
        flag = r.scan_log_domain()
    var mode = _RE_DIV | (_RE_ROOT if root else 0)
    var y_off = r.y_off
    var p_off = r.p_off
    var mean = r.stage_mean(metric_term(kind), mode, y_off, p_off, False)
    var out = r.average(mean, False, _NONE)
    var arena = metric_plan_run(r.plan, pins, args[4], context)
    if flag != _NONE and arena.i32(flag) != 0:
        raise Error(log_domain + " cannot be used when targets contain values less than or equal to -1.")
    return _reg_result(r, out, arena, context)


def _median_absolute_error(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """args = (y_true, y_pred, sample_weight, multioutput, mode)."""
    var pins = List[PythonObject]()
    var r = _reg_new(args, args[3], "median_absolute_error", False, pins, context)
    r.program(0.0)
    var y_off = r.y_off
    var p_off = r.p_off
    var t = r.term(metric_term("abs"), y_off, p_off, False)
    var median = r.percentile(t, 50.0, False)
    var out = r.average(median, False, _NONE)
    var arena = metric_plan_run(r.plan, pins, args[4], context)
    return _reg_result(r, out, arena, context)


def _max_error(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """args = (y_true, y_pred, mode): the largest |y - p| (single output)."""
    var pins = List[PythonObject]()
    var r = _reg_new(Python.tuple(args[0], args[1], PythonObject(None)), PythonObject("uniform_average"),
                     "max_error", False, pins, context)
    if r.d != 1:
        raise Error("Multioutput not supported in max_error")
    var prog = MetricDoorProg()
    var y = prog.put(pins[0])
    var p = prog.put(pins[1])
    var t = prog.scratch(r.n)
    prog.stage("reg_term", r.n, [y, p, t, 1, metric_term("abs"), y, 0])
    var out = prog.want(prog.alloc(1), 1)
    prog.stage("col_max", 1, [t, r.n, 1, out])
    var arena = metric_door_run(prog, args[2], context)
    # Read through the helper: `arena` is the only owner of the store, and
    # Mojo releases a local at its last named use, so a pointer taken from
    # `arena.base()` here would read freed words (different max_error bits
    # run to run on CUDA).
    return PythonObject(arena.f32(out))


def _explained_variance(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """args = (y_true, y_pred, sample_weight, multioutput, force_finite, mode)."""
    var b = native_loaded_module("builtins")
    var pins = List[PythonObject]()
    var r = _reg_new(args, args[3], "explained_variance_score", True, pins, context)
    var sq = metric_term("sq")
    r.program(0.0)
    var y_off = r.y_off
    var p_off = r.p_off
    # `_diff_and_y_means`, then `_centered` twice.
    var diff = r.term(metric_term("diff"), y_off, p_off, False)
    var diff_sums = r.sums(diff)
    var diff_mean = r.fin32(diff_sums, _RE_DIV)
    var y_sums = r.sums(y_off)
    var y_mean = r.fin32(y_sums, _RE_DIV)
    var num_terms = r.term(sq, diff, diff_mean, True)
    var num_sums = r.sums(num_terms)
    var num = r.fin(num_sums, _RE_DIV)
    var den_terms = r.term(sq, y_off, y_mean, True)
    var den_sums = r.sums(den_terms)
    var den = r.fin(den_sums, _RE_DIV)
    var out = r.assemble(num, den, Bool(py=py_call(b.bool, args[4])), True)
    var arena = metric_plan_run(r.plan, pins, args[5], context)
    return _reg_result(r, out, arena, context)


def _tweedie_power(power: PythonObject, context: PythonObject) raises -> Float64:
    if not _metric_real(power, context):
        raise Error("power must be a real number")
    return Float64(py=family_call(native_loaded_module("builtins").float, power))


def _tweedie(args: PythonObject, power_obj: PythonObject, caller: String, d2: Bool,
             context: PythonObject) raises -> PythonObject:
    """mean_tweedie_deviance (d2 False) or d2_tweedie_score (d2 True);
    args = (y_true, y_pred, sample_weight, mode)."""
    var power = _tweedie_power(power_obj, context)
    var pins = List[PythonObject]()
    var r = _reg_new(Python.tuple(args[0], args[1], args[2]), PythonObject("uniform_average"), caller, False, pins, context)
    if r.d != 1:
        raise Error("Multioutput not supported in " + caller)
    if d2 and r.n < 2:
        _undefined("D^2 score is not well-defined with less than two samples.", context)
        return py_call(native_loaded_module("builtins").float, PythonObject("nan"))
    var tweedie = metric_term("tweedie")
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
    var arena = metric_plan_run(r.plan, pins, args[3], context)
    if fy != _NONE:
        tweedie_check(fy, fp, power, family_str(power_obj), arena)
    return PythonObject(arena.f64(out))


def _d2_pinball(args: PythonObject, alpha_obj: PythonObject, context: PythonObject) raises -> PythonObject:
    """d2_pinball_score; args = (y_true, y_pred, sample_weight, multioutput, mode)."""
    var alpha = _check_alpha(alpha_obj, context)
    var pins = List[PythonObject]()
    var r = _reg_new(args, args[3], "d2_pinball_score", False, pins, context)
    if r.n < 2:
        _undefined("D^2 score is not well-defined with less than two samples.", context)
        return py_call(native_loaded_module("builtins").float, PythonObject("nan"))
    var pinball = metric_term("pinball")
    r.program(alpha)
    var y_off = r.y_off
    var p_off = r.p_off
    var num = r.stage_mean(pinball, _RE_DIV, y_off, p_off, False)
    var quant = r.percentile(y_off, alpha * 100, True)
    var den = r.stage_mean(pinball, _RE_DIV, y_off, quant, True)
    var out = r.assemble(num, den, True, True)
    var arena = metric_plan_run(r.plan, pins, args[4], context)
    return _reg_result(r, out, arena, context)


def _r2_options(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`r2_score_options`: SS_res over SS_tot per output, one program;
    args = (y_true, y_pred, sample_weight, multioutput, force_finite, mode)."""
    var b = native_loaded_module("builtins")
    var pins = List[PythonObject]()
    var r = _reg_new(args, args[3], "r2_score", True, pins, context)
    if r.n < 2:
        _undefined("R^2 score is not well-defined with less than two samples.", context)
        return py_call(b.float, PythonObject("nan"))
    var sq = metric_term("sq")
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
    var out = r.assemble(num, den, Bool(py=py_call(b.bool, args[4])), True)
    var arena = metric_plan_run(r.plan, pins, args[5], context)
    return _reg_result(r, out, arena, context)


def _regression_report(args: PythonObject, context: PythonObject) raises -> PythonObject:
    # The existing report's stage order and shared target loads are retained.
    # This is orchestration migration, not a new performance/default decision.
    var b = native_loaded_module("builtins")
    if py_type(args[5]) is not b.str or String(args[5]) != "identical":
        raise Error("regression_report requires numeric_mode='identical'")
    from bindings.host_inference_scalars_door import host_inference_scalar
    host_inference_scalar(args[4], context, True)
    var pins = List[PythonObject]()
    var r = _reg_new(args, args[3], "regression_report", False, pins, context)
    r.program(0.0)
    var y = r.y_off
    var p = r.p_off
    var squared = r.term(metric_term("sq"), y, p, False)
    var absolute = r.term(metric_term("abs"), y, p, False)
    var squared_sums = r.sums(squared)
    var squared_means = r.fin(squared_sums, _RE_DIV)
    var mse = r.average(squared_means, False, _NONE)
    var absolute_sums = r.sums(absolute)
    var absolute_means = r.fin(absolute_sums, _RE_DIV)
    var mae = r.average(absolute_means, False, _NONE)
    var y_sums = r.sums(y)
    var y_mean = r.fin32(y_sums, _RE_DIV)
    var centered = r.term(metric_term("sq"), y, y_mean, True)
    var centered_sums = r.sums(centered)
    var denominator = r.fin(centered_sums, 0)
    var numerator = r.fin(squared_sums, 0)
    var r2 = r.assemble(numerator, denominator, Bool(py=args[4]), True)
    var arena = metric_plan_run(r.plan, pins, args[5], context)
    var result = Python.dict()
    py_setitem(result, "mean_squared_error", _reg_result(r, mse, arena, context))
    py_setitem(result, "mean_absolute_error", _reg_result(r, mae, arena, context))
    if r.n < 2:
        _undefined("R^2 score is not well-defined with less than two samples.", context)
        py_setitem(result, "r2_score", py_call(b.float, PythonObject("nan")))
    else:
        py_setitem(result, "r2_score", _reg_result(r, r2, arena, context))
    return result


def metrics_reg_operation(operation: String, var args: PythonObject, context: PythonObject) raises -> PythonObject:
    """The regression metrics; each shell passes its arguments in its
    signature's order (see python/mojolearn/_expansion_metrics.py)."""
    classical_connect_numpy(context)
    classical_admit_value(args, context)
    args = metric_arguments(args, 0, context)
    if operation == "family_metric_regression_report":
        return _regression_report(args, context)
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
