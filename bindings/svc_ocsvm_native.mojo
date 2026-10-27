# SPDX-License-Identifier: Apache-2.0
# OneClassSVM `fit`, `score_samples`, `decision_function`, `predict` and
# `fit_predict` (python/mojolearn/_expansion_neighbors.py), one native call
# each. The owner runs the x_neighbors primitives the Python `_XNeighbors`
# methods ran (`xn_pos_compact`, `xn_take_rows`, `xn_take_cols`,
# `xn_variance`, `xn_kernel`, `xn_ocsvm`, `x_neighbors_ocsvm_alpha_init`, the
# resident solve `x_neighbors_ocsvm_resident[_idn]`, `xn_kernel_matmul`,
# `xn_matmul`, `xn_unary`, `xn_p2m_sign_label`) with the same buffers and the
# same int and float parameter lists, in the same order, plus the base
# binding's `gather_i32` and `cast_elements`: the bits do not move.
from providers.python.x_neighbors_descriptor_pyapi import xn_admit_arguments, xn_execute, xn_module, xn_is_module
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem, py_fields
from bindings.classical_arrays import classical_address, classical_wrap_array
from bindings.classical_attributes import classical_connect_numpy
from bindings.family_common import (
    FAMILY_F32P, FAMILY_I32P, family_repr, family_isinstance, family_call, family_entry, family_array, family_min,
)
from core.family_weights_impl import family_fill_f32
from bindings.svc_common import (
    svc_builtins, svc_field, svc_has, svc_float, svc_int, svc_truthy, svc_base, svc_prefix, svc_scalar_array,
    svc_read_f32, svc_read_i32, svc_f32,
)

# `_expansion_neighbors._U_IDENTITY`
comptime _U_IDENTITY = 6


def _a(value: PythonObject) raises -> PythonObject:
    return classical_address(value)


def _op(binding: PythonObject, name: String, addrs: PythonObject, ints: PythonObject, floats: PythonObject,
        context: PythonObject) raises:
    """`_XNeighbors._op`: the binding's `xn_<name>(addresses, ints, floats)`."""
    xn_execute(binding, name, addrs, ints, floats)


def _f32_scalar(value: Float64) -> Float64:
    """A Python double rounded once to float32."""
    return Float64(Float32(value))


def _dims(*values: Int) raises -> PythonObject:
    """A Python list of ints."""
    var out = Python.list()
    for i in range(len(values)):  # small-loop(op ints: at most six parameters)
        _ = py_call(out.append, PythonObject(values[i]))
    return out


def _take_rows(binding: PythonObject, x: PythonObject, rows: PythonObject, count: Int,
               context: PythonObject) raises -> PythonObject:
    var cols = Int(py=x.shape[1])
    var out = family_array(Python.tuple(PythonObject(count), PythonObject(cols)), "<f4", context)
    _op(binding, "take_rows", Python.list(_a(x), _a(rows), _a(out)), _dims(count, cols, Int(py=x.shape[0])),
        Python.list(), context)
    _ = out  # keep the Python object alive through the native call above
    return out


def _take_cols(binding: PythonObject, x: PythonObject, cols: PythonObject, count: Int,
               context: PythonObject) raises -> PythonObject:
    var rows = Int(py=x.shape[0])
    var out = family_array(Python.tuple(PythonObject(rows), PythonObject(count)), "<f4", context)
    _op(binding, "take_cols", Python.list(_a(x), _a(cols), _a(out)), _dims(rows, Int(py=x.shape[1]), count),
        Python.list(), context)
    _ = out  # keep the Python object alive through the native call above
    return out


def _kernel(binding: PythonObject, a: PythonObject, b: PythonObject, code: Int, gamma: Float64, coef0: Float64,
            degree: Int, context: PythonObject) raises -> PythonObject:
    var n = Int(py=a.shape[0])
    var d = Int(py=a.shape[1])
    var m = Int(py=b.shape[0])
    var out = family_array(Python.tuple(PythonObject(n), PythonObject(m)), "<f4", context)
    _op(binding, "kernel", Python.list(_a(a), _a(b), _a(out)), _dims(n, m, d, code, degree),
        Python.list(PythonObject(_f32_scalar(gamma)), PythonObject(_f32_scalar(coef0))), context)
    _ = out  # keep the Python object alive through the native call above
    return out


def _i32_of(value: PythonObject, context: PythonObject) raises -> PythonObject:
    """`as_i32_c(value, ndim=1)` for the fitted int32 index Arrays."""
    if String(value.dtype) == "<i4":
        return value
    var b = svc_builtins()
    var store = family_call(context["array_type"], PythonObject("i"), family_call(b.map, b.int, family_call(value.tolist)))
    return classical_wrap_array(store, Python.tuple(PythonObject(len(store))), "<i4", "C", context)


def _fast_tier(binding: PythonObject, context: PythonObject) -> Bool:
    """`_XNeighbors._fast_tier`: the bound binary's compile-time numeric
    mode (`numeric_mode_used`) is 'fast'; False when it cannot be read."""
    try:
        if not xn_is_module(binding):
            return False
    except:
        pass
    try:
        var b = svc_builtins()
        var full = String(xn_module(binding).__name__)
        var parts = family_call(PythonObject(full).split, PythonObject("."))
        var name = String(parts[len(parts) - 1])
        var prefix = String("mojolearn") if name == "_mojolearn" else String(
            py_call(PythonObject(name).removeprefix, PythonObject("_mojolearn_")))
        var getter = py_call(b.getattr, xn_module(binding), PythonObject(prefix + "_numeric_mode"), PythonObject(None))
        if getter is not None:
            var mode = py_get(context["backend_fields"]["_CODE_MODE"], py_call(getter), PythonObject(None))
            return mode is not None and String(mode) == "fast"
        if len(parts) < 2:
            return False
        return String(parts[len(parts) - 2]) == "fast"
    except:
        return False


def _resolve_gamma(binding: PythonObject, gamma: PythonObject, x: PythonObject, context: PythonObject) raises -> Float64:
    """`_expansion_neighbors._resolve_gamma`."""
    var operator = Python.import_module("operator")
    var d = Int(py=x.shape[1])
    if gamma is None or Bool(py=family_call(operator.eq, gamma, PythonObject("auto"))):
        return 1.0 / Float64(d)
    if Bool(py=family_call(operator.eq, gamma, PythonObject("scale"))):
        var out = family_array(Python.tuple(PythonObject(1)), "<f4", context)
        _op(binding, "variance", Python.list(_a(x), _a(out)), _dims(Int(py=x.size)), Python.list(), context)
        _ = out  # keep the Python object alive through the native call above
        var var_value = svc_read_f32(out, 0)
        return 1.0 / (Float64(d) * var_value) if var_value != 0 else 1.0
    return svc_float(gamma)


def _ocsvm_fit(model: PythonObject, binding: PythonObject, source: PythonObject, sample_weight: PythonObject,
               context: PythonObject) raises:
    """`OneClassSVM.fit(X, y, sample_weight)`."""
    var b = svc_builtins()
    var fields = py_fields(model)
    var kernel = svc_field(fields, "kernel")
    var names = Python.tuple(PythonObject("linear"), PythonObject("poly"), PythonObject("rbf"), PythonObject("sigmoid"),
                             PythonObject("precomputed"))
    if Bool(py=family_call(b.callable, kernel)) or not py_contains(names, kernel):
        raise Error("NotImplementedError: OneClassSVM: kernel=" + family_repr(kernel) + " is not implemented")
    var nu = svc_float(svc_field(fields, "nu"))
    if not (0.0 < nu and nu <= 1.0):
        raise Error("nu must be in (0, 1]")
    var x = svc_f32(source, 2, "X", True, context)[0]
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var name = String(kernel)
    var precomputed = name == "precomputed"
    if precomputed and n != d:
        raise Error("Precomputed matrix must be a square matrix.")
    var rows = PythonObject(None)
    var cv: PythonObject
    var m = n
    if sample_weight is None:
        cv = family_array(Python.tuple(PythonObject(n)), "<f4", context)
        family_fill_f32(FAMILY_F32P(unsafe_from_address=Int(py=_a(cv))), n, Float32(1.0))
    else:
        var wv = svc_f32(sample_weight, 1, "sample_weight", False, context)[0]
        if Int(py=wv.shape[0]) != n:
            raise Error("sample_weight and X have different numbers of samples")
        if n > 0 and family_min(wv) < 0:
            raise Error("negative sample_weight is not supported")
        # libsvm's remove_zero_weight as a device compaction (pos_compact)
        var prow = family_array(Python.tuple(PythonObject(n)), "<i4", context)
        var pval = family_array(Python.tuple(PythonObject(n)), "<f4", context)
        var pinfo = family_array(Python.tuple(PythonObject(1)), "<i4", context)
        _op(binding, "pos_compact", Python.list(_a(wv), _a(prow), _a(pval), _a(pinfo)), _dims(n), Python.list(), context)
        _ = pinfo  # keep the Python object alive through the native call above
        _ = pval  # keep the Python object alive through the native call above
        _ = prow  # keep the Python object alive through the native call above
        _ = wv  # keep the Python object alive through the native call above
        var kept = svc_read_i32(pinfo, 0)
        if kept == 0:
            raise Error("Invalid input - all samples have zero or negative weights.")
        rows = svc_prefix(prow, kept, Python.tuple(PythonObject(kept)), "<i4", context)
        cv = svc_prefix(pval, kept, Python.tuple(PythonObject(kept)), "<f4", context)
        m = kept
        _ = wv^
    var q = PythonObject(None)
    var xw = x
    var res_fn = PythonObject(None)
    var degree = 0
    var coef0 = Float64(0.0)
    if precomputed:
        py_setitem(fields, PythonObject("_gamma"), PythonObject(0.0))
        if rows is None:
            q = x
        else:
            q = _take_cols(binding, _take_rows(binding, x, rows, m, context), rows, m, context)
    else:
        degree = svc_int(svc_field(fields, "degree"))
        coef0 = svc_float(svc_field(fields, "coef0"))
        var gamma = _f32_scalar(_resolve_gamma(binding, svc_field(fields, "gamma"), x, context))
        py_setitem(fields, PythonObject("_gamma"), PythonObject(gamma))
        if rows is not None:
            xw = _take_rows(binding, x, rows, m, context)
        # the resident solve: the Gram on the device, solved there
        var entry = "x_neighbors_ocsvm_resident" if _fast_tier(binding, context) else "x_neighbors_ocsvm_resident_idn"
        res_fn = py_call(b.getattr, xn_module(binding), PythonObject(entry), PythonObject(None))
        if res_fn is None:
            q = _kernel(binding, xw, xw, Int(py=context["family_svc_xn_kernels"][kernel]), gamma, coef0, degree, context)
    # libsvm's start (solve_one_class), three device launches
    var alpha = family_array(Python.tuple(PythonObject(m)), "<f4", context)
    _ = family_call(family_entry(xn_module(binding), "x_neighbors_ocsvm_alpha_init", context), Python.list(_a(cv), _a(alpha)),
                    _dims(m), Python.list(PythonObject(nu)))
    _ = alpha  # keep the Python object alive through the native call above
    var info = family_array(Python.tuple(PythonObject(1)), "<f4", context)
    var iters = family_array(Python.tuple(PythonObject(1)), "<i4", context)
    var max_iter = svc_int(svc_field(fields, "max_iter"))
    var cap = 10000000 if max_iter < 0 else max_iter
    var tol = _f32_scalar(svc_float(svc_field(fields, "tol")))
    if q is None:
        _ = family_call(res_fn, Python.list(_a(xw), _a(cv), _a(alpha), _a(info), _a(iters)),
                        _dims(m, d, Int(py=context["family_svc_xn_kernels"][kernel]), degree, cap),
                        Python.list(PythonObject(svc_float(svc_field(fields, "_gamma"))), PythonObject(_f32_scalar(coef0)),
                                    PythonObject(tol)))
        _ = iters  # keep the Python object alive through the native call above
        _ = info  # keep the Python object alive through the native call above
        _ = alpha  # keep the Python object alive through the native call above
        _ = xw  # keep the Python object alive through the native call above
    else:
        _op(binding, "ocsvm", Python.list(_a(q), _a(cv), _a(alpha), _a(info), _a(iters)), _dims(m, cap),
            Python.list(PythonObject(tol)), context)
        _ = iters  # keep the Python object alive through the native call above
        _ = info  # keep the Python object alive through the native call above
        _ = alpha  # keep the Python object alive through the native call above
        _ = q  # keep the Python object alive through the native call above
    var rho = svc_read_f32(info, 0)
    # the support (alpha > 0, in order) and its coefficients by pos_compact
    var srow = family_array(Python.tuple(PythonObject(max(m, 1))), "<i4", context)
    var sval = family_array(Python.tuple(PythonObject(max(m, 1))), "<f4", context)
    var sinfo = family_array(Python.tuple(PythonObject(1)), "<i4", context)
    if m > 0:
        _op(binding, "pos_compact", Python.list(_a(alpha), _a(srow), _a(sval), _a(sinfo)), _dims(m), Python.list(), context)
        _ = sinfo  # keep the Python object alive through the native call above
        _ = sval  # keep the Python object alive through the native call above
        _ = srow  # keep the Python object alive through the native call above
        _ = alpha  # keep the Python object alive through the native call above
    var n_sv = svc_read_i32(sinfo, 0) if m > 0 else 0
    var support = svc_prefix(srow, n_sv, Python.tuple(PythonObject(n_sv)), "<i4", context)
    if rows is not None and n_sv > 0:
        var mapped = family_array(Python.tuple(PythonObject(n_sv)), "<i4", context)
        _ = family_call(svc_base("gather_i32", context), _a(rows), PythonObject(m), _a(support), PythonObject(n_sv),
                        _a(mapped))
        _ = support  # keep the Python object alive through the native call above
        _ = rows  # keep the Python object alive through the native call above
        _ = mapped  # keep the Python object alive through the native call above
        support = mapped
    py_setitem(fields, PythonObject("support_"), support)
    if precomputed:
        py_setitem(fields, PythonObject("support_vectors_"),
                   family_array(Python.tuple(PythonObject(n_sv), PythonObject(0)), "<f4", context))
    else:
        py_setitem(fields, PythonObject("support_vectors_"), _take_rows(binding, x, support, n_sv, context))
    py_setitem(fields, PythonObject("dual_coef_"),
               svc_prefix(sval, n_sv, Python.tuple(PythonObject(1), PythonObject(n_sv)), "<f4", context))
    py_setitem(fields, PythonObject("intercept_"), svc_scalar_array(-rho, "<f4", context))
    py_setitem(fields, PythonObject("offset_"), PythonObject(rho))
    py_setitem(fields, PythonObject("n_iter_"), PythonObject(svc_read_i32(iters, 0)))
    py_setitem(fields, PythonObject("n_features_in_"), PythonObject(d))
    py_setitem(fields, PythonObject("fit_status_"), PythonObject(0))
    _ = x^
    _ = xw^
    _ = q^
    _ = cv^
    _ = rows^
    _ = alpha^


def _ocsvm_scores(model: PythonObject, binding: PythonObject, source: PythonObject,
                  context: PythonObject) raises -> PythonObject:
    """`OneClassSVM.score_samples`: (n,) float32."""
    var fields = py_fields(model)
    var q = svc_f32(source, 2, "X", True, context)[0]
    var dual = svc_field(fields, "dual_coef_")
    var n_sv = Int(py=dual.shape[1])
    # the (n_sv, 1) column of the C-order float32 coefficients
    var coef = svc_f32(dual, 2, "dual_coef_", False, context)[0]
    var kernel = svc_field(fields, "kernel")
    var precomputed = String(kernel) == "precomputed"
    var nq = Int(py=q.shape[0])
    var sv = svc_field(fields, "support_vectors_")
    if not precomputed and n_sv > 0 and nq > 0 and not svc_truthy(context["family_svc_xn_old_items"]):
        var degree = svc_int(svc_field(fields, "degree"))
        var gamma = svc_float(svc_field(fields, "_gamma"))
        var coef0 = svc_float(svc_field(fields, "coef0"))
        # the fused chain: kernel then matmul per row tile on the device
        var d = Int(py=q.shape[1])
        var s = family_array(Python.tuple(PythonObject(nq)), "<f4", context)
        _op(binding, "kernel_matmul", Python.list(_a(q), _a(sv), _a(coef), _a(s)),
            _dims(nq, n_sv, d, 1, Int(py=context["family_svc_xn_kernels"][kernel]), degree),
            Python.list(PythonObject(_f32_scalar(gamma)), PythonObject(_f32_scalar(coef0))), context)
        _ = s  # keep the Python object alive through the native call above
        _ = sv  # keep the Python object alive through the native call above
        _ = coef  # keep the Python object alive through the native call above
        _ = q  # keep the Python object alive through the native call above
        _ = q^
        _ = coef^
        return s
    var k: PythonObject
    if precomputed:
        var support = _i32_of(svc_field(fields, "support_"), context)
        k = _take_cols(binding, q, support, Int(py=support.shape[0]), context)
        _ = support^
    else:
        k = _kernel(binding, q, sv, Int(py=context["family_svc_xn_kernels"][kernel]), svc_float(svc_field(fields, "_gamma")),
                    svc_float(svc_field(fields, "coef0")), svc_int(svc_field(fields, "degree")), context)
    var s = family_array(Python.tuple(PythonObject(nq)), "<f4", context)
    _op(binding, "matmul", Python.list(_a(k), _a(coef), _a(s)), _dims(Int(py=k.shape[0]), Int(py=k.shape[1]), 1),
        Python.list(), context)
    _ = s  # keep the Python object alive through the native call above
    _ = q^
    _ = k^
    _ = coef^
    return s


def _ocsvm_decision(model: PythonObject, binding: PythonObject, source: PythonObject,
                    context: PythonObject) raises -> PythonObject:
    """`OneClassSVM.decision_function`: score_samples(X) - offset_ (`xn_unary`)."""
    var scores = _ocsvm_scores(model, binding, source, context)
    var out = family_array(Python.tuple(PythonObject(Int(py=scores.size))), "<f4", context)
    var offset = svc_float(svc_field(py_fields(model), "offset_"))
    _op(binding, "unary", Python.list(_a(scores), _a(out)), _dims(Int(py=scores.size), _U_IDENTITY),
        Python.list(PythonObject(1.0), PythonObject(-offset)), context)
    _ = out  # keep the Python object alive through the native call above
    _ = scores  # keep the Python object alive through the native call above
    _ = scores^
    return out


def _ocsvm_predict(model: PythonObject, binding: PythonObject, source: PythonObject,
                   context: PythonObject) raises -> PythonObject:
    """`OneClassSVM.predict`: 1 inliers, -1 outliers, int64."""
    var dec = _ocsvm_decision(model, binding, source, context)
    var n = Int(py=dec.size)
    var lab = family_array(Python.tuple(PythonObject(n)), "<i4", context)
    if n > 0:
        _op(binding, "p2m_sign_label", Python.list(_a(dec), _a(lab)), _dims(n, 2), Python.list(PythonObject(0.0)), context)
        _ = lab  # keep the Python object alive through the native call above
        _ = dec  # keep the Python object alive through the native call above
    var out = family_array(Python.tuple(PythonObject(n)), "<i8", context)
    if n > 0:
        _ = family_call(svc_base("cast_elements", context), _a(lab), PythonObject(2), _a(out), PythonObject(3),
                        PythonObject(n))
        _ = out  # keep the Python object alive through the native call above
        _ = lab  # keep the Python object alive through the native call above
    _ = dec^
    _ = lab^
    return out


def svc_ocsvm_operation(model: PythonObject, operation: String, var args: PythonObject,
                        context: PythonObject) raises -> PythonObject:
    """OneClassSVM; args[0] is the resolved `_mojolearn_x_neighbors` binding."""
    args = xn_admit_arguments(args)
    classical_connect_numpy(context)
    if operation == "family_svc_ocsvm_fit":
        _ocsvm_fit(model, args[0], args[1], args[2], context)
        return model
    if operation == "family_svc_ocsvm_score_samples":
        return _ocsvm_scores(model, args[0], args[1], context)
    if operation == "family_svc_ocsvm_decision":
        return _ocsvm_decision(model, args[0], args[1], context)
    if operation == "family_svc_ocsvm_predict":
        return _ocsvm_predict(model, args[0], args[1], context)
    if operation == "family_svc_ocsvm_fit_predict":
        # `fit(X).predict(X)`
        _ocsvm_fit(model, args[0], args[1], PythonObject(None), context)
        return _ocsvm_predict(model, args[0], args[1], context)
    raise Error("RuntimeError: unknown OneClassSVM operation " + operation)
