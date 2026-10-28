# SPDX-License-Identifier: Apache-2.0
# The public functions of python/mojolearn/resample.py (bootstrap,
# permutation_test, monte_carlo_integrate, resample_indices, resample), one
# native call each (`family_rs_*`). The shell hands over the resample binding
# selected inside the native boundary; this owner
# makes the Python refusals in their order and words, the compiled-mode
# cross-check (`_extension`), the input casts and outputs, calls the same
# binding entries with the same address and params lists in the same order,
# and builds the result objects. The same kernels on the same bytes: the bits
# do not move.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_get, py_type, py_setitem
from bindings.py_attr import py_attr
from bindings.classical_pandas import classical_admit_value
from bindings.classical_validate import classical_view
from bindings.classical_arrays import classical_is_array
from bindings.metrics_native_inputs import metric_shape, metric_flatten
from bindings.native_scalar_decode import NativeScalarBuffer
from core.native_scalar_decode import NativeRealBits, native_scalar_real
from bindings.mode_metadata_native import _fields
from bindings.host_inference_scalars import host_inference_scalar
from bindings.native_loaded_module import native_loaded_module
from providers.python.neural_descriptor_pyapi import neural_descriptor_entry
from providers.python.resample_public_pyapi import resample_arguments, resample_result_initialize, resample_result_layout
from providers.python.resample_rows_operation_pyapi import resample_rows_operation
from bindings.classical_attributes import classical_connect_numpy
from bindings.native_buffer_owner import native_storage_allocate
from bindings.family_common import (
    FAMILY_F64P, FAMILY_I32P, family_call, family_repr, family_type_name, family_isinstance, family_is_str,
)
from bindings.linalg_ops_buffers import LinBuf, LinAdmission, lin_admitted_c, lin_truth, lin_new, lin_tuple



def rs_extension(binding: PythonObject, numeric_mode: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_extension(numeric_mode)`: the binary's own compile-time mode must be
    the one requested."""
    var b = native_loaded_module("builtins")
    var want = numeric_mode
    if want is None:
        want = context["backend_fields"]["_DEFAULT_MODE"]
    if py_type(want) is not b.str:
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    want = want.strip().lower()
    var fn_obj = neural_descriptor_entry(binding, "resample_numeric_mode", False)
    if fn_obj is not None:
        var got = Int(py=family_call(fn_obj))
        var mode = String(want)
        if (mode != "fast" and mode != "identical") or got != (1 if mode == "identical" else 0):
            raise Error("RuntimeError: mojolearn resample: requested mode differs from compiled mode; rebuild it with bash bindings/build_resample.sh")
    return binding



def rs_entry(binding: PythonObject, name: String) raises -> PythonObject:
    return neural_descriptor_entry(binding, name)


def _rs_is_int(value: PythonObject) raises -> Bool:
    var b = native_loaded_module("builtins")
    return family_isinstance(value, b.int) and not family_isinstance(value, b.bool)


def rs_code(table_name: String, value: PythonObject, name: String, where: String,
            context: PythonObject) raises -> PythonObject:
    """`_code(table, value, name, where)`."""
    var b = native_loaded_module("builtins")
    var table = context["family_rs_codes"][table_name]
    if family_is_str(value):
        if not lin_truth(py_call(py_attr(table, "__contains__"), value)):
            raise Error("mojolearn " + where + ": " + name + " must be one of "
                        + family_repr(py_call(b.sorted, table)) + ", got " + family_repr(value))
        return table[value]
    if not _rs_is_int(value):
        raise Error("TypeError: mojolearn " + where + ": " + name + " must be a name or its code")
    return family_call(b.int, value)


def rs_real(value: PythonObject, name: String, where: String) raises -> PythonObject:
    """`_real(v, name, where)`."""
    var b = native_loaded_module("builtins")
    if family_isinstance(value, b.bool) or not family_isinstance(value, Python.tuple(b.int, b.float)):
        raise Error("TypeError: mojolearn " + where + ": " + name + " must be a real number, got "
                    + family_type_name(value))
    return family_call(b.float, value)


def rs_int(value: PythonObject, name: String, where: String) raises -> PythonObject:
    """`_int(v, name, where)`."""
    if not _rs_is_int(value):
        raise Error("TypeError: mojolearn " + where + ": " + name + " must be an int, got "
                    + family_type_name(value))
    return family_call(native_loaded_module("builtins").int, value)


def _rs_nonneg(value: PythonObject) raises -> Int:
    """`max(value, 0)` of a Python int, as a length."""
    var b = native_loaded_module("builtins")
    return Int(py=family_call(b.max, value, PythonObject(0)))


def _rs_typed(source: PythonObject, typestr: String, ndim: Int, name: String,
              context: PythonObject) raises -> LinBuf:
    # Capture ordinary input-buffer metadata once. Lists, pandas and Torch
    # use their existing native materialization before capturing the new owner.
    classical_admit_value(source, context)
    var value = source
    if not classical_is_array(value, context) and py_type(value) is not py_get(context, "ndarray"):
        value = classical_view(value, context, name)
    var admitted = LinAdmission(value, context)
    var rank = len(admitted.view.shape)
    var shape = family_repr(lin_tuple(admitted.view.shape))
    if ndim >= 0 and rank != ndim:
        raise Error("mojolearn: " + name + " must be " + String(ndim) + "-D, got " + String(rank) + "-D shape " + shape)
    if admitted.view.count() == 0:
        var what = String("sample(s)")
        if rank == 2 and admitted.view.shape[0] != 0:
            what = "feature(s)"
        raise Error("mojolearn: " + name + " is empty, shape " + shape + ": Found array with 0 " + what
                    + " (shape=" + shape + ") while a minimum of 1 is required.")
    return lin_admitted_c(admitted^, typestr, name, context)


def _rs_word(scalars: LinBuf, i: Int) -> Float64:
    return FAMILY_F64P(unsafe_from_address=scalars.addr)[i]


def _rs_object(cls: PythonObject) raises -> PythonObject:
    """A new instance of a plain result class, its fields set by the caller
    (no Python `__init__` runs)."""
    return native_storage_allocate(cls)


def _rs_bootstrap_result(dist: LinBuf, sdist: LinBuf, scalars: LinBuf, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var result = _rs_object(resample_result_layout(0, context))
    py_setitem(_fields(result), "point_estimate", PythonObject(_rs_word(scalars, 0)))
    py_setitem(_fields(result), "distribution", dist.keep)
    py_setitem(_fields(result), "sorted_distribution", sdist.keep)
    py_setitem(_fields(result), "standard_error", PythonObject(_rs_word(scalars, 1)))
    var interval = py_call(b.tuple.__new__, resample_result_layout(3, context),
                           Python.tuple(PythonObject(_rs_word(scalars, 2)), PythonObject(_rs_word(scalars, 3))))
    py_setitem(_fields(result), "confidence_interval", interval)
    py_setitem(_fields(result), "order_low", PythonObject(Int(_rs_word(scalars, 4))))
    py_setitem(_fields(result), "order_high", PythonObject(Int(_rs_word(scalars, 5))))
    return result


def _rs_permutation_result(null: LinBuf, scalars: LinBuf, context: PythonObject) raises -> PythonObject:
    var result = _rs_object(resample_result_layout(1, context))
    py_setitem(_fields(result), "statistic", PythonObject(_rs_word(scalars, 0)))
    py_setitem(_fields(result), "null_distribution", null.keep)
    py_setitem(_fields(result), "pvalue", PythonObject(_rs_word(scalars, 1)))
    py_setitem(_fields(result), "count_less", PythonObject(Int(_rs_word(scalars, 2))))
    py_setitem(_fields(result), "count_greater", PythonObject(Int(_rs_word(scalars, 3))))
    return result


def _rs_addr(buf: LinBuf) -> PythonObject:
    return PythonObject(buf.addr)


# ---------------------------------------------------------------------------
# bootstrap(data, statistic, n_resamples, confidence_level, method,
#           alternative, random_state, q_or_prop, r_first, numeric_mode, paired)


def _rs_method(method: PythonObject, where: String, context: PythonObject) raises -> PythonObject:
    var value = py_call(method.lower) if family_is_str(method) else method
    return rs_code("METHODS", value, "method", where, context)


def _rs_bootstrap_unpaired(args: PythonObject, context: PythonObject) raises -> PythonObject:
    var where = String("bootstrap(paired=False)")
    var b = native_loaded_module("builtins")
    var data = args[1]
    var statistic = args[2]
    if len(data) != 2:
        raise Error("mojolearn " + where + ": data must be two samples (x, y), got " + String(len(data)))
    if (not family_is_str(statistic) or String(statistic) != "diff_means"):
        raise Error("mojolearn " + where + ": statistic " + family_repr(statistic) + " is refused by name: the unpaired bootstrap's"
                    + " two-sample statistic is 'diff_means' (a one-sample statistic has no pairing; pearson needs pairs)")
    var xx = _rs_typed(data[0], "<f4", 1, "x", context)
    var yy = _rs_typed(data[1], "<f4", 1, "y", context)
    var meth = _rs_method(args[5], where, context)
    var alt = rs_code("ALTERNATIVES", args[6], "alternative", where, context)
    var r = rs_int(args[3], "n_resamples", where)
    var dist = lin_new([_rs_nonneg(r)], "<f4", context)
    var sdist = lin_new([_rs_nonneg(r)], "<f4", context)
    var scalars = lin_new([6], "<f8", context)
    var fn_obj = rs_entry(rs_extension(args[0], args[10], context), "bootstrap_unpaired")
    # ORDER MATCHES bindings/_mojolearn_resample.mojo::bootstrap_unpaired_binding:
    # n_x, n_y, n_resamples, seed, method, confidence_level, alternative, r_first
    var addrs = Python.list(_rs_addr(xx), _rs_addr(yy), _rs_addr(dist), _rs_addr(sdist), _rs_addr(scalars))
    _ = scalars  # keep the Python object alive through the native call above
    _ = sdist  # keep the Python object alive through the native call above
    _ = dist  # keep the Python object alive through the native call above
    _ = yy  # keep the Python object alive through the native call above
    _ = xx  # keep the Python object alive through the native call above
    var params = Python.list(PythonObject(xx.dim(0)), PythonObject(yy.dim(0)), r,
                             rs_int(args[7], "random_state", where), meth,
                             rs_real(args[4], "confidence_level", where), alt, rs_int(args[9], "r_first", where))
    _ = family_call(fn_obj, addrs, params)
    _ = scalars  # keep the Python object alive through the native call above
    _ = sdist  # keep the Python object alive through the native call above
    _ = dist  # keep the Python object alive through the native call above
    _ = yy  # keep the Python object alive through the native call above
    _ = xx  # keep the Python object alive through the native call above
    _ = xx^
    _ = yy^
    return _rs_bootstrap_result(dist, sdist, scalars, context)


def _rs_bootstrap(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """args: binding, data, statistic, n_resamples, confidence_level, method,
    alternative, random_state, q_or_prop, r_first, numeric_mode, paired."""
    var where = String("bootstrap")
    var b = native_loaded_module("builtins")
    var data = args[1]
    if not lin_truth(args[11]) and family_isinstance(data, Python.tuple(b.tuple, b.list)):
        return _rs_bootstrap_unpaired(args, context)
    var x = _rs_typed(data, "<f4", -1, "data", context)
    var n: Int
    var d: Int
    if x.rank() == 1:
        n = x.dim(0)
        d = 1
    elif x.rank() == 2:
        n = x.dim(0)
        d = x.dim(1)
    else:
        raise Error("mojolearn " + where + ": data must be 1-D or 2-D, got " + String(x.rank()) + "-D")
    var stat = rs_code("STATISTICS", args[2], "statistic", where, context)
    var meth = _rs_method(args[5], where, context)
    var alt = rs_code("ALTERNATIVES", args[6], "alternative", where, context)
    var r = rs_int(args[3], "n_resamples", where)
    var rf = rs_int(args[9], "r_first", where)
    var seed = rs_int(args[7], "random_state", where)
    var cl = rs_real(args[4], "confidence_level", where)
    var qp = rs_real(args[8], "q_or_prop", where)
    var dist = lin_new([_rs_nonneg(r)], "<f4", context)
    var sdist = lin_new([_rs_nonneg(r)], "<f4", context)
    var scalars = lin_new([6], "<f8", context)
    var fn_obj = rs_entry(rs_extension(args[0], args[10], context), "bootstrap")
    # ORDER MATCHES bindings/_mojolearn_resample.mojo::bootstrap_binding.
    # x (the C-order sample, flat), distribution_out, sorted_out, scalars_out;
    # n, d, statistic, n_resamples, seed, method, confidence_level,
    # alternative, q_or_prop, r_first
    var addrs = Python.list(_rs_addr(x), _rs_addr(dist), _rs_addr(sdist), _rs_addr(scalars))
    _ = scalars  # keep the Python object alive through the native call above
    _ = sdist  # keep the Python object alive through the native call above
    _ = dist  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    var params = Python.list(PythonObject(n), PythonObject(d), stat, r, seed, meth, cl, alt, qp, rf)
    _ = family_call(fn_obj, addrs, params)
    _ = scalars  # keep the Python object alive through the native call above
    _ = sdist  # keep the Python object alive through the native call above
    _ = dist  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    _ = x^
    return _rs_bootstrap_result(dist, sdist, scalars, context)


# ---------------------------------------------------------------------------
# permutation_test(x, y, statistic, n_resamples, alternative, random_state,
#                  r_first, numeric_mode, permutation_type)


def _rs_permutation_samples(args: PythonObject, context: PythonObject) raises -> PythonObject:
    var where = String("permutation_test(permutation_type='samples')")
    var y = args[2]
    var statistic = args[3]
    var xx = _rs_typed(args[1], "<f4", 1, "x", context)
    var want = String("mean") if y is None else String("diff_means")
    if (not family_is_str(statistic) or String(statistic) != want):
        var arm = String("one sample") if y is None else String("two paired samples")
        raise Error("mojolearn " + where + ": statistic " + family_repr(statistic) + " is refused by name; the implemented arm for "
                    + arm + " is " + family_repr(PythonObject(want)))
    var ny = 0
    var yy: LinBuf
    if y is None:
        yy = LinBuf(xx.keep, xx.addr, xx.shape.copy())
        _ = xx  # keep the Python object alive through the native call above
    else:
        yy = _rs_typed(y, "<f4", 1, "y", context)
        ny = yy.dim(0)
    var r = rs_int(args[4], "n_resamples", where)
    var null = lin_new([_rs_nonneg(r)], "<f4", context)
    var scalars = lin_new([4], "<f8", context)
    var fn_obj = rs_entry(rs_extension(args[0], args[8], context), "permutation_samples")
    # ORDER MATCHES bindings/_mojolearn_resample.mojo::permutation_samples_binding:
    # n, n_y, n_resamples, seed, alternative, r_first
    var addrs = Python.list(_rs_addr(xx), _rs_addr(yy), _rs_addr(null), _rs_addr(scalars))
    _ = scalars  # keep the Python object alive through the native call above
    _ = null  # keep the Python object alive through the native call above
    _ = xx  # keep the Python object alive through the native call above
    var params = Python.list(PythonObject(xx.dim(0)), PythonObject(ny), r, rs_int(args[6], "random_state", where),
                             rs_code("ALTERNATIVES", args[5], "alternative", where, context),
                             rs_int(args[7], "r_first", where))
    _ = family_call(fn_obj, addrs, params)
    _ = scalars  # keep the Python object alive through the native call above
    _ = null  # keep the Python object alive through the native call above
    _ = xx  # keep the Python object alive through the native call above
    _ = xx^
    _ = yy^
    return _rs_permutation_result(null, scalars, context)


def _rs_permutation_test(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """args: binding, x, y, statistic, n_resamples, alternative,
    random_state, r_first, numeric_mode, permutation_type."""
    var where = String("permutation_test")
    var permutation_type = args[9]
    var ptype = String("")
    var named = family_is_str(permutation_type)
    if named:
        ptype = String(py=py_call(permutation_type.lower))
    if named and ptype == "samples":
        return _rs_permutation_samples(args, context)
    if named and ptype == "pairings":
        raise Error("mojolearn " + where + ": permutation_type='pairings' is refused by name: its null permutes every"
                    + " sample's observation order, which for the implemented statistics (mean, std, diff_means)"
                    + " leaves the statistic unchanged -- the null is the observed value R times; pearson, the"
                    + " statistic it exists for, has no permutation arm (resample/NOT_IMPLEMENTED.tsv)")
    if not named or ptype != "independent":
        raise Error("mojolearn " + where + ": permutation_type must be 'independent', 'samples' or"
                    + " 'pairings', got " + family_repr(permutation_type))
    if args[2] is None:
        raise Error("mojolearn " + where + ": permutation_type='independent' needs two samples x and y")
    var xx = _rs_typed(args[1], "<f4", 1, "x", context)
    var yy = _rs_typed(args[2], "<f4", 1, "y", context)
    var stat = rs_code("STATISTICS", args[3], "statistic", where, context)
    var alt = rs_code("ALTERNATIVES", args[5], "alternative", where, context)
    var r = rs_int(args[4], "n_resamples", where)
    var rf = rs_int(args[7], "r_first", where)
    var seed = rs_int(args[6], "random_state", where)
    var null = lin_new([_rs_nonneg(r)], "<f4", context)
    var scalars = lin_new([4], "<f8", context)
    var fn_obj = rs_entry(rs_extension(args[0], args[8], context), "permutation_test")
    # ORDER MATCHES bindings/_mojolearn_resample.mojo::permutation_test_binding:
    # x, y, null_out, scalars_out; n_x, n_y, statistic, n_resamples, seed,
    # alternative, r_first
    var addrs = Python.list(_rs_addr(xx), _rs_addr(yy), _rs_addr(null), _rs_addr(scalars))
    _ = scalars  # keep the Python object alive through the native call above
    _ = null  # keep the Python object alive through the native call above
    _ = yy  # keep the Python object alive through the native call above
    _ = xx  # keep the Python object alive through the native call above
    var params = Python.list(PythonObject(xx.dim(0)), PythonObject(yy.dim(0)), stat, r, seed, alt, rf)
    _ = family_call(fn_obj, addrs, params)
    _ = scalars  # keep the Python object alive through the native call above
    _ = null  # keep the Python object alive through the native call above
    _ = yy  # keep the Python object alive through the native call above
    _ = xx  # keep the Python object alive through the native call above
    _ = xx^
    _ = yy^
    return _rs_permutation_result(null, scalars, context)


# ---------------------------------------------------------------------------
# monte_carlo_integrate(integrand, lower, upper, n_samples, random_state,
#                       i_first, numeric_mode)


def _rs_corner(values: PythonObject, name: String, where: String, context: PythonObject) raises -> LinBuf:
    """`Array.from_list([_real(v, name, where) for v in values], "<f4")`."""
    var b = native_loaded_module("builtins")
    var items = Python.list()
    if classical_is_array(values, context):
        var admitted = LinAdmission(values, context)
        ref view = admitted.view
        if len(view.shape) != 1:
            raise Error("TypeError: mojolearn " + where + ": " + name + " must be a one-dimensional native sequence")
        # Reuse metric label boxing semantics, but read every Array slot once
        # and retain the captured owners through all raw storage access.
        var kind = String(view.dtype[byte=1:2])
        var code = String("i") if kind == "i" else String("I")
        if kind == "f":
            code = "e" if view.itemsize == 2 else "f" if view.itemsize == 4 else "d"
        var data = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=view.addr)
        for i in range(view.shape[0]):  # driver-loop: corner coordinate boxing from retained Array metadata
            var decoded = NativeScalarBuffer(native_scalar_real(data, i * view.strides[0], code, view.itemsize, True),
                                             NativeRealBits(), False, False)
            py_call(items.append, decoded.to_python_number(kind != "f"))
        _ = admitted^
    else:
        var source = values
        if py_type(values) is py_get(context, "ndarray"):
            # The exact native view snapshots ndarray metadata before the
            # shape check and native storage boxing consume that snapshot.
            source = py_call(py_attr(context["ndarray"], "view"), values)
        elif py_type(values) is b.list:
            source = py_call(b.tuple, values)
        var shape = metric_shape(source, context)
        if len(shape) != 1:
            raise Error("TypeError: mojolearn " + where + ": " + name + " must be a one-dimensional native sequence")
        items = metric_flatten(source, context)
    var count = len(items)
    var words = List[Float64]()
    for i in range(count):  # driver-loop: over the corner's coordinates (the box's dimensions)
        words.append(Float64(py=rs_real(items[i], name, where)))
    var out = lin_new([count], "<f4", context)
    var p = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=out.addr)
    _ = out  # keep the Python object alive through the native call above
    for i in range(count):  # driver-loop: over the corner's coordinates (the box's dimensions)
        p[i] = words[i].cast[DType.float32]()
        _ = out  # keep the Python object alive through the native call above
    return out^


def _rs_monte_carlo(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """args: binding, integrand, lower, upper, n_samples, random_state,
    i_first, numeric_mode."""
    var where = String("monte_carlo_integrate")
    var f_id = rs_code("INTEGRANDS", args[1], "integrand", where, context)
    var lo = _rs_corner(args[2], "lower", where, context)
    var hi = _rs_corner(args[3], "upper", where, context)
    if lo.dim(0) != 2 or hi.dim(0) != 2:
        raise Error("mojolearn " + where + ": lower and upper must each hold 2 values "
                    + "(MC_DIMS), got " + String(lo.dim(0)) + " and " + String(hi.dim(0)))
    var scalars = lin_new([4], "<f8", context)
    var fn_obj = rs_entry(rs_extension(args[0], args[7], context), "monte_carlo_integrate")
    # ORDER MATCHES bindings/_mojolearn_resample.mojo::monte_carlo_integrate_binding:
    # lower, upper, scalars_out; f_id, n_samples, seed, i_first
    var addrs = Python.list(_rs_addr(lo), _rs_addr(hi), _rs_addr(scalars))
    _ = scalars  # keep the Python object alive through the native call above
    _ = hi  # keep the Python object alive through the native call above
    _ = lo  # keep the Python object alive through the native call above
    var params = Python.list(f_id, rs_int(args[4], "n_samples", where), rs_int(args[5], "random_state", where),
                             rs_int(args[6], "i_first", where))
    _ = family_call(fn_obj, addrs, params)
    _ = scalars  # keep the Python object alive through the native call above
    _ = hi  # keep the Python object alive through the native call above
    _ = lo  # keep the Python object alive through the native call above
    var result = _rs_object(resample_result_layout(2, context))
    py_setitem(_fields(result), "integral", PythonObject(_rs_word(scalars, 0)))
    py_setitem(_fields(result), "mean", PythonObject(_rs_word(scalars, 1)))
    py_setitem(_fields(result), "volume", PythonObject(_rs_word(scalars, 2)))
    py_setitem(_fields(result), "closed_form", PythonObject(_rs_word(scalars, 3)))
    _ = lo^
    _ = hi^
    return result


# ---------------------------------------------------------------------------
# resample_indices(n, n_samples, replace, random_state, numeric_mode)


def rs_indices(binding: PythonObject, n_obj: PythonObject, n_samples: PythonObject, replace: PythonObject,
               random_state: PythonObject, numeric_mode: PythonObject, context: PythonObject) raises -> LinBuf:
    var where = String("resample")
    var n = rs_int(n_obj, "n", where)
    var count = n if n_samples is None else rs_int(n_samples, "n_samples", where)
    var idx = lin_new([_rs_nonneg(count)], "<i4", context)
    var fn_obj = rs_entry(rs_extension(binding, numeric_mode, context), "resample_indices")
    # ORDER MATCHES bindings/_mojolearn_resample.mojo::resample_indices_binding.
    var params = Python.list(n, count, PythonObject(1 if lin_truth(replace) else 0),
                             rs_int(random_state, "random_state", where))
    _ = family_call(fn_obj, Python.list(_rs_addr(idx)), params)
    _ = idx  # keep the Python object alive through the native call above
    return idx^


def _rs_resample_indices(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """args: binding, n, n_samples, replace, random_state, numeric_mode."""
    var idx = rs_indices(args[0], args[1], args[2], args[3], args[4], args[5], context)
    return idx.keep


# ---------------------------------------------------------------------------
# resample(*arrays, replace, n_samples, random_state, stratify,
#          sample_weight, numeric_mode)


def rs_operation(operation: String, var args: PythonObject, context: PythonObject) raises -> PythonObject:
    if operation == "family_rs_result_initialize":
        return resample_result_initialize(args, context)
    classical_connect_numpy(context)
    if operation == "family_rs_resample":
        return resample_rows_operation(args, context)
    args = resample_arguments(operation, args, context)
    for i in range(1, len(args)):  # driver-loop: admit native public inputs before any legacy scalar/container operations
        classical_admit_value(args[i], context)
        var data = ((operation == "family_rs_bootstrap" and i == 1)
                    or (operation == "family_rs_permutation_test" and (i == 1 or i == 2))
                    or (operation == "family_rs_monte_carlo_integrate" and (i == 2 or i == 3)))
        if not data:
            host_inference_scalar(args[i], context, True)
    if operation == "family_rs_bootstrap":
        return _rs_bootstrap(args, context)
    if operation == "family_rs_permutation_test":
        return _rs_permutation_test(args, context)
    if operation == "family_rs_monte_carlo_integrate":
        return _rs_monte_carlo(args, context)
    if operation == "family_rs_resample_indices":
        return _rs_resample_indices(args, context)
    raise Error("RuntimeError: unknown resample operation " + operation)
