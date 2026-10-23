# SPDX-License-Identifier: Apache-2.0
# The public functions of python/mojolearn/resample.py (bootstrap,
# permutation_test, monte_carlo_integrate, resample_indices, resample), one
# native call each (`family_rs_*`). The shell hands over the resample binding
# `_backend.binding("_mojolearn_resample", numeric_mode)` resolved; this owner
# makes the Python refusals in their order and words, the compiled-mode
# cross-check (`_extension`), the input casts and outputs, calls the same
# binding entries with the same address and params lists in the same order,
# and builds the result objects. The same kernels on the same bytes: the bits
# do not move.
from std.memory import memcpy
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_get, py_type, py_setattr
from bindings.py_attr import py_attr
from bindings.classical_arrays import classical_address
from bindings.classical_attributes import classical_connect_numpy
from bindings.native_buffer_owner import native_storage_allocate
from bindings.sequential_arrays import sequential_array
from bindings.family_common import (
    FAMILY_F64P, FAMILY_I32P, family_call, family_repr, family_type_name, family_isinstance, family_is_str,
)
from bindings.optimizer_public_door import TensorView
from bindings.linalg_ops_buffers import LinBuf, lin_truth, lin_typed, lin_new, lin_view, lin_tuple

comptime RS_BP = MutPointer[UInt8, MutUntrackedOrigin]


def rs_state(context: PythonObject) raises -> PythonObject:
    """python/mojolearn/resample.py's module dictionary (its tables and result classes)."""
    return context["family_rs_state"]


def rs_extension(binding: PythonObject, numeric_mode: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_extension(numeric_mode)`: the binary's own compile-time mode must be
    the one requested."""
    var b = Python.import_module("builtins")
    var want = numeric_mode
    if not lin_truth(want):
        want = context["backend_fields"]["_DEFAULT_MODE"]
        if want is None:
            want = family_call(context["backend_fields"]["default_mode"])
    var fn_obj = py_call(b.getattr, binding, "resample_numeric_mode", PythonObject(None))
    if fn_obj is not None:
        var got = family_call(b.int, family_call(fn_obj))
        var code = py_get(rs_state(context)["_MODE_CODE"], want, PythonObject(None))
        if code is None or Int(py=got) != Int(py=code):
            raise Error("RuntimeError: mojolearn resample: numeric_mode=" + family_repr(want) + " was requested but "
                        + String(py=py_call(b.getattr, binding, "__name__")) + " reports compile-time mode code "
                        + String(py=family_call(b.str, got)) + "; rebuild it "
                        + "with bash bindings/build_resample.sh")
    return binding


def rs_entry(binding: PythonObject, name: String) raises -> PythonObject:
    return family_call(Python.import_module("builtins").getattr, binding, PythonObject(name))


def _rs_is_int(value: PythonObject) raises -> Bool:
    var b = Python.import_module("builtins")
    return family_isinstance(value, b.int) and not family_isinstance(value, b.bool)


def rs_code(table_name: String, value: PythonObject, name: String, where: String,
            context: PythonObject) raises -> PythonObject:
    """`_code(table, value, name, where)`."""
    var b = Python.import_module("builtins")
    var table = rs_state(context)[table_name]
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
    var b = Python.import_module("builtins")
    if family_isinstance(value, b.bool) or not family_isinstance(value, Python.tuple(b.int, b.float)):
        raise Error("TypeError: mojolearn " + where + ": " + name + " must be a real number, got "
                    + family_type_name(value))
    return family_call(b.float, value)


def rs_int(value: PythonObject, name: String, where: String) raises -> PythonObject:
    """`_int(v, name, where)`."""
    if not _rs_is_int(value):
        raise Error("TypeError: mojolearn " + where + ": " + name + " must be an int, got "
                    + family_type_name(value))
    return family_call(Python.import_module("builtins").int, value)


def _rs_nonneg(value: PythonObject) raises -> Int:
    """`max(value, 0)` of a Python int, as a length."""
    var b = Python.import_module("builtins")
    return Int(py=family_call(b.max, value, PythonObject(0)))


def _rs_word(scalars: LinBuf, i: Int) -> Float64:
    return FAMILY_F64P(unsafe_from_address=scalars.addr)[i]


def _rs_object(cls: PythonObject) raises -> PythonObject:
    """A new instance of a plain result class, its fields set by the caller
    (no Python `__init__` runs)."""
    return native_storage_allocate(cls)


def _rs_bootstrap_result(dist: LinBuf, sdist: LinBuf, scalars: LinBuf, context: PythonObject) raises -> PythonObject:
    var state = rs_state(context)
    var b = Python.import_module("builtins")
    var result = _rs_object(state["BootstrapResult"])
    py_setattr(result, "point_estimate", PythonObject(_rs_word(scalars, 0)))
    py_setattr(result, "distribution", dist.keep)
    py_setattr(result, "sorted_distribution", sdist.keep)
    py_setattr(result, "standard_error", PythonObject(_rs_word(scalars, 1)))
    var interval = py_call(b.tuple.__new__, state["ConfidenceInterval"],
                           Python.tuple(PythonObject(_rs_word(scalars, 2)), PythonObject(_rs_word(scalars, 3))))
    py_setattr(result, "confidence_interval", interval)
    py_setattr(result, "order_low", PythonObject(Int(_rs_word(scalars, 4))))
    py_setattr(result, "order_high", PythonObject(Int(_rs_word(scalars, 5))))
    return result


def _rs_permutation_result(null: LinBuf, scalars: LinBuf, context: PythonObject) raises -> PythonObject:
    var result = _rs_object(rs_state(context)["PermutationTestResult"])
    py_setattr(result, "statistic", PythonObject(_rs_word(scalars, 0)))
    py_setattr(result, "null_distribution", null.keep)
    py_setattr(result, "pvalue", PythonObject(_rs_word(scalars, 1)))
    py_setattr(result, "count_less", PythonObject(Int(_rs_word(scalars, 2))))
    py_setattr(result, "count_greater", PythonObject(Int(_rs_word(scalars, 3))))
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
    var b = Python.import_module("builtins")
    var data = args[1]
    var statistic = args[2]
    if len(data) != 2:
        raise Error("mojolearn " + where + ": data must be two samples (x, y), got " + String(len(data)))
    if lin_truth(py_call(Python.import_module("operator").ne, statistic, PythonObject("diff_means"))):
        raise Error("mojolearn " + where + ": statistic " + family_repr(statistic) + " is refused by name: the unpaired bootstrap's"
                    + " two-sample statistic is 'diff_means' (a one-sample statistic has no pairing; pearson needs pairs)")
    var xx = lin_typed(data[0], "<f4", 1, "x", context)
    var yy = lin_typed(data[1], "<f4", 1, "y", context)
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
    var b = Python.import_module("builtins")
    var data = args[1]
    if not lin_truth(args[11]) and family_isinstance(data, Python.tuple(b.tuple, b.list)):
        return _rs_bootstrap_unpaired(args, context)
    var x = lin_typed(data, "<f4", -1, "data", context)
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
    var xx = lin_typed(args[1], "<f4", 1, "x", context)
    var want = String("mean") if y is None else String("diff_means")
    if lin_truth(py_call(Python.import_module("operator").ne, statistic, PythonObject(want))):
        var arm = String("one sample") if y is None else String("two paired samples")
        raise Error("mojolearn " + where + ": statistic " + family_repr(statistic) + " is refused by name; the implemented arm for "
                    + arm + " is " + family_repr(PythonObject(want)))
    var ny = 0
    var yy: LinBuf
    if y is None:
        yy = LinBuf(xx.keep, xx.addr, xx.shape.copy())
        _ = xx  # keep the Python object alive through the native call above
    else:
        yy = lin_typed(y, "<f4", 1, "y", context)
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
    var xx = lin_typed(args[1], "<f4", 1, "x", context)
    var yy = lin_typed(args[2], "<f4", 1, "y", context)
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
    var b = Python.import_module("builtins")
    var items = family_call(b.list, values)
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
    var result = _rs_object(rs_state(context)["MonteCarloResult"])
    py_setattr(result, "integral", PythonObject(_rs_word(scalars, 0)))
    py_setattr(result, "mean", PythonObject(_rs_word(scalars, 1)))
    py_setattr(result, "volume", PythonObject(_rs_word(scalars, 2)))
    py_setattr(result, "closed_form", PythonObject(_rs_word(scalars, 3)))
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


def _rs_np(context: PythonObject) raises -> PythonObject:
    """`require_numpy("resample")`: the loaded NumPy module, imported when
    it is installed but not loaded yet, its native types connected."""
    var numpy = py_get(context["modules"], PythonObject("numpy"))
    if numpy is None:
        try:
            numpy = Python.import_module("numpy")
        except:
            raise Error("ImportError: mojolearn: resample requires optional NumPy support. "
                        + 'Install it with: python -m pip install "mojolearn[numpy]" '
                        + '("mojolearn[verify]" also includes NumPy).')
    classical_connect_numpy(context)
    return numpy


def _rs_numpy(context: PythonObject) raises -> Bool:
    return py_get(context, PythonObject("ndarray")) is not None


def _rs_is_ndarray(x: PythonObject, context: PythonObject) raises -> Bool:
    return _rs_numpy(context) and family_isinstance(x, context["ndarray"])


def _rs_is_f32(x: PythonObject, context: PythonObject) raises -> Bool:
    return lin_truth(py_call(Python.import_module("operator").eq, x.dtype, context["float32"]))


def _rs_output(x: PythonObject, count: Int, dtype: PythonObject, context: PythonObject) raises -> PythonObject:
    """`np.empty((count,) + x.shape[1:], dtype=dtype)`."""
    var shape = Python.list(PythonObject(count))
    var dims = x.shape
    for k in range(1, len(dims)):  # driver-loop: over array rank metadata
        _ = py_call(shape.append, dims[k])
    return sequential_array(py_call(Python.import_module("builtins").tuple, shape), dtype, context)


def _rs_gpu_gather(binding: PythonObject, arrays: PythonObject, n: Int, count: Int, seed: PythonObject,
                   numeric_mode: PythonObject, replace: Bool, context: PythonObject) raises -> PythonObject:
    """`_gpu_gather`: the device draw and gather of float32 C-contiguous
    ndarrays, or None for the unchanged fallback."""
    var b = Python.import_module("builtins")
    if count <= 0 or n <= 0:
        return PythonObject(None)
    var mod = rs_extension(binding, numeric_mode, context)
    var enabled = py_call(b.getattr, mod, "resample_gpu_gather_enabled", PythonObject(None))
    if enabled is None or Int(py=family_call(b.int, family_call(enabled))) == 0:
        return PythonObject(None)
    if not _rs_numpy(context):
        return PythonObject(None)
    for i in range(len(arrays)):  # driver-loop: over the argument arrays
        var x = arrays[i]
        if not _rs_is_ndarray(x, context) or not _rs_is_f32(x, context):
            return PythonObject(None)
        var nd = Int(py=x.ndim)
        if (nd != 1 and nd != 2) or not lin_truth(x.flags.c_contiguous):
            return PythonObject(None)
        if nd == 2 and not (0 < Int(py=x.shape[1]) and Int(py=x.shape[1]) <= 2147483647):
            return PythonObject(None)
    var outputs = Python.list()
    var addresses = Python.list()
    var params = Python.list(PythonObject(n), PythonObject(count), seed)
    for i in range(len(arrays)):  # driver-loop: over the argument arrays
        var x = arrays[i]
        var output = _rs_output(x, count, context["float32"], context)
        _ = py_call(outputs.append, output)
        _ = py_call(addresses.append, classical_address(x))
        _ = x  # keep the Python object alive through the native call above
        _ = py_call(addresses.append, classical_address(output))
        _ = output  # keep the Python object alive through the native call above
        _ = x  # keep the Python object alive through the native call above
        _ = py_call(params.append, PythonObject(1) if Int(py=x.ndim) == 1 else x.shape[1])
    var route: PythonObject
    if replace:
        route = rs_entry(mod, "resample_gather_gpu")
    else:
        route = py_call(b.getattr, mod, "resample_permutation_gather_gpu", PythonObject(None))
    if route is not None and Int(py=family_call(b.int, family_call(route, addresses, params))) != 0:
        return outputs
    return PythonObject(None)


def _rs_fast_defines(binding: PythonObject, numeric_mode: PythonObject, context: PythonObject) raises -> Int:
    """`fast_defines(numeric_mode)`."""
    var b = Python.import_module("builtins")
    var mode = numeric_mode
    if not lin_truth(mode):
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    if mode is None or String(py=family_call(b.str, mode)) != "fast":
        return 0
    var mod: PythonObject
    try:
        mod = rs_extension(binding, numeric_mode, context)
    except:
        return 0
    var fn_obj = py_call(b.getattr, mod, "resample_fast_defines", PythonObject(None))
    if fn_obj is None:
        return 0
    return Int(py=family_call(b.int, family_call(fn_obj)))


def _rs_take(a: PythonObject, idx: LinBuf, context: PythonObject) raises -> PythonObject:
    """`_take(a, idx)`: the rows `idx` of an array (its dtype and trailing
    shape kept), or of a sequence as a list."""
    var b = Python.import_module("builtins")
    var count = idx.dim(0)
    var rows = FAMILY_I32P(unsafe_from_address=idx.addr)
    if lin_truth(py_call(b.hasattr, a, "__array__")) and lin_truth(py_call(b.hasattr, a, "shape")):
        var numpy = _rs_np(context)
        var arr = family_call(numpy.asarray, a)
        var v = TensorView()
        if not lin_view(arr, context, v) or v.fmt == "O" or len(v.shape) == 0:
            # an object-dtype (or 0-d) array: NumPy's own row take
            var picked = Python.list()
            for t in range(count):  # cpu-route: object rows taken by index [py-data-loop]
                _ = py_call(picked.append, PythonObject(Int(rows[t])))
            return family_call(py_attr(arr, "__getitem__"), family_call(numpy.asarray, picked, numpy.intp))
        var out = _rs_output(arr, count, arr.dtype, context)
        var dst = Int(py=classical_address(out))
        _ = out  # keep the Python object alive through the native call above
        var nd = len(v.shape)
        var row_items = 1
        for k in range(1, nd):  # driver-loop: over array rank metadata
            row_items *= v.shape[k]
        var item = v.itemsize
        var strides = List[Int](length=nd, fill=0)
        if v.has_strides:
            for k in range(nd):  # driver-loop: over stride words
                strides[k] = v.strides[k]
        else:
            var step = item
            for k in range(nd - 1, -1, -1):  # driver-loop: over stride words
                strides[k] = step
                step *= v.shape[k]
        var idx_k = List[Int](length=nd, fill=0)
        var w = 0
        for t in range(count):  # cpu-route: the host row take of the fallback route (NumPy's take)
            var base = v.addr + Int(rows[t]) * strides[0]
            _ = v  # keep the Python object alive through the native call above
            var off = 0
            for e in range(row_items):  # cpu-route: one row's items in C order
                memcpy(dest=RS_BP(unsafe_from_address=dst + w * item),
                       src=RS_BP(unsafe_from_address=base + off), count=item)
                _ = out  # keep the Python object alive through the native call above
                _ = v  # keep the Python object alive through the native call above
                w += 1
                var k = nd - 1
                while k >= 1:  # driver-loop: over the axes of one index step
                    idx_k[k] += 1
                    off += strides[k]
                    if idx_k[k] < v.shape[k]:
                        break
                    off -= strides[k] * v.shape[k]
                    idx_k[k] = 0
                    k -= 1
        _ = arr^
        return out
    var picked = Python.list()
    for t in range(count):  # cpu-route: Python list input taken by index [py-data-loop]
        _ = py_call(picked.append, family_call(py_attr(a, "__getitem__"), PythonObject(Int(rows[t]))))
    return picked


def _rs_narrow_gather(binding: PythonObject, arrays: PythonObject, n: Int, count: Int, seed: PythonObject,
                      numeric_mode: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_narrow_gather` (-D MOJOLEARN_RESAMPLE_FAST_GATHER_NARROW), or None."""
    var b = Python.import_module("builtins")
    var bit = Int(py=rs_state(context)["FAST_DEFINE_GATHER_NARROW"])
    if count <= 0 or n <= 0 or (_rs_fast_defines(binding, numeric_mode, context) & bit) == 0:
        return PythonObject(None)
    var mod = rs_extension(binding, numeric_mode, context)
    var fn_obj = py_call(b.getattr, mod, "resample_gather_narrow", PythonObject(None))
    if fn_obj is None or not _rs_numpy(context):
        return PythonObject(None)
    var bound = Int(py=family_call(b.int, family_call(rs_entry(mod, "resample_gather_narrow_bytes"))))
    var idx = lin_new([count], "<i4", context)
    var outs = Python.list()
    var addresses = Python.list(_rs_addr(idx))
    _ = idx  # keep the Python object alive through the native call above
    var params = Python.list(PythonObject(n), PythonObject(count), seed)
    for i in range(len(arrays)):  # driver-loop: over the argument arrays
        var x = arrays[i]
        var w = 0
        if _rs_is_ndarray(x, context):
            var nd = Int(py=x.ndim)
            if nd == 1:
                w = 1
            elif nd == 2:
                w = Int(py=x.shape[1])
        if not (w > 0 and 4 * w <= bound and _rs_is_f32(x, context) and lin_truth(x.flags.c_contiguous)):
            w = 0
        if w > 0:
            var out = _rs_output(x, count, context["float32"], context)
            _ = py_call(outs.append, out)
            _ = py_call(addresses.append, classical_address(x))
            _ = x  # keep the Python object alive through the native call above
            _ = idx  # keep the Python object alive through the native call above
            _ = py_call(addresses.append, classical_address(out))
            _ = out  # keep the Python object alive through the native call above
            _ = x  # keep the Python object alive through the native call above
            _ = idx  # keep the Python object alive through the native call above
        else:
            _ = py_call(outs.append, PythonObject(None))
            _ = py_call(addresses.append, PythonObject(0))
            _ = x  # keep the Python object alive through the native call above
            _ = idx  # keep the Python object alive through the native call above
            _ = py_call(addresses.append, PythonObject(0))
            _ = x  # keep the Python object alive through the native call above
            _ = idx  # keep the Python object alive through the native call above
        _ = py_call(params.append, PythonObject(w))
    var done = Int(py=family_call(b.int, family_call(fn_obj, addresses, params)))
    _ = idx  # keep the Python object alive through the native call above
    if done < 0:
        return PythonObject(None)
    var result = Python.list()
    for i in range(len(arrays)):  # driver-loop: over the argument arrays
        if ((done >> i) & 1) != 0:
            _ = py_call(result.append, outs[i])
        else:
            _ = py_call(result.append, _rs_take(arrays[i], idx, context))
    _ = idx^
    return result


def _rs_one_or_all(gathered: PythonObject) raises -> PythonObject:
    return gathered[0] if len(gathered) == 1 else gathered


def _rs_resample(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """args: binding, arrays, replace, n_samples, random_state, stratify,
    sample_weight, numeric_mode."""
    var where = String("resample")
    var b = Python.import_module("builtins")
    var binding = args[0]
    var arrays = args[1]
    var replace_obj = args[2]
    var n_samples = args[3]
    var random_state = args[4]
    var numeric_mode = args[7]
    if args[5] is not None:
        raise Error("mojolearn " + where + ": stratify= is refused by name (resample/NOT_IMPLEMENTED.tsv): "
                    + "scikit-learn's per-class allocation (_approximate_mode) breaks ties with its RNG stream")
    if args[6] is not None:
        raise Error("mojolearn " + where + ": sample_weight= is refused by name (resample/NOT_IMPLEMENTED.tsv): "
                    + "a weighted draw is an inverse-CDF lookup over a float cumulative sum not yet pinned")
    if len(arrays) == 0:
        return PythonObject(None)
    var n = Int(py=family_call(b.len, arrays[0]))
    for i in range(1, len(arrays)):  # driver-loop: over the argument arrays
        if Int(py=family_call(b.len, arrays[i])) != n:
            var lengths = Python.list()
            for j in range(len(arrays)):  # driver-loop: over the argument arrays
                _ = py_call(lengths.append, family_call(b.len, arrays[j]))
            raise Error("mojolearn " + where + ": Found input variables with inconsistent numbers of samples: "
                        + family_repr(lengths))
    var count_obj = PythonObject(n) if n_samples is None else rs_int(n_samples, "n_samples", where)
    var count = Int(py=count_obj)
    var replace = lin_truth(replace_obj)
    if not replace:
        if count < 0 or count > n:
            # the existing refusal, before any native buffer is requested
            _ = rs_indices(binding, PythonObject(n), n_samples, replace_obj, random_state, numeric_mode, context)
        var gathered = _rs_gpu_gather(binding, arrays, n, count, rs_int(random_state, "random_state", where),
                                      numeric_mode, False, context)
        if gathered is not None:
            return _rs_one_or_all(gathered)
    if replace:
        var gathered = _rs_gpu_gather(binding, arrays, n, count, rs_int(random_state, "random_state", where),
                                      numeric_mode, True, context)
        if gathered is not None:
            return _rs_one_or_all(gathered)
        gathered = _rs_narrow_gather(binding, arrays, n, count, rs_int(random_state, "random_state", where),
                                     numeric_mode, context)
        if gathered is not None:
            return _rs_one_or_all(gathered)
    var idx = rs_indices(binding, PythonObject(n), n_samples, replace_obj, random_state, numeric_mode, context)
    var out = Python.list()
    for i in range(len(arrays)):  # driver-loop: over the argument arrays
        _ = py_call(out.append, _rs_take(arrays[i], idx, context))
    _ = idx^
    return _rs_one_or_all(out)


def rs_operation(operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    if operation == "family_rs_bootstrap":
        return _rs_bootstrap(args, context)
    if operation == "family_rs_permutation_test":
        return _rs_permutation_test(args, context)
    if operation == "family_rs_monte_carlo_integrate":
        return _rs_monte_carlo(args, context)
    if operation == "family_rs_resample_indices":
        return _rs_resample_indices(args, context)
    if operation == "family_rs_resample":
        return _rs_resample(args, context)
    raise Error("RuntimeError: unknown resample operation " + operation)
