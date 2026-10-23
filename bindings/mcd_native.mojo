# SPDX-License-Identifier: Apache-2.0
# The native owner of python/mojolearn/_expansion_decomp.py's MinCovDet and
# EllipticEnvelope: fit, score, mahalanobis, get_precision, score_samples,
# decision_function, predict and fit_predict are each one call into mcd_c.
# The owner runs the steps the Python shells ran (fast_mcd's one-feature
# shortcut or the one `x_decomp_mcd` call, correct_covariance,
# reweight_covariance, `_pinvh`, `_mahal`, `_slogdet`, the offset's order
# statistics) through bindings/decomp_kit_native.mojo: the same x_decomp
# entries with the same words in the same order, so the bits are unchanged.
# fit_predict labels the X the fit converted and uploaded (one of each).
#
# The fitted location and precision are read back from `location_` and the
# private `_precision` (kept also when store_precision=False, for
# get_precision), the values the Python shell held as device matrices.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from std.math import ceil, floor
from std.memory import bitcast
from bindings.optimizer_public_words import PYTHON_ERROR_PENDING, optimizer_python_error
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_arrays import classical_address, classical_host_f32, classical_wrap_array
from bindings.decomp_kit_native import (
    DecompKit, DecompF32, decomp_float, decomp_int, decomp_seed, decomp_seed_word, decomp_mode, decomp_entries,
    decomp_input, decomp_is_a, XD_SUB, XD_ADD, XD_MUL, XD_SCALE, XD_ABS, XD_RECIP, XD_SELECT, XD_LOGS,
    XD_ADDS, XD_LE,
)
from core.decomp_native_impl import (
    decomp_f32_to_u8, decomp_i32_to_u8, decomp_u8_to_f32, decomp_f32_to_i32, decomp_count_u8,
)

comptime MCD_F32_EPS = 1.1920928955078125e-07
comptime MCD_LOG_2PI = 1.8378770664093453
comptime MCD_TINY = 1.1754943508222875e-38


def _name(model: PythonObject) raises -> String:
    return String(py=py_type(model).__name__)


def _matrix(mut kit: DecompKit, source: PythonObject, name: String, context: PythonObject) raises -> Int:
    var array = decomp_input(source, name, context)
    return kit.input(array, Int(py=array.shape[0]), Int(py=array.shape[1]))


def _fitted(mut kit: DecompKit, fields: PythonObject, name: String, rows: Int, context: PythonObject) raises -> Int:
    # A fitted float32 Array (location_ (p,) as 1 x p, _precision p x p).
    var array = classical_host_f32(fields[name], 1 if rows == 1 else 2, name, context)
    var size = Int(py=array.size)
    return kit.input(array, rows, size // rows)


def _typed(n: Int, typestr: String, code: String, item: Int, context: PythonObject) raises -> PythonObject:
    # A zero-filled (n,) Array of a fixed small dtype.
    var output = context["output_context"]
    var store = py_call(output["array_type"], PythonObject(code), py_call(output["native_bytes_type"], n * item))
    return classical_wrap_array(store, Python.tuple(n), typestr, "C", output)


def _f32_below(x: Float64) -> Float32:
    # `_f32_below`: the largest float32 strictly below x.
    var f = Float32(x)
    if Float64(f) < x:
        return f
    var u = bitcast[DType.uint32](f)
    if f > 0.0:
        u -= 1
    elif f == 0.0:
        u = UInt32(0x80000001)
    else:
        u += 1
    return bitcast[DType.float32](u)


def _pinvh(mut kit: DecompKit, a: Int) raises -> Int:
    # `_pinvh`: V diag(1/w) V^T over |w| > max|w| * n * float32 eps.
    var wv = kit.eigh(a)
    var w = wv[0]
    var wmax = Float64(0.0)
    if kit.ncols(w):
        var top = kit.reduce(w, 0)
        wmax = kit.word(top)
    var cut = Float64(Float32(wmax * Float64(kit.nrows(a)) * MCD_F32_EPS))
    var absolute = kit.ew(XD_ABS, w)
    var inverse = kit.ew(XD_RECIP, w)
    var zero = kit.zeros(1, 1)
    var kept = kit.ew(XD_SELECT, absolute, inverse, zero, cut)
    var scaled = kit.ew(XD_MUL, wv[1], kept)
    return kit.mm(scaled, wv[1], tb=True)


def _emp_cov(mut kit: DecompKit, xs: Int, assume_centered: Bool) raises -> Int:
    # `_emp_cov`: (X - mean)^T (X - mean) / n.
    var rows = kit.nrows(xs)
    if assume_centered:
        var gram = kit.mm(xs, xs, ta=True)
        return kit.ew(XD_SCALE, gram, scalar=1.0 / Float64(rows))
    var mean = kit.colmean(xs)
    var centered = kit.ew(XD_SUB, xs, mean)
    var gram = kit.mm(centered, centered, ta=True)
    return kit.ew(XD_SCALE, gram, scalar=1.0 / Float64(rows))


def _mahal(mut kit: DecompKit, x: Int, loc: Int, precision: Int) raises -> Int:
    var centered = kit.ew(XD_SUB, x, loc)
    var product = kit.mm(centered, precision)
    var terms = kit.ew(XD_MUL, product, centered)
    return kit.rowsum(terms)


def _masked_cov(mut kit: DecompKit, m: Int, mask: Int, assume_centered: Bool) raises -> Tuple[Int, Int]:
    # `_masked_cov`: the location and covariance of the masked rows.
    var count = kit.total_word(mask)
    var masked = kit.ew(XD_MUL, m, mask)
    var loc: Int
    var centered: Int
    if assume_centered:
        loc = kit.zeros(1, kit.ncols(m))
        centered = masked
    else:
        var sums = kit.colsum(masked)
        loc = kit.ew(XD_SCALE, sums, scalar=1.0 / count)
        var shifted = kit.ew(XD_SUB, m, loc)
        centered = kit.ew(XD_MUL, shifted, mask)
    var rows = kit.nrows(centered)
    var cols = kit.ncols(centered)
    var gram: Int
    if kit.has("x_decomp_dev_mcd_cov") and kit.use(centered) and rows > 0 and cols > 0:
        gram = kit.device_output(cols, cols)
        var dc = kit.device(centered)
        _ = py_call(kit.entries["x_decomp_dev_mcd_cov"], dc, kit.matrices[gram].device, Python.list(rows, cols))
    else:
        gram = kit.mm(centered, centered, ta=True)
    return (loc, kit.ew(XD_SCALE, gram, scalar=1.0 / count))


def _slogdet(mut kit: DecompKit, a: Int) raises -> Tuple[Float64, Float64]:
    # `_slogdet`: (sign, log|det|) from the LU factor (sums ascending).
    var factor = kit.lu(a)
    var aux = kit.lu_aux(factor[0], factor[1])
    if kit.word(aux[0], 1) > 0:
        return (Float64(0.0), -Float64.MAX * 2.0)
    var negative = Int(kit.word(aux[0], 2)) + Int(kit.word(aux[0], 3))
    var absolute = kit.ew(XD_ABS, aux[1])
    var logs = kit.ew(XD_LOGS, absolute, scalar=MCD_TINY)
    var ld = kit.total_word(logs)
    return (Float64(-1.0) if negative % 2 else Float64(1.0), ld)


def _mask_store(mut kit: DecompKit, mask: Int, n: Int, context: PythonObject) raises -> PythonObject:
    # An n x 1 0/1 float32 matrix as the public '<u1' (n,) support.
    var support = _typed(n, "<u1", "B", 1, context)
    var source = kit.host(mask)
    decomp_f32_to_u8(DecompF32(unsafe_from_address=Int(py=classical_address(source))),
                     MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=support._addr)), n)
    _ = source^
    return support


def _int_store(store: PythonObject, n: Int, context: PythonObject) raises -> PythonObject:
    # `_support_of`: a 0/1 int32 membership store as the public '<u1' (n,) support.
    var support = _typed(n, "<u1", "B", 1, context)
    decomp_i32_to_u8(MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=py_call(store.buffer_info)[0])),
                     MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=support._addr)), n)
    return support


def _support_count(support: PythonObject, n: Int) raises -> Int:
    return decomp_count_u8(MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=support._addr)), n)


def _mcd_1d(mut kit: DecompKit, x: Int, h: Int, context: PythonObject) raises -> Tuple[Int, Int, PythonObject, Int]:
    # `MinCovDet._mcd_1d_native`: (loc, cov, support, dist) for one feature.
    var n = kit.nrows(x)
    var order = kit.order(x)
    var xs = kit.take_rows(x, order)
    var loc: Int
    var support: PythonObject
    var chosen: Int
    if h < n:
        var upper = kit.rows_of(xs, h, n)
        var lower = kit.rows_of(xs, 0, n - h)
        var diff = kit.ew(XD_SUB, upper, lower)
        var starts = kit.argmin_all(diff)
        var ends = kit.take_rows(xs, starts, h)
        var begins = kit.take_rows(xs, starts)
        var sums = kit.ew(XD_ADD, ends, begins)
        var mids = kit.ew(XD_SCALE, sums, scalar=0.5)
        loc = kit.colmean(mids)
        var shifted = kit.ew(XD_SUB, x, loc)
        var centered = kit.ew(XD_ABS, shifted)
        var selected = kit.select_smallest(centered, h)
        chosen = kit.take_rows(x, selected[0])
        if selected[1] >= 0:
            support = _mask_store(kit, selected[1], n, context)
        else:
            support = _int_store(selected[2], n, context)
    else:
        loc = kit.colmean(x)
        chosen = x
        # every row is in the support: `_support_of(array("i", [1]) * n, n)`
        var ones = py_call(context["output_context"]["array_type"], PythonObject("i"), Python.list(1)) * PythonObject(n)
        support = _int_store(ones, n, context)
    var cov = _emp_cov(kit, chosen, False)
    var precision = _pinvh(kit, cov)
    return (loc, cov, support, _mahal(kit, x, loc, precision))


def mcd_fit(model: PythonObject, args: PythonObject, mut kit: DecompKit, context: PythonObject) raises -> Tuple[Int, Int]:
    # `MinCovDet.fit` after the kit is connected: (X, dist_) as kit matrices.
    var fields = py_fields(model)
    var x = _matrix(kit, args[0], "X", context)
    var n = kit.nrows(x)
    var p = kit.ncols(x)
    if n < 2:
        raise Error("Found array with " + String(n) + " sample(s) (shape=(" + String(n) + ", " + String(p)
                    + ")) while a minimum of 2 is required by MinCovDet.")
    var seed = decomp_seed(fields["random_state"])
    py_setitem(fields, "_seed", seed)
    py_setitem(fields, "_draws", PythonObject(0))
    var fraction = fields["support_fraction"]
    var h: Int
    if fraction is None:
        h = min((n + p + 2) // 2, n)        # min(int(ceil(0.5 * (n + p + 1))), n)
    else:
        h = decomp_int(fraction * PythonObject(n))
    var loc: Int
    var cov: Int
    var support: PythonObject
    var dist: Int
    if p == 1:
        var one = _mcd_1d(kit, x, h, context)
        loc = one[0]
        cov = one[1]
        support = one[2]
        dist = one[3]
    else:
        # fast_mcd in ONE binding call; only the O(1) plan integers here, with
        # the float expressions they always used.
        var plan = Python.list(0, 0, 0, 0, 0, 0, 0)
        if n > 500:
            var n_sub = n // 300
            var n_ss = n // n_sub
            var n_m = min(1500, n)
            var share = Float64(h) / Float64(n)
            plan = Python.list(n_sub, n_ss, Int(ceil(Float64(n_ss) * share)), max(10, 500 // n_sub), n_m,
                               Int(ceil(Float64(n_m) * share)), 10 if n > 1500 else 1)
        loc = kit.zeros(1, p)
        cov = kit.zeros(p, p)
        dist = kit.zeros(n, 1)
        var store = py_call(context["output_context"]["array_type"], PythonObject("i"),
                            py_call(context["output_context"]["native_bytes_type"], 4 * n))
        var params = Python.list(n, p, h, decomp_seed_word(seed))
        _ = py_call(params.extend, plan)
        var ax = kit.address(x)
        _ = py_call(kit.entries["x_decomp_mcd"], ax, kit.address(loc), kit.address(cov), py_call(store.buffer_info)[0],
                    kit.address(dist), params)
        kit.wrote_host(loc)
        kit.wrote_host(cov)
        kit.wrote_host(dist)
        support = _int_store(store, n, context)
    var assume_centered = Bool(py=fields["assume_centered"])
    if assume_centered:
        loc = kit.zeros(1, p)
        var weights = kit.zeros(n, 1)
        decomp_u8_to_f32(MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=support._addr)),
                         DecompF32(unsafe_from_address=Int(py=kit.address(weights))), n)
        cov = _masked_cov(kit, x, weights, True)[1]
        var inverse = _pinvh(kit, cov)
        var product = kit.mm(x, inverse)
        var terms = kit.ew(XD_MUL, product, x)
        dist = kit.rowsum(terms)
    py_setitem(fields, "raw_location_", kit.out_shape(loc, Python.tuple(p)))
    py_setitem(fields, "raw_covariance_", kit.publish(cov))
    py_setitem(fields, "raw_support_", support)
    # correct_covariance: consistency at the normal model (dist_ is rescaled)
    var n_support = _support_count(support, n)
    var corr = decomp_float(py_call(kit.entries["x_decomp_consistency_factor"], PythonObject(Float64(p)), PythonObject(Float64(n_support)),
                                     PythonObject(Float64(n))))
    dist = kit.ew(XD_SCALE, dist, scalar=1.0 / corr)
    # reweight_covariance: the mask dist < thr (`le` against the float32 just below thr)
    var thr = decomp_float(py_call(kit.entries["x_decomp_chi2_quantile"], PythonObject(Float64(p)), PythonObject(Float64(0.025))))
    var below = kit.const(_f32_below(thr))
    var mask = kit.ew(XD_LE, dist, below)
    var moments = _masked_cov(kit, x, mask, assume_centered)
    var factor = decomp_float(py_call(kit.entries["x_decomp_consistency_factor"], PythonObject(Float64(p)), PythonObject(Float64(0.975)),
                                       PythonObject(Float64(1.0))))
    var covr = kit.ew(XD_SCALE, moments[1], scalar=factor)
    var reweighted = _mask_store(kit, mask, n, context)
    var precision = _pinvh(kit, covr)
    py_setitem(fields, "location_", kit.out_shape(moments[0], Python.tuple(p)))
    py_setitem(fields, "covariance_", kit.publish(covr))
    var stored = kit.publish(precision)
    py_setitem(fields, "_precision", stored)
    py_setitem(fields, "precision_", stored if Bool(py=fields["store_precision"]) else PythonObject(None))
    py_setitem(fields, "support_", reweighted)
    var distances = _mahal(kit, x, moments[0], precision)
    py_setitem(fields, "dist_", kit.out_shape(distances, Python.tuple(n)))
    py_setitem(fields, "n_features_in_", PythonObject(p))
    return (x, distances)


def _offset(model: PythonObject, mut kit: DecompKit, dist: Int) raises:
    # EllipticEnvelope's offset_: the contamination percentile of -dist_.
    var fields = py_fields(model)
    var nv = kit.size(dist)
    var contamination = decomp_float(fields["contamination"])
    var q = 100.0 * contamination / 100.0 * Float64(nv - 1)
    var lo = Int(floor(q))
    var hi = min(lo + 1, nv - 1)
    var order = kit.order(dist)
    var vlo = -kit.word(dist, Int(kit.word(order, nv - 1 - lo)))
    var vhi = -kit.word(dist, Int(kit.word(order, nv - 1 - hi)))
    py_setitem(fields, "offset_", PythonObject(Float64(Float32(vlo + (vhi - vlo) * (q - Float64(lo))))))


def _check_fitted(model: PythonObject, fields: PythonObject) raises:
    if not py_contains(fields, "_precision"):
        raise Error("RuntimeError: " + _name(model) + " is not fitted; call fit first")


def _distances(mut kit: DecompKit, x: Int, fields: PythonObject, context: PythonObject) raises -> Int:
    var p = kit.ncols(x)
    var loc = _fitted(kit, fields, "location_", 1, context)
    var precision = _fitted(kit, fields, "_precision", p, context)
    return _mahal(kit, x, loc, precision)


def _labels(mut kit: DecompKit, x: Int, fields: PythonObject, context: PythonObject) raises -> PythonObject:
    # EllipticEnvelope.predict: 1 where -dist - offset_ >= 0, else -1.
    var d = _distances(kit, x, fields, context)
    var rows = kit.nrows(d)
    var negated = kit.ew(XD_SCALE, d, scalar=-1.0)
    var decision = kit.ew(XD_ADDS, negated, scalar=-Float64(py=fields["offset_"]))
    var labels = _typed(rows, "<i4", "i", 4, context)
    if not rows:
        return labels
    var flipped = kit.ew(XD_SCALE, decision, scalar=-1.0)
    var zero = kit.const(0.0)
    var ge = kit.ew(XD_LE, flipped, zero)
    var plus = kit.const(1.0)
    var minus = kit.const(-1.0)
    var chosen = kit.ew(XD_SELECT, ge, plus, minus, 0.5)
    var source = kit.host(chosen)
    decomp_f32_to_i32(DecompF32(unsafe_from_address=Int(py=classical_address(source))),
                      MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=labels._addr)), rows)
    _ = source^
    return labels


def mcd_operation(model: PythonObject, operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var envelope = decomp_is_a(model, context["elliptic_envelope_class"])
    if operation == "fit" or operation == "fit_predict":
        if envelope:
            var c = fields["contamination"]
            if not (Bool(py=PythonObject(0) < c) and Bool(py=c <= PythonObject(0.5))):
                raise Error("contamination must be in (0, 0.5]")
        var mode = decomp_mode(fields["numeric_mode"], context)
        py_setitem(fields, "numeric_mode_", mode)
        var kit = DecompKit(decomp_entries(mode, context), context["output_context"])
        try:
            var held = mcd_fit(model, args, kit, context)
            if envelope:
                _offset(model, kit, held[1])
            if operation == "fit":
                return model
            # predict(X) on the X the fit converted and uploaded.
            return _labels(kit, held[0], fields, context)
        finally:
            kit.close()
    if operation == "get_precision":
        _check_fitted(model, fields)
        var kit = DecompKit(decomp_entries(py_get(fields, "numeric_mode_"), context), context["output_context"])
        try:
            var p = Int(py=fields["_precision"].shape[0])
            return kit.publish(_fitted(kit, fields, "_precision", p, context))
        finally:
            kit.close()
    _check_fitted(model, fields)
    var kit = DecompKit(decomp_entries(py_get(fields, "numeric_mode_"), context), context["output_context"])
    try:
        var x = _matrix(kit, args[0], "X", context)
        if operation == "mahalanobis":
            var d = _distances(kit, x, fields, context)
            return kit.out_shape(d, Python.tuple(kit.nrows(d)))
        if operation == "score":
            # EmpiricalCovariance.score: X's covariance around location_ under the precision.
            var p = kit.ncols(x)
            var loc = _fitted(kit, fields, "location_", 1, context)
            var precision = _fitted(kit, fields, "_precision", p, context)
            var shifted = kit.ew(XD_SUB, x, loc)
            var cov = _emp_cov(kit, shifted, True)
            var product = kit.ew(XD_MUL, cov, precision)
            var trace = kit.total_word(product)
            var logdet = _slogdet(kit, precision)
            return PythonObject(-(Float64(p) * MCD_LOG_2PI) / 2.0 - 0.5 * trace + 0.5 * logdet[1])
        if operation == "score_samples" or operation == "decision_function":
            var d = _distances(kit, x, fields, context)
            var negated = kit.ew(XD_SCALE, d, scalar=-1.0)
            if operation == "score_samples":
                return kit.out_shape(negated, Python.tuple(kit.nrows(d)))
            var decision = kit.ew(XD_ADDS, negated, scalar=-Float64(py=fields["offset_"]))
            return kit.out_shape(decision, Python.tuple(kit.nrows(d)))
        if operation == "predict":
            return _labels(kit, x, fields, context)
        raise Error("unknown native MinCovDet operation")
    finally:
        kit.close()


def mcd_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        if len(call) != 4:
            raise Error("TypeError: mcd_native_call requires model, operation, arguments and connected context")
        var b = Python.import_module("builtins")
        if py_type(call[1]) is not b.str or py_type(call[2]) is not b.tuple or py_type(call[3]) is not b.dict:
            raise Error("TypeError: native MinCovDet requires an operation, an argument tuple and its context")
        var context = call[3]
        classical_connect_numpy(context)
        var result = mcd_operation(call[0], String(py=call[1]), call[2], context)
        return result.steal_data()
    except error:
        if String(error) == PYTHON_ERROR_PENDING:
            return optimizer_python_error(error^)
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if String(error).startswith("NotImplementedError:"):
            return raise_python_exception(error_without_class(error, "NotImplementedError"), ExceptionType("PyExc_NotImplementedError"))
        if String(error).startswith("AttributeError:"):
            return raise_python_exception(error_without_class(error, "AttributeError"), ExceptionType("PyExc_AttributeError"))
        if String(error).startswith("ImportError:"):
            return raise_python_exception(error_without_class(error, "ImportError"), ExceptionType("PyExc_ImportError"))
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
