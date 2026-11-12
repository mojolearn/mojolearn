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
from bindings.native_loaded_module_door import native_loaded_module
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.optimizer_public_words_unpack import PYTHON_ERROR_PENDING, optimizer_python_error
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.classical_labels import classical_accuracy, labels_shape_of
from bindings.admitted_input import classical_view
from bindings.classical_arrays_door import classical_address, classical_wrap_array
from bindings.admitted_input import admit_x
from bindings.decomp_public_lifecycle_unpack import decomp_public_begin
from bindings.decomp_public_lifecycle_pack import decomp_public_finish
from x_decomp.decomp_public_lifecycle_run import decomp_public_method
from bindings.public_lifecycle_owner_door import lifecycle_end
from bindings.decomp_kit_native_door import (
    DecompDoor, decomp_float, decomp_int, decomp_seed, decomp_seed_word, decomp_mode, decomp_entries, decomp_is_a,
    decomp_kit_unpack, decomp_kit_publish, decomp_kit_out_shape, decomp_door_borrow, decomp_door_input,
    decomp_door_resident_hit, decomp_door_resident_input,
)
from bindings.decomp_kit_run import DKit, DK_SCALE, DK_ADDS
from bindings.mcd_run import McdFit, mcd_fit_run, mcd_labels_run, mcd_score_run, mcd_mahal
from core.decomp_native_impl import decomp_f32_to_i32
from std.python._cpython import GILReleased
from std.memory import unsafe_memcpy


def _name(model: PythonObject) raises -> String:
    return String(py=py_type(model).__name__)


def _fitted(mut door: DecompDoor, mut kit: DKit, fields: PythonObject, name: String, rows: Int,
            context: PythonObject) raises -> Int:
    # UNPACK a fitted float32 Array (location_ (p,) as 1 x p, _precision p x p).
    var hit = decomp_door_resident_hit(door, kit, fields[name], name)  # frozen and resident: no re-admission
    if hit >= 0:
        return hit
    var array = admit_x(fields[name], 1 if rows == 1 else 2, name, context)
    var size = Int(py=array.size)
    # On a query its device copy is kept across calls (the model's record).
    var held = decomp_door_resident_input(door, kit, fields[name], name, array, Int(py=classical_address(array)),
                                          array, rows, size // rows)
    if held >= 0:
        return held
    return decomp_door_borrow(door, kit, array, rows, size // rows)


def _typed(n: Int, typestr: String, code: String, item: Int, context: PythonObject) raises -> PythonObject:
    # A zero-filled (n,) Array of a fixed small dtype.
    var output = context["output_context"]
    var store = py_call(output["array_type"], PythonObject(code), py_call(output["native_bytes_type"], n * item))
    return classical_wrap_array(store, Python.tuple(n), typestr, "C", output)


def _support_pack(support: List[UInt8], n: Int, context: PythonObject) raises -> PythonObject:
    # PACK: the support words as the public '<u1' (n,) Array.
    var out = _typed(n, "<u1", "B", 1, context)
    if n:
        unsafe_memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=out._addr)),
                      src=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(support.unsafe_ptr())), count=n)
        _ = out  # keep the Python object alive through the copy above
    return out


def _labels_pack(mut kit: DKit, chosen: Int, rows: Int, context: PythonObject) raises -> PythonObject:
    # PACK EllipticEnvelope.predict: the +1 / -1 float32 matrix as int32 labels.
    var labels = _typed(rows, "<i4", "i", 4, context)
    if chosen >= 0:
        decomp_f32_to_i32(kit.ptr(chosen), MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=labels._addr)),
                          rows)
        _ = labels  # keep the Python object alive through the native call above
    return labels


def _check_fitted(model: PythonObject, fields: PythonObject) raises:
    if not py_contains(fields, "_precision"):
        raise Error("RuntimeError: " + _name(model) + " is not fitted; call fit first")


def mcd_fit(model: PythonObject, args: PythonObject, envelope: Bool, predict: Bool,
            context: PythonObject) raises -> PythonObject:
    # `MinCovDet.fit` (and fit_predict): one UNPACK, one RUN, one PACK.
    var fields = py_fields(model)
    var mode = decomp_mode(fields["numeric_mode"], context)
    py_setitem(fields, "numeric_mode_", mode)
    var entries = decomp_entries(mode, context)
    var door = DecompDoor(entries, context["output_context"])
    var kit = decomp_kit_unpack(entries)
    try:
        # UNPACK: X, the seed, h and the options.
        var x = decomp_door_input(door, kit, args[0], "X", context)
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
        var seed_word = decomp_seed_word(seed)
        var assume_centered = Bool(py=fields["assume_centered"])
        var contamination = decomp_float(fields["contamination"]) if envelope else 0.0
        # RUN: fast_mcd, the corrections, the precision, the distances (and
        # fit_predict's labels over the fit's own distances: predict(X) on the
        # X the fit converted computes exactly these mahalanobis words again).
        var fit = McdFit()
        var chosen = -1
        var py = Python()
        with GILReleased(py):
            fit = mcd_fit_run(kit, x, h, seed_word, assume_centered, envelope, contamination)
            if predict:
                chosen = mcd_labels_run(kit, fit.dist, fit.offset)
        # PACK: the fitted fields, once.
        py_setitem(fields, "raw_location_", decomp_kit_out_shape(door, kit, fit.raw_loc, Python.tuple(p)))
        py_setitem(fields, "raw_covariance_", decomp_kit_publish(door, kit, fit.raw_cov))
        py_setitem(fields, "raw_support_", _support_pack(fit.raw_support, n, context))
        py_setitem(fields, "location_", decomp_kit_out_shape(door, kit, fit.loc, Python.tuple(p)))
        py_setitem(fields, "covariance_", decomp_kit_publish(door, kit, fit.cov))
        var stored = decomp_kit_publish(door, kit, fit.precision)
        py_setitem(fields, "_precision", stored)
        py_setitem(fields, "precision_", stored if Bool(py=fields["store_precision"]) else PythonObject(None))
        py_setitem(fields, "support_", _support_pack(fit.support, n, context))
        py_setitem(fields, "dist_", decomp_kit_out_shape(door, kit, fit.dist, Python.tuple(n)))
        py_setitem(fields, "n_features_in_", PythonObject(p))
        if envelope:
            py_setitem(fields, "offset_", PythonObject(fit.offset))
        if not predict:
            return model
        return _labels_pack(kit, chosen, n, context)
    finally:
        kit.close()
        door.keep_alive()  # the borrowed owners outlive every RUN read


def mcd_operation(model: PythonObject, operation: String, args: PythonObject, envelope: Bool,
                  context: PythonObject) raises -> PythonObject:
    # `envelope`: the model is an EllipticEnvelope, resolved once in mcd_c
    # and passed down (N345; it was asked twice per call).
    var fields = py_fields(model)
    if operation == "fit" or operation == "fit_predict":
        if envelope:
            var c = fields["contamination"]
            var op = native_loaded_module("operator")
            if not (Bool(py=py_call(op.lt, PythonObject(0), c)) and Bool(py=py_call(op.le, c, PythonObject(0.5)))):
                raise Error("contamination must be in (0, 0.5]")
        return mcd_fit(model, args, envelope, operation == "fit_predict", context)
    _check_fitted(model, fields)
    var entries = decomp_entries(py_get(fields, "numeric_mode_"), context)
    var door = DecompDoor(entries, context["output_context"])
    var kit = decomp_kit_unpack(entries)
    try:
        if operation == "get_precision":
            var p = Int(py=fields["_precision"].shape[0])
            return decomp_kit_publish(door, kit, _fitted(door, kit, fields, "_precision", p, context))
        # A query: the fitted location and precision stay on the device across calls.
        door.owner = model
        # UNPACK: X, then the fitted location and precision (and offset_).
        var x = decomp_door_input(door, kit, args[0], "X", context)
        var p = kit.ncols(x)
        var loc = _fitted(door, kit, fields, "location_", 1, context)
        var precision = _fitted(door, kit, fields, "_precision", p, context)
        var py = Python()
        if operation == "score":
            var value = Float64(0.0)
            with GILReleased(py):
                value = mcd_score_run(kit, x, loc, precision)
            return PythonObject(value)
        if operation == "mahalanobis" or operation == "score_samples":
            var out = -1
            with GILReleased(py):
                out = mcd_mahal(kit, x, loc, precision)
                if operation == "score_samples":
                    out = kit.ew(DK_SCALE, out, scalar=-1.0)
            return decomp_kit_out_shape(door, kit, out, Python.tuple(kit.nrows(out)))
        if operation == "decision_function" or operation == "predict":
            var offset = Float64(py=fields["offset_"])
            var out = -1
            var rows = 0
            with GILReleased(py):
                var d = mcd_mahal(kit, x, loc, precision)
                rows = kit.nrows(d)
                if operation == "predict":
                    out = mcd_labels_run(kit, d, offset)
                else:
                    var negated = kit.ew(DK_SCALE, d, scalar=-1.0)
                    out = kit.ew(DK_ADDS, negated, scalar=-offset)
            if operation == "predict":
                return _labels_pack(kit, out, rows, context)
            return decomp_kit_out_shape(door, kit, out, Python.tuple(rows))
        raise Error("unknown native MinCovDet operation")
    finally:
        kit.close()
        door.keep_alive()  # the borrowed owners outlive every RUN read


def elliptic_score(model: PythonObject, args: PythonObject, envelope: Bool, context: PythonObject) raises -> PythonObject:
    """EllipticEnvelope.score: the native predict, then the (weighted) match
    fraction of the x_metrics accuracy program in the fitted mode, the
    prediction handed straight to it (`accuracy_fraction`)."""
    var prediction = mcd_operation(model, "predict", Python.tuple(args[0]), envelope, context)
    var n = labels_shape_of(prediction, context)[0]
    var y = args[1]
    var sample_weight = args[2]
    # `len(y)`: the row count from the shape (an Array's slots, a buffer's
    # shape, a list's length).
    var y_shape = labels_shape_of(y, context)
    if len(y_shape) == 0 or y_shape[0] != n:
        raise Error("y and X have different numbers of rows")
    if sample_weight is not None:
        var view = classical_view(sample_weight, context, "sample_weight")
        if Int(py=view.ndim) != 1:
            raise Error("mojolearn: sample_weight must be 1-D, got " + String(Int(py=view.ndim)) + "-D")
        if Int(py=view.shape[0]) != n:
            raise Error("sample_weight and X have different numbers of rows")
    try:
        return PythonObject(classical_accuracy(y, prediction, sample_weight, py_fields(model)["numeric_mode_"], context))
    except error:
        if String(error).startswith("ZeroDivisionError"):
            raise Error("ZeroDivisionError: Weights sum to zero, can't be normalized")
        raise error^


def mcd_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        if len(call) != 4:
            raise Error("TypeError: mcd_native_call requires model, operation, arguments and connected context")
        var b = native_loaded_module("builtins")
        if py_type(call[1]) is not b.str or py_type(call[2]) is not b.tuple or py_type(call[3]) is not b.dict:
            raise Error("TypeError: native MinCovDet requires an operation, an argument tuple and its context")
        var context = call[3]
        classical_connect_numpy(context)
        var model = call[0]
        var operation = String(py=call[1])
        var args = call[2]
        var kind = String("")
        var envelope = decomp_is_a(model, context["elliptic_envelope_class"])
        if envelope:
            kind = "outlier_detector"  # float64 scores for callers outside the package
        var method = decomp_public_method(operation)
        var depth = decomp_public_begin(model, method, args, True, True, context)
        var result = PythonObject(None)
        try:
            if operation == "elliptic_score":
                result = elliptic_score(model, args, envelope, context)
            else:
                result = mcd_operation(model, operation, args, envelope, context)
        finally:
            lifecycle_end(model, method, depth)
        result = decomp_public_finish(model, method, args, result, kind, context)
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
        if String(error).startswith("ZeroDivisionError:"):
            return raise_python_exception(error_without_class(error, "ZeroDivisionError"), ExceptionType("PyExc_ZeroDivisionError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
