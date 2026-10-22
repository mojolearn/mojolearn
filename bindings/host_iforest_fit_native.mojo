# SPDX-License-Identifier: Apache-2.0
# Explicit CPU IsolationForest fit retains its existing seeded numerical ABI.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.classical_attributes import classical_connect_numpy, classical_check_layout, classical_getattribute
from bindings.classical_arrays import classical_address, classical_is_array, classical_is_f32, classical_c_order, classical_kind, classical_f64
from bindings.classical_pandas import classical_admit_value, _pd_type_fields
from bindings.classical_validate import RESHAPE_HINT
from bindings.host_tree_native import host_tree_constructor_fields
from bindings.host_kde_fit_native import _kde_prepare, _kde_cast, _kde_y, _kde_shape, _kde_fit_mode
from bindings.host_kde_native import _kde_option_value
from bindings.host_inference_scalars import host_inference_scalar
from bindings.host_iforest_scalar_repr import host_iforest_scalar_repr
from bindings.host_kernel_inference_native import _kernel_entry
from bindings.host_archive_native import host_archive_typed
from bindings.host_sparse_native import host_sparse_format
from bindings.native_owner_lifetime import native_release_admit
from bindings.native_buffer_owner import native_buffer_view
from bindings.native_numpy_metadata import native_numpy_dtype
from bindings.optimizer_public_words import stash_python_error
from bindings.sequential_admission import _one_of
from core.host_inference_impl import host_inference_finite


def _if_fit_alias(value: PythonObject, context: PythonObject, integer_only: Bool = False) raises -> Bool:
    if py_get(context, "ndarray") is None:
        return False
    var constructor = native_numpy_dtype(context)
    var codes = Python.tuple("q", "Q", "g")
    for i in range(2 if integer_only else 3):  # driver-loop: exact native numeric scalar aliases
        if py_type(value) is py_call(constructor, codes[i]).type:
            return True
    return False


def _if_fit_repr(value: PythonObject, context: PythonObject) raises -> String:
    # Numeric ndarray repr enters NumPy's Python array-printing module. Its
    # scalar contents and native dtype metadata suffice for these scalar knobs.
    var native_str = py_type(PythonObject(""))
    if py_type(value) is py_get(context, "ndarray"):
        host_inference_scalar(value, context)
        return host_iforest_scalar_repr(value, context)
    var admitted = value if _if_fit_alias(value, context) else _kde_option_value(value, context, True)
    var original = value
    if py_type(admitted) is native_str and py_type(value) is not native_str:
        var numpy_str = PythonObject(None)
        if py_get(context, "ndarray") is not None:
            numpy_str = py_call(native_numpy_dtype(context), "U").type
        if py_type(value) is not numpy_str:
            var mro = py_call(py_type(native_str).__dict__["__mro__"].__get__, py_type(value))
            for i in range(len(mro)):  # driver-loop: benign string subclass repr admission
                if mro[i] is native_str:
                    break
                var slot = py_get(_pd_type_fields(mro[i]), "__repr__")
                if slot is not None and slot is not native_str.__dict__["__repr__"]:
                    raise Error("TypeError: IsolationForest parameter errors cannot invoke Python repr hooks")
            original = admitted
    ref cpy = Python().cpython()
    var result = cpy.lib.get_function[PyObjectPtr]("PyObject_Repr")(original._obj_ptr)
    if not result:
        raise cpy.unsafe_get_error()
    return String(PythonObject(from_owned=result))


def _if_fit_number(value: PythonObject, context: PythonObject, integer: Bool = False) raises -> PythonObject:
    var admitted = value if _if_fit_alias(value, context) else _kde_option_value(value, context, True)
    ref cpy = Python().cpython()
    var pointer = cpy.lib.get_function[PyObjectPtr]("PyNumber_Long" if integer else "PyNumber_Float")(admitted._obj_ptr)
    if not pointer:
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=pointer)


def _if_fit_truth(value: PythonObject, context: PythonObject) raises -> Bool:
    var cls = py_type(value)
    if classical_is_array(value, context):
        native_release_admit(value, context)
        var slots = context["owner_array_slots"]
        var size = py_call(slots["size"].__get__, value, cls)
        if Int(py=size) != 1:
            raise Error("mojolearn: the truth value of an Array with more than one element is ambiguous")
        var view = py_call(slots["_mv"].__get__, value, cls)
        var dtype = py_call(slots["dtype"].__get__, value, cls)
        if py_type(view) is not context["host_iforest_memoryview_type"] or len(view) != 1:
            raise Error("TypeError: IsolationForest Array controls require one native buffer element")
        if py_type(dtype) is not py_type(PythonObject("")):
            raise Error("TypeError: IsolationForest Array controls require native dtype metadata")
        # Half Arrays store unsigned bit patterns. Ignore the sign of zero;
        # every nonzero finite, infinite or NaN magnitude has a true value.
        var result = (Int(py=view[0]) & 32767) != 0 if String(dtype) == "<f2" else Bool(py=view[0])
        _ = view^
        return result
    if py_type(value) is py_get(context, "ndarray"):
        host_inference_scalar(value, context, True)
        return Bool(py=value)
    if cls is py_type(Python.list()) or cls is py_type(Python.tuple()) or _one_of(cls, context["mapping_types"]):
        classical_admit_value(value, context)
        return Bool(py=value)
    var admitted = value if _if_fit_alias(value, context) else _kde_option_value(value, context, True)
    _ = admitted^
    return Bool(py=value)


def _if_fit_marker_after(marker: PythonObject, context: PythonObject) raises -> PythonObject:
    # The Python fit guard incremented then decremented its depth. Preserve
    # native scalar type promotion and half/single rounding in that state.
    var value = marker
    var dtype = PythonObject(None)
    var ndarray = py_type(marker) is py_get(context, "ndarray")
    if ndarray:
        dtype = marker.dtype
        value = marker.flat[0]
    elif _one_of(py_type(marker), context["scalar_types"]) or _if_fit_alias(marker, context):
        dtype = marker.dtype
    if dtype is None:
        ref cpy = Python().cpython()
        var one = PythonObject(1)
        var incremented = cpy.lib.get_function[PyObjectPtr]("PyNumber_Add")(value._obj_ptr, one._obj_ptr)
        if not incremented:
            var failure = stash_python_error()
            raise failure^
        var owner = PythonObject(from_owned=incremented)
        var result = cpy.lib.get_function[PyObjectPtr]("PyNumber_Subtract")(owner._obj_ptr, one._obj_ptr)
        if not result:
            var failure = stash_python_error()
            raise failure^
        return PythonObject(from_owned=result)
    var constructor = native_numpy_dtype(context)
    var numpy = py_get(context["modules"], "numpy")
    if py_type(numpy) is not context["host_iforest_module_type"]:
        raise Error("TypeError: IsolationForest fit depth requires native NumPy metadata")
    var version = py_get(py_call(context["host_iforest_module_dict"], numpy), "__version__")
    if py_type(version) is not py_type(PythonObject("")):
        raise Error("TypeError: IsolationForest fit depth requires native NumPy version text")
    var release = Int(String(String(version).split(".")[0]))
    var kind = String(dtype.kind)
    var width = Int(py=dtype.itemsize)
    var scalar = not ndarray or Int(py=marker.ndim) == 0
    if kind == "b":
        dtype = py_call(constructor, "i8")
        value = PythonObject(Int(Bool(py=value)))
        kind = "i"
    elif release < 2 and scalar:
        # NumPy1's strong scalar promotion predates NumPy2's weak Python int.
        if kind == "f" and width <= 8 or (kind == "u" and width == 8):
            dtype = py_call(constructor, "f8")
            kind = "f"
            width = 8
        elif kind == "i" or kind == "u":
            dtype = py_call(constructor, "i8")
    var result = value
    if kind == "f":
        var number = Float64(py=value)
        if width == 2:
            number = Float64(Float16(Float16(number) + Float16(1)) - Float16(1))
        elif width == 4:
            number = Float64(Float32(Float32(number) + Float32(1)) - Float32(1))
        else:
            number = (number + Float64(1)) - Float64(1)
        result = py_call(dtype.type, PythonObject(number))
    elif kind == "i" or kind == "u":
        result = py_call(dtype.type, value)
    else:
        raise Error("TypeError: IsolationForest fit depth requires native real scalar state")
    if ndarray and not scalar:
        # A one-item vector/matrix keeps its shape; a 0-D result is a scalar.
        var array = py_call(context["ndarray"], marker.shape, dtype=dtype)
        py_setitem(array.flat, 0, result)
        return array
    return result


def _if_fit_options(model: PythonObject, fields: PythonObject, context: PythonObject) raises:
    var options = Python.dict()
    var names = Python.tuple("n_estimators", "max_samples", "max_depth", "max_features", "bootstrap", "contamination", "random_state", "warm_start", "n_jobs", "verbose", "output_type")
    for i in range(len(names)):  # driver-loop: native attribute precedence for fixed constructor options
        options[names[i]] = classical_getattribute(model, names[i], context)
    var trees = _if_fit_number(options["n_estimators"], context, True)
    if Bool(py=trees < 1):
        raise Error("mojolearn IsolationForest: n_estimators must be at least 1, got " + _if_fit_repr(trees, context))
    var native_str = py_type(PythonObject(""))
    var native_bool = py_type(PythonObject(True))
    var samples = options["max_samples"]
    var admitted = samples if _if_fit_alias(samples, context) else _kde_option_value(samples, context, True)
    if py_type(admitted) is native_str:
        if String(py_call(admitted.lower)) != "auto":
            raise Error("mojolearn IsolationForest: max_samples=" + _if_fit_repr(samples, context) + " is not a name; it is 'auto', a positive int, or a float in (0.0, 1.0]")
        py_setitem(fields, "_max_samples_mode", 0)
        py_setitem(fields, "_max_samples_int", 256)
        py_setitem(fields, "_max_samples_frac", 1.0)
    elif _one_of(py_type(samples), py_get(context, "bool_types", Python.tuple(native_bool))):
        raise Error("mojolearn IsolationForest: max_samples must be 'auto', an int, or a float, not a bool")
    elif _one_of(py_type(samples), context["integer_types"]) or _if_fit_alias(samples, context, True):
        var count = _if_fit_number(samples, context, True)
        if Bool(py=count <= 0):
            raise Error("mojolearn IsolationForest: max_samples must be a positive integer, got " + _if_fit_repr(samples, context))
        py_setitem(fields, "_max_samples_mode", 1)
        fields["_max_samples_int"] = count
        py_setitem(fields, "_max_samples_frac", 1.0)
    else:
        var fraction = _if_fit_number(samples, context)
        if not (Float64(py=fraction) > 0.0 and Float64(py=fraction) <= 1.0):
            raise Error("mojolearn IsolationForest: float max_samples must be in (0.0, 1.0], got " + _if_fit_repr(samples, context))
        py_setitem(fields, "_max_samples_mode", 2)
        py_setitem(fields, "_max_samples_int", 256)
        fields["_max_samples_frac"] = fraction
    var depth = options["max_depth"]
    fields["_max_depth"] = PythonObject(-1) if depth is None else _if_fit_number(depth, context, True)
    if depth is not None and Bool(py=fields["_max_depth"] < 1):
        raise Error("mojolearn IsolationForest: max_depth is None (their auto) or a positive int, got " + _if_fit_repr(depth, context))
    var features = options["max_features"]
    if _one_of(py_type(features), py_get(context, "bool_types", Python.tuple(native_bool))):
        raise Error("mojolearn IsolationForest: max_features must be an int or a float, not a bool")
    var features_error = String("mojolearn IsolationForest: max_features must be an int in [1, n_features] or a float in (0.0, 1.0], got ")
    if _one_of(py_type(features), context["integer_types"]) or _if_fit_alias(features, context, True):
        var count = _if_fit_number(features, context, True)
        if Bool(py=count < 1):
            raise Error(features_error + _if_fit_repr(features, context))
        py_setitem(fields, "_max_features_mode", 1)
        fields["_max_features_int"] = count
        py_setitem(fields, "_max_features_frac", 1.0)
    else:
        var fraction = _if_fit_number(features, context)
        if not (Float64(py=fraction) > 0.0 and Float64(py=fraction) <= 1.0):
            raise Error(features_error + _if_fit_repr(features, context))
        py_setitem(fields, "_max_features_mode", 0)
        py_setitem(fields, "_max_features_int", 0)
        fields["_max_features_frac"] = fraction
    var contamination = options["contamination"]
    admitted = contamination if _if_fit_alias(contamination, context) else _kde_option_value(contamination, context, True)
    var contamination_error = String("mojolearn IsolationForest: contamination must be 'auto' or a float in the range (0.0, 0.5], got ")
    if py_type(admitted) is native_str:
        if String(py_call(admitted.lower)) != "auto":
            raise Error(contamination_error + _if_fit_repr(contamination, context))
        py_setitem(fields, "_contamination_auto", True)
        py_setitem(fields, "_contamination", 0.0)
    else:
        var fraction = _if_fit_number(contamination, context)
        if not (Float64(py=fraction) > 0.0 and Float64(py=fraction) <= 0.5):
            raise Error(contamination_error + _if_fit_repr(contamination, context))
        py_setitem(fields, "_contamination_auto", False)
        fields["_contamination"] = fraction
    var random_state = options["random_state"]
    var seed = PythonObject(0) if random_state is None else _if_fit_number(random_state, context, True)
    if Bool(py=seed < 0) or Bool(py=seed > PythonObject(Int64(4294967295))):
        raise Error("mojolearn IsolationForest: expected 0 <= random_state <= 2**32 - 1, got " + _if_fit_repr(random_state, context))
    if _if_fit_truth(options["warm_start"], context):
        raise Error("NotImplementedError: mojolearn IsolationForest: warm_start=True is not supported (cuML raises UnsupportedOnGPU for it too)")
    if options["n_jobs"] is not None:
        raise Error("NotImplementedError: mojolearn IsolationForest: n_jobs is refused; it is scikit-learn's CPU thread count and there is no CPU path in this library to spread")
    if _if_fit_truth(options["verbose"], context):
        raise Error("NotImplementedError: mojolearn IsolationForest: verbose is refused; this implementation prints no log lines, so accepting it would be accepting-and-ignoring")
    if options["output_type"] is not None:
        raise Error("NotImplementedError: mojolearn IsolationForest: output_type is a cuML-internal array-type selector; this package returns mojolearn Arrays")
    fields["_n_estimators"] = trees
    py_setitem(fields, "_bootstrap", _if_fit_truth(options["bootstrap"], context))
    fields["_seed"] = seed


def _if_fit_body(model: PythonObject, fields: PythonObject, raw: PythonObject, copied: Bool, est: String, context: PythonObject) raises:
    _if_fit_options(model, fields, context)
    if Int(py=raw.ndim) != 2:
        raise Error("mojolearn: X must be 2-D, got " + String(Int(py=raw.ndim)) + "-D shape " + _kde_shape(raw.shape))
    var kind = classical_kind(raw, context)
    if kind == "U" or kind == "S":
        raise Error("mojolearn: X has buffer format '" + String(native_buffer_view(raw).format) + "', which is not a numeric array")
    if kind == "O":
        var flat = raw.flat
        for i in range(Int(py=raw.size)):  # driver-loop: native object scalars before the numeric packing phase
            var item = flat[i]
            if py_type(item) is py_type(PythonObject("")):
                raise Error("mojolearn: X holds a str, not a number")
    var x = _kde_cast(raw, context, "X")
    py_setitem(fields, "input_copied_", copied)
    var mode = _kde_fit_mode(model, fields, context)
    if mode is not None and py_type(mode) is not py_type(PythonObject("")):
        raise Error("TypeError: IsolationForest numeric mode requires an exact native string")
    if mode is not None and String(mode) != "identical":
        raise Error("mojolearn: " + est + " runs IDENTICAL only on the host; this model was saved '" + String(mode) + "'")
    try:
        host_inference_finite(Int(py=classical_address(x)), Int(py=x.size))
    except:
        raise Error("mojolearn IsolationForest: X contains NaN or infinity")
    # Explicit CPU kernels have no resident registry. Old CPU release simply
    # removed the optional bookkeeping; callback-bearing finalizers are refused.
    if py_get(fields, "_if_finalizer") is not None:
        raise Error("TypeError: IsolationForest CPU fit cannot invoke a Python resident finalizer")
    _ = py_call(fields.pop, "_if_token", PythonObject(None))
    _ = py_call(fields.pop, "_if_finalizer", PythonObject(None))
    fields["_x"] = x
    fields["n_features_in_"] = x.shape[1]
    var output = host_archive_typed(Python.tuple(1), "<f4", context["output_context"])
    var info = host_archive_typed(Python.tuple(3), "<f8", context["output_context"])
    var params = Python.list(x.shape[0], x.shape[1], 1, fields["_n_estimators"], fields["_max_samples_mode"], fields["_max_samples_int"], fields["_max_samples_frac"], fields["_max_depth"], fields["_max_features_mode"], fields["_max_features_int"], fields["_max_features_frac"], Int(Bool(py=fields["_bootstrap"])), fields["_seed"], Int(Bool(py=fields["_contamination_auto"])), fields["_contamination"], 0)
    var entry = _kernel_entry("iforest_run", context)
    try:
        # One query row is the same first row previously copied by x[:1].
        _ = py_call(entry, classical_address(x), classical_address(x), classical_address(output), 0, classical_address(info), params)
    except error:
        raise Error("HostIForestKernelError: " + String(error))
    _ = x^
    _ = output^
    py_setitem(fields, "offset_", Float64(Float32(classical_f64(info, 0))))
    py_setitem(fields, "max_samples_", Int(classical_f64(info, 1)))
    _ = info^


def host_iforest_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    classical_check_layout(model, context)
    var fields = host_tree_constructor_fields(model, context)
    var native_type = py_type(py_type(PythonObject(0)))
    var mro = py_call(native_type.__dict__["__mro__"].__get__, py_type(model))
    var known = False
    var setter_found = False
    for i in range(len(mro)):  # driver-loop: fixed native estimator ancestry
        var family = py_get(context["host_inference_kinds"], mro[i])
        known = known or (family is not None and String(family) == "iforest")
        if py_get(_pd_type_fields(mro[i]), "_mojolearn_after_fit") is not None:
            raise Error("TypeError: IsolationForest fit does not invoke Python after-fit hooks")
        var setter = py_get(_pd_type_fields(mro[i]), "__setattr__")
        if not setter_found and setter is not None:
            if setter is not context["host_density_object_setattr"]:
                raise Error("TypeError: IsolationForest fit does not invoke Python attribute assignment hooks")
            setter_found = True
    if not known:
        raise Error("TypeError: native IsolationForest fit requires a connected host class")
    var replacements = Python.tuple("_max_samples_mode", "_max_samples_int", "_max_samples_frac", "_max_depth", "_max_features_mode", "_max_features_int", "_max_features_frac", "_contamination_auto", "_contamination", "_n_estimators", "_bootstrap", "_seed", "_x", "n_features_in_", "offset_", "max_samples_", "input_copied_", "_mojolearn_fitting", "_mojolearn_fitted", "feature_names_in_", "_if_token", "_if_finalizer")
    for i in range(len(replacements)):  # driver-loop: old state lifetime admission before any replacement
        if py_contains(fields, replacements[i]):
            native_release_admit(fields[replacements[i]], context)
    var marker = py_get(fields, "_mojolearn_fitting", 0)
    host_inference_scalar(marker, context, True)
    var nested = Bool(py=marker)
    var est = String(py_call(native_type.__dict__["__name__"].__get__, py_type(model)))
    if host_sparse_format(args[0]).byte_length() > 0:
        raise Error("TypeError: Sparse data was passed for X, but dense data is required. Use '.toarray()' to convert to a dense numpy array. (" + est + " takes dense X only.)")
    var pair = _kde_prepare(args[0], context, "X")
    var raw = pair[0]
    var kind = classical_kind(raw, context)
    var rank = Int(py=raw.ndim)
    var described = kind == "b" or kind == "i" or kind == "u" or kind == "f" or kind == "c"
    if not nested and described:
        if kind == "c":
            raise Error("Complex data not supported: " + est + " got a complex X.")
        if rank < 2:
            raise Error("Expected 2D array, got " + ("scalar" if rank == 0 else "1D") + " array instead (shape=" + _kde_shape(raw.shape) + "). " + RESHAPE_HINT)
        if rank == 2 and (Int(py=raw.shape[0]) == 0 or Int(py=raw.shape[1]) == 0):
            var dimension = "sample(s)" if Int(py=raw.shape[0]) == 0 else "feature(s)"
            raise Error("Found array with 0 " + dimension + " (shape=" + _kde_shape(raw.shape) + ") while a minimum of 1 is required by " + est + ".")
    if not nested:
        _kde_y(args[1], Int(py=raw.shape[0]) if described and rank > 0 else -1, est, context)
    var copied = Bool(py=pair[1]) or not classical_is_f32(raw, context) or not classical_c_order(raw, context) or (not classical_is_array(raw, context) and not Bool(py=raw.dtype.isnative))
    var remaining_depth = _if_fit_marker_after(marker, context)
    try:
        _if_fit_body(model, fields, raw, copied, est, context)
    finally:
        if Bool(py=remaining_depth > 0):
            fields["_mojolearn_fitting"] = remaining_depth
        else:
            _ = py_call(fields.pop, "_mojolearn_fitting", PythonObject(None))
    py_setitem(fields, "_mojolearn_fitted", True)
    if not nested:
        _ = py_call(fields.pop, "feature_names_in_", PythonObject(None))
    _ = raw^
    _ = pair^
    return model
