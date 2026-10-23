# SPDX-License-Identifier: Apache-2.0
# RobustScaler owns its existing x_prep stage sequence and public lifecycle.
from std.os import getenv
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains
from bindings.scaler_tags_native import scaler_tags
from bindings.classical_attributes import classical_connect_numpy, classical_check_layout, classical_getattribute
from bindings.classical_arrays import classical_typed, classical_is_array, classical_input
from bindings.native_numpy_metadata import native_numpy_dtype
from bindings.classical_warnings import classical_warning, _warning_fields
from bindings.classical_metadata import classical_is_fitted
from bindings.classical_exceptions import classical_not_fitted_class
from bindings.causal_exception import causal_borrowed_item, causal_set_exception
from bindings.sequential_admission import _one_of, sequential_admit_value
from bindings.x_prep_native_inputs import _target
from bindings.scaler_native import _input, _ptr, _stage
from bindings.scaler_lifecycle_native import _entries, _public_result
from bindings.x_prep_native_owner import xprep_r3, xprep_storage, xprep_take, xprep_values, xprep_run
from gemm.f16_widen import f16_bits_to_f32

def _truth(value: PythonObject, context: PythonObject) raises -> Bool:
    var b = Python.import_module("builtins")
    var cls = py_type(value)
    if value is None or cls is b.bool or cls is b.int or cls is b.float or cls is b.complex or cls is b.str or cls is b.bytes or cls is b.list or cls is b.tuple or cls is b.dict or cls is b.range or _one_of(cls,context["scalar_types"]):
        return Bool(py=value)
    raise Error("TypeError: RobustScaler options require native values; Python truth hooks cannot execute")


def _complex_warning(context: PythonObject) raises:
    var b = Python.import_module("builtins")
    var module = py_get(context["modules"],"numpy.exceptions")
    if module is None:
        module = py_get(context["modules"],"numpy")
    if module is None or py_type(module) is not py_type(Python.import_module("sys")):
        raise Error("TypeError: complex quantiles require native NumPy warning metadata")
    var category = py_get(_warning_fields(py_fields(module)),"ComplexWarning")
    if py_type(category) is not b.type:
        raise Error("TypeError: complex quantiles require a native warning category")
    var ancestry = py_call(b.type.__dict__["__mro__"].__get__,category)
    for i in range(len(ancestry)):  # driver-loop: admit warning class metadata before its native slot guard
        var mapping = py_call(b.type.__dict__["__dict__"].__get__,ancestry[i])
        var keys = py_call(b.list,mapping)
        for k in range(len(keys)):  # driver-loop: class keys may otherwise invoke Python equality during lookup
            if py_type(keys[k]) is not b.str:
                raise Error("TypeError: complex quantile warning keys require native strings")
    if py_type(py_call(b.type.__getattribute__,category,"__name__")) is not b.str:
        raise Error("TypeError: complex quantile warning names require native strings")
    classical_warning("Casting complex values to real discards the imaginary part",category,context,PythonObject(None))


def _number(value: PythonObject, context: PythonObject) raises -> Float64:
    var b = Python.import_module("builtins")
    var cls = py_type(value)
    if cls is b.bool or cls is b.int or cls is b.float or cls is b.str or cls is b.bytes or _one_of(cls,context["scalar_types"]):
        return Float64(py=py_call(b.float,value))
    if py_get(context,"ndarray") is not None:
        var constructor = native_numpy_dtype(context)
        for code in ["S", "U", "g", "c8", "c16", "G"]:  # small-loop(params: supported native quantile scalar kinds)
            if cls is py_call(constructor,code).type:
                if code == "c8" or code == "c16" or code == "G":
                    _complex_warning(context)
                    return Float64(py=py_call(b.float,value.real))
                return Float64(py=py_call(b.float,value))
    raise Error("TypeError: RobustScaler quantiles require native numbers; Python conversion hooks cannot execute")


def _quantile_array(value: PythonObject, context: PythonObject) raises:
    # Exact ndarray alone is insufficient: user DTypes can own element
    # access. Match the descriptor's exact native class before dtype fields
    # or __getitem__. Object elements are admitted separately by _number.
    var dtype = value.dtype
    var constructor = native_numpy_dtype(context)
    var admitted = False
    for code in ["?","i1","u1","i2","u2","i4","u4","i8","u8","f2","f4","f8","g","c8","c16","G","S","U","O"]:  # small-loop(params: builtin native quantile descriptor kinds)
        if py_type(dtype) is py_type(py_call(constructor,code)):
            admitted = True
            break
    if not admitted:
        raise Error("TypeError: RobustScaler quantiles require native numeric, string or object storage; custom and structured dtypes cannot execute")
    if Int(py=value.ndim) != 1:
        raise Error("TypeError: RobustScaler quantile_range requires one-dimensional scalar metadata")


def _quantiles(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var cls = py_type(value)
    var source = value
    if classical_is_array(value,context):
        var shape = py_call(context["array_fields"]["shape"],value,context["array_class"])
        var dtype = py_call(context["array_fields"]["dtype"],value,context["array_class"])
        var order = py_call(context["array_fields"]["order"],value,context["array_class"])
        if py_type(dtype) is not b.str:
            raise Error("TypeError: RobustScaler quantile Array dtype must be native metadata")
        if py_type(shape) is not b.tuple or len(shape) != 1 or py_type(shape[0]) is not b.int or py_type(order) is not b.str or (String(order) != "C" and String(order) != "F"):
            raise Error("TypeError: RobustScaler quantile Array shape and order must be native metadata")
        var code = String(dtype)
        if code != "<f2" and code != "<f4" and code != "<f8" and code != "<i1" and code != "<i4" and code != "<i8" and code != "<u1" and code != "<u2" and code != "<u4":
            raise Error("TypeError: RobustScaler quantile Array requires supported numeric storage")
        source = py_call(context["array_fields"]["_mv"],value,context["array_class"])
        if py_type(source) is not b.memoryview:
            raise Error("TypeError: RobustScaler quantile Array requires a native memoryview")
        sequential_admit_value(source,context)
        if py_get(context,"ndarray") is not None:
            source = classical_input(value,context)
        elif code == "<f2":
            if String(source.format) != "H":
                raise Error("TypeError: RobustScaler half quantile Array requires native half-word storage")
            if len(source) != 2:
                raise Error("RobustScaler quantile_range must contain two values")
            return Python.tuple(Float64(f16_bits_to_f32(UInt16(Int(py=source[0])))),Float64(f16_bits_to_f32(UInt16(Int(py=source[1])))))
    if py_type(source) is b.memoryview:
        # Admission checks the exact exporter before any native buffer
        # acquisition/release or element access. Object/structured formats
        # are refused before they can produce callback-bearing scalars.
        sequential_admit_value(source,context)
        var format = String(source.format)
        if format.byte_length() == 2 and (format.startswith("@") or format.startswith("=") or format.startswith("<") or format.startswith(">") or format.startswith("!")):
            var scalar_format = String(format[byte=1:2])
            format = scalar_format
        if format != "?" and format != "b" and format != "B" and format != "h" and format != "H" and format != "i" and format != "I" and format != "l" and format != "L" and format != "q" and format != "Q" and format != "n" and format != "N" and format != "e" and format != "f" and format != "d":
            raise Error("TypeError: RobustScaler quantile buffers require native numeric formats")
        if Int(py=source.ndim) != 1:
            raise Error("TypeError: RobustScaler quantile_range requires one-dimensional scalar metadata")
        if py_get(context,"ndarray") is not None:
            # Native ndarray buffer access also preserves half and endian
            # formats that CPython memoryview scalar indexing cannot read.
            source = classical_input(source,context)
    if py_type(source) is py_get(context,"ndarray"):
        _quantile_array(source,context)
    elif cls is not b.list and cls is not b.tuple and cls is not b.range and cls is not b.str and cls is not b.bytes and cls is not b.memoryview and cls is not py_get(context,"ndarray"):
        if not classical_is_array(value,context):
            raise Error("TypeError: RobustScaler quantile_range requires native sequence metadata")
    if len(source) != 2:
        raise Error("RobustScaler quantile_range must contain two values")
    return Python.tuple(_number(source[0],context),_number(source[1],context))


def robust_fit(model: PythonObject, source: PythonObject, target: PythonObject, context: PythonObject,
               project: Bool = False) raises -> PythonObject:
    """The fit; with `project` (fit_transform) the transform's affine stage
    runs in the fit's own program on the X it already uploaded and the
    scaled X is returned (None when the output is large enough for the
    device output region, which the separate transform keeps)."""
    var fields = py_fields(model)
    # Preserve the public wrapper's input/target validation before fit begins.
    var mode = context["backend_fields"]["_DEFAULT_MODE"]
    var entries = _entries(mode,context)
    var values = xprep_values(source,entries,context)
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    _target(target,n,context)
    var quantiles = _quantiles(fields["quantile_range"],context)
    var lo = Float64(py=quantiles[0])
    var hi = Float64(py=quantiles[1])
    if not (0.0 <= lo and lo <= hi and hi <= 100.0):
        var b = Python.import_module("builtins")
        var given = fields["quantile_range"]
        # Only exact builtin sequence/scalar repr is allowed on this error
        # path; ndarray repr delegates to NumPy Python formatting.
        if py_type(given) is b.tuple or py_type(given) is b.list:
            var safe = True
            for k in range(2):  # small-loop(params: the two admitted quantile metadata values)
                var cls = py_type(given[k])
                safe = safe and (cls is b.float or cls is b.int or cls is b.bool or cls is b.str or cls is b.bytes)
            if safe:
                raise Error("mojolearn: invalid quantile range " + String(py_call(b.repr,given)))
        raise Error("mojolearn: invalid quantile range")
    var unit = _truth(fields["unit_variance"],context)
    if unit and not (0.0 < lo and lo < hi and hi < 100.0):
        raise Error("mojolearn: RobustScaler(unit_variance=True) needs 0 < q_min < q_max < 100 (norm.ppf of 0 or 1 is infinite)")
    var centering = _truth(fields["with_centering"],context)
    var scaling = _truth(fields["with_scaling"],context)
    var select = False
    if String(mode) == "fast" and Bool(py=context["robust_qselect"]) and not py_contains(entries,"x_prep_host_column"):
        var vendor = py_get(entries,"x_prep_vendor")
        select = vendor is not None and String(py_call(vendor)) == "metal"
    # Preserve the existing 2048-row statistics blocks and optional radix
    # quantile selection; no benchmark dimension selects a route.
    var blocked = String(getenv("MOJOLEARN_XPREP_BLOCKED","1").strip()) != "0"
    var blocks = (n+2047)//2048
    var work = xprep_r3("work")
    var sort_words = 0 if select else n*d
    var part_words = 5*blocks*d if blocked else 0
    var scratch = sort_words+part_words if work else 0
    var sorted = n*d
    var stats = n*d+(0 if work else sort_words)
    var fractions = stats+6*d
    var q = fractions+3
    var center = q+3*d
    var scale = center+d
    var host = scale+d+(0 if work else part_words)
    # fit_transform: the scaled X takes n*d words after the fit's host words;
    # the same affine stage and operands the transform program runs.
    var fused = project and n*d < 134217728
    var dest = host
    if fused:
        host += n*d
    var part = scale+d
    if work:
        sorted = host
        part = host+sort_words
    if host+scratch > 2147483647:
        raise Error("x_prep: the program exceeds the native Int32 indexing bound")
    var device_scratch = scratch > 0 and (py_get(entries,"x_prep_run_out") is not None or py_get(entries,"x_prep_run_scratch") is not None)
    var arena = xprep_storage(host+(0 if device_scratch else scratch),context)
    var fraction_values = classical_typed(Python.tuple(3),"<f4",context["output_context"])
    var fp = _ptr(fraction_values)
    fp[0] = Float32(lo/100.0)
    fp[1] = 0.5
    fp[2] = Float32(hi/100.0)
    var inputs = Python.tuple(Python.tuple(0,values),Python.tuple(fractions,Python.tuple(0,fraction_values,PythonObject(None),3,1)))
    var program = List[Int32]()
    if not select:
        _stage(program,0,d,0,n,d,sorted,0)
    if blocked:
        _stage(program,124,blocks*d,0,n,d,part,blocks)
        _stage(program,125,d,part,blocks,d,stats)
    else:
        _stage(program,1,d,0,n,d,stats)
    _stage(program,2,3*d,0 if select else sorted,n,d,fractions,3,q,stats,1 if select else 0)
    _stage(program,4,d,q,3,d,center,scale,0,0,2,1)
    if unit:
        _stage(program,73,d,scale,fractions)
    if fused:
        _stage(program,3,n*d,0,n*d,d,center if centering else -1,scale if scaling else -1,dest)
    xprep_run(arena,host,scratch,PythonObject(None),0,program^,inputs,entries,context)
    var scaled = xprep_take(arena[0],dest,n,d,context["output_context"]) if fused else PythonObject(None)
    py_setitem(fields,"center_",xprep_take(arena[0],center,d,0,context["output_context"]) if centering else PythonObject(None))
    py_setitem(fields,"scale_",xprep_take(arena[0],scale,d,0,context["output_context"]) if scaling else PythonObject(None))
    py_setitem(fields,"numeric_mode_",mode)
    py_setitem(fields,"n_features_in_",d)
    py_setitem(fields,"_mojolearn_fitted",True)
    if py_contains(fields,"feature_names_in_"):
        _ = py_call(fields.pop,"feature_names_in_")
    _ = arena^
    _ = values^
    return scaled if project else model


def robust_transform(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    if not Bool(py=classical_is_fitted(model)):
        _ = classical_not_fitted_class(context)
        raise Error("NotFittedError: This RobustScaler instance is not fitted yet. Call 'fit' with appropriate arguments before using transform.")
    var fields = py_fields(model)
    if not py_contains(fields,"n_features_in_"):
        raise Error("RuntimeError: mojolearn: this RobustScaler instance is not fitted yet")
    var entries = _entries(fields["numeric_mode_"],context)
    var values = xprep_values(source,entries,context)
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    if py_type(fields["n_features_in_"]) is not Python.import_module("builtins").int or Int(py=fields["n_features_in_"]) != d:
        raise Error("RobustScaler input feature count differs from fit")
    var center = fields["center_"]
    var scale = fields["scale_"]
    if center is not None:
        center = _input(center,1,False,context)
        if Int(py=center.size) != d:
            raise Error("RobustScaler center_ must hold one value per feature")
    if scale is not None:
        scale = _input(scale,1,False,context)
        if Int(py=scale.size) != d:
            raise Error("RobustScaler scale_ must hold one divisor per feature")
    var center_at = n*d if center is not None else -1
    var scale_at = n*d+(d if center is not None else 0) if scale is not None else -1
    var dest = n*d+(d if center is not None else 0)+(d if scale is not None else 0)
    if dest+n*d > 2147483647:
        raise Error("x_prep: the program exceeds the native Int32 indexing bound")
    var dev_out = n*d >= 134217728 and py_get(entries,"x_prep_run_out") is not None and String(getenv("MOJOLEARN_XPREP_OUT","1")) != "0"
    var host = dest+(0 if dev_out else n*d)
    var arena = xprep_storage(host,context)
    var output_store = xprep_storage(n*d,context) if dev_out else PythonObject(None)
    var inputs = Python.list()
    _ = py_call(inputs.append,Python.tuple(0,values))
    if center is not None:
        _ = py_call(inputs.append,Python.tuple(center_at,Python.tuple(0,center,PythonObject(None),d,1)))
    if scale is not None:
        _ = py_call(inputs.append,Python.tuple(scale_at,Python.tuple(0,scale,PythonObject(None),d,1)))
    var program = List[Int32]()
    _stage(program,3,n*d,0,n*d,d,center_at,scale_at,dest)
    xprep_run(arena,host,0,output_store,n*d if dev_out else 0,program^,inputs,entries,context,n*d if n*d >= 134217728 else 0)
    var output = xprep_take(output_store[0] if dev_out else arena[0],0 if dev_out else dest,n,d,context["output_context"])
    _ = arena^
    _ = values^
    return output


def robust_lifecycle_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        var model = call[0]
        var operation = String(call[1])
        var args = call[2]
        var context = call[3]
        classical_connect_numpy(context)
        classical_check_layout(model,context)
        var fields = py_fields(model)
        var names = Python.tuple("with_centering","with_scaling","quantile_range","copy","unit_variance")
        var result = PythonObject(None)
        if operation == "init":
            for k in range(5):  # small-loop(params: the five stored constructor parameters)
                py_setitem(fields,names[k],args[k])
        elif operation == "get_params":
            result = Python.dict()
            for k in range(5):  # small-loop(params: the five constructor parameters)
                py_setitem(result,names[k],fields[names[k]])
        elif operation == "set_params":
            var options = args[0]
            var keys = py_call(Python.import_module("builtins").list,options)
            for k in range(len(keys)):  # driver-loop: proposed parameter names in caller order
                if py_type(keys[k]) is not Python.import_module("builtins").str or not py_contains(names,keys[k]):
                    raise Error("mojolearn: invalid parameter for RobustScaler")
                py_setitem(fields,keys[k],options[keys[k]])
            result = model
        elif operation == "tags":
            result = scaler_tags(model,context,PythonObject(None))
        elif operation == "is_fitted":
            result = classical_is_fitted(model)
        elif operation == "fit" or operation == "fit_transform":
            if operation == "fit_transform" and len(args[2]) != 0:
                raise Error("TypeError: RobustScaler.fit got an unexpected keyword argument")
            if operation == "fit_transform":
                # An inherited combined method must never silently bypass
                # application overrides or invoke them as Python callbacks.
                var fit = classical_getattribute(model,PythonObject("fit"),context)
                var transform = classical_getattribute(model,PythonObject("transform"),context)
                if py_type(fit) is not context["robust_method_type"] or py_type(transform) is not context["robust_method_type"]:
                    raise Error("TypeError: native fit_transform requires native fit and transform; call application operations explicitly")
                if fit.__func__ is not context["robust_fit_method"] or transform.__func__ is not context["robust_transform_method"]:
                    raise Error("TypeError: native fit_transform cannot invoke Python estimator overrides; call application operations explicitly")
            result = robust_fit(model,args[0],args[1],context,operation == "fit_transform")
            if operation == "fit_transform":
                if result is None:
                    result = robust_transform(model,args[0],context)
                result = _public_result(result,context)
        elif operation == "transform":
            result = _public_result(robust_transform(model,args[0],context),context)
        else:
            raise Error("unknown native RobustScaler operation")
        return result.steal_data()
    except error:
        # Input/target admission is the same native preparation shared with
        # MaxAbsScaler. Keep this estimator's public diagnostic identity.
        var own = Error(String(error).replace("MaxAbsScaler","RobustScaler"))
        if String(own).startswith("NotFittedError:") or String(own).startswith("DataConversionWarning:"):
            try:
                ref cpy = Python().cpython()
                var context = cpy.PyTuple_GetItem(args_ptr,3)
                var warning = String(own).startswith("DataConversionWarning:")
                var cls = causal_borrowed_item(context,"warning_category" if warning else "active_not_fitted_class")
                return causal_set_exception(error_without_class(own,"DataConversionWarning" if warning else "NotFittedError"),cls)
            except nested:
                return raise_python_exception(nested)
        if String(own).startswith("TypeError:"):
            return raise_python_exception(error_without_class(own,"TypeError"),ExceptionType.TypeError)
        if String(own).startswith("AttributeError:"):
            return raise_python_exception(error_without_class(own,"AttributeError"),ExceptionType("PyExc_AttributeError"))
        if String(own).startswith("ImportError:"):
            return raise_python_exception(error_without_class(own,"ImportError"),ExceptionType("PyExc_ImportError"))
        if String(own).startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(own,"RuntimeError"),ExceptionType("PyExc_RuntimeError"))
        return raise_python_exception(error_without_class(own,"ValueError"),ExceptionType.ValueError)
