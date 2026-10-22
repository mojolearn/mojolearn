# SPDX-License-Identifier: Apache-2.0
# RobustScaler owns its existing x_prep stage sequence and public lifecycle.
from std.memory import memcpy
from std.os import getenv
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains
from bindings.classical_attributes import classical_connect_numpy, classical_check_layout, classical_getattribute
from bindings.classical_arrays import classical_typed, classical_is_array
from bindings.classical_metadata import classical_is_fitted
from bindings.classical_exceptions import classical_not_fitted_class
from bindings.causal_exception import causal_borrowed_item, causal_set_exception
from bindings.sequential_admission import _one_of
from bindings.maxabs_native import _values, _target
from bindings.scaler_native import _input, _ptr, _stage, _copy_row
from bindings.scaler_lifecycle_native import _run, _entries, _public_result

alias F32P = MutPointer[Float32, MutUntrackedOrigin]


def _truth(value: PythonObject, context: PythonObject) raises -> Bool:
    var b = Python.import_module("builtins")
    var cls = py_type(value)
    if value is None or cls is b.bool or cls is b.int or cls is b.float or cls is b.complex or cls is b.str or cls is b.bytes or cls is b.list or cls is b.tuple or cls is b.dict or cls is b.range or _one_of(cls,context["scalar_types"]):
        return Bool(py=value)
    raise Error("TypeError: RobustScaler options require native values; Python truth hooks cannot execute")


def _number(value: PythonObject, context: PythonObject) raises -> Float64:
    var b = Python.import_module("builtins")
    var cls = py_type(value)
    if cls is b.bool or cls is b.int or cls is b.float or cls is b.str or cls is b.bytes or _one_of(cls,context["scalar_types"]):
        return Float64(py=py_call(b.float,value))
    raise Error("TypeError: RobustScaler quantiles require native numbers; Python conversion hooks cannot execute")


def _quantiles(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var cls = py_type(value)
    var source = value
    if classical_is_array(value,context):
        source = value._mv
    elif cls is not b.list and cls is not b.tuple and cls is not b.range and cls is not b.str and cls is not b.bytes and cls is not b.memoryview and cls is not py_get(context,"ndarray"):
        raise Error("TypeError: RobustScaler quantile_range requires native sequence metadata")
    if len(source) != 2:
        raise Error("RobustScaler quantile_range must contain two values")
    return Python.tuple(_number(source[0],context),_number(source[1],context))


def robust_fit(model: PythonObject, source: PythonObject, target: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    # Preserve the public wrapper's input/target validation before fit begins.
    var values = _values(source,context)
    var n = Int(py=values.shape[0])
    var d = Int(py=values.shape[1])
    _target(target,n,context)
    var quantiles = _quantiles(fields["quantile_range"],context)
    var lo = Float64(py=quantiles[0])
    var hi = Float64(py=quantiles[1])
    if not (0 <= lo and lo <= hi and hi <= 100):
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
    if unit and not (0 < lo and lo < hi and hi < 100):
        raise Error("mojolearn: RobustScaler(unit_variance=True) needs 0 < q_min < q_max < 100 (norm.ppf of 0 or 1 is infinite)")
    var centering = _truth(fields["with_centering"],context)
    var scaling = _truth(fields["with_scaling"],context)
    var mode = context["backend_fields"]["_DEFAULT_MODE"]
    var entries = _entries(mode,context)
    var select = False
    if String(mode) == "fast" and Bool(py=context["robust_qselect"]) and not py_contains(entries,"x_prep_host_column"):
        var vendor = py_get(entries,"x_prep_vendor")
        select = vendor is not None and String(py_call(vendor)) == "metal"
    # Preserve the existing 2048-row statistics blocks and optional radix
    # quantile selection; no benchmark dimension selects a route.
    var blocked = String(getenv("MOJOLEARN_XPREP_BLOCKED","1").strip()) != "0"
    var blocks = (n+2047)//2048
    var sorted = n*d
    var stats = sorted+(0 if select else n*d)
    var fractions = stats+6*d
    var q = fractions+3
    var center = q+3*d
    var scale = center+d
    var part = scale+d
    var size = part+(5*blocks*d if blocked else 0)
    if size > 2147483647:
        raise Error("x_prep: the program exceeds the native Int32 indexing bound")
    var arena = List[Float32](length=size,fill=0)
    var ap = F32P(unsafe_from_address=Int(arena.unsafe_ptr()))
    memcpy(dest=ap,src=_ptr(values),count=n*d)
    ap[fractions] = Float32(lo/100.0)
    ap[fractions+1] = 0.5
    ap[fractions+2] = Float32(hi/100.0)
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
    _run(arena,program,entries)
    py_setitem(fields,"center_",_copy_row(ap+center,d,context) if centering else PythonObject(None))
    py_setitem(fields,"scale_",_copy_row(ap+scale,d,context) if scaling else PythonObject(None))
    py_setitem(fields,"numeric_mode_",mode)
    py_setitem(fields,"n_features_in_",d)
    py_setitem(fields,"_mojolearn_fitted",True)
    if py_contains(fields,"feature_names_in_"):
        _ = py_call(fields.pop,"feature_names_in_")
    _ = arena^
    _ = values^
    return model


def robust_transform(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    if not Bool(py=classical_is_fitted(model)):
        _ = classical_not_fitted_class(context)
        raise Error("NotFittedError: This RobustScaler instance is not fitted yet. Call 'fit' with appropriate arguments before using transform.")
    var fields = py_fields(model)
    if not py_contains(fields,"n_features_in_"):
        raise Error("RuntimeError: mojolearn: this RobustScaler instance is not fitted yet")
    var values = _values(source,context)
    var n = Int(py=values.shape[0])
    var d = Int(py=values.shape[1])
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
    var dest = n*d+2*d
    if dest+n*d > 2147483647:
        raise Error("x_prep: the program exceeds the native Int32 indexing bound")
    var arena = List[Float32](length=dest+n*d,fill=0)
    var ap = F32P(unsafe_from_address=Int(arena.unsafe_ptr()))
    memcpy(dest=ap,src=_ptr(values),count=n*d)
    if center is not None:
        memcpy(dest=ap+n*d,src=_ptr(center),count=d)
    if scale is not None:
        memcpy(dest=ap+n*d+d,src=_ptr(scale),count=d)
    var program = List[Int32]()
    _stage(program,3,n*d,0,n*d,d,n*d if center is not None else -1,n*d+d if scale is not None else -1,dest)
    _run(arena,program,_entries(fields["numeric_mode_"],context))
    var output = classical_typed(Python.tuple(n,d),"<f4",context["output_context"])
    memcpy(dest=_ptr(output),src=ap+dest,count=n*d)
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
            result = robust_fit(model,args[0],args[1],context)
            if operation == "fit_transform":
                result = _public_result(robust_transform(model,args[0],context),context)
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
        if String(own).startswith("ImportError:"):
            return raise_python_exception(error_without_class(own,"ImportError"),ExceptionType("PyExc_ImportError"))
        if String(own).startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(own,"RuntimeError"),ExceptionType("PyExc_RuntimeError"))
        return raise_python_exception(error_without_class(own,"ValueError"),ExceptionType.ValueError)
