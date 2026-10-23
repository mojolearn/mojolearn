# SPDX-License-Identifier: Apache-2.0
# MaxAbsScaler owns its existing x_prep program and complete public operation.
from std.os import getenv
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains
from bindings.scaler_tags_native import scaler_tags
from bindings.classical_attributes import classical_connect_numpy, classical_check_layout
from bindings.classical_arrays import classical_address, classical_typed
from bindings.classical_metadata import classical_is_fitted
from bindings.classical_exceptions import classical_not_fitted_class
from bindings.causal_exception import causal_borrowed_item, causal_set_exception
from bindings.scaler_native import _input, _ptr, _stage, _copy_row
from bindings.scaler_lifecycle_native import _entries, _public_result
from bindings.x_prep_native_inputs import _target, _values
from bindings.x_prep_native_owner import xprep_r3, xprep_storage, xprep_take, xprep_values, xprep_run
from core.scaler_fit_impl import scaler_refuse_infinite


def maxabs_fit(model: PythonObject, source: PythonObject, target: PythonObject, context: PythonObject,
               project: Bool = False) raises -> PythonObject:
    """The fit; with `project` (fit_transform) the transform's affine stage
    runs in the fit's own program on the X it already uploaded, and the
    scaled X is returned (None when this route cannot carry it: the direct
    Apple fast entry, or an output large enough for the device output
    region, which the separate transform keeps)."""
    var fields = py_fields(model)
    var mode = context["backend_fields"]["_DEFAULT_MODE"]
    var entries = _entries(mode,context)
    var direct = py_get(entries,"x_prep_maxabs_fit_direct")
    # The existing Apple fast direct entry owns its upload; retain that
    # route, including its host row materialization. Other routes preserve
    # the established device rows, resident cache and scratch transports.
    var values = PythonObject(None)
    if direct is not None:
        var dense = _values(source,context,True)
        values = Python.tuple(0,dense,PythonObject(None),dense.shape[0],dense.shape[1])
    else:
        values = xprep_values(source,entries,context)
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    _target(target,n,context)
    var maximum = PythonObject(None)
    var scale = PythonObject(None)
    var fused = False
    var scaled = PythonObject(None)
    if direct is not None:
        var output = classical_typed(Python.tuple(2*d),"<f4",context["output_context"])
        _ = py_call(direct,classical_address(values[1]),classical_address(output),Python.list(n,d))
        _ = output  # keep the Python object alive through the native call above
        maximum = _copy_row(_ptr(output),d,context)
        _ = output  # keep the Python object alive through the native call above
        scale = _copy_row(_ptr(output)+d,d,context)
        _ = output  # keep the Python object alive through the native call above
        _ = output^
    else:
        # Existing row blocks are 2048 rows, independent of benchmark shapes.
        var blocked = String(getenv("MOJOLEARN_XPREP_BLOCKED","1").strip()) != "0"
        var blocks = (n+2047)//2048
        var work = blocked and xprep_r3("work")
        var scratch = 5*blocks*d if work else 0
        var host = n*d+(2*d+(0 if work else 5*blocks*d) if blocked else 7*d)
        # fit_transform: the scaled X takes n*d words after the fit's host
        # words. The same affine stage, operands and order the transform
        # program runs, so the words are the same.
        fused = project and n*d < 134217728
        var dest = host
        if fused:
            host += n*d
        if host+scratch > 2147483647:
            raise Error("x_prep: the program exceeds the native Int32 indexing bound")
        var device_scratch = scratch > 0 and (py_get(entries,"x_prep_run_out") is not None or py_get(entries,"x_prep_run_scratch") is not None)
        var arena = xprep_storage(host+(0 if device_scratch else scratch),context)
        var sc = n*d
        var ma = 0
        var program = List[Int32]()
        if blocked:
            var part = host if work else sc+d
            ma = sc+d if work else part+5*blocks*d
            _stage(program,124,blocks*d,0,n,d,part,blocks)
            _stage(program,128,d,part,blocks,d,ma,sc)
        else:
            var st = sc+d
            ma = st+5*d
            _stage(program,1,d,0,n,d,st)
            _stage(program,4,d,ma,1,d,-1,sc,1)
        if fused:
            _stage(program,3,n*d,0,n*d,d,-1,sc,dest)
        xprep_run(arena,host,scratch,PythonObject(None),0,program^,Python.tuple(Python.tuple(0,values)),entries,context)
        maximum = xprep_take(arena[0],ma,d,0,context["output_context"])
        scale = xprep_take(arena[0],sc,d,0,context["output_context"])
        if fused:
            scaled = xprep_take(arena[0],dest,n,d,context["output_context"])
        _ = arena^
    # scikit-learn refuses an infinite input (NaN alone is allowed); the
    # native column maxima of |X| are infinite exactly for those columns.
    scaler_refuse_infinite(_ptr(maximum),d,"ValueError: Input X contains infinity or a value too large for dtype('float64').")
    _ = maximum  # keep the Python object alive through the native call above
    py_setitem(fields,"max_abs_",maximum)
    py_setitem(fields,"scale_",scale)
    py_setitem(fields,"n_features_in_",d)
    py_setitem(fields,"n_samples_seen_",n)
    py_setitem(fields,"numeric_mode_",mode)
    py_setitem(fields,"_mojolearn_fitted",True)
    if py_contains(fields,"feature_names_in_"):
        _ = py_call(fields.pop,"feature_names_in_")
    _ = values^
    return scaled if project else model


def maxabs_transform(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    if not Bool(py=classical_is_fitted(model)):
        _ = classical_not_fitted_class(context)
        raise Error("NotFittedError: This MaxAbsScaler instance is not fitted yet. Call 'fit' with appropriate arguments before using transform.")
    var fields = py_fields(model)
    if not py_contains(fields,"n_features_in_"):
        raise Error("RuntimeError: mojolearn: this MaxAbsScaler instance is not fitted yet")
    var entries = _entries(fields["numeric_mode_"],context)
    var values = xprep_values(source,entries,context)
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    if py_type(fields["n_features_in_"]) is not Python.import_module("builtins").int or Int(py=fields["n_features_in_"]) != d:
        raise Error("MaxAbsScaler input feature count differs from fit")
    var scale = fields["scale_"]
    if scale is not None:
        scale = _input(scale,1,False,context)
        if Int(py=scale.size) != d:
            raise Error("MaxAbsScaler scale_ must hold one divisor per feature")
    var dest = n*d+(d if scale is not None else 0)
    if dest+n*d > 2147483647:
        raise Error("x_prep: the program exceeds the native Int32 indexing bound")
    var dev_out = n*d >= 134217728 and py_get(entries,"x_prep_run_out") is not None and String(getenv("MOJOLEARN_XPREP_OUT","1")) != "0"
    var host = dest+(0 if dev_out else n*d)
    var arena = xprep_storage(host,context)
    var output_store = xprep_storage(n*d,context) if dev_out else PythonObject(None)
    var inputs = Python.list(Python.tuple(0,values))
    if scale is not None:
        _ = py_call(inputs.append,Python.tuple(n*d,Python.tuple(0,scale,PythonObject(None),d,1)))
    var program = List[Int32]()
    _stage(program,3,n*d,0,n*d,d,-1,n*d if scale is not None else -1,dest)
    xprep_run(arena,host,0,output_store,n*d if dev_out else 0,program^,inputs,entries,context,n*d if n*d >= 134217728 else 0)
    var output = xprep_take(output_store[0] if dev_out else arena[0],0 if dev_out else dest,n,d,context["output_context"])
    _ = arena^
    _ = values^
    return output


def maxabs_lifecycle_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        var model = call[0]
        var operation = String(call[1])
        var args = call[2]
        var context = call[3]
        classical_connect_numpy(context)
        classical_check_layout(model,context)
        var fields = py_fields(model)
        var result = PythonObject(None)
        if operation == "init":
            py_setitem(fields,"copy",args[0])
        elif operation == "get_params":
            result = Python.dict(copy=fields["copy"])
        elif operation == "set_params":
            var options = args[0]
            var names = py_call(Python.import_module("builtins").list,options)
            for k in range(len(names)):  # driver-loop: proposed parameter names, in declared order
                if py_type(names[k]) is not Python.import_module("builtins").str or String(names[k]) != "copy":
                    raise Error("mojolearn: invalid parameter for MaxAbsScaler")
                py_setitem(fields,names[k],options[names[k]])
            result = model
        elif operation == "tags":
            result = scaler_tags(model,context,PythonObject(None))
        elif operation == "is_fitted":
            result = classical_is_fitted(model)
        elif operation == "fit" or operation == "fit_transform":
            if operation == "fit_transform" and len(args[2]) != 0:
                raise Error("TypeError: MaxAbsScaler.fit got an unexpected keyword argument")
            result = maxabs_fit(model,args[0],args[1],context,operation == "fit_transform")
            if operation == "fit_transform":
                if result is None:
                    result = maxabs_transform(model,args[0],context)
                result = _public_result(result,context)
        elif operation == "transform":
            result = _public_result(maxabs_transform(model,args[0],context),context)
        else:
            raise Error("unknown native MaxAbsScaler operation")
        return result.steal_data()
    except error:
        if String(error).startswith("NotFittedError:") or String(error).startswith("DataConversionWarning:"):
            try:
                ref cpy = Python().cpython()
                var context = cpy.PyTuple_GetItem(args_ptr,3)
                var warning = String(error).startswith("DataConversionWarning:")
                var cls = causal_borrowed_item(context,"warning_category" if warning else "active_not_fitted_class")
                return causal_set_exception(error_without_class(error,"DataConversionWarning" if warning else "NotFittedError"),cls)
            except nested:
                return raise_python_exception(nested)
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error_without_class(error,"TypeError"),ExceptionType.TypeError)
        if String(error).startswith("AttributeError:"):
            return raise_python_exception(error_without_class(error,"AttributeError"),ExceptionType("PyExc_AttributeError"))
        if String(error).startswith("ImportError:"):
            return raise_python_exception(error_without_class(error,"ImportError"),ExceptionType("PyExc_ImportError"))
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error,"RuntimeError"),ExceptionType("PyExc_RuntimeError"))
        return raise_python_exception(error_without_class(error,"ValueError"),ExceptionType.ValueError)
