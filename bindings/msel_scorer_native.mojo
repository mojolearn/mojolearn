# SPDX-License-Identifier: Apache-2.0
# Fixed scorer records call verified compiled estimator/metric entries. The
# Python function objects identify import-time plans; they are never invoked.
from bindings.native_loaded_module import native_loaded_module
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains
from bindings.family_common import family_call, family_array
from bindings.classical_arrays import classical_address
from bindings.classical_attributes import classical_connect_numpy
from bindings.metrics_native_support import metric_is_real, metric_float
from bindings.msel_inputs import msel_input, msel_admit
from bindings.native_buffer_owner import native_storage_allocate
from bindings.msel_estimator_native import msel_estimator_call, msel_estimator_plan


def msel_metric_plan(function: PythonObject, context: PythonObject) raises -> PythonObject:
    if py_type(function) is not context["family_msel_metric_function_type"]:
        raise Error("TypeError: Python metric callbacks are not accepted; use a connected native mojolearn metric")
    var plan = py_get(context["family_msel_metric_plans"], function)
    if plan is None:
        raise Error("TypeError: Python metric callbacks are not accepted; use a connected native mojolearn metric")
    if function.__code__ is not plan["code"]:
        raise Error("TypeError: the connected metric function was modified")
    return plan


def _scorer_class(context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var cls = context["family_msel_scorer_class"]
    if py_type(cls) is not b.type:
        raise Error("TypeError: native scorer metadata refuses metaclass hooks")
    var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
    if len(mro) != 2 or mro[1] is not b.object:
        raise Error("TypeError: native scorer metadata refuses inherited lifecycle hooks")
    var source = py_call(b.type.__dict__["__dict__"].__get__, cls)
    var hooks = context["family_msel_scorer_hooks"]
    for k in range(len(hooks)):  # driver-loop: connected scorer storage and method identities before allocation or access
        if py_get(source, hooks[k][0]) is not hooks[k][1]:
            raise Error("TypeError: native scorer methods or storage hooks changed")
    return cls


def msel_scorer_create(function: PythonObject, response: PythonObject, sign: Int,
                       keywords: PythonObject, context: PythonObject) raises -> PythonObject:
    _ = msel_metric_plan(function, context)
    msel_admit(response, context)
    msel_admit(keywords, context)
    var cls = _scorer_class(context)
    var scorer = native_storage_allocate(cls)
    var fields = py_call(context["family_msel_scorer_fields"], scorer, cls)
    # Fresh admitted native storage; Python __setattr__ and property setters
    # are never involved and no pre-existing owner can be released here.
    py_setitem(fields, "_score_func", function)
    py_setitem(fields, "_sign", PythonObject(sign))
    py_setitem(fields, "_kwargs", keywords)
    py_setitem(fields, "_response_method", response)
    py_setitem(fields, "_name", PythonObject(None))
    return scorer


def msel_scorer_fields(scorer: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var cls = _scorer_class(context)
    if py_type(scorer) is not cls:
        raise Error("TypeError: Python scorer callbacks are not accepted; use get_scorer or make_scorer with a native metric")
    var fields = py_call(context["family_msel_scorer_fields"], scorer, cls)
    var keys = py_call(b.list, fields.keys())
    for k in range(len(keys)):  # driver-loop: exact scorer keys before metadata lookup
        if py_type(keys[k]) is not b.str:
            raise Error("TypeError: native scorer fields require exact string keys")
    _ = msel_metric_plan(fields["_score_func"], context)
    return fields


def msel_metric_call(function: PythonObject, y: PythonObject, predicted: PythonObject,
                     keywords: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var plan = msel_metric_plan(function, context)
    var entry = plan["entry"]
    if entry is None or py_type(entry) is not py_type(b.len):
        raise Error("ImportError: native scorer metric binding is unavailable")
    if py_type(keywords) is not b.dict:
        raise Error("TypeError: native metric keywords must be an exact dict")
    msel_admit(keywords, context)
    var values = py_call(b.dict, plan["defaults"])
    py_setitem(values, plan["inputs"][0], y)
    py_setitem(values, plan["inputs"][1], predicted)
    var keys = py_call(b.list, keywords.keys())
    for k in range(len(keys)):  # driver-loop: admitted metric keyword metadata
        var name = keys[k]
        if py_type(name) is not b.str or not py_contains(plan["allowed"], name):
            raise Error("TypeError: native metric received an unexpected keyword argument")
        if name is plan["inputs"][0] or name is plan["inputs"][1] or Bool(py=name == plan["inputs"][0]) or Bool(py=name == plan["inputs"][1]):
            raise Error("TypeError: native metric received multiple values for a positional input")
        py_setitem(values, name, keywords[name])
    var envelope = Python.list()
    var recipe = plan["recipe"]
    for k in range(len(recipe)):  # driver-loop: fixed native metric ABI arguments
        var item = recipe[k]
        var tag = String(item[0])
        var value: PythonObject
        if tag == "parameter":
            value = values[item[1]]
        elif tag == "binding":
            var mode = PythonObject(None) if item[1] is None else values[item[1]]
            if mode is None:
                mode = context["family_msel_backend_fields"]["_DEFAULT_MODE"]
            if py_type(mode) is not b.str:
                raise Error("ValueError: numeric_mode must be 'fast' or 'identical'")
            var clean = py_call(py_call(mode.strip).lower)
            if String(clean) != "identical" and String(clean) != "fast":
                raise Error("ValueError: numeric_mode must be 'fast' or 'identical'")
            value = context["family_msel_metric_bindings"][clean]
            if py_type(value) is b.str:
                raise Error(String(value))
        else:
            value = item[1]
        _ = py_call(envelope.append, value)
    return family_call(entry, PythonObject(None), plan["operation"], py_call(b.tuple, envelope), plan["context"])


def _scorer_column(predicted: PythonObject, context: PythonObject) raises -> PythonObject:
    var array = predicted
    var dtype = String(array.dtype)
    if dtype != "<f4" and dtype != "<f8":
        raise Error("TypeError: native probability output must be float32 or float64")
    var rows = Int(py=array.shape[0])
    var result = family_array(Python.tuple(rows), "<f4", context)
    var entry = context["family_msel_entries"]["msel_proba_column"]
    if py_type(entry) is not py_type(native_loaded_module("builtins").len):
        raise Error("ImportError: native probability column entry is unavailable")
    _ = family_call(entry, classical_address(array), context["family_msel_codes"][array.dtype],
                    PythonObject(rows), PythonObject(2), PythonObject(1), classical_address(result))
    _ = array  # keep the Python object alive through the native call above
    _ = result  # keep the Python object alive through the native call above
    return result


def msel_scorer_call(scorer: PythonObject, estimator: PythonObject, x: PythonObject, y: PythonObject,
                     sample_weight: PythonObject, memo: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var fields = msel_scorer_fields(scorer, context)
    var model = msel_estimator_plan(estimator, context)
    var methods = fields["_response_method"]
    if py_type(methods) is b.str:
        methods = Python.tuple(methods)
    elif py_type(methods) is not b.tuple:
        raise Error("TypeError: native scorer response_method requires a string or tuple of strings")
    var method = PythonObject(None)
    for k in range(len(methods)):  # driver-loop: fixed response preference order
        if py_type(methods[k]) is not b.str:
            raise Error("TypeError: native scorer response method names must be exact strings")
        if py_contains(model["methods"], methods[k]):
            method = methods[k]
            break
    if method is None:
        raise Error("AttributeError: native estimator has none of the requested scorer response methods")
    if memo is not None:
        if py_type(memo) is not b.dict:
            raise Error("TypeError: native scorer prediction cache must be an exact dict")
        msel_admit(memo, context)
    var predicted = PythonObject(None) if memo is None else py_get(memo, method)
    if predicted is None:
        predicted = msel_estimator_call(estimator, String(method), Python.tuple(x), context, True)
        if memo is not None:
            py_setitem(memo, method, predicted)
    var name = fields["_name"]
    if name is not None and py_type(name) is not b.str:
        raise Error("TypeError: native scorer name must be an exact string")
    if String(method) == "predict_proba" and py_contains(context["family_msel_binary_proba"], name):
        var array = msel_input(predicted, "pred", context)
        if Int(py=array.ndim) == 2 and Int(py=array.shape[1]) == 2:
            var column = PythonObject(None) if memo is None else py_get(memo, "predict_proba[:, 1]")
            if column is None:
                column = _scorer_column(array, context)
                if memo is not None:
                    py_setitem(memo, "predict_proba[:, 1]", column)
            predicted = column
    var keywords = fields["_kwargs"]
    if py_type(keywords) is not b.dict:
        raise Error("TypeError: native scorer keyword options must be an exact dict")
    msel_admit(keywords, context)
    keywords = py_call(b.dict, keywords)
    if sample_weight is not None:
        py_setitem(keywords, "sample_weight", sample_weight)
    var result = msel_metric_call(fields["_score_func"], y, predicted, keywords, context)
    classical_connect_numpy(context)
    if not metric_is_real(result, context):
        raise Error("TypeError: scoring must return a real scalar")
    var sign = fields["_sign"]
    if py_type(sign) is not b.int or (Int(py=sign) != 1 and Int(py=sign) != -1):
        raise Error("TypeError: native scorer sign must be 1 or -1")
    return PythonObject(Float64(Int(py=sign)) * metric_float(result, context))
