# SPDX-License-Identifier: Apache-2.0
# Native `get_feature_names_out` for the neighbors lane's transformers
# (python/mojolearn/_expansion_neighbors.py): KernelPCA, AdditiveChi2Sampler,
# SkewedChi2Sampler and KNNImputer. Each public method is one native call;
# the names, the fitted-state reads and the input-name admission run here.
# Input names are admitted without invoking element or container protocols:
# None, an exact list or tuple, or a NumPy ndarray (its native `tolist`
# slot), holding str values (read from their unicode storage) or exact ints
# (formatted natively, as `str(int)` formats them).
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields
from bindings.classical_attributes import classical_connect_numpy, _type_mro
from bindings.classical_validate import classical_view
from bindings.host_inference_scalars import host_inference_scalar, host_inference_primitive_number
from providers.python.array_metadata_pyapi import array_shape_once


def _class_name(model: PythonObject) raises -> String:
    var b = native_loaded_module("builtins")
    return String(py_call(b.type.__dict__["__name__"].__get__, py_type(model)))


def _missing(model: PythonObject, name: String) raises -> String:
    return "AttributeError: '" + _class_name(model) + "' object has no attribute '" + name + "'"


def _fitted(model: PythonObject, fields: PythonObject, name: String) raises -> PythonObject:
    var value = py_get(fields, PythonObject(name), PythonObject(None))
    if value is None and not py_contains(fields, PythonObject(name)):
        var message = _missing(model, name)
        raise Error(message)
    return value


def _int_value(value: PythonObject, context: PythonObject) raises -> Int:
    # `int(value)` over an admitted primitive scalar only.
    var b = native_loaded_module("builtins")
    host_inference_scalar(value, context)
    return Int(py=py_call(b.int, host_inference_primitive_number(value, context, True)))


def _truth_value(value: PythonObject, context: PythonObject) raises -> Bool:
    # `bool(value)` over an admitted primitive scalar only.
    host_inference_scalar(value, context, True)
    return Bool(py=value)


def _is_str(value: PythonObject) raises -> Bool:
    var b = native_loaded_module("builtins")
    var mro = _type_mro(py_type(value))
    for i in range(len(mro)):  # driver-loop: over native class ancestry metadata
        if mro[i] is b.str:
            return True
    return False


def _names_in(model: PythonObject, fields: PythonObject, source: PythonObject,
              context: PythonObject) raises -> List[String]:
    """`_feature_names_in(est, input_features)`."""
    var b = native_loaded_module("builtins")
    var d = _int_value(_fitted(model, fields, "n_features_in_"), context)
    var names = List[String]()
    if source is None:
        for i in range(d):  # driver-loop: finished public feature-name metadata over the d columns
            names.append("x" + String(i))
        return names^
    var cls = py_type(source)
    var items = source
    if cls is py_get(context, "ndarray") and py_get(context, "ndarray") is not None:
        items = py_call(cls.__dict__["tolist"], source)
    elif cls is not b.list and cls is not b.tuple:
        raise Error("TypeError: input_features must be None, a list, a tuple or a NumPy array of str")
    if py_type(items) is not b.list and py_type(items) is not b.tuple:
        raise Error("TypeError: input_features must be a one-dimensional sequence of str")
    for i in range(len(items)):  # driver-loop: finished public feature-name metadata over the given names
        var item = items[i]
        if py_type(item) is b.int:
            names.append(String(Int(py=item)))
        elif _is_str(item):
            names.append(String(py=item))
        else:
            raise Error("TypeError: input_features entries must be str")
    if len(names) != d:
        raise Error("ValueError: input_features should have length equal to number of features ("
                    + String(d) + "), got " + String(len(names)))
    return names^


def _to_list(names: List[String]) raises -> PythonObject:
    var result = Python.list()
    for i in range(len(names)):  # driver-loop: finished public feature-name metadata
        _ = py_call(result.append, PythonObject(names[i]))
    return result


def _prefixed(model: PythonObject, count: Int) raises -> PythonObject:
    """sklearn's ClassNamePrefixFeaturesOutMixin: `<classname lower><i>`."""
    var base = _class_name(model).lower()
    var names = List[String]()
    for i in range(max(count, 0)):  # driver-loop: finished public feature-name metadata
        names.append(base + String(i))
    return _to_list(names)


def _kpca_names(model: PythonObject, fields: PythonObject, context: PythonObject) raises -> PythonObject:
    """KernelPCA: `kernelpca<i>` over `eigenvalues_.shape[0]`."""
    var values = _fitted(model, fields, "eigenvalues_")
    var shape = array_shape_once(classical_view(values, context, "eigenvalues_"), context)
    if len(shape) < 1:
        raise Error("IndexError: tuple index out of range")
    return _prefixed(model, shape[0])


def _achi2_names(model: PythonObject, fields: PythonObject, source: PythonObject,
                 context: PythonObject) raises -> PythonObject:
    """AdditiveChi2Sampler: `<base>_<f>_sqrt`, then per step j from 1 the
    `<base>_<f>_cos<j>` and `<base>_<f>_sin<j>` names."""
    var names = _names_in(model, fields, source, context)
    var steps = _int_value(_fitted(model, fields, "sample_steps"), context)
    var base = _class_name(model).lower()
    var out = List[String]()
    for i in range(len(names)):  # driver-loop: finished public feature-name metadata over the d columns
        out.append(base + "_" + names[i] + "_sqrt")
    for j in range(1, steps):  # driver-loop: finished public feature-name metadata over the sample steps
        for i in range(len(names)):  # driver-loop: finished public feature-name metadata over the d columns
            out.append(base + "_" + names[i] + "_cos" + String(j))
        for i in range(len(names)):  # driver-loop: finished public feature-name metadata over the d columns
            out.append(base + "_" + names[i] + "_sin" + String(j))
    return _to_list(out)


def _imputer_names(model: PythonObject, fields: PythonObject, source: PythonObject,
                   context: PythonObject) raises -> PythonObject:
    """KNNImputer: the kept input names, then `missingindicator_<name>` for
    each indicator column when `add_indicator` is True."""
    var b = native_loaded_module("builtins")
    var names = _names_in(model, fields, source, context)
    var valid = _fitted(model, fields, "_valid")
    var keep_empty = _truth_value(_fitted(model, fields, "keep_empty_features"), context)
    if py_type(valid) is not b.list:
        raise Error("TypeError: KNNImputer fitted column masks must be the native list fit stored")
    if len(valid) < len(names):
        raise Error("IndexError: list index out of range")
    var out = List[String]()
    for f in range(len(names)):  # driver-loop: finished public feature-name metadata over the d columns
        if keep_empty or _truth_value(valid[f], context):
            out.append(names[f])
    if _truth_value(_fitted(model, fields, "add_indicator"), context):
        var miss = _fitted(model, fields, "_miss_cols")
        if py_type(miss) is not b.list:
            raise Error("TypeError: KNNImputer indicator columns must be the native list fit stored")
        for i in range(len(miss)):  # driver-loop: finished public feature-name metadata over the indicator columns
            var f = _int_value(miss[i], context)
            if f < 0 or f >= len(names):
                raise Error("IndexError: list index out of range")
            out.append("missingindicator_" + names[f])
    return _to_list(out)


def neighbors_feature_names(model: PythonObject, operation: String, args: PythonObject,
                            context: PythonObject) raises -> PythonObject:
    """`family_xn_names_*`: args = (input_features,)."""
    classical_connect_numpy(context)
    var fields = py_fields(model)
    var source = args[0] if len(args) > 0 else PythonObject(None)
    if operation == "family_xn_names_kpca":
        return _kpca_names(model, fields, context)
    if operation == "family_xn_names_schi2":
        return _prefixed(model, _int_value(_fitted(model, fields, "n_components"), context))
    if operation == "family_xn_names_achi2":
        return _achi2_names(model, fields, source, context)
    if operation == "family_xn_names_imputer":
        return _imputer_names(model, fields, source, context)
    raise Error("RuntimeError: unknown neighbors operation " + operation)
