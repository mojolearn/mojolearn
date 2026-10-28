# SPDX-License-Identifier: Apache-2.0
# PolynomialCountSketch public fit/transform envelope. One X snapshot feeds
# finite/width checks and both fit_transform stages; target owners survive it.
from std.math import isfinite
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.native_loaded_module import native_loaded_module, _loaded_fields
from bindings.mode_metadata_native import _fields, _class_fields, _fitted
from bindings.classical_attributes import classical_connect_numpy, _type_mro
from bindings.classical_exceptions import classical_not_fitted_class
from bindings.classical_validate import classical_estimator_name, classical_check_width
from bindings.public_attributes_native import _plan, _lookup
from bindings.host_inference_scalars import host_inference_scalar, host_inference_primitive_number
from bindings.family_common import FAMILY_F32P, family_f32_snapshot
from providers.python.array_metadata_pyapi import ArrayMetadata
from providers.python.radius_public_lifecycle_pyapi import radius_target
from providers.python.neighbors_public_lifecycle_pyapi import NeighborPublicCall
from providers.python.estimator_lifecycle_pyapi import estimator_outside, estimator_fit_begin, estimator_feature_check


struct PCSPublicCall(Movable):
    var call: NeighborPublicCall
    var settings: PythonObject
    var x: ArrayMetadata
    var target_owner: PythonObject

    def __init__(out self, var call: NeighborPublicCall, settings: PythonObject,
                 var x: ArrayMetadata, target_owner: PythonObject):
        self.call = call^
        self.settings = settings
        self.x = x^
        self.target_owner = target_owner


def pcs_public_begin(model: PythonObject, operation: String, args: PythonObject,
                     context: PythonObject) raises -> PCSPublicCall:
    classical_connect_numpy(context)
    var b = native_loaded_module("builtins")
    var fields = _fields(model)
    var attributes = py_get(_loaded_fields(native_loaded_module("mojolearn._native_public_attributes")), "_CONTEXT")
    var plan = _plan(model, attributes)
    var fitting = operation != "family_xn_pcs_transform"
    var method = String("fit") if operation == "family_xn_pcs_fit" else String("fit_transform") if fitting else String("transform")
    var ancestry = _type_mro(py_type(model))
    var setter_seen = False
    for i in range(len(ancestry)):  # driver-loop: connected native model field/callback admission
        var names = _class_fields(ancestry[i])
        if not setter_seen and py_contains(names, "__setattr__"):
            if names["__setattr__"] is not b.object.__dict__["__setattr__"]:
                raise Error("TypeError: native PCS cannot invoke Python field setters")
            setter_seen = True
        if fitting and py_contains(names, "_mojolearn_after_fit"):
            raise Error("TypeError: native PCS cannot invoke Python after-fit callbacks")
    if fitting and py_contains(fields, "_mojolearn_after_fit"):
        raise Error("TypeError: native PCS cannot invoke Python after-fit callbacks")
    var depth = py_get(fields, "_mojolearn_fitting", PythonObject(0))
    var fitted = py_get(fields, "_mojolearn_fitted", PythonObject(False))
    if py_type(depth) is not b.int or (py_type(fitted) is not b.int and py_type(fitted) is not b.bool):
        raise Error("TypeError: native PCS fitted flags/depth require exact integer or boolean values")
    if not fitting and not Bool(py=_fitted(model)):
        _ = classical_not_fitted_class(context)
        raise Error("NotFittedError: This PolynomialCountSketch instance is not fitted yet. Call 'fit' with appropriate arguments before using transform.")
    var settings = Python.dict()
    var names = Python.tuple("degree", "n_components", "gamma", "coef0")
    for i in range(len(names)):  # small-loop(four captured sketch scalar parameters)
        var value = _lookup(model, names[i], fields, plan, attributes)
        host_inference_scalar(value, context)
        py_setitem(settings, names[i], value)
    if fitting:
        py_setitem(settings, "random_state", _lookup(model, PythonObject("random_state"), fields, plan, attributes))
    else:
        var learned = Python.tuple("indexHash_", "bitHash_", "n_features_in_")
        for i in range(len(learned)):  # small-loop(three fitted sketch input owners)
            py_setitem(settings, learned[i], _lookup(model, learned[i], fields, plan, attributes))
    var x = family_f32_snapshot(args[1], 2, "X", context, True)
    var data = FAMILY_F32P(unsafe_from_address=x.source.addr)
    for i in range(x.count):  # driver-loop: the public PCS finite-input check, once for fit_transform
        if not isfinite(data[i]):
            raise Error("Input X contains NaN or infinity (PolynomialCountSketch).")
    var target = PythonObject(None)
    if Int(py=depth) == 0:
        if fitting:
            target = radius_target(args[2] if len(args) > 2 else PythonObject(None), x.shape[0], context,
                                   False, PythonObject(None), "PolynomialCountSketch")
        else:
            host_inference_scalar(settings["n_features_in_"], context)
            classical_check_width(x.shape[1], Int(py=settings["n_features_in_"]), "PolynomialCountSketch")
            estimator_feature_check(model, fields, args[1], context)
    var previous = estimator_fit_begin(fields) if fitting else Int(py=depth)
    var call = NeighborPublicCall(args, fields, args[1], method, fitting, estimator_outside(), previous, x.shape[0], x.shape[1])
    return PCSPublicCall(call^, settings, x^, target)


def pcs_feature_names(model: PythonObject, context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    var b = native_loaded_module("builtins")
    var fields = _fields(model)
    var attributes = py_get(_loaded_fields(native_loaded_module("mojolearn._native_public_attributes")), "_CONTEXT")
    var plan = _plan(model, attributes)
    var count = _lookup(model, PythonObject("n_components"), fields, plan, attributes)
    host_inference_scalar(count, context)
    var length = Int(py=py_call(b.int, host_inference_primitive_number(count, context, True)))
    var cls_name = String(py_call(b.type.__dict__["__name__"].__get__, py_type(model))).lower()
    var result = Python.list()
    for i in range(max(length, 0)):  # driver-loop: finished public feature-name metadata
        _ = py_call(result.append, PythonObject(cls_name + String(i)))
    return result
