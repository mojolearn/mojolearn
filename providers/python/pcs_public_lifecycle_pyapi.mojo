# SPDX-License-Identifier: Apache-2.0
# PolynomialCountSketch public fit/transform envelope. One X snapshot feeds
# finite/width checks and both fit_transform stages; target owners survive it.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.native_loaded_module import native_loaded_module, _loaded_fields
from bindings.mode_metadata_native import _fields, _class_fields, _fitted
from bindings.classical_attributes import classical_connect_numpy, _type_mro
from bindings.classical_exceptions import classical_not_fitted_class
from bindings.classical_validate import classical_estimator_name, classical_check_width
from bindings.public_attributes_native import _plan, _lookup
from bindings.host_inference_scalars import host_inference_scalar, host_inference_primitive_number
from bindings.family_common import family_input_snapshot, family_snapshot_f32
from bindings.native_numpy_metadata import native_numpy_dtype
from core.native_scalar_decode import native_scalar_real
from core.cast_in import CAST_KIND_FLOAT
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


def _pcs_finite_policy(source: PythonObject, context: PythonObject) raises -> Int:
    # Match _sk_validate.describe/_check_X after native admission, without
    # rereading an exporter's metadata. 0 skips; 1 is an ordinary buffer;
    # 2 is a nested sequence whose first leaf is a real floating scalar.
    var b = native_loaded_module("builtins")
    var cls = py_type(source)
    if cls is b.list or cls is b.tuple:
        var first = source
        var depth = 0
        while py_type(first) is b.list or py_type(first) is b.tuple:  # driver-loop: first-element sequence kind, never buffer metadata
            if len(first) == 0:
                return 0
            first = first[0]
            depth += 1
            if depth > 64:
                raise Error("TypeError: native PCS input nesting exceeds the supported depth")
        var first_cls = py_type(first)
        if first_cls is b.float:
            return 2
        if py_get(context, "ndarray") is not None:
            var dtype = native_numpy_dtype(context)
            var codes = Python.tuple("e", "f", "d", "g")
            for i in range(len(codes)):  # small-loop(four exact native NumPy floating scalar identities)
                if first_cls is py_call(dtype, codes[i]).type:
                    return 2
        return 0
    if (cls is context["array_class"] or cls is py_get(context, "ndarray")
            or cls is b.memoryview or cls is context["array_type"] or cls is b.bytearray):
        return 1
    # Sparse, pandas, Torch and other admitted frame inputs did not run the
    # old buffer/list-only finite check. Their native preparation still runs.
    return 0


def _pcs_check_finite(source: PythonObject, snapshot: ArrayMetadata,
                      context: PythonObject) raises:
    var policy = _pcs_finite_policy(source, context)
    var data = snapshot.source
    if policy == 0 or data.kind != CAST_KIND_FLOAT:
        return
    # _all_finite accepted only the materialized f4/f8 Array. A direct half
    # buffer stayed f2 and skipped the scan; a list of half scalars materialized
    # as f8 and was scanned. Preserve that distinction before the f32 cast.
    if policy == 1 and data.itemsize == 2:
        return
    var code = String("e") if data.itemsize == 2 else String("f") if data.itemsize == 4 else String("d")
    var bytes = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=data.addr)
    for row in range(data.rows):  # driver-loop: finite check over the captured pre-cast input
        for column in range(data.cols):
            if native_scalar_real(bytes, row * data.s0 + column * data.s1, code, data.itemsize, True).special != 0:
                raise Error("Input X contains NaN or infinity (PolynomialCountSketch).")


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
    var x = family_input_snapshot(args[1], 2, "X", context, True)
    _pcs_check_finite(args[1], x, context)
    family_snapshot_f32(x, context)
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
