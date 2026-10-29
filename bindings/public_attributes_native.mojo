from bindings.trees_bagging_members_native import bagging_members
# SPDX-License-Identifier: Apache-2.0
# Native public attribute resolution and zero-copy fitted-output boundary.
# No generic object getter, property invocation, Array.__array__, or Python
# caller-frame inspection is part of the operation.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_loaded_module import native_loaded_module, NativeModules, native_namespace_string_keys
from bindings.native_caller_frame import native_frame_globals
from bindings.mode_metadata_native import _fields, _fields_from_admitted_rows, _class_fields, _truth
from bindings.classical_attributes import _type_mro, _classical_descriptor_get, classical_connect_numpy
from bindings.native_array_view import native_array_ndarray_view as _array_view
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.native_error_class import error_without_class
from bindings.native_instance_item import native_instance_find_admitted
from bindings.native_mode_recipe import mode_recipe_call
from bindings.optimizer_public_words import optimizer_python_error, stash_python_error, PYTHON_ERROR_PENDING
from bindings.host_inference_scalars import host_inference_scalar
from providers.python.neural_descriptor_pyapi import neural_descriptor_entry


def _string_keys(fields: PythonObject) raises:
    # Both callers supply an exact dictionary (native frame globals or an
    # explicitly admitted state dict). Avoid key views, lists and indexed boxes.
    if not native_namespace_string_keys(fields._obj_ptr, py_type(PythonObject(""))):
        raise Error("TypeError: native attribute metadata requires exact string keys")


def _outside() raises -> Bool:
    ref cpy = Python().cpython()
    var pointer = cpy.lib.get_function[PyObjectPtr]("PyEval_GetFrame")()
    if not pointer:
        return True
    var frame = PythonObject(from_borrowed=pointer)
    while True:  # driver-loop: skip only the old public wrapper frames
        var fields = native_frame_globals(frame)
        _string_keys(fields)
        var module = py_get(fields, "__name__")
        var skip = False
        if py_type(module) is py_type(PythonObject("")) and String(module) == "mojolearn._mode":
            pointer = cpy.lib.get_function[PyObjectPtr]("PyFrame_GetCode")(frame._obj_ptr)
            if not pointer:
                raise cpy.unsafe_get_error()
            var code = PythonObject(from_owned=pointer)
            var name = String(code.co_name)
            # __getattribute__ is the generic one-call slot below
            # (PublicOutputsMixin.__getattribute__): its caller decides.
            skip = (name == "public" or name == "guarded" or name == "tracked" or name == "load"
                    or name == "__getattribute__")
        if not skip:
            if py_get(fields, "__mojolearn_user_boundary__") is PythonObject(True):
                return True
            if py_get(fields, "__mojolearn_internal__") is PythonObject(True):
                return False
            if not _truth(module):
                module = PythonObject("")
            if py_type(module) is not py_type(PythonObject("")):
                raise Error("TypeError: native output caller module name must be a string")
            var name = String(module)
            return name != "mojolearn" and not name.startswith("mojolearn.") and not name.startswith("mojolearn_verify_all_")
        pointer = cpy.lib.get_function[PyObjectPtr]("PyFrame_GetBack")(frame._obj_ptr)
        _ = frame
        if not pointer:
            if cpy.lib.get_function[PyObjectPtr]("PyErr_Occurred")():
                raise cpy.unsafe_get_error()
            return True
        frame = PythonObject(from_owned=pointer)


struct _AttributeAdmission(Movable):
    var plan: PythonObject
    var ancestry: PythonObject
    var rows: List[PythonObject]

    def __init__(out self, plan: PythonObject, ancestry: PythonObject, var rows: List[PythonObject]):
        self.plan = plan
        self.ancestry = ancestry
        self.rows = rows^


def _plan(mut mods: NativeModules, model: PythonObject, context: PythonObject) raises -> _AttributeAdmission:
    var b = mods.module("builtins")
    var cls = py_type(model)
    if py_type(cls) is not b.type:
        raise Error("TypeError: native attributes refuse Python metaclass protocols")
    var mro = _type_mro(cls)
    var rows = List[PythonObject](capacity=len(mro))
    var plan = PythonObject(None)
    var getter = context["missing"]
    for i in range(len(mro)):  # driver-loop: validate native slot and passive subclass ancestry
        if py_type(mro[i]) is not b.type:
            raise Error("TypeError: native attributes refuse Python metaclass protocols")
        var namespace = _class_fields(mods, mro[i])
        rows.append(namespace)
        if py_contains(namespace, "__getattr__"):
            raise Error("TypeError: native attributes refuse Python __getattr__ hooks")
        if getter is context["missing"] and py_contains(namespace, "__getattribute__"):
            getter = namespace["__getattribute__"]
        if plan is None:
            plan = py_get(context["plans"], mro[i])
    if plan is None or getter is not context["descriptor"]:
        raise Error("TypeError: native attribute class has no connected slot")
    return _AttributeAdmission(plan, mro, rows^)


def _has_slot(cls: PythonObject, name: String) raises -> Bool:
    var ancestry = _type_mro(cls)
    for i in range(len(ancestry)):  # driver-loop: descriptor metadata without foreign key equality hooks
        if py_contains(_class_fields(ancestry[i]), name):
            return True
    return False


def _member(admission: _AttributeAdmission, name: PythonObject) raises -> PythonObject:
    for i in range(len(admission.rows)):  # driver-loop: exact native attribute resolution
        var namespace = admission.rows[i]
        if py_contains(namespace, name):
            return Python.tuple(True, namespace[name])
    return Python.tuple(False, None)


def _bound_function(function: PythonObject, model: PythonObject) raises -> PythonObject:
    # Exact Python function binding is a native CPython operation. A function
    # can have an instance attribute named __get__; never consult that mutable
    # attribute or invoke it as a callback while binding a public method.
    if py_type(function) is not native_loaded_module("types").FunctionType:
        raise Error("TypeError: native method binding requires an exact function")
    ref cpy = Python().cpython()
    var pointer = cpy.lib.get_function[PyObjectPtr]("PyMethod_New")(function._obj_ptr, model._obj_ptr)
    if not pointer:
        raise cpy.unsafe_get_error()
    return PythonObject(from_owned=pointer)


def _availability_member(cls: PythonObject, name: PythonObject, missing: PythonObject) raises -> PythonObject:
    var ancestry = _type_mro(cls)
    for i in range(len(ancestry)):  # driver-loop: raw descriptor protocol metadata only
        var fields = _class_fields(ancestry[i])
        if py_contains(fields, name):
            return fields[name]
    return missing


def _available(model: PythonObject, member: PythonObject, schema: PythonObject,
               fields: PythonObject, admission: _AttributeAdmission, context: PythonObject) raises -> PythonObject:
    # Admitted rules are presence of transform and LOF's novelty word. Never
    # call the Python available_if descriptor, its check, or its wrapped body.
    var ancestry = _type_mro(py_type(member))
    if len(ancestry) != len(schema["mro"]):
        raise Error("TypeError: native availability descriptor ancestry changed")
    for i in range(len(ancestry)):  # driver-loop: frozen descriptor class identities
        if ancestry[i] is not schema["mro"][i]:
            raise Error("TypeError: native availability descriptor ancestry changed")
    var hooks = schema["hooks"]
    for i in range(len(hooks)):  # driver-loop: changed Python descriptor hooks are never executed
        if _availability_member(py_type(member), hooks[i][0], context["missing"]) is not hooks[i][1]:
            raise Error("TypeError: native availability descriptor hooks changed")
    var metadata = _fields(member)
    if (py_get(metadata, "check") is not schema["check"]
            or py_get(metadata, "fn") is not schema["function"]
            or py_get(metadata, "attribute_name", context["missing"]) is not schema["attribute_name"]):
        raise Error("TypeError: native availability descriptor data changed")
    ref cpy = Python().cpython()
    # Direct CPython function metadata access avoids the object.__getattr__
    # audit event emitted by Python's function.__code__ getset descriptor.
    var code = cpy.lib.get_function[PyObjectPtr]("PyFunction_GetCode")(schema["check"]._obj_ptr)
    if not code:
        raise cpy.unsafe_get_error()
    if PythonObject(from_borrowed=code) is not schema["code"]:
        raise Error("TypeError: native availability predicate code changed")
    try:
        var condition = py_get(schema, "condition")
        if condition is not None:
            if py_type(condition) is not py_type(PythonObject("")):
                raise Error("TypeError: native availability requires its connected condition")
            var rule = String(condition)
            if rule == "density_predict" or rule == "svm_probability":
                var requested = _lookup_impl[False](model, PythonObject("probability" if rule == "svm_probability" else "prediction_data"), fields, admission, context)
                var b = native_loaded_module("builtins")
                var kind = py_type(requested)
                var enabled = False
                if requested is None or kind is b.bool or kind is b.int or kind is b.float or kind is b.str or kind is b.bytes or kind is b.list or kind is b.tuple or kind is b.dict:
                    enabled = _truth(requested)
                else:
                    host_inference_scalar(requested, context, True)
                    enabled = Bool(py=requested)
                if not enabled:
                    raise Error("AttributeError: predict_proba is not available when probability=False" if rule == "svm_probability" else "AttributeError: prediction data was not requested")
            elif rule == "sgd_probability":
                var loss = _lookup_impl[False](model, PythonObject("loss"), fields, admission, context)
                if py_type(loss) is not py_type(PythonObject("")):
                    raise Error("TypeError: SGD probability availability requires an exact native loss name")
                if String(loss) != "log_loss":
                    raise Error("AttributeError: probabilities require loss='log_loss'")
            elif rule == "spectral_predict":
                var requested = _lookup_impl[False](model, PythonObject("prediction_data"), fields, admission, context)
                host_inference_scalar(requested, context, True)
                if not Bool(py=requested):
                    var references = _lookup_impl[False](model, PythonObject("_fit_X"), fields, admission, context)
                    if references is None:
                        raise Error("AttributeError: spectral prediction data was not requested or stored")
            elif rule == "hierarchy_predict":
                var requested = _lookup_impl[False](model, PythonObject("prediction_data"), fields, admission, context)
                if py_type(requested) is not py_type(PythonObject(True)) and py_type(requested) is not py_type(PythonObject(0)):
                    raise Error("TypeError: prediction_data availability requires an exact native boolean or integer")
                if not Bool(py=requested):
                    var references = _lookup_impl[False](model, PythonObject("_fit_X"), fields, admission, context)
                    if references is None:
                        raise Error("AttributeError: prediction data was not requested or stored")
            elif rule == "novelty" or rule == "not_novelty":
                var novelty = _lookup_impl[False](model, PythonObject("novelty"), fields, admission, context)
                if py_type(novelty) is not py_type(PythonObject(True)) and py_type(novelty) is not py_type(PythonObject(0)):
                    raise Error("TypeError: novelty must be an exact native boolean or integer")
                if Bool(py=novelty) != (rule == "novelty"):
                    raise Error("AttributeError: method unavailable for this novelty setting")
            else:
                raise Error("TypeError: unknown native availability condition")
        else:
            _ = _lookup_impl[False](model, PythonObject("transform"), fields, admission, context)
    except error:
        if String(error).startswith("AttributeError:"):
            var name = schema["attribute_name"]
            if name is context["missing"]:
                name = schema["function"].__name__
            if py_type(name) is not py_type(PythonObject("")):
                raise Error("TypeError: native availability method name must be an exact string")
            var cls_name = py_call(native_loaded_module("builtins").type.__dict__["__name__"].__get__, py_type(model))
            raise Error("AttributeError: This '" + String(cls_name) + "' has no attribute '" + String(name) + "'")
        raise error^
    return _bound_function(schema["function"], model)


def _property(model: PythonObject, name: PythonObject, kind: String, fields: PythonObject,
              admission: _AttributeAdmission, context: PythonObject) raises -> PythonObject:
    if kind == "svm_svc_coef" or kind == "svm_svr_coef":
        var owner = context["svm_owner"]
        var entry = owner[0]
        if py_type(entry) is not native_loaded_module("types").BuiltinFunctionType:
            raise Error("TypeError: SVM coefficient property requires its compiled owner")
        if neural_descriptor_entry(entry.__self__, "classical_native_call", False) is not entry:
            raise Error("TypeError: SVM coefficient property owner changed after connection")
        return py_call(entry, model, PythonObject("family_svc_svc_coef" if kind == "svm_svc_coef" else "family_svc_svr_coef"), Python.tuple(None), owner[1])
    if kind == "bagging_estimators":
        return bagging_members(model)
    if kind == "cholesky_logdet":
        var info = PythonObject(None)
        try:
            info = _lookup_impl[False](model, PythonObject("info_"), fields, admission, context)
        except error:
            if String(error).startswith("AttributeError:"):
                raise Error("AttributeError: mojolearn Cholesky: logdet_ is set by fit")
            raise error^
        var b = native_loaded_module("builtins")
        if py_type(info) is not b.int and py_type(info) is not b.bool and py_type(info) is not b.float:
            raise Error("TypeError: Cholesky info_ must be native numeric status")
        if _truth(info):
            raise Error("ValueError: mojolearn Cholesky: the factorization failed (info=" + String(py_call(b.str, info)) + "), so there is no determinant to report (DEVIATION 1634)")
        return _lookup_impl[False](model, PythonObject("_logdet"), fields, admission, context)
    if kind == "stl_config":
        var out = Python.dict()
        var names = Python.tuple("period", "seasonal", "seasonal_deg", "seasonal_jump", "trend", "trend_deg", "trend_jump", "low_pass", "low_pass_deg", "low_pass_jump", "robust")
        for i in range(len(names)):  # driver-loop: supported STL configuration fields
            py_setitem(out, names[i], _lookup_impl[False](model, names[i], fields, admission, context))
        return out
    var recipes = py_get(context, "native_property_recipes")
    var recipe = PythonObject(None) if recipes is None else py_get(recipes, PythonObject(kind))
    if recipe is not None:
        return mode_recipe_call(recipe["entry"], Python.tuple(model, recipe["operation"], recipe["arguments"], recipe["context"]))
    raise Error("TypeError: unconnected native property schema")


# Connected property/predicate schemas depend on ordinary fields or methods,
# never another connected property/predicate. Keep those dependency reads in
# the leaf specialization while preserving exact descriptor precedence and
# native_instance_item ownership. This makes the finite schema explicit.
# The former mutually recursive resolver stalled compilation on Mojo 1.0.0
# ed45d567 (including the unchanged 7de54456bd baseline); the internal compiler
# cause is unconfirmed. Reproduction and complete logs are retained under
# experiments/peer1_neighbors_kernels_20261027 and the corresponding evidence
# directory's cpu-compile results. No Python fallback or toolchain change.
def _lookup_impl[allow_special: Bool](model: PythonObject, name: PythonObject,
                                    fields: PythonObject, admission: _AttributeAdmission,
                                    context: PythonObject) raises -> PythonObject:
    var plan = admission.plan
    var found = _member(admission, name)
    var present = Bool(py=found[0])
    var member = found[1]
    var cls = py_type(model)
    var member_type = py_type(member)
    var descriptor = present and _has_slot(member_type, "__get__")
    var data = descriptor and (_has_slot(member_type, "__set__") or _has_slot(member_type, "__delete__"))
    var binding = py_get(plan["bindings"], name)
    var known = binding is not None and member is binding[0]
    if data:
        comptime if allow_special:
            var property_plan = py_get(plan["properties"], name)
            if property_plan is not None and member is property_plan[0]:
                return _property(model, name, String(property_plan[1]), fields, admission, context)
        if not known:
            raise Error("TypeError: Python attribute properties/descriptors require a native owner")
        return _classical_descriptor_get(binding, model, cls, plan)
    # _fields and the public/dependency-name boundary already admitted exact
    # dict/string storage. Preserve the authorized lookup repair with one probe
    # and no temporary dict allocation for a redundant type check.
    var item = native_instance_find_admitted(fields, name)
    if item[0]:
        return item[1]
    if not present:
        raise Error("AttributeError: native model has no attribute '" + String(name) + "'")
    comptime if allow_special:
        var availability = py_get(plan, "availability")
        if availability is not None:
            var schema = py_get(availability, name)
            if schema is not None and member is schema["member"]:
                return _available(model, member, schema, fields, admission, context)
    if descriptor:
        # Creating a bound method does not execute its body. Preserve ordinary
        # subclass functions and later import-time native setter connections.
        # Use CPython's method constructor, not mutable function.__get__.
        if member_type is native_loaded_module("types").FunctionType:
            return _bound_function(member, model)
        if not known:
            raise Error("TypeError: Python attribute descriptor requires a native owner")
        return _classical_descriptor_get(binding, model, cls, plan)
    return member


def _lookup(model: PythonObject, name: PythonObject, fields: PythonObject,
            admission: _AttributeAdmission, context: PythonObject) raises -> PythonObject:
    return _lookup_impl[True](model, name, fields, admission, context)


def public_attribute_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        var context = args[0]
        var model = args[1]
        var name = args[2]
        if py_type(name) is not py_type(PythonObject("")):
            raise Error("TypeError: native model attribute name must be an exact string")
        # Admission holds the GIL and invokes no application protocol. Share
        # its builtins connection across plan and instance-storage validation.
        var mods = NativeModules()
        var admission = _plan(mods, model, context)
        var fields = _fields_from_admitted_rows(mods, model, admission.ancestry, admission.rows)
        var result = _lookup(model, name, fields, admission, context)
        var text = String(name)
        if (text.byte_length() > 0 and text.endswith("_") and not text.startswith("_")
                and py_type(result) is context["array_class"]):
            classical_connect_numpy(context)
            if py_get(context, "ndarray") is not None and _outside():
                result = _array_view(result, context)
        return result.steal_data()
    except error:
        if String(error) == PYTHON_ERROR_PENDING:
            return optimizer_python_error(error)
        if String(error).startswith("AttributeError:"):
            return raise_python_exception(error_without_class(error, "AttributeError"), ExceptionType("PyExc_AttributeError"))
        if String(error).startswith("ValueError:"):
            return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
        return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)


# Generic native attribute slot installed on PublicOutputsMixin once, for every
# class without a connected native descriptor plan: application subclasses,
# private package classes and the built-ins whose property owners are pending.
# The lookup is CPython's own generic attribute operation, exactly
# object.__getattribute__: it runs no package Python. A property or other
# descriptor declared by the class is that class's own public operation and is
# audited as its own endpoint (pending owners 2026-10-28: RandomProjection
# components_, SkewedChi2Sampler.random_weights_, Bagging estimators_). The
# fitted-output rule then matches public_attribute_c: a public trailing-
# underscore name holding a mojolearn Array reads as a zero-copy NumPy view for
# callers outside the package once NumPy is loaded. Its admission context is
# the public attributes context, connected at package import
# (_native_public_attributes.connect_classes); before that connection no
# caller outside the package holds an estimator, so the value is returned.
def public_attribute_generic_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    # Retain the earlier three-argument internal entry for binary consumers.
    return _public_attribute_generic_entry[False](args_ptr)


def public_attribute_generic_slot_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    # PyInstanceMethod_New binds the model after the import-bound context.
    return _public_attribute_generic_entry[True](args_ptr)


def _public_attribute_generic_entry[slot: Bool](args_ptr: PyObjectPtr) -> PyObjectPtr:
    try:
        ref cpy = Python().cpython()
        if cpy.lib.get_function[Int]("PyTuple_Size")(args_ptr) != 3:
            raise Error("TypeError: generic public attribute expects model, name and native context")
        # CPython owns this METH_VARARGS tuple throughout the C call. Borrow
        # its items directly, so a failed lookup has no PythonObject tuple/item
        # temporaries to destroy while the original AttributeError is pending.
        # The earlier wrapper path still lost one owner reference in the
        # installed 543875 regression; its failure is retained under
        # ~/mojolearn-evidence/host-attribute-error-repair-20261029/focused.log.
        # This uses documented CPython APIs; the Mojo ownership failure's
        # internal cause remains unconfirmed.
        var model_ptr = cpy.PyTuple_GetItem(args_ptr, 1 if slot else 0)
        var name_ptr = cpy.PyTuple_GetItem(args_ptr, 2 if slot else 1)
        var raw = cpy.lib.get_function[PyObjectPtr]("PyObject_GenericGetAttr")(model_ptr, name_ptr)
        if not raw:
            # This is already a CPython error boundary. Return its original
            # exception directly, as array_native does; no fetch/restore or
            # Mojo-error round trip is needed. The former path reproduced a
            # lost instance reference on getattr(obj, missing, default), then
            # a saved-host-model/traceback cleanup crash. Internal cause is
            # unconfirmed; retained reproduction: experiments/
            # host_attribute_error_20261029/HANDOFF.txt.
            return PyObjectPtr()
        var result = PythonObject(from_owned=raw)
        var name = PythonObject(from_borrowed=name_ptr)
        var context = PythonObject(from_borrowed=cpy.PyTuple_GetItem(args_ptr, 0 if slot else 2))
        if py_type(name) is not py_type(PythonObject("")):
            return result.steal_data()
        var attributes = py_get(context, "attributes")
        if attributes is None or py_type(result) is not attributes["array_class"]:
            return result.steal_data()
        var text = String(name)
        if text.byte_length() > 0 and text.endswith("_") and not text.startswith("_"):
            classical_connect_numpy(attributes)
            if py_get(attributes, "ndarray") is not None and _outside():
                result = _array_view(result, attributes)
        return result.steal_data()
    except error:
        if String(error) == PYTHON_ERROR_PENDING:
            return optimizer_python_error(error)
        return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
