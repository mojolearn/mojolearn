# SPDX-License-Identifier: Apache-2.0
# Direct native estimator ABI calls. Python only connects these fixed plans
# at package import; CV never calls a constructor, method or clone callback.
from bindings.native_array_storage import native_array_empty
from bindings.native_loaded_module import native_loaded_module
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.native_buffer_owner import native_storage_allocate, native_buffer_view
from bindings.native_numpy_metadata import native_numpy_type
from bindings.classical_arrays import classical_wrap_array, _store_code
from bindings.msel_inputs import msel_input
from bindings.classical_attributes import classical_connect_numpy
from bindings.family_common import family_call
from bindings.sequential_admission import sequential_admit_value


def msel_estimator_plan(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var cls = py_type(model)
    if py_type(cls) is not b.type:
        raise Error("TypeError: native cross-validation refuses Python estimator metaclass hooks")
    var plan = py_get(context["family_msel_estimators"], cls)
    if plan is None:
        # Only an exact import-connected built-in can be pending migration.
        # An external class may use any __module__ string (including our
        # namespace); that is neither a native identity nor an ABI promise.
        if py_contains(context["family_msel_pending_estimators"], cls):
            raise Error("RuntimeError: this built-in estimator's native cross-validation schema is pending migration")
        raise Error("TypeError: Python estimator callbacks are not accepted; use a connected native estimator")
    var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
    if len(mro) != len(plan["mro"]):
        raise Error("TypeError: native estimator class ancestry changed")
    for k in range(len(mro)):  # driver-loop: connected estimator class ancestry
        if mro[k] is not plan["mro"][k]:
            raise Error("TypeError: native estimator class ancestry changed")
    var hooks = plan["hooks"]
    for k in range(len(hooks)):  # driver-loop: connected estimator method and storage identities
        var fields = py_call(b.type.__dict__["__dict__"].__get__, hooks[k][0])
        if py_get(fields, hooks[k][1]) is not hooks[k][2]:
            raise Error("TypeError: native estimator methods or storage hooks changed")
    var fields = py_call(plan["fields_get"], model, cls)
    var names = py_call(b.list, fields.keys())
    for k in range(len(names)):  # driver-loop: instance keys before dictionary lookup
        if py_type(names[k]) is not b.str:
            raise Error("TypeError: native estimator state requires exact string keys")
    var protected = plan["protected"]
    for k in range(len(protected)):  # driver-loop: refuse instance method/protocol overrides before dispatch
        if String(protected[k]) == "_has_constructor_parameters":
            # Forest constructors deliberately store this native bool on
            # each instance; the class identity is still protected above.
            var flag = py_get(fields, protected[k])
            if flag is not None and py_type(flag) is not b.bool:
                raise Error("TypeError: native forest constructor-state flag must be an exact bool")
            continue
        if py_contains(fields, protected[k]):
            raise Error("TypeError: native estimator instance overrides a connected method or protocol")
    return plan


def msel_estimator_call(model: PythonObject, method: String, args: PythonObject,
                        context: PythonObject, scorer_response: Bool = False,
                        keywords: PythonObject = PythonObject(None)) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var plan = msel_estimator_plan(model, context)
    if method == "__admit__":
        # Validate constructor configuration without executing any of its
        # Python protocols. Native clone owns the recursive value schemas.
        _ = msel_native_clone(model, context)
        return PythonObject(None)
    var entry = plan["entry"]
    if entry is None or py_type(entry) is not py_type(b.len):
        raise Error("ImportError: native cross-validation estimator binding is unavailable")
    var method_plan = py_get(plan["methods"], PythonObject(method))
    if method_plan is None:
        raise Error("AttributeError: native estimator has no " + method)
    var allowed = py_get(py_get(plan, "keywords", Python.dict()), PythonObject(method), Python.tuple())
    if keywords is not None:
        if py_type(keywords) is not b.dict:
            raise Error("TypeError: native estimator keywords require an exact dict")
        var keys = py_call(b.list, keywords.keys())
        for k in range(len(keys)):  # driver-loop: named fit arguments before native dispatch
            if py_type(keys[k]) is not b.str:
                raise Error("TypeError: native estimator keyword names must be exact strings")
            if not py_contains(allowed, keys[k]):
                raise Error("TypeError: " + String(py_type(model).__name__) + "." + method
                            + " got an unexpected keyword argument '" + String(keys[k]) + "'")
            sequential_admit_value(keywords[keys[k]], context)
    if method == "fit" and py_contains(plan, "fit_entry"):
        var fit = plan["fit_entry"]
        if py_type(fit) is not py_type(b.len):
            raise Error("TypeError: native scaler fitting entry is unavailable")
        return family_call(fit, model, args[0], args[1], py_get(keywords, "sample_weight")
                           if keywords is not None else PythonObject(None), plan["context"])
    var values = Python.list()
    var recipe = method_plan[1]
    for k in range(len(recipe)):  # driver-loop: fixed native ABI arguments
        var item = recipe[k]
        var kind = String(item[0])
        var value = item[1]
        if kind == "arg":
            value = args[Int(py=item[1])]
        elif kind == "keyword":
            value = py_get(keywords, item[1], item[2]) if keywords is not None else item[2]
        elif kind == "binding":
            var fields = py_call(plan["fields_get"], model, py_type(model))
            var mode = py_get(fields, "numeric_mode")
            if mode is None:
                mode = context["family_msel_backend_fields"]["_DEFAULT_MODE"]
            if py_type(mode) is not b.str:
                raise Error("TypeError: native estimator numeric_mode must be an exact string or None")
            var modes = context["family_msel_bindings"][item[1]]
            value = py_get(modes, mode)
            if value is None:
                raise Error("ImportError: native estimator binding is unavailable for " + String(mode))
            if py_type(value) is b.str:
                raise Error(String(value))
        _ = py_call(values.append, value)
    var envelope = py_call(b.tuple, values)
    if method == "fit" and plan["forest"] is not None:
        var fields = py_call(plan["fields_get"], model, py_type(model))
        var before = py_get(fields, "_mojolearn_fitting", PythonObject(0))
        if py_type(before) is not b.int:
            raise Error("TypeError: native estimator fit depth must be an exact integer")
        var depth = Int(py=before)
        py_setitem(fields, "_mojolearn_fitting", PythonObject(depth + 1))
        var result = PythonObject(None)
        try:
            result = family_call(entry, model, method_plan[0], envelope, plan["context"])
        finally:
            fields = py_call(plan["fields_get"], model, py_type(model))
            if depth > 0:
                py_setitem(fields, "_mojolearn_fitting", PythonObject(depth))
            else:
                _ = py_call(fields.pop, "_mojolearn_fitting", PythonObject(None))
        py_setitem(fields, "_mojolearn_fitted", PythonObject(True))
        return result
    var operation = method_plan[0]
    var override = py_get(py_get(plan, "method_entries", Python.dict()), PythonObject(method))
    if override is not None:
        if py_type(override[0]) is not py_type(b.len):
            raise Error("ImportError: native estimator method binding is unavailable")
        return family_call(override[0], model, operation, envelope, override[1])
    if scorer_response and method == "predict":
        var internal_operation = py_get(plan, "scorer_predict_operation")
        if internal_operation is not None:
            operation = internal_operation
    return family_call(entry, model, operation, envelope, plan["context"])


def msel_clone_parameter(value: PythonObject, context: PythonObject, depth: Int = 0) raises -> PythonObject:
    if depth > 64:
        raise Error("TypeError: native estimator configuration exceeds the supported nesting depth")
    var b = native_loaded_module("builtins")
    var kind = py_type(value)
    if value is None or kind is b.bool or kind is b.int or kind is b.float or kind is b.complex or kind is b.str or kind is b.bytes or kind is b.range:
        return value
    if kind is b.type and py_contains(context["family_msel_splitter_layouts"], value):
        # Repeated splitters store their exact native splitter class as data.
        return value
    if kind is b.dict:
        var result = Python.dict()
        var pairs = py_call(b.list, value.items())
        for k in range(len(pairs)):  # driver-loop: native constructor parameter mapping
            var key = msel_clone_parameter(pairs[k][0], context, depth + 1)
            py_setitem(result, key, msel_clone_parameter(pairs[k][1], context, depth + 1))
        return result
    if kind is b.list or kind is b.tuple or kind is b.set or kind is b.frozenset:
        var items = py_call(b.list, value)
        var result = Python.list()
        for k in range(len(items)):  # driver-loop: native constructor parameter containers
            _ = py_call(result.append, msel_clone_parameter(items[k], context, depth + 1))
        return result if kind is b.list else py_call(kind, result)
    classical_connect_numpy(context)
    var scalars = context["scalar_types"]
    for k in range(len(scalars)):  # driver-loop: connected immutable NumPy scalar classes
        if kind is scalars[k]:
            return value
    if kind is context["array_class"]:
        sequential_admit_value(value, context)
        var packed = msel_input(value, "constructor parameter", context)
        var store = native_array_empty(String(_store_code(String(packed.dtype))))
        _ = py_call(store.frombytes, py_call(packed._mv.tobytes))
        _ = packed  # keep the native buffer owner alive through the copy
        return classical_wrap_array(store, packed.shape, String(packed.dtype), "C", context)
    if kind is py_get(context, "ndarray"):
        sequential_admit_value(value, context)
        # Numeric primitive dtype bytes own no Python object references.
        # Copy the admitted native buffer, then wrap independent storage;
        # neither ndarray.copy nor deepcopy/NumPy Python hooks are called.
        if Bool(py=value.dtype.hasobject):
            raise Error("TypeError: native constructor buffers cannot contain Python object references")
        var view = native_buffer_view(value)
        var order = PythonObject("F" if Bool(py=view.f_contiguous) and not Bool(py=view.c_contiguous) else "C")
        var store = py_call(b.bytearray, py_call(view.tobytes, order))
        var cls = native_numpy_type("ndarray", context)
        return py_call(cls, view.shape, dtype=value.dtype, buffer=store, order=order)
    if kind is context["array_type"]:
        var result = native_array_empty(String(value.typecode))
        var view = native_buffer_view(value)
        _ = py_call(result.frombytes, py_call(view.tobytes))
        _ = view  # keep the acquired native buffer alive through the copy
        return result
    if kind is b.bytearray:
        return py_call(b.bytearray, value)
    if py_type(kind) is b.type and py_contains(context["family_msel_estimators"], kind):
        return msel_native_clone(value, context, depth + 1)
    # Never invoke deepcopy/__reduce__/buffer protocols of an arbitrary
    # object. Additional native buffer/schema migration remains explicit
    # debt rather than sharing mutable constructor or fitted state.
    sequential_admit_value(value, context)
    raise Error("RuntimeError: native cloning of this constructor parameter remains pending migration")


def msel_native_clone(model: PythonObject, context: PythonObject, depth: Int = 0) raises -> PythonObject:
    if depth > 64:
        raise Error("TypeError: native estimator configuration exceeds the supported nesting depth")
    var b = native_loaded_module("builtins")
    var plan = msel_estimator_plan(model, context)
    var names = plan["parameters"]
    var given: PythonObject
    if plan["forest"] is not None:
        given = family_call(plan["entry"], model, "trees_forest_get_params", Python.tuple(), plan["context"])
    elif plan["init"] is None:
        given = py_call(plan["fields_get"], model, py_type(model))
    else:
        given = msel_estimator_call(model, "get_params", Python.tuple(), context)
    var parameters = Python.dict()
    var args = Python.list()
    for k in range(len(names)):  # driver-loop: fixed constructor parameters, no fitted state
        if not py_contains(given, names[k]):
            raise Error("TypeError: native constructor parameter is missing from estimator state")
        var value = msel_clone_parameter(given[names[k]], context, depth + 1)
        py_setitem(parameters, names[k], value)
        _ = py_call(args.append, value)
    var result = native_storage_allocate(py_type(model))
    if plan["forest"] is not None:
        var fields = py_call(plan["fields_get"], result, py_type(result))
        _ = py_call(fields.update, parameters)
        py_setitem(fields, "_has_constructor_parameters", PythonObject(True))
        # Public forest constructors store parameters verbatim; native fit
        # derives/validates configuration in its original owner later.
    elif plan["init"] is None:
        # Exact fixed constructor schemas containing only field assignments.
        var fields = py_call(plan["fields_get"], result, py_type(result))
        _ = py_call(fields.update, parameters)
        var fixed = py_get(plan, "initial_fields")
        if fixed is not None:
            _ = py_call(fields.update, msel_clone_parameter(fixed, context))
    else:
        var entry = plan["entry"]
        if entry is None or py_type(entry) is not py_type(b.len):
            raise Error("ImportError: native estimator constructor binding is unavailable")
        _ = family_call(entry, result, plan["init"], py_call(b.tuple, args), plan["context"])
    return result


def msel_native_set_params(model: PythonObject, params: PythonObject,
                           context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var plan = msel_estimator_plan(model, context)
    if py_type(params) is not b.dict:
        raise Error("TypeError: native estimator parameters require a native dict")
    var keys = py_call(b.list, params.keys())
    for k in range(len(keys)):  # driver-loop: parameter admission before hashing or native setter dispatch
        if py_type(keys[k]) is not b.str:
            raise Error("TypeError: native estimator parameter names must be exact strings")
        sequential_admit_value(params[keys[k]], context)
    if len(keys) == 0:
        return model
    if plan["init"] is None and plan["forest"] is None and not py_contains(plan["methods"], PythonObject("set_params")):
        # These exact constructors only assign their stored parameter schema.
        # Native search can reset them without a Python ParamsMixin call.
        var fields = py_call(plan["fields_get"], model, py_type(model))
        var names = plan["parameters"]
        var given = Python.dict()
        for k in range(len(names)):  # driver-loop: retain constructor configuration only
            py_setitem(given, names[k], fields[names[k]])
        for k in range(len(keys)):  # driver-loop: validate all names before mutating state
            if not py_contains(names, keys[k]):
                raise Error("ValueError: invalid native estimator parameter " + String(keys[k]))
            py_setitem(given, keys[k], msel_clone_parameter(params[keys[k]], context))
        _ = py_call(fields.clear)
        _ = py_call(fields.update, given)
        var fixed = py_get(plan, "initial_fields")
        if fixed is not None:
            _ = py_call(fields.update, msel_clone_parameter(fixed, context))
        return model
    if plan["forest"] is None:
        return msel_estimator_call(model, "set_params", Python.tuple(params), context)
    var values = family_call(plan["entry"], model, "trees_forest_get_params", Python.tuple(), plan["context"])
    var unknown = Python.list()
    for k in range(len(keys)):  # driver-loop: complete forest parameter name validation before mutation
        if not py_contains(values, keys[k]):
            _ = py_call(unknown.append, keys[k])
    if len(unknown):
        raise Error("Invalid parameter(s) " + String(py_call(b.repr, py_call(b.sorted, unknown)))
                    + " for " + String(py_type(model).__name__))
    _ = py_call(values.update, params)
    var fields = py_call(plan["fields_get"], model, py_type(model))
    _ = py_call(fields.clear)
    _ = py_call(fields.update, values)
    py_setitem(fields, "_has_constructor_parameters", PythonObject(True))
    return model
