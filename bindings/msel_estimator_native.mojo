# SPDX-License-Identifier: Apache-2.0
# Direct native estimator ABI calls. Python only connects these fixed plans
# at package import; CV never calls a constructor, method or clone callback.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.native_buffer_owner import native_storage_allocate
from bindings.classical_attributes import classical_connect_numpy
from bindings.family_common import family_call
from bindings.sequential_admission import sequential_admit_value
from bindings.trees_public_forest import forest_get_params, forest_refresh


def msel_estimator_plan(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var cls = py_type(model)
    if py_type(cls) is not b.type:
        raise Error("TypeError: native cross-validation refuses Python estimator metaclass hooks")
    var plan = py_get(context["family_msel_estimators"], cls)
    if plan is None:
        var module = py_call(b.type.__dict__["__module__"].__get__, cls)
        if py_contains(context["family_msel_pending_estimators"], cls) or (py_type(module) is b.str and String(module).startswith("mojolearn.")):
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
        if py_contains(fields, protected[k]):
            raise Error("TypeError: native estimator instance overrides a connected method or protocol")
    return plan


def msel_estimator_call(model: PythonObject, method: String, args: PythonObject,
                        context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var plan = msel_estimator_plan(model, context)
    var entry = plan["entry"]
    if entry is None or py_type(entry) is not py_type(b.len):
        raise Error("ImportError: native cross-validation estimator binding is unavailable")
    var method_plan = py_get(plan["methods"], PythonObject(method))
    if method_plan is None:
        raise Error("AttributeError: native estimator has no " + method)
    if method == "fit" and py_contains(plan, "fit_entry"):
        var fit = plan["fit_entry"]
        if py_type(fit) is not py_type(b.len):
            raise Error("TypeError: native scaler fitting entry is unavailable")
        return family_call(fit, model, args[0], args[1], PythonObject(None), plan["context"])
    var values = Python.list()
    var recipe = method_plan[1]
    for k in range(len(recipe)):  # driver-loop: fixed native ABI arguments
        var item = recipe[k]
        var value = args[Int(py=item[1])] if String(item[0]) == "arg" else item[1]
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
    return family_call(entry, model, method_plan[0], envelope, plan["context"])


def msel_clone_parameter(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var kind = py_type(value)
    if value is None or kind is b.bool or kind is b.int or kind is b.float or kind is b.complex or kind is b.str or kind is b.bytes or kind is b.range:
        return value
    if kind is b.dict:
        var result = Python.dict()
        var pairs = py_call(b.list, value.items())
        for k in range(len(pairs)):  # driver-loop: native constructor parameter mapping
            var key = msel_clone_parameter(pairs[k][0], context)
            py_setitem(result, key, msel_clone_parameter(pairs[k][1], context))
        return result
    if kind is b.list or kind is b.tuple or kind is b.set or kind is b.frozenset:
        var items = py_call(b.list, value)
        var result = Python.list()
        for k in range(len(items)):  # driver-loop: native constructor parameter containers
            _ = py_call(result.append, msel_clone_parameter(items[k], context))
        return result if kind is b.list else py_call(kind, result)
    classical_connect_numpy(context)
    var scalars = context["scalar_types"]
    for k in range(len(scalars)):  # driver-loop: connected immutable NumPy scalar classes
        if kind is scalars[k]:
            return value
    if py_type(kind) is b.type and py_contains(context["family_msel_estimators"], kind):
        return msel_native_clone(value, context)
    # Never invoke deepcopy/__reduce__/buffer protocols of an arbitrary
    # object. Mutable constructor-buffer and additional native schema
    # migration remain explicit debt rather than sharing fitted state.
    sequential_admit_value(value, context)
    raise Error("RuntimeError: native cloning of this constructor parameter remains pending migration")


def msel_native_clone(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var plan = msel_estimator_plan(model, context)
    var names = plan["parameters"]
    var given: PythonObject
    if plan["forest"] is not None:
        given = forest_get_params(model, plan["context"])
    else:
        given = msel_estimator_call(model, "get_params", Python.tuple(), context)
    var parameters = Python.dict()
    var args = Python.list()
    for k in range(len(names)):  # driver-loop: fixed constructor parameters, no fitted state
        var value = msel_clone_parameter(given[names[k]], context)
        py_setitem(parameters, names[k], value)
        _ = py_call(args.append, value)
    var result = native_storage_allocate(py_type(model))
    if plan["forest"] is not None:
        var fields = py_call(plan["fields_get"], result, py_type(result))
        _ = py_call(fields.update, parameters)
        py_setitem(fields, "_has_constructor_parameters", PythonObject(True))
        _ = forest_refresh(result, String(plan["forest"]), plan["context"])
    else:
        var entry = plan["entry"]
        if entry is None or py_type(entry) is not py_type(b.len):
            raise Error("ImportError: native estimator constructor binding is unavailable")
        _ = family_call(entry, result, plan["init"], py_call(b.tuple, args), plan["context"])
    return result
