# SPDX-License-Identifier: Apache-2.0
# Exact native search/container storage. No Python attribute or subclass hooks.
from bindings.native_loaded_module import native_loaded_module
from std.python import Python, PythonObject
from bindings.py_slice_native import py_slice, py_slice_from
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.msel_inputs import msel_input, msel_admit
from bindings.arena_materialize_native import _array
from bindings.msel_estimator_native import msel_estimator_plan
from bindings.msel_scorer_native import msel_scorer_fields
from bindings.native_owner_lifetime import native_release_weakrefs
from bindings.declared_params_native import _release as declared_release
from bindings.params_metadata_native import _resolved


def msel_search_plan(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var cls = py_type(model)
    if py_type(cls) is not b.type:
        raise Error("TypeError: native search refuses Python metaclass hooks")
    var plan = py_get(context["family_msel_search_layouts"], cls)
    if plan is None:
        raise Error("TypeError: use a connected native search or parameter container; Python subclass hooks are not supported")
    var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
    if len(mro) != len(plan["mro"]):
        raise Error("TypeError: native search ancestry changed")
    for k in range(len(mro)):  # driver-loop: retained native class ancestry
        if mro[k] is not plan["mro"][k]:
            raise Error("TypeError: native search ancestry changed")
    var hooks = plan["hooks"]
    for k in range(len(hooks)):  # driver-loop: class storage/method identity admission
        var row = py_call(b.type.__dict__["__dict__"].__get__, hooks[k][0])
        if py_get(row, hooks[k][1]) is not hooks[k][2]:
            raise Error("TypeError: native search class hooks changed")
    return plan


def msel_search_fields(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var plan = msel_search_plan(model, context)
    var fields = py_call(plan["fields_get"], model, py_type(model))
    var keys = py_call(b.list, fields.keys())
    for k in range(len(keys)):  # driver-loop: stored field names before dictionary lookup
        if py_type(keys[k]) is not b.str:
            raise Error("TypeError: native search state requires exact string keys")
        if py_contains(plan["protected"], keys[k]):
            raise Error("TypeError: native search instance overrides a method or storage hook")
    return fields


def _search_release(values: PythonObject, retained: PythonObject,
                    context: PythonObject) raises:
    # Replacement is an ownership operation: the input-admission table alone
    # cannot rule out transitive __del__/weakref callbacks in old model state.
    # Reuse the declared setter's exact native model/finalizer plans, with the
    # search/scorer records and container cycle walk specific to this owner.
    var b = native_loaded_module("builtins")
    var release = context["family_msel_release_context"]
    if not Bool(py=release["ready"]):
        raise Error("RuntimeError: native search owner-release plans are not connected")
    var todo = py_call(b.list, values)
    var seen = Python.dict()
    var cursor = 0
    while cursor < len(todo):  # driver-loop: cycle-safe old ownership traversal before any replacement
        var item = todo[cursor]
        cursor += 1
        var identity = PythonObject(Int(item._obj_ptr))
        if py_contains(seen, identity):
            continue
        py_setitem(seen, identity, item)
        var kept = False
        for i in range(len(retained)):  # driver-loop: unchanged or replacement fields independently retain identical owners
            if item is retained[i]:
                kept = True
                break
        if kept:
            continue
        var cls = py_type(item)
        # Connected metric functions are keys retained by the import-time
        # registry; dropping a scorer cannot destroy that function object.
        if cls is context["family_msel_metric_function_type"] and py_contains(context["family_msel_metric_plans"], item):
            continue
        if cls is b.range:
            continue
        native_release_weakrefs(item, release["ownership"])
        if cls is b.list or cls is b.tuple:
            for i in range(len(item)):  # driver-loop: exact native sequence owners
                _ = py_call(todo.append, item[i])
        elif cls is b.set or cls is b.frozenset:
            var members = py_call(b.list, item)
            for i in range(len(members)):  # driver-loop: native set iteration does not hash or compare its elements
                _ = py_call(todo.append, members[i])
        elif cls is b.dict:
            var pairs = py_call(b.list, item.items())
            for i in range(len(pairs)):  # driver-loop: inspect owners without application-key hashing/equality
                _ = py_call(todo.append, pairs[i][0])
                _ = py_call(todo.append, pairs[i][1])
        elif cls is context["family_msel_scorer_class"]:
            var ancestry = py_call(b.type.__dict__["__mro__"].__get__, cls)
            if len(ancestry) != 2 or ancestry[1] is not b.object:
                raise Error("TypeError: native scorer release refuses inherited lifecycle hooks")
            _ = py_call(todo.append, msel_scorer_fields(item, context))
        elif py_type(cls) is b.type and py_contains(context["family_msel_search_layouts"], cls):
            _ = py_call(todo.append, msel_search_fields(item, context))
        elif py_type(cls) is b.type and py_contains(release["release_plans"], cls):
            var plan = msel_estimator_plan(item, context)
            var finalizer = release["release_plans"][cls]["finalizer"]
            if _resolved(cls, PythonObject("__del__"), release["missing"]) is not finalizer:
                raise Error("TypeError: native search refuses changed model finalizers")
            _ = py_call(todo.append, py_call(plan["fields_get"], item, cls))
        else:
            # This existing owner admits native buffers/dtypes, connected
            # module identities and known native archive finalizers; Python
            # finalizers or callback-bearing array owners are refused.
            declared_release(item, release)


def msel_search_update(fields: PythonObject, updates: PythonObject,
                       context: PythonObject) raises:
    var b = native_loaded_module("builtins")
    var names = py_call(b.list, updates)
    var existing = py_call(b.list, fields)
    var retained = py_call(b.list, updates.values())
    var previous = Python.list()
    for i in range(len(existing)):  # driver-loop: preserve owners reachable through unchanged public fields
        var name = existing[i]
        if py_contains(updates, name):
            if fields[name] is not updates[name]:
                _ = py_call(previous.append, fields[name])
        else:
            _ = py_call(retained.append, fields[name])
    if len(previous) > 0:
        _search_release(previous, retained, context)
    for i in range(len(names)):  # driver-loop: all old owner checks finish before the first dictionary write
        py_setitem(fields, names[i], updates[names[i]])


def msel_search_initialize(model: PythonObject, args: PythonObject,
                           context: PythonObject) raises -> PythonObject:
    var plan = msel_search_plan(model, context)
    var fields = msel_search_fields(model, context)
    var names = plan["parameters"]
    if len(args) != len(names):
        raise Error("TypeError: native search constructor argument count disagrees with its schema")
    var updates = Python.dict()
    for k in range(len(names)):  # driver-loop: borrowed constructor values, no clone or conversion callbacks
        py_setitem(updates, names[k], args[k])
    msel_search_update(fields, updates, context)
    return PythonObject(None)


def msel_parameter_values(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    msel_admit(value, context)
    var cls = py_type(value)
    if cls is b.str or cls is b.bytes:
        raise Error("TypeError: parameter values require a native sequence, not a string")
    if cls is context["array_class"]:
        var packed = msel_input(value, "parameter values", context)
        var rank = Int(py=packed.ndim)
        if rank == 0:
            raise Error("TypeError: iteration over a 0-d Array")
        if rank == 1:
            return py_call(b.list, packed._mv)
        var shape = py_slice_from(packed.shape, 1)
        var n = Int(py=packed.shape[0])
        var step = 1
        for k in range(len(shape)):  # driver-loop: native row view extents
            step *= Int(py=shape[k])
        var rows = Python.list()
        for k in range(n):  # driver-loop: parameter row views retain their original packed owner
            var view = py_slice(packed._mv, k * step, (k + 1) * step)
            var row = _array(shape, String(packed.dtype), view, packed._store, packed._pin,
                             context["family_msel_resident_context"], packed)
            _ = py_call(rows.append, row)
        return rows
    if (cls is b.list or cls is b.tuple or cls is b.range or cls is b.set or cls is b.frozenset
            or cls is b.dict or cls is b.memoryview or cls is context["array_type"]
            or cls is py_get(context, "ndarray")):
        # Exact built-in / immutable extension iterator slots only. ndarray
        # iteration yields native scalar/row views, without NumPy Python code.
        return py_call(b.list, value)
    raise Error("TypeError: parameter values require native containers or buffers; Python iterator/distribution callbacks are not supported")
