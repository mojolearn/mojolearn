# SPDX-License-Identifier: Apache-2.0
# Native Signature.bind/apply_defaults shared by registered tree
# constructors. Preserve parameter object identities and defer value checks
# until fit, exactly as the public constructor contract requires.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.mode_metadata_native import _fields
from bindings.params_metadata_native import _resolved
from bindings.trees_public_state import _release_tree_values


def tree_parameter_storage(model: PythonObject, names: PythonObject) raises:
    for i in range(len(names)):  # driver-loop: refuse parameter setters before direct constructor storage writes
        var declared = _resolved(py_type(model), names[i], PythonObject(None))
        if declared is not None:
            var cls = py_type(declared)
            if _resolved(cls, "__set__", PythonObject(None)) is not None or _resolved(cls, "__delete__", PythonObject(None)) is not None:
                raise Error("TypeError: native tree lifecycle requires ordinary parameter storage, not descriptor setters")


def tree_store_parameters(fields: PythonObject, values: PythonObject,
                          context: PythonObject, reset: Bool) raises:
    var b = Python.import_module("builtins")
    var previous = py_call(b.dict, fields)
    var replacement = Python.dict() if reset else py_call(b.dict, previous)
    _ = py_call(replacement.update, values)
    _release_tree_values(py_call(b.list, previous.values()), py_call(b.list, replacement.values()), context)
    if reset:
        _ = py_call(fields.clear)
    _ = py_call(fields.update, replacement)
    _ = previous


def tree_constructor_values(model: PythonObject, args: PythonObject, constructors: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var owner = args[0]
    var cls = py_type(model)
    if py_type(owner) is not b.type or py_type(cls) is not b.type:
        raise Error("TypeError: native forest constructors require ordinary class metadata")
    var plan = py_get(constructors, owner)
    if plan is None:
        raise Error("TypeError: the forest constructor is not connected")
    var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
    var found = False
    for i in range(len(mro)):  # driver-loop: native constructor ancestry metadata
        found = found or mro[i] is owner
    if not found:
        raise Error("TypeError: forest constructor requires an instance of its connected class")
    # A registered decision-tree constructor may call its registered forest
    # base initializer. Validate the actual class's connected native slots;
    # its descriptor intentionally differs from the base's descriptor.
    # Unregistered subclasses still inherit the base's strict hook identities.
    var actual_plan = py_get(constructors, cls)
    var hooks = plan[2] if actual_plan is None else actual_plan[2]
    for i in range(len(hooks)):  # driver-loop: fixed storage hook identities before native dictionary access
        var actual = PythonObject(None)
        for j in range(len(mro)):  # driver-loop: inherited native storage metadata
            var namespace = py_call(b.type.__dict__["__dict__"].__get__, mro[j])
            if py_contains(namespace, hooks[i][0]):
                actual = namespace[hooks[i][0]]
                break
        if actual is not hooks[i][1]:
            raise Error("TypeError: native forest constructors cannot invoke modified Python storage hooks")
    var positional = args[1]
    var keywords = args[2]
    if py_type(positional) is not b.tuple or py_type(keywords) is not b.dict:
        raise Error("TypeError: native forest constructor arguments require ordinary tuple/dict storage")
    var names = py_call(b.list, keywords)
    for i in range(len(names)):  # driver-loop: validate keyword storage before hashing/comparing its keys
        if py_type(names[i]) is not b.str:
            raise Error("TypeError: native forest parameter names must be exact strings")
        if String(names[i]) == "self":
            raise Error("TypeError: multiple values for argument 'self'")
    var rows = plan[0]
    var values = Python.dict()
    var consumed = 0
    for i in range(len(rows)):  # driver-loop: fixed public signature, storing each supplied object unchanged
        var row = rows[i]
        var name = row[0]
        var kind = Int(py=row[1])
        var has_keyword = py_contains(keywords, name)
        if kind != 3 and consumed < len(positional):
            if has_keyword and kind != 0:
                raise Error("TypeError: multiple values for argument '" + String(name) + "'")
            py_setitem(values, name, positional[consumed])
            consumed += 1
        elif has_keyword:
            if kind == 0:
                raise Error("TypeError: '" + String(name) + "' parameter is positional only, but was passed as a keyword")
            py_setitem(values, name, keywords[name])
        elif Bool(py=row[2]):
            raise Error("TypeError: missing a required argument: '" + String(name) + "'")
        else:
            py_setitem(values, name, row[3])
    if consumed != len(positional):
        raise Error("TypeError: too many positional arguments")
    for i in range(len(names)):  # driver-loop: caller keyword names, never their values or protocols
        if not py_contains(values, names[i]):
            raise Error("TypeError: got an unexpected keyword argument '" + String(names[i]) + "'")
    var fields = py_call(plan[1], model, cls)
    tree_parameter_storage(model, py_call(b.list, values))
    return Python.tuple(fields, values, rows)


def forest_construct(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _fields(model)
    var prepared = tree_constructor_values(model, args, context["trees_constructors"])
    if fields is not prepared[0]:
        raise Error("TypeError: native forest constructor requires stable instance storage")
    var values = prepared[1]
    py_setitem(values, "_has_constructor_parameters", PythonObject(True))
    tree_store_parameters(fields, values, context, False)
    return PythonObject(None)
