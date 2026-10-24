# SPDX-License-Identifier: Apache-2.0
# Native Signature.bind/apply_defaults for the six registered forest/tree
# constructors. Preserve parameter object identities and defer value checks
# until fit, exactly as the public constructor contract requires.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem


def forest_construct(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var owner = args[0]
    var cls = py_type(model)
    if py_type(owner) is not b.type or py_type(cls) is not b.type:
        raise Error("TypeError: native forest constructors require ordinary class metadata")
    var plan = py_get(context["trees_constructors"], owner)
    if plan is None:
        raise Error("TypeError: the forest constructor is not connected")
    var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
    var found = False
    for i in range(len(mro)):  # driver-loop: native constructor ancestry metadata
        found = found or mro[i] is owner
    if not found:
        raise Error("TypeError: forest constructor requires an instance of its connected class")
    var hooks = plan[2]
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
    for i in range(len(rows)):  # driver-loop: constructor assignments after argument binding succeeds
        py_setitem(fields, rows[i][0], values[rows[i][0]])
    py_setitem(fields, "_has_constructor_parameters", PythonObject(True))
    return PythonObject(None)
