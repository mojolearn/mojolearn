# SPDX-License-Identifier: Apache-2.0
# The nineteen linear constructors assign existing references only. Public
# arguments are bound here; no Python initializer, signature binder, descriptor
# or caller protocol runs. Reinitialization preserves fitted/user fields and
# preserves an existing numeric_mode when the supplied mode is None, matching
# NumericModeMixin's original post-constructor assignment rule.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_loaded_module import native_loaded_module
from bindings.native_error_class import error_without_class
from bindings.native_instance_item import native_instance_item
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.mode_metadata_native import _fields, _class_fields
from bindings.params_setter_native import _release_admit
from bindings.params_metadata_native import _connected_plan, _resolved
from bindings.classical_attributes import classical_connect_numpy
from bindings.optimizer_public_words import optimizer_python_error, PYTHON_ERROR_PENDING


def _constructor_plan(model: PythonObject, fields: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var cls = py_type(model)
    if py_type(cls) is not b.type:
        raise Error("TypeError: native constructors cannot invoke Python metaclass hooks")
    var plan = _connected_plan(cls, context)
    if plan is None:
        raise Error("TypeError: native linear constructor requires a connected class")
    var ancestry = py_call(b.type.__dict__["__mro__"].__get__, cls)
    var base = plan["mro"]
    var connected_ancestry = py_call(b.type.__dict__["__mro__"].__get__, base[0])
    if len(connected_ancestry) != len(base):
        raise Error("TypeError: native constructor ancestry changed")
    for i in range(len(base)):  # driver-loop: freeze the connected base while admitting passive inherited ancestry
        if connected_ancestry[i] is not base[i]:
            raise Error("TypeError: native constructor ancestry changed")
    for i in range(len(ancestry)):  # driver-loop: every class namespace must use ordinary native type metadata
        if py_type(ancestry[i]) is not b.type:
            raise Error("TypeError: native constructors cannot invoke Python metaclass hooks")
    var layout = plan["layout_hooks"]
    for i in range(len(layout)):  # driver-loop: original base method/storage identities remain frozen
        if py_get(_class_fields(layout[i][0]), layout[i][1]) is not layout[i][2]:
            raise Error("TypeError: native constructor base hooks changed after connection")
    var inherited = plan["inherited_hooks"]
    for i in range(len(inherited)):  # driver-loop: passive subclasses may not replace constructor or lifecycle protocols
        if _resolved(cls, inherited[i][0], context["missing"]) is not inherited[i][1]:
            raise Error("TypeError: native constructor inherited hooks changed")
    var protected = plan["protected"]
    for i in range(len(protected)):  # driver-loop: stored protocol hooks are never product callbacks
        if py_contains(fields, protected[i]):
            raise Error("TypeError: native constructors refuse instance protocol overrides")
    var hooks = plan["field_hooks"]
    for i in range(len(hooks)):  # driver-loop: reject changed parameter descriptors without invoking them
        if py_get(_class_fields(hooks[i][0]), hooks[i][1]) is not hooks[i][2]:
            raise Error("TypeError: native constructor parameter descriptors changed after connection")
    var names = plan["names"]
    for i in range(len(names)):  # driver-loop: subclass data descriptors would intercept original field assignment
        var value = _resolved(cls, names[i], context["missing"])
        if value is context["missing"]:
            continue
        var kind = py_type(value)
        if (_resolved(kind, "__set__", context["missing"]) is not context["missing"]
                or _resolved(kind, "__delete__", context["missing"]) is not context["missing"]
                or (String(names[i]) == "numeric_mode"
                    and _resolved(kind, "__get__", context["missing"]) is not context["missing"])):
            raise Error("TypeError: native constructors refuse parameter descriptor callbacks")
    return plan


def _initialize(model: PythonObject, positional: PythonObject, keywords: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if py_type(positional) is not b.tuple or py_type(keywords) is not b.dict:
        raise Error("TypeError: native constructor arguments require an ordinary tuple and dict")
    if not Bool(py=context["ready"]):
        raise Error("RuntimeError: native linear constructor schemas are not connected")
    var fields = _fields(model)
    var plan = _constructor_plan(model, fields, context)
    var keys = py_call(b.list, keywords.keys())
    for i in range(len(keys)):  # driver-loop: reject user-key hashing/equality before argument lookup
        if py_type(keys[i]) is not b.str:
            raise Error("TypeError: native constructor keyword names must be exact strings")
    var names = plan["names"]
    var allowed = plan["positional"]
    var defaults = plan["defaults"]
    var label = String(plan["label"])
    # Python's old mode wrapper removed numeric_mode before binding the
    # estimator's original signature; it is still keyword-only here.
    var values = Python.dict()
    for i in range(min(len(positional), len(allowed))):  # driver-loop: native positional argument binding
        py_setitem(values, allowed[i], positional[i])
    for i in range(len(keys)):  # driver-loop: duplicate and unexpected keywords before any state mutation
        var name = keys[i]
        if not py_contains(names, name):
            raise Error("TypeError: " + label + "() got an unexpected keyword argument '" + String(name) + "'")
        if py_contains(values, name):
            raise Error("TypeError: " + label + "() got multiple values for argument '" + String(name) + "'")
        py_setitem(values, name, keywords[name])
    if len(positional) > len(allowed):
        var required = 1  # the already bound self still appears in Python's error counts
        for i in range(len(allowed)):  # driver-loop: exact constructor positional arity metadata
            if not py_contains(defaults, allowed[i]):
                required += 1
        var maximum = len(allowed) + 1
        var expected = String(maximum)
        if required != maximum:
            expected = "from " + String(required) + " to " + String(maximum)
        var noun = String(" positional arguments")
        if maximum == 1:
            noun = " positional argument"
        var supplied = String(len(positional) + 1)
        var kwonly = 0
        for i in range(len(keys)):  # driver-loop: numeric_mode was stripped by the original wrapper
            if String(keys[i]) != "numeric_mode" and not py_contains(allowed, keys[i]):
                kwonly += 1
        if kwonly > 0:
            var suffix = String(" keyword-only argument")
            if kwonly != 1:
                suffix += "s"
            supplied += " positional arguments (and " + String(kwonly) + suffix + ")"
        raise Error("TypeError: " + label + "() takes " + expected + noun + " but " + supplied + " were given")
    for i in range(len(names)):  # driver-loop: borrow the import-time defaults without copies/conversions
        var name = names[i]
        if not py_contains(values, name):
            if not py_contains(defaults, name):
                raise Error("TypeError: " + label + "() missing required argument '" + String(name) + "'")
            py_setitem(values, name, defaults[name])
    var write_mode = True
    if values["numeric_mode"] is None:
        if py_contains(fields, "numeric_mode"):
            # Keep instance reads on the supported accessor documented for
            # the retained Mojo instance-dictionary lookup failure.
            py_setitem(values, "numeric_mode", native_instance_item(fields, "numeric_mode"))
            write_mode = False
        elif _resolved(py_type(model), "numeric_mode", context["missing"]) is not context["missing"]:
            # Original hasattr() also observed an ordinary class attribute;
            # None must not introduce a shadowing instance field. Descriptors
            # were already refused without calling their hooks above.
            write_mode = False
    var seen = Python.dict()
    var ownership_connected = False
    for i in range(len(names)):  # driver-loop: all overwritten owners must have callback-free destruction before commit
        var name = names[i]
        if String(name) == "numeric_mode" and not write_mode:
            continue
        if not py_contains(fields, name):
            continue
        var previous = native_instance_item(fields, name)
        if previous is not values[name]:
            if not ownership_connected:
                classical_connect_numpy(context["setters"]["ownership"])
                ownership_connected = True
            _release_admit(previous, context["setters"], seen)
    var order = plan["order"]
    for i in range(len(order)):  # driver-loop: original field assignment order; fitted and unrelated fields remain intact
        if String(order[i]) == "numeric_mode" and not write_mode:
            continue
        py_setitem(fields, order[i], values[order[i]])
    return PythonObject(None)


def linear_constructor_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        if len(args) != 4:
            raise Error("TypeError: native constructor expects model, positional arguments, keywords and context")
        var result = _initialize(args[0], args[1], args[2], args[3])
        return result.steal_data()
    except error:
        if String(error) == PYTHON_ERROR_PENDING:
            return optimizer_python_error(error)
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        if String(error).startswith("AttributeError:"):
            return raise_python_exception(error_without_class(error, "AttributeError"), ExceptionType("PyExc_AttributeError"))
        return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
