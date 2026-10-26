# SPDX-License-Identifier: Apache-2.0
# Native public constructors for the 77 source-vetted full-reset owners
# (python/mojolearn/_native_full_reset_params.py, proof
# tools/native_full_reset_constructor_schemas.json) and the 52 field-only
# declared prep/decomp and plain owners (python/mojolearn/_native_declared_constructors.py,
# schemas derived by tools/declared_constructor_schemas.py). The constructor event
# schema the full-reset setter already replays is the whole original
# constructor: field writes in source order, repeated writes, fixed constants
# and the NumericModeMixin post-constructor numeric_mode rule. Here the public
# arguments are bound natively with CPython's own argument errors, then the
# events are written into the existing instance dictionary. No Python
# initializer, signature binder, wrapper, descriptor or caller protocol runs.
#
# Differences from the full-reset setter: a constructor keeps every unrelated
# field (re-initialization overwrites only the fields it writes, in place,
# preserving dictionary order), and the numeric_mode rule observes the
# existing instance dictionary as the original `hasattr(self, ...)` did.
# A user subclass that only overrides `__init__` and calls `super().__init__`
# is admitted: its own Python initializer is the caller, never a callee here.
# Subclasses that replace attribute lookup/assignment with Python hooks, or
# add data descriptors over constructor fields, are refused before any write.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_loaded_module import native_loaded_module
from bindings.native_error_class import error_without_class
from bindings.native_instance_item import native_instance_item
from bindings.mode_metadata_native import _fields, _class_fields
from bindings.params_metadata_native import _resolved
from bindings.full_reset_params_native import _release_owned
from bindings.declared_params_native import _release as _declared_release
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.optimizer_public_words import optimizer_python_error, PYTHON_ERROR_PENDING


def _quoted_names(names: PythonObject) raises -> String:
    """CPython's `format_missing` list: 'a' / 'a' and 'b' / 'a', 'b', and 'c'."""
    var b = native_loaded_module("builtins")
    var count = len(names)
    var text = String("")
    for i in range(count):  # driver-loop: over missing constructor parameter names
        if i > 0:
            if count == 2:
                text += " and "
            elif i == count - 1:
                text += ", and "
            else:
                text += ", "
        text += String(py_call(b.repr, names[i]))
    return text


def _missing(label: String, kind: String, names: PythonObject) raises:
    if len(names) == 0:
        return
    var noun = String(" argument")
    if len(names) != 1:
        noun += "s"
    raise Error("TypeError: " + label + "() missing " + String(len(names)) + " required " + kind
                + noun + ": " + _quoted_names(names))


def _bind(positional: PythonObject, keywords: PythonObject, plan: PythonObject) raises -> PythonObject:
    """The public signature's bound arguments, raising the original
    constructor's exact argument errors in CPython's order."""
    var b = native_loaded_module("builtins")
    var keys = py_call(b.list, keywords.keys())
    for i in range(len(keys)):  # driver-loop: reject user-key hashing/equality before argument lookup
        if py_type(keys[i]) is not b.str:
            raise Error("TypeError: native constructor keyword names must be exact strings")
    var names = plan["names"]
    var allowed = plan["positional"]
    var defaults = plan["defaults"]
    var label = String(plan["label"])
    var values = Python.dict()
    for i in range(min(len(positional), len(allowed))):  # driver-loop: native positional argument binding
        py_setitem(values, allowed[i], positional[i])
    for i in range(len(keys)):  # driver-loop: unexpected and duplicate keywords before any state mutation
        var name = keys[i]
        if not py_contains(names, name):
            raise Error("TypeError: " + label + "() got an unexpected keyword argument '" + String(name) + "'")
        if py_contains(values, name):
            raise Error("TypeError: " + label + "() got multiple values for argument '" + String(name) + "'")
        py_setitem(values, name, native_instance_item(keywords, name))
    if len(positional) > len(allowed):
        var required = 1  # the bound self still appears in Python's error counts
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
        var counted = plan["kwonly"]
        for i in range(len(keys)):  # driver-loop: keyword-only parameters the original initializer received
            if py_contains(counted, keys[i]):
                kwonly += 1
        if kwonly > 0:
            var suffix = String(" keyword-only argument")
            if kwonly != 1:
                suffix += "s"
            supplied += " positional arguments (and " + String(kwonly) + suffix + ")"
        raise Error("TypeError: " + label + "() takes " + expected + noun + " but " + supplied + " were given")
    var absent = Python.list()
    var required_positional = plan["required_positional"]
    for i in range(len(required_positional)):  # driver-loop: CPython reports missing positionals first
        if not py_contains(values, required_positional[i]):
            _ = py_call(absent.append, required_positional[i])
    _missing(label, "positional", absent)
    var required_keywords = plan["required_kwonly"]
    for i in range(len(required_keywords)):  # driver-loop: then missing keyword-only parameters
        if not py_contains(values, required_keywords[i]):
            _ = py_call(absent.append, required_keywords[i])
    _missing(label, "keyword-only", absent)
    for i in range(len(names)):  # driver-loop: borrow the import-time defaults without copies/conversions
        var name = names[i]
        if not py_contains(values, name):
            py_setitem(values, name, native_instance_item(defaults, name))
    return values


def _admit(model: PythonObject, owner: PythonObject, plan: PythonObject, context: PythonObject) raises:
    var b = native_loaded_module("builtins")
    var missing = context["missing"]
    var cls = py_type(model)
    if py_type(cls) is not b.type:
        raise Error("TypeError: native constructors cannot invoke Python metaclass hooks")
    var ancestry = py_call(b.type.__dict__["__mro__"].__get__, cls)
    var owned = False
    for i in range(len(ancestry)):  # driver-loop: ordinary class metadata and the connected owner
        if py_type(ancestry[i]) is not b.type:
            raise Error("TypeError: native constructors cannot invoke Python metaclass hooks")
        if ancestry[i] is owner:
            owned = True
    if not owned:
        raise Error("TypeError: " + String(plan["label"]) + "() requires a "
                    + String(owner.__name__) + " instance")
    var hooks = plan["constructor_hooks"]
    var admitted = context["constructor_hook_types"]
    for i in range(len(hooks)):  # driver-loop: instance lookup/assignment protocols stay native
        var current = _resolved(cls, hooks[i][0], missing)
        if current is hooks[i][1]:
            continue
        var trusted = False
        if current is not missing:
            var kind = py_type(current)
            for j in range(len(admitted)):  # driver-loop: fixed native descriptor types, by identity
                if kind is admitted[j]:
                    trusted = True
        if not trusted:
            raise Error("TypeError: native constructors refuse Python attribute protocol overrides")
    var field_hooks = plan["field_hooks"]
    for i in range(len(field_hooks)):  # driver-loop: connected bases keep their field metadata
        if py_get(_class_fields(field_hooks[i][0]), field_hooks[i][1]) is not field_hooks[i][2]:
            raise Error("TypeError: native constructor parameter descriptors changed after connection")
    var names = plan["field_names"]
    var rule = plan["rule_fields"]
    for i in range(len(names)):  # driver-loop: subclass descriptors would intercept the original assignment
        var value = _resolved(cls, names[i], missing)
        if value is missing:
            continue
        var kind = py_type(value)
        if (_resolved(kind, "__set__", missing) is not missing
                or _resolved(kind, "__delete__", missing) is not missing
                or (py_contains(rule, names[i]) and _resolved(kind, "__get__", missing) is not missing)):
            raise Error("TypeError: native constructors refuse parameter descriptor callbacks")


def _construct(model: PythonObject, positional: PythonObject, keywords: PythonObject,
               context: PythonObject, owner: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if py_type(positional) is not b.tuple or py_type(keywords) is not b.dict:
        raise Error("TypeError: native constructor arguments require an ordinary tuple and dict")
    if not Bool(py=context["ready"]):
        raise Error("RuntimeError: native full-reset constructor schemas are not connected")
    var plan = py_get(context["plans"], owner)
    if plan is None:
        raise Error("TypeError: native constructor requires a connected class")
    var values = _bind(positional, keywords, plan)
    _admit(model, owner, plan, context)
    var missing = context["missing"]
    var cls = py_type(model)
    var fields = _fields(model)
    # Replay the original writes. The NumericModeMixin rule writes a None mode
    # only when hasattr(self, name) was false: no instance field (old or
    # written earlier in this constructor) and no class attribute.
    var events = plan["events"]
    var written = Python.dict()
    var targets = Python.list()
    var sources = Python.list()
    for i in range(len(events)):  # driver-loop: original constructor assignment and wrapper order
        var event = events[i]
        var name = event[0]
        var value = event[2]
        if String(event[1]) == "parameter":
            value = native_instance_item(values, event[2])
        if Bool(py=event[3]) and value is None:
            if (py_contains(written, name) or py_contains(fields, name)
                    or _resolved(cls, name, missing) is not missing):
                continue
        py_setitem(written, name, value)
        _ = py_call(targets.append, name)
        _ = py_call(sources.append, value)
    # Re-initialization drops the overwritten owners: admit their release
    # before the first write, exactly as the full-reset setter does.
    var previous = Python.list()
    var retained = py_call(b.list, written.values())
    var existing = py_call(b.list, fields.keys())
    for i in range(len(existing)):  # driver-loop: fields this constructor overwrites or keeps
        var name = existing[i]
        var old = native_instance_item(fields, name)
        if py_contains(written, name):
            if old is not native_instance_item(written, name):
                _ = py_call(previous.append, old)
        else:
            _ = py_call(retained.append, old)
    if len(previous) > 0:
        if py_contains(context, "release_declared"):
            # The declared families (_native_declared_constructors.py) release
            # an overwritten value under their shallow setter's admission.
            for i in range(len(previous)):  # driver-loop: each overwritten constructor field
                _declared_release(previous[i], context["declared"])
        else:
            _release_owned(previous, retained, context)
    for i in range(len(targets)):  # driver-loop: original field assignment order into the same dictionary
        py_setitem(fields, targets[i], sources[i])
    return PythonObject(None)


def full_reset_constructor_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        if len(args) != 5:
            raise Error("TypeError: native constructor expects model, positional arguments, keywords, context and owner")
        var result = _construct(args[0], args[1], args[2], args[3], args[4])
        return result.steal_data()
    except error:
        if String(error) == PYTHON_ERROR_PENDING:
            return optimizer_python_error(error)
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        if String(error).startswith("AttributeError:"):
            return raise_python_exception(error_without_class(error, "AttributeError"), ExceptionType("PyExc_AttributeError"))
        return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
