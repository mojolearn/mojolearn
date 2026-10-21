# SPDX-License-Identifier: Apache-2.0
# Construct the standard sklearn tag records through their native slots.
# Dataclass constructors and Python setattr/getattribute never run here.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains


def _tag_record_plan(cls: PythonObject, records: PythonObject, context: PythonObject) raises -> PythonObject:
    # One record type's slots: (name, member descriptor, nested record index
    # or -1, immutable value). The explicit values are the ones the regressor
    # tags always carry: estimator_type "regressor", a required target, and
    # the nested target/input/regressor records; every other field keeps its
    # declared dataclass default. No dataclass constructor runs.
    var b = Python.import_module("builtins")
    if py_type(cls) is not b.type:
        raise Error("TypeError: sklearn tags require standard classes without metaclass hooks")
    var class_fields = py_call(b.type.__dict__["__dict__"].__get__, cls)
    var declared = py_get(class_fields, "__dataclass_fields__")
    if declared is None or py_type(declared) is not b.dict:
        raise Error("TypeError: sklearn tag records must be standard dataclasses")
    var missing = py_get(py_call(b.object.__getattribute__, Python.import_module("dataclasses"), "__dict__"), "MISSING")
    var names = py_call(b.list, declared.keys())
    var slots = Python.list()
    for j in range(len(names)):  # driver-loop: over the record type's declared tag fields
        var name = names[j]
        if py_type(name) is not b.str:
            raise Error("TypeError: sklearn tag field names must be native strings")
        var descriptor = py_get(class_fields, name)
        if py_type(descriptor) is not context["native_member_descriptor"]:
            raise Error("TypeError: sklearn tag fields must be native member descriptors")
        var nested = -1
        var value = PythonObject(None)
        var field = String(name)
        if cls is records[3] and field == "target_tags":
            nested = 0
        elif cls is records[3] and field == "input_tags":
            nested = 1
        elif cls is records[3] and field == "regressor_tags":
            nested = 2
        elif cls is records[3] and field == "estimator_type":
            value = PythonObject("regressor")
        elif cls is records[0] and field == "required":
            value = PythonObject(True)
        else:
            value = declared[name].default
            if value is missing:
                raise Error("TypeError: sklearn tag field has no immutable default: " + field)
            if value is not None and py_type(value) is not b.bool and py_type(value) is not b.str:
                raise Error("TypeError: sklearn tag defaults must be immutable native metadata")
        py_call(slots.append, Python.tuple(name, descriptor, nested, value))
    return Python.tuple(cls, py_call(b.tuple, slots))


def classical_tag_plan(context: PythonObject) raises -> PythonObject:
    """The sklearn tag record layout, read from the loaded sklearn module the
    first time sklearn asks for tags, then kept on the connected context."""
    var plan = context["classical_tag_plan"]
    if plan is not None:
        return plan
    var b = Python.import_module("builtins")
    var module = py_get(Python.import_module("sys").modules, "sklearn.utils._tags")
    if module is None:
        try:
            module = Python.import_module("sklearn.utils._tags")
        except error:
            raise Error("ImportError: " + String(error))
    var fields = py_call(b.object.__getattribute__, module, "__dict__")
    var records = Python.list()
    var classes = Python.tuple("TargetTags", "InputTags", "RegressorTags", "Tags")
    for k in range(len(classes)):  # small-loop(classes: sklearn tag record type names): the four standard records
        var cls = py_get(fields, classes[k])
        if cls is None:
            raise Error("ImportError: sklearn.utils lacks " + String(classes[k]))
        py_call(records.append, cls)
    var built = Python.list()
    for k in range(len(records)):  # small-loop(records: sklearn tag record types): the four standard records
        py_call(built.append, _tag_record_plan(records[k], records, context))
    plan = py_call(b.tuple, built)
    py_setitem(context, "classical_tag_plan", plan)
    return plan


def classical_tags(context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var plan = classical_tag_plan(context)
    if py_type(plan) is b.str:
        raise Error("ImportError: " + String(plan))
    if py_type(plan) is not b.tuple:
        raise Error("TypeError: sklearn tag metadata must be a native tuple")
    var objects = Python.list()
    for k in range(len(plan)):  # driver-loop: over the four connected sklearn metadata record types
        var record = plan[k]
        if py_type(record) is not b.tuple or len(record) != 2:
            raise Error("TypeError: sklearn tag record metadata differs")
        var cls = record[0]
        if py_type(cls) is not b.type:
            raise Error("TypeError: sklearn tags require standard classes without metaclass hooks")
        var fields = py_call(b.type.__dict__["__dict__"].__get__, cls)
        var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
        if len(mro) != 2 or mro[1] is not b.object or py_get(fields, "__del__") is not None:
            raise Error("TypeError: sklearn tags require standard native record lifetime")
        var slots = record[1]
        if py_type(slots) is not b.tuple:
            raise Error("TypeError: sklearn tag slots must be native metadata")
        var value = py_call(b.object.__new__, cls)
        for j in range(len(slots)):  # driver-loop: over connected native member descriptors
            var slot = slots[j]
            if py_type(slot) is not b.tuple or len(slot) != 4 or py_type(slot[0]) is not b.str:
                raise Error("TypeError: sklearn tag slot metadata differs")
            var descriptor = slot[1]
            if py_type(descriptor) is not context["native_member_descriptor"] or py_get(fields, slot[0]) is not descriptor:
                raise Error("TypeError: sklearn tag slot was replaced by a Python descriptor")
            var nested = Int(py=slot[2])
            var content = slot[3]
            if nested >= 0:
                if nested >= k:
                    raise Error("TypeError: sklearn tag reference order differs")
                content = objects[nested]
            elif content is not None and py_type(content) is not b.bool and py_type(content) is not b.str:
                raise Error("TypeError: sklearn tag defaults must be native immutable metadata")
            py_call(descriptor.__set__, value, content)
        py_call(objects.append, value)
    if len(objects) != 4:
        raise Error("TypeError: sklearn tag metadata must contain its four standard records")
    return objects[3]
