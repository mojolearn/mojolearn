# SPDX-License-Identifier: Apache-2.0
# Construct the standard sklearn tag records through their native slots.
# Dataclass constructors and Python setattr/getattribute never run here.
from std.python import Python, PythonObject


def classical_tags(context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var plan = context["classical_tag_plan"]
    if b.type(plan) is b.str:
        raise Error("ImportError: " + String(plan))
    if b.type(plan) is not b.tuple:
        raise Error("TypeError: sklearn tag metadata must be a native tuple")
    var objects = Python.list()
    for k in range(len(plan)):  # driver-loop: over the four connected sklearn metadata record types
        var record = plan[k]
        if b.type(record) is not b.tuple or len(record) != 2:
            raise Error("TypeError: sklearn tag record metadata differs")
        var cls = record[0]
        if b.type(cls) is not b.type:
            raise Error("TypeError: sklearn tags require standard classes without metaclass hooks")
        var fields = b.type.__dict__["__dict__"].__get__(cls)
        var mro = b.type.__dict__["__mro__"].__get__(cls)
        if len(mro) != 2 or mro[1] is not b.object or fields.get("__del__") is not None:
            raise Error("TypeError: sklearn tags require standard native record lifetime")
        var slots = record[1]
        if b.type(slots) is not b.tuple:
            raise Error("TypeError: sklearn tag slots must be native metadata")
        var value = b.object.__new__(cls)
        for j in range(len(slots)):  # driver-loop: over connected native member descriptors
            var slot = slots[j]
            if b.type(slot) is not b.tuple or len(slot) != 4 or b.type(slot[0]) is not b.str:
                raise Error("TypeError: sklearn tag slot metadata differs")
            var descriptor = slot[1]
            if b.type(descriptor) is not context["native_member_descriptor"] or fields.get(slot[0]) is not descriptor:
                raise Error("TypeError: sklearn tag slot was replaced by a Python descriptor")
            var nested = Int(py=slot[2])
            var content = slot[3]
            if nested >= 0:
                if nested >= k:
                    raise Error("TypeError: sklearn tag reference order differs")
                content = objects[nested]
            elif content is not None and b.type(content) is not b.bool and b.type(content) is not b.str:
                raise Error("TypeError: sklearn tag defaults must be native immutable metadata")
            descriptor.__set__(value, content)
        objects.append(value)
    if len(objects) != 4:
        raise Error("TypeError: sklearn tag metadata must contain its four standard records")
    return objects[3]
