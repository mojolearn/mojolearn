# SPDX-License-Identifier: Apache-2.0
# sklearn's transformer tag records assembled through native member slots.
# The caller supplies sklearn by importing it; product calls never import
# optional Python modules or invoke dataclass constructors/default factories.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains
from bindings.classical_attributes import _type_fields, _type_mro


def _metadata(model: PythonObject, name: String, default: PythonObject) raises -> PythonObject:
    var bases = _type_mro(py_type(model))
    for k in range(len(bases)):  # driver-loop: native estimator class metadata ancestry
        var fields = _type_fields(bases[k])
        if py_contains(fields,name):
            return fields[name]
    return default


def _tag_overrides(value: PythonObject) raises:
    var b = Python.import_module("builtins")
    if py_type(value) is not b.dict:
        raise Error("TypeError: sklearn tag overrides require an ordinary dictionary of native booleans")
    var names = py_call(b.list,value)
    for k in range(len(names)):  # driver-loop: explicit tag metadata override names
        if py_type(names[k]) is not b.str or py_type(value[names[k]]) is not b.bool:
            raise Error("TypeError: sklearn tag overrides require native string names and boolean values")


def scaler_tags(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var modules = context["modules"]
    var module = py_get(modules,"sklearn.utils._tags")
    if module is None:
        # Importing sklearn executes its Python modules. It belongs in the
        # application's integration setup, before a native product operation.
        raise Error("ImportError: import sklearn.utils before requesting sklearn tags; native operations do not import Python integrations")
    if py_type(module) is not context["native_module_type"]:
        raise Error("TypeError: sklearn tags require the ordinary sklearn metadata module")
    var tag_fields = py_fields(module)
    var dataclasses = py_get(modules,"dataclasses")
    if dataclasses is None or py_type(dataclasses) is not context["native_module_type"]:
        raise Error("TypeError: sklearn tag dataclass metadata is unavailable")
    var missing = py_get(py_fields(dataclasses),"MISSING")
    var input_overrides = _metadata(model,"_sklearn_input_tags",Python.dict())
    var target_overrides = _metadata(model,"_sklearn_target_tags",Python.dict())
    _tag_overrides(input_overrides)
    _tag_overrides(target_overrides)
    if _metadata(model,"_sklearn_instance_target_tags",PythonObject(None)) is not None:
        raise Error("TypeError: Python tag callbacks cannot execute; use native _sklearn_target_tags metadata")
    var requires_fit = _metadata(model,"_sklearn_requires_fit",PythonObject(True))
    if py_type(requires_fit) is not b.bool:
        raise Error("TypeError: sklearn requires_fit must be a native boolean")
    var names = Python.tuple("TargetTags","InputTags","TransformerTags","Tags")
    var objects = Python.list()
    for k in range(4):  # small-loop(records: four standard transformer metadata records)
        var cls = py_get(tag_fields,names[k])
        if cls is None or py_type(cls) is not b.type:
            raise Error("TypeError: sklearn tag records must retain their ordinary native metaclass")
        var fields = _type_fields(cls)
        var bases = _type_mro(cls)
        if len(bases) != 2 or bases[1] is not b.object or py_get(fields,"__del__") is not None:
            raise Error("TypeError: sklearn tag records must retain standard native record lifetime")
        var declared = py_get(fields,"__dataclass_fields__")
        if declared is None or py_type(declared) is not b.dict:
            raise Error("TypeError: sklearn tag records require ordinary dataclass field metadata")
        if k < 2:
            var overrides = target_overrides if k == 0 else input_overrides
            var overridden = py_call(b.list,overrides)
            for j in range(len(overridden)):  # driver-loop: explicitly provided tag field names
                if not py_contains(declared,overridden[j]):
                    raise Error("AttributeError: sklearn tag record has no field " + String(overridden[j]))
        var slots = py_call(b.list,declared)
        var value = py_call(b.object.__new__,cls)
        for j in range(len(slots)):  # driver-loop: native member metadata for each tag record
            var name = slots[j]
            if py_type(name) is not b.str:
                raise Error("TypeError: sklearn tag field names must be native strings")
            var descriptor = py_get(fields,name)
            if py_type(descriptor) is not context["native_member_descriptor"]:
                raise Error("TypeError: sklearn tag fields must remain native member descriptors")
            var field = String(name)
            var content = PythonObject(None)
            var nested = False
            if k == 3 and field == "target_tags":
                content = objects[0]
                nested = True
            elif k == 3 and field == "input_tags":
                content = objects[1]
                nested = True
            elif k == 3 and field == "transformer_tags":
                content = objects[2]
                nested = True
            elif k == 3 and field == "estimator_type":
                content = PythonObject(None)
            elif k == 3 and field == "requires_fit":
                content = requires_fit
            elif k == 0 and field == "required":
                content = py_get(target_overrides,name,PythonObject(False))
            elif k == 2 and field == "preserves_dtype":
                # _ScalerProtocol fixes float32, irrespective of the generic
                # mixin's preserves_dtype default or subclass metadata.
                content = Python.list("float32")
                nested = True
            elif k == 0 and py_contains(target_overrides,name):
                content = target_overrides[name]
            elif k == 1 and py_contains(input_overrides,name):
                content = input_overrides[name]
            else:
                var declaration = declared[name]
                if py_type(declaration) is not py_get(py_fields(dataclasses),"Field"):
                    raise Error("TypeError: sklearn tag fields require standard native dataclass field storage")
                # Field.default is an admitted native member slot, not a
                # property or a factory called back into Python.
                var default_slot = py_get(_type_fields(py_type(declaration)),"default")
                if py_type(default_slot) is not context["native_member_descriptor"]:
                    raise Error("TypeError: dataclass defaults require native member descriptors")
                content = py_call(default_slot.__get__,declaration)
                if content is missing:
                    raise Error("TypeError: sklearn tag field lacks native default metadata: " + field)
            if not nested and content is not None and py_type(content) is not b.bool and py_type(content) is not b.str:
                raise Error("TypeError: sklearn tag defaults must remain native immutable metadata")
            _ = py_call(descriptor.__set__,value,content)
        py_call(objects.append,value)
    return objects[3]
