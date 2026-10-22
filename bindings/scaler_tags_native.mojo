# SPDX-License-Identifier: Apache-2.0
# scikit-learn's tag records for the scalers (StandardScaler, MinMaxScaler and
# their subclasses), built through the records' native member slots.
#
# Cold contract: scikit-learn calls `__sklearn_tags__`, so the tag module is
# loaded here on first use when the caller has not imported scikit-learn yet
# (API shell work: one module lookup, the same as bindings/classical_tags.mojo).
# `import mojolearn` itself never imports scikit-learn or NumPy, and the
# records returned are always scikit-learn's own dataclass types. No dataclass
# constructor, default factory, setattr or Python tag callback runs.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_contains
from bindings.classical_attributes import _type_fields, _type_mro


def _tags_module() raises -> PythonObject:
    # The loaded module when scikit-learn is already imported; otherwise load
    # it now. scikit-learn absent raises ImportError, as before.
    var module = py_get(Python.import_module("sys").modules, "sklearn.utils._tags")
    if module is None:
        try:
            module = Python.import_module("sklearn.utils._tags")
        except error:
            raise Error("ImportError: " + String(error))
    return module


def _class_metadata(model: PythonObject, name: String, default: PythonObject) raises -> PythonObject:
    # Class attribute lookup through the MRO's own dictionaries (no
    # __getattribute__/__getattr__ hooks, no descriptors invoked).
    var bases = _type_mro(py_type(model))
    for k in range(len(bases)):  # driver-loop: over the estimator class ancestry
        var fields = _type_fields(bases[k])
        if py_contains(fields, name):
            return fields[name]
    return default


def _overrides(value: PythonObject, declared: PythonObject, what: String) raises -> PythonObject:
    # Admission of a `_sklearn_input_tags` / `_sklearn_target_tags` override:
    # an ordinary dict of declared field name -> bool. Anything else is refused
    # before any record is built.
    var b = Python.import_module("builtins")
    if py_type(value) is not b.dict:
        raise Error("TypeError: " + what + " must be a dict of tag names to booleans")
    var names = py_call(b.list, value)
    for k in range(len(names)):  # driver-loop: over the override field names
        var name = names[k]
        if py_type(name) is not b.str:
            raise Error("TypeError: " + what + " names must be strings")
        if py_type(value[name]) is not b.bool:
            raise Error("TypeError: " + what + "[" + String(name) + "] must be a bool")
        if not py_contains(declared, name):
            raise Error("AttributeError: " + what + " names an unknown tag field: " + String(name))
    return value


def _record_class(fields: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var cls = py_get(fields, name)
    if cls is None or py_type(cls) is not b.type:
        raise Error("ImportError: sklearn.utils lacks the standard " + name + " record")
    var own = _type_fields(cls)
    var mro = _type_mro(cls)
    if len(mro) != 2 or mro[1] is not b.object or py_get(own, "__del__") is not None:
        raise Error("TypeError: sklearn tags require standard native record lifetime")
    var declared = py_get(own, "__dataclass_fields__")
    if declared is None or py_type(declared) is not b.dict:
        raise Error("TypeError: sklearn tag records must be standard dataclasses")
    return cls


def _build(cls: PythonObject, values: PythonObject, missing: PythonObject, context: PythonObject) raises -> PythonObject:
    # One record: every declared field from `values` when named there,
    # otherwise its declared immutable default.
    var b = Python.import_module("builtins")
    var own = _type_fields(cls)
    var declared = own["__dataclass_fields__"]
    var names = py_call(b.list, declared)
    var record = py_call(b.object.__new__, cls)
    for j in range(len(names)):  # driver-loop: over the record type's declared tag fields
        var name = names[j]
        if py_type(name) is not b.str:
            raise Error("TypeError: sklearn tag field names must be native strings")
        var descriptor = py_get(own, name)
        if py_type(descriptor) is not context["native_member_descriptor"]:
            raise Error("TypeError: sklearn tag fields must be native member descriptors")
        var content: PythonObject
        if py_contains(values, name):
            content = values[name]
        else:
            content = declared[name].default
            if content is missing:
                raise Error("TypeError: sklearn tag field has no immutable default: " + String(name))
            if content is not None and py_type(content) is not b.bool and py_type(content) is not b.str:
                raise Error("TypeError: sklearn tag defaults must be immutable native metadata")
        _ = py_call(descriptor.__set__, record, content)
    return record


def scaler_tags(model: PythonObject, context: PythonObject) raises -> PythonObject:
    """scikit-learn's `Tags` for a scaler instance: the shared estimator
    answer (estimator type, target/input overrides, requires_fit) with
    float32-preserving transformer tags."""
    var b = Python.import_module("builtins")
    var module = _tags_module()
    if py_type(module) is not context["native_module_type"]:
        raise Error("TypeError: sklearn tags require the ordinary sklearn metadata module")
    var fields = py_fields(module)
    var target_cls = _record_class(fields, "TargetTags", context)
    var input_cls = _record_class(fields, "InputTags", context)
    var transformer_cls = _record_class(fields, "TransformerTags", context)
    var classifier_cls = _record_class(fields, "ClassifierTags", context)
    var regressor_cls = _record_class(fields, "RegressorTags", context)
    var tags_cls = _record_class(fields, "Tags", context)
    var missing = py_get(py_fields(Python.import_module("dataclasses")), "MISSING")

    # Metadata admission, all before any record is built.
    var kind = py_get(py_fields(model), "_estimator_type", _class_metadata(model, "_estimator_type", PythonObject(None)))
    if kind is not None and py_type(kind) is not b.str:
        raise Error("TypeError: _estimator_type must be a string or None")
    var kind_name = String("") if kind is None else String(py=kind)
    var requires_fit = _class_metadata(model, "_sklearn_requires_fit", PythonObject(True))
    if py_type(requires_fit) is not b.bool:
        raise Error("TypeError: _sklearn_requires_fit must be a bool")
    if py_get(py_fields(model), "_sklearn_instance_target_tags") is not None or _class_metadata(model, "_sklearn_instance_target_tags", PythonObject(None)) is not None:
        raise Error("TypeError: scaler tags take _sklearn_target_tags metadata, not an _sklearn_instance_target_tags callback")
    var target_declared = _type_fields(target_cls)["__dataclass_fields__"]
    var input_declared = _type_fields(input_cls)["__dataclass_fields__"]
    var target_overrides = _overrides(_class_metadata(model, "_sklearn_target_tags", Python.dict()), target_declared, "_sklearn_target_tags")
    var input_overrides = _overrides(_class_metadata(model, "_sklearn_input_tags", Python.dict()), input_declared, "_sklearn_input_tags")
    var accepts_sparse = context["sklearn_accepts_sparse"]
    if py_type(accepts_sparse) is not b.frozenset:
        raise Error("TypeError: scaler tag metadata lacks the sparse-input class names")

    # TargetTags: a required target for classifiers and regressors, then the
    # class's own overrides.
    var target_values = Python.dict()
    target_values["required"] = PythonObject(kind_name == "classifier" or kind_name == "regressor")
    _ = py_call(target_values.update, target_overrides)
    # InputTags: the shared sparse-input class names (_sk_validate's
    # input_tag_overrides), then the class's own overrides.
    var input_values = Python.dict()
    var bases = _type_mro(py_type(model))
    for k in range(len(bases)):  # driver-loop: over the estimator class ancestry
        var name = py_call(b.type.__dict__["__name__"].__get__, bases[k])
        if py_contains(accepts_sparse, name):
            input_values["sparse"] = PythonObject(True)
            break
    _ = py_call(input_values.update, input_overrides)
    var transformer_values = Python.dict()
    transformer_values["preserves_dtype"] = Python.list("float32")

    var tag_values = Python.dict()
    tag_values["estimator_type"] = kind
    tag_values["target_tags"] = _build(target_cls, target_values, missing, context)
    tag_values["input_tags"] = _build(input_cls, input_values, missing, context)
    tag_values["transformer_tags"] = _build(transformer_cls, transformer_values, missing, context)
    tag_values["classifier_tags"] = _build(classifier_cls, Python.dict(), missing, context) if kind_name == "classifier" else PythonObject(None)
    tag_values["regressor_tags"] = _build(regressor_cls, Python.dict(), missing, context) if kind_name == "regressor" else PythonObject(None)
    tag_values["requires_fit"] = requires_fit
    return _build(tags_cls, tag_values, missing, context)
