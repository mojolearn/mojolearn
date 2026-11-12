# SPDX-License-Identifier: Apache-2.0
# Admission/parameter updates for exact built-in estimator schemas whose
# constructors only store the supplied parameter objects. Numerical work
# continues through each estimator's existing native owner.
# The PACK half of the former `bindings/msel_fixed_owner.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/msel_fixed_owner_unpack.mojo`.
from bindings.native_loaded_module_door import native_loaded_module
from bindings.native_model_records import (
    RECORDS_PREPARED, ContentRecord, content_fields, content_object, records_stamped_get, records_stamped_put,
)
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.classical_attributes_door import classical_check_layout, _type_fields
from bindings.classical_pandas_door import classical_admit_value
from bindings.native_typed_context_pack import native_typed_context
from bindings.classifier_public_lifecycle_door import classifier_label_list
from bindings.sequential_admission_door import _one_of
from bindings.native_model_records import bump_model_generation


from bindings.msel_fixed_owner_unpack import (
    _parameter,
    fixed_owner_fields,
)


def fixed_owner_set_params(model: PythonObject, values: PythonObject,
                           context: PythonObject) raises -> PythonObject:
    # Generation word (bindings/native_model_records.mojo, lane ne1-e-records):
    # every stamped record of this object misses from here on.
    _ = bump_model_generation(model)
    var b = native_loaded_module("builtins")
    var fields = fixed_owner_fields(model, context)
    var plan = context["cv_fixed_owner"]
    if py_type(values) is not b.dict:
        raise Error("TypeError: native parameter updates require an ordinary dictionary")
    var names = plan["parameters"]
    var keys = py_call(b.list, values)
    for i in range(len(keys)):  # driver-loop: validate all names before hashing or formatting
        if py_type(keys[i]) is not b.str:
            raise Error("TypeError: native parameter names must be exact strings")
    if len(keys) == 0:
        return model
    if Bool(py=plan["reset"]):
        var unknown = Python.list()
        for i in range(len(keys)):  # driver-loop: adapter rejects unknown names before any mutation
            if not py_contains(names, keys[i]):
                _ = py_call(unknown.append, keys[i])
        if len(unknown):
            raise Error("Invalid " + String(plan["name"]) + " parameters: " + String(py_call(b.repr, py_call(b.sorted, unknown))))
        var replacement = Python.dict()
        for i in range(len(names)):  # driver-loop: constructor reconstruction keeps only parameters
            var value = values[names[i]] if py_contains(values, names[i]) else fields[names[i]]
            _parameter(names[i], value, context)
            py_setitem(replacement, names[i], value)
        _ = py_call(fields.clear)
        _ = py_call(fields.update, replacement)
    else:
        for i in range(len(keys)):  # driver-loop: preserve _PrepBase's ordered parameter assignments
            if not py_contains(names, keys[i]):
                raise Error("mojolearn: invalid parameter " + String(py_call(b.repr, keys[i])) + " for " + String(plan["name"]))
            _parameter(keys[i], values[keys[i]], context)
            py_setitem(fields, keys[i], values[keys[i]])
    return model
