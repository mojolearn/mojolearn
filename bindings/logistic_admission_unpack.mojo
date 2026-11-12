# SPDX-License-Identifier: Apache-2.0
# Fixed LogisticRegression metadata; never invoke ParamsMixin, a Python
# constructor, user scalar/protocol hooks, or mutable class descriptors.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.classical_attributes_door import classical_check_layout
from bindings.classical_pandas_door import classical_admit_value
from bindings.sequential_admission_door import _one_of
from bindings.classifier_public_lifecycle_door import classifier_label_list
from bindings.native_owner_lifetime_door import native_release_admit, native_release_weakrefs


def _parameter(name: PythonObject, value: PythonObject, context: PythonObject, configure: Bool = False) raises:
    classical_admit_value(value, context)
    if configure:
        # Constructor, get_params and set_params store and return values
        # verbatim and validate nothing (scikit-learn's convention, its
        # check_do_not_raise_errors_in_init_or_set_params). Admission above
        # already refuses Python hooks; the scalar check runs when a value is
        # consumed (fit, predict, ...).
        return
    var b = native_loaded_module("builtins")
    if String(name) == "class_weight" and _one_of(py_type(value), context["mapping_types"]):
        var keys = py_call(b.list, value)
        for i in range(len(keys)):  # driver-loop: native scalar class labels and scalar weights only
            _parameter(PythonObject("class label"), keys[i], context)
            _parameter(PythonObject("class weight"), value[keys[i]], context)
        return
    var cls = py_type(value)
    if value is not None and cls is not b.str and cls is not b.bool and cls is not b.int and cls is not b.float and not _one_of(cls, context["scalar_types"]):
        raise Error("TypeError: LogisticRegression parameter " + String(name) + " requires native scalar configuration without conversion hooks")


def logistic_storage(model: PythonObject, context: PythonObject) raises -> PythonObject:
    classical_check_layout(model, context)
    var layout = py_get(context["classical_public_layouts"], py_type(model))
    if layout is None:
        raise Error("TypeError: LogisticRegression requires its connected native class")
    return py_call(layout["fields_get"], model, py_type(model))


def logistic_fields(model: PythonObject, context: PythonObject, configure: Bool = False) raises -> PythonObject:
    var fields = logistic_storage(model, context)
    var names = context["lr_parameters"]
    for i in range(len(names)):  # driver-loop: fixed constructor parameter values before comparisons/conversion
        if not py_contains(fields, names[i]):
            raise Error("AttributeError: missing LogisticRegression parameter " + String(names[i]))
        _parameter(names[i], fields[names[i]], context, configure)
    names = context["lr_consumed_fields"]
    for i in range(len(names)):  # driver-loop: consumed fitted state and optional native solver configuration
        if py_contains(fields, names[i]):
            var value = classifier_label_list(fields[names[i]], context) if String(names[i]) == "classes_" else fields[names[i]]
            classical_admit_value(value, context)
    return fields


def logistic_release_admit(value: PythonObject, context: PythonObject) raises:
    if py_type(value) is py_get(context, "classifier_labels_class"):
        # ClassLabels is the fixed slots-only list subclass exposed by the
        # classifier boundary. Its native copier checks the frozen class/MRO;
        # inspect its retained elements and weakrefs without invoking list hooks.
        var labels = classifier_label_list(value, context)
        native_release_weakrefs(value, context)
        native_release_admit(labels, context)
        return
    native_release_admit(value, context)


def logistic_fit_replacements_admit(fields: PythonObject, context: PythonObject) raises:
    var names = Python.tuple("_l1_ratio", "input_copied_", "classes_", "_w", "coef_", "intercept_", "n_iter_", "objective_", "retcode_", "n_features_in_", "feature_names_in_", "_mojolearn_fitted")
    for i in range(len(names)):  # driver-loop: every owner changed by fit and classifier_after_fit, before either mutates state
        if py_contains(fields, names[i]):
            logistic_release_admit(fields[names[i]], context)
