# SPDX-License-Identifier: Apache-2.0
# PCA fitting keeps admitted model references separately from its immutable
# operation setting references. Each consumed scalar is decoded once before
# the solver, at its original refusal point. Native scalar conversion preserves NumPy's public
# parsing/version/warning contract through the shared scalar provider.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_contains, py_get, py_setitem, py_type
from bindings.sequential_admission import sequential_admit_value
from bindings.native_loaded_module import native_loaded_module
from bindings.training_scalar_native import training_scalar_number
from providers.python.pca_model_pyapi import _native_number, _entry_dictionary


def pca_fit_fields(state: PythonObject, operation: String, context: PythonObject) raises -> PythonObject:
    var fields = py_call(py_type(Python.dict()).__dict__["copy"], state)
    var owners = Python.list()
    var names = context["pca_parameters"] + context["pca_state"]
    for index in range(len(names)):  # driver-loop: admitted references retained before settings are replaced
        if py_contains(fields, names[index]):
            sequential_admit_value(fields[names[index]], context)
            _ = py_call(owners.append, fields[names[index]])
    var b = native_loaded_module("builtins")
    var mode = py_get(fields, "numeric_mode")
    if mode is None:
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    if py_type(mode) is not b.str:
        raise Error("TypeError: numeric_mode must be identical, fast or None")
    py_setitem(fields, "numeric_mode", mode)
    py_setitem(fields, "_pca_fit_owners", owners)
    return fields


def pca_fit_preflight(fields: PythonObject, operation: String, context: PythonObject) raises -> String:
    var b = native_loaded_module("builtins")
    var solver = fields["svd_solver"]
    var text = String(solver) if py_type(solver) is b.str else String("")
    if text != "auto" and text != "covariance_eigh" and text != "jacobi" and text != "full" and text != "randomized":
        # Invalid scalar spellings keep the established solver refusal. Do
        # not repr containers/arrays whose contained objects could run hooks.
        var cls = py_type(solver)
        if solver is not None and cls is not b.str and cls is not b.int and cls is not b.float and cls is not b.bool and cls is not b.bytes:
            raise Error("TypeError: PCA svd_solver requires a native string without Python representation hooks")
        raise Error("NotImplementedError: mojolearn PCA: svd_solver=" + String(py_call(b.repr, solver))
                    + " is not implemented; this class runs ('auto', 'covariance_eigh', 'jacobi', 'full', "
                    + "'randomized'). 'arpack' is scikit-learn's implicitly restarted Lanczos, a third "
                    + "algorithm; accepting the name and running an exact arm would be a silent substitution, "
                    + "which is why this raises (decomposition/NOT_IMPLEMENTED.tsv)")
    var scalar = context["pca_fit_scalar_context"]
    py_setitem(fields, "tol", training_scalar_number(fields["tol"], operation, scalar, False))
    if Float64(py=fields["tol"]) != 0:
        raise Error("NotImplementedError: mojolearn PCA: tol is ARPACK's convergence tolerance and "
                    + "svd_solver='arpack' is not implemented; leave tol=0.0")
    py_setitem(fields, "whiten", _native_number(fields["whiten"], context, True))
    return text


def pca_fit_entries(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    var mode = py_call(py_call(fields["numeric_mode"].strip).lower)
    if String(mode) != "identical" and String(mode) != "fast":
        raise Error("numeric_mode must be identical, fast or None")
    var bindings = _entry_dictionary(context["pca_bindings"])
    var selected = bindings[mode]
    if py_type(selected) is py_type(PythonObject("")):
        raise Error("ImportError: " + String(selected))
    var entries = _entry_dictionary(selected)
    var records = context["pca_fit_connections"]
    if py_type(records) is not py_type(Python.tuple()):
        raise Error("TypeError: PCA fit connections require their import-time tuple")
    var connected = PythonObject(None)
    for index in range(len(records)):  # driver-loop: fixed import-connected numeric modes
        var record = records[index]
        if py_type(record) is not py_type(Python.tuple()) or len(record) != 2 or py_type(record[0]) is not py_type(PythonObject("")) or py_type(record[1]) is not py_type(Python.tuple()) or len(record[1]) != 7:
            raise Error("TypeError: PCA fit connections require native mode/entry records")
        if String(record[0]) == String(mode):
            connected = record[1]
    if connected is None:
        raise Error("TypeError: PCA fit mode was not connected at import")
    var names = Python.tuple("estimators_vendor", "estimators_numeric_mode", "pca_fit_owned", "pca_fit_randomized",
                             "pca_fit_full", "pca_whiten_transform", "pca_whiten_inverse_transform")
    var builtin_type = py_type(native_loaded_module("builtins").len)
    for index in range(7):  # small-loop(seven consumed compiled entry identities)
        var entry = py_get(entries, names[index])
        if index >= 2 and entry is None:
            continue
        var registration = connected[index]
        if py_type(registration) is not py_type(Python.tuple()) or len(registration) != 3:
            raise Error("TypeError: PCA fit requires its import-connected entry record")
        if entry is None or py_type(entry) is not builtin_type or entry is not registration[0]:
            raise Error("TypeError: PCA fit requires its connected compiled entry " + String(names[index]))
        if entry.__self__ is not registration[1] or py_type(registration[2]) is not py_type(PythonObject("")) or String(entry.__name__) != String(registration[2]):
            raise Error("TypeError: PCA fit compiled entry metadata changed after import")
    if String(py_call(entries["estimators_vendor"])) != String(context["vendor"]):
        raise Error("RuntimeError: PCA native vendor differs from selected execution")
    var numeric = Int(py=py_call(entries["estimators_numeric_mode"]))
    if numeric != (1 if String(mode) == "identical" else 0):
        raise Error("RuntimeError: PCA native mode differs from selected execution")
    py_setitem(entries, "_pca_numeric_mode", numeric)
    return entries


def pca_fit_host_only(model: PythonObject) raises -> Bool:
    var b = native_loaded_module("builtins")
    var mro = py_call(b.type.__dict__["__mro__"].__get__, py_type(model))
    for index in range(len(mro)):  # driver-loop: admitted native class metadata only
        var fields = py_call(b.type.__dict__["__dict__"].__get__, mro[index])
        if py_get(fields, "_HOST_INFERENCE_ONLY") is not None:
            return True
    return False
