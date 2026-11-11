# SPDX-License-Identifier: Apache-2.0
# PCA fitting keeps admitted model references separately from its immutable
# operation setting references. Each consumed scalar is decoded once before
# the solver, at its original refusal point. Native scalar conversion preserves NumPy's public
# parsing/version/warning contract through the shared scalar provider.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_contains, py_get, py_setitem, py_type
from bindings.sequential_admission import sequential_admit_value
from bindings.native_loaded_module import native_loaded_module, NativeModules
from providers.python.pca_fit_scalar_pyapi import PCAFitScalar, pca_fit_scalar_admit
from bindings.pca_descriptor_boundary import pca_connection, pca_mode_name


def pca_fit_fields(state: PythonObject, operation: String, context: PythonObject) raises -> PythonObject:
    var fields = py_call(py_type(Python.dict()).__dict__["copy"], state)
    var owners = Python.list()
    var names = context["pca_parameters"] + context["pca_state"]
    # Row N262: one module view for the field admissions of this call, not
    # one NativeModules per value; a scalar admission (NumPy's scalar
    # protocol) is followed by released(), so the next value re-checks.
    var mods = NativeModules()
    for index in range(len(names)):  # driver-loop: admitted references retained before settings are replaced
        if py_contains(fields, names[index]):
            if index < len(context["pca_parameters"]):
                pca_fit_scalar_admit(fields[names[index]], context)
                mods.released()
            else:
                sequential_admit_value(mods, fields[names[index]], context)
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


def pca_fit_preflight(fields: PythonObject, operation: String, context: PythonObject) raises -> Tuple[String, Bool]:
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
    var tol = PCAFitScalar(fields["tol"], context)
    var tolerance = tol.convert(operation, context, False)
    if Float64(py=tolerance) != 0:
        raise Error("NotImplementedError: mojolearn PCA: tol is ARPACK's convergence tolerance and "
                    + "svd_solver='arpack' is not implemented; leave tol=0.0")
    var whiten = PCAFitScalar(fields["whiten"], context, True, operation)
    return (text, whiten.truth(operation, context))


def pca_fit_entries(fields: PythonObject, context: PythonObject) raises -> Int:
    return pca_connection(pca_mode_name(fields["numeric_mode"]), context)


def pca_fit_host_only(model: PythonObject) raises -> Bool:
    var b = native_loaded_module("builtins")
    var mro = py_call(b.type.__dict__["__mro__"].__get__, py_type(model))
    for index in range(len(mro)):  # driver-loop: admitted native class metadata only
        var fields = py_call(b.type.__dict__["__dict__"].__get__, mro[index])
        if py_get(fields, "_HOST_INFERENCE_ONLY") is not None:
            return True
    return False
