# SPDX-License-Identifier: Apache-2.0
# PCA fitting keeps admitted model references separately from its immutable
# operation setting references. Each consumed scalar is decoded once before
# the solver, at its original refusal point. Native scalar conversion preserves NumPy's public
# parsing/version/warning contract through the shared scalar provider.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_contains, py_get, py_setitem, py_type
from bindings.sequential_admission_door import sequential_admit_value
from bindings.native_loaded_module_door import native_loaded_module, NativeModules
from providers.python.pca_fit_scalar_pyapi_unpack import PCAFitScalar, pca_fit_scalar_admit
from bindings.pca_descriptor_boundary_door import pca_route, pca_mode_name
from core.device_route_dw_decomp import DecompRoute
from bindings.pca_keys_unpack import pca_keys
from bindings.py_balanced import py_item


def pca_fit_preflight(fields: PythonObject, operation: String, context: PythonObject) raises -> Tuple[String, Bool]:
    var nk = pca_keys()
    var b = native_loaded_module("builtins")
    var solver = py_item(fields, nk[].k_svd_solver)
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
    var tol = PCAFitScalar(py_item(fields, nk[].k_tol), context)
    var tolerance = tol.convert(operation, context, False)
    if Float64(py=tolerance) != 0:
        raise Error("NotImplementedError: mojolearn PCA: tol is ARPACK's convergence tolerance and "
                    + "svd_solver='arpack' is not implemented; leave tol=0.0")
    var whiten = PCAFitScalar(py_item(fields, nk[].k_whiten), context, True, operation)
    return (text, whiten.truth(operation, context))


def pca_fit_route(model: PythonObject, fields: PythonObject, context: PythonObject) raises -> DecompRoute:
    """The estimator's resolved route for a fit: `fields` is the fit's
    snapshot, whose `numeric_mode` already is the mode object (the process
    default object when the estimator's is None)."""
    var nk = pca_keys()
    return pca_route(model, fields, py_item(fields, nk[].k_numeric_mode), context)


def pca_fit_host_only(model: PythonObject) raises -> Bool:
    var nk = pca_keys()
    var b = native_loaded_module("builtins")
    var mro = py_call(py_item(b.type.__dict__, nk[].k___mro__).__get__, py_type(model))
    for index in range(len(mro)):  # driver-loop: admitted native class metadata only
        var fields = py_call(py_item(b.type.__dict__, nk[].k___dict__).__get__, mro[index])
        if py_get(fields, nk[].k__HOST_INFERENCE_ONLY) is not None:
            return True
    return False
