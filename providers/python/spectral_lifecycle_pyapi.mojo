# SPDX-License-Identifier: Apache-2.0
# Whole estimator metadata/lifecycle boundary for the spectral public methods.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.mode_metadata_native import _fields, _class_fields
from bindings.classical_attributes import _type_mro
from bindings.classical_pandas import classical_admit_value
from bindings.classical_arrays import classical_kind
from bindings.classical_validate import classical_view, classical_estimator_name, RESHAPE_HINT
from bindings.host_inference_scalars import host_inference_scalar
from bindings.native_owner_lifetime import native_release_admit
from bindings.family_raw_attr import family_raw_get, family_raw_get_or
from providers.python.classical_family_admission_pyapi import classical_family_admit
from providers.python.array_metadata_pyapi import array_shape_once


def spectral_admit(model: PythonObject, clustering: Bool, replacing: Bool,
                   context: PythonObject) raises -> PythonObject:
    var fields = classical_family_admit(model)
    var b = native_loaded_module("builtins")
    var names = context["spectral_clustering_parameters"] if clustering else context["spectral_embedding_parameters"]
    for i in range(len(names)):  # driver-loop: fixed consumed scalar configuration before comparisons/hash/conversion
        var value = family_raw_get_or(model, names[i], PythonObject(None)) if String(names[i]) == "numeric_mode" else family_raw_get(model, names[i])
        if String(names[i]) == "kernel_params":
            # The algorithm refuses any non-None kernel_params; admission
            # prevents its diagnostic from traversing custom state.
            classical_admit_value(value, context)
        else:
            host_inference_scalar(value, context, True)
    host_inference_scalar(family_raw_get_or(model, "numeric_mode", PythonObject(None)), context, True)
    var scalars = Python.tuple("_n_clusters", "_n_neighbors", "_n_init", "_eigen_tol", "_gamma", "_seed", "_n_components", "n_features_in_", "input_copied_", "n_components_", "n_iter_assign_", "gamma_", "n_neighbors_", "_mojolearn_fitted", "_mojolearn_fitting")
    var arrays = Python.tuple("_pd_eigenvalues", "_pd_eigenvectors", "_pd_diag", "_pd_centroids", "_fit_X", "affinity_matrix_", "embedding_", "labels_", "feature_names_in_")
    for i in range(len(scalars)):  # driver-loop: consumed learned scalar state
        if py_contains(fields, scalars[i]):
            host_inference_scalar(fields[scalars[i]], context, True)
            if replacing:
                native_release_admit(fields[scalars[i]], context)
    for i in range(len(arrays)):  # driver-loop: retained fitted buffers, admitted before removal or replacement
        if py_contains(fields, arrays[i]):
            classical_admit_value(fields[arrays[i]], context)
            if replacing:
                native_release_admit(fields[arrays[i]], context)
    if py_type(py_get(fields, "_mojolearn_fitting", PythonObject(0))) is not b.int:
        raise Error("TypeError: spectral fitting depth requires an exact integer")
    var fitted = py_get(fields, "_mojolearn_fitted", PythonObject(False))
    if py_type(fitted) is not b.bool and py_type(fitted) is not b.int:
        raise Error("TypeError: spectral fitted state requires a native boolean or integer")
    var ancestry = _type_mro(py_type(model))
    for i in range(len(ancestry)):  # driver-loop: no completion callback may execute from native fit
        if py_contains(_class_fields(ancestry[i]), "_mojolearn_after_fit"):
            raise Error("TypeError: spectral fitting cannot invoke Python completion callbacks")
    if py_contains(fields, "_mojolearn_after_fit"):
        raise Error("TypeError: spectral fitting cannot invoke Python completion callbacks")
    if not clustering:
        host_inference_scalar(family_raw_get(model, "_drop_first"), context, True)
        host_inference_scalar(family_raw_get(model, "_norm_laplacian"), context, True)
    return fields


def spectral_input(source: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    # Preserve the former _sk_validate dense-only class contract without
    # reading a custom metaclass's __module__ attribute.
    var module = py_call(b.type.__dict__["__module__"].__get__, py_type(source))
    if py_type(module) is b.str and String(module).startswith("scipy.sparse"):
        raise Error("TypeError: Sparse data was passed for X, but dense data is required. Use '.toarray()' to convert to a dense numpy array. (" + name + " takes dense X only.)")
    var view = classical_view(source, context, "X")
    var kind = classical_kind(view, context)
    if kind == "c":
        raise Error("ValueError: Complex data not supported: " + name + " got a complex X.")
    var shape = array_shape_once(view, context)
    if len(shape) < 2:
        raise Error("ValueError: Expected 2D array, got " + ("scalar" if len(shape) == 0 else "1D") + " array instead. " + RESHAPE_HINT)
    if len(shape) == 2 and shape[0] == 0:
        raise Error("ValueError: Found array with 0 sample(s) while a minimum of 1 is required by " + name + ".")
    if len(shape) == 2 and shape[1] == 0:
        raise Error("ValueError: Found array with 0 feature(s) while a minimum of 1 is required by " + name + ".")
    # Keep the original dtype: precomputed affinity's COO route consumes
    # float64 before its established conversion; validation must not narrow it.
    return view
