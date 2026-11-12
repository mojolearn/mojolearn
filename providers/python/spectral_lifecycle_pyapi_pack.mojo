# SPDX-License-Identifier: Apache-2.0
# Whole estimator metadata/lifecycle boundary for the spectral public methods.
# The PACK half of the former `providers/python/spectral_lifecycle_pyapi.mojo` (fence-flip split, 2026-11-12);
# the UNPACK half is `providers/python/spectral_lifecycle_pyapi_unpack.mojo`.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.mode_metadata_native import _fields, _class_fields
from bindings.classical_attributes import _type_mro
from bindings.classical_pandas import classical_admit_value
from bindings.classical_arrays_door import classical_kind
from bindings.classical_validate import classical_view, classical_estimator_name, RESHAPE_HINT
from bindings.host_inference_scalars_door import host_inference_scalar
from bindings.native_owner_lifetime import native_release_admit
from bindings.family_raw_attr_pack import family_raw_get, family_raw_get_or
from providers.python.classical_family_admission_pyapi import classical_family_admit
from providers.python.array_metadata_pyapi import array_shape_once


def spectral_admit(model: PythonObject, clustering: Bool, replacing: Bool,
                   context: PythonObject) raises -> PythonObject:
    var fields = classical_family_admit(model)
    var b = native_loaded_module("builtins")
    var names = context["spectral_clustering_parameters"] if clustering else context["spectral_embedding_parameters"]
    for i in range(len(names)):  # driver-loop: fixed consumed scalar configuration before comparisons/hash/conversion
        if py_type(names[i]) is not b.str:
            raise Error("TypeError: spectral parameter schema requires exact names")
        var key = String(py=names[i])
        var value = family_raw_get_or(model, key, PythonObject(None)) if key == "numeric_mode" else family_raw_get(model, key)
        if key == "kernel_params":
            # The algorithm refuses any non-None kernel_params; admission
            # prevents its diagnostic from traversing custom state.
            classical_admit_value(value, context)
        else:
            host_inference_scalar(value, context, True)
    host_inference_scalar(family_raw_get_or(model, "numeric_mode", PythonObject(None)), context, True)
    var scalars: List[String] = ["_n_clusters", "_n_neighbors", "_n_init", "_eigen_tol", "_gamma", "_seed", "_n_components", "n_features_in_", "input_copied_", "n_components_", "n_iter_assign_", "gamma_", "n_neighbors_", "_mojolearn_fitted", "_mojolearn_fitting"]
    var arrays: List[String] = ["_pd_eigenvalues", "_pd_eigenvectors", "_pd_diag", "_pd_centroids", "_fit_X", "affinity_matrix_", "embedding_", "labels_", "feature_names_in_"]
    for i in range(len(scalars)):  # driver-loop: consumed learned scalar state
        var key = PythonObject(scalars[i])
        if py_contains(fields, key):
            host_inference_scalar(fields[key], context, True)
            if replacing:
                native_release_admit(fields[key], context)
    for i in range(len(arrays)):  # driver-loop: retained fitted buffers, admitted before removal or replacement
        var key = PythonObject(arrays[i])
        if py_contains(fields, key):
            classical_admit_value(fields[key], context)
            if replacing:
                native_release_admit(fields[key], context)
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
