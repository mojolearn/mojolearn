# SPDX-License-Identifier: Apache-2.0
# The incumbent mojolearn-pca-1 ZIP/NPY schema, using the shared native archive
# reader/writer. This private owner never calls PCA.load or _serialize.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.sequential_arrays_door import sequential_array, sequential_address
from bindings.sequential_description_unpack import sequential_fields
from bindings.classical_arrays_door import classical_array, classical_word
from bindings.classical_archive_door import classical_archive_write
from bindings.classical_archive_read_door import classical_archive_read, _archive_exact, _archive_string
from bindings.pca_metadata_unpack import pca_mode
from bindings.pca_metadata_pack import pca_initialize
from bindings.py_balanced import py_call, py_contains, py_setitem
from bindings.pca_keys_unpack import pca_keys
from bindings.py_balanced import py_item



def _meta_word(meta: PythonObject, index: Int) raises -> Int:
    # `meta` is the admitted 1-D <i8 member (an Array or an ndarray): read the
    # word at its data address; neither `.flat` nor an item protocol runs.
    var address = Int(py=sequential_address(meta))
    return Int(MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=address)[index])


def pca_archive_save(model: PythonObject, path: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = pca_keys()
    var fields = sequential_fields(model)
    if not py_contains(fields, nk[].k_components_):
        raise Error("RuntimeError: this estimator is not fitted yet")
    var values = Python.dict()
    py_setitem(values, nk[].k_format, "mojolearn-pca-1")
    py_setitem(values, nk[].k_estimator, "PCA")
    py_setitem(values, nk[].k_numeric_mode, pca_mode(fields, context))
    py_setitem(values, nk[].k_svd_solver, py_call(native_loaded_module("builtins").str, py_item(fields, nk[].k_svd_solver)))
    var names = Python.tuple("components", "mean", "singular_values", "explained_variance", "explained_variance_ratio")
    for k in range(len(names)):  # driver-loop: over the five existing PCA archive array members
        py_setitem(values, names[k], fields[names[k] + "_"])
    var noise = classical_array(1, 0, context, True)
    var np = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(noise)))
    np[0] = Float64(py=py_item(fields, nk[].k_noise_variance_))
    py_setitem(values, nk[].k_noise_variance, noise)
    var meta = sequential_array(Python.tuple(4), py_item(context, nk[].k_archive_npy_dtypes)["<i8"], context)
    var mp = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(meta)))
    mp[0] = Int64(py=py_item(fields, nk[].k_n_components_))
    mp[1] = Int64(py=py_item(fields, nk[].k_n_features_in_))
    mp[2] = Int64(py=py_item(fields, nk[].k_n_samples_))
    mp[3] = 1 if Bool(py=py_item(fields, nk[].k_whiten)) else 0
    py_setitem(values, nk[].k_meta, meta)
    var result = classical_archive_write(path, values, context)
    _ = noise^
    _ = meta^
    _ = values^
    return result


def pca_archive_load(cls: PythonObject, path: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = pca_keys()
    var arrays = classical_archive_read(path, context)
    if String(_archive_string(arrays, "format", context)) != "mojolearn-pca-1":
        raise Error("model archive format does not match mojolearn-pca-1")
    if String(_archive_string(arrays, "estimator", context)) != "PCA":
        raise Error("model archive was saved by a different estimator")
    var meta = _archive_exact(arrays, "meta", "<i8", context)
    if Int(py=meta.size) != 4:
        raise Error("PCA model meta must hold four fields")
    var nc = _meta_word(meta, 0)
    var nf = _meta_word(meta, 1)
    var mode = _archive_string(arrays, "numeric_mode", context)
    if String(mode) != "identical" and String(mode) != "fast":
        raise Error("invalid saved numeric_mode")
    var components = _archive_exact(arrays, "components", "<f4", context)
    if components.shape != Python.tuple(nc, nf):
        raise Error("PCA model components do not match saved dimensions")
    var mean = _archive_exact(arrays, "mean", "<f4", context)
    if Int(py=mean.size) != nf:
        raise Error("PCA model mean does not match n_features_in_")
    var names = Python.tuple("singular_values", "explained_variance", "explained_variance_ratio")
    var spectral = Python.list()
    for k in range(len(names)):  # driver-loop: over the three saved PCA spectrum members
        var value = _archive_exact(arrays, String(names[k]), "<f4", context)
        if Int(py=value.size) != nc:
            raise Error("PCA model spectrum does not match n_components_")
        _ = py_call(spectral.append, value)
    var noise = _archive_exact(arrays, "noise_variance", "<f8", context)
    if Int(py=noise.size) != 1:
        raise Error("PCA model noise_variance must hold one value")
    var solver = _archive_string(arrays, "svd_solver", context)
    var model = py_call(native_loaded_module("builtins").object.__new__, cls)
    # A loaded model runs this installation's device (`device=None`): the
    # device is a constructor parameter, not model state, and is not saved.
    _ = pca_initialize(model, Python.tuple(nc, True, _meta_word(meta, 3) != 0, solver, 0.0,
                                          "auto", 10, "auto", PythonObject(None), mode, PythonObject(None)), context)
    var fields = sequential_fields(model)
    py_setitem(fields, nk[].k_components_, components)
    py_setitem(fields, nk[].k_mean_, mean)
    for k in range(len(names)):  # driver-loop: over fully validated saved spectral arrays
        py_setitem(fields, names[k] + "_", spectral[k])
    py_setitem(fields, nk[].k_noise_variance_, PythonObject(classical_word(noise)))
    py_setitem(fields, nk[].k_n_components_, nc)
    py_setitem(fields, nk[].k_n_features_in_, nf)
    py_setitem(fields, nk[].k_n_samples_, _meta_word(meta, 2))
    py_setitem(fields, nk[].k__mojolearn_fitted, True)
    return model
