# SPDX-License-Identifier: Apache-2.0
# The incumbent mojolearn-pca-1 ZIP/NPY schema, using the shared native archive
# reader/writer. This private owner never calls PCA.load or _serialize.
from std.python import Python, PythonObject
from bindings.sequential_arrays import sequential_array, sequential_address
from bindings.sequential_description import sequential_fields
from bindings.classical_arrays import classical_array, classical_word
from bindings.classical_archive import classical_archive_write
from bindings.classical_archive_read import classical_archive_read, _archive_exact, _archive_string
from bindings.pca_metadata import pca_mode, pca_initialize


def pca_archive_save(model: PythonObject, path: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = sequential_fields(model)
    if not Bool(py=fields.__contains__("components_")):
        raise Error("RuntimeError: this estimator is not fitted yet")
    var values = Python.dict()
    values["format"] = "mojolearn-pca-1"
    values["estimator"] = "PCA"
    values["numeric_mode"] = pca_mode(fields, context)
    values["svd_solver"] = Python.import_module("builtins").str(fields["svd_solver"])
    var names = Python.tuple("components", "mean", "singular_values", "explained_variance", "explained_variance_ratio")
    for k in range(len(names)):  # driver-loop: over the five existing PCA archive array members
        values[names[k]] = fields[names[k] + "_"]
    var noise = classical_array(1, 0, context, True)
    var np = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(noise)))
    np[0] = Float64(py=fields["noise_variance_"])
    values["noise_variance"] = noise
    var meta = sequential_array(Python.tuple(4), context["archive_npy_dtypes"]["<i8"], context)
    var mp = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(meta)))
    mp[0] = Int64(py=fields["n_components_"])
    mp[1] = Int64(py=fields["n_features_in_"])
    mp[2] = Int64(py=fields["n_samples_"])
    mp[3] = 1 if Bool(py=fields["whiten"]) else 0
    values["meta"] = meta
    var result = classical_archive_write(path, values, context)
    _ = noise^
    _ = meta^
    _ = values^
    return result


def pca_archive_load(cls: PythonObject, path: PythonObject, context: PythonObject) raises -> PythonObject:
    var arrays = classical_archive_read(path, context)
    if String(_archive_string(arrays, "format", context)) != "mojolearn-pca-1":
        raise Error("model archive format does not match mojolearn-pca-1")
    if String(_archive_string(arrays, "estimator", context)) != "PCA":
        raise Error("model archive was saved by a different estimator")
    var meta = _archive_exact(arrays, "meta", "<i8", context)
    if Int(py=meta.size) != 4:
        raise Error("PCA model meta must hold four fields")
    var nc = Int(py=meta.flat[0])
    var nf = Int(py=meta.flat[1])
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
        spectral.append(value)
    var noise = _archive_exact(arrays, "noise_variance", "<f8", context)
    if Int(py=noise.size) != 1:
        raise Error("PCA model noise_variance must hold one value")
    var solver = _archive_string(arrays, "svd_solver", context)
    var model = Python.import_module("builtins").object.__new__(cls)
    _ = pca_initialize(model, Python.tuple(nc, True, Bool(py=meta.flat[3]), solver, 0.0,
                                          "auto", 10, "auto", PythonObject(None), mode))
    var fields = sequential_fields(model)
    fields["components_"] = components
    fields["mean_"] = mean
    for k in range(len(names)):  # driver-loop: over fully validated saved spectral arrays
        fields[names[k] + "_"] = spectral[k]
    fields["noise_variance_"] = PythonObject(classical_word(noise))
    fields["n_components_"] = nc
    fields["n_features_in_"] = nf
    fields["n_samples_"] = Int(py=meta.flat[2])
    fields["_mojolearn_fitted"] = True
    return model
