# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the Spectral family (SpectralClustering, kind 0).

UNPACK / PACK of core/spectral_member.mojo, in the family's place in
`bindings/member_boundary_door.mojo`. Admit answers True only for the form
with a predict: affinity='nearest_neighbors' and prediction_data=True,
after the class's own checks (`_sc_validate`, which writes the instance's
`_n_clusters` / `_n_neighbors` as the class's fit does). Every other
instance (an affinity matrix, 'rbf', 'precomputed_nearest_neighbors', no
prediction data) answers False and keeps the class's own doors.
SpectralEmbedding is not a member (fit / fit_transform only, no
out-of-sample transform).

Route: `route.entry` = the class's fit binding (`_mojolearn_metrics`, the
process default mode) `spectral_native_api_v1` entry; `route.second` = its
predict binding (the host binding of the estimator's route, else
`_mojolearn_metrics` in the estimator's mode), whose mode and vendor are the
route's and whose capability word is `route.flags`; `route.third` = the
x_cluster image's `spectral_assign_native_api_v1` entry (the estimator's
mode). Pack publishes what the class's dataset-route fit publishes:
input_copied_ False, _fit_X, n_features_in_, embedding_, n_components_,
n_iter_assign_ ('discretize' only), labels_ and the four prediction-data
arrays, and forgets the last fit's resident prediction state.
"""
from std.memory import bitcast, unsafe_memcpy
from std.python import Python, PythonObject
from bindings.admitted_family_door import admit_family
from bindings.cfb_spectral_door import _routed, _sc_validate
from bindings.classical_arrays_door import classical_address
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.family_common_pack import family_array
from bindings.family_common_unpack import family_is_str
from bindings.member_owner_unpack import MemberOwner
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_get, py_fields, py_setitem, py_type
from bindings.py_cluster_resident import CRES_SPECTRAL, cres_forget
from bindings.public_lifecycle_owner_door import lifecycle_admit_fields
from bindings.spectral_native_boundary_unpack import SpectralConnection
from bindings.spectral_umap_routes_unpack import spectral_umap_entry
from core.device_route import DeviceRoute, DEVICE_CPU, DEVICE_GPU, device_vendor_code
from core.entry_table import EntryRef, entry_ref_word
from core.hostptr import f32_ptr
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free,
    MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_IMPORT_DONE,
    MEMBER_F64,
)
from core.spectral_assignment_api import SPECTRAL_ASSIGN_ABI, SPECTRAL_ASSIGN_COUNT
from core.spectral_native_api import SPECTRAL_MODE, SPECTRAL_NATIVE_COUNT
from providers.python.host_estimator_binding_pyapi_door import host_estimator_binding
from providers.python.neural_descriptor_pyapi_door import (
    NeuralDescriptorProvider, neural_descriptor_connect_factory_once,
)

comptime _KIND_SPECTRAL = 0
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]


def _vendor(binding: PythonObject) raises -> Int:
    var vendor = device_vendor_code(String(py_call(binding.metrics_vendor)))
    if vendor < 0:
        raise Error("ImportError: mojolearn: the metrics binding names no vendor")
    return vendor


def spectral_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted estimator. False where the class's own doors run the
    instance (see the module notes)."""
    if kind != _KIND_SPECTRAL:
        return False
    classical_connect_numpy(context)
    var affinity = py_get(fields, PythonObject("affinity"), PythonObject(None))
    if not family_is_str(affinity) or String(py=affinity) != "nearest_neighbors":
        return False
    # The class's fit checks, in its order (`_sc_fit`).
    var config = _sc_validate(fields, context)
    var builtins = native_loaded_module("builtins")
    var prediction_data = py_get(fields, PythonObject("prediction_data"), PythonObject(None))
    if py_type(prediction_data) is not builtins.bool:
        raise Error("TypeError: mojolearn SpectralClustering: prediction_data must be a bool, "
                    + "got " + String(py_type(prediction_data).__name__))
    if not config.fit.keep_state:
        return False
    var mode_word = py_get(fields, PythonObject("numeric_mode"), PythonObject(None))
    # The fit binding: the process default mode, as `_routes` asks it.
    var fit_binding = _routed(model, "metrics", "_mojolearn_metrics", PythonObject(None), context)
    var fit = SpectralConnection(fit_binding, context)
    var fit_mode = fit.driver.metadata(SPECTRAL_MODE)
    var fit_vendor = _vendor(fit_binding)
    # The predict binding, as the class's predict door asks it.
    var predict_binding = host_estimator_binding(model, "_mojolearn_metrics", mode_word, context)
    if predict_binding is None:
        predict_binding = _routed(model, "metrics", "_mojolearn_metrics", mode_word, context)
    var query = SpectralConnection(predict_binding, context)
    var mode = query.driver.metadata(SPECTRAL_MODE)
    var vendor = _vendor(predict_binding)
    var route = DeviceRoute(DEVICE_CPU if vendor == 0 else DEVICE_GPU)
    route.mode = mode
    route.vendor = vendor
    route.entry = EntryRef(fit.driver.entry, entry_ref_word(fit_mode, fit_vendor, SPECTRAL_NATIVE_COUNT))
    route.second = EntryRef(query.driver.entry, entry_ref_word(mode, vendor, SPECTRAL_NATIVE_COUNT))
    route.has_second = True
    route.flags = query.capabilities
    # The assignment entry (x_cluster, the estimator's mode), as `_sc_fit` runs it.
    var cluster_binding = _routed(model, "cluster", "_mojolearn_x_cluster", mode_word, context)
    var factory = spectral_umap_entry(cluster_binding, "spectral_assign_native_api_v1", context)
    var assign = neural_descriptor_connect_factory_once(factory, String(SPECTRAL_ASSIGN_ABI), SPECTRAL_ASSIGN_COUNT)
    route.third = EntryRef(assign.downcast_value_ptr[NeuralDescriptorProvider]()[].record[0].entry,
        entry_ref_word(mode, vendor, SPECTRAL_ASSIGN_COUNT))
    route.has_third = True
    owner.plan.route = route
    var options = config.fit
    var words: List[Float64] = [Float64(_KIND_SPECTRAL), Float64(options.clusters), Float64(options.components),
        Float64(options.initializations), Float64(options.neighbors), options.tolerance,
        bitcast[DType.float64](Int64(options.seed)), Float64(config.assignment)]
    owner.plan.params = words^
    owner.pins = Python.tuple(fit_binding, fit.provider, predict_binding, query.provider, cluster_binding, assign)
    return True


def _export(plan: MemberPlan, mut call: MemberCall, block: Int) raises:
    call.block = block
    call.flags = 0
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)


def _f32_array(view: MemberMatrix, shape: PythonObject, context: PythonObject) raises -> PythonObject:
    var array = family_array(shape, "<f4", context)
    var count = view.rows * view.cols
    if count > 0:
        unsafe_memcpy(dest=f32_ptr(Int(py=classical_address(array))), src=f32_ptr(view.data), count=count)
    return array


def spectral_member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: write the fitted attributes of state `state` into `model`, as
    the class's dataset-route fit publishes them (same names, dtypes, shapes
    and order)."""
    classical_connect_numpy(context)
    var model_fields = py_fields(model)
    lifecycle_admit_fields(model_fields)
    var clusters = Int(plan.params[1])
    var neighbors = Int(plan.params[4])
    var assignment = Int(plan.params[7])
    var call = MemberCall(plan.route, state)
    _export(plan, call, MEMBER_BLOCK_SCALARS)
    var scalars = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data)
    var k = Int(scalars[0])
    var iterations = Int(scalars[1])
    _export(plan, call, MEMBER_BLOCK_FAMILY + 6)
    var n = call.dst.rows
    var d = call.dst.cols
    var fit_x = _f32_array(call.dst, Python.tuple(n, d), context)
    _export(plan, call, MEMBER_BLOCK_FAMILY)
    var labels = family_array(Python.tuple(n), "<i4", context)
    if n > 0:
        unsafe_memcpy(dest=_I32P(unsafe_from_address=Int(py=classical_address(labels))),
            src=_I32P(unsafe_from_address=call.dst.data), count=n)
    _export(plan, call, MEMBER_BLOCK_FAMILY + 1)
    var embedding = _f32_array(call.dst, Python.tuple(n, k), context)
    _export(plan, call, MEMBER_BLOCK_FAMILY + 2)
    var eigenvalues = _f32_array(call.dst, Python.tuple(k), context)
    _export(plan, call, MEMBER_BLOCK_FAMILY + 3)
    var eigenvectors = _f32_array(call.dst, Python.tuple(n, k), context)
    _export(plan, call, MEMBER_BLOCK_FAMILY + 4)
    var diagonal = _f32_array(call.dst, Python.tuple(n), context)
    _export(plan, call, MEMBER_BLOCK_FAMILY + 5)
    var centroids = _f32_array(call.dst, Python.tuple(clusters, k), context)
    var dropped: List[String] = ["_pd_eigenvalues", "_pd_eigenvectors", "_pd_diag", "_pd_centroids", "_fit_X",
        "affinity_matrix_"]
    for i in range(len(dropped)):  # small-loop(prediction data and affinity: six cached attribute names)
        _ = py_call(model_fields.pop, PythonObject(dropped[i]), PythonObject(None))
    py_setitem(model_fields, "_n_clusters", PythonObject(clusters))
    py_setitem(model_fields, "_n_neighbors", PythonObject(neighbors))
    py_setitem(model_fields, "input_copied_", PythonObject(False))
    py_setitem(model_fields, "_fit_X", fit_x)
    py_setitem(model_fields, "n_features_in_", PythonObject(d))
    py_setitem(model_fields, "embedding_", embedding)
    py_setitem(model_fields, "n_components_", PythonObject(k))
    _ = py_call(model_fields.pop, PythonObject("n_iter_assign_"), PythonObject(None))
    if assignment == 0:
        py_setitem(model_fields, "n_iter_assign_", PythonObject(iterations))
    # A fit releases the last fit's resident prediction state at once (N326).
    cres_forget(model, CRES_SPECTRAL)
    py_setitem(model_fields, "labels_", labels)
    py_setitem(model_fields, "_pd_eigenvalues", eigenvalues)
    py_setitem(model_fields, "_pd_eigenvectors", eigenvectors)
    py_setitem(model_fields, "_pd_diag", diagonal)
    py_setitem(model_fields, "_pd_centroids", centroids)


def _import(plan: MemberPlan, mut call: MemberCall, block: Int, view: MemberMatrix, flags: Int = 0) raises:
    call.block = block
    call.x = view
    call.flags = flags
    _ = member_call(plan.member, MEMBER_OP_IMPORT, call)


def _view(array: PythonObject, rows: Int, cols: Int) raises -> MemberMatrix:
    if rows * cols == 0:
        return MemberMatrix(0, rows, cols)
    return MemberMatrix(Int(py=classical_address(array)), rows, cols)


def spectral_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding the prediction data of a fitted
    model (`_fit_X`, `_pd_eigenvalues`, `_pd_eigenvectors`, `_pd_diag`,
    `_pd_centroids`, n_components_, n_iter_assign_). The caller frees it
    (`member_free`)."""
    classical_connect_numpy(context)
    var source = py_get(fields, PythonObject("_fit_X"), PythonObject(None))
    var values_source = py_get(fields, PythonObject("_pd_eigenvalues"), PythonObject(None))
    if source is None or values_source is None:
        raise Error("mojolearn SpectralClustering.predict: prediction data was not stored. Fit with"
            + " SpectralClustering(prediction_data=True), which keeps the eigenpairs, degrees and centroids"
            + " the Nystrom extension needs (DEVIATION 2860)")
    var training = admit_family(source, 2, "fitted training rows", context).array
    var eigenvalues = admit_family(values_source, 1, "fitted eigenvalues", context).array
    var eigenvectors = admit_family(py_get(fields, PythonObject("_pd_eigenvectors"), PythonObject(None)), 2,
        "fitted eigenvectors", context).array
    var diagonal = admit_family(py_get(fields, PythonObject("_pd_diag"), PythonObject(None)), 1,
        "fitted degree diagonal", context).array
    var centroids = admit_family(py_get(fields, PythonObject("_pd_centroids"), PythonObject(None)), 2,
        "fitted centroids", context).array
    var n = Int(py=training.shape[0])
    var d = Int(py=training.shape[1])
    var k = Int(py=eigenvalues.shape[0])
    var iterations = py_get(fields, PythonObject("n_iter_assign_"), PythonObject(None))
    var scalars: List[Float64] = [Float64(k), -1.0 if iterations is None else Float64(py=iterations)]
    var call = member_new(plan)
    try:
        _import(plan, call, MEMBER_BLOCK_FAMILY + 6, _view(training, n, d))
        _import(plan, call, MEMBER_BLOCK_FAMILY + 2, _view(eigenvalues, 1, k))
        _import(plan, call, MEMBER_BLOCK_FAMILY + 3, _view(eigenvectors, n, k))
        _import(plan, call, MEMBER_BLOCK_FAMILY + 4, _view(diagonal, 1, n))
        _import(plan, call, MEMBER_BLOCK_FAMILY + 5,
            _view(centroids, Int(py=centroids.shape[0]), Int(py=centroids.shape[1])))
        _import(plan, call, MEMBER_BLOCK_SCALARS, MemberMatrix(Int(scalars.unsafe_ptr()), 1, 2, MEMBER_F64),
            MEMBER_IMPORT_DONE)
    except error:
        member_free(plan, call)
        raise error
    _ = training
    _ = eigenvalues
    _ = eigenvectors
    _ = diagonal
    _ = centroids
    _ = source
    _ = values_source
    _ = len(scalars)
    return call.state
