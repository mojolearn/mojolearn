# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the Spectral family (SpectralClustering, kind 0).

UNPACK / PACK of core/spectral_member.mojo, in the family's place in
`bindings/member_boundary_door.mojo`. Admit answers True for every
instance the class's fit admits (lane mf-clu6): every affinity, with or
without prediction data, after the class's own checks (`_sc_validate`, run
on a SCRATCH copy of the instance dictionary: the derived `_n_clusters`,
`_n_neighbors`, `_n_init`, `_eigen_tol`, `_gamma`, `_seed` it writes land in
the copy, so the template estimator a caller holds is never written; PACK
runs it again on the clone, which receives them as the class's fit writes
them).
SpectralEmbedding (kind 1, lane mf-member8) is a fit + export member (no
transform, no predict): `_se_admit` runs the class's checks (`_se_validate`)
on a scratch copy and builds the route of its fit (`_routes`: the metrics
image's spectral entry, the base image's preparation entry for
'precomputed_nearest_neighbors' or a graph affinity on a fit entry without
the dense capability, the x_decomp image for 'rbf'); `_se_pack` publishes what
`_se_execute` publishes for the affinity.

Route: `route.entry` = the class's fit binding (`_mojolearn_metrics`, the
process default mode) `spectral_native_api_v1` entry; `route.second` = its
predict binding (the host binding of the estimator's route, else
`_mojolearn_metrics` in the estimator's mode), whose mode and vendor are the
route's and whose capability word is `route.flags`; `route.third` = the
x_cluster image's `spectral_assign_native_api_v1` entry (the estimator's
mode); `route.fourth` = the base image's `spectral_preparation_native_api_v1`
entry (identical; 'precomputed_nearest_neighbors', or a graph affinity on a
fit entry without the dense capability); `route.fifth` = the x_decomp
image's entry ('rbf'). Pack publishes what the class's fit publishes for the
affinity: affinity_matrix_ (the precomputed X's copy, or the built affinity)
and n_features_in_ ('rbf'), or input_copied_ False, _fit_X (with prediction
data) and n_features_in_ (the dataset route); then embedding_,
n_components_, n_iter_assign_ ('discretize' only), labels_ and, with
prediction data, the four prediction-data arrays; it forgets the last fit's
resident prediction state.
"""
from std.memory import bitcast, unsafe_memcpy
from std.python import Python, PythonObject
from bindings.admitted_family_door import admit_family
from bindings.cfb_spectral_door import _routed, _sc_validate, SpectralPreparationConnection
from bindings.cfb_spectral_door import _se_validate
from bindings.family_raw_attr_pack import family_raw_get
from core.spectral_native_api import SPECTRAL_CAP_DENSE
from core.spectral_preparation_api import SPECTRAL_PREPARATION_COUNT
from x_decomp.descriptor_schema import XD_ABI_NAME, XD_OPERATION_COUNT
from bindings.classical_arrays_door import classical_address
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.family_common_pack import family_array
from bindings.family_common_unpack import family_is_str
from bindings.member_owner_unpack import MemberOwner, member_host_only
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
from providers.python.estimator_binding_pyapi_door import estimator_binding_host
from providers.python.host_estimator_binding_pyapi_door import host_estimator_binding, estimator_member_binding
from providers.python.neural_descriptor_pyapi_door import (
    NeuralDescriptorProvider, neural_descriptor_connect_factory_once, neural_descriptor_connect_once,
)

comptime _KIND_SPECTRAL = 0
comptime _KIND_EMBEDDING = 1
comptime _AFF_NN = 0
comptime _AFF_PRECOMPUTED = 1
comptime _AFF_RBF = 2
comptime _AFF_PNN = 3
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
    if kind == _KIND_EMBEDDING:
        return _se_admit(model, fields, context, owner)
    if kind != _KIND_SPECTRAL:
        # Kinds 0 and 1 are marked (python/mojolearn/_spectral_impl.py); any
        # other mark is the member's own NEW refusal, raised here.
        raise Error("member ABI: spectral member kind " + String(kind) + " is not published")
    classical_connect_numpy(context)
    # The class's fit checks, in its order (`_sc_fit`), on a scratch copy:
    # `_sc_validate` writes its derived fields into the dictionary it reads.
    var builtins = native_loaded_module("builtins")
    var scratch = py_call(builtins.dict, fields)
    var config = _sc_validate(scratch, context)
    var prediction_data = py_get(fields, PythonObject("prediction_data"), PythonObject(None))
    if py_type(prediction_data) is not builtins.bool:
        raise Error("TypeError: mojolearn SpectralClustering: prediction_data must be a bool, "
                    + "got " + String(py_type(prediction_data).__name__))
    var affinity_code = _affinity_code(config.affinity)
    var mode_word = py_get(fields, PythonObject("numeric_mode"), PythonObject(None))
    # The fit binding: the process default mode, as `_routes` asks it. An explicit HostSpectralClustering (its
    # own host-only word, lane mf-iterimp3b): its import-connected host image for every binding of the route
    # (`estimator_member_binding`, `estimator_binding_host`), never a device one beside it.
    var host_class = member_host_only(model)
    var fit_binding = (estimator_member_binding(model, "_mojolearn_metrics", context) if host_class
                       else _routed(model, "metrics", "_mojolearn_metrics", PythonObject(None), context))
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
    var cluster_binding = (estimator_binding_host(model, "_mojolearn_x_cluster", context) if host_class
                           else _routed(model, "cluster", "_mojolearn_x_cluster", mode_word, context))
    var factory = spectral_umap_entry(cluster_binding, "spectral_assign_native_api_v1", context)
    var assign = neural_descriptor_connect_factory_once(factory, String(SPECTRAL_ASSIGN_ABI), SPECTRAL_ASSIGN_COUNT)
    route.third = EntryRef(assign.downcast_value_ptr[NeuralDescriptorProvider]()[].record[0].entry,
        entry_ref_word(mode, vendor, SPECTRAL_ASSIGN_COUNT))
    route.has_third = True
    # A graph affinity goes to the fit entry dense when it serves dense
    # matrices, else as COO triples (`_sc_fit`'s `use_dense`).
    var fit_dense = (fit.capabilities & SPECTRAL_CAP_DENSE) != 0
    var base_binding = PythonObject(None)
    var preparation = PythonObject(None)
    if affinity_code == _AFF_PNN or (affinity_code != _AFF_NN and not fit_dense):
        # The base image's preparation entry (COO count/fill, the k-NN
        # affinity), identical only, as `_routes` asks it.
        base_binding = (estimator_binding_host(model, "_mojolearn", context) if host_class
                        else _routed(model, "base", "_mojolearn", context["family_cfb_identical"], context))
        var prep = SpectralPreparationConnection(base_binding, context)
        route.fourth = EntryRef(prep.driver.entry, entry_ref_word(mode, vendor, SPECTRAL_PREPARATION_COUNT))
        route.has_fourth = True
        preparation = prep.provider
    var decomp_binding = PythonObject(None)
    var decomp = PythonObject(None)
    if affinity_code == _AFF_RBF:
        # The x_decomp image's RBF cells (process default mode), as `_routes` asks it.
        decomp_binding = (estimator_binding_host(model, "_mojolearn_x_decomp", context) if host_class
                          else _routed(model, "decomp", "_mojolearn_x_decomp", PythonObject(None), context))
        decomp = neural_descriptor_connect_once(decomp_binding, "x_decomp_native_api_v1", String(XD_ABI_NAME),
            XD_OPERATION_COUNT)
        route.fifth = EntryRef(decomp.downcast_value_ptr[NeuralDescriptorProvider]()[].record[0].entry,
            entry_ref_word(mode, vendor, XD_OPERATION_COUNT))
        route.has_fifth = True
    owner.plan.route = route
    var options = config.fit
    var words: List[Float64] = [Float64(_KIND_SPECTRAL), Float64(options.clusters), Float64(options.components),
        Float64(options.initializations), Float64(options.neighbors), options.tolerance,
        bitcast[DType.float64](Int64(options.seed)), Float64(config.assignment), Float64(affinity_code),
        1.0 if options.keep_state else 0.0, config.gamma, 1.0 if fit_dense else 0.0]
    owner.plan.params = words^
    owner.pins = Python.tuple(fit_binding, fit.provider, predict_binding, query.provider, cluster_binding, assign,
        base_binding, preparation, decomp_binding, decomp)
    return True


def _se_admit(model: PythonObject, fields: PythonObject, context: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """Lane mf-member8: UNPACK of a SpectralEmbedding member: the class's
    checks (`_se_validate`, on a scratch copy of the instance dictionary: the
    derived `_n_components`, `_n_neighbors`, `_seed`, `_gamma` land in the
    copy; PACK runs it again on the clone), its two flags, and the route of
    its fit (`_routes`: metrics and x_decomp in the process default mode, the
    base image identical)."""
    classical_connect_numpy(context)
    var builtins = native_loaded_module("builtins")
    var scratch = py_call(builtins.dict, fields)
    var config = _se_validate(scratch, context)
    var drop_first = Bool(py=py_call(builtins.bool, family_raw_get(model, "_drop_first")))
    var norm_laplacian = Bool(py=py_call(builtins.bool, family_raw_get(model, "_norm_laplacian")))
    var affinity_code = _affinity_code(config.affinity)
    var host_class = member_host_only(model)
    var fit_binding = (estimator_member_binding(model, "_mojolearn_metrics", context) if host_class
                       else _routed(model, "metrics", "_mojolearn_metrics", PythonObject(None), context))
    var fit = SpectralConnection(fit_binding, context)
    var mode = fit.driver.metadata(SPECTRAL_MODE)
    var vendor = _vendor(fit_binding)
    var route = DeviceRoute(DEVICE_CPU if vendor == 0 else DEVICE_GPU)
    route.mode = mode
    route.vendor = vendor
    route.entry = EntryRef(fit.driver.entry, entry_ref_word(mode, vendor, SPECTRAL_NATIVE_COUNT))
    route.flags = fit.capabilities
    var fit_dense = (fit.capabilities & SPECTRAL_CAP_DENSE) != 0
    var base_binding = PythonObject(None)
    var preparation = PythonObject(None)
    if affinity_code == _AFF_PNN or (affinity_code != _AFF_NN and not fit_dense):
        base_binding = (estimator_binding_host(model, "_mojolearn", context) if host_class
                        else _routed(model, "base", "_mojolearn", context["family_cfb_identical"], context))
        var prep = SpectralPreparationConnection(base_binding, context)
        route.fourth = EntryRef(prep.driver.entry, entry_ref_word(mode, vendor, SPECTRAL_PREPARATION_COUNT))
        route.has_fourth = True
        preparation = prep.provider
    var decomp_binding = PythonObject(None)
    var decomp = PythonObject(None)
    if affinity_code == _AFF_RBF:
        decomp_binding = (estimator_binding_host(model, "_mojolearn_x_decomp", context) if host_class
                          else _routed(model, "decomp", "_mojolearn_x_decomp", PythonObject(None), context))
        decomp = neural_descriptor_connect_once(decomp_binding, "x_decomp_native_api_v1", String(XD_ABI_NAME),
            XD_OPERATION_COUNT)
        route.fifth = EntryRef(decomp.downcast_value_ptr[NeuralDescriptorProvider]()[].record[0].entry,
            entry_ref_word(mode, vendor, XD_OPERATION_COUNT))
        route.has_fifth = True
    owner.plan.route = route
    var automatic = (1 if config.automatic_neighbors else 0) + (2 if config.automatic_gamma else 0)
    var words: List[Float64] = [Float64(_KIND_EMBEDDING), 0.0, Float64(config.components), 0.0,
        Float64(config.neighbors), config.tolerance, bitcast[DType.float64](Int64(config.seed)), -1.0,
        Float64(affinity_code), 0.0, config.gamma, 1.0 if fit_dense else 0.0, 1.0 if drop_first else 0.0,
        1.0 if norm_laplacian else 0.0, Float64(automatic)]
    owner.plan.params = words^
    owner.pins = Python.tuple(fit_binding, fit.provider, base_binding, preparation, decomp_binding, decomp)
    return True


def _se_pack(model: PythonObject, context: PythonObject, plan: MemberPlan, state: Int) raises:
    """Lane mf-member8: `_se_execute`'s publications for the affinity, in its
    order: 'rbf' gamma_, affinity_matrix_ and n_features_in_; the dataset
    route input_copied_ (False: the member reads the caller's admitted X) and
    n_features_in_; n_neighbors_ (every route but a sparse precomputed one);
    then embedding_. The derived fields of `_se_validate` are written on the
    clone first, as the class's fit writes them."""
    var model_fields = py_fields(model)
    lifecycle_admit_fields(model_fields)
    _ = _se_validate(model_fields, context)
    var affinity = Int(plan.params[8])
    var call = MemberCall(plan.route, state)
    _export(plan, call, MEMBER_BLOCK_SCALARS)
    var scalars = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data)
    var k = Int(scalars[0])
    var neighbors = Int(scalars[1])
    var features = Int(scalars[2])
    var gamma = scalars[3]
    _export(plan, call, MEMBER_BLOCK_FAMILY + 1)
    var n = call.dst.rows
    var embedding = _f32_array(call.dst, Python.tuple(n, k), context)
    if affinity == _AFF_RBF:
        _export(plan, call, MEMBER_BLOCK_FAMILY + 7)
        var built = _f32_array(call.dst, Python.tuple(call.dst.rows, call.dst.cols), context)
        py_setitem(model_fields, "gamma_", PythonObject(gamma))
        py_setitem(model_fields, "affinity_matrix_", built)
        py_setitem(model_fields, "n_features_in_", PythonObject(features))
    if affinity == _AFF_NN:
        py_setitem(model_fields, "input_copied_", PythonObject(False))
    if neighbors >= 0:
        py_setitem(model_fields, "n_neighbors_", PythonObject(neighbors))
    if affinity == _AFF_NN:
        py_setitem(model_fields, "n_features_in_", PythonObject(features))
    py_setitem(model_fields, "embedding_", embedding)


def _affinity_code(affinity: String) raises -> Int:
    """The affinity names `_sc_validate` admits, as NEW word 8 carries them."""
    if affinity == "nearest_neighbors":
        return _AFF_NN
    if affinity == "precomputed":
        return _AFF_PRECOMPUTED
    if affinity == "rbf":
        return _AFF_RBF
    if affinity == "precomputed_nearest_neighbors":
        return _AFF_PNN
    raise Error("mojolearn SpectralClustering: affinity='" + affinity + "' passed the class's checks but has"
                + " no member code")


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
    the class's fit publishes them for its affinity (`_sc_fit`: same names,
    dtypes, shapes and order)."""
    classical_connect_numpy(context)
    if Int(plan.params[0]) == _KIND_EMBEDDING:
        _se_pack(model, context, plan, state)
        return
    var model_fields = py_fields(model)
    lifecycle_admit_fields(model_fields)
    # The derived fields the class's fit writes (`_n_clusters`, `_n_neighbors`,
    # `_n_init`, `_eigen_tol`, `_gamma`, `_seed`), into the clone, by the
    # class's own checks (the admit ran them on a scratch copy).
    _ = _sc_validate(model_fields, context)
    var clusters = Int(plan.params[1])
    var assignment = Int(plan.params[7])
    var affinity = Int(plan.params[8])
    var keep = plan.params[9] != 0.0
    var call = MemberCall(plan.route, state)
    _export(plan, call, MEMBER_BLOCK_SCALARS)
    var scalars = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data)
    var k = Int(scalars[0])
    var iterations = Int(scalars[1])
    _export(plan, call, MEMBER_BLOCK_FAMILY)
    var n = call.dst.cols
    var labels = family_array(Python.tuple(n), "<i4", context)
    if n > 0:
        unsafe_memcpy(dest=_I32P(unsafe_from_address=Int(py=classical_address(labels))),
            src=_I32P(unsafe_from_address=call.dst.data), count=n)
    _export(plan, call, MEMBER_BLOCK_FAMILY + 6)
    var d = call.dst.cols
    var training = _f32_array(call.dst, Python.tuple(call.dst.rows, d), context)
    var built = PythonObject(None)
    if affinity == _AFF_RBF or affinity == _AFF_PNN:
        _export(plan, call, MEMBER_BLOCK_FAMILY + 7)
        built = _f32_array(call.dst, Python.tuple(n, n), context)
    _export(plan, call, MEMBER_BLOCK_FAMILY + 1)
    var embedding = _f32_array(call.dst, Python.tuple(n, k), context)
    var eigenvalues = PythonObject(None)
    var eigenvectors = PythonObject(None)
    var diagonal = PythonObject(None)
    var centroids = PythonObject(None)
    if keep:
        _export(plan, call, MEMBER_BLOCK_FAMILY + 2)
        eigenvalues = _f32_array(call.dst, Python.tuple(k), context)
        _export(plan, call, MEMBER_BLOCK_FAMILY + 3)
        eigenvectors = _f32_array(call.dst, Python.tuple(n, k), context)
        _export(plan, call, MEMBER_BLOCK_FAMILY + 4)
        diagonal = _f32_array(call.dst, Python.tuple(n), context)
        _export(plan, call, MEMBER_BLOCK_FAMILY + 5)
        centroids = _f32_array(call.dst, Python.tuple(clusters, k), context)
    var dropped: List[String] = ["_pd_eigenvalues", "_pd_eigenvectors", "_pd_diag", "_pd_centroids", "_fit_X",
        "affinity_matrix_"]
    for i in range(len(dropped)):  # small-loop(prediction data and affinity: six cached attribute names)
        _ = py_call(model_fields.pop, PythonObject(dropped[i]), PythonObject(None))
    if affinity == _AFF_PRECOMPUTED:
        # The class publishes the X it was given; the member's copy of it.
        py_setitem(model_fields, "affinity_matrix_", training)
    elif affinity == _AFF_RBF:
        py_setitem(model_fields, "affinity_matrix_", built)
        py_setitem(model_fields, "n_features_in_", PythonObject(d))
    elif affinity == _AFF_PNN:
        py_setitem(model_fields, "affinity_matrix_", built)
    else:
        py_setitem(model_fields, "input_copied_", PythonObject(False))
        if keep:
            py_setitem(model_fields, "_fit_X", training)
        py_setitem(model_fields, "n_features_in_", PythonObject(d))
    py_setitem(model_fields, "embedding_", embedding)
    py_setitem(model_fields, "n_components_", PythonObject(k))
    _ = py_call(model_fields.pop, PythonObject("n_iter_assign_"), PythonObject(None))
    if assignment == 0:
        py_setitem(model_fields, "n_iter_assign_", PythonObject(iterations))
    # A fit releases the last fit's resident prediction state at once (N326).
    cres_forget(model, CRES_SPECTRAL)
    py_setitem(model_fields, "labels_", labels)
    if keep:
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
    if Int(plan.params[0]) == _KIND_EMBEDDING:
        # Lane mf-member8: a fitted SpectralEmbedding serves no member query.
        raise Error("NotImplementedError: member ABI: a SpectralEmbedding member is fitted, not imported "
                    + "(SpectralEmbedding has no transform)")
    # A precomputed fit keeps no training rows (its queries are affinities).
    var precomputed = Int(plan.params[8]) == _AFF_PRECOMPUTED
    var source = py_get(fields, PythonObject("_fit_X"), PythonObject(None))
    var values_source = py_get(fields, PythonObject("_pd_eigenvalues"), PythonObject(None))
    if (source is None and not precomputed) or values_source is None:
        raise Error("mojolearn SpectralClustering.predict: prediction data was not stored. Fit with"
            + " SpectralClustering(prediction_data=True), which keeps the eigenpairs, degrees and centroids"
            + " the Nystrom extension needs (DEVIATION 2860)")
    var training = PythonObject(None)
    if not precomputed:
        training = admit_family(source, 2, "fitted training rows", context).array
    var eigenvalues = admit_family(values_source, 1, "fitted eigenvalues", context).array
    var eigenvectors = admit_family(py_get(fields, PythonObject("_pd_eigenvectors"), PythonObject(None)), 2,
        "fitted eigenvectors", context).array
    var diagonal = admit_family(py_get(fields, PythonObject("_pd_diag"), PythonObject(None)), 1,
        "fitted degree diagonal", context).array
    var centroids = admit_family(py_get(fields, PythonObject("_pd_centroids"), PythonObject(None)), 2,
        "fitted centroids", context).array
    var n = Int(py=diagonal.shape[0]) if precomputed else Int(py=training.shape[0])
    var d = 0 if precomputed else Int(py=training.shape[1])
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
