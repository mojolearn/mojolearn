# SPDX-License-Identifier: Apache-2.0
# KMeans' public methods, one native call each (python/mojolearn/cluster.py:
# `fit`, `predict`, `transform`, `fit_predict`, `fit_transform`). The shell
# hands over the binding `_bind` resolved for the model's numeric mode; this
# owner reads the parameters, makes the refusals the Python method made (same
# wording, same order), allocates the outputs, executes typed native requests
# through the admitted compiled provider, and stores the fitted attributes.
# Numerical kernels, fused operations and their inputs remain unchanged.
# Lane migrate-classical-b2: `fit_transform` uploads X once
# (`kmeans_fit_transform`), and `predict` / `transform` read the fitted
# centroids from a device copy minted at the first query after a fit
# (`kmeans_centers_prepare`, cluster/resident_centers.mojo) instead of
# uploading them per call; a binding without those doors (the CPU host
# binding) takes the per-call entries as before.
# The PACK half of the former `bindings/kmeans_native.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/kmeans_native_unpack.mojo`.
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.admitted_family_door import (
    admit_family, admit_family_weights_array, admit_family_input, admit_family_cast, admitted_family_x,
)
from providers.python.array_metadata_pyapi_door import ArrayPin, array_python_shape
from bindings.admitted_input_unpack import AdmittedX, admit_x_unrecorded, admitted_finite_refuse
from bindings.device_rows_door import door_is_device_rows, door_rows_refuse
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields, py_setitem
from bindings.classical_arrays_door import classical_address
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.py_cpython_table import cpython_table
from bindings.host_inference_scalars_door import host_inference_scalar
from bindings.native_numpy_metadata_unpack import native_numpy_dtype
from bindings.family_common_unpack import (
    FAMILY_F32P,
    family_repr,
    family_str,
)
from bindings.family_common_pack import (
    family_call,
    family_array,
    family_field,
)
from bindings.classical_arrays_door import classical_is_f32, classical_c_order
from core.host_finite_scan import host_all_finite_f32
from bindings.p3nc_fitted_freeze_unpack import fitted_is_freezable, fitted_freeze
from providers.python.family_public_lifecycle_pyapi_door import family_lifecycle_begin, family_lifecycle_finish
from providers.python.family_public_lifecycle_run import FamilyLifecycle
from bindings.operation_names_kmeans import (
    KMEANS_OPN_FAMILY_KMEANS_FIT, KMEANS_OPN_FAMILY_KMEANS_FIT_PREDICT, KMEANS_OPN_FAMILY_KMEANS_FIT_TRANSFORM,
    KMEANS_OPN_FAMILY_KMEANS_PREDICT, KMEANS_OPN_FAMILY_KMEANS_TRANSFORM, KMEANS_OPN_FAMILY_KMEANS_GETSTATE,
    KMEANS_OPN_FAMILY_KMEANS_SETSTATE, KMEANS_OPN_FAMILY_KMEANS_SCORE)
from providers.python.neighbors_public_lifecycle_pyapi_door import neighbors_public_end, neighbors_public_refusal
from bindings.py_optional import py_or_none, py_present
from core.kmeans_native_api import (
    KMeansFoldStore, KMeansFitRequest, KMeansQueryRequest, KMeansScoreRequest,
    KMEANS_CAP_JOINT, KMEANS_CAP_RESIDENT, KMEANS_CAP_FOLD, KMEANS_CAP_HOST,
)
from bindings.kmeans_descriptor_boundary_door import (
    kmeans_connection,
    kmeans_run_fit,
    kmeans_run_query,
    kmeans_run_score,
    kmeans_integer,
)
from x_cluster.kmeans_descriptor_boundary_run import kmeans_capabilities
from bindings.kmeans_owner_boundary_door import (
    kmeans_owner_matching, kmeans_owner_handle, kmeans_owner_prepare,
    kmeans_owner_getstate, kmeans_owner_setstate,
)

from bindings.kmeans_native_unpack import (
    _INIT_ARRAY,
    KMeansFitInit,
    KMeansFitScalars,
    kmeans_fit_init,
    kmeans_fit_scalars,
    _HOLDER,
    _KMEANS_EST,
    _kmeans_admitted,
    _kmeans_centers_ready,
    _kmeans_conversion_scalar,
    _kmeans_host_x,
    _kmeans_is_name,
    _kmeans_metric,
    _kmeans_release,
    _kmeans_x,
)
from bindings.cluster_keys_unpack import cluster_keys
from bindings.py_balanced import py_item


def _kmeans_held(model: PythonObject, fields: PythonObject, binding: PythonObject,
                 centers: PythonObject, k: Int, d: Int, connection: Int) raises -> PythonObject:
    var owner = py_get(fields, _HOLDER, PythonObject(None))
    return owner if kmeans_owner_matching(owner, model, centers, binding, connection, k, d) else PythonObject(None)


def _kmeans_resident(model: PythonObject, binding: PythonObject, centers: PythonObject,
    c: PythonObject, k: Int, d: Int, connection: Int, context: PythonObject) raises -> PythonObject:
    if (kmeans_capabilities(connection) & KMEANS_CAP_RESIDENT) == 0:
        return PythonObject(None)
    var fields = py_fields(model)
    var held = _kmeans_held(model, fields, binding, centers, k, d, connection)
    if held is not None:
        return held
    # Writable/rebound caller arrays still use per-query uploads. Only the
    # same freezable Array owner can cache its already-admitted device copy.
    if not fitted_is_freezable(centers, context):
        return PythonObject(None)
    _ = fitted_freeze(centers, context)
    _kmeans_release(model)
    var owner = kmeans_owner_prepare(model, centers, binding, connection,
        Int(py=classical_address(c)), k, d)
    py_setitem(fields, _HOLDER, owner)
    return owner


def _kmeans_fit(model: PythonObject, args: PythonObject, context: PythonObject,
                transform: Bool = False) raises -> PythonObject:
    """`KMeans.fit(X, sample_weight)`; returns the fitted labels, or with
    `transform` the distances of the fitted rows to the final centroids
    (`KMeans.fit_transform`): one `kmeans_fit_transform` call, X uploaded
    once, where the binding has that door; otherwise the fit and then the
    transform, as before."""
    classical_connect_numpy(context)
    var b = native_loaded_module("builtins")
    var binding = args[0]
    var source = args[1]
    var sample_weight = args[2]
    var fields = py_fields(model)
    var nk = cluster_keys()
    # UNPACK, first stage: init, metric and oversampling, typed.
    var settings = kmeans_fit_init(fields, b, context)
    var k_obj = settings.clusters
    var init_code = settings.init_code
    var metric_code = settings.metric_code
    var x = _kmeans_x(source, context)
    var connection = kmeans_connection(binding)
    var capabilities = kmeans_capabilities(connection)
    # THE DOOR'S WEIGHTED FOLD (lane vdf-scope-cluster3): a weighted DeviceRows fold stays on the device too;
    # its fold-length sample_weight (already in fold order) is uploaded by the fold entry (KMeansFoldStore
    # .host_weights), X's rows are gathered there. The host gather of X is only for an image with no fold entry.
    var fold_fit = x.store.active() and (capabilities & KMEANS_CAP_FOLD) != 0
    if x.store.active() and not fold_fit:
        x = _kmeans_host_x(source, x, context)
    var n = x.rows
    var d = x.cols
    host_inference_scalar(k_obj, context)
    if Bool(py=family_call(native_loaded_module("operator").gt, k_obj, PythonObject(n))):
        raise Error("mojolearn: n_clusters=" + family_str(k_obj) + " exceeds n_samples=" + String(n))
    var k = Int(py=family_call(b.int, k_obj))
    var centers = family_array(Python.tuple(k, d), "<f4", context)
    if init_code == _INIT_ARRAY:
        if settings.init_centroids is None:
            raise Error("mojolearn: init='array' needs init_centroids")
        # The one admission path in its two steps: the rank and emptiness
        # refusals on the metadata, the shape refusal, then the cast (or the
        # borrow) with the finiteness verdict (family_view deleted).
        var pin = ArrayPin()
        var snapshot = admit_family_input(settings.init_centroids, 2, "init_centroids", context, pin)
        if snapshot.shape[0] != k or snapshot.shape[1] != d:
            raise Error("mojolearn: init_centroids must be (" + family_str(k_obj) + ", " + String(d)
                        + "), got " + family_repr(array_python_shape(snapshot)))
        var c0 = PythonObject(None)
        admit_family_cast(snapshot, pin, "init_centroids", context, True,
                          "Input init_centroids contains NaN or infinity (" + String(_KMEANS_EST) + ").")
        c0 = pin.keep
        # A COPY, on purpose: the kernel writes the centroids in place and
        # `c0` may borrow the caller's array.
        unsafe_memcpy(dest=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(centers))),
               src=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(c0))), count=k * d)
        _ = centers  # keep the Python object alive through the native call above
        _ = c0  # keep the Python object alive through the native call above
        _ = c0^
        _ = snapshot^
        _ = pin^
    # int32 cluster ids: the kernel writes uint32 ids below 2**31, the same
    # bytes, into the array the model keeps (DEVIATION 2672).
    var labels = family_array(Python.tuple(n), "<i4", context)
    var n_weights = 0
    var weights = x.array
    if sample_weight is not None:
        weights = admit_family_weights_array(sample_weight, n, True, context)
        n_weights = n
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
    var distances = Optional[PythonObject]()
    var joint = transform and not fold_fit and (capabilities & KMEANS_CAP_JOINT) != 0
    if joint:
        distances = family_array(Python.tuple(n, k), "<f4", context)
    # UNPACK, second stage: the scalar settings in their original admission
    # order, narrowed to machine words (host integer slots keep index
    # semantics). The joint distances were allocated between the reads and
    # the narrowing; they are allocated first now (no refusal depends on it).
    var scalars = kmeans_fit_scalars(fields, k_obj, (capabilities & KMEANS_CAP_HOST) != 0, b, context)
    var fold = KMeansFoldStore()
    if fold_fit:
        fold = KMeansFoldStore(x.store.base_id, x.store.base_rows, x.store.index_id, -1, n_weights != 0)
    var request = KMeansFitRequest(x.address, Int(py=classical_address(centers)),
        Int(py=classical_address(labels)), Int(py=classical_address(weights)) if n_weights != 0 else 0,
        Int(py=classical_address(distances.value())) if joint else 0,
        n, d, scalars.clusters, n_weights, scalars.iterations, scalars.seed, scalars.restarts,
        init_code, metric_code, not x.finite, scalars.tolerance, settings.oversampling, fold)
    var result = kmeans_run_fit(connection, request)
    _ = labels
    _ = centers
    _ = distances
    _ = weights
    if fold_fit:
        door_rows_refuse(result.finite, result.range_error, _KMEANS_EST, x.store.base_rows)
    elif not x.finite:
        admitted_finite_refuse(result.finite, _KMEANS_EST)
    _ = x^
    _ = weights^
    # p3-neighbors-cluster: the fitted centroids are read-only from fit on,
    # so no NumPy view of them taken before the first query is writeable
    # (the resident device copy is keyed by identity).
    _ = fitted_freeze(centers, context)
    py_setitem(fields, nk[].k_cluster_centers_, centers)
    py_setitem(fields, nk[].k_labels_, labels)
    py_setitem(fields, nk[].k_inertia_, PythonObject(result.inertia))
    py_setitem(fields, nk[].k_n_iter_, PythonObject(result.iterations))
    # `fit_loop_` (lane ab-loop-mode): (Lloyd iterations summed over the
    # restarts, loop milliseconds), built from the fit's FitLoop; not saved
    # in the archive, so a loaded model raises AttributeError for it.
    py_setitem(fields, nk[].k_fit_loop_,
        Python.tuple(result.loop.iterations, result.loop.milliseconds()))
    py_setitem(fields, nk[].k_sum_scale_, PythonObject(result.sum_scale))
    py_setitem(fields, nk[].k_weight_scale_, PythonObject(result.weight_scale))
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(d))
    # The last fit's resident centroids are released; the next query
    # uploads the new ones once (`_kmeans_resident`).
    _kmeans_release(model)
    if transform:
        if not distances:
            return _kmeans_inference(model, binding, source, True, context)
        return py_or_none(distances)
    return labels


comptime _QUERY_PREDICT = 0
comptime _QUERY_TRANSFORM = 1
comptime _QUERY_SCORE = 2


def _kmeans_inference(model: PythonObject, binding: PythonObject, source: PythonObject, transform: Bool,
                      context: PythonObject, admitted: Bool = False) raises -> PythonObject:
    """`KMeans.predict` (labels) or `KMeans.transform` (distances)."""
    return _kmeans_query(model, binding, source, _QUERY_TRANSFORM if transform else _QUERY_PREDICT,
                         PythonObject(None), context, admitted)


def _kmeans_query(model: PythonObject, binding: PythonObject, source: PythonObject, query: Int,
                  sample_weight: PythonObject, context: PythonObject, admitted: Bool) raises -> PythonObject:
    """`KMeans.predict` (labels), `KMeans.transform` (distances) or
    `KMeans.score(X, sample_weight)` (lane mf-kmeans5: scikit-learn's
    `-inertia` of X on `cluster_centers_`, a Python float; the image's
    KMEANS_SCORE op, the fit's post-loop cost, on the same centers homes as
    the queries; `sample_weight` admitted as the fit admits it).

    Perf (lane perf2-cluster): with `admitted`, `source` is the snapshot the
    public lifecycle admitted and cast (a 2-D, non-empty float32 C-order
    array), so X is only scanned for the by-name refusal here, not admitted
    and cast a second time; the scan reads the same float32 words the cast
    made, so the refusal fires for exactly the same inputs. The fitted
    centroids are admitted only when a device copy is minted or the binding
    has no resident door: a resident hit (same `cluster_centers_` object,
    binding and shape; a record is held only for a read-only Array)
    reuses the copy without uploading them. On the host column a
    ready float32 C-order `cluster_centers_` is read in place after one
    native finite scan."""
    classical_connect_numpy(context)
    var fields = py_fields(model)
    var centers = py_get(fields, PythonObject("cluster_centers_"), PythonObject(None))
    if centers is None:
        raise Error("RuntimeError: this estimator is not fitted yet")
    var metric_code = _kmeans_metric(fields, context)
    # Lane a7d: no host scan of X. The admitted snapshot (or a borrowed X)
    # is tested by the kernel entry over its own upload (finite flag in
    # params, the status word returned); a resident fold is gathered and
    # tested on the device by the fold entry.
    var x: AdmittedX
    if admitted and _kmeans_admitted(source, context):
        x = AdmittedX(source, source, Int(py=classical_address(source)), Int(py=source.shape[0]),
                      Int(py=source.shape[1]), 2, True, False)
    else:
        x = _kmeans_x(source, context)
    var connection = kmeans_connection(binding)
    var capabilities = kmeans_capabilities(connection)
    # A weighted fold score stays on the device as a weighted fold fit does
    # (lane vdf-scope-cluster3: the fold-length weights uploaded by the fold entry).
    var fold_query = x.store.active() and (capabilities & KMEANS_CAP_FOLD) != 0
    if x.store.active() and not fold_query:
        x = _kmeans_host_x(source, x, context)
    var n = x.rows
    var d = x.cols
    var shape = centers.shape
    var dc = Int(py=shape[1])
    if d != dc:
        raise Error("mojolearn: X has " + String(d) + " features, KMeans was fitted with " + String(dc))
    var k = Int(py=shape[0])
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
    var c = Optional[PythonObject]()
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
    var resident = Optional[PythonObject]()
    var resolved = False
    if _kmeans_centers_ready(centers, context):
        # Perf + correctness (lane pass2-cluster-nbr, hazard H3): the
        # centroids' own float32 words are read in place. A resident hit
        # needs the same object, binding and shape; records are held only
        # for a read-only Array (p3-neighbors-cluster, replacing the per-query
        # SHA-256), so an in-place edit raises instead of serving stale words.
        var held = _kmeans_held(model, fields, binding, centers, k, d, connection)
        if held is not None:
            # Minted only for this read-only Array (`_kmeans_resident`).
            resident = py_present(held)
            resolved = True
        elif (capabilities & KMEANS_CAP_RESIDENT) == 0:
            # Host column (no resident door): the admission of a ready
            # array is its finite scan; the same refusal as the cast's.
            if not host_all_finite_f32(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(centers))), k * d):
                raise Error("Input cluster_centers_ contains NaN or infinity")
            c = py_present(centers)
            resolved = True
        _ = centers  # keep the Python object alive through the native passes above
    if not resolved:
        c = admit_family(centers, 2, "cluster_centers_", context, finite=True).array
        k = Int(py=c.value().shape[0])
        resident = py_present(_kmeans_resident(model, binding, centers, c.value(), k, d, connection, context))
    var transform = query == _QUERY_TRANSFORM
    var destination = PythonObject(None)
    var weights = PythonObject(None)
    var n_weights = 0
    if query == _QUERY_SCORE:
        if sample_weight is not None:
            weights = admit_family_weights_array(sample_weight, n, True, context)
            n_weights = n
    elif transform:
        destination = family_array(Python.tuple(n, k), "<f4", context)
    else:
        destination = family_array(Python.tuple(n), "<i4", context)
    var fold = KMeansFoldStore()
    if fold_query:
        # Fold kernels run on the store context and retain their existing
        # host-centroid upload, even when another context caches the centers.
        if not c:
            c = admit_family(centers, 2, "cluster_centers_", context, finite=True).array
        fold = KMeansFoldStore(x.store.base_id, x.store.base_rows, x.store.index_id, -1, n_weights != 0)
    var handle = kmeans_owner_handle(resident.value()) if Bool(resident) and not fold_query else -1
    if query == _QUERY_SCORE:
        var score_request = KMeansScoreRequest(x.address,
            Int(py=classical_address(c.value())) if Bool(c) else 0,
            Int(py=classical_address(weights)) if n_weights != 0 else 0, n, d, k, n_weights, metric_code,
            not x.finite, handle, fold)
        var scored = kmeans_run_score(connection, score_request)
        # In-flight scores pin the native owner and the admitted words, as a query.
        _ = resident
        _ = centers
        _ = c
        _ = weights
        if fold_query:
            door_rows_refuse(scored.finite, scored.range_error, _KMEANS_EST, x.store.base_rows)
        elif not x.finite:
            admitted_finite_refuse(scored.finite, _KMEANS_EST)
        _ = x^
        _ = c^
        _ = weights^
        return PythonObject(-scored.inertia)
    var request = KMeansQueryRequest(x.address,
        Int(py=classical_address(c.value())) if Bool(c) else 0,
        Int(py=classical_address(destination)), n, d, k, metric_code,
        not x.finite, transform, handle, fold)
    var result = kmeans_run_query(connection, request)
    # In-flight queries pin the exact native owner even if another call
    # replaces the model's centroid field or resident owner while GIL is free.
    _ = resident
    _ = centers
    _ = c
    if fold_query:
        door_rows_refuse(result.finite, result.range_error, _KMEANS_EST, x.store.base_rows)
    elif not x.finite:
        admitted_finite_refuse(result.finite, _KMEANS_EST)
    _ = x^
    _ = c^
    return destination


def kmeans_unknown_operation(operation: String) raises:
    """A `family_kmeans_*` name the operation table does not hold (the shells
    pass table names only): the text the name chain raised."""
    raise Error("unknown KMeans operation " + operation)


def _kmeans_lifecycle(operation: Int) -> FamilyLifecycle:
    """The lifecycle of the KMeans public operations (lane bw2-cluster: moved
    from the shared name chain of
    providers/python/family_public_lifecycle_run.mojo, same flags; the pickle
    state runs outside it). A fit's X reaches the owner as the caller passed
    it (array init receives the rows). Perf (lane perf2-cluster): predict and
    transform take the admitted float32 snapshot, so X is admitted and cast
    once per query, not twice. Lane a7d: no scan here; the owner's kernel
    entry returns X's finite status word over its own upload, and a resident
    fold reaches the owner as the DeviceRows for its device-X entries
    (`device_x`)."""
    if operation == KMEANS_OPN_FAMILY_KMEANS_FIT:
        return FamilyLifecycle("fit", replace=False, device_x=True)
    if operation == KMEANS_OPN_FAMILY_KMEANS_FIT_PREDICT:
        return FamilyLifecycle("fit_predict", replace=False, device_x=True)
    if operation == KMEANS_OPN_FAMILY_KMEANS_FIT_TRANSFORM:
        return FamilyLifecycle("fit_transform", replace=False, device_x=True)
    if operation == KMEANS_OPN_FAMILY_KMEANS_PREDICT:
        return FamilyLifecycle("predict", device_x=True)
    if operation == KMEANS_OPN_FAMILY_KMEANS_TRANSFORM:
        return FamilyLifecycle("transform", device_x=True)
    if operation == KMEANS_OPN_FAMILY_KMEANS_SCORE:
        return FamilyLifecycle("score", device_x=True)
    return FamilyLifecycle("")


def _kmeans_operation_route(model: PythonObject, operation: Int, args: PythonObject,
                            context: PythonObject, admitted: Bool = False) raises -> PythonObject:
    """The KMeans public operations: args[0] is the resolved binding.
    `admitted`: the public lifecycle replaced X (args[1]) by its admitted
    float32 snapshot. `operation` is the leaf code of
    bindings/operation_names_kmeans.mojo (lane bw2-cluster)."""
    if operation == KMEANS_OPN_FAMILY_KMEANS_FIT:
        _ = _kmeans_fit(model, args, context)
        return model
    if operation == KMEANS_OPN_FAMILY_KMEANS_FIT_PREDICT:
        return _kmeans_fit(model, args, context)
    if operation == KMEANS_OPN_FAMILY_KMEANS_FIT_TRANSFORM:
        return _kmeans_fit(model, args, context, True)
    if operation == KMEANS_OPN_FAMILY_KMEANS_GETSTATE:
        # A pickle or a deepcopy carries no device handle: it is meaningful
        # only in the process and registry that minted it. The copy uploads
        # its own centroids at its first query.
        return kmeans_owner_getstate(model)
    if operation == KMEANS_OPN_FAMILY_KMEANS_SETSTATE:
        kmeans_owner_setstate(model, args[0])
        return PythonObject(None)
    if operation == KMEANS_OPN_FAMILY_KMEANS_PREDICT:
        return _kmeans_inference(model, args[0], args[1], False, context, admitted)
    if operation == KMEANS_OPN_FAMILY_KMEANS_TRANSFORM:
        return _kmeans_inference(model, args[0], args[1], True, context, admitted)
    if operation == KMEANS_OPN_FAMILY_KMEANS_SCORE:
        # args: (binding, X, sample_weight); KMeans.score's y is not read.
        return _kmeans_query(model, args[0], args[1], _QUERY_SCORE, args[2], context, admitted)
    raise Error("unknown KMeans operation code " + String(operation))


def kmeans_operation(model: PythonObject, operation: Int, args: PythonObject,
                     context: PythonObject) raises -> PythonObject:
    """The public operation: the estimators that formerly took the `_mode`
    class-creation wrappers run their whole lifecycle around the owner
    (providers/python/family_public_lifecycle_pyapi.mojo); every other
    operation goes straight to the owner. `operation` is the leaf code of
    bindings/operation_names_kmeans.mojo (lane bw2-cluster: the lifecycle is
    an integer switch; it walked the shared chain of up to 22 String
    compares, then up to seven more here)."""
    var lifecycle = _kmeans_lifecycle(operation)
    if lifecycle.method.byte_length() == 0:
        return _kmeans_operation_route(model, operation, args, context)
    # The lifecycle's operation parameter is not read (its text is the spec's method).
    var call = family_lifecycle_begin(model, "", lifecycle, args, 1, context)
    var result = PythonObject(None)
    try:
        result = _kmeans_operation_route(model, operation, call.args, context, lifecycle.replace)
    except error:
        var failure = neighbors_public_refusal(error, call, model)
        raise failure^
    finally:
        neighbors_public_end(model, call)
    return family_lifecycle_finish(model, lifecycle, result, call, context)
