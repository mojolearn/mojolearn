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
from std.memory import memcpy
from std.python import Python, PythonObject
from bindings.admitted_family import (
    admit_family, admit_family_weights, admit_family_input, admit_family_cast, admitted_family_x,
)
from bindings.admitted_input import AdmittedX, admit_x_unrecorded, admitted_finite_refuse
from bindings.device_rows_door import door_is_device_rows, door_rows_refuse
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields, py_setitem
from bindings.classical_arrays import classical_address
from bindings.classical_attributes import classical_connect_numpy
from bindings.cpython_table import cpython_table
from bindings.host_inference_scalars import host_inference_scalar
from bindings.native_numpy_metadata import native_numpy_dtype
from bindings.family_common import (
    FAMILY_F32P, family_repr, family_str,
    family_call, family_array, family_field,
)
from bindings.classical_arrays import classical_is_f32, classical_c_order
from core.host_finite_scan import host_all_finite_f32
from bindings.p3nc_fitted_freeze import fitted_is_freezable, fitted_freeze
from providers.python.family_public_lifecycle_pyapi import family_lifecycle_spec, family_lifecycle_begin, family_lifecycle_finish
from providers.python.neighbors_public_lifecycle_pyapi import neighbors_public_end, neighbors_public_refusal
from core.kmeans_native_api import (
    KMeansFoldStore, KMeansFitRequest, KMeansQueryRequest,
    KMEANS_CAP_JOINT, KMEANS_CAP_RESIDENT, KMEANS_CAP_FOLD, KMEANS_CAP_HOST,
)
from bindings.kmeans_descriptor_boundary import (
    kmeans_connection, kmeans_capabilities, kmeans_run_fit, kmeans_run_query, kmeans_integer,
)
from bindings.kmeans_owner_boundary import (
    kmeans_owner_matching, kmeans_owner_handle, kmeans_owner_prepare,
    kmeans_owner_getstate, kmeans_owner_setstate,
)

comptime _INIT_ARRAY = 2
comptime _HOLDER = "_resident_holder"


def _kmeans_release(model: PythonObject) raises:
    # Removing the model's reference retires the owner. A query still using
    # it pins the native handle until its own execution has finished.
    py_setitem(py_fields(model), _HOLDER, PythonObject(None))


def _kmeans_centers_ready(centers: PythonObject, context: PythonObject) raises -> Bool:
    """`cluster_centers_` is an exact Array or ndarray, 2-D, non-empty,
    float32 in C order: `as_f32_c` would return its own storage, so its
    words can be read (scanned) in place."""
    var cls = py_type(centers)
    if cls is not context["array_class"] and cls is not py_get(context, "ndarray"):
        return False
    if Int(py=centers.ndim) != 2 or Int(py=centers.size) <= 0:
        return False
    return classical_is_f32(centers, context) and classical_c_order(centers, context)


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


# TOMBSTONE: _kmeans_nonfinite / _kmeans_f32 (KMeans' own "X contains a NaN or an infinity; ... refused by name" wording and its host scan of X) deleted 2026-11-09 by lane a7d-device-x-decomp-cluster; code recoverable at 5b5490960d.
# Restore: experiments/removed/A7D_KMEANS_HOST_FINITE.patch. The kernel entry
# returns X's finite status word over its own upload; the refusal is the one
# admission wording (admitted_finite_refuse, "Input X contains NaN or infinity (KMeans).").
comptime _KMEANS_EST = "KMeans"


def _kmeans_x(source: PythonObject, context: PythonObject) raises -> AdmittedX:
    """X of a KMeans operation on the one admission path, without a host
    scan of a borrowed X (lane a7d): `finite` False means the kernel entry
    tests the words it uploads and returns the status word (the door then
    refuses through `admitted_finite_refuse`). A cast X is tested in the
    cast pass. A model-selection fold (`_arena_io.DeviceRows`) keeps its
    resident store ids (`store.active()`) for the device-X entries; where
    they do not resolve, its rows are gathered on the host and tested in
    the gather."""
    if door_is_device_rows(source, context):
        return admit_x_unrecorded(source, _KMEANS_EST, context, device_x=True)
    var snapshot = admit_family_input(source, 2, "X", context)
    var borrowed = snapshot.f32_c()
    admit_family_cast(snapshot, "X", context, not borrowed,
                      "Input X contains NaN or infinity (" + String(_KMEANS_EST) + ").")
    return admitted_family_x(snapshot, source, not borrowed)


def _kmeans_host_x(source: PythonObject, x: AdmittedX, context: PythonObject) raises -> AdmittedX:
    """`x` with host words: a resident fold whose operation has no device-X
    entry (sample weights, the host binding) is gathered on
    the host out of model selection's host base, tested in the gather."""
    if x.store.active():
        return admit_x_unrecorded(source, _KMEANS_EST, context)
    return x.copy()


def _kmeans_is_name(value: PythonObject, context: PythonObject) raises -> Bool:
    # isinstance may consult a rebound object's __class__. Inspect the real
    # type's C MRO instead, admitting only exact builtin/NumPy text storage
    # before dictionary hashing or equality can execute.
    var cls = py_type(value)
    var string_type = py_type(PythonObject(""))
    if cls is string_type:
        return True
    if cpython_table()[].PyType_IsSubtype(cls._obj_ptr, string_type._obj_ptr) == 0:
        return False
    if py_get(context, "ndarray") is not None and cls is py_call(native_numpy_dtype(context), "U").type:
        return True
    raise Error("TypeError: scalar model state requires native values without Python conversion hooks")


def _kmeans_conversion_scalar(value: PythonObject, context: PythonObject) raises:
    # tol/seed were converted before the compiled integer admission. Keep
    # their exact native NumPy text/byte and extended-real conversion forms.
    # Common builtin/numeric fields return through the shared admission.
    try:
        host_inference_scalar(value, context)
    except failure:
        if py_get(context, "ndarray") is not None:
            var cls = py_type(value)
            var constructor = native_numpy_dtype(context)
            var aliases = Python.tuple("U", "S", "q", "Q", "g")
            for i in range(len(aliases)):  # driver-loop: exact native scalar ABI aliases
                if cls is py_call(constructor, aliases[i]).type:
                    return
        raise failure^


def _kmeans_metric(fields: PythonObject, context: PythonObject) raises -> Int:
    """`KMeans._metric_code`: a name from the table, or `int(metric)`."""
    var metric = family_field(fields, "metric")
    if _kmeans_is_name(metric, context):
        var table = context["family_kmeans_metrics"]
        if not py_contains(table, metric):
            raise Error("mojolearn: metric must be one of "
                        + family_repr(py_call(native_loaded_module("builtins").sorted, table))
                        + ", got " + family_repr(metric))
        return Int(py=table[metric])
    _kmeans_conversion_scalar(metric, context)
    return Int(py=family_call(native_loaded_module("builtins").int, metric))


# TOMBSTONE: P1_KMEANS_CALLABLE_INIT (removed: the user's callable init plus numpy.asarray / RandomState called inside KMeans.fit) deleted 2026-11-09 by lane p1-python-callbacks-worst; code recoverable at e68654a757.
# Restore: experiments/removed/P1_KMEANS_CALLABLE_INIT.patch.
# P1 (audit A #2, audit B #3): _kmeans_callable_init removed; it imported
# NumPy, built numpy.random.RandomState and called the user's init(X, k,
# random_state=...) inside fit. A callable init is now refused before it is
# called (as MiniBatchKMeans already does, bindings/cluster_fit_a_native.mojo
# _init_array); the seeds of 'k-means++' and 'random' are drawn natively.


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
    var k_obj = family_field(fields, "n_clusters")
    var init = family_field(fields, "init")
    var init_centroids = family_field(fields, "init_centroids")
    var init_code: Int
    if _kmeans_is_name(init, context):
        var inits = context["family_kmeans_inits"]
        if not py_contains(inits, init):
            raise Error("mojolearn: init must be one of " + family_repr(py_call(b.sorted, inits))
                        + ", got " + family_repr(init))
        init_code = Int(py=inits[init])
    elif Bool(py=py_call(b.callable, init)):
        # Refused before it is called: a Python init would run Python code
        # inside the fit. PyCallable_Check through builtins.callable runs no
        # user code.
        raise Error("TypeError: Python init callbacks are not accepted; pass explicit initial centers "
                    + "(an array of shape (n_clusters, n_features)) or use 'k-means++' or 'random'")
    elif py_type(init) is b.int:
        init_code = Int(py=family_call(b.int, init))
    else:
        if (py_type(init) is not b.bool
                and cpython_table()[].PyType_IsSubtype(py_type(init)._obj_ptr, b.int._obj_ptr) != 0):
            raise Error("TypeError: scalar model state requires native values without Python conversion hooks")
        init_code = _INIT_ARRAY
        init_centroids = init
    var metric_code = _kmeans_metric(fields, context)
    var factor = family_field(fields, "oversampling_factor")
    var factor_type = py_type(factor)
    var factor_admitted = factor_type is b.int or factor_type is b.float
    if not factor_admitted and py_get(context, "ndarray") is not None:
        factor_admitted = factor_type is context["float64"].type
    if not factor_admitted:
        # Use the canonical type getset directly; a custom metaclass can
        # override __name__ even through type.__getattribute__.
        var name = String(py_call(b.type.__dict__["__name__"].__get__, factor_type))
        raise Error("TypeError: mojolearn: oversampling_factor must be a float, got " + name)
    var oversampling = family_call(b.float, factor)
    var x = _kmeans_x(source, context)
    var connection = kmeans_connection(binding)
    var capabilities = kmeans_capabilities(connection)
    var fold_fit = x.store.active() and (capabilities & KMEANS_CAP_FOLD) != 0 and sample_weight is None
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
        if init_centroids is None:
            raise Error("mojolearn: init='array' needs init_centroids")
        # The one admission path in its two steps: the rank and emptiness
        # refusals on the metadata, the shape refusal, then the cast (or the
        # borrow) with the finiteness verdict (family_view deleted).
        var snapshot = admit_family_input(init_centroids, 2, "init_centroids", context)
        if snapshot.shape[0] != k or snapshot.shape[1] != d:
            raise Error("mojolearn: init_centroids must be (" + family_str(k_obj) + ", " + String(d)
                        + "), got " + family_repr(snapshot.python_shape()))
        var c0 = PythonObject(None)
        admit_family_cast(snapshot, "init_centroids", context, True,
                          "Input init_centroids contains NaN or infinity (" + String(_KMEANS_EST) + ").")
        c0 = snapshot.keep
        # A COPY, on purpose: the kernel writes the centroids in place and
        # `c0` may borrow the caller's array.
        memcpy(dest=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(centers))),
               src=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(c0))), count=k * d)
        _ = centers  # keep the Python object alive through the native call above
        _ = c0  # keep the Python object alive through the native call above
        _ = c0^
        _ = snapshot^
    # int32 cluster ids: the kernel writes uint32 ids below 2**31, the same
    # bytes, into the array the model keeps (DEVIATION 2672).
    var labels = family_array(Python.tuple(n), "<i4", context)
    var n_weights = 0
    var weights = x.array
    if sample_weight is not None:
        weights = admit_family_weights(sample_weight, n, True, context).keep
        n_weights = n
    # Retain the original scalar admission order, then discard the former
    # Python parameter list before RUN. Host integer slots keep index semantics.
    var max_iter = family_field(fields, "max_iter")
    var tolerance = family_field(fields, "tol")
    _kmeans_conversion_scalar(tolerance, context)
    tolerance = family_call(b.float, tolerance)
    var seed = family_field(fields, "random_state")
    _kmeans_conversion_scalar(seed, context)
    seed = family_call(b.int, seed)
    var n_init = family_field(fields, "n_init")
    var distances = PythonObject(None)
    var joint = transform and not fold_fit and (capabilities & KMEANS_CAP_JOINT) != 0
    if joint:
        distances = family_array(Python.tuple(n, k), "<f4", context)
    var host = (capabilities & KMEANS_CAP_HOST) != 0
    var native_clusters = kmeans_integer(k_obj, host, context)
    var native_iterations = kmeans_integer(max_iter, host, context)
    var native_tolerance = Float64(py=tolerance)
    var native_seed = kmeans_integer(seed, host, context)
    var native_restarts = kmeans_integer(n_init, host, context)
    var native_oversampling = Float64(py=oversampling)
    var fold = KMeansFoldStore()
    if fold_fit:
        fold = KMeansFoldStore(x.store.base_id, x.store.base_rows, x.store.index_id)
    var request = KMeansFitRequest(x.address, Int(py=classical_address(centers)),
        Int(py=classical_address(labels)), 0 if fold_fit else Int(py=classical_address(weights)),
        Int(py=classical_address(distances)) if joint else 0,
        n, d, native_clusters, n_weights, native_iterations, native_seed, native_restarts,
        init_code, metric_code, not x.finite, native_tolerance, native_oversampling, fold)
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
    py_setitem(fields, "cluster_centers_", centers)
    py_setitem(fields, "labels_", labels)
    py_setitem(fields, "inertia_", PythonObject(result.inertia))
    py_setitem(fields, "n_iter_", PythonObject(result.iterations))
    py_setitem(fields, "sum_scale_", PythonObject(result.sum_scale))
    py_setitem(fields, "weight_scale_", PythonObject(result.weight_scale))
    py_setitem(fields, "n_features_in_", PythonObject(d))
    # The last fit's resident centroids are released; the next query
    # uploads the new ones once (`_kmeans_resident`).
    _kmeans_release(model)
    if transform:
        if distances is None:
            return _kmeans_inference(model, binding, source, True, context)
        return distances
    return labels


def _kmeans_admitted(source: PythonObject, context: PythonObject) raises -> Bool:
    """`source` is the public lifecycle's admitted dense snapshot: an exact
    Array or ndarray (a sparse X passes through unreplaced and is neither),
    float32 in C order."""
    var cls = py_type(source)
    if cls is not context["array_class"] and cls is not py_get(context, "ndarray"):
        return False
    return classical_is_f32(source, context) and classical_c_order(source, context)


def _kmeans_inference(model: PythonObject, binding: PythonObject, source: PythonObject, transform: Bool,
                      context: PythonObject, admitted: Bool = False) raises -> PythonObject:
    """`KMeans.predict` (labels) or `KMeans.transform` (distances).

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
    var c = PythonObject(None)
    var resident = PythonObject(None)
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
            resident = held
            resolved = True
        elif (capabilities & KMEANS_CAP_RESIDENT) == 0:
            # Host column (no resident door): the admission of a ready
            # array is its finite scan; the same refusal as the cast's.
            if not host_all_finite_f32(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(centers))), k * d):
                raise Error("Input cluster_centers_ contains NaN or infinity")
            c = centers
            resolved = True
        _ = centers  # keep the Python object alive through the native passes above
    if not resolved:
        c = admit_family(centers, 2, "cluster_centers_", context, finite=True).array
        k = Int(py=c.shape[0])
        resident = _kmeans_resident(model, binding, centers, c, k, d, connection, context)
    var destination: PythonObject
    if transform:
        destination = family_array(Python.tuple(n, k), "<f4", context)
    else:
        destination = family_array(Python.tuple(n), "<i4", context)
    var fold = KMeansFoldStore()
    if fold_query:
        # Fold kernels run on the store context and retain their existing
        # host-centroid upload, even when another context caches the centers.
        if c is None:
            c = admit_family(centers, 2, "cluster_centers_", context, finite=True).array
        fold = KMeansFoldStore(x.store.base_id, x.store.base_rows, x.store.index_id)
    var handle = kmeans_owner_handle(resident) if resident is not None and not fold_query else -1
    var request = KMeansQueryRequest(x.address,
        Int(py=classical_address(c)) if c is not None else 0,
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


def _kmeans_operation_route(model: PythonObject, operation: String, args: PythonObject,
                            context: PythonObject, admitted: Bool = False) raises -> PythonObject:
    """The KMeans public operations: args[0] is the resolved binding.
    `admitted`: the public lifecycle replaced X (args[1]) by its admitted
    float32 snapshot."""
    if operation == "family_kmeans_fit":
        _ = _kmeans_fit(model, args, context)
        return model
    if operation == "family_kmeans_fit_predict":
        return _kmeans_fit(model, args, context)
    if operation == "family_kmeans_fit_transform":
        return _kmeans_fit(model, args, context, True)
    if operation == "family_kmeans_getstate":
        # A pickle or a deepcopy carries no device handle: it is meaningful
        # only in the process and registry that minted it. The copy uploads
        # its own centroids at its first query.
        return kmeans_owner_getstate(model)
    if operation == "family_kmeans_setstate":
        kmeans_owner_setstate(model, args[0])
        return PythonObject(None)
    if operation == "family_kmeans_predict":
        return _kmeans_inference(model, args[0], args[1], False, context, admitted)
    if operation == "family_kmeans_transform":
        return _kmeans_inference(model, args[0], args[1], True, context, admitted)
    raise Error("unknown KMeans operation " + operation)


def kmeans_operation(model: PythonObject, operation: String, args: PythonObject,
                     context: PythonObject) raises -> PythonObject:
    """The public operation: the estimators that formerly took the `_mode`
    class-creation wrappers run their whole lifecycle around the owner
    (providers/python/family_public_lifecycle_pyapi.mojo); every other
    operation goes straight to the owner."""
    var lifecycle = family_lifecycle_spec(operation)
    if lifecycle.method.byte_length() == 0:
        return _kmeans_operation_route(model, operation, args, context)
    var call = family_lifecycle_begin(model, operation, lifecycle, args, 1, context)
    var result = PythonObject(None)
    try:
        result = _kmeans_operation_route(model, operation, call.args, context, lifecycle.replace)
    except error:
        var failure = neighbors_public_refusal(error, call, model)
        raise failure^
    finally:
        neighbors_public_end(model, call)
    return family_lifecycle_finish(model, lifecycle, result, call, context)
