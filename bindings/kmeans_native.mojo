# SPDX-License-Identifier: Apache-2.0
# KMeans' public methods, one native call each (python/mojolearn/cluster.py:
# `fit`, `predict`, `transform`, `fit_predict`, `fit_transform`). The shell
# hands over the binding `_bind` resolved for the model's numeric mode; this
# owner reads the parameters, makes the refusals the Python method made (same
# wording, same order), allocates the outputs, runs the binding's
# `kmeans_fit` / `kmeans_predict` / `kmeans_transform` with the argument lists
# `bindings/_mojolearn.mojo` documents, and stores the fitted attributes. The
# kernels and their inputs are unchanged, so the bits are too.
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
from bindings.admitted_input import AdmittedX, admit_x_unrecorded, admit_rows_host, admitted_finite_refuse
from bindings.device_rows_door import door_is_device_rows, door_rows_refuse
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields, py_setitem
from bindings.classical_arrays import classical_address
from bindings.classical_attributes import classical_connect_numpy
from bindings.optimizer_public_words import PYTHON_ERROR_PENDING, optimizer_python_error
from bindings.family_common import (
    FAMILY_F32P, family_repr, family_str, family_is_str, family_is_int, family_isinstance, family_type_name,
    family_call, family_call_kw, family_entry, family_array, family_nonfinite, family_field,
)
from bindings.classical_arrays import classical_is_f32, classical_c_order
from core.host_finite_scan import host_all_finite_f32
from bindings.p3nc_fitted_freeze import fitted_is_freezable, fitted_freeze
from providers.python.family_public_lifecycle_pyapi import family_lifecycle_spec, family_lifecycle_begin, family_lifecycle_finish
from providers.python.neighbors_public_lifecycle_pyapi import neighbors_public_end, neighbors_public_refusal
from bindings.cpython_table import cpython_table
from core.shared_slot import SharedSlot

comptime _INIT_ARRAY = 2
comptime _HOLDER = "_resident_holder"


def _kmeans_optional_entry(binding: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """The binding's compiled entry `name`, or None where the loaded binding
    has no such door (the CPU host binding refuses an absent name with
    ImportError, a module with AttributeError: both mean "no residency
    here", and the pending exception is discarded)."""
    if _kmeans_door_absent(binding, name):
        return PythonObject(None)
    try:
        return family_entry(binding, name, context)
    except error:
        if String(error) != PYTHON_ERROR_PENDING:
            raise error
        _ = optimizer_python_error(error)
        ref cpy = Python().cpython()
        cpython_table()[].PyErr_Clear()
        _kmeans_mark_absent(binding, name)
        return PythonObject(None)


def _kmeans_release(model: PythonObject) raises:
    """Drop the resident centroids, if held, through the minting binding's
    `kmeans_centers_release`. Quiet on a binding that cannot be reached any
    more and on a handle already released (a copied instance)."""
    var b = native_loaded_module("builtins")
    var holder = py_get(py_fields(model), PythonObject(_HOLDER), PythonObject(None))
    if holder is None or py_type(holder) is not b.dict:
        return
    var held = py_get(holder, PythonObject("index"), PythonObject(None))
    if held is None:
        return
    py_setitem(holder, "index", PythonObject(None))
    try:
        _ = py_call(py_call(b.getattr, held[2], PythonObject("kmeans_centers_release")), held[1])
    except:
        pass


struct _KMeansDoorless(Defaultable, Movable):
    # Bindings found to have no `kmeans_centers_prepare` door (the CPU host
    # binding): probing them raised and cleared an ImportError on every
    # query. Strong references, so an address is never reused while listed;
    # the binding modules live for the process anyway.
    # p3-neighbors-cluster: (binding, door name) pairs, so every optional
    # door (`kmeans_fit_transform` too, probed on every fit_transform) is
    # probed once per binding; the layout changed, so the global's name did.
    var bindings: List[PythonObject]
    var names: List[String]

    def __init__(out self):
        self.bindings = List[PythonObject]()
        self.names = List[String]()


comptime _KMEANS_DOORLESS = SharedSlot[_KMeansDoorless, "MojolearnKMeansDoorlessV3"]


def _kmeans_door_absent(binding: PythonObject, name: String) raises -> Bool:
    var g = _KMEANS_DOORLESS.get_or_create_ptr()
    for i in range(len(g[].bindings)):  # small-loop(bindings: the few connected KMeans bindings without a door)
        if g[].bindings[i] is binding and g[].names[i] == name:
            return True
    return False


def _kmeans_mark_absent(binding: PythonObject, name: String) raises:
    var g = _KMEANS_DOORLESS.get_or_create_ptr()
    if len(g[].bindings) < 64:
        g[].bindings.append(binding)
        g[].names.append(name)


def _kmeans_doorless(binding: PythonObject) raises -> Bool:
    return _kmeans_door_absent(binding, "kmeans_centers_prepare")


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


def _kmeans_held(fields: PythonObject, binding: PythonObject, centers: PythonObject, k: Int,
                 d: Int) raises -> PythonObject:
    """The held resident record `((centers, k, d), handle, binding)` when it
    was minted from this very `cluster_centers_` object (the record keeps a
    reference to it, so it cannot be replaced by a new array at the same
    address) by this binding at this shape, else None. A record is minted
    only for a read-only mojolearn Array (p3-neighbors-cluster, replacing the
    per-query SHA-256 of the centroid words): an in-place edit raises
    NumPy's read-only error instead of diverging from the device copy."""
    var b = native_loaded_module("builtins")
    var holder = py_get(fields, PythonObject(_HOLDER), PythonObject(None))
    if holder is None or py_type(holder) is not b.dict:
        return PythonObject(None)
    var held = py_get(holder, PythonObject("index"), PythonObject(None))
    if held is None:
        return PythonObject(None)
    if held[0][0] is centers and held[2] is binding and Int(py=held[0][1]) == k and Int(py=held[0][2]) == d:
        return held
    return PythonObject(None)


def _kmeans_resident(model: PythonObject, binding: PythonObject, centers: PythonObject, c: PythonObject,
                     k: Int, d: Int, context: PythonObject) raises -> PythonObject:
    """The handle of the device copy of the fitted centroids `c` (the
    admitted float32 words of `cluster_centers_`), uploaded at the first
    query after a fit and reused while `cluster_centers_` is the same object,
    the binding is the one that minted it (the object is read-only)
    (`_kmeans_held`); None where the binding has no resident door. The
    holder dictionary changes, never the model's `__dict__` keys after a
    fit."""
    var fields = py_fields(model)
    if _kmeans_doorless(binding):
        return PythonObject(None)
    var held = _kmeans_held(fields, binding, centers, k, d)
    if held is not None:
        return held[1]
    # p3-neighbors-cluster (hazard class A): the record is keyed by the
    # identity of `cluster_centers_`, never its values, so only storage that
    # can be made read-only is held on the device; any other (a
    # caller-assigned ndarray) takes the per-call upload, which reads the
    # current words every query.
    if not fitted_is_freezable(centers, context):
        return PythonObject(None)
    _ = fitted_freeze(centers, context)
    var prepare = _kmeans_optional_entry(binding, "kmeans_centers_prepare", context)
    if prepare is None:
        return PythonObject(None)
    var b = native_loaded_module("builtins")
    var holder = py_get(fields, PythonObject(_HOLDER), PythonObject(None))
    if holder is None or py_type(holder) is not b.dict:
        holder = Python.dict()
        py_setitem(fields, _HOLDER, holder)
    if py_get(holder, PythonObject("index"), PythonObject(None)) is not None:
        _kmeans_release(model)
    var handle = family_call(prepare, classical_address(c), Python.list(PythonObject(k), PythonObject(d)))
    # The minting binding travels with the handle: the release (here, and
    # the native finalizer when the instance is dropped) is its own
    # `kmeans_centers_release`.
    # The native finalizer (bindings/host_archive_lifetime.mojo) reads a
    # 3-tuple (key, handle, binding).
    py_setitem(holder, "index", Python.tuple(Python.tuple(centers, PythonObject(k), PythonObject(d)),
                                             handle, binding))
    return handle


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
    entry (sample weights, a callable init, the host binding) is gathered on
    the host out of model selection's host base, tested in the gather."""
    if x.store.active():
        return admit_x_unrecorded(source, _KMEANS_EST, context)
    return x.copy()


def _kmeans_metric(fields: PythonObject, context: PythonObject) raises -> Int:
    """`KMeans._metric_code`: a name from the table, or `int(metric)`."""
    var metric = family_field(fields, "metric")
    if family_is_str(metric):
        var table = context["family_kmeans_metrics"]
        if not py_contains(table, metric):
            raise Error("mojolearn: metric must be one of "
                        + family_repr(py_call(native_loaded_module("builtins").sorted, table))
                        + ", got " + family_repr(metric))
        return Int(py=table[metric])
    return Int(py=family_call(native_loaded_module("builtins").int, metric))


def _kmeans_callable_init(init: PythonObject, source: PythonObject, k: PythonObject,
                          random_state: PythonObject) raises -> PythonObject:
    """scikit-learn's callable `init` (`init(X, n_clusters,
    random_state=random_state)`), called once with the fit's rows: a NumPy
    array and a RandomState seeded by `random_state` when NumPy is
    installed, the input and the int otherwise."""
    var data = source
    var state = random_state
    var numpy = PythonObject(None)
    try:
        numpy = Python.import_module("numpy")
    except:
        numpy = PythonObject(None)
    if numpy is not None:
        data = family_call(numpy.asarray, source)
        state = family_call(numpy.random.RandomState, random_state)
    var keywords = Python.dict()
    py_setitem(keywords, "random_state", state)
    return family_call_kw(init, keywords, data, k)


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
    if family_is_str(init):
        var inits = context["family_kmeans_inits"]
        if not py_contains(inits, init):
            raise Error("mojolearn: init must be one of " + family_repr(py_call(b.sorted, inits))
                        + ", got " + family_repr(init))
        init_code = Int(py=inits[init])
    elif Bool(py=py_call(b.callable, init)):
        init_code = _INIT_ARRAY
        if door_is_device_rows(source, context):
            # scikit-learn's callable receives the fit's rows: a fold's host
            # rows (gathered once, tested; the fit then reads them).
            source = admit_rows_host(source, _KMEANS_EST, context)
        init_centroids = _kmeans_callable_init(init, source, k_obj, family_field(fields, "random_state"))
    elif family_is_int(init):
        init_code = Int(py=family_call(b.int, init))
    else:
        init_code = _INIT_ARRAY
        init_centroids = init
    var metric_code = _kmeans_metric(fields, context)
    var factor = family_field(fields, "oversampling_factor")
    if family_isinstance(factor, b.bool) or not family_isinstance(factor, Python.tuple(b.int, b.float)):
        raise Error("TypeError: mojolearn: oversampling_factor must be a float, got " + family_type_name(factor))
    var oversampling = family_call(b.float, factor)
    var x = _kmeans_x(source, context)
    var fold_fit = PythonObject(None)
    if x.store.active():
        fold_fit = _kmeans_optional_entry(binding, "kmeans_fit_fold", context)
        if fold_fit is None or sample_weight is not None:
            x = _kmeans_host_x(source, x, context)
            fold_fit = PythonObject(None)
    var n = x.rows
    var d = x.cols
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
    # ORDER MATCHES bindings/_mojolearn.mojo::kmeans_fit_binding: n_samples,
    # n_features, n_clusters, n_weights, max_iter, tol, seed, n_init, init,
    # metric, oversampling_factor.
    var params = Python.list()
    _ = py_call(params.append, PythonObject(n))
    _ = py_call(params.append, PythonObject(d))
    _ = py_call(params.append, k_obj)
    _ = py_call(params.append, PythonObject(n_weights))
    _ = py_call(params.append, family_field(fields, "max_iter"))
    _ = py_call(params.append, family_call(b.float, family_field(fields, "tol")))
    _ = py_call(params.append, family_call(b.int, family_field(fields, "random_state")))
    _ = py_call(params.append, family_field(fields, "n_init"))
    _ = py_call(params.append, PythonObject(init_code))
    _ = py_call(params.append, PythonObject(metric_code))
    _ = py_call(params.append, oversampling)
    var result: PythonObject
    var distances = PythonObject(None)
    var joint = PythonObject(None)
    if fold_fit is not None:
        # Lane a7d: the fold's rows gathered on the device out of model
        # selection's resident base (finite and range words returned).
        result = family_call(fold_fit, Python.list(x.store.base_id, x.store.base_rows, x.store.index_id),
                             classical_address(centers), classical_address(labels), params)
        _ = labels  # keep the Python object alive through the native call above
        _ = centers  # keep the Python object alive through the native call above
        door_rows_refuse(Int(py=result[4]), Int(py=result[5]), _KMEANS_EST, x.store.base_rows)
    else:
        # Lane a7d: a borrowed X is tested by the entry (finite flag, the
        # status word comes back as result[4]).
        var test = not x.finite
        if test:
            _ = py_call(params.append, PythonObject(1))
        if transform:
            joint = _kmeans_optional_entry(binding, "kmeans_fit_transform", context)
        if joint is not None:
            distances = family_array(Python.tuple(n, k), "<f4", context)
            result = family_call(joint, PythonObject(x.address), classical_address(centers), classical_address(labels),
                                 classical_address(weights), classical_address(distances), params)
            _ = labels  # keep the Python object alive through the native call above
            _ = centers  # keep the Python object alive through the native call above
            _ = distances  # keep the Python object alive through the native call above
            _ = weights  # keep the Python object alive through the native call above
        else:
            var fit = family_entry(binding, "kmeans_fit", context)
            result = family_call(fit, PythonObject(x.address), classical_address(centers), classical_address(labels),
                                 classical_address(weights), params)
            _ = labels  # keep the Python object alive through the native call above
            _ = centers  # keep the Python object alive through the native call above
            _ = weights  # keep the Python object alive through the native call above
        if test:
            admitted_finite_refuse(Int(py=result[4]), _KMEANS_EST)
    _ = x^
    _ = weights^
    # p3-neighbors-cluster: the fitted centroids are read-only from fit on,
    # so no NumPy view of them taken before the first query is writeable
    # (the resident device copy is keyed by identity).
    _ = fitted_freeze(centers, context)
    py_setitem(fields, "cluster_centers_", centers)
    py_setitem(fields, "labels_", labels)
    py_setitem(fields, "inertia_", result[0])
    py_setitem(fields, "n_iter_", result[1])
    py_setitem(fields, "sum_scale_", result[2])
    py_setitem(fields, "weight_scale_", result[3])
    py_setitem(fields, "n_features_in_", PythonObject(d))
    # The last fit's resident centroids are released; the next query
    # uploads the new ones once (`_kmeans_resident`).
    _kmeans_release(model)
    py_setitem(fields, _HOLDER, Python.dict())
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
    var fold_query = PythonObject(None)
    if x.store.active():
        fold_query = _kmeans_optional_entry(binding, "kmeans_transform_fold" if transform else "kmeans_predict_fold",
                                            context)
        if fold_query is None:
            x = _kmeans_host_x(source, x, context)
    var n = x.rows
    var d = x.cols
    var shape = centers.shape
    var dc = Int(py=shape[1])
    if d != dc:
        raise Error("mojolearn: X has " + String(d) + " features, KMeans was fitted with " + String(dc))
    var k = Int(py=shape[0])
    var c = PythonObject(None)
    var handle = PythonObject(None)
    var resolved = False
    if _kmeans_centers_ready(centers, context):
        # Perf + correctness (lane pass2-cluster-nbr, hazard H3): the
        # centroids' own float32 words are read in place. A resident hit
        # needs the same object, binding and shape; records are held only
        # for a read-only Array (p3-neighbors-cluster, replacing the per-query
        # SHA-256), so an in-place edit raises instead of serving stale words.
        var held = _kmeans_held(fields, binding, centers, k, d)
        if held is not None:
            # Minted only for this read-only Array (`_kmeans_resident`).
            handle = held[1]
            resolved = True
        elif _kmeans_doorless(binding):
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
        handle = _kmeans_resident(model, binding, centers, c, k, d, context)
    # ORDER MATCHES bindings/_mojolearn.mojo::kmeans_predict_binding and
    # kmeans_transform_binding: n_samples, n_features, n_clusters, metric.
    var params = Python.list(PythonObject(n), PythonObject(d), PythonObject(k), PythonObject(metric_code))
    var out: PythonObject
    if transform:
        out = family_array(Python.tuple(n, k), "<f4", context)
    else:
        out = family_array(Python.tuple(n), "<i4", context)
    if fold_query is not None:
        # The fold entry reads the host centroid words (a resident handle is
        # the per-binding copy of another context; the fold runs on the
        # store's own context).
        if c is None:
            c = admit_family(centers, 2, "cluster_centers_", context, finite=True).array
        var words = family_call(fold_query, Python.list(x.store.base_id, x.store.base_rows, x.store.index_id),
                                classical_address(c), classical_address(out), params)
        _ = c  # keep the Python object alive through the native call above
        door_rows_refuse(Int(py=words[0]), Int(py=words[1]), _KMEANS_EST, x.store.base_rows)
        _ = x^
        _ = c^
        return out
    var test = not x.finite
    if test:
        _ = py_call(params.append, PythonObject(1))
    var status: PythonObject
    if transform:
        if handle is not None:
            status = family_call(family_entry(binding, "kmeans_transform_resident", context), handle,
                                 PythonObject(x.address), classical_address(out), params)
        else:
            status = family_call(family_entry(binding, "kmeans_transform", context), PythonObject(x.address),
                                 classical_address(c), classical_address(out), params)
            _ = c  # keep the Python object alive through the native call above
    else:
        if handle is not None:
            status = family_call(family_entry(binding, "kmeans_predict_resident", context), handle,
                                 PythonObject(x.address), classical_address(out), params)
        else:
            status = family_call(family_entry(binding, "kmeans_predict", context), PythonObject(x.address),
                                 classical_address(c), classical_address(out), params)
            _ = c  # keep the Python object alive through the native call above
    if test:
        admitted_finite_refuse(Int(py=status), _KMEANS_EST)
    _ = x^
    _ = c^
    return out


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
        var state = py_call(native_loaded_module("builtins").dict, py_fields(model))
        _ = py_call(state.pop, PythonObject(_HOLDER), PythonObject(None))
        return state
    if operation == "family_kmeans_setstate":
        var fields = py_fields(model)
        _ = py_call(fields.update, args[0])
        py_setitem(fields, _HOLDER, Python.dict())
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
