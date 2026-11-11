# SPDX-License-Identifier: Apache-2.0
# The x_cluster estimators' inference methods, one native call each
# (python/mojolearn/_expansion_cluster.py): `_CentersMixin.predict`,
# `transform` and `score` (MiniBatchKMeans, BisectingKMeans), MeanShift
# `predict`, BisectingKMeans `predict` (the bisection tree descent) and
# AffinityPropagation `predict`. The shell hands over the binding `_bind`
# resolved; this owner makes the Python methods' refusals, calls the
# binding's `x_cluster_call(which, x, x_size, aux, aux_size, ints, floats)`
# with the same words, and makes the `array.array` results it returns the
# Arrays' stores (no second copy). Same entry, same inputs: the bits do not
# move. The fits of these classes are bindings/cluster_fit_a_native.mojo's.
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.admitted_family import admit_family
from bindings.py_balanced import py_call, py_type, py_get, py_fields
from bindings.classical_arrays import (
    classical_address, classical_wrap_array, classical_c_bytes, classical_is_f32, classical_c_order,
)
from core.admitted_weights import admitted_fill
from core.host_finite_scan import host_all_finite_f32
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_pandas import classical_admit_value
from bindings.host_inference_scalars import host_inference_scalar
from bindings.agglo_native_boundary import agglo_reference_labels, agglo_scalar_repr, agglo_hash_admitted
from core.agglo_native_state import agglo_metric_code
from bindings.native_loaded_module import native_loaded_module, NativeModules
from bindings.perf2_shared_fitted_field import native_fitted_get
from bindings.p3nc_fitted_freeze import fitted_is_freezable, fitted_freeze
from bindings.cluster_resident_table import (
    cres_variant_lookup, cres_variant_store,
)
from bindings.kmeans_variants_boundary import (
    variants_connection, variants_span, variants_pack_f32, variants_pack_i32,
)
from bindings.kmeans_variants_resident_boundary import VariantsResidentLease, variants_resident_prepare, variants_resident_retire
from bindings.native_buffer_address import native_buffer_address
from core.kmeans_variants_api import (
    VariantsSpan, VariantsResultSlot, VariantsQueryRequest, BisectPredictRequest, VARIANTS_CAP_RESIDENT,
)
from bindings.family_common import (
    FAMILY_F32P, FAMILY_I32P, family_repr, family_array, family_nonfinite,
    family_shape_repr,
)

# The query kinds of the GPU binding's `x_cluster_resident_query`
# (x_cluster/resident.mojo XC_RES_*).
comptime _XC_RES_PREDICT = 0
comptime _XC_RES_SCORE = 1
comptime _XC_RES_TRANSFORM = 2


def _xc_fitted(mut mods: NativeModules, model: PythonObject, attr: String) raises -> PythonObject:
    """`_XCluster._check_fitted`: the attribute, or the refusal."""
    # Package-internal read: the instance field without the public
    # attribute hook (bindings/perf2_shared_fitted_field.mojo).
    var value = native_fitted_get(mods, model, attr, PythonObject(None))
    if value is None:
        raise Error("RuntimeError: this " + String(py_type(model).__name__) + " is not fitted yet")
    return value


def _xc_f32_ready(source: PythonObject, context: PythonObject) raises -> Bool:
    """`source` is an exact Array or ndarray, 2-D, non-empty, float32 in C
    order: `as_f32_c(source, ndim=2)` would admit it and return its own
    storage (the test of bindings/mixture_native.mojo `_f32_ready`)."""
    var cls = py_type(source)
    if cls is not context["array_class"] and cls is not py_get(context, "ndarray"):
        return False
    if Int(py=source.ndim) != 2 or Int(py=source.size) <= 0:
        return False
    return classical_is_f32(source, context) and classical_c_order(source, context)


def _xc_input(mut mods: NativeModules, model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_XCluster._input_like_fit`: `_f32(X)` (as_f32_c, nonempty, finite)
    and the fitted feature count."""
    var x = PythonObject(None)
    if _xc_f32_ready(source, context):
        # Perf (lane pass2-cluster-nbr): the public door hands over its
        # admitted X (an exact float32 C-order Array or ndarray), which
        # `as_f32_c` would return as is. Only the finite refusal is repeated,
        # as one native scan of the same words, in the same wording.
        if not host_all_finite_f32(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(source))),
                                     Int(py=source.size)):
            raise Error("mojolearn: X contains NaN or infinity")
        _ = source  # keep the Python object alive through the native scan above
        x = source
    else:
        try:
            x = admit_family(source, 2, "X", context, finite=True).array
        except error:
            if family_nonfinite(error, "X"):
                raise Error("mojolearn: X contains NaN or infinity")
            raise Error(String(error))
    var d = Int(py=x.shape[1])
    var fitted = native_fitted_get(mods, model, "n_features_in_")
    if d != Int(py=fitted):
        raise Error("mojolearn: X has " + String(d) + " features, " + String(py_type(model).__name__)
                    + " was fitted with " + String(Int(py=fitted)))
    return x


def _xc_array(flat: PythonObject, shape: PythonObject, typestr: String, code: String,
              context: PythonObject) raises -> PythonObject:
    """`Array._from_flat(flat, shape, typestr)` without its copy: the
    entry returns each result as one `array.array` of the dtype's store code
    (providers/python/x_cluster_out_pyapi.mojo, one memcpy from the result), which becomes the
    Array's store as it is. Anything else is copied once, as `_from_flat` did.
    Same values either way."""
    var store = flat
    if py_type(flat) is not context["array_type"] or String(py=flat.typecode) != code:
        store = py_call(context["array_type"], PythonObject(code), flat)
    return classical_wrap_array(store, shape, typestr, "C", context)


def _centers_resident(model: PythonObject, binding: PythonObject, centers: PythonObject,
                      features: Int, context: PythonObject) raises -> Optional[VariantsResidentLease]:
    if not fitted_is_freezable(centers, context):
        return None
    if not _xc_f32_ready(centers, context) or Int(py=centers.shape[1]) != features:
        return None
    var clusters = Int(py=centers.shape[0])
    var dims: List[Int] = [clusters, features]
    var connection = variants_connection(binding)
    var held = cres_variant_lookup(model, binding, centers, connection.provider, dims)
    if held:
        return held^
    if (connection.capabilities & VARIANTS_CAP_RESIDENT) == 0:
        return None
    _ = fitted_freeze(centers, context)
    var lease = variants_resident_prepare(connection.driver, connection.provider, centers,
        Int(py=classical_address(centers)), clusters, features)
    # Table and caller each own a lease. Eviction or concurrent replacement
    # retires only the table's reference, never this active query's handle.
    cres_variant_store(model, binding, dims^, lease.copy(), clusters * features * 4)
    return lease^


def _centers_query(model: PythonObject, binding: PythonObject, x: PythonObject,
                   centers_field: PythonObject, kind: Int, context: PythonObject) raises -> PythonObject:
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var held = _centers_resident(model, binding, centers_field, d, context)
    var result = VariantsResultSlot()
    if held:
        var k = Int(py=centers_field.shape[0])
        var output = PythonObject(None)
        if kind == _XC_RES_PREDICT:
            output = family_array(Python.tuple(n), "<i4", context)
        elif kind == _XC_RES_TRANSFORM:
            output = family_array(Python.tuple(n, k), "<f4", context)
        var address = 0 if output is None else Int(py=classical_address(output))
        var source = variants_span(x)
        var total = Float64(0)
        try:
            with GILReleased(Python()):
                total = held.value().native[].driver.resident_query(held.value().native[].handle,
                    kind, source.address, n, k, d, address, result)
        finally:
            variants_resident_retire(held.take())
            _ = x
            _ = output
        return PythonObject(-total) if kind == _XC_RES_SCORE else output
    var centers = admit_family(centers_field, 2, "cluster_centers_", context).array
    var k = Int(py=centers.shape[0])
    var connection = variants_connection(binding)
    var center_span = variants_span(centers)
    var source_span = variants_span(x)
    var request = VariantsQueryRequest(source_span, center_span, n, k, d, kind == _XC_RES_SCORE)
    try:
        with GILReleased(Python()):
            connection.driver.query(request, kind == _XC_RES_TRANSFORM, result)
    finally:
        _ = connection.provider
        _ = centers
        _ = x
    # Pack only the public result. The old nested result envelope also copied
    # discarded distances/labels to Python for predict/score; those copies
    # have no observable owner and are unnecessary with typed result fields.
    if kind == _XC_RES_SCORE:
        return PythonObject(-result.storage[0].inertia)
    if kind == _XC_RES_TRANSFORM:
        return _xc_array(variants_pack_f32(result.storage[0].distances), Python.tuple(n, k), "<f4", "f", context)
    return _xc_array(variants_pack_i32(result.storage[0].labels), Python.tuple(n), "<i4", "i", context)


def _reference_queries(source: PythonObject, n_features: Int, where: String, context: PythonObject) raises -> PythonObject:
    """`density._check_queries`: float32 C order, the fitted feature count,
    at least one row, every value finite."""
    var q = admit_family(source, 2, "X", context).array
    var d = Int(py=q.shape[1])
    if d != n_features:
        raise Error("mojolearn " + where + ": X has " + String(d) + " features, the fit saw " + String(n_features))
    if Int(py=q.shape[0]) < 1:
        raise Error("mojolearn " + where + ": X has no rows; refused by name")
    if not host_all_finite_f32(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(q))), Int(py=q.size)):
        raise Error("mojolearn " + where + ": X contains a NaN or an infinity; a non-finite "
                    + "query has no distance order to take a nearest row from, so it is "
                    + "refused by name")
    _ = q  # keep the Python object alive through the native calls above
    return q


def _agglo_predict(model: PythonObject, binding: PythonObject, source: PythonObject,
                   context: PythonObject) raises -> PythonObject:
    """`AgglomerativeClustering.predict`: the nearest training row's cluster
    (single linkage, euclidean)."""
    var b = native_loaded_module("builtins")
    # One attribute read for `labels_`: the fitted check (`hasattr`) and the
    # value used below. A fresh sentinel keeps "absent" apart from a stored None.
    var mods = NativeModules()
    var absent = py_call(b.object)
    var labels = native_fitted_get(mods, model, "labels_", absent)
    if labels is absent:
        raise Error("mojolearn AgglomerativeClustering.predict: this instance is not "
                    + "fitted yet; call fit first")
    var linkage = native_fitted_get(mods, model, "linkage")
    var metric = native_fitted_get(mods, model, "metric")
    host_inference_scalar(linkage, context, True)
    host_inference_scalar(metric, context, True)
    # `_predicts`: single linkage under a euclidean metric name.
    var predicts = py_type(linkage) is b.str and String(linkage) == "single"
    if predicts:
        agglo_hash_admitted(metric)
        predicts = (agglo_metric_code(String(metric)) == -1) if Bool(py=py_call(b.isinstance, metric, b.str)) else False
    if not predicts:
        raise Error("NotImplementedError: mojolearn AgglomerativeClustering.predict: linkage=" + agglo_scalar_repr(linkage, context)
                    + ", metric=" + agglo_scalar_repr(metric, context) + " has no predict rule; DEVIATION 2740 is single "
                    + "linkage's own criterion (the nearest training row's cluster) under "
                    + "the euclidean metric only")
    var refs = native_fitted_get(mods, model, "_fit_X", PythonObject(None))
    if refs is None:
        raise Error("mojolearn AgglomerativeClustering.predict: prediction data was not "
                    + "stored. Fit with AgglomerativeClustering(prediction_data=True), "
                    + "which keeps the training rows this rule needs (DEVIATION 2740)")
    var feature_value = native_fitted_get(mods, model, "n_features_in_")
    host_inference_scalar(feature_value, context)
    var n_features = Int(py=feature_value)
    var q = _reference_queries(source, n_features, "AgglomerativeClustering.predict", context)
    classical_admit_value(refs, context)
    classical_admit_value(labels, context)
    # Historical internal entry: retain direct uploads without minting a holder.
    var result = agglo_reference_labels(py_fields(model), binding, refs, labels, q,
        Int(py=classical_address(refs)), Int(py=classical_address(labels)), Int(py=classical_address(q)),
        Int(py=refs.shape[0]), Int(py=q.shape[0]), n_features, context, False)
    _ = q^
    return result


def cluster_operation(model: PythonObject, operation: String, args: PythonObject,
                      context: PythonObject) raises -> PythonObject:
    """args[0] is the resolved binding, args[1] X."""
    classical_connect_numpy(context)
    var b = native_loaded_module("builtins")
    # One module handle for every fitted-field read of this call.
    var mods = NativeModules()
    var binding = args[0]
    if operation == "family_xc_agglo_predict":
        return _agglo_predict(model, binding, args[1], context)
    if (operation == "family_xc_variants_predict" or operation == "family_xc_variants_transform"
            or operation == "family_xc_variants_score"
            or operation == "family_xc_transform" or operation == "family_xc_score"):
        var centers = _xc_fitted(mods, model, "cluster_centers_")
        var x = _xc_input(mods, model, args[1], context)
        var kind = _XC_RES_PREDICT if operation == "family_xc_variants_predict" else (
            _XC_RES_TRANSFORM if (operation == "family_xc_variants_transform" or operation == "family_xc_transform") else _XC_RES_SCORE)
        return _centers_query(model, binding, x, centers, kind, context)
    if operation == "family_xc_predict":
        var centers = _xc_fitted(mods, model, "cluster_centers_")
        var x = _xc_input(mods, model, args[1], context)
        return _centers_query(model, binding, x, centers, _XC_RES_PREDICT, context)
    if operation == "family_xc_bisect_predict":
        _ = _xc_fitted(mods, model, "cluster_centers_")
        var x = _xc_input(mods, model, args[1], context)
        var n = Int(py=x.shape[0])
        var d = Int(py=x.shape[1])
        var fields = py_fields(model)
        var node_source = fields["_tree_nodes"]
        # Preserve the existing raw C-order byte interpretation, including
        # rebound storage. The extra array('i') and boxed Int list disappear.
        var nodes = classical_c_bytes(node_source, "_tree_nodes", context)
        var byte_count = len(nodes)
        if byte_count % 4 != 0:
            raise Error("ValueError: bytes length not a multiple of item size")
        var centers = fields["_tree_centers"]
        var connection = variants_connection(binding)
        var center_span = variants_span(centers)
        var source_span = variants_span(x)
        var node_span = VariantsSpan(native_buffer_address(nodes) if byte_count else 0, byte_count // 4)
        var request = BisectPredictRequest(source_span, center_span, node_span, n, d)
        var result = VariantsResultSlot()
        try:
            with GILReleased(Python()):
                connection.driver.bisect_predict(request, result)
        finally:
            _ = connection.provider
            _ = node_source
            _ = nodes
            _ = centers
            _ = x
        return _xc_array(variants_pack_i32(result.storage[0].labels), Python.tuple(n), "<i4", "i", context)
    if operation == "family_xc_ap_predict":
        _ = _xc_fitted(mods, model, "labels_")
        var affinity = native_fitted_get(mods, model, "affinity")
        if py_type(affinity) is b.str and String(affinity) == "precomputed":
            raise Error("Predict method is not supported when affinity='precomputed'.")
        var x = _xc_input(mods, model, args[1], context)
        var centers = native_fitted_get(mods, model, "cluster_centers_")
        if Int(py=centers.shape[0]) == 0:
            var labels = family_array(Python.tuple(Int(py=x.shape[0])), "<i4", context)
            var words = FAMILY_I32P(unsafe_from_address=Int(py=classical_address(labels)))
            _ = labels  # keep the Python object alive through the native call above
            admitted_fill[DType.int32](words, Int(py=x.shape[0]), Int32(-1))
            _ = labels  # keep the Python object alive through the native call above
            return labels
        return _centers_query(model, binding, x, centers, _XC_RES_PREDICT, context)
    raise Error("RuntimeError: unknown x_cluster operation " + operation)
