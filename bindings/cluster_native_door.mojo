# SPDX-License-Identifier: Apache-2.0
# What is left of the x_cluster query door after its split (UNPACK
# bindings/cluster_public_native_unpack.mojo, RUN cluster/cluster_door_run.mojo,
# PACK and the operator bindings/cluster_public_native_pack.mojo): the two
# reference-row helpers other doors import. `_reference_queries` is the
# query admission of the stored-rows predicts (DBSCAN, SpectralClustering,
# AgglomerativeClustering); `_agglo_predict` is AgglomerativeClustering's
# predict, which the x_cluster operator dispatches by its leaf code.
from std.python import PythonObject
from bindings.admitted_family_door import admit_family
from bindings.py_balanced import py_call, py_type, py_fields
from bindings.classical_arrays_door import classical_address
from core.host_finite_scan import host_all_finite_f32
from bindings.classical_pandas_door import classical_admit_value
from bindings.host_inference_scalars_door import host_inference_scalar
from bindings.agglo_native_boundary_door import agglo_reference_labels, agglo_scalar_repr, agglo_hash_admitted
from core.agglo_native_state import agglo_metric_code
from bindings.native_loaded_module_door import native_loaded_module, NativeModules
from bindings.perf2_shared_fitted_field_pack import native_fitted_get
from bindings.family_common_unpack import FAMILY_F32P


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
