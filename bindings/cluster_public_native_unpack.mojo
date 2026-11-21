# SPDX-License-Identifier: Apache-2.0
# UNPACK part of the x_cluster query door (python/mojolearn/_expansion_cluster.py:
# `_CentersMixin.predict`, `transform` and `score` of MiniBatchKMeans and
# BisectingKMeans, MeanShift `predict`, BisectingKMeans `predict` (the
# bisection tree descent) and AffinityPropagation `predict`): the Python
# reads. The fitted fields and X are read here, once per operation and in the
# order the Python methods refused them, into the typed stages `CentersQuery`
# and `BisectQuery`; the resident centers lease and the typed provider calls
# under GIL release are here too. The number refusals and the request words
# are RUN's (cluster/cluster_door_run.mojo). PACK and the operator are
# bindings/cluster_public_native_pack.mojo. Same entry, same inputs: the bits
# do not move.
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.admitted_family_door import admit_family
from bindings.py_balanced import py_type, py_get, py_fields, py_item
from bindings.classical_arrays_door import (
    classical_address, classical_c_bytes, classical_is_f32, classical_c_order,
)
from core.host_finite_scan import host_all_finite_f32
from bindings.native_loaded_module_door import native_loaded_module, NativeModules
from bindings.perf2_shared_fitted_field_pack import native_fitted_get
from bindings.p3nc_fitted_freeze_unpack import fitted_is_freezable, fitted_freeze
from bindings.py_cluster_resident import (
    cres_variant_lookup, cres_variant_store,
)
from bindings.kmeans_variants_boundary_door import variants_connection, variants_span
from bindings.kmeans_variants_resident_boundary_unpack import (
    VariantsResidentLease, variants_resident_prepare, variants_resident_retire,
)
from bindings.native_buffer_address_unpack import native_buffer_address
from core.kmeans_variants_api import VariantsResultSlot, VARIANTS_CAP_RESIDENT
from bindings.family_common_unpack import FAMILY_F32P, family_nonfinite
from bindings.cluster_keys_unpack import cluster_keys
from cluster.cluster_door_run import (
    XC_RES_TRANSFORM,
    cluster_features_text, cluster_node_words, cluster_query_request, cluster_bisect_request,
    cluster_resident_bytes,
)


# ------------------------------------------------------------------ helpers
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
    storage (the test of bindings/mixture_public_native_unpack.mojo `_f32_ready`)."""
    var cls = py_type(source)
    var nk = cluster_keys()
    if cls is not py_item(context, nk[].k_array_class) and cls is not py_get(context, nk[].k_ndarray):
        return False
    if Int(py=source.ndim) != 2 or Int(py=source.size) <= 0:
        return False
    return classical_is_f32(source, context) and classical_c_order(source, context)


def _xc_input(mut mods: NativeModules, model: PythonObject, source: PythonObject, context: PythonObject,
              admitted: Bool = False) raises -> PythonObject:
    """`_XCluster._input_like_fit`: `_f32(X)` (as_f32_c, nonempty, finite)
    and the fitted feature count. `admitted`: the public door's
    classical_check_x already refused a non-finite X with scikit-learn's
    wording and returned this float32 C-order block (lane fb2-neighbors:
    the second full scan was HEAD-only work, one read of all n x d words
    per predict/transform/score)."""
    var x = PythonObject(None)
    if admitted and _xc_f32_ready(source, context):
        x = source
    elif _xc_f32_ready(source, context):
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
        raise Error(cluster_features_text(d, String(py_type(model).__name__), Int(py=fitted)))
    return x


# ------------------------------------------------------------ centers query
@fieldwise_init
struct CentersQuery(Movable):
    """One centers query as typed values. `x` is the admitted float32 X and
    `centers` the fitted `cluster_centers_` field, both pinned through the
    call; `kind` is the XC_RES_* query kind. `empty`: AffinityPropagation
    fitted no exemplar, so every label is -1 and no entry runs."""
    var x: PythonObject
    var centers: PythonObject
    var n: Int
    var d: Int
    var kind: Int
    var empty: Bool


def centers_query_stage(mut mods: NativeModules, model: PythonObject, source: PythonObject, kind: Int,
                        context: PythonObject, admitted: Bool) raises -> CentersQuery:
    """`_CentersMixin.predict` / `transform` / `score` and MeanShift
    `predict`: the fitted check, then X."""
    var centers = _xc_fitted(mods, model, "cluster_centers_")
    var x = _xc_input(mods, model, source, context, admitted)
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    return CentersQuery(x=x, centers=centers, n=n, d=d, kind=kind, empty=False)


def affinity_query_stage(mut mods: NativeModules, model: PythonObject, source: PythonObject, kind: Int,
                         context: PythonObject, admitted: Bool) raises -> CentersQuery:
    """`AffinityPropagation.predict`: the fitted check, the precomputed
    refusal, X, then the exemplar rows."""
    var b = native_loaded_module("builtins")
    _ = _xc_fitted(mods, model, "labels_")
    var affinity = native_fitted_get(mods, model, "affinity")
    if py_type(affinity) is b.str and String(affinity) == "precomputed":
        raise Error("Predict method is not supported when affinity='precomputed'.")
    var x = _xc_input(mods, model, source, context, admitted)
    var centers = native_fitted_get(mods, model, "cluster_centers_")
    var empty = Int(py=centers.shape[0]) == 0
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    return CentersQuery(x=x, centers=centers, n=n, d=d, kind=kind, empty=empty)


def centers_resident(model: PythonObject, binding: PythonObject, centers: PythonObject,
                     features: Int, context: PythonObject) raises -> Optional[VariantsResidentLease]:
    """The model's resident device copy of `centers` for this binding, minted
    on the first query when the provider keeps one; None otherwise."""
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
    cres_variant_store(model, binding, dims^, lease.copy(), cluster_resident_bytes(clusters, features))
    return lease^


def centers_resident_call(mut held: Optional[VariantsResidentLease], stage: CentersQuery, k: Int,
                          address: Int, mut result: VariantsResultSlot) raises -> Float64:
    """The resident query on the typed transport: X against the device copy
    of the centers; labels or distances are written at `address` (0 for
    score). Returns the inertia sum. The lease is retired here."""
    var source = variants_span(stage.x)
    var total = Float64(0)
    try:
        with GILReleased(Python()):
            total = held.value().native[].driver.resident_query(held.value().native[].handle,
                stage.kind, source.address, stage.n, k, stage.d, address, result)
    finally:
        variants_resident_retire(held.take())
        _ = stage.x
    return total


def centers_query_call(binding: PythonObject, stage: CentersQuery, context: PythonObject,
                       mut result: VariantsResultSlot) raises -> Int:
    """The centers query on the typed transport with the centers handed over
    per call. Returns the cluster count."""
    var centers = admit_family(stage.centers, 2, "cluster_centers_", context).array
    var k = Int(py=centers.shape[0])
    var connection = variants_connection(binding)
    var center_span = variants_span(centers)
    var source_span = variants_span(stage.x)
    var request = cluster_query_request(source_span, center_span, stage.n, k, stage.d, stage.kind)
    try:
        with GILReleased(Python()):
            connection.driver.query(request, stage.kind == XC_RES_TRANSFORM, result)
    finally:
        _ = connection.provider
        _ = centers
        _ = stage.x
    return k


# ------------------------------------------------------- bisection descent
@fieldwise_init
struct BisectQuery(Movable):
    """BisectingKMeans `predict` as typed values. `x`, the tree's node field,
    its C-order bytes and the tree centers are pinned through the call;
    `words` is the node table's int32 count."""
    var x: PythonObject
    var node_source: PythonObject
    var nodes: PythonObject
    var centers: PythonObject
    var n: Int
    var d: Int
    var words: Int


def bisect_query_stage(mut mods: NativeModules, model: PythonObject, source: PythonObject,
                       context: PythonObject, admitted: Bool) raises -> BisectQuery:
    """The fitted check, X, then the tree."""
    var nk = cluster_keys()
    _ = _xc_fitted(mods, model, "cluster_centers_")
    var x = _xc_input(mods, model, source, context, admitted)
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var fields = py_fields(model)
    var node_source = py_item(fields, nk[].k_tree_nodes_)
    # Preserve the existing raw C-order byte interpretation, including
    # rebound storage. The extra array('i') and boxed Int list disappear.
    var nodes = classical_c_bytes(node_source, "tree_nodes_", context)
    var words = cluster_node_words(len(nodes))
    var centers = py_item(fields, nk[].k__tree_centers)
    return BisectQuery(x=x, node_source=node_source, nodes=nodes, centers=centers, n=n, d=d, words=words)


def bisect_query_call(binding: PythonObject, stage: BisectQuery, mut result: VariantsResultSlot) raises:
    """The tree descent on the typed transport."""
    var connection = variants_connection(binding)
    var center_span = variants_span(stage.centers)
    var source_span = variants_span(stage.x)
    var address = native_buffer_address(stage.nodes) if stage.words else 0
    var request = cluster_bisect_request(source_span, center_span, address, stage.words, stage.n, stage.d)
    try:
        with GILReleased(Python()):
            connection.driver.bisect_predict(request, result)
    finally:
        _ = connection.provider
        _ = stage.node_source
        _ = stage.nodes
        _ = stage.centers
        _ = stage.x
