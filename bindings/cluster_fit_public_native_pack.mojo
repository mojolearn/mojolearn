# SPDX-License-Identifier: Apache-2.0
# PACK part and operator of the x_cluster fit door: the compiled owner of the
# fits of MiniBatchKMeans (fit, partial_fit), BisectingKMeans, MeanShift,
# OPTICS and AffinityPropagation (python/mojolearn/_expansion_cluster.py, with
# the centers' fit_predict and fit_transform), one native call per public
# method, reached from bindings/family_native_door.mojo for every
# `family_cfa_*` operation. Each fit is UNPACK
# (bindings/cluster_fit_public_native_unpack.mojo: the Python method's reads
# and refusals in its order, into one typed stage, and the typed provider
# call), RUN (cluster/cluster_fit_door_run.mojo: the number refusals and the
# request) and the publication here: the attributes the Python set, with the
# same names, dtypes and shapes, each result `array.array` becoming its
# Array's store. Same entries, same operands, same order: the bits do not
# move.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_item
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.family_common_unpack import family_isinstance
from bindings.family_common_pack import family_call
from bindings.cluster_keys_unpack import cluster_keys
from bindings.hdbscan_fit_boundary_pack import hdbscan_fit
from bindings.operation_names_xc import XC_OPN_FAMILY_XC_VARIANTS_TRANSFORM
from bindings.operation_names_cfa import (
    CFA_OPN_FAMILY_CFA_MBK_FIT, CFA_OPN_FAMILY_CFA_MBK_PARTIAL_FIT, CFA_OPN_FAMILY_CFA_BISECT_FIT,
    CFA_OPN_FAMILY_CFA_FIT_PREDICT, CFA_OPN_FAMILY_CFA_FIT_TRANSFORM, CFA_OPN_FAMILY_CFA_MEANSHIFT_FIT,
    CFA_OPN_FAMILY_CFA_MEANSHIFT_FIT_PREDICT, CFA_OPN_FAMILY_CFA_OPTICS_FIT, CFA_OPN_FAMILY_CFA_OPTICS_FIT_PREDICT,
    CFA_OPN_FAMILY_CFA_AP_FIT, CFA_OPN_FAMILY_CFA_AP_FIT_PREDICT, CFA_OPN_FAMILY_CFA_HDBSCAN_FIT)
from core.kmeans_variants_api import VariantsResultSlot
from bindings.kmeans_variants_boundary_door import VariantsPackedResult, variants_connection
from core.cluster_centroid_fit_api import CentroidFitResultSlot
from bindings.cluster_centroid_fit_boundary_door import CentroidPackedResult
from core.optics_fit_api import OpticsFitResultSlot
from bindings.optics_fit_boundary_door import OpticsPackedResult
from cluster.cluster_fit_door_run import optics_hierarchy_rows, bisect_tree_rows, cluster_fit_unknown_code
from bindings.cluster_public_native_pack import cluster_operation, _xc_array
from bindings.cluster_fit_public_native_unpack import (
    _set, _attr,
    meanshift_fit_stage, meanshift_fit_call,
    optics_fit_stage, optics_fit_call,
    bisect_fit_stage, bisect_fit_call,
    affinity_fit_stage, affinity_fit_call,
    minibatch_fit_stage, minibatch_fit_call, minibatch_partial_stage, minibatch_partial_call,
)


# ---------------------------------------------------------------- MeanShift
def meanshift_fit(model: PythonObject, binding: PythonObject, source: PythonObject,
                  context: PythonObject) raises:
    """`MeanShift.fit(X)`."""
    var stage = meanshift_fit_stage(model, source, context)
    var result = CentroidFitResultSlot()
    meanshift_fit_call(binding, stage, result)
    var n = stage.n
    var d = stage.d
    var packed = CentroidPackedResult(result, False)
    var kc = result.storage[0].clusters
    _set(model, "cluster_centers_", _xc_array(packed.centers, Python.tuple(kc, d), "<f4", "f", context))
    _set(model, "labels_", _xc_array(packed.labels, Python.tuple(n), "<i4", "i", context))
    _set(model, "bandwidth_", PythonObject(result.storage[0].bandwidth))
    _set(model, "n_iter_", PythonObject(result.storage[0].iterations))
    _set(model, "n_features_in_", PythonObject(d))


# ---------------------------------------------------------------- OPTICS
def optics_fit(model: PythonObject, binding: PythonObject, source: PythonObject,
               context: PythonObject) raises:
    """`OPTICS.fit(X)`."""
    var stage = optics_fit_stage(model, source, context)
    var result = OpticsFitResultSlot()
    optics_fit_call(binding, stage, result)
    var n = stage.n
    var packed = OpticsPackedResult(result)
    _set(model, "core_distances_", _xc_array(packed.core_distances, Python.tuple(n), "<f4", "f", context))
    _set(model, "reachability_", _xc_array(packed.reachability, Python.tuple(n), "<f4", "f", context))
    _set(model, "ordering_", _xc_array(packed.ordering, Python.tuple(n), "<i4", "i", context))
    _set(model, "predecessor_", _xc_array(packed.predecessor, Python.tuple(n), "<i4", "i", context))
    _set(model, "labels_", _xc_array(packed.labels, Python.tuple(n), "<i4", "i", context))
    if stage.is_xi:
        var rows = optics_hierarchy_rows(len(result.storage[0].hierarchy))
        _set(model, "cluster_hierarchy_", _xc_array(packed.hierarchy, Python.tuple(rows, 2), "<i4", "i", context))
    _set(model, "n_features_in_", PythonObject(stage.d))


# ---------------------------------------------------------------- BisectingKMeans
def bisect_fit(model: PythonObject, binding: PythonObject, source: PythonObject, sample_weight: PythonObject,
               context: PythonObject) raises:
    """`BisectingKMeans.fit(X, sample_weight)`."""
    var stage = bisect_fit_stage(model, source, sample_weight, context)
    var result = VariantsResultSlot()
    bisect_fit_call(binding, stage, result)
    var n = stage.n
    var d = stage.d
    var k = stage.k
    var packed = VariantsPackedResult(result, True)
    _set(model, "cluster_centers_", _xc_array(packed.centers, Python.tuple(k, d), "<f4", "f", context))
    var m = bisect_tree_rows(len(result.storage[0].tree_nodes))
    _set(model, "_tree_centers", _xc_array(packed.tree_centers, Python.tuple(m, d), "<f4", "f", context))
    _set(model, "tree_nodes_", _xc_array(packed.tree_nodes, Python.tuple(m, 3), "<i4", "i", context))
    _set(model, "labels_", _xc_array(packed.labels, Python.tuple(n), "<i4", "i", context))
    _set(model, "inertia_", PythonObject(result.storage[0].inertia))
    _set(model, "n_features_in_", PythonObject(d))


# ---------------------------------------------------------------- AffinityPropagation
def affinity_fit(model: PythonObject, binding: PythonObject, source: PythonObject,
                 context: PythonObject) raises:
    """`AffinityPropagation.fit(X)`."""
    var stage = affinity_fit_stage(model, source, context)
    var result = CentroidFitResultSlot()
    affinity_fit_call(binding, stage, result)
    var n = stage.n
    var d = stage.d
    var packed = CentroidPackedResult(result, True)
    var k = len(result.storage[0].center_indices)
    _set(model, "cluster_centers_indices_", _xc_array(packed.center_indices, Python.tuple(k), "<i4", "i", context))
    _set(model, "labels_", _xc_array(packed.labels, Python.tuple(n), "<i4", "i", context))
    _set(model, "affinity_matrix_", _xc_array(packed.affinity, Python.tuple(n, n), "<f4", "f", context))
    # Observable final availability/responsibility diagonals stay public.
    _set(model, "message_diag_", _xc_array(packed.message_diagonal, Python.tuple(len(result.storage[0].message_diagonal)), "<f4", "f", context))
    _set(model, "n_iter_", PythonObject(result.storage[0].iterations))
    _set(model, "n_features_in_", PythonObject(d))
    if not stage.precomputed:
        _set(model, "cluster_centers_", _xc_array(packed.centers, Python.tuple(k, d), "<f4", "f", context))


# ---------------------------------------------------------------- MiniBatchKMeans
def minibatch_fit(model: PythonObject, binding: PythonObject, source: PythonObject, sample_weight: PythonObject,
                  context: PythonObject) raises:
    """`MiniBatchKMeans.fit(X, sample_weight)`. The provider is connected
    first, once: its capabilities decide who refuses a non-finite X."""
    var connection = variants_connection(binding)
    var stage = minibatch_fit_stage(model, source, sample_weight, connection.capabilities, context)
    var result = VariantsResultSlot()
    minibatch_fit_call(connection, stage, result)
    var n = stage.n
    var d = stage.d
    var k = stage.k
    var packed = VariantsPackedResult(result, False)
    _set(model, "cluster_centers_", _xc_array(packed.centers, Python.tuple(k, d), "<f4", "f", context))
    _set(model, "counts_", _xc_array(packed.counts, Python.tuple(k), "<f4", "f", context))
    _set(model, "labels_", _xc_array(packed.labels, Python.tuple(n), "<i4", "i", context))
    _set(model, "inertia_", PythonObject(result.storage[0].inertia))
    _set(model, "n_steps_", PythonObject(result.storage[0].steps))
    _set(model, "n_iter_", PythonObject(result.storage[0].iterations))
    # `fit_loop_` (lane ab-loop-mode): (mini-batch steps, step loop
    # milliseconds) from the fit's FitLoop; not saved in the archive.
    _set(model, "fit_loop_", Python.tuple(result.storage[0].loop.iterations,
        result.storage[0].loop.milliseconds()))
    _set(model, "n_features_in_", PythonObject(d))


def minibatch_partial_fit(model: PythonObject, binding: PythonObject, source: PythonObject,
                          sample_weight: PythonObject, context: PythonObject) raises:
    """`MiniBatchKMeans.partial_fit(X, sample_weight)`: one step on all of X;
    the random stream and the reassignment counter carry over."""
    var stage = minibatch_partial_stage(model, source, sample_weight, context)
    var result = VariantsResultSlot()
    minibatch_partial_call(binding, stage, result)
    var n = stage.n
    var d = stage.d
    var k = stage.k
    var packed = VariantsPackedResult(result, False)
    _set(model, "cluster_centers_", _xc_array(packed.centers, Python.tuple(k, d), "<f4", "f", context))
    _set(model, "counts_", _xc_array(packed.counts, Python.tuple(k), "<f4", "f", context))
    _set(model, "labels_", _xc_array(packed.labels, Python.tuple(n), "<i4", "i", context))
    _set(model, "inertia_", PythonObject(result.storage[0].inertia))
    _set(model, "_partial_state", Python.tuple(Python.tuple(PythonObject(result.storage[0].state_hi),
        PythonObject(result.storage[0].state_lo)), PythonObject(result.storage[0].since_reassignment)))
    var steps = family_call(native_loaded_module("builtins").getattr, model, PythonObject("n_steps_"),
                            PythonObject(0))
    _set(model, "n_steps_", family_call(native_loaded_module("operator").add, steps, PythonObject(1)))
    _set(model, "n_features_in_", PythonObject(d))


def _centers_fit(model: PythonObject, binding: PythonObject, source: PythonObject,
                 context: PythonObject) raises:
    """`self.fit(X)` of a centers estimator (fit_predict and fit_transform
    pass no sample_weight)."""
    var nk = cluster_keys()
    if family_isinstance(model, py_item(context, nk[].k_family_cfa_minibatch)):
        minibatch_fit(model, binding, source, PythonObject(None), context)
    else:
        bisect_fit(model, binding, source, PythonObject(None), context)


# ---------------------------------------------------------------- HDBSCAN
# HDBSCAN.fit is `hdbscan_fit` (bindings/hdbscan_fit_boundary_pack.mojo: UNPACK,
# typed RUN, PACK), called directly by HDBSCAN's lifecycle. The one code
# `family_cfa_hdbscan_fit` below stays for the parallel fit plan
# (bindings/parallel_plan_specs.mojo), which enters here.


# ---------------------------------------------------------------- door
def cluster_fit_a_operation(model: PythonObject, operation: Int, args: PythonObject,
         context: PythonObject) raises -> PythonObject:
    """args[0] is the resolved binding, args[1] X, args[2] sample_weight
    where the method takes one. `operation` is the leaf code of
    bindings/operation_names_cfa.mojo (lane bw2-cluster; it was the name,
    matched by up to thirteen String compares per call)."""
    classical_connect_numpy(context)
    var binding = args[0]
    if operation == CFA_OPN_FAMILY_CFA_MBK_FIT:
        minibatch_fit(model, binding, args[1], args[2], context)
        return model
    if operation == CFA_OPN_FAMILY_CFA_MBK_PARTIAL_FIT:
        minibatch_partial_fit(model, binding, args[1], args[2], context)
        return model
    if operation == CFA_OPN_FAMILY_CFA_BISECT_FIT:
        bisect_fit(model, binding, args[1], args[2], context)
        return model
    if operation == CFA_OPN_FAMILY_CFA_FIT_PREDICT:
        _centers_fit(model, binding, args[1], context)
        return _attr(model, "labels_")
    if operation == CFA_OPN_FAMILY_CFA_FIT_TRANSFORM:
        _centers_fit(model, binding, args[1], context)
        return cluster_operation(model, XC_OPN_FAMILY_XC_VARIANTS_TRANSFORM, Python.tuple(binding, args[1]), context)
    if operation == CFA_OPN_FAMILY_CFA_MEANSHIFT_FIT:
        meanshift_fit(model, binding, args[1], context)
        return model
    if operation == CFA_OPN_FAMILY_CFA_MEANSHIFT_FIT_PREDICT:
        meanshift_fit(model, binding, args[1], context)
        return _attr(model, "labels_")
    if operation == CFA_OPN_FAMILY_CFA_OPTICS_FIT:
        optics_fit(model, binding, args[1], context)
        return model
    if operation == CFA_OPN_FAMILY_CFA_OPTICS_FIT_PREDICT:
        optics_fit(model, binding, args[1], context)
        return _attr(model, "labels_")
    if operation == CFA_OPN_FAMILY_CFA_AP_FIT:
        affinity_fit(model, binding, args[1], context)
        return model
    if operation == CFA_OPN_FAMILY_CFA_AP_FIT_PREDICT:
        affinity_fit(model, binding, args[1], context)
        return _attr(model, "labels_")
    if operation == CFA_OPN_FAMILY_CFA_HDBSCAN_FIT:
        _ = hdbscan_fit(model, binding, args[1], context)
        return model
    cluster_fit_unknown_code(operation)
    return PythonObject(None)
