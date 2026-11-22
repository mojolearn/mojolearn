# SPDX-License-Identifier: Apache-2.0
# Public cluster lifecycle formerly implemented by _mode's Python decorators:
# the operator and PACK. Validation and admission are UNPACK
# (providers/python/cluster_public_lifecycle_pyapi_unpack.mojo, one typed
# `ClusterLifecycleCall` per call); fitted tracking, the owner dispatch, the
# refusal rewriting, the admission record of a passed query and the final
# output wrapping are here. The algorithm owners are
# bindings/cluster_fit_public_native_pack.mojo, bindings/cluster_public_native_pack.mojo
# and bindings/mixture_public_native_pack.mojo.
from std.python import Python, PythonObject
from bindings.py_balanced import py_type, py_get, py_item
from bindings.cluster_side_records_unpack import cluster_record_put
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.native_array_view_pack import native_array_ndarray_view
from bindings.cluster_fit_public_native_pack import cluster_fit_a_operation
from cluster.cluster_fit_door_run import cluster_fit_a_unknown_operation
from bindings.cluster_public_native_pack import cluster_operation
from cluster.cluster_door_run import cluster_unknown_operation
from bindings.mixture_public_native_pack import mixture_operation
from mixture.mixture_door_run import mixture_unknown_operation
from providers.python.estimator_lifecycle_pyapi_unpack import estimator_outside, estimator_fit_begin, estimator_fit_end, estimator_fit_complete
from providers.python.cluster_constructor_pyapi_pack import cluster_initialize_py
from providers.python.estimator_tabular_boundary_pyapi_unpack import _refusal
from providers.python.cluster_public_lifecycle_pyapi_unpack import _XC_ADMIT_SLOT, cluster_lifecycle_unpack
from bindings.native_operation_codes import FR_CFA, FR_XC, OP_LEAF_NONE
from bindings.operation_names_xc import (
    XC_OPN_FAMILY_XC_INITIALIZE, XC_OPN_FAMILY_XC_PREDICT, XC_OPN_FAMILY_XC_VARIANTS_PREDICT, XC_OPN_FAMILY_XC_BISECT_PREDICT,
    XC_OPN_FAMILY_XC_AP_PREDICT, XC_OPN_FAMILY_XC_AGGLO_PREDICT, XC_OPN_FAMILY_XC_DBSCAN_PREDICT,
)
from bindings.operation_names_mix import MIX_OPN_FAMILY_MIX_GMM_PREDICT, MIX_OPN_FAMILY_MIX_BGMM_PREDICT
from bindings.cluster_keys_unpack import cluster_keys


def cluster_public_operation(model: PythonObject, owner: Int, operation: Int, operation_name: String, args: PythonObject,
                             context: PythonObject) raises -> PythonObject:
    """The public lifecycle of the flagged cluster operations (the family
    door's FR_CFA, FR_XC and `family_mix_bgmm_*` names called with no
    binding). `owner` is the route owner and `operation` the name's leaf
    code in that owner's numbering (lane bw2-cluster: it was the name, with a
    method String built per call from suffix tests and two prefix tests to
    pick the owner); `operation_name` is read only for a name the table does not hold
    (OP_LEAF_NONE), whose owner's unknown-operation text is raised."""
    if operation == OP_LEAF_NONE:
        if owner == FR_CFA:
            cluster_fit_a_unknown_operation(operation_name)
        elif owner == FR_XC:
            cluster_unknown_operation(operation_name)
        else:
            classical_connect_numpy(context)
            mixture_unknown_operation(operation_name)
    if owner == FR_XC and operation == XC_OPN_FAMILY_XC_INITIALIZE:
        return cluster_initialize_py(model, args[1], context)
    classical_connect_numpy(context)
    var call = cluster_lifecycle_unpack(model, owner, operation, args, context)
    var fitting = call.fitting
    var prepared = call.prepared
    # predict has one result for every caller (Andrew, 2026-11-22, lane
    # predict-dtype): the labels as a NumPy view, with no caller test. The
    # other public results keep the caller test.
    var predicting = (owner == FR_XC and (operation == XC_OPN_FAMILY_XC_PREDICT or operation == XC_OPN_FAMILY_XC_VARIANTS_PREDICT
                                          or operation == XC_OPN_FAMILY_XC_BISECT_PREDICT or operation == XC_OPN_FAMILY_XC_AP_PREDICT
                                          or operation == XC_OPN_FAMILY_XC_AGGLO_PREDICT or operation == XC_OPN_FAMILY_XC_DBSCAN_PREDICT)) or (
        owner != FR_CFA and owner != FR_XC and (operation == MIX_OPN_FAMILY_MIX_GMM_PREDICT or operation == MIX_OPN_FAMILY_MIX_BGMM_PREDICT))
    var outside = predicting or estimator_outside()
    var depth = estimator_fit_begin(call.fields) if fitting else 0
    var result = PythonObject(None)
    try:
        if owner == FR_CFA:
            result = cluster_fit_a_operation(model, operation, prepared, context)
        elif owner == FR_XC:
            # prepared[1] is the X classical_check_x admitted above (finite,
            # float32 C order): the cluster queries skip a second scan.
            result = cluster_operation(model, operation, prepared, context, True)
        else:
            # X is the one classical_check_x admitted above (2-D, non-empty,
            # finite, float32 C order): the BayesianGaussianMixture queries
            # skip a second finite scan of it.
            result = mixture_operation(model, operation, prepared, context, True)
    except error:
        var refused = _refusal(error, fitting, call.nested, call.rows, call.columns, call.name)
        raise refused^
    finally:
        if fitting:
            estimator_fit_end(model, depth)
    if fitting:
        estimator_fit_complete(model, call.source, False, context)
    elif call.direct and not call.recorded:
        # Written only after the full admission and the query succeeded.
        cluster_record_put(_XC_ADMIT_SLOT, model, call.key, PythonObject(True))
    var nk = cluster_keys()
    if outside and py_type(result) is py_item(context, nk[].k_array_class) and py_get(context, nk[].k_ndarray) is not None:
        return native_array_ndarray_view(result, context)
    return result
