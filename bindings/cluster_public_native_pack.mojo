# SPDX-License-Identifier: Apache-2.0
# PACK part and operator of the x_cluster query door: the x_cluster
# estimators' inference methods, one native call each
# (python/mojolearn/_expansion_cluster.py): `_CentersMixin.predict`,
# `transform` and `score` (MiniBatchKMeans, BisectingKMeans), MeanShift
# `predict`, BisectingKMeans `predict` (the bisection tree descent) and
# AffinityPropagation `predict`. The shell hands over the binding `_bind`
# resolved. UNPACK (bindings/cluster_public_native_unpack.mojo) reads the
# model and X into a typed stage and makes the typed provider call; RUN
# (cluster/cluster_door_run.mojo) classifies the operation and builds the
# request words; this file allocates the output arrays, makes the
# `array.array` results the Arrays' stores (no second copy) and dispatches.
# Same entry, same inputs: the bits do not move. The fits of these classes
# are bindings/cluster_fit_public_native_pack.mojo's.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_item
from bindings.classical_arrays_door import classical_address, classical_wrap_array
from core.admitted_weights import admitted_fill
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.native_loaded_module_door import NativeModules
from bindings.kmeans_variants_boundary_door import variants_pack_f32, variants_pack_i32
from bindings.py_optional import py_or_none
from bindings.operation_names_xc import (
    XC_OPN_FAMILY_XC_PREDICT, XC_OPN_FAMILY_XC_BISECT_PREDICT, XC_OPN_FAMILY_XC_AP_PREDICT,
    XC_OPN_FAMILY_XC_AGGLO_PREDICT)
from core.kmeans_variants_api import VariantsResultSlot
from bindings.family_common_unpack import FAMILY_I32P
from bindings.family_common_pack import family_array
from bindings.cluster_keys_unpack import cluster_keys
from bindings.cluster_native_door import _agglo_predict
from bindings.cluster_public_native_unpack import (
    CentersQuery, centers_query_stage, affinity_query_stage, centers_resident, centers_resident_call,
    centers_query_call, bisect_query_stage, bisect_query_call,
)
from cluster.cluster_door_run import (
    XC_RES_PREDICT, XC_RES_SCORE, XC_RES_TRANSFORM,
    cluster_query_variants, cluster_query_kind, cluster_unknown_code,
)


def _xc_array(flat: PythonObject, shape: PythonObject, typestr: String, code: String,
              context: PythonObject) raises -> PythonObject:
    """`Array._from_flat(flat, shape, typestr)` without its copy: the
    entry returns each result as one `array.array` of the dtype's store code
    (providers/python/x_cluster_out_pyapi.mojo, one memcpy from the result), which becomes the
    Array's store as it is. Anything else is copied once, as `_from_flat` did.
    Same values either way."""
    var store = flat
    var nk = cluster_keys()
    if py_type(flat) is not py_item(context, nk[].k_array_type) or String(py=flat.typecode) != code:
        store = py_call(py_item(context, nk[].k_array_type), PythonObject(code), flat)
    return classical_wrap_array(store, shape, typestr, "C", context)


def _centers_query(model: PythonObject, binding: PythonObject, stage: CentersQuery,
                   context: PythonObject) raises -> PythonObject:
    """One centers query: against the model's resident device copy of the
    centers when the provider keeps one, else with the centers handed over."""
    var n = stage.n
    var kind = stage.kind
    var held = centers_resident(model, binding, stage.centers, stage.d, context)
    var result = VariantsResultSlot()
    if held:
        var k = Int(py=stage.centers.shape[0])
        # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
        var output = Optional[PythonObject]()
        if kind == XC_RES_PREDICT:
            output = family_array(Python.tuple(n), "<i4", context)
        elif kind == XC_RES_TRANSFORM:
            output = family_array(Python.tuple(n, k), "<f4", context)
        var address = 0 if not output else Int(py=classical_address(output.value()))
        var total = centers_resident_call(held, stage, k, address, result)
        _ = output  # the output array stays alive through the native write above
        return PythonObject(-total) if kind == XC_RES_SCORE else py_or_none(output)
    var k = centers_query_call(binding, stage, context, result)
    # Pack only the public result. The old nested result envelope also copied
    # discarded distances/labels to Python for predict/score; those copies
    # have no observable owner and are unnecessary with typed result fields.
    if kind == XC_RES_SCORE:
        return PythonObject(-result.storage[0].inertia)
    if kind == XC_RES_TRANSFORM:
        return _xc_array(variants_pack_f32(result.storage[0].distances), Python.tuple(n, k), "<f4", "f", context)
    return _xc_array(variants_pack_i32(result.storage[0].labels), Python.tuple(n), "<i4", "i", context)


def cluster_operation(model: PythonObject, operation: Int, args: PythonObject,
                      context: PythonObject, x_admitted: Bool = False) raises -> PythonObject:
    """args[0] is the resolved binding, args[1] X. `x_admitted`: args[1] is
    the X the public door already admitted (classical_check_x: 2-D,
    non-empty, finite, C-ordered float32), so the finite scan is not run a
    second time over the same words. `operation` is the leaf code of
    bindings/operation_names_xc.mojo (lane bw2-cluster; it was the name,
    matched by up to eight String compares per call)."""
    classical_connect_numpy(context)
    # One module handle for every fitted-field read of this call.
    var mods = NativeModules()
    var binding = args[0]
    if operation == XC_OPN_FAMILY_XC_AGGLO_PREDICT:
        return _agglo_predict(model, binding, args[1], context)
    if cluster_query_variants(operation):
        var stage = centers_query_stage(mods, model, args[1], cluster_query_kind(operation), context, x_admitted)
        return _centers_query(model, binding, stage, context)
    if operation == XC_OPN_FAMILY_XC_PREDICT:
        var stage = centers_query_stage(mods, model, args[1], XC_RES_PREDICT, context, x_admitted)
        return _centers_query(model, binding, stage, context)
    if operation == XC_OPN_FAMILY_XC_BISECT_PREDICT:
        var stage = bisect_query_stage(mods, model, args[1], context, x_admitted)
        var result = VariantsResultSlot()
        bisect_query_call(binding, stage, result)
        return _xc_array(variants_pack_i32(result.storage[0].labels), Python.tuple(stage.n), "<i4", "i", context)
    if operation == XC_OPN_FAMILY_XC_AP_PREDICT:
        var stage = affinity_query_stage(mods, model, args[1], XC_RES_PREDICT, context, x_admitted)
        if stage.empty:
            var labels = family_array(Python.tuple(stage.n), "<i4", context)
            var words = FAMILY_I32P(unsafe_from_address=Int(py=classical_address(labels)))
            admitted_fill[DType.int32](words, stage.n, Int32(-1))
            _ = labels  # keep the Python object alive through the native call above
            return labels
        return _centers_query(model, binding, stage, context)
    cluster_unknown_code(operation)
    return PythonObject(None)
