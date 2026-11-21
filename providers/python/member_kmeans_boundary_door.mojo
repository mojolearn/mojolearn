# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the KMeans family (KMeans, kind 0;
`_mojolearn_member = (MEMBER_FAMILY_KMEANS, 0)` on the class). Recipe:
MEMBER-ABI-V2.md; RUN: core/kmeans_member.mojo.

  * `kmeans_member_admit` (UNPACK, once per member): the estimator's own
    parameter admission (bindings/kmeans_native_unpack.mojo
    `kmeans_fit_init`, `kmeans_fit_scalars`: same refusals, same order),
    its binding through the estimator's own binding route (`device=` and
    the numeric mode, `estimator_binding`), and that binding's KMeans entry
    as the route's `entry` with its capability word in `route.flags`.
    Explicit starting centers (`init='array'` or an array `init`) are not
    parameter words: admit answers False and the caller keeps the class's
    own doors for that instance.
  * `kmeans_member_pack` (PACK): the fitted attributes the estimator's own
    fit writes (cluster_centers_ frozen, labels_, inertia_, n_iter_,
    sum_scale_, weight_scale_, n_features_in_) and the release of its
    resident holder, from the member state's blocks.
  * `kmeans_member_state` (UNPACK): a member state from a fitted KMeans.

SERIAL BY NATURE: one member's admission or publication (a few record reads
and two block copies); a caller runs its members in parallel.
"""
from std.memory import bitcast, unsafe_memcpy
from std.python import Python, PythonObject
from bindings.admitted_family_door import admit_family
from bindings.classical_arrays_door import classical_address
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.cluster_keys_unpack import cluster_keys
from bindings.family_common_pack import family_array
from bindings.kmeans_descriptor_boundary_door import kmeans_connection, kmeans_provider
from bindings.kmeans_native_unpack import kmeans_fit_init, kmeans_fit_scalars
from bindings.member_owner_unpack import MemberOwner
from bindings.p3nc_fitted_freeze_unpack import fitted_freeze
from bindings.py_balanced import py_call, py_get, py_setitem
from core.device_route import DeviceRoute, DEVICE_CPU, DEVICE_GPU, device_vendor_code
from core.entry_table import EntryRef, entry_ref_word
from core.hostptr import f32_ptr
from core.kmeans_native_api import KMEANS_MODE, KMEANS_CAP_HOST, KMEANS_NATIVE_COUNT
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free,
    MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_IMPORT_DONE, MEMBER_F64,
)
from providers.python.estimator_binding_pyapi_door import estimator_binding
from x_cluster.kmeans_descriptor_boundary_run import kmeans_capabilities, kmeans_driver

comptime _INIT_ARRAY = 2


def kmeans_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted estimator. False for explicit starting centers (the
    class's own doors run that instance)."""
    classical_connect_numpy(context)
    var settings = kmeans_fit_init(fields, b, context)
    if settings.init_code == _INIT_ARRAY:
        return False
    var binding = estimator_binding(model, "_mojolearn", context)
    var connection = kmeans_connection(binding)
    var capabilities = kmeans_capabilities(connection)
    var host = (capabilities & KMEANS_CAP_HOST) != 0
    var scalars = kmeans_fit_scalars(fields, settings.clusters, host, b, context)
    var driver = kmeans_driver(connection)
    var mode = driver.metadata(KMEANS_MODE)
    var vendor = 0
    if not host:
        vendor = device_vendor_code(String(py_call(binding.mojolearn_vendor)))
        if vendor <= 0:
            raise Error("ImportError: mojolearn: the KMeans binding names no GPU vendor")
    var route = DeviceRoute(DEVICE_CPU if host else DEVICE_GPU)
    route.mode = mode
    route.vendor = vendor
    route.entry = EntryRef(driver.entry, entry_ref_word(mode, vendor, KMEANS_NATIVE_COUNT))
    route.flags = capabilities
    owner.plan.route = route
    var seed_bits = bitcast[DType.float64](Int64(scalars.seed))
    var words: List[Float64] = [Float64(scalars.clusters), Float64(settings.init_code), Float64(scalars.restarts),
        Float64(scalars.iterations), scalars.tolerance, seed_bits, Float64(settings.metric_code),
        settings.oversampling]
    owner.plan.params = words^
    owner.pins = Python.tuple(binding, kmeans_provider(connection))
    return True


def kmeans_member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: write the fitted attributes of state `state` into `model`, as
    the estimator's own fit does (bindings/kmeans_native_pack.mojo
    `_kmeans_fit`: the same names, frozen centers, the resident holder
    released so the next query uploads the new centers once)."""
    classical_connect_numpy(context)
    var nk = cluster_keys()
    var call = MemberCall(plan.route, state)
    call.block = MEMBER_BLOCK_FAMILY
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var k = call.dst.rows
    var d = call.dst.cols
    var centers = family_array(Python.tuple(k, d), "<f4", context)
    unsafe_memcpy(dest=f32_ptr(Int(py=classical_address(centers))), src=f32_ptr(call.dst.data), count=k * d)
    call.block = MEMBER_BLOCK_FAMILY + 1
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var n = call.dst.cols
    var labels = family_array(Python.tuple(n), "<i4", context)
    if n > 0:
        unsafe_memcpy(dest=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(labels))),
            src=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=call.dst.data), count=n)
    call.block = MEMBER_BLOCK_SCALARS
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var scalars = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data)
    var inertia = scalars[0]
    var iterations = Int(scalars[1])
    var sum_scale = scalars[2]
    var weight_scale = scalars[3]
    _ = fitted_freeze(centers, context)
    py_setitem(fields, nk[].k_cluster_centers_, centers)
    py_setitem(fields, nk[].k_labels_, labels)
    py_setitem(fields, nk[].k_inertia_, PythonObject(inertia))
    py_setitem(fields, nk[].k_n_iter_, PythonObject(iterations))
    py_setitem(fields, nk[].k_sum_scale_, PythonObject(sum_scale))
    py_setitem(fields, nk[].k_weight_scale_, PythonObject(weight_scale))
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(d))
    # The estimator's own `_kmeans_release`: the last fit's resident centers go.
    py_setitem(fields, "_resident_holder", PythonObject(None))


def kmeans_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding the fitted centers and scalars of
    `model` (labels_ are not needed to predict or transform and are not
    imported). The caller frees it (`member_free`)."""
    classical_connect_numpy(context)
    var source = py_get(fields, PythonObject("cluster_centers_"), PythonObject(None))
    if source is None:
        raise Error("NotFittedError: this member is not fitted yet")
    var centers = admit_family(source, 2, "cluster_centers_", context, finite=True).array
    var k = Int(py=centers.shape[0])
    var d = Int(py=centers.shape[1])
    var scalars: List[Float64] = [0.0, 0.0, 0.0, 0.0]
    var names: List[String] = ["inertia_", "n_iter_", "sum_scale_", "weight_scale_"]
    for i in range(4):  # driver-loop: four fitted scalar attributes
        var value = py_get(fields, PythonObject(names[i]), PythonObject(None))
        if value is not None:
            scalars[i] = Float64(py=value)
    var call = member_new(plan)
    try:
        call.block = MEMBER_BLOCK_FAMILY
        call.x = MemberMatrix(Int(py=classical_address(centers)), k, d)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, 4, MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = centers
    _ = source
    _ = len(scalars)
    return call.state
