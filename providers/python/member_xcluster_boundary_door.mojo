# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the XCluster family (x_cluster estimators). Recipe:
MEMBER-ABI-V2.md; RUN: core/xcluster_member.mojo.

Published kinds: 1 BisectingKMeans (`_mojolearn_member = (25, 1)` on the
class). Every other kind answers False here (MiniBatchKMeans 0, MeanShift 2,
OPTICS 3, AffinityPropagation 4, BayesianGaussianMixture 5: not published
yet) and the caller keeps the class's own doors.

  * `xcluster_member_admit` (UNPACK, once per member): the estimator's own
    parameter reads and refusals in its fit's order
    (bindings/cluster_fit_public_native_unpack.mojo `bisect_fit_stage`:
    init, bisecting_strategy, algorithm, n_clusters, n_init, max_iter,
    random_state, tol; the X-dependent refusals run in RUN at fit), its
    binding through the estimator's own binding route (`device=` and the
    numeric mode, `estimator_binding(model, "_mojolearn_x_cluster")`), and
    that binding's KMeans-variants entry as the route's `entry` with its
    capability word in `route.flags`.
  * `xcluster_member_pack` (PACK): the attributes `bisect_fit` writes
    (bindings/cluster_fit_public_native_pack.mojo: cluster_centers_,
    _tree_centers, tree_nodes_, labels_, inertia_, n_features_in_; the
    model's resident centers released by `_set`).
  * `xcluster_member_state` (UNPACK): a member state from a fitted model.

SERIAL BY NATURE: one member's admission or publication (a few attribute
reads and block copies); a caller runs its members in parallel.
"""
from std.memory import bitcast, unsafe_memcpy
from std.python import Python, PythonObject
from bindings.admitted_family_door import admit_family
from bindings.classical_arrays_door import classical_address, classical_c_bytes
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.cluster_fit_public_native_unpack import _attr, _int, _float, _eq, _one_of, _seed, _set
from bindings.family_common_pack import family_array
from bindings.family_common_unpack import family_repr
from bindings.kmeans_variants_boundary_door import variants_connection
from bindings.member_owner_unpack import MemberOwner
from bindings.native_buffer_address_unpack import native_buffer_address
from bindings.py_balanced import py_call, py_get
from cluster.cluster_fit_door_run import cluster_n_init_positive
from cluster.cluster_door_run import cluster_node_words
from core.device_route import DeviceRoute, DEVICE_CPU, DEVICE_GPU, device_vendor_code
from core.entry_table import EntryRef, entry_ref_word
from core.hostptr import f32_ptr
from core.kmeans_variants_api import VARIANTS_MODE, KMEANS_VARIANTS_COUNT
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free,
    MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_IMPORT_DONE,
    MEMBER_F64, MEMBER_I32,
)
from providers.python.estimator_binding_pyapi_door import estimator_binding

comptime _KIND_BISECT = 1
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]


def _variants_route(model: PythonObject, context: PythonObject, mut owner: MemberOwner) raises:
    """The estimator's x_cluster binding and its variants entry -> the route."""
    var binding = estimator_binding(model, "_mojolearn_x_cluster", context)
    var connection = variants_connection(binding)
    var mode = connection.driver.metadata(VARIANTS_MODE)
    var vendor = device_vendor_code(String(py_call(binding.x_cluster_vendor)))
    if vendor < 0:
        raise Error("ImportError: mojolearn: the x_cluster binding names no vendor")
    var route = DeviceRoute(DEVICE_CPU if vendor == 0 else DEVICE_GPU)
    route.mode = mode
    route.vendor = vendor
    route.entry = EntryRef(connection.driver.entry, entry_ref_word(mode, vendor, KMEANS_VARIANTS_COUNT))
    route.flags = connection.capabilities
    owner.plan.route = route
    owner.pins = Python.tuple(binding, connection.provider)


def _bisect_admit(model: PythonObject, context: PythonObject, mut owner: MemberOwner) raises:
    """`bisect_fit_stage`'s reads, in its order, without X."""
    var init = _attr(model, "init")
    if not _one_of(init, "random", "k-means++"):
        raise Error("NotImplementedError: mojolearn BisectingKMeans: init=" + family_repr(init)
                    + " is not implemented; 'random' or 'k-means++'")
    var strategy = _attr(model, "bisecting_strategy")
    if not _one_of(strategy, "biggest_inertia", "largest_cluster"):
        raise Error("bisecting_strategy must be 'biggest_inertia' or 'largest_cluster', got " + family_repr(strategy))
    var algorithm = _attr(model, "algorithm")
    if not _one_of(algorithm, "lloyd", "elkan"):
        raise Error("algorithm must be 'lloyd' or 'elkan', got " + family_repr(algorithm))
    var k = Int(py=_int(_attr(model, "n_clusters")))
    if k < 1:
        raise Error("ValueError: n_clusters must be at least 1, got " + String(k))
    var n_init = Int(py=_int(_attr(model, "n_init")))
    cluster_n_init_positive(n_init)
    var random_init = _eq(init, PythonObject("random"))
    var max_iter = Int(py=_int(_attr(model, "max_iter")))
    var seed = _seed(_attr(model, "random_state"))
    var largest_cluster = _eq(strategy, PythonObject("largest_cluster"))
    var tolerance = Float64(py=_float(_attr(model, "tol")))
    _variants_route(model, context, owner)
    var words: List[Float64] = [Float64(_KIND_BISECT), Float64(k), 0.0 if random_init else 1.0, Float64(n_init),
        Float64(max_iter), tolerance, bitcast[DType.float64](Int64(seed)), 1.0 if largest_cluster else 0.0]
    owner.plan.params = words^


def xcluster_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted estimator. False for a kind not published yet (the
    class's own doors run that instance)."""
    if kind != _KIND_BISECT:
        return False
    classical_connect_numpy(context)
    _bisect_admit(model, context, owner)
    return True


def _export(plan: MemberPlan, mut call: MemberCall, block: Int) raises:
    call.block = block
    call.flags = 0
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)


def _f32_array(view: MemberMatrix, context: PythonObject) raises -> PythonObject:
    var array = family_array(Python.tuple(view.rows, view.cols), "<f4", context)
    if view.rows * view.cols > 0:
        unsafe_memcpy(dest=f32_ptr(Int(py=classical_address(array))), src=f32_ptr(view.data),
            count=view.rows * view.cols)
    return array


def _i32_array(view: MemberMatrix, shape: PythonObject, context: PythonObject) raises -> PythonObject:
    var array = family_array(shape, "<i4", context)
    if view.rows * view.cols > 0:
        unsafe_memcpy(dest=_I32P(unsafe_from_address=Int(py=classical_address(array))),
            src=_I32P(unsafe_from_address=view.data), count=view.rows * view.cols)
    return array


def xcluster_member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: write the fitted attributes of state `state` into `model`, as
    the estimator's own fit does (the same names, dtypes and shapes)."""
    if Int(plan.params[0]) != _KIND_BISECT:
        raise Error("RuntimeError: mojolearn: this x_cluster member kind is not published")
    classical_connect_numpy(context)
    var call = MemberCall(plan.route, state)
    _export(plan, call, MEMBER_BLOCK_FAMILY)
    var d = call.dst.cols
    var centers = _f32_array(call.dst, context)
    _export(plan, call, MEMBER_BLOCK_FAMILY + 2)
    var tree_centers = _f32_array(call.dst, context)
    _export(plan, call, MEMBER_BLOCK_FAMILY + 3)
    var tree_nodes = _i32_array(call.dst, Python.tuple(call.dst.rows, 3), context)
    _export(plan, call, MEMBER_BLOCK_FAMILY + 1)
    var labels = _i32_array(call.dst, Python.tuple(call.dst.cols), context)
    _export(plan, call, MEMBER_BLOCK_SCALARS)
    var inertia = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data)[0]
    _set(model, "cluster_centers_", centers)
    _set(model, "_tree_centers", tree_centers)
    _set(model, "tree_nodes_", tree_nodes)
    _set(model, "labels_", labels)
    _set(model, "inertia_", PythonObject(inertia))
    _set(model, "n_features_in_", PythonObject(d))


def xcluster_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding the fitted leaf centers, tree and
    inertia of `model` (labels_ are not needed to predict and are not
    imported). The caller frees it (`member_free`)."""
    if Int(plan.params[0]) != _KIND_BISECT:
        raise Error("RuntimeError: mojolearn: this x_cluster member kind is not published")
    classical_connect_numpy(context)
    var source = py_get(fields, PythonObject("cluster_centers_"), PythonObject(None))
    if source is None:
        raise Error("NotFittedError: this member is not fitted yet")
    var centers = admit_family(source, 2, "cluster_centers_", context, finite=True).array
    var tree_source = py_get(fields, PythonObject("_tree_centers"), PythonObject(None))
    var node_source = py_get(fields, PythonObject("tree_nodes_"), PythonObject(None))
    if tree_source is None or node_source is None:
        raise Error("NotFittedError: this member is not fitted yet")
    var tree = admit_family(tree_source, 2, "_tree_centers", context, finite=True).array
    # The tree's raw C-order int32 words, read as the estimator's own predict reads them.
    var nodes = classical_c_bytes(node_source, "tree_nodes_", context)
    var words = cluster_node_words(len(nodes))
    var scalars: List[Float64] = [0.0]
    var value = py_get(fields, PythonObject("inertia_"), PythonObject(None))
    if value is not None:
        scalars[0] = Float64(py=value)
    var call = member_new(plan)
    try:
        call.block = MEMBER_BLOCK_FAMILY
        call.x = MemberMatrix(Int(py=classical_address(centers)), Int(py=centers.shape[0]), Int(py=centers.shape[1]))
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = MEMBER_BLOCK_FAMILY + 2
        call.x = MemberMatrix(Int(py=classical_address(tree)), Int(py=tree.shape[0]), Int(py=tree.shape[1]))
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = MEMBER_BLOCK_FAMILY + 3
        call.x = MemberMatrix(native_buffer_address(nodes) if words else 0, words // 3, 3, MEMBER_I32)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, 1, MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = centers
    _ = source
    _ = tree
    _ = tree_source
    _ = nodes
    _ = node_source
    _ = len(scalars)
    return call.state
