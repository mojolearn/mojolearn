# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the XCluster family (x_cluster estimators). Recipe:
MEMBER-ABI-V2.md; RUN: core/xcluster_member.mojo.

Published kinds: 0 MiniBatchKMeans, 1 BisectingKMeans
(`_mojolearn_member = (25, kind)` on each class). Every other kind answers
False here (MeanShift 2, OPTICS 3, AffinityPropagation 4,
BayesianGaussianMixture 5: not published yet), as does a MiniBatchKMeans
with explicit starting centers (an array `init`), and the caller keeps the
class's own doors.

  * `xcluster_member_admit` (UNPACK, once per member): the estimator's own
    parameter reads and refusals in its fit's order
    (bindings/cluster_fit_public_native_unpack.mojo `minibatch_fit_stage`:
    n_clusters, init, n_init ('auto' by `minibatch_auto_n_init`),
    max_no_improvement, max_iter, batch_size, init_size, random_state, tol,
    reassignment_ratio; `bisect_fit_stage`:
    init, bisecting_strategy, algorithm, n_clusters, n_init, max_iter,
    random_state, tol; the X-dependent refusals run in RUN at fit), its
    binding through the estimator's own binding route (`device=` and the
    numeric mode, `estimator_binding(model, "_mojolearn_x_cluster")`), and
    that binding's KMeans-variants entry as the route's `entry` with its
    capability word in `route.flags`.
  * `xcluster_member_pack` (PACK): the attributes the class's fit writes
    (bindings/cluster_fit_public_native_pack.mojo `minibatch_fit`:
    cluster_centers_, counts_, labels_, inertia_, n_steps_, n_iter_,
    n_features_in_; `bisect_fit`: cluster_centers_,
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
from bindings.family_common_unpack import family_repr, family_is_str
from bindings.kmeans_variants_boundary_door import variants_connection
from bindings.member_owner_unpack import MemberOwner
from bindings.native_buffer_address_unpack import native_buffer_address
from bindings.py_balanced import py_call, py_get
from cluster.cluster_fit_door_run import cluster_n_init_positive, minibatch_auto_n_init
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

comptime _KIND_MINIBATCH = 0
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


def _minibatch_admit(model: PythonObject, context: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """`minibatch_fit_stage`'s reads, in its order, without X and weights.
    False for explicit starting centers (the class's own doors)."""
    var k = Int(py=_int(_attr(model, "n_clusters")))
    if k < 1:
        raise Error("ValueError: n_clusters must be at least 1, got " + String(k))
    var init = _attr(model, "init")
    if not family_is_str(init):
        return False
    if not _one_of(init, "k-means++", "random"):
        raise Error("init must be 'k-means++', 'random' or an array, got " + family_repr(init))
    var init_random = _eq(init, PythonObject("random"))
    var n_init_field = _attr(model, "n_init")
    var n_init: Int
    if _eq(n_init_field, PythonObject("auto")):
        n_init = minibatch_auto_n_init(False, init_random)
    else:
        n_init = Int(py=_int(n_init_field))
    cluster_n_init_positive(n_init)
    var mni_field = _attr(model, "max_no_improvement")
    var mni = -1 if mni_field is None else Int(py=_int(mni_field))
    var max_iter = Int(py=_int(_attr(model, "max_iter")))
    var batch_size = Int(py=_int(_attr(model, "batch_size")))
    var init_size_field = _attr(model, "init_size")
    var init_size = 0 if init_size_field is None else Int(py=_int(init_size_field))
    var seed = _seed(_attr(model, "random_state"))
    var tolerance = Float64(py=_float(_attr(model, "tol")))
    var ratio = Float64(py=_float(_attr(model, "reassignment_ratio")))
    _variants_route(model, context, owner)
    var words: List[Float64] = [Float64(_KIND_MINIBATCH), Float64(k), 1.0 if init_random else 0.0,
        Float64(n_init), Float64(max_iter), tolerance, bitcast[DType.float64](Int64(seed)), Float64(batch_size),
        Float64(mni), Float64(init_size), ratio]
    owner.plan.params = words^
    return True


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
    if kind == _KIND_MINIBATCH:
        classical_connect_numpy(context)
        return _minibatch_admit(model, context, owner)
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
    return _f32_shaped(view, Python.tuple(view.rows, view.cols), context)


def _f32_shaped(view: MemberMatrix, shape: PythonObject, context: PythonObject) raises -> PythonObject:
    var array = family_array(shape, "<f4", context)
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
    var kind = Int(plan.params[0])
    if kind != _KIND_BISECT and kind != _KIND_MINIBATCH:
        raise Error("RuntimeError: mojolearn: this x_cluster member kind is not published")
    classical_connect_numpy(context)
    var call = MemberCall(plan.route, state)
    if kind == _KIND_MINIBATCH:
        _export(plan, call, MEMBER_BLOCK_FAMILY)
        var cols = call.dst.cols
        var mb_centers = _f32_array(call.dst, context)
        _export(plan, call, MEMBER_BLOCK_FAMILY + 4)
        var counts = _f32_shaped(call.dst, Python.tuple(call.dst.cols), context)
        _export(plan, call, MEMBER_BLOCK_FAMILY + 1)
        var mb_labels = _i32_array(call.dst, Python.tuple(call.dst.cols), context)
        _export(plan, call, MEMBER_BLOCK_SCALARS)
        var mb = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data)
        var mb_inertia = mb[0]
        var steps = Int(mb[1])
        var iterations = Int(mb[2])
        _set(model, "cluster_centers_", mb_centers)
        _set(model, "counts_", counts)
        _set(model, "labels_", mb_labels)
        _set(model, "inertia_", PythonObject(mb_inertia))
        _set(model, "n_steps_", PythonObject(steps))
        _set(model, "n_iter_", PythonObject(iterations))
        _set(model, "n_features_in_", PythonObject(cols))
        return
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


def _minibatch_state(fields: PythonObject, plan: MemberPlan, centers: PythonObject) raises -> Int:
    var scalars: List[Float64] = [0.0, 0.0, 0.0]
    var names: List[String] = ["inertia_", "n_steps_", "n_iter_"]
    for i in range(3):  # driver-loop: three fitted scalar attributes
        var value = py_get(fields, PythonObject(names[i]), PythonObject(None))
        if value is not None:
            scalars[i] = Float64(py=value)
    var call = member_new(plan)
    try:
        call.block = MEMBER_BLOCK_FAMILY
        call.x = MemberMatrix(Int(py=classical_address(centers)), Int(py=centers.shape[0]), Int(py=centers.shape[1]))
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, 3, MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = centers
    _ = len(scalars)
    return call.state


def xcluster_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding the fitted leaf centers, tree and
    inertia of `model` (labels_ are not needed to predict and are not
    imported). A MiniBatchKMeans state holds its centers and
    [inertia_, n_steps_, n_iter_]. The caller frees it (`member_free`)."""
    var kind = Int(plan.params[0])
    if kind != _KIND_BISECT and kind != _KIND_MINIBATCH:
        raise Error("RuntimeError: mojolearn: this x_cluster member kind is not published")
    classical_connect_numpy(context)
    var source = py_get(fields, PythonObject("cluster_centers_"), PythonObject(None))
    if source is None:
        raise Error("NotFittedError: this member is not fitted yet")
    var centers = admit_family(source, 2, "cluster_centers_", context, finite=True).array
    if kind == _KIND_MINIBATCH:
        return _minibatch_state(fields, plan, centers)
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
