# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the Dart family (DARTClassifier DARTRegressor),
UNPACK / PACK of core/dart_member.mojo.

  * `dart_member_admit` (UNPACK, once per member): the estimator's DART
    settings with the door's own checks (`dart_words`), its first one-tree
    member validated once (`dart_first_request`: the member's constructor
    checks, read on a fresh member; the template estimator is not mutated)
    and the members' typed configuration (`dart_member_config`); the route (the run
    context of the estimator's device, `trees_route_context`; its x_trees
    image's DART entry, glue entry and key; the rf image's forest entry);
    the parameter words of core/dart_member.mojo.
  * `dart_member_pack` (PACK): the fitted attributes the estimator's own fit
    writes (`dart_pack_rounds`: `init_score_`, `n_classes_`, the `trees_`
    members with their model arrays, `tree_values_`, the coefficients and
    weights, `n_features_in_`; `_KIND` for a classifier) from the state's
    blocks. The caller publishes `classes_`.
  * `dart_member_state` (UNPACK): a member state from a fitted estimator's
    members, leaf values, coefficients, weights and starts.

SERIAL BY NATURE: one member's admission or publication (driver loops over
its fitted trees); a caller runs its members in parallel.
"""
from std.memory import unsafe_memcpy, bitcast
from std.python import Python, PythonObject
from bindings.member_owner_unpack import MemberOwner
from bindings.py_balanced import py_call, py_get, py_item, py_setitem, py_fields
from bindings.admitted_input_unpack import classical_view, admitted_dtype
from bindings.classical_arrays_door import classical_address
from bindings.trees_route_boundary_door import trees_route_context
from bindings.trees_side_records_unpack import trees_side_drop
from bindings.py_trees_resident import trees_resident_forget
from bindings.py_model_records import model_identity_key
from bindings.trees_public_boost_door import (
    dart_words, dart_first_request, dart_pack_rounds, _xt, _xk, _DART_SLOT, dart_member_shape,
)
from bindings.trees_public_forest_door import forest_key
from bindings.trees_dart_rounds_run import (
    DartRounds, DART_M_WORDS, DART_M_OFF_START, DART_M_OFF_N, DART_M_NODE_START, DART_M_NODES,
    DART_M_LEAF_START, DART_M_LEAF_N, DART_M_VAL_START, DART_M_VAL_N, DART_M_COLS, DART_M_TREES, DART_M_OUTPUTS,
    dart_member_config,
)
from xtrees.dart_calls import xd_ref
from xtrees.glue_calls import xt_ref
from core.device_route import DeviceRoute, DEVICE_CPU, DEVICE_GPU
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free,
    MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_SCALARS, MEMBER_IMPORT_DONE,
    MEMBER_F32, MEMBER_F64, MEMBER_I32,
)

comptime _ARENAS = 6


def _int_word(value: Int) -> Float64:
    return bitcast[DType.float64](Int64(value))


def dart_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted estimator (see the module docstring)."""
    var run = trees_route_context(model, context)
    var words = dart_words(fields, run)
    var xt = _xt(fields, run)
    var key = _xk(xt)
    var route = DeviceRoute(DEVICE_CPU if key.vendor == 0 else DEVICE_GPU)
    route.mode = key.mode
    route.vendor = key.vendor
    route.entry = xd_ref(key)
    route.second = xt_ref(key)
    route.has_second = True
    var params = List[Float64](capacity=27 + 60)
    params.append(0.0 if kind == 0 else 1.0)
    params.append(Float64(words.n_iters))
    params.append(Float64(words.node_cap))
    params.append(_int_word(words.skip_thr))
    params.append(Float64(words.uniform))
    params.append(Float64(words.xgb))
    params.append(Float64(words.max_drop))
    params.append(words.drop_rate)
    params.append(words.lr)
    params.append(words.lam)
    params.append(words.l1)
    params.append(words.mds)
    params.append(1.0 if words.bag else 0.0)
    params.append(Float64(words.bag_freq))
    params.append(words.bag_frac)
    params.append(_int_word(words.bag_seed))
    params.append(words.col_frac)
    params.append(_int_word(words.col_seed))
    params.append(1.0 if words.col_seed_refusal != "" else 0.0)
    params.append(_int_word(words.seed))
    params.append(_int_word(words.drop_seed))
    var pins = Python.list(xt, run)
    if words.n_iters > 0:
        # The first member's constructor checks, once (the estimator's fit
        # runs them on its first member the same way).
        var built = dart_first_request(fields, words.seed, run)
        # The members' configuration, typed (lane bm-trees:
        # bindings/trees_dart_rounds_run.mojo `dart_member_config`), as the
        # door's fit takes it; the constructor above ran its checks.
        var config = dart_member_config(dart_member_shape(fields), words.seed)
        route.third = forest_key(built[2])
        route.has_third = True
        _ = py_call(pins.append, built[2])
        params.append(Float64(config.criterion))
        params.append(Float64(config.features.form))
        params.append(Float64(config.features.count))
        params.append(config.features.fraction)
        params.append(1.0 if config.bins_image else 0.0)
        var slots = len(config.ints)
        params.append(Float64(slots))
        for k in range(slots):  # driver-loop: over the member configuration's parameter slots (about twenty)
            params.append(_int_word(config.ints[k]))
            params.append(config.floats[k])
            params.append(1.0 if config.bad[k] != 0 else 0.0)
    else:
        for _ in range(5):  # small-loop: the five member configuration words, unused without trees
            params.append(0.0)
        params.append(0.0)
    owner.plan.route = route
    owner.plan.params = params^
    owner.pins = pins
    return True


def _block_into_i32(plan: MemberPlan, mut call: MemberCall, block: Int) raises -> List[Int32]:
    call.block = block
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var count = call.dst.rows * call.dst.cols
    var words = List[Int32](length=count, fill=0)
    if count > 0:
        unsafe_memcpy(dest=words.unsafe_ptr(), src=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=call.dst.data),
            count=count)
    return words^


def _block_into_f32(plan: MemberPlan, mut call: MemberCall, block: Int) raises -> List[Float32]:
    call.block = block
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var count = call.dst.rows * call.dst.cols
    var words = List[Float32](length=count, fill=0.0)
    if count > 0:
        unsafe_memcpy(dest=words.unsafe_ptr(),
            src=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=call.dst.data), count=count)
    return words^


def _block_into_f64(plan: MemberPlan, mut call: MemberCall, block: Int) raises -> List[Float64]:
    call.block = block
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var count = call.dst.rows * call.dst.cols
    var words = List[Float64](length=count, fill=0.0)
    if count > 0:
        unsafe_memcpy(dest=words.unsafe_ptr(),
            src=MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data), count=count)
    return words^


def dart_member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: write the fitted attributes of state `state` into `model`, as
    the estimator's own fit does (see the module docstring)."""
    var run = trees_route_context(model, context)
    trees_side_drop(_DART_SLOT, model)
    trees_resident_forget(model_identity_key(model))
    var call = MemberCall(plan.route, state)
    var rounds = DartRounds()
    rounds.offsets = _block_into_i32(plan, call, MEMBER_BLOCK_FAMILY)
    rounds.colid = _block_into_i32(plan, call, MEMBER_BLOCK_FAMILY + 1)
    rounds.quesval = _block_into_f32(plan, call, MEMBER_BLOCK_FAMILY + 2)
    rounds.left = _block_into_i32(plan, call, MEMBER_BLOCK_FAMILY + 3)
    rounds.leaves = _block_into_f32(plan, call, MEMBER_BLOCK_FAMILY + 4)
    rounds.values = _block_into_f32(plan, call, MEMBER_BLOCK_FAMILY + 5)
    var member_words = _block_into_f64(plan, call, MEMBER_BLOCK_FAMILY + 6)
    rounds.members = List[Int](length=len(member_words), fill=0)
    for w in range(len(member_words)):  # driver-loop: over the members' words (11 per tree, PACK)
        rounds.members[w] = Int(member_words[w])
    rounds.coefs = _block_into_f64(plan, call, MEMBER_BLOCK_FAMILY + 7)
    rounds.weights = _block_into_f64(plan, call, MEMBER_BLOCK_FAMILY + 8)
    var scalars = _block_into_f64(plan, call, MEMBER_BLOCK_SCALARS)
    var d = Int(scalars[0])
    var K = Int(scalars[1])
    var kind = Int(scalars[2])
    var starts = List[Float64]()
    for c in range(K):  # small-loop(K): the class starts
        starts.append(scalars[4 + c])
    var words = dart_words(fields, run)
    var first = PythonObject(None)
    var first_fields = PythonObject(None)
    if rounds.count() > 0:
        var built = dart_first_request(fields, words.seed, run)
        first = built[0]
        first_fields = built[1]
    if kind != 0:
        py_setitem(fields, "_KIND", PythonObject(kind))
    dart_pack_rounds(model, fields, rounds, starts, K, d, first, first_fields, words.seed, run)


def _view_of(source: PythonObject, typestr: String, context: PythonObject) raises -> PythonObject:
    var view = classical_view(source, context, "DART model")
    if admitted_dtype(view, context) != typestr:
        raise Error("TypeError: mojolearn: a DART model array of dtype " + admitted_dtype(view, context)
            + " where " + typestr + " was expected")
    return view


def dart_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding the fitted members of `model`
    (their five model arrays and leaf values, laid into the six arenas), its
    coefficients, weights and starts. The caller frees it (`member_free`)."""
    var trees_source = py_get(fields, PythonObject("trees_"))
    if trees_source is None:
        raise Error("NotFittedError: this member is not fitted yet")
    var trees = trees_source
    var values = py_item(fields, PythonObject("tree_values_"))
    var count = len(trees)
    if len(values) != count:
        raise Error("ValueError: a DART model holds one leaf-value array per tree")
    var names = Python.tuple("_offsets", "_colid", "_quesval", "_left_child", "_leaves")
    var typestrs: List[String] = ["<i4", "<i4", "<f4", "<i4", "<f4", "<f4"]
    # Per tree: six views (the five model arrays and its leaf values).
    var views = List[PythonObject]()
    var sizes = List[Int]()
    var totals = List[Int](length=_ARENAS, fill=0)
    var members = List[Float64](length=count * DART_M_WORDS, fill=0.0)
    for j in range(count):  # driver-loop: over the fitted member trees, once per import
        var tf = py_fields(trees[j])
        var base = j * DART_M_WORDS
        for f in range(_ARENAS):  # driver-loop: over the six arrays of one tree
            var source = values[j] if f == 5 else py_item(tf, names[f])
            var view = _view_of(source, typestrs[f], context)
            var size = Int(py=view.size)
            views.append(view)
            sizes.append(size)
        members[base + DART_M_OFF_START] = Float64(totals[0])
        members[base + DART_M_OFF_N] = Float64(sizes[j * _ARENAS])
        members[base + DART_M_NODE_START] = Float64(totals[1])
        members[base + DART_M_NODES] = Float64(sizes[j * _ARENAS + 1])
        members[base + DART_M_LEAF_START] = Float64(totals[4])
        members[base + DART_M_LEAF_N] = Float64(sizes[j * _ARENAS + 4])
        members[base + DART_M_VAL_START] = Float64(totals[5])
        members[base + DART_M_VAL_N] = Float64(sizes[j * _ARENAS + 5])
        members[base + DART_M_COLS] = Float64(py=py_item(tf, PythonObject("n_features_in_")))
        members[base + DART_M_TREES] = Float64(py=py_item(tf, PythonObject("_n_trees")))
        members[base + DART_M_OUTPUTS] = Float64(py=py_item(tf, PythonObject("_num_outputs")))
        for f in range(_ARENAS):  # driver-loop: over the six arrays of one tree
            totals[f] += sizes[j * _ARENAS + f]
    if totals[1] != totals[2] or totals[1] != totals[3]:
        raise Error("ValueError: a DART member's node arrays differ in length")
    var arena_i32 = List[List[Int32]]()
    var arena_f32 = List[List[Float32]]()
    for f in range(_ARENAS):  # driver-loop: over the six arenas
        if typestrs[f] == "<i4":
            arena_i32.append(List[Int32](length=max(totals[f], 1), fill=0))
        else:
            arena_f32.append(List[Float32](length=max(totals[f], 1), fill=0.0))
    var at = List[Int](length=_ARENAS, fill=0)
    for j in range(count):  # driver-loop: over the fitted member trees (one copy per array)
        var i32 = 0
        var f32 = 0
        for f in range(_ARENAS):  # driver-loop: over the six arrays of one tree
            var size = sizes[j * _ARENAS + f]
            var src = Int(py=classical_address(views[j * _ARENAS + f]))
            var dst = 0
            if typestrs[f] == "<i4":
                dst = Int(arena_i32[i32].unsafe_ptr()) + 4 * at[f]
                i32 += 1
            else:
                dst = Int(arena_f32[f32].unsafe_ptr()) + 4 * at[f]
                f32 += 1
            if size > 0:
                unsafe_memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=dst),
                    src=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=src), count=4 * size)
            at[f] += size
    var coefs = List[Float64]()
    var coef_source = py_item(fields, PythonObject("tree_coefs_"))
    for i in range(len(coef_source)):  # driver-loop: over the tree coefficients (trees-sized, UNPACK)
        coefs.append(Float64(py=coef_source[i]))
    var weights = List[Float64]()
    var weight_source = py_item(fields, PythonObject("tree_weights_"))
    for i in range(len(weight_source)):  # driver-loop: over the tree weights (rounds-sized, UNPACK)
        weights.append(Float64(py=weight_source[i]))
    var K = Int(py=py_item(fields, PythonObject("n_classes_")))
    var kind_source = py_get(fields, PythonObject("_KIND"))
    var kind = 0 if plan.kind != 0 or kind_source is None else Int(py=kind_source)
    var classes_source = py_get(fields, PythonObject("classes_"))
    var classes = 0 if plan.kind != 0 or classes_source is None else len(classes_source)
    var scalars: List[Float64] = [Float64(py=py_item(fields, PythonObject("n_features_in_"))), Float64(K),
        Float64(kind), Float64(classes)]
    var init = py_item(fields, PythonObject("init_score_"))
    if K == 1:
        scalars.append(Float64(py=init))
    else:
        for c in range(K):  # small-loop(K): the class starts
            scalars.append(Float64(py=init[c]))
    var call = member_new(plan)
    try:
        var blocks: List[Int] = [Int(arena_i32[0].unsafe_ptr()), Int(arena_i32[1].unsafe_ptr()),
            Int(arena_f32[0].unsafe_ptr()), Int(arena_i32[2].unsafe_ptr()), Int(arena_f32[1].unsafe_ptr()),
            Int(arena_f32[2].unsafe_ptr())]
        for f in range(_ARENAS):  # driver-loop: over the six arenas
            call.block = MEMBER_BLOCK_FAMILY + f
            call.x = MemberMatrix(blocks[f], 1, totals[f], MEMBER_I32 if typestrs[f] == "<i4" else MEMBER_F32)
            _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = MEMBER_BLOCK_FAMILY + 6
        call.x = MemberMatrix(Int(members.unsafe_ptr()), count, DART_M_WORDS, MEMBER_F64)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = MEMBER_BLOCK_FAMILY + 7
        call.x = MemberMatrix(Int(coefs.unsafe_ptr()), 1, len(coefs), MEMBER_F64)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = MEMBER_BLOCK_FAMILY + 8
        call.x = MemberMatrix(Int(weights.unsafe_ptr()), 1, len(weights), MEMBER_F64)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, len(scalars), MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = views^
    _ = len(members)
    _ = len(coefs)
    _ = len(weights)
    _ = len(scalars)
    _ = len(arena_i32)
    _ = len(arena_f32)
    return call.state
