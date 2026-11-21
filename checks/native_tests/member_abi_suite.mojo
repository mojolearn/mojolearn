# SPDX-License-Identifier: Apache-2.0
"""The member ABI (core/member_abi.mojo) and the lifetime of the linear
member state store (core/linear_member.mojo), without a kernel: five
independent fixtures, one host task each (capabilities; handles; refusals,
with every version 2 operation the linear member lacks; the fitted-state
blocks in and out; the family table and a planned member), then one fixture
per further published family (`_PUBLISHED` of them, one host task each): its
capability mask says what its kind of estimator is and nothing it is not, a
call without a state is refused in words, and the width of an output is
answered without a state where the family's words decide it (KMeans). A fit
or a prediction with numbers needs the family's kernel tables and belongs to
the installed-wheel tests.

A family joins by one row in `_published` and `_published_masks` when its
publication merges (MEMBER-ABI-V2.md)."""
from core.device_route import DeviceRoute
from core.entry_table import EntryRef
from core.host_parallel import host_parallelize
from core.decomposition_member import decomposition_member_v2
from core.extra_trees_member import extra_trees_member_v2
from core.forest_member import forest_member_v2
from core.kernel_svm_member import kernel_svm_member_v2
from core.kmeans_member import kmeans_member_v2
from core.linear_member import linear_member_v2
from core.logistic_member import logistic_member_v2
from core.neighbors_member import neighbors_member_v2
from core.scaler_member import scaler_member_v2
from core.tree_member import tree_member_v2
from core.xlinear_member import xlinear_member_v2
from core.member_abi import (
    MemberCall, MemberMatrix, MemberPlan, MemberRef, member_call, member_publish, member_resolve_family,
    member_new, member_free, member_require, member_outputs, MEMBER_NEEDS_PREDICTOR, MEMBER_NEEDS_TRANSFORMER,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT,
    MEMBER_OP_PREDICT, MEMBER_OP_PREDICT_PROBA, MEMBER_OP_DECISION, MEMBER_OP_OUTPUTS, MEMBER_OP_TRANSFORM,
    MEMBER_OP_INVERSE_TRANSFORM, MEMBER_OP_PARTIAL_FIT, MEMBER_OP_PREDICT_STD, MEMBER_OP_SCORE_SAMPLES,
    MEMBER_OP_EXPORT, MEMBER_OP_IMPORT,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_SCORE, MEMBER_CAN_PREDICT_PROBA, MEMBER_CAN_EXPORT,
    MEMBER_CAN_IMPORT, MEMBER_CAN_SAMPLE_WEIGHT, MEMBER_CAN_COEF, MEMBER_CAN_TRANSFORM,
    MEMBER_CAN_WEIGHTED_SCORE, MEMBER_CAN_DEVICE_VIEWS, MEMBER_CAN_CLASSIFIER,
    MEMBER_BLOCK_COEF, MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_BY_INDEX, MEMBER_IMPORT_DONE,
    MEMBER_F32, MEMBER_F64, MEMBER_HOME_DEVICE,
)
from core.dart_member import dart_member_v2
from core.isolation_forest_member import isolation_forest_member_v2
from core.xcluster_member import xcluster_member_v2
from core.mixture_member import mixture_member_v2
from core.density_member import density_member_v2
from core.hierarchy_member import hierarchy_member_v2
from core.spectral_member import spectral_member_v2
from core.member_abi import member_output_dtype, MEMBER_I32, MEMBER_CAN_SCORE_SAMPLES, MEMBER_DTYPE_UNSAID
from core.member_families import MEMBER_FAMILY_SPARE3, MEMBER_FAMILY_COUNT, member_family_name

comptime _LINEAR_FIXTURES = 5
#: Published families beyond linear, in `_published` order: Logistic 1,
#: XLinear 2, Neighbors 11, Tree 13, Forest 14, ExtraTrees 15, KernelSvm 6,
#: KMeans 24, Decomposition 31, Scaler 34.
comptime _PUBLISHED = 10
#: Sweep-b families (2026-11-21), in `_sweep_b` order: Dart 17,
#: IsolationForest 18, XCluster 25, Mixture 26, Density 27, Hierarchy 29,
#: Spectral 30.
comptime _SWEEP_B = 7
comptime MEMBER_ABI_FIXTURES = _LINEAR_FIXTURES + _PUBLISHED + _SWEEP_B


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _member() -> MemberRef:
    return MemberRef(EntryRef(linear_member_v2, 0), 0)


def _new(kind: Float64, alpha: Float64) raises -> Int:
    var params: List[Float64] = [kind, 1.0, alpha]
    var call = MemberCall(DeviceRoute(0))
    call.params = Int(params.unsafe_ptr())
    call.param_count = 3
    _ = member_call(_member(), MEMBER_OP_NEW, call)
    _ = len(params)
    return call.state


def _refused(operation: Int, mut call: MemberCall, words: String) raises:
    var refused = False
    try:
        _ = member_call(_member(), operation, call)
    except error:
        refused = String(error).find(words) >= 0
    require(refused, "member ABI: expected a refusal holding '" + words + "'")


def _published(which: Int) raises -> MemberRef:
    """The member entry of published family `which` (no table: the entry
    itself, as the linear fixtures call theirs)."""
    if which == 0:
        return MemberRef(EntryRef(logistic_member_v2, 0), 0)
    if which == 1:
        return MemberRef(EntryRef(xlinear_member_v2, 0), 0)
    if which == 2:
        return MemberRef(EntryRef(neighbors_member_v2, 0), 0)
    if which == 3:
        return MemberRef(EntryRef(tree_member_v2, 0), 0)
    if which == 4:
        return MemberRef(EntryRef(forest_member_v2, 0), 0)
    if which == 5:
        return MemberRef(EntryRef(extra_trees_member_v2, 0), 0)
    if which == 6:
        return MemberRef(EntryRef(kernel_svm_member_v2, 0), 0)
    if which == 7:
        return MemberRef(EntryRef(kmeans_member_v2, 0), 0)
    if which == 8:
        return MemberRef(EntryRef(decomposition_member_v2, 0), 0)
    if which == 9:
        return MemberRef(EntryRef(scaler_member_v2, 0), 0)
    raise Error("member ABI: no published fixture " + String(which))


def _published_fixture(which: Int) raises:
    """One published family, without a kernel and without a state."""
    var member = _published(which)
    var probe = MemberCall(DeviceRoute(0))
    member.capabilities = member_call(member, MEMBER_OP_CAPABILITIES, probe)
    var name = String(which)
    # what the family is: predictors predict, transformers transform, and
    # the form a caller asks for (`has`: every bit) follows
    var transformer = which == 8 or which == 9
    if transformer:
        require(member.has(MEMBER_NEEDS_TRANSFORMER) and not member.can(MEMBER_CAN_PREDICT)
            and not member.has(MEMBER_NEEDS_PREDICTOR), "member ABI: published " + name + " is a transformer")
    else:
        require(member.has(MEMBER_NEEDS_PREDICTOR), "member ABI: published " + name + " fits and predicts")
    var classifier = which == 0 or which == 2
    if classifier:
        require(member.can(MEMBER_CAN_CLASSIFIER), "member ABI: published " + name + " has classifier kinds")
    if which == 1 or which >= 7:
        require(not member.can(MEMBER_CAN_CLASSIFIER), "member ABI: published " + name + " has no classifier kind")
    if which == 7:
        require(member.can(MEMBER_CAN_TRANSFORM), "member ABI: KMeans transforms")
    # a call without a state is refused in words, never a wild read
    var stateless = MemberCall(DeviceRoute(0))
    var refused = False
    try:
        _ = member_call(member, MEMBER_OP_TRANSFORM if transformer else MEMBER_OP_PREDICT, stateless)
    except error:
        refused = len(String(error)) > 0
    require(refused, "member ABI: published " + name + " refuses a call without a state")
    if which == 7:
        # the width of an output before any state: KMeans' words decide it
        var plan = MemberPlan()
        plan.member = member
        plan.route = DeviceRoute(0)
        var words: List[Float64] = [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        plan.params = words^
        require(member_outputs(plan, MEMBER_OP_TRANSFORM, 3, 0) == 5 and member_outputs(plan, MEMBER_OP_PREDICT, 3, 0) == 1,
            "member ABI: KMeans answers its output widths without a state")


def _fixture(unit: Int) raises:
    if unit >= _LINEAR_FIXTURES + _PUBLISHED:
        _sweep_b_fixture(unit - _LINEAR_FIXTURES - _PUBLISHED)
        return
    if unit >= _LINEAR_FIXTURES:
        _published_fixture(unit - _LINEAR_FIXTURES)
        return
    if unit == 0:
        # capabilities and the output width
        var call = MemberCall(DeviceRoute(0))
        var mask = member_call(_member(), MEMBER_OP_CAPABILITIES, call)
        require(mask == (MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_SCORE | MEMBER_CAN_EXPORT
            | MEMBER_CAN_IMPORT | MEMBER_CAN_SAMPLE_WEIGHT | MEMBER_CAN_COEF), "member ABI: linear capability mask")
        require((mask & (MEMBER_CAN_PREDICT_PROBA | MEMBER_CAN_TRANSFORM | MEMBER_CAN_WEIGHTED_SCORE
            | MEMBER_CAN_DEVICE_VIEWS | MEMBER_CAN_CLASSIFIER)) == 0,
            "member ABI: a linear regressor has no predict_proba, transform, weighted score, device views or classes")
        call.classes = 7
        _ = member_call(_member(), MEMBER_OP_OUTPUTS, call)
        require(call.out_cols == 1 and call.classes == 0, "member ABI: one output column, no classes")
        # a contiguous fold of a host view is a view: no copy
        var view = MemberMatrix(4096, 10, 3)
        var fold = view.rows_from(2, 5)
        require(fold.data == 4096 + 2 * 3 * 4 and fold.rows == 5 and fold.cols == 3 and fold.dense_host(MEMBER_F32),
            "member ABI: a host fold view")
        view.home = MEMBER_HOME_DEVICE
        fold = view.rows_from(2, 5)
        require(fold.data == 4096 and fold.offset == 2 and not fold.dense_host(MEMBER_F32),
            "member ABI: a device fold view moves its offset")
    elif unit == 1:
        # two live states are two handles; a released handle is refused, never read
        var first = _new(1.0, 0.5)
        var second = _new(0.0, 0.0)
        require(first != 0 and second != 0 and first != second, "member ABI: two states, two handles")
        var call = MemberCall(DeviceRoute(0), first)
        _ = member_call(_member(), MEMBER_OP_FREE, call)
        require(call.state == 0, "member ABI: free clears the caller's handle")
        var stale = MemberCall(DeviceRoute(0), first)
        _refused(MEMBER_OP_FREE, stale, "released or never created")
        var third = _new(1.0, 1.0)
        require(third != first, "member ABI: a reused slot is a new generation")
        var other = MemberCall(DeviceRoute(0), second)
        _ = member_call(_member(), MEMBER_OP_FREE, other)
        var last = MemberCall(DeviceRoute(0), third)
        _ = member_call(_member(), MEMBER_OP_FREE, last)
    elif unit == 3:
        # fitted-state blocks: import three, read them back as views of the state, release
        var handle = _new(1.0, 0.25)
        var coef: List[Float32] = [1.5, -2.0, 0.5]
        var mean: List[Float32] = [0.1, 0.2, 0.3]
        var scalars: List[Float64] = [4.0, 9.0]
        var call = MemberCall(DeviceRoute(0), handle)
        _refused(MEMBER_OP_EXPORT, call, "not fitted yet")
        call.block = MEMBER_BLOCK_COEF
        call.x = MemberMatrix(Int(coef.unsafe_ptr()), 1, 3)
        _ = member_call(_member(), MEMBER_OP_IMPORT, call)
        call.block = MEMBER_BLOCK_FAMILY
        call.x = MemberMatrix(Int(mean.unsafe_ptr()), 1, 3)
        _ = member_call(_member(), MEMBER_OP_IMPORT, call)
        _refused(MEMBER_OP_EXPORT, call, "not fitted yet")
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, 2, MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(_member(), MEMBER_OP_IMPORT, call)
        call.flags = 0
        call.block = MEMBER_BLOCK_COEF
        _ = member_call(_member(), MEMBER_OP_EXPORT, call)
        require(call.out_cols == 3 and call.dst.rows == 1 and call.dst.cols == 3 and call.dst.dense_host(MEMBER_F32),
            "member ABI: the coefficient block's shape")
        require(call.dst.data != Int(coef.unsafe_ptr()), "member ABI: an import copies; the state owns its blocks")
        var back = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=call.dst.data)
        require(back[0] == 1.5 and back[1] == -2.0 and back[2] == 0.5, "member ABI: the coefficient block's words")
        call.flags = MEMBER_BLOCK_BY_INDEX
        call.block = 1
        _ = member_call(_member(), MEMBER_OP_EXPORT, call)
        require(call.block == MEMBER_BLOCK_SCALARS and call.dst.dense_host(MEMBER_F64) and call.dst.cols == 2,
            "member ABI: block 1 is the scalars")
        var words = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data)
        require(words[0] == 4.0 and words[1] == 9.0, "member ABI: the scalars block's words")
        call.block = 2
        _ = member_call(_member(), MEMBER_OP_EXPORT, call)
        require(call.block == MEMBER_BLOCK_FAMILY and call.dst.cols == 3, "member ABI: block 2 is the column means")
        call.block = 3
        _refused(MEMBER_OP_EXPORT, call, "three fitted blocks")
        call.flags = 0
        call.block = 999
        _refused(MEMBER_OP_IMPORT, call, "no fitted block")
        _ = len(coef)
        _ = len(mean)
        _ = len(scalars)
        _ = member_call(_member(), MEMBER_OP_FREE, call)
    elif unit == 4:
        # the family table: publish in a spare family's CPU identical slot, resolve by number, plan a member
        require(member_family_name(MEMBER_FAMILY_SPARE3) == "Spare3" and member_family_name(MEMBER_FAMILY_COUNT) == "unknown",
            "member ABI: family names")
        var anchor: List[Int] = [1]
        member_publish[MEMBER_FAMILY_SPARE3](1, 0, Int(anchor.unsafe_ptr()), linear_member_v2)
        member_publish[MEMBER_FAMILY_SPARE3](1, 0, Int(anchor.unsafe_ptr()), linear_member_v2)
        var route = DeviceRoute(0)
        route.mode = 1
        route.vendor = 0
        var plan = MemberPlan()
        plan.family = MEMBER_FAMILY_SPARE3
        plan.member = member_resolve_family(plan.family, route)
        plan.route = route
        var ridge_words: List[Float64] = [1.0, 1.0, 0.5]
        plan.params = ridge_words^
        require(plan.member.can(MEMBER_CAN_FIT) and not plan.member.can(MEMBER_CAN_PREDICT_PROBA),
            "member ABI: the resolved entry's capability mask")
        member_require(plan, MEMBER_CAN_SCORE, "score")
        require(plan.member.has(MEMBER_NEEDS_PREDICTOR) and not plan.member.has(MEMBER_NEEDS_TRANSFORMER)
            and member_outputs(plan, MEMBER_OP_PREDICT, 3, 1) == 1,
            "member ABI: a predictor plan, its width asked without a state")
        var partly = False
        try:
            member_require(plan, MEMBER_CAN_FIT | MEMBER_CAN_TRANSFORM, "fit and transform")
        except error:
            partly = String(error).find("has no fit and transform") >= 0
        require(partly, "member ABI: member_require tests every bit of its mask")
        var lacks = False
        try:
            member_require(plan, MEMBER_CAN_TRANSFORM, "transform")
        except error:
            lacks = String(error).find("has no transform") >= 0
        require(lacks, "member ABI: a missing capability is refused at admission")
        var call = member_new(plan)
        require(call.state != 0, "member ABI: a planned member mints a state")
        var first = call.state
        member_free(plan, call)
        require(call.state == 0, "member ABI: member_free clears the handle")
        member_free(plan, call)
        var stale = MemberCall(route, first)
        _refused(MEMBER_OP_FREE, stale, "released or never created")
        route.vendor = 3
        var absent = False
        try:
            _ = member_resolve_family(plan.family, route)
        except error:
            absent = String(error).find("not published") >= 0
        require(absent, "member ABI: an image that published nothing is refused at resolution")
        _ = len(anchor)
    else:
        # refusals in words: bad parameters, an unfitted predict, an operation the family lacks
        var bad = MemberCall(DeviceRoute(0))
        _refused(MEMBER_OP_NEW, bad, "three parameters")
        var handle = _new(1.0, 0.0)
        var storage: List[Float32] = [1.0, 2.0, 0.0]
        var call = MemberCall(DeviceRoute(0), handle)
        call.x = MemberMatrix(Int(storage.unsafe_ptr()), 1, 2)
        call.dst = MemberMatrix(Int(storage.unsafe_ptr()) + 8, 1, 1)
        _refused(MEMBER_OP_PREDICT, call, "call fit before predict")
        _refused(MEMBER_OP_PREDICT_PROBA, call, "has no predict_proba")
        _refused(MEMBER_OP_DECISION, call, "has no decision_function")
        _refused(MEMBER_OP_TRANSFORM, call, "has no transform")
        _refused(MEMBER_OP_INVERSE_TRANSFORM, call, "has no inverse_transform")
        _refused(MEMBER_OP_PARTIAL_FIT, call, "has no partial_fit")
        _refused(MEMBER_OP_PREDICT_STD, call, "has no predict_std")
        _refused(MEMBER_OP_SCORE_SAMPLES, call, "has no score_samples")
        # a selection or an output the linear member does not read is refused before any kernel
        var picks: List[Int] = [0]
        call.row_index = MemberMatrix(Int(picks.unsafe_ptr()), 1, 1)
        _refused(MEMBER_OP_FIT, call, "no row or column index")
        _refused(MEMBER_OP_PREDICT, call, "no row or column index")
        call.row_index = MemberMatrix()
        call.fit_output = MEMBER_OP_TRANSFORM
        _refused(MEMBER_OP_FIT, call, "has no transform")
        call.fit_output = 0
        _ = len(picks)
        _ = len(storage)
        _ = member_call(_member(), MEMBER_OP_FREE, call)


def run_checks() raises:
    var flags = List[UInt8](length=MEMBER_ABI_FIXTURES, fill=UInt8(0))
    var flag_ptr = rebind[MutPointer[UInt8, MutUntrackedOrigin]](flags.unsafe_ptr())

    def _one(unit: Int) {imm flag_ptr}:
        try:
            _fixture(unit)
        except:
            flag_ptr.unsafe_store(unit, UInt8(1))

    host_parallelize(_one, MEMBER_ABI_FIXTURES)
    for unit in range(MEMBER_ABI_FIXTURES):  # driver-loop: over the fixtures' flags
        if flags[unit] != UInt8(0):
            # Raise the failing fixture's own words.
            _fixture(unit)
            raise Error("member ABI: fixture " + String(unit) + " failed on its host task and passed alone")
    _ = len(flags)


def _sweep_b_member(which: Int) raises -> MemberRef:
    """The member entry of sweep-b family `which` (the entry itself)."""
    if which == 0:
        return MemberRef(EntryRef(dart_member_v2, 0), 0)
    if which == 1:
        return MemberRef(EntryRef(isolation_forest_member_v2, 0), 0)
    if which == 2:
        return MemberRef(EntryRef(xcluster_member_v2, 0), 0)
    if which == 3:
        return MemberRef(EntryRef(mixture_member_v2, 0), 0)
    if which == 4:
        return MemberRef(EntryRef(density_member_v2, 0), 0)
    if which == 5:
        return MemberRef(EntryRef(hierarchy_member_v2, 0), 0)
    if which == 6:
        return MemberRef(EntryRef(spectral_member_v2, 0), 0)
    raise Error("member ABI: no sweep-b fixture " + String(which))


def _answers(member: MemberRef, var words: List[Float64], classes: Int, operation: Int, width: Int, dtype: Int,
    name: String) raises:
    """OUTPUTS without a state (items 10 and 26): the width and the element
    type of `operation`, from the parameter words alone."""
    var plan = MemberPlan()
    plan.member = member
    plan.route = DeviceRoute(0)
    plan.params = words^
    plan.classes = classes
    require(member_outputs(plan, operation, 3, 1) == width,
        "member ABI: " + name + " answers the width of operation " + String(operation) + " without a state")
    require(member_output_dtype(plan, operation, 3, 1, MEMBER_DTYPE_UNSAID) == dtype,
        "member ABI: " + name + " answers the element type of operation " + String(operation) + " without a state")


def _sweep_b_fixture(which: Int) raises:
    """One sweep-b family, without a kernel and without a state: what its
    kind is, a stateless call refused in words, OUTPUTS widths and element
    types answered from the parameter words."""
    var member = _sweep_b_member(which)
    var probe = MemberCall(DeviceRoute(0))
    member.capabilities = member_call(member, MEMBER_OP_CAPABILITIES, probe)
    var name = String("sweep-b ") + String(which)
    require(member.has(MEMBER_NEEDS_PREDICTOR) or member.can(MEMBER_CAN_SCORE_SAMPLES),
        "member ABI: " + name + " fits and predicts or scores")
    require(member.can(MEMBER_CAN_CLASSIFIER) == (which == 0),
        "member ABI: " + name + " is a classifier family only for DART")
    var stateless = MemberCall(DeviceRoute(0))
    var refused = False
    try:
        _ = member_call(member, MEMBER_OP_PREDICT, stateless)
    except error:
        refused = len(String(error)) > 0
    require(refused, "member ABI: " + name + " refuses a call without a state")
    if which == 0:
        # DART: kind word 0 classifier (codes int32, proba float64), 1 regressor
        _answers(member, [0.0], 3, MEMBER_OP_PREDICT, 1, MEMBER_I32, name)
        _answers(member, [0.0], 3, MEMBER_OP_PREDICT_PROBA, 3, MEMBER_F64, name)
        _answers(member, [1.0], 0, MEMBER_OP_PREDICT, 1, MEMBER_F32, name)
    elif which == 1:
        _answers(member, List[Float64](length=13, fill=0.0), 0, MEMBER_OP_PREDICT, 1, MEMBER_I32, name)
        _answers(member, List[Float64](length=13, fill=0.0), 0, MEMBER_OP_DECISION, 1, MEMBER_F32, name)
    elif which == 2:
        # MiniBatchKMeans with four clusters
        var words: List[Float64] = [0.0, 4.0, 0.0, 1.0, 10.0, 0.0, 0.0, 16.0, -1.0, 0.0, 0.01]
        _answers(member, words.copy(), 0, MEMBER_OP_PREDICT, 1, MEMBER_I32, name)
        _answers(member, words^, 0, MEMBER_OP_TRANSFORM, 4, MEMBER_F32, name)
    elif which == 3:
        var words: List[Float64] = [3.0, 100.0, 0.001, 0.000001, 0.0, 0.0]
        _answers(member, words.copy(), 0, MEMBER_OP_PREDICT, 1, MEMBER_I32, name)
        _answers(member, words^, 0, MEMBER_OP_PREDICT_PROBA, 3, MEMBER_F32, name)
    elif which == 4:
        _answers(member, [1.0, 0.5, 5.0, 0.0, 0.0, 0.0, 0.0], 0, MEMBER_OP_PREDICT, 1, MEMBER_I32, name)
        _answers(member, [0.0, 1.0, 0.0, 0.0], 0, MEMBER_OP_SCORE_SAMPLES, 1, MEMBER_F32, name)
    else:
        _answers(member, List[Float64](length=8, fill=0.0), 0, MEMBER_OP_PREDICT, 1, MEMBER_I32, name)
