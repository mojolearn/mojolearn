# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
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
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require
from core.device_route import DeviceRoute
from core.entry_table import EntryRef
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
from core.sgd_member import sgd_member_v2
from core.cd_member import cd_member_v2
from core.linear_svm_member import linear_svm_member_v2
from core.kernel_ridge_member import kernel_ridge_member_v2
from core.kernel_approx_member import kernel_approx_member_v2
from core.gp_member import gp_member_v2
from core.mlp_member import mlp_member_v2
from core.sequence_member import sequence_member_v2
from core.cnn_member import cnn_member_v2
from core.pls_member import pls_member_v2
from core.forecast_member import forecast_member_v2
from core.forecast_seq_member import forecast_seq_member_v2
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
    MEMBER_F32, MEMBER_F64, MEMBER_HOME_DEVICE, MEMBER_I32, MEMBER_CAN_TEST_ROWS, MEMBER_CAN_DECISION,
)
from core.dart_member import dart_member_v2
from core.isolation_forest_member import isolation_forest_member_v2
from core.xcluster_member import xcluster_member_v2
from core.mixture_member import mixture_member_v2
from core.density_member import density_member_v2
from core.hierarchy_member import hierarchy_member_v2
from core.spectral_member import spectral_member_v2
from core.member_abi import member_output_dtype, MEMBER_I32, MEMBER_CAN_SCORE_SAMPLES, MEMBER_DTYPE_UNSAID
from core.member_abi import MEMBER_BORROWS_FIT, MEMBER_GIL_FREE, member_own
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
#: Sweep-a families without a fixture before (OUTPUTS width and element
#: type without a state, kind-exact capabilities from the NEW words), in
#: `_sweep_a_member` order: Sgd 3, Cd 4, LinearSvm 5, KernelRidge 7,
#: KernelApprox 8, Gp 23, Mlp 19, Sequence 20, Cnn 21, Pls 22, Forecast 37,
#: ForecastSeq 43.
comptime _SWEEP_A = 12
#: THE LIFETIME RULE (core/member_abi.mojo; lane fx-borrow, 2026-11-21): a
#: borrowing kind (the k-NN regressor) fitted on a buffer that is then
#: overwritten: 0 after the export of both blocks, 1 after MEMBER_EXPORT_OWN
#: (`member_own`).
comptime _BORROW = 2
comptime MEMBER_ABI_FIXTURES = _LINEAR_FIXTURES + _PUBLISHED + _SWEEP_B + _SWEEP_A + _BORROW


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
    # x_linear (1) publishes RidgeClassifier (kind 9) since 01d51610db: its family mask has the classifier bit
    var classifier = which == 0 or which == 1 or which == 2
    if classifier:
        require(member.can(MEMBER_CAN_CLASSIFIER), "member ABI: published " + name + " has classifier kinds")
    if which >= 7:
        require(not member.can(MEMBER_CAN_CLASSIFIER), "member ABI: published " + name + " has no classifier kind")
    if which == 7:
        require(member.can(MEMBER_CAN_TRANSFORM), "member ABI: KMeans transforms")
    # a call without a state is refused in words, never a wild read
    var stateless = MemberCall(DeviceRoute(0))
    var refused = False
    try:
        _ = member_call(member, MEMBER_OP_TRANSFORM if transformer else MEMBER_OP_PREDICT, stateless)
    except error:
        refused = String(error).byte_length() > 0
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


def _sweep_a_member(which: Int) raises -> MemberRef:
    if which == 0:
        return MemberRef(EntryRef(sgd_member_v2, 0), 0)
    if which == 1:
        return MemberRef(EntryRef(cd_member_v2, 0), 0)
    if which == 2:
        return MemberRef(EntryRef(linear_svm_member_v2, 0), 0)
    if which == 3:
        return MemberRef(EntryRef(kernel_ridge_member_v2, 0), 0)
    if which == 4:
        return MemberRef(EntryRef(kernel_approx_member_v2, 0), 0)
    if which == 5:
        return MemberRef(EntryRef(gp_member_v2, 0), 0)
    if which == 6:
        return MemberRef(EntryRef(mlp_member_v2, 0), 0)
    if which == 7:
        return MemberRef(EntryRef(sequence_member_v2, 0), 0)
    if which == 8:
        return MemberRef(EntryRef(cnn_member_v2, 0), 0)
    if which == 9:
        return MemberRef(EntryRef(pls_member_v2, 0), 0)
    if which == 10:
        return MemberRef(EntryRef(forecast_member_v2, 0), 0)
    if which == 11:
        return MemberRef(EntryRef(forecast_seq_member_v2, 0), 0)
    raise Error("member ABI: no sweep-a fixture " + String(which))


def _words(count: Int, kind: Float64) -> List[Float64]:
    """`count` NEW words, all zero but the kind word 0."""
    var words = List[Float64](length=count, fill=0.0)
    if count > 0:
        words[0] = kind
    return words^


def _ask(member: MemberRef, words: List[Float64], operation: Int, features: Int, targets: Int,
    classes: Int) raises -> MemberCall:
    """OUTPUTS for `operation` without a state, from the NEW words and the
    shape words (what `member_outputs` / `member_output_dtype` send)."""
    var call = MemberCall(DeviceRoute(0))
    if len(words) > 0:
        call.params = Int(words.unsafe_ptr())
        call.param_count = len(words)
    call.flags = operation
    call.x.cols = features
    call.y.cols = targets
    call.classes = classes
    _ = member_call(member, MEMBER_OP_OUTPUTS, call)
    _ = len(words)
    return call^


def _answers(member: MemberRef, words: List[Float64], operation: Int, features: Int, targets: Int, classes: Int,
    cols: Int, dtype: Int, what: String) raises:
    var call = _ask(member, words, operation, features, targets, classes)
    require(call.out_cols == cols and call.out_dtype == dtype, "member ABI: " + what + " answers "
        + String(cols) + " column(s) of type " + String(dtype) + " without a state, got " + String(call.out_cols)
        + " of type " + String(call.out_dtype))


def _refuses(member: MemberRef, words: List[Float64], operation: Int, what: String) raises:
    var refused = False
    try:
        _ = _ask(member, words, operation, 4, 1, 3)
    except error:
        refused = String(error).byte_length() > 0
    require(refused, "member ABI: " + what + " refuses to size an operation it lacks")


def _kind_mask(member: MemberRef, words: List[Float64]) raises -> Int:
    """CAPABILITIES without a state, kind-exact from the NEW words."""
    var call = MemberCall(DeviceRoute(0))
    call.params = Int(words.unsafe_ptr())
    call.param_count = len(words)
    var mask = member_call(member, MEMBER_OP_CAPABILITIES, call)
    _ = len(words)
    return mask


def _sweep_a_fixture(which: Int) raises:
    """One sweep-a family, without a kernel and without a state."""
    var member = _sweep_a_member(which)
    var none = List[Float64]()
    if which == 0:
        var cls = _words(20, 2.0)  # SGDClassifier, loss='log_loss' (word 19)
        cls[19] = 1.0
        var reg = _words(20, 0.0)  # SGDRegressor
        require((_kind_mask(member, cls) & (MEMBER_CAN_CLASSIFIER | MEMBER_CAN_PREDICT_PROBA))
            == (MEMBER_CAN_CLASSIFIER | MEMBER_CAN_PREDICT_PROBA)
            and (_kind_mask(member, reg) & MEMBER_CAN_CLASSIFIER) == 0, "member ABI: SGD kinds are kind-exact")
        _answers(member, cls, MEMBER_OP_PREDICT, 4, 1, 3, 1, MEMBER_I32, "SGDClassifier predict")
        _answers(member, cls, MEMBER_OP_PREDICT_PROBA, 4, 1, 3, 3, MEMBER_F32, "SGDClassifier predict_proba")
        _answers(member, reg, MEMBER_OP_PREDICT, 4, 1, 0, 1, MEMBER_F32, "SGDRegressor predict")
        _refuses(member, reg, MEMBER_OP_PREDICT_PROBA, "SGDRegressor")
        cls[19] = 0.0
        _refuses(member, cls, MEMBER_OP_PREDICT_PROBA, "SGDClassifier without log_loss")
    elif which == 1:
        var words = _words(7, 0.0)
        _answers(member, words, MEMBER_OP_PREDICT, 4, 1, 0, 1, MEMBER_F32, "ElasticNet predict")
        _refuses(member, words, MEMBER_OP_PREDICT_PROBA, "ElasticNet")
    elif which == 2:
        var svc = _words(11, 0.0)
        var svr = _words(11, 1.0)
        require((_kind_mask(member, svc) & MEMBER_CAN_CLASSIFIER) != 0
            and (_kind_mask(member, svr) & (MEMBER_CAN_CLASSIFIER | MEMBER_CAN_DECISION)) == 0,
            "member ABI: LinearSVC / LinearSVR are kind-exact")
        _answers(member, svc, MEMBER_OP_PREDICT, 4, 1, 3, 1, MEMBER_I32, "LinearSVC predict")
        _answers(member, svc, MEMBER_OP_DECISION, 4, 1, 3, 3, MEMBER_F32, "LinearSVC decision_function")
        _answers(member, svr, MEMBER_OP_PREDICT, 4, 1, 0, 1, MEMBER_F32, "LinearSVR predict")
        _refuses(member, svr, MEMBER_OP_DECISION, "LinearSVR")
    elif which == 3:
        require((_kind_mask(member, _words(6, 2.0)) & MEMBER_CAN_TEST_ROWS) != 0,
            "member ABI: KernelRidge tests its rows")
        _answers(member, none, MEMBER_OP_PREDICT, 4, 2, 0, 2, MEMBER_F32, "KernelRidge predict")
        _refuses(member, none, MEMBER_OP_PREDICT_PROBA, "KernelRidge")
    elif which == 4:
        var rbf = _words(9, 1.0)
        rbf[1] = 2.0
        rbf[6] = 7.0
        var nystroem = _words(9, 0.0)
        nystroem[6] = 7.0
        require((_kind_mask(member, rbf) & MEMBER_CAN_TEST_ROWS) != 0, "member ABI: RBFSampler tests its rows")
        _answers(member, rbf, MEMBER_OP_TRANSFORM, 4, 0, 0, 7, MEMBER_F32, "RBFSampler transform")
        _refuses(member, nystroem, MEMBER_OP_TRANSFORM, "Nystroem (its width is chosen by the fit)")
        _refuses(member, rbf, MEMBER_OP_PREDICT, "RBFSampler")
    elif which == 5:
        var gpc = _words(1, 1.0)
        var gpr = _words(1, 0.0)
        require((_kind_mask(member, gpc) & (MEMBER_CAN_CLASSIFIER | MEMBER_CAN_TEST_ROWS))
            == (MEMBER_CAN_CLASSIFIER | MEMBER_CAN_TEST_ROWS)
            and (_kind_mask(member, gpr) & MEMBER_CAN_CLASSIFIER) == 0, "member ABI: GP kinds are kind-exact")
        _answers(member, gpc, MEMBER_OP_PREDICT, 4, 1, 3, 1, MEMBER_I32, "GaussianProcessClassifier predict")
        _answers(member, gpc, MEMBER_OP_PREDICT_PROBA, 4, 1, 3, 3, MEMBER_F64, "GaussianProcessClassifier proba")
        _answers(member, gpr, MEMBER_OP_PREDICT_STD, 4, 1, 0, 1, MEMBER_F32, "GaussianProcessRegressor std")
        _refuses(member, gpr, MEMBER_OP_PREDICT_PROBA, "GaussianProcessRegressor")
    elif which == 6 or which == 7:
        var count = 1 if which == 6 else 20
        var cls = _words(count, 1.0)
        var reg = _words(count, 0.0)
        var name = String("MLP") if which == 6 else String("recurrent")
        require((_kind_mask(member, cls) & MEMBER_CAN_CLASSIFIER) != 0
            and (_kind_mask(member, reg) & MEMBER_CAN_CLASSIFIER) == 0, "member ABI: " + name + " kinds are kind-exact")
        _answers(member, cls, MEMBER_OP_PREDICT, 4, 1, 4, 1, MEMBER_I32, name + " classifier predict")
        _answers(member, cls, MEMBER_OP_PREDICT_PROBA, 4, 1, 4, 4, MEMBER_F32, name + " classifier predict_proba")
        _answers(member, reg, MEMBER_OP_PREDICT, 4, 2, 0, 2, MEMBER_F32, name + " regressor predict")
        _refuses(member, reg, MEMBER_OP_PREDICT_PROBA, name + " regressor")
    elif which == 8:
        _answers(member, none, MEMBER_OP_PREDICT, 12, 1, 5, 1, MEMBER_I32, "CNNClassifier predict")
        _answers(member, none, MEMBER_OP_PREDICT_PROBA, 12, 1, 5, 5, MEMBER_F32, "CNNClassifier predict_proba")
        _refuses(member, none, MEMBER_OP_DECISION, "CNNClassifier")
    elif which == 9:
        var words = _words(8, 0.0)
        words[1] = 2.0
        _answers(member, words, MEMBER_OP_PREDICT, 4, 3, 0, 3, MEMBER_F32, "PLSRegression predict")
        _answers(member, words, MEMBER_OP_TRANSFORM, 4, 3, 0, 2, MEMBER_F32, "PLSRegression transform")
        _answers(member, words, MEMBER_OP_INVERSE_TRANSFORM, 4, 3, 0, 4, MEMBER_F32, "PLSRegression inverse")
        _refuses(member, words, MEMBER_OP_PREDICT_PROBA, "PLSRegression")
    else:
        var name = String("ARIMA") if which == 10 else String("ETS / Theta")
        _answers(member, none, MEMBER_OP_PREDICT, 0, 2, 0, 2, MEMBER_F32, name + " forecast")
        _refuses(member, none, MEMBER_OP_TRANSFORM, name)
    _ = len(none)


def _borrow_new(member: MemberRef) raises -> Int:
    """A KNeighborsRegressor member state (kind 1, one neighbor, euclidean,
    uniform weights)."""
    var words: List[Float64] = [1.0, 1.0, 1.0, 2.0, 0.0, 0.0]
    var call = MemberCall(DeviceRoute(0))
    call.params = Int(words.unsafe_ptr())
    call.param_count = 6
    _ = member_call(member, MEMBER_OP_NEW, call)
    _ = len(words)
    return call.state


def _borrow_same(view: MemberMatrix, expected: List[Float32]) -> Bool:
    """The float32 host block `view` holds exactly the words `expected`."""
    if view.data == 0 or view.rows * view.cols != len(expected):
        return False
    var words = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=view.data)
    for i in range(len(expected)):  # driver-loop: over a six-word fixture block
        if words[i] != expected[i]:
            return False
    return True


def _borrow_block(member: MemberRef, state: Int, block: Int) raises -> MemberMatrix:
    var call = MemberCall(DeviceRoute(0), state)
    call.block = block
    _ = member_call(member, MEMBER_OP_EXPORT, call)
    return call.dst


def _borrow_fixture(which: Int) raises:
    """THE LIFETIME RULE on a borrowing kind, without a kernel. Fit a k-NN
    regressor on a buffer; make the state own its rows (`which` 0: the
    export of both blocks, the copy that always existed; 1: `member_own`,
    what an adoption sends); OVERWRITE the fit buffers; the state's blocks
    still hold the fitted words, in the state's own storage; IMPORT them
    into a second state, free the first, and the second state's blocks are
    the exported ones. A k-NN answer is a function of exactly these two
    blocks and the query, so the predict of the imported state is the
    exported model's; the predict call itself needs the family's kernel
    entry in the route, which this suite (no image) does not hold.
    SERIAL BY NATURE: one state, six words; the fixtures are the parallel units."""
    var member = MemberRef(EntryRef(neighbors_member_v2, 0), 0)
    # a kind that owns its state does not say it borrows, and OWN is not sent to it
    var linear_probe = MemberCall(DeviceRoute(0))
    require((member_call(_member(), MEMBER_OP_CAPABILITIES, linear_probe) & MEMBER_BORROWS_FIT) == 0,
        "member ABI: the linear member owns its state (no MEMBER_BORROWS_FIT)")
    var points: List[Float32] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    var targets: List[Float32] = [10.0, 20.0, 30.0]
    var fitted_points = points.copy()
    var fitted_targets = targets.copy()
    var first = _borrow_new(member)
    var second = 0
    try:
        var call = MemberCall(DeviceRoute(0), first)
        require((member_call(member, MEMBER_OP_CAPABILITIES, call) & MEMBER_BORROWS_FIT) != 0,
            "member ABI: a k-NN state says it borrows its fit buffers (kind-exact)")
        call.x = MemberMatrix(Int(points.unsafe_ptr()), 3, 2)
        call.y = MemberMatrix(Int(targets.unsafe_ptr()), 3, 1)
        _ = member_call(member, MEMBER_OP_FIT, call)
        if which == 0:
            _ = _borrow_block(member, first, MEMBER_BLOCK_FAMILY)
            _ = _borrow_block(member, first, MEMBER_BLOCK_FAMILY + 1)
        else:
            var plan = MemberPlan()
            plan.member = member
            plan.route = DeviceRoute(0)
            member_own(plan, first)
            member_own(plan, first)  # a second OWN copies nothing and changes nothing
        # the caller's buffers are gone (overwritten in place)
        for i in range(6):  # driver-loop: over the fixture's six words
            points[i] = Float32(-7.0)
        for i in range(3):  # driver-loop: over the fixture's three targets
            targets[i] = Float32(-7.0)
        var point_view = _borrow_block(member, first, MEMBER_BLOCK_FAMILY)
        var target_view = _borrow_block(member, first, MEMBER_BLOCK_FAMILY + 1)
        require(point_view.data != Int(points.unsafe_ptr()) and target_view.data != Int(targets.unsafe_ptr()),
            "member ABI: an owning k-NN state hands out its own words, not the caller's fit buffers")
        require(_borrow_same(point_view, fitted_points) and _borrow_same(target_view, fitted_targets),
            "member ABI: a k-NN state that owns its rows keeps the fitted words after the fit buffers changed")
        second = _borrow_new(member)
        var into = MemberCall(DeviceRoute(0), second)
        into.block = MEMBER_BLOCK_FAMILY
        into.x = point_view
        _ = member_call(member, MEMBER_OP_IMPORT, into)
        into.block = MEMBER_BLOCK_FAMILY + 1
        into.x = target_view
        into.flags = MEMBER_IMPORT_DONE
        _ = member_call(member, MEMBER_OP_IMPORT, into)
        # the first state goes; the imported one is whole on its own
        var gone = MemberCall(DeviceRoute(0), first)
        _ = member_call(member, MEMBER_OP_FREE, gone)
        first = 0
        require(_borrow_same(_borrow_block(member, second, MEMBER_BLOCK_FAMILY), fitted_points)
            and _borrow_same(_borrow_block(member, second, MEMBER_BLOCK_FAMILY + 1), fitted_targets),
            "member ABI: the imported k-NN state holds the exported model, not the overwritten buffers")
    finally:
        if first != 0:
            var last = MemberCall(DeviceRoute(0), first)
            _ = member_call(member, MEMBER_OP_FREE, last)
        if second != 0:
            var other = MemberCall(DeviceRoute(0), second)
            _ = member_call(member, MEMBER_OP_FREE, other)
        _ = len(points)
        _ = len(targets)
        _ = len(fitted_points)
        _ = len(fitted_targets)


def _fixture(unit: Int) raises:
    if unit >= _LINEAR_FIXTURES + _PUBLISHED:
        var extra = unit - _LINEAR_FIXTURES - _PUBLISHED
        if extra < _SWEEP_B:
            _sweep_b_fixture(extra)
        elif extra < _SWEEP_B + _SWEEP_A:
            _sweep_a_fixture(extra - _SWEEP_B)
        else:
            _borrow_fixture(extra - _SWEEP_B - _SWEEP_A)
        return
    if unit >= _LINEAR_FIXTURES:
        _published_fixture(unit - _LINEAR_FIXTURES)
        return
    if unit == 0:
        # capabilities and the output width
        var call = MemberCall(DeviceRoute(0))
        var mask = member_call(_member(), MEMBER_OP_CAPABILITIES, call)
        require(mask == (MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_SCORE | MEMBER_CAN_EXPORT
            | MEMBER_CAN_IMPORT | MEMBER_CAN_SAMPLE_WEIGHT | MEMBER_CAN_COEF | MEMBER_GIL_FREE), "member ABI: linear capability mask")
        require((mask & (MEMBER_CAN_PREDICT_PROBA | MEMBER_CAN_TRANSFORM | MEMBER_CAN_WEIGHTED_SCORE
            | MEMBER_CAN_DEVICE_VIEWS | MEMBER_CAN_CLASSIFIER)) == 0,
            "member ABI: a linear regressor has no predict_proba, transform, weighted score, device views or classes")
        call.classes = 7
        call.flags = MEMBER_OP_PREDICT
        _ = member_call(_member(), MEMBER_OP_OUTPUTS, call)
        require(call.out_cols == 1 and call.classes == 0 and call.out_dtype == MEMBER_F32,
            "member ABI: one float32 output column, no classes")
        call.flags = 0
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
        refused = String(error).byte_length() > 0
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


def test_member_abi() raises:
    check_cases("member_abi", MEMBER_ABI_FIXTURES, _fixture)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
