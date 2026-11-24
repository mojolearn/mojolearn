# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""Every typed meta-estimator RUN of xtrees/trees_meta_vote_run.mojo over NATIVE members (lane meta-gil,
2026-11-23), run on host workers of a process that has no Python at all.

The doors (bindings/trees_public_meta_door.mojo, bindings/trees_public_boost_door.mojo) release the GIL around
these RUNs whenever a member is native (a kind with MEMBER_GIL_FREE or MEMBER_GIL_DOOR). nn-integrate found a
segfault there (CalibratedClassifierCV.score over a LogisticRegression member; Bagging OOB). The defect was a RUN
argument read through Python inside the release, not the member, but no test had ever run a RUN over a native
member: checks/native_tests/trees_meta_runs.mojo feeds door blocks only. Here every case:

- fits LogisticRegression members (core/logistic_member.mojo, kind 0, MEMBER_GIL_FREE) through the member ABI
  on the CPU image's QN entry (glm/host/logistic_native_host.mojo `qn_host_native_v1`), on the case's worker;
- runs the RUN with those members NATIVE (RUN makes the member calls) on that worker, which holds no GIL and has
  no Python thread state: any CPython call on the path crashes this binary;
- runs the same RUN again with DOOR members whose blocks are the members' own direct answers (float64 rounded to
  float32 and int64 codes narrowed to int32, as the door's admission does), and requires every output word to
  be equal (the native route and the door route are the same ops over the same words).

Cases (one per RUN entry the doors release around): calibrated binary over decision_function (the self-test's
shape), calibrated k = 3 over predict_proba, Bagging OOB (classifier), Bagging predict, soft and hard vote,
MultiOutput proba and codes, Stacking placement, OneVsRest proba. The cases run concurrently as one case table
across the host cores, so the logistic member store is also exercised from several threads at once (the
MEMBER_GIL_FREE claim).

Sabotage (to prove it can fail): put a Python call into `_member_f32` (xtrees/trees_meta_vote_run.mojo), or
narrow the native F64 answer differently from the door's rounding: every case fails.
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.device_route import DeviceRoute, DEVICE_CPU
from core.entry_table import EntryRef
from core.logistic_member import logistic_member_v2
from core.logistic_native_api import QnNativeDriver
from core.logistic_route import LogisticRoute, logistic_route_pack
from core.member_abi import (
    MemberCall, MemberMatrix, MemberPlan, MemberRef, member_call, member_call_of, member_new, member_free,
    member_kind_capabilities, MEMBER_OP_CAPABILITIES, MEMBER_OP_FIT, MEMBER_OP_OUTPUTS, MEMBER_OP_PREDICT,
    MEMBER_OP_PREDICT_PROBA, MEMBER_OP_DECISION, MEMBER_F32, MEMBER_F64, MEMBER_I32, MEMBER_I64, MEMBER_GIL_FREE,
)
from core.trees_entry_table import TreesEntryKey
from glm.host.logistic_native_host import qn_host_native_v1
from xtrees.glue_runs import publish_x_trees_glue
from xtrees.glue_calls import xt_ref
from xtrees.trees_meta_vote_run import (
    VoteMember, VoteRows, VOTE_ALL_ANSWERED, vote_soft_run, vote_codes_run, ovr_assemble_run,
    multioutput_codes_run, multioutput_proba_run, bag_predict_run, bag_oob_run, stack_place_run, STACK_PROBA,
    CalPair, CalFit, calibrated_proba_run,
)

comptime N = 48
comptime D = 3
comptime CASES = 9

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _I64P = MutPointer[Int64, MutUntrackedOrigin]


def _glue() raises -> EntryRef:
    """The glue entry `test_meta_runs_gil_free` published before the cases."""
    return xt_ref(TreesEntryKey(Int(GLOBAL_NUMERIC_MODE), 0))


def _x() -> List[Float32]:
    """One N x D block whose rows the classes below separate only in part (a fit with real iterations)."""
    var x = List[Float32](length=N * D, fill=0.0)
    for i in range(N):  # small-loop(fixture)
        for j in range(D):  # small-loop(fixture)
            x[i * D + j] = Float32((i * 7 + j * 5) % 11) * 0.25 - 1.0 + Float32(i % 3) * 0.5 * Float32(j + 1)
    return x^


def _codes(k: Int, positive: Int) -> List[Int32]:
    """Class codes 0 .. k-1 (`positive` < 0), else the one-vs-rest indicator of class `positive` of 3."""
    var y = List[Int32](length=N, fill=Int32(0))
    for i in range(N):  # small-loop(fixture)
        var c = (i + i // 5) % 3
        if positive >= 0:
            y[i] = Int32(1) if c == positive else Int32(0)
        else:
            y[i] = Int32(c % k)
    return y^


struct _Fitted(Movable):
    """One fitted LogisticRegression member: its plan (RUN reads it by address: keep this value unmoved while a
    VoteMember points at it) and its state."""
    var plan: MemberPlan
    var state: Int

    def __init__(out self, var plan: MemberPlan, state: Int):
        self.plan = plan^
        self.state = state


def _fit(x: List[Float32], k: Int, positive: Int) raises -> _Fitted:
    """A LogisticRegression member (l2 1.0, intercept, lbfgs) fitted on `x` and `_codes(k, positive)` through the
    member ABI, on the calling thread."""
    var plan = MemberPlan()
    plan.route = logistic_route_pack(LogisticRoute(DEVICE_CPU, Int(GLOBAL_NUMERIC_MODE), 0, False, False,
                                                   QnNativeDriver(qn_host_native_v1)))
    plan.member = MemberRef(EntryRef(logistic_member_v2, 0), 0)
    var probe = MemberCall(plan.route)
    plan.member.capabilities = member_call(plan.member, MEMBER_OP_CAPABILITIES, probe)
    # LogisticRegression()'s own words as providers/python/member_logistic_boundary_door.mojo admits them:
    # max_iter 1000, linesearch 50, lbfgs_memory 5, penalty_normalized 1 (was 100, 20, 10, 0: the old words,
    # an unnormalized penalty N times the estimator's; lane mf-iterimp-logit, w3-cleanup).
    plan.params = [0.0, 1.0, 0.0, 1.0, 1000.0, 50.0, 5.0, 1.0, 1e-4, 1e-6]
    plan.classes = k
    require((member_kind_capabilities(plan) & MEMBER_GIL_FREE) != 0,
            "the logistic kind lost MEMBER_GIL_FREE: the meta doors would answer it in UNPACK, not in RUN")
    var call = member_new(plan)
    var y = _codes(k, positive)
    call.x = MemberMatrix(Int(x.unsafe_ptr()), N, D)
    call.y = MemberMatrix(Int(y.unsafe_ptr()), N, 1, MEMBER_I32)
    call.classes = k
    try:
        _ = member_call(plan.member, MEMBER_OP_FIT, call)
    except e:
        member_free(plan, call)
        raise e^
    _ = y^  # read through the fit above
    var state = call.state
    return _Fitted(plan^, state)


def _free(mut f: _Fitted):
    var call = MemberCall(f.plan.route, f.state)
    member_free(f.plan, call)
    f.state = 0


def _outputs(mut f: _Fitted, operation: Int) raises -> MemberCall:
    """The member's OUTPUTS answer for `operation` (`out_cols`, `out_dtype`)."""
    var call = member_call_of(f.plan, f.state)
    call.flags = operation
    _ = member_call(f.plan.member, MEMBER_OP_OUTPUTS, call)
    return call^


def _native(mut f: _Fitted, operation: Int, class_cols: Int, weight: Float64) raises -> VoteMember:
    """The member as a native VoteMember: RUN calls it (its plan by address)."""
    var asked = _outputs(f, operation)
    return VoteMember(Int(UnsafePointer(to=f.plan)), f.state, operation, asked.out_cols, asked.out_dtype,
                      class_cols, weight)


def _block(mut f: _Fitted, operation: Int, x: Int, n: Int) raises -> List[Float32]:
    """The member's own float32 answer over `n` rows at `x` (a float64 answer rounded word by word, as the
    door's admission rounds it): the door block."""
    var asked = _outputs(f, operation)
    var cols = asked.out_cols
    var words = List[Float32](length=n * cols, fill=0.0)
    var call = member_call_of(f.plan, f.state)
    call.x = MemberMatrix(x, n, D)
    if asked.out_dtype == MEMBER_F64:
        var wide = List[Float64](length=n * cols, fill=0.0)
        call.dst = MemberMatrix(Int(wide.unsafe_ptr()), n, cols, MEMBER_F64)
        _ = member_call(f.plan.member, operation, call)
        for i in range(n * cols):  # small-loop(oracle: the door's float32 admission)
            words[i] = wide[i].cast[DType.float32]()
        _ = wide^
    else:
        call.dst = MemberMatrix(Int(words.unsafe_ptr()), n, cols, MEMBER_F32)
        _ = member_call(f.plan.member, operation, call)
    return words^


def _codes_block(mut f: _Fitted, x: Int, n: Int) raises -> List[Int32]:
    """The member's own predicted codes (int64 narrowed to int32, as the door's cast): the door block."""
    var wide = List[Int64](length=n, fill=0)
    var call = member_call_of(f.plan, f.state)
    call.x = MemberMatrix(x, n, D)
    call.dst = MemberMatrix(Int(wide.unsafe_ptr()), n, 1, MEMBER_I64)
    _ = member_call(f.plan.member, MEMBER_OP_PREDICT, call)
    var codes = List[Int32](length=n, fill=Int32(0))
    for i in range(n):  # small-loop(oracle: the door's int32 cast)
        codes[i] = Int32(wide[i])
    _ = wide^
    return codes^


def _same(a: List[Float64], b: List[Float64], what: String) raises:
    require(len(a) == len(b), what + ": lengths differ")
    for i in range(len(a)):  # small-loop(oracle: every word)
        require(a[i] == b[i], what + " word " + String(i) + ": native " + String(a[i]) + " != door "
                + String(b[i]))


def _same_i32(a: List[Int32], b: List[Int32], what: String) raises:
    require(len(a) == len(b), what + ": lengths differ")
    for i in range(len(a)):  # small-loop(oracle: every code)
        require(a[i] == b[i], what + " code " + String(i) + ": native " + String(a[i]) + " != door " + String(b[i]))


def _door(block: List[Float32], cols: Int, class_cols: Int, weight: Float64) -> VoteMember:
    return VoteMember(Int(block.unsafe_ptr()), cols, class_cols, weight)


def check_calibrated_binary() raises:
    """CalibratedClassifierCV.predict_proba, k = 2, two pairs scored by decision_function (one column each),
    Platt calibrators: the shape verify --self-test crashed on."""
    var x = _x()
    var a = _fit(x, 2, -1)
    var b = _fit(x, 2, -1)
    var rows = VoteRows(Int(x.unsafe_ptr()), N, D)
    var pairs: List[CalPair] = [CalPair(0, 0, 0, 0, 1), CalPair(0, 0, 0, 1, 1)]
    var cals: List[CalFit] = [CalFit(False, -1.25, 0.125, 0, 0, 0), CalFit(False, -0.75, -0.25, 0, 0, 0)]
    var native: List[VoteMember] = [_native(a, MEMBER_OP_DECISION, 0, 1.0), _native(b, MEMBER_OP_DECISION, 0, 1.0)]
    var got = List[Float64](length=N * 2, fill=0.0)
    require(calibrated_proba_run(_glue(), rows, native, pairs, cals, 2, Int(got.unsafe_ptr())) == VOTE_ALL_ANSWERED,
            "a native member refused")
    var sa = _block(a, MEMBER_OP_DECISION, Int(x.unsafe_ptr()), N)
    var sb = _block(b, MEMBER_OP_DECISION, Int(x.unsafe_ptr()), N)
    var door: List[VoteMember] = [_door(sa, 1, 0, 1.0), _door(sb, 1, 0, 1.0)]
    var want = List[Float64](length=N * 2, fill=0.0)
    _ = calibrated_proba_run(_glue(), rows, door, pairs, cals, 2, Int(want.unsafe_ptr()))
    _same(got, want, "calibrated binary")
    _ = sa^
    _ = sb^
    _free(a)
    _free(b)
    _ = x^


def check_calibrated_multiclass() raises:
    """CalibratedClassifierCV.predict_proba, k = 3, one pair scored by predict_proba (float64 answer, rounded
    to float32 by RUN as by the door), three Platt calibrators."""
    var x = _x()
    var a = _fit(x, 3, -1)
    var cols: List[Int32] = [0, 1, 2]
    var rows = VoteRows(Int(x.unsafe_ptr()), N, D)
    var pairs: List[CalPair] = [CalPair(0, Int(cols.unsafe_ptr()), 3, 0, 3)]
    var cals: List[CalFit] = [CalFit(False, -1.5, 0.25, 0, 0, 0), CalFit(False, -0.5, -0.125, 0, 0, 0),
                              CalFit(False, -2.0, 0.5, 0, 0, 0)]
    var native: List[VoteMember] = [_native(a, MEMBER_OP_PREDICT_PROBA, Int(cols.unsafe_ptr()), 1.0)]
    var got = List[Float64](length=N * 3, fill=0.0)
    require(calibrated_proba_run(_glue(), rows, native, pairs, cals, 3, Int(got.unsafe_ptr())) == VOTE_ALL_ANSWERED,
            "a native member refused")
    var p = _block(a, MEMBER_OP_PREDICT_PROBA, Int(x.unsafe_ptr()), N)
    var door: List[VoteMember] = [_door(p, 3, Int(cols.unsafe_ptr()), 1.0)]
    var want = List[Float64](length=N * 3, fill=0.0)
    _ = calibrated_proba_run(_glue(), rows, door, pairs, cals, 3, Int(want.unsafe_ptr()))
    _same(got, want, "calibrated k=3")
    _ = p^
    _ = cols^
    _free(a)
    _ = x^


def check_bag_oob() raises:
    """BaggingClassifier OOB sums: two members, each over its own out-of-bag rows (consecutive rows of X, row
    numbers given), probabilities placed at the class columns and counted."""
    var x = _x()
    var a = _fit(x, 3, -1)
    var b = _fit(x, 3, -1)
    var cols: List[Int32] = [0, 1, 2]
    var first: List[Int] = [5, 20]
    var ns: List[Int] = [17, 23]
    var oob_a = List[Int32](length=ns[0], fill=Int32(0))
    var oob_b = List[Int32](length=ns[1], fill=Int32(0))
    for i in range(ns[0]):  # small-loop(fixture)
        oob_a[i] = Int32(first[0] + i)
    for i in range(ns[1]):  # small-loop(fixture)
        oob_b[i] = Int32(first[1] + i)
    var base = Int(x.unsafe_ptr())
    var xs: List[Int] = [base + first[0] * D * 4, base + first[1] * D * 4]
    var ds: List[Int] = [D, D]
    var oobs: List[Int] = [Int(oob_a.unsafe_ptr()), Int(oob_b.unsafe_ptr())]
    var native: List[VoteMember] = [_native(a, MEMBER_OP_PREDICT_PROBA, Int(cols.unsafe_ptr()), 1.0),
                                    _native(b, MEMBER_OP_PREDICT_PROBA, Int(cols.unsafe_ptr()), 1.0)]
    var got = List[Float64](length=N * 3, fill=0.0)
    var got_n = List[Int32](length=N, fill=Int32(0))
    require(bag_oob_run(_glue(), native, xs, ns, ds, oobs, N, 3, True, Int(got.unsafe_ptr()),
                        Int(got_n.unsafe_ptr()), 0) == VOTE_ALL_ANSWERED, "a native member refused")
    var pa = _block(a, MEMBER_OP_PREDICT_PROBA, xs[0], ns[0])
    var pb = _block(b, MEMBER_OP_PREDICT_PROBA, xs[1], ns[1])
    var door: List[VoteMember] = [_door(pa, 3, Int(cols.unsafe_ptr()), 1.0), _door(pb, 3, Int(cols.unsafe_ptr()), 1.0)]
    var want = List[Float64](length=N * 3, fill=0.0)
    var want_n = List[Int32](length=N, fill=Int32(0))
    _ = bag_oob_run(_glue(), door, xs, ns, ds, oobs, N, 3, True, Int(want.unsafe_ptr()), Int(want_n.unsafe_ptr()), 0)
    _same(got, want, "bagging oob sums")
    _same_i32(got_n, want_n, "bagging oob counts")
    _ = pa^
    _ = pb^
    _ = oob_a^
    _ = oob_b^
    _ = cols^
    _free(a)
    _free(b)
    _ = x^


def check_bag_predict() raises:
    """BaggingClassifier.predict_proba: two members over the whole of X, placed and divided by the count."""
    var x = _x()
    var a = _fit(x, 3, -1)
    var b = _fit(x, 3, -1)
    var cols: List[Int32] = [0, 1, 2]
    var xs: List[Int] = [Int(x.unsafe_ptr()), Int(x.unsafe_ptr())]
    var ds: List[Int] = [D, D]
    var native: List[VoteMember] = [_native(a, MEMBER_OP_PREDICT_PROBA, Int(cols.unsafe_ptr()), 1.0),
                                    _native(b, MEMBER_OP_PREDICT_PROBA, Int(cols.unsafe_ptr()), 1.0)]
    var got = List[Float64](length=N * 3, fill=0.0)
    require(bag_predict_run(_glue(), native, xs, ds, N, 3, True, Int(got.unsafe_ptr()), 0) == VOTE_ALL_ANSWERED,
            "a native member refused")
    var pa = _block(a, MEMBER_OP_PREDICT_PROBA, xs[0], N)
    var pb = _block(b, MEMBER_OP_PREDICT_PROBA, xs[1], N)
    var door: List[VoteMember] = [_door(pa, 3, Int(cols.unsafe_ptr()), 1.0), _door(pb, 3, Int(cols.unsafe_ptr()), 1.0)]
    var want = List[Float64](length=N * 3, fill=0.0)
    _ = bag_predict_run(_glue(), door, xs, ds, N, 3, True, Int(want.unsafe_ptr()), 0)
    _same(got, want, "bagging predict_proba")
    _ = pa^
    _ = pb^
    _ = cols^
    _free(a)
    _free(b)
    _ = x^


def check_vote() raises:
    """VotingClassifier: the soft vote (weights 1 and 2) and the hard vote's codes."""
    var x = _x()
    var a = _fit(x, 3, -1)
    var b = _fit(x, 3, -1)
    var cols: List[Int32] = [0, 1, 2]
    var rows = VoteRows(Int(x.unsafe_ptr()), N, D)
    var soft: List[VoteMember] = [_native(a, MEMBER_OP_PREDICT_PROBA, Int(cols.unsafe_ptr()), 1.0),
                                  _native(b, MEMBER_OP_PREDICT_PROBA, Int(cols.unsafe_ptr()), 2.0)]
    var got = List[Float64](length=N * 3, fill=0.0)
    require(vote_soft_run(_glue(), rows, soft, 3, Int(got.unsafe_ptr())) == VOTE_ALL_ANSWERED, "soft refused")
    var pa = _block(a, MEMBER_OP_PREDICT_PROBA, Int(x.unsafe_ptr()), N)
    var pb = _block(b, MEMBER_OP_PREDICT_PROBA, Int(x.unsafe_ptr()), N)
    var soft_door: List[VoteMember] = [_door(pa, 3, Int(cols.unsafe_ptr()), 1.0),
                                       _door(pb, 3, Int(cols.unsafe_ptr()), 2.0)]
    var want = List[Float64](length=N * 3, fill=0.0)
    _ = vote_soft_run(_glue(), rows, soft_door, 3, Int(want.unsafe_ptr()))
    _same(got, want, "soft vote")
    var hard: List[VoteMember] = [_native(a, MEMBER_OP_PREDICT, 0, 1.0), _native(b, MEMBER_OP_PREDICT, 0, 2.0)]
    var acc = List[Float64](length=N * 3, fill=0.0)
    var codes = List[Int32](length=N, fill=Int32(-1))
    require(vote_codes_run(_glue(), rows, hard, 3, False, Int(acc.unsafe_ptr()), Int(codes.unsafe_ptr()))
            == VOTE_ALL_ANSWERED, "hard refused")
    var ca = _codes_block(a, Int(x.unsafe_ptr()), N)
    var cb = _codes_block(b, Int(x.unsafe_ptr()), N)
    var hard_door: List[VoteMember] = [VoteMember(Int(ca.unsafe_ptr()), 1, 0, 1.0),
                                       VoteMember(Int(cb.unsafe_ptr()), 1, 0, 2.0)]
    var acc2 = List[Float64](length=N * 3, fill=0.0)
    var codes2 = List[Int32](length=N, fill=Int32(-1))
    _ = vote_codes_run(_glue(), rows, hard_door, 3, False, Int(acc2.unsafe_ptr()), Int(codes2.unsafe_ptr()))
    _same_i32(codes, codes2, "hard vote")
    _ = acc^  # the hard votes' scratch, read through vote_codes_run
    _ = acc2^
    _ = pa^
    _ = pb^
    _ = ca^
    _ = cb^
    _ = cols^
    _free(a)
    _free(b)
    _ = x^


def check_multioutput_proba() raises:
    """MultiOutputClassifier.predict_proba: output 0 a k = 3 member, output 1 a k = 2 member."""
    var x = _x()
    var a = _fit(x, 3, -1)
    var b = _fit(x, 2, -1)
    var c3: List[Int32] = [0, 1, 2]
    var c2: List[Int32] = [0, 1]
    var rows = VoteRows(Int(x.unsafe_ptr()), N, D)
    var ks: List[Int] = [3, 2]
    var native: List[VoteMember] = [_native(a, MEMBER_OP_PREDICT_PROBA, Int(c3.unsafe_ptr()), 1.0),
                                    _native(b, MEMBER_OP_PREDICT_PROBA, Int(c2.unsafe_ptr()), 1.0)]
    var g0 = List[Float64](length=N * 3, fill=0.0)
    var g1 = List[Float64](length=N * 2, fill=0.0)
    var gs: List[Int] = [Int(g0.unsafe_ptr()), Int(g1.unsafe_ptr())]
    require(multioutput_proba_run(_glue(), rows, native, ks, gs, 0) == VOTE_ALL_ANSWERED, "a native member refused")
    var pa = _block(a, MEMBER_OP_PREDICT_PROBA, Int(x.unsafe_ptr()), N)
    var pb = _block(b, MEMBER_OP_PREDICT_PROBA, Int(x.unsafe_ptr()), N)
    var door: List[VoteMember] = [_door(pa, 3, Int(c3.unsafe_ptr()), 1.0), _door(pb, 2, Int(c2.unsafe_ptr()), 1.0)]
    var w0 = List[Float64](length=N * 3, fill=0.0)
    var w1 = List[Float64](length=N * 2, fill=0.0)
    var ws: List[Int] = [Int(w0.unsafe_ptr()), Int(w1.unsafe_ptr())]
    _ = multioutput_proba_run(_glue(), rows, door, ks, ws, 0)
    _same(g0, w0, "multioutput proba output 0")
    _same(g1, w1, "multioutput proba output 1")
    _ = pa^
    _ = pb^
    _ = c3^
    _ = c2^
    _free(a)
    _free(b)
    _ = x^


def check_multioutput_codes() raises:
    """MultiOutputClassifier.predict before the decode: each native member's codes into its output column."""
    var x = _x()
    var a = _fit(x, 3, -1)
    var b = _fit(x, 2, -1)
    var rows = VoteRows(Int(x.unsafe_ptr()), N, D)
    var native: List[VoteMember] = [_native(a, MEMBER_OP_PREDICT, 0, 1.0), _native(b, MEMBER_OP_PREDICT, 0, 1.0)]
    var d0 = List[Int32](length=N, fill=Int32(-1))
    var d1 = List[Int32](length=N, fill=Int32(-1))
    var dst: List[Int] = [Int(d0.unsafe_ptr()), Int(d1.unsafe_ptr())]
    require(multioutput_codes_run(rows, native, dst, 0) == VOTE_ALL_ANSWERED, "a native member refused")
    _same_i32(d0, _codes_block(a, Int(x.unsafe_ptr()), N), "multioutput codes output 0")
    _same_i32(d1, _codes_block(b, Int(x.unsafe_ptr()), N), "multioutput codes output 1")
    _free(a)
    _free(b)
    _ = x^


def check_stack_place() raises:
    """StackingClassifier meta features: two k = 3 predict_proba members placed side by side (six columns)."""
    var x = _x()
    var a = _fit(x, 3, -1)
    var b = _fit(x, 3, -1)
    var cols: List[Int32] = [0, 1, 2]
    var rows = VoteRows(Int(x.unsafe_ptr()), N, D)
    var kinds: List[Int] = [STACK_PROBA, STACK_PROBA]
    var counts: List[Int] = [3, 3]
    var place = List[Int32](length=N, fill=Int32(0))
    for i in range(N):  # small-loop(fixture: the identity placement)
        place[i] = Int32(i)
    var native: List[VoteMember] = [_native(a, MEMBER_OP_PREDICT_PROBA, Int(cols.unsafe_ptr()), 1.0),
                                    _native(b, MEMBER_OP_PREDICT_PROBA, Int(cols.unsafe_ptr()), 1.0)]
    var got = List[Float64](length=N * 6, fill=0.0)
    var got_w = List[Int]()
    require(stack_place_run(_glue(), rows, native, kinds, counts, 3, False, Int(got.unsafe_ptr()), N, 6,
                            Int(place.unsafe_ptr()), 0, got_w, 0) == VOTE_ALL_ANSWERED, "a native member refused")
    var pa = _block(a, MEMBER_OP_PREDICT_PROBA, Int(x.unsafe_ptr()), N)
    var pb = _block(b, MEMBER_OP_PREDICT_PROBA, Int(x.unsafe_ptr()), N)
    var door: List[VoteMember] = [_door(pa, 3, Int(cols.unsafe_ptr()), 1.0), _door(pb, 3, Int(cols.unsafe_ptr()), 1.0)]
    var want = List[Float64](length=N * 6, fill=0.0)
    var want_w = List[Int]()
    _ = stack_place_run(_glue(), rows, door, kinds, counts, 3, False, Int(want.unsafe_ptr()), N, 6,
                        Int(place.unsafe_ptr()), 0, want_w, 0)
    _same(got, want, "stacking meta features")
    require(len(got_w) == 2 and got_w[0] == 3 and got_w[1] == 3, "stacking widths")
    _ = pa^
    _ = pb^
    _ = place^
    _ = cols^
    _free(a)
    _free(b)
    _ = x^


def check_ovr_proba() raises:
    """OneVsRestClassifier.predict_proba, k = 3: three binary members, each scored by its decision column."""
    var x = _x()
    var m0 = _fit(x, 2, 0)
    var m1 = _fit(x, 2, 1)
    var m2 = _fit(x, 2, 2)
    var rows = VoteRows(Int(x.unsafe_ptr()), N, D)
    var at: List[Int] = [0, 0, 0]
    var native: List[VoteMember] = [_native(m0, MEMBER_OP_DECISION, 0, 1.0), _native(m1, MEMBER_OP_DECISION, 0, 1.0),
                                    _native(m2, MEMBER_OP_DECISION, 0, 1.0)]
    var got = List[Float64](length=N * 3, fill=0.0)
    require(ovr_assemble_run(_glue(), rows, native, at, 3, 1, Int(got.unsafe_ptr())) == VOTE_ALL_ANSWERED,
            "a native member refused")
    var s0 = _block(m0, MEMBER_OP_DECISION, Int(x.unsafe_ptr()), N)
    var s1 = _block(m1, MEMBER_OP_DECISION, Int(x.unsafe_ptr()), N)
    var s2 = _block(m2, MEMBER_OP_DECISION, Int(x.unsafe_ptr()), N)
    var door: List[VoteMember] = [_door(s0, 1, 0, 1.0), _door(s1, 1, 0, 1.0), _door(s2, 1, 0, 1.0)]
    var want = List[Float64](length=N * 3, fill=0.0)
    _ = ovr_assemble_run(_glue(), rows, door, at, 3, 1, Int(want.unsafe_ptr()))
    _same(got, want, "one-vs-rest proba")
    _ = s0^
    _ = s1^
    _ = s2^
    _free(m0)
    _free(m1)
    _free(m2)
    _ = x^


def test_meta_runs_gil_free() raises:
    # The publication _mojolearn_x_trees_host makes at module init, once before the parallel cases.
    publish_x_trees_glue("cpu")
    def run_case(which: Int) raises:
        if which == 0:
            check_calibrated_binary()
        elif which == 1:
            check_calibrated_multiclass()
        elif which == 2:
            check_bag_oob()
        elif which == 3:
            check_bag_predict()
        elif which == 4:
            check_vote()
        elif which == 5:
            check_multioutput_proba()
        elif which == 6:
            check_multioutput_codes()
        elif which == 7:
            check_stack_place()
        else:
            check_ovr_proba()
    check_cases("meta_runs_gil_free", CASES, run_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
