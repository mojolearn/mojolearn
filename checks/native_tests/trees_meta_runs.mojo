# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""The typed meta-estimator RUNs of lane bm-meta-3 (2026-11-23), reached natively: MultiOutputClassifier
(`multioutput_proba_run`, `multioutput_codes_run`), Bagging (`bag_predict_run`, `bag_oob_run`,
`bag_oob_classifier_finish`, `bag_oob_r2_run`), Stacking (`stack_place_run`, including the fold that lacks a class),
CalibratedClassifierCV (`calibrated_proba_run`), AdaBoost (`ada_classifier_run`, `ada_regressor_run`), all in
xtrees/trees_meta_vote_run.mojo, and the micro one-vs-rest roc_auc (`roc_ovr_micro_run`,
bindings/mcls_ranking_run.mojo, over the one typed program runner `metric_inputs_run`).

The member blocks are small fixed blocks written here (a member's door answer is a float32 block or int32 codes;
RUN reads it by address), each checked against the fold its scikit-learn / Python door form writes, computed here
in the same order. The roc case replaces the numeric part of
python/mojolearn/tests/test_meta_members_native.py test_roc_auc_ovr_micro_is_the_raveled_binary_auc (the Python test
stays): its y, s and w (`fixtures/meta_runs/`, tools/py2mojo_native_fixtures_bm_meta_3.py) and scikit-learn's raveled
binary AUC, unweighted and weighted, as the independent oracle (rel 1e-6, the Python test's tolerance). The
independent cases run as one case table across the host cores.
"""
from std.math import exp
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, fixture_f32, fixture_i32, require
from checks.native_tests.py2mojo_fixture_words import fixture_f64
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.entry_table import EntryRef
from core.member_abi import MemberPlan, MEMBER_OP_PREDICT, MEMBER_I32, MEMBER_F32
from core.trees_entry_table import TreesEntryKey
from xtrees.glue_runs import publish_x_trees_glue
from xtrees.glue_calls import xt_ref
from xtrees.trees_meta_vote_run import (
    VoteMember, VoteRows, VOTE_ALL_ANSWERED, multioutput_codes_run, multioutput_proba_run, bag_predict_run,
    bag_oob_run, bag_oob_classifier_finish, bag_oob_r2_run, stack_place_run, STACK_PREDICT, STACK_PROBA,
    STACK_DECISION, CalPair, CalFit, calibrated_proba_run, ada_classifier_run, ada_regressor_run, ADA_DECISION,
    ADA_CODES,
)
from bindings.arena_prog_run import MetricRoute
from bindings.mcls_ranking_run import roc_ovr_micro_run
from x_metrics.descriptor_host import metrics_program_host_v1

comptime CASES = 9
comptime ROC_N = 80
comptime ROC_K = 3
comptime ROC_DIR = "checks/native_tests/fixtures/meta_runs/"


def _glue() raises -> EntryRef:
    """The glue entry `test_meta_runs` published before the cases."""
    return xt_ref(TreesEntryKey(Int(GLOBAL_NUMERIC_MODE), 0))


def _near(got: Float64, want: Float64, tol: Float64, what: String) raises:
    require(abs(got - want) <= tol, what + ": " + String(got) + " != " + String(want))


def _door(block: List[Float32], cols: Int, class_cols: List[Int32]) -> VoteMember:
    """A member whose door answered: its float32 block and its class columns."""
    return VoteMember(Int(block.unsafe_ptr()), cols, Int(class_cols.unsafe_ptr()), 1.0)


def check_multioutput_proba() raises:
    """MultiOutputClassifier.predict_proba: output 1's member saw classes 0 and 2 of 3: its columns land at 0 and
    2, column 1 stays 0."""
    var p0: List[Float32] = [0.5, 0.25, 0.25, 0.125, 0.5, 0.375]
    var p1: List[Float32] = [0.75, 0.25, 0.5, 0.5]
    var c0: List[Int32] = [0, 1, 2]
    var c1: List[Int32] = [0, 2]
    var members: List[VoteMember] = [_door(p0, 3, c0), _door(p1, 2, c1)]
    var a0 = List[Float64](length=6, fill=0.0)
    var a1 = List[Float64](length=6, fill=0.0)
    var ks: List[Int] = [3, 3]
    var accs: List[Int] = [Int(a0.unsafe_ptr()), Int(a1.unsafe_ptr())]
    var refused = multioutput_proba_run(_glue(), VoteRows(0, 2, 1), members, ks, accs, 0)
    require(refused == VOTE_ALL_ANSWERED, "proba refused member " + String(refused))
    for i in range(6):  # small-loop(oracle: output 0 is its member's block)
        _near(a0[i], Float64(p0[i]), 0.0, "output 0 cell " + String(i))
    var want1: List[Float64] = [0.75, 0.0, 0.25, 0.5, 0.0, 0.5]
    for i in range(6):  # small-loop(oracle: output 1 placed at its classes)
        _near(a1[i], want1[i], 0.0, "output 1 cell " + String(i))
    _ = p0^
    _ = p1^
    _ = c0^
    _ = c1^


def check_multioutput_codes_refusal() raises:
    """MultiOutputClassifier predict: a door member's codes are its column (nothing written); a native member
    that refuses is returned by index; a run from past it answers every member."""
    var codes: List[Int32] = [2, 0, 1]
    var plan = MemberPlan()
    var members: List[VoteMember] = [VoteMember(Int(codes.unsafe_ptr()), 1, 0, 1.0),
                                     VoteMember(Int(UnsafePointer(to=plan)), 0, MEMBER_OP_PREDICT, 1, MEMBER_I32, 0,
                                                1.0)]
    var dst1 = List[Int32](length=3, fill=Int32(-7))
    var dst: List[Int] = [Int(codes.unsafe_ptr()), Int(dst1.unsafe_ptr())]
    var refused = multioutput_codes_run(VoteRows(0, 3, 1), members, dst, 0)
    require(refused == 1, "refused index " + String(refused))
    require(codes[0] == 2 and codes[1] == 0 and codes[2] == 1, "a door member's codes were written")
    require(multioutput_codes_run(VoteRows(0, 3, 1), members, dst, 2) == VOTE_ALL_ANSWERED, "start past the end")
    _ = dst1^
    _ = codes^
    _ = plan^


def check_bag_predict() raises:
    """Bagging predict_proba (the mean of the members' placed probabilities) and BaggingRegressor.predict (the
    mean of the members' predictions)."""
    var p0: List[Float32] = [0.5, 0.5, 0.0, 0.25, 0.25, 0.5]
    var p1: List[Float32] = [1.0, 0.0, 0.5, 0.5]
    var c0: List[Int32] = [0, 1, 2]
    var c1: List[Int32] = [1, 2]
    var members: List[VoteMember] = [_door(p0, 3, c0), _door(p1, 2, c1)]
    var xs: List[Int] = [0, 0]
    var ds: List[Int] = [1, 1]
    var acc = List[Float64](length=6, fill=0.0)
    var refused = bag_predict_run(_glue(), members, xs, ds, 2, 3, True, Int(acc.unsafe_ptr()), 0)
    require(refused == VOTE_ALL_ANSWERED, "bag proba refused " + String(refused))
    var want: List[Float64] = [0.25, 0.75, 0.0, 0.125, 0.375, 0.5]
    for i in range(6):  # small-loop(oracle)
        _near(acc[i], want[i], 0.0, "bag proba cell " + String(i))
    var r0: List[Float32] = [1.0, 2.0, 3.0]
    var r1: List[Float32] = [2.0, 2.0, 0.5]
    var none = List[Int32]()
    var regs: List[VoteMember] = [VoteMember(Int(r0.unsafe_ptr()), 1, 0, 1.0), VoteMember(Int(r1.unsafe_ptr()), 1, 0, 1.0)]
    var racc = List[Float64](length=3, fill=0.0)
    refused = bag_predict_run(_glue(), regs, xs, ds, 3, 1, False, Int(racc.unsafe_ptr()), 0)
    require(refused == VOTE_ALL_ANSWERED, "bag regress refused " + String(refused))
    var rwant: List[Float64] = [1.5, 2.0, 1.75]
    for i in range(3):  # small-loop(oracle)
        _near(racc[i], rwant[i], 0.0, "bag regress row " + String(i))
    _ = p0^
    _ = p1^
    _ = c0^
    _ = c1^
    _ = r0^
    _ = r1^
    _ = none^


def check_bag_oob_classifier() raises:
    """BaggingClassifier OOB: each member's probabilities summed at its out-of-bag rows, counted, normalized,
    argmax, hits against the true codes; a member with no out-of-bag rows is skipped."""
    var p0: List[Float32] = [0.5, 0.25, 0.25, 0.0, 1.0, 0.0, 0.25, 0.25, 0.5]
    var p1: List[Float32] = [0.75, 0.25, 0.5, 0.5, 0.0, 1.0]
    var c0: List[Int32] = [0, 1, 2]
    var c1: List[Int32] = [0, 2]
    var o0: List[Int32] = [0, 2, 4]
    var o1: List[Int32] = [1, 3, 5]
    var empty = List[Float32]()
    var members: List[VoteMember] = [_door(p0, 3, c0), _door(p1, 2, c1), VoteMember(0, 0, 0, 1.0)]
    var xs: List[Int] = [0, 0, 0]
    var ns: List[Int] = [3, 3, 0]
    var ds: List[Int] = [1, 1, 0]
    var oobs: List[Int] = [Int(o0.unsafe_ptr()), Int(o1.unsafe_ptr()), 0]
    var n = 6
    var k = 3
    var acc = List[Float64](length=n * k, fill=0.0)
    var counts = List[Int32](length=n, fill=Int32(0))
    var refused = bag_oob_run(_glue(), members, xs, ns, ds, oobs, n, k, True, Int(acc.unsafe_ptr()),
                              Int(counts.unsafe_ptr()), 0)
    require(refused == VOTE_ALL_ANSWERED, "oob refused " + String(refused))
    # oracle: member 0 at rows 0, 2, 4 (its 3 columns); member 1 at rows 1, 3, 5 (columns 0 and 2)
    var want = List[Float64](length=n * k, fill=0.0)
    for r in range(3):  # small-loop(oracle)
        for c in range(3):  # small-loop(oracle)
            want[Int(o0[r]) * k + c] += Float64(p0[r * 3 + c])
        want[Int(o1[r]) * k + 0] += Float64(p1[r * 2 + 0])
        want[Int(o1[r]) * k + 2] += Float64(p1[r * 2 + 1])
    for i in range(n * k):  # small-loop(oracle)
        _near(acc[i], want[i], 0.0, "oob sum cell " + String(i))
    for i in range(n):  # small-loop(oracle)
        require(counts[i] == 1, "oob count row " + String(i) + ": " + String(counts[i]))
    var truth: List[Int32] = [0, 0, 1, 0, 2, 2]
    var pred = List[Int32](length=n, fill=Int32(-1))
    var hits = bag_oob_classifier_finish(_glue(), Int(acc.unsafe_ptr()), Int(truth.unsafe_ptr()),
                                         Int(pred.unsafe_ptr()), n, k)
    var want_pred: List[Int32] = [0, 0, 1, 0, 2, 2]  # row 3 ties 0.5 / 0.5: the first
    for i in range(n):  # small-loop(oracle: the first argmax of each normalized row)
        require(pred[i] == want_pred[i], "oob argmax row " + String(i) + ": " + String(pred[i]))
    require(hits == 6, "oob hits " + String(hits))
    var row_sum = acc[0] + acc[1] + acc[2]
    _near(row_sum, 1.0, 1e-12, "oob normalized row 0")
    _ = p0^
    _ = p1^
    _ = c0^
    _ = c1^
    _ = o0^
    _ = o1^
    _ = empty^
    _ = truth^
    _ = pred^
    _ = counts^


def check_bag_oob_regressor() raises:
    """BaggingRegressor OOB: the members' predictions widened and summed at their out-of-bag rows, then the
    out-of-bag prediction (sum / count) and the R^2 sums."""
    var p0: List[Float32] = [1.0, 2.0]
    var p1: List[Float32] = [3.0, 5.0, 0.5]
    var o0: List[Int32] = [0, 1]
    var o1: List[Int32] = [1, 2, 3]
    var members: List[VoteMember] = [VoteMember(Int(p0.unsafe_ptr()), 1, 0, 1.0),
                                     VoteMember(Int(p1.unsafe_ptr()), 1, 0, 1.0)]
    var xs: List[Int] = [0, 0]
    var ns: List[Int] = [2, 3]
    var ds: List[Int] = [1, 1]
    var oobs: List[Int] = [Int(o0.unsafe_ptr()), Int(o1.unsafe_ptr())]
    var n = 4
    var acc = List[Float64](length=n, fill=0.0)
    var counts = List[Int32](length=n, fill=Int32(0))
    var refused = bag_oob_run(_glue(), members, xs, ns, ds, oobs, n, 1, False, Int(acc.unsafe_ptr()),
                              Int(counts.unsafe_ptr()), 0)
    require(refused == VOTE_ALL_ANSWERED, "oob regress refused " + String(refused))
    var want: List[Float64] = [1.0, 5.0, 5.0, 0.5]
    var want_counts: List[Int32] = [1, 2, 1, 1]
    for i in range(n):  # small-loop(oracle)
        _near(acc[i], want[i], 0.0, "oob regress sum row " + String(i))
        require(counts[i] == want_counts[i], "oob regress count row " + String(i))
    var y: List[Float32] = [1.0, 2.0, 4.0, 1.0]
    var pred = List[Float64](length=n, fill=0.0)
    var r2 = bag_oob_r2_run(_glue(), Int(acc.unsafe_ptr()), Int(counts.unsafe_ptr()), Int(y.unsafe_ptr()),
                            Int(pred.unsafe_ptr()), n)
    require(r2.finite, "oob r2 flagged a finite input")
    var mean = 0.0
    for i in range(n):  # small-loop(oracle)
        _near(pred[i], want[i] / Float64(want_counts[i]), 1e-12, "oob prediction row " + String(i))
        mean += Float64(y[i])
    mean /= Float64(n)
    var ss_res = 0.0
    var ss_tot = 0.0
    for i in range(n):  # small-loop(oracle)
        var e = Float64(y[i]) - pred[i]
        ss_res += e * e
        ss_tot += (Float64(y[i]) - mean) * (Float64(y[i]) - mean)
    _near(r2.ss_res, ss_res, 1e-9, "oob ss_res")
    _near(r2.ss_tot, ss_tot, 1e-9, "oob ss_tot")
    _ = p0^
    _ = p1^
    _ = o0^
    _ = o1^
    _ = y^
    _ = acc^
    _ = counts^


def check_stack_place() raises:
    """Stacking meta features: a fold member that saw classes 0 and 2 of 3 (predict_proba) is placed at its
    classes, column 1 = 0; a predict member after it takes one column; a binary predict_proba keeps its second
    column; a decision_function member that saw 3 of 4 classes fills the missing one with the lowest float32; one
    that saw 2 of 3 is refused in scikit-learn's words."""
    var g = _glue()
    var pa: List[Float32] = [0.25, 0.75, 0.5, 0.5]
    var ca: List[Int32] = [0, 2]
    var pr: List[Float32] = [7.0, 9.0]
    var members: List[VoteMember] = [_door(pa, 2, ca), VoteMember(Int(pr.unsafe_ptr()), 1, 0, 1.0)]
    var kinds: List[Int] = [STACK_PROBA, STACK_PREDICT]
    var counts: List[Int] = [2, 0]
    var place: List[Int32] = [1, 0]
    var meta = List[Float64](length=2 * 4, fill=-1.0)
    var widths = List[Int]()
    var refused = stack_place_run(g, VoteRows(0, 2, 1), members, kinds, counts, 3, False, Int(meta.unsafe_ptr()), 2,
                                  4, Int(place.unsafe_ptr()), 0, widths, 0)
    require(refused == VOTE_ALL_ANSWERED, "stack refused " + String(refused))
    require(len(widths) == 2 and widths[0] == 3 and widths[1] == 1, "stack widths")
    # fold row 0 goes to meta row 1, fold row 1 to meta row 0
    var want: List[Float64] = [0.5, 0.0, 0.5, 9.0, 0.25, 0.0, 0.75, 7.0]
    for i in range(8):  # small-loop(oracle)
        _near(meta[i], want[i], 0.0, "stack meta cell " + String(i))
    # binary predict_proba: the second column
    var pb: List[Float32] = [0.25, 0.75, 0.875, 0.125]
    var cb: List[Int32] = [0, 1]
    var bin_members: List[VoteMember] = [_door(pb, 2, cb)]
    var bin_kinds: List[Int] = [STACK_PROBA]
    var bin_counts: List[Int] = [2]
    var order: List[Int32] = [0, 1]
    var bmeta = List[Float64](length=2, fill=-1.0)
    var bw = List[Int]()
    _ = stack_place_run(g, VoteRows(0, 2, 1), bin_members, bin_kinds, bin_counts, 2, True, Int(bmeta.unsafe_ptr()), 2,
                        1, Int(order.unsafe_ptr()), 0, bw, 0)
    _near(bmeta[0], 0.75, 0.0, "binary proba row 0")
    _near(bmeta[1], 0.125, 0.0, "binary proba row 1")
    require(bw[0] == 1, "binary proba width")
    # decision_function, 3 of 4 classes: the missing class 1 is the lowest float32
    var dd: List[Float32] = [1.0, 2.0, 3.0]
    var cd: List[Int32] = [0, 2, 3]
    var dec_members: List[VoteMember] = [_door(dd, 3, cd)]
    var dec_kinds: List[Int] = [STACK_DECISION]
    var dec_counts: List[Int] = [3]
    var one: List[Int32] = [0]
    var dmeta = List[Float64](length=4, fill=0.0)
    var dw = List[Int]()
    _ = stack_place_run(g, VoteRows(0, 1, 1), dec_members, dec_kinds, dec_counts, 4, False, Int(dmeta.unsafe_ptr()),
                        1, 4, Int(one.unsafe_ptr()), 0, dw, 0)
    _near(dmeta[0], 1.0, 0.0, "decision class 0")
    _near(dmeta[1], Float64(Float32.MIN_FINITE), 0.0, "decision missing class 1")
    _near(dmeta[2], 2.0, 0.0, "decision class 2")
    _near(dmeta[3], 3.0, 0.0, "decision class 3")
    # decision_function, 2 of 3 classes: refused
    var d2: List[Float32] = [1.0, 2.0]
    var c2: List[Int32] = [0, 2]
    var two_members: List[VoteMember] = [_door(d2, 2, c2)]
    var two_counts: List[Int] = [2]
    var tw = List[Int]()
    var raised = False
    try:
        _ = stack_place_run(g, VoteRows(0, 1, 1), two_members, dec_kinds, two_counts, 3, False,
                            Int(dmeta.unsafe_ptr()), 1, 4, Int(one.unsafe_ptr()), 0, tw, 0)
    except e:
        raised = String(e).find("imbalanced folds") >= 0
    require(raised, "a two-class decision_function fold was not refused in scikit-learn's words")
    _ = pa^
    _ = ca^
    _ = pr^
    _ = place^
    _ = meta^
    _ = pb^
    _ = cb^
    _ = order^
    _ = bmeta^
    _ = dd^
    _ = cd^
    _ = one^
    _ = dmeta^
    _ = d2^
    _ = c2^


def check_calibrated_sigmoid() raises:
    """CalibratedClassifierCV.predict_proba, one pair, k = 3, Platt calibrators: p_c = 1 / (1 + exp(a_c s_c +
    b_c)), each row normalized, the mean over the one pair (atol 1e-6: the program narrows through float32)."""
    var s: List[Float32] = [0.5, -1.0, 2.0, 1.5, 0.0, -0.5]
    var cols: List[Int32] = [0, 1, 2]
    var members: List[VoteMember] = [_door(s, 3, cols)]
    var pairs: List[CalPair] = [CalPair(0, Int(cols.unsafe_ptr()), 3, 0, 3)]
    var a: List[Float64] = [-1.5, -0.5, -2.0]
    var b: List[Float64] = [0.25, -0.125, 0.5]
    var cals: List[CalFit] = [CalFit(False, a[0], b[0], 0, 0, 0), CalFit(False, a[1], b[1], 0, 0, 0),
                              CalFit(False, a[2], b[2], 0, 0, 0)]
    var acc = List[Float64](length=6, fill=0.0)
    var refused = calibrated_proba_run(_glue(), VoteRows(0, 2, 1), members, pairs, cals, 3, Int(acc.unsafe_ptr()))
    require(refused == VOTE_ALL_ANSWERED, "calibrated refused " + String(refused))
    for r in range(2):  # small-loop(oracle)
        var p = List[Float64](length=3, fill=0.0)
        var t = 0.0
        for c in range(3):  # small-loop(oracle)
            p[c] = 1.0 / (1.0 + exp(a[c] * Float64(s[r * 3 + c]) + b[c]))
            t += p[c]
        for c in range(3):  # small-loop(oracle)
            _near(acc[r * 3 + c], p[c] / t, 1e-6, "calibrated cell " + String(r) + "," + String(c))
    _ = s^
    _ = cols^


def check_ada() raises:
    """AdaBoostClassifier decision (votes +a, -a/(k-1), over the sum of the weights) and its argmax codes;
    AdaBoostRegressor's weighted median (scikit-learn `_get_median_predict`: the first sorted prediction whose
    cumulative weight reaches half the total)."""
    var g = _glue()
    var m0: List[Int32] = [0, 1, 2, 2]
    var m1: List[Int32] = [1, 1, 0, 2]
    var codes: List[Int] = [Int(m0.unsafe_ptr()), Int(m1.unsafe_ptr())]
    var alphas: List[Float64] = [0.75, 0.25]
    var n = 4
    var k = 3
    var acc = List[Float64](length=n * k, fill=0.0)
    ada_classifier_run(g, codes, alphas, n, k, Int(acc.unsafe_ptr()), ADA_DECISION, 0)
    var both = List[List[Int32]]()
    both.append(m0.copy())
    both.append(m1.copy())
    for i in range(n):  # small-loop(oracle)
        for c in range(k):  # small-loop(oracle)
            var v = 0.0
            for j in range(2):  # small-loop(the members, in order)
                v += alphas[j] if Int(both[j][i]) == c else -alphas[j] / 2.0
            _near(acc[i * k + c], v / 1.0, 1e-15, "ada decision " + String(i) + "," + String(c))
    var acc2 = List[Float64](length=n * k, fill=0.0)
    var got = List[Int32](length=n, fill=Int32(-1))
    ada_classifier_run(g, codes, alphas, n, k, Int(acc2.unsafe_ptr()), ADA_CODES, Int(got.unsafe_ptr()))
    var want: List[Int32] = [0, 1, 2, 2]
    for i in range(n):  # small-loop(oracle)
        require(got[i] == want[i], "ada code row " + String(i) + ": " + String(got[i]))
    # weighted median of 3 members over 2 rows
    var r0: List[Float32] = [1.0, 5.0]
    var r1: List[Float32] = [2.0, 4.0]
    var r2: List[Float32] = [3.0, 3.0]
    var blocks: List[Int] = [Int(r0.unsafe_ptr()), Int(r1.unsafe_ptr()), Int(r2.unsafe_ptr())]
    var w: List[Float64] = [0.25, 0.5, 0.25]
    var med = List[Float32](length=2, fill=Float32(-1.0))
    ada_regressor_run(g, blocks, w, 2, Int(med.unsafe_ptr()))
    # row 0: sorted 1, 2, 3 with cdf .25, .75, 1: first >= .5 is 2; row 1: sorted 3, 4, 5 (w .25, .5, .25): 4
    _near(Float64(med[0]), 2.0, 0.0, "ada median row 0")
    _near(Float64(med[1]), 4.0, 0.0, "ada median row 1")
    _ = m0^
    _ = m1^
    _ = acc^
    _ = acc2^
    _ = got^
    _ = r0^
    _ = r1^
    _ = r2^
    _ = med^


def check_roc_micro() raises:
    """test_roc_auc_ovr_micro_is_the_raveled_binary_auc: the micro one-vs-rest AUC equals scikit-learn's binary
    AUC of the raveled one-hot labels and scores (unweighted, and weighted by w repeated per class); a row that
    does not sum to 1 is refused before the AUC."""
    var y = fixture_i32(ROC_DIR + "roc-y.i32", ROC_N)
    var s = fixture_f32(ROC_DIR + "roc-s.f32", ROC_N * ROC_K)
    var w = fixture_f32(ROC_DIR + "roc-w.f32", ROC_N)
    var want = fixture_f64(ROC_DIR + "roc-auc.f64", 2)
    var route = MetricRoute(metrics_program_host_v1, False, False)
    var got = roc_ovr_micro_run(route, Int(s.unsafe_ptr()), ROC_N * ROC_K, Int(y.unsafe_ptr()), ROC_N, 0, -1, ROC_N,
                                ROC_K)
    require(got.rows_ok and got.auc.defined, "micro auc refused")
    _near(got.auc.value, want[0], 1e-6 * abs(want[0]), "micro auc")
    var got_w = roc_ovr_micro_run(route, Int(s.unsafe_ptr()), ROC_N * ROC_K, Int(y.unsafe_ptr()), ROC_N,
                                  Int(w.unsafe_ptr()), ROC_N, ROC_N, ROC_K)
    require(got_w.rows_ok and got_w.auc.defined, "weighted micro auc refused")
    _near(got_w.auc.value, want[1], 1e-6 * abs(want[1]), "weighted micro auc")
    var bad = s.copy()
    for c in range(ROC_K):  # small-loop(the Python test's bad[0] *= 2)
        bad[c] = bad[c] * 2.0
    var refused = roc_ovr_micro_run(route, Int(bad.unsafe_ptr()), ROC_N * ROC_K, Int(y.unsafe_ptr()), ROC_N, 0, -1,
                                    ROC_N, ROC_K)
    require(not refused.rows_ok, "a row summing to 2 was not refused")
    _ = y^
    _ = s^
    _ = w^
    _ = bad^


def test_meta_runs() raises:
    # The publication _mojolearn_x_trees_host makes at module init, once before the parallel cases.
    publish_x_trees_glue("cpu")
    def run_case(which: Int) raises:
        if which == 0:
            check_multioutput_proba()
        elif which == 1:
            check_multioutput_codes_refusal()
        elif which == 2:
            check_bag_predict()
        elif which == 3:
            check_bag_oob_classifier()
        elif which == 4:
            check_bag_oob_regressor()
        elif which == 5:
            check_stack_place()
        elif which == 6:
            check_calibrated_sigmoid()
        elif which == 7:
            check_ada()
        else:
            check_roc_micro()
    check_cases("meta_runs", CASES, run_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
