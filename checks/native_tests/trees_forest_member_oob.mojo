# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""The forest member's out-of-bag output (core/forest_member_run.mojo, parameter word 11, blocks `FM_BLOCK_OOB` /
`FM_BLOCK_OOB_SCORE`; lane mf-oob3, 2026-11-23) against the forest's OWN fit body.

X = default_rng(1).random((120, 4), float32), y = X0 + 0.5 X1, labels (X0 + X1 > 1) (`fixtures/trees_ada/`, the
pack lane mf-ada's generator writes; no new fixture). The member is the plan `forest_member_admit_common` writes for
the estimator (`member_plan_of`, checks/native_tests/mf_ada_fits.mojo) on the RF host image's forest entry. The
oracle is what `RandomForest*.fit` runs with oob_score=True on the host column: the door's request
(`forest_request` over `rf_config`, the column-major X `_x_for(X, False)` stages, the ForestFit oob and score words)
and the host image's fit body `_rf_fit_forest` (core/mojolearn_rf_host_run.mojo, `rf_host_oob` inside it). Each
case requires the member's five model blocks and both out-of-bag blocks equal to the oracle's word for word:

  0 rfc    RandomForestClassifier(n_estimators=8, max_depth=3, oob_score=True, random_state=0) on the labels
  1 rfcw   the same with sample_weight w_i = 1 + (i mod 3)
  2 rfr    RandomForestRegressor(n_estimators=8, max_depth=3, oob_score=True, random_state=0) on y
  3 rfrw   the regressor with sample_weight: the estimator's refusal words, and its kind's mask has no
           MEMBER_CAN_SAMPLE_WEIGHT
  4 et     an ExtraTrees plan with word 11 set: refused at MEMBER_OP_NEW in words
  5 plain  rfc without word 11: both out-of-bag blocks refused in words; the index walk of an oob state is still
           six blocks (the out-of-bag blocks are by name only)

The cases are independent fits and run across the host cores. Fail-proof edits (each must fail, then be
restored): in `_fit`, `column_major` without `or slot[].oob` fails 0 and 2 ("oob" words differ: `rf_host_oob` reads
X column-major); `score=` passed 0 fails 0-2 (the fit's "requested together" refusal); `_capabilities`' weights
rule without `or known[].oob` fails 3; the `_mint` ExtraTrees out-of-bag refusal removed fails 4; `_export`'s
`len(slot[].oob_score) == 0` refusal removed fails 5.
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require, fixture_f32, assert_same_words
from checks.native_tests.bm_rest_boost_fits import publish_host_images
from checks.native_tests.mf_ada_fits import publish_host_members, member_plan_of
from checks.native_tests.py2mojo_forest_fits import RfSpec, rf_config, column_major
from checks.native_tests.bm_trees_et_fits import EtSpec, et_config
from core.entry_table import entry_ref_unresolved
from core.member_abi import (
    MemberPlan, MemberMatrix, member_new, member_call, member_call_of, member_free, member_kind_capabilities,
    MEMBER_OP_FIT, MEMBER_OP_EXPORT, MEMBER_F32, MEMBER_I32, MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_BY_INDEX,
    MEMBER_CAN_SAMPLE_WEIGHT,
)
from core.member_families import MEMBER_FAMILY_FOREST, MEMBER_FAMILY_EXTRA_TREES
from core.forest_member_run import FM_BLOCK_OOB, FM_BLOCK_OOB_SCORE
from core.mojolearn_rf_host_run import _rf_fit_forest
from bindings.forest_config_run import forest_request
from xtrees.trees_forest_run import ForestFit

comptime DIR = "checks/native_tests/fixtures/trees_ada/"
comptime N = 120
comptime D = 4
comptime CASES = 6
comptime REFUSE_WEIGHTS = "sample_weight is not implemented together with oob_score"
comptime REFUSE_ET = "the ExtraTrees fit computes no out-of-bag outputs"
comptime REFUSE_PLAIN = "fitted without out-of-bag outputs"


def _labels(x: List[Float32]) -> List[Int32]:
    var out = List[Int32](length=N, fill=Int32(0))
    for i in range(N):  # small-loop(fixture labels: one per row)
        out[i] = Int32(1) if x[i * D] + x[i * D + 1] > Float32(1.0) else Int32(0)
    return out^


def _weights() -> List[Float32]:
    var w = List[Float32](length=N, fill=Float32(0))
    for i in range(N):  # small-loop(fixture weights: one per row)
        w[i] = Float32(1 + i % 3)
    return w^


def _spec(classifier: Bool) -> RfSpec:
    var s = RfSpec(classifier, 8, 0)
    s.max_depth = 3
    return s^


struct Oracle(Movable):
    """The forest's own oob fit: the five model arrays, the out-of-bag words and the score."""
    var offsets: List[Int32]
    var colid: List[Int32]
    var quesval: List[Float32]
    var left: List[Int32]
    var leaves: List[Float32]
    var oob: List[Float64]
    var score: List[Float64]

    def __init__(out self, s: RfSpec, x: List[Float32], codes: List[Int32], y: List[Float32],
                 weights: List[Float32]) raises:
        var classifier = s.classifier
        var weighted = len(weights) > 0
        var staged = column_major(x, N, D)
        var oob = List[Float64](length=N * (2 if classifier else 1), fill=0.0)
        var score = List[Float64](length=1, fill=0.0)
        var fit = ForestFit(classifier, rowmajor_ok=False, weights=Int(weights.unsafe_ptr()) if weighted else 0,
                            oob=Int(oob.unsafe_ptr()), score=Int(score.unsafe_ptr()))
        var y_addr = Int(codes.unsafe_ptr()) if classifier else Int(y.unsafe_ptr())
        var request = forest_request(rf_config(s), entry_ref_unresolved(), N, D, 2 if classifier else 0,
                                     Int(staged.unsafe_ptr()), y_addr, fit, False)
        request.gil_free = True
        if classifier:
            var r = _rf_fit_forest[True, False](request)
            self.offsets = r.offsets.copy()
            self.colid = r.colid.copy()
            self.quesval = r.quesval.copy()
            self.left = r.left_child.copy()
            self.leaves = r.leaves.copy()
        else:
            var r = _rf_fit_forest[False, False](request)
            self.offsets = r.offsets.copy()
            self.colid = r.colid.copy()
            self.quesval = r.quesval.copy()
            self.left = r.left_child.copy()
            self.leaves = r.leaves.copy()
        self.oob = oob^  # the fit wrote these words through their addresses (their last named use)
        self.score = score^
        _ = staged^  # the fit read these words through their addresses
        _ = request^
        _ = len(codes)
        _ = len(y)
        _ = len(weights)


def _member_fit(plan: MemberPlan, x: List[Float32], codes: List[Int32], y: List[Float32],
                weights: List[Float32]) raises -> Int:
    """A new state of `plan` fitted on row-major X (the member stages the column-major copy itself)."""
    var call = member_new(plan)
    try:
        call.x = MemberMatrix(Int(x.unsafe_ptr()), N, D, MEMBER_F32)
        if plan.kind == 0:
            call.y = MemberMatrix(Int(codes.unsafe_ptr()), N, 1, MEMBER_I32)
            call.classes = 2
        else:
            call.y = MemberMatrix(Int(y.unsafe_ptr()), N, 1, MEMBER_F32)
        if len(weights) > 0:
            call.weights = MemberMatrix(Int(weights.unsafe_ptr()), N, 1, MEMBER_F32)
        _ = member_call(plan.member, MEMBER_OP_FIT, call)
    except error:
        member_free(plan, call)
        raise error^
    _ = len(x)  # the fit read these words through their addresses
    _ = len(codes)
    _ = len(y)
    _ = len(weights)
    return call.state


def _block[dt: DType](plan: MemberPlan, state: Int, block: Int) raises -> List[Scalar[dt]]:
    """A copy of one exported block (by name)."""
    var call = member_call_of(plan, state)
    call.block = block
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var count = call.dst.rows * call.dst.cols
    var src = MutPointer[Scalar[dt], MutUntrackedOrigin](unsafe_from_address=call.dst.data)
    var out = List[Scalar[dt]](length=count, fill=Scalar[dt](0))
    for i in range(count):  # small-loop(one exported block's words)
        out[i] = src[i]
    return out^


def _refused(plan: MemberPlan, state: Int, block: Int, words: String, what: String) raises:
    try:
        _ = _block[DType.float64](plan, state, block)
    except problem:
        require(words in String(problem), what + ": unexpected refusal: " + String(problem))
        return
    raise Error(what + ": the block was not refused")


def _against_oracle(name: String, classifier: Bool, x: List[Float32], codes: List[Int32], y: List[Float32],
                    weights: List[Float32]) raises:
    var s = _spec(classifier)
    var want = Oracle(s, x, codes, y, weights)
    var plan = member_plan_of(rf_config(s), MEMBER_FAMILY_FOREST, classifier, True)
    var state = _member_fit(plan, x, codes, y, weights)
    var call = member_call_of(plan, state)
    try:
        assert_same_words[DType.int32](_block[DType.int32](plan, state, MEMBER_BLOCK_FAMILY), want.offsets,
                                       name + ": offsets")
        assert_same_words[DType.int32](_block[DType.int32](plan, state, MEMBER_BLOCK_FAMILY + 1), want.colid,
                                       name + ": colid")
        assert_same_words[DType.float32](_block[DType.float32](plan, state, MEMBER_BLOCK_FAMILY + 2), want.quesval,
                                         name + ": quesval")
        assert_same_words[DType.int32](_block[DType.int32](plan, state, MEMBER_BLOCK_FAMILY + 3), want.left,
                                       name + ": left_child")
        assert_same_words[DType.float32](_block[DType.float32](plan, state, MEMBER_BLOCK_FAMILY + 4), want.leaves,
                                         name + ": leaves")
        assert_same_words[DType.float64](_block[DType.float64](plan, state, FM_BLOCK_OOB), want.oob,
                                         name + (": oob_decision_function_" if classifier else ": oob_prediction_"))
        assert_same_words[DType.float64](_block[DType.float64](plan, state, FM_BLOCK_OOB_SCORE), want.score,
                                         name + ": oob_score_")
        # The index walk is unchanged: six blocks (no importances), the out-of-bag blocks by name only.
        var walk = member_call_of(plan, state)
        walk.block = 0
        walk.flags = MEMBER_BLOCK_BY_INDEX
        _ = member_call(plan.member, MEMBER_OP_EXPORT, walk)
        require(walk.out_cols == 6, name + ": an out-of-bag state walks " + String(walk.out_cols) + " blocks")
    finally:
        member_free(plan, call)


def run_case(unit: Int) raises:
    var x = fixture_f32(DIR + "ada-x.f32", N * D)
    var codes = _labels(x)
    var y = fixture_f32(DIR + "ada-y.f32", N)
    var none = List[Float32]()
    if unit == 0:
        _against_oracle("rfc", True, x, codes, y, none)
    elif unit == 1:
        _against_oracle("rfcw", True, x, codes, y, _weights())
    elif unit == 2:
        _against_oracle("rfr", False, x, codes, y, none)
    elif unit == 3:
        var plan = member_plan_of(rf_config(_spec(False)), MEMBER_FAMILY_FOREST, False, True)
        require((member_kind_capabilities(plan) & MEMBER_CAN_SAMPLE_WEIGHT) == 0,
                "rfrw: an out-of-bag regressor advertises sample weights")
        try:
            var state = _member_fit(plan, x, codes, y, _weights())
            var call = member_call_of(plan, state)
            member_free(plan, call)
        except problem:
            require(REFUSE_WEIGHTS in String(problem), "rfrw: unexpected refusal: " + String(problem))
            return
        raise Error("rfrw: the weighted out-of-bag regressor fit did not refuse")
    elif unit == 4:
        var e = EtSpec(True, 5, 0)
        e.max_depth = 2
        var plan = member_plan_of(et_config(e), MEMBER_FAMILY_EXTRA_TREES, True, True)
        try:
            var call = member_new(plan)
            member_free(plan, call)
        except problem:
            require(REFUSE_ET in String(problem), "et: unexpected refusal: " + String(problem))
            return
        raise Error("et: an ExtraTrees state with the out-of-bag word was minted")
    else:
        var plan = member_plan_of(rf_config(_spec(True)), MEMBER_FAMILY_FOREST, True, False)
        var state = _member_fit(plan, x, codes, y, none)
        var call = member_call_of(plan, state)
        try:
            _refused(plan, state, FM_BLOCK_OOB, REFUSE_PLAIN, "plain: oob block")
            _refused(plan, state, FM_BLOCK_OOB_SCORE, REFUSE_PLAIN, "plain: oob score block")
        finally:
            member_free(plan, call)


def test_forest_member_oob_matches_the_forest_fit() raises:
    publish_host_images()
    publish_host_members()
    check_cases("forest_member_oob", CASES, run_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
