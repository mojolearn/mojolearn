# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""The last classes with no member entry, as member kinds (lane mf-hostent5, 2026-11-23), in a process with no
Python and no tree image: the typed bodies the door and the member share and the kinds' words, capabilities and
refusals.

Cases (one row each, run across the host cores by `check_cases`):
  0 OrderedRMSE's permutation body (gbdt/trees_gbdt_fit_run.mojo `gbdt_ordered_order`, which the door's
    `ordered_rmse_unpack` and the Gbdt member's kind 4 both run): the int64 bijection [3, 0, 2, 1] gives the
    uint32 order [3, 0, 2, 1]; a length other than the rows, a repeated id, a negative id and an id at n are each
    refused in the door's words (`GBDT_ORDERED_BIJECTION`). Fail-proof: drop the `seen[v]` test (the repeat is
    accepted), or write `UInt32(i)` (the order is the identity).
  1 OrderedRMSE's X and y bodies (`gbdt_ordered_rows_check`, `gbdt_ordered_target_check`): 3 rows and 0
    features are refused with ">=4 rows and >=1 feature", a NaN cell of X with "X must be finite", an infinite
    target with "y must be finite"; 4 finite rows pass. Fail-proof: `n < 3` in the rows check.
  2 the permutation through a caller's fit parameters (core/member_set_run.mojo `MemberFitExtras.permutation`,
    `member_fit_extras_apply` / `member_fit_extras_clear`): given() with only a permutation; the FIT call's
    `permutation` view is rows x 1 int64 at the caller's address; cleared after the fit. Fail-proof: drop the
    `call.permutation =` line in `member_fit_extras_apply`.
  3 the Gbdt kinds (core/gbdt_member.mojo): kind 4 (OrderedRMSE) mints from its config, five settings and tail,
    fits (MEMBER_CAN_FIT, MEMBER_CAN_SAMPLE_WEIGHT), has no predict_proba, answers one float32 predict column,
    and its FIT without a permutation raises Python's missing-argument words before any work; kind 5 (HostGBDT)
    has no fit (MEMBER_CAN_FIT absent, FIT raises "'HostGBDT' object has no attribute 'fit'"), predict_proba only
    for Logloss / CrossEntropy, and a MultiClass proba raises HostGBDT's own words. Fail-proof: drop the
    `_KIND_HOST` branch of `_capabilities` (MEMBER_CAN_FIT appears), or the permutation test of `_fit_ordered`.
  4 the forest kinds 2 / 3 (core/forest_member_run.mojo, the words `forest_member_host_admit` writes): no fit, no
    score, a classifier's predict_proba and MEMBER_CAN_CLASSIFIER; FIT and SCORE raise HostForest's missing
    attribute words; OUTPUTS with no state answers int32 codes for kind 2 and float32 values for kind 3.
    Fail-proof: drop the `predict_only` branch of `_capabilities` (MEMBER_CAN_FIT appears), or the `words[0] ==
    2.0` term of `classifier` (kind 2 has no predict_proba).
The kinds against each class's public RUN (an OrderedRMSE member fit and the door's fit, the same text; a
HostForest / HostGBDT held state's predict and the class's own predict, the same bits) need the tree images:
they are the digest lines of tools/mf_del5_digests.py (section hostent5) and the Python tests named in the
lane's result.
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require
from core.member_abi import (
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_I32, MEMBER_I64, MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW,
    MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_SCORE, MEMBER_OP_OUTPUTS, MEMBER_OP_PREDICT, MEMBER_OP_PREDICT_PROBA,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_PREDICT_PROBA, MEMBER_CAN_SAMPLE_WEIGHT, MEMBER_CAN_SCORE,
    MEMBER_CAN_CLASSIFIER, MEMBER_CAN_IMPORT,
)
from core.device_route import DeviceRoute, DEVICE_CPU
from core.member_set_run import MemberFitExtras, member_fit_extras_apply, member_fit_extras_clear
from gbdt.trees_gbdt_fit_run import (
    gbdt_ordered_order, gbdt_ordered_rows_check, gbdt_ordered_target_check, GBDT_ORDERED_BIJECTION,
)
from bindings.gbdt_config_run import GbdtConfig, gbdt_config_words
from core.gbdt_member import GbdtMemberImage, _member as gbdt_member
from core.forest_member_run import fm_member, FM_HEAD

comptime CASES = 5
comptime _FOREST_STORE: StaticString = "MojolearnForestMemberStoreHostent5Test"


struct _NoImage(GbdtMemberImage):
    """A gbdt image that holds no resident model (the host image's answer)."""

    @staticmethod
    def release(handle: Int) raises:
        _ = handle


def _refusal_of(ids: List[Int64], n: Int) -> String:
    var words = String("")
    try:
        _ = gbdt_ordered_order(Int(ids.unsafe_ptr()), len(ids), n)
    except error:
        words = String(error)
    return words^


def check_order() raises:
    var ids: List[Int64] = [3, 0, 2, 1]
    var order = gbdt_ordered_order(Int(ids.unsafe_ptr()), 4, 4)
    require(len(order) == 4 and order[0] == 3 and order[1] == 0 and order[2] == 2 and order[3] == 1,
            "the bijection's uint32 order is its ids")
    var short: List[Int64] = [0, 1, 2]
    require(_refusal_of(short, 4) == String(GBDT_ORDERED_BIJECTION), "a length other than the rows")
    var repeat: List[Int64] = [0, 1, 1, 3]
    require(_refusal_of(repeat, 4) == String(GBDT_ORDERED_BIJECTION), "a repeated id")
    var negative: List[Int64] = [0, -1, 2, 3]
    require(_refusal_of(negative, 4) == String(GBDT_ORDERED_BIJECTION), "a negative id")
    var high: List[Int64] = [0, 1, 2, 4]
    require(_refusal_of(high, 4) == String(GBDT_ORDERED_BIJECTION), "an id at n")
    _ = ids^


def _rows_refusal(x: List[Float32], n: Int, d: Int) -> String:
    var words = String("")
    try:
        gbdt_ordered_rows_check(Int(x.unsafe_ptr()), n, d)
    except error:
        words = String(error)
    return words^


def check_rows() raises:
    var x = List[Float32](length=8, fill=Float32(0.5))
    require("requires >=4 rows and >=1 feature" in _rows_refusal(x, 3, 2), "3 rows are refused")
    require("requires >=4 rows and >=1 feature" in _rows_refusal(x, 4, 0), "0 features are refused")
    require(_rows_refusal(x, 4, 2).byte_length() == 0, "4 finite rows pass")
    var bad = x.copy()
    bad[5] = Float32(0.0) / Float32(0.0)
    require("ordered RMSE X must be finite" in _rows_refusal(bad, 4, 2), "a NaN cell is refused")
    var y = List[Float32](length=4, fill=Float32(1.0))
    gbdt_ordered_target_check(Int(y.unsafe_ptr()), 4)
    y[2] = Float32(1.0) / Float32(0.0)
    var words = String("")
    try:
        gbdt_ordered_target_check(Int(y.unsafe_ptr()), 4)
    except error:
        words = String(error)
    require(words == "mojolearn: y must be finite", "an infinite target is refused: " + words)
    _ = x^
    _ = bad^
    _ = y^


def check_extras() raises:
    var ids: List[Int64] = [1, 0, 3, 2]
    var extras = MemberFitExtras()
    extras.rows = 4
    require(not extras.given(), "nothing given")
    extras.permutation = Int(ids.unsafe_ptr())
    require(extras.given(), "a permutation alone is given")
    var call = MemberCall(DeviceRoute(DEVICE_CPU))
    var held = List[Float32]()
    member_fit_extras_apply(extras, call, 0, 0, False, held)
    require(call.permutation.rows == 4 and call.permutation.cols == 1 and call.permutation.dtype == MEMBER_I64
            and call.permutation.data == Int(ids.unsafe_ptr()), "the permutation view: rows x 1 int64, the caller's")
    require(not call.weights.given(), "no weights view without weights")
    member_fit_extras_clear(call)
    require(not call.permutation.given(), "cleared after the fit")
    _ = ids^


def _gbdt_words(kind: Float64, loss: String, var extra: List[Float64]) -> List[Float64]:
    var config = GbdtConfig()
    config.loss = loss
    var words: List[Float64] = [kind]
    words.extend(gbdt_config_words(config))
    words.extend(extra^)
    words.append(0.0)
    words.append(0.0)
    return words^


def _gbdt_new(words: List[Float64]) raises -> Int:
    var call = MemberCall(DeviceRoute(DEVICE_CPU))
    call.params = Int(words.unsafe_ptr())
    call.param_count = len(words)
    _ = gbdt_member[_NoImage](MEMBER_OP_NEW, Int(MutPointer(to=call)))
    _ = len(words)
    return call.state


def _gbdt_ask(state: Int, operation: Int) -> String:
    """The words a FIT / PREDICT_PROBA of `state` raises with empty views ("" when it does not raise)."""
    var call = MemberCall(DeviceRoute(DEVICE_CPU), state)
    var words = String("")
    try:
        _ = gbdt_member[_NoImage](operation, Int(MutPointer(to=call)))
    except error:
        words = String(error)
    return words^


def _gbdt_mask(state: Int) raises -> Int:
    var call = MemberCall(DeviceRoute(DEVICE_CPU), state)
    return gbdt_member[_NoImage](MEMBER_OP_CAPABILITIES, Int(MutPointer(to=call)))


def _gbdt_free(state: Int) raises:
    var call = MemberCall(DeviceRoute(DEVICE_CPU), state)
    _ = gbdt_member[_NoImage](MEMBER_OP_FREE, Int(MutPointer(to=call)))


def check_gbdt_kinds() raises:
    var settings: List[Float64] = [100.0, 6.0, 128.0, 0.03, 3.0]
    var ordered = _gbdt_new(_gbdt_words(4.0, "RMSE", settings^))
    var mask = _gbdt_mask(ordered)
    require((mask & MEMBER_CAN_FIT) != 0 and (mask & MEMBER_CAN_SAMPLE_WEIGHT) != 0, "OrderedRMSE fits, weighted")
    require((mask & MEMBER_CAN_PREDICT_PROBA) == 0, "OrderedRMSE has no predict_proba")
    var outputs = MemberCall(DeviceRoute(DEVICE_CPU), ordered)
    outputs.flags = MEMBER_OP_PREDICT
    _ = gbdt_member[_NoImage](MEMBER_OP_OUTPUTS, Int(MutPointer(to=outputs)))
    require(outputs.out_cols == 1 and outputs.out_dtype == MEMBER_F32, "OrderedRMSE predicts one float32 column")
    require(_gbdt_ask(ordered, MEMBER_OP_FIT)
            == "TypeError: OrderedRMSE.fit() missing 1 required keyword-only argument: 'permutation'",
            "a fit without the permutation: Python's words")
    _gbdt_free(ordered)
    var multi = _gbdt_new(_gbdt_words(5.0, "MultiClass", List[Float64]()))
    var host_mask = _gbdt_mask(multi)
    require((host_mask & MEMBER_CAN_FIT) == 0 and (host_mask & MEMBER_CAN_SAMPLE_WEIGHT) == 0, "HostGBDT has no fit")
    require((host_mask & MEMBER_CAN_PREDICT) != 0 and (host_mask & MEMBER_CAN_IMPORT) != 0, "HostGBDT predicts imported")
    require((host_mask & MEMBER_CAN_PREDICT_PROBA) == 0, "a MultiClass HostGBDT has no predict_proba")
    require(_gbdt_ask(multi, MEMBER_OP_FIT) == "AttributeError: 'HostGBDT' object has no attribute 'fit'",
            "a HostGBDT fit: its missing attribute")
    require(_gbdt_ask(multi, MEMBER_OP_PREDICT_PROBA)
            == "NotImplementedError: HostGBDT.predict_proba has no qualified multiclass transform",
            "a MultiClass HostGBDT proba: its words")
    _gbdt_free(multi)
    var binary = _gbdt_new(_gbdt_words(5.0, "Logloss", List[Float64]()))
    require((_gbdt_mask(binary) & MEMBER_CAN_PREDICT_PROBA) != 0, "a Logloss HostGBDT has predict_proba")
    _gbdt_free(binary)


def _forest_words(kind: Float64, et: Bool) -> List[Float64]:
    """What `forest_member_host_admit` writes (providers/python/forest_member_common_boundary_door.mojo)."""
    var words: List[Float64] = [kind, 1.0 if et else 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0]
    for _ in range(3):  # small-loop(the three unread slots)
        words.append(0.0)
        words.append(0.0)
        words.append(1.0)
    return words^


def _forest(operation: Int, mut call: MemberCall) raises -> Int:
    return fm_member[_FOREST_STORE, True](operation, Int(MutPointer(to=call)), "RandomForest")


def _forest_refusal(state: Int, operation: Int) -> String:
    var call = MemberCall(DeviceRoute(DEVICE_CPU), state)
    var words = String("")
    try:
        _ = _forest(operation, call)
    except error:
        words = String(error)
    return words^


def check_forest_kinds() raises:
    for which in range(2):  # small-loop(kinds 2 and 3)
        var classifier = which == 0
        var words = _forest_words(2.0 if classifier else 3.0, False)
        require(len(words) == FM_HEAD + 9, "the predict-only words")
        var minted = MemberCall(DeviceRoute(DEVICE_CPU))
        minted.params = Int(words.unsafe_ptr())
        minted.param_count = len(words)
        _ = _forest(MEMBER_OP_NEW, minted)
        var state = minted.state
        var ask = MemberCall(DeviceRoute(DEVICE_CPU), state)
        var mask = _forest(MEMBER_OP_CAPABILITIES, ask)
        require((mask & MEMBER_CAN_FIT) == 0 and (mask & MEMBER_CAN_SCORE) == 0, "a saved forest has no fit or score")
        require((mask & MEMBER_CAN_PREDICT) != 0 and (mask & MEMBER_CAN_IMPORT) != 0, "a saved forest predicts imported")
        require(((mask & MEMBER_CAN_PREDICT_PROBA) != 0) == classifier, "predict_proba for a classifier file only")
        require(((mask & MEMBER_CAN_CLASSIFIER) != 0) == classifier, "MEMBER_CAN_CLASSIFIER for a classifier file")
        require(_forest_refusal(state, MEMBER_OP_FIT) == "AttributeError: 'HostForest' object has no attribute 'fit'",
                "a HostForest fit: its missing attribute")
        require(_forest_refusal(state, MEMBER_OP_SCORE) == "AttributeError: 'HostForest' object has no attribute 'score'",
                "a HostForest score: its missing attribute")
        var outputs = MemberCall(DeviceRoute(DEVICE_CPU))
        outputs.params = Int(words.unsafe_ptr())
        outputs.param_count = len(words)
        outputs.flags = MEMBER_OP_PREDICT
        outputs.classes = 3 if classifier else 0
        _ = _forest(MEMBER_OP_OUTPUTS, outputs)
        require(outputs.out_dtype == (MEMBER_I32 if classifier else MEMBER_F32), "predict's element type")
        var released = MemberCall(DeviceRoute(DEVICE_CPU), state)
        _ = _forest(MEMBER_OP_FREE, released)
        _ = len(words)


def test_mf_hostent5_kinds() raises:
    def run_case(which: Int) raises:
        if which == 0:
            check_order()
        elif which == 1:
            check_rows()
        elif which == 2:
            check_extras()
        elif which == 3:
            check_gbdt_kinds()
        else:
            check_forest_kinds()
    check_cases("mf_hostent5_kinds", CASES, run_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
