# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""GaussianMixture's MEMBER_OP_SCORE (lane mf-rows8c, 2026-11-23).

The mixture member (core/mixture_member.mojo `_score_mean`) over the host column's GMM entry
(mixture/host/gmm_native_host.mojo), so a search with scoring=None scores a GaussianMixture unit by the class's own
`score` (core/msel_search_run.mojo `_member_score`; providers/python/member_ensemble_boundary_door.mojo no longer
asks MEMBER_CAN_SCORE of a kind without it, it scores the typed scorer its class's score is). The data are the two
tight, far-apart groups of four points of checks/native_tests/mixture_native.mojo.

    unit 0   the kind's capability mask carries MEMBER_CAN_SCORE
    unit 1   SCORE's value == word 0 of the entry's MIXTURE_NATIVE_SCORE_BIC_AIC over the exported fitted blocks
             (bit-equal: the public `score`'s one body, bindings/mixture_public_native_pack.mojo `_gmm_bic_aic`)
    unit 2   SCORE's value against the mean of the member's own SCORE_SAMPLES (an independent fold, rtol 1e-5)
    unit 3   SCORE before a fit refuses ("call fit before score")

Fail-proof: in core/mixture_member.mojo `_score_mean` write `call[].value = words[1]` (the BIC): units 1 and 2
fail; drop MEMBER_CAN_SCORE from the capability answer: unit 0 fails; drop the `fitted` check: unit 3 fails.
The units are independent and run across the host cores (`check_cases`).
"""
from std.math import isfinite
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.device_route import DeviceRoute, DEVICE_CPU
from core.entry_table import EntryRef, entry_ref_word
from core.member_abi import (
    MemberPlan, MemberRef, MemberCall, MemberMatrix, MEMBER_OP_CAPABILITIES, MEMBER_OP_EXPORT, MEMBER_OP_FIT,
    MEMBER_OP_SCORE, MEMBER_OP_SCORE_SAMPLES, MEMBER_CAN_SCORE, MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_SCALARS,
    member_call, member_call_of, member_new, member_free, member_kind_capabilities,
)
from core.member_families import MEMBER_FAMILY_MIXTURE
from core.mixture_member import mixture_member_v2
from core.mixture_native_api import GmmDriver, GmmScoreRequest, MIXTURE_NATIVE_SCORE_BIC_AIC
from mixture.host.gmm_native_host import gmm_native_host_v1

comptime ROWS = 8
comptime FEATURES = 2
comptime K = 2


def _data() -> List[Float32]:
    var x: List[Float32] = [0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1,
                            10.0, 10.0, 10.1, 10.0, 10.0, 10.1, 10.1, 10.1]
    return x^


def _plan() raises -> MemberPlan:
    """GaussianMixture(n_components=2, random_state=0) as its UNPACK admits it, on the host GMM entry."""
    var mode = Int(GLOBAL_NUMERIC_MODE)
    var plan = MemberPlan()
    var route = DeviceRoute(DEVICE_CPU)
    route.mode = mode
    route.vendor = 0
    route.entry = EntryRef(gmm_native_host_v1, entry_ref_word(mode, 0, 0))
    route.flags = 0  # the host entry has no fit_predict (MIXTURE_CAP_FIT_PREDICT clear)
    plan.family = MEMBER_FAMILY_MIXTURE
    plan.kind = 0
    plan.member = MemberRef(EntryRef(mixture_member_v2, 0), 0)
    # [n_components, max_iter, tol, reg_covar, random_state bits (0), init_params 0 'kmeans']
    plan.params = [2.0, 100.0, 1e-3, 1e-6, 0.0, 0.0]
    plan.route = route
    var probe = MemberCall(route)
    plan.member.capabilities = member_call(plan.member, MEMBER_OP_CAPABILITIES, probe)
    return plan^


def _block_address(plan: MemberPlan, state: Int, block: Int) raises -> Int:
    var call = member_call_of(plan, state)
    call.block = block
    call.flags = 0
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    return call.dst.data


def score_case(unit: Int) raises:
    var plan = _plan()
    if unit == 0:
        require((member_kind_capabilities(plan) & MEMBER_CAN_SCORE) != 0, "GaussianMixture publishes its score")
        return
    var x = _data()
    var call = member_new(plan)
    try:
        if unit == 3:
            var refused = False
            var ask = member_call_of(plan, call.state)
            ask.x = MemberMatrix(Int(x.unsafe_ptr()), ROWS, FEATURES)
            try:
                _ = member_call(plan.member, MEMBER_OP_SCORE, ask)
            except problem:
                refused = String(problem).find("call fit before score") >= 0
            require(refused, "score before fit refuses in its words")
            return
        call.x = MemberMatrix(Int(x.unsafe_ptr()), ROWS, FEATURES)
        _ = member_call(plan.member, MEMBER_OP_FIT, call)
        var ask = member_call_of(plan, call.state)
        ask.x = MemberMatrix(Int(x.unsafe_ptr()), ROWS, FEATURES)
        _ = member_call(plan.member, MEMBER_OP_SCORE, ask)
        var value = ask.value
        require(isfinite(value), "a finite mean log-likelihood")
        if unit == 1:
            var scalars = MutPointer[Float64, MutUntrackedOrigin](
                unsafe_from_address=_block_address(plan, call.state, MEMBER_BLOCK_SCALARS))
            var words = List[Float64](length=3, fill=0.0)
            GmmDriver(gmm_native_host_v1).score(MIXTURE_NATIVE_SCORE_BIC_AIC, GmmScoreRequest(
                _block_address(plan, call.state, MEMBER_BLOCK_FAMILY),
                _block_address(plan, call.state, MEMBER_BLOCK_FAMILY + 1),
                _block_address(plan, call.state, MEMBER_BLOCK_FAMILY + 2),
                _block_address(plan, call.state, MEMBER_BLOCK_FAMILY + 3),
                _block_address(plan, call.state, MEMBER_BLOCK_FAMILY + 4),
                Int(x.unsafe_ptr()), Int(words.unsafe_ptr()), K, FEATURES, Int(scalars[0]), scalars[1] != 0.0, ROWS,
                scalars[2]))
            require(value == words[0], "score is the entry's BIC/AIC word 0: " + String(value) + " vs "
                    + String(words[0]))
            _ = len(words)
        else:
            var samples = List[Float32](length=ROWS, fill=0.0)
            var each = member_call_of(plan, call.state)
            each.x = MemberMatrix(Int(x.unsafe_ptr()), ROWS, FEATURES)
            each.dst = MemberMatrix(Int(samples.unsafe_ptr()), ROWS, 1)
            _ = member_call(plan.member, MEMBER_OP_SCORE_SAMPLES, each)
            var total = 0.0
            for r in range(ROWS):  # small-loop(oracle: the fixture's rows, in order)
                total += Float64(samples[r])
            var mean = total / Float64(ROWS)
            require(abs(value - mean) <= 1e-5 * max(abs(mean), 1.0), "score is the mean of score_samples: "
                    + String(value) + " vs " + String(mean))
            _ = len(samples)
    finally:
        member_free(plan, call)
    _ = len(x)  # the fit's and the queries' words stay alive past the member calls


def test_mixture_member_score() raises:
    check_cases("mixture_member_score", 4, score_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
