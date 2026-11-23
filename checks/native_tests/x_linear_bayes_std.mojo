# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""BayesianRidge's and ARDRegression's posterior against scikit-learn (lane mf-iterimp2, 2026-11-23).

The x_linear member (core/xlinear_member.mojo) over the host column's operations entry
(x_linear/ops_entry_host.mojo): one FIT, then its posterior blocks (X_OFFSET; BayesianRidge's FACTOR, V and ev,
ARDRegression's SIGMA) and one PREDICT_STD with the mean in `call.y` (XL_OP_BAYES_STD: BayesianRidge from the factor,
never forming sigma_), against scikit-learn's X_offset_, sigma_ and predict(X, return_std=True) on the same float32
inputs (fitted in float64). BayesianRidge's sigma_ is the public attribute's lazy first read: x_linear/door_run.mojo
`xl_bayes_sigma_run` over the exported factor. Inputs and references:
tools/py2mojo_native_fixtures_mf_iterimp2.py (pack `mf_bayes_std`). Tolerance rtol 2e-3, atol 2e-3 (sigma_ atol
1e-4: its entries are small), the float32 fit against a float64 reference.

    unit 0   BayesianRidge(), X 200 x 5
    unit 1   BayesianRidge(fit_intercept=False)
    unit 2   ARDRegression() on X 200 x 6 whose target reads two columns: the others are pruned (dk < d)
    unit 3   ARDRegression(fit_intercept=False) on the same data

Each unit checks dk (sigma_'s side), X_offset_, sigma_, the mean and the standard deviation of 50 query rows.
Fail-proof: in x_linear/bayes.mojo `bayes_factor_std_row` return `fsqrt(s)` (drop the 1 / alpha_ term): units 0
and 1 fail naming a std word (`ard_std_row` the same: units 2 and 3); in `bayes_sigma_cell` read `v + j * d + m`
twice (sigma_[j, j] for every k): units 0 and 1 fail naming a sigma_ word. The units are independent and run across the host cores.
"""
from std.testing import TestSuite
from checks.native_tests.py2mojo_fixture_words import fixture_f64
from checks.native_tests.py2mojo_host_members import rows_close
from checks.native_tests.testsuite_cases import check_cases, fixture_f32, require
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.device_route import DeviceRoute, DEVICE_CPU
from core.entry_table import EntryRef, entry_ref_word
from core.member_abi import (
    MemberPlan, MemberRef, MemberCall, MemberMatrix, MEMBER_OP_CAPABILITIES, MEMBER_OP_EXPORT, MEMBER_OP_FIT,
    MEMBER_OP_PREDICT_STD, MEMBER_CAN_PREDICT_STD, member_call, member_call_of, member_new, member_free,
)
from core.member_families import MEMBER_FAMILY_XLINEAR
from core.xlinear_member import (
    xlinear_member_v2, XL_MEMBER_BLOCK_SIGMA, XL_MEMBER_BLOCK_X_OFFSET, XL_MEMBER_BLOCK_FACTOR,
)
from core.member_abi import MEMBER_BLOCK_SCALARS, MEMBER_F64
from x_linear.door_run import xl_bayes_sigma_run
from x_linear.ops_entry_host import x_linear_ops_host_v1

comptime DIR = "checks/native_tests/fixtures/mf_bayes_std/"
comptime N = 200
comptime Q = 50


def _plan(unit: Int) raises -> MemberPlan:
    """The member plan the x_linear provider admits (providers/python/member_xlinear_boundary_door.mojo
    `_xl_bayes_words` / `_xl_ard_words`) for the unit's estimator at its default keywords, on the host operations
    entry."""
    var mode = Int(GLOBAL_NUMERIC_MODE)
    var plan = MemberPlan()
    var route = DeviceRoute(DEVICE_CPU)
    route.mode = mode
    route.vendor = 0
    route.entry = EntryRef(x_linear_ops_host_v1, entry_ref_word(mode, 0, 0))
    plan.family = MEMBER_FAMILY_XLINEAR
    plan.member = MemberRef(EntryRef(xlinear_member_v2, 0), 0)
    var intercept = 1.0 if unit % 2 == 0 else 0.0
    if unit < 2:
        plan.kind = 4
        # [kind 4, fit_intercept, max_iter, tol, alpha_1, alpha_2, lambda_1, lambda_2, alpha_init None, lambda_init None]
        plan.params = [4.0, intercept, 300.0, 1e-3, 1e-6, 1e-6, 1e-6, 1e-6, -1.0, -1.0]
    else:
        plan.kind = 5
        # [kind 5, fit_intercept, max_iter, tol, alpha_1, alpha_2, lambda_1, lambda_2, threshold_lambda]
        plan.params = [5.0, intercept, 300.0, 1e-3, 1e-6, 1e-6, 1e-6, 1e-6, 1e4]
    plan.route = route
    var probe = MemberCall(route)
    plan.member.capabilities = member_call(plan.member, MEMBER_OP_CAPABILITIES, probe)
    return plan^


def _block(plan: MemberPlan, state: Int, block: Int) raises -> List[Float32]:
    """A copy of float32 fitted block `block` of `state`."""
    var call = member_call_of(plan, state)
    call.block = block
    call.flags = 0
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var view = call.dst
    var count = view.rows * view.cols
    var out = List[Float32](length=count, fill=0.0)
    var src = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=view.data)
    for i in range(count):  # small-loop(oracle: one fitted block)
        out[i] = src[i]
    return out^


def bayes_case(unit: Int) raises:
    var names: List[String] = ["BayesianRidge()", "BayesianRidge(fit_intercept=False)", "ARDRegression()",
                               "ARDRegression(fit_intercept=False)"]
    var label = names[unit]
    var d = 5 if unit < 2 else 6
    var data = "bayes-" if unit < 2 else "ard-"
    var x = fixture_f32(DIR + data + "x.f32", N * d)
    var y = fixture_f32(DIR + data + "y.f32", N)
    var xq = fixture_f32(DIR + data + "xq.f32", Q * d)
    var name = DIR + "u" + String(unit)
    var dk = Int(fixture_f64(name + "-dk.f64", 1)[0])
    var plan = _plan(unit)
    require((plan.member.capabilities & MEMBER_CAN_PREDICT_STD) != 0, label + ": PREDICT_STD is published")
    var call = member_new(plan)
    try:
        call.x = MemberMatrix(Int(x.unsafe_ptr()), N, d)
        call.y = MemberMatrix(Int(y.unsafe_ptr()), N, 1)
        _ = member_call(plan.member, MEMBER_OP_FIT, call)
        var sigma: List[Float32]
        if unit < 2:
            # the factor (d + 1) x d, then sigma_ as its first read forms it
            var factor = _block(plan, call.state, XL_MEMBER_BLOCK_FACTOR)
            require(len(factor) == d * d + d, label + ": the factor has " + String(len(factor)) + " words")
            var sc = member_call_of(plan, call.state)
            sc.block = MEMBER_BLOCK_SCALARS
            sc.flags = 0
            _ = member_call(plan.member, MEMBER_OP_EXPORT, sc)
            var scalars = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=sc.dst.data)
            sigma = List[Float32](length=d * d, fill=0.0)
            xl_bayes_sigma_run(Int(factor.unsafe_ptr()), Int(factor.unsafe_ptr()) + 4 * d * d, d, scalars[1],
                               scalars[2], Int(sigma.unsafe_ptr()))
            _ = len(factor)  # the factor stays alive past the RUN that read it
        else:
            sigma = _block(plan, call.state, XL_MEMBER_BLOCK_SIGMA)
        require(len(sigma) == dk * dk, label + ": sigma_ has " + String(len(sigma)) + " words, expected "
            + String(dk * dk))
        rows_close(_block(plan, call.state, XL_MEMBER_BLOCK_X_OFFSET), fixture_f64(name + "-offset.f64", d), 2e-3,
            2e-3, label + ": X_offset_")
        rows_close(sigma, fixture_f64(name + "-sigma.f64", dk * dk), 2e-3, 1e-4, label + ": sigma_")
        var mean = List[Float32](length=Q, fill=0.0)
        var std = List[Float32](length=Q, fill=0.0)
        var ask = member_call_of(plan, call.state)
        ask.x = MemberMatrix(Int(xq.unsafe_ptr()), Q, d)
        ask.dst = MemberMatrix(Int(std.unsafe_ptr()), Q, 1)
        ask.y = MemberMatrix(Int(mean.unsafe_ptr()), Q, 1)
        _ = member_call(plan.member, MEMBER_OP_PREDICT_STD, ask)
        rows_close(mean, fixture_f64(name + "-mean.f64", Q), 2e-3, 2e-3, label + ": predict mean")
        rows_close(std, fixture_f64(name + "-std.f64", Q), 2e-3, 2e-3, label + ": predict std")
    finally:
        member_free(plan, call)
    _ = len(x)  # the fit's and the query's words stay alive past the member calls
    _ = len(y)
    _ = len(xq)


def test_bayes_predict_std() raises:
    check_cases("bayes_predict_std", 4, bayes_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
