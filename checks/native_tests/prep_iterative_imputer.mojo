# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""IterativeImputer with the built-in Bayesian ridge against scikit-learn (lane bm-rest-iterimp, 2026-11-23; ported
from python/mojolearn/tests/test_x_prep_iterative.py `test_iterative` and the built-in-estimator parts of
`test_iterative_options`).

The typed RUN (x_prep/iterimp_run.mojo `iterimp_fit` / `iterimp_transform`) runs over the host column's x_prep
transport (checks/native_tests/bm_rest_iterimp_host.mojo `HostIterRunner`) with the request words the door's
UNPACK writes (bindings/prep_iterimp_native_door.mojo `_fit_request`). Inputs: the original's `_data(seed)` (400 x 5,
correlated, 15% NaN) for seeds 0 to 3, with scikit-learn 1.9.1's results, in `fixtures/prep_iterimp/`
(tools/py2mojo_native_fixtures_bm_rest_iterimp.py). Tolerance rtol 2e-3, atol 2e-3, as the original.

    iterative units 0-2   {}, {max_iter 3, descending, median}, {min_value -1, max_value 1}: fit_transform(X0),
                          n_iter_ and transform(X1)
    indicator             add_indicator: the same three checks on X2 / X3 (the width included)
    random units 0-2      {random, rs 3}, {n_nearest_features 3, rs 3}, {nnf 2, random, rs 5, max_iter 20}: the same
                          bits twice, and the mean distance to the reference on the missing cells under twice the
                          reference's own spread (another seed) + 0.02
    posterior units 0-1   sample_posterior, and with bounds [-0.5, 0.5]: over random_state 0..11 the draws' mean
                          within 0.05 and their mean per-cell spread within 20% of the reference's (and inside the
                          bounds)

    estimator units 0-2   (lane mf-iterimp) a supplied estimator as a native member, the rounds of
                          x_prep/iterimp_run.mojo `iterimp_member_rounds` after `iterimp_prepare`, on X2 with the
                          default keywords: Ridge(alpha=1), LinearRegression(), BayesianRidge() (linear and x_linear
                          members over the host entries). Against the door's words before the member route
                          (bindings/prep_iterimp_native_door.mojo's per-step estimator rounds, the base build;
                          tools/py2mojo_native_fixtures_mf_iterimp.py, pack `mf_iterimp`): Xt word for word,
                          n_iter_, the step count, and each step's feature, coef_ words and intercept_ (as float32).
                          Fail-proof: in `_member_step`, read the predict block at `a + 4 * go` instead of
                          `a + 4 * gm`; unit 0 fails naming an Xt word.

    estimator transform units 0-2   (lane mf-iterimp2) transform(X3) of the same three fits through
                          x_prep/iterimp_run.mojo `iterimp_member_transform` over each step's member state (the held
                          state the fit hands its estimator), against the door's per-step transform rounds over the
                          public fitted estimators before this lane (the base build;
                          tools/py2mojo_native_fixtures_mf_iterimp2.py, pack `mf_iterimp2`): word for word.
                          Fail-proof: in `iterimp_member_transform` run the steps from the last to the first; unit 0
                          fails naming an Xt word.
    estimator posterior units 0-1   (lane mf-iterimp2) sample_posterior with estimator=BayesianRidge() through the
                          member rounds (PREDICT_STD with the mean, core/xlinear_member.mojo), and with bounds
                          [-0.5, 0.5]: the posterior sets' statistical checks against scikit-learn's reference (whose
                          default estimator is BayesianRidge()). Fail-proof: in x_linear/bayes.mojo `bayes_factor_std_row`
                          return `fsqrt(s)` (no 1 / alpha_ term): the spread check fails.

Lane mf-iterimp3 (exact oracles, no fixture):
    single feature        X0 with columns 1-4 all NaN (dropped): one kept feature is scikit-learn's edge case, n_iter_
                          0, no step, Xt = the initial imputation (each missing cell the fitted fill, the observed
                          cells as given). Fail-proof: delete the `len(fitted.keep) < 2` branch of `iterimp_fit`;
                          n_iter_ is not 0.
    empty kept units 0-1  X2 with column 2 all NaN and keep_empty_features: scikit-learn marks it not missing, so it
                          is first in the ascending order (0 missing), counts in n_features_with_missing_ (skip_complete
                          False), and keeps its fill in every row; unit 0 the built-in ridge, unit 1 Ridge(alpha=1)
                          through the member rounds (its step fits on every row). Fail-proof: delete the
                          `iterimp_unmark_empty` call of `iterimp_member_rounds` (unit 1 fails on a column-2 word), or
                          the missing-count scatter of `iterimp_prepare` (the order check fails).
    class labels units 0-3  `iterimp_class_values` / `iterimp_class_codes`: a small target, a 300000-row target split
                          across the host cores, a continuous target (scikit-learn's words) and a NaN target.
                          Fail-proof: in `iterimp_class_codes` test `cp[mid] <= v`; units 0 and 1 fail on a code.
    classifier units 0-1  estimator=GaussianNB() (the naive Bayes member, a classifier kind) through the member rounds
                          on a 400 x 4 matrix whose every column is the row's label (0, 1, 2) with at most one NaN a
                          row: every imputed cell is its row's label, n_iter_ stops at 2; unit 1 puts 0.5 in column 0
                          and the fit refuses in scikit-learn's continuous-target words. Fail-proof: in
                          `_class_of_codes` write `cp[0]` for every row; unit 0 fails on an imputed cell.

Lane mf-iterimp3b (exact oracle, no fixture):
    logistic classifier   estimator=LogisticRegression() (C 1, intercept, lbfgs) as the logistic member (a classifier
                          kind) on the CPU image's QN entry (glm/host/logistic_native_host.mojo `qn_host_native_v1`):
                          the route an explicit HostLogisticRegression's member admission binds (its host-only word;
                          providers/python/member_logistic_boundary_door.mojo walks the class to its host context),
                          and LogisticRegression(device='cpu')'s. The classifier matrix above: every step has three
                          classes, every imputed cell is one of the labels 0, 1, 2 (the codes mapped back), and at
                          least 95% of them are their row's label. Fail-proof: in `_class_of_codes` write `cp[0]` for
                          every row; the 95% check fails.

The GaussianProcess member's PREDICT_STD route (the same query's mean in `call.y`) needs the GP provider's host route
and kernel words: it is in tools/mf_iterimp_digests.py, as are the public refusals. Every unit is independent and
runs across the host cores; the posterior sets run one after the other, each with its twelve seeds across the host
cores.
"""
from std.math import isnan, sqrt
from std.testing import TestSuite
from core.host_parallel import host_parallelize
from checks.native_tests.py2mojo_fixture_words import fixture_f64
from checks.native_tests.py2mojo_host_members import rows_close, same_bits
from checks.native_tests.testsuite_cases import check_cases, check_cases_serial, fixture_f32, require
from checks.native_tests.bm_rest_iterimp_host import (
    HostIterRunner, PREP_ITERIMP_DIR, ii_request, ii_output, ii_transform, pos_inf, neg_inf,
)
from bindings.prep_encode_imputer_run import IMP_MEAN, IMP_MEDIAN
from x_prep.iterimp_run import (
    iterimp_fit, iterimp_prepare, iterimp_member_rounds, iterimp_member_free, IterImpFitted, IterImpMember,
    IterImpMemberFit, II_NONE, II_ASCENDING, II_DESCENDING, II_RANDOM, IterImpHeldStep, IterImpApply,
    IterImpStep, iterimp_member_transform, iterimp_class_values, iterimp_class_codes, II_CONTINUOUS_WORDS,
)
from bindings.prep_encode_imputer_run import ImputerApply
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.device_route import DeviceRoute, DEVICE_CPU
from core.entry_table import EntryRef, entry_ref_word
from core.member_abi import (
    MemberPlan, MemberRef, MemberCall, MEMBER_F64, MEMBER_I32, MEMBER_OP_CAPABILITIES, MEMBER_OP_EXPORT,
    MEMBER_OP_PREDICT, MEMBER_BORROWS_FIT,
    MEMBER_BLOCK_COEF, MEMBER_BLOCK_SCALARS, member_call, member_call_of, member_kind_capabilities,
    member_output_dtype,
)
from core.member_families import (
    MEMBER_FAMILY_LINEAR, MEMBER_FAMILY_XLINEAR, MEMBER_FAMILY_NAIVE_BAYES, MEMBER_FAMILY_LOGISTIC,
)
from core.logistic_member import logistic_member_v2
from core.logistic_native_api import QnNativeDriver
from core.logistic_route import LogisticRoute, logistic_route_pack
from glm.host.logistic_native_host import qn_host_native_v1
from core.naive_bayes_member import naive_bayes_member_v2
from x_prep.transport_host import x_prep_transport_host_v1
from x_prep.transport_abi import XP_OP_COUNT
from core.linear_member import linear_member_v2
from core.xlinear_member import xlinear_member_v2
from glm.host.linear_fit_native import linear_fit_host_v1
from glm.host.linear_predict_native import linear_predict_host_v1
from x_linear.ridge_descriptor_host import ridge_fit_host_v1
from x_decomp.descriptor_host import x_decomp_host_v1
from x_linear.ops_entry_host import x_linear_ops_host_v1

comptime N = 400
comptime D = 5
comptime SEEDS = 12
comptime MF_ITERIMP_DIR = "checks/native_tests/fixtures/mf_iterimp/"
comptime MF_ITERIMP2_DIR = "checks/native_tests/fixtures/mf_iterimp2/"


def _x(seed: Int) raises -> List[Float32]:
    return fixture_f32(PREP_ITERIMP_DIR + "x" + String(seed) + ".f32", N * D)


def iterative_case(unit: Int) raises:
    var label = String("IterativeImputer set ") + String(unit)
    var strategy = IMP_MEDIAN if unit == 1 else IMP_MEAN
    var order = II_DESCENDING if unit == 1 else II_ASCENDING
    var max_iter = 3 if unit == 1 else 10
    var lo = -1.0 if unit == 2 else neg_inf()
    var hi = 1.0 if unit == 2 else pos_inf()
    var x = _x(0)
    var xh = _x(1)
    var request = ii_request(N, D, strategy, order, max_iter, II_NONE, 0, lo, hi, False, False)
    var runner = HostIterRunner(Int(x.unsafe_ptr()))
    var fitted = iterimp_fit(runner, request)
    var name = PREP_ITERIMP_DIR + "it-" + String(unit)
    rows_close(ii_output(runner, fitted, N), fixture_f64(name + "-ft.f64", N * D), 2e-3, 2e-3, label + ": fit_transform")
    var niter = Int(fixture_f64(name + "-niter.f64", 1)[0])
    require(fitted.n_iter == niter, label + ": n_iter_ " + String(fitted.n_iter) + " expected " + String(niter))
    rows_close(ii_transform(runner, fitted, request, xh, N), fixture_f64(name + "-tr.f64", N * D), 2e-3, 2e-3,
        label + ": transform")
    _ = len(x)  # X stays alive past every program that read it


def indicator_case(unit: Int) raises:
    var label = String("IterativeImputer add_indicator")
    var x = _x(2)
    var xh = _x(3)
    var width = Int(fixture_f64(PREP_ITERIMP_DIR + "ind-width.f64", 1)[0])
    var request = ii_request(N, D, IMP_MEAN, II_ASCENDING, 10, II_NONE, 0, neg_inf(), pos_inf(), False, True)
    var runner = HostIterRunner(Int(x.unsafe_ptr()))
    var fitted = iterimp_fit(runner, request)
    require(fitted.cols == width, label + ": width " + String(fitted.cols) + " expected " + String(width))
    rows_close(ii_output(runner, fitted, N), fixture_f64(PREP_ITERIMP_DIR + "ind-ft.f64", N * width), 2e-3, 2e-3,
        label + ": fit_transform")
    var niter = Int(fixture_f64(PREP_ITERIMP_DIR + "ind-niter.f64", 1)[0])
    require(fitted.n_iter == niter, label + ": n_iter_ " + String(fitted.n_iter) + " expected " + String(niter))
    rows_close(ii_transform(runner, fitted, request, xh, N), fixture_f64(PREP_ITERIMP_DIR + "ind-tr.f64", N * width),
        2e-3, 2e-3, label + ": transform")
    _ = len(x)
    _ = unit


def _random_fit(unit: Int, x: List[Float32]) raises -> List[Float32]:
    # {random, rs 3}, {n_nearest_features 3, rs 3}, {n_nearest_features 2, random, rs 5, max_iter 20}
    var order = II_ASCENDING if unit == 1 else II_RANDOM
    var nnf = II_NONE if unit == 0 else (3 if unit == 1 else 2)
    var seed = UInt64(5) if unit == 2 else UInt64(3)
    var request = ii_request(N, D, IMP_MEAN, order, 20 if unit == 2 else 10, nnf, seed, neg_inf(), pos_inf(), False,
                             False)
    var runner = HostIterRunner(Int(x.unsafe_ptr()))
    var fitted = iterimp_fit(runner, request)
    var values = ii_output(runner, fitted, N)
    _ = len(x)
    return values^


def random_case(unit: Int) raises:
    var label = String("IterativeImputer random set ") + String(unit)
    var x = _x(2)
    var a = _random_fit(unit, x)
    var again = _random_fit(unit, x)
    for i in range(len(a)):  # small-loop(oracle: one result block)
        require(a[i] == again[i] or (isnan(a[i]) and isnan(again[i])),
            label + ": word " + String(i) + " differs between two fits of one seed")
    var name = PREP_ITERIMP_DIR + "rnd-" + String(unit)
    var b = fixture_f64(name + "-b.f64", N * D)
    var b2 = fixture_f64(name + "-b2.f64", N * D)
    var dist = 0.0
    var spread = 0.0
    var count = 0
    for i in range(N * D):  # small-loop(oracle: the missing cells of one block)
        if isnan(x[i]):
            dist += abs(Float64(a[i]) - b[i])
            spread += abs(b[i] - b2[i])
            count += 1
    require(count > 0, label + ": no missing cell")
    dist /= Float64(count)
    spread /= Float64(count)
    require(dist < 2.0 * spread + 0.02, label + ": mean distance " + String(dist) + " spread " + String(spread))


def posterior_case(unit: Int) raises:
    _posterior_set(unit, False)


def estimator_posterior_case(unit: Int) raises:
    _posterior_set(unit, True)


def _posterior_set(unit: Int, with_member: Bool) raises:
    var label = String("IterativeImputer sample_posterior set ") + String(unit)
    if with_member:
        label += " (estimator=BayesianRidge(), member rounds)"
    var x = _x(2)
    var cells = List[Int]()
    for i in range(N * D):  # small-loop(oracle: the missing cells of one block)
        if isnan(x[i]):
            cells.append(i)
    var m = len(cells)
    var draws = List[Float32](length=SEEDS * m, fill=0.0)
    var failed = List[Int](length=SEEDS, fill=0)
    var messages = List[String](length=SEEDS, fill=String(""))
    var dp = draws.unsafe_ptr()
    var fp = failed.unsafe_ptr()
    var mp = messages.unsafe_ptr()
    var xp = x.unsafe_ptr()
    var cp = cells.unsafe_ptr()
    def seed_fit(seed: Int) {imm dp, imm fp, imm mp, imm xp, imm cp, imm unit, imm m, imm with_member}:
        try:
            var lo = -0.5 if unit == 1 else neg_inf()
            var hi = 0.5 if unit == 1 else pos_inf()
            var request = ii_request(N, D, IMP_MEAN, II_ASCENDING, 10, II_NONE, UInt64(seed), lo, hi, True, False)
            var runner = HostIterRunner(Int(xp))
            if with_member:
                var fitted = IterImpFitted()
                var plan = iterimp_prepare(runner, request, fitted)
                var member = IterImpMember(_estimator_plan(2), False, True)
                var fit = IterImpMemberFit()
                iterimp_member_rounds(runner, request, plan, fitted, member, fit)
                var src = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=runner.words(fit.out_handle))
                for c in range(m):  # small-loop(oracle: the missing cells of one fit)
                    dp.unsafe_store(seed * m + c, src[fit.out_offset + Int(cp[c])])
                iterimp_member_free(member, fit)
            else:
                var fitted = iterimp_fit(runner, request)
                var values = ii_output(runner, fitted, N)
                for c in range(m):  # small-loop(oracle: the missing cells of one fit)
                    dp.unsafe_store(seed * m + c, values[Int(cp[c])])
        except problem:
            mp[unsafe_offset=seed] = String(problem)
            fp.unsafe_store(seed, 1)
    host_parallelize(seed_fit, SEEDS)
    for s in range(SEEDS):  # driver-loop: over the seeds, after the parallel fits
        require(failed[s] == 0, label + " seed " + String(s) + ": " + messages[s])
    _ = len(x)  # X and the cell list stay alive past the parallel fits
    _ = len(cells)
    var stats = fixture_f64(PREP_ITERIMP_DIR + "post-" + String(unit) + "-stats.f64", 2)
    var total = 0.0
    var low = Float64(draws[0])
    var high = Float64(draws[0])
    for i in range(SEEDS * m):  # small-loop(oracle: every draw)
        total += Float64(draws[i])
        low = min(low, Float64(draws[i]))
        high = max(high, Float64(draws[i]))
    var mean = total / Float64(SEEDS * m)
    var spread = 0.0
    for c in range(m):  # small-loop(oracle: the per-cell spread over the seeds, population std)
        var cell_mean = 0.0
        for s in range(SEEDS):  # small-loop(oracle: the seeds)
            cell_mean += Float64(draws[s * m + c])
        cell_mean /= Float64(SEEDS)
        var var_sum = 0.0
        for s in range(SEEDS):  # small-loop(oracle: the seeds)
            var e = Float64(draws[s * m + c]) - cell_mean
            var_sum += e * e
        spread += sqrt(var_sum / Float64(SEEDS))
    spread /= Float64(m)
    if unit == 1:
        require(low >= -0.5 and high <= 0.5, label + ": draws outside the bounds " + String(low) + " " + String(high))
    require(abs(mean - stats[0]) < 0.05, label + ": mean " + String(mean) + " expected " + String(stats[0]))
    require(abs(spread / stats[1] - 1.0) < 0.2, label + ": spread " + String(spread) + " expected " + String(stats[1]))


def _estimator_plan(unit: Int) raises -> MemberPlan:
    """The member plan the door's UNPACK admits for unit 0 Ridge(alpha=1), 1 LinearRegression(), 2 BayesianRidge()
    on the host images: the family's member entry, its route over the host kernel entries the estimator's own doors
    read (linear: fit, predict, the Ridge solve, the decomposition; x_linear: the operations entry), the
    MEMBER_OP_NEW words of providers/python/member_linear_boundary_door.mojo / member_xlinear_boundary_door.mojo for
    the default keywords, and the family's capability mask."""
    var mode = Int(GLOBAL_NUMERIC_MODE)
    var plan = MemberPlan()
    var route = DeviceRoute(DEVICE_CPU)
    route.mode = mode
    route.vendor = 0
    if unit == 2:
        plan.family = MEMBER_FAMILY_XLINEAR
        plan.kind = 4
        route.entry = EntryRef(x_linear_ops_host_v1, entry_ref_word(mode, 0, 0))
        plan.member = MemberRef(EntryRef(xlinear_member_v2, 0), 0)
        # [kind 4, fit_intercept, max_iter 300, tol 1e-3, alpha_1, alpha_2, lambda_1, lambda_2, alpha_init None,
        #  lambda_init None]
        plan.params = [4.0, 1.0, 300.0, 1e-3, 1e-6, 1e-6, 1e-6, 1e-6, -1.0, -1.0]
    else:
        plan.family = MEMBER_FAMILY_LINEAR
        plan.kind = 1 if unit == 0 else 0
        route.entry = EntryRef(linear_fit_host_v1, entry_ref_word(mode, 0, 0))
        route.second = EntryRef(linear_predict_host_v1, entry_ref_word(mode, 0, 0))
        route.third = EntryRef(ridge_fit_host_v1, entry_ref_word(mode, 0, 0))
        route.fourth = EntryRef(x_decomp_host_v1, entry_ref_word(mode, 0, 0))
        route.has_second = True
        route.has_third = True
        route.has_fourth = True
        plan.member = MemberRef(EntryRef(linear_member_v2, 0), 0)
        # [kind (1 Ridge, 0 LinearRegression), fit_intercept, alpha]
        if unit == 0:
            plan.params = [1.0, 1.0, 1.0]
        else:
            plan.params = [0.0, 1.0, 0.0]
    plan.route = route
    var probe = MemberCall(route)
    plan.member.capabilities = member_call(plan.member, MEMBER_OP_CAPABILITIES, probe)
    return plan^


def _exported(plan: MemberPlan, state: Int, block: Int) raises -> List[Float64]:
    """A copy of fitted block `block` of `state` (float32 COEF words widened, or float64 SCALARS words)."""
    var call = member_call_of(plan, state)
    call.block = block
    call.flags = 0
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var view = call.dst
    var count = view.rows * view.cols
    var out = List[Float64](length=count, fill=0.0)
    for i in range(count):  # small-loop(oracle: one fitted block of a few words)
        if view.dtype == MEMBER_F64:
            out[i] = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=view.data)[i]
        else:
            out[i] = Float64(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=view.data)[i])
    return out^


def estimator_case(unit: Int) raises:
    var names: List[String] = ["Ridge(alpha=1)", "LinearRegression()", "BayesianRidge()"]
    var label = String("IterativeImputer estimator=") + names[unit]
    var x = fixture_f32(MF_ITERIMP_DIR + "x.f32", N * D)
    var request = ii_request(N, D, IMP_MEAN, II_ASCENDING, 10, II_NONE, 0, neg_inf(), pos_inf(), False, False)
    var runner = HostIterRunner(Int(x.unsafe_ptr()))
    var fitted = IterImpFitted()
    var plan = iterimp_prepare(runner, request, fitted)
    require(len(fitted.keep) == D and len(fitted.empty) == 0, label + ": every column is kept and observed")
    var member = IterImpMember(_estimator_plan(unit), False, False)
    var fit = IterImpMemberFit()
    iterimp_member_rounds(runner, request, plan, fitted, member, fit)
    try:
        var name = MF_ITERIMP_DIR + "est-" + String(unit)
        var meta = fixture_f64(name + "-meta.f64", 2)
        require(fit.n_iter == Int(meta[0]), label + ": n_iter_ " + String(fit.n_iter) + " expected " + String(Int(meta[0])))
        var steps = Int(meta[1])
        require(len(fit.states) == steps and len(fit.seq_j) == steps,
            label + ": " + String(len(fit.states)) + " steps, expected " + String(steps))
        require(fit.cols == D, label + ": width " + String(fit.cols))
        var got = List[Float32](length=N * D, fill=0.0)
        var src = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=runner.words(fit.out_handle))
        for i in range(N * D):  # small-loop(oracle: one result block)
            got[i] = src[fit.out_offset + i]
        same_bits(got, fixture_f32(name + "-xt.f32", N * D), label + ": fit_transform")
        var js = fixture_f64(name + "-j.f64", steps)
        var coef = fixture_f32(name + "-coef.f32", steps * (D - 1))
        var inter = fixture_f32(name + "-inter.f32", steps)
        for s in range(steps):  # small-loop(oracle: the fitted steps)
            require(fit.seq_j[s] == Int(js[s]), label + ": step " + String(s) + " feature " + String(fit.seq_j[s]))
            require(len(fit.seq_nbl[s]) == D - 1, label + ": step " + String(s) + " neighbours")
            var c = _exported(member.plan, fit.states[s], MEMBER_BLOCK_COEF)
            require(len(c) == D - 1, label + ": step " + String(s) + " coef_ has " + String(len(c)) + " words")
            for k in range(D - 1):  # small-loop(oracle: one step's coefficients)
                require(Float32(c[k]) == coef[s * (D - 1) + k], label + ": step " + String(s) + " coef_[" + String(k)
                    + "] " + String(c[k]) + " expected " + String(coef[s * (D - 1) + k]))
            var scalars = _exported(member.plan, fit.states[s], MEMBER_BLOCK_SCALARS)
            require(Float32(scalars[0]) == inter[s], label + ": step " + String(s) + " intercept_ " + String(scalars[0])
                + " expected " + String(inter[s]))
    finally:
        iterimp_member_free(member, fit)
    _ = len(x)  # X stays alive past every program that read it


def estimator_transform_case(unit: Int) raises:
    var names: List[String] = ["Ridge(alpha=1)", "LinearRegression()", "BayesianRidge()"]
    var label = String("IterativeImputer estimator=") + names[unit] + " transform"
    var x = fixture_f32(MF_ITERIMP2_DIR + "x.f32", N * D)
    var xh = fixture_f32(MF_ITERIMP2_DIR + "xh.f32", N * D)
    var request = ii_request(N, D, IMP_MEAN, II_ASCENDING, 10, II_NONE, 0, neg_inf(), pos_inf(), False, False)
    var runner = HostIterRunner(Int(x.unsafe_ptr()))
    var fitted = IterImpFitted()
    var plan = iterimp_prepare(runner, request, fitted)
    var member = IterImpMember(_estimator_plan(unit), False, False)
    var fit = IterImpMemberFit()
    iterimp_member_rounds(runner, request, plan, fitted, member, fit)
    try:
        # the door's UNPACK of transform: each fitted step's held state (here the fit's own states), the initial
        # imputer's fill, the fitted fields
        var steps = List[IterImpHeldStep]()
        for s in range(len(fit.states)):  # small-loop(oracle: the fitted steps)
            steps.append(IterImpHeldStep(fit.seq_j[s], fit.seq_nbl[s].copy(), member.plan.copy(), fit.states[s]))
        var init = ImputerApply(N, D, 0, request.init.mark, request.init.missing, 1, len(fitted.fill),
                                fitted.keep.copy(), List[Int]())
        var apply = IterImpApply(init^, Int(fitted.fill.unsafe_ptr()), N, D, request.mark, request.missing,
                                 request.work_on, False, fitted.keep.copy(), fitted.bounds.copy(),
                                 fitted.indicator.copy(), False, 0, 0, List[IterImpStep]())
        var rng = fit.rng
        var other = HostIterRunner(Int(xh.unsafe_ptr()))
        var applied = iterimp_member_transform(other, apply, steps, rng)
        require(applied.cols == D, label + ": width " + String(applied.cols))
        var got = List[Float32](length=N * D, fill=0.0)
        var src = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=other.words(applied.out_handle))
        for i in range(N * D):  # small-loop(oracle: one result block)
            got[i] = src[applied.out_offset + i]
        same_bits(got, fixture_f32(MF_ITERIMP2_DIR + "est-" + String(unit) + "-tr.f32", N * D), label)
    finally:
        iterimp_member_free(member, fit)
    _ = len(x)  # X, Xh and the fill stay alive past every program that read them
    _ = len(xh)
    _ = len(fitted.fill)


def _nan() -> Float32:
    return Float32(0.0) / Float32(0.0)


def single_feature_case(unit: Int) raises:
    var label = String("IterativeImputer one kept feature")
    var x = _x(0)
    for i in range(N):  # small-loop(fixture: drop columns 1-4)
        for j in range(1, D):  # small-loop(fixture: the dropped columns)
            x[i * D + j] = _nan()
    var request = ii_request(N, D, IMP_MEAN, II_ASCENDING, 10, II_NONE, 0, neg_inf(), pos_inf(), False, False)
    var runner = HostIterRunner(Int(x.unsafe_ptr()))
    var fitted = iterimp_fit(runner, request)
    require(len(fitted.keep) == 1 and fitted.keep[0] == 0, label + ": kept " + String(len(fitted.keep)))
    require(fitted.n_iter == 0, label + ": n_iter_ " + String(fitted.n_iter) + " expected 0")
    require(len(fitted.seq_j) == 0, label + ": " + String(len(fitted.seq_j)) + " steps, expected none")
    require(fitted.cols == 1, label + ": width " + String(fitted.cols))
    var got = ii_output(runner, fitted, N)
    for i in range(N):  # small-loop(oracle: the one kept column)
        var want = fitted.fill[0] if isnan(x[i * D]) else x[i * D]
        require(got[i] == want, label + ": row " + String(i) + " " + String(got[i]) + " expected " + String(want))
    _ = len(x)  # X stays alive past every program that read it
    _ = unit


def empty_kept_case(unit: Int) raises:
    var label = String("IterativeImputer keep_empty_features, an all-NaN column, ")
    if unit == 0:
        label += "built-in ridge"
    else:
        label += "estimator=Ridge(alpha=1)"
    var x = _x(2)
    for i in range(N):  # small-loop(fixture: column 2 has no observed value)
        x[i * D + 2] = _nan()
    var request = ii_request(N, D, IMP_MEAN, II_ASCENDING, 10, II_NONE, 0, neg_inf(), pos_inf(), False, False)
    request.init.keep_empty = True
    var runner = HostIterRunner(Int(x.unsafe_ptr()))
    var fitted = IterImpFitted()
    var got = List[Float32](length=N * D, fill=0.0)
    if unit == 0:
        fitted = iterimp_fit(runner, request)
        var values = ii_output(runner, fitted, N)
        for i in range(N * D):  # small-loop(oracle: one result block)
            got[i] = values[i]
    else:
        var plan = iterimp_prepare(runner, request, fitted)
        require(len(plan.orders) > 0 and len(plan.orders[0]) == D and plan.orders[0][0] == 2,
            label + ": the empty column (no missing value) is not first in the ascending order")
        var member = IterImpMember(_estimator_plan(0), False, False)
        var fit = IterImpMemberFit()
        iterimp_member_rounds(runner, request, plan, fitted, member, fit)
        try:
            require(fit.cols == D, label + ": width " + String(fit.cols))
            var src = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=runner.words(fit.out_handle))
            for i in range(N * D):  # small-loop(oracle: one result block)
                got[i] = src[fit.out_offset + i]
        finally:
            iterimp_member_free(member, fit)
    require(len(fitted.keep) == D and len(fitted.empty) == 1 and fitted.empty[0] == 2,
        label + ": the empty kept positions")
    require(fitted.n_missing == D, label + ": n_features_with_missing_ " + String(fitted.n_missing) + " expected " + String(D))
    for i in range(N):  # small-loop(oracle: every row)
        require(got[i * D + 2] == fitted.fill[2], label + ": row " + String(i) + " column 2 " + String(got[i * D + 2])
            + " expected its fill " + String(fitted.fill[2]))
        for j in range(D):  # small-loop(oracle: one row)
            if j != 2 and not isnan(x[i * D + j]):
                require(got[i * D + j] == x[i * D + j], label + ": observed cell " + String(i) + ", " + String(j))
            require(not isnan(got[i * D + j]), label + ": cell " + String(i) + ", " + String(j) + " not imputed")
    _ = len(x)  # X stays alive past every program that read it


def class_labels_case(unit: Int) raises:
    var label = String("IterativeImputer classifier labels unit ") + String(unit)
    var y = List[Float32]()
    if unit == 0:
        for v in range(5):  # small-loop(fixture: the target 2, 0, 2, 1, 0)
            y.append(Float32(2 if v == 0 or v == 2 else (1 if v == 3 else 0)))
    elif unit == 1:
        for i in range(300000):  # small-loop(fixture: a target split across the host cores)
            y.append(Float32((i * 7919) % 11 - 5))
    elif unit == 2:
        y.append(Float32(0.0))
        y.append(Float32(1.5))
        y.append(Float32(1.0))
    else:
        y.append(Float32(1.0))
        y.append(_nan())
        y.append(Float32(0.0))
    var m = len(y)
    if unit >= 2:
        var raised = String("")
        try:
            _ = iterimp_class_values(Int(y.unsafe_ptr()), m)
        except problem:
            raised = String(problem)
        var want = String("Input y contains NaN.")
        if unit == 2:
            want = String(II_CONTINUOUS_WORDS)
        require(raised == want, label + ": raised '" + raised + "' expected '" + want + "'")
        _ = len(y)
        return
    var classes = iterimp_class_values(Int(y.unsafe_ptr()), m)
    var codes = iterimp_class_codes(Int(y.unsafe_ptr()), m, classes)
    var k = 3 if unit == 0 else 11
    require(len(classes) == k, label + ": " + String(len(classes)) + " classes, expected " + String(k))
    for c in range(k):  # small-loop(oracle: the ascending classes)
        var want = Float32(c) if unit == 0 else Float32(c - 5)
        require(classes[c] == want, label + ": class " + String(c) + " " + String(classes[c]))
    for i in range(m):  # small-loop(oracle: every code)
        var want = Int32(Int(y[i])) if unit == 0 else Int32(Int(y[i]) + 5)
        require(codes[i] == want, label + ": code " + String(i) + " " + String(codes[i]) + " expected " + String(want))
    _ = len(y)  # the target stays alive past the reads


def _gnb_plan() raises -> MemberPlan:
    """GaussianNB() as the naive Bayes member (kind 0) over the host x_prep transport: var_smoothing 1e-9, no
    blocked statistics, the R3 work region off, priors None (core/naive_bayes_member.mojo NEW words)."""
    var mode = Int(GLOBAL_NUMERIC_MODE)
    var route = DeviceRoute(DEVICE_CPU)
    route.mode = mode
    route.vendor = 0
    route.entry = EntryRef(x_prep_transport_host_v1, entry_ref_word(mode, 0, XP_OP_COUNT))
    route.flags = 0
    var plan = MemberPlan()
    plan.route = route
    plan.family = MEMBER_FAMILY_NAIVE_BAYES
    plan.kind = 0
    plan.member = MemberRef(EntryRef(naive_bayes_member_v2, 0), 0)
    var probe = MemberCall(plan.route)
    plan.member.capabilities = member_call(plan.member, MEMBER_OP_CAPABILITIES, probe)
    plan.params = [0.0, 1e-9, 0.0, 0.0, 0.0]
    return plan^


def classifier_case(unit: Int) raises:
    comptime C = 4
    var label = String("IterativeImputer estimator=GaussianNB() unit ") + String(unit)
    var x = List[Float32](length=N * C, fill=0.0)
    for i in range(N):  # small-loop(fixture: every column the row's label, at most one NaN a row)
        for j in range(C):  # small-loop(fixture: one row)
            x[i * C + j] = _nan() if (i + j) % 6 == 0 else Float32(i % 3)
    if unit == 1:
        x[1 * C + 0] = 0.5  # row 1 is observed in column 0: a continuous target for its step
    var request = ii_request(N, C, IMP_MEAN, II_ASCENDING, 10, II_NONE, 0, neg_inf(), pos_inf(), False, False)
    var runner = HostIterRunner(Int(x.unsafe_ptr()))
    var fitted = IterImpFitted()
    var plan = iterimp_prepare(runner, request, fitted)
    var gnb = _gnb_plan()
    var mask = member_kind_capabilities(gnb)
    var two = gnb.copy()
    two.classes = 2
    var dtype = member_output_dtype(two, MEMBER_OP_PREDICT, C - 1, 1, MEMBER_I32)
    var member = IterImpMember(gnb^, (mask & MEMBER_BORROWS_FIT) != 0, False)
    member.classifier = True
    member.code_dtype = dtype
    var fit = IterImpMemberFit()
    if unit == 1:
        var raised = String("")
        try:
            iterimp_member_rounds(runner, request, plan, fitted, member, fit)
        except problem:
            raised = String(problem)
        iterimp_member_free(member, fit)
        require(raised == String(II_CONTINUOUS_WORDS), label + ": raised '" + raised + "'")
        _ = len(x)
        return
    iterimp_member_rounds(runner, request, plan, fitted, member, fit)
    try:
        require(fit.n_iter == 2, label + ": n_iter_ " + String(fit.n_iter) + " expected 2")
        require(len(fit.classes) == len(fit.states), label + ": the steps' classes")
        for s in range(len(fit.classes)):  # small-loop(oracle: the fitted steps)
            require(len(fit.classes[s]) == 3, label + ": step " + String(s) + " has " + String(len(fit.classes[s]))
                + " classes")
        var src = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=runner.words(fit.out_handle))
        for i in range(N * C):  # small-loop(oracle: one result block)
            var want = Float32((i // C) % 3)
            require(src[fit.out_offset + i] == want, label + ": cell " + String(i) + " "
                + String(src[fit.out_offset + i]) + " expected " + String(want))
    finally:
        iterimp_member_free(member, fit)
    _ = len(x)  # X stays alive past every program that read it


def _logistic_plan() raises -> MemberPlan:
    """LogisticRegression() as the logistic member (kind 0) on the CPU image's QN entry: the NEW words of
    providers/python/member_logistic_boundary_door.mojo for C 1 (l1 0, l2 1), fit_intercept, max_iter 100,
    linesearch 20, lbfgs memory 10, no normalized penalty, tol 1e-4 (change tolerance 1e-6)."""
    var plan = MemberPlan()
    plan.family = MEMBER_FAMILY_LOGISTIC
    plan.kind = 0
    plan.route = logistic_route_pack(LogisticRoute(DEVICE_CPU, Int(GLOBAL_NUMERIC_MODE), 0, False, False,
                                                   QnNativeDriver(qn_host_native_v1)))
    plan.member = MemberRef(EntryRef(logistic_member_v2, 0), 0)
    var probe = MemberCall(plan.route)
    plan.member.capabilities = member_call(plan.member, MEMBER_OP_CAPABILITIES, probe)
    plan.params = [0.0, 1.0, 0.0, 1.0, 100.0, 20.0, 10.0, 0.0, 1e-4, 1e-6]
    return plan^


def logistic_classifier_case(unit: Int) raises:
    comptime C = 4
    var label = String("IterativeImputer estimator=LogisticRegression() on the host QN entry")
    var x = List[Float32](length=N * C, fill=0.0)
    var missing = 0
    for i in range(N):  # small-loop(fixture: every column the row's label, at most one NaN a row)
        for j in range(C):  # small-loop(fixture: one row)
            if (i + j) % 6 == 0:
                x[i * C + j] = _nan()
                missing += 1
            else:
                x[i * C + j] = Float32(i % 3)
    var request = ii_request(N, C, IMP_MEAN, II_ASCENDING, 10, II_NONE, 0, neg_inf(), pos_inf(), False, False)
    var runner = HostIterRunner(Int(x.unsafe_ptr()))
    var fitted = IterImpFitted()
    var plan = iterimp_prepare(runner, request, fitted)
    var logistic = _logistic_plan()
    var mask = member_kind_capabilities(logistic)
    var two = logistic.copy()
    two.classes = 2
    var dtype = member_output_dtype(two, MEMBER_OP_PREDICT, C - 1, 1, MEMBER_I32)
    var member = IterImpMember(logistic^, (mask & MEMBER_BORROWS_FIT) != 0, False)
    member.classifier = True
    member.code_dtype = dtype
    var fit = IterImpMemberFit()
    iterimp_member_rounds(runner, request, plan, fitted, member, fit)
    try:
        require(fit.n_iter >= 1 and fit.n_iter <= 10, label + ": n_iter_ " + String(fit.n_iter))
        require(len(fit.classes) == len(fit.states) and len(fit.states) > 0, label + ": the steps' classes")
        for s in range(len(fit.classes)):  # small-loop(oracle: the fitted steps)
            require(len(fit.classes[s]) == 3, label + ": step " + String(s) + " has " + String(len(fit.classes[s]))
                + " classes")
        var src = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=runner.words(fit.out_handle))
        var right = 0
        for i in range(N):  # small-loop(oracle: one result block)
            for j in range(C):  # small-loop(oracle: one row)
                var got = src[fit.out_offset + i * C + j]
                var want = Float32(i % 3)
                if not isnan(x[i * C + j]):
                    require(got == want, label + ": observed cell " + String(i) + ", " + String(j))
                    continue
                require(got == 0.0 or got == 1.0 or got == 2.0, label + ": imputed cell " + String(i) + ", "
                    + String(j) + " " + String(got) + " is not a class label")
                if got == want:
                    right += 1
        require(right * 100 >= missing * 95, label + ": " + String(right) + " of " + String(missing)
            + " imputed cells are their row's label")
    finally:
        iterimp_member_free(member, fit)
    _ = len(x)  # X stays alive past every program that read it
    _ = unit


def test_iterative() raises:
    check_cases("iterative", 3, iterative_case)


def test_iterative_single_feature() raises:
    check_cases("iterative_single_feature", 1, single_feature_case)


def test_iterative_empty_kept() raises:
    check_cases("iterative_empty_kept", 2, empty_kept_case)


def test_iterative_class_labels() raises:
    check_cases("iterative_class_labels", 4, class_labels_case)


def test_iterative_classifier() raises:
    check_cases("iterative_classifier", 2, classifier_case)


def test_iterative_logistic_classifier() raises:
    check_cases("iterative_logistic_classifier", 1, logistic_classifier_case)


def test_iterative_indicator() raises:
    check_cases("iterative_indicator", 1, indicator_case)


def test_iterative_random() raises:
    check_cases("iterative_random", 3, random_case)


def test_iterative_posterior() raises:
    # the two sets one after the other: each runs its twelve seeds across the host cores
    check_cases_serial("iterative_posterior", 2, posterior_case)


def test_iterative_estimator() raises:
    check_cases("iterative_estimator", 3, estimator_case)


def test_iterative_estimator_transform() raises:
    check_cases("iterative_estimator_transform", 3, estimator_transform_case)


def test_iterative_estimator_posterior() raises:
    # the two sets one after the other: each runs its twelve seeds across the host cores
    check_cases_serial("iterative_estimator_posterior", 2, estimator_posterior_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
