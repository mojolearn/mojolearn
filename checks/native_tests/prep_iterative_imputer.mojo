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

The supplied-estimator rounds (`estimator=Ridge(...)`) and the public refusals stay in the Python file: those rounds
call a public estimator object per feature. Every unit is independent and runs across the host cores; the two
posterior sets run one after the other, each with its twelve seeds across the host cores.
"""
from std.math import isnan, sqrt
from std.testing import TestSuite
from core.host_parallel import host_parallelize
from checks.native_tests.py2mojo_fixture_words import fixture_f64
from checks.native_tests.py2mojo_host_members import rows_close
from checks.native_tests.testsuite_cases import check_cases, check_cases_serial, fixture_f32, require
from checks.native_tests.bm_rest_iterimp_host import (
    HostIterRunner, PREP_ITERIMP_DIR, ii_request, ii_output, ii_transform, pos_inf, neg_inf,
)
from bindings.prep_encode_imputer_run import IMP_MEAN, IMP_MEDIAN
from x_prep.iterimp_run import iterimp_fit, II_NONE, II_ASCENDING, II_DESCENDING, II_RANDOM

comptime N = 400
comptime D = 5
comptime SEEDS = 12


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
    var label = String("IterativeImputer sample_posterior set ") + String(unit)
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
    def seed_fit(seed: Int) {imm dp, imm fp, imm mp, imm xp, imm cp, imm unit, imm m}:
        try:
            var lo = -0.5 if unit == 1 else neg_inf()
            var hi = 0.5 if unit == 1 else pos_inf()
            var request = ii_request(N, D, IMP_MEAN, II_ASCENDING, 10, II_NONE, UInt64(seed), lo, hi, True, False)
            var runner = HostIterRunner(Int(xp))
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


def test_iterative() raises:
    check_cases("iterative", 3, iterative_case)


def test_iterative_indicator() raises:
    check_cases("iterative_indicator", 1, indicator_case)


def test_iterative_random() raises:
    check_cases("iterative_random", 3, random_case)


def test_iterative_posterior() raises:
    # the two sets one after the other: each runs its twelve seeds across the host cores
    check_cases_serial("iterative_posterior", 2, posterior_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
