# SPDX-License-Identifier: Apache-2.0
"""Native fixtures for the two typed mixture transports the mixture door
calls (lane thin-unpack-mixture-cluster): `mixture_native_api_v1`
(core/mixture_native_api.mojo over mixture/host/gmm_native_host.mojo) and
`bgmm_native_api_v1` (core/bgmm_native_api.mojo over
x_cluster/bgmm_native_host.mojo), on the CPU providers.

Each fixture is independent and runs on its own worker. The data are two
tight, far-apart groups of four points, so every fit has an analytic
answer for its labels: the first four rows share one label, the last four
the other.
"""
from std.math import isfinite
from core.host_parallel import host_parallelize
from core.kmeans_variants_api import VariantsSpan
from core.native_descriptor_fixed_call import native_descriptor_call_fixed
from core.mixture_native_api import (
    GmmDriver,
    GmmFitRequest,
    GmmScoreRequest,
    MIXTURE_NATIVE_MODE,
    MIXTURE_NATIVE_CAPABILITIES,
    MIXTURE_NATIVE_FIT,
    MIXTURE_NATIVE_PREDICT,
    MIXTURE_NATIVE_SCORE_SAMPLES,
    MIXTURE_NATIVE_SCORE_BIC_AIC,
)
from core.bgmm_native_api import BgmmDriver, BgmmFitRequest, BgmmResultSlot, BGMM_NATIVE_FIT
from mixture.host.gmm_native_host import gmm_native_host_v1
from x_cluster.bgmm_native_host import bgmm_native_host_v1
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


# ---- The host column (was mixture_native_cpu_suite.mojo) ----
comptime MIXTURE_NATIVE_CASES = 6
comptime ROWS = 8
comptime FEATURES = 2
comptime K = 2


def _data() -> List[Float32]:
    var x: List[Float32] = [0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1,
                            10.0, 10.0, 10.1, 10.0, 10.0, 10.1, 10.1, 10.1]
    return x^


def _two_groups(labels: List[Int32]) raises:
    require(len(labels) == ROWS, "label count")
    for r in range(1, 4):  # fixed fixture rows of the first group
        require(labels[r] == labels[0], "first group shares one label")
    for r in range(5, ROWS):  # fixed fixture rows of the second group
        require(labels[r] == labels[4], "second group shares one label")
    require(labels[0] != labels[4], "the two groups differ")


def _gmm_fit(driver: GmmDriver, x: List[Float32], mut weights: List[Float32], mut means: List[Float32],
             mut covariances: List[Float32], mut precisions: List[Float32], mut log_det: List[Float32],
             mut scalars: List[Float64], var covariance_type: String) raises -> Int:
    var init_params = String("kmeans")
    var request = GmmFitRequest(Int(x.unsafe_ptr()), Int(weights.unsafe_ptr()), Int(means.unsafe_ptr()),
        Int(covariances.unsafe_ptr()), Int(precisions.unsafe_ptr()), Int(log_det.unsafe_ptr()),
        Int(scalars.unsafe_ptr()), 0, Int(covariance_type.as_c_string_slice().unsafe_ptr()),
        Int(init_params.as_c_string_slice().unsafe_ptr()), ROWS, FEATURES, K, 100, 0, 1e-3, 1e-6)
    var n_iter = driver.fit(request)
    _ = covariance_type
    _ = init_params
    return n_iter


def fixture(unit: Int) raises:
    var x = _data()
    if unit <= 3:
        var driver = GmmDriver(gmm_native_host_v1)
        if unit == 0:
            # Metadata: the binary's mode word and no fit_predict on the host.
            var mode = driver.metadata(MIXTURE_NATIVE_MODE)
            require(mode == 0 or mode == 1, "mode word")
            require(driver.metadata(MIXTURE_NATIVE_CAPABILITIES) == 0, "host has no fit_predict")
            return
        var weights = List[Float32](length=K, fill=0)
        var means = List[Float32](length=K * FEATURES, fill=0)
        var covariances = List[Float32](length=K * FEATURES * FEATURES, fill=0)
        var precisions = List[Float32](length=K * FEATURES * FEATURES, fill=0)
        var log_det = List[Float32](length=K, fill=0)
        var scalars = List[Float64](length=3, fill=0)
        if unit == 1:
            # The by-name refusal crosses the descriptor with its text.
            var refused = False
            try:
                _ = _gmm_fit(driver, x, weights, means, covariances, precisions, log_det, scalars, "tied")
            except problem:
                refused = String(problem).find("covariance_type='tied' is NOT IMPLEMENTED") >= 0
            require(refused, "tied refusal text")
            return
        var n_iter = _gmm_fit(driver, x, weights, means, covariances, precisions, log_det, scalars, "full")
        require(n_iter >= 1 and Int(scalars[0]) == n_iter, "n_iter word")
        require(abs(Float64(weights[0]) + Float64(weights[1]) - 1.0) < 1e-5, "weights sum to one")
        var labels = List[Int32](length=ROWS, fill=-1)
        var score = List[Float32](length=ROWS, fill=0)
        var bic_aic = List[Float64](length=3, fill=0)
        var operation = MIXTURE_NATIVE_PREDICT if unit == 2 else MIXTURE_NATIVE_SCORE_SAMPLES
        var destination = Int(labels.unsafe_ptr()) if unit == 2 else Int(score.unsafe_ptr())
        driver.score(operation, GmmScoreRequest(Int(weights.unsafe_ptr()), Int(means.unsafe_ptr()),
            Int(covariances.unsafe_ptr()), Int(precisions.unsafe_ptr()), Int(log_det.unsafe_ptr()),
            Int(x.unsafe_ptr()), destination, K, FEATURES, n_iter, scalars[1] != 0, ROWS, scalars[2]))
        if unit == 2:
            _two_groups(labels)
        else:
            for r in range(ROWS):  # fixed fixture rows
                require(isfinite(score[r]), "finite log likelihood")
            driver.score(MIXTURE_NATIVE_SCORE_BIC_AIC, GmmScoreRequest(Int(weights.unsafe_ptr()),
                Int(means.unsafe_ptr()), Int(covariances.unsafe_ptr()), Int(precisions.unsafe_ptr()),
                Int(log_det.unsafe_ptr()), Int(x.unsafe_ptr()), Int(bic_aic.unsafe_ptr()), K, FEATURES, n_iter,
                scalars[1] != 0, ROWS, scalars[2]))
            require(isfinite(bic_aic[0]) and isfinite(bic_aic[1]) and isfinite(bic_aic[2]), "score, bic, aic")
        _ = weights
        _ = means
        _ = covariances
        _ = precisions
        _ = log_det
        _ = scalars
        _ = labels
        _ = score
        _ = bic_aic
        _ = x
        return
    var driver = BgmmDriver(bgmm_native_host_v1)
    if unit == 4:
        # A fit request with a missing result slot is refused, not run.
        var a: Array[Int, 3] = [Int(x.unsafe_ptr()), 0, 0]
        var p: Array[Int, 18] = [ROWS * FEATURES, 0, ROWS, FEATURES, K, 1, 100, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        var f: Array[Float64, 6] = [-1, -1, -1, 1e-6, 1e-3, 0]
        var refused = False
        try:
            _ = native_descriptor_call_fixed(driver.entry, BGMM_NATIVE_FIT, a, p, f)
        except problem:
            refused = String(problem).find("invalid request") >= 0
        require(refused, "missing BGMM result slot was accepted")
        _ = x
        return
    # BayesianGaussianMixture (dirichlet_process, kmeans init), then its
    # score on the fitted block: the fit's labels and the score's argmax
    # agree on the two groups.
    var ints: List[Int] = [ROWS, FEATURES, K, 1, 100, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    var floats: List[Float64] = [-1.0, -1.0, -1.0, 1e-6, 1e-3, 0.0]
    var fit = BgmmResultSlot()
    driver.fit(BgmmFitRequest(VariantsSpan(Int(x.unsafe_ptr()), len(x)), VariantsSpan(0, 0), ints^, floats^), fit)
    require(len(fit.storage[0].f) >= 16 and len(fit.storage[0].i) == 1 and len(fit.storage[0].s) >= 3,
            "BGMM result slots")
    require(len(fit.storage[0].f[0]) == K and len(fit.storage[0].f[1]) == K * FEATURES, "weights and means")
    _two_groups(fit.storage[0].i[0])
    var aux = List[Float32]()
    for block in [1, 3, 8]:  # the three model blocks of the score (means, Cholesky, constants)
        for t in range(len(fit.storage[0].f[block])):  # fixture block words
            aux.append(fit.storage[0].f[block][t])
    var scored = BgmmResultSlot()
    driver.score(VariantsSpan(Int(x.unsafe_ptr()), len(x)), VariantsSpan(Int(aux.unsafe_ptr()), len(aux)),
                 ROWS, FEATURES, K, 3, scored)
    _two_groups(scored.storage[0].i[0])
    require(len(scored.storage[0].f[1]) == ROWS, "score_samples length")
    _ = aux
    _ = x
    _ = fit
    _ = scored


def _run_cpu(mut report: Report) raises:
    var cases = report.cases(MIXTURE_NATIVE_CASES)
    def run(unit: Int) {imm cases}:
        try:
            fixture(unit)
        except problem:
            cases.fail(unit, problem)
    report.parallel(run, MIXTURE_NATIVE_CASES)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if CPU_COLUMN_BUILD:
            _run_cpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("mixture_native", cpu=True, gpu=False, serial="", run=run_cases)
