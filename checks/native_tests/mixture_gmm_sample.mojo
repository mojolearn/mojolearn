# SPDX-License-Identifier: Apache-2.0
"""GaussianMixture.sample statistics from test_gmm_sample.py, on the actual entries.

Each fixture fits GaussianMixture(n_components=2, random_state=3) (full
covariance, kmeans init, tol 1e-3, reg_covar 1e-6, max_iter 100) to the
original blobs (recorded under fixtures/mixture_gmm_sample/) and draws with
the model's random_state as the key, as the public `sample` does:

  0  REPEAT   300 rows twice with key 3 are the same words (rows and
              components); key 4 gives other rows; key 2^64-1 draws finite rows
  1  MOMENTS  20000 rows: each component's count within 5 sd of n * weight,
              its sample mean within 6 standard errors of means_, its sample
              covariance (ddof 1) within 0.1 correlation units of covariances_
  2  WHITEN   4000 rows: (X - mean_j) P_j over every row has column means
              below 0.08 and population standard deviations within 0.06 of 1

The SHAPE arm (public dtypes and grouping) and the refusals stay in Python.
"""
from std.math import isfinite, sqrt
from std.memory import bitcast
from core.host_parallel import host_parallelize
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.hostptr import f32_ptr, i32_ptr
from core.mixture_native_api import GmmDriver, GmmFitRequest
from mixture.host.gmm_native_host import gmm_native_host_v1
from mixture.host.gmm_host_entries import gmm_sample_into
from mixture.estimator import GmmParams, gaussian_mixture_fit, gaussian_mixture_sample
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


comptime GMM_SAMPLE_FIXTURES = 3
comptime GMM_K = 2
comptime GMM_D = 3
comptime GMM_SEED = UInt64(3)


@fieldwise_init
struct GmmRun(Movable):
    """The fitted parameters and the draws of each requested key, flat:
    draw `s` is rows[s*n*d : (s+1)*n*d] and labels[s*n : (s+1)*n]."""
    var weights: List[Float32]
    var means: List[Float32]
    var covariances: List[Float32]
    var precisions: List[Float32]
    var rows: List[Float32]
    var labels: List[Int32]


trait GmmSampleCalls:
    @staticmethod
    def run(x: List[Float32], n_fit: Int, n_draw: Int, keys: List[UInt64]) raises -> GmmRun:
        """Fit to `x` (n_fit x 3), then draw `n_draw` rows for every key."""
        ...


def blobs(seed: Int, n: Int) raises -> List[Float32]:
    var count = n * GMM_D
    var result = List[Float32](unsafe_uninit_length=count)
    with open("checks/native_tests/fixtures/mixture_gmm_sample/blobs-seed" + String(seed) + "-" + String(n) + "x3.f32", "r") as file:
        var bytes = file.read_bytes()
        require(len(bytes) == count * 4, "GMM blob fixture extent changed")
        var bp = bytes.unsafe_ptr()
        var rp = result.unsafe_ptr()
        def word(index: Int) {imm bp, imm rp}:
            var at = index * 4
            var bits = UInt32(bp[at]) | (UInt32(bp[at + 1]) << 8) | (UInt32(bp[at + 2]) << 16) | (UInt32(bp[at + 3]) << 24)
            rp.unsafe_store(index, bitcast[DType.float32](bits))
        host_parallelize(word, count)
        _ = bytes^
    return result^


def repeat_case(run: GmmRun, n: Int) raises:
    var size = n * GMM_D
    var flags = List[Int](length=size, fill=0)
    var rp = run.rows.unsafe_ptr()
    var lp = run.labels.unsafe_ptr()
    var fp = flags.unsafe_ptr()
    def cell(index: Int) {imm rp, imm lp, imm fp, imm size, imm n}:
        var bad = 0
        if bitcast[DType.uint32](rp[index]) != bitcast[DType.uint32](rp[size + index]):
            bad |= 1  # key 3 twice: the same words
        if bitcast[DType.uint32](rp[index]) != bitcast[DType.uint32](rp[2 * size + index]):
            bad |= 2  # key 4: at least one word differs (tallied below)
        if not isfinite(rp[3 * size + index]):
            bad |= 4  # key 2^64-1: finite
        if index < n and lp[index] != lp[n + index]:
            bad |= 1
        fp.unsafe_store(index, bad)
    host_parallelize(cell, size)
    var repeated = True
    var other = False
    var finite = True
    for value in flags:  # tally of independent word verdicts
        repeated = repeated and (value & 1) == 0
        other = other or (value & 2) != 0
        finite = finite and (value & 4) == 0
    require(repeated, "two calls with one random_state give the same bits")
    require(other, "another random_state gives other rows")
    require(finite, "the largest random_state is a valid key")


def moments_case(run: GmmRun, n: Int) raises:
    # Cells (component, statistic): statistic 0 is the count, 1..3 the sums.
    var first = List[Float64](length=GMM_K * 4, fill=0)
    var rp = run.rows.unsafe_ptr()
    var lp = run.labels.unsafe_ptr()
    var sp = first.unsafe_ptr()
    def sums(cell: Int) {imm rp, imm lp, imm sp, imm n}:
        var j = cell // 4
        var s = cell % 4
        var value = Float64(0)
        for r in range(n):  # one ordered reduction over the drawn rows
            if lp[r] == Int32(j):
                value += Float64(1.0) if s == 0 else Float64(rp[r * GMM_D + s - 1])
        sp.unsafe_store(cell, value)
    host_parallelize(sums, GMM_K * 4)
    var second = List[Float64](length=GMM_K * GMM_D * GMM_D, fill=0)
    var cp = second.unsafe_ptr()
    def products(cell: Int) {imm rp, imm lp, imm sp, imm cp, imm n}:
        var j = cell // (GMM_D * GMM_D)
        var a = (cell // GMM_D) % GMM_D
        var b = cell % GMM_D
        var count = sp[j * 4]
        var ma = sp[j * 4 + 1 + a] / count
        var mb = sp[j * 4 + 1 + b] / count
        var value = Float64(0)
        for r in range(n):  # one ordered reduction over the drawn rows
            if lp[r] == Int32(j):
                value += (Float64(rp[r * GMM_D + a]) - ma) * (Float64(rp[r * GMM_D + b]) - mb)
        cp.unsafe_store(cell, value / (count - 1.0))  # np.cov: ddof 1
    host_parallelize(products, GMM_K * GMM_D * GMM_D)
    for j in range(GMM_K):  # two components, each checked against its own parameters
        var count = first[j * 4]
        var w = Float64(run.weights[j])
        var sd = sqrt(Float64(n) * w * (1.0 - w))
        require(abs(count - Float64(n) * w) < 5.0 * sd + 1.0,
            "component " + String(j) + ": count " + String(count) + " not within 5 sd of n * weight")
        for a in range(GMM_D):  # three features
            var covariance = Float64(run.covariances[j * 9 + a * 3 + a])
            var se = sqrt(covariance / count)
            var mean = first[j * 4 + 1 + a] / count
            require(abs(mean - Float64(run.means[j * GMM_D + a])) < 6.0 * se,
                "component " + String(j) + ": sample mean not within 6 standard errors of means_")
            for b in range(GMM_D):  # three features
                var cb = Float64(run.covariances[j * 9 + b * 3 + b])
                var rel = abs(second[j * 9 + a * 3 + b] - Float64(run.covariances[j * 9 + a * 3 + b])) / sqrt(covariance * cb)
                require(rel < 0.1, "component " + String(j) + ": sample covariance not within 0.1 correlation units")


def whiten_case(run: GmmRun, n: Int) raises:
    var z = List[Float64](length=n * GMM_D, fill=0)
    var rp = run.rows.unsafe_ptr()
    var lp = run.labels.unsafe_ptr()
    var mp = run.means.unsafe_ptr()
    var pp = run.precisions.unsafe_ptr()
    var zp = z.unsafe_ptr()
    def row(r: Int) {imm rp, imm lp, imm mp, imm pp, imm zp}:
        var j = Int(lp[r])
        for c in range(GMM_D):  # one row times P_j
            var value = Float64(0)
            for a in range(GMM_D):  # the row's three centered features
                value += (Float64(rp[r * GMM_D + a]) - Float64(mp[j * GMM_D + a])) * Float64(pp[j * 9 + a * 3 + c])
            zp.unsafe_store(r * GMM_D + c, value)
    host_parallelize(row, n)
    var moments = List[Float64](length=2 * GMM_D, fill=0)
    var op = moments.unsafe_ptr()
    def column(c: Int) {imm zp, imm op, imm n}:
        var sum = Float64(0)
        for r in range(n):  # one ordered reduction per column
            sum += zp[r * GMM_D + c]
        var mean = sum / Float64(n)
        var squares = Float64(0)
        for r in range(n):  # one ordered reduction per column
            var delta = zp[r * GMM_D + c] - mean
            squares += delta * delta
        op.unsafe_store(c, mean)
        op.unsafe_store(GMM_D + c, sqrt(squares / Float64(n)))  # np.std: ddof 0
    host_parallelize(column, GMM_D)
    for c in range(GMM_D):  # three whitened columns
        require(abs(moments[c]) < 0.08 and abs(moments[GMM_D + c] - 1.0) < 0.06,
            "(X - mean) P is not standard normal in column " + String(c))


def fixture[Calls: GmmSampleCalls](unit: Int) raises:
    if unit == 0:
        var keys: List[UInt64] = [GMM_SEED, GMM_SEED, UInt64(4), ~UInt64(0)]
        var run = Calls.run(blobs(1, 3000), 3000, 300, keys)
        repeat_case(run, 300)
    elif unit == 1:
        var keys: List[UInt64] = [GMM_SEED]
        var run = Calls.run(blobs(2, 6000), 6000, 20000, keys)
        moments_case(run, 20000)
    else:
        var keys: List[UInt64] = [GMM_SEED]
        var run = Calls.run(blobs(3, 3000), 3000, 4000, keys)
        whiten_case(run, 4000)


# ---- The host column (was mixture_gmm_sample_cpu_suite.mojo) ----
# GaussianMixture.sample statistics on the host fit and host sample entries
# the mixture host binding calls; the three fixtures run across host cores and
# each fixture's independent draws run across host cores too.
struct HostGmmSample(GmmSampleCalls):
    @staticmethod
    def run(x: List[Float32], n_fit: Int, n_draw: Int, keys: List[UInt64]) raises -> GmmRun:
        var weights = List[Float32](length=GMM_K, fill=0)
        var means = List[Float32](length=GMM_K * GMM_D, fill=0)
        var covariances = List[Float32](length=GMM_K * GMM_D * GMM_D, fill=0)
        var precisions = List[Float32](length=GMM_K * GMM_D * GMM_D, fill=0)
        var log_det = List[Float32](length=GMM_K, fill=0)
        var scalars = List[Float64](length=3, fill=0)
        var covariance_type = String("full")
        var init_params = String("kmeans")
        var driver = GmmDriver(gmm_native_host_v1)
        _ = driver.fit(GmmFitRequest(Int(x.unsafe_ptr()), Int(weights.unsafe_ptr()), Int(means.unsafe_ptr()),
            Int(covariances.unsafe_ptr()), Int(precisions.unsafe_ptr()), Int(log_det.unsafe_ptr()),
            Int(scalars.unsafe_ptr()), 0, Int(covariance_type.as_c_string_slice().unsafe_ptr()),
            Int(init_params.as_c_string_slice().unsafe_ptr()), n_fit, GMM_D, GMM_K, 100, 3, 1e-3, 1e-6))
        _ = covariance_type
        _ = init_params
        var draws = len(keys)
        var rows = List[Float32](length=draws * n_draw * GMM_D, fill=0)
        var labels = List[Int32](length=draws * n_draw, fill=-1)
        var failures = List[Int](length=draws, fill=0)
        var fp = failures.unsafe_ptr()
        var kp = keys.unsafe_ptr()
        var rp = rows.unsafe_ptr()
        var lp = labels.unsafe_ptr()
        var wp = weights.unsafe_ptr()
        var mp = means.unsafe_ptr()
        var pp = precisions.unsafe_ptr()
        def draw(s: Int) {imm fp, imm kp, imm rp, imm lp, imm wp, imm mp, imm pp, imm n_draw}:
            try:
                var w = List[Float32](length=GMM_K, fill=0)
                var m = List[Float32](length=GMM_K * GMM_D, fill=0)
                var p = List[Float32](length=GMM_K * GMM_D * GMM_D, fill=0)
                for i in range(GMM_K * GMM_D * GMM_D):  # this task's private copy of the 18 parameter words
                    p[i] = pp[i]
                    if i < GMM_K * GMM_D:
                        m[i] = mp[i]
                    if i < GMM_K:
                        w[i] = wp[i]
                gmm_sample_into(w, m, p, GMM_K, GMM_D, n_draw, kp[s],
                    f32_ptr(Int(rp) + 4 * s * n_draw * GMM_D),
                    i32_ptr(Int(lp) + 4 * s * n_draw))
            except error:
                fp.unsafe_store(s, 1)
                print("FAIL mixture_gmm_sample_cpu draw", s, String(error))
        host_parallelize(draw, draws)
        for value in failures:
            require(value == 0, "GMM host sample draw failed")
        _ = keys
        _ = x
        _ = log_det^
        _ = scalars^
        return GmmRun(weights^, means^, covariances^, precisions^, rows^, labels^)


def _run_cpu(mut report: Report) raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var cases = report.cases(GMM_SAMPLE_FIXTURES)
    def task(unit: Int) {imm cases}:
        try:
            fixture[HostGmmSample](unit)
        except error:
            cases.fail(unit, error)
    report.parallel(task, GMM_SAMPLE_FIXTURES)


# ---- The device column (was mixture_gmm_sample_gpu_suite.mojo) ----
# GaussianMixture.sample statistics on the device fit and sample entries.
#
# The three fixtures, and the draws inside each, run in order: they submit to
# the shared family DeviceContext, and concurrent submission from several host
# threads to one context is a separately recorded open failure. The EM and
# sample kernels keep their own parallelism; the statistics use host cores.
struct DeviceGmmSample(GmmSampleCalls):
    @staticmethod
    def run(x: List[Float32], n_fit: Int, n_draw: Int, keys: List[UInt64]) raises -> GmmRun:
        var params = GmmParams.default()
        params.n_components = GMM_K
        params.random_state = UInt64(3)
        var model = gaussian_mixture_fit(x, n_fit, GMM_D, params)
        var rows = List[Float32]()
        var labels = List[Int32]()
        for s in range(len(keys)):  # driver-loop: over the requested keys, ordered on one device
            var drawn = List[Int32](length=n_draw, fill=-1)
            var values = gaussian_mixture_sample(model, n_draw, keys[s], drawn)
            require(len(values) == n_draw * GMM_D and len(drawn) == n_draw, "GMM device sample extents")
            rows.extend(values^)
            labels.extend(drawn^)
        var result = GmmRun(model.weights.copy(), model.means.copy(), model.covariances.copy(),
                            model.precisions_cholesky.copy(), rows^, labels^)
        _ = model^
        return result^


def _run_gpu(mut report: Report) raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var cases = report.cases(GMM_SAMPLE_FIXTURES)
    for unit in range(GMM_SAMPLE_FIXTURES):
        try:
            fixture[DeviceGmmSample](unit)
        except error:
            cases.fail(unit, error)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if CPU_COLUMN_BUILD:
            _run_cpu(report)
        else:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("mixture_gmm_sample", cpu=True, gpu=True, serial="", run=run_cases)
