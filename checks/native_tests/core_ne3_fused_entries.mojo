# SPDX-License-Identifier: Apache-2.0
"""The fused whole-step entries of lane ne3-g-svc-gp-neighbors against the
per-op chains and reference relations (ported from
`python/mojolearn/tests/test_ne3_fused_entries.py`; the six ported functions
are deleted there, the SVC score and empty-query boundary tests stay).

    units 0-3   OneClassSVM (rbf, linear, poly, sigmoid; nu 0.3, gamma
                'scale', degree 3, coef0 0): the fused `ocsvm_labels`
                (`oc_fused_labels`, the public predict) equals the sign of
                the decision function (`oc_kernel_product` then
                `oc_decision`, the public decision_function), `+1` where
                the decision is > 0, else `-1`, on the 37 queries and on
                the training rows (fit_predict)
    units 4-5   NearestCentroid uniform priors (euclidean, manhattan): the
                fused `nc_labels` (mode 0 / 1) is the argmin of the float64
                squared-L2 / L1 distance to the fitted centroids
    unit 6      NearestCentroid empirical priors, non-uniform class counts:
                the fused `nc_labels` (mode 2) is the argmax of the decision
                scores (`NcQueryPlan.decision`); the fused `nc_proba` (mode
                `nc_proba_mode`: float64 under the NC_PROBA64 capability) is
                the float64 softmax of the scores (rtol 2e-5, atol 2e-6),
                its log form (mode 1) their log (rtol 2e-4, atol 2e-5), and
                a repeated fused proba gives the same bits
    unit 7      homogeneity_completeness_v_measure, beta 1, 0.5, 2: the one
                fused entry equals the three single entries bit for bit;
                a perfect labelling equals them too and is (1, 1, 1) within
                atol 1e-12
    units 8-9   LabelPropagation / LabelSpreading rbf (gamma 0.5, every
                fourth row unlabelled): the fused `lp_dense_proba` rows sum
                to 1 within 1e-5 and its code output is their argmax
    units 10-11 LabelPropagation / LabelSpreading knn (n_neighbors 5,
                max_iter 200, every second row unlabelled): two fits give
                the same label_distributions_ bits and n_iter_, and
                transduction_ is the argmax of label_distributions_

The fits are the sequences the doors run (`bindings/svc_ocsvm_native_door.mojo`
`_ocsvm_fit_native`, `bindings/neighbors_local_native_door.mojo` `_nc_fit`,
`bindings/neighbors_semi_native_door.mojo` `_lp_fit`) over the typed plans and
requests they call (`core/ocsvm_plan.mojo`, `core/nearest_centroid_plan.mojo`,
`neighbors/neighbors_semi_run.mojo`), at the constructor defaults. The data are
the original shapes (`_blobs`: 96 rows, 5 features, class `i % 3` shifted by
2 * class; 37 queries shifted by 1.5; 200 labels in [0, 4) and [0, 6)) drawn
from a SplitMix64 Box-Muller stream regenerated here instead of NumPy's PCG64:
every assertion is a relation that holds for any such data.
"""
from std.math import exp, log, sqrt, cos
from std.memory import bitcast
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call
from core.neighbors_native_driver import XnDriver
from core.ocsvm_plan import (
    OcWeights, OcFitState, oc_f32_address, oc_i32_address, oc_take_rows, oc_kernel, oc_variance,
    oc_kernel_product, oc_fused_labels, oc_decision,
)
from core.nearest_centroid_plan import NcFitPlan, NcQueryPlan, NcLabelCodes
from neighbors.neighbors_local_run import nc_proba_mode
from neighbors.neighbors_graph_run import XnOpRequest, xn_f32_word
from neighbors.neighbors_semi_run import (
    xn_semi_knn_k, xn_semi_knn_sq_request, xn_lp_knn_graph_request, xn_semi_rbf_request,
    xn_semi_row_normalize_request, xn_semi_row_argmax_request, xn_lp_col_degree_request,
    xn_lp_laplacian_deg_request, xn_lp_static_weight, xn_lp_labels_request, xn_lp_clamp_request, xn_lp_mode,
    xn_lp_iterate_knn_request, xn_lp_iterate_request, xn_lp_iterations, xn_lp_dense_proba_request,
)
# TOMBSTONE: MOJOLEARN_C43_RESIDENT_NORMALIZATION (unmeasured opt-in arm, owner expdel2-05-base; non-owner sites) deleted 2026-11-15 by lane/expdel2-10-cluster; code recoverable at f379abdf05.
# Restore: git apply experiments/removed/MOJOLEARN_C43_RESIDENT_NORMALIZATION.expdel2-10-cluster.patch; record in docs/TOMBSTONES.md.
from x_neighbors.descriptor_schema import xn_operation_id
from x_neighbors.native_descriptor_host import xn_native_v1
from x_neighbors.native_descriptor_device import xn_native_v1 as xn_native_v1_gpu
from metrics.mojolearn_metrics_host_run import homogeneity_completeness_v_measure_run
from metrics.host.metrics_oracle import host_homogeneity_score, host_v_measure
from metrics.estimator import (
    homogeneity_score_host, completeness_score_host, v_measure_score_host,
    homogeneity_completeness_v_measure_host,
)
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


comptime NE3_UNITS = 12
# `_expansion_neighbors._KERNELS`
comptime KERNEL_LINEAR = 0
comptime KERNEL_POLY = 1
comptime KERNEL_RBF = 2
comptime KERNEL_SIGMOID = 3


# ------------------------------------------------------------------ data


def _next(mut state: UInt64) -> UInt64:
    """SplitMix64."""
    state += UInt64(0x9E3779B97F4A7C15)
    var z = state
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def _unit(mut state: UInt64) -> Float64:
    return Float64(Int(_next(state) >> 11)) * (1.0 / 9007199254740992.0)


def _normal(mut state: UInt64) -> Float64:
    var u1 = _unit(state) + (1.0 / 9007199254740992.0)
    var u2 = _unit(state)
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2)


struct Blobs(Movable):
    """`_blobs(seed)`: row-major x (n, d), class codes `i % k`, queries (nq, d)."""
    var x: List[Float32]
    var y: List[Int32]
    var q: List[Float32]
    var n: Int
    var d: Int
    var k: Int
    var nq: Int

    def __init__(out self, seed: Int, n: Int = 96, d: Int = 5, k: Int = 3):
        self.n = n
        self.d = d
        self.k = k
        self.nq = 37
        var state = UInt64(seed) * UInt64(0x2545F4914F6CDD1D) + UInt64(1)
        self.y = List[Int32](length=n, fill=0)
        self.x = List[Float32](length=n * d, fill=0)
        self.q = List[Float32](length=self.nq * d, fill=0)
        for i in range(n):  # small-loop(one fixture's sequential draw stream, inside one unit's task)
            self.y[i] = Int32(i % k)
            for j in range(d):  # small-loop(features of one row)
                self.x[i * d + j] = Float32(_normal(state) + 2.0 * Float64(i % k))
        for i in range(self.nq * d):  # small-loop(one fixture's query draws)
            self.q[i] = Float32(_normal(state) + 1.5)


def _argmax_f32(values: List[Float32], row: Int, width: Int) -> Int:
    """`np.argmax` of one row: the first maximum."""
    var best = 0
    for c in range(1, width):  # small-loop(classes of one row)
        if values[row * width + c] > values[row * width + best]:
            best = c
    return best


def _close(got: Float64, expected: Float64, rtol: Float64, atol: Float64) -> Bool:
    """`np.testing.assert_allclose`'s element rule."""
    return abs(got - expected) <= atol + rtol * abs(expected)


def _capabilities(driver: XnDriver) raises -> Int:
    var a: Array[Int, 0] = []
    var p: Array[Int, 0] = []
    var f: Array[Float64, 0] = []
    return driver.run("capabilities", a, p, f)


def _op(driver: XnDriver, request: XnOpRequest) raises:
    """`_semi_op`: one x_neighbors request through the native descriptor."""
    _ = native_descriptor_call(driver.entry, xn_operation_id(request.name), request.a, request.p, request.f)


# ------------------------------------------------------------- OneClassSVM


def _ocsvm_signs(driver: XnDriver, query: Int, rows: Int, supports: Int, coefficients: Int, ns: Int, d: Int,
                 kernel: Int, gamma: Float64, rho: Float64, what: String) raises:
    var scores = List[Float32](length=rows, fill=0)
    var decision = List[Float32](length=rows, fill=0)
    var labels = List[Int32](length=rows, fill=7)
    oc_kernel_product(driver, query, supports, coefficients, Int(scores.unsafe_ptr()), rows, ns, d, kernel, 3,
                      gamma, 0.0)
    oc_decision(driver, Int(scores.unsafe_ptr()), Int(decision.unsafe_ptr()), rows, rho)
    oc_fused_labels(driver, query, supports, coefficients, Int(labels.unsafe_ptr()), rows, ns, d, kernel, 3,
                    gamma, 0.0, rho)
    _ = scores
    var bad = 0
    for r in range(rows):  # small-loop(rows of one fixture's query block)
        var expected = Int32(1) if decision[r] > Float32(0) else Int32(-1)
        if labels[r] != expected:
            bad += 1
    require(bad == 0, "OneClassSVM kernel " + String(kernel) + " " + what + ": " + String(bad)
            + " fused labels differ from the sign of decision_function")
    _ = decision^
    _ = labels^


def ocsvm_unit(driver: XnDriver, kernel: Int) raises:
    var b = Blobs(11)
    var n = b.n
    var d = b.d
    var xp = Int(b.x.unsafe_ptr())
    var variance = oc_variance(driver, xp, n * d)
    var gamma = Float64(Float32(1.0 / (Float64(d) * variance) if variance != 0 else 1.0))
    var gram = oc_kernel(driver, xp, xp, n, n, d, kernel, 3, gamma, 0.0)
    var weights = OcWeights()
    weights.unit(n)
    var fitted = OcFitState(n)
    fitted.initialize(driver, oc_f32_address(weights.values), 0.3)
    fitted.prepare_metadata()
    fitted.solve_gram(driver, gram.address(), oc_f32_address(weights.values), 10000000, Float64(Float32(0.001)))
    var rho = Float64(fitted.info[0])
    fitted.compact_support(driver)
    var ns = fitted.supports
    require(ns > 0, "OneClassSVM kernel " + String(kernel) + " fitted no support vectors")
    var supports = oc_take_rows(driver, xp, n, d, oc_i32_address(fitted.support_rows), ns)
    var coefficients = oc_f32_address(fitted.support_values)
    _ocsvm_signs(driver, Int(b.q.unsafe_ptr()), b.nq, supports.address(), coefficients, ns, d, kernel, gamma, rho,
                 "predict")
    _ocsvm_signs(driver, xp, n, supports.address(), coefficients, ns, d, kernel, gamma, rho, "fit_predict")
    _ = b^
    _ = gram^
    _ = weights^
    _ = fitted^
    _ = supports^


# ---------------------------------------------------------- NearestCentroid


@fieldwise_init
struct NcFitted(Movable):
    var centroids: List[Float32]
    var stats: List[Float32]
    var prior: List[Float32]


def _nc_fit(driver: XnDriver, b: Blobs, euclidean: Bool, empirical: Bool) raises -> NcFitted:
    """`_nc_fit` at shrink_threshold None (the MOJOLEARN_NC_SPLIT_OPS default off)."""
    var n = b.n
    var d = b.d
    var c = b.k
    var plan = NcFitPlan(n, d, c)
    plan.count(driver, Int(b.y.unsafe_ptr()))
    var prior = List[Float32](length=c, fill=0)
    for k in range(c):  # small-loop(classes): class_prior_ narrowed as nc_narrow_priors narrows it
        prior[k] = Float32(Float64(plan.wide_counts[k]) / Float64(n)) if empirical else Float32(1.0 / Float64(c))
    plan.split = False
    plan.euclidean = euclidean
    var stats = List[Float32](length=d, fill=0)
    var centroids = List[Float32](length=c * d, fill=0)
    var deviations = List[Float32](length=c * d, fill=0)
    require(plan.statistics(driver, Int(b.x.unsafe_ptr()), Int(b.y.unsafe_ptr()), Int(stats.unsafe_ptr())) == 0,
            "NearestCentroid finite fit status")
    var summary = plan.median_std(driver, Int(stats.unsafe_ptr()))
    require(not summary[1], "NearestCentroid fixture has varying features")
    plan.shrink(driver, Int(b.x.unsafe_ptr()), Int(stats.unsafe_ptr()), Int(centroids.unsafe_ptr()),
                Int(deviations.unsafe_ptr()), summary[0], 0.0)
    _ = plan^
    _ = deviations^
    _ = b
    return NcFitted(centroids^, stats^, prior^)


def nc_uniform_unit(driver: XnDriver, euclidean: Bool) raises:
    require((_capabilities(driver) & 32) != 0, "the x_neighbors binary carries nc_labels")
    var b = Blobs(12)
    var fitted = _nc_fit(driver, b, euclidean, False)
    var qa = Int(b.q.unsafe_ptr())
    var request = NcQueryPlan(qa, Int(fitted.centroids.unsafe_ptr()), qa, qa, b.nq, b.d, b.k)
    var codes = NcLabelCodes(b.nq)
    require(request.fused(driver, "nc_labels", codes.address(), 0 if euclidean else 1) == 0,
            "NearestCentroid fused labels finite status")
    var bad = 0
    for r in range(b.nq):  # small-loop(queries of one fixture)
        var best = 0
        var best_distance = Float64(0)
        for k in range(b.k):  # small-loop(centroids)
            var total = Float64(0)
            for j in range(b.d):  # small-loop(features)
                var diff = Float64(b.q[r * b.d + j]) - Float64(fitted.centroids[k * b.d + j])
                total += diff * diff if euclidean else abs(diff)
            if k == 0 or total < best_distance:
                best = k
                best_distance = total
        if codes.values[r] != Int32(best):
            bad += 1
    require(bad == 0, "NearestCentroid " + ("euclidean" if euclidean else "manhattan") + ": " + String(bad)
            + " fused labels are not the nearest centroid")
    _ = b^
    _ = fitted^
    _ = codes^


def nc_empirical_unit(driver: XnDriver) raises:
    var capabilities = _capabilities(driver)
    require((capabilities & 32) != 0, "the x_neighbors binary carries nc_labels and nc_proba")
    var b = Blobs(13)
    for i in range(b.n):  # small-loop(rows): non-uniform class counts
        if i % 5 == 0:
            b.y[i] = 0
    var fitted = _nc_fit(driver, b, True, True)
    var nq = b.nq
    var c = b.k
    var request = NcQueryPlan(Int(b.q.unsafe_ptr()), Int(fitted.centroids.unsafe_ptr()),
                              Int(fitted.stats.unsafe_ptr()), Int(fitted.prior.unsafe_ptr()), nq, b.d, c)
    var scores = List[Float32](length=nq * c, fill=0)
    var status = List[Int32](length=1, fill=0)
    request.decision(driver, Int(scores.unsafe_ptr()), Int(status.unsafe_ptr()))
    require(status[0] == 0, "NearestCentroid decision finite status")
    var codes = NcLabelCodes(nq)
    require(request.fused(driver, "nc_labels", codes.address(), 2) == 0, "NearestCentroid fused labels status")
    var mode = nc_proba_mode(False, (capabilities & 128) != 0)
    var wide = List[Float64](length=nq * c, fill=-1.0)
    var narrow = List[Float32](length=nq * c, fill=-1.0)
    var again64 = List[Float64](length=nq * c, fill=-2.0)
    var again32 = List[Float32](length=nq * c, fill=-2.0)
    var log_proba = List[Float32](length=nq * c, fill=1.0)
    var dst = Int(wide.unsafe_ptr()) if mode == 2 else Int(narrow.unsafe_ptr())
    var dst_again = Int(again64.unsafe_ptr()) if mode == 2 else Int(again32.unsafe_ptr())
    require(request.fused(driver, "nc_proba", dst, mode) == 0, "NearestCentroid fused proba status")
    require(request.fused(driver, "nc_proba", dst_again, mode) == 0, "NearestCentroid repeated proba status")
    require(request.fused(driver, "nc_proba", Int(log_proba.unsafe_ptr()), 1) == 0,
            "NearestCentroid fused log proba status")
    var moved = 0
    var far = 0
    var far_log = 0
    var repeat = 0
    for r in range(nq):  # small-loop(queries of one fixture)
        if codes.values[r] != Int32(_argmax_f32(scores, r, c)):
            moved += 1
        var top = Float64(scores[r * c])
        for k in range(1, c):  # small-loop(classes)
            top = max(top, Float64(scores[r * c + k]))
        var total = Float64(0)
        for k in range(c):  # small-loop(classes)
            total += exp(Float64(scores[r * c + k]) - top)
        for k in range(c):  # small-loop(classes)
            var i = r * c + k
            var expected = exp(Float64(scores[i]) - top) / total
            var got = wide[i] if mode == 2 else Float64(narrow[i])
            if not _close(got, expected, 2e-5, 2e-6):
                far += 1
            if not _close(Float64(log_proba[i]), log(expected), 2e-4, 2e-5):
                far_log += 1
            if mode == 2:
                if bitcast[DType.uint64](wide[i]) != bitcast[DType.uint64](again64[i]):
                    repeat += 1
            elif bitcast[DType.uint32](narrow[i]) != bitcast[DType.uint32](again32[i]):
                repeat += 1
    require(moved == 0, String(moved) + " NearestCentroid fused labels are not the argmax of decision_function")
    require(far == 0, String(far) + " NearestCentroid fused probabilities differ from the softmax of the scores")
    require(far_log == 0, String(far_log) + " NearestCentroid fused log probabilities differ from log(softmax)")
    require(repeat == 0, String(repeat) + " NearestCentroid repeated fused probabilities changed bits")
    _ = b^
    _ = fitted^
    _ = scores^
    _ = status^
    _ = codes^
    _ = wide^
    _ = narrow^
    _ = again64^
    _ = again32^
    _ = log_proba^


# ---------------------------------------- homogeneity / completeness / V


trait HcvCalls:
    @staticmethod
    def fused(a: List[Int32], b: List[Int32], n: Int, upper: Int32, beta: Float64) raises -> Tuple[
            Float64, Float64, Float64]:
        ...

    @staticmethod
    def homogeneity(a: List[Int32], b: List[Int32], n: Int, upper: Int32) raises -> Float64:
        ...

    @staticmethod
    def completeness(a: List[Int32], b: List[Int32], n: Int, upper: Int32) raises -> Float64:
        ...

    @staticmethod
    def v_measure(a: List[Int32], b: List[Int32], n: Int, upper: Int32, beta: Float64) raises -> Float64:
        ...


struct HostHcv(HcvCalls):
    """The host binding's four entries (`bindings/_mojolearn_metrics_host.mojo`)."""
    @staticmethod
    def fused(a: List[Int32], b: List[Int32], n: Int, upper: Int32, beta: Float64) raises -> Tuple[
            Float64, Float64, Float64]:
        return homogeneity_completeness_v_measure_run(a, b, n, 0, upper, beta)

    @staticmethod
    def homogeneity(a: List[Int32], b: List[Int32], n: Int, upper: Int32) raises -> Float64:
        return host_homogeneity_score(a, b, n, 0, upper)

    @staticmethod
    def completeness(a: List[Int32], b: List[Int32], n: Int, upper: Int32) raises -> Float64:
        return host_homogeneity_score(b, a, n, 0, upper)

    @staticmethod
    def v_measure(a: List[Int32], b: List[Int32], n: Int, upper: Int32, beta: Float64) raises -> Float64:
        return host_v_measure(a, b, n, 0, upper, beta)


struct DeviceHcv(HcvCalls):
    """The device binding's four entries (`bindings/_mojolearn_metrics.mojo`)."""
    @staticmethod
    def fused(a: List[Int32], b: List[Int32], n: Int, upper: Int32, beta: Float64) raises -> Tuple[
            Float64, Float64, Float64]:
        var h = Float64(0)
        var c = Float64(0)
        var v = Float64(0)
        homogeneity_completeness_v_measure_host(a, b, n, 0, upper, beta, h, c, v)
        return (h, c, v)

    @staticmethod
    def homogeneity(a: List[Int32], b: List[Int32], n: Int, upper: Int32) raises -> Float64:
        return homogeneity_score_host(a, b, n, 0, upper)

    @staticmethod
    def completeness(a: List[Int32], b: List[Int32], n: Int, upper: Int32) raises -> Float64:
        return completeness_score_host(a, b, n, 0, upper)

    @staticmethod
    def v_measure(a: List[Int32], b: List[Int32], n: Int, upper: Int32, beta: Float64) raises -> Float64:
        return v_measure_score_host(a, b, n, 0, upper, beta)


def _same64(a: Float64, b: Float64) -> Bool:
    return bitcast[DType.uint64](a) == bitcast[DType.uint64](b)


def hcv_unit[H: HcvCalls]() raises:
    # `_cluster_codes` maps both arrays onto 0..k-1 over the sorted union of
    # their labels. Every value of [0, 4) and [0, 6) is planted, so the codes
    # are the labels and upper = 5 (3 for the pair (a, a)).
    comptime N = 200
    var state = UInt64(15) * UInt64(0x2545F4914F6CDD1D) + UInt64(1)
    var a = List[Int32](length=N, fill=0)
    var b = List[Int32](length=N, fill=0)
    for i in range(N):  # small-loop(one fixture's sequential draw stream)
        a[i] = Int32(i) if i < 4 else Int32(Int(_next(state) % UInt64(4)))
        b[i] = Int32(i) if i < 6 else Int32(Int(_next(state) % UInt64(6)))
    var betas: List[Float64] = [1.0, 0.5, 2.0]
    for beta in betas:  # small-loop(the three betas of the original test)
        var hcv = H.fused(a, b, N, 5, beta)
        require(_same64(hcv[0], H.homogeneity(a, b, N, 5)), "fused homogeneity differs from homogeneity_score")
        require(_same64(hcv[1], H.completeness(a, b, N, 5)), "fused completeness differs from completeness_score")
        require(_same64(hcv[2], H.v_measure(a, b, N, 5, beta)), "fused V-measure differs from v_measure_score")
    var same = H.fused(a, a, N, 3, 1.0)
    require(_same64(same[0], H.homogeneity(a, a, N, 3)) and _same64(same[1], H.completeness(a, a, N, 3))
            and _same64(same[2], H.v_measure(a, a, N, 3, 1.0)), "perfect labelling: fused differs from the three")
    require(abs(same[0] - 1.0) <= 1e-12 and abs(same[1] - 1.0) <= 1e-12 and abs(same[2] - 1.0) <= 1e-12,
            "a perfect labelling scores (1, 1, 1)")
    _ = a^
    _ = b^


# ---------------------------------------------- LabelPropagation/Spreading


@fieldwise_init
struct LpFitted(Movable):
    var distributions: List[Float32]
    var transduction: List[Int32]
    var n_iter: Int


def _lp_fit(driver: XnDriver, b: Blobs, spreading: Bool, knn: Bool, gamma: Float64, n_neighbors: Int,
            max_iter: Int) raises -> LpFitted:
    """`_lp_fit` over the typed requests; the unlabelled marker -1 is the
    first encoded class, so the codes are y + 1 with marker 0."""
    var n = b.n
    var d = b.d
    var nc = b.k
    var xa = Int(b.x.unsafe_ptr())
    var codes = List[Int32](length=n, fill=0)
    for i in range(n):  # small-loop(rows): the encoded labels
        codes[i] = b.y[i] + 1
    var ld = List[Float32](length=n * nc, fill=0)
    var ys = List[Float32](length=n * nc, fill=0)
    var unlabeled = List[Int32](length=n, fill=0)
    var weight = xn_lp_static_weight(0.2) if spreading else 1.0
    _op(driver, xn_lp_labels_request(Int(codes.unsafe_ptr()), Int(ld.unsafe_ptr()), Int(ys.unsafe_ptr()),
                                     Int(unlabeled.unsafe_ptr()), n, nc, 0, weight))
    _ = codes
    var status = List[Int32](length=1, fill=0)
    var k = xn_semi_knn_k(n_neighbors, n)
    var dist = List[Float32]()
    var idx = List[Int32]()
    var cols = List[Int32]()
    var vals = List[Float32]()
    var affinity = List[Float32]()
    var dense = List[Float32]()
    var degree = List[Float32]()
    # TOMBSTONE: MOJOLEARN_C43_RESIDENT_NORMALIZATION (unmeasured opt-in arm, owner expdel2-05-base; non-owner sites) deleted 2026-11-15 by lane/expdel2-10-cluster; code recoverable at f379abdf05.
    # Restore: git apply experiments/removed/MOJOLEARN_C43_RESIDENT_NORMALIZATION.expdel2-10-cluster.patch; record in docs/TOMBSTONES.md.
    if knn:
        dist = List[Float32](length=n * k, fill=0)
        idx = List[Int32](length=n * k, fill=0)
        _op(driver, xn_semi_knn_sq_request(False, xa, xa, Int(dist.unsafe_ptr()), Int(idx.unsafe_ptr()), n, n, d, k,
                                           Int(status.unsafe_ptr())))
        cols = List[Int32](length=n * k, fill=0)
        vals = List[Float32](length=n * k, fill=0)
        _op(driver, xn_lp_knn_graph_request(Int(idx.unsafe_ptr()), Int(cols.unsafe_ptr()), Int(vals.unsafe_ptr()),
                                            n, n, k, 1 if spreading else 0))
    else:
        affinity = List[Float32](length=n * n, fill=0)
        _op(driver, xn_semi_rbf_request(xa, xa, Int(affinity.unsafe_ptr()), n, n, d, KERNEL_RBF,
                                        Int(status.unsafe_ptr()), gamma))
        if not spreading:
            dense = List[Float32](length=n * n, fill=0)
            _op(driver, xn_semi_row_normalize_request(Int(affinity.unsafe_ptr()), Int(dense.unsafe_ptr()), n, n))
        else:
            degree = List[Float32](length=n, fill=0)
            dense = List[Float32](length=n * n, fill=0)
            _op(driver, xn_lp_col_degree_request(Int(affinity.unsafe_ptr()), Int(degree.unsafe_ptr()), n))
            _op(driver, xn_lp_laplacian_deg_request(Int(affinity.unsafe_ptr()), Int(degree.unsafe_ptr()),
                                                    Int(dense.unsafe_ptr()), n))
    require(status[0] == 0, "LabelPropagation finite X status")
    var ystatic = ys.copy()
    if not spreading:
        _op(driver, xn_lp_clamp_request(Int(ld.unsafe_ptr()), Int(unlabeled.unsafe_ptr()), Int(ystatic.unsafe_ptr()),
                                        n, nc))
    var info = List[Int32](length=2, fill=0)
    var alpha = xn_f32_word(0.2) if spreading else 0.0
    if knn:
        _op(driver, xn_lp_iterate_knn_request(Int(cols.unsafe_ptr()), Int(vals.unsafe_ptr()), Int(ld.unsafe_ptr()),
                                              Int(ystatic.unsafe_ptr()), Int(unlabeled.unsafe_ptr()),
                                              Int(info.unsafe_ptr()), n, k, nc, max_iter, spreading, 0.001, alpha))
    else:
        _op(driver, xn_lp_iterate_request(Int(dense.unsafe_ptr()), Int(ld.unsafe_ptr()), Int(ystatic.unsafe_ptr()),
                                          Int(unlabeled.unsafe_ptr()), Int(info.unsafe_ptr()), n, nc, max_iter,
                                          xn_lp_mode(spreading, False), 0.001, alpha))
    var n_iter = xn_lp_iterations(Int(info.unsafe_ptr()))
    var final = List[Float32](length=n * nc, fill=0)
    _op(driver, xn_semi_row_normalize_request(Int(ld.unsafe_ptr()), Int(final.unsafe_ptr()), n, nc))
    var transduction = List[Int32](length=n, fill=-1)
    _op(driver, xn_semi_row_argmax_request(Int(final.unsafe_ptr()), Int(transduction.unsafe_ptr()), n, nc))
    _ = b
    _ = ld^
    _ = ys^
    _ = unlabeled^
    _ = status^
    _ = dist^
    _ = idx^
    _ = cols^
    _ = vals^
    _ = affinity^
    _ = dense^
    _ = degree^
    _ = ystatic^
    _ = info^
    return LpFitted(final^, transduction^, n_iter)


def lp_rbf_unit(driver: XnDriver, spreading: Bool) raises:
    require((_capabilities(driver) & 64) != 0, "the x_neighbors binary carries lp_dense_proba")
    var b = Blobs(18)
    for i in range(0, b.n, 4):  # small-loop(rows): every class keeps labelled rows
        b.y[i] = -1
    var max_iter = 30 if spreading else 1000
    var fitted = _lp_fit(driver, b, spreading, False, 0.5, 7, max_iter)
    var nq = b.nq
    var c = b.k
    var proba = List[Float32](length=nq * c, fill=-1.0)
    var codes = List[Int32](length=nq, fill=-1)
    var placeholder_codes = List[Int32](length=1, fill=0)
    var placeholder_proba = List[Float32](length=1, fill=0)
    var status = List[Int32](length=1, fill=0)
    var qa = Int(b.q.unsafe_ptr())
    var xa = Int(b.x.unsafe_ptr())
    var la = Int(fitted.distributions.unsafe_ptr())
    _op(driver, xn_lp_dense_proba_request(qa, xa, la, Int(proba.unsafe_ptr()), Int(placeholder_codes.unsafe_ptr()),
                                          nq, b.n, b.d, c, KERNEL_RBF, False, Int(status.unsafe_ptr()), 0.5))
    _op(driver, xn_lp_dense_proba_request(qa, xa, la, Int(placeholder_proba.unsafe_ptr()), Int(codes.unsafe_ptr()),
                                          nq, b.n, b.d, c, KERNEL_RBF, True, Int(status.unsafe_ptr()), 0.5))
    require(status[0] == 0, "LabelPropagation finite query status")
    var unnormalized = 0
    var moved = 0
    for r in range(nq):  # small-loop(queries of one fixture)
        var total = Float64(0)
        for k in range(c):  # small-loop(classes)
            total += Float64(proba[r * c + k])
        if abs(total - 1.0) > 1e-5:
            unnormalized += 1
        if codes[r] != Int32(_argmax_f32(proba, r, c)):
            moved += 1
    var tag = String("LabelSpreading") if spreading else String("LabelPropagation")
    require(unnormalized == 0, tag + " rbf: " + String(unnormalized) + " fused proba rows do not sum to 1")
    require(moved == 0, tag + " rbf: " + String(moved) + " fused predict codes are not the argmax of predict_proba")
    _ = b^
    _ = fitted^
    _ = proba^
    _ = codes^
    _ = placeholder_codes^
    _ = placeholder_proba^
    _ = status^


def lp_knn_unit(driver: XnDriver, spreading: Bool) raises:
    var b = Blobs(19)
    for i in range(0, b.n, 2):  # small-loop(rows)
        b.y[i] = -1
    var first = _lp_fit(driver, b, spreading, True, 20.0, 5, 200)
    var second = _lp_fit(driver, b, spreading, True, 20.0, 5, 200)
    var tag = String("LabelSpreading") if spreading else String("LabelPropagation")
    var c = b.k
    var differ = 0
    for i in range(b.n * c):  # small-loop(one fit's distributions)
        if bitcast[DType.uint32](first.distributions[i]) != bitcast[DType.uint32](second.distributions[i]):
            differ += 1
    require(differ == 0, tag + " knn: two fits gave different label_distributions_ bits")
    require(first.n_iter == second.n_iter, tag + " knn: two fits gave different n_iter_")
    var moved = 0
    for r in range(b.n):  # small-loop(rows)
        if first.transduction[r] != Int32(_argmax_f32(first.distributions, r, c)):
            moved += 1
    require(moved == 0, tag + " knn: " + String(moved) + " transduction_ rows are not the argmax of label_distributions_")
    _ = b^
    _ = first^
    _ = second^


# ------------------------------------------------------------------ driver


def fixture[H: HcvCalls](entry: NativeDescriptorCall, unit: Int) raises:
    var driver = XnDriver(entry)
    if unit < 4:
        var kernels: List[Int] = [KERNEL_RBF, KERNEL_LINEAR, KERNEL_POLY, KERNEL_SIGMOID]
        ocsvm_unit(driver, kernels[unit])
    elif unit < 6:
        nc_uniform_unit(driver, unit == 4)
    elif unit == 6:
        nc_empirical_unit(driver)
    elif unit == 7:
        hcv_unit[H]()
    elif unit < 10:
        lp_rbf_unit(driver, unit == 9)
    else:
        lp_knn_unit(driver, unit == 11)


# ---- The host column: independent fixtures across the host cores ----
def _run_cpu(mut report: Report) raises:
    var cases = report.cases(NE3_UNITS)
    def task(unit: Int) {imm cases}:
        try:
            fixture[HostHcv](xn_native_v1, unit)
        except error:
            cases.fail(unit, error)
    report.parallel(task, NE3_UNITS)


# ---- The device column ----
# The descriptor's graphs and the metrics entries share one DeviceContext, so
# the fixtures are submitted in order (as nearest_centroid_numeric does); the
# kernels keep their own parallelism.
def _run_gpu(mut report: Report) raises:
    var cases = report.cases(NE3_UNITS)
    for unit in range(NE3_UNITS):  # driver-loop: over the fixtures on the one device context
        try:
            fixture[DeviceHcv](xn_native_v1_gpu, unit)
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
    return Check("core_ne3_fused_entries", cpu=True, gpu=True, serial="", run=run_cases)
