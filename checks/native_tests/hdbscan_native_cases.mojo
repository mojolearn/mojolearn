# SPDX-License-Identifier: Apache-2.0
"""Actual HDBSCAN doors against independent separated-chain oracles.

Two unit-spaced chains have two components below their separating edge.
With min_samples=2, core distances are two at endpoints and one inside.
The three-point chains die at lambda=1/2; the sixty-point chains shed
endpoints at 1/2 and die at 1. These facts give labels and probabilities
without consulting another execution of the implementation. Exact interior
exemplars have one-hot membership. Existing random-blob Python oracles stay.
"""
from core.host_parallel import host_parallelize
from core.kmeans_variants_api import VariantsSpan
from core.native_descriptor_fixed_call import native_descriptor_call_fixed
from core.hdbscan_fit_api import (
    HdbscanFitDriver, HdbscanFitOptions, HdbscanFitRequest, HdbscanFitPlan,
    HdbscanPredictionRequest, HdbscanPredictionPlan, HDBSCAN_FIT_HOST,
)
from core.hdbscan_query_api import (
    HdbscanQueryDriver, HdbscanQueryDimensions, HdbscanQueryBuffers,
    HdbscanQueryRequest, HdbscanQueryPlan, HDBSCAN_QUERY_APPROXIMATE,
    HDBSCAN_QUERY_MEMBERSHIP, HDBSCAN_QUERY_ALL_POINTS,
)

comptime HDBSCAN_NATIVE_CASES = 8


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def check_flags(flags: List[Int], where: String) raises:
    var failed = 0
    for flag in flags:  # metadata: independent row verdicts
        failed += flag
    require(failed == 0, where + ": " + String(failed) + " rows differ from analytic oracle")


def fixture(fit_driver: HdbscanFitDriver, query_driver: HdbscanQueryDriver, unit: Int) raises:
    if unit == 6:
        var a: Array[Int, 1] = [0]
        var p: Array[Int, 0] = []
        var f: Array[Float64, 0] = []
        var caught = False
        try:
            _ = native_descriptor_call_fixed(query_driver.entry, HDBSCAN_QUERY_APPROXIMATE, a, p, f)
        except problem:
            require(String(problem) == "HDBSCAN query ABI: missing prepared owner", "missing-owner refusal")
            caught = True
        require(caught, "null query owner must refuse before any kernel")
        return
    var wide = unit == 4 or unit == 5
    var half = 60 if wide else 3
    var n = half * 2
    var d = 2 if wide else 1
    var gap = 600 if wide else 10
    var source = List[Float32](length=n * d, fill=0)
    var xp = source.unsafe_ptr()
    def fill(row: Int) {imm xp, imm half, imm d, imm gap}:
        xp[row * d] = Float32(row % half + (gap if row >= half else 0))
    host_parallelize(fill, n)
    var labels = List[Int32](length=n, fill=-9)
    var core = List[Float32](length=n, fill=-9)
    var probabilities = List[Float32](length=n, fill=-9)
    var lp = labels.unsafe_ptr()
    var cp = core.unsafe_ptr()
    var pp = probabilities.unsafe_ptr()
    var want_tree = unit != 0
    var options = HdbscanFitOptions(n, d, 2, 8 if wide else 2, 0, False,
        1 if unit == 2 else 0, 1, want_tree, 1, 0)
    var request = HdbscanFitRequest(VariantsSpan(Int(xp), n * d),
        VariantsSpan(Int(lp), n), VariantsSpan(Int(cp), n), VariantsSpan(Int(pp), n), options.copy())
    var fit = HdbscanFitPlan(want_tree)
    if unit == 7:
        # Deliberately wrong info layout is refused before source capture or
        # output pointers could reach either provider's numerical entry.
        fit.storage[0].info = List[Int32](length=4, fill=0)
        var caught = False
        try:
            _ = fit_driver.fit(request, fit)
        except problem:
            require(String(problem).find("info owner does not match") >= 0, "fit owner-mode refusal")
            caught = True
        require(caught, "wrong fit owner must refuse")
        _ = source
        _ = labels
        _ = core
        _ = probabilities
        return
    if want_tree:
        fit.storage[0].prepare_tree(n)
    var host = fit_driver.metadata(HDBSCAN_FIT_HOST) != 0
    if host:
        fit.capture_host_source(request)
    var clusters = fit_driver.fit(request, fit)
    require(clusters == 2 and fit.storage[0].info[0] == 2 and fit.storage[0].info[1] == 0,
        "two separated chains produce two clusters without noise")
    var left = labels[1]
    var right = labels[half + 1]
    require(left >= 0 and right >= 0 and left != right, "cluster labels may permute but must separate chains")
    var flags = List[Int](length=n, fill=0)
    var fp = flags.unsafe_ptr()
    def check_fit(row: Int) {imm lp, imm cp, imm pp, imm fp, imm half, imm wide, imm left, imm right}:
        var endpoint = row % half == 0 or row % half == half - 1
        var expected_core = Float32(2) if endpoint else Float32(1)
        var expected_probability = Float32(0.5) if wide and endpoint else Float32(1)
        if lp[row] != (left if row < half else right) or abs(cp[row] - expected_core) > 0.000001 or abs(pp[row] - expected_probability) > 0.000001:
            fp[row] = 1
    host_parallelize(check_fit, n)
    check_flags(flags, "fit labels/core/probability")
    if want_tree:
        var edges = Int(fit.storage[0].info[4])
        var condensed = Int(fit.storage[0].info[3])
        var deaths = List[Float32](length=condensed, fill=0)
        var selected = List[Int32](length=clusters, fill=0)
        var offsets = List[Int32](length=clusters + 1, fill=0)
        var index_into = List[Int32](length=edges + 1, fill=0)
        var prediction = HdbscanPredictionPlan(n)
        var preparation = HdbscanPredictionRequest(VariantsSpan(Int(lp), n),
            VariantsSpan(Int(fit.storage[0].tree_parents.unsafe_ptr()), edges),
            VariantsSpan(Int(fit.storage[0].tree_children.unsafe_ptr()), edges),
            VariantsSpan(Int(fit.storage[0].tree_lambdas.unsafe_ptr()), edges),
            VariantsSpan(Int(fit.storage[0].tree_sizes.unsafe_ptr()), edges),
            VariantsSpan(Int(fit.storage[0].inverse.unsafe_ptr()), clusters),
            VariantsSpan(Int(deaths.unsafe_ptr()), condensed),
            VariantsSpan(Int(selected.unsafe_ptr()), clusters),
            VariantsSpan(Int(offsets.unsafe_ptr()), clusters + 1),
            VariantsSpan(Int(index_into.unsafe_ptr()), edges + 1), n, edges, condensed, clusters)
        prediction.storage[0].capture(preparation)
        var exemplars = fit_driver.prediction(preparation, prediction)
        require(exemplars >= 2 and offsets[0] == 0 and offsets[2] == Int32(exemplars)
            and offsets[1] > 0 and offsets[1] < Int32(exemplars), "both selected clusters own exemplars")
        prediction.storage[0].release_inputs()
        var query = List[Float32](length=3 * d, fill=0)
        query[0] = 1
        query[d] = Float32(gap + 1)
        query[2 * d] = 10000
        var output_labels = List[Int32](length=3, fill=-9)
        var output_probabilities = List[Float32](length=3, fill=-9)
        var output_membership = List[Float32](length=n * clusters, fill=-9)
        var buffers = HdbscanQueryBuffers(request.source, request.core_distances, request.labels,
            preparation.parents, preparation.lambdas, preparation.deaths, preparation.selected,
            preparation.index_into, VariantsSpan(Int(prediction.storage[0].exemplars.unsafe_ptr()), exemplars),
            preparation.offsets, VariantsSpan(Int(query.unsafe_ptr()), 3 * d),
            VariantsSpan(Int(output_labels.unsafe_ptr()), 3), VariantsSpan(Int(output_probabilities.unsafe_ptr()), 3))
        var dimensions = HdbscanQueryDimensions(n, d, edges, condensed, exemplars, clusters, 2)
        var operation = HDBSCAN_QUERY_APPROXIMATE
        var nq = 3
        var row_start = 0
        if unit == 2:
            operation = HDBSCAN_QUERY_MEMBERSHIP
            nq = 2
        elif unit == 3 or unit == 5:
            operation = HDBSCAN_QUERY_ALL_POINTS
            row_start = 1
            nq = n - 2
        if operation != HDBSCAN_QUERY_APPROXIMATE:
            buffers.output = VariantsSpan(Int(output_membership.unsafe_ptr()), nq * clusters)
        var query_request = HdbscanQueryRequest(dimensions.copy(), buffers.copy(), nq, row_start, 3)
        var plan = HdbscanQueryPlan(query_request, operation, host)
        require(query_driver.run(plan) == nq, "query returns consumed row count")
        if operation == HDBSCAN_QUERY_APPROXIMATE:
            require(output_labels[0] == left and output_labels[1] == right and output_labels[2] == -1,
                "exact interior exemplars retain labels and distant query is noise")
            require(abs(output_probabilities[0] - 1) < 0.000001 and abs(output_probabilities[1] - 1) < 0.000001
                and output_probabilities[2] == 0, "approximate prediction strength oracle")
        else:
            var mp = output_membership.unsafe_ptr()
            var verdicts = List[Int](length=nq, fill=0)
            var vp = verdicts.unsafe_ptr()
            var all_points = operation == HDBSCAN_QUERY_ALL_POINTS
            def check_membership(row: Int) {imm mp, imm vp, imm half, imm wide, imm left, imm right, imm row_start, imm all_points}:
                var source_row = row + row_start
                var label = (left if source_row < half else right) if all_points else (left if row == 0 else right)
                var own = mp[row * 2 + Int(label)]
                var other = mp[row * 2 + 1 - Int(label)]
                var endpoint = source_row % half == 0 or source_row % half == half - 1
                if all_points and wide and endpoint:
                    # These endpoints are not exemplars; their total membership
                    # is 1/2, and their own chain remains most probable.
                    if abs(own + other - 0.5) > 0.00001 or own <= other or other < 0:
                        vp[row] = 1
                elif abs(own - 1) > 0.00001 or abs(other) > 0.00001:
                    vp[row] = 1
            host_parallelize(check_membership, nq)
            check_flags(verdicts, "membership exact exemplars/endpoint probability")
        _ = plan
        _ = query
        _ = output_labels
        _ = output_probabilities
        _ = output_membership
        _ = prediction
        _ = deaths
        _ = selected
        _ = offsets
        _ = index_into
    _ = source
    _ = labels
    _ = core
    _ = probabilities
    _ = fit
