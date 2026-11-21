# SPDX-License-Identifier: Apache-2.0
"""LocalOutlierFactor RUN: typed requests in, typed results out.

The boundary (bindings/neighbors_local_native_door.mojo) admits the public
arguments into these requests, releases the GIL once for each function here
and publishes the results. Nothing in this module imports Python. The staged
plan and its kernels are core/lof_plan.mojo's, in the order the door ran them.
"""
from core.neighbors_native_driver import XnDriver
from core.lof_plan import LofLabelsPlan, lof_f32_address, lof_offset
from neighbors.neighbors_local_run import (
    LOF_RESULT_SCORES, LOF_RESULT_DECISION, LOF_AUTO_OFFSET, lof_plan,
)


struct LofFitRequest(Copyable, Movable):
    # Addresses of admitted storage the boundary keeps alive: the float32
    # rows and the three published float32 arrays (`rows * neighbors` root
    # distances, `rows` densities, `rows` scores).
    var x: Int
    var distances: Int
    var densities: Int
    var scores: Int
    var rows: Int
    var features: Int
    var neighbors: Int
    var unfused: Bool
    var old_items: Bool
    var auto: Bool
    var contamination: Float64

    def __init__(out self, x: Int, rows: Int, features: Int, neighbors: Int, unfused: Bool, old_items: Bool,
                 auto: Bool, contamination: Float64):
        self.x = x
        self.distances = 0
        self.densities = 0
        self.scores = 0
        self.rows = rows
        self.features = features
        self.neighbors = neighbors
        self.unfused = unfused
        self.old_items = old_items
        self.auto = auto
        self.contamination = contamination


struct LofFitOutcome(Copyable, Movable):
    # `status` is the deferred finite word of the neighbor selection (0 =
    # every word finite); `offset` is the fitted `offset_`.
    var status: Int
    var offset: Float64

    def __init__(out self):
        self.status = 0
        self.offset = LOF_AUTO_OFFSET


def lof_fit_run(driver: XnDriver, request: LofFitRequest) raises -> LofFitOutcome:
    """Neighbors, root distances, densities, scores and the offset. A
    non-zero finite word stops after the root distances, as the door did."""
    var outcome = LofFitOutcome()
    var n = request.rows
    var plan = lof_plan(n, n, request.features, request.neighbors, True, request.unfused, request.old_items)
    outcome.status = plan.select(driver, request.x, request.x)
    plan.roots(driver, request.distances)
    if outcome.status != 0:
        return outcome^
    plan.density(driver, request.distances, request.distances, request.densities, n)
    plan.score(driver, request.densities, request.densities, request.scores, n)
    if not request.auto:
        outcome.offset = lof_offset(driver, request.scores, n, request.contamination)
    _ = plan^
    return outcome^


def lof_fit_labels_run(driver: XnDriver, scores: Int, rows: Int, offset: Float64, dst: Int) raises:
    """fit_predict's labels: `rows` int64 words at `dst` from the fitted
    scores against the fitted offset."""
    var labels = LofLabelsPlan(rows)
    if rows != 0:
        labels.classify(driver, scores, 0, offset)
    labels.write_i64(dst)
    _ = labels^


struct LofQueryRequest(Copyable, Movable):
    # Addresses of admitted storage the boundary keeps alive: the float32
    # query rows, the fitted rows, root distances and densities, and the
    # result (`rows` float32 words for scores and decisions, int64 for labels).
    var query: Int
    var fit_x: Int
    var fit_distances: Int
    var fit_densities: Int
    var dst: Int
    var rows: Int
    var reference_rows: Int
    var features: Int
    var neighbors: Int
    var fit_rows: Int
    var unfused: Bool
    var old_items: Bool
    var kind: Int
    var shift: Float64

    def __init__(out self, query: Int, fit_x: Int, fit_distances: Int, fit_densities: Int, rows: Int,
                 reference_rows: Int, features: Int, neighbors: Int, fit_rows: Int, unfused: Bool,
                 old_items: Bool, kind: Int, shift: Float64):
        self.query = query
        self.fit_x = fit_x
        self.fit_distances = fit_distances
        self.fit_densities = fit_densities
        self.dst = 0
        self.rows = rows
        self.reference_rows = reference_rows
        self.features = features
        self.neighbors = neighbors
        self.fit_rows = fit_rows
        self.unfused = unfused
        self.old_items = old_items
        self.kind = kind
        self.shift = shift


def lof_query_run(driver: XnDriver, request: LofQueryRequest) raises -> Int:
    """score_samples, decision_function or predict of new rows, by
    `request.kind`, into `request.dst`. Returns the deferred finite word of
    the neighbor query; on a non-zero word nothing is written."""
    var plan = lof_plan(request.rows, request.reference_rows, request.features, request.neighbors, False,
                        request.unfused, request.old_items)
    var status = plan.query_neighbors(driver, request.query, request.fit_x)
    if status != 0:
        return status
    plan.query_density(driver, request.fit_distances, request.fit_rows)
    if request.kind == LOF_RESULT_SCORES:
        plan.score(driver, request.fit_densities, lof_f32_address(plan.densities), request.dst, request.fit_rows)
        _ = plan^
        return 0
    plan.query_score(driver, request.fit_densities, request.fit_rows)
    if request.kind == LOF_RESULT_DECISION:
        plan.decision(driver, request.dst, request.shift)
        _ = plan^
        return 0
    plan.query_decision(driver, request.shift)
    var labels = LofLabelsPlan(request.rows)
    labels.classify(driver, lof_f32_address(plan.decisions), 1, 0.0)
    labels.write_i64(request.dst)
    _ = labels^
    _ = plan^  # the plan's decisions outlive the classification that reads them
    return 0
