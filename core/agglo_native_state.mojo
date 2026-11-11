# SPDX-License-Identifier: Apache-2.0
"""AgglomerativeClustering route plans and native resident-reference identity."""
from core.dbscan_native_api import DBSCANNativeDriver


comptime AGGLO_PAIRWISE_MAX_ROWS = 46340
comptime AGGLO_FULL_AUTO = 0
comptime AGGLO_FULL_FALSE = 1
comptime AGGLO_FULL_TRUE = 2


def agglo_linkage_code(name: String) -> Int:
    if name == "ward":
        return 0
    if name == "complete":
        return 1
    if name == "average":
        return 2
    if name == "single":
        return 3
    return -1


def agglo_metric_code(name: String) -> Int:
    if name == "euclidean" or name == "l2":
        return -1
    if name == "l1" or name == "manhattan" or name == "cityblock":
        return 1
    if name == "cosine":
        return 4
    if name == "precomputed":
        return 5
    return -2


def agglo_connectivity_code(name: String) -> Int:
    if name == "pairwise":
        return 0
    if name == "knn":
        return 1
    return -1


def agglo_predicts(linkage: Int, metric: Int) -> Bool:
    return linkage == 3 and metric == -1


@fieldwise_init
struct AggloOptions(Copyable, Movable):
    var linkage: Int
    var metric: Int
    var constrained: Bool
    var has_threshold: Bool
    var compute_distances: Bool
    var prediction_data: Bool
    var full_tree: Int

    def solver_route(self) -> Bool:
        return (agglo_predicts(self.linkage, self.metric) and not self.constrained
            and not self.has_threshold and not self.compute_distances)


@fieldwise_init
struct AggloTreePlan(Copyable, Movable):
    var full: Bool
    var merges: Int
    var cut: Int
    var use_threshold: Bool


def agglo_tree_plan(options: AggloOptions, rows: Int, clusters: Int) -> AggloTreePlan:
    # This is the existing full-tree cost rule, independent of benchmark shape.
    var full = True
    if options.constrained:
        if options.full_tree == AGGLO_FULL_AUTO:
            var bound = max(100.0, 0.02 * Float64(rows))
            full = options.has_threshold or Float64(clusters) < bound
        else:
            full = options.full_tree == AGGLO_FULL_TRUE
    var merges = rows - 1 if full else rows - clusters
    var cut = (1 if options.has_threshold else clusters) if full else 0
    return AggloTreePlan(full, merges, cut, full and options.has_threshold)


@fieldwise_init
struct AggloReferenceKey(Copyable, Movable):
    var refs_owner: Int
    var labels_owner: Int
    var rows: Int
    var features: Int

    def matches(self, other: AggloReferenceKey) -> Bool:
        # Retain the existing stored-owner identity cache. Admission snapshots
        # may be temporary views/copies; their identities were never this key.
        return (self.refs_owner == other.refs_owner and self.labels_owner == other.labels_owner
            and self.rows == other.rows and self.features == other.features)


@fieldwise_init
struct AggloReferenceState(Copyable, Movable):
    var fields_owner: Int
    var provider: Int
    var key: AggloReferenceKey
    var handle: Int
    var driver: DBSCANNativeDriver

    def matches(self, fields_owner: Int, provider: Int, key: AggloReferenceKey) -> Bool:
        return (self.handle >= 0 and self.fields_owner == fields_owner
            and self.provider == provider and self.key.matches(key))

    def take_handle(mut self) -> Int:
        var handle = self.handle
        self.handle = -1
        return handle
