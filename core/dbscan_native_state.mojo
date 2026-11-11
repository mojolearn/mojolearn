# SPDX-License-Identifier: Apache-2.0
"""Typed reference identity and take-once handle state for DBSCAN prediction."""
from core.dbscan_native_api import DBSCANNativeDriver


@fieldwise_init
struct DBSCANReferenceKey(Copyable, Movable):
    var refs_owner: Int
    var keys_owner: Int
    var labels_owner: Int
    var refs: Int
    var keys: Int
    var labels: Int
    var rows: Int
    var features: Int
    var metric: Int

    def matches(self, other: Self) -> Bool:
        return (self.refs_owner == other.refs_owner and self.keys_owner == other.keys_owner
            and self.labels_owner == other.labels_owner and self.refs == other.refs
            and self.keys == other.keys and self.labels == other.labels
            and self.rows == other.rows and self.features == other.features and self.metric == other.metric)


@fieldwise_init
struct DBSCANReferenceState(Movable):
    var owner: Int
    var provider: Int
    var key: DBSCANReferenceKey
    var handle: Int
    var driver: DBSCANNativeDriver

    def matches(self, owner: Int, provider: Int, key: DBSCANReferenceKey) -> Bool:
        return self.handle >= 0 and self.owner == owner and self.provider == provider and self.key.matches(key)

    def take_handle(mut self) -> Int:
        var handle = self.handle
        self.handle = -1
        return handle


def dbscan_metric_code(name: String) -> Int:
    if name == "euclidean" or name == "l2":
        return 0
    if name == "manhattan" or name == "l1" or name == "cityblock":
        return 1
    if name == "cosine":
        return 2
    if name == "precomputed":
        return 3
    return -1


def dbscan_method_code(name: String) -> Int:
    if name == "brute":
        return 0
    if name == "rbc":
        return 1
    return -1


def dbscan_method_accepts(method: Int, metric: Int) -> Bool:
    # The public RBC spelling admits precomputed matrices, then fit forces
    # that metric onto the existing brute adjacency route.
    return (method == 0 and metric >= 0 and metric <= 3) or (method == 1 and (metric == 0 or metric == 3))
