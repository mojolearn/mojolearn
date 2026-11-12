# SPDX-License-Identifier: Apache-2.0
"""Operation-local public parameter pins before native Agglomerative parsing."""
from std.python import PythonObject


@fieldwise_init
struct AggloSettings(Movable):
    # These are the exact admitted public owners. Numeric conversion remains
    # at its established refusal point, then typed options/requests carry RUN.
    var numeric_mode: PythonObject
    var n_clusters: PythonObject
    var metric: PythonObject
    var connectivity: PythonObject
    var linkage: PythonObject
    var c: PythonObject
    var memory: PythonObject
    var compute_full_tree: PythonObject
    var distance_threshold: PythonObject
    var compute_distances: PythonObject
    var prediction_data: PythonObject
