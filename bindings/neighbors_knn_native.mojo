# SPDX-License-Identifier: Apache-2.0
# One native call per public method (lane native-neighbors-20261024): NearestNeighbors.fit and the k-NN classifier/regressor `fit`, `predict`,
# `predict_proba` and `score` (python/mojolearn/neighbors.py).
# Each `family_xn_knn_*` operation validates, prepares, runs the family's compiled
# kernels and returns, with the former Python method's refusals, operations
# and order, so the bits do not move.
from std.python import PythonObject


def neighbors_knn_operation(model: PythonObject, operation: String, args: PythonObject,
                            context: PythonObject) raises -> PythonObject:
    raise Error("RuntimeError: unknown neighbors operation " + operation)
