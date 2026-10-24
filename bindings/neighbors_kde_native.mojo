# SPDX-License-Identifier: Apache-2.0
# One native call per public method (lane native-neighbors-20261024): KernelDensity `fit` / `score` (python/mojolearn/density.py) and
# parallel_neighbors_reference.ReferenceShardedNeighbors `predict` /
# `predict_proba`.
# Each `family_xn_kde_*` operation validates, prepares, runs the family's compiled
# kernels and returns, with the former Python method's refusals, operations
# and order, so the bits do not move.
from std.python import PythonObject


def neighbors_kde_operation(model: PythonObject, operation: String, args: PythonObject,
                            context: PythonObject) raises -> PythonObject:
    raise Error("RuntimeError: unknown neighbors operation " + operation)
