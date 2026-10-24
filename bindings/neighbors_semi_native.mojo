# SPDX-License-Identifier: Apache-2.0
# One native call per public method (lane native-neighbors-20261024): LabelPropagation / LabelSpreading and the AdditiveChi2Sampler /
# SkewedChi2Sampler public methods (python/mojolearn/_expansion_neighbors.py).
# Each `family_xn_semi_*` operation validates, prepares, runs the family's compiled
# kernels and returns, with the former Python method's refusals, operations
# and order, so the bits do not move.
from std.python import PythonObject


def neighbors_semi_operation(model: PythonObject, operation: String, args: PythonObject,
                             context: PythonObject) raises -> PythonObject:
    raise Error("RuntimeError: unknown neighbors operation " + operation)
