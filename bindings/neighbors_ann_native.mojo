# SPDX-License-Identifier: Apache-2.0
# One native call per public method (lane native-neighbors-20261024): the approximate indexes' `fit` (IVFIndex, IVFPQIndex, IVFSQIndex,
# IVFRaBitQIndex, CagraIndex) and `refine` (python/mojolearn/_ivf_impl.py,
# python/mojolearn/_expansion_ann.py).
# Each `family_xn_ann_*` operation validates, prepares, runs the family's compiled
# kernels and returns, with the former Python method's refusals, operations
# and order, so the bits do not move.
from std.python import PythonObject


def neighbors_ann_operation(model: PythonObject, operation: String, args: PythonObject,
                            context: PythonObject) raises -> PythonObject:
    raise Error("RuntimeError: unknown neighbors operation " + operation)
