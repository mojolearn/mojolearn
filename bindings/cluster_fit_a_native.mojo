# SPDX-License-Identifier: Apache-2.0
# The compiled owner of the fits of MiniBatchKMeans, BisectingKMeans, MeanShift, AffinityPropagation (python/mojolearn/_expansion_cluster.py) and HDBSCAN (hdbscan.py): one native call per public method,
# reached from bindings/family_native.mojo for every `family_cfa_*` operation.
from std.python import Python, PythonObject


def cluster_fit_a_operation(model: PythonObject, operation: String, args: PythonObject,
         context: PythonObject) raises -> PythonObject:
    raise Error("RuntimeError: unknown family operation " + operation)
