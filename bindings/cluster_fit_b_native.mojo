# SPDX-License-Identifier: Apache-2.0
# The compiled owner of the fits of DBSCAN (python/mojolearn/density.py), AgglomerativeClustering (_hierarchy_impl.py), SpectralClustering and SpectralEmbedding / spectral_embedding (_spectral_impl.py): one native call per public method,
# reached from bindings/family_native.mojo for every `family_cfb_*` operation.
from std.python import Python, PythonObject
from bindings.classical_attributes import classical_connect_numpy
from bindings.cfb_dbscan import cfb_dbscan_operation


def cluster_fit_b_operation(model: PythonObject, operation: String, args: PythonObject,
         context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    if operation.startswith("family_cfb_dbscan_"):
        return cfb_dbscan_operation(model, operation, args, context)
    raise Error("RuntimeError: unknown family operation " + operation)
