# SPDX-License-Identifier: Apache-2.0
# The compiled owner of the fits of DBSCAN (python/mojolearn/density.py), AgglomerativeClustering (_hierarchy_impl.py), SpectralClustering and SpectralEmbedding / spectral_embedding (_spectral_impl.py): one native call per public method,
# reached from bindings/family_native.mojo for every `family_cfb_*` operation.
from std.python import Python, PythonObject


def cluster_fit_b_operation(model: PythonObject, operation: String, args: PythonObject,
         context: PythonObject) raises -> PythonObject:
    raise Error("RuntimeError: unknown family operation " + operation)
