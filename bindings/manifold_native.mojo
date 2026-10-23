# SPDX-License-Identifier: Apache-2.0
# The compiled owner of TSNE (python/mojolearn/_expansion_ann.py), UMAP (_umap_impl.py), Isomap, MDS, ClassicalMDS, LocallyLinearEmbedding (_expansion_decomp.py): one native call per public method,
# reached from bindings/family_native.mojo for every `family_mani_*` operation.
from std.python import Python, PythonObject
from bindings.mani_tsne_native import tsne_operation


def manifold_operation(model: PythonObject, operation: String, args: PythonObject,
         context: PythonObject) raises -> PythonObject:
    if operation.startswith("family_mani_tsne_"):
        return tsne_operation(model, operation, args, context)
    raise Error("RuntimeError: unknown family operation " + operation)
