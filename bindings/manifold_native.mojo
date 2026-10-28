# SPDX-License-Identifier: Apache-2.0
# The compiled owner of TSNE (python/mojolearn/_expansion_ann.py), UMAP (_umap_impl.py), Isomap, MDS, ClassicalMDS, LocallyLinearEmbedding (_expansion_decomp.py): one native call per public method,
# reached from bindings/family_native.mojo for every `family_mani_*` operation:
# TSNE in bindings/mani_tsne_native.mojo, UMAP in bindings/mani_umap_native.mojo,
# the decomp lane's four in bindings/mani_decomp_native.mojo (on the decomp kit,
# bindings/mani_kit.mojo).
from std.python import Python, PythonObject
from bindings.mani_tsne_native import tsne_operation
from bindings.mani_umap_native import umap_operation
from bindings.mani_decomp_native import mani_decomp_operation
from bindings.manifold_settings_native import ManifoldSettings


def manifold_operation(model: PythonObject, operation: String, args: PythonObject,
         context: PythonObject) raises -> PythonObject:
    var settings = ManifoldSettings(model, operation, context)
    if operation.startswith("family_mani_tsne_"):
        return tsne_operation(model, settings, operation, args, context)
    if operation.startswith("family_mani_umap_"):
        return umap_operation(model, settings, operation, args, context)
    return mani_decomp_operation(model, settings, operation, args, context)
