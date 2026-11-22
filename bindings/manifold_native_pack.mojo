# SPDX-License-Identifier: Apache-2.0
# The compiled owner of TSNE (python/mojolearn/_expansion_ann.py), UMAP (_umap_impl.py), Isomap, MDS, ClassicalMDS, LocallyLinearEmbedding (_expansion_decomp.py): one native call per public method,
# reached from bindings/family_native.mojo for every `family_mani_*` operation:
# TSNE in bindings/mani_tsne_native.mojo, UMAP in bindings/mani_umap_native.mojo,
# the decomp lane's four in bindings/mani_decomp_native.mojo (on the decomp kit,
# bindings/mani_kit.mojo).
from std.python import Python, PythonObject
from bindings.mani_tsne_native_unpack import tsne_operation
from bindings.mani_umap_native_door import umap_operation
from bindings.mani_decomp_native_door import mani_decomp_operation, mani_decomp_vendor_used
from bindings.manifold_settings_native_door import ManifoldSettings, manifold_transforms
from bindings.manifold_lifecycle_native_door import (
    manifold_method, manifold_require_fitted, manifold_input, manifold_ignored_target,
    manifold_check_width, manifold_finish,
)
from bindings.classifier_public_lifecycle_door import _feature_names
from bindings.py_balanced import py_type, py_call, py_setitem, py_get
from bindings.native_loaded_module_door import native_loaded_module
from bindings.sequence_array_native_unpack import sequence_array_release
from providers.python.array_metadata_pyapi_door import ArrayMetadata, ArrayPin
from bindings.manifold_errors_native_unpack import manifold_error
from bindings.public_lifecycle_owner_door import lifecycle_begin, lifecycle_end
from bindings.operation_names_mani import (
    MANI_OPN_FAMILY_MANI_ISO_RECONSTRUCTION_ERROR, MANI_OPN_FAMILY_MANI_MDS_FIT, MANI_OPN_FAMILY_MANI_MDS_FIT_TRANSFORM,
    MANI_OPN_FAMILY_MANI_TSNE_FIT, MANI_OPN_FAMILY_MANI_TSNE_FIT_TRANSFORM, MANI_OPN_FAMILY_MANI_UMAP_FIT,
    MANI_OPN_FAMILY_MANI_UMAP_TRANSFORM, MANI_OPN_FAMILY_MANI_VENDOR_USED,
)
from bindings.mani_keys_unpack import mani_keys
from bindings.py_balanced import py_item


def manifold_unknown_operation(operation: String) raises:
    """A `family_mani_*` name that is not in the operation table (lane
    bw2-manifold): the texts the owners' chains raised, chosen by the same
    prefixes. Runs only for such a name, before any admission."""
    if operation.startswith("family_mani_umap_"):
        raise Error("RuntimeError: unknown UMAP operation " + operation)
    if operation.startswith("family_mani_tsne_"):
        raise Error("RuntimeError: unknown TSNE operation " + operation)
    raise Error("RuntimeError: unknown manifold operation " + operation)


def manifold_operation(model: PythonObject, operation: Int, args: PythonObject,
         shared_context: PythonObject) raises -> PythonObject:
    # Scalar snapshots belong to this operation, never a mutable global cache.
    # Retained connection tables remain shared; only this exact native mapping
    # receives private scalar owners and operation-local helper cache entries.
    var context = py_call(native_loaded_module("builtins").dict, shared_context)
    # The kit's cross-call constants (the host 1 x 1 zero, the device zero
    # per binding) live in the long-lived shared context (row N290); only
    # the scalar snapshots and helper entries below are operation-local.
    var nk = mani_keys()
    py_setitem(context, nk[].k_family_mani_shared, shared_context)
    py_setitem(context, nk[].k_family_mani_scalar_cache, Python.dict())
    py_setitem(context, nk[].k_module_type, py_type(native_loaded_module("sys")))
    var release = sequence_array_release(context) if py_get(py_item(context, nk[].k_modules), nk[].k_numpy) is not None else 0
    py_setitem(context, nk[].k_family_mani_numpy_release, PythonObject(release))
    var settings = ManifoldSettings(model, operation, context)
    if operation == MANI_OPN_FAMILY_MANI_VENDOR_USED:
        # Metadata, no input and no lifecycle step (as PLS / the decomp kit).
        return mani_decomp_vendor_used(model, settings, context)
    if operation >= MANI_OPN_FAMILY_MANI_UMAP_FIT and operation <= MANI_OPN_FAMILY_MANI_UMAP_TRANSFORM:
        # UMAP's whole public lifecycle (the former `_mode` wrappers): the
        # fitted check (NotFittedError), X refusals, width and feature-name
        # checks, fit depth, then manifold_finish's fit bookkeeping and NumPy
        # results for callers outside the package.
        var umap_method = manifold_method(operation)
        manifold_require_fitted(model, settings, umap_method, context)
        var umap_source = args[0]
        var umap_names = _feature_names(umap_source, context)
        var depth = lifecycle_begin(model, umap_method, umap_source, True, False, True, context)
        var umap_result: PythonObject
        try:
            umap_result = umap_operation(model, settings, operation, args, context)
        finally:
            lifecycle_end(model, umap_method, depth)
        return manifold_finish(settings, umap_result, umap_method, umap_names, context)
    var method = manifold_method(operation)
    if operation == MANI_OPN_FAMILY_MANI_ISO_RECONSTRUCTION_ERROR:
        return mani_decomp_operation(model, settings, operation, args, Optional[ArrayMetadata](), context)
    manifold_require_fitted(model, settings, method, context)
    var name = String(py_type(model).__name__)
    var tsne = operation == MANI_OPN_FAMILY_MANI_TSNE_FIT or operation == MANI_OPN_FAMILY_MANI_TSNE_FIT_TRANSFORM
    var source = args[2]
    var x_pin = ArrayPin()
    var x = manifold_input(source, name, not tsne, context, x_pin)
    var names = _feature_names(source, context)
    var transforms = manifold_transforms(operation)
    if transforms:
        manifold_check_width(settings, x, names, name, context)
    else:
        # Staged public shells append the formerly ignored y. Old internal
        # callers remain valid while the import-time front doors are connected.
        var y_at = 4 if (operation == MANI_OPN_FAMILY_MANI_MDS_FIT or operation == MANI_OPN_FAMILY_MANI_MDS_FIT_TRANSFORM) else 3
        if len(args) > y_at:
            manifold_ignored_target(args[y_at], x.source.rows, name, context)
    var result: PythonObject
    try:
        if tsne:
            result = tsne_operation(model, settings, operation, args, x, x_pin, context)
        else:
            result = mani_decomp_operation(model, settings, operation, args, Optional(x.copy()), context)
    except error:
        var refined = manifold_error(error, x.source.rows, x.source.cols, not transforms, name)
        raise refined^
    _ = source
    _ = x
    _ = x_pin^  # the door's pin held X through every RUN read
    return manifold_finish(settings, result, method, names, context)
