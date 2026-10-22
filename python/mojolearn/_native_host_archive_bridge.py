# SPDX-License-Identifier: Apache-2.0
"""Import-time archive schemas for the compiled saved-model owner."""


def connect(context, formats, cleanup_entries):
    # glue: fixed native type and format metadata, connected once at import.
    # No class.load/from_file/constructor callback is installed here.
    from ._native_classical_bridge import _numpy_free_context, _archive_npy_dtypes
    from types import GetSetDescriptorType, BuiltinFunctionType, MethodType
    from .tokenizer import _binding
    from .cluster import _METRIC_NAMES, _INIT_NAMES
    from .density import _METRICS
    from ._hierarchy_impl import _ALL_METRICS
    from .neighbors import _METRIC_TABLE, _RBC_METRIC_TABLE
    from ._umap_impl import _METRICS as umap_metrics
    from ._mode import SklearnTagsMixin
    from ._gp_impl import _SavedKernel as gp_saved_kernel
    from ._gpc_impl import _SavedKernel as gpc_saved_kernel, _BinaryLaplace

    archive = dict(context)
    archive.update(_numpy_free_context())
    for key in ("ndarray", "scalar_type", "integer", "numeric_dtypes",  # glue: remove optional NumPy ABI metadata once at package connection
                "class_dtypes", "float_types", "float32", "float64", "int64"):
        archive.pop(key, None)
    archive["archive_npy_dtypes"] = _archive_npy_dtypes()
    archive["host_archive_dict_descriptor_type"] = GetSetDescriptorType
    archive["host_archive_kmeans_metrics"] = dict(_METRIC_NAMES)
    archive["host_archive_kmeans_starts"] = dict(_INIT_NAMES)
    archive["host_archive_dbscan_metrics"] = tuple(_METRICS)
    archive["host_archive_agglo_metrics"] = tuple(_ALL_METRICS)
    archive["host_archive_neighbor_metrics"] = dict(_METRIC_TABLE)
    archive["host_archive_rbc_metrics"] = dict(_RBC_METRIC_TABLE)
    archive["host_archive_umap_metrics"] = dict(umap_metrics)
    archive["host_archive_gp_saved_kernel"] = gp_saved_kernel
    archive["host_archive_gpc_saved_kernel"] = gpc_saved_kernel
    archive["host_archive_binary_laplace"] = _BinaryLaplace
    archive["host_archive_formats"] = {
        fmt: dict(classes) for fmt, classes in formats.items()  # glue: copy the fixed native class registry once at package connection
    }
    archive["host_archive_fitted_estimators"] = tuple(
        name for classes in formats.values() for name, cls in classes.items()  # glue: fixed fitted-state ownership metadata at import
        if issubclass(cls, SklearnTagsMixin)
    )
    archive["host_archive_lifetime_layouts"] = {}
    module = _binding()._module
    finalize = getattr(module, "host_archive_finalize_call", None)
    if finalize is not None:
        if not isinstance(finalize, BuiltinFunctionType):
            raise TypeError("saved host model cleanup requires a compiled native entry")
        descriptor = module.classical_attribute_descriptor(MethodType(finalize, archive))
        archive["host_archive_finalizer_descriptor"] = descriptor
        for classes in formats.values():  # glue: connect fixed host class lifetime metadata at import
            for name, cls in classes.items():  # glue: fixed saved estimator class metadata
                if not any("__del__" in base.__dict__ for base in cls.__mro__):  # glue: inspect fixed finalizer slots once
                    continue
                field_descriptor = next(base.__dict__["__dict__"] for base in cls.__mro__  # glue: fixed native instance layout
                                        if "__dict__" in base.__dict__)
                if not isinstance(field_descriptor, GetSetDescriptorType):
                    raise TypeError("saved host model cleanup requires a native instance dictionary")
                release = cleanup_entries.get(name)
                if release is not None and not isinstance(release, BuiltinFunctionType):
                    raise TypeError("saved host model release requires a compiled native entry")
                archive["host_archive_lifetime_layouts"][cls] = (release, field_descriptor)
                cls.__del__ = descriptor
    context["host_archive_context"] = archive
