# SPDX-License-Identifier: Apache-2.0
"""Import-time archive schemas for the compiled saved-model owner."""


def connect(context, formats):
    # glue: fixed native type and format metadata, connected once at import.
    # No class.load/from_file/constructor callback is installed here.
    from ._native_classical_bridge import _numpy_free_context, _archive_npy_dtypes
    from types import GetSetDescriptorType
    from .cluster import _METRIC_NAMES, _INIT_NAMES
    from .density import _METRICS
    from ._hierarchy_impl import _ALL_METRICS

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
    archive["host_archive_formats"] = {
        fmt: dict(classes) for fmt, classes in formats.items()  # glue: copy the fixed native class registry once at package connection
    }
    context["host_archive_context"] = archive
