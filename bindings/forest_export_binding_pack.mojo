# SPDX-License-Identifier: Apache-2.0
"""WP2a/DEVIATION 2482: shared ownership for fitted-forest host export.

A family binding specializes ForestExportRegistry with its native model type;
no prediction snapshot or device allocation is created here. The caller holds
the GIL for insert/export/diagnostic/release. Training may release it before
insertion. Handles are monotonically increasing and never aliases for pointers.
The family-specific exporter reads its existing tree/node representation into
validated caller buffers; this module never builds a second flattened model.
"""
# The PACK half of the former `bindings/forest_export_binding.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/forest_export_binding_unpack.mojo`.
from bindings.py_lists import py_list_i64, py_list_f64
from std.python import Python, PythonObject
from core.forest_export_binding_impl import (
    validate_forest_export_destinations,
    copy_forest_export_leaves,
)
from core.forest_export_run import ForestExportRegistry


def forest_export_descriptor_py[Model: Movable & Deinitable](
    registry: ForestExportRegistry[Model], handle: Int,
) raises -> PythonObject:
    """`[handle, trees, nodes, outputs, [meta...]]` (PACK)."""
    if handle not in registry.entries:
        raise Error("unknown or released fitted forest export handle")
    ref entry = registry.entries[handle]
    var descriptor = Python.list()
    descriptor.append(PythonObject(handle))
    descriptor.append(PythonObject(entry.trees))
    descriptor.append(PythonObject(entry.nodes))
    descriptor.append(PythonObject(entry.outputs))
    var metadata = Python.list()
    for m in range(len(entry.meta)):  # small-loop(meta: at most six export metadata words): descriptor glue, not data
        metadata.append(PythonObject(Int(entry.meta[m])))
    descriptor.append(metadata)
    return descriptor
