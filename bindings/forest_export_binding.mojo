# SPDX-License-Identifier: Apache-2.0
"""WP2a/DEVIATION 2482: shared ownership for fitted-forest host export.

A family binding specializes ForestExportRegistry with its native model type;
no prediction snapshot or device allocation is created here. The caller holds
the GIL for insert/export/diagnostic/release. Training may release it before
insertion. Handles are monotonically increasing and never aliases for pointers.
The family-specific exporter reads its existing tree/node representation into
validated caller buffers; this module never builds a second flattened model.
"""
from bindings.py_lists import py_list_i64, py_list_f64
from std.python import Python, PythonObject
from core.forest_export_binding_impl import (
    validate_forest_export_destinations,
    copy_forest_export_leaves,
)
from core.forest_export_run import ForestExportRegistry


def forest_export_insert_py[Model: Movable & Deinitable](
    mut registry: ForestExportRegistry[Model], var model: Model, trees: Int, nodes: Int,
    outputs: Int, var meta: List[Int64],
) raises -> PythonObject:
    """`registry.insert_native`, then the Python descriptor (PACK of the
    Python entries); the entry is released again when the descriptor cannot
    be built, so no handle is stranded."""
    var handle = registry.insert_native(model^, trees, nodes, outputs, meta^)
    try:
        return forest_export_descriptor_py(registry, handle)
    except error:
        registry.release(handle)
        raise error^


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


def forest_export_py_list_i64(values: List[Int64]) raises -> PythonObject:
    """Publish native signed export words through the shared sized-list owner."""
    return py_list_i64(values)


def forest_export_py_list_f64(values: List[Float64]) raises -> PythonObject:
    """Publish native export doubles without an intermediate NumPy array."""
    return py_list_f64(values)
