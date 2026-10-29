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
    validate_forest_export_counts,
    validate_forest_export_destinations,
    copy_forest_export_leaves,
)


@fieldwise_init
struct ForestExportEntry[Model: Movable & Deinitable](Movable):
    var model: Self.Model
    var trees: Int
    var nodes: Int
    var outputs: Int
    var meta: List[Int64]


struct ForestExportRegistry[Model: Movable & Deinitable](Defaultable, Movable):
    var entries: Dict[Int, ForestExportEntry[Self.Model]]
    var next_id: Int

    def __init__(out self):
        self.entries = Dict[Int, ForestExportEntry[Self.Model]]()
        self.next_id = 1

    def insert(mut self, var model: Self.Model, trees: Int, nodes: Int,
               outputs: Int, var meta: List[Int64]) raises -> PythonObject:
        """Move one fitted native result into ownership; return only scalars."""
        validate_forest_export_counts(trees, nodes, outputs)
        if self.next_id == 9223372036854775807:
            raise Error("fitted forest export handle space exhausted")
        var handle = self.next_id
        self.next_id += 1
        # Build all Python objects before inserting, so an allocation failure
        # cannot strand a registry entry the caller has never received.
        var descriptor = Python.list()
        descriptor.append(PythonObject(handle))
        descriptor.append(PythonObject(trees))
        descriptor.append(PythonObject(nodes))
        descriptor.append(PythonObject(outputs))
        var metadata = Python.list()
        for m in range(len(meta)):  # small-loop(meta: at most six export metadata words): descriptor glue, not data
            metadata.append(PythonObject(Int(meta[m])))
        descriptor.append(metadata)
        self.entries[handle] = ForestExportEntry[Self.Model](model^, trees, nodes,
                                                       outputs, meta^)
        return descriptor

    def validate(self, handle: Int, trees: Int, nodes: Int, outputs: Int) raises:
        if handle not in self.entries:
            raise Error("unknown or released fitted forest export handle")
        ref entry = self.entries[handle]
        if entry.trees != trees or entry.nodes != nodes or entry.outputs != outputs:
            raise Error("fitted forest export destination sizes differ from owned model")

    def release(mut self, handle: Int) raises:
        if handle not in self.entries:
            raise Error("unknown or released fitted forest export handle")
        var released = self.entries.pop(handle)
        _ = released^


def forest_export_py_list_i64(values: List[Int64]) raises -> PythonObject:
    """Publish native signed export words through the shared sized-list owner."""
    return py_list_i64(values)


def forest_export_py_list_f64(values: List[Float64]) raises -> PythonObject:
    """Publish native export doubles without an intermediate NumPy array."""
    return py_list_f64(values)
