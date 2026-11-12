# SPDX-License-Identifier: Apache-2.0
"""RUN of `bindings/forest_export_binding.mojo` (wave 8 run split): the typed
fitted-forest export registry (insert, sizes, validate, release). Imports no
Python; the binding keeps the Python descriptor (`forest_export_insert_py`,
`forest_export_descriptor_py`)."""
from core.forest_fit_request import forest_export_sizes_into
from core.forest_export_binding_impl import validate_forest_export_counts


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

    def insert_native(mut self, var model: Self.Model, trees: Int, nodes: Int,
                      outputs: Int, var meta: List[Int64]) raises -> Int:
        """Move one fitted native result into ownership; return its handle
        (the forest entry's fits, lane calltable-forest)."""
        validate_forest_export_counts(trees, nodes, outputs)
        if self.next_id == 9223372036854775807:
            raise Error("fitted forest export handle space exhausted")
        var handle = self.next_id
        self.next_id += 1
        self.entries[handle] = ForestExportEntry[Self.Model](model^, trees, nodes,
                                                       outputs, meta^)
        return handle

    def sizes_into(self, handle: Int, out_addr: Int) raises:
        """XF_EXPORT_SIZES (core/forest_fit_request.mojo)."""
        if handle not in self.entries:
            raise Error("unknown or released fitted forest export handle")
        ref entry = self.entries[handle]
        forest_export_sizes_into(out_addr, entry.trees, entry.nodes, entry.outputs, entry.meta)

    def meta_of(self, handle: Int) raises -> List[Int64]:
        if handle not in self.entries:
            raise Error("unknown or released fitted forest export handle")
        return self.entries[handle].meta.copy()

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
