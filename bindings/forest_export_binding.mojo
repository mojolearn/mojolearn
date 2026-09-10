# SPDX-License-Identifier: Apache-2.0
"""WP2a/DEVIATION 2482: shared ownership for fitted-forest host export.

A family binding specializes ForestExportRegistry with its native model type;
no prediction snapshot or device allocation is created here. The caller holds
the GIL for insert/export/diagnostic/release. Training may release it before
insertion. Handles are monotonically increasing and never aliases for pointers.
The family-specific exporter reads its existing tree/node representation into
validated caller buffers; this module never builds a second flattened model.
"""
from std.memory import memcpy
from std.python import Python, PythonObject


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
        for item in meta:
            metadata.append(PythonObject(Int(item)))
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


def validate_forest_export_counts(trees: Int, nodes: Int, outputs: Int) raises:
    if trees < 1 or trees >= 2147483647 or nodes < trees or outputs < 1:
        raise Error("invalid fitted forest export counts")
    if nodes > 2147483647 // outputs:
        raise Error("fitted forest export exceeds Int32 indexing bounds")


def validate_forest_export_destinations(offsets: Int, columns: Int,
    thresholds: Int, left: Int, leaves: Int) raises:
    # ABI output arrays all have four-byte scalar storage. Sizes must be
    # checked against the registry before any pointer is written.
    var addresses: List[Int] = [offsets, columns, thresholds, left, leaves]
    for address in addresses:
        if address <= 0 or address % 4 != 0:
            raise Error("invalid fitted forest export destination pointer")


def copy_forest_export_leaves(values: List[Float32], address: Int,
                              offset: Int) raises:
    """One contiguous tree leaf-vector copy; no Float64/Python scalar detour."""
    if address <= 0 or address % 4 != 0 or offset < 0:
        raise Error("invalid fitted forest leaf export pointer or offset")
    var destination = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=address)
    if len(values):
        memcpy(dest=destination + offset, src=values.unsafe_ptr(), count=len(values))
