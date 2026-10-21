# SPDX-License-Identifier: Apache-2.0
"""Typed export checks and the leaf copy, moved word for word from
bindings/forest_export_binding.mojo (re-exported there under the same names)."""
from std.memory import memcpy


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
