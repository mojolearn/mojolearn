# SPDX-License-Identifier: Apache-2.0
"""WP2a host ownership/count/copy check; does not train or launch a GPU."""
from std.python import Python
from std.memory import bitcast
from forest_export_binding import (
    ForestExportRegistry, validate_forest_export_counts,
    validate_forest_export_destinations, copy_forest_export_leaves,
)


@fieldwise_init
struct CheckModel(Movable):
    var marker: Int


def main() raises:
    _ = Python.import_module("builtins")
    var registry = ForestExportRegistry[CheckModel]()
    var meta: List[Int64] = [2, 1]
    var descriptor = registry.insert(CheckModel(73), 2, 4, 1, meta^)
    var handle = Int(py=descriptor[0])
    registry.validate(handle, 2, 4, 1)
    if registry.entries[handle].model.marker != 73:
        raise Error("export registry lost its typed owned model")
    var rejected = False
    try:
        registry.validate(handle, 2, 5, 1)
    except:
        rejected = True
    if not rejected:
        raise Error("export accepted mismatched capacities")
    registry.release(handle)
    rejected = False
    try:
        registry.release(handle)
    except:
        rejected = True
    if not rejected:
        raise Error("export accepted double release")
    rejected = False
    try:
        validate_forest_export_counts(1, 2147483647, 2)
    except:
        rejected = True
    if not rejected:
        raise Error("export accepted overflowing leaf count")
    var input: List[Float32] = [bitcast[DType.float32](UInt32(0x80000000)),
        bitcast[DType.float32](UInt32(1)), Float32(0.125), Float32(-8)]
    var output = List[Float32](length=6, fill=Float32(19))
    copy_forest_export_leaves(input, Int(output.unsafe_ptr()), 1)
    if output[0] != 19 or output[5] != 19:
        raise Error("export copy wrote outside caller capacity")
    for i in range(4):
        if bitcast[DType.uint32](input[i]) != bitcast[DType.uint32](output[i + 1]):
            raise Error("export copy changed leaf bytes")
    print("PASS forest export typed ownership/counts/double-release/copy bytes")
