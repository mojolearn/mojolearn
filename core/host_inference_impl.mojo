# SPDX-License-Identifier: Apache-2.0
from std.math import isfinite


def host_inference_finite(source: Int, count: Int) raises:
    var values = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=source)
    for i in range(count):
        if not isfinite(values[i]):
            raise Error("X contains NaN or infinity")


def host_inference_raw_finite(source: Int, stride0: Int, stride1: Int, rows: Int, columns: Int, width: Int, native: Bool) raises:
    for row in range(rows):
        for col in range(columns):
            var data = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=source + row * stride0 + col * stride1)
            var word = UInt64(0)
            for i in range(width):
                word |= UInt64(data[i if native else width - i - 1]) << UInt64(8 * i)
            var mask = UInt64(0x7C00) if width == 2 else (UInt64(0x7F800000) if width == 4 else UInt64(0x7FF0000000000000))
            if (word & mask) == mask:
                raise Error("Input X contains NaN or infinity")


def host_inference_nd_finite(source: Int, shape: List[Int], strides: List[Int], width: Int, native: Bool) raises:
    """Scan admitted strided storage without copying or Python dispatch."""
    var rank = len(shape)
    var count = 1
    for axis in range(rank):
        count *= shape[axis]
    var columns = 1 if rank == 0 else shape[rank - 1]
    var stride = width if rank == 0 else strides[rank - 1]
    var rows = 0 if columns == 0 else count // columns
    for row in range(rows):
        var remaining = row
        var offset = 0
        for reverse_axis in range(1, rank):
            var axis = rank - reverse_axis - 1
            offset += (remaining % shape[axis]) * strides[axis]
            remaining //= shape[axis]
        host_inference_raw_finite(source + offset, 0, stride, 1, columns, width, native)
