# SPDX-License-Identifier: Apache-2.0
"""Finite admission of host inference inputs. Every scan here is the one
parallel SIMD pass of `core/host_finite_scan.mojo` over the admitted words
(float16, float32 or float64, either byte order, any strides); the callers
keep their refusal wording."""
from core.host_finite_scan import (
    host_all_finite_f32,
    host_all_finite_view,
    host_all_finite_nd,
    host_finite_width_supported,
)


def host_inference_finite(source: Int, count: Int) raises:
    if not host_all_finite_f32(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=source), count):
        raise Error("X contains NaN or infinity")


def _host_inference_bytes_finite(source: Int, stride0: Int, stride1: Int, rows: Int, columns: Int, width: Int, native: Bool) raises:
    """The byte-assembling scan, kept only for a float width the word scan
    does not take (a 16-byte long double): serial because that storage has
    no fixed-width word to load; its verdict is unchanged."""
    for row in range(rows):
        for col in range(columns):
            var data = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=source + row * stride0 + col * stride1)
            var word = UInt64(0)
            for i in range(width):
                word |= UInt64(data[i if native else width - i - 1]) << UInt64(8 * i)
            var mask = UInt64(0x7C00) if width == 2 else (UInt64(0x7F800000) if width == 4 else UInt64(0x7FF0000000000000))
            if (word & mask) == mask:
                raise Error("Input X contains NaN or infinity")


def host_inference_raw_finite(source: Int, stride0: Int, stride1: Int, rows: Int, columns: Int, width: Int, native: Bool) raises:
    if not host_finite_width_supported(width):
        _host_inference_bytes_finite(source, stride0, stride1, rows, columns, width, native)
        return
    if not host_all_finite_view(source, stride0, stride1, rows, columns, width, native):
        raise Error("Input X contains NaN or infinity")


def host_inference_nd_finite(source: Int, shape: List[Int], strides: List[Int], width: Int, native: Bool) raises:
    """Scan admitted strided storage without copying or Python dispatch."""
    if host_finite_width_supported(width):
        if not host_all_finite_nd(source, shape, strides, width, native):
            raise Error("Input X contains NaN or infinity")
        return
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
        _host_inference_bytes_finite(source + offset, 0, stride, 1, columns, width, native)
