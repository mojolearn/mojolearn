# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/apple-fast-w4-small, opt-in (-D MOJOLEARN_RESAMPLE_FAST_ROW_GATHER),
FAST + Apple: `resample(X, y)`'s row gathers as one native copy loop.

Main gathers each array in Python with numpy fancy indexing
(python/mojolearn/resample.py `_take`: `np.asarray(a)[np.asarray(idx,
dtype=np.intp)]`), which first converts the 1M int32 indices to a new intp
array, then walks numpy's general index iterator row by row. Here the
binding copies row idx[i] of the caller's C-contiguous array into row i of
a caller-allocated output (np.empty, glue), reading the int32 indices as
drawn: data movement only, the same bytes (the output is byte-identical to
main's gather). Rows of 4 or 8 bytes (the targets, a 1-column or
2-column float32 array) are one word move each; wider rows one memcpy each.
An index outside [0, n) is refused (nothing past the source is read)."""
from std.memory import memcpy
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST

comptime RESAMPLE_ROW_GATHER = (GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
                                and is_defined["MOJOLEARN_RESAMPLE_FAST_ROW_GATHER"]())

comptime BP = MutPointer[UInt8, MutUntrackedOrigin]
comptime IP32 = MutPointer[Int32, MutUntrackedOrigin]
comptime WP32 = MutPointer[UInt32, MutUntrackedOrigin]
comptime WP64 = MutPointer[UInt64, MutUntrackedOrigin]


def gather_rows(idx: IP32, src: Int, dst: Int, count: Int, row_bytes: Int, n: Int) -> Int:
    """Row idx[i] of src (n rows of row_bytes) to row i of dst for i < count.
    Returns -1, or the first position whose index lies outside [0, n)
    (rows before it are copied, nothing after)."""
    var aligned8 = (src | dst) % 8 == 0
    var aligned4 = (src | dst) % 4 == 0
    if row_bytes == 4 and aligned4:
        var s = WP32(unsafe_from_address=src)
        var d = WP32(unsafe_from_address=dst)
        for i in range(count):
            var r = Int(idx[i])
            if r < 0 or r >= n:
                return i
            d[i] = s[r]
        return -1
    if row_bytes == 8 and aligned8:
        var s = WP64(unsafe_from_address=src)
        var d = WP64(unsafe_from_address=dst)
        for i in range(count):
            var r = Int(idx[i])
            if r < 0 or r >= n:
                return i
            d[i] = s[r]
        return -1
    var s = BP(unsafe_from_address=src)
    var d = BP(unsafe_from_address=dst)
    for i in range(count):
        var r = Int(idx[i])
        if r < 0 or r >= n:
            return i
        memcpy(dest=d + i * row_bytes, src=s + r * row_bytes, count=row_bytes)
    return -1
