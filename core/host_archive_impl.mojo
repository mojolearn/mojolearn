# SPDX-License-Identifier: Apache-2.0
# Saved-state layout copies preserve every stored float32 bit, including
# NaN payloads. They perform no numerical normalization or validation.
from std.builtin.sort import sort
from std.collections import Dict
from core.hotpath_impl import reduce_stat, HP_F64, HP_SUM


def host_archive_copy_f32(source: Int, stride0: Int, stride1: Int, rows: Int, cols: Int,
                          destination: Int):
    var dst = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=destination)
    for row in range(rows):
        for col in range(cols):
            var src = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=source + row * stride0 + col * stride1)
            dst[row * cols + col] = src[0]


def host_archive_unique_i32(source: Int, stride: Int, count: Int) -> List[Int32]:
    var labels = List[Int32]()
    for i in range(count):
        var value = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=source + i * stride)[0]
        labels.append(value)
    sort(labels)
    var result = List[Int32]()
    for i in range(len(labels)):
        if i == 0 or labels[i] != labels[i - 1]:
            result.append(labels[i])
    return result^


def host_archive_mean_f64(source: Int, count: Int) raises -> Float64:
    # Reuse Array.sum's ascending binary64 reduction and sabotage arm.
    return reduce_stat(source, HP_F64, count, HP_SUM).f / Float64(count)


def host_archive_reorder_f32(source: Int, shape: List[Int], strides: List[Int], count: Int,
                             destination: Int):
    var dst = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=destination)
    for i in range(count):
        var index = i
        var offset = 0
        for d in range(len(shape) - 1, -1, -1):
            offset += (index % shape[d]) * strides[d]
            index //= shape[d]
        dst[i] = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=source + offset)[0]


def host_archive_reorder_bytes(source: Int, shape: List[Int], strides: List[Int], count: Int,
                               width: Int, destination: Int):
    # _serialize.exact returns C-order storage for every supported numeric
    # dtype. Copy words as bytes so int64 and floating NaN payloads survive.
    var dst = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=destination)
    for i in range(count):
        var index = i
        var offset = 0
        for d in range(len(shape) - 1, -1, -1):
            offset += (index % shape[d]) * strides[d]
            index //= shape[d]
        var src = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=source + offset)
        for byte in range(width):
            dst[i * width + byte] = src[byte]


def host_archive_copy_words(source: Int, count: Int, destination: Int):
    var src = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=source)
    var dst = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=destination)
    for i in range(count):
        dst[i] = src[i]


def host_archive_support_bound(source: Int, count: Int, columns: Int = -1) raises -> Int:
    var src = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=source)
    var bound = 0
    for i in range(count):
        var row = Int(src[i])
        if row < 0 or (columns >= 0 and row >= columns):
            raise Error("saved precomputed support index is outside the model columns")
        bound = max(bound, row + 1)
    return bound


def host_archive_svc_gather(pair_support: Int, pair_rows: Int, vectors: Int, columns: Int,
                            support: Int, rows: Int, destination: Int) raises:
    var source_indices = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=pair_support)
    var result_indices = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=support)
    var positions = Dict[Int, Int]()
    for i in range(pair_rows):
        positions[Int(source_indices[i])] = i
    for i in range(rows):
        var at = positions[Int(result_indices[i])]
        host_archive_copy_words(vectors + at * columns * 4, columns, destination + i * columns * 4)
