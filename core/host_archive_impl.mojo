# SPDX-License-Identifier: Apache-2.0
# Saved-state layout copies preserve every stored float32 bit, including
# NaN payloads. They perform no numerical normalization or validation.
from std.builtin.sort import sort


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
