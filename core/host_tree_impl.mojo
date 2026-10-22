# SPDX-License-Identifier: Apache-2.0
# Completed host-tree output layout and label selection. Prediction arithmetic
# remains in the existing RF/ET/GBDT inference kernels.


def host_tree_widen(source: Int, count: Int, destination: Int):
    var src = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=source)
    var dst = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=destination)
    for i in range(count):
        dst[i] = Float64(src[i])


def host_tree_argmax(source: Int, rows: Int, columns: Int) -> List[Int]:
    var src = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=source)
    var result = List[Int](capacity=rows)
    for row in range(rows):
        var best = 0
        var value = src[row * columns]
        for col in range(1, columns):
            var candidate = src[row * columns + col]
            if candidate > value:
                best = col
                value = candidate
        result.append(best)
    return result^


def host_tree_column_copy(source: Int, rows: Int, columns: Int, destination: Int):
    var src = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=source)
    var dst = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=destination)
    for row in range(rows):
        for col in range(columns):
            dst[col * rows + row] = src[row * columns + col]
