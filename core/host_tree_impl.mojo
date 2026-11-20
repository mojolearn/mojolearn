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


# Completed class-code label publication: each predicted row's class word is
# gathered from k per-class words converted once by the caller (never one
# Python object per row).
def host_label_codes_in_range(indices: List[Int], classes: Int) -> Bool:
    for i in range(len(indices)):
        if indices[i] < 0 or indices[i] >= classes:
            return False
    return True


def host_label_first_unconverted(indices: List[Int], converted: List[Bool]) -> Int:
    """The first row whose class has no native word, or -1."""
    for i in range(len(indices)):
        if not converted[indices[i]]:
            return i
    return -1


def host_label_gather_i64(indices: List[Int], words: List[Int64], destination: Int):
    var dst = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=destination)
    for i in range(len(indices)):
        dst[i] = words[indices[i]]


def host_label_gather_f64(indices: List[Int], words: List[Float64], destination: Int):
    var dst = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=destination)
    for i in range(len(indices)):
        dst[i] = words[indices[i]]
