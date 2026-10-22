# SPDX-License-Identifier: Apache-2.0
# Sparse storage is expanded in its original dtype before the established
# classical float32 cast. Duplicate order, integer wrap and Boolean OR match
# the dense representation that the public CPU methods previously consumed.
from std.memory import bitcast
from core.classical_pandas_impl import pd_raw_at


def host_sparse_uint64_bound(source: Int, rows: Int, columns: Int, stride0: Int, stride1: Int, native: Bool) raises:
    # The existing Array preparation normalizes uint64 through int64 before
    # its float32 cast. Preserve its overflow refusal after duplicate sums.
    for row in range(rows):
        for col in range(columns):
            if pd_raw_at(source, row * stride0 + col * stride1, 8, native) > UInt64(0x7FFFFFFFFFFFFFFF):
                raise Error("OverflowError: int too big to convert")


def host_sparse_write(destination: Int, word: UInt64, width: Int):
    var out = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=destination)
    for i in range(width):
        out[i] = UInt8(word >> UInt64(8 * i))


def host_sparse_place(destination: Int, word: UInt64, width: Int, kind: Int, add: Bool):
    var result = word
    if add:
        var prior = pd_raw_at(destination, 0, width, True)
        if kind == 2:
            if width == 4:
                var value = bitcast[DType.float32](UInt32(prior)) + bitcast[DType.float32](UInt32(word))
                result = UInt64(bitcast[DType.uint32](value))
            else:
                var value = bitcast[DType.float64](prior) + bitcast[DType.float64](word)
                result = bitcast[DType.uint64](value)
        elif kind == 3:
            result = UInt64(prior != 0 or word != 0)
        else:
            # Unsigned words implement every signed/unsigned source dtype's
            # modular addition; the store retains only that dtype's width.
            result = prior + word
    host_sparse_write(destination, result, width)


def host_sparse_index(source: Int, stride: Int, width: Int, native: Bool, index: Int) -> Int:
    var word = pd_raw_at(source, index * stride, width, native)
    return Int(bitcast[DType.int32](UInt32(word))) if width == 4 else Int(bitcast[DType.int64](word))


def host_sparse_compressed(
    destination: Int, rows: Int, columns: Int, column_storage: Bool,
    pointers: Int, pointer_stride: Int, pointer_width: Int, pointer_native: Bool,
    indices: Int, index_stride: Int, index_width: Int, index_native: Bool,
    data: Int, data_stride: Int, row_stride: Int, col_stride: Int,
    count: Int, width: Int, kind: Int, native: Bool, block_rows: Int = 1, block_cols: Int = 1,
) raises:
    var outer = columns if column_storage else rows // block_rows
    var inner = rows if column_storage else columns // block_cols
    if host_sparse_index(pointers, pointer_stride, pointer_width, pointer_native, 0) != 0:
        raise Error("sparse pointers must start at zero")
    var start = 0
    for major in range(outer):
        var end = host_sparse_index(pointers, pointer_stride, pointer_width, pointer_native, major + 1)
        if end < start or end > count:
            raise Error("sparse pointers exceed the stored data or are not ordered")
        for entry in range(start, end):
            var minor = host_sparse_index(indices, index_stride, index_width, index_native, entry)
            if minor < 0 or minor >= inner:
                raise Error("sparse index exceeds its declared shape")
            for r in range(block_rows):
                for c in range(block_cols):
                    var row = minor if column_storage else major * block_rows + r
                    var col = major if column_storage else minor * block_cols + c
                    var word = pd_raw_at(data, entry * data_stride + r * row_stride + c * col_stride, width, native)
                    host_sparse_place(destination + (row * columns + col) * width, word, width, kind, True)
        start = end


def host_sparse_coo(
    destination: Int, rows: Int, columns: Int,
    row_indices: Int, row_stride: Int, row_width: Int, row_native: Bool,
    col_indices: Int, col_stride: Int, col_width: Int, col_native: Bool,
    data: Int, data_stride: Int, count: Int, width: Int, kind: Int, native: Bool,
) raises:
    for entry in range(count):
        var row = host_sparse_index(row_indices, row_stride, row_width, row_native, entry)
        var col = host_sparse_index(col_indices, col_stride, col_width, col_native, entry)
        if row < 0 or row >= rows or col < 0 or col >= columns:
            raise Error("sparse coordinate exceeds its declared shape")
        var word = pd_raw_at(data, entry * data_stride, width, native)
        host_sparse_place(destination + (row * columns + col) * width, word, width, kind, True)


def host_sparse_dia(
    destination: Int, rows: Int, columns: Int,
    offsets: Int, offset_stride: Int, offset_width: Int, offset_native: Bool,
    data: Int, row_stride: Int, col_stride: Int, diagonals: Int, stored_columns: Int,
    width: Int, kind: Int, native: Bool,
) raises:
    for diagonal in range(diagonals):
        var offset = host_sparse_index(offsets, offset_stride, offset_width, offset_native, diagonal)
        for col in range(min(columns, stored_columns)):
            var row = col - offset
            if row >= 0 and row < rows:
                var word = pd_raw_at(data, diagonal * row_stride + col * col_stride, width, native)
                host_sparse_place(destination + (row * columns + col) * width, word, width, kind, True)
