# SPDX-License-Identifier: Apache-2.0
"""Typed pandas block preparation; external object decoding stays at UNPACK."""
from std.memory import bitcast
from core.cast_in import cast_in_kind, cast_w64_bits, f64_to_f32_bits
from core.classical_pandas_impl import pd_raw_at, pd_mask_check, pd_scatter_numeric
from core.sequential_input_plan import sequence_promote


def pandas_block_count(count: Int, series: Bool) raises:
    if count == 0 or (series and count != 1):
        raise Error("pandas input is empty or has invalid block metadata")


def pandas_nullable_check(data_rank: Int, mask_rank: Int, data_size: Int,
    mask_size: Int, mask_type: Int, address: Int, stride: Int, itemsize: Int,
    native: Bool) raises:
    if data_rank != 1 or mask_rank != 1 or data_size != mask_size or mask_type != 0:
        raise Error("TypeError: pandas nullable storage metadata differs")
    pd_mask_check(address, mask_size, stride, itemsize, native)


def pandas_positions_slice(start: Int, stop: Int, step: Int) raises -> List[Int]:
    if step == 0:
        raise Error("ValueError: range() arg 3 must not be zero")
    var count = max(0, (stop - start + step - 1) // step) if step > 0 else max(0, (start - stop - step - 1) // -step)
    var result = List[Int](capacity=count)
    for k in range(count):  # driver-loop: over pandas block column placements
        result.append(start + k * step)
    return result^


def pandas_positions_array(rank: Int, dtype: String, count: Int, base: Int,
    itemsize: Int, typecode: Int, stride: Int, native: Bool) raises -> List[Int]:
    if rank != 1 or dtype != "i":
        raise Error("TypeError: pandas block placement requires native integer storage")
    var kind = cast_in_kind(typecode, itemsize, "pandas placement")
    var result = List[Int](capacity=count)
    for k in range(count):  # driver-loop: over pandas block column placements
        result.append(Int(bitcast[DType.int64](cast_w64_bits(
            pd_raw_at(base, k * stride, itemsize, native), kind, itemsize))))
    return result^


struct PandasBlock(Copyable, Movable):
    var address: Int
    var rows: Int
    var cols: Int
    var row_stride: Int
    var col_stride: Int
    var itemsize: Int
    var kind: Int
    var native: Bool
    var objects: Bool
    var positions: Int

    def __init__(out self, address: Int, shape: List[Int], strides: List[Int],
        itemsize: Int, typecode: Int, dtype: String, native: Bool,
        series: Bool, prior_rows: Int) raises:
        var rank = len(shape)
        if rank != 1 and rank != 2:
            raise Error("pandas numeric block rank must be one or two")
        self.rows = shape[rank - 1]
        self.cols = 1 if rank == 1 else shape[0]
        if self.rows < 1 or self.cols < 1 or (prior_rows >= 0 and self.rows != prior_rows) or (series and rank != 1):
            raise Error("pandas numeric block dimensions differ or are empty")
        self.address = address
        self.row_stride = strides[rank - 1]
        self.col_stride = 0 if rank == 1 else strides[0]
        self.itemsize = itemsize
        self.native = native
        self.objects = dtype == "O"
        # Dtype admission/promotion precedes conversion-kind admission, as in
        # the original block walk. The typed kind is set by add_block below.
        self.kind = typecode
        self.positions = 0


struct PandasInputPlan(Movable):
    var blocks: List[PandasBlock]
    var placements: List[Int]
    var rows: Int
    var cols: Int
    var common_kind: String
    var common_size: Int
    var objects: Bool
    var booleans: Bool
    var numbers: Bool
    var double_round: Bool
    var allow_nonfinite: Bool
    var seen: List[Bool]

    def __init__(out self, allow_nonfinite: Bool):
        self.blocks = List[PandasBlock]()
        self.placements = List[Int]()
        self.rows = -1
        self.cols = 0
        self.common_kind = String()
        self.common_size = 0
        self.objects = False
        self.booleans = False
        self.numbers = False
        self.double_round = False
        self.allow_nonfinite = allow_nonfinite
        self.seen = List[Bool]()

    def add_block(mut self, var block: PandasBlock, positions: List[Int],
        dtype: String, object_promoted: Bool) raises:
        if len(positions) != block.cols:
            raise Error("pandas block column positions differ from numeric storage")
        self.rows = block.rows
        self.cols += block.cols
        self.objects = self.objects or block.objects or object_promoted
        if not block.objects:
            if dtype == "c":
                raise Error("Complex data not supported")
            sequence_promote(self.common_kind, self.common_size, dtype, block.itemsize)
            self.booleans = self.booleans or dtype == "b"
            self.numbers = self.numbers or dtype != "b"
        block.positions = len(self.placements)
        for position in positions:  # driver-loop: over native block placement metadata
            self.placements.append(position)
        self.blocks.append(block^)

    def finish(mut self, strict_f32: Bool, refuse_double: String) raises:
        if strict_f32 and (self.objects or self.booleans or self.common_kind != "f" or self.common_size != 4):
            raise Error("TypeError: Scaler input must have dtype float32")
        self.double_round = self.objects or (self.booleans and self.numbers) or (self.common_kind == "f" and self.common_size == 8)
        if self.double_round and refuse_double.byte_length() > 0:
            raise Error(refuse_double)
        self.seen = List[Bool](length=self.cols, fill=False)

    def column(mut self, block: Int, column: Int) raises -> Int:
        var destination = self.placements[self.blocks[block].positions + column]
        if destination < 0 or destination >= self.cols or self.seen[destination]:
            raise Error("pandas block column positions overlap or exceed the matrix")
        self.seen[destination] = True
        return destination

    def numeric_block(mut self, index: Int, destination: Int) raises:
        var block = self.blocks[index].copy()
        var kind = cast_in_kind(block.kind, block.itemsize, "pandas input")
        var output = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=destination)
        for col in range(block.cols):  # driver-loop: over pandas columns; each scatter spans parallel row tasks
            var dest = self.column(index, col)
            pd_scatter_numeric(block.address, block.itemsize, block.native, kind,
                block.row_stride, block.col_stride, col, dest, self.rows, self.cols,
                self.double_round, output, self.allow_nonfinite)


struct PandasObjectWriter(Copyable, Movable):
    var source: Int
    var stride: Int
    var destination: Int
    var cols: Int
    var column: Int
    var allow_nonfinite: Bool

    def __init__(out self, block: PandasBlock, col: Int, destination: Int,
        columns: Int, column: Int, allow_nonfinite: Bool):
        self.source = block.address + col * block.col_stride
        self.stride = block.row_stride
        self.destination = destination
        self.cols = columns
        self.column = column
        self.allow_nonfinite = allow_nonfinite

    def object_address(self, row: Int) -> Int:
        return self.source + row * self.stride

    def put(self, row: Int, number: Float64) raises:
        var bits = f64_to_f32_bits(bitcast[DType.uint64](number))
        if not self.allow_nonfinite and (bits & UInt32(0x7F800000)) == UInt32(0x7F800000):
            raise Error("Input contains NaN or infinity")
        MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=self.destination)[row * self.cols + self.column] = bitcast[DType.float32](bits)
