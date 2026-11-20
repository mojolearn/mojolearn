# SPDX-License-Identifier: Apache-2.0
"""Literal native sequence format/shape/promotion contracts, no Python objects."""
from std.memory import bitcast
from core.borrowed_buffer import BorrowedBufferLayout
from core.sequential_input_plan import SequenceInputPlan, SequenceBufferFormat, sequence_buffer_shape, sequence_array_typestr, sequence_write

comptime SEQUENCE_METADATA_FIXTURES = 78


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _format(fixture_id: Int) raises:
    var formats: List[String] = ["?", "b", "B", "h", "H", "i", "I", "l", "L", "q", "Q", "n", "N", "e", "f", "d", ">i", "!H", "<f", "@q", "=d"]
    var widths: List[Int] = [1, 1, 1, 2, 2, 4, 4, 8, 8, 8, 8, 8, 8, 2, 4, 8, 4, 2, 4, 8, 8]
    var kinds: List[String] = ["b", "i", "u", "i", "u", "i", "u", "i", "u", "i", "u", "i", "u", "f", "f", "f", "i", "u", "f", "i", "f"]
    var parsed = SequenceBufferFormat(formats[fixture_id], widths[fixture_id])
    require(parsed.kind == kinds[fixture_id] and parsed.itemsize == widths[fixture_id], "numeric buffer kind/width")
    require(parsed.native == (fixture_id != 16 and fixture_id != 17), "buffer endian policy")
    var prefix = String(">") if fixture_id == 16 or fixture_id == 17 else (String("<") if fixture_id == 18 else String())
    require(parsed.prefix == prefix, "buffer format prefix")


def _format_refusal(fixture_id: Int) raises:
    var formats: List[String] = ["O", "T{f:x}", "2f", "x", "", "f", "i", "?"]
    var widths: List[Int] = [8, 4, 8, 1, 1, 1, 3, 0]
    var expected = String("numeric buffer") if fixture_id < 5 else String("supported numeric scalars")
    var refused = False
    try:
        var parsed = SequenceBufferFormat(formats[fixture_id], widths[fixture_id])
        _ = parsed
    except error:
        refused = String(error).find(expected) >= 0
    require(refused, "unsupported numeric buffer accepted")


def _buffer_shape(fixture_id: Int) raises:
    var ranks: List[Int] = [1, 2, 3, 3, 8, 8]
    var c_orders: List[Bool] = [False, False, False, True, False, True]
    var samples: List[Bool] = [True, True, True, True, False, True]
    var refused = False
    try:
        sequence_buffer_shape(ranks[fixture_id], c_orders[fixture_id], samples[fixture_id])
    except error:
        refused = String(error).find("more than two dimensions must be C-contiguous") >= 0
    require(refused == (fixture_id == 2), "sample buffer layout admission")


def _typestr(fixture_id: Int) raises:
    var kinds: List[String] = ["b", "i", "f", "u", "i", "f"]
    var widths: List[Int] = [1, 8, 8, 8, 4, 4]
    var expected: List[String] = ["<i1", "<i8", "<f8"]
    var refused = False
    var result = String()
    var message = String()
    try:
        result = sequence_array_typestr(kinds[fixture_id], widths[fixture_id], "fixture input")
    except error:
        refused = True
        message = String(error)
    require(refused == (fixture_id >= 3), "sequence Array dtype admission")
    if fixture_id < 3:
        require(result == expected[fixture_id], "sequence Array dtype")
    else:
        var needle = String("integers beyond the int64 range") if fixture_id == 3 else String("unsupported numbers")
        require(message.find("fixture input") >= 0 and message.find(needle) >= 0, "sequence Array refusal name/reason")


def _shape_refusal(fixture_id: Int) raises:
    var plan = SequenceInputPlan()
    var expected = String("rectangular")
    var refused = False
    try:
        if fixture_id == 0:
            expected = "nonempty"
            plan.container(0, 0)
        elif fixture_id == 1:
            expected = "rank at most eight"
            plan.container(8, 1)
        elif fixture_id == 2:
            plan.container(0, 2)
            plan.container(1, 2)
            plan.scalar(2, 1, "i", 8)
            plan.container(1, 3)
        elif fixture_id == 3:
            plan.container(0, 2)
            plan.scalar(1, 1, "i", 8)
            plan.container(1, 2)
        elif fixture_id == 4:
            plan.container(0, 2)
            plan.container(1, 2)
            plan.scalar(2, 1, "i", 8)
            plan.scalar(1, 2, "i", 8)
        elif fixture_id == 5:
            expected = "nonempty"
            var dims: List[Int] = [0]
            plan.leaf(0, dims)
        elif fixture_id == 6:
            expected = "rank at most eight"
            var dims: List[Int] = [1, 1]
            plan.leaf(7, dims)
        elif fixture_id == 7:
            plan.container(0, 2)
            var dims: List[Int] = [3]
            plan.leaf(1, dims)
            dims[0] = 4
            plan.leaf(1, dims)
        elif fixture_id == 8:
            plan.container(0, 2)
            plan.container(1, 3)
            plan.scalar(1, 1, "i", 8)
        else:
            expected = "element count exceeds native storage"
            # Metadata-only overflow: layouts refer to the sentinel address,
            # but no data read/write is attempted for the impossible span.
            var sentinel: List[UInt8] = [0]
            var dims: List[Int] = [9223372036854775807]
            var strides: List[Int] = [1]
            var suboffsets: List[Int] = [-1]
            var layout = BorrowedBufferLayout(Int(sentinel.unsafe_ptr()), 1, True, dims^, strides^, suboffsets^)
            plan.container(0, 2)
            plan.array(1, layout.copy(), "u", True)
            plan.array(1, layout.copy(), "u", True)
            _ = sentinel
    except error:
        refused = String(error).find(expected) >= 0
    require(refused, "wrong/absent sequence shape refusal")


def _scalar_pair(fixture_id: Int) raises:
    var plan = SequenceInputPlan()
    plan.container(0, 2)
    var kinds_a: List[String] = ["b", "i", "i", "i", "i", "f", "f", "f", "f", "u", "i", "f"]
    var kinds_b: List[String] = ["b", "u", "u", "u", "u", "b", "i", "i", "i", "u", "i", "f"]
    var wa: List[Int] = [1, 1, 2, 4, 8, 2, 2, 2, 4, 8, 8, 8]
    var wb: List[Int] = [1, 1, 1, 4, 8, 1, 1, 2, 4, 8, 8, 8]
    var a: List[UInt64] = [1, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFF8000,
        0xFFFFFFFF80000000, 0x8000000000000000, 0x3FF8000000000000,
        0x3FF8000000000000, 0x3FF8000000000000, 0x3FE0000000000000,
        0xFFFFFFFFFFFFFFFF, 0x8000000000000000, 0x8000000000000000]
    var b: List[UInt64] = [0, 255, 255, 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF,
        1, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFF8000, 2147483647,
        0xFFFFFFFFFFFFFFFE, 0x7FFFFFFFFFFFFFFF, 0x3FF0000000000000]
    var kind: List[String] = ["b", "i", "i", "i", "f", "f", "f", "f", "f", "u", "i", "f"]
    var width: List[Int] = [1, 2, 2, 8, 8, 2, 2, 4, 8, 8, 8, 8]
    plan.scalar(1, a[fixture_id], kinds_a[fixture_id], wa[fixture_id])
    plan.scalar(1, b[fixture_id], kinds_b[fixture_id], wb[fixture_id])
    require(plan.count == 2 and plan.kind == kind[fixture_id] and plan.itemsize == width[fixture_id], "scalar promotion")
    require(len(plan.segments) == 1 and plan.segments[0].source == -1 and plan.segments[0].start == 0 and plan.segments[0].count == 2, "compact scalar segment")
    require(len(plan.words) == 2 and len(plan.kinds) == 2 and len(plan.arrays) == 0, "compact scalar storage")
    var dst = List[UInt64](length=4, fill=0xABABABABABABABAB)
    sequence_write(plan, Int(dst.unsafe_ptr()) + 8)
    var expected_a = a[fixture_id]
    var expected_b = b[fixture_id]
    if fixture_id == 4:
        expected_a = 0xC3E0000000000000  # -2^63
        expected_b = 0x43F0000000000000  # uint64 MAX rounds to 2^64 in f64
    elif fixture_id == 5 or fixture_id == 6:
        expected_a = 0x3E00
        expected_b = 0x3C00 if fixture_id == 5 else 0xBC00
    elif fixture_id == 7:
        expected_a = 0x3FC00000
        expected_b = 0xC7000000
    elif fixture_id == 8:
        expected_b = bitcast[DType.uint64](Float64(2147483647))
    var bytes = Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=Int(dst.unsafe_ptr()))
    for i in range(32):
        var expected = UInt8(0xAB)
        if i >= 8 and i < 8 + 2 * width[fixture_id]:
            var element = (i - 8) // width[fixture_id]
            var shift = ((i - 8) % width[fixture_id]) * 8
            var word = expected_a if element == 0 else expected_b
            expected = UInt8((word >> UInt64(shift)) & 255)
        require(bytes[i] == expected, "scalar output word/guard")
    _ = dst


def _float32(fixture_id: Int) raises:
    var source: List[UInt64] = [0, 0x8000000000000000, 0x3FF0000010000000,
        0x3FF0000030000000, 0x3FF0000010000001, 0x3690000000000000,
        0x3690000000000001, 0x47EFFFFFE0000000, 0x47F0000000000000,
        0x7FF8000000000001, 0x4000004000000001, 0xBFFFFFBFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF, 0x8000000000000000]
    var expected: List[UInt32] = [0, 0x80000000, 0x3F800000,
        0x3F800002, 0x3F800001, 0, 1, 0x7F7FFFFF,
        0x7F800000, 0x7FC00000, 0x5E800000, 0xDE800000,
        0x5F800000, 0xDF000000]
    var kind = String("f") if fixture_id < 10 else (String("u") if fixture_id == 12 else String("i"))
    var plan = SequenceInputPlan()
    plan.scalar(0, source[fixture_id], kind, 8)
    var dst: List[UInt32] = [0xDEADBEEF, 0, 0xDEADBEEF]
    sequence_write(plan, Int(dst.unsafe_ptr()) + 4, True)
    require(dst[0] == 0xDEADBEEF and dst[2] == 0xDEADBEEF and dst[1] == expected[fixture_id], "forced f32 rounding/guard")


def _maximum_rank() raises:
    var plan = SequenceInputPlan()
    for depth in range(8):  # dependent construction of one nested metadata tree
        plan.container(depth, 1)
    plan.scalar(8, 7, "i", 8)
    require(len(plan.shape) == 8 and plan.leaf_depth == 8 and plan.count == 1, "rank-eight admission")
    var dst: List[Int64] = [-99, 0, -99]
    sequence_write(plan, Int(dst.unsafe_ptr()) + 8)
    require(dst[0] == -99 and dst[1] == 7 and dst[2] == -99, "rank-eight scalar output")


def run_metadata_fixture(fixture_id: Int) raises:
    if fixture_id < 21:
        _format(fixture_id)
    elif fixture_id < 29:
        _format_refusal(fixture_id - 21)
    elif fixture_id < 35:
        _buffer_shape(fixture_id - 29)
    elif fixture_id < 41:
        _typestr(fixture_id - 35)
    elif fixture_id < 51:
        _shape_refusal(fixture_id - 41)
    elif fixture_id < 63:
        _scalar_pair(fixture_id - 51)
    elif fixture_id < 77:
        _float32(fixture_id - 63)
    else:
        _maximum_rank()
