# SPDX-License-Identifier: Apache-2.0
"""Direct typed pandas-plan checks; no pandas objects or Python runtime.

Expected dtype joins, output cells and IEEE words are literal fixtures, not
answers computed by the product's promotion/cast/scatter helpers. Small cases
run independently on the existing host pool; production row tasks are unchanged.
Borrowed data stays owned by named Mojo Lists until the last pointer access.
"""
from std.memory import bitcast
from core.pandas_input_plan import (
    PandasBlock, PandasInputPlan, PandasObjectWriter, pandas_block_count,
    pandas_nullable_check, pandas_positions_slice, pandas_positions_array,
)

comptime PANDAS_PLAN_FIXTURES = 111


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _typecode(kind: String, width: Int) -> Int:
    if kind == "b":
        return 0
    if kind == "i":
        return 1
    if kind == "u":
        return 2
    if kind == "O":
        return 17
    return 23 if width == 2 else (11 if width == 4 else 12)


def _block(address: Int, rows: Int, cols: Int, width: Int, code: Int,
    kind: String, row_stride: Int, col_stride: Int, native: Bool = True) raises -> PandasBlock:
    var shape: List[Int] = [cols, rows]
    var strides: List[Int] = [col_stride, row_stride]
    return PandasBlock(address, shape, strides, width, code, kind, native, False, -1)


def _words(actual: List[Float32], expected: List[UInt32]) raises:
    require(len(actual) == len(expected) + 2, "output guard extent")
    require(actual[0] == -999 and actual[len(actual) - 1] == -999, "output guard overwritten")
    for i in range(len(expected)):
        require(bitcast[DType.uint32](actual[i + 1]) == expected[i], "output word " + String(i))


def _promotion(fixture_id: Int) raises:
    # Each pair also runs in reverse order. Positions are metadata, never data.
    var left: List[String] = ["b", "b", "f", "f", "f", "f", "f", "f", "i", "i", "i", "i", "u", "i", "u", "f", "f", "b"]
    var right: List[String] = ["b", "f", "i", "i", "i", "i", "i", "u", "u", "u", "u", "u", "i", "i", "u", "f", "f", "i"]
    var lw: List[Int] = [1, 1, 2, 2, 2, 4, 4, 8, 2, 1, 4, 8, 8, 2, 1, 2, 4, 1]
    var rw: List[Int] = [1, 4, 1, 2, 4, 2, 4, 8, 1, 1, 4, 8, 4, 8, 2, 4, 4, 8]
    var expected_kind: List[String] = ["b", "f", "f", "f", "f", "f", "f", "f", "i", "i", "i", "f", "f", "i", "u", "f", "f", "i"]
    var expected_width: List[Int] = [1, 4, 2, 4, 8, 4, 8, 8, 2, 2, 8, 8, 8, 8, 2, 4, 4, 8]
    var rounded: List[Bool] = [False, True, False, False, True, False, True, True, False, False, False, True, True, False, False, False, False, True]
    for reverse in range(2):  # dependent order-invariance check of one dtype pair
        var plan = PandasInputPlan(False)
        var ak = left[fixture_id] if reverse == 0 else right[fixture_id]
        var bk = right[fixture_id] if reverse == 0 else left[fixture_id]
        var aw = lw[fixture_id] if reverse == 0 else rw[fixture_id]
        var bw = rw[fixture_id] if reverse == 0 else lw[fixture_id]
        var a = _block(0, 3, 1, aw, _typecode(ak, aw), ak, aw, 0)
        var b = _block(0, 3, 1, bw, _typecode(bk, bw), bk, bw, 0)
        var first: List[Int] = [1]
        var second: List[Int] = [0]
        plan.add_block(a^, first, ak, False)
        plan.add_block(b^, second, bk, False)
        first[0] = 99
        second[0] = 99
        plan.finish(False, "")
        require(plan.rows == 3 and plan.cols == 2, "promoted plan shape")
        require(plan.common_kind == expected_kind[fixture_id] and plan.common_size == expected_width[fixture_id], "dtype promotion")
        require(plan.double_round == rounded[fixture_id], "dtype conversion policy")
        require(plan.column(0, 0) == 1 and plan.column(1, 0) == 0, "plan owns placement metadata")


def _strides(fixture_id: Int) raises:
    var values: List[Float32] = [1, 2, 3, 11, 12, 13]
    var original = values.copy()
    var reverse_rows = fixture_id == 1 or fixture_id == 3
    var reverse_cols = fixture_id == 2 or fixture_id == 3
    var row_stride = -4 if reverse_rows else (0 if fixture_id == 4 else 4)
    var col_stride = -12 if reverse_cols else 12
    var base = Int(values.unsafe_ptr()) + (8 if reverse_rows else 0) + (12 if reverse_cols else 0)
    var shape: List[Int] = [2, 3]
    var strides: List[Int] = [col_stride, row_stride]
    var block = PandasBlock(base, shape, strides, 4, 11, "f", True, False, -1)
    shape[0] = 99
    strides[0] = 999
    var positions: List[Int] = [1, 0]
    var plan = PandasInputPlan(False)
    plan.add_block(block^, positions, "f", False)
    plan.finish(True, "")
    require(not plan.double_round, "strict float32 plan policy")
    var dst = List[Float32](length=8, fill=-999)
    plan.numeric_block(0, Int(dst.unsafe_ptr()) + 4)
    var expected: List[Float32] = [11, 1, 12, 2, 13, 3]
    if fixture_id == 1:
        expected = [13, 3, 12, 2, 11, 1]
    elif fixture_id == 2:
        expected = [1, 11, 2, 12, 3, 13]
    elif fixture_id == 3:
        expected = [3, 13, 2, 12, 1, 11]
    elif fixture_id == 4:
        expected = [11, 1, 11, 1, 11, 1]
    require(dst[0] == -999 and dst[7] == -999, "scatter output guards")
    for i in range(6):
        require(dst[i + 1] == expected[i], "signed/zero stride output order")
        require(values[i] == original[i], "scatter mutated source")
    values[0] = 77
    require(dst[1] == expected[0] and dst[2] == expected[1], "prepared output aliases source")


def _integer_rounding(fixture_id: Int) raises:
    # 2^62 + 2^38 +/- 1 straddles a float32 midpoint. Classical integer
    # conversion goes through binary64 even for a homogeneous integer frame.
    var midpoint = (Int64(1) << 62) + (Int64(1) << 38)
    var values: List[Int64] = [midpoint + 1, -midpoint - 1, midpoint - 1, -midpoint + 1]
    var zeros: List[Float32] = [0, 0, 0, 0]
    var unsigned: List[UInt64] = [0, 0, 0, 0]
    var bools: List[UInt8] = [0, 1, 0, 1]
    var plan = PandasInputPlan(False)
    var block = _block(Int(values.unsafe_ptr()), 4, 1, 8, 7, "i", 8, 0)
    var position: List[Int] = [0]
    plan.add_block(block^, position, "i", False)
    var cols = 1 if fixture_id == 0 else 2
    if fixture_id != 0:
        var kind = String("f") if fixture_id == 1 else (String("u") if fixture_id == 2 else String("b"))
        var width = 4 if fixture_id == 1 else (8 if fixture_id == 2 else 1)
        var address = Int(zeros.unsafe_ptr()) if fixture_id == 1 else (Int(unsigned.unsafe_ptr()) if fixture_id == 2 else Int(bools.unsafe_ptr()))
        var other = _block(address, 4, 1, width, _typecode(kind, width), kind, width, 0)
        var second: List[Int] = [1]
        plan.add_block(other^, second, kind, fixture_id == 3)
    plan.finish(False, "")
    require(plan.double_round == (fixture_id != 0), "integer mix promotion policy")
    var dst = List[Float32](length=4 * cols + 2, fill=-999)
    plan.numeric_block(0, Int(dst.unsafe_ptr()) + 4)
    if fixture_id != 0:
        plan.numeric_block(1, Int(dst.unsafe_ptr()) + 4)
    require(dst[0] == -999 and dst[len(dst) - 1] == -999, "integer output guards")
    for row in range(4):
        var expected = UInt32(0x5E800000) if row % 2 == 0 else UInt32(0xDE800000)
        require(bitcast[DType.uint32](dst[1 + row * cols]) == expected, "double-rounding midpoint word")
        if cols == 2:
            require(dst[2 + row * cols] == Float32(row % 2 if fixture_id == 3 else 0), "mixed second column")
    require(values[0] == midpoint + 1 and values[1] == -midpoint - 1 and values[2] == midpoint - 1 and values[3] == -midpoint + 1, "integer source mutated")
    for row in range(4):
        require(zeros[row] == 0 and unsigned[row] == 0 and bools[row] == UInt8(row % 2), "mixed source mutated")


def _endian(fixture_id: Int) raises:
    var bytes: List[UInt8] = [0x3F, 0xC0, 0, 0, 0xC0, 0, 0, 0, 0x3E, 0x80, 0, 0]
    var original = bytes.copy()
    var block = _block(Int(bytes.unsafe_ptr()) + (8 if fixture_id else 0), 3, 1, 4, 11, "f", -4 if fixture_id else 4, 0, False)
    var plan = PandasInputPlan(False)
    var positions: List[Int] = [0]
    plan.add_block(block^, positions, "f", False)
    plan.finish(True, "")
    var dst = List[Float32](length=5, fill=-999)
    plan.numeric_block(0, Int(dst.unsafe_ptr()) + 4)
    var expected: List[UInt32] = [0x3FC00000, 0xC0000000, 0x3E800000]
    if fixture_id:
        expected = [0x3E800000, 0xC0000000, 0x3FC00000]
    _words(dst, expected)
    for i in range(12):
        require(bytes[i] == original[i], "endian source mutated")


def _slice(fixture_id: Int) raises:
    var starts: List[Int] = [0, 4, 0, 0, 5, 0, 5]
    var stops: List[Int] = [5, -1, 0, 5, 0, 6, -1]
    var steps: List[Int] = [2, -2, 1, -1, 1, 2, -2]
    var expected: List[Int] = [0, 2, 4]
    if fixture_id == 1:
        expected = [4, 2, 0]
    elif fixture_id >= 2 and fixture_id <= 4:
        expected = List[Int]()
    elif fixture_id == 6:
        expected = [5, 3, 1]
    var actual = pandas_positions_slice(starts[fixture_id], stops[fixture_id], steps[fixture_id])
    require(len(actual) == len(expected), "placement slice length")
    for i in range(len(expected)):
        require(actual[i] == expected[i], "placement slice order")


def _placement_array(fixture_id: Int) raises:
    var slots: List[Int64] = [2, -99, 0, -99, 1]
    var bytes: List[UInt8] = [0, 2, 0, 0, 0, 1]
    var reversed = fixture_id % 2 == 1
    var native = fixture_id < 2
    var base = Int(slots.unsafe_ptr()) + (32 if reversed else 0) if native else Int(bytes.unsafe_ptr()) + (4 if reversed else 0)
    var stride = (-16 if reversed else 16) if native else (-2 if reversed else 2)
    var actual = pandas_positions_array(1, "i", 3, base, 8 if native else 2, 7 if native else 3, stride, native)
    var expected: List[Int] = [2, 0, 1]
    if reversed:
        expected = [1, 0, 2]
    require(len(actual) == 3, "placement array extent")
    for i in range(3):
        require(actual[i] == expected[i], "placement array endian/stride order")
    slots[0] = 88
    bytes[1] = 88
    require(actual[0] == expected[0] and actual[2] == expected[2], "placement result aliases source")


def _refusal(fixture_id: Int) raises:
    var refused = False
    var message = String()
    var expected = String()
    try:
        if fixture_id == 0:
            expected = "invalid block metadata"
            pandas_block_count(0, False)
        elif fixture_id == 1:
            expected = "invalid block metadata"
            pandas_block_count(2, True)
        elif fixture_id >= 2 and fixture_id <= 6:
            var shape: List[Int] = [2, 3]
            var strides: List[Int] = [12, 4]
            expected = "dimensions differ or are empty"
            if fixture_id == 2:
                shape = [1, 2, 3]
                expected = "rank must be one or two"
            elif fixture_id == 3:
                shape = [2, 0]
            elif fixture_id == 4:
                shape = [0, 3]
            var block = PandasBlock(0, shape, strides, 4, 11, "f", True, fixture_id == 5, 4 if fixture_id == 6 else -1)
            _ = block
        elif fixture_id == 7:
            # Product names the Python-compatible range argument, not a custom error.
            expected = "arg 3 must not be zero"
            var positions = pandas_positions_slice(0, 2, 0)
            _ = positions
        elif fixture_id == 8 or fixture_id == 9:
            expected = "placement requires native integer storage"
            var positions = pandas_positions_array(2 if fixture_id == 8 else 1, "i" if fixture_id == 8 else "u", 0, 0, 8, 7, 8, True)
            _ = positions
        else:
            var plan = PandasInputPlan(False)
            var kind = String("c") if fixture_id == 11 else String("i")
            var block = _block(0, 1, 1, 8, 7, kind, 8, 0)
            var positions: List[Int] = [0]
            if fixture_id == 10:
                expected = "column positions differ"
                positions = List[Int]()
            elif fixture_id == 11:
                expected = "Complex data not supported"
            plan.add_block(block^, positions, kind, fixture_id == 13)
            if fixture_id == 12:
                expected = "Scaler input must have dtype float32"
                plan.finish(True, "")
            elif fixture_id == 13:
                expected = "no double conversion admitted"
                plan.finish(False, expected)
    except error:
        refused = True
        message = String(error)
    require(refused and expected.byte_length() > 0 and message.find(expected) >= 0, "wrong/absent metadata refusal: " + message)


def _columns(fixture_id: Int) raises:
    var input: List[Float32] = [1, 2, 3, 4]
    var block = _block(Int(input.unsafe_ptr()), 2, 2, 4, 11, "f", 4, 8)
    var positions: List[Int] = [0, 0]
    if fixture_id == 1:
        positions = [-1, 1]
    elif fixture_id == 2:
        positions = [2, 1]
    elif fixture_id == 3:
        positions = [1, 0]
    var plan = PandasInputPlan(False)
    plan.add_block(block^, positions, "f", False)
    plan.finish(False, "")
    var dst = List[Float32](length=6, fill=-999)
    var refused = False
    try:
        plan.numeric_block(0, Int(dst.unsafe_ptr()) + 4)
        if fixture_id == 3:
            plan.numeric_block(0, Int(dst.unsafe_ptr()) + 4)
    except error:
        refused = String(error).find("column positions overlap or exceed the matrix") >= 0
    require(refused, "column overlap/bounds/replay accepted")
    require(dst[0] == -999 and dst[5] == -999, "refused placement escaped output")
    if fixture_id == 1 or fixture_id == 2:
        for i in range(1, 5):
            require(dst[i] == -999, "invalid first column wrote output")
    for i in range(4):
        require(input[i] == Float32(i + 1), "refused placement mutated source")


def _nullable(fixture_id: Int) raises:
    var mask: List[UInt8] = [99, 0, 0, 0, 99]
    if fixture_id == 2 or fixture_id == 3:
        mask[3] = 1
    var original = mask.copy()
    var reversed = fixture_id == 1 or fixture_id == 3
    var address = Int(mask.unsafe_ptr()) + (3 if reversed else 1)
    var refused = False
    var message = String()
    try:
        pandas_nullable_check(2 if fixture_id == 4 else 1, 2 if fixture_id == 5 else 1,
            4 if fixture_id == 6 else 3, 3, 1 if fixture_id == 7 else 0,
            address, -1 if reversed else 1, 1, True)
    except error:
        refused = True
        message = String(error)
    require(refused == (fixture_id >= 2), "nullable admission outcome")
    if fixture_id >= 2:
        var expected = String("missing nullable value") if fixture_id < 4 else String("nullable storage metadata differs")
        require(message.find(expected) >= 0, "nullable refusal reason")
    for i in range(5):
        require(mask[i] == original[i], "nullable mask mutated")


def _object_scalar(fixture_id: Int) raises:
    var word_id = fixture_id // 2
    var allow = fixture_id % 2 == 1
    # Literal binary64 values and independently specified binary32 answers:
    # signed zero, exact integers, two tie parities, just above a tie,
    # subnormal underflow/tie, largest finite f32, overflow, infinities/NaNs.
    var source: List[UInt64] = [0, 0x8000000000000000, 0x3FF0000000000000,
        0xC004000000000000, 0x3FF0000010000000, 0x3FF0000030000000,
        0x3FF0000010000001, 0x3690000000000000, 0x3690000000000001,
        0xB690000000000000, 0x47EFFFFFE0000000, 0x47F0000000000000,
        0x7FF0000000000000, 0xFFF0000000000000, 0x7FF8000000000001,
        0xFFF8000000000001, 0x36A0000000000000, 0x1]
    var expected: List[UInt32] = [0, 0x80000000, 0x3F800000, 0xC0200000,
        0x3F800000, 0x3F800002, 0x3F800001, 0, 1, 0x80000000,
        0x7F7FFFFF, 0x7F800000, 0x7F800000, 0xFF800000,
        0x7FC00000, 0x7FC00000, 1, 0]
    var slots: List[UInt64] = [10, 20, 30, 40, 50, 60]
    var block = _block(Int(slots.unsafe_ptr()) + 40, 3, 2, 8, 17, "O", -8, -24)
    var plan = PandasInputPlan(allow)
    var positions: List[Int] = [2, 0]
    plan.add_block(block.copy(), positions, "O", False)
    var extra = _block(0, 3, 1, 4, 11, "f", 4, 0)
    var middle: List[Int] = [1]
    plan.add_block(extra^, middle, "f", False)
    plan.finish(False, "")
    require(plan.objects and plan.double_round, "object scalar promotion policy")
    var col = plan.column(0, 1)
    require(col == 0, "object column order")
    var dst = List[Float32](length=11, fill=-999)
    var writer = PandasObjectWriter(block, 1, Int(dst.unsafe_ptr()) + 4, 3, col, allow)
    require(writer.object_address(0) == Int(slots.unsafe_ptr()) + 16 and writer.object_address(2) == Int(slots.unsafe_ptr()), "object slot signed strides")
    var refused = False
    var before = source[word_id]
    try:
        writer.put(1, bitcast[DType.float64](before))
    except error:
        refused = String(error).find("Input contains NaN or infinity") >= 0
    var nonfinite = word_id >= 11 and word_id <= 15
    require(refused == (nonfinite and not allow), "object finite policy")
    for i in range(11):
        if i == 4 and not refused:
            require(bitcast[DType.uint32](dst[i]) == expected[word_id], "object narrowing word")
        else:
            require(dst[i] == -999, "object write escaped cell or wrote before refusal")
    require(source[word_id] == before, "object scalar source mutated")
    for i in range(6):
        require(slots[i] == UInt64((i + 1) * 10), "object address slots mutated")


def _numeric_finite(fixture_id: Int) raises:
    var allow = fixture_id % 2 == 1
    var specials: List[UInt32] = [0x7FC00000, 0x7F800000, 0xFF800000]
    var value = specials[fixture_id // 2]
    var source: List[UInt32] = [0x3F800000, value, 0x40000000]
    var block = _block(Int(source.unsafe_ptr()), 3, 1, 4, 11, "f", 4, 0)
    var positions: List[Int] = [0]
    var plan = PandasInputPlan(allow)
    plan.add_block(block^, positions, "f", False)
    plan.finish(True, "")
    var dst = List[Float32](length=5, fill=-999)
    var refused = False
    try:
        plan.numeric_block(0, Int(dst.unsafe_ptr()) + 4)
    except error:
        refused = String(error).find("Input contains NaN or infinity") >= 0
    require(refused == (not allow), "numeric finite policy")
    # Numeric scatter fills the fresh private output before aggregate refusal.
    # The object writer above instead refuses before its individual store.
    _words(dst, source)
    require(source[0] == 0x3F800000 and source[1] == value and source[2] == 0x40000000, "numeric nonfinite source mutated")


def _series_nullable_policy(fixture_id: Int) raises:
    if fixture_id < 2:
        # Nullable DataFrame storage has object-style promotion even when all
        # columns are boolean; plain homogeneous bool storage does not.
        var source: List[UInt8] = [0, 1, 0]
        var block = _block(Int(source.unsafe_ptr()), 3, 1, 1, 0, "b", 1, 0)
        var positions: List[Int] = [0]
        var plan = PandasInputPlan(False)
        plan.add_block(block^, positions, "b", fixture_id == 1)
        plan.finish(False, "")
        require(plan.double_round == (fixture_id == 1) and plan.objects == (fixture_id == 1), "nullable boolean promotion policy")
        var dst = List[Float32](length=5, fill=-999)
        plan.numeric_block(0, Int(dst.unsafe_ptr()) + 4)
        var expected: List[UInt32] = [0, 0x3F800000, 0]
        _words(dst, expected)
        require(source[0] == 0 and source[1] == 1 and source[2] == 0, "boolean source mutated")
    else:
        pandas_block_count(1, True)
        pandas_block_count(2, False)
        var source: List[Float64] = [1.5, -2, 0.25]
        var shape: List[Int] = [3]
        var strides: List[Int] = [-8]
        var block = PandasBlock(Int(source.unsafe_ptr()) + 16, shape, strides, 8, 12, "f", True, True, 3)
        var positions: List[Int] = [0]
        var plan = PandasInputPlan(False)
        plan.add_block(block^, positions, "f", False)
        plan.finish(False, "")
        require(plan.rows == 3 and plan.cols == 1 and plan.double_round, "series plan shape/policy")
        var dst = List[Float32](length=5, fill=-999)
        plan.numeric_block(0, Int(dst.unsafe_ptr()) + 4)
        var expected: List[UInt32] = [0x3E800000, 0xC0000000, 0x3FC00000]
        _words(dst, expected)
        require(source[0] == 1.5 and source[1] == -2 and source[2] == 0.25, "series source mutated")


def run_pandas_plan_fixture(fixture_id: Int) raises:
    if fixture_id < 18:
        _promotion(fixture_id)
    elif fixture_id < 23:
        _strides(fixture_id - 18)
    elif fixture_id < 27:
        _integer_rounding(fixture_id - 23)
    elif fixture_id < 29:
        _endian(fixture_id - 27)
    elif fixture_id < 36:
        _slice(fixture_id - 29)
    elif fixture_id < 40:
        _placement_array(fixture_id - 36)
    elif fixture_id < 54:
        _refusal(fixture_id - 40)
    elif fixture_id < 58:
        _columns(fixture_id - 54)
    elif fixture_id < 66:
        _nullable(fixture_id - 58)
    elif fixture_id < 102:
        _object_scalar(fixture_id - 66)
    elif fixture_id < 108:
        _numeric_finite(fixture_id - 102)
    else:
        _series_nullable_policy(fixture_id - 108)
