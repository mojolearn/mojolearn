# SPDX-License-Identifier: Apache-2.0
"""Direct native checkpoint conversion contracts, independent literal words.

Conversion chunks may complete independently in private scratch. The earliest
invalid input index determines the refusal; source and destination bounds stay
protected. Independent formats/bounds/strides run across the existing host pool. The public Python
checkpoint and error-class tests remain distinct from these raw typed spans.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from core.cast_in import CAST_KIND_FLOAT, CAST_KIND_SIGNED, CAST_KIND_UNSIGNED, CAST_KIND_BOOL
from core.optimizer_state_impl import optimizer_state_cast

comptime STATE_INPUT_FIXTURES = 39


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _integer_scalar(fixture_id: Int) raises:
    # Fractional values truncate toward zero; the negative admissible open
    # bound is -2147483649, not -2147483648. All expectations are literals.
    var numbers: List[Float64] = [0, -0.0, 1.75, -1.75, 2147483647,
        2147483647.75, -2147483648, -2147483648.75,
        2147483648, -2147483649,
        bitcast[DType.float64](UInt64(0x7FF8000000000001)),
        bitcast[DType.float64](UInt64(0x7FF0000000000000)),
        bitcast[DType.float64](UInt64(0xFFF0000000000000))]
    var expected: List[UInt32] = [0, 0, 1, 0xFFFFFFFF, 0x7FFFFFFF,
        0x7FFFFFFF, 0x80000000, 0x80000000]
    var source: List[Float64] = [numbers[fixture_id]]
    var original = bitcast[DType.uint64](source[0])
    var dst: List[UInt32] = [0xAABBCCDD, 0xAABBCCDD, 0xAABBCCDD]
    var output = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=Int(dst.unsafe_ptr()) + 4)
    var message = String()
    var refused = False
    try:
        optimizer_state_cast(Int(source.unsafe_ptr()), 8, 1, 8, CAST_KIND_FLOAT, True, output, True)
    except error:
        refused = True
        message = String(error)
    require(refused == (fixture_id >= 8), "int32 cast admission")
    if refused:
        var wanted = String("cannot convert float NaN to integer") if fixture_id == 10 else String("buf_initialized value does not fit int32")
        require(message.find(wanted) >= 0 and dst[1] == 0xAABBCCDD, "int32 refusal reason or pre-store mutation")
    else:
        require(dst[1] == expected[fixture_id], "int32 truncation/boundary word")
    require(dst[0] == 0xAABBCCDD and dst[2] == 0xAABBCCDD and bitcast[DType.uint64](source[0]) == original, "int32 cast escaped span or changed input")


def _float_scalar(fixture_id: Int) raises:
    var words: List[UInt64] = [0, 0x8000000000000000, 0x3FF0000010000000,
        0x3FF0000030000000, 0x3FF0000010000001, 0x3690000000000000,
        0x3690000000000001, 0x47EFFFFFE0000000, 0x47F0000000000000,
        0x7FF8000000000001, 0xFFF0000000000000]
    var expected: List[UInt32] = [0, 0x80000000, 0x3F800000, 0x3F800002,
        0x3F800001, 0, 1, 0x7F7FFFFF, 0x7F800000, 0x7FC00000, 0xFF800000]
    var source: List[UInt64] = [words[fixture_id]]
    var dst: List[UInt32] = [0xAABBCCDD, 0xAABBCCDD, 0xAABBCCDD]
    var output = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=Int(dst.unsafe_ptr()) + 4)
    optimizer_state_cast(Int(source.unsafe_ptr()), 8, 1, 8, CAST_KIND_FLOAT, True, output, False)
    require(dst[1] == expected[fixture_id], "moment float32 narrowing word")
    require(dst[0] == 0xAABBCCDD and dst[2] == 0xAABBCCDD and source[0] == words[fixture_id], "moment conversion guard/input")


def _span(fixture_id: Int) raises:
    var signed: List[Int16] = [-2, 3, -4]
    var unsigned: List[UInt16] = [0, 65535, 32768]
    var bools: List[UInt8] = [0, 1, 0]
    var half: List[UInt16] = [0x3E00, 0xC000, 0x3400]
    var endian: List[UInt8] = [0xFF, 0xFE, 0, 3, 0xFF, 0xFC]
    var wide_signed: List[Int64] = [4611686293305294849, -4611686293305294849, 0]
    var wide_unsigned: List[UInt64] = [0xFFFFFFFFFFFFFFFF, 0x8000000000000000, 0]
    var base = Int(signed.unsafe_ptr())
    var width = 2
    var stride = 2
    var kind = CAST_KIND_SIGNED
    var native = True
    var integer = fixture_id == 0 or fixture_id == 1 or fixture_id == 5
    var expected: List[UInt32] = [0xFFFFFFFE, 3, 0xFFFFFFFC]
    if fixture_id == 1:
        base += 4
        stride = -2
        expected = [0xFFFFFFFC, 3, 0xFFFFFFFE]
    elif fixture_id == 2:
        base = Int(unsigned.unsafe_ptr())
        kind = CAST_KIND_UNSIGNED
        expected = [0, 0x477FFF00, 0x47000000]
    elif fixture_id == 3:
        base = Int(bools.unsafe_ptr())
        width = 1
        stride = 1
        kind = CAST_KIND_BOOL
        expected = [0, 0x3F800000, 0]
    elif fixture_id == 4:
        base = Int(half.unsafe_ptr())
        kind = CAST_KIND_FLOAT
        expected = [0x3FC00000, 0xC0000000, 0x3E800000]
    elif fixture_id == 5:
        base = Int(endian.unsafe_ptr()) + 4
        stride = -2
        native = False
        expected = [0xFFFFFFFC, 3, 0xFFFFFFFE]
    elif fixture_id == 6:
        base = Int(wide_signed.unsafe_ptr())
        width = 8
        stride = 8
        expected = [0x5E800000, 0xDE800000, 0]
    elif fixture_id == 7:
        base = Int(wide_unsigned.unsafe_ptr())
        width = 8
        stride = 8
        kind = CAST_KIND_UNSIGNED
        expected = [0x5F800000, 0x5F000000, 0]
    elif fixture_id == 8:
        stride = 0
        expected = [0xC0000000, 0xC0000000, 0xC0000000]
    var dst = List[UInt32](length=5, fill=0xAABBCCDD)
    var output = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=Int(dst.unsafe_ptr()) + 4)
    optimizer_state_cast(base, stride, 3, width, kind, native, output, integer)
    require(dst[0] == 0xAABBCCDD and dst[4] == 0xAABBCCDD, "state span guards")
    for i in range(3):
        require(dst[i + 1] == expected[i], "state endian/stride/kind conversion")
    require(signed[0] == -2 and signed[1] == 3 and signed[2] == -4, "signed source mutated")
    require(unsigned[1] == 65535 and bools[1] == 1 and half[0] == 0x3E00 and endian[4] == 0xFF and endian[5] == 0xFC, "packed source mutated")
    require(wide_signed[0] == 4611686293305294849 and wide_unsigned[0] == 0xFFFFFFFFFFFFFFFF, "wide source mutated")
    _ = signed
    _ = unsigned
    _ = bools
    _ = half
    _ = endian
    _ = wide_signed
    _ = wide_unsigned


def _prefix_refusal() raises:
    var source: List[Float64] = [1.75, bitcast[DType.float64](UInt64(0x7FF8000000000001)), 3]
    var dst = List[UInt32](length=5, fill=0xAABBCCDD)
    var output = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=Int(dst.unsafe_ptr()) + 4)
    var refused = False
    try:
        optimizer_state_cast(Int(source.unsafe_ptr()), 8, 3, 8, CAST_KIND_FLOAT, True, output, True)
    except error:
        refused = String(error).find("cannot convert float NaN to integer") >= 0
    # This converter writes private scratch. Public checkpoint publication
    # atomicity is tested at the Python seam; scratch-prefix writes are allowed.
    require(refused and dst[1] == 1 and dst[2] == 0xAABBCCDD, "state scratch refusal prefix")
    require(dst[3] == 0xAABBCCDD or dst[3] == 3, "later private scratch is either unwritten or correctly converted")
    require(dst[0] == 0xAABBCCDD and dst[4] == 0xAABBCCDD, "state scratch guard")
    require(source[0] == 1.75 and bitcast[DType.uint64](source[1]) == 0x7FF8000000000001 and source[2] == 3, "state scratch source mutated")


def _parallel_earliest_refusal(fixture_id: Int) raises:
    # Large enough to cross the native worker-policy threshold. The test does
    # not prescribe a worker count or completion order; on a one-worker host
    # it still checks the same earliest-input contract.
    comptime COUNT = 262147
    comptime FIRST = 4097
    comptime SECOND = COUNT - 4097
    var reverse = fixture_id >= 2
    var nan_first = fixture_id % 2 == 0
    var source = List[Float64](length=COUNT, fill=1.75)
    var sp = source.unsafe_ptr()
    var first_word = UInt64(0x7FF8000000000001) if nan_first else UInt64(0x7FF0000000000000)
    var second_word = UInt64(0x7FF0000000000000) if nan_first else UInt64(0x7FF8000000000001)
    sp[COUNT - 1 - FIRST if reverse else FIRST] = bitcast[DType.float64](first_word)
    sp[COUNT - 1 - SECOND if reverse else SECOND] = bitcast[DType.float64](second_word)
    var dst = List[UInt32](length=COUNT + 2, fill=0xAABBCCDD)
    var output = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=Int(dst.unsafe_ptr()) + 4)
    var address = Int(source.unsafe_ptr()) + (8 * (COUNT - 1) if reverse else 0)
    var refused = False
    try:
        optimizer_state_cast(address, -8 if reverse else 8, COUNT, 8, CAST_KIND_FLOAT, True, output, True)
    except error:
        var expected = String("ValueError: cannot convert float NaN to integer" if nan_first else
            "OverflowError: buf_initialized value does not fit int32")
        require(String(error) == expected, "parallel checkpoint selected a later input's refusal kind")
        refused = True
    require(refused, "parallel checkpoint accepted invalid integer input")
    require(dst[0] == 0xAABBCCDD and dst[COUNT + 1] == 0xAABBCCDD, "parallel checkpoint scratch guard")
    require(dst[1] == 1, "valid prefix before earliest refusal was not converted")
    require(dst[FIRST + 1] == 0xAABBCCDD and dst[SECOND + 1] == 0xAABBCCDD,
        "invalid cells must not be published into private scratch")
    var tasks = host_predict_task_count_work(COUNT, COUNT * 8)
    var chunk = host_predict_chunk(COUNT, tasks)
    var failures = List[Int](length=tasks, fill=0)
    var fp = failures.unsafe_ptr()
    var src = Pointer[Float64, ImmUntrackedOrigin](unsafe_from_address=Int(source.unsafe_ptr()))
    var result = Pointer[UInt32, ImmUntrackedOrigin](unsafe_from_address=Int(dst.unsafe_ptr()) + 4)
    def inspect(unit: Int) {imm fp, imm src, imm result, imm reverse, imm first_word, imm second_word, imm chunk}:
        var bad = 0
        for index in range(unit * chunk, min(COUNT, (unit + 1) * chunk)):
            var original_index = COUNT - 1 - index if reverse else index
            var expected_word = first_word if index == FIRST else (second_word if index == SECOND else UInt64(0x3FFC000000000000))
            if bitcast[DType.uint64](src[original_index]) != expected_word:
                bad = 1
            # Other independent workers may finish after the first bad input.
            # Their private output must be correct if written, never arbitrary.
            if index != FIRST and index != SECOND and result[index] != 0xAABBCCDD and result[index] != 1:
                bad = 1
        fp.unsafe_store(unit, bad)
    host_parallelize(inspect, tasks)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "parallel checkpoint changed source or wrote incorrect scratch values")
    _ = source
    _ = dst


def run_fixture(fixture_id: Int) raises:
    if fixture_id < 13:
        _integer_scalar(fixture_id)
    elif fixture_id < 24:
        _float_scalar(fixture_id - 13)
    elif fixture_id < 33:
        _span(fixture_id - 24)
    elif fixture_id == 33:
        _prefix_refusal()
    elif fixture_id == 34:
        # An empty input needs no backing storage. The output API is a
        # non-nullable pointer, so prove its guarded destination is untouched.
        var guard = List[UInt32](length=1, fill=0xAABBCCDD)
        var output = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=Int(guard.unsafe_ptr()))
        optimizer_state_cast(0, 0, 0, 8, CAST_KIND_FLOAT, True, output, True)
        require(guard[0] == 0xAABBCCDD, "empty state cast wrote output")

    else:
        _parallel_earliest_refusal(fixture_id - 35)


def run_checks() raises:
    var failures = List[Int](length=STATE_INPUT_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(fixture_id: Int) {imm fp}:
        try:
            run_fixture(fixture_id)
        except error:
            print("FAIL optimizer_state_input fixture", fixture_id, String(error))
            fp.unsafe_store(fixture_id, 1)
    host_parallelize(task, STATE_INPUT_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "optimizer_state_input failed " + String(failed) + " fixtures")
    print("PASS optimizer_state_input:", STATE_INPUT_FIXTURES, "native checkpoint conversion fixtures, parallel")
