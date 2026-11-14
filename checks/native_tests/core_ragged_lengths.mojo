# SPDX-License-Identifier: Apache-2.0
"""The ragged (right-padded) batch passes of `lengths=`, moved from
test_ragged_lengths.py (2026-11-14): the byte passes that
python/mojolearn/_ragged.py hands to the native `ragged_rows_bytes`
(core/array_helpers_impl.mojo on the host, core/input_device.mojo
device_ragged_rows on a GPU).

  fixture 0 (padded_copy, mode 0): float32 (2, 4, 3) = arange + 1, lengths
    (4, 2): row 0 copied whole, row 1's first 2 positions copied, the rest
    zero bytes; the source untouched.
  fixture 1 (zero_padding, mode 1, in place): float32 (3, 5, 2) of -1.0,
    lengths (5, 1, 3): real positions stay -1.0, padding becomes +0.0 bytes
    (never -0.0).
  fixture 2 (padded_copy of int32 ids, mode 0): [[5, 6, 7], [8, 9, 10]],
    lengths (1, 3) -> [[5, 0, 0], [8, 9, 10]].
  fixture 3 (last_real_rows, mode 2): float32 logits (2, 3, 4) = arange,
    lengths (2, 3) -> (2, 1, 4) holding logits[0, 1] and logits[1, 2].

The destination of modes 0 and 2 starts as 0xAB bytes, so every byte the
pass must write is seen written. The lengths admission, refusals and the
ragged_forward composition with the caller's run stay in the Python test.
"""
from std.memory import bitcast
from core.array_helpers_impl import ragged_rows_bytes
from core.host_parallel import host_parallelize
from max.gpu.host import DeviceContext
from core.input_device import device_ragged_rows
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


comptime RAGGED_FIXTURES = 4
comptime _GARBAGE = UInt8(0xAB)


struct RaggedFixture(Movable):
    var src: List[UInt8]
    var dst: List[UInt8]
    var lens: List[Int64]
    var n: Int
    var row: Int
    var pos: Int
    var mode: Int

    def __init__(out self, var src: List[UInt8], dst_bytes: Int, var lens: List[Int64], row: Int, pos: Int, mode: Int):
        self.n = len(lens)
        self.dst = List[UInt8](length=dst_bytes, fill=_GARBAGE)
        if mode == 1:
            self.dst = src.copy()
        self.src = src^
        self.lens = lens^
        self.row = row
        self.pos = pos
        self.mode = mode


def _f32_bytes(values: List[Float32]) -> List[UInt8]:
    var out = List[UInt8](length=4 * len(values), fill=0)
    for i in range(len(values)):
        var w = bitcast[DType.uint32](values[i])
        for b in range(4):
            out[4 * i + b] = UInt8((w >> UInt32(8 * b)) & 0xFF)
    return out^


def _i32_bytes(values: List[Int32]) -> List[UInt8]:
    var out = List[UInt8](length=4 * len(values), fill=0)
    for i in range(len(values)):
        var w = bitcast[DType.uint32](values[i])
        for b in range(4):
            out[4 * i + b] = UInt8((w >> UInt32(8 * b)) & 0xFF)
    return out^


def _word(bytes: List[UInt8], i: Int) -> UInt32:
    return (UInt32(bytes[4 * i]) | (UInt32(bytes[4 * i + 1]) << 8)
            | (UInt32(bytes[4 * i + 2]) << 16) | (UInt32(bytes[4 * i + 3]) << 24))


def ragged_fixture(unit: Int) raises -> RaggedFixture:
    if unit == 0:
        var x = List[Float32](length=24, fill=0)
        for i in range(24):
            x[i] = Float32(i + 1)
        var lens: List[Int64] = [4, 2]
        return RaggedFixture(_f32_bytes(x), 96, lens^, 48, 12, 0)
    if unit == 1:
        var y = List[Float32](length=30, fill=-1.0)
        var lens: List[Int64] = [5, 1, 3]
        return RaggedFixture(_f32_bytes(y), 120, lens^, 40, 8, 1)
    if unit == 2:
        var ids: List[Int32] = [5, 6, 7, 8, 9, 10]
        var lens: List[Int64] = [1, 3]
        return RaggedFixture(_i32_bytes(ids), 24, lens^, 12, 4, 0)
    require(unit == 3, "unknown ragged fixture")
    var logits = List[Float32](length=24, fill=0)
    for i in range(24):
        logits[i] = Float32(i)
    var lens: List[Int64] = [2, 3]
    # mode 2: row = V * 4 bytes, pos = L * V * 4 bytes (_ragged.last_real_rows)
    return RaggedFixture(_f32_bytes(logits), 32, lens^, 16, 48, 2)


def check_ragged(unit: Int, f: RaggedFixture) raises:
    var tag = "ragged fixture " + String(unit)
    if unit == 0:
        for i in range(24):
            require(_word(f.src, i) == bitcast[DType.uint32](Float32(i + 1)), tag + ": the source changed")
        for i in range(12):
            require(_word(f.dst, i) == _word(f.src, i), tag + ": row 0 real position bytes")
        for i in range(12, 18):
            require(_word(f.dst, i) == _word(f.src, i), tag + ": row 1 real position bytes")
        for i in range(18, 24):
            require(_word(f.dst, i) == 0, tag + ": row 1 padding is zero bytes")
    elif unit == 1:
        var lens: List[Int] = [5, 1, 3]
        for b in range(3):
            for t in range(5):
                for c in range(2):
                    var w = _word(f.dst, (b * 5 + t) * 2 + c)
                    if t < lens[b]:
                        require(w == bitcast[DType.uint32](Float32(-1.0)), tag + ": a real position changed")
                    else:
                        require(w == 0, tag + ": padding is +0.0 (zero bytes), never -0.0")
    elif unit == 2:
        var want: List[Int32] = [5, 0, 0, 8, 9, 10]
        for i in range(6):
            require(bitcast[DType.int32](_word(f.dst, i)) == want[i], tag + ": ids by position")
    else:
        for k in range(4):
            require(_word(f.dst, k) == bitcast[DType.uint32](Float32(1 * 4 + k)), tag + ": row 0 reads position 1")
            require(_word(f.dst, 4 + k) == bitcast[DType.uint32](Float32(12 + 2 * 4 + k)), tag + ": row 1 reads position 2")


# ---- The host column (was core_ragged_lengths_cpu_suite.mojo) ----
# The host ragged passes; the four independent fixtures run across host
# cores, each with its own buffers and failure slot.
def check_unit(unit: Int) raises:
    var f = ragged_fixture(unit)
    var dst = Int(f.dst.unsafe_ptr())
    var src = Int(f.src.unsafe_ptr()) if f.mode != 1 else dst
    ragged_rows_bytes(
        MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=dst),
        MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=src),
        MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(f.lens.unsafe_ptr())),
        f.n, f.row, f.pos, f.mode,
    )
    check_ragged(unit, f)


def _run_cpu(mut report: Report) raises:
    var cases = report.cases(RAGGED_FIXTURES)
    def task(unit: Int) {imm cases}:
        try:
            check_unit(unit)
        except error:
            cases.fail(unit, error)
    report.parallel(task, RAGGED_FIXTURES)


# ---- The device column (was core_ragged_lengths_gpu_suite.mojo) ----
# The device ragged passes, in order on one DeviceContext: each pass is one
# thread per destination byte, and the four fixtures share the context's
# stream, so ordering them costs nothing.
def _run_gpu(mut report: Report) raises:
    var ctx = DeviceContext()
    var cases = report.cases(RAGGED_FIXTURES)
    for unit in range(RAGGED_FIXTURES):
        try:
            var f = ragged_fixture(unit)
            var src = Int(f.src.unsafe_ptr()) if f.mode != 1 else Int(f.dst.unsafe_ptr())
            var wrote = device_ragged_rows(ctx, src, Int(f.dst.unsafe_ptr()), Int(f.lens.unsafe_ptr()),
                                           f.n, f.row, f.pos, f.mode)
            require(wrote, "the device pass declined fixture " + String(unit))
            check_ragged(unit, f)
        except error:
            cases.fail(unit, error)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if CPU_COLUMN_BUILD:
            _run_cpu(report)
        else:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("core_ragged_lengths", cpu=True, gpu=True, serial="", run=run_cases)
