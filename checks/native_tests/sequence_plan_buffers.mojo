# SPDX-License-Identifier: Apache-2.0
"""Borrowed native spans and parallel C-order output, independent expectations."""
from std.memory import bitcast
from core.borrowed_buffer import BorrowedBufferLayout
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work
from core.sequential_input_plan import SequenceInputPlan, sequence_write
from checks.native_tests.sequence_plan_metadata import require

comptime SEQUENCE_BUFFER_FIXTURES = 8


def _borrowed(fixture_id: Int) raises:
    var direct: List[Float32] = [1, 2, 3, 4, 5, 6]
    var fortran: List[Float32] = [1, 4, 2, 5, 3, 6]
    var endian: List[UInt8] = [0xFF, 0xFE, 0, 3, 0xFF, 0xFC, 0, 5, 0xFF, 0xFA, 0, 7]
    var row_a: List[Float32] = [-99, 1, 2, 3]
    var row_b: List[Float32] = [-99, 4, 5, 6]
    var row_pointers: List[Int] = [Int(row_a.unsafe_ptr()), Int(row_b.unsafe_ptr())]
    var scalar: List[Float64] = [-2.5]
    var dims: List[Int] = [2, 3]
    var strides: List[Int] = [12, 4]
    var suboffsets: List[Int] = [-1, -1]
    var address = Int(direct.unsafe_ptr())
    var kind = String("f")
    var width = 4
    var native = True
    var expected: List[Float32] = [1, 2, 3, 4, 5, 6]
    if fixture_id == 1:
        address = Int(fortran.unsafe_ptr())
        strides = [4, 8]
    elif fixture_id == 2:
        address += 20
        strides = [-12, -4]
        expected = [6, 5, 4, 3, 2, 1]
    elif fixture_id == 3:
        strides = [0, 4]
        expected = [1, 2, 3, 1, 2, 3]
    elif fixture_id == 4:
        address = Int(endian.unsafe_ptr()) + 10
        strides = [-6, -2]
        width = 2
        kind = "i"
        native = False
        expected = [7, -6, 5, -4, 3, -2]
    elif fixture_id == 5:
        address = Int(row_pointers.unsafe_ptr())
        strides = [8, 4]
        suboffsets = [4, -1]
    elif fixture_id == 6:
        address = Int(scalar.unsafe_ptr())
        dims = List[Int]()
        strides = List[Int]()
        suboffsets = List[Int]()
        width = 8
        expected = [-2.5]
    var layout = BorrowedBufferLayout(address, width, True, dims.copy(), strides.copy(), suboffsets.copy())
    var plan = SequenceInputPlan()
    plan.array(0, layout.copy(), kind, native)
    require(plan.count == len(expected) and len(plan.segments) == 1 and len(plan.arrays) == 1 and len(plan.words) == 0, "borrowed array stored as one span")
    require(plan.arrays[0].layout.address == address, "array leaf copied instead of borrowed")
    if fixture_id != 6:
        dims[0] = 99
        strides[0] = 999
        layout.shape[0] = 77
        require(plan.shape[0] == 2 and plan.shape[1] == 3 and plan.arrays[0].layout.shape[0] == 2, "borrowed metadata aliases caller")
    # A descriptor is a live borrow, not an eager snapshot. Mutating direct
    # storage before execution must affect the prepared value, without making
    # the resulting output alias that storage after execution.
    if fixture_id == 0:
        direct[0] = 17
        expected[0] = 17
    var dst = List[Float32](length=len(expected) + 2, fill=-999)
    sequence_write(plan, Int(dst.unsafe_ptr()) + 4, True)
    require(dst[0] == -999 and dst[len(dst) - 1] == -999, "borrowed output guards")
    for i in range(len(expected)):
        require(dst[i + 1] == expected[i], "borrowed C-order output")
    for i in range(6):
        require(direct[i] == Float32(17 if fixture_id == 0 and i == 0 else i + 1), "borrowed source mutated")
    var expected_fortran: List[Float32] = [1, 4, 2, 5, 3, 6]
    for i in range(6):
        require(fortran[i] == expected_fortran[i], "Fortran source mutated")
    var expected_endian: List[UInt8] = [0xFF, 0xFE, 0, 3, 0xFF, 0xFC, 0, 5, 0xFF, 0xFA, 0, 7]
    for i in range(12):
        require(endian[i] == expected_endian[i], "endian source mutated")
    require(row_a[0] == -99 and row_b[0] == -99, "indirect row prefixes mutated")
    for i in range(1, 4):
        require(row_a[i] == Float32(i) and row_b[i] == Float32(i + 3), "indirect source mutated")
    require(row_pointers[0] == Int(row_a.unsafe_ptr()) and row_pointers[1] == Int(row_b.unsafe_ptr()) and scalar[0] == -2.5, "pointer/scalar source mutated")
    if fixture_id == 0:
        direct[0] = 23
        require(dst[1] == 17, "final output aliases borrowed source")


def _parallel_spans() raises:
    comptime N = 32769
    comptime CHUNK = 1024
    comptime PREP_UNITS = (N + CHUNK - 1) // CHUNK
    comptime CHECK_UNITS = (3 * N + CHUNK - 1) // CHUNK
    var a = List[Float32](length=N, fill=0)
    var b = List[Float32](length=N, fill=0)
    var ap = a.unsafe_ptr()
    var bp = b.unsafe_ptr()
    def prepare(unit: Int) {imm ap, imm bp}:
        for i in range(unit * CHUNK, min((unit + 1) * CHUNK, N)):
            ap.unsafe_store(i, Float32(i % 17 - 8))
            bp.unsafe_store(i, Float32(i % 11 - 5))
    host_parallelize(prepare, PREP_UNITS)
    var dims: List[Int] = [N]
    var forward: List[Int] = [4]
    var reverse: List[Int] = [-4]
    var suboffsets: List[Int] = [-1]
    var left = BorrowedBufferLayout(Int(a.unsafe_ptr()), 4, True, dims.copy(), forward^, suboffsets.copy())
    var right = BorrowedBufferLayout(Int(b.unsafe_ptr()) + 4 * (N - 1), 4, True, dims.copy(), reverse^, suboffsets.copy())
    var plan = SequenceInputPlan()
    plan.container(0, 3)
    plan.array(1, left^, "f", True)
    plan.container(1, N)
    # This ordered metadata append is one plan's state transition, not
    # independent row computation. Source preparation/output checks and the
    # product's final write run across the existing host pool.
    for i in range(N):
        plan.scalar(2, bitcast[DType.uint64](Float64(i % 13 - 6)), "f", 8)
    plan.array(1, right^, "f", True)
    require(plan.count == 3 * N and plan.kind == "f" and plan.itemsize == 8 and len(plan.shape) == 2 and plan.shape[0] == 3 and plan.shape[1] == N, "mixed-run output shape/promotion")
    require(len(plan.words) == N and len(plan.arrays) == 2 and len(plan.segments) == 3, "scalar run must not create per-value descriptors")
    require(plan.segments[0].start == 0 and plan.segments[1].start == N and plan.segments[2].start == 2 * N, "mixed segment starts")
    require(plan.segments[0].source == 0 and plan.segments[1].source == -1 and plan.segments[2].source == 1 and plan.segments[1].scalar_start == 0, "mixed segment sources")
    a[7] = 777
    b[N - 8] = 888
    var dst = List[Float32](length=3 * N + 2, fill=-999)
    sequence_write(plan, Int(dst.unsafe_ptr()) + 4, True)
    require(dst[0] == -999 and dst[3 * N + 1] == -999, "parallel write guards")
    var dp = dst.unsafe_ptr()
    var failures = List[Int](length=CHECK_UNITS, fill=0)
    var fp = failures.unsafe_ptr()
    def check(unit: Int) {imm dp, imm ap, imm bp, imm fp}:
        for i in range(unit * CHUNK, min((unit + 1) * CHUNK, 3 * N)):
            var row = i // N
            var column = i % N
            var expected = Float32(column % 17 - 8)
            if row == 0 and column == 7:
                expected = 777
            elif row == 1:
                expected = Float32(column % 13 - 6)
            elif row == 2:
                expected = Float32(888 if column == 7 else (N - 1 - column) % 11 - 5)
            if dp.unsafe_load(i + 1) != expected:
                fp.unsafe_store(unit, 1)
            if i < N:
                var expected_a = Float32(777 if i == 7 else i % 17 - 8)
                var expected_b = Float32(888 if i == N - 8 else i % 11 - 5)
                if ap.unsafe_load(i) != expected_a or bp.unsafe_load(i) != expected_b:
                    fp.unsafe_store(unit, 1)
    host_parallelize(check, CHECK_UNITS)
    for failed in failures:
        require(failed == 0, "parallel span output/owner preservation")
    # Report the resolved product split, without claiming measured busy cores.
    print("sequence parallel output:", 3 * N, "cells,", host_predict_task_count_work(3 * N, 3 * N * 8), "resolved write tasks")
    _ = plan
    _ = a
    _ = b
    _ = dst


def run_buffer_fixture(fixture_id: Int) raises:
    if fixture_id < 7:
        _borrowed(fixture_id)
    else:
        _parallel_spans()
