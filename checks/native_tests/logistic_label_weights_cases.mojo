# SPDX-License-Identifier: Apache-2.0
"""Actual class-count/gather/multiply descriptors with independent expectations."""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.label_weights_native_api import (
    LabelWeightsDriver, LabelWeightsRequest, LABEL_WEIGHTS_COUNTS,
    LABEL_WEIGHTS_ROWS, LABEL_WEIGHTS_MULTIPLY,
)

comptime LABEL_WEIGHT_FIXTURES = 8
comptime WEIGHT_GUARD = UInt32(0xCD123456)
comptime COUNT_GUARD = Int32(-1234567)


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def code_at(index: Int, fixture_id: Int, classes: Int) -> Int32:
    var literal: Array[Int32, 8] = [0, 2, 1, 2, 0, 2, 0, 1]
    return literal[index] if fixture_id == 1 else Int32(index % classes)


def class_word(index: Int, raw: Bool) -> UInt32:
    var words: Array[UInt32, 5] = [0x80000000, 0x00000000, 0x7FC00042, 0x7F800123, 0x3F800000]
    return words[index] if raw else bitcast[DType.uint32](Float32(index + 1) * Float32(0.125))


def pipeline(driver: LabelWeightsDriver, fixture_id: Int) raises:
    var rows = 0 if fixture_id == 0 else (8 if fixture_id == 1 else (262271 if fixture_id == 2 else 17))
    var classes = 3 if fixture_id < 2 else (7 if fixture_id == 2 else 5)
    var raw = fixture_id == 3
    var codes = List[Int32](length=rows + 2, fill=COUNT_GUARD)
    var values = List[UInt32](length=classes + 2, fill=WEIGHT_GUARD)
    var sample = List[Float32](length=rows + 2, fill=bitcast[DType.float32](WEIGHT_GUARD))
    var cp = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(codes.unsafe_ptr()) + 4)
    var vp = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=Int(values.unsafe_ptr()) + 4)
    var sw = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(sample.unsafe_ptr()) + 4)
    def prepare(index: Int) {imm cp, imm vp, imm sw, imm fixture_id, imm classes, imm rows, imm raw}:
        if index < rows:
            cp[index] = code_at(index, fixture_id, classes)
            sw[index] = Float32(index % 13 + 1) * Float32(0.25)
        if index < classes:
            vp[index] = class_word(index, raw)
    host_parallelize(prepare, max(rows, classes))
    var counts = List[Int32](length=classes + 2, fill=COUNT_GUARD)
    var gathered = List[UInt32](length=rows + 2, fill=WEIGHT_GUARD)
    var multiplied = List[UInt32](length=rows + 2, fill=WEIGHT_GUARD)
    var source_address = Int(codes.unsafe_ptr()) + 4 if rows else 0
    var value_address = Int(values.unsafe_ptr()) + 4
    var row_address = Int(gathered.unsafe_ptr()) + 4 if rows else 0
    require(driver.run(LABEL_WEIGHTS_COUNTS, LabelWeightsRequest(source_address,
        Int(counts.unsafe_ptr()) + 4, 0, rows, classes)) == classes, "class count descriptor result")
    require(driver.run(LABEL_WEIGHTS_ROWS, LabelWeightsRequest(source_address, value_address,
        row_address, rows, classes)) == rows, "class-row descriptor result")
    if not raw:
        require(driver.run(LABEL_WEIGHTS_MULTIPLY, LabelWeightsRequest(row_address,
            Int(sample.unsafe_ptr()) + 4 if rows else 0,
            Int(multiplied.unsafe_ptr()) + 4 if rows else 0, rows, 0)) == rows,
            "weight multiplication descriptor result")
    var np = Pointer[Int32, ImmUntrackedOrigin](unsafe_from_address=Int(counts.unsafe_ptr()) + 4)
    var rp = Pointer[UInt32, ImmUntrackedOrigin](unsafe_from_address=Int(gathered.unsafe_ptr()) + 4)
    var pp = Pointer[UInt32, ImmUntrackedOrigin](unsafe_from_address=Int(multiplied.unsafe_ptr()) + 4)
    var failures = List[Int](length=max(rows, classes), fill=0)
    var fp = failures.unsafe_ptr()
    def check(index: Int) {imm cp, imm vp, imm sw, imm np, imm rp, imm pp, imm fp,
            imm fixture_id, imm classes, imm rows, imm raw}:
        if index < classes:
            var literal_counts: Array[Int32, 3] = [3, 2, 3]
            var expected = literal_counts[index] if fixture_id == 1 else Int32(rows // classes + Int(index < rows % classes))
            if np[index] != expected or vp[index] != class_word(index, raw):
                fp.unsafe_store(index, 1)
        if index < rows:
            var code = code_at(index, fixture_id, classes)
            if cp[index] != code or rp[index] != class_word(Int(code), raw):
                fp.unsafe_store(index, 1)
            var sample_expected = Float32(index % 13 + 1) * Float32(0.25)
            if bitcast[DType.uint32](sw[index]) != bitcast[DType.uint32](sample_expected):
                fp.unsafe_store(index, 1)
            if not raw:
                # Two FP32 factors have an exact Float64 product. Independent
                # widening then one narrowing is the kernel's numerical contract.
                var expected = Float32((Float64(Int(code) + 1) / 8) * (Float64(index % 13 + 1) / 4))
                if pp[index] != bitcast[DType.uint32](expected):
                    fp.unsafe_store(index, 1)
    host_parallelize(check, max(rows, classes))
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "native count/class-row/sample-weight product differs")
    require(codes[0] == COUNT_GUARD and codes[len(codes) - 1] == COUNT_GUARD
        and counts[0] == COUNT_GUARD and counts[len(counts) - 1] == COUNT_GUARD, "class count source/output guards")
    require(values[0] == WEIGHT_GUARD and values[len(values) - 1] == WEIGHT_GUARD
        and gathered[0] == WEIGHT_GUARD and gathered[len(gathered) - 1] == WEIGHT_GUARD
        and multiplied[0] == WEIGHT_GUARD and multiplied[len(multiplied) - 1] == WEIGHT_GUARD
        and bitcast[DType.uint32](sample[0]) == WEIGHT_GUARD
        and bitcast[DType.uint32](sample[len(sample) - 1]) == WEIGHT_GUARD, "class-weight source/output guards")
    _ = codes
    _ = values
    _ = sample
    _ = counts
    _ = gathered
    _ = multiplied


def invalid_code(driver: LabelWeightsDriver, fixture_id: Int) raises:
    var codes: List[Int32] = [COUNT_GUARD, 0, 1, 2, 0, 1, 2, 0, 1, COUNT_GUARD]
    codes[4] = -1 if fixture_id % 2 == 0 else 3
    var values: List[Float32] = [1, 2, 3]
    var counts = List[Int32](length=5, fill=COUNT_GUARD)
    var rows = List[UInt32](length=10, fill=WEIGHT_GUARD)
    var count_operation = fixture_id < 2
    var operation = LABEL_WEIGHTS_COUNTS if count_operation else LABEL_WEIGHTS_ROWS
    var second = Int(counts.unsafe_ptr()) + 4 if count_operation else Int(values.unsafe_ptr())
    var refused = False
    try:
        _ = driver.run(operation, LabelWeightsRequest(Int(codes.unsafe_ptr()) + 4,
            second, Int(rows.unsafe_ptr()) + 4, 8, 3))
    except error:
        var expected = String("x_trees code_counts: a class code outside [0, n_classes)") if count_operation else String("x_trees class_rows: a class code outside [0, n_classes)")
        refused = String(error) == expected
    require(refused, "invalid native class code did not reach descriptor error reporting")
    require(counts[0] == COUNT_GUARD and counts[4] == COUNT_GUARD, "refused class count guards")
    # Counts may be partial on refusal; do not impose an atomic contract the
    # existing kernels do not promise. Class-row publication is atomic.
    var rp = Pointer[UInt32, ImmUntrackedOrigin](unsafe_from_address=Int(rows.unsafe_ptr()))
    var cp = Pointer[Int32, ImmUntrackedOrigin](unsafe_from_address=Int(codes.unsafe_ptr()))
    var failures = List[Int](length=10, fill=0)
    var fp = failures.unsafe_ptr()
    def unchanged(index: Int) {imm rp, imm cp, imm fp, imm fixture_id}:
        var expected_code = COUNT_GUARD if index == 0 or index == 9 else (
            Int32(-1 if fixture_id % 2 == 0 else 3) if index == 4 else Int32((index - 1) % 3))
        if cp[index] != expected_code or rp[index] != WEIGHT_GUARD:
            fp.unsafe_store(index, 1)
    host_parallelize(unchanged, 10)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "refused class code mutated source or published rows")
    _ = codes
    _ = rows
    _ = values


def fixture(driver: LabelWeightsDriver, fixture_id: Int) raises:
    if fixture_id < 4:
        pipeline(driver, fixture_id)
    else:
        invalid_code(driver, fixture_id - 4)
