# SPDX-License-Identifier: Apache-2.0
"""Original linalg numerical cases, independent of the Python binding layer.

Inputs preserve NumPy seed 5 and every original dimension. QR uses its Gram
identity; the diagonal and duplicated-column expectations are analytic. The
constant-column spectrum is recorded from independent NumPy Float64 SVD.
No output hashes, repeated-execution comparison or alternative solver.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize

comptime LINALG_NUMERICAL_FIXTURES = 5
comptime LINALG_GUARD = Float32(27182.5)


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


trait LinalgNumericalCalls:
    @staticmethod
    def qr(source: List[Float32], rows: Int, cols: Int, dst: Int) raises: ...
    @staticmethod
    def singular(source: List[Float32], rows: Int, cols: Int, dst: Int) raises: ...
    @staticmethod
    def eigen(source: Int, values: Int, vectors: Int, n: Int) raises: ...


def input_values(name: String, count: Int) raises -> List[Float32]:
    var result = List[Float32](unsafe_uninit_length=count)
    with open("checks/native_tests/fixtures/linalg_numerical/" + name + ".f32", "r") as file:
        var bytes = file.read_bytes()
        require(len(bytes) == count * 4, "linalg input fixture extent changed")
        var bp = bytes.unsafe_ptr()
        var rp = result.unsafe_ptr()
        def word(index: Int) {imm bp, imm rp}:
            var at = index * 4
            var bits = UInt32(bp[at]) | (UInt32(bp[at + 1]) << 8) | (UInt32(bp[at + 2]) << 16) | (UInt32(bp[at + 3]) << 24)
            rp.unsafe_store(index, bitcast[DType.float32](bits))
        host_parallelize(word, count)
        _ = bytes^
    return result^


def constant_reference() raises -> List[Float64]:
    var result = List[Float64](unsafe_uninit_length=10)
    with open("checks/native_tests/fixtures/linalg_numerical/constant-64x10-singular.f64", "r") as file:
        var bytes = file.read_bytes()
        require(len(bytes) == 80, "linalg reference fixture extent changed")
        var bp = bytes.unsafe_ptr()
        var rp = result.unsafe_ptr()
        def word(index: Int) {imm bp, imm rp}:
            var bits = UInt64(0)
            for byte in range(8):  # Fixed-width file decoding, not a numerical oracle.
                bits |= UInt64(bp[index * 8 + byte]) << UInt64(byte * 8)
            rp.unsafe_store(index, bitcast[DType.float64](bits))
        host_parallelize(word, 10)
        _ = bytes^
    return result^


def guards(value: List[Float32], name: String) raises:
    require(value[0] == LINALG_GUARD and value[len(value) - 1] == LINALG_GUARD, name + " output guards")


def source_unchanged(source: List[Float32], original: List[Float32]) raises:
    # Storage ownership assertion, not a repeatability/identity run.
    var count = len(source)
    var failures = List[Int](length=count, fill=0)
    var fp = failures.unsafe_ptr()
    var ap = source.unsafe_ptr()
    var bp = original.unsafe_ptr()
    def cell(index: Int) {imm fp, imm ap, imm bp}:
        fp.unsafe_store(index, Int(bitcast[DType.uint32](ap[index]) != bitcast[DType.uint32](bp[index])))
    host_parallelize(cell, count)
    for failure in failures:
        require(failure == 0, "linalg operation changed its caller's source")
    _ = source
    _ = original


def qr_expectations(source: List[Float32], result: List[Float32], rows: Int, cols: Int) raises:
    var failures = List[Int](length=cols * cols, fill=0)
    var fp = failures.unsafe_ptr()
    var ap = source.unsafe_ptr()
    var rp = result.unsafe_ptr() + 1
    def cell(index: Int) {imm fp, imm ap, imm rp, imm rows, imm cols}:
        var i = index // cols
        var j = index % cols
        var expected = Float64(0)
        var actual = Float32(0)
        for row in range(rows):  # Independent Gram cell; columns/cells split across the host.
            expected += Float64(ap[row * cols + i]) * Float64(ap[row * cols + j])
        for row in range(cols):
            actual += rp[row * cols + i] * rp[row * cols + j]
        var triangular = i <= j or abs(rp[index]) <= Float32(1e-5)
        var close = abs(Float64(actual) - expected) <= Float64(1e-4) + Float64(1e-4) * abs(expected)
        fp.unsafe_store(index, Int(not triangular or not close))
    host_parallelize(cell, cols * cols)
    for failure in failures:
        require(failure == 0, "QR triangularity or original Gram tolerance")
    _ = source
    _ = result


def singular_expectations(result: List[Float32], reference: List[Float64], cols: Int, constant: Bool) raises:
    var failures = List[Int](length=cols, fill=0)
    var fp = failures.unsafe_ptr()
    var sp = result.unsafe_ptr() + 1
    var rp = reference.unsafe_ptr()
    def cell(index: Int) {imm fp, imm sp, imm rp, imm cols, imm constant}:
        var ok = index == 0 or sp[index - 1] >= sp[index]
        if constant:
            if index < 5:
                ok = ok and abs(Float64(sp[index]) - rp[index]) <= Float64(1e-4) * abs(rp[index]) + Float64(1e-5) * rp[0]
            else:
                ok = ok and abs(Float64(sp[index])) <= Float64(1e-3) * rp[0]
        elif index == cols - 1:
            ok = ok and abs(sp[index]) < Float32(1e-3)
        fp.unsafe_store(index, Int(not ok))
    host_parallelize(cell, cols)
    for failure in failures:
        require(failure == 0, "SVD ordering/rank/independent spectrum tolerance")
    _ = result
    _ = reference


def fixture[Calls: LinalgNumericalCalls](index: Int) raises:
    if index < 2:
        var rows = 48 if index == 0 else 16
        var cols = 5 if index == 0 else 4
        var source = input_values("qr-48x5" if index == 0 else "qr-16x4", rows * cols)
        var original = source.copy()
        var result = List[Float32](length=cols * cols + 2, fill=LINALG_GUARD)
        Calls.qr(source, rows, cols, Int(result.unsafe_ptr()) + 4)
        guards(result, "QR")
        qr_expectations(source, result, rows, cols)
        source_unchanged(source, original)
        _ = result^
    elif index == 2:
        var source: List[Float32] = [3, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 2, 0, 0, 0, 0, 9]
        var original = source.copy()
        var values = List[Float32](length=6, fill=LINALG_GUARD)
        var vectors = List[Float32](length=18, fill=LINALG_GUARD)
        Calls.eigen(Int(source.unsafe_ptr()), Int(values.unsafe_ptr()) + 4, Int(vectors.unsafe_ptr()) + 4, 4)
        guards(values, "eigh values")
        guards(vectors, "eigh vectors")
        var expected: List[Float32] = [0.5, 2, 3, 9]
        var failures = List[Int](length=4, fill=0)
        var fp = failures.unsafe_ptr()
        var vp = values.unsafe_ptr() + 1
        var ep = expected.unsafe_ptr()
        def cell(at: Int) {imm fp, imm vp, imm ep}:
            fp.unsafe_store(at, Int(not (abs(vp[at] - ep[at]) <= Float32(1e-6))))
        host_parallelize(cell, 4)
        for failure in failures:
            require(failure == 0, "diagonal eigh must return ascending diagonal")
        source_unchanged(source, original)
        _ = expected^
        _ = values^
        _ = vectors^
    else:
        var rows = 48 if index == 3 else 64
        var cols = 4 if index == 3 else 10
        var source = List[Float32]()
        var reference = List[Float64](length=1, fill=0)
        if index == 3:
            var base = input_values("rank-48x3", 144)
            source = List[Float32](unsafe_uninit_length=192)
            var bp = base.unsafe_ptr()
            var sp = source.unsafe_ptr()
            def duplicate(at: Int) {imm bp, imm sp}:
                var column = at % 4
                sp.unsafe_store(at, bp[(at // 4) * 3 + (column if column < 2 else column - 1)])
            host_parallelize(duplicate, 192)
            _ = base^
        else:
            source = input_values("constant-64x10", 640)
            reference = constant_reference()
        var original = source.copy()
        var result = List[Float32](length=cols + 2, fill=LINALG_GUARD)
        Calls.singular(source, rows, cols, Int(result.unsafe_ptr()) + 4)
        guards(result, "svdvals")
        singular_expectations(result, reference, cols, index == 4)
        source_unchanged(source, original)
        _ = result^

