# SPDX-License-Identifier: Apache-2.0
"""Original dense COO/neighbor-affinity workloads without Python execution.

Raw reference words were prepared from the original independent Python scan
and sorted-candidate oracle. Product kernels, dimensions and tie rules stay
unchanged. Native owners remain live through every borrowed pointer use.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.spectral_preparation_api import SpectralPreparationDriver, SpectralCOOCountRequest, SpectralCOOFillRequest

comptime SPECTRAL_COO_FIXTURES = 6
comptime SPECTRAL_AFFINITY_FIXTURES = 8
comptime SPECTRAL_DENSE_GUARD = Float32(12671.5)


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def clear(failures: List[Int], message: String) raises:
    var count = 0
    for value in failures:
        count += value
    require(count == 0, message + ": " + String(count) + " mismatched cells")


def read_words(name: String, count: Int) raises -> List[UInt32]:
    var values = List[UInt32](unsafe_uninit_length=count)
    with open("checks/native_tests/fixtures/spectral/" + name, "r") as file:
        var bytes = file.read_bytes()
        require(len(bytes) == 4 * count, "Spectral fixture extent")
        var bp = bytes.unsafe_ptr()
        var vp = values.unsafe_ptr()
        def word(index: Int) {imm bp, imm vp}:
            var bits = UInt32(0)
            for offset in range(4):  # Fixed-width little-endian word decode.
                bits |= UInt32(bp[index * 4 + offset]) << UInt32(offset * 8)
            vp.unsafe_store(index, bits)
        host_parallelize(word, count)
        _ = bytes^
    return values^


def floats(words: List[UInt32]) -> List[Float32]:
    var values = List[Float32](unsafe_uninit_length=len(words))
    var wp = words.unsafe_ptr()
    var vp = values.unsafe_ptr()
    def word(index: Int) {imm wp, imm vp}:
        vp.unsafe_store(index, bitcast[DType.float32](wp[index]))
    host_parallelize(word, len(words))
    _ = words
    return values^


def unchanged(values: List[Float32], words: List[UInt32]) raises:
    require(len(values) == len(words), "Spectral borrowed source extent")
    var failures = List[Int](length=len(words), fill=0)
    var fp = failures.unsafe_ptr()
    var vp = values.unsafe_ptr()
    var wp = words.unsafe_ptr()
    def word(index: Int) {imm fp, imm vp, imm wp}:
        fp.unsafe_store(index, Int(bitcast[DType.uint32](vp[index]) != wp[index]))
    host_parallelize(word, len(words))
    clear(failures, "Spectral borrowed source raw words unchanged")
    _ = values
    _ = words


def coo_case(driver: SpectralPreparationDriver, unit: Int) raises:
    var n = 17 + unit
    var counts: List[Int] = [196, 232, 244, 283, 315, 340]
    var count = counts[unit]
    var prefix = "coo-" + String(unit)
    var words = read_words(prefix + "-input.f32", n * n)
    var source = floats(words)
    var expected_rows = read_words(prefix + "-rows.i32", count)
    var expected_columns = read_words(prefix + "-cols.i32", count)
    var expected_values = read_words(prefix + "-values.f32", count)
    var rows = List[Int32](length=count + 2, fill=-31)
    var columns = List[Int32](length=count + 2, fill=-31)
    var values = List[Float32](length=count + 2, fill=SPECTRAL_DENSE_GUARD)
    try:
        var address = Int(source.unsafe_ptr())
        require(driver.count(SpectralCOOCountRequest(address, n * n, False)) == count,
            "Spectral original Float32 nonzero count")
        require(driver.fill(SpectralCOOFillRequest(address, Int(rows.unsafe_ptr() + 1),
            Int(columns.unsafe_ptr() + 1), Int(values.unsafe_ptr() + 1), n, n, count, False)) == count,
            "Spectral original Float32 fill count")
        var failures = List[Int](length=count, fill=0)
        var fp = failures.unsafe_ptr()
        var rp = rows.unsafe_ptr() + 1
        var cp = columns.unsafe_ptr() + 1
        var vp = values.unsafe_ptr() + 1
        var er = expected_rows.unsafe_ptr()
        var ec = expected_columns.unsafe_ptr()
        var ev = expected_values.unsafe_ptr()
        def cell(index: Int) {imm fp, imm rp, imm cp, imm vp, imm er, imm ec, imm ev}:
            var okay = bitcast[DType.uint32](rp[index]) == er[index]
            okay = okay and bitcast[DType.uint32](cp[index]) == ec[index]
            okay = okay and bitcast[DType.uint32](vp[index]) == ev[index]
            fp.unsafe_store(index, Int(not okay))
        host_parallelize(cell, count)
        clear(failures, "Spectral original row-major COO words, including NaN/Inf/subnormal values")
        require(rows[0] == Int32(-31) and rows[count + 1] == Int32(-31)
            and columns[0] == Int32(-31) and columns[count + 1] == Int32(-31), "Spectral COO index guards")
        require(values[0] == SPECTRAL_DENSE_GUARD and values[count + 1] == SPECTRAL_DENSE_GUARD, "Spectral COO value guards")
        unchanged(source, words)
    finally:
        _ = source^
        _ = words^
        _ = rows^
        _ = columns^
        _ = values^
        _ = expected_rows^
        _ = expected_columns^
        _ = expected_values^


trait SpectralDenseCalls:
    @staticmethod
    def affinity(source: Int, rows: Int, neighbors: Int, dst: Int, status: Int) raises: ...


def affinity_case[Calls: SpectralDenseCalls](unit: Int) raises:
    var n = 23 + unit
    var k = 3 + unit
    if unit >= 6:
        n = 6
        k = 2
    var words = List[UInt32]()
    var expected = List[UInt32]()
    if unit < 6:
        var prefix = "knn-" + String(unit)
        words = read_words(prefix + "-input.f32", n * n)
        expected = read_words(prefix + "-affinity.f32", n * n)
    else:
        words = List[UInt32](length=n * n, fill=0x3f800000)
        words[2 * n + 4] = UInt32(0xbf800000) if unit == 6 else UInt32(0x7fc00000)
        expected = List[UInt32](length=n * n, fill=0)
    var source = floats(words)
    var destination = List[Float32](length=n * n + 2, fill=0)
    destination[0] = SPECTRAL_DENSE_GUARD
    destination[n * n + 1] = SPECTRAL_DENSE_GUARD
    var status: List[Int32] = [-31, -7, -7, -7, -31]
    try:
        Calls.affinity(Int(source.unsafe_ptr()), n, k, Int(destination.unsafe_ptr() + 1), Int(status.unsafe_ptr() + 1))
        if unit < 6:
            require(status[1] == Int32(0) and status[2] == Int32(0) and status[3] == Int32(0), "Spectral dense affinity successful status")
        else:
            require(status[1] == Int32(1) and status[2] == Int32(2) and status[3] == Int32(6),
                "Spectral negative/NaN refusal identifies first bad row and candidate count")
        var failures = List[Int](length=n * n, fill=0)
        var fp = failures.unsafe_ptr()
        var dp = destination.unsafe_ptr() + 1
        var ep = expected.unsafe_ptr()
        def cell(index: Int) {imm fp, imm dp, imm ep}:
            fp.unsafe_store(index, Int(bitcast[DType.uint32](dp[index]) != ep[index]))
        host_parallelize(cell, n * n)
        clear(failures, "Spectral original sorted-distance/index and symmetric affinity oracle, or refused zero output")
        require(destination[0] == SPECTRAL_DENSE_GUARD and destination[n * n + 1] == SPECTRAL_DENSE_GUARD,
            "Spectral affinity output guards")
        require(status[0] == Int32(-31) and status[4] == Int32(-31), "Spectral affinity status guards")
        unchanged(source, words)
    finally:
        _ = source^
        _ = words^
        _ = expected^
        _ = destination^
        _ = status^
