# SPDX-License-Identifier: Apache-2.0
"""Raw little-endian fixture words of the tsa and metrics native checks
(`tools/tsa_metrics_native_fixtures.py` writes them). Importable; no
`main()`. Paths are relative to the repository root, where the pixi tasks
run. The word decode is one independent cell per word across host cores.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize

comptime TSA_FIXTURE_DIR = "checks/native_tests/fixtures/tsa/"
comptime METRICS_FIXTURE_DIR = "checks/native_tests/fixtures/metrics/"


def fixture_words(path: String, count: Int) raises -> List[UInt32]:
    var values = List[UInt32](length=count, fill=UInt32(0))
    with open(path, "r") as file:
        var bytes = file.read_bytes()
        if len(bytes) != 4 * count:
            raise Error("fixture extent: " + path + " holds " + String(len(bytes)) + " bytes, expected "
                        + String(4 * count))
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


def fixture_f32(path: String, count: Int) raises -> List[Float32]:
    var words = fixture_words(path, count)
    var values = List[Float32](length=count, fill=Float32(0))
    var wp = words.unsafe_ptr()
    var vp = values.unsafe_ptr()
    def cell(index: Int) {imm wp, imm vp}:
        vp.unsafe_store(index, bitcast[DType.float32](wp[index]))
    host_parallelize(cell, count)
    _ = words^
    return values^


def fixture_i32(path: String, count: Int) raises -> List[Int32]:
    var words = fixture_words(path, count)
    var values = List[Int32](length=count, fill=Int32(0))
    var wp = words.unsafe_ptr()
    var vp = values.unsafe_ptr()
    def cell(index: Int) {imm wp, imm vp}:
        vp.unsafe_store(index, bitcast[DType.int32](wp[index]))
    host_parallelize(cell, count)
    _ = words^
    return values^
