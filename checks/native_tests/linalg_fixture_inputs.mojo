# SPDX-License-Identifier: Apache-2.0
"""The stored linalg inputs (fixtures/linalg_numerical/<name>.f32, raw little-endian float32 words), read for
linalg_numerical and linalg_decompositions. A helper module: no `main()`, no `test_*`."""
from std.memory import bitcast
from core.host_parallel import host_parallelize


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


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
