# SPDX-License-Identifier: Apache-2.0
"""Original GP fixture bytes and declarative kernel settings; no Python."""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.gp_native_storage import GPKernelStorage
from checks.native_tests.gp_state_suite import require

comptime GP_GUARD = Float32(123456.75)
comptime GP_GUARD64 = Float64(987654.125)


def input_values(name: String, part: String, count: Int) raises -> List[Float32]:
    var values = List[Float32](unsafe_uninit_length=count)
    with open("checks/native_tests/fixtures/gp/" + name + "-" + part + ".f32", "r") as file:
        var bytes = file.read_bytes()
        require(len(bytes) == 4 * count, "GP input fixture extent")
        var bp = bytes.unsafe_ptr()
        var vp = values.unsafe_ptr()
        def word(index: Int) {imm bp, imm vp}:
            var at = 4 * index
            var bits = UInt32(bp[at]) | (UInt32(bp[at + 1]) << 8) | (UInt32(bp[at + 2]) << 16) | (UInt32(bp[at + 3]) << 24)
            vp.unsafe_store(index, bitcast[DType.float32](bits))
        host_parallelize(word, count)
        _ = bytes^
    return values^


def reference_values(name: String, part: String, count: Int) raises -> List[Float64]:
    var values = List[Float64](unsafe_uninit_length=count)
    with open("checks/native_tests/fixtures/gp/" + name + "-" + part + ".f64", "r") as file:
        var bytes = file.read_bytes()
        require(len(bytes) == 8 * count, "GP reference fixture extent")
        var bp = bytes.unsafe_ptr()
        var vp = values.unsafe_ptr()
        def word(index: Int) {imm bp, imm vp}:
            var bits = UInt64(0)
            for byte in range(8):  # Fixed-width word decoding.
                bits |= UInt64(bp[index * 8 + byte]) << UInt64(8 * byte)
            vp.unsafe_store(index, bitcast[DType.float64](bits))
        host_parallelize(word, count)
        _ = bytes^
    return values^


def kernel_storage(kind: Int) -> GPKernelStorage:
    """0=surface,1=default,2..5=original gradients,6=restart,7=singular."""
    var value = GPKernelStorage()
    value.kinds = [0, 2, 5, 1, 4]
    value.parameters = [1.0, 0.0, 0.0, 0.1, 0.0]
    value.lengths = [0, 1, 0, 0, 0]
    value.scales = [1.0]
    if kind == 0:
        value.parameters[3] = 0.0
        value.lengths[1] = 2
        value.scales = [0.25, 0.25]
    elif kind == 2:
        value.parameters[0] = 1.5
        value.parameters[3] = 0.2
        value.lengths[1] = 3
        value.scales = [0.8, 1.2, 2.0]
    elif kind == 3:
        value.kinds[1] = 3
        value.parameters[1] = 0.5
        value.scales = [1.1]
    elif kind == 4:
        value.kinds[1] = 3
        value.parameters[0] = 0.7
        value.parameters[1] = 1.5
        value.lengths[1] = 3
        value.scales = [1.0, 0.6, 1.4]
    elif kind == 5:
        value.kinds = [3, 0, 4, 1, 4]
        value.parameters = [2.5, 0.3, 0.0, 0.2, 0.0]
        value.lengths = [1, 0, 0, 0, 0]
        value.scales = [0.9]
    elif kind == 6:
        value.kinds[1] = 3
        value.parameters[1] = 2.5
        value.lengths[1] = 3
        value.scales = [1.0, 1.0, 1.0]
    elif kind == 7:
        value.kinds = [2]
        value.parameters = [0.0]
        value.lengths = [1]
    value.finish()
    return value^


def guards(values: List[Float32], message: String) raises:
    require(values[0] == GP_GUARD and values[len(values) - 1] == GP_GUARD, message)
    _ = values


def guards64(values: List[Float64], message: String) raises:
    require(values[0] == GP_GUARD64 and values[len(values) - 1] == GP_GUARD64, message)
    _ = values


def failures_clear(values: List[Int], message: String) raises:
    for value in values:  # Reporting the independently evaluated predicates.
        require(value == 0, message)
    _ = values


def pairwise_level(source: List[Float64], rows: Int, width: Int) -> List[Float64]:
    var next_width = (width + 1) // 2
    var result = List[Float64](unsafe_uninit_length=rows * next_width)
    var sp = source.unsafe_ptr()
    var rp = result.unsafe_ptr()
    def cell(index: Int) {imm sp, imm rp, imm width, imm next_width}:
        var row = index // next_width
        var column = 2 * (index % next_width)
        var value = sp[row * width + column]
        if column + 1 < width:
            value += sp[row * width + column + 1]
        rp.unsafe_store(index, value)
    host_parallelize(cell, rows * next_width)
    _ = source
    return result^


def sample_terms(values: List[Float32], offset: Int, rows: Int, count: Int,
                 centers: List[Float64], squared: Bool) -> List[Float64]:
    var work = List[Float64](unsafe_uninit_length=rows * count)
    var vp = values.unsafe_ptr() + offset
    var wp = work.unsafe_ptr()
    var cp = centers.unsafe_ptr()
    def cell(index: Int) {imm vp, imm wp, imm cp, imm squared, imm count}:
        var value = Float64(vp[index])
        if squared:
            value -= cp[index // count]
            value *= value
        wp.unsafe_store(index, value)
    host_parallelize(cell, rows * count)
    _ = values
    _ = centers
    return work^


def row_sums(values: List[Float32], offset: Int, rows: Int, count: Int,
             centers: List[Float64], squared: Bool) -> List[Float64]:
    """Independent Float64 test reduction; all cells/each tree level parallel."""
    var work = sample_terms(values, offset, rows, count, centers, squared)
    var width = count
    while width > 1:  # Dependent reduction levels; each level runs in parallel.
        var next = pairwise_level(work, rows, width)
        work = next^
        width = (width + 1) // 2
    _ = values
    _ = centers
    return work^
