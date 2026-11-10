# SPDX-License-Identifier: Apache-2.0
"""Independent kernel cells and the original Nystroem basis reconstruction.

Recorded sklearn Float64 matrices use the original Float32 inputs. Native
fixture I/O/preparation/comparisons are parallel. IdentityTrace is disabled
in the device adapter; no hashes, repeatability run or performance claim.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.family_cast_impl import family_min_f32

comptime KERNEL_METHODS_CELLS_GPU_FIXTURES = 21
comptime KERNEL_METHODS_CELLS_CPU_FIXTURES = 27
comptime CELL_ATOL = Float64(5e-3)
comptime CELL_RTOL = Float64(1e-7)


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


trait KernelMethodsCellCalls:
    @staticmethod
    def matrix(a: List[Float32], b: List[Float32], m: Int, n: Int, d: Int,
               kernel: Int, gamma: Float64) raises -> List[Float32]: ...
    @staticmethod
    def basis(x: List[Float32], kernel: Int) raises -> Tuple[List[Int32], List[Float32]]: ...


def input_values(name: String, count: Int, recorded_count: Int) raises -> List[Float32]:
    var result = List[Float32](length=count, fill=Float32(0))
    with open("checks/native_tests/fixtures/kernel_methods_cells/" + name + "-input.f32", "r") as file:
        var bytes = file.read_bytes()
        require(len(bytes) == recorded_count * 4 and count <= recorded_count, "kernel input fixture extent changed")
        var bp = bytes.unsafe_ptr()
        var rp = result.unsafe_ptr()
        def word(index: Int) {imm bp, imm rp}:
            var at = index * 4
            var bits = UInt32(bp[at]) | (UInt32(bp[at + 1]) << 8) | (UInt32(bp[at + 2]) << 16) | (UInt32(bp[at + 3]) << 24)
            rp.unsafe_store(index, bitcast[DType.float32](bits))
        host_parallelize(word, count)
        _ = bytes^
    return result^


def reference_values(name: String, kind: Int, count: Int) raises -> List[Float64]:
    var names: List[String] = ["cosine", "chi2", "chi2-g03", "additive-chi2"]
    var result = List[Float64](length=count, fill=Float64(0))
    with open("checks/native_tests/fixtures/kernel_methods_cells/" + name + "-" + names[kind] + ".f64", "r") as file:
        var bytes = file.read_bytes()
        require(len(bytes) == count * 8, "kernel reference fixture extent changed")
        var bp = bytes.unsafe_ptr()
        var rp = result.unsafe_ptr()
        def word(index: Int) {imm bp, imm rp}:
            var bits = UInt64(0)
            for byte in range(8):  # Fixed binary64 decoding, no numerical oracle.
                bits |= UInt64(bp[index * 8 + byte]) << UInt64(byte * 8)
            rp.unsafe_store(index, bitcast[DType.float64](bits))
        host_parallelize(word, count)
        _ = bytes^
    return result^


def check_cells(actual: List[Float32], expected: List[Float64], rows: Int, cols: Int, stride: Int) raises:
    require(len(actual) == rows * cols, "native kernel cell extent changed")
    var failures = List[Int](length=rows * cols, fill=0)
    var ap = actual.unsafe_ptr()
    var ep = expected.unsafe_ptr()
    var fp = failures.unsafe_ptr()
    def cell(index: Int) {imm ap, imm ep, imm fp, imm cols, imm stride}:
        var reference = ep[(index // cols) * stride + index % cols]
        var difference = abs(Float64(ap[index]) - reference)
        if not (difference <= CELL_ATOL + CELL_RTOL * abs(reference)):
            fp.unsafe_store(index, 1)
    host_parallelize(cell, rows * cols)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "native kernel matrix differs from independent sklearn cells: " + String(failed))
    _ = actual
    _ = expected


def cell_fixture[Calls: KernelMethodsCellCalls](index: Int) raises:
    var name = String("original")
    var kind = index % 4
    var rows = 79 if index < 4 else (80 if index < 8 else 81)
    var cols = 81 if index < 4 else (80 if index < 8 else 79)
    var stride = 81
    var dimensions = 6
    if index >= 12:
        name = "edges"
        rows = 7
        cols = 7
        stride = 7
        dimensions = 4
    if index == 16:
        name = "negative"
        kind = 0
        rows = 4
        cols = 4
        stride = 4
        dimensions = 2
    var kernel = 6 if kind == 0 else (8 if kind == 3 else 7)
    var gamma = Float64(0.3) if kind == 2 else (Float64(1) / Float64(dimensions) if kind == 0 else Float64(1))
    var a = input_values(name, rows * dimensions, stride * dimensions)
    var b = input_values(name, cols * dimensions, stride * dimensions)
    var expected = reference_values(name, kind, stride * stride)
    var actual = Calls.matrix(a, b, rows, cols, dimensions, kernel, gamma)
    check_cells(actual, expected, rows, cols, stride)
    _ = a^
    _ = b^
    _ = actual^
    _ = expected^


def empty_fixture[Calls: KernelMethodsCellCalls](index: Int) raises:
    # Raw matrix drivers allow an empty result; public estimators still
    # require positive input dimensions at their separate admission boundary.
    var rows = 0 if index == 0 else 4
    var cols = 4 if index == 0 else 0
    var a = List[Float32](length=rows * 6, fill=Float32(1))
    var b = List[Float32](length=cols * 6, fill=Float32(1))
    var result = Calls.matrix(a, b, rows, cols, 6, 7, 1.0)
    require(len(result) == 0, "raw empty kernel matrix should contain no cells")


def nystroem_fixture[Calls: KernelMethodsCellCalls](index: Int) raises:
    # Exact numerical portion moved from test_km_kernels.py:80x6,12basis
    # rows,random_state1,cosine/chi2 and the same5e-3+1e-7relative bound.
    var x = input_values("original", 80 * 6, 81 * 6)
    var expected = reference_values("original", index, 81 * 81)
    var result = Calls.basis(x, 6 if index == 0 else 7)
    var indices = result[0].copy()
    var embedded = result[1].copy()
    require(len(indices) == 12 and len(embedded) == 144, "Nystroem basis output extent changed")
    for i in range(12):  # Fixed model metadata, checked before reference indexing.
        require(0 <= indices[i] and indices[i] < 80, "Nystroem basis index outside original input")
        for j in range(i):
            require(indices[i] != indices[j], "Nystroem basis index repeated")
    var failures = List[Int](length=144, fill=0)
    var ip = indices.unsafe_ptr()
    var zp = embedded.unsafe_ptr()
    var ep = expected.unsafe_ptr()
    var fp = failures.unsafe_ptr()
    def cell(at: Int) {imm ip, imm zp, imm ep, imm fp}:
        var row = at // 12
        var col = at % 12
        var actual = Float64(0)
        for component in range(12):  # Independent Float64 Gram of the produced basis embedding.
            actual += Float64(zp[row * 12 + component]) * Float64(zp[col * 12 + component])
        var reference = ep[Int(ip[row]) * 81 + Int(ip[col])]
        if not (abs(actual - reference) <= CELL_ATOL + CELL_RTOL * abs(reference)):
            fp.unsafe_store(at, 1)
    host_parallelize(cell, 144)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "Nystroem basis Gram differs from original sklearn reference: " + String(failed))
    _ = result^
    _ = indices^
    _ = embedded^
    _ = expected^
    _ = x^


def negative_minimum_fixture(index: Int) raises:
    # Exact production reduction consumed by the chi2 public refusal. The
    # actual exception/estimator integration remains in the Python seam.
    var count = 65537
    var values = List[Float32](length=count, fill=Float32(1))
    var expected = Float32(1)
    if index == 1:
        values[0] = bitcast[DType.float32](UInt32(0x80000000))
        expected = 0
    elif index >= 2:
        var at = 0 if index == 2 else (count // 2 if index == 3 else (count - 1 if index == 4 else 32767))
        values[at] = -1
        expected = -1
    var pointer = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(values.unsafe_ptr()))
    var actual = family_min_f32(pointer, count)
    require(actual == expected and (actual < 0) == (index >= 2), "chi2 negative-input reduction changed")
    _ = values^


def fixture[Calls: KernelMethodsCellCalls](index: Int) raises:
    if index < 17:
        cell_fixture[Calls](index)
    elif index < 19:
        empty_fixture[Calls](index - 17)
    elif index < 21:
        nystroem_fixture[Calls](index - 19)
    else:
        negative_minimum_fixture(index - 21)
