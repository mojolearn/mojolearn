# SPDX-License-Identifier: Apache-2.0
"""Independent kernel cells and the original Nystroem basis reconstruction.

Recorded sklearn Float64 matrices use the original Float32 inputs. Native
fixture I/O/preparation/comparisons are parallel. IdentityTrace is disabled
in the device adapter; no hashes, repeatability run or performance claim.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.family_cast_impl import family_min_f32
from kernel_methods.host.km_host_oracle import kmh_kernel_matrix, kmh_nystroem_fit, kmh_nystroem_transform
from core.identity_trace import IdentityTrace
from svm.impl.svm_parameter import KernelParams
from kernel_methods.estimator import _family_ctx, nystroem_fit_host, nystroem_transform_host
from kernel_methods.checks.kernel_matrix import km_kernel_matrix, km_kernel_workspace_floats
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


comptime KERNEL_METHODS_CELLS_GPU_FIXTURES = 21
comptime KERNEL_METHODS_CELLS_CPU_FIXTURES = 27
comptime CELL_ATOL = Float64(5e-3)
comptime CELL_RTOL = Float64(1e-7)


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
        require(indices[i] >= 0 and indices[i] < 80, "Nystroem basis index outside original input")
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


# ---- The host column (was kernel_methods_cells_cpu_suite.mojo) ----
# Actual host kernel-method operations; independent fixtures run in parallel.
struct HostKernelCells(KernelMethodsCellCalls):
    @staticmethod
    def matrix(a: List[Float32], b: List[Float32], m: Int, n: Int, d: Int,
               kernel: Int, gamma: Float64) raises -> List[Float32]:
        return kmh_kernel_matrix(kernel, 3, gamma, 1.0, a, b, m, n, d)

    @staticmethod
    def basis(x: List[Float32], kernel: Int) raises -> Tuple[List[Int32], List[Float32]]:
        var gamma = Float64(1) / Float64(6) if kernel == 6 else Float64(1)
        var model = kmh_nystroem_fit(x, 80, 6, kernel, 3, gamma, 1.0, 12, UInt64(1))
        require(len(model.indices) == 12, "Nystroem host basis index extent")
        for i in range(12):
            require(model.indices[i] >= 0 and model.indices[i] < 80, "Nystroem host basis index outside input")
        var basis = List[Float32](length=72, fill=Float32(0))
        var ip = Pointer[Int32, ImmUntrackedOrigin](unsafe_from_address=Int(model.indices.unsafe_ptr()))
        var xp = x.unsafe_ptr()
        var bp = basis.unsafe_ptr()
        def gather(at: Int) {imm ip, imm xp, imm bp}:
            bp.unsafe_store(at, xp[Int(ip[at // 6]) * 6 + at % 6])
        host_parallelize(gather, 72)
        var embedded = kmh_nystroem_transform(model.components, model.normalization,
            12, 6, kernel, 3, gamma, 1.0, basis, 12)
        var indices = model.indices.copy()
        _ = model^
        _ = basis^
        _ = x
        return (indices^, embedded^)


def _run_cpu(mut report: Report) raises:
    var cases = report.cases(KERNEL_METHODS_CELLS_CPU_FIXTURES)
    def task(index: Int) {imm cases}:
        try:
            fixture[HostKernelCells](index)
        except error:
            cases.fail(index, error)
    report.parallel(task, KERNEL_METHODS_CELLS_CPU_FIXTURES)


# ---- The device column (was kernel_methods_cells_gpu_suite.mojo) ----
# Actual GPU matrix/Nystroem paths, ordered on their existing device context.
#
# A device stream is serial by contract; fixture submissions are ordered.
# Independent input decoding, gathers and output checks use host_parallelize.
# No tuned kernel, dispatch choice, fusion or numerical setting is replaced.
comptime CELL_OUTPUT_GUARD = Float32(27182.5)


struct DeviceKernelCells(KernelMethodsCellCalls):
    @staticmethod
    def matrix(a: List[Float32], b: List[Float32], m: Int, n: Int, d: Int,
               kernel: Int, gamma: Float64) raises -> List[Float32]:
        var ctx = _family_ctx()
        var da = ctx.enqueue_create_buffer[DType.float32](max(1, len(a)))
        var db = ctx.enqueue_create_buffer[DType.float32](max(1, len(b)))
        var dst = ctx.enqueue_create_buffer[DType.float32](m * n + 2)
        var norm_a = ctx.enqueue_create_buffer[DType.float32](max(1, m))
        var norm_b = ctx.enqueue_create_buffer[DType.float32](max(1, n))
        var workspace = ctx.enqueue_create_buffer[DType.float32](max(1, km_kernel_workspace_floats(m, n, d)))
        if len(a):
            ctx.enqueue_copy(dst_buf=da, src_ptr=a.unsafe_ptr())
        if len(b):
            ctx.enqueue_copy(dst_buf=db, src_ptr=b.unsafe_ptr())
        ctx.enqueue_memset(dst, CELL_OUTPUT_GUARD)
        var parameters = KernelParams(kernel, 3, gamma, 1.0)
        km_kernel_matrix(ctx, parameters, dst, da, db, m, n, d, norm_a, norm_b, workspace)
        var downloaded = List[Float32](length=m * n + 2, fill=Float32(0))
        ctx.enqueue_copy(dst_ptr=downloaded.unsafe_ptr(), src_buf=dst)
        ctx.synchronize()
        require(downloaded[m * n] == CELL_OUTPUT_GUARD and downloaded[m * n + 1] == CELL_OUTPUT_GUARD,
            "GPU matrix wrote beyond its requested cells or touched empty output")
        var result = List[Float32](length=m * n, fill=Float32(0))
        var sp = downloaded.unsafe_ptr()
        var rp = result.unsafe_ptr()
        def copy(at: Int) {imm sp, imm rp}:
            rp.unsafe_store(at, sp[at])
        host_parallelize(copy, m * n)
        _ = downloaded^
        _ = da^
        _ = db^
        _ = dst^
        _ = norm_a^
        _ = norm_b^
        _ = workspace^
        _ = a
        _ = b
        _ = ctx^
        return result^

    @staticmethod
    def basis(x: List[Float32], kernel: Int) raises -> Tuple[List[Int32], List[Float32]]:
        var trace = IdentityTrace.disabled()
        var gamma = Float64(1) / Float64(6) if kernel == 6 else Float64(1)
        var parameters = KernelParams(kernel, 3, gamma, 1.0)
        var model = nystroem_fit_host(x, 80, 6, parameters, 12, UInt64(1), trace)
        require(len(model.component_indices) == 12, "Nystroem device basis index extent")
        for i in range(12):
            require(model.component_indices[i] >= 0 and model.component_indices[i] < 80, "Nystroem device basis index outside input")
        var basis = List[Float32](length=72, fill=Float32(0))
        var ip = Pointer[Int32, ImmUntrackedOrigin](unsafe_from_address=Int(model.component_indices.unsafe_ptr()))
        var xp = x.unsafe_ptr()
        var bp = basis.unsafe_ptr()
        def gather(at: Int) {imm ip, imm xp, imm bp}:
            bp.unsafe_store(at, xp[Int(ip[at // 6]) * 6 + at % 6])
        host_parallelize(gather, 72)
        var embedded = nystroem_transform_host(model, basis, 12, trace)
        var indices = model.component_indices.copy()
        _ = model^
        _ = trace^
        _ = basis^
        _ = x
        return (indices^, embedded^)


def _run_gpu(mut report: Report) raises:
    var cases = report.cases(KERNEL_METHODS_CELLS_GPU_FIXTURES)
    for index in range(KERNEL_METHODS_CELLS_GPU_FIXTURES):
        try:
            fixture[DeviceKernelCells](index)
        except error:
            cases.fail(index, error)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if CPU_COLUMN_BUILD:
            _run_cpu(report)
        else:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("kernel_methods_cells", cpu=True, gpu=True, serial="", run=run_cases)
