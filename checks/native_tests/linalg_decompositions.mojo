# SPDX-License-Identifier: Apache-2.0
"""Device and host linalg entries return the same words on the original inputs.

Ported from test_linalg_decompositions.py (the four `*_device_equals_host_byte_for_byte`
tests). The Python tests handed the same NumPy seed-5 matrices to the device
binding and the host binding through raw addresses; this calls the same entry
bodies those bindings call (`qr_r_device_into` / `qr_r_host_run`,
`svdvals_device_into` / `svdvals_host_run`, `DevExec.eigh` / `HostExec.eigh`
with UPLO 0) and compares every output word, sign bits and NaN payloads
included. The QR fixtures cover both slice arms (48x5 two slices, 16x4 one
slice); the eigh comparison includes the eigenvector sign flip.

Inputs: the linalg_numerical fixtures (qr-48x5, qr-16x4, constant-64x10) and
fixtures/linalg_decompositions/eigh-64x5-sym.f32 (generate.py beside it).
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.hostptr import f32_ptr
from decomposition.linalg_host_impl import qr_r_host_run, svdvals_host_run
from decomposition.linalg_door_impl import qr_r_device_into, svdvals_device_into
from x_decomp.cells import F32Ptr
from x_decomp.device import DevExec
from x_decomp.host import HostExec
from checks.native_tests.linalg_numerical import input_values
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_CPU
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


comptime LINALG_DECOMPOSITIONS_FIXTURES = 5
comptime DECOMP_GUARD = Float32(27182.5)


def eigh_input() raises -> List[Float32]:
    var result = List[Float32](unsafe_uninit_length=25)
    with open("checks/native_tests/fixtures/linalg_decompositions/eigh-64x5-sym.f32", "r") as file:
        var bytes = file.read_bytes()
        require(len(bytes) == 100, "eigh input fixture extent changed")
        var bp = bytes.unsafe_ptr()
        var rp = result.unsafe_ptr()
        def word(index: Int) {imm bp, imm rp}:
            var at = index * 4
            var bits = UInt32(bp[at]) | (UInt32(bp[at + 1]) << 8) | (UInt32(bp[at + 2]) << 16) | (UInt32(bp[at + 3]) << 24)
            rp.unsafe_store(index, bitcast[DType.float32](bits))
        host_parallelize(word, 25)
        _ = bytes^
    return result^


def same_words(device: List[Float32], host: List[Float32], name: String) raises:
    """Every word between the guards, compared as bits; the guards stay put."""
    require(len(device) == len(host), name + " output extents differ")
    var last = len(device) - 1
    require(device[0] == DECOMP_GUARD and device[last] == DECOMP_GUARD, name + " device output guards")
    require(host[0] == DECOMP_GUARD and host[last] == DECOMP_GUARD, name + " host output guards")
    var count = len(device) - 2
    var failures = List[Int](length=count, fill=0)
    var fp = failures.unsafe_ptr()
    var dp = device.unsafe_ptr() + 1
    var hp = host.unsafe_ptr() + 1
    def cell(index: Int) {imm fp, imm dp, imm hp}:
        fp.unsafe_store(index, Int(bitcast[DType.uint32](dp[index]) != bitcast[DType.uint32](hp[index])))
    host_parallelize(cell, count)
    var differ = 0
    for value in failures:
        differ += value
    require(differ == 0, name + ": device and host disagree in " + String(differ) + " words")
    _ = device
    _ = host


def unchanged(source: List[Float32], original: List[Float32], name: String) raises:
    var failures = List[Int](length=len(source), fill=0)
    var fp = failures.unsafe_ptr()
    var sp = source.unsafe_ptr()
    var op = original.unsafe_ptr()
    def cell(index: Int) {imm fp, imm sp, imm op}:
        fp.unsafe_store(index, Int(bitcast[DType.uint32](sp[index]) != bitcast[DType.uint32](op[index])))
    host_parallelize(cell, len(source))
    for value in failures:
        require(value == 0, name + " input was written")
    _ = source
    _ = original


def fixture(index: Int) raises:
    """One original comparison. Device work is submitted from this thread;
    the GPU suite runs the fixtures in order (see its docstring)."""
    if index < 2:
        var rows = 48 if index == 0 else 16
        var cols = 5 if index == 0 else 4
        var source = input_values("qr-48x5" if index == 0 else "qr-16x4", rows * cols)
        var original = source.copy()
        var device = List[Float32](length=cols * cols + 2, fill=DECOMP_GUARD)
        var host = List[Float32](length=cols * cols + 2, fill=DECOMP_GUARD)
        qr_r_device_into(source, rows, cols, f32_ptr(Int(device.unsafe_ptr()) + 4))
        qr_r_host_run(source, rows, cols, f32_ptr(Int(host.unsafe_ptr()) + 4))
        same_words(device, host, "QR R " + String(rows) + "x" + String(cols))
        unchanged(source, original, "QR")
        _ = device^
        _ = host^
    elif index < 4:
        var rows = 48 if index == 2 else 64
        var cols = 5 if index == 2 else 10
        var source = input_values("qr-48x5" if index == 2 else "constant-64x10", rows * cols)
        var original = source.copy()
        var device = List[Float32](length=cols + 2, fill=DECOMP_GUARD)
        var host = List[Float32](length=cols + 2, fill=DECOMP_GUARD)
        svdvals_device_into(source, rows, cols, f32_ptr(Int(device.unsafe_ptr()) + 4))
        svdvals_host_run(source, rows, cols, f32_ptr(Int(host.unsafe_ptr()) + 4))
        same_words(device, host, "svdvals " + String(rows) + "x" + String(cols))
        unchanged(source, original, "svdvals")
        _ = device^
        _ = host^
    else:
        var source = eigh_input()
        var device_source = source.copy()
        var host_source = source.copy()
        var device_values = List[Float32](length=7, fill=DECOMP_GUARD)
        var device_vectors = List[Float32](length=27, fill=DECOMP_GUARD)
        var host_values = List[Float32](length=7, fill=DECOMP_GUARD)
        var host_vectors = List[Float32](length=27, fill=DECOMP_GUARD)
        DevExec.eigh(F32Ptr(unsafe_from_address=Int(device_source.unsafe_ptr())),
                     F32Ptr(unsafe_from_address=Int(device_values.unsafe_ptr()) + 4),
                     F32Ptr(unsafe_from_address=Int(device_vectors.unsafe_ptr()) + 4), 5, 0)
        HostExec.eigh(F32Ptr(unsafe_from_address=Int(host_source.unsafe_ptr())),
                      F32Ptr(unsafe_from_address=Int(host_values.unsafe_ptr()) + 4),
                      F32Ptr(unsafe_from_address=Int(host_vectors.unsafe_ptr()) + 4), 5, 0)
        same_words(device_values, host_values, "eigh eigenvalues")
        same_words(device_vectors, host_vectors, "eigh eigenvectors")
        unchanged(device_source, source, "eigh device")
        unchanged(host_source, source, "eigh host")
        _ = device_values^
        _ = device_vectors^
        _ = host_values^
        _ = host_vectors^
        _ = device_source^
        _ = host_source^


# ---- The device column (was linalg_decompositions_gpu_suite.mojo) ----
# Device and host linalg entries agree word for word (from test_linalg_decompositions.py).
#
# The five comparisons run in order: each one submits graphs to the shared
# production DeviceContext, and concurrent submission from several host threads
# to that one context is a separately recorded open failure. The device kernels
# keep their own parallelism and the word comparisons run across host cores.
# A CPU-only column has one route and nothing to compare, so this suite is GPU only.
def _run_gpu(mut report: Report) raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and TARGET_COLUMN != COLUMN_CPU
    var cases = report.cases(LINALG_DECOMPOSITIONS_FIXTURES)
    for index in range(LINALG_DECOMPOSITIONS_FIXTURES):
        try:
            fixture(index)
        except error:
            cases.fail(index, error)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if not CPU_COLUMN_BUILD:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("linalg_decompositions", cpu=False, gpu=True, serial="", run=run_cases)
