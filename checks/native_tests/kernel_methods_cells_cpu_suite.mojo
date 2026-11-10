# SPDX-License-Identifier: Apache-2.0
"""Actual host kernel-method operations; independent fixtures run in parallel."""
from core.host_parallel import host_parallelize
from kernel_methods.host.km_host_oracle import kmh_kernel_matrix, kmh_nystroem_fit, kmh_nystroem_transform
from checks.native_tests.kernel_methods_cells_cases import (
    KernelMethodsCellCalls, KERNEL_METHODS_CELLS_CPU_FIXTURES, fixture, require,
)


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


def run_checks() raises:
    var failures = List[Int](length=KERNEL_METHODS_CELLS_CPU_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(index: Int) {imm fp}:
        try:
            fixture[HostKernelCells](index)
        except error:
            fp.unsafe_store(index, 1)
            print("FAIL kernel_methods_cells_cpu fixture", index, String(error))
    host_parallelize(task, KERNEL_METHODS_CELLS_CPU_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "kernel_methods_cells_cpu failed " + String(failed) + " fixtures")
    print("PASS kernel_methods_cells_cpu:", KERNEL_METHODS_CELLS_CPU_FIXTURES, "native matrix/basis/admission fixtures")
