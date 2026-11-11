# SPDX-License-Identifier: Apache-2.0
"""Independent actual host KernelPCA graphs run across host cores."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.host_parallel import host_parallelize
from core.neighbors_native_driver import XnDriver
from core.kpca_native_api import KPCANativeDriver, kpca_native_v1
from core.kpca_native_provider import KPCAEigenKernels
from core.kpca_native_plan import KPCANeighborDriver
from x_neighbors.native_descriptor_host import xn_native_v1
from x_decomp.descriptor_host import x_decomp_host_v1
from x_decomp.host import HostExec
from x_decomp.pca_randomized import NoRDev
from x_decomp.decomp_owner_host import HostDecomp
from checks.native_tests.kpca_native_cases import KPCA_NATIVE_FIXTURES, fixture, require


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var failures = List[Int](length=KPCA_NATIVE_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(KPCANativeDriver(kpca_native_v1[KPCAEigenKernels[HostExec, NoRDev, False, HostDecomp]]),
                KPCANeighborDriver(XnDriver(xn_native_v1)), x_decomp_host_v1, unit, True)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL kpca_native_cpu fixture", unit, String(error))
    host_parallelize(task, KPCA_NATIVE_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "kpca_native_cpu failed " + String(failed) + " fixtures")
    print("PASS kpca_native_cpu:", KPCA_NATIVE_FIXTURES, "native centering/eigen/query/refusal fixtures")
