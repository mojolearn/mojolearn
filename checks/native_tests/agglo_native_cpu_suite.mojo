# SPDX-License-Identifier: Apache-2.0
"""Independent Agglomerative numerical graphs run across host cores.

Existing kernels and independent oracle cells remain parallel. Ordered GPU
execution does not clear the separately recorded shared-context submission bug.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.host_parallel import host_parallelize
from core.neighbors_native_driver import XnDriver
from core.agglo_native_api import AggloSolverDriver, AggloClusterDriver, AggloZeroRowsDriver
from core.dbscan_native_api import DBSCANNativeDriver, dbscan_native_v1
from core.dbscan_native_host import DBSCANHostKernels
from hierarchy.solver_descriptor_host import hierarchy_solver_host_v1
from hierarchy.cluster_descriptor_host import hierarchy_cluster_host_v1
from x_neighbors.native_descriptor_host import xn_native_v1
from checks.native_tests.agglo_native_cases import AGGLO_NATIVE_FIXTURES, fixture, require


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var failures = List[Int](length=AGGLO_NATIVE_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(AggloSolverDriver(hierarchy_solver_host_v1), AggloClusterDriver(hierarchy_cluster_host_v1),
                AggloZeroRowsDriver(XnDriver(xn_native_v1)), DBSCANNativeDriver(dbscan_native_v1[DBSCANHostKernels]),
                unit, True)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL agglo_native_cpu fixture", unit, String(error))
    host_parallelize(task, AGGLO_NATIVE_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "agglo_native_cpu failed " + String(failed) + " fixtures")
    print("PASS agglo_native_cpu:", AGGLO_NATIVE_FIXTURES, "native solver/tree/query/zero-row fixtures")
