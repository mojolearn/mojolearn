# SPDX-License-Identifier: Apache-2.0
"""Agglomerative graphs run one at a time on one physical GPU/shared contexts.

Existing kernels and independent oracle cells remain parallel. Ordered GPU
execution does not clear the separately recorded shared-context submission bug.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.host_parallel import host_parallelize
from core.neighbors_native_driver import XnDriver
from core.agglo_native_api import AggloSolverDriver, AggloClusterDriver, AggloZeroRowsDriver
from core.dbscan_native_api import DBSCANNativeDriver, dbscan_native_v1
from core.dbscan_native_device import DBSCANDeviceKernels
from hierarchy.solver_descriptor_device import hierarchy_solver_device_v1
from hierarchy.cluster_descriptor_device import hierarchy_cluster_device_v1
from x_neighbors.native_descriptor_device import xn_native_v1
from checks.native_tests.agglo_native_cases import AGGLO_NATIVE_FIXTURES, fixture, require
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_CPU


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and TARGET_COLUMN != COLUMN_CPU
    var failed = 0
    for unit in range(AGGLO_NATIVE_FIXTURES):
        try:
            fixture(AggloSolverDriver(hierarchy_solver_device_v1), AggloClusterDriver(hierarchy_cluster_device_v1),
                AggloZeroRowsDriver(XnDriver(xn_native_v1)), DBSCANNativeDriver(dbscan_native_v1[DBSCANDeviceKernels]),
                unit, False)
        except error:
            failed += 1
            print("FAIL agglo_native_gpu fixture", unit, String(error))
    require(failed == 0, "agglo_native_gpu failed " + String(failed) + " fixtures")
    print("PASS agglo_native_gpu:", AGGLO_NATIVE_FIXTURES, "native solver/tree/direct/resident/zero-row fixtures")
