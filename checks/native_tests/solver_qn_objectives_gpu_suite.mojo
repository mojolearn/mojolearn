# SPDX-License-Identifier: Apache-2.0
"""LinearSVC/LinearSVR documented objectives on the device QN provider.

The ten fixtures run in order: they submit to the shared family
DeviceContext, and concurrent submission from several host threads to one
context is a separately recorded open failure. The QN kernels keep their own
parallelism; the Float64 objective cells use host cores.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.linear_svc_native_api import LinearSvcDriver
from glm.linear_svc_native_device import linear_svc_device_v1
from checks.native_tests.solver_qn_objectives_cases import QN_OBJECTIVE_FIXTURES, fixture, require


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var failed = 0
    for unit in range(QN_OBJECTIVE_FIXTURES):
        try:
            fixture(LinearSvcDriver(linear_svc_device_v1), unit)
        except error:
            failed += 1
            print("FAIL solver_qn_objectives_gpu fixture", unit, String(error))
    require(failed == 0, "solver_qn_objectives_gpu failed " + String(failed) + " fixtures")
    print("PASS solver_qn_objectives_gpu:", QN_OBJECTIVE_FIXTURES, "documented QN objectives")
