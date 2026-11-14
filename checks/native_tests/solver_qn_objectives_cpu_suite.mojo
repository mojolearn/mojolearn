# SPDX-License-Identifier: Apache-2.0
"""LinearSVC/LinearSVR documented objectives on the host QN provider; the ten
independent fixtures run across host cores."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.host_parallel import host_parallelize
from core.linear_svc_native_api import LinearSvcDriver
from glm.host.linear_svc_native_host import linear_svc_host_v1
from checks.native_tests.solver_qn_objectives_cases import QN_OBJECTIVE_FIXTURES, fixture, require


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var failures = List[Int](length=QN_OBJECTIVE_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(LinearSvcDriver(linear_svc_host_v1), unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL solver_qn_objectives_cpu fixture", unit, String(error))
    host_parallelize(task, QN_OBJECTIVE_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "solver_qn_objectives_cpu failed " + String(failed) + " fixtures")
    print("PASS solver_qn_objectives_cpu:", QN_OBJECTIVE_FIXTURES, "documented QN objectives")
