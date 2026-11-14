# SPDX-License-Identifier: Apache-2.0
"""AutoTheta batch-equals-single and seasonal decomposition: the independent fixtures run across host cores, each on its own
HostExec with a private failure slot."""
from core.host_parallel import host_parallelize
from sequence.exec import HostExec
from checks.native_tests.sequence_port_common import require, count_failures
from checks.native_tests.sequence_theta_cases import THETA_FIXTURES, run_theta_fixture


def run_checks() raises:
    var failures = List[Int](length=THETA_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            var ex = HostExec()
            run_theta_fixture(ex, unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL sequence_theta_cpu fixture", unit, String(error))
    host_parallelize(task, THETA_FIXTURES)
    var failed = count_failures(failures)
    require(failed == 0, "sequence_theta_cpu failed " + String(failed) + " fixtures")
    print("PASS sequence_theta_cpu:", THETA_FIXTURES, "fixtures")
