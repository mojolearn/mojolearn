# SPDX-License-Identifier: Apache-2.0
"""VAR(p) params and sigma_u against float64 least squares: the independent fixtures run across host cores, each on its own
HostExec with a private failure slot."""
from core.host_parallel import host_parallelize
from sequence.exec import HostExec
from checks.native_tests.sequence_port_common import require, count_failures
from checks.native_tests.sequence_var_cases import VAR_FIXTURES, run_var_fixture


def run_checks() raises:
    var failures = List[Int](length=VAR_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            var ex = HostExec()
            run_var_fixture(ex, unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL sequence_var_cpu fixture", unit, String(error))
    host_parallelize(task, VAR_FIXTURES)
    var failed = count_failures(failures)
    require(failed == 0, "sequence_var_cpu failed " + String(failed) + " fixtures")
    print("PASS sequence_var_cpu:", VAR_FIXTURES, "fixtures")
