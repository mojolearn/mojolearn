# SPDX-License-Identifier: Apache-2.0
"""GARCH(1, 1) recovery of the stored simulated process: the independent fixtures run across host cores, each on its own
HostExec with a private failure slot."""
from core.host_parallel import host_parallelize
from sequence.exec import HostExec
from checks.native_tests.sequence_port_common import require, count_failures
from checks.native_tests.sequence_garch_cases import GARCH_FIXTURES, run_garch_fixture


def run_checks() raises:
    var failures = List[Int](length=GARCH_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            var ex = HostExec()
            run_garch_fixture(ex, unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL sequence_garch_cpu fixture", unit, String(error))
    host_parallelize(task, GARCH_FIXTURES)
    var failed = count_failures(failures)
    require(failed == 0, "sequence_garch_cpu failed " + String(failed) + " fixtures")
    print("PASS sequence_garch_cpu:", GARCH_FIXTURES, "fixtures")
