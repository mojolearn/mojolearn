# SPDX-License-Identifier: Apache-2.0
"""STL decomposition identities, robust weights and batch rows: the independent fixtures run across host cores, each on its own
HostExec with a private failure slot."""
from core.host_parallel import host_parallelize
from sequence.exec import HostExec
from checks.native_tests.sequence_port_common import require, count_failures
from checks.native_tests.sequence_stl_cases import STL_FIXTURES, run_stl_fixture


def run_checks() raises:
    var failures = List[Int](length=STL_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            var ex = HostExec()
            run_stl_fixture(ex, unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL sequence_stl_cpu fixture", unit, String(error))
    host_parallelize(task, STL_FIXTURES)
    var failed = count_failures(failures)
    require(failed == 0, "sequence_stl_cpu failed " + String(failed) + " fixtures")
    print("PASS sequence_stl_cpu:", STL_FIXTURES, "fixtures")
