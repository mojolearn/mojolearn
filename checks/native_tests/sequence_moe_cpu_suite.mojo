# SPDX-License-Identifier: Apache-2.0
"""MoEBlock forward against the float64 Mixtral block and the routing tie rule: the independent fixtures run across host cores, each on its own
HostExec with a private failure slot."""
from core.host_parallel import host_parallelize
from sequence.exec import HostExec
from checks.native_tests.sequence_port_common import require, count_failures
from checks.native_tests.sequence_moe_cases import MOE_FIXTURES, run_moe_fixture


def run_checks() raises:
    var failures = List[Int](length=MOE_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            var ex = HostExec()
            run_moe_fixture(ex, unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL sequence_moe_cpu fixture", unit, String(error))
    host_parallelize(task, MOE_FIXTURES)
    var failed = count_failures(failures)
    require(failed == 0, "sequence_moe_cpu failed " + String(failed) + " fixtures")
    print("PASS sequence_moe_cpu:", MOE_FIXTURES, "fixtures")
