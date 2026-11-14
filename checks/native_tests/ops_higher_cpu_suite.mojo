# SPDX-License-Identifier: Apache-2.0
"""Independent higher-order gradient fixtures, each on its own two recording
host sessions, run across host cores."""
from autograd.ops_api import OpsSession
from checks.native_tests.ops_higher_cases import OPS_HIGHER_FIXTURES, require, run_ops_higher_fixture
from core.host_parallel import host_parallelize
from sequence.exec import HostExec


def run_checks() raises:
    var failures = List[Int](length=OPS_HIGHER_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(fixture_id: Int) {imm fp}:
        try:
            var s = OpsSession[HostExec](HostExec(), True)
            var again = OpsSession[HostExec](HostExec(), True)
            try:
                run_ops_higher_fixture(s, again, fixture_id)
            except error:
                s.close()
                again.close()
                raise error
            s.close()
            again.close()
        except error:
            print("FAIL ops_higher_cpu fixture", fixture_id, String(error))
            fp.unsafe_store(fixture_id, 1)
    host_parallelize(task, OPS_HIGHER_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "ops_higher_cpu failed " + String(failed) + " of " + String(OPS_HIGHER_FIXTURES))
    print("PASS ops_higher_cpu:", OPS_HIGHER_FIXTURES, "higher-order gradient fixtures, parallel")
