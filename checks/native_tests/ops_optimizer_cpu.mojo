# SPDX-License-Identifier: Apache-2.0
"""Native optimizer fixtures; independent host sessions run in parallel.

Rows and parameter cells in the independent oracle use the host pool.
"""
from autograd.ops_api import OpsSession
from checks.native_tests.ops_optimizer_cases import OPS_OPTIMIZER_FIXTURES, require, run_ops_optimizer_fixture
from sequence.exec import HostExec
from core.host_parallel import host_parallelize


def main() raises:
    var failures = List[Int](length=OPS_OPTIMIZER_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(fixture_id: Int) {imm fp}:
        try:
            var s = OpsSession[HostExec](HostExec(), True)
            var restored = OpsSession[HostExec](HostExec(), fixture_id != 9)
            var model_ex = HostExec()
            try:
                run_ops_optimizer_fixture(s, restored, model_ex, fixture_id)
            except error:
                restored.close()
                s.close()
                raise error
            restored.close()
            s.close()
        except error:
            print("FAIL ops_optimizer_cpu fixture", fixture_id, String(error))
            fp.unsafe_store(fixture_id, 1)
    host_parallelize(task, OPS_OPTIMIZER_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "ops_optimizer_cpu failed " + String(failed) + " of " + String(OPS_OPTIMIZER_FIXTURES))
    print("PASS ops_optimizer_cpu:", OPS_OPTIMIZER_FIXTURES, "native fixtures; parallel host sessions")
