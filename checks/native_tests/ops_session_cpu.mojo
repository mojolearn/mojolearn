# SPDX-License-Identifier: Apache-2.0
"""Independent typed host sessions run in parallel on the existing host pool."""
from autograd.ops_api import OpsSession
from checks.native_tests.ops_session_cases import OPS_SESSION_FIXTURES, require, run_ops_session_fixture
from core.host_parallel import host_parallelize
from sequence.exec import HostExec


def main() raises:
    var failures = List[Int](length=OPS_SESSION_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(fixture_id: Int) {imm fp}:
        try:
            var s = OpsSession[HostExec](HostExec(), False)
            try:
                run_ops_session_fixture(s, fixture_id)
            except error:
                s.close()
                raise error
            s.close()
        except error:
            print("FAIL ops_session_cpu fixture", fixture_id, String(error))
            fp.unsafe_store(fixture_id, 1)
    host_parallelize(task, OPS_SESSION_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "ops_session_cpu failed " + String(failed) + " of " + String(OPS_SESSION_FIXTURES))
    print("PASS ops_session_cpu:", OPS_SESSION_FIXTURES, "native fixtures, parallel")
