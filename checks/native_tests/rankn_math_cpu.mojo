# SPDX-License-Identifier: Apache-2.0
"""Four independent native session fixtures run through the host pool."""
from autograd.ops_api import OpsSession
from checks.native_tests.rankn_math_cases import RANKN_MATH_FIXTURES, require, run_rankn_math_fixture
from core.host_parallel import host_parallelize
from sequence.exec import HostExec


def main() raises:
    var failures = List[Int](length=RANKN_MATH_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(fixture_id: Int) {imm fp}:
        try:
            var s = OpsSession[HostExec](HostExec(), fixture_id == 0)
            try:
                run_rankn_math_fixture(s, fixture_id)
            except error:
                s.close()
                raise error
            s.close()
        except error:
            print("FAIL rankn_math_cpu fixture", fixture_id, String(error))
            fp.unsafe_store(fixture_id, 1)
    host_parallelize(task, RANKN_MATH_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "rankn_math_cpu failed " + String(failed) + " fixtures")
    print("PASS rankn_math_cpu:", RANKN_MATH_FIXTURES, "native broadcast-gradient/softmax fixtures, parallel")
