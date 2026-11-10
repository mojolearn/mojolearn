# SPDX-License-Identifier: Apache-2.0
"""Independent typed host sessions run on the existing parallel host pool."""
from autograd.ops_api import OpsSession
from checks.native_tests.rankn_views_cases import RANKN_VIEW_FIXTURES, require, run_rankn_view_fixture
from core.host_parallel import host_parallelize
from sequence.exec import HostExec


def main() raises:
    var failures = List[Int](length=RANKN_VIEW_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(fixture_id: Int) {imm fp}:
        try:
            var s = OpsSession[HostExec](HostExec(), fixture_id == 16)
            try:
                run_rankn_view_fixture(s, fixture_id)
            except error:
                s.close()
                raise error
            s.close()
        except error:
            print("FAIL rankn_views_cpu fixture", fixture_id, String(error))
            fp.unsafe_store(fixture_id, 1)
    host_parallelize(task, RANKN_VIEW_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "rankn_views_cpu failed " + String(failed) + " of " + String(RANKN_VIEW_FIXTURES))
    print("PASS rankn_views_cpu:", RANKN_VIEW_FIXTURES, "native fixtures, parallel")
