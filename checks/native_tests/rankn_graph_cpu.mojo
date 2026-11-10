# SPDX-License-Identifier: Apache-2.0
"""Independent graph sessions and attention-oracle queries run in parallel."""
from autograd.ops_api import OpsSession
from checks.native_tests.rankn_graph_cases import RANKN_GRAPH_FIXTURES, run_rankn_graph_fixture
from checks.native_tests.rankn_math_cases import require
from core.host_parallel import host_parallelize
from sequence.exec import HostExec


def main() raises:
    var failures = List[Int](length=RANKN_GRAPH_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(fixture_id: Int) {imm fp}:
        try:
            var s = OpsSession[HostExec](HostExec(), True)
            try:
                run_rankn_graph_fixture(s, fixture_id)
            except error:
                s.close()
                raise error
            s.close()
        except error:
            print("FAIL rankn_graph_cpu fixture", fixture_id, String(error))
            fp.unsafe_store(fixture_id, 1)
    host_parallelize(task, RANKN_GRAPH_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "rankn_graph_cpu failed " + String(failed) + " fixtures")
    print("PASS rankn_graph_cpu:", RANKN_GRAPH_FIXTURES, "native attention/layout graphs; parallel fixtures and oracle queries")
