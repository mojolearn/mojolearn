# SPDX-License-Identifier: Apache-2.0
"""Native GPU attention/layout graphs; ordered shared-context submissions.

DeviceExec owns a shared sequence context/pool. Existing kernels and host
oracle queries provide parallel work; session concurrency is not claimed.
"""
from autograd.ops_api import OpsSession
from checks.native_tests.rankn_graph_cases import RANKN_GRAPH_FIXTURES, run_rankn_graph_fixture
from checks.native_tests.rankn_math_cases import require
from sequence.exec_device import DeviceExec


def main() raises:
    var failed = 0
    for fixture_id in range(RANKN_GRAPH_FIXTURES):
        try:
            var s = OpsSession[DeviceExec](DeviceExec(), True)
            try:
                run_rankn_graph_fixture(s, fixture_id)
            except error:
                s.close()
                raise error
            s.close()
        except error:
            print("FAIL rankn_graph_gpu fixture", fixture_id, String(error))
            failed += 1
    require(failed == 0, "rankn_graph_gpu failed " + String(failed) + " fixtures")
    print("PASS rankn_graph_gpu:", RANKN_GRAPH_FIXTURES, "native graphs; shared-context submission ordered")
