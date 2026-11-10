# SPDX-License-Identifier: Apache-2.0
"""Parallel native Theta fitted-state fixtures, private HostExec per fixture."""
from core.host_parallel import host_parallelize
from sequence.exec import HostExec
from checks.native_tests.theta_state_cases import THETA_STATE_FIXTURES, require, run_theta_state_fixture


def run_checks() raises:
    var failures = List[Int](length=THETA_STATE_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(fixture_id: Int) {imm fp}:
        try:
            var ex = HostExec()
            run_theta_state_fixture(ex, fixture_id)
        except error:
            fp.unsafe_store(fixture_id, 1)
            print("FAIL theta_state_cpu fixture", fixture_id, String(error))
    host_parallelize(task, THETA_STATE_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "theta_state_cpu failed " + String(failed) + " fixtures")
    print("PASS theta_state_cpu:", THETA_STATE_FIXTURES, "parallel native fit/tail fixtures")
