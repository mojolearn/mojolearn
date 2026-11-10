# SPDX-License-Identifier: Apache-2.0
"""Canonical typed optimizer requests: parallel host fixtures.

Metadata cases use no executor; math cases own independent sessions.
"""
from autograd.ops_api import OpsSession
from sequence.exec import HostExec
from checks.native_tests.ops_optimizer_requests_cases import (
    OPS_REQUEST_FIXTURES, OPS_REQUEST_MATH_FIXTURES, require,
    run_typed_optimizer, run_request_metadata,
)
from core.host_parallel import host_parallelize


def main() raises:
    var failures = List[Int](length=OPS_REQUEST_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(fixture_id: Int) {imm fp}:
        try:
            if fixture_id >= OPS_REQUEST_MATH_FIXTURES:
                run_request_metadata(fixture_id)
                return
            var s = OpsSession[HostExec](HostExec(), True)
            var restored = OpsSession[HostExec](HostExec(), True)
            try:
                run_typed_optimizer(s, restored, fixture_id)
            except error:
                restored.close()
                s.close()
                raise error
            restored.close()
            s.close()
        except error:
            print("FAIL ops_optimizer_requests_cpu fixture", fixture_id, String(error))
            fp.unsafe_store(fixture_id, 1)
    host_parallelize(task, OPS_REQUEST_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "ops_optimizer_requests_cpu failed " + String(failed) + " fixtures")
    print("PASS ops_optimizer_requests_cpu:", OPS_REQUEST_FIXTURES, "typed request/state fixtures")
