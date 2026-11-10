# SPDX-License-Identifier: Apache-2.0
"""Canonical typed optimizer requests: ordered shared-context GPU fixtures.

DeviceExec shares sequence_ctx/pool; existing numerical kernels remain parallel. No concurrent-session proof is claimed.
"""
from autograd.ops_api import OpsSession
from sequence.exec_device import DeviceExec
from checks.native_tests.ops_optimizer_requests_cases import (
    OPS_REQUEST_FIXTURES, OPS_REQUEST_MATH_FIXTURES, require,
    run_typed_optimizer, run_request_metadata,
)


def main() raises:
    var failed = 0
    for fixture_id in range(OPS_REQUEST_MATH_FIXTURES):
        try:
            var s = OpsSession[DeviceExec](DeviceExec(), True)
            var restored = OpsSession[DeviceExec](DeviceExec(), True)
            try:
                run_typed_optimizer(s, restored, fixture_id)
            except error:
                restored.close()
                s.close()
                raise error
            restored.close()
            s.close()
        except error:
            print("FAIL ops_optimizer_requests_gpu fixture", fixture_id, String(error))
            failed += 1
    require(failed == 0, "ops_optimizer_requests_gpu failed " + String(failed) + " fixtures")
    print("PASS ops_optimizer_requests_gpu:", OPS_REQUEST_MATH_FIXTURES, "typed request/state fixtures")
