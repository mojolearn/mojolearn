# SPDX-License-Identifier: Apache-2.0
"""Native GPU correctness coverage using the existing DeviceExec context.

GPU element work runs in its existing parallel kernels. Fixture submission
is ordered because DeviceExec shares sequence_ctx() and its pool; independent
concurrent sessions would race that context's ownership. This check records
that limitation explicitly and does not claim concurrent-session coverage.
Use the smallest existing GPU host; this is not a CPU scaling measurement.
"""
from autograd.ops_api import OpsSession
from checks.native_tests.ops_session_cases import OPS_SESSION_FIXTURES, require, run_ops_session_fixture
from sequence.exec_device import DeviceExec


def main() raises:
    var failed = 0
    for fixture_id in range(OPS_SESSION_FIXTURES):
        try:
            var s = OpsSession[DeviceExec](DeviceExec(), False)
            try:
                run_ops_session_fixture(s, fixture_id)
            except error:
                s.close()
                raise error
            s.close()
        except error:
            print("FAIL ops_session_gpu fixture", fixture_id, String(error))
            failed += 1
    require(failed == 0, "ops_session_gpu failed " + String(failed) + " of " + String(OPS_SESSION_FIXTURES))
    print("PASS ops_session_gpu:", OPS_SESSION_FIXTURES, "native fixtures; shared-context submission ordered")
