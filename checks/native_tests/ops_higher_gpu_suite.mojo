# SPDX-License-Identifier: Apache-2.0
"""Higher-order gradient fixtures on the device, in order.

DeviceExec shares sequence_ctx() and its pool, so independent concurrent
sessions would race that context's ownership (as in ops_session_gpu); the
fixtures run one after another and each kernel keeps its own parallelism.
"""
from autograd.ops_api import OpsSession
from checks.native_tests.ops_higher_cases import OPS_HIGHER_FIXTURES, require, run_ops_higher_fixture
from sequence.exec_device import DeviceExec


def run_checks() raises:
    var failed = 0
    for fixture_id in range(OPS_HIGHER_FIXTURES):
        try:
            var s = OpsSession[DeviceExec](DeviceExec(), True)
            var again = OpsSession[DeviceExec](DeviceExec(), True)
            try:
                run_ops_higher_fixture(s, again, fixture_id)
            except error:
                s.close()
                again.close()
                raise error
            s.close()
            again.close()
        except error:
            print("FAIL ops_higher_gpu fixture", fixture_id, String(error))
            failed += 1
    require(failed == 0, "ops_higher_gpu failed " + String(failed) + " of " + String(OPS_HIGHER_FIXTURES))
    print("PASS ops_higher_gpu:", OPS_HIGHER_FIXTURES, "higher-order gradient fixtures; shared-context submission ordered")
