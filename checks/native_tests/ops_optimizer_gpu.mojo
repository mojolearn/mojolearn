# SPDX-License-Identifier: Apache-2.0
"""Native optimizer fixtures; shared-context GPU submission is ordered.

DeviceExec shares sequence_ctx() and its pool. GPU tensor work uses the existing parallel kernels; ordered fixtures do not prove concurrent-session safety. Use the smallest existing GPU host.
"""
from autograd.ops_api import OpsSession
from checks.native_tests.ops_optimizer_cases import OPS_OPTIMIZER_FIXTURES, require, run_ops_optimizer_fixture
from sequence.exec_device import DeviceExec


def main() raises:
    var failed = 0
    for fixture_id in range(OPS_OPTIMIZER_FIXTURES):
        try:
            var s = OpsSession[DeviceExec](DeviceExec(), True)
            var restored = OpsSession[DeviceExec](DeviceExec(), fixture_id != 9)
            var model_ex = DeviceExec()
            try:
                run_ops_optimizer_fixture(s, restored, model_ex, fixture_id)
            except error:
                restored.close()
                s.close()
                raise error
            restored.close()
            s.close()
        except error:
            print("FAIL ops_optimizer_gpu fixture", fixture_id, String(error))
            failed += 1
    require(failed == 0, "ops_optimizer_gpu failed " + String(failed) + " of " + String(OPS_OPTIMIZER_FIXTURES))
    print("PASS ops_optimizer_gpu:", OPS_OPTIMIZER_FIXTURES, "native fixtures; shared-context submission ordered")
