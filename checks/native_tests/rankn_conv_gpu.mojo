# SPDX-License-Identifier: Apache-2.0
"""Native GPU convolution graphs with independent parallel host oracles.

DeviceExec owns a shared context/pool; submit fixtures in order and use
existing parallel kernels. Session concurrency/scaling is not claimed.
"""
from autograd.ops_api import OpsSession
from checks.native_tests.rankn_conv_cases import RANKN_CONV_FIXTURES, run_rankn_conv_fixture
from checks.native_tests.rankn_math_cases import require
from sequence.exec_device import DeviceExec


def main() raises:
    var failed = 0
    for fixture_id in range(RANKN_CONV_FIXTURES):
        try:
            var s = OpsSession[DeviceExec](DeviceExec(), True)
            try:
                run_rankn_conv_fixture(s, fixture_id)
            except error:
                s.close()
                raise error
            s.close()
        except error:
            print("FAIL rankn_conv_gpu fixture", fixture_id, String(error))
            failed += 1
    require(failed == 0, "rankn_conv_gpu failed " + String(failed) + " fixtures")
    print("PASS rankn_conv_gpu:", RANKN_CONV_FIXTURES, "native fixtures; shared-context submission ordered")
