# SPDX-License-Identifier: Apache-2.0
"""Theta state GPU fixtures; ordered shared sequence context, parallel kernels."""
from sequence.exec_device import DeviceExec
from checks.native_tests.theta_state_cases import THETA_STATE_FIXTURES, require, run_theta_state_fixture


def main() raises:
    var failed = 0
    for fixture_id in range(THETA_STATE_FIXTURES):
        try:
            var ex = DeviceExec()
            run_theta_state_fixture(ex, fixture_id)
        except error:
            failed += 1
            print("FAIL theta_state_gpu fixture", fixture_id, String(error))
    require(failed == 0, "theta_state_gpu failed " + String(failed) + " fixtures")
    print("PASS theta_state_gpu:", THETA_STATE_FIXTURES, "native fit/tail fixtures")
