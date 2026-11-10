# SPDX-License-Identifier: Apache-2.0
"""Direct native GPU operations, with ordered shared-context submissions.

DeviceExec shares sequence_ctx() and its pool. Respect that ownership and
use the existing parallel kernels; this does not prove session concurrency.
"""
from autograd.ops_api import OpsSession
from checks.native_tests.rankn_math_cases import RANKN_MATH_FIXTURES, require, run_rankn_math_fixture
from sequence.exec_device import DeviceExec


def main() raises:
    var failed = 0
    for fixture_id in range(RANKN_MATH_FIXTURES):
        try:
            var s = OpsSession[DeviceExec](DeviceExec(), fixture_id == 0)
            try:
                run_rankn_math_fixture(s, fixture_id)
            except error:
                s.close()
                raise error
            s.close()
        except error:
            print("FAIL rankn_math_gpu fixture", fixture_id, String(error))
            failed += 1
    require(failed == 0, "rankn_math_gpu failed " + String(failed) + " fixtures")
    print("PASS rankn_math_gpu:", RANKN_MATH_FIXTURES, "native fixtures; shared-context submission ordered")
