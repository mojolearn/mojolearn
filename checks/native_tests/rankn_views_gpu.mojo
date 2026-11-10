# SPDX-License-Identifier: Apache-2.0
"""Native GPU element kernels, with ordered shared-context submissions.

DeviceExec shares sequence_ctx() and its pool; concurrent sessions race its
ownership. The same explicit limitation as the ops-session check applies.
This is correctness coverage, not concurrent-session or scaling evidence.
"""
from autograd.ops_api import OpsSession
from checks.native_tests.rankn_views_cases import RANKN_VIEW_FIXTURES, require, run_rankn_view_fixture
from sequence.exec_device import DeviceExec


def main() raises:
    var failed = 0
    for fixture_id in range(RANKN_VIEW_FIXTURES):
        try:
            var s = OpsSession[DeviceExec](DeviceExec(), fixture_id == 16)
            try:
                run_rankn_view_fixture(s, fixture_id)
            except error:
                s.close()
                raise error
            s.close()
        except error:
            print("FAIL rankn_views_gpu fixture", fixture_id, String(error))
            failed += 1
    require(failed == 0, "rankn_views_gpu failed " + String(failed) + " of " + String(RANKN_VIEW_FIXTURES))
    print("PASS rankn_views_gpu:", RANKN_VIEW_FIXTURES, "native fixtures; shared-context submission ordered")
