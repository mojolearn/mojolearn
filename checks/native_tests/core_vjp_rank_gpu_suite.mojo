# SPDX-License-Identifier: Apache-2.0
"""First-derivative ranks of the session's vjp on the device, in order.

DeviceExec shares sequence_ctx() and its pool, so independent concurrent
sessions would race that context's ownership (as in ops_higher_gpu); the
fixtures run one after another and each kernel keeps its own parallelism.
"""
from autograd.ops_api import OpsSession
from checks.native_tests.ops_higher_cases import require
from checks.native_tests.core_vjp_rank_cases import CORE_VJP_RANK_FIXTURES, run_core_vjp_rank_fixture
from sequence.exec_device import DeviceExec


def run_checks() raises:
    var failed = 0
    for fixture_id in range(CORE_VJP_RANK_FIXTURES):
        try:
            var record = True
            var s = OpsSession[DeviceExec](DeviceExec(), record)
            var again = OpsSession[DeviceExec](DeviceExec(), record)
            try:
                run_core_vjp_rank_fixture(s, again, fixture_id)
            except error:
                s.close()
                again.close()
                raise error
            s.close()
            again.close()
        except error:
            print("FAIL core_vjp_rank_gpu fixture", fixture_id, String(error))
            failed += 1
    require(failed == 0, "core_vjp_rank_gpu failed " + String(failed) + " of " + String(CORE_VJP_RANK_FIXTURES))
    print("PASS core_vjp_rank_gpu:", CORE_VJP_RANK_FIXTURES, "fixtures; shared-context submission ordered")
