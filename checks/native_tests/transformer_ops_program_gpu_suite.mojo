# SPDX-License-Identifier: Apache-2.0
"""The transformer-primitive session fixtures on the device, in order.

DeviceExec shares sequence_ctx() and its pool, so independent concurrent
sessions would race that context's ownership (as in ops_higher_gpu); the
fixtures run one after another and each kernel keeps its own parallelism.
"""
from autograd.ops_api import OpsSession
from checks.native_tests.transformer_ops_program_cases import (
    TRANSFORMER_OPS_PROGRAM_FIXTURES, run_transformer_ops_program_fixture,
)
from checks.native_tests.ops_higher_cases import require
from sequence.exec_device import DeviceExec


def run_checks() raises:
    var failed = 0
    for fixture_id in range(TRANSFORMER_OPS_PROGRAM_FIXTURES):
        try:
            var s = OpsSession[DeviceExec](DeviceExec(), True)
            try:
                run_transformer_ops_program_fixture(s, fixture_id)
            except error:
                s.close()
                raise error
            s.close()
        except error:
            print("FAIL transformer_ops_program_gpu fixture", fixture_id, String(error))
            failed += 1
    require(failed == 0, "transformer_ops_program_gpu failed " + String(failed) + " of " + String(TRANSFORMER_OPS_PROGRAM_FIXTURES))
    print("PASS transformer_ops_program_gpu:", TRANSFORMER_OPS_PROGRAM_FIXTURES, "transformer primitive fixtures; shared-context submission ordered")
