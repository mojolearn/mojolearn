# SPDX-License-Identifier: Apache-2.0
"""The import elementwise kinds against float64 oracles on the device, in order.

DeviceExec shares sequence_ctx() and its pool, so independent concurrent
sessions would race that context's ownership (as in ops_higher_gpu); the
fixtures run one after another and each kernel keeps its own parallelism.
"""
from autograd.ops_api import OpsSession
from checks.native_tests.ops_higher_cases import require
from checks.native_tests.core_ops_import_kinds_cases import CORE_OPS_IMPORT_KINDS_FIXTURES, run_core_ops_import_kinds_fixture, core_ops_import_kinds_records
from sequence.exec_device import DeviceExec


def run_checks() raises:
    var failed = 0
    for fixture_id in range(CORE_OPS_IMPORT_KINDS_FIXTURES):
        try:
            var record = core_ops_import_kinds_records(fixture_id)
            var s = OpsSession[DeviceExec](DeviceExec(), record)
            var again = OpsSession[DeviceExec](DeviceExec(), record)
            try:
                run_core_ops_import_kinds_fixture(s, again, fixture_id)
            except error:
                s.close()
                again.close()
                raise error
            s.close()
            again.close()
        except error:
            print("FAIL core_ops_import_kinds_gpu fixture", fixture_id, String(error))
            failed += 1
    require(failed == 0, "core_ops_import_kinds_gpu failed " + String(failed) + " of " + String(CORE_OPS_IMPORT_KINDS_FIXTURES))
    print("PASS core_ops_import_kinds_gpu:", CORE_OPS_IMPORT_KINDS_FIXTURES, "fixtures; shared-context submission ordered")
