# SPDX-License-Identifier: Apache-2.0
"""Importable typed optimizer owner CPU lifecycle checks."""
from core.host_parallel import host_parallelize
from training.host.optimizer_native import optimizer_host_v1
from training.optimizer_native_api import OptimizerNativeDriver
from checks.native_tests.optimizer_owner_cases import OWNER_PURE, OWNER_RESOURCE, pure, resource
from checks.native_tests.optimizer_descriptor_cases import require


def run_checks() raises:
    var failures = List[Int](length=OWNER_PURE, fill=0)
    var fp = failures.unsafe_ptr()
    def task(fixture_id: Int) {imm fp}:
        try:
            pure(OptimizerNativeDriver(optimizer_host_v1, True), fixture_id)
        except error:
            fp.unsafe_store(fixture_id, 1)
            print("FAIL optimizer_owner_cpu control", fixture_id, String(error))
    host_parallelize(task, OWNER_PURE)
    var failed = 0
    for value in failures:
        failed += value
    # The existing host and device optimizer pools use shared handle tables.
    # Resource transitions are ordered; independent pure controls run above.
    for fixture_id in range(OWNER_RESOURCE):
        try:
            resource(OptimizerNativeDriver(optimizer_host_v1, True), fixture_id)
        except error:
            failed += 1
            print("FAIL optimizer_owner_cpu resource", fixture_id, String(error))
    require(failed == 0, "optimizer_owner_cpu failed " + String(failed) + " fixtures")
    print("PASS optimizer_owner_cpu:", OWNER_PURE + OWNER_RESOURCE, "typed owner/control fixtures")
