# SPDX-License-Identifier: Apache-2.0
"""Independent typed sequence fixtures use the existing host worker pool."""
from core.host_parallel import host_parallelize
from checks.native_tests.sequence_plan_metadata import SEQUENCE_METADATA_FIXTURES, require, run_metadata_fixture
from checks.native_tests.sequence_plan_buffers import SEQUENCE_BUFFER_FIXTURES, run_buffer_fixture

comptime FIXTURES = SEQUENCE_METADATA_FIXTURES + SEQUENCE_BUFFER_FIXTURES


def main() raises:
    var failures = List[Int](length=FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(fixture_id: Int) {imm fp}:
        try:
            if fixture_id < SEQUENCE_METADATA_FIXTURES:
                run_metadata_fixture(fixture_id)
            else:
                run_buffer_fixture(fixture_id - SEQUENCE_METADATA_FIXTURES)
        except error:
            print("FAIL sequence_input_plan fixture", fixture_id, String(error))
            fp.unsafe_store(fixture_id, 1)
    host_parallelize(task, FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "sequence_input_plan failed " + String(failed) + " fixtures")
    print("PASS sequence_input_plan:", FIXTURES, "native format/shape/scalar/span/ownership fixtures, parallel")
