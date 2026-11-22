# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""Independent typed sequence fixtures use the existing host worker pool."""
from checks.native_tests.sequence_plan_metadata import SEQUENCE_METADATA_FIXTURES, run_metadata_fixture
from checks.native_tests.sequence_plan_buffers import SEQUENCE_BUFFER_FIXTURES, run_buffer_fixture
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require

comptime FIXTURES = SEQUENCE_METADATA_FIXTURES + SEQUENCE_BUFFER_FIXTURES


def test_metadata() raises:
    check_cases("metadata", SEQUENCE_METADATA_FIXTURES, run_metadata_fixture)


def test_buffers() raises:
    check_cases("buffers", FIXTURES - SEQUENCE_METADATA_FIXTURES, run_buffer_fixture)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
