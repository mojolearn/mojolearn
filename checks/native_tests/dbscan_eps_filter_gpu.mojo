# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""Lane sp-dbscan-filter (2026-11-20): the GEMM filter of the brute epsilon pass (`dbscan/impl/brute_eps_filter.mojo`)
against the exact pass, every bit word and degree.
"""
from std.testing import TestSuite
from dbscan.checks.dbscan_eps_filter_suite import run_checks


def test_eps_filter() raises:
    run_checks()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
