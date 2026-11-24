# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""The nearest_centroid_native native check: the four fit/query plan fixtures and three empty-query contracts of
`nearest_centroid_descriptor_cases.mojo` through the column's real native descriptor (host on the host column, the
device descriptor on the device column), independent rows across the host workers."""
from std.testing import TestSuite
from std.sys.compile import is_defined
from checks.native_tests.testsuite_cases import check_cases, CPU_COLUMN_BUILD
from checks.native_tests.nearest_centroid_descriptor_cases import fixture, empty_case
from core.native_descriptor_abi import NativeDescriptorCall
from x_neighbors.native_descriptor_host import xn_native_v1
from x_neighbors.native_descriptor_device import xn_native_v1 as xn_native_v1_gpu



def _entry() -> NativeDescriptorCall:
    comptime if CPU_COLUMN_BUILD:
        return xn_native_v1
    else:
        return xn_native_v1_gpu


def _fit_query(unit: Int) raises:
    fixture(_entry(), unit)


def _empty(unit: Int) raises:
    empty_case(_entry(), unit)


def test_fit_query() raises:
    check_cases("fit_query", 4, _fit_query)


def test_empty() raises:
    check_cases("empty", 3, _empty)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
