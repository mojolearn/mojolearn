# SPDX-License-Identifier: Apache-2.0
from checks.native_tests.nearest_centroid_descriptor_suite import run_checks
from x_neighbors.native_descriptor_host import xn_native_v1


def main() raises:
    run_checks(xn_native_v1)
