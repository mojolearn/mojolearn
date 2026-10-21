# SPDX-License-Identifier: Apache-2.0
# Native isolation forest fit/path scoring consumed against independent oracle.
from isolation_forest.checks.if_check import check_if_device_equals_oracle


def main() raises:
    check_if_device_equals_oracle()
    print("NATIVE_ONLY_PASS isolation_forest")
