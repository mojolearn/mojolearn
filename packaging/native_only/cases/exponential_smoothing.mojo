# SPDX-License-Identifier: Apache-2.0
# Native Holt-Winters fit/forecast consumed against scalar recurrence oracle.
from holtwinters.checks.hw_check import check_hw_device_equals_oracle


def main() raises:
    check_hw_device_equals_oracle()
    print("NATIVE_ONLY_PASS exponential_smoothing")
