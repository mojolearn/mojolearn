# SPDX-License-Identifier: Apache-2.0
# Native density evaluation and consumed output against scalar oracle; public validation adapter excluded.
from kde.checks.kde_check import check_kde_device_equals_oracle


def main() raises:
    check_kde_device_equals_oracle()
    print("NATIVE_ONLY_PASS kernel_density")
